from __future__ import annotations

import csv
import statistics
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from evaluation.errors import EvaluationError, MetricUnavailableError
from evaluation.io import iter_jsonl, read_json, update_latest, utc_timestamp, write_json, write_jsonl
from evaluation.manifest import is_successful_generation, sample_id, source_record
from evaluation.metrics.external import backend_metadata, require_configured_backend
from evaluation.metrics.registry import MetricSpec
from evaluation.metrics.text import normalize_text, word_error_rate
from project_conventions import eval_model_root


@dataclass(frozen=True)
class EvalOptions:
    model: str
    task: str
    dataset: str
    split: str
    manifest_path: Path
    dataset_config_path: Path
    metrics_config_path: Path
    output_root: Path
    metric_specs: list[MetricSpec]
    max_samples: int | None
    cache_dir: Path
    device: str
    command: list[str]


def run_evaluation(options: EvalOptions) -> dict[str, Path]:
    dataset_config = read_json(options.dataset_config_path)
    metrics_config = read_json(options.metrics_config_path).get("metrics", {})
    dataset_root = Path(dataset_config["dataset_root"])
    rows = [item.record for item in iter_jsonl(options.manifest_path, options.max_samples)]

    _validate_rows(rows, options.metric_specs, dataset_root)
    metric_backend_metadata = _prepare_metric_backends(options.metric_specs, metrics_config, options.device, options.cache_dir)

    per_sample_rows = []
    for ordinal, row in enumerate(rows):
        per_sample_rows.append(_evaluate_row(row, ordinal, options.metric_specs, dataset_root))
    _compute_external_metrics(options, rows, per_sample_rows, dataset_root, metrics_config)

    summary = _summarize(options, per_sample_rows, metric_backend_metadata)
    output_dir = eval_model_root(options.task, options.dataset, options.model, options.output_root)
    timestamp = utc_timestamp()

    paths = {
        "per_sample": output_dir / f"per_sample_{options.split}_{timestamp}.jsonl",
        "summary": output_dir / f"summary_{options.split}_{timestamp}.json",
        "summary_csv": output_dir / f"summary_{options.split}_{timestamp}.csv",
        "report": output_dir / f"report_{options.split}_{timestamp}.md",
        "run_config": output_dir / f"run_config_{options.split}_{timestamp}.json",
    }

    run_config = _run_config(options, metric_backend_metadata)
    write_jsonl(paths["per_sample"], per_sample_rows)
    write_json(paths["summary"], summary)
    _write_summary_csv(paths["summary_csv"], summary)
    _write_report(paths["report"], summary, run_config)
    write_json(paths["run_config"], run_config)

    latest = {
        "per_sample": output_dir / f"latest_per_sample_{options.split}.jsonl",
        "summary": output_dir / f"latest_summary_{options.split}.json",
        "summary_csv": output_dir / f"latest_summary_{options.split}.csv",
        "report": output_dir / f"latest_report_{options.split}.md",
        "run_config": output_dir / f"latest_run_config_{options.split}.json",
    }
    for key, path in paths.items():
        update_latest(path, latest[key])
    return paths


def dry_run(options: EvalOptions) -> dict[str, Any]:
    dataset_config = read_json(options.dataset_config_path)
    rows = [item.record for item in iter_jsonl(options.manifest_path, options.max_samples)]
    dataset_root = Path(dataset_config["dataset_root"])
    _validate_rows(rows, options.metric_specs, dataset_root)
    output_dir = eval_model_root(options.task, options.dataset, options.model, options.output_root)
    return {
        "dry_run": True,
        "model": options.model,
        "task": options.task,
        "dataset": options.dataset,
        "split": options.split,
        "manifest": str(options.manifest_path),
        "output_dir": str(output_dir),
        "metrics": [spec.name for spec in options.metric_specs],
        "rows": len(rows),
        "cache_dir": str(options.cache_dir),
    }


def _validate_rows(rows: list[dict], specs: list[MetricSpec], dataset_root: Path) -> None:
    if not rows:
        raise EvaluationError("Manifest contains no rows after --max-samples filtering.")
    errors = []
    for ordinal, row in enumerate(rows):
        source = source_record(row)
        sid = sample_id(row, ordinal)
        if not is_successful_generation(row):
            continue
        if any(spec.requires_output_file for spec in specs):
            output = row.get("output_file")
            if not isinstance(output, str) or not output:
                errors.append(f"{sid}: missing output_file")
            elif not Path(output).is_file():
                errors.append(f"{sid}: output_file does not exist: {output}")
        for spec in specs:
            for field in spec.required_source_fields:
                value = source.get(field)
                if not isinstance(value, str) or not value.strip():
                    errors.append(f"{sid}: metric {spec.name} requires source.{field}")
                elif field in {"audio_file", "speech_file"} and not _resolve_reference(dataset_root, value).is_file():
                    errors.append(f"{sid}: metric {spec.name} reference path does not exist: {_resolve_reference(dataset_root, value)}")
            if spec.requires_hypothesis_text and _hypothesis_text(row, spec.name) is None:
                errors.append(f"{sid}: metric {spec.name} requires a text prediction in model_extra")
    if errors:
        preview = "\n".join(f"  - {item}" for item in errors[:20])
        suffix = "" if len(errors) <= 20 else f"\n  ... {len(errors) - 20} more"
        raise EvaluationError(f"Evaluation inputs are incomplete:\n{preview}{suffix}")


def _prepare_metric_backends(
    specs: list[MetricSpec],
    metrics_config: dict[str, dict],
    device: str,
    cache_dir: Path,
) -> dict[str, dict]:
    metadata = {}
    for spec in specs:
        config = metrics_config.get(spec.name, {})
        if spec.external:
            require_configured_backend(spec.name, config)
        metadata[spec.name] = backend_metadata(spec.name, config, device, cache_dir)
    return metadata


def _evaluate_row(row: dict, ordinal: int, specs: list[MetricSpec], dataset_root: Path) -> dict:
    source = source_record(row)
    sid = sample_id(row, ordinal)
    output_file = row.get("output_file")
    result = {
        "row_index": row.get("row_index", ordinal),
        "sample_id": sid,
        "status": "success" if is_successful_generation(row) else "skipped",
        "output_file": output_file,
        "references": _references(source, specs, dataset_root),
        "predictions": _predictions(row, specs),
        "metrics": {},
        "errors": {},
    }
    if not is_successful_generation(row):
        result["errors"]["row"] = row.get("error", "generation was not successful")
        return result

    for spec in specs:
        if spec.corpus_only:
            continue
        try:
            value = _compute_per_sample_metric(spec, row, source)
            if value is not None:
                result["metrics"][spec.name] = value
        except Exception as exc:
            result["status"] = "error"
            result["errors"][spec.name] = f"{type(exc).__name__}: {exc}"
    return result


def _compute_per_sample_metric(spec: MetricSpec, row: dict, source: dict) -> float | None:
    if spec.name == "wer":
        reference = normalize_text(source["speech_text"])
        hypothesis = normalize_text(_hypothesis_text(row, spec.name) or "")
        return word_error_rate(reference, hypothesis)
    if spec.external:
        return None
    raise EvaluationError(f"No implementation for metric {spec.name!r}")


def _compute_external_metrics(
    options: EvalOptions,
    rows: list[dict],
    per_sample_rows: list[dict],
    dataset_root: Path,
    metrics_config: dict[str, dict],
) -> None:
    selected = {spec.name for spec in options.metric_specs if spec.external}
    aac_metrics = [name for name in ("mace", "spider") if name in selected]
    unsupported = sorted(selected - set(aac_metrics))
    if unsupported:
        raise MetricUnavailableError(
            "Executable adapters are not implemented yet for selected external metrics: "
            f"{', '.join(unsupported)}. Configure/download the backend first, then add the adapter."
        )
    if aac_metrics:
        _compute_aac_metrics(options, rows, per_sample_rows, dataset_root, metrics_config, aac_metrics)


def _compute_aac_metrics(
    options: EvalOptions,
    rows: list[dict],
    per_sample_rows: list[dict],
    dataset_root: Path,
    metrics_config: dict[str, dict],
    metric_names: list[str],
) -> None:
    try:
        from aac_metrics import evaluate as aac_evaluate
        from aac_metrics.functional.mace import mace as mace_fn
    except Exception as exc:
        raise MetricUnavailableError(
            "AAC metrics require the 'aac-metrics' package and its cached assets. "
            "Install/cache them before running MACE or SPIDEr."
        ) from exc

    candidates = []
    references = []
    audio_paths = []
    row_map = []
    for index, row in enumerate(rows):
        if not is_successful_generation(row):
            continue
        source = source_record(row)
        candidate = _hypothesis_text(row, "mace")
        if candidate is None:
            continue
        candidates.append(candidate)
        references.append([source["audio_cap"]])
        audio_paths.append(str(_resolve_reference(dataset_root, source["audio_file"])))
        row_map.append(index)

    if "spider" in metric_names:
        cfg = metrics_config.get("spider", {})
        corpus_scores, sent_scores = aac_evaluate(
            candidates,
            references,
            metrics=["spider"],
            cache_path=cfg.get("cache_dir") or str(options.cache_dir),
            device=options.device,
        )
        _apply_sentence_scores(per_sample_rows, row_map, "spider", sent_scores.get("spider"))

    if "mace" in metric_names:
        cfg = metrics_config.get("mace", {})
        _corpus_scores, sent_scores = mace_fn(
            candidates,
            references,
            audio_paths=audio_paths,
            mace_method=cfg.get("mace_method", "combined"),
            clap_model=cfg.get("clap_model", "MS-CLAP-2023"),
            echecker=cfg.get("echecker", "echecker_clotho_audiocaps_base"),
            device=options.device,
            return_all_scores=True,
        )
        _apply_sentence_scores(per_sample_rows, row_map, "mace", sent_scores.get("mace"))


def _apply_sentence_scores(per_sample_rows: list[dict], row_map: list[int], metric: str, scores: object) -> None:
    if scores is None:
        return
    values = scores.detach().cpu().tolist() if hasattr(scores, "detach") else scores
    for row_index, value in zip(row_map, values):
        per_sample_rows[row_index]["metrics"][metric] = float(value)


def _hypothesis_text(row: dict, metric_name: str | None = None) -> str | None:
    extra = row.get("model_extra")
    if not isinstance(extra, dict):
        return None
    if metric_name in {"wer", "semantic_equivalence"}:
        keys = ("understanding_text", "decoded_text", "converted_prompt", "generation_prompt")
    elif metric_name in {"mace", "spider"}:
        keys = ("converted_prompt", "generation_prompt", "decoded_text", "understanding_text")
    else:
        keys = ("generation_prompt", "converted_prompt", "understanding_text", "decoded_text")
    for key in keys:
        value = extra.get(key)
        if isinstance(value, str) and value.strip():
            return value
        if isinstance(value, list) and value and isinstance(value[0], str):
            return value[0]
    return None


def _predictions(row: dict, specs: list[MetricSpec]) -> dict:
    result = {}
    for spec in specs:
        if spec.requires_hypothesis_text:
            result[spec.name] = _hypothesis_text(row, spec.name)
    return result


def _references(source: dict, specs: list[MetricSpec], dataset_root: Path) -> dict:
    fields = sorted({field for spec in specs for field in spec.required_source_fields})
    refs = {}
    for field in fields:
        value = source.get(field)
        if field in {"audio_file", "speech_file"} and isinstance(value, str):
            refs[field] = str(_resolve_reference(dataset_root, value))
        else:
            refs[field] = value
    return refs


def _resolve_reference(dataset_root: Path, value: str) -> Path:
    path = Path(value)
    return path if path.is_absolute() else dataset_root / path


def _summarize(options: EvalOptions, rows: list[dict], backend_metadata: dict[str, dict]) -> dict:
    metric_names = [spec.name for spec in options.metric_specs]
    stats = {}
    for name in metric_names:
        values = [
            row["metrics"][name]
            for row in rows
            if isinstance(row.get("metrics"), dict) and isinstance(row["metrics"].get(name), (int, float))
        ]
        if values:
            stats[name] = _numeric_summary(values)
        elif any(spec.name == name and spec.corpus_only for spec in options.metric_specs):
            stats[name] = {"level": "corpus", "status": "not_computed_in_scaffold"}

    grouped_errors: dict[str, dict[str, int]] = {}
    for row in rows:
        for metric, message in row.get("errors", {}).items():
            grouped_errors.setdefault(metric, {})
            grouped_errors[metric][message] = grouped_errors[metric].get(message, 0) + 1

    return {
        "model": options.model,
        "task": options.task,
        "dataset": options.dataset,
        "split": options.split,
        "manifest": str(options.manifest_path),
        "metrics": metric_names,
        "rows": len(rows),
        "successful_rows": sum(1 for row in rows if row["status"] == "success"),
        "skipped_or_error_rows": sum(1 for row in rows if row["status"] != "success"),
        "metric_summaries": stats,
        "errors_by_metric": grouped_errors,
        "metric_backends": backend_metadata,
    }


def _numeric_summary(values: list[float]) -> dict:
    return {
        "count": len(values),
        "mean": statistics.fmean(values),
        "std": statistics.pstdev(values) if len(values) > 1 else 0.0,
        "min": min(values),
        "max": max(values),
        "median": statistics.median(values),
    }


def _run_config(options: EvalOptions, backend_metadata: dict[str, dict]) -> dict:
    return {
        "model": options.model,
        "task": options.task,
        "dataset": options.dataset,
        "split": options.split,
        "manifest": str(options.manifest_path),
        "dataset_config": str(options.dataset_config_path),
        "metrics_config": str(options.metrics_config_path),
        "metrics": [spec.name for spec in options.metric_specs],
        "metric_backends": backend_metadata,
        "cache_dir": str(options.cache_dir),
        "device": options.device,
        "command": options.command,
        "timestamp_utc": utc_timestamp(),
    }


def _write_summary_csv(path: Path, summary: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["metric", "count", "mean", "std", "min", "max", "median", "status"])
        for metric, stats in summary.get("metric_summaries", {}).items():
            writer.writerow(
                [
                    metric,
                    stats.get("count"),
                    stats.get("mean"),
                    stats.get("std"),
                    stats.get("min"),
                    stats.get("max"),
                    stats.get("median"),
                    stats.get("status"),
                ]
            )


def _write_report(path: Path, summary: dict, run_config: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    lines = [
        f"# Evaluation Report: {summary['model']} {summary['task']}",
        "",
        f"- Dataset: `{summary['dataset']}`",
        f"- Split: `{summary['split']}`",
        f"- Manifest: `{summary['manifest']}`",
        f"- Rows: {summary['rows']}",
        f"- Successful rows: {summary['successful_rows']}",
        f"- Skipped/error rows: {summary['skipped_or_error_rows']}",
        "",
        "## Metrics",
        "",
    ]
    for metric, stats in summary.get("metric_summaries", {}).items():
        if "mean" in stats:
            lines.append(
                f"- `{metric}`: mean={stats['mean']:.6g}, std={stats['std']:.6g}, "
                f"min={stats['min']:.6g}, max={stats['max']:.6g}, median={stats['median']:.6g}"
            )
        else:
            lines.append(f"- `{metric}`: {stats.get('status', 'not computed')}")
    lines.extend(["", "## Dependency Notes", ""])
    lines.append("- aac-metrics / SPIDEr / MACE may require Java and cached assets.")
    lines.append("- KAD/kadtk may have Python version constraints and should be run corpus-level.")
    lines.append("- ASR/CLAP/UTMOS/DNSMOS/FAD/KLD/IS require external model weights or toolkits.")
    lines.append("- Generated-speech dWER is directional: WER(ASR(reference_speech), ASR(generated_speech)).")
    lines.append("- FAD, KAD, KLD, and Inception Score are corpus-level metrics only.")
    lines.extend(["", "## Command", "", "```text", " ".join(run_config.get("command", [])), "```", ""])
    path.write_text("\n".join(lines), encoding="utf-8")
