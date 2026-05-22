from __future__ import annotations

import csv
import json
import os
import statistics
import subprocess
import sys
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
    task_metric_map_path: Path
    output_root: Path
    metric_specs: list[MetricSpec]
    max_samples: int | None
    start_index: int
    cache_dir: Path
    device: str
    command: list[str]


def run_evaluation(options: EvalOptions) -> dict[str, Path]:
    dataset_config = read_json(options.dataset_config_path)
    metrics_config = read_json(options.metrics_config_path).get("metrics", {})
    dataset_root = Path(dataset_config["dataset_root"])
    rows = [
        item.record
        for item in iter_jsonl(
            options.manifest_path,
            max_samples=options.max_samples,
            start_index=options.start_index,
        )
    ]

    _validate_rows(rows, options.metric_specs, dataset_root)
    metric_backend_metadata = _prepare_metric_backends(options.metric_specs, metrics_config, options.device, options.cache_dir)

    per_sample_rows = []
    for ordinal, row in enumerate(rows):
        per_sample_rows.append(_evaluate_row(row, ordinal, options.metric_specs, dataset_root))

    output_dir = eval_model_root(options.task, options.dataset, options.model, options.output_root)
    timestamp = utc_timestamp()
    work_dir = output_dir / f"work_{options.split}_{timestamp}"
    corpus_metrics = _compute_external_metrics(
        options,
        rows,
        per_sample_rows,
        dataset_root,
        metrics_config,
        work_dir,
        metric_backend_metadata,
    )

    summary = _summarize(options, per_sample_rows, metric_backend_metadata, corpus_metrics)

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
    rows = [
        item.record
        for item in iter_jsonl(
            options.manifest_path,
            max_samples=options.max_samples,
            start_index=options.start_index,
        )
    ]
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
        "start_index": options.start_index,
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
            if spec.requires_hypothesis_text and _prediction_text_for_metric(row, spec.name) is None:
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
        if spec.level == "corpus":
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
        hypothesis = normalize_text(_prediction_text_for_metric(row, spec.name) or "")
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
    work_dir: Path,
    metric_backend_metadata: dict[str, dict],
) -> dict[str, Any]:
    selected = {spec.name for spec in options.metric_specs if spec.external}
    corpus_metrics: dict[str, Any] = {}
    backend_metrics = {
        "aac-metrics": ["mace", "spider", "aac_clapsim"],
        "openai-whisper": ["asr_wer", "dwer"],
        "utmosv2": ["utmos"],
        "audioldm_eval": ["fad", "inception_score"],
        "kadtk": ["kad"],
        "stable-audio-metrics": ["clap_score_t2a", "kl_passt"],
    }
    supported = {name for names in backend_metrics.values() for name in names}
    unsupported = sorted(selected - supported)
    if unsupported:
        raise MetricUnavailableError(
            "Executable adapters are not implemented yet for selected external metrics: "
            f"{', '.join(unsupported)}. Configure/download the backend first, then add the adapter."
        )
    for backend, candidate_metrics in backend_metrics.items():
        metric_names = [name for name in candidate_metrics if name in selected]
        if not metric_names:
            continue
        result = _run_backend_worker(options, rows, dataset_root, metrics_config, work_dir, backend, metric_names)
        _merge_backend_result(per_sample_rows, result)
        for metric_name, metadata in result.get("metric_backends", {}).items():
            if metric_name in metric_backend_metadata:
                metric_backend_metadata[metric_name].update(metadata)
        corpus_level_names = {spec.name for spec in options.metric_specs if spec.level == "corpus"}
        corpus_metrics.update(
            {
                name: value
                for name, value in result.get("corpus_metrics", {}).items()
                if name in corpus_level_names
            }
        )
    return corpus_metrics


def _run_backend_worker(
    options: EvalOptions,
    rows: list[dict],
    dataset_root: Path,
    metrics_config: dict[str, dict],
    work_dir: Path,
    backend: str,
    metric_names: list[str],
) -> dict[str, Any]:
    metric_configs = {name: metrics_config.get(name, {}) for name in metric_names}
    python = _backend_python(metric_configs)
    backend_rows = []
    for ordinal, row in enumerate(rows):
        if not is_successful_generation(row):
            continue
        row_index = row.get("row_index", ordinal)
        source = source_record(row)
        backend_source = dict(source)
        for field in ("audio_file", "speech_file"):
            value = backend_source.get(field)
            if isinstance(value, str) and value:
                backend_source[field] = str(_resolve_reference(dataset_root, value))
        backend_rows.append(
            {
                "row_index": row_index,
                "sample_id": sample_id(row, ordinal),
                "output_file": row.get("output_file"),
                "candidate_text": _prediction_text_for_metric(row, metric_names[0]),
                "source": backend_source,
            }
        )

    payload = {
        "backend": backend,
        "metrics": metric_names,
        "metric_configs": metric_configs,
        "rows": backend_rows,
        "cache_dir": str(options.cache_dir),
        "device": options.device,
        "work_dir": str(work_dir),
    }
    backend_work_dir = work_dir / backend
    backend_work_dir.mkdir(parents=True, exist_ok=True)
    input_path = backend_work_dir / "input.json"
    output_path = backend_work_dir / "output.json"
    stdout_path = backend_work_dir / "stdout.log"
    input_path.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")

    env = _backend_env(options, metric_configs)
    script = Path(__file__).resolve().parents[1] / "scripts" / "eval_backend_worker.py"
    completed = subprocess.run(
        [str(python), str(script), "--backend", backend, "--input", str(input_path), "--output", str(output_path)],
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        env=env,
        cwd=Path(__file__).resolve().parents[1],
    )
    stdout_path.write_text(completed.stdout, encoding="utf-8")
    if completed.returncode != 0:
        raise MetricUnavailableError(
            f"Backend {backend!r} failed for metrics {', '.join(metric_names)}. "
            f"See {stdout_path} for details."
        )
    if not output_path.is_file():
        raise MetricUnavailableError(f"Backend {backend!r} did not produce {output_path}.")
    return json.loads(output_path.read_text(encoding="utf-8"))


def _backend_python(metric_configs: dict[str, dict]) -> Path:
    for config in metric_configs.values():
        configured = config.get("python")
        if configured:
            path = Path(configured)
            if not path.is_file():
                raise MetricUnavailableError(f"Configured metric Python does not exist: {path}")
            return path
    return Path(sys.executable)


def _backend_env(options: EvalOptions, metric_configs: dict[str, dict]) -> dict[str, str]:
    env = os.environ.copy()
    cache_dir = _first_metric_config_value(metric_configs, "cache_dir") or str(options.cache_dir)
    cache_path = Path(cache_dir)
    hf_home = _first_metric_config_value(metric_configs, "hf_home")
    transformers_cache = _first_metric_config_value(metric_configs, "transformers_cache")
    torch_home = _first_metric_config_value(metric_configs, "torch_home")
    env["XDG_CACHE_HOME"] = str(cache_path)
    env["HF_HOME"] = str(Path(hf_home)) if hf_home else str(cache_path.parent / "huggingface")
    env["TRANSFORMERS_CACHE"] = (
        str(Path(transformers_cache)) if transformers_cache else env["HF_HOME"]
    )
    env["TORCH_HOME"] = str(Path(torch_home)) if torch_home else str(cache_path.parent / "torch")
    env["NUMBA_CACHE_DIR"] = str(cache_path.parent / "numba")
    env["MPLCONFIGDIR"] = str(cache_path.parent / "matplotlib")
    env["UTMOSV2_CHACHE"] = str(cache_path)
    hf_hub_offline = _first_metric_config_value(metric_configs, "hf_hub_offline")
    transformers_offline = _first_metric_config_value(metric_configs, "transformers_offline")
    if hf_hub_offline is not None:
        env["HF_HUB_OFFLINE"] = hf_hub_offline
    if transformers_offline is not None:
        env["TRANSFORMERS_OFFLINE"] = transformers_offline
    env["TF_CPP_MIN_LOG_LEVEL"] = "3"
    pythonpath = _first_metric_config_value(metric_configs, "pythonpath")
    if pythonpath:
        existing = env.get("PYTHONPATH")
        env["PYTHONPATH"] = pythonpath if not existing else f"{pythonpath}{os.pathsep}{existing}"
    return env


def _first_metric_config_value(metric_configs: dict[str, dict], key: str) -> str | None:
    for config in metric_configs.values():
        value = config.get(key)
        if isinstance(value, str) and value:
            return value
    return None


def _merge_backend_result(per_sample_rows: list[dict], result: dict[str, Any]) -> None:
    row_by_index = {str(row["row_index"]): row for row in per_sample_rows}
    for row_index, metrics in result.get("sample_metrics", {}).items():
        row = row_by_index.get(str(row_index))
        if row is not None:
            row["metrics"].update(metrics)
    for row_index, predictions in result.get("predictions", {}).items():
        row = row_by_index.get(str(row_index))
        if row is not None:
            row["predictions"].update(predictions)
    for row_index, errors in result.get("errors", {}).items():
        row = row_by_index.get(str(row_index))
        if row is not None:
            row["status"] = "error"
            row["errors"].update(errors)


def _prediction_text_for_metric(row: dict, metric_name: str | None = None) -> str | None:
    extra = row.get("model_extra")
    if not isinstance(extra, dict):
        return None
    if metric_name in {"wer", "semantic_equivalence"}:
        keys = ("understanding_text", "decoded_text", "converted_prompt", "generation_prompt")
    elif metric_name in {"mace", "spider", "aac_clapsim"}:
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
    result: dict = {}
    output_file = row.get("output_file")
    if isinstance(output_file, str) and output_file:
        if output_file.endswith(".wav"):
            if row.get("task") in {"T2S", "A2S"}:
                result["generated_speech"] = output_file
            else:
                result["generated_audio"] = output_file
        elif output_file.endswith(".json"):
            result["generated_text_file"] = output_file

    extra = row.get("model_extra")
    if isinstance(extra, dict):
        text = _text_value(extra.get("understanding_text"))
        if text:
            result["generated_text"] = text
            result["transcript"] = text
        converted = _text_value(extra.get("converted_prompt"))
        if converted:
            result["converted_text"] = converted
        prompt = _text_value(extra.get("generation_prompt"))
        if prompt:
            result["generation_prompt"] = prompt
        decoded = _text_value(extra.get("decoded_text"))
        if decoded and "generated_text" not in result:
            result["generated_text"] = decoded
    return result


def _text_value(value: object) -> str | None:
    if isinstance(value, str) and value.strip():
        return value
    if isinstance(value, list) and value and isinstance(value[0], str) and value[0].strip():
        return value[0]
    return None


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


def _summarize(
    options: EvalOptions,
    rows: list[dict],
    backend_metadata: dict[str, dict],
    corpus_metrics: dict[str, Any] | None = None,
) -> dict:
    corpus_metrics = corpus_metrics or {}
    metric_names = [spec.name for spec in options.metric_specs]
    stats = {}
    for name in metric_names:
        if name in corpus_metrics:
            stats[name] = _corpus_summary(corpus_metrics[name])
            continue
        values = [
            row["metrics"][name]
            for row in rows
            if isinstance(row.get("metrics"), dict) and isinstance(row["metrics"].get(name), (int, float))
        ]
        if values:
            stats[name] = _numeric_summary(values)
        elif any(spec.name == name and spec.level == "corpus" for spec in options.metric_specs):
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


def _corpus_summary(value: Any) -> dict:
    if isinstance(value, dict):
        return value
    numeric = float(value)
    return {
        "level": "corpus",
        "count": 1,
        "mean": numeric,
        "std": 0.0,
        "min": numeric,
        "max": numeric,
        "median": numeric,
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
        "task_metric_map": str(options.task_metric_map_path),
        "metrics": [spec.name for spec in options.metric_specs],
        "metric_backends": backend_metadata,
        "start_index": options.start_index,
        "max_samples": options.max_samples,
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
    lines.append("- ASR/CLAP/UTMOS/DNSMOS/FAD/KL_PaSST/IS require external model weights or toolkits.")
    lines.append("- Generated-speech dWER is directional: WER(ASR(reference_speech), ASR(generated_speech)).")
    lines.append("- FAD, KAD, KL_PaSST, and Inception Score are corpus-level metrics only.")
    lines.extend(["", "## Command", "", "```text", " ".join(run_config.get("command", [])), "```", ""])
    path.write_text("\n".join(lines), encoding="utf-8")
