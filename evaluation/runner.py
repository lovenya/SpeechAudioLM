from __future__ import annotations

import math
import json
import os
import statistics
import subprocess
import sys
import tempfile
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from evaluation.artifacts import (
    write_json_create_only,
    write_jsonl_create_only,
    write_text_create_only,
)
from evaluation.errors import EvaluationError, MetricUnavailableError
from evaluation.io import iter_jsonl, read_json, utc_timestamp, write_json, write_jsonl
from evaluation.manifest import is_successful_generation, sample_id, source_record
from evaluation.metrics.external import backend_metadata, require_configured_backend
from evaluation.metrics.registry import MetricSpec
from evaluation.metrics.text import normalize_text, word_error_rate
from evaluation.result_validation import validate_final_result_completeness
from project_conventions import eval_stage_root
from project_paths import model_path


GEVAL_METRICS = {
    "g_eval_prometheus_plausibility",
    "g_eval_prometheus_reference",
}


@dataclass(frozen=True)
class EvalOptions:
    model: str
    task: str
    stage: str
    direction: str
    evaluated_run_id: str
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
    temporary: bool = False
    eval_run_id: str | None = None


def run_evaluation(options: EvalOptions) -> dict[str, Path]:
    started_monotonic = time.perf_counter()
    started_at = utc_timestamp()
    dataset_config = read_json(options.dataset_config_path)
    metrics_config = read_json(options.metrics_config_path).get("metrics", {})
    dataset_root = Path(dataset_config["dataset_root"])
    rows, selected_sample_ids = _evaluation_rows(options)

    preflight_errors = _validate_rows(
        rows,
        options.metric_specs,
        dataset_root,
        require_success=False,
        allow_attributable_errors=True,
    )
    metric_backend_metadata = _prepare_metric_backends(options.metric_specs, metrics_config, options.device, options.cache_dir)

    per_sample_rows = []
    for ordinal, row in enumerate(rows):
        per_sample_rows.append(_evaluate_row(row, ordinal, options.metric_specs, dataset_root))
    _merge_backend_result(per_sample_rows, {"errors": preflight_errors})

    eval_run_id = options.eval_run_id or utc_timestamp()
    output_dir = _evaluation_output_dir(options, eval_run_id)
    timestamp = eval_run_id
    artifact_dir = _evaluation_artifact_dir(output_dir, options.split, timestamp)
    work_dir = _evaluation_work_dir(output_dir, options.split, timestamp)
    corpus_metrics = _compute_external_metrics(
        options,
        rows,
        per_sample_rows,
        dataset_root,
        metrics_config,
        work_dir,
        artifact_dir,
        metric_backend_metadata,
    )
    _record_nonfinite_metric_errors(per_sample_rows, corpus_metrics)
    validation_report = validate_final_result_completeness(
        options.metric_specs,
        selected_sample_ids,
        per_sample_rows,
        corpus_metrics,
        diagnostic=options.temporary,
    ).to_dict()

    summary = _summarize(
        options,
        eval_run_id,
        per_sample_rows,
        metric_backend_metadata,
        corpus_metrics,
        validation_report,
        total_seconds=time.perf_counter() - started_monotonic,
    )

    paths = {
        "run": output_dir / "run.json",
        "config": output_dir / "config.json",
        "manifest": output_dir / "manifest.jsonl",
        "per_sample": output_dir / "manifest.jsonl",
        "inspect": output_dir / "inspect.json",
        "errors": output_dir / "errors.json",
        "summary": output_dir / "summary.json",
        "report": output_dir / "report.md",
        "run_config": output_dir / "config.json",
    }

    run_config = _run_config(
        options,
        metric_backend_metadata,
        validation_report,
        num_samples=len(rows),
    )
    run_doc = _run_document(options, eval_run_id, started_at, validation_report, paths)
    errors_doc = _errors_document(options, eval_run_id, per_sample_rows, validation_report)
    manifest_rows = _manifest_rows(options, per_sample_rows)
    inspect_doc = _inspect_document(
        options,
        eval_run_id,
        per_sample_rows,
        metric_backend_metadata,
        summary,
        validation_report,
    )
    write_json_create_only(paths["run"], run_doc)
    write_json_create_only(paths["config"], run_config)
    write_jsonl_create_only(paths["manifest"], manifest_rows)
    write_json_create_only(paths["inspect"], inspect_doc)
    write_json_create_only(paths["errors"], errors_doc)
    write_json_create_only(paths["summary"], summary)
    write_text_create_only(paths["report"], _report_text(summary, run_config))
    _write_eval_log(run_doc, summary, run_config)

    return paths


def _evaluation_artifact_dir(output_dir: Path, split: str, timestamp: str) -> Path:
    return output_dir / "artifacts"


def _evaluation_work_dir(output_dir: Path, split: str, timestamp: str) -> Path:
    slurm_tmpdir = os.environ.get("SLURM_TMPDIR")
    if slurm_tmpdir:
        root = Path(slurm_tmpdir) / "speechaudiolm_evaluation"
        root.mkdir(parents=True, exist_ok=True)
        return Path(tempfile.mkdtemp(prefix=f"work_{split}_{timestamp}_", dir=root))
    return output_dir / "artifacts" / "work"


def dry_run(options: EvalOptions) -> dict[str, Any]:
    dataset_config = read_json(options.dataset_config_path)
    rows, selected_sample_ids = _evaluation_rows(options)
    dataset_root = Path(dataset_config["dataset_root"])
    _validate_rows(rows, options.metric_specs, dataset_root, require_success=False)
    output_dir = _evaluation_output_dir(options, options.eval_run_id or "<eval_run_id>")
    return {
        "dry_run": True,
        "status": _evaluation_status(options),
        "model": options.model,
        "task": options.task,
        "stage": options.stage,
        "direction": options.direction,
        "evaluated_run_id": options.evaluated_run_id,
        "dataset": options.dataset,
        "split": options.split,
        "manifest": str(options.manifest_path),
        "output_dir": str(output_dir),
        "metrics": [spec.name for spec in options.metric_specs],
        "temporary": options.temporary,
        "start_index": options.start_index if options.temporary else None,
        "max_samples": options.max_samples if options.temporary else None,
        "rows": len(rows),
        "selected_sample_count": len(selected_sample_ids),
        "cache_dir": str(options.cache_dir),
    }


VALID_EVAL_STAGES = {"a2t", "s2t", "t2t", "t2a", "t2s", "s2a", "a2s"}
VALID_EVAL_DIRECTIONS = {"a2s", "s2a"}
IMPLIED_STAGE_DIRECTIONS = {
    "a2t": "a2s",
    "t2s": "a2s",
    "a2s": "a2s",
    "s2t": "s2a",
    "t2a": "s2a",
    "s2a": "s2a",
}


def resolve_stage_direction(stage: str, direction: str | None) -> str:
    if stage not in VALID_EVAL_STAGES:
        raise EvaluationError(f"Invalid eval stage {stage!r}; expected one of {sorted(VALID_EVAL_STAGES)}.")
    if direction is not None and direction not in VALID_EVAL_DIRECTIONS:
        raise EvaluationError(f"Invalid eval direction {direction!r}; expected one of {sorted(VALID_EVAL_DIRECTIONS)}.")
    implied = IMPLIED_STAGE_DIRECTIONS.get(stage)
    if implied is None:
        if direction is None:
            raise EvaluationError("--direction is required when --stage t2t is selected.")
        return direction
    if direction is not None and direction != implied:
        raise EvaluationError(f"Stage {stage!r} implies direction {implied!r}; got {direction!r}.")
    return implied


def _evaluation_output_dir(options: EvalOptions, eval_run_id: str) -> Path:
    if options.temporary and "temporary" not in options.output_root.parts:
        raise EvaluationError("Temporary evaluations must write under a temporary results root.")
    direction = resolve_stage_direction(options.stage, options.direction)
    if not options.evaluated_run_id:
        raise EvaluationError("evaluated_run_id is required.")
    if len(options.metric_specs) != 1:
        raise EvaluationError("Each eval run must contain exactly one metric; split multi-metric commands into per-metric runs.")
    metric_id = options.metric_specs[0].name
    return eval_stage_root(
        direction,
        options.stage,
        options.model,
        options.dataset,
        options.split,
        options.evaluated_run_id,
        metric_id,
        eval_run_id,
        root=options.output_root,
    )


def _evaluation_rows(options: EvalOptions) -> tuple[list[dict], tuple[str, ...]]:
    if options.start_index < 0:
        raise EvaluationError("start_index must be non-negative.")
    if options.max_samples is not None and options.max_samples < 1:
        raise EvaluationError("max_samples must be positive when set.")
    if not options.temporary and (options.start_index != 0 or options.max_samples is not None):
        raise EvaluationError("Canonical evaluation must use the full manifest; use --temporary for sliced checks.")
    all_rows = [item.record for item in iter_jsonl(options.manifest_path)]
    stop_index = None if options.max_samples is None else options.start_index + options.max_samples
    selected_rows = all_rows[options.start_index:stop_index]
    selected_ids = tuple(
        sample_id(row, ordinal)
        for ordinal, row in enumerate(selected_rows, start=options.start_index)
    )
    if not selected_rows:
        raise EvaluationError("Manifest contains no rows after --max-samples filtering.")
    return selected_rows, selected_ids


def _evaluation_status(options: EvalOptions) -> str:
    return "diagnostic_non_authoritative" if options.temporary else "final_scientific"


def _validate_rows(
    rows: list[dict],
    specs: list[MetricSpec],
    dataset_root: Path,
    *,
    require_success: bool = False,
    allow_attributable_errors: bool = False,
) -> dict[str, dict[str, str]]:
    if not rows:
        raise EvaluationError("Manifest contains no rows after --max-samples filtering.")
    errors: list[str] = []
    attributable: dict[str, dict[str, str]] = {}
    for ordinal, row in enumerate(rows):
        source = source_record(row)
        sid = sample_id(row, ordinal)
        row_key = str(row.get("row_index", ordinal))
        if not is_successful_generation(row):
            if require_success:
                errors.append(f"{sid}: generation was not successful")
            for spec in specs:
                attributable.setdefault(row_key, {})[spec.name] = str(
                    row.get("error") or "generation was not successful"
                )
            continue
        for spec in specs:
            metric_errors: list[str] = []
            if spec.requires_output_file:
                output = row.get("output_file")
                if not isinstance(output, str) or not output:
                    metric_errors.append("missing output_file")
                elif not Path(output).is_file():
                    metric_errors.append(f"output_file does not exist: {output}")
            for field in spec.required_source_fields:
                value = _source_field_value(source, field, dataset_root)
                if not isinstance(value, str) or not value.strip():
                    metric_errors.append(f"requires source.{field}")
                elif field in {"audio_file", "speech_file"} and not _resolve_reference(dataset_root, value).is_file():
                    metric_errors.append(
                        f"reference path does not exist: {_resolve_reference(dataset_root, value)}"
                    )
            if spec.requires_hypothesis_text and _prediction_text_for_metric(row, spec.name) is None:
                if spec.name == "prompt_adherence_clap_score":
                    metric_errors.append("requires the exact persisted generation prompt")
                else:
                    metric_errors.append("requires a text prediction in model_extra")
            if metric_errors:
                message = "; ".join(metric_errors)
                errors.append(f"{sid}: metric {spec.name}: {message}")
                attributable.setdefault(row_key, {})[spec.name] = message
    if errors and not allow_attributable_errors:
        preview = "\n".join(f"  - {item}" for item in errors[:20])
        suffix = "" if len(errors) <= 20 else f"\n  ... {len(errors) - 20} more"
        raise EvaluationError(f"Evaluation inputs are incomplete:\n{preview}{suffix}")
    return attributable


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
    row_status = "success" if is_successful_generation(row) else "skipped"
    if row_status == "success" and all(spec.level == "corpus" for spec in specs):
        row_status = "included"
    result = {
        "row_index": row.get("row_index", ordinal),
        "sample_id": sid,
        "status": row_status,
        "output_file": output_file,
        "source": _source_context(source, dataset_root),
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



def _manifest_rows(options: EvalOptions, rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    metric = options.metric_specs[0]
    return [_manifest_row(metric, row) for row in rows]


def _manifest_row(metric: MetricSpec, row: dict[str, Any]) -> dict[str, Any]:
    return {
        "schema_version": 1,
        "row_index": row.get("row_index"),
        "sample_id": row.get("sample_id"),
        "metric_id": metric.name,
        "status": row.get("status"),
        "input": _manifest_input(metric, row),
        "output": _manifest_output(metric, row),
        "error": _manifest_error(metric, row),
    }


def _manifest_input(metric: MetricSpec, row: dict[str, Any]) -> dict[str, Any]:
    if metric.name == "nli_contradiction" or _is_geval_metric(metric.name):
        details = _metric_prediction_details(row, metric.name)
        source = {
            "direction": details.get("direction"),
            "stage_input_text": details.get("input_text"),
        }
        generated = {"stage_output_text": details.get("generated_text")}
        reference: dict[str, Any] = {}
        if metric.name == "g_eval_prometheus_reference":
            anchors = details.get("reference_anchors")
            reference["reference_anchors"] = anchors if isinstance(anchors, list) else []
        return {
            "source": {key: value for key, value in source.items() if value is not None},
            "generated": {key: value for key, value in generated.items() if value is not None},
            "reference": reference,
        }
    return {
        "source": row.get("source") if isinstance(row.get("source"), dict) else {},
        "generated": _manifest_generated_input(metric, row),
        "reference": row.get("references") if isinstance(row.get("references"), dict) else {},
    }


def _manifest_generated_input(metric: MetricSpec, row: dict[str, Any]) -> dict[str, Any]:
    generated = _generated_input(row)
    predictions = generated.pop("predictions", None)
    if isinstance(predictions, dict):
        for key, value in predictions.items():
            if key == metric.name:
                continue
            generated.setdefault(key, value)
    return generated


def _manifest_output(metric: MetricSpec, row: dict[str, Any]) -> dict[str, Any]:
    metrics = row.get("metrics") if isinstance(row.get("metrics"), dict) else {}
    predictions = row.get("predictions") if isinstance(row.get("predictions"), dict) else {}
    metric_details = predictions.get(metric.name) if isinstance(predictions.get(metric.name), dict) else {}
    score = metrics.get(metric.name)
    if metric.level == "corpus" and score is None:
        details = {"score_location": "summary.json"}
        if metric_details:
            details["backend_output"] = _canonical_metric_details(metric_details)
    elif metric.name == "nli_contradiction":
        details = _nli_metric_details(metric_details)
    elif _is_geval_metric(metric.name):
        details = _geval_metric_details(metric_details)
    else:
        details = _canonical_metric_details(metric_details)
    output = {
        "score": score if isinstance(score, (int, float)) and not isinstance(score, bool) else None,
        "score_name": _metric_score_name(metric.name),
        "higher_is_better": _higher_is_better(metric.name),
    }
    score_range = _metric_score_range(metric.name)
    if score_range is not None:
        output["range"] = score_range
    interpretation = _metric_score_interpretation(metric.name)
    if interpretation is not None:
        output["interpretation"] = interpretation
    output["details"] = details
    return output


def _is_geval_metric(metric_name: str) -> bool:
    return metric_name in GEVAL_METRICS


def _metric_prediction_details(row: dict[str, Any], metric_name: str) -> dict[str, Any]:
    predictions = row.get("predictions") if isinstance(row.get("predictions"), dict) else {}
    details = predictions.get(metric_name)
    return details if isinstance(details, dict) else {}


def _geval_metric_details(metric_details: dict[str, Any]) -> dict[str, Any]:
    canonical = _canonical_metric_details(metric_details)
    details: dict[str, Any] = {
        "final_score": canonical.get("final_score"),
        "status": canonical.get("status"),
        "evaluation_variant": canonical.get("evaluation_variant"),
        "criteria": _geval_criteria_details(canonical),
    }
    for key in ("reference_anchors", "diagnostics", "incomplete_reason"):
        if key in canonical:
            details[key] = canonical[key]
    return details


def _geval_criteria_details(metric_details: dict[str, Any]) -> dict[str, Any]:
    criteria = metric_details.get("criteria")
    weights = metric_details.get("criterion_weights")
    if not isinstance(criteria, dict):
        return {}
    weights = weights if isinstance(weights, dict) else {}
    compact: dict[str, Any] = {}
    for criterion, result in sorted(criteria.items()):
        if not isinstance(result, dict):
            continue
        outcome = _geval_criterion_status(result)
        item: dict[str, Any] = {
            "status": outcome,
            "score": result.get("score"),
            "score_name": "criterion_score",
            "range": {"min": 1, "max": 5},
            "higher_is_better": True,
            "weight": weights.get(criterion),
            "feedback": result.get("feedback"),
            "parse_status": result.get("parse_status"),
        }
        for key in (
            "parse_repair_attempted",
            "parse_repair_succeeded",
            "parse_repair_method",
            "abstention_reason",
            "judge_error",
            "parse_error",
        ):
            if key in result:
                item[key] = result[key]
        compact[str(criterion)] = item
    return compact


def _geval_criterion_status(result: dict[str, Any]) -> str:
    outcome = result.get("parse_status")
    if outcome == "parsed":
        return "success"
    if outcome == "abstained":
        return "abstained"
    return "error"


def _nli_metric_details(metric_details: dict[str, Any]) -> dict[str, Any]:
    canonical = _canonical_metric_details(metric_details)
    source_to_candidate = {
        "entailment_probability": canonical.get("forward_entailment_probability"),
        "neutral_probability": canonical.get("forward_neutral_probability"),
        "contradiction_probability": canonical.get("forward_contradiction_probability"),
    }
    candidate_to_source = {
        "entailment_probability": canonical.get("reverse_entailment_probability"),
        "neutral_probability": canonical.get("reverse_neutral_probability"),
        "contradiction_probability": canonical.get("reverse_contradiction_probability"),
    }
    if "forward_tokenization" in canonical:
        source_to_candidate["tokenization"] = canonical["forward_tokenization"]
    if "reverse_tokenization" in canonical:
        candidate_to_source["tokenization"] = canonical["reverse_tokenization"]
    details = {
        "label": canonical.get("predicted_label_using_max_contradiction"),
        "source_to_candidate": source_to_candidate,
        "candidate_to_source": candidate_to_source,
    }
    for key in ("direction", "input_text", "generated_text", "row_index", "pair_id", "sample_id", "split", "primary_audio_class"):
        if key in canonical:
            details[key] = canonical[key]
    return details


def _canonical_metric_details(value: Any) -> Any:
    if isinstance(value, dict):
        return {
            key: _canonical_metric_details(item)
            for key, item in value.items()
            if not _is_hash_or_fingerprint_field(key)
        }
    if isinstance(value, list):
        return [_canonical_metric_details(item) for item in value]
    return value


def _is_hash_or_fingerprint_field(key: object) -> bool:
    if not isinstance(key, str):
        return False
    normalized = key.lower().replace("-", "_")
    return "hash" in normalized or "fingerprint" in normalized


def _manifest_error(metric: MetricSpec, row: dict[str, Any]) -> dict[str, Any] | None:
    errors = row.get("errors") if isinstance(row.get("errors"), dict) else {}
    if not errors:
        return None
    return {
        "type": _error_type(errors, metric.name, False),
        "message": _error_message(errors),
    }

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
    artifact_dir: Path,
    metric_backend_metadata: dict[str, dict],
) -> dict[str, Any]:
    selected = {spec.name for spec in options.metric_specs if spec.external}
    corpus_metrics: dict[str, Any] = {}
    backend_metrics = {
        "aac-metrics": ["mace", "spider", "aac_clapsim", "fense"],
        "openai-whisper": ["asr_wer", "dwer"],
        "utmosv2": ["utmos"],
        "torchmetrics-dnsmos": ["dnsmos"],
        "audioldm_eval": ["fad", "inception_score"],
        "kadtk": ["kad"],
        "stable-audio-metrics": [
            "target_caption_clap_score",
            "prompt_adherence_clap_score",
            "kl_passt",
        ],
        "dedicated-script": [
            "nli_contradiction",
            "g_eval_prometheus_plausibility",
            "g_eval_prometheus_reference",
        ],
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
        for metric_name in metric_names:
            if backend == "dedicated-script":
                result = _run_dedicated_script_metrics(
                    options, rows, metrics_config, artifact_dir, [metric_name]
                )
            else:
                valid_rows = _rows_without_metric_errors(rows, per_sample_rows, metric_name)
                if valid_rows:
                    result = _run_backend_worker(
                        options, valid_rows, dataset_root, metrics_config, work_dir,
                        artifact_dir, backend, [metric_name]
                    )
                else:
                    result = {"sample_metrics": {}, "corpus_metrics": {}, "predictions": {}, "errors": {}}
            _merge_backend_result(per_sample_rows, result)
            for name, metadata in result.get("metric_backends", {}).items():
                if name in metric_backend_metadata:
                    metric_backend_metadata[name].update(metadata)
            # Prefer backend-provided official corpus scores in summaries while
            # retaining sentence-level scores in the per-sample JSONL.
            corpus_metrics.update(result.get("corpus_metrics", {}))
    return corpus_metrics


def _run_dedicated_script_metrics(
    options: EvalOptions,
    rows: list[dict],
    metrics_config: dict[str, dict],
    work_dir: Path,
    metric_names: list[str],
) -> dict[str, Any]:
    supported_order = [
        "nli_contradiction",
        "g_eval_prometheus_plausibility",
        "g_eval_prometheus_reference",
    ]
    ordered_names = [name for name in supported_order if name in metric_names]
    if sorted(ordered_names) != sorted(metric_names):
        raise MetricUnavailableError(f"Unsupported dedicated-script metrics: {metric_names}")

    merged: dict[str, Any] = {
        "sample_metrics": {},
        "predictions": {},
        "errors": {},
        "metric_backends": {},
    }
    prediction_context: dict[str, dict[str, Any]] = {}
    for metric_name in ordered_names:
        result = _run_single_dedicated_script_metric(
            options,
            rows,
            metrics_config,
            work_dir,
            metric_name,
            prediction_context,
        )
        _merge_result_dicts(merged, result)
        for row_index, predictions in result.get("predictions", {}).items():
            prediction_context.setdefault(str(row_index), {}).update(predictions)
    return merged


def _run_single_dedicated_script_metric(
    options: EvalOptions,
    rows: list[dict],
    metrics_config: dict[str, dict],
    work_dir: Path,
    metric_name: str,
    prediction_context: dict[str, dict[str, Any]],
) -> dict[str, Any]:
    config = metrics_config.get(metric_name, {})
    script = Path(config.get("script", _default_dedicated_script(metric_name)))
    if not script.is_absolute():
        script = Path(__file__).resolve().parents[1] / script
    if not script.is_file():
        raise MetricUnavailableError(f"Dedicated metric script does not exist for {metric_name}: {script}")

    metric_work_dir = work_dir / "dedicated-script" / metric_name
    metric_work_dir.mkdir(parents=True, exist_ok=True)
    input_path = metric_work_dir / "input.jsonl"
    output_path = metric_work_dir / "output.jsonl"
    summary_path = metric_work_dir / "summary.json"
    stdout_path = metric_work_dir / "stdout.log"

    helper_rows, errors = _dedicated_text_pair_rows(options, rows, metric_name, prediction_context)
    write_jsonl(input_path, helper_rows)
    if helper_rows:
        python = Path(config.get("python") or sys.executable)
        command = _dedicated_script_command(
            metric_name,
            python,
            script,
            input_path,
            output_path,
            summary_path,
            metric_work_dir,
            config,
            options,
        )
        with stdout_path.open("w", encoding="utf-8") as stdout_file:
            process = subprocess.Popen(
                command,
                text=True,
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                cwd=Path(__file__).resolve().parents[1],
                env=_backend_env(options, {metric_name: config}),
            )
            assert process.stdout is not None
            with process.stdout:
                for line in process.stdout:
                    stdout_file.write(line)
                    stdout_file.flush()
                    print(line, end="", flush=True)
            returncode = process.wait()
        if returncode != 0:
            raise MetricUnavailableError(f"Dedicated metric {metric_name!r} failed. See {stdout_path} for details.")
    else:
        output_path.write_text("", encoding="utf-8")
        write_json(summary_path, {"total_pairs_processed": 0, "metric": metric_name, "judge_model": config.get("model_name")})

    sample_metrics: dict[str, dict[str, float]] = {}
    predictions: dict[str, dict[str, Any]] = {}
    if output_path.is_file():
        for item in iter_jsonl(output_path):
            result = item.record
            row_index = str(result["row_index"])
            metric_value = _dedicated_metric_value(metric_name, result)
            if metric_value is not None:
                sample_metrics[row_index] = {metric_name: metric_value}
            elif metric_name in {
                "g_eval_prometheus_plausibility",
                "g_eval_prometheus_reference",
            } and result.get("status") != "complete":
                errors.setdefault(row_index, {})[metric_name] = str(
                    result.get("incomplete_reason") or "incomplete_criterion_judgment"
                )
            predictions[row_index] = {metric_name: result}
    return {
        "sample_metrics": sample_metrics,
        "predictions": predictions,
        "errors": errors,
        "metric_backends": {
            metric_name: {
                "backend": "dedicated-script",
                "script": str(script),
                "work_dir": str(metric_work_dir),
                "output": str(output_path),
                "summary": str(summary_path),
                "criterion_judgments": str(metric_work_dir / "criterion_judgments.jsonl"),
            }
        },
    }


def _default_dedicated_script(metric_name: str) -> str:
    defaults = {
        "nli_contradiction": "scripts/run_nli_contradiction_helper.py",
        "g_eval_prometheus_plausibility": "scripts/run_geval_prometheus_criterion_v2.py",
        "g_eval_prometheus_reference": "scripts/run_geval_prometheus_criterion_v2.py",
    }
    if metric_name not in defaults:
        raise MetricUnavailableError(f"No default dedicated script for metric {metric_name!r}")
    return defaults[metric_name]


def _dedicated_text_pair_rows(
    options: EvalOptions,
    rows: list[dict],
    metric_name: str,
    prediction_context: dict[str, dict[str, Any]],
) -> tuple[list[dict[str, Any]], dict[str, dict[str, str]]]:
    helper_rows: list[dict[str, Any]] = []
    errors: dict[str, dict[str, str]] = {}
    for ordinal, row in enumerate(rows):
        row_index = row.get("row_index", ordinal)
        if not is_successful_generation(row):
            continue
        try:
            source = source_record(row)
            if metric_name in {
                "g_eval_prometheus_plausibility",
                "g_eval_prometheus_reference",
            }:
                stage_input, stage_output, stage_direction = _final_text_stage(row)
                generated_text = stage_output or _prediction_text_for_metric(row, metric_name)
                input_text = stage_input
                direction = stage_direction
            elif metric_name == "nli_contradiction":
                stage_input, stage_output, stage_direction = _final_text_stage(row)
                if stage_input or stage_output or stage_direction:
                    generated_text = stage_output or _prediction_text_for_metric(row, metric_name)
                    input_text = stage_input
                    direction = stage_direction
                else:
                    generated_text = _prediction_text_for_metric(row, metric_name)
                    input_text, direction = _nli_input_text_and_direction(options.task, row, source)
            else:
                generated_text = _prediction_text_for_metric(row, metric_name)
                input_text, direction = _nli_input_text_and_direction(options.task, row, source)
            if not generated_text:
                raise EvaluationError(f"{metric_name} requires generated text in candidate_text/generated_text/model_extra")
            if not input_text:
                input_text, fallback_direction = _nli_input_text_and_direction(options.task, row, source)
                direction = direction or fallback_direction
            if direction not in {"speech_to_audio", "audio_to_speech"}:
                direction = _infer_t2t_direction(row, source)
            helper_row = {
                "row_index": row_index,
                "pair_id": sample_id(row, ordinal),
                "sample_id": sample_id(row, ordinal),
                "split": options.split,
                "direction": direction,
                "primary_audio_class": source.get("primary_audio_class") or row.get("primary_audio_class"),
                "input_text": input_text,
                "generated_text": generated_text,
            }
            if metric_name == "g_eval_prometheus_reference":
                reference_text = _paired_reference_text(source, direction)
                helper_row["reference_anchors"] = (
                    [
                        {
                            "anchor_id": sample_id(row, ordinal),
                            "text": reference_text,
                            "source": "paired_dataset_reference",
                        }
                    ]
                    if reference_text
                    else []
                )
            helper_rows.append(helper_row)
        except Exception as exc:
            errors[str(row_index)] = {metric_name: f"{type(exc).__name__}: {exc}"}
    return helper_rows, errors


def _dedicated_script_command(
    metric_name: str,
    python: Path,
    script: Path,
    input_path: Path,
    output_path: Path,
    summary_path: Path,
    metric_work_dir: Path,
    config: dict[str, Any],
    options: EvalOptions,
) -> list[str]:
    if metric_name == "nli_contradiction":
        command = [
            str(python),
            str(script),
            "--input",
            str(input_path),
            "--output",
            str(output_path),
            "--summary-output",
            str(summary_path),
            "--backend",
            str(config.get("script_backend", config.get("runtime_backend", "hf"))),
            "--model",
            str(
                config.get(
                    "model_name",
                    config.get("model_path")
                    or "MoritzLaurer/DeBERTa-v3-large-mnli-fever-anli-ling-wanli",
                )
            ),
            "--device",
            options.device,
            "--batch-size",
            str(config.get("batch_size", 8)),
            "--contradiction-threshold",
            str(config.get("contradiction_threshold", 0.7)),
        ]
        if (
            _truthy_config(config.get("local_files_only"))
            or _truthy_config(config.get("hf_hub_offline"))
            or _truthy_config(config.get("transformers_offline"))
        ):
            command.append("--local-files-only")
        return command
    if metric_name in {
        "g_eval_prometheus_plausibility",
        "g_eval_prometheus_reference",
    }:
        variant = "reference_grounded" if metric_name == "g_eval_prometheus_reference" else "plausibility"
        command = [
            str(python), str(script), "--input", str(input_path), "--output", str(output_path),
            "--summary-output", str(summary_path), "--summary-csv-output", str(metric_work_dir / "summary.csv"),
            "--criterion-judgments-output", str(metric_work_dir / "criterion_judgments.jsonl"),
            "--variant", variant,
            "--backend", str(config.get("script_backend", config.get("runtime_backend", "vllm"))),
            "--model", str(config.get("model_name", config.get("model_path") or str(model_path("prometheus_8x7b_v2")))),
            "--batch-size", str(config.get("batch_size", 8)),
            "--cache-dir", str(config.get("cache_dir") or metric_work_dir / "cache"),
            "--tensor-parallel-size", str(config.get("tensor_parallel_size", 4)),
            "--temperature", str(config.get("temperature", 0.0)),
            "--max-tokens", str(config.get("max_tokens", 512)),
            "--gpu-memory-utilization", str(config.get("gpu_memory_utilization", 0.9)),
        ]
        if config.get("max_model_len") is not None:
            command.extend(["--max-model-len", str(config["max_model_len"])])
        if config.get("enforce_eager"):
            command.append("--enforce-eager")
        if config.get("trust_remote_code", True):
            command.append("--trust-remote-code")
        return command
    raise MetricUnavailableError(f"Unsupported dedicated-script metric: {metric_name}")


def _truthy_config(value: Any) -> bool:
    if isinstance(value, bool):
        return value
    if isinstance(value, str):
        return value.strip().lower() in {"1", "true", "yes", "on"}
    return bool(value)

def _dedicated_metric_value(metric_name: str, result: dict[str, Any]) -> float | None:
    if metric_name == "nli_contradiction":
        return float(result["max_contradiction_probability"])
    if metric_name in {
        "g_eval_prometheus_plausibility",
        "g_eval_prometheus_reference",
    } and isinstance(result.get("final_score"), (int, float)):
        return float(result["final_score"])
    return None


def _paired_reference_text(source: dict[str, Any], direction: str) -> str | None:
    field = "speech_text" if direction == "audio_to_speech" else "audio_cap"
    value = source.get(field)
    if not value and field == "audio_cap":
        value = source.get("audio_caption")
    return value.strip() if isinstance(value, str) and value.strip() else None


def _merge_result_dicts(target: dict[str, Any], source: dict[str, Any]) -> None:
    for key in ("sample_metrics", "predictions", "errors"):
        for row_index, values in source.get(key, {}).items():
            target.setdefault(key, {}).setdefault(str(row_index), {}).update(values)
    target.setdefault("metric_backends", {}).update(source.get("metric_backends", {}))


def _rows_without_metric_errors(
    rows: list[dict],
    per_sample_rows: list[dict],
    metric_name: str,
) -> list[dict]:
    errors_by_index = {
        str(row["row_index"]): row.get("errors", {})
        for row in per_sample_rows
    }
    return [
        row
        for ordinal, row in enumerate(rows)
        if metric_name not in errors_by_index.get(str(row.get("row_index", ordinal)), {})
    ]


def _run_backend_worker(
    options: EvalOptions,
    rows: list[dict],
    dataset_root: Path,
    metrics_config: dict[str, dict],
    work_dir: Path,
    artifact_dir: Path,
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
            value = _source_field_value(source, field, dataset_root)
            if isinstance(value, str) and value:
                backend_source[field] = str(_resolve_reference(dataset_root, value))
        backend_rows.append(
            {
                "row_index": row_index,
                "sample_id": sample_id(row, ordinal),
                "output_file": row.get("output_file"),
                "candidate_text": _prediction_text_for_metric(row, metric_names[0]),
                "conditioning_prompt": row.get("prompt"),
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
    backend_artifact_dir = artifact_dir / backend
    backend_artifact_dir.mkdir(parents=True, exist_ok=True)
    input_path = backend_artifact_dir / "input.json"
    output_path = backend_artifact_dir / "output.json"
    stdout_path = backend_artifact_dir / "stdout.log"
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
    repo_root = Path(__file__).resolve().parents[1]
    cache_dir = _first_metric_config_value(metric_configs, "cache_dir") or str(options.cache_dir)
    cache_path = Path(cache_dir)
    hf_home = _first_metric_config_value(metric_configs, "hf_home")
    transformers_cache = _first_metric_config_value(metric_configs, "transformers_cache")
    torch_home = _first_metric_config_value(metric_configs, "torch_home")
    env["XDG_CACHE_HOME"] = str(cache_path)
    env["HF_HOME"] = str(Path(hf_home)) if hf_home else str(cache_path.parent / "huggingface")
    env["HF_HUB_CACHE"] = env["HF_HOME"]
    env["HUGGINGFACE_HUB_CACHE"] = env["HF_HOME"]
    env["TRANSFORMERS_CACHE"] = (
        str(Path(transformers_cache)) if transformers_cache else env["HF_HOME"]
    )
    env["TORCH_HOME"] = str(Path(torch_home)) if torch_home else str(cache_path.parent / "torch")
    env["FENSE_DATA"] = str(cache_path / "fense_data")
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
    configured_pythonpath = _first_metric_config_value(metric_configs, "pythonpath")
    pythonpath_entries = [
        str(repo_root),
        *_split_pythonpath(configured_pythonpath),
        *_split_pythonpath(env.get("PYTHONPATH")),
    ]
    unique_pythonpath_entries: list[str] = []
    seen_pythonpath_entries: set[Path] = set()
    for entry in pythonpath_entries:
        path = Path(entry)
        resolved = path.resolve() if path.is_absolute() else (repo_root / path).resolve()
        if resolved in seen_pythonpath_entries:
            continue
        seen_pythonpath_entries.add(resolved)
        unique_pythonpath_entries.append(entry)
    env["PYTHONPATH"] = os.pathsep.join(unique_pythonpath_entries)
    return env


def _split_pythonpath(value: str | None) -> list[str]:
    return [entry for entry in value.split(os.pathsep) if entry] if value else []


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


def _record_nonfinite_metric_errors(
    per_sample_rows: list[dict[str, Any]],
    corpus_metrics: dict[str, Any],
) -> None:
    for row in per_sample_rows:
        metrics = row.get("metrics")
        if not isinstance(metrics, dict):
            continue
        for metric, value in list(metrics.items()):
            if _contains_nonfinite_number(value):
                metrics.pop(metric)
                row["status"] = "error"
                row.setdefault("errors", {})[metric] = (
                    "metric returned a non-finite numeric value"
                )

    for metric, value in list(corpus_metrics.items()):
        if not _contains_nonfinite_number(value):
            continue
        corpus_metrics.pop(metric)
        error_name = f"{metric}_corpus_summary"
        for row in per_sample_rows:
            row.setdefault("errors", {})[error_name] = (
                "corpus metric returned a non-finite numeric value"
            )


def _contains_nonfinite_number(value: Any) -> bool:
    if isinstance(value, bool):
        return False
    if isinstance(value, (int, float)):
        return not math.isfinite(float(value))
    if isinstance(value, dict):
        return any(_contains_nonfinite_number(item) for item in value.values())
    if isinstance(value, (list, tuple)):
        return any(_contains_nonfinite_number(item) for item in value)
    return False


def _nli_input_text_and_direction(task: str, row: dict, source: dict) -> tuple[str, str]:
    direction = row.get("direction")
    if direction in {"speech_to_audio", "audio_to_speech"}:
        candidate_fields = ("speech_text",) if direction == "speech_to_audio" else ("audio_cap", "audio_caption")
    elif task == "T2T":
        explicit = _first_text_value(row, source, ("input_text", "prompt"))
        if explicit:
            return explicit, _infer_t2t_direction(row, source)
        direction = _infer_t2t_direction(row, source)
        candidate_fields = ("speech_text",) if direction == "speech_to_audio" else ("audio_cap", "audio_caption", "speech_text")
    else:
        raise EvaluationError("nli_contradiction is defined for the T2T/intermediate-text stage or explicit text-pair rows")
    for field in candidate_fields:
        value = source.get(field) or row.get(field)
        if isinstance(value, str) and value.strip():
            return value, str(direction)
    raise EvaluationError(f"nli_contradiction could not find input text for direction {direction}")


def _infer_t2t_direction(row: dict, source: dict) -> str:
    _, _, stage_direction = _final_text_stage(row)
    if stage_direction:
        return stage_direction
    mode = (
        _nested_text(row, ("model_extra", "mode"))
        or _nested_text(row, ("text_mode",))
        or _nested_text(source, ("stage_provenance", "mode"))
    )
    pipeline = row.get("pipeline") or source.get("pipeline") or _nested_text(source, ("stage_provenance", "pipeline"))
    variant = row.get("variant")
    if mode == "speech_to_audio_description" or pipeline == "S2A" or _looks_like_lane(variant, "s2a"):
        return "speech_to_audio"
    if mode == "audio_description_to_speech" or pipeline == "A2S" or _looks_like_lane(variant, "a2s"):
        return "audio_to_speech"
    return "text_to_text"


def _final_text_stage(row: dict) -> tuple[str | None, str | None, str | None]:
    extra = row.get("model_extra") if isinstance(row.get("model_extra"), dict) else {}
    stages = extra.get("stages") if isinstance(extra.get("stages"), list) else []
    stage = stages[-1] if stages and isinstance(stages[-1], dict) else {}
    stage_input = stage.get("input") if isinstance(stage.get("input"), dict) else {}
    stage_output = stage.get("output") if isinstance(stage.get("output"), dict) else {}
    mode = stage.get("mode")
    if mode == "speech_to_audio_description":
        direction = "speech_to_audio"
    elif mode == "audio_description_to_speech":
        direction = "audio_to_speech"
    else:
        direction = None
    return (
        _nonempty_text(stage_input.get("text")),
        _nonempty_text(stage_output.get("text")),
        direction,
    )


def _nonempty_text(value: object) -> str | None:
    return value.strip() if isinstance(value, str) and value.strip() else None


def _looks_like_lane(value: object, token: str) -> bool:
    return isinstance(value, str) and token in value.lower().split("_")


def _first_text_value(row: dict, source: dict, fields: tuple[str, ...]) -> str | None:
    extra = row.get("model_extra") if isinstance(row.get("model_extra"), dict) else {}
    for field in fields:
        for record in (row, source, extra):
            value = record.get(field) if isinstance(record, dict) else None
            if isinstance(value, str) and value.strip():
                return value
    return None


def _nested_text(record: dict, keys: tuple[str, ...]) -> str | None:
    value: object = record
    for key in keys:
        if not isinstance(value, dict):
            return None
        value = value.get(key)
    return value.strip() if isinstance(value, str) and value.strip() else None


def _prediction_text_for_metric(row: dict, metric_name: str | None = None) -> str | None:
    if metric_name == "prompt_adherence_clap_score":
        prompt = row.get("prompt")
        return prompt if isinstance(prompt, str) and prompt.strip() else None
    candidate_text = row.get("candidate_text")
    if isinstance(candidate_text, str) and candidate_text.strip():
        return candidate_text
    generated_text = row.get("generated_text")
    if isinstance(generated_text, str) and generated_text.strip():
        return generated_text
    extra = row.get("model_extra")
    if not isinstance(extra, dict):
        return None
    if metric_name in {"wer", "semantic_equivalence"}:
        keys = ("understanding_text", "decoded_text", "converted_prompt", "generation_prompt")
    elif metric_name in {"mace", "spider", "aac_clapsim", "fense"}:
        keys = ("converted_prompt", "generation_prompt", "decoded_text", "understanding_text")
    elif metric_name == "nli_contradiction":
        keys = ("generated_text", "understanding_text", "decoded_text")
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


def _source_context(source: dict, dataset_root: Path) -> dict:
    context: dict[str, Any] = {}
    for field in ("audio_file", "speech_file"):
        value = _source_field_value(source, field, dataset_root)
        if isinstance(value, str) and value:
            context[field] = str(_resolve_reference(dataset_root, value))
    for field in ("primary_audio_class", "primary_audio_tag", "pipeline"):
        value = source.get(field)
        if value is not None:
            context[field] = value
    provenance = source.get("stage_provenance")
    if isinstance(provenance, dict):
        for field in ("stage", "direction", "mode", "pipeline"):
            value = provenance.get(field)
            if value is not None:
                context[f"source_{field}"] = value
    return context


def _references(source: dict, specs: list[MetricSpec], dataset_root: Path) -> dict:
    fields = sorted({field for spec in specs for field in spec.required_source_fields})
    refs = {}
    for field in fields:
        value = _source_field_value(source, field, dataset_root)
        if field in {"audio_file", "speech_file"} and isinstance(value, str):
            refs[field] = str(_resolve_reference(dataset_root, value))
        else:
            refs[field] = value
    return refs


def _source_field_value(source: dict, field: str, dataset_root: Path) -> Any:
    value = source.get(field)
    if field not in {"audio_file", "speech_file"}:
        return value
    if isinstance(value, str) and value.strip() and _resolve_reference(dataset_root, value).is_file():
        return value

    # Cascaded stage manifests may store synthetic stage-local paths at the
    # top level while preserving the original TTAS pair under nested source keys.
    for candidate in _nested_source_values(source, field):
        if isinstance(candidate, str) and candidate.strip() and _resolve_reference(dataset_root, candidate).is_file():
            return candidate
    return value


def _nested_source_values(source: dict, field: str) -> list[Any]:
    candidates: list[Any] = []
    original_row = source.get("original_row")
    if isinstance(original_row, dict):
        for key in ("_demo_source", "source", "original_source"):
            nested = original_row.get(key)
            if isinstance(nested, dict):
                candidates.append(nested.get(field))
        nested_original = original_row.get("original_row")
        if isinstance(nested_original, dict):
            candidates.append(nested_original.get(field))
    for key in ("_demo_source", "original_source"):
        nested = source.get(key)
        if isinstance(nested, dict):
            candidates.append(nested.get(field))
    return candidates


def _resolve_reference(dataset_root: Path, value: str) -> Path:
    path = Path(value)
    return path if path.is_absolute() else dataset_root / path


def _summarize(
    options: EvalOptions,
    eval_run_id: str,
    rows: list[dict],
    backend_metadata: dict[str, dict],
    corpus_metrics: dict[str, Any] | None,
    validation_report: dict[str, Any],
    total_seconds: float | None = None,
) -> dict:
    corpus_metrics = corpus_metrics or {}
    metric = options.metric_specs[0]
    stats = _metric_statistics(metric, rows, corpus_metrics)
    return {
        "schema_version": 1,
        "run_id": eval_run_id,
        "status": "complete",
        "metric": {
            "metric_id": metric.name,
            "level": metric.level,
            "higher_is_better": _higher_is_better(metric.name),
        },
        "target": {
            "kind": "stage",
            "stage": options.stage,
            "direction": options.direction,
            "model": options.model,
            "dataset": options.dataset,
            "split": options.split,
            "run_id": options.evaluated_run_id,
            "manifest": str(options.manifest_path),
        },
        "samples": _sample_counts(metric, rows),
        "scores": _score_summary(metric, stats),
        "errors": _summary_errors(rows, validation_report),
        "artifacts": {
            "manifest": "manifest.jsonl",
            "errors": "errors.json",
            "inspect": "inspect.json",
            "backend_outputs": "artifacts/",
        },
        "timing": {
            "total_seconds": round(total_seconds, 3) if isinstance(total_seconds, (int, float)) else None,
            "backend_seconds": None,
            "mean_seconds_per_sample": None,
        },
    }


def _metric_statistics(metric: MetricSpec, rows: list[dict], corpus_metrics: dict[str, Any]) -> dict | None:
    if metric.name in corpus_metrics:
        return _corpus_summary(corpus_metrics[metric.name])
    values = [
        row["metrics"][metric.name]
        for row in rows
        if isinstance(row.get("metrics"), dict) and isinstance(row["metrics"].get(metric.name), (int, float))
    ]
    if values:
        return _numeric_summary(values)
    if metric.level == "corpus":
        return {"level": "corpus", "status": "not_computed_in_scaffold"}
    return None


def _sample_counts(metric: MetricSpec, rows: list[dict]) -> dict[str, Any]:
    skipped = sum(1 for row in rows if row.get("status") == "skipped")
    failed = sum(1 for row in rows if row.get("status") == "error")
    included = max(0, len(rows) - skipped - failed)
    if metric.level == "corpus":
        return {
            "total": len(rows),
            "successful": None,
            "included": included,
            "skipped": skipped,
            "failed": failed,
            "successful_note": "not_applicable_for_corpus_metric",
        }
    successful = sum(1 for row in rows if row.get("status") == "success")
    return {
        "total": len(rows),
        "successful": successful,
        "included": successful,
        "skipped": skipped,
        "failed": failed,
    }


def _score_summary(metric: MetricSpec, stats: dict | None) -> dict[str, Any]:
    primary = {
        "name": _metric_score_name(metric.name),
        "value": _primary_score_value(stats),
        "aggregation": "corpus" if metric.level == "corpus" else "mean",
        "higher_is_better": _higher_is_better(metric.name),
    }
    score_range = _metric_score_range(metric.name)
    if score_range is not None:
        primary["range"] = score_range
    interpretation = _metric_score_interpretation(metric.name)
    if interpretation is not None:
        primary["interpretation"] = interpretation
    return {
        "primary": primary,
        "statistics": stats if metric.level == "sample" else None,
    }


def _primary_score_value(stats: dict | None) -> float | int | None:
    if not isinstance(stats, dict):
        return None
    for key in ("mean", "score", "value"):
        value = stats.get(key)
        if isinstance(value, (int, float)) and not isinstance(value, bool):
            return value
    return None


def _summary_errors(rows: list[dict], validation_report: dict[str, Any]) -> dict[str, Any]:
    by_type: dict[str, int] = {}
    count = 0
    for row in rows:
        errors = row.get("errors") if isinstance(row.get("errors"), dict) else {}
        if not errors:
            continue
        count += 1
        error_type = "TargetRowNotSuccessful" if "row" in errors else "MetricError"
        by_type[error_type] = by_type.get(error_type, 0) + 1
    return {
        "count": count,
        "by_type": by_type,
        "validation": {
            "status": validation_report.get("status"),
            "result_status": validation_report.get("result_status"),
            "scientifically_valid": validation_report.get("scientifically_valid"),
        },
    }


def _higher_is_better(metric_name: str) -> bool | None:
    lower_is_better = {"wer", "asr_wer", "dwer", "fad", "kl_passt", "nli_contradiction"}
    if metric_name in lower_is_better:
        return False
    return True


def _metric_score_name(metric_name: str) -> str:
    if metric_name == "nli_contradiction":
        return "max_contradiction_probability"
    if _is_geval_metric(metric_name):
        return "weighted_final_score"
    return metric_name


def _metric_score_range(metric_name: str) -> dict[str, float] | None:
    if metric_name == "nli_contradiction":
        return {"min": 0.0, "max": 1.0}
    if metric_name in {
        "g_eval_prometheus_plausibility",
        "g_eval_prometheus_reference",
    }:
        return {"min": 1.0, "max": 5.0}
    return None


def _metric_score_interpretation(metric_name: str) -> str | None:
    if metric_name == "nli_contradiction":
        return "contradiction_risk"
    return None


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


def _run_document(
    options: EvalOptions,
    eval_run_id: str,
    started_at: str,
    validation_report: dict[str, Any],
    paths: dict[str, Path],
) -> dict:
    metric = options.metric_specs[0]
    return {
        "schema_version": 1,
        "run_id": eval_run_id,
        "temporary": options.temporary,
        "status": "complete",
        "metric": {
            "metric_id": metric.name,
            "level": metric.level,
            "reference_pairing_required": bool(metric.required_source_fields),
        },
        "target": {
            "kind": "stage",
            "category": "cascaded",
            "stage": options.stage,
            "direction": options.direction,
            "model": options.model,
            "dataset": options.dataset,
            "split": options.split,
            "run_id": options.evaluated_run_id,
            "run_path": str(options.manifest_path.parent),
            "manifest": str(options.manifest_path),
            "summary": str(options.manifest_path.with_name("summary.json")),
        },
        "dataset": options.dataset,
        "split": options.split,
        "started_at": started_at,
        "completed_at": utc_timestamp(),
        "code": _code_state(),
        "slurm": _slurm_identity(),
        "paths": {
            "result": str(paths["run"].parent),
            "logs": _eval_logs_path(paths["run"].parent),
        },
    }


def _write_eval_log(run_doc: dict[str, Any], summary: dict[str, Any], run_config: dict[str, Any]) -> None:
    logs_path = ((run_doc.get("paths") or {}).get("logs"))
    if not isinstance(logs_path, str) or not logs_path:
        return
    metric = summary.get("metric") if isinstance(summary.get("metric"), dict) else {}
    samples = summary.get("samples") if isinstance(summary.get("samples"), dict) else {}
    lines = [
        f"run_id={run_doc.get('run_id')}",
        f"status={run_doc.get('status')}",
        f"metric_id={metric.get('metric_id')}",
        f"num_samples={(run_config.get('execution') or {}).get('num_samples')}",
        f"sample_total={samples.get('total')}",
        f"result_path={(run_doc.get('paths') or {}).get('result')}",
    ]
    write_text_create_only(Path(logs_path) / "run.log", "\n".join(lines) + "\n")


def _code_state() -> dict[str, Any]:
    return {
        "git_commit": _git_output("rev-parse", "HEAD"),
        "git_dirty": _git_dirty(),
    }


def _git_output(*args: str) -> str | None:
    try:
        completed = subprocess.run(
            ["git", *args],
            cwd=Path(__file__).resolve().parents[1],
            check=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.DEVNULL,
            text=True,
        )
    except Exception:
        return None
    value = completed.stdout.strip()
    return value or None


def _git_dirty() -> bool | None:
    value = _git_output("status", "--porcelain")
    if value is None:
        return None
    return bool(value)


def _eval_logs_path(result_dir: Path) -> str | None:
    parts = list(result_dir.parts)
    for index in range(len(parts) - 2):
        if parts[index:index + 3] == ["temporary", "results", "evals"]:
            parts[index:index + 3] = ["temporary", "logs", "evals"]
            return str(Path(*parts))
    for index in range(len(parts) - 1):
        if parts[index:index + 2] == ["results", "evals"]:
            parts[index:index + 2] = ["logs", "evals"]
            return str(Path(*parts))
    for index, part in enumerate(parts):
        if part == "evals":
            parts[index:index + 1] = ["logs", "evals"]
            return str(Path(*parts))
    return None


def _slurm_identity() -> dict[str, str | None]:
    return {
        "job_id": os.environ.get("SLURM_JOB_ID"),
        "array_job_id": os.environ.get("SLURM_ARRAY_JOB_ID"),
        "array_task_id": os.environ.get("SLURM_ARRAY_TASK_ID"),
    }


def _errors_document(
    options: EvalOptions,
    eval_run_id: str,
    rows: list[dict[str, Any]],
    validation_report: dict[str, Any],
) -> dict:
    metric_id = options.metric_specs[0].name
    failed_sample_ids = set(validation_report.get("failed_sample_ids", {}).get(metric_id, []))
    entries = [
        _error_entry(row, options.metric_specs[0], row.get("sample_id") in failed_sample_ids)
        for row in rows
        if row.get("status") in {"skipped", "error"}
        or row.get("errors")
        or (row.get("status") != "included" and row.get("sample_id") in failed_sample_ids)
    ]
    return {
        "schema_version": 1,
        "run_id": eval_run_id,
        "metric_id": metric_id,
        "error_count": sum(1 for entry in entries if entry["status"] == "error"),
        "skipped_count": sum(1 for entry in entries if entry["status"] == "skipped"),
        "errors": entries,
    }


def _error_entry(row: dict[str, Any], metric: MetricSpec, validation_failed: bool) -> dict:
    errors = row.get("errors") if isinstance(row.get("errors"), dict) else {}
    error_type = _error_type(errors, metric.name, validation_failed)
    status = "skipped" if error_type == "TargetRowNotSuccessful" else "error"
    message = _error_message(errors)
    if message is None and validation_failed:
        message = "result validation failed for this sample"
    return {
        "row_index": row.get("row_index"),
        "sample_id": row.get("sample_id"),
        "status": status,
        "error_type": error_type,
        "message": message,
        "input": _manifest_input(metric, row),
        "log_ref": None,
        "artifact_ref": None,
    }


def _error_type(errors: dict[str, Any], metric_id: str, validation_failed: bool) -> str:
    if "row" in errors:
        return "TargetRowNotSuccessful"
    if metric_id in errors:
        return "MetricError"
    if errors:
        return "EvaluationError"
    if validation_failed:
        return "ResultValidationError"
    return "EvaluationError"


def _error_message(errors: dict[str, Any]) -> str | None:
    if not errors:
        return None
    if len(errors) == 1:
        return str(next(iter(errors.values())))
    return json.dumps(errors, ensure_ascii=False, sort_keys=True)


def _generated_input(row: dict[str, Any]) -> dict[str, Any]:
    generated = {}
    output_file = row.get("output_file")
    if output_file is not None:
        generated["output_file"] = output_file
    predictions = row.get("predictions")
    if isinstance(predictions, dict) and predictions:
        generated["predictions"] = predictions
    return generated


def _inspect_document(
    options: EvalOptions,
    eval_run_id: str,
    rows: list[dict[str, Any]],
    backend_metadata: dict[str, dict],
    summary: dict,
    validation_report: dict[str, Any],
) -> dict:
    metric = options.metric_specs[0]
    include_samples = metric.level == "sample"
    return {
        "schema_version": 1,
        "run_id": eval_run_id,
        "metric": {
            "metric_id": metric.name,
            "level": metric.level,
        },
        "target": {
            "kind": "stage",
            "stage": options.stage,
            "direction": options.direction,
            "model": options.model,
            "dataset": options.dataset,
            "split": options.split,
            "run_id": options.evaluated_run_id,
            "manifest": str(options.manifest_path),
        },
        "config_summary": {
            "backend": backend_metadata.get(metric.name, {}),
            "input_contract": {
                "prediction_fields": list(metric.required_prediction_fields),
                "reference_fields": list(metric.required_source_fields),
                "context_fields": [],
                "requires_output_file": metric.requires_output_file,
                "requires_hypothesis_text": metric.requires_hypothesis_text,
            },
        },
        "summary": {
            "status": summary.get("status"),
            "samples": summary.get("samples", {}),
            "scores": summary.get("scores", {}),
            "result_validation": validation_report,
        },
        "sample_policy": {
            "mode": "all" if include_samples else "omitted",
            "count": len(rows) if include_samples else 0,
            **({} if include_samples else {"reason": "corpus_metric_has_no_sample_level_outputs; manifest.jsonl records included rows"}),
        },
        "samples": _inspect_samples(metric, rows) if include_samples else [],
        "backend_artifacts": {
            "directory": "artifacts/",
        },
    }


def _inspect_samples(metric: MetricSpec, rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    return [_inspect_sample(metric, row) for row in rows]


def _inspect_sample(metric: MetricSpec, row: dict[str, Any]) -> dict[str, Any]:
    return {
        "row_index": row.get("row_index"),
        "sample_id": row.get("sample_id"),
        "status": row.get("status"),
        "input": _manifest_input(metric, row),
        "intermediate": {},
        "normalization": {},
        "output": _manifest_output(metric, row),
        "error": _manifest_error(metric, row),
    }

def _run_config(
    options: EvalOptions,
    backend_metadata: dict[str, dict],
    validation_report: dict[str, Any],
    num_samples: int,
) -> dict:
    metric = options.metric_specs[0]
    return {
        "schema_version": 1,
        "input": {
            "target_manifest": str(options.manifest_path),
            "target_run": str(options.manifest_path.parent),
            "dataset": options.dataset,
            "split": options.split,
            "start_index": options.start_index if options.temporary else None,
            "max_samples": options.max_samples if options.temporary else None,
        },
        "metric": {
            "metric_id": metric.name,
            "display_name": metric.name,
            "level": metric.level,
            "reference_pairing_required": bool(metric.required_source_fields),
            "higher_is_better": _higher_is_better(metric.name),
            "score_name": _metric_score_name(metric.name),
            "score_range": _metric_score_range(metric.name),
            "score_interpretation": _metric_score_interpretation(metric.name),
            "input_contract": {
                "prediction_fields": list(metric.required_prediction_fields),
                "reference_fields": list(metric.required_source_fields),
                "context_fields": [],
            },
        },
        "backend": _config_backend(metric, backend_metadata.get(metric.name, {})),
        "execution": {
            "command": options.command,
            "num_samples": num_samples,
            "temporary": options.temporary,
            "device": options.device,
            "cache_dir": str(options.cache_dir),
            "timestamp_utc": utc_timestamp(),
        },
    }


def _config_backend(metric: MetricSpec, metadata: dict[str, Any]) -> dict[str, Any]:
    if not metric.external:
        return {
            "name": "metric_algorithm",
            "implementation": _internal_metric_implementation(metric.name),
            "parameters": {},
        }
    return {
        "name": metadata.get("backend"),
        "implementation": metadata.get("package"),
        "python": metadata.get("python"),
        "model": {
            "name": metadata.get("model_name"),
            "path": metadata.get("model_path"),
        },
        "cache_dir": metadata.get("cache_dir"),
        "device": metadata.get("device"),
        "parameters": {
            key: value
            for key, value in metadata.items()
            if key not in {"metric", "backend", "package", "python", "model_name", "model_path", "cache_dir", "device"}
            and value is not None
        },
    }


def _internal_metric_implementation(metric_name: str) -> str:
    if metric_name == "wer":
        return "evaluation.metrics.text.word_error_rate"
    return "metric_algorithm"


def _report_text(summary: dict, run_config: dict) -> str:
    metric = summary.get("metric", {})
    target = summary.get("target", {})
    samples = summary.get("samples", {})
    lines = [
        f"# Evaluation Report: {target.get('model')} {target.get('stage')}",
        "",
        f"- Status: `{summary.get('status')}`",
        f"- Metric: `{metric.get('metric_id')}`",
        f"- Dataset: `{target.get('dataset')}`",
        f"- Split: `{target.get('split')}`",
        f"- Target manifest: `{target.get('manifest')}`",
        f"- Rows: {samples.get('total')}",
        f"- Included rows: {samples.get('included')}",
        f"- Skipped rows: {samples.get('skipped')}",
        f"- Failed rows: {samples.get('failed')}",
        "",
        "## Score",
        "",
    ]
    primary = summary.get("scores", {}).get("primary", {})
    for key, value in primary.items():
        lines.append(f"- {key}: `{value}`")
    if summary.get("errors", {}).get("count"):
        lines.extend(["", "## Errors", ""])
        for error_type, count in summary["errors"].get("by_type", {}).items():
            lines.append(f"- {count} x {error_type}")
    command = run_config.get("execution", {}).get("command", [])
    lines.extend(["", "## Command", "", "```text", " ".join(command), "```", ""])
    return "\n".join(lines)
