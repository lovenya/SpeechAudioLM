from __future__ import annotations

import csv
import hashlib
import io
import math
import json
import os
import statistics
import subprocess
import sys
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from evaluation.artifacts import (
    write_json_create_only,
    write_jsonl_create_only,
    write_text_create_only,
)
from evaluation.errors import EvaluationError, MetricUnavailableError
from evaluation.io import iter_jsonl, read_json, update_latest, utc_timestamp, write_json, write_jsonl
from evaluation.manifest import is_successful_generation, sample_id, source_record
from evaluation.metric_contracts import (
    ContractError,
    EvaluationMode,
    EvaluationProvenance,
    EvaluationScope,
    sha256_config,
    sha256_manifest,
    sha256_policy,
    sha256_sample_id_set,
    sha256_selected_rows,
)
from evaluation.metrics.external import backend_metadata, require_configured_backend
from evaluation.metrics.registry import MetricSpec
from evaluation.metrics.text import normalize_text, word_error_rate
from evaluation.result_validation import validate_final_result_completeness
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
    evaluation_mode: EvaluationMode
    common_sample_ids: tuple[str, ...] | None = None
    common_sample_ids_path: Path | None = None


def run_evaluation(options: EvalOptions) -> dict[str, Path]:
    dataset_config = read_json(options.dataset_config_path)
    metrics_config = read_json(options.metrics_config_path).get("metrics", {})
    dataset_root = Path(dataset_config["dataset_root"])
    rows, scope, provenance = _evaluation_scope_and_provenance(options)

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

    output_dir = _evaluation_output_dir(options)
    timestamp = utc_timestamp()
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
        scope.selected_sample_ids,
        per_sample_rows,
        corpus_metrics,
        diagnostic=options.evaluation_mode is EvaluationMode.BACKEND_SMOKE,
    ).to_dict()

    summary = _summarize(
        options,
        per_sample_rows,
        metric_backend_metadata,
        corpus_metrics,
        scope,
        provenance,
        validation_report,
    )

    paths = {
        "per_sample": output_dir / f"per_sample_{options.split}_{timestamp}.jsonl",
        "summary": output_dir / f"summary_{options.split}_{timestamp}.json",
        "summary_csv": output_dir / f"summary_{options.split}_{timestamp}.csv",
        "report": output_dir / f"report_{options.split}_{timestamp}.md",
        "run_config": output_dir / f"run_config_{options.split}_{timestamp}.json",
    }

    run_config = _run_config(
        options,
        metric_backend_metadata,
        scope,
        provenance,
        validation_report,
    )
    write_jsonl_create_only(paths["per_sample"], per_sample_rows)
    write_json_create_only(paths["summary"], summary)
    write_text_create_only(paths["summary_csv"], _summary_csv_text(summary))
    write_text_create_only(paths["report"], _report_text(summary, run_config))
    write_json_create_only(paths["run_config"], run_config)

    if scope.publishable:
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


def _evaluation_artifact_dir(output_dir: Path, split: str, timestamp: str) -> Path:
    return output_dir / f"work_{split}_{timestamp}"


def _evaluation_work_dir(output_dir: Path, split: str, timestamp: str) -> Path:
    slurm_tmpdir = os.environ.get("SLURM_TMPDIR")
    if slurm_tmpdir:
        root = Path(slurm_tmpdir) / "speechaudiolm_evaluation"
        root.mkdir(parents=True, exist_ok=True)
        return Path(tempfile.mkdtemp(prefix=f"work_{split}_{timestamp}_", dir=root))
    return output_dir / f"work_{split}_{timestamp}"


def dry_run(options: EvalOptions) -> dict[str, Any]:
    dataset_config = read_json(options.dataset_config_path)
    rows, scope, provenance = _evaluation_scope_and_provenance(options)
    dataset_root = Path(dataset_config["dataset_root"])
    _validate_rows(rows, options.metric_specs, dataset_root, require_success=False)
    output_dir = _evaluation_output_dir(options)
    return {
        "dry_run": True,
        "status": _evaluation_status(scope),
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
        "scope": scope.report_fields(),
        "provenance": provenance.report_fields(),
    }


def _evaluation_output_dir(options: EvalOptions) -> Path:
    root = eval_model_root(options.task, options.dataset, options.model, options.output_root)
    if options.evaluation_mode is EvaluationMode.BACKEND_SMOKE:
        return root / EvaluationMode.BACKEND_SMOKE.value
    return root


def _evaluation_scope_and_provenance(options: EvalOptions) -> tuple[list[dict], EvaluationScope, EvaluationProvenance]:
    try:
        all_rows = [item.record for item in iter_jsonl(options.manifest_path)]
        stop_index = None if options.max_samples is None else options.start_index + options.max_samples
        selected_rows = all_rows[options.start_index:stop_index]
        expected_ids = tuple(sample_id(row, ordinal) for ordinal, row in enumerate(all_rows))
        selected_ids = tuple(
            sample_id(row, ordinal)
            for ordinal, row in enumerate(selected_rows, start=options.start_index)
        )
        if options.evaluation_mode is EvaluationMode.FINAL and (
            options.common_sample_ids is None or options.common_sample_ids_path is None
        ):
            raise EvaluationError(
                "Final evaluation requires explicit --common-sample-ids identical to all manifest IDs."
            )
        common_ids = options.common_sample_ids or expected_ids
        common_ids_hash = (
            sha256_manifest(options.common_sample_ids_path)
            if options.common_sample_ids_path is not None
            else sha256_sample_id_set(common_ids)
        )
        git_worktree_hash, git_untracked_files = _git_worktree_provenance()
        scope = EvaluationScope(
            mode=options.evaluation_mode,
            expected_sample_ids=expected_ids,
            common_sample_ids=common_ids,
            selected_sample_ids=selected_ids,
            start_index=options.start_index,
            max_samples=options.max_samples,
        )
        provenance = EvaluationProvenance(
            manifest_sha256=sha256_manifest(options.manifest_path),
            config_sha256=sha256_config(_combined_configs(options)),
            policy_sha256=sha256_policy(_evaluation_policy(scope)),
            selected_rows_sha256=sha256_selected_rows(selected_rows),
            sample_id_set_sha256=sha256_sample_id_set(selected_ids),
            common_sample_ids_sha256=common_ids_hash,
            git_commit=_git_output(["rev-parse", "HEAD"]).decode("ascii").strip(),
            git_worktree_sha256=git_worktree_hash,
            git_untracked_files=git_untracked_files,
        )
    except ContractError as exc:
        raise EvaluationError(f"Invalid evaluation scope or provenance: {exc}") from None
    return selected_rows, scope, provenance


def _combined_configs(options: EvalOptions) -> dict[str, Any]:
    return {
        "dataset_config": _config_value(options.dataset_config_path),
        "metrics_config": _config_value(options.metrics_config_path),
        "task_metric_map": _config_value(options.task_metric_map_path),
    }


def _config_value(path: Path) -> dict[str, Any]:
    return {"path": str(path), "content": read_json(path) if path.is_file() else None}


def _evaluation_policy(scope: EvaluationScope) -> dict[str, Any]:
    return {
        "scope": scope.report_fields(),
        "common_sample_ids": list(scope.common_sample_ids),
        "reject_unsuccessful_generation_rows": scope.scientific,
        "latest_alias_policy": "final_only",
        "evaluation_sharding": "disallowed_for_final",
    }


def _git_output(arguments: list[str]) -> bytes:
    completed = subprocess.run(
        ["git", *arguments],
        cwd=Path(__file__).resolve().parents[1],
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
    )
    if completed.returncode != 0:
        message = completed.stderr.decode("utf-8", errors="replace").strip()
        raise EvaluationError(f"Unable to record git provenance: {message}")
    return completed.stdout


def _git_worktree_provenance() -> tuple[str, tuple[str, ...]]:
    repo_root = Path(__file__).resolve().parents[1]
    hasher = hashlib.sha256()
    hasher.update(b"tracked-diff\0")
    hasher.update(_git_output(["diff", "--binary", "HEAD", "--"]))

    untracked_text = _git_output(["ls-files", "--others", "--exclude-standard"]).decode(
        "utf-8", errors="strict"
    )
    relevant_roots = ("configs/", "evaluation/", "scripts/", "submission_scripts/", "tests/")
    relevant_suffixes = {".py", ".json", ".yaml", ".yml", ".toml", ".sh", ".sbatch"}
    relevant_paths: list[str] = []
    for relative in sorted(line for line in untracked_text.splitlines() if line):
        path = Path(relative)
        is_root_source = len(path.parts) == 1 and path.suffix in relevant_suffixes
        if not (relative.startswith(relevant_roots) or is_root_source):
            continue
        absolute = repo_root / path
        if not absolute.is_file():
            continue
        relevant_paths.append(relative)
        hasher.update(b"untracked-file\0")
        hasher.update(relative.encode("utf-8"))
        hasher.update(b"\0")
        hasher.update(hashlib.sha256(absolute.read_bytes()).digest())
    return hasher.hexdigest(), tuple(relevant_paths)


def _evaluation_status(scope: EvaluationScope) -> str:
    return "final_scientific" if scope.scientific else "diagnostic_non_authoritative"


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
        "stable-audio-metrics": ["clap_score_t2a", "kl_passt"],
        "dedicated-script": [
            "nli_contradiction",
            "g_eval_reference_free",
            "g_eval_prometheus_criterion_v2_plausibility",
            "g_eval_prometheus_criterion_v2_reference_grounded",
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
        "g_eval_reference_free",
        "g_eval_prometheus_criterion_v2_plausibility",
        "g_eval_prometheus_criterion_v2_reference_grounded",
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
                "g_eval_prometheus_criterion_v2_plausibility",
                "g_eval_prometheus_criterion_v2_reference_grounded",
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
            }
        },
    }


def _default_dedicated_script(metric_name: str) -> str:
    defaults = {
        "nli_contradiction": "scripts/run_nli_contradiction_helper.py",
        "g_eval_reference_free": "scripts/run_g_eval_reference_free.py",
        "g_eval_prometheus_criterion_v2_plausibility": "scripts/run_geval_prometheus_criterion_v2.py",
        "g_eval_prometheus_criterion_v2_reference_grounded": "scripts/run_geval_prometheus_criterion_v2.py",
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
                "g_eval_prometheus_criterion_v2_plausibility",
                "g_eval_prometheus_criterion_v2_reference_grounded",
            }:
                stage_input, stage_output, stage_direction = _final_text_stage(row)
                generated_text = stage_output or _prediction_text_for_metric(row, metric_name)
                input_text = stage_input
                direction = stage_direction
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
            if metric_name == "g_eval_prometheus_criterion_v2_reference_grounded":
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
        if config.get("allow_stub_fallback"):
            command.append("--allow-stub-fallback")
        return command
    if metric_name == "g_eval_reference_free":
        command = [
            str(python),
            str(script),
            "--input",
            str(input_path),
            "--output",
            str(output_path),
            "--summary-output",
            str(summary_path),
            "--summary-csv-output",
            str(metric_work_dir / "summary.csv"),
            "--backend",
            str(config.get("script_backend", config.get("runtime_backend", "vllm"))),
            "--model",
            str(config.get("model_name", config.get("model_path") or "/scratch/l/lovenya/salm_migration_clean/models/eval/prometheus-8x7b-v2.0")),
            "--batch-size",
            str(config.get("batch_size", 8)),
            "--cache-dir",
            str(config.get("cache_dir") or metric_work_dir / "cache"),
            "--tensor-parallel-size",
            str(config.get("tensor_parallel_size", 4)),
            "--temperature",
            str(config.get("temperature", 0.0)),
            "--max-tokens",
            str(config.get("max_tokens", 512)),
            "--gpu-memory-utilization",
            str(config.get("gpu_memory_utilization", 0.9)),
        ]
        if config.get("max_model_len") is not None:
            command.extend(["--max-model-len", str(config["max_model_len"])])
        if config.get("enforce_eager"):
            command.append("--enforce-eager")
        if config.get("trust_remote_code", True):
            command.append("--trust-remote-code")
        return command
    if metric_name in {
        "g_eval_prometheus_criterion_v2_plausibility",
        "g_eval_prometheus_criterion_v2_reference_grounded",
    }:
        variant = "reference_grounded" if metric_name.endswith("reference_grounded") else "plausibility"
        command = [
            str(python), str(script), "--input", str(input_path), "--output", str(output_path),
            "--summary-output", str(summary_path), "--summary-csv-output", str(metric_work_dir / "summary.csv"),
            "--variant", variant,
            "--backend", str(config.get("script_backend", config.get("runtime_backend", "vllm"))),
            "--model", str(config.get("model_name", config.get("model_path") or "/scratch/l/lovenya/salm_migration_clean/models/eval/prometheus-8x7b-v2.0")),
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
    if metric_name == "g_eval_reference_free" and isinstance(result.get("overall_score"), (int, float)):
        return float(result["overall_score"])
    if metric_name in {
        "g_eval_prometheus_criterion_v2_plausibility",
        "g_eval_prometheus_criterion_v2_reference_grounded",
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
        keys = ("generated_text", "generation_prompt", "converted_prompt", "understanding_text", "decoded_text")
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
    rows: list[dict],
    backend_metadata: dict[str, dict],
    corpus_metrics: dict[str, Any] | None,
    scope: EvaluationScope,
    provenance: EvaluationProvenance,
    validation_report: dict[str, Any],
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
            normalized_message = (
                json.dumps(message, ensure_ascii=False, sort_keys=True)
                if isinstance(message, (dict, list))
                else str(message)
            )
            grouped_errors[metric][normalized_message] = grouped_errors[metric].get(normalized_message, 0) + 1

    return {
        "status": validation_report["status"],
        "scope": scope.report_fields(),
        "provenance": provenance.report_fields(),
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
        "result_validation": validation_report,
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


def _run_config(
    options: EvalOptions,
    backend_metadata: dict[str, dict],
    scope: EvaluationScope,
    provenance: EvaluationProvenance,
    validation_report: dict[str, Any],
) -> dict:
    return {
        "status": validation_report["status"],
        "scope": scope.report_fields(),
        "provenance": provenance.report_fields(),
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
        "result_validation": validation_report,
        "start_index": options.start_index,
        "max_samples": options.max_samples,
        "cache_dir": str(options.cache_dir),
        "device": options.device,
        "command": options.command,
        "timestamp_utc": utc_timestamp(),
    }


def _summary_csv_text(summary: dict) -> str:
    output = io.StringIO(newline="")
    writer = csv.writer(output)
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
    return output.getvalue()


def _report_text(summary: dict, run_config: dict) -> str:
    lines = [
        f"# Evaluation Report: {summary['model']} {summary['task']}",
        "",
        f"- Scientific status: `{summary['status']}`",
        f"- Evaluation mode: `{summary['scope']['evaluation_mode']}`",
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
            fields = [f"mean={stats['mean']:.6g}"]
            for name in ("std", "min", "max", "median"):
                value = stats.get(name)
                if isinstance(value, (int, float)) and not isinstance(value, bool):
                    fields.append(f"{name}={value:.6g}")
            lines.append(f"- `{metric}`: " + ", ".join(fields))
        else:
            lines.append(f"- `{metric}`: {stats.get('status', 'not computed')}")
    lines.extend(["", "## Dependency Notes", ""])
    lines.append("- aac-metrics / SPIDEr / MACE may require Java and cached assets.")
    lines.append("- KAD/kadtk may have Python version constraints and should be run corpus-level.")
    lines.append("- ASR/CLAP/UTMOS/DNSMOS/FAD/KL_PaSST/IS require external model weights or toolkits.")
    lines.append("- Generated-speech dWER is directional: WER(ASR(reference_speech), ASR(generated_speech)).")
    lines.append("- FAD, KAD, KL_PaSST, and Inception Score are corpus-level metrics only.")
    lines.extend(["", "## Command", "", "```text", " ".join(run_config.get("command", [])), "```", ""])
    return "\n".join(lines)
