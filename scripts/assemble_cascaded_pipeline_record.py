#!/usr/bin/env python
"""Assemble a new-schema cascaded pipeline record from completed stage runs."""

from __future__ import annotations

import argparse
import json
import os
import shutil
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Literal

from results_structure import relative_to_project_root


Direction = Literal["a2s", "s2a"]

STAGES_BY_DIRECTION: dict[str, tuple[str, str, str]] = {
    "a2s": ("a2t", "t2t", "t2s"),
    "s2a": ("s2t", "t2t", "t2a"),
}
INPUT_FIELD_BY_DIRECTION = {"a2s": "audio_file", "s2a": "speech_file"}
FINAL_OUTPUT_FIELDS = {"audio_file", "speech_file"}
PIPELINE_ARTIFACTS = ("run.json", "pipeline.json", "manifest.jsonl", "inspect.json", "errors.json", "summary.json")
TERMINAL_STAGE_STATUSES = {"complete", "complete_with_errors"}
ROW_STATUSES = {"success", "error"}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--pipeline-dir", required=True, help="Target pipeline run directory.")
    parser.add_argument("--direction", required=True, choices=sorted(STAGES_BY_DIRECTION))
    parser.add_argument("--stage1-run-dir", required=True, help="A2T run for A2S, or S2T run for S2A.")
    parser.add_argument("--t2t-run-dir", required=True)
    parser.add_argument("--final-run-dir", required=True, help="T2S run for A2S, or T2A run for S2A.")
    parser.add_argument("--dataset", default=None)
    parser.add_argument("--split", default=None)
    parser.add_argument("--pipeline-name", default=None)
    parser.add_argument("--temporary", action="store_true")
    parser.add_argument("--logs-dir", default=None)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    report = assemble_pipeline_record(
        Path(args.pipeline_dir),
        direction=args.direction,
        stage_run_dirs=[Path(args.stage1_run_dir), Path(args.t2t_run_dir), Path(args.final_run_dir)],
        dataset=args.dataset,
        split=args.split,
        pipeline_name=args.pipeline_name,
        temporary=args.temporary,
        logs_dir=Path(args.logs_dir) if args.logs_dir else None,
    )
    summary = report["summary"]["samples"]
    print(
        "assembled cascaded pipeline record: "
        f"run_id={report['run_id']} direction={report['direction']} "
        f"success={summary['successful']} incomplete={summary['incomplete']}"
    )


def assemble_pipeline_record(
    pipeline_dir: Path,
    *,
    direction: Direction,
    stage_run_dirs: list[Path] | tuple[Path, Path, Path],
    dataset: str | None = None,
    split: str | None = None,
    pipeline_name: str | None = None,
    temporary: bool = False,
    logs_dir: Path | None = None,
    now: datetime | None = None,
) -> dict[str, Any]:
    """Write create-only pipeline artifacts by joining stage manifests on sample_id."""
    pipeline_dir = Path(pipeline_dir)
    _require_direction(direction)
    if len(stage_run_dirs) != 3:
        raise ValueError("stage_run_dirs must contain exactly three stage run directories")
    _fail_if_outputs_exist(pipeline_dir)

    stage_runs = [
        _load_stage_run(Path(stage_run_dir), expected_stage=expected_stage, position=position)
        for position, (stage_run_dir, expected_stage) in enumerate(
            zip(stage_run_dirs, STAGES_BY_DIRECTION[direction], strict=True),
            start=1,
        )
    ]
    _validate_stage_directions(stage_runs, direction)

    rows = _assemble_manifest_rows(direction, stage_runs)
    errors = _errors_document(pipeline_dir, rows)
    summary = _summary_document(pipeline_dir, rows, direction)
    status = summary["status"]
    run_id = pipeline_dir.name
    completed_at = (now or datetime.now(timezone.utc)).astimezone().isoformat()
    started_at = _pipeline_started_at(stage_runs) or completed_at

    pipeline_doc = _pipeline_document(run_id, direction, stage_runs)
    run_doc = _run_document(
        run_id=run_id,
        direction=direction,
        status=status,
        dataset=dataset or _first_string(stage_runs, "dataset"),
        split=split or _first_string(stage_runs, "split"),
        pipeline_name=pipeline_name,
        temporary=temporary,
        pipeline_dir=pipeline_dir,
        logs_dir=logs_dir,
        started_at=started_at,
        completed_at=completed_at,
        stage_runs=stage_runs,
    )
    inspect_doc = _inspect_document(
        run_id=run_id,
        direction=direction,
        status=status,
        dataset=run_doc["dataset"],
        split=run_doc["split"],
        pipeline_name=pipeline_name,
        stage_runs=stage_runs,
        summary=summary["samples"],
        rows=rows,
    )

    staging = pipeline_dir / ".assemble_pipeline_staging"
    if staging.exists():
        raise FileExistsError(f"staging directory already exists: {staging}")
    pipeline_dir.mkdir(parents=True, exist_ok=True)
    staging.mkdir()
    try:
        _write_json_create_only(staging / "run.json", run_doc)
        _write_json_create_only(staging / "pipeline.json", pipeline_doc)
        _write_jsonl_create_only(staging / "manifest.jsonl", rows)
        _write_json_create_only(staging / "inspect.json", inspect_doc)
        _write_json_create_only(staging / "errors.json", errors)
        _write_json_create_only(staging / "summary.json", summary)
        _publish_staging(pipeline_dir, staging)
    except Exception:
        shutil.rmtree(staging, ignore_errors=True)
        raise

    return {
        "schema_version": 1,
        "run_id": run_id,
        "direction": direction,
        "pipeline_dir": pipeline_dir.as_posix(),
        "summary": summary,
    }


def _load_stage_run(run_dir: Path, *, expected_stage: str, position: int) -> dict[str, Any]:
    manifest_path = run_dir / "manifest.jsonl"
    run_json = _read_json(run_dir / "run.json")
    rows = _read_manifest(manifest_path)
    rows_by_sample_id = _index_rows_by_sample_id(rows, manifest_path)
    stage = _stage_name(run_json, rows, run_dir)
    if stage != expected_stage:
        raise ValueError(f"{run_dir}: expected stage {expected_stage!r}; got {stage!r}")
    _validate_stage_run_status(run_json, run_dir)
    _validate_stage_row_statuses(rows, manifest_path)
    return {
        "position": position,
        "stage": stage,
        "model": _stage_model(run_json, rows),
        "run_id": _string_or_default(run_json.get("run_id"), run_dir.name),
        "run_path": _path_reference(run_dir),
        "manifest": _path_reference(manifest_path),
        "config": _path_reference(run_dir / "config.json"),
        "_run_dir": run_dir,
        "run": run_json,
        "summary": _read_optional_json(run_dir / "summary.json"),
        "inspect": _read_optional_json(run_dir / "inspect.json"),
        "rows": rows,
        "rows_by_sample_id": rows_by_sample_id,
        "dataset": run_json.get("dataset"),
        "split": run_json.get("split"),
    }


def _assemble_manifest_rows(direction: str, stage_runs: list[dict[str, Any]]) -> list[dict[str, Any]]:
    stage1, stage2, final_stage = stage_runs
    _validate_no_downstream_extras(stage1, stage2)
    _validate_no_downstream_extras(stage2, final_stage)

    ordered_stage1_rows = sorted(
        stage1["rows"],
        key=lambda row: row.get("row_index") if isinstance(row.get("row_index"), int) else 10**12,
    )
    rows = []
    for stage1_row in ordered_stage1_rows:
        sample_id = _sample_id(stage1_row, Path(stage1["manifest"]))
        stage2_row = stage2["rows_by_sample_id"].get(sample_id)
        final_row = final_stage["rows_by_sample_id"].get(sample_id)
        _validate_downstream_row(
            stage2_row,
            stage_run=stage2,
            sample_id=sample_id,
            expected_source_stage_index=1,
            upstream_status=_row_status(stage1_row),
        )
        _validate_downstream_row(
            final_row,
            stage_run=final_stage,
            sample_id=sample_id,
            expected_source_stage_index=2,
            upstream_status=_row_status(stage2_row) if stage2_row is not None else "missing",
        )
        joined = [
            _stage_entry(stage1, stage1_row),
            _stage_entry(stage2, stage2_row),
            _stage_entry(final_stage, final_row),
        ]
        failed_stage = _first_failed_stage(joined)
        success = failed_stage is None and all(entry["status"] == "success" for entry in joined)
        final_output = _final_output(final_stage, final_row) if success and final_row is not None else None
        rows.append(
            {
                "schema_version": 1,
                "sample_id": sample_id,
                "status": "success" if success else "incomplete",
                "direction": direction,
                "input": _pipeline_input(direction, stage1_row),
                "stages": [_public_stage_entry(stage) for stage in joined],
                "final_output": final_output,
                "error": None if success else _pipeline_error(failed_stage),
            }
        )
    return rows


def _validate_downstream_row(
    row: dict[str, Any] | None,
    *,
    stage_run: dict[str, Any],
    sample_id: str,
    expected_source_stage_index: int,
    upstream_status: str,
) -> None:
    if row is None:
        return
    input_value = row.get("input")
    if not isinstance(input_value, dict):
        raise ValueError(f"{stage_run['manifest']}: sample_id={sample_id!r} input must be an object")
    source_stage_index = input_value.get("source_stage_index")
    if source_stage_index != expected_source_stage_index:
        raise ValueError(
            f"{stage_run['manifest']}: sample_id={sample_id!r} source_stage_index must be "
            f"{expected_source_stage_index}; got {source_stage_index!r}"
        )
    source_sample_id = input_value.get("source_sample_id")
    if source_sample_id != sample_id:
        raise ValueError(
            f"{stage_run['manifest']}: sample_id={sample_id!r} source_sample_id must match; "
            f"got {source_sample_id!r}"
        )
    source_status = input_value.get("source_status")
    if source_status is not None and source_status != upstream_status:
        raise ValueError(
            f"{stage_run['manifest']}: sample_id={sample_id!r} source_status={source_status!r} "
            f"does not match upstream status {upstream_status!r}"
        )
    if row.get("status") == "success" and upstream_status != "success":
        raise ValueError(
            f"{stage_run['manifest']}: sample_id={sample_id!r} cannot be successful when "
            f"upstream status is {upstream_status!r}"
        )


def _row_status(row: dict[str, Any]) -> str:
    status = row.get("status")
    return status if isinstance(status, str) and status else "missing"


def _stage_entry(stage_run: dict[str, Any], row: dict[str, Any] | None) -> dict[str, Any]:
    status = row.get("status") if isinstance(row, dict) else "missing"
    return {
        "stage_index": stage_run["position"],
        "stage": stage_run["stage"],
        "model": stage_run["model"],
        "run_id": stage_run["run_id"],
        "status": status,
        "_error": row.get("error") if isinstance(row, dict) else None,
    }


def _public_stage_entry(stage: dict[str, Any]) -> dict[str, Any]:
    return {
        "stage_index": stage["stage_index"],
        "stage": stage["stage"],
        "model": stage["model"],
        "run_id": stage["run_id"],
        "status": stage["status"],
    }


def _first_failed_stage(stages: list[dict[str, Any]]) -> dict[str, Any] | None:
    for stage in stages:
        if stage["status"] != "success":
            return stage
    return None


def _pipeline_error(stage: dict[str, Any] | None) -> dict[str, Any]:
    if stage is None:
        return {
            "stage_index": None,
            "stage": None,
            "type": "PipelineIncomplete",
            "message": "Pipeline did not produce a final successful output.",
        }
    error = stage.get("_error") if isinstance(stage.get("_error"), dict) else {}
    if stage["status"] == "missing":
        error_type = "MissingStageRow"
        message = f"Missing manifest row for stage {stage['stage']}."
    else:
        error_type = _string_or_default(error.get("type"), "StageError")
        message = _string_or_default(error.get("message"), f"Stage {stage['stage']} status is {stage['status']}.")
    return {
        "stage_index": stage["stage_index"],
        "stage": stage["stage"],
        "type": error_type,
        "message": message,
    }


def _pipeline_input(direction: str, stage1_row: dict[str, Any]) -> dict[str, Any]:
    input_value = stage1_row.get("input")
    if not isinstance(input_value, dict):
        raise ValueError(f"sample_id={stage1_row.get('sample_id')!r}: stage-one input must be an object")
    field = INPUT_FIELD_BY_DIRECTION[direction]
    value = input_value.get(field)
    if not isinstance(value, str) or not value:
        raise ValueError(f"sample_id={stage1_row.get('sample_id')!r}: missing input.{field}")
    return {field: value}


def _final_output(final_stage: dict[str, Any], final_row: dict[str, Any]) -> dict[str, Any] | None:
    output = final_row.get("output")
    if output is None:
        raise ValueError(f"sample_id={final_row.get('sample_id')!r}: successful final stage row has no output")
    if not isinstance(output, dict):
        raise ValueError(f"sample_id={final_row.get('sample_id')!r}: final output must be an object")
    rewritten = dict(output)
    media_field = "speech_file" if final_stage["stage"] == "t2s" else "audio_file"
    if not isinstance(rewritten.get(media_field), str) or not rewritten.get(media_field):
        raise ValueError(f"sample_id={final_row.get('sample_id')!r}: final output missing {media_field}")
    for field in FINAL_OUTPUT_FIELDS:
        value = rewritten.get(field)
        if isinstance(value, str) and value:
            path = Path(value)
            if not path.is_absolute():
                path = Path(final_stage["_run_dir"]) / path
            if field == media_field and not path.is_file():
                raise FileNotFoundError(f"sample_id={final_row.get('sample_id')!r}: missing final output file: {path}")
            rewritten[field] = _path_reference(path)
    return rewritten


def _pipeline_document(run_id: str, direction: str, stage_runs: list[dict[str, Any]]) -> dict[str, Any]:
    final_stage = stage_runs[-1]
    return {
        "schema_version": 1,
        "run_id": run_id,
        "direction": direction,
        "stages": [
            {
                "position": stage["position"],
                "stage": stage["stage"],
                "model": stage["model"],
                "run_id": stage["run_id"],
                "run_path": stage["run_path"],
                "manifest": stage["manifest"],
            }
            for stage in stage_runs
        ],
        "final_output": {
            "stage": final_stage["stage"],
            "run_id": final_stage["run_id"],
            "manifest": final_stage["manifest"],
        },
    }


def _run_document(
    *,
    run_id: str,
    direction: str,
    status: str,
    dataset: str | None,
    split: str | None,
    pipeline_name: str | None,
    temporary: bool,
    pipeline_dir: Path,
    logs_dir: Path | None,
    started_at: str,
    completed_at: str,
    stage_runs: list[dict[str, Any]],
) -> dict[str, Any]:
    return {
        "schema_version": 1,
        "run_id": run_id,
        "temporary": temporary,
        "stage": "pipeline",
        "direction": direction,
        "status": status,
        "dataset": dataset,
        "split": split,
        "model": pipeline_name,
        "started_at": started_at,
        "completed_at": completed_at,
        "code": _pipeline_code_state(stage_runs),
        "slurm": _slurm_state(),
        "paths": {
            "result": _path_reference(pipeline_dir),
            "logs": _path_reference(logs_dir) if logs_dir else None,
        },
        "lineage": {
            "direction": direction,
            "current_stage_index": 3,
            "stages": [
                {
                    "stage_index": stage["position"],
                    "stage": stage["stage"],
                    "model": stage["model"],
                    "run_id": stage["run_id"],
                    "run_path": stage["run_path"],
                    "manifest": stage["manifest"],
                    "config": stage["config"],
                }
                for stage in stage_runs
            ],
        },
    }


def _pipeline_started_at(stage_runs: list[dict[str, Any]]) -> str | None:
    parsed: list[tuple[datetime, str]] = []
    for stage in stage_runs:
        value = stage["run"].get("started_at") if isinstance(stage.get("run"), dict) else None
        if not isinstance(value, str):
            continue
        try:
            timestamp = datetime.fromisoformat(value)
        except ValueError:
            continue
        if timestamp.tzinfo is None:
            timestamp = timestamp.replace(tzinfo=timezone.utc)
        parsed.append((timestamp.astimezone(timezone.utc), value))
    if not parsed:
        return None
    return min(parsed, key=lambda item: item[0])[1]


def _pipeline_code_state(stage_runs: list[dict[str, Any]]) -> dict[str, Any]:
    return {
        "stage_runs": [
            {
                "stage_index": stage["position"],
                "stage": stage["stage"],
                "run_id": stage["run_id"],
                "code": stage["run"].get("code") if isinstance(stage.get("run"), dict) else None,
            }
            for stage in stage_runs
        ]
    }


def _slurm_state() -> dict[str, str | None]:
    return {
        "job_id": os.environ.get("SLURM_JOB_ID"),
        "array_job_id": os.environ.get("SLURM_ARRAY_JOB_ID"),
        "array_task_id": os.environ.get("SLURM_ARRAY_TASK_ID"),
    }


def _inspect_document(
    *,
    run_id: str,
    direction: str,
    status: str,
    dataset: str | None,
    split: str | None,
    pipeline_name: str | None,
    stage_runs: list[dict[str, Any]],
    summary: dict[str, Any],
    rows: list[dict[str, Any]],
) -> dict[str, Any]:
    return {
        "schema_version": 1,
        "run_id": run_id,
        "stage": "pipeline",
        "direction": direction,
        "dataset": dataset,
        "split": split,
        "status": status,
        "lineage": {
            "direction": direction,
            "current_stage_index": 3,
            "stages": [
                {
                    "stage_index": stage["position"],
                    "stage": stage["stage"],
                    "model": stage["model"],
                    "run_id": stage["run_id"],
                    "run_path": stage["run_path"],
                    "manifest": stage["manifest"],
                    "config": stage["config"],
                }
                for stage in stage_runs
            ],
        },
        "stage_configs": _stage_configs(stage_runs),
        "summary": summary,
        "samples": [_inspect_sample(row, stage_runs) for row in rows],
        "pipeline_name": pipeline_name,
    }


def _inspect_sample(row: dict[str, Any], stage_runs: list[dict[str, Any]]) -> dict[str, Any]:
    sample_id = row["sample_id"]
    return {
        "sample_id": sample_id,
        "status": row["status"],
        "reference": _reference_from_inspect(stage_runs, sample_id),
        "input": row["input"],
        "stages": [_inspect_stage(stage_run, sample_id) for stage_run in stage_runs],
        "final_output": row["final_output"],
        "error": row["error"],
    }


def _reference_from_inspect(stage_runs: list[dict[str, Any]], sample_id: str) -> dict[str, Any]:
    merged: dict[str, Any] = {}
    for stage_run in stage_runs:
        samples = stage_run.get("inspect", {}).get("samples")
        if not isinstance(samples, list):
            continue
        for sample in samples:
            if not isinstance(sample, dict) or sample.get("sample_id") != sample_id:
                continue
            reference = sample.get("reference")
            if not isinstance(reference, dict):
                continue
            for key, value in reference.items():
                if key in merged and merged[key] != value:
                    raise ValueError(f"sample_id={sample_id!r}: conflicting reference.{key} across stage inspect records")
                merged[key] = value
    return merged


def _inspect_stage(stage_run: dict[str, Any], sample_id: str) -> dict[str, Any]:
    row = stage_run["rows_by_sample_id"].get(sample_id)
    if row is None:
        return {
            "stage_index": stage_run["position"],
            "stage": stage_run["stage"],
            "model": stage_run["model"],
            "run_id": stage_run["run_id"],
            "status": "missing",
            "input": None,
            "runtime_prompt": None,
            "output": None,
            "timing": None,
            "error": None,
        }
    return {
        "stage_index": stage_run["position"],
        "stage": stage_run["stage"],
        "model": stage_run["model"],
        "run_id": stage_run["run_id"],
        "status": row.get("status"),
        "input": row.get("input"),
        "runtime_prompt": _runtime_prompt_from_inspect(stage_run, sample_id),
        "output": row.get("output"),
        "timing": row.get("timing"),
        "error": row.get("error"),
    }


def _runtime_prompt_from_inspect(stage_run: dict[str, Any], sample_id: str) -> Any:
    samples = stage_run.get("inspect", {}).get("samples")
    if not isinstance(samples, list):
        return None
    for sample in samples:
        if not isinstance(sample, dict) or sample.get("sample_id") != sample_id:
            continue
        stages = sample.get("stages")
        if not isinstance(stages, list):
            return None
        for stage in stages:
            if isinstance(stage, dict) and stage.get("stage") == stage_run["stage"]:
                return stage.get("runtime_prompt")
    return None


def _stage_configs(stage_runs: list[dict[str, Any]]) -> list[dict[str, Any]]:
    configs = []
    for stage in stage_runs:
        config = _read_optional_json(Path(stage["_run_dir"]) / "config.json")
        if not config:
            continue
        compact = {
            "stage_index": stage["position"],
            "stage": stage["stage"],
        }
        for key in ("model", "input", "prompt", "generation"):
            if key in config:
                compact[key] = config[key]
        configs.append(compact)
    return configs


def _summary_document(pipeline_dir: Path, rows: list[dict[str, Any]], direction: str) -> dict[str, Any]:
    successful = sum(row["status"] == "success" for row in rows)
    incomplete = len(rows) - successful
    errors = _pipeline_errors(rows)
    errors_by_type = Counter(error["type"] for error in errors if isinstance(error.get("type"), str))
    status = "complete" if incomplete == 0 else "complete_with_errors"
    return {
        "schema_version": 1,
        "run_id": pipeline_dir.name,
        "status": status,
        "direction": direction,
        "samples": {
            "requested": len(rows),
            "processed": len(rows),
            "successful": successful,
            "incomplete": incomplete,
        },
        "outputs": {
            "pipeline": "pipeline.json",
            "manifest": "manifest.jsonl",
            "inspect": "inspect.json",
            "errors": "errors.json",
            "generated_media_files": None,
        },
        "errors": {
            "count": incomplete,
            "by_type": dict(errors_by_type),
        },
    }


def _errors_document(pipeline_dir: Path, rows: list[dict[str, Any]]) -> dict[str, Any]:
    errors = _pipeline_errors(rows)
    return {
        "schema_version": 1,
        "run_id": pipeline_dir.name,
        "error_count": len(errors),
        "errors": errors,
    }


def _pipeline_errors(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    errors = []
    for row in rows:
        if row.get("status") == "success":
            continue
        error = row.get("error") if isinstance(row.get("error"), dict) else {}
        errors.append(
            {
                "sample_id": row.get("sample_id"),
                "stage_index": error.get("stage_index"),
                "stage": error.get("stage"),
                "type": error.get("type"),
                "message": error.get("message"),
            }
        )
    return errors


def _read_manifest(path: Path) -> list[dict[str, Any]]:
    if not path.is_file():
        raise FileNotFoundError(f"missing manifest: {path}")
    rows = []
    with path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            row = json.loads(line)
            if not isinstance(row, dict):
                raise ValueError(f"{path}:{line_number} is not a JSON object")
            rows.append(row)
    return rows


def _read_json(path: Path) -> dict[str, Any]:
    if not path.is_file():
        raise FileNotFoundError(f"missing JSON file: {path}")
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"{path} is not a JSON object")
    return value


def _read_optional_json(path: Path) -> dict[str, Any]:
    if not path.is_file():
        return {}
    return _read_json(path)


def _index_rows_by_sample_id(rows: list[dict[str, Any]], manifest_path: Path) -> dict[str, dict[str, Any]]:
    indexed: dict[str, dict[str, Any]] = {}
    for row in rows:
        sample_id = _sample_id(row, manifest_path)
        if sample_id in indexed:
            raise ValueError(f"{manifest_path}: duplicate sample_id={sample_id!r}")
        indexed[sample_id] = row
    return indexed


def _sample_id(row: dict[str, Any], manifest_path: Path) -> str:
    value = row.get("sample_id")
    if not isinstance(value, str) or not value:
        raise ValueError(f"{manifest_path}: sample_id must be a non-empty string")
    return value


def _stage_name(run_json: dict[str, Any], rows: list[dict[str, Any]], run_dir: Path) -> str:
    stage = run_json.get("stage")
    if isinstance(stage, str) and stage:
        return stage
    for row in rows:
        row_stage = row.get("stage")
        if isinstance(row_stage, str) and row_stage:
            return row_stage
    raise ValueError(f"{run_dir}: cannot determine stage name")


def _stage_model(run_json: dict[str, Any], rows: list[dict[str, Any]]) -> str | None:
    model = run_json.get("model")
    if isinstance(model, str) and model:
        return model
    for row in rows:
        row_model = row.get("model")
        if isinstance(row_model, str) and row_model:
            return row_model
    return None


def _validate_stage_directions(stage_runs: list[dict[str, Any]], direction: str) -> None:
    for stage in stage_runs:
        stage_direction = stage["run"].get("direction")
        if isinstance(stage_direction, str) and stage_direction and stage_direction != direction:
            raise ValueError(
                f"{stage['run_path']}: direction {stage_direction!r} does not match pipeline direction {direction!r}"
            )


def _validate_stage_run_status(run_json: dict[str, Any], run_dir: Path) -> None:
    status = run_json.get("status")
    if status not in TERMINAL_STAGE_STATUSES:
        expected = ", ".join(sorted(TERMINAL_STAGE_STATUSES))
        raise ValueError(f"{run_dir}: stage run status must be terminal ({expected}); got {status!r}")


def _validate_stage_row_statuses(rows: list[dict[str, Any]], manifest_path: Path) -> None:
    for row in rows:
        status = row.get("status")
        if status not in ROW_STATUSES:
            raise ValueError(
                f"{manifest_path}: sample_id={row.get('sample_id')!r} has unsupported row status {status!r}"
            )


def _validate_no_downstream_extras(upstream: dict[str, Any], downstream: dict[str, Any]) -> None:
    upstream_ids = set(upstream["rows_by_sample_id"])
    extras = sorted(set(downstream["rows_by_sample_id"]) - upstream_ids)
    if extras:
        preview = ", ".join(extras[:10])
        raise ValueError(
            f"{downstream['manifest']}: sample_id values are not present in upstream stage "
            f"{upstream['stage']}: {preview}"
        )


def _first_string(stage_runs: list[dict[str, Any]], key: str) -> str | None:
    for stage in stage_runs:
        value = stage.get(key)
        if isinstance(value, str) and value:
            return value
    return None


def _require_direction(direction: str) -> None:
    if direction not in STAGES_BY_DIRECTION:
        expected = ", ".join(sorted(STAGES_BY_DIRECTION))
        raise ValueError(f"unsupported direction: {direction!r}; expected one of {expected}")


def _string_or_default(value: Any, default: str) -> str:
    return value if isinstance(value, str) and value else default


def _fail_if_outputs_exist(pipeline_dir: Path) -> None:
    for name in PIPELINE_ARTIFACTS:
        target = pipeline_dir / name
        if target.exists():
            raise FileExistsError(f"pipeline artifact already exists: {target}")


def _path_reference(path: Path) -> str:
    return relative_to_project_root(path)


def _publish_staging(pipeline_dir: Path, staging: Path) -> None:
    _fail_if_outputs_exist(pipeline_dir)
    for name in PIPELINE_ARTIFACTS:
        (staging / name).rename(pipeline_dir / name)
    staging.rmdir()


def _write_json_create_only(path: Path, value: Any) -> None:
    if path.exists():
        raise FileExistsError(f"artifact already exists: {path}")
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + "\n", encoding="utf-8")


def _write_jsonl_create_only(path: Path, rows: list[dict[str, Any]]) -> None:
    if path.exists():
        raise FileExistsError(f"artifact already exists: {path}")
    payload = "".join(json.dumps(row, ensure_ascii=False, separators=(",", ":"), allow_nan=False) + "\n" for row in rows)
    path.write_text(payload, encoding="utf-8")


if __name__ == "__main__":
    main()
