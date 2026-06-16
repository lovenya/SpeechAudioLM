#!/usr/bin/env python
"""Merge new-schema cascaded stage shards into a canonical stage run root."""

from __future__ import annotations

import argparse
import json
import shutil
from collections import Counter
from pathlib import Path
from typing import Any, Literal


MediaOutput = Literal["none", "audio", "speech"]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", required=True, help="Stage run root containing shards/shard_*/.")
    parser.add_argument("--expected-count", type=int, required=True)
    parser.add_argument("--media-output", choices=["none", "audio", "speech"], default="none")
    parser.add_argument("--keep-shards", action="store_true", help="Keep shard directories after a clean merge.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    report = merge_stage_shards(
        Path(args.run_dir),
        expected_count=args.expected_count,
        media_output=args.media_output,
        delete_shards=not args.keep_shards,
    )
    print(
        "merged stage shards: "
        f"run_id={report['run_id']} rows={report['merged']['rows_written']} "
        f"success={report['merged']['successful']} failed={report['merged']['failed']}"
    )


def merge_stage_shards(
    run_dir: Path,
    *,
    expected_count: int,
    media_output: MediaOutput = "none",
    delete_shards: bool = True,
) -> dict[str, Any]:
    run_dir = Path(run_dir)
    if expected_count < 0:
        raise ValueError("expected_count must be non-negative")
    shards_dir = run_dir / "shards"
    shard_dirs = sorted(path for path in shards_dir.glob("shard_*") if path.is_dir())
    if not shard_dirs:
        raise FileNotFoundError(f"no shard directories under {shards_dir}")
    _require_root_artifacts(run_dir)
    _fail_if_published_outputs_exist(run_dir, media_output)

    rows: dict[int, dict[str, Any]] = {}
    shard_reports: list[dict[str, Any]] = []
    referenced_media: dict[Path, str] = {}
    inspect_by_sample_id: dict[str, dict[str, Any]] = {}
    first_inspect: dict[str, Any] = {}

    for shard_dir in shard_dirs:
        shard_rows = _read_manifest(shard_dir / "manifest.jsonl")
        shard_inspect = _read_optional_json(shard_dir / "inspect.json")
        if shard_inspect and not first_inspect:
            first_inspect = shard_inspect
        for sample in shard_inspect.get("samples", []) if shard_inspect else []:
            if isinstance(sample, dict) and isinstance(sample.get("sample_id"), str):
                inspect_by_sample_id[sample["sample_id"]] = sample
        successful = 0
        failed = 0
        for row in shard_rows:
            row_index = _row_index(row, shard_dir)
            sample_id = _sample_id(row, shard_dir)
            _validate_sample_link(row, sample_id, shard_dir)
            if row_index in rows:
                raise ValueError(f"duplicate row_index={row_index}")
            if row.get("status") == "success":
                successful += 1
                _stage_media_reference(
                    row,
                    shard_dir=shard_dir,
                    media_output=media_output,
                    referenced_media=referenced_media,
                )
            elif row.get("status") == "error":
                failed += 1
            else:
                raise ValueError(f"{shard_dir}: row {row_index} has unsupported status {row.get('status')!r}")
            rows[row_index] = row
        _fail_on_stray_media(shard_dir, referenced_media)
        shard_reports.append(
            {
                "shard_id": shard_dir.name,
                "manifest": f"shards/{shard_dir.name}/manifest.jsonl",
                "requested": len(shard_rows),
                "successful": successful,
                "failed": failed,
            }
        )

    _validate_expected_rows(rows, expected_count)
    _validate_unique_sample_ids(rows)

    staging = run_dir / ".merge_stage_staging"
    if staging.exists():
        raise FileExistsError(f"staging directory already exists: {staging}")
    staging.mkdir()
    try:
        media_copied, missing_media = _copy_media_to_staging(
            referenced_media,
            staging_outputs=staging / "outputs",
            enabled=media_output != "none",
        )
        ordered_rows = [rows[index] for index in sorted(rows)]
        report = _merge_report(
            run_dir=run_dir,
            shard_reports=shard_reports,
            rows=ordered_rows,
            expected_count=expected_count,
            media_copied=media_copied,
            missing_media=missing_media,
        )
        _write_jsonl_create_only(staging / "manifest.jsonl", ordered_rows)
        _write_json_create_only(staging / "errors.json", _errors_document(run_dir, ordered_rows))
        _write_json_create_only(staging / "summary.json", _summary_document(run_dir, ordered_rows, media_copied, media_output))
        _write_json_create_only(staging / "inspect.json", _inspect_document(run_dir, first_inspect, inspect_by_sample_id, ordered_rows))
        _write_json_create_only(staging / "merge_report.json", report)
        _publish_staging(run_dir, staging, media_output)
    except Exception:
        shutil.rmtree(staging, ignore_errors=True)
        raise

    if delete_shards:
        shutil.rmtree(shards_dir)
    return report


def _read_manifest(path: Path) -> list[dict[str, Any]]:
    if not path.is_file():
        raise FileNotFoundError(f"missing shard manifest: {path}")
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
        raise FileNotFoundError(f"missing shard inspect file: {path}")
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"{path} is not a JSON object")
    return value


def _read_optional_json(path: Path) -> dict[str, Any]:
    if not path.is_file():
        return {}
    return _read_json(path)


def _require_root_artifacts(run_dir: Path) -> None:
    missing = [name for name in ("run.json", "config.json") if not (run_dir / name).is_file()]
    if missing:
        raise FileNotFoundError(f"{run_dir}: missing required stage root artifact(s): {', '.join(missing)}")


def _row_index(row: dict[str, Any], shard_dir: Path) -> int:
    value = row.get("row_index")
    if not isinstance(value, int) or value < 0:
        raise ValueError(f"{shard_dir}: row_index must be a non-negative integer")
    return value


def _sample_id(row: dict[str, Any], shard_dir: Path) -> str:
    value = row.get("sample_id")
    if not isinstance(value, str) or not value:
        raise ValueError(f"{shard_dir}: sample_id must be a non-empty string")
    return value


def _validate_sample_link(row: dict[str, Any], sample_id: str, shard_dir: Path) -> None:
    input_value = row.get("input")
    if not isinstance(input_value, dict):
        raise ValueError(f"{shard_dir}: row {row.get('row_index')} input must be an object")
    source_sample_id = input_value.get("source_sample_id")
    if source_sample_id is not None and source_sample_id != sample_id:
        raise ValueError(
            f"{shard_dir}: row {row.get('row_index')} sample_id={sample_id!r} "
            f"does not match source_sample_id={source_sample_id!r}"
        )


def _stage_media_reference(
    row: dict[str, Any],
    *,
    shard_dir: Path,
    media_output: MediaOutput,
    referenced_media: dict[Path, str],
) -> None:
    output = row.get("output")
    if media_output == "none":
        if output is not None and isinstance(output, dict) and output.get("type") in {"audio", "speech"}:
            raise ValueError(f"{shard_dir}: media row requires --media-output")
        return
    if not isinstance(output, dict):
        raise ValueError(f"{shard_dir}: successful media row has no output object")
    expected_type = media_output
    if output.get("type") != expected_type:
        raise ValueError(f"{shard_dir}: output.type must be {expected_type!r}")
    key = "audio_file" if media_output == "audio" else "speech_file"
    value = output.get(key)
    if not isinstance(value, str) or not value:
        raise ValueError(f"{shard_dir}: output.{key} must be a non-empty path")
    source = Path(value)
    if not source.is_absolute():
        source = shard_dir / source
    source = source.resolve()
    target_name = source.name
    if target_name in referenced_media.values():
        raise ValueError(f"duplicate media file name after merge: {target_name}")
    referenced_media[source] = target_name
    output[key] = f"outputs/{target_name}"


def _fail_on_stray_media(shard_dir: Path, referenced_media: dict[Path, str]) -> None:
    outputs_dir = shard_dir / "outputs"
    if not outputs_dir.exists():
        return
    referenced = set(referenced_media)
    stray = sorted(path for path in outputs_dir.rglob("*") if path.is_file() and path.resolve() not in referenced)
    if stray:
        formatted = ", ".join(str(path) for path in stray[:5])
        raise ValueError(f"{shard_dir}: output files without matching manifest rows: {formatted}")


def _validate_expected_rows(rows: dict[int, dict[str, Any]], expected_count: int) -> None:
    if len(rows) != expected_count:
        raise ValueError(f"merged rows={len(rows)}; expected={expected_count}")


def _validate_unique_sample_ids(rows: dict[int, dict[str, Any]]) -> None:
    counts = Counter(str(row["sample_id"]) for row in rows.values())
    duplicates = sorted(sample_id for sample_id, count in counts.items() if count > 1)
    if duplicates:
        raise ValueError(f"duplicate sample_id values: {', '.join(duplicates[:10])}")


def _copy_media_to_staging(
    referenced_media: dict[Path, str],
    *,
    staging_outputs: Path,
    enabled: bool,
) -> tuple[int | None, list[str]]:
    if not enabled:
        return None, []
    missing = [str(path) for path in referenced_media if not path.is_file()]
    if missing:
        raise FileNotFoundError(f"missing referenced media files: {', '.join(missing[:5])}")
    staging_outputs.mkdir()
    for source, target_name in sorted(referenced_media.items(), key=lambda item: item[1]):
        shutil.copy2(source, staging_outputs / target_name)
    return len(referenced_media), []


def _merge_report(
    *,
    run_dir: Path,
    shard_reports: list[dict[str, Any]],
    rows: list[dict[str, Any]],
    expected_count: int,
    media_copied: int | None,
    missing_media: list[str],
) -> dict[str, Any]:
    failed = sum(row.get("status") == "error" for row in rows)
    successful = sum(row.get("status") == "success" for row in rows)
    return {
        "schema_version": 1,
        "run_id": run_dir.name,
        "shards": shard_reports,
        "merged": {
            "requested": expected_count,
            "rows_written": len(rows),
            "successful": successful,
            "failed": failed,
            "missing": expected_count - len(rows),
            "duplicate_sample_ids": [],
            "duplicate_row_indices": [],
        },
        "outputs": {
            "media_files_copied": media_copied,
            "missing_media_files": missing_media,
        },
        "status": "complete" if failed == 0 else "complete_with_errors",
    }


def _errors_document(run_dir: Path, rows: list[dict[str, Any]]) -> dict[str, Any]:
    errors = []
    for row in rows:
        if row.get("status") != "error":
            continue
        error = row.get("error") if isinstance(row.get("error"), dict) else {}
        errors.append(
            {
                "sample_id": row.get("sample_id"),
                "row_index": row.get("row_index"),
                "stage": row.get("stage"),
                "error_type": error.get("type"),
                "message": error.get("message"),
                "log_ref": error.get("log_ref"),
            }
        )
    return {
        "schema_version": 1,
        "run_id": run_dir.name,
        "error_count": len(errors),
        "errors": errors,
    }


def _summary_document(
    run_dir: Path,
    rows: list[dict[str, Any]],
    media_copied: int | None,
    media_output: MediaOutput,
) -> dict[str, Any]:
    timings = [
        row.get("timing", {}).get("inference_seconds")
        for row in rows
        if isinstance(row.get("timing"), dict) and isinstance(row["timing"].get("inference_seconds"), (int, float))
    ]
    failed = sum(row.get("status") == "error" for row in rows)
    successful = sum(row.get("status") == "success" for row in rows)
    return {
        "schema_version": 1,
        "run_id": run_dir.name,
        "status": "complete" if failed == 0 else "complete_with_errors",
        "samples": {
            "requested": len(rows),
            "processed": len(rows),
            "successful": successful,
            "failed": failed,
            "skipped": 0,
        },
        "timing": {
            "model_load_seconds": None,
            "total_inference_seconds": round(sum(timings), 6) if timings else None,
            "total_seconds": None,
            "mean_inference_seconds_per_sample": round(sum(timings) / len(timings), 6) if timings else None,
            "median_inference_seconds_per_sample": _median(timings),
            "min_inference_seconds_per_sample": min(timings) if timings else None,
            "max_inference_seconds_per_sample": max(timings) if timings else None,
        },
        "outputs": {
            "manifest": "manifest.jsonl",
            "inspect": "inspect.json",
            "generated_media_files": media_copied if media_output != "none" else None,
        },
        "errors": {
            "count": failed,
            "by_type": dict(Counter(_error_type(row) for row in rows if row.get("status") == "error")),
        },
    }


def _inspect_document(
    run_dir: Path,
    first: dict[str, Any],
    inspect_by_sample_id: dict[str, dict[str, Any]],
    rows: list[dict[str, Any]],
) -> dict[str, Any]:
    samples = []
    for row in rows:
        sample_id = row.get("sample_id")
        if isinstance(sample_id, str) and sample_id in inspect_by_sample_id:
            samples.append(inspect_by_sample_id[sample_id])
        else:
            samples.append(
                {
                    "sample_id": sample_id,
                    "status": row.get("status"),
                    "stages": [
                        {
                            "stage": row.get("stage"),
                            "model": row.get("model"),
                            "status": row.get("status"),
                            "input": row.get("input"),
                            "runtime_prompt": None,
                            "output": row.get("output"),
                            "timing": row.get("timing"),
                            "error": row.get("error"),
                        }
                    ],
                    "final_output": row.get("output") if row.get("status") == "success" else None,
                }
            )
    return {
        "schema_version": 1,
        "run_id": run_dir.name,
        "stage": first.get("stage"),
        "direction": first.get("direction"),
        "dataset": first.get("dataset"),
        "split": first.get("split"),
        "status": "complete" if all(row.get("status") == "success" for row in rows) else "complete_with_errors",
        "lineage": first.get("lineage"),
        "stage_configs": first.get("stage_configs", []),
        "summary": {
            "requested": len(rows),
            "successful": sum(row.get("status") == "success" for row in rows),
            "failed": sum(row.get("status") == "error" for row in rows),
        },
        "samples": samples,
    }


def _publish_staging(run_dir: Path, staging: Path, media_output: MediaOutput) -> None:
    for name in ("manifest.jsonl", "errors.json", "summary.json", "inspect.json", "merge_report.json"):
        target = run_dir / name
        if target.exists():
            raise FileExistsError(f"canonical artifact already exists: {target}")
    if media_output != "none" and (run_dir / "outputs").exists():
        raise FileExistsError(f"canonical outputs directory already exists: {run_dir / 'outputs'}")
    for name in ("manifest.jsonl", "errors.json", "summary.json", "inspect.json", "merge_report.json"):
        (staging / name).rename(run_dir / name)
    if media_output != "none":
        (staging / "outputs").rename(run_dir / "outputs")
    staging.rmdir()


def _fail_if_published_outputs_exist(run_dir: Path, media_output: MediaOutput) -> None:
    for name in ("manifest.jsonl", "errors.json", "summary.json", "inspect.json", "merge_report.json"):
        if (run_dir / name).exists():
            raise FileExistsError(f"canonical artifact already exists: {run_dir / name}")
    if media_output != "none" and (run_dir / "outputs").exists():
        raise FileExistsError(f"canonical outputs directory already exists: {run_dir / 'outputs'}")


def _write_json_create_only(path: Path, value: Any) -> None:
    if path.exists():
        raise FileExistsError(f"artifact already exists: {path}")
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + "\n", encoding="utf-8")


def _write_jsonl_create_only(path: Path, rows: list[dict[str, Any]]) -> None:
    if path.exists():
        raise FileExistsError(f"artifact already exists: {path}")
    payload = "".join(json.dumps(row, ensure_ascii=False, separators=(",", ":"), allow_nan=False) + "\n" for row in rows)
    path.write_text(payload, encoding="utf-8")


def _median(values: list[float]) -> float | None:
    if not values:
        return None
    ordered = sorted(values)
    midpoint = len(ordered) // 2
    if len(ordered) % 2:
        return ordered[midpoint]
    return (ordered[midpoint - 1] + ordered[midpoint]) / 2


def _error_type(row: dict[str, Any]) -> str:
    error = row.get("error")
    if isinstance(error, dict) and isinstance(error.get("type"), str):
        return error["type"]
    return "UnknownError"


if __name__ == "__main__":
    main()
