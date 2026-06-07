from __future__ import annotations

import json
from datetime import datetime
from pathlib import Path
from typing import Any, Iterable
from zoneinfo import ZoneInfo

SCHEMA_VERSION = "1.0"
EASTERN_TIME = ZoneInfo("America/Toronto")


def eastern_now() -> str:
    return datetime.now(EASTERN_TIME).isoformat(timespec="seconds")


def unwrap_sample(record: dict[str, Any]) -> dict[str, Any]:
    sample = record.get("sample")
    return sample if isinstance(sample, dict) else record


def wrap_sample(metadata: dict[str, Any], sample: dict[str, Any]) -> dict[str, Any]:
    return {
        "schema_version": SCHEMA_VERSION,
        "current_stage": len(sample.get("stages", [])),
        "lane": metadata["lane"],
        "pipeline": metadata["pipeline"],
        "dataset": metadata["dataset"],
        "split": metadata["split"],
        "config_file": metadata.get("config_file"),
        "sample": sample,
    }


def reference_from_record(record: dict[str, Any]) -> dict[str, Any]:
    carried = record.get("_cascaded_sample")
    if isinstance(carried, dict) and isinstance(carried.get("reference"), dict):
        return dict(carried["reference"])
    original = _original_record(record)
    reference = {
        "audio_file": original.get("audio_file"),
        "audio_caption": original.get("audio_cap") or original.get("audio_caption"),
        "speech_file": original.get("speech_file"),
        "speech_text": original.get("speech_text") or original.get("speech_transcript"),
    }
    metadata = _audio_metadata(original)
    if metadata:
        reference["audio_metadata"] = metadata
    return {key: value for key, value in reference.items() if value is not None}


def carried_stages(record: dict[str, Any]) -> list[dict[str, Any]]:
    sample = record.get("_cascaded_sample")
    stages = sample.get("stages") if isinstance(sample, dict) else None
    return [dict(stage) for stage in stages] if isinstance(stages, list) else []


def carried_stage_definitions(record: dict[str, Any]) -> list[dict[str, Any]]:
    definitions = record.get("_cascaded_stage_definitions")
    return [dict(item) for item in definitions] if isinstance(definitions, list) else []


def build_sample(
    *,
    row_index: int,
    sample_id: str,
    record: dict[str, Any],
    stage: dict[str, Any],
    status: str,
    error: dict[str, str] | None = None,
) -> dict[str, Any]:
    stages = carried_stages(record)
    expected_stage = len(stages) + 1
    if stage.get("stage") != expected_stage:
        raise ValueError(f"Stage {stage.get('stage')} does not follow {len(stages)} carried stages")
    stages.append(stage)
    final_output = stage.get("output") if status == "success" and stage["task"] in {"text_to_audio", "text_to_speech"} else None
    return {
        "row_index": row_index,
        "sample_id": sample_id,
        "status": status,
        "reference": reference_from_record(record),
        "stages": stages,
        "final_output": final_output,
        "error": error,
    }


def write_inspection_manifest(
    path: Path,
    *,
    metadata: dict[str, Any],
    stage_definitions: list[dict[str, Any]],
    samples: Iterable[dict[str, Any]],
    created_at: str,
    started_at: str,
    completed_at: str,
    status: str,
) -> None:
    sample_list = list(samples)
    success = sum(sample.get("status") == "success" for sample in sample_list)
    document = {
        "schema_version": SCHEMA_VERSION,
        "current_stage": max((len(sample.get("stages", [])) for sample in sample_list), default=len(stage_definitions)),
        "lane": metadata["lane"],
        "pipeline": metadata["pipeline"],
        "dataset": metadata["dataset"],
        "split": metadata["split"],
        "config_file": metadata.get("config_file"),
        "created_at": created_at,
        "started_at": started_at,
        "completed_at": completed_at,
        "status": status,
        "stage_definitions": stage_definitions,
        "summary": {"total": len(sample_list), "success": success, "failed": len(sample_list) - success},
        "samples": sample_list,
    }
    path.write_text(json.dumps(document, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def read_inspection_definitions(manifest_path: Path) -> list[dict[str, Any]]:
    inspection_path = manifest_path.with_name("manifest.inspect.json")
    if not inspection_path.is_file():
        return []
    document = json.loads(inspection_path.read_text(encoding="utf-8"))
    definitions = document.get("stage_definitions")
    return [dict(item) for item in definitions] if isinstance(definitions, list) else []


def output_duration_seconds(extra: dict[str, Any], sample_rate: int | None) -> float | None:
    if not sample_rate:
        return None
    shape = extra.get("shape")
    if isinstance(shape, list) and shape and isinstance(shape[-1], (int, float)):
        return round(float(shape[-1]) / sample_rate, 3)
    return None


def _original_record(record: dict[str, Any]) -> dict[str, Any]:
    for key in ("original_row", "source"):
        nested = record.get(key)
        if isinstance(nested, dict):
            return _original_record(nested)
    return record


def _audio_metadata(record: dict[str, Any]) -> dict[str, Any]:
    metadata: dict[str, Any] = {}
    if isinstance(record.get("retained_audio_tags"), list):
        metadata["audio_tags"] = record["retained_audio_tags"]
    if isinstance(record.get("retained_audio_tag_scores"), dict):
        metadata["audio_tag_scores"] = record["retained_audio_tag_scores"]
    for key in ("primary_audio_tag", "primary_audio_class"):
        if record.get(key) is not None:
            metadata[key] = record[key]
    pools: dict[str, Any] = {}
    for key in ("audio_to_speech", "speech_to_audio"):
        value = record.get(key)
        if isinstance(value, dict) and value.get("reference_pool_class"):
            pools[key] = value["reference_pool_class"]
    if pools:
        metadata["reference_pool"] = pools
    return metadata
