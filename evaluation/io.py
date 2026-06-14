from __future__ import annotations

import json
import shutil
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Iterator

from project_paths import resolve_config_value


@dataclass(frozen=True)
class ManifestRow:
    ordinal: int
    record: dict


def read_json(path: Path) -> dict:
    with path.open("r", encoding="utf-8") as f:
        return resolve_config_value(json.load(f))


def iter_jsonl(path: Path, max_samples: int | None = None, start_index: int = 0) -> Iterator[ManifestRow]:
    if start_index < 0:
        raise ValueError("--start-index must be >= 0")
    if max_samples is not None and max_samples <= 0:
        raise ValueError("--max-samples must be a positive integer")

    emitted = 0
    seen = 0
    with path.open("r", encoding="utf-8") as f:
        for ordinal, line in enumerate(f):
            line = line.strip()
            if not line:
                continue
            if seen < start_index:
                seen += 1
                continue
            yield ManifestRow(ordinal=ordinal, record=_normalize_generation_record(json.loads(line)))
            seen += 1
            emitted += 1
            if max_samples is not None and emitted >= max_samples:
                break


def _legacy_task_name(task: object) -> object:
    return {
        "audio_to_text": "A2T",
        "speech_to_text": "S2T",
        "text_to_text": "T2T",
        "text_to_speech": "T2S",
        "text_to_audio": "T2A",
    }.get(task, task)

def _normalize_generation_record(record: dict) -> dict:
    sample = record.get("sample")
    if not isinstance(sample, dict):
        return record
    stages = sample.get("stages") if isinstance(sample.get("stages"), list) else []
    last_stage = stages[-1] if stages else {}
    output = last_stage.get("output") if isinstance(last_stage.get("output"), dict) else {}
    reference = dict(sample.get("reference") or {})
    if "audio_caption" in reference and "audio_cap" not in reference:
        reference["audio_cap"] = reference["audio_caption"]
    normalized = {
        key: value
        for key, value in record.items()
        if key != "sample"
    }
    normalized.update(
        {
            "status": sample.get("status"),
            "task": _legacy_task_name(last_stage.get("task")),
            "model": last_stage.get("model"),
            "row_index": sample.get("row_index"),
            "sample_id": sample.get("sample_id"),
            "source": reference,
            "candidate_text": output.get("text"),
            "output_file": output.get("file"),
            "sample_rate": output.get("sample_rate"),
            "generation_seconds": last_stage.get("inference_seconds"),
            "prompt": (last_stage.get("input") or {}).get("text"),
            "model_extra": {"stages": stages},
            "error": sample.get("error"),
        }
    )
    return normalized

def write_json(path: Path, data: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, ensure_ascii=False, sort_keys=True)
        f.write("\n")


def write_jsonl(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")


def utc_timestamp() -> str:
    return datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")


def update_latest(path: Path, latest_path: Path) -> None:
    latest_path.parent.mkdir(parents=True, exist_ok=True)
    shutil.copyfile(path, latest_path)
