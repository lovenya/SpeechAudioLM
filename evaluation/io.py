from __future__ import annotations

import json
import shutil
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Iterator


@dataclass(frozen=True)
class ManifestRow:
    ordinal: int
    record: dict


def read_json(path: Path) -> dict:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


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
            yield ManifestRow(ordinal=ordinal, record=json.loads(line))
            seen += 1
            emitted += 1
            if max_samples is not None and emitted >= max_samples:
                break


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
