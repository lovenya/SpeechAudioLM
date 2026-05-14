from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Iterator


@dataclass(frozen=True)
class PromptRow:
    row_index: int
    sample_id: str
    prompt: str
    record: dict


def load_dataset_config(path: Path) -> dict:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def iter_jsonl(path: Path) -> Iterator[dict]:
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                yield json.loads(line)


def iter_split_rows(
    split_path: Path,
    prompt_field: str,
    start_index: int = 0,
    num_samples: int = -1,
) -> Iterator[PromptRow]:
    if start_index < 0:
        raise ValueError("--start-index must be >= 0")
    if num_samples == 0 or num_samples < -1:
        raise ValueError("--num-samples must be -1 or a positive integer")

    emitted = 0
    for row_index, record in enumerate(iter_jsonl(split_path)):
        if row_index < start_index:
            continue
        prompt = record.get(prompt_field)
        if not isinstance(prompt, str) or not prompt.strip():
            raise ValueError(f"Row {row_index} has no usable prompt field {prompt_field!r}")
        sample_id = sample_id_from_record(record, row_index)
        yield PromptRow(
            row_index=row_index,
            sample_id=sample_id,
            prompt=prompt.strip(),
            record=record,
        )
        emitted += 1
        if num_samples != -1 and emitted >= num_samples:
            break


def sample_id_from_record(record: dict, row_index: int) -> str:
    for key in ("audio_file", "speech_file"):
        value = record.get(key)
        if isinstance(value, str) and value:
            return Path(value).stem
    original = record.get("_original_line_index")
    if original is not None:
        return f"line_{original}"
    return f"row_{row_index}"
