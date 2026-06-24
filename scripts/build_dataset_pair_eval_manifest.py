#!/usr/bin/env python
"""Build evaluation manifests from paired TTAS dataset examples.

This is intended for metric sanity checks: the paired dataset target is treated
as the generated output so existing metrics can score known-valid pairs.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path
from typing import Any, Iterable

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from evaluation.io import read_json
from project_paths import root_path


DIRECTIONS = {"a2s", "s2a"}
STAGES = {"t2t", "t2s", "t2a"}


def main() -> None:
    args = _build_parser().parse_args()
    dataset_config = read_json(args.dataset_config)
    dataset_root = Path(dataset_config["dataset_root"])
    split_path = _split_path(dataset_config, args.split)
    rows = list(_iter_jsonl(split_path))
    selected = _select_rows(rows, args.start_index, args.max_samples)
    output = args.output or _default_output_path(
        dataset=dataset_config.get("dataset", "ttas"),
        split=args.split,
        stage=args.stage,
        direction=args.direction,
    )
    if output.exists() and not args.overwrite:
        raise SystemExit(f"error: output already exists: {output} (pass --overwrite to replace it)")
    manifest_rows = [
        _manifest_row(
            source=row,
            ordinal=args.start_index + offset,
            dataset_root=dataset_root,
            stage=args.stage,
            direction=args.direction,
            skip_missing=args.skip_missing,
        )
        for offset, row in enumerate(selected)
    ]
    manifest_rows = [row for row in manifest_rows if row is not None]
    output.parent.mkdir(parents=True, exist_ok=True)
    _write_jsonl(output, manifest_rows)
    print(f"manifest={output}")
    print(f"rows={len(manifest_rows)}")


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset-config", type=Path, default=Path("configs/ttas_dataset.json"))
    parser.add_argument("--split", default="test", choices=["train", "val", "test"])
    parser.add_argument("--stage", required=True, choices=sorted(STAGES))
    parser.add_argument("--direction", required=True, choices=sorted(DIRECTIONS))
    parser.add_argument("--output", type=Path, help="Output manifest path. Defaults under temporary results.")
    parser.add_argument("--start-index", type=int, default=0)
    parser.add_argument("--max-samples", type=int)
    parser.add_argument("--skip-missing", action="store_true", help="Skip rows missing required paired fields/files.")
    parser.add_argument("--overwrite", action="store_true")
    return parser


def _split_path(dataset_config: dict[str, Any], split: str) -> Path:
    splits = dataset_config.get("splits")
    if not isinstance(splits, dict) or split not in splits:
        raise SystemExit(f"error: split {split!r} is not configured in dataset config")
    path = Path(splits[split])
    return path if path.is_absolute() else Path.cwd() / path


def _iter_jsonl(path: Path) -> Iterable[dict[str, Any]]:
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if line:
                yield json.loads(line)


def _select_rows(rows: list[dict[str, Any]], start_index: int, max_samples: int | None) -> list[dict[str, Any]]:
    if start_index < 0:
        raise SystemExit("error: --start-index must be non-negative")
    if max_samples is not None and max_samples < 1:
        raise SystemExit("error: --max-samples must be positive")
    selected = rows[start_index:]
    return selected[:max_samples] if max_samples is not None else selected


def _manifest_row(
    *,
    source: dict[str, Any],
    ordinal: int,
    dataset_root: Path,
    stage: str,
    direction: str,
    skip_missing: bool,
) -> dict[str, Any] | None:
    try:
        sample = _sample_id(source, ordinal)
        source_copy = dict(source)
        source_copy["stage_provenance"] = {
            "stage": stage,
            "direction": direction,
            "pipeline": direction.upper(),
            "mode": _mode(direction),
            "source": "dataset_pair_eval_manifest",
        }
        if stage == "t2t":
            input_text, output_text = _text_pair(source, direction)
            return {
                "row_index": ordinal,
                "sample_id": sample,
                "task": "T2T",
                "stage": "t2t",
                "direction": _long_direction(direction),
                "status": "success",
                "input_text": input_text,
                "prompt": input_text,
                "candidate_text": output_text,
                "generated_text": output_text,
                "source": source_copy,
                "model_extra": {
                    "mode": _mode(direction),
                    "dataset_pair_eval": True,
                    "stages": [
                        {
                            "stage": "t2t",
                            "task": "text_to_text",
                            "model": "paired_dataset_reference",
                            "mode": _mode(direction),
                            "input": {"type": _input_text_type(direction), "text": input_text},
                            "output": {"type": "text", "text": output_text},
                        }
                    ],
                },
            }
        if stage == "t2s":
            if direction != "a2s":
                raise ValueError("--stage t2s requires --direction a2s")
            output_file = _required_file(source, dataset_root, "speech_file")
            _required_file(source, dataset_root, "audio_file")
            task = "T2S"
        elif stage == "t2a":
            if direction != "s2a":
                raise ValueError("--stage t2a requires --direction s2a")
            output_file = _required_file(source, dataset_root, "audio_file")
            _required_file(source, dataset_root, "speech_file")
            task = "T2A"
        else:
            raise ValueError(f"unsupported stage: {stage}")
        input_text, _ = _text_pair(source, direction)
        return {
            "row_index": ordinal,
            "sample_id": sample,
            "task": task,
            "stage": stage,
            "direction": _long_direction(direction),
            "status": "success",
            "prompt": input_text,
            "conditioning_prompt": input_text,
            "output_file": str(output_file),
            "source": source_copy,
            "model_extra": {
                "mode": _mode(direction),
                "dataset_pair_eval": True,
                "output_source": "paired_dataset_audio",
            },
        }
    except Exception:
        if skip_missing:
            return None
        raise


def _text_pair(source: dict[str, Any], direction: str) -> tuple[str, str]:
    if direction == "a2s":
        return _required_text(source, "audio_cap"), _required_text(source, "speech_text")
    if direction == "s2a":
        return _required_text(source, "speech_text"), _required_text(source, "audio_cap")
    raise ValueError(f"unsupported direction: {direction}")


def _required_text(source: dict[str, Any], field: str) -> str:
    value = source.get(field)
    if isinstance(value, str) and value.strip():
        return value.strip()
    raise ValueError(f"missing required text field: {field}")


def _required_file(source: dict[str, Any], dataset_root: Path, field: str) -> Path:
    value = source.get(field)
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"missing required file field: {field}")
    path = Path(value)
    resolved = path if path.is_absolute() else dataset_root / path
    if not resolved.is_file():
        raise ValueError(f"required file does not exist for {field}: {resolved}")
    return resolved


def _sample_id(source: dict[str, Any], ordinal: int) -> str:
    value = source.get("sample_id")
    if isinstance(value, str) and value.strip():
        return value.strip()
    audio_file = source.get("audio_file")
    if isinstance(audio_file, str) and audio_file.strip():
        return Path(audio_file).stem
    return f"sample_{ordinal:08d}"


def _mode(direction: str) -> str:
    return "audio_description_to_speech" if direction == "a2s" else "speech_to_audio_description"


def _long_direction(direction: str) -> str:
    return "audio_to_speech" if direction == "a2s" else "speech_to_audio"


def _input_text_type(direction: str) -> str:
    return "audio_caption" if direction == "a2s" else "speech_transcript"


def _default_output_path(*, dataset: str, split: str, stage: str, direction: str) -> Path:
    root = os.environ.get("SALM_TEMPORARY_RESULTS_ROOT")
    base = Path(root) if root else root_path("temporary_results_root")
    return base / "dataset_pair_eval_manifests" / dataset / split / f"{stage}_{direction}_dataset_pair_manifest.jsonl"


def _write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")


if __name__ == "__main__":
    main()
