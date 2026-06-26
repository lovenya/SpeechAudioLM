#!/usr/bin/env python
"""Build split-aware reference pools from primary audio class manifests."""

from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path
from typing import Any


DEFAULT_INPUT_MANIFESTS = {
    "train": Path("data/manifests/audio_classes/ttas_train_panns_cnn14_classes.jsonl"),
    "val": Path("data/manifests/audio_classes/ttas_val_panns_cnn14_classes.jsonl"),
    "test": Path("data/manifests/audio_classes/ttas_test_panns_cnn14_classes.jsonl"),
}
DEFAULT_OUTPUT_DIR = Path("data/reports/audio_classes/reference_pools")
DEFAULT_POOL_OUTPUT = DEFAULT_OUTPUT_DIR / "class_reference_pools.json"
DEFAULT_MAPPING_OUTPUT = DEFAULT_OUTPUT_DIR / "pair_reference_mapping.jsonl"
DEFAULT_SUMMARY_OUTPUT = DEFAULT_OUTPUT_DIR / "reference_pool_summary.json"
SPLITS = ("train", "val", "test")


def build_reference_pools(
    *,
    input_manifests: dict[str, Path],
    pool_output: Path,
    mapping_output: Path,
    summary_output: Path,
    splits: list[str] | None = None,
) -> dict[str, Any]:
    split_names = splits or list(input_manifests)
    pools: dict[str, dict[str, list[dict[str, Any]]]] = {}
    mappings: list[dict[str, Any]] = []
    class_counts: Counter[str] = Counter()
    class_counts_by_split: dict[str, Counter[str]] = {}

    for split in split_names:
        if split not in input_manifests:
            available = ", ".join(input_manifests)
            raise ValueError(f"Unknown split {split!r}; available splits: {available}")
        split_counts: Counter[str] = Counter()
        for row in _iter_jsonl(input_manifests[split]):
            record = _reference_record(row)
            class_name = _required_str(row, "primary_audio_class")
            pools.setdefault(class_name, {name: [] for name in SPLITS})[split].append(record)
            mapping = _pair_mapping(row, split=split, class_name=class_name, record=record)
            _validate_mapping_points_to_pool(mapping, pools)
            mappings.append(mapping)
            class_counts.update([class_name])
            split_counts.update([class_name])
        class_counts_by_split[split] = split_counts

    summary = _summary(
        input_manifests=input_manifests,
        pool_output=pool_output,
        mapping_output=mapping_output,
        splits=split_names,
        pools=pools,
        mappings=mappings,
        class_counts=class_counts,
        class_counts_by_split=class_counts_by_split,
    )
    _write_json(pool_output, pools)
    _write_jsonl(mapping_output, mappings)
    _write_json(summary_output, summary)
    print(f"wrote {pool_output} classes={summary['num_classes']}")
    print(f"wrote {mapping_output} rows={summary['total_pairs']}")
    print(f"wrote {summary_output}")
    return summary


def _reference_record(row: dict[str, Any]) -> dict[str, str]:
    audio_path = _required_str(row, "audio_file")
    speech_path = _required_str(row, "speech_file")
    return {
        "pair_id": _pair_id(row),
        "audio_id": Path(audio_path).stem,
        "speech_id": Path(speech_path).stem,
        "audio_path": audio_path,
        "speech_path": speech_path,
        "speech_transcript": _required_str(row, "speech_text"),
        "audio_caption": str(row.get("audio_cap") or ""),
    }


def _pair_mapping(row: dict[str, Any], *, split: str, class_name: str, record: dict[str, str]) -> dict[str, Any]:
    mapping = {
        "pair_id": record["pair_id"],
        "split": split,
        "audio_id": record["audio_id"],
        "speech_id": record["speech_id"],
        "audio_path": record["audio_path"],
        "speech_path": record["speech_path"],
        "speech_transcript": record["speech_transcript"],
        "audio_caption": record["audio_caption"],
        "primary_audio_class": class_name,
        "audio_to_speech": {
            "reference_pool_class": class_name,
            "reference_pool_split": split,
        },
        "speech_to_audio": {
            "reference_pool_class": class_name,
            "reference_pool_split": split,
        },
    }
    return mapping


def _validate_mapping_points_to_pool(mapping: dict[str, Any], pools: dict[str, dict[str, list[dict[str, Any]]]]) -> None:
    primary = mapping["primary_audio_class"]
    for task in ("audio_to_speech", "speech_to_audio"):
        target = mapping[task]
        if target["reference_pool_class"] != primary:
            raise ValueError(f"{task} class does not match primary_audio_class for {mapping['pair_id']}")
        split = target["reference_pool_split"]
        if primary not in pools or split not in pools[primary] or not pools[primary][split]:
            raise ValueError(f"{task} points to missing pool {primary}/{split} for {mapping['pair_id']}")


def _summary(
    *,
    input_manifests: dict[str, Path],
    pool_output: Path,
    mapping_output: Path,
    splits: list[str],
    pools: dict[str, dict[str, list[dict[str, Any]]]],
    mappings: list[dict[str, Any]],
    class_counts: Counter[str],
    class_counts_by_split: dict[str, Counter[str]],
) -> dict[str, Any]:
    pool_sizes_by_class_split = {
        class_name: {split: len(records) for split, records in split_records.items()}
        for class_name, split_records in sorted(pools.items())
    }
    empty_pools = [
        {"primary_audio_class": class_name, "split": split}
        for class_name, split_records in sorted(pools.items())
        for split, records in split_records.items()
        if not records
    ]
    pair_ids = [mapping["pair_id"] for mapping in mappings]
    duplicate_pair_ids = _duplicates(pair_ids)
    invalid_reference_targets = [
        mapping["pair_id"]
        for mapping in mappings
        if mapping["audio_to_speech"]["reference_pool_class"] != mapping["primary_audio_class"]
        or mapping["speech_to_audio"]["reference_pool_class"] != mapping["primary_audio_class"]
    ]
    return {
        "schema_version": 1,
        "input_manifests": {split: str(input_manifests[split]) for split in splits},
        "class_reference_pools": str(pool_output),
        "pair_reference_mapping": str(mapping_output),
        "splits": splits,
        "num_classes": len(pools),
        "total_pairs": len(mappings),
        "unique_pair_ids": len(set(pair_ids)),
        "duplicate_pair_ids": duplicate_pair_ids,
        "unknown_count": class_counts.get("unknown", 0),
        "class_counts": _sorted_counter(class_counts),
        "class_counts_by_split": {split: _sorted_counter(counts) for split, counts in class_counts_by_split.items()},
        "pool_sizes_by_class_split": pool_sizes_by_class_split,
        "empty_pools": empty_pools,
        "invalid_reference_targets": invalid_reference_targets,
        "validation": {
            "all_pair_ids_unique": not duplicate_pair_ids,
            "all_mapping_classes_match_primary_audio_class": not invalid_reference_targets,
            "all_mapping_targets_exist": True,
        },
        "schema_notes": [
            "Pools are class -> split -> list of reference records, not parallel arrays.",
            "pair_id is sample_id when present; otherwise it is derived from audio_file stem.",
            "audio_id and speech_id are derived from audio_file and speech_file stems.",
            "audio_to_speech and speech_to_audio store direction-specific reference pool class/split fields.",
        ],
    }


def _pair_id(row: dict[str, Any]) -> str:
    value = row.get("sample_id")
    if isinstance(value, str) and value:
        return value
    return Path(_required_str(row, "audio_file")).stem


def _required_str(row: dict[str, Any], field: str) -> str:
    value = row.get(field)
    if not isinstance(value, str) or not value:
        raise ValueError(f"Row has no usable {field!r}: {row}")
    return value


def _duplicates(values: list[str]) -> list[str]:
    counts = Counter(values)
    return sorted(value for value, count in counts.items() if count > 1)


def _sorted_counter(counter: Counter[str]) -> dict[str, int]:
    return dict(sorted(counter.items(), key=lambda item: (-item[1], item[0])))


def _iter_jsonl(path: Path):
    with path.open("r", encoding="utf-8") as src:
        for line in src:
            if line.strip():
                yield json.loads(line)


def _write_json(path: Path, data: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")


def _write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as dst:
        for row in rows:
            dst.write(json.dumps(row, ensure_ascii=False) + "\n")


def _parse_manifest_arg(values: list[str] | None) -> dict[str, Path]:
    if not values:
        return dict(DEFAULT_INPUT_MANIFESTS)
    manifests: dict[str, Path] = {}
    for value in values:
        if "=" not in value:
            raise ValueError("--input-manifest values must use SPLIT=PATH")
        split, path = value.split("=", 1)
        manifests[split] = Path(path)
    return manifests


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-manifest", action="append", help="Input manifest as SPLIT=PATH. Repeat for multiple splits.")
    parser.add_argument("--pool-output", type=Path, default=DEFAULT_POOL_OUTPUT)
    parser.add_argument("--mapping-output", type=Path, default=DEFAULT_MAPPING_OUTPUT)
    parser.add_argument("--summary-output", type=Path, default=DEFAULT_SUMMARY_OUTPUT)
    parser.add_argument("--split", action="append", choices=list(SPLITS), help="Split to process. Repeat for multiple splits.")
    return parser


def main() -> None:
    args = _build_parser().parse_args()
    build_reference_pools(
        input_manifests=_parse_manifest_arg(args.input_manifest),
        pool_output=args.pool_output,
        mapping_output=args.mapping_output,
        summary_output=args.summary_output,
        splits=args.split,
    )


if __name__ == "__main__":
    main()
