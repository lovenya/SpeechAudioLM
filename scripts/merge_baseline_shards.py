#!/usr/bin/env python
"""Merge separately written baseline shard manifests into the canonical manifest."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from baselines.cascaded_manifest import unwrap_sample, write_inspection_manifest
from project_conventions import baseline_output_root


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", required=True)
    parser.add_argument("--task", required=True)
    parser.add_argument("--dataset", default="ttas")
    parser.add_argument("--split", default="test")
    parser.add_argument(
        "--shard-root",
        default="/scratch/l/lovenya/salm_migration_clean/results/baselines_shards",
        help="Root containing shard output directories.",
    )
    parser.add_argument(
        "--output-root",
        default="/scratch/l/lovenya/salm_migration_clean/results/baselines",
        help="Canonical baseline output root to write merged manifest under.",
    )
    parser.add_argument("--overwrite", action="store_true", help="Replace an existing merged manifest.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    shard_root = Path(args.shard_root)
    output_root = baseline_output_root(args.model, args.task, args.dataset, args.split, Path(args.output_root))
    output_manifest = output_root / "manifest.jsonl"
    output_inspection = output_root / "manifest.inspect.json"

    if output_manifest.exists() and not args.overwrite:
        raise SystemExit(f"error: {output_manifest} exists; pass --overwrite to replace it")

    if args.overwrite and output_inspection.exists():
        output_inspection.unlink()

    manifests = sorted(
        shard_root.glob(f"*/{args.model}/{args.task}/{args.dataset}/{args.split}/manifest.jsonl")
    )
    if not manifests:
        raise SystemExit(
            "error: no shard manifests found under "
            f"{shard_root} for {args.model}/{args.task}/{args.dataset}/{args.split}"
        )

    rows: list[dict] = []
    samples: list[dict] = []
    inspection_metadata: dict | None = None
    stage_definitions: list[dict] | None = None
    created_at: str | None = None
    started_at: str | None = None
    completed_at: str | None = None
    wrapped_format: bool | None = None
    seen: set[tuple[int | str, str]] = set()
    duplicates: list[str] = []
    for manifest in manifests:
        inspection_path = manifest.with_name("manifest.inspect.json")
        if inspection_path.is_file():
            inspection = json.loads(inspection_path.read_text(encoding="utf-8"))
            current_metadata = {key: inspection.get(key) for key in ("lane", "pipeline", "dataset", "split", "config_file")}
            current_definitions = inspection.get("stage_definitions") or []
            if inspection_metadata is None:
                inspection_metadata = current_metadata
                stage_definitions = current_definitions
                created_at = inspection.get("created_at")
                started_at = inspection.get("started_at")
            else:
                if current_metadata != inspection_metadata or not _compatible_stage_definitions(current_definitions, stage_definitions or []):
                    raise SystemExit(f"error: incompatible shard inspection metadata: {inspection_path}")
                stage_definitions = _merge_stage_definitions(stage_definitions or [], current_definitions)
            completed_at = inspection.get("completed_at") or completed_at
        with manifest.open("r", encoding="utf-8") as handle:
            for line_number, line in enumerate(handle, start=1):
                line = line.strip()
                if not line:
                    continue
                row = json.loads(line)
                is_wrapped = isinstance(row.get("sample"), dict)
                if wrapped_format is None:
                    wrapped_format = is_wrapped
                elif wrapped_format != is_wrapped:
                    raise SystemExit("error: cannot merge mixed wrapped and legacy shard manifests")
                if is_wrapped:
                    _validate_wrapper(row, inspection_metadata, manifest, line_number)
                sample = unwrap_sample(row)
                key = (sample.get("row_index", ""), str(sample.get("sample_id", "")))
                if key in seen:
                    duplicates.append(f"{manifest}:{line_number}: row_index={key[0]} sample_id={key[1]}")
                    continue
                seen.add(key)
                rows.append(row)
                samples.append(sample)

    if duplicates:
        preview = "\n".join(duplicates[:20])
        raise SystemExit(f"error: duplicate rows found while merging:\n{preview}")

    rows.sort(key=lambda item: (int(unwrap_sample(item).get("row_index", 0)), str(unwrap_sample(item).get("sample_id", ""))))
    samples.sort(key=lambda item: (int(item.get("row_index", 0)), str(item.get("sample_id", ""))))
    output_root.mkdir(parents=True, exist_ok=True)
    with output_manifest.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")

    if inspection_metadata is not None and stage_definitions is not None:
        write_inspection_manifest(
            output_inspection,
            metadata=inspection_metadata,
            stage_definitions=stage_definitions,
            samples=samples,
            created_at=created_at or "",
            started_at=started_at or "",
            completed_at=completed_at or "",
            status="completed" if all(sample.get("status") == "success" for sample in samples) else "completed_with_errors",
        )

    print(f"merged_manifests={len(manifests)}")
    print(f"merged_rows={len(rows)}")
    print(f"output_manifest={output_manifest}")


def _static_stage_definition(definition: dict) -> dict:
    static = dict(definition)
    runtime = dict(static.get("runtime") or {})
    runtime.pop("model_load_seconds", None)
    static["runtime"] = runtime
    return static


def _compatible_stage_definitions(left: list[dict], right: list[dict]) -> bool:
    return [_static_stage_definition(item) for item in left] == [_static_stage_definition(item) for item in right]


def _merge_stage_definitions(left: list[dict], right: list[dict]) -> list[dict]:
    merged = []
    for first, second in zip(left, right, strict=True):
        item = dict(first)
        runtime = dict(item.get("runtime") or {})
        values = [
            value
            for value in ((first.get("runtime") or {}).get("model_load_seconds"), (second.get("runtime") or {}).get("model_load_seconds"))
            if isinstance(value, (int, float))
        ]
        runtime["model_load_seconds"] = max(values) if values else None
        item["runtime"] = runtime
        merged.append(item)
    return merged


def _validate_wrapper(row: dict, inspection_metadata: dict | None, manifest: Path, line_number: int) -> None:
    if inspection_metadata is None:
        raise SystemExit(f"error: wrapped shard lacks manifest.inspect.json: {manifest}")
    expected = {
        "schema_version": "1.0",
        **inspection_metadata,
    }
    for key, value in expected.items():
        if row.get(key) != value:
            raise SystemExit(f"error: {manifest}:{line_number} wrapper {key}={row.get(key)!r}; expected {value!r}")
    sample = row["sample"]
    stages = sample.get("stages") if isinstance(sample.get("stages"), list) else []
    if row.get("current_stage") != len(stages):
        raise SystemExit(f"error: {manifest}:{line_number} current_stage does not match sample stages")


if __name__ == "__main__":
    main()
