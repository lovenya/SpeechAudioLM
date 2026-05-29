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

    if output_manifest.exists() and not args.overwrite:
        raise SystemExit(f"error: {output_manifest} exists; pass --overwrite to replace it")

    manifests = sorted(
        shard_root.glob(f"*/{args.model}/{args.task}/{args.dataset}/{args.split}/manifest.jsonl")
    )
    if not manifests:
        raise SystemExit(
            "error: no shard manifests found under "
            f"{shard_root} for {args.model}/{args.task}/{args.dataset}/{args.split}"
        )

    rows: list[dict] = []
    seen: set[tuple[int | str, str]] = set()
    duplicates: list[str] = []
    for manifest in manifests:
        with manifest.open("r", encoding="utf-8") as handle:
            for line_number, line in enumerate(handle, start=1):
                line = line.strip()
                if not line:
                    continue
                row = json.loads(line)
                key = (row.get("row_index", ""), str(row.get("sample_id", "")))
                if key in seen:
                    duplicates.append(f"{manifest}:{line_number}: row_index={key[0]} sample_id={key[1]}")
                    continue
                seen.add(key)
                rows.append(row)

    if duplicates:
        preview = "\n".join(duplicates[:20])
        raise SystemExit(f"error: duplicate rows found while merging:\n{preview}")

    rows.sort(key=lambda item: (int(item.get("row_index", 0)), str(item.get("sample_id", ""))))
    output_root.mkdir(parents=True, exist_ok=True)
    with output_manifest.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")

    print(f"merged_manifests={len(manifests)}")
    print(f"merged_rows={len(rows)}")
    print(f"output_manifest={output_manifest}")


if __name__ == "__main__":
    main()
