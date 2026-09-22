#!/usr/bin/env python
"""Prepare immutable, one-GPU-per-shard v8 scenario-plausibility eval plans."""

from __future__ import annotations

import argparse
import copy
import json
import math
import sys
from datetime import datetime
from pathlib import Path
from typing import Any, Iterable
from zoneinfo import ZoneInfo

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from project_paths import root_path


METRIC = "geval_audio_plausibility"
CRITERIA = ["scenario_plausibility"]
DATASET = "ttas"
SOURCES = (
    {
        "label": "cascaded_a2s",
        "model": "cascaded",
        "stage": "t2s",
        "direction": "a2s",
        "evaluated_run_id": "20260618_1211",
        "manifest": "/scratch/lovenya/salm_migration_clean/results/evals/target_manifests/cascaded/ttas/test/clean_v2/20260618_1211/a2s_manifest.jsonl",
        "expected_rows": 2164,
    },
    {
        "label": "cascaded_s2a",
        "model": "cascaded",
        "stage": "t2a",
        "direction": "s2a",
        "evaluated_run_id": "20260618_1211",
        "manifest": "/scratch/lovenya/salm_migration_clean/results/evals/target_manifests/cascaded/ttas/test/clean_v2/20260618_1211/s2a_manifest.jsonl",
        "expected_rows": 1878,
    },
    {
        "label": "ground_truth_a2s",
        "model": "dataset_pair",
        "stage": "t2s",
        "direction": "a2s",
        "evaluated_run_id": "clean_v2_a2s_ground_truth_20260917_0001",
        "manifest": "/scratch/lovenya/salm_migration_clean/results/evals/target_manifests/dataset_pair/ttas/test/clean_v2/t2s_a2s_ground_truth.jsonl",
        "expected_rows": 2197,
    },
    {
        "label": "ground_truth_s2a",
        "model": "dataset_pair",
        "stage": "t2a",
        "direction": "s2a",
        "evaluated_run_id": "clean_v2_s2a_ground_truth_20260917_0002",
        "manifest": "/scratch/lovenya/salm_migration_clean/results/evals/target_manifests/dataset_pair/ttas/test/clean_v2/t2a_s2a_ground_truth.jsonl",
        "expected_rows": 1878,
    },
    {
        "label": "salm_flat_a2s",
        "model": "salm_e2e",
        "stage": "a2s",
        "direction": "a2s",
        "evaluated_run_id": "flat_k2048_ep11_seed1337_steps100",
        "manifest": "/scratch/lovenya/salm_model_outputs/eval_exports/a2s_clean_v2_test/flat_k2048_ep11_seed1337_steps100/manifest.jsonl",
        "expected_rows": 2197,
    },
    {
        "label": "salm_hier_a2s",
        "model": "salm_e2e",
        "stage": "a2s",
        "direction": "a2s",
        "evaluated_run_id": "hier_k2048_s12_ep12_seed1337_steps100",
        "manifest": "/scratch/lovenya/salm_model_outputs/eval_exports/a2s_clean_v2_test/hier_k2048_s12_ep12_seed1337_steps100/manifest.jsonl",
        "expected_rows": 2197,
    },
    {
        "label": "salm_ep70_s2a",
        "model": "salm_e2e",
        "stage": "s2a",
        "direction": "s2a",
        "evaluated_run_id": "s2a_clean_v2_test_ep70_seed1337_steps100",
        "manifest": "/scratch/lovenya/salm_model_outputs/runs/s2a_audioldm_whisper_xformer/long100_L2_o2_rorqual_b64_2gpu/eval_export/s2a_clean_v2_test_ep70_seed1337_steps100/manifest.jsonl",
        "expected_rows": 1878,
    },
)


def main() -> None:
    args = parse_args()
    run_group = args.run_group or f"geval_audio_v8_scenario_{datetime.now(ZoneInfo('America/Toronto')).strftime('%Y%m%d_%H%M')}"
    root = args.output_root / run_group
    if root.exists():
        raise SystemExit(f"refusing existing run-plan root: {root}")
    workers: list[dict[str, Any]] = []
    collections: list[dict[str, Any]] = []
    for spec in SOURCES:
        manifest = Path(spec["manifest"])
        rows = [normalise_output_paths(row, manifest.parent) for row in iter_jsonl(manifest) if row.get("status") in (None, "success")]
        if len(rows) != spec["expected_rows"]:
            raise SystemExit(f"unexpected included-row count for {spec['label']}: {len(rows)} != {spec['expected_rows']}")
        collection_root = root / spec["label"]
        shard_count = math.ceil(len(rows) / args.shard_size)
        collections.append({**spec, "row_count": len(rows), "shard_count": shard_count, "shard_size": args.shard_size})
        for shard_index in range(shard_count):
            start = shard_index * args.shard_size
            shard_rows = rows[start : start + args.shard_size]
            shard_path = collection_root / f"shard_{shard_index:03d}" / "manifest.jsonl"
            write_jsonl(shard_path, shard_rows)
            workers.append(
                {
                    "worker_index": len(workers),
                    "label": f"{spec['label']}_shard_{shard_index:03d}",
                    "gpu": 0,
                    "model": spec["model"],
                    "stage": spec["stage"],
                    "direction": spec["direction"],
                    "dataset": DATASET,
                    "split": "test",
                    "metric": METRIC,
                    "criteria": CRITERIA,
                    "row_count": len(shard_rows),
                    "shard_index": shard_index,
                    "manifest": str(shard_path),
                    "source_manifest": str(manifest),
                    "evaluated_run_id": spec["evaluated_run_id"],
                    "eval_run_id": f"{run_group}_{spec['label']}_shard_{shard_index:03d}",
                }
            )
    tasks = [{"task_index": worker["worker_index"], "workers": [worker]} for worker in workers]
    plan = {
        "schema_version": 1,
        "run_group": run_group,
        "metric": METRIC,
        "criteria": CRITERIA,
        "prompt_version_expected": "geval_audio_plausibility_v8_independent_anchor_coverage",
        "created_at": datetime.now().astimezone().isoformat(),
        "shard_size": args.shard_size,
        "worker_count": len(workers),
        "task_count": len(tasks),
        "collections": collections,
        "tasks": tasks,
    }
    write_json(root / "task_plan.json", plan)
    write_jsonl(root / "workers.jsonl", workers)
    print(f"run_group={run_group}")
    print(f"task_plan={root / 'task_plan.json'}")
    print(f"task_count={len(tasks)}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-group")
    parser.add_argument("--shard-size", type=int, default=456)
    parser.add_argument("--output-root", type=Path, default=root_path("results_root") / "evals/shard_manifests/geval_audio_v8_scenario")
    args = parser.parse_args()
    if args.shard_size < 1:
        parser.error("--shard-size must be positive")
    return args


def iter_jsonl(path: Path) -> Iterable[dict[str, Any]]:
    with path.open(encoding="utf-8") as src:
        for line in src:
            if line.strip():
                yield json.loads(line)


def normalise_output_paths(row: dict[str, Any], manifest_dir: Path) -> dict[str, Any]:
    copied = copy.deepcopy(row)
    output = copied.get("output")
    if isinstance(output, dict):
        for key in ("audio_file", "speech_file", "file"):
            value = output.get(key)
            if isinstance(value, str) and value and not Path(value).is_absolute():
                output[key] = str(manifest_dir / value)
    return copied


def write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def write_jsonl(path: Path, rows: Iterable[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as dst:
        for row in rows:
            dst.write(json.dumps(row, ensure_ascii=False) + "\n")


if __name__ == "__main__":
    main()
