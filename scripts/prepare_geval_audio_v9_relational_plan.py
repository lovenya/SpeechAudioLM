#!/usr/bin/env python
"""Derive a resumable v9 relational-criteria plan from the v8 shard plan."""

from __future__ import annotations

import argparse
import copy
import json
import sys
from datetime import datetime
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from project_paths import root_path


CRITERIA = {
    "a2s": ["controlled_detail", "expressive_delivery_fit"],
    "s2a": ["controlled_detail", "contextual_acoustic_fit"],
}
PROMPT_VERSION = "geval_audio_plausibility_v9_criterion_evidence_boundaries"


def main() -> None:
    args = parse_args()
    source = json.loads(args.source_task_plan.read_text(encoding="utf-8"))
    run_root = args.output_root / args.run_group
    if run_root.exists():
        raise SystemExit(f"refusing existing run-plan root: {run_root}")

    tasks: list[dict[str, Any]] = []
    workers: list[dict[str, Any]] = []
    for source_task in source.get("tasks", []):
        source_workers = source_task.get("workers")
        if not isinstance(source_workers, list) or len(source_workers) != 1:
            raise SystemExit("source plan must contain exactly one worker per task")
        worker = copy.deepcopy(source_workers[0])
        direction = str(worker.get("direction"))
        if direction not in CRITERIA:
            raise SystemExit(f"unsupported worker direction: {direction}")
        worker["run_group"] = args.run_group
        worker["criteria"] = CRITERIA[direction]
        worker["eval_run_id"] = f"{args.run_group}_{worker['label']}"
        workers.append(worker)
        tasks.append({"task_index": len(tasks), "workers": [worker]})

    plan = {
        "schema_version": 1,
        "run_group": args.run_group,
        "source_task_plan": str(args.source_task_plan),
        "source_run_group": source.get("run_group"),
        "metric": "geval_audio_plausibility",
        "criteria_by_direction": CRITERIA,
        "prompt_version_expected": PROMPT_VERSION,
        "created_at": datetime.now().astimezone().isoformat(),
        "worker_count": len(workers),
        "task_count": len(tasks),
        "collections": source.get("collections", []),
        "tasks": tasks,
    }
    write_json(run_root / "task_plan.json", plan)
    write_jsonl(run_root / "workers.jsonl", workers)
    print(f"run_group={args.run_group}")
    print(f"task_plan={run_root / 'task_plan.json'}")
    print(f"task_count={len(tasks)}")


def parse_args() -> argparse.Namespace:
    default_source = (
        root_path("results_root")
        / "evals/shard_manifests/geval_audio_v8_scenario/"
        / "geval_audio_v8_scenario_20260921_1710/task_plan.json"
    )
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-group", required=True)
    parser.add_argument("--source-task-plan", type=Path, default=default_source)
    parser.add_argument(
        "--output-root",
        type=Path,
        default=root_path("results_root") / "evals/shard_manifests/geval_audio_v9_relational",
    )
    return parser.parse_args()


def write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as dst:
        for row in rows:
            dst.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")


if __name__ == "__main__":
    main()
