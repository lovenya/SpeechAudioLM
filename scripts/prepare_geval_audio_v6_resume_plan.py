#!/usr/bin/env python
from __future__ import annotations

import argparse
import json
import math
import sys
from datetime import datetime
from pathlib import Path
from typing import Any
from zoneinfo import ZoneInfo

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from project_paths import root_path

METRIC = "geval_audio_plausibility"


def main() -> None:
    args = parse_args()
    source_plan = read_json(args.source_task_plan)
    run_group = args.run_group or f"{source_plan['run_group']}_resume_{datetime.now(ZoneInfo('America/Toronto')).strftime('%Y%m%d_%H%M')}"
    output_root = args.output_root or root_path("results_root") / "evals" / "shard_manifests" / "geval_audio_v6_plausibility"
    run_root = output_root / run_group
    if run_root.exists() and not args.overwrite:
        raise SystemExit(f"error: run root already exists: {run_root}")
    run_root.mkdir(parents=True, exist_ok=True)

    output_eval_root = args.eval_output_root or root_path("results_root") / "evals"
    resume_workers: list[dict[str, Any]] = []
    restart_workers: list[dict[str, Any]] = []
    skipped_complete: list[dict[str, Any]] = []

    for task in source_plan.get("tasks", []):
        for worker in task.get("workers", []):
            status = inspect_worker(worker, output_eval_root)
            if status["top_rows"] == worker.get("row_count"):
                skipped_complete.append({**worker, **status})
                continue
            item = dict(worker)
            item["source_run_group"] = source_plan["run_group"]
            item["retry_of_task_index"] = task.get("task_index")
            item["eval_run_id"] = status.get("eval_run_id")
            item["existing_eval_dir"] = status.get("eval_dir")
            item["existing_backend_rows"] = status.get("backend_rows", 0)
            item["existing_complete_rows"] = status.get("complete_rows", 0)
            item["existing_pending_rows"] = status.get("pending_rows", 0)
            item["run_group"] = run_group
            if status.get("backend_rows", 0) > 0:
                resume_workers.append(item)
            else:
                restart_workers.append(item)

    resume_tasks = pack_tasks(resume_workers, workers_per_task=args.workers_per_task)
    restart_tasks = pack_tasks(restart_workers, workers_per_task=args.workers_per_task)
    write_plan(run_root / "partial_resume_task_plan.json", source_plan, run_group, run_root, resume_workers, resume_tasks, skipped_complete, "partial_resume", args)
    write_plan(run_root / "restart_empty_task_plan.json", source_plan, run_group, run_root, restart_workers, restart_tasks, skipped_complete, "restart_empty", args)
    write_jsonl(run_root / "partial_resume_workers.jsonl", resume_workers)
    write_jsonl(run_root / "restart_empty_workers.jsonl", restart_workers)
    write_jsonl(run_root / "skipped_complete_workers.jsonl", skipped_complete)
    write_text_env(run_root / "run_env.sh", run_group, run_root, len(resume_tasks), len(restart_tasks))

    print(f"run_group={run_group}")
    print(f"run_root={run_root}")
    print(f"partial_resume_task_plan={run_root / 'partial_resume_task_plan.json'}")
    print(f"partial_resume_workers={len(resume_workers)}")
    print(f"partial_resume_tasks={len(resume_tasks)}")
    print(f"restart_empty_task_plan={run_root / 'restart_empty_task_plan.json'}")
    print(f"restart_empty_workers={len(restart_workers)}")
    print(f"restart_empty_tasks={len(restart_tasks)}")
    print(f"skipped_complete_workers={len(skipped_complete)}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Prepare retry plans for incomplete v6 audio GEval shard workers.")
    parser.add_argument("--source-task-plan", type=Path, required=True)
    parser.add_argument("--run-group")
    parser.add_argument("--output-root", type=Path)
    parser.add_argument("--eval-output-root", type=Path)
    parser.add_argument("--workers-per-task", type=int, default=4)
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def inspect_worker(worker: dict[str, Any], output_root: Path) -> dict[str, Any]:
    parent = metric_output_parent(worker, output_root)
    eval_dirs = sorted([path for path in parent.iterdir() if path.is_dir()]) if parent.exists() else []
    eval_dir = eval_dirs[-1] if eval_dirs else None
    manifest = eval_dir / "manifest.jsonl" if eval_dir else None
    top_rows = count_jsonl(manifest)
    backend = eval_dir / "artifacts" / "dedicated-script" / worker["metric"] if eval_dir else None
    output = backend / "output.jsonl" if backend else None
    backend_rows = count_jsonl(output)
    complete_rows = 0
    pending_rows = 0
    summary_path = backend / "summary.json" if backend else None
    if summary_path and summary_path.is_file():
        summary = read_json(summary_path)
        counts = summary.get("status_counts") if isinstance(summary.get("status_counts"), dict) else {}
        complete_rows = int(counts.get("complete", 0) or 0) + int(counts.get("complete_with_abstention", 0) or 0)
        pending_rows = int(counts.get("pending", 0) or 0)
    return {
        "eval_run_id": eval_dir.name if eval_dir else None,
        "eval_dir": str(eval_dir) if eval_dir else None,
        "top_rows": top_rows,
        "backend_rows": backend_rows,
        "complete_rows": complete_rows,
        "pending_rows": pending_rows,
    }


def metric_output_parent(worker: dict[str, Any], output_root: Path) -> Path:
    return (
        output_root
        / "stages"
        / str(worker["direction"])
        / str(worker["stage"])
        / str(worker["model"])
        / str(worker["dataset"])
        / str(worker["split"])
        / str(worker["evaluated_run_id"])
        / str(worker["metric"])
    )


def pack_tasks(workers: list[dict[str, Any]], *, workers_per_task: int) -> list[dict[str, Any]]:
    if workers_per_task <= 0:
        raise ValueError("workers_per_task must be positive")
    tasks: list[dict[str, Any]] = []
    for start in range(0, len(workers), workers_per_task):
        packed = []
        for gpu, worker in enumerate(workers[start:start + workers_per_task]):
            item = dict(worker)
            item["gpu"] = gpu
            item["worker_index"] = start + gpu
            packed.append(item)
        tasks.append({"task_index": len(tasks), "workers": packed})
    return tasks


def write_plan(
    path: Path,
    source_plan: dict[str, Any],
    run_group: str,
    run_root: Path,
    workers: list[dict[str, Any]],
    tasks: list[dict[str, Any]],
    skipped_complete: list[dict[str, Any]],
    retry_mode: str,
    args: argparse.Namespace,
) -> None:
    plan = {
        "schema_version": 1,
        "run_group": run_group,
        "source_run_group": source_plan["run_group"],
        "retry_mode": retry_mode,
        "metric": source_plan.get("metric", METRIC),
        "prompt_version_expected": source_plan.get("prompt_version_expected"),
        "created_at": datetime.now().astimezone().isoformat(),
        "repo_root": str(REPO_ROOT),
        "run_root": str(run_root),
        "source_task_plan": str(args.source_task_plan),
        "workers_per_task": args.workers_per_task,
        "worker_count": len(workers),
        "task_count": len(tasks),
        "skipped_complete_count": len(skipped_complete),
        "tasks": tasks,
    }
    write_json(path, plan)


def count_jsonl(path: Path | None) -> int:
    if path is None or not path.is_file():
        return 0
    with path.open("r", encoding="utf-8") as src:
        return sum(1 for line in src if line.strip())


def read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def write_json(path: Path, data: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as dst:
        for row in rows:
            dst.write(json.dumps(row, sort_keys=True) + "\n")


def write_text_env(path: Path, run_group: str, run_root: Path, resume_tasks: int, restart_tasks: int) -> None:
    lines = [
        f"export RUN_GROUP={shell_quote(run_group)}",
        f"export RESUME_TASK_PLAN={shell_quote(str(run_root / 'partial_resume_task_plan.json'))}",
        f"export RESTART_TASK_PLAN={shell_quote(str(run_root / 'restart_empty_task_plan.json'))}",
        f"export RESUME_TASK_COUNT={resume_tasks}",
        f"export RESTART_TASK_COUNT={restart_tasks}",
        "",
    ]
    path.write_text("\n".join(lines), encoding="utf-8")


def shell_quote(value: str) -> str:
    return "'" + value.replace("'", "'\\''") + "'"


if __name__ == "__main__":
    main()
