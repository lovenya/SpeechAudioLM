#!/usr/bin/env python
from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time
from pathlib import Path
from typing import Any


def main() -> None:
    args = parse_args()
    task_index = args.task_index
    if task_index is None:
        value = os.environ.get("SLURM_ARRAY_TASK_ID")
        if value is None:
            raise SystemExit("error: --task-index or SLURM_ARRAY_TASK_ID is required")
        task_index = int(value)
    plan = json.loads(args.task_plan.read_text(encoding="utf-8"))
    tasks = plan.get("tasks")
    if not isinstance(tasks, list) or task_index < 0 or task_index >= len(tasks):
        raise SystemExit(f"error: task index {task_index} outside task plan range")
    task = tasks[task_index]
    workers = task.get("workers")
    if not isinstance(workers, list) or not workers:
        raise SystemExit(f"error: task {task_index} has no workers")

    log_root = args.log_root / f"task_{task_index:03d}"
    log_root.mkdir(parents=True, exist_ok=True)
    (log_root / "task.json").write_text(json.dumps(task, indent=2, sort_keys=True) + "\n", encoding="utf-8")

    processes: list[tuple[dict[str, Any], subprocess.Popen[bytes], Any, Path]] = []
    for worker in workers:
        gpu = int(worker["gpu"])
        output_parent = metric_output_parent(worker, args.output_root)
        if output_parent.exists() and any(output_parent.iterdir()) and not args.allow_existing_outputs:
            raise SystemExit(f"error: refusing to launch because output already exists: {output_parent}")
        command = evaluate_command(worker, args)
        command_path = log_root / f"{worker['label']}.command.txt"
        command_path.write_text(" ".join(quote(part) for part in command) + "\n", encoding="utf-8")
        log_path = log_root / f"{worker['label']}.log"
        env = os.environ.copy()
        env["CUDA_VISIBLE_DEVICES"] = str(gpu)
        env["PYTHONUNBUFFERED"] = "1"
        env.setdefault("TOKENIZERS_PARALLELISM", "false")
        handle = log_path.open("wb")
        print(f"launch label={worker['label']} gpu={gpu} rows={worker['row_count']} log={log_path}", flush=True)
        process = subprocess.Popen(command, cwd=args.repo_root, env=env, stdout=handle, stderr=subprocess.STDOUT)
        processes.append((worker, process, handle, log_path))
        time.sleep(args.launch_stagger_seconds)

    failures = 0
    results = []
    for worker, process, handle, log_path in processes:
        returncode = process.wait()
        handle.close()
        ok = returncode == 0
        failures += 0 if ok else 1
        results.append({"label": worker["label"], "returncode": returncode, "ok": ok, "log": str(log_path)})
        print(f"complete label={worker['label']} returncode={returncode} log={log_path}", flush=True)
    (log_root / "results.json").write_text(json.dumps({"task_index": task_index, "failures": failures, "results": results}, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    raise SystemExit(failures)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run one packed GEval audio v6 task from a prepared task plan.")
    parser.add_argument("--task-plan", type=Path, required=True)
    parser.add_argument("--task-index", type=int)
    parser.add_argument("--log-root", type=Path, required=True)
    parser.add_argument("--repo-root", type=Path, default=Path.cwd())
    parser.add_argument("--output-root", type=Path, default=Path(os.environ.get("SALM_RESULTS_ROOT", "/scratch/l/lovenya/salm_migration_clean/results")) / "evals")
    parser.add_argument("--cache-dir", type=Path, default=Path(os.environ.get("SALM_CACHE_ROOT", "/scratch/l/lovenya/salm_migration_clean/cache")) / "evaluations")
    parser.add_argument("--metrics-config", default="configs/eval_metrics.json")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--launch-stagger-seconds", type=float, default=2.0)
    parser.add_argument("--allow-existing-outputs", action="store_true")
    return parser.parse_args()


def evaluate_command(worker: dict[str, Any], args: argparse.Namespace) -> list[str]:
    command = [
        sys.executable,
        "evaluate.py",
        "--model", str(worker["model"]),
        "--stage", str(worker["stage"]),
        "--direction", str(worker["direction"]),
        "--evaluated-run-id", str(worker["evaluated_run_id"]),
        "--dataset", str(worker["dataset"]),
        "--split", str(worker["split"]),
        "--manifest", str(worker["manifest"]),
        "--metrics", str(worker["metric"]),
        "--metrics-config", args.metrics_config,
        "--output-root", str(args.output_root),
        "--cache-dir", str(args.cache_dir),
        "--device", args.device,
    ]
    if worker.get("eval_run_id"):
        command.extend(["--eval-run-id", str(worker["eval_run_id"])])
    return command


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


def quote(value: str) -> str:
    if not value or any(char.isspace() or char in "'\"$`\\" for char in value):
        return "'" + value.replace("'", "'\\''") + "'"
    return value


if __name__ == "__main__":
    main()
