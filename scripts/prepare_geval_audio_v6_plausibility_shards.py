#!/usr/bin/env python
from __future__ import annotations

import argparse
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
from scripts.build_dataset_pair_eval_manifest import _manifest_row

DEFAULT_CASCADED_A2S = Path(
    "/scratch/l/lovenya/salm_migration_clean/results/cascaded/stages/t2s/"
    "cosyvoice3/from-t2t/llama3_3_70b_instruct/ttas/test/20260618_1211/manifest.jsonl"
)
DEFAULT_CASCADED_S2A = Path(
    "/scratch/l/lovenya/salm_migration_clean/results/cascaded/stages/t2a/"
    "audio_omni/from-t2t/llama3_3_70b_instruct/ttas/test/20260618_1211/manifest.jsonl"
)
METRIC = "geval_audio_plausibility"
MODEL = "cascaded"
DATASET = "ttas"


def main() -> None:
    args = parse_args()
    run_group = args.run_group or f"geval_audio_v6_plausibility_{datetime.now(ZoneInfo('America/Toronto')).strftime('%Y%m%d_%H%M')}"
    output_root = args.output_root or root_path("results_root") / "evals" / "shard_manifests" / "geval_audio_v6_plausibility"
    run_root = output_root / run_group
    if run_root.exists() and not args.overwrite:
        raise SystemExit(f"error: run root already exists: {run_root}")
    run_root.mkdir(parents=True, exist_ok=True)

    workers: list[dict[str, Any]] = []
    collections: list[dict[str, Any]] = []
    dataset_config = read_json(args.dataset_config)
    dataset_root = Path(dataset_config["dataset_root"])

    for split in args.dataset_splits:
        source_rows = list(iter_jsonl(split_path(dataset_config, split)))
        for direction, stage in (("a2s", "t2s"), ("s2a", "t2a")):
            rows = [
                _manifest_row(
                    source=row,
                    ordinal=index,
                    dataset_root=dataset_root,
                    stage=stage,
                    direction=direction,
                    skip_missing=False,
                )
                for index, row in enumerate(source_rows)
            ]
            collections.append(
                write_shards(
                    run_root=run_root,
                    run_group=run_group,
                    source_scope="dataset",
                    split=split,
                    direction=direction,
                    stage=stage,
                    rows=[row for row in rows if row is not None],
                    shard_size=args.dataset_shard_size,
                    workers=workers,
                    inspect_samples_by_id=None,
                    source_manifest=str(split_path(dataset_config, split)),
                )
            )

    if args.include_cascaded_test:
        for direction, stage, manifest in (
            ("a2s", "t2s", args.cascaded_a2s_manifest),
            ("s2a", "t2a", args.cascaded_s2a_manifest),
        ):
            rows = [rewrite_cascaded_output_paths(row, manifest.parent) for row in iter_jsonl(manifest) if row.get("status") == "success"]
            collections.append(
                write_shards(
                    run_root=run_root,
                    run_group=run_group,
                    source_scope="cascaded",
                    split="test",
                    direction=direction,
                    stage=stage,
                    rows=rows,
                    shard_size=args.cascaded_shard_size,
                    workers=workers,
                    inspect_samples_by_id=inspect_samples_by_id(manifest.with_name("inspect.json")),
                    source_manifest=str(manifest),
                )
            )

    tasks = pack_tasks(workers, workers_per_task=args.workers_per_task)
    plan = {
        "schema_version": 1,
        "run_group": run_group,
        "metric": METRIC,
        "prompt_version_expected": "geval_audio_plausibility_v6",
        "created_at": datetime.now().astimezone().isoformat(),
        "repo_root": str(REPO_ROOT),
        "run_root": str(run_root),
        "dataset_shard_size": args.dataset_shard_size,
        "cascaded_shard_size": args.cascaded_shard_size,
        "workers_per_task": args.workers_per_task,
        "worker_count": len(workers),
        "task_count": len(tasks),
        "collections": collections,
        "tasks": tasks,
    }
    plan_path = run_root / "task_plan.json"
    write_json(plan_path, plan)
    write_jsonl(run_root / "workers.jsonl", workers)
    write_jsonl(run_root / "tasks.jsonl", tasks)
    env_path = run_root / "run_env.sh"
    env_path.write_text(
        "\n".join(
            [
                f"export RUN_GROUP={shell_quote(run_group)}",
                f"export TASK_PLAN={shell_quote(str(plan_path))}",
                f"export SHARD_MANIFEST_ROOT={shell_quote(str(run_root))}",
                f"export TASK_COUNT={len(tasks)}",
                "",
            ]
        ),
        encoding="utf-8",
    )
    print(f"run_group={run_group}")
    print(f"run_root={run_root}")
    print(f"task_plan={plan_path}")
    print(f"task_count={len(tasks)}")
    print(f"worker_count={len(workers)}")
    print(f"env_file={env_path}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Prepare canonical shard manifests for v6 audio GEval plausibility runs.")
    parser.add_argument("--run-group")
    parser.add_argument("--output-root", type=Path)
    parser.add_argument("--dataset-config", type=Path, default=Path("configs/ttas_dataset.json"))
    parser.add_argument("--dataset-splits", nargs="+", default=["train", "test", "val"], choices=["train", "val", "test"])
    parser.add_argument("--include-cascaded-test", action="store_true", default=True)
    parser.add_argument("--no-cascaded-test", action="store_false", dest="include_cascaded_test")
    parser.add_argument("--cascaded-a2s-manifest", type=Path, default=DEFAULT_CASCADED_A2S)
    parser.add_argument("--cascaded-s2a-manifest", type=Path, default=DEFAULT_CASCADED_S2A)
    parser.add_argument("--dataset-shard-size", type=int, default=456)
    parser.add_argument("--cascaded-shard-size", type=int, default=228)
    parser.add_argument("--workers-per-task", type=int, default=4)
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def iter_jsonl(path: Path) -> Iterable[dict[str, Any]]:
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                yield json.loads(line)


def split_path(dataset_config: dict[str, Any], split: str) -> Path:
    path = Path(dataset_config["splits"][split])
    return path if path.is_absolute() else REPO_ROOT / path


def write_shards(
    *,
    run_root: Path,
    run_group: str,
    source_scope: str,
    split: str,
    direction: str,
    stage: str,
    rows: list[dict[str, Any]],
    shard_size: int,
    workers: list[dict[str, Any]],
    inspect_samples_by_id: dict[str, dict[str, Any]] | None,
    source_manifest: str,
) -> dict[str, Any]:
    if shard_size <= 0:
        raise ValueError("shard_size must be positive")
    shard_count = math.ceil(len(rows) / shard_size)
    collection_root = run_root / source_scope / split / direction
    missing_inspect = 0
    for shard_index in range(shard_count):
        start = shard_index * shard_size
        shard_rows = rows[start:start + shard_size]
        shard_dir = collection_root / f"shard_{shard_index:03d}"
        shard_dir.mkdir(parents=True, exist_ok=True)
        manifest_path = shard_dir / "manifest.jsonl"
        write_jsonl(manifest_path, shard_rows)
        if inspect_samples_by_id is not None:
            shard_samples = []
            for row in shard_rows:
                sample_id = str(row.get("sample_id"))
                sample = inspect_samples_by_id.get(sample_id)
                if sample is None:
                    missing_inspect += 1
                else:
                    shard_samples.append(sample)
            write_json(shard_dir / "inspect.json", {"samples": shard_samples})
        workers.append(
            {
                "worker_index": len(workers),
                "label": f"{source_scope}_{split}_{direction}_shard_{shard_index:03d}",
                "run_group": run_group,
                "source_scope": source_scope,
                "split": split,
                "direction": direction,
                "stage": stage,
                "model": MODEL,
                "dataset": DATASET,
                "metric": METRIC,
                "shard_index": shard_index,
                "shard_start": start,
                "row_count": len(shard_rows),
                "manifest": str(manifest_path),
                "evaluated_run_id": f"{run_group}_{source_scope}_{split}_{direction}_shard_{shard_index:03d}",
            }
        )
    if missing_inspect:
        raise RuntimeError(f"missing {missing_inspect} inspect samples for {source_scope}/{split}/{direction}")
    return {
        "source_scope": source_scope,
        "split": split,
        "direction": direction,
        "stage": stage,
        "source_manifest": source_manifest,
        "row_count": len(rows),
        "shard_size": shard_size,
        "shard_count": shard_count,
        "root": str(collection_root),
    }


def pack_tasks(workers: list[dict[str, Any]], *, workers_per_task: int) -> list[dict[str, Any]]:
    if workers_per_task <= 0:
        raise ValueError("workers_per_task must be positive")
    tasks: list[dict[str, Any]] = []
    for start in range(0, len(workers), workers_per_task):
        packed = []
        for gpu, worker in enumerate(workers[start:start + workers_per_task]):
            item = dict(worker)
            item["gpu"] = gpu
            packed.append(item)
        tasks.append({"task_index": len(tasks), "workers": packed})
    return tasks


def rewrite_cascaded_output_paths(row: dict[str, Any], manifest_dir: Path) -> dict[str, Any]:
    copied = json.loads(json.dumps(row))
    output = copied.get("output")
    if isinstance(output, dict):
        for key in ("audio_file", "speech_file", "file"):
            value = output.get(key)
            if isinstance(value, str) and value and not Path(value).is_absolute():
                output[key] = str(manifest_dir / value)
    return copied


def inspect_samples_by_id(path: Path) -> dict[str, dict[str, Any]]:
    doc = read_json(path)
    samples = doc.get("samples")
    if not isinstance(samples, list):
        raise ValueError(f"inspect file has no samples list: {path}")
    result: dict[str, dict[str, Any]] = {}
    for sample in samples:
        if isinstance(sample, dict) and isinstance(sample.get("sample_id"), str):
            result[sample["sample_id"]] = sample
    return result


def write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, ensure_ascii=False, sort_keys=True) + "\n", encoding="utf-8")


def write_jsonl(path: Path, rows: Iterable[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")


def shell_quote(value: str) -> str:
    return "'" + value.replace("'", "'\\''") + "'"


if __name__ == "__main__":
    main()
