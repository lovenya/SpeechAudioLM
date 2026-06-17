#!/usr/bin/env python
"""Evaluate baseline outputs from a manifest."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from evaluation.capabilities import load_capabilities, validate_model_task
from evaluation.errors import EvaluationError
from evaluation.io import utc_timestamp
from evaluation.manifest import resolve_manifest_path
from evaluation.metrics import get_metric_specs
from evaluation.runner import EvalOptions, dry_run, resolve_stage_direction, run_evaluation
from evaluation.task_metrics import load_task_metric_map, select_metrics
from project_conventions import DEFAULT_EVAL_ROOT
from project_paths import result_path, root_path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate SpeechAudio LM baseline outputs.")
    parser.add_argument("--model", required=True, help="Evaluated model/backend name, e.g. audio_omni.")
    parser.add_argument(
        "--stage",
        required=True,
        choices=["a2t", "s2t", "t2t", "t2a", "t2s", "s2a", "a2s"],
        help="Evaluated stage/task: a2t, s2t, t2t, t2a, t2s, s2a, or a2s.",
    )
    parser.add_argument("--direction", choices=["a2s", "s2a"], help="Pipeline direction. Required for --stage t2t; otherwise inferred and validated when supplied.")
    parser.add_argument("--evaluated-run-id", required=True, help="Run ID of the result being evaluated.")
    parser.add_argument("--dataset", default="ttas", help="Dataset name. Defaults to ttas.")
    parser.add_argument("--split", default="test", help="Dataset split. Defaults to test.")
    parser.add_argument("--manifest", default=None, help="Optional explicit baseline manifest path.")
    parser.add_argument("--metrics", required=True, help='Comma-separated metrics, or "all".')
    parser.add_argument("--dataset-config", default="configs/ttas_dataset.json")
    parser.add_argument("--metrics-config", default="configs/eval_metrics.json")
    parser.add_argument("--task-metric-map", default="configs/task_metric_map.json")
    parser.add_argument("--capabilities-config", default="configs/model_capabilities.json")
    parser.add_argument("--baseline-root", default=str(result_path("baselines")))
    parser.add_argument("--output-root", default=str(DEFAULT_EVAL_ROOT))
    parser.add_argument("--temporary", action="store_true", help="Write outputs under temporary results roots.")
    parser.add_argument("--cache-dir", default=str(result_path("evaluation_cache")))
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--start-index", type=int, default=0, help="Skip this many manifest rows before evaluation.")
    parser.add_argument("--max-samples", type=int, default=None, help="Limit rows for debugging.")
    parser.add_argument("--dry-run", action="store_true", help="Validate inputs and print planned evaluation.")
    return parser.parse_args()



def main() -> None:
    args = parse_args()
    try:
        task = args.stage.upper()
        direction = resolve_stage_direction(args.stage, args.direction)
        capabilities = load_capabilities(Path(args.capabilities_config))
        validate_model_task(args.model, task, capabilities)
        task_metric_map = load_task_metric_map(Path(args.task_metric_map))
        selection = select_metrics(task, args.metrics, task_metric_map)
        manifest_path = resolve_manifest_path(
            model=args.model,
            task=task,
            dataset=args.dataset,
            split=args.split,
            explicit_manifest=args.manifest,
            baseline_root=Path(args.baseline_root),
        )
        output_root = Path(args.output_root)
        if args.temporary:
            output_root = root_path("temporary_results_root") / "evals"
        metric_specs = get_metric_specs(selection.metric_names)
        eval_run_id = utc_timestamp()
        shared_options = dict(
            model=args.model,
            task=task,
            stage=args.stage,
            direction=direction,
            evaluated_run_id=args.evaluated_run_id,
            eval_run_id=eval_run_id,
            dataset=args.dataset,
            split=args.split,
            manifest_path=manifest_path,
            dataset_config_path=Path(args.dataset_config),
            metrics_config_path=Path(args.metrics_config),
            task_metric_map_path=Path(args.task_metric_map),
            output_root=output_root,
            max_samples=args.max_samples,
            start_index=args.start_index,
            cache_dir=Path(args.cache_dir),
            device=args.device,
            command=sys.argv,
            temporary=args.temporary,
        )
        if args.dry_run:
            dry_runs = []
            for metric_spec in metric_specs:
                options = EvalOptions(metric_specs=[metric_spec], **shared_options)
                dry_runs.append(dry_run(options))
            print(json.dumps(dry_runs[0] if len(dry_runs) == 1 else dry_runs, indent=2, ensure_ascii=False, sort_keys=True))
            return
        all_paths = []
        for metric_spec in metric_specs:
            options = EvalOptions(metric_specs=[metric_spec], **shared_options)
            all_paths.append((metric_spec.name, run_evaluation(options)))
    except EvaluationError as exc:
        raise SystemExit(f"error: {exc}") from None

    assert all_paths
    if len(all_paths) == 1:
        for name, path in all_paths[0][1].items():
            print(f"{name}={path}")
        return
    for metric_name, paths in all_paths:
        for name, path in paths.items():
            print(f"{metric_name}.{name}={path}")


if __name__ == "__main__":
    main()
