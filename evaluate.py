#!/usr/bin/env python
"""Evaluate baseline outputs from a manifest."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from evaluation.capabilities import load_capabilities, validate_model_task
from evaluation.errors import EvaluationError
from evaluation.manifest import resolve_manifest_path
from evaluation.metric_contracts import EvaluationMode
from evaluation.metrics import get_metric_specs
from evaluation.runner import EvalOptions, dry_run, run_evaluation
from evaluation.task_metrics import load_task_metric_map, select_metrics
from project_paths import result_path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate SpeechAudio LM baseline outputs.")
    parser.add_argument("--model", required=True, help="Model/backend name, e.g. audio_omni.")
    parser.add_argument("--task", required=True, help="Task name, e.g. T2A, T2S, S2A, or A2S.")
    parser.add_argument("--dataset", default="ttas", help="Dataset name. Defaults to ttas.")
    parser.add_argument("--split", default="test", help="Dataset split. Defaults to test.")
    parser.add_argument("--manifest", default=None, help="Optional explicit baseline manifest path.")
    parser.add_argument("--metrics", default=None, help="Comma-separated metrics. Defaults depend on --task.")
    parser.add_argument("--dataset-config", default="configs/ttas_dataset.json")
    parser.add_argument("--metrics-config", default="configs/eval_metrics.json")
    parser.add_argument("--task-metric-map", default="configs/task_metric_map.json")
    parser.add_argument("--capabilities-config", default="configs/model_capabilities.json")
    parser.add_argument("--baseline-root", default=str(result_path("baselines")))
    parser.add_argument("--output-root", default=str(result_path("evaluations")))
    parser.add_argument("--cache-dir", default=str(result_path("evaluation_cache")))
    parser.add_argument("--device", default="cuda")
    parser.add_argument(
        "--evaluation-mode",
        choices=[mode.value for mode in EvaluationMode],
        default=EvaluationMode.FINAL.value,
        help="Final scientific evaluation or non-authoritative backend smoke test.",
    )
    parser.add_argument(
        "--common-sample-ids",
        default=None,
        help="Path to an explicit JSON list/object or newline-delimited common sample-ID file.",
    )
    parser.add_argument("--start-index", type=int, default=0, help="Skip this many manifest rows before evaluation.")
    parser.add_argument("--max-samples", type=int, default=None, help="Limit rows for debugging.")
    parser.add_argument("--dry-run", action="store_true", help="Validate inputs and print planned evaluation.")
    return parser.parse_args()


def _load_common_sample_ids(path_text: str | None) -> tuple[str, ...] | None:
    if path_text is None:
        return None
    path = Path(path_text)
    if not path.is_file():
        raise EvaluationError(f"Common sample-ID file does not exist: {path}")
    text = path.read_text(encoding="utf-8")
    try:
        value = json.loads(text)
    except json.JSONDecodeError:
        values = [line.strip() for line in text.splitlines() if line.strip()]
    else:
        if isinstance(value, dict):
            value = value.get("common_sample_ids", value.get("sample_ids"))
        if not isinstance(value, list):
            raise EvaluationError(
                "Common sample-ID JSON must be a list or an object containing "
                "'common_sample_ids' or 'sample_ids'."
            )
        values = value
    if not values or any(not isinstance(value, str) or not value.strip() for value in values):
        raise EvaluationError("Common sample IDs must be a non-empty list of non-empty strings.")
    if len(values) != len(set(values)):
        raise EvaluationError("Common sample IDs must not contain duplicates.")
    return tuple(values)


def main() -> None:
    args = parse_args()
    try:
        capabilities = load_capabilities(Path(args.capabilities_config))
        validate_model_task(args.model, args.task, capabilities)
        task_metric_map = load_task_metric_map(Path(args.task_metric_map))
        selection = select_metrics(args.task, args.metrics, task_metric_map)
        manifest_path = resolve_manifest_path(
            model=args.model,
            task=args.task,
            dataset=args.dataset,
            split=args.split,
            explicit_manifest=args.manifest,
            baseline_root=Path(args.baseline_root),
        )
        metric_specs = get_metric_specs(selection.metric_names)
        options = EvalOptions(
            model=args.model,
            task=args.task,
            dataset=args.dataset,
            split=args.split,
            manifest_path=manifest_path,
            dataset_config_path=Path(args.dataset_config),
            metrics_config_path=Path(args.metrics_config),
            task_metric_map_path=Path(args.task_metric_map),
            output_root=Path(args.output_root),
            metric_specs=metric_specs,
            max_samples=args.max_samples,
            start_index=args.start_index,
            cache_dir=Path(args.cache_dir),
            device=args.device,
            command=sys.argv,
            evaluation_mode=EvaluationMode(args.evaluation_mode),
            common_sample_ids=_load_common_sample_ids(args.common_sample_ids),
            common_sample_ids_path=(Path(args.common_sample_ids) if args.common_sample_ids else None),
        )
        if args.dry_run:
            print(json.dumps(dry_run(options), indent=2, ensure_ascii=False, sort_keys=True))
            return
        paths = run_evaluation(options)
    except EvaluationError as exc:
        raise SystemExit(f"error: {exc}") from None

    for name, path in paths.items():
        print(f"{name}={path}")


if __name__ == "__main__":
    main()
