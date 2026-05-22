#!/usr/bin/env python
"""Smoke-check aac-metrics for SPIDEr/MACE."""

from __future__ import annotations

import argparse
import json
import os
import shutil


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Smoke test aac-metrics.")
    parser.add_argument("--cache-dir", default="/scratch/lovenya/SpeechAudioLM/cache/metrics/aac_metrics")
    parser.add_argument("--metrics", default="spider", help="Comma-separated AAC metrics for a tiny text-only run.")
    parser.add_argument("--device", default="cpu")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    from aac_metrics import evaluate

    java_tool_options = os.environ.pop("JAVA_TOOL_OPTIONS", None)
    java_path = shutil.which("java")
    if java_path is None:
        raise SystemExit("java was not found on PATH. Load java/11.0.22 before running AAC metrics.")
    metrics = [item.strip() for item in args.metrics.split(",") if item.strip()]
    candidates = [
        "rain is falling on a street",
        "a dog barks in a park",
    ]
    references = [
        ["rain falls on pavement"],
        ["a dog is barking outdoors"],
    ]
    try:
        corpus_scores, sample_scores = evaluate(
            candidates,
            references,
            metrics=metrics,
            cache_path=args.cache_dir,
            java_path=java_path,
            device=args.device,
        )
    finally:
        if java_tool_options is not None:
            os.environ["JAVA_TOOL_OPTIONS"] = java_tool_options
    result = {
        "backend": "aac-metrics",
        "metrics": metrics,
        "cache_dir": args.cache_dir,
        "java_path": java_path,
        "java_tool_options_removed": java_tool_options is not None,
        "corpus_score_keys": sorted(corpus_scores.keys()),
        "sample_score_keys": sorted(sample_scores.keys()),
    }
    print(json.dumps(result, indent=2, sort_keys=True, default=str))


if __name__ == "__main__":
    main()
