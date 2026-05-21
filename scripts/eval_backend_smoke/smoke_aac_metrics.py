#!/usr/bin/env python
"""Smoke-check aac-metrics for SPIDEr/MACE."""

from __future__ import annotations

import argparse
import json


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Smoke test aac-metrics.")
    parser.add_argument("--cache-dir", default="/project/rrg-csubakan/lovenya/cache/metrics/aac_metrics")
    parser.add_argument("--metrics", default="spider", help="Comma-separated AAC metrics for a tiny text-only run.")
    parser.add_argument("--device", default="cpu")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    from aac_metrics import evaluate

    metrics = [item.strip() for item in args.metrics.split(",") if item.strip()]
    candidates = ["rain is falling on a street"]
    references = [["rain falls on pavement"]]
    corpus_scores, sample_scores = evaluate(
        candidates,
        references,
        metrics=metrics,
        cache_path=args.cache_dir,
        device=args.device,
    )
    result = {
        "backend": "aac-metrics",
        "metrics": metrics,
        "cache_dir": args.cache_dir,
        "corpus_score_keys": sorted(corpus_scores.keys()),
        "sample_score_keys": sorted(sample_scores.keys()),
    }
    print(json.dumps(result, indent=2, sort_keys=True, default=str))


if __name__ == "__main__":
    main()
