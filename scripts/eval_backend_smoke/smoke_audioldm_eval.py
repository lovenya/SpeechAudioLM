#!/usr/bin/env python
"""Smoke-check audioldm_eval for FAD/Inception Score."""

from __future__ import annotations

import argparse
import json


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Smoke test audioldm_eval.")
    parser.add_argument("--reference-dir", default=None)
    parser.add_argument("--generated-dir", default=None)
    parser.add_argument("--sample-rate", type=int, default=16000)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--run", action="store_true", help="Run EvaluationHelper.main on the two folders.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    from audioldm_eval import EvaluationHelper
    import torch

    result = {
        "backend": "audioldm_eval",
        "sample_rate": args.sample_rate,
        "device": args.device,
        "helper_imported": True,
        "metrics": None,
    }
    if args.run:
        if not args.reference_dir or not args.generated_dir:
            raise SystemExit("--run requires --reference-dir and --generated-dir")
        evaluator = EvaluationHelper(args.sample_rate, torch.device(args.device))
        result["metrics"] = evaluator.main(args.generated_dir, args.reference_dir, limit_num=2)
    print(json.dumps(result, indent=2, sort_keys=True, default=str))


if __name__ == "__main__":
    main()
