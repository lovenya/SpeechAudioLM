#!/usr/bin/env python
"""Smoke-check UTMOS through SpeechMOS torch.hub."""

from __future__ import annotations

import argparse
import json
import os


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Smoke test UTMOS/SpeechMOS.")
    parser.add_argument("--cache-dir", default="/project/rrg-csubakan/lovenya/models/metrics/torchhub")
    parser.add_argument("--load-model", action="store_true", help="Load UTMOS22 strong. Use compute node if needed.")
    parser.add_argument("--sample-rate", type=int, default=16000)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    os.environ.setdefault("TORCH_HOME", args.cache_dir)

    import torch

    result = {
        "backend": "SpeechMOS/UTMOS",
        "cache_dir": args.cache_dir,
        "torch_version": torch.__version__,
        "loaded": False,
        "score_shape": None,
    }
    if args.load_model:
        predictor = torch.hub.load("tarepan/SpeechMOS:v1.2.0", "utmos22_strong", trust_repo=True)
        wave = torch.zeros((1, args.sample_rate), dtype=torch.float32)
        score = predictor(wave, sr=args.sample_rate)
        result["loaded"] = True
        result["score_shape"] = list(score.shape)
        result["score_preview"] = [float(item) for item in score.detach().cpu().flatten()[:3]]
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
