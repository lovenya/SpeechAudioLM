#!/usr/bin/env python
"""Smoke-check DNSMOS dependencies and cached ONNX assets."""

from __future__ import annotations

import argparse
import importlib
import inspect
import json
from pathlib import Path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Smoke test DNSMOS package/assets.")
    parser.add_argument("--cache-dir", default="/project/rrg-csubakan/lovenya/models/metrics/dnsmos")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    module = importlib.import_module("torchmetrics.functional.audio.dnsmos")
    expected_assets = [
        Path(args.cache_dir) / "DNSMOS" / "model_v8.onnx",
        Path(args.cache_dir) / "DNSMOS" / "sig_bak_ovr.onnx",
        Path(args.cache_dir) / "pDNSMOS" / "sig_bak_ovr.onnx",
    ]
    result = {
        "backend": "torchmetrics-dnsmos",
        "cache_dir": args.cache_dir,
        "module": module.__name__,
        "functions": [name for name in ("deep_noise_suppression_mean_opinion_score", "dnsmos") if hasattr(module, name)],
        "asset_status": {str(path): path.is_file() for path in expected_assets},
    }
    if hasattr(module, "deep_noise_suppression_mean_opinion_score"):
        result["signature"] = str(inspect.signature(module.deep_noise_suppression_mean_opinion_score))
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
