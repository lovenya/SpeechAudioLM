#!/usr/bin/env python
"""Smoke-check UTMOSv2 import/model availability."""

from __future__ import annotations

import argparse
import json
import os


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Smoke test UTMOSv2.")
    parser.add_argument("--cache-dir", default="/project/rrg-csubakan/lovenya/cache/metrics/utmosv2")
    parser.add_argument("--load-model", action="store_true", help="Reserved for a later UTMOSv2 model-load smoke.")
    parser.add_argument("--sample-rate", type=int, default=16000)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    os.environ.setdefault("XDG_CACHE_HOME", args.cache_dir)

    import importlib.util

    result = {
        "backend": "UTMOSv2",
        "cache_dir": args.cache_dir,
        "package_importable": importlib.util.find_spec("utmosv2") is not None,
        "loaded": False,
        "score_shape": None,
    }
    if args.load_model:
        raise SystemExit("UTMOSv2 model-load smoke is not wired yet; first verify the package API after install.")
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
