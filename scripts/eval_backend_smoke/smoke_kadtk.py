#!/usr/bin/env python
"""Smoke-check kadtk for KAD."""

from __future__ import annotations

import argparse
import json
import shutil
import subprocess
import sys


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Smoke test kadtk.")
    parser.add_argument("--model", default="panns-wavegram-logmel")
    parser.add_argument("--reference-dir", default=None)
    parser.add_argument("--generated-dir", default=None)
    parser.add_argument("--run", action="store_true", help="Run kadtk on the two folders.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    import kadtk  # noqa: F401

    exe = shutil.which("kadtk")
    result = {
        "backend": "kadtk",
        "cli": exe,
        "model": args.model,
        "ran": False,
        "stdout_preview": None,
    }
    if args.run:
        if not exe:
            raise SystemExit("kadtk CLI was not found on PATH")
        if not args.reference_dir or not args.generated_dir:
            raise SystemExit("--run requires --reference-dir and --generated-dir")
        completed = subprocess.run(
            [exe, args.model, args.reference_dir, args.generated_dir],
            check=True,
            text=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
        )
        result["ran"] = True
        result["stdout_preview"] = completed.stdout[:2000]
    json.dump(result, sys.stdout, indent=2, sort_keys=True)
    sys.stdout.write("\n")


if __name__ == "__main__":
    main()
