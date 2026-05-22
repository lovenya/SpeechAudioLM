#!/usr/bin/env python
"""Smoke-check audioldm_eval for FAD/Inception Score."""

from __future__ import annotations

import argparse
import json
import os


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Smoke test audioldm_eval.")
    parser.add_argument("--cache-dir", default="/scratch/lovenya/SpeechAudioLM/cache/metrics/audioldm_eval")
    parser.add_argument("--reference-dir", default=None)
    parser.add_argument("--generated-dir", default=None)
    parser.add_argument("--sample-rate", type=int, default=16000)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--run", action="store_true", help="Run EvaluationHelper.main on the two folders.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    cache_dir = args.cache_dir
    os.environ.setdefault("XDG_CACHE_HOME", cache_dir)
    os.environ.setdefault("NUMBA_CACHE_DIR", os.path.join(os.path.dirname(cache_dir), "numba"))
    os.environ.setdefault("MPLCONFIGDIR", os.path.join(os.path.dirname(cache_dir), "matplotlib"))
    os.makedirs(os.environ["NUMBA_CACHE_DIR"], exist_ok=True)
    os.makedirs(os.environ["MPLCONFIGDIR"], exist_ok=True)
    _disable_numba_function_cache()

    from audioldm_eval import EvaluationHelper
    import torch

    result = {
        "backend": "audioldm_eval",
        "cache_dir": cache_dir,
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


def _disable_numba_function_cache() -> None:
    try:
        import numba
        import numba.np.ufunc.decorators as ufunc_decorators
    except Exception:
        return

    def patch_decorator(decorator):
        def wrapper(*args, **kwargs):
            kwargs["cache"] = False
            return decorator(*args, **kwargs)

        return wrapper

    numba.jit = patch_decorator(numba.jit)
    numba.vectorize = patch_decorator(numba.vectorize)
    numba.guvectorize = patch_decorator(numba.guvectorize)
    ufunc_decorators.vectorize = patch_decorator(ufunc_decorators.vectorize)
    ufunc_decorators.guvectorize = patch_decorator(ufunc_decorators.guvectorize)


if __name__ == "__main__":
    main()
