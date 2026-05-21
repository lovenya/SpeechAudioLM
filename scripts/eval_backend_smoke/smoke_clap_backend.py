#!/usr/bin/env python
"""Report available CLAP backend packages for AAC and T2A metrics."""

from __future__ import annotations

import importlib.util
import json


def main() -> None:
    candidates = {
        "aac_metrics": importlib.util.find_spec("aac_metrics") is not None,
        "msclap": importlib.util.find_spec("msclap") is not None,
        "laion_clap": importlib.util.find_spec("laion_clap") is not None,
        "stable_audio_metrics": importlib.util.find_spec("stable_audio_metrics") is not None,
    }
    result = {
        "backend": "split-by-task",
        "metric_backends": {
            "aac_clapsim": "aac-metrics",
            "clap_score_t2a": "stable-audio-metrics",
        },
        "installed_candidates": candidates,
        "note": "AAC CLAPSim and T2A CLAPscore are intentionally separate metrics with separate backends.",
    }
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
