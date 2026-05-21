#!/usr/bin/env python
"""Report available CLAP candidates without choosing a backend."""

from __future__ import annotations

import importlib.util
import json


def main() -> None:
    candidates = {
        "aac_metrics": importlib.util.find_spec("aac_metrics") is not None,
        "msclap": importlib.util.find_spec("msclap") is not None,
        "laion_clap": importlib.util.find_spec("laion_clap") is not None,
    }
    result = {
        "backend": "clap-decision-pending",
        "installed_candidates": candidates,
        "note": "Do not implement CLAP until one backend/checkpoint is selected for both reporting and reproducibility.",
    }
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
