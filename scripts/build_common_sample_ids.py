#!/usr/bin/env python
from __future__ import annotations

import argparse
from pathlib import Path

from evaluation.common_ids import CommonIdError, write_common_successful_ids


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Create an immutable common successful sample-ID artifact."
    )
    parser.add_argument("--manifest", action="append", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    try:
        artifact = write_common_successful_ids(args.manifest, args.output)
    except (CommonIdError, OSError, ValueError) as exc:
        raise SystemExit(f"error: {exc}") from None
    print(f"output={args.output}")
    print(f"common_sample_count={artifact['common_sample_count']}")
    print(f"common_sample_id_set_sha256={artifact['common_sample_id_set_sha256']}")


if __name__ == "__main__":
    main()
