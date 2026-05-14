from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Iterable

from baselines.dataset import PromptRow
from baselines.models.base import BaselineRunner


def run_inference(
    runner: BaselineRunner,
    rows: Iterable[PromptRow],
    output_root: Path,
    overwrite: bool,
    run_config: dict,
) -> None:
    output_root.mkdir(parents=True, exist_ok=True)
    audio_dir = output_root / "audio"
    audio_dir.mkdir(parents=True, exist_ok=True)
    manifest_path = output_root / "manifest.jsonl"

    print(f"output_root={output_root}")
    print("loading model...")
    load_start = time.time()
    runner.load()
    print(f"model_loaded_seconds={time.time() - load_start:.2f}")

    with manifest_path.open("a", encoding="utf-8") as manifest:
        for row in rows:
            output_path = audio_dir / f"{row.row_index:06d}_{row.sample_id}{runner.extension}"
            if output_path.exists() and not overwrite:
                print(f"skip existing {output_path}")
                continue

            print(f"generate row={row.row_index} sample_id={row.sample_id}")
            start = time.time()
            result = runner.generate(row.prompt, output_path)
            elapsed = time.time() - start
            record = {
                **run_config,
                "row_index": row.row_index,
                "sample_id": row.sample_id,
                "prompt": row.prompt,
                "output_file": str(result.output_path),
                "sample_rate": result.sample_rate,
                "generation_seconds": round(elapsed, 3),
                "source": row.record,
                "model_extra": result.extra,
            }
            manifest.write(json.dumps(record, ensure_ascii=False) + "\n")
            manifest.flush()
            print(f"wrote {result.output_path} seconds={elapsed:.2f}")

    print(f"manifest={manifest_path}")
