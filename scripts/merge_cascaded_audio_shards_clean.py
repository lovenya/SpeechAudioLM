#!/usr/bin/env python
"""Merge final-stage audio shards without retaining canonical backup clutter."""

from __future__ import annotations

import argparse
import json
import os
import shutil
import tempfile
from pathlib import Path
from typing import Any

from baselines.cascaded_manifest import unwrap_sample, write_inspection_manifest


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-dir", action="append", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--lane", required=True)
    parser.add_argument("--expected-count", type=int, required=True)
    parser.add_argument("--replace-existing", action="store_true")
    parser.add_argument(
        "--link-audio",
        action="store_true",
        help="Hard-link audio into the canonical directory instead of copying it.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    output_dir = Path(args.output_dir)
    rows: dict[int, dict[str, Any]] = {}
    samples: dict[int, dict[str, Any]] = {}
    inspections = []
    staging = output_dir.parent / f".{output_dir.name}.merge-{os.getpid()}"
    staging_audio = staging / "audio"
    staging_audio.mkdir(parents=True)

    try:
        copied = 0
        for value in args.input_dir:
            shard = Path(value)
            inspection_path = shard / "manifest.inspect.json"
            manifest_path = shard / "manifest.jsonl"
            if not inspection_path.is_file() or not manifest_path.is_file():
                raise RuntimeError(f"Incomplete shard directory: {shard}")
            inspections.append(json.loads(inspection_path.read_text(encoding="utf-8")))

            for artifact in sorted((shard / "audio").glob("*.wav")):
                target = staging_audio / artifact.name
                if target.exists():
                    raise RuntimeError(f"Duplicate audio artifact: {artifact.name}")
                if args.link_audio:
                    os.link(artifact, target)
                else:
                    shutil.copy2(artifact, target)
                copied += 1

            with manifest_path.open(encoding="utf-8") as handle:
                for line in handle:
                    if not line.strip():
                        continue
                    row = json.loads(line)
                    sample = unwrap_sample(row)
                    index = int(sample["row_index"])
                    if index in rows:
                        raise RuntimeError(f"Duplicate row_index: {index}")
                    row["lane"] = args.lane
                    output = sample.get("final_output")
                    if isinstance(output, dict) and isinstance(output.get("file"), str):
                        path = str(output_dir / "audio" / Path(output["file"]).name)
                        output["file"] = path
                        sample["stages"][-1]["output"]["file"] = path
                    rows[index] = row
                    samples[index] = sample

        if len(rows) != args.expected_count:
            raise RuntimeError(f"Merged rows={len(rows)} expected={args.expected_count}")
        successful = sum(sample.get("status") == "success" for sample in samples.values())
        if copied != successful:
            raise RuntimeError(f"Copied WAVs={copied} successful rows={successful}")

        ordered = sorted(rows)
        metadata = {
            key: inspections[0].get(key)
            for key in ("pipeline", "dataset", "split", "config_file")
        }
        definitions = _merge_definitions(inspections)
        manifest_path = staging / "manifest.jsonl"
        manifest_path.write_text(
            "".join(json.dumps(rows[index], ensure_ascii=False) + "\n" for index in ordered),
            encoding="utf-8",
        )
        write_inspection_manifest(
            staging / "manifest.inspect.json",
            metadata={"lane": args.lane, **metadata},
            stage_definitions=definitions,
            samples=[samples[index] for index in ordered],
            created_at=inspections[0].get("created_at") or "",
            started_at=min(item.get("started_at") or "" for item in inspections),
            completed_at=max(item.get("completed_at") or "" for item in inspections),
            status="completed" if successful == len(rows) else "completed_with_errors",
        )

        if output_dir.exists():
            if not args.replace_existing:
                raise RuntimeError(f"Canonical output exists: {output_dir}")
            shutil.rmtree(output_dir)
        staging.rename(output_dir)
        print(
            f"canonicalized {args.lane}: rows={len(rows)} "
            f"success={successful} failed={len(rows) - successful} wavs={copied}"
        )
    except BaseException:
        shutil.rmtree(staging, ignore_errors=True)
        raise


def _merge_definitions(inspections: list[dict[str, Any]]) -> list[dict[str, Any]]:
    definitions = [dict(item) for item in inspections[0].get("stage_definitions") or []]
    for index, definition in enumerate(definitions):
        runtime = dict(definition.get("runtime") or {})
        loads = []
        for inspection in inspections:
            other = (inspection.get("stage_definitions") or [])[index]
            value = (other.get("runtime") or {}).get("model_load_seconds")
            if isinstance(value, (int, float)):
                loads.append(value)
        runtime["model_load_seconds"] = max(loads) if loads else None
        definition["runtime"] = runtime
    return definitions


if __name__ == "__main__":
    main()
