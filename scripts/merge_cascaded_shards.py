#!/usr/bin/env python
"""Merge cascaded stage shard directories into one canonical lane directory."""

from __future__ import annotations

import argparse
import json
import shutil
from pathlib import Path
from typing import Any

from baselines.cascaded_manifest import unwrap_sample, write_inspection_manifest


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-dir", action="append", required=True, help="Shard lane directory containing manifest.jsonl.")
    parser.add_argument("--output-dir", required=True, help="Canonical lane directory to write.")
    parser.add_argument("--lane", required=True, help="Canonical lane name to store in wrapped row metadata.")
    parser.add_argument("--expected-count", type=int, required=True)
    parser.add_argument("--artifact-subdir", choices=["text", "audio"], required=True)
    parser.add_argument("--backup-label", required=True)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    output_dir = Path(args.output_dir)
    output_manifest = output_dir / "manifest.jsonl"
    output_inspection = output_dir / "manifest.inspect.json"
    rows: dict[int, dict[str, Any]] = {}
    samples: dict[int, dict[str, Any]] = {}
    metadata: dict[str, Any] | None = None
    definitions: list[dict[str, Any]] | None = None
    created_at: str | None = None
    started_at: str | None = None
    completed_at: str | None = None
    copied = 0
    staging_root = output_dir / f".merge_staging_{args.backup_label}"
    staging_artifacts = staging_root / args.artifact_subdir
    if staging_root.exists():
        raise SystemExit(f"staging path exists {staging_root}")

    for input_value in args.input_dir:
        lane_dir = Path(input_value)
        manifest = lane_dir / "manifest.jsonl"
        inspection_path = lane_dir / "manifest.inspect.json"
        if not manifest.is_file():
            raise SystemExit(f"missing shard manifest {manifest}")
        if not inspection_path.is_file():
            raise SystemExit(f"missing shard inspection manifest {inspection_path}")
        inspection = json.loads(inspection_path.read_text(encoding="utf-8"))
        current_metadata = {key: inspection.get(key) for key in ("pipeline", "dataset", "split", "config_file")}
        current_definitions = inspection.get("stage_definitions") or []
        if metadata is None:
            metadata = current_metadata
            definitions = current_definitions
            created_at = inspection.get("created_at")
            started_at = inspection.get("started_at")
        else:
            if current_metadata != metadata or not _compatible_stage_definitions(current_definitions, definitions or []):
                raise SystemExit(f"incompatible shard metadata: {inspection_path}")
            definitions = _merge_stage_definitions(definitions or [], current_definitions)
        completed_at = inspection.get("completed_at") or completed_at

        for artifact in sorted((lane_dir / args.artifact_subdir).glob("*")):
            if not artifact.is_file():
                continue
            target = staging_artifacts / artifact.name
            target.parent.mkdir(parents=True, exist_ok=True)
            if target.exists():
                raise SystemExit(f"duplicate artifact name {artifact.name}")
            shutil.copy2(artifact, target)
            copied += 1

        with manifest.open("r", encoding="utf-8") as handle:
            for line_number, line in enumerate(handle, start=1):
                if not line.strip():
                    continue
                row = json.loads(line)
                if not isinstance(row.get("sample"), dict):
                    raise SystemExit(f"legacy row is not supported in cascaded shard merge: {manifest}:{line_number}")
                sample = unwrap_sample(row)
                row_index = int(sample["row_index"])
                if row_index in rows:
                    raise SystemExit(f"duplicate row_index={row_index}")
                row["lane"] = args.lane
                output = sample.get("final_output") or (sample.get("stages") or [{}])[-1].get("output")
                if isinstance(output, dict) and isinstance(output.get("file"), str):
                    new_path = str(output_dir / args.artifact_subdir / Path(output["file"]).name)
                    output["file"] = new_path
                    final_output = sample.get("final_output")
                    if isinstance(final_output, dict):
                        final_output["file"] = new_path
                    stages = sample.get("stages")
                    if isinstance(stages, list) and stages and isinstance(stages[-1].get("output"), dict):
                        stages[-1]["output"]["file"] = new_path
                rows[row_index] = row
                samples[row_index] = sample

    missing = [index for index in range(args.expected_count) if index not in rows]
    if missing:
        raise SystemExit(f"missing rows {missing[:20]} count={len(missing)}")
    output_dir.mkdir(parents=True, exist_ok=True)
    if args.artifact_subdir == "text":
        expected_artifacts = sum(sample.get("status") == "success" for sample in samples.values())
    else:
        expected_artifacts = sum(
            sample.get("status") == "success"
            and isinstance(sample.get("final_output"), dict)
            and isinstance(sample["final_output"].get("file"), str)
            for sample in samples.values()
        )
    if copied != expected_artifacts:
        raise SystemExit(f"copied artifacts={copied}; expected successful artifacts={expected_artifacts}")
    canonical_artifacts = output_dir / args.artifact_subdir
    backup_artifacts = output_dir / f"{args.artifact_subdir}.pre_{args.backup_label}"
    if backup_artifacts.exists():
        raise SystemExit(f"backup exists {backup_artifacts}")
    _backup(output_manifest, args.backup_label)
    _backup(output_inspection, args.backup_label)
    if canonical_artifacts.exists():
        canonical_artifacts.rename(backup_artifacts)
    staging_artifacts.rename(canonical_artifacts)
    staging_root.rmdir()
    merged_rows = [rows[index] for index in range(args.expected_count)]
    output_manifest.write_text("".join(json.dumps(row, ensure_ascii=False) + "\n" for row in merged_rows), encoding="utf-8")
    if metadata is None or definitions is None:
        raise SystemExit("no shard metadata found")
    write_inspection_manifest(
        output_inspection,
        metadata={"lane": args.lane, **metadata},
        stage_definitions=definitions,
        samples=[samples[index] for index in range(args.expected_count)],
        created_at=created_at or "",
        started_at=started_at or "",
        completed_at=completed_at or "",
        status="completed" if all(samples[index].get("status") == "success" for index in range(args.expected_count)) else "completed_with_errors",
    )
    ok = sum(samples[index].get("status") == "success" for index in range(args.expected_count))
    print(f"canonicalized {args.lane}: rows={len(merged_rows)} ok={ok} err={len(merged_rows)-ok} copied_artifacts={copied}")


def _backup(path: Path, label: str) -> None:
    if not path.exists():
        return
    backup = path.with_name(f"{path.name}.pre_{label}")
    if backup.exists():
        raise SystemExit(f"backup exists {backup}")
    shutil.copy2(path, backup)


def _static_stage_definition(definition: dict[str, Any]) -> dict[str, Any]:
    static = dict(definition)
    runtime = dict(static.get("runtime") or {})
    runtime.pop("model_load_seconds", None)
    static["runtime"] = runtime
    return static


def _compatible_stage_definitions(left: list[dict[str, Any]], right: list[dict[str, Any]]) -> bool:
    return [_static_stage_definition(item) for item in left] == [_static_stage_definition(item) for item in right]


def _merge_stage_definitions(left: list[dict[str, Any]], right: list[dict[str, Any]]) -> list[dict[str, Any]]:
    merged = []
    for first, second in zip(left, right, strict=True):
        item = dict(first)
        runtime = dict(item.get("runtime") or {})
        values = [
            value
            for value in ((first.get("runtime") or {}).get("model_load_seconds"), (second.get("runtime") or {}).get("model_load_seconds"))
            if isinstance(value, (int, float))
        ]
        runtime["model_load_seconds"] = max(values) if values else None
        item["runtime"] = runtime
        merged.append(item)
    return merged


if __name__ == "__main__":
    main()
