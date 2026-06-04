#!/usr/bin/env python
"""Build stage input split files from upstream cascaded manifests.

Generated split files should be written outside git, typically under
/scratch/l/lovenya/salm_migration_clean/results/cascaded_full_validation_metadata/ttas_val/stage_inputs/.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)

    stage1 = subparsers.add_parser("stage1-to-t2t")
    stage1.add_argument("--manifest", required=True)
    stage1.add_argument("--output", required=True)
    stage1.add_argument("--pipeline", required=True, choices=["S2A", "A2S"])
    stage1.add_argument("--upstream-backend", required=True)
    stage1.add_argument("--text-model", required=True)
    stage1.add_argument("--expected-count", type=int, required=True)
    stage1.add_argument("--allow-partial", action="store_true", help="Use successful rows only when upstream has errors.")

    final = subparsers.add_parser("t2t-to-final")
    final.add_argument("--manifest", required=True)
    final.add_argument("--output", required=True)
    final.add_argument("--pipeline", required=True, choices=["S2A", "A2S"])
    final.add_argument("--upstream-backend", required=True)
    final.add_argument("--text-model", required=True)
    final.add_argument("--final-backend", required=True)
    final.add_argument("--expected-count", type=int, required=True)
    final.add_argument("--allow-partial", action="store_true", help="Use successful rows only when upstream has errors.")

    report = subparsers.add_parser("report")
    report.add_argument("--root", required=True)
    report.add_argument("--output", required=True)
    report.add_argument("--expected-count", type=int, default=None)
    report.add_argument("--require-all-success", action="store_true")

    args = parser.parse_args()
    if args.command == "stage1-to-t2t":
        rows = _stage1_to_t2t(args)
        _write_jsonl(Path(args.output), rows)
        _require_count(args.output, rows, args.expected_count, allow_partial=args.allow_partial)
    elif args.command == "t2t-to-final":
        rows = _t2t_to_final(args)
        _write_jsonl(Path(args.output), rows)
        _require_count(args.output, rows, args.expected_count, allow_partial=args.allow_partial)
    elif args.command == "report":
        _write_report(
            Path(args.root),
            Path(args.output),
            expected_count=args.expected_count,
            require_all_success=args.require_all_success,
        )


def _stage1_to_t2t(args: argparse.Namespace) -> list[dict]:
    rows = []
    for index, record in enumerate(_successful_records(Path(args.manifest), allow_partial=args.allow_partial)):
        text = _required_text(record, "candidate_text")
        sample_id = str(record.get("sample_id") or f"row_{index}")
        stage_id = _safe_id(f"{args.pipeline}_{args.upstream_backend}_{args.text_model}_{sample_id}")
        rows.append(
            {
                "audio_file": _source_field(record, "audio_file", f"stage_inputs/{stage_id}.wav"),
                "speech_file": _source_field(record, "speech_file", f"stage_inputs/{stage_id}.wav"),
                "speech_text": text,
                "audio_cap": text,
                "stage_provenance": {
                    "stage": "t2t_input",
                    "pipeline": args.pipeline,
                    "upstream_backend": args.upstream_backend,
                    "text_model": args.text_model,
                    "upstream_manifest": args.manifest,
                    "upstream_sample_id": sample_id,
                    "upstream_output_file": record.get("output_file"),
                    "upstream_text": text,
                },
                "original_row": record.get("source", {}),
            }
        )
    return rows


def _t2t_to_final(args: argparse.Namespace) -> list[dict]:
    rows = []
    for index, record in enumerate(_successful_records(Path(args.manifest), allow_partial=args.allow_partial)):
        text = _required_text(record, "candidate_text")
        sample_id = str(record.get("sample_id") or f"row_{index}")
        stage_id = _safe_id(f"{args.pipeline}_{args.upstream_backend}_{args.text_model}_{args.final_backend}_{sample_id}")
        source = record.get("source") or {}
        original = _original_row_from_stage_source(source)
        row = {
            "audio_file": original.get("audio_file") if isinstance(original.get("audio_file"), str) else f"stage_inputs/{stage_id}.wav",
            "speech_file": original.get("speech_file") if isinstance(original.get("speech_file"), str) else f"stage_inputs/{stage_id}.wav",
            "speech_text": text,
            "audio_cap": text,
            "stage_provenance": {
                "stage": "final_input",
                "pipeline": args.pipeline,
                "upstream_backend": args.upstream_backend,
                "text_model": args.text_model,
                "final_backend": args.final_backend,
                "t2t_manifest": args.manifest,
                "t2t_sample_id": sample_id,
                "t2t_output_file": record.get("output_file"),
                "t2t_output_text": text,
                "t2t_input_text": source.get("speech_text") if isinstance(source, dict) else None,
                "upstream_stage_provenance": source.get("stage_provenance") if isinstance(source, dict) else None,
            },
            "original_row": original,
        }
        if args.pipeline == "A2S" and isinstance(original.get("speech_file"), str):
            row["speech_file"] = original["speech_file"]
            row["reference_speech_file"] = original["speech_file"]
        if args.pipeline == "A2S" and isinstance(original.get("speech_text"), str):
            row["reference_speech_text"] = original["speech_text"]
        rows.append(row)
    return rows



def _original_row_from_stage_source(source: dict) -> dict:
    original = source.get("original_row")
    return original if isinstance(original, dict) else source


def _source_field(record: dict, field: str, fallback: str) -> str:
    source = record.get("source")
    if isinstance(source, dict) and isinstance(source.get(field), str) and source[field].strip():
        return source[field]
    return fallback


def _successful_records(path: Path, *, allow_partial: bool = False) -> list[dict]:
    if not path.exists():
        raise FileNotFoundError(path)
    rows = []
    errors = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            record = json.loads(line)
            if record.get("status") == "success":
                rows.append(record)
            else:
                errors.append(record)
    if errors and not allow_partial:
        raise RuntimeError(f"{path} contains {len(errors)} non-success records")
    if errors:
        print(f"warning: {path} contains {len(errors)} non-success records; wrote successful rows only")
    return rows


def _required_text(record: dict, key: str) -> str:
    value = record.get(key)
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"Record row={record.get('row_index')} has no usable {key}")
    return value.strip()


def _write_jsonl(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")


def _require_count(path: str, rows: list[dict], expected: int, *, allow_partial: bool = False) -> None:
    if len(rows) != expected:
        message = f"{path} wrote {len(rows)} rows; expected {expected}"
        if not allow_partial:
            raise RuntimeError(message)
        print("warning: " + message)
    print(f"wrote {path} rows={len(rows)}")


def _write_report(root: Path, output: Path, expected_count: int | None, require_all_success: bool) -> None:
    manifests = sorted(root.glob("**/manifest.jsonl"))
    report = []
    failures = []
    for manifest in manifests:
        counts: dict[str, int] = {}
        seconds = []
        with manifest.open("r", encoding="utf-8") as f:
            for line in f:
                if not line.strip():
                    continue
                record = json.loads(line)
                status = str(record.get("status", "unknown"))
                counts[status] = counts.get(status, 0) + 1
                value = record.get("generation_seconds")
                if isinstance(value, (int, float)):
                    seconds.append(float(value))
        report.append(
            {
                "manifest": str(manifest),
                "counts": counts,
                "avg_generation_seconds": round(sum(seconds) / len(seconds), 3) if seconds else None,
            }
        )
        success_count = counts.get("success", 0)
        if expected_count is not None and success_count != expected_count:
            failures.append(f"{manifest}: success_count={success_count}, expected={expected_count}")
        if require_all_success and any(status != "success" for status in counts):
            failures.append(f"{manifest}: non-success counts={counts}")
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(f"wrote {output} manifests={len(report)}")
    if failures:
        raise RuntimeError("Manifest validation failed:\n" + "\n".join(failures))


def _safe_id(value: str) -> str:
    return "".join(char if char.isalnum() or char in {"-", "_", "."} else "_" for char in value)


if __name__ == "__main__":
    main()
