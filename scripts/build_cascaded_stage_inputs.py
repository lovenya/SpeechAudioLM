#!/usr/bin/env python
"""Build stage input split files from upstream cascaded manifests.

Generated split files should be written outside git, typically under
/scratch/l/lovenya/salm_migration_clean/results/cascaded_full_validation_metadata/ttas_val/stage_inputs/.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from baselines.cascaded_manifest import build_sample, read_inspection_definitions, reference_from_record, unwrap_sample


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
    manifest = Path(args.manifest)
    definitions = read_inspection_definitions(manifest)
    rows = []
    for index, record in enumerate(_successful_records(manifest, allow_partial=args.allow_partial)):
        sample = _cumulative_sample(record, pipeline=args.pipeline, model=args.upstream_backend, stage_number=1)
        text = _sample_output_text(sample)
        sample_id = str(sample.get("sample_id") or f"row_{index}")
        reference = sample["reference"]
        rows.append(
            {
                "audio_file": reference.get("audio_file") or f"stage_inputs/{sample_id}.wav",
                "speech_file": reference.get("speech_file") or f"stage_inputs/{sample_id}.wav",
                "speech_text": text,
                "audio_cap": text,
                "_cascaded_pipeline": args.pipeline,
                "_cascaded_sample": sample,
                "_cascaded_stage_definitions": definitions or [_legacy_stage_definition(sample["stages"][-1])],
            }
        )
    return rows


def _t2t_to_final(args: argparse.Namespace) -> list[dict]:
    manifest = Path(args.manifest)
    definitions = read_inspection_definitions(manifest)
    rows = []
    for index, record in enumerate(_successful_records(manifest, allow_partial=args.allow_partial)):
        sample = _cumulative_sample(record, pipeline=args.pipeline, model=args.text_model, stage_number=2)
        text = _sample_output_text(sample)
        sample_id = str(sample.get("sample_id") or f"row_{index}")
        reference = sample["reference"]
        row = {
            "audio_file": reference.get("audio_file") or f"stage_inputs/{sample_id}.wav",
            "speech_file": reference.get("speech_file") or f"stage_inputs/{sample_id}.wav",
            "speech_text": text,
            "audio_cap": text,
            "_cascaded_pipeline": args.pipeline,
            "_cascaded_sample": sample,
            "_cascaded_stage_definitions": definitions or [_legacy_stage_definition(stage) for stage in sample["stages"]],
        }
        if args.pipeline == "A2S":
            if reference.get("speech_file"):
                row["reference_speech_file"] = reference["speech_file"]
            if reference.get("speech_text"):
                row["reference_speech_text"] = reference["speech_text"]
        rows.append(row)
    return rows


def _cumulative_sample(record: dict, *, pipeline: str, model: str, stage_number: int) -> dict:
    sample = unwrap_sample(record)
    if isinstance(record.get("sample"), dict):
        return sample
    carried = record.get("_cascaded_sample")
    if isinstance(carried, dict):
        return carried

    task = "audio_to_text" if pipeline == "A2S" and stage_number == 1 else "speech_to_text"
    if stage_number == 2:
        task = "text_to_text"
    source = record.get("source") if isinstance(record.get("source"), dict) else record
    carried_record = source
    if stage_number == 2 and isinstance(source.get("original_row"), dict):
        original = source["original_row"]
        upstream_text = source.get("speech_text") or source.get("audio_cap")
        upstream_task = "audio_to_text" if pipeline == "A2S" else "speech_to_text"
        upstream_input = {
            "type": "audio" if pipeline == "A2S" else "speech_audio",
            "file": original.get("audio_file") if pipeline == "A2S" else original.get("speech_file"),
        }
        upstream_stage = {
            "stage": 1,
            "task": upstream_task,
            "model": (source.get("stage_provenance") or {}).get("upstream_backend", "unknown"),
            "input": upstream_input,
            "prompt": None,
            "inference_seconds": None,
            "output": {"type": "audio_caption" if pipeline == "A2S" else "speech_text", "text": upstream_text},
            "error": None,
        }
        prior_sample = build_sample(
            row_index=int(record.get("row_index", 0)),
            sample_id=str(record.get("sample_id") or "unknown"),
            record=original,
            stage=upstream_stage,
            status="success",
        )
        carried_record = {"_cascaded_sample": prior_sample}
    input_value = source.get("audio_file") if task == "audio_to_text" else source.get("speech_file")
    input_type = "audio" if task == "audio_to_text" else "speech_audio"
    if task == "text_to_text":
        input_value = source.get("speech_text") or source.get("audio_cap") or record.get("prompt")
        input_type = "audio_caption" if pipeline == "A2S" else "speech_text"
    extra = record.get("model_extra") if isinstance(record.get("model_extra"), dict) else {}
    rendered = extra.get("generation_prompt") or extra.get("understanding_prompt") or record.get("prompt")
    messages = extra.get("messages") if isinstance(extra.get("messages"), list) else []
    output_text = record.get("candidate_text") or extra.get("understanding_text")
    output_type = "audio_caption" if task == "audio_to_text" else "speech_text"
    if task == "text_to_text" and pipeline == "S2A":
        output_type = "audio_description"
    stage = {
        "stage": stage_number,
        "task": task,
        "model": model,
        "input": {"type": input_type, "file" if task != "text_to_text" else "text": input_value},
        "prompt": {"rendered": rendered, "messages": messages} if rendered is not None or messages else None,
        "inference_seconds": record.get("generation_seconds"),
        "output": {"type": output_type, "text": output_text},
        "error": None,
    }
    if task == "text_to_text":
        stage["mode"] = "audio_description_to_speech" if pipeline == "A2S" else "speech_to_audio_description"
    return build_sample(
        row_index=int(record.get("row_index", 0)),
        sample_id=str(record.get("sample_id") or "unknown"),
        record=carried_record,
        stage=stage,
        status="success",
    )


def _sample_output_text(sample: dict) -> str:
    stages = sample.get("stages")
    if not isinstance(stages, list) or not stages:
        raise ValueError(f"Sample {sample.get('sample_id')} has no stages")
    output = stages[-1].get("output")
    text = output.get("text") if isinstance(output, dict) else None
    if not isinstance(text, str) or not text.strip():
        raise ValueError(f"Sample {sample.get('sample_id')} has no usable final stage text")
    return text.strip()


def _legacy_stage_definition(stage: dict) -> dict:
    return {
        "stage": stage.get("stage"),
        "task": stage.get("task"),
        "model": stage.get("model"),
        "checkpoint": None,
        "dtype": None,
        "prompt_template": None,
        "generation": {},
        "runtime": {"device": None, "model_load_seconds": None},
    }


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
            sample = unwrap_sample(record)
            if sample.get("status") == "success":
                rows.append(record)
            else:
                errors.append(record)
    if errors and not allow_partial:
        raise RuntimeError(f"{path} contains {len(errors)} non-success records")
    if errors:
        print(f"warning: {path} contains {len(errors)} non-success records; wrote successful rows only")
    return rows


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
                sample = unwrap_sample(record)
                status = str(sample.get("status", "unknown"))
                counts[status] = counts.get(status, 0) + 1
                stages = sample.get("stages")
                value = stages[-1].get("inference_seconds") if isinstance(stages, list) and stages else record.get("generation_seconds")
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
