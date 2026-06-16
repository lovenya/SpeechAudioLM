#!/usr/bin/env python
"""Build stage input split files from upstream cascaded manifests.

Generated split files should be written outside git, typically under
/scratch/l/lovenya/salm_migration_clean/results/cascaded_full_validation_metadata/ttas_val/stage_inputs/.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from baselines.cascaded_manifest import build_sample, normalize_prompt, read_inspection_definitions, unwrap_sample


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
    inspect_samples = _read_v1_inspect_samples(manifest)
    rows = []
    for index, record in enumerate(_successful_records(manifest, allow_partial=args.allow_partial)):
        sample = _cumulative_sample(
            record,
            pipeline=args.pipeline,
            model=args.upstream_backend,
            stage_number=1,
            inspect_sample=inspect_samples.get(_record_sample_id(record)),
        )
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
    inspect_samples = _read_v1_inspect_samples(manifest)
    rows = []
    for index, record in enumerate(_successful_records(manifest, allow_partial=args.allow_partial)):
        sample = _cumulative_sample(
            record,
            pipeline=args.pipeline,
            model=args.text_model,
            stage_number=2,
            inspect_sample=inspect_samples.get(_record_sample_id(record)),
        )
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


def _cumulative_sample(
    record: dict,
    *,
    pipeline: str,
    model: str,
    stage_number: int,
    inspect_sample: dict | None = None,
) -> dict:
    if _is_v1_manifest_row(record):
        return _v1_cumulative_sample(
            record,
            pipeline=pipeline,
            model=model,
            stage_number=stage_number,
            inspect_sample=inspect_sample,
        )
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
    prompt = _legacy_prompt(task, model, source, extra)
    output_text = record.get("candidate_text") or extra.get("understanding_text")
    output_type = "audio_caption" if task == "audio_to_text" else "speech_text"
    if task == "text_to_text" and pipeline == "S2A":
        output_type = "audio_description"
    stage = {
        "stage": stage_number,
        "task": task,
        "model": model,
        "input": {"type": input_type, "file" if task != "text_to_text" else "text": input_value},
        "prompt": prompt,
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


def _v1_cumulative_sample(
    record: dict,
    *,
    pipeline: str,
    model: str,
    stage_number: int,
    inspect_sample: dict | None,
) -> dict:
    _validate_v1_stage_for_command(record, pipeline=pipeline, stage_number=stage_number)
    if inspect_sample and isinstance(inspect_sample.get("stages"), list):
        return _legacy_sample_from_v1_inspect(record, inspect_sample, stage_number=stage_number)
    if stage_number != 1:
        raise ValueError(
            f"v1 {record.get('stage')} manifest row for sample {record.get('sample_id')} requires sibling inspect.json"
        )
    stage = record.get("stage")
    expected_stage = "a2t" if pipeline == "A2S" else "s2t"
    if stage != expected_stage:
        raise ValueError(f"Expected v1 stage {expected_stage!r}; got {stage!r}")
    task = "audio_to_text" if pipeline == "A2S" else "speech_to_text"
    reference = _v1_reference(record, inspect_sample)
    output_text = _v1_output_text(record)
    output_type = "audio_caption" if pipeline == "A2S" else "speech_text"
    input_file = record.get("input", {}).get("audio_file" if pipeline == "A2S" else "speech_file")
    stage_doc = {
        "stage": 1,
        "task": task,
        "model": record.get("model") or model,
        "input": {"type": "audio" if pipeline == "A2S" else "speech_audio", "file": input_file},
        "prompt": None,
        "inference_seconds": _v1_inference_seconds(record),
        "output": {"type": output_type, "text": output_text},
        "error": None,
    }
    return build_sample(
        row_index=int(record.get("row_index", 0)),
        sample_id=str(record.get("sample_id") or "unknown"),
        record=reference,
        stage=stage_doc,
        status="success",
    )


def _validate_v1_stage_for_command(record: dict, *, pipeline: str, stage_number: int) -> None:
    expected = None
    if stage_number == 1:
        expected = "a2t" if pipeline == "A2S" else "s2t"
    elif stage_number == 2:
        expected = "t2t"
    actual = record.get("stage")
    if expected is not None and actual != expected:
        raise ValueError(f"Expected v1 stage {expected!r} for {pipeline} stage {stage_number}; got {actual!r}")


def _legacy_sample_from_v1_inspect(record: dict, inspect_sample: dict, *, stage_number: int) -> dict:
    stages = inspect_sample.get("stages")
    if not isinstance(stages, list) or len(stages) < stage_number:
        raise ValueError(f"inspect sample {record.get('sample_id')} does not contain stage {stage_number}")
    reference = _reference_from_v1_inspect(inspect_sample)
    sample: dict | None = None
    carried_record = reference
    for stage in stages[:stage_number]:
        legacy_stage = _legacy_stage_from_v1(stage)
        sample = build_sample(
            row_index=int(record.get("row_index", inspect_sample.get("row_index", 0) or 0)),
            sample_id=str(record.get("sample_id") or inspect_sample.get("sample_id") or "unknown"),
            record=carried_record,
            stage=legacy_stage,
            status="success",
        )
        carried_record = {"_cascaded_sample": sample}
    if sample is None:
        raise ValueError(f"inspect sample {record.get('sample_id')} has no usable stages")
    return sample


def _legacy_stage_from_v1(stage: dict) -> dict:
    design_stage = stage.get("stage")
    task = {
        "a2t": "audio_to_text",
        "s2t": "speech_to_text",
        "t2t": "text_to_text",
        "t2a": "text_to_audio",
        "t2s": "text_to_speech",
    }.get(design_stage)
    if task is None:
        raise ValueError(f"unsupported v1 inspect stage: {design_stage!r}")
    legacy = {
        "stage": stage.get("stage_index"),
        "task": task,
        "model": stage.get("model"),
        "input": _legacy_input_from_v1(stage),
        "prompt": stage.get("runtime_prompt"),
        "inference_seconds": (stage.get("timing") or {}).get("inference_seconds")
        if isinstance(stage.get("timing"), dict)
        else None,
        "output": _legacy_output_from_v1(stage),
        "error": stage.get("error"),
    }
    mode = _legacy_mode_from_v1(stage)
    if mode:
        legacy["mode"] = mode
    return legacy


def _legacy_input_from_v1(stage: dict) -> dict | None:
    value = stage.get("input")
    if not isinstance(value, dict):
        return None
    design_stage = stage.get("stage")
    if design_stage == "a2t":
        return {"type": "audio", "file": value.get("audio_file")}
    if design_stage == "s2t":
        return {"type": "speech_audio", "file": value.get("speech_file")}
    if design_stage == "t2t":
        if value.get("audio_caption") is not None:
            return {"type": "audio_caption", "text": value.get("audio_caption")}
        if value.get("speech_transcription") is not None:
            return {"type": "speech_text", "text": value.get("speech_transcription")}
    if design_stage == "t2s":
        return {"type": "speech_text", "text": value.get("generated_speech_text")}
    if design_stage == "t2a":
        return {"type": "audio_description", "text": value.get("generated_audio_caption")}
    return dict(value)


def _legacy_output_from_v1(stage: dict) -> dict | None:
    output = stage.get("output")
    if not isinstance(output, dict):
        return None
    text = _v1_text_from_output(output)
    if text is not None:
        key = _v1_text_key(output)
        output_type = {
            "audio_caption": "audio_caption",
            "speech_transcription": "speech_text",
            "generated_speech_text": "speech_text",
            "generated_audio_caption": "audio_description",
            "text": "text",
        }[key]
        return {"type": output_type, "text": text}
    if output.get("audio_file") or output.get("speech_file"):
        return {
            "type": "environmental_audio" if output.get("audio_file") else "speech_audio",
            "file": output.get("audio_file") or output.get("speech_file"),
            "sample_rate": output.get("sample_rate"),
            "duration_seconds": output.get("duration_seconds"),
        }
    return dict(output)


def _legacy_mode_from_v1(stage: dict) -> str | None:
    output = stage.get("output")
    if not isinstance(output, dict):
        return None
    if output.get("generated_speech_text") is not None:
        return "audio_description_to_speech"
    if output.get("generated_audio_caption") is not None:
        return "speech_to_audio_description"
    return None


def _is_v1_manifest_row(record: dict) -> bool:
    return record.get("schema_version") == 1 and isinstance(record.get("output"), dict) and isinstance(record.get("stage"), str)


def _read_v1_inspect_samples(manifest: Path) -> dict[str, dict]:
    inspect_path = manifest.with_name("inspect.json")
    if not inspect_path.is_file():
        return {}
    document = json.loads(inspect_path.read_text(encoding="utf-8"))
    samples = document.get("samples")
    if not isinstance(samples, list):
        return {}
    indexed = {}
    for sample in samples:
        if isinstance(sample, dict) and isinstance(sample.get("sample_id"), str):
            indexed[sample["sample_id"]] = sample
    return indexed


def _record_sample_id(record: dict) -> str | None:
    value = record.get("sample_id")
    return value if isinstance(value, str) and value else None


def _v1_reference(record: dict, inspect_sample: dict | None) -> dict:
    if inspect_sample:
        reference = _reference_from_v1_inspect(inspect_sample)
        if reference:
            return reference
    input_value = record.get("input") if isinstance(record.get("input"), dict) else {}
    return {
        "audio_file": input_value.get("audio_file"),
        "speech_file": input_value.get("speech_file"),
    }


def _reference_from_v1_inspect(inspect_sample: dict) -> dict:
    reference = inspect_sample.get("reference")
    if not isinstance(reference, dict):
        return {}
    return {
        "audio_file": reference.get("audio_file"),
        "speech_file": reference.get("speech_file"),
        "audio_cap": reference.get("reference_audio_caption") or reference.get("audio_caption") or reference.get("audio_cap"),
        "speech_text": reference.get("reference_speech_text") or reference.get("speech_text"),
        "audio_metadata": reference.get("audio_metadata"),
    }


def _v1_output_text(record: dict) -> str:
    output = record.get("output")
    if not isinstance(output, dict):
        raise ValueError(f"v1 row {record.get('sample_id')} has no output object")
    text = _v1_text_from_output(output)
    if not isinstance(text, str) or not text.strip():
        raise ValueError(f"v1 row {record.get('sample_id')} has no usable text output")
    return text.strip()


def _v1_text_from_output(output: dict) -> str | None:
    key = _v1_text_key(output)
    return output.get(key) if key else None


def _v1_text_key(output: dict) -> str | None:
    for key in ("audio_caption", "speech_transcription", "generated_speech_text", "generated_audio_caption", "text"):
        if isinstance(output.get(key), str):
            return key
    return None


def _v1_inference_seconds(record: dict) -> float | None:
    timing = record.get("timing")
    if not isinstance(timing, dict):
        return None
    seconds = timing.get("inference_seconds")
    return seconds if isinstance(seconds, (int, float)) else None


def _legacy_prompt(task: str, model: str, source: dict, extra: dict) -> dict | None:
    rendered = extra.get("generation_prompt") or extra.get("understanding_prompt")
    messages = extra.get("messages") if isinstance(extra.get("messages"), list) else []
    if rendered is not None or messages:
        return normalize_prompt({"rendered": rendered, "messages": messages})
    if task in {"speech_to_text", "text_to_text"}:
        return None
    audio_file = source.get("audio_file")
    if model == "audio_flamingo_next":
        prompt = "Generate a caption for the input audio."
        content = [{"type": "text", "text": prompt}]
        if isinstance(audio_file, str):
            content.append({"type": "audio", "path": audio_file})
        return {"messages": [{"role": "user", "content": content}]}
    if model == "qwen3_omni_captioner":
        content = []
        if isinstance(audio_file, str):
            content.append({"type": "audio", "audio": audio_file})
        return {"messages": [{"role": "user", "content": content}]}
    return None


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
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


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
