from __future__ import annotations

import json
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Iterator

from project_paths import resolve_config_value


@dataclass(frozen=True)
class ManifestRow:
    ordinal: int
    record: dict


def read_json(path: Path) -> dict:
    with path.open("r", encoding="utf-8") as f:
        return resolve_config_value(json.load(f))


def iter_jsonl(path: Path, max_samples: int | None = None, start_index: int = 0) -> Iterator[ManifestRow]:
    if start_index < 0:
        raise ValueError("--start-index must be >= 0")
    if max_samples is not None and max_samples <= 0:
        raise ValueError("--max-samples must be a positive integer")

    inspect_samples = _read_v1_inspect_samples(path)
    emitted = 0
    seen = 0
    with path.open("r", encoding="utf-8") as f:
        for ordinal, line in enumerate(f):
            line = line.strip()
            if not line:
                continue
            record = json.loads(line)
            if seen < start_index:
                seen += 1
                continue
            yield ManifestRow(
                ordinal=ordinal,
                record=_normalize_generation_record(
                    record,
                    manifest_dir=path.parent,
                    inspect_sample=inspect_samples.get(_record_sample_id(record)),
                ),
            )
            seen += 1
            emitted += 1
            if max_samples is not None and emitted >= max_samples:
                break


def _legacy_task_name(task: object) -> object:
    return {
        "audio_to_text": "A2T",
        "speech_to_text": "S2T",
        "text_to_text": "T2T",
        "text_to_speech": "T2S",
        "text_to_audio": "T2A",
    }.get(task, task)


def _normalize_generation_record(
    record: dict,
    manifest_dir: Path | None = None,
    inspect_sample: dict | None = None,
) -> dict:
    if _is_v1_cascaded_record(record):
        return _normalize_v1_cascaded_record(
            record,
            manifest_dir=manifest_dir,
            inspect_sample=inspect_sample,
        )
    sample = record.get("sample")
    if not isinstance(sample, dict):
        return record
    stages = sample.get("stages") if isinstance(sample.get("stages"), list) else []
    last_stage = stages[-1] if stages else {}
    output = last_stage.get("output") if isinstance(last_stage.get("output"), dict) else {}
    reference = dict(sample.get("reference") or {})
    if "audio_caption" in reference and "audio_cap" not in reference:
        reference["audio_cap"] = reference["audio_caption"]
    normalized = {
        key: value
        for key, value in record.items()
        if key != "sample"
    }
    normalized.update(
        {
            "status": sample.get("status"),
            "task": _legacy_task_name(last_stage.get("task")),
            "model": last_stage.get("model"),
            "row_index": sample.get("row_index"),
            "sample_id": sample.get("sample_id"),
            "source": reference,
            "candidate_text": output.get("text"),
            "output_file": output.get("file"),
            "sample_rate": output.get("sample_rate"),
            "generation_seconds": last_stage.get("inference_seconds"),
            "prompt": (last_stage.get("input") or {}).get("text"),
            "model_extra": {"stages": stages},
            "error": sample.get("error"),
        }
    )
    return normalized


def _is_v1_cascaded_record(record: dict) -> bool:
    return record.get("schema_version") == 1 and (
        record.get("stage") is not None
        or record.get("direction") is not None
        or isinstance(record.get("output"), dict)
        or isinstance(record.get("final_output"), dict)
    )


def _normalize_v1_cascaded_record(
    record: dict,
    manifest_dir: Path | None = None,
    inspect_sample: dict | None = None,
) -> dict:
    output = record.get("output") if isinstance(record.get("output"), dict) else record.get("final_output")
    output = output if isinstance(output, dict) else {}
    if inspect_sample is not None:
        _validate_v1_inspect_join(record, inspect_sample)
    normalized = dict(record)
    source = _v1_source(record, inspect_sample=inspect_sample)
    task = _v1_task(record)
    input_text = _v1_input_text(record, inspect_sample=inspect_sample)
    direction = _v1_text_direction(output)
    prompt = _v1_runtime_prompt(record, inspect_sample=inspect_sample)
    model_extra = {"v1_output": output, "stages": record.get("stages", [])}
    if direction is not None:
        model_extra["mode"] = {
            "audio_to_speech": "audio_description_to_speech",
            "speech_to_audio": "speech_to_audio_description",
        }[direction]
    normalized.update(
        {
            "task": task,
            "model": record.get("model"),
            "source": source,
            "candidate_text": _v1_output_text(output),
            "output_file": _v1_output_file(output, manifest_dir),
            "sample_rate": output.get("sample_rate"),
            "generation_seconds": _v1_generation_seconds(record),
            "model_extra": model_extra,
        }
    )
    if input_text is not None:
        normalized["input_text"] = input_text
    if task == "T2T" and direction is not None:
        normalized["direction"] = direction
    if prompt is not None:
        normalized["prompt"] = prompt
    return normalized


def _v1_source(record: dict, inspect_sample: dict | None = None) -> dict:
    source = _v1_reference(inspect_sample)
    input_value = record.get("input")
    if isinstance(input_value, dict):
        for key, value in input_value.items():
            source.setdefault(key, value)
    return source


def _read_v1_inspect_samples(manifest_path: Path) -> dict[str, dict]:
    inspect_path = manifest_path.with_name("inspect.json")
    if not inspect_path.exists():
        inspect_path = manifest_path.with_name("manifest.inspect.json")
    if not inspect_path.exists():
        return {}
    with inspect_path.open("r", encoding="utf-8") as f:
        document = resolve_config_value(json.load(f))
    samples = document.get("samples") if isinstance(document, dict) else None
    if not isinstance(samples, list):
        return {}
    result: dict[str, dict] = {}
    for sample in samples:
        if not isinstance(sample, dict):
            continue
        sample_id = _record_sample_id(sample)
        if sample_id is not None:
            result[sample_id] = sample
    return result


def _validate_v1_inspect_join(record: dict, inspect_sample: dict) -> None:
    row_sample_id = _record_sample_id(record)
    inspect_sample_id = _record_sample_id(inspect_sample)
    if row_sample_id is not None and inspect_sample_id != row_sample_id:
        raise ValueError(f"inspect.json sample_id mismatch for {row_sample_id}: {inspect_sample_id}")
    input_value = record.get("input")
    if not isinstance(input_value, dict):
        return
    source_sample_id = input_value.get("source_sample_id")
    if isinstance(source_sample_id, str) and row_sample_id is not None and source_sample_id != row_sample_id:
        raise ValueError(f"v1 row {row_sample_id} references source_sample_id {source_sample_id}")
    source_stage_index = input_value.get("source_stage_index")
    source_status = input_value.get("source_status")
    if not isinstance(source_stage_index, int):
        return
    stage = _v1_inspect_stage_by_index(inspect_sample, source_stage_index)
    if stage is None:
        raise ValueError(f"v1 row {row_sample_id} references missing source_stage_index {source_stage_index}")
    if isinstance(source_status, str) and stage.get("status") != source_status:
        raise ValueError(
            f"v1 row {row_sample_id} source_status {source_status} does not match inspect stage {source_stage_index}"
        )


def _record_sample_id(record: dict) -> str | None:
    sample_id = record.get("sample_id")
    return sample_id if isinstance(sample_id, str) else None


def _v1_reference(inspect_sample: dict | None) -> dict:
    if not isinstance(inspect_sample, dict):
        return {}
    reference = inspect_sample.get("reference")
    if not isinstance(reference, dict):
        return {}
    source = dict(reference)
    audio_caption = (
        reference.get("reference_audio_caption")
        or reference.get("audio_caption")
        or reference.get("audio_cap")
    )
    speech_text = (
        reference.get("reference_speech_text")
        or reference.get("speech_text")
        or reference.get("speech_transcript")
    )
    if isinstance(audio_caption, str):
        source.setdefault("reference_audio_caption", audio_caption)
        source.setdefault("audio_caption", audio_caption)
        source.setdefault("audio_cap", audio_caption)
    if isinstance(speech_text, str):
        source.setdefault("reference_speech_text", speech_text)
        source.setdefault("speech_text", speech_text)
    return source


def _v1_task(record: dict) -> str | None:
    stage = record.get("stage")
    if isinstance(stage, str):
        return {
            "a2t": "A2T",
            "s2t": "S2T",
            "t2t": "T2T",
            "t2s": "T2S",
            "t2a": "T2A",
        }.get(stage)
    direction = record.get("direction")
    if direction == "a2s":
        return "A2S"
    if direction == "s2a":
        return "S2A"
    return None


def _v1_output_text(output: dict) -> str | None:
    for key in (
        "audio_caption",
        "speech_transcription",
        "generated_speech_text",
        "generated_audio_caption",
        "text",
    ):
        value = output.get(key)
        if isinstance(value, str):
            return value
    return None


def _v1_input_text(record: dict, inspect_sample: dict | None = None) -> str | None:
    for value in (_v1_stage_input(record, inspect_sample), record.get("input")):
        if not isinstance(value, dict):
            continue
        text = _v1_text_value(value)
        if text is not None:
            return text
    return None


def _v1_stage_input(record: dict, inspect_sample: dict | None = None) -> dict | None:
    stage = _v1_inspect_stage_for_record(record, inspect_sample)
    if stage is not None:
        return stage.get("input") if isinstance(stage.get("input"), dict) else None
    return None


def _v1_runtime_prompt(record: dict, inspect_sample: dict | None = None) -> str | None:
    stage = _v1_inspect_stage_for_record(record, inspect_sample)
    if stage is None:
        return None
    prompt = stage.get("runtime_prompt")
    return prompt if isinstance(prompt, str) else None


def _v1_inspect_stage_for_record(record: dict, inspect_sample: dict | None = None) -> dict | None:
    if not isinstance(inspect_sample, dict):
        return None
    stages = inspect_sample.get("stages")
    if not isinstance(stages, list):
        return None
    record_stage = record.get("stage")
    if record_stage is None:
        record_stage = {"a2s": "t2s", "s2a": "t2a"}.get(record.get("direction"))
    record_model = record.get("model")
    fallback: dict | None = None
    for stage in stages:
        if not isinstance(stage, dict):
            continue
        if stage.get("stage") == record_stage and (
            record_model is None or stage.get("model") == record_model
        ):
            return stage
        if stage.get("stage") == record_stage:
            fallback = stage
    return fallback


def _v1_inspect_stage_by_index(inspect_sample: dict, stage_index: int) -> dict | None:
    stages = inspect_sample.get("stages")
    if not isinstance(stages, list):
        return None
    for stage in stages:
        if isinstance(stage, dict) and stage.get("stage_index") == stage_index:
            return stage
    return None


def _v1_text_value(value: dict) -> str | None:
    for key in (
        "audio_caption",
        "audio_cap",
        "speech_transcription",
        "speech_text",
        "generated_speech_text",
        "generated_audio_caption",
        "text",
    ):
        text = value.get(key)
        if isinstance(text, str) and text.strip():
            return text
    return None


def _v1_text_direction(output: dict) -> str | None:
    if isinstance(output.get("generated_speech_text"), str):
        return "audio_to_speech"
    if isinstance(output.get("generated_audio_caption"), str):
        return "speech_to_audio"
    return None


def _v1_output_file(output: dict, manifest_dir: Path | None) -> str | None:
    value = output.get("audio_file") or output.get("speech_file") or output.get("file")
    if not isinstance(value, str) or not value:
        return None
    path = Path(value)
    if path.is_absolute() or value.startswith("results/") or value.startswith("temporary/results/"):
        return value
    if manifest_dir is not None and path.parts and path.parts[0] == "outputs":
        return (manifest_dir / path).as_posix()
    return value


def _v1_generation_seconds(record: dict) -> float | None:
    timing = record.get("timing")
    if not isinstance(timing, dict):
        return None
    seconds = timing.get("inference_seconds")
    return seconds if isinstance(seconds, (int, float)) else None


def write_json(path: Path, data: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, ensure_ascii=False, sort_keys=True)
        f.write("\n")


def write_jsonl(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")


def utc_timestamp() -> str:
    return datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
