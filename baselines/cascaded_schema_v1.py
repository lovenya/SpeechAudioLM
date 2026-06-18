from __future__ import annotations

from pathlib import Path
from typing import Any


SCHEMA_VERSION = 1

TASK_TO_STAGE = {
    "audio_captioning": "a2t",
    "audio_to_text": "a2t",
    "asr": "s2t",
    "speech_to_text": "s2t",
    "text_to_text": "t2t",
    "text_to_audio": "t2a",
    "text_to_speech": "t2s",
}

A2S_MODE = "audio_description_to_speech"
S2A_MODE = "speech_to_audio_description"


def legacy_sample_to_stage_manifest_row(record: dict[str, Any], *, stage_index: int | None = None) -> dict[str, Any]:
    """Convert one legacy cascaded sample or wrapped sample to a v1 stage manifest row."""
    sample = _unwrap_sample(record)
    stages = _stages(sample)
    index = _stage_index(stages, stage_index)
    stage = stages[index - 1]
    design_stage = _design_stage(stage)
    status = _stage_status(sample, stage, index)
    error = _stage_error(sample, stage, index) if status == "error" else None
    return {
        "schema_version": SCHEMA_VERSION,
        "row_index": sample.get("row_index"),
        "sample_id": sample.get("sample_id"),
        "stage": design_stage,
        "model": stage.get("model"),
        "status": status,
        "input": _manifest_input(sample, stages, index, design_stage),
        "output": None if status == "error" else _design_output(stage, design_stage),
        "timing": _timing(stage),
        "error": error,
    }


def legacy_sample_to_inspect_sample(record: dict[str, Any], *, stage_index: int | None = None) -> dict[str, Any]:
    """Convert one legacy cascaded sample or wrapped sample to the v1 inspect sample shape."""
    sample = _unwrap_sample(record)
    stages = _stages(sample)
    limit = _stage_index(stages, stage_index)
    selected_stages = stages[:limit]
    final_stage = selected_stages[-1]
    final_design_stage = _design_stage(final_stage)
    final_status = _stage_status(sample, final_stage, limit)
    return {
        "sample_id": sample.get("sample_id"),
        "row_index": sample.get("row_index"),
        "status": final_status,
        "reference": _design_reference(sample.get("reference")),
        "stages": [_inspect_stage(sample, selected_stages, index) for index in range(1, limit + 1)],
        "final_output": _final_output(sample, final_stage, final_design_stage, final_status),
        "error": _stage_error(sample, final_stage, limit) if final_status == "error" else None,
    }


def _unwrap_sample(record: dict[str, Any]) -> dict[str, Any]:
    sample = record.get("sample")
    if isinstance(sample, dict):
        return sample
    return record


def _stages(sample: dict[str, Any]) -> list[dict[str, Any]]:
    stages = sample.get("stages")
    if not isinstance(stages, list) or not stages:
        raise ValueError("legacy sample must contain at least one stage")
    if not all(isinstance(stage, dict) for stage in stages):
        raise ValueError("legacy sample stages must be objects")
    return stages


def _stage_index(stages: list[dict[str, Any]], stage_index: int | None) -> int:
    index = len(stages) if stage_index is None else stage_index
    if index < 1 or index > len(stages):
        raise ValueError(f"stage_index must be between 1 and {len(stages)}")
    return index


def _design_stage(stage: dict[str, Any]) -> str:
    task = stage.get("task")
    if not isinstance(task, str) or task not in TASK_TO_STAGE:
        raise ValueError(f"unsupported legacy stage task: {task!r}")
    return TASK_TO_STAGE[task]


def _stage_status(sample: dict[str, Any], stage: dict[str, Any], index: int) -> str:
    if _stage_error(sample, stage, index) is not None:
        return "error"
    status = sample.get("status") if index == len(sample.get("stages", [])) else None
    return "success" if status in {None, "success"} else str(status)


def _stage_error(sample: dict[str, Any], stage: dict[str, Any], index: int) -> dict[str, Any] | None:
    error = stage.get("error")
    if error is None and index == len(sample.get("stages", [])):
        error = sample.get("error")
    if error is None and index == len(sample.get("stages", [])) and sample.get("status") == "error":
        error = {"type": "StageError", "message": "Legacy sample status is error."}
    if error is None:
        return None
    if isinstance(error, dict):
        return dict(error)
    return {"type": "StageError", "message": str(error)}


def _manifest_input(
    sample: dict[str, Any],
    stages: list[dict[str, Any]],
    index: int,
    design_stage: str,
) -> dict[str, Any]:
    if design_stage == "a2t":
        return _file_input(stages[index - 1].get("input"), "audio_file")
    if design_stage == "s2t":
        return _file_input(stages[index - 1].get("input"), "speech_file")
    previous = stages[index - 2] if index > 1 else None
    result = (
        {"source_stage_index": None, "source_sample_id": sample.get("sample_id"), "source_status": None}
        if previous is None
        else {
            "source_stage_index": index - 1,
            "source_sample_id": sample.get("sample_id"),
            "source_status": _stage_status(sample, previous, index - 1),
        }
    )
    if design_stage == "t2s":
        reference = _stage_reference(stages[index - 1].get("input"))
        if reference is not None:
            result["reference"] = reference
    return result


def _stage_reference(value: Any) -> dict[str, Any] | None:
    if not isinstance(value, dict):
        return None
    reference = value.get("reference")
    if not isinstance(reference, dict):
        return None
    return dict(reference)


def _file_input(value: Any, field_name: str) -> dict[str, Any]:
    if not isinstance(value, dict):
        return {field_name: None}
    path = value.get(field_name) or value.get("file")
    return {field_name: path}


def _inspect_stage(sample: dict[str, Any], stages: list[dict[str, Any]], index: int) -> dict[str, Any]:
    stage = stages[index - 1]
    design_stage = _design_stage(stage)
    status = _stage_status(sample, stage, index)
    return {
        "stage_index": index,
        "stage": design_stage,
        "model": stage.get("model"),
        "status": status,
        "input": _runtime_input(stage, design_stage),
        "runtime_prompt": stage.get("prompt"),
        "output": None if status == "error" else _design_output(stage, design_stage),
        "timing": _timing(stage),
        "error": _stage_error(sample, stage, index) if status == "error" else None,
    }


def _runtime_input(stage: dict[str, Any], design_stage: str) -> dict[str, Any] | None:
    legacy_input = stage.get("input")
    if not isinstance(legacy_input, dict):
        return None
    if design_stage == "a2t":
        return _file_input(legacy_input, "audio_file")
    if design_stage == "s2t":
        return _file_input(legacy_input, "speech_file")
    text = legacy_input.get("text")
    if not isinstance(text, str):
        return dict(legacy_input)
    if design_stage == "t2t":
        if _t2t_output_name(stage) == "generated_speech_text":
            return {"audio_caption": text}
        if _t2t_output_name(stage) == "generated_audio_caption":
            return {"speech_transcription": text}
    if design_stage == "t2s":
        result = {"generated_speech_text": text}
        reference = _stage_reference(legacy_input)
        if reference is not None:
            result["reference"] = reference
        return result
    if design_stage == "t2a":
        return {"generated_audio_caption": text}
    return {"text": text}


def _design_output(stage: dict[str, Any], design_stage: str) -> dict[str, Any] | None:
    legacy_output = stage.get("output")
    if not isinstance(legacy_output, dict):
        return None
    if design_stage in {"a2t", "s2t", "t2t"}:
        return _text_output(stage, legacy_output, design_stage)
    if design_stage in {"t2a", "t2s"}:
        return _media_output(legacy_output, design_stage)
    raise ValueError(f"unsupported design stage: {design_stage!r}")


def _text_output(stage: dict[str, Any], legacy_output: dict[str, Any], design_stage: str) -> dict[str, Any]:
    text = _text_value(legacy_output)
    if design_stage == "a2t":
        return {"type": "text", "audio_caption": text}
    if design_stage == "s2t":
        return {"type": "text", "speech_transcription": text}
    output_name = _t2t_output_name(stage)
    if output_name is None:
        return {"type": "text", "text": text}
    return {"type": "text", output_name: text}


def _text_value(value: dict[str, Any]) -> str | None:
    for key in ("text", "audio_caption", "speech_transcription", "generated_speech_text", "generated_audio_caption"):
        text = value.get(key)
        if isinstance(text, str):
            return text
    return None


def _t2t_output_name(stage: dict[str, Any]) -> str | None:
    mode = stage.get("mode")
    if mode == A2S_MODE:
        return "generated_speech_text"
    if mode == S2A_MODE:
        return "generated_audio_caption"
    output = stage.get("output")
    output_type = output.get("type") if isinstance(output, dict) else None
    if output_type in {"speech_text", "generated_speech_text"}:
        return "generated_speech_text"
    if output_type in {"audio_caption", "audio_description", "generated_audio_caption"}:
        return "generated_audio_caption"
    legacy_input = stage.get("input")
    input_type = legacy_input.get("type") if isinstance(legacy_input, dict) else None
    if input_type == "audio_caption":
        return "generated_speech_text"
    if input_type in {"speech_text", "speech_transcription"}:
        return "generated_audio_caption"
    return None


def _media_output(legacy_output: dict[str, Any], design_stage: str) -> dict[str, Any]:
    media_type = "speech" if design_stage == "t2s" else "audio"
    path_field = "speech_file" if design_stage == "t2s" else "audio_file"
    output = {
        "type": media_type,
        path_field: _outputs_path(legacy_output.get(path_field) or legacy_output.get("file")),
    }
    for key in ("sample_rate", "duration_seconds"):
        if legacy_output.get(key) is not None:
            output[key] = legacy_output[key]
    return output


def _outputs_path(value: Any) -> str | None:
    if not isinstance(value, str) or not value:
        return None
    normalized = value.replace("\\", "/")
    if normalized == "outputs" or normalized.startswith("outputs/"):
        return normalized
    return (Path("outputs") / Path(normalized).name).as_posix()


def _timing(stage: dict[str, Any]) -> dict[str, Any]:
    timing: dict[str, Any] = {}
    if stage.get("inference_seconds") is not None:
        timing["inference_seconds"] = stage["inference_seconds"]
    return timing


def _design_reference(value: Any) -> dict[str, Any]:
    if not isinstance(value, dict):
        return {}
    reference = {
        "audio_file": value.get("audio_file"),
        "speech_file": value.get("speech_file"),
        "reference_audio_caption": value.get("reference_audio_caption")
        or value.get("audio_caption")
        or value.get("audio_cap"),
        "reference_speech_text": value.get("reference_speech_text")
        or value.get("speech_text")
        or value.get("speech_transcript"),
    }
    if value.get("audio_metadata") is not None:
        reference["audio_metadata"] = value["audio_metadata"]
    return {key: item for key, item in reference.items() if item is not None}


def _final_output(
    sample: dict[str, Any],
    final_stage: dict[str, Any],
    final_design_stage: str,
    final_status: str,
) -> dict[str, Any] | None:
    if final_status != "success" or final_design_stage not in {"t2a", "t2s"}:
        return None
    legacy_final = sample.get("final_output")
    if isinstance(legacy_final, dict):
        synthetic_stage = dict(final_stage)
        synthetic_stage["output"] = legacy_final
        return _design_output(synthetic_stage, final_design_stage)
    return _design_output(final_stage, final_design_stage)
