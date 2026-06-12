"""Strict scientific semantics for stable-audio-metrics CLAPScore.

CLAPScore is a cosine similarity between generated audio and one text input.
The score's scientific meaning therefore depends on which text was embedded:
a target caption measures target-caption alignment, while the exact text used
to condition generation measures prompt adherence.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from numbers import Real
from typing import Any, Mapping, Sequence


STABLE_AUDIO_METRICS_COMMIT = "fd55536cc812c460ecc421220864993c7f168184"
LAION_CLAP_VERSION = "1.1.4"
LAION_CLAP_CHECKPOINT = "630k-audioset-fusion-best.pt"
TARGET_CAPTION_ALIGNMENT = "target_caption_alignment"
PROMPT_ADHERENCE = "prompt_adherence"
TARGET_CAPTION_CLAP_SCORE = "target_caption_clap_score"
PROMPT_ADHERENCE_CLAP_SCORE = "prompt_adherence_clap_score"

_SEMANTICS = {TARGET_CAPTION_ALIGNMENT, PROMPT_ADHERENCE}
_TEXT_FIELDS = {
    TARGET_CAPTION_ALIGNMENT: "target_caption",
    PROMPT_ADHERENCE: "conditioning_prompt",
}
_METRIC_NAMES = {
    TARGET_CAPTION_ALIGNMENT: TARGET_CAPTION_CLAP_SCORE,
    PROMPT_ADHERENCE: PROMPT_ADHERENCE_CLAP_SCORE,
}


class CLAPScoreValidationError(ValueError):
    """Raised when CLAPScore inputs or outputs are scientifically invalid."""


@dataclass(frozen=True)
class CLAPScoreInput:
    sample_id: str
    generated_id: str
    semantics: str
    text: str


@dataclass(frozen=True)
class CLAPScoreRecord:
    sample_id: str
    generated_id: str
    semantics: str
    text: str
    score: float


@dataclass(frozen=True)
class CLAPScoreAggregation:
    records: tuple[CLAPScoreRecord, ...]
    corpus_mean: float | None
    sample_count: int
    expected_sample_count: int
    metadata: Mapping[str, Any]


def official_clap_score_constraints() -> dict[str, Any]:
    """Return constraints verified from the pinned vendor and adapter code."""

    return {
        "implementation": "stable-audio-metrics",
        "implementation_commit": STABLE_AUDIO_METRICS_COMMIT,
        "model": "LAION-CLAP",
        "model_package": "laion-clap",
        "model_package_version": LAION_CLAP_VERSION,
        "checkpoint": LAION_CLAP_CHECKPOINT,
        "checkpoint_fusion_enabled": True,
        "score": "cosine_similarity(generated_audio_embedding, text_embedding)",
        "text_encoder": "roberta-base",
        "text_context_tokens": 77,
        "text_padding": "max_length",
        "text_truncation": True,
        "text_tokenizer_adapter": (
            "repository adapter preserves the batch dimension while using the "
            "LAION-CLAP 77-token RoBERTa policy"
        ),
        "sample_rate_hz": 48_000,
        "channel_policy": "librosa.load(..., mono=True)",
        "resampling_policy": "librosa.load(..., sr=48000)",
        "amplitude_policy": "pyloudnorm peak normalization to -1.0 dBFS",
        "pcm_roundtrip": (
            "clip to [-1, 1], multiply by 32767 and cast to int16, then "
            "divide by 32767 and cast to float32"
        ),
        "requires_generated_audio": True,
        "requires_nonempty_selected_text": True,
        "partial_results_accepted": True,
    }


official_clap_preprocessing = official_clap_score_constraints


def select_clap_score_text(record: Mapping[str, Any], *, semantics: str) -> str:
    """Select the text that defines the score without cross-field fallback."""

    semantics = _validate_semantics(semantics)
    if not isinstance(record, Mapping):
        raise CLAPScoreValidationError("record must be a mapping")
    field = _TEXT_FIELDS[semantics]
    if field not in record:
        raise CLAPScoreValidationError(
            f"{field} is required for {semantics}; fallback to another caption is forbidden"
        )
    return _text(record[field], field)


def prepare_clap_score_inputs(
    records: Any,
    *,
    expected_ids: Sequence[str],
    semantics: str,
) -> tuple[CLAPScoreInput, ...]:
    """Build one explicit text/audio input for every expected sample."""

    semantics = _validate_semantics(semantics)
    expected = _validated_ids(expected_ids, "expected_ids")
    values = _record_sequence(records)
    parsed: list[CLAPScoreInput] = []
    for index, value in enumerate(values):
        if not isinstance(value, Mapping):
            raise CLAPScoreValidationError(f"records[{index}] must be a mapping")
        _require_success(value, index)
        sample_id = _identifier(value.get("sample_id"), f"records[{index}].sample_id")
        generated_id = _identifier(value.get("generated_id"), f"records[{index}].generated_id")
        if generated_id != sample_id:
            raise CLAPScoreValidationError(
                f"records[{index}] has mismatched IDs: sample={sample_id!r}, "
                f"generated={generated_id!r}"
            )
        parsed.append(
            CLAPScoreInput(
                sample_id=sample_id,
                generated_id=generated_id,
                semantics=semantics,
                text=select_clap_score_text(value, semantics=semantics),
            )
        )
    return _order_complete(tuple(parsed), expected, "CLAPScore inputs")


def validate_clap_score_records(
    records: Any,
    *,
    expected_ids: Sequence[str],
    semantics: str,
    official_corpus_score: Any | None = None,
) -> CLAPScoreAggregation:
    """Aggregate passed scores while requiring exact attribution for every attempt."""

    semantics = _validate_semantics(semantics)
    expected = _validated_ids(expected_ids, "expected_ids")
    values = _record_sequence(records)
    attempted = tuple(_attempted_score(value, index, semantics) for index, value in enumerate(values))
    _validate_attempted_ids(attempted, expected, "CLAPScore records")
    by_id = {sample_id: (record, error) for sample_id, record, error in attempted}
    ordered = tuple(by_id[value][0] for value in expected if by_id[value][0] is not None)
    failed_ids = tuple(value for value in expected if by_id[value][1] is not None)
    corpus_mean = sum(record.score for record in ordered) / len(ordered) if ordered else None
    if official_corpus_score is not None:
        if corpus_mean is None:
            raise CLAPScoreValidationError("official CLAPScore corpus score cannot be provided when all samples failed")
        official = _score(official_corpus_score, "official CLAPScore corpus score")
        if not math.isclose(official, corpus_mean, rel_tol=1e-7, abs_tol=1e-12):
            raise CLAPScoreValidationError(
                f"official CLAPScore corpus score {official} does not match passed sample mean {corpus_mean}"
            )

    status = "passed" if not failed_ids else ("passed_with_errors" if ordered else "error")
    metadata = official_clap_score_constraints() | {
        "metric": _METRIC_NAMES[semantics],
        "status": status,
        "attempted_count": len(expected),
        "passed_count": len(ordered),
        "error_count": len(failed_ids),
        "error_sample_ids": failed_ids,
        "semantics": semantics,
        "selected_text_field": _TEXT_FIELDS[semantics],
        "target_caption_is_prompt_adherence": False,
        "expected_sample_count": len(expected),
        "completed_sample_count": len(ordered),
        "failed_sample_count": len(failed_ids),
        "sample_ids_validated": True,
    }
    return CLAPScoreAggregation(ordered, corpus_mean, len(ordered), len(expected), metadata)


aggregate_clap_score_records = validate_clap_score_records


def _score_record(value: Any, index: int, semantics: str) -> CLAPScoreRecord:
    if isinstance(value, CLAPScoreRecord):
        record = value
    elif isinstance(value, Mapping):
        record = CLAPScoreRecord(
            sample_id=_identifier(value.get("sample_id"), f"records[{index}].sample_id"),
            generated_id=_identifier(value.get("generated_id"), f"records[{index}].generated_id"),
            semantics=_validate_semantics(value.get("semantics")),
            text=_text(value.get("text"), f"records[{index}].text"),
            score=_score(value.get("score"), f"records[{index}].score"),
        )
    else:
        raise CLAPScoreValidationError(f"records[{index}] must be a mapping or CLAPScoreRecord")

    sample_id = _identifier(record.sample_id, f"records[{index}].sample_id")
    generated_id = _identifier(record.generated_id, f"records[{index}].generated_id")
    record_semantics = _validate_semantics(record.semantics)
    text = _text(record.text, f"records[{index}].text")
    score = _score(record.score, f"records[{index}].score")
    if generated_id != sample_id:
        raise CLAPScoreValidationError(
            f"records[{index}] has mismatched IDs: sample={sample_id!r}, generated={generated_id!r}"
        )
    if record_semantics != semantics:
        raise CLAPScoreValidationError(
            f"records[{index}] semantics {record_semantics!r} does not match requested semantics {semantics!r}"
        )
    return CLAPScoreRecord(sample_id, generated_id, record_semantics, text, score)


def _attempted_score(value: Any, index: int, semantics: str) -> tuple[str, CLAPScoreRecord | None, str | None]:
    if not isinstance(value, Mapping):
        record = _score_record(value, index, semantics)
        return record.sample_id, record, None
    sample_id = _identifier(value.get("sample_id"), f"records[{index}].sample_id")
    generated_id = value.get("generated_id")
    if generated_id is not None and _identifier(generated_id, f"records[{index}].generated_id") != sample_id:
        raise CLAPScoreValidationError(
            f"records[{index}] has mismatched IDs: sample={sample_id!r}, generated={generated_id!r}"
        )
    status, error = value.get("status", "success"), value.get("error")
    if status != "success" or error not in (None, ""):
        return sample_id, None, str(error or f"status={status!r}")
    return sample_id, _score_record(value, index, semantics), None


def _validate_attempted_ids(attempted: Sequence[tuple[str, Any, str | None]], expected: tuple[str, ...], label: str) -> None:
    actual_ids = tuple(sample_id for sample_id, _, _ in attempted)
    duplicates = _duplicates(actual_ids)
    if duplicates:
        raise CLAPScoreValidationError(f"duplicate sample IDs: {duplicates}")
    expected_set, actual_set = set(expected), set(actual_ids)
    missing = tuple(value for value in expected if value not in actual_set)
    extra = tuple(value for value in actual_ids if value not in expected_set)
    if missing or extra or len(attempted) != len(expected):
        raise CLAPScoreValidationError(
            f"{label} do not exactly match expected IDs; missing={missing}, extra={extra}, "
            f"expected_count={len(expected)}, actual_count={len(attempted)}"
        )


def _record_sequence(records: Any) -> Sequence[Any]:
    if isinstance(records, (str, bytes, Mapping)) or not isinstance(records, Sequence):
        raise CLAPScoreValidationError("records must be a sequence")
    return records


def _require_success(value: Mapping[str, Any], index: int) -> None:
    status, error = value.get("status", "success"), value.get("error")
    if status != "success" or error not in (None, ""):
        identifier = value.get("sample_id", index)
        raise CLAPScoreValidationError(
            f"sample {identifier!r} failed: status={status!r}, error={error!r}"
        )


def _order_complete(
    records: Sequence[Any],
    expected: tuple[str, ...],
    label: str,
) -> tuple[Any, ...]:
    actual_ids = tuple(record.sample_id for record in records)
    duplicates = _duplicates(actual_ids)
    if duplicates:
        raise CLAPScoreValidationError(f"duplicate sample IDs: {duplicates}")
    expected_set, actual_set = set(expected), set(actual_ids)
    missing = tuple(value for value in expected if value not in actual_set)
    extra = tuple(value for value in actual_ids if value not in expected_set)
    if missing or extra or len(records) != len(expected):
        raise CLAPScoreValidationError(
            f"{label} do not exactly match expected IDs; missing={missing}, extra={extra}, "
            f"expected_count={len(expected)}, actual_count={len(records)}"
        )
    by_id = {record.sample_id: record for record in records}
    return tuple(by_id[value] for value in expected)


def _validated_ids(values: Any, name: str) -> tuple[str, ...]:
    if isinstance(values, (str, bytes)) or not isinstance(values, Sequence) or not values:
        raise CLAPScoreValidationError(f"{name} must be a non-empty sequence")
    identifiers = tuple(_identifier(value, f"{name}[{index}]") for index, value in enumerate(values))
    duplicates = _duplicates(identifiers)
    if duplicates:
        raise CLAPScoreValidationError(f"{name} contains duplicate IDs: {duplicates}")
    return identifiers


def _identifier(value: Any, name: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise CLAPScoreValidationError(f"{name} must be a non-empty string")
    return value


def _text(value: Any, name: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise CLAPScoreValidationError(f"{name} must be a non-empty string")
    return value


def _score(value: Any, name: str) -> float:
    if hasattr(value, "item"):
        try:
            value = value.item()
        except (RuntimeError, TypeError, ValueError) as exc:
            raise CLAPScoreValidationError(f"{name} must be a scalar") from exc
    if not isinstance(value, Real) or isinstance(value, bool):
        raise CLAPScoreValidationError(f"{name} must be a real numeric scalar")
    result = float(value)
    if not math.isfinite(result):
        raise CLAPScoreValidationError(f"{name} must be finite")
    if not -1.0 <= result <= 1.0:
        raise CLAPScoreValidationError(f"{name} must be a cosine similarity in [-1, 1]")
    return result


def _validate_semantics(value: Any) -> str:
    if value not in _SEMANTICS:
        raise CLAPScoreValidationError(
            f"semantics must be {TARGET_CAPTION_ALIGNMENT!r} or {PROMPT_ADHERENCE!r}"
        )
    return value


def _duplicates(values: Sequence[str]) -> tuple[str, ...]:
    seen: set[str] = set()
    duplicates: list[str] = []
    for value in values:
        if value in seen and value not in duplicates:
            duplicates.append(value)
        seen.add(value)
    return tuple(duplicates)
