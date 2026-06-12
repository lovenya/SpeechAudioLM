"""Strict semantic helpers for the official stable-audio-metrics KL-PaSST."""

from __future__ import annotations

import math
from dataclasses import dataclass
from numbers import Real
from typing import Any, Mapping, Sequence


STABLE_AUDIO_METRICS_COMMIT = "fd55536cc812c460ecc421220864993c7f168184"
HEAR21PASST_VERSION = "0.0.16"
OFFICIAL_REFERENCE_EPSILON = 1e-6
KL_DIRECTION = "KL(P_eval || P_reference_plus_1e-6)"


class KLPaSSTValidationError(ValueError):
    """Raised when KL-PaSST output cannot represent the complete corpus."""


@dataclass(frozen=True)
class KLPaSSTPairRecord:
    sample_id: str
    reference_id: str
    generated_id: str
    kl_eval_to_reference: float


@dataclass(frozen=True)
class KLPaSSTAggregation:
    records: tuple[KLPaSSTPairRecord, ...]
    corpus_mean: float
    pair_count: int
    expected_pair_count: int
    metadata: Mapping[str, Any]


def official_passt_preprocessing(*, collect: str = "mean") -> dict[str, Any]:
    """Describe behavior observed in the installed official implementation."""

    collect = _validate_collect(collect)
    return {
        "implementation": "stable-audio-metrics",
        "implementation_commit": STABLE_AUDIO_METRICS_COMMIT,
        "audio_encoder": "PaSST AudioSet classifier",
        "audio_encoder_package": "hear21passt",
        "audio_encoder_package_version": HEAR21PASST_VERSION,
        "class_count": 527,
        "sample_rate_hz": 32_000,
        "channel_policy": "librosa.load(..., mono=True)",
        "resampling_policy": "librosa.load(..., sr=32000)",
        "amplitude_policy": "pyloudnorm peak normalization to -1.0 dBFS",
        "window_seconds": 10,
        "overlap_seconds": 5,
        "window_step_seconds": 5,
        "short_window_policy": (
            "zero-pad tails longer than 15% of 10 seconds; pass shorter tails "
            "through unpadded as implemented upstream"
        ),
        "window_collect": collect,
        "window_collect_semantics": f"{collect} over window logits, then softmax",
        "probability_normalization": "softmax over 527 AudioSet logits",
        "reference_epsilon": OFFICIAL_REFERENCE_EPSILON,
        "kl_direction": KL_DIRECTION,
        "pair_aggregation": "arithmetic mean over all expected pairs",
        "backend_partial_failure_behavior": "upstream skips failed pairs; project validation rejects them",
        "requires_cuda": True,
    }


def collect_window_probabilities(window_logits: Any, *, collect: str = "mean") -> tuple[float, ...]:
    """Pool window logits as upstream does, then apply softmax."""

    collect = _validate_collect(collect)
    rows = _numeric_matrix(window_logits, "window_logits")
    if collect == "mean":
        pooled = tuple(sum(items) / len(items) for items in zip(*rows, strict=True))
    else:
        pooled = tuple(max(items) for items in zip(*rows, strict=True))
    maximum = max(pooled)
    exponentials = tuple(math.exp(value - maximum) for value in pooled)
    denominator = sum(exponentials)
    return tuple(value / denominator for value in exponentials)


def pair_kl_passt(
    reference_probabilities: Any,
    eval_probabilities: Any,
    *,
    epsilon: Real = OFFICIAL_REFERENCE_EPSILON,
) -> float:
    """Compute the official pair value, explicitly KL(eval || reference)."""

    reference = _probability_vector(reference_probabilities, "reference_probabilities")
    evaluated = _probability_vector(eval_probabilities, "eval_probabilities")
    if len(reference) != len(evaluated):
        raise KLPaSSTValidationError(
            f"probability dimensions differ: reference={len(reference)}, eval={len(evaluated)}"
        )
    epsilon_value = _finite_scalar(epsilon, "epsilon")
    if epsilon_value <= 0.0:
        raise KLPaSSTValidationError("epsilon must be positive")
    value = sum(
        target * (math.log(target) - math.log(source + epsilon_value))
        for source, target in zip(reference, evaluated, strict=True)
        if target > 0.0
    )
    if not math.isfinite(value):
        raise KLPaSSTValidationError("pair KL must be finite")
    return value


def validate_kl_passt_records(
    records: Any,
    *,
    expected_ids: Sequence[str],
    collect: str = "mean",
    official_corpus_score: Any | None = None,
) -> KLPaSSTAggregation:
    """Require one successful finite pair record for every expected ID."""

    collect = _validate_collect(collect)
    expected = _validated_ids(expected_ids, "expected_ids")
    if isinstance(records, (str, bytes, Mapping)) or not isinstance(records, Sequence):
        raise KLPaSSTValidationError("records must be a sequence of per-pair records")
    if not records:
        raise KLPaSSTValidationError("all KL-PaSST pairs failed or no records were returned")

    parsed = tuple(_pair_record(value, index) for index, value in enumerate(records))
    actual_ids = tuple(record.sample_id for record in parsed)
    duplicates = _duplicates(actual_ids)
    if duplicates:
        raise KLPaSSTValidationError(f"duplicate pair IDs: {duplicates}")
    expected_set, actual_set = set(expected), set(actual_ids)
    missing = tuple(value for value in expected if value not in actual_set)
    extra = tuple(value for value in actual_ids if value not in expected_set)
    if missing or extra or len(parsed) != len(expected):
        raise KLPaSSTValidationError(
            "KL-PaSST pair IDs do not exactly match expected IDs; "
            f"missing={missing}, extra={extra}, expected_count={len(expected)}, actual_count={len(parsed)}"
        )

    by_id = {record.sample_id: record for record in parsed}
    ordered = tuple(by_id[value] for value in expected)
    corpus_mean = sum(record.kl_eval_to_reference for record in ordered) / len(ordered)
    if not math.isfinite(corpus_mean):
        raise KLPaSSTValidationError("KL-PaSST corpus mean must be finite")
    if official_corpus_score is not None:
        official = _finite_scalar(official_corpus_score, "official KL-PaSST corpus score")
        if not math.isclose(official, corpus_mean, rel_tol=1e-7, abs_tol=1e-12):
            raise KLPaSSTValidationError(
                f"official KL-PaSST corpus score {official} does not match complete pair mean {corpus_mean}"
            )

    metadata = official_passt_preprocessing(collect=collect) | {
        "metric": "kl_passt",
        "expected_pair_count": len(expected),
        "completed_pair_count": len(ordered),
        "failed_pair_count": 0,
        "pair_ids_validated": True,
        "partial_results_accepted": False,
    }
    return KLPaSSTAggregation(ordered, corpus_mean, len(ordered), len(expected), metadata)


aggregate_kl_passt_pairs = validate_kl_passt_records


def _pair_record(value: Any, index: int) -> KLPaSSTPairRecord:
    if isinstance(value, KLPaSSTPairRecord):
        sample_id, reference_id, generated_id, score = (
            value.sample_id,
            value.reference_id,
            value.generated_id,
            value.kl_eval_to_reference,
        )
    elif isinstance(value, Mapping):
        status, error = value.get("status", "success"), value.get("error")
        if status != "success" or error not in (None, ""):
            identifier = value.get("sample_id", value.get("id", index))
            raise KLPaSSTValidationError(f"pair {identifier!r} failed: status={status!r}, error={error!r}")
        sample_id = value.get("sample_id", value.get("id"))
        reference_id = value.get("reference_id", sample_id)
        generated_id = value.get("generated_id", sample_id)
        score = value.get("kl_eval_to_reference", value.get("kl"))
    else:
        raise KLPaSSTValidationError(f"records[{index}] must be a mapping or KLPaSSTPairRecord")

    sample_id = _identifier(sample_id, f"records[{index}].sample_id")
    reference_id = _identifier(reference_id, f"records[{index}].reference_id")
    generated_id = _identifier(generated_id, f"records[{index}].generated_id")
    score = _finite_scalar(score, f"records[{index}].kl_eval_to_reference")
    if reference_id != sample_id or generated_id != sample_id:
        raise KLPaSSTValidationError(
            f"records[{index}] has mismatched IDs: sample={sample_id!r}, "
            f"reference={reference_id!r}, generated={generated_id!r}"
        )
    return KLPaSSTPairRecord(sample_id, reference_id, generated_id, score)


def _validated_ids(values: Any, name: str) -> tuple[str, ...]:
    if isinstance(values, (str, bytes)) or not isinstance(values, Sequence) or not values:
        raise KLPaSSTValidationError(f"{name} must be a non-empty sequence")
    identifiers = tuple(_identifier(value, f"{name}[{index}]") for index, value in enumerate(values))
    duplicates = _duplicates(identifiers)
    if duplicates:
        raise KLPaSSTValidationError(f"{name} contains duplicate IDs: {duplicates}")
    return identifiers


def _identifier(value: Any, name: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise KLPaSSTValidationError(f"{name} must be a non-empty string")
    return value


def _duplicates(values: Sequence[str]) -> tuple[str, ...]:
    seen: set[str] = set()
    duplicates: list[str] = []
    for value in values:
        if value in seen and value not in duplicates:
            duplicates.append(value)
        seen.add(value)
    return tuple(duplicates)


def _validate_collect(value: Any) -> str:
    if value not in {"mean", "max"}:
        raise KLPaSSTValidationError("collect must be 'mean' or 'max'")
    return value


def _numeric_matrix(value: Any, name: str) -> tuple[tuple[float, ...], ...]:
    if hasattr(value, "tolist"):
        value = value.tolist()
    if isinstance(value, (str, bytes, Mapping)) or not isinstance(value, Sequence) or not value:
        raise KLPaSSTValidationError(f"{name} must be a non-empty two-dimensional sequence")
    rows = tuple(_numeric_vector(row, f"{name}[{index}]") for index, row in enumerate(value))
    if any(len(row) != len(rows[0]) for row in rows):
        raise KLPaSSTValidationError(f"{name} rows must have equal dimensions")
    return rows


def _numeric_vector(value: Any, name: str) -> tuple[float, ...]:
    if hasattr(value, "tolist"):
        value = value.tolist()
    if isinstance(value, (str, bytes, Mapping)) or not isinstance(value, Sequence) or not value:
        raise KLPaSSTValidationError(f"{name} must be a non-empty one-dimensional sequence")
    return tuple(_finite_scalar(item, f"{name}[{index}]") for index, item in enumerate(value))


def _probability_vector(value: Any, name: str) -> tuple[float, ...]:
    probabilities = _numeric_vector(value, name)
    if any(item < 0.0 for item in probabilities):
        raise KLPaSSTValidationError(f"{name} cannot contain negative values")
    total = sum(probabilities)
    if not math.isclose(total, 1.0, rel_tol=1e-6, abs_tol=1e-8):
        raise KLPaSSTValidationError(f"{name} must sum to 1, found {total}")
    return probabilities


def _finite_scalar(value: Any, name: str) -> float:
    if hasattr(value, "item"):
        try:
            value = value.item()
        except (RuntimeError, TypeError, ValueError) as exc:
            raise KLPaSSTValidationError(f"{name} must be a scalar") from exc
    if not isinstance(value, Real) or isinstance(value, bool):
        raise KLPaSSTValidationError(f"{name} must be a real numeric scalar")
    result = float(value)
    if not math.isfinite(result):
        raise KLPaSSTValidationError(f"{name} must be finite")
    return result
