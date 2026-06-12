"""Scientific policy and strict result validation for official UTMOSv2."""

from __future__ import annotations

import math
from dataclasses import dataclass
from numbers import Real
from typing import Any, Mapping, Sequence


UTMOSV2_PACKAGE_VERSION = "1.3.1.dev0"
UTMOSV2_SOURCE_COMMIT = "cc2700db57bb83ee13dc31ebe1b868c254e15d09"
UTMOSV2_CONFIG = "fusion_stage3"
UTMOSV2_CHECKPOINT_FOLD = 0
UTMOSV2_CHECKPOINT_SEED = 42
UTMOSV2_CHECKPOINT_FILENAME = "fold0_s42_best_model.pth"
UTMOSV2_SAMPLE_RATE_HZ = 16_000
UTMOSV2_NUM_REPETITIONS = 5
UTMOSV2_PREDICT_DATASET = "sarulab"


class UTMOSValidationError(ValueError):
    """Raised when UTMOS policy or output is scientifically invalid."""


@dataclass(frozen=True)
class UTMOSRecord:
    sample_id: str
    score: float


@dataclass(frozen=True)
class UTMOSAggregation:
    records: tuple[UTMOSRecord, ...]
    corpus_mean: float
    sample_count: int
    expected_sample_count: int
    metadata: Mapping[str, Any]


def official_utmosv2_policy() -> dict[str, Any]:
    """Describe the paper-comparable official ``fusion_stage3`` behavior.

    UTMOSv2's user-facing API defaults to silence removal, but its source states
    that this feature was added after the VoiceMOS 2024 and paper experiments.
    Scientific evaluation therefore disables only that optional post-release
    transform and preserves the model's native stochastic segmentation.
    """

    return {
        "metric": "utmos",
        "implementation": "official sarulab-speech/UTMOSv2",
        "implementation_version": UTMOSV2_PACKAGE_VERSION,
        "implementation_commit": UTMOSV2_SOURCE_COMMIT,
        "config": UTMOSV2_CONFIG,
        "sample_rate_hz": UTMOSV2_SAMPLE_RATE_HZ,
        "channel_policy": "librosa.load default mono=True for file input",
        "resampling_policy": (
            "load at native rate with librosa.load(sr=None), then librosa.resample "
            "to 16000 Hz"
        ),
        "amplitude_normalization": "none",
        "silence_removal": False,
        "silence_removal_policy": (
            "disabled; optional threshold-based removal was added after the "
            "VoiceMOS 2024 and paper experiments"
        ),
        "ssl_segment_seconds": 3.0,
        "ssl_segment_policy": (
            "tile clips at or below 3.0 seconds, then select one random contiguous "
            "3.0-second segment per repetition"
        ),
        "spectrogram_frame_seconds": 1.4,
        "spectrogram_frame_count": 2,
        "spectrogram_segment_policy": (
            "tile clips at or below 1.4 seconds; for each of two frames select a "
            "random 1.4-second segment"
        ),
        "spectrogram_inner_mixup": True,
        "spectrogram_inner_mixup_policy": (
            "for every frame and each of four mel configurations, mix the selected "
            "segment with another random 1.4-second segment using Beta(0.4, 0.4)"
        ),
        "crop_policy": "preserve official random segment selection; no project crop",
        "padding_policy": "official waveform tiling only; no zero-padding",
        "truncation_policy": "none outside official random segment selection",
        "num_repetitions": UTMOSV2_NUM_REPETITIONS,
        "repetition_aggregation": "arithmetic mean of five stochastic predictions",
        "random_seed_policy": (
            "official prediction API does not seed NumPy; record the externally "
            "controlled NumPy RNG seed and execution provenance"
        ),
        "checkpoint": {
            "pretrained": True,
            "config": UTMOSV2_CONFIG,
            "fold": UTMOSV2_CHECKPOINT_FOLD,
            "training_seed": UTMOSV2_CHECKPOINT_SEED,
            "filename": UTMOSV2_CHECKPOINT_FILENAME,
            "repository": "sarulab-speech/UTMOSv2 on Hugging Face",
            "custom_checkpoint_allowed": False,
        },
        "predict_dataset": UTMOSV2_PREDICT_DATASET,
        "metric_level": "per-sample",
        "higher_is_better": True,
        "corpus_aggregation": "arithmetic mean over the complete expected sample set",
        "partial_results_accepted": False,
        "nonfinite_results_accepted": False,
    }


def scientific_prediction_kwargs() -> dict[str, Any]:
    """Return mandatory official prediction options for a scientific run."""

    return {
        "predict_dataset": UTMOSV2_PREDICT_DATASET,
        "num_repetitions": UTMOSV2_NUM_REPETITIONS,
        "remove_silent_section": False,
    }


def validate_prediction_policy(
    *,
    config: Any,
    fold: Any,
    checkpoint_seed: Any,
    pretrained: Any,
    checkpoint_path: Any,
    predict_dataset: Any,
    num_repetitions: Any,
    remove_silent_section: Any,
) -> None:
    """Reject settings that depart from the selected official scientific policy."""

    expected = {
        "config": UTMOSV2_CONFIG,
        "fold": UTMOSV2_CHECKPOINT_FOLD,
        "checkpoint_seed": UTMOSV2_CHECKPOINT_SEED,
        "pretrained": True,
        "checkpoint_path": None,
        "predict_dataset": UTMOSV2_PREDICT_DATASET,
        "num_repetitions": UTMOSV2_NUM_REPETITIONS,
        "remove_silent_section": False,
    }
    actual = {
        "config": config,
        "fold": fold,
        "checkpoint_seed": checkpoint_seed,
        "pretrained": pretrained,
        "checkpoint_path": checkpoint_path,
        "predict_dataset": predict_dataset,
        "num_repetitions": num_repetitions,
        "remove_silent_section": remove_silent_section,
    }
    mismatches = tuple(
        f"{name}={actual[name]!r} (required {required!r})"
        for name, required in expected.items()
        if actual[name] != required
    )
    if mismatches:
        raise UTMOSValidationError(
            "UTMOSv2 settings depart from the paper-comparable official policy: "
            + ", ".join(mismatches)
        )


def validate_utmos_records(
    records: Any,
    *,
    expected_ids: Sequence[str],
    official_corpus_score: Any | None = None,
) -> UTMOSAggregation:
    """Require exactly one successful finite UTMOS value per expected sample."""

    expected = _validated_ids(expected_ids)
    if isinstance(records, (str, bytes, Mapping)) or not isinstance(records, Sequence):
        raise UTMOSValidationError("records must be a sequence of per-sample records")
    if not records:
        raise UTMOSValidationError("all UTMOS samples failed or no records were returned")

    parsed = tuple(_record(value, index) for index, value in enumerate(records))
    actual_ids = tuple(record.sample_id for record in parsed)
    duplicates = _duplicates(actual_ids)
    if duplicates:
        raise UTMOSValidationError(f"duplicate UTMOS sample IDs: {duplicates}")

    expected_set, actual_set = set(expected), set(actual_ids)
    missing = tuple(value for value in expected if value not in actual_set)
    extra = tuple(value for value in actual_ids if value not in expected_set)
    if missing or extra or len(parsed) != len(expected):
        raise UTMOSValidationError(
            "UTMOS sample IDs do not exactly match expected IDs; "
            f"missing={missing}, extra={extra}, expected_count={len(expected)}, "
            f"actual_count={len(parsed)}"
        )

    by_id = {record.sample_id: record for record in parsed}
    ordered = tuple(by_id[sample_id] for sample_id in expected)
    corpus_mean = sum(record.score for record in ordered) / len(ordered)
    if not math.isfinite(corpus_mean):
        raise UTMOSValidationError("UTMOS corpus mean must be finite")
    if official_corpus_score is not None:
        official = _finite_scalar(official_corpus_score, "official UTMOS corpus score")
        if not math.isclose(official, corpus_mean, rel_tol=1e-7, abs_tol=1e-12):
            raise UTMOSValidationError(
                f"official UTMOS corpus score {official} does not match "
                f"complete sample mean {corpus_mean}"
            )

    metadata = official_utmosv2_policy() | {
        "expected_sample_count": len(expected),
        "completed_sample_count": len(ordered),
        "failed_sample_count": 0,
        "sample_ids_validated": True,
    }
    return UTMOSAggregation(ordered, corpus_mean, len(ordered), len(expected), metadata)


aggregate_utmos_samples = validate_utmos_records


def _record(value: Any, index: int) -> UTMOSRecord:
    if isinstance(value, UTMOSRecord):
        sample_id, score = value.sample_id, value.score
    elif isinstance(value, Mapping):
        status, error = value.get("status", "success"), value.get("error")
        if status != "success" or error not in (None, ""):
            identifier = value.get("sample_id", value.get("id", index))
            raise UTMOSValidationError(
                f"UTMOS sample {identifier!r} failed: status={status!r}, error={error!r}"
            )
        sample_id = value.get("sample_id", value.get("id"))
        score = value.get("utmos", value.get("score"))
    else:
        raise UTMOSValidationError(
            f"records[{index}] must be a mapping or UTMOSRecord"
        )
    return UTMOSRecord(
        _identifier(sample_id, f"records[{index}].sample_id"),
        _finite_scalar(score, f"records[{index}].utmos"),
    )


def _validated_ids(values: Any) -> tuple[str, ...]:
    if isinstance(values, (str, bytes)) or not isinstance(values, Sequence) or not values:
        raise UTMOSValidationError("expected_ids must be a non-empty sequence")
    identifiers = tuple(
        _identifier(value, f"expected_ids[{index}]")
        for index, value in enumerate(values)
    )
    duplicates = _duplicates(identifiers)
    if duplicates:
        raise UTMOSValidationError(f"expected_ids contains duplicate IDs: {duplicates}")
    return identifiers


def _identifier(value: Any, name: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise UTMOSValidationError(f"{name} must be a non-empty string")
    return value


def _duplicates(values: Sequence[str]) -> tuple[str, ...]:
    seen: set[str] = set()
    duplicates: list[str] = []
    for value in values:
        if value in seen and value not in duplicates:
            duplicates.append(value)
        seen.add(value)
    return tuple(duplicates)


def _finite_scalar(value: Any, name: str) -> float:
    if hasattr(value, "item"):
        try:
            value = value.item()
        except (RuntimeError, TypeError, ValueError) as exc:
            raise UTMOSValidationError(f"{name} must be a scalar") from exc
    if not isinstance(value, Real) or isinstance(value, bool):
        raise UTMOSValidationError(f"{name} must be a real numeric scalar")
    result = float(value)
    if not math.isfinite(result):
        raise UTMOSValidationError(f"{name} must be finite")
    return result
