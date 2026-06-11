"""Strict reconstruction helpers for AAC Metrics 0.5.5 audio metrics.

AAC Metrics 0.5.5 returns the corpus MACE scalar in the sentence-score
mapping.  Per-sample MACE values must therefore be reconstructed from the
sentence-level CLAP similarity and FER vectors using the upstream formula.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from numbers import Integral, Real
from typing import Any, Mapping, Sequence


AAC_METRICS_VERSION = "0.5.5"
MS_CLAP_VERSION = "1.3.4"
MS_CLAP_2023_SAMPLE_RATE_HZ = 44_100
MS_CLAP_2023_CLIP_DURATION_SECONDS = 7
MS_CLAP_2023_TEXT_CONTEXT_TOKENS = 77
PROJECT_EVALUATION_SEED = 1337


class AACAudioValidationError(ValueError):
    """Raised when AAC audio metric outputs cannot be used exactly."""


@dataclass(frozen=True)
class SampleReconstruction:
    """Validated sample scores, their exact arithmetic mean, and policy data."""

    scores: tuple[float, ...]
    corpus_mean: float
    metadata: Mapping[str, Any]


def ms_clap_2023_backend_behavior(
    *, seed: int | None = PROJECT_EVALUATION_SEED
) -> dict[str, Any]:
    """Describe observed upstream behavior, not an approved project policy.

    The values describe AAC Metrics 0.5.5 with msclap 1.3.4.  In particular,
    the seed is set once before the ordered audio list is encoded; it is not a
    per-sample seed.
    """

    if seed is not None and (not isinstance(seed, Integral) or isinstance(seed, bool)):
        raise AACAudioValidationError("seed must be an integer or None")
    seed_value = int(seed) if seed is not None else None
    return {
        "implementation": "aac-metrics",
        "implementation_version": AAC_METRICS_VERSION,
        "audio_encoder": "MS-CLAP-2023",
        "audio_encoder_package": "msclap",
        "audio_encoder_package_version": MS_CLAP_VERSION,
        "sample_rate_hz": MS_CLAP_2023_SAMPLE_RATE_HZ,
        "clip_duration_seconds": MS_CLAP_2023_CLIP_DURATION_SECONDS,
        "channel_policy": "torchaudio channels are flattened before encoding",
        "short_audio_policy": "repeat waveform from the start, then truncate to 7 seconds",
        "long_audio_policy": "one contiguous 7-second crop selected with Python random.randrange",
        "crop_seed": seed_value,
        "project_evaluation_seed": PROJECT_EVALUATION_SEED,
        "crop_seed_scope": "set once per AAC Metrics invocation before encoding the ordered audio list",
        "order_invariant": False,
        "crop_offset_stream_internal_batch_size_invariant": True,
        "model_score_internal_batch_size_invariant": "not_claimed",
        "separate_invocation_partition_invariant": False,
        "crop_offsets_recorded": False,
        "scientifically_approved_project_policy": True,
        "project_policy_status": "official_seeded_backend_cropping_accepted_and_disclosed",
        "text_context_tokens": MS_CLAP_2023_TEXT_CONTEXT_TOKENS,
        "text_overflow_policy": "no explicit tokenizer truncation in msclap 1.3.4",
    }


def validate_clap_audio_samples(
    sentence_scores: Any,
    *,
    expected_count: int,
    corpus_score: Any | None = None,
    seed: int | None = PROJECT_EVALUATION_SEED,
) -> SampleReconstruction:
    """Validate official per-sample audio CLAP similarities without reshaping."""

    scores = _score_vector(
        _mapping_value(sentence_scores, "clap_sim"),
        name="CLAP sentence scores",
        expected_count=expected_count,
        minimum=-1.0,
        maximum=1.0,
    )
    corpus_mean = _mean(scores)
    if corpus_score is not None:
        _require_matching_corpus_score(corpus_score, corpus_mean, "CLAP")
    metadata = ms_clap_2023_backend_behavior(seed=seed) | {
        "metric": "aac_clapsim",
        "sample_reconstruction": "official sentence-level clap_sim vector",
        "corpus_aggregation": "arithmetic mean over the complete sample vector",
    }
    return SampleReconstruction(scores, corpus_mean, metadata)


def reconstruct_combined_clap_samples(
    text_sentence_scores: Any,
    audio_sentence_scores: Any,
    *,
    expected_count: int,
    corpus_score: Any | None = None,
    seed: int | None = PROJECT_EVALUATION_SEED,
) -> SampleReconstruction:
    """Reconstruct combined MACE CLAP values by pairing samples exactly.

    AAC Metrics combines the text and audio CLAP vectors elementwise.  Combining
    their corpus means cannot recover sample values and is therefore rejected by
    this API, which requires both complete vectors.
    """

    text_scores = _score_vector(
        _mapping_value(text_sentence_scores, "clap_sim"),
        name="text CLAP sentence scores",
        expected_count=expected_count,
        minimum=-1.0,
        maximum=1.0,
    )
    audio_scores = _score_vector(
        _mapping_value(audio_sentence_scores, "clap_sim"),
        name="audio CLAP sentence scores",
        expected_count=expected_count,
        minimum=-1.0,
        maximum=1.0,
    )
    scores = tuple((text + audio) / 2.0 for text, audio in zip(text_scores, audio_scores, strict=True))
    corpus_mean = _mean(scores)
    if corpus_score is not None:
        _require_matching_corpus_score(corpus_score, corpus_mean, "combined CLAP")
    metadata = ms_clap_2023_backend_behavior(seed=seed) | {
        "metric": "mace_combined_clap_component",
        "sample_reconstruction": "elementwise mean of paired text and audio CLAP similarities",
        "corpus_aggregation": "arithmetic mean only after per-sample pairing",
    }
    return SampleReconstruction(scores, corpus_mean, metadata)


def reconstruct_mace_samples(
    sentence_scores: Mapping[str, Any],
    *,
    expected_count: int,
    penalty: Real = 0.3,
    corpus_score: Any | None = None,
    mace_method: str = "combined",
    seed: int | None = PROJECT_EVALUATION_SEED,
) -> SampleReconstruction:
    """Reconstruct exact per-sample MACE values from CLAP and FER outputs."""

    if not isinstance(sentence_scores, Mapping):
        raise AACAudioValidationError("MACE sentence scores must be a mapping")
    if mace_method not in {"text", "audio", "combined"}:
        raise AACAudioValidationError("mace_method must be 'text', 'audio', or 'combined'")
    penalty_value = _finite_scalar(penalty, "MACE penalty")
    if not 0.0 <= penalty_value <= 1.0:
        raise AACAudioValidationError("MACE penalty must be in [0, 1]")

    clap_scores = _score_vector(
        _mapping_value(sentence_scores, "clap_sim"),
        name="MACE CLAP sentence scores",
        expected_count=expected_count,
        minimum=-1.0,
        maximum=1.0,
    )
    fer_scores = _score_vector(
        _mapping_value(sentence_scores, "fer"),
        name="MACE FER sentence scores",
        expected_count=expected_count,
        allowed_values=(0.0, 1.0),
    )
    scores = tuple(
        clap * (1.0 - penalty_value * fer)
        for clap, fer in zip(clap_scores, fer_scores, strict=True)
    )
    corpus_mean = _mean(scores)
    if corpus_score is not None:
        _require_matching_corpus_score(corpus_score, corpus_mean, "MACE")

    metadata: dict[str, Any] = {
        "implementation": "aac-metrics",
        "implementation_version": AAC_METRICS_VERSION,
        "metric": "mace",
        "mace_method": mace_method,
        "penalty": penalty_value,
        "fer_values": "binary thresholded fluency-error indicators",
        "sample_reconstruction": "clap_sim_i * (1 - penalty * fer_i)",
        "backend_sentence_mace_policy": "ignore AAC Metrics 0.5.5 sentence 'mace'; it is the corpus scalar",
        "corpus_aggregation": "arithmetic mean of reconstructed sample MACE values",
    }
    if mace_method in {"audio", "combined"}:
        metadata.update(ms_clap_2023_backend_behavior(seed=seed))
        metadata["metric"] = "mace"
        metadata["mace_method"] = mace_method
        metadata["penalty"] = penalty_value
    return SampleReconstruction(scores, corpus_mean, metadata)


def _mapping_value(value: Any, key: str) -> Any:
    if isinstance(value, Mapping):
        if key not in value:
            raise AACAudioValidationError(f"missing required sentence score '{key}'")
        return value[key]
    return value


def _score_vector(
    values: Any,
    *,
    name: str,
    expected_count: int,
    minimum: float | None = None,
    maximum: float | None = None,
    allowed_values: tuple[float, ...] | None = None,
) -> tuple[float, ...]:
    if not isinstance(expected_count, Integral) or isinstance(expected_count, bool) or expected_count < 1:
        raise AACAudioValidationError("expected_count must be a positive integer")
    expected_count = int(expected_count)

    shape = getattr(values, "shape", None)
    if shape is not None and len(shape) != 1:
        raise AACAudioValidationError(f"{name} must be one-dimensional, found shape {tuple(shape)}")
    if hasattr(values, "tolist"):
        values = values.tolist()
    if isinstance(values, (str, bytes, Mapping)) or not isinstance(values, Sequence):
        raise AACAudioValidationError(f"{name} must be a complete one-dimensional sequence")
    if len(values) != expected_count:
        raise AACAudioValidationError(
            f"{name} has {len(values)} values; expected exactly {expected_count}; refusing duplication or truncation"
        )

    scores = tuple(_finite_scalar(value, f"{name}[{index}]") for index, value in enumerate(values))
    for index, score in enumerate(scores):
        if minimum is not None and score < minimum:
            raise AACAudioValidationError(f"{name}[{index}]={score} is below {minimum}")
        if maximum is not None and score > maximum:
            raise AACAudioValidationError(f"{name}[{index}]={score} is above {maximum}")
        if allowed_values is not None and score not in allowed_values:
            raise AACAudioValidationError(
                f"{name}[{index}]={score} is invalid; expected one of {allowed_values}"
            )
    return scores


def _finite_scalar(value: Any, name: str) -> float:
    if hasattr(value, "item"):
        try:
            value = value.item()
        except (RuntimeError, TypeError, ValueError) as exc:
            raise AACAudioValidationError(f"{name} must be a scalar") from exc
    if not isinstance(value, Real) or isinstance(value, bool):
        raise AACAudioValidationError(f"{name} must be a real numeric scalar")
    result = float(value)
    if not math.isfinite(result):
        raise AACAudioValidationError(f"{name} must be finite")
    return result


def _mean(scores: tuple[float, ...]) -> float:
    return math.fsum(scores) / len(scores)


def _require_matching_corpus_score(value: Any, reconstructed: float, metric: str) -> None:
    official = _finite_scalar(value, f"official {metric} corpus score")
    if not math.isclose(official, reconstructed, rel_tol=1e-7, abs_tol=1e-12):
        raise AACAudioValidationError(
            f"official {metric} corpus score {official} does not match reconstructed mean {reconstructed}"
        )
