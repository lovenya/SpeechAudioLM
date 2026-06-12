"""Reusable, serialization-friendly WER scoring for Whisper ASR outputs.

The primary aggregate is corpus WER, computed from summed edit counts. Macro
utterance WER is retained as a secondary diagnostic and must not be substituted
for corpus WER in reported results.
"""

from __future__ import annotations

from collections.abc import Iterable, Mapping, Sequence
from os import PathLike
from typing import Any

from evaluation.metrics.wer import (
    CONSERVATIVE_ENGLISH_V1,
    RAW_ORTHOGRAPHIC_V1,
    NormalizationPolicy,
    WerCounts,
    macro_utterance_wer,
    sum_wer_counts,
    utterance_wer_counts,
)


NORMALIZED_SCORE = "normalized"
RAW_SCORE = "raw"


def whisper_transcribe_kwargs(device: str) -> dict[str, Any]:
    """Return the fixed English transcription contract for OpenAI Whisper."""

    return {
        "language": "en",
        "task": "transcribe",
        "fp16": str(device).lower().startswith("cuda"),
    }


def transcribe_whisper(model: Any, audio_path: str | PathLike[str], device: str) -> str:
    """Transcribe one file under the fixed English Whisper contract."""

    result = model.transcribe(str(audio_path), **whisper_transcribe_kwargs(device))
    if not isinstance(result, Mapping):
        raise TypeError("Whisper transcribe() must return a mapping")
    return str(result.get("text", "")).strip()


def score_wer_pair(reference: str, hypothesis: str) -> dict[str, dict[str, Any]]:
    """Score one directional reference/hypothesis pair under both policies.

    Empty references, including references emptied by normalization, raise
    ``EmptyReferenceError`` from :mod:`evaluation.metrics.wer`.
    """

    return {
        NORMALIZED_SCORE: _score_with_policy(
            reference,
            hypothesis,
            CONSERVATIVE_ENGLISH_V1,
        ),
        RAW_SCORE: _score_with_policy(reference, hypothesis, RAW_ORTHOGRAPHIC_V1),
    }


def score_asr_wer_pairs(
    reference_transcripts: Sequence[str],
    generated_asr_transcripts: Sequence[str],
) -> dict[str, Any]:
    """Score pairwise ASR-WER in its required direction.

    Each sample is ``WER(reference_transcript, ASR(generated_speech))``. Inputs
    must have identical lengths; pairing by truncation would silently corrupt
    the evaluation.
    """

    if len(reference_transcripts) != len(generated_asr_transcripts):
        raise ValueError(
            "Pairwise ASR-WER requires equal reference and hypothesis counts: "
            f"got {len(reference_transcripts)} and {len(generated_asr_transcripts)}"
        )
    if not reference_transcripts:
        raise ValueError("Cannot score ASR-WER for an empty corpus")

    samples = [
        score_wer_pair(reference, hypothesis)
        for reference, hypothesis in zip(reference_transcripts, generated_asr_transcripts)
    ]
    return {
        "semantics": "WER(reference_transcript, ASR(generated_speech))",
        "samples": samples,
        "corpus": aggregate_wer_scores(samples),
    }


def aggregate_wer_scores(samples: Iterable[Mapping[str, Mapping[str, Any]]]) -> dict[str, Any]:
    """Aggregate sample score records using summed counts and macro WER."""

    items = list(samples)
    if not items:
        raise ValueError("Cannot aggregate WER scores from an empty corpus")
    return {
        NORMALIZED_SCORE: _aggregate_policy_scores(items, NORMALIZED_SCORE),
        RAW_SCORE: _aggregate_policy_scores(items, RAW_SCORE),
    }


def _score_with_policy(
    reference: str,
    hypothesis: str,
    policy: NormalizationPolicy,
) -> dict[str, Any]:
    counts = utterance_wer_counts(reference, hypothesis, policy)
    return {
        "normalization_id": policy.identifier,
        "reference": policy.normalize(reference),
        "hypothesis": policy.normalize(hypothesis),
        **_counts_record(counts),
    }


def _aggregate_policy_scores(
    samples: Sequence[Mapping[str, Mapping[str, Any]]],
    score_name: str,
) -> dict[str, Any]:
    records = [sample[score_name] for sample in samples]
    normalization_ids = {str(record["normalization_id"]) for record in records}
    if len(normalization_ids) != 1:
        raise ValueError(f"Cannot aggregate mixed {score_name} normalization policies")

    counts = [_counts_from_record(record) for record in records]
    total = sum_wer_counts(counts)
    return {
        "normalization_id": normalization_ids.pop(),
        "sample_count": len(records),
        **_counts_record(total),
        "macro_wer": macro_utterance_wer(counts),
    }


def _counts_record(counts: WerCounts) -> dict[str, int | float]:
    return {
        "substitutions": counts.substitutions,
        "deletions": counts.deletions,
        "insertions": counts.insertions,
        "errors": counts.errors,
        "reference_words": counts.reference_words,
        "wer": counts.wer,
    }


def _counts_from_record(record: Mapping[str, Any]) -> WerCounts:
    try:
        return WerCounts(
            substitutions=record["substitutions"],
            deletions=record["deletions"],
            insertions=record["insertions"],
            reference_words=record["reference_words"],
        )
    except KeyError as exc:
        raise ValueError(f"Incomplete WER score record: missing {exc.args[0]!r}") from exc
