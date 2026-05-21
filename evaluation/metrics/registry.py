from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

from evaluation.errors import EvaluationError

MetricLevel = Literal["per_sample", "corpus"]


@dataclass(frozen=True)
class MetricSpec:
    name: str
    level: MetricLevel
    required_source_fields: tuple[str, ...] = ()
    requires_output_file: bool = False
    requires_hypothesis_text: bool = False
    external: bool = False
    corpus_only: bool = False


_SPECS = {
    "wer": MetricSpec(
        name="wer",
        level="per_sample",
        required_source_fields=("speech_text",),
        requires_hypothesis_text=True,
    ),
    "semantic_equivalence": MetricSpec(
        name="semantic_equivalence",
        level="per_sample",
        required_source_fields=("speech_text",),
        requires_hypothesis_text=True,
        external=True,
    ),
    "asr_wer": MetricSpec(
        name="asr_wer",
        level="per_sample",
        required_source_fields=("speech_text",),
        requires_output_file=True,
        external=True,
    ),
    "dwer": MetricSpec(
        name="dwer",
        level="per_sample",
        required_source_fields=("speech_file",),
        requires_output_file=True,
        external=True,
    ),
    "mace": MetricSpec(
        name="mace",
        level="per_sample",
        required_source_fields=("audio_cap", "audio_file"),
        requires_hypothesis_text=True,
        external=True,
    ),
    "spider": MetricSpec(
        name="spider",
        level="per_sample",
        required_source_fields=("audio_cap",),
        requires_hypothesis_text=True,
        external=True,
    ),
    "clap": MetricSpec(
        name="clap",
        level="per_sample",
        required_source_fields=("audio_cap",),
        requires_output_file=True,
        external=True,
    ),
    "fad": MetricSpec(
        name="fad",
        level="corpus",
        required_source_fields=("audio_file",),
        requires_output_file=True,
        external=True,
        corpus_only=True,
    ),
    "kad": MetricSpec(
        name="kad",
        level="corpus",
        required_source_fields=("audio_file",),
        requires_output_file=True,
        external=True,
        corpus_only=True,
    ),
    "kld": MetricSpec(
        name="kld",
        level="corpus",
        required_source_fields=("audio_file",),
        requires_output_file=True,
        external=True,
        corpus_only=True,
    ),
    "inception_score": MetricSpec(
        name="inception_score",
        level="corpus",
        requires_output_file=True,
        external=True,
        corpus_only=True,
    ),
    "utmos": MetricSpec(
        name="utmos",
        level="per_sample",
        requires_output_file=True,
        external=True,
    ),
    "dnsmos": MetricSpec(
        name="dnsmos",
        level="per_sample",
        requires_output_file=True,
        external=True,
    ),
}


DEFAULT_TASK_METRICS = {
    "T2A": ("clap", "fad", "kad", "kld", "inception_score"),
    "T2S": ("asr_wer", "dwer", "utmos", "dnsmos"),
    "S2A": ("wer", "mace", "spider", "clap", "fad", "kad", "kld"),
    "A2S": ("mace", "spider", "asr_wer", "dwer", "utmos", "dnsmos"),
    "pseudo_t2a_direct": ("asr_wer", "utmos", "dnsmos"),
    "pseudo_t2a_phoneme": ("asr_wer", "utmos", "dnsmos"),
}


def parse_metric_names(value: str | None, task: str) -> list[str]:
    if value:
        names = [item.strip() for item in value.split(",") if item.strip()]
    else:
        names = list(DEFAULT_TASK_METRICS.get(task, ()))
    if not names:
        raise EvaluationError("No metrics selected. Pass --metrics metric1,metric2.")
    unknown = sorted(set(names) - set(_SPECS))
    if unknown:
        known = ", ".join(sorted(_SPECS))
        raise EvaluationError(f"Unknown metrics {unknown}. Known metrics: {known}")
    return names


def get_metric_specs(names: list[str]) -> list[MetricSpec]:
    return [_SPECS[name] for name in names]
