from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

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
    "clap_similarity": MetricSpec(
        name="clap_similarity",
        level="per_sample",
        required_source_fields=("audio_cap",),
        requires_output_file=True,
        external=True,
    ),
    "fense": MetricSpec(
        name="fense",
        level="per_sample",
        required_source_fields=("audio_cap",),
        requires_hypothesis_text=True,
        external=True,
    ),
    "spider_max": MetricSpec(
        name="spider_max",
        level="per_sample",
        required_source_fields=("audio_cap",),
        requires_hypothesis_text=True,
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
    "urgentmos": MetricSpec(
        name="urgentmos",
        level="per_sample",
        requires_output_file=True,
        external=True,
    ),
    "spk_sim": MetricSpec(
        name="spk_sim",
        level="per_sample",
        required_source_fields=("speech_file",),
        requires_output_file=True,
        external=True,
    ),
}


def get_metric_specs(names: list[str]) -> list[MetricSpec]:
    return [_SPECS[name] for name in names]


def known_metric_names() -> set[str]:
    return set(_SPECS)
