from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

MetricLevel = Literal["sample", "corpus"]


@dataclass(frozen=True)
class MetricSpec:
    name: str
    level: MetricLevel
    required_source_fields: tuple[str, ...] = ()
    requires_output_file: bool = False
    requires_hypothesis_text: bool = False
    external: bool = False


_SPECS = {
    "wer": MetricSpec(
        name="wer",
        level="sample",
        required_source_fields=("speech_text",),
        requires_hypothesis_text=True,
    ),
    "semantic_equivalence": MetricSpec(
        name="semantic_equivalence",
        level="sample",
        required_source_fields=("speech_text",),
        requires_hypothesis_text=True,
        external=True,
    ),
    "nli_contradiction": MetricSpec(
        name="nli_contradiction",
        level="sample",
        requires_hypothesis_text=True,
        external=True,
    ),
    "g_eval_reference_free": MetricSpec(
        name="g_eval_reference_free",
        level="sample",
        requires_hypothesis_text=True,
        external=True,
    ),
    "g_eval_prometheus_criterion_v2_plausibility": MetricSpec(
        name="g_eval_prometheus_criterion_v2_plausibility",
        level="sample",
        requires_hypothesis_text=True,
        external=True,
    ),
    "g_eval_prometheus_criterion_v2_reference_grounded": MetricSpec(
        name="g_eval_prometheus_criterion_v2_reference_grounded",
        level="sample",
        requires_hypothesis_text=True,
        external=True,
    ),
    "asr_wer": MetricSpec(
        name="asr_wer",
        level="sample",
        required_source_fields=("speech_text",),
        requires_output_file=True,
        external=True,
    ),
    "dwer": MetricSpec(
        name="dwer",
        level="sample",
        required_source_fields=("speech_file",),
        requires_output_file=True,
        external=True,
    ),
    "mace": MetricSpec(
        name="mace",
        level="sample",
        required_source_fields=("audio_cap", "audio_file"),
        requires_hypothesis_text=True,
        external=True,
    ),
    "spider": MetricSpec(
        name="spider",
        level="sample",
        required_source_fields=("audio_cap",),
        requires_hypothesis_text=True,
        external=True,
    ),
    "aac_clapsim": MetricSpec(
        name="aac_clapsim",
        level="sample",
        required_source_fields=("audio_file",),
        requires_hypothesis_text=True,
        external=True,
    ),
    "target_caption_clap_score": MetricSpec(
        name="target_caption_clap_score",
        level="sample",
        required_source_fields=("audio_cap",),
        requires_output_file=True,
        external=True,
    ),
    "prompt_adherence_clap_score": MetricSpec(
        name="prompt_adherence_clap_score",
        level="sample",
        requires_hypothesis_text=True,
        requires_output_file=True,
        external=True,
    ),
    "fense": MetricSpec(
        name="fense",
        level="sample",
        required_source_fields=("audio_cap",),
        requires_hypothesis_text=True,
        external=True,
    ),
    "spider_max": MetricSpec(
        name="spider_max",
        level="sample",
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
    ),
    "kad": MetricSpec(
        name="kad",
        level="corpus",
        required_source_fields=("audio_file",),
        requires_output_file=True,
        external=True,
    ),
    "kl_passt": MetricSpec(
        name="kl_passt",
        level="corpus",
        required_source_fields=("audio_file",),
        requires_output_file=True,
        external=True,
    ),
    "inception_score": MetricSpec(
        name="inception_score",
        level="corpus",
        required_source_fields=("audio_file",),
        requires_output_file=True,
        external=True,
    ),
    "utmos": MetricSpec(
        name="utmos",
        level="sample",
        requires_output_file=True,
        external=True,
    ),
    "dnsmos": MetricSpec(
        name="dnsmos",
        level="sample",
        requires_output_file=True,
        external=True,
    ),
    "urgentmos": MetricSpec(
        name="urgentmos",
        level="sample",
        requires_output_file=True,
        external=True,
    ),
    "spk_sim": MetricSpec(
        name="spk_sim",
        level="sample",
        required_source_fields=("speech_file",),
        requires_output_file=True,
        external=True,
    ),
}


def get_metric_specs(names: list[str]) -> list[MetricSpec]:
    return [_SPECS[name] for name in names]


def known_metric_names() -> set[str]:
    return set(_SPECS)
