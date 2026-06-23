from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

MetricLevel = Literal["sample", "corpus"]


@dataclass(frozen=True)
class MetricSpec:
    name: str
    level: MetricLevel
    required_source_fields: tuple[str, ...] = ()
    required_prediction_fields: tuple[str, ...] = ()
    requires_output_file: bool = False
    requires_hypothesis_text: bool = False
    external: bool = False


_SPECS = {
    "wer": MetricSpec(
        name="wer",
        level="sample",
        required_source_fields=("speech_text",),
        required_prediction_fields=("generated_text",),
        requires_hypothesis_text=True,
    ),
    "semantic_equivalence": MetricSpec(
        name="semantic_equivalence",
        level="sample",
        required_source_fields=("speech_text",),
        required_prediction_fields=("generated_text",),
        requires_hypothesis_text=True,
        external=True,
    ),
    "nli_contradiction": MetricSpec(
        name="nli_contradiction",
        level="sample",
        required_prediction_fields=("generated_text",),
        requires_hypothesis_text=True,
        external=True,
    ),
    "geval_text_plausibility": MetricSpec(
        name="geval_text_plausibility",
        level="sample",
        required_prediction_fields=("generated_text",),
        requires_hypothesis_text=True,
        external=True,
    ),
    "geval_text_reference": MetricSpec(
        name="geval_text_reference",
        level="sample",
        required_prediction_fields=("generated_text",),
        requires_hypothesis_text=True,
        external=True,
    ),
    "geval_audio_plausibility": MetricSpec(
        name="geval_audio_plausibility",
        level="sample",
        required_prediction_fields=("output_file",),
        requires_output_file=True,
        external=True,
    ),
    "geval_audio_reference": MetricSpec(
        name="geval_audio_reference",
        level="sample",
        required_prediction_fields=("output_file",),
        requires_output_file=True,
        external=True,
    ),
    "asr_wer": MetricSpec(
        name="asr_wer",
        level="sample",
        required_source_fields=("speech_text",),
        required_prediction_fields=("output_file",),
        requires_output_file=True,
        external=True,
    ),
    "dwer": MetricSpec(
        name="dwer",
        level="sample",
        required_source_fields=("speech_file",),
        required_prediction_fields=("output_file",),
        requires_output_file=True,
        external=True,
    ),
    "mace": MetricSpec(
        name="mace",
        level="sample",
        required_source_fields=("audio_cap", "audio_file"),
        required_prediction_fields=("generated_text",),
        requires_hypothesis_text=True,
        external=True,
    ),
    "spider": MetricSpec(
        name="spider",
        level="sample",
        required_source_fields=("audio_cap",),
        required_prediction_fields=("generated_text",),
        requires_hypothesis_text=True,
        external=True,
    ),
    "aac_clapsim": MetricSpec(
        name="aac_clapsim",
        level="sample",
        required_source_fields=("audio_file",),
        required_prediction_fields=("generated_text",),
        requires_hypothesis_text=True,
        external=True,
    ),
    "target_caption_clap_score": MetricSpec(
        name="target_caption_clap_score",
        level="sample",
        required_source_fields=("audio_cap",),
        required_prediction_fields=("output_file",),
        requires_output_file=True,
        external=True,
    ),
    "prompt_adherence_clap_score": MetricSpec(
        name="prompt_adherence_clap_score",
        level="sample",
        required_prediction_fields=("output_file", "conditioning_prompt"),
        requires_hypothesis_text=True,
        requires_output_file=True,
        external=True,
    ),
    "fense": MetricSpec(
        name="fense",
        level="sample",
        required_source_fields=("audio_cap",),
        required_prediction_fields=("generated_text",),
        requires_hypothesis_text=True,
        external=True,
    ),
    "spider_max": MetricSpec(
        name="spider_max",
        level="sample",
        required_source_fields=("audio_cap",),
        required_prediction_fields=("generated_text",),
        requires_hypothesis_text=True,
        external=True,
    ),
    "fad": MetricSpec(
        name="fad",
        level="corpus",
        required_source_fields=("audio_file",),
        required_prediction_fields=("output_file",),
        requires_output_file=True,
        external=True,
    ),
    "kad": MetricSpec(
        name="kad",
        level="corpus",
        required_source_fields=("audio_file",),
        required_prediction_fields=("output_file",),
        requires_output_file=True,
        external=True,
    ),
    "kl_passt": MetricSpec(
        name="kl_passt",
        level="corpus",
        required_source_fields=("audio_file",),
        required_prediction_fields=("output_file",),
        requires_output_file=True,
        external=True,
    ),
    "inception_score": MetricSpec(
        name="inception_score",
        level="corpus",
        required_source_fields=("audio_file",),
        required_prediction_fields=("output_file",),
        requires_output_file=True,
        external=True,
    ),
    "utmos": MetricSpec(
        name="utmos",
        level="sample",
        required_prediction_fields=("output_file",),
        requires_output_file=True,
        external=True,
    ),
    "dnsmos": MetricSpec(
        name="dnsmos",
        level="sample",
        required_prediction_fields=("output_file",),
        requires_output_file=True,
        external=True,
    ),
    "urgentmos": MetricSpec(
        name="urgentmos",
        level="sample",
        required_prediction_fields=("output_file",),
        requires_output_file=True,
        external=True,
    ),
    "spk_sim": MetricSpec(
        name="spk_sim",
        level="sample",
        required_source_fields=("speech_file",),
        required_prediction_fields=("output_file",),
        requires_output_file=True,
        external=True,
    ),
}


def get_metric_specs(names: list[str]) -> list[MetricSpec]:
    return [_SPECS[name] for name in names]


def known_metric_names() -> set[str]:
    return set(_SPECS)
