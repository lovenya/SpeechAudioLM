#!/usr/bin/env python
"""Run criterion-wise Prometheus G-Eval v2 for T2T SpeechAudio LM outputs."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import re
from dataclasses import dataclass
from pathlib import Path

from project_paths import model_path
from statistics import mean, median, multimode, pstdev
from typing import Any, Protocol


EVALUATOR_VERSION = "geval_prometheus_criterion_v3"
SCORING_POLICY_VERSION = "geval_weighted_mean_v1_no_caps"
DEFAULT_MODEL_PATH = str(model_path("prometheus_8x7b_v2"))
METRIC_IDS = {
    "plausibility": "g_eval_prometheus_plausibility",
    "reference_grounded": "g_eval_prometheus_reference",
}
PROMPT_VERSIONS = {
    "plausibility": "plausibility_v3",
    "reference_grounded": "reference_grounded_v3",
}
PROMPT_TEMPLATE_NAME = "prometheus_direct_assessment_v1"
PROMPT_TEMPLATE_TEXT = """###Task Description:
You are evaluating a creative cross-modal generation system.

{task_description}

Judge only the requested criterion using the score rubric.
Do not let qualities assigned to other criteria affect this score.

Criterion boundary:
{criterion_boundary}

Direction: {direction}

For audio_to_speech, the input is an environmental audio description and the response is a generated human utterance that could occur in, after, or because of the audio scene.

For speech_to_audio, the input is a speech utterance and the response is a generated environmental audio description that could contextualize, accompany, explain, or be caused by the speech.

Write concise feedback strictly based on the criterion and rubric. Then output the final score as:
Feedback: <feedback> [RESULT] <integer score from 1 to 5>

If the criterion genuinely cannot be assessed from the response, you may instead output [RESULT] N/A. Do not use
N/A merely because the response is poor, short, vague, or low quality; use scores 1 or 2 for assessable weak outputs.

###The instruction to evaluate:
Input text:
{input_text}

###Response to evaluate:
{generated_text}{reference_section}

###Score Rubric:
Criterion: {criterion}

{rubric}

###Feedback:
"""

CRITERION_WEIGHTS = {
    ("plausibility", "audio_to_speech"): {
        "non_contradiction": 0.20,
        "meaningful_scene_to_utterance_connection": 0.25,
        "human_reaction_plausibility": 0.20,
        "contextual_emotional_fit": 0.15,
        "specificity_without_over_invention": 0.10,
        "spoken_naturalness": 0.10,
    },
    ("plausibility", "speech_to_audio"): {
        "non_contradiction": 0.20,
        "meaningful_utterance_to_scene_connection": 0.25,
        "causal_contextual_plausibility": 0.20,
        "physical_acoustic_coherence": 0.15,
        "specificity_without_over_invention": 0.10,
        "audio_description_naturalness": 0.10,
    },
    ("reference_grounded", "audio_to_speech"): {
        "non_contradiction": 0.15,
        "meaningful_scene_to_utterance_connection": 0.20,
        "reference_anchor_grounding": 0.25,
        "human_reaction_plausibility": 0.15,
        "contextual_emotional_fit": 0.10,
        "specificity_without_over_invention": 0.075,
        "spoken_naturalness": 0.075,
    },
    ("reference_grounded", "speech_to_audio"): {
        "non_contradiction": 0.15,
        "meaningful_utterance_to_scene_connection": 0.20,
        "reference_anchor_grounding": 0.25,
        "causal_contextual_plausibility": 0.15,
        "physical_acoustic_coherence": 0.10,
        "specificity_without_over_invention": 0.075,
        "audio_description_naturalness": 0.075,
    },
}

RUBRICS = {
    "non_contradiction": """Evaluate the degree to which the response conflicts with the input text.

Score 5: No contradiction or meaningful tension; the response is fully compatible with the input.
Score 4: No likely conflict, with only minor ambiguity or mild tension.
Score 3: Mostly compatible, but includes detail that creates a genuine possible inconsistency with the input.
Score 2: Likely or significant conflict with the input, though not a direct explicit contradiction.
Score 1: Clear contradiction of the input.""",
    "meaningful_scene_to_utterance_connection": """Evaluate whether the generated utterance has a meaningful situational, causal, contextual, or reactive relationship to the input audio scene.

Score 5: Strong, direct, specific relationship; the utterance clearly makes sense in, after, or because of the audio scene.
Score 4: Clear plausible relationship with minor weakness or missing specificity.
Score 3: Identifiable but generic, weak, or incomplete relationship.
Score 2: Very tenuous relationship; the utterance is mostly independent of the audio scene.
Score 1: No meaningful relationship to the audio scene.""",
    "meaningful_utterance_to_scene_connection": """Evaluate whether the generated audio description has a meaningful situational, causal, contextual, or accompanying relationship to the input speech utterance.

Score 5: Strong, direct, specific relationship; the audio scene clearly contextualizes, accompanies, explains, or could be caused by the speech.
Score 4: Clear plausible relationship with minor weakness or missing specificity.
Score 3: Identifiable but generic, weak, or incomplete relationship.
Score 2: Very tenuous relationship; the audio scene is mostly independent of the speech.
Score 1: No meaningful relationship to the speech.""",
    "human_reaction_plausibility": """Evaluate whether a human would plausibly say the generated utterance in response to, during, or because of the described audio scene.

Score 5: Highly plausible human utterance for the situation implied by the audio scene.
Score 4: Plausible utterance with minor contextual or pragmatic weakness.
Score 3: Possible but weakly motivated, atypical, or pragmatically strained for the situation.
Score 2: Unlikely or forced human reaction to the audio scene.
Score 1: Implausible as something a human would say in relation to the audio scene.""",
    "causal_contextual_plausibility": """Evaluate whether the generated audio scene could plausibly contextualize, accompany, explain, or be caused by the input speech.

Score 5: Highly plausible causal/contextual relationship between the speech and the audio scene.
Score 4: Plausible relationship with minor weakness or missing detail.
Score 3: Possible but generic, weak, or only partially motivated relationship.
Score 2: Unlikely or forced relationship between the speech and the audio scene.
Score 1: Implausible relationship between the speech and the audio scene.""",
    "physical_acoustic_coherence": """Evaluate whether the generated audio scene is physically and acoustically coherent.

Score 5: Sound sources, events, and acoustic scene are realistic and internally coherent.
Score 4: Mostly coherent with only minor physical or acoustic weakness.
Score 3: Generally possible but includes mildly questionable interactions among sound sources, events, or acoustic conditions.
Score 2: Contains significant physical or acoustic implausibility.
Score 1: Physically impossible, acoustically incoherent, or internally contradictory.""",
    "contextual_emotional_fit": """Evaluate whether the response fits the implied situation, intent, mood, urgency, or emotional context of the input.

Score 5: Strong fit to the implied context, mood, intent, or human reaction.
Score 4: Good fit with minor weakness or missing nuance.
Score 3: Partially fitting but generic, emotionally flat, or only weakly aligned.
Score 2: Poor contextual or emotional fit.
Score 1: Completely mismatched context, mood, intent, or emotional reaction.""",
    "specificity_without_over_invention": """Evaluate whether the response is concrete and useful without inventing unsupported excessive detail.

Score 5: Concrete, informative, and appropriately specific without unsupported over-invention.
Score 4: Mostly useful and specific, with minor vagueness or minor extra detail.
Score 3: Understandable but generic, under-specified, or slightly over-invented.
Score 2: Too vague or substantially over-invented relative to the input.
Score 1: Empty, uninformative, or dominated by unsupported invented details.""",
    "spoken_naturalness": """Evaluate whether the generated speech text sounds like a natural human utterance.

Score 5: Natural, fluent, and plausible as spoken human language.
Score 4: Mostly natural with minor awkwardness.
Score 3: Understandable but somewhat stiff, repetitive, or template-like.
Score 2: Unnatural, awkward, or difficult to accept as human speech.
Score 1: Broken, incoherent, excessively repetitive, or not a plausible utterance.""",
    "audio_description_naturalness": """Evaluate whether the generated audio description is a coherent natural description of an audio scene.

Score 5: Natural, coherent, and well-formed audio scene description.
Score 4: Mostly natural with minor awkwardness or wording issue.
Score 3: Understandable but somewhat stiff, repetitive, or template-like.
Score 2: Unnatural, awkward, or poorly formed as an audio description.
Score 1: Broken, incoherent, excessively repetitive, or not a meaningful audio description.""",
}

REFERENCE_GROUNDING_RUBRICS = {
    "audio_to_speech": """Evaluate whether the generated utterance is grounded in the reference speech anchor(s) while allowing valid alternative wording and reasonable variation.

The reference anchors are not exact gold answers. A high score does not require copying the reference. It requires preserving the important intended speech-output space, such as core situation, intent, reaction, or communicative function.

Score 5: Strongly grounded in the reference anchor(s); preserves the important intended speech situation/intent/reaction while allowing natural variation.
Score 4: Mostly grounded in the reference anchor(s), with minor omission, shift, or extra detail.
Score 3: Partially grounded, but misses or changes an important aspect of the reference output space.
Score 2: Weakly grounded; mostly an alternative utterance not well supported by the reference anchor(s).
Score 1: Not grounded in, or clearly conflicts with, the reference anchor(s).""",
    "speech_to_audio": """Evaluate whether the generated audio description is grounded in the reference audio-caption anchor(s) while allowing valid alternative realizations.

The reference anchors are not exact gold answers. A high score does not require copying the reference. It requires preserving the important intended audio-output space, such as core sound sources, scene type, events, or acoustic context.

Score 5: Strongly grounded in the reference anchor(s); preserves the important intended audio scene/sound sources/events while allowing reasonable variation.
Score 4: Mostly grounded in the reference anchor(s), with minor omission, shift, or extra detail.
Score 3: Partially grounded, but misses or changes an important aspect of the reference output space.
Score 2: Weakly grounded; mostly an alternative audio scene not well supported by the reference anchor(s).
Score 1: Not grounded in, or clearly conflicts with, the reference anchor(s).""",
}

CRITERION_BOUNDARIES = {
    "non_contradiction": (
        "Judge only whether the response conflicts with the input. Do not lower the score merely because the "
        "response is weakly related, generic, missing details, or contains plausible unsupported details. Those "
        "properties are evaluated by other criteria. Unsupported detail matters here only when it creates a "
        "genuine possible, likely, or actual conflict with the input."
    ),
    "meaningful_scene_to_utterance_connection": (
        "Judge only the strength of the scene-to-utterance relationship. Do not score language fluency, emotional "
        "tone, or reference similarity here."
    ),
    "meaningful_utterance_to_scene_connection": (
        "Judge only the strength of the utterance-to-scene relationship. Do not score prose quality, internal "
        "acoustic realism, or reference similarity here."
    ),
    "human_reaction_plausibility": (
        "Judge whether the utterance is a plausible thing for a human to say in this situation. Do not require it "
        "to mention every input detail or match a reference utterance."
    ),
    "causal_contextual_plausibility": (
        "Judge only whether the described audio could plausibly contextualize, accompany, explain, or result from "
        "the speech. Do not score prose quality, acoustic realism, or reference similarity here."
    ),
    "physical_acoustic_coherence": (
        "Judge only whether the generated audio scene is internally physically and acoustically possible. Do not "
        "lower the score for weak relevance to the input or disagreement with a reference anchor."
    ),
    "contextual_emotional_fit": (
        "Judge only situational, pragmatic, emotional, and urgency fit. Do not score wording naturalness, exact "
        "reference similarity, or omission of acoustically irrelevant details here."
    ),
    "specificity_without_over_invention": (
        "Judge only whether the response has an appropriate level of supported detail. Do not treat every "
        "reasonable unstated detail as a contradiction, and do not score general cross-modal relevance here."
    ),
    "spoken_naturalness": (
        "Judge only whether the generated utterance itself sounds natural and fluent as spoken language. Ignore "
        "its relevance to the input, factual grounding, and similarity to any reference."
    ),
    "audio_description_naturalness": (
        "Judge only whether the generated text itself is a natural, coherent audio-scene description. Ignore its "
        "relevance to the input, factual grounding, and similarity to any reference."
    ),
    "reference_anchor_grounding": (
        "Judge grounding in the intended output space represented by the anchor, not exact matching. A different "
        "utterance or scene may receive a high score when it preserves the anchor's important situation, intent, "
        "sound source, event, or communicative function. Do not penalize wording or harmless realization changes."
    ),
}


class Judge(Protocol):
    name: str
    cache_identity: str

    def generate_with_traces(self, prompts: list[str]) -> list["JudgeGeneration"]:
        """Return one raw judge response and exact backend prompt per prompt."""


@dataclass(frozen=True)
class CriterionItem:
    row_position: int
    criterion: str
    prompt: str
    prompt_hash: str
    prompt_record: dict[str, Any]


@dataclass(frozen=True)
class JudgeGeneration:
    raw_response: str
    backend_formatted_prompt: str
    backend_format: str


class StubJudge:
    name = "stub_geval_prometheus_criterion_v2"
    cache_identity = "stub_geval_prometheus_criterion_v2_v1"

    def generate_with_traces(self, prompts: list[str]) -> list[JudgeGeneration]:
        return [
            JudgeGeneration(
                raw_response="Feedback: Stub deterministic criterion feedback. [RESULT] 4",
                backend_formatted_prompt=prompt,
                backend_format="identity",
            )
            for prompt in prompts
        ]


class VllmPrometheusJudge:
    def __init__(
        self,
        model_path: str,
        *,
        tensor_parallel_size: int,
        temperature: float,
        max_tokens: int,
        gpu_memory_utilization: float,
        max_model_len: int | None,
        enforce_eager: bool,
        trust_remote_code: bool,
    ) -> None:
        try:
            from vllm import LLM, SamplingParams
        except ImportError as exc:
            raise RuntimeError("Prometheus G-Eval requires vllm in the configured metric environment.") from exc

        self.name = model_path
        self._sampling_params = SamplingParams(temperature=temperature, max_tokens=max_tokens)
        options: dict[str, Any] = {
            "model": model_path,
            "tensor_parallel_size": tensor_parallel_size,
            "trust_remote_code": trust_remote_code,
            "gpu_memory_utilization": gpu_memory_utilization,
            "enforce_eager": enforce_eager,
        }
        if max_model_len is not None:
            options["max_model_len"] = max_model_len
        self._llm = LLM(**options)
        self._tokenizer = self._llm.get_tokenizer()
        self.cache_identity = json.dumps(
            {
                **options,
                "temperature": temperature,
                "max_tokens": max_tokens,
                "chat_template": getattr(self._tokenizer, "chat_template", None),
            },
            sort_keys=True,
        )

    def render_runtime_prompt(self, prompt: str) -> str:
        return self._tokenizer.apply_chat_template(
            [{"role": "user", "content": prompt}],
            tokenize=False,
            add_generation_prompt=True,
        )

    def generate_with_traces(self, prompts: list[str]) -> list[JudgeGeneration]:
        formatted = [self.render_runtime_prompt(prompt) for prompt in prompts]
        responses = self._llm.generate(formatted, self._sampling_params)
        return [
            JudgeGeneration(
                raw_response=response.outputs[0].text.strip(),
                backend_formatted_prompt=formatted_prompt,
                backend_format="chat_template" if formatted_prompt != prompt else "identity",
            )
            for prompt, formatted_prompt, response in zip(prompts, formatted, responses)
        ]


def parse_prometheus_result(raw: str) -> dict[str, Any]:
    matches = list(re.finditer(r"\[RESULT\]\s*([^\r\n]*)", raw, flags=re.IGNORECASE))
    if not matches:
        raise ValueError("Missing [RESULT] marker")
    result_text = matches[-1].group(1).strip()
    feedback = raw[: matches[-1].start()].strip()
    feedback = re.sub(r"^\s*Feedback:\s*", "", feedback, flags=re.IGNORECASE).strip()
    if not feedback:
        raise ValueError("Feedback must be non-empty")
    if re.fullmatch(r"N/?A|null", result_text, flags=re.IGNORECASE):
        return {
            "score": None,
            "feedback": feedback,
            "parse_status": "abstained",
            "parse_repair_attempted": False,
            "parse_repair_succeeded": False,
            "parse_repair_method": None,
            "abstention_reason": "judge_returned_no_score",
        }
    score_match = re.fullmatch(r"([1-5])", result_text)
    if not score_match:
        raise ValueError(f"Final [RESULT] must contain one integer from 1 to 5 or N/A: {result_text!r}")
    return {
        "score": int(score_match.group(1)),
        "feedback": feedback,
        "parse_status": "parsed",
        "parse_repair_attempted": False,
        "parse_repair_succeeded": False,
        "parse_repair_method": None,
    }


def weighted_final_score(criteria: dict[str, dict[str, Any]], weights: dict[str, float]) -> float | None:
    if any(not _is_score((criteria.get(name) or {}).get("score")) for name in weights):
        return None
    denominator = sum(weights.values())
    if denominator <= 0:
        raise ValueError("Criterion weights must have a positive sum")
    return sum(float(criteria[name]["score"]) * weight for name, weight in weights.items()) / denominator


def deterministic_diagnostics(criteria: dict[str, dict[str, Any]]) -> dict[str, Any]:
    scores = {name: value.get("score") for name, value in criteria.items() if isinstance(value, dict)}
    failure_modes: list[str] = []

    def low(name: str) -> bool:
        return _is_score(scores.get(name)) and float(scores[name]) <= 2

    mappings = [
        (low("non_contradiction"), "clear_contradiction_or_major_tension"),
        (
            low("meaningful_scene_to_utterance_connection") or low("meaningful_utterance_to_scene_connection"),
            "weak_or_no_meaningful_connection",
        ),
        (low("human_reaction_plausibility"), "implausible_human_reaction"),
        (low("causal_contextual_plausibility"), "weak_causal_contextual_plausibility"),
        (low("physical_acoustic_coherence"), "physically_or_acoustically_incoherent"),
        (low("reference_anchor_grounding"), "weak_reference_anchor_grounding"),
        (low("specificity_without_over_invention"), "too_vague_or_overinvented"),
        (low("spoken_naturalness") or low("audio_description_naturalness"), "unnatural_output_text"),
    ]
    for applies, label in mappings:
        if applies:
            failure_modes.append(label)
    return {"failure_modes": failure_modes, "failure_mode_source": "deterministic"}


def build_prompt_record(
    *,
    variant: str,
    direction: str,
    criterion: str,
    input_text: str,
    generated_text: str,
    reference_anchors: list[dict[str, str]] | None = None,
) -> dict[str, Any]:
    if (variant, direction) not in CRITERION_WEIGHTS:
        raise ValueError(f"Unsupported variant/direction: {variant}/{direction}")
    rubric = REFERENCE_GROUNDING_RUBRICS[direction] if criterion == "reference_anchor_grounding" else RUBRICS[criterion]
    if variant == "reference_grounded" and criterion == "reference_anchor_grounding":
        anchors = format_reference_anchors(reference_anchors or [])
        task_description = """The task is not exact reference matching. Multiple outputs can be valid. The reference anchors are examples of the intended target-output space, not the only correct answers. Do not require exact wording or exact scene matching. Use them only to judge whether the response is grounded in the dataset's intended paired output space."""
        reference_section = f"\n\n###Reference Anchors:\n{anchors}"
    else:
        task_description = "Multiple outputs can be valid. Judge only the requested criterion using the score rubric. No reference comparison is part of this criterion."
        reference_section = ""
    criterion_boundary = CRITERION_BOUNDARIES[criterion]
    rendered_user_message = PROMPT_TEMPLATE_TEXT.format(
        task_description=task_description,
        criterion_boundary=criterion_boundary,
        direction=direction,
        input_text=input_text,
        generated_text=generated_text,
        reference_section=reference_section,
        criterion=criterion,
        rubric=rubric,
    )
    return {
        "prompt_version": PROMPT_VERSIONS[variant],
        "template_name": PROMPT_TEMPLATE_NAME,
        "template": PROMPT_TEMPLATE_TEXT,
        "messages": [{"role": "user", "content": rendered_user_message}],
        "rendered_user_message": rendered_user_message,
        "task_description": task_description,
        "criterion_boundary": criterion_boundary,
        "rubric": rubric,
        "reference_section": reference_section,
    }


def build_prompt(
    *,
    variant: str,
    direction: str,
    criterion: str,
    input_text: str,
    generated_text: str,
    reference_anchors: list[dict[str, str]] | None = None,
) -> str:
    return build_prompt_record(
        variant=variant,
        direction=direction,
        criterion=criterion,
        input_text=input_text,
        generated_text=generated_text,
        reference_anchors=reference_anchors,
    )["rendered_user_message"]

def format_reference_anchors(anchors: list[dict[str, str]]) -> str:
    return "\n\n".join(
        f"[Anchor {index} | source={anchor.get('source', 'unknown')} | id={anchor.get('anchor_id', 'unknown')}]\n{anchor.get('text', '')}"
        for index, anchor in enumerate(anchors, start=1)
    )


def build_repair_prompt(raw_response: str) -> str:
    return f"""Repair the evaluator response into the native Prometheus direct-assessment format.

Return exactly one of:
Feedback: <non-empty concise feedback> [RESULT] <one integer from 1 to 5>
Feedback: <non-empty concise feedback> [RESULT] N/A

Do not return JSON. Preserve an explicit N/A rather than forcing a score.

Response to repair:
{raw_response}"""


def run_geval(
    *,
    input_path: Path,
    output_path: Path,
    summary_output: Path,
    summary_csv_output: Path | None,
    judge: Judge,
    variant: str,
    cache_dir: Path | None,
    batch_size: int,
    max_samples: int | None = None,
    criterion_judgments_output: Path | None = None,
) -> dict[str, Any]:
    rows = _read_input_rows(input_path, max_samples=max_samples)
    cache_identity = getattr(judge, "cache_identity", judge.name)
    states = [_initial_row_state(row, variant=variant, judge_name=judge.name) for row in rows]
    items: list[CriterionItem] = []
    if cache_dir:
        cache_dir.mkdir(parents=True, exist_ok=True)

    for position, state in enumerate(states):
        if state["status"] == "incomplete":
            continue
        for criterion in state["criterion_weights"]:
            item = _make_item(rows[position], position, criterion, variant=variant, cache_identity=cache_identity)
            cached = _read_cache(cache_dir, item.prompt_hash)
            if cached is not None:
                state["criteria"][criterion] = cached
            else:
                items.append(item)

    for start in range(0, len(items), batch_size):
        chunk = items[start : start + batch_size]
        generated = _generate_with_fallback(judge, chunk)
        for item, generation, generation_error in generated:
            if generation_error:
                result = {
                    "score": None,
                    "feedback": "",
                    "parse_status": "judge_error",
                    "parse_repair_attempted": False,
                    "parse_repair_succeeded": False,
                    "parse_repair_method": None,
                    "judge_error": generation_error,
                    "raw_response": "",
                }
            else:
                assert generation is not None
                result = _parse_with_repair(judge, generation.raw_response)
                result["runtime_prompt"] = _runtime_prompt_trace(generation)
            states[item.row_position]["criteria"][item.criterion] = result
            if result.get("parse_status") in {"parsed", "abstained"}:
                _write_cache(cache_dir, item.prompt_hash, result)
        _write_progress(
            states,
            requested_rows=len(rows),
            input_path=input_path,
            output_path=output_path,
            summary_output=summary_output,
            summary_csv_output=summary_csv_output,
            judge_name=judge.name,
            variant=variant,
            input_rows=rows,
            judge=judge,
            cache_identity=cache_identity,
            criterion_judgments_output=criterion_judgments_output,
        )

    summary = _write_progress(
        states,
        requested_rows=len(rows),
        input_path=input_path,
        output_path=output_path,
        summary_output=summary_output,
        summary_csv_output=summary_csv_output,
        judge_name=judge.name,
        variant=variant,
        input_rows=rows,
        judge=judge,
        cache_identity=cache_identity,
        criterion_judgments_output=criterion_judgments_output,
    )
    print(f"wrote {output_path} rows={summary['total_pairs_processed']}")
    print(f"wrote {summary_output}")
    return summary


def _initial_row_state(row: dict[str, Any], *, variant: str, judge_name: str) -> dict[str, Any]:
    direction = _required_str(row, "direction")
    weights = CRITERION_WEIGHTS.get((variant, direction))
    if weights is None:
        raise ValueError(f"Unsupported variant/direction: {variant}/{direction}")
    anchors = row.get("reference_anchors") if isinstance(row.get("reference_anchors"), list) else []
    missing_reference = variant == "reference_grounded" and not _valid_reference_anchors(anchors)
    state = {
        "schema_version": 2,
        "metric": METRIC_IDS[variant],
        "evaluator_version": EVALUATOR_VERSION,
        "evaluation_variant": variant,
        "row_index": row.get("row_index"),
        "pair_id": row.get("pair_id") or row.get("sample_id"),
        "sample_id": row.get("sample_id"),
        "split": row.get("split"),
        "direction": direction,
        "primary_audio_class": row.get("primary_audio_class"),
        "input_text": _required_str(row, "input_text"),
        "generated_text": _required_str(row, "generated_text"),
        "criteria": {},
        "criterion_weights": weights,
        "final_score": None,
        "scoring_policy_version": SCORING_POLICY_VERSION,
        "prompt_version": PROMPT_VERSIONS[variant],
        "judge_model": judge_name,
        "status": "incomplete" if missing_reference else "pending",
        "diagnostics": {"failure_modes": [], "failure_mode_source": "deterministic"},
    }
    if variant == "reference_grounded":
        state["reference_anchors"] = anchors
    if missing_reference:
        state["incomplete_reason"] = "missing_required_reference_anchor"
    return state


def _valid_reference_anchors(anchors: list[Any]) -> bool:
    return bool(anchors) and all(isinstance(anchor, dict) and isinstance(anchor.get("text"), str) and anchor["text"].strip() for anchor in anchors)


def _make_item(row: dict[str, Any], row_position: int, criterion: str, *, variant: str, cache_identity: str) -> CriterionItem:
    direction = _required_str(row, "direction")
    anchors = row.get("reference_anchors") if isinstance(row.get("reference_anchors"), list) else []
    prompt = build_prompt(
        variant=variant,
        direction=direction,
        criterion=criterion,
        input_text=_required_str(row, "input_text"),
        generated_text=_required_str(row, "generated_text"),
        reference_anchors=anchors,
    )
    cache_key = json.dumps(
        {
            "metric": METRIC_IDS[variant],
            "evaluator_version": EVALUATOR_VERSION,
            "evaluation_variant": variant,
            "direction": direction,
            "criterion": criterion,
            "input_text": row["input_text"],
            "generated_text": row["generated_text"],
            "reference_anchors": anchors if criterion == "reference_anchor_grounding" else [],
            "prompt_version": PROMPT_VERSIONS[variant],
            "scoring_policy_version": SCORING_POLICY_VERSION,
            "judge": cache_identity,
        },
        ensure_ascii=False,
        sort_keys=True,
    )
    prompt_record = build_prompt_record(
        variant=variant,
        direction=direction,
        criterion=criterion,
        input_text=_required_str(row, "input_text"),
        generated_text=_required_str(row, "generated_text"),
        reference_anchors=anchors,
    )
    return CriterionItem(
        row_position,
        criterion,
        prompt,
        hashlib.sha256(cache_key.encode("utf-8")).hexdigest(),
        prompt_record,
    )


def _generate_with_fallback(
    judge: Judge,
    items: list[CriterionItem],
) -> list[tuple[CriterionItem, JudgeGeneration | None, str | None]]:
    try:
        generations = judge.generate_with_traces([item.prompt for item in items])
        if len(generations) != len(items):
            raise RuntimeError(f"Judge returned {len(generations)} responses for {len(items)} prompts")
        return [(item, generation, None) for item, generation in zip(items, generations)]
    except Exception:
        generated = []
        for item in items:
            try:
                generations = judge.generate_with_traces([item.prompt])
                if len(generations) != 1:
                    raise RuntimeError(f"Judge returned {len(generations)} responses for one prompt")
                generated.append((item, generations[0], None))
            except Exception as exc:
                generated.append((item, None, f"{type(exc).__name__}: {exc}"))
        return generated


def _runtime_prompt_trace(generation: JudgeGeneration) -> dict[str, Any]:
    return {
        "backend_formatted_prompt": generation.backend_formatted_prompt,
        "backend_format": generation.backend_format,
        "trace_source": "generation_call",
    }


def _repair_prompt_trace(prompt: str, generation: JudgeGeneration | None) -> dict[str, Any]:
    trace: dict[str, Any] = {
        "messages": [{"role": "user", "content": prompt}],
        "rendered_user_message": prompt,
    }
    if generation is None:
        trace.update({"backend_formatted_prompt": None, "backend_format": "generation_failed"})
    else:
        trace.update(_runtime_prompt_trace(generation))
    return trace


def _parse_with_repair(judge: Judge, raw: str) -> dict[str, Any]:
    try:
        parsed = parse_prometheus_result(raw)
        parsed["raw_response"] = raw
        return parsed
    except Exception as exc:
        initial_error = f"{type(exc).__name__}: {exc}"
        repair_prompt = build_repair_prompt(raw)
        repair_generation: JudgeGeneration | None = None
        try:
            repair_generations = judge.generate_with_traces([repair_prompt])
            if len(repair_generations) != 1:
                raise RuntimeError(f"Judge returned {len(repair_generations)} responses for one repair prompt")
            repair_generation = repair_generations[0]
            repaired_raw = repair_generation.raw_response
            parsed = parse_prometheus_result(repaired_raw)
            parsed.update(
                {
                    "parse_repair_attempted": True,
                    "parse_repair_succeeded": True,
                    "parse_repair_method": "judge_reformat",
                    "parse_repair": {
                        "attempted": True,
                        "succeeded": True,
                        "method": "judge_reformat",
                        "prompt": _repair_prompt_trace(repair_prompt, repair_generation),
                        "raw_response": repaired_raw,
                    },
                    "initial_parse_error": initial_error,
                    "initial_raw_response": raw,
                    "raw_response": repaired_raw,
                }
            )
            return parsed
        except Exception as repair_exc:
            repaired_raw = repair_generation.raw_response if repair_generation is not None else ""
            return {
                "score": None,
                "feedback": "",
                "parse_status": "parse_failed",
                "parse_repair_attempted": True,
                "parse_repair_succeeded": False,
                "parse_repair_method": "judge_reformat",
                "parse_repair": {
                    "attempted": True,
                    "succeeded": False,
                    "method": "judge_reformat",
                    "prompt": _repair_prompt_trace(repair_prompt, repair_generation),
                    "raw_response": repaired_raw,
                },
                "parse_error": f"initial={initial_error}; repair={type(repair_exc).__name__}: {repair_exc}",
                "initial_parse_error": initial_error,
                "initial_raw_response": raw,
                "raw_response": repaired_raw,
            }


def _finalize_state(state: dict[str, Any]) -> None:
    if state.get("incomplete_reason") == "missing_required_reference_anchor":
        state["status"] = "incomplete"
        return
    expected = set(state["criterion_weights"])
    if not expected.issubset(state["criteria"]):
        state["status"] = "pending"
        state["final_score"] = None
        state["incomplete_reason"] = "criteria_pending"
        return
    statuses = {result.get("parse_status") for result in state["criteria"].values()}
    state["final_score"] = weighted_final_score(state["criteria"], state["criterion_weights"])
    state["diagnostics"] = deterministic_diagnostics(state["criteria"])
    if statuses.issubset({"parsed", "abstained"}) and "abstained" in statuses:
        state["status"] = "complete_with_abstention"
        state["incomplete_reason"] = "one_or_more_criteria_abstained"
    elif state["final_score"] is not None:
        state["status"] = "complete"
        state.pop("incomplete_reason", None)
    else:
        state["status"] = "incomplete"
        state["incomplete_reason"] = "one_or_more_required_criteria_failed"


def _write_progress(
    states: list[dict[str, Any]],
    *,
    requested_rows: int,
    input_path: Path,
    output_path: Path,
    summary_output: Path,
    summary_csv_output: Path | None,
    judge_name: str,
    variant: str,
    input_rows: list[dict[str, Any]] | None = None,
    judge: Judge | None = None,
    cache_identity: str | None = None,
    criterion_judgments_output: Path | None = None,
) -> dict[str, Any]:
    for state in states:
        _finalize_state(state)
    states.sort(key=lambda row: int(row.get("row_index", 0)))
    summary = _summary(states, input_path=input_path, output_path=output_path, judge_name=judge_name, variant=variant)
    summary["requested_pairs"] = requested_rows
    summary["remaining_pairs"] = sum(state.get("status") == "pending" for state in states)
    summary["failed_pairs"] = sum(state.get("status") == "incomplete" for state in states)
    summary["complete"] = summary["remaining_pairs"] == 0 and summary["failed_pairs"] == 0
    _write_jsonl(output_path, states)
    if criterion_judgments_output is not None and input_rows is not None and judge is not None:
        _write_jsonl(
            criterion_judgments_output,
            _criterion_judgment_rows(
                input_rows,
                states,
                variant=variant,
                judge=judge,
                cache_identity=cache_identity or judge_name,
            ),
        )
    _write_json(summary_output, summary)
    if summary_csv_output:
        _write_summary_csv(summary_csv_output, summary)
    return summary


def _criterion_judgment_rows(
    input_rows: list[dict[str, Any]],
    states: list[dict[str, Any]],
    *,
    variant: str,
    judge: Judge,
    cache_identity: str,
) -> list[dict[str, Any]]:
    input_by_row_index = {str(row.get("row_index", index)): row for index, row in enumerate(input_rows)}
    rows: list[dict[str, Any]] = []
    for state in states:
        input_row = input_by_row_index.get(str(state.get("row_index")), {})
        for criterion, weight in state.get("criterion_weights", {}).items():
            item = _make_item(input_row or state, int(state.get("row_index") or 0), criterion, variant=variant, cache_identity=cache_identity)
            result = state.get("criteria", {}).get(criterion)
            rows.append(_criterion_judgment_row(state, item, result, weight, judge))
    rows.sort(key=lambda row: (int(row.get("row_index") or 0), str(row.get("criterion_id") or "")))
    return rows


def _criterion_judgment_row(
    state: dict[str, Any],
    item: CriterionItem,
    result: dict[str, Any] | None,
    weight: float,
    judge: Judge,
) -> dict[str, Any]:
    prompt_record = dict(item.prompt_record)
    runtime_prompt = result.get("runtime_prompt") if isinstance(result, dict) else None
    if isinstance(runtime_prompt, dict):
        prompt_record.update(runtime_prompt)
    else:
        prompt_record["backend_formatted_prompt"] = None
        prompt_record["backend_format"] = "not_generated"
        prompt_record["trace_source"] = "no_generation_result"
    status = _criterion_judgment_status(state, result)
    row = {
        "schema_version": 1,
        "row_index": state.get("row_index"),
        "pair_id": state.get("pair_id"),
        "sample_id": state.get("sample_id"),
        "split": state.get("split"),
        "metric_id": state.get("metric"),
        "evaluator_version": state.get("evaluator_version"),
        "evaluation_variant": state.get("evaluation_variant"),
        "criterion_id": item.criterion,
        "criterion_weight": weight,
        "status": status,
        "input": {
            "direction": state.get("direction"),
            "stage_input_text": state.get("input_text"),
            "generated_text": state.get("generated_text"),
            "reference_anchors": state.get("reference_anchors", []),
        },
        "prompt": prompt_record,
        "judge": {"model": state.get("judge_model")},
        "output": _criterion_judgment_output(result),
        "error": _criterion_judgment_error(state, result),
    }
    return row


def _criterion_judgment_status(state: dict[str, Any], result: dict[str, Any] | None) -> str:
    if state.get("incomplete_reason") == "missing_required_reference_anchor" and not result:
        return "skipped"
    if not isinstance(result, dict):
        return "pending" if state.get("status") == "pending" else "error"
    outcome = _criterion_outcome(result)
    if outcome == "parsed":
        return "success"
    if outcome == "abstained":
        return "abstained"
    return "error"


def _criterion_judgment_output(result: dict[str, Any] | None) -> dict[str, Any]:
    if not isinstance(result, dict):
        return {
            "score": None,
            "range": {"min": 1, "max": 5},
            "higher_is_better": True,
            "feedback": None,
            "raw_response": None,
            "parse_status": "missing",
        }
    output = {
        "score": result.get("score"),
        "range": {"min": 1, "max": 5},
        "higher_is_better": True,
        "feedback": result.get("feedback"),
        "raw_response": result.get("raw_response"),
        "parse_status": result.get("parse_status"),
        "parse_repair_attempted": result.get("parse_repair_attempted"),
        "parse_repair_succeeded": result.get("parse_repair_succeeded"),
        "parse_repair_method": result.get("parse_repair_method"),
    }
    for key in ("parse_repair", "initial_parse_error", "initial_raw_response", "abstention_reason", "judge_error", "parse_error"):
        if key in result:
            output[key] = result[key]
    return output


def _criterion_judgment_error(state: dict[str, Any], result: dict[str, Any] | None) -> dict[str, Any] | None:
    if state.get("incomplete_reason") == "missing_required_reference_anchor" and not result:
        return {"type": "MissingReferenceAnchor", "message": "missing_required_reference_anchor"}
    if not isinstance(result, dict):
        return {"type": "CriterionPending", "message": "criterion has no result"}
    outcome = _criterion_outcome(result)
    if outcome in {"parsed", "abstained"}:
        return None
    message = result.get("judge_error") or result.get("parse_error") or outcome
    return {"type": str(result.get("parse_status") or "CriterionError"), "message": str(message)}


def _summary(rows: list[dict[str, Any]], *, input_path: Path, output_path: Path, judge_name: str, variant: str) -> dict[str, Any]:
    final_scores = [float(row["final_score"]) for row in rows if _is_score(row.get("final_score"))]
    criterion_scores: dict[str, list[float]] = {}
    criterion_metadata: dict[str, dict[str, int]] = {}
    criterion_parse_status_counts: dict[str, int] = {}
    status_counts: dict[str, int] = {}

    for row in rows:
        status = _effective_row_status(row)
        status_counts[status] = status_counts.get(status, 0) + 1
        for criterion in row.get("criterion_weights", {}):
            metadata = criterion_metadata.setdefault(
                criterion,
                {"eligible_count": 0, "abstention_count": 0, "failure_count": 0, "missing_count": 0},
            )
            metadata["eligible_count"] += 1
            result = row.get("criteria", {}).get(criterion)
            if not isinstance(result, dict):
                metadata["missing_count"] += 1
                continue
            outcome = _criterion_outcome(result)
            criterion_parse_status_counts[outcome] = criterion_parse_status_counts.get(outcome, 0) + 1
            if _is_score(result.get("score")):
                criterion_scores.setdefault(criterion, []).append(float(result["score"]))
            elif outcome == "abstained":
                metadata["abstention_count"] += 1
            else:
                metadata["failure_count"] += 1

    criterion_statistics = {}
    for name, metadata in sorted(criterion_metadata.items()):
        criterion_statistics[name] = {
            **_score_statistics(criterion_scores.get(name, []), total_count=metadata["eligible_count"], distribution="exact_1_to_5"),
            **metadata,
        }

    prompt_versions = sorted({str(row.get("prompt_version")) for row in rows if row.get("prompt_version")})
    prompt_version = prompt_versions[0] if len(prompt_versions) == 1 else prompt_versions
    final_statistics = _score_statistics(final_scores, total_count=len(rows), distribution="fixed_half_point_bins")
    abstained_judgments = status_counts.get("complete_with_abstention", 0)
    failed_judgments = status_counts.get("incomplete", 0)
    existing_rows_reinterpreted_as_abstentions = sum(
        _effective_row_status(row) == "complete_with_abstention"
        and row.get("status") != "complete_with_abstention"
        for row in rows
    )
    return {
        "schema_version": 3,
        "metric": METRIC_IDS[variant],
        "evaluator_version": EVALUATOR_VERSION,
        "evaluation_variant": variant,
        "prompt_version": prompt_version or PROMPT_VERSIONS[variant],
        "scoring_policy_version": SCORING_POLICY_VERSION,
        "aggregation_policy_version": "geval_descriptive_statistics_v1_null_excluded",
        "input_path": str(input_path),
        "output_path": str(output_path),
        "judge_model": judge_name,
        "total_pairs_processed": len(rows),
        "scored_judgments": len(final_scores),
        "abstained_judgments": abstained_judgments,
        "failed_judgments": failed_judgments,
        "existing_rows_reinterpreted_as_abstentions": existing_rows_reinterpreted_as_abstentions,
        "status_counts": dict(sorted(status_counts.items())),
        "criterion_parse_status_counts": dict(sorted(criterion_parse_status_counts.items())),
        "final_score_statistics": final_statistics,
        "criterion_score_statistics": criterion_statistics,
    }


def _effective_row_status(row: dict[str, Any]) -> str:
    if row.get("incomplete_reason") == "missing_required_reference_anchor":
        return "incomplete"
    expected = set(row.get("criterion_weights", {}))
    criteria = row.get("criteria", {})
    if not expected.issubset(criteria):
        return "pending" if row.get("status") == "pending" else "incomplete"
    outcomes = {_criterion_outcome(criteria[name]) for name in expected}
    if outcomes.issubset({"parsed", "abstained"}):
        return "complete_with_abstention" if "abstained" in outcomes else "complete"
    return "incomplete"


def _criterion_outcome(result: dict[str, Any]) -> str:
    if result.get("parse_status") == "abstained" or _is_existing_na_abstention(result):
        return "abstained"
    return str(result.get("parse_status") or "missing")


def _is_existing_na_abstention(result: dict[str, Any]) -> bool:
    if result.get("score") is not None or result.get("parse_status") != "parse_failed":
        return False
    old_parser_message = "Final [RESULT] must contain exactly one integer from 1 to 5"
    if old_parser_message not in str(result.get("parse_error") or ""):
        return False
    responses = (
        str(result.get("initial_raw_response") or ""),
        str(result.get("raw_response") or ""),
    )
    for response in responses:
        matches = list(re.finditer(r"\[RESULT\]\s*([^\r\n]*)", response, flags=re.IGNORECASE))
        if not matches or not re.fullmatch(r"N/?A|null", matches[-1].group(1).strip(), flags=re.IGNORECASE):
            return False
    return True


def _score_statistics(
    values: list[float],
    *,
    total_count: int,
    distribution: str,
) -> dict[str, Any]:
    sorted_values = sorted(values)
    modes = sorted(multimode(sorted_values)) if sorted_values else []
    mode_count = max((sorted_values.count(value) for value in modes), default=0)
    if mode_count <= 1:
        modes = []
    result = {
        "count": len(sorted_values),
        "null_count": total_count - len(sorted_values),
        "mean": mean(sorted_values) if sorted_values else None,
        "std": pstdev(sorted_values) if sorted_values else None,
        "min": min(sorted_values) if sorted_values else None,
        "max": max(sorted_values) if sorted_values else None,
        "median": median(sorted_values) if sorted_values else None,
        "mode": modes[0] if len(modes) == 1 else None,
        "modes": modes,
        "mode_count": mode_count if modes else 0,
        "distribution_type": distribution,
    }
    if distribution == "exact_1_to_5":
        counts = {str(score): 0 for score in range(1, 6)}
        for value in sorted_values:
            if value.is_integer() and 1 <= value <= 5:
                counts[str(int(value))] += 1
    elif distribution == "fixed_half_point_bins":
        labels = [
            "[1.0,1.5)", "[1.5,2.0)", "[2.0,2.5)", "[2.5,3.0)",
            "[3.0,3.5)", "[3.5,4.0)", "[4.0,4.5)", "[4.5,5.0]",
        ]
        counts = {label: 0 for label in labels}
        for value in sorted_values:
            index = min(max(int((value - 1.0) / 0.5), 0), len(labels) - 1)
            counts[labels[index]] += 1
    else:
        raise ValueError(f"Unsupported distribution: {distribution}")
    result["score_distribution"] = counts
    return result


def _is_score(value: Any) -> bool:
    return not isinstance(value, bool) and isinstance(value, (int, float))


def _required_str(row: dict[str, Any], field: str) -> str:
    value = row.get(field)
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"Row has no usable {field!r}: {row}")
    return value.strip()


def _read_input_rows(path: Path, *, max_samples: int | None) -> list[dict[str, Any]]:
    rows = []
    with path.open("r", encoding="utf-8") as src:
        for line in src:
            if line.strip():
                rows.append(json.loads(line))
                if max_samples is not None and len(rows) >= max_samples:
                    break
    return rows


def _read_cache(cache_dir: Path | None, prompt_hash: str) -> dict[str, Any] | None:
    path = cache_dir / f"{prompt_hash}.json" if cache_dir else None
    if not path or not path.is_file():
        return None
    try:
        cached = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return None
    if not isinstance(cached, dict):
        return None
    score = cached.get("score")
    status = cached.get("parse_status")
    valid_score = status == "parsed" and not isinstance(score, bool) and isinstance(score, int) and 1 <= score <= 5
    valid_abstention = status == "abstained" and score is None
    if not (valid_score or valid_abstention):
        return None
    if not isinstance(cached.get("feedback"), str) or not isinstance(cached.get("raw_response"), str):
        return None
    return cached


def _write_cache(cache_dir: Path | None, prompt_hash: str, data: dict[str, Any]) -> None:
    if cache_dir:
        (cache_dir / f"{prompt_hash}.json").write_text(json.dumps(data, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")


def _write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as dst:
        for row in rows:
            dst.write(json.dumps(row, ensure_ascii=False) + "\n")


def _write_json(path: Path, data: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")


def _write_summary_csv(path: Path, summary: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as dst:
        writer = csv.writer(dst)
        fields = [
            "count", "null_count", "mean", "std", "min", "max", "median", "mode", "modes",
            "mode_count", "distribution_type", "score_distribution", "eligible_count",
            "abstention_count", "failure_count", "missing_count",
        ]
        writer.writerow(["scope", "criterion", *fields])
        statistics = {"final_score": summary.get("final_score_statistics", {})}
        statistics.update(summary.get("criterion_score_statistics", {}))
        for criterion, values in statistics.items():
            writer.writerow(
                ["all", criterion]
                + [
                    json.dumps(values.get(field), sort_keys=True)
                    if field in {"modes", "score_distribution"}
                    else values.get(field)
                    for field in fields
                ]
            )


def reaggregate_existing_output(
    *,
    input_path: Path,
    output_path: Path,
    summary_output: Path,
    summary_csv_output: Path | None,
    variant: str,
) -> dict[str, Any]:
    rows = _read_input_rows(output_path, max_samples=None)
    judge_names = sorted({str(row.get("judge_model")) for row in rows if row.get("judge_model")})
    judge_name = judge_names[0] if len(judge_names) == 1 else json.dumps(judge_names)
    summary = _summary(rows, input_path=input_path, output_path=output_path, judge_name=judge_name, variant=variant)
    summary["requested_pairs"] = len(rows)
    summary["remaining_pairs"] = 0
    summary["failed_pairs"] = summary["failed_judgments"]
    summary["complete"] = summary["failed_pairs"] == 0
    _write_json(summary_output, summary)
    if summary_csv_output:
        _write_summary_csv(summary_csv_output, summary)
    return summary


def _build_judge(args: argparse.Namespace) -> Judge:
    if args.backend == "stub":
        return StubJudge()
    return VllmPrometheusJudge(
        args.model,
        tensor_parallel_size=args.tensor_parallel_size,
        temperature=args.temperature,
        max_tokens=args.max_tokens,
        gpu_memory_utilization=args.gpu_memory_utilization,
        max_model_len=args.max_model_len,
        enforce_eager=args.enforce_eager,
        trust_remote_code=args.trust_remote_code,
    )


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--summary-output", type=Path, required=True)
    parser.add_argument("--summary-csv-output", type=Path, default=None)
    parser.add_argument("--criterion-judgments-output", type=Path, default=None)
    parser.add_argument("--variant", choices=sorted(METRIC_IDS), required=True)
    parser.add_argument("--cache-dir", type=Path, default=None)
    parser.add_argument("--backend", choices=["vllm", "stub"], default="vllm")
    parser.add_argument("--model", default=DEFAULT_MODEL_PATH)
    parser.add_argument("--tensor-parallel-size", type=int, default=4)
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--max-tokens", type=int, default=512)
    parser.add_argument("--max-model-len", type=int, default=None)
    parser.add_argument("--gpu-memory-utilization", type=float, default=0.9)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--max-samples", type=int, default=None)
    parser.add_argument("--trust-remote-code", action="store_true")
    parser.add_argument("--enforce-eager", action="store_true")
    parser.add_argument("--reaggregate-existing", action="store_true")
    return parser


def main() -> None:
    args = _build_parser().parse_args()
    if args.reaggregate_existing:
        reaggregate_existing_output(
            input_path=args.input,
            output_path=args.output,
            summary_output=args.summary_output,
            summary_csv_output=args.summary_csv_output,
            variant=args.variant,
        )
        return
    if args.batch_size <= 0:
        raise ValueError("--batch-size must be positive")
    run_geval(
        input_path=args.input,
        output_path=args.output,
        summary_output=args.summary_output,
        summary_csv_output=args.summary_csv_output,
        judge=_build_judge(args),
        variant=args.variant,
        cache_dir=args.cache_dir,
        batch_size=args.batch_size,
        max_samples=args.max_samples,
        criterion_judgments_output=args.criterion_judgments_output,
    )


if __name__ == "__main__":
    main()
