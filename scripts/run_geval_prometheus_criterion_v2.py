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
from statistics import mean
from typing import Any, Protocol


EVALUATOR_VERSION = "geval_prometheus_criterion_v2"
SCORING_POLICY_VERSION = "geval_weighted_mean_v1_no_caps"
DEFAULT_MODEL_PATH = "/scratch/l/lovenya/salm_migration_clean/models/eval/prometheus-8x7b-v2.0"
METRIC_IDS = {
    "plausibility": "g_eval_prometheus_criterion_v2_plausibility",
    "reference_grounded": "g_eval_prometheus_criterion_v2_reference_grounded",
}
PROMPT_VERSIONS = {
    "plausibility": "geval_prometheus_plausibility_direct_assessment_v1",
    "reference_grounded": "geval_prometheus_reference_grounded_direct_assessment_v1",
}

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
    "non_contradiction": """Evaluate whether the response explicitly conflicts with the input text.

Score 5: No contradiction; the response is fully compatible with the input.
Score 4: No clear contradiction, with only minor ambiguity or mild tension.
Score 3: Mostly non-contradictory, but includes some confusing, unsupported, or potentially inconsistent detail.
Score 2: Significant tension with the input, though not a direct explicit contradiction.
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
Score 3: Possible but generic, weakly motivated, or somewhat unnatural for the situation.
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
Score 3: Generally possible but includes vague, awkward, or mildly questionable acoustic details.
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
Score 3: Understandable but somewhat stiff, generic, repetitive, or template-like.
Score 2: Unnatural, awkward, or difficult to accept as human speech.
Score 1: Broken, incoherent, excessively repetitive, or not a plausible utterance.""",
    "audio_description_naturalness": """Evaluate whether the generated audio description is a coherent natural description of an audio scene.

Score 5: Natural, coherent, and well-formed audio scene description.
Score 4: Mostly natural with minor awkwardness or wording issue.
Score 3: Understandable but somewhat generic, stiff, repetitive, or template-like.
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


class Judge(Protocol):
    name: str
    cache_identity: str

    def generate(self, prompts: list[str]) -> list[str]:
        """Return one raw judge response per prompt."""


@dataclass(frozen=True)
class CriterionItem:
    row_position: int
    criterion: str
    prompt: str
    prompt_hash: str


class StubJudge:
    name = "stub_geval_prometheus_criterion_v2"
    cache_identity = "stub_geval_prometheus_criterion_v2_v1"

    def generate(self, prompts: list[str]) -> list[str]:
        return ["Feedback: Stub deterministic criterion feedback. [RESULT] 4" for _ in prompts]


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

    def generate(self, prompts: list[str]) -> list[str]:
        formatted = [
            self._tokenizer.apply_chat_template(
                [{"role": "user", "content": prompt}],
                tokenize=False,
                add_generation_prompt=True,
            )
            for prompt in prompts
        ]
        responses = self._llm.generate(formatted, self._sampling_params)
        return [item.outputs[0].text.strip() for item in responses]


def parse_prometheus_result(raw: str) -> dict[str, Any]:
    matches = list(re.finditer(r"\[RESULT\]\s*([^\r\n]*)", raw, flags=re.IGNORECASE))
    if not matches:
        raise ValueError("Missing [RESULT] marker")
    result_text = matches[-1].group(1).strip()
    score_match = re.fullmatch(r"([1-5])", result_text)
    if not score_match:
        raise ValueError(f"Final [RESULT] must contain exactly one integer from 1 to 5: {result_text!r}")
    feedback = raw[: matches[-1].start()].strip()
    feedback = re.sub(r"^\s*Feedback:\s*", "", feedback, flags=re.IGNORECASE).strip()
    if not feedback:
        raise ValueError("Feedback must be non-empty")
    return {"score": int(score_match.group(1)), "feedback": feedback, "parse_status": "parsed", "repair_applied": False}


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


def build_prompt(
    *,
    variant: str,
    direction: str,
    criterion: str,
    input_text: str,
    generated_text: str,
    reference_anchors: list[dict[str, str]] | None = None,
) -> str:
    if (variant, direction) not in CRITERION_WEIGHTS:
        raise ValueError(f"Unsupported variant/direction: {variant}/{direction}")
    rubric = REFERENCE_GROUNDING_RUBRICS[direction] if criterion == "reference_anchor_grounding" else RUBRICS[criterion]
    if variant == "reference_grounded":
        anchors = format_reference_anchors(reference_anchors or [])
        task_description = """The task is not exact reference matching. Multiple outputs can be valid. The reference anchors are examples of the intended target-output space, not the only correct answers. Do not require exact wording or exact scene matching. Use them to judge whether the response is grounded in the dataset's intended paired output space."""
        reference_section = f"\n\n###Reference Anchors:\n{anchors}"
    else:
        task_description = "The task is not paraphrase matching and not exact reference matching. Multiple outputs can be valid. Judge only the requested criterion using the score rubric."
        reference_section = ""
    return f"""###Task Description:
You are evaluating a creative cross-modal generation system.

{task_description}

Judge only the requested criterion using the score rubric.

Direction: {direction}

For audio_to_speech, the input is an environmental audio description and the response is a generated human utterance that could occur in, after, or because of the audio scene.

For speech_to_audio, the input is a speech utterance and the response is a generated environmental audio description that could contextualize, accompany, explain, or be caused by the speech.

Write concise feedback strictly based on the criterion and rubric. Then output the final score as:
Feedback: <feedback> [RESULT] <integer score from 1 to 5>

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


def format_reference_anchors(anchors: list[dict[str, str]]) -> str:
    return "\n\n".join(
        f"[Anchor {index} | source={anchor.get('source', 'unknown')} | id={anchor.get('anchor_id', 'unknown')}]\n{anchor.get('text', '')}"
        for index, anchor in enumerate(anchors, start=1)
    )


def build_repair_prompt(raw_response: str) -> str:
    return f"""Repair the evaluator response into the native Prometheus direct-assessment format.

Return exactly:
Feedback: <non-empty concise feedback> [RESULT] <one integer from 1 to 5>

Do not return JSON. Preserve the intended feedback and score when possible.

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
        for item, raw, generation_error in generated:
            if generation_error:
                result = {
                    "score": None,
                    "feedback": "",
                    "parse_status": "judge_error",
                    "repair_applied": False,
                    "judge_error": generation_error,
                    "raw_response": "",
                }
            else:
                assert raw is not None
                result = _parse_with_repair(judge, raw)
            states[item.row_position]["criteria"][item.criterion] = result
            if result.get("parse_status") == "parsed":
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
            "reference_anchors": anchors if variant == "reference_grounded" else [],
            "prompt_version": PROMPT_VERSIONS[variant],
            "scoring_policy_version": SCORING_POLICY_VERSION,
            "judge": cache_identity,
        },
        ensure_ascii=False,
        sort_keys=True,
    )
    return CriterionItem(row_position, criterion, prompt, hashlib.sha256(cache_key.encode("utf-8")).hexdigest())


def _generate_with_fallback(judge: Judge, items: list[CriterionItem]) -> list[tuple[CriterionItem, str | None, str | None]]:
    try:
        responses = judge.generate([item.prompt for item in items])
        if len(responses) != len(items):
            raise RuntimeError(f"Judge returned {len(responses)} responses for {len(items)} prompts")
        return [(item, raw, None) for item, raw in zip(items, responses)]
    except Exception:
        generated = []
        for item in items:
            try:
                responses = judge.generate([item.prompt])
                if len(responses) != 1:
                    raise RuntimeError(f"Judge returned {len(responses)} responses for one prompt")
                generated.append((item, responses[0], None))
            except Exception as exc:
                generated.append((item, None, f"{type(exc).__name__}: {exc}"))
        return generated


def _parse_with_repair(judge: Judge, raw: str) -> dict[str, Any]:
    try:
        parsed = parse_prometheus_result(raw)
        parsed["raw_response"] = raw
        return parsed
    except Exception as exc:
        initial_error = f"{type(exc).__name__}: {exc}"
        try:
            responses = judge.generate([build_repair_prompt(raw)])
            if len(responses) != 1:
                raise RuntimeError(f"Judge returned {len(responses)} responses for one repair prompt")
            repaired_raw = responses[0]
            parsed = parse_prometheus_result(repaired_raw)
            parsed.update(
                {
                    "repair_applied": True,
                    "initial_raw_response": raw,
                    "raw_response": repaired_raw,
                }
            )
            return parsed
        except Exception as repair_exc:
            return {
                "score": None,
                "feedback": "",
                "parse_status": "parse_failed",
                "repair_applied": True,
                "parse_error": f"initial={initial_error}; repair={type(repair_exc).__name__}: {repair_exc}",
                "initial_raw_response": raw,
                "raw_response": repaired_raw if "repaired_raw" in locals() else "",
            }


def _finalize_state(state: dict[str, Any]) -> None:
    if state.get("incomplete_reason") == "missing_required_reference_anchor":
        state["status"] = "incomplete"
        return
    state["final_score"] = weighted_final_score(state["criteria"], state["criterion_weights"])
    state["diagnostics"] = deterministic_diagnostics(state["criteria"])
    state["status"] = "complete" if state["final_score"] is not None else "incomplete"
    if state["final_score"] is None:
        state["incomplete_reason"] = "one_or_more_required_criteria_failed"
    else:
        state.pop("incomplete_reason", None)


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
) -> dict[str, Any]:
    for state in states:
        _finalize_state(state)
    states.sort(key=lambda row: int(row.get("row_index", 0)))
    summary = _summary(states, input_path=input_path, output_path=output_path, judge_name=judge_name, variant=variant)
    summary["requested_pairs"] = requested_rows
    summary["remaining_pairs"] = sum(state.get("status") != "complete" for state in states)
    summary["complete"] = summary["remaining_pairs"] == 0
    _write_jsonl(output_path, states)
    _write_json(summary_output, summary)
    if summary_csv_output:
        _write_summary_csv(summary_csv_output, summary)
    return summary


def _summary(rows: list[dict[str, Any]], *, input_path: Path, output_path: Path, judge_name: str, variant: str) -> dict[str, Any]:
    valid = [row for row in rows if _is_score(row.get("final_score"))]
    criterion_scores: dict[str, list[float]] = {}
    for row in rows:
        for criterion, result in row.get("criteria", {}).items():
            if _is_score(result.get("score")):
                criterion_scores.setdefault(criterion, []).append(float(result["score"]))
    return {
        "schema_version": 2,
        "metric": METRIC_IDS[variant],
        "evaluator_version": EVALUATOR_VERSION,
        "evaluation_variant": variant,
        "prompt_version": PROMPT_VERSIONS[variant],
        "scoring_policy_version": SCORING_POLICY_VERSION,
        "input_path": str(input_path),
        "output_path": str(output_path),
        "judge_model": judge_name,
        "total_pairs_processed": len(rows),
        "valid_judgments": len(valid),
        "incomplete_judgments": len(rows) - len(valid),
        "mean_final_score": mean(float(row["final_score"]) for row in valid) if valid else None,
        "mean_criterion_scores": {name: mean(values) for name, values in sorted(criterion_scores.items())},
    }


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
    if cached.get("parse_status") != "parsed" or isinstance(score, bool) or not isinstance(score, int) or not 1 <= score <= 5:
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
        writer.writerow(["scope", "criterion", "mean"])
        if summary.get("mean_final_score") is not None:
            writer.writerow(["all", "final_score", summary["mean_final_score"]])
        for criterion, value in summary.get("mean_criterion_scores", {}).items():
            writer.writerow(["all", criterion, value])


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
    return parser


def main() -> None:
    args = _build_parser().parse_args()
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
    )


if __name__ == "__main__":
    main()
