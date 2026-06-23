#!/usr/bin/env python
"""Run criterion-wise LALM-as-a-judge G-Eval for generated audio outputs."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import time
from dataclasses import dataclass
from pathlib import Path
from statistics import mean, median, multimode, pstdev
from typing import Any, Protocol

from project_paths import model_path
from scripts.run_geval_prometheus_criterion_v2 import (
    SCORING_POLICY_VERSION,
    parse_prometheus_result,
    weighted_final_score,
)

EVALUATOR_VERSION = "geval_audio_qwen3_criterion_v1"
DEFAULT_MODEL_PATH = str(model_path("qwen3_omni"))
METRIC_IDS = {
    "plausibility": "geval_audio_plausibility",
    "reference_grounded": "geval_audio_reference",
}
PROMPT_VERSIONS = {
    "plausibility": "geval_audio_plausibility_v3_end_to_end_audio",
    "reference_grounded": "geval_audio_reference_v3_end_to_end_audio",
}
PROMPT_TEMPLATE_NAME = "geval_audio_end_to_end_direct_assessment_v1"
PROMPT_TEMPLATE_TEXT = """###Task Description:
An instruction, audio clips, and a score rubric representing one evaluation criterion are given.
1. Listen carefully to the attached audio clips.
2. Write concise, objective feedback assessing the output audio strictly according to the given score rubric, not in general.
3. After writing the feedback, write a score that is an integer between 1 and 5.
4. The output format should be exactly:
Feedback: (write concise feedback for this criterion) [RESULT] (integer score from 1 to 5)
5. Do not generate any other opening, closing, or explanation.
6. If the criterion genuinely cannot be assessed from the attached audio, write [RESULT] N/A instead of an integer. Do not use N/A merely because the output audio is poor, short, vague, or low quality; use scores 1 or 2 for assessable weak outputs.

###The instruction to evaluate:
{instruction}

###Audio clips:
{audio_clip_description}

###Score Rubric:
{rubric}

###Feedback:
"""

DIRECTION_INSTRUCTIONS = {
    "audio_to_speech": """You are evaluating an end-to-end SpeechAudio LM generation. Listen carefully to both audio clips. The output audio should be a generated human speech response that could plausibly be spoken in, after, or because of the input environmental audio scene.

Evaluate only the specified criterion. Do not evaluate overall quality. Do not assume access to captions, transcripts, or reference text. Base your judgment only on what you can hear in the input audio and output audio.""",
    "speech_to_audio": """You are evaluating an end-to-end SpeechAudio LM generation. Listen carefully to both audio clips. The output audio should be an environmental audio scene that could plausibly contextualize, accompany, explain, or be caused by the input spoken utterance.

Evaluate only the specified criterion. Do not evaluate overall quality. Do not assume access to captions, transcripts, or reference text. Base your judgment only on what you can hear in the input audio and output audio.""",
}

REFERENCE_DIRECTION_INSTRUCTIONS = {
    "audio_to_speech": """You are evaluating an end-to-end SpeechAudio LM generation with reference audio anchors. Listen carefully to the input audio, output audio, and any reference audio clips. The output audio should be generated human speech that could plausibly be spoken in, after, or because of the input environmental audio scene, and should remain compatible with the reference speech audio anchors as soft examples of valid output space.

Evaluate only the specified criterion. Do not evaluate overall quality. Do not assume access to captions, transcripts, or reference text. Base your judgment only on what you can hear in the attached audio clips.""",
    "speech_to_audio": """You are evaluating an end-to-end SpeechAudio LM generation with reference audio anchors. Listen carefully to the input audio, output audio, and any reference audio clips. The output audio should be an environmental audio scene that could plausibly contextualize, accompany, explain, or be caused by the input spoken utterance, and should remain compatible with the reference environmental audio anchors as soft examples of valid output space.

Evaluate only the specified criterion. Do not evaluate overall quality. Do not assume access to captions, transcripts, or reference text. Base your judgment only on what you can hear in the attached audio clips.""",
}

CRITERION_WEIGHTS = {
    ("plausibility", "audio_to_speech"): {
        "cross_modal_non_contradiction": 1.0,
        "cross_modal_grounding": 1.0,
        "scenario_plausibility": 1.0,
        "controlled_detail": 1.0,
        "target_modality_success": 1.0,
        "perceptual_audio_quality": 1.0,
        "expressive_delivery_fit": 1.0,
    },
    ("plausibility", "speech_to_audio"): {
        "cross_modal_non_contradiction": 1.0,
        "cross_modal_grounding": 1.0,
        "scenario_plausibility": 1.0,
        "controlled_detail": 1.0,
        "target_modality_success": 1.0,
        "perceptual_audio_quality": 1.0,
        "contextual_acoustic_fit": 1.0,
    },
    ("reference_grounded", "audio_to_speech"): {
        "cross_modal_non_contradiction": 1.0,
        "cross_modal_grounding": 1.0,
        "scenario_plausibility": 1.0,
        "controlled_detail": 1.0,
        "target_modality_success": 1.0,
        "perceptual_audio_quality": 1.0,
        "expressive_delivery_fit": 1.0,
        "reference_audio_compatibility": 1.0,
    },
    ("reference_grounded", "speech_to_audio"): {
        "cross_modal_non_contradiction": 1.0,
        "cross_modal_grounding": 1.0,
        "scenario_plausibility": 1.0,
        "controlled_detail": 1.0,
        "target_modality_success": 1.0,
        "perceptual_audio_quality": 1.0,
        "contextual_acoustic_fit": 1.0,
        "reference_audio_compatibility": 1.0,
    },
}

RUBRICS = {
    ("audio_to_speech", "cross_modal_non_contradiction"): """Evaluate whether the generated speech audio avoids contradicting the heard input environmental audio.

Focus only on contradiction. The speech may add reasonable unstated context, but it should not deny, reverse, or conflict with explicit or strongly implied events, conditions, or situations heard in the input audio.

Score 1: The generated speech directly and severely contradicts the input audio scene, or is incoherent with it.
Score 2: The generated speech contains a major contradiction with the input audio scene or with a strongly implied fact from it.
Score 3: The generated speech has a noticeable possible contradiction or tension, but the conflict is ambiguous or not central.
Score 4: The generated speech is mostly compatible with the input audio scene, with only minor ambiguity or weak tension.
Score 5: The generated speech is fully compatible with the input audio scene and contains no meaningful contradiction.""",
    ("audio_to_speech", "cross_modal_grounding"): """Evaluate whether the generated speech audio is meaningfully grounded in the heard input environmental audio.

The speech should use, respond to, or be motivated by the sound event, acoustic context, or situation heard in the input audio. It does not need to describe the audio literally, but it should not be generic, unrelated, or only accidentally compatible.

Score 1: The generated speech is unrelated to the input audio.
Score 2: The generated speech has only a very weak, vague, or accidental connection to the input audio.
Score 3: The generated speech is somewhat connected, but the connection is generic, indirect, or underdeveloped.
Score 4: The generated speech is clearly connected to the input audio, with only minor genericness or missed opportunity.
Score 5: The generated speech is strongly and specifically grounded in the input audio.""",
    ("audio_to_speech", "scenario_plausibility"): """Evaluate whether the input environmental audio and generated speech audio form a believable shared situation.

The generated speech should sound like something a human could plausibly say during, after, or because of the heard audio scene. Consider temporal fit, causal fit, contextual emotional fit, pragmatic fit, and human reaction plausibility. These are not separate sub-scores; use them only to judge overall scenario plausibility.

Score 1: The input audio and generated speech form an absurd, impossible, incoherent, or extremely unnatural situation.
Score 2: The generated speech is difficult to believe in the heard scene because of a major temporal, causal, emotional, pragmatic, or human-reaction mismatch.
Score 3: The generated speech could occur in the heard scene, but the situation feels somewhat strained, weakly motivated, generic, or dependent on extra context.
Score 4: The generated speech forms a plausible human situation with the heard scene, requiring at most a small reasonable assumption.
Score 5: The generated speech feels like a very natural and believable human reaction to the heard scene.""",
    ("audio_to_speech", "controlled_detail"): """Evaluate whether the generated speech audio uses an appropriate amount of inferred detail: not too vague, and not too over-specific.

The speech may infer reasonable context from the heard input audio because the task is one-to-many. However, it should not invent highly specific unsupported details, identities, locations, causes, or events. It should also not be so vague that it barely uses the input audio.

Score 1: The generated speech is either almost empty/generic or wildly over-invented with unsupported details.
Score 2: The generated speech is mostly too vague or mostly too over-specific, making it poorly controlled.
Score 3: The generated speech has some useful detail, but is noticeably too generic or includes somewhat unsupported details.
Score 4: The generated speech has mostly appropriate detail, with only minor vagueness or minor over-inference.
Score 5: The generated speech is appropriately detailed: concrete enough to be useful, but not over-invented beyond the input audio.""",
    ("audio_to_speech", "target_modality_success"): """Evaluate whether the output audio succeeds as generated human speech.

The output should contain intelligible, speakable human speech. It should not be only environmental sound, music, silence, noise, a non-speech vocalization, a spoken audio caption, or an incoherent waveform. Judge the success of the target modality, not whether the content is semantically perfect.

Score 1: The output is not recognizable as valid human speech, is silent/noise, or is completely in the wrong modality.
Score 2: The output contains little usable speech, or the speech is mostly unintelligible, broken, or not a plausible utterance.
Score 3: The output is partly valid speech, but has noticeable intelligibility, modality, or utterance-form problems.
Score 4: The output is mostly successful speech audio, with only minor intelligibility or utterance-form issues.
Score 5: The output is clearly successful human speech audio: intelligible, speakable, and in the correct target modality.""",
    ("audio_to_speech", "perceptual_audio_quality"): """Evaluate the perceptual quality and naturalness of the generated speech audio.

Focus on audio quality, not semantic alignment. Consider naturalness, clarity, distortion, clipping, glitches, robotic artifacts, unstable voice, excessive noise, unnatural pauses, and synthesis artifacts.

Score 1: The generated speech audio is extremely degraded, unnatural, distorted, or difficult to listen to.
Score 2: The generated speech audio has major quality problems such as strong artifacts, distortion, instability, or poor clarity.
Score 3: The generated speech audio is understandable but has noticeable artifacts, unnaturalness, noise, or synthesis issues.
Score 4: The generated speech audio is mostly natural and clear, with only minor artifacts or quality issues.
Score 5: The generated speech audio is highly natural, clear, stable, and free of meaningful perceptual artifacts.""",
    ("audio_to_speech", "expressive_delivery_fit"): """Evaluate whether the expressive delivery of the generated speech fits the heard input audio scene.

Focus on non-textual speech cues: emotion, prosody, tone, urgency, loudness, speaking pace, rhythm, hesitation, stress, surprise, fear, calmness, laughter, or other vocal expression. The words may be plausible, but the delivery should also fit the situation implied by the input audio. Do not require strong emotion when the situation is neutral.

Score 1: The expressive delivery is completely inappropriate for the heard scene, or the speech has no usable expressive character.
Score 2: The delivery has a major mismatch in emotion, prosody, urgency, loudness, pace, or tone.
Score 3: The delivery is somewhat plausible, but feels flat, exaggerated, emotionally mismatched, or weakly adapted to the scene.
Score 4: The delivery mostly fits the heard scene, with only minor expressive mismatch or missed nuance.
Score 5: The delivery strongly fits the heard scene, with natural emotion, prosody, urgency, pace, and tone for the situation.""",
    ("speech_to_audio", "cross_modal_non_contradiction"): """Evaluate whether the generated environmental audio avoids contradicting the heard input speech.

Focus only on contradiction. The environmental audio may add reasonable unstated context, but it should not deny, reverse, or conflict with explicit or strongly implied facts, intent, emotion, or context from the spoken utterance.

Score 1: The generated environmental audio directly and severely contradicts the input speech, or is incoherent with it.
Score 2: The generated environmental audio contains a major contradiction with the input speech or with a strongly implied fact from it.
Score 3: The generated environmental audio has a noticeable possible contradiction or tension, but the conflict is ambiguous or not central.
Score 4: The generated environmental audio is mostly compatible with the input speech, with only minor ambiguity or weak tension.
Score 5: The generated environmental audio is fully compatible with the input speech and contains no meaningful contradiction.""",
    ("speech_to_audio", "cross_modal_grounding"): """Evaluate whether the generated environmental audio is meaningfully grounded in the heard input speech.

The audio scene should contextualize, accompany, explain, or be caused by the spoken utterance. It does not need to be the only possible scene, but it should not be generic, unrelated, or only accidentally compatible.

Score 1: The generated environmental audio is unrelated to the input speech.
Score 2: The generated environmental audio has only a very weak, vague, or accidental connection to the input speech.
Score 3: The generated environmental audio is somewhat connected, but the connection is generic, indirect, or underdeveloped.
Score 4: The generated environmental audio is clearly connected to the input speech, with only minor genericness or missed opportunity.
Score 5: The generated environmental audio is strongly and specifically grounded in the input speech.""",
    ("speech_to_audio", "scenario_plausibility"): """Evaluate whether the input speech and generated environmental audio form a believable shared situation.

The generated audio scene should realistically contextualize, accompany, explain, or result from the heard utterance. Consider temporal fit, causal fit, contextual emotional fit, pragmatic fit, and whether the described sound world is believable. These are not separate sub-scores; use them only to judge overall scenario plausibility.

Score 1: The input speech and generated environmental audio form an absurd, impossible, incoherent, or extremely unnatural situation.
Score 2: The generated audio scene is difficult to believe with the input speech because of a major temporal, causal, emotional, pragmatic, or sound-world mismatch.
Score 3: The generated audio scene could fit the input speech, but the situation feels somewhat strained, weakly motivated, generic, or dependent on extra context.
Score 4: The generated audio scene forms a plausible situation with the input speech, requiring at most a small reasonable assumption.
Score 5: The generated audio scene feels like a very natural and believable context, accompaniment, explanation, or consequence of the input speech.""",
    ("speech_to_audio", "controlled_detail"): """Evaluate whether the generated environmental audio uses an appropriate amount of acoustic detail: not too sparse, and not too overloaded or over-specific.

The audio may infer reasonable context from the heard input speech because the task is one-to-many. However, it should not introduce many unsupported or overly specific sound events, identities, locations, or causes. It should also not be so sparse or generic that it barely uses the input speech.

Score 1: The generated audio is either almost empty/generic or wildly overloaded with unsupported acoustic details.
Score 2: The generated audio is mostly too sparse/generic or mostly too over-specific/overcrowded, making it poorly controlled.
Score 3: The generated audio has some useful acoustic detail, but is noticeably too generic, too sparse, or somewhat over-invented.
Score 4: The generated audio has mostly appropriate acoustic detail, with only minor sparseness or minor over-inference.
Score 5: The generated audio is appropriately detailed: rich enough to be useful, but not overloaded or over-invented beyond the input speech.""",
    ("speech_to_audio", "target_modality_success"): """Evaluate whether the output audio succeeds as generated environmental audio.

The output should contain an environmental sound scene, sound event, ambience, or acoustic context. It should not be generated speech replying to the speaker, a spoken caption, pure silence, unrelated music, random noise, or an incoherent waveform. Judge the success of the target modality, not whether the semantic alignment is perfect.

Score 1: The output is not recognizable as valid environmental audio, is silent/noise, or is completely in the wrong modality.
Score 2: The output contains little usable environmental audio, or is mostly speech, music, silence, random noise, or broken audio.
Score 3: The output is partly valid environmental audio, but has noticeable modality, scene, or acoustic-content problems.
Score 4: The output is mostly successful environmental audio, with only minor modality or acoustic-content issues.
Score 5: The output is clearly successful environmental audio: audible, scene-like, and in the correct target modality.""",
    ("speech_to_audio", "perceptual_audio_quality"): """Evaluate the perceptual quality and realism of the generated environmental audio.

Focus on audio quality, not semantic alignment. Consider realism, clarity, natural sound texture, distortion, clipping, glitches, looping artifacts, harsh synthetic noise, poor mixing, unstable ambience, and unnatural transitions.

Score 1: The generated environmental audio is extremely degraded, unrealistic, distorted, or difficult to listen to.
Score 2: The generated environmental audio has major quality problems such as strong artifacts, distortion, poor realism, or unstable sound texture.
Score 3: The generated environmental audio is usable but has noticeable artifacts, unnaturalness, poor mixing, looping, or synthetic texture.
Score 4: The generated environmental audio is mostly realistic and clear, with only minor artifacts or quality issues.
Score 5: The generated environmental audio is highly realistic, clear, coherent, and free of meaningful perceptual artifacts.""",
    ("speech_to_audio", "contextual_acoustic_fit"): """Evaluate whether the acoustic character of the generated environmental audio fits the heard input speech.

Focus on non-textual acoustic cues: loudness, intensity, ambience, mood, tension, calmness, event timing, foreground/background balance, density, and soundscape energy. The audio scene may be semantically plausible, but its acoustic character should also fit the utterance's intent, emotion, urgency, and pragmatic context. Do not require dramatic sound when the speech implies a neutral or quiet situation.

Score 1: The acoustic character is completely inappropriate for the heard speech, or the output has no usable acoustic character.
Score 2: The generated audio has a major mismatch in loudness, intensity, ambience, timing, density, or acoustic mood.
Score 3: The generated audio is somewhat plausible, but feels too flat, too intense, emotionally mismatched, poorly timed, or weakly adapted to the speech.
Score 4: The generated audio mostly fits the heard speech, with only minor acoustic mismatch or missed nuance.
Score 5: The generated audio strongly fits the heard speech, with natural loudness, intensity, ambience, timing, density, and acoustic mood for the situation.""",
    ("audio_to_speech", "reference_audio_compatibility"): """Evaluate whether the generated speech audio is compatible with the attached reference speech audio anchor(s) as soft examples of valid responses.

The generated speech does not need to match the exact wording, voice, timing, or delivery of the reference audio. It should remain compatible with the reference's core audible situation, communicative role, reaction, intent, or emotional stance when those are clear.

Score 1: The generated speech is incompatible with the reference audio's core response, reaction, intent, or communicative role.
Score 2: The generated speech has only weak or strained compatibility with the reference audio and misses or conflicts with an important anchor.
Score 3: The generated speech is partially compatible with the reference audio, but changes, weakens, or omits a noticeable part of the reference anchor.
Score 4: The generated speech is mostly compatible with the reference audio's core anchor, with only minor differences in emphasis, specificity, emotion, wording, or delivery.
Score 5: The generated speech is strongly compatible with the reference audio's core anchor while allowing natural variation in wording, phrasing, voice, and delivery.""",
    ("speech_to_audio", "reference_audio_compatibility"): """Evaluate whether the generated environmental audio is compatible with the attached reference environmental audio anchor(s) as soft examples of valid output scenes.

The generated environmental audio does not need to match the exact same sound list, timing, recording conditions, or scene details. It should remain compatible with the reference's core sound event, acoustic environment, causal context, scene type, or contextual role when those are clear.

Score 1: The generated environmental audio is incompatible with the reference audio's core sound event, acoustic environment, causal context, or scene role.
Score 2: The generated environmental audio has only weak or strained compatibility with the reference audio and misses or conflicts with an important anchor.
Score 3: The generated environmental audio is partially compatible with the reference audio, but changes, weakens, or omits a noticeable part of the reference acoustic/contextual anchor.
Score 4: The generated environmental audio is mostly compatible with the reference audio's core anchor, with only minor differences in emphasis, specificity, sound details, timing, or texture.
Score 5: The generated environmental audio is strongly compatible with the reference audio's core anchor while allowing natural variation in sound detail, scene realization, and acoustic texture.""",
}

CRITERION_BOUNDARIES = {
    "cross_modal_non_contradiction": "Judge only contradiction between the heard input audio and generated output audio.",
    "cross_modal_grounding": "Judge only whether the output audio is meaningfully grounded in the heard input audio.",
    "scenario_plausibility": "Judge only whether the input and output audio clips form a believable shared situation.",
    "controlled_detail": "Judge only whether the output audio uses an appropriate amount of inferred detail.",
    "target_modality_success": "Judge only whether the output audio succeeds in the required target modality.",
    "perceptual_audio_quality": "Judge only perceptual audio quality and naturalness/realism, not semantic alignment.",
    "expressive_delivery_fit": "Judge only non-textual expressive speech delivery fit to the input audio scene.",
    "contextual_acoustic_fit": "Judge only non-textual acoustic character fit to the input speech.",
    "reference_audio_compatibility": "Judge compatibility with attached reference audio anchors as soft examples, not exact matching.",
}


class Judge(Protocol):
    name: str
    cache_identity: str

    def generate_with_traces(self, items: list["CriterionItem"]) -> list["JudgeGeneration"]:
        """Return one raw judge response and exact backend prompt per item."""


@dataclass(frozen=True)
class CriterionItem:
    row_position: int
    criterion: str
    prompt: str
    prompt_hash: str
    prompt_record: dict[str, Any]
    audio_files: list[dict[str, str]]


@dataclass(frozen=True)
class JudgeGeneration:
    raw_response: str
    backend_formatted_prompt: str
    backend_format: str
    generation_seconds: float | None = None


class StubJudge:
    name = "stub_geval_audio_qwen3"
    cache_identity = "stub_geval_audio_qwen3_v1"

    def generate_with_traces(self, items: list[CriterionItem]) -> list[JudgeGeneration]:
        return [
            JudgeGeneration(
                raw_response="Feedback: Stub deterministic multimodal criterion feedback. [RESULT] 4",
                backend_formatted_prompt=item.prompt,
                backend_format="identity",
            )
            for item in items
        ]


class TransformersQwen3OmniJudge:
    def __init__(
        self,
        model_path: str,
        *,
        temperature: float,
        max_tokens: int,
        local_files_only: bool,
        trust_remote_code: bool,
    ) -> None:
        try:
            import torch
            from transformers import Qwen3OmniMoeForConditionalGeneration, Qwen3OmniMoeProcessor
        except ImportError as exc:
            raise RuntimeError("GEval audio requires torch and transformers in the Qwen3-Omni metric environment.") from exc

        self.name = model_path
        self.temperature = temperature
        self.max_tokens = max_tokens
        load_started = time.perf_counter()
        self.processor = Qwen3OmniMoeProcessor.from_pretrained(
            model_path,
            local_files_only=local_files_only,
            trust_remote_code=trust_remote_code,
            use_fast=False,
        )
        self.model = Qwen3OmniMoeForConditionalGeneration.from_pretrained(
            model_path,
            local_files_only=local_files_only,
            trust_remote_code=trust_remote_code,
            dtype=torch.bfloat16 if torch.cuda.is_available() else torch.float32,
            device_map="auto" if torch.cuda.is_available() else None,
        )
        self.model_load_seconds = time.perf_counter() - load_started
        self.cache_identity = json.dumps(
            {
                "model": model_path,
                "backend": "transformers_qwen3_omni",
                "temperature": temperature,
                "max_tokens": max_tokens,
                "local_files_only": local_files_only,
                "trust_remote_code": trust_remote_code,
            },
            sort_keys=True,
        )

    def generate_with_traces(self, items: list[CriterionItem]) -> list[JudgeGeneration]:
        # Qwen3-Omni audio preprocessing is easier and safer per item because each
        # criterion row can carry a different number of audio attachments.
        return [self._generate_one(item) for item in items]

    def _generate_one(self, item: CriterionItem) -> JudgeGeneration:
        import torch
        from qwen_omni_utils import process_mm_info

        generation_started = time.perf_counter()
        conversation = [{"role": "user", "content": _qwen_content(item)}]
        text = self.processor.apply_chat_template(conversation, add_generation_prompt=True, tokenize=False)
        audios, images, videos = process_mm_info(conversation, use_audio_in_video=False)
        inputs = self.processor(
            text=text,
            audio=audios,
            images=images,
            videos=videos,
            return_tensors="pt",
            padding=True,
            use_audio_in_video=False,
        )
        inputs = inputs.to(self.model.device)
        for key, value in list(inputs.items()):
            if getattr(value, "is_floating_point", lambda: False)():
                inputs[key] = value.to(self.model.dtype)
        generate_kwargs = {
            "max_new_tokens": self.max_tokens,
            "return_audio": False,
            "thinker_return_dict_in_generate": True,
        }
        if self.temperature > 0:
            generate_kwargs.update({"do_sample": True, "temperature": self.temperature})
        else:
            generate_kwargs.update({"do_sample": False})
        with torch.inference_mode():
            output_ids = self.model.generate(**inputs, **generate_kwargs)
        sequences = getattr(output_ids, "sequences", output_ids)
        decoded = self.processor.batch_decode(
            sequences[:, inputs["input_ids"].shape[1] :],
            skip_special_tokens=True,
            clean_up_tokenization_spaces=False,
        )[0].strip()
        generation_seconds = time.perf_counter() - generation_started
        return JudgeGeneration(
            raw_response=decoded,
            backend_formatted_prompt=text,
            backend_format="qwen3_omni_chat_template",
            generation_seconds=generation_seconds,
        )


def _qwen_content(item: CriterionItem) -> list[dict[str, str]]:
    content: list[dict[str, str]] = [{"type": "text", "text": item.prompt}]
    for audio in item.audio_files:
        content.append({"type": "audio", "audio": audio["path"]})
    return content


def build_prompt_record(
    *,
    variant: str,
    direction: str,
    criterion: str,
    generated_audio_file: str,
    source_audio_file: str,
    reference_anchors: list[dict[str, Any]] | None = None,
    input_text: str | None = None,
    generated_text: str | None = None,
) -> dict[str, Any]:
    if (variant, direction) not in CRITERION_WEIGHTS:
        raise ValueError(f"Unsupported variant/direction: {variant}/{direction}")
    if direction not in DIRECTION_INSTRUCTIONS:
        raise ValueError(f"Unsupported direction: {direction}")
    rubric = RUBRICS[(direction, criterion)]
    if variant == "reference_grounded" and criterion == "reference_audio_compatibility":
        instruction = REFERENCE_DIRECTION_INSTRUCTIONS[direction]
    else:
        instruction = DIRECTION_INSTRUCTIONS[direction]
    criterion_boundary = CRITERION_BOUNDARIES[criterion]
    audio_clip_description = _audio_clip_description(generated_audio_file, source_audio_file, reference_anchors or [])
    rendered_rubric = f"Criterion: {criterion}\n\n{rubric}"
    rendered_user_message = PROMPT_TEMPLATE_TEXT.format(
        instruction=instruction,
        audio_clip_description=audio_clip_description,
        rubric=rendered_rubric,
    )
    return {
        "prompt_version": PROMPT_VERSIONS[variant],
        "template_name": PROMPT_TEMPLATE_NAME,
        "template": PROMPT_TEMPLATE_TEXT,
        "messages": [{"role": "user", "content": rendered_user_message}],
        "rendered_user_message": rendered_user_message,
        "instruction": instruction,
        "criterion_boundary": criterion_boundary,
        "rubric": rubric,
        "rendered_rubric": rendered_rubric,
        "audio_clip_description": audio_clip_description,
        "reference_section": _reference_audio_section(reference_anchors or []),
        "audio_attachments": _audio_attachments(generated_audio_file, source_audio_file, reference_anchors or []),
    }


def _audio_clip_description(generated_audio_file: str, source_audio_file: str, reference_anchors: list[dict[str, Any]]) -> str:
    lines = [
        f"Input audio: attached file {Path(source_audio_file).name}",
        f"Output audio to evaluate: attached file {Path(generated_audio_file).name}",
    ]
    for index, anchor in enumerate(reference_anchors, start=1):
        audio_file = anchor.get("audio_file")
        if isinstance(audio_file, str) and audio_file.strip():
            lines.append(f"Reference audio anchor {index}: attached file {Path(audio_file).name}")
    return "\n".join(lines)


def _reference_audio_section(anchors: list[dict[str, Any]]) -> str:
    parts = []
    for index, anchor in enumerate(anchors, start=1):
        audio_file = anchor.get("audio_file")
        if isinstance(audio_file, str) and audio_file.strip():
            parts.append(f"Reference audio anchor {index}: {Path(audio_file).name}")
    return "\n".join(parts)


def _audio_attachments(generated_audio_file: str, source_audio_file: str, reference_anchors: list[dict[str, Any]]) -> list[dict[str, str]]:
    attachments = [
        {"role": "input_audio", "path": source_audio_file},
        {"role": "generated_audio_to_evaluate", "path": generated_audio_file},
    ]
    for anchor in reference_anchors:
        audio_file = anchor.get("audio_file")
        if isinstance(audio_file, str) and audio_file.strip():
            attachments.append({"role": "reference_anchor_audio", "path": audio_file})
    return attachments


def deterministic_diagnostics(criteria: dict[str, dict[str, Any]]) -> dict[str, Any]:
    scores = {name: value.get("score") for name, value in criteria.items() if isinstance(value, dict)}
    failure_modes: list[str] = []

    def low(name: str) -> bool:
        score = scores.get(name)
        return isinstance(score, (int, float)) and 1 <= float(score) <= 2

    mappings = [
        (low("cross_modal_non_contradiction"), "clear_cross_modal_contradiction_or_major_tension"),
        (low("cross_modal_grounding"), "weak_or_no_cross_modal_grounding"),
        (low("scenario_plausibility"), "implausible_shared_audio_situation"),
        (low("controlled_detail"), "poorly_controlled_audio_detail"),
        (low("target_modality_success"), "target_modality_failure"),
        (low("perceptual_audio_quality"), "poor_perceptual_audio_quality"),
        (low("expressive_delivery_fit"), "poor_expressive_delivery_fit"),
        (low("contextual_acoustic_fit"), "poor_contextual_acoustic_fit"),
        (low("reference_audio_compatibility"), "weak_reference_audio_compatibility"),
    ]
    for applies, label in mappings:
        if applies:
            failure_modes.append(label)
    return {"failure_modes": failure_modes, "failure_mode_source": "deterministic"}


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
                result = _parse_no_repair(generation.raw_response)
                result["runtime_prompt"] = _runtime_prompt_trace(generation)
                if generation.generation_seconds is not None:
                    result["generation_seconds"] = round(generation.generation_seconds, 6)
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
        "input_text": row.get("input_text"),
        "generated_text": row.get("generated_text"),
        "generated_audio_file": _required_existing_file(row, "generated_audio_file"),
        "generated_audio_kind": row.get("generated_audio_kind"),
        "source_audio_file": _required_existing_file(row, "source_audio_file"),
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
    return bool(anchors) and all(
        isinstance(anchor, dict)
        and isinstance(anchor.get("audio_file"), str)
        and Path(anchor["audio_file"]).is_file()
        for anchor in anchors
    )


def _make_item(row: dict[str, Any], row_position: int, criterion: str, *, variant: str, cache_identity: str) -> CriterionItem:
    direction = _required_str(row, "direction")
    anchors = row.get("reference_anchors") if isinstance(row.get("reference_anchors"), list) else []
    generated_audio_file = _required_existing_file(row, "generated_audio_file")
    source_audio_file = _required_existing_file(row, "source_audio_file")
    prompt_record = build_prompt_record(
        variant=variant,
        direction=direction,
        criterion=criterion,
        generated_audio_file=generated_audio_file,
        source_audio_file=source_audio_file,
        reference_anchors=anchors,
        input_text=row.get("input_text") if isinstance(row.get("input_text"), str) else None,
        generated_text=row.get("generated_text") if isinstance(row.get("generated_text"), str) else None,
    )
    prompt = prompt_record["rendered_user_message"]
    cache_key = json.dumps(
        {
            "metric": METRIC_IDS[variant],
            "evaluator_version": EVALUATOR_VERSION,
            "evaluation_variant": variant,
            "direction": direction,
            "criterion": criterion,
            "input_audio_file": source_audio_file,
            "generated_audio_file": generated_audio_file,
            "generated_audio_kind": row.get("generated_audio_kind"),
            "reference_anchors": anchors if criterion == "reference_audio_compatibility" else [],
            "prompt_version": PROMPT_VERSIONS[variant],
            "scoring_policy_version": SCORING_POLICY_VERSION,
            "judge": cache_identity,
        },
        ensure_ascii=False,
        sort_keys=True,
    )
    return CriterionItem(
        row_position=row_position,
        criterion=criterion,
        prompt=prompt,
        prompt_hash=hashlib.sha256(cache_key.encode("utf-8")).hexdigest(),
        prompt_record=prompt_record,
        audio_files=prompt_record["audio_attachments"],
    )


def _generate_with_fallback(judge: Judge, items: list[CriterionItem]) -> list[tuple[CriterionItem, JudgeGeneration | None, str | None]]:
    try:
        generations = judge.generate_with_traces(items)
        if len(generations) != len(items):
            raise RuntimeError(f"Judge returned {len(generations)} responses for {len(items)} prompts")
        return [(item, generation, None) for item, generation in zip(items, generations)]
    except Exception:
        generated = []
        for item in items:
            try:
                generations = judge.generate_with_traces([item])
                if len(generations) != 1:
                    raise RuntimeError(f"Judge returned {len(generations)} responses for one prompt")
                generated.append((item, generations[0], None))
            except Exception as exc:
                generated.append((item, None, f"{type(exc).__name__}: {exc}"))
        return generated


def _parse_no_repair(raw: str) -> dict[str, Any]:
    try:
        parsed = parse_prometheus_result(raw)
        parsed["raw_response"] = raw
        return parsed
    except Exception as exc:
        return {
            "score": None,
            "feedback": "",
            "parse_status": "parse_failed",
            "parse_repair_attempted": False,
            "parse_repair_succeeded": False,
            "parse_repair_method": None,
            "parse_error": f"{type(exc).__name__}: {exc}",
            "initial_raw_response": raw,
            "raw_response": raw,
        }


def _runtime_prompt_trace(generation: JudgeGeneration) -> dict[str, Any]:
    return {
        "backend_formatted_prompt": generation.backend_formatted_prompt,
        "backend_format": generation.backend_format,
        "trace_source": "generation_call",
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
    _attach_runtime_statistics(summary, states, judge=judge)
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


def _attach_runtime_statistics(summary: dict[str, Any], states: list[dict[str, Any]], *, judge: Judge | None) -> None:
    generation_times = [
        float(result["generation_seconds"])
        for state in states
        for result in state.get("criteria", {}).values()
        if isinstance(result, dict) and isinstance(result.get("generation_seconds"), (int, float))
    ]
    processed_rows = max(1, len(states))
    runtime = {
        "model_load_seconds": round(float(getattr(judge, "model_load_seconds", 0.0)), 6) if judge is not None else None,
        "judge_generations": len(generation_times),
        "judge_generation_seconds_total": round(sum(generation_times), 6),
        "judge_generation_seconds_mean": round(mean(generation_times), 6) if generation_times else None,
        "judge_generation_seconds_median": round(median(generation_times), 6) if generation_times else None,
        "judge_generation_seconds_max": round(max(generation_times), 6) if generation_times else None,
        "judge_generation_seconds_per_sample": round(sum(generation_times) / processed_rows, 6) if generation_times else None,
    }
    summary["runtime"] = runtime


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
    return {
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
            "generated_audio_file": state.get("generated_audio_file"),
            "generated_audio_kind": state.get("generated_audio_kind"),
            "source_audio_file": state.get("source_audio_file"),
            "reference_anchors": state.get("reference_anchors", []),
        },
        "prompt": prompt_record,
        "judge": {"model": state.get("judge_model")},
        "output": _criterion_judgment_output(result),
        "error": _criterion_judgment_error(state, result),
    }


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
        return {"score": None, "range": {"min": 1, "max": 5}, "higher_is_better": True, "feedback": None, "raw_response": None, "parse_status": "missing"}
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
    for key in ("initial_parse_error", "initial_raw_response", "abstention_reason", "judge_error", "parse_error"):
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
            metadata = criterion_metadata.setdefault(criterion, {"eligible_count": 0, "abstention_count": 0, "failure_count": 0, "missing_count": 0})
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
    criterion_statistics = {name: {**_score_statistics(criterion_scores.get(name, []), total_count=metadata["eligible_count"], distribution="exact_1_to_5"), **metadata} for name, metadata in sorted(criterion_metadata.items())}
    prompt_versions = sorted({str(row.get("prompt_version")) for row in rows if row.get("prompt_version")})
    prompt_version = prompt_versions[0] if len(prompt_versions) == 1 else prompt_versions
    final_statistics = _score_statistics(final_scores, total_count=len(rows), distribution="fixed_half_point_bins")
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
        "abstained_judgments": status_counts.get("complete_with_abstention", 0),
        "failed_judgments": status_counts.get("incomplete", 0),
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
    return str(result.get("parse_status") or "missing")


def _score_statistics(values: list[float], *, total_count: int, distribution: str) -> dict[str, Any]:
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
        labels = ["[1.0,1.5)", "[1.5,2.0)", "[2.0,2.5)", "[2.5,3.0)", "[3.0,3.5)", "[3.5,4.0)", "[4.0,4.5)", "[4.5,5.0]"]
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


def _required_existing_file(row: dict[str, Any], field: str) -> str:
    value = _required_str(row, field)
    if not Path(value).is_file():
        raise ValueError(f"Row {row.get('row_index')} has no existing {field}: {value}")
    return value


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
        fields = ["count", "null_count", "mean", "std", "min", "max", "median", "mode", "modes", "mode_count", "distribution_type", "score_distribution", "eligible_count", "abstention_count", "failure_count", "missing_count"]
        writer.writerow(["scope", "criterion", *fields])
        statistics = {"final_score": summary.get("final_score_statistics", {})}
        statistics.update(summary.get("criterion_score_statistics", {}))
        for criterion, values in statistics.items():
            writer.writerow(["all", criterion] + [json.dumps(values.get(field), sort_keys=True) if field in {"modes", "score_distribution"} else values.get(field) for field in fields])


def _build_judge(args: argparse.Namespace) -> Judge:
    if args.backend == "stub":
        return StubJudge()
    return TransformersQwen3OmniJudge(
        args.model,
        temperature=args.temperature,
        max_tokens=args.max_tokens,
        local_files_only=args.local_files_only,
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
    parser.add_argument("--backend", choices=["transformers", "stub"], default="transformers")
    parser.add_argument("--model", default=DEFAULT_MODEL_PATH)
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--max-tokens", type=int, default=512)
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--max-samples", type=int, default=None)
    parser.add_argument("--local-files-only", action="store_true")
    parser.add_argument("--trust-remote-code", action="store_true")
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
        criterion_judgments_output=args.criterion_judgments_output,
    )


if __name__ == "__main__":
    main()
