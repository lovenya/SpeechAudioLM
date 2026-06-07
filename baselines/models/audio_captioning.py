from __future__ import annotations

import json
from pathlib import Path

from baselines.dataset import PromptRow
from baselines.models.base import BaselineRunner, GenerationResult

DEFAULT_CAPTION_PROMPT = (
    "Describe the environmental audio in this recording. Focus on sound events, setting, intensity, "
    "and mood. Return only one concise caption."
)
AFN_SHORT_CAPTION_PROMPT = "Generate a caption for the input audio."


class Qwen3OmniCaptionerRunner(BaselineRunner):
    output_subdir = "text"
    extension = ".json"

    def __init__(self, **kwargs: object) -> None:
        super().__init__(**kwargs)
        if self.task != "A2T":
            raise ValueError("Qwen3-Omni captioner supports only task A2T")
        self.model_path = self.model_path or "/scratch/l/lovenya/salm_migration_clean/models/cascaded/captioning/qwen3-omni-30b-a3b-captioner"
        self.processor = None
        self.model = None

    def load(self) -> None:
        import torch
        from transformers import Qwen3OmniMoeForConditionalGeneration, Qwen3OmniMoeProcessor

        self.processor = Qwen3OmniMoeProcessor.from_pretrained(
            self.model_path,
            local_files_only=True,
            use_fast=False,
        )
        self.model = Qwen3OmniMoeForConditionalGeneration.from_pretrained(
            self.model_path,
            local_files_only=True,
            dtype=torch.bfloat16,
            device_map="auto" if self.device.startswith("cuda") else None,
        )

    def generate(self, row: PromptRow, output_path: Path) -> GenerationResult:
        if self.processor is None or self.model is None:
            raise RuntimeError("Qwen3-Omni captioner is not loaded")
        import torch
        from qwen_omni_utils import process_mm_info

        audio_path = self._record_path(row, "audio_file")
        conversation = [
            {
                "role": "user",
                "content": [
                    {"type": "audio", "audio": str(audio_path)},
                ],
            }
        ]
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
        with torch.inference_mode():
            output_ids = self.model.generate(**inputs, max_new_tokens=128, do_sample=False)
        sequences = getattr(output_ids, "sequences", output_ids)
        caption = self.processor.batch_decode(
            sequences[:, inputs["input_ids"].shape[1] :],
            skip_special_tokens=True,
            clean_up_tokenization_spaces=False,
        )[0].strip()
        extra = {
            "source_input_file": str(audio_path),
            "understanding_prompt": None,
            "messages": conversation,
            "understanding_text": caption,
            "backend": "qwen3_omni_captioner",
            "stages": [
                {
                    "stage": "audio_captioning",
                    "method": "Qwen3OmniMoeForConditionalGeneration.generate",
                    "input_field": "audio_file",
                    "input_file": str(audio_path),
                    "prompt": None,
                    "output_text": caption,
                    "output_file": str(output_path),
                }
            ],
        }
        _write_text_result(output_path, row, caption, extra)
        return GenerationResult(output_path=output_path, sample_rate=None, extra=extra)


class AudioFlamingoCaptionerRunner(BaselineRunner):
    output_subdir = "text"
    extension = ".json"

    def __init__(self, **kwargs: object) -> None:
        super().__init__(**kwargs)
        if self.task != "A2T":
            raise ValueError("Audio Flamingo captioner supports only task A2T")
        self.model_path = self.model_path or "/scratch/l/lovenya/salm_migration_clean/models/cascaded/captioning/audio-flamingo-next-captioner-hf"
        self.processor = None
        self.model = None

    def load(self) -> None:
        import torch

        _patch_audio_flamingo_next_transformers_compat()
        from transformers import AutoConfig

        model_cls, processor_cls, method_name = _audio_flamingo_next_classes()
        config = AutoConfig.from_pretrained(self.model_path, local_files_only=True)
        self.processor = processor_cls.from_pretrained(self.model_path, local_files_only=True)
        self.model = model_cls.from_pretrained(
            self.model_path,
            config=config,
            local_files_only=True,
            torch_dtype=torch.bfloat16 if self.device.startswith("cuda") else torch.float32,
            device_map="auto" if self.device.startswith("cuda") else None,
        )
        self.method_name = method_name

    def generate(self, row: PromptRow, output_path: Path) -> GenerationResult:
        if self.processor is None or self.model is None:
            raise RuntimeError("Audio Flamingo captioner is not loaded")
        import torch

        audio_path = self._record_path(row, "audio_file")
        prompt = self._format_prompt(row.prompt, AFN_SHORT_CAPTION_PROMPT)
        conversation = [
            [
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": prompt},
                        {"type": "audio", "path": str(audio_path)},
                    ],
                }
            ]
        ]
        inputs = self.processor.apply_chat_template(
            conversation,
            tokenize=True,
            add_generation_prompt=True,
            return_dict=True,
        ).to(self.model.device)
        if "input_features" in inputs:
            inputs["input_features"] = inputs["input_features"].to(self.model.dtype)
        with torch.inference_mode():
            output_ids = self.model.generate(**inputs, max_new_tokens=256, repetition_penalty=1.2, do_sample=False)
        prompt_len = inputs["input_ids"].shape[1]
        caption = self.processor.batch_decode(
            output_ids[:, prompt_len:],
            skip_special_tokens=True,
            clean_up_tokenization_spaces=False,
        )[0].strip()
        extra = {
            "source_input_file": str(audio_path),
            "understanding_prompt": prompt,
            "messages": conversation[0],
            "understanding_text": caption,
            "backend": "audio_flamingo_next",
            "stages": [
                {
                    "stage": "audio_captioning",
                    "method": getattr(self, "method_name", "Audio Flamingo generate"),
                    "input_field": "audio_file",
                    "input_file": str(audio_path),
                    "prompt": prompt,
                    "output_text": caption,
                    "output_file": str(output_path),
                }
            ],
        }
        _write_text_result(output_path, row, caption, extra)
        return GenerationResult(output_path=output_path, sample_rate=None, extra=extra)


def _audio_flamingo_next_classes():
    import transformers

    if hasattr(transformers, "MusicFlamingoForConditionalGeneration") and hasattr(transformers, "MusicFlamingoProcessor"):
        return (
            transformers.MusicFlamingoForConditionalGeneration,
            transformers.MusicFlamingoProcessor,
            "MusicFlamingoForConditionalGeneration.generate",
        )
    return (
        transformers.AudioFlamingo3ForConditionalGeneration,
        transformers.AudioFlamingo3Processor,
        "AudioFlamingo3ForConditionalGeneration.generate",
    )


def _patch_audio_flamingo_next_transformers_compat() -> None:
    try:
        import transformers
        from transformers import (
            AudioFlamingo3Config,
            AudioFlamingo3ForConditionalGeneration,
            AudioFlamingo3Processor,
            AutoConfig,
            AutoProcessor,
        )
    except (AttributeError, ImportError):
        return
    if hasattr(transformers, "MusicFlamingoConfig") and hasattr(transformers, "MusicFlamingoProcessor"):
        return

    class MusicFlamingoConfig(AudioFlamingo3Config):
        model_type = "musicflamingo"

    MusicFlamingoConfig.__name__ = "MusicFlamingoConfig"
    MusicFlamingoConfig.__qualname__ = "MusicFlamingoConfig"

    _register_auto(AutoConfig, "musicflamingo", MusicFlamingoConfig)
    _register_auto(AutoProcessor, MusicFlamingoConfig, AudioFlamingo3Processor)

    transformers.MusicFlamingoConfig = MusicFlamingoConfig
    transformers.MusicFlamingoForConditionalGeneration = AudioFlamingo3ForConditionalGeneration
    transformers.MusicFlamingoProcessor = AudioFlamingo3Processor


def _register_auto(auto_cls, *args: object) -> None:
    register = getattr(auto_cls, "register", None)
    if register is None:
        return
    try:
        register(*args, exist_ok=True)
    except TypeError:
        try:
            register(*args)
        except ValueError:
            pass
    except ValueError:
        pass


def _write_text_result(output_path: Path, row: PromptRow, text: str, extra: dict) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(
        json.dumps(
            {
                "row_index": row.row_index,
                "sample_id": row.sample_id,
                "task": "A2T",
                "text": text,
                "extra": extra,
            },
            ensure_ascii=False,
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )
