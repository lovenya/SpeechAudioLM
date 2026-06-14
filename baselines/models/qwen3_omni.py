from __future__ import annotations

from pathlib import Path

from baselines.dataset import PromptRow
from baselines.models.base import BaselineRunner, GenerationResult
from project_paths import model_path

QWEN_T2S_TEMPLATE = "Read the following text aloud exactly, without adding any extra words: {prompt}"
QWEN_PSEUDO_T2A_DIRECT_TEMPLATE = (
    "You are given a description of an audio scene. Generate the audio based on this description. "
    "Do not explain the scene. Produce the audio itself.\n\n"
    "Audio description: {prompt}"
)
QWEN_PSEUDO_T2A_PHONEME_TEMPLATE = (
    "You are given a description of an audio scene. Try to generate the audio itself, not an explanation. "
    "Use only vocal sound effects, phonemes, rhythm, breaths, and onomatopoeia if needed. The scene may "
    "include environmental sounds, animal sounds, background voices, impacts, weather, or other sound events.\n\n"
    "Audio description: {prompt}"
)


class Qwen3OmniRunner(BaselineRunner):
    def __init__(
        self,
        qwen_speaker: str = "Ethan",
        qwen_template: str | None = None,
        **kwargs: object,
    ) -> None:
        super().__init__(**kwargs)
        if self.task not in {"T2S", "A2S", "pseudo_t2a_direct", "pseudo_t2a_phoneme"}:
            raise ValueError("Qwen3-Omni supports --task T2S, A2S, pseudo_t2a_direct, or pseudo_t2a_phoneme")
        self.model_path = self.model_path or str(model_path("qwen3_omni"))
        self.speaker = qwen_speaker
        self.template = self.prompt_template or qwen_template or self._default_template()
        self.model = None
        self.processor = None

    def load(self) -> None:
        import torch
        from transformers import Qwen3OmniMoeForConditionalGeneration, Qwen3OmniMoeProcessor

        torch.manual_seed(self.seed)
        self.processor = Qwen3OmniMoeProcessor.from_pretrained(
            self.model_path,
            local_files_only=True,
            use_fast=False,
        )
        self.model = Qwen3OmniMoeForConditionalGeneration.from_pretrained(
            self.model_path,
            local_files_only=True,
            dtype=torch.bfloat16,
            device_map="auto",
        )

    def generate(self, row: PromptRow, output_path: Path) -> GenerationResult:
        if self.model is None or self.processor is None:
            raise RuntimeError("Qwen3-Omni model is not loaded")

        import soundfile as sf
        import torch

        prompt = self._prompt_for_task(row)
        instruction = self.template.format(prompt=prompt)
        conversation = [{"role": "user", "content": instruction}]
        text = self.processor.apply_chat_template(conversation, add_generation_prompt=True, tokenize=False)
        inputs = self.processor(text=text, return_tensors="pt", padding=True)
        inputs = inputs.to(self.model.device)

        text_ids, audio = self.model.generate(
            **inputs,
            return_audio=True,
            speaker=self.speaker,
            thinker_return_dict_in_generate=True,
            thinker_max_new_tokens=64,
            talker_max_new_tokens=768,
        )
        sequences = getattr(text_ids, "sequences", text_ids)
        decoded = self.processor.batch_decode(
            sequences[:, inputs["input_ids"].shape[1] :],
            skip_special_tokens=True,
            clean_up_tokenization_spaces=False,
        )
        if audio is None:
            raise RuntimeError("Qwen3-Omni returned no audio")

        sf.write(str(output_path), audio.reshape(-1).detach().cpu().float().numpy(), samplerate=24000)
        extra = {
            "shape": list(audio.shape),
            "decoded_text": decoded,
            "generation_prompt": instruction,
            "cuda_reserved_gb": round(torch.cuda.memory_reserved() / 1024**3, 2)
            if torch.cuda.is_available()
            else None,
        }
        if self.task == "A2S":
            extra.update(
                {
                    "understanding_stage": "skipped",
                    "understanding_text": None,
                    "converted_prompt": prompt,
                    "intermediate_transcript": prompt,
                    "final_generation_method": "Qwen3-Omni TTS",
                    "stages": [
                        {
                            "stage": "audio_understanding",
                            "method": "skipped",
                            "reason": "Qwen3-Omni A2S baseline uses the paired transcript as the intermediate text.",
                        },
                        {
                            "stage": "text_to_speech",
                            "method": "Qwen3-Omni talker",
                            "prompt": prompt,
                            "output_file": str(output_path),
                        },
                    ],
                }
            )
            try:
                extra["source_input_file"] = str(self._record_path(row, "audio_file"))
            except ValueError:
                extra["source_input_file"] = None
        return GenerationResult(
            output_path=output_path,
            sample_rate=24000,
            extra=extra,
        )

    def _default_template(self) -> str:
        if self.task in {"T2S", "A2S"}:
            return QWEN_T2S_TEMPLATE
        if self.task == "pseudo_t2a_direct":
            return QWEN_PSEUDO_T2A_DIRECT_TEMPLATE
        if self.task == "pseudo_t2a_phoneme":
            return QWEN_PSEUDO_T2A_PHONEME_TEMPLATE
        raise ValueError(f"Unsupported Qwen3-Omni task: {self.task}")

    def _prompt_for_task(self, row: PromptRow) -> str:
        if self.task == "A2S":
            speech_text = row.record.get("speech_text")
            if not isinstance(speech_text, str) or not speech_text.strip():
                raise ValueError(f"Row {row.row_index} has no usable speech_text for Qwen3-Omni A2S")
            return speech_text
        return row.prompt
