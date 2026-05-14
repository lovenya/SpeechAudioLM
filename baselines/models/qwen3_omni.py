from __future__ import annotations

from pathlib import Path

from baselines.models.base import BaselineRunner, GenerationResult


class Qwen3OmniRunner(BaselineRunner):
    def __init__(
        self,
        qwen_speaker: str = "Ethan",
        qwen_template: str = "Read the following text aloud exactly, without adding any extra words: {prompt}",
        **kwargs: object,
    ) -> None:
        super().__init__(**kwargs)
        self.model_path = self.model_path or "/project/rrg-csubakan/lovenya/models/qwen3-omni-30b-a3b-instruct"
        self.speaker = qwen_speaker
        self.template = qwen_template
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

    def generate(self, prompt: str, output_path: Path) -> GenerationResult:
        if self.model is None or self.processor is None:
            raise RuntimeError("Qwen3-Omni model is not loaded")

        import soundfile as sf
        import torch

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
        decoded = self.processor.batch_decode(
            text_ids[:, inputs["input_ids"].shape[1] :],
            skip_special_tokens=True,
            clean_up_tokenization_spaces=False,
        )
        if audio is None:
            raise RuntimeError("Qwen3-Omni returned no audio")

        sf.write(str(output_path), audio.reshape(-1).detach().cpu().float().numpy(), samplerate=24000)
        return GenerationResult(
            output_path=output_path,
            sample_rate=24000,
            extra={
                "shape": list(audio.shape),
                "decoded_text": decoded,
                "cuda_reserved_gb": round(torch.cuda.memory_reserved() / 1024**3, 2)
                if torch.cuda.is_available()
                else None,
            },
        )
