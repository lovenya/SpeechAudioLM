from __future__ import annotations

import json
from pathlib import Path

from baselines.dataset import PromptRow
from baselines.models.base import BaselineRunner, GenerationResult


class TextLLMRunner(BaselineRunner):
    output_subdir = "text"
    extension = ".json"

    def __init__(self, **kwargs: object) -> None:
        super().__init__(**kwargs)
        if self.task != "text_to_text":
            raise ValueError("Text LLM runner supports only task text_to_text")
        self.tokenizer = None
        self.model = None

    def load(self) -> None:
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer

        if not self.model_path:
            raise ValueError("Text LLM runner requires a model_path")
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_path, local_files_only=True, trust_remote_code=True)
        self.model = AutoModelForCausalLM.from_pretrained(
            self.model_path,
            local_files_only=True,
            trust_remote_code=True,
            torch_dtype=torch.bfloat16 if self.device.startswith("cuda") else torch.float32,
            device_map="auto" if self.device.startswith("cuda") else None,
        )

    def generate(self, row: PromptRow, output_path: Path) -> GenerationResult:
        if self.model is None or self.tokenizer is None:
            raise RuntimeError("Text LLM is not loaded")
        import torch

        messages = [{"role": "user", "content": row.prompt}]
        if hasattr(self.tokenizer, "apply_chat_template"):
            text = self.tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        else:
            text = row.prompt
        inputs = self.tokenizer(text, return_tensors="pt").to(self.model.device)
        with torch.inference_mode():
            output_ids = self.model.generate(
                **inputs,
                max_new_tokens=min(max(self.steps, 32), 512),
                do_sample=False,
                pad_token_id=self.tokenizer.eos_token_id,
            )
        generated = output_ids[0, inputs["input_ids"].shape[1] :]
        answer = self.tokenizer.decode(generated, skip_special_tokens=True).strip()
        extra = {
            "generation_prompt": row.prompt,
            "understanding_text": answer,
            "backend": "text_llm",
            "stages": [
                {
                    "stage": "text_to_text",
                    "method": "transformers.AutoModelForCausalLM.generate",
                    "input_text": row.prompt,
                    "output_text": answer,
                    "output_file": str(output_path),
                }
            ],
        }
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(
            json.dumps(
                {
                    "row_index": row.row_index,
                    "sample_id": row.sample_id,
                    "task": "text_to_text",
                    "text": answer,
                    "extra": extra,
                },
                ensure_ascii=False,
                indent=2,
            )
            + "\n",
            encoding="utf-8",
        )
        return GenerationResult(output_path=output_path, sample_rate=None, extra=extra)
