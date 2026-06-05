from __future__ import annotations

import json
from pathlib import Path

from baselines.dataset import PromptRow
from baselines.models.base import BaselineRunner, GenerationResult


def _strip_thinking_output(text: str) -> str:
    text = text.strip()
    if "</think>" in text:
        text = text.split("</think>", 1)[1].strip()
    if text.startswith("<think>"):
        lines = [line.strip() for line in text.splitlines() if line.strip() and line.strip() != "<think>"]
        text = lines[-1] if lines else text
    return text.strip()


def _patch_transformers_optional_scipy_import() -> None:
    """Avoid importing broken optional scipy stacks for CausalLM text generation."""
    import importlib.machinery
    import sys
    import types

    if "scipy.optimize" in sys.modules:
        return

    scipy = sys.modules.get("scipy")
    if scipy is None:
        scipy = types.ModuleType("scipy")
        scipy.__path__ = []
        scipy.__spec__ = importlib.machinery.ModuleSpec("scipy", loader=None, is_package=True)
        sys.modules["scipy"] = scipy
    elif getattr(scipy, "__spec__", None) is None:
        scipy.__spec__ = importlib.machinery.ModuleSpec("scipy", loader=None, is_package=True)

    optimize = types.ModuleType("scipy.optimize")
    optimize.__spec__ = importlib.machinery.ModuleSpec("scipy.optimize", loader=None)

    def linear_sum_assignment(*args, **kwargs):
        raise RuntimeError("scipy.optimize.linear_sum_assignment is not available in this text LLM environment")

    optimize.linear_sum_assignment = linear_sum_assignment
    scipy.optimize = optimize
    sys.modules["scipy.optimize"] = optimize


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
        _patch_transformers_optional_scipy_import()

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

        formatted_prompt = self._format_prompt(row.prompt)
        messages = [{"role": "user", "content": formatted_prompt}]
        if hasattr(self.tokenizer, "apply_chat_template"):
            try:
                text = self.tokenizer.apply_chat_template(
                    messages,
                    tokenize=False,
                    add_generation_prompt=True,
                    enable_thinking=False,
                )
            except TypeError:
                text = self.tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        else:
            text = formatted_prompt
        inputs = self.tokenizer(text, return_tensors="pt").to(self.model.device)
        with torch.inference_mode():
            output_ids = self.model.generate(
                **inputs,
                max_new_tokens=min(max(self.steps, 32), 512),
                do_sample=False,
                pad_token_id=self.tokenizer.eos_token_id,
            )
        generated = output_ids[0, inputs["input_ids"].shape[1] :]
        answer = _strip_thinking_output(self.tokenizer.decode(generated, skip_special_tokens=True).strip())
        extra = {
            "generation_prompt": formatted_prompt,
            "input_text": row.prompt,
            "understanding_text": answer,
            "backend": "text_llm",
            "stages": [
                {
                    "stage": "text_to_text",
                    "method": "transformers.AutoModelForCausalLM.generate",
                    "input_text": row.prompt,
                    "generation_prompt": formatted_prompt,
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
