from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from baselines.dataset import PromptRow


@dataclass
class GenerationResult:
    output_path: Path
    sample_rate: int | None
    extra: dict


class BaselineRunner:
    extension = ".wav"
    output_subdir = "audio"

    def __init__(
        self,
        model_path: str | None,
        device: str,
        duration: float,
        steps: int,
        seed: int,
        task: str,
        prompt_template: str | None = None,
        dataset_root: str | None = None,
        **_: object,
    ) -> None:
        self.model_path = model_path
        self.device = device
        self.duration = duration
        self.steps = steps
        self.seed = seed
        self.task = task
        self.prompt_template = prompt_template
        self.dataset_root = Path(dataset_root) if dataset_root else None

    def load(self) -> None:
        raise NotImplementedError

    def generate(self, row: PromptRow, output_path: Path) -> GenerationResult:
        raise NotImplementedError

    def _format_prompt(self, prompt: str, default_template: str | None = None) -> str:
        template = self.prompt_template or default_template
        if not template:
            return prompt
        return template.format(prompt=prompt)

    def _record_path(self, row: PromptRow, field: str) -> Path:
        value = row.record.get(field)
        if not isinstance(value, str) or not value.strip():
            raise ValueError(f"Row {row.row_index} has no usable {field!r}")
        path = Path(value)
        if path.is_absolute():
            return path
        if self.dataset_root is None:
            raise ValueError(f"Cannot resolve relative {field!r} without dataset_root")
        return self.dataset_root / path
