from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path


@dataclass
class GenerationResult:
    output_path: Path
    sample_rate: int
    extra: dict


class BaselineRunner:
    extension = ".wav"

    def __init__(
        self,
        model_path: str | None,
        device: str,
        duration: float,
        steps: int,
        seed: int,
        **_: object,
    ) -> None:
        self.model_path = model_path
        self.device = device
        self.duration = duration
        self.steps = steps
        self.seed = seed

    def load(self) -> None:
        raise NotImplementedError

    def generate(self, prompt: str, output_path: Path) -> GenerationResult:
        raise NotImplementedError
