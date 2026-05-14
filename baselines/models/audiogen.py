from __future__ import annotations

from pathlib import Path

from baselines.dataset import PromptRow
from baselines.models.base import BaselineRunner, GenerationResult


class AudioGenRunner(BaselineRunner):
    def __init__(self, **kwargs: object) -> None:
        super().__init__(**kwargs)
        if self.task != "T2A":
            raise ValueError("AudioGen supports only --task T2A")
        self.model_path = self.model_path or "/project/rrg-csubakan/lovenya/models/audiogen-medium"
        self.model = None

    def load(self) -> None:
        import torch
        from audiocraft.models import AudioGen

        torch.manual_seed(self.seed)
        self.model = AudioGen.get_pretrained(self.model_path, device=self.device)
        self.model.set_generation_params(duration=self.duration)

    def generate(self, row: PromptRow, output_path: Path) -> GenerationResult:
        if self.model is None:
            raise RuntimeError("AudioGen model is not loaded")

        from audiocraft.data.audio import audio_write

        prompt = self._format_prompt(row.prompt)
        wav = self.model.generate([prompt], progress=True)
        prefix = output_path.with_suffix("")
        audio_write(str(prefix), wav[0].detach().cpu(), self.model.sample_rate, strategy="loudness")
        return GenerationResult(
            output_path=prefix.with_suffix(".wav"),
            sample_rate=int(self.model.sample_rate),
            extra={
                "shape": list(wav.shape),
                "channels": int(self.model.audio_channels),
                "generation_prompt": prompt,
            },
        )
