from __future__ import annotations

import os
from contextlib import contextmanager
from pathlib import Path

from baselines.models.base import BaselineRunner, GenerationResult


class AudioOmniRunner(BaselineRunner):
    def __init__(
        self,
        audio_omni_config: str = "/project/rrg-csubakan/lovenya/models/audio-omni/Audio-Omni.json",
        audio_omni_ckpt: str = "/project/rrg-csubakan/lovenya/models/audio-omni/model.ckpt",
        audio_omni_source_dir: str = "/project/rrg-csubakan/lovenya/src/Audio-Omni",
        **kwargs: object,
    ) -> None:
        super().__init__(**kwargs)
        self.config_path = audio_omni_config
        self.ckpt_path = audio_omni_ckpt
        self.source_dir = Path(audio_omni_source_dir)
        self.model = None

    def load(self) -> None:
        import sys
        import torch

        os.environ.setdefault("QWEN_OMNI_MODEL_PATH", "/project/rrg-csubakan/lovenya/models/qwen2.5-omni-3b")
        os.environ.setdefault("SKIP_QWEN_WEIGHTS", "1")
        source = str(self.source_dir)
        if source not in sys.path:
            sys.path.insert(0, source)
        from audio_omni import AudioOmni

        torch.manual_seed(self.seed)
        with _pushd(self.source_dir):
            self.model = AudioOmni(self.config_path, self.ckpt_path, device=self.device)

    def generate(self, prompt: str, output_path: Path) -> GenerationResult:
        if self.model is None:
            raise RuntimeError("Audio-Omni model is not loaded")

        import torch
        import torchaudio

        # The released config has fixed-shape mel conditioning that expects 10 seconds.
        duration = int(round(self.duration))
        if duration != 10:
            raise ValueError("Audio-Omni baseline currently requires --duration 10")

        with _pushd(self.source_dir):
            audio = self.model.generate(
                "T2A",
                prompt=prompt,
                steps=self.steps,
                seconds_total=duration,
                seed=self.seed,
            )
        audio = audio.detach().cpu()
        if audio.ndim == 1:
            audio = audio.unsqueeze(0)
        sample_rate = int(getattr(self.model, "sample_rate", 44100))
        torchaudio.save(str(output_path), audio, sample_rate)
        return GenerationResult(
            output_path=output_path,
            sample_rate=sample_rate,
            extra={
                "shape": list(audio.shape),
                "cuda_reserved_gb": round(torch.cuda.memory_reserved() / 1024**3, 2)
                if torch.cuda.is_available()
                else None,
            },
        )


@contextmanager
def _pushd(path: Path):
    previous = Path.cwd()
    os.chdir(path)
    try:
        yield
    finally:
        os.chdir(previous)
