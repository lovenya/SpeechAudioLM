from __future__ import annotations

import importlib.machinery
import sys
import types
from pathlib import Path

from baselines.dataset import PromptRow
from baselines.models.base import BaselineRunner, GenerationResult


class TangoFluxRunner(BaselineRunner):
    def __init__(
        self,
        tangoflux_source_dir: str = "/scratch/l/lovenya/salm_migration_clean/models/tangoflux",
        **kwargs: object,
    ) -> None:
        super().__init__(**kwargs)
        if self.task != "T2A":
            raise ValueError("TangoFlux supports only --task T2A")
        self.model_path = self.model_path or "/scratch/l/lovenya/salm_migration_clean/models/tangoflux"
        self.source_dir = tangoflux_source_dir
        self.model = None

    def load(self) -> None:
        import torch

        _install_datasets_stub()
        if self.source_dir not in sys.path:
            sys.path.insert(0, self.source_dir)
        import tangoflux as tangoflux_module
        from tangoflux import TangoFluxInference

        torch.manual_seed(self.seed)
        tangoflux_module.snapshot_download = lambda repo_id: self.model_path
        self.model = TangoFluxInference(name="declare-lab/TangoFlux", device=self.device)

    def generate(self, row: PromptRow, output_path: Path) -> GenerationResult:
        if self.model is None:
            raise RuntimeError("TangoFlux model is not loaded")

        import torchaudio

        prompt = self._format_prompt(row.prompt)
        audio = self.model.generate(prompt, steps=self.steps, duration=self.duration)
        audio = audio.detach().cpu()
        torchaudio.save(str(output_path), audio, 44100)
        return GenerationResult(
            output_path=output_path,
            sample_rate=44100,
            extra={"shape": list(audio.shape), "generation_prompt": prompt},
        )


def _install_datasets_stub() -> None:
    if "datasets" in sys.modules:
        return
    datasets = types.ModuleType("datasets")
    datasets.__spec__ = importlib.machinery.ModuleSpec("datasets", loader=None)
    datasets.load_dataset = lambda *args, **kwargs: (_ for _ in ()).throw(
        RuntimeError("datasets stub called during TangoFlux inference")
    )
    datasets.Audio = object
    sys.modules["datasets"] = datasets
