from __future__ import annotations

from baselines.models.base import BaselineRunner


def get_runner(name: str, **kwargs) -> BaselineRunner:
    if name == "audiogen":
        from baselines.models.audiogen import AudioGenRunner

        return AudioGenRunner(**kwargs)
    if name == "tangoflux":
        from baselines.models.tangoflux import TangoFluxRunner

        return TangoFluxRunner(**kwargs)
    if name == "qwen3_omni":
        from baselines.models.qwen3_omni import Qwen3OmniRunner

        return Qwen3OmniRunner(**kwargs)
    raise ValueError(f"Unknown baseline model: {name}")
