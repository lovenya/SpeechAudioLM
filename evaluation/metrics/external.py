from __future__ import annotations

import importlib.metadata
from pathlib import Path
from typing import Any

from evaluation.errors import MetricUnavailableError


PACKAGE_BY_BACKEND = {
    "aac-metrics": "aac-metrics",
    "audioldm_eval": "audioldm-eval",
    "kadtk": "kadtk",
    "openai-whisper": "openai-whisper",
    "stable-audio-metrics": "stable-audio-metrics",
    "utmosv2": "utmosv2",
}


def backend_metadata(name: str, config: dict[str, Any], device: str, cache_dir: Path) -> dict[str, Any]:
    backend = config.get("backend")
    package_name = PACKAGE_BY_BACKEND.get(str(backend), str(backend)) if backend else None
    version = None
    if package_name:
        try:
            version = importlib.metadata.version(package_name)
        except importlib.metadata.PackageNotFoundError:
            version = None
    return {
        "metric": name,
        "backend": backend,
        "python": config.get("python"),
        "package": package_name,
        "package_version": version,
        "git_commit": config.get("git_commit"),
        "model_name": config.get("model_name"),
        "model_path": config.get("model_path"),
        "checkpoint": config.get("checkpoint"),
        "config_name": config.get("config_name"),
        "device": device,
        "cache_dir": str(config.get("cache_dir") or cache_dir),
        "audio_preprocessing": {
            "sample_rate": config.get("sample_rate"),
            "resampling_policy": config.get("resampling_policy", "backend_default"),
            "mono_stereo_policy": config.get("mono_stereo_policy", "backend_default"),
            "loudness_normalization": config.get("loudness_normalization", "backend_default"),
            "duration_policy": config.get("duration_policy", "backend_default"),
        },
    }


def require_configured_backend(name: str, config: dict[str, Any]) -> None:
    backend = config.get("backend")
    if not backend:
        raise MetricUnavailableError(
            f"Metric {name!r} requires an external backend/checkpoint. "
            "Set it in --metrics-config before running this metric."
        )
    configured_python = config.get("python")
    if configured_python:
        if Path(configured_python).is_file():
            return
        raise MetricUnavailableError(
            f"Metric {name!r} is configured to use backend Python {configured_python!r}, "
            "but that interpreter does not exist."
        )
    package_name = PACKAGE_BY_BACKEND.get(str(backend))
    if package_name:
        try:
            importlib.metadata.version(package_name)
        except importlib.metadata.PackageNotFoundError as exc:
            raise MetricUnavailableError(
                f"Metric {name!r} uses backend {backend!r}, but package {package_name!r} is not installed."
            ) from exc
