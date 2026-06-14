from __future__ import annotations

import json
import os
from functools import lru_cache
from pathlib import Path
from typing import Any


REPO_ROOT = Path(__file__).resolve().parent
DEFAULT_CONFIG_PATH = REPO_ROOT / "configs" / "project_paths.json"

_ROOT_ENV_VARS = {
    "repo_root": "SALM_REPO_ROOT",
    "results_root": "SALM_RESULTS_ROOT",
    "environments_root": "SALM_ENVIRONMENTS_ROOT",
    "models_root": "SALM_MODELS_ROOT",
    "cache_root": "SALM_CACHE_ROOT",
    "datasets_root": "SALM_DATASETS_ROOT",
    "sources_root": "SALM_SOURCES_ROOT",
    "logs_root": "SALM_LOGS_ROOT",
}


@lru_cache(maxsize=None)
def load_project_paths(config_path: str | Path = DEFAULT_CONFIG_PATH) -> dict[str, Any]:
    path = Path(config_path)
    with path.open(encoding="utf-8") as handle:
        config = json.load(handle)
    if config.get("schema_version") != 2:
        raise ValueError(f"Unsupported project path schema: {config.get('schema_version')!r}")
    return config


def root_path(name: str, config_path: str | Path = DEFAULT_CONFIG_PATH) -> Path:
    config = load_project_paths(config_path)
    roots = config["roots"]
    if name not in roots:
        raise KeyError(f"Unknown project root: {name}")
    override = os.environ.get(_ROOT_ENV_VARS.get(name, ""))
    value = override if override else roots[name]
    if name == "repo_root":
        return _absolute(Path(value), REPO_ROOT)
    repo_root = root_path("repo_root", config_path)
    return _absolute(Path(resolve_config_value(value, config_path)), repo_root)


def model_path(name: str, config_path: str | Path = DEFAULT_CONFIG_PATH) -> Path:
    return _mapped_path("models", name, "models_root", config_path)


def environment_path(name: str, config_path: str | Path = DEFAULT_CONFIG_PATH) -> Path:
    return _mapped_path("environments", name, "environments_root", config_path)


def cache_path(name: str, config_path: str | Path = DEFAULT_CONFIG_PATH) -> Path:
    return _mapped_path("caches", name, "cache_root", config_path)


def dataset_path(name: str, config_path: str | Path = DEFAULT_CONFIG_PATH) -> Path:
    return _mapped_path("datasets", name, "datasets_root", config_path)


def source_path(name: str, config_path: str | Path = DEFAULT_CONFIG_PATH) -> Path:
    return _mapped_path("sources", name, "sources_root", config_path)


def result_path(name: str, config_path: str | Path = DEFAULT_CONFIG_PATH) -> Path:
    return _mapped_path("results", name, "results_root", config_path)


def log_path(name: str, config_path: str | Path = DEFAULT_CONFIG_PATH) -> Path:
    return _mapped_path("logs", name, "logs_root", config_path)


def resolve_config_value(value: Any, config_path: str | Path = DEFAULT_CONFIG_PATH) -> Any:
    if isinstance(value, dict):
        return {key: resolve_config_value(item, config_path) for key, item in value.items()}
    if isinstance(value, list):
        return [resolve_config_value(item, config_path) for item in value]
    if not isinstance(value, str):
        return value

    roots = load_project_paths(config_path)["roots"]
    resolved = value
    for name in roots:
        token = "{" + name + "}"
        if token in resolved:
            resolved = resolved.replace(token, str(root_path(name, config_path)))
    return resolved


def _mapped_path(
    section: str,
    name: str,
    root_name: str,
    config_path: str | Path,
) -> Path:
    mappings = load_project_paths(config_path).get(section, {})
    if name not in mappings:
        raise KeyError(f"Unknown {section} path: {name}")
    value = resolve_config_value(mappings[name], config_path)
    return _absolute(Path(value), root_path(root_name, config_path))


def _absolute(path: Path, base: Path) -> Path:
    return path if path.is_absolute() else (base / path).resolve()
