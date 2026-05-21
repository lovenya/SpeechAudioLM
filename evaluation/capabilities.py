from __future__ import annotations

from pathlib import Path

from evaluation.errors import EvaluationError
from evaluation.io import read_json


def load_capabilities(path: Path) -> dict[str, set[str]]:
    data = read_json(path)
    models = data.get("models")
    if not isinstance(models, dict):
        raise EvaluationError(f"Capability config has no 'models' object: {path}")
    capabilities: dict[str, set[str]] = {}
    for model, record in models.items():
        tasks = record.get("tasks") if isinstance(record, dict) else None
        if not isinstance(tasks, list) or not all(isinstance(item, str) for item in tasks):
            raise EvaluationError(f"Capability config for {model!r} must define a string task list")
        capabilities[str(model)] = set(tasks)
    return capabilities


def validate_model_task(model: str, task: str, capabilities: dict[str, set[str]]) -> None:
    if model not in capabilities:
        known = ", ".join(sorted(capabilities))
        raise EvaluationError(f"Unknown model {model!r}. Known models: {known}")
    if task not in capabilities[model]:
        known = ", ".join(sorted(capabilities[model]))
        raise EvaluationError(f"Model {model!r} does not support task {task!r}. Supported tasks: {known}")

