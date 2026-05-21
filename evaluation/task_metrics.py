from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from evaluation.errors import EvaluationError
from evaluation.io import read_json
from evaluation.metrics.registry import known_metric_names


@dataclass(frozen=True)
class MetricSelection:
    task: str
    stage: str
    metric_names: list[str]
    output_type: str | None


def load_task_metric_map(path: Path) -> dict:
    data = read_json(path)
    if not isinstance(data, dict):
        raise EvaluationError(f"Task metric map must be a JSON object: {path}")
    return data


def select_metrics(
    task: str,
    stage: str | None,
    metrics_arg: str | None,
    task_metric_map: dict,
) -> MetricSelection:
    task_config = _resolve_task_config(task, task_metric_map)
    resolved_stage = stage or task_config.get("default_stage")
    if not isinstance(resolved_stage, str) or not resolved_stage:
        raise EvaluationError(f"Task {task!r} has no default stage. Pass --stage explicitly.")
    _validate_stage(task, task_config, resolved_stage)

    allowed = _allowed_metrics_for_stage(task_config, resolved_stage)
    metric_names = _parse_metric_arg(metrics_arg) if metrics_arg else list(_default_metrics_for_stage(task_config, resolved_stage))
    _validate_known_metrics(metric_names)
    invalid = [name for name in metric_names if name not in allowed]
    if invalid:
        raise EvaluationError(_invalid_metric_message(task, resolved_stage, invalid, task_metric_map))
    if not metric_names:
        raise EvaluationError(f"No metrics selected for task {task!r} stage {resolved_stage!r}.")
    output_type = task_config.get("output_type")
    return MetricSelection(
        task=task,
        stage=resolved_stage,
        metric_names=metric_names,
        output_type=str(output_type) if output_type is not None else None,
    )


def _validate_stage(task: str, task_config: dict, stage: str) -> None:
    default_stage = task_config.get("default_stage")
    overrides = task_config.get("stage_metric_overrides")
    valid_stages = {default_stage} if isinstance(default_stage, str) else set()
    if isinstance(overrides, dict):
        valid_stages.update(key for key in overrides if isinstance(key, str))
    if stage not in valid_stages:
        known = ", ".join(sorted(valid_stages)) or "none"
        raise EvaluationError(f"Stage {stage!r} is not defined for task {task!r}. Known stages: {known}")


def _resolve_task_config(task: str, task_metric_map: dict) -> dict:
    config = task_metric_map.get(task)
    if not isinstance(config, dict):
        raise EvaluationError(f"No metric mapping is defined for task {task!r}.")
    alias = config.get("alias_of")
    if alias:
        aliased = task_metric_map.get(alias)
        if not isinstance(aliased, dict):
            raise EvaluationError(f"Task {task!r} aliases unknown metric task {alias!r}.")
        merged = dict(aliased)
        merged.update({key: value for key, value in config.items() if key != "alias_of"})
        return merged
    return config


def _allowed_metrics_for_stage(task_config: dict, stage: str) -> list[str]:
    overrides = task_config.get("stage_metric_overrides")
    if isinstance(overrides, dict) and stage in overrides:
        metrics = overrides[stage]
    else:
        metrics = task_config.get("allowed_metrics", [])
    if not isinstance(metrics, list) or not all(isinstance(item, str) for item in metrics):
        raise EvaluationError(f"Invalid allowed metric list for stage {stage!r}.")
    return metrics


def _default_metrics_for_stage(task_config: dict, stage: str) -> list[str]:
    overrides = task_config.get("stage_metric_overrides")
    if isinstance(overrides, dict) and stage in overrides:
        default_stage = task_config.get("default_stage")
        if stage != default_stage:
            return list(overrides[stage])
    metrics = task_config.get("default_metrics", [])
    if not isinstance(metrics, list) or not all(isinstance(item, str) for item in metrics):
        raise EvaluationError(f"Invalid default metric list for stage {stage!r}.")
    return metrics


def _parse_metric_arg(value: str) -> list[str]:
    return [item.strip() for item in value.split(",") if item.strip()]


def _validate_known_metrics(metric_names: list[str]) -> None:
    unknown = sorted(set(metric_names) - known_metric_names())
    if unknown:
        known = ", ".join(sorted(known_metric_names()))
        raise EvaluationError(f"Unknown metrics {unknown}. Known metrics: {known}")


def _invalid_metric_message(task: str, stage: str, invalid: list[str], task_metric_map: dict) -> str:
    invalid_text = ", ".join(f"`{name}`" for name in invalid)
    message = f"Metric {invalid_text} is not valid for task `{task}` stage `{stage}`."
    if task == "S2A" and "wer" in invalid:
        message += (
            " WER is valid for S2T transcription or S2A intermediate stage "
            "`s2t_intermediate`, not the final S2A audio output."
        )
    if task == "A2S" and any(name in {"mace", "spider", "clap_similarity"} for name in invalid):
        message += (
            " A2T caption metrics are valid only for A2S intermediate stage "
            "`a2t_intermediate`, not the final A2S speech output."
        )
    return message
