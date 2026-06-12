from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from evaluation.errors import EvaluationError
from evaluation.io import read_json
from evaluation.metrics.registry import known_metric_names


@dataclass(frozen=True)
class MetricSelection:
    task: str
    metric_names: list[str]
    output_type: str | None


def load_task_metric_map(path: Path) -> dict:
    data = read_json(path)
    if not isinstance(data, dict):
        raise EvaluationError(f"Task metric map must be a JSON object: {path}")
    return data


def select_metrics(
    task: str,
    metrics_arg: str | None,
    task_metric_map: dict,
) -> MetricSelection:
    task_config = _resolve_task_config(task, task_metric_map)
    allowed = _metric_list(task_config, "allowed_metrics", task)
    metric_names = _parse_metric_arg(metrics_arg) if metrics_arg else list(_metric_list(task_config, "default_metrics", task))
    _validate_known_metrics(metric_names)
    invalid = [name for name in metric_names if name not in allowed]
    if invalid:
        raise EvaluationError(_invalid_metric_message(task, invalid))
    if not metric_names:
        raise EvaluationError(f"No metrics selected for task {task!r}.")
    output_type = task_config.get("output_type")
    return MetricSelection(
        task=task,
        metric_names=metric_names,
        output_type=str(output_type) if output_type is not None else None,
    )


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


def _metric_list(task_config: dict, key: str, task: str) -> list[str]:
    metrics = task_config.get(key, [])
    if not isinstance(metrics, list) or not all(isinstance(item, str) for item in metrics):
        raise EvaluationError(f"Invalid {key} list for task {task!r}.")
    return metrics


def _parse_metric_arg(value: str) -> list[str]:
    return [item.strip() for item in value.split(",") if item.strip()]


def _validate_known_metrics(metric_names: list[str]) -> None:
    unknown = sorted(set(metric_names) - known_metric_names())
    if unknown:
        known = ", ".join(sorted(known_metric_names()))
        raise EvaluationError(f"Unknown metrics {unknown}. Known metrics: {known}")


def _invalid_metric_message(task: str, invalid: list[str]) -> str:
    invalid_text = ", ".join(f"`{name}`" for name in invalid)
    message = f"Metric {invalid_text} is not valid for task `{task}`."
    if task == "S2A" and "wer" in invalid:
        message += (
            " WER is valid for S2T transcription. Evaluate the S2T component manifest directly, "
            "not the combined S2A final-audio manifest."
        )
    if task == "A2S" and any(name in {"mace", "spider", "aac_clapsim"} for name in invalid):
        message += (
            " A2T caption metrics are valid for A2T captioning. Evaluate the A2T component "
            "manifest directly, not the combined A2S final-speech manifest."
        )
    if task in {"S2A", "A2S"} and "nli_contradiction" in invalid:
        message += " NLI contradiction is a T2T/intermediate-text metric; evaluate the text-to-text stage manifest directly."
    if task in {"T2A", "S2A"} and any(name in {"mace", "spider", "aac_clapsim"} for name in invalid):
        message += " AAC caption metrics are valid for A2T captioning, not final generated-audio evaluation."
    if task == "A2T" and any(
        name in {
            "target_caption_clap_score",
            "prompt_adherence_clap_score",
            "kl_passt",
            "fad",
            "kad",
            "inception_score",
        }
        for name in invalid
    ):
        message += " Audio-generation metrics are valid for T2A/S2A final audio, not A2T captioning."
    return message
