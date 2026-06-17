from __future__ import annotations

from pathlib import Path

from project_paths import result_path, root_path


DEFAULT_BASELINE_ROOT = result_path("baselines")
DEFAULT_EVAL_ROOT = root_path("results_root") / "evals"


def baseline_output_root(
    model: str,
    task: str,
    dataset: str,
    split: str,
    root: Path = DEFAULT_BASELINE_ROOT,
) -> Path:
    return root / model / task / dataset / split


def legacy_baseline_output_roots(
    model: str,
    task: str,
    split: str,
    prompt_field: str | None,
    root: Path = DEFAULT_BASELINE_ROOT,
) -> list[Path]:
    candidates = [root / model / task / split]
    if prompt_field:
        candidates.append(root / model / task / split / prompt_field)
        candidates.append(root / model / split / prompt_field)
    return candidates


def eval_stage_root(
    direction: str,
    stage: str,
    model: str,
    dataset: str,
    split: str,
    evaluated_run_id: str,
    metric_id: str,
    eval_run_id: str,
    root: Path = DEFAULT_EVAL_ROOT,
) -> Path:
    return root / "stages" / direction / stage / model / dataset / split / evaluated_run_id / metric_id / eval_run_id


def default_prompt_field(task: str) -> str:
    if task in {"T2A", "A2T", "A2S", "pseudo_t2a_direct", "pseudo_t2a_phoneme"}:
        return "audio_cap"
    if task in {"T2S", "S2T", "S2A"}:
        return "speech_text"
    raise ValueError(f"Unsupported task: {task}")
