from __future__ import annotations

from pathlib import Path

from baselines.dataset import sample_id_from_record
from evaluation.errors import EvaluationError
from project_conventions import (
    baseline_output_root,
    default_prompt_field,
    legacy_baseline_output_roots,
)


def resolve_manifest_path(
    model: str,
    task: str,
    dataset: str,
    split: str,
    explicit_manifest: str | None,
    baseline_root: Path,
) -> Path:
    if explicit_manifest:
        path = Path(explicit_manifest)
        if not path.is_file():
            raise EvaluationError(f"Manifest does not exist: {path}")
        return path

    canonical = baseline_output_root(model, task, dataset, split, baseline_root) / "manifest.jsonl"
    candidates = [canonical]
    try:
        prompt_field = default_prompt_field(task)
    except ValueError:
        prompt_field = None
    candidates.extend(
        path / "manifest.jsonl"
        for path in legacy_baseline_output_roots(model, task, split, prompt_field, baseline_root)
    )

    for path in candidates:
        if path.is_file():
            return path

    candidate_text = "\n".join(f"  - {path}" for path in candidates)
    raise EvaluationError(
        "Could not find a baseline manifest. Checked:\n"
        f"{candidate_text}\n"
        "Pass --manifest to evaluate a non-standard output path."
    )


def source_record(row: dict) -> dict:
    source = row.get("source")
    return source if isinstance(source, dict) else row


def sample_id(row: dict, ordinal: int) -> str:
    existing = row.get("sample_id")
    if isinstance(existing, str) and existing:
        return existing
    return sample_id_from_record(source_record(row), ordinal)


def is_successful_generation(row: dict) -> bool:
    return row.get("status", "success") == "success"

