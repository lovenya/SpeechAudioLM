from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Literal
from zoneinfo import ZoneInfo

from project_paths import root_path


TORONTO_TZ = ZoneInfo("America/Toronto")
CASCADED_STAGES = {"a2t", "s2t", "t2t", "t2a", "t2s"}
DIRECTIONS = {"a2s", "s2a"}
STAGE_ONE_BY_DIRECTION = {"a2s": "a2t", "s2a": "s2t"}
FINAL_STAGE_BY_DIRECTION = {"a2s": "t2s", "s2a": "t2a"}


@dataclass(frozen=True)
class RunPaths:
    result: Path
    logs: Path
    result_relative: Path
    logs_relative: Path


def generate_run_id(now: datetime | None = None) -> str:
    """Return a design-compliant minute-resolution run ID in Montreal/Toronto time."""
    if now is None:
        current = datetime.now(TORONTO_TZ)
    elif now.tzinfo is None:
        current = now.replace(tzinfo=TORONTO_TZ)
    else:
        current = now.astimezone(TORONTO_TZ)
    return current.strftime("%Y%m%d_%H%M")


def cascaded_stage_run_paths(
    *,
    stage: str,
    model: str,
    dataset: str,
    split: str,
    run_id: str,
    temporary: bool = False,
    direction: str | None = None,
    upstream_stage: str | None = None,
    upstream_model: str | None = None,
) -> RunPaths:
    relative = Path("cascaded") / "stages" / _stage_relative_parts(
        stage=stage,
        model=model,
        dataset=dataset,
        split=split,
        run_id=run_id,
        direction=direction,
        upstream_stage=upstream_stage,
        upstream_model=upstream_model,
    )
    return _run_paths(relative, temporary=temporary)


def cascaded_pipeline_run_paths(
    *,
    direction: str,
    pipeline_name: str,
    dataset: str,
    split: str,
    run_id: str,
    temporary: bool = False,
) -> RunPaths:
    _require_choice("direction", direction, DIRECTIONS)
    relative = (
        Path("cascaded")
        / "pipelines"
        / _safe_part(direction)
        / _safe_part(pipeline_name)
        / _safe_part(dataset)
        / _safe_part(split)
        / _safe_part(run_id)
    )
    return _run_paths(relative, temporary=temporary)


def create_run_directories(paths: RunPaths) -> None:
    if paths.result.exists():
        raise FileExistsError(f"result run directory already exists: {paths.result}")
    if paths.logs.exists():
        raise FileExistsError(f"log run directory already exists: {paths.logs}")
    result_created = False
    try:
        paths.result.mkdir(parents=True)
        result_created = True
        paths.logs.mkdir(parents=True)
    except Exception:
        if result_created:
            try:
                paths.result.rmdir()
            except OSError:
                pass
        raise


def validate_t2t_source(direction: str, upstream_stage: str) -> None:
    _require_choice("direction", direction, DIRECTIONS)
    expected = STAGE_ONE_BY_DIRECTION[direction]
    if upstream_stage != expected:
        raise ValueError(
            f"direction {direction!r} requires upstream stage {expected!r}; got {upstream_stage!r}"
        )


def relative_to_project_root(path: Path) -> str:
    repo_root = root_path("repo_root")
    try:
        return path.resolve().relative_to(repo_root).as_posix()
    except ValueError:
        return path.as_posix()


def _stage_relative_parts(
    *,
    stage: str,
    model: str,
    dataset: str,
    split: str,
    run_id: str,
    direction: str | None,
    upstream_stage: str | None,
    upstream_model: str | None,
) -> Path:
    _require_choice("stage", stage, CASCADED_STAGES)
    model_part = _safe_part(model)
    dataset_part = _safe_part(dataset)
    split_part = _safe_part(split)
    run_part = _safe_part(run_id)

    if stage in {"a2t", "s2t"}:
        _reject_extra(stage, direction=direction, upstream_stage=upstream_stage, upstream_model=upstream_model)
        return Path(stage) / model_part / dataset_part / split_part / run_part

    if stage == "t2t":
        if direction is None or upstream_stage is None or upstream_model is None:
            raise ValueError("t2t stage paths require direction, upstream_stage, and upstream_model")
        validate_t2t_source(direction, upstream_stage)
        return (
            Path("t2t")
            / _safe_part(direction)
            / model_part
            / f"from-{_safe_part(upstream_stage)}"
            / _safe_part(upstream_model)
            / dataset_part
            / split_part
            / run_part
        )

    expected_direction = "s2a" if stage == "t2a" else "a2s"
    if direction is not None and direction != expected_direction:
        raise ValueError(f"{stage} belongs to direction {expected_direction!r}; got {direction!r}")
    if upstream_stage not in {None, "t2t"}:
        raise ValueError(f"{stage} upstream_stage must be 't2t' when provided")
    if upstream_model is None:
        raise ValueError(f"{stage} stage paths require upstream_model for the source t2t model")
    return (
        Path(stage)
        / model_part
        / "from-t2t"
        / _safe_part(upstream_model)
        / dataset_part
        / split_part
        / run_part
    )


def _run_paths(relative: Path, *, temporary: bool) -> RunPaths:
    result_root_name: Literal["results_root", "temporary_results_root"] = (
        "temporary_results_root" if temporary else "results_root"
    )
    logs_root_name: Literal["logs_root", "temporary_logs_root"] = (
        "temporary_logs_root" if temporary else "logs_root"
    )
    return RunPaths(
        result=root_path(result_root_name) / relative,
        logs=root_path(logs_root_name) / relative,
        result_relative=Path("temporary/results" if temporary else "results") / relative,
        logs_relative=Path("temporary/logs" if temporary else "logs") / relative,
    )


def _safe_part(value: str) -> str:
    if not isinstance(value, str) or not value:
        raise ValueError("path part must be a non-empty string")
    if value in {".", ".."} or "/" in value or "\\" in value:
        raise ValueError(f"unsafe path part: {value!r}")
    return value


def _require_choice(name: str, value: str, choices: set[str]) -> None:
    if value not in choices:
        expected = ", ".join(sorted(choices))
        raise ValueError(f"unsupported {name}: {value!r}; expected one of {expected}")


def _reject_extra(stage: str, **values: str | None) -> None:
    extras = {name: value for name, value in values.items() if value is not None}
    if extras:
        names = ", ".join(sorted(extras))
        raise ValueError(f"{stage} stage paths do not accept {names}")
