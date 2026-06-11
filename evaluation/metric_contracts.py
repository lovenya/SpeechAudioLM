from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from enum import Enum
from pathlib import Path
from typing import Any, Mapping, Sequence


class ContractError(ValueError):
    """Raised when an evaluation contract or scope is scientifically invalid."""


class EvaluationMode(str, Enum):
    FINAL = "final"
    BACKEND_SMOKE = "backend_smoke"


class MetricLevel(str, Enum):
    SAMPLE = "sample"
    PAIRED_CORPUS = "paired-corpus"
    DISTRIBUTIONAL_CORPUS = "distributional-corpus"


def _require_nonempty_text(value: str, field_name: str) -> None:
    if not isinstance(value, str) or not value.strip():
        raise ContractError(f"{field_name} must be non-empty text")


def _require_nonempty_text_tuple(values: tuple[str, ...], field_name: str) -> None:
    if not isinstance(values, tuple) or not values:
        raise ContractError(f"{field_name} must be a non-empty tuple")
    for value in values:
        _require_nonempty_text(value, field_name)
    if len(values) != len(set(values)):
        raise ContractError(f"{field_name} must not contain duplicates")


@dataclass(frozen=True)
class MetricContract:
    name: str
    level: MetricLevel
    academic_definition: str
    implementation: str
    implementation_version: str
    higher_is_better: bool
    evaluated_fields: tuple[str, ...]
    reference_construction: str
    successful_id_policy: str
    constraints: tuple[str, ...]
    output_validation_rules: tuple[str, ...]
    shard_composable: bool
    minimum_valid_corpus_size: int
    known_limitations: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        for field_name in (
            "name",
            "academic_definition",
            "implementation",
            "implementation_version",
            "reference_construction",
            "successful_id_policy",
        ):
            _require_nonempty_text(getattr(self, field_name), field_name)
        if not isinstance(self.level, MetricLevel):
            raise ContractError("level must be a MetricLevel")
        if not isinstance(self.higher_is_better, bool):
            raise ContractError("higher_is_better must be boolean")
        if not isinstance(self.shard_composable, bool):
            raise ContractError("shard_composable must be boolean")
        if (
            not isinstance(self.minimum_valid_corpus_size, int)
            or isinstance(self.minimum_valid_corpus_size, bool)
            or self.minimum_valid_corpus_size < 1
        ):
            raise ContractError("minimum_valid_corpus_size must be a positive integer")
        _require_nonempty_text_tuple(self.evaluated_fields, "evaluated_fields")
        _require_nonempty_text_tuple(self.constraints, "constraints")
        _require_nonempty_text_tuple(
            self.output_validation_rules, "output_validation_rules"
        )
        if not isinstance(self.known_limitations, tuple):
            raise ContractError("known_limitations must be a tuple")
        for limitation in self.known_limitations:
            _require_nonempty_text(limitation, "known_limitations")

    def report_fields(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "metric_level": self.level.value,
            "higher_is_better": self.higher_is_better,
            "evaluated_fields": list(self.evaluated_fields),
            "constraints": list(self.constraints),
            "shard_composable": self.shard_composable,
            "minimum_valid_corpus_size": self.minimum_valid_corpus_size,
            "output_validation_rules": list(self.output_validation_rules),
            "known_limitations": list(self.known_limitations),
        }


def _validated_ids(values: tuple[str, ...], field_name: str) -> frozenset[str]:
    if not isinstance(values, tuple) or not values:
        raise ContractError(f"{field_name} must be a non-empty tuple")
    for sample_id in values:
        _require_nonempty_text(sample_id, field_name)
    if len(values) != len(set(values)):
        raise ContractError(f"{field_name} must not contain duplicate sample IDs")
    return frozenset(values)


@dataclass(frozen=True)
class EvaluationScope:
    mode: EvaluationMode
    expected_sample_ids: tuple[str, ...]
    common_sample_ids: tuple[str, ...]
    selected_sample_ids: tuple[str, ...]
    start_index: int = 0
    max_samples: int | None = None

    def __post_init__(self) -> None:
        if not isinstance(self.mode, EvaluationMode):
            raise ContractError("mode must be an EvaluationMode")
        if (
            not isinstance(self.start_index, int)
            or isinstance(self.start_index, bool)
            or self.start_index < 0
        ):
            raise ContractError("start_index must be a non-negative integer")
        if self.max_samples is not None and (
            not isinstance(self.max_samples, int)
            or isinstance(self.max_samples, bool)
            or self.max_samples < 1
        ):
            raise ContractError("max_samples must be a positive integer or None")

        expected = _validated_ids(self.expected_sample_ids, "expected_sample_ids")
        common = _validated_ids(self.common_sample_ids, "common_sample_ids")
        selected = _validated_ids(self.selected_sample_ids, "selected_sample_ids")

        if self.mode is EvaluationMode.FINAL:
            if self.start_index != 0 or self.max_samples is not None:
                raise ContractError("final evaluation disallows slicing")
            if selected != expected:
                raise ContractError(
                    "final evaluation requires the complete expected sample-ID set"
                )
            if selected != common:
                raise ContractError(
                    "final evaluation requires the complete common sample-ID set"
                )

        if not selected.issubset(expected):
            raise ContractError("selected_sample_ids must be a subset of expected_sample_ids")
        if not selected.issubset(common):
            raise ContractError("selected_sample_ids must be a subset of common_sample_ids")

    @property
    def scientific(self) -> bool:
        return self.mode is EvaluationMode.FINAL

    @property
    def publishable(self) -> bool:
        return self.mode is EvaluationMode.FINAL

    def report_fields(self) -> dict[str, Any]:
        return {
            "evaluation_mode": self.mode.value,
            "scientific": self.scientific,
            "publishable": self.publishable,
            "scope_label": (
                "final_scientific_evaluation"
                if self.mode is EvaluationMode.FINAL
                else "backend_smoke_non_scientific_non_publishable"
            ),
            "expected_sample_count": len(self.expected_sample_ids),
            "common_sample_count": len(self.common_sample_ids),
            "selected_sample_count": len(self.selected_sample_ids),
            "start_index": self.start_index,
            "max_samples": self.max_samples,
        }


def _sha256(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def _canonical_json_bytes(value: Any) -> bytes:
    try:
        encoded = json.dumps(
            value,
            allow_nan=False,
            ensure_ascii=False,
            separators=(",", ":"),
            sort_keys=True,
        )
    except (TypeError, ValueError) as exc:
        raise ContractError(f"value is not canonical JSON: {exc}") from exc
    return encoded.encode("utf-8")


def sha256_manifest(path: Path) -> str:
    """Hash exact manifest bytes so any byte-level manifest change is visible."""
    return _sha256(path.read_bytes())


def _json_source(source: Path | Mapping[str, Any]) -> Any:
    if isinstance(source, Path):
        try:
            return json.loads(source.read_text(encoding="utf-8"))
        except (OSError, UnicodeError, json.JSONDecodeError) as exc:
            raise ContractError(f"unable to load canonical JSON source {source}: {exc}") from exc
    if not isinstance(source, Mapping):
        raise ContractError("config and policy sources must be a Path or mapping")
    return source


def sha256_config(source: Path | Mapping[str, Any]) -> str:
    return _sha256(_canonical_json_bytes(_json_source(source)))


def sha256_policy(source: Path | Mapping[str, Any]) -> str:
    return _sha256(_canonical_json_bytes(_json_source(source)))


def sha256_selected_rows(rows: Sequence[Mapping[str, Any]]) -> str:
    if not isinstance(rows, Sequence) or isinstance(rows, (str, bytes)):
        raise ContractError("selected rows must be a sequence of mappings")
    if not rows:
        raise ContractError("selected rows must not be empty")
    if any(not isinstance(row, Mapping) for row in rows):
        raise ContractError("each selected row must be a mapping")
    return _sha256(_canonical_json_bytes(list(rows)))


def sha256_sample_id_set(sample_ids: Sequence[str]) -> str:
    if not isinstance(sample_ids, Sequence) or isinstance(sample_ids, (str, bytes)):
        raise ContractError("sample IDs must be a sequence")
    ids = tuple(sample_ids)
    _validated_ids(ids, "sample_ids")
    return _sha256(_canonical_json_bytes(sorted(ids)))


def _validate_sha256(value: str, field_name: str) -> None:
    if (
        not isinstance(value, str)
        or len(value) != 64
        or any(character not in "0123456789abcdef" for character in value)
    ):
        raise ContractError(f"{field_name} must be a lowercase SHA-256 hex digest")


@dataclass(frozen=True)
class EvaluationProvenance:
    manifest_sha256: str
    config_sha256: str
    policy_sha256: str
    selected_rows_sha256: str
    sample_id_set_sha256: str
    git_commit: str
    git_worktree_sha256: str

    def __post_init__(self) -> None:
        for field_name in (
            "manifest_sha256",
            "config_sha256",
            "policy_sha256",
            "selected_rows_sha256",
            "sample_id_set_sha256",
            "git_worktree_sha256",
        ):
            _validate_sha256(getattr(self, field_name), field_name)
        if (
            not isinstance(self.git_commit, str)
            or len(self.git_commit) not in (40, 64)
            or any(character not in "0123456789abcdef" for character in self.git_commit)
        ):
            raise ContractError("git_commit must be a lowercase Git object ID")

    def report_fields(self) -> dict[str, str]:
        return {
            "manifest_sha256": self.manifest_sha256,
            "config_sha256": self.config_sha256,
            "policy_sha256": self.policy_sha256,
            "selected_rows_sha256": self.selected_rows_sha256,
            "sample_id_set_sha256": self.sample_id_set_sha256,
            "git_commit": self.git_commit,
            "git_worktree_sha256": self.git_worktree_sha256,
        }
