from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from typing import Any


class ContractError(ValueError):
    """Raised when an evaluation contract or scope is scientifically invalid."""



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
