from __future__ import annotations

import math
from collections import Counter
from dataclasses import asdict, dataclass
from numbers import Real
from typing import Any, Iterable, Mapping, Sequence

from evaluation.errors import EvaluationError
from evaluation.metrics.registry import MetricSpec


@dataclass(frozen=True)
class ValidationIssue:
    code: str
    message: str
    metric: str | None = None
    sample_id: str | None = None


@dataclass(frozen=True)
class ResultValidationReport:
    mode: str
    status: str
    scientifically_valid: bool
    expected_row_count: int
    observed_row_count: int
    sample_metric_coverage: dict[str, int]
    corpus_metric_coverage: dict[str, bool]
    issues: tuple[ValidationIssue, ...]

    @property
    def ok(self) -> bool:
        return not self.issues

    def to_dict(self) -> dict[str, Any]:
        result = asdict(self)
        result["ok"] = self.ok
        return result


class FinalResultValidationError(EvaluationError):
    def __init__(self, report: ResultValidationReport) -> None:
        self.report = report
        preview = "; ".join(issue.message for issue in report.issues[:5])
        suffix = "" if len(report.issues) <= 5 else f"; ... {len(report.issues) - 5} more"
        super().__init__(f"Final scientific result validation failed: {preview}{suffix}")


def validate_final_result_completeness(
    metric_specs: Sequence[MetricSpec],
    selected_row_ids: Iterable[str],
    per_sample_rows: Sequence[Mapping[str, Any]],
    corpus_metrics: Mapping[str, Any] | None,
    *,
    diagnostic: bool = False,
) -> ResultValidationReport:
    """Validate complete outputs, raising on invalid final scientific results."""
    issues: list[ValidationIssue] = []
    specs = list(metric_specs)
    expected_ids = list(selected_row_ids)
    corpus_metrics = corpus_metrics if corpus_metrics is not None else {}

    _validate_specs(specs, issues)
    _validate_expected_ids(expected_ids, issues)
    if not isinstance(corpus_metrics, Mapping):
        issues.append(ValidationIssue("malformed_corpus_metrics", "corpus_metrics must be a mapping"))
        corpus_metrics = {}

    rows_by_id = _index_rows(per_sample_rows, expected_ids, issues)
    sample_coverage: dict[str, int] = {}
    corpus_coverage: dict[str, bool] = {}

    for spec in specs:
        if spec.level == "sample":
            covered = 0
            for sample_id in expected_ids:
                row = rows_by_id.get(sample_id)
                if row is None:
                    continue
                metrics = row.get("metrics")
                if not isinstance(metrics, Mapping) or spec.name not in metrics:
                    issues.append(
                        ValidationIssue(
                            "missing_sample_metric",
                            f"{sample_id}: missing sample metric {spec.name}",
                            metric=spec.name,
                            sample_id=sample_id,
                        )
                    )
                    continue
                if _validate_numeric(
                    metrics[spec.name],
                    f"{sample_id}: metric {spec.name}",
                    issues,
                    spec.name,
                    sample_id,
                ):
                    covered += 1
            sample_coverage[spec.name] = covered
        elif spec.level == "corpus":
            present = spec.name in corpus_metrics
            corpus_coverage[spec.name] = present
            if not present:
                issues.append(
                    ValidationIssue(
                        "missing_corpus_metric",
                        f"missing corpus metric {spec.name}",
                        metric=spec.name,
                    )
                )
            else:
                _validate_corpus_output(spec.name, corpus_metrics[spec.name], len(expected_ids), issues)
        else:
            issues.append(
                ValidationIssue(
                    "invalid_metric_level",
                    f"metric {spec.name} has unsupported level {spec.level!r}",
                    metric=spec.name,
                )
            )

    selected_names = {spec.name for spec in specs}
    for metric_name, value in corpus_metrics.items():
        if metric_name in selected_names and metric_name not in corpus_coverage:
            _validate_corpus_output(metric_name, value, len(expected_ids), issues)

    mode = "diagnostic" if diagnostic else "final"
    status = "diagnostic_non_authoritative" if diagnostic else (
        "invalid" if issues else "final_scientific_valid"
    )
    report = ResultValidationReport(
        mode=mode,
        status=status,
        scientifically_valid=not diagnostic and not issues,
        expected_row_count=len(expected_ids),
        observed_row_count=len(per_sample_rows),
        sample_metric_coverage=sample_coverage,
        corpus_metric_coverage=corpus_coverage,
        issues=tuple(issues),
    )
    if issues and not diagnostic:
        raise FinalResultValidationError(report)
    return report


def _validate_specs(specs: list[MetricSpec], issues: list[ValidationIssue]) -> None:
    for name, count in Counter(spec.name for spec in specs).items():
        if count > 1:
            issues.append(
                ValidationIssue(
                    "duplicate_metric_spec",
                    f"metric {name} was selected {count} times",
                    metric=name,
                )
            )


def _validate_expected_ids(expected_ids: list[str], issues: list[ValidationIssue]) -> None:
    for index, sample_id in enumerate(expected_ids):
        if not isinstance(sample_id, str) or not sample_id.strip():
            issues.append(
                ValidationIssue(
                    "invalid_selected_row_id",
                    f"selected row ID at position {index} is invalid",
                )
            )
    for sample_id, count in Counter(expected_ids).items():
        if count > 1:
            issues.append(
                ValidationIssue(
                    "duplicate_selected_row_id",
                    f"selected row ID {sample_id!r} appears {count} times",
                    sample_id=sample_id if isinstance(sample_id, str) else None,
                )
            )


def _index_rows(
    rows: Sequence[Mapping[str, Any]],
    expected_ids: list[str],
    issues: list[ValidationIssue],
) -> dict[str, Mapping[str, Any]]:
    indexed: dict[str, Mapping[str, Any]] = {}
    counts: Counter[str] = Counter()
    expected = set(expected_ids)
    for index, row in enumerate(rows):
        if not isinstance(row, Mapping):
            issues.append(ValidationIssue("malformed_row", f"per-sample row {index} is not a mapping"))
            continue
        sample_id = row.get("sample_id")
        if not isinstance(sample_id, str) or not sample_id.strip():
            issues.append(ValidationIssue("missing_row_id", f"per-sample row {index} has no valid sample_id"))
            continue
        counts[sample_id] += 1
        indexed.setdefault(sample_id, row)
        if sample_id not in expected:
            issues.append(
                ValidationIssue(
                    "unexpected_row_id",
                    f"unexpected per-sample row ID {sample_id!r}",
                    sample_id=sample_id,
                )
            )
        if row.get("errors"):
            issues.append(
                ValidationIssue("row_errors", f"{sample_id}: result contains errors", sample_id=sample_id)
            )
        if row.get("status") != "success":
            issues.append(
                ValidationIssue(
                    "unsuccessful_row",
                    f"{sample_id}: row status is {row.get('status')!r}, expected 'success'",
                    sample_id=sample_id,
                )
            )
        if not isinstance(row.get("metrics"), Mapping):
            issues.append(
                ValidationIssue(
                    "malformed_sample_metrics",
                    f"{sample_id}: metrics must be a mapping",
                    sample_id=sample_id,
                )
            )

    for sample_id, count in counts.items():
        if count > 1:
            issues.append(
                ValidationIssue(
                    "duplicate_row_id",
                    f"per-sample row ID {sample_id!r} appears {count} times",
                    sample_id=sample_id,
                )
            )
    for sample_id in expected_ids:
        if counts[sample_id] == 0:
            issues.append(
                ValidationIssue(
                    "missing_row_id",
                    f"missing per-sample row ID {sample_id!r}",
                    sample_id=sample_id,
                )
            )
    return indexed


def _validate_numeric(
    value: Any,
    label: str,
    issues: list[ValidationIssue],
    metric: str,
    sample_id: str | None = None,
) -> bool:
    if isinstance(value, bool) or not isinstance(value, Real):
        issues.append(
            ValidationIssue(
                "non_numeric_metric_value",
                f"{label} must be a numeric non-bool value",
                metric=metric,
                sample_id=sample_id,
            )
        )
        return False
    if not math.isfinite(float(value)):
        issues.append(
            ValidationIssue(
                "non_finite_metric_value",
                f"{label} must be finite",
                metric=metric,
                sample_id=sample_id,
            )
        )
        return False
    return True


def _validate_corpus_output(
    metric: str,
    value: Any,
    expected_count: int,
    issues: list[ValidationIssue],
) -> None:
    if not isinstance(value, Mapping):
        _validate_numeric(value, f"corpus metric {metric}", issues, metric)
        return

    if value.get("level") != "corpus":
        issues.append(
            ValidationIssue(
                "malformed_corpus_summary",
                f"corpus metric {metric} summary must declare level='corpus'",
                metric=metric,
            )
        )
    if "mean" not in value:
        issues.append(
            ValidationIssue(
                "malformed_corpus_summary",
                f"corpus metric {metric} summary is missing mean",
                metric=metric,
            )
        )
    for field, field_value in value.items():
        if isinstance(field_value, bool):
            issues.append(
                ValidationIssue(
                    "boolean_corpus_value",
                    f"corpus metric {metric}.{field} must not be bool",
                    metric=metric,
                )
            )
        elif isinstance(field_value, Real) and not math.isfinite(float(field_value)):
            issues.append(
                ValidationIssue(
                    "non_finite_metric_value",
                    f"corpus metric {metric}.{field} must be finite",
                    metric=metric,
                )
            )
    if "mean" in value:
        _validate_numeric(value["mean"], f"corpus metric {metric}.mean", issues, metric)
    for field in ("std", "min", "max", "median"):
        if field in value:
            _validate_numeric(value[field], f"corpus metric {metric}.{field}", issues, metric)
    if "std" in value and isinstance(value["std"], Real) and not isinstance(value["std"], bool):
        if math.isfinite(float(value["std"])) and value["std"] < 0:
            issues.append(
                ValidationIssue(
                    "malformed_corpus_summary",
                    f"corpus metric {metric}.std must be non-negative",
                    metric=metric,
                )
            )
    for field in ("count", "input_count"):
        if field not in value:
            continue
        count = value[field]
        if isinstance(count, bool) or not isinstance(count, int) or count < 1:
            issues.append(
                ValidationIssue(
                    "malformed_corpus_summary",
                    f"corpus metric {metric}.{field} must be a positive integer",
                    metric=metric,
                )
            )
        elif field == "input_count" and count != expected_count:
            issues.append(
                ValidationIssue(
                    "corpus_input_count_mismatch",
                    f"corpus metric {metric}.input_count is {count}, expected {expected_count}",
                    metric=metric,
                )
            )


validate_results = validate_final_result_completeness
