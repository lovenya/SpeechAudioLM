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
    comparison_ready: bool
    expected_row_count: int
    observed_row_count: int
    metric_counts: dict[str, dict[str, int]]
    failed_sample_ids: dict[str, tuple[str, ...]]
    issues: tuple[ValidationIssue, ...]
    blocking_issues: tuple[ValidationIssue, ...]

    @property
    def ok(self) -> bool:
        return not self.blocking_issues

    def to_dict(self) -> dict[str, Any]:
        result = asdict(self)
        result["ok"] = self.ok
        result["has_errors"] = bool(self.issues)
        result["result_status"] = (
            "error"
            if self.metric_counts and all(
                counts["passed_count"] == 0 for counts in self.metric_counts.values()
            )
            else "passed_with_errors"
            if any(counts["error_count"] for counts in self.metric_counts.values())
            else "passed"
        )
        result["union_of_failures_recompute_todo"] = True
        return result


class FinalResultValidationError(EvaluationError):
    def __init__(self, report: ResultValidationReport) -> None:
        self.report = report
        preview = "; ".join(issue.message for issue in report.blocking_issues[:5])
        super().__init__(f"Final result structure is invalid: {preview}")


def validate_final_result_completeness(
    metric_specs: Sequence[MetricSpec],
    selected_row_ids: Iterable[str],
    per_sample_rows: Sequence[Mapping[str, Any]],
    corpus_metrics: Mapping[str, Any] | None,
    *,
    diagnostic: bool = False,
) -> ResultValidationReport:
    """Validate attribution while retaining final results with reported errors."""

    issues: list[ValidationIssue] = []
    blockers: list[ValidationIssue] = []
    specs = list(metric_specs)
    expected_ids = list(selected_row_ids)
    corpus_metrics = corpus_metrics if isinstance(corpus_metrics, Mapping) else {}

    _validate_structure(specs, expected_ids, per_sample_rows, issues, blockers)
    rows_by_id = {
        row["sample_id"]: row
        for row in per_sample_rows
        if isinstance(row, Mapping) and isinstance(row.get("sample_id"), str)
    }

    metric_counts: dict[str, dict[str, int]] = {}
    failed_sample_ids: dict[str, tuple[str, ...]] = {}

    for spec in specs:
        failed: list[str] = []
        passed = 0

        if spec.level == "sample":
            for sample_id in expected_ids:
                row = rows_by_id.get(sample_id)
                metrics = row.get("metrics") if isinstance(row, Mapping) else None
                value = metrics.get(spec.name) if isinstance(metrics, Mapping) else None
                if _finite_numeric(value):
                    passed += 1
                else:
                    failed.append(sample_id)
                    issues.append(
                        ValidationIssue(
                            "metric_sample_error",
                            f"{sample_id}: no valid {spec.name} score",
                            metric=spec.name,
                            sample_id=sample_id,
                        )
                    )
        elif spec.level == "corpus":
            value = corpus_metrics.get(spec.name)
            if isinstance(value, Mapping):
                passed = _declared_count(value, "passed_count", "valid_count", "input_count")
                declared_failed = value.get("error_sample_ids", value.get("failed_sample_ids", []))
                if isinstance(declared_failed, Sequence) and not isinstance(declared_failed, (str, bytes)):
                    failed = [str(item) for item in declared_failed]
                if passed == 0 and _finite_numeric(value.get("mean")):
                    passed = max(0, len(expected_ids) - len(failed))
            elif _finite_numeric(value):
                passed = len(expected_ids)

            if passed > len(expected_ids):
                issue = ValidationIssue(
                    "invalid_metric_count",
                    f"{spec.name}: passed_count exceeds attempted_count",
                    metric=spec.name,
                )
                issues.append(issue)
                blockers.append(issue)
                passed = len(expected_ids)

            if not failed and passed < len(expected_ids):
                failed = expected_ids[passed:]

            if passed == 0:
                issues.append(
                    ValidationIssue(
                        "metric_error",
                        f"{spec.name}: no valid corpus score",
                        metric=spec.name,
                    )
                )
        else:
            issue = ValidationIssue(
                "invalid_metric_level",
                f"{spec.name}: unsupported level {spec.level!r}",
                metric=spec.name,
            )
            issues.append(issue)
            blockers.append(issue)

        failed = list(dict.fromkeys(failed))
        attempted = len(expected_ids)
        error_count = max(attempted - passed, len(failed))
        metric_counts[spec.name] = {
            "attempted_count": attempted,
            "passed_count": passed,
            "error_count": error_count,
        }
        failed_sample_ids[spec.name] = tuple(failed)

    has_errors = any(counts["error_count"] for counts in metric_counts.values())
    if diagnostic:
        status = "diagnostic_non_authoritative"
        scientifically_valid = False
    else:
        status = "final_scientific"
        scientifically_valid = not blockers

    report = ResultValidationReport(
        mode="diagnostic" if diagnostic else "final",
        status=status,
        scientifically_valid=scientifically_valid,
        comparison_ready=not has_errors and not blockers,
        expected_row_count=len(expected_ids),
        observed_row_count=len(per_sample_rows),
        metric_counts=metric_counts,
        failed_sample_ids=failed_sample_ids,
        issues=tuple(issues),
        blocking_issues=tuple(blockers),
    )
    if blockers and not diagnostic:
        raise FinalResultValidationError(report)
    return report


def _validate_structure(
    specs: list[MetricSpec],
    expected_ids: list[str],
    rows: Sequence[Mapping[str, Any]],
    issues: list[ValidationIssue],
    blockers: list[ValidationIssue],
) -> None:
    for name, count in Counter(spec.name for spec in specs).items():
        if count > 1:
            issue = ValidationIssue("duplicate_metric_spec", f"metric {name} selected {count} times")
            issues.append(issue)
            blockers.append(issue)

    if any(not isinstance(value, str) or not value.strip() for value in expected_ids):
        issue = ValidationIssue("invalid_selected_row_id", "selected row IDs must be non-empty strings")
        issues.append(issue)
        blockers.append(issue)
    if len(expected_ids) != len(set(expected_ids)):
        issue = ValidationIssue("duplicate_selected_row_id", "selected row IDs contain duplicates")
        issues.append(issue)
        blockers.append(issue)

    actual_ids: list[str] = []
    for index, row in enumerate(rows):
        if not isinstance(row, Mapping):
            issue = ValidationIssue("malformed_row", f"row {index} is not a mapping")
            issues.append(issue)
            blockers.append(issue)
            continue
        sample_id = row.get("sample_id")
        if not isinstance(sample_id, str) or not sample_id.strip():
            issue = ValidationIssue("missing_row_id", f"row {index} has no valid sample_id")
            issues.append(issue)
            blockers.append(issue)
            continue
        actual_ids.append(sample_id)

    if len(actual_ids) != len(set(actual_ids)):
        issue = ValidationIssue("duplicate_row_id", "result rows contain duplicate sample IDs")
        issues.append(issue)
        blockers.append(issue)

    expected_set = set(expected_ids)
    actual_set = set(actual_ids)
    if expected_set != actual_set:
        issue = ValidationIssue(
            "row_id_set_mismatch",
            f"result row IDs differ from attempted IDs; missing={sorted(expected_set-actual_set)}, "
            f"unexpected={sorted(actual_set-expected_set)}",
        )
        issues.append(issue)
        blockers.append(issue)


def _finite_numeric(value: Any) -> bool:
    return (
        isinstance(value, Real)
        and not isinstance(value, bool)
        and math.isfinite(float(value))
    )


def _declared_count(value: Mapping[str, Any], *fields: str) -> int:
    for field in fields:
        count = value.get(field)
        if isinstance(count, int) and not isinstance(count, bool) and count >= 0:
            return count
    return 0


validate_results = validate_final_result_completeness
