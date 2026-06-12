from __future__ import annotations

import unittest

from evaluation.metrics.registry import MetricSpec
from evaluation.result_validation import (
    FinalResultValidationError,
    validate_final_result_completeness,
)


SAMPLE_METRIC = MetricSpec(name="sample_score", level="sample")
CORPUS_METRIC = MetricSpec(name="corpus_score", level="corpus")


def _row(
    sample_id: str,
    *,
    metrics: object = None,
    status: str = "success",
    errors: object = None,
) -> dict:
    return {
        "sample_id": sample_id,
        "status": status,
        "errors": [] if errors is None else errors,
        "metrics": {"sample_score": 0.5} if metrics is None else metrics,
    }


def _issue_codes(error: FinalResultValidationError) -> set[str]:
    return {issue.code for issue in error.report.issues}


class FinalResultCompletenessTest(unittest.TestCase):
    def test_accepts_complete_sample_and_corpus_outputs(self) -> None:
        report = validate_final_result_completeness(
            [SAMPLE_METRIC, CORPUS_METRIC],
            ["a", "b"],
            [_row("a"), _row("b", metrics={"sample_score": 0.75})],
            {
                "corpus_score": {
                    "level": "corpus",
                    "mean": 0.625,
                    "std": 0.125,
                    "min": 0.5,
                    "max": 0.75,
                    "median": 0.625,
                    "count": 2,
                    "input_count": 2,
                }
            },
        )

        self.assertTrue(report.ok)
        self.assertTrue(report.scientifically_valid)
        self.assertEqual(report.status, "final_scientific_valid")
        self.assertEqual(report.sample_metric_coverage, {"sample_score": 2})
        self.assertEqual(report.corpus_metric_coverage, {"corpus_score": True})

    def test_rejects_row_errors_unsuccessful_status_and_missing_metrics(self) -> None:
        with self.assertRaises(FinalResultValidationError) as raised:
            validate_final_result_completeness(
                [SAMPLE_METRIC],
                ["a"],
                [_row("a", metrics={}, status="failed", errors=["backend failed"])],
                {},
            )

        self.assertEqual(
            _issue_codes(raised.exception),
            {"row_errors", "unsuccessful_row", "missing_sample_metric"},
        )
        self.assertEqual(
            raised.exception.report.sample_metric_coverage,
            {"sample_score": 0},
        )

    def test_rejects_duplicate_missing_and_unexpected_row_ids(self) -> None:
        with self.assertRaises(FinalResultValidationError) as raised:
            validate_final_result_completeness(
                [SAMPLE_METRIC],
                ["a", "b"],
                [_row("a"), _row("a"), _row("extra")],
                {},
            )

        codes = _issue_codes(raised.exception)
        self.assertIn("duplicate_row_id", codes)
        self.assertIn("missing_row_id", codes)
        self.assertIn("unexpected_row_id", codes)

    def test_rejects_boolean_and_nonfinite_metric_values(self) -> None:
        with self.assertRaises(FinalResultValidationError) as raised:
            validate_final_result_completeness(
                [
                    MetricSpec(name="boolean_sample", level="sample"),
                    MetricSpec(name="nonfinite_sample", level="sample"),
                    MetricSpec(name="boolean_corpus", level="corpus"),
                    MetricSpec(name="nonfinite_corpus", level="corpus"),
                ],
                ["a"],
                [
                    _row(
                        "a",
                        metrics={
                            "boolean_sample": True,
                            "nonfinite_sample": float("nan"),
                        },
                    )
                ],
                {
                    "boolean_corpus": {
                        "level": "corpus",
                        "mean": False,
                        "input_count": 1,
                    },
                    "nonfinite_corpus": {
                        "level": "corpus",
                        "mean": float("inf"),
                        "input_count": 1,
                    },
                },
            )

        codes = _issue_codes(raised.exception)
        self.assertIn("non_numeric_metric_value", codes)
        self.assertIn("boolean_corpus_value", codes)
        self.assertIn("non_finite_metric_value", codes)

    def test_rejects_missing_malformed_and_wrong_sized_corpus_outputs(self) -> None:
        specs = [
            MetricSpec(name="missing", level="corpus"),
            MetricSpec(name="malformed", level="corpus"),
            MetricSpec(name="wrong_size", level="corpus"),
        ]
        with self.assertRaises(FinalResultValidationError) as raised:
            validate_final_result_completeness(
                specs,
                ["a", "b"],
                [_row("a"), _row("b")],
                {
                    "malformed": {"level": "sample", "input_count": 2},
                    "wrong_size": {
                        "level": "corpus",
                        "mean": 0.25,
                        "input_count": 1,
                    },
                },
            )

        codes = _issue_codes(raised.exception)
        self.assertIn("missing_corpus_metric", codes)
        self.assertIn("malformed_corpus_summary", codes)
        self.assertIn("corpus_input_count_mismatch", codes)
        self.assertEqual(
            raised.exception.report.corpus_metric_coverage,
            {"missing": False, "malformed": True, "wrong_size": True},
        )

    def test_diagnostic_mode_never_claims_scientific_validity(self) -> None:
        valid_report = validate_final_result_completeness(
            [SAMPLE_METRIC],
            ["a"],
            [_row("a")],
            {},
            diagnostic=True,
        )
        invalid_report = validate_final_result_completeness(
            [SAMPLE_METRIC],
            ["a"],
            [_row("a", metrics={})],
            {},
            diagnostic=True,
        )

        for report in (valid_report, invalid_report):
            with self.subTest(ok=report.ok):
                self.assertEqual(report.mode, "diagnostic")
                self.assertEqual(report.status, "diagnostic_non_authoritative")
                self.assertFalse(report.scientifically_valid)


if __name__ == "__main__":
    unittest.main()
