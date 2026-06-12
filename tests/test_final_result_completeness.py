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
        self.assertEqual(report.status, "final_scientific")
        self.assertTrue(report.comparison_ready)
        self.assertEqual(report.metric_counts["sample_score"], {
            "attempted_count": 2, "passed_count": 2, "error_count": 0,
        })
        self.assertEqual(report.metric_counts["corpus_score"], {
            "attempted_count": 2, "passed_count": 2, "error_count": 0,
        })
        self.assertEqual(report.to_dict()["result_status"], "passed")

    def test_retains_attributable_sample_errors_in_final_report(self) -> None:
        report = validate_final_result_completeness(
            [SAMPLE_METRIC], ["a"],
            [_row("a", metrics={}, status="failed", errors=["backend failed"])], {},
        )
        self.assertTrue(report.ok)
        self.assertTrue(report.scientifically_valid)
        self.assertFalse(report.comparison_ready)
        self.assertEqual({issue.code for issue in report.issues}, {"metric_sample_error"})
        self.assertEqual(report.metric_counts["sample_score"], {
            "attempted_count": 1, "passed_count": 0, "error_count": 1,
        })
        self.assertEqual(report.failed_sample_ids, {"sample_score": ("a",)})
        self.assertEqual(report.to_dict()["result_status"], "error")

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
        self.assertIn("row_id_set_mismatch", codes)

    def test_retains_boolean_and_nonfinite_metric_values_as_errors(self) -> None:
        report = validate_final_result_completeness(
            [
                MetricSpec(name="boolean_sample", level="sample"),
                MetricSpec(name="nonfinite_sample", level="sample"),
                MetricSpec(name="boolean_corpus", level="corpus"),
                MetricSpec(name="nonfinite_corpus", level="corpus"),
            ],
            ["a"],
            [_row("a", metrics={"boolean_sample": True, "nonfinite_sample": float("nan")})],
            {
                "boolean_corpus": {"level": "corpus", "mean": False, "input_count": 1},
                "nonfinite_corpus": {"level": "corpus", "mean": float("inf"), "input_count": 1},
            },
        )
        self.assertTrue(report.ok)
        self.assertEqual(report.to_dict()["result_status"], "passed_with_errors")
        self.assertEqual(report.metric_counts["boolean_sample"]["passed_count"], 0)
        self.assertEqual(report.metric_counts["nonfinite_sample"]["passed_count"], 0)
        self.assertEqual(report.metric_counts["boolean_corpus"]["passed_count"], 1)
        self.assertEqual(report.metric_counts["nonfinite_corpus"]["passed_count"], 1)

    def test_retains_missing_malformed_and_wrong_sized_corpus_outputs(self) -> None:
        specs = [
            MetricSpec(name="missing", level="corpus"),
            MetricSpec(name="malformed", level="corpus"),
            MetricSpec(name="wrong_size", level="corpus"),
        ]
        report = validate_final_result_completeness(
            specs, ["a", "b"], [_row("a"), _row("b")],
            {
                "malformed": {"level": "sample", "input_count": 2},
                "wrong_size": {"level": "corpus", "mean": 0.25, "input_count": 1},
            },
        )
        self.assertTrue(report.ok)
        self.assertEqual(report.to_dict()["result_status"], "passed_with_errors")
        self.assertEqual(report.failed_sample_ids["missing"], ("a", "b"))
        self.assertEqual(report.metric_counts["missing"]["passed_count"], 0)
        self.assertEqual(report.metric_counts["malformed"]["passed_count"], 2)
        self.assertEqual(report.metric_counts["wrong_size"]["passed_count"], 1)

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
