from __future__ import annotations

import math
import unittest

from evaluation.metrics.kl_passt import (
    KL_DIRECTION,
    KLPaSSTPairRecord,
    KLPaSSTValidationError,
    collect_window_probabilities,
    official_passt_preprocessing,
    pair_kl_passt,
    validate_kl_passt_records,
)


class KLPaSSTSemanticsTest(unittest.TestCase):
    def test_pair_kl_direction_is_eval_to_smoothed_reference(self) -> None:
        reference = [0.8, 0.2]
        evaluated = [0.25, 0.75]
        actual = pair_kl_passt(reference, evaluated)
        expected = sum(
            target * (math.log(target) - math.log(source + 1e-6))
            for source, target in zip(reference, evaluated, strict=True)
        )
        reverse = sum(
            source * (math.log(source) - math.log(target + 1e-6))
            for source, target in zip(reference, evaluated, strict=True)
        )
        self.assertAlmostEqual(actual, expected)
        self.assertNotAlmostEqual(actual, reverse)
        self.assertEqual(KL_DIRECTION, "KL(P_eval || P_reference_plus_1e-6)")

    def test_collect_mean_pools_logits_before_softmax(self) -> None:
        probabilities = collect_window_probabilities([[4.0, 0.0], [0.0, 0.0]], collect="mean")
        expected_first = math.exp(2.0) / (math.exp(2.0) + 1.0)
        mean_of_softmaxes = (math.exp(4.0) / (math.exp(4.0) + 1.0) + 0.5) / 2
        self.assertAlmostEqual(probabilities[0], expected_first)
        self.assertAlmostEqual(sum(probabilities), 1.0)
        self.assertNotAlmostEqual(probabilities[0], mean_of_softmaxes)

    def test_collect_max_pools_logits_before_softmax(self) -> None:
        probabilities = collect_window_probabilities([[4.0, 1.0], [0.0, 3.0]], collect="max")
        expected_first = math.exp(4.0) / (math.exp(4.0) + math.exp(3.0))
        self.assertAlmostEqual(probabilities[0], expected_first)
        with self.assertRaisesRegex(KLPaSSTValidationError, "mean.*max"):
            collect_window_probabilities([[1.0]], collect="median")

    def test_complete_records_are_ordered_and_averaged_like_official_corpus(self) -> None:
        result = validate_kl_passt_records(
            [
                {"sample_id": "b", "reference_id": "b", "generated_id": "b", "kl": 0.6},
                KLPaSSTPairRecord("a", "a", "a", 0.2),
            ],
            expected_ids=["a", "b"],
            official_corpus_score=0.4,
            collect="mean",
        )
        self.assertEqual([record.sample_id for record in result.records], ["a", "b"])
        self.assertEqual(result.corpus_mean, 0.4)
        self.assertEqual(result.pair_count, 2)
        self.assertEqual(result.expected_pair_count, 2)
        self.assertEqual(result.metadata["completed_pair_count"], 2)
        self.assertEqual(result.metadata["failed_pair_count"], 0)
        self.assertEqual(result.metadata["window_collect_semantics"], "mean over window logits, then softmax")

    def test_rejects_partial_and_all_failed_results(self) -> None:
        with self.assertRaisesRegex(KLPaSSTValidationError, "do not exactly match"):
            validate_kl_passt_records([{"sample_id": "a", "kl": 0.2}], expected_ids=["a", "b"])
        with self.assertRaisesRegex(KLPaSSTValidationError, "all.*failed"):
            validate_kl_passt_records([], expected_ids=["a"])
        with self.assertRaisesRegex(KLPaSSTValidationError, "failed"):
            validate_kl_passt_records(
                [{"sample_id": "a", "status": "error", "error": "decode failed"}],
                expected_ids=["a"],
            )

    def test_rejects_nonfinite_duplicate_extra_and_mismatched_ids(self) -> None:
        for invalid in (math.nan, math.inf, -math.inf):
            with self.subTest(invalid=invalid), self.assertRaisesRegex(KLPaSSTValidationError, "finite"):
                validate_kl_passt_records([{"sample_id": "a", "kl": invalid}], expected_ids=["a"])
        with self.assertRaisesRegex(KLPaSSTValidationError, "duplicate pair IDs"):
            validate_kl_passt_records(
                [{"sample_id": "a", "kl": 0.1}, {"sample_id": "a", "kl": 0.2}],
                expected_ids=["a", "b"],
            )
        with self.assertRaisesRegex(KLPaSSTValidationError, "extra=.*b"):
            validate_kl_passt_records([{"sample_id": "b", "kl": 0.1}], expected_ids=["a"])
        with self.assertRaisesRegex(KLPaSSTValidationError, "mismatched IDs"):
            validate_kl_passt_records(
                [{"sample_id": "a", "reference_id": "other", "generated_id": "a", "kl": 0.1}],
                expected_ids=["a"],
            )

    def test_rejects_bad_probabilities_and_official_mean_mismatch(self) -> None:
        with self.assertRaisesRegex(KLPaSSTValidationError, "dimensions differ"):
            pair_kl_passt([1.0], [0.5, 0.5])
        with self.assertRaisesRegex(KLPaSSTValidationError, "sum to 1"):
            pair_kl_passt([0.2, 0.2], [0.5, 0.5])
        with self.assertRaisesRegex(KLPaSSTValidationError, "does not match"):
            validate_kl_passt_records(
                [{"sample_id": "a", "kl": 0.2}],
                expected_ids=["a"],
                official_corpus_score=0.3,
            )

    def test_metadata_discloses_official_preprocessing_constraints(self) -> None:
        metadata = official_passt_preprocessing(collect="max")
        self.assertEqual(metadata["sample_rate_hz"], 32_000)
        self.assertIn("mono=True", metadata["channel_policy"])
        self.assertIn("-1.0 dBFS", metadata["amplitude_policy"])
        self.assertEqual(metadata["window_seconds"], 10)
        self.assertEqual(metadata["overlap_seconds"], 5)
        self.assertIn("15%", metadata["short_window_policy"])
        self.assertEqual(metadata["window_collect_semantics"], "max over window logits, then softmax")
        self.assertEqual(metadata["kl_direction"], KL_DIRECTION)
        self.assertTrue(metadata["requires_cuda"])
        self.assertIn("rejects", metadata["backend_partial_failure_behavior"])


if __name__ == "__main__":
    unittest.main()
