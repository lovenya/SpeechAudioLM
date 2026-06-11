from __future__ import annotations

import math
import unittest

from evaluation.metrics.aac_audio import (
    AACAudioValidationError,
    ms_clap_2023_backend_behavior,
    reconstruct_combined_clap_samples,
    reconstruct_mace_samples,
    validate_clap_audio_samples,
)


class AACAudioReconstructionTest(unittest.TestCase):
    def test_reconstructs_mace_samples_from_components_not_backend_mace_scalar(self) -> None:
        result = reconstruct_mace_samples(
            {
                "clap_sim": [0.8, 0.5, -0.2],
                "fer": [0.0, 1.0, 1.0],
                "mace": 0.31666666666666665,
            },
            expected_count=3,
            penalty=0.3,
            corpus_score=(0.8 + 0.35 - 0.14) / 3,
        )

        for actual, expected in zip(result.scores, (0.8, 0.35, -0.14), strict=True):
            self.assertAlmostEqual(actual, expected)
        self.assertAlmostEqual(result.corpus_mean, (0.8 + 0.35 - 0.14) / 3)
        self.assertIn("corpus scalar", result.metadata["backend_sentence_mace_policy"])

    def test_combined_clap_is_reconstructed_elementwise(self) -> None:
        result = reconstruct_combined_clap_samples(
            [0.9, 0.1],
            [0.3, 0.7],
            expected_count=2,
            corpus_score=0.5,
        )

        self.assertEqual(result.scores, (0.6, 0.39999999999999997))
        self.assertEqual(result.corpus_mean, 0.5)
        self.assertIn("elementwise", result.metadata["sample_reconstruction"])

    def test_clap_audio_preserves_complete_official_sample_vector(self) -> None:
        result = validate_clap_audio_samples(
            {"clap_sim": [0.25, -0.5]},
            expected_count=2,
            corpus_score=-0.125,
            seed=7,
        )

        self.assertEqual(result.scores, (0.25, -0.5))
        self.assertEqual(result.metadata["crop_seed"], 7)
        self.assertFalse(result.metadata["order_invariant"])
        self.assertTrue(result.metadata["crop_offset_stream_internal_batch_size_invariant"])
        self.assertEqual(result.metadata["model_score_internal_batch_size_invariant"], "not_claimed")
        self.assertFalse(result.metadata["separate_invocation_partition_invariant"])
        self.assertTrue(result.metadata["scientifically_approved_project_policy"])

    def test_rejects_scalar_or_wrong_length_instead_of_duplication_or_truncation(self) -> None:
        with self.assertRaisesRegex(AACAudioValidationError, "one-dimensional sequence"):
            validate_clap_audio_samples(0.5, expected_count=2)
        with self.assertRaisesRegex(AACAudioValidationError, "refusing duplication or truncation"):
            validate_clap_audio_samples([0.1, 0.2, 0.3], expected_count=2)
        with self.assertRaisesRegex(AACAudioValidationError, "refusing duplication or truncation"):
            reconstruct_combined_clap_samples([0.1], [0.2, 0.3], expected_count=2)

    def test_rejects_nonfinite_out_of_range_and_nonbinary_components(self) -> None:
        for invalid in (math.nan, math.inf, -math.inf):
            with self.subTest(invalid=invalid), self.assertRaisesRegex(AACAudioValidationError, "finite"):
                validate_clap_audio_samples([invalid], expected_count=1)
        with self.assertRaisesRegex(AACAudioValidationError, "above 1.0"):
            validate_clap_audio_samples([1.01], expected_count=1)
        with self.assertRaisesRegex(AACAudioValidationError, "expected one of"):
            reconstruct_mace_samples(
                {"clap_sim": [0.5], "fer": [0.5]},
                expected_count=1,
            )
        with self.assertRaisesRegex(AACAudioValidationError, "real numeric scalar"):
            reconstruct_mace_samples(
                {"clap_sim": [True], "fer": [0.0]},
                expected_count=1,
            )

    def test_rejects_invalid_penalty_and_corpus_mean_mismatch(self) -> None:
        with self.assertRaisesRegex(AACAudioValidationError, "penalty must be in"):
            reconstruct_mace_samples(
                {"clap_sim": [0.5], "fer": [0.0]},
                expected_count=1,
                penalty=1.1,
            )
        with self.assertRaisesRegex(AACAudioValidationError, "does not match"):
            reconstruct_mace_samples(
                {"clap_sim": [0.5], "fer": [0.0]},
                expected_count=1,
                corpus_score=0.4,
            )

    def test_policy_reports_upstream_audio_constraints(self) -> None:
        policy = ms_clap_2023_backend_behavior(seed=1337)

        self.assertEqual(policy["sample_rate_hz"], 44_100)
        self.assertEqual(policy["clip_duration_seconds"], 7)
        self.assertIn("repeat", policy["short_audio_policy"])
        self.assertIn("random.randrange", policy["long_audio_policy"])
        self.assertEqual(policy["crop_seed"], 1337)
        self.assertEqual(policy["text_context_tokens"], 77)
        self.assertEqual(policy["text_overflow_policy"], "no explicit tokenizer truncation in msclap 1.3.4")
        self.assertFalse(policy["crop_offsets_recorded"])
        self.assertIn("accepted_and_disclosed", policy["project_policy_status"])


if __name__ == "__main__":
    unittest.main()
