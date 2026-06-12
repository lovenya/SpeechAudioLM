from __future__ import annotations

import math
import unittest

from evaluation.metrics.utmos_policy import (
    UTMOSV2_CHECKPOINT_FILENAME,
    UTMOSV2_NUM_REPETITIONS,
    UTMOSRecord,
    UTMOSValidationError,
    official_utmosv2_policy,
    scientific_prediction_kwargs,
    validate_prediction_policy,
    validate_utmos_records,
)


class UTMOSPolicyTest(unittest.TestCase):
    def test_discloses_official_audio_and_segmentation_behavior(self) -> None:
        policy = official_utmosv2_policy()

        self.assertEqual(policy["sample_rate_hz"], 16_000)
        self.assertIn("mono=True", policy["channel_policy"])
        self.assertIn("librosa.resample", policy["resampling_policy"])
        self.assertEqual(policy["amplitude_normalization"], "none")
        self.assertFalse(policy["silence_removal"])
        self.assertIn("paper experiments", policy["silence_removal_policy"])
        self.assertEqual(policy["ssl_segment_seconds"], 3.0)
        self.assertIn("tile", policy["ssl_segment_policy"])
        self.assertEqual(policy["spectrogram_frame_seconds"], 1.4)
        self.assertEqual(policy["spectrogram_frame_count"], 2)
        self.assertTrue(policy["spectrogram_inner_mixup"])
        self.assertIn("Beta(0.4, 0.4)", policy["spectrogram_inner_mixup_policy"])
        self.assertIn("random", policy["crop_policy"])
        self.assertIn("no zero-padding", policy["padding_policy"])

    def test_discloses_official_checkpoint_and_repetition_policy(self) -> None:
        policy = official_utmosv2_policy()
        checkpoint = policy["checkpoint"]

        self.assertEqual(policy["implementation_version"], "1.3.1.dev0")
        self.assertEqual(policy["config"], "fusion_stage3")
        self.assertEqual(checkpoint["fold"], 0)
        self.assertEqual(checkpoint["training_seed"], 42)
        self.assertEqual(checkpoint["filename"], UTMOSV2_CHECKPOINT_FILENAME)
        self.assertFalse(checkpoint["custom_checkpoint_allowed"])
        self.assertEqual(policy["num_repetitions"], UTMOSV2_NUM_REPETITIONS)
        self.assertIn("five", policy["repetition_aggregation"])
        self.assertIn("NumPy RNG seed", policy["random_seed_policy"])

    def test_scientific_kwargs_disable_post_release_silence_removal(self) -> None:
        self.assertEqual(
            scientific_prediction_kwargs(),
            {
                "predict_dataset": "sarulab",
                "num_repetitions": 5,
                "remove_silent_section": False,
            },
        )

    def test_policy_validator_accepts_only_selected_official_configuration(self) -> None:
        validate_prediction_policy(
            config="fusion_stage3",
            fold=0,
            checkpoint_seed=42,
            pretrained=True,
            checkpoint_path=None,
            predict_dataset="sarulab",
            num_repetitions=5,
            remove_silent_section=False,
        )

        invalid_settings = {
            "config": "spec_only",
            "fold": 1,
            "checkpoint_seed": 7,
            "pretrained": False,
            "checkpoint_path": "/tmp/custom.pth",
            "predict_dataset": "bvcc",
            "num_repetitions": 1,
            "remove_silent_section": True,
        }
        for name, invalid in invalid_settings.items():
            kwargs = {
                "config": "fusion_stage3",
                "fold": 0,
                "checkpoint_seed": 42,
                "pretrained": True,
                "checkpoint_path": None,
                "predict_dataset": "sarulab",
                "num_repetitions": 5,
                "remove_silent_section": False,
            }
            kwargs[name] = invalid
            with self.subTest(name=name), self.assertRaisesRegex(
                UTMOSValidationError, name
            ):
                validate_prediction_policy(**kwargs)


class UTMOSValidationTest(unittest.TestCase):
    def test_complete_records_are_ordered_and_averaged(self) -> None:
        result = validate_utmos_records(
            [
                {"sample_id": "b", "utmos": 3.8},
                UTMOSRecord("a", 4.2),
            ],
            expected_ids=["a", "b"],
            official_corpus_score=4.0,
        )

        self.assertEqual([record.sample_id for record in result.records], ["a", "b"])
        self.assertEqual(result.corpus_mean, 4.0)
        self.assertEqual(result.sample_count, 2)
        self.assertEqual(result.expected_sample_count, 2)
        self.assertEqual(result.metadata["completed_sample_count"], 2)
        self.assertEqual(result.metadata["failed_sample_count"], 0)
        self.assertFalse(result.metadata["partial_results_accepted"])

    def test_rejects_partial_empty_failed_duplicate_and_extra_results(self) -> None:
        with self.assertRaisesRegex(UTMOSValidationError, "do not exactly match"):
            validate_utmos_records([{"sample_id": "a", "utmos": 4.0}], expected_ids=["a", "b"])
        with self.assertRaisesRegex(UTMOSValidationError, "all UTMOS samples failed"):
            validate_utmos_records([], expected_ids=["a"])
        with self.assertRaisesRegex(UTMOSValidationError, "failed"):
            validate_utmos_records(
                [{"sample_id": "a", "status": "error", "error": "decode failed"}],
                expected_ids=["a"],
            )
        with self.assertRaisesRegex(UTMOSValidationError, "duplicate UTMOS sample IDs"):
            validate_utmos_records(
                [
                    {"sample_id": "a", "utmos": 4.0},
                    {"sample_id": "a", "utmos": 3.0},
                ],
                expected_ids=["a", "b"],
            )
        with self.assertRaisesRegex(UTMOSValidationError, "extra=.*b"):
            validate_utmos_records([{"sample_id": "b", "utmos": 4.0}], expected_ids=["a"])

    def test_rejects_nonfinite_missing_and_nonnumeric_values(self) -> None:
        for invalid in (math.nan, math.inf, -math.inf):
            with self.subTest(invalid=invalid), self.assertRaisesRegex(
                UTMOSValidationError, "finite"
            ):
                validate_utmos_records(
                    [{"sample_id": "a", "utmos": invalid}],
                    expected_ids=["a"],
                )
        with self.assertRaisesRegex(UTMOSValidationError, "real numeric scalar"):
            validate_utmos_records([{"sample_id": "a"}], expected_ids=["a"])
        with self.assertRaisesRegex(UTMOSValidationError, "real numeric scalar"):
            validate_utmos_records(
                [{"sample_id": "a", "utmos": True}],
                expected_ids=["a"],
            )

    def test_rejects_duplicate_expected_ids_and_corpus_mean_mismatch(self) -> None:
        with self.assertRaisesRegex(UTMOSValidationError, "duplicate IDs"):
            validate_utmos_records(
                [{"sample_id": "a", "utmos": 4.0}],
                expected_ids=["a", "a"],
            )
        with self.assertRaisesRegex(UTMOSValidationError, "does not match"):
            validate_utmos_records(
                [{"sample_id": "a", "utmos": 4.0}],
                expected_ids=["a"],
                official_corpus_score=3.9,
            )


if __name__ == "__main__":
    unittest.main()
