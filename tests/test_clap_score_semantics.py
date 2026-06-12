from __future__ import annotations

import math
import unittest

from evaluation.metrics.clap_score import (
    LAION_CLAP_CHECKPOINT,
    PROMPT_ADHERENCE,
    PROMPT_ADHERENCE_CLAP_SCORE,
    TARGET_CAPTION_ALIGNMENT,
    TARGET_CAPTION_CLAP_SCORE,
    CLAPScoreRecord,
    CLAPScoreValidationError,
    official_clap_score_constraints,
    prepare_clap_score_inputs,
    select_clap_score_text,
    validate_clap_score_records,
)


class CLAPScoreSemanticsTest(unittest.TestCase):
    def test_target_caption_and_prompt_adherence_select_different_text(self) -> None:
        record = {
            "target_caption": "A dog barks near traffic.",
            "conditioning_prompt": "Create two short barks followed by a car horn.",
        }

        self.assertEqual(
            select_clap_score_text(record, semantics=TARGET_CAPTION_ALIGNMENT),
            record["target_caption"],
        )
        self.assertEqual(
            select_clap_score_text(record, semantics=PROMPT_ADHERENCE),
            record["conditioning_prompt"],
        )

    def test_prompt_adherence_never_falls_back_to_target_caption(self) -> None:
        with self.assertRaisesRegex(
            CLAPScoreValidationError, "conditioning_prompt.*fallback.*forbidden"
        ):
            select_clap_score_text(
                {"target_caption": "A usable target caption."},
                semantics=PROMPT_ADHERENCE,
            )
        with self.assertRaisesRegex(CLAPScoreValidationError, "non-empty"):
            select_clap_score_text(
                {"conditioning_prompt": "   "},
                semantics=PROMPT_ADHERENCE,
            )

    def test_prepared_inputs_are_complete_ordered_and_keep_exact_text(self) -> None:
        result = prepare_clap_score_inputs(
            [
                {
                    "sample_id": "b",
                    "generated_id": "b",
                    "conditioning_prompt": "  Preserve exact whitespace.  ",
                },
                {
                    "sample_id": "a",
                    "generated_id": "a",
                    "conditioning_prompt": "First prompt",
                },
            ],
            expected_ids=["a", "b"],
            semantics=PROMPT_ADHERENCE,
        )

        self.assertEqual([item.sample_id for item in result], ["a", "b"])
        self.assertEqual(result[1].text, "  Preserve exact whitespace.  ")
        self.assertTrue(all(item.semantics == PROMPT_ADHERENCE for item in result))

    def test_validated_scores_preserve_semantics_and_complete_mean(self) -> None:
        result = validate_clap_score_records(
            [
                {
                    "sample_id": "b",
                    "generated_id": "b",
                    "semantics": TARGET_CAPTION_ALIGNMENT,
                    "text": "Caption B",
                    "score": 0.6,
                },
                CLAPScoreRecord(
                    "a", "a", TARGET_CAPTION_ALIGNMENT, "Caption A", 0.2
                ),
            ],
            expected_ids=["a", "b"],
            semantics=TARGET_CAPTION_ALIGNMENT,
            official_corpus_score=0.4,
        )

        self.assertEqual([record.sample_id for record in result.records], ["a", "b"])
        self.assertEqual(result.corpus_mean, 0.4)
        self.assertEqual(result.metadata["metric"], TARGET_CAPTION_CLAP_SCORE)
        self.assertEqual(result.metadata["selected_text_field"], "target_caption")
        self.assertFalse(result.metadata["target_caption_is_prompt_adherence"])

    def test_prompt_score_uses_distinct_metric_name(self) -> None:
        result = validate_clap_score_records(
            [
                {
                    "sample_id": "a",
                    "generated_id": "a",
                    "semantics": PROMPT_ADHERENCE,
                    "text": "The exact generation prompt",
                    "score": 0.5,
                }
            ],
            expected_ids=["a"],
            semantics=PROMPT_ADHERENCE,
        )
        self.assertEqual(result.metadata["metric"], PROMPT_ADHERENCE_CLAP_SCORE)
        self.assertEqual(result.metadata["selected_text_field"], "conditioning_prompt")

    def test_partial_and_all_failed_records_are_attributed(self) -> None:
        passed = {
            "sample_id": "a",
            "generated_id": "a",
            "semantics": TARGET_CAPTION_ALIGNMENT,
            "text": "Caption",
            "score": 0.2,
        }
        partial = validate_clap_score_records(
            [passed, {"sample_id": "b", "status": "error", "error": "decode failed"}],
            expected_ids=["a", "b"],
            semantics=TARGET_CAPTION_ALIGNMENT,
            official_corpus_score=0.2,
        )
        self.assertEqual(partial.corpus_mean, 0.2)
        self.assertEqual(partial.metadata["status"], "passed_with_errors")
        self.assertEqual(partial.metadata["attempted_count"], 2)
        self.assertEqual(partial.metadata["passed_count"], 1)
        self.assertEqual(partial.metadata["error_count"], 1)
        self.assertEqual(partial.metadata["error_sample_ids"], ("b",))

        failed = validate_clap_score_records(
            [{"sample_id": "a", "status": "error", "error": "decode failed"}],
            expected_ids=["a"],
            semantics=TARGET_CAPTION_ALIGNMENT,
        )
        self.assertIsNone(failed.corpus_mean)
        self.assertEqual(failed.records, ())
        self.assertEqual(failed.metadata["status"], "error")

        with self.assertRaisesRegex(CLAPScoreValidationError, "do not exactly match"):
            validate_clap_score_records([passed], expected_ids=["a", "b"], semantics=TARGET_CAPTION_ALIGNMENT)
        with self.assertRaisesRegex(CLAPScoreValidationError, "duplicate sample IDs"):
            validate_clap_score_records([passed, passed], expected_ids=["a", "b"], semantics=TARGET_CAPTION_ALIGNMENT)
        with self.assertRaisesRegex(CLAPScoreValidationError, "mismatched IDs"):
            validate_clap_score_records(
                [{"sample_id": "a", "generated_id": "other", "status": "error", "error": "failed"}],
                expected_ids=["a"],
                semantics=TARGET_CAPTION_ALIGNMENT,
            )

    def test_rejects_nonfinite_out_of_range_and_wrong_semantics(self) -> None:
        base = {
            "sample_id": "a",
            "generated_id": "a",
            "semantics": TARGET_CAPTION_ALIGNMENT,
            "text": "Caption",
            "score": 0.2,
        }
        for invalid in (math.nan, math.inf, -math.inf):
            with self.subTest(invalid=invalid), self.assertRaisesRegex(
                CLAPScoreValidationError, "finite"
            ):
                validate_clap_score_records(
                    [base | {"score": invalid}],
                    expected_ids=["a"],
                    semantics=TARGET_CAPTION_ALIGNMENT,
                )
        for invalid in (-1.01, 1.01):
            with self.subTest(invalid=invalid), self.assertRaisesRegex(
                CLAPScoreValidationError, r"\[-1, 1\]"
            ):
                validate_clap_score_records(
                    [base | {"score": invalid}],
                    expected_ids=["a"],
                    semantics=TARGET_CAPTION_ALIGNMENT,
                )
        with self.assertRaisesRegex(CLAPScoreValidationError, "does not match"):
            validate_clap_score_records(
                [base],
                expected_ids=["a"],
                semantics=PROMPT_ADHERENCE,
            )

    def test_metadata_discloses_verified_vendor_and_adapter_constraints(self) -> None:
        metadata = official_clap_score_constraints()

        self.assertEqual(metadata["checkpoint"], LAION_CLAP_CHECKPOINT)
        self.assertTrue(metadata["checkpoint_fusion_enabled"])
        self.assertEqual(metadata["text_encoder"], "roberta-base")
        self.assertEqual(metadata["text_context_tokens"], 77)
        self.assertTrue(metadata["text_truncation"])
        self.assertIn("batch dimension", metadata["text_tokenizer_adapter"])
        self.assertEqual(metadata["sample_rate_hz"], 48_000)
        self.assertIn("mono=True", metadata["channel_policy"])
        self.assertIn("sr=48000", metadata["resampling_policy"])
        self.assertIn("-1.0 dBFS", metadata["amplitude_policy"])
        self.assertIn("int16", metadata["pcm_roundtrip"])
        self.assertIn("32767", metadata["pcm_roundtrip"])
        self.assertTrue(metadata["partial_results_accepted"])


if __name__ == "__main__":
    unittest.main()
