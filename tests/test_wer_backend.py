from __future__ import annotations

import unittest
from unittest.mock import Mock

from evaluation.metrics.wer import EmptyReferenceError
from scripts.wer_backend import (
    aggregate_wer_scores,
    score_asr_wer_pairs,
    score_wer_pair,
    transcribe_whisper,
    whisper_transcribe_kwargs,
)


class WerBackendTest(unittest.TestCase):
    def test_sample_reports_normalized_and_raw_counts(self) -> None:
        score = score_wer_pair("Hello, world", "hello world extra")

        self.assertEqual(score["normalized"]["normalization_id"], "english_conservative@1")
        self.assertEqual(score["normalized"]["reference"], "hello world")
        self.assertEqual(score["normalized"]["hypothesis"], "hello world extra")
        self.assertEqual(score["normalized"]["insertions"], 1)
        self.assertEqual(score["normalized"]["errors"], 1)
        self.assertEqual(score["normalized"]["reference_words"], 2)
        self.assertEqual(score["normalized"]["wer"], 0.5)

        self.assertEqual(score["raw"]["normalization_id"], "raw_orthographic@1")
        self.assertEqual(score["raw"]["substitutions"], 1)
        self.assertEqual(score["raw"]["deletions"], 0)
        self.assertEqual(score["raw"]["insertions"], 1)
        self.assertEqual(score["raw"]["errors"], 2)
        self.assertEqual(score["raw"]["reference_words"], 2)
        self.assertEqual(score["raw"]["wer"], 1.0)

    def test_corpus_wer_uses_summed_counts_and_macro_is_secondary(self) -> None:
        samples = [
            score_wer_pair("one", "wrong"),
            score_wer_pair("one two three four five six seven eight nine", "one two three four five six seven eight wrong"),
        ]

        corpus = aggregate_wer_scores(samples)["normalized"]

        self.assertEqual(corpus["sample_count"], 2)
        self.assertEqual(corpus["substitutions"], 2)
        self.assertEqual(corpus["errors"], 2)
        self.assertEqual(corpus["reference_words"], 10)
        self.assertEqual(corpus["wer"], 0.2)
        self.assertAlmostEqual(corpus["macro_wer"], (1.0 + 1 / 9) / 2)

    def test_empty_reference_fails_strictly_for_both_policies(self) -> None:
        for reference in ("", "   ", "..."):
            with self.subTest(reference=reference), self.assertRaises(EmptyReferenceError):
                score_wer_pair(reference, "some words")

    def test_pairwise_asr_wer_preserves_direction_and_rejects_mismatch(self) -> None:
        result = score_asr_wer_pairs(["one two", "three"], ["one", "wrong"])

        self.assertEqual(
            result["semantics"],
            "WER(reference_transcript, ASR(generated_speech))",
        )
        self.assertEqual(result["samples"][0]["normalized"]["deletions"], 1)
        self.assertEqual(result["samples"][0]["normalized"]["insertions"], 0)
        self.assertEqual(result["corpus"]["normalized"]["wer"], 2 / 3)
        with self.assertRaisesRegex(ValueError, "equal reference and hypothesis counts"):
            score_asr_wer_pairs(["one"], [])

    def test_empty_pairwise_corpus_is_rejected(self) -> None:
        with self.assertRaisesRegex(ValueError, "empty corpus"):
            score_asr_wer_pairs([], [])

    def test_whisper_kwargs_fix_language_task_and_device_precision(self) -> None:
        self.assertEqual(
            whisper_transcribe_kwargs("cuda:2"),
            {"language": "en", "task": "transcribe", "fp16": True},
        )
        self.assertEqual(
            whisper_transcribe_kwargs("cpu"),
            {"language": "en", "task": "transcribe", "fp16": False},
        )

    def test_transcribe_whisper_applies_contract_and_strips_text(self) -> None:
        model = Mock()
        model.transcribe.return_value = {"text": "  hello world  "}

        transcript = transcribe_whisper(model, "/audio/generated.wav", "cuda")

        self.assertEqual(transcript, "hello world")
        model.transcribe.assert_called_once_with(
            "/audio/generated.wav",
            language="en",
            task="transcribe",
            fp16=True,
        )


if __name__ == "__main__":
    unittest.main()
