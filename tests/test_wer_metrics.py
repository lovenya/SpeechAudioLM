from __future__ import annotations

import unittest

from evaluation.metrics.wer import (
    CONSERVATIVE_ENGLISH_V1,
    RAW_ORTHOGRAPHIC_V1,
    EmptyReferenceError,
    WerCounts,
    corpus_wer,
    macro_utterance_wer,
    sum_wer_counts,
    utterance_wer_counts,
    word_edit_counts,
)


class WerMetricsTest(unittest.TestCase):
    def test_reports_explicit_edit_counts(self) -> None:
        counts = word_edit_counts(
            "start delete a b c old end".split(),
            "start a b c new end insert".split(),
        )

        self.assertEqual(counts.substitutions, 1)
        self.assertEqual(counts.deletions, 1)
        self.assertEqual(counts.insertions, 1)
        self.assertEqual(counts.reference_words, 7)
        self.assertEqual(counts.errors, 3)
        self.assertEqual(counts.wer, 3 / 7)

    def test_empty_hypothesis_is_all_deletions(self) -> None:
        counts = utterance_wer_counts("one two three", "")

        self.assertEqual(counts, WerCounts(0, 3, 0, 3))
        self.assertEqual(counts.wer, 1.0)

    def test_corpus_wer_sums_counts_instead_of_averaging_utterances(self) -> None:
        short = WerCounts(substitutions=1, deletions=0, insertions=0, reference_words=1)
        long = WerCounts(substitutions=1, deletions=0, insertions=0, reference_words=9)

        self.assertEqual(corpus_wer([short, long]), 0.2)
        self.assertAlmostEqual(macro_utterance_wer([short, long]), (1.0 + 1 / 9) / 2)

    def test_summed_counts_are_shard_composable(self) -> None:
        first = WerCounts(1, 2, 0, 10)
        second = WerCounts(0, 1, 3, 20)

        self.assertEqual(sum_wer_counts([first, second]), WerCounts(1, 3, 3, 30))

    def test_conservative_normalization_policy_is_declared_and_versioned(self) -> None:
        self.assertEqual(CONSERVATIVE_ENGLISH_V1.identifier, "english_conservative@1")
        self.assertEqual(
            CONSERVATIVE_ENGLISH_V1.normalize("  DON’T-stop,  Café 42! "),
            "don t stop café 42",
        )

    def test_conservative_policy_normalizes_before_scoring(self) -> None:
        counts = utterance_wer_counts("Hello—WORLD!", "hello world")

        self.assertEqual(counts, WerCounts(0, 0, 0, 2))

    def test_raw_orthographic_policy_preserves_case_and_punctuation(self) -> None:
        self.assertEqual(RAW_ORTHOGRAPHIC_V1.identifier, "raw_orthographic@1")
        counts = utterance_wer_counts("Hello, world", "hello world", RAW_ORTHOGRAPHIC_V1)

        self.assertEqual(counts, WerCounts(1, 0, 0, 2))

    def test_empty_reference_is_invalid_before_and_after_normalization(self) -> None:
        for reference in ("", " \t\n ", "..."):
            with self.subTest(reference=reference), self.assertRaises(EmptyReferenceError):
                utterance_wer_counts(reference, "hypothesis")

    def test_invalid_or_empty_aggregates_are_rejected(self) -> None:
        with self.assertRaises(EmptyReferenceError):
            WerCounts(0, 0, 0, 0)
        with self.assertRaises(ValueError):
            corpus_wer([])
        with self.assertRaises(ValueError):
            macro_utterance_wer([])


if __name__ == "__main__":
    unittest.main()
