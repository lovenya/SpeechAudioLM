from __future__ import annotations

import unittest

from evaluation.metrics.aac_preprocessing import (
    AAC_CAPTION_PREPROCESSING_POLICY_VERSION,
    AAC_METRICS_PACKAGE_VERSION,
    SpiderFlCandidate,
    caption_preprocessing_metadata,
    fense_raw_candidates,
    prepare_rouge_l_tokens,
    prepare_spice_caption,
    prepare_spider_fl_candidate,
    rouge_l_tokenizer,
    spider_sentences,
)


class SpiceCaptionPreprocessingTest(unittest.TestCase):
    def test_only_collapses_whitespace_and_preserves_complete_caption(self) -> None:
        caption = "  Dogs' BARK!\n\tThen 100 birds fly.  "

        prepared = prepare_spice_caption(caption)

        self.assertEqual(prepared, "Dogs' BARK! Then 100 birds fly.")
        self.assertEqual(len(prepared.split()), 6)

    def test_empty_and_whitespace_only_captions_remain_explicitly_empty(self) -> None:
        self.assertEqual(prepare_spice_caption(""), "")
        self.assertEqual(prepare_spice_caption(" \n\t "), "")

    def test_rejects_non_text_instead_of_stringifying_it(self) -> None:
        for value in (None, 0, ["caption"]):
            with self.subTest(value=value), self.assertRaises(TypeError):
                prepare_spice_caption(value)  # type: ignore[arg-type]


class RougeLCaptionPreprocessingTest(unittest.TestCase):
    def test_aac_metrics_055_policy_is_whitespace_tokenization_without_stemming(self) -> None:
        self.assertEqual(
            prepare_rouge_l_tokens("dogs running dog's"),
            ("dogs", "running", "dog's"),
        )

    def test_optional_stemmer_runs_after_tokenization_once_per_token(self) -> None:
        seen: list[str] = []

        def stem(token: str) -> str:
            seen.append(token)
            return token.removesuffix("ing")

        self.assertEqual(
            prepare_rouge_l_tokens(" running\t ringing ", stemmer=stem),
            ("runn", "ring"),
        )
        self.assertEqual(seen, ["running", "ringing"])

    def test_tokenizer_adapter_returns_the_list_shape_expected_by_backend(self) -> None:
        tokenizer = rouge_l_tokenizer()

        self.assertEqual(tokenizer("one  two\tthree"), ["one", "two", "three"])

    def test_invalid_stemmer_and_invalid_stem_output_fail(self) -> None:
        with self.assertRaises(TypeError):
            rouge_l_tokenizer(stemmer="porter")  # type: ignore[arg-type]
        with self.assertRaises(TypeError):
            prepare_rouge_l_tokens("word", stemmer=lambda _token: None)  # type: ignore[arg-type,return-value]
        with self.assertRaises(TypeError):
            prepare_rouge_l_tokens(None)  # type: ignore[arg-type]


class SpiderFlCandidateRepresentationTest(unittest.TestCase):
    def test_fense_receives_raw_caption_not_ptb_sentence_or_sentence_fragments(self) -> None:
        raw = "First sentence!\nSecond sentence?"
        candidate = prepare_spider_fl_candidate(
            raw_text=raw,
            spider_sentence="first sentence second sentence",
        )

        self.assertEqual(spider_sentences([candidate]), ["first sentence second sentence"])
        self.assertEqual(fense_raw_candidates([candidate]), [raw])
        self.assertEqual(len(fense_raw_candidates([candidate])), 1)

    def test_representation_preserves_empty_and_whitespace_raw_text_verbatim(self) -> None:
        candidate = prepare_spider_fl_candidate(" \n ", "")

        self.assertEqual(candidate.fense_raw_text, " \n ")
        self.assertEqual(candidate.spider_sentence, "")

    def test_rejects_wrong_field_and_collection_types(self) -> None:
        with self.assertRaises(TypeError):
            SpiderFlCandidate(spider_sentence=None, fense_raw_text="raw")  # type: ignore[arg-type]
        with self.assertRaises(TypeError):
            prepare_spider_fl_candidate(raw_text="raw", spider_sentence=1)  # type: ignore[arg-type]
        with self.assertRaises(TypeError):
            fense_raw_candidates(["raw caption"])  # type: ignore[list-item]


class CaptionPreprocessingMetadataTest(unittest.TestCase):
    def test_policy_is_versioned_and_records_metric_specific_semantics(self) -> None:
        metadata = caption_preprocessing_metadata()

        self.assertEqual(AAC_METRICS_PACKAGE_VERSION, "0.5.5")
        self.assertEqual(metadata["policy_version"], AAC_CAPTION_PREPROCESSING_POLICY_VERSION)
        self.assertEqual(metadata["aac_metrics_package_version"], "0.5.5")
        self.assertEqual(metadata["rouge_l"]["stemming"], "none")  # type: ignore[index]
        self.assertEqual(metadata["spice"]["truncation"], "none")  # type: ignore[index]
        self.assertEqual(metadata["spider_fl"]["fense_candidate"], "raw_stored_caption")  # type: ignore[index]
        self.assertEqual(metadata["spider_fl"]["fer_tokenizer_max_length"], 64)  # type: ignore[index]
        self.assertEqual(  # type: ignore[index]
            metadata["spider_fl"]["fer_text_normalization"],
            "lowercase_and_remove_punctuation_before_tokenization",
        )
        self.assertTrue(metadata["spider_fl"]["fer_tokenizer_truncation"])  # type: ignore[index]
        self.assertFalse(metadata["spider_fl"]["effective_complete_caption_evaluation"])  # type: ignore[index]

    def test_metadata_callers_cannot_mutate_future_results(self) -> None:
        first = caption_preprocessing_metadata()
        first["policy"] = "changed"

        self.assertEqual(
            caption_preprocessing_metadata()["policy"],
            "speech_audio_lm_aac_caption_preprocessing",
        )


if __name__ == "__main__":
    unittest.main()
