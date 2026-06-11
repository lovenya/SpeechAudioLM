"""Versioned caption views for AAC metric backends.

These helpers make preprocessing boundaries explicit without reimplementing
the Stanford PTB tokenizer or model tokenizers owned by ``aac-metrics``.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Iterable


AAC_METRICS_PACKAGE_VERSION = "0.5.5"
AAC_CAPTION_PREPROCESSING_POLICY_VERSION = "1"


def _require_text(value: object, name: str) -> str:
    if not isinstance(value, str):
        raise TypeError(f"{name} must be a string")
    return value


def prepare_spice_caption(caption: str) -> str:
    """Return caption text accepted by the package PTB preprocessing stage.

    ``aac_metrics.utils.tokenization.ptb_tokenize_batch`` rejects embedded
    newlines because one input caption must occupy one line. Collapsing Unicode
    whitespace is the only local transformation. Case, punctuation, and words
    are preserved, but exact whitespace and character length are not.
    """

    caption = _require_text(caption, "caption")
    return " ".join(caption.split())


def prepare_rouge_l_tokens(
    ptb_preprocessed_caption: str,
    *,
    stemmer: Callable[[str], str] | None = None,
) -> tuple[str, ...]:
    """Tokenize a PTB-preprocessed caption for AAC-metrics ROUGE-L.

    AAC-metrics 0.5.5 uses ``str.split`` in ROUGE-L after its optional global
    PTB preprocessing and does not stem. The default therefore returns exact
    whitespace tokens with no case folding, punctuation removal, or stemming.
    A stemmer is accepted only to make any non-package comparison policy
    explicit; it is applied independently after tokenization.
    """

    caption = _require_text(ptb_preprocessed_caption, "ptb_preprocessed_caption")
    tokens = tuple(caption.split())
    if stemmer is None:
        return tokens
    if not callable(stemmer):
        raise TypeError("stemmer must be callable or None")

    stemmed: list[str] = []
    for token in tokens:
        stem = stemmer(token)
        if not isinstance(stem, str):
            raise TypeError("stemmer must return a string for every token")
        stemmed.append(stem)
    return tuple(stemmed)


def rouge_l_tokenizer(
    *,
    stemmer: Callable[[str], str] | None = None,
) -> Callable[[str], list[str]]:
    """Build the tokenizer callable expected by ``aac_metrics.rouge_l``."""

    if stemmer is not None and not callable(stemmer):
        raise TypeError("stemmer must be callable or None")

    def tokenize(caption: str) -> list[str]:
        return list(prepare_rouge_l_tokens(caption, stemmer=stemmer))

    return tokenize


@dataclass(frozen=True)
class SpiderFlCandidate:
    """Separate text views required by a scientifically explicit SPIDEr-FL.

    ``spider_sentence`` is the PTB-preprocessed sentence consumed by CIDEr-D
    and SPICE. ``fense_raw_text`` is the stored candidate passed into FENSE.
    FENSE then applies its own embedding and fluency-checker preprocessing;
    preserving this input view does not imply complete-caption evaluation.
    AAC-metrics 0.5.5's high-level ``evaluate(preprocess=True)`` cannot preserve
    this distinction because it passes one PTB-preprocessed string to both.
    """

    spider_sentence: str
    fense_raw_text: str

    def __post_init__(self) -> None:
        _require_text(self.spider_sentence, "spider_sentence")
        _require_text(self.fense_raw_text, "fense_raw_text")


def prepare_spider_fl_candidate(
    raw_text: str,
    spider_sentence: str,
) -> SpiderFlCandidate:
    """Construct SPIDEr and FENSE views without normalizing the FENSE text."""

    return SpiderFlCandidate(
        spider_sentence=_require_text(spider_sentence, "spider_sentence"),
        fense_raw_text=_require_text(raw_text, "raw_text"),
    )


def spider_sentences(candidates: Iterable[SpiderFlCandidate]) -> list[str]:
    """Extract the package-tokenized candidate view for SPIDEr."""

    return [_require_candidate(candidate).spider_sentence for candidate in candidates]


def fense_raw_candidates(candidates: Iterable[SpiderFlCandidate]) -> list[str]:
    """Extract stored captions passed to FENSE, one string per sample."""

    return [_require_candidate(candidate).fense_raw_text for candidate in candidates]


def _require_candidate(candidate: object) -> SpiderFlCandidate:
    if not isinstance(candidate, SpiderFlCandidate):
        raise TypeError("candidates must contain SpiderFlCandidate values")
    return candidate


def caption_preprocessing_metadata() -> dict[str, object]:
    """Return JSON-serializable provenance for these caption policies."""

    return {
        "policy": "speech_audio_lm_aac_caption_preprocessing",
        "policy_version": AAC_CAPTION_PREPROCESSING_POLICY_VERSION,
        "aac_metrics_package_version": AAC_METRICS_PACKAGE_VERSION,
        "spice": {
            "local_input_policy": "unicode_whitespace_to_single_ascii_spaces",
            "exact_stored_text_preserved": False,
            "backend_preprocessing": "stanford_ptb_tokenizer_3.4.1_lowercase_remove_ptb_punctuation",
            "truncation": "none",
        },
        "rouge_l": {
            "input": "ptb_preprocessed_caption",
            "tokenizer": "python_str_split",
            "stemming": "none",
            "beta": 1.2,
        },
        "spider_fl": {
            "spider_candidate": "ptb_preprocessed_sentence",
            "fense_candidate": "raw_stored_caption",
            "fense_sentence_segmentation": "none_one_caption_string_per_sample",
            "raw_text_reaches_fense_api": True,
            "fer_text_normalization": "lowercase_and_remove_punctuation_before_tokenization",
            "fer_tokenizer_max_length": 64,
            "fer_tokenizer_truncation": True,
            "sentence_embedding_context": "finite_backend_model_context",
            "effective_complete_caption_evaluation": False,
            "truncation_coverage_reporting": "required_before_final_scientific_run",
        },
    }
