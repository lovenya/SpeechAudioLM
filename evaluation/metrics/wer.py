"""Scientifically explicit word-error-rate primitives.

The primary WER statistic is corpus WER: summed edit errors divided by summed
reference words. Macro utterance WER is exposed separately because it weights
short and long utterances equally and is not standard corpus WER.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
import unicodedata
from typing import Iterable, Sequence


class EmptyReferenceError(ValueError):
    """Raised when a reference has no words under the selected policy."""


class NormalizationPolicy(ABC):
    """A named, versioned text-normalization contract."""

    name: str
    version: str

    @property
    def identifier(self) -> str:
        return f"{self.name}@{self.version}"

    @abstractmethod
    def normalize(self, text: str) -> str:
        """Return text ready for whitespace tokenization."""


class ConservativeEnglishNormalizationV1(NormalizationPolicy):
    """Dependency-free English normalization without lexical rewriting.

    The policy applies Unicode NFKC normalization, Unicode case folding,
    replaces every Unicode punctuation character with a space, and collapses
    whitespace. It intentionally does not expand numbers, abbreviations, or
    contractions and does not remove non-punctuation symbols.
    """

    name = "english_conservative"
    version = "1"

    def normalize(self, text: str) -> str:
        normalized = unicodedata.normalize("NFKC", text).casefold()
        without_punctuation = "".join(
            " " if unicodedata.category(character).startswith("P") else character
            for character in normalized
        )
        return " ".join(without_punctuation.split())


class RawOrthographicNormalizationV1(NormalizationPolicy):
    """Preserve orthographic tokens exactly, apart from whitespace layout."""

    name = "raw_orthographic"
    version = "1"

    def normalize(self, text: str) -> str:
        return " ".join(text.split())


CONSERVATIVE_ENGLISH_V1 = ConservativeEnglishNormalizationV1()
RAW_ORTHOGRAPHIC_V1 = RawOrthographicNormalizationV1()


@dataclass(frozen=True)
class WerCounts:
    """Minimum word-edit counts for one or more valid references."""

    substitutions: int
    deletions: int
    insertions: int
    reference_words: int

    def __post_init__(self) -> None:
        values = (
            self.substitutions,
            self.deletions,
            self.insertions,
            self.reference_words,
        )
        if any(isinstance(value, bool) or not isinstance(value, int) for value in values):
            raise TypeError("WER counts must be integers")
        if any(value < 0 for value in values):
            raise ValueError("WER counts must be non-negative")
        if self.reference_words == 0:
            raise EmptyReferenceError("WER is undefined for an empty reference")

    @property
    def errors(self) -> int:
        return self.substitutions + self.deletions + self.insertions

    @property
    def wer(self) -> float:
        return self.errors / self.reference_words


def word_edit_counts(reference_words: Sequence[str], hypothesis_words: Sequence[str]) -> WerCounts:
    """Return deterministic counts from a minimum-word-edit alignment.

    Multiple minimum-distance alignments can have different operation
    breakdowns. Ties here prefer substitution, then deletion, then insertion;
    total errors and WER are invariant to that tie policy.
    """

    if not reference_words:
        raise EmptyReferenceError("WER is undefined for an empty reference")

    rows = len(reference_words) + 1
    columns = len(hypothesis_words) + 1
    distances = [[0] * columns for _ in range(rows)]
    for row in range(rows):
        distances[row][0] = row
    for column in range(columns):
        distances[0][column] = column

    for row in range(1, rows):
        for column in range(1, columns):
            substitution_cost = int(reference_words[row - 1] != hypothesis_words[column - 1])
            distances[row][column] = min(
                distances[row - 1][column - 1] + substitution_cost,
                distances[row - 1][column] + 1,
                distances[row][column - 1] + 1,
            )

    substitutions = deletions = insertions = 0
    row = len(reference_words)
    column = len(hypothesis_words)
    while row or column:
        if (
            row
            and column
            and reference_words[row - 1] == hypothesis_words[column - 1]
            and distances[row][column] == distances[row - 1][column - 1]
        ):
            row -= 1
            column -= 1
        elif row and column and distances[row][column] == distances[row - 1][column - 1] + 1:
            substitutions += 1
            row -= 1
            column -= 1
        elif row and distances[row][column] == distances[row - 1][column] + 1:
            deletions += 1
            row -= 1
        else:
            insertions += 1
            column -= 1

    return WerCounts(
        substitutions=substitutions,
        deletions=deletions,
        insertions=insertions,
        reference_words=len(reference_words),
    )


def utterance_wer_counts(
    reference: str,
    hypothesis: str,
    normalization_policy: NormalizationPolicy = CONSERVATIVE_ENGLISH_V1,
) -> WerCounts:
    """Normalize and score one utterance, rejecting an empty normalized reference."""

    normalized_reference = normalization_policy.normalize(reference)
    normalized_hypothesis = normalization_policy.normalize(hypothesis)
    return word_edit_counts(normalized_reference.split(), normalized_hypothesis.split())


def sum_wer_counts(counts: Iterable[WerCounts]) -> WerCounts:
    """Sum composable WER sufficient statistics."""

    items = list(counts)
    if not items:
        raise ValueError("Cannot aggregate WER counts from an empty corpus")
    return WerCounts(
        substitutions=sum(item.substitutions for item in items),
        deletions=sum(item.deletions for item in items),
        insertions=sum(item.insertions for item in items),
        reference_words=sum(item.reference_words for item in items),
    )


def corpus_wer(counts: Iterable[WerCounts]) -> float:
    """Return standard corpus WER from summed edit and reference-word counts."""

    return sum_wer_counts(counts).wer


def macro_utterance_wer(counts: Iterable[WerCounts]) -> float:
    """Return the explicitly secondary, equal-utterance-weighted mean WER."""

    items = list(counts)
    if not items:
        raise ValueError("Cannot calculate macro utterance WER for an empty corpus")
    return sum(item.wer for item in items) / len(items)
