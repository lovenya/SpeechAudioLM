from __future__ import annotations

import unittest
from dataclasses import FrozenInstanceError

from evaluation.metric_contracts import (
    ContractError,
    MetricContract,
    MetricLevel,
)


def _metric_contract(**overrides: object) -> MetricContract:
    fields: dict[str, object] = {
        "name": "fad",
        "level": MetricLevel.DISTRIBUTIONAL_CORPUS,
        "academic_definition": "Distance between embedding distributions.",
        "implementation": "official-package",
        "implementation_version": "1.0",
        "higher_is_better": False,
        "evaluated_fields": ("reference_audio", "generated_audio"),
        "reference_construction": "Reference audio from the dataset split.",
        "successful_id_policy": "Report rows that were evaluated and rows that failed.",
        "constraints": ("Complete canonical manifest required.",),
        "output_validation_rules": ("Reject non-finite and sentinel values.",),
        "shard_composable": False,
        "minimum_valid_corpus_size": 2,
    }
    fields.update(overrides)
    return MetricContract(**fields)  # type: ignore[arg-type]


class MetricContractTest(unittest.TestCase):
    def test_contract_is_immutable_and_reports_level_and_constraints(self) -> None:
        contract = _metric_contract()

        with self.assertRaises(FrozenInstanceError):
            contract.name = "changed"  # type: ignore[misc]

        report = contract.report_fields()
        self.assertEqual(report["metric_level"], "distributional-corpus")
        self.assertEqual(report["constraints"], ["Complete canonical manifest required."])
        self.assertFalse(report["shard_composable"])

    def test_contract_fails_closed_on_missing_or_ambiguous_semantics(self) -> None:
        for override in (
            {"level": "corpus"},
            {"constraints": ()},
            {"evaluated_fields": ()},
            {"output_validation_rules": ()},
            {"minimum_valid_corpus_size": 0},
        ):
            with self.subTest(override=override), self.assertRaises(ContractError):
                _metric_contract(**override)

    def test_contract_rejects_duplicate_tuple_fields(self) -> None:
        for override in (
            {"evaluated_fields": ("audio", "audio")},
            {"constraints": ("same", "same")},
            {"output_validation_rules": ("finite", "finite")},
        ):
            with self.subTest(override=override), self.assertRaises(ContractError):
                _metric_contract(**override)


if __name__ == "__main__":
    unittest.main()
