from __future__ import annotations

import json
import tempfile
import unittest
from dataclasses import FrozenInstanceError
from pathlib import Path

from evaluation.metric_contracts import (
    ContractError,
    EvaluationMode,
    EvaluationProvenance,
    EvaluationScope,
    MetricContract,
    MetricLevel,
    sha256_config,
    sha256_manifest,
    sha256_policy,
    sha256_sample_id_set,
    sha256_selected_rows,
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
        "reference_construction": "Reference audio for the common selected IDs.",
        "successful_id_policy": "Every common selected ID must succeed.",
        "constraints": ("Complete common corpus required.",),
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
        self.assertEqual(report["constraints"], ["Complete common corpus required."])
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

    def test_final_scope_requires_complete_common_ids_and_disallows_slicing(self) -> None:
        ids = ("a", "b", "c")
        scope = EvaluationScope(
            mode=EvaluationMode.FINAL,
            expected_sample_ids=ids,
            common_sample_ids=("c", "b", "a"),
            selected_sample_ids=("b", "a", "c"),
        )

        self.assertTrue(scope.scientific)
        self.assertTrue(scope.publishable)
        self.assertEqual(scope.report_fields()["scope_label"], "final_scientific_evaluation")

        invalid_scopes = (
            {"selected_sample_ids": ("a", "b")},
            {"common_sample_ids": ("a", "b")},
            {"start_index": 1},
            {"max_samples": 2},
        )
        for override in invalid_scopes:
            kwargs = {
                "mode": EvaluationMode.FINAL,
                "expected_sample_ids": ids,
                "common_sample_ids": ids,
                "selected_sample_ids": ids,
            }
            kwargs.update(override)
            with self.subTest(override=override), self.assertRaises(ContractError):
                EvaluationScope(**kwargs)  # type: ignore[arg-type]

    def test_backend_smoke_may_be_partial_but_is_never_scientific_or_publishable(self) -> None:
        scope = EvaluationScope(
            mode=EvaluationMode.BACKEND_SMOKE,
            expected_sample_ids=("a", "b", "c"),
            common_sample_ids=("a", "b", "c"),
            selected_sample_ids=("b",),
            start_index=1,
            max_samples=1,
        )

        report = scope.report_fields()
        self.assertFalse(report["scientific"])
        self.assertFalse(report["publishable"])
        self.assertEqual(
            report["scope_label"], "backend_smoke_non_scientific_non_publishable"
        )

    def test_scope_rejects_duplicate_unknown_and_empty_ids(self) -> None:
        for override in (
            {"selected_sample_ids": ("a", "a")},
            {"selected_sample_ids": ("unknown",)},
            {"common_sample_ids": ()},
        ):
            kwargs = {
                "mode": EvaluationMode.BACKEND_SMOKE,
                "expected_sample_ids": ("a", "b"),
                "common_sample_ids": ("a", "b"),
                "selected_sample_ids": ("a",),
            }
            kwargs.update(override)
            with self.subTest(override=override), self.assertRaises(ContractError):
                EvaluationScope(**kwargs)  # type: ignore[arg-type]


class ProvenanceHashTest(unittest.TestCase):
    def test_manifest_hash_covers_exact_bytes(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            manifest = Path(tmpdir) / "manifest.jsonl"
            manifest.write_bytes(b'{"sample_id":"a"}\n')
            first = sha256_manifest(manifest)
            manifest.write_bytes(b'{"sample_id": "a"}\n')
            second = sha256_manifest(manifest)

        self.assertNotEqual(first, second)

    def test_config_and_policy_hash_canonical_json(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            config = Path(tmpdir) / "config.json"
            config.write_text(json.dumps({"b": 2, "a": 1}, indent=2), encoding="utf-8")

            self.assertEqual(sha256_config(config), sha256_config({"a": 1, "b": 2}))
            self.assertEqual(sha256_policy({"b": 2, "a": 1}), sha256_policy({"a": 1, "b": 2}))

    def test_selected_row_hash_preserves_order_but_canonicalizes_keys(self) -> None:
        rows = [{"sample_id": "a", "value": 1}, {"sample_id": "b", "value": 2}]

        self.assertEqual(
            sha256_selected_rows(rows),
            sha256_selected_rows(
                [{"value": 1, "sample_id": "a"}, {"value": 2, "sample_id": "b"}]
            ),
        )
        self.assertNotEqual(sha256_selected_rows(rows), sha256_selected_rows(list(reversed(rows))))

    def test_sample_id_set_hash_is_order_independent_and_rejects_duplicates(self) -> None:
        self.assertEqual(
            sha256_sample_id_set(("a", "b", "c")),
            sha256_sample_id_set(("c", "a", "b")),
        )
        with self.assertRaises(ContractError):
            sha256_sample_id_set(("a", "a"))

    def test_provenance_is_immutable_and_rejects_non_sha256_values(self) -> None:
        digest = "a" * 64
        provenance = EvaluationProvenance(
            manifest_sha256=digest,
            config_sha256=digest,
            policy_sha256=digest,
            selected_rows_sha256=digest,
            sample_id_set_sha256=digest,
            git_commit="b" * 40,
            git_worktree_sha256=digest,
        )

        with self.assertRaises(FrozenInstanceError):
            provenance.config_sha256 = "b" * 64  # type: ignore[misc]
        with self.assertRaises(ContractError):
            EvaluationProvenance(
                manifest_sha256="not-a-hash",
                config_sha256=digest,
                policy_sha256=digest,
                selected_rows_sha256=digest,
                sample_id_set_sha256=digest,
                git_commit="b" * 40,
                git_worktree_sha256=digest,
            )
        with self.assertRaises(ContractError):
            EvaluationProvenance(
                manifest_sha256=digest,
                config_sha256=digest,
                policy_sha256=digest,
                selected_rows_sha256=digest,
                sample_id_set_sha256=digest,
                git_commit="not-a-git-object-id",
                git_worktree_sha256=digest,
            )

    def test_hash_helpers_reject_non_finite_or_empty_inputs(self) -> None:
        with self.assertRaises(ContractError):
            sha256_config({"invalid": float("nan")})
        with self.assertRaises(ContractError):
            sha256_selected_rows([])
        with self.assertRaises(ContractError):
            sha256_sample_id_set(())


if __name__ == "__main__":
    unittest.main()
