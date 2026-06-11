from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from evaluate import _load_common_sample_ids
from evaluation.errors import EvaluationError
from evaluation.metric_contracts import EvaluationMode
from evaluation.metrics.registry import get_metric_specs
from evaluation.runner import EvalOptions, dry_run, run_evaluation


class EvaluationScopeIntegrationTest(unittest.TestCase):
    def _fixture(self, root: Path, *, failed_second: bool = False) -> tuple[Path, Path, Path]:
        dataset_root = root / "dataset"
        dataset_root.mkdir()
        dataset_config = root / "dataset.json"
        dataset_config.write_text(json.dumps({"dataset_root": str(dataset_root)}), encoding="utf-8")
        metrics_config = root / "metrics.json"
        metrics_config.write_text(json.dumps({"metrics": {"wer": {"backend": "internal"}}}), encoding="utf-8")
        manifest = root / "manifest.jsonl"
        rows = [
            {
                "row_index": index,
                "sample_id": sample_id,
                "status": "error" if failed_second and index == 1 else "success",
                "candidate_text": "hello world",
                "source": {"speech_text": "hello world"},
                "error": "generation failed" if failed_second and index == 1 else None,
            }
            for index, sample_id in enumerate(("sample-a", "sample-b"))
        ]
        manifest.write_text("".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8")
        return manifest, dataset_config, metrics_config

    def _options(
        self,
        root: Path,
        *,
        mode: EvaluationMode,
        common_ids: tuple[str, ...] | None,
        start_index: int = 0,
        max_samples: int | None = None,
        failed_second: bool = False,
    ) -> EvalOptions:
        manifest, dataset_config, metrics_config = self._fixture(root, failed_second=failed_second)
        return EvalOptions(
            model="cascaded",
            task="S2T",
            dataset="ttas",
            split="val",
            manifest_path=manifest,
            dataset_config_path=dataset_config,
            metrics_config_path=metrics_config,
            task_metric_map_path=root / "missing-task-map.json",
            output_root=root / "evals",
            metric_specs=get_metric_specs(["wer"]),
            max_samples=max_samples,
            start_index=start_index,
            cache_dir=root / "cache",
            device="cpu",
            command=["evaluate.py"],
            evaluation_mode=mode,
            common_sample_ids=common_ids,
        )

    def test_final_requires_explicit_complete_common_ids_and_disallows_slicing(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            with self.assertRaisesRegex(EvaluationError, "requires explicit --common-sample-ids"):
                dry_run(self._options(root, mode=EvaluationMode.FINAL, common_ids=None))

        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            with self.assertRaisesRegex(EvaluationError, "complete common sample-ID set"):
                dry_run(
                    self._options(
                        root,
                        mode=EvaluationMode.FINAL,
                        common_ids=("sample-a",),
                    )
                )

        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            with self.assertRaisesRegex(EvaluationError, "disallows slicing"):
                dry_run(
                    self._options(
                        root,
                        mode=EvaluationMode.FINAL,
                        common_ids=("sample-a", "sample-b"),
                        max_samples=1,
                    )
                )

    def test_final_rejects_unsuccessful_generation_rows(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            options = self._options(
                root,
                mode=EvaluationMode.FINAL,
                common_ids=("sample-a", "sample-b"),
                failed_second=True,
            )
            with self.assertRaisesRegex(EvaluationError, "sample-b: generation was not successful"):
                dry_run(options)

    def test_backend_smoke_is_isolated_nonpublishable_and_does_not_update_latest(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            options = self._options(
                root,
                mode=EvaluationMode.BACKEND_SMOKE,
                common_ids=None,
                start_index=1,
                max_samples=1,
            )
            paths = run_evaluation(options)
            summary = json.loads(paths["summary"].read_text(encoding="utf-8"))

            self.assertEqual(summary["status"], "diagnostic_non_authoritative")
            self.assertIn("backend_smoke", paths["summary"].parts)
            self.assertFalse(list((root / "evals").rglob("latest_*")))
            self.assertEqual(summary["scope"]["selected_sample_count"], 1)

    def test_final_publishes_latest_and_records_immutable_scope_hashes(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            options = self._options(
                root,
                mode=EvaluationMode.FINAL,
                common_ids=("sample-b", "sample-a"),
            )
            paths = run_evaluation(options)
            summary = json.loads(paths["summary"].read_text(encoding="utf-8"))
            run_config = json.loads(paths["run_config"].read_text(encoding="utf-8"))

            self.assertEqual(summary["status"], "final_scientific")
            self.assertNotIn("backend_smoke", paths["summary"].parts)
            self.assertTrue(paths["summary"].with_name("latest_summary_val.json").is_file())
            self.assertEqual(summary["scope"]["selected_sample_count"], 2)
            for key in (
                "manifest_sha256",
                "config_sha256",
                "policy_sha256",
                "selected_rows_sha256",
                "sample_id_set_sha256",
                "git_worktree_sha256",
            ):
                self.assertEqual(len(summary["provenance"][key]), 64)
            self.assertEqual(summary["provenance"], run_config["provenance"])

    def test_common_id_loader_accepts_json_and_rejects_invalid_values(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            ids_path = root / "ids.json"
            ids_path.write_text(json.dumps({"common_sample_ids": ["a", "b"]}), encoding="utf-8")
            self.assertEqual(_load_common_sample_ids(str(ids_path)), ("a", "b"))
            ids_path.write_text(json.dumps({"wrong": ["a"]}), encoding="utf-8")
            with self.assertRaises(EvaluationError):
                _load_common_sample_ids(str(ids_path))


if __name__ == "__main__":
    unittest.main()
