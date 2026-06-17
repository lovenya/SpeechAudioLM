from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from evaluation.errors import EvaluationError
from evaluation.metrics.registry import get_metric_specs
from evaluation.runner import EvalOptions, dry_run, run_evaluation


class EvaluationScopeIntegrationTest(unittest.TestCase):
    def _fixture(self, root: Path, *, failed_second: bool = False) -> tuple[Path, Path, Path]:
        dataset_root = root / "dataset"
        dataset_root.mkdir(exist_ok=True)
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
        temporary: bool = False,
        start_index: int = 0,
        max_samples: int | None = None,
        failed_second: bool = False,
    ) -> EvalOptions:
        manifest, dataset_config, metrics_config = self._fixture(root, failed_second=failed_second)
        return EvalOptions(
            model="cascaded",
            task="S2T",
            stage="s2t",
            direction="s2a",
            evaluated_run_id="20260616_1500",
            dataset="ttas",
            split="val",
            manifest_path=manifest,
            dataset_config_path=dataset_config,
            metrics_config_path=metrics_config,
            task_metric_map_path=root / "missing-task-map.json",
            output_root=(root / "temporary" / "results" / "evals") if temporary else root / "evals",
            metric_specs=get_metric_specs(["wer"]),
            max_samples=max_samples,
            start_index=start_index,
            cache_dir=root / "cache",
            device="cpu",
            command=["evaluate.py"],
            temporary=temporary,
        )

    def test_canonical_eval_uses_full_manifest_without_id_filtering(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            preview = dry_run(self._options(root))

            self.assertFalse(preview["temporary"])
            self.assertIsNone(preview["start_index"])
            self.assertIsNone(preview["max_samples"])
            self.assertEqual(preview["selected_sample_count"], 2)

    def test_canonical_eval_rejects_slicing(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            with self.assertRaisesRegex(EvaluationError, "full manifest"):
                dry_run(self._options(root, max_samples=1))

    def test_temporary_eval_rejects_invalid_slicing_bounds(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            with self.assertRaisesRegex(EvaluationError, "start_index"):
                dry_run(self._options(root, temporary=True, start_index=-1))
            with self.assertRaisesRegex(EvaluationError, "max_samples"):
                dry_run(self._options(root, temporary=True, max_samples=0))

    def test_temporary_eval_allows_slicing_without_latest_aliases(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            options = self._options(root, temporary=True, start_index=1, max_samples=1)
            paths = run_evaluation(options)
            summary = json.loads(paths["summary"].read_text(encoding="utf-8"))

            self.assertTrue(summary["temporary"])
            self.assertEqual(summary["status"], "diagnostic_non_authoritative")
            self.assertEqual(summary["result_validation"]["expected_row_count"], 1)
            self.assertEqual(list((root / "evals").rglob("latest_*")), [])

    def test_canonical_eval_retains_unsuccessful_generation_rows_with_errors(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            options = self._options(root, failed_second=True)
            paths = run_evaluation(options)
            summary = json.loads(paths["summary"].read_text(encoding="utf-8"))
            validation = summary["result_validation"]

            self.assertEqual(validation["result_status"], "passed_with_errors")
            self.assertEqual(validation["metric_counts"]["wer"], {
                "attempted_count": 2, "passed_count": 1, "error_count": 1,
            })
            self.assertEqual(validation["failed_sample_ids"]["wer"], ["sample-b"])
            self.assertNotIn("provenance", summary)


if __name__ == "__main__":
    unittest.main()
