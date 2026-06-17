from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from evaluation.artifacts import ArtifactAlreadyExistsError
from evaluation.metrics.registry import get_metric_specs
from evaluation.runner import EvalOptions, run_evaluation


class RunnerResultValidationTest(unittest.TestCase):
    def _options(self, root: Path, *, temporary: bool = False) -> EvalOptions:
        dataset_root = root / "dataset"
        dataset_root.mkdir()
        dataset_config = root / "dataset.json"
        dataset_config.write_text(
            json.dumps({"dataset_root": str(dataset_root)}), encoding="utf-8"
        )
        metrics_config = root / "metrics.json"
        metrics_config.write_text(
            json.dumps({"metrics": {"wer": {"backend": "internal"}}}),
            encoding="utf-8",
        )
        manifest = root / "manifest.jsonl"
        rows = [
            {
                "row_index": index,
                "sample_id": sample_id,
                "status": "success",
                "candidate_text": "hello world",
                "source": {"speech_text": "hello world"},
            }
            for index, sample_id in enumerate(("sample-a", "sample-b"))
        ]
        manifest.write_text(
            "".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8"
        )

        return EvalOptions(
            model="cascaded",
            task="S2T",
            dataset="ttas",
            split="val",
            manifest_path=manifest,
            dataset_config_path=dataset_config,
            metrics_config_path=metrics_config,
            task_metric_map_path=root / "task_metric_map.json",
            output_root=(root / "temporary" / "results" / "evals") if temporary else root / "evals",
            metric_specs=get_metric_specs(["wer"]),
            max_samples=None,
            start_index=0,
            cache_dir=root / "cache",
            device="cpu",
            command=["evaluate.py"],
            temporary=temporary,
        )

    @staticmethod
    def _incomplete_row(row: dict, ordinal: int, specs: list, dataset_root: Path) -> dict:
        return {
            "row_index": row["row_index"],
            "sample_id": row["sample_id"],
            "status": "success",
            "output_file": None,
            "references": {},
            "predictions": {},
            "metrics": {},
            "errors": {},
        }

    def test_final_with_attributable_errors_writes_result_artifacts(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            options = self._options(root)

            with patch("evaluation.runner._evaluate_row", side_effect=self._incomplete_row):
                paths = run_evaluation(options)

            self.assertTrue(all(path.is_file() for path in paths.values()))
            summary = json.loads(paths["summary"].read_text(encoding="utf-8"))
            validation = summary["result_validation"]
            self.assertEqual(validation["result_status"], "error")
            self.assertEqual(validation["metric_counts"]["wer"], {
                "attempted_count": 2, "passed_count": 0, "error_count": 2,
            })
            self.assertEqual(validation["failed_sample_ids"]["wer"], ["sample-a", "sample-b"])

    def test_valid_final_writes_result_artifacts(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            paths = run_evaluation(self._options(root))

            self.assertTrue(all(path.is_file() for path in paths.values()))
            summary = json.loads(paths["summary"].read_text(encoding="utf-8"))
            run_config = json.loads(paths["run_config"].read_text(encoding="utf-8"))
            report = summary["result_validation"]
            self.assertTrue(report["ok"])
            self.assertTrue(report["scientifically_valid"])
            self.assertEqual(report["status"], "final_scientific")
            self.assertEqual(report["result_status"], "passed")
            self.assertEqual(run_config["result_validation"], report)

    def test_temporary_eval_publishes_diagnostic_validation_report(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            options = self._options(root, temporary=True)

            with patch("evaluation.runner._evaluate_row", side_effect=self._incomplete_row):
                paths = run_evaluation(options)

            summary = json.loads(paths["summary"].read_text(encoding="utf-8"))
            run_config = json.loads(paths["run_config"].read_text(encoding="utf-8"))
            report = summary["result_validation"]
            self.assertTrue(report["ok"])
            self.assertFalse(report["scientifically_valid"])
            self.assertEqual(report["status"], "diagnostic_non_authoritative")
            self.assertEqual(run_config["result_validation"], report)
            self.assertEqual(list((root / "evals").rglob("latest_*")), [])

    def test_nonfinite_sample_score_becomes_explicit_error_and_writes(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            options = self._options(root)

            def nonfinite_row(row: dict, ordinal: int, specs: list, dataset_root: Path) -> dict:
                return {
                    "row_index": row["row_index"],
                    "sample_id": row["sample_id"],
                    "status": "success",
                    "output_file": None,
                    "references": {},
                    "predictions": {},
                    "metrics": {"wer": float("nan")},
                    "errors": {},
                }

            with patch("evaluation.runner._evaluate_row", side_effect=nonfinite_row):
                paths = run_evaluation(options)

            rows = [json.loads(line) for line in paths["per_sample"].read_text().splitlines()]
            self.assertTrue(all(row["status"] == "error" for row in rows))
            self.assertTrue(all("wer" not in row["metrics"] for row in rows))
            self.assertTrue(all("non-finite" in row["errors"]["wer"] for row in rows))
            summary = json.loads(paths["summary"].read_text())
            self.assertEqual(summary["result_validation"]["result_status"], "error")

    def test_timestamped_artifacts_refuse_overwrite(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            options = self._options(root)

            with patch("evaluation.runner.utc_timestamp", return_value="20260101T000000Z"):
                paths = run_evaluation(options)
                original = {key: path.read_bytes() for key, path in paths.items()}
                with self.assertRaises(ArtifactAlreadyExistsError):
                    run_evaluation(options)

            self.assertEqual(
                {key: path.read_bytes() for key, path in paths.items()}, original
            )


if __name__ == "__main__":
    unittest.main()
