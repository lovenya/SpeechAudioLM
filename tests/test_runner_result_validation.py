from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from evaluation.artifacts import ArtifactAlreadyExistsError
from evaluation.metrics.registry import MetricSpec, get_metric_specs
from evaluation.runner import EvalOptions, run_evaluation


class RunnerResultValidationTest(unittest.TestCase):
    def _options(
        self,
        root: Path,
        *,
        temporary: bool = False,
        metric_specs: list[MetricSpec] | None = None,
        eval_run_id: str | None = None,
    ) -> EvalOptions:
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
            stage="s2t",
            direction="s2a",
            evaluated_run_id="20260616_1500",
            dataset="ttas",
            split="val",
            manifest_path=manifest,
            dataset_config_path=dataset_config,
            metrics_config_path=metrics_config,
            task_metric_map_path=root / "task_metric_map.json",
            output_root=(root / "temporary" / "results" / "evals") if temporary else root / "evals",
            metric_specs=metric_specs or get_metric_specs(["wer"]),
            max_samples=None,
            start_index=0,
            cache_dir=root / "cache",
            device="cpu",
            command=["evaluate.py"],
            temporary=temporary,
            eval_run_id=eval_run_id,
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
            inspect = json.loads(paths["inspect"].read_text(encoding="utf-8"))
            validation = inspect["summary"]["result_validation"]
            errors = json.loads(paths["errors"].read_text(encoding="utf-8"))
            self.assertEqual(
                set(errors),
                {"schema_version", "run_id", "metric_id", "error_count", "skipped_count", "errors"},
            )
            self.assertEqual(errors["schema_version"], 1)
            self.assertTrue(errors["run_id"])
            self.assertEqual(errors["metric_id"], "wer")
            self.assertEqual(errors["error_count"], 2)
            self.assertEqual(errors["skipped_count"], 0)
            self.assertEqual([entry["sample_id"] for entry in errors["errors"]], ["sample-a", "sample-b"])
            self.assertEqual(validation["result_status"], "error")
            self.assertEqual(validation["metric_counts"]["wer"], {
                "attempted_count": 2, "passed_count": 0, "error_count": 2,
            })
            self.assertEqual(validation["failed_sample_ids"]["wer"], ["sample-a", "sample-b"])

    def test_valid_final_writes_result_artifacts(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            paths = run_evaluation(self._options(root, eval_run_id="manual_eval_run"))

            self.assertTrue(all(path.is_file() for path in paths.values()))
            self.assertEqual(paths["run"].name, "run.json")
            self.assertEqual(paths["config"].name, "config.json")
            self.assertEqual(paths["manifest"].name, "manifest.jsonl")
            self.assertEqual(paths["inspect"].name, "inspect.json")
            self.assertEqual(paths["summary"].name, "summary.json")
            self.assertEqual(paths["report"].name, "report.md")
            self.assertEqual(paths["errors"].name, "errors.json")
            self.assertTrue((paths["run"].parent / "artifacts").is_dir())
            summary = json.loads(paths["summary"].read_text(encoding="utf-8"))
            inspect = json.loads(paths["inspect"].read_text(encoding="utf-8"))
            run_doc = json.loads(paths["run"].read_text(encoding="utf-8"))
            run_config = json.loads(paths["run_config"].read_text(encoding="utf-8"))
            report = inspect["summary"]["result_validation"]
            self.assertTrue(report["ok"])
            self.assertTrue(report["scientifically_valid"])
            self.assertEqual(report["status"], "final_scientific")
            self.assertEqual(report["result_status"], "passed")
            self.assertEqual(inspect["summary"]["result_validation"], report)
            self.assertEqual(summary["schema_version"], 1)
            self.assertEqual(summary["metric"]["metric_id"], "wer")
            self.assertEqual(summary["samples"]["total"], 2)
            self.assertEqual(summary["scores"]["primary"]["name"], "wer")
            self.assertEqual(run_config["schema_version"], 1)
            self.assertEqual(set(run_config), {"schema_version", "input", "metric", "backend", "execution"})
            self.assertEqual(run_config["metric"]["input_contract"]["prediction_fields"], ["generated_text"])
            self.assertEqual(run_config["backend"]["name"], "metric_algorithm")
            self.assertEqual(
                set(run_doc),
                {"schema_version", "run_id", "temporary", "status", "metric", "target", "dataset", "split", "started_at", "completed_at", "code", "slurm", "paths"},
            )
            self.assertEqual(run_doc["run_id"], "manual_eval_run")
            self.assertNotEqual(run_doc["started_at"], "manual_eval_run")
            self.assertEqual(run_doc["status"], "complete")
            self.assertEqual(run_doc["metric"], {"metric_id": "wer", "level": "sample", "reference_pairing_required": True})
            self.assertEqual(run_doc["target"]["category"], "cascaded")
            self.assertEqual(run_doc["target"]["run_path"], str(root))
            self.assertEqual(run_doc["target"]["manifest"], str(root / "manifest.jsonl"))
            self.assertEqual(run_doc["target"]["summary"], str(root / "summary.json"))
            self.assertEqual(set(run_doc["code"]), {"git_commit", "git_dirty"})
            self.assertEqual(set(run_doc["slurm"]), {"job_id", "array_job_id", "array_task_id"})
            expected_logs = root / "logs" / "evals" / paths["run"].parent.relative_to(root / "evals")
            self.assertEqual(run_doc["paths"], {"result": str(paths["run"].parent), "logs": str(expected_logs)})
            self.assertEqual(
                set(inspect),
                {"schema_version", "run_id", "metric", "target", "config_summary", "summary", "sample_policy", "samples", "backend_artifacts"},
            )
            self.assertEqual(inspect["schema_version"], 1)
            self.assertTrue(inspect["run_id"])
            self.assertEqual(inspect["metric"]["metric_id"], "wer")
            self.assertEqual(inspect["target"]["stage"], "s2t")
            self.assertEqual(inspect["target"]["direction"], "s2a")
            self.assertEqual(inspect["config_summary"]["input_contract"]["prediction_fields"], ["generated_text"])
            self.assertEqual(inspect["sample_policy"], {"mode": "all", "count": 2})
            self.assertEqual(len(inspect["samples"]), 2)

    def test_corpus_metric_manifest_marks_participants_included(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            metric = MetricSpec(name="corpus_stub", level="corpus")
            with patch("evaluation.runner._compute_external_metrics", return_value={"corpus_stub": 1.0}):
                paths = run_evaluation(self._options(root, metric_specs=[metric]))

            rows = [json.loads(line) for line in paths["manifest"].read_text(encoding="utf-8").splitlines()]
            summary = json.loads(paths["summary"].read_text(encoding="utf-8"))
            self.assertEqual([row["status"] for row in rows], ["included", "included"])
            self.assertTrue(all(set(row) == {"schema_version", "row_index", "sample_id", "metric_id", "status", "input", "output", "error"} for row in rows))
            self.assertTrue(all(row["output"]["details"] == {"score_location": "summary.json"} for row in rows))
            self.assertIsNone(summary["samples"]["successful"])
            self.assertEqual(summary["samples"]["included"], 2)
            errors = json.loads(paths["errors"].read_text(encoding="utf-8"))
            self.assertEqual(errors["error_count"], 0)
            self.assertEqual(errors["skipped_count"], 0)
            self.assertEqual(errors["errors"], [])

    def test_corpus_metric_validation_failure_does_not_emit_included_rows_as_errors(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            metric = MetricSpec(name="corpus_stub", level="corpus")
            paths = run_evaluation(self._options(root, metric_specs=[metric]))

            rows = [json.loads(line) for line in paths["manifest"].read_text(encoding="utf-8").splitlines()]
            errors = json.loads(paths["errors"].read_text(encoding="utf-8"))
            summary = json.loads(paths["summary"].read_text(encoding="utf-8"))
            self.assertEqual([row["status"] for row in rows], ["included", "included"])
            self.assertEqual(errors["errors"], [])
            self.assertEqual(errors["error_count"], 0)
            self.assertEqual(summary["errors"]["validation"]["result_status"], "error")

    def test_temporary_eval_publishes_diagnostic_validation_report(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            options = self._options(root, temporary=True)

            with patch("evaluation.runner._evaluate_row", side_effect=self._incomplete_row):
                paths = run_evaluation(options)

            inspect = json.loads(paths["inspect"].read_text(encoding="utf-8"))
            report = inspect["summary"]["result_validation"]
            self.assertTrue(report["ok"])
            self.assertFalse(report["scientifically_valid"])
            self.assertEqual(report["status"], "diagnostic_non_authoritative")
            self.assertEqual(inspect["summary"]["result_validation"], report)
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
            self.assertTrue(all(row["output"]["score"] is None for row in rows))
            self.assertTrue(all("non-finite" in row["error"]["message"] for row in rows))
            summary = json.loads(paths["summary"].read_text())
            errors = json.loads(paths["errors"].read_text())
            self.assertEqual(errors["error_count"], 2)
            self.assertTrue(all(entry["error_type"] == "MetricError" for entry in errors["errors"]))
            self.assertEqual(summary["errors"]["validation"]["result_status"], "error")

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
