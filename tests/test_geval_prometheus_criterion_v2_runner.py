from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from evaluation.io import _normalize_generation_record
from evaluation.metrics.registry import get_metric_specs
from evaluation.runner import EvalOptions, _dedicated_text_pair_rows, run_evaluation


class GEvalPrometheusCriterionV2RunnerTest(unittest.TestCase):
    def _options(self, root: Path, metrics: list[str]) -> EvalOptions:
        return EvalOptions(
            model="cascaded",
            task="T2T",
            stage="t2t",
            direction="a2s",
            evaluated_run_id="20260616_1500",
            dataset="ttas",
            split="val",
            manifest_path=root / "manifest.jsonl",
            dataset_config_path=root / "dataset.json",
            metrics_config_path=root / "metrics.json",
            task_metric_map_path=root / "task_metric_map.json",
            output_root=root / "temporary" / "results" / "evals",
            metric_specs=get_metric_specs(metrics),
            max_samples=None,
            start_index=0,
            cache_dir=root / "cache",
            device="cpu",
            command=["evaluate.py"],
            temporary=True,
        )

    def test_reference_extraction_from_wrapped_cascaded_rows(self) -> None:
        cases = [
            ("A2S", "audio_to_speech", "Reference speech.", "audio_caption", "Input audio scene."),
            ("S2A", "speech_to_audio", "Reference audio scene.", "speech_text", "Input speech."),
        ]
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            options = self._options(root, ["geval_text_reference"])
            for pipeline, direction, expected_reference, input_field, input_text in cases:
                reference = {
                    "speech_text": "Reference speech.",
                    "audio_caption": "Reference audio scene.",
                }
                wrapped = {
                    "pipeline": pipeline,
                    "sample": {
                        "row_index": 0,
                        "sample_id": "sample",
                        "status": "success",
                        "reference": reference,
                        "stages": [
                            {
                                "stage": 2,
                                "task": "text_to_text",
                                "model": "text-model",
                                "mode": "audio_description_to_speech" if pipeline == "A2S" else "speech_to_audio_description",
                                "input": {"type": input_field, "text": input_text},
                                "output": {"type": "text", "text": "Generated output."},
                            }
                        ],
                    },
                }
                row = _normalize_generation_record(wrapped)
                helper_rows, errors = _dedicated_text_pair_rows(
                    options,
                    [row],
                    "geval_text_reference",
                    {},
                )
                self.assertFalse(errors)
                self.assertEqual(helper_rows[0]["direction"], direction)
                self.assertEqual(helper_rows[0]["input_text"], input_text)
                self.assertEqual(helper_rows[0]["generated_text"], "Generated output.")
                self.assertEqual(helper_rows[0]["reference_anchors"][0]["text"], expected_reference)

    def test_canonical_stage_fields_override_conflicting_flattened_fields(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            options = self._options(root, ["geval_text_plausibility"])
            row = {
                "status": "success",
                "row_index": 0,
                "sample_id": "sample",
                "pipeline": "A2S",
                "prompt": "Wrong flattened input.",
                "candidate_text": "Wrong flattened output.",
                "source": {"audio_cap": "Dataset audio caption."},
                "model_extra": {"stages": [{
                    "mode": "speech_to_audio_description",
                    "input": {"text": "Canonical speech input."},
                    "output": {"text": "Canonical audio output."},
                }]},
            }
            helper_rows, errors = _dedicated_text_pair_rows(
                options, [row], "geval_text_plausibility", {},
            )
            self.assertFalse(errors)
            self.assertEqual(helper_rows[0]["direction"], "speech_to_audio")
            self.assertEqual(helper_rows[0]["input_text"], "Canonical speech input.")
            self.assertEqual(helper_rows[0]["generated_text"], "Canonical audio output.")

    def test_runner_executes_both_new_metric_ids_separately(self) -> None:
        metric_ids = [
            "geval_text_plausibility",
            "geval_text_reference",
        ]
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            (root / "dataset").mkdir()
            (root / "dataset.json").write_text(json.dumps({"dataset_root": str(root / "dataset")}), encoding="utf-8")
            metric_config = {
                metric: {
                    "backend": "dedicated-script",
                    "script": "scripts/run_geval_prometheus_criterion_v2.py",
                    "script_backend": "stub",
                    "batch_size": 20,
                    "cache_dir": str(root / "cache" / metric),
                }
                for metric in metric_ids
            }
            (root / "metrics.json").write_text(json.dumps({"metrics": metric_config}), encoding="utf-8")
            (root / "manifest.jsonl").write_text(
                json.dumps(
                    {
                        "model": "cascaded",
                        "task": "T2T",
                        "pipeline": "A2S",
                        "row_index": 0,
                        "sample_id": "sample",
                        "generated_text": "Please let the dog inside.",
                        "source": {
                            "audio_cap": "A dog is barking at a door.",
                            "speech_text": "Please let the dog inside.",
                        },
                    }
                )
                + "\n",
                encoding="utf-8",
            )
            rows_by_metric = {}
            for metric in metric_ids:
                paths = run_evaluation(self._options(root, [metric]))
                rows_by_metric[metric] = json.loads(paths["per_sample"].read_text(encoding="utf-8"))
                self.assertIn(f"/stages/a2s/t2t/cascaded/ttas/val/20260616_1500/{metric}/", str(paths["per_sample"]))
                artifact = paths["per_sample"].parent / "artifacts" / "dedicated-script" / metric / "criterion_judgments.jsonl"
                self.assertTrue(artifact.is_file())
                judgment = json.loads(artifact.read_text(encoding="utf-8").splitlines()[0])
                self.assertEqual(judgment["metric_id"], metric)
                self.assertIn("messages", judgment["prompt"])
                self.assertIn("Score 5:", judgment["prompt"]["rubric"])
                backend = json.loads(paths["inspect"].read_text(encoding="utf-8"))["config_summary"]["backend"]
                self.assertEqual(backend["criterion_judgments"], str(artifact))

            for metric, row in rows_by_metric.items():
                self.assertEqual(row["metric_id"], metric)
                self.assertEqual(row["output"]["score"], 4)
                self.assertEqual(row["output"]["score_name"], "weighted_final_score")
                self.assertEqual(row["output"]["details"]["final_score"], 4)
                criterion = next(iter(row["output"]["details"]["criteria"].values()))
                self.assertEqual(criterion["status"], "success")
                self.assertEqual(criterion["score"], 4)
                self.assertNotIn("raw_response", criterion)
                self.assertEqual(row["input"]["source"]["stage_input_text"], "A dog is barking at a door.")
                self.assertEqual(row["input"]["generated"]["stage_output_text"], "Please let the dog inside.")
            anchors = rows_by_metric["geval_text_reference"]["input"]["reference"]["reference_anchors"]
            self.assertEqual(anchors[0]["text"], "Please let the dog inside.")


    def test_missing_grounded_reference_marks_top_level_row_error(self) -> None:
        metric = "geval_text_reference"
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            (root / "dataset").mkdir()
            (root / "dataset.json").write_text(json.dumps({"dataset_root": str(root / "dataset")}), encoding="utf-8")
            (root / "metrics.json").write_text(json.dumps({"metrics": {metric: {
                "backend": "dedicated-script",
                "script": "scripts/run_geval_prometheus_criterion_v2.py",
                "script_backend": "stub",
                "batch_size": 20,
                "cache_dir": str(root / "cache"),
            }}}), encoding="utf-8")
            (root / "manifest.jsonl").write_text(json.dumps({
                "model": "cascaded",
                "task": "T2T",
                "pipeline": "A2S",
                "row_index": 0,
                "sample_id": "sample",
                "generated_text": "Please let the dog inside.",
                "source": {"audio_cap": "A dog is barking at a door."},
            }) + "\n", encoding="utf-8")
            paths = run_evaluation(self._options(root, [metric]))
            row = json.loads(paths["per_sample"].read_text(encoding="utf-8"))
            self.assertEqual(row["status"], "error")
            self.assertEqual(row["error"]["message"], "missing_required_reference_anchor")
            self.assertIsNone(row["output"]["details"]["final_score"])


if __name__ == "__main__":
    unittest.main()
