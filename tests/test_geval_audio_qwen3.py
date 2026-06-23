from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from evaluation.metrics.registry import get_metric_specs
from evaluation.runner import EvalOptions, _dedicated_lalm_audio_rows, run_evaluation
from scripts.run_geval_audio_qwen3_criterion import build_prompt_record


class GEvalLALMQwen3Test(unittest.TestCase):
    def _options(self, root: Path, metrics: list[str], *, stage: str = "t2s", task: str = "T2S") -> EvalOptions:
        return EvalOptions(
            model="cascaded",
            task=task,
            stage=stage,
            direction="a2s" if task in {"T2S", "A2S"} else "s2a",
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

    def test_audio_rows_include_generated_source_and_reference_audio(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            dataset_root = root / "dataset"
            dataset_root.mkdir()
            generated = root / "generated.wav"
            source_audio = dataset_root / "audio.wav"
            reference_speech = dataset_root / "speech.wav"
            for path in (generated, source_audio, reference_speech):
                path.write_bytes(b"RIFF....WAVE")
            (root / "dataset.json").write_text(json.dumps({"dataset_root": str(dataset_root)}), encoding="utf-8")
            row = {
                "row_index": 0,
                "sample_id": "sample",
                "task": "T2S",
                "status": "success",
                "prompt": "A dog barks near a door.",
                "output_file": str(generated),
                "source": {
                    "audio_file": "audio.wav",
                    "speech_file": "speech.wav",
                    "speech_text": "Please let the dog inside.",
                    "audio_cap": "A dog barks near a door.",
                },
            }
            rows, errors = _dedicated_lalm_audio_rows(
                self._options(root, ["geval_audio_reference"]),
                [row],
                "geval_audio_reference",
            )
            self.assertFalse(errors)
            self.assertEqual(rows[0]["direction"], "audio_to_speech")
            self.assertEqual(rows[0]["generated_audio_file"], str(generated))
            self.assertEqual(rows[0]["source_audio_file"], str(source_audio))
            self.assertEqual(rows[0]["reference_anchors"][0]["audio_file"], str(reference_speech))
            self.assertEqual(rows[0]["reference_anchors"][0]["text"], "Please let the dog inside.")

    def test_audio_prompt_uses_direction_specific_input_and_rubric(self) -> None:
        prompt = build_prompt_record(
            variant="plausibility",
            direction="speech_to_audio",
            criterion="conditional_relevance",
            input_text="Please let the dog inside.",
            generated_audio_file="/tmp/generated.wav",
        )["rendered_user_message"]

        self.assertIn("Input speech transcript / utterance:\nPlease let the dog inside.", prompt)
        self.assertIn("generated environmental audio clip", prompt)
        self.assertIn("generated audio caption is meaningfully grounded in the input utterance", prompt)
        self.assertNotIn("generated utterance is meaningfully grounded in the input audio caption", prompt)

    def test_runner_executes_lalm_stub_and_preserves_audio_manifest_fields(self) -> None:
        metric = "geval_audio_plausibility"
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            dataset_root = root / "dataset"
            dataset_root.mkdir()
            generated = root / "generated.wav"
            source_audio = dataset_root / "audio.wav"
            generated.write_bytes(b"RIFF....WAVE")
            source_audio.write_bytes(b"RIFF....WAVE")
            (root / "dataset.json").write_text(json.dumps({"dataset_root": str(dataset_root)}), encoding="utf-8")
            (root / "metrics.json").write_text(json.dumps({"metrics": {metric: {
                "backend": "dedicated-script",
                "script": "scripts/run_geval_audio_qwen3_criterion.py",
                "script_backend": "stub",
                "batch_size": 20,
                "cache_dir": str(root / "cache" / metric),
            }}}), encoding="utf-8")
            (root / "manifest.jsonl").write_text(json.dumps({
                "row_index": 0,
                "sample_id": "sample",
                "task": "T2S",
                "status": "success",
                "prompt": "A dog barks near a door.",
                "output_file": str(generated),
                "source": {
                    "audio_file": "audio.wav",
                    "speech_file": "speech.wav",
                    "speech_text": "Please let the dog inside.",
                    "audio_cap": "A dog barks near a door.",
                },
            }) + "\n", encoding="utf-8")
            paths = run_evaluation(self._options(root, [metric]))
            row = json.loads(paths["manifest"].read_text(encoding="utf-8"))
            self.assertEqual(row["metric_id"], metric)
            self.assertEqual(row["status"], "success")
            self.assertEqual(row["output"]["score"], 4)
            self.assertEqual(row["output"]["score_name"], "weighted_final_score")
            self.assertEqual(row["input"]["generated"]["generated_audio_file"], str(generated))
            self.assertEqual(row["input"]["source"]["source_audio_file"], str(source_audio))
            artifact = paths["manifest"].parent / "artifacts" / "dedicated-script" / metric / "criterion_judgments.jsonl"
            self.assertTrue(artifact.is_file())
            judgment = json.loads(artifact.read_text(encoding="utf-8").splitlines()[0])
            self.assertEqual(judgment["metric_id"], metric)
            self.assertEqual(judgment["prompt"]["audio_attachments"][0]["role"], "generated_audio_to_evaluate")


if __name__ == "__main__":
    unittest.main()
