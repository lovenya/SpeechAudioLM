from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from evaluation.metrics.registry import get_metric_specs
from evaluation.runner import EvalOptions, _dedicated_lalm_audio_rows, run_evaluation
from scripts.run_geval_audio_qwen3_criterion import PROMPT_VERSIONS, build_prompt_record


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
            self.assertNotIn("input_text", rows[0])
            self.assertNotIn("generated_text", rows[0])
            self.assertNotIn("text", rows[0]["reference_anchors"][0])

    def test_audio_prompt_uses_only_audio_inputs_and_direction_specific_rubric(self) -> None:
        prompt_record = build_prompt_record(
            variant="plausibility",
            direction="speech_to_audio",
            criterion="scenario_plausibility",
            source_audio_file="/tmp/source.wav",
            generated_audio_file="/tmp/generated.wav",
            input_text="Please let the dog inside.",
            generated_text="A dog barks by a door.",
        )
        prompt = prompt_record["rendered_user_message"]
        self.assertEqual(prompt_record["prompt_version"], PROMPT_VERSIONS["plausibility"])

        self.assertIn("Clip 1 - INPUT SPEECH AUDIO (context): attached file input_audio_source.wav", prompt)
        self.assertIn("Clip 2 - OUTPUT ENVIRONMENTAL AUDIO TO EVALUATE: attached file output_audio_generated.wav", prompt)
        self.assertIn("output environmental audio should be an audio scene", prompt)
        self.assertIn("Criterion: input_speech_and_output_environmental_audio_grounded_scenario_plausibility", prompt)
        self.assertIn("Internal criterion id: scenario_plausibility", prompt)
        self.assertIn("Criterion boundary:", prompt)
        self.assertIn("form a believable, meaningfully connected shared situation", prompt)
        self.assertIn("Focus on two questions", prompt)
        self.assertIn("situationally connected", prompt)
        self.assertIn("physically and practically plausible", prompt)
        self.assertIn("Do not penalize emotional contrast by itself", prompt)
        self.assertNotIn("Please let the dog inside.", prompt)
        self.assertNotIn("A dog barks by a door.", prompt)
        self.assertNotIn("generated speech audio is meaningfully grounded in the heard input environmental audio", prompt)

        a2s_quality_record = build_prompt_record(
            variant="plausibility",
            direction="audio_to_speech",
            criterion="perceptual_audio_quality",
            source_audio_file="/tmp/water.wav",
            generated_audio_file="/tmp/speech.wav",
        )
        a2s_quality = a2s_quality_record["rendered_user_message"]
        self.assertIn("Clip 1 - OUTPUT SPEECH AUDIO TO EVALUATE: attached file output_audio_speech.wav", a2s_quality)
        self.assertNotIn("input_audio_water.wav", a2s_quality)
        self.assertNotIn("Clip 2 -", a2s_quality)
        self.assertIn("Criterion: output_speech_perceptual_quality", a2s_quality)
        self.assertIn("The input/context clip is intentionally not provided", a2s_quality)
        self.assertEqual(
            a2s_quality_record["audio_attachments"],
            [{"role": "generated_audio_to_evaluate", "path": "/tmp/speech.wav"}],
        )

        s2a_modality_record = build_prompt_record(
            variant="plausibility",
            direction="speech_to_audio",
            criterion="target_modality_success",
            source_audio_file="/tmp/speech.wav",
            generated_audio_file="/tmp/environment.wav",
        )
        s2a_modality = s2a_modality_record["rendered_user_message"]
        self.assertEqual(s2a_modality_record["prompt_version"], "geval_audio_plausibility_v6")
        self.assertIn(
            "Clip 1 - OUTPUT ENVIRONMENTAL AUDIO TO EVALUATE: attached file output_audio_environment.wav",
            s2a_modality,
        )
        self.assertIn("Do not judge relevance, scenario plausibility, emotional fit, or amount of detail", s2a_modality)
        self.assertIn("essentially a single spoken reply", s2a_modality)
        self.assertNotIn("INPUT SPEECH AUDIO", s2a_modality)
        self.assertNotIn("Clip 2 -", s2a_modality)
        self.assertEqual(
            s2a_modality_record["audio_attachments"],
            [{"role": "generated_audio_to_evaluate", "path": "/tmp/environment.wav"}],
        )

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
            self.assertNotIn("evaluator_version", judgment)
            self.assertEqual(judgment["prompt"]["prompt_version"], PROMPT_VERSIONS["plausibility"])
            self.assertEqual(judgment["prompt"]["audio_attachments"][0]["role"], "input_audio")
            self.assertEqual(judgment["prompt"]["audio_attachments"][0]["path"], str(source_audio))
            self.assertEqual(judgment["prompt"]["audio_attachments"][1]["role"], "generated_audio_to_evaluate")
            self.assertEqual(judgment["prompt"]["audio_attachments"][1]["path"], str(generated))
            rendered = judgment["prompt"]["rendered_user_message"]
            self.assertIn("Clip 1 - INPUT ENVIRONMENTAL AUDIO (context): attached file input_audio_audio.wav", rendered)
            self.assertIn("Clip 2 - OUTPUT SPEECH AUDIO TO EVALUATE: attached file output_audio_generated.wav", rendered)
            self.assertNotIn("A dog barks near a door.", rendered)
            self.assertNotIn("Please let the dog inside.", rendered)


if __name__ == "__main__":
    unittest.main()
