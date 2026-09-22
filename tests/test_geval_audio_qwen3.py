from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from evaluation.metrics.registry import get_metric_specs
from evaluation.runner import EvalOptions, _dedicated_lalm_audio_rows, _dedicated_script_command, run_evaluation
from scripts.run_geval_audio_qwen3_criterion import (
    CRITERION_WEIGHTS,
    PROMPT_VERSIONS,
    SCORING_POLICY_VERSION,
    StubJudge,
    build_prompt_record,
    run_geval,
)


class GEvalLALMQwen3Test(unittest.TestCase):
    def _options(
        self,
        root: Path,
        metrics: list[str],
        *,
        stage: str = "t2s",
        task: str = "T2S",
        criteria: tuple[str, ...] | None = None,
    ) -> EvalOptions:
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
            criteria=criteria,
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
        self.assertEqual(
            prompt_record["prompt_version"],
            "geval_audio_plausibility_v9_criterion_evidence_boundaries",
        )

        self.assertIn("Clip 1 - INPUT SPEECH AUDIO (context): attached file input_audio_source.wav", prompt)
        self.assertIn("Clip 2 - OUTPUT ENVIRONMENTAL AUDIO TO EVALUATE: attached file output_audio_generated.wav", prompt)
        self.assertIn("output environmental audio should be an audio scene", prompt)
        self.assertIn("Criterion: input_speech_and_output_environmental_audio_grounded_scenario_plausibility", prompt)
        self.assertIn("Internal criterion id: scenario_plausibility", prompt)
        self.assertIn("Required independent-listening procedure", prompt)
        self.assertIn("Clip 1 evidence: ...; Clip 2 evidence: ...; Connection: ...", prompt)
        self.assertIn("Do not let the content of either clip overwrite", prompt)
        self.assertIn("inventory all salient audible evidence", prompt)
        self.assertIn("Do not cherry-pick one vaguely compatible cue", prompt)
        self.assertIn("Broad attribute overlap", prompt)
        self.assertIn("share the same specific core situation", prompt)
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
        self.assertEqual(
            s2a_modality_record["prompt_version"],
            "geval_audio_plausibility_v9_criterion_evidence_boundaries",
        )
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

        a2s_detail = build_prompt_record(
            variant="plausibility",
            direction="audio_to_speech",
            criterion="controlled_detail",
            source_audio_file="/tmp/source.wav",
            generated_audio_file="/tmp/speech.wav",
        )["rendered_user_message"]
        self.assertIn("Required independent-evidence check", a2s_detail)
        self.assertIn("distinguishing supported detail, reasonable inference", a2s_detail)
        self.assertIn("Do not rescore overall scenario plausibility", a2s_detail)

        a2s_delivery = build_prompt_record(
            variant="plausibility",
            direction="audio_to_speech",
            criterion="expressive_delivery_fit",
            source_audio_file="/tmp/source.wav",
            generated_audio_file="/tmp/speech.wav",
        )["rendered_user_message"]
        self.assertIn("actually audible delivery cues", a2s_delivery)
        self.assertIn("Judge audible delivery fit", a2s_delivery)

        s2a_detail = build_prompt_record(
            variant="plausibility",
            direction="speech_to_audio",
            criterion="controlled_detail",
            source_audio_file="/tmp/speech.wav",
            generated_audio_file="/tmp/environment.wav",
        )["rendered_user_message"]
        self.assertIn("inventory the audible events and layers", s2a_detail)
        self.assertIn("unsupported acoustic detail", s2a_detail)

        s2a_fit = build_prompt_record(
            variant="plausibility",
            direction="speech_to_audio",
            criterion="contextual_acoustic_fit",
            source_audio_file="/tmp/speech.wav",
            generated_audio_file="/tmp/environment.wav",
        )["rendered_user_message"]
        self.assertIn("acoustic character actually audible", s2a_fit)
        self.assertIn("not sound-source identity", s2a_fit)
        self.assertNotIn("Required independent-evidence check", a2s_quality)
        self.assertNotIn("Required independent-evidence check", s2a_modality)

    def test_lalm_audio_disable_cache_omits_cache_argument(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            command = _dedicated_script_command(
                "geval_audio_plausibility",
                Path("python"),
                Path("scripts/run_geval_audio_qwen3_criterion.py"),
                root / "input.jsonl",
                root / "output.jsonl",
                root / "summary.json",
                root / "work",
                {"script_backend": "stub", "disable_cache": True},
                self._options(root, ["geval_audio_plausibility"]),
            )
            self.assertNotIn("--cache-dir", command)

            cached_command = _dedicated_script_command(
                "geval_audio_plausibility",
                Path("python"),
                Path("scripts/run_geval_audio_qwen3_criterion.py"),
                root / "input.jsonl",
                root / "output.jsonl",
                root / "summary.json",
                root / "work",
                {"script_backend": "stub"},
                self._options(root, ["geval_audio_plausibility"]),
            )
            self.assertIn("--cache-dir", cached_command)

            scenario_command = _dedicated_script_command(
                "geval_audio_plausibility",
                Path("python"),
                Path("scripts/run_geval_audio_qwen3_criterion.py"),
                root / "input.jsonl",
                root / "output.jsonl",
                root / "summary.json",
                root / "work",
                {"script_backend": "stub", "disable_cache": True},
                self._options(
                    root,
                    ["geval_audio_plausibility"],
                    criteria=("scenario_plausibility",),
                ),
            )
            criteria_index = scenario_command.index("--criteria")
            self.assertEqual(scenario_command[criteria_index + 1], "scenario_plausibility")

    def test_audio_geval_can_select_only_scenario_plausibility(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            generated = root / "generated.wav"
            source_audio = root / "source.wav"
            generated.write_bytes(b"RIFF....WAVE")
            source_audio.write_bytes(b"RIFF....WAVE")
            input_path = root / "input.jsonl"
            output_path = root / "output.jsonl"
            summary_path = root / "summary.json"
            input_path.write_text(json.dumps({
                "row_index": 0,
                "sample_id": "sample",
                "split": "test",
                "direction": "audio_to_speech",
                "generated_audio_file": str(generated),
                "generated_audio_kind": "speech",
                "source_audio_file": str(source_audio),
            }) + "\n", encoding="utf-8")

            run_geval(
                input_path=input_path,
                output_path=output_path,
                summary_output=summary_path,
                summary_csv_output=None,
                judge=StubJudge(),
                variant="plausibility",
                cache_dir=None,
                batch_size=1,
                criteria=["scenario_plausibility"],
            )

            row = json.loads(output_path.read_text(encoding="utf-8"))
            self.assertEqual(list(row["criteria"]), ["scenario_plausibility"])
            self.assertEqual(row["criterion_weights"], {"scenario_plausibility": 1.0})
            self.assertEqual(row["final_score"], 4.0)

    def test_audio_geval_resumes_from_partial_output(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            generated = root / "generated.wav"
            source_audio = root / "source.wav"
            generated.write_bytes(b"RIFF....WAVE")
            source_audio.write_bytes(b"RIFF....WAVE")
            input_path = root / "input.jsonl"
            output_path = root / "output.jsonl"
            summary_path = root / "summary.json"
            row = {
                "row_index": 0,
                "pair_id": "sample",
                "sample_id": "sample",
                "split": "val",
                "direction": "audio_to_speech",
                "generated_audio_file": str(generated),
                "generated_audio_kind": "speech",
                "source_audio_file": str(source_audio),
            }
            input_path.write_text(json.dumps(row) + "\n", encoding="utf-8")
            partial = {
                "schema_version": 2,
                "metric": "geval_audio_plausibility",
                "evaluation_variant": "plausibility",
                "row_index": 0,
                "pair_id": "sample",
                "sample_id": "sample",
                "split": "val",
                "direction": "audio_to_speech",
                "primary_audio_class": None,
                "input_text": None,
                "generated_text": None,
                "generated_audio_file": str(generated),
                "generated_audio_kind": "speech",
                "source_audio_file": str(source_audio),
                "criteria": {
                    "scenario_plausibility": {
                        "score": 5,
                        "feedback": "Already judged.",
                        "parse_status": "parsed",
                        "parse_repair_attempted": False,
                        "parse_repair_succeeded": False,
                        "parse_repair_method": None,
                        "raw_response": "Feedback: Already judged. [RESULT] 5",
                    }
                },
                "criterion_weights": CRITERION_WEIGHTS[("plausibility", "audio_to_speech")],
                "final_score": None,
                "scoring_policy_version": SCORING_POLICY_VERSION,
                "prompt_version": PROMPT_VERSIONS["plausibility"],
                "judge_model": StubJudge.name,
                "status": "pending",
                "diagnostics": {"failure_modes": [], "failure_mode_source": "deterministic"},
            }
            output_path.write_text(json.dumps(partial) + "\n", encoding="utf-8")

            summary = run_geval(
                input_path=input_path,
                output_path=output_path,
                summary_output=summary_path,
                summary_csv_output=None,
                judge=StubJudge(),
                variant="plausibility",
                cache_dir=None,
                batch_size=20,
                criterion_judgments_output=root / "criterion_judgments.jsonl",
                resume_from_output=True,
            )

            self.assertTrue(summary["complete"])
            resumed = json.loads(output_path.read_text(encoding="utf-8"))
            self.assertEqual(resumed["status"], "complete")
            self.assertEqual(resumed["criteria"]["scenario_plausibility"]["score"], 5)
            self.assertEqual(resumed["criteria"]["controlled_detail"]["score"], 4)
            self.assertAlmostEqual(resumed["final_score"], 4.2)

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
