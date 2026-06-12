from __future__ import annotations

import json
import sys
import tempfile
import unittest
from unittest.mock import patch
from pathlib import Path

from evaluation.capabilities import load_capabilities, validate_model_task
from evaluation.errors import EvaluationError
from evaluation.metric_contracts import EvaluationMode
from evaluation.metrics.registry import get_metric_specs
from evaluation.metrics.text import word_error_rate
from evaluation.runner import EvalOptions, _dedicated_script_command, _evaluation_artifact_dir, _evaluation_work_dir, _prediction_text_for_metric, _run_backend_worker, dry_run, run_evaluation
from evaluation.task_metrics import load_task_metric_map, select_metrics
from project_conventions import baseline_output_root, eval_model_root


class EvaluationScaffoldTest(unittest.TestCase):
    def test_evaluation_work_dir_uses_unique_slurm_tmpdir_workspace(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            slurm_tmpdir = Path(tmpdir) / "node-local"
            slurm_tmpdir.mkdir()
            output_dir = Path(tmpdir) / "persistent-output"
            with patch.dict("os.environ", {"SLURM_TMPDIR": str(slurm_tmpdir)}):
                first = _evaluation_work_dir(output_dir, "val", "timestamp")
                second = _evaluation_work_dir(output_dir, "val", "timestamp")

        self.assertNotEqual(first, second)
        self.assertEqual(first.parent, slurm_tmpdir / "speechaudiolm_evaluation")
        self.assertFalse(str(first).startswith(str(output_dir)))

    def test_evaluation_work_dir_falls_back_outside_slurm(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            output_dir = Path(tmpdir) / "persistent-output"
            with patch.dict("os.environ", {}, clear=True):
                work_dir = _evaluation_work_dir(output_dir, "val", "timestamp")

        self.assertEqual(work_dir, output_dir / "work_val_timestamp")

    def test_evaluation_artifact_dir_stays_on_persistent_output_root(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            output_dir = Path(tmpdir) / "persistent-output"
            artifact_dir = _evaluation_artifact_dir(output_dir, "val", "timestamp")

        self.assertEqual(artifact_dir, output_dir / "work_val_timestamp")

    def test_path_conventions(self) -> None:
        self.assertEqual(
            baseline_output_root("audio_omni", "T2A", "ttas", "test"),
            Path("results/baselines/audio_omni/T2A/ttas/test"),
        )
        self.assertEqual(
            eval_model_root("T2A", "ttas", "audio_omni"),
            Path("results/evals/T2A/ttas/audio_omni"),
        )

    def test_capability_validation(self) -> None:
        caps = load_capabilities(Path("configs/model_capabilities.json"))
        validate_model_task("tangoflux", "T2A", caps)
        validate_model_task("audio_omni", "A2T", caps)
        validate_model_task("audio_omni", "S2T", caps)
        validate_model_task("qwen3_omni", "A2S", caps)
        validate_model_task("cascaded", "T2T", caps)
        validate_model_task("qwen3_32b", "T2T", caps)
        with self.assertRaises(EvaluationError):
            validate_model_task("tangoflux", "T2S", caps)

    def test_word_error_rate_is_directional(self) -> None:
        self.assertEqual(word_error_rate("a b", "a b c"), 0.5)
        self.assertAlmostEqual(word_error_rate("a b c", "a b"), 1 / 3)

    def test_corpus_metrics_are_marked_by_level(self) -> None:
        specs = get_metric_specs([
            "fad",
            "kad",
            "kl_passt",
            "inception_score",
            "target_caption_clap_score",
            "prompt_adherence_clap_score",
        ])
        corpus_metrics = {spec.name for spec in specs if spec.level == "corpus"}
        self.assertEqual(corpus_metrics, {"fad", "kad", "kl_passt", "inception_score"})
        self.assertEqual(
            {spec.name for spec in specs if spec.level == "sample"},
            {"target_caption_clap_score", "prompt_adherence_clap_score"},
        )

    def test_aac_clapsim_validates_text_against_reference_audio(self) -> None:
        spec = get_metric_specs(["aac_clapsim"])[0]
        self.assertEqual(spec.required_source_fields, ("audio_file",))
        self.assertTrue(spec.requires_hypothesis_text)
        self.assertFalse(spec.requires_output_file)

    def test_task_metric_mapping_allows_valid_pairs(self) -> None:
        mapping = load_task_metric_map(Path("configs/task_metric_map.json"))
        self.assertEqual(select_metrics("S2T", "wer", mapping).metric_names, ["wer"])
        self.assertEqual(select_metrics("T2A", "fad", mapping).metric_names, ["fad"])
        self.assertEqual(select_metrics("T2A", "kl_passt", mapping).metric_names, ["kl_passt"])
        self.assertEqual(
            select_metrics("T2A", "target_caption_clap_score", mapping).metric_names,
            ["target_caption_clap_score"],
        )
        self.assertEqual(
            select_metrics("T2A", "prompt_adherence_clap_score", mapping).metric_names,
            ["prompt_adherence_clap_score"],
        )
        self.assertEqual(select_metrics("S2A", "fad", mapping).metric_names, ["fad"])
        self.assertEqual(select_metrics("A2S", "asr_wer", mapping).metric_names, ["asr_wer"])
        self.assertEqual(select_metrics("T2T", "nli_contradiction", mapping).metric_names, ["nli_contradiction"])
        self.assertEqual(select_metrics("T2T", "g_eval_reference_free", mapping).metric_names, ["g_eval_reference_free"])
        self.assertEqual(select_metrics("A2T", "mace", mapping).metric_names, ["mace"])
        self.assertEqual(select_metrics("A2T", "aac_clapsim", mapping).metric_names, ["aac_clapsim"])
        self.assertEqual(select_metrics("A2T", "fense", mapping).metric_names, ["fense"])

    def test_task_metric_mapping_rejects_invalid_pairs(self) -> None:
        mapping = load_task_metric_map(Path("configs/task_metric_map.json"))
        with self.assertRaises(EvaluationError):
            select_metrics("T2A", "asr_wer", mapping)
        with self.assertRaises(EvaluationError):
            select_metrics("T2A", "aac_clapsim", mapping)
        with self.assertRaises(EvaluationError):
            select_metrics("A2T", "target_caption_clap_score", mapping)
        with self.assertRaises(EvaluationError):
            select_metrics("A2T", "prompt_adherence_clap_score", mapping)
        with self.assertRaises(EvaluationError):
            select_metrics("T2A", "clap_score_t2a", mapping)
        with self.assertRaises(EvaluationError):
            select_metrics("T2A", "kld", mapping)
        with self.assertRaises(EvaluationError):
            select_metrics("A2T", "clap_similarity", mapping)
        with self.assertRaises(EvaluationError):
            select_metrics("S2A", "wer", mapping)
        with self.assertRaises(EvaluationError):
            select_metrics("A2S", "mace", mapping)
        with self.assertRaises(EvaluationError):
            select_metrics("S2A", "nli_contradiction", mapping)
        with self.assertRaises(EvaluationError):
            select_metrics("A2S", "nli_contradiction", mapping)

    def test_task_metric_mapping_selects_defaults(self) -> None:
        mapping = load_task_metric_map(Path("configs/task_metric_map.json"))
        self.assertEqual(select_metrics("A2T", None, mapping).metric_names, ["mace", "spider", "aac_clapsim"])
        self.assertEqual(select_metrics("T2A", None, mapping).metric_names, ["fad", "kad", "inception_score"])
        self.assertEqual(select_metrics("S2A", None, mapping).metric_names, ["fad", "kad", "inception_score"])
        self.assertEqual(select_metrics("T2S", None, mapping).metric_names, ["asr_wer", "dwer", "utmos"])
        self.assertEqual(select_metrics("T2T", None, mapping).metric_names, ["nli_contradiction"])

    def test_prompt_adherence_uses_only_exact_persisted_generation_prompt(self) -> None:
        row = {
            "prompt": "  exact conditioning prompt  ",
            "candidate_text": "wrong fallback",
            "model_extra": {"generation_prompt": "also wrong"},
        }
        self.assertEqual(
            _prediction_text_for_metric(row, "prompt_adherence_clap_score"),
            "  exact conditioning prompt  ",
        )
        self.assertIsNone(
            _prediction_text_for_metric(
                {"candidate_text": "must not be used"},
                "prompt_adherence_clap_score",
            )
        )

    def test_prompt_adherence_backend_payload_has_explicit_conditioning_prompt(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            captured = {}

            def fake_run(command, **kwargs):
                input_path = Path(command[command.index("--input") + 1])
                output_path = Path(command[command.index("--output") + 1])
                captured.update(json.loads(input_path.read_text(encoding="utf-8")))
                output_path.write_text(
                    json.dumps({
                        "sample_metrics": {},
                        "corpus_metrics": {},
                        "predictions": {},
                        "errors": {},
                        "metric_backends": {
                            "prompt_adherence_clap_score": {}
                        },
                    }),
                    encoding="utf-8",
                )
                return type("Completed", (), {"returncode": 0, "stdout": ""})()

            options = EvalOptions(
                model="cascaded",
                task="T2A",
                dataset="ttas",
                split="val",
                manifest_path=root / "manifest.jsonl",
                dataset_config_path=root / "dataset.json",
                metrics_config_path=root / "metrics.json",
                task_metric_map_path=root / "task_metric_map.json",
                output_root=root / "evals",
                metric_specs=get_metric_specs(["prompt_adherence_clap_score"]),
                max_samples=1,
                start_index=0,
                cache_dir=root / "cache",
                device="cpu",
                command=["evaluate.py"],
                evaluation_mode=EvaluationMode.BACKEND_SMOKE,
            )
            rows = [{
                "row_index": 0,
                "sample_id": "sample-a",
                "status": "success",
                "candidate_text": "must not become the prompt",
                "prompt": "  exact persisted prompt  ",
                "output_file": str(root / "generated.wav"),
                "source": {"audio_cap": "dataset target caption"},
            }]
            metrics_config = {
                "prompt_adherence_clap_score": {
                    "python": sys.executable,
                    "cache_dir": str(root / "cache"),
                }
            }

            with patch("evaluation.runner.subprocess.run", side_effect=fake_run):
                _run_backend_worker(
                    options,
                    rows,
                    root,
                    metrics_config,
                    root / "work",
                    root / "artifacts",
                    "stable-audio-metrics",
                    ["prompt_adherence_clap_score"],
                )

            backend_row = captured["rows"][0]
            self.assertEqual(backend_row["conditioning_prompt"], "  exact persisted prompt  ")
            self.assertEqual(backend_row["candidate_text"], "  exact persisted prompt  ")
            self.assertNotEqual(backend_row["conditioning_prompt"], rows[0]["candidate_text"])

    def test_metric_specific_validation_does_not_require_unused_fields(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            dataset_root = root / "dataset"
            baseline_root = root / "baseline"
            ref_audio = dataset_root / "audio" / "sample.wav"
            output_audio = baseline_root / "audio" / "sample.wav"
            ref_audio.parent.mkdir(parents=True)
            output_audio.parent.mkdir(parents=True)
            ref_audio.write_bytes(b"RIFFfake")
            output_audio.write_bytes(b"RIFFfake")

            dataset_config = root / "dataset.json"
            dataset_config.write_text(
                json.dumps({"dataset_root": str(dataset_root)}),
                encoding="utf-8",
            )
            metrics_config = root / "metrics.json"
            metrics_config.write_text(json.dumps({"metrics": {}}), encoding="utf-8")
            manifest = root / "manifest.jsonl"
            manifest.write_text(
                json.dumps(
                    {
                        "model": "audio_omni",
                        "task": "T2A",
                        "row_index": 0,
                        "sample_id": "sample",
                        "output_file": str(output_audio),
                        "source": {
                            "audio_file": "audio/sample.wav",
                            "audio_cap": "A tone plays",
                        },
                    }
                )
                + "\n",
                encoding="utf-8",
            )

            options = EvalOptions(
                model="audio_omni",
                task="T2A",
                dataset="ttas",
                split="test",
                manifest_path=manifest,
                dataset_config_path=dataset_config,
                metrics_config_path=metrics_config,
                task_metric_map_path=root / "task_metric_map.json",
                output_root=root / "evals",
                metric_specs=get_metric_specs(["fad"]),
                max_samples=None,
                start_index=0,
                cache_dir=root / "cache",
                device="cpu",
                command=["evaluate.py"],
                evaluation_mode=EvaluationMode.BACKEND_SMOKE,
            )
            preview = dry_run(options)
            self.assertEqual(preview["rows"], 1)

    def test_corpus_audio_metrics_use_preserved_original_audio_when_stage_path_is_missing(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            dataset_root = root / "dataset"
            baseline_root = root / "baseline"
            ref_audio = dataset_root / "audio" / "sample.wav"
            output_audio = baseline_root / "audio" / "sample.wav"
            ref_audio.parent.mkdir(parents=True)
            output_audio.parent.mkdir(parents=True)
            ref_audio.write_bytes(b"RIFFfake")
            output_audio.write_bytes(b"RIFFfake")

            dataset_config = root / "dataset.json"
            dataset_config.write_text(json.dumps({"dataset_root": str(dataset_root)}), encoding="utf-8")
            metrics_config = root / "metrics.json"
            metrics_config.write_text(json.dumps({"metrics": {}}), encoding="utf-8")
            manifest = root / "manifest.jsonl"
            manifest.write_text(
                json.dumps(
                    {
                        "model": "cascaded",
                        "task": "T2A",
                        "row_index": 0,
                        "sample_id": "sample",
                        "output_file": str(output_audio),
                        "source": {
                            "audio_file": "stagewise_demo_ids/sample.wav",
                            "audio_cap": "A tone plays",
                            "original_row": {
                                "_demo_source": {
                                    "audio_file": "audio/sample.wav",
                                }
                            },
                        },
                    }
                )
                + "\n",
                encoding="utf-8",
            )

            options = EvalOptions(
                model="cascaded",
                task="T2A",
                dataset="ttas",
                split="test",
                manifest_path=manifest,
                dataset_config_path=dataset_config,
                metrics_config_path=metrics_config,
                task_metric_map_path=root / "task_metric_map.json",
                output_root=root / "evals",
                metric_specs=get_metric_specs(["fad"]),
                max_samples=None,
                start_index=0,
                cache_dir=root / "cache",
                device="cpu",
                command=["evaluate.py"],
                evaluation_mode=EvaluationMode.BACKEND_SMOKE,
            )
            preview = dry_run(options)
            self.assertEqual(preview["rows"], 1)

    def test_eval_dry_run_supports_start_index(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            dataset_root = root / "dataset"
            baseline_root = root / "baseline"
            ref_audio = dataset_root / "audio" / "sample.wav"
            output_audio = baseline_root / "audio" / "sample.wav"
            ref_audio.parent.mkdir(parents=True)
            output_audio.parent.mkdir(parents=True)
            ref_audio.write_bytes(b"RIFFfake")
            output_audio.write_bytes(b"RIFFfake")

            dataset_config = root / "dataset.json"
            dataset_config.write_text(json.dumps({"dataset_root": str(dataset_root)}), encoding="utf-8")
            metrics_config = root / "metrics.json"
            metrics_config.write_text(json.dumps({"metrics": {}}), encoding="utf-8")
            manifest = root / "manifest.jsonl"
            rows = [
                {
                    "model": "audio_omni",
                    "task": "T2A",
                    "row_index": index,
                    "sample_id": f"sample_{index}",
                    "output_file": str(output_audio),
                    "source": {"audio_file": "audio/sample.wav", "audio_cap": "A tone plays"},
                }
                for index in range(3)
            ]
            manifest.write_text("\n".join(json.dumps(row) for row in rows) + "\n", encoding="utf-8")

            options = EvalOptions(
                model="audio_omni",
                task="T2A",
                dataset="ttas",
                split="test",
                manifest_path=manifest,
                dataset_config_path=dataset_config,
                metrics_config_path=metrics_config,
                task_metric_map_path=root / "task_metric_map.json",
                output_root=root / "evals",
                metric_specs=get_metric_specs(["fad"]),
                max_samples=1,
                start_index=2,
                cache_dir=root / "cache",
                device="cpu",
                command=["evaluate.py"],
                evaluation_mode=EvaluationMode.BACKEND_SMOKE,
            )
            preview = dry_run(options)
            self.assertEqual(preview["rows"], 1)

    def test_nli_dedicated_command_uses_local_files_only_for_offline_config(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            options = EvalOptions(
                model="cascaded",
                task="T2T",
                dataset="ttas",
                split="val",
                manifest_path=root / "manifest.jsonl",
                dataset_config_path=root / "dataset.json",
                metrics_config_path=root / "metrics.json",
                task_metric_map_path=root / "task_metric_map.json",
                output_root=root / "evals",
                metric_specs=get_metric_specs(["nli_contradiction"]),
                max_samples=None,
                start_index=0,
                cache_dir=root / "cache",
                device="cuda",
                command=["evaluate.py"],
                evaluation_mode=EvaluationMode.BACKEND_SMOKE,
            )

            command = _dedicated_script_command(
                "nli_contradiction",
                Path("python"),
                Path("scripts/run_nli_contradiction_helper.py"),
                root / "input.jsonl",
                root / "output.jsonl",
                root / "summary.json",
                root / "work",
                {"hf_hub_offline": "1", "transformers_offline": "1"},
                options,
            )

            self.assertIn("--local-files-only", command)

    def test_nli_contradiction_runs_as_dedicated_script_metric(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            dataset_root = root / "dataset"
            output_root = root / "evals"
            dataset_root.mkdir()
            dataset_config = root / "dataset.json"
            dataset_config.write_text(json.dumps({"dataset_root": str(dataset_root)}), encoding="utf-8")
            metrics_config = root / "metrics.json"
            metrics_config.write_text(
                json.dumps(
                    {
                        "metrics": {
                            "nli_contradiction": {
                                "backend": "dedicated-script",
                                "script": "scripts/run_nli_contradiction_helper.py",
                                "script_backend": "stub",
                                "contradiction_threshold": 0.7,
                            }
                        }
                    }
                ),
                encoding="utf-8",
            )
            manifest = root / "manifest.jsonl"
            manifest.write_text(
                json.dumps(
                    {
                        "model": "cascaded",
                        "task": "T2T",
                        "pipeline": "S2A",
                        "row_index": 0,
                        "sample_id": "sample",
                        "generated_text": "A dog is barking loudly outside.",
                        "source": {
                            "speech_text": "Can someone close the window?",
                            "primary_audio_class": "domestic_pets",
                        },
                    }
                )
                + "\n",
                encoding="utf-8",
            )
            options = EvalOptions(
                model="cascaded",
                task="T2T",
                dataset="ttas",
                split="val",
                manifest_path=manifest,
                dataset_config_path=dataset_config,
                metrics_config_path=metrics_config,
                task_metric_map_path=root / "task_metric_map.json",
                output_root=output_root,
                metric_specs=get_metric_specs(["nli_contradiction"]),
                max_samples=None,
                start_index=0,
                cache_dir=root / "cache",
                device="cpu",
                command=["evaluate.py"],
                evaluation_mode=EvaluationMode.BACKEND_SMOKE,
            )

            paths = run_evaluation(options)
            per_sample = [json.loads(line) for line in paths["per_sample"].read_text(encoding="utf-8").splitlines()]
            self.assertIn("nli_contradiction", per_sample[0]["metrics"])
            self.assertIn("nli_contradiction", per_sample[0]["predictions"])
            self.assertEqual(per_sample[0]["predictions"]["nli_contradiction"]["direction"], "speech_to_audio")
            summary = json.loads(paths["summary"].read_text(encoding="utf-8"))
            self.assertIn("nli_contradiction", summary["metric_summaries"])


    def test_g_eval_reference_free_runs_after_nli_when_selected(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            dataset_root = root / "dataset"
            output_root = root / "evals"
            dataset_root.mkdir()
            dataset_config = root / "dataset.json"
            dataset_config.write_text(json.dumps({"dataset_root": str(dataset_root)}), encoding="utf-8")
            metrics_config = root / "metrics.json"
            metrics_config.write_text(
                json.dumps(
                    {
                        "metrics": {
                            "nli_contradiction": {
                                "backend": "dedicated-script",
                                "script": "scripts/run_nli_contradiction_helper.py",
                                "script_backend": "stub",
                                "contradiction_threshold": 0.7,
                            },
                            "g_eval_reference_free": {
                                "backend": "dedicated-script",
                                "script": "scripts/run_g_eval_reference_free.py",
                                "script_backend": "stub",
                                "batch_size": 1,
                            },
                        }
                    }
                ),
                encoding="utf-8",
            )
            manifest = root / "manifest.jsonl"
            manifest.write_text(
                json.dumps(
                    {
                        "model": "cascaded",
                        "task": "T2T",
                        "pipeline": "A2S",
                        "row_index": 0,
                        "sample_id": "sample",
                        "generated_text": "Can someone close the window?",
                        "source": {
                            "audio_cap": "A dog is barking loudly outside.",
                            "primary_audio_class": "domestic_pets",
                        },
                    }
                )
                + "\n",
                encoding="utf-8",
            )
            options = EvalOptions(
                model="cascaded",
                task="T2T",
                dataset="ttas",
                split="val",
                manifest_path=manifest,
                dataset_config_path=dataset_config,
                metrics_config_path=metrics_config,
                task_metric_map_path=root / "task_metric_map.json",
                output_root=output_root,
                metric_specs=get_metric_specs(["nli_contradiction", "g_eval_reference_free"]),
                max_samples=None,
                start_index=0,
                cache_dir=root / "cache",
                device="cpu",
                command=["evaluate.py"],
                evaluation_mode=EvaluationMode.BACKEND_SMOKE,
            )

            paths = run_evaluation(options)
            per_sample = [json.loads(line) for line in paths["per_sample"].read_text(encoding="utf-8").splitlines()]
            self.assertIn("nli_contradiction", per_sample[0]["metrics"])
            self.assertIn("g_eval_reference_free", per_sample[0]["metrics"])
            g_eval = per_sample[0]["predictions"]["g_eval_reference_free"]
            self.assertEqual(g_eval["direction"], "audio_to_speech")
            self.assertNotIn("nli_contradiction", g_eval)


if __name__ == "__main__":
    unittest.main()

class WrappedCascadedManifestTest(unittest.TestCase):
    def test_evaluation_reader_normalizes_wrapped_sample(self) -> None:
        from evaluation.io import _normalize_generation_record

        wrapped = {
            "schema_version": "1.0",
            "current_stage": 3,
            "lane": "full_a2s_afn_qwen32_f5_tts",
            "pipeline": "A2S",
            "dataset": "ttas",
            "split": "val",
            "config_file": "configs/example.json",
            "sample": {
                "row_index": 4,
                "sample_id": "sample",
                "status": "success",
                "reference": {
                    "audio_file": "audio/sample.wav",
                    "audio_caption": "A motorcycle starts",
                    "speech_file": "speech/sample.wav",
                    "speech_text": "Start already.",
                },
                "stages": [
                    {
                        "stage": 3,
                        "task": "text_to_speech",
                        "model": "f5_tts",
                        "input": {"type": "speech_text", "text": "Come on, start."},
                        "prompt": None,
                        "inference_seconds": 3.4,
                        "output": {
                            "type": "speech_audio",
                            "file": "audio/generated.wav",
                            "sample_rate": 24000,
                            "duration_seconds": 4.2,
                        },
                        "error": None,
                    }
                ],
                "final_output": {"type": "speech_audio", "file": "audio/generated.wav"},
                "error": None,
            },
        }
        row = _normalize_generation_record(wrapped)
        self.assertEqual(row["task"], "T2S")
        self.assertEqual(row["model"], "f5_tts")
        self.assertEqual(row["output_file"], "audio/generated.wav")
        self.assertEqual(row["source"]["audio_cap"], "A motorcycle starts")
        self.assertEqual(row["prompt"], "Come on, start.")
        self.assertEqual(row["generation_seconds"], 3.4)
        from evaluation.runner import _predictions
        self.assertEqual(_predictions(row, []), {"generated_speech": "audio/generated.wav"})
