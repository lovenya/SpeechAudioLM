from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from evaluation.capabilities import load_capabilities, validate_model_task
from evaluation.errors import EvaluationError
from evaluation.metrics.registry import get_metric_specs
from evaluation.metrics.text import word_error_rate
from evaluation.runner import EvalOptions, dry_run
from evaluation.task_metrics import load_task_metric_map, select_metrics
from project_conventions import baseline_output_root, eval_model_root


class EvaluationScaffoldTest(unittest.TestCase):
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
        with self.assertRaises(EvaluationError):
            validate_model_task("tangoflux", "T2S", caps)

    def test_word_error_rate_is_directional(self) -> None:
        self.assertEqual(word_error_rate("a b", "a b c"), 0.5)
        self.assertAlmostEqual(word_error_rate("a b c", "a b"), 1 / 3)

    def test_corpus_metrics_are_marked_by_level(self) -> None:
        specs = get_metric_specs(["fad", "kad", "kl_passt", "inception_score", "clap_score_t2a"])
        corpus_metrics = {spec.name for spec in specs if spec.level == "corpus"}
        self.assertEqual(corpus_metrics, {"fad", "kad", "kl_passt", "inception_score"})
        self.assertEqual(next(spec for spec in specs if spec.name == "clap_score_t2a").level, "sample")

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
        self.assertEqual(select_metrics("T2A", "clap_score_t2a", mapping).metric_names, ["clap_score_t2a"])
        self.assertEqual(select_metrics("S2A", "fad", mapping).metric_names, ["fad"])
        self.assertEqual(select_metrics("A2S", "asr_wer", mapping).metric_names, ["asr_wer"])
        self.assertEqual(select_metrics("A2T", "mace", mapping).metric_names, ["mace"])
        self.assertEqual(select_metrics("A2T", "aac_clapsim", mapping).metric_names, ["aac_clapsim"])

    def test_task_metric_mapping_rejects_invalid_pairs(self) -> None:
        mapping = load_task_metric_map(Path("configs/task_metric_map.json"))
        with self.assertRaises(EvaluationError):
            select_metrics("T2A", "asr_wer", mapping)
        with self.assertRaises(EvaluationError):
            select_metrics("T2A", "aac_clapsim", mapping)
        with self.assertRaises(EvaluationError):
            select_metrics("A2T", "clap_score_t2a", mapping)
        with self.assertRaises(EvaluationError):
            select_metrics("T2A", "kld", mapping)
        with self.assertRaises(EvaluationError):
            select_metrics("A2T", "clap_similarity", mapping)
        with self.assertRaises(EvaluationError):
            select_metrics("S2A", "wer", mapping)
        with self.assertRaises(EvaluationError):
            select_metrics("A2S", "mace", mapping)

    def test_task_metric_mapping_selects_defaults(self) -> None:
        mapping = load_task_metric_map(Path("configs/task_metric_map.json"))
        self.assertEqual(select_metrics("A2T", None, mapping).metric_names, ["mace", "spider", "aac_clapsim"])
        self.assertEqual(select_metrics("T2A", None, mapping).metric_names, ["fad", "kad", "inception_score"])
        self.assertEqual(select_metrics("S2A", None, mapping).metric_names, ["fad", "kad", "inception_score"])
        self.assertEqual(select_metrics("T2S", None, mapping).metric_names, ["asr_wer", "dwer", "utmos"])

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
            )
            preview = dry_run(options)
            self.assertEqual(preview["rows"], 1)


if __name__ == "__main__":
    unittest.main()
