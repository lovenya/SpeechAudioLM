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
        with self.assertRaises(EvaluationError):
            validate_model_task("tangoflux", "T2S", caps)

    def test_word_error_rate_is_directional(self) -> None:
        self.assertEqual(word_error_rate("a b", "a b c"), 0.5)
        self.assertAlmostEqual(word_error_rate("a b c", "a b"), 1 / 3)

    def test_corpus_only_metrics_are_marked(self) -> None:
        specs = get_metric_specs(["fad", "kad", "kld", "inception_score", "clap"])
        corpus_only = {spec.name for spec in specs if spec.corpus_only}
        self.assertEqual(corpus_only, {"fad", "kad", "kld", "inception_score"})
        self.assertFalse(next(spec for spec in specs if spec.name == "clap").corpus_only)

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
                output_root=root / "evals",
                metric_specs=get_metric_specs(["fad"]),
                max_samples=None,
                cache_dir=root / "cache",
                device="cpu",
                command=["evaluate.py"],
            )
            preview = dry_run(options)
            self.assertEqual(preview["rows"], 1)


if __name__ == "__main__":
    unittest.main()
