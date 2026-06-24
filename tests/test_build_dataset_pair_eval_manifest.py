from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from evaluation.io import iter_jsonl
from evaluation.metrics.registry import get_metric_specs
from evaluation.runner import EvalOptions, _dedicated_lalm_audio_rows, _dedicated_text_pair_rows
from scripts.build_dataset_pair_eval_manifest import _manifest_row


class BuildDatasetPairEvalManifestTest(unittest.TestCase):
    def test_text_a2s_manifest_feeds_geval_text_pair_builder(self) -> None:
        source = {
            "audio_file": "audio/sample.wav",
            "speech_file": "speech/sample.wav",
            "audio_cap": "A motorcycle engine is idling",
            "speech_text": "Need to adjust these mirrors before hitting the road.",
        }
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            manifest = root / "manifest.jsonl"
            row = _manifest_row(
                source=source,
                ordinal=0,
                dataset_root=root,
                stage="t2t",
                direction="a2s",
                skip_missing=False,
            )
            manifest.write_text(json.dumps(row) + "\n", encoding="utf-8")
            normalized = [item.record for item in iter_jsonl(manifest)]
            helper_rows, errors = _dedicated_text_pair_rows(
                self._options(root, manifest, ["geval_text_reference"], stage="t2t", task="T2T", direction="a2s"),
                normalized,
                "geval_text_reference",
                {},
            )
        self.assertFalse(errors)
        self.assertEqual(helper_rows[0]["direction"], "audio_to_speech")
        self.assertEqual(helper_rows[0]["input_text"], source["audio_cap"])
        self.assertEqual(helper_rows[0]["generated_text"], source["speech_text"])
        self.assertEqual(helper_rows[0]["reference_anchors"][0]["text"], source["speech_text"])

    def test_audio_s2a_manifest_feeds_geval_audio_pair_builder(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            dataset_root = root / "dataset"
            (dataset_root / "audio").mkdir(parents=True)
            (dataset_root / "speech").mkdir()
            audio = dataset_root / "audio" / "sample.wav"
            speech = dataset_root / "speech" / "sample.wav"
            audio.write_bytes(b"RIFF....WAVE")
            speech.write_bytes(b"RIFF....WAVE")
            source = {
                "audio_file": "audio/sample.wav",
                "speech_file": "speech/sample.wav",
                "audio_cap": "Rain falls on pavement",
                "speech_text": "Grab your umbrella before you leave.",
            }
            manifest = root / "manifest.jsonl"
            row = _manifest_row(
                source=source,
                ordinal=3,
                dataset_root=dataset_root,
                stage="t2a",
                direction="s2a",
                skip_missing=False,
            )
            manifest.write_text(json.dumps(row) + "\n", encoding="utf-8")
            normalized = [item.record for item in iter_jsonl(manifest)]
            helper_rows, errors = _dedicated_lalm_audio_rows(
                self._options(root, manifest, ["geval_audio_reference"], stage="t2a", task="T2A", direction="s2a"),
                normalized,
                "geval_audio_reference",
            )
        self.assertFalse(errors)
        self.assertEqual(helper_rows[0]["direction"], "speech_to_audio")
        self.assertEqual(helper_rows[0]["source_audio_file"], str(speech))
        self.assertEqual(helper_rows[0]["generated_audio_file"], str(audio))
        self.assertEqual(helper_rows[0]["reference_anchors"][0]["audio_file"], str(audio))

    def _options(
        self,
        root: Path,
        manifest: Path,
        metrics: list[str],
        *,
        stage: str,
        task: str,
        direction: str,
    ) -> EvalOptions:
        dataset_config = root / "dataset.json"
        dataset_config.write_text(json.dumps({"dataset_root": str(root / "dataset")}), encoding="utf-8")
        return EvalOptions(
            model="dataset_pair",
            task=task,
            stage=stage,
            direction=direction,
            evaluated_run_id="dataset_pair",
            dataset="ttas",
            split="test",
            manifest_path=manifest,
            dataset_config_path=dataset_config,
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


if __name__ == "__main__":
    unittest.main()
