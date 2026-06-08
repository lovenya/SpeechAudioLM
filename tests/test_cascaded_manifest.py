from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from baselines.cascaded_manifest import build_sample, unwrap_sample, wrap_sample, write_inspection_manifest


class CascadedManifestTest(unittest.TestCase):
    def setUp(self) -> None:
        self.source = {
            "audio_file": "audio/sample.wav",
            "audio_cap": "A motorcycle starts",
            "speech_file": "speech/sample.wav",
            "speech_text": "Let's see if it starts.",
            "retained_audio_tags": ["Vehicle", "Motorcycle"],
            "retained_audio_tag_scores": {"Vehicle": 0.8, "Motorcycle": 0.7},
            "primary_audio_tag": "Vehicle",
            "primary_audio_class": "road_vehicle_traffic",
            "audio_to_speech": {"reference_pool_class": "road_vehicle_traffic"},
            "speech_to_audio": {"reference_pool_class": "road_vehicle_traffic"},
        }

    def test_cumulative_sample_preserves_reference_and_stage_chain(self) -> None:
        stage1 = {
            "stage": 1,
            "task": "audio_to_text",
            "model": "audio_flamingo_next",
            "input": {"type": "audio", "file": "audio/sample.wav"},
            "prompt": {"messages": [{"role": "user", "content": "Generate a caption for the input audio."}]},
            "inference_seconds": 1.2,
            "output": {"type": "audio_caption", "text": "A motorcycle engine starts."},
            "error": None,
        }
        sample1 = build_sample(
            row_index=0,
            sample_id="sample",
            record=self.source,
            stage=stage1,
            status="success",
        )
        stage2 = {
            "stage": 2,
            "task": "text_to_text",
            "model": "qwen3_32b",
            "mode": "audio_description_to_speech",
            "input": {"type": "audio_caption", "text": stage1["output"]["text"]},
            "prompt": {"messages": [{"role": "user", "content": "full prompt"}]},
            "inference_seconds": 2.1,
            "output": {"type": "speech_text", "text": "Come on, start."},
            "error": None,
        }
        sample2 = build_sample(
            row_index=0,
            sample_id="sample",
            record={"_cascaded_sample": sample1},
            stage=stage2,
            status="success",
        )

        self.assertEqual(sample2["reference"]["speech_text"], "Let's see if it starts.")
        self.assertEqual(sample2["reference"]["audio_metadata"]["primary_audio_class"], "road_vehicle_traffic")
        self.assertEqual(sample2["stages"][0]["output"], sample2["stages"][1]["input"])
        self.assertEqual(sample2["stages"][1]["inference_seconds"], 2.1)
        self.assertIsNone(sample2["final_output"])

    def test_jsonl_and_inspection_json_share_identical_sample(self) -> None:
        stage = {
            "stage": 1,
            "task": "speech_to_text",
            "model": "whisper",
            "input": {"type": "speech_audio", "file": "speech/sample.wav"},
            "prompt": None,
            "inference_seconds": 0.8,
            "output": {"type": "speech_text", "text": "Let's see if it starts."},
            "error": None,
        }
        sample = build_sample(row_index=0, sample_id="sample", record=self.source, stage=stage, status="success")
        metadata = {
            "lane": "full_s2a_whisper_qwen32_audiogen",
            "pipeline": "S2A",
            "dataset": "ttas",
            "split": "val",
            "config_file": "configs/example.json",
        }
        wrapped = wrap_sample(metadata, sample)
        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "manifest.inspect.json"
            write_inspection_manifest(
                path,
                metadata=metadata,
                stage_definitions=[],
                samples=[sample],
                created_at="2026-06-07T10:00:00-04:00",
                started_at="2026-06-07T10:00:01-04:00",
                completed_at="2026-06-07T10:00:02-04:00",
                status="completed",
            )
            inspection = json.loads(path.read_text(encoding="utf-8"))

        self.assertEqual(wrapped["current_stage"], 1)
        self.assertEqual(unwrap_sample(wrapped), inspection["samples"][0])
        self.assertEqual(inspection["summary"], {"total": 1, "success": 1, "failed": 0})

    def test_nonconsecutive_stage_is_rejected(self) -> None:
        with self.assertRaisesRegex(ValueError, "does not follow"):
            build_sample(
                row_index=0,
                sample_id="sample",
                record=self.source,
                stage={"stage": 2, "task": "text_to_text", "output": None},
                status="success",
            )


if __name__ == "__main__":
    unittest.main()

class FullPipelineFailureTest(unittest.TestCase):
    def test_a2s_stage_two_failure_preserves_completed_stage(self) -> None:
        import argparse
        from baseline_cascaded import ModularCascade
        from baselines.dataset import PromptRow

        args = argparse.Namespace(
            pipeline="A2S",
            captioner="audio_flamingo_next",
            text_model="qwen3_32b",
            text_mode="speech_to_audio_description",
            tts="f5_tts",
            asr="whisper",
            t2a="audiogen",
            prompt_template=None,
        )
        cascade = ModularCascade(args, dataset_root=None)
        cascade._caption_audio = lambda row, root: "A motorcycle engine starts."
        cascade._convert_text = lambda *args, **kwargs: (_ for _ in ()).throw(RuntimeError("text failure"))
        row = PromptRow(
            row_index=0,
            sample_id="sample",
            prompt="reference caption",
            record={"audio_file": "audio/sample.wav", "speech_file": "speech/sample.wav"},
        )
        with tempfile.TemporaryDirectory() as tmpdir:
            with self.assertRaisesRegex(RuntimeError, "text failure"):
                cascade._run_full_sample(row, Path(tmpdir), Path(tmpdir) / "out.wav")
        failed = cascade._full_error_sample(row, 0.5, RuntimeError("text failure"))

        self.assertEqual(failed["status"], "error")
        self.assertEqual(len(failed["stages"]), 2)
        self.assertEqual(failed["stages"][0]["output"], failed["stages"][1]["input"])
        self.assertEqual(failed["stages"][1]["mode"], "audio_description_to_speech")
        self.assertIn("Audio description", failed["stages"][1]["prompt"]["messages"][1]["content"])
