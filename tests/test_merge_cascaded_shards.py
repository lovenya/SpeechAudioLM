from __future__ import annotations

import json
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from baselines.cascaded_manifest import build_sample, wrap_sample, write_inspection_manifest
from scripts.merge_cascaded_shards import main


class MergeCascadedShardsTest(unittest.TestCase):
    def test_merges_final_audio_and_rewrites_nested_output_paths(self) -> None:
        metadata = {
            "lane": "shard_lane",
            "pipeline": "A2S",
            "dataset": "ttas",
            "split": "val",
            "config_file": "configs/example.json",
        }
        definition = {
            "stage": 3,
            "task": "text_to_speech",
            "model": "f5_tts",
            "checkpoint": "F5-TTS",
            "dtype": "float16",
            "prompt_template": None,
            "generation": {"fix_duration_seconds": 10.0},
            "runtime": {"device": "cuda", "model_load_seconds": 2.0},
        }
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            inputs = []
            for index in (0, 1):
                lane = root / f"shard_{index}"
                audio = lane / "audio" / f"{index:06d}_{index}.wav"
                audio.parent.mkdir(parents=True)
                audio.write_bytes(b"RIFFfake")
                prior = build_sample(
                    row_index=index,
                    sample_id=str(index),
                    record={"speech_file": f"speech/{index}.wav", "speech_text": f"reference {index}"},
                    stage={
                        "stage": 1,
                        "task": "audio_to_text",
                        "model": "audio_flamingo_next",
                        "input": {"type": "audio", "file": f"audio/{index}.wav"},
                        "prompt": None,
                        "inference_seconds": 0.2,
                        "output": {"type": "audio_caption", "text": "engine"},
                        "error": None,
                    },
                    status="success",
                )
                prior = build_sample(
                    row_index=index,
                    sample_id=str(index),
                    record={"_cascaded_sample": prior},
                    stage={
                        "stage": 2,
                        "task": "text_to_text",
                        "model": "qwen3_32b",
                        "mode": "audio_description_to_speech",
                        "input": {"type": "audio_caption", "text": "engine"},
                        "prompt": None,
                        "inference_seconds": 0.3,
                        "output": {"type": "speech_text", "text": "Start now."},
                        "error": None,
                    },
                    status="success",
                )
                sample = build_sample(
                    row_index=index,
                    sample_id=str(index),
                    record={"_cascaded_sample": prior},
                    stage={
                        "stage": 3,
                        "task": "text_to_speech",
                        "model": "f5_tts",
                        "input": {"type": "speech_text", "text": "Start now."},
                        "prompt": None,
                        "inference_seconds": 0.4,
                        "output": {"type": "speech_audio", "file": str(audio), "sample_rate": 24000},
                        "error": None,
                    },
                    status="success",
                )
                (lane / "manifest.jsonl").write_text(json.dumps(wrap_sample(metadata, sample)) + "\n", encoding="utf-8")
                write_inspection_manifest(
                    lane / "manifest.inspect.json",
                    metadata=metadata,
                    stage_definitions=[definition],
                    samples=[sample],
                    created_at="t",
                    started_at="t",
                    completed_at="t",
                    status="completed",
                )
                inputs.append(lane)
            output = root / "canonical"
            old_audio = output / "audio" / "000000_0.wav"
            old_audio.parent.mkdir(parents=True)
            old_audio.write_bytes(b"OLD")
            argv = ["merge_cascaded_shards.py"]
            for lane in inputs:
                argv.extend(["--input-dir", str(lane)])
            error_lane = root / "shard_2"
            error_lane.mkdir(parents=True)
            failed = build_sample(
                row_index=2,
                sample_id="2",
                record={"speech_file": "speech/2.wav", "speech_text": "reference 2"},
                stage={
                    "stage": 1,
                    "task": "text_to_speech",
                    "model": "f5_tts",
                    "input": {"type": "speech_text", "text": "failed"},
                    "prompt": None,
                    "inference_seconds": 0.4,
                    "output": None,
                    "error": {"type": "RuntimeError", "message": "failed"},
                },
                status="error",
                error={"type": "RuntimeError", "message": "failed"},
            )
            (error_lane / "manifest.jsonl").write_text(json.dumps(wrap_sample(metadata, failed)) + "\n", encoding="utf-8")
            error_definition = dict(definition)
            error_definition["stage"] = 1
            write_inspection_manifest(
                error_lane / "manifest.inspect.json",
                metadata=metadata,
                stage_definitions=[error_definition],
                samples=[failed],
                created_at="t",
                started_at="t",
                completed_at="t",
                status="completed_with_errors",
            )
            # Use a matching one-stage definition for every shard in this synthetic error-row test.
            for lane in inputs:
                inspection_path = lane / "manifest.inspect.json"
                inspection = json.loads(inspection_path.read_text())
                inspection["stage_definitions"] = [error_definition]
                inspection_path.write_text(json.dumps(inspection) + "\n")
            argv.extend(["--input-dir", str(error_lane)])
            argv.extend([
                "--output-dir", str(output),
                "--lane", "full_a2s_afn_qwen32_f5_tts",
                "--expected-count", "3",
                "--artifact-subdir", "audio",
                "--backup-label", "test",
            ])
            with patch.object(sys, "argv", argv):
                main()
            rows = [json.loads(line) for line in (output / "manifest.jsonl").read_text().splitlines()]
            inspection = json.loads((output / "manifest.inspect.json").read_text())
            backed_up_audio = (output / "audio.pre_test" / "000000_0.wav").read_bytes()
            promoted_audio = (output / "audio" / "000000_0.wav").read_bytes()

        self.assertEqual(backed_up_audio, b"OLD")
        self.assertEqual(promoted_audio, b"RIFFfake")
        self.assertEqual(rows[0]["lane"], "full_a2s_afn_qwen32_f5_tts")
        expected = str(output / "audio" / "000000_0.wav")
        self.assertEqual(rows[0]["sample"]["stages"][-1]["output"]["file"], expected)
        self.assertEqual(rows[0]["sample"]["final_output"]["file"], expected)
        self.assertEqual(inspection["summary"], {"total": 3, "success": 2, "failed": 1})
        self.assertEqual(rows[2]["sample"]["status"], "error")
        self.assertIsNone(rows[2]["sample"]["final_output"])

    def test_merges_successful_t2t_text_artifact_without_final_output(self) -> None:
        metadata = {
            "lane": "shard_t2t",
            "pipeline": "A2S",
            "dataset": "ttas",
            "split": "val",
            "config_file": "configs/example.json",
        }
        definition = {
            "stage": 2,
            "task": "text_to_text",
            "model": "qwen3_32b",
            "checkpoint": "Qwen3-32B",
            "dtype": "bfloat16",
            "prompt_template": "{input}",
            "generation": {},
            "runtime": {"device": "cuda", "model_load_seconds": 2.0},
        }
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            lane = root / "shard"
            artifact = lane / "text" / "000000_sample.json"
            artifact.parent.mkdir(parents=True)
            artifact.write_text('{"text":"hello"}\n')
            sample = build_sample(
                row_index=0,
                sample_id="sample",
                record={"audio_file": "audio/sample.wav", "audio_cap": "engine"},
                stage={
                    "stage": 1,
                    "task": "text_to_text",
                    "model": "qwen3_32b",
                    "mode": "audio_description_to_speech",
                    "input": {"type": "audio_caption", "text": "engine"},
                    "prompt": None,
                    "inference_seconds": 0.3,
                    "output": {"type": "speech_text", "text": "Start now."},
                    "error": None,
                },
                status="success",
            )
            (lane / "manifest.jsonl").write_text(json.dumps(wrap_sample(metadata, sample)) + "\n")
            write_inspection_manifest(
                lane / "manifest.inspect.json",
                metadata=metadata,
                stage_definitions=[definition],
                samples=[sample],
                created_at="t",
                started_at="t",
                completed_at="t",
                status="completed",
            )
            output = root / "canonical"
            argv = [
                "merge_cascaded_shards.py",
                "--input-dir", str(lane),
                "--output-dir", str(output),
                "--lane", "full_a2s_afn_qwen32_t2t",
                "--expected-count", "1",
                "--artifact-subdir", "text",
                "--backup-label", "test",
            ]
            with patch.object(sys, "argv", argv):
                main()

            merged = json.loads((output / "manifest.jsonl").read_text())
            promoted_exists = (output / "text" / artifact.name).is_file()

        self.assertTrue(promoted_exists)
        self.assertEqual(merged["sample"]["final_output"], None)
        self.assertEqual(merged["sample"]["stages"][-1]["output"]["text"], "Start now.")


if __name__ == "__main__":
    unittest.main()
