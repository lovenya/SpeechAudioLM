from __future__ import annotations

import json
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from baselines.cascaded_manifest import build_sample, wrap_sample, write_inspection_manifest
from scripts.merge_baseline_shards import main


class MergeBaselineShardsTest(unittest.TestCase):
    def test_merges_wrapped_rows_and_inspection_manifest(self) -> None:
        metadata = {
            "lane": "full_s2a_whisper_qwen32_audiogen",
            "pipeline": "S2A",
            "dataset": "ttas",
            "split": "val",
            "config_file": "configs/example.json",
        }
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            shard_root = root / "shards"
            output_root = root / "merged"
            for index in (0, 1):
                lane = shard_root / f"s{index}" / "whisper" / "S2T" / "ttas" / "val"
                lane.mkdir(parents=True)
                stage = {
                    "stage": 1,
                    "task": "speech_to_text",
                    "model": "whisper",
                    "input": {"type": "speech_audio", "file": f"speech/{index}.wav"},
                    "prompt": None,
                    "inference_seconds": 0.5,
                    "output": {"type": "speech_text", "text": f"text {index}"},
                    "error": None,
                }
                sample = build_sample(
                    row_index=index,
                    sample_id=str(index),
                    record={"speech_file": f"speech/{index}.wav", "speech_text": f"reference {index}"},
                    stage=stage,
                    status="success",
                )
                definition = {
                    "stage": 1,
                    "task": "speech_to_text",
                    "model": "whisper",
                    "checkpoint": "whisper-large-v3",
                    "dtype": "float16",
                    "prompt_template": None,
                    "generation": {},
                    "runtime": {"device": "cuda", "model_load_seconds": float(index + 1)},
                }
                (lane / "manifest.jsonl").write_text(json.dumps(wrap_sample(metadata, sample)) + "\n", encoding="utf-8")
                write_inspection_manifest(
                    lane / "manifest.inspect.json",
                    metadata=metadata,
                    stage_definitions=[definition],
                    samples=[sample],
                    created_at="2026-06-07T10:00:00-04:00",
                    started_at="2026-06-07T10:00:01-04:00",
                    completed_at="2026-06-07T10:00:02-04:00",
                    status="completed",
                )
            argv = [
                "merge_baseline_shards.py",
                "--model", "whisper",
                "--task", "S2T",
                "--dataset", "ttas",
                "--split", "val",
                "--shard-root", str(shard_root),
                "--output-root", str(output_root),
            ]
            with patch.object(sys, "argv", argv):
                main()
            merged_root = output_root / "whisper" / "S2T" / "ttas" / "val"
            rows = [json.loads(line) for line in (merged_root / "manifest.jsonl").read_text().splitlines()]
            inspection = json.loads((merged_root / "manifest.inspect.json").read_text())

        self.assertEqual([row["sample"]["row_index"] for row in rows], [0, 1])
        self.assertEqual(inspection["summary"], {"total": 2, "success": 2, "failed": 0})
        self.assertEqual(inspection["stage_definitions"][0]["runtime"]["model_load_seconds"], 2.0)
        static = dict(inspection["stage_definitions"][0])
        static["runtime"] = dict(static["runtime"])
        static["runtime"].pop("model_load_seconds")
        expected_static = {
            "stage": 1,
            "task": "speech_to_text",
            "model": "whisper",
            "checkpoint": "whisper-large-v3",
            "dtype": "float16",
            "prompt_template": None,
            "generation": {},
            "runtime": {"device": "cuda", "model_load_seconds": 1.0},
        }
        expected_static["runtime"] = dict(expected_static["runtime"])
        expected_static["runtime"].pop("model_load_seconds")
        self.assertEqual(static, expected_static)

    def test_rejects_conflicting_wrapper_metadata(self) -> None:
        metadata = {
            "lane": "lane_a",
            "pipeline": "S2A",
            "dataset": "ttas",
            "split": "val",
            "config_file": "configs/example.json",
        }
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            lane = root / "shards" / "s0" / "whisper" / "S2T" / "ttas" / "val"
            lane.mkdir(parents=True)
            sample = build_sample(
                row_index=0,
                sample_id="0",
                record={"speech_file": "speech/0.wav", "speech_text": "reference"},
                stage={
                    "stage": 1,
                    "task": "speech_to_text",
                    "model": "whisper",
                    "input": {"type": "speech_audio", "file": "speech/0.wav"},
                    "prompt": None,
                    "inference_seconds": 0.5,
                    "output": {"type": "speech_text", "text": "text"},
                    "error": None,
                },
                status="success",
            )
            row = wrap_sample({**metadata, "lane": "wrong_lane"}, sample)
            (lane / "manifest.jsonl").write_text(json.dumps(row) + "\n", encoding="utf-8")
            write_inspection_manifest(
                lane / "manifest.inspect.json",
                metadata=metadata,
                stage_definitions=[],
                samples=[sample],
                created_at="t",
                started_at="t",
                completed_at="t",
                status="completed",
            )
            argv = [
                "merge_baseline_shards.py",
                "--model", "whisper",
                "--task", "S2T",
                "--dataset", "ttas",
                "--split", "val",
                "--shard-root", str(root / "shards"),
                "--output-root", str(root / "merged"),
            ]
            with patch.object(sys, "argv", argv), self.assertRaises(SystemExit):
                main()


if __name__ == "__main__":
    unittest.main()
