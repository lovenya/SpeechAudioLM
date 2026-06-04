from __future__ import annotations

import argparse
import json
import tempfile
import unittest
from pathlib import Path

from scripts.build_cascaded_stage_inputs import _stage1_to_t2t, _t2t_to_final


class CascadedStageInputBuilderTest(unittest.TestCase):
    def test_stage1_to_t2t_uses_generated_text_and_preserves_original_row(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            manifest = Path(tmpdir) / "a2t_manifest.jsonl"
            manifest.write_text(
                json.dumps(
                    {
                        "status": "success",
                        "sample_id": "sample_a",
                        "candidate_text": "generated caption",
                        "output_file": "caption.json",
                        "source": {
                            "audio_file": "audio/a.wav",
                            "speech_file": "speech/a.wav",
                            "speech_text": "ground truth speech",
                            "audio_cap": "ground truth caption",
                        },
                    }
                )
                + "\n",
                encoding="utf-8",
            )
            rows = _stage1_to_t2t(
                argparse.Namespace(
                    manifest=str(manifest),
                    pipeline="A2S",
                    upstream_backend="audio_flamingo_next",
                    text_model="qwen3_32b",
                    allow_partial=False,
                )
            )

        self.assertEqual(len(rows), 1)
        row = rows[0]
        self.assertEqual(row["audio_file"], "audio/a.wav")
        self.assertEqual(row["speech_file"], "speech/a.wav")
        self.assertEqual(row["speech_text"], "generated caption")
        self.assertEqual(row["audio_cap"], "generated caption")
        self.assertEqual(row["original_row"]["speech_text"], "ground truth speech")
        self.assertEqual(row["stage_provenance"]["upstream_backend"], "audio_flamingo_next")
        self.assertFalse(any(key.startswith("_de" + "mo") for key in row))

    def test_t2t_to_final_preserves_reference_speech_for_a2s(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            manifest = Path(tmpdir) / "t2t_manifest.jsonl"
            stage_input = {
                "audio_file": "audio/a.wav",
                "speech_file": "speech/a.wav",
                "speech_text": "caption input",
                "audio_cap": "caption input",
                "stage_provenance": {"stage": "t2t_input"},
                "original_row": {
                    "audio_file": "audio/a.wav",
                    "speech_file": "speech/a.wav",
                    "speech_text": "ground truth speech",
                    "audio_cap": "ground truth caption",
                },
            }
            manifest.write_text(
                json.dumps(
                    {
                        "status": "success",
                        "sample_id": "sample_a",
                        "candidate_text": "generated speech text",
                        "output_file": "t2t.json",
                        "source": stage_input,
                    }
                )
                + "\n",
                encoding="utf-8",
            )
            rows = _t2t_to_final(
                argparse.Namespace(
                    manifest=str(manifest),
                    pipeline="A2S",
                    upstream_backend="audio_flamingo_next",
                    text_model="qwen3_32b",
                    final_backend="f5_tts",
                    allow_partial=False,
                )
            )

        self.assertEqual(len(rows), 1)
        row = rows[0]
        self.assertEqual(row["speech_text"], "generated speech text")
        self.assertEqual(row["reference_speech_file"], "speech/a.wav")
        self.assertEqual(row["reference_speech_text"], "ground truth speech")
        self.assertEqual(row["original_row"]["audio_cap"], "ground truth caption")
        self.assertEqual(row["stage_provenance"]["upstream_stage_provenance"], {"stage": "t2t_input"})
        self.assertFalse(any(key.startswith("_de" + "mo") for key in row))


if __name__ == "__main__":
    unittest.main()
