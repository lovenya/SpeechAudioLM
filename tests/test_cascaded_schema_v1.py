from __future__ import annotations

import unittest

from baselines.cascaded_schema_v1 import (
    legacy_sample_to_inspect_sample,
    legacy_sample_to_stage_manifest_row,
)


class CascadedSchemaV1Test(unittest.TestCase):
    def test_converts_wrapped_t2t_a2s_sample_to_compact_manifest_row(self) -> None:
        sample = _sample(
            stages=[
                {
                    "stage": 1,
                    "task": "audio_captioning",
                    "model": "audio_flamingo_next",
                    "input": {"type": "audio", "file": "audio/source.wav"},
                    "prompt": {"messages": [{"role": "user", "content": "caption prompt"}]},
                    "inference_seconds": 1.2,
                    "output": {"type": "audio_caption", "text": "A motor starts."},
                    "error": None,
                },
                {
                    "stage": 2,
                    "task": "text_to_text",
                    "model": "qwen3_32b",
                    "mode": "audio_description_to_speech",
                    "input": {"type": "audio_caption", "text": "A motor starts."},
                    "prompt": {"messages": [{"role": "user", "content": "runtime a2s prompt"}]},
                    "inference_seconds": 2.3,
                    "output": {"type": "speech_text", "text": "Try the starter again."},
                    "error": None,
                },
            ]
        )

        row = legacy_sample_to_stage_manifest_row({"sample": sample})

        self.assertEqual(
            list(row),
            ["schema_version", "row_index", "sample_id", "stage", "model", "status", "input", "output", "timing", "error"],
        )
        self.assertEqual(row["stage"], "t2t")
        self.assertEqual(row["input"], {"source_stage_index": 1, "source_sample_id": "sample-a", "source_status": "success"})
        self.assertEqual(row["output"], {"type": "text", "generated_speech_text": "Try the starter again."})
        self.assertEqual(row["timing"], {"inference_seconds": 2.3})
        self.assertIsNone(row["error"])

    def test_maps_stage_one_asr_output_to_speech_transcription(self) -> None:
        sample = _sample(
            stages=[
                {
                    "stage": 1,
                    "task": "asr",
                    "model": "whisper",
                    "input": {"type": "speech_audio", "file": "speech/source.wav"},
                    "prompt": None,
                    "inference_seconds": 0.8,
                    "output": {"type": "speech_text", "text": "Start the engine."},
                    "error": None,
                }
            ]
        )

        row = legacy_sample_to_stage_manifest_row(sample)

        self.assertEqual(row["stage"], "s2t")
        self.assertEqual(row["input"], {"speech_file": "speech/source.wav"})
        self.assertEqual(row["output"], {"type": "text", "speech_transcription": "Start the engine."})

    def test_rewrites_generated_media_paths_under_outputs(self) -> None:
        sample = _sample(
            stages=[
                {
                    "stage": 1,
                    "task": "speech_to_text",
                    "model": "whisper",
                    "input": {"type": "speech_audio", "file": "speech/source.wav"},
                    "prompt": None,
                    "inference_seconds": 0.8,
                    "output": {"type": "speech_text", "text": "Start the engine."},
                    "error": None,
                },
                {
                    "stage": 2,
                    "task": "text_to_text",
                    "model": "qwen3_32b",
                    "mode": "speech_to_audio_description",
                    "input": {"type": "speech_text", "text": "Start the engine."},
                    "prompt": {"messages": [{"role": "user", "content": "runtime s2a prompt"}]},
                    "inference_seconds": 2.0,
                    "output": {"type": "audio_description", "text": "A starter motor turns."},
                    "error": None,
                },
                {
                    "stage": 3,
                    "task": "text_to_audio",
                    "model": "audiogen",
                    "input": {"type": "audio_description", "text": "A starter motor turns."},
                    "prompt": None,
                    "inference_seconds": 4.0,
                    "output": {
                        "type": "environmental_audio",
                        "file": "/tmp/cascade/shard_000/generated/source.wav",
                        "sample_rate": 32000,
                        "duration_seconds": 10.0,
                    },
                    "error": None,
                },
            ],
            final_output={
                "type": "environmental_audio",
                "file": "/tmp/cascade/shard_000/generated/source.wav",
                "sample_rate": 32000,
                "duration_seconds": 10.0,
            },
        )

        row = legacy_sample_to_stage_manifest_row(sample)
        inspect = legacy_sample_to_inspect_sample(sample)

        expected = {
            "type": "audio",
            "audio_file": "outputs/source.wav",
            "sample_rate": 32000,
            "duration_seconds": 10.0,
        }
        self.assertEqual(row["stage"], "t2a")
        self.assertEqual(row["output"], expected)
        self.assertEqual(inspect["final_output"], expected)

    def test_t2s_manifest_and_inspect_keep_conditioning_reference(self) -> None:
        reference = {
            "speech_file": "speech/train-ref.wav",
            "speech_text": "reference words",
            "sample_id": "train-ref",
            "source_split": "train",
            "selection": {"bank_file": "configs/tts_reference_banks/cosyvoice3_train_25.json"},
        }
        sample = _sample(
            stages=[
                {
                    "stage": 1,
                    "task": "audio_captioning",
                    "model": "audio_flamingo_next",
                    "input": {"type": "audio", "file": "audio/source.wav"},
                    "prompt": None,
                    "inference_seconds": 1.2,
                    "output": {"type": "audio_caption", "text": "A motor starts."},
                    "error": None,
                },
                {
                    "stage": 2,
                    "task": "text_to_text",
                    "model": "qwen3_32b",
                    "mode": "audio_description_to_speech",
                    "input": {"type": "audio_caption", "text": "A motor starts."},
                    "prompt": None,
                    "inference_seconds": 2.3,
                    "output": {"type": "speech_text", "text": "Try the starter again."},
                    "error": None,
                },
                {
                    "stage": 3,
                    "task": "text_to_speech",
                    "model": "cosyvoice3",
                    "input": {"type": "speech_text", "text": "Try the starter again.", "reference": reference},
                    "prompt": None,
                    "inference_seconds": 3.4,
                    "output": {
                        "type": "speech_audio",
                        "file": "/tmp/cascade/shard_000/generated/source.wav",
                        "sample_rate": 24000,
                        "duration_seconds": 4.2,
                    },
                    "error": None,
                },
            ],
            final_output={
                "type": "speech_audio",
                "file": "/tmp/cascade/shard_000/generated/source.wav",
                "sample_rate": 24000,
                "duration_seconds": 4.2,
            },
        )

        row = legacy_sample_to_stage_manifest_row(sample)
        inspect = legacy_sample_to_inspect_sample(sample)

        self.assertEqual(row["stage"], "t2s")
        self.assertEqual(row["input"]["reference"], reference)
        self.assertEqual(inspect["stages"][2]["input"], {"generated_speech_text": "Try the starter again.", "reference": reference})

    def test_error_row_preserves_structured_error_and_null_output(self) -> None:
        sample = _sample(
            status="error",
            error={"type": "RuntimeError", "message": "model failed", "retryable": False},
            stages=[
                {
                    "stage": 1,
                    "task": "audio_to_text",
                    "model": "audio_flamingo_next",
                    "input": {"type": "audio", "file": "audio/source.wav"},
                    "prompt": {"messages": [{"role": "user", "content": "caption prompt"}]},
                    "inference_seconds": 1.0,
                    "output": None,
                    "error": {"type": "RuntimeError", "message": "model failed", "retryable": False},
                }
            ],
        )

        row = legacy_sample_to_stage_manifest_row(sample)

        self.assertEqual(row["status"], "error")
        self.assertIsNone(row["output"])
        self.assertEqual(row["error"], {"type": "RuntimeError", "message": "model failed", "retryable": False})

    def test_inspect_sample_keeps_captured_runtime_prompt(self) -> None:
        prompt = {
            "messages": [
                {"role": "system", "content": "Use the captured prompt."},
                {"role": "user", "content": "Audio description: A motor starts."},
            ]
        }
        sample = _sample(
            stages=[
                {
                    "stage": 1,
                    "task": "audio_to_text",
                    "model": "audio_flamingo_next",
                    "input": {"type": "audio", "file": "audio/source.wav"},
                    "prompt": None,
                    "inference_seconds": 1.2,
                    "output": {"type": "audio_caption", "text": "A motor starts."},
                    "error": None,
                },
                {
                    "stage": 2,
                    "task": "text_to_text",
                    "model": "qwen3_32b",
                    "mode": "audio_description_to_speech",
                    "input": {"type": "audio_caption", "text": "A motor starts."},
                    "prompt": prompt,
                    "inference_seconds": 2.3,
                    "output": {"type": "speech_text", "text": "Try the starter again."},
                    "error": None,
                },
            ]
        )

        inspect = legacy_sample_to_inspect_sample(sample)

        self.assertEqual(
            inspect["reference"],
            {
                "audio_file": "audio/source.wav",
                "speech_file": "speech/source.wav",
                "reference_audio_caption": "Reference caption.",
                "reference_speech_text": "Reference speech.",
                "audio_metadata": {"primary_audio_class": "vehicle"},
            },
        )
        self.assertIs(inspect["stages"][1]["runtime_prompt"], prompt)
        self.assertEqual(inspect["stages"][1]["input"], {"audio_caption": "A motor starts."})
        self.assertEqual(inspect["stages"][1]["output"], {"type": "text", "generated_speech_text": "Try the starter again."})


def _sample(
    *,
    stages: list[dict],
    status: str = "success",
    error: dict | None = None,
    final_output: dict | None = None,
) -> dict:
    return {
        "row_index": 7,
        "sample_id": "sample-a",
        "status": status,
        "reference": {
            "audio_file": "audio/source.wav",
            "audio_caption": "Reference caption.",
            "speech_file": "speech/source.wav",
            "speech_text": "Reference speech.",
            "audio_metadata": {"primary_audio_class": "vehicle"},
        },
        "stages": stages,
        "final_output": final_output,
        "error": error,
    }


if __name__ == "__main__":
    unittest.main()
