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
        sample = row["_cascaded_sample"]
        self.assertEqual(sample["reference"]["speech_text"], "ground truth speech")
        self.assertEqual(sample["stages"][0]["model"], "audio_flamingo_next")
        self.assertEqual(sample["stages"][0]["output"]["text"], "generated caption")
        self.assertFalse(any(key.startswith("_de" + "mo") for key in row))

    def test_legacy_stage_prompts_do_not_use_reference_text(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            cases = [
                ("audio_flamingo_next", "A2S", "Generate a caption for the input audio."),
                ("qwen3_omni_captioner", "A2S", None),
                ("whisper", "S2A", None),
            ]
            for model, pipeline, expected_prompt in cases:
                manifest = root / f"{model}.jsonl"
                manifest.write_text(
                    json.dumps(
                        {
                            "status": "success",
                            "row_index": 0,
                            "sample_id": model,
                            "prompt": "REFERENCE TEXT MUST NOT BECOME A MODEL PROMPT",
                            "candidate_text": "generated text",
                            "model_extra": {"stages": []},
                            "source": {
                                "audio_file": "audio/a.wav",
                                "speech_file": "speech/a.wav",
                                "speech_text": "reference speech",
                                "audio_cap": "reference caption",
                            },
                        }
                    )
                    + "\n",
                    encoding="utf-8",
                )
                row = _stage1_to_t2t(
                    argparse.Namespace(
                        manifest=str(manifest),
                        pipeline=pipeline,
                        upstream_backend=model,
                        text_model="qwen3_32b",
                        allow_partial=False,
                    )
                )[0]
                prompt = row["_cascaded_sample"]["stages"][0]["prompt"]
                messages = prompt["messages"] if prompt else []
                text_parts = [
                    part.get("text")
                    for message in messages
                    for part in (message.get("content") if isinstance(message.get("content"), list) else [])
                    if isinstance(part, dict)
                ]
                rendered = text_parts[0] if text_parts else None
                self.assertEqual(rendered, expected_prompt)
                self.assertNotEqual(rendered, "REFERENCE TEXT MUST NOT BECOME A MODEL PROMPT")

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
        sample = row["_cascaded_sample"]
        self.assertEqual(sample["reference"]["audio_caption"], "ground truth caption")
        self.assertEqual(len(sample["stages"]), 2)
        self.assertEqual(sample["stages"][0]["output"]["text"], "caption input")
        self.assertEqual(sample["stages"][1]["output"]["text"], "generated speech text")
        self.assertFalse(any(key.startswith("_de" + "mo") for key in row))

    def test_v1_stage1_to_t2t_uses_stage_output_and_inspect_reference(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            manifest = root / "manifest.jsonl"
            manifest.write_text(
                json.dumps(
                    {
                        "schema_version": 1,
                        "row_index": 4,
                        "sample_id": "sample_a",
                        "stage": "a2t",
                        "model": "captioner",
                        "status": "success",
                        "input": {"audio_file": "audio/a.wav"},
                        "output": {"type": "text", "audio_caption": "generated caption"},
                        "timing": {"inference_seconds": 1.5},
                        "error": None,
                    }
                )
                + "\n",
                encoding="utf-8",
            )
            (root / "inspect.json").write_text(
                json.dumps(
                    {
                        "schema_version": 1,
                        "stage_configs": [
                            {
                                "stage_index": 1,
                                "stage": "a2t",
                                "model": {
                                    "name": "captioner",
                                    "path": "{model_root}/captioner",
                                    "device": "cuda",
                                    "dtype": "bf16",
                                },
                                "prompt": {
                                    "mode": "audio_description_to_speech",
                                    "template": "caption template",
                                    "system_prompt": "caption system",
                                },
                                "generation": {"temperature": 0.2},
                            }
                        ],
                        "samples": [
                            {
                                "sample_id": "sample_a",
                                "reference": {
                                    "audio_file": "audio/a.wav",
                                    "speech_file": "speech/a.wav",
                                    "reference_audio_caption": "reference caption",
                                    "reference_speech_text": "reference speech",
                                },
                            }
                        ],
                    }
                )
                + "\n",
                encoding="utf-8",
            )
            (root / "run.json").write_text(
                json.dumps(
                    {
                        "schema_version": 1,
                        "run_id": "20260616_1000",
                        "lineage": {
                            "direction": "a2s",
                            "current_stage_index": 1,
                            "stages": [
                                {
                                    "stage_index": 1,
                                    "stage": "a2t",
                                    "model": "captioner",
                                    "run_id": "20260616_1000",
                                    "run_path": "results/cascaded/stages/a2t/captioner/ttas/val/20260616_1000",
                                    "manifest": "results/cascaded/stages/a2t/captioner/ttas/val/20260616_1000/manifest.jsonl",
                                    "config": "results/cascaded/stages/a2t/captioner/ttas/val/20260616_1000/config.json",
                                }
                            ],
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
                    upstream_backend="captioner",
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
        sample = row["_cascaded_sample"]
        self.assertEqual(sample["row_index"], 4)
        self.assertEqual(sample["reference"]["speech_text"], "reference speech")
        self.assertEqual(sample["reference"]["audio_caption"], "reference caption")
        self.assertEqual(sample["stages"][0]["task"], "audio_to_text")
        self.assertEqual(sample["stages"][0]["output"], {"type": "audio_caption", "text": "generated caption"})
        self.assertEqual(row["_cascaded_stage_lineage"][0]["run_id"], "20260616_1000")

    def test_v1_stage1_to_t2t_supports_s2a_s2t_manifest(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            manifest = Path(tmpdir) / "manifest.jsonl"
            manifest.write_text(
                json.dumps(
                    {
                        "schema_version": 1,
                        "row_index": 3,
                        "sample_id": "sample_s",
                        "stage": "s2t",
                        "model": "asr",
                        "status": "success",
                        "input": {"speech_file": "speech/s.wav"},
                        "output": {"type": "text", "speech_transcription": "spoken words"},
                        "timing": {"inference_seconds": 0.5},
                        "error": None,
                    }
                )
                + "\n",
                encoding="utf-8",
            )

            rows = _stage1_to_t2t(
                argparse.Namespace(
                    manifest=str(manifest),
                    pipeline="S2A",
                    upstream_backend="asr",
                    text_model="qwen3_32b",
                    allow_partial=False,
                )
            )

        self.assertEqual(rows[0]["speech_text"], "spoken words")
        self.assertEqual(rows[0]["audio_cap"], "spoken words")
        sample = rows[0]["_cascaded_sample"]
        self.assertEqual(sample["stages"][0]["task"], "speech_to_text")
        self.assertEqual(sample["stages"][0]["input"], {"type": "speech_audio", "file": "speech/s.wav"})
        self.assertEqual(sample["stages"][0]["output"], {"type": "speech_text", "text": "spoken words"})

    def test_v1_stage1_to_t2t_prefers_manifest_output_over_stale_inspect(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            manifest = root / "manifest.jsonl"
            manifest.write_text(
                json.dumps(
                    {
                        "schema_version": 1,
                        "row_index": 4,
                        "sample_id": "sample_a",
                        "stage": "a2t",
                        "model": "captioner",
                        "status": "success",
                        "input": {"audio_file": "audio/a.wav"},
                        "output": {"type": "text", "audio_caption": "manifest caption"},
                        "timing": {"inference_seconds": 1.5},
                        "error": None,
                    }
                )
                + "\n",
                encoding="utf-8",
            )
            (root / "inspect.json").write_text(
                json.dumps(
                    {
                        "schema_version": 1,
                        "stage_configs": [
                            {
                                "stage_index": 1,
                                "stage": "a2t",
                                "model": {
                                    "name": "captioner",
                                    "path": "{model_root}/captioner",
                                    "device": "cuda",
                                    "dtype": "bf16",
                                },
                                "prompt": {
                                    "mode": "audio_description_to_speech",
                                    "template": "caption template",
                                    "system_prompt": "caption system",
                                },
                                "generation": {"temperature": 0.2},
                            }
                        ],
                        "samples": [
                            {
                                "sample_id": "sample_a",
                                "reference": {
                                    "audio_file": "audio/a.wav",
                                    "speech_file": "speech/a.wav",
                                    "reference_audio_caption": "reference caption",
                                    "reference_speech_text": "reference speech",
                                },
                                "stages": [
                                    {
                                        "stage_index": 1,
                                        "stage": "a2t",
                                        "model": "old-captioner",
                                        "status": "success",
                                        "input": {"audio_file": "audio/a.wav"},
                                        "runtime_prompt": None,
                                        "output": {"type": "text", "audio_caption": "stale inspect caption"},
                                        "timing": {"inference_seconds": 999.0},
                                        "error": None,
                                    }
                                ],
                            }
                        ],
                    }
                )
                + "\n",
                encoding="utf-8",
            )

            rows = _stage1_to_t2t(
                argparse.Namespace(
                    manifest=str(manifest),
                    pipeline="A2S",
                    upstream_backend="captioner",
                    text_model="qwen3_32b",
                    allow_partial=False,
                )
            )

        sample = rows[0]["_cascaded_sample"]
        self.assertEqual(rows[0]["speech_text"], "manifest caption")
        self.assertEqual(sample["stages"][0]["model"], "captioner")
        self.assertEqual(sample["stages"][0]["output"], {"type": "audio_caption", "text": "manifest caption"})
        self.assertEqual(sample["stages"][0]["inference_seconds"], 1.5)
        definitions = rows[0]["_cascaded_stage_definitions"]
        self.assertEqual(definitions[0]["task"], "audio_to_text")
        self.assertEqual(definitions[0]["model"], "captioner")
        self.assertEqual(definitions[0]["checkpoint"], "{model_root}/captioner")
        self.assertEqual(definitions[0]["dtype"], "bf16")
        self.assertEqual(definitions[0]["prompt_template"], "caption template")
        self.assertEqual(definitions[0]["system_prompt"], "caption system")
        self.assertEqual(definitions[0]["generation"], {"temperature": 0.2})
        self.assertEqual(definitions[0]["runtime"]["device"], "cuda")

    def test_v1_stage1_to_t2t_rejects_stage_direction_mismatch(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            manifest = Path(tmpdir) / "manifest.jsonl"
            manifest.write_text(
                json.dumps(
                    {
                        "schema_version": 1,
                        "row_index": 3,
                        "sample_id": "sample_s",
                        "stage": "s2t",
                        "model": "asr",
                        "status": "success",
                        "input": {"speech_file": "speech/s.wav"},
                        "output": {"type": "text", "speech_transcription": "spoken words"},
                        "timing": {"inference_seconds": 0.5},
                        "error": None,
                    }
                )
                + "\n",
                encoding="utf-8",
            )

            with self.assertRaisesRegex(ValueError, "Expected v1 stage 'a2t'"):
                _stage1_to_t2t(
                    argparse.Namespace(
                        manifest=str(manifest),
                        pipeline="A2S",
                        upstream_backend="asr",
                        text_model="qwen3_32b",
                        allow_partial=False,
                    )
                )

    def test_v1_t2t_to_final_uses_inspect_cumulative_stages(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            manifest = root / "manifest.jsonl"
            manifest.write_text(
                json.dumps(
                    {
                        "schema_version": 1,
                        "row_index": 4,
                        "sample_id": "sample_a",
                        "stage": "t2t",
                        "model": "qwen3_32b",
                        "status": "success",
                        "input": {"source_stage_index": 1, "source_sample_id": "sample_a", "source_status": "success"},
                        "output": {"type": "text", "generated_speech_text": "generated speech text"},
                        "timing": {"inference_seconds": 2.5},
                        "error": None,
                    }
                )
                + "\n",
                encoding="utf-8",
            )
            (root / "inspect.json").write_text(
                json.dumps(
                    {
                        "schema_version": 1,
                        "samples": [
                            {
                                "sample_id": "sample_a",
                                "reference": {
                                    "audio_file": "audio/a.wav",
                                    "speech_file": "speech/a.wav",
                                    "reference_audio_caption": "reference caption",
                                    "reference_speech_text": "reference speech",
                                },
                                "stages": [
                                    {
                                        "stage_index": 1,
                                        "stage": "a2t",
                                        "model": "captioner",
                                        "status": "success",
                                        "input": {"audio_file": "audio/a.wav"},
                                        "runtime_prompt": None,
                                        "output": {"type": "text", "audio_caption": "generated caption"},
                                        "timing": {"inference_seconds": 1.5},
                                        "error": None,
                                    },
                                    {
                                        "stage_index": 2,
                                        "stage": "t2t",
                                        "model": "qwen3_32b",
                                        "status": "success",
                                        "input": {"audio_caption": "generated caption"},
                                        "runtime_prompt": {"messages": [{"role": "user", "content": "convert"}]},
                                        "output": {"type": "text", "generated_speech_text": "stale inspect speech"},
                                        "timing": {"inference_seconds": 999.0},
                                        "error": None,
                                    },
                                ],
                            }
                        ],
                    }
                )
                + "\n",
                encoding="utf-8",
            )
            (root / "run.json").write_text(
                json.dumps(
                    {
                        "schema_version": 1,
                        "run_id": "20260616_1100",
                        "lineage": {
                            "direction": "a2s",
                            "current_stage_index": 2,
                            "stages": [
                                {
                                    "stage_index": 1,
                                    "stage": "a2t",
                                    "model": "captioner",
                                    "run_id": "20260616_1000",
                                    "run_path": "results/cascaded/stages/a2t/captioner/ttas/val/20260616_1000",
                                    "manifest": "results/cascaded/stages/a2t/captioner/ttas/val/20260616_1000/manifest.jsonl",
                                    "config": "results/cascaded/stages/a2t/captioner/ttas/val/20260616_1000/config.json",
                                },
                                {
                                    "stage_index": 2,
                                    "stage": "t2t",
                                    "model": "qwen3_32b",
                                    "run_id": "20260616_1100",
                                    "run_path": "results/cascaded/stages/t2t/a2s/qwen3_32b/from-a2t/captioner/ttas/val/20260616_1100",
                                    "manifest": "results/cascaded/stages/t2t/a2s/qwen3_32b/from-a2t/captioner/ttas/val/20260616_1100/manifest.jsonl",
                                    "config": "results/cascaded/stages/t2t/a2s/qwen3_32b/from-a2t/captioner/ttas/val/20260616_1100/config.json",
                                },
                            ],
                        },
                    }
                )
                + "\n",
                encoding="utf-8",
            )

            rows = _t2t_to_final(
                argparse.Namespace(
                    manifest=str(manifest),
                    pipeline="A2S",
                    upstream_backend="captioner",
                    text_model="qwen3_32b",
                    final_backend="f5_tts",
                    allow_partial=False,
                )
            )

        self.assertEqual(len(rows), 1)
        row = rows[0]
        self.assertEqual(row["speech_text"], "generated speech text")
        self.assertEqual(row["reference_speech_file"], "speech/a.wav")
        self.assertEqual(row["reference_speech_text"], "reference speech")
        sample = row["_cascaded_sample"]
        self.assertEqual(len(sample["stages"]), 2)
        self.assertEqual(sample["stages"][0]["output"], {"type": "audio_caption", "text": "generated caption"})
        self.assertEqual(sample["stages"][1]["mode"], "audio_description_to_speech")
        self.assertEqual(sample["stages"][1]["output"], {"type": "speech_text", "text": "generated speech text"})
        self.assertEqual(sample["stages"][1]["inference_seconds"], 2.5)
        self.assertEqual([stage["run_id"] for stage in row["_cascaded_stage_lineage"]], ["20260616_1000", "20260616_1100"])

    def test_v1_t2t_to_final_requires_inspect_for_cumulative_context(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            manifest = Path(tmpdir) / "manifest.jsonl"
            manifest.write_text(
                json.dumps(
                    {
                        "schema_version": 1,
                        "row_index": 4,
                        "sample_id": "sample_a",
                        "stage": "t2t",
                        "model": "qwen3_32b",
                        "status": "success",
                        "input": {"source_stage_index": 1, "source_sample_id": "sample_a", "source_status": "success"},
                        "output": {"type": "text", "generated_speech_text": "generated speech text"},
                        "timing": {"inference_seconds": 2.5},
                        "error": None,
                    }
                )
                + "\n",
                encoding="utf-8",
            )

            with self.assertRaisesRegex(ValueError, "requires sibling inspect.json"):
                _t2t_to_final(
                    argparse.Namespace(
                        manifest=str(manifest),
                        pipeline="A2S",
                        upstream_backend="captioner",
                        text_model="qwen3_32b",
                        final_backend="f5_tts",
                        allow_partial=False,
                    )
                )

    def test_v1_t2t_to_final_rejects_non_t2t_manifest(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            manifest = Path(tmpdir) / "manifest.jsonl"
            manifest.write_text(
                json.dumps(
                    {
                        "schema_version": 1,
                        "row_index": 4,
                        "sample_id": "sample_a",
                        "stage": "a2t",
                        "model": "captioner",
                        "status": "success",
                        "input": {"audio_file": "audio/a.wav"},
                        "output": {"type": "text", "audio_caption": "generated caption"},
                        "timing": {"inference_seconds": 1.5},
                        "error": None,
                    }
                )
                + "\n",
                encoding="utf-8",
            )

            with self.assertRaisesRegex(ValueError, "Expected v1 stage 't2t'"):
                _t2t_to_final(
                    argparse.Namespace(
                        manifest=str(manifest),
                        pipeline="A2S",
                        upstream_backend="captioner",
                        text_model="qwen3_32b",
                        final_backend="f5_tts",
                        allow_partial=False,
                    )
                )


if __name__ == "__main__":
    unittest.main()
