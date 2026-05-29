from __future__ import annotations

import argparse
import unittest
from pathlib import Path

from baseline_cascaded import _cascade_output_root, _cascade_variant_name, _pipeline_prompt_field, _planned_stages
from baselines.models.text_llm import _strip_thinking_output


class CascadedBaselineTest(unittest.TestCase):
    def test_prompt_fields_match_pipeline_inputs(self) -> None:
        self.assertEqual(_pipeline_prompt_field("T2A"), "audio_cap")
        self.assertEqual(_pipeline_prompt_field("A2T"), "audio_cap")
        self.assertEqual(_pipeline_prompt_field("A2S"), "audio_cap")
        self.assertEqual(_pipeline_prompt_field("T2S"), "speech_text")
        self.assertEqual(_pipeline_prompt_field("S2T"), "speech_text")
        self.assertEqual(_pipeline_prompt_field("S2A"), "speech_text")

    def test_cascade_output_root_includes_backend_variant(self) -> None:
        args = argparse.Namespace(
            output_dir="results/baselines",
            pipeline="S2A",
            dataset="ttas",
            split="test",
            run_name=None,
            asr="reference",
            text_model="template",
            t2a="audiogen",
        )
        self.assertEqual(
            _cascade_output_root(args),
            Path("results/baselines/cascaded/S2A/asr-reference__text_to_text-template__text_to_audio-audiogen/ttas/test"),
        )

    def test_cascade_variant_distinguishes_backend_choices(self) -> None:
        audiogen = argparse.Namespace(pipeline="S2A", run_name=None, asr="reference", text_model="template", t2a="audiogen")
        audio_omni = argparse.Namespace(pipeline="S2A", run_name=None, asr="reference", text_model="template", t2a="audio_omni")
        self.assertNotEqual(_cascade_variant_name(audiogen), _cascade_variant_name(audio_omni))

    def test_cascade_variant_can_be_overridden(self) -> None:
        args = argparse.Namespace(pipeline="S2A", run_name="smoke lane/0", asr="reference", text_model="template", t2a="audiogen")
        self.assertEqual(_cascade_variant_name(args), "smoke_lane_0")

    def test_planned_stages_for_s2a(self) -> None:
        args = argparse.Namespace(pipeline="S2A", asr="whisper", text_model="qwen3_32b", t2a="tangoflux")
        self.assertEqual(
            _planned_stages(args),
            [
                {"stage": "asr", "backend": "whisper"},
                {"stage": "text_to_text", "backend": "qwen3_32b"},
                {"stage": "text_to_audio", "backend": "tangoflux"},
            ],
        )

    def test_planned_stages_for_a2s(self) -> None:
        args = argparse.Namespace(
            pipeline="A2S",
            captioner="audio_flamingo_next",
            text_model="llama3_3_70b_instruct",
            tts="f5_tts",
        )
        self.assertEqual(
            _planned_stages(args),
            [
                {"stage": "audio_captioning", "backend": "audio_flamingo_next"},
                {"stage": "text_to_text", "backend": "llama3_3_70b_instruct"},
                {"stage": "text_to_speech", "backend": "f5_tts"},
            ],
        )

    def test_planned_stages_for_t2t(self) -> None:
        args = argparse.Namespace(pipeline="T2T", text_model="qwen3_32b")
        self.assertEqual(_planned_stages(args), [{"stage": "text_to_text", "backend": "qwen3_32b"}])

    def test_strip_thinking_output_prefers_final_answer(self) -> None:
        self.assertEqual(_strip_thinking_output("<think>reasoning</think>rain on pavement"), "rain on pavement")

    def test_strip_thinking_output_handles_orphan_think_prefix(self) -> None:
        self.assertEqual(
            _strip_thinking_output("<think>\nreasoning line\ncity traffic with idling motorcycle"),
            "city traffic with idling motorcycle",
        )

if __name__ == "__main__":
    unittest.main()
