from __future__ import annotations

import argparse
import sys
import types
import unittest
from pathlib import Path

from baseline_cascaded import _cascade_output_root, _cascade_variant_name, _pipeline_prompt_field, _planned_stages
from baselines.models.audio_captioning import _patch_audio_flamingo_next_transformers_compat
from baselines.models.speech_asr import _patch_qwen_asr_transformers_compat
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

    def test_planned_stages_for_cosyvoice3_t2s(self) -> None:
        args = argparse.Namespace(pipeline="T2S", asr="reference", captioner="reference", text_model="template", tts="cosyvoice3", t2a="audiogen")
        self.assertEqual(_planned_stages(args), [{"stage": "text_to_speech", "backend": "cosyvoice3"}])

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

    def test_audio_flamingo_next_compat_patch_registers_musicflamingo_alias(self) -> None:
        transformers = types.ModuleType("transformers")

        class AudioFlamingo3Config:
            pass

        class AudioFlamingo3ForConditionalGeneration:
            pass

        class AudioFlamingo3Processor:
            pass

        class AutoRegistry:
            registered = []

            @classmethod
            def register(cls, *args, **kwargs):
                cls.registered.append((args, kwargs))

        class AutoConfig(AutoRegistry):
            registered = []

        class AutoProcessor(AutoRegistry):
            registered = []

        transformers.AudioFlamingo3Config = AudioFlamingo3Config
        transformers.AudioFlamingo3ForConditionalGeneration = AudioFlamingo3ForConditionalGeneration
        transformers.AudioFlamingo3Processor = AudioFlamingo3Processor
        transformers.AutoConfig = AutoConfig
        transformers.AutoProcessor = AutoProcessor
        old_transformers = sys.modules.get("transformers")
        sys.modules["transformers"] = transformers
        try:
            _patch_audio_flamingo_next_transformers_compat()
            self.assertTrue(issubclass(transformers.MusicFlamingoConfig, AudioFlamingo3Config))
            self.assertEqual(transformers.MusicFlamingoConfig.model_type, "musicflamingo")
            self.assertIs(transformers.MusicFlamingoForConditionalGeneration, AudioFlamingo3ForConditionalGeneration)
            self.assertIs(transformers.MusicFlamingoProcessor, AudioFlamingo3Processor)
            self.assertEqual(AutoConfig.registered[0][0], ("musicflamingo", transformers.MusicFlamingoConfig))
            self.assertEqual(AutoProcessor.registered[0][0], (transformers.MusicFlamingoConfig, AudioFlamingo3Processor))
        finally:
            if old_transformers is None:
                sys.modules.pop("transformers", None)
            else:
                sys.modules["transformers"] = old_transformers

    def test_qwen_asr_compat_patch_adds_transformers_helpers(self) -> None:
        generic = types.ModuleType("transformers.utils.generic")
        rope_utils = types.ModuleType("transformers.modeling_rope_utils")
        rope_utils.ROPE_INIT_FUNCTIONS = {}
        utils = types.ModuleType("transformers.utils")
        transformers = types.ModuleType("transformers")
        qwen_asr = types.ModuleType("qwen_asr")
        qwen_core = types.ModuleType("qwen_asr.core")
        qwen_backend = types.ModuleType("qwen_asr.core.transformers_backend")
        qwen_config = types.ModuleType("qwen_asr.core.transformers_backend.configuration_qwen3_asr")

        class Qwen3ASRThinkerConfig:
            def __init__(self) -> None:
                self.audio_token_id = 151676

        qwen_config.Qwen3ASRThinkerConfig = Qwen3ASRThinkerConfig
        utils.generic = generic
        transformers.utils = utils
        transformers.modeling_rope_utils = rope_utils
        old_transformers = sys.modules.get("transformers")
        old_utils = sys.modules.get("transformers.utils")
        old_generic = sys.modules.get("transformers.utils.generic")
        old_rope_utils = sys.modules.get("transformers.modeling_rope_utils")
        old_qwen_asr = sys.modules.get("qwen_asr")
        old_qwen_core = sys.modules.get("qwen_asr.core")
        old_qwen_backend = sys.modules.get("qwen_asr.core.transformers_backend")
        old_qwen_config = sys.modules.get("qwen_asr.core.transformers_backend.configuration_qwen3_asr")
        sys.modules["transformers"] = transformers
        sys.modules["transformers.utils"] = utils
        sys.modules["transformers.utils.generic"] = generic
        sys.modules["transformers.modeling_rope_utils"] = rope_utils
        sys.modules["qwen_asr"] = qwen_asr
        sys.modules["qwen_asr.core"] = qwen_core
        sys.modules["qwen_asr.core.transformers_backend"] = qwen_backend
        sys.modules["qwen_asr.core.transformers_backend.configuration_qwen3_asr"] = qwen_config
        try:
            _patch_qwen_asr_transformers_compat()
            def direct_fn():
                return None

            def decorated_fn():
                return None

            self.assertIs(generic.check_model_inputs(direct_fn), direct_fn)
            self.assertIs(generic.check_model_inputs()(decorated_fn), decorated_fn)
            self.assertIn("default", rope_utils.ROPE_INIT_FUNCTIONS)
            self.assertNotIn("nagisa", sys.modules)
            thinker_config = Qwen3ASRThinkerConfig()
            self.assertIsNone(thinker_config.pad_token_id)
        finally:
            if old_transformers is None:
                sys.modules.pop("transformers", None)
            else:
                sys.modules["transformers"] = old_transformers
            if old_utils is None:
                sys.modules.pop("transformers.utils", None)
            else:
                sys.modules["transformers.utils"] = old_utils
            if old_generic is None:
                sys.modules.pop("transformers.utils.generic", None)
            else:
                sys.modules["transformers.utils.generic"] = old_generic
            if old_rope_utils is None:
                sys.modules.pop("transformers.modeling_rope_utils", None)
            else:
                sys.modules["transformers.modeling_rope_utils"] = old_rope_utils
            for name, old_module in (
                ("qwen_asr", old_qwen_asr),
                ("qwen_asr.core", old_qwen_core),
                ("qwen_asr.core.transformers_backend", old_qwen_backend),
                ("qwen_asr.core.transformers_backend.configuration_qwen3_asr", old_qwen_config),
            ):
                if old_module is None:
                    sys.modules.pop(name, None)
                else:
                    sys.modules[name] = old_module

if __name__ == "__main__":
    unittest.main()
