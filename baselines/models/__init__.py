from __future__ import annotations

from baselines.models.base import BaselineRunner
from project_paths import model_path


def get_runner(name: str, **kwargs) -> BaselineRunner:
    if name == "audiogen":
        from baselines.models.audiogen import AudioGenRunner

        return AudioGenRunner(**kwargs)
    if name == "tangoflux":
        from baselines.models.tangoflux import TangoFluxRunner

        return TangoFluxRunner(**kwargs)
    if name == "qwen3_omni":
        from baselines.models.qwen3_omni import Qwen3OmniRunner

        return Qwen3OmniRunner(**kwargs)
    if name == "audio_omni":
        from baselines.models.audio_omni import AudioOmniRunner

        return AudioOmniRunner(**kwargs)
    if name == "whisper":
        from baselines.models.speech_asr import WhisperASRRunner

        return WhisperASRRunner(**kwargs)
    if name == "qwen3_asr":
        from baselines.models.speech_asr import Qwen3ASRRunner

        return Qwen3ASRRunner(**kwargs)
    if name == "qwen3_32b":
        from baselines.models.text_llm import TextLLMRunner

        if not kwargs.get("model_path"):
            kwargs["model_path"] = str(model_path("qwen3_32b"))
        return TextLLMRunner(**kwargs)
    if name == "llama3_3_70b_instruct":
        from baselines.models.text_llm import TextLLMRunner

        if not kwargs.get("model_path"):
            kwargs["model_path"] = str(model_path("llama3_3_70b_instruct"))
        return TextLLMRunner(**kwargs)
    if name == "qwen3_omni_captioner":
        from baselines.models.audio_captioning import Qwen3OmniCaptionerRunner

        return Qwen3OmniCaptionerRunner(**kwargs)
    if name == "audio_flamingo_next":
        from baselines.models.audio_captioning import AudioFlamingoCaptionerRunner

        return AudioFlamingoCaptionerRunner(**kwargs)
    if name == "f5_tts":
        from baselines.models.tts_cascade import F5TTSRunner

        return F5TTSRunner(**kwargs)
    if name == "cosyvoice3":
        from baselines.models.tts_cascade import CosyVoice3Runner

        return CosyVoice3Runner(**kwargs)
    raise ValueError(f"Unknown baseline model: {name}")
