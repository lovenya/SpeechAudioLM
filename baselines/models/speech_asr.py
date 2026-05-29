from __future__ import annotations

import json
from pathlib import Path
from typing import Callable
import sys
import types

from baselines.dataset import PromptRow
from baselines.models.base import BaselineRunner, GenerationResult


class WhisperASRRunner(BaselineRunner):
    output_subdir = "text"
    extension = ".json"

    def __init__(self, **kwargs: object) -> None:
        super().__init__(**kwargs)
        if self.task != "S2T":
            raise ValueError("Whisper ASR supports only task S2T")
        self.model_path = self.model_path or "/scratch/l/lovenya/salm_migration_clean/models/cascaded/asr/whisper-large-v3"
        self.pipe = None

    def load(self) -> None:
        import torch
        from transformers import AutoModelForSpeechSeq2Seq, AutoProcessor, pipeline

        dtype = torch.float16 if self.device.startswith("cuda") and torch.cuda.is_available() else torch.float32
        device = _pipeline_device(self.device) if torch.cuda.is_available() else -1
        processor = AutoProcessor.from_pretrained(self.model_path, local_files_only=True)
        model = AutoModelForSpeechSeq2Seq.from_pretrained(
            self.model_path,
            local_files_only=True,
            dtype=dtype,
        )
        self.pipe = pipeline(
            "automatic-speech-recognition",
            model=model,
            tokenizer=processor.tokenizer,
            feature_extractor=processor.feature_extractor,
            device=device,
        )

    def generate(self, row: PromptRow, output_path: Path) -> GenerationResult:
        if self.pipe is None:
            raise RuntimeError("Whisper ASR model is not loaded")
        speech_path = self._record_path(row, "speech_file")
        result = self.pipe(str(speech_path), return_timestamps=False)
        text = _extract_text(result)
        extra = {
            "source_input_file": str(speech_path),
            "understanding_text": text,
            "generation_prompt": row.prompt,
            "backend": "whisper",
            "stages": [
                {
                    "stage": "asr",
                    "method": "transformers.pipeline(automatic-speech-recognition)",
                    "input_field": "speech_file",
                    "input_file": str(speech_path),
                    "output_text": text,
                    "output_file": str(output_path),
                }
            ],
        }
        _write_text_result(output_path, row, "S2T", text, extra)
        return GenerationResult(output_path=output_path, sample_rate=None, extra=extra)


class Qwen3ASRRunner(BaselineRunner):
    output_subdir = "text"
    extension = ".json"

    def __init__(self, **kwargs: object) -> None:
        super().__init__(**kwargs)
        if self.task != "S2T":
            raise ValueError("Qwen3-ASR supports only task S2T")
        self.model_path = self.model_path or "/scratch/l/lovenya/salm_migration_clean/models/cascaded/asr/qwen3-asr-1.7b"
        self.model = None

    def load(self) -> None:
        import torch

        _patch_qwen_asr_transformers_compat()
        from qwen_asr import Qwen3ASRModel

        self.model = Qwen3ASRModel.from_pretrained(
            self.model_path,
            dtype=torch.bfloat16,
            device_map=_torch_device(self.device) if self.device.startswith("cuda") else "cpu",
            max_inference_batch_size=1,
            max_new_tokens=256,
        )

    def generate(self, row: PromptRow, output_path: Path) -> GenerationResult:
        if self.model is None:
            raise RuntimeError("Qwen3-ASR model is not loaded")
        speech_path = self._record_path(row, "speech_file")
        results = self.model.transcribe(audio=str(speech_path), language=None)
        first = results[0]
        text = str(getattr(first, "text", first)).strip()
        language = getattr(first, "language", None)
        extra = {
            "source_input_file": str(speech_path),
            "understanding_text": text,
            "language": language,
            "generation_prompt": row.prompt,
            "backend": "qwen3_asr",
            "stages": [
                {
                    "stage": "asr",
                    "method": "qwen_asr.Qwen3ASRModel.transcribe",
                    "input_field": "speech_file",
                    "input_file": str(speech_path),
                    "language": language,
                    "output_text": text,
                    "output_file": str(output_path),
                }
            ],
        }
        _write_text_result(output_path, row, "S2T", text, extra)
        return GenerationResult(output_path=output_path, sample_rate=None, extra=extra)


def _patch_qwen_asr_transformers_compat() -> None:
    import transformers.utils.generic as generic

    if not hasattr(generic, "check_model_inputs"):
        def check_model_inputs(*args, **kwargs) -> Callable:
            def decorator(fn: Callable) -> Callable:
                return fn

            return decorator

        generic.check_model_inputs = check_model_inputs

    try:
        from transformers import modeling_rope_utils
    except ImportError:
        modeling_rope_utils = None
    if modeling_rope_utils is not None and "default" not in modeling_rope_utils.ROPE_INIT_FUNCTIONS:
        modeling_rope_utils.ROPE_INIT_FUNCTIONS["default"] = _compute_default_rope_parameters

    if "nagisa" not in sys.modules:
        nagisa = types.ModuleType("nagisa")

        class _NagisaResult:
            def __init__(self, text: str) -> None:
                self.words = [text] if text else []

        def tagging(text: str) -> _NagisaResult:
            return _NagisaResult(text)

        nagisa.tagging = tagging
        sys.modules["nagisa"] = nagisa

    _patch_qwen_asr_config_compat()


def _patch_qwen_asr_config_compat() -> None:
    try:
        from qwen_asr.core.transformers_backend.configuration_qwen3_asr import Qwen3ASRThinkerConfig
    except ImportError:
        return
    if getattr(Qwen3ASRThinkerConfig, "_salm_pad_token_patch", False):
        return

    original_init = Qwen3ASRThinkerConfig.__init__

    def patched_init(self, *args, **kwargs):
        original_init(self, *args, **kwargs)
        if not hasattr(self, "pad_token_id"):
            self.pad_token_id = None

    Qwen3ASRThinkerConfig.__init__ = patched_init
    Qwen3ASRThinkerConfig._salm_pad_token_patch = True


def _compute_default_rope_parameters(config=None, device=None, seq_len: int | None = None, layer_type: str | None = None):
    import torch

    if config is None:
        raise ValueError("config is required for default RoPE parameters")
    if hasattr(config, "standardize_rope_params"):
        config.standardize_rope_params()
    rope_parameters = getattr(config, "rope_parameters", None)
    if isinstance(rope_parameters, dict):
        if layer_type is not None and isinstance(rope_parameters.get(layer_type), dict):
            rope_parameters = rope_parameters[layer_type]
    else:
        rope_parameters = getattr(config, "rope_scaling", {}) or {}

    base = rope_parameters.get("rope_theta", getattr(config, "rope_theta", 10000.0))
    partial_rotary_factor = rope_parameters.get(
        "partial_rotary_factor",
        getattr(config, "partial_rotary_factor", 1.0),
    )
    head_dim = getattr(config, "head_dim", None) or config.hidden_size // config.num_attention_heads
    dim = int(head_dim * partial_rotary_factor)
    inv_freq = 1.0 / (base ** (torch.arange(0, dim, 2, dtype=torch.int64).to(device=device, dtype=torch.float) / dim))
    return inv_freq, 1.0


def _extract_text(result: object) -> str:
    if isinstance(result, dict):
        value = result.get("text")
    else:
        value = getattr(result, "text", result)
    return str(value or "").strip()


def _pipeline_device(device: str) -> int:
    if device == "cuda":
        return 0
    if device.startswith("cuda:"):
        return int(device.split(":", 1)[1])
    return -1


def _torch_device(device: str) -> str:
    if device == "cuda":
        return "cuda:0"
    return device


def _write_text_result(output_path: Path, row: PromptRow, task: str, text: str, extra: dict) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(
        json.dumps(
            {
                "row_index": row.row_index,
                "sample_id": row.sample_id,
                "task": task,
                "text": text,
                "extra": extra,
            },
            ensure_ascii=False,
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )
