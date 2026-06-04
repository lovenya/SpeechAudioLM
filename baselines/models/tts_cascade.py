from __future__ import annotations

from pathlib import Path

from baselines.dataset import PromptRow
from baselines.models.base import BaselineRunner, GenerationResult


class F5TTSRunner(BaselineRunner):
    output_subdir = "audio"
    extension = ".wav"

    def __init__(
        self,
        f5_tts_source_dir: str = "/scratch/l/lovenya/salm_migration_clean/code/cascaded/F5-TTS",
        f5_tts_vocoder_path: str = "/scratch/l/lovenya/salm_migration_clean/models/cascaded/tts/vocos-mel-24khz",
        **kwargs: object,
    ) -> None:
        super().__init__(**kwargs)
        if self.task != "T2S":
            raise ValueError("F5-TTS supports only task T2S")
        self.model_path = self.model_path or "/scratch/l/lovenya/salm_migration_clean/models/cascaded/tts/f5-tts"
        self.source_dir = Path(f5_tts_source_dir)
        self.vocoder_path = Path(f5_tts_vocoder_path)
        self.model = None

    def load(self) -> None:
        import sys

        source = str(self.source_dir / "src")
        if source not in sys.path:
            sys.path.insert(0, source)
        from f5_tts.api import F5TTS

        root = Path(self.model_path)
        self.model = F5TTS(
            model="F5TTS_v1_Base",
            ckpt_file=str(root / "F5TTS_v1_Base" / "model_1250000.safetensors"),
            vocab_file=str(root / "F5TTS_v1_Base" / "vocab.txt"),
            device=self.device,
            vocoder_local_path=str(self.vocoder_path),
            hf_cache_dir="/scratch/l/lovenya/salm_migration_clean/models/hf_cache",
        )

    def generate(self, row: PromptRow, output_path: Path) -> GenerationResult:
        if self.model is None:
            raise RuntimeError("F5-TTS model is not loaded")
        ref_audio = self._record_path(row, _reference_field(row, "reference_speech_file", "speech_file"))
        ref_text = _required_record_text(row, _reference_field(row, "reference_speech_text", "speech_text"))
        gen_text = self._format_prompt(row.prompt)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        wav, sr, _ = self.model.infer(
            ref_file=str(ref_audio),
            ref_text=ref_text,
            gen_text=gen_text,
            nfe_step=self.steps,
            fix_duration=self.duration if self.duration > 0 else None,
            file_wave=str(output_path),
            seed=self.seed,
        )
        extra = {
            "shape": list(getattr(wav, "shape", [])),
            "source_reference_file": str(ref_audio),
            "reference_text": ref_text,
            "generation_prompt": gen_text,
            "backend": "f5_tts",
            "stages": [
                {
                    "stage": "text_to_speech",
                    "method": "f5_tts.api.F5TTS.infer",
                    "prompt": gen_text,
                    "reference_file": str(ref_audio),
                    "output_file": str(output_path),
                }
            ],
        }
        return GenerationResult(output_path=output_path, sample_rate=int(sr), extra=extra)


class CosyVoice3Runner(BaselineRunner):
    output_subdir = "audio"
    extension = ".wav"

    def __init__(
        self,
        cosyvoice_source_dir: str = "/scratch/l/lovenya/salm_migration_clean/code/cascaded/CosyVoice",
        **kwargs: object,
    ) -> None:
        super().__init__(**kwargs)
        if self.task != "T2S":
            raise ValueError("CosyVoice3 supports only task T2S")
        self.model_path = self.model_path or "/scratch/l/lovenya/salm_migration_clean/models/cascaded/tts/fun-cosyvoice3-0.5b-2512"
        self.source_dir = Path(cosyvoice_source_dir)
        self.model = None

    def load(self) -> None:
        import sys

        source = str(self.source_dir)
        matcha = str(self.source_dir / "third_party" / "Matcha-TTS")
        for path in (source, matcha):
            if path not in sys.path:
                sys.path.insert(0, path)
        from cosyvoice.cli.cosyvoice import AutoModel

        self.model = AutoModel(model_dir=self.model_path, fp16=self.device.startswith("cuda"))

    def generate(self, row: PromptRow, output_path: Path) -> GenerationResult:
        if self.model is None:
            raise RuntimeError("CosyVoice3 model is not loaded")
        import torch
        import torchaudio

        ref_audio = self._record_path(row, _reference_field(row, "reference_speech_file", "speech_file"))
        ref_text = _required_record_text(row, _reference_field(row, "reference_speech_text", "speech_text"))
        gen_text = self._format_prompt(row.prompt)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        chunks = []
        for item in self.model.inference_zero_shot(
            gen_text,
            f"You are a helpful assistant.<|endofprompt|>{ref_text}",
            str(ref_audio),
            stream=False,
        ):
            chunks.append(item["tts_speech"].detach().cpu())
        if not chunks:
            raise RuntimeError("CosyVoice3 returned no audio chunks")
        audio = torch.cat(chunks, dim=1) if len(chunks) > 1 else chunks[0]
        torchaudio.save(str(output_path), audio, self.model.sample_rate)
        extra = {
            "shape": list(audio.shape),
            "source_reference_file": str(ref_audio),
            "reference_text": ref_text,
            "generation_prompt": gen_text,
            "backend": "cosyvoice3",
            "stages": [
                {
                    "stage": "text_to_speech",
                    "method": "cosyvoice.cli.cosyvoice.AutoModel.inference_zero_shot",
                    "prompt": gen_text,
                    "reference_file": str(ref_audio),
                    "output_file": str(output_path),
                }
            ],
        }
        return GenerationResult(output_path=output_path, sample_rate=int(self.model.sample_rate), extra=extra)


def _reference_field(row: PromptRow, preferred: str, fallback: str) -> str:
    value = row.record.get(preferred)
    return preferred if isinstance(value, str) and value.strip() else fallback


def _required_record_text(row: PromptRow, field: str) -> str:
    value = row.record.get(field)
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"Row {row.row_index} has no usable {field!r}")
    return value.strip()
