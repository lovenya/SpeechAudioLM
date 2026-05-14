from __future__ import annotations

import json
import os
import tempfile
from pathlib import Path

from baselines.dataset import PromptRow
from baselines.models.base import BaselineRunner, GenerationResult

S2A_SPEECH_UNDERSTANDING_PROMPT = (
    "Transcribe the speech in this audio. If exact transcription is hard, summarize what the speaker is saying. "
    "Return only the speech content."
)
S2A_AUDIO_DESCRIPTION_PROMPT = (
    "Based on this speech content, describe a likely environmental audio scene that could correspond to it. "
    "Return only a concise audio description suitable for text-to-audio generation.\n\n"
    "Speech content: {understanding}"
)
A2S_AUDIO_UNDERSTANDING_PROMPT = (
    "Describe the environmental audio in this recording. Focus on the sound events, setting, intensity, "
    "and mood. Return only the audio description."
)
A2S_SPEECH_TRANSCRIPT_PROMPT = (
    "Based on this environmental audio description, write a short speech transcript that could naturally "
    "correspond to the scene. Return only the spoken words, with no explanation.\n\n"
    "Audio description: {understanding}"
)


class AudioOmniRunner(BaselineRunner):
    def __init__(
        self,
        audio_omni_config: str = "/project/rrg-csubakan/lovenya/models/audio-omni/Audio-Omni.json",
        audio_omni_ckpt: str = "/project/rrg-csubakan/lovenya/models/audio-omni/model.ckpt",
        audio_omni_source_dir: str = "/project/rrg-csubakan/lovenya/src/Audio-Omni",
        **kwargs: object,
    ) -> None:
        super().__init__(**kwargs)
        if self.task not in {"T2A", "T2S", "S2A", "A2S"}:
            raise ValueError("Audio-Omni supports --task T2A, T2S, S2A, or A2S")
        self.config_path = audio_omni_config
        self.ckpt_path = audio_omni_ckpt
        self.source_dir = Path(audio_omni_source_dir)
        self.model = None
        self._patched_config_path: str | None = None

    def load(self) -> None:
        import sys
        import torch

        os.environ.setdefault("QWEN_OMNI_MODEL_PATH", "/project/rrg-csubakan/lovenya/models/qwen2.5-omni-3b")
        os.environ.setdefault("SKIP_QWEN_WEIGHTS", "1")
        source = str(self.source_dir)
        if source not in sys.path:
            sys.path.insert(0, source)
        from audio_omni import AudioOmni

        torch.manual_seed(self.seed)
        config_path = self._prepare_config()
        self.model = AudioOmni(config_path, self.ckpt_path, device=self.device)

    def generate(self, row: PromptRow, output_path: Path) -> GenerationResult:
        if self.model is None:
            raise RuntimeError("Audio-Omni model is not loaded")

        import torch
        import torchaudio

        # The released config has fixed-shape mel conditioning that expects 10 seconds.
        duration = int(round(self.duration))
        if duration != 10:
            raise ValueError("Audio-Omni baseline currently requires --duration 10")

        if self.task == "T2A":
            generation_prompt = self._format_prompt(row.prompt)
            audio = self._generate_audio_omni_audio("T2A", generation_prompt, duration)
            extra = {"shape": list(audio.shape), "generation_prompt": generation_prompt}
        elif self.task == "T2S":
            generation_prompt = self._format_prompt(row.prompt)
            audio = self._generate_audio_omni_audio("TTS", generation_prompt, duration)
            extra = {"shape": list(audio.shape), "generation_prompt": generation_prompt}
        elif self.task == "S2A":
            speech_path = self._record_path(row, "speech_file")
            speech_understanding = self.model.understand(
                S2A_SPEECH_UNDERSTANDING_PROMPT,
                audio=str(speech_path),
            )
            audio_description = self.model.understand(
                self._format_prompt(
                    S2A_AUDIO_DESCRIPTION_PROMPT.format(understanding=speech_understanding),
                )
            )
            audio = self._generate_audio_omni_audio("T2A", audio_description, duration)
            extra = {
                "shape": list(audio.shape),
                "source_input_file": str(speech_path),
                "understanding_text": speech_understanding,
                "converted_prompt": audio_description,
                "generation_prompt": audio_description,
                "final_generation_method": "T2A",
            }
        elif self.task == "A2S":
            audio_path = self._record_path(row, "audio_file")
            audio_understanding = self.model.understand(
                A2S_AUDIO_UNDERSTANDING_PROMPT,
                audio=str(audio_path),
            )
            speech_transcript = self.model.understand(
                self._format_prompt(
                    A2S_SPEECH_TRANSCRIPT_PROMPT.format(understanding=audio_understanding),
                )
            )
            audio = self._generate_audio_omni_audio("TTS", speech_transcript, duration)
            extra = {
                "shape": list(audio.shape),
                "source_input_file": str(audio_path),
                "understanding_text": audio_understanding,
                "converted_prompt": speech_transcript,
                "generation_prompt": speech_transcript,
                "final_generation_method": "TTS",
            }
        else:
            raise ValueError(f"Unsupported Audio-Omni task: {self.task}")

        audio = audio.detach().cpu()
        if audio.ndim == 1:
            audio = audio.unsqueeze(0)
        sample_rate = int(getattr(self.model, "sample_rate", 44100))
        torchaudio.save(str(output_path), audio, sample_rate)
        extra.update(
            {
                "cuda_reserved_gb": round(torch.cuda.memory_reserved() / 1024**3, 2)
                if torch.cuda.is_available()
                else None,
            }
        )
        return GenerationResult(
            output_path=output_path,
            sample_rate=sample_rate,
            extra=extra,
        )

    def _generate_audio_omni_audio(self, task: str, prompt: str, duration: int):
        audio = self.model.generate(
            task,
            prompt=prompt,
            steps=self.steps,
            seconds_total=duration,
            seed=self.seed,
        )
        return audio

    def _prepare_config(self) -> str:
        with open(self.config_path, "r", encoding="utf-8") as f:
            config = json.load(f)

        changed = False
        for conditioner in config.get("model", {}).get("conditioning", {}).get("configs", []):
            cfg = conditioner.get("config", {})
            vocab_file = cfg.get("vocab_file")
            if isinstance(vocab_file, str) and vocab_file.startswith("./"):
                cfg["vocab_file"] = str((self.source_dir / vocab_file[2:]).resolve())
                changed = True

        if not changed:
            return self.config_path

        tmp = tempfile.NamedTemporaryFile(
            mode="w",
            suffix=".json",
            prefix="audio_omni_",
            delete=False,
            encoding="utf-8",
        )
        with tmp:
            json.dump(config, tmp, indent=2)
        self._patched_config_path = tmp.name
        return tmp.name
