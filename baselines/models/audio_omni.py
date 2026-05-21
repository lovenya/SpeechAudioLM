from __future__ import annotations

import json
import os
import tempfile
from pathlib import Path

from baselines.dataset import PromptRow
from baselines.models.base import BaselineRunner, GenerationResult

S2A_SPEECH_UNDERSTANDING_PROMPT = (
    "Transcribe the speech in this audio. If exact transcription is hard, summarize what the speaker is saying. "
    "Return only the speech content. Do not ask follow-up questions."
)
S2A_AUDIO_DESCRIPTION_PROMPT = (
    "Based on this speech content, describe a likely environmental audio scene that could correspond to it. "
    "Return only one concise audio description suitable for text-to-audio generation. "
    "Do not explain your reasoning and do not ask follow-up questions.\n\n"
    "Speech content: {understanding}"
)
A2S_AUDIO_UNDERSTANDING_PROMPT = (
    "Describe the environmental audio in this recording. Focus on the sound events, setting, intensity, "
    "and mood. Return only the audio description. Do not ask follow-up questions."
)
A2S_SPEECH_TRANSCRIPT_PROMPT = (
    "Based on this environmental audio description, write a short speech transcript that could naturally "
    "correspond to the scene. Return only the spoken words, with no explanation and no follow-up question.\n\n"
    "Audio description: {understanding}"
)


class AudioOmniRunner(BaselineRunner):
    output_subdir = "audio"
    extension = ".wav"

    def __init__(
        self,
        audio_omni_config: str = "/project/rrg-csubakan/lovenya/models/audio-omni/Audio-Omni.json",
        audio_omni_ckpt: str = "/project/rrg-csubakan/lovenya/models/audio-omni/model.ckpt",
        audio_omni_source_dir: str = "/project/rrg-csubakan/lovenya/src/Audio-Omni",
        **kwargs: object,
    ) -> None:
        super().__init__(**kwargs)
        if self.task not in {"T2A", "T2S", "A2T", "S2T", "S2A", "A2S"}:
            raise ValueError("Audio-Omni supports --task T2A, T2S, A2T, S2T, S2A, or A2S")
        if self.task in {"A2T", "S2T"}:
            self.output_subdir = "text"
            self.extension = ".json"
        self.config_path = audio_omni_config
        self.ckpt_path = audio_omni_ckpt
        self.source_dir = Path(audio_omni_source_dir)
        self.model = None
        self._patched_config_path: str | None = None

    def load(self) -> None:
        import sys
        import torch

        if self.task in {"A2T", "S2T", "S2A", "A2S"}:
            self._check_understanding_dependencies()

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
            extra = {
                "shape": list(audio.shape),
                "generation_prompt": generation_prompt,
                "stages": [
                    {
                        "stage": "text_to_audio",
                        "method": "AudioOmni.generate",
                        "audio_omni_task": "T2A",
                        "input_field": "audio_cap",
                        "prompt": generation_prompt,
                        "output_file": str(output_path),
                    }
                ],
            }
        elif self.task == "T2S":
            generation_prompt = self._format_prompt(row.prompt)
            audio = self._generate_audio_omni_audio("TTS", generation_prompt, duration)
            extra = {
                "shape": list(audio.shape),
                "generation_prompt": generation_prompt,
                "stages": [
                    {
                        "stage": "text_to_speech",
                        "method": "AudioOmni.generate",
                        "audio_omni_task": "TTS",
                        "input_field": "speech_text",
                        "prompt": generation_prompt,
                        "output_file": str(output_path),
                    }
                ],
            }
        elif self.task == "A2T":
            audio_path = self._record_path(row, "audio_file")
            audio_understanding = self._understand(
                A2S_AUDIO_UNDERSTANDING_PROMPT,
                audio=str(audio_path),
            )
            extra = {
                "source_input_file": str(audio_path),
                "understanding_prompt": A2S_AUDIO_UNDERSTANDING_PROMPT,
                "understanding_text": audio_understanding,
                "generation_prompt": audio_understanding,
                "stages": [
                    {
                        "stage": "audio_understanding",
                        "method": "Qwen2.5-Omni thinker",
                        "input_field": "audio_file",
                        "input_file": str(audio_path),
                        "prompt": A2S_AUDIO_UNDERSTANDING_PROMPT,
                        "output_text": audio_understanding,
                        "output_file": str(output_path),
                    }
                ],
            }
            self._write_text_result(output_path, row, extra)
            return GenerationResult(output_path=output_path, sample_rate=None, extra=extra)
        elif self.task == "S2T":
            speech_path = self._record_path(row, "speech_file")
            speech_understanding = self._understand(
                S2A_SPEECH_UNDERSTANDING_PROMPT,
                audio=str(speech_path),
            )
            extra = {
                "source_input_file": str(speech_path),
                "understanding_prompt": S2A_SPEECH_UNDERSTANDING_PROMPT,
                "understanding_text": speech_understanding,
                "generation_prompt": speech_understanding,
                "stages": [
                    {
                        "stage": "speech_understanding",
                        "method": "Qwen2.5-Omni thinker",
                        "input_field": "speech_file",
                        "input_file": str(speech_path),
                        "prompt": S2A_SPEECH_UNDERSTANDING_PROMPT,
                        "output_text": speech_understanding,
                        "output_file": str(output_path),
                    }
                ],
            }
            self._write_text_result(output_path, row, extra)
            return GenerationResult(output_path=output_path, sample_rate=None, extra=extra)
        elif self.task == "S2A":
            speech_path = self._record_path(row, "speech_file")
            speech_understanding = self._understand(
                S2A_SPEECH_UNDERSTANDING_PROMPT,
                audio=str(speech_path),
            )
            conversion_prompt = self._format_prompt(
                S2A_AUDIO_DESCRIPTION_PROMPT.format(understanding=speech_understanding),
            )
            audio_description = self._understand(
                conversion_prompt,
            )
            audio = self._generate_audio_omni_audio("T2A", audio_description, duration)
            extra = {
                "shape": list(audio.shape),
                "source_input_file": str(speech_path),
                "understanding_prompt": S2A_SPEECH_UNDERSTANDING_PROMPT,
                "understanding_text": speech_understanding,
                "conversion_prompt": conversion_prompt,
                "converted_prompt": audio_description,
                "generation_prompt": audio_description,
                "final_generation_method": "T2A",
                "stages": [
                    {
                        "stage": "speech_understanding",
                        "method": "Qwen2.5-Omni thinker",
                        "input_field": "speech_file",
                        "input_file": str(speech_path),
                        "prompt": S2A_SPEECH_UNDERSTANDING_PROMPT,
                        "output_text": speech_understanding,
                    },
                    {
                        "stage": "speech_to_audio_description",
                        "method": "Qwen2.5-Omni thinker",
                        "input_text": speech_understanding,
                        "prompt": conversion_prompt,
                        "output_text": audio_description,
                    },
                    {
                        "stage": "text_to_audio",
                        "method": "AudioOmni.generate",
                        "audio_omni_task": "T2A",
                        "prompt": audio_description,
                        "output_file": str(output_path),
                    },
                ],
            }
        elif self.task == "A2S":
            audio_path = self._record_path(row, "audio_file")
            audio_understanding = self._understand(
                A2S_AUDIO_UNDERSTANDING_PROMPT,
                audio=str(audio_path),
            )
            conversion_prompt = self._format_prompt(
                A2S_SPEECH_TRANSCRIPT_PROMPT.format(understanding=audio_understanding),
            )
            speech_transcript = self._understand(
                conversion_prompt,
            )
            audio = self._generate_audio_omni_audio("TTS", speech_transcript, duration)
            extra = {
                "shape": list(audio.shape),
                "source_input_file": str(audio_path),
                "understanding_prompt": A2S_AUDIO_UNDERSTANDING_PROMPT,
                "understanding_text": audio_understanding,
                "conversion_prompt": conversion_prompt,
                "converted_prompt": speech_transcript,
                "generation_prompt": speech_transcript,
                "final_generation_method": "TTS",
                "stages": [
                    {
                        "stage": "audio_understanding",
                        "method": "Qwen2.5-Omni thinker",
                        "input_field": "audio_file",
                        "input_file": str(audio_path),
                        "prompt": A2S_AUDIO_UNDERSTANDING_PROMPT,
                        "output_text": audio_understanding,
                    },
                    {
                        "stage": "audio_description_to_speech_transcript",
                        "method": "Qwen2.5-Omni thinker",
                        "input_text": audio_understanding,
                        "prompt": conversion_prompt,
                        "output_text": speech_transcript,
                    },
                    {
                        "stage": "text_to_speech",
                        "method": "AudioOmni.generate",
                        "audio_omni_task": "TTS",
                        "prompt": speech_transcript,
                        "output_file": str(output_path),
                    },
                ],
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

    def _write_text_result(self, output_path: Path, row: PromptRow, extra: dict) -> None:
        output_path.parent.mkdir(parents=True, exist_ok=True)
        with output_path.open("w", encoding="utf-8") as f:
            json.dump(
                {
                    "row_index": row.row_index,
                    "sample_id": row.sample_id,
                    "task": self.task,
                    "text": extra["understanding_text"],
                    "source_input_file": extra["source_input_file"],
                    "prompt": extra["understanding_prompt"],
                    "stages": extra["stages"],
                },
                f,
                ensure_ascii=False,
                indent=2,
            )
            f.write("\n")

    def _understand(self, question: str, audio: str | None = None, video: str | None = None) -> str:
        if self.model is None:
            raise RuntimeError("Audio-Omni model is not loaded")

        import torch
        from audio_omni.prompts import SYSTEM_UNDERSTANDING
        from qwen_omni_utils import process_mm_info

        omni_cond = self.model._get_omni_conditioner()
        processor = omni_cond.processor
        omni_model = omni_cond.model

        if audio:
            self._validate_understanding_audio(Path(audio))

        user_content = []
        if video:
            user_content.append({"type": "video", "video": video})
        if audio:
            user_content.append({"type": "audio", "audio": audio})
        user_content.append({"type": "text", "text": question})

        conversation = [
            {"role": "system", "content": [{"type": "text", "text": SYSTEM_UNDERSTANDING}]},
            {"role": "user", "content": user_content},
        ]

        text = processor.apply_chat_template(conversation, add_generation_prompt=True, tokenize=False)
        audios, images, videos = process_mm_info(conversation, use_audio_in_video=True)
        inputs = processor(
            text=text,
            audio=audios,
            images=images,
            videos=videos,
            return_tensors="pt",
            padding=True,
            use_audio_in_video=True,
        )
        thinker = omni_model.thinker if hasattr(omni_model, "thinker") else omni_model
        thinker_device = next(thinker.parameters()).device
        inputs = {
            key: value.to(thinker_device) if isinstance(value, torch.Tensor) else value
            for key, value in inputs.items()
        }

        input_ids = inputs.get("input_ids")
        input_len = input_ids.shape[1] if input_ids is not None else 0
        with torch.no_grad():
            text_ids = thinker.generate(**inputs, use_audio_in_video=True, max_new_tokens=512)
        generated_ids = text_ids[:, input_len:]
        output = processor.batch_decode(
            generated_ids,
            skip_special_tokens=True,
            clean_up_tokenization_spaces=False,
        )
        return output[0].strip() if output else ""

    def _validate_understanding_audio(self, path: Path) -> None:
        if not path.exists():
            raise FileNotFoundError(f"Audio-Omni understanding input does not exist: {path}")
        size = path.stat().st_size
        if size < 4096:
            raise ValueError(
                "Audio-Omni understanding input is too small to decode reliably: "
                f"{path} ({size} bytes)"
            )

    def _check_understanding_dependencies(self) -> None:
        try:
            import qwen_omni_utils  # noqa: F401
        except ModuleNotFoundError as exc:
            raise ModuleNotFoundError(
                "Audio-Omni understanding tasks and cascades require qwen-omni-utils because they call "
                "AudioOmni.understand()."
            ) from exc

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
