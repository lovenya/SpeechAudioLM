#!/usr/bin/env python
"""Smoke-check OpenAI Whisper for ASR-WER/dWER."""

from __future__ import annotations

import argparse
import json
from pathlib import Path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Smoke test the Whisper ASR backend.")
    parser.add_argument("--cache-dir", default="/project/rrg-csubakan/lovenya/models/metrics/whisper")
    parser.add_argument("--model", default="large-v3")
    parser.add_argument("--audio", default=None, help="Optional audio file to transcribe.")
    parser.add_argument("--load-model", action="store_true", help="Actually load the model. Use compute node for large-v3.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    import whisper

    result = {
        "backend": "openai-whisper",
        "model": args.model,
        "cache_dir": args.cache_dir,
        "available": args.model in whisper.available_models(),
        "loaded": False,
        "transcription": None,
    }
    if args.load_model:
        model = whisper.load_model(args.model, download_root=args.cache_dir)
        result["loaded"] = True
        if args.audio:
            audio_path = Path(args.audio)
            transcription = model.transcribe(str(audio_path), fp16=False)
            result["transcription"] = transcription.get("text", "").strip()
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
