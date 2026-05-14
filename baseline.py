#!/usr/bin/env python
"""Run baseline model inference over a dataset split."""

from __future__ import annotations

import argparse
from pathlib import Path

from baselines.dataset import iter_split_rows, load_dataset_config
from baselines.models import get_runner
from baselines.run import run_inference


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run baseline inference on a TTAS split.")
    parser.add_argument(
        "--model",
        required=True,
        choices=["audiogen", "tangoflux", "qwen3_omni"],
        help="Baseline model backend to run.",
    )
    parser.add_argument(
        "--dataset-config",
        default="configs/ttas_dataset.json",
        help="Dataset config JSON with split paths.",
    )
    parser.add_argument(
        "--split",
        default="test",
        choices=["train", "val", "test"],
        help="Named split from the dataset config.",
    )
    parser.add_argument(
        "--split-file",
        default=None,
        help="Optional explicit JSONL split path. Overrides --split.",
    )
    parser.add_argument(
        "--prompt-field",
        default="speech_text",
        help="Manifest field used as the model prompt, e.g. speech_text or audio_cap.",
    )
    parser.add_argument(
        "--num-samples",
        type=int,
        default=5,
        help="Number of rows to run. Use -1 for the full split.",
    )
    parser.add_argument(
        "--start-index",
        type=int,
        default=0,
        help="Number of split rows to skip before running.",
    )
    parser.add_argument(
        "--output-dir",
        default="/project/rrg-csubakan/lovenya/outputs/baselines",
        help="Output root outside git.",
    )
    parser.add_argument("--device", default="cuda", help="Torch device for model loading.")
    parser.add_argument("--duration", type=float, default=10.0, help="Requested output duration in seconds.")
    parser.add_argument("--steps", type=int, default=25, help="Diffusion/sampling steps where supported.")
    parser.add_argument("--seed", type=int, default=1337, help="Torch seed used before generation.")
    parser.add_argument("--overwrite", action="store_true", help="Regenerate files that already exist.")
    parser.add_argument("--model-path", default=None, help="Override local model path for the selected backend.")
    parser.add_argument(
        "--qwen-speaker",
        default="Ethan",
        help="Qwen3-Omni voice name. Known upstream options include Ethan, Chelsie, and Aiden.",
    )
    parser.add_argument(
        "--qwen-template",
        default="Read the following text aloud exactly, without adding any extra words: {prompt}",
        help="Instruction template for Qwen3-Omni voice generation.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config = load_dataset_config(Path(args.dataset_config))
    split_path = Path(args.split_file) if args.split_file else Path(config["splits"][args.split])

    rows = iter_split_rows(
        split_path,
        prompt_field=args.prompt_field,
        start_index=args.start_index,
        num_samples=args.num_samples,
    )
    runner = get_runner(
        args.model,
        model_path=args.model_path,
        device=args.device,
        duration=args.duration,
        steps=args.steps,
        seed=args.seed,
        qwen_speaker=args.qwen_speaker,
        qwen_template=args.qwen_template,
    )

    output_root = Path(args.output_dir) / args.model / args.split / args.prompt_field
    run_inference(
        runner=runner,
        rows=rows,
        output_root=output_root,
        overwrite=args.overwrite,
        run_config={
            "model": args.model,
            "split": args.split,
            "split_file": str(split_path),
            "prompt_field": args.prompt_field,
            "duration": args.duration,
            "steps": args.steps,
            "seed": args.seed,
        },
    )


if __name__ == "__main__":
    main()
