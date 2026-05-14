#!/usr/bin/env python
"""Run baseline model inference over a dataset split."""

from __future__ import annotations

import argparse
from pathlib import Path

from baselines.dataset import count_jsonl_rows, iter_split_rows, load_dataset_config
from baselines.models import get_runner
from baselines.run import dry_run_inference, run_inference


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run baseline inference on a TTAS split.")
    parser.add_argument(
        "--model",
        required=True,
        choices=["audiogen", "tangoflux", "qwen3_omni", "audio_omni"],
        help="Baseline model backend to run.",
    )
    parser.add_argument(
        "--task",
        default="T2A",
        choices=["T2A", "T2S", "S2A", "A2S", "pseudo_t2a_direct", "pseudo_t2a_phoneme"],
        help="Inference task. Backends validate which tasks they support.",
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
        default=None,
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
        default="results/baselines",
        help="Output root. The default repo-local results/ directory is ignored by git.",
    )
    parser.add_argument(
        "--log-dir",
        default="logs/baselines",
        help="Log root. The default repo-local logs/ directory is ignored by git.",
    )
    parser.add_argument("--device", default="cuda", help="Torch device for model loading.")
    parser.add_argument("--duration", type=float, default=10.0, help="Requested output duration in seconds.")
    parser.add_argument("--steps", type=int, default=25, help="Diffusion/sampling steps where supported.")
    parser.add_argument("--seed", type=int, default=1337, help="Torch seed used before generation.")
    parser.add_argument(
        "--resource-interval",
        type=float,
        default=10.0,
        help="Seconds between resource-monitor samples. Use 0 to disable.",
    )
    parser.add_argument("--overwrite", action="store_true", help="Regenerate files that already exist.")
    parser.add_argument("--model-path", default=None, help="Override local model path for the selected backend.")
    parser.add_argument(
        "--prompt-template",
        default=None,
        help="Optional instruction template. Use {prompt} where the row prompt should be inserted.",
    )
    parser.add_argument(
        "--qwen-speaker",
        default="Ethan",
        help="Qwen3-Omni voice name. Known upstream options include Ethan, Chelsie, and Aiden.",
    )
    parser.add_argument(
        "--qwen-template",
        default=None,
        help="Backward-compatible Qwen3-Omni instruction template alias. Prefer --prompt-template.",
    )
    parser.add_argument(
        "--tangoflux-source-dir",
        default="/project/rrg-csubakan/lovenya/models/tangoflux",
        help="TangoFlux source/model directory.",
    )
    parser.add_argument(
        "--audio-omni-config",
        default="/project/rrg-csubakan/lovenya/models/audio-omni/Audio-Omni.json",
        help="Audio-Omni config path.",
    )
    parser.add_argument(
        "--audio-omni-ckpt",
        default="/project/rrg-csubakan/lovenya/models/audio-omni/model.ckpt",
        help="Audio-Omni checkpoint path.",
    )
    parser.add_argument(
        "--audio-omni-source-dir",
        default="/project/rrg-csubakan/lovenya/src/Audio-Omni",
        help="Audio-Omni source directory. Used internally for relative config assets.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Validate dataset rows and output paths without loading a model.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config = load_dataset_config(Path(args.dataset_config))
    split_path = Path(args.split_file) if args.split_file else Path(config["splits"][args.split])
    prompt_field = args.prompt_field or default_prompt_field(args.task)

    rows = iter_split_rows(
        split_path,
        prompt_field=prompt_field,
        start_index=args.start_index,
        num_samples=args.num_samples,
    )
    output_root = Path(args.output_dir) / args.model / args.task / args.split / prompt_field
    log_root = Path(args.log_dir) / args.model / args.task / args.split / prompt_field
    total_rows = count_jsonl_rows(
        split_path,
        start_index=args.start_index,
        num_samples=args.num_samples,
    )

    run_config = {
        "model": args.model,
        "task": args.task,
        "split": args.split,
        "split_file": str(split_path),
        "prompt_field": prompt_field,
        "duration": args.duration,
        "steps": args.steps,
        "seed": args.seed,
    }
    try:
        runner = get_runner(
            args.model,
            model_path=args.model_path,
            device=args.device,
            duration=args.duration,
            steps=args.steps,
            seed=args.seed,
            task=args.task,
            prompt_template=args.prompt_template,
            dataset_root=config.get("dataset_root"),
            qwen_speaker=args.qwen_speaker,
            qwen_template=args.qwen_template,
            tangoflux_source_dir=args.tangoflux_source_dir,
            audio_omni_config=args.audio_omni_config,
            audio_omni_ckpt=args.audio_omni_ckpt,
            audio_omni_source_dir=args.audio_omni_source_dir,
        )
    except ValueError as exc:
        raise SystemExit(f"error: {exc}") from None
    if args.dry_run:
        dry_run_inference(
            rows=rows,
            output_root=output_root,
            output_subdir=runner.output_subdir,
            extension=runner.extension,
            run_config=run_config,
            total_rows=total_rows,
        )
        return

    run_inference(
        runner=runner,
        rows=rows,
        output_root=output_root,
        log_root=log_root,
        overwrite=args.overwrite,
        run_config=run_config,
        total_rows=total_rows,
        resource_interval=args.resource_interval,
    )


def default_prompt_field(task: str) -> str:
    if task in {"T2A", "A2S", "pseudo_t2a_direct", "pseudo_t2a_phoneme"}:
        return "audio_cap"
    if task in {"T2S", "S2A"}:
        return "speech_text"
    raise ValueError(f"Unsupported task: {task}")


if __name__ == "__main__":
    main()
