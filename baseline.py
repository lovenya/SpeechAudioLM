#!/usr/bin/env python
"""Run baseline model inference over a dataset split."""

from __future__ import annotations

import argparse
from pathlib import Path

from baselines.dataset import count_jsonl_rows, iter_split_rows, load_dataset_config
from baselines.models import get_runner
from baselines.run import dry_run_inference, run_inference
from project_conventions import baseline_output_root, default_prompt_field



TEXT_CONVERSION_TEMPLATES = {
    "speech_to_audio_description": (
        "Based on this speech content, describe a likely environmental audio scene that could correspond to it. "
        "Return only one concise audio description suitable for text-to-audio generation. "
        "Do not explain your reasoning and do not ask follow-up questions.\n\n"
        "Speech content: {prompt}"
    ),
    "audio_description_to_speech": (
        "Based on this environmental audio description, write a short speech transcript that could naturally "
        "correspond to the scene. Return only the spoken words, with no explanation and no follow-up question.\n\n"
        "Audio description: {prompt}"
    ),
}

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run baseline inference on a TTAS split.")
    parser.add_argument(
        "--model",
        required=True,
        choices=[
            "audiogen",
            "tangoflux",
            "qwen3_omni",
            "audio_omni",
            "whisper",
            "qwen3_asr",
            "qwen3_32b",
            "llama3_3_70b_instruct",
            "qwen3_omni_captioner",
            "audio_flamingo_next",
            "f5_tts",
            "cosyvoice3",
        ],
        help="Baseline model backend to run.",
    )
    parser.add_argument(
        "--task",
        default="T2A",
        choices=["T2A", "T2S", "A2T", "S2T", "T2T", "text_to_text", "S2A", "A2S", "pseudo_t2a_direct", "pseudo_t2a_phoneme"],
        help="Inference task. Backends validate which tasks they support.",
    )
    parser.add_argument(
        "--dataset-config",
        default="configs/ttas_dataset.json",
        help="Dataset config JSON with split paths.",
    )
    parser.add_argument(
        "--dataset",
        default="ttas",
        help="Dataset name used in output paths. Defaults to ttas.",
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
    parser.add_argument(
        "--continue-on-error",
        action="store_true",
        help="Record per-sample errors in the manifest and continue instead of stopping the run.",
    )
    parser.add_argument("--model-path", default=None, help="Override local model path for the selected backend.")
    parser.add_argument(
        "--prompt-template",
        default=None,
        help="Optional instruction template. Use {prompt} where the row prompt should be inserted.",
    )
    parser.add_argument(
        "--text-mode",
        default=None,
        choices=["speech_to_audio_description", "audio_description_to_speech"],
        help="Direction-specific T2T template to apply when --prompt-template is not provided.",
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
        default="/scratch/l/lovenya/salm_migration_clean/models/tangoflux",
        help="TangoFlux source/model directory.",
    )
    parser.add_argument(
        "--audio-omni-config",
        default="/scratch/l/lovenya/salm_migration_clean/models/audio-omni/Audio-Omni.json",
        help="Audio-Omni config path.",
    )
    parser.add_argument(
        "--audio-omni-ckpt",
        default="/scratch/l/lovenya/salm_migration_clean/models/audio-omni/model.ckpt",
        help="Audio-Omni checkpoint path.",
    )
    parser.add_argument(
        "--audio-omni-source-dir",
        default="/scratch/l/lovenya/salm_migration_clean/code/Audio-Omni",
        help="Audio-Omni source directory. Used internally for relative config assets.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Validate dataset rows and output paths without loading a model.",
    )
    return parser.parse_args()


def _default_prompt_field(model: str, task: str) -> str:
    if model == "qwen3_omni" and task == "A2S":
        return "speech_text"
    if task in {"T2T", "text_to_text"}:
        return "speech_text"
    return default_prompt_field(task)


def main() -> None:
    args = parse_args()
    config = load_dataset_config(Path(args.dataset_config))
    split_path = Path(args.split_file) if args.split_file else Path(config["splits"][args.split])
    prompt_field = args.prompt_field or _default_prompt_field(args.model, args.task)
    runner_task = "text_to_text" if args.task == "T2T" else args.task
    prompt_template = args.prompt_template
    if runner_task == "text_to_text" and prompt_template is None:
        if args.text_mode is None:
            raise SystemExit("T2T/text_to_text requires --prompt-template or --text-mode")
        prompt_template = TEXT_CONVERSION_TEMPLATES[args.text_mode]

    rows = iter_split_rows(
        split_path,
        prompt_field=prompt_field,
        start_index=args.start_index,
        num_samples=args.num_samples,
    )
    output_root = baseline_output_root(args.model, args.task, args.dataset, args.split, Path(args.output_dir))
    log_root = baseline_output_root(args.model, args.task, args.dataset, args.split, Path(args.log_dir))
    total_rows = count_jsonl_rows(
        split_path,
        start_index=args.start_index,
        num_samples=args.num_samples,
    )

    run_config = {
        "model": args.model,
        "task": args.task,
        "dataset": args.dataset,
        "split": args.split,
        "split_file": str(split_path),
        "prompt_field": prompt_field,
        "duration": args.duration,
        "steps": args.steps,
        "seed": args.seed,
        "text_mode": args.text_mode,
        "prompt_template": prompt_template,
    }
    try:
        runner = get_runner(
            args.model,
            model_path=args.model_path,
            device=args.device,
            duration=args.duration,
            steps=args.steps,
            seed=args.seed,
            task=runner_task,
            prompt_template=prompt_template,
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
        continue_on_error=args.continue_on_error,
    )


if __name__ == "__main__":
    main()
