#!/usr/bin/env python
"""Run modular baseline stages and cascades over a dataset split."""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path
from typing import Iterable

from baselines.dataset import PromptRow, count_jsonl_rows, iter_split_rows, load_dataset_config
from baselines.models import get_runner
from baselines.models.base import BaselineRunner, GenerationResult
from project_conventions import baseline_output_root, default_prompt_field


PIPELINES = ("T2A", "T2S", "A2T", "S2T", "A2S", "S2A")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run modular cascaded baseline stages.")
    parser.add_argument("--pipeline", required=True, choices=PIPELINES)
    parser.add_argument("--dataset-config", default="configs/ttas_dataset.json")
    parser.add_argument("--dataset", default="ttas")
    parser.add_argument("--split", default="test", choices=["train", "val", "test"])
    parser.add_argument("--split-file", default=None)
    parser.add_argument("--num-samples", type=int, default=5)
    parser.add_argument("--start-index", type=int, default=0)
    parser.add_argument("--output-dir", default="results/baselines")
    parser.add_argument("--log-dir", default="logs/baselines")
    parser.add_argument("--run-name", default=None, help="Optional output variant name; defaults to selected stage backends.")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--duration", type=float, default=10.0)
    parser.add_argument("--steps", type=int, default=25)
    parser.add_argument("--seed", type=int, default=1337)
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--continue-on-error", action="store_true")
    parser.add_argument("--dry-run", action="store_true")

    parser.add_argument("--asr", default="audio_omni", choices=["audio_omni", "reference", "whisper", "qwen3_asr"])
    parser.add_argument("--captioner", default="audio_omni", choices=["audio_omni", "reference", "audio_flamingo_next", "qwen3_omni_captioner"])
    parser.add_argument("--text-model", default="template", choices=["template", "identity", "qwen3_32b", "llama3_3_70b_instruct"])
    parser.add_argument("--tts", default="audio_omni", choices=["audio_omni", "qwen3_omni", "f5_tts", "cosyvoice3"])
    parser.add_argument("--t2a", default="audiogen", choices=["audiogen", "tangoflux", "audio_omni"])

    parser.add_argument("--model-path", default=None, help="Fallback model path override for single-backend runs.")
    parser.add_argument("--asr-model-path", default=None)
    parser.add_argument("--captioner-model-path", default=None)
    parser.add_argument("--text-model-path", default=None)
    parser.add_argument("--tts-model-path", default=None)
    parser.add_argument("--t2a-model-path", default=None)
    parser.add_argument("--prompt-template", default=None)
    parser.add_argument("--qwen-speaker", default="Ethan")
    parser.add_argument("--qwen-template", default=None)
    parser.add_argument("--tangoflux-source-dir", default="/scratch/l/lovenya/salm_migration_clean/models/tangoflux")
    parser.add_argument("--audio-omni-config", default="/scratch/l/lovenya/salm_migration_clean/models/audio-omni/Audio-Omni.json")
    parser.add_argument("--audio-omni-ckpt", default="/scratch/l/lovenya/salm_migration_clean/models/audio-omni/model.ckpt")
    parser.add_argument("--audio-omni-source-dir", default="/scratch/l/lovenya/salm_migration_clean/code/Audio-Omni")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config = load_dataset_config(Path(args.dataset_config))
    split_path = Path(args.split_file) if args.split_file else Path(config["splits"][args.split])
    prompt_field = _pipeline_prompt_field(args.pipeline)
    rows = list(
        iter_split_rows(
            split_path,
            prompt_field=prompt_field,
            start_index=args.start_index,
            num_samples=args.num_samples,
        )
    )
    total_rows = count_jsonl_rows(split_path, args.start_index, args.num_samples)
    output_root = _cascade_output_root(args)
    log_root = _cascade_output_root(args, root=Path(args.log_dir))
    run_config = _run_config(args, split_path, prompt_field)

    cascade = ModularCascade(args, dataset_root=config.get("dataset_root"))
    if args.dry_run:
        _dry_run(args, rows, output_root, run_config, total_rows)
        return
    cascade.run(rows, output_root, log_root, run_config, total_rows)


class ModularCascade:
    def __init__(self, args: argparse.Namespace, dataset_root: str | None) -> None:
        self.args = args
        self.dataset_root = dataset_root
        self._runners: dict[tuple[str, str], BaselineRunner] = {}

    def run(
        self,
        rows: Iterable[PromptRow],
        output_root: Path,
        log_root: Path,
        run_config: dict,
        total_rows: int,
    ) -> None:
        output_root.mkdir(parents=True, exist_ok=True)
        log_root.mkdir(parents=True, exist_ok=True)
        final_manifest = output_root / "manifest.jsonl"
        run_config_path = output_root / "run_config.json"
        run_config_path.write_text(json.dumps(run_config, indent=2, sort_keys=True) + "\n", encoding="utf-8")

        processed = 0
        skipped = 0
        failed = 0
        started = time.time()
        print(f"output_root={output_root}")
        print(f"manifest={final_manifest}")
        print(f"run_config={run_config_path}")
        print(f"total_rows={total_rows}")

        with final_manifest.open("a", encoding="utf-8") as manifest:
            for ordinal, row in enumerate(rows, start=1):
                output_path = self._final_output_path(output_root, row)
                if output_path.exists() and not self.args.overwrite:
                    skipped += 1
                    print(f"skip [{ordinal}/{total_rows}] row={row.row_index} output={output_path}")
                    continue
                print(f"run [{ordinal}/{total_rows}] row={row.row_index} sample_id={row.sample_id}")
                row_start = time.time()
                try:
                    record = self._run_row(row, output_root, output_path, run_config)
                except Exception as exc:
                    if not self.args.continue_on_error:
                        raise
                    failed += 1
                    record = {
                        **run_config,
                        "status": "error",
                        "row_index": row.row_index,
                        "sample_id": row.sample_id,
                        "prompt": row.prompt,
                        "output_file": str(output_path),
                        "generation_seconds": round(time.time() - row_start, 3),
                        "source": row.record,
                        "error": {"type": type(exc).__name__, "message": str(exc)},
                    }
                else:
                    processed += 1
                    record["generation_seconds"] = round(time.time() - row_start, 3)
                manifest.write(json.dumps(record, ensure_ascii=False) + "\n")
                manifest.flush()
        print(
            f"done processed={processed} skipped={skipped} failed={failed} "
            f"elapsed={time.time() - started:.2f}s"
        )

    def _run_row(self, row: PromptRow, output_root: Path, output_path: Path, run_config: dict) -> dict:
        pipeline = self.args.pipeline
        if pipeline == "T2A":
            result = self._generate_t2a(row.prompt, row, output_path)
            self._append_component_manifest(output_root, "T2A", row, result.output_path, result.sample_rate, result.extra)
            stages = [_stage_record("text_to_audio", self.args.t2a, row.prompt, result.output_path)]
            return self._success_record(row, output_path, result.sample_rate, stages, result.extra, run_config)
        if pipeline == "T2S":
            result = self._generate_tts(row.prompt, row, output_path)
            self._append_component_manifest(output_root, "T2S", row, result.output_path, result.sample_rate, result.extra)
            stages = [_stage_record("text_to_speech", self.args.tts, row.prompt, result.output_path)]
            return self._success_record(row, output_path, result.sample_rate, stages, result.extra, run_config)
        if pipeline == "A2T":
            text = self._caption_audio(row, output_root)
            stages = [_text_stage_record("audio_captioning", self.args.captioner, text)]
            output_file = self._write_final_text(output_root, row, text, stages)
            return self._text_success_record(row, text, output_file, stages, run_config)
        if pipeline == "S2T":
            text = self._transcribe_speech(row, output_root)
            stages = [_text_stage_record("asr", self.args.asr, text)]
            output_file = self._write_final_text(output_root, row, text, stages)
            return self._text_success_record(row, text, output_file, stages, run_config)
        if pipeline == "A2S":
            audio_description = self._caption_audio(row, output_root)
            speech_text = self._audio_description_to_speech(audio_description)
            self._write_component_text(output_root, "text_to_text", row, speech_text, {"input_text": audio_description})
            result = self._generate_tts(speech_text, row, output_path)
            self._append_component_manifest(output_root, "T2S", row, result.output_path, result.sample_rate, result.extra)
            stages = [
                _text_stage_record("audio_captioning", self.args.captioner, audio_description),
                _text_stage_record("text_to_text", self.args.text_model, speech_text),
                _stage_record("text_to_speech", self.args.tts, speech_text, result.output_path),
            ]
            return self._success_record(row, output_path, result.sample_rate, stages, result.extra, run_config)
        if pipeline == "S2A":
            transcript = self._transcribe_speech(row, output_root)
            audio_prompt = self._speech_to_audio_description(transcript)
            self._write_component_text(output_root, "text_to_text", row, audio_prompt, {"input_text": transcript})
            result = self._generate_t2a(audio_prompt, row, output_path)
            self._append_component_manifest(output_root, "T2A", row, result.output_path, result.sample_rate, result.extra)
            stages = [
                _text_stage_record("asr", self.args.asr, transcript),
                _text_stage_record("text_to_text", self.args.text_model, audio_prompt),
                _stage_record("text_to_audio", self.args.t2a, audio_prompt, result.output_path),
            ]
            return self._success_record(row, output_path, result.sample_rate, stages, result.extra, run_config)
        raise ValueError(f"Unsupported pipeline: {pipeline}")

    def _caption_audio(self, row: PromptRow, output_root: Path) -> str:
        if self.args.captioner == "reference":
            text = _required_text(row, "audio_cap")
            self._write_component_text(output_root, "A2T", row, text, {"backend": "reference"})
            return text
        if self.args.captioner != "audio_omni":
            raise ValueError(f"Captioner backend is not wired yet: {self.args.captioner}")
        text_output = self._component_output_path(output_root, "A2T", row, ".json")
        result = self._runner("audio_omni", "A2T").generate(row, text_output)
        text = str(result.extra.get("understanding_text") or "").strip()
        self._append_component_manifest(output_root, "A2T", row, text_output, None, result.extra, text)
        return text

    def _transcribe_speech(self, row: PromptRow, output_root: Path) -> str:
        if self.args.asr == "reference":
            text = _required_text(row, "speech_text")
            self._write_component_text(output_root, "S2T", row, text, {"backend": "reference"})
            return text
        if self.args.asr != "audio_omni":
            raise ValueError(f"ASR backend is not wired yet: {self.args.asr}")
        text_output = self._component_output_path(output_root, "S2T", row, ".json")
        result = self._runner("audio_omni", "S2T").generate(row, text_output)
        text = str(result.extra.get("understanding_text") or "").strip()
        self._append_component_manifest(output_root, "S2T", row, text_output, None, result.extra, text)
        return text

    def _audio_description_to_speech(self, audio_description: str) -> str:
        if self.args.text_model == "identity":
            return audio_description
        if self.args.text_model != "template":
            raise ValueError(f"Text model backend is not wired yet: {self.args.text_model}")
        return (
            "I can hear it clearly. "
            f"{audio_description.strip()} "
            "We should respond to what is happening there."
        ).strip()

    def _speech_to_audio_description(self, transcript: str) -> str:
        if self.args.text_model == "identity":
            return transcript
        if self.args.text_model != "template":
            raise ValueError(f"Text model backend is not wired yet: {self.args.text_model}")
        return f"Environmental audio matching this spoken situation: {transcript.strip()}"

    def _generate_t2a(self, prompt: str, row: PromptRow, output_path: Path) -> GenerationResult:
        prompt_row = _with_prompt(row, prompt)
        return self._runner(self.args.t2a, "T2A").generate(prompt_row, output_path)

    def _generate_tts(self, prompt: str, row: PromptRow, output_path: Path) -> GenerationResult:
        prompt_row = _with_prompt(row, prompt)
        if self.args.tts == "audio_omni":
            return self._runner("audio_omni", "T2S").generate(prompt_row, output_path)
        if self.args.tts == "qwen3_omni":
            return self._runner("qwen3_omni", "T2S").generate(prompt_row, output_path)
        raise ValueError(f"TTS backend is not wired yet: {self.args.tts}")

    def _runner(self, model: str, task: str) -> BaselineRunner:
        key = (model, task)
        if key not in self._runners:
            runner = get_runner(
                model,
                model_path=self._model_path_for(model, task),
                device=self.args.device,
                duration=self.args.duration,
                steps=self.args.steps,
                seed=self.args.seed,
                task=task,
                prompt_template=self.args.prompt_template,
                dataset_root=self.dataset_root,
                qwen_speaker=self.args.qwen_speaker,
                qwen_template=self.args.qwen_template,
                tangoflux_source_dir=self.args.tangoflux_source_dir,
                audio_omni_config=self.args.audio_omni_config,
                audio_omni_ckpt=self.args.audio_omni_ckpt,
                audio_omni_source_dir=self.args.audio_omni_source_dir,
            )
            print(f"loading stage model={model} task={task}")
            runner.load()
            self._runners[key] = runner
        return self._runners[key]


    def _model_path_for(self, model: str, task: str) -> str | None:
        if task == "S2T":
            return self.args.asr_model_path or self.args.model_path
        if task == "A2T":
            return self.args.captioner_model_path or self.args.model_path
        if task == "T2S":
            return self.args.tts_model_path or self.args.model_path
        if task == "T2A":
            return self.args.t2a_model_path or self.args.model_path
        if model in {"qwen3_32b", "llama3_3_70b_instruct", "template", "identity"}:
            return self.args.text_model_path or self.args.model_path
        return self.args.model_path

    def _component_output_path(self, output_root: Path, task: str, row: PromptRow, extension: str) -> Path:
        path = output_root / "components" / task / "text" / f"{row.row_index:06d}_{row.sample_id}{extension}"
        path.parent.mkdir(parents=True, exist_ok=True)
        return path

    def _final_output_path(self, output_root: Path, row: PromptRow) -> Path:
        if self.args.pipeline in {"A2T", "S2T"}:
            path = output_root / "text" / f"{row.row_index:06d}_{row.sample_id}.json"
        else:
            path = output_root / "audio" / f"{row.row_index:06d}_{row.sample_id}.wav"
        path.parent.mkdir(parents=True, exist_ok=True)
        return path

    def _write_component_text(self, output_root: Path, task: str, row: PromptRow, text: str, extra: dict) -> None:
        output_path = self._component_output_path(output_root, task, row, ".json")
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
        self._append_component_manifest(output_root, task, row, output_path, None, extra, text)

    def _write_final_text(self, output_root: Path, row: PromptRow, text: str, stages: list[dict]) -> Path:
        output_path = output_root / "text" / f"{row.row_index:06d}_{row.sample_id}.json"
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(
            json.dumps(
                {
                    "row_index": row.row_index,
                    "sample_id": row.sample_id,
                    "pipeline": self.args.pipeline,
                    "text": text,
                    "stages": stages,
                },
                ensure_ascii=False,
                indent=2,
            )
            + "\n",
            encoding="utf-8",
        )
        return output_path

    def _append_component_manifest(
        self,
        output_root: Path,
        task: str,
        row: PromptRow,
        output_path: Path,
        sample_rate: int | None,
        extra: dict,
        text: str | None = None,
    ) -> None:
        manifest_path = output_root / "components" / task / "manifest.jsonl"
        manifest_path.parent.mkdir(parents=True, exist_ok=True)
        record = {
            "model": "cascaded",
            "task": task,
            "status": "success",
            "row_index": row.row_index,
            "sample_id": row.sample_id,
            "prompt": row.prompt,
            "output_file": str(output_path),
            "sample_rate": sample_rate,
            "source": row.record,
            "model_extra": extra,
        }
        if text is not None:
            record["candidate_text"] = text
        with manifest_path.open("a", encoding="utf-8") as f:
            f.write(json.dumps(record, ensure_ascii=False) + "\n")

    def _success_record(
        self,
        row: PromptRow,
        output_path: Path,
        sample_rate: int | None,
        stages: list[dict],
        extra: dict,
        run_config: dict,
    ) -> dict:
        return {
            **run_config,
            "status": "success",
            "row_index": row.row_index,
            "sample_id": row.sample_id,
            "prompt": row.prompt,
            "output_file": str(output_path),
            "sample_rate": sample_rate,
            "source": row.record,
            "model_extra": {**extra, "stages": stages},
        }

    def _text_success_record(
        self,
        row: PromptRow,
        text: str,
        output_path: Path,
        stages: list[dict],
        run_config: dict,
    ) -> dict:
        return {
            **run_config,
            "status": "success",
            "row_index": row.row_index,
            "sample_id": row.sample_id,
            "prompt": row.prompt,
            "candidate_text": text,
            "output_file": str(output_path),
            "sample_rate": None,
            "source": row.record,
            "model_extra": {"stages": stages},
        }


def _pipeline_prompt_field(pipeline: str) -> str:
    if pipeline in {"T2A", "A2T", "A2S"}:
        return default_prompt_field("A2T" if pipeline in {"A2T", "A2S"} else "T2A")
    if pipeline in {"T2S", "S2T", "S2A"}:
        return default_prompt_field("S2T" if pipeline in {"S2T", "S2A"} else "T2S")
    raise ValueError(f"Unsupported pipeline: {pipeline}")


def _cascade_output_root(args: argparse.Namespace, root: Path | None = None) -> Path:
    base = Path(args.output_dir) if root is None else root
    return baseline_output_root("cascaded", args.pipeline, _cascade_variant_name(args), args.dataset, base) / args.split


def _cascade_variant_name(args: argparse.Namespace) -> str:
    if args.run_name:
        return _safe_name(args.run_name)
    stage_backends = _planned_stages(args)
    return _safe_name("__".join(f"{stage['stage']}-{stage['backend']}" for stage in stage_backends))


def _safe_name(value: str) -> str:
    safe = []
    for char in value.strip():
        if char.isalnum() or char in {"-", "_", "."}:
            safe.append(char)
        else:
            safe.append("_")
    name = "".join(safe).strip("._-")
    if not name:
        raise ValueError("Output variant name cannot be empty")
    return name


def _run_config(args: argparse.Namespace, split_path: Path, prompt_field: str) -> dict:
    return {
        "model": "cascaded",
        "pipeline": args.pipeline,
        "task": args.pipeline,
        "variant": _cascade_variant_name(args),
        "dataset": args.dataset,
        "split": args.split,
        "split_file": str(split_path),
        "prompt_field": prompt_field,
        "asr": args.asr,
        "captioner": args.captioner,
        "text_model": args.text_model,
        "tts": args.tts,
        "t2a": args.t2a,
        "duration": args.duration,
        "steps": args.steps,
        "seed": args.seed,
    }


def _dry_run(
    args: argparse.Namespace,
    rows: Iterable[PromptRow],
    output_root: Path,
    run_config: dict,
    total_rows: int,
) -> None:
    print("dry_run=true")
    print(f"output_root={output_root}")
    print(f"total_rows={total_rows}")
    print(f"run_config={json.dumps(run_config, ensure_ascii=False, sort_keys=True)}")
    for row in rows:
        print(
            json.dumps(
                {
                    "row_index": row.row_index,
                    "sample_id": row.sample_id,
                    "pipeline": args.pipeline,
                    "stages": _planned_stages(args),
                },
                ensure_ascii=False,
            )
        )


def _planned_stages(args: argparse.Namespace) -> list[dict]:
    if args.pipeline == "A2S":
        return [
            {"stage": "audio_captioning", "backend": args.captioner},
            {"stage": "text_to_text", "backend": args.text_model},
            {"stage": "text_to_speech", "backend": args.tts},
        ]
    if args.pipeline == "S2A":
        return [
            {"stage": "asr", "backend": args.asr},
            {"stage": "text_to_text", "backend": args.text_model},
            {"stage": "text_to_audio", "backend": args.t2a},
        ]
    stage = {
        "A2T": ("audio_captioning", args.captioner),
        "S2T": ("asr", args.asr),
        "T2A": ("text_to_audio", args.t2a),
        "T2S": ("text_to_speech", args.tts),
    }[args.pipeline]
    return [{"stage": stage[0], "backend": stage[1]}]


def _required_text(row: PromptRow, field: str) -> str:
    value = row.record.get(field)
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"Row {row.row_index} has no usable {field!r}")
    return value.strip()


def _with_prompt(row: PromptRow, prompt: str) -> PromptRow:
    return PromptRow(row_index=row.row_index, sample_id=row.sample_id, prompt=prompt, record=row.record)


def _stage_record(stage: str, backend: str, prompt: str, output_path: Path) -> dict:
    return {"stage": stage, "backend": backend, "prompt": prompt, "output_file": str(output_path)}


def _text_stage_record(stage: str, backend: str, text: str) -> dict:
    return {"stage": stage, "backend": backend, "output_text": text}


if __name__ == "__main__":
    main()
