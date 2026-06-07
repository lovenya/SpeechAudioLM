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
from baselines.cascaded_manifest import (
    build_sample,
    carried_stage_definitions,
    carried_stages,
    eastern_now,
    output_duration_seconds,
    read_inspection_definitions,
    unwrap_sample,
    wrap_sample,
    write_inspection_manifest,
)
from project_conventions import baseline_output_root, default_prompt_field


PIPELINES = ("T2A", "T2S", "A2T", "S2T", "T2T", "A2S", "S2A")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run modular cascaded baseline stages.")
    parser.add_argument("--pipeline", required=True, choices=PIPELINES)
    parser.add_argument("--dataset-config", default="configs/ttas_dataset.json")
    parser.add_argument("--config-file", default=None, help="Lane configuration recorded in output metadata.")
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
    parser.add_argument(
        "--text-mode",
        default="speech_to_audio_description",
        choices=["speech_to_audio_description", "audio_description_to_speech"],
        help="Standalone T2T conversion mode; cascades set this implicitly.",
    )
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
    parser.add_argument("--f5-tts-source-dir", default="/scratch/l/lovenya/salm_migration_clean/code/cascaded/F5-TTS")
    parser.add_argument("--f5-tts-vocoder-path", default="/scratch/l/lovenya/salm_migration_clean/models/cascaded/tts/vocos-mel-24khz")
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
    rows = _enrich_rows_with_audio_metadata(rows, args.split)
    total_rows = count_jsonl_rows(split_path, args.start_index, args.num_samples)
    output_root = _cascade_output_root(args)
    log_root = _cascade_output_root(args, root=Path(args.log_dir))
    run_config = _run_config(args, split_path, prompt_field)
    if rows and isinstance(rows[0].record.get("_cascaded_pipeline"), str):
        run_config["pipeline"] = rows[0].record["_cascaded_pipeline"]

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
        self._run_config: dict = {}
        self._load_times: dict[str, float] = {}
        self._last_stage_extra: dict = {}
        self._partial_sample: dict | None = None
        self._stage_started_at: float | None = None

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
        inspection_manifest = output_root / "manifest.inspect.json"
        run_config_path = output_root / "run_config.json"
        self._run_config = run_config
        if self.args.overwrite:
            _unlink_if_exists(final_manifest)
            _unlink_if_exists(inspection_manifest)
            for component_manifest in (output_root / "components").glob("*/manifest.jsonl"):
                _unlink_if_exists(component_manifest)
        else:
            _validate_existing_run_config(run_config_path, final_manifest, run_config)
        run_config_path.write_text(json.dumps(run_config, indent=2, sort_keys=True) + "\n", encoding="utf-8")

        processed = 0
        skipped = 0
        failed = 0
        created_at = _existing_created_at(inspection_manifest) or eastern_now()
        started_at = eastern_now()
        started = time.time()
        existing_samples = _read_existing_samples(final_manifest) if not self.args.overwrite else {}
        samples: dict[tuple[int, str], dict] = dict(existing_samples)
        existing_definitions = read_inspection_definitions(final_manifest) if existing_samples else []
        prior_definitions = existing_definitions[:-1] if existing_definitions else []
        print(f"output_root={output_root}")
        print(f"manifest={final_manifest}")
        print(f"inspection_manifest={inspection_manifest}")
        print(f"run_config={run_config_path}")
        print(f"total_rows={total_rows}")

        with final_manifest.open("a", encoding="utf-8") as manifest:
            for ordinal, row in enumerate(rows, start=1):
                row_definitions = carried_stage_definitions(row.record)
                if not prior_definitions:
                    prior_definitions = row_definitions
                elif row_definitions != prior_definitions:
                    raise ValueError(f"Row {row.row_index} carries different stage definitions")
                output_path = self._final_output_path(output_root, row)
                sample_key = (row.row_index, row.sample_id)
                if not self.args.overwrite and sample_key in existing_samples:
                    skipped += 1
                    print(f"skip [{ordinal}/{total_rows}] row={row.row_index} output={output_path}")
                    continue
                if not self.args.overwrite and output_path.exists():
                    raise RuntimeError(f"Output exists without a matching manifest row: {output_path}")
                print(f"run [{ordinal}/{total_rows}] row={row.row_index} sample_id={row.sample_id}")
                row_start = time.time()
                try:
                    if self.args.pipeline in {"A2S", "S2A"}:
                        sample = self._run_full_sample(row, output_root, output_path)
                    else:
                        legacy_record = self._run_row(row, output_root, output_path, run_config)
                        inference_seconds = round(time.time() - row_start, 3)
                        sample = self._sample_from_record(row, legacy_record, inference_seconds)
                except Exception as exc:
                    if not self.args.continue_on_error:
                        raise
                    failed += 1
                    inference_seconds = round(time.time() - (self._stage_started_at or row_start), 3)
                    if self.args.pipeline in {"A2S", "S2A"}:
                        sample = self._full_error_sample(row, inference_seconds, exc)
                    else:
                        sample = self._error_sample(row, output_path, inference_seconds, exc)
                else:
                    if sample.get("status") == "success":
                        processed += 1
                    else:
                        failed += 1
                samples[sample_key] = sample
                manifest.write(json.dumps(wrap_sample(run_config, sample), ensure_ascii=False) + "\n")
                manifest.flush()

        definitions = self._full_stage_definitions() if self.args.pipeline in {"A2S", "S2A"} else prior_definitions + [self._current_stage_definition(len(prior_definitions) + 1)]
        ordered_samples = [samples[key] for key in sorted(samples)]
        final_status = "completed" if all(sample.get("status") == "success" for sample in ordered_samples) else "completed_with_errors"
        write_inspection_manifest(
            inspection_manifest,
            metadata=run_config,
            stage_definitions=definitions,
            samples=ordered_samples,
            created_at=created_at,
            started_at=started_at,
            completed_at=eastern_now(),
            status=final_status,
        )
        (output_root / "model_load_times.json").write_text(
            json.dumps(self._load_times, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        print(
            f"done processed={processed} skipped={skipped} failed={failed} "
            f"elapsed={time.time() - started:.2f}s"
        )

    def _run_row(self, row: PromptRow, output_root: Path, output_path: Path, run_config: dict) -> dict:
        self._last_stage_extra = {}
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
        if pipeline == "T2T":
            text = self._convert_text(row.prompt, self.args.text_mode, output_root, row)
            stages = [_text_stage_record("text_to_text", self.args.text_model, text)]
            output_file = self._write_final_text(output_root, row, text, stages)
            return self._text_success_record(row, text, output_file, stages, run_config)
        if pipeline == "A2S":
            audio_description = self._caption_audio(row, output_root)
            speech_text = self._convert_text(audio_description, "audio_description_to_speech", output_root, row)
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
            audio_prompt = self._convert_text(transcript, "speech_to_audio_description", output_root, row)
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
        self._last_stage_extra = {}
        if self.args.captioner == "reference":
            text = _required_text(row, "audio_cap")
            self._write_component_text(output_root, "A2T", row, text, {"backend": "reference"})
            return text
        if self.args.captioner not in {"audio_omni", "audio_flamingo_next", "qwen3_omni_captioner"}:
            raise ValueError(f"Captioner backend is not wired yet: {self.args.captioner}")
        text_output = self._component_output_path(output_root, "A2T", row, ".json")
        result = self._runner(self.args.captioner, "A2T").generate(row, text_output)
        self._last_stage_extra = dict(result.extra)
        text = str(result.extra.get("understanding_text") or "").strip()
        self._append_component_manifest(output_root, "A2T", row, text_output, None, result.extra, text)
        return text

    def _transcribe_speech(self, row: PromptRow, output_root: Path) -> str:
        self._last_stage_extra = {}
        if self.args.asr == "reference":
            text = _required_text(row, "speech_text")
            self._write_component_text(output_root, "S2T", row, text, {"backend": "reference"})
            return text
        if self.args.asr not in {"audio_omni", "whisper", "qwen3_asr"}:
            raise ValueError(f"ASR backend is not wired yet: {self.args.asr}")
        text_output = self._component_output_path(output_root, "S2T", row, ".json")
        result = self._runner(self.args.asr, "S2T").generate(row, text_output)
        self._last_stage_extra = dict(result.extra)
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

    def _convert_text(self, input_text: str, mode: str, output_root: Path, row: PromptRow) -> str:
        self._last_stage_extra = {}
        if self.args.text_model == "identity":
            text = input_text
            self._write_component_text(output_root, "text_to_text", row, text, {"input_text": input_text, "mode": mode})
            return text
        if self.args.text_model == "template":
            if mode == "audio_description_to_speech":
                text = self._audio_description_to_speech(input_text)
            elif mode == "speech_to_audio_description":
                text = self._speech_to_audio_description(input_text)
            else:
                raise ValueError(f"Unsupported text conversion mode: {mode}")
            self._write_component_text(output_root, "text_to_text", row, text, {"input_text": input_text, "mode": mode})
            return text
        if self.args.text_model not in {"qwen3_32b", "llama3_3_70b_instruct"}:
            raise ValueError(f"Text model backend is not wired yet: {self.args.text_model}")
        prompt = _text_conversion_prompt(input_text, mode)
        text_output = self._component_output_path(output_root, "text_to_text", row, ".json")
        runner = self._runner(self.args.text_model, "text_to_text")
        runner.system_prompt = _text_system_prompt(mode)
        result = runner.generate(_with_prompt(row, prompt), text_output)
        self._last_stage_extra = dict(result.extra)
        text = str(result.extra.get("understanding_text") or "").strip()
        self._append_component_manifest(output_root, "text_to_text", row, text_output, None, {**result.extra, "input_text": input_text, "mode": mode}, text)
        return text

    def _generate_t2a(self, prompt: str, row: PromptRow, output_path: Path) -> GenerationResult:
        prompt_row = _with_prompt(row, prompt)
        return self._runner(self.args.t2a, "T2A").generate(prompt_row, output_path)

    def _generate_tts(self, prompt: str, row: PromptRow, output_path: Path) -> GenerationResult:
        prompt_row = _with_prompt(row, prompt)
        if self.args.tts == "audio_omni":
            return self._runner("audio_omni", "T2S").generate(prompt_row, output_path)
        if self.args.tts == "qwen3_omni":
            return self._runner("qwen3_omni", "T2S").generate(prompt_row, output_path)
        if self.args.tts in {"f5_tts", "cosyvoice3"}:
            return self._runner(self.args.tts, "T2S").generate(prompt_row, output_path)
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
                system_prompt=_text_system_prompt(self.args.text_mode) if task == "text_to_text" else None,
                dataset_root=self.dataset_root,
                qwen_speaker=self.args.qwen_speaker,
                qwen_template=self.args.qwen_template,
                tangoflux_source_dir=self.args.tangoflux_source_dir,
                audio_omni_config=self.args.audio_omni_config,
                audio_omni_ckpt=self.args.audio_omni_ckpt,
                audio_omni_source_dir=self.args.audio_omni_source_dir,
                f5_tts_source_dir=self.args.f5_tts_source_dir,
                f5_tts_vocoder_path=self.args.f5_tts_vocoder_path,
            )
            print(f"loading stage model={model} task={task}")
            load_start = time.time()
            runner.load()
            load_seconds = round(time.time() - load_start, 3)
            load_key = f"{task}:{model}"
            self._load_times[load_key] = load_seconds
            print(f"model_loaded stage={load_key} seconds={load_seconds}")
            self._runners[key] = runner
        return self._runners[key]


    def _run_full_sample(self, row: PromptRow, output_root: Path, output_path: Path) -> dict:
        self._last_stage_extra = {}
        self._partial_sample = None
        self._stage_started_at = time.time()
        sample_record = row.record
        if self.args.pipeline == "A2S":
            started = time.time()
            self._stage_started_at = started
            caption = self._caption_audio(row, output_root)
            sample = build_sample(
                row_index=row.row_index,
                sample_id=row.sample_id,
                record=sample_record,
                stage=self._stage_from_values(
                    stage=1,
                    task="audio_to_text",
                    model=self.args.captioner,
                    input_value={"type": "audio", "file": _required_text(row, "audio_file")},
                    output={"type": "audio_caption", "text": caption},
                    inference_seconds=round(time.time() - started, 3),
                    extra=self._last_stage_extra,
                ),
                status="success",
            )
            self._partial_sample = sample
            started = time.time()
            self._stage_started_at = started
            speech_text = self._convert_text(caption, "audio_description_to_speech", output_root, row)
            sample = build_sample(
                row_index=row.row_index,
                sample_id=row.sample_id,
                record={"_cascaded_sample": sample},
                stage=self._stage_from_values(
                    stage=2,
                    task="text_to_text",
                    model=self.args.text_model,
                    mode="audio_description_to_speech",
                    input_value={"type": "audio_caption", "text": caption},
                    output={"type": "speech_text", "text": speech_text},
                    inference_seconds=round(time.time() - started, 3),
                    extra=self._last_stage_extra,
                ),
                status="success",
            )
            self._partial_sample = sample
            started = time.time()
            self._stage_started_at = started
            result = self._generate_tts(speech_text, row, output_path)
            sample = build_sample(
                row_index=row.row_index,
                sample_id=row.sample_id,
                record={"_cascaded_sample": sample},
                stage=self._stage_from_values(
                    stage=3,
                    task="text_to_speech",
                    model=self.args.tts,
                    input_value={"type": "speech_text", "text": speech_text},
                    output=self._audio_output("speech_audio", result),
                    inference_seconds=round(time.time() - started, 3),
                    extra=result.extra,
                ),
                status="success",
            )
            self._partial_sample = sample
            return sample

        started = time.time()
        self._stage_started_at = started
        transcript = self._transcribe_speech(row, output_root)
        sample = build_sample(
            row_index=row.row_index,
            sample_id=row.sample_id,
            record=sample_record,
            stage=self._stage_from_values(
                stage=1,
                task="speech_to_text",
                model=self.args.asr,
                input_value={"type": "speech_audio", "file": _required_text(row, "speech_file")},
                output={"type": "speech_text", "text": transcript},
                inference_seconds=round(time.time() - started, 3),
                extra=self._last_stage_extra,
            ),
            status="success",
        )
        self._partial_sample = sample
        started = time.time()
        self._stage_started_at = started
        audio_description = self._convert_text(transcript, "speech_to_audio_description", output_root, row)
        sample = build_sample(
            row_index=row.row_index,
            sample_id=row.sample_id,
            record={"_cascaded_sample": sample},
            stage=self._stage_from_values(
                stage=2,
                task="text_to_text",
                model=self.args.text_model,
                mode="speech_to_audio_description",
                input_value={"type": "speech_text", "text": transcript},
                output={"type": "audio_caption", "text": audio_description},
                inference_seconds=round(time.time() - started, 3),
                extra=self._last_stage_extra,
            ),
            status="success",
        )
        self._partial_sample = sample
        started = time.time()
        self._stage_started_at = started
        result = self._generate_t2a(audio_description, row, output_path)
        final_sample = build_sample(
            row_index=row.row_index,
            sample_id=row.sample_id,
            record={"_cascaded_sample": sample},
            stage=self._stage_from_values(
                stage=3,
                task="text_to_audio",
                model=self.args.t2a,
                input_value={"type": "audio_caption", "text": audio_description},
                output=self._audio_output("environmental_audio", result),
                inference_seconds=round(time.time() - started, 3),
                extra=result.extra,
            ),
            status="success",
        )
        self._partial_sample = final_sample
        return final_sample

    def _full_error_sample(self, row: PromptRow, inference_seconds: float, exc: Exception) -> dict:
        completed = self._partial_sample
        stage_number = len(completed.get("stages", [])) + 1 if completed else 1
        specs = (
            [("audio_to_text", self.args.captioner, None), ("text_to_text", self.args.text_model, "audio_description_to_speech"), ("text_to_speech", self.args.tts, None)]
            if self.args.pipeline == "A2S"
            else [("speech_to_text", self.args.asr, None), ("text_to_text", self.args.text_model, "speech_to_audio_description"), ("text_to_audio", self.args.t2a, None)]
        )
        task, model, mode = specs[stage_number - 1]
        if completed:
            previous_output = completed["stages"][-1]["output"]
            input_value = dict(previous_output)
        elif task == "audio_to_text":
            input_value = {"type": "audio", "file": _required_text(row, "audio_file")}
        else:
            input_value = {"type": "speech_audio", "file": _required_text(row, "speech_file")}
        prompt_row = row
        if isinstance(input_value.get("text"), str):
            prompt_row = _with_prompt(row, input_value["text"])
        stage = {
            "stage": stage_number,
            "task": task,
            "model": model,
            "input": input_value,
            "prompt": self._attempted_prompt(task, model, prompt_row, mode=mode),
            "inference_seconds": inference_seconds,
            "output": None,
            "error": {"type": type(exc).__name__, "message": str(exc)},
        }
        if mode:
            stage["mode"] = mode
        source = {"_cascaded_sample": completed} if completed else row.record
        return build_sample(
            row_index=row.row_index,
            sample_id=row.sample_id,
            record=source,
            stage=stage,
            status="error",
            error={"type": type(exc).__name__, "message": str(exc)},
        )

    def _stage_from_values(
        self,
        *,
        stage: int,
        task: str,
        model: str,
        input_value: dict,
        output: dict,
        inference_seconds: float,
        extra: dict,
        mode: str | None = None,
    ) -> dict:
        rendered = extra.get("generation_prompt") or extra.get("understanding_prompt")
        messages = extra.get("messages")
        prompt = None
        if task in {"audio_to_text", "text_to_text"} and (rendered is not None or isinstance(messages, list)):
            prompt = {"rendered": rendered, "messages": messages if isinstance(messages, list) else []}
        result = {
            "stage": stage,
            "task": task,
            "model": model,
            "input": input_value,
            "prompt": prompt,
            "inference_seconds": inference_seconds,
            "output": output,
            "error": None,
        }
        if mode:
            result["mode"] = mode
        return result

    def _audio_output(self, output_type: str, result: GenerationResult) -> dict:
        output = {
            "type": output_type,
            "file": str(result.output_path),
            "sample_rate": result.sample_rate,
            "duration_seconds": output_duration_seconds(result.extra, result.sample_rate),
        }
        return {key: value for key, value in output.items() if value is not None}

    def _sample_from_record(self, row: PromptRow, record: dict, inference_seconds: float) -> dict:
        task, model = self._current_task_and_model()
        extra = dict(record.get("model_extra") or {})
        extra.update(self._last_stage_extra)
        stage_number = len(carried_stages(row.record)) + 1
        output = self._stage_output(task, record, extra)
        rendered_prompt = extra.get("generation_prompt") or extra.get("understanding_prompt")
        messages = extra.get("messages")
        prompt = None
        if task in {"audio_to_text", "text_to_text"} and (rendered_prompt is not None or isinstance(messages, list)):
            prompt = {
                "rendered": rendered_prompt,
                "messages": messages if isinstance(messages, list) else [],
            }
        stage = {
            "stage": stage_number,
            "task": task,
            "model": model,
            "input": self._stage_input(task, row),
            "prompt": prompt,
            "inference_seconds": inference_seconds,
            "output": output,
            "error": None,
        }
        if task == "text_to_text":
            stage["mode"] = self.args.text_mode
        return build_sample(
            row_index=row.row_index,
            sample_id=row.sample_id,
            record=row.record,
            stage=stage,
            status="success",
        )

    def _error_sample(self, row: PromptRow, output_path: Path, inference_seconds: float, exc: Exception) -> dict:
        task, model = self._current_task_and_model()
        stage = {
            "stage": len(carried_stages(row.record)) + 1,
            "task": task,
            "model": model,
            "input": self._stage_input(task, row),
            "prompt": self._attempted_prompt(task, model, row),
            "inference_seconds": inference_seconds,
            "output": None,
            "error": {"type": type(exc).__name__, "message": str(exc)},
        }
        if task == "text_to_text":
            stage["mode"] = self.args.text_mode
        return build_sample(
            row_index=row.row_index,
            sample_id=row.sample_id,
            record=row.record,
            stage=stage,
            status="error",
            error={"type": type(exc).__name__, "message": str(exc)},
        )

    def _stage_input(self, task: str, row: PromptRow) -> dict:
        if task == "audio_to_text":
            return {"type": "audio", "file": _required_text(row, "audio_file")}
        if task == "speech_to_text":
            return {"type": "speech_audio", "file": _required_text(row, "speech_file")}
        if task == "text_to_text":
            input_type = "audio_caption" if self.args.text_mode == "audio_description_to_speech" else "speech_text"
            return {"type": input_type, "text": row.prompt}
        if task == "text_to_speech":
            return {"type": "speech_text", "text": row.prompt}
        if task == "text_to_audio":
            return {"type": "audio_description", "text": row.prompt}
        raise ValueError(f"Unsupported stage task: {task}")

    def _stage_output(self, task: str, record: dict, extra: dict) -> dict:
        if task in {"audio_to_text", "speech_to_text", "text_to_text"}:
            output_type = {
                "audio_to_text": "audio_caption",
                "speech_to_text": "speech_text",
                "text_to_text": "speech_text" if self.args.text_mode == "audio_description_to_speech" else "audio_caption",
            }[task]
            return {"type": output_type, "text": record.get("candidate_text") or extra.get("understanding_text")}
        output_type = "speech_audio" if task == "text_to_speech" else "environmental_audio"
        sample_rate = record.get("sample_rate")
        output = {
            "type": output_type,
            "file": record.get("output_file"),
            "sample_rate": sample_rate,
            "duration_seconds": output_duration_seconds(extra, sample_rate),
        }
        return {key: value for key, value in output.items() if value is not None}

    def _current_task_and_model(self) -> tuple[str, str]:
        return {
            "A2T": ("audio_to_text", self.args.captioner),
            "S2T": ("speech_to_text", self.args.asr),
            "T2T": ("text_to_text", self.args.text_model),
            "T2S": ("text_to_speech", self.args.tts),
            "T2A": ("text_to_audio", self.args.t2a),
        }.get(self.args.pipeline, (self.args.pipeline, "cascaded"))

    def _attempted_prompt(self, task: str, model: str, row: PromptRow, mode: str | None = None) -> dict | None:
        active_mode = mode or self.args.text_mode
        if task == "text_to_text":
            rendered = _text_conversion_prompt(row.prompt, active_mode)
            return {
                "rendered": rendered,
                "messages": [
                    {"role": "system", "content": _text_system_prompt(active_mode)},
                    {"role": "user", "content": rendered},
                ],
            }
        template = _stage_prompt_template(task, model, active_mode, self.args.prompt_template)
        if template:
            rendered = template.format(prompt=row.prompt, input=row.prompt)
            return {"rendered": rendered, "messages": [{"role": "user", "content": rendered}]}
        return None

    def _current_stage_definition(self, stage_number: int) -> dict:
        task, model = self._current_task_and_model()
        return self._definition_for(stage_number, task, model, self.args.text_mode)

    def _full_stage_definitions(self) -> list[dict]:
        if self.args.pipeline == "A2S":
            specs = [
                (1, "audio_to_text", self.args.captioner, None),
                (2, "text_to_text", self.args.text_model, "audio_description_to_speech"),
                (3, "text_to_speech", self.args.tts, None),
            ]
        else:
            specs = [
                (1, "speech_to_text", self.args.asr, None),
                (2, "text_to_text", self.args.text_model, "speech_to_audio_description"),
                (3, "text_to_audio", self.args.t2a, None),
            ]
        return [self._definition_for(*spec) for spec in specs]

    def _definition_for(self, stage_number: int, task: str, model: str, mode: str | None) -> dict:
        runner_task = {"audio_to_text": "A2T", "speech_to_text": "S2T", "text_to_text": "text_to_text", "text_to_speech": "T2S", "text_to_audio": "T2A"}[task]
        runner = self._runners.get((model, runner_task))
        checkpoint = getattr(runner, "model_path", None) or self._model_path_for(model, runner_task)
        is_rule_based = model in {"reference", "template", "identity"}
        definition = {
            "stage": stage_number,
            "task": task,
            "model": model,
            "checkpoint": None if is_rule_based else checkpoint,
            "dtype": None if is_rule_based else _model_dtype(model, self.args.device),
            "prompt_template": _stage_prompt_template(task, model, mode or self.args.text_mode, self.args.prompt_template),
            "generation": {} if is_rule_based else _stage_generation_settings(task, model, self.args),
            "runtime": {
                "device": None if is_rule_based else self.args.device,
                "model_load_seconds": None if is_rule_based else self._load_times.get(f"{runner_task}:{model}"),
            },
        }
        if task == "text_to_text":
            definition["mode"] = mode
            definition["system_prompt"] = _text_system_prompt(mode or self.args.text_mode)
        return definition

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
            "pipeline": self._run_config.get("pipeline"),
            "variant": self._run_config.get("variant"),
            "dataset": self._run_config.get("dataset"),
            "split": self._run_config.get("split"),
            "split_file": self._run_config.get("split_file"),
            "prompt_field": self._run_config.get("prompt_field"),
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


def _enrich_rows_with_audio_metadata(rows: list[PromptRow], split: str) -> list[PromptRow]:
    if all(isinstance(row.record.get("_cascaded_sample"), dict) for row in rows):
        return rows
    class_manifest = Path(f"data/manifests/audio_classes/ttas_{split}_panns_cnn14_classes.jsonl")
    pool_manifest = Path("data/reports/audio_classes/reference_pools/pair_reference_mapping.jsonl")
    metadata_by_id = _records_by_sample_id(class_manifest)
    pools_by_id = _records_by_sample_id(pool_manifest, split=split)
    enriched: list[PromptRow] = []
    for row in rows:
        metadata = metadata_by_id.get(row.sample_id, {})
        pools = pools_by_id.get(row.sample_id, {})
        record = dict(row.record)
        for key in (
            "retained_audio_tags",
            "retained_audio_tag_scores",
            "primary_audio_tag",
            "primary_audio_class",
        ):
            if key not in record and key in metadata:
                record[key] = metadata[key]
        for key in ("audio_to_speech", "speech_to_audio"):
            if key not in record and key in pools:
                record[key] = pools[key]
        enriched.append(PromptRow(row.row_index, row.sample_id, row.prompt, record))
    return enriched


def _records_by_sample_id(path: Path, split: str | None = None) -> dict[str, dict]:
    if not path.is_file():
        return {}
    records: dict[str, dict] = {}
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            if not line.strip():
                continue
            record = json.loads(line)
            if split is not None and record.get("split") != split:
                continue
            sample_id = record.get("sample_id") or record.get("pair_id")
            if not isinstance(sample_id, str):
                for key in ("audio_file", "audio_path", "speech_file", "speech_path"):
                    value = record.get(key)
                    if isinstance(value, str) and value:
                        sample_id = Path(value).stem
                        break
            if isinstance(sample_id, str):
                records[sample_id] = record
    return records


def _pipeline_prompt_field(pipeline: str) -> str:
    if pipeline in {"T2A", "A2T", "A2S"}:
        return default_prompt_field("A2T" if pipeline in {"A2T", "A2S"} else "T2A")
    if pipeline in {"T2S", "S2T", "S2A"}:
        return default_prompt_field("S2T" if pipeline in {"S2T", "S2A"} else "T2S")
    if pipeline == "T2T":
        return default_prompt_field("S2T")
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
        "lane": _cascade_variant_name(args),
        "config_file": args.config_file or args.dataset_config,
        "dataset": args.dataset,
        "split": args.split,
        "split_file": str(split_path),
        "prompt_field": prompt_field,
        "asr": args.asr,
        "captioner": args.captioner,
        "text_model": args.text_model,
        "text_mode": args.text_mode,
        "tts": args.tts,
        "t2a": args.t2a,
        "duration": args.duration,
        "steps": args.steps,
        "seed": args.seed,
        "device": args.device,
        "prompt_template": args.prompt_template,
        "qwen_speaker": args.qwen_speaker,
        "qwen_template": args.qwen_template,
        "model_paths": {
            "fallback": args.model_path,
            "asr": args.asr_model_path,
            "captioner": args.captioner_model_path,
            "text": args.text_model_path,
            "tts": args.tts_model_path,
            "t2a": args.t2a_model_path,
            "tangoflux_source": args.tangoflux_source_dir,
            "audio_omni_config": args.audio_omni_config,
            "audio_omni_checkpoint": args.audio_omni_ckpt,
            "audio_omni_source": args.audio_omni_source_dir,
            "f5_tts_source": args.f5_tts_source_dir,
            "f5_tts_vocoder": args.f5_tts_vocoder_path,
        },
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
    if args.pipeline == "T2T":
        return [{"stage": "text_to_text", "backend": args.text_model}]
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


def _validate_existing_run_config(path: Path, manifest_path: Path, run_config: dict) -> None:
    if not manifest_path.exists() or not path.exists():
        return
    existing = json.loads(path.read_text(encoding="utf-8"))
    keys = sorted(set(existing) | set(run_config))
    mismatches = {key: (existing.get(key), run_config.get(key)) for key in keys if existing.get(key) != run_config.get(key)}
    if mismatches:
        raise RuntimeError(f"Existing output directory belongs to a different run: {mismatches}")


def _existing_created_at(path: Path) -> str | None:
    if not path.is_file():
        return None
    value = json.loads(path.read_text(encoding="utf-8")).get("created_at")
    return value if isinstance(value, str) else None


def _read_existing_samples(path: Path) -> dict[tuple[int, str], dict]:
    if not path.is_file():
        return {}
    samples: dict[tuple[int, str], dict] = {}
    with path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            sample = unwrap_sample(json.loads(line))
            row_index = sample.get("row_index")
            sample_id = sample.get("sample_id")
            if not isinstance(row_index, int) or not isinstance(sample_id, str):
                raise ValueError(f"Invalid existing manifest row at {path}:{line_number}")
            key = (row_index, sample_id)
            if key in samples:
                raise ValueError(f"Duplicate existing manifest row {key} in {path}")
            samples[key] = sample
    return samples


def _unlink_if_exists(path: Path) -> None:
    try:
        path.unlink()
    except FileNotFoundError:
        pass


def _text_conversion_template(mode: str) -> str:
    if mode == "audio_description_to_speech":
        return (
            "Based on this environmental audio description, write a short natural speech transcript that could "
            "realistically be spoken in the scene. Return only the spoken words. Do not include reasoning, "
            "explanations, stage directions, sound effects, audio descriptions, follow-up questions, or "
            "presentation-style openings.\n\nAudio description:\n{input}"
        )
    if mode == "speech_to_audio_description":
        return (
            "Based on this speech content, write one concise environmental audio description suitable for "
            "text-to-audio generation. Describe only plausible audible sound events and setting. Return only the "
            "audio description, without reasoning, dialogue, labels, or follow-up questions.\n\n"
            "Speech content:\n{input}"
        )
    raise ValueError(f"Unsupported text conversion mode: {mode}")


def _text_system_prompt(mode: str) -> str:
    if mode == "audio_description_to_speech":
        return (
            "Generate only natural spoken words. Never provide explanations, reasoning, stage directions, sound "
            "effects, audio descriptions, or presentation-style openings."
        )
    if mode == "speech_to_audio_description":
        return (
            "Generate only a concise environmental audio description. Never provide reasoning, dialogue, labels, "
            "or follow-up questions."
        )
    raise ValueError(f"Unsupported text conversion mode: {mode}")


def _text_conversion_prompt(input_text: str, mode: str) -> str:
    return _text_conversion_template(mode).format(input=input_text.strip())


def _stage_prompt_template(task: str, model: str, mode: str, override: str | None) -> str | None:
    if override:
        return override
    if task == "text_to_text":
        return _text_conversion_template(mode)
    if task == "audio_to_text" and model == "audio_flamingo_next":
        return "Generate a caption for the input audio."
    return None


def _stage_generation_settings(task: str, model: str, args: argparse.Namespace) -> dict:
    if task == "text_to_text":
        return {
            "max_new_tokens": min(max(args.steps, 32), 512),
            "do_sample": False,
            "temperature": None,
            "top_p": None,
            "seed": args.seed,
        }
    if task == "audio_to_text":
        return {"max_new_tokens": 256 if model == "audio_flamingo_next" else 128, "do_sample": False}
    if task == "speech_to_text":
        return {"max_new_tokens": 256 if model == "qwen3_asr" else None, "do_sample": False}
    settings = {"steps": args.steps, "duration_seconds": args.duration, "seed": args.seed}
    if task == "text_to_speech" and model == "f5_tts":
        settings["fix_duration_seconds"] = args.duration if args.duration > 0 else None
    return settings


def _model_dtype(model: str, device: str) -> str:
    if not device.startswith("cuda"):
        return "float32"
    if model in {"whisper", "f5_tts", "cosyvoice3"}:
        return "float16"
    return "bfloat16"


def _stage_record(stage: str, backend: str, prompt: str, output_path: Path) -> dict:
    return {"stage": stage, "backend": backend, "prompt": prompt, "output_file": str(output_path)}


def _text_stage_record(stage: str, backend: str, text: str) -> dict:
    return {"stage": stage, "backend": backend, "output_text": text}


if __name__ == "__main__":
    main()
