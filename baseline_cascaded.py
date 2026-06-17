#!/usr/bin/env python
"""Run modular baseline stages and cascades over a dataset split."""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import time
from collections import Counter
from pathlib import Path
from typing import Iterable

from baselines.dataset import PromptRow, count_jsonl_rows, iter_split_rows, load_dataset_config
from baselines.cascaded_schema_v1 import (
    TASK_TO_STAGE,
    legacy_sample_to_inspect_sample,
    legacy_sample_to_stage_manifest_row,
)
from baselines.models import get_runner
from baselines.models.base import BaselineRunner, GenerationResult
from baselines.cascaded_manifest import (
    build_sample,
    carried_stage_definitions,
    carried_stages,
    eastern_now,
    normalize_prompt,
    output_duration_seconds,
    read_inspection_definitions,
    unwrap_sample,
    wrap_sample,
    write_inspection_manifest,
)
from project_conventions import baseline_output_root, default_prompt_field
from project_paths import log_path, model_path, result_path, root_path, source_path
from results_structure import (
    cascaded_stage_run_paths,
    create_run_directories,
    generate_run_id,
    relative_to_project_root,
)


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
    parser.add_argument("--output-dir", default=str(result_path("baselines")))
    parser.add_argument("--log-dir", default=str(log_path("baselines")))
    parser.add_argument("--temporary", action="store_true", help="Write results and logs under configured temporary roots.")
    parser.add_argument("--run-id", default=None, help="Optional run ID for results-schema v1 runs; defaults to Toronto local YYYYMMDD_HHMM.")
    parser.add_argument("--run-name", default=None, help="Optional output variant name; defaults to selected stage backends.")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--duration", type=float, default=10.0)
    parser.add_argument("--steps", type=int, default=25)
    parser.add_argument("--seed", type=int, default=1337)
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--continue-on-error", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument(
        "--results-schema",
        choices=["legacy", "v1"],
        default="legacy",
        help="Output artifact schema. v1 writes design-schema stage artifacts for single-stage runs.",
    )
    parser.add_argument("--shard-index", type=int, default=None, help="Optional v1 stage shard index to write under shards/shard_NNN/.")
    parser.add_argument("--num-shards", type=int, default=None, help="Total v1 stage shard count for this run.")

    parser.add_argument("--asr", default="audio_omni", choices=["audio_omni", "reference", "whisper", "qwen3_asr"])
    parser.add_argument("--captioner", default="audio_omni", choices=["audio_omni", "reference", "audio_flamingo_next", "qwen3_omni_captioner"])
    parser.add_argument("--text-model", default="template", choices=["template", "qwen3_32b", "llama3_3_70b_instruct"])
    parser.add_argument(
        "--text-mode",
        default="speech_to_audio_description",
        choices=["speech_to_audio_description", "audio_description_to_speech"],
        help="Standalone T2T conversion mode; cascades set this implicitly.",
    )
    parser.add_argument("--tts", default="audio_omni", choices=["audio_omni", "qwen3_omni", "f5_tts", "cosyvoice3"])
    parser.add_argument("--t2a", default="audiogen", choices=["audiogen", "tangoflux", "audio_omni"])

    parser.add_argument("--model-path", default=None, help="Fallback model path override for single-backend runs.")
    parser.add_argument(
        "--model-cache-dir",
        default=None,
        help="Optional runtime model cache directory recorded in v1 config metadata.",
    )
    parser.add_argument("--asr-model-path", default=None)
    parser.add_argument("--captioner-model-path", default=None)
    parser.add_argument("--text-model-path", default=None)
    parser.add_argument("--tts-model-path", default=None)
    parser.add_argument("--t2a-model-path", default=None)
    parser.add_argument("--prompt-template", default=None)
    parser.add_argument("--qwen-speaker", default="Ethan")
    parser.add_argument("--qwen-template", default=None)
    parser.add_argument("--tangoflux-source-dir", default=str(model_path("tangoflux")))
    parser.add_argument("--audio-omni-config", default=str(model_path("audio_omni_config")))
    parser.add_argument("--audio-omni-ckpt", default=str(model_path("audio_omni_checkpoint")))
    parser.add_argument("--audio-omni-source-dir", default=str(source_path("audio_omni")))
    parser.add_argument("--f5-tts-source-dir", default=str(source_path("f5_tts")))
    parser.add_argument("--f5-tts-vocoder-path", default=str(model_path("f5_tts_vocoder")))
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config = load_dataset_config(Path(args.dataset_config))
    split_path = Path(args.split_file) if args.split_file else Path(config["splits"][args.split])
    prompt_field = _pipeline_prompt_field(args.pipeline)
    _validate_v1_shard_args(args)
    selected_total_rows = count_jsonl_rows(split_path, args.start_index, args.num_samples)
    row_start_index = args.start_index
    row_num_samples = args.num_samples
    if args.results_schema == "v1" and args.shard_index is not None:
        shard_offset, shard_count = _shard_window(selected_total_rows, args.shard_index, args.num_shards)
        row_start_index = args.start_index + shard_offset
        row_num_samples = shard_count
    rows = list(
        iter_split_rows(
            split_path,
            prompt_field=prompt_field,
            start_index=row_start_index,
            num_samples=row_num_samples,
        )
    )
    rows = _enrich_rows_with_audio_metadata(rows, args.split)
    total_rows = count_jsonl_rows(split_path, row_start_index, row_num_samples)
    run_config = _run_config(args, split_path, prompt_field)
    if args.results_schema == "v1" and args.shard_index is not None:
        run_config["input_start_index"] = args.start_index
        run_config["input_num_samples"] = selected_total_rows
        run_config["current_shard_index"] = args.shard_index
        run_config["shard_start_index"] = row_start_index
        run_config["shard_num_samples"] = total_rows
    if rows and isinstance(rows[0].record.get("_cascaded_pipeline"), str):
        run_config["cascade_pipeline"] = rows[0].record["_cascaded_pipeline"]
    if args.results_schema == "v1" and args.pipeline in {"A2S", "S2A"}:
        raise ValueError("results-schema v1 is only supported for individual stage runs; assemble pipelines separately")
    if args.results_schema == "v1" and args.overwrite:
        raise ValueError("results-schema v1 runs do not support --overwrite; use the same run ID to resume")

    if args.results_schema == "v1":
        paths = _v1_stage_paths(args, run_config, run_id=args.run_id or generate_run_id())
        output_root = paths.result
        log_root = paths.logs
    else:
        if args.temporary:
            args.output_dir = str(_temporary_path(Path(args.output_dir), root_path("results_root"), root_path("temporary_results_root")))
            args.log_dir = str(_temporary_path(Path(args.log_dir), root_path("logs_root"), root_path("temporary_logs_root")))
        output_root = _cascade_output_root(args)
        log_root = _cascade_output_root(args, root=Path(args.log_dir))

    cascade = ModularCascade(args, dataset_root=config.get("dataset_root"))
    if args.dry_run:
        _dry_run(args, rows, output_root, run_config, total_rows)
        return
    if args.results_schema == "v1":
        if not paths.result.exists() and not paths.logs.exists():
            create_run_directories(paths)
        else:
            paths.result.mkdir(parents=True, exist_ok=True)
            paths.logs.mkdir(parents=True, exist_ok=True)
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
        schema_v1 = getattr(self.args, "results_schema", "legacy") == "v1"
        if schema_v1 and self.args.pipeline in {"A2S", "S2A"}:
            raise ValueError("results-schema v1 is only supported for individual stage runs; assemble pipelines separately")
        if schema_v1 and self.args.overwrite:
            raise ValueError("results-schema v1 runs do not support --overwrite; use the same run ID to resume")
        if schema_v1:
            rows = list(rows)
            _validate_v1_stage_inputs(self.args, rows)
        is_v1_shard = schema_v1 and getattr(self.args, "shard_index", None) is not None
        artifact_root = output_root / "shards" / _v1_shard_name(self.args.shard_index) if is_v1_shard else output_root
        output_root.mkdir(parents=True, exist_ok=True)
        log_root.mkdir(parents=True, exist_ok=True)
        artifact_root.mkdir(parents=True, exist_ok=True)
        final_manifest = artifact_root / "manifest.jsonl"
        inspection_manifest = artifact_root / ("inspect.json" if schema_v1 else "manifest.inspect.json")
        run_config_path = output_root / "config.json" if is_v1_shard else artifact_root / ("config.json" if schema_v1 else "run_config.json")
        self._run_config = run_config
        config_document = _design_config_document(run_config, total_rows) if schema_v1 else run_config
        config_payload = json.dumps(config_document, indent=2, sort_keys=not schema_v1) + "\n"
        resume_event: dict | None = None
        existing_run_doc: dict | None = None
        if self.args.overwrite:
            _unlink_if_exists(final_manifest)
            _unlink_if_exists(inspection_manifest)
            _unlink_if_exists(run_config_path)
            if schema_v1:
                for artifact in (output_root / "run.json", output_root / "errors.json", output_root / "summary.json"):
                    _unlink_if_exists(artifact)
            for component_manifest in (output_root / "components").glob("*/manifest.jsonl"):
                _unlink_if_exists(component_manifest)
        elif schema_v1:
            existing_run_doc = _read_optional_json(output_root / "run.json") if is_v1_shard else _read_optional_json(artifact_root / "run.json")
            existing_samples = self._read_existing_v1_samples(final_manifest, rows, artifact_root)
            if is_v1_shard and not _has_v1_shard_artifacts(artifact_root):
                resume_event = None
            else:
                resume_event = _validate_v1_resume_state(
                    output_root=artifact_root,
                    config_path=run_config_path,
                    manifest_path=final_manifest,
                    config_document=config_document,
                    requested_keys=[(row.row_index, row.sample_id) for row in rows],
                    existing_samples=existing_samples,
                    existing_run_doc=existing_run_doc,
                )
            if resume_event is not None:
                self._validate_v1_output_files_match_manifest(artifact_root, rows, existing_samples)
        else:
            _validate_existing_run_config(run_config_path, final_manifest, run_config)
            existing_samples = _read_existing_samples(final_manifest)
        if schema_v1:
            _validate_v1_artifact_targets(
                artifact_root,
                allow_resume=resume_event is not None,
                include_root_artifacts=not is_v1_shard,
            )
        if schema_v1:
            if run_config_path.exists():
                existing_config = json.loads(run_config_path.read_text(encoding="utf-8"))
                if existing_config != config_document:
                    raise RuntimeError("Existing v1 output directory belongs to a different config.json")
            else:
                _write_text_create_or_verify(run_config_path, config_payload)
        else:
            run_config_path.write_text(config_payload, encoding="utf-8")
        code_state = _git_state() if schema_v1 else None
        if is_v1_shard:
            _write_v1_shard_root_run(
                output_root=output_root,
                log_root=log_root,
                run_config=run_config,
                code_state=code_state,
            )

        processed = 0
        skipped = 0
        failed = 0
        created_at = _existing_created_at(inspection_manifest) or eastern_now()
        started_at = _v1_existing_started_at(existing_run_doc) if schema_v1 else eastern_now()
        started_at = started_at or eastern_now()
        started = time.time()
        samples: dict[tuple[int, str], dict] = dict(existing_samples)
        existing_definitions = read_inspection_definitions(final_manifest) if existing_samples else []
        prior_definitions = existing_definitions[:-1] if existing_definitions else []
        prior_lineage: list[dict] = []
        print(f"output_root={output_root}")
        print(f"manifest={final_manifest}")
        print(f"inspection_manifest={inspection_manifest}")
        print(f"run_config={run_config_path}")
        print(f"total_rows={total_rows}")

        manifest_mode = "a"
        with final_manifest.open(manifest_mode, encoding="utf-8") as manifest:
            for ordinal, row in enumerate(rows, start=1):
                row_definitions = carried_stage_definitions(row.record) or [
                    _stage_definition_from_carried_stage(stage) for stage in carried_stages(row.record)
                ]
                if not prior_definitions:
                    prior_definitions = row_definitions
                elif row_definitions != prior_definitions:
                    raise ValueError(f"Row {row.row_index} carries different stage definitions")
                row_lineage = _carried_stage_lineage(row.record)
                if not prior_lineage:
                    prior_lineage = row_lineage
                elif row_lineage != prior_lineage:
                    raise ValueError(f"Row {row.row_index} carries different stage lineage")
                output_path = self._final_output_path(artifact_root, row)
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
                        sample = self._run_full_sample(row, artifact_root, output_path)
                    else:
                        legacy_record = self._run_row(row, artifact_root, output_path, run_config)
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
                output_row = (
                    legacy_sample_to_stage_manifest_row(sample)
                    if schema_v1
                    else wrap_sample(run_config, sample)
                )
                manifest.write(json.dumps(output_row, ensure_ascii=False) + "\n")
                manifest.flush()

        definitions = self._full_stage_definitions() if self.args.pipeline in {"A2S", "S2A"} else prior_definitions + [self._current_stage_definition(len(prior_definitions) + 1)]
        ordered_samples = [samples[key] for key in sorted(samples)]
        final_status = "completed" if all(sample.get("status") == "success" for sample in ordered_samples) else "completed_with_errors"
        completed_at = eastern_now()
        if schema_v1:
            _write_design_run_artifacts(
                output_root=artifact_root,
                log_root=log_root,
                run_config=run_config,
                samples=ordered_samples,
                definitions=definitions,
                load_times=self._load_times,
                code_state=code_state,
                started_at=started_at,
                completed_at=completed_at,
                processed=processed,
                skipped=skipped,
                upstream_lineage=prior_lineage,
                resume_event=resume_event,
                existing_run_doc=existing_run_doc,
                run_root=output_root if is_v1_shard else None,
                write_run_json=not is_v1_shard,
            )
        else:
            write_inspection_manifest(
                inspection_manifest,
                metadata=run_config,
                stage_definitions=definitions,
                samples=ordered_samples,
                created_at=created_at,
                started_at=started_at,
                completed_at=completed_at,
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
        if self.args.text_model != "template":
            raise ValueError(f"Text model backend is not wired yet: {self.args.text_model}")
        return (
            "I can hear it clearly. "
            f"{audio_description.strip()} "
            "We should respond to what is happening there."
        ).strip()

    def _speech_to_audio_description(self, transcript: str) -> str:
        if self.args.text_model != "template":
            raise ValueError(f"Text model backend is not wired yet: {self.args.text_model}")
        return f"Environmental audio matching this spoken situation: {transcript.strip()}"

    def _convert_text(self, input_text: str, mode: str, output_root: Path, row: PromptRow) -> str:
        self._last_stage_extra = {}
        if self.args.text_model == "template":
            rendered_prompt = _text_conversion_prompt(input_text, mode)
            if mode == "audio_description_to_speech":
                text = self._audio_description_to_speech(input_text)
            elif mode == "speech_to_audio_description":
                text = self._speech_to_audio_description(input_text)
            else:
                raise ValueError(f"Unsupported text conversion mode: {mode}")
            extra = {"input_text": input_text, "mode": mode, "generation_prompt": rendered_prompt}
            self._last_stage_extra = extra
            self._write_component_text(output_root, "text_to_text", row, text, extra)
            return text
        if self.args.text_model not in {"qwen3_32b", "llama3_3_70b_instruct"}:
            raise ValueError(f"Text model backend is not wired yet: {self.args.text_model}")
        prompt = _text_conversion_prompt(input_text, mode)
        text_output = self._component_output_path(output_root, "text_to_text", row, ".json")
        runner = self._runner(self.args.text_model, "text_to_text")
        runner.system_prompt = _text_system_prompt(mode)
        inference_started = time.time()
        result = runner.generate(_with_prompt(row, prompt), text_output)
        self._last_stage_extra = {
            **result.extra,
            "inference_seconds": round(time.time() - inference_started, 3),
        }
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
        prompt = None
        if task in {"audio_to_text", "text_to_text"}:
            prompt = _prompt_from_extra(extra)
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
        prompt = None
        if task in {"audio_to_text", "text_to_text"}:
            prompt = _prompt_from_extra(extra)
        stage = {
            "stage": stage_number,
            "task": task,
            "model": model,
            "input": self._stage_input(task, row),
            "prompt": prompt,
            "inference_seconds": extra.get("inference_seconds", inference_seconds),
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

    def _read_existing_v1_samples(
        self,
        manifest_path: Path,
        rows: list[PromptRow],
        output_root: Path,
    ) -> dict[tuple[int, str], dict]:
        if not manifest_path.is_file():
            return {}
        rows_by_key = {(row.row_index, row.sample_id): row for row in rows}
        samples: dict[tuple[int, str], dict] = {}
        seen_row_indices: set[int] = set()
        seen_sample_ids: set[str] = set()
        with manifest_path.open("r", encoding="utf-8") as handle:
            for line_number, line in enumerate(handle, start=1):
                if not line.strip():
                    continue
                record = json.loads(line)
                row_index = record.get("row_index")
                sample_id = record.get("sample_id")
                if not isinstance(row_index, int) or not isinstance(sample_id, str):
                    raise ValueError(f"Invalid existing v1 manifest row at {manifest_path}:{line_number}")
                if row_index in seen_row_indices:
                    raise ValueError(f"Duplicate existing v1 row_index {row_index} in {manifest_path}")
                if sample_id in seen_sample_ids:
                    raise ValueError(f"Duplicate existing v1 sample_id {sample_id!r} in {manifest_path}")
                seen_row_indices.add(row_index)
                seen_sample_ids.add(sample_id)
                key = (row_index, sample_id)
                source_row = rows_by_key.get(key)
                if source_row is None:
                    raise ValueError(f"Existing v1 manifest row {key} is not present in this run input")
                sample = self._sample_from_v1_manifest_row(record, source_row, output_root)
                samples[key] = sample
        return samples

    def _validate_v1_output_files_match_manifest(
        self,
        output_root: Path,
        rows: list[PromptRow],
        existing_samples: dict[tuple[int, str], dict],
    ) -> None:
        allowed_paths = {
            self._final_output_path(output_root, row).resolve()
            for row in rows
            if (row.row_index, row.sample_id) in existing_samples
        }
        for dirname in ("text", "outputs"):
            directory = output_root / dirname
            if not directory.is_dir():
                continue
            for path in directory.rglob("*"):
                if path.is_file() and path.resolve() not in allowed_paths:
                    raise RuntimeError(f"Orphan v1 output exists without a matching manifest row: {path}")

    def _sample_from_v1_manifest_row(self, record: dict, row: PromptRow, output_root: Path) -> dict:
        task, model = self._current_task_and_model()
        stage_number = len(carried_stages(row.record)) + 1
        stage = {
            "stage": stage_number,
            "task": task,
            "model": record.get("model") or model,
            "input": self._stage_input(task, row),
            "prompt": None,
            "inference_seconds": _v1_manifest_inference_seconds(record),
            "output": _legacy_output_from_v1_stage_row(record, output_root),
            "error": record.get("error"),
        }
        if task == "text_to_text":
            stage["mode"] = self.args.text_mode
        status = record.get("status") if isinstance(record.get("status"), str) else "error"
        return build_sample(
            row_index=row.row_index,
            sample_id=row.sample_id,
            record=row.record,
            stage=stage,
            status=status,
            error=record.get("error") if status != "success" and isinstance(record.get("error"), dict) else None,
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
                "text_to_text": "speech_text" if self.args.text_mode == "audio_description_to_speech" else "audio_description",
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
                "messages": [
                    {"role": "system", "content": _text_system_prompt(active_mode)},
                    {"role": "user", "content": rendered},
                ],
            }
        template = _stage_prompt_template(task, model, active_mode, self.args.prompt_template)
        if template:
            rendered = template.format(prompt=row.prompt, input=row.prompt)
            return {"messages": [{"role": "user", "content": rendered}]}
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
        is_rule_based = model in {"reference", "template"}
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
                "cache_dir": None if is_rule_based else getattr(self.args, "model_cache_dir", None),
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
        if model in {"qwen3_32b", "llama3_3_70b_instruct", "template"}:
            return self.args.text_model_path or self.args.model_path
        return self.args.model_path

    def _component_output_path(self, output_root: Path, task: str, row: PromptRow, extension: str) -> Path:
        path = output_root / "components" / task / "text" / f"{row.row_index:06d}_{row.sample_id}{extension}"
        path.parent.mkdir(parents=True, exist_ok=True)
        return path

    def _final_output_path(self, output_root: Path, row: PromptRow) -> Path:
        if self.args.pipeline in {"A2T", "S2T"}:
            path = output_root / "text" / f"{row.row_index:06d}_{row.sample_id}.json"
        elif getattr(self.args, "results_schema", "legacy") == "v1":
            path = output_root / "outputs" / f"{row.row_index:06d}_{row.sample_id}.wav"
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


def _validate_v1_shard_args(args: argparse.Namespace) -> None:
    shard_index = getattr(args, "shard_index", None)
    num_shards = getattr(args, "num_shards", None)
    if shard_index is None and num_shards is None:
        return
    if getattr(args, "results_schema", "legacy") != "v1":
        raise ValueError("--shard-index/--num-shards require --results-schema v1")
    if shard_index is None or num_shards is None:
        raise ValueError("--shard-index and --num-shards must be provided together")
    if not isinstance(shard_index, int) or shard_index < 0:
        raise ValueError("--shard-index must be a non-negative integer")
    if not isinstance(num_shards, int) or num_shards <= 0:
        raise ValueError("--num-shards must be a positive integer")
    if shard_index >= num_shards:
        raise ValueError("--shard-index must be smaller than --num-shards")


def _shard_window(total_rows: int, shard_index: int, num_shards: int) -> tuple[int, int]:
    if total_rows < 0:
        raise ValueError("total_rows must be non-negative")
    base = total_rows // num_shards
    extra = total_rows % num_shards
    if shard_index < extra:
        return shard_index * (base + 1), base + 1
    return extra * (base + 1) + (shard_index - extra) * base, base


def _v1_shard_name(shard_index: int) -> str:
    return f"shard_{shard_index:03d}"


def _stage_name_for_run_config(run_config: dict) -> str | None:
    return {
        "A2T": "a2t",
        "S2T": "s2t",
        "T2T": "t2t",
        "T2S": "t2s",
        "T2A": "t2a",
    }.get(run_config.get("pipeline"))


def _write_v1_shard_root_run(
    *,
    output_root: Path,
    log_root: Path,
    run_config: dict,
    code_state: dict | None,
) -> None:
    path = output_root / "run.json"
    if path.exists():
        document = json.loads(path.read_text(encoding="utf-8"))
        if document.get("status") in {"complete", "complete_with_errors"}:
            raise RuntimeError(f"v1 run is already complete: {output_root}")
        for key, expected in (
            ("run_id", output_root.name),
            ("stage", _stage_name_for_run_config(run_config)),
            ("direction", _direction_for_run_config(run_config)),
            ("dataset", run_config.get("dataset")),
            ("split", run_config.get("split")),
        ):
            if expected is not None and document.get(key) != expected:
                raise RuntimeError(f"Existing v1 shard root run.json has different {key}")
        return
    model = _design_model_config(run_config).get("name")
    started_at = eastern_now()
    document = {
        "schema_version": 1,
        "run_id": output_root.name,
        "temporary": _is_temporary_path(output_root),
        "stage": _stage_name_for_run_config(run_config),
        "direction": _direction_for_run_config(run_config),
        "status": "running",
        "dataset": run_config.get("dataset"),
        "split": run_config.get("split"),
        "model": model,
        "started_at": started_at,
        "completed_at": None,
        "code": code_state,
        "slurm": _slurm_state(),
        "paths": {
            "result": _path_reference(output_root),
            "logs": _path_reference(log_root),
        },
        "lineage": {
            "direction": _direction_for_run_config(run_config),
            "current_stage_index": None,
            "stages": [],
        },
        "events": [
            {
                "type": "started",
                "time": started_at,
                "status": "running",
            }
        ],
    }
    _write_json_create_or_verify(path, document)


def _v1_stage_paths(args: argparse.Namespace, run_config: dict, *, run_id: str):
    pipeline = run_config.get("pipeline")
    if pipeline == "A2T":
        return cascaded_stage_run_paths(
            stage="a2t",
            model=args.captioner,
            dataset=args.dataset,
            split=args.split,
            run_id=run_id,
            temporary=args.temporary,
        )
    if pipeline == "S2T":
        return cascaded_stage_run_paths(
            stage="s2t",
            model=args.asr,
            dataset=args.dataset,
            split=args.split,
            run_id=run_id,
            temporary=args.temporary,
        )
    if pipeline == "T2T":
        direction = "a2s" if args.text_mode == "audio_description_to_speech" else "s2a"
        upstream_stage = "a2t" if direction == "a2s" else "s2t"
        upstream_model = args.captioner if direction == "a2s" else args.asr
        return cascaded_stage_run_paths(
            stage="t2t",
            direction=direction,
            model=args.text_model,
            upstream_stage=upstream_stage,
            upstream_model=upstream_model,
            dataset=args.dataset,
            split=args.split,
            run_id=run_id,
            temporary=args.temporary,
        )
    if pipeline == "T2S":
        return cascaded_stage_run_paths(
            stage="t2s",
            direction="a2s",
            model=args.tts,
            upstream_stage="t2t",
            upstream_model=args.text_model,
            dataset=args.dataset,
            split=args.split,
            run_id=run_id,
            temporary=args.temporary,
        )
    if pipeline == "T2A":
        return cascaded_stage_run_paths(
            stage="t2a",
            direction="s2a",
            model=args.t2a,
            upstream_stage="t2t",
            upstream_model=args.text_model,
            dataset=args.dataset,
            split=args.split,
            run_id=run_id,
            temporary=args.temporary,
        )
    raise ValueError(f"Unsupported v1 stage pipeline: {pipeline}")


def _validate_v1_stage_inputs(args: argparse.Namespace, rows: list[PromptRow]) -> None:
    expected = {
        "T2T": 1,
        "T2S": 2,
        "T2A": 2,
    }.get(args.pipeline)
    if expected is None:
        return
    allowed_previous_tasks = _v1_allowed_previous_tasks(args)
    expected_previous_model = _v1_expected_previous_model(args)
    for row in rows:
        stages = carried_stages(row.record)
        if len(stages) != expected:
            raise ValueError(
                f"results-schema v1 {args.pipeline} requires rows with exactly {expected} carried "
                f"upstream stage(s); row {row.row_index} sample_id={row.sample_id!r} has {len(stages)}"
            )
        carried_sample_id = _carried_sample_id(row.record)
        if carried_sample_id != row.sample_id:
            raise ValueError(
                f"results-schema v1 {args.pipeline} requires carried sample_id to match row sample_id; "
                f"row {row.row_index} has row sample_id={row.sample_id!r} carried sample_id={carried_sample_id!r}"
            )
        previous_task = stages[-1].get("task") if stages else None
        if previous_task not in allowed_previous_tasks:
            expected_tasks = ", ".join(sorted(allowed_previous_tasks))
            raise ValueError(
                f"results-schema v1 {args.pipeline} requires previous carried task in "
                f"{{{expected_tasks}}}; row {row.row_index} sample_id={row.sample_id!r} has {previous_task!r}"
            )
        previous_model = stages[-1].get("model") if stages else None
        if expected_previous_model is not None and previous_model != expected_previous_model:
            raise ValueError(
                f"results-schema v1 {args.pipeline} requires previous carried model "
                f"{expected_previous_model!r}; row {row.row_index} sample_id={row.sample_id!r} has {previous_model!r}"
            )


def _carried_stage_lineage(record: dict) -> list[dict]:
    lineage = record.get("_cascaded_stage_lineage")
    if not isinstance(lineage, list):
        return []
    return [dict(item) for item in lineage if isinstance(item, dict)]


def _carried_sample_id(record: dict) -> str | None:
    sample = record.get("_cascaded_sample")
    value = sample.get("sample_id") if isinstance(sample, dict) else None
    return value if isinstance(value, str) and value else None


def _stage_definition_from_carried_stage(stage: dict) -> dict:
    return {
        "stage": stage.get("stage"),
        "task": stage.get("task"),
        "model": stage.get("model"),
        "checkpoint": None,
        "dtype": None,
        "prompt_template": None,
        "generation": {},
        "runtime": {},
        "mode": stage.get("mode"),
        "system_prompt": None,
    }


def _v1_allowed_previous_tasks(args: argparse.Namespace) -> set[str]:
    if args.pipeline == "T2T":
        if args.text_mode == "audio_description_to_speech":
            return {"audio_captioning", "audio_to_text"}
        return {"asr", "speech_to_text"}
    if args.pipeline in {"T2A", "T2S"}:
        return {"text_to_text"}
    return set()


def _v1_expected_previous_model(args: argparse.Namespace) -> str | None:
    if args.pipeline == "T2T":
        if args.text_mode == "audio_description_to_speech":
            return args.captioner
        return args.asr
    if args.pipeline in {"T2A", "T2S"}:
        return args.text_model
    return None


def _temporary_path(path: Path, canonical_root: Path, temporary_root: Path) -> Path:
    resolved_path = path.resolve()
    resolved_root = canonical_root.resolve()
    try:
        relative = resolved_path.relative_to(resolved_root)
    except ValueError as exc:
        raise ValueError(f"--temporary can only redirect paths under {canonical_root}: {path}") from exc
    return temporary_root / relative


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
        "start_index": getattr(args, "start_index", 0),
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
        "continue_on_error": args.continue_on_error,
        "prompt_template": args.prompt_template,
        "qwen_speaker": args.qwen_speaker,
        "qwen_template": args.qwen_template,
        "model_cache_dir": getattr(args, "model_cache_dir", None),
        "shard_index": None,
        "num_shards": getattr(args, "num_shards", None),
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


def _write_design_run_artifacts(
    *,
    output_root: Path,
    log_root: Path,
    run_config: dict,
    samples: list[dict],
    definitions: list[dict],
    load_times: dict[str, float],
    code_state: dict | None,
    started_at: str,
    completed_at: str,
    processed: int,
    skipped: int,
    upstream_lineage: list[dict] | None = None,
    resume_event: dict | None = None,
    existing_run_doc: dict | None = None,
    run_root: Path | None = None,
    write_run_json: bool = True,
) -> None:
    status = "complete" if all(sample.get("status") == "success" for sample in samples) else "complete_with_errors"
    current_stage_index = max((len(sample.get("stages", [])) for sample in samples), default=len(definitions))
    stage_name = _design_stage_name(definitions[-1].get("task") if definitions else run_config.get("pipeline"))
    lineage_root = run_root or output_root
    run_id = lineage_root.name
    inspect_samples = [legacy_sample_to_inspect_sample(sample) for sample in samples]
    summary = _design_summary_document(
        run_id=run_id,
        status=status,
        samples=samples,
        load_times=load_times,
        processed=processed,
        skipped=skipped,
    )
    lineage = _design_lineage(
        run_config=run_config,
        definitions=definitions,
        output_root=lineage_root,
        current_stage_index=current_stage_index,
        upstream_lineage=upstream_lineage or [],
    )
    events = _design_run_events(
        existing_run_doc=existing_run_doc,
        started_at=started_at,
        completed_at=completed_at,
        samples=samples,
        resume_event=resume_event,
    )
    artifacts = {
        "run.json": {
            "schema_version": 1,
            "run_id": run_id,
            "temporary": _is_temporary_path(lineage_root),
            "stage": stage_name,
            "direction": _direction_for_run_config(run_config),
            "status": status,
            "dataset": run_config.get("dataset"),
            "split": run_config.get("split"),
            "model": definitions[-1].get("model") if definitions else run_config.get("model"),
            "started_at": started_at,
            "completed_at": completed_at,
            "code": code_state,
            "slurm": _slurm_state(),
            "paths": {
                "result": _path_reference(lineage_root),
                "logs": _path_reference(log_root),
            },
            "lineage": lineage,
            "events": events,
        },
        "inspect.json": {
            "schema_version": 1,
            "run_id": run_id,
            "stage": stage_name,
            "direction": _direction_for_run_config(run_config),
            "dataset": run_config.get("dataset"),
            "split": run_config.get("split"),
            "status": status,
            "lineage": lineage,
            "stage_configs": [_stage_config_summary(index, definition) for index, definition in enumerate(definitions, start=1)],
            "summary": {
                "requested": len(samples),
                "successful": sum(sample.get("status") == "success" for sample in samples),
                "failed": sum(sample.get("status") != "success" for sample in samples),
            },
            "samples": inspect_samples,
        },
        "errors.json": _design_errors_document(run_id, samples),
        "summary.json": summary,
    }
    if not write_run_json:
        artifacts.pop("run.json", None)
    for name, document in artifacts.items():
        _write_json_replace(output_root / name, document)


def _design_config_document(run_config: dict, total_rows: int) -> dict:
    pipeline = run_config.get("pipeline")
    return {
        "schema_version": 1,
        "input": {
            "manifest": run_config.get("split_file"),
            "input_field": run_config.get("prompt_field"),
            "start_index": run_config.get("input_start_index", run_config.get("start_index")),
            "num_samples": run_config.get("input_num_samples", total_rows),
        },
        "model": _design_model_config(run_config),
        "prompt": _design_prompt_config(run_config),
        "generation": {
            "seed": run_config.get("seed"),
            "steps": run_config.get("steps"),
            "duration_seconds": run_config.get("duration"),
        },
        "execution": {
            "continue_on_error": run_config.get("continue_on_error"),
            "shard_index": run_config.get("shard_index"),
            "num_shards": run_config.get("num_shards"),
            "legacy_pipeline": pipeline,
            "cascade_pipeline": run_config.get("cascade_pipeline"),
        },
    }


def _design_model_config(run_config: dict) -> dict:
    pipeline = run_config.get("pipeline")
    model_key = {
        "A2T": "captioner",
        "S2T": "asr",
        "T2T": "text_model",
        "T2S": "tts",
        "T2A": "t2a",
    }.get(pipeline)
    model_name = run_config.get(model_key) if model_key else run_config.get("model")
    model_paths = run_config.get("model_paths") if isinstance(run_config.get("model_paths"), dict) else {}
    path_key = {"text_model": "text"}.get(model_key, model_key)
    is_rule_based = model_name in {"reference", "template"}
    cache_dir = None if is_rule_based else _named_root_reference(run_config.get("model_cache_dir"))
    return {
        "name": model_name,
        "path": _named_root_reference(model_paths.get(path_key or "fallback") or model_paths.get("fallback")),
        "cache_dir": cache_dir,
        "dtype": None,
        "device": run_config.get("device"),
    }


def _design_prompt_config(run_config: dict) -> dict | None:
    if run_config.get("pipeline") not in {"A2T", "S2T", "T2T"} and not run_config.get("prompt_template"):
        return None
    return {
        "mode": run_config.get("text_mode") if run_config.get("pipeline") == "T2T" else run_config.get("pipeline"),
        "template": run_config.get("prompt_template"),
        "system_prompt": None,
    }


def _design_summary_document(
    *,
    run_id: str,
    status: str,
    samples: list[dict],
    load_times: dict[str, float],
    processed: int,
    skipped: int,
) -> dict:
    timings = [
        stage.get("inference_seconds")
        for sample in samples
        for stage in sample.get("stages", [])
        if isinstance(stage, dict) and isinstance(stage.get("inference_seconds"), (int, float))
    ]
    failed = sum(sample.get("status") != "success" for sample in samples)
    return {
        "schema_version": 1,
        "run_id": run_id,
        "status": status,
        "samples": {
            "requested": len(samples),
            "processed": processed + failed,
            "successful": sum(sample.get("status") == "success" for sample in samples),
            "failed": failed,
            "skipped": skipped,
        },
        "timing": {
            "model_load_seconds": round(sum(load_times.values()), 6) if load_times else None,
            "total_inference_seconds": round(sum(timings), 6) if timings else None,
            "total_seconds": None,
            "mean_inference_seconds_per_sample": round(sum(timings) / len(timings), 6) if timings else None,
            "median_inference_seconds_per_sample": _median(timings),
            "min_inference_seconds_per_sample": min(timings) if timings else None,
            "max_inference_seconds_per_sample": max(timings) if timings else None,
        },
        "outputs": {
            "manifest": "manifest.jsonl",
            "inspect": "inspect.json",
            "generated_media_files": _generated_media_count(samples),
        },
        "errors": {
            "count": failed,
            "by_type": dict(Counter(_sample_error_type(sample) for sample in samples if sample.get("status") != "success")),
        },
    }


def _design_errors_document(run_id: str, samples: list[dict]) -> dict:
    errors = []
    for sample in samples:
        if sample.get("status") == "success":
            continue
        error = sample.get("error") if isinstance(sample.get("error"), dict) else {}
        stage = (sample.get("stages") or [{}])[-1]
        errors.append(
            {
                "sample_id": sample.get("sample_id"),
                "row_index": sample.get("row_index"),
                "stage": _design_stage_name(stage.get("task")) if isinstance(stage, dict) else None,
                "error_type": error.get("type"),
                "message": error.get("message"),
                "log_ref": None,
            }
        )
    return {
        "schema_version": 1,
        "run_id": run_id,
        "error_count": len(errors),
        "errors": errors,
    }


def _design_lineage(
    *,
    run_config: dict,
    definitions: list[dict],
    output_root: Path,
    current_stage_index: int,
    upstream_lineage: list[dict] | None = None,
) -> dict:
    upstream_entries = upstream_lineage or []
    return {
        "direction": _direction_for_run_config(run_config),
        "current_stage_index": current_stage_index,
        "stages": [
            _design_lineage_entry(
                index=index,
                definition=definition,
                output_root=output_root,
                upstream_entries=upstream_entries,
            )
            for index, definition in enumerate(definitions, start=1)
        ],
    }


def _design_lineage_entry(
    *,
    index: int,
    definition: dict,
    output_root: Path,
    upstream_entries: list[dict],
) -> dict:
    if index <= len(upstream_entries):
        entry = dict(upstream_entries[index - 1])
        entry.setdefault("stage_index", index)
        entry.setdefault("stage", _design_stage_name(definition.get("task")))
        entry.setdefault("model", definition.get("model"))
        return entry
    return {
        "stage_index": index,
        "stage": _design_stage_name(definition.get("task")),
        "model": definition.get("model"),
        "run_id": output_root.name,
        "run_path": _path_reference(output_root),
        "manifest": _path_reference(output_root / "manifest.jsonl"),
        "config": _path_reference(output_root / "config.json"),
    }


def _stage_config_summary(index: int, definition: dict) -> dict:
    model = {
        "name": definition.get("model"),
        "path": _named_root_reference(definition.get("checkpoint")),
        "cache_dir": _named_root_reference((definition.get("runtime") or {}).get("cache_dir")),
        "device": (definition.get("runtime") or {}).get("device"),
        "dtype": definition.get("dtype"),
    }
    summary = {
        "stage_index": index,
        "stage": _design_stage_name(definition.get("task")),
        "model": model,
        "input": {},
        "generation": definition.get("generation") or {},
    }
    if definition.get("prompt_template") is not None or definition.get("system_prompt") is not None:
        summary["prompt"] = {
            "mode": definition.get("mode"),
            "template_name": None,
            "template": definition.get("prompt_template"),
            "system_prompt": definition.get("system_prompt"),
        }
    return summary


def _design_stage_name(task: object) -> str | None:
    return TASK_TO_STAGE.get(task) if isinstance(task, str) else None


def _direction_for_run_config(run_config: dict) -> str | None:
    pipeline = run_config.get("pipeline")
    if pipeline == "T2T":
        if run_config.get("text_mode") == "audio_description_to_speech":
            return "a2s"
        if run_config.get("text_mode") == "speech_to_audio_description":
            return "s2a"
    return _direction_for_pipeline(pipeline)


def _direction_for_pipeline(pipeline: object) -> str | None:
    if pipeline in {"A2T", "T2S", "A2S"}:
        return "a2s"
    if pipeline in {"S2T", "T2A", "S2A"}:
        return "s2a"
    if pipeline == "T2T":
        return None
    return None


def _is_temporary_path(path: Path) -> bool:
    return "temporary" in path.parts


def _generated_media_count(samples: list[dict]) -> int | None:
    count = sum(
        1
        for sample in samples
        if isinstance(sample.get("final_output"), dict)
        and (sample["final_output"].get("file") or sample["final_output"].get("audio_file") or sample["final_output"].get("speech_file"))
    )
    return count or None


def _sample_error_type(sample: dict) -> str:
    error = sample.get("error")
    if isinstance(error, dict) and isinstance(error.get("type"), str):
        return error["type"]
    return "StageError"


def _median(values: list[float]) -> float | None:
    if not values:
        return None
    ordered = sorted(values)
    midpoint = len(ordered) // 2
    if len(ordered) % 2:
        return ordered[midpoint]
    return (ordered[midpoint - 1] + ordered[midpoint]) / 2


def _has_v1_shard_artifacts(artifact_root: Path) -> bool:
    return any((artifact_root / name).exists() for name in ("manifest.jsonl", "inspect.json", "errors.json", "summary.json"))


def _validate_v1_artifact_targets(
    output_root: Path,
    *,
    allow_resume: bool = False,
    include_root_artifacts: bool = True,
) -> None:
    names = ["manifest.jsonl", "inspect.json", "errors.json", "summary.json"]
    if include_root_artifacts:
        names.extend(["config.json", "run.json"])
    for name in names:
        path = output_root / name
        if path.exists() and not allow_resume:
            raise FileExistsError(f"v1 artifact already exists: {path}")


def _write_json_create_or_verify(path: Path, document: dict) -> None:
    payload = json.dumps(document, ensure_ascii=False, indent=2) + "\n"
    _write_text_create_or_verify(path, payload)


def _write_text_create_or_verify(path: Path, payload: str) -> None:
    try:
        _write_text_create_only(path, payload)
    except FileExistsError:
        if path.read_text(encoding="utf-8") != payload:
            raise RuntimeError(f"Existing artifact differs from requested payload: {path}")


def _write_json_create_only(path: Path, document: dict) -> None:
    _write_text_create_only(
        path,
        json.dumps(document, ensure_ascii=False, indent=2) + "\n",
    )


def _write_json_replace(path: Path, document: dict) -> None:
    path.write_text(json.dumps(document, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def _write_text_create_only(path: Path, payload: str) -> None:
    with path.open("x", encoding="utf-8") as handle:
        handle.write(payload)


def _path_reference(path: Path) -> str:
    return relative_to_project_root(path)


def _slurm_state() -> dict[str, str | None]:
    return {
        "job_id": os.environ.get("SLURM_JOB_ID"),
        "array_job_id": os.environ.get("SLURM_ARRAY_JOB_ID"),
        "array_task_id": os.environ.get("SLURM_ARRAY_TASK_ID"),
    }


def _named_root_reference(value: object) -> str | None:
    if value is None:
        return None
    if not isinstance(value, str):
        value = str(value)
    if not value:
        return None
    normalized = value.replace("\\", "/")
    if normalized.startswith("{"):
        return normalized
    path = Path(normalized)
    if not path.is_absolute():
        return path.as_posix()
    resolved = path.resolve()
    for root_name in ("model_root", "cache_root", "sources_root", "environment_root", "datasets_root", "repo_root"):
        try:
            root = root_path(root_name).resolve()
            relative = resolved.relative_to(root)
        except (KeyError, ValueError):
            continue
        token = "{" + root_name + "}"
        return token if relative.as_posix() == "." else f"{token}/{relative.as_posix()}"
    return relative_to_project_root(path)


def _git_state() -> dict[str, object]:
    try:
        commit = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            check=True,
            capture_output=True,
            text=True,
        ).stdout.strip()
        dirty = bool(
            subprocess.run(
                ["git", "status", "--porcelain"],
                check=True,
                capture_output=True,
                text=True,
            ).stdout.strip()
        )
    except (OSError, subprocess.CalledProcessError) as exc:
        return {"git_commit": None, "git_dirty": None, "error": str(exc)}
    return {"git_commit": commit, "git_dirty": dirty}


def _read_optional_json(path: Path) -> dict | None:
    if not path.is_file():
        return None
    return json.loads(path.read_text(encoding="utf-8"))


def _validate_v1_resume_state(
    *,
    output_root: Path,
    config_path: Path,
    manifest_path: Path,
    config_document: dict,
    requested_keys: list[tuple[int, str]],
    existing_samples: dict[tuple[int, str], dict],
    existing_run_doc: dict | None,
) -> dict | None:
    has_resume_artifact = (
        config_path.exists()
        or manifest_path.exists()
        or existing_run_doc is not None
        or (output_root / "inspect.json").exists()
        or (output_root / "errors.json").exists()
        or (output_root / "summary.json").exists()
    )
    if not has_resume_artifact:
        return None
    if not config_path.is_file():
        raise FileNotFoundError(f"v1 resume requires existing config.json: {config_path}")
    existing_config = json.loads(config_path.read_text(encoding="utf-8"))
    if existing_config != config_document:
        raise RuntimeError("Existing v1 output directory belongs to a different config.json")
    terminal_statuses = {"complete", "complete_with_errors"}
    if existing_run_doc and existing_run_doc.get("status") in terminal_statuses:
        raise RuntimeError(f"v1 run is already complete: {output_root}")
    if requested_keys and all(key in existing_samples for key in requested_keys):
        raise RuntimeError(f"v1 run is already complete: {output_root}")
    successful = sum(sample.get("status") == "success" for sample in existing_samples.values())
    errors = len(existing_samples) - successful
    missing_row_indices = [row_index for row_index, sample_id in requested_keys if (row_index, sample_id) not in existing_samples]
    start_row_index = missing_row_indices[0] if missing_row_indices else 0
    return {
        "type": "resumed",
        "time": eastern_now(),
        "start_row_index": start_row_index,
        "existing_successful_rows": successful,
        "existing_error_rows": errors,
        "reason": "manual_resume",
        "manifest": manifest_path.name,
    }


def _v1_existing_started_at(existing_run_doc: dict | None) -> str | None:
    if not isinstance(existing_run_doc, dict):
        return None
    value = existing_run_doc.get("started_at")
    return value if isinstance(value, str) else None


def _design_run_events(
    *,
    existing_run_doc: dict | None,
    started_at: str,
    completed_at: str,
    samples: list[dict],
    resume_event: dict | None,
) -> list[dict]:
    existing_events = existing_run_doc.get("events") if isinstance(existing_run_doc, dict) else None
    events = [dict(event) for event in existing_events] if isinstance(existing_events, list) else []
    if not events:
        events.append({"type": "started", "time": started_at, "start_row_index": 0})
    if resume_event is not None:
        events.append(resume_event)
    events.append(
        {
            "type": "completed",
            "time": completed_at,
            "successful_rows": sum(sample.get("status") == "success" for sample in samples),
            "error_rows": sum(sample.get("status") != "success" for sample in samples),
        }
    )
    return events


def _v1_manifest_inference_seconds(record: dict) -> float | None:
    timing = record.get("timing")
    if not isinstance(timing, dict):
        return None
    seconds = timing.get("inference_seconds")
    return seconds if isinstance(seconds, (int, float)) else None


def _legacy_output_from_v1_stage_row(record: dict, output_root: Path) -> dict | None:
    output = record.get("output")
    if not isinstance(output, dict):
        return None
    text = _v1_text_output(output)
    if text is not None:
        key, value = text
        output_type = {
            "audio_caption": "audio_caption",
            "speech_transcription": "speech_text",
            "generated_speech_text": "speech_text",
            "generated_audio_caption": "audio_description",
            "text": "text",
        }[key]
        return {"type": output_type, "text": value}
    for field, output_type in (("audio_file", "environmental_audio"), ("speech_file", "speech_audio"), ("file", "media")):
        value = output.get(field)
        if isinstance(value, str) and value:
            resolved = _resolve_v1_output_reference(output_root, value)
            if record.get("status") == "success" and not resolved.is_file():
                raise FileNotFoundError(f"successful v1 row points to missing media file: {resolved}")
            legacy = {"type": output_type, "file": value}
            for key in ("sample_rate", "duration_seconds"):
                if output.get(key) is not None:
                    legacy[key] = output[key]
            return legacy
    return dict(output)


def _v1_text_output(output: dict) -> tuple[str, str] | None:
    for key in ("audio_caption", "speech_transcription", "generated_speech_text", "generated_audio_caption", "text"):
        value = output.get(key)
        if isinstance(value, str):
            return key, value
    return None


def _resolve_v1_output_reference(output_root: Path, value: str) -> Path:
    path = Path(value)
    if path.is_absolute():
        return path
    return output_root / path


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


def _prompt_from_extra(extra: dict) -> dict | None:
    rendered = extra.get("generation_prompt") or extra.get("understanding_prompt")
    return normalize_prompt({"rendered": rendered, "messages": extra.get("messages")})

if __name__ == "__main__":
    main()
