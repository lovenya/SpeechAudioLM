from __future__ import annotations

import argparse
import contextlib
import json
import sys
import tempfile
import threading
import time
import types
import unittest
from pathlib import Path
from unittest.mock import patch

from project_paths import root_path

from baseline_cascaded import (
    ModularCascade,
    _cascade_output_root,
    _cascade_variant_name,
    _apply_tts_reference_bank,
    _design_config_document,
    _load_stage_rows,
    _load_tts_reference_bank,
    _pipeline_prompt_field,
    _planned_stages,
    _read_optional_json,
    _run_config,
    _temporary_path,
    _validate_v1_shard_args,
    _write_text_create_or_verify,
    main,
)
from baselines.models.audio_captioning import _patch_audio_flamingo_next_transformers_compat
from baselines.models.speech_asr import _patch_qwen_asr_transformers_compat
from baselines.dataset import PromptRow, iter_split_rows
from baselines.models.text_llm import TextLLMRunner, _strip_thinking_output


class CascadedBaselineTest(unittest.TestCase):
    def test_tts_reference_bank_uses_fixed_train_rows(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            ref_file = Path(tmpdir) / "train_refs.jsonl"
            rows = [
                {"speech_file": f"speech/train-{idx}.wav", "speech_text": f"train text {idx}"}
                for idx in range(6)
            ]
            ref_file.write_text("".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8")

            refs = _load_tts_reference_bank(ref_file, count=3, stride=2)
            self.assertEqual([ref["sample_id"] for ref in refs], ["train-0", "train-2", "train-4"])

            bank_file = Path(tmpdir) / "bank.json"
            bank_file.write_text(
                json.dumps(
                    {
                        "source_split_file": str(ref_file),
                        "source_split": "train",
                        "assignment": "row_index_mod_count",
                        "sample_ids": ["train-0", "train-2", "train-4"],
                    }
                ),
                encoding="utf-8",
            )
            args = argparse.Namespace(
                pipeline="T2S",
                tts_reference_bank_file=str(bank_file),
                tts_reference_split_file=None,
                tts_reference_sample_ids=None,
                tts_reference_count=0,
                tts_reference_stride=110,
            )
            target_rows = [
                PromptRow(
                    i,
                    f"test-{i}",
                    "generated speech",
                    {"speech_file": f"speech/test-{i}.wav", "speech_text": "test ground truth"},
                )
                for i in range(4)
            ]

            enriched = _apply_tts_reference_bank(target_rows, args)

            self.assertEqual([row.record["reference_sample_id"] for row in enriched], ["train-0", "train-2", "train-4", "train-0"])
            self.assertEqual(enriched[0].record["reference_speech_file"], "speech/train-0.wav")
            self.assertEqual(enriched[0].record["reference_speech_text"], "train text 0")
            self.assertEqual(enriched[0].record["speech_text"], "test ground truth")

            cascade = ModularCascade(argparse.Namespace(pipeline="T2S"), dataset_root=None)
            stage_input = cascade._stage_input("text_to_speech", enriched[0])
            self.assertEqual(stage_input["reference"]["speech_file"], "speech/train-0.wav")
            self.assertEqual(stage_input["reference"]["source_split"], "train")
            self.assertEqual(stage_input["reference"]["selection"]["assignment"], "row_index_mod_count")
            self.assertEqual(stage_input["reference"]["selection"]["bank_file"], str(bank_file))

            run_config = {
                "pipeline": "T2S",
                "split_file": "input.jsonl",
                "prompt_field": "generated_speech_text",
                "start_index": 0,
                "input_num_samples": 4,
                "tts": "cosyvoice3",
                "device": "cuda",
                "seed": 1337,
                "steps": 32,
                "duration": 0,
                "tts_reference_bank_file": str(bank_file),
            }
            config = _design_config_document(run_config, total_rows=4)
            self.assertEqual(config["generation"]["tts_reference_bank"]["bank_file"], str(bank_file))
            self.assertEqual(config["generation"]["tts_reference_bank"]["sample_ids"], ["train-0", "train-2", "train-4"])

    def test_stage_uses_generation_only_inference_seconds(self) -> None:
        cascade = ModularCascade(
            argparse.Namespace(
                pipeline="T2T",
                text_model="qwen3_32b",
                text_mode="speech_to_audio_description",
                captioner="audio_flamingo_next",
                asr="whisper",
                tts="f5_tts",
                t2a="audiogen",
            ),
            dataset_root=None,
        )
        cascade._last_stage_extra = {"inference_seconds": 1.25}
        row = PromptRow(
            row_index=0,
            sample_id="sample",
            prompt="spoken words",
            record={"speech_text": "spoken words"},
        )
        sample = cascade._sample_from_record(
            row,
            {"candidate_text": "engine sound", "model_extra": {}},
            inference_seconds=99.0,
        )

        self.assertEqual(sample["stages"][-1]["inference_seconds"], 1.25)

    def test_s2a_text_stage_output_is_audio_description(self) -> None:
        cascade = ModularCascade(
            argparse.Namespace(text_mode="speech_to_audio_description"),
            dataset_root=None,
        )
        output = cascade._stage_output(
            "text_to_text",
            {"candidate_text": "A motorcycle engine starts."},
            {},
        )

        self.assertEqual(output["type"], "audio_description")

    def test_stage_input_preserves_carried_sample_identity(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            split = Path(tmpdir) / "stage_input.jsonl"
            split.write_text(
                '{"speech_text":"generated caption","audio_file":"audio/a.wav",'
                '"_cascaded_sample":{"row_index":17,"sample_id":"original-id"}}\n',
                encoding="utf-8",
            )
            rows = list(iter_split_rows(split, prompt_field="speech_text"))

        self.assertEqual(rows[0].row_index, 17)
        self.assertEqual(rows[0].sample_id, "original-id")

    def test_prompt_fields_match_pipeline_inputs(self) -> None:
        self.assertEqual(_pipeline_prompt_field("T2A"), "audio_cap")
        self.assertEqual(_pipeline_prompt_field("A2T"), "audio_cap")
        self.assertEqual(_pipeline_prompt_field("A2S"), "audio_cap")
        self.assertEqual(_pipeline_prompt_field("T2S"), "speech_text")
        self.assertEqual(_pipeline_prompt_field("S2T"), "speech_text")
        self.assertEqual(_pipeline_prompt_field("S2A"), "speech_text")

    def test_cascade_output_root_includes_backend_variant(self) -> None:
        args = argparse.Namespace(
            output_dir="results/baselines",
            pipeline="S2A",
            dataset="ttas",
            split="test",
            run_name=None,
            asr="reference",
            text_model="template",
            t2a="audiogen",
        )
        self.assertEqual(
            _cascade_output_root(args),
            Path("results/baselines/cascaded/S2A/asr-reference__text_to_text-template__text_to_audio-audiogen/ttas/test"),
        )

    def test_temporary_path_redirects_canonical_root(self) -> None:
        self.assertEqual(
            _temporary_path(
                Path("/repo/results/baselines"),
                Path("/repo/results"),
                Path("/repo/temporary/results"),
            ),
            Path("/repo/temporary/results/baselines"),
        )

    def test_temporary_path_rejects_noncanonical_root(self) -> None:
        with self.assertRaisesRegex(ValueError, "can only redirect paths under"):
            _temporary_path(
                Path("/other/results/baselines"),
                Path("/repo/results"),
                Path("/repo/temporary/results"),
            )

    def test_v1_stage_run_writes_design_artifacts(self) -> None:
        args = _minimal_cascade_args(
            pipeline="A2T",
            captioner="reference",
            results_schema="v1",
            continue_on_error=False,
            start_index=5,
            num_samples=1,
        )
        row = PromptRow(
            row_index=0,
            sample_id="sample-a",
            prompt="reference caption",
            record={
                "audio_file": "audio/source.wav",
                "audio_cap": "A motor starts.",
                "speech_file": "speech/source.wav",
                "speech_text": "Try the starter again.",
            },
        )
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            output_root = root / "results" / "cascaded" / "stages" / "a2t" / "reference" / "ttas" / "val" / "20260616_1437"
            log_root = root / "logs" / "cascaded" / "stages" / "a2t" / "reference" / "ttas" / "val" / "20260616_1437"
            run_config = _run_config(args, Path("data/splits/ttas_val.jsonl"), "audio_cap")

            with patch.dict("os.environ", {"SLURM_JOB_ID": "1234", "SLURM_ARRAY_JOB_ID": "1200", "SLURM_ARRAY_TASK_ID": "7"}):
                ModularCascade(args, dataset_root=None).run([row], output_root, log_root, run_config, total_rows=1)

            manifest_rows = _read_jsonl(output_root / "manifest.jsonl")
            run_doc = json.loads((output_root / "run.json").read_text(encoding="utf-8"))
            config = json.loads((output_root / "config.json").read_text(encoding="utf-8"))
            inspect = json.loads((output_root / "inspect.json").read_text(encoding="utf-8"))
            errors = json.loads((output_root / "errors.json").read_text(encoding="utf-8"))
            summary = json.loads((output_root / "summary.json").read_text(encoding="utf-8"))
            legacy_files = [
                (output_root / "manifest.inspect.json").exists(),
                (output_root / "run_config.json").exists(),
                (output_root / "model_load_times.json").exists(),
            ]

        self.assertEqual(
            manifest_rows,
            [
                {
                    "schema_version": 1,
                    "row_index": 0,
                    "sample_id": "sample-a",
                    "stage": "a2t",
                    "model": "reference",
                    "status": "success",
                    "input": {"audio_file": "audio/source.wav"},
                    "output": {"type": "text", "audio_caption": "A motor starts."},
                    "timing": {"inference_seconds": manifest_rows[0]["timing"]["inference_seconds"]},
                    "error": None,
                }
            ],
        )
        self.assertEqual(run_doc["run_id"], "20260616_1437")
        self.assertFalse(run_doc["temporary"])
        self.assertEqual(run_doc["stage"], "a2t")
        self.assertIn("git_commit", run_doc["code"])
        self.assertIn("git_dirty", run_doc["code"])
        self.assertEqual(run_doc["slurm"], {"job_id": "1234", "array_job_id": "1200", "array_task_id": "7"})
        self.assertEqual(run_doc["paths"], {"result": output_root.as_posix(), "logs": log_root.as_posix()})
        self.assertEqual(config["input"]["manifest"], "data/splits/ttas_val.jsonl")
        self.assertEqual(config["input"]["input_field"], "audio_cap")
        self.assertEqual(config["input"]["start_index"], 5)
        self.assertEqual(config["input"]["num_samples"], 1)
        self.assertEqual(inspect["samples"][0]["stages"][0]["output"], {"type": "text", "audio_caption": "A motor starts."})
        self.assertEqual(errors, {"schema_version": 1, "run_id": "20260616_1437", "error_count": 0, "errors": []})
        self.assertEqual(summary["samples"]["successful"], 1)
        self.assertEqual(summary["outputs"]["manifest"], "manifest.jsonl")
        self.assertEqual(legacy_files, [False, False, False])

    def test_v1_schema_rejects_full_cascade_mode(self) -> None:
        args = _minimal_cascade_args(pipeline="A2S", results_schema="v1")
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            output_root = root / "results" / "20260616_1437"
            with self.assertRaisesRegex(ValueError, "individual stage runs"):
                # The explicit guard lives in main(), but run() is kept from
                # writing deprecated full-cascade v1 artifacts as well.
                ModularCascade(args, dataset_root=None).run(
                    [],
                    output_root,
                    root / "logs" / "20260616_1437",
                    {"pipeline": "A2S", "lane": "lane", "dataset": "ttas", "split": "val"},
                    total_rows=0,
                )
            self.assertFalse(output_root.exists())

    def test_v1_schema_fails_before_work_when_artifact_exists(self) -> None:
        args = _minimal_cascade_args(pipeline="A2T", captioner="reference", results_schema="v1")
        row = PromptRow(
            row_index=0,
            sample_id="sample-a",
            prompt="reference caption",
            record={"audio_file": "audio/source.wav", "audio_cap": "A motor starts."},
        )
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            output_root = root / "results" / "20260616_1437"
            output_root.mkdir(parents=True)
            (output_root / "run.json").write_text("{}\n", encoding="utf-8")

            with self.assertRaisesRegex(FileNotFoundError, "requires existing config"):
                ModularCascade(args, dataset_root=None).run(
                    [row],
                    output_root,
                    root / "logs" / "20260616_1437",
                    _run_config(args, Path("data/splits/ttas_val.jsonl"), "audio_cap"),
                    total_rows=1,
                )

            self.assertFalse((output_root / "manifest.jsonl").exists())

    def test_v1_schema_rejects_overwrite_without_deleting_existing_artifacts(self) -> None:
        args = _minimal_cascade_args(pipeline="A2T", captioner="reference", results_schema="v1", overwrite=True)
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            output_root = root / "results" / "20260616_1437"
            output_root.mkdir(parents=True)
            manifest = output_root / "manifest.jsonl"
            run_json = output_root / "run.json"
            manifest.write_text("existing manifest\n", encoding="utf-8")
            run_json.write_text("existing run\n", encoding="utf-8")

            with self.assertRaisesRegex(ValueError, "do not support --overwrite"):
                ModularCascade(args, dataset_root=None).run(
                    [],
                    output_root,
                    root / "logs" / "20260616_1437",
                    _run_config(args, Path("data/splits/ttas_val.jsonl"), "audio_cap"),
                    total_rows=0,
                )

            self.assertEqual(manifest.read_text(encoding="utf-8"), "existing manifest\n")
            self.assertEqual(run_json.read_text(encoding="utf-8"), "existing run\n")

    def test_v1_schema_resumes_partial_manifest_without_inspect(self) -> None:
        args = _minimal_cascade_args(pipeline="A2T", captioner="reference", results_schema="v1")
        rows = [
            PromptRow(
                row_index=0,
                sample_id="sample-a",
                prompt="first caption",
                record={"audio_file": "audio/a.wav", "audio_cap": "first caption"},
            ),
            PromptRow(
                row_index=1,
                sample_id="sample-b",
                prompt="second caption",
                record={"audio_file": "audio/b.wav", "audio_cap": "second caption"},
            ),
        ]
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            output_root = root / "results" / "20260616_1437"
            log_root = root / "logs" / "20260616_1437"
            run_config = _run_config(args, Path("data/splits/ttas_val.jsonl"), "audio_cap")
            cascade = ModularCascade(args, dataset_root=None)
            cascade.run(rows[:1], output_root, log_root, run_config, total_rows=2)
            for name in ("run.json", "inspect.json", "errors.json", "summary.json"):
                (output_root / name).unlink()
            self.assertFalse((output_root / "inspect.json").exists())

            ModularCascade(args, dataset_root=None).run(rows, output_root, log_root, run_config, total_rows=2)
            manifest = _read_jsonl(output_root / "manifest.jsonl")
            inspect = json.loads((output_root / "inspect.json").read_text(encoding="utf-8"))
            summary = json.loads((output_root / "summary.json").read_text(encoding="utf-8"))
            run_doc = json.loads((output_root / "run.json").read_text(encoding="utf-8"))

        self.assertEqual([row["sample_id"] for row in manifest], ["sample-a", "sample-b"])
        self.assertEqual([sample["sample_id"] for sample in inspect["samples"]], ["sample-a", "sample-b"])
        self.assertEqual(summary["samples"]["successful"], 2)
        self.assertEqual(summary["samples"]["processed"], 1)
        self.assertEqual(summary["samples"]["skipped"], 1)
        self.assertEqual([event["type"] for event in run_doc["events"]], ["started", "resumed", "completed"])
        self.assertEqual(run_doc["events"][1]["start_row_index"], 1)
        self.assertEqual(run_doc["events"][1]["existing_successful_rows"], 1)

    def test_v1_config_records_non_rule_model_cache_dir_with_named_root(self) -> None:
        args = _minimal_cascade_args(
            pipeline="A2T",
            captioner="audio_flamingo_next",
            model_cache_dir=(root_path("cache_root") / "huggingface").as_posix(),
            results_schema="v1",
        )
        run_config = _run_config(args, Path("data/splits/ttas_val.jsonl"), "audio_cap")
        config = _design_config_document(run_config, total_rows=1)

        self.assertEqual(config["model"]["cache_dir"], "{cache_root}/huggingface")

    def test_v1_stage_shards_write_only_shard_local_artifacts(self) -> None:
        args0 = _minimal_cascade_args(
            pipeline="A2T",
            captioner="reference",
            results_schema="v1",
            shard_index=0,
            num_shards=2,
            num_samples=2,
        )
        args1 = _minimal_cascade_args(
            pipeline="A2T",
            captioner="reference",
            results_schema="v1",
            shard_index=1,
            num_shards=2,
            num_samples=2,
        )
        rows = [
            PromptRow(
                row_index=0,
                sample_id="sample-a",
                prompt="first caption",
                record={"audio_file": "audio/a.wav", "audio_cap": "first caption"},
            ),
            PromptRow(
                row_index=1,
                sample_id="sample-b",
                prompt="second caption",
                record={"audio_file": "audio/b.wav", "audio_cap": "second caption"},
            ),
        ]
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            output_root = root / "results" / "20260616_1437"
            log_root = root / "logs" / "20260616_1437"
            run_config0 = _run_config(args0, Path("data/splits/ttas_val.jsonl"), "audio_cap")
            run_config0["input_start_index"] = 0
            run_config0["input_num_samples"] = 2
            run_config1 = _run_config(args1, Path("data/splits/ttas_val.jsonl"), "audio_cap")
            run_config1["input_start_index"] = 0
            run_config1["input_num_samples"] = 2

            ModularCascade(args0, dataset_root=None).run([rows[0]], output_root, log_root, run_config0, total_rows=1)
            ModularCascade(args1, dataset_root=None).run([rows[1]], output_root, log_root, run_config1, total_rows=1)

            root_run = json.loads((output_root / "run.json").read_text(encoding="utf-8"))
            root_config = json.loads((output_root / "config.json").read_text(encoding="utf-8"))
            shard0 = output_root / "shards" / "shard_000"
            shard1 = output_root / "shards" / "shard_001"
            shard0_manifest = _read_jsonl(shard0 / "manifest.jsonl")
            shard1_manifest = _read_jsonl(shard1 / "manifest.jsonl")
            shard0_inspect = json.loads((shard0 / "inspect.json").read_text(encoding="utf-8"))

            shard0_config = json.loads((shard0 / "config.json").read_text(encoding="utf-8"))
            shard1_config = json.loads((shard1 / "config.json").read_text(encoding="utf-8"))

            self.assertEqual(root_run["run_id"], "20260616_1437")
            self.assertEqual(root_run["status"], "running")
            self.assertEqual(root_config["input"]["num_samples"], 2)
            self.assertEqual(root_config["execution"]["shard_index"], None)
            self.assertEqual(root_config["execution"]["num_shards"], 2)
            self.assertEqual(shard0_config["input"]["start_index"], 0)
            self.assertEqual(shard0_config["input"]["num_samples"], 1)
            self.assertEqual(shard0_config["execution"]["shard_index"], 0)
            self.assertEqual(shard0_config["execution"]["shard_start_index"], 0)
            self.assertEqual(shard1_config["input"]["start_index"], 1)
            self.assertEqual(shard1_config["input"]["num_samples"], 1)
            self.assertEqual(shard1_config["execution"]["shard_index"], 1)
            self.assertEqual(shard1_config["execution"]["shard_start_index"], 1)
            for name in ("manifest.jsonl", "inspect.json", "errors.json", "summary.json"):
                self.assertFalse((output_root / name).exists())
            self.assertEqual([row["row_index"] for row in shard0_manifest], [0])
            self.assertEqual([row["row_index"] for row in shard1_manifest], [1])
            self.assertEqual(shard0_inspect["run_id"], "20260616_1437")
            self.assertEqual(shard0_inspect["stage"], root_run["stage"])
            self.assertEqual(shard0_inspect["direction"], root_run["direction"])

    def test_create_or_verify_allows_matching_concurrent_artifact(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "config.json"
            _write_text_create_or_verify(path, "{\n  \"schema_version\": 1\n}\n")
            _write_text_create_or_verify(path, "{\n  \"schema_version\": 1\n}\n")

            with self.assertRaisesRegex(RuntimeError, "differs"):
                _write_text_create_or_verify(path, "{\n  \"schema_version\": 2\n}\n")

    def test_optional_json_tolerates_concurrent_publish(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "run.json"
            path.write_text("", encoding="utf-8")

            def publish() -> None:
                time.sleep(0.1)
                path.write_text('{"status": "running"}\n', encoding="utf-8")

            writer = threading.Thread(target=publish)
            writer.start()
            try:
                self.assertEqual(_read_optional_json(path), {"status": "running"})
            finally:
                writer.join()

    def test_v1_stage_shard_args_are_v1_only_and_paired(self) -> None:
        with self.assertRaisesRegex(ValueError, "require --results-schema v1"):
            _validate_v1_shard_args(_minimal_cascade_args(shard_index=0, num_shards=2))
        with self.assertRaisesRegex(ValueError, "must be provided together"):
            _validate_v1_shard_args(_minimal_cascade_args(results_schema="v1", shard_index=0))
        with self.assertRaisesRegex(ValueError, "non-negative"):
            _validate_v1_shard_args(_minimal_cascade_args(results_schema="v1", shard_index=-1, num_shards=2))
        with self.assertRaisesRegex(ValueError, "smaller"):
            _validate_v1_shard_args(_minimal_cascade_args(results_schema="v1", shard_index=2, num_shards=2))

    def test_v1_schema_resumes_config_only_interrupted_run(self) -> None:
        args = _minimal_cascade_args(pipeline="A2T", captioner="reference", results_schema="v1")
        row = PromptRow(
            row_index=0,
            sample_id="sample-a",
            prompt="first caption",
            record={"audio_file": "audio/a.wav", "audio_cap": "first caption"},
        )
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            output_root = root / "results" / "20260616_1437"
            log_root = root / "logs" / "20260616_1437"
            output_root.mkdir(parents=True)
            run_config = _run_config(args, Path("data/splits/ttas_val.jsonl"), "audio_cap")
            (output_root / "config.json").write_text(
                json.dumps(_design_config_document(run_config, total_rows=1), indent=2) + "\n",
                encoding="utf-8",
            )

            ModularCascade(args, dataset_root=None).run([row], output_root, log_root, run_config, total_rows=1)
            manifest = _read_jsonl(output_root / "manifest.jsonl")
            run_doc = json.loads((output_root / "run.json").read_text(encoding="utf-8"))

        self.assertEqual([entry["sample_id"] for entry in manifest], ["sample-a"])
        self.assertEqual([event["type"] for event in run_doc["events"]], ["started", "resumed", "completed"])
        self.assertEqual(run_doc["events"][1]["start_row_index"], 0)
        self.assertEqual(run_doc["events"][1]["existing_successful_rows"], 0)

    def test_v1_schema_resume_rejects_config_mismatch(self) -> None:
        args = _minimal_cascade_args(pipeline="A2T", captioner="reference", results_schema="v1")
        rows = [
            PromptRow(
                row_index=0,
                sample_id="sample-a",
                prompt="first caption",
                record={"audio_file": "audio/a.wav", "audio_cap": "first caption"},
            ),
            PromptRow(
                row_index=1,
                sample_id="sample-b",
                prompt="second caption",
                record={"audio_file": "audio/b.wav", "audio_cap": "second caption"},
            ),
        ]
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            output_root = root / "results" / "20260616_1437"
            log_root = root / "logs" / "20260616_1437"
            run_config = _run_config(args, Path("data/splits/ttas_val.jsonl"), "audio_cap")
            ModularCascade(args, dataset_root=None).run(rows[:1], output_root, log_root, run_config, total_rows=2)
            for name in ("run.json", "inspect.json", "errors.json", "summary.json"):
                (output_root / name).unlink()
            mismatched = dict(run_config)
            mismatched["seed"] = 999

            with self.assertRaisesRegex(RuntimeError, "different config"):
                ModularCascade(args, dataset_root=None).run(rows, output_root, log_root, mismatched, total_rows=2)

    def test_v1_schema_resume_rejects_model_cache_dir_mismatch(self) -> None:
        args = _minimal_cascade_args(
            pipeline="A2T",
            captioner="audio_flamingo_next",
            model_cache_dir=(root_path("cache_root") / "huggingface-a").as_posix(),
            results_schema="v1",
        )
        row = PromptRow(
            row_index=0,
            sample_id="sample-a",
            prompt="first caption",
            record={"audio_file": "audio/a.wav", "audio_cap": "first caption"},
        )
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            output_root = root / "results" / "20260616_1437"
            log_root = root / "logs" / "20260616_1437"
            output_root.mkdir(parents=True)
            run_config = _run_config(args, Path("data/splits/ttas_val.jsonl"), "audio_cap")
            (output_root / "config.json").write_text(
                json.dumps(_design_config_document(run_config, total_rows=1), indent=2) + "\n",
                encoding="utf-8",
            )
            mismatched = dict(run_config)
            mismatched["model_cache_dir"] = (root_path("cache_root") / "huggingface-b").as_posix()

            with self.assertRaisesRegex(RuntimeError, "different config"):
                ModularCascade(args, dataset_root=None).run([row], output_root, log_root, mismatched, total_rows=1)

    def test_v1_schema_resume_rejects_start_index_mismatch(self) -> None:
        args = _minimal_cascade_args(pipeline="A2T", captioner="reference", results_schema="v1")
        row = PromptRow(
            row_index=0,
            sample_id="sample-a",
            prompt="first caption",
            record={"audio_file": "audio/a.wav", "audio_cap": "first caption"},
        )
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            output_root = root / "results" / "20260616_1437"
            log_root = root / "logs" / "20260616_1437"
            run_config = _run_config(args, Path("data/splits/ttas_val.jsonl"), "audio_cap")
            ModularCascade(args, dataset_root=None).run([row], output_root, log_root, run_config, total_rows=1)
            for name in ("run.json", "inspect.json", "errors.json", "summary.json"):
                (output_root / name).unlink()
            mismatched = dict(run_config)
            mismatched["start_index"] = 1

            with self.assertRaisesRegex(RuntimeError, "different config"):
                ModularCascade(args, dataset_root=None).run([row], output_root, log_root, mismatched, total_rows=1)

    def test_v1_schema_resume_rejects_selected_count_mismatch(self) -> None:
        args = _minimal_cascade_args(pipeline="A2T", captioner="reference", results_schema="v1")
        row = PromptRow(
            row_index=0,
            sample_id="sample-a",
            prompt="first caption",
            record={"audio_file": "audio/a.wav", "audio_cap": "first caption"},
        )
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            output_root = root / "results" / "20260616_1437"
            log_root = root / "logs" / "20260616_1437"
            run_config = _run_config(args, Path("data/splits/ttas_val.jsonl"), "audio_cap")
            ModularCascade(args, dataset_root=None).run([row], output_root, log_root, run_config, total_rows=1)
            for name in ("run.json", "inspect.json", "errors.json", "summary.json"):
                (output_root / name).unlink()

            with self.assertRaisesRegex(RuntimeError, "different config"):
                ModularCascade(args, dataset_root=None).run([row], output_root, log_root, run_config, total_rows=2)

    def test_v1_schema_resume_rejects_missing_successful_media_output(self) -> None:
        args = _minimal_cascade_args(pipeline="T2A", t2a="audiogen", results_schema="v1")

        class FakeRunner:
            model_path = None

            def generate(self, row, output_path):
                output_path.write_bytes(b"RIFFfake")
                from baselines.models.base import GenerationResult

                return GenerationResult(output_path=output_path, sample_rate=32000, extra={"shape": [320000]})

        row = PromptRow(
            row_index=0,
            sample_id="sample-a",
            prompt="engine turning",
            record={
                "audio_cap": "engine turning",
                "_cascaded_sample": _carried_sample(
                    row_index=0,
                    sample_id="sample-a",
                    stages=[
                        {
                            "stage": 1,
                            "task": "asr",
                            "model": "reference",
                            "input": {"type": "speech_audio", "file": "speech/source.wav"},
                            "output": {"type": "speech_text", "text": "turn the engine"},
                        },
                        {
                            "stage": 2,
                            "task": "text_to_text",
                            "model": "template",
                            "mode": "speech_to_audio_description",
                            "input": {"type": "speech_text", "text": "turn the engine"},
                            "output": {"type": "audio_description", "text": "engine turning"},
                        },
                    ],
                ),
            },
        )
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            output_root = root / "results" / "cascaded" / "stages" / "t2a" / "audiogen" / "from-t2t" / "template" / "ttas" / "val" / "20260616_1437"
            log_root = root / "logs" / "cascaded" / "stages" / "t2a" / "audiogen" / "from-t2t" / "template" / "ttas" / "val" / "20260616_1437"
            run_config = _run_config(args, Path("data/splits/ttas_val.jsonl"), "audio_cap")
            cascade = ModularCascade(args, dataset_root=None)
            cascade._runners[("audiogen", "T2A")] = FakeRunner()
            cascade.run([row], output_root, log_root, run_config, total_rows=1)
            manifest = _read_jsonl(output_root / "manifest.jsonl")
            (output_root / manifest[0]["output"]["audio_file"]).unlink()
            for name in ("run.json", "inspect.json", "errors.json", "summary.json"):
                (output_root / name).unlink()

            with self.assertRaisesRegex(FileNotFoundError, "missing media file"):
                ModularCascade(args, dataset_root=None).run([row], output_root, log_root, run_config, total_rows=1)

    def test_v1_schema_resume_rejects_orphan_output_file(self) -> None:
        args = _minimal_cascade_args(pipeline="A2T", captioner="reference", results_schema="v1")
        rows = [
            PromptRow(
                row_index=0,
                sample_id="sample-a",
                prompt="first caption",
                record={"audio_file": "audio/a.wav", "audio_cap": "first caption"},
            ),
            PromptRow(
                row_index=1,
                sample_id="sample-b",
                prompt="second caption",
                record={"audio_file": "audio/b.wav", "audio_cap": "second caption"},
            ),
        ]
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            output_root = root / "results" / "20260616_1437"
            log_root = root / "logs" / "20260616_1437"
            run_config = _run_config(args, Path("data/splits/ttas_val.jsonl"), "audio_cap")
            ModularCascade(args, dataset_root=None).run(rows[:1], output_root, log_root, run_config, total_rows=2)
            for name in ("run.json", "inspect.json", "errors.json", "summary.json"):
                (output_root / name).unlink()
            orphan = output_root / "text" / "000001_sample-b.json"
            orphan.write_text("{}\n", encoding="utf-8")

            with self.assertRaisesRegex(RuntimeError, "Orphan v1 output"):
                ModularCascade(args, dataset_root=None).run(rows, output_root, log_root, run_config, total_rows=2)

    def test_v1_schema_rejects_rerun_after_complete(self) -> None:
        args = _minimal_cascade_args(pipeline="A2T", captioner="reference", results_schema="v1")
        row = PromptRow(
            row_index=0,
            sample_id="sample-a",
            prompt="first caption",
            record={"audio_file": "audio/a.wav", "audio_cap": "first caption"},
        )
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            output_root = root / "results" / "20260616_1437"
            log_root = root / "logs" / "20260616_1437"
            run_config = _run_config(args, Path("data/splits/ttas_val.jsonl"), "audio_cap")
            ModularCascade(args, dataset_root=None).run([row], output_root, log_root, run_config, total_rows=1)

            with self.assertRaisesRegex(RuntimeError, "already complete"):
                ModularCascade(args, dataset_root=None).run([row], output_root, log_root, run_config, total_rows=1)

    def test_v1_schema_rejects_manifest_complete_without_run_json(self) -> None:
        args = _minimal_cascade_args(pipeline="A2T", captioner="reference", results_schema="v1")
        row = PromptRow(
            row_index=0,
            sample_id="sample-a",
            prompt="first caption",
            record={"audio_file": "audio/a.wav", "audio_cap": "first caption"},
        )
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            output_root = root / "results" / "20260616_1437"
            log_root = root / "logs" / "20260616_1437"
            run_config = _run_config(args, Path("data/splits/ttas_val.jsonl"), "audio_cap")
            ModularCascade(args, dataset_root=None).run([row], output_root, log_root, run_config, total_rows=1)
            for name in ("run.json", "inspect.json", "errors.json", "summary.json"):
                (output_root / name).unlink()

            with self.assertRaisesRegex(RuntimeError, "already complete"):
                ModularCascade(args, dataset_root=None).run([row], output_root, log_root, run_config, total_rows=1)

    def test_v1_schema_rejects_complete_with_errors_as_terminal(self) -> None:
        args = _minimal_cascade_args(pipeline="A2T", captioner="reference", results_schema="v1")
        row = PromptRow(
            row_index=0,
            sample_id="sample-a",
            prompt="first caption",
            record={"audio_file": "audio/a.wav", "audio_cap": "first caption"},
        )
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            output_root = root / "results" / "20260616_1437"
            log_root = root / "logs" / "20260616_1437"
            run_config = _run_config(args, Path("data/splits/ttas_val.jsonl"), "audio_cap")
            ModularCascade(args, dataset_root=None).run([row], output_root, log_root, run_config, total_rows=1)
            run_json = output_root / "run.json"
            run_doc = json.loads(run_json.read_text(encoding="utf-8"))
            run_doc["status"] = "complete_with_errors"
            run_json.write_text(json.dumps(run_doc) + "\n", encoding="utf-8")

            with self.assertRaisesRegex(RuntimeError, "already complete"):
                ModularCascade(args, dataset_root=None).run([row], output_root, log_root, run_config, total_rows=1)

    def test_v1_dry_run_does_not_create_run_directory(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            split = root / "ttas_val.jsonl"
            split.write_text(
                json.dumps({"audio_file": "audio/source.wav", "audio_cap": "A motor starts."}) + "\n",
                encoding="utf-8",
            )
            config = root / "dataset.json"
            config.write_text(
                json.dumps({"dataset_root": str(root), "splits": {"val": str(split), "train": str(split), "test": str(split)}}) + "\n",
                encoding="utf-8",
            )
            argv = [
                "baseline_cascaded.py",
                "--pipeline",
                "A2T",
                "--captioner",
                "reference",
                "--dataset-config",
                str(config),
                "--split",
                "val",
                "--results-schema",
                "v1",
                "--run-id",
                "20260616_1437",
                "--dry-run",
            ]
            env = {"SALM_RESULTS_ROOT": str(root / "results"), "SALM_LOGS_ROOT": str(root / "logs")}
            with patch.object(sys, "argv", argv), patch.dict("os.environ", env):
                main()

            self.assertFalse((root / "results").exists())
            self.assertFalse((root / "logs").exists())

    def test_v1_t2t_direction_and_text_model_path_are_recorded(self) -> None:
        args = _minimal_cascade_args(
            pipeline="T2T",
            text_model="template",
            text_mode="speech_to_audio_description",
            text_model_path=(root_path("model_root") / "text-model").as_posix(),
            model_cache_dir=(root_path("cache_root") / "huggingface").as_posix(),
            results_schema="v1",
        )
        row = PromptRow(
            row_index=0,
            sample_id="sample-a",
            prompt="spoken words",
            record={
                "speech_text": "spoken words",
                "_cascaded_sample": _carried_sample(
                    row_index=0,
                    sample_id="sample-a",
                    stages=[
                        {
                            "stage": 1,
                            "task": "asr",
                            "model": "reference",
                            "input": {"type": "speech_audio", "file": "speech/source.wav"},
                            "output": {"type": "speech_text", "text": "spoken words"},
                        }
                    ],
                ),
                "_cascaded_stage_definitions": [
                    {
                        "stage": 1,
                        "task": "speech_to_text",
                        "model": "reference",
                        "checkpoint": (root_path("model_root") / "asr-reference").as_posix(),
                        "generation": {},
                        "runtime": {
                            "device": "cpu",
                            "model_load_seconds": None,
                            "cache_dir": (root_path("cache_root") / "asr-cache").as_posix(),
                        },
                    }
                ],
                "_cascaded_stage_lineage": [
                    {
                        "stage_index": 1,
                        "stage": "s2t",
                        "model": "reference",
                        "run_id": "20260616_1200",
                        "run_path": "results/cascaded/stages/s2t/reference/ttas/val/20260616_1200",
                        "manifest": "results/cascaded/stages/s2t/reference/ttas/val/20260616_1200/manifest.jsonl",
                        "config": "results/cascaded/stages/s2t/reference/ttas/val/20260616_1200/config.json",
                    }
                ],
            },
        )
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            output_root = root / "results" / "cascaded" / "stages" / "t2t" / "s2a" / "template" / "from-s2t" / "reference" / "ttas" / "val" / "20260616_1437"
            log_root = root / "logs" / "cascaded" / "stages" / "t2t" / "s2a" / "template" / "from-s2t" / "reference" / "ttas" / "val" / "20260616_1437"
            run_config = _run_config(args, Path("data/splits/ttas_val.jsonl"), "speech_text")

            ModularCascade(args, dataset_root=None).run([row], output_root, log_root, run_config, total_rows=1)
            run_doc = json.loads((output_root / "run.json").read_text(encoding="utf-8"))
            config = json.loads((output_root / "config.json").read_text(encoding="utf-8"))
            inspect = json.loads((output_root / "inspect.json").read_text(encoding="utf-8"))

        self.assertEqual(run_doc["direction"], "s2a")
        self.assertEqual(run_doc["lineage"]["direction"], "s2a")
        self.assertEqual(run_doc["lineage"]["stages"][0]["run_id"], "20260616_1200")
        self.assertEqual(
            run_doc["lineage"]["stages"][0]["run_path"],
            "results/cascaded/stages/s2t/reference/ttas/val/20260616_1200",
        )
        self.assertEqual(run_doc["lineage"]["stages"][1]["run_id"], "20260616_1437")
        self.assertEqual(run_doc["lineage"]["stages"][1]["run_path"], output_root.as_posix())
        self.assertEqual([stage["stage"] for stage in inspect["stage_configs"]], ["s2t", "t2t"])
        runtime_prompt = inspect["samples"][0]["stages"][1]["runtime_prompt"]
        self.assertEqual(runtime_prompt["messages"][0]["role"], "user")
        self.assertIn("Speech content:\nspoken words", runtime_prompt["messages"][0]["content"])
        self.assertEqual(config["model"]["path"], "{model_root}/text-model")
        self.assertIsNone(config["model"]["cache_dir"])
        self.assertEqual(inspect["stage_configs"][0]["model"]["path"], "{model_root}/asr-reference")
        self.assertEqual(inspect["stage_configs"][0]["model"]["cache_dir"], "{cache_root}/asr-cache")
        self.assertIsNone(inspect["stage_configs"][1]["model"]["cache_dir"])

    def test_v1_t2t_requires_carried_stage1_sample(self) -> None:
        args = _minimal_cascade_args(pipeline="T2T", text_model="template", results_schema="v1")
        row = PromptRow(row_index=0, sample_id="sample-a", prompt="raw text", record={"audio_cap": "raw text"})
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            output_root = root / "results" / "20260616_1437"

            with self.assertRaisesRegex(ValueError, "requires rows with exactly 1 carried upstream stage"):
                ModularCascade(args, dataset_root=None).run(
                    [row],
                    output_root,
                    root / "logs" / "20260616_1437",
                    _run_config(args, Path("data/splits/ttas_val.jsonl"), "audio_cap"),
                    total_rows=1,
                )

            self.assertFalse(output_root.exists())

    def test_v1_t2t_rejects_mismatched_carried_sample_id(self) -> None:
        args = _minimal_cascade_args(pipeline="T2T", text_model="template", results_schema="v1")
        row = PromptRow(
            row_index=0,
            sample_id="sample-a",
            prompt="raw text",
            record={
                "audio_cap": "raw text",
                "_cascaded_sample": _carried_sample(
                    row_index=0,
                    sample_id="other-sample",
                    stages=[
                        {
                            "stage": 1,
                            "task": "audio_to_text",
                            "model": "reference",
                            "input": {"type": "audio", "file": "audio/source.wav"},
                            "output": {"type": "audio_caption", "text": "raw text"},
                        }
                    ],
                ),
            },
        )
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            output_root = root / "results" / "20260616_1437"

            with self.assertRaisesRegex(ValueError, "carried sample_id to match"):
                ModularCascade(args, dataset_root=None).run(
                    [row],
                    output_root,
                    root / "logs" / "20260616_1437",
                    _run_config(args, Path("data/splits/ttas_val.jsonl"), "audio_cap"),
                    total_rows=1,
                )

            self.assertFalse(output_root.exists())

    def test_v1_t2t_rejects_unexpected_upstream_model(self) -> None:
        args = _minimal_cascade_args(pipeline="T2T", captioner="reference", text_model="template", results_schema="v1")
        row = PromptRow(
            row_index=0,
            sample_id="sample-a",
            prompt="raw text",
            record={
                "audio_cap": "raw text",
                "_cascaded_sample": _carried_sample(
                    row_index=0,
                    sample_id="sample-a",
                    stages=[
                        {
                            "stage": 1,
                            "task": "audio_to_text",
                            "model": "other-captioner",
                            "input": {"type": "audio", "file": "audio/source.wav"},
                            "output": {"type": "audio_caption", "text": "raw text"},
                        }
                    ],
                ),
            },
        )
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            output_root = root / "results" / "20260616_1437"

            with self.assertRaisesRegex(ValueError, "requires previous carried model 'reference'"):
                ModularCascade(args, dataset_root=None).run(
                    [row],
                    output_root,
                    root / "logs" / "20260616_1437",
                    _run_config(args, Path("data/splits/ttas_val.jsonl"), "audio_cap"),
                    total_rows=1,
                )

            self.assertFalse(output_root.exists())

    def test_v1_t2a_writes_media_under_outputs(self) -> None:
        args = _minimal_cascade_args(pipeline="T2A", t2a="audiogen", results_schema="v1")

        class FakeRunner:
            model_path = None

            def generate(self, row, output_path):
                output_path.write_bytes(b"RIFFfake")
                from baselines.models.base import GenerationResult

                return GenerationResult(output_path=output_path, sample_rate=32000, extra={"shape": [320000]})

        row = PromptRow(
            row_index=0,
            sample_id="sample-a",
            prompt="engine turning",
            record={
                "audio_cap": "engine turning",
                "_cascaded_sample": _carried_sample(
                    row_index=0,
                    sample_id="sample-a",
                    stages=[
                        {
                            "stage": 1,
                            "task": "asr",
                            "model": "reference",
                            "input": {"type": "speech_audio", "file": "speech/source.wav"},
                            "output": {"type": "speech_text", "text": "turn the engine"},
                        },
                        {
                            "stage": 2,
                            "task": "text_to_text",
                            "model": "template",
                            "mode": "speech_to_audio_description",
                            "input": {"type": "speech_text", "text": "turn the engine"},
                            "output": {"type": "audio_description", "text": "engine turning"},
                        },
                    ],
                ),
            },
        )
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            output_root = root / "results" / "cascaded" / "stages" / "t2a" / "audiogen" / "from-t2t" / "template" / "ttas" / "val" / "20260616_1437"
            log_root = root / "logs" / "cascaded" / "stages" / "t2a" / "audiogen" / "from-t2t" / "template" / "ttas" / "val" / "20260616_1437"
            run_config = _run_config(args, Path("data/splits/ttas_val.jsonl"), "audio_cap")
            cascade = ModularCascade(args, dataset_root=None)
            cascade._runners[("audiogen", "T2A")] = FakeRunner()

            cascade.run([row], output_root, log_root, run_config, total_rows=1)
            manifest = _read_jsonl(output_root / "manifest.jsonl")
            media = output_root / manifest[0]["output"]["audio_file"]
            media_bytes = media.read_bytes()

        self.assertEqual(manifest[0]["output"]["audio_file"], "outputs/000000_sample-a.wav")
        self.assertEqual(media_bytes, b"RIFFfake")

    def test_v1_t2a_requires_two_carried_stages(self) -> None:
        args = _minimal_cascade_args(pipeline="T2A", t2a="audiogen", results_schema="v1")
        row = PromptRow(row_index=0, sample_id="sample-a", prompt="engine turning", record={"audio_cap": "engine turning"})
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            output_root = root / "results" / "20260616_1437"

            with self.assertRaisesRegex(ValueError, "requires rows with exactly 2 carried upstream stage"):
                ModularCascade(args, dataset_root=None).run(
                    [row],
                    output_root,
                    root / "logs" / "20260616_1437",
                    _run_config(args, Path("data/splits/ttas_val.jsonl"), "audio_cap"),
                    total_rows=1,
                )

            self.assertFalse(output_root.exists())

    def test_v1_main_uses_design_stage_path_with_run_id(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            split = root / "ttas_val.jsonl"
            split.write_text(
                json.dumps(
                    {
                        "sample_id": "sample-a",
                        "audio_file": "audio/source.wav",
                        "audio_cap": "A motor starts.",
                        "speech_file": "speech/source.wav",
                        "speech_text": "Try the starter again.",
                    }
                )
                + "\n",
                encoding="utf-8",
            )
            config = root / "dataset.json"
            config.write_text(
                json.dumps(
                    {
                        "dataset_root": str(root),
                        "splits": {"val": str(split), "train": str(split), "test": str(split)},
                    }
                )
                + "\n",
                encoding="utf-8",
            )
            argv = [
                "baseline_cascaded.py",
                "--pipeline",
                "A2T",
                "--captioner",
                "reference",
                "--dataset-config",
                str(config),
                "--split",
                "val",
                "--num-samples",
                "1",
                "--results-schema",
                "v1",
                "--run-id",
                "20260616_1437",
                "--device",
                "cpu",
            ]
            env = {
                "SALM_RESULTS_ROOT": str(root / "results"),
                "SALM_LOGS_ROOT": str(root / "logs"),
            }
            with patch.object(sys, "argv", argv), patch.dict("os.environ", env):
                main()

            design_root = root / "results" / "cascaded" / "stages" / "a2t" / "reference" / "ttas" / "val" / "20260616_1437"
            legacy_root = root / "results" / "baselines" / "cascaded" / "A2T" / "audio_captioning-reference" / "ttas" / "val"
            manifest_exists = (design_root / "manifest.jsonl").is_file()
            legacy_exists = legacy_root.exists()
            run_doc = json.loads((design_root / "run.json").read_text(encoding="utf-8"))

        self.assertTrue(manifest_exists)
        self.assertFalse(legacy_exists)
        self.assertEqual(run_doc["run_id"], "20260616_1437")
        self.assertEqual(run_doc["paths"]["result"], design_root.as_posix())

    def test_v1_t2t_consumes_direct_a2t_manifest(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            split = root / "ttas_val.jsonl"
            split.write_text(
                json.dumps(
                    {
                        "sample_id": "sample-a",
                        "audio_file": "audio/source.wav",
                        "audio_cap": "reference caption",
                        "speech_file": "speech/source.wav",
                        "speech_text": "reference speech",
                    }
                )
                + "\n",
                encoding="utf-8",
            )
            config = root / "dataset.json"
            config.write_text(
                json.dumps({"dataset_root": str(root), "splits": {"val": str(split), "train": str(split), "test": str(split)}}) + "\n",
                encoding="utf-8",
            )
            env = {"SALM_RESULTS_ROOT": str(root / "results"), "SALM_LOGS_ROOT": str(root / "logs")}
            with patch.dict("os.environ", env):
                with patch.object(
                    sys,
                    "argv",
                    [
                        "baseline_cascaded.py",
                        "--pipeline",
                        "A2T",
                        "--captioner",
                        "reference",
                        "--dataset-config",
                        str(config),
                        "--split",
                        "val",
                        "--num-samples",
                        "1",
                        "--results-schema",
                        "v1",
                        "--run-id",
                        "20260616_1200",
                        "--device",
                        "cpu",
                    ],
                ):
                    main()
                a2t_manifest = (
                    root
                    / "results"
                    / "cascaded"
                    / "stages"
                    / "a2t"
                    / "reference"
                    / "ttas"
                    / "val"
                    / "20260616_1200"
                    / "manifest.jsonl"
                )
                with patch.object(
                    sys,
                    "argv",
                    [
                        "baseline_cascaded.py",
                        "--pipeline",
                        "T2T",
                        "--text-model",
                        "template",
                        "--text-mode",
                        "audio_description_to_speech",
                        "--captioner",
                        "reference",
                        "--dataset-config",
                        str(config),
                        "--split",
                        "val",
                        "--split-file",
                        str(a2t_manifest),
                        "--num-samples",
                        "1",
                        "--results-schema",
                        "v1",
                        "--run-id",
                        "20260616_1300",
                        "--device",
                        "cpu",
                    ],
                ):
                    main()

            t2t_root = (
                root
                / "results"
                / "cascaded"
                / "stages"
                / "t2t"
                / "a2s"
                / "template"
                / "from-a2t"
                / "reference"
                / "ttas"
                / "val"
                / "20260616_1300"
            )
            manifest_row = json.loads((t2t_root / "manifest.jsonl").read_text(encoding="utf-8").splitlines()[0])
            config_doc = json.loads((t2t_root / "config.json").read_text(encoding="utf-8"))
            inspect_doc = json.loads((t2t_root / "inspect.json").read_text(encoding="utf-8"))

        self.assertEqual(config_doc["input"]["input_field"], "audio_caption")
        self.assertEqual(manifest_row["input"]["source_sample_id"], "source")
        self.assertEqual(manifest_row["output"]["type"], "text")
        self.assertEqual(manifest_row["output"]["generated_speech_text"], "I can hear it clearly. reference caption We should respond to what is happening there.")
        self.assertEqual([stage["stage"] for stage in inspect_doc["stage_configs"]], ["a2t", "t2t"])
        self.assertEqual(len(inspect_doc["samples"][0]["stages"]), 2)

    def test_v1_t2t_consumes_direct_s2t_manifest(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            split = root / "ttas_val.jsonl"
            split.write_text(
                json.dumps(
                    {
                        "sample_id": "sample-a",
                        "audio_file": "audio/source.wav",
                        "audio_cap": "reference caption",
                        "speech_file": "speech/source.wav",
                        "speech_text": "start the engine",
                    }
                )
                + "\n",
                encoding="utf-8",
            )
            config = root / "dataset.json"
            config.write_text(
                json.dumps({"dataset_root": str(root), "splits": {"val": str(split), "train": str(split), "test": str(split)}}) + "\n",
                encoding="utf-8",
            )
            env = {"SALM_RESULTS_ROOT": str(root / "results"), "SALM_LOGS_ROOT": str(root / "logs")}
            with patch.dict("os.environ", env):
                with patch.object(
                    sys,
                    "argv",
                    [
                        "baseline_cascaded.py",
                        "--pipeline",
                        "S2T",
                        "--asr",
                        "reference",
                        "--dataset-config",
                        str(config),
                        "--split",
                        "val",
                        "--num-samples",
                        "1",
                        "--results-schema",
                        "v1",
                        "--run-id",
                        "20260616_1200",
                        "--device",
                        "cpu",
                    ],
                ):
                    main()
                s2t_manifest = (
                    root
                    / "results"
                    / "cascaded"
                    / "stages"
                    / "s2t"
                    / "reference"
                    / "ttas"
                    / "val"
                    / "20260616_1200"
                    / "manifest.jsonl"
                )
                with patch.object(
                    sys,
                    "argv",
                    [
                        "baseline_cascaded.py",
                        "--pipeline",
                        "T2T",
                        "--text-model",
                        "template",
                        "--text-mode",
                        "speech_to_audio_description",
                        "--asr",
                        "reference",
                        "--dataset-config",
                        str(config),
                        "--split",
                        "val",
                        "--split-file",
                        str(s2t_manifest),
                        "--num-samples",
                        "1",
                        "--results-schema",
                        "v1",
                        "--run-id",
                        "20260616_1300",
                        "--device",
                        "cpu",
                    ],
                ):
                    main()

            t2t_root = (
                root
                / "results"
                / "cascaded"
                / "stages"
                / "t2t"
                / "s2a"
                / "template"
                / "from-s2t"
                / "reference"
                / "ttas"
                / "val"
                / "20260616_1300"
            )
            manifest_row = json.loads((t2t_root / "manifest.jsonl").read_text(encoding="utf-8").splitlines()[0])
            config_doc = json.loads((t2t_root / "config.json").read_text(encoding="utf-8"))
            inspect_doc = json.loads((t2t_root / "inspect.json").read_text(encoding="utf-8"))

        self.assertEqual(config_doc["input"]["input_field"], "speech_transcription")
        self.assertEqual(manifest_row["input"]["source_sample_id"], "source")
        self.assertEqual(manifest_row["output"]["generated_audio_caption"], "Environmental audio matching this spoken situation: start the engine")
        self.assertEqual([stage["stage"] for stage in inspect_doc["stage_configs"]], ["s2t", "t2t"])
        self.assertEqual(len(inspect_doc["samples"][0]["stages"]), 2)

    def test_v1_direct_stage_manifest_skips_failed_upstream_rows(self) -> None:
        args = _minimal_cascade_args(
            pipeline="T2T",
            text_mode="audio_description_to_speech",
            captioner="reference",
            results_schema="v1",
        )
        with tempfile.TemporaryDirectory() as tmpdir:
            manifest = Path(tmpdir) / "manifest.jsonl"
            rows = [
                {
                    "schema_version": 1,
                    "row_index": 0,
                    "sample_id": "ok-a",
                    "stage": "a2t",
                    "model": "reference",
                    "status": "success",
                    "input": {"audio_file": "audio/a.wav"},
                    "output": {"type": "text", "audio_caption": "engine idling"},
                    "timing": {"inference_seconds": 0.1},
                    "error": None,
                },
                {
                    "schema_version": 1,
                    "row_index": 1,
                    "sample_id": "bad-audio",
                    "stage": "a2t",
                    "model": "reference",
                    "status": "error",
                    "input": {"audio_file": "audio/b.wav"},
                    "output": None,
                    "timing": {"inference_seconds": 0.01},
                    "error": {"type": "IndexError", "message": "empty features"},
                },
                {
                    "schema_version": 1,
                    "row_index": 2,
                    "sample_id": "ok-b",
                    "stage": "a2t",
                    "model": "reference",
                    "status": "success",
                    "input": {"audio_file": "audio/c.wav"},
                    "output": {"type": "text", "audio_caption": "rain falling"},
                    "timing": {"inference_seconds": 0.1},
                    "error": None,
                },
            ]
            manifest.write_text("\n".join(json.dumps(row) for row in rows) + "\n", encoding="utf-8")

            loaded, input_field = _load_stage_rows(
                args,
                manifest,
                prompt_field="audio_cap",
                start_index=0,
                num_samples=-1,
            )
            windowed, _ = _load_stage_rows(
                args,
                manifest,
                prompt_field="audio_cap",
                start_index=0,
                num_samples=2,
            )

        self.assertEqual(input_field, "audio_caption")
        self.assertEqual([row.sample_id for row in loaded], ["ok-a", "ok-b"])
        self.assertEqual([row.row_index for row in loaded], [0, 2])
        self.assertEqual([row.prompt for row in loaded], ["engine idling", "rain falling"])
        self.assertEqual([row.sample_id for row in windowed], ["ok-a"])

    def test_v1_main_stage_input_keeps_cli_stage_pipeline(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            split = root / "t2t_stage_input.jsonl"
            split.write_text(
                json.dumps(
                    {
                        "sample_id": "sample-a",
                        "speech_text": "start the engine",
                        "_cascaded_pipeline": "S2A",
                        "_cascaded_sample": _carried_sample(
                            row_index=0,
                            sample_id="sample-a",
                            stages=[
                                {
                                    "stage": 1,
                                    "task": "asr",
                                    "model": "reference",
                                    "input": {"type": "speech_audio", "file": "speech/source.wav"},
                                    "output": {"type": "speech_text", "text": "start the engine"},
                                }
                            ],
                        ),
                    }
                )
                + "\n",
                encoding="utf-8",
            )
            config = root / "dataset.json"
            config.write_text(
                json.dumps({"dataset_root": str(root), "splits": {"val": str(split), "train": str(split), "test": str(split)}}) + "\n",
                encoding="utf-8",
            )
            argv = [
                "baseline_cascaded.py",
                "--pipeline",
                "T2T",
                "--text-model",
                "template",
                "--asr",
                "reference",
                "--dataset-config",
                str(config),
                "--split",
                "val",
                "--num-samples",
                "1",
                "--results-schema",
                "v1",
                "--run-id",
                "20260616_1437",
                "--device",
                "cpu",
            ]
            env = {"SALM_RESULTS_ROOT": str(root / "results"), "SALM_LOGS_ROOT": str(root / "logs")}
            with patch.object(sys, "argv", argv), patch.dict("os.environ", env):
                main()

            design_root = root / "results" / "cascaded" / "stages" / "t2t" / "s2a" / "template" / "from-s2t" / "reference" / "ttas" / "val" / "20260616_1437"
            run_doc = json.loads((design_root / "run.json").read_text(encoding="utf-8"))
            config_doc = json.loads((design_root / "config.json").read_text(encoding="utf-8"))

        self.assertEqual(run_doc["stage"], "t2t")
        self.assertEqual(run_doc["direction"], "s2a")
        self.assertEqual(config_doc["execution"]["legacy_pipeline"], "T2T")
        self.assertEqual(config_doc["execution"]["cascade_pipeline"], "S2A")

    def test_cascade_variant_distinguishes_backend_choices(self) -> None:
        audiogen = argparse.Namespace(pipeline="S2A", run_name=None, asr="reference", text_model="template", t2a="audiogen")
        audio_omni = argparse.Namespace(pipeline="S2A", run_name=None, asr="reference", text_model="template", t2a="audio_omni")
        self.assertNotEqual(_cascade_variant_name(audiogen), _cascade_variant_name(audio_omni))

    def test_cascade_variant_can_be_overridden(self) -> None:
        args = argparse.Namespace(pipeline="S2A", run_name="smoke lane/0", asr="reference", text_model="template", t2a="audiogen")
        self.assertEqual(_cascade_variant_name(args), "smoke_lane_0")

    def test_planned_stages_for_s2a(self) -> None:
        args = argparse.Namespace(pipeline="S2A", asr="whisper", text_model="qwen3_32b", t2a="tangoflux")
        self.assertEqual(
            _planned_stages(args),
            [
                {"stage": "asr", "backend": "whisper"},
                {"stage": "text_to_text", "backend": "qwen3_32b"},
                {"stage": "text_to_audio", "backend": "tangoflux"},
            ],
        )

    def test_planned_stages_for_a2s(self) -> None:
        args = argparse.Namespace(
            pipeline="A2S",
            captioner="audio_flamingo_next",
            text_model="llama3_3_70b_instruct",
            tts="f5_tts",
        )
        self.assertEqual(
            _planned_stages(args),
            [
                {"stage": "audio_captioning", "backend": "audio_flamingo_next"},
                {"stage": "text_to_text", "backend": "llama3_3_70b_instruct"},
                {"stage": "text_to_speech", "backend": "f5_tts"},
            ],
        )

    def test_planned_stages_for_cosyvoice3_t2s(self) -> None:
        args = argparse.Namespace(pipeline="T2S", asr="reference", captioner="reference", text_model="template", tts="cosyvoice3", t2a="audiogen")
        self.assertEqual(_planned_stages(args), [{"stage": "text_to_speech", "backend": "cosyvoice3"}])

    def test_planned_stages_for_t2t(self) -> None:
        args = argparse.Namespace(pipeline="T2T", text_model="qwen3_32b")
        self.assertEqual(_planned_stages(args), [{"stage": "text_to_text", "backend": "qwen3_32b"}])


    def test_text_llm_applies_prompt_template(self) -> None:
        class FakeTensor:
            shape = (1, 3)

            def to(self, device):
                return self

        class FakeInputs(dict):
            def to(self, device):
                return self

        class FakeTokenizer:
            eos_token_id = 0

            def __init__(self):
                self.messages = None
                self.encoded_text = None

            def apply_chat_template(self, messages, tokenize=False, add_generation_prompt=True, enable_thinking=False):
                self.messages = messages
                return messages[0]["content"]

            def __call__(self, text, return_tensors):
                self.encoded_text = text
                return FakeInputs({"input_ids": FakeTensor()})

            def decode(self, generated, skip_special_tokens=True):
                return "converted text"

        class FakeGenerated:
            def __getitem__(self, key):
                return [1, 2, 3]

        class FakeModel:
            device = "cpu"

            def generate(self, **kwargs):
                return FakeGenerated()

        old_torch = sys.modules.get("torch")
        sys.modules["torch"] = types.SimpleNamespace(inference_mode=lambda: contextlib.nullcontext())
        try:
            runner = TextLLMRunner(
                model_path=None,
                device="cpu",
                duration=0,
                steps=25,
                seed=0,
                task="text_to_text",
                prompt_template="Direction instruction: {prompt}",
            )
            runner.tokenizer = FakeTokenizer()
            runner.model = FakeModel()
            row = PromptRow(row_index=0, sample_id="sample", prompt="raw upstream text", record={})
            with tempfile.TemporaryDirectory() as tmp:
                result = runner.generate(row, Path(tmp) / "out.json")
            self.assertEqual(runner.tokenizer.messages[0]["content"], "Direction instruction: raw upstream text")
            self.assertEqual(runner.tokenizer.encoded_text, "Direction instruction: raw upstream text")
            self.assertEqual(result.extra["input_text"], "raw upstream text")
            self.assertEqual(result.extra["generation_prompt"], "Direction instruction: raw upstream text")
        finally:
            if old_torch is None:
                sys.modules.pop("torch", None)
            else:
                sys.modules["torch"] = old_torch

    def test_strip_thinking_output_prefers_final_answer(self) -> None:
        self.assertEqual(_strip_thinking_output("<think>reasoning</think>rain on pavement"), "rain on pavement")

    def test_strip_thinking_output_handles_orphan_think_prefix(self) -> None:
        self.assertEqual(
            _strip_thinking_output("<think>\nreasoning line\ncity traffic with idling motorcycle"),
            "city traffic with idling motorcycle",
        )

    def test_audio_flamingo_next_compat_patch_registers_musicflamingo_alias(self) -> None:
        transformers = types.ModuleType("transformers")

        class AudioFlamingo3Config:
            pass

        class AudioFlamingo3ForConditionalGeneration:
            pass

        class AudioFlamingo3Processor:
            pass

        class AutoRegistry:
            registered = []

            @classmethod
            def register(cls, *args, **kwargs):
                cls.registered.append((args, kwargs))

        class AutoConfig(AutoRegistry):
            registered = []

        class AutoProcessor(AutoRegistry):
            registered = []

        transformers.AudioFlamingo3Config = AudioFlamingo3Config
        transformers.AudioFlamingo3ForConditionalGeneration = AudioFlamingo3ForConditionalGeneration
        transformers.AudioFlamingo3Processor = AudioFlamingo3Processor
        transformers.AutoConfig = AutoConfig
        transformers.AutoProcessor = AutoProcessor
        old_transformers = sys.modules.get("transformers")
        sys.modules["transformers"] = transformers
        try:
            _patch_audio_flamingo_next_transformers_compat()
            self.assertTrue(issubclass(transformers.MusicFlamingoConfig, AudioFlamingo3Config))
            self.assertEqual(transformers.MusicFlamingoConfig.model_type, "musicflamingo")
            self.assertIs(transformers.MusicFlamingoForConditionalGeneration, AudioFlamingo3ForConditionalGeneration)
            self.assertIs(transformers.MusicFlamingoProcessor, AudioFlamingo3Processor)
            self.assertEqual(AutoConfig.registered[0][0], ("musicflamingo", transformers.MusicFlamingoConfig))
            self.assertEqual(AutoProcessor.registered[0][0], (transformers.MusicFlamingoConfig, AudioFlamingo3Processor))
        finally:
            if old_transformers is None:
                sys.modules.pop("transformers", None)
            else:
                sys.modules["transformers"] = old_transformers

    def test_qwen_asr_compat_patch_adds_transformers_helpers(self) -> None:
        generic = types.ModuleType("transformers.utils.generic")
        rope_utils = types.ModuleType("transformers.modeling_rope_utils")
        rope_utils.ROPE_INIT_FUNCTIONS = {}
        utils = types.ModuleType("transformers.utils")
        transformers = types.ModuleType("transformers")
        qwen_asr = types.ModuleType("qwen_asr")
        qwen_core = types.ModuleType("qwen_asr.core")
        qwen_backend = types.ModuleType("qwen_asr.core.transformers_backend")
        qwen_config = types.ModuleType("qwen_asr.core.transformers_backend.configuration_qwen3_asr")

        class Qwen3ASRThinkerConfig:
            def __init__(self) -> None:
                self.audio_token_id = 151676

        qwen_config.Qwen3ASRThinkerConfig = Qwen3ASRThinkerConfig
        utils.generic = generic
        transformers.utils = utils
        transformers.modeling_rope_utils = rope_utils
        old_transformers = sys.modules.get("transformers")
        old_utils = sys.modules.get("transformers.utils")
        old_generic = sys.modules.get("transformers.utils.generic")
        old_rope_utils = sys.modules.get("transformers.modeling_rope_utils")
        old_qwen_asr = sys.modules.get("qwen_asr")
        old_qwen_core = sys.modules.get("qwen_asr.core")
        old_qwen_backend = sys.modules.get("qwen_asr.core.transformers_backend")
        old_qwen_config = sys.modules.get("qwen_asr.core.transformers_backend.configuration_qwen3_asr")
        sys.modules["transformers"] = transformers
        sys.modules["transformers.utils"] = utils
        sys.modules["transformers.utils.generic"] = generic
        sys.modules["transformers.modeling_rope_utils"] = rope_utils
        sys.modules["qwen_asr"] = qwen_asr
        sys.modules["qwen_asr.core"] = qwen_core
        sys.modules["qwen_asr.core.transformers_backend"] = qwen_backend
        sys.modules["qwen_asr.core.transformers_backend.configuration_qwen3_asr"] = qwen_config
        try:
            _patch_qwen_asr_transformers_compat()
            def direct_fn():
                return None

            def decorated_fn():
                return None

            self.assertIs(generic.check_model_inputs(direct_fn), direct_fn)
            self.assertIs(generic.check_model_inputs()(decorated_fn), decorated_fn)
            self.assertIn("default", rope_utils.ROPE_INIT_FUNCTIONS)
            self.assertNotIn("nagisa", sys.modules)
            thinker_config = Qwen3ASRThinkerConfig()
            self.assertIsNone(thinker_config.pad_token_id)
        finally:
            if old_transformers is None:
                sys.modules.pop("transformers", None)
            else:
                sys.modules["transformers"] = old_transformers
            if old_utils is None:
                sys.modules.pop("transformers.utils", None)
            else:
                sys.modules["transformers.utils"] = old_utils
            if old_generic is None:
                sys.modules.pop("transformers.utils.generic", None)
            else:
                sys.modules["transformers.utils.generic"] = old_generic
            if old_rope_utils is None:
                sys.modules.pop("transformers.modeling_rope_utils", None)
            else:
                sys.modules["transformers.modeling_rope_utils"] = old_rope_utils
            for name, old_module in (
                ("qwen_asr", old_qwen_asr),
                ("qwen_asr.core", old_qwen_core),
                ("qwen_asr.core.transformers_backend", old_qwen_backend),
                ("qwen_asr.core.transformers_backend.configuration_qwen3_asr", old_qwen_config),
            ):
                if old_module is None:
                    sys.modules.pop(name, None)
                else:
                    sys.modules[name] = old_module


def _minimal_cascade_args(**overrides) -> argparse.Namespace:
    values = {
        "pipeline": "A2T",
        "dataset": "ttas",
        "split": "val",
        "run_name": None,
        "device": "cpu",
        "duration": 10.0,
        "steps": 25,
        "seed": 1337,
        "overwrite": False,
        "continue_on_error": False,
        "dry_run": False,
        "results_schema": "legacy",
        "shard_index": None,
        "num_shards": None,
        "asr": "reference",
        "captioner": "reference",
        "text_model": "template",
        "text_mode": "audio_description_to_speech",
        "tts": "f5_tts",
        "t2a": "audiogen",
        "model_path": None,
        "model_cache_dir": None,
        "asr_model_path": None,
        "captioner_model_path": None,
        "text_model_path": None,
        "tts_model_path": None,
        "t2a_model_path": None,
        "prompt_template": None,
        "qwen_speaker": "Ethan",
        "qwen_template": None,
        "tangoflux_source_dir": None,
        "audio_omni_config": None,
        "audio_omni_ckpt": None,
        "audio_omni_source_dir": None,
        "f5_tts_source_dir": None,
        "f5_tts_vocoder_path": None,
        "config_file": None,
        "dataset_config": "configs/ttas_dataset.json",
        "output_dir": "results/baselines",
        "log_dir": "logs/baselines",
        "temporary": False,
        "start_index": 0,
        "num_samples": 5,
    }
    values.update(overrides)
    return argparse.Namespace(**values)


def _carried_sample(*, row_index: int, sample_id: str, stages: list[dict]) -> dict:
    normalized_stages = []
    for stage in stages:
        item = {
            "stage": stage["stage"],
            "task": stage["task"],
            "model": stage["model"],
            "input": stage.get("input"),
            "prompt": stage.get("prompt"),
            "inference_seconds": stage.get("inference_seconds", 0.0),
            "output": stage.get("output"),
            "error": stage.get("error"),
        }
        if stage.get("mode") is not None:
            item["mode"] = stage["mode"]
        normalized_stages.append(item)
    return {
        "row_index": row_index,
        "sample_id": sample_id,
        "status": "success",
        "reference": {
            "audio_file": "audio/source.wav",
            "audio_caption": "engine idling",
            "speech_file": "speech/source.wav",
            "speech_text": "turn the engine",
        },
        "stages": normalized_stages,
        "final_output": None,
        "error": None,
    }


def _read_jsonl(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line]


if __name__ == "__main__":
    unittest.main()
