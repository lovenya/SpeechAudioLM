from __future__ import annotations

import argparse
import contextlib
import json
import sys
import tempfile
import types
import unittest
from pathlib import Path
from unittest.mock import patch

from baseline_cascaded import (
    ModularCascade,
    _cascade_output_root,
    _cascade_variant_name,
    _pipeline_prompt_field,
    _planned_stages,
    _run_config,
    _temporary_path,
    main,
)
from baselines.models.audio_captioning import _patch_audio_flamingo_next_transformers_compat
from baselines.models.speech_asr import _patch_qwen_asr_transformers_compat
from baselines.dataset import PromptRow, iter_split_rows
from baselines.models.text_llm import TextLLMRunner, _strip_thinking_output


class CascadedBaselineTest(unittest.TestCase):
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
        self.assertEqual(run_doc["paths"], {"result": output_root.as_posix(), "logs": log_root.as_posix()})
        self.assertEqual(config["input"]["manifest"], "data/splits/ttas_val.jsonl")
        self.assertEqual(config["input"]["input_field"], "audio_cap")
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

            with self.assertRaisesRegex(FileExistsError, "v1 artifact already exists"):
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
        self.assertEqual(run_doc["events"][1]["existing_successful_rows"], 1)

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
            text_model_path="{model_root}/text-model",
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
        self.assertEqual(config["model"]["path"], "{model_root}/text-model")

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
        "asr": "reference",
        "captioner": "reference",
        "text_model": "template",
        "text_mode": "audio_description_to_speech",
        "tts": "f5_tts",
        "t2a": "audiogen",
        "model_path": None,
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
