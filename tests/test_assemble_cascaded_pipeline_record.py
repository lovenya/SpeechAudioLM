from __future__ import annotations

import json
import tempfile
import unittest
from datetime import datetime, timezone
from pathlib import Path
from unittest.mock import patch

from scripts.assemble_cascaded_pipeline_record import assemble_pipeline_record


class AssembleCascadedPipelineRecordTest(unittest.TestCase):
    def test_a2s_joins_by_sample_id_not_row_position(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            a2t = _write_stage_run(
                root / "a2t",
                stage="a2t",
                model="captioner",
                direction="a2s",
                rows=[
                    _row(0, "sample-b", stage="a2t", model="captioner", input_value={"audio_file": "b-input.wav"}),
                    _row(1, "sample-a", stage="a2t", model="captioner", input_value={"audio_file": "a-input.wav"}),
                ],
            )
            t2t = _write_stage_run(
                root / "t2t",
                stage="t2t",
                model="text",
                direction="a2s",
                rows=[
                    _row(0, "sample-a", stage="t2t", model="text"),
                    _row(1, "sample-b", stage="t2t", model="text"),
                ],
            )
            t2s = _write_stage_run(
                root / "t2s",
                stage="t2s",
                model="tts",
                direction="a2s",
                rows=[
                    _row(
                        0,
                        "sample-a",
                        stage="t2s",
                        model="tts",
                        output={"type": "speech", "speech_file": "outputs/sample-a.wav"},
                    ),
                    _row(
                        1,
                        "sample-b",
                        stage="t2s",
                        model="tts",
                        output={"type": "speech", "speech_file": "outputs/sample-b.wav"},
                    ),
                ],
            )
            _update_run_json(a2t, started_at="2026-06-16T23:00:00+02:00", code={"git_commit": "a2t"})
            _update_run_json(t2t, started_at="2026-06-16T18:30:00-04:00", code={"git_commit": "t2t"})
            _update_run_json(t2s, started_at="2026-06-16T19:10:00+00:00", code={"git_commit": "t2s"})
            pipeline_dir = root / "pipeline" / "20260616_1530"

            with patch.dict("os.environ", {"SLURM_JOB_ID": "1234", "SLURM_ARRAY_JOB_ID": "1200", "SLURM_ARRAY_TASK_ID": "7"}):
                report = assemble_pipeline_record(
                    pipeline_dir,
                    direction="a2s",
                    stage_run_dirs=[a2t, t2t, t2s],
                    dataset="ttas",
                    split="val",
                    pipeline_name="caption_text_tts",
                    now=datetime(2026, 6, 16, 19, 30, tzinfo=timezone.utc),
                )
            rows = _read_jsonl(pipeline_dir / "manifest.jsonl")
            run = json.loads((pipeline_dir / "run.json").read_text(encoding="utf-8"))
            pipeline = json.loads((pipeline_dir / "pipeline.json").read_text(encoding="utf-8"))
            inspect = json.loads((pipeline_dir / "inspect.json").read_text(encoding="utf-8"))

        self.assertEqual(report["summary"]["samples"]["successful"], 2)
        self.assertEqual([row["sample_id"] for row in rows], ["sample-b", "sample-a"])
        self.assertEqual(rows[0]["input"], {"audio_file": "b-input.wav"})
        self.assertEqual(rows[0]["final_output"]["speech_file"], (t2s / "outputs" / "sample-b.wav").as_posix())
        self.assertEqual([stage["stage"] for stage in rows[0]["stages"]], ["a2t", "t2t", "t2s"])
        self.assertEqual(pipeline["final_output"]["stage"], "t2s")
        self.assertEqual(run["started_at"], "2026-06-16T19:10:00+00:00")
        self.assertEqual(run["completed_at"], "2026-06-16T15:30:00-04:00")
        self.assertEqual(run["code"]["stage_runs"][0]["code"], {"git_commit": "a2t"})
        self.assertEqual(run["code"]["stage_runs"][2]["code"], {"git_commit": "t2s"})
        self.assertEqual(run["slurm"], {"job_id": "1234", "array_job_id": "1200", "array_task_id": "7"})
        self.assertEqual(inspect["samples"][0]["reference"], {"audio_file": "b-input.wav", "reference_audio_caption": "reference caption for sample-b"})
        self.assertEqual(inspect["lineage"]["stages"][0]["config"], (a2t / "config.json").as_posix())
        self.assertFalse((pipeline_dir / "outputs").exists())

    def test_incomplete_row_reports_first_failed_stage_before_final_stage(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            a2t = _write_stage_run(
                root / "a2t",
                stage="a2t",
                model="captioner",
                direction="a2s",
                rows=[_row(0, "sample-a", stage="a2t", model="captioner", input_value={"audio_file": "a.wav"})],
            )
            t2t = _write_stage_run(
                root / "t2t",
                stage="t2t",
                model="text",
                direction="a2s",
                rows=[
                    _row(
                        0,
                        "sample-a",
                        stage="t2t",
                        model="text",
                        status="error",
                        output=None,
                        error={"type": "RuntimeError", "message": "text model failed"},
                    )
                ],
            )
            t2s = _write_stage_run(root / "t2s", stage="t2s", model="tts", direction="a2s", rows=[])
            pipeline_dir = root / "pipeline" / "20260616_1530"

            assemble_pipeline_record(pipeline_dir, direction="a2s", stage_run_dirs=[a2t, t2t, t2s])
            row = _read_jsonl(pipeline_dir / "manifest.jsonl")[0]
            summary = json.loads((pipeline_dir / "summary.json").read_text(encoding="utf-8"))
            errors = json.loads((pipeline_dir / "errors.json").read_text(encoding="utf-8"))

        self.assertEqual(row["status"], "incomplete")
        self.assertIsNone(row["final_output"])
        self.assertEqual(
            row["error"],
            {"stage_index": 2, "stage": "t2t", "type": "RuntimeError", "message": "text model failed"},
        )
        self.assertEqual(row["stages"][2]["status"], "missing")
        self.assertEqual(summary["errors"]["by_type"], {"RuntimeError": 1})
        self.assertEqual(summary["outputs"]["errors"], "errors.json")
        self.assertEqual(
            errors,
            {
                "schema_version": 1,
                "run_id": "20260616_1530",
                "error_count": 1,
                "errors": [
                    {
                        "sample_id": "sample-a",
                        "stage_index": 2,
                        "stage": "t2t",
                        "type": "RuntimeError",
                        "message": "text model failed",
                    }
                ],
            },
        )

    def test_s2a_uses_speech_input_and_t2a_final_output(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            s2t = _write_stage_run(
                root / "s2t",
                stage="s2t",
                model="asr",
                direction="s2a",
                rows=[_row(0, "sample-a", stage="s2t", model="asr", input_value={"speech_file": "speech.wav"})],
            )
            t2t = _write_stage_run(
                root / "t2t",
                stage="t2t",
                model="text",
                direction="s2a",
                rows=[_row(0, "sample-a", stage="t2t", model="text")],
            )
            t2a = _write_stage_run(
                root / "t2a",
                stage="t2a",
                model="audiogen",
                direction="s2a",
                rows=[
                    _row(
                        0,
                        "sample-a",
                        stage="t2a",
                        model="audiogen",
                        output={"type": "audio", "audio_file": "outputs/sample-a.wav"},
                    )
                ],
            )

            assemble_pipeline_record(root / "pipeline", direction="s2a", stage_run_dirs=[s2t, t2t, t2a])
            row = _read_jsonl(root / "pipeline" / "manifest.jsonl")[0]
            inspect = json.loads((root / "pipeline" / "inspect.json").read_text(encoding="utf-8"))

        self.assertEqual(row["input"], {"speech_file": "speech.wav"})
        self.assertEqual(row["final_output"]["audio_file"], (t2a / "outputs" / "sample-a.wav").as_posix())
        self.assertEqual([stage["stage"] for stage in row["stages"]], ["s2t", "t2t", "t2a"])
        self.assertEqual(inspect["samples"][0]["reference"], {"speech_file": "speech.wav", "reference_audio_caption": "reference caption for sample-a"})

    def test_fails_if_pipeline_artifact_already_exists_without_rewriting(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            a2t = _write_stage_run(
                root / "a2t",
                stage="a2t",
                model="captioner",
                direction="a2s",
                rows=[_row(0, "sample-a", stage="a2t", model="captioner", input_value={"audio_file": "a.wav"})],
            )
            t2t = _write_stage_run(root / "t2t", stage="t2t", model="text", direction="a2s", rows=[])
            t2s = _write_stage_run(root / "t2s", stage="t2s", model="tts", direction="a2s", rows=[])
            pipeline_dir = root / "pipeline"
            pipeline_dir.mkdir()
            existing = pipeline_dir / "manifest.jsonl"
            existing.write_text("existing\n", encoding="utf-8")

            with self.assertRaisesRegex(FileExistsError, "pipeline artifact already exists"):
                assemble_pipeline_record(pipeline_dir, direction="a2s", stage_run_dirs=[a2t, t2t, t2s])

            self.assertEqual(existing.read_text(encoding="utf-8"), "existing\n")
            self.assertFalse((pipeline_dir / "run.json").exists())

    def test_pipeline_references_repo_owned_stage_paths_relative_to_project_root(self) -> None:
        repo_root = Path.cwd()
        with tempfile.TemporaryDirectory(dir=repo_root) as tmpdir:
            root = Path(tmpdir)
            a2t = _write_stage_run(
                root / "results" / "cascaded" / "stages" / "a2t" / "captioner" / "ttas" / "val" / "20260616_1000",
                stage="a2t",
                model="captioner",
                direction="a2s",
                rows=[_row(0, "sample-a", stage="a2t", model="captioner", input_value={"audio_file": "a.wav"})],
            )
            t2t = _write_stage_run(
                root / "results" / "cascaded" / "stages" / "t2t" / "a2s" / "text" / "from-a2t" / "captioner" / "ttas" / "val" / "20260616_1100",
                stage="t2t",
                model="text",
                direction="a2s",
                rows=[_row(0, "sample-a", stage="t2t", model="text")],
            )
            t2s = _write_stage_run(
                root / "results" / "cascaded" / "stages" / "t2s" / "tts" / "from-t2t" / "text" / "ttas" / "val" / "20260616_1200",
                stage="t2s",
                model="tts",
                direction="a2s",
                rows=[
                    _row(
                        0,
                        "sample-a",
                        stage="t2s",
                        model="tts",
                        output={"type": "speech", "speech_file": "outputs/sample-a.wav"},
                    )
                ],
            )

            pipeline_dir = root / "results" / "cascaded" / "pipelines" / "a2s" / "caption_text_tts" / "ttas" / "val" / "20260616_1300"
            assemble_pipeline_record(pipeline_dir, direction="a2s", stage_run_dirs=[a2t, t2t, t2s])
            run = json.loads((pipeline_dir / "run.json").read_text(encoding="utf-8"))
            pipeline = json.loads((pipeline_dir / "pipeline.json").read_text(encoding="utf-8"))
            row = _read_jsonl(pipeline_dir / "manifest.jsonl")[0]

        self.assertFalse(run["paths"]["result"].startswith("/"))
        self.assertFalse(pipeline["stages"][0]["run_path"].startswith("/"))
        self.assertFalse(pipeline["stages"][0]["manifest"].startswith("/"))
        self.assertFalse(run["lineage"]["stages"][0]["config"].startswith("/"))
        self.assertFalse(row["final_output"]["speech_file"].startswith("/"))


    def test_rejects_non_terminal_stage_run_status(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            a2t = _write_stage_run(
                root / "a2t",
                stage="a2t",
                model="captioner",
                direction="a2s",
                rows=[_row(0, "sample-a", stage="a2t", model="captioner", input_value={"audio_file": "a.wav"})],
            )
            t2t = _write_stage_run(root / "t2t", stage="t2t", model="text", direction="a2s", rows=[])
            t2s = _write_stage_run(root / "t2s", stage="t2s", model="tts", direction="a2s", rows=[])
            _update_run_json(t2t, status="running")

            with self.assertRaisesRegex(ValueError, "stage run status must be terminal"):
                assemble_pipeline_record(root / "pipeline", direction="a2s", stage_run_dirs=[a2t, t2t, t2s])

    def test_rejects_unsupported_stage_manifest_row_status(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            a2t = _write_stage_run(
                root / "a2t",
                stage="a2t",
                model="captioner",
                direction="a2s",
                rows=[_row(0, "sample-a", stage="a2t", model="captioner", input_value={"audio_file": "a.wav"})],
            )
            t2t = _write_stage_run(
                root / "t2t",
                stage="t2t",
                model="text",
                direction="a2s",
                rows=[_row(0, "sample-a", stage="t2t", model="text", status="skipped")],
            )
            t2s = _write_stage_run(root / "t2s", stage="t2s", model="tts", direction="a2s", rows=[])

            with self.assertRaisesRegex(ValueError, "unsupported row status"):
                assemble_pipeline_record(root / "pipeline", direction="a2s", stage_run_dirs=[a2t, t2t, t2s])

    def test_rejects_successful_final_row_with_missing_media_file(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            a2t = _write_stage_run(
                root / "a2t",
                stage="a2t",
                model="captioner",
                direction="a2s",
                rows=[_row(0, "sample-a", stage="a2t", model="captioner", input_value={"audio_file": "a.wav"})],
            )
            t2t = _write_stage_run(
                root / "t2t",
                stage="t2t",
                model="text",
                direction="a2s",
                rows=[_row(0, "sample-a", stage="t2t", model="text")],
            )
            t2s = _write_stage_run(
                root / "t2s",
                stage="t2s",
                model="tts",
                direction="a2s",
                rows=[
                    _row(
                        0,
                        "sample-a",
                        stage="t2s",
                        model="tts",
                        output={"type": "speech", "speech_file": "outputs/sample-a.wav"},
                    )
                ],
            )
            (t2s / "outputs" / "sample-a.wav").unlink()

            with self.assertRaisesRegex(FileNotFoundError, "missing final output file"):
                assemble_pipeline_record(root / "pipeline", direction="a2s", stage_run_dirs=[a2t, t2t, t2s])

    def test_pipeline_inspect_merges_reference_keys_from_stage_inspects(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            a2t = _write_stage_run(
                root / "a2t",
                stage="a2t",
                model="captioner",
                direction="a2s",
                rows=[_row(0, "sample-a", stage="a2t", model="captioner", input_value={"audio_file": "a-input.wav"})],
            )
            t2t = _write_stage_run(
                root / "t2t",
                stage="t2t",
                model="text",
                direction="a2s",
                rows=[_row(0, "sample-a", stage="t2t", model="text")],
            )
            t2s = _write_stage_run(
                root / "t2s",
                stage="t2s",
                model="tts",
                direction="a2s",
                rows=[_row(0, "sample-a", stage="t2s", model="tts", output={"type": "speech", "speech_file": "outputs/sample-a.wav"})],
            )
            a2t_inspect = json.loads((a2t / "inspect.json").read_text(encoding="utf-8"))
            a2t_inspect["samples"][0]["reference"] = {"audio_file": "a-input.wav"}
            (a2t / "inspect.json").write_text(json.dumps(a2t_inspect) + "\n", encoding="utf-8")
            t2t_inspect = json.loads((t2t / "inspect.json").read_text(encoding="utf-8"))
            t2t_inspect["samples"][0].pop("reference")
            (t2t / "inspect.json").write_text(json.dumps(t2t_inspect) + "\n", encoding="utf-8")
            t2s_inspect = json.loads((t2s / "inspect.json").read_text(encoding="utf-8"))
            t2s_inspect["samples"][0]["reference"] = {"reference_speech_text": "spoken reference"}
            (t2s / "inspect.json").write_text(json.dumps(t2s_inspect) + "\n", encoding="utf-8")

            assemble_pipeline_record(root / "pipeline", direction="a2s", stage_run_dirs=[a2t, t2t, t2s])
            inspect = json.loads((root / "pipeline" / "inspect.json").read_text(encoding="utf-8"))

        self.assertEqual(inspect["samples"][0]["reference"], {"audio_file": "a-input.wav", "reference_speech_text": "spoken reference"})

    def test_pipeline_inspect_rejects_conflicting_stage_reference(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            a2t = _write_stage_run(
                root / "a2t",
                stage="a2t",
                model="captioner",
                direction="a2s",
                rows=[_row(0, "sample-a", stage="a2t", model="captioner", input_value={"audio_file": "a-input.wav"})],
            )
            t2t = _write_stage_run(
                root / "t2t",
                stage="t2t",
                model="text",
                direction="a2s",
                rows=[_row(0, "sample-a", stage="t2t", model="text")],
            )
            t2s = _write_stage_run(
                root / "t2s",
                stage="t2s",
                model="tts",
                direction="a2s",
                rows=[_row(0, "sample-a", stage="t2s", model="tts", output={"type": "speech", "speech_file": "outputs/sample-a.wav"})],
            )
            inspect = json.loads((t2t / "inspect.json").read_text(encoding="utf-8"))
            inspect["samples"][0]["reference"]["reference_audio_caption"] = "conflicting caption"
            (t2t / "inspect.json").write_text(json.dumps(inspect) + "\n", encoding="utf-8")

            with self.assertRaisesRegex(ValueError, "conflicting reference"):
                assemble_pipeline_record(root / "pipeline", direction="a2s", stage_run_dirs=[a2t, t2t, t2s])

    def test_rejects_downstream_source_sample_mismatch(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            a2t = _write_stage_run(
                root / "a2t",
                stage="a2t",
                model="captioner",
                direction="a2s",
                rows=[_row(0, "sample-a", stage="a2t", model="captioner", input_value={"audio_file": "a.wav"})],
            )
            t2t = _write_stage_run(
                root / "t2t",
                stage="t2t",
                model="text",
                direction="a2s",
                rows=[
                    _row(
                        0,
                        "sample-a",
                        stage="t2t",
                        model="text",
                        input_value={"source_stage_index": 1, "source_sample_id": "other", "source_status": "success"},
                    )
                ],
            )
            t2s = _write_stage_run(root / "t2s", stage="t2s", model="tts", direction="a2s", rows=[])

            with self.assertRaisesRegex(ValueError, "source_sample_id must match"):
                assemble_pipeline_record(root / "pipeline", direction="a2s", stage_run_dirs=[a2t, t2t, t2s])

    def test_rejects_downstream_source_stage_index_mismatch(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            a2t = _write_stage_run(
                root / "a2t",
                stage="a2t",
                model="captioner",
                direction="a2s",
                rows=[_row(0, "sample-a", stage="a2t", model="captioner", input_value={"audio_file": "a.wav"})],
            )
            t2t = _write_stage_run(
                root / "t2t",
                stage="t2t",
                model="text",
                direction="a2s",
                rows=[_row(0, "sample-a", stage="t2t", model="text")],
            )
            t2s = _write_stage_run(
                root / "t2s",
                stage="t2s",
                model="tts",
                direction="a2s",
                rows=[
                    _row(
                        0,
                        "sample-a",
                        stage="t2s",
                        model="tts",
                        input_value={"source_stage_index": 1, "source_sample_id": "sample-a", "source_status": "success"},
                        output={"type": "speech", "speech_file": "outputs/sample-a.wav"},
                    )
                ],
            )

            with self.assertRaisesRegex(ValueError, "source_stage_index must be 2"):
                assemble_pipeline_record(root / "pipeline", direction="a2s", stage_run_dirs=[a2t, t2t, t2s])

    def test_rejects_successful_downstream_row_after_failed_upstream(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            a2t = _write_stage_run(
                root / "a2t",
                stage="a2t",
                model="captioner",
                direction="a2s",
                rows=[_row(0, "sample-a", stage="a2t", model="captioner", input_value={"audio_file": "a.wav"})],
            )
            t2t = _write_stage_run(
                root / "t2t",
                stage="t2t",
                model="text",
                direction="a2s",
                rows=[
                    _row(
                        0,
                        "sample-a",
                        stage="t2t",
                        model="text",
                        status="error",
                        error={"type": "RuntimeError", "message": "text model failed"},
                    )
                ],
            )
            t2s = _write_stage_run(
                root / "t2s",
                stage="t2s",
                model="tts",
                direction="a2s",
                rows=[
                    _row(
                        0,
                        "sample-a",
                        stage="t2s",
                        model="tts",
                        input_value={"source_stage_index": 2, "source_sample_id": "sample-a", "source_status": "error"},
                        output={"type": "speech", "speech_file": "outputs/sample-a.wav"},
                    )
                ],
            )

            with self.assertRaisesRegex(ValueError, "cannot be successful"):
                assemble_pipeline_record(root / "pipeline", direction="a2s", stage_run_dirs=[a2t, t2t, t2s])


def _write_stage_run(
    run_dir: Path,
    *,
    stage: str,
    model: str,
    direction: str,
    rows: list[dict],
) -> Path:
    run_dir.mkdir(parents=True)
    (run_dir / "run.json").write_text(
        json.dumps(
            {
                "schema_version": 1,
                "run_id": run_dir.name,
                "stage": stage,
                "direction": direction,
                "status": "complete",
                "dataset": "ttas",
                "split": "val",
                "model": model,
            }
        )
        + "\n",
        encoding="utf-8",
    )
    _write_referenced_media(run_dir, rows)
    (run_dir / "manifest.jsonl").write_text(
        "".join(json.dumps(row) + "\n" for row in rows),
        encoding="utf-8",
    )
    (run_dir / "config.json").write_text(
        json.dumps(
            {
                "schema_version": 1,
                "input": {"manifest": "input.jsonl"},
                "model": {"name": model},
                "generation": {"seed": 1337},
            }
        )
        + "\n",
        encoding="utf-8",
    )
    (run_dir / "inspect.json").write_text(
        json.dumps(
            {
                "schema_version": 1,
                "run_id": run_dir.name,
                "samples": [_inspect_sample_fixture(row, stage) for row in rows],
            }
        )
        + "\n",
        encoding="utf-8",
    )
    return run_dir


def _inspect_sample_fixture(row: dict, stage: str) -> dict:
    input_value = row.get("input") if isinstance(row.get("input"), dict) else {}
    reference = {"reference_audio_caption": f"reference caption for {row.get('sample_id')}"}
    if isinstance(input_value.get("audio_file"), str):
        reference["audio_file"] = input_value["audio_file"]
    if isinstance(input_value.get("speech_file"), str):
        reference["speech_file"] = input_value["speech_file"]
    return {
        "sample_id": row.get("sample_id"),
        "reference": reference,
        "stages": [{"stage": stage, "runtime_prompt": None}],
    }


def _write_referenced_media(run_dir: Path, rows: list[dict]) -> None:
    for row in rows:
        if row.get("status") != "success":
            continue
        output = row.get("output")
        if not isinstance(output, dict):
            continue
        for key in ("audio_file", "speech_file"):
            value = output.get(key)
            if not isinstance(value, str) or not value:
                continue
            path = Path(value)
            if not path.is_absolute():
                path = run_dir / path
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_bytes(b"RIFFfake")


def _update_run_json(run_dir: Path, **updates: object) -> None:
    path = run_dir / "run.json"
    document = json.loads(path.read_text(encoding="utf-8"))
    document.update(updates)
    path.write_text(json.dumps(document) + "\n", encoding="utf-8")


def _row(
    row_index: int,
    sample_id: str,
    *,
    stage: str,
    model: str,
    status: str = "success",
    input_value: dict | None = None,
    output: dict | None = None,
    error: dict | None = None,
) -> dict:
    if input_value is None:
        source_stage_index = 2 if stage in {"t2a", "t2s"} else 1
        input_value = {
            "source_stage_index": source_stage_index,
            "source_sample_id": sample_id,
            "source_status": "success",
        }
    return {
        "schema_version": 1,
        "row_index": row_index,
        "sample_id": sample_id,
        "stage": stage,
        "model": model,
        "status": status,
        "input": input_value,
        "output": output if status == "success" else None,
        "timing": {"inference_seconds": 1.0},
        "error": error,
    }


def _read_jsonl(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line]


if __name__ == "__main__":
    unittest.main()
