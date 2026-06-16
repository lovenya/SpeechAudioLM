from __future__ import annotations

import json
import tempfile
import unittest
from datetime import datetime, timezone
from pathlib import Path

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
            pipeline_dir = root / "pipeline" / "20260616_1530"

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
            pipeline = json.loads((pipeline_dir / "pipeline.json").read_text(encoding="utf-8"))

        self.assertEqual(report["summary"]["samples"]["successful"], 2)
        self.assertEqual([row["sample_id"] for row in rows], ["sample-b", "sample-a"])
        self.assertEqual(rows[0]["input"], {"audio_file": "b-input.wav"})
        self.assertEqual(rows[0]["final_output"]["speech_file"], (t2s / "outputs" / "sample-b.wav").as_posix())
        self.assertEqual([stage["stage"] for stage in rows[0]["stages"]], ["a2t", "t2t", "t2s"])
        self.assertEqual(pipeline["final_output"]["stage"], "t2s")
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

        self.assertEqual(row["status"], "incomplete")
        self.assertIsNone(row["final_output"])
        self.assertEqual(
            row["error"],
            {"stage_index": 2, "stage": "t2t", "type": "RuntimeError", "message": "text model failed"},
        )
        self.assertEqual(row["stages"][2]["status"], "missing")
        self.assertEqual(summary["errors"]["by_type"], {"RuntimeError": 1})

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

        self.assertEqual(row["input"], {"speech_file": "speech.wav"})
        self.assertEqual(row["final_output"]["audio_file"], (t2a / "outputs" / "sample-a.wav").as_posix())
        self.assertEqual([stage["stage"] for stage in row["stages"]], ["s2t", "t2t", "t2a"])

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
    (run_dir / "manifest.jsonl").write_text(
        "".join(json.dumps(row) + "\n" for row in rows),
        encoding="utf-8",
    )
    return run_dir


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
    return {
        "schema_version": 1,
        "row_index": row_index,
        "sample_id": sample_id,
        "stage": stage,
        "model": model,
        "status": status,
        "input": input_value
        or {"source_stage_index": 1, "source_sample_id": sample_id, "source_status": "success"},
        "output": output if status == "success" else None,
        "timing": {"inference_seconds": 1.0},
        "error": error,
    }


def _read_jsonl(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line]


if __name__ == "__main__":
    unittest.main()
