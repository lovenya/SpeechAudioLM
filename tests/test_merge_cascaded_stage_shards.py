from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from scripts.merge_cascaded_stage_shards import merge_stage_shards


class MergeCascadedStageShardsTest(unittest.TestCase):
    def test_merges_stage_shards_and_deletes_shard_directory(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            run_dir = Path(tmpdir) / "20260616_1437"
            _write_shard(
                run_dir,
                "shard_000",
                [
                    _row(1, "sample-b", status="error", output=None, error={"type": "RuntimeError", "message": "boom"}),
                    _row(0, "sample-a", output={"type": "text", "generated_speech_text": "hello"}),
                ],
            )
            _write_shard(
                run_dir,
                "shard_001",
                [_row(2, "sample-c", output={"type": "text", "generated_speech_text": "there"})],
            )

            report = merge_stage_shards(run_dir, expected_count=3)

            rows = [json.loads(line) for line in (run_dir / "manifest.jsonl").read_text().splitlines()]
            errors = json.loads((run_dir / "errors.json").read_text())
            summary = json.loads((run_dir / "summary.json").read_text())
            run_doc = json.loads((run_dir / "run.json").read_text())

        self.assertEqual([row["row_index"] for row in rows], [0, 1, 2])
        self.assertFalse((run_dir / "shards").exists())
        self.assertEqual(report["merged"]["successful"], 2)
        self.assertEqual(report["merged"]["failed"], 1)
        self.assertEqual(errors["error_count"], 1)
        self.assertEqual(summary["outputs"]["generated_media_files"], None)
        self.assertEqual(run_doc["status"], "complete_with_errors")
        self.assertIsInstance(run_doc.get("completed_at"), str)

    def test_merges_generated_media_into_outputs_and_rewrites_manifest(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            run_dir = Path(tmpdir) / "20260616_1500"
            shard = run_dir / "shards" / "shard_000"
            media = shard / "outputs" / "000000_sample-a.wav"
            media.parent.mkdir(parents=True)
            media.write_bytes(b"RIFFfake")
            _write_shard(
                run_dir,
                "shard_000",
                [
                    _row(
                        0,
                        "sample-a",
                        stage="t2s",
                        input_value={"source_stage_index": 2, "source_sample_id": "sample-a", "source_status": "success"},
                        output={
                            "type": "speech",
                            "speech_file": "outputs/000000_sample-a.wav",
                            "sample_rate": 24000,
                            "duration_seconds": 1.0,
                        },
                    )
                ],
            )

            report = merge_stage_shards(run_dir, expected_count=1, media_output="speech")
            row = json.loads((run_dir / "manifest.jsonl").read_text())
            promoted_audio = (run_dir / "outputs" / "000000_sample-a.wav").read_bytes()

        self.assertEqual(row["output"]["speech_file"], "outputs/000000_sample-a.wav")
        self.assertEqual(promoted_audio, b"RIFFfake")
        self.assertEqual(report["outputs"]["media_files_copied"], 1)

    def test_allows_subset_runs_with_nonzero_row_indices(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            run_dir = Path(tmpdir) / "20260616_1600"
            _write_shard(
                run_dir,
                "shard_000",
                [
                    _row(6, "sample-b", output={"type": "text", "generated_speech_text": "second"}),
                    _row(5, "sample-a", output={"type": "text", "generated_speech_text": "first"}),
                ],
            )

            report = merge_stage_shards(run_dir, expected_count=2)
            rows = [json.loads(line) for line in (run_dir / "manifest.jsonl").read_text().splitlines()]

        self.assertEqual([row["row_index"] for row in rows], [5, 6])
        self.assertEqual(report["merged"]["rows_written"], 2)
        self.assertEqual(report["merged"]["requested"], 2)

    def test_fails_when_merged_row_count_does_not_match_expected_count(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            run_dir = Path(tmpdir) / "20260616_1600"
            _write_shard(run_dir, "shard_000", [_row(5, "sample-a")])

            with self.assertRaisesRegex(ValueError, "merged rows=1; expected=2"):
                merge_stage_shards(run_dir, expected_count=2)

            self.assertFalse((run_dir / "manifest.jsonl").exists())
            self.assertTrue((run_dir / "shards").is_dir())

    def test_fails_for_duplicate_sample_id_without_publishing(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            run_dir = Path(tmpdir) / "20260616_1437"
            _write_shard(run_dir, "shard_000", [_row(0, "same")])
            _write_shard(run_dir, "shard_001", [_row(1, "same")])

            with self.assertRaisesRegex(ValueError, "duplicate sample_id"):
                merge_stage_shards(run_dir, expected_count=2)

            self.assertFalse((run_dir / "manifest.jsonl").exists())
            self.assertTrue((run_dir / "shards").is_dir())

    def test_fails_if_canonical_manifest_already_exists(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            run_dir = Path(tmpdir) / "20260616_1437"
            run_dir.mkdir()
            (run_dir / "manifest.jsonl").write_text("{}\n")
            _write_shard(run_dir, "shard_000", [_row(0, "sample-a")])

            with self.assertRaisesRegex(FileExistsError, "canonical artifact already exists"):
                merge_stage_shards(run_dir, expected_count=1)

    def test_fails_for_mismatched_shard_inspect_direction_without_publishing(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            run_dir = Path(tmpdir) / "20260616_1437"
            _write_shard(run_dir, "shard_000", [_row(0, "sample-a")])
            _write_shard(run_dir, "shard_001", [_row(1, "sample-b")], inspect_updates={"direction": "s2a"})

            with self.assertRaisesRegex(ValueError, "inspect.json direction does not match"):
                merge_stage_shards(run_dir, expected_count=2)

            self.assertFalse((run_dir / "manifest.jsonl").exists())
            self.assertTrue((run_dir / "shards").is_dir())

    def test_fails_for_mismatched_shard_inspect_stage_without_publishing(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            run_dir = Path(tmpdir) / "20260616_1437"
            _write_shard(run_dir, "shard_000", [_row(0, "sample-a")])
            _write_shard(run_dir, "shard_001", [_row(1, "sample-b")], inspect_updates={"stage": "t2s"})

            with self.assertRaisesRegex(ValueError, "inspect.json stage does not match"):
                merge_stage_shards(run_dir, expected_count=2)

            self.assertFalse((run_dir / "manifest.jsonl").exists())
            self.assertTrue((run_dir / "shards").is_dir())

    def test_fails_for_mismatched_shard_inspect_lineage_without_publishing(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            run_dir = Path(tmpdir) / "20260616_1437"
            _write_shard(run_dir, "shard_000", [_row(0, "sample-a")])
            _write_shard(
                run_dir,
                "shard_001",
                [_row(1, "sample-b")],
                inspect_updates={"lineage": {"direction": "a2s", "current_stage_index": 3, "stages": []}},
            )

            with self.assertRaisesRegex(ValueError, "inspect.json lineage does not match"):
                merge_stage_shards(run_dir, expected_count=2)

            self.assertFalse((run_dir / "manifest.jsonl").exists())
            self.assertTrue((run_dir / "shards").is_dir())

    def test_fails_for_mismatched_shard_stage_configs_without_publishing(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            run_dir = Path(tmpdir) / "20260616_1437"
            _write_shard(run_dir, "shard_000", [_row(0, "sample-a")])
            _write_shard(
                run_dir,
                "shard_001",
                [_row(1, "sample-b")],
                inspect_updates={"stage_configs": [{"stage": "t2t", "model": {"name": "other"}}]},
            )

            with self.assertRaisesRegex(ValueError, "inspect.json stage_configs does not match"):
                merge_stage_shards(run_dir, expected_count=2)

            self.assertFalse((run_dir / "manifest.jsonl").exists())
            self.assertTrue((run_dir / "shards").is_dir())

    def test_fails_when_shard_inspect_is_missing_without_publishing(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            run_dir = Path(tmpdir) / "20260616_1437"
            _write_shard(run_dir, "shard_000", [_row(0, "sample-a")])
            (run_dir / "shards" / "shard_000" / "inspect.json").unlink()

            with self.assertRaisesRegex(FileNotFoundError, "missing shard inspect file"):
                merge_stage_shards(run_dir, expected_count=1)

            self.assertFalse((run_dir / "manifest.jsonl").exists())
            self.assertTrue((run_dir / "shards").is_dir())

    def test_fails_for_shard_inspect_root_identity_mismatch_without_publishing(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            run_dir = Path(tmpdir) / "20260616_1437"
            _write_shard(run_dir, "shard_000", [_row(0, "sample-a")], inspect_updates={"dataset": "other"})

            with self.assertRaisesRegex(ValueError, "inspect.json dataset does not match root run.json"):
                merge_stage_shards(run_dir, expected_count=1)

            self.assertFalse((run_dir / "manifest.jsonl").exists())
            self.assertTrue((run_dir / "shards").is_dir())

    def test_fails_for_shard_inspect_sample_coverage_mismatch_without_publishing(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            run_dir = Path(tmpdir) / "20260616_1437"
            _write_shard(
                run_dir,
                "shard_000",
                [_row(0, "sample-a")],
                inspect_updates={"samples": [{"sample_id": "other", "status": "success"}]},
            )

            with self.assertRaisesRegex(ValueError, "inspect.json sample IDs do not match"):
                merge_stage_shards(run_dir, expected_count=1)

            self.assertFalse((run_dir / "manifest.jsonl").exists())
            self.assertTrue((run_dir / "shards").is_dir())

    def test_fails_if_root_run_artifacts_are_missing(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            run_dir = Path(tmpdir) / "20260616_1437"
            _write_shard(run_dir, "shard_000", [_row(0, "sample-a")], write_root_artifacts=False)

            with self.assertRaisesRegex(FileNotFoundError, "missing required stage root artifact"):
                merge_stage_shards(run_dir, expected_count=1)

            self.assertFalse((run_dir / "manifest.jsonl").exists())


def _write_shard(
    run_dir: Path,
    shard_id: str,
    rows: list[dict],
    *,
    write_root_artifacts: bool = True,
    inspect_updates: dict | None = None,
) -> None:
    if write_root_artifacts:
        _write_root_artifacts(run_dir, rows)
    shard = run_dir / "shards" / shard_id
    shard.mkdir(parents=True, exist_ok=True)
    (shard / "manifest.jsonl").write_text(
        "".join(json.dumps(row) + "\n" for row in rows),
        encoding="utf-8",
    )
    inspect = {
        "schema_version": 1,
        "run_id": run_dir.name,
        "stage": rows[0].get("stage", "t2t"),
        "direction": "a2s",
        "dataset": "ttas",
        "split": "val",
        "status": "complete",
        "lineage": {"direction": "a2s", "current_stage_index": 2, "stages": []},
        "stage_configs": [],
        "summary": {"requested": len(rows), "successful": len(rows), "failed": 0},
        "samples": [{"sample_id": row["sample_id"], "status": row["status"]} for row in rows],
    }
    if inspect_updates:
        inspect.update(inspect_updates)
    (shard / "inspect.json").write_text(
        json.dumps(inspect) + "\n",
        encoding="utf-8",
    )


def _write_root_artifacts(run_dir: Path, rows: list[dict]) -> None:
    run_dir.mkdir(parents=True, exist_ok=True)
    stage = rows[0].get("stage", "t2t")
    model = rows[0].get("model", "llama3_3b_instruct")
    if not (run_dir / "run.json").exists():
        (run_dir / "run.json").write_text(
            json.dumps(
                {
                    "schema_version": 1,
                    "run_id": run_dir.name,
                    "stage": stage,
                    "direction": "a2s",
                    "status": "running",
                    "dataset": "ttas",
                    "split": "val",
                    "model": model,
                }
            )
            + "\n",
            encoding="utf-8",
        )
    if not (run_dir / "config.json").exists():
        (run_dir / "config.json").write_text(
            json.dumps(
                {
                    "schema_version": 1,
                    "input": {"manifest": "input.jsonl"},
                    "model": {"name": model},
                    "execution": {"shard_index": None, "num_shards": None},
                }
            )
            + "\n",
            encoding="utf-8",
        )


def _row(
    row_index: int,
    sample_id: str,
    *,
    stage: str = "t2t",
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
        "model": "llama3_3b_instruct",
        "status": status,
        "input": input_value
        or {"source_stage_index": 1, "source_sample_id": sample_id, "source_status": "success"},
        "output": output if status == "success" else None,
        "timing": {"inference_seconds": 1.0},
        "error": error,
    }


if __name__ == "__main__":
    unittest.main()
