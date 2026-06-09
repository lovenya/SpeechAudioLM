from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from scripts.aggregate_evaluation_shards import aggregate, discover_per_sample_files, main


def write_jsonl(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8")


def read_jsonl(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()]


class AggregateEvaluationShardsTest(unittest.TestCase):
    def test_merges_sorted_rows_preserves_errors_and_summarizes_metrics(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            first = root / "shard-1" / "T2T" / "ttas" / "cascaded"
            second = root / "shard-2" / "T2T" / "ttas" / "cascaded"
            write_jsonl(
                first / "latest_per_sample_val.jsonl",
                [
                    {"row_index": 2, "status": "success", "metrics": {"g_eval_reference_free": 5.0}, "errors": {}},
                    {"row_index": 0, "status": "success", "metrics": {"g_eval_reference_free": 3.0}, "errors": {}},
                ],
            )
            write_jsonl(
                second / "latest_per_sample_val.jsonl",
                [
                    {
                        "row_index": 1,
                        "status": "error",
                        "metrics": {},
                        "errors": {"g_eval_reference_free": {"type": "judge_runtime_error"}},
                    }
                ],
            )

            paths = aggregate([root], root / "merged", expected_count=3)
            rows = read_jsonl(paths["per_sample"])
            summary = json.loads(paths["summary"].read_text(encoding="utf-8"))
            report = json.loads(paths["merge_report"].read_text(encoding="utf-8"))

        self.assertEqual([row["row_index"] for row in rows], [0, 1, 2])
        self.assertEqual(rows[1]["errors"]["g_eval_reference_free"]["type"], "judge_runtime_error")
        self.assertEqual(summary["metric_summaries"]["g_eval_reference_free"]["count"], 2)
        self.assertEqual(summary["metric_summaries"]["g_eval_reference_free"]["mean"], 4.0)
        self.assertEqual(summary["successful_rows"], 2)
        self.assertEqual(summary["skipped_or_error_rows"], 1)
        self.assertEqual(report["status"], "complete")

    def test_rejects_duplicates_and_writes_report(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            one = root / "one.jsonl"
            two = root / "two.jsonl"
            write_jsonl(one, [{"row_index": 0, "status": "success", "metrics": {}, "errors": {}}])
            write_jsonl(two, [{"row_index": 0, "status": "success", "metrics": {}, "errors": {}}])
            output = root / "merged"

            with self.assertRaisesRegex(ValueError, "duplicate_row_indices"):
                aggregate([one, two], output, expected_count=1)

            report = json.loads((output / "merge_report.json").read_text(encoding="utf-8"))
            self.assertEqual(report["status"], "invalid")
            self.assertEqual(report["duplicate_row_indices"], [0])
            self.assertFalse((output / "per_sample.jsonl").exists())

    def test_rejects_missing_expected_rows_unless_incomplete_is_allowed(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            shard = root / "latest_per_sample_val.jsonl"
            write_jsonl(
                shard,
                [
                    {"row_index": 0, "status": "success", "metrics": {}, "errors": {}},
                    {"row_index": 2, "status": "success", "metrics": {}, "errors": {}},
                ],
            )
            strict_output = root / "strict"
            with self.assertRaisesRegex(ValueError, "missing_expected_row_indices"):
                aggregate([shard], strict_output, expected_count=3)
            strict_report = json.loads((strict_output / "merge_report.json").read_text(encoding="utf-8"))
            self.assertEqual(strict_report["missing_expected_ranges"], [[1, 1]])

            allowed_paths = aggregate([shard], root / "allowed", expected_count=3, allow_incomplete=True)
            allowed_report = json.loads(allowed_paths["merge_report"].read_text(encoding="utf-8"))

        self.assertEqual(allowed_report["status"], "incomplete_allowed")
        self.assertEqual(allowed_report["missing_expected_row_indices"], [1])

    def test_rejects_rows_outside_expected_range(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            shard = root / "latest_per_sample_val.jsonl"
            write_jsonl(
                shard,
                [
                    {"row_index": 0, "status": "success", "metrics": {}, "errors": {}},
                    {"row_index": 1, "status": "success", "metrics": {}, "errors": {}},
                ],
            )

            with self.assertRaisesRegex(ValueError, "unexpected_row_indices"):
                aggregate([shard], root / "merged", expected_count=1)

    def test_rejects_internal_gaps_without_an_explicit_expected_range(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            shard = root / "latest_per_sample_val.jsonl"
            write_jsonl(
                shard,
                [
                    {"row_index": 4, "status": "success", "metrics": {}, "errors": {}},
                    {"row_index": 6, "status": "success", "metrics": {}, "errors": {}},
                ],
            )

            with self.assertRaisesRegex(ValueError, "missing_internal_row_indices"):
                aggregate([shard], root / "merged")

            report = json.loads((root / "merged" / "merge_report.json").read_text(encoding="utf-8"))

        self.assertEqual(report["internal_gap_indices"], [5])
        self.assertEqual(report["internal_gap_ranges"], [[5, 5]])

    def test_preserves_corpus_shard_summaries_without_invalid_averaging(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            for index, value in enumerate((1.2, 1.8)):
                shard = root / f"shard-{index}"
                write_jsonl(
                    shard / "latest_per_sample_val.jsonl",
                    [{"row_index": index, "status": "success", "metrics": {}, "errors": {}}],
                )
                (shard / "latest_summary_val.json").write_text(
                    json.dumps(
                        {
                            "model": "cascaded",
                            "task": "T2A",
                            "dataset": "ttas",
                            "split": "val",
                            "metric_summaries": {
                                "fad": {"level": "corpus", "count": 1, "mean": value}
                            },
                        }
                    ),
                    encoding="utf-8",
                )

            paths = aggregate([root], root / "merged", expected_count=2)
            summary = json.loads(paths["summary"].read_text(encoding="utf-8"))

        fad = summary["corpus_metric_summaries"]["fad"]
        self.assertEqual(fad["status"], "requires_full_corpus_recomputation")
        self.assertEqual(len(fad["shards"]), 2)
        self.assertNotIn("mean", fad)
        self.assertEqual(summary["metric_summaries"]["fad"], fad)

    def test_discovers_latest_files_and_cli_returns_nonzero_for_incomplete_merge(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            latest = root / "shard" / "latest_per_sample_val.jsonl"
            write_jsonl(latest, [{"row_index": 1, "status": "success", "metrics": {}, "errors": {}}])

            self.assertEqual(discover_per_sample_files([root]), [latest.resolve()])
            exit_code = main([str(root), "--output-dir", str(root / "merged"), "--expected-count", "2"])

        self.assertEqual(exit_code, 1)


if __name__ == "__main__":
    unittest.main()
