from __future__ import annotations

import json
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

from evaluation.artifacts import ArtifactAlreadyExistsError
from evaluation.common_ids import (
    CommonIdError,
    derive_common_attempted_ids,
    derive_common_successful_ids,
    write_common_attempted_ids,
    write_common_successful_ids,
)


def _write_manifest(path: Path, rows: list[tuple[str, str]]) -> None:
    path.write_text(
        "".join(
            json.dumps({"sample_id": sample_id, "status": status}) + "\n"
            for sample_id, status in rows
        ),
        encoding="utf-8",
    )


class CommonIdTest(unittest.TestCase):
    def test_intersects_successful_ids_in_first_manifest_order(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            first, second = root / "first.jsonl", root / "second.jsonl"
            _write_manifest(first, [("c", "success"), ("a", "success"), ("b", "error")])
            _write_manifest(second, [("a", "success"), ("b", "success"), ("c", "success")])

            result = derive_common_successful_ids([first, second])

            self.assertEqual(result["common_sample_ids"], ["c", "a"])
            self.assertEqual(result["common_sample_count"], 2)
            self.assertEqual(result["policy"], "intersection_of_successful_generation_rows")
            self.assertEqual(len(result["source_manifests"][0]["manifest_sha256"]), 64)

    def test_rejects_duplicate_ids_and_empty_intersection(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            duplicate = root / "duplicate.jsonl"
            _write_manifest(duplicate, [("a", "success"), ("a", "success")])
            with self.assertRaisesRegex(CommonIdError, "duplicate"):
                derive_common_successful_ids([duplicate])

            first, second = root / "first.jsonl", root / "second.jsonl"
            _write_manifest(first, [("a", "success")])
            _write_manifest(second, [("b", "success")])
            with self.assertRaisesRegex(CommonIdError, "no common successful"):
                derive_common_successful_ids([first, second])

    def test_final_scope_retains_errors_and_requires_identical_attempted_ids(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            first, second = root / "first.jsonl", root / "second.jsonl"
            _write_manifest(first, [("c", "success"), ("a", "error"), ("b", "success")])
            _write_manifest(second, [("a", "success"), ("b", "error"), ("c", "success")])

            result = derive_common_attempted_ids([first, second])

            self.assertEqual(result["common_sample_ids"], ["c", "a", "b"])
            self.assertEqual(result["policy"], "identical_attempted_generation_rows")
            self.assertEqual(result["source_manifests"][0]["error_row_count"], 1)
            output = root / "final_scope.json"
            write_common_attempted_ids([first, second], output)
            self.assertTrue(output.is_file())

            mismatch = root / "mismatch.jsonl"
            _write_manifest(mismatch, [("a", "success"), ("other", "success")])
            with self.assertRaisesRegex(CommonIdError, "identical attempted"):
                derive_common_attempted_ids([first, mismatch])

    def test_final_scope_cli_runs_from_repository_root(self) -> None:
        completed = subprocess.run(
            [sys.executable, "scripts/build_final_scope_ids.py", "--help"],
            cwd=Path(__file__).resolve().parents[1],
            text=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
        )
        self.assertEqual(completed.returncode, 0, completed.stderr)
        self.assertIn("--manifest", completed.stdout)

    def test_output_is_create_only(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            manifest, output = root / "manifest.jsonl", root / "ids.json"
            _write_manifest(manifest, [("a", "success")])
            write_common_successful_ids([manifest], output)
            original = output.read_bytes()
            with self.assertRaises(ArtifactAlreadyExistsError):
                write_common_successful_ids([manifest], output)
            self.assertEqual(output.read_bytes(), original)


if __name__ == "__main__":
    unittest.main()
