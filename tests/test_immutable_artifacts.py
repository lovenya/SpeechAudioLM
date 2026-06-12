from __future__ import annotations

import math
import multiprocessing
import tempfile
import unittest
from pathlib import Path

from evaluation.artifacts import (
    ArtifactAlreadyExistsError,
    write_bytes_create_only,
    write_json_create_only,
    write_jsonl_create_only,
    write_text_create_only,
)


def _race_writer(path: str, content: bytes, start, results) -> None:
    start.wait()
    try:
        write_bytes_create_only(Path(path), content)
    except ArtifactAlreadyExistsError:
        results.put(("exists", content))
    else:
        results.put(("created", content))


class ImmutableArtifactTest(unittest.TestCase):
    def assert_no_temporary_files(self, root: Path) -> None:
        self.assertEqual(
            [path for path in root.iterdir() if path.name.endswith(".tmp")],
            [],
        )

    def test_bytes_and_text_refuse_to_overwrite(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            path = root / "artifact.txt"
            write_text_create_only(path, "first")

            with self.assertRaises(ArtifactAlreadyExistsError):
                write_bytes_create_only(path, b"second")

            self.assertEqual(path.read_bytes(), b"first")
            self.assert_no_temporary_files(root)

    def test_json_is_sorted_deterministic_and_rejects_non_finite_numbers(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            first = root / "first.json"
            second = root / "second.json"
            value = {"z": "caf\u00e9", "a": {"y": 2, "x": 1}}

            write_json_create_only(first, value)
            write_json_create_only(second, value)

            expected = (
                '{\n'
                '  "a": {\n'
                '    "x": 1,\n'
                '    "y": 2\n'
                '  },\n'
                '  "z": "caf\u00e9"\n'
                '}\n'
            )
            self.assertEqual(first.read_text(encoding="utf-8"), expected)
            self.assertEqual(first.read_bytes(), second.read_bytes())

            failed = root / "failed.json"
            with self.assertRaises(ValueError):
                write_json_create_only(failed, {"value": math.inf})
            self.assertFalse(failed.exists())
            self.assert_no_temporary_files(root)

    def test_jsonl_validates_rows_and_serializes_deterministically(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            path = root / "rows.jsonl"
            write_jsonl_create_only(
                path,
                [{"z": 2, "a": 1}, {"text": "caf\u00e9", "nested": {"b": 2, "a": 1}}],
            )

            self.assertEqual(
                path.read_text(encoding="utf-8"),
                '{"a":1,"z":2}\n{"nested":{"a":1,"b":2},"text":"caf\u00e9"}\n',
            )

            invalid_row = root / "invalid-row.jsonl"
            with self.assertRaisesRegex(TypeError, "row 1 must be a mapping"):
                write_jsonl_create_only(invalid_row, [{"valid": True}, ["invalid"]])
            self.assertFalse(invalid_row.exists())

            invalid_number = root / "invalid-number.jsonl"
            with self.assertRaises(ValueError):
                write_jsonl_create_only(invalid_number, [{"value": math.nan}])
            self.assertFalse(invalid_number.exists())
            self.assert_no_temporary_files(root)

    def test_concurrent_creation_has_exactly_one_winner(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            path = root / "raced.bin"
            context = multiprocessing.get_context("spawn")
            start = context.Event()
            results = context.Queue()
            contents = [b"first", b"second"]
            processes = [
                context.Process(
                    target=_race_writer,
                    args=(str(path), content, start, results),
                )
                for content in contents
            ]

            for process in processes:
                process.start()
            start.set()
            outcomes = [results.get(timeout=10) for _ in processes]
            for process in processes:
                process.join(timeout=10)
                self.assertEqual(process.exitcode, 0)

            created = [content for status, content in outcomes if status == "created"]
            rejected = [content for status, content in outcomes if status == "exists"]
            self.assertEqual(len(created), 1)
            self.assertEqual(len(rejected), 1)
            self.assertEqual(path.read_bytes(), created[0])
            self.assert_no_temporary_files(root)


if __name__ == "__main__":
    unittest.main()
