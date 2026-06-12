from __future__ import annotations

import os
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

from evaluation.runner import _backend_env


class BackendEnvironmentTest(unittest.TestCase):
    def test_pythonpath_includes_repo_configured_and_existing_paths_without_duplicates(self) -> None:
        repo_root = Path(__file__).resolve().parents[1]
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            vendor = root / "vendor"
            inherited = root / "inherited"
            configured = os.pathsep.join((str(vendor), str(repo_root), str(vendor)))
            existing = os.pathsep.join((str(inherited), str(vendor), "."))

            with patch.dict(os.environ, {"PYTHONPATH": existing}, clear=True):
                env = _backend_env(
                    SimpleNamespace(cache_dir=root / "cache"),
                    {"metric": {"pythonpath": configured}},
                )

        self.assertEqual(
            env["PYTHONPATH"].split(os.pathsep),
            [str(repo_root), str(vendor), str(inherited)],
        )

    def test_pythonpath_always_starts_with_repo_root(self) -> None:
        repo_root = Path(__file__).resolve().parents[1]
        with tempfile.TemporaryDirectory() as tmpdir:
            with patch.dict(os.environ, {}, clear=True):
                env = _backend_env(
                    SimpleNamespace(cache_dir=Path(tmpdir) / "cache"),
                    {},
                )

        self.assertEqual(env["PYTHONPATH"], str(repo_root))


if __name__ == "__main__":
    unittest.main()
