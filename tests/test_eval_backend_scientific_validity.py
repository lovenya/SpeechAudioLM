from __future__ import annotations

import math
import sys
import tempfile
import types
import unittest
from pathlib import Path
from unittest.mock import patch

from scripts.eval_backend_worker import (
    _assign_sample_scores,
    _reject_short_audio_files,
    _require_minimum_audio_files,
    _scalar,
)


class EvalBackendScientificValidityTest(unittest.TestCase):
    def test_rejects_small_corpus_instead_of_duplicating_samples(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            (root / "only.wav").write_bytes(b"audio")
            with self.assertRaisesRegex(ValueError, "Refusing to duplicate"):
                _require_minimum_audio_files(root, 2, "generated")
            self.assertEqual([path.name for path in root.iterdir()], ["only.wav"])

    def test_rejects_short_audio_instead_of_padding(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            audio = root / "short.wav"
            audio.write_bytes(b"unchanged")
            fake_soundfile = types.ModuleType("soundfile")
            fake_soundfile.info = lambda path: types.SimpleNamespace(frames=8000, samplerate=16000)
            with patch.dict(sys.modules, {"soundfile": fake_soundfile}):
                with self.assertRaisesRegex(ValueError, "Refusing to pad"):
                    _reject_short_audio_files(root, 1.0, "reference")
            self.assertEqual(audio.read_bytes(), b"unchanged")

    def test_scalar_rejects_missing_nonfinite_and_sentinel_values(self) -> None:
        for value in (None, math.nan, math.inf, -math.inf):
            with self.subTest(value=value), self.assertRaises(ValueError):
                _scalar(value, metric="fad")
        with self.assertRaisesRegex(ValueError, "sentinel"):
            _scalar(-1, metric="fad", rejected_values=(-1.0,))

    def test_sample_score_assignment_rejects_nonfinite_values(self) -> None:
        rows = [{"row_index": 0}]
        sample_metrics = {"0": {}}
        with self.assertRaisesRegex(ValueError, "non-finite"):
            _assign_sample_scores(rows, sample_metrics, "metric", [math.nan])
        self.assertEqual(sample_metrics, {"0": {}})


if __name__ == "__main__":
    unittest.main()
