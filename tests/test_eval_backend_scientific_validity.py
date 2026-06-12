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
    _stable_clap_text_embeddings,
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

    def test_clap_text_embedding_fallback_isolates_failed_samples(self) -> None:
        class NoGrad:
            def __enter__(self):
                return None

            def __exit__(self, exc_type, exc, traceback):
                return False

        class Cuda:
            emptied = False

            @staticmethod
            def is_available() -> bool:
                return True

            @classmethod
            def empty_cache(cls) -> None:
                cls.emptied = True

        fake_torch = types.SimpleNamespace(no_grad=NoGrad, cuda=Cuda)

        class Model:
            def get_text_embedding(self, texts, **kwargs):
                if len(texts) > 1:
                    raise RuntimeError("batch failed")
                if texts[0] == "bad":
                    raise ValueError("bad prompt")
                return [f"embedding:{texts[0]}"]

        embeddings, errors = _stable_clap_text_embeddings(
            Model(), ["good", "bad", "also good"], lambda value: value, fake_torch
        )

        self.assertEqual(embeddings, ["embedding:good", None, "embedding:also good"])
        self.assertIsNone(errors[0])
        self.assertIn("bad prompt", errors[1])
        self.assertIsNone(errors[2])
        self.assertTrue(Cuda.emptied)


if __name__ == "__main__":
    unittest.main()
