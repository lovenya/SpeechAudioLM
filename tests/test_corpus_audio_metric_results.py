from __future__ import annotations

import math
import unittest

from scripts.eval_backend_worker import _audioldm_corpus_metrics


class CorpusAudioMetricResultTest(unittest.TestCase):
    def test_inception_nonfinite_result_is_rejected(self) -> None:
        common = {"attempted_count": 2, "passed_count": 2}
        with self.assertRaisesRegex(ValueError, "non-finite"):
            _audioldm_corpus_metrics(
                "inception_score",
                {"inception_score_mean": math.nan, "inception_score_std": math.nan},
                common,
            )

    def test_finite_inception_result_preserves_counts(self) -> None:
        common = {"attempted_count": 50, "passed_count": 49}
        result = _audioldm_corpus_metrics(
            "inception_score",
            {"inception_score_mean": 2.5, "inception_score_std": 0.2},
            common,
        )["inception_score"]
        self.assertEqual(result["mean"], 2.5)
        self.assertEqual(result["std"], 0.2)
        self.assertEqual(result["passed_count"], 49)


if __name__ == "__main__":
    unittest.main()
