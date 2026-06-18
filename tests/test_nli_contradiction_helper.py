from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from scripts.run_nli_contradiction_helper import run_nli_helper, score_pair


class FixedScorer:
    name = "fixed_nli"

    def predict(self, premise: str, hypothesis: str) -> dict[str, float]:
        if "silent" in hypothesis.lower() or "silent" in premise.lower():
            return {"entailment": 0.04, "neutral": 0.16, "contradiction": 0.8}
        return {"entailment": 0.1, "neutral": 0.85, "contradiction": 0.05}


class NliContradictionHelperTest(unittest.TestCase):
    def test_score_pair_uses_max_bidirectional_contradiction(self) -> None:
        result = score_pair(
            "I cannot sleep with all this noise outside.",
            "A silent peaceful forest.",
            FixedScorer(),
            contradiction_threshold=0.7,
        )

        self.assertEqual(result.max_contradiction_probability, 0.8)
        self.assertEqual(result.predicted_label_using_max_contradiction, "contradiction")

    def test_neutral_is_not_failure_when_contradiction_is_low(self) -> None:
        result = score_pair(
            "Can someone close the window?",
            "A dog is barking outside.",
            FixedScorer(),
            contradiction_threshold=0.7,
        )

        self.assertEqual(result.forward["neutral"], 0.85)
        self.assertEqual(result.predicted_label_using_max_contradiction, "not_contradiction")

    def test_run_nli_helper_writes_rows_and_summary(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            input_path = root / "pairs.jsonl"
            output_path = root / "nli_results.jsonl"
            summary_path = root / "summary.json"
            rows = [
                {
                    "pair_id": "a",
                    "split": "val",
                    "direction": "speech_to_audio",
                    "primary_audio_class": "road_vehicle_traffic",
                    "input_text": "I cannot sleep with all this noise outside.",
                    "generated_text": "A silent peaceful forest.",
                },
                {
                    "sample_id": "b",
                    "split": "val",
                    "direction": "audio_to_speech",
                    "input_text": "A dog is barking outside.",
                    "generated_text": "Can someone close the window?",
                },
            ]
            input_path.write_text("\n".join(json.dumps(row) for row in rows) + "\n", encoding="utf-8")

            summary = run_nli_helper(
                input_path=input_path,
                output_path=output_path,
                summary_output=summary_path,
                scorer=FixedScorer(),
                contradiction_threshold=0.7,
            )

            output_rows = [json.loads(line) for line in output_path.read_text(encoding="utf-8").splitlines()]
            self.assertEqual(len(output_rows), 2)
            self.assertEqual(output_rows[0]["pair_id"], "a")
            self.assertEqual(output_rows[1]["pair_id"], "b")
            self.assertIn("forward_entailment_probability", output_rows[0])
            self.assertIn("reverse_contradiction_probability", output_rows[0])
            self.assertEqual(output_rows[0]["predicted_label_using_max_contradiction"], "contradiction")
            self.assertEqual(output_rows[1]["predicted_label_using_max_contradiction"], "not_contradiction")
            self.assertEqual(summary["total_pairs_processed"], 2)
            self.assertEqual(summary["contradiction_rate_at_thresholds"]["0.7"], 0.5)
            self.assertEqual(json.loads(summary_path.read_text(encoding="utf-8"))["nli_model"], "fixed_nli")


if __name__ == "__main__":
    unittest.main()
