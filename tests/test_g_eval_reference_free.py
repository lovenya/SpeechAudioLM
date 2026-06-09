from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from scripts.run_g_eval_reference_free import StubJudge, _parse_judge_json, _should_flag, build_prompt, run_g_eval


class GEvalReferenceFreeTest(unittest.TestCase):
    def test_prompt_contains_reference_free_criteria(self) -> None:
        prompt = build_prompt(
            direction="audio_to_speech",
            input_text="A dog is barking outside.",
            generated_text="Can someone close the window?",
        )

        self.assertIn("not exact reference matching", prompt)
        self.assertIn("Causal / situational plausibility", prompt)
        self.assertEqual(prompt.count("Input text:"), 1)
        self.assertEqual(prompt.count("Direction:"), 1)
        self.assertNotIn("NLI", prompt)
        self.assertIn('"overall_score": <1-5>', prompt)

    def test_parser_preserves_decimal_scores(self) -> None:
        response = json.dumps({
            "non_contradiction": 5,
            "causal_situational_plausibility": 4,
            "physical_grounding": 5,
            "contextual_emotional_fit": 4,
            "specificity_without_over_invention": 4,
            "creativity_valid_diversity": 5,
            "naturalness": 5,
            "overall_score": 4.5,
            "should_flag": False,
            "failure_modes": [],
            "short_rationale": "Good with minor issues.",
        })
        parsed = _parse_judge_json(response)
        self.assertEqual(parsed["overall_score"], 4.5)

    def test_should_flag_is_deterministic(self) -> None:
        self.assertTrue(_should_flag(4, ["situationally_implausible"]))
        self.assertTrue(_should_flag(2, []))
        self.assertFalse(_should_flag(3, ["too_vague"]))

    def test_run_g_eval_writes_rows_summary_and_cache(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            input_path = root / "pairs.jsonl"
            output_path = root / "g_eval.jsonl"
            summary_path = root / "summary.json"
            summary_csv_path = root / "summary.csv"
            cache_dir = root / "cache"
            rows = [
                {
                    "row_index": 0,
                    "pair_id": "a",
                    "split": "val",
                    "direction": "audio_to_speech",
                    "input_text": "A dog is barking outside.",
                    "generated_text": "Can someone close the window?",
                    "primary_audio_class": "domestic_pets",
                }
            ]
            input_path.write_text("\n".join(json.dumps(row) for row in rows) + "\n", encoding="utf-8")

            summary = run_g_eval(
                input_path=input_path,
                output_path=output_path,
                summary_output=summary_path,
                summary_csv_output=summary_csv_path,
                judge=StubJudge(),
                cache_dir=cache_dir,
                batch_size=1,
            )

            output_rows = [json.loads(line) for line in output_path.read_text(encoding="utf-8").splitlines()]
            self.assertEqual(output_rows[0]["overall_score"], 4)
            self.assertEqual(output_rows[0]["judge_model"], "stub_g_eval_reference_free")
            self.assertEqual(summary["valid_judgments"], 1)
            self.assertTrue(summary_csv_path.is_file())
            self.assertEqual(len(list(cache_dir.glob("*.json"))), 1)


    def test_run_g_eval_repairs_malformed_judgment(self) -> None:
        class RepairJudge:
            name = "repair-test"

            def generate(self, prompts: list[str]) -> list[str]:
                if prompts[0].startswith("Repair the judge response"):
                    return [StubJudge().generate(["Input text: ordinary"])[0]]
                return ["not json"]

        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            input_path = root / "pairs.jsonl"
            output_path = root / "g_eval.jsonl"
            summary_path = root / "summary.json"
            input_path.write_text(json.dumps({
                "row_index": 0,
                "pair_id": "a",
                "split": "val",
                "direction": "audio_to_speech",
                "input_text": "A dog is barking outside.",
                "generated_text": "Can someone close the window?",
            }) + "\n", encoding="utf-8")

            summary = run_g_eval(
                input_path=input_path,
                output_path=output_path,
                summary_output=summary_path,
                summary_csv_output=None,
                judge=RepairJudge(),
                cache_dir=None,
                batch_size=1,
            )

            row = json.loads(output_path.read_text(encoding="utf-8"))
            self.assertTrue(row["repair_applied"])
            self.assertEqual(row["initial_raw_judge_response"], "not json")
            self.assertTrue(summary["complete"])


if __name__ == "__main__":
    unittest.main()
