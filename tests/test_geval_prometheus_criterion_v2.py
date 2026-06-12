from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from scripts.run_geval_prometheus_criterion_v2 import (
    CRITERION_WEIGHTS,
    deterministic_diagnostics,
    build_prompt,
    parse_prometheus_result,
    run_geval,
    weighted_final_score,
)


class GEvalPrometheusCriterionV2Test(unittest.TestCase):
    def test_result_parser(self) -> None:
        parsed = parse_prometheus_result("Feedback: Strong connection. [RESULT] 5")
        self.assertEqual(parsed["score"], 5)
        self.assertEqual(parsed["feedback"], "Strong connection.")

        parsed = parse_prometheus_result("Preface\nFeedback: First. [RESULT] 2\nFeedback: Final. [RESULT] 4")
        self.assertEqual(parsed["score"], 4)
        self.assertIn("Final.", parsed["feedback"])

        for invalid in ("Feedback: missing", "Feedback: bad [RESULT] 0", "Feedback: bad [RESULT] 6", "Feedback: bad [RESULT] 4.5", "Feedback: bad [RESULT] 4."):
            with self.subTest(invalid=invalid), self.assertRaises(ValueError):
                parse_prometheus_result(invalid)

    def test_weighted_final_score_requires_every_criterion_and_normalizes(self) -> None:
        criteria = {"a": {"score": 5}, "b": {"score": 3}}
        self.assertEqual(weighted_final_score(criteria, {"a": 2, "b": 1}), 13 / 3)
        self.assertIsNone(weighted_final_score({"a": {"score": 5}}, {"a": 2, "b": 1}))

    def test_prompt_rendering(self) -> None:
        plausibility = build_prompt(
            variant="plausibility",
            direction="audio_to_speech",
            criterion="spoken_naturalness",
            input_text="A dog barks.",
            generated_text="Please let the dog inside.",
        )
        self.assertNotIn("###Reference Anchors:", plausibility)
        self.assertIn("Direction: audio_to_speech", plausibility)
        self.assertIn("sounds like a natural human utterance", plausibility)
        self.assertIn("Ignore its relevance to the input", plausibility)

        grounded = build_prompt(
            variant="reference_grounded",
            direction="speech_to_audio",
            criterion="reference_anchor_grounding",
            input_text="Please let the dog inside.",
            generated_text="A dog barks by a door.",
            reference_anchors=[{"anchor_id": "a", "text": "A dog is barking.", "source": "paired_dataset_reference"}],
        )
        self.assertIn("###Reference Anchors:", grounded)
        self.assertIn("[Anchor 1 | source=paired_dataset_reference | id=a]", grounded)
        self.assertIn("reference audio-caption anchor", grounded)

        anchor = {"anchor_id": "a", "text": "UNIQUE_REFERENCE_ANCHOR_TEXT", "source": "paired_dataset_reference"}
        for direction in ("audio_to_speech", "speech_to_audio"):
            for criterion in CRITERION_WEIGHTS[("reference_grounded", direction)]:
                with self.subTest(direction=direction, criterion=criterion):
                    prompt = build_prompt(
                        variant="reference_grounded",
                        direction=direction,
                        criterion=criterion,
                        input_text="Input text.",
                        generated_text="Generated text.",
                        reference_anchors=[anchor],
                    )
                    if criterion == "reference_anchor_grounding":
                        self.assertIn("###Reference Anchors:", prompt)
                        self.assertIn(anchor["text"], prompt)
                    else:
                        self.assertNotIn("###Reference Anchors:", prompt)
                        self.assertNotIn(anchor["text"], prompt)

        grounded_non_contradiction = build_prompt(
            variant="reference_grounded",
            direction="speech_to_audio",
            criterion="non_contradiction",
            input_text="Please let the dog inside.",
            generated_text="A dog barks by a door.",
            reference_anchors=[anchor],
        )
        self.assertIn(
            "Do not lower the score merely because the response is weakly related",
            grounded_non_contradiction,
        )

    def test_diagnostics_do_not_change_final_score(self) -> None:
        weights = {"non_contradiction": 0.5, "spoken_naturalness": 0.5}
        criteria = {"non_contradiction": {"score": 2}, "spoken_naturalness": {"score": 4}}
        score = weighted_final_score(criteria, weights)
        diagnostics = deterministic_diagnostics(criteria)
        self.assertEqual(score, 3)
        self.assertEqual(score, weighted_final_score(criteria, weights))
        self.assertIn("clear_contradiction_or_major_tension", diagnostics["failure_modes"])

    def test_minimal_integration_and_missing_reference(self) -> None:
        class FakeJudge:
            name = "fake-prometheus"
            cache_identity = "fake-prometheus-v1"

            def __init__(self) -> None:
                self.calls = 0

            def generate(self, prompts: list[str]) -> list[str]:
                self.calls += len(prompts)
                return ["Feedback: Criterion is good. [RESULT] 4" for _ in prompts]

        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            input_path = root / "input.jsonl"
            output_path = root / "output.jsonl"
            summary_path = root / "summary.json"
            input_path.write_text(
                json.dumps(
                    {
                        "row_index": 0,
                        "sample_id": "sample",
                        "direction": "audio_to_speech",
                        "input_text": "A dog barks.",
                        "generated_text": "Please let the dog inside.",
                    }
                )
                + "\n",
                encoding="utf-8",
            )
            judge = FakeJudge()
            run_geval(
                input_path=input_path,
                output_path=output_path,
                summary_output=summary_path,
                summary_csv_output=None,
                judge=judge,
                variant="plausibility",
                cache_dir=root / "cache",
                batch_size=3,
            )
            row = json.loads(output_path.read_text(encoding="utf-8"))
            self.assertEqual(judge.calls, len(CRITERION_WEIGHTS[("plausibility", "audio_to_speech")]))
            self.assertEqual(row["final_score"], 4)
            self.assertEqual(row["status"], "complete")
            self.assertEqual(row["scoring_policy_version"], "geval_weighted_mean_v1_no_caps")
            self.assertTrue(all(result["feedback"] for result in row["criteria"].values()))
            self.assertTrue(all("raw_response" in result for result in row["criteria"].values()))

            run_geval(
                input_path=input_path,
                output_path=output_path,
                summary_output=summary_path,
                summary_csv_output=None,
                judge=FakeJudge(),
                variant="reference_grounded",
                cache_dir=None,
                batch_size=3,
            )
            row = json.loads(output_path.read_text(encoding="utf-8"))
            self.assertIsNone(row["final_score"])
            self.assertEqual(row["status"], "incomplete")
            self.assertEqual(row["incomplete_reason"], "missing_required_reference_anchor")

    def test_failed_criterion_makes_final_score_null_after_failed_repair(self) -> None:
        class OneFailureJudge:
            name = "one-failure"
            cache_identity = "one-failure-v1"

            def generate(self, prompts: list[str]) -> list[str]:
                outputs = []
                for prompt in prompts:
                    if "Criterion: non_contradiction" in prompt or prompt.startswith("Repair the evaluator response"):
                        outputs.append("unparseable")
                    else:
                        outputs.append("Feedback: Fine. [RESULT] 4")
                return outputs

        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            input_path = root / "input.jsonl"
            output_path = root / "output.jsonl"
            input_path.write_text(
                json.dumps(
                    {
                        "row_index": 0,
                        "sample_id": "sample",
                        "direction": "speech_to_audio",
                        "input_text": "Please shut the window.",
                        "generated_text": "Strong wind blows through an open window.",
                    }
                )
                + "\n",
                encoding="utf-8",
            )
            run_geval(
                input_path=input_path,
                output_path=output_path,
                summary_output=root / "summary.json",
                summary_csv_output=None,
                judge=OneFailureJudge(),
                variant="plausibility",
                cache_dir=None,
                batch_size=20,
            )
            row = json.loads(output_path.read_text(encoding="utf-8"))
            self.assertIsNone(row["final_score"])
            self.assertEqual(row["status"], "incomplete")
            self.assertEqual(row["criteria"]["non_contradiction"]["parse_status"], "parse_failed")
            self.assertEqual(row["criteria"]["non_contradiction"]["initial_raw_response"], "unparseable")
            self.assertEqual(row["criteria"]["non_contradiction"]["raw_response"], "unparseable")


    def test_successful_repair_and_cache_resume(self) -> None:
        class RepairJudge:
            name = "repair-judge"
            cache_identity = "repair-judge-v1"

            def __init__(self) -> None:
                self.calls = 0

            def generate(self, prompts: list[str]) -> list[str]:
                self.calls += len(prompts)
                return [
                    "Feedback: Repaired feedback. [RESULT] 3"
                    if prompt.startswith("Repair the evaluator response")
                    else "not parseable"
                    for prompt in prompts
                ]

        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            input_path = root / "input.jsonl"
            output_path = root / "output.jsonl"
            input_path.write_text(
                json.dumps({
                    "row_index": 0,
                    "sample_id": "sample",
                    "direction": "audio_to_speech",
                    "input_text": "A dog barks.",
                    "generated_text": "Please let the dog inside.",
                }) + "\n",
                encoding="utf-8",
            )
            cache_dir = root / "cache"
            first_judge = RepairJudge()
            run_geval(
                input_path=input_path, output_path=output_path, summary_output=root / "summary.json",
                summary_csv_output=None, judge=first_judge, variant="plausibility",
                cache_dir=cache_dir, batch_size=20,
            )
            row = json.loads(output_path.read_text(encoding="utf-8"))
            self.assertEqual(row["final_score"], 3)
            self.assertTrue(all(result["repair_applied"] for result in row["criteria"].values()))
            self.assertTrue(all("initial_raw_response" in result for result in row["criteria"].values()))

            second_judge = RepairJudge()
            run_geval(
                input_path=input_path, output_path=output_path, summary_output=root / "summary.json",
                summary_csv_output=None, judge=second_judge, variant="plausibility",
                cache_dir=cache_dir, batch_size=20,
            )
            self.assertEqual(second_judge.calls, 0)


if __name__ == "__main__":
    unittest.main()
