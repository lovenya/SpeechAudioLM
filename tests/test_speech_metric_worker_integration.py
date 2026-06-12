from __future__ import annotations

import math
import sys
import types
import unittest
from unittest.mock import Mock, patch

from evaluation.metrics.registry import MetricSpec
from evaluation.metrics.utmos_policy import UTMOSValidationError
from evaluation.result_validation import validate_final_result_completeness
from scripts.eval_backend_worker import run_utmosv2, run_whisper


def _whisper_payload() -> dict:
    return {
        "metrics": ["asr_wer", "dwer"],
        "metric_configs": {
            "asr_wer": {"model_name": "large-v3"},
            "dwer": {"model_name": "large-v3"},
        },
        "cache_dir": "/cache",
        "device": "cuda:1",
        "rows": [
            {
                "row_index": 10,
                "output_file": "/generated/a.wav",
                "source": {"speech_text": "one", "speech_file": "/reference/a.wav"},
            },
            {
                "row_index": 11,
                "output_file": "/generated/b.wav",
                "source": {
                    "speech_text": "one two three four five six seven eight nine",
                    "speech_file": "/reference/b.wav",
                },
            },
        ],
    }


def _utmos_payload(**config: object) -> dict:
    return {
        "metrics": ["utmos"],
        "metric_configs": {"utmos": config},
        "cache_dir": "/cache",
        "device": "cuda:0",
        "rows": [
            {"row_index": 3, "output_file": "/generated/a.wav"},
            {"row_index": 7, "output_file": "/generated/b.wav"},
        ],
    }


class SpeechMetricWorkerIntegrationTest(unittest.TestCase):
    def test_whisper_uses_committed_semantics_counts_and_corpus_aggregation(self) -> None:
        model = Mock()
        transcripts = {
            "/generated/a.wav": "wrong",
            "/reference/a.wav": "one",
            "/generated/b.wav": "one two three four five six seven eight wrong",
            "/reference/b.wav": "one two three four five six seven eight nine",
        }
        model.transcribe.side_effect = lambda path, **kwargs: {"text": transcripts[path]}
        whisper = types.ModuleType("whisper")
        whisper.load_model = Mock(return_value=model)

        with patch.dict(sys.modules, {"whisper": whisper}):
            result = run_whisper(_whisper_payload())

        whisper.load_model.assert_called_once_with("large-v3", download_root="/cache", device="cuda:1")
        for call in model.transcribe.call_args_list:
            self.assertEqual(
                call.kwargs,
                {"language": "en", "task": "transcribe", "fp16": True},
            )
        self.assertEqual(result["sample_metrics"]["10"], {"asr_wer": 1.0, "dwer": 1.0})
        self.assertAlmostEqual(result["sample_metrics"]["11"]["asr_wer"], 1 / 9)
        details = result["predictions"]["10"]["asr_wer_details"]
        self.assertEqual(details["normalized"]["substitutions"], 1)
        self.assertEqual(details["raw"]["reference_words"], 1)
        corpus = result["corpus_metrics"]["asr_wer"]
        self.assertEqual(corpus["semantics"], "WER(reference_transcript, ASR(generated_speech))")
        self.assertEqual(corpus["normalized"]["wer"], 0.2)
        self.assertAlmostEqual(corpus["normalized"]["macro_wer"], (1 + 1 / 9) / 2)
        self.assertNotEqual(corpus["normalized"]["wer"], corpus["normalized"]["macro_wer"])
        self.assertEqual(
            result["corpus_metrics"]["dwer"]["semantics"],
            "WER(ASR(reference_speech), ASR(generated_speech))",
        )
        self.assertEqual(corpus["mean"], corpus["normalized"]["wer"])
        self.assertEqual(corpus["input_count"], 2)
        self.assertIn("not sample mean", corpus["mean_semantics"])
        self.assertEqual(
            result["metric_backends"]["asr_wer"]["policy_metadata"]["corpus"],
            corpus,
        )
        validation = validate_final_result_completeness(
            [MetricSpec("asr_wer", "sample"), MetricSpec("dwer", "sample")],
            ["10", "11"],
            [
                {"sample_id": key, "status": "success", "errors": [], "metrics": metrics}
                for key, metrics in result["sample_metrics"].items()
            ],
            result["corpus_metrics"],
        )
        self.assertTrue(validation.ok)

    def test_whisper_retains_bad_reference_as_attributable_error(self) -> None:
        for reference in (None, "", "..."):
            model = Mock()
            model.transcribe.side_effect = lambda path, **kwargs: {
                "text": "one" if path.endswith("a.wav") else "one two three four five six seven eight nine"
            }
            whisper = types.ModuleType("whisper")
            whisper.load_model = Mock(return_value=model)
            payload = _whisper_payload()
            payload["rows"][0]["source"]["speech_text"] = reference
            with self.subTest(reference=reference), patch.dict(sys.modules, {"whisper": whisper}):
                result = run_whisper(payload)
            self.assertNotIn("asr_wer", result["sample_metrics"]["10"])
            self.assertIn("asr_wer", result["errors"]["10"])
            corpus = result["corpus_metrics"]["asr_wer"]
            self.assertEqual(corpus["attempted_count"], 2)
            self.assertEqual(corpus["passed_count"], 1)
            self.assertEqual(corpus["error_count"], 1)
            self.assertEqual(corpus["error_sample_ids"], ["10"])

    def test_utmos_enforces_official_factory_and_prediction_contract(self) -> None:
        model = Mock()
        model.predict.side_effect = [3.5, 4.5]
        utmosv2 = types.ModuleType("utmosv2")
        utmosv2.create_model = Mock(return_value=model)

        with patch.dict(sys.modules, {"utmosv2": utmosv2}):
            result = run_utmosv2(_utmos_payload(num_workers=2, batch_size=4))

        utmosv2.create_model.assert_called_once_with(
            pretrained=True,
            config="fusion_stage3",
            fold=0,
            checkpoint_path=None,
            seed=42,
            device="cuda:0",
        )
        for call in model.predict.call_args_list:
            self.assertEqual(call.kwargs["predict_dataset"], "sarulab")
            self.assertEqual(call.kwargs["num_repetitions"], 5)
            self.assertFalse(call.kwargs["remove_silent_section"])
            self.assertEqual(call.kwargs["num_workers"], 2)
            self.assertEqual(call.kwargs["batch_size"], 4)
        self.assertEqual(result["sample_metrics"], {"3": {"utmos": 3.5}, "7": {"utmos": 4.5}})
        self.assertEqual(result["corpus_metrics"]["utmos"]["mean"], 4.0)
        self.assertEqual(result["corpus_metrics"]["utmos"]["passed_count"], 2)
        self.assertEqual(result["corpus_metrics"]["utmos"]["error_count"], 0)
        policy = result["metric_backends"]["utmos"]["policy_metadata"]
        self.assertFalse(policy["silence_removal"])
        self.assertEqual(policy["num_repetitions"], 5)
        self.assertEqual(policy["predict_dataset"], "sarulab")
        self.assertEqual(policy["completed_sample_count"], 2)

    def test_utmos_rejects_policy_deviation_and_nonfinite_score(self) -> None:
        utmosv2 = types.ModuleType("utmosv2")
        utmosv2.create_model = Mock()
        with patch.dict(sys.modules, {"utmosv2": utmosv2}):
            with self.assertRaisesRegex(UTMOSValidationError, "paper-comparable"):
                run_utmosv2(_utmos_payload(num_repetitions=1))
        utmosv2.create_model.assert_not_called()

        model = Mock()
        model.predict.side_effect = [4.0, math.nan]
        utmosv2.create_model.return_value = model
        with patch.dict(sys.modules, {"utmosv2": utmosv2}):
            result = run_utmosv2(_utmos_payload())
        self.assertEqual(result["sample_metrics"], {"3": {"utmos": 4.0}})
        self.assertIn("utmos", result["errors"]["7"])
        corpus = result["corpus_metrics"]["utmos"]
        self.assertEqual(corpus["passed_count"], 1)
        self.assertEqual(corpus["error_count"], 1)
        self.assertEqual(corpus["error_sample_ids"], ["7"])


if __name__ == "__main__":
    unittest.main()
