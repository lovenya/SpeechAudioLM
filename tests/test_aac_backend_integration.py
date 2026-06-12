from __future__ import annotations

import math
import sys
import types
import unittest
from unittest.mock import patch

from scripts.eval_backend_worker import run_aac_metrics


def _modules(*, spider=None, mace=None, clap=None, fense=None):
    modules = {
        "aac_metrics": types.ModuleType("aac_metrics"),
        "aac_metrics.functional": types.ModuleType("aac_metrics.functional"),
    }
    for name, function_name, function in (
        ("clap_sim", "clap_sim", clap),
        ("fense", "fense", fense),
        ("mace", "mace", mace),
        ("spider", "spider", spider),
    ):
        module = types.ModuleType(f"aac_metrics.functional.{name}")
        setattr(module, function_name, function or (lambda *args, **kwargs: None))
        modules[f"aac_metrics.functional.{name}"] = module
    return modules


def _payload(metrics, configs=None):
    return {
        "metrics": metrics,
        "metric_configs": configs or {metric: {} for metric in metrics},
        "rows": [
            {"row_index": 0, "candidate_text": " First\ncaption! ", "source": {"audio_cap": " Ref\none ", "audio_file": "a.wav"}},
            {"row_index": 1, "candidate_text": "Second caption?", "source": {"audio_cap": "Ref two", "audio_file": "b.wav"}},
        ],
        "cache_dir": "cache",
        "device": "cpu",
    }


class AACBackendIntegrationTest(unittest.TestCase):
    def test_spider_accepts_official_corpus_score_different_from_sample_mean(self):
        captured = {}
        def spider(candidates, references, **kwargs):
            captured.update(candidates=candidates, references=references)
            return {"spider": 0.73}, {"spider": [0.2, 0.4]}
        with patch.dict(sys.modules, _modules(spider=spider)), patch(
            "scripts.eval_backend_worker._resolve_java_path", return_value="/usr/bin/java"
        ):
            result = run_aac_metrics(_payload(["spider"]))
        self.assertEqual(captured["candidates"], ["First caption!", "Second caption?"])
        self.assertEqual(captured["references"], [["Ref one"], ["Ref two"]])
        self.assertEqual(result["corpus_metrics"]["spider"], 0.73)
        self.assertEqual([result["sample_metrics"][str(i)]["spider"] for i in range(2)], [0.2, 0.4])
        self.assertIn("do not compare", result["metric_backends"]["spider"]["policy_metadata"]["corpus_score_policy"])

    def test_mace_and_clapsim_use_seed_and_validate_official_means(self):
        captured = {}
        def mace(candidates, references, **kwargs):
            captured["mace"] = kwargs
            return {"mace": 0.325}, {"clap_sim": [0.5, 0.5], "fer": [0.0, 1.0], "mace": 0.325}
        def clap(candidates, **kwargs):
            captured["clap"] = kwargs
            return {"clap_sim": 0.25}, {"clap_sim": [0.1, 0.4]}
        configs = {
            "mace": {"penalty": 0.7, "error_threshold": 0.8},
            "aac_clapsim": {},
        }
        fake_tokenizer = types.SimpleNamespace(
            encode_plus=lambda **kwargs: {"input_ids": types.SimpleNamespace(shape=(1, 77))}
        )
        with patch.dict(sys.modules, _modules(mace=mace, clap=clap)), patch(
            "scripts.eval_backend_worker._ms_clap_gpt2_tokenizer",
            return_value=fake_tokenizer,
        ):
            result = run_aac_metrics(_payload(["mace", "aac_clapsim"], configs))
        self.assertEqual(captured["mace"]["seed"], 1337)
        self.assertEqual(captured["mace"]["penalty"], 0.7)
        self.assertEqual(captured["mace"]["error_threshold"], 0.8)
        self.assertEqual(captured["clap"]["seed"], 1337)
        self.assertEqual(result["metric_backends"]["mace"]["policy_metadata"]["crop_seed"], 1337)
        self.assertEqual(result["sample_metrics"]["1"]["mace"], 0.15000000000000002)

    def test_fense_receives_raw_text_and_passes_policy_parameters(self):
        captured = {}
        def fense(candidates, references, **kwargs):
            captured.update(candidates=candidates, references=references, kwargs=kwargs)
            return {"fense": 0.5}, {"fense": [0.25, 0.75]}
        configs = {"fense": {"penalty": 0.6, "error_threshold": 0.85}}
        with patch.dict(sys.modules, _modules(fense=fense)):
            result = run_aac_metrics(_payload(["fense"], configs))
        self.assertEqual(captured["candidates"][0], " First\ncaption! ")
        self.assertEqual(captured["references"][0][0], " Ref\none ")
        self.assertEqual(captured["kwargs"]["penalty"], 0.6)
        self.assertEqual(captured["kwargs"]["error_threshold"], 0.85)
        self.assertTrue(result["metric_backends"]["fense"]["policy_metadata"]["spider_fl"]["raw_text_reaches_fense_api"])

    def test_rejects_incomplete_and_nonfinite_sample_vectors(self):
        def spider(*args, **kwargs):
            return {"spider": 0.2}, {"spider": [0.2]}
        with patch.dict(sys.modules, _modules(spider=spider)), patch(
            "scripts.eval_backend_worker._resolve_java_path", return_value="/usr/bin/java"
        ):
            with self.assertRaisesRegex(ValueError, "incomplete or misaligned"):
                run_aac_metrics(_payload(["spider"]))
        def fense(*args, **kwargs):
            return {"fense": 0.5}, {"fense": [0.5, math.nan]}
        with patch.dict(sys.modules, _modules(fense=fense)):
            with self.assertRaisesRegex(ValueError, "non-finite"):
                run_aac_metrics(_payload(["fense"]))


if __name__ == "__main__":
    unittest.main()
