from __future__ import annotations

import sys
import types
import unittest
from unittest.mock import patch

from scripts.eval_backend_worker import run_aac_metrics


class SpiderSemanticsTest(unittest.TestCase):
    def test_spider_receives_complete_candidate_without_local_word_cap(self) -> None:
        captured: dict[str, object] = {}

        def spider(candidates, references, **kwargs):
            captured["candidates"] = candidates
            return {"spider": 0.5}, {"spider": [0.5]}

        aac_metrics = types.ModuleType("aac_metrics")
        aac_metrics.evaluate = lambda *args, **kwargs: None
        functional = types.ModuleType("aac_metrics.functional")
        clap_module = types.ModuleType("aac_metrics.functional.clap_sim")
        clap_module.clap_sim = lambda *args, **kwargs: None
        mace_module = types.ModuleType("aac_metrics.functional.mace")
        mace_module.mace = lambda *args, **kwargs: None
        spider_module = types.ModuleType("aac_metrics.functional.spider")
        spider_module.spider = spider
        modules = {
            "aac_metrics": aac_metrics,
            "aac_metrics.functional": functional,
            "aac_metrics.functional.clap_sim": clap_module,
            "aac_metrics.functional.mace": mace_module,
            "aac_metrics.functional.spider": spider_module,
        }
        candidate = " ".join(f"word{index}" for index in range(150))
        payload = {
            "metrics": ["spider"],
            "metric_configs": {"spider": {"java_max_memory": "1G"}},
            "rows": [
                {
                    "row_index": 0,
                    "candidate_text": candidate,
                    "source": {"audio_cap": "reference", "audio_file": "unused.wav"},
                }
            ],
            "cache_dir": "cache",
            "device": "cpu",
        }

        with patch.dict(sys.modules, modules), patch(
            "scripts.eval_backend_worker._resolve_java_path", return_value="/usr/bin/java"
        ):
            result = run_aac_metrics(payload)

        self.assertEqual(captured["candidates"], [candidate])
        self.assertEqual(len(captured["candidates"][0].split()), 150)
        self.assertEqual(result["corpus_metrics"]["spider"], 0.5)
        self.assertEqual(result["sample_metrics"]["0"]["spider"], 0.5)


if __name__ == "__main__":
    unittest.main()
