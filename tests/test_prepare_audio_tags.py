from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from scripts.prepare_audio_tags import enrich_splits, retain_tags


class FakeTagger:
    name = "fake_tagger"

    def __init__(self, predictions: dict[str, list[tuple[str, float]]]) -> None:
        self.predictions = predictions

    def predict(self, audio_path: Path) -> list[tuple[str, float]]:
        return self.predictions[audio_path.name]


class AudioTagPrepTest(unittest.TestCase):
    def test_threshold_retains_all_qualifying_tags(self) -> None:
        retained = retain_tags(
            [("speech", 0.91), ("engine", 0.82), ("music", 0.12)],
            tag_threshold=0.6,
            min_tags=1,
            fallback_top_k=3,
        )
        self.assertEqual(retained.tags, ["speech", "engine"])
        self.assertEqual(retained.primary, "speech")
        self.assertFalse(retained.used_fallback)

    def test_fewer_than_min_tags_falls_back_to_top_k(self) -> None:
        retained = retain_tags(
            [("speech", 0.91), ("engine", 0.32), ("music", 0.12), ("wind", 0.08)],
            tag_threshold=0.95,
            min_tags=1,
            fallback_top_k=3,
        )
        self.assertEqual(retained.tags, ["speech", "engine", "music"])
        self.assertEqual(retained.primary, "speech")
        self.assertTrue(retained.used_fallback)

    def test_gap_limited_fallback_stops_at_confidence_cliff(self) -> None:
        retained = retain_tags(
            [("clear", 0.8), ("also_clear", 0.78), ("weak", 0.2)],
            tag_threshold=0.9,
            min_tags=3,
            fallback_top_k=3,
            fallback_max_score_gap=0.25,
        )
        self.assertEqual(retained.tags, ["clear", "also_clear"])
        self.assertTrue(retained.used_fallback)

    def test_no_predictions_has_no_primary_tag(self) -> None:
        retained = retain_tags([], tag_threshold=0.6, min_tags=1, fallback_top_k=3)
        self.assertEqual(retained.tags, [])
        self.assertIsNone(retained.primary)
        self.assertTrue(retained.used_fallback)
        self.assertTrue(retained.no_valid_tags)

    def test_enrich_splits_preserves_original_rows_and_writes_summary(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            dataset_root = root / "dataset"
            split_dir = root / "splits"
            output_root = root / "audio_tags"
            report = root / "reports" / "summary.json"
            dataset_root.mkdir()
            split_dir.mkdir()

            split_path = split_dir / "ttas_val.jsonl"
            rows = [
                {"audio_file": "audio/a.wav", "audio_cap": "Engine starts", "row": 1},
                {"audio_file": "audio/b.wav", "audio_cap": "Rain falls", "row": 2},
            ]
            original_text = "\n".join(json.dumps(row) for row in rows) + "\n"
            split_path.write_text(original_text, encoding="utf-8")
            config_path = root / "ttas_dataset.json"
            config_path.write_text(
                json.dumps({"dataset_root": str(dataset_root), "splits": {"val": str(split_path)}}),
                encoding="utf-8",
            )

            tagger = FakeTagger(
                {
                    "a.wav": [("engine", 0.9), ("vehicle", 0.7), ("music", 0.1)],
                    "b.wav": [("rain", 0.5), ("water", 0.4), ("speech", 0.3)],
                }
            )
            summary = enrich_splits(
                dataset_config_path=config_path,
                output_root=output_root,
                summary_output=report,
                splits=["val"],
                tagger=tagger,
                tag_threshold=0.6,
                min_tags=1,
                fallback_top_k=2,
                fallback_max_score_gap=0.25,
                max_samples=None,
                store_all_tag_scores=True,
                store_top_tag_scores=2,
            )

            self.assertEqual(split_path.read_text(encoding="utf-8"), original_text)
            enriched_rows = [
                json.loads(line)
                for line in (output_root / "ttas_val_panns_cnn14.jsonl").read_text(encoding="utf-8").splitlines()
            ]
            self.assertEqual(enriched_rows[0]["audio_cap"], "Engine starts")
            self.assertEqual(enriched_rows[0]["retained_audio_tags"], ["engine", "vehicle"])
            self.assertEqual(enriched_rows[0]["primary_audio_tag"], "engine")
            self.assertEqual(enriched_rows[0]["audio_tagger_name"], "fake_tagger")
            self.assertEqual(list(enriched_rows[0]["all_audio_tag_scores"]), ["engine", "vehicle", "music"])
            self.assertEqual(enriched_rows[0]["all_audio_tag_scores"]["music"], 0.1)
            self.assertEqual(list(enriched_rows[0]["top_audio_tag_scores"]), ["engine", "vehicle"])
            self.assertEqual(enriched_rows[1]["retained_audio_tags"], ["rain", "water"])
            self.assertTrue(summary["store_all_tag_scores"])
            self.assertEqual(summary["store_top_tag_scores"], 2)
            self.assertEqual(summary["total_samples_processed"], 2)
            self.assertEqual(summary["fallback_top_k_used"], 1)
            self.assertEqual(summary["no_valid_tags"], 0)
            self.assertEqual(
                list(summary["tag_frequencies"].items()),
                [("engine", 1), ("rain", 1), ("vehicle", 1), ("water", 1)],
            )
            self.assertEqual(json.loads(report.read_text(encoding="utf-8"))["splits"]["val"]["processed"], 2)


if __name__ == "__main__":
    unittest.main()
