from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from scripts.build_reference_pools import build_reference_pools


class BuildReferencePoolsTest(unittest.TestCase):
    def test_builds_split_aware_pools_and_pair_mapping(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            train = root / "train.jsonl"
            val = root / "val.jsonl"
            pool_output = root / "class_reference_pools.json"
            mapping_output = root / "pair_reference_mapping.jsonl"
            summary_output = root / "reference_pool_summary.json"

            train.write_text(
                json.dumps(
                    {
                        "sample_id": "a",
                        "audio_file": "audio/a.wav",
                        "speech_file": "speech/a.wav",
                        "speech_text": "Traffic is loud.",
                        "audio_cap": "A car passes",
                        "primary_audio_class": "road_vehicle_traffic",
                    }
                )
                + "\n",
                encoding="utf-8",
            )
            val.write_text(
                json.dumps(
                    {
                        "sample_id": "b",
                        "audio_file": "audio/b.wav",
                        "speech_file": "speech/b.wav",
                        "speech_text": "It is raining.",
                        "audio_cap": "Rain falls",
                        "primary_audio_class": "rain_thunder_weather",
                    }
                )
                + "\n",
                encoding="utf-8",
            )

            summary = build_reference_pools(
                input_manifests={"train": train, "val": val},
                pool_output=pool_output,
                mapping_output=mapping_output,
                summary_output=summary_output,
                splits=["train", "val"],
            )

            pools = json.loads(pool_output.read_text(encoding="utf-8"))
            self.assertEqual(pools["road_vehicle_traffic"]["train"][0]["pair_id"], "a")
            self.assertEqual(pools["road_vehicle_traffic"]["train"][0]["audio_id"], "a")
            self.assertEqual(pools["road_vehicle_traffic"]["val"], [])
            self.assertEqual(pools["rain_thunder_weather"]["val"][0]["speech_transcript"], "It is raining.")

            mappings = [json.loads(line) for line in mapping_output.read_text(encoding="utf-8").splitlines()]
            self.assertEqual(len(mappings), 2)
            self.assertEqual(mappings[0]["audio_to_speech"]["reference_pool_class"], "road_vehicle_traffic")
            self.assertEqual(mappings[0]["audio_to_speech"]["reference_pool_split"], "train")
            self.assertEqual(mappings[0]["speech_to_audio"]["reference_pool_class"], "road_vehicle_traffic")
            self.assertEqual(mappings[0]["speech_to_audio"]["reference_pool_split"], "train")
            self.assertEqual(mappings[0]["audio_to_speech"]["reference_pool_class"], mappings[0]["primary_audio_class"])
            self.assertEqual(mappings[0]["speech_to_audio"]["reference_pool_class"], mappings[0]["primary_audio_class"])

            self.assertEqual(summary["total_pairs"], 2)
            self.assertEqual(summary["num_classes"], 2)
            self.assertTrue(summary["validation"]["all_pair_ids_unique"])
            self.assertTrue(summary["validation"]["all_mapping_classes_match_primary_audio_class"])
            self.assertEqual(summary["pool_sizes_by_class_split"]["road_vehicle_traffic"]["train"], 1)
            self.assertEqual(summary["pool_sizes_by_class_split"]["road_vehicle_traffic"]["val"], 0)
            self.assertEqual(json.loads(summary_output.read_text(encoding="utf-8"))["total_pairs"], 2)

    def test_derives_pair_id_when_sample_id_is_missing(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            manifest = root / "val.jsonl"
            pool_output = root / "pools.json"
            mapping_output = root / "mapping.jsonl"
            summary_output = root / "summary.json"
            manifest.write_text(
                json.dumps(
                    {
                        "audio_file": "audio/no_sample_id.wav",
                        "speech_file": "speech/no_sample_id.wav",
                        "speech_text": "A bell rings.",
                        "primary_audio_class": "bells_alarms_phones",
                    }
                )
                + "\n",
                encoding="utf-8",
            )

            build_reference_pools(
                input_manifests={"val": manifest},
                pool_output=pool_output,
                mapping_output=mapping_output,
                summary_output=summary_output,
                splits=["val"],
            )

            mapping = json.loads(mapping_output.read_text(encoding="utf-8").strip())
            self.assertEqual(mapping["pair_id"], "no_sample_id")
            self.assertEqual(mapping["audio_id"], "no_sample_id")
            self.assertEqual(mapping["speech_id"], "no_sample_id")


if __name__ == "__main__":
    unittest.main()
