from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from scripts.assign_audio_classes import assign_primary_audio_class, assign_splits


CONFIG = {
    "assignment_rule": "priority_first_v1",
    "available_assignment_rules": ["priority_first_v1", "priority_with_evidence_guard"],
    "unknown_class": "unknown",
    "source_document": "speechaudio_audio_tag_class_pooling.md",
    "priority_order": [
        "emergency_siren",
        "road_vehicle_traffic",
        "domestic_pets",
        "human_voice_vocal",
        "unknown",
    ],
    "evidence_guard": {
        "max_score_threshold": 0.6,
        "sum_score_threshold": 0.6,
        "strongest_sum_ratio": 0.5,
    },
    "class_map": {
        "emergency_siren": ["Siren", "Alarm"],
        "road_vehicle_traffic": ["Vehicle", "Car", "Engine"],
        "domestic_pets": ["Dog", "Bark"],
        "human_voice_vocal": ["Speech", "Laughter"],
    },
}


class AudioClassAssignmentTest(unittest.TestCase):
    def test_real_config_priority_first_keeps_voice_below_concrete_environmental_classes(self) -> None:
        real_config = json.loads(Path("configs/audio_tag_class_pooling.json").read_text(encoding="utf-8"))

        priority = real_config["priority_order"]
        self.assertLess(priority.index("domestic_pets"), priority.index("human_voice_vocal"))
        self.assertLess(priority.index("road_vehicle_traffic"), priority.index("human_voice_vocal"))
        assignment = assign_primary_audio_class(
            ["Speech", "Dog"],
            {"Speech": 0.95, "Dog": 0.31},
            real_config,
            assignment_rule="priority_first_v1",
        )
        self.assertEqual(assignment.primary_audio_class, "domestic_pets")

    def test_real_config_default_evidence_guard_keeps_stronger_voice_evidence(self) -> None:
        real_config = json.loads(Path("configs/audio_tag_class_pooling.json").read_text(encoding="utf-8"))

        assignment = assign_primary_audio_class(
            ["Speech", "Dog"],
            {"Speech": 0.95, "Dog": 0.31},
            real_config,
        )
        self.assertEqual(assignment.primary_audio_class, "human_voice_vocal")
        self.assertEqual(assignment.debug["assignment_rule"], "priority_with_evidence_guard")

    def test_real_config_maps_new_unknown_tags(self) -> None:
        real_config = json.loads(Path("configs/audio_tag_class_pooling.json").read_text(encoding="utf-8"))

        priority = real_config["priority_order"]
        self.assertLess(priority.index("rain_thunder_weather"), priority.index("nature_ambient"))
        self.assertLess(priority.index("nature_ambient"), priority.index("water_nature"))
        expected = {
            "Whistling": "human_voice_vocal",
            "Whistle": "human_voice_vocal",
            "Bird flight, flapping wings": "birds",
            "Crumpling, crinkling": "indoor_household",
            "Tearing": "indoor_household",
            "Scissors": "tools_machinery_construction",
            "Wood": "tools_machinery_construction",
            "Rub": "tools_machinery_construction",
            "Splinter": "tools_machinery_construction",
            "Rustling leaves": "nature_ambient",
            "Rustle": "nature_ambient",
        }
        for tag, class_name in expected.items():
            with self.subTest(tag=tag):
                assignment = assign_primary_audio_class([tag], {tag: 0.7}, real_config)
                self.assertEqual(assignment.primary_audio_class, class_name)

    def test_real_config_documents_ambiguous_tag_priority(self) -> None:
        real_config = json.loads(Path("configs/audio_tag_class_pooling.json").read_text(encoding="utf-8"))

        mouse = assign_primary_audio_class(["Mouse"], {"Mouse": 0.7}, real_config)
        bell = assign_primary_audio_class(["Bell"], {"Bell": 0.7}, real_config)

        self.assertEqual(mouse.primary_audio_class, "other_animals_insects_farm")
        self.assertEqual(mouse.debug["ambiguous_matched_tags"]["Mouse"], ["other_animals_insects_farm", "office_typing_clock"])
        self.assertEqual(bell.primary_audio_class, "music_party")
        self.assertEqual(bell.debug["ambiguous_matched_tags"]["Bell"], ["music_party", "bells_alarms_phones"])

    def test_priority_assigns_specific_environmental_class_over_voice(self) -> None:
        assignment = assign_primary_audio_class(
            ["Speech", "Dog", "Bark"],
            {"Speech": 0.92, "Dog": 0.71, "Bark": 0.69},
            CONFIG,
        )

        self.assertEqual(assignment.primary_audio_class, "domestic_pets")
        self.assertEqual(assignment.matched_audio_class_tags, ["Dog", "Bark"])
        self.assertEqual(assignment.debug["strongest_class_by_score"], "domestic_pets")

    def test_priority_first_keeps_high_priority_class_and_records_score_override(self) -> None:
        assignment = assign_primary_audio_class(
            ["Vehicle", "Car", "Siren"],
            {"Vehicle": 0.72, "Car": 0.7, "Siren": 0.22},
            CONFIG,
        )

        self.assertEqual(assignment.primary_audio_class, "emergency_siren")
        self.assertEqual(assignment.debug["strongest_class_by_score"], "road_vehicle_traffic")
        self.assertTrue(assignment.debug["priority_overrode_stronger_score"])
        self.assertTrue(assignment.debug["evidence_guard_disagrees"])
        self.assertEqual(assignment.debug["evidence_guard_class"], "road_vehicle_traffic")

    def test_evidence_guard_can_override_weak_priority_candidate(self) -> None:
        assignment = assign_primary_audio_class(
            ["Vehicle", "Car", "Siren"],
            {"Vehicle": 0.72, "Car": 0.7, "Siren": 0.22},
            CONFIG,
            assignment_rule="priority_with_evidence_guard",
        )

        self.assertEqual(assignment.primary_audio_class, "road_vehicle_traffic")
        self.assertEqual(assignment.matched_audio_class_tags, ["Vehicle", "Car"])

    def test_unknown_when_no_retained_tag_matches_class_map(self) -> None:
        assignment = assign_primary_audio_class(["Wind"], {"Wind": 0.8}, CONFIG)

        self.assertEqual(assignment.primary_audio_class, "unknown")
        self.assertEqual(assignment.matched_audio_class_tags, [])
        self.assertEqual(assignment.debug["candidate_class_scores"], {})

    def test_assign_splits_preserves_rows_and_writes_summary_and_unknowns(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            config_path = root / "config.json"
            input_path = root / "ttas_val_panns_cnn14.jsonl"
            output_root = root / "classes"
            summary_path = root / "reports" / "counts.json"
            unknown_path = root / "reports" / "unknown.jsonl"
            config_path.write_text(json.dumps(CONFIG), encoding="utf-8")

            rows = [
                {
                    "sample_id": "a",
                    "audio_file": "a.wav",
                    "retained_audio_tags": ["Speech", "Dog"],
                    "retained_audio_tag_scores": {"Speech": 0.91, "Dog": 0.75},
                    "primary_audio_tag": "Speech",
                },
                {
                    "sample_id": "b",
                    "audio_file": "b.wav",
                    "retained_audio_tags": ["Wind"],
                    "retained_audio_tag_scores": {"Wind": 0.8},
                    "primary_audio_tag": "Wind",
                },
            ]
            input_path.write_text("\n".join(json.dumps(row) for row in rows) + "\n", encoding="utf-8")

            summary = assign_splits(
                config_path=config_path,
                input_manifests={"val": input_path},
                output_root=output_root,
                summary_output=summary_path,
                unknown_output=unknown_path,
                splits=["val"],
                assignment_rule=None,
                include_debug=True,
            )

            output_rows = [
                json.loads(line)
                for line in (output_root / "ttas_val_panns_cnn14_classes.jsonl").read_text(encoding="utf-8").splitlines()
            ]
            self.assertEqual(output_rows[0]["sample_id"], "a")
            self.assertEqual(output_rows[0]["primary_audio_tag"], "Speech")
            self.assertEqual(output_rows[0]["primary_audio_class"], "domestic_pets")
            self.assertEqual(output_rows[0]["matched_audio_class_tags"], ["Dog"])
            self.assertIn("audio_class_assignment_debug", output_rows[0])
            self.assertEqual(output_rows[1]["primary_audio_class"], "unknown")
            self.assertEqual(summary["total_samples_processed"], 2)
            self.assertEqual(summary["unknown_count"], 1)
            self.assertEqual(summary["primary_audio_class_counts"], {"domestic_pets": 1, "unknown": 1})
            self.assertEqual(json.loads(summary_path.read_text(encoding="utf-8"))["splits"]["val"]["processed"], 2)
            unknown_rows = [json.loads(line) for line in unknown_path.read_text(encoding="utf-8").splitlines()]
            self.assertEqual([row["sample_id"] for row in unknown_rows], ["b"])


if __name__ == "__main__":
    unittest.main()
