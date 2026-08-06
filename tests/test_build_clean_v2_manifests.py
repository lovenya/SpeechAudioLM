import unittest

from scripts.build_clean_v2_manifests import _exclusion_reasons


POLICY = {
    "null_final_score": True,
    "final_score_lte": 3.0,
    "scenario_plausibility_lte": 2.0,
    "target_modality_success_eq": 1.0,
}


class ExclusionPolicyTests(unittest.TestCase):
    def test_keeps_row_above_all_thresholds(self):
        self.assertEqual(_exclusion_reasons(3.1, 3, 2, POLICY), [])

    def test_excludes_null_final_score(self):
        self.assertEqual(
            _exclusion_reasons(None, None, None, POLICY),
            ["null_final_score"],
        )

    def test_excludes_each_threshold_and_records_overlap(self):
        self.assertEqual(
            _exclusion_reasons(3, 2, 1, POLICY),
            [
                "final_score_lte_3",
                "scenario_plausibility_lte_2",
                "target_modality_success_eq_1",
            ],
        )

    def test_target_modality_two_is_not_excluded(self):
        self.assertEqual(_exclusion_reasons(4, 4, 2, POLICY), [])


if __name__ == "__main__":
    unittest.main()
