from __future__ import annotations

import subprocess
import unittest
from pathlib import Path


SCHEMA_T2T_RUN_SCRIPTS = (
    Path("submission_scripts/run_tamia_schema_t2t_qwen_array.sbatch"),
    Path("submission_scripts/run_tamia_schema_t2t_llama_array.sbatch"),
)
SCHEMA_T2T_MERGE_SCRIPTS = (
    Path("submission_scripts/merge_tamia_schema_t2t_llama_array.sbatch"),
)
SCHEMA_T2T_HELPERS = (
    Path("submission_scripts/lib_resource_monitor.sh"),
)
SCHEMA_T2T_SCRIPTS = SCHEMA_T2T_RUN_SCRIPTS + SCHEMA_T2T_MERGE_SCRIPTS


class SubmissionScriptShapeTest(unittest.TestCase):
    def test_schema_t2t_scripts_use_v1_stage_shards(self) -> None:
        for script in SCHEMA_T2T_RUN_SCRIPTS:
            with self.subTest(script=script.as_posix()):
                text = script.read_text(encoding="utf-8")

                self.assertIn("--results-schema v1", text)
                self.assertIn("--shard-index", text)
                self.assertIn("--num-shards", text)
                self.assertIn("--run-id", text)
                self.assertIn("SALM_RESULTS_ROOT", text)
                self.assertIn("SALM_LOGS_ROOT", text)
                self.assertIn("upstream_model=", text)
                self.assertNotIn("--overwrite", text)
                self.assertNotIn("--run-name", text)
                self.assertNotIn("scripts/merge_cascaded_shards.py", text)

    def test_schema_t2t_merge_scripts_use_stage_merge_helper(self) -> None:
        for script in SCHEMA_T2T_MERGE_SCRIPTS:
            with self.subTest(script=script.as_posix()):
                text = script.read_text(encoding="utf-8")

                self.assertIn("scripts/merge_cascaded_stage_shards.py", text)
                self.assertIn("--run-dir", text)
                self.assertIn("--expected-count", text)
                self.assertIn("--media-output none", text)
                self.assertIn("SALM_RESULTS_ROOT", text)
                self.assertNotIn("scripts/merge_cascaded_shards.py", text)

    def test_schema_t2t_scripts_parse_as_bash(self) -> None:
        subprocess.run(
            [
                "bash",
                "-n",
                *(script.as_posix() for script in SCHEMA_T2T_SCRIPTS + SCHEMA_T2T_HELPERS),
            ],
            check=True,
        )


if __name__ == "__main__":
    unittest.main()
