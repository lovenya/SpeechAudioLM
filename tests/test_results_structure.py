from __future__ import annotations

import json
import tempfile
import unittest
from datetime import datetime, timezone
from pathlib import Path
from unittest.mock import patch

from results_structure import (
    cascaded_pipeline_run_paths,
    cascaded_stage_run_paths,
    create_run_directories,
    generate_run_id,
)


class ResultsStructureTest(unittest.TestCase):
    def test_run_id_uses_toronto_minute_time(self) -> None:
        self.assertEqual(
            generate_run_id(datetime(2026, 6, 16, 18, 37, 59, tzinfo=timezone.utc)),
            "20260616_1437",
        )

    def test_stage_paths_follow_cascaded_design(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            with patch("results_structure.root_path") as root_path:
                root_path.side_effect = lambda name: {
                    "results_root": root / "results",
                    "logs_root": root / "logs",
                    "temporary_results_root": root / "temporary" / "results",
                    "temporary_logs_root": root / "temporary" / "logs",
                }[name]

                t2t = cascaded_stage_run_paths(
                    stage="t2t",
                    direction="a2s",
                    model="llama3_3b_instruct",
                    upstream_stage="a2t",
                    upstream_model="af_next_aac",
                    dataset="ttas",
                    split="val",
                    run_id="20260616_1437",
                )
                t2s = cascaded_stage_run_paths(
                    stage="t2s",
                    direction="a2s",
                    model="f5_tts",
                    upstream_stage="t2t",
                    upstream_model="llama3_3b_instruct",
                    dataset="ttas",
                    split="val",
                    run_id="20260616_1500",
                    temporary=True,
                )

        self.assertEqual(
            t2t.result_relative.as_posix(),
            "results/cascaded/stages/t2t/a2s/llama3_3b_instruct/from-a2t/af_next_aac/ttas/val/20260616_1437",
        )
        self.assertEqual(
            t2t.logs_relative.as_posix(),
            "logs/cascaded/stages/t2t/a2s/llama3_3b_instruct/from-a2t/af_next_aac/ttas/val/20260616_1437",
        )
        self.assertEqual(
            t2s.result_relative.as_posix(),
            "temporary/results/cascaded/stages/t2s/f5_tts/from-t2t/llama3_3b_instruct/ttas/val/20260616_1500",
        )

    def test_rejects_incompatible_t2t_direction_and_upstream_stage(self) -> None:
        with self.assertRaisesRegex(ValueError, "requires upstream stage 's2t'"):
            cascaded_stage_run_paths(
                stage="t2t",
                direction="s2a",
                model="llama3_3b_instruct",
                upstream_stage="a2t",
                upstream_model="af_next_aac",
                dataset="ttas",
                split="val",
                run_id="20260616_1437",
            )

    def test_pipeline_paths_and_create_only_directories(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            with patch("results_structure.root_path") as root_path:
                root_path.side_effect = lambda name: {
                    "results_root": root / "results",
                    "logs_root": root / "logs",
                    "temporary_results_root": root / "temporary" / "results",
                    "temporary_logs_root": root / "temporary" / "logs",
                }[name]
                paths = cascaded_pipeline_run_paths(
                    direction="a2s",
                    pipeline_name="afn_llama_f5",
                    dataset="ttas",
                    split="val",
                    run_id="20260616_1530",
                )
                create_run_directories(paths)

                self.assertTrue(paths.result.is_dir())
                self.assertTrue(paths.logs.is_dir())
                with self.assertRaises(FileExistsError):
                    create_run_directories(paths)

    def test_create_run_directories_rolls_back_result_dir_if_logs_fail(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            result = root / "results" / "run"
            logs = root / "logs" / "run"
            logs.parent.mkdir(parents=True)
            logs.write_text("not a directory")

            from results_structure import RunPaths

            with self.assertRaises(FileExistsError):
                create_run_directories(
                    RunPaths(
                        result=result,
                        logs=logs,
                        result_relative=Path("results/run"),
                        logs_relative=Path("logs/run"),
                    )
                )

            self.assertFalse(result.exists())

    def test_project_paths_prefer_design_singular_roots_with_plural_env_fallback(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            config = root / "project_paths.json"
            config.write_text(
                json.dumps(
                    {
                        "schema_version": 2,
                        "roots": {
                            "repo_root": ".",
                            "results_root": "results",
                            "logs_root": "logs",
                            "temporary_results_root": "temporary/results",
                            "temporary_logs_root": "temporary/logs",
                            "exports_root": "exports",
                            "environment_root": "envs",
                            "model_root": "models",
                            "cache_root": "cache",
                        },
                        "models": {"demo": "demo-model"},
                        "environments": {"demo": "demo-env"},
                    }
                ),
                encoding="utf-8",
            )

            from project_paths import environment_path, load_project_paths, model_path

            load_project_paths.cache_clear()
            with patch.dict("os.environ", {"SALM_REPO_ROOT": str(root), "SALM_MODELS_ROOT": str(root / "legacy-models")}):
                self.assertEqual(model_path("demo", config), root / "legacy-models" / "demo-model")
                self.assertEqual(environment_path("demo", config), root / "envs" / "demo-env")


if __name__ == "__main__":
    unittest.main()
