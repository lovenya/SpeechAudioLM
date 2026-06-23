from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from scripts.export_presentation_bundle import BundleOptions, export_bundle


class ExportPresentationBundleTest(unittest.TestCase):
    def test_filters_inspects_copies_media_and_exports_metrics(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            dataset = root / "dataset"
            inspect_root = root / "results" / "cascaded"
            eval_root = root / "results" / "evals" / "final_validation"
            output = root / "bundle"
            for sample_id in ("a", "b", "c"):
                for category in ("audio", "speech"):
                    path = dataset / category / f"{sample_id}.wav"
                    path.parent.mkdir(parents=True, exist_ok=True)
                    path.write_bytes(f"{category}-{sample_id}".encode())

            t2t = inspect_root / "T2T" / "lane_t2t" / "ttas" / "val" / "manifest.inspect.json"
            t2a = inspect_root / "T2A" / "lane_t2a" / "ttas" / "val" / "manifest.inspect.json"
            a2t = inspect_root / "A2T" / "full_a2t_captioner" / "ttas" / "val" / "manifest.jsonl"
            generated = {}
            for sample_id in ("a", "b", "c"):
                path = t2a.parent / "audio" / f"{sample_id}.wav"
                path.parent.mkdir(parents=True, exist_ok=True)
                path.write_bytes(f"generated-{sample_id}".encode())
                generated[sample_id] = path

            _write_inspect(t2t, "lane_t2t", [_sample("a"), _sample("b"), _sample("c")])
            _write_inspect(
                t2a,
                "lane_t2a",
                [
                    _sample("a", generated["a"]),
                    _sample("b", generated["b"], status="error"),
                    _sample("c", generated["c"]),
                ],
            )
            a2t.parent.mkdir(parents=True)
            a2t.write_text(
                "".join(
                    json.dumps(
                        {
                            "pipeline": "A2T",
                            "task": "A2T",
                            "variant": "full_a2t_captioner",
                            "dataset": "ttas",
                            "split": "val",
                            "captioner": "captioner",
                            "status": "success",
                            "row_index": index,
                            "sample_id": sample_id,
                            "candidate_text": f"generated caption {sample_id}",
                            "generation_seconds": 0.5,
                            "source": {
                                "audio_file": f"audio/{sample_id}.wav",
                                "speech_file": f"speech/{sample_id}.wav",
                                "audio_cap": f"caption {sample_id}",
                                "speech_text": f"speech {sample_id}",
                            },
                        }
                    )
                    + "\n"
                    for index, sample_id in enumerate(("a", "b", "c"))
                ),
                encoding="utf-8",
            )

            metric_dir = eval_root / "T2A" / "lane_t2a" / "target_caption_clap_score" / "T2A" / "ttas" / "cascaded"
            metric_dir.mkdir(parents=True)
            metric_rows = [
                {"sample_id": sample_id, "status": "success", "metrics": {"target_caption_clap_score": index / 10}}
                for index, sample_id in enumerate(("a", "b", "c"), start=1)
            ]
            (metric_dir / "latest_per_sample_val.jsonl").write_text(
                "".join(json.dumps(row) + "\n" for row in metric_rows), encoding="utf-8"
            )
            (metric_dir / "latest_summary_val.json").write_text('{"score": 0.2}\n', encoding="utf-8")
            (metric_dir / "latest_run_config_val.json").write_text('{"mode": "final"}\n', encoding="utf-8")

            result = export_bundle(
                BundleOptions(
                    inspect_root=inspect_root,
                    dataset_root=dataset,
                    eval_roots=(eval_root,),
                    output_dir=output,
                    count=2,
                    metric_plan=_write_metric_plan(root / "metric_plan.json"),
                )
            )

            selected = json.loads((output / "sample_ids.json").read_text())
            filtered = json.loads((output / "inspect" / "T2A" / "lane_t2a.inspect.json").read_text())
            metrics = [json.loads(line) for line in (output / "metrics.jsonl").read_text().splitlines()]
            metric_sources = json.loads((output / "metric_sources.json").read_text())
            metric_artifacts = [
                json.loads(line) for line in (output / "metric_artifacts.jsonl").read_text().splitlines()
            ]
            bundle = json.loads((output / "bundle.json").read_text())
            metric_runs = [
                json.loads(line) for line in (output / "metric_runs.jsonl").read_text().splitlines()
            ]
            metric_values = [
                json.loads(line) for line in (output / "metric_values.jsonl").read_text().splitlines()
            ]
            coverage = json.loads((output / "metric_coverage.json").read_text())
            media_index = json.loads((output / "media_index.json").read_text())
            a2t_inspect = json.loads(
                (output / "inspect" / "A2T" / "full_a2t_captioner.inspect.json").read_text()
            )

            self.assertEqual(selected, ["a", "c"])
            self.assertEqual([sample["sample_id"] for sample in filtered["samples"]], ["a", "c"])
            self.assertEqual(
                filtered["samples"][0]["reference"]["audio_file"],
                "media/references/audio/a.wav",
            )
            self.assertEqual(
                filtered["samples"][0]["final_output"]["file"],
                "media/generated/T2A/lane_t2a/a.wav",
            )
            self.assertTrue((output / "media" / "generated" / "T2A" / "lane_t2a" / "a.wav").is_file())
            self.assertEqual([row["result"]["sample_id"] for row in metrics], ["a", "c"])
            self.assertEqual(metric_sources[0]["selected_row_count"], 2)
            self.assertEqual(metric_artifacts[0]["summary"], {"score": 0.2})
            self.assertFalse((output / "metric_sources").exists())
            self.assertEqual(a2t_inspect["samples"][0]["stages"][0]["task"], "audio_to_text")
            self.assertEqual(
                media_index[0]["direction_roles"]["A2S"]["pipeline_input_audio"],
                "media/references/audio/a.wav",
            )
            self.assertEqual(
                media_index[0]["direction_roles"]["A2S"]["paired_reference_output_audio"],
                "media/references/speech/a.wav",
            )
            self.assertEqual(bundle["selection"]["rejected_before_completion"][0]["sample_id"], "b")
            self.assertEqual(bundle["schema_version"], "presentation_bundle_v2")
            self.assertEqual(metric_runs[0]["metric_id"], "target_caption_clap_score")
            self.assertEqual([row["sample_id"] for row in metric_values], ["a", "c"])
            self.assertEqual(coverage["status_counts"], {"complete": 1, "in_progress": 1})
            self.assertIsNone(coverage["entries"][1]["summary"])
            self.assertEqual(result["media_file_count"], 6)


def _sample(sample_id: str, output: Path | None = None, status: str = "success") -> dict:
    stages = [
        {
            "stage": 1,
            "task": "speech_to_text",
            "model": "model",
            "input": {"type": "speech_audio", "file": f"speech/{sample_id}.wav"},
            "output": {"type": "speech_text", "text": sample_id},
            "error": None if status == "success" else {"message": "failed"},
        }
    ]
    final_output = None
    if output is not None:
        stages.append(
            {
                "stage": 2,
                "task": "text_to_audio",
                "model": "audio-model",
                "input": {"type": "audio_description", "text": sample_id},
                "output": {"type": "audio", "file": str(output)},
                "error": None if status == "success" else {"message": "failed"},
            }
        )
        final_output = {"type": "audio", "file": str(output)} if status == "success" else None
    return {
        "row_index": ord(sample_id) - ord("a"),
        "sample_id": sample_id,
        "status": status,
        "reference": {
            "audio_caption": f"caption {sample_id}",
            "audio_file": f"audio/{sample_id}.wav",
            "speech_file": f"speech/{sample_id}.wav",
            "speech_text": f"speech {sample_id}",
        },
        "stages": stages,
        "final_output": final_output,
        "error": None if status == "success" else {"message": "failed"},
    }


def _write_inspect(path: Path, lane: str, samples: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(
            {
                "schema_version": "1.0",
                "current_stage": 2,
                "lane": lane,
                "pipeline": "S2A",
                "dataset": "ttas",
                "split": "val",
                "status": "completed",
                "stage_definitions": [],
                "summary": {
                    "total": len(samples),
                    "success": sum(sample["status"] == "success" for sample in samples),
                    "failed": sum(sample["status"] != "success" for sample in samples),
                },
                "samples": samples,
            }
        )
        + "\n",
        encoding="utf-8",
    )


def _write_metric_plan(path: Path) -> Path:
    path.write_text(
        json.dumps(
            {
                "name": "test-plan",
                "task_metrics": {
                    "T2A": [
                        {"metric_id": "target_caption_clap_score", "source_families": ["target_caption_clap_score"]},
                        {"metric_id": "utmos"},
                    ]
                },
            }
        ),
        encoding="utf-8",
    )
    return path


if __name__ == "__main__":
    unittest.main()
