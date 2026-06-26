#!/usr/bin/env python
"""Export a portable, inspect-oriented subset of cascaded results and evals."""

from __future__ import annotations

import argparse
import copy
import json
import statistics
import shutil
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable


TASK_NAMES = {"A2T", "S2T", "T2T", "T2A", "T2S"}


@dataclass(frozen=True)
class BundleOptions:
    inspect_root: Path
    dataset_root: Path
    eval_roots: tuple[Path, ...]
    output_dir: Path
    count: int = 100
    source_inspect: Path | None = None
    geval_root: Path | None = None
    geval_a2s_lane: str = "full_a2s_afn_qwen32_t2t"
    geval_s2a_lane: str = "full_s2a_whisper_qwen32_t2t"
    metric_plan: Path | None = None
    overwrite: bool = False
    make_zip: bool = False


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--inspect-root", type=Path, required=True)
    parser.add_argument("--dataset-root", type=Path, required=True)
    parser.add_argument("--eval-root", type=Path, action="append", default=[])
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--count", type=int, default=100)
    parser.add_argument("--source-inspect", type=Path)
    parser.add_argument("--geval-root", type=Path)
    parser.add_argument("--geval-a2s-lane", default="full_a2s_afn_qwen32_t2t")
    parser.add_argument("--geval-s2a-lane", default="full_s2a_whisper_qwen32_t2t")
    parser.add_argument("--metric-plan", type=Path)
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--zip", action="store_true", dest="make_zip")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    result = export_bundle(
        BundleOptions(
            inspect_root=args.inspect_root,
            dataset_root=args.dataset_root,
            eval_roots=tuple(args.eval_root),
            output_dir=args.output_dir,
            count=args.count,
            source_inspect=args.source_inspect,
            geval_root=args.geval_root,
            geval_a2s_lane=args.geval_a2s_lane,
            geval_s2a_lane=args.geval_s2a_lane,
            metric_plan=args.metric_plan,
            overwrite=args.overwrite,
            make_zip=args.make_zip,
        )
    )
    print(
        f"bundle={result['output_dir']} samples={result['sample_count']} "
        f"lanes={result['lane_count']} metric_sources={result['metric_source_count']} "
        f"media_files={result['media_file_count']}"
    )


def export_bundle(options: BundleOptions) -> dict[str, Any]:
    if options.count <= 0:
        raise ValueError("count must be positive")

    inspections = _discover_inspections(options.inspect_root)
    if not inspections:
        raise FileNotFoundError(f"no manifest.inspect.json files under {options.inspect_root}")
    source_path = options.source_inspect or _choose_source_inspect(inspections)
    if source_path not in inspections:
        source_path = source_path.resolve()
        inspections = {path.resolve(): value for path, value in inspections.items()}
    if source_path not in inspections:
        raise FileNotFoundError(f"source inspect is not under inspect root: {source_path}")

    selected_ids, rejected = _select_sample_ids(inspections, source_path, options.count)
    if len(selected_ids) != options.count:
        raise RuntimeError(
            f"only {len(selected_ids)} samples succeeded in every inspect manifest; "
            f"requested {options.count}"
        )

    output_dir = options.output_dir
    if output_dir.exists():
        if not options.overwrite:
            raise FileExistsError(f"output directory exists: {output_dir}")
        shutil.rmtree(output_dir)
    output_dir.mkdir(parents=True)

    selected = set(selected_ids)
    media_map: dict[Path, str] = {}
    lane_catalog = []
    canonical_samples: dict[str, dict[str, Any]] = {}

    for path, inspection in inspections.items():
        filtered = copy.deepcopy(inspection)
        samples = [sample for sample in filtered.get("samples", []) if sample.get("sample_id") in selected]
        samples.sort(key=lambda sample: selected_ids.index(str(sample["sample_id"])))
        lane = str(filtered.get("lane") or path.parent.parent.parent.parent.name)
        task = path.relative_to(options.inspect_root).parts[0]
        for sample in samples:
            _rewrite_sample_media(
                sample,
                dataset_root=options.dataset_root,
                output_dir=output_dir,
                task=task,
                lane=lane,
                media_map=media_map,
            )
            canonical_samples.setdefault(
                str(sample["sample_id"]),
                {
                    "sample_id": sample["sample_id"],
                    "row_index": sample.get("row_index"),
                    "reference": copy.deepcopy(sample.get("reference") or {}),
                },
            )
        filtered["samples"] = samples
        filtered["summary"] = {"total": len(samples), "success": len(samples), "failed": 0}
        source_bundle_metadata = filtered.get("presentation_bundle") or {}
        filtered["presentation_bundle"] = {
            **source_bundle_metadata,
            "source_manifest": str(path),
            "selected_sample_count": len(samples),
            "source_status": filtered.get("status"),
        }
        filtered["status"] = "completed"
        relative_output = Path("inspect") / task / f"{lane}.inspect.json"
        _write_json(output_dir / relative_output, filtered)
        lane_catalog.append(
            {
                "lane": lane,
                "pipeline": filtered.get("pipeline"),
                "task": task,
                "current_stage": filtered.get("current_stage"),
                "inspect_file": relative_output.as_posix(),
                "stage_definitions": filtered.get("stage_definitions") or [],
            }
        )

    ordered_samples = [canonical_samples[sample_id] for sample_id in selected_ids]
    _write_json(output_dir / "sample_ids.json", selected_ids)
    _write_json(output_dir / "samples.json", ordered_samples)
    _write_json(output_dir / "media_index.json", [_media_roles(sample) for sample in ordered_samples])
    _write_json(output_dir / "lanes.json", lane_catalog)
    _write_json(
        output_dir / "audio_class_summary.json",
        _audio_class_summary(
            selected_samples=ordered_samples,
            source_samples=inspections[source_path].get("samples") or [],
        ),
    )

    metric_catalog, metric_rows, metric_runs, metric_summaries, metric_values, definitions = _export_metrics(
        eval_roots=options.eval_roots,
        selected_ids=selected,
        output_dir=output_dir,
        media_map=media_map,
    )
    if options.geval_root:
        _export_geval(
            root=options.geval_root,
            selected_ids=selected,
            direction_lanes={
                "audio_to_speech": options.geval_a2s_lane,
                "speech_to_audio": options.geval_s2a_lane,
            },
            runs=metric_runs,
            summaries=metric_summaries,
            values=metric_values,
            definitions=definitions,
        )
    _write_jsonl(output_dir / "metrics.jsonl", metric_rows)
    _write_json(output_dir / "metric_sources.json", metric_catalog)
    _write_json(
        output_dir / "metric_definitions.json",
        sorted(definitions.values(), key=lambda row: row["metric_id"]),
    )
    _write_jsonl(output_dir / "metric_runs.jsonl", metric_runs)
    _write_jsonl(output_dir / "metric_summaries.jsonl", metric_summaries)
    _write_jsonl(output_dir / "metric_values.jsonl", metric_values)
    metric_coverage = _build_metric_coverage(
        lane_catalog=lane_catalog,
        metric_runs=metric_runs,
        plan=_read_json(options.metric_plan) if options.metric_plan else None,
    )
    _write_json(output_dir / "metric_coverage.json", metric_coverage)
    logical_runs, logical_summaries, logical_values = _merge_logical_metrics(
        metric_runs=metric_runs,
        metric_summaries=metric_summaries,
        metric_values=metric_values,
        metric_coverage=metric_coverage,
    )
    _write_jsonl(output_dir / "logical_metric_runs.jsonl", logical_runs)
    _write_jsonl(output_dir / "logical_metric_summaries.jsonl", logical_summaries)
    _write_jsonl(output_dir / "logical_metric_values.jsonl", logical_values)
    if options.geval_root:
        _write_json(output_dir / "geval_evaluator_metadata.json", _geval_evaluator_metadata())

    manifest = {
        "schema_version": "presentation_bundle_v2",
        "created_at": datetime.now(timezone.utc).isoformat(),
        "selection": {
            "strategy": "source_order_first_successful_across_all_inspect_manifests",
            "requested_count": options.count,
            "selected_count": len(selected_ids),
            "source_inspect": str(source_path),
            "rejected_before_completion": rejected,
        },
        "dataset_root_at_export": str(options.dataset_root),
        "inspect_root_at_export": str(options.inspect_root),
        "eval_roots_at_export": [str(path) for path in options.eval_roots],
        "geval_root_at_export": str(options.geval_root) if options.geval_root else None,
        "files": {
            "sample_ids": "sample_ids.json",
            "samples": "samples.json",
            "media_index": "media_index.json",
            "lanes": "lanes.json",
            "audio_class_summary": "audio_class_summary.json",
            "metrics": "metrics.jsonl",
            "metric_sources": "metric_sources.json",
            "metric_artifacts": "metric_artifacts.jsonl",
            "metric_definitions": "metric_definitions.json",
            "metric_runs": "metric_runs.jsonl",
            "metric_summaries": "metric_summaries.jsonl",
            "metric_values": "metric_values.jsonl",
            "metric_coverage": "metric_coverage.json",
            "logical_metric_runs": "logical_metric_runs.jsonl",
            "logical_metric_summaries": "logical_metric_summaries.jsonl",
            "logical_metric_values": "logical_metric_values.jsonl",
            "geval_evaluator_metadata": "geval_evaluator_metadata.json" if options.geval_root else None,
            "inspect_root": "inspect",
            "media_root": "media",
        },
        "counts": {
            "samples": len(selected_ids),
            "lanes": len(lane_catalog),
            "metric_sources": len(metric_catalog),
            "metric_rows": len(metric_rows),
            "metric_runs": len(metric_runs),
            "metric_values": len(metric_values),
            "metric_coverage_entries": len(metric_coverage["entries"]),
            "logical_metric_runs": len(logical_runs),
            "logical_metric_values": len(logical_values),
            "media_files": len(media_map),
        },
        "metric_sources": metric_catalog,
    }
    _write_json(output_dir / "bundle.json", manifest)

    archive = None
    if options.make_zip:
        archive = shutil.make_archive(str(output_dir), "zip", root_dir=output_dir.parent, base_dir=output_dir.name)

    return {
        "output_dir": str(output_dir),
        "archive": archive,
        "sample_count": len(selected_ids),
        "lane_count": len(lane_catalog),
        "metric_source_count": len(metric_catalog),
        "metric_row_count": len(metric_rows),
        "media_file_count": len(media_map),
    }


def _choose_source_inspect(inspections: dict[Path, dict[str, Any]]) -> Path:
    return sorted(
        inspections,
        key=lambda path: (-len(inspections[path].get("samples") or []), str(path)),
    )[0]


def _discover_inspections(root: Path) -> dict[Path, dict[str, Any]]:
    inspections = {path: _read_json(path) for path in sorted(root.rglob("manifest.inspect.json"))}
    for task in ("A2T", "S2T"):
        for path in sorted((root / task).glob("full_*/ttas/val/manifest.jsonl")):
            lane = path.parents[2].name
            if "_shard" in lane or "_corrected" in lane:
                continue
            inspection_path = path.with_name("manifest.inspect.json")
            if inspection_path in inspections:
                continue
            inspections[path] = _inspection_from_legacy_manifest(path, task, lane)
    return inspections


def _inspection_from_legacy_manifest(path: Path, task: str, lane: str) -> dict[str, Any]:
    rows = list(_read_jsonl(path))
    if not rows:
        raise RuntimeError(f"empty canonical manifest: {path}")
    first = rows[0]
    if task == "A2T":
        model = first.get("captioner") or _legacy_backend(first)
        stage_task = "audio_to_text"
        input_type = "environmental_audio"
        input_key = "audio_file"
        output_type = "audio_caption"
    else:
        model = first.get("asr") or _legacy_backend(first)
        stage_task = "speech_to_text"
        input_type = "speech_audio"
        input_key = "speech_file"
        output_type = "speech_text"

    samples = []
    for row in rows:
        source = row.get("source") or {}
        status = row.get("status") or "error"
        error = row.get("error")
        if error is None and status != "success":
            error = {"message": row.get("error_message") or "legacy manifest row failed"}
        samples.append(
            {
                "row_index": row.get("row_index"),
                "sample_id": row.get("sample_id"),
                "status": status,
                "reference": {
                    key: value
                    for key, value in {
                        "audio_caption": source.get("audio_cap") or source.get("audio_caption"),
                        "audio_file": source.get("audio_file"),
                        "speech_file": source.get("speech_file"),
                        "speech_text": source.get("speech_text") or source.get("speech_transcript"),
                    }.items()
                    if value is not None
                },
                "stages": [
                    {
                        "stage": 1,
                        "task": stage_task,
                        "model": model,
                        "input": {"type": input_type, "file": source.get(input_key)},
                        "prompt": None,
                        "inference_seconds": row.get("generation_seconds"),
                        "output": (
                            {"type": output_type, "text": row.get("candidate_text")}
                            if status == "success"
                            else None
                        ),
                        "error": error,
                    }
                ],
                "final_output": None,
                "error": error,
            }
        )

    success = sum(sample["status"] == "success" for sample in samples)
    return {
        "schema_version": "1.0",
        "current_stage": 1,
        "lane": lane,
        "pipeline": task,
        "dataset": first.get("dataset"),
        "split": first.get("split"),
        "config_file": None,
        "created_at": None,
        "started_at": None,
        "completed_at": None,
        "status": "completed" if success == len(samples) else "completed_with_errors",
        "stage_definitions": [
            {
                "stage": 1,
                "task": stage_task,
                "model": model,
                "checkpoint": None,
                "dtype": None,
                "prompt_template": None,
                "generation": {},
                "runtime": {
                    "device": None,
                    "model_load_seconds": first.get("model_load_seconds_total"),
                },
            }
        ],
        "summary": {"total": len(samples), "success": success, "failed": len(samples) - success},
        "samples": samples,
        "presentation_bundle": {"synthesized_from_legacy_manifest": True},
    }


def _legacy_backend(row: dict[str, Any]) -> str | None:
    stages = (row.get("model_extra") or {}).get("stages") or []
    return stages[0].get("backend") if stages else None


def _select_sample_ids(
    inspections: dict[Path, dict[str, Any]], source_path: Path, count: int
) -> tuple[list[str], list[dict[str, Any]]]:
    status_by_path = {
        path: {str(sample.get("sample_id")): sample.get("status") for sample in data.get("samples", [])}
        for path, data in inspections.items()
    }
    selected: list[str] = []
    rejected: list[dict[str, Any]] = []
    for sample in inspections[source_path].get("samples", []):
        sample_id = str(sample.get("sample_id"))
        failures = [
            str(path)
            for path, statuses in status_by_path.items()
            if statuses.get(sample_id) != "success"
        ]
        if failures:
            rejected.append({"sample_id": sample_id, "failed_or_missing_manifests": failures})
            continue
        selected.append(sample_id)
        if len(selected) == count:
            break
    return selected, rejected


def _rewrite_sample_media(
    sample: dict[str, Any],
    *,
    dataset_root: Path,
    output_dir: Path,
    task: str,
    lane: str,
    media_map: dict[Path, str],
) -> None:
    reference = sample.get("reference") or {}
    for key, category in (("audio_file", "audio"), ("speech_file", "speech")):
        value = reference.get(key)
        if isinstance(value, str):
            source = _resolve_media(value, dataset_root)
            relative = Path("media") / "references" / category / source.name
            reference[key] = _copy_media(source, relative, output_dir, media_map)

    def visit(value: Any) -> None:
        if isinstance(value, dict):
            for key, child in list(value.items()):
                if key == "file" and isinstance(child, str):
                    source = _resolve_media(child, dataset_root)
                    if _is_relative_to(source, dataset_root):
                        category = "speech" if "speech" in source.relative_to(dataset_root).parts else "audio"
                        relative = Path("media") / "references" / category / source.name
                    else:
                        relative = Path("media") / "generated" / task / lane / source.name
                    value[key] = _copy_media(source, relative, output_dir, media_map)
                else:
                    visit(child)
        elif isinstance(value, list):
            for child in value:
                visit(child)

    visit(sample.get("stages") or [])
    visit(sample.get("final_output"))


def _media_roles(sample: dict[str, Any]) -> dict[str, Any]:
    reference = sample.get("reference") or {}
    environmental_audio = reference.get("audio_file")
    speech_audio = reference.get("speech_file")
    return {
        "sample_id": sample.get("sample_id"),
        "paired_dataset_media": {
            "environmental_audio": environmental_audio,
            "speech_audio": speech_audio,
        },
        "direction_roles": {
            "A2S": {
                "pipeline_input_audio": environmental_audio,
                "paired_reference_output_audio": speech_audio,
                "input_reference_text": reference.get("audio_caption"),
                "target_reference_text": reference.get("speech_text"),
            },
            "S2A": {
                "pipeline_input_audio": speech_audio,
                "paired_reference_output_audio": environmental_audio,
                "input_reference_text": reference.get("speech_text"),
                "target_reference_text": reference.get("audio_caption"),
            },
        },
    }


def _audio_class_summary(
    *,
    selected_samples: list[dict[str, Any]],
    source_samples: list[dict[str, Any]],
) -> dict[str, Any]:
    def summarize(samples: list[dict[str, Any]]) -> tuple[dict[str, int], int]:
        counts: dict[str, int] = {}
        missing = 0
        for sample in samples:
            metadata = ((sample.get("reference") or {}).get("audio_metadata") or {})
            audio_class = metadata.get("primary_audio_class")
            if not audio_class:
                missing += 1
                continue
            key = str(audio_class)
            counts[key] = counts.get(key, 0) + 1
        return counts, missing

    selected_counts, selected_missing = summarize(selected_samples)
    source_counts, source_missing = summarize(source_samples)
    selected_total = len(selected_samples)
    source_total = len(source_samples)
    classes = []
    for audio_class in sorted(set(selected_counts) | set(source_counts)):
        selected_count = selected_counts.get(audio_class, 0)
        source_count = source_counts.get(audio_class, 0)
        classes.append(
            {
                "audio_class": audio_class,
                "selected_count": selected_count,
                "selected_percent": selected_count * 100.0 / selected_total if selected_total else 0.0,
                "source_count": source_count,
                "source_percent": source_count * 100.0 / source_total if source_total else 0.0,
            }
        )
    classes.sort(key=lambda row: (-row["selected_count"], -row["source_count"], row["audio_class"]))
    return {
        "schema_version": "audio_class_summary_v1",
        "selected_total": selected_total,
        "selected_missing_class": selected_missing,
        "source_total": source_total,
        "source_missing_class": source_missing,
        "classes": classes,
    }


def _export_metrics(
    *,
    eval_roots: Iterable[Path],
    selected_ids: set[str],
    output_dir: Path,
    media_map: dict[Path, str],
) -> tuple[
    list[dict[str, Any]],
    list[dict[str, Any]],
    list[dict[str, Any]],
    list[dict[str, Any]],
    list[dict[str, Any]],
    dict[str, dict[str, Any]],
]:
    catalog: list[dict[str, Any]] = []
    consolidated: list[dict[str, Any]] = []
    artifacts: list[dict[str, Any]] = []
    runs: list[dict[str, Any]] = []
    summaries: list[dict[str, Any]] = []
    values: list[dict[str, Any]] = []
    definitions: dict[str, dict[str, Any]] = {}
    for eval_root in eval_roots:
        if not eval_root.exists():
            continue
        for source in sorted(eval_root.rglob("latest_per_sample_val.jsonl")):
            metadata = _metric_metadata(eval_root, source)
            source_id = metadata["source_id"]
            rows = []
            for row in _read_jsonl(source):
                if str(row.get("sample_id")) not in selected_ids:
                    continue
                rewritten = copy.deepcopy(row)
                _rewrite_known_media_paths(rewritten, media_map)
                rows.append(rewritten)
                consolidated.append({**metadata, "result": rewritten})
            summary_path = source.with_name("latest_summary_val.json")
            run_config_path = source.with_name("latest_run_config_val.json")
            summary_csv_path = source.with_name("latest_summary_val.csv")
            report_path = source.with_name("latest_report_val.md")
            artifact = {
                **metadata,
                "summary": _read_json(summary_path) if summary_path.is_file() else None,
                "run_config": _read_json(run_config_path) if run_config_path.is_file() else None,
                "summary_csv": summary_csv_path.read_text(encoding="utf-8") if summary_csv_path.is_file() else None,
                "report_markdown": report_path.read_text(encoding="utf-8") if report_path.is_file() else None,
            }
            artifacts.append(artifact)
            catalog.append(
                {
                    **metadata,
                    "source_file": str(source),
                    "selected_row_count": len(rows),
                    "has_summary": artifact["summary"] is not None,
                    "has_run_config": artifact["run_config"] is not None,
                    "has_summary_csv": artifact["summary_csv"] is not None,
                    "has_report": artifact["report_markdown"] is not None,
                }
            )
            metric_ids = sorted(
                {
                    str(metric_id)
                    for row in rows
                    for metric_id in (row.get("metrics") or {}).keys()
                }
            ) or ([str(metadata["metric"])] if metadata.get("metric") else [])
            for metric_id in metric_ids:
                run_id = _safe_name(f"{source_id}__{metric_id}")
                renderer = "nli" if metric_id.startswith("nli") else "scalar"
                definitions.setdefault(
                    metric_id,
                    {
                        "metric_id": metric_id,
                        "label": metric_id.replace("_", " ").title(),
                        "renderer": renderer,
                        "level": "per_sample",
                        "description": None,
                        "range": None,
                        "better": None,
                    },
                )
                runs.append(
                    {
                        "metric_run_id": run_id,
                        "metric_id": metric_id,
                        "renderer": renderer,
                        "status": "complete",
                        **metadata,
                        "selected_row_count": len(rows),
                        "source_file": str(source),
                    }
                )
                summaries.append(
                    {
                        "metric_run_id": run_id,
                        "metric_id": metric_id,
                        "scope": "source_summary",
                        "summary": artifact["summary"],
                    }
                )
                for row in rows:
                    predictions = row.get("predictions") or {}
                    values.append(
                        {
                            "metric_run_id": run_id,
                            "metric_id": metric_id,
                            "sample_id": row.get("sample_id"),
                            "status": row.get("status"),
                            "value": (row.get("metrics") or {}).get(metric_id),
                            "details": predictions.get(metric_id) if renderer == "nli" else row,
                        }
                    )
    _write_jsonl(output_dir / "metric_artifacts.jsonl", artifacts)
    return catalog, consolidated, runs, summaries, values, definitions


def _export_geval(
    *,
    root: Path,
    selected_ids: set[str],
    direction_lanes: dict[str, str],
    runs: list[dict[str, Any]],
    summaries: list[dict[str, Any]],
    values: list[dict[str, Any]],
    definitions: dict[str, dict[str, Any]],
) -> None:
    for variant in ("plausibility", "reference_grounded"):
        variant_root = root / variant
        candidates = (
            sorted(path for path in variant_root.iterdir() if path.is_dir())
            if variant_root.is_dir()
            else []
        )
        if not candidates:
            continue
        run_dir = candidates[-1]
        output_path = run_dir / "output.jsonl"
        summary_path = run_dir / "summary.json"
        if not output_path.is_file() or not summary_path.is_file():
            continue
        official_summary = _read_json(summary_path)
        rows_by_direction: dict[str, list[dict[str, Any]]] = {}
        selected_by_direction: dict[str, list[dict[str, Any]]] = {}
        for row in _read_jsonl(output_path):
            direction = str(row.get("direction"))
            rows_by_direction.setdefault(direction, []).append(row)
            if str(row.get("sample_id")) in selected_ids:
                selected_by_direction.setdefault(direction, []).append(row)

        metric_id = str(official_summary.get("metric") or f"g_eval_{variant}")
        definitions.setdefault(
            metric_id,
            {
                "metric_id": metric_id,
                "label": f"G-Eval {variant.replace('_', ' ').title()}",
                "renderer": "geval",
                "level": "judge",
                "description": "Prometheus criterion-level text judgment with weighted final score.",
                "range": [1, 5],
                "better": "higher",
            },
        )
        for direction, all_rows in sorted(rows_by_direction.items()):
            lane = direction_lanes.get(direction)
            run_id = _safe_name(f"geval__{variant}__{direction}__{lane or 'unknown_lane'}")
            selected_rows = selected_by_direction.get(direction, [])
            first = all_rows[0] if all_rows else {}
            runs.append(
                {
                    "metric_run_id": run_id,
                    "metric_id": metric_id,
                    "renderer": "geval",
                    "status": "complete",
                    "task": "T2T",
                    "lane": lane,
                    "direction": direction,
                    "evaluation_variant": variant,
                    "judge_model": official_summary.get("judge_model"),
                    "evaluator_version": official_summary.get("evaluator_version"),
                    "prompt_version": official_summary.get("prompt_version"),
                    "scoring_policy_version": official_summary.get("scoring_policy_version"),
                    "aggregation_policy_version": official_summary.get("aggregation_policy_version"),
                    "source_directory": str(run_dir),
                    "source_row_schema_version": first.get("schema_version"),
                    "source_summary_schema_version": official_summary.get("schema_version"),
                    "selected_row_count": len(selected_rows),
                }
            )
            summaries.append(
                {
                    "metric_run_id": run_id,
                    "metric_id": metric_id,
                    "scope": "direction",
                    "direction": direction,
                    "statistics": _geval_direction_statistics(all_rows),
                    "official_variant_summary": official_summary,
                }
            )
            for row in selected_rows:
                values.append(
                    {
                        "metric_run_id": run_id,
                        "metric_id": metric_id,
                        "sample_id": row.get("sample_id"),
                        "pair_id": row.get("pair_id"),
                        "status": row.get("status"),
                        "value": row.get("final_score"),
                        "details": row,
                    }
                )


def _geval_direction_statistics(rows: list[dict[str, Any]]) -> dict[str, Any]:
    scores = [
        float(row["final_score"])
        for row in rows
        if isinstance(row.get("final_score"), (int, float))
    ]
    criterion_scores: dict[str, list[float]] = {}
    for row in rows:
        for name, result in (row.get("criteria") or {}).items():
            score = result.get("score") if isinstance(result, dict) else None
            if isinstance(score, (int, float)):
                criterion_scores.setdefault(str(name), []).append(float(score))
    return {
        "total_rows": len(rows),
        "status_counts": _counts(str(row.get("status")) for row in rows),
        "final_score": _describe(scores, len(rows)),
        "criteria": {
            name: _describe(values, len(rows))
            for name, values in sorted(criterion_scores.items())
        },
    }


def _describe(values: list[float], eligible_count: int) -> dict[str, Any]:
    if not values:
        return {"count": 0, "null_count": eligible_count}
    return {
        "count": len(values),
        "null_count": eligible_count - len(values),
        "mean": statistics.fmean(values),
        "std": statistics.pstdev(values),
        "min": min(values),
        "max": max(values),
        "median": statistics.median(values),
    }


def _counts(values: Iterable[str]) -> dict[str, int]:
    counts: dict[str, int] = {}
    for value in values:
        counts[value] = counts.get(value, 0) + 1
    return counts


def _build_metric_coverage(
    *,
    lane_catalog: list[dict[str, Any]],
    metric_runs: list[dict[str, Any]],
    plan: dict[str, Any] | None,
) -> dict[str, Any]:
    if not plan:
        return {"schema_version": "metric_coverage_v1", "entries": [], "status_counts": {}}

    lanes_by_task: dict[str, list[str]] = {}
    for lane in lane_catalog:
        lanes_by_task.setdefault(str(lane.get("task")), []).append(str(lane.get("lane")))

    expected = []
    for task, metrics in (plan.get("task_metrics") or {}).items():
        for lane in sorted(set(lanes_by_task.get(task, []))):
            for metric in metrics:
                expected.append({"task": task, "lane": lane, **metric})
    expected.extend(plan.get("specific_runs") or [])

    entries = []
    for item in expected:
        source_families = item.get("source_families") or [item["metric_id"]]
        matches = [
            run
            for run in metric_runs
            if run.get("task") == item.get("task")
            and run.get("lane") == item.get("lane")
            and (run.get("metric") in source_families or run.get("metric_id") in source_families)
            and all(run.get(key) == value for key, value in (item.get("match") or {}).items())
        ]
        full_runs = [run for run in matches if not run.get("shard")]
        shards = sorted({str(run["shard"]) for run in matches if run.get("shard")})
        required_shards = item.get("required_shards")
        complete = bool(full_runs) or bool(required_shards and len(shards) >= required_shards)
        entries.append(
            {
                "coverage_id": _safe_name(
                    f"{item.get('task')}__{item.get('lane')}__{item.get('metric_id')}"
                    f"__{(item.get('match') or {}).get('evaluation_variant', '')}"
                ),
                "task": item.get("task"),
                "lane": item.get("lane"),
                "metric_id": item.get("metric_id"),
                "label": item.get("label") or str(item.get("metric_id")).replace("_", " ").title(),
                "status": "complete" if complete else item.get("missing_status", "in_progress"),
                "summary": None,
                "values": None,
                "metric_run_ids": [run["metric_run_id"] for run in matches],
                "available_shards": shards,
                "required_shards": required_shards,
                "notes": item.get("notes"),
            }
        )
    for override in plan.get("status_overrides") or []:
        for entry in entries:
            if all(entry.get(key) == value for key, value in override.items() if key in {"task", "lane", "metric_id"}):
                entry["status"] = override["status"]
                entry["notes"] = override.get("notes") or entry.get("notes")
    return {
        "schema_version": "metric_coverage_v1",
        "generated_from_plan": plan.get("name"),
        "entries": entries,
        "status_counts": _counts(entry["status"] for entry in entries),
    }


def _merge_logical_metrics(
    *,
    metric_runs: list[dict[str, Any]],
    metric_summaries: list[dict[str, Any]],
    metric_values: list[dict[str, Any]],
    metric_coverage: dict[str, Any],
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], list[dict[str, Any]]]:
    runs_by_id = {str(run["metric_run_id"]): run for run in metric_runs}
    summaries_by_id: dict[str, list[dict[str, Any]]] = {}
    values_by_id: dict[str, list[dict[str, Any]]] = {}
    for summary in metric_summaries:
        summaries_by_id.setdefault(str(summary["metric_run_id"]), []).append(summary)
    for value in metric_values:
        values_by_id.setdefault(str(value["metric_run_id"]), []).append(value)

    logical_runs = []
    logical_summaries = []
    logical_values = []
    for coverage in metric_coverage.get("entries") or []:
        logical_id = str(coverage["coverage_id"])
        source_ids = [str(value) for value in coverage.get("metric_run_ids") or []]
        source_runs = [runs_by_id[source_id] for source_id in source_ids if source_id in runs_by_id]
        full_runs = [run for run in source_runs if not run.get("shard")]
        chosen_runs = full_runs[:1] if full_runs else source_runs
        chosen_ids = [str(run["metric_run_id"]) for run in chosen_runs]
        merged_by_sample: dict[str, dict[str, Any]] = {}
        if coverage.get("status") == "complete":
            for source_id in chosen_ids:
                for value in values_by_id.get(source_id, []):
                    sample_id = str(value.get("sample_id"))
                    merged_by_sample.setdefault(
                        sample_id,
                        {
                            **value,
                            "logical_metric_run_id": logical_id,
                            "source_metric_run_id": source_id,
                        },
                    )
        logical_runs.append(
            {
                "logical_metric_run_id": logical_id,
                "task": coverage.get("task"),
                "lane": coverage.get("lane"),
                "metric_id": coverage.get("metric_id"),
                "label": coverage.get("label"),
                "status": coverage.get("status"),
                "selected_row_count": len(merged_by_sample),
                "source_metric_run_ids": chosen_ids,
                "all_physical_metric_run_ids": source_ids,
                "merge_strategy": "prefer_full_run_else_union_shards_by_sample_id",
                "notes": coverage.get("notes"),
            }
        )
        source_summary_rows = [
            summary
            for source_id in chosen_ids
            for summary in summaries_by_id.get(source_id, [])
        ]
        logical_summaries.append(
            {
                "logical_metric_run_id": logical_id,
                "status": coverage.get("status"),
                "summary": source_summary_rows[0] if len(source_summary_rows) == 1 else None,
                "source_summaries": source_summary_rows,
            }
        )
        logical_values.extend(merged_by_sample.values())
    return logical_runs, logical_summaries, logical_values


def _geval_evaluator_metadata() -> dict[str, Any]:
    from scripts.run_geval_prometheus_criterion_v2 import (
        CRITERION_BOUNDARIES,
        CRITERION_WEIGHTS,
        EVALUATOR_VERSION,
        PROMPT_VERSIONS,
        REFERENCE_GROUNDING_RUBRICS,
        RUBRICS,
        SCORING_POLICY_VERSION,
        build_prompt,
        build_repair_prompt,
    )

    prompt_examples = {}
    for (variant, direction), weights in CRITERION_WEIGHTS.items():
        key = f"{variant}__{direction}"
        prompt_examples[key] = {
            criterion: build_prompt(
                variant=variant,
                direction=direction,
                criterion=criterion,
                input_text="{{input_text}}",
                generated_text="{{generated_text}}",
                reference_anchors=[
                    {"anchor_id": "{{anchor_id}}", "text": "{{reference_text}}", "source": "{{source}}"}
                ],
            )
            for criterion in weights
        }
    return {
        "schema_version": "geval_evaluator_metadata_v1",
        "evaluator_version": EVALUATOR_VERSION,
        "prompt_versions": PROMPT_VERSIONS,
        "scoring_policy_version": SCORING_POLICY_VERSION,
        "message_protocol": {
            "system_prompt": None,
            "system_prompt_note": "The evaluated runs used no separate system-role prompt.",
            "judge_messages": [{"role": "user", "content": "criterion_prompt"}],
        },
        "criterion_weights": {
            f"{variant}__{direction}": weights
            for (variant, direction), weights in CRITERION_WEIGHTS.items()
        },
        "rubrics": RUBRICS,
        "reference_grounding_rubrics": REFERENCE_GROUNDING_RUBRICS,
        "criterion_boundaries": CRITERION_BOUNDARIES,
        "criterion_prompts": prompt_examples,
        "repair_prompt_template": build_repair_prompt("{{raw_response}}"),
    }


def _metric_metadata(eval_root: Path, source: Path) -> dict[str, Any]:
    relative = source.relative_to(eval_root)
    parts = relative.parts
    task_index = next((index for index, part in enumerate(parts) if part in TASK_NAMES), None)
    task = parts[task_index] if task_index is not None else None
    lane = parts[task_index + 1] if task_index is not None and task_index + 1 < len(parts) else None
    shard = next((part for part in parts if part.startswith("shard_")), None)
    metric = None
    if lane in parts:
        index = parts.index(lane) + 1
        if index < len(parts) and parts[index].startswith("shard_"):
            index += 1
        if index < len(parts):
            metric = parts[index]
    source_key = "__".join(relative.parent.parts[:-3])
    source_id = _safe_name(f"{eval_root.name}__{source_key}")
    return {
        "source_id": source_id,
        "evaluation_scope": eval_root.name,
        "task": task,
        "lane": lane,
        "shard": shard,
        "metric": metric,
    }


def _rewrite_known_media_paths(value: Any, media_map: dict[Path, str]) -> None:
    if isinstance(value, dict):
        for key, child in list(value.items()):
            if isinstance(child, str) and (key == "output_file" or key.endswith("audio")):
                mapped = media_map.get(Path(child).resolve())
                if mapped:
                    value[key] = mapped
            else:
                _rewrite_known_media_paths(child, media_map)
    elif isinstance(value, list):
        for child in value:
            _rewrite_known_media_paths(child, media_map)


def _resolve_media(value: str, dataset_root: Path) -> Path:
    path = Path(value)
    return path.resolve() if path.is_absolute() else (dataset_root / path).resolve()


def _copy_media(source: Path, relative: Path, output_dir: Path, media_map: dict[Path, str]) -> str:
    source = source.resolve()
    existing = media_map.get(source)
    if existing:
        return existing
    if not source.is_file():
        raise FileNotFoundError(f"referenced media file does not exist: {source}")
    target = output_dir / relative
    target.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(source, target)
    portable = relative.as_posix()
    media_map[source] = portable
    return portable


def _is_relative_to(path: Path, root: Path) -> bool:
    try:
        path.relative_to(root.resolve())
        return True
    except ValueError:
        return False


def _safe_name(value: str) -> str:
    return "".join(character if character.isalnum() or character in "._-" else "_" for character in value)


def _read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _read_jsonl(path: Path) -> Iterable[dict[str, Any]]:
    with path.open(encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                yield json.loads(line)


def _write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def _write_jsonl(path: Path, rows: Iterable[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        "".join(json.dumps(row, ensure_ascii=False) + "\n" for row in rows),
        encoding="utf-8",
    )


if __name__ == "__main__":
    main()
