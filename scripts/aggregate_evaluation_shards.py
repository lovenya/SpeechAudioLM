#!/usr/bin/env python
"""Merge sharded evaluation outputs and validate their row-index coverage."""

from __future__ import annotations

import argparse
import json
import math
import statistics
import sys
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Iterable


NON_COMPOSABLE_CORPUS_METRICS = {
    "fad",
    "inception_score",
    "kad",
    "kl_passt",
    "spider",
}

SPIDER_INVALIDITY = {
    "status": "requires_full_corpus_recomputation",
    "reason": (
        "Historical shard-derived SPIDEr values are scientifically invalid because "
        "CIDEr document-frequency statistics differ across shards. Run SPIDEr once "
        "on the complete caption corpus."
    ),
}

SHARD_AGGREGATION_VALIDITY = {
    "status": "diagnostic_only",
    "reason": (
        "Evaluation sharding is unsupported for final scientific evaluation. "
        "Merged outputs may be used only for diagnostics; run evaluation on the "
        "complete corpus for final scientific results."
    ),
}


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "inputs",
        nargs="+",
        type=Path,
        help="Shard evaluation directories or latest_per_sample_*.jsonl files.",
    )
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument(
        "--expected-start",
        type=int,
        default=0,
        help="First expected row_index when --expected-count or --expected-end is supplied.",
    )
    expected = parser.add_mutually_exclusive_group()
    expected.add_argument("--expected-count", type=int)
    expected.add_argument("--expected-end", type=int, help="Exclusive expected row_index end.")
    parser.add_argument(
        "--allow-incomplete",
        action="store_true",
        help="Write merged outputs and exit successfully despite duplicate or missing rows.",
    )
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args(argv)


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            line = line.strip()
            if not line:
                continue
            value = json.loads(line)
            if not isinstance(value, dict):
                raise ValueError(f"{path}:{line_number}: expected a JSON object")
            if "row_index" not in value:
                raise ValueError(f"{path}:{line_number}: missing row_index")
            try:
                value["row_index"] = int(value["row_index"])
            except (TypeError, ValueError) as exc:
                raise ValueError(f"{path}:{line_number}: invalid row_index={value['row_index']!r}") from exc
            rows.append(value)
    return rows


def discover_per_sample_files(inputs: Iterable[Path]) -> list[Path]:
    discovered: set[Path] = set()
    for item in inputs:
        item = item.resolve()
        if item.is_file():
            discovered.add(item)
            continue
        if not item.is_dir():
            raise FileNotFoundError(f"input does not exist: {item}")
        latest = sorted(item.rglob("latest_per_sample_*.jsonl"))
        if latest:
            discovered.update(latest)
            continue
        by_parent: dict[Path, list[Path]] = defaultdict(list)
        for path in item.rglob("per_sample_*.jsonl"):
            by_parent[path.parent].append(path)
        for paths in by_parent.values():
            discovered.add(max(paths, key=lambda path: (path.stat().st_mtime_ns, path.name)))
    if not discovered:
        raise FileNotFoundError("no latest_per_sample_*.jsonl or per_sample_*.jsonl inputs found")
    return sorted(discovered)


def companion_summary_path(per_sample_path: Path) -> Path | None:
    name = per_sample_path.name
    candidates: list[Path] = []
    if name.startswith("latest_per_sample_"):
        candidates.append(per_sample_path.with_name(name.replace("latest_per_sample_", "latest_summary_", 1).replace(".jsonl", ".json")))
    if name.startswith("per_sample_"):
        candidates.append(per_sample_path.with_name(name.replace("per_sample_", "summary_", 1).replace(".jsonl", ".json")))
    return next((path for path in candidates if path.is_file()), None)


def expected_indices(start: int, count: int | None, end: int | None) -> set[int] | None:
    if count is None and end is None:
        return None
    if count is not None:
        if count < 0:
            raise ValueError("--expected-count must be nonnegative")
        end = start + count
    assert end is not None
    if end < start:
        raise ValueError("--expected-end must be greater than or equal to --expected-start")
    return set(range(start, end))


def numeric_summary(values: list[float]) -> dict[str, Any]:
    return {
        "count": len(values),
        "mean": statistics.fmean(values),
        "std": statistics.pstdev(values) if len(values) > 1 else 0.0,
        "min": min(values),
        "max": max(values),
        "median": statistics.median(values),
    }


def summarize_rows(rows: list[dict[str, Any]]) -> dict[str, Any]:
    metric_values: dict[str, list[float]] = defaultdict(list)
    errors_by_metric: dict[str, Counter[str]] = defaultdict(Counter)
    for row in rows:
        for metric, value in (row.get("metrics") or {}).items():
            if isinstance(value, (int, float)) and not isinstance(value, bool) and math.isfinite(float(value)):
                metric_values[str(metric)].append(float(value))
        for metric, error in (row.get("errors") or {}).items():
            normalized = json.dumps(error, ensure_ascii=False, sort_keys=True) if isinstance(error, (dict, list)) else str(error)
            errors_by_metric[str(metric)][normalized] += 1
    return {
        "rows": len(rows),
        "successful_rows": sum(row.get("status") == "success" for row in rows),
        "skipped_or_error_rows": sum(row.get("status") != "success" for row in rows),
        "metric_summaries": {name: numeric_summary(values) for name, values in sorted(metric_values.items())},
        "errors_by_metric": {
            metric: dict(sorted(messages.items()))
            for metric, messages in sorted(errors_by_metric.items())
        },
    }


def invalidate_spider_values(rows: list[dict[str, Any]]) -> int:
    invalidated = 0
    for row in rows:
        metrics = row.get("metrics")
        if not isinstance(metrics, dict) or "spider" not in metrics:
            continue
        metrics.pop("spider")
        errors = row.setdefault("errors", {})
        if not isinstance(errors, dict):
            errors = {}
            row["errors"] = errors
        errors["spider"] = {
            "type": "invalid_historical_shard_metric",
            **SPIDER_INVALIDITY,
        }
        invalidated += 1
    return invalidated


def corpus_metric_summary(shard_summaries: list[tuple[Path, dict[str, Any]]], sample_metrics: set[str]) -> dict[str, Any]:
    values: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for path, summary in shard_summaries:
        for metric, metric_summary in (summary.get("metric_summaries") or {}).items():
            if (
                metric in sample_metrics
                and metric not in NON_COMPOSABLE_CORPUS_METRICS
            ) or not isinstance(metric_summary, dict):
                continue
            values[str(metric)].append({"source": str(path), "summary": metric_summary})
    result: dict[str, Any] = {}
    for metric, shards in sorted(values.items()):
        if metric == "spider":
            result[metric] = dict(SPIDER_INVALIDITY)
            continue
        if len(shards) == 1:
            result[metric] = {**shards[0]["summary"], "source": shards[0]["source"]}
        else:
            result[metric] = {
                "level": "corpus",
                "status": "requires_full_corpus_recomputation",
                "reason": "Corpus-level shard scores are not generally composable.",
                "shards": shards,
            }
    return result


def common_metadata(shard_summaries: list[tuple[Path, dict[str, Any]]]) -> dict[str, Any]:
    keys = ("model", "task", "dataset", "split", "manifest", "metrics")
    result: dict[str, Any] = {}
    for key in keys:
        values = [summary.get(key) for _, summary in shard_summaries if key in summary]
        if values and all(value == values[0] for value in values):
            result[key] = values[0]
    return result


def compact_ranges(indices: Iterable[int]) -> list[list[int]]:
    ordered = sorted(set(indices))
    if not ordered:
        return []
    ranges: list[list[int]] = []
    start = previous = ordered[0]
    for current in ordered[1:]:
        if current != previous + 1:
            ranges.append([start, previous])
            start = current
        previous = current
    ranges.append([start, previous])
    return ranges


def aggregate(
    inputs: Iterable[Path],
    output_dir: Path,
    *,
    expected_start: int = 0,
    expected_count: int | None = None,
    expected_end: int | None = None,
    allow_incomplete: bool = False,
    overwrite: bool = False,
) -> dict[str, Path]:
    per_sample_files = discover_per_sample_files(inputs)
    rows_by_index: dict[int, dict[str, Any]] = {}
    row_sources: dict[int, str] = {}
    duplicates: list[dict[str, Any]] = []
    shard_summaries: list[tuple[Path, dict[str, Any]]] = []
    input_reports: list[dict[str, Any]] = []

    for path in per_sample_files:
        rows = read_jsonl(path)
        input_reports.append(
            {
                "per_sample": str(path),
                "rows": len(rows),
                "min_row_index": min((row["row_index"] for row in rows), default=None),
                "max_row_index": max((row["row_index"] for row in rows), default=None),
            }
        )
        for row in rows:
            index = row["row_index"]
            if index in rows_by_index:
                duplicates.append(
                    {
                        "row_index": index,
                        "first_source": row_sources[index],
                        "duplicate_source": str(path),
                    }
                )
                continue
            rows_by_index[index] = row
            row_sources[index] = str(path)
        summary_path = companion_summary_path(path)
        if summary_path is not None:
            summary = json.loads(summary_path.read_text(encoding="utf-8"))
            if isinstance(summary, dict):
                shard_summaries.append((summary_path, summary))

    rows = [rows_by_index[index] for index in sorted(rows_by_index)]
    spider_in_shard_summaries = any(
        "spider" in (summary.get("metric_summaries") or {})
        for _, summary in shard_summaries
    )
    invalidated_spider_per_sample_values = invalidate_spider_values(rows)
    observed = set(rows_by_index)
    expected = expected_indices(expected_start, expected_count, expected_end)
    missing = sorted(expected - observed) if expected is not None else []
    unexpected = sorted(observed - expected) if expected is not None else []
    internal_gaps = (
        sorted(set(range(min(observed), max(observed) + 1)) - observed)
        if observed
        else []
    )
    issues = []
    if duplicates:
        issues.append("duplicate_row_indices")
    if missing:
        issues.append("missing_expected_row_indices")
    elif expected is None and internal_gaps:
        issues.append("missing_internal_row_indices")
    if unexpected:
        issues.append("unexpected_row_indices")

    output_dir.mkdir(parents=True, exist_ok=True)
    paths = {
        "per_sample": output_dir / "per_sample.jsonl",
        "summary": output_dir / "summary.json",
        "merge_report": output_dir / "merge_report.json",
    }
    if not overwrite:
        existing = [str(path) for path in paths.values() if path.exists()]
        if existing:
            raise FileExistsError(f"output files exist; pass --overwrite: {', '.join(existing)}")

    report = {
        "status": "complete" if not issues else ("incomplete_allowed" if allow_incomplete else "invalid"),
        "input_files": input_reports,
        "input_file_count": len(per_sample_files),
        "merged_unique_rows": len(rows),
        "duplicates": duplicates,
        "duplicate_count": len(duplicates),
        "duplicate_row_indices": sorted({item["row_index"] for item in duplicates}),
        "expected": (
            {
                "start": expected_start,
                "end_exclusive": expected_start + expected_count if expected_count is not None else expected_end,
                "count": len(expected),
            }
            if expected is not None
            else None
        ),
        "missing_expected_row_indices": missing,
        "missing_expected_count": len(missing),
        "missing_expected_ranges": compact_ranges(missing),
        "unexpected_row_indices": unexpected,
        "unexpected_row_count": len(unexpected),
        "internal_gap_indices": internal_gaps,
        "internal_gap_ranges": compact_ranges(internal_gaps),
        "issues": issues,
        "scientific_validity": SHARD_AGGREGATION_VALIDITY,
        "invalidated_spider_per_sample_values": invalidated_spider_per_sample_values,
        "spider_requires_full_corpus_recomputation": bool(
            invalidated_spider_per_sample_values or spider_in_shard_summaries
        ),
    }
    paths["merge_report"].write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")

    if issues and not allow_incomplete:
        raise ValueError(f"invalid shard merge: {', '.join(issues)}; see {paths['merge_report']}")

    row_summary = summarize_rows(rows)
    corpus_summaries = corpus_metric_summary(
        shard_summaries,
        {metric for row in rows for metric in (row.get("metrics") or {})},
    )
    if invalidated_spider_per_sample_values and "spider" not in corpus_summaries:
        corpus_summaries["spider"] = dict(SPIDER_INVALIDITY)
    summary = {
        **common_metadata(shard_summaries),
        **row_summary,
        "metric_summaries": {
            **row_summary["metric_summaries"],
            **corpus_summaries,
        },
        "corpus_metric_summaries": corpus_summaries,
        "scientific_validity": SHARD_AGGREGATION_VALIDITY,
        "invalidated_spider_per_sample_values": invalidated_spider_per_sample_values,
        "merge": {
            "status": report["status"],
            "input_file_count": len(per_sample_files),
            "merge_report": str(paths["merge_report"]),
        },
    }
    with paths["per_sample"].open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")
    paths["summary"].write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return paths


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    try:
        paths = aggregate(
            args.inputs,
            args.output_dir,
            expected_start=args.expected_start,
            expected_count=args.expected_count,
            expected_end=args.expected_end,
            allow_incomplete=args.allow_incomplete,
            overwrite=args.overwrite,
        )
    except Exception as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1
    for name, path in paths.items():
        print(f"{name}={path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
