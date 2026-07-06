#!/usr/bin/env python
"""Export a complete, frontend-ready GEval inspection bundle.

The bundle is intentionally data-rich: it preserves all criterion feedback,
rendered judge prompts, raw rows, and referenced audio files while also writing
a normalized case table for filtering and sorting in a separate website.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import shutil
import sys
import time
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable


CANONICAL_RELATIVE_OUTPUTS = (
    ("dataset", "audio", "a2s"),
    ("dataset", "audio", "s2a"),
    ("dataset", "text", "a2s"),
    ("dataset", "text", "s2a"),
    ("generated", "audio", "a2s"),
    ("generated", "audio", "s2a"),
    ("generated", "text", "a2s"),
    ("generated", "text", "s2a"),
)

SCORE_BINS = ("[1,2)", "[2,3)", "[3,4)", "[4,5)", "5")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--merged-root", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument(
        "--copy-media",
        choices=("all", "none"),
        default="all",
        help="Copy referenced audio into the bundle. Use 'none' only for smoke tests.",
    )
    parser.add_argument("--detail-shard-rows", type=int, default=1000)
    parser.add_argument("--zip", action="store_true", dest="make_zip")
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument(
        "--max-rows",
        type=int,
        help="Optional global row cap for smoke exports. Do not use for final bundles.",
    )
    parser.add_argument("--quiet", action="store_true", help="Suppress progress messages.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    result = export_bundle(args)
    print(
        "bundle={output_dir} rows={case_count} details={detail_count} "
        "media={copied_media_count}/{media_source_count} zip={zip_path}".format(**result)
    )


def export_bundle(args: argparse.Namespace) -> dict[str, Any]:
    if args.detail_shard_rows <= 0:
        raise ValueError("--detail-shard-rows must be positive")
    if args.max_rows is not None and args.max_rows <= 0:
        raise ValueError("--max-rows must be positive when provided")

    started_at = time.monotonic()
    _progress(args, f"starting export output_dir={args.output_dir}")
    canonical_root = args.merged_root / "canonical"
    source_files = _resolve_source_files(canonical_root)
    _progress(args, f"resolved {len(source_files)} canonical output files")
    output_dir = args.output_dir
    if output_dir.exists():
        if not args.overwrite:
            raise FileExistsError(f"output directory already exists: {output_dir}")
        shutil.rmtree(output_dir)
    _ensure_dirs(output_dir)

    _progress(args, "collecting referenced audio paths")
    media_by_case_key, media_source_paths = _collect_audio_paths(source_files, args.max_rows, args=args)
    _progress(args, f"found {len(media_source_paths)} unique referenced audio files")
    media_index = _build_media_index(media_source_paths, output_dir, copy_media=args.copy_media, args=args)
    _progress(args, f"media indexing complete copied={sum(1 for row in media_index.values() if row.get('copied'))}/{len(media_index)}")

    cases: list[dict[str, Any]] = []
    detail_count = 0
    criteria_by_suite: dict[str, set[str]] = defaultdict(set)
    source_row_counts: dict[str, int] = {}
    source_status_counts: dict[str, Counter[str]] = {}
    detail_file = None
    detail_writer_path: Path | None = None
    detail_line = 0
    detail_shard_index = -1

    try:
        for source_file in source_files:
            source_label = _source_label(canonical_root, source_file)
            _progress(args, f"processing rows from {source_label}")
            source_row_counts[source_label] = 0
            source_status_counts[source_label] = Counter()
            for raw in _iter_jsonl(source_file):
                if args.max_rows is not None and len(cases) >= args.max_rows:
                    break
                _validate_test_split(raw, source_file)
                source, metric_suite, direction = source_label.split("/")
                case = _normalize_case(
                    raw=raw,
                    source=source,
                    metric_suite=metric_suite,
                    direction=direction,
                    source_label=source_label,
                    source_file=source_file,
                    media_by_case_key=media_by_case_key,
                    media_index=media_index,
                )
                criteria_by_suite[metric_suite].update(case["criterion_scores"])

                if detail_count % args.detail_shard_rows == 0:
                    if detail_file:
                        detail_file.close()
                    detail_shard_index += 1
                    detail_writer_path = Path("details") / f"details_{detail_shard_index:03d}.jsonl"
                    detail_file = (output_dir / detail_writer_path).open("w", encoding="utf-8")
                    detail_line = 0
                assert detail_file is not None
                assert detail_writer_path is not None
                case["detail_ref"] = {"path": detail_writer_path.as_posix(), "line": detail_line}
                detail_row = {
                    "case_id": case["case_id"],
                    "case": case,
                    "raw_geval_row": raw,
                    "criteria": raw.get("criteria") or {},
                    "diagnostics": raw.get("diagnostics"),
                }
                detail_file.write(json.dumps(detail_row, ensure_ascii=False) + "\n")
                detail_line += 1
                detail_count += 1
                cases.append(case)
                source_row_counts[source_label] += 1
                source_status_counts[source_label][str(raw.get("status"))] += 1
                if len(cases) % 1000 == 0:
                    _progress(args, f"processed {len(cases)} GEval rows; details={detail_count}; current_source={source_label}")
            if args.max_rows is not None and len(cases) >= args.max_rows:
                break
    finally:
        if detail_file:
            detail_file.close()

    _progress(args, f"writing normalized cases rows={len(cases)}")
    _write_jsonl(output_dir / "cases" / "cases.jsonl", cases)
    _write_cases_csv(output_dir / "cases" / "cases.csv", cases)
    _progress(args, "writing saved views")
    saved_views = _write_saved_views(output_dir, cases)
    _write_media_index(output_dir / "media_index.json", media_index)

    manifest = _build_manifest(
        args=args,
        canonical_root=canonical_root,
        source_files=source_files,
        source_row_counts=source_row_counts,
        source_status_counts=source_status_counts,
        criteria_by_suite=criteria_by_suite,
        cases=cases,
        detail_count=detail_count,
        media_index=media_index,
        saved_views=saved_views,
    )
    _write_json(output_dir / "manifest.json", manifest)
    _write_handoff(output_dir, manifest)
    _progress(args, "validating bundle")
    validation = _validate_bundle(output_dir, manifest, cases, media_index, saved_views)
    _write_json(output_dir / "validation.json", validation)
    if not validation["ok"]:
        raise RuntimeError(f"bundle validation failed: {validation}")

    zip_path = None
    if args.make_zip:
        _progress(args, "creating zip archive; this can take a while")
        zip_path = _make_zip(output_dir, args=args)
        _progress(args, f"zip archive complete: {zip_path}")

    _progress(args, f"export complete in {time.monotonic() - started_at:.1f}s")
    return {
        "output_dir": str(output_dir),
        "case_count": len(cases),
        "detail_count": detail_count,
        "media_source_count": len(media_index),
        "copied_media_count": sum(1 for row in media_index.values() if row.get("copied")),
        "zip_path": str(zip_path) if zip_path else "not_requested",
    }


def _progress(args: argparse.Namespace, message: str) -> None:
    if getattr(args, "quiet", False):
        return
    timestamp = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
    print(f"[{timestamp}] {message}", file=sys.stderr, flush=True)


def _resolve_source_files(canonical_root: Path) -> list[Path]:
    files = [canonical_root.joinpath(*parts, "output.jsonl") for parts in CANONICAL_RELATIVE_OUTPUTS]
    missing = [str(path) for path in files if not path.exists()]
    if missing:
        raise FileNotFoundError("missing canonical merged GEval outputs:\n" + "\n".join(missing))
    return files


def _ensure_dirs(output_dir: Path) -> None:
    for relative in ("cases", "details", "views", "media"):
        (output_dir / relative).mkdir(parents=True, exist_ok=True)


def _iter_jsonl(path: Path) -> Iterable[dict[str, Any]]:
    with path.open(encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, 1):
            if not line.strip():
                continue
            try:
                yield json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(f"invalid JSON in {path}:{line_number}: {exc}") from exc


def _source_label(canonical_root: Path, output_jsonl: Path) -> str:
    return str(output_jsonl.parent.relative_to(canonical_root))


def _validate_test_split(raw: dict[str, Any], source_file: Path) -> None:
    if raw.get("split") != "test":
        raise ValueError(f"non-test row in {source_file}: sample_id={raw.get('sample_id')} split={raw.get('split')}")


def _collect_audio_paths(
    source_files: list[Path],
    max_rows: int | None,
    *,
    args: argparse.Namespace,
) -> tuple[dict[tuple[str, str, str], dict[str, str]], set[str]]:
    media_by_case_key: dict[tuple[str, str, str], dict[str, str]] = {}
    media_source_paths: set[str] = set()
    seen_rows = 0
    for source_file in source_files:
        source, metric_suite, direction = source_file.parent.parts[-3:]
        for raw in _iter_jsonl(source_file):
            if max_rows is not None and seen_rows >= max_rows:
                break
            seen_rows += 1
            sample_id = str(raw.get("sample_id") or "")
            key = (source, direction, sample_id)
            entry = media_by_case_key.setdefault(key, {})
            for field in ("source_audio_file", "generated_audio_file"):
                value = raw.get(field)
                if value:
                    entry.setdefault(field, str(value))
                    media_source_paths.add(str(value))
            if raw.get("generated_audio_kind"):
                entry.setdefault("generated_audio_kind", str(raw["generated_audio_kind"]))
            if seen_rows % 2000 == 0:
                _progress(args, f"scanned {seen_rows} rows for audio paths; unique_audio={len(media_source_paths)}")
        if max_rows is not None and seen_rows >= max_rows:
            break
    return media_by_case_key, media_source_paths


def _build_media_index(
    media_source_paths: set[str],
    output_dir: Path,
    copy_media: str,
    *,
    args: argparse.Namespace,
) -> dict[str, dict[str, Any]]:
    media_index: dict[str, dict[str, Any]] = {}
    total = len(media_source_paths)
    copied_count = 0
    copied_bytes = 0
    for index, source_path in enumerate(sorted(media_source_paths), 1):
        path = Path(source_path)
        exists = path.exists()
        relative_path = _portable_media_path(path)
        copied = False
        size_bytes = path.stat().st_size if exists else None
        if copy_media == "all":
            if not exists:
                raise FileNotFoundError(f"referenced media does not exist: {source_path}")
            destination = output_dir / relative_path
            destination.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(path, destination)
            copied = True
            copied_count += 1
            copied_bytes += size_bytes or 0
            if copied_count == 1 or copied_count % 500 == 0 or copied_count == total:
                _progress(args, f"copied media {copied_count}/{total} ({copied_bytes / 1024**3:.2f} GiB) latest={destination}")
        media_index[source_path] = {
            "source_path": source_path,
            "bundle_path": relative_path,
            "exists": exists,
            "copied": copied,
            "size_bytes": size_bytes,
        }
    return media_index


def _portable_media_path(path: Path) -> str:
    digest = hashlib.sha1(str(path).encode("utf-8")).hexdigest()[:12]
    name = f"{digest}_{path.name}"
    path_text = str(path)
    if "/datasets/ttas/extracted/audio/" in path_text:
        bucket = "dataset/audio"
    elif "/datasets/ttas/extracted/speech/" in path_text:
        bucket = "dataset/speech"
    elif "/results/cascaded/stages/t2s/" in path_text:
        bucket = "generated/speech"
    elif "/results/cascaded/stages/t2a/" in path_text:
        bucket = "generated/audio"
    else:
        bucket = "other"
    return (Path("media") / bucket / name).as_posix()


def _normalize_case(
    *,
    raw: dict[str, Any],
    source: str,
    metric_suite: str,
    direction: str,
    source_label: str,
    source_file: Path,
    media_by_case_key: dict[tuple[str, str, str], dict[str, str]],
    media_index: dict[str, dict[str, Any]],
) -> dict[str, Any]:
    sample_id = str(raw.get("sample_id") or "")
    row_index = raw.get("row_index")
    case_id = f"{source}__{metric_suite}__{direction}__{row_index}__{sample_id}"
    criteria = raw.get("criteria") or {}
    criterion_scores: dict[str, int | float | None] = {}
    criterion_feedback: dict[str, str | None] = {}
    criterion_parse_status: dict[str, str | None] = {}
    prompt_available: dict[str, bool] = {}
    feedback_preview: dict[str, str | None] = {}
    for criterion, info in criteria.items():
        info = info or {}
        criterion_scores[criterion] = info.get("score")
        feedback = info.get("feedback")
        criterion_feedback[criterion] = feedback
        criterion_parse_status[criterion] = info.get("parse_status")
        prompt_available[criterion] = bool(info.get("runtime_prompt"))
        feedback_preview[criterion] = _preview(feedback)

    media = dict(media_by_case_key.get((source, direction, sample_id), {}))
    for field in ("source_audio_file", "generated_audio_file", "generated_audio_kind"):
        if raw.get(field):
            media[field] = str(raw[field])

    source_audio_file = media.get("source_audio_file")
    generated_audio_file = media.get("generated_audio_file")
    case: dict[str, Any] = {
        "case_id": case_id,
        "sample_id": sample_id,
        "row_index": row_index,
        "pair_id": raw.get("pair_id"),
        "source": source,
        "metric_suite": metric_suite,
        "direction": direction,
        "raw_direction": raw.get("direction"),
        "metric_id": raw.get("metric"),
        "status": raw.get("status"),
        "split": raw.get("split"),
        "final_score": raw.get("final_score"),
        "input_text": raw.get("input_text"),
        "generated_text": raw.get("generated_text"),
        "source_audio_file": source_audio_file,
        "generated_audio_file": generated_audio_file,
        "source_audio_bundle_path": _bundle_path_for(source_audio_file, media_index),
        "generated_audio_bundle_path": _bundle_path_for(generated_audio_file, media_index),
        "generated_audio_kind": media.get("generated_audio_kind"),
        "primary_audio_class": raw.get("primary_audio_class"),
        "criterion_scores": criterion_scores,
        "criterion_feedback": criterion_feedback,
        "criterion_feedback_preview": feedback_preview,
        "criterion_parse_status": criterion_parse_status,
        "criterion_prompt_available": prompt_available,
        "criterion_weights": raw.get("criterion_weights"),
        "judge_model": raw.get("judge_model"),
        "prompt_version": raw.get("prompt_version"),
        "schema_version": raw.get("schema_version"),
        "evaluation_variant": raw.get("evaluation_variant"),
        "scoring_policy_version": raw.get("scoring_policy_version"),
        "source_label": source_label,
        "source_output_jsonl": str(source_file),
        "detail_ref": None,
    }
    for criterion, score in criterion_scores.items():
        case[f"score__{criterion}"] = score
    for criterion, feedback in criterion_feedback.items():
        case[f"feedback__{criterion}"] = feedback
    return case


def _bundle_path_for(source_path: str | None, media_index: dict[str, dict[str, Any]]) -> str | None:
    if not source_path:
        return None
    row = media_index.get(source_path)
    return str(row["bundle_path"]) if row else None


def _preview(text: Any, limit: int = 240) -> str | None:
    if not isinstance(text, str):
        return None
    one_line = " ".join(text.split())
    if len(one_line) <= limit:
        return one_line
    return one_line[: limit - 1] + "…"


def _write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, ensure_ascii=False, sort_keys=True) + "\n", encoding="utf-8")


def _write_jsonl(path: Path, rows: Iterable[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")


def _write_cases_csv(path: Path, cases: list[dict[str, Any]]) -> None:
    criteria = sorted({criterion for case in cases for criterion in case["criterion_scores"]})
    fields = [
        "case_id",
        "sample_id",
        "row_index",
        "source",
        "metric_suite",
        "direction",
        "metric_id",
        "status",
        "final_score",
        "input_text",
        "generated_text",
        "source_audio_bundle_path",
        "generated_audio_bundle_path",
        "generated_audio_kind",
        "primary_audio_class",
        "detail_ref",
    ]
    fields += [f"score__{criterion}" for criterion in criteria]
    fields += [f"feedback__{criterion}" for criterion in criteria]
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        for case in cases:
            row = dict(case)
            row["detail_ref"] = json.dumps(case["detail_ref"], ensure_ascii=False)
            writer.writerow(row)


def _write_media_index(path: Path, media_index: dict[str, dict[str, Any]]) -> None:
    rows = [media_index[key] for key in sorted(media_index)]
    _write_json(path, {"media": rows, "by_source_path": media_index})


def _write_saved_views(output_dir: Path, cases: list[dict[str, Any]]) -> list[dict[str, Any]]:
    views: list[dict[str, Any]] = []
    target_rows = [
        case
        for case in cases
        if case["metric_suite"] == "audio"
        and _numeric(case["criterion_scores"].get("target_modality_success")) is not None
        and _numeric(case["criterion_scores"].get("target_modality_success")) < 5
    ]
    target_rows.sort(
        key=lambda case: (
            _sort_number(case["criterion_scores"].get("target_modality_success")),
            _sort_number(case.get("final_score")),
            str(case["source"]),
            str(case["direction"]),
            int(case["row_index"] or 0),
        )
    )
    views.append(
        _write_view(
            output_dir,
            name="target_modality_success_lt_5",
            description="All GEval audio rows with target_modality_success score below 5.",
            filters={
                "metric_suite": ["audio"],
                "criterion_ranges": {"target_modality_success": {"min": 1, "max": 4}},
            },
            sort=[
                {"field": "score__target_modality_success", "direction": "asc"},
                {"field": "final_score", "direction": "asc"},
                {"field": "row_index", "direction": "asc"},
            ],
            rows=target_rows,
        )
    )

    scenario_rows: list[dict[str, Any]] = []
    grouped: dict[tuple[str, str, str], list[dict[str, Any]]] = defaultdict(list)
    for case in cases:
        if _numeric(case["criterion_scores"].get("scenario_plausibility")) == 5:
            grouped[(case["metric_suite"], case["source"], case["direction"])].append(case)
    for group_key, group_rows in sorted(grouped.items()):
        high = sorted(group_rows, key=lambda case: (-_sort_number(case.get("final_score")), int(case["row_index"] or 0)))[:5]
        low = sorted(group_rows, key=lambda case: (_sort_number(case.get("final_score")), int(case["row_index"] or 0)))[:5]
        for selection, selected in (("highest_final_score", high), ("lowest_final_score", low)):
            for rank, case in enumerate(selected, 1):
                row = dict(case)
                row["view_group"] = "/".join(group_key)
                row["view_selection"] = selection
                row["view_rank"] = rank
                scenario_rows.append(row)
    views.append(
        _write_view(
            output_dir,
            name="scenario_plausibility_eq_5_high_low_final",
            description=(
                "Rows with scenario_plausibility score 5, grouped by metric_suite/source/direction, "
                "with five highest and five lowest final_score rows per group."
            ),
            filters={"criterion_ranges": {"scenario_plausibility": {"equals": 5}}},
            sort=[
                {"field": "metric_suite", "direction": "asc"},
                {"field": "source", "direction": "asc"},
                {"field": "direction", "direction": "asc"},
                {"field": "view_selection", "direction": "asc"},
                {"field": "final_score", "direction": "desc_or_asc_by_selection"},
            ],
            rows=scenario_rows,
        )
    )
    _write_json(output_dir / "views" / "views.json", {"views": views})
    return views


def _write_view(
    output_dir: Path,
    *,
    name: str,
    description: str,
    filters: dict[str, Any],
    sort: list[dict[str, str]],
    rows: list[dict[str, Any]],
) -> dict[str, Any]:
    path = Path("views") / f"{name}.jsonl"
    _write_jsonl(output_dir / path, rows)
    return {
        "name": name,
        "description": description,
        "path": path.as_posix(),
        "row_count": len(rows),
        "filters": filters,
        "sort": sort,
    }


def _numeric(value: Any) -> float | None:
    if isinstance(value, (int, float)):
        return float(value)
    return None


def _sort_number(value: Any) -> float:
    numeric = _numeric(value)
    return numeric if numeric is not None else float("inf")


def _build_manifest(
    *,
    args: argparse.Namespace,
    canonical_root: Path,
    source_files: list[Path],
    source_row_counts: dict[str, int],
    source_status_counts: dict[str, Counter[str]],
    criteria_by_suite: dict[str, set[str]],
    cases: list[dict[str, Any]],
    detail_count: int,
    media_index: dict[str, dict[str, Any]],
    saved_views: list[dict[str, Any]],
) -> dict[str, Any]:
    return {
        "schema_version": "geval_inspection_bundle_v1",
        "created_at": datetime.now(timezone.utc).isoformat(),
        "merged_root": str(args.merged_root),
        "canonical_root": str(canonical_root),
        "split": "test",
        "included_runs": "canonical_only",
        "copy_media": args.copy_media,
        "max_rows": args.max_rows,
        "detail_shard_rows": args.detail_shard_rows,
        "counts": {
            "cases": len(cases),
            "details": detail_count,
            "media_sources": len(media_index),
            "media_copied": sum(1 for row in media_index.values() if row.get("copied")),
            "media_missing": sum(1 for row in media_index.values() if not row.get("exists")),
        },
        "criteria_by_suite": {key: sorted(value) for key, value in criteria_by_suite.items()},
        "source_outputs": [
            {
                "label": _source_label(canonical_root, source_file),
                "path": str(source_file),
                "rows": source_row_counts.get(_source_label(canonical_root, source_file), 0),
                "status_counts": dict(source_status_counts.get(_source_label(canonical_root, source_file), Counter())),
            }
            for source_file in source_files
        ],
        "files": {
            "cases_jsonl": "cases/cases.jsonl",
            "cases_csv": "cases/cases.csv",
            "details_glob": "details/details_*.jsonl",
            "views_catalog": "views/views.json",
            "media_index": "media_index.json",
            "validation": "validation.json",
            "website_handoff": "WEBSITE_CODEX_HANDOFF.md",
        },
        "saved_views": saved_views,
    }


def _write_handoff(output_dir: Path, manifest: dict[str, Any]) -> None:
    lines = [
        "# Website Codex Handoff: GEval Inspection Bundle",
        "",
        "This bundle is a complete data/metadata export for inspecting GEval text and audio judgments.",
        "Build the frontend around the exported data; do not rerun metrics or infer missing fields.",
        "",
        "## Primary Files",
        "",
        "- `cases/cases.jsonl`: main table source, one normalized row per GEval case.",
        "- `cases/cases.csv`: spreadsheet/debug copy of the main table.",
        "- `details/details_*.jsonl`: full details for every case, including raw GEval rows, criterion feedback, rendered prompts, diagnostics, and parser metadata.",
        "- `media_index.json`: maps original absolute audio paths to bundle-relative copied audio paths.",
        "- `views/views.json`: saved views and their filter/sort definitions.",
        "",
        "## Required UI Behavior",
        "",
        "- Filter by `source`, `metric_suite`, `direction`, `status`, `sample_id`, `final_score`, and any `score__<criterion>` column.",
        "- Support contrasting criterion filters, for example `score__criterion_a in [4,5]` AND `score__criterion_b in [1,2]`.",
        "- Support hierarchical sorting with multiple ordered sort keys, e.g. criterion 1, then criterion 2, then final score, then row index.",
        "- Show all criterion scores and feedback. Full rendered prompts should be available from the detail rows.",
        "- Use `source_audio_bundle_path` and `generated_audio_bundle_path` for audio playback when present.",
        "- Treat saved views as quick-start filters/bookmarks, not hardcoded logic.",
        "",
        "## Criteria By Suite",
        "",
    ]
    for suite, criteria in sorted((manifest.get("criteria_by_suite") or {}).items()):
        lines.append(f"- `{suite}`: " + ", ".join(f"`{criterion}`" for criterion in criteria))
    lines.extend(
        [
            "",
            "## Bundle Counts",
            "",
            "```json",
            json.dumps(manifest.get("counts"), indent=2, sort_keys=True),
            "```",
            "",
        ]
    )
    (output_dir / "WEBSITE_CODEX_HANDOFF.md").write_text("\n".join(lines), encoding="utf-8")


def _validate_bundle(
    output_dir: Path,
    manifest: dict[str, Any],
    cases: list[dict[str, Any]],
    media_index: dict[str, dict[str, Any]],
    saved_views: list[dict[str, Any]],
) -> dict[str, Any]:
    problems: list[str] = []
    detail_refs = Counter()
    for case in cases:
        ref = case.get("detail_ref") or {}
        detail_path = output_dir / str(ref.get("path", ""))
        if not detail_path.exists():
            problems.append(f"missing detail file for {case['case_id']}: {ref}")
        detail_refs[(ref.get("path"), ref.get("line"))] += 1
        if case.get("split") != "test":
            problems.append(f"non-test case exported: {case['case_id']}")
    duplicate_refs = [ref for ref, count in detail_refs.items() if count != 1]
    if duplicate_refs:
        problems.append(f"detail refs are not unique: {duplicate_refs[:5]}")
    for source_path, row in media_index.items():
        if not row.get("exists"):
            problems.append(f"media source missing: {source_path}")
        if row.get("copied") and not (output_dir / row["bundle_path"]).exists():
            problems.append(f"copied media missing: {row['bundle_path']}")
    for view in saved_views:
        path = output_dir / view["path"]
        if not path.exists():
            problems.append(f"missing saved view file: {view['path']}")
        else:
            row_count = sum(1 for _ in path.open(encoding="utf-8"))
            if row_count != view["row_count"]:
                problems.append(f"saved view count mismatch for {view['name']}: {row_count} != {view['row_count']}")
    return {
        "ok": not problems,
        "problems": problems,
        "case_count": len(cases),
        "detail_count": manifest["counts"]["details"],
        "media_source_count": len(media_index),
        "saved_view_count": len(saved_views),
    }


def _make_zip(output_dir: Path, *, args: argparse.Namespace) -> Path:
    final_zip = output_dir.with_suffix(".zip")
    tmp_base = output_dir.parent / f"{output_dir.name}.tmp"
    tmp_zip = Path(str(tmp_base) + ".zip")
    if tmp_zip.exists():
        tmp_zip.unlink()
    _progress(args, f"zipping {output_dir} -> {tmp_zip}")
    shutil.make_archive(str(tmp_base), "zip", root_dir=output_dir.parent, base_dir=output_dir.name)
    if final_zip.exists():
        final_zip.unlink()
    tmp_zip.replace(final_zip)
    return final_zip


if __name__ == "__main__":
    main()
