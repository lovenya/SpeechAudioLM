#!/usr/bin/env python
"""Assign coarse primary audio classes from retained PANNs/AudioSet tags."""

from __future__ import annotations

import argparse
import json
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any


DEFAULT_CONFIG = Path("configs/audio_tag_class_pooling.json")
DEFAULT_INPUT_MANIFESTS = {
    "train": Path("data/manifests/audio_tags/ttas_train_panns_cnn14.jsonl"),
    "val": Path("data/manifests/audio_tags/ttas_val_panns_cnn14.jsonl"),
    "test": Path("data/manifests/audio_tags/ttas_test_panns_cnn14.jsonl"),
}
DEFAULT_OUTPUT_ROOT = Path("data/manifests/audio_classes")
DEFAULT_SUMMARY_OUTPUT = Path("data/reports/audio_classes/ttas_panns_cnn14_primary_audio_class_counts.json")
DEFAULT_UNKNOWN_OUTPUT = Path("data/reports/audio_classes/ttas_panns_cnn14_unknown_audio_class_samples.jsonl")


@dataclass(frozen=True)
class ClassAssignment:
    primary_audio_class: str
    matched_audio_class_tags: list[str]
    debug: dict[str, Any]


def assign_primary_audio_class(
    retained_tags: list[str],
    retained_scores: dict[str, float],
    config: dict[str, Any],
    *,
    assignment_rule: str | None = None,
    tag_to_classes: dict[str, list[str]] | None = None,
) -> ClassAssignment:
    """Return the coarse class for one sample and compact audit metadata."""
    rule = assignment_rule or config["assignment_rule"]
    _validate_assignment_rule(rule, config)
    unknown_class = config["unknown_class"]
    tag_to_classes = tag_to_classes or _build_tag_to_classes(config["class_map"])
    candidate_scores = _candidate_class_scores(retained_tags, retained_scores, tag_to_classes)
    guard_class = _choose_with_evidence_guard(candidate_scores, config) if candidate_scores else None

    if not candidate_scores:
        return ClassAssignment(
            primary_audio_class=unknown_class,
            matched_audio_class_tags=[],
            debug={
                "assignment_rule": rule,
                "candidate_class_scores": {},
                "strongest_class_by_score": None,
                "evidence_guard_class": None,
                "priority_overrode_stronger_score": False,
                "weak_priority_assignment": False,
                "ambiguous_matched_tags": {},
            },
        )

    priority_class = _choose_by_priority(candidate_scores, config["priority_order"])
    if rule == "priority_first_v1":
        assigned_class = priority_class
    elif rule == "priority_with_evidence_guard":
        assigned_class = guard_class or priority_class
    else:  # Guarded by _validate_assignment_rule; kept for defensive completeness.
        raise ValueError(f"Unhandled assignment_rule {rule!r}")

    strongest_class = _choose_by_score(candidate_scores)
    assigned_scores = candidate_scores[assigned_class]
    strongest_sum = candidate_scores[strongest_class]["score_sum"]
    assigned_sum = assigned_scores["score_sum"]
    priority_overrode = assigned_class != strongest_class and strongest_sum > assigned_sum

    debug = {
        "assignment_rule": rule,
        "candidate_class_scores": candidate_scores,
        "strongest_class_by_score": strongest_class,
        "strongest_class_score_sum": strongest_sum,
        "assigned_class_score_sum": assigned_sum,
        "assigned_class_score_max": assigned_scores["score_max"],
        "evidence_guard_class": guard_class,
        "evidence_guard_disagrees": guard_class is not None and guard_class != priority_class,
        "priority_overrode_stronger_score": priority_overrode,
        "weak_priority_assignment": _is_weak_assignment(assigned_scores, config),
        "ambiguous_matched_tags": _ambiguous_matched_tags(retained_tags, tag_to_classes),
    }
    return ClassAssignment(
        primary_audio_class=assigned_class,
        matched_audio_class_tags=list(assigned_scores["matched_tags"]),
        debug=debug,
    )


def assign_splits(
    *,
    config_path: Path,
    input_manifests: dict[str, Path],
    output_root: Path,
    summary_output: Path,
    unknown_output: Path | None,
    splits: list[str] | None,
    assignment_rule: str | None,
    include_debug: bool,
    max_samples: int | None = None,
) -> dict[str, Any]:
    config = _read_json(config_path)
    rule = assignment_rule or config["assignment_rule"]
    _validate_assignment_rule(rule, config)
    tag_to_classes = _build_tag_to_classes(config["class_map"])
    split_names = splits or list(input_manifests)
    include_debug = include_debug and config.get("debug", {}).get("write_candidate_class_scores", True)
    summary = _empty_summary(config, config_path, rule, include_debug)
    class_counts: Counter[str] = Counter()
    class_counts_by_split: dict[str, Counter[str]] = {}
    unknown_rows: list[dict[str, Any]] = []

    for split in split_names:
        if split not in input_manifests:
            available = ", ".join(input_manifests)
            raise ValueError(f"Unknown split {split!r}; available splits: {available}")
        input_path = input_manifests[split]
        output_path = output_root / f"ttas_{split}_panns_cnn14_classes.jsonl"
        split_counts: Counter[str] = Counter()
        processed = _assign_manifest(
            split=split,
            input_path=input_path,
            output_path=output_path,
            config=config,
            assignment_rule=rule,
            tag_to_classes=tag_to_classes,
            include_debug=include_debug,
            max_samples=max_samples,
            class_counts=class_counts,
            split_counts=split_counts,
            summary=summary,
            unknown_rows=unknown_rows,
        )
        class_counts_by_split[split] = split_counts
        summary["splits"][split] = {
            "input_manifest": str(input_path),
            "output_manifest": str(output_path),
            "processed": processed,
            "primary_audio_class_counts": _sorted_counter(split_counts),
        }
        print(f"wrote {output_path} split={split} rows={processed}")

    summary["total_samples_processed"] = sum(item["processed"] for item in summary["splits"].values())
    summary["primary_audio_class_counts"] = _sorted_counter(class_counts)
    summary["primary_audio_class_counts_by_split"] = {
        split: _sorted_counter(counts) for split, counts in class_counts_by_split.items()
    }
    summary["unknown_count"] = class_counts.get(config["unknown_class"], 0)

    if unknown_output is not None:
        unknown_output.parent.mkdir(parents=True, exist_ok=True)
        with unknown_output.open("w", encoding="utf-8") as dst:
            for row in unknown_rows:
                dst.write(json.dumps(row, ensure_ascii=False) + "\n")
        summary["unknown_samples_output"] = str(unknown_output)

    summary = _finalize_summary(summary)
    summary_output.parent.mkdir(parents=True, exist_ok=True)
    summary_output.write_text(json.dumps(summary, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(f"wrote {summary_output} total_samples_processed={summary['total_samples_processed']}")
    return summary


def _assign_manifest(
    *,
    split: str,
    input_path: Path,
    output_path: Path,
    config: dict[str, Any],
    assignment_rule: str | None,
    tag_to_classes: dict[str, list[str]],
    include_debug: bool,
    max_samples: int | None,
    class_counts: Counter[str],
    split_counts: Counter[str],
    summary: dict[str, Any],
    unknown_rows: list[dict[str, Any]],
) -> int:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    processed = 0
    with input_path.open("r", encoding="utf-8") as src, output_path.open("w", encoding="utf-8") as dst:
        for line in src:
            if max_samples is not None and processed >= max_samples:
                break
            if not line.strip():
                continue
            row = json.loads(line)
            assignment = assign_primary_audio_class(
                list(row.get("retained_audio_tags") or []),
                _normalize_score_map(row.get("retained_audio_tag_scores") or {}),
                config,
                assignment_rule=assignment_rule,
                tag_to_classes=tag_to_classes,
            )
            row.setdefault("sample_id", Path(row["audio_file"]).stem)
            row["primary_audio_class"] = assignment.primary_audio_class
            row["matched_audio_class_tags"] = assignment.matched_audio_class_tags
            if include_debug:
                row["audio_class_assignment_debug"] = assignment.debug

            class_counts.update([assignment.primary_audio_class])
            split_counts.update([assignment.primary_audio_class])
            _update_debug_summary(summary, assignment.debug)
            if assignment.primary_audio_class == config["unknown_class"]:
                unknown_rows.append(row)
            dst.write(json.dumps(row, ensure_ascii=False) + "\n")
            processed += 1
    return processed


def _validate_assignment_rule(rule: str, config: dict[str, Any]) -> None:
    if rule not in config.get("available_assignment_rules", [config["assignment_rule"]]):
        available = ", ".join(config.get("available_assignment_rules", []))
        raise ValueError(f"Unknown assignment_rule {rule!r}; available rules: {available}")


def _build_tag_to_classes(class_map: dict[str, list[str]]) -> dict[str, list[str]]:
    tag_to_classes: dict[str, list[str]] = defaultdict(list)
    for class_name, tags in class_map.items():
        for tag in tags:
            tag_to_classes[tag].append(class_name)
    return dict(tag_to_classes)


def _candidate_class_scores(
    retained_tags: list[str],
    retained_scores: dict[str, float],
    tag_to_classes: dict[str, list[str]],
) -> dict[str, dict[str, Any]]:
    candidates: dict[str, dict[str, Any]] = {}
    for tag in retained_tags:
        if tag not in tag_to_classes:
            continue
        score = float(retained_scores.get(tag, 0.0))
        for class_name in tag_to_classes[tag]:
            candidate = candidates.setdefault(
                class_name,
                {"score_sum": 0.0, "score_max": 0.0, "matched_tag_count": 0, "matched_tags": []},
            )
            candidate["score_sum"] += score
            candidate["score_max"] = max(candidate["score_max"], score)
            candidate["matched_tag_count"] += 1
            candidate["matched_tags"].append(tag)
    for candidate in candidates.values():
        candidate["score_sum"] = float(candidate["score_sum"])
        candidate["score_max"] = float(candidate["score_max"])
    return candidates


def _choose_by_priority(candidate_scores: dict[str, dict[str, Any]], priority_order: list[str]) -> str:
    for class_name in priority_order:
        if class_name in candidate_scores:
            return class_name
    return next(iter(candidate_scores))


def _choose_by_score(candidate_scores: dict[str, dict[str, Any]]) -> str:
    return max(
        candidate_scores,
        key=lambda class_name: (
            candidate_scores[class_name]["score_sum"],
            candidate_scores[class_name]["score_max"],
            candidate_scores[class_name]["matched_tag_count"],
            class_name,
        ),
    )


def _choose_with_evidence_guard(candidate_scores: dict[str, dict[str, Any]], config: dict[str, Any]) -> str | None:
    guard = config["evidence_guard"]
    strongest_class = _choose_by_score(candidate_scores)
    strongest_sum = candidate_scores[strongest_class]["score_sum"]
    passing: dict[str, dict[str, Any]] = {}
    for class_name, scores in candidate_scores.items():
        if (
            scores["score_max"] >= guard["max_score_threshold"]
            or scores["score_sum"] >= guard["sum_score_threshold"]
            or scores["score_sum"] >= strongest_sum * guard["strongest_sum_ratio"]
        ):
            passing[class_name] = scores
    if passing:
        return _choose_by_priority(passing, config["priority_order"])
    return strongest_class


def _is_weak_assignment(scores: dict[str, Any], config: dict[str, Any]) -> bool:
    guard = config["evidence_guard"]
    return scores["score_max"] < guard["max_score_threshold"] and scores["score_sum"] < guard["sum_score_threshold"]


def _ambiguous_matched_tags(retained_tags: list[str], tag_to_classes: dict[str, list[str]]) -> dict[str, list[str]]:
    return {
        tag: tag_to_classes[tag]
        for tag in retained_tags
        if tag in tag_to_classes and len(tag_to_classes[tag]) > 1
    }


def _update_debug_summary(summary: dict[str, Any], debug: dict[str, Any]) -> None:
    if debug["priority_overrode_stronger_score"]:
        summary["priority_overrode_stronger_score_count"] += 1
    if debug["weak_priority_assignment"]:
        summary["weak_priority_assignment_count"] += 1
    if debug.get("evidence_guard_disagrees"):
        summary["evidence_guard_disagreement_count"] += 1
    for tag in debug.get("ambiguous_matched_tags", {}):
        summary["ambiguous_matched_tag_counts"][tag] += 1


def _empty_summary(config: dict[str, Any], config_path: Path, assignment_rule: str, include_debug: bool) -> dict[str, Any]:
    return {
        "schema_version": 1,
        "config_path": str(config_path),
        "source_document": config.get("source_document"),
        "assignment_rule": assignment_rule,
        "unknown_class": config["unknown_class"],
        "include_debug": include_debug,
        "splits": {},
        "total_samples_processed": 0,
        "unknown_count": 0,
        "primary_audio_class_counts": {},
        "primary_audio_class_counts_by_split": {},
        "priority_overrode_stronger_score_count": 0,
        "weak_priority_assignment_count": 0,
        "evidence_guard_disagreement_count": 0,
        "ambiguous_matched_tag_counts": Counter(),
    }


def _finalize_summary(summary: dict[str, Any]) -> dict[str, Any]:
    summary["ambiguous_matched_tag_counts"] = _sorted_counter(summary["ambiguous_matched_tag_counts"])
    return summary


def _sorted_counter(counter: Counter[str]) -> dict[str, int]:
    return dict(sorted(counter.items(), key=lambda item: (-item[1], item[0])))


def _normalize_score_map(value: dict[str, Any]) -> dict[str, float]:
    return {str(tag): float(score) for tag, score in value.items()}


def _read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _parse_manifest_arg(values: list[str] | None) -> dict[str, Path]:
    if not values:
        return dict(DEFAULT_INPUT_MANIFESTS)
    manifests: dict[str, Path] = {}
    for value in values:
        if "=" not in value:
            raise ValueError("--input-manifest values must use SPLIT=PATH")
        split, path = value.split("=", 1)
        manifests[split] = Path(path)
    return manifests


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument(
        "--input-manifest",
        action="append",
        help="Input manifest as SPLIT=PATH. Repeat for multiple splits. Defaults to TTAS train/val/test tag manifests.",
    )
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT)
    parser.add_argument("--summary-output", type=Path, default=DEFAULT_SUMMARY_OUTPUT)
    parser.add_argument("--unknown-output", type=Path, default=DEFAULT_UNKNOWN_OUTPUT)
    parser.add_argument("--no-unknown-output", action="store_true")
    parser.add_argument("--split", action="append", help="Split to process. Repeat to process multiple splits.")
    parser.add_argument("--assignment-rule", default=None)
    parser.add_argument("--max-samples", type=int, default=None)
    parser.add_argument("--include-debug", dest="include_debug", action="store_true", default=True)
    parser.add_argument("--no-debug", dest="include_debug", action="store_false")
    return parser


def main() -> None:
    args = _build_parser().parse_args()
    if args.max_samples is not None and args.max_samples < 0:
        raise ValueError("--max-samples must be non-negative")
    assign_splits(
        config_path=args.config,
        input_manifests=_parse_manifest_arg(args.input_manifest),
        output_root=args.output_root,
        summary_output=args.summary_output,
        unknown_output=None if args.no_unknown_output else args.unknown_output,
        splits=args.split,
        assignment_rule=args.assignment_rule,
        include_debug=args.include_debug,
        max_samples=args.max_samples,
    )


if __name__ == "__main__":
    main()
