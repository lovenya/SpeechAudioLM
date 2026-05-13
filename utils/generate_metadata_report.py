#!/usr/bin/env python3
"""Generate metadata and split-balance reports for JSONL dataset manifests."""

from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path
from typing import Any


DEFAULT_CONFIG = Path("configs/ttas_dataset.json")
DEFAULT_OUTPUT = Path("data/reports/ttas_dataset_report.json")
SPLIT_ORDER = ("train", "val", "test")


def read_json(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def read_jsonl(path: Path) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    rows: list[dict[str, Any]] = []
    errors: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as f:
        for line_no, line in enumerate(f, start=1):
            try:
                rows.append(json.loads(line))
            except json.JSONDecodeError as exc:
                errors.append({"line": line_no, "error": str(exc)})
    return rows, errors


def pct(part: int | float, total: int | float) -> float:
    if not total:
        return 0.0
    return round((part / total) * 100, 4)


def label_value(row: dict[str, Any], field: str, common_emotions: set[str], rare_bucket: str) -> str:
    if field == "speaker_emotion_bucket":
        raw = str(row.get("speaker_emotion") or "").strip() or "<missing>"
        return raw if raw in common_emotions else rare_bucket
    return str(row.get(field) or "").strip() or "<missing>"


def label_distribution(
    rows: list[dict[str, Any]],
    split_rows: dict[str, list[dict[str, Any]]],
    field: str,
    split_ratio: dict[str, float],
    common_emotions: set[str],
    rare_bucket: str,
) -> dict[str, Any]:
    totals = Counter(label_value(row, field, common_emotions, rare_bucket) for row in rows)
    split_counts = {
        split: Counter(label_value(row, field, common_emotions, rare_bucket) for row in values)
        for split, values in split_rows.items()
    }

    labels: dict[str, Any] = {}
    for label, total in sorted(totals.items(), key=lambda item: (-item[1], item[0])):
        per_split: dict[str, Any] = {}
        max_abs_delta = 0.0
        for split, ratio in split_ratio.items():
            count = split_counts.get(split, Counter()).get(label, 0)
            expected = total * ratio
            delta = count - expected
            max_abs_delta = max(max_abs_delta, abs(delta))
            per_split[split] = {
                "count": count,
                "percent_of_label": pct(count, total),
                "expected_count": round(expected, 4),
                "delta_from_expected": round(delta, 4),
            }
        labels[label] = {
            "total": total,
            "percent_of_dataset": pct(total, len(rows)),
            "splits": per_split,
            "max_abs_delta_from_expected": round(max_abs_delta, 4),
        }
    return labels


def stratification_distribution(
    rows: list[dict[str, Any]],
    split_rows: dict[str, list[dict[str, Any]]],
    split_ratio: dict[str, float],
    common_emotions: set[str],
    rare_bucket: str,
) -> dict[str, Any]:
    def key(row: dict[str, Any]) -> str:
        gender = label_value(row, "speaker_gender", common_emotions, rare_bucket)
        emotion = label_value(row, "speaker_emotion_bucket", common_emotions, rare_bucket)
        return f"{gender}|{emotion}"

    totals = Counter(key(row) for row in rows)
    split_counts = {split: Counter(key(row) for row in values) for split, values in split_rows.items()}
    result: dict[str, Any] = {}
    for label, total in sorted(totals.items(), key=lambda item: (-item[1], item[0])):
        per_split: dict[str, Any] = {}
        for split, ratio in split_ratio.items():
            count = split_counts.get(split, Counter()).get(label, 0)
            expected = total * ratio
            per_split[split] = {
                "count": count,
                "percent_of_stratum": pct(count, total),
                "expected_count": round(expected, 4),
                "delta_from_expected": round(count - expected, 4),
            }
        result[label] = {"total": total, "splits": per_split}
    return result


def balance_summary(labels: dict[str, Any], top_k: int = 10) -> dict[str, Any]:
    by_delta = sorted(
        (
            {
                "label": label,
                "total": item["total"],
                "max_abs_delta_from_expected": item.get("max_abs_delta_from_expected", 0.0),
                "splits": item["splits"],
            }
            for label, item in labels.items()
        ),
        key=lambda item: (-item["max_abs_delta_from_expected"], item["label"]),
    )
    return {
        "largest_count_deltas_from_expected": by_delta[:top_k],
        "note": "Deltas are expected because split counts are integers.",
    }


def stratification_balance_summary(strata: dict[str, Any], top_k: int = 10) -> dict[str, Any]:
    rows = []
    for label, item in strata.items():
        max_delta = max(abs(split["delta_from_expected"]) for split in item["splits"].values())
        rows.append(
            {
                "label": label,
                "total": item["total"],
                "max_abs_delta_from_expected": round(max_delta, 4),
                "splits": item["splits"],
            }
        )
    rows.sort(key=lambda item: (-item["max_abs_delta_from_expected"], item["label"]))
    return {
        "largest_count_deltas_from_expected": rows[:top_k],
        "note": "This is the actual split stratification unit: speaker_gender plus bucketed speaker_emotion.",
    }


def path_validation(
    rows: list[dict[str, Any]],
    dataset_root: Path,
    check_exists: bool,
    max_examples: int,
) -> dict[str, Any]:
    malformed = []
    missing = []
    missing_audio = 0
    missing_speech = 0

    for i, row in enumerate(rows):
        audio_file = str(row.get("audio_file") or "")
        speech_file = str(row.get("speech_file") or "")
        if not audio_file.startswith("audio/") or not speech_file.startswith("speech/"):
            malformed.append({"row": i, "audio_file": audio_file, "speech_file": speech_file})

        if check_exists:
            audio_exists = (dataset_root / audio_file).is_file()
            speech_exists = (dataset_root / speech_file).is_file()
            missing_audio += int(not audio_exists)
            missing_speech += int(not speech_exists)
            if (not audio_exists or not speech_exists) and len(missing) < max_examples:
                missing.append(
                    {
                        "row": i,
                        "audio_file": audio_file,
                        "audio_exists": audio_exists,
                        "speech_file": speech_file,
                        "speech_exists": speech_exists,
                    }
                )

    return {
        "malformed_relative_path_count": len(malformed),
        "malformed_relative_paths": malformed[:max_examples],
        "path_existence_checked": check_exists,
        "missing_audio_paths": missing_audio if check_exists else None,
        "missing_speech_paths": missing_speech if check_exists else None,
        "missing_or_malformed_examples": missing,
    }


def split_membership(rows: list[dict[str, Any]], split_rows: dict[str, list[dict[str, Any]]]) -> dict[str, Any]:
    manifest_ids = Counter(str(row.get("audio_file") or "") for row in rows)
    split_ids: Counter[str] = Counter()
    duplicate_examples: dict[str, int] = {}

    for values in split_rows.values():
        for row in values:
            split_ids[str(row.get("audio_file") or "")] += 1

    for sample_id, count in split_ids.items():
        if count > 1:
            duplicate_examples[sample_id] = count

    missing_from_splits = sorted(sample_id for sample_id in manifest_ids if split_ids[sample_id] == 0)
    extra_in_splits = sorted(sample_id for sample_id in split_ids if manifest_ids[sample_id] == 0)

    return {
        "manifest_rows": len(rows),
        "rows_across_splits": sum(len(values) for values in split_rows.values()),
        "duplicate_audio_files_across_splits": duplicate_examples,
        "missing_from_splits_count": len(missing_from_splits),
        "missing_from_splits_examples": missing_from_splits[:20],
        "extra_in_splits_count": len(extra_in_splits),
        "extra_in_splits_examples": extra_in_splits[:20],
    }


def generate_report(
    config_path: Path,
    check_paths: bool,
    max_examples: int,
) -> dict[str, Any]:
    config = read_json(config_path)
    dataset_root = Path(config["dataset_root"])
    manifest_path = Path(config["canonical_manifest"])
    split_paths = {name: Path(path) for name, path in config["splits"].items()}
    split_ratio = {name: float(value) for name, value in config["split_ratio"].items()}
    stratification = config.get("stratification", {})
    rare_threshold = int(stratification.get("speaker_emotion_rare_threshold", 20))
    rare_bucket = str(stratification.get("rare_bucket", "other"))

    rows, manifest_errors = read_jsonl(manifest_path)
    raw_emotion_counts = Counter(str(row.get("speaker_emotion") or "").strip() or "<missing>" for row in rows)
    common_emotions = {
        label
        for label, count in raw_emotion_counts.items()
        if count >= rare_threshold
    }
    rare_emotions = {
        label: count
        for label, count in sorted(raw_emotion_counts.items(), key=lambda item: (-item[1], item[0]))
        if label not in common_emotions
    }

    split_rows: dict[str, list[dict[str, Any]]] = {}
    split_errors: dict[str, list[dict[str, Any]]] = {}
    for split, path in split_paths.items():
        split_rows[split], split_errors[split] = read_jsonl(path)

    duplicate_ids = {
        sample_id: count
        for sample_id, count in Counter(str(row.get("audio_file") or "") for row in rows).items()
        if count > 1
    }

    gender_labels = label_distribution(
        rows, split_rows, "speaker_gender", split_ratio, common_emotions, rare_bucket
    )
    raw_emotion_labels = label_distribution(
        rows, split_rows, "speaker_emotion", split_ratio, common_emotions, rare_bucket
    )
    bucketed_emotion_labels = label_distribution(
        rows, split_rows, "speaker_emotion_bucket", split_ratio, common_emotions, rare_bucket
    )
    strata_labels = stratification_distribution(rows, split_rows, split_ratio, common_emotions, rare_bucket)

    report = {
        "inputs": {
            "config": str(config_path),
            "dataset_root": str(dataset_root),
            "source_manifest": config.get("source_manifest"),
            "canonical_manifest": str(manifest_path),
            "global_seed": config.get("global_seed"),
            "split_ratio": split_ratio,
        },
        "split_policy": {
            "intended_ratio": split_ratio,
            "stratified_by": ["speaker_gender", "speaker_emotion_bucket"],
            "rare_label_handling": (
                f"speaker_emotion labels with fewer than {rare_threshold} total examples "
                f"are grouped into {rare_bucket!r} for stratified splitting. "
                "Raw emotion labels are still reported individually."
            ),
            "speaker_emotion_rare_threshold": rare_threshold,
            "rare_bucket": rare_bucket,
            "common_emotions": sorted(common_emotions),
            "rare_emotions_bucketed_for_split": rare_emotions,
        },
        "manifest_validation": {
            "rows": len(rows),
            "json_errors": manifest_errors,
            "duplicate_sample_ids": duplicate_ids,
            **path_validation(rows, dataset_root, check_paths, max_examples),
        },
        "split_validation": split_membership(rows, split_rows),
        "split_counts": {
            split: {
                "rows": len(values),
                "percent_of_dataset": pct(len(values), len(rows)),
                "json_errors": split_errors[split],
            }
            for split, values in split_rows.items()
        },
        "proportionality": {
            "summary": (
                "The split is proportional by speaker_gender plus bucketed speaker_emotion, "
                "within integer rounding. Raw rare emotion labels were grouped into the "
                f"{rare_bucket!r} bucket for splitting, so those individual rare labels are "
                "reported but were not independently balanced."
            ),
            "speaker_gender": balance_summary(gender_labels),
            "speaker_emotion_bucket": balance_summary(bucketed_emotion_labels),
            "speaker_gender_x_emotion_bucket": stratification_balance_summary(strata_labels),
            "speaker_emotion_raw": balance_summary(raw_emotion_labels),
        },
        "labels": {
            "speaker_gender": gender_labels,
            "speaker_emotion": raw_emotion_labels,
            "speaker_emotion_bucket": bucketed_emotion_labels,
            "speaker_gender_x_emotion_bucket": strata_labels,
        },
    }
    return report


def format_int(value: int | float) -> str:
    return f"{int(value):,}"


def format_pct(value: int | float) -> str:
    return f"{float(value):.2f}%"


def format_delta(value: int | float) -> str:
    value = float(value)
    if value > 0:
        return f"+{value:.2f}"
    return f"{value:.2f}"


def print_section(title: str) -> None:
    print()
    print(title)
    print("=" * len(title))


def print_table(headers: list[str], rows: list[list[str]]) -> None:
    widths = [len(header) for header in headers]
    for row in rows:
        for i, cell in enumerate(row):
            widths[i] = max(widths[i], len(cell))

    def fmt_row(row: list[str]) -> str:
        cells = []
        for i, cell in enumerate(row):
            if i == 0:
                cells.append(cell.ljust(widths[i]))
            else:
                cells.append(cell.rjust(widths[i]))
        return "  ".join(cells)

    print(fmt_row(headers))
    print(fmt_row(["-" * width for width in widths]))
    for row in rows:
        print(fmt_row(row))


def split_row(label: str, item: dict[str, Any], show_delta: bool) -> list[str]:
    percent_of_dataset = item.get("percent_of_dataset")
    if percent_of_dataset is None:
        percent_of_dataset = pct(item["total"], item.get("dataset_total", 0))
    row = [
        label,
        format_int(item["total"]),
        format_pct(percent_of_dataset),
    ]
    for split in SPLIT_ORDER:
        values = item["splits"][split]
        split_percent = values.get("percent_of_label")
        if split_percent is None:
            split_percent = values.get("percent_of_stratum", 0.0)
        row.extend(
            [
                format_int(values["count"]),
                format_pct(split_percent),
            ]
        )
    if show_delta:
        row.append(format_delta(item.get("max_abs_delta_from_expected", 0.0)))
    return row


def with_dataset_total(labels: dict[str, Any], dataset_total: int) -> dict[str, Any]:
    result = {}
    for label, item in labels.items():
        copied = dict(item)
        copied["dataset_total"] = dataset_total
        copied["percent_of_dataset"] = pct(copied["total"], dataset_total)
        if "max_abs_delta_from_expected" not in copied:
            copied["max_abs_delta_from_expected"] = round(
                max(abs(split.get("delta_from_expected", 0.0)) for split in copied["splits"].values()),
                4,
            )
        result[label] = copied
    return result


def print_label_table(
    title: str,
    labels: dict[str, Any],
    limit: int | None = None,
    omitted_message: str | None = None,
    show_delta: bool = False,
) -> None:
    print_section(title)
    headers = [
        "label",
        "total",
        "% dataset",
        "train",
        "% label",
        "val",
        "% label",
        "test",
        "% label",
    ]
    if show_delta:
        headers.append("max delta")
    items = list(labels.items())
    shown = items if limit is None or limit <= 0 else items[:limit]
    print_table(headers, [split_row(label, item, show_delta) for label, item in shown])
    if limit and limit > 0 and len(items) > limit:
        if omitted_message:
            print(omitted_message.format(count=len(items) - limit))
        else:
            print(f"... {len(items) - limit} more labels omitted.")


def print_other_raw_emotions(report: dict[str, Any], limit: int | None, show_delta: bool) -> None:
    rare_labels = report["split_policy"]["rare_emotions_bucketed_for_split"]
    raw_labels = report["labels"]["speaker_emotion"]
    rows = {
        label: raw_labels[label]
        for label in rare_labels
        if label in raw_labels
    }
    print_label_table(
        "Raw Emotion Labels Inside 'other'",
        rows,
        limit,
        "... {count} more 'other' labels omitted; rerun with --other-labels-limit 0 to show all.",
        show_delta,
    )


def print_balance_table(title: str, balance: dict[str, Any]) -> None:
    print_section(title)
    headers = ["label", "total", "max delta", "train delta", "val delta", "test delta"]
    rows = []
    for item in balance["largest_count_deltas_from_expected"]:
        rows.append(
            [
                item["label"],
                format_int(item["total"]),
                format_delta(item["max_abs_delta_from_expected"]),
                format_delta(item["splits"]["train"]["delta_from_expected"]),
                format_delta(item["splits"]["val"]["delta_from_expected"]),
                format_delta(item["splits"]["test"]["delta_from_expected"]),
            ]
        )
    print_table(headers, rows)
    print(balance["note"])


def print_summary(
    report: dict[str, Any],
    raw_emotions_limit: int | None = None,
    show_delta: bool = False,
) -> None:
    rows = report["manifest_validation"]["rows"]
    policy = report["split_policy"]

    print("TTAS DATASET METADATA REPORT")
    print("============================")
    print(f"Manifest rows       : {format_int(rows)}")
    print(f"Global seed         : {report['inputs']['global_seed']}")
    print(f"Intended split      : train=80%, val=10%, test=10%")
    print(f"Stratified by       : {', '.join(policy['stratified_by'])}")
    print(f"Rare emotion bucket : < {policy['speaker_emotion_rare_threshold']} examples -> {policy['rare_bucket']!r}")
    print(f"JSON errors         : {len(report['manifest_validation']['json_errors'])}")
    print(f"Duplicate IDs       : {len(report['manifest_validation']['duplicate_sample_ids'])}")
    print(f"Missing audio paths : {report['manifest_validation']['missing_audio_paths']}")
    print(f"Missing speech paths: {report['manifest_validation']['missing_speech_paths']}")

    print_section("Split Counts")
    split_rows = []
    for split in SPLIT_ORDER:
        item = report["split_counts"][split]
        split_rows.append([split, format_int(item["rows"]), format_pct(item["percent_of_dataset"])])
    print_table(["split", "rows", "% dataset"], split_rows)

    print_section("Split Policy")
    print(policy["rare_label_handling"])
    print(report["proportionality"]["summary"])
    if show_delta:
        print("Delta means actual split count minus the ideal expected count for an 80/10/10 split.")
    print(f"Common emotion labels: {len(policy['common_emotions'])}")
    print(f"Rare emotion labels bucketed for split: {len(policy['rare_emotions_bucketed_for_split'])}")

    print_label_table("Gender Distribution", report["labels"]["speaker_gender"], show_delta=show_delta)
    print_label_table(
        "Primary Stratification: Gender x Bucketed Emotion",
        with_dataset_total(report["labels"]["speaker_gender_x_emotion_bucket"], rows),
        show_delta=show_delta,
    )
    print_label_table(
        "Bucketed Emotion Distribution",
        report["labels"]["speaker_emotion_bucket"],
        show_delta=show_delta,
    )
    print_other_raw_emotions(report, raw_emotions_limit, show_delta)

    if show_delta:
        print_balance_table(
            "Largest Proportionality Deltas: Stratification Buckets",
            report["proportionality"]["speaker_gender_x_emotion_bucket"],
        )

    print()
    print(f"JSON report written with full details: {report['inputs'].get('output_path', DEFAULT_OUTPUT)}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--print", action="store_true", help="Print a presentation-friendly CLI report.")
    parser.add_argument(
        "--show-delta",
        action="store_true",
        help="Show actual-minus-expected split-count deltas in CLI tables.",
    )
    parser.add_argument(
        "--other-labels-limit",
        type=int,
        default=0,
        help="Limit raw emotion rows inside the 'other' bucket in CLI output. Use 0 to show all.",
    )
    parser.add_argument(
        "--raw-emotions-limit",
        type=int,
        help=argparse.SUPPRESS,
    )
    parser.add_argument(
        "--no-path-check",
        action="store_true",
        help="Skip filesystem existence checks for audio_file and speech_file.",
    )
    parser.add_argument("--max-examples", type=int, default=20)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    report = generate_report(
        config_path=args.config,
        check_paths=not args.no_path_check,
        max_examples=args.max_examples,
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    if args.print:
        report["inputs"]["output_path"] = str(args.output)
        other_labels_limit = args.raw_emotions_limit
        if other_labels_limit is None:
            other_labels_limit = args.other_labels_limit
        print_summary(report, raw_emotions_limit=other_labels_limit, show_delta=args.show_delta)
    else:
        print(f"Wrote {args.output}")


if __name__ == "__main__":
    main()
