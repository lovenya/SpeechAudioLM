#!/usr/bin/env python
"""Build direction-specific TTAS clean-v2 manifests from G-Eval audio v6."""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from collections import Counter
from pathlib import Path
from typing import Any, Iterable

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from project_paths import root_path


SPLITS = ("train", "val", "test")
DIRECTIONS = ("a2s", "s2a")


def main() -> None:
    args = _parser().parse_args()
    config = _read_json(args.config)
    dataset_config = _read_json(_repo_path(config["source_split_config"]))
    output_root = _repo_path(config["outputs"]["manifest_root"])
    report_root = _repo_path(config["outputs"]["report_root"])
    output_paths = {
        (split, direction): output_root / direction / f"ttas_{split}.jsonl"
        for split in SPLITS
        for direction in DIRECTIONS
    }
    ledger_path = report_root / config["outputs"]["eligibility_ledger"]
    summary_json_path = report_root / config["outputs"]["summary_json"]
    summary_md_path = report_root / config["outputs"]["summary_markdown"]
    all_outputs = [*output_paths.values(), ledger_path, summary_json_path, summary_md_path]
    existing = [path for path in all_outputs if path.exists()]
    if existing and not args.overwrite:
        paths = "\n".join(f"  {path}" for path in existing)
        raise SystemExit(f"error: outputs already exist (pass --overwrite):\n{paths}")

    merged_root = root_path("results_root") / config["source_result"]["merged_root_relative_to_results"]
    split_rows = {
        split: list(_iter_jsonl(_repo_path(dataset_config["splits"][split])))
        for split in SPLITS
    }
    split_ids = {split: [_sample_id(row) for row in rows] for split, rows in split_rows.items()}
    _validate_source_ids(split_ids)

    decisions: dict[tuple[str, str], dict[str, dict[str, Any]]] = {}
    for split in SPLITS:
        expected_ids = set(split_ids[split])
        for direction in DIRECTIONS:
            score_path = merged_root / f"dataset_{split}_{direction}" / "backend_output.jsonl"
            decisions[(split, direction)] = _load_decisions(
                score_path=score_path,
                expected_ids=expected_ids,
                config=config,
            )

    for path in all_outputs:
        path.parent.mkdir(parents=True, exist_ok=True)

    stats: dict[str, dict[str, Any]] = {direction: {} for direction in DIRECTIONS}
    for split in SPLITS:
        rows = split_rows[split]
        for direction in DIRECTIONS:
            by_id = decisions[(split, direction)]
            kept_rows = [row for row in rows if by_id[_sample_id(row)]["keep"]]
            output_path = output_paths[(split, direction)]
            _write_jsonl(output_path, kept_rows)
            reason_counts = Counter(
                reason
                for decision in by_id.values()
                for reason in decision["exclusion_reasons"]
            )
            stats[direction][split] = {
                "source_rows": len(rows),
                "kept_rows": len(kept_rows),
                "excluded_rows": len(rows) - len(kept_rows),
                "keep_percentage": round(100.0 * len(kept_rows) / len(rows), 6),
                "exclusion_reason_counts": dict(sorted(reason_counts.items())),
                "manifest": str(output_path.relative_to(REPO_ROOT)),
                "sha256": _sha256(output_path),
            }

    _write_ledger(ledger_path, split_rows, decisions, config)
    summary = _summary(config, merged_root, stats, ledger_path, summary_md_path)
    _write_json(summary_json_path, summary)
    summary_md_path.write_text(_summary_markdown(summary), encoding="utf-8")

    print(f"policy={config['dataset_version']}")
    for direction in DIRECTIONS:
        for split in SPLITS:
            item = stats[direction][split]
            print(
                f"{direction}/{split}: source={item['source_rows']} "
                f"kept={item['kept_rows']} excluded={item['excluded_rows']}"
            )
    print(f"ledger={ledger_path.relative_to(REPO_ROOT)}")
    print(f"summary={summary_json_path.relative_to(REPO_ROOT)}")


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--config",
        type=Path,
        default=REPO_ROOT / "configs" / "ttas_geval_audio_v6_clean_v2.json",
    )
    parser.add_argument("--overwrite", action="store_true")
    return parser


def _repo_path(value: str | Path) -> Path:
    path = Path(value)
    return path if path.is_absolute() else REPO_ROOT / path


def _read_json(path: Path) -> dict[str, Any]:
    with path.open(encoding="utf-8") as handle:
        return json.load(handle)


def _iter_jsonl(path: Path) -> Iterable[dict[str, Any]]:
    with path.open(encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if line.strip():
                try:
                    yield json.loads(line)
                except json.JSONDecodeError as exc:
                    raise ValueError(f"Invalid JSON in {path}:{line_number}: {exc}") from exc


def _sample_id(row: dict[str, Any]) -> str:
    audio_file = row.get("audio_file")
    if not isinstance(audio_file, str) or not audio_file:
        raise ValueError(f"Source row lacks audio_file: {row}")
    return Path(audio_file).stem


def _validate_source_ids(split_ids: dict[str, list[str]]) -> None:
    seen: set[str] = set()
    for split, ids in split_ids.items():
        if len(ids) != len(set(ids)):
            raise ValueError(f"Duplicate sample IDs in source {split} split")
        overlap = seen.intersection(ids)
        if overlap:
            raise ValueError(f"Sample IDs cross source splits: {sorted(overlap)[:5]}")
        seen.update(ids)


def _load_decisions(
    *,
    score_path: Path,
    expected_ids: set[str],
    config: dict[str, Any],
) -> dict[str, dict[str, Any]]:
    expected_prompt = config["source_result"]["prompt_version"]
    expected_scoring = config["source_result"]["scoring_policy_version"]
    policy = config["exclusion_policy"]
    by_id: dict[str, dict[str, Any]] = {}
    for row in _iter_jsonl(score_path):
        sample_id = row.get("sample_id")
        if not isinstance(sample_id, str) or not sample_id:
            raise ValueError(f"Score row lacks sample_id in {score_path}")
        if sample_id in by_id:
            raise ValueError(f"Duplicate score for {sample_id} in {score_path}")
        if row.get("prompt_version") != expected_prompt:
            raise ValueError(f"Unexpected prompt version for {sample_id} in {score_path}")
        if row.get("scoring_policy_version") != expected_scoring:
            raise ValueError(f"Unexpected scoring policy for {sample_id} in {score_path}")
        criteria = row.get("criteria") if isinstance(row.get("criteria"), dict) else {}
        scenario = _criterion_score(criteria, "scenario_plausibility")
        target = _criterion_score(criteria, "target_modality_success")
        final_score = row.get("final_score")
        reasons = _exclusion_reasons(final_score, scenario, target, policy)
        by_id[sample_id] = {
            "keep": not reasons,
            "status": row.get("status"),
            "final_score": final_score,
            "scenario_plausibility": scenario,
            "target_modality_success": target,
            "exclusion_reasons": reasons,
        }
    actual_ids = set(by_id)
    if actual_ids != expected_ids:
        missing = sorted(expected_ids - actual_ids)[:5]
        extra = sorted(actual_ids - expected_ids)[:5]
        raise ValueError(
            f"Score/source membership mismatch for {score_path}: "
            f"missing={missing}, extra={extra}"
        )
    return by_id


def _criterion_score(criteria: dict[str, Any], name: str) -> Any:
    item = criteria.get(name)
    return item.get("score") if isinstance(item, dict) else None


def _exclusion_reasons(
    final_score: Any,
    scenario: Any,
    target: Any,
    policy: dict[str, Any],
) -> list[str]:
    reasons: list[str] = []
    if final_score is None and policy["null_final_score"]:
        reasons.append("null_final_score")
    elif final_score is not None and final_score <= policy["final_score_lte"]:
        reasons.append("final_score_lte_3")
    if scenario is not None and scenario <= policy["scenario_plausibility_lte"]:
        reasons.append("scenario_plausibility_lte_2")
    if target is not None and target == policy["target_modality_success_eq"]:
        reasons.append("target_modality_success_eq_1")
    return reasons


def _write_jsonl(path: Path, rows: Iterable[dict[str, Any]]) -> None:
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")


def _write_json(path: Path, value: Any) -> None:
    path.write_text(json.dumps(value, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")


def _write_ledger(
    path: Path,
    split_rows: dict[str, list[dict[str, Any]]],
    decisions: dict[tuple[str, str], dict[str, dict[str, Any]]],
    config: dict[str, Any],
) -> None:
    with path.open("w", encoding="utf-8") as handle:
        for split in SPLITS:
            for source in split_rows[split]:
                sample_id = _sample_id(source)
                row = {
                    "schema_version": 1,
                    "policy_version": config["dataset_version"],
                    "prompt_version": config["source_result"]["prompt_version"],
                    "scoring_policy_version": config["source_result"]["scoring_policy_version"],
                    "sample_id": sample_id,
                    "split": split,
                    "audio_file": source.get("audio_file"),
                    "speech_file": source.get("speech_file"),
                    "a2s": decisions[(split, "a2s")][sample_id],
                    "s2a": decisions[(split, "s2a")][sample_id],
                }
                handle.write(json.dumps(row, ensure_ascii=False) + "\n")


def _summary(
    config: dict[str, Any],
    merged_root: Path,
    stats: dict[str, dict[str, Any]],
    ledger_path: Path,
    summary_md_path: Path,
) -> dict[str, Any]:
    totals: dict[str, Any] = {}
    for direction in DIRECTIONS:
        source = sum(stats[direction][split]["source_rows"] for split in SPLITS)
        kept = sum(stats[direction][split]["kept_rows"] for split in SPLITS)
        totals[direction] = {
            "source_rows": source,
            "kept_rows": kept,
            "excluded_rows": source - kept,
            "keep_percentage": round(100.0 * kept / source, 6),
        }
    return {
        "schema_version": 1,
        "dataset_version": config["dataset_version"],
        "source_split_config": config["source_split_config"],
        "source_result_root": str(merged_root),
        "prompt_version": config["source_result"]["prompt_version"],
        "scoring_policy_version": config["source_result"]["scoring_policy_version"],
        "exclusion_policy": config["exclusion_policy"],
        "directions": stats,
        "totals": totals,
        "eligibility_ledger": str(ledger_path.relative_to(REPO_ROOT)),
        "eligibility_ledger_sha256": _sha256(ledger_path),
        "summary_markdown": str(summary_md_path.relative_to(REPO_ROOT)),
    }


def _summary_markdown(summary: dict[str, Any]) -> str:
    lines = [
        "# TTAS G-Eval Audio v6 clean-v2 summary",
        "",
        "Rows are retained independently for A2S and S2A. Original split membership and row order are preserved.",
        "",
        "| Direction | Split | Source | Kept | Excluded | Kept % |",
        "|---|---:|---:|---:|---:|---:|",
    ]
    for direction in DIRECTIONS:
        for split in SPLITS:
            item = summary["directions"][direction][split]
            lines.append(
                f"| {direction.upper()} | {split} | {item['source_rows']:,} | "
                f"{item['kept_rows']:,} | {item['excluded_rows']:,} | "
                f"{item['keep_percentage']:.4f}% |"
            )
        total = summary["totals"][direction]
        lines.append(
            f"| **{direction.upper()}** | **total** | **{total['source_rows']:,}** | "
            f"**{total['kept_rows']:,}** | **{total['excluded_rows']:,}** | "
            f"**{total['keep_percentage']:.4f}%** |"
        )
    lines.extend(
        [
            "",
            "Exclusion is the union of: null final score, final score <= 3, "
            "scenario plausibility <= 2, or target-modality success = 1.",
            "",
        ]
    )
    return "\n".join(lines)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


if __name__ == "__main__":
    main()
