#!/usr/bin/env python
"""Enrich TTAS split manifests with AudioSet tags from PANNs CNN14."""

from __future__ import annotations

import argparse
import json
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Protocol


DEFAULT_DATASET_CONFIG = Path("configs/ttas_dataset.json")
DEFAULT_OUTPUT_ROOT = Path("data/manifests/audio_tags")
DEFAULT_SUMMARY_OUTPUT = Path("data/reports/audio_tags/ttas_panns_cnn14_tag_frequency_summary.json")
DEFAULT_TAGGER_NAME = "panns_cnn14_audioset"


class AudioTagger(Protocol):
    name: str

    def predict(self, audio_path: Path) -> list[tuple[str, float]]:
        """Return ``(tag, score)`` pairs for one audio file."""


class PannsCnn14AudioSetTagger:
    """PANNs CNN14 AudioSet tagger using the panns_inference package."""

    name = DEFAULT_TAGGER_NAME

    def __init__(self, device: str = "cuda", checkpoint_path: str | None = None) -> None:
        try:
            import librosa
            from panns_inference import AudioTagging, labels
        except ImportError as exc:
            raise RuntimeError(
                "PANNs tagging requires panns_inference and librosa. "
                "Install them in the metadata/tagging environment before running this CLI."
            ) from exc

        self._audio_tagging = AudioTagging(checkpoint_path=checkpoint_path, device=device)
        self._labels = list(labels)
        self._librosa = librosa

    def predict(self, audio_path: Path) -> list[tuple[str, float]]:
        waveform, _ = self._librosa.load(str(audio_path), sr=32000, mono=True)
        clipwise_output, _embedding = self._audio_tagging.inference(waveform[None, :])
        scores = clipwise_output[0]
        return [(self._labels[index], float(score)) for index, score in enumerate(scores)]


@dataclass(frozen=True)
class RetainedTags:
    tags: list[str]
    scores: dict[str, float]
    primary: str | None
    used_fallback: bool
    no_valid_tags: bool


def retain_tags(
    predictions: list[tuple[str, float]],
    *,
    tag_threshold: float,
    min_tags: int,
    fallback_top_k: int,
    fallback_max_score_gap: float | None = None,
) -> RetainedTags:
    sorted_predictions = sorted(predictions, key=lambda item: item[1], reverse=True)
    threshold_retained = [(tag, score) for tag, score in sorted_predictions if score >= tag_threshold]
    used_fallback = len(threshold_retained) < min_tags
    retained = (
        _fallback_retained(
            sorted_predictions,
            threshold_retained,
            fallback_top_k=fallback_top_k,
            fallback_max_score_gap=fallback_max_score_gap,
        )
        if used_fallback
        else threshold_retained
    )

    tags = [tag for tag, _score in retained]
    scores = {tag: float(score) for tag, score in retained}
    return RetainedTags(
        tags=tags,
        scores=scores,
        primary=tags[0] if tags else None,
        used_fallback=used_fallback,
        no_valid_tags=not tags,
    )


def enrich_splits(
    *,
    dataset_config_path: Path,
    output_root: Path,
    summary_output: Path,
    splits: list[str] | None,
    tagger: AudioTagger,
    tag_threshold: float,
    min_tags: int,
    fallback_top_k: int,
    fallback_max_score_gap: float | None,
    max_samples: int | None,
    store_all_tag_scores: bool = False,
    store_top_tag_scores: int = 0,
    audio_field: str = "audio_file",
) -> dict:
    dataset_config = _read_json(dataset_config_path)
    dataset_root = Path(dataset_config["dataset_root"])
    split_paths = dataset_config["splits"]
    split_names = splits or list(split_paths.keys())

    total_samples = 0
    fallback_top_k_used = 0
    no_valid_tags = 0
    tag_frequencies: Counter[str] = Counter()
    tag_frequencies_by_split: dict[str, Counter[str]] = {}
    split_summary: dict[str, dict[str, str | int]] = {}

    for split in split_names:
        if split not in split_paths:
            raise ValueError(f"Unknown split {split!r}; available splits: {', '.join(split_paths)}")
        input_path = Path(split_paths[split])
        output_path = output_root / f"ttas_{split}_panns_cnn14.jsonl"
        split_counter: Counter[str] = Counter()
        processed = _enrich_manifest(
            split=split,
            input_path=input_path,
            output_path=output_path,
            dataset_root=dataset_root,
            tagger=tagger,
            tag_threshold=tag_threshold,
            min_tags=min_tags,
            fallback_top_k=fallback_top_k,
            fallback_max_score_gap=fallback_max_score_gap,
            max_samples=max_samples,
            store_all_tag_scores=store_all_tag_scores,
            store_top_tag_scores=store_top_tag_scores,
            audio_field=audio_field,
            tag_frequencies=tag_frequencies,
            split_counter=split_counter,
            counters={"fallback_top_k_used": 0, "no_valid_tags": 0},
        )
        total_samples += processed["processed"]
        fallback_top_k_used += processed["fallback_top_k_used"]
        no_valid_tags += processed["no_valid_tags"]
        tag_frequencies_by_split[split] = split_counter
        split_summary[split] = {
            "input_manifest": str(input_path),
            "output_manifest": str(output_path),
            "processed": processed["processed"],
        }

    summary = {
        "audio_tagger_name": tagger.name,
        "tag_threshold": tag_threshold,
        "min_tags": min_tags,
        "fallback_top_k": fallback_top_k,
        "fallback_max_score_gap": fallback_max_score_gap,
        "store_all_tag_scores": store_all_tag_scores,
        "store_top_tag_scores": store_top_tag_scores,
        "splits": split_summary,
        "total_samples_processed": total_samples,
        "fallback_top_k_used": fallback_top_k_used,
        "no_valid_tags": no_valid_tags,
        "tag_frequencies": _sorted_counter(tag_frequencies),
        "tag_frequencies_by_split": {
            split: _sorted_counter(counter) for split, counter in tag_frequencies_by_split.items()
        },
    }
    summary_output.parent.mkdir(parents=True, exist_ok=True)
    summary_output.write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    return summary


def _enrich_manifest(
    *,
    split: str,
    input_path: Path,
    output_path: Path,
    dataset_root: Path,
    tagger: AudioTagger,
    tag_threshold: float,
    min_tags: int,
    fallback_top_k: int,
    fallback_max_score_gap: float | None,
    max_samples: int | None,
    store_all_tag_scores: bool,
    store_top_tag_scores: int,
    audio_field: str,
    tag_frequencies: Counter[str],
    split_counter: Counter[str],
    counters: dict[str, int],
) -> dict[str, int]:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    processed = 0
    with input_path.open("r", encoding="utf-8") as src, output_path.open("w", encoding="utf-8") as dst:
        for line in src:
            if max_samples is not None and processed >= max_samples:
                break
            if not line.strip():
                continue
            row = json.loads(line)
            audio_path = _resolve_audio_path(row, audio_field=audio_field, dataset_root=dataset_root)
            retained, predictions = _predict_retained_tags(
                tagger,
                audio_path=audio_path,
                tag_threshold=tag_threshold,
                min_tags=min_tags,
                fallback_top_k=fallback_top_k,
                fallback_max_score_gap=fallback_max_score_gap,
            )
            if retained.used_fallback:
                counters["fallback_top_k_used"] += 1
            if retained.no_valid_tags:
                counters["no_valid_tags"] += 1

            tag_frequencies.update(retained.tags)
            split_counter.update(retained.tags)
            row.update(
                {
                    "retained_audio_tags": retained.tags,
                    "retained_audio_tag_scores": retained.scores,
                    "primary_audio_tag": retained.primary,
                    "audio_tagger_name": tagger.name,
                }
            )
            if store_all_tag_scores:
                row["all_audio_tag_scores"] = _score_map(predictions)
            if store_top_tag_scores > 0:
                row["top_audio_tag_scores"] = _score_map(predictions, limit=store_top_tag_scores)
            dst.write(json.dumps(row, ensure_ascii=False) + "\n")
            processed += 1
    print(f"wrote {output_path} split={split} rows={processed}")
    return {
        "processed": processed,
        "fallback_top_k_used": counters["fallback_top_k_used"],
        "no_valid_tags": counters["no_valid_tags"],
    }


def _predict_retained_tags(
    tagger: AudioTagger,
    *,
    audio_path: Path,
    tag_threshold: float,
    min_tags: int,
    fallback_top_k: int,
    fallback_max_score_gap: float | None,
) -> tuple[RetainedTags, list[tuple[str, float]]]:
    try:
        predictions = tagger.predict(audio_path)
    except Exception as exc:
        print(f"warning: failed to tag {audio_path}: {exc}")
        predictions = []
    retained = retain_tags(
        predictions,
        tag_threshold=tag_threshold,
        min_tags=min_tags,
        fallback_top_k=fallback_top_k,
        fallback_max_score_gap=fallback_max_score_gap,
    )
    return retained, predictions


def _fallback_retained(
    sorted_predictions: list[tuple[str, float]],
    threshold_retained: list[tuple[str, float]],
    *,
    fallback_top_k: int,
    fallback_max_score_gap: float | None,
) -> list[tuple[str, float]]:
    if fallback_top_k <= 0 or not sorted_predictions:
        return []
    if fallback_max_score_gap is None:
        return sorted_predictions[:fallback_top_k]

    retained = list(threshold_retained)
    start_index = len(threshold_retained)
    if not retained:
        retained.append(sorted_predictions[0])
        start_index = 1

    for tag, score in sorted_predictions[start_index:fallback_top_k]:
        if len(retained) >= fallback_top_k:
            break
        previous_score = retained[-1][1]
        if previous_score - score >= fallback_max_score_gap:
            break
        retained.append((tag, score))
    return retained


def _resolve_audio_path(row: dict, *, audio_field: str, dataset_root: Path) -> Path:
    value = row.get(audio_field)
    if not isinstance(value, str) or not value:
        raise ValueError(f"Row has no usable {audio_field!r}: {row}")
    path = Path(value)
    return path if path.is_absolute() else dataset_root / path


def _sorted_counter(counter: Counter[str]) -> dict[str, int]:
    return dict(sorted(counter.items(), key=lambda item: (-item[1], item[0])))


def _score_map(predictions: list[tuple[str, float]], limit: int | None = None) -> dict[str, float]:
    sorted_predictions = sorted(predictions, key=lambda item: item[1], reverse=True)
    if limit is not None:
        sorted_predictions = sorted_predictions[:limit]
    return {tag: float(score) for tag, score in sorted_predictions}


def _read_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset-config", type=Path, default=DEFAULT_DATASET_CONFIG)
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT)
    parser.add_argument("--summary-output", type=Path, default=DEFAULT_SUMMARY_OUTPUT)
    parser.add_argument("--split", action="append", choices=["train", "val", "test"], help="Split to process. Repeat to process multiple splits.")
    parser.add_argument("--audio-field", default="audio_file")
    parser.add_argument("--tag-threshold", type=float, default=0.6)
    parser.add_argument("--min-tags", type=int, default=2)
    parser.add_argument("--fallback-top-k", type=int, default=3)
    parser.add_argument(
        "--fallback-max-score-gap",
        type=float,
        default=None,
        help="When fallback is used, stop adding lower tags after this score drop.",
    )
    parser.add_argument("--max-samples", type=int, default=None)
    parser.add_argument(
        "--store-all-tag-scores",
        action="store_true",
        help="Add all PANNs AudioSet tag scores to each enriched row.",
    )
    parser.add_argument(
        "--store-top-tag-scores",
        type=int,
        default=0,
        help="Add the top N PANNs AudioSet tag scores to each enriched row.",
    )
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--checkpoint-path", default=None)
    return parser


def main() -> None:
    args = _build_parser().parse_args()
    if args.min_tags < 0:
        raise ValueError("--min-tags must be non-negative")
    if args.fallback_top_k < 0:
        raise ValueError("--fallback-top-k must be non-negative")
    if args.fallback_max_score_gap is not None and args.fallback_max_score_gap < 0:
        raise ValueError("--fallback-max-score-gap must be non-negative")
    if args.max_samples is not None and args.max_samples < 0:
        raise ValueError("--max-samples must be non-negative")
    if args.store_top_tag_scores < 0:
        raise ValueError("--store-top-tag-scores must be non-negative")

    tagger = PannsCnn14AudioSetTagger(device=args.device, checkpoint_path=args.checkpoint_path)
    summary = enrich_splits(
        dataset_config_path=args.dataset_config,
        output_root=args.output_root,
        summary_output=args.summary_output,
        splits=args.split,
        tagger=tagger,
        tag_threshold=args.tag_threshold,
        min_tags=args.min_tags,
        fallback_top_k=args.fallback_top_k,
        fallback_max_score_gap=args.fallback_max_score_gap,
        max_samples=args.max_samples,
        store_all_tag_scores=args.store_all_tag_scores,
        store_top_tag_scores=args.store_top_tag_scores,
        audio_field=args.audio_field,
    )
    print(f"wrote {args.summary_output} total_samples_processed={summary['total_samples_processed']}")


if __name__ == "__main__":
    main()
