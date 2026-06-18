#!/usr/bin/env python
"""Run directional NLI contradiction checks for generated text pairs."""

from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from pathlib import Path
from statistics import mean
from typing import Any, Protocol


DEFAULT_MODEL_NAME = "MoritzLaurer/DeBERTa-v3-large-mnli-fever-anli-ling-wanli"
LABELS = ("entailment", "neutral", "contradiction")
THRESHOLDS = (0.5, 0.7, 0.9)


class NliScorer(Protocol):
    name: str

    def predict(self, premise: str, hypothesis: str) -> dict[str, float]:
        """Return entailment, neutral, and contradiction probabilities."""


@dataclass(frozen=True)
class NliResult:
    forward: dict[str, float]
    reverse: dict[str, float]
    max_contradiction_probability: float
    predicted_label_using_max_contradiction: str


class HuggingFaceNliScorer:
    """Hugging Face sequence-classification NLI adapter."""

    def __init__(
        self,
        model_name_or_path: str,
        *,
        device: str = "cpu",
        batch_size: int = 8,
        local_files_only: bool = False,
    ) -> None:
        try:
            import torch
            from transformers import AutoModelForSequenceClassification, AutoTokenizer
        except ImportError as exc:
            raise RuntimeError("Hugging Face NLI requires transformers and torch.") from exc

        self.name = model_name_or_path
        self._tokenizer = AutoTokenizer.from_pretrained(model_name_or_path, local_files_only=local_files_only)
        self._model = AutoModelForSequenceClassification.from_pretrained(
            model_name_or_path,
            local_files_only=local_files_only,
        )
        self._torch = torch
        self._device = _resolve_device(device, torch)
        self._batch_size = batch_size
        self._model.to(self._device)
        self._model.eval()
        self._label_indices = _label_indices(self._model.config)

    def predict(self, premise: str, hypothesis: str) -> dict[str, float]:
        batch = self.predict_batch([(premise, hypothesis)])
        return batch[0]

    def predict_batch(self, pairs: list[tuple[str, str]]) -> list[dict[str, float]]:
        outputs: list[dict[str, float]] = []
        with self._torch.inference_mode():
            for start in range(0, len(pairs), self._batch_size):
                chunk = pairs[start : start + self._batch_size]
                encoded = self._tokenizer(
                    [premise for premise, _hypothesis in chunk],
                    [hypothesis for _premise, hypothesis in chunk],
                    truncation=True,
                    padding=True,
                    return_tensors="pt",
                )
                encoded = {key: value.to(self._device) for key, value in encoded.items()}
                probabilities = self._torch.softmax(self._model(**encoded).logits, dim=-1).cpu()
                for row in probabilities:
                    outputs.append(
                        {
                            label: float(row[index])
                            for label, index in self._label_indices.items()
                        }
                    )
        return outputs

    def tokenization_metadata(self, premise: str, hypothesis: str) -> dict[str, Any]:
        return _tokenization_metadata(self._tokenizer, premise, hypothesis)


def score_pair(input_text: str, generated_text: str, scorer: NliScorer, *, contradiction_threshold: float) -> NliResult:
    forward = scorer.predict(input_text, generated_text)
    reverse = scorer.predict(generated_text, input_text)
    max_contradiction = max(forward["contradiction"], reverse["contradiction"])
    predicted = "contradiction" if max_contradiction >= contradiction_threshold else "not_contradiction"
    return NliResult(
        forward=forward,
        reverse=reverse,
        max_contradiction_probability=max_contradiction,
        predicted_label_using_max_contradiction=predicted,
    )


def run_nli_helper(
    *,
    input_path: Path,
    output_path: Path,
    summary_output: Path,
    scorer: NliScorer,
    contradiction_threshold: float,
    max_samples: int | None = None,
) -> dict[str, Any]:
    rows = _read_input_rows(input_path, max_samples=max_samples)
    scored_rows: list[dict[str, Any]] = []
    for row in rows:
        input_text = _required_str(row, "input_text")
        generated_text = _required_str(row, "generated_text")
        result = score_pair(
            input_text,
            generated_text,
            scorer,
            contradiction_threshold=contradiction_threshold,
        )
        scored_rows.append(
            _output_row(
                row,
                input_text,
                generated_text,
                result,
                scorer.name,
                tokenization=_bidirectional_tokenization_metadata(scorer, input_text, generated_text),
            )
        )

    summary = _summary(scored_rows, input_path=input_path, output_path=output_path, scorer_name=scorer.name)
    _write_jsonl(output_path, scored_rows)
    _write_json(summary_output, summary)
    print(f"wrote {output_path} rows={summary['total_pairs_processed']}")
    print(f"wrote {summary_output}")
    return summary


def _output_row(
    row: dict[str, Any],
    input_text: str,
    generated_text: str,
    result: NliResult,
    model_name: str,
    *,
    tokenization: dict[str, dict[str, Any]] | None = None,
) -> dict[str, Any]:
    pair_id = row.get("pair_id") or row.get("sample_id")
    output = {
        "row_index": row.get("row_index"),
        "pair_id": pair_id,
        "sample_id": row.get("sample_id"),
        "split": row.get("split"),
        "direction": row.get("direction"),
        "primary_audio_class": row.get("primary_audio_class"),
        "input_text": input_text,
        "generated_text": generated_text,
        "forward_entailment_probability": result.forward["entailment"],
        "forward_neutral_probability": result.forward["neutral"],
        "forward_contradiction_probability": result.forward["contradiction"],
        "reverse_entailment_probability": result.reverse["entailment"],
        "reverse_neutral_probability": result.reverse["neutral"],
        "reverse_contradiction_probability": result.reverse["contradiction"],
        "max_contradiction_probability": result.max_contradiction_probability,
        "predicted_label_using_max_contradiction": result.predicted_label_using_max_contradiction,
    }
    if tokenization:
        output.update(tokenization)
    return {key: value for key, value in output.items() if value is not None}


def _bidirectional_tokenization_metadata(scorer: NliScorer, input_text: str, generated_text: str) -> dict[str, dict[str, Any]]:
    metadata_fn = getattr(scorer, "tokenization_metadata", None)
    if not callable(metadata_fn):
        return {}
    return {
        "forward_tokenization": metadata_fn(input_text, generated_text),
        "reverse_tokenization": metadata_fn(generated_text, input_text),
    }


def _tokenization_metadata(tokenizer: Any, premise: str, hypothesis: str) -> dict[str, Any]:
    premise_tokens = tokenizer(premise, add_special_tokens=False)["input_ids"]
    hypothesis_tokens = tokenizer(hypothesis, add_special_tokens=False)["input_ids"]
    raw_pair = tokenizer(
        premise,
        hypothesis,
        truncation=False,
        padding=False,
        add_special_tokens=True,
    )["input_ids"]
    truncated_pair = tokenizer(
        premise,
        hypothesis,
        truncation=True,
        padding=False,
        add_special_tokens=True,
    )["input_ids"]
    return {
        "tokenizer_max_length": _tokenizer_max_length(tokenizer),
        "truncation_enabled": True,
        "truncation_strategy": "longest_first",
        "truncation_side": getattr(tokenizer, "truncation_side", None),
        "premise_token_count_raw": len(premise_tokens),
        "hypothesis_token_count_raw": len(hypothesis_tokens),
        "special_token_count_pair": int(tokenizer.num_special_tokens_to_add(pair=True)),
        "pair_token_count_raw": len(raw_pair),
        "pair_token_count_after_truncation": len(truncated_pair),
        "was_truncated": len(raw_pair) > len(truncated_pair),
    }


def _tokenizer_max_length(tokenizer: Any) -> int | None:
    max_length = getattr(tokenizer, "model_max_length", None)
    if isinstance(max_length, int) and 0 < max_length < 1_000_000:
        return max_length
    return None


def _summary(rows: list[dict[str, Any]], *, input_path: Path, output_path: Path, scorer_name: str) -> dict[str, Any]:
    max_scores = [float(row["max_contradiction_probability"]) for row in rows]
    return {
        "schema_version": 1,
        "input_path": str(input_path),
        "output_path": str(output_path),
        "nli_model": scorer_name,
        "total_pairs_processed": len(rows),
        "mean_max_contradiction_probability": mean(max_scores) if max_scores else 0.0,
        "contradiction_rate_at_thresholds": {
            str(threshold): _rate(max_scores, threshold) for threshold in THRESHOLDS
        },
        "predicted_label_counts": _label_counts(rows),
        "top_flagged_examples": sorted(
            rows,
            key=lambda row: float(row["max_contradiction_probability"]),
            reverse=True,
        )[:20],
        "interpretation_note": "High contradiction is a helper flag. Neutral is not failure and entailment is not required.",
    }


def _read_input_rows(input_path: Path, *, max_samples: int | None) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with input_path.open("r", encoding="utf-8") as src:
        for line in src:
            if not line.strip():
                continue
            rows.append(json.loads(line))
            if max_samples is not None and len(rows) >= max_samples:
                break
    return rows


def _label_counts(rows: list[dict[str, Any]]) -> dict[str, int]:
    counts: dict[str, int] = {}
    for row in rows:
        label = row["predicted_label_using_max_contradiction"]
        counts[label] = counts.get(label, 0) + 1
    return dict(sorted(counts.items()))


def _rate(scores: list[float], threshold: float) -> float:
    if not scores:
        return 0.0
    return sum(score >= threshold for score in scores) / len(scores)


def _required_str(row: dict[str, Any], field: str) -> str:
    value = row.get(field)
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"Row has no usable {field!r}: {row}")
    return value

def _label_indices(config: Any) -> dict[str, int]:
    id2label = {int(index): str(label).lower() for index, label in config.id2label.items()}
    indices: dict[str, int] = {}
    for label in LABELS:
        matches = [index for index, name in id2label.items() if label in name]
        if not matches:
            raise RuntimeError(f"Could not find NLI label {label!r} in model labels: {id2label}")
        indices[label] = matches[0]
    return indices


def _resolve_device(device: str, torch_module: Any) -> str:
    if device == "cuda" and not torch_module.cuda.is_available():
        return "cpu"
    return device


def _write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as dst:
        for row in rows:
            dst.write(json.dumps(row, ensure_ascii=False) + "\n")


def _write_json(path: Path, data: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")


def _build_scorer(args: argparse.Namespace) -> NliScorer:
    return HuggingFaceNliScorer(
        args.model,
        device=args.device,
        batch_size=args.batch_size,
        local_files_only=args.local_files_only,
    )


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, required=True, help="Input JSONL with direction, input_text, and generated_text.")
    parser.add_argument("--output", type=Path, required=True, help="Output NLI result JSONL.")
    parser.add_argument("--summary-output", type=Path, required=True)
    parser.add_argument("--backend", choices=["hf"], default="hf")
    parser.add_argument("--model", default=DEFAULT_MODEL_NAME)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--contradiction-threshold", type=float, default=0.7)
    parser.add_argument("--max-samples", type=int, default=None)
    parser.add_argument(
        "--local-files-only",
        action="store_true",
        help="Do not contact Hugging Face; load only local model files.",
    )
    return parser


def main() -> None:
    args = _build_parser().parse_args()
    if args.batch_size <= 0:
        raise ValueError("--batch-size must be positive")
    if args.max_samples is not None and args.max_samples < 0:
        raise ValueError("--max-samples must be non-negative")
    scorer = _build_scorer(args)
    run_nli_helper(
        input_path=args.input,
        output_path=args.output,
        summary_output=args.summary_output,
        scorer=scorer,
        contradiction_threshold=args.contradiction_threshold,
        max_samples=args.max_samples,
    )


if __name__ == "__main__":
    main()
