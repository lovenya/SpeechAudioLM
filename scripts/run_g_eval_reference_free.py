#!/usr/bin/env python
"""Run reference-free G-Eval judging for T2T SpeechAudio LM outputs."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import re
from dataclasses import dataclass
from pathlib import Path
from statistics import mean
from typing import Any, Protocol


PROMPT_VERSION = "g_eval_reference_free_v1"
DEFAULT_MODEL_PATH = "/scratch/l/lovenya/salm_migration_clean/models/eval/prometheus-8x7b-v2.0"
CRITERIA = (
    "non_contradiction",
    "causal_situational_plausibility",
    "physical_grounding",
    "contextual_emotional_fit",
    "specificity_without_over_invention",
    "creativity_valid_diversity",
    "naturalness",
    "overall_score",
)


class Judge(Protocol):
    name: str

    def generate(self, prompts: list[str]) -> list[str]:
        """Return one raw judge response per prompt."""


@dataclass(frozen=True)
class EvalItem:
    row: dict[str, Any]
    prompt: str
    prompt_hash: str


class StubJudge:
    name = "stub_g_eval_reference_free"

    def generate(self, prompts: list[str]) -> list[str]:
        outputs = []
        for prompt in prompts:
            lower = prompt.lower()
            contradiction = "silent" in lower and any(token in lower for token in ("noise", "loud", "traffic"))
            score = 2 if contradiction else 4
            outputs.append(
                json.dumps(
                    {
                        "non_contradiction": score,
                        "causal_situational_plausibility": score,
                        "physical_grounding": 4,
                        "contextual_emotional_fit": score,
                        "specificity_without_over_invention": 4,
                        "creativity_valid_diversity": 4,
                        "naturalness": 4,
                        "overall_score": score,
                        "should_flag": contradiction,
                        "failure_modes": ["possible_contradiction"] if contradiction else [],
                        "short_rationale": "Stub deterministic judgment for pipeline tests.",
                    }
                )
            )
        return outputs


class VllmPrometheusJudge:
    def __init__(
        self,
        model_path: str,
        *,
        tensor_parallel_size: int,
        temperature: float,
        max_tokens: int,
        gpu_memory_utilization: float,
        trust_remote_code: bool,
    ) -> None:
        try:
            from vllm import LLM, SamplingParams
        except ImportError as exc:
            raise RuntimeError("Prometheus G-Eval requires vllm in the configured metric environment.") from exc

        self.name = model_path
        self._sampling_params = SamplingParams(temperature=temperature, max_tokens=max_tokens)
        self._llm = LLM(
            model=model_path,
            tensor_parallel_size=tensor_parallel_size,
            trust_remote_code=trust_remote_code,
            gpu_memory_utilization=gpu_memory_utilization,
        )
        self._tokenizer = self._llm.get_tokenizer()

    def generate(self, prompts: list[str]) -> list[str]:
        formatted = [self._format_prompt(prompt) for prompt in prompts]
        responses = self._llm.generate(formatted, self._sampling_params)
        return [item.outputs[0].text.strip() for item in responses]

    def _format_prompt(self, prompt: str) -> str:
        if hasattr(self._tokenizer, "apply_chat_template"):
            try:
                return self._tokenizer.apply_chat_template(
                    [{"role": "user", "content": prompt}],
                    tokenize=False,
                    add_generation_prompt=True,
                )
            except Exception:
                pass
        return prompt


def run_g_eval(
    *,
    input_path: Path,
    output_path: Path,
    summary_output: Path,
    summary_csv_output: Path | None,
    judge: Judge,
    cache_dir: Path | None,
    batch_size: int,
    max_samples: int | None = None,
) -> dict[str, Any]:
    rows = _read_input_rows(input_path, max_samples=max_samples)
    items = [_make_item(row) for row in rows]
    cache_dir.mkdir(parents=True, exist_ok=True) if cache_dir else None

    output_rows: list[dict[str, Any]] = []
    pending: list[EvalItem] = []
    for item in items:
        cached = _read_cache(cache_dir, item.prompt_hash)
        if cached is None:
            pending.append(item)
        else:
            output_rows.append(_row_from_judgment(item.row, item.prompt_hash, cached, cached.get("raw_judge_response", ""), judge.name))

    for start in range(0, len(pending), batch_size):
        chunk = pending[start : start + batch_size]
        raw_responses = judge.generate([item.prompt for item in chunk])
        for item, raw in zip(chunk, raw_responses):
            try:
                parsed = _parse_judge_json(raw)
                parsed["parse_error"] = None
            except Exception as exc:
                parsed = _error_judgment(raw, f"{type(exc).__name__}: {exc}")
            parsed["raw_judge_response"] = raw
            _write_cache(cache_dir, item.prompt_hash, parsed)
            output_rows.append(_row_from_judgment(item.row, item.prompt_hash, parsed, raw, judge.name))

    output_rows.sort(key=lambda row: int(row.get("row_index", 0)))
    summary = _summary(output_rows, input_path=input_path, output_path=output_path, judge_name=judge.name)
    _write_jsonl(output_path, output_rows)
    _write_json(summary_output, summary)
    if summary_csv_output:
        _write_summary_csv(summary_csv_output, summary)
    print(f"wrote {output_path} rows={summary['total_pairs_processed']}")
    print(f"wrote {summary_output}")
    return summary


def _make_item(row: dict[str, Any]) -> EvalItem:
    input_text = _required_str(row, "input_text")
    generated_text = _required_str(row, "generated_text")
    prompt = build_prompt(
        direction=str(row.get("direction") or "text_to_text"),
        input_text=input_text,
        generated_text=generated_text,
        nli_summary=_nli_summary(row),
    )
    prompt_hash = hashlib.sha256(prompt.encode("utf-8")).hexdigest()
    return EvalItem(row=row, prompt=prompt, prompt_hash=prompt_hash)


def build_prompt(*, direction: str, input_text: str, generated_text: str, nli_summary: str) -> str:
    return f'''You are evaluating a creative cross-modal generation system.

The task is not paraphrase matching and not exact reference matching. The generated output may be creative. Multiple different outputs can be valid.

Direction: {direction}

Input text:
{input_text}

Generated output text:
{generated_text}

Optional NLI contradiction signal:
{nli_summary}

Evaluate the generated output using the criteria below.

Criteria:

1. Non-contradiction
   Does the generated output contradict the input text?
   Score 1 if it clearly contradicts the input.
   Score 5 if it is fully non-contradictory.

2. Causal / situational plausibility
   For speech_to_audio: could the described audio scene plausibly cause, explain, or contextualize the speech?
   For audio_to_speech: could a human plausibly say this utterance in or because of the described audio scene?

3. Physical grounding
   Are the sound sources, scene, and implied events physically coherent and realistic?

4. Contextual / emotional fit
   Does the generated output fit the implied situation, intent, mood, or human reaction?

5. Specificity without unsupported over-invention
   Reward concrete and useful outputs.
   Penalize outputs that are too vague, but also penalize outputs that invent overly specific unsupported details.
   Do not reward verbosity for its own sake.

6. Creativity / valid diversity
   Allow multiple plausible alternatives.
   Do not penalize a valid creative output simply because it is not the most obvious output.

7. Naturalness
   For generated speech text: does it sound like a natural human utterance?
   For generated audio descriptions: does it sound like a coherent natural audio scene description?

Return strict JSON only:

{{
"non_contradiction": <1-5>,
"causal_situational_plausibility": <1-5>,
"physical_grounding": <1-5>,
"contextual_emotional_fit": <1-5>,
"specificity_without_over_invention": <1-5>,
"creativity_valid_diversity": <1-5>,
"naturalness": <1-5>,
"overall_score": <1-5>,
"should_flag": true/false,
"failure_modes": ["..."],
"short_rationale": "..."
}}

Scoring guidance:

* 5 = excellent
* 4 = good with minor issues
* 3 = partially plausible but flawed
* 2 = weak or poorly grounded
* 1 = contradictory, irrelevant, or incoherent

If the output is strongly contradictory, the overall score should be low even if it is fluent.
Do not reward long outputs just because they are detailed.'''


def _nli_summary(row: dict[str, Any]) -> str:
    nli = row.get("nli_contradiction")
    if not isinstance(nli, dict):
        return "Not available."
    parts = []
    for key in (
        "forward_contradiction_probability",
        "reverse_contradiction_probability",
        "max_contradiction_probability",
        "predicted_label_using_max_contradiction",
    ):
        if key in nli:
            parts.append(f"{key}={nli[key]}")
    return "; ".join(parts) if parts else "Not available."


def _parse_judge_json(raw: str) -> dict[str, Any]:
    text = raw.strip()
    if not text.startswith("{"):
        match = re.search(r"\{.*\}", text, flags=re.DOTALL)
        if not match:
            raise ValueError("No JSON object found in judge response")
        text = match.group(0)
    data = json.loads(text)
    if not isinstance(data, dict):
        raise ValueError("Judge response JSON is not an object")
    result: dict[str, Any] = {}
    for key in CRITERIA:
        value = int(data[key])
        if value < 1 or value > 5:
            raise ValueError(f"{key} score out of range: {value}")
        result[key] = value
    result["should_flag"] = bool(data.get("should_flag", result["overall_score"] <= 2))
    failure_modes = data.get("failure_modes", [])
    result["failure_modes"] = [str(item) for item in failure_modes] if isinstance(failure_modes, list) else [str(failure_modes)]
    result["short_rationale"] = str(data.get("short_rationale", "")).strip()
    return result


def _error_judgment(raw: str, message: str) -> dict[str, Any]:
    return {
        **{key: None for key in CRITERIA},
        "should_flag": True,
        "failure_modes": ["judge_parse_error"],
        "short_rationale": "Judge response could not be parsed as valid scoring JSON.",
        "parse_error": message,
        "raw_judge_response": raw,
    }


def _row_from_judgment(row: dict[str, Any], prompt_hash: str, judgment: dict[str, Any], raw: str, judge_name: str) -> dict[str, Any]:
    output = {
        "row_index": row.get("row_index"),
        "pair_id": row.get("pair_id") or row.get("sample_id"),
        "sample_id": row.get("sample_id"),
        "split": row.get("split"),
        "direction": row.get("direction"),
        "primary_audio_class": row.get("primary_audio_class"),
        "input_text": row.get("input_text"),
        "generated_text": row.get("generated_text"),
        "nli_contradiction": row.get("nli_contradiction"),
        "prompt_hash": prompt_hash,
        "judge_model": judge_name,
        "prompt_version": PROMPT_VERSION,
        "raw_judge_response": raw,
    }
    output.update(judgment)
    return {key: value for key, value in output.items() if value is not None}


def _summary(rows: list[dict[str, Any]], *, input_path: Path, output_path: Path, judge_name: str) -> dict[str, Any]:
    valid_rows = [row for row in rows if isinstance(row.get("overall_score"), int)]
    by_direction: dict[str, list[dict[str, Any]]] = {}
    for row in valid_rows:
        by_direction.setdefault(str(row.get("direction") or "unknown"), []).append(row)
    return {
        "schema_version": 1,
        "metric": "g_eval_reference_free",
        "prompt_version": PROMPT_VERSION,
        "input_path": str(input_path),
        "output_path": str(output_path),
        "judge_model": judge_name,
        "total_pairs_processed": len(rows),
        "valid_judgments": len(valid_rows),
        "parse_error_count": sum(1 for row in rows if row.get("parse_error")),
        "should_flag_count": sum(1 for row in rows if row.get("should_flag") is True),
        "mean_scores": _mean_scores(valid_rows),
        "mean_scores_by_direction": {direction: _mean_scores(items) for direction, items in sorted(by_direction.items())},
        "top_flagged_examples": sorted(rows, key=lambda row: (row.get("should_flag") is True, -(row.get("overall_score") or 0)), reverse=True)[:20],
    }


def _mean_scores(rows: list[dict[str, Any]]) -> dict[str, float]:
    return {key: mean([float(row[key]) for row in rows if isinstance(row.get(key), int)]) for key in CRITERIA if any(isinstance(row.get(key), int) for row in rows)}


def _read_input_rows(input_path: Path, *, max_samples: int | None) -> list[dict[str, Any]]:
    rows = []
    with input_path.open("r", encoding="utf-8") as src:
        for line in src:
            if not line.strip():
                continue
            rows.append(json.loads(line))
            if max_samples is not None and len(rows) >= max_samples:
                break
    return rows


def _required_str(row: dict[str, Any], field: str) -> str:
    value = row.get(field)
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"Row has no usable {field!r}: {row}")
    return value


def _read_cache(cache_dir: Path | None, prompt_hash: str) -> dict[str, Any] | None:
    if cache_dir is None:
        return None
    path = cache_dir / f"{prompt_hash}.json"
    if not path.is_file():
        return None
    return json.loads(path.read_text(encoding="utf-8"))


def _write_cache(cache_dir: Path | None, prompt_hash: str, data: dict[str, Any]) -> None:
    if cache_dir is None:
        return
    path = cache_dir / f"{prompt_hash}.json"
    path.write_text(json.dumps(data, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")


def _write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as dst:
        for row in rows:
            dst.write(json.dumps(row, ensure_ascii=False) + "\n")


def _write_json(path: Path, data: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")


def _write_summary_csv(path: Path, summary: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["scope", "criterion", "mean"])
        for key, value in summary.get("mean_scores", {}).items():
            writer.writerow(["all", key, value])
        for direction, scores in summary.get("mean_scores_by_direction", {}).items():
            for key, value in scores.items():
                writer.writerow([direction, key, value])


def _build_judge(args: argparse.Namespace) -> Judge:
    if args.backend == "stub":
        return StubJudge()
    return VllmPrometheusJudge(
        args.model,
        tensor_parallel_size=args.tensor_parallel_size,
        temperature=args.temperature,
        max_tokens=args.max_tokens,
        gpu_memory_utilization=args.gpu_memory_utilization,
        trust_remote_code=args.trust_remote_code,
    )


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--summary-output", type=Path, required=True)
    parser.add_argument("--summary-csv-output", type=Path, default=None)
    parser.add_argument("--cache-dir", type=Path, default=None)
    parser.add_argument("--backend", choices=["vllm", "stub"], default="vllm")
    parser.add_argument("--model", default=DEFAULT_MODEL_PATH)
    parser.add_argument("--tensor-parallel-size", type=int, default=4)
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--max-tokens", type=int, default=512)
    parser.add_argument("--gpu-memory-utilization", type=float, default=0.9)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--max-samples", type=int, default=None)
    parser.add_argument("--trust-remote-code", action="store_true")
    return parser


def main() -> None:
    args = _build_parser().parse_args()
    if args.batch_size <= 0:
        raise ValueError("--batch-size must be positive")
    if args.tensor_parallel_size <= 0:
        raise ValueError("--tensor-parallel-size must be positive")
    judge = _build_judge(args)
    run_g_eval(
        input_path=args.input,
        output_path=args.output,
        summary_output=args.summary_output,
        summary_csv_output=args.summary_csv_output,
        judge=judge,
        cache_dir=args.cache_dir,
        batch_size=args.batch_size,
        max_samples=args.max_samples,
    )


if __name__ == "__main__":
    main()
