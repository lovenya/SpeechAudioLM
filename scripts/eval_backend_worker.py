#!/usr/bin/env python
"""Run one metric backend inside its dedicated environment.

The main evaluator invokes this script with the backend-specific Python
interpreter configured in configs/eval_metrics.json.
"""

from __future__ import annotations

import argparse
import importlib.metadata
import json
import math
import os
import shutil
import sys
from pathlib import Path
from typing import Any


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run a SpeechAudio LM metric backend worker.")
    parser.add_argument("--backend", required=True)
    parser.add_argument("--input", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    with args.input.open("r", encoding="utf-8") as f:
        payload = json.load(f)

    backend = args.backend
    if backend == "aac-metrics":
        result = run_aac_metrics(payload)
    elif backend == "openai-whisper":
        result = run_whisper(payload)
    elif backend == "utmosv2":
        result = run_utmosv2(payload)
    elif backend == "torchmetrics-dnsmos":
        result = run_dnsmos(payload)
    elif backend == "audioldm_eval":
        result = run_audioldm_eval(payload)
    elif backend == "kadtk":
        result = run_kadtk(payload)
    elif backend == "stable-audio-metrics":
        result = run_stable_audio_metrics(payload)
    else:
        raise SystemExit(f"Unsupported backend worker: {backend}")

    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", encoding="utf-8") as f:
        json.dump(result, f, indent=2, sort_keys=True)
        f.write("\n")


def run_aac_metrics(payload: dict[str, Any]) -> dict[str, Any]:
    from evaluation.metrics.aac_audio import (
        MS_CLAP_2023_TEXT_CONTEXT_TOKENS,
        MS_CLAP_GPT2_END_OF_TEXT_SUFFIX,
        MS_CLAP_GPT2_HARD_CONTEXT_TOKENS,
        PROJECT_EVALUATION_SEED,
        reconstruct_mace_samples,
        validate_clap_audio_samples,
    )
    from evaluation.metrics.aac_preprocessing import (
        caption_preprocessing_metadata,
        prepare_spice_caption,
    )

    metrics = set(payload["metrics"])
    rows = payload["rows"]
    cache_path = _first_config_value(payload, "cache_dir") or payload["cache_dir"]
    device = payload["device"]
    sample_metrics: dict[str, dict[str, float]] = {str(row["row_index"]): {} for row in rows}
    corpus_metrics: dict[str, Any] = {}
    policy_metadata: dict[str, dict[str, Any]] = {}
    errors: dict[str, dict[str, str]] = {}

    java_path = _resolve_java_path(payload) if "spider" in metrics else None
    java_tool_options = os.environ.pop("JAVA_TOOL_OPTIONS", None)
    try:
        if "spider" in metrics:
            from aac_metrics.functional.spider import spider as spider_fn

            metric_rows = _valid_aac_rows(rows, "spider", errors)
            if not metric_rows:
                policy_metadata["spider"] = caption_preprocessing_metadata() | _metric_attempt_metadata(rows, metric_rows, errors, "spider")
            else:
                cfg = payload["metric_configs"].get("spider", {})
                raw_candidates = [row["candidate_text"] for row in metric_rows]
                raw_references = [[row["source"]["audio_cap"]] for row in metric_rows]
                spider_candidates = [prepare_spice_caption(candidate) for candidate in raw_candidates]
                spider_references = [[prepare_spice_caption(ref) for ref in refs] for refs in raw_references]
                corpus_scores, sent_scores = spider_fn(
                    spider_candidates,
                    spider_references,
                    cache_path=cache_path,
                    java_path=java_path,
                    java_max_memory=str(cfg.get("java_max_memory", "32G")),
                    return_all_scores=True,
                )
                _assign_sample_scores(metric_rows, sample_metrics, "spider", sent_scores.get("spider"))
                corpus_metrics["spider"] = _scalar(corpus_scores.get("spider"), metric="spider")
                policy_metadata["spider"] = caption_preprocessing_metadata() | {
                    "corpus_score_policy": "preserve official corpus SPIDEr; do not compare with sentence-score mean",
                } | _metric_attempt_metadata(rows, metric_rows, errors, "spider")

        if "mace" in metrics:
            from aac_metrics.functional.mace import mace as mace_fn

            metric_rows = _valid_aac_rows(rows, "mace", errors)
            cfg = payload["metric_configs"].get("mace", {})
            mace_method = cfg.get("mace_method", "combined")
            overflow_metadata = {}
            if mace_method in {"text", "combined"}:
                metric_rows, overflow_metadata = _filter_ms_clap_text_overflow(
                    metric_rows, "mace", errors, include_references=True
                )
            if not metric_rows:
                policy_metadata["mace"] = overflow_metadata | _metric_attempt_metadata(
                    rows, metric_rows, errors, "mace"
                )
            else:
                raw_candidates = [row["candidate_text"] for row in metric_rows]
                raw_references = [[row["source"]["audio_cap"]] for row in metric_rows]
                audio_paths = [row["source"]["audio_file"] for row in metric_rows]
                penalty = float(cfg.get("penalty", 0.3))
                error_threshold = float(cfg.get("error_threshold", 0.97))
                corpus_scores, sent_scores = mace_fn(
                    raw_candidates,
                    raw_references,
                    audio_paths=audio_paths,
                    mace_method=mace_method,
                    penalty=penalty,
                    clap_model=cfg.get("clap_model", "MS-CLAP-2023"),
                    seed=PROJECT_EVALUATION_SEED,
                    echecker=cfg.get("echecker", "echecker_clotho_audiocaps_base"),
                    error_threshold=error_threshold,
                    device=device,
                    batch_size=int(cfg.get("batch_size", 1)),
                    return_all_scores=True,
                )
                official_corpus_score = _scalar(corpus_scores.get("mace"), metric="mace")
                reconstruction = reconstruct_mace_samples(
                    sent_scores,
                    expected_count=len(metric_rows),
                    penalty=penalty,
                    corpus_score=official_corpus_score,
                    mace_method=mace_method,
                    seed=PROJECT_EVALUATION_SEED,
                )
                _assign_sample_scores(metric_rows, sample_metrics, "mace", reconstruction.scores)
                corpus_metrics["mace"] = official_corpus_score
                policy_metadata["mace"] = dict(reconstruction.metadata) | {
                    "error_threshold": error_threshold,
                } | overflow_metadata | _metric_attempt_metadata(
                    rows, metric_rows, errors, "mace"
                )

        if "aac_clapsim" in metrics:
            from aac_metrics.functional.clap_sim import clap_sim

            metric_rows = _valid_aac_rows(rows, "aac_clapsim", errors)
            metric_rows, overflow_metadata = _filter_ms_clap_text_overflow(
                metric_rows, "aac_clapsim", errors, include_references=False
            )
            if not metric_rows:
                policy_metadata["aac_clapsim"] = overflow_metadata | _metric_attempt_metadata(
                    rows, metric_rows, errors, "aac_clapsim"
                )
            else:
                raw_candidates = [row["candidate_text"] for row in metric_rows]
                audio_paths = [row["source"]["audio_file"] for row in metric_rows]
                cfg = payload["metric_configs"].get("aac_clapsim", {})
                corpus_scores, sent_scores = clap_sim(
                    raw_candidates,
                    audio_paths=audio_paths,
                    clap_method="audio",
                    clap_model=cfg.get("model_name", "MS-CLAP-2023"),
                    device=device,
                    batch_size=int(cfg.get("batch_size", 1)),
                    seed=PROJECT_EVALUATION_SEED,
                    return_all_scores=True,
                )
                official_corpus_score = _scalar(corpus_scores.get("clap_sim"), metric="aac_clapsim")
                validated = validate_clap_audio_samples(
                    sent_scores,
                    expected_count=len(metric_rows),
                    corpus_score=official_corpus_score,
                    seed=PROJECT_EVALUATION_SEED,
                )
                _assign_sample_scores(metric_rows, sample_metrics, "aac_clapsim", validated.scores)
                corpus_metrics["aac_clapsim"] = official_corpus_score
                policy_metadata["aac_clapsim"] = (
                    dict(validated.metadata)
                    | overflow_metadata
                    | _metric_attempt_metadata(rows, metric_rows, errors, "aac_clapsim")
                )

        if "fense" in metrics:
            from aac_metrics.functional.fense import fense as fense_fn

            metric_rows = _valid_aac_rows(rows, "fense", errors)
            if not metric_rows:
                policy_metadata["fense"] = caption_preprocessing_metadata() | _metric_attempt_metadata(rows, metric_rows, errors, "fense")
            else:
                raw_candidates = [row["candidate_text"] for row in metric_rows]
                raw_references = [[row["source"]["audio_cap"]] for row in metric_rows]
                cfg = payload["metric_configs"].get("fense", {})
                penalty = float(cfg.get("penalty", 0.9))
                error_threshold = float(cfg.get("error_threshold", 0.9))
                corpus_scores, sent_scores = fense_fn(
                    raw_candidates,
                    raw_references,
                    sbert_model=cfg.get("sbert_model", "paraphrase-TinyBERT-L6-v2"),
                    echecker=cfg.get("echecker", "echecker_clotho_audiocaps_base"),
                    error_threshold=error_threshold,
                    device=device,
                    batch_size=int(cfg.get("batch_size", 1)),
                    penalty=penalty,
                    return_all_scores=True,
                )
                fense_scores = _sample_score_vector(sent_scores.get("fense"), len(metric_rows), "fense")
                official_corpus_score = _scalar(corpus_scores.get("fense"), metric="fense")
                _require_sample_mean(official_corpus_score, fense_scores, "fense")
                _assign_sample_scores(metric_rows, sample_metrics, "fense", fense_scores)
                corpus_metrics["fense"] = official_corpus_score
                policy_metadata["fense"] = caption_preprocessing_metadata() | {
                    "penalty": penalty,
                    "error_threshold": error_threshold,
                    "corpus_aggregation": "arithmetic mean over the passed official sample vector",
                } | _metric_attempt_metadata(rows, metric_rows, errors, "fense")
    finally:
        if java_tool_options is not None:
            os.environ["JAVA_TOOL_OPTIONS"] = java_tool_options

    result = _result(
        payload,
        sample_metrics=sample_metrics,
        corpus_metrics=corpus_metrics,
        predictions={},
        errors=errors,
        packages=["aac-metrics", "torch", "numpy"],
    )
    for metric, metadata in policy_metadata.items():
        result["metric_backends"][metric]["policy_metadata"] = metadata
    return result



def _valid_aac_rows(
    rows: list[dict[str, Any]],
    metric: str,
    errors: dict[str, dict[str, str]],
) -> list[dict[str, Any]]:
    requires_reference = metric in {"spider", "mace", "fense"}
    requires_audio = metric in {"mace", "aac_clapsim"}
    valid: list[dict[str, Any]] = []
    for row in rows:
        row_key = str(row["row_index"])
        try:
            candidate = row.get("candidate_text")
            if not isinstance(candidate, str) or not candidate.strip():
                raise ValueError("candidate caption is empty")
            if requires_reference:
                reference = row.get("source", {}).get("audio_cap")
                if not isinstance(reference, str) or not reference.strip():
                    raise ValueError("reference caption is empty")
            if requires_audio:
                audio_path = Path(row.get("source", {}).get("audio_file", ""))
                if audio_path.is_file():
                    import soundfile as sf
                    sf.info(str(audio_path))
            valid.append(row)
        except Exception as exc:
            errors.setdefault(row_key, {})[metric] = _metric_error(exc)
    return valid


def _metric_attempt_metadata(
    attempted_rows: list[dict[str, Any]],
    passed_rows: list[dict[str, Any]],
    errors: dict[str, dict[str, str]],
    metric: str,
) -> dict[str, Any]:
    failed_rows = [
        row for row in attempted_rows
        if metric in errors.get(str(row["row_index"]), {})
    ]
    return {
        "status": "passed" if not failed_rows else ("passed_with_errors" if passed_rows else "error"),
        "attempted_count": len(attempted_rows),
        "passed_count": len(passed_rows),
        "error_count": len(failed_rows),
        "error_row_indices": [str(row["row_index"]) for row in failed_rows],
        "error_sample_ids": [str(row.get("sample_id", row["row_index"])) for row in failed_rows],
    }


def _filter_ms_clap_text_overflow(rows, metric, errors, *, include_references):
    from evaluation.metrics.aac_audio import (
        MS_CLAP_2023_TEXT_CONTEXT_TOKENS,
        MS_CLAP_GPT2_END_OF_TEXT_SUFFIX,
        MS_CLAP_GPT2_HARD_CONTEXT_TOKENS,
    )

    valid = []
    rejected = []
    tokenizer = _ms_clap_gpt2_tokenizer()
    for row in rows:
        fields = [("candidate", row["candidate_text"])]
        if include_references:
            fields.append(("reference[0]", row["source"]["audio_cap"]))
        overflow = None
        for field, text in fields:
            encoded = tokenizer.encode_plus(
                text=f"{text}{MS_CLAP_GPT2_END_OF_TEXT_SUFFIX}",
                add_special_tokens=True,
                max_length=MS_CLAP_2023_TEXT_CONTEXT_TOKENS,
                padding="max_length",
                return_tensors="pt",
            )
            token_count = int(encoded["input_ids"].shape[-1])
            if token_count > MS_CLAP_GPT2_HARD_CONTEXT_TOKENS:
                overflow = field, token_count
                break
        if overflow is None:
            valid.append(row)
            continue
        field, token_count = overflow
        row_key = str(row["row_index"])
        errors.setdefault(row_key, {})[metric] = (
            f"MS-CLAP GPT-2 tokenized {field} length {token_count} exceeds its hard "
            f"{MS_CLAP_GPT2_HARD_CONTEXT_TOKENS}-position limit; text was not truncated or modified"
        )
        rejected.append({
            "row_index": row["row_index"],
            "sample_id": str(row.get("sample_id", row["row_index"])),
            "field": field,
            "token_count": token_count,
        })
    return valid, {
        "text_preflight": "exact released MS-CLAP GPT-2 tokenization before CUDA scoring",
        "nominal_text_context_tokens": MS_CLAP_2023_TEXT_CONTEXT_TOKENS,
        "gpt2_hard_context_tokens": MS_CLAP_GPT2_HARD_CONTEXT_TOKENS,
        "project_text_truncation": False,
        "overflow_action": "record attributable error and score ordered survivors once",
        "rejected_text_inputs": rejected,
        "survivor_order": "original relative input order",
        "official_metric_invocations": 0 if not valid else 1,
        "seeded_audio_crop_stream_note": "filtering changes the seeded crop stream; independently scored subsets are not merged",
    }


def _ms_clap_gpt2_tokenizer():
    from transformers import AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained("gpt2", local_files_only=True)
    tokenizer.add_special_tokens({"pad_token": "!"})
    return tokenizer


def run_whisper(payload: dict[str, Any]) -> dict[str, Any]:
    from scripts.wer_backend import (
        NORMALIZED_SCORE,
        aggregate_wer_scores,
        score_wer_pair,
        transcribe_whisper,
    )

    metrics = set(payload["metrics"])
    rows = payload["rows"]
    if not rows:
        raise ValueError("Cannot score WER for an empty corpus")

    import whisper

    cfg = payload["metric_configs"].get("asr_wer") or payload["metric_configs"].get("dwer") or {}
    model_name = cfg.get("model_name", "large-v3")
    cache_dir = cfg.get("cache_dir") or payload["cache_dir"]
    device = payload["device"]
    model = whisper.load_model(model_name, download_root=cache_dir, device=device)

    sample_metrics: dict[str, dict[str, float]] = {}
    predictions: dict[str, dict[str, Any]] = {}
    errors: dict[str, dict[str, str]] = {}
    scores_by_metric: dict[str, list[dict[str, dict[str, Any]]]] = {metric: [] for metric in metrics}
    failed_ids: dict[str, list[str]] = {metric: [] for metric in metrics}
    for row in rows:
        row_key = str(row["row_index"])
        sample_metrics[row_key] = {}
        predictions[row_key] = {}
        try:
            generated_text = transcribe_whisper(model, row["output_file"], device)
            predictions[row_key]["generated_speech_asr"] = generated_text
        except Exception as exc:
            for metric in metrics:
                errors.setdefault(row_key, {})[metric] = _metric_error(exc)
                failed_ids[metric].append(row_key)
            continue

        if "asr_wer" in metrics:
            try:
                score = score_wer_pair(row["source"]["speech_text"], generated_text)
                scores_by_metric["asr_wer"].append(score)
                sample_metrics[row_key]["asr_wer"] = float(score[NORMALIZED_SCORE]["wer"])
                predictions[row_key]["asr_wer_details"] = score
            except Exception as exc:
                errors.setdefault(row_key, {})["asr_wer"] = _metric_error(exc)
                failed_ids["asr_wer"].append(row_key)
        if "dwer" in metrics:
            try:
                reference_text = transcribe_whisper(model, row["source"]["speech_file"], device)
                score = score_wer_pair(reference_text, generated_text)
                scores_by_metric["dwer"].append(score)
                predictions[row_key]["reference_speech_asr"] = reference_text
                predictions[row_key]["dwer_details"] = score
                sample_metrics[row_key]["dwer"] = float(score[NORMALIZED_SCORE]["wer"])
            except Exception as exc:
                errors.setdefault(row_key, {})["dwer"] = _metric_error(exc)
                failed_ids["dwer"].append(row_key)

    semantics = {
        "asr_wer": "WER(reference_transcript, ASR(generated_speech))",
        "dwer": "WER(ASR(reference_speech), ASR(generated_speech))",
    }
    corpus_metrics: dict[str, Any] = {}
    for metric, scores in scores_by_metric.items():
        if not scores:
            continue
        aggregates = aggregate_wer_scores(scores)
        corpus_metrics[metric] = {
            "level": "corpus",
            "status": "passed_with_errors" if failed_ids[metric] else "passed",
            "attempted_count": len(rows),
            "passed_count": len(scores),
            "error_count": len(failed_ids[metric]),
            "error_sample_ids": failed_ids[metric],
            "count": 1,
            "input_count": len(scores),
            "mean": aggregates[NORMALIZED_SCORE]["wer"],
            "mean_semantics": "normalized corpus WER from summed edit counts; not sample mean",
            "semantics": semantics[metric],
            **aggregates,
        }
    result = _result(
        payload, sample_metrics=sample_metrics, corpus_metrics=corpus_metrics,
        predictions=predictions, errors=errors, packages=["openai-whisper", "torch", "numpy"],
    )
    for metric in metrics:
        result["metric_backends"][metric]["policy_metadata"] = {
            "semantics": semantics[metric],
            "sample_metric": "normalized utterance WER",
            "corpus_aggregation": "WER from summed edit counts over passed pairs; macro_wer is secondary",
            "attempted_count": len(rows),
            "passed_count": len(scores_by_metric[metric]),
            "error_count": len(failed_ids[metric]),
            "error_sample_ids": failed_ids[metric],
            "corpus": corpus_metrics.get(metric),
        }
    return result


def run_utmosv2(payload: dict[str, Any]) -> dict[str, Any]:
    import utmosv2
    from evaluation.metrics.utmos_policy import (
        UTMOSV2_CHECKPOINT_FOLD,
        UTMOSV2_CHECKPOINT_SEED,
        UTMOSV2_CONFIG,
        official_utmosv2_policy,
        scientific_prediction_kwargs,
        validate_prediction_policy,
        validate_utmos_records,
    )

    cfg = payload["metric_configs"].get("utmos", {})
    device = payload["device"]
    config = cfg.get("config_name", UTMOSV2_CONFIG)
    fold = cfg.get("fold", UTMOSV2_CHECKPOINT_FOLD)
    checkpoint_seed = cfg.get("checkpoint_seed", cfg.get("seed", UTMOSV2_CHECKPOINT_SEED))
    pretrained = cfg.get("pretrained", True)
    checkpoint_path = cfg.get("model_path")
    prediction_kwargs = scientific_prediction_kwargs()
    predict_dataset = cfg.get("predict_dataset", prediction_kwargs["predict_dataset"])
    num_repetitions = cfg.get("num_repetitions", prediction_kwargs["num_repetitions"])
    remove_silent_section = cfg.get(
        "remove_silent_section", prediction_kwargs["remove_silent_section"]
    )
    validate_prediction_policy(
        config=config,
        fold=fold,
        checkpoint_seed=checkpoint_seed,
        pretrained=pretrained,
        checkpoint_path=checkpoint_path,
        predict_dataset=predict_dataset,
        num_repetitions=num_repetitions,
        remove_silent_section=remove_silent_section,
    )
    model = utmosv2.create_model(
        pretrained=pretrained,
        config=config,
        fold=fold,
        checkpoint_path=checkpoint_path,
        seed=checkpoint_seed,
        device=device,
    )
    records: list[dict[str, Any]] = []
    errors: dict[str, dict[str, str]] = {}
    for row in payload["rows"]:
        row_key = str(row["row_index"])
        try:
            score = model.predict(
                input_path=row["output_file"], device=device,
                num_workers=int(cfg.get("num_workers", 0)),
                batch_size=int(cfg.get("batch_size", 1)),
                predict_dataset=predict_dataset, num_repetitions=num_repetitions,
                remove_silent_section=remove_silent_section, verbose=False,
            )
            records.append({"sample_id": row_key, "utmos": _scalar(score, metric="utmos")})
        except Exception as exc:
            errors[row_key] = {"utmos": _metric_error(exc)}

    sample_metrics: dict[str, dict[str, float]] = {}
    corpus_metrics: dict[str, Any] = {}
    policy_metadata = official_utmosv2_policy() | {
        "attempted_count": len(payload["rows"]),
        "passed_count": 0,
        "error_count": len(errors),
        "error_sample_ids": list(errors),
    }
    if records:
        try:
            aggregation = validate_utmos_records(
                records, expected_ids=[record["sample_id"] for record in records]
            )
        except Exception as exc:
            for record in records:
                row_key = record["sample_id"]
                errors[row_key] = {"utmos": _metric_error(exc)}
        else:
            sample_metrics = {record.sample_id: {"utmos": record.score} for record in aggregation.records}
            failed_ids = list(errors)
            corpus_metrics["utmos"] = {
                "level": "corpus",
                "status": "passed_with_errors" if failed_ids else "passed",
                "attempted_count": len(payload["rows"]),
                "passed_count": len(aggregation.records),
                "error_count": len(failed_ids),
                "error_sample_ids": failed_ids,
                "input_count": len(aggregation.records),
                "mean": aggregation.corpus_mean,
            }
            policy_metadata = dict(aggregation.metadata) | {
                "attempted_count": len(payload["rows"]),
                "passed_count": len(aggregation.records),
                "error_count": len(failed_ids),
                "error_sample_ids": failed_ids,
            }
    result = _result(
        payload, sample_metrics=sample_metrics, corpus_metrics=corpus_metrics,
        predictions={}, errors=errors, packages=["utmosv2", "torch", "numpy"],
    )
    result["metric_backends"]["utmos"]["policy_metadata"] = policy_metadata
    return result


def run_dnsmos(payload: dict[str, Any]) -> dict[str, Any]:
    import torch
    import torchaudio
    import torchmetrics.functional.audio.dnsmos as dnsmos_module

    from torchmetrics.functional.audio.dnsmos import deep_noise_suppression_mean_opinion_score

    cfg = payload["metric_configs"].get("dnsmos", {})
    cache_dir = Path(cfg.get("cache_dir") or payload["cache_dir"])
    dnsmos_module.DNSMOS_DIR = str(cache_dir)
    personalized = bool(cfg.get("personalized", False))
    num_threads = cfg.get("num_threads")
    num_threads = int(num_threads) if num_threads is not None else None
    device = cfg.get("runtime_device", "cpu")
    if device == "cpu" and str(payload.get("device", "")).startswith("cuda") and cfg.get("use_gpu", False):
        device = payload["device"]

    sample_metrics: dict[str, dict[str, float]] = {}
    errors: dict[str, dict[str, str]] = {}
    for row in payload["rows"]:
        row_key = str(row["row_index"])
        try:
            waveform, fs = torchaudio.load(row["output_file"])
            waveform = waveform.mean(dim=0)
            scores = deep_noise_suppression_mean_opinion_score(
                waveform,
                int(fs),
                personalized,
                device=device,
                num_threads=num_threads,
            ).detach().cpu().tolist()
            p808_mos, mos_sig, mos_bak, mos_ovr = [float(value) for value in scores]
            sample_metrics[row_key] = {
                "dnsmos": mos_ovr,
                "dnsmos_p808_mos": p808_mos,
                "dnsmos_mos_sig": mos_sig,
                "dnsmos_mos_bak": mos_bak,
                "dnsmos_mos_ovr": mos_ovr,
            }
        except Exception as exc:
            errors[row_key] = {"dnsmos": _metric_error(exc)}
    return _result(
        payload,
        sample_metrics=sample_metrics,
        corpus_metrics={},
        predictions={},
        errors=errors,
        packages=["torchmetrics", "torch", "torchaudio", "onnxruntime", "librosa", "requests"],
    )

def run_audioldm_eval(payload: dict[str, Any]) -> dict[str, Any]:
    _disable_numba_function_cache()
    from audioldm_eval import EvaluationHelper
    import torch

    original_torch_load = torch.load

    def load_legacy_checkpoint(*args: Any, **kwargs: Any) -> Any:
        kwargs.setdefault("weights_only", False)
        return original_torch_load(*args, **kwargs)

    metric = payload["metrics"][0]
    cfg = payload["metric_configs"].get(metric, {})
    runtime_home = Path(cfg.get("runtime_home") or Path(cfg.get("cache_dir") or payload["cache_dir"]).parent / "runtime_home")
    runtime_home.mkdir(parents=True, exist_ok=True)
    generated_dir, reference_dir, valid_rows, errors = _stage_valid_audio_pairs(
        payload, reference_field="audio_file", metric=metric
    )
    minimum_size = int(cfg.get("minimum_valid_corpus_size", 2))
    if len(valid_rows) < minimum_size:
        _mark_metric_constraint_errors(
            payload, errors, metric,
            f"valid paired corpus has {len(valid_rows)} samples; backend requires at least {minimum_size}",
        )
        return _result(
            payload, sample_metrics={}, corpus_metrics={}, predictions={}, errors=errors,
            packages=["audioldm_eval", "torch", "numpy"],
        )

    previous_home = os.environ.get("HOME")
    try:
        torch.load = load_legacy_checkpoint
        os.environ["HOME"] = str(runtime_home)
        evaluator = EvaluationHelper(
            int(cfg.get("sample_rate", 16000)), torch.device(payload["device"]),
            backbone=cfg.get("backbone", "cnn14"),
        )
        raw = evaluator.main(str(generated_dir), str(reference_dir), limit_num=len(valid_rows))
    finally:
        torch.load = original_torch_load
        if previous_home is None:
            os.environ.pop("HOME", None)
        else:
            os.environ["HOME"] = previous_home

    failed_ids = list(errors)
    common = {
        "level": "corpus",
        "status": "passed_with_errors" if failed_ids else "passed",
        "attempted_count": len(payload["rows"]),
        "passed_count": len(valid_rows),
        "error_count": len(failed_ids),
        "error_sample_ids": failed_ids,
        "input_count": len(valid_rows),
    }
    try:
        corpus_metrics = _audioldm_corpus_metrics(metric, raw, common)
    except ValueError as exc:
        _mark_metric_constraint_errors(payload, errors, metric, str(exc))
        corpus_metrics = {}
    return _result(
        payload, sample_metrics={}, corpus_metrics=corpus_metrics, predictions={}, errors=errors,
        packages=["audioldm_eval", "torch", "numpy"],
    )


def _audioldm_corpus_metrics(
    metric: str,
    raw: dict[str, Any],
    common: dict[str, Any],
) -> dict[str, dict[str, Any]]:
    if metric == "fad":
        return {
            "fad": common | {
                "mean": _scalar(
                    raw.get("frechet_audio_distance"),
                    metric="fad",
                    rejected_values=(-1.0,),
                )
            }
        }
    if metric == "inception_score":
        return {
            "inception_score": common | {
                "mean": _scalar(
                    raw.get("inception_score_mean"), metric="inception_score"
                ),
                "std": _scalar(
                    raw.get("inception_score_std"), metric="inception_score"
                ),
            }
        }
    raise ValueError(f"Unsupported AudioLDM Eval metric: {metric}")


def run_kadtk(payload: dict[str, Any]) -> dict[str, Any]:
    metric = "kad"
    cfg = payload["metric_configs"].get(metric, {})
    minimum_duration_seconds = float(cfg.get("minimum_duration_seconds", 1.0))
    generated_dir, reference_dir, valid_rows, errors = _stage_valid_audio_pairs(
        payload, reference_field="audio_file", metric=metric,
        minimum_duration_seconds=minimum_duration_seconds,
    )
    minimum_size = int(cfg.get("minimum_valid_corpus_size", 2))
    if len(valid_rows) < minimum_size:
        _mark_metric_constraint_errors(
            payload, errors, metric,
            f"valid paired corpus has {len(valid_rows)} samples; backend requires at least {minimum_size}",
        )
        return _result(
            payload, sample_metrics={}, corpus_metrics={}, predictions={}, errors=errors,
            packages=["kadtk", "torch", "numpy", "tensorflow"],
        )

    import kadtk.model_loader as model_loader
    import torch
    from kadtk.emb_loader import EmbeddingLoader
    from kadtk.kad import KernelAudioDistance

    cache_dir = Path(cfg.get("cache_dir") or payload["cache_dir"])
    cache_dir.mkdir(parents=True, exist_ok=True)
    model_loader.__file__ = str(cache_dir / "model_loader.py")
    model_name = cfg.get("model_name", "panns-wavegram-logmel")
    if model_name == "panns-wavegram-logmel":
        model = model_loader.PANNsModel("wavegram-logmel")
    elif model_name == "panns-cnn14-16k":
        model = model_loader.PANNsModel("cnn14-16k")
    elif model_name == "panns-cnn14-32k":
        model = model_loader.PANNsModel("cnn14-32k")
    else:
        raise ValueError(f"Unsupported kadtk model for direct worker: {model_name}")

    if not str(payload["device"]).startswith("cuda"):
        original_torch_load = torch.load

        def load_on_cpu(*args: Any, **kwargs: Any) -> Any:
            kwargs.setdefault("map_location", torch.device("cpu"))
            return original_torch_load(*args, **kwargs)

        torch.load = load_on_cpu

    workers = int(cfg.get("workers", 1))
    emb_loader = EmbeddingLoader(model, audio_load_worker=1, load_model=True)
    for audio_dir in (reference_dir, generated_dir):
        if cfg.get("force_emb_encode", False):
            emb_path = audio_dir / "embeddings" / model.name
            if emb_path.exists():
                shutil.rmtree(emb_path)
        for audio_file in sorted(audio_dir.glob("*.*")):
            emb_loader.cache_embedding_file(audio_file)
    kad = KernelAudioDistance(
        model, payload["device"], audio_load_worker=workers,
        force_stats_calc=bool(cfg.get("force_stats_calc", False)),
    )
    value = _scalar(kad.score(reference_dir, generated_dir), metric=metric)
    failed_ids = list(errors)
    corpus = {
        "level": "corpus",
        "status": "passed_with_errors" if failed_ids else "passed",
        "attempted_count": len(payload["rows"]),
        "passed_count": len(valid_rows),
        "error_count": len(failed_ids),
        "error_sample_ids": failed_ids,
        "input_count": len(valid_rows),
        "mean": value,
    }
    return _result(
        payload, sample_metrics={}, corpus_metrics={metric: corpus}, predictions={}, errors=errors,
        packages=["kadtk", "torch", "numpy", "tensorflow"],
    )


def run_stable_audio_metrics(payload: dict[str, Any]) -> dict[str, Any]:
    metrics = set(payload["metrics"])
    sample_metrics: dict[str, dict[str, float]] = {}
    corpus_metrics: dict[str, Any] = {}
    errors: dict[str, dict[str, str]] = {}

    policy_metadata: dict[str, dict[str, Any]] = {}
    clap_metrics = (
        ("target_caption_clap_score", "target_caption_alignment"),
        ("prompt_adherence_clap_score", "prompt_adherence"),
    )
    for metric_name, semantics in clap_metrics:
        if metric_name not in metrics:
            continue
        clap_scores, clap_errors, clap_corpus, clap_policy = _stable_clap_scores(
            payload, metric_name=metric_name, semantics=semantics
        )
        for row_key, row_metrics in clap_scores.items():
            sample_metrics.setdefault(row_key, {}).update(row_metrics)
        for row_key, row_errors in clap_errors.items():
            errors.setdefault(row_key, {}).update(row_errors)
        if clap_corpus is not None:
            corpus_metrics[metric_name] = clap_corpus
        policy_metadata[metric_name] = clap_policy
    if "kl_passt" in metrics:
        kl_corpus, kl_errors, kl_policy = _stable_kl_passt(payload)
        errors.update(kl_errors)
        if kl_corpus is not None:
            corpus_metrics["kl_passt"] = kl_corpus
        policy_metadata["kl_passt"] = kl_policy

    result = _result(
        payload,
        sample_metrics=sample_metrics,
        corpus_metrics=corpus_metrics,
        predictions={},
        errors=errors,
        packages=["laion-clap", "hear21passt", "torch", "numpy", "librosa", "pyloudnorm"],
    )
    for metric, metadata in policy_metadata.items():
        result["metric_backends"][metric]["policy_metadata"] = metadata
    return result


def _stable_clap_scores(
    payload: dict[str, Any],
    *,
    metric_name: str,
    semantics: str,
) -> tuple[
    dict[str, dict[str, float]],
    dict[str, dict[str, str]],
    dict[str, Any] | None,
    dict[str, Any],
]:
    import numpy as np
    import requests
    import torch
    import laion_clap
    import librosa
    import pyloudnorm as pyln
    from clap_module.factory import load_state_dict
    from tqdm import tqdm

    cfg = payload["metric_configs"].get(metric_name, {})
    generated_dir = _stage_generated_audio_dir(payload)
    cache_dir = Path(cfg.get("cache_dir") or payload["cache_dir"])
    checkpoint = cfg.get("checkpoint", "630k-audioset-fusion-best")
    checkpoint_file = checkpoint if checkpoint.endswith(".pt") else f"{checkpoint}.pt"
    checkpoint_path = cache_dir / "load" / "clap_score" / checkpoint_file
    _download_clap_checkpoint(checkpoint_file, checkpoint_path, requests, tqdm)

    enable_fusion = checkpoint_file == "630k-audioset-fusion-best.pt"
    model_kwargs = {"enable_fusion": enable_fusion, "device": payload["device"]}
    if not enable_fusion:
        model_kwargs["amodel"] = "HTSAT-base"
    model = laion_clap.CLAP_Module(**model_kwargs)
    original_torch_load = torch.load

    def load_legacy_checkpoint(*args: Any, **kwargs: Any) -> Any:
        kwargs.setdefault("weights_only", False)
        return original_torch_load(*args, **kwargs)

    try:
        torch.load = load_legacy_checkpoint
        state_dict = load_state_dict(str(checkpoint_path))
    finally:
        torch.load = original_torch_load
    state_dict.pop("text_branch.embeddings.position_ids", None)
    model.model.load_state_dict(state_dict)
    model.eval()

    from evaluation.metrics.clap_score import (
        PROMPT_ADHERENCE,
        TARGET_CAPTION_ALIGNMENT,
        validate_clap_score_records,
    )

    ids = [_stage_id(index, row) for index, row in enumerate(payload["rows"])]
    if semantics == TARGET_CAPTION_ALIGNMENT:
        texts = [row["source"]["audio_cap"] for row in payload["rows"]]
    elif semantics == PROMPT_ADHERENCE:
        texts = [row["conditioning_prompt"] for row in payload["rows"]]
    else:
        raise ValueError(f"Unsupported CLAPScore semantics: {semantics}")

    def batched_tokenizer(text: list[str]) -> dict[str, Any]:
        return model.tokenize(
            text,
            padding="max_length",
            truncation=True,
            max_length=77,
            return_tensors="pt",
        )

    text_embeddings, text_embedding_errors = _stable_clap_text_embeddings(
        model, texts, batched_tokenizer, torch
    )

    attempted_records: list[dict[str, Any]] = []
    errors: dict[str, dict[str, str]] = {}
    for row, stem, text, text_embedding, text_embedding_error in zip(
        payload["rows"], ids, texts, text_embeddings, text_embedding_errors
    ):
        row_key = str(row["row_index"])
        try:
            if text_embedding_error is not None or text_embedding is None:
                raise RuntimeError(text_embedding_error or "CLAP text embedding was not produced")
            audio_path = generated_dir / f"{stem}.wav"
            audio, _ = librosa.load(str(audio_path), sr=int(cfg.get("sample_rate", 48000)), mono=True)
            audio = pyln.normalize.peak(audio, -1.0)
            audio = audio.reshape(1, -1)
            audio = torch.from_numpy(_int16_to_float32(_float32_to_int16(audio, np), np)).float()
            with torch.no_grad():
                audio_embedding = model.get_audio_embedding_from_data(x=audio, use_tensor=True)
                score = torch.nn.functional.cosine_similarity(
                    audio_embedding,
                    text_embedding.unsqueeze(0),
                    dim=1,
                    eps=1e-8,
                )[0]
            value = float(score.detach().cpu().item())
            attempted_records.append({
                "sample_id": row_key, "generated_id": row_key,
                "semantics": semantics, "text": text, "score": value,
            })
        except Exception as exc:
            message = _metric_error(exc)
            errors[row_key] = {metric_name: message}
            attempted_records.append({
                "sample_id": row_key,
                "generated_id": row_key,
                "status": "error",
                "error": message,
            })
    expected_ids = [str(row["row_index"]) for row in payload["rows"]]
    aggregation = validate_clap_score_records(
        attempted_records, expected_ids=expected_ids, semantics=semantics
    )
    sample_metrics = {
        record.sample_id: {metric_name: record.score}
        for record in aggregation.records
    }
    corpus = None if aggregation.corpus_mean is None else {
        "level": "corpus", "status": aggregation.metadata["status"],
        "attempted_count": aggregation.metadata["attempted_count"],
        "passed_count": aggregation.metadata["passed_count"],
        "error_count": aggregation.metadata["error_count"],
        "error_sample_ids": list(aggregation.metadata["error_sample_ids"]),
        "input_count": aggregation.sample_count, "mean": aggregation.corpus_mean,
    }
    return sample_metrics, errors, corpus, dict(aggregation.metadata)


def _stable_clap_text_embeddings(
    model: Any,
    texts: list[str],
    tokenizer: Any,
    torch_module: Any,
) -> tuple[list[Any | None], list[str | None]]:
    """Embed text in a batch, then isolate individual failures if needed."""

    embeddings: list[Any | None] = [None] * len(texts)
    errors: list[str | None] = [None] * len(texts)
    try:
        with torch_module.no_grad():
            batch = model.get_text_embedding(
                texts, tokenizer=tokenizer, use_tensor=True
            )
        batch_embeddings = list(batch)
        if len(batch_embeddings) != len(texts):
            raise ValueError(
                f"CLAP returned {len(batch_embeddings)} text embeddings for {len(texts)} texts"
            )
        return batch_embeddings, errors
    except Exception:
        cuda = getattr(torch_module, "cuda", None)
        if cuda is not None and cuda.is_available():
            cuda.empty_cache()

    for index, text in enumerate(texts):
        try:
            with torch_module.no_grad():
                embedding = model.get_text_embedding(
                    [text], tokenizer=tokenizer, use_tensor=True
                )
            embeddings[index] = embedding[0]
        except Exception as exc:
            errors[index] = _metric_error(exc)
    return embeddings, errors

def _stable_kl_passt(
    payload: dict[str, Any],
) -> tuple[dict[str, Any] | None, dict[str, dict[str, str]], dict[str, Any]]:
    import contextlib
    from hear21passt.base import get_basic_model
    from passt_kld import return_probabilities
    from evaluation.metrics.kl_passt import pair_kl_passt, validate_kl_passt_records

    cfg = payload["metric_configs"].get("kl_passt", {})
    collect = cfg.get("collect", "mean")
    generated_dir, reference_dir = _stage_audio_dirs(payload, reference_field="audio_file")
    ids = [_stage_id(index, row) for index, row in enumerate(payload["rows"])]
    with open(os.devnull, "w") as sink, contextlib.redirect_stdout(sink):
        model = get_basic_model(mode="logits")
        model.eval()
        model = model.cuda()

    attempted_records: list[dict[str, Any]] = []
    errors: dict[str, dict[str, str]] = {}
    for row, stem in zip(payload["rows"], ids):
        row_key = str(row["row_index"])
        try:
            reference = return_probabilities(model, reference_dir / f"{stem}.wav", collect=collect)
            generated = return_probabilities(model, generated_dir / f"{stem}.wav", collect=collect)
            value = pair_kl_passt(reference, generated)
            attempted_records.append({
                "sample_id": row_key, "reference_id": row_key,
                "generated_id": row_key, "kl_eval_to_reference": value,
            })
        except Exception as exc:
            message = _metric_error(exc)
            errors[row_key] = {"kl_passt": message}
            attempted_records.append({"sample_id": row_key, "status": "error", "error": message})

    expected_ids = [str(row["row_index"]) for row in payload["rows"]]
    aggregation = validate_kl_passt_records(
        attempted_records, expected_ids=expected_ids, collect=collect
    )
    corpus = None if aggregation.corpus_mean is None else {
        "level": "corpus", "status": aggregation.metadata["status"],
        "attempted_count": aggregation.metadata["attempted_count"],
        "passed_count": aggregation.metadata["passed_count"],
        "error_count": aggregation.metadata["error_count"],
        "error_sample_ids": list(aggregation.metadata["error_sample_ids"]),
        "input_count": aggregation.pair_count, "mean": aggregation.corpus_mean,
    }
    return corpus, errors, dict(aggregation.metadata)


def _download_clap_checkpoint(checkpoint_file: str, checkpoint_path: Path, requests: Any, tqdm: Any) -> None:
    urls = {
        "music_speech_audioset_epoch_15_esc_89.98.pt": "https://huggingface.co/lukewys/laion_clap/resolve/main/music_speech_audioset_epoch_15_esc_89.98.pt",
        "music_audioset_epoch_15_esc_90.14.pt": "https://huggingface.co/lukewys/laion_clap/resolve/main/music_audioset_epoch_15_esc_90.14.pt",
        "music_speech_epoch_15_esc_89.25.pt": "https://huggingface.co/lukewys/laion_clap/resolve/main/music_speech_epoch_15_esc_89.25.pt",
        "630k-audioset-fusion-best.pt": "https://huggingface.co/lukewys/laion_clap/resolve/main/630k-audioset-fusion-best.pt",
    }
    if checkpoint_path.exists():
        return
    url = urls.get(checkpoint_file)
    if not url:
        raise ValueError(f"Unsupported LAION-CLAP checkpoint: {checkpoint_file}")
    checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
    response = requests.get(url, stream=True, timeout=30)
    response.raise_for_status()
    total_size = int(response.headers.get("content-length", 0))
    with checkpoint_path.open("wb") as f:
        with tqdm(total=total_size, unit="B", unit_scale=True) as progress_bar:
            for data in response.iter_content(chunk_size=8192):
                f.write(data)
                progress_bar.update(len(data))


def _float32_to_int16(x: Any, np: Any) -> Any:
    x = np.clip(x, a_min=-1.0, a_max=1.0)
    return (x * 32767.0).astype(np.int16)


def _int16_to_float32(x: Any, np: Any) -> Any:
    return (x / 32767.0).astype(np.float32)


def _disable_numba_function_cache() -> None:
    try:
        import numba
        import numba.np.ufunc.decorators as ufunc_decorators
    except Exception:
        return

    def patch_decorator(decorator: Any) -> Any:
        def wrapper(*args: Any, **kwargs: Any) -> Any:
            kwargs["cache"] = False
            return decorator(*args, **kwargs)

        return wrapper

    numba.jit = patch_decorator(numba.jit)
    numba.vectorize = patch_decorator(numba.vectorize)
    numba.guvectorize = patch_decorator(numba.guvectorize)
    ufunc_decorators.vectorize = patch_decorator(ufunc_decorators.vectorize)
    ufunc_decorators.guvectorize = patch_decorator(ufunc_decorators.guvectorize)


def _stage_valid_audio_pairs(
    payload: dict[str, Any],
    *,
    reference_field: str,
    metric: str,
    minimum_duration_seconds: float = 0.0,
) -> tuple[Path, Path, list[dict[str, Any]], dict[str, dict[str, str]]]:
    import soundfile as sf

    work_dir = Path(payload["work_dir"]) / payload["backend"]
    generated_dir = work_dir / "generated"
    reference_dir = work_dir / "reference"
    _reset_staging_dir(generated_dir)
    _reset_staging_dir(reference_dir)
    valid_rows: list[dict[str, Any]] = []
    errors: dict[str, dict[str, str]] = {}
    for index, row in enumerate(payload["rows"]):
        row_key = str(row["row_index"])
        stem = _stage_id(index, row)
        try:
            generated = Path(row["output_file"]).resolve(strict=True)
            reference = Path(row["source"][reference_field]).resolve(strict=True)
            for label, path in (("generated", generated), ("reference", reference)):
                info = sf.info(str(path))
                duration = info.frames / info.samplerate if info.samplerate else 0.0
                if minimum_duration_seconds > 0 and duration < minimum_duration_seconds:
                    raise ValueError(
                        f"{label} audio duration {duration:.6g}s is below the declared "
                        f"{minimum_duration_seconds:g}s backend constraint"
                    )
            _link_or_copy(generated, generated_dir / f"{stem}.wav")
            _link_or_copy(reference, reference_dir / f"{stem}.wav")
            valid_rows.append(row)
        except Exception as exc:
            errors[row_key] = {metric: _metric_error(exc)}
    return generated_dir, reference_dir, valid_rows, errors


def _reset_staging_dir(path: Path) -> None:
    if path.exists():
        shutil.rmtree(path)
    path.mkdir(parents=True, exist_ok=True)


def _mark_metric_constraint_errors(
    payload: dict[str, Any],
    errors: dict[str, dict[str, str]],
    metric: str,
    message: str,
) -> None:
    for row in payload["rows"]:
        errors.setdefault(str(row["row_index"]), {}).setdefault(metric, message)


def _stage_audio_dirs(payload: dict[str, Any], reference_field: str) -> tuple[Path, Path]:
    work_dir = Path(payload["work_dir"])
    generated_dir = _stage_generated_audio_dir(payload)
    reference_dir = work_dir / payload["backend"] / "reference"
    reference_dir.mkdir(parents=True, exist_ok=True)
    for index, row in enumerate(payload["rows"]):
        stem = _stage_id(index, row)
        _link_or_copy(Path(row["source"][reference_field]), reference_dir / f"{stem}.wav")
    return generated_dir, reference_dir


def _stage_generated_audio_dir(payload: dict[str, Any]) -> Path:
    work_dir = Path(payload["work_dir"])
    generated_dir = work_dir / payload["backend"] / "generated"
    generated_dir.mkdir(parents=True, exist_ok=True)
    for index, row in enumerate(payload["rows"]):
        stem = _stage_id(index, row)
        _link_or_copy(Path(row["output_file"]), generated_dir / f"{stem}.wav")
    return generated_dir


def _stage_id(index: int, row: dict[str, Any]) -> str:
    return f"{index:06d}_{row['sample_id']}"


def _audio_files(directory: Path) -> list[Path]:
    return sorted(path for path in directory.iterdir() if path.is_file() or path.is_symlink())


def _require_minimum_audio_files(directory: Path, minimum: int, corpus_name: str) -> None:
    if minimum < 1:
        raise ValueError("minimum_valid_corpus_size must be a positive integer")
    count = len(_audio_files(directory))
    if count < minimum:
        raise ValueError(
            f"{corpus_name} corpus has {count} audio files; metric requires at least {minimum}. "
            "Refusing to duplicate samples to force a score."
        )


def _reject_short_audio_files(directory: Path, minimum_duration_seconds: float, corpus_name: str) -> None:
    if minimum_duration_seconds <= 0:
        return
    import soundfile as sf

    too_short: list[str] = []
    for path in _audio_files(directory):
        info = sf.info(str(path))
        duration = info.frames / info.samplerate if info.samplerate else 0.0
        if duration < minimum_duration_seconds:
            too_short.append(f"{path.name} ({duration:.6g}s)")
    if too_short:
        preview = ", ".join(too_short[:10])
        suffix = "" if len(too_short) <= 10 else f", ... {len(too_short) - 10} more"
        raise ValueError(
            f"{corpus_name} corpus contains {len(too_short)} audio files shorter than the "
            f"declared {minimum_duration_seconds:g}s backend constraint: {preview}{suffix}. "
            "Refusing to pad or modify audio to force a score."
        )


def _link_or_copy(source: Path, target: Path) -> None:
    source = source.resolve(strict=True)
    if target.exists() or target.is_symlink():
        target.unlink()
    try:
        target.symlink_to(source)
    except OSError:
        shutil.copyfile(source, target)


def _assign_sample_scores(rows: list[dict[str, Any]], sample_metrics: dict[str, dict[str, float]], metric: str, scores: Any) -> None:
    values = _sample_score_vector(scores, len(rows), metric)
    for row, value in zip(rows, values):
        sample_metrics[str(row["row_index"])][metric] = value


def _sample_score_vector(scores: Any, expected_count: int, metric: str) -> tuple[float, ...]:
    values = scores.detach().cpu().tolist() if hasattr(scores, "detach") else scores
    if values is None or isinstance(values, (str, bytes, dict, int, float)):
        raise ValueError(f"{metric} must return a complete one-dimensional sample score vector")
    shape = getattr(values, "shape", None)
    if shape is not None and len(shape) != 1:
        raise ValueError(f"{metric} sample scores must be one-dimensional, found shape {tuple(shape)}")
    if hasattr(values, "tolist"):
        values = values.tolist()
    try:
        count = len(values)
    except TypeError as exc:
        raise ValueError(f"{metric} must return a complete one-dimensional sample score vector") from exc
    if count != expected_count:
        raise ValueError(
            f"{metric} returned {count} sample scores for {expected_count} rows; "
            "refusing to attach incomplete or misaligned scores"
        )
    return tuple(_scalar(value, metric=metric) for value in values)


def _require_sample_mean(corpus_score: float, sample_scores: tuple[float, ...], metric: str) -> None:
    sample_mean = math.fsum(sample_scores) / len(sample_scores)
    if not math.isclose(corpus_score, sample_mean, rel_tol=1e-7, abs_tol=1e-12):
        raise ValueError(
            f"official {metric} corpus score {corpus_score} does not match sample mean {sample_mean}"
        )


def _resolve_java_path(payload: dict[str, Any]) -> str:
    configured = _first_config_value(payload, "java_path")
    candidate = configured or os.environ.get("AAC_METRICS_JAVA_PATH") or shutil.which("java")
    if not candidate:
        raise RuntimeError(
            "SPIDEr/FENSE require Java, but no executable was found. "
            "Load Java and export AAC_METRICS_JAVA_PATH=$(command -v java)."
        )
    path = Path(candidate).expanduser()
    if not path.is_absolute():
        resolved = shutil.which(str(path))
        if not resolved:
            raise RuntimeError(f"Configured Java executable is not resolvable: {candidate}")
        path = Path(resolved)
    if not path.is_file() or not os.access(path, os.X_OK):
        raise RuntimeError(f"Configured Java executable is not executable: {path}")
    return str(path.resolve())


def _scalar(value: Any, *, metric: str = "metric", rejected_values: tuple[float, ...] = ()) -> float:
    if value is None:
        raise ValueError(f"{metric} returned no value")
    if hasattr(value, "detach"):
        value = value.detach().cpu().item()
    if hasattr(value, "item"):
        value = value.item()
    numeric = float(value)
    if not math.isfinite(numeric):
        raise ValueError(f"{metric} returned non-finite value {numeric!r}")
    if numeric in rejected_values:
        raise ValueError(f"{metric} returned rejected sentinel value {numeric!r}")
    return numeric


def _first_config_value(payload: dict[str, Any], key: str) -> Any:
    for name in payload["metrics"]:
        value = payload["metric_configs"].get(name, {}).get(key)
        if value:
            return value
    return None


def _result(
    payload: dict[str, Any],
    *,
    sample_metrics: dict[str, Any],
    corpus_metrics: dict[str, Any],
    predictions: dict[str, Any],
    errors: dict[str, Any],
    packages: list[str],
) -> dict[str, Any]:
    return {
        "sample_metrics": sample_metrics,
        "corpus_metrics": corpus_metrics,
        "predictions": predictions,
        "errors": errors,
        "metric_backends": _backend_metadata(payload, packages),
    }


def _backend_metadata(payload: dict[str, Any], packages: list[str]) -> dict[str, dict[str, Any]]:
    versions = {}
    for package in packages:
        try:
            versions[package] = importlib.metadata.version(package)
        except importlib.metadata.PackageNotFoundError:
            versions[package] = None
    executable = sys.executable
    return {
        metric: {
            "runtime_python": executable,
            "runtime_package_versions": versions,
        }
        for metric in payload["metrics"]
    }


def _transcribe_whisper(model: Any, audio_path: str, device: str) -> str:
    result = model.transcribe(audio_path, fp16=str(device).startswith("cuda"))
    return str(result.get("text", "")).strip()


def _normalize_text(text: str) -> str:
    return " ".join(str(text).strip().split())


def _metric_error(exc: Exception) -> str:
    return f"{type(exc).__name__}: {exc}"


def _single_line_text(value: Any) -> str:
    return " ".join(str(value or "").split())


def _word_error_rate(reference: str, hypothesis: str) -> float:
    ref_words = reference.split()
    hyp_words = hypothesis.split()
    if not ref_words:
        return 0.0 if not hyp_words else 1.0
    previous = list(range(len(hyp_words) + 1))
    for i, ref_word in enumerate(ref_words, start=1):
        current = [i]
        for j, hyp_word in enumerate(hyp_words, start=1):
            substitution = previous[j - 1] + int(ref_word != hyp_word)
            insertion = current[j - 1] + 1
            deletion = previous[j] + 1
            current.append(min(substitution, insertion, deletion))
        previous = current
    return previous[-1] / len(ref_words)


if __name__ == "__main__":
    main()
