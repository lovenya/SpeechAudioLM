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
    from aac_metrics.functional.clap_sim import clap_sim
    from aac_metrics.functional.fense import fense as fense_fn
    from aac_metrics.functional.mace import mace as mace_fn
    from aac_metrics.functional.spider import spider as spider_fn
    from evaluation.metrics.aac_audio import (
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
    raw_candidates = [row["candidate_text"] for row in rows]
    raw_references = [[row["source"]["audio_cap"]] for row in rows]
    spider_candidates = [prepare_spice_caption(candidate) for candidate in raw_candidates]
    spider_references = [
        [prepare_spice_caption(reference) for reference in sample_references]
        for sample_references in raw_references
    ]
    audio_paths = [row["source"]["audio_file"] for row in rows]
    cache_path = _first_config_value(payload, "cache_dir") or payload["cache_dir"]
    device = payload["device"]
    sample_metrics: dict[str, dict[str, float]] = {str(row["row_index"]): {} for row in rows}
    corpus_metrics: dict[str, Any] = {}
    policy_metadata: dict[str, dict[str, Any]] = {}

    java_path = _resolve_java_path(payload) if "spider" in metrics else None
    java_tool_options = os.environ.pop("JAVA_TOOL_OPTIONS", None)
    try:
        if "spider" in metrics:
            cfg = payload["metric_configs"].get("spider", {})
            corpus_scores, sent_scores = spider_fn(
                spider_candidates,
                spider_references,
                cache_path=cache_path,
                java_path=java_path,
                java_max_memory=str(cfg.get("java_max_memory", "32G")),
                return_all_scores=True,
            )
            _assign_sample_scores(rows, sample_metrics, "spider", sent_scores.get("spider"))
            corpus_metrics["spider"] = _scalar(corpus_scores.get("spider"), metric="spider")
            policy_metadata["spider"] = caption_preprocessing_metadata() | {
                "corpus_score_policy": "preserve official corpus SPIDEr; do not compare with sentence-score mean",
            }

        if "mace" in metrics:
            cfg = payload["metric_configs"].get("mace", {})
            penalty = float(cfg.get("penalty", 0.3))
            error_threshold = float(cfg.get("error_threshold", 0.97))
            mace_method = cfg.get("mace_method", "combined")
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
                expected_count=len(rows),
                penalty=penalty,
                corpus_score=official_corpus_score,
                mace_method=mace_method,
                seed=PROJECT_EVALUATION_SEED,
            )
            _assign_sample_scores(rows, sample_metrics, "mace", reconstruction.scores)
            corpus_metrics["mace"] = official_corpus_score
            policy_metadata["mace"] = dict(reconstruction.metadata) | {
                "error_threshold": error_threshold,
            }

        if "aac_clapsim" in metrics:
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
                expected_count=len(rows),
                corpus_score=official_corpus_score,
                seed=PROJECT_EVALUATION_SEED,
            )
            _assign_sample_scores(rows, sample_metrics, "aac_clapsim", validated.scores)
            corpus_metrics["aac_clapsim"] = official_corpus_score
            policy_metadata["aac_clapsim"] = dict(validated.metadata)

        if "fense" in metrics:
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
            fense_scores = _sample_score_vector(sent_scores.get("fense"), len(rows), "fense")
            official_corpus_score = _scalar(corpus_scores.get("fense"), metric="fense")
            _require_sample_mean(official_corpus_score, fense_scores, "fense")
            _assign_sample_scores(rows, sample_metrics, "fense", fense_scores)
            corpus_metrics["fense"] = official_corpus_score
            policy_metadata["fense"] = caption_preprocessing_metadata() | {
                "penalty": penalty,
                "error_threshold": error_threshold,
                "corpus_aggregation": "arithmetic mean over the complete official sample vector",
            }
    finally:
        if java_tool_options is not None:
            os.environ["JAVA_TOOL_OPTIONS"] = java_tool_options

    result = _result(
        payload,
        sample_metrics=sample_metrics,
        corpus_metrics=corpus_metrics,
        predictions={},
        errors={},
        packages=["aac-metrics", "torch", "numpy"],
    )
    for metric, metadata in policy_metadata.items():
        result["metric_backends"][metric]["policy_metadata"] = metadata
    return result



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
    if "asr_wer" in metrics:
        for row in rows:
            reference = row.get("source", {}).get("speech_text")
            if not isinstance(reference, str) or not reference.strip():
                raise ValueError(
                    f"asr_wer requires a non-empty reference transcript for row {row.get('row_index')!r}"
                )
            score_wer_pair(reference, reference)

    import whisper

    cfg = payload["metric_configs"].get("asr_wer") or payload["metric_configs"].get("dwer") or {}
    model_name = cfg.get("model_name", "large-v3")
    cache_dir = cfg.get("cache_dir") or payload["cache_dir"]
    device = payload["device"]
    model = whisper.load_model(model_name, download_root=cache_dir, device=device)

    sample_metrics: dict[str, dict[str, float]] = {}
    predictions: dict[str, dict[str, Any]] = {}
    scores_by_metric: dict[str, list[dict[str, dict[str, Any]]]] = {
        metric: [] for metric in metrics
    }
    for row in rows:
        row_key = str(row["row_index"])
        generated_text = transcribe_whisper(model, row["output_file"], device)
        sample_metrics[row_key] = {}
        predictions[row_key] = {"generated_speech_asr": generated_text}
        if "asr_wer" in metrics:
            score = score_wer_pair(row["source"]["speech_text"], generated_text)
            scores_by_metric["asr_wer"].append(score)
            sample_metrics[row_key]["asr_wer"] = float(score[NORMALIZED_SCORE]["wer"])
            predictions[row_key]["asr_wer_details"] = score
        if "dwer" in metrics:
            reference_text = transcribe_whisper(model, row["source"]["speech_file"], device)
            score = score_wer_pair(reference_text, generated_text)
            scores_by_metric["dwer"].append(score)
            predictions[row_key]["reference_speech_asr"] = reference_text
            predictions[row_key]["dwer_details"] = score
            sample_metrics[row_key]["dwer"] = float(score[NORMALIZED_SCORE]["wer"])

    semantics = {
        "asr_wer": "WER(reference_transcript, ASR(generated_speech))",
        "dwer": "WER(ASR(reference_speech), ASR(generated_speech))",
    }
    corpus_metrics = {}
    for metric, scores in scores_by_metric.items():
        aggregates = aggregate_wer_scores(scores)
        corpus_metrics[metric] = {
            "level": "corpus",
            "count": 1,
            "input_count": len(scores),
            "mean": aggregates[NORMALIZED_SCORE]["wer"],
            "mean_semantics": "normalized corpus WER from summed edit counts; not sample mean",
            "semantics": semantics[metric],
            **aggregates,
        }
    result = _result(
        payload,
        sample_metrics=sample_metrics,
        corpus_metrics=corpus_metrics,
        predictions=predictions,
        errors={},
        packages=["openai-whisper", "torch", "numpy"],
    )
    for metric in metrics:
        result["metric_backends"][metric]["policy_metadata"] = {
            "semantics": semantics[metric],
            "sample_metric": "normalized utterance WER",
            "corpus_aggregation": "WER from summed edit counts; macro_wer is secondary",
            "corpus": corpus_metrics[metric],
        }
    return result


def run_utmosv2(payload: dict[str, Any]) -> dict[str, Any]:
    import utmosv2
    from evaluation.metrics.utmos_policy import (
        UTMOSV2_CHECKPOINT_FOLD,
        UTMOSV2_CHECKPOINT_SEED,
        UTMOSV2_CONFIG,
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
    for row in payload["rows"]:
        row_key = str(row["row_index"])
        score = model.predict(
            input_path=row["output_file"],
            device=device,
            num_workers=int(cfg.get("num_workers", 0)),
            batch_size=int(cfg.get("batch_size", 1)),
            predict_dataset=predict_dataset,
            num_repetitions=num_repetitions,
            remove_silent_section=remove_silent_section,
            verbose=False,
        )
        records.append({"sample_id": row_key, "utmos": score})

    aggregation = validate_utmos_records(
        records,
        expected_ids=[str(row["row_index"]) for row in payload["rows"]],
    )
    sample_metrics = {
        record.sample_id: {"utmos": record.score}
        for record in aggregation.records
    }
    result = _result(
        payload,
        sample_metrics=sample_metrics,
        corpus_metrics={"utmos": aggregation.corpus_mean},
        predictions={},
        errors={},
        packages=["utmosv2", "torch", "numpy"],
    )
    result["metric_backends"]["utmos"]["policy_metadata"] = dict(aggregation.metadata)
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

    metrics = set(payload["metrics"])
    cfg = payload["metric_configs"].get("fad") or payload["metric_configs"].get("inception_score") or {}
    runtime_home = Path(cfg.get("runtime_home") or Path(cfg.get("cache_dir") or payload["cache_dir"]).parent / "runtime_home")
    runtime_home.mkdir(parents=True, exist_ok=True)
    generated_dir, reference_dir = _stage_audio_dirs(payload, reference_field="audio_file")
    minimum_size = int(cfg.get("minimum_valid_corpus_size", 2))
    _require_minimum_audio_files(generated_dir, minimum_size, "generated")
    _require_minimum_audio_files(reference_dir, minimum_size, "reference")
    previous_home = os.environ.get("HOME")
    try:
        torch.load = load_legacy_checkpoint
        os.environ["HOME"] = str(runtime_home)
        evaluator = EvaluationHelper(
            int(cfg.get("sample_rate", 16000)),
            torch.device(payload["device"]),
            backbone=cfg.get("backbone", "cnn14"),
        )
        raw = evaluator.main(str(generated_dir), str(reference_dir), limit_num=len(payload["rows"]))
    finally:
        torch.load = original_torch_load
        if previous_home is None:
            os.environ.pop("HOME", None)
        else:
            os.environ["HOME"] = previous_home
    corpus_metrics: dict[str, Any] = {}
    if "fad" in metrics:
        corpus_metrics["fad"] = _scalar(raw.get("frechet_audio_distance"), metric="fad", rejected_values=(-1.0,))
    if "inception_score" in metrics:
        mean = _scalar(raw.get("inception_score_mean"))
        std = _scalar(raw.get("inception_score_std"))
        corpus_metrics["inception_score"] = {
            "level": "corpus",
            "status": "backend_corpus_summary",
            "input_count": len(payload["rows"]),
            "mean": mean,
            "std": std,
        }
    return _result(
        payload,
        sample_metrics={},
        corpus_metrics=corpus_metrics,
        predictions={},
        errors={},
        packages=["audioldm_eval", "torch", "numpy"],
    )


def run_kadtk(payload: dict[str, Any]) -> dict[str, Any]:
    cfg = payload["metric_configs"].get("kad", {})
    generated_dir, reference_dir = _stage_audio_dirs(payload, reference_field="audio_file")
    minimum_size = int(cfg.get("minimum_valid_corpus_size", 2))
    _require_minimum_audio_files(generated_dir, minimum_size, "generated")
    _require_minimum_audio_files(reference_dir, minimum_size, "reference")
    minimum_duration_seconds = float(cfg.get("minimum_duration_seconds", 1.0))
    _reject_short_audio_files(generated_dir, minimum_duration_seconds, "generated")
    _reject_short_audio_files(reference_dir, minimum_duration_seconds, "reference")
    import kadtk.model_loader as model_loader
    import torch
    from kadtk.emb_loader import EmbeddingLoader
    from kadtk.kad import KernelAudioDistance

    cache_dir = Path(cfg.get("cache_dir") or payload["cache_dir"])
    cache_dir.mkdir(parents=True, exist_ok=True)
    # kadtk derives checkpoint location from model_loader.__file__; redirect it
    # away from site-packages so project inode quota is not consumed.
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
    metric = KernelAudioDistance(
        model,
        payload["device"],
        audio_load_worker=workers,
        force_stats_calc=bool(cfg.get("force_stats_calc", False)),
    )
    score = metric.score(reference_dir, generated_dir)
    value = _scalar(score, metric="kad")
    return _result(
        payload,
        sample_metrics={},
        corpus_metrics={"kad": value},
        predictions={},
        errors={},
        packages=["kadtk", "torch", "numpy", "tensorflow"],
    )



def run_stable_audio_metrics(payload: dict[str, Any]) -> dict[str, Any]:
    metrics = set(payload["metrics"])
    sample_metrics: dict[str, dict[str, float]] = {}
    corpus_metrics: dict[str, Any] = {}
    errors: dict[str, dict[str, str]] = {}

    if "clap_score_t2a" in metrics:
        clap_scores, clap_errors = _stable_clap_scores(payload)
        sample_metrics.update(clap_scores)
        errors.update(clap_errors)
    if "kl_passt" in metrics:
        corpus_metrics["kl_passt"] = _stable_kl_passt(payload)

    return _result(
        payload,
        sample_metrics=sample_metrics,
        corpus_metrics=corpus_metrics,
        predictions={},
        errors=errors,
        packages=["laion-clap", "hear21passt", "torch", "numpy", "librosa", "pyloudnorm"],
    )


def _stable_clap_scores(payload: dict[str, Any]) -> tuple[dict[str, dict[str, float]], dict[str, dict[str, str]]]:
    import numpy as np
    import requests
    import torch
    import laion_clap
    import librosa
    import pyloudnorm as pyln
    from clap_module.factory import load_state_dict
    from tqdm import tqdm

    cfg = payload["metric_configs"].get("clap_score_t2a", {})
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

    ids = [_stage_id(index, row) for index, row in enumerate(payload["rows"])]
    texts = [row["source"].get("audio_cap") or row.get("candidate_text") or "" for row in payload["rows"]]

    def batched_tokenizer(text: list[str]) -> dict[str, Any]:
        return model.tokenize(
            text,
            padding="max_length",
            truncation=True,
            max_length=77,
            return_tensors="pt",
        )

    with torch.no_grad():
        text_embeddings = model.get_text_embedding(texts, tokenizer=batched_tokenizer, use_tensor=True)

    sample_metrics: dict[str, dict[str, float]] = {}
    errors: dict[str, dict[str, str]] = {}
    for row, stem, text_embedding in zip(payload["rows"], ids, text_embeddings):
        row_key = str(row["row_index"])
        try:
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
            sample_metrics[row_key] = {"clap_score_t2a": float(score.detach().cpu().item())}
        except Exception as exc:
            errors[row_key] = {"clap_score_t2a": _metric_error(exc)}
    return sample_metrics, errors

def _stable_kl_passt(payload: dict[str, Any]) -> float:
    from passt_kld import passt_kld

    cfg = payload["metric_configs"].get("kl_passt", {})
    generated_dir, reference_dir = _stage_audio_dirs(payload, reference_field="audio_file")
    ids = [_stage_id(index, row) for index, row in enumerate(payload["rows"])]
    cache_dir = Path(cfg.get("cache_dir") or payload["cache_dir"])
    cache_dir.mkdir(parents=True, exist_ok=True)
    previous_cwd = Path.cwd()
    try:
        os.chdir(cache_dir)
        value = passt_kld(
            ids=ids,
            eval_path=str(generated_dir),
            eval_files_extension=".wav",
            ref_path=str(reference_dir),
            ref_files_extension=".wav",
            collect=cfg.get("collect", "mean"),
        )
    finally:
        os.chdir(previous_cwd)
    return _scalar(value)


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
