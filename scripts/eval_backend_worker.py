#!/usr/bin/env python
"""Run one metric backend inside its dedicated environment.

The main evaluator invokes this script with the backend-specific Python
interpreter configured in configs/eval_metrics.json.
"""

from __future__ import annotations

import argparse
import importlib.metadata
import json
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
    elif backend == "audioldm_eval":
        result = run_audioldm_eval(payload)
    elif backend == "kadtk":
        result = run_kadtk(payload)
    else:
        raise SystemExit(f"Unsupported backend worker: {backend}")

    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", encoding="utf-8") as f:
        json.dump(result, f, indent=2, sort_keys=True)
        f.write("\n")


def run_aac_metrics(payload: dict[str, Any]) -> dict[str, Any]:
    from aac_metrics import evaluate as aac_evaluate
    from aac_metrics.functional.clap_sim import clap_sim
    from aac_metrics.functional.mace import mace as mace_fn

    metrics = set(payload["metrics"])
    rows = payload["rows"]
    candidates = [row["candidate_text"] for row in rows]
    references = [[row["source"]["audio_cap"]] for row in rows]
    audio_paths = [row["source"]["audio_file"] for row in rows]
    cache_path = _first_config_value(payload, "cache_dir") or payload["cache_dir"]
    device = payload["device"]
    sample_metrics: dict[str, dict[str, float]] = {str(row["row_index"]): {} for row in rows}
    corpus_metrics: dict[str, Any] = {}

    java_path = shutil.which("java")
    java_tool_options = os.environ.pop("JAVA_TOOL_OPTIONS", None)
    try:
        if "spider" in metrics:
            corpus_scores, sent_scores = aac_evaluate(
                candidates,
                references,
                metrics=["spider"],
                cache_path=cache_path,
                java_path=java_path,
                device=device,
            )
            _assign_sample_scores(rows, sample_metrics, "spider", sent_scores.get("spider"))
            corpus_metrics["spider"] = _scalar(corpus_scores.get("spider"))

        if "mace" in metrics:
            cfg = payload["metric_configs"].get("mace", {})
            corpus_scores, sent_scores = mace_fn(
                candidates,
                references,
                audio_paths=audio_paths,
                mace_method=cfg.get("mace_method", "combined"),
                clap_model=cfg.get("clap_model", "MS-CLAP-2023"),
                echecker=cfg.get("echecker", "echecker_clotho_audiocaps_base"),
                device=device,
                return_all_scores=True,
            )
            _assign_sample_scores(rows, sample_metrics, "mace", sent_scores.get("mace"))
            corpus_metrics["mace"] = _scalar(corpus_scores.get("mace"))

        if "aac_clapsim" in metrics:
            corpus_scores, sent_scores = clap_sim(
                candidates,
                audio_paths=audio_paths,
                clap_method="audio",
                clap_model=payload["metric_configs"].get("aac_clapsim", {}).get("model_name", "MS-CLAP-2023"),
                device=device,
                return_all_scores=True,
            )
            _assign_sample_scores(rows, sample_metrics, "aac_clapsim", sent_scores.get("clap_sim"))
            corpus_metrics["aac_clapsim"] = _scalar(corpus_scores.get("clap_sim"))
    finally:
        if java_tool_options is not None:
            os.environ["JAVA_TOOL_OPTIONS"] = java_tool_options

    return _result(
        payload,
        sample_metrics=sample_metrics,
        corpus_metrics=corpus_metrics,
        predictions={},
        errors={},
        packages=["aac-metrics", "torch", "numpy"],
    )


def run_whisper(payload: dict[str, Any]) -> dict[str, Any]:
    import whisper

    metrics = set(payload["metrics"])
    cfg = payload["metric_configs"].get("asr_wer") or payload["metric_configs"].get("dwer") or {}
    model_name = cfg.get("model_name", "large-v3")
    cache_dir = cfg.get("cache_dir") or payload["cache_dir"]
    device = payload["device"]
    model = whisper.load_model(model_name, download_root=cache_dir, device=device)

    sample_metrics: dict[str, dict[str, float]] = {}
    predictions: dict[str, dict[str, str]] = {}
    for row in payload["rows"]:
        row_key = str(row["row_index"])
        generated_text = _transcribe_whisper(model, row["output_file"], device)
        sample_metrics[row_key] = {}
        predictions[row_key] = {"generated_speech_asr": generated_text}
        if "asr_wer" in metrics:
            reference = _normalize_text(row["source"]["speech_text"])
            sample_metrics[row_key]["asr_wer"] = _word_error_rate(reference, _normalize_text(generated_text))
        if "dwer" in metrics:
            reference_text = _transcribe_whisper(model, row["source"]["speech_file"], device)
            predictions[row_key]["reference_speech_asr"] = reference_text
            sample_metrics[row_key]["dwer"] = _word_error_rate(_normalize_text(reference_text), _normalize_text(generated_text))
    return _result(
        payload,
        sample_metrics=sample_metrics,
        corpus_metrics={},
        predictions=predictions,
        errors={},
        packages=["openai-whisper", "torch", "numpy"],
    )


def run_utmosv2(payload: dict[str, Any]) -> dict[str, Any]:
    import utmosv2

    cfg = payload["metric_configs"].get("utmos", {})
    device = payload["device"]
    model = utmosv2.create_model(
        pretrained=True,
        config=cfg.get("config_name", "fusion_stage3"),
        checkpoint_path=cfg.get("model_path"),
        device=device,
    )
    sample_metrics: dict[str, dict[str, float]] = {}
    for row in payload["rows"]:
        score = model.predict(
            input_path=row["output_file"],
            device=device,
            num_workers=int(cfg.get("num_workers", 0)),
            batch_size=int(cfg.get("batch_size", 1)),
            verbose=False,
        )
        sample_metrics[str(row["row_index"])] = {"utmos": float(score)}
    return _result(
        payload,
        sample_metrics=sample_metrics,
        corpus_metrics={},
        predictions={},
        errors={},
        packages=["utmosv2", "torch", "numpy"],
    )


def run_audioldm_eval(payload: dict[str, Any]) -> dict[str, Any]:
    from audioldm_eval import EvaluationHelper
    import torch

    metrics = set(payload["metrics"])
    cfg = payload["metric_configs"].get("fad") or payload["metric_configs"].get("inception_score") or {}
    generated_dir, reference_dir = _stage_audio_dirs(payload, reference_field="audio_file")
    evaluator = EvaluationHelper(
        int(cfg.get("sample_rate", 16000)),
        torch.device(payload["device"]),
        backbone=cfg.get("backbone", "cnn14"),
    )
    raw = evaluator.main(str(generated_dir), str(reference_dir), limit_num=len(payload["rows"]))
    corpus_metrics: dict[str, Any] = {}
    if "fad" in metrics:
        corpus_metrics["fad"] = _scalar(raw.get("frechet_audio_distance"))
    if "inception_score" in metrics:
        mean = _scalar(raw.get("inception_score_mean"))
        std = _scalar(raw.get("inception_score_std"))
        corpus_metrics["inception_score"] = {
            "level": "corpus",
            "count": len(payload["rows"]),
            "mean": mean,
            "std": std,
            "min": mean,
            "max": mean,
            "median": mean,
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
    value = float(score.item() if hasattr(score, "item") else score)
    return _result(
        payload,
        sample_metrics={},
        corpus_metrics={"kad": value},
        predictions={},
        errors={},
        packages=["kadtk", "torch", "numpy", "tensorflow"],
    )


def _stage_audio_dirs(payload: dict[str, Any], reference_field: str) -> tuple[Path, Path]:
    work_dir = Path(payload["work_dir"])
    generated_dir = work_dir / payload["backend"] / "generated"
    reference_dir = work_dir / payload["backend"] / "reference"
    generated_dir.mkdir(parents=True, exist_ok=True)
    reference_dir.mkdir(parents=True, exist_ok=True)
    for index, row in enumerate(payload["rows"]):
        stem = f"{index:06d}_{row['sample_id']}"
        _link_or_copy(Path(row["output_file"]), generated_dir / f"{stem}.wav")
        _link_or_copy(Path(row["source"][reference_field]), reference_dir / f"{stem}.wav")
    return generated_dir, reference_dir


def _link_or_copy(source: Path, target: Path) -> None:
    source = source.resolve(strict=True)
    if target.exists() or target.is_symlink():
        target.unlink()
    try:
        target.symlink_to(source)
    except OSError:
        shutil.copyfile(source, target)


def _assign_sample_scores(rows: list[dict[str, Any]], sample_metrics: dict[str, dict[str, float]], metric: str, scores: Any) -> None:
    values = scores.detach().cpu().tolist() if hasattr(scores, "detach") else scores
    for row, value in zip(rows, values):
        sample_metrics[str(row["row_index"])][metric] = float(value)


def _scalar(value: Any) -> float:
    if hasattr(value, "detach"):
        value = value.detach().cpu().item()
    if hasattr(value, "item"):
        value = value.item()
    return float(value)


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
