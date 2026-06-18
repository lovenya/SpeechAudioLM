#!/usr/bin/env bash
# Interactive GPU demo for canonical test-split generated-audio/speech metrics.
# Scope: audio_omni T2A and cosyvoice3 T2S only. Writes temporary eval outputs.

set -euo pipefail

cd /scratch/l/lovenya/SpeechAudioLM

module --force purge
module load StdEnv/2023 gcc cuda/12.6 opencv/4.13.0 java/11.0.22 python/3.12.4 scipy-stack
unset JAVA_TOOL_OPTIONS
export AAC_METRICS_JAVA_PATH="$(command -v java)"

test -x "$AAC_METRICS_JAVA_PATH" || { echo "missing Java executable" >&2; exit 2; }

export SALM_SCRATCH="${SALM_SCRATCH:-/scratch/l/lovenya/salm_migration_clean}"
export SALM_RESULTS_ROOT="$SALM_SCRATCH/results"
export SALM_TEMPORARY_RESULTS_ROOT="$SALM_SCRATCH/temporary/results"
export SALM_LOGS_ROOT="$SALM_SCRATCH/logs"
export SALM_TEMPORARY_LOGS_ROOT="$SALM_SCRATCH/temporary/logs"
export SALM_CACHE_ROOT="$SALM_SCRATCH/cache"
export SALM_MODEL_ROOT="$SALM_SCRATCH/models"
export SALM_ENVIRONMENT_ROOT="$SALM_SCRATCH/.venvs"
export SALM_ENVIRONMENTS_ROOT="$SALM_SCRATCH/.venvs"

export PYTHONNOUSERSITE=1
export PYTHONDONTWRITEBYTECODE=1
export HF_HOME="$SALM_SCRATCH/cache/metrics/huggingface"
export HF_HUB_CACHE="$HF_HOME"
export HUGGINGFACE_HUB_CACHE="$HF_HOME"
export TRANSFORMERS_CACHE="$HF_HOME"
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export TORCH_HOME="$SALM_SCRATCH/cache/metrics/torch"
export TOKENIZERS_PARALLELISM=false
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-1}"
export MKL_NUM_THREADS="${MKL_NUM_THREADS:-1}"
export NUMEXPR_NUM_THREADS="${NUMEXPR_NUM_THREADS:-1}"
export RESOURCE_INTERVAL="${RESOURCE_INTERVAL:-30}"

MAX_SAMPLES="${MAX_SAMPLES:-50}"
SHARED_PYTHON="${SHARED_PYTHON:-/home/l/lovenya/.venvs/speechaudiolm/.venv-eval-metrics-shared-py312/bin/python}"
AUDIOLDM_PYTHON="${AUDIOLDM_PYTHON:-$SALM_SCRATCH/.venvs/metrics/.venv-eval-audioldm-eval/bin/python}"
KADTK_PYTHON="${KADTK_PYTHON:-$SALM_SCRATCH/.venvs/metrics/.venv-eval-kadtk/bin/python}"
STABLE_AUDIO_PYTHON="${STABLE_AUDIO_PYTHON:-$SALM_SCRATCH/.venvs/metrics/.venv-eval-stable-audio-metrics/bin/python}"
STABLE_AUDIO_SOURCE="${STABLE_AUDIO_SOURCE:-$SALM_SCRATCH/code/metrics/stable-audio-metrics/src}"

RUN_ID="$(TZ=America/Toronto date +%Y%m%dT%H%M%S%z)"
LOG_DIR="$SALM_SCRATCH/logs/evals/audio_omni_cosyvoice3_gpu_demo/$RUN_ID"
RUNTIME_CONFIG="$LOG_DIR/eval_metrics.runtime.json"
T2A_MANIFEST="$SALM_SCRATCH/results/cascaded/stages/t2a/audio_omni/from-t2t/llama3_3_70b_instruct/ttas/test/20260618_1211/manifest.jsonl"
T2S_MANIFEST="$SALM_SCRATCH/results/cascaded/stages/t2s/cosyvoice3/from-t2t/llama3_3_70b_instruct/ttas/test/20260618_1211/manifest.jsonl"
export MAX_SAMPLES SHARED_PYTHON AUDIOLDM_PYTHON KADTK_PYTHON STABLE_AUDIO_PYTHON STABLE_AUDIO_SOURCE
export RUN_ID LOG_DIR RUNTIME_CONFIG T2A_MANIFEST T2S_MANIFEST

mkdir -p "$LOG_DIR" "$SALM_SCRATCH/logs/resources" "$SALM_SCRATCH/logs/slurm"
test -x "$SHARED_PYTHON" || { echo "missing shared python: $SHARED_PYTHON" >&2; exit 2; }
test -x "$AUDIOLDM_PYTHON" || { echo "missing AudioLDM Eval python: $AUDIOLDM_PYTHON" >&2; exit 2; }
test -x "$KADTK_PYTHON" || { echo "missing KADTK python: $KADTK_PYTHON" >&2; exit 2; }
test -x "$STABLE_AUDIO_PYTHON" || { echo "missing stable-audio-metrics python: $STABLE_AUDIO_PYTHON" >&2; exit 2; }
test -d "$STABLE_AUDIO_SOURCE" || { echo "missing stable-audio-metrics source: $STABLE_AUDIO_SOURCE" >&2; exit 2; }
test -f "$T2A_MANIFEST" || { echo "missing T2A manifest: $T2A_MANIFEST" >&2; exit 2; }
test -f "$T2S_MANIFEST" || { echo "missing T2S manifest: $T2S_MANIFEST" >&2; exit 2; }

python - <<'PYCONFIG'
import json
import os
import subprocess
from pathlib import Path

runtime_config = Path(os.environ["RUNTIME_CONFIG"])
cfg = json.loads(Path("configs/eval_metrics.json").read_text(encoding="utf-8"))
for name in ("asr_wer", "dwer", "utmos", "dnsmos"):
    cfg["metrics"][name]["python"] = os.environ["SHARED_PYTHON"]
for name in ("fad", "inception_score"):
    cfg["metrics"][name]["python"] = os.environ["AUDIOLDM_PYTHON"]
cfg["metrics"]["kad"]["python"] = os.environ["KADTK_PYTHON"]
for name in ("target_caption_clap_score", "prompt_adherence_clap_score", "kl_passt"):
    cfg["metrics"][name]["python"] = os.environ["STABLE_AUDIO_PYTHON"]
    cfg["metrics"][name]["pythonpath"] = os.environ["STABLE_AUDIO_SOURCE"]
runtime_config.write_text(json.dumps(cfg, indent=2) + "\n", encoding="utf-8")

metadata = {
    "script": "submission_scripts/run_tamia_eval_audio_omni_cosyvoice3_gpu_demo_interactive.sh",
    "job_id": os.environ.get("SLURM_JOB_ID", "interactive"),
    "run_id": os.environ["RUN_ID"],
    "scope": "temporary_test_audio_omni_t2a_and_cosyvoice3_t2s",
    "max_samples": int(os.environ["MAX_SAMPLES"]),
    "repo": str(Path.cwd()),
    "git_commit": subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip(),
    "git_dirty": bool(subprocess.check_output(["git", "status", "--porcelain"], text=True).strip()),
    "metrics_config": str(runtime_config),
    "t2a_manifest": os.environ["T2A_MANIFEST"],
    "t2s_manifest": os.environ["T2S_MANIFEST"],
}
Path(os.environ["LOG_DIR"], "job_metadata.json").write_text(json.dumps(metadata, indent=2) + "\n", encoding="utf-8")
print(f"runtime_config={runtime_config}")
print(f"job_metadata={Path(os.environ['LOG_DIR'], 'job_metadata.json')}")
PYCONFIG

source submission_scripts/lib_resource_monitor.sh
start_resource_monitor "audio_omni_cosyvoice3_gpu_demo_${RUN_ID}_n${MAX_SAMPLES}"

run_eval() {
  local label="$1"
  local stage="$2"
  local metrics="$3"
  local manifest="$4"
  local gpu="$5"
  local log="$LOG_DIR/${label}.log"

  echo "START label=$label stage=$stage split=test metrics=$metrics evaluated_run_id=20260618_1211 gpu=$gpu max_samples=$MAX_SAMPLES" | tee "$log"

  CUDA_VISIBLE_DEVICES="$gpu" python evaluate.py \
    --model cascaded \
    --stage "$stage" \
    --evaluated-run-id 20260618_1211 \
    --dataset ttas \
    --split test \
    --manifest "$manifest" \
    --metrics "$metrics" \
    --metrics-config "$RUNTIME_CONFIG" \
    --cache-dir "$SALM_SCRATCH/cache/metrics/eval_runtime" \
    --device cuda \
    --temporary \
    --max-samples "$MAX_SAMPLES" >>"$log" 2>&1

  echo "PASS label=$label" | tee -a "$log"
}

failures=0
pids=()

run_eval t2a_audio_omni_audioldm t2a fad,inception_score "$T2A_MANIFEST" 0 & pids+=("$!")
run_eval t2a_audio_omni_kad t2a kad "$T2A_MANIFEST" 1 & pids+=("$!")
run_eval t2a_audio_omni_stable t2a target_caption_clap_score,prompt_adherence_clap_score,kl_passt "$T2A_MANIFEST" 2 & pids+=("$!")
run_eval t2s_cosyvoice3_quality t2s asr_wer,dwer,utmos,dnsmos "$T2S_MANIFEST" 3 & pids+=("$!")

for pid in "${pids[@]}"; do
  wait "$pid" || failures=$((failures + 1))
done

stop_resource_monitor || true

echo "AUDIO_OMNI_COSYVOICE3_GPU_DEMO_DONE failures=$failures log_dir=$LOG_DIR"
exit "$failures"
