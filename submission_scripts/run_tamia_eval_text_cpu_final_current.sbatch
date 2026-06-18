#!/usr/bin/env bash
# Canonical non-G-Eval CPU metric submission for current eval CLI:
# S2T WER and A2T SPIDEr over full validation manifests.

#SBATCH --job-name=salm-text-cpu-eval
#SBATCH --account=aip-ravanelm
#SBATCH --time=04:00:00
#SBATCH --cpus-per-task=64
#SBATCH --mem=128G
#SBATCH --output=/scratch/l/lovenya/salm_migration_clean/logs/slurm/%x-%j.out

set -euo pipefail

cd /scratch/l/lovenya/SpeechAudioLM

module --force purge
module load StdEnv/2023 gcc opencv/4.13.0 java/11.0.22 python/3.12.4 scipy-stack
unset JAVA_TOOL_OPTIONS
export AAC_METRICS_JAVA_PATH="$(command -v java)"
test -x "$AAC_METRICS_JAVA_PATH" || { echo "Java executable unavailable" >&2; exit 2; }

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
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-4}"
export MKL_NUM_THREADS="${MKL_NUM_THREADS:-4}"
export NUMEXPR_NUM_THREADS="${NUMEXPR_NUM_THREADS:-4}"
export RESOURCE_INTERVAL="${RESOURCE_INTERVAL:-60}"

SHARED_PYTHON="${SHARED_PYTHON:-/home/l/lovenya/.venvs/speechaudiolm/.venv-eval-metrics-shared-py312/bin/python}"
BASE="$SALM_SCRATCH/results/cascaded_full_validation/ttas_val/cascaded"
RUN_ID="$(TZ=America/Toronto date +%Y%m%dT%H%M%S%z)"
LOG_DIR="$SALM_SCRATCH/logs/evals/text_cpu_final_current/$RUN_ID"
RUNTIME_CONFIG="$LOG_DIR/eval_metrics.runtime.json"

mkdir -p "$LOG_DIR" "$SALM_SCRATCH/logs/resources" "$SALM_SCRATCH/logs/slurm"
test -x "$SHARED_PYTHON" || { echo "missing shared python: $SHARED_PYTHON" >&2; exit 2; }

python - <<PY
import json
import subprocess
from pathlib import Path

cfg = json.loads(Path("configs/eval_metrics.json").read_text(encoding="utf-8"))
cfg["metrics"]["spider"]["python"] = "$SHARED_PYTHON"
cfg["metrics"]["spider"]["java_max_memory"] = "96G"
Path("$RUNTIME_CONFIG").write_text(json.dumps(cfg, indent=2) + "\n", encoding="utf-8")

metadata = {
    "script": "submission_scripts/run_tamia_eval_text_cpu_final_current.sbatch",
    "job_id": "${SLURM_JOB_ID:-manual}",
    "run_id": "$RUN_ID",
    "repo": str(Path.cwd()),
    "git_commit": subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip(),
    "git_dirty": bool(subprocess.check_output(["git", "status", "--porcelain"], text=True).strip()),
    "metrics_config": "$RUNTIME_CONFIG",
    "g_eval_excluded": True,
}
Path("$LOG_DIR/job_metadata.json").write_text(json.dumps(metadata, indent=2) + "\n", encoding="utf-8")
print("runtime_config=$RUNTIME_CONFIG")
print("job_metadata=$LOG_DIR/job_metadata.json")
PY

source submission_scripts/lib_resource_monitor.sh
start_resource_monitor "text_cpu_final_current_${RUN_ID}"

run_eval() {
  local label="$1"
  local stage="$2"
  local metrics="$3"
  local manifest="$4"
  local device="$5"
  local evaluated_run_id="ttas_val_${label}"
  local log="$LOG_DIR/${label}.log"

  echo "START label=$label stage=$stage metrics=$metrics evaluated_run_id=$evaluated_run_id device=$device" | tee "$log"

  python evaluate.py \
    --model cascaded \
    --stage "$stage" \
    --evaluated-run-id "$evaluated_run_id" \
    --dataset ttas \
    --split val \
    --manifest "$manifest" \
    --metrics "$metrics" \
    --metrics-config "$RUNTIME_CONFIG" \
    --cache-dir "$SALM_SCRATCH/cache/metrics/eval_runtime" \
    --device "$device" >>"$log" 2>&1

  echo "PASS label=$label" | tee -a "$log"
}

failures=0
wer_pids=()

run_eval s2t_whisper s2t wer "$BASE/S2T/full_s2t_whisper/ttas/val/manifest.jsonl" cpu & wer_pids+=("$!")
run_eval s2t_qwen3_asr s2t wer "$BASE/S2T/full_s2t_qwen3_asr/ttas/val/manifest.jsonl" cpu & wer_pids+=("$!")

for pid in "${wer_pids[@]}"; do
  wait "$pid" || failures=$((failures + 1))
done

run_eval a2t_afn_spider a2t spider "$BASE/A2T/full_a2t_audio_flamingo_next/ttas/val/manifest.jsonl" cpu || failures=$((failures + 1))
run_eval a2t_qwen3_captioner_spider a2t spider "$BASE/A2T/full_a2t_qwen3_omni_captioner/ttas/val/manifest.jsonl" cpu || failures=$((failures + 1))

stop_resource_monitor || true

echo "TEXT_CPU_FINAL_CURRENT_DONE failures=$failures log_dir=$LOG_DIR"
exit "$failures"
