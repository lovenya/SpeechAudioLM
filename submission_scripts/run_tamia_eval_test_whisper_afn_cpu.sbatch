#!/usr/bin/env bash
# Canonical test-split CPU evals for the confirmed minimal scope:
# Whisper S2T WER and Audio Flamingo Next A2T SPIDEr only.

#SBATCH --job-name=salm-test-w-afn-cpu
#SBATCH --account=aip-ravanelm
#SBATCH --time=02:00:00
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
RUN_ID="$(TZ=America/Toronto date +%Y%m%dT%H%M%S%z)"
LOG_DIR="$SALM_SCRATCH/logs/evals/test_whisper_afn_cpu/$RUN_ID"
RUNTIME_CONFIG="$LOG_DIR/eval_metrics.runtime.json"
WHISPER_MANIFEST="$SALM_SCRATCH/results/cascaded/stages/s2t/whisper/ttas/test/20260618_0001/manifest.jsonl"
AFN_MANIFEST="$SALM_SCRATCH/results/cascaded/stages/a2t/audio_flamingo_next/ttas/test/20260618_0001/manifest.jsonl"

mkdir -p "$LOG_DIR" "$SALM_SCRATCH/logs/resources" "$SALM_SCRATCH/logs/slurm"
test -x "$SHARED_PYTHON" || { echo "missing shared python: $SHARED_PYTHON" >&2; exit 2; }
test -f "$WHISPER_MANIFEST" || { echo "missing manifest: $WHISPER_MANIFEST" >&2; exit 2; }
test -f "$AFN_MANIFEST" || { echo "missing manifest: $AFN_MANIFEST" >&2; exit 2; }

python - <<PY
import json
import subprocess
from pathlib import Path

cfg = json.loads(Path("configs/eval_metrics.json").read_text(encoding="utf-8"))
cfg["metrics"]["spider"]["python"] = "$SHARED_PYTHON"
cfg["metrics"]["spider"]["java_max_memory"] = "96G"
Path("$RUNTIME_CONFIG").write_text(json.dumps(cfg, indent=2) + "\n", encoding="utf-8")

metadata = {
    "script": "submission_scripts/run_tamia_eval_test_whisper_afn_cpu.sbatch",
    "job_id": "${SLURM_JOB_ID:-manual}",
    "run_id": "$RUN_ID",
    "scope": "test_only_whisper_and_audio_flamingo_next",
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
start_resource_monitor "test_whisper_afn_cpu_${RUN_ID}"

run_eval() {
  local label="$1"
  local stage="$2"
  local metrics="$3"
  local manifest="$4"
  local log="$LOG_DIR/${label}.log"

  echo "START label=$label stage=$stage split=test metrics=$metrics evaluated_run_id=20260618_0001 device=cpu" | tee "$log"

  python evaluate.py \
    --model cascaded \
    --stage "$stage" \
    --evaluated-run-id 20260618_0001 \
    --dataset ttas \
    --split test \
    --manifest "$manifest" \
    --metrics "$metrics" \
    --metrics-config "$RUNTIME_CONFIG" \
    --cache-dir "$SALM_SCRATCH/cache/metrics/eval_runtime" \
    --device cpu >>"$log" 2>&1

  echo "PASS label=$label" | tee -a "$log"
}

failures=0
run_eval s2t_whisper_test_wer s2t wer "$WHISPER_MANIFEST" || failures=$((failures + 1))
run_eval a2t_afn_test_spider a2t spider "$AFN_MANIFEST" || failures=$((failures + 1))

stop_resource_monitor || true

echo "TEST_WHISPER_AFN_CPU_DONE failures=$failures log_dir=$LOG_DIR"
exit "$failures"
