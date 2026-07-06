#!/usr/bin/env bash
# Packed canonical v6 audio GEval plausibility task runner.

#SBATCH --job-name=geval-audio-v6p
#SBATCH --account=aip-csubakan
#SBATCH --nodes=1
#SBATCH --gpus=h100:4
#SBATCH --cpus-per-task=16
#SBATCH --mem=128G
#SBATCH --time=02:00:00
#SBATCH --output=/scratch/l/lovenya/salm_migration_clean/logs/slurm/%x-%A_%a.out

set -euo pipefail

cd /scratch/l/lovenya/SpeechAudioLM

module --force purge
module load StdEnv/2023 gcc cuda/12.6 python/3.12.4

export SALM_SCRATCH="${SALM_SCRATCH:-/scratch/l/lovenya/salm_migration_clean}"
export SALM_RESULTS_ROOT="$SALM_SCRATCH/results"
export SALM_TEMPORARY_RESULTS_ROOT="$SALM_SCRATCH/temporary/results"
export SALM_CACHE_ROOT="$SALM_SCRATCH/cache"
export SALM_MODEL_ROOT="$SALM_SCRATCH/models"
export SALM_ENVIRONMENT_ROOT="$SALM_SCRATCH/.venvs"
export SALM_ENVIRONMENTS_ROOT="$SALM_SCRATCH/.venvs"
export SALM_LOGS_ROOT="$SALM_SCRATCH/logs"
export SALM_TEMPORARY_LOGS_ROOT="$SALM_SCRATCH/temporary/logs"
export HF_HOME="$SALM_SCRATCH/cache/metrics/huggingface"
export HF_HUB_CACHE="$HF_HOME"
export HUGGINGFACE_HUB_CACHE="$HF_HOME"
export TRANSFORMERS_CACHE="$HF_HOME"
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export TOKENIZERS_PARALLELISM=false
export PYTHONNOUSERSITE=1
export PYTHONDONTWRITEBYTECODE=1
export PYTHONUNBUFFERED=1
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-2}"
export MKL_NUM_THREADS="${MKL_NUM_THREADS:-2}"
export NUMEXPR_NUM_THREADS="${NUMEXPR_NUM_THREADS:-2}"
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"

source "$SALM_SCRATCH/.venvs/models/.venv-qwen3-omni/bin/activate"

TASK_PLAN="${TASK_PLAN:?Submit with --export=ALL,TASK_PLAN=/path/to/task_plan.json,RUN_GROUP=<group>}"
RUN_GROUP="${RUN_GROUP:?Submit with --export=ALL,TASK_PLAN=/path/to/task_plan.json,RUN_GROUP=<group>}"
TASK_INDEX="${SLURM_ARRAY_TASK_ID:-0}"
JOB_KEY="${SLURM_ARRAY_JOB_ID:-manual}_${TASK_INDEX}"
LOG_ROOT="$SALM_SCRATCH/logs/evals/$RUN_GROUP/packed_${JOB_KEY}"
MONITOR_DIR="$SALM_SCRATCH/logs/resources/$RUN_GROUP/packed_${JOB_KEY}"
METADATA="$LOG_ROOT/job_metadata.json"
mkdir -p "$LOG_ROOT" "$MONITOR_DIR" "$SALM_SCRATCH/logs/slurm"
export RUN_GROUP TASK_PLAN METADATA

python - <<'PY_METADATA'
import json
import os
import subprocess
from pathlib import Path
metadata = {
    "script": "submission_scripts/run_tamia_geval_audio_v6_plausibility_packed.sbatch",
    "job_id": os.environ.get("SLURM_JOB_ID"),
    "array_job_id": os.environ.get("SLURM_ARRAY_JOB_ID"),
    "array_task_id": os.environ.get("SLURM_ARRAY_TASK_ID"),
    "run_group": os.environ["RUN_GROUP"],
    "task_plan": os.environ["TASK_PLAN"],
    "repo": str(Path.cwd()),
    "git_commit": subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip(),
    "git_dirty": bool(subprocess.check_output(["git", "status", "--porcelain"], text=True).strip()),
}
Path(os.environ["METADATA"]).write_text(json.dumps(metadata, indent=2, sort_keys=True) + "\n", encoding="utf-8")
print(f"metadata={os.environ['METADATA']}")
PY_METADATA

python scripts/resource_monitor.py \
  --interval "${RESOURCE_INTERVAL:-60}" \
  --label "${RUN_GROUP}_${JOB_KEY}" \
  --log "$MONITOR_DIR/resource_samples.jsonl" \
  --summary "$MONITOR_DIR/resource_summary.json" &
MONITOR_PID=$!

cleanup() {
  local status=$?
  kill -TERM "$MONITOR_PID" 2>/dev/null || true
  wait "$MONITOR_PID" 2>/dev/null || true
  echo "run_group=$RUN_GROUP"
  echo "task_index=$TASK_INDEX"
  echo "status=$status"
  echo "resource_summary=$MONITOR_DIR/resource_summary.json"
  exit "$status"
}
trap cleanup EXIT INT TERM

python scripts/run_geval_audio_v6_plausibility_task.py \
  --task-plan "$TASK_PLAN" \
  --task-index "$TASK_INDEX" \
  --log-root "$LOG_ROOT" \
  --repo-root /scratch/l/lovenya/SpeechAudioLM \
  --output-root "$SALM_RESULTS_ROOT/evals" \
  --cache-dir "$SALM_CACHE_ROOT/evaluations" \
  --device cuda
