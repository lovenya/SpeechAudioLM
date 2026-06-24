#!/usr/bin/env bash
# Full test-split dataset-pair GEval audio plausibility shards.

#SBATCH --job-name=salm-geval-audio-p
#SBATCH --account=aip-csubakan
#SBATCH --nodes=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=128G
#SBATCH --gpus=h100:4
#SBATCH --time=02:20:00
#SBATCH --array=0-4
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
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-2}"
export MKL_NUM_THREADS="${MKL_NUM_THREADS:-2}"
export NUMEXPR_NUM_THREADS="${NUMEXPR_NUM_THREADS:-2}"
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"

RUN_GROUP="${RUN_GROUP:?Submit with --export=ALL,RUN_GROUP=<group>,MANIFEST_ROOT=<manifest-root>}"
MANIFEST_ROOT="${MANIFEST_ROOT:?Submit with --export=ALL,RUN_GROUP=<group>,MANIFEST_ROOT=<manifest-root>}"
ARRAY_TASK_ID="${SLURM_ARRAY_TASK_ID:-0}"
JOB_KEY="${SLURM_ARRAY_JOB_ID:-manual}_${ARRAY_TASK_ID}"
LOG_ROOT="$SALM_SCRATCH/logs/evals/$RUN_GROUP/array_${JOB_KEY}"
MONITOR_DIR="$SALM_SCRATCH/logs/resources/$RUN_GROUP/array_${JOB_KEY}"
METADATA="$LOG_ROOT/job_metadata.json"

A2S_MANIFEST="$MANIFEST_ROOT/t2s_a2s.jsonl"
S2A_MANIFEST="$MANIFEST_ROOT/t2a_s2a.jsonl"
export RUN_GROUP MANIFEST_ROOT A2S_MANIFEST S2A_MANIFEST METADATA

mkdir -p "$LOG_ROOT" "$MONITOR_DIR" "$SALM_SCRATCH/logs/slurm"
test -s "$A2S_MANIFEST" || { echo "missing A2S manifest: $A2S_MANIFEST" >&2; exit 2; }
test -s "$S2A_MANIFEST" || { echo "missing S2A manifest: $S2A_MANIFEST" >&2; exit 2; }

python - <<'PY'
import json
import os
import subprocess
from pathlib import Path

metadata = {
    "script": "submission_scripts/run_tamia_dataset_pair_geval_audio_plausibility_array.sbatch",
    "job_id": os.environ.get("SLURM_JOB_ID"),
    "array_job_id": os.environ.get("SLURM_ARRAY_JOB_ID"),
    "array_task_id": os.environ.get("SLURM_ARRAY_TASK_ID"),
    "run_group": os.environ["RUN_GROUP"],
    "manifest_root": os.environ["MANIFEST_ROOT"],
    "a2s_manifest": os.environ["A2S_MANIFEST"],
    "s2a_manifest": os.environ["S2A_MANIFEST"],
    "repo": str(Path.cwd()),
    "git_commit": subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip(),
    "git_dirty": bool(subprocess.check_output(["git", "status", "--porcelain"], text=True).strip()),
}
Path(os.environ["METADATA"]).write_text(json.dumps(metadata, indent=2, sort_keys=True) + "\n", encoding="utf-8")
print(f"metadata={os.environ['METADATA']}")
PY

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
  exit "$status"
}
trap cleanup EXIT INT TERM

shard_start() {
  echo $(( $1 * 228 ))
}

shard_count() {
  if [ "$1" -eq 9 ]; then
    echo 225
  else
    echo 228
  fi
}

run_eval() {
  local direction="$1"
  local stage="$2"
  local manifest="$3"
  local shard="$4"
  local gpu="$5"
  local start count evaluated_run_id log
  start="$(shard_start "$shard")"
  count="$(shard_count "$shard")"
  evaluated_run_id="${RUN_GROUP}_${direction}_shard_$(printf '%03d' "$shard")"
  log="$LOG_ROOT/${direction}_plausibility_shard_$(printf '%03d' "$shard").log"
  echo "launch direction=$direction shard=$shard start=$start count=$count gpu=$gpu evaluated_run_id=$evaluated_run_id log=$log"
  CUDA_VISIBLE_DEVICES="$gpu" python evaluate.py \
    --model cascaded \
    --stage "$stage" \
    --direction "$direction" \
    --evaluated-run-id "$evaluated_run_id" \
    --dataset ttas \
    --split test \
    --manifest "$manifest" \
    --metrics geval_audio_plausibility \
    --start-index "$start" \
    --max-samples "$count" \
    --temporary \
    --device cuda \
    > "$log" 2>&1
}

first_shard=$(( ARRAY_TASK_ID * 2 ))
second_shard=$(( first_shard + 1 ))

run_eval a2s t2s "$A2S_MANIFEST" "$first_shard" 0 & pid_a2s_0=$!
sleep 2
run_eval a2s t2s "$A2S_MANIFEST" "$second_shard" 1 & pid_a2s_1=$!
sleep 2
run_eval s2a t2a "$S2A_MANIFEST" "$first_shard" 2 & pid_s2a_0=$!
sleep 2
run_eval s2a t2a "$S2A_MANIFEST" "$second_shard" 3 & pid_s2a_1=$!

failures=0
wait "$pid_a2s_0" || failures=$((failures + 1))
wait "$pid_a2s_1" || failures=$((failures + 1))
wait "$pid_s2a_0" || failures=$((failures + 1))
wait "$pid_s2a_1" || failures=$((failures + 1))

echo "run_group=$RUN_GROUP"
echo "array_task=$ARRAY_TASK_ID"
echo "failures=$failures"
exit "$failures"
