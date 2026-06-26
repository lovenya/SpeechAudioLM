#!/usr/bin/env bash
# Sharded test-split GEval audio rerun after prompt-boundary and clip-label fixes.

#SBATCH --job-name=geval-audio-pfix
#SBATCH --account=aip-csubakan
#SBATCH --nodes=1
#SBATCH --gpus=h100:4
#SBATCH --cpus-per-task=16
#SBATCH --mem=128G
#SBATCH --time=03:30:00
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
export PYTHONUNBUFFERED=1
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-2}"
export MKL_NUM_THREADS="${MKL_NUM_THREADS:-2}"
export NUMEXPR_NUM_THREADS="${NUMEXPR_NUM_THREADS:-2}"
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"

source "$SALM_SCRATCH/.venvs/models/.venv-qwen3-omni/bin/activate"

SCOPE="${SCOPE:-dataset}"
RUN_GROUP="${RUN_GROUP:-geval_audio_promptfix_$(TZ=America/Toronto date +%Y%m%d_%H%M)}"
METRIC="${METRIC:-geval_audio_plausibility}"
TEMPORARY="${TEMPORARY:-1}"
ARRAY_TASK_ID="${SLURM_ARRAY_TASK_ID:-0}"
JOB_KEY="${SLURM_ARRAY_JOB_ID:-manual}_${ARRAY_TASK_ID}"

case "$SCOPE" in
  dataset)
    MANIFEST_ROOT="${MANIFEST_ROOT:?dataset scope requires MANIFEST_ROOT from prepare_tamia_geval_promptfix_dataset_pair_manifests.sh}"
    A2S_MANIFEST="${A2S_MANIFEST:-$MANIFEST_ROOT/t2s_a2s.jsonl}"
    S2A_MANIFEST="${S2A_MANIFEST:-$MANIFEST_ROOT/t2a_s2a.jsonl}"
    ;;
  generated)
    T2S_RUN_ID="${T2S_RUN_ID:-20260618_1211}"
    T2A_RUN_ID="${T2A_RUN_ID:-20260618_1211}"
    A2S_MANIFEST="${A2S_MANIFEST:-$SALM_RESULTS_ROOT/cascaded/stages/t2s/cosyvoice3/from-t2t/llama3_3_70b_instruct/ttas/test/$T2S_RUN_ID/manifest.jsonl}"
    S2A_MANIFEST="${S2A_MANIFEST:-$SALM_RESULTS_ROOT/cascaded/stages/t2a/audio_omni/from-t2t/llama3_3_70b_instruct/ttas/test/$T2A_RUN_ID/manifest.jsonl}"
    ;;
  *)
    echo "error: SCOPE must be dataset or generated, got: $SCOPE" >&2
    exit 2
    ;;
esac

test -s "$A2S_MANIFEST" || { echo "missing A2S manifest: $A2S_MANIFEST" >&2; exit 2; }
test -s "$S2A_MANIFEST" || { echo "missing S2A manifest: $S2A_MANIFEST" >&2; exit 2; }

LOG_ROOT="$SALM_SCRATCH/logs/evals/$RUN_GROUP/audio_${SCOPE}_array_${JOB_KEY}"
MONITOR_DIR="$SALM_SCRATCH/logs/resources/$RUN_GROUP/audio_${SCOPE}_array_${JOB_KEY}"
METADATA="$LOG_ROOT/job_metadata.json"
mkdir -p "$LOG_ROOT" "$MONITOR_DIR" "$SALM_SCRATCH/logs/slurm"
export SCOPE RUN_GROUP METRIC A2S_MANIFEST S2A_MANIFEST METADATA

python - <<'PY'
import json
import os
import subprocess
from pathlib import Path

metadata = {
    "script": "submission_scripts/run_tamia_geval_audio_promptfix_array.sbatch",
    "job_id": os.environ.get("SLURM_JOB_ID"),
    "array_job_id": os.environ.get("SLURM_ARRAY_JOB_ID"),
    "array_task_id": os.environ.get("SLURM_ARRAY_TASK_ID"),
    "scope": os.environ["SCOPE"],
    "run_group": os.environ["RUN_GROUP"],
    "metric": os.environ["METRIC"],
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
  --label "${RUN_GROUP}_audio_${SCOPE}_${JOB_KEY}" \
  --log "$MONITOR_DIR/resource_samples.jsonl" \
  --summary "$MONITOR_DIR/resource_summary.json" &
MONITOR_PID=$!

cleanup() {
  local status=$?
  kill -TERM "$MONITOR_PID" 2>/dev/null || true
  wait "$MONITOR_PID" 2>/dev/null || true
  echo "scope=$SCOPE"
  echo "run_group=$RUN_GROUP"
  echo "array_task=$ARRAY_TASK_ID"
  echo "status=$status"
  echo "resource_summary=$MONITOR_DIR/resource_summary.json"
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

metric_stage() {
  case "$1" in
    a2s) echo t2s ;;
    s2a) echo t2a ;;
    *) return 2 ;;
  esac
}

run_eval() {
  local direction="$1"
  local manifest="$2"
  local shard="$3"
  local gpu="$4"
  local stage start count evaluated_run_id log
  stage="$(metric_stage "$direction")"
  start="$(shard_start "$shard")"
  count="$(shard_count "$shard")"
  evaluated_run_id="${RUN_GROUP}_${SCOPE}_${direction}_shard_$(printf '%03d' "$shard")"
  log="$LOG_ROOT/${direction}_${METRIC}_shard_$(printf '%03d' "$shard").log"
  echo "launch scope=$SCOPE direction=$direction stage=$stage shard=$shard start=$start count=$count gpu=$gpu evaluated_run_id=$evaluated_run_id log=$log"
  local args=(
    evaluate.py
    --model cascaded
    --stage "$stage"
    --direction "$direction"
    --evaluated-run-id "$evaluated_run_id"
    --dataset ttas
    --split test
    --manifest "$manifest"
    --metrics "$METRIC"
    --metrics-config configs/eval_metrics.json
    --start-index "$start"
    --max-samples "$count"
    --device cuda
  )
  if [ "$TEMPORARY" = "1" ]; then
    args+=(--temporary)
  fi
  printf '%q ' python "${args[@]}" > "$LOG_ROOT/${direction}_shard_$(printf '%03d' "$shard").command.txt"
  printf '\n' >> "$LOG_ROOT/${direction}_shard_$(printf '%03d' "$shard").command.txt"
  CUDA_VISIBLE_DEVICES="$gpu" python "${args[@]}" > "$log" 2>&1
}

first_shard=$(( ARRAY_TASK_ID * 2 ))
second_shard=$(( first_shard + 1 ))

run_eval a2s "$A2S_MANIFEST" "$first_shard" 0 & pid_a2s_0=$!
sleep 2
run_eval a2s "$A2S_MANIFEST" "$second_shard" 1 & pid_a2s_1=$!
sleep 2
run_eval s2a "$S2A_MANIFEST" "$first_shard" 2 & pid_s2a_0=$!
sleep 2
run_eval s2a "$S2A_MANIFEST" "$second_shard" 3 & pid_s2a_1=$!

failures=0
wait "$pid_a2s_0" || failures=$((failures + 1))
wait "$pid_a2s_1" || failures=$((failures + 1))
wait "$pid_s2a_0" || failures=$((failures + 1))
wait "$pid_s2a_1" || failures=$((failures + 1))

echo "run_group=$RUN_GROUP"
echo "scope=$SCOPE"
echo "array_task=$ARRAY_TASK_ID"
echo "failures=$failures"
exit "$failures"
