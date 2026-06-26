#!/usr/bin/env bash
# Full test-split GEval text rerun after prompt-boundary fixes.

#SBATCH --job-name=geval-text-pfix
#SBATCH --account=aip-csubakan
#SBATCH --nodes=1
#SBATCH --gpus=h100:4
#SBATCH --cpus-per-task=10
#SBATCH --mem=128G
#SBATCH --time=05:00:00
#SBATCH --output=/scratch/l/lovenya/salm_migration_clean/logs/slurm/%x-%j.out

set -euo pipefail

cd /scratch/l/lovenya/SpeechAudioLM

module --force purge
module load StdEnv/2023 gcc cuda python scipy-stack

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
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0,1,2,3}"
export VLLM_USE_DEEP_GEMM=0
export VLLM_MOE_USE_DEEP_GEMM=0
export VLLM_DEEP_GEMM_WARMUP=skip

source /home/l/lovenya/.venvs/speechaudiolm/.venv-eval-prometheus-vllm-py312/bin/activate

SCOPE="${SCOPE:-dataset}"
DIRECTION="${DIRECTION:?Submit with --export=ALL,SCOPE=<dataset|generated>,DIRECTION=<a2s|s2a>[,RUN_GROUP=...,MANIFEST_ROOT=...]}"
METRICS="${METRICS:-geval_text_plausibility}"
TEMPORARY="${TEMPORARY:-1}"
MAX_SAMPLES="${MAX_SAMPLES:-}"

case "$SCOPE" in
  dataset)
    RUN_GROUP="${RUN_GROUP:?dataset scope requires RUN_GROUP from prepare_tamia_geval_promptfix_dataset_pair_manifests.sh}"
    MANIFEST_ROOT="${MANIFEST_ROOT:?dataset scope requires MANIFEST_ROOT from prepare_tamia_geval_promptfix_dataset_pair_manifests.sh}"
    EVALUATED_RUN_ID="${EVALUATED_RUN_ID:-$RUN_GROUP}"
    case "$DIRECTION" in
      a2s) MANIFEST="$MANIFEST_ROOT/t2t_a2s.jsonl" ;;
      s2a) MANIFEST="$MANIFEST_ROOT/t2t_s2a.jsonl" ;;
      *) echo "error: DIRECTION must be a2s or s2a, got: $DIRECTION" >&2; exit 2 ;;
    esac
    ;;
  generated)
    EVALUATED_RUN_ID="${EVALUATED_RUN_ID:-20260618_0001}"
    case "$DIRECTION" in
      a2s) MANIFEST="${MANIFEST:-$SALM_RESULTS_ROOT/cascaded/stages/t2t/a2s/llama3_3_70b_instruct/from-a2t/audio_flamingo_next/ttas/test/$EVALUATED_RUN_ID/manifest.jsonl}" ;;
      s2a) MANIFEST="${MANIFEST:-$SALM_RESULTS_ROOT/cascaded/stages/t2t/s2a/llama3_3_70b_instruct/from-s2t/whisper/ttas/test/$EVALUATED_RUN_ID/manifest.jsonl}" ;;
      *) echo "error: DIRECTION must be a2s or s2a, got: $DIRECTION" >&2; exit 2 ;;
    esac
    RUN_GROUP="${RUN_GROUP:-generated_geval_text_promptfix_$EVALUATED_RUN_ID}"
    ;;
  *)
    echo "error: SCOPE must be dataset or generated, got: $SCOPE" >&2
    exit 2
    ;;
esac

test -s "$MANIFEST" || { echo "error: missing manifest: $MANIFEST" >&2; exit 2; }

JOB_KEY="${SLURM_JOB_ID:-manual}"
LOG_ROOT="$SALM_SCRATCH/logs/evals/$RUN_GROUP/text_${SCOPE}_${DIRECTION}_${JOB_KEY}"
MONITOR_DIR="$SALM_SCRATCH/logs/resources/$RUN_GROUP/text_${SCOPE}_${DIRECTION}_${JOB_KEY}"
METADATA="$LOG_ROOT/job_metadata.json"
mkdir -p "$LOG_ROOT" "$MONITOR_DIR" "$SALM_SCRATCH/logs/slurm"
export SCOPE RUN_GROUP DIRECTION MANIFEST METRICS EVALUATED_RUN_ID METADATA

python - <<'PY'
import json
import os
import subprocess
from pathlib import Path

metadata = {
    "script": "submission_scripts/run_tamia_geval_text_promptfix_full.sbatch",
    "job_id": os.environ.get("SLURM_JOB_ID"),
    "scope": os.environ["SCOPE"],
    "run_group": os.environ["RUN_GROUP"],
    "direction": os.environ["DIRECTION"],
    "manifest": os.environ["MANIFEST"],
    "metrics": os.environ["METRICS"],
    "evaluated_run_id": os.environ["EVALUATED_RUN_ID"],
    "repo": str(Path.cwd()),
    "git_commit": subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip(),
    "git_dirty": bool(subprocess.check_output(["git", "status", "--porcelain"], text=True).strip()),
}
Path(os.environ["METADATA"]).write_text(json.dumps(metadata, indent=2, sort_keys=True) + "\n", encoding="utf-8")
print(f"metadata={os.environ['METADATA']}")
PY

python scripts/resource_monitor.py \
  --interval "${RESOURCE_INTERVAL:-60}" \
  --label "${RUN_GROUP}_text_${SCOPE}_${DIRECTION}_${JOB_KEY}" \
  --log "$MONITOR_DIR/resource_samples.jsonl" \
  --summary "$MONITOR_DIR/resource_summary.json" &
MONITOR_PID=$!

cleanup() {
  local status=$?
  kill -TERM "$MONITOR_PID" 2>/dev/null || true
  wait "$MONITOR_PID" 2>/dev/null || true
  echo "scope=$SCOPE"
  echo "direction=$DIRECTION"
  echo "status=$status"
  echo "manifest=$MANIFEST"
  echo "resource_summary=$MONITOR_DIR/resource_summary.json"
  exit "$status"
}
trap cleanup EXIT INT TERM

args=(
  evaluate.py
  --model cascaded
  --stage t2t
  --direction "$DIRECTION"
  --evaluated-run-id "$EVALUATED_RUN_ID"
  --dataset ttas
  --split test
  --manifest "$MANIFEST"
  --metrics "$METRICS"
  --metrics-config configs/eval_metrics.json
  --device cuda
)

if [ "$TEMPORARY" = "1" ]; then
  args+=(--temporary)
fi
if [ -n "$MAX_SAMPLES" ]; then
  args+=(--max-samples "$MAX_SAMPLES")
fi

printf '%q ' python "${args[@]}" | tee "$LOG_ROOT/command.txt"
printf '\n' | tee -a "$LOG_ROOT/command.txt"
python "${args[@]}" 2>&1 | tee "$LOG_ROOT/evaluate.log"
