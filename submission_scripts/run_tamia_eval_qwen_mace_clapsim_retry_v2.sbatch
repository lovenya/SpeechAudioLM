#!/usr/bin/env bash
#SBATCH --job-name=salm-qwen-mace-clapsim-retry-v2
#SBATCH --account=aip-csubakan
#SBATCH --time=00:45:00
#SBATCH --cpus-per-task=24
#SBATCH --mem=0
#SBATCH --gpus=h100:4
#SBATCH --output=/scratch/l/lovenya/salm_migration_clean/logs/slurm/%x-%j.out

set -uo pipefail

cd /scratch/l/lovenya/SpeechAudioLM
module --force purge
module load StdEnv/2023 gcc cuda/12.6 opencv/4.13.0 python/3.12.4 scipy-stack java/11.0.22
unset JAVA_TOOL_OPTIONS
export AAC_METRICS_JAVA_PATH="$(command -v java)"
test -x "$AAC_METRICS_JAVA_PATH" || { echo "Java executable unavailable"; exit 1; }

export SALM_SCRATCH="${SALM_SCRATCH:-/scratch/l/lovenya/salm_migration_clean}"
export PYTHONNOUSERSITE=1 PYTHONDONTWRITEBYTECODE=1 TOKENIZERS_PARALLELISM=false
export HF_HOME="$SALM_SCRATCH/cache/metrics/huggingface"
export HF_HUB_CACHE="$HF_HOME" HUGGINGFACE_HUB_CACHE="$HF_HOME" TRANSFORMERS_CACHE="$HF_HOME"
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1
export TORCH_HOME="$SALM_SCRATCH/cache/metrics/torch"
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-1}"
export MKL_NUM_THREADS="${MKL_NUM_THREADS:-1}"
export NUMEXPR_NUM_THREADS="${NUMEXPR_NUM_THREADS:-1}"
export RESOURCE_INTERVAL="${RESOURCE_INTERVAL:-30}"

SCOPE_TAG="${SCOPE_TAG:?Submit with --export=ALL,SCOPE_TAG=<prepared-scope-tag>}"
manifest="$SALM_SCRATCH/results/cascaded_full_validation/ttas_val/cascaded/A2T/full_a2t_qwen3_omni_captioner/ttas/val/manifest.jsonl"
scope="$SALM_SCRATCH/results/evaluation_scopes/ttas_val/$SCOPE_TAG/A2T.json"
out_root="$SALM_SCRATCH/results/evals/final_metrics_v2/$SCOPE_TAG/A2T/full_a2t_qwen3_omni_captioner"
cache="$SALM_SCRATCH/cache/metrics/eval_runtime"
log_root="$SALM_SCRATCH/logs/eval_packed/qwen_mace_clapsim_retry_v2/${SLURM_JOB_ID:-manual}"
mkdir -p "$log_root" "$out_root" "$SALM_SCRATCH/logs/slurm"

test -f "$manifest" || { echo "missing full Qwen A2T manifest: $manifest"; exit 2; }
test -f "$scope" || { echo "missing final A2T scope: $scope"; exit 2; }

source submission_scripts/lib_resource_monitor.sh
start_resource_monitor qwen_mace_clapsim_retry_v2

run_metric() {
  local metric="$1" gpu="$2"
  local output="$out_root/$metric"
  local log="$log_root/$metric.log"

  CUDA_VISIBLE_DEVICES="$gpu" python evaluate.py \
    --model cascaded \
    --task A2T \
    --dataset ttas \
    --split val \
    --manifest "$manifest" \
    --metrics "$metric" \
    --evaluation-mode final \
    --common-sample-ids "$scope" \
    --metrics-config configs/eval_metrics.json \
    --output-root "$output" \
    --cache-dir "$cache" \
    --device cuda \
    >"$log" 2>&1
}

run_metric mace 0 &
mace_pid=$!
run_metric aac_clapsim 1 &
clapsim_pid=$!

wait "$mace_pid"
mace_status=$?
wait "$clapsim_pid"
clapsim_status=$?

stop_resource_monitor
echo "qwen_mace_clapsim_retry_v2_complete mace_status=$mace_status clapsim_status=$clapsim_status logs=$log_root outputs=$out_root"

if [ "$mace_status" -ne 0 ]; then
  exit "$mace_status"
fi
exit "$clapsim_status"
