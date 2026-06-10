#!/usr/bin/env bash
#SBATCH --job-name=salm-a2t-qwen-debug
#SBATCH --account=aip-csubakan
#SBATCH --time=00:25:00
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
export HF_HUB_CACHE="$HF_HOME" TRANSFORMERS_CACHE="$HF_HOME"
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1
export TORCH_HOME="$SALM_SCRATCH/cache/metrics/torch"
export RESOURCE_INTERVAL="${RESOURCE_INTERVAL:-30}"

lane=full_a2t_qwen3_omni_captioner
manifest="$SALM_SCRATCH/results/cascaded_full_validation/ttas_val/cascaded/A2T/$lane/ttas/val/manifest.jsonl"
log_root="$SALM_SCRATCH/logs/eval_packed/final_validation/a2t_qwen_debug/${SLURM_JOB_ID}"
out_root="$SALM_SCRATCH/results/evals/final_validation_debug/A2T/$lane"
cache="$SALM_SCRATCH/cache/metrics/eval_runtime"
mkdir -p "$log_root"

source submission_scripts/lib_resource_monitor.sh
start_resource_monitor final_a2t_qwen_debug

run_eval() {
  local label="$1"
  shift
  if "$@" >"$log_root/$label.log" 2>&1; then
    echo "PASS $label" | tee -a "$log_root/$label.log"
  else
    echo "FAIL $label" | tee -a "$log_root/$label.log"
    return 1
  fi
}

pids=()
for shard in 0 1 2 3; do
  gpu="$shard"
  start=$((shard * 600))
  (
    shard_failures=0
    for metric in mace aac_clapsim; do
      run_eval "${lane}.shard_${shard}.${metric}" env CUDA_VISIBLE_DEVICES="$gpu" python evaluate.py \
        --model cascaded --task A2T --dataset ttas --split val --manifest "$manifest" \
        --metrics "$metric" --metrics-config configs/eval_metrics.json \
        --output-root "$out_root/shard_${shard}/$metric" \
        --cache-dir "$cache" --device cuda --start-index "$start" --max-samples 600 \
        || shard_failures=$((shard_failures + 1))
    done
    exit "$shard_failures"
  ) &
  pids+=("$!")
done

failures=0
for pid in "${pids[@]}"; do
  wait "$pid" || failures=$((failures + 1))
done
stop_resource_monitor
echo "a2t_qwen_debug_complete failures=$failures logs=$log_root"
exit "$failures"
