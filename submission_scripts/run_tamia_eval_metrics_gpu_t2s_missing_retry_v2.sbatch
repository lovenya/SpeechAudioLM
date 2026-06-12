#!/usr/bin/env bash
#SBATCH --job-name=salm-metrics-t2s-missing-retry-v2
#SBATCH --account=aip-csubakan
#SBATCH --time=03:00:00
#SBATCH --cpus-per-task=48
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
export RESOURCE_INTERVAL="${RESOURCE_INTERVAL:-60}"

SCOPE_TAG="${SCOPE_TAG:?Submit with --export=ALL,SCOPE_TAG=<prepared-scope-tag>}"
base="$SALM_SCRATCH/results/cascaded_full_validation/ttas_val/cascaded"
scope="$SALM_SCRATCH/results/evaluation_scopes/ttas_val/$SCOPE_TAG/T2S.json"
out_root="$SALM_SCRATCH/results/evals/final_metrics_v2/$SCOPE_TAG"
cache="$SALM_SCRATCH/cache/metrics/eval_runtime"
log_root="$SALM_SCRATCH/logs/eval_packed/t2s_missing_retry_v2/${SLURM_JOB_ID:-manual}"
mkdir -p "$log_root" "$out_root" "$SALM_SCRATCH/logs/slurm"

test -f "$scope" || { echo "missing final T2S scope: $scope"; exit 2; }

source submission_scripts/lib_resource_monitor.sh
start_resource_monitor t2s_missing_retry_v2

pids=()
cleanup() {
  local status=$?
  trap - EXIT INT TERM
  if [ "${#pids[@]}" -gt 0 ]; then
    kill "${pids[@]}" 2>/dev/null || true
    wait "${pids[@]}" 2>/dev/null || true
  fi
  stop_resource_monitor
  exit "$status"
}
trap cleanup EXIT INT TERM

run_metric() {
  local lane="$1" metric="$2" gpu="$3"
  local manifest="$base/T2S/$lane/ttas/val/manifest.jsonl"
  local out="$out_root/T2S/$lane/$metric"
  local log="$log_root/T2S.${lane}.${metric}.log"
  if [ ! -f "$manifest" ]; then
    echo "missing manifest: $manifest" | tee "$log"
    return 2
  fi

  local extra_args=()
  if [ "${DRY_RUN:-0}" = "1" ]; then extra_args+=(--dry-run); fi
  if CUDA_VISIBLE_DEVICES="$gpu" python evaluate.py \
    --model cascaded --task T2S --dataset ttas --split val \
    --manifest "$manifest" --metrics "$metric" \
    --evaluation-mode final --common-sample-ids "$scope" \
    --metrics-config configs/eval_metrics.json \
    --output-root "$out" --cache-dir "$cache" --device cuda \
    "${extra_args[@]}" >"$log" 2>&1; then
    echo "PASS task=T2S lane=$lane metric=$metric" | tee -a "$log"
  else
    echo "FAIL task=T2S lane=$lane metric=$metric" | tee -a "$log"
    return 1
  fi
}

launch_metric() {
  run_metric "$1" "$2" "$3" &
  pids+=("$!")
}

# Retry only the pairs left incomplete when job 320913 reached its time limit.
# Each invocation has an independent output directory and writes immutable,
# timestamped evaluation artifacts under the existing final-metrics root.
launch_metric full_a2s_afn_llama_audio_omni utmos 0
launch_metric full_a2s_afn_llama_cosyvoice3 utmos 1
launch_metric full_a2s_afn_llama_f5_tts utmos 2
launch_metric full_a2s_afn_qwen32_audio_omni utmos 3
launch_metric full_a2s_afn_qwen32_cosyvoice3 utmos 0
launch_metric full_a2s_afn_qwen32_f5_tts utmos 1
launch_metric full_a2s_qwencap_llama_audio_omni utmos 2
launch_metric full_a2s_qwencap_llama_cosyvoice3 utmos 3
launch_metric full_a2s_qwencap_llama_f5_tts utmos 0
launch_metric full_a2s_qwencap_qwen32_audio_omni utmos 1
launch_metric full_a2s_qwencap_qwen32_cosyvoice3 utmos 2
launch_metric full_a2s_qwencap_qwen32_f5_tts utmos 3

launch_metric full_a2s_afn_qwen32_audio_omni dwer 0
launch_metric full_a2s_qwencap_llama_audio_omni dwer 1
launch_metric full_a2s_qwencap_llama_cosyvoice3 dwer 2
launch_metric full_a2s_qwencap_qwen32_audio_omni dwer 3
launch_metric full_a2s_qwencap_qwen32_cosyvoice3 dwer 0
launch_metric full_a2s_qwencap_qwen32_f5_tts dwer 1

launch_metric full_a2s_qwencap_qwen32_cosyvoice3 asr_wer 2

failures=0
for pid in "${pids[@]}"; do
  wait "$pid" || failures=$((failures + 1))
done

echo "t2s_missing_retry_v2_complete failures=$failures logs=$log_root outputs=$out_root"
exit "$failures"
