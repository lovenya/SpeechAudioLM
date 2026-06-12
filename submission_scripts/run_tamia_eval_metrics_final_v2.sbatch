#!/usr/bin/env bash
#SBATCH --job-name=salm-metrics-v2
#SBATCH --account=aip-csubakan
#SBATCH --array=0-26%3
#SBATCH --time=08:00:00
#SBATCH --cpus-per-task=48
#SBATCH --mem=0
#SBATCH --gpus=h100:4
#SBATCH --output=/scratch/l/lovenya/salm_migration_clean/logs/slurm/%x-%A_%a.out

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
export RESOURCE_INTERVAL="${RESOURCE_INTERVAL:-60}"

SCOPE_TAG="${SCOPE_TAG:?Submit with --export=ALL,SCOPE_TAG=<prepared-scope-tag>}"
base="$SALM_SCRATCH/results/cascaded_full_validation/ttas_val/cascaded"
scope_root="$SALM_SCRATCH/results/evaluation_scopes/ttas_val/$SCOPE_TAG"
out_root="$SALM_SCRATCH/results/evals/final_metrics_v2/$SCOPE_TAG"
cache="$SALM_SCRATCH/cache/metrics/eval_runtime"
log_root="$SALM_SCRATCH/logs/eval_packed/final_metrics_v2/${SLURM_ARRAY_JOB_ID}_${SLURM_ARRAY_TASK_ID}"
mkdir -p "$log_root" "$out_root" "$SALM_SCRATCH/logs/slurm"

source submission_scripts/lib_resource_monitor.sh
start_resource_monitor "final_metrics_v2_${SLURM_ARRAY_TASK_ID}"

run_metric() {
  local task="$1" lane="$2" metric="$3" gpu="$4"
  local manifest="$base/$task/$lane/ttas/val/manifest.jsonl"
  local scope="$scope_root/$task.json"
  local out="$out_root/$task/$lane/$metric"
  local log="$log_root/${task}.${lane}.${metric}.log"
  if [ ! -f "$manifest" ]; then echo "missing manifest: $manifest" | tee "$log"; return 2; fi
  if [ ! -f "$scope" ]; then echo "missing final scope: $scope" | tee "$log"; return 2; fi
  local extra_args=()
  if [ "${DRY_RUN:-0}" = "1" ]; then extra_args+=(--dry-run); fi
  if CUDA_VISIBLE_DEVICES="$gpu" python evaluate.py \
    --model cascaded --task "$task" --dataset ttas --split val \
    --manifest "$manifest" --metrics "$metric" \
    --evaluation-mode final --common-sample-ids "$scope" \
    --metrics-config configs/eval_metrics.json \
    --output-root "$out" --cache-dir "$cache" --device cuda \
    "${extra_args[@]}" >"$log" 2>&1; then
    echo "PASS task=$task lane=$lane metric=$metric" | tee -a "$log"
  else
    echo "FAIL task=$task lane=$lane metric=$metric" | tee -a "$log"
    return 1
  fi
}

run_parallel() {
  local failures=0
  local pids=()
  while IFS='|' read -r task lane metric gpu; do
    [ -z "$task" ] && continue
    run_metric "$task" "$lane" "$metric" "$gpu" &
    pids+=("$!")
  done
  for pid in "${pids[@]}"; do wait "$pid" || failures=$((failures + 1)); done
  return "$failures"
}

index="$SLURM_ARRAY_TASK_ID"
failures=0

if [ "$index" -lt 2 ]; then
  lanes=(full_a2t_audio_flamingo_next full_a2t_qwen3_omni_captioner)
  lane="${lanes[$index]}"
  run_parallel <<EOF || failures=$?
A2T|$lane|mace|0
A2T|$lane|spider|1
A2T|$lane|aac_clapsim|2
A2T|$lane|fense|3
EOF
elif [ "$index" -lt 14 ]; then
  lanes=(
    full_s2a_qwen3_asr_llama_audio_omni full_s2a_qwen3_asr_llama_audiogen full_s2a_qwen3_asr_llama_tangoflux
    full_s2a_qwen3_asr_qwen32_audio_omni full_s2a_qwen3_asr_qwen32_audiogen full_s2a_qwen3_asr_qwen32_tangoflux
    full_s2a_whisper_llama_audio_omni full_s2a_whisper_llama_audiogen full_s2a_whisper_llama_tangoflux
    full_s2a_whisper_qwen32_audio_omni full_s2a_whisper_qwen32_audiogen full_s2a_whisper_qwen32_tangoflux
  )
  lane="${lanes[$((index - 2))]}"
  run_metric T2A "$lane" fad 0 & p0=$!
  run_metric T2A "$lane" kad 1 & p1=$!
  run_metric T2A "$lane" clap_score_t2a 2 & p2=$!
  run_metric T2A "$lane" kl_passt 3 & p3=$!
  wait "$p0" || failures=$((failures + 1))
  run_metric T2A "$lane" inception_score 0 || failures=$((failures + 1))
  wait "$p1" || failures=$((failures + 1))
  wait "$p2" || failures=$((failures + 1))
  wait "$p3" || failures=$((failures + 1))
elif [ "$index" -lt 26 ]; then
  lanes=(
    full_a2s_afn_llama_audio_omni full_a2s_afn_llama_cosyvoice3 full_a2s_afn_llama_f5_tts
    full_a2s_afn_qwen32_audio_omni full_a2s_afn_qwen32_cosyvoice3 full_a2s_afn_qwen32_f5_tts
    full_a2s_qwencap_llama_audio_omni full_a2s_qwencap_llama_cosyvoice3 full_a2s_qwencap_llama_f5_tts
    full_a2s_qwencap_qwen32_audio_omni full_a2s_qwencap_qwen32_cosyvoice3 full_a2s_qwencap_qwen32_f5_tts
  )
  lane="${lanes[$((index - 14))]}"
  run_parallel <<EOF || failures=$?
T2S|$lane|asr_wer|0
T2S|$lane|dwer|1
T2S|$lane|utmos|2
T2S|$lane|dnsmos|3
EOF
else
  run_parallel <<'EOF' || failures=$?
S2T|full_s2t_whisper|wer|0
S2T|full_s2t_qwen3_asr|wer|1
EOF
fi

stop_resource_monitor
echo "final_metrics_v2_complete index=$index failures=$failures logs=$log_root outputs=$out_root"
exit "$failures"
