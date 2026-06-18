#!/usr/bin/env bash
# Run from an active TamIA GPU allocation. Dense temporary fit for non-G-Eval
# GPU-backed metrics: generated audio/speech, A2T neural metrics, and NLI.

set -euo pipefail

cd /scratch/l/lovenya/SpeechAudioLM

module --force purge
module load StdEnv/2023 gcc cuda/12.6 opencv/4.13.0 java/11.0.22 python/3.12.4 scipy-stack
unset JAVA_TOOL_OPTIONS

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
export RESOURCE_INTERVAL="${RESOURCE_INTERVAL:-10}"

MAX_SAMPLES="${MAX_SAMPLES:-50}"
SHARED_PYTHON="${SHARED_PYTHON:-/home/l/lovenya/.venvs/speechaudiolm/.venv-eval-metrics-shared-py312/bin/python}"
BASE="$SALM_SCRATCH/results/cascaded_full_validation/ttas_val/cascaded"
RUN_ID="$(TZ=America/Toronto date +%Y%m%dT%H%M%S%z)"
LOG_DIR="$SALM_SCRATCH/logs/evals/gpu_metric_dense_fit/$RUN_ID"
RUNTIME_CONFIG="$LOG_DIR/eval_metrics.runtime.json"

mkdir -p "$LOG_DIR" "$SALM_SCRATCH/logs/resources"
test -x "$SHARED_PYTHON" || { echo "missing shared python: $SHARED_PYTHON" >&2; exit 2; }

python - <<PY
import json
from pathlib import Path

cfg = json.loads(Path("configs/eval_metrics.json").read_text(encoding="utf-8"))
for name in (
    "mace", "aac_clapsim", "fense",
    "asr_wer", "dwer", "utmos", "dnsmos",
    "nli_contradiction",
):
    cfg["metrics"][name]["python"] = "$SHARED_PYTHON"
cfg["metrics"]["nli_contradiction"]["local_files_only"] = True
cfg["metrics"]["nli_contradiction"].pop("allow_stub_fallback", None)
Path("$RUNTIME_CONFIG").write_text(json.dumps(cfg, indent=2) + "\n", encoding="utf-8")
print("runtime_config=$RUNTIME_CONFIG")
PY

source submission_scripts/lib_resource_monitor.sh
start_resource_monitor "gpu_metric_dense_fit_${RUN_ID}_n${MAX_SAMPLES}"

run_eval() {
  local label="$1"
  local stage="$2"
  local direction="$3"
  local metrics="$4"
  local manifest="$5"
  local gpu="$6"
  local evaluated_run_id="ttas_val_${label}"
  local log="$LOG_DIR/${label}.log"

  echo "START label=$label stage=$stage direction=$direction metrics=$metrics gpu=$gpu max_samples=$MAX_SAMPLES" | tee "$log"

  local direction_args=()
  if [ "$direction" != "none" ]; then
    direction_args=(--direction "$direction")
  fi

  CUDA_VISIBLE_DEVICES="$gpu" python evaluate.py \
    --model cascaded \
    --stage "$stage" \
    "${direction_args[@]}" \
    --evaluated-run-id "$evaluated_run_id" \
    --dataset ttas \
    --split val \
    --manifest "$manifest" \
    --metrics "$metrics" \
    --metrics-config "$RUNTIME_CONFIG" \
    --cache-dir "$SALM_SCRATCH/cache/metrics/eval_runtime" \
    --device cuda \
    --max-samples "$MAX_SAMPLES" \
    --temporary >>"$log" 2>&1

  echo "PASS label=$label" | tee -a "$log"
}

failures=0
pids=()

run_eval t2a_whisper_qwen32_audiogen t2a none fad,kad,inception_score,kl_passt,target_caption_clap_score,prompt_adherence_clap_score "$BASE/T2A/full_s2a_whisper_qwen32_audiogen/ttas/val/manifest.jsonl" 0 & pids+=("$!")
run_eval t2t_a2s_afn_qwen32_nli t2t a2s nli_contradiction "$BASE/T2T/full_a2s_afn_qwen32_t2t/ttas/val/manifest.jsonl" 0 & pids+=("$!")

run_eval t2a_whisper_qwen32_tangoflux t2a none fad,kad,inception_score,kl_passt,target_caption_clap_score,prompt_adherence_clap_score "$BASE/T2A/full_s2a_whisper_qwen32_tangoflux/ttas/val/manifest.jsonl" 1 & pids+=("$!")
run_eval t2t_s2a_whisper_qwen32_nli t2t s2a nli_contradiction "$BASE/T2T/full_s2a_whisper_qwen32_t2t/ttas/val/manifest.jsonl" 1 & pids+=("$!")

run_eval t2s_afn_qwen32_f5 t2s none asr_wer,dwer,utmos,dnsmos "$BASE/T2S/full_a2s_afn_qwen32_f5_tts/ttas/val/manifest.jsonl" 2 & pids+=("$!")
run_eval a2t_afn_neural a2t none mace,aac_clapsim,fense "$BASE/A2T/full_a2t_audio_flamingo_next/ttas/val/manifest.jsonl" 2 & pids+=("$!")

run_eval t2s_afn_qwen32_cosyvoice t2s none asr_wer,dwer,utmos,dnsmos "$BASE/T2S/full_a2s_afn_qwen32_cosyvoice3/ttas/val/manifest.jsonl" 3 & pids+=("$!")
run_eval a2t_qwen3_captioner_neural a2t none mace,aac_clapsim,fense "$BASE/A2T/full_a2t_qwen3_omni_captioner/ttas/val/manifest.jsonl" 3 & pids+=("$!")

for pid in "${pids[@]}"; do
  wait "$pid" || failures=$((failures + 1))
done

stop_resource_monitor || true

echo "DENSE_FIT_DONE failures=$failures log_dir=$LOG_DIR"
exit "$failures"
