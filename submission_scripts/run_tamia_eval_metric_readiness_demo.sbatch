#!/usr/bin/env bash
#SBATCH --job-name=salm-eval-ready
#SBATCH --account=aip-csubakan
#SBATCH --time=01:30:00
#SBATCH --cpus-per-task=48
#SBATCH --mem=0
#SBATCH --gpus=h100:4
#SBATCH --output=/scratch/l/lovenya/salm_migration_clean/logs/slurm/%x-%j.out

set -uo pipefail

cd /scratch/l/lovenya/SpeechAudioLM
module --force purge
module load StdEnv/2023 gcc cuda java/11.0.22 python/3.10.13
unset JAVA_TOOL_OPTIONS

export SALM_SCRATCH="${SALM_SCRATCH:-/scratch/l/lovenya/salm_migration_clean}"
export PYTHONDONTWRITEBYTECODE=1
export XDG_CACHE_HOME="$SALM_SCRATCH/cache/metrics"
export HOME="$SALM_SCRATCH/cache/metrics/runtime_home"
export HF_HOME="$SALM_SCRATCH/cache/metrics/huggingface"
export HF_HUB_CACHE="$HF_HOME"
export HUGGINGFACE_HUB_CACHE="$HF_HOME"
export TRANSFORMERS_CACHE="$HF_HOME"
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export TOKENIZERS_PARALLELISM=false
export TORCH_HOME="$SALM_SCRATCH/cache/metrics/torch"
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-2}"
export MKL_NUM_THREADS="${MKL_NUM_THREADS:-2}"
export NUMEXPR_NUM_THREADS="${NUMEXPR_NUM_THREADS:-2}"
export RESOURCE_INTERVAL="${RESOURCE_INTERVAL:-30}"

MAX_SAMPLES="${MAX_SAMPLES:-50}"
STRICT_FAILURES="${STRICT_FAILURES:-0}"
RUN_ID="${RUN_ID:-$(TZ=America/Toronto date +%Y%m%dT%H%M%S%z)}"
BASELINE_ROOT="$SALM_SCRATCH/results/cascaded_full_validation/ttas_val/cascaded"
OUT_ROOT="${OUT_ROOT:-$SALM_SCRATCH/results/evals/validation_metric_readiness_demo}"
CACHE_ROOT="${CACHE_ROOT:-$SALM_SCRATCH/cache/metrics/eval_runtime}"
RUN_LOG_DIR="$SALM_SCRATCH/logs/eval_packed/metric_readiness/$RUN_ID"
RUNTIME_METRICS_CONFIG="$RUN_LOG_DIR/eval_metrics.runtime.json"
NLI_MODEL="${NLI_MODEL:-MoritzLaurer/DeBERTa-v3-large-mnli-fever-anli-ling-wanli}"
NLI_PYTHON="${NLI_PYTHON:-$SALM_SCRATCH/.venvs/metrics/.venv-eval-aac/bin/python}"

export MAX_SAMPLES STRICT_FAILURES RUN_ID BASELINE_ROOT OUT_ROOT CACHE_ROOT RUN_LOG_DIR
export RUNTIME_METRICS_CONFIG NLI_MODEL NLI_PYTHON
mkdir -p "$OUT_ROOT" "$CACHE_ROOT" "$RUN_LOG_DIR" "$SALM_SCRATCH/logs/slurm"

NLI_SNAPSHOT="$("$SALM_SCRATCH/.venvs/download/.venv-hf-download/bin/python" - <<'PY_NLI'
import os
from huggingface_hub import snapshot_download
print(snapshot_download(os.environ['NLI_MODEL'], cache_dir=os.environ['HF_HOME'], local_files_only=True))
PY_NLI
)"
export NLI_SNAPSHOT

python - <<'PY_CONFIG'
import json
import os
from pathlib import Path
src = Path('configs/eval_metrics.json')
dst = Path(os.environ['RUNTIME_METRICS_CONFIG'])
data = json.loads(src.read_text(encoding='utf-8'))
nli = data['metrics']['nli_contradiction']
nli.update({
    'model_name': os.environ['NLI_SNAPSHOT'],
    'python': os.environ['NLI_PYTHON'],
    'hf_home': os.environ['HF_HOME'],
    'transformers_cache': os.environ['HF_HOME'],
    'hf_hub_offline': '1',
    'transformers_offline': '1',
    'local_files_only': True,
    'allow_stub_fallback': False,
})
dst.write_text(json.dumps(data, indent=2, ensure_ascii=False) + '\n', encoding='utf-8')
print(f'runtime_metrics_config={dst}')
print(f'nli_snapshot={os.environ["NLI_SNAPSHOT"]}')
PY_CONFIG

source submission_scripts/lib_resource_monitor.sh
start_resource_monitor "eval_metric_readiness_n${MAX_SAMPLES}"

run_eval() {
  local lane="$1"
  local task="$2"
  local metric="$3"
  local manifest="$4"
  local gpu="$5"
  local metric_root="$OUT_ROOT/$lane/$metric"
  local metric_log="$RUN_LOG_DIR/${lane}.${metric}.log"

  echo "start lane=$lane task=$task metric=$metric gpu=$gpu manifest=$manifest" | tee "$metric_log"
  if [ ! -f "$manifest" ]; then
    echo "failed missing_manifest=$manifest" | tee -a "$metric_log"
    return 2
  fi

  if (
    export CUDA_VISIBLE_DEVICES="$gpu"
    python evaluate.py \
      --model cascaded \
      --task "$task" \
      --dataset ttas \
      --split val \
      --manifest "$manifest" \
      --metrics "$metric" \
      --metrics-config "$RUNTIME_METRICS_CONFIG" \
      --output-root "$metric_root" \
      --cache-dir "$CACHE_ROOT" \
      --device cuda \
      --max-samples "$MAX_SAMPLES"
  ) >>"$metric_log" 2>&1; then
    echo "PASS lane=$lane metric=$metric" | tee -a "$metric_log"
    return 0
  fi

  echo "FAIL lane=$lane metric=$metric" | tee -a "$metric_log"
  return 1
}

pids=()
labels=()
launch() {
  local label="$1"
  shift
  run_eval "$@" &
  pids+=("$!")
  labels+=("$label")
}

AFN_A2T="$BASELINE_ROOT/A2T/full_a2t_audio_flamingo_next/ttas/val/manifest.jsonl"
QWEN_A2T="$BASELINE_ROOT/A2T/full_a2t_qwen3_omni_captioner/ttas/val/manifest.jsonl"
T2A_MANIFEST="$BASELINE_ROOT/T2A/full_s2a_qwen3_asr_llama_audiogen/ttas/val/manifest.jsonl"
AFN_QWEN_T2T="$BASELINE_ROOT/T2T/full_a2s_afn_qwen32_t2t/ttas/val/manifest.jsonl"
AFN_LLAMA_T2T="$BASELINE_ROOT/T2T/full_a2s_afn_llama_t2t/ttas/val/manifest.jsonl"

# A2T reference-based caption metrics on both full validation caption manifests.
launch afn_mace       full_a2t_audio_flamingo_next A2T mace        "$AFN_A2T" 0
launch afn_spider     full_a2t_audio_flamingo_next A2T spider      "$AFN_A2T" 1
launch afn_clapsim    full_a2t_audio_flamingo_next A2T aac_clapsim "$AFN_A2T" 2
launch afn_fense      full_a2t_audio_flamingo_next A2T fense       "$AFN_A2T" 3
launch qwen_mace      full_a2t_qwen3_omni_captioner A2T mace        "$QWEN_A2T" 2
launch qwen_spider    full_a2t_qwen3_omni_captioner A2T spider      "$QWEN_A2T" 3
launch qwen_clapsim   full_a2t_qwen3_omni_captioner A2T aac_clapsim "$QWEN_A2T" 0
launch qwen_fense     full_a2t_qwen3_omni_captioner A2T fense       "$QWEN_A2T" 1

# One representative T2A lane exercises every audio metric backend after the
# original-reference path fallback fix. Full-lane expansion comes after this passes.
launch t2a_fad        full_s2a_qwen3_asr_llama_audiogen T2A fad             "$T2A_MANIFEST" 0
launch t2a_kad        full_s2a_qwen3_asr_llama_audiogen T2A kad             "$T2A_MANIFEST" 1
launch t2a_inception  full_s2a_qwen3_asr_llama_audiogen T2A inception_score "$T2A_MANIFEST" 2
launch t2a_clap       full_s2a_qwen3_asr_llama_audiogen T2A clap_score_t2a  "$T2A_MANIFEST" 3
launch t2a_kl         full_s2a_qwen3_asr_llama_audiogen T2A kl_passt        "$T2A_MANIFEST" 3

# Both AFN T2T lanes verify the corrected three-way NLI checkpoint and mapping.
launch nli_qwen       full_a2s_afn_qwen32_t2t T2T nli_contradiction "$AFN_QWEN_T2T" 0
launch nli_llama      full_a2s_afn_llama_t2t T2T nli_contradiction "$AFN_LLAMA_T2T" 1

failures=0
for index in "${!pids[@]}"; do
  if ! wait "${pids[$index]}"; then
    failures=$((failures + 1))
    echo "failed_task=${labels[$index]}"
  fi
done

pass_count="$(grep -l '^PASS ' "$RUN_LOG_DIR"/*.log 2>/dev/null | wc -l)"
fail_count="$(grep -El '^FAIL |^failed ' "$RUN_LOG_DIR"/*.log 2>/dev/null | wc -l)"
echo "metric_readiness_complete pass=$pass_count fail=$fail_count process_failures=$failures"
find "$OUT_ROOT" -path '*latest_summary_val.json' -type f -mmin -180 -print | sort

if [ "$STRICT_FAILURES" = "1" ] && [ "$failures" -ne 0 ]; then
  exit 1
fi
exit 0
