#!/usr/bin/env bash
#SBATCH --job-name=salm-eval-val-smoke
#SBATCH --account=aip-csubakan
#SBATCH --time=00:40:00
#SBATCH --cpus-per-task=48
#SBATCH --mem=0
#SBATCH --gpus=h100:4
#SBATCH --output=/scratch/l/lovenya/salm_migration_clean/logs/slurm/%x-%j.out

set -euo pipefail

cd /scratch/l/lovenya/SpeechAudioLM
module --force purge
module load StdEnv/2023 gcc cuda java/11.0.22 python/3.10.13
unset JAVA_TOOL_OPTIONS

export SALM_SCRATCH="${SALM_SCRATCH:-/scratch/l/lovenya/salm_migration_clean}"
export PYTHONDONTWRITEBYTECODE=1
export XDG_CACHE_HOME="$SALM_SCRATCH/cache/metrics"
export HF_HOME="$SALM_SCRATCH/cache/metrics/huggingface"
export HF_HUB_CACHE="$HF_HOME"
export HUGGINGFACE_HUB_CACHE="$HF_HOME"
export TRANSFORMERS_CACHE="$HF_HOME"
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export TORCH_HOME="$SALM_SCRATCH/cache/metrics/torch"
export RESOURCE_INTERVAL="${RESOURCE_INTERVAL:-30}"
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-2}"
export MKL_NUM_THREADS="${MKL_NUM_THREADS:-2}"
export STRICT_FAILURES="${STRICT_FAILURES:-0}"

MAX_SAMPLES="${MAX_SAMPLES:-50}"
INCLUDE_NLI="${INCLUDE_NLI:-1}"
SPLIT="val"
DATASET="ttas"
RUN_ID="${RUN_ID:-$(TZ=America/Toronto date +%Y%m%dT%H%M%S%z)}"
BASELINE_ROOT="$SALM_SCRATCH/results/cascaded_full_validation/ttas_val/cascaded"
EVAL_SUITE_ROOT="${EVAL_SUITE_ROOT:-$SALM_SCRATCH/results/evals/validation_smoke_packed}"
CACHE_ROOT="${CACHE_ROOT:-$SALM_SCRATCH/cache/metrics/eval_runtime}"
RUN_LOG_DIR="$SALM_SCRATCH/logs/eval_packed/validation_smoke/$RUN_ID"
RUNTIME_METRICS_CONFIG="$RUN_LOG_DIR/eval_metrics.runtime.json"
NLI_PYTHON="${NLI_PYTHON:-$SALM_SCRATCH/.venvs/metrics/.venv-eval-aac/bin/python}"
NLI_MODEL="${NLI_MODEL:-MoritzLaurer/DeBERTa-v3-large-mnli-fever-anli-ling-wanli}"
NLI_SNAPSHOT="${NLI_SNAPSHOT:-}"
export MAX_SAMPLES INCLUDE_NLI SPLIT DATASET RUN_ID BASELINE_ROOT EVAL_SUITE_ROOT CACHE_ROOT RUN_LOG_DIR RUNTIME_METRICS_CONFIG NLI_PYTHON NLI_MODEL NLI_SNAPSHOT STRICT_FAILURES

mkdir -p "$EVAL_SUITE_ROOT" "$CACHE_ROOT" "$RUN_LOG_DIR" "$SALM_SCRATCH/logs/slurm"

if [ "$INCLUDE_NLI" = "1" ] && [ -z "$NLI_SNAPSHOT" ]; then
  NLI_SNAPSHOT="$("$SALM_SCRATCH/.venvs/download/.venv-hf-download/bin/python" - <<'PY_NLI_SNAPSHOT'
import os
from huggingface_hub import snapshot_download
print(snapshot_download(
    os.environ["NLI_MODEL"],
    cache_dir=os.environ["HF_HOME"],
    local_files_only=True,
))
PY_NLI_SNAPSHOT
)"
  export NLI_SNAPSHOT
fi

python - <<'PY_RUNTIME_CONFIG'
import json
import os
from pathlib import Path
src = Path('configs/eval_metrics.json')
dst = Path(os.environ['RUNTIME_METRICS_CONFIG'])
data = json.loads(src.read_text(encoding='utf-8'))
nli = data['metrics'].setdefault('nli_contradiction', {})
nli.update({
    'backend': 'dedicated-script',
    'script': 'scripts/run_nli_contradiction_helper.py',
    'script_backend': 'hf',
    'python': os.environ['NLI_PYTHON'],
    'model_name': os.environ.get('NLI_SNAPSHOT') or os.environ['NLI_MODEL'],
    'hf_home': os.environ['HF_HOME'],
    'transformers_cache': os.environ['HF_HOME'],
    'hf_hub_offline': '1',
    'transformers_offline': '1',
    'local_files_only': True,
    'batch_size': 8,
    'contradiction_threshold': 0.7,
    'allow_stub_fallback': False,
})
dst.parent.mkdir(parents=True, exist_ok=True)
dst.write_text(json.dumps(data, indent=2, ensure_ascii=False) + '\n', encoding='utf-8')
print(f'runtime_metrics_config={dst}')
PY_RUNTIME_CONFIG

if [ "$INCLUDE_NLI" = "1" ]; then
  if [ ! -x "$NLI_PYTHON" ]; then
    echo "missing NLI python: $NLI_PYTHON" >&2
    exit 2
  fi
  "$NLI_PYTHON" - <<'PY_NLI_PREFLIGHT'
import os
from transformers import AutoModelForSequenceClassification, AutoTokenizer
model = os.environ.get('NLI_SNAPSHOT') or os.environ['NLI_MODEL']
AutoTokenizer.from_pretrained(model, local_files_only=True)
AutoModelForSequenceClassification.from_pretrained(model, local_files_only=True)
print(f'nli_model_available={model}')
PY_NLI_PREFLIGHT
fi

source submission_scripts/lib_resource_monitor.sh
start_resource_monitor "eval_validation_packed_smoke_n${MAX_SAMPLES}"

run_eval() {
  local lane="$1"
  local task="$2"
  local metrics="$3"
  local manifest="$4"
  local gpu="$5"
  local lane_root="$EVAL_SUITE_ROOT/$lane"
  local lane_log="$RUN_LOG_DIR/${lane}.log"
  local lane_failures=0

  if [ ! -f "$manifest" ]; then
    echo "missing manifest for $lane: $manifest" >&2
    return 2
  fi

  echo "start lane=$lane task=$task metrics=$metrics gpu=$gpu manifest=$manifest" | tee "$lane_log"
  IFS=',' read -r -a metric_list <<< "$metrics"
  for metric in "${metric_list[@]}"; do
    local metric_root="$lane_root/$metric"
    local metric_log="$RUN_LOG_DIR/${lane}.${metric}.log"
    echo "start metric lane=$lane metric=$metric gpu=$gpu output_root=$metric_root" | tee "$metric_log" >>"$lane_log"
    if (
      export CUDA_VISIBLE_DEVICES="$gpu"
      python evaluate.py \
        --model cascaded \
        --task "$task" \
        --dataset "$DATASET" \
        --split "$SPLIT" \
        --manifest "$manifest" \
        --metrics "$metric" \
        --metrics-config "$RUNTIME_METRICS_CONFIG" \
        --output-root "$metric_root" \
        --cache-dir "$CACHE_ROOT" \
        --device cuda \
        --max-samples "$MAX_SAMPLES"
    ) >>"$metric_log" 2>&1; then
      echo "done metric lane=$lane metric=$metric" | tee -a "$metric_log" >>"$lane_log"
    else
      lane_failures=$((lane_failures + 1))
      echo "failed metric lane=$lane metric=$metric; continuing" | tee -a "$metric_log" >>"$lane_log"
    fi
  done

  if [ "$lane_failures" -ne 0 ]; then
    echo "done lane=$lane metric_failures=$lane_failures" >>"$lane_log"
    return 1
  fi
  echo "done lane=$lane metric_failures=0" >>"$lane_log"
}

pids=()
failures=0
launch() {
  run_eval "$@" &
  pids+=("$!")
}

T2A_METRICS="fad,kad,inception_score,clap_score_t2a,kl_passt"
launch full_s2a_whisper_qwen32_audiogen T2A "$T2A_METRICS" "$BASELINE_ROOT/T2A/full_s2a_whisper_qwen32_audiogen/ttas/val/manifest.jsonl" 0
launch full_s2a_whisper_qwen32_tangoflux T2A "$T2A_METRICS" "$BASELINE_ROOT/T2A/full_s2a_whisper_qwen32_tangoflux/ttas/val/manifest.jsonl" 1
launch full_s2a_qwen3_asr_qwen32_audiogen T2A "$T2A_METRICS" "$BASELINE_ROOT/T2A/full_s2a_qwen3_asr_qwen32_audiogen/ttas/val/manifest.jsonl" 2
launch full_s2a_qwen3_asr_qwen32_tangoflux T2A "$T2A_METRICS" "$BASELINE_ROOT/T2A/full_s2a_qwen3_asr_qwen32_tangoflux/ttas/val/manifest.jsonl" 3
launch full_s2a_whisper_llama_audiogen T2A "$T2A_METRICS" "$BASELINE_ROOT/T2A/full_s2a_whisper_llama_audiogen/ttas/val/manifest.jsonl" 0
launch full_s2a_whisper_llama_tangoflux T2A "$T2A_METRICS" "$BASELINE_ROOT/T2A/full_s2a_whisper_llama_tangoflux/ttas/val/manifest.jsonl" 1
launch full_s2a_qwen3_asr_llama_audiogen T2A "$T2A_METRICS" "$BASELINE_ROOT/T2A/full_s2a_qwen3_asr_llama_audiogen/ttas/val/manifest.jsonl" 2
launch full_s2a_qwen3_asr_llama_tangoflux T2A "$T2A_METRICS" "$BASELINE_ROOT/T2A/full_s2a_qwen3_asr_llama_tangoflux/ttas/val/manifest.jsonl" 3

T2S_METRICS="asr_wer,dwer,utmos,dnsmos"
launch full_a2s_qwencap_qwen32_f5_tts T2S "$T2S_METRICS" "$BASELINE_ROOT/T2S/full_a2s_qwencap_qwen32_f5_tts/ttas/val/manifest.jsonl" 0
launch full_a2s_qwencap_qwen32_cosyvoice3 T2S "$T2S_METRICS" "$BASELINE_ROOT/T2S/full_a2s_qwencap_qwen32_cosyvoice3/ttas/val/manifest.jsonl" 1
launch full_a2s_qwencap_llama_f5_tts T2S "$T2S_METRICS" "$BASELINE_ROOT/T2S/full_a2s_qwencap_llama_f5_tts/ttas/val/manifest.jsonl" 2
launch full_a2s_qwencap_llama_cosyvoice3 T2S "$T2S_METRICS" "$BASELINE_ROOT/T2S/full_a2s_qwencap_llama_cosyvoice3/ttas/val/manifest.jsonl" 3

if [ "$INCLUDE_NLI" = "1" ]; then
  launch full_a2s_afn_qwen32_t2t T2T nli_contradiction "$BASELINE_ROOT/T2T/full_a2s_afn_qwen32_t2t/ttas/val/manifest.jsonl" 0
  launch full_a2s_afn_llama_t2t T2T nli_contradiction "$BASELINE_ROOT/T2T/full_a2s_afn_llama_t2t/ttas/val/manifest.jsonl" 1
fi

for pid in "${pids[@]}"; do
  if ! wait "$pid"; then
    failures=$((failures + 1))
  fi
done

python - <<'PY_LIST_SUMMARIES'
import os
from pathlib import Path
root = Path(os.environ['EVAL_SUITE_ROOT'])
print('eval_suite_root=', root)
for path in sorted(root.glob('*/**/latest_summary_val.json')):
    print(path)
PY_LIST_SUMMARIES

if [ "$failures" -ne 0 ]; then
  echo "packed eval smoke completed with lane failures=$failures" >&2
  if [ "$STRICT_FAILURES" = "1" ]; then
    exit 1
  fi
fi

echo "packed eval smoke completed; lane_failures=$failures strict_failures=$STRICT_FAILURES"
