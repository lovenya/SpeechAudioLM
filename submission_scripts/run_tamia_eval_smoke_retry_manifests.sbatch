#!/usr/bin/env bash
#SBATCH --job-name=salm-eval-smoke
#SBATCH --account=aip-csubakan
#SBATCH --time=00:30:00
#SBATCH --nodes=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
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
export HOME="$SALM_SCRATCH/cache/metrics/runtime_home"
export HF_HOME="$SALM_SCRATCH/cache/metrics/huggingface"
export HF_HUB_CACHE="$HF_HOME"
export HUGGINGFACE_HUB_CACHE="$HF_HOME"
export TRANSFORMERS_CACHE="$HF_HOME"
export TORCH_HOME="$SALM_SCRATCH/cache/metrics/torch"
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export TOKENIZERS_PARALLELISM=false
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-4}"
export MKL_NUM_THREADS="${MKL_NUM_THREADS:-4}"
export NUMEXPR_NUM_THREADS="${NUMEXPR_NUM_THREADS:-4}"

OUT_ROOT="$SALM_SCRATCH/results/eval_smoke_retry"
CACHE_ROOT="$SALM_SCRATCH/cache/metrics/eval_smoke_runtime"
mkdir -p "$OUT_ROOT" "$CACHE_ROOT" "$SALM_SCRATCH/logs/slurm"

source submission_scripts/lib_resource_monitor.sh
start_resource_monitor "tamia_eval_smoke_retry_manifests"

status=0
run_eval() {
  local label="$1"
  shift
  echo "[$label] started=$(date -Is)"
  if python evaluate.py "$@"; then
    echo "[$label] PASS finished=$(date -Is)"
  else
    echo "[$label] FAIL finished=$(date -Is)"
    status=1
  fi
}

run_eval audiogen_t2a_audioldm_metrics   --model audiogen   --task T2A   --manifest "$SALM_SCRATCH/results/baselines_smoke_retry/audiogen/T2A/ttas/test/manifest.jsonl"   --metrics fad,inception_score   --max-samples 1   --output-root "$OUT_ROOT"   --cache-dir "$CACHE_ROOT"   --device cuda

run_eval audiogen_t2a_kad_metric   --model audiogen   --task T2A   --manifest "$SALM_SCRATCH/results/baselines_smoke_retry/audiogen/T2A/ttas/test/manifest.jsonl"   --metrics kad   --max-samples 1   --output-root "$OUT_ROOT"   --cache-dir "$CACHE_ROOT"   --device cuda

run_eval audiogen_t2a_stable_audio_metrics   --model audiogen   --task T2A   --manifest "$SALM_SCRATCH/results/baselines_smoke_retry/audiogen/T2A/ttas/test/manifest.jsonl"   --metrics clap_score_t2a,kl_passt   --max-samples 1   --output-root "$OUT_ROOT"   --cache-dir "$CACHE_ROOT"   --device cuda

run_eval audio_omni_s2a_audioldm_metrics   --model audio_omni   --task S2A   --manifest "$SALM_SCRATCH/results/baselines_smoke_retry/audio_omni/S2A/ttas/test/manifest.jsonl"   --metrics fad,inception_score   --max-samples 1   --output-root "$OUT_ROOT"   --cache-dir "$CACHE_ROOT"   --device cuda

run_eval audio_omni_s2a_kad_metric   --model audio_omni   --task S2A   --manifest "$SALM_SCRATCH/results/baselines_smoke_retry/audio_omni/S2A/ttas/test/manifest.jsonl"   --metrics kad   --max-samples 1   --output-root "$OUT_ROOT"   --cache-dir "$CACHE_ROOT"   --device cuda

run_eval audio_omni_s2a_stable_audio_metrics   --model audio_omni   --task S2A   --manifest "$SALM_SCRATCH/results/baselines_smoke_retry/audio_omni/S2A/ttas/test/manifest.jsonl"   --metrics clap_score_t2a,kl_passt   --max-samples 1   --output-root "$OUT_ROOT"   --cache-dir "$CACHE_ROOT"   --device cuda

run_eval audio_omni_a2s_speech_metrics   --model audio_omni   --task A2S   --manifest "$SALM_SCRATCH/results/baselines_smoke_retry/audio_omni/A2S/ttas/test/manifest.jsonl"   --metrics asr_wer,dwer,utmos   --max-samples 1   --output-root "$OUT_ROOT"   --cache-dir "$CACHE_ROOT"   --device cuda

run_eval qwen3_omni_a2s_speech_metrics   --model qwen3_omni   --task A2S   --manifest "$SALM_SCRATCH/results/baselines_smoke_retry/qwen3_omni/A2S/ttas/test/manifest.jsonl"   --metrics asr_wer,dwer,utmos   --max-samples 1   --output-root "$OUT_ROOT"   --cache-dir "$CACHE_ROOT"   --device cuda

find "$OUT_ROOT" -name 'latest_summary_test.json' -print
exit "$status"
