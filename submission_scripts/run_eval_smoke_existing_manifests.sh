#!/usr/bin/env bash
set -euo pipefail

# Run this inside an interactive compute allocation. It validates the metric
# backend adapters against existing small baseline manifests without submitting
# a Slurm job array.

cd /lustre09/project/6090520/lovenya/SpeechAudioLM

module purge
module load StdEnv/2023 gcc cuda java/11.0.22 python/3.10.13
unset JAVA_TOOL_OPTIONS

export SALM_SCRATCH="${SALM_SCRATCH:-/scratch/lovenya/SpeechAudioLM}"
export PYTHONDONTWRITEBYTECODE=1
export XDG_CACHE_HOME="$SALM_SCRATCH/cache/metrics"
export HF_HOME="$SALM_SCRATCH/cache/metrics/huggingface"
export TORCH_HOME="$SALM_SCRATCH/cache/metrics/torch"

OUT_ROOT="$SALM_SCRATCH/results/eval_smoke_existing"
CACHE_ROOT="$SALM_SCRATCH/cache/metrics/eval_smoke_runtime"
mkdir -p "$OUT_ROOT" "$CACHE_ROOT"

python evaluate.py \
  --model audio_omni \
  --task T2A \
  --metrics fad,kad,inception_score,clap_score_t2a,kl_passt \
  --max-samples 2 \
  --baseline-root results/baselines \
  --output-root "$OUT_ROOT" \
  --cache-dir "$CACHE_ROOT" \
  --device cuda

python evaluate.py \
  --model audio_omni \
  --task T2S \
  --metrics asr_wer,dwer,utmos \
  --max-samples 2 \
  --baseline-root results/baselines \
  --output-root "$OUT_ROOT" \
  --cache-dir "$CACHE_ROOT" \
  --device cuda

python evaluate.py \
  --model audio_omni \
  --task A2T \
  --manifest results/baselines/audio_omni/A2S/test/audio_cap/manifest.jsonl \
  --metrics mace,spider,aac_clapsim \
  --max-samples 2 \
  --baseline-root results/baselines \
  --output-root "$OUT_ROOT" \
  --cache-dir "$CACHE_ROOT" \
  --device cuda

python evaluate.py \
  --model audio_omni \
  --task S2T \
  --manifest results/baselines/audio_omni/S2A/test/speech_text/manifest.jsonl \
  --metrics wer \
  --max-samples 2 \
  --baseline-root results/baselines \
  --output-root "$OUT_ROOT" \
  --cache-dir "$CACHE_ROOT" \
  --device cuda

find "$OUT_ROOT" -name 'latest_summary_test.json' -print
