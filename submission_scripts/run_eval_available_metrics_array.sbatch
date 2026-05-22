#!/usr/bin/env bash
#SBATCH --job-name=salm-eval
#SBATCH --account=rrg-ravanelm
#SBATCH --time=04:00:00
#SBATCH --cpus-per-task=4
#SBATCH --mem=96G
#SBATCH --gpus=h100_3g.40gb:1
#SBATCH --array=0-8
#SBATCH --output=/scratch/lovenya/SpeechAudioLM/logs/slurm/%x-%A_%a.out

set -euo pipefail

# Format: model|task|metrics
JOBS=(
  "audiogen|T2A|fad,kad,inception_score,clap_score_t2a,kl_passt"
  "tangoflux|T2A|fad,kad,inception_score,clap_score_t2a,kl_passt"
  "audio_omni|T2A|fad,kad,inception_score,clap_score_t2a,kl_passt"
  "audio_omni|S2A|fad,kad,inception_score,clap_score_t2a,kl_passt"
  "audio_omni|T2S|asr_wer,dwer,utmos"
  "audio_omni|A2S|asr_wer,dwer,utmos"
  "audio_omni|A2T|mace,spider,aac_clapsim"
  "audio_omni|S2T|wer"
  "qwen3_omni|A2S|asr_wer,dwer,utmos"
)

IFS='|' read -r MODEL TASK METRICS <<< "${JOBS[$SLURM_ARRAY_TASK_ID]}"

cd /lustre09/project/6090520/lovenya/SpeechAudioLM
module --force purge
module load StdEnv/2023 gcc cuda java/11.0.22 python/3.10.13
unset JAVA_TOOL_OPTIONS

export SALM_SCRATCH="${SALM_SCRATCH:-/scratch/lovenya/SpeechAudioLM}"
export PYTHONDONTWRITEBYTECODE=1
export XDG_CACHE_HOME="$SALM_SCRATCH/cache/metrics"
export HF_HOME="$SALM_SCRATCH/cache/metrics/huggingface"
export TORCH_HOME="$SALM_SCRATCH/cache/metrics/torch"

mkdir -p "$SALM_SCRATCH/results/evals" "$SALM_SCRATCH/logs/slurm"

EXTRA_ARGS=()
if [ -n "${START_INDEX:-}" ]; then
  EXTRA_ARGS+=(--start-index "$START_INDEX")
fi
if [ -n "${MAX_SAMPLES:-}" ]; then
  EXTRA_ARGS+=(--max-samples "$MAX_SAMPLES")
fi
if [ "${DRY_RUN:-0}" = "1" ]; then
  EXTRA_ARGS+=(--dry-run)
fi

EVAL_ROOT="$SALM_SCRATCH/results/evals"
if [ "${SHARDED:-0}" = "1" ]; then
  SHARD_LABEL="start${START_INDEX:-0}_n${MAX_SAMPLES:-all}"
  EVAL_ROOT="$SALM_SCRATCH/results/eval_shards/$SHARD_LABEL"
fi
source submission_scripts/lib_resource_monitor.sh
start_resource_monitor "eval_${MODEL}_${TASK}_${METRICS}_start${START_INDEX:-0}_n${MAX_SAMPLES:-all}"

python evaluate.py \
  --model "$MODEL" \
  --task "$TASK" \
  --metrics "$METRICS" \
  --baseline-root "$SALM_SCRATCH/results/baselines" \
  --output-root "$EVAL_ROOT" \
  --cache-dir "$SALM_SCRATCH/cache/metrics/eval_runtime" \
  --device cuda \
  "${EXTRA_ARGS[@]}"
