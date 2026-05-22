#!/usr/bin/env bash
#SBATCH --job-name=salm-audio-omni
#SBATCH --account=rrg-ravanelm
#SBATCH --time=08:00:00
#SBATCH --cpus-per-task=4
#SBATCH --mem=96G
#SBATCH --gpus=h100_3g.40gb:1
#SBATCH --array=0-5
#SBATCH --output=/scratch/lovenya/SpeechAudioLM/logs/slurm/%x-%A_%a.out

set -euo pipefail

TASKS=(T2A T2S A2T S2T S2A A2S)
TASK="${TASKS[$SLURM_ARRAY_TASK_ID]}"

cd /lustre09/project/6090520/lovenya/SpeechAudioLM
module --force purge
module load StdEnv/2023 gcc cuda python/3.10.13
source .venv-audio-omni/bin/activate

export SALM_SCRATCH="${SALM_SCRATCH:-/scratch/lovenya/SpeechAudioLM}"
export QWEN_OMNI_MODEL_PATH="/project/rrg-csubakan/lovenya/models/qwen2.5-omni-3b"
export HF_HOME="/project/rrg-csubakan/lovenya/models/hf_cache"
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export PYTHONDONTWRITEBYTECODE=1

MODEL="audio_omni"
START_INDEX="${START_INDEX:-0}"
NUM_SAMPLES_VALUE="${NUM_SAMPLES:--1}"
BASELINE_ROOT="$SALM_SCRATCH/results/baselines"
LOG_ROOT="$SALM_SCRATCH/logs/baselines"
if [ "${SHARDED:-0}" = "1" ]; then
  SHARD_LABEL="${MODEL}_${TASK}_start${START_INDEX}_n${NUM_SAMPLES_VALUE}"
  BASELINE_ROOT="$SALM_SCRATCH/results/baselines_shards/$SHARD_LABEL"
  LOG_ROOT="$SALM_SCRATCH/logs/baseline_shards/$SHARD_LABEL"
fi

mkdir -p "$BASELINE_ROOT" "$LOG_ROOT" "$SALM_SCRATCH/logs/slurm"
source submission_scripts/lib_resource_monitor.sh
start_resource_monitor "baseline_${MODEL}_${TASK}_start${START_INDEX}_n${NUM_SAMPLES_VALUE}"

python baseline.py \
  --model "$MODEL" \
  --task "$TASK" \
  --split test \
  --start-index "$START_INDEX" \
  --num-samples "$NUM_SAMPLES_VALUE" \
  --duration "${DURATION:-10}" \
  --output-dir "$BASELINE_ROOT" \
  --log-dir "$LOG_ROOT" \
  --continue-on-error
