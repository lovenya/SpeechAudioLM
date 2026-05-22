#!/usr/bin/env bash
#SBATCH --job-name=salm-audio-omni
#SBATCH --account=rrg-ravanelm
#SBATCH --time=08:00:00
#SBATCH --cpus-per-task=8
#SBATCH --mem=96G
#SBATCH --gpus=h100_3g.40gb:1
#SBATCH --array=0-5
#SBATCH --output=/scratch/lovenya/SpeechAudioLM/logs/slurm/%x-%A_%a.out

set -euo pipefail

TASKS=(T2A T2S A2T S2T S2A A2S)
TASK="${TASKS[$SLURM_ARRAY_TASK_ID]}"

cd /lustre09/project/6090520/lovenya/SpeechAudioLM
module purge
module load StdEnv/2023 gcc cuda python/3.10.13
source .venv-audio-omni/bin/activate

export SALM_SCRATCH="${SALM_SCRATCH:-/scratch/lovenya/SpeechAudioLM}"
export QWEN_OMNI_MODEL_PATH="/project/rrg-csubakan/lovenya/models/qwen2.5-omni-3b"
export HF_HOME="/project/rrg-csubakan/lovenya/models/hf_cache"
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export PYTHONDONTWRITEBYTECODE=1

mkdir -p "$SALM_SCRATCH/results/baselines" "$SALM_SCRATCH/logs/baselines" "$SALM_SCRATCH/logs/slurm"

python baseline.py \
  --model audio_omni \
  --task "$TASK" \
  --split test \
  --num-samples "${NUM_SAMPLES:--1}" \
  --duration "${DURATION:-10}" \
  --output-dir "$SALM_SCRATCH/results/baselines" \
  --log-dir "$SALM_SCRATCH/logs/baselines" \
  --continue-on-error

