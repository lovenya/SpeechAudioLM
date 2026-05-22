#!/usr/bin/env bash
#SBATCH --job-name=salm-qwen3-a2s
#SBATCH --account=rrg-ravanelm
#SBATCH --time=08:00:00
#SBATCH --cpus-per-task=8
#SBATCH --mem=160G
#SBATCH --gpus=h100:1
#SBATCH --output=/scratch/lovenya/SpeechAudioLM/logs/slurm/%x-%j.out

set -euo pipefail

cd /lustre09/project/6090520/lovenya/SpeechAudioLM
module purge
module load StdEnv/2023 gcc cuda python/3.10.13
source .venv-qwen3-omni/bin/activate

export SALM_SCRATCH="${SALM_SCRATCH:-/scratch/lovenya/SpeechAudioLM}"
export HF_HOME="/project/rrg-csubakan/lovenya/models/hf_cache"
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export PYTHONDONTWRITEBYTECODE=1

mkdir -p "$SALM_SCRATCH/results/baselines" "$SALM_SCRATCH/logs/baselines" "$SALM_SCRATCH/logs/slurm"

python baseline.py \
  --model qwen3_omni \
  --task A2S \
  --split test \
  --prompt-field speech_text \
  --num-samples "${NUM_SAMPLES:--1}" \
  --duration "${DURATION:-10}" \
  --output-dir "$SALM_SCRATCH/results/baselines" \
  --log-dir "$SALM_SCRATCH/logs/baselines" \
  --continue-on-error

