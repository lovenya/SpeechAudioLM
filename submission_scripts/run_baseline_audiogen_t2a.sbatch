#!/usr/bin/env bash
#SBATCH --job-name=salm-audiogen-t2a
#SBATCH --account=rrg-ravanelm
#SBATCH --time=06:00:00
#SBATCH --cpus-per-task=8
#SBATCH --mem=48G
#SBATCH --gpus=h100_2g.20gb:1
#SBATCH --output=/scratch/lovenya/SpeechAudioLM/logs/slurm/%x-%j.out

set -euo pipefail

cd /lustre09/project/6090520/lovenya/SpeechAudioLM
module purge
module load StdEnv/2023 gcc cuda rust arrow python/3.10.13
source .venv-audiocraft/bin/activate

export SALM_SCRATCH="${SALM_SCRATCH:-/scratch/lovenya/SpeechAudioLM}"
export HF_HOME="/project/rrg-csubakan/lovenya/models/hf_cache"
export HF_HUB_CACHE="/project/rrg-csubakan/lovenya/models/hf_cache/transformers"
export TRANSFORMERS_CACHE="/project/rrg-csubakan/lovenya/models/hf_cache/transformers"
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export PYTHONDONTWRITEBYTECODE=1

mkdir -p "$SALM_SCRATCH/results/baselines" "$SALM_SCRATCH/logs/baselines" "$SALM_SCRATCH/logs/slurm"

python baseline.py \
  --model audiogen \
  --task T2A \
  --split test \
  --num-samples "${NUM_SAMPLES:--1}" \
  --output-dir "$SALM_SCRATCH/results/baselines" \
  --log-dir "$SALM_SCRATCH/logs/baselines" \
  --continue-on-error

