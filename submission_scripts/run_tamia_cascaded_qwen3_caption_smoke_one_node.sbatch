#!/usr/bin/env bash
#SBATCH --job-name=salm-qwen3-cap-smoke
#SBATCH --account=aip-csubakan
#SBATCH --time=00:30:00
#SBATCH --nodes=1
#SBATCH --cpus-per-task=48
#SBATCH --mem=0
#SBATCH --gpus=h100:4
#SBATCH --output=/scratch/l/lovenya/salm_migration_clean/logs/slurm/%x-%j.out

set -euo pipefail

cd /scratch/l/lovenya/SpeechAudioLM
module --force purge
module load StdEnv/2023 gcc cuda rust arrow python/3.10.13

export SALM_SCRATCH="${SALM_SCRATCH:-/scratch/l/lovenya/salm_migration_clean}"
export HF_HOME="$SALM_SCRATCH/models/hf_cache"
export HF_HUB_CACHE="$HF_HOME"
export HUGGINGFACE_HUB_CACHE="$HF_HOME"
export TRANSFORMERS_CACHE="$HF_HOME"
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export PYTHONDONTWRITEBYTECODE=1
export TOKENIZERS_PARALLELISM=false
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-8}"
export MKL_NUM_THREADS="${MKL_NUM_THREADS:-8}"
export NUMEXPR_NUM_THREADS="${NUMEXPR_NUM_THREADS:-8}"

START_INDEX="${START_INDEX:-0}"
NUM_SAMPLES_VALUE="${NUM_SAMPLES:-1}"
STEPS="${STEPS:-16}"
CAPTION_CUDA_VISIBLE_DEVICES="${CAPTION_CUDA_VISIBLE_DEVICES:-}"
BASELINE_ROOT="$SALM_SCRATCH/results/baselines_cascaded_qwen3_caption_smoke"
LOG_ROOT="$SALM_SCRATCH/logs/baselines_cascaded_qwen3_caption_smoke"
JOB_LOG_ROOT="$SALM_SCRATCH/logs/cascaded_qwen3_caption_smokes/${SLURM_JOB_ID:-manual}"

mkdir -p "$BASELINE_ROOT" "$LOG_ROOT" "$JOB_LOG_ROOT" "$SALM_SCRATCH/logs/slurm"

source submission_scripts/lib_resource_monitor.sh
start_resource_monitor "tamia_cascaded_qwen3_caption_smoke_one_node"

source "$SALM_SCRATCH/.venvs/models/.venv-cascade-caption/bin/activate"
if [ -n "$CAPTION_CUDA_VISIBLE_DEVICES" ]; then
  export CUDA_VISIBLE_DEVICES="$CAPTION_CUDA_VISIBLE_DEVICES"
fi
echo "[cascaded_a2t_qwen3_omni_captioner] host=$(hostname) cuda_visible_devices=${CUDA_VISIBLE_DEVICES:-all} started=$(date -Is)"
python baseline_cascaded.py \
  --pipeline A2T \
  --captioner qwen3_omni_captioner \
  --split test \
  --start-index "$START_INDEX" \
  --num-samples "$NUM_SAMPLES_VALUE" \
  --steps "$STEPS" \
  --output-dir "$BASELINE_ROOT" \
  --log-dir "$LOG_ROOT" \
  --overwrite
echo "[cascaded_a2t_qwen3_omni_captioner] finished=$(date -Is)"

echo "Generated cascaded Qwen3 caption smoke outputs: $BASELINE_ROOT"
find "$BASELINE_ROOT" -name manifest.jsonl -print
