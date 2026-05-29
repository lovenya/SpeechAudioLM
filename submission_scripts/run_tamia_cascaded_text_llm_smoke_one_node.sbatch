#!/usr/bin/env bash
#SBATCH --job-name=salm-text-llm-smoke
#SBATCH --account=aip-csubakan
#SBATCH --time=00:45:00
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

TEXT_MODEL="${TEXT_MODEL:-qwen3_32b}"
START_INDEX="${START_INDEX:-0}"
NUM_SAMPLES_VALUE="${NUM_SAMPLES:-1}"
STEPS="${STEPS:-32}"
BASELINE_ROOT="$SALM_SCRATCH/results/baselines_cascaded_text_llm_smoke"
LOG_ROOT="$SALM_SCRATCH/logs/baselines_cascaded_text_llm_smoke"
JOB_LOG_ROOT="$SALM_SCRATCH/logs/cascaded_text_llm_smokes/${SLURM_JOB_ID:-manual}"

mkdir -p "$BASELINE_ROOT" "$LOG_ROOT" "$JOB_LOG_ROOT" "$SALM_SCRATCH/logs/slurm"

case "$TEXT_MODEL" in
  qwen3_32b|llama3_3_70b_instruct) ;;
  *)
    echo "Unsupported TEXT_MODEL=$TEXT_MODEL; use qwen3_32b or llama3_3_70b_instruct" >&2
    exit 2
    ;;
esac

source submission_scripts/lib_resource_monitor.sh
start_resource_monitor "tamia_cascaded_text_llm_smoke_one_node_${TEXT_MODEL}"

source "$SALM_SCRATCH/.venvs/models/.venv-cascade-text/bin/activate"
echo "[cascaded_t2t_${TEXT_MODEL}] host=$(hostname) cuda_visible_devices=${CUDA_VISIBLE_DEVICES:-all} started=$(date -Is)"
python baseline_cascaded.py \
  --pipeline T2T \
  --text-model "$TEXT_MODEL" \
  --text-mode speech_to_audio_description \
  --split test \
  --start-index "$START_INDEX" \
  --num-samples "$NUM_SAMPLES_VALUE" \
  --steps "$STEPS" \
  --output-dir "$BASELINE_ROOT" \
  --log-dir "$LOG_ROOT" \
  --overwrite
echo "[cascaded_t2t_${TEXT_MODEL}] finished=$(date -Is)"

echo "Generated cascaded text LLM smoke outputs: $BASELINE_ROOT"
find "$BASELINE_ROOT" -name manifest.jsonl -print
