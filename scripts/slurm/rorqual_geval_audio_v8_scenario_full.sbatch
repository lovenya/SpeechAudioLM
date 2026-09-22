#!/usr/bin/env bash
# Full clean-v2 v8 scenario-plausibility G-Eval. One 456-row shard per H100.

#SBATCH --job-name=geval-v8-scenario
#SBATCH --account=rrg-csubakan_gpu
#SBATCH --nodes=1
#SBATCH --gpus=h100:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=02:00:00
#SBATCH --array=0-34
#SBATCH --output=/scratch/lovenya/salm_migration_clean/logs/slurm/%x-%A_%a.out

set -euo pipefail

REPO=/scratch/lovenya/SpeechAudioLM
PYTHON=/scratch/lovenya/salm_migration_clean/.venvs/salm-qwen3-omni-env/bin/python
RUN_GROUP=geval_audio_v8_scenario_20260921_1710
TASK_PLAN=/scratch/lovenya/salm_migration_clean/results/evals/shard_manifests/geval_audio_v8_scenario/${RUN_GROUP}/task_plan.json
LOG_ROOT=/scratch/lovenya/salm_migration_clean/logs/evals/${RUN_GROUP}
OUTPUT_ROOT=/scratch/lovenya/salm_migration_clean/results/evals
CACHE_DIR=/scratch/lovenya/salm_migration_clean/cache/evaluations

module --force purge
module load StdEnv/2023 gcc/12.3 cuda/12.6 python/3.12.4

export HF_HOME=/scratch/lovenya/salm_migration_clean/cache/metrics/huggingface
export HF_HUB_CACHE="$HF_HOME"
export HUGGINGFACE_HUB_CACHE="$HF_HOME"
export TRANSFORMERS_CACHE="$HF_HOME"
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export TOKENIZERS_PARALLELISM=false
export PYTHONNOUSERSITE=1
export PYTHONDONTWRITEBYTECODE=1
export PYTHONUNBUFFERED=1
export OMP_NUM_THREADS=2
export MKL_NUM_THREADS=2
export NUMEXPR_NUM_THREADS=2
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

cd "$REPO"
[[ -x "$PYTHON" ]] || { echo "missing Qwen3-Omni environment: $PYTHON" >&2; exit 2; }
[[ -s "$TASK_PLAN" ]] || { echo "missing task plan: $TASK_PLAN" >&2; exit 2; }
mkdir -p "$LOG_ROOT" /scratch/lovenya/salm_migration_clean/logs/slurm

echo "job_id=${SLURM_JOB_ID:-unknown} task_index=${SLURM_ARRAY_TASK_ID:?array task ID is required} host=$(hostname)"
nvidia-smi --query-gpu=index,name,memory.total --format=csv,noheader

exec "$PYTHON" scripts/run_geval_audio_v6_plausibility_task.py \
    --task-plan "$TASK_PLAN" \
    --log-root "$LOG_ROOT" \
    --repo-root "$REPO" \
    --output-root "$OUTPUT_ROOT" \
    --cache-dir "$CACHE_DIR" \
    --device cuda \
    --allow-existing-outputs
