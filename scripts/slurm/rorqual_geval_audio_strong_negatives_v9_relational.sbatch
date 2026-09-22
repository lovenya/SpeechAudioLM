#!/usr/bin/env bash
# V9 diagnostic for relational criteria over 120 matched/mismatched pairs.
# Output-only modality and perceptual-quality criteria are intentionally excluded.

#SBATCH --job-name=ttas-v9-relational
#SBATCH --account=rrg-ravanelm_gpu
#SBATCH --nodes=1
#SBATCH --gpus=h100:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=02:00:00
#SBATCH --array=0-3
#SBATCH --output=/scratch/lovenya/salm_migration_clean/logs/slurm/%x-%A_%a.out

set -euo pipefail

REPO=/scratch/lovenya/SpeechAudioLM
PYTHON=/scratch/lovenya/salm_migration_clean/.venvs/salm-qwen3-omni-env/bin/python
TARGET_RUN_ID=20260920_1852
MANIFEST_ROOT=/scratch/lovenya/salm_migration_clean/results/evals/target_manifests/dataset_pair/ttas/test/geval_audio_strong_negatives/${TARGET_RUN_ID}
CACHE_DIR=/scratch/lovenya/salm_migration_clean/cache/evaluations
TASK_INDEX="${SLURM_ARRAY_TASK_ID:?array task ID is required}"

case "${TASK_INDEX}" in
    0)
        STAGE=t2s
        DIRECTION=a2s
        PAIRING=matched
        CRITERIA=controlled_detail,expressive_delivery_fit
        EVAL_RUN_ID=20260922_0303
        MANIFEST=${MANIFEST_ROOT}/t2s_a2s_matched.jsonl
        ;;
    1)
        STAGE=t2s
        DIRECTION=a2s
        PAIRING=mismatched
        CRITERIA=controlled_detail,expressive_delivery_fit
        EVAL_RUN_ID=20260922_0304
        MANIFEST=${MANIFEST_ROOT}/t2s_a2s_mismatch.jsonl
        ;;
    2)
        STAGE=t2a
        DIRECTION=s2a
        PAIRING=matched
        CRITERIA=controlled_detail,contextual_acoustic_fit
        EVAL_RUN_ID=20260922_0305
        MANIFEST=${MANIFEST_ROOT}/t2a_s2a_matched.jsonl
        ;;
    3)
        STAGE=t2a
        DIRECTION=s2a
        PAIRING=mismatched
        CRITERIA=controlled_detail,contextual_acoustic_fit
        EVAL_RUN_ID=20260922_0306
        MANIFEST=${MANIFEST_ROOT}/t2a_s2a_mismatch.jsonl
        ;;
    *)
        echo "unexpected array task index: ${TASK_INDEX}" >&2
        exit 2
        ;;
esac

TEMP_EVAL_ROOT=/scratch/lovenya/salm_migration_clean/temporary/results/evals
RESULT_DIR=${TEMP_EVAL_ROOT}/stages/${DIRECTION}/${STAGE}/dataset_pair/ttas/test/${TARGET_RUN_ID}/geval_audio_plausibility/${EVAL_RUN_ID}

module --force purge
module load StdEnv/2023 gcc/12.3 cuda/12.6 python/3.12.4

export HF_HOME=/scratch/lovenya/salm_migration_clean/cache/metrics/huggingface
export HF_HUB_CACHE="${HF_HOME}"
export HUGGINGFACE_HUB_CACHE="${HF_HOME}"
export TRANSFORMERS_CACHE="${HF_HOME}"
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

cd "${REPO}"
[[ -x "${PYTHON}" ]] || { echo "missing Qwen3-Omni environment: ${PYTHON}" >&2; exit 2; }
[[ -s "${MANIFEST}" ]] || { echo "missing pair manifest: ${MANIFEST}" >&2; exit 2; }
[[ $(wc -l < "${MANIFEST}") -eq 120 ]] || { echo "expected 120 rows: ${MANIFEST}" >&2; exit 2; }
[[ ! -e "${RESULT_DIR}" ]] || { echo "refusing to overwrite existing result: ${RESULT_DIR}" >&2; exit 2; }
mkdir -p /scratch/lovenya/salm_migration_clean/logs/slurm

echo "job_id=${SLURM_JOB_ID:-unknown} task_index=${TASK_INDEX} direction=${DIRECTION} pairing=${PAIRING} criteria=${CRITERIA} host=$(hostname)"
echo "manifest=${MANIFEST}"
echo "result_dir=${RESULT_DIR}"
nvidia-smi --query-gpu=index,name,memory.total --format=csv,noheader

exec "${PYTHON}" evaluate.py \
    --model dataset_pair \
    --stage "${STAGE}" \
    --direction "${DIRECTION}" \
    --evaluated-run-id "${TARGET_RUN_ID}" \
    --eval-run-id "${EVAL_RUN_ID}" \
    --dataset ttas \
    --split test \
    --manifest "${MANIFEST}" \
    --metrics geval_audio_plausibility \
    --criteria "${CRITERIA}" \
    --temporary \
    --cache-dir "${CACHE_DIR}" \
    --device cuda
