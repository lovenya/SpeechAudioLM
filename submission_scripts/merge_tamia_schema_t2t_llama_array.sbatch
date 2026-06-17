#!/usr/bin/env bash
#SBATCH --account=aip-csubakan
#SBATCH --nodes=1
#SBATCH --cpus-per-task=2
#SBATCH --mem=16G
#SBATCH --time=00:20:00
#SBATCH --array=0-3
#SBATCH --output=/scratch/l/lovenya/salm_migration_clean/logs/slurm/%x-%A_%a.out

set -euo pipefail

cd /scratch/l/lovenya/SpeechAudioLM
module --force purge
module load StdEnv/2023 python/3.10.13

export SALM_SCRATCH="${SALM_SCRATCH:-/scratch/l/lovenya/salm_migration_clean}"
export SALM_RESULTS_ROOT="$SALM_SCRATCH/results"
export SALM_LOGS_ROOT="$SALM_SCRATCH/logs"
export PYTHONDONTWRITEBYTECODE=1
mkdir -p "$SALM_SCRATCH/logs/slurm"

case "${SLURM_ARRAY_TASK_ID:-0}" in
  0)
    lane="full_a2s_afn_llama_t2t"
    direction="a2s"
    upstream_stage="a2t"
    upstream_model="audio_flamingo_next"
    total=2258
    ;;
  1)
    lane="full_a2s_qwencap_llama_t2t"
    direction="a2s"
    upstream_stage="a2t"
    upstream_model="qwen3_omni_captioner"
    total=2258
    ;;
  2)
    lane="full_s2a_whisper_llama_t2t"
    direction="s2a"
    upstream_stage="s2t"
    upstream_model="whisper"
    total=2277
    ;;
  3)
    lane="full_s2a_qwen3_asr_llama_t2t"
    direction="s2a"
    upstream_stage="s2t"
    upstream_model="qwen3_asr"
    total=2277
    ;;
  *)
    echo "invalid SLURM_ARRAY_TASK_ID=${SLURM_ARRAY_TASK_ID:-unset}" >&2
    exit 2
    ;;
esac

run_dir="$SALM_RESULTS_ROOT/cascaded/stages/t2t/$direction/llama3_3_70b_instruct/from-$upstream_stage/$upstream_model/ttas/val/$lane"
python scripts/merge_cascaded_stage_shards.py \
  --run-dir "$run_dir" \
  --expected-count "$total" \
  --media-output none
