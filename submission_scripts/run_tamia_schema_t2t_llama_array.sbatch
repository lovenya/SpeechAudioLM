#!/usr/bin/env bash
#SBATCH --account=aip-csubakan
#SBATCH --nodes=1
#SBATCH --cpus-per-task=10
#SBATCH --mem=180G
#SBATCH --gpus=h100:4
#SBATCH --time=01:00:00
#SBATCH --array=0-5
#SBATCH --output=/scratch/l/lovenya/salm_migration_clean/logs/slurm/%x-%A_%a.out

set -euo pipefail

cd /scratch/l/lovenya/SpeechAudioLM
module --force purge
module load StdEnv/2023 gcc cuda/12.6 rust arrow/17.0.0 python/3.10.13

export SALM_SCRATCH="${SALM_SCRATCH:-/scratch/l/lovenya/salm_migration_clean}"
export HF_HOME="$SALM_SCRATCH/models/hf_cache"
export HF_HUB_CACHE="$HF_HOME"
export HUGGINGFACE_HUB_CACHE="$HF_HOME"
export TRANSFORMERS_CACHE="$HF_HOME"
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export PYTHONDONTWRITEBYTECODE=1
export TOKENIZERS_PARALLELISM=false
export RESOURCE_INTERVAL="${RESOURCE_INTERVAL:-30}"
export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
export NUMEXPR_NUM_THREADS=1
export MPLCONFIGDIR="${SLURM_TMPDIR:-/tmp}/$USER/matplotlib"
mkdir -p "$MPLCONFIGDIR" "$SALM_SCRATCH/logs/slurm"

RUN_ID="ttas_val"
BASELINE_ROOT="$SALM_SCRATCH/results/cascaded_full_validation/$RUN_ID"
INPUT_ROOT="$BASELINE_ROOT/stage_inputs/schema_v1"
LOG_ROOT="$SALM_SCRATCH/logs/cascaded_full_validation/$RUN_ID"
JOB_KEY="${SLURM_ARRAY_JOB_ID:-manual}_${SLURM_ARRAY_TASK_ID:-0}"
JOB_LOG_ROOT="$LOG_ROOT/lanes/${JOB_KEY}_schema_t2t_llama"
CONFIG_FILE="configs/baselines/cascaded_t2t_schema_v1.json"
STEPS_TEXT="${STEPS_TEXT:-64}"
mkdir -p "$JOB_LOG_ROOT"

source submission_scripts/lib_resource_monitor.sh
start_resource_monitor "tamia_schema_t2t_llama_${JOB_KEY}"
source "$SALM_SCRATCH/.venvs/models/.venv-cascade-text/bin/activate"

case "${SLURM_ARRAY_TASK_ID:-0}" in
  0)
    lane="full_a2s_afn_llama_t2t"
    input="$INPUT_ROOT/a2s_afn.jsonl"
    mode="audio_description_to_speech"
    total=2258
    total_shards=4
    shard_offset=0
    ;;
  1)
    lane="full_a2s_afn_llama_t2t"
    input="$INPUT_ROOT/a2s_afn.jsonl"
    mode="audio_description_to_speech"
    total=2258
    total_shards=4
    shard_offset=2
    ;;
  2)
    lane="full_a2s_qwencap_llama_t2t"
    input="$INPUT_ROOT/a2s_qwencap.jsonl"
    mode="audio_description_to_speech"
    total=2258
    total_shards=4
    shard_offset=0
    ;;
  3)
    lane="full_a2s_qwencap_llama_t2t"
    input="$INPUT_ROOT/a2s_qwencap.jsonl"
    mode="audio_description_to_speech"
    total=2258
    total_shards=4
    shard_offset=2
    ;;
  4)
    lane="full_s2a_whisper_llama_t2t"
    input="$INPUT_ROOT/s2a_whisper.jsonl"
    mode="speech_to_audio_description"
    total=2277
    total_shards=2
    shard_offset=0
    ;;
  5)
    lane="full_s2a_qwen3_asr_llama_t2t"
    input="$INPUT_ROOT/s2a_qwen3_asr.jsonl"
    mode="speech_to_audio_description"
    total=2277
    total_shards=2
    shard_offset=0
    ;;
  *)
    echo "invalid SLURM_ARRAY_TASK_ID=${SLURM_ARRAY_TASK_ID:-unset}" >&2
    exit 2
    ;;
esac

test -s "$input" || { echo "missing input: $input" >&2; exit 2; }
pids=()
names=()

run_shard() {
  local shard="$1" gpus="$2"
  local base=$((total / total_shards)) extra=$((total % total_shards)) count start
  count="$base"
  if (( shard < extra )); then
    count=$((base + 1))
    start=$((shard * (base + 1)))
  else
    start=$((extra * (base + 1) + (shard - extra) * base))
  fi
  local run_name="${lane}_schema_${JOB_KEY}_s${shard}"
  (
    set -euo pipefail
    export CUDA_VISIBLE_DEVICES="$gpus"
    echo "[$run_name] gpus=$gpus start=$start count=$count started=$(date -Is)"
    python baseline_cascaded.py \
      --pipeline T2T \
      --text-model llama3_3_70b_instruct \
      --text-mode "$mode" \
      --config-file "$CONFIG_FILE" \
      --split val \
      --split-file "$input" \
      --start-index "$start" \
      --num-samples "$count" \
      --steps "$STEPS_TEXT" \
      --output-dir "$BASELINE_ROOT" \
      --log-dir "$LOG_ROOT" \
      --run-name "$run_name" \
      --overwrite \
      --continue-on-error
    echo "[$run_name] finished=$(date -Is)"
  ) >"$JOB_LOG_ROOT/${run_name}.out" 2>&1 &
  pids+=("$!")
  names+=("$run_name")
}

run_shard "$shard_offset" "0,1"
run_shard "$((shard_offset + 1))" "2,3"
for i in "${!pids[@]}"; do
  wait "${pids[i]}" || { echo "FAIL ${names[i]} see $JOB_LOG_ROOT/${names[i]}.out" >&2; exit 1; }
  echo "PASS ${names[i]}"
done

