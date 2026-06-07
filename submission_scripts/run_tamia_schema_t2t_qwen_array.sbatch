#!/usr/bin/env bash
#SBATCH --account=aip-ravanelm
#SBATCH --nodes=1
#SBATCH --cpus-per-task=10
#SBATCH --mem=160G
#SBATCH --gpus=h100:4
#SBATCH --time=01:00:00
#SBATCH --array=0-3
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
JOB_LOG_ROOT="$LOG_ROOT/lanes/${JOB_KEY}_schema_t2t_qwen"
CONFIG_FILE="configs/baselines/cascaded_t2t_schema_v1.json"
STEPS_TEXT="${STEPS_TEXT:-64}"
mkdir -p "$JOB_LOG_ROOT"

source submission_scripts/lib_resource_monitor.sh
start_resource_monitor "tamia_schema_t2t_qwen_${JOB_KEY}"
source "$SALM_SCRATCH/.venvs/models/.venv-cascade-text/bin/activate"

pids=()
names=()
lanes=()
inputs=()
modes=()
totals=()
shard_counts=()

case "${SLURM_ARRAY_TASK_ID:-0}" in
  0)
    lanes=("full_a2s_qwencap_qwen32_t2t")
    inputs=("$INPUT_ROOT/a2s_qwencap.jsonl")
    modes=("audio_description_to_speech")
    totals=(2258)
    shard_counts=(4)
    ;;
  1)
    lanes=("full_a2s_afn_qwen32_t2t")
    inputs=("$INPUT_ROOT/a2s_afn.jsonl")
    modes=("audio_description_to_speech")
    totals=(2258)
    shard_counts=(4)
    ;;
  2)
    lanes=("full_s2a_whisper_qwen32_t2t")
    inputs=("$INPUT_ROOT/s2a_whisper.jsonl")
    modes=("speech_to_audio_description")
    totals=(2277)
    shard_counts=(4)
    ;;
  3)
    lanes=("full_s2a_qwen3_asr_qwen32_t2t")
    inputs=("$INPUT_ROOT/s2a_qwen3_asr.jsonl")
    modes=("speech_to_audio_description")
    totals=(2277)
    shard_counts=(4)
    ;;
  *)
    echo "invalid SLURM_ARRAY_TASK_ID=${SLURM_ARRAY_TASK_ID:-unset}" >&2
    exit 2
    ;;
esac

for input in "${inputs[@]}"; do
  test -s "$input" || { echo "missing input: $input" >&2; exit 2; }
done

run_shard() {
  local lane="$1" input="$2" mode="$3" total="$4" shard="$5" num_shards="$6" gpu="$7"
  local base=$((total / num_shards)) extra=$((total % num_shards)) count start
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
    export CUDA_VISIBLE_DEVICES="$gpu"
    echo "[$run_name] gpu=$gpu start=$start count=$count started=$(date -Is)"
    python baseline_cascaded.py \
      --pipeline T2T \
      --text-model qwen3_32b \
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

gpu=0
for lane_index in "${!lanes[@]}"; do
  for ((shard=0; shard<shard_counts[lane_index]; shard++)); do
    run_shard "${lanes[lane_index]}" "${inputs[lane_index]}" "${modes[lane_index]}" \
      "${totals[lane_index]}" "$shard" "${shard_counts[lane_index]}" "$gpu"
    gpu=$((gpu + 1))
  done
done
if (( gpu != 4 )); then
  echo "packing error: assigned $gpu GPUs, expected 4" >&2
  exit 2
fi

for i in "${!pids[@]}"; do
  wait "${pids[i]}" || { echo "FAIL ${names[i]} see $JOB_LOG_ROOT/${names[i]}.out" >&2; exit 1; }
  echo "PASS ${names[i]}"
done

for lane_index in "${!lanes[@]}"; do
  merge_args=()
  for ((shard=0; shard<shard_counts[lane_index]; shard++)); do
    merge_args+=(--input-dir "$BASELINE_ROOT/cascaded/T2T/${lanes[lane_index]}_schema_${JOB_KEY}_s${shard}/ttas/val")
  done
  python scripts/merge_cascaded_shards.py \
    "${merge_args[@]}" \
    --output-dir "$BASELINE_ROOT/cascaded/T2T/${lanes[lane_index]}/ttas/val" \
    --lane "${lanes[lane_index]}" \
    --expected-count "${totals[lane_index]}" \
    --artifact-subdir text \
    --backup-label "schema_${JOB_KEY}"
done
