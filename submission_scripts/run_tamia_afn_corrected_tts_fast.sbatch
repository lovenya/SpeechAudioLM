#!/usr/bin/env bash
#SBATCH --account=aip-csubakan
#SBATCH --nodes=1
#SBATCH --cpus-per-task=32
#SBATCH --mem=256G
#SBATCH --gpus=h100:4
#SBATCH --time=00:30:00
#SBATCH --output=/scratch/l/lovenya/salm_migration_clean/logs/slurm/%x-%j.out

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

RUN_ID="${RUN_ID:-ttas_val}"
TOTAL_A2S="${TOTAL_A2S:-2258}"
TEXT_MODEL_KIND="${TEXT_MODEL_KIND:?TEXT_MODEL_KIND must be qwen32 or llama}"
TTS_SHARDS_PER_BACKEND="${TTS_SHARDS_PER_BACKEND:-16}"
DURATION="${DURATION:-10}"
STEPS_AUDIO="${STEPS_AUDIO:-12}"
BASELINE_ROOT="$SALM_SCRATCH/results/cascaded_full_validation/$RUN_ID"
LOG_ROOT="$SALM_SCRATCH/logs/cascaded_full_validation/$RUN_ID"
JOB_LOG_ROOT="$LOG_ROOT/lanes/${SLURM_JOB_ID:-manual}_afn_corrected_tts_${TEXT_MODEL_KIND}"
SPLIT_ROOT="$BASELINE_ROOT/intermediate_splits"
METADATA_ROOT="$BASELINE_ROOT/metadata"
mkdir -p "$JOB_LOG_ROOT" "$SPLIT_ROOT" "$METADATA_ROOT"

source submission_scripts/lib_resource_monitor.sh
start_resource_monitor "tamia_cascaded_${RUN_ID}_afn_corrected_tts_fast_${TEXT_MODEL_KIND}"

if [ "$TEXT_MODEL_KIND" = "qwen32" ]; then
  TEXT_MODEL="qwen3_32b"
  T2T_LANE="full_a2s_afn_qwen32_t2t"
  F5_CANONICAL="full_a2s_afn_qwen32_f5_tts"
  COSY_CANONICAL="full_a2s_afn_qwen32_cosyvoice3"
elif [ "$TEXT_MODEL_KIND" = "llama" ]; then
  TEXT_MODEL="llama3_3_70b_instruct"
  T2T_LANE="full_a2s_afn_llama_t2t"
  F5_CANONICAL="full_a2s_afn_llama_f5_tts"
  COSY_CANONICAL="full_a2s_afn_llama_cosyvoice3"
else
  echo "unknown TEXT_MODEL_KIND=$TEXT_MODEL_KIND" >&2
  exit 2
fi

cat >"$METADATA_ROOT/afn_corrected_tts_${TEXT_MODEL_KIND}_${SLURM_JOB_ID:-manual}.json" <<META
{"job_id":"${SLURM_JOB_ID:-manual}","stage":"afn_corrected_tts","run_id":"$RUN_ID","text_model_kind":"$TEXT_MODEL_KIND","text_model":"$TEXT_MODEL","started_at":"$(date -Is)","requested":{"nodes":1,"gpus":4,"cpus":32,"memory_gb":256,"walltime_minutes":30},"tts_shards_per_backend":$TTS_SHARDS_PER_BACKEND,"total_a2s":$TOTAL_A2S}
META

if [ "$TEXT_MODEL_KIND" = "llama" ]; then
  merge_args=()
  for shard in $(seq 0 7); do
    merge_args+=(--input-dir "$BASELINE_ROOT/cascaded/T2T/full_a2s_afn_llama_t2t_corrected_s${shard}/ttas/val")
  done
  python scripts/merge_cascaded_shards.py \
    "${merge_args[@]}" \
    --output-dir "$BASELINE_ROOT/cascaded/T2T/full_a2s_afn_llama_t2t/ttas/val" \
    --lane full_a2s_afn_llama_t2t \
    --expected-count "$TOTAL_A2S" \
    --artifact-subdir text \
    --backup-label "afn_corrected_llama_${SLURM_JOB_ID:-manual}"
fi

T2T_MANIFEST="$BASELINE_ROOT/cascaded/T2T/$T2T_LANE/ttas/val/manifest.jsonl"
F5_SPLIT="$SPLIT_ROOT/a2s_afn_corrected_${TEXT_MODEL_KIND}_to_f5.jsonl"
COSY_SPLIT="$SPLIT_ROOT/a2s_afn_corrected_${TEXT_MODEL_KIND}_to_cosy.jsonl"
python scripts/build_cascaded_stage_inputs.py t2t-to-final --pipeline A2S --upstream-backend audio_flamingo_next --text-model "$TEXT_MODEL" --final-backend f5_tts --manifest "$T2T_MANIFEST" --output "$F5_SPLIT" --expected-count "$TOTAL_A2S" --allow-partial
python scripts/build_cascaded_stage_inputs.py t2t-to-final --pipeline A2S --upstream-backend audio_flamingo_next --text-model "$TEXT_MODEL" --final-backend cosyvoice3 --manifest "$T2T_MANIFEST" --output "$COSY_SPLIT" --expected-count "$TOTAL_A2S" --allow-partial

pids=()
names=()

run_tts_shard() {
  local backend="$1"
  local split_file="$2"
  local canonical="$3"
  local shard="$4"
  local count="$5"
  local start="$6"
  local gpu="$7"
  local venv="$8"
  local run_name="${canonical}_corrected_${SLURM_JOB_ID:-manual}_s${shard}"
  (
    set -euo pipefail
    export CUDA_VISIBLE_DEVICES="$gpu"
    source "$venv/bin/activate"
    echo "[${backend}_s${shard}] gpu=$gpu start=$start count=$count started=$(date -Is)"
    python baseline_cascaded.py \
      --pipeline T2S \
      --tts "$backend" \
      --split val \
      --split-file "$split_file" \
      --start-index "$start" \
      --num-samples "$count" \
      --duration "$DURATION" \
      --steps "$STEPS_AUDIO" \
      --output-dir "$BASELINE_ROOT" \
      --log-dir "$LOG_ROOT" \
      --run-name "$run_name" \
      --overwrite \
      --continue-on-error
    echo "[${backend}_s${shard}] finished=$(date -Is)"
  ) >"$JOB_LOG_ROOT/${backend}_s${shard}.out" 2>&1 &
  pids+=("$!")
  names+=("${backend}_s${shard}")
}

launch_backend() {
  local backend="$1"
  local split_file="$2"
  local canonical="$3"
  local venv="$4"
  local total="$5"
  local base=$((total / TTS_SHARDS_PER_BACKEND))
  local extra=$((total % TTS_SHARDS_PER_BACKEND))
  for shard in $(seq 0 $((TTS_SHARDS_PER_BACKEND - 1))); do
    local count="$base"
    local start
    if (( shard < extra )); then
      count=$((base + 1))
      start=$((shard * (base + 1)))
    else
      start=$((extra * (base + 1) + (shard - extra) * base))
    fi
    gpu=$((shard % 4))
    run_tts_shard "$backend" "$split_file" "$canonical" "$shard" "$count" "$start" "$gpu" "$venv"
  done
}

COSY_COUNT=$(wc -l < "$COSY_SPLIT")
F5_COUNT=$(wc -l < "$F5_SPLIT")
if (( COSY_COUNT <= 0 || F5_COUNT <= 0 )); then
  echo "empty final-stage input: cosy=$COSY_COUNT f5=$F5_COUNT" >&2
  exit 2
fi
launch_backend cosyvoice3 "$COSY_SPLIT" "$COSY_CANONICAL" "$SALM_SCRATCH/.venvs/models/.venv-cascade-cosyvoice3-py310" "$COSY_COUNT"
launch_backend f5_tts "$F5_SPLIT" "$F5_CANONICAL" "$SALM_SCRATCH/.venvs/models/.venv-cascade-f5-tts" "$F5_COUNT"

for i in "${!pids[@]}"; do
  wait "${pids[$i]}" || { echo "FAIL ${names[$i]} see $JOB_LOG_ROOT/${names[$i]}.out" >&2; exit 1; }
  echo "PASS ${names[$i]}"
done

for canonical in "$F5_CANONICAL" "$COSY_CANONICAL"; do
  if [ "$canonical" = "$F5_CANONICAL" ]; then
    expected_count="$F5_COUNT"
  else
    expected_count="$COSY_COUNT"
  fi
  merge_args=()
  for shard in $(seq 0 $((TTS_SHARDS_PER_BACKEND - 1))); do
    merge_args+=(--input-dir "$BASELINE_ROOT/cascaded/T2S/${canonical}_corrected_${SLURM_JOB_ID:-manual}_s${shard}/ttas/val")
  done
  python scripts/merge_cascaded_shards.py \
    "${merge_args[@]}" \
    --output-dir "$BASELINE_ROOT/cascaded/T2S/${canonical}/ttas/val" \
    --lane "$canonical" \
    --expected-count "$expected_count" \
    --artifact-subdir audio \
    --backup-label "afn_corrected_tts_${SLURM_JOB_ID:-manual}"
done
