#!/usr/bin/env bash
#SBATCH --job-name=salm-test-t2t-s2a-llama
#SBATCH --account=aip-ravanelm
#SBATCH --nodes=1
#SBATCH --cpus-per-task=24
#SBATCH --mem=180G
#SBATCH --gpus=h100:4
#SBATCH --time=01:00:00
#SBATCH --array=0-1
#SBATCH --output=/scratch/l/lovenya/salm_migration_clean/logs/slurm/%x-%A_%a.out

set -euo pipefail

cd /scratch/l/lovenya/SpeechAudioLM
module --force purge
module load StdEnv/2023 gcc cuda/12.6 rust arrow/17.0.0 python/3.10.13

export SALM_SCRATCH="${SALM_SCRATCH:-/scratch/l/lovenya/salm_migration_clean}"
export SALM_RESULTS_ROOT="$SALM_SCRATCH/results"
export SALM_LOGS_ROOT="$SALM_SCRATCH/logs"
export SALM_MODEL_ROOT="$SALM_SCRATCH/models"
export SALM_CACHE_ROOT="$SALM_SCRATCH/cache"
export SALM_DATASETS_ROOT="$SALM_SCRATCH/datasets"
export SALM_SOURCES_ROOT="$SALM_SCRATCH/code"
export HF_HOME="$SALM_SCRATCH/models/hf_cache"
export HF_HUB_CACHE="$HF_HOME"
export HUGGINGFACE_HUB_CACHE="$HF_HOME"
export TRANSFORMERS_CACHE="$HF_HOME"
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export PYTHONPATH="$PWD${PYTHONPATH:+:$PYTHONPATH}"
export PYTHONDONTWRITEBYTECODE=1
export TOKENIZERS_PARALLELISM=false
export RESOURCE_INTERVAL="${RESOURCE_INTERVAL:-300}"
export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
export NUMEXPR_NUM_THREADS=1
export MPLCONFIGDIR="${SLURM_TMPDIR:-/tmp}/$USER/matplotlib"

mkdir -p "$MPLCONFIGDIR" "$SALM_SCRATCH/logs/slurm"

RUN_GROUP="ttas_test"
SPLIT="test"
RUN_ID="${RUN_ID:?Set RUN_ID=yyyymmdd_hhmm so all T2T array tasks write to the same run root}"
S2T_RUN_ID="${S2T_RUN_ID:-$RUN_ID}"
LANE="full_s2a_whisper_llama_t2t"
INPUT="$SALM_RESULTS_ROOT/cascaded/stages/s2t/whisper/ttas/$SPLIT/$S2T_RUN_ID/manifest.jsonl"
LOG_ROOT="$SALM_SCRATCH/logs/cascaded_full_validation/$RUN_GROUP"
JOB_KEY="${SLURM_ARRAY_JOB_ID:-manual}_${SLURM_ARRAY_TASK_ID:-0}"
JOB_LOG_ROOT="$LOG_ROOT/lanes/${JOB_KEY}_t2t_s2a_whisper_llama"
CONFIG_FILE="configs/baselines/cascaded_t2t_schema_v1.json"
TOTAL="${TOTAL_SAMPLES:-2277}"
TOTAL_SHARDS="${TOTAL_SHARDS:-4}"
STEPS_TEXT="${STEPS_TEXT:-64}"
SHARD_OFFSET=$(( (${SLURM_ARRAY_TASK_ID:-0}) * 2 ))
RUN_DIR="$SALM_RESULTS_ROOT/cascaded/stages/t2t/s2a/llama3_3_70b_instruct/from-s2t/whisper/ttas/$SPLIT/$RUN_ID"
LOG_RUN_DIR="$SALM_LOGS_ROOT/cascaded/stages/t2t/s2a/llama3_3_70b_instruct/from-s2t/whisper/ttas/$SPLIT/$RUN_ID"

mkdir -p "$JOB_LOG_ROOT"
test -s "$INPUT" || { echo "missing input: $INPUT" >&2; exit 2; }
test ! -f "$RUN_DIR/manifest.jsonl" || { echo "Refusing to overwrite merged output: $RUN_DIR/manifest.jsonl" >&2; exit 2; }
mkdir -p "$RUN_DIR" "$LOG_RUN_DIR"

source submission_scripts/lib_resource_monitor.sh
start_resource_monitor "tamia_test_t2t_s2a_whisper_llama_${JOB_KEY}"
source "$SALM_SCRATCH/.venvs/models/.venv-cascade-text/bin/activate"

pids=()
names=()

run_shard() {
  local shard="$1" gpus="$2"
  local base=$((TOTAL / TOTAL_SHARDS)) extra=$((TOTAL % TOTAL_SHARDS)) count start
  count="$base"
  if (( shard < extra )); then
    count=$((base + 1))
    start=$((shard * (base + 1)))
  else
    start=$((extra * (base + 1) + (shard - extra) * base))
  fi
  local shard_name="${LANE}.shard_${shard}"
  (
    set -euo pipefail
    export CUDA_VISIBLE_DEVICES="$gpus"
    echo "[$shard_name] gpus=$gpus start=$start count=$count run_id=$LANE started=$(date -Is)"
    python baseline_cascaded.py \
      --results-schema v1 \
      --pipeline T2T \
      --text-model llama3_3_70b_instruct \
      --text-mode speech_to_audio_description \
      --asr whisper \
      --config-file "$CONFIG_FILE" \
      --split "$SPLIT" \
      --split-file "$INPUT" \
      --start-index 0 \
      --num-samples "$TOTAL" \
      --shard-index "$shard" \
      --num-shards "$TOTAL_SHARDS" \
      --run-id "$RUN_ID" \
      --steps "$STEPS_TEXT" \
      --continue-on-error
    echo "[$shard_name] finished=$(date -Is)"
  ) >"$JOB_LOG_ROOT/${shard_name}.out" 2>&1 &
  pids+=("$!")
  names+=("$shard_name")
}

run_shard "$SHARD_OFFSET" "0,1"
run_shard "$((SHARD_OFFSET + 1))" "2,3"

for i in "${!pids[@]}"; do
  wait "${pids[i]}" || { echo "FAIL ${names[i]} see $JOB_LOG_ROOT/${names[i]}.out" >&2; exit 1; }
  echo "PASS ${names[i]}"
done

echo "test_t2t_s2a_whisper_llama_partial_shards_complete lane=$LANE run_id=$RUN_ID run_dir=$RUN_DIR shards=${SHARD_OFFSET},$((SHARD_OFFSET + 1)) total_shards=$TOTAL_SHARDS"
echo "merge later from a login node after auditing all shards:"
echo "python scripts/merge_cascaded_stage_shards.py --run-dir '$RUN_DIR' --expected-count '$TOTAL' --media-output none"
