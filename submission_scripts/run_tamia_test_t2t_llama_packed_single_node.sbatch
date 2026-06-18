#!/usr/bin/env bash
#SBATCH --job-name=salm-test-t2t-one-node
#SBATCH --account=aip-ravanelm
#SBATCH --nodes=1
#SBATCH --cpus-per-task=24
#SBATCH --mem=180G
#SBATCH --gpus=h100:4
#SBATCH --time=01:30:00
#SBATCH --output=/scratch/l/lovenya/salm_migration_clean/logs/slurm/%x-%j.out

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
RUN_ID="${RUN_ID:?Set RUN_ID=yyyymmdd_hhmm so both T2T lanes write to the intended run roots}"
S2T_RUN_ID="${S2T_RUN_ID:-$RUN_ID}"
A2T_RUN_ID="${A2T_RUN_ID:-$RUN_ID}"
CONFIG_FILE="configs/baselines/cascaded_t2t_schema_v1.json"
TOTAL_S2A="${TOTAL_S2A:-2277}"
TOTAL_A2S="${TOTAL_A2S:-2277}"
EXPECTED_S2A_SUCCESS="${EXPECTED_S2A_SUCCESS:-2277}"
EXPECTED_A2S_SUCCESS="${EXPECTED_A2S_SUCCESS:-2254}"
TOTAL_SHARDS="${TOTAL_SHARDS:-4}"
STEPS_TEXT="${STEPS_TEXT:-64}"

S2A_INPUT="$SALM_RESULTS_ROOT/cascaded/stages/s2t/whisper/ttas/$SPLIT/$S2T_RUN_ID/manifest.jsonl"
A2S_INPUT="$SALM_RESULTS_ROOT/cascaded/stages/a2t/audio_flamingo_next/ttas/$SPLIT/$A2T_RUN_ID/manifest.jsonl"
S2A_RUN_DIR="$SALM_RESULTS_ROOT/cascaded/stages/t2t/s2a/llama3_3_70b_instruct/from-s2t/whisper/ttas/$SPLIT/$RUN_ID"
A2S_RUN_DIR="$SALM_RESULTS_ROOT/cascaded/stages/t2t/a2s/llama3_3_70b_instruct/from-a2t/audio_flamingo_next/ttas/$SPLIT/$RUN_ID"
LOG_ROOT="$SALM_SCRATCH/logs/cascaded_full_validation/$RUN_GROUP"
JOB_KEY="${SLURM_JOB_ID:-manual}"
JOB_LOG_ROOT="$LOG_ROOT/lanes/${JOB_KEY}_t2t_packed_single_node"

mkdir -p "$JOB_LOG_ROOT"
test -s "$S2A_INPUT" || { echo "missing S2A input: $S2A_INPUT" >&2; exit 2; }
test -s "$A2S_INPUT" || { echo "missing A2S input: $A2S_INPUT" >&2; exit 2; }
test ! -f "$S2A_RUN_DIR/manifest.jsonl" || { echo "Refusing to overwrite merged S2A T2T output: $S2A_RUN_DIR/manifest.jsonl" >&2; exit 2; }
test ! -f "$A2S_RUN_DIR/manifest.jsonl" || { echo "Refusing to overwrite merged A2S T2T output: $A2S_RUN_DIR/manifest.jsonl" >&2; exit 2; }
mkdir -p "$S2A_RUN_DIR" "$A2S_RUN_DIR"

source submission_scripts/lib_resource_monitor.sh
start_resource_monitor "tamia_test_t2t_packed_single_node_${JOB_KEY}"
source "$SALM_SCRATCH/.venvs/models/.venv-cascade-text/bin/activate"

run_s2a_shard() {
  local shard="$1"
  local name="s2a_whisper_llama.shard_${shard}"
  (
    set -euo pipefail
    export CUDA_VISIBLE_DEVICES="0,1"
    echo "[$name] gpus=$CUDA_VISIBLE_DEVICES run_id=$RUN_ID s2t_run_id=$S2T_RUN_ID started=$(date -Is)"
    python baseline_cascaded.py \
      --results-schema v1 \
      --pipeline T2T \
      --text-model llama3_3_70b_instruct \
      --text-mode speech_to_audio_description \
      --asr whisper \
      --config-file "$CONFIG_FILE" \
      --split "$SPLIT" \
      --split-file "$S2A_INPUT" \
      --start-index 0 \
      --num-samples "$TOTAL_S2A" \
      --shard-index "$shard" \
      --num-shards "$TOTAL_SHARDS" \
      --run-id "$RUN_ID" \
      --steps "$STEPS_TEXT" \
      --continue-on-error
    echo "[$name] finished=$(date -Is)"
  ) >"$JOB_LOG_ROOT/${name}.out" 2>&1 &
  LAST_PID="$!"
  LAST_NAME="$name"
}

run_a2s_shard() {
  local shard="$1"
  local name="a2s_afn_llama.shard_${shard}"
  (
    set -euo pipefail
    export CUDA_VISIBLE_DEVICES="2,3"
    echo "[$name] gpus=$CUDA_VISIBLE_DEVICES run_id=$RUN_ID a2t_run_id=$A2T_RUN_ID started=$(date -Is)"
    python baseline_cascaded.py \
      --results-schema v1 \
      --pipeline T2T \
      --text-model llama3_3_70b_instruct \
      --text-mode audio_description_to_speech \
      --captioner audio_flamingo_next \
      --config-file "$CONFIG_FILE" \
      --split "$SPLIT" \
      --split-file "$A2S_INPUT" \
      --start-index 0 \
      --num-samples "$TOTAL_A2S" \
      --shard-index "$shard" \
      --num-shards "$TOTAL_SHARDS" \
      --run-id "$RUN_ID" \
      --steps "$STEPS_TEXT" \
      --continue-on-error
    echo "[$name] finished=$(date -Is)"
  ) >"$JOB_LOG_ROOT/${name}.out" 2>&1 &
  LAST_PID="$!"
  LAST_NAME="$name"
}

for shard in $(seq 0 $((TOTAL_SHARDS - 1))); do
  echo "[wave $shard] started=$(date -Is)"
  run_s2a_shard "$shard"
  s2a_pid="$LAST_PID"
  s2a_name="$LAST_NAME"
  run_a2s_shard "$shard"
  a2s_pid="$LAST_PID"
  a2s_name="$LAST_NAME"
  wait "$s2a_pid" || { echo "FAIL $s2a_name see $JOB_LOG_ROOT/${s2a_name}.out" >&2; exit 1; }
  echo "PASS $s2a_name"
  wait "$a2s_pid" || { echo "FAIL $a2s_name see $JOB_LOG_ROOT/${a2s_name}.out" >&2; exit 1; }
  echo "PASS $a2s_name"
  echo "[wave $shard] finished=$(date -Is)"
done

echo "packed_t2t_all_shards_complete run_id=$RUN_ID total_shards=$TOTAL_SHARDS"
echo "merge later from a login node after auditing all shards:"
echo "python scripts/merge_cascaded_stage_shards.py --run-dir '$S2A_RUN_DIR' --expected-count '$EXPECTED_S2A_SUCCESS' --media-output none --keep-shards"
echo "python scripts/merge_cascaded_stage_shards.py --run-dir '$A2S_RUN_DIR' --expected-count '$EXPECTED_A2S_SUCCESS' --media-output none --keep-shards"
