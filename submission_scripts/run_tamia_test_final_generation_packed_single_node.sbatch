#!/usr/bin/env bash
#SBATCH --job-name=salm-test-finalgen-one-node
#SBATCH --account=aip-ravanelm
#SBATCH --nodes=1
#SBATCH --cpus-per-task=32
#SBATCH --mem=240G
#SBATCH --gpus=h100:4
#SBATCH --time=04:00:00
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
RUN_ID="${RUN_ID:?Set RUN_ID=yyyymmdd_hhmm for final generation outputs}"
T2T_RUN_ID="${T2T_RUN_ID:-20260618_0001}"
TEXT_MODEL="llama3_3_70b_instruct"
T2A_BACKEND="${T2A_BACKEND:-audio_omni}"
T2S_BACKEND="${T2S_BACKEND:-cosyvoice3}"
TOTAL_T2A="${TOTAL_T2A:-2277}"
TOTAL_T2S="${TOTAL_T2S:-2254}"
T2A_SHARDS="${T2A_SHARDS:-4}"
T2S_SHARDS="${T2S_SHARDS:-4}"
T2A_MAX_PARALLEL="${T2A_MAX_PARALLEL:-4}"
T2S_MAX_PARALLEL="${T2S_MAX_PARALLEL:-4}"
T2A_STEPS="${T2A_STEPS:-25}"
T2A_DURATION="${T2A_DURATION:-10}"
T2S_STEPS="${T2S_STEPS:-32}"
T2S_DURATION="${T2S_DURATION:-0}"
COSYVOICE_REFERENCE_BANK="${COSYVOICE_REFERENCE_BANK:-configs/tts_reference_banks/cosyvoice3_train_25.json}"

S2A_T2T_INPUT="$SALM_RESULTS_ROOT/cascaded/stages/t2t/s2a/$TEXT_MODEL/from-s2t/whisper/ttas/$SPLIT/$T2T_RUN_ID/manifest.jsonl"
A2S_T2T_INPUT="$SALM_RESULTS_ROOT/cascaded/stages/t2t/a2s/$TEXT_MODEL/from-a2t/audio_flamingo_next/ttas/$SPLIT/$T2T_RUN_ID/manifest.jsonl"
T2A_RUN_DIR="$SALM_RESULTS_ROOT/cascaded/stages/t2a/$T2A_BACKEND/from-t2t/$TEXT_MODEL/ttas/$SPLIT/$RUN_ID"
T2S_RUN_DIR="$SALM_RESULTS_ROOT/cascaded/stages/t2s/$T2S_BACKEND/from-t2t/$TEXT_MODEL/ttas/$SPLIT/$RUN_ID"
LOG_ROOT="$SALM_SCRATCH/logs/cascaded_full_validation/$RUN_GROUP"
JOB_KEY="${SLURM_JOB_ID:-manual}"
JOB_LOG_ROOT="$LOG_ROOT/lanes/${JOB_KEY}_final_generation_packed"

mkdir -p "$JOB_LOG_ROOT"
test -s "$S2A_T2T_INPUT" || { echo "missing S2A T2T input: $S2A_T2T_INPUT" >&2; exit 2; }
test -s "$A2S_T2T_INPUT" || { echo "missing A2S T2T input: $A2S_T2T_INPUT" >&2; exit 2; }
test -s "$COSYVOICE_REFERENCE_BANK" || { echo "missing CosyVoice reference bank: $COSYVOICE_REFERENCE_BANK" >&2; exit 2; }
test ! -e "$T2A_RUN_DIR" || { echo "Refusing existing T2A run dir: $T2A_RUN_DIR" >&2; exit 2; }
test ! -e "$T2S_RUN_DIR" || { echo "Refusing existing T2S run dir: $T2S_RUN_DIR" >&2; exit 2; }
mkdir -p "$T2A_RUN_DIR" "$T2S_RUN_DIR"

source submission_scripts/lib_resource_monitor.sh
start_resource_monitor "tamia_test_final_generation_packed_${JOB_KEY}"
trap 'stop_resource_monitor || true' EXIT

run_packed_shards() {
  local stage_label="$1" total_rows="$2" total_shards="$3" max_parallel="$4" venv="$5" gpus_per_proc="$6"
  shift 6
  local -a base_args=("$@")
  local shard=0
  while (( shard < total_shards )); do
    local -a pids=()
    local -a names=()
    local launched=0
    while (( shard < total_shards && launched < max_parallel )); do
      local gpu_start=$(( (launched * gpus_per_proc) % 4 ))
      local gpu_list="$gpu_start"
      if (( gpus_per_proc == 2 )); then
        gpu_list="$gpu_start,$((gpu_start + 1))"
      fi
      local name="${stage_label}.shard_${shard}"
      (
        set -euo pipefail
        export CUDA_VISIBLE_DEVICES="$gpu_list"
        source "$venv/bin/activate"
        echo "[$name] gpus=$gpu_list rows=$total_rows shard=$shard/$total_shards started=$(date -Is)"
        python baseline_cascaded.py \
          "${base_args[@]}" \
          --num-samples "$total_rows" \
          --shard-index "$shard" \
          --num-shards "$total_shards" \
          --run-id "$RUN_ID" \
          --continue-on-error
        echo "[$name] finished=$(date -Is)"
      ) >"$JOB_LOG_ROOT/${name}.out" 2>&1 &
      pids+=("$!")
      names+=("$name")
      shard=$((shard + 1))
      launched=$((launched + 1))
    done
    for i in "${!pids[@]}"; do
      wait "${pids[$i]}" || { echo "FAIL ${names[$i]} see $JOB_LOG_ROOT/${names[$i]}.out" >&2; exit 1; }
      echo "PASS ${names[$i]}"
    done
  done
}

echo "FINAL_GENERATION run_id=$RUN_ID t2t_run_id=$T2T_RUN_ID logs=$JOB_LOG_ROOT"
echo "T2A input=$S2A_T2T_INPUT"
echo "T2S input=$A2S_T2T_INPUT"
echo "CosyVoice reference bank=$COSYVOICE_REFERENCE_BANK"
git rev-parse HEAD >"$JOB_LOG_ROOT/git_commit.txt" 2>/dev/null || echo unknown >"$JOB_LOG_ROOT/git_commit.txt"
git status --short >"$JOB_LOG_ROOT/git_status_short.txt" 2>/dev/null || true
echo "$PWD" >"$JOB_LOG_ROOT/repo_path.txt"
nvidia-smi || true

run_packed_shards \
  "t2a_${T2A_BACKEND}" "$TOTAL_T2A" "$T2A_SHARDS" "$T2A_MAX_PARALLEL" "$SALM_SCRATCH/.venvs/models/.venv-audio-omni" 1 \
  --results-schema v1 --pipeline T2A --t2a "$T2A_BACKEND" --text-model "$TEXT_MODEL" \
  --split "$SPLIT" --split-file "$S2A_T2T_INPUT" --duration "$T2A_DURATION" --steps "$T2A_STEPS"

echo "T2A shards complete: run_dir=$T2A_RUN_DIR"

run_packed_shards \
  "t2s_${T2S_BACKEND}" "$TOTAL_T2S" "$T2S_SHARDS" "$T2S_MAX_PARALLEL" "$SALM_SCRATCH/.venvs/models/.venv-cascade-cosyvoice3-py310" 1 \
  --results-schema v1 --pipeline T2S --tts "$T2S_BACKEND" --text-model "$TEXT_MODEL" \
  --split "$SPLIT" --split-file "$A2S_T2T_INPUT" --duration "$T2S_DURATION" --steps "$T2S_STEPS" \
  --tts-reference-bank-file "$COSYVOICE_REFERENCE_BANK"

echo "T2S shards complete: run_dir=$T2S_RUN_DIR"
echo "merge later from a login node after auditing all shards:"
echo "python scripts/merge_cascaded_stage_shards.py --run-dir '$T2A_RUN_DIR' --expected-count '$TOTAL_T2A' --media-output audio --keep-shards"
echo "python scripts/merge_cascaded_stage_shards.py --run-dir '$T2S_RUN_DIR' --expected-count '$TOTAL_T2S' --media-output speech --keep-shards"
