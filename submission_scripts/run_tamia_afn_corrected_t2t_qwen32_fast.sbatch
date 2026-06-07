#!/usr/bin/env bash
#SBATCH --account=aip-csubakan
#SBATCH --nodes=1
#SBATCH --cpus-per-task=10
#SBATCH --mem=160G
#SBATCH --gpus=h100:4
#SBATCH --time=00:35:00
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
STEPS_TEXT="${STEPS_TEXT:-64}"
BASELINE_ROOT="$SALM_SCRATCH/results/cascaded_full_validation/$RUN_ID"
LOG_ROOT="$SALM_SCRATCH/logs/cascaded_full_validation/$RUN_ID"
JOB_LOG_ROOT="$LOG_ROOT/lanes/${SLURM_JOB_ID:-manual}_afn_corrected_qwen32_t2t"
SPLIT_ROOT="$BASELINE_ROOT/intermediate_splits"
METADATA_ROOT="$BASELINE_ROOT/metadata"
mkdir -p "$JOB_LOG_ROOT" "$SPLIT_ROOT" "$METADATA_ROOT"

source submission_scripts/lib_resource_monitor.sh
start_resource_monitor "tamia_cascaded_${RUN_ID}_afn_corrected_qwen32_t2t_fast"

INPUT_SPLIT="$SPLIT_ROOT/a2s_afn_corrected_to_qwen32.jsonl"
if [ ! -s "$INPUT_SPLIT" ]; then
  echo "missing input split: $INPUT_SPLIT" >&2
  exit 2
fi

cat >"$METADATA_ROOT/afn_corrected_qwen32_t2t_${SLURM_JOB_ID:-manual}.json" <<META
{"job_id":"${SLURM_JOB_ID:-manual}","stage":"afn_corrected_qwen32_t2t","run_id":"$RUN_ID","started_at":"$(date -Is)","requested":{"nodes":1,"gpus":4,"cpus":10,"memory_gb":160,"walltime_minutes":35},"input_split":"$INPUT_SPLIT","total_a2s":$TOTAL_A2S}
META

pids=()
names=()

run_shard() {
  local shard="$1"
  local count="$2"
  local start="$3"
  local gpu="$4"
  local run_name="full_a2s_afn_qwen32_t2t_corrected_${SLURM_JOB_ID:-manual}_s${shard}"
  (
    set -euo pipefail
    export CUDA_VISIBLE_DEVICES="$gpu"
    source "$SALM_SCRATCH/.venvs/models/.venv-cascade-text/bin/activate"
    echo "[qwen32_s${shard}] gpu=$gpu start=$start count=$count started=$(date -Is)"
    python baseline_cascaded.py \
      --pipeline T2T \
      --text-model qwen3_32b \
      --text-mode audio_description_to_speech \
      --split val \
      --split-file "$INPUT_SPLIT" \
      --start-index "$start" \
      --num-samples "$count" \
      --steps "$STEPS_TEXT" \
      --output-dir "$BASELINE_ROOT" \
      --log-dir "$LOG_ROOT" \
      --run-name "$run_name" \
      --overwrite \
      --continue-on-error
    echo "[qwen32_s${shard}] finished=$(date -Is)"
  ) >"$JOB_LOG_ROOT/qwen32_s${shard}.out" 2>&1 &
  pids+=("$!")
  names+=("qwen32_s${shard}")
}

base=$((TOTAL_A2S / 4))
extra=$((TOTAL_A2S % 4))
for shard in 0 1 2 3; do
  count="$base"
  if (( shard < extra )); then
    count=$((base + 1))
    start=$((shard * (base + 1)))
  else
    start=$((extra * (base + 1) + (shard - extra) * base))
  fi
  run_shard "$shard" "$count" "$start" "$shard"
done

for i in "${!pids[@]}"; do
  wait "${pids[$i]}" || { echo "FAIL ${names[$i]} see $JOB_LOG_ROOT/${names[$i]}.out" >&2; exit 1; }
  echo "PASS ${names[$i]}"
done

merge_args=()
for shard in 0 1 2 3; do
  merge_args+=(--input-dir "$BASELINE_ROOT/cascaded/T2T/full_a2s_afn_qwen32_t2t_corrected_${SLURM_JOB_ID:-manual}_s${shard}/ttas/val")
done
python scripts/merge_cascaded_shards.py \
  "${merge_args[@]}" \
  --output-dir "$BASELINE_ROOT/cascaded/T2T/full_a2s_afn_qwen32_t2t/ttas/val" \
  --lane full_a2s_afn_qwen32_t2t \
  --expected-count "$TOTAL_A2S" \
  --artifact-subdir text \
  --backup-label "afn_corrected_qwen32_${SLURM_JOB_ID:-manual}"
