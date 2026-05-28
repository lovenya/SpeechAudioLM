#!/usr/bin/env bash
#SBATCH --job-name=salm-cascade-smoke
#SBATCH --account=aip-csubakan
#SBATCH --time=00:20:00
#SBATCH --nodes=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=192G
#SBATCH --gpus=h100:4
#SBATCH --output=/scratch/l/lovenya/salm_migration_clean/logs/slurm/%x-%j.out

set -euo pipefail

cd /scratch/l/lovenya/SpeechAudioLM
module --force purge
module load StdEnv/2023 gcc cuda rust arrow python/3.10.13

export SALM_SCRATCH="${SALM_SCRATCH:-/scratch/l/lovenya/salm_migration_clean}"
export HF_HOME="$SALM_SCRATCH/models/hf_cache"
export HF_HUB_CACHE="$HF_HOME"
export TRANSFORMERS_CACHE="$HF_HOME"
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export PYTHONDONTWRITEBYTECODE=1
export TOKENIZERS_PARALLELISM=false
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-4}"
export MKL_NUM_THREADS="${MKL_NUM_THREADS:-4}"
export NUMEXPR_NUM_THREADS="${NUMEXPR_NUM_THREADS:-4}"

START_INDEX="${START_INDEX:-0}"
NUM_SAMPLES_VALUE="${NUM_SAMPLES:-1}"
DURATION="${DURATION:-10}"
BASELINE_ROOT="$SALM_SCRATCH/results/baselines_cascaded_smoke"
LOG_ROOT="$SALM_SCRATCH/logs/baselines_cascaded_smoke"
JOB_LOG_ROOT="$SALM_SCRATCH/logs/cascaded_smokes/${SLURM_JOB_ID:-manual}"

mkdir -p "$BASELINE_ROOT" "$LOG_ROOT" "$JOB_LOG_ROOT" "$SALM_SCRATCH/logs/slurm"

source submission_scripts/lib_resource_monitor.sh
start_resource_monitor "tamia_cascaded_smokes_one_node"

pids=()
names=()

run_smoke() {
  local name="$1"
  local gpu="$2"
  local venv="$3"
  shift 3

  (
    set -euo pipefail
    export CUDA_VISIBLE_DEVICES="$gpu"
    source "$venv/bin/activate"
    echo "[$name] host=$(hostname) cuda_visible_devices=$CUDA_VISIBLE_DEVICES started=$(date -Is)"
    python baseline_cascaded.py "$@"
    echo "[$name] finished=$(date -Is)"
  ) >"$JOB_LOG_ROOT/${name}.out" 2>&1 &

  pids+=("$!")
  names+=("$name")
}

run_smoke \
  cascaded_s2a_reference_audiogen \
  0 \
  "$SALM_SCRATCH/.venvs/models/.venv-audiocraft" \
  --pipeline S2A \
  --asr reference \
  --text-model template \
  --t2a audiogen \
  --split test \
  --start-index "$START_INDEX" \
  --num-samples "$NUM_SAMPLES_VALUE" \
  --duration "$DURATION" \
  --output-dir "$BASELINE_ROOT" \
  --log-dir "$LOG_ROOT" \
  --overwrite

run_smoke \
  cascaded_s2a_reference_audio_omni \
  1 \
  "$SALM_SCRATCH/.venvs/models/.venv-audio-omni" \
  --pipeline S2A \
  --asr reference \
  --text-model template \
  --t2a audio_omni \
  --split test \
  --start-index "$START_INDEX" \
  --num-samples "$NUM_SAMPLES_VALUE" \
  --duration "$DURATION" \
  --output-dir "$BASELINE_ROOT" \
  --log-dir "$LOG_ROOT" \
  --overwrite

run_smoke \
  cascaded_a2s_reference_audio_omni \
  2 \
  "$SALM_SCRATCH/.venvs/models/.venv-audio-omni" \
  --pipeline A2S \
  --captioner reference \
  --text-model template \
  --tts audio_omni \
  --split test \
  --start-index "$START_INDEX" \
  --num-samples "$NUM_SAMPLES_VALUE" \
  --duration "$DURATION" \
  --output-dir "$BASELINE_ROOT" \
  --log-dir "$LOG_ROOT" \
  --overwrite

run_smoke \
  cascaded_t2s_qwen3_omni \
  3 \
  "$SALM_SCRATCH/.venvs/models/.venv-qwen3-omni" \
  --pipeline T2S \
  --tts qwen3_omni \
  --split test \
  --start-index "$START_INDEX" \
  --num-samples "$NUM_SAMPLES_VALUE" \
  --duration "$DURATION" \
  --output-dir "$BASELINE_ROOT" \
  --log-dir "$LOG_ROOT" \
  --overwrite

status=0
for i in "${!pids[@]}"; do
  if wait "${pids[$i]}"; then
    echo "PASS ${names[$i]}"
  else
    echo "FAIL ${names[$i]} - see $JOB_LOG_ROOT/${names[$i]}.out"
    status=1
  fi
done

echo "Per-cascade logs: $JOB_LOG_ROOT"
echo "Generated cascaded smoke outputs: $BASELINE_ROOT"
find "$BASELINE_ROOT" -name manifest.jsonl -print
exit "$status"
