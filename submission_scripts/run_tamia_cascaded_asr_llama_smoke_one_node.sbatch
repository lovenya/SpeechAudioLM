#!/usr/bin/env bash
#SBATCH --job-name=salm-asr-llama-smoke
#SBATCH --account=aip-csubakan
#SBATCH --time=00:45:00
#SBATCH --nodes=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=128G
#SBATCH --gpus=h100:4
#SBATCH --output=/scratch/l/lovenya/salm_migration_clean/logs/slurm/%x-%j.out

set -euo pipefail

cd /scratch/l/lovenya/SpeechAudioLM
module --force purge
module load StdEnv/2023 gcc cuda rust arrow python/3.10.13

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
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-2}"
export MKL_NUM_THREADS="${MKL_NUM_THREADS:-2}"
export NUMEXPR_NUM_THREADS="${NUMEXPR_NUM_THREADS:-2}"

START_INDEX="${START_INDEX:-0}"
NUM_SAMPLES_VALUE="${NUM_SAMPLES:-1}"
ASR_STEPS="${ASR_STEPS:-16}"
TEXT_STEPS="${TEXT_STEPS:-32}"
ASR_CUDA_VISIBLE_DEVICES="${ASR_CUDA_VISIBLE_DEVICES:-0}"
LLAMA_CUDA_VISIBLE_DEVICES="${LLAMA_CUDA_VISIBLE_DEVICES:-1,2,3}"
BASELINE_ROOT="$SALM_SCRATCH/results/baselines_cascaded_asr_llama_smoke"
LOG_ROOT="$SALM_SCRATCH/logs/baselines_cascaded_asr_llama_smoke"
JOB_LOG_ROOT="$SALM_SCRATCH/logs/cascaded_asr_llama_smokes/${SLURM_JOB_ID:-manual}"

mkdir -p "$BASELINE_ROOT" "$LOG_ROOT" "$JOB_LOG_ROOT" "$SALM_SCRATCH/logs/slurm"

source submission_scripts/lib_resource_monitor.sh
start_resource_monitor "tamia_cascaded_asr_llama_smoke_one_node"

pids=()
names=()

run_smoke() {
  local name="$1"
  local gpu_list="$2"
  local venv="$3"
  shift 3

  (
    set -euo pipefail
    export CUDA_VISIBLE_DEVICES="$gpu_list"
    source "$venv/bin/activate"
    echo "[$name] host=$(hostname) cuda_visible_devices=$CUDA_VISIBLE_DEVICES started=$(date -Is)"
    python baseline_cascaded.py "$@"
    echo "[$name] finished=$(date -Is)"
  ) >"$JOB_LOG_ROOT/${name}.out" 2>&1 &

  pids+=("$!")
  names+=("$name")
}

run_smoke   packed_s2t_qwen3_asr   "$ASR_CUDA_VISIBLE_DEVICES"   "$SALM_SCRATCH/.venvs/models/.venv-cascade-asr"   --pipeline S2T   --asr qwen3_asr   --split test   --start-index "$START_INDEX"   --num-samples "$NUM_SAMPLES_VALUE"   --steps "$ASR_STEPS"   --output-dir "$BASELINE_ROOT"   --log-dir "$LOG_ROOT"   --overwrite

run_smoke   packed_t2t_llama3_3_70b_instruct   "$LLAMA_CUDA_VISIBLE_DEVICES"   "$SALM_SCRATCH/.venvs/models/.venv-cascade-text"   --pipeline T2T   --text-model llama3_3_70b_instruct   --text-mode speech_to_audio_description   --split test   --start-index "$START_INDEX"   --num-samples "$NUM_SAMPLES_VALUE"   --steps "$TEXT_STEPS"   --output-dir "$BASELINE_ROOT"   --log-dir "$LOG_ROOT"   --overwrite

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
echo "Generated ASR/Llama smoke outputs: $BASELINE_ROOT"
find "$BASELINE_ROOT" -name manifest.jsonl -print
exit "$status"
