#!/usr/bin/env bash
# Canonical test-split GPU evals for the confirmed minimal scope:
# Audio Flamingo Next A2T neural metrics plus NLI on the AFN and Whisper T2T
# stage outputs only.

#SBATCH --job-name=salm-test-w-afn-gpu
#SBATCH --account=aip-csubakan
#SBATCH --time=01:15:00
#SBATCH --cpus-per-task=32
#SBATCH --mem=128G
#SBATCH --gpus=h100:4
#SBATCH --output=/scratch/l/lovenya/salm_migration_clean/logs/slurm/%x-%j.out

set -euo pipefail

cd /scratch/l/lovenya/SpeechAudioLM

module --force purge
module load StdEnv/2023 gcc cuda/12.6 opencv/4.13.0 java/11.0.22 python/3.12.4 scipy-stack
unset JAVA_TOOL_OPTIONS

export SALM_SCRATCH="${SALM_SCRATCH:-/scratch/l/lovenya/salm_migration_clean}"
export SALM_RESULTS_ROOT="$SALM_SCRATCH/results"
export SALM_TEMPORARY_RESULTS_ROOT="$SALM_SCRATCH/temporary/results"
export SALM_LOGS_ROOT="$SALM_SCRATCH/logs"
export SALM_TEMPORARY_LOGS_ROOT="$SALM_SCRATCH/temporary/logs"
export SALM_CACHE_ROOT="$SALM_SCRATCH/cache"
export SALM_MODEL_ROOT="$SALM_SCRATCH/models"
export SALM_ENVIRONMENT_ROOT="$SALM_SCRATCH/.venvs"
export SALM_ENVIRONMENTS_ROOT="$SALM_SCRATCH/.venvs"

export PYTHONNOUSERSITE=1
export PYTHONDONTWRITEBYTECODE=1
export HF_HOME="$SALM_SCRATCH/cache/metrics/huggingface"
export HF_HUB_CACHE="$HF_HOME"
export HUGGINGFACE_HUB_CACHE="$HF_HOME"
export TRANSFORMERS_CACHE="$HF_HOME"
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export TORCH_HOME="$SALM_SCRATCH/cache/metrics/torch"
export TOKENIZERS_PARALLELISM=false
export RESOURCE_INTERVAL="${RESOURCE_INTERVAL:-60}"

SHARED_PYTHON="${SHARED_PYTHON:-/home/l/lovenya/.venvs/speechaudiolm/.venv-eval-metrics-shared-py312/bin/python}"
RUN_ID="$(TZ=America/Toronto date +%Y%m%dT%H%M%S%z)"
LOG_DIR="$SALM_SCRATCH/logs/evals/test_whisper_afn_gpu/$RUN_ID"
RUNTIME_CONFIG="$LOG_DIR/eval_metrics.runtime.json"
AFN_MANIFEST="$SALM_SCRATCH/results/cascaded/stages/a2t/audio_flamingo_next/ttas/test/20260618_0001/manifest.jsonl"
AFN_T2T_MANIFEST="$SALM_SCRATCH/results/cascaded/stages/t2t/a2s/llama3_3_70b_instruct/from-a2t/audio_flamingo_next/ttas/test/20260618_0001/manifest.jsonl"
WHISPER_T2T_MANIFEST="$SALM_SCRATCH/results/cascaded/stages/t2t/s2a/llama3_3_70b_instruct/from-s2t/whisper/ttas/test/20260618_0001/manifest.jsonl"

mkdir -p "$LOG_DIR" "$SALM_SCRATCH/logs/resources" "$SALM_SCRATCH/logs/slurm"
test -x "$SHARED_PYTHON" || { echo "missing shared python: $SHARED_PYTHON" >&2; exit 2; }
test -f "$AFN_MANIFEST" || { echo "missing manifest: $AFN_MANIFEST" >&2; exit 2; }
test -f "$AFN_T2T_MANIFEST" || { echo "missing manifest: $AFN_T2T_MANIFEST" >&2; exit 2; }
test -f "$WHISPER_T2T_MANIFEST" || { echo "missing manifest: $WHISPER_T2T_MANIFEST" >&2; exit 2; }

python - <<PY
import json
import subprocess
from pathlib import Path

cfg = json.loads(Path("configs/eval_metrics.json").read_text(encoding="utf-8"))
for name in ("mace", "aac_clapsim", "fense", "nli_contradiction"):
    cfg["metrics"][name]["python"] = "$SHARED_PYTHON"
cfg["metrics"]["nli_contradiction"]["local_files_only"] = True
cfg["metrics"]["nli_contradiction"].pop("allow_stub_fallback", None)
Path("$RUNTIME_CONFIG").write_text(json.dumps(cfg, indent=2) + "\n", encoding="utf-8")

metadata = {
    "script": "submission_scripts/run_tamia_eval_test_whisper_afn_gpu.sbatch",
    "job_id": "${SLURM_JOB_ID:-manual}",
    "run_id": "$RUN_ID",
    "scope": "test_only_whisper_and_audio_flamingo_next",
    "repo": str(Path.cwd()),
    "git_commit": subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip(),
    "git_dirty": bool(subprocess.check_output(["git", "status", "--porcelain"], text=True).strip()),
    "metrics_config": "$RUNTIME_CONFIG",
    "g_eval_excluded": True,
}
Path("$LOG_DIR/job_metadata.json").write_text(json.dumps(metadata, indent=2) + "\n", encoding="utf-8")
print("runtime_config=$RUNTIME_CONFIG")
print("job_metadata=$LOG_DIR/job_metadata.json")
PY

source submission_scripts/lib_resource_monitor.sh
start_resource_monitor "test_whisper_afn_gpu_${RUN_ID}"

run_eval() {
  local label="$1"
  local stage="$2"
  local direction="$3"
  local metrics="$4"
  local manifest="$5"
  local gpu="$6"
  local log="$LOG_DIR/${label}.log"

  echo "START label=$label stage=$stage direction=$direction split=test metrics=$metrics evaluated_run_id=20260618_0001 gpu=$gpu" | tee "$log"

  local direction_args=()
  if [ "$direction" != "none" ]; then
    direction_args=(--direction "$direction")
  fi

  CUDA_VISIBLE_DEVICES="$gpu" python evaluate.py \
    --model cascaded \
    --stage "$stage" \
    "${direction_args[@]}" \
    --evaluated-run-id 20260618_0001 \
    --dataset ttas \
    --split test \
    --manifest "$manifest" \
    --metrics "$metrics" \
    --metrics-config "$RUNTIME_CONFIG" \
    --cache-dir "$SALM_SCRATCH/cache/metrics/eval_runtime" \
    --device cuda >>"$log" 2>&1

  echo "PASS label=$label" | tee -a "$log"
}

failures=0
pids=()

run_eval a2t_afn_test_neural a2t none mace,aac_clapsim,fense "$AFN_MANIFEST" 0 & pids+=("$!")
run_eval t2t_a2s_afn_test_nli t2t a2s nli_contradiction "$AFN_T2T_MANIFEST" 1 & pids+=("$!")
run_eval t2t_s2a_whisper_test_nli t2t s2a nli_contradiction "$WHISPER_T2T_MANIFEST" 2 & pids+=("$!")

for pid in "${pids[@]}"; do
  wait "$pid" || failures=$((failures + 1))
done

stop_resource_monitor || true

echo "TEST_WHISPER_AFN_GPU_DONE failures=$failures log_dir=$LOG_DIR"
exit "$failures"
