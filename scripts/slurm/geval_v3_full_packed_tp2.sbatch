#!/usr/bin/env bash
# Full-scale packed Prometheus G-Eval for merged TTAS test T2T outputs.
#
# Packing strategy:
#   wave 1: A2S plausibility on GPUs 0,1 and S2A plausibility on GPUs 2,3
#   wave 2: A2S reference     on GPUs 0,1 and S2A reference     on GPUs 2,3
#
# Each process uses tensor_parallel_size=2 via a runtime metrics config.

#SBATCH --job-name=geval-v3-tp2-full
#SBATCH --account=aip-ravanelm
#SBATCH --partition=gpubase_bynode_b2
#SBATCH --time=04:00:00
#SBATCH --nodes=1
#SBATCH --gpus-per-node=h100:4
#SBATCH --cpus-per-task=32
#SBATCH --mem=256G
#SBATCH --output=/scratch/l/lovenya/salm_migration_clean/logs/slurm/%x-%j.out

set -euo pipefail

REPO="${REPO:-/scratch/l/lovenya/SpeechAudioLM}"
SALM_SCRATCH="${SALM_SCRATCH:-/scratch/l/lovenya/salm_migration_clean}"
PYTHON="${PYTHON:-/home/l/lovenya/.venvs/speechaudiolm/.venv-eval-prometheus-vllm-py312/bin/python}"

DATASET="${DATASET:-ttas}"
SPLIT="${SPLIT:-test}"
STAGE="${STAGE:-t2t}"
EVALUATED_RUN_ID="${EVALUATED_RUN_ID:-20260618_0001}"
MODEL="${MODEL:-cascaded}"
MAX_SAMPLES="${MAX_SAMPLES:-}"
RUN_TAG="${RUN_TAG:-$(TZ=America/Toronto date +%Y%m%dT%H%M%S%z)_${SLURM_JOB_ID:-manual}}"
WORKLOAD_TIMEOUT="${WORKLOAD_TIMEOUT:-115m}"
METRICS_TO_RUN="${METRICS_TO_RUN:-g_eval_prometheus_plausibility,g_eval_prometheus_reference}"

A2S_MANIFEST="${A2S_MANIFEST:-${SALM_SCRATCH}/results/cascaded/stages/t2t/a2s/llama3_3_70b_instruct/from-a2t/audio_flamingo_next/ttas/test/20260618_0001/manifest.jsonl}"
S2A_MANIFEST="${S2A_MANIFEST:-${SALM_SCRATCH}/results/cascaded/stages/t2t/s2a/llama3_3_70b_instruct/from-s2t/whisper/ttas/test/20260618_0001/manifest.jsonl}"

LOG_ROOT="${LOG_ROOT:-${SALM_SCRATCH}/logs/geval_prometheus_v3_full/${RUN_TAG}}"
OUTPUT_ROOT="${OUTPUT_ROOT:-${SALM_SCRATCH}/results/evals/geval_prometheus_v3_full/${RUN_TAG}}"
CACHE_ROOT="${CACHE_ROOT:-${SALM_SCRATCH}/cache/metrics/geval_prometheus_v3_full/${RUN_TAG}}"
METRICS_CONFIG="${LOG_ROOT}/eval_metrics_tp2.json"

mkdir -p "${LOG_ROOT}" "${OUTPUT_ROOT}" "${CACHE_ROOT}" "${SALM_SCRATCH}/logs/slurm"
cd "${REPO}"

module --force purge
module load StdEnv/2023 gcc cuda/12.6 python/3.12.4 scipy-stack

if [[ -f /home/l/lovenya/.venvs/speechaudiolm/.venv-eval-prometheus-vllm-py312/bin/activate ]]; then
  # shellcheck source=/dev/null
  source /home/l/lovenya/.venvs/speechaudiolm/.venv-eval-prometheus-vllm-py312/bin/activate
fi

export PYTHONNOUSERSITE=1
export PYTHONDONTWRITEBYTECODE=1
export TOKENIZERS_PARALLELISM=false
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export HF_HOME="${SALM_SCRATCH}/cache/metrics/huggingface"
export HF_HUB_CACHE="${HF_HOME}"
export HUGGINGFACE_HUB_CACHE="${HF_HOME}"
export TRANSFORMERS_CACHE="${HF_HOME}"
export TORCH_HOME="${SALM_SCRATCH}/cache/metrics/torch"
export VLLM_CACHE_ROOT="${SALM_SCRATCH}/cache/vllm"
export TORCHINDUCTOR_CACHE_DIR="${SALM_SCRATCH}/cache/vllm/torchinductor"
export VLLM_USE_DEEP_GEMM=0
export VLLM_MOE_USE_DEEP_GEMM=0
export VLLM_DEEP_GEMM_WARMUP=skip
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-1}"
export MKL_NUM_THREADS="${MKL_NUM_THREADS:-1}"
export NUMEXPR_NUM_THREADS="${NUMEXPR_NUM_THREADS:-1}"

for manifest in "${A2S_MANIFEST}" "${S2A_MANIFEST}"; do
  if [[ ! -f "${manifest}" ]]; then
    echo "Missing manifest: ${manifest}" >&2
    exit 2
  fi
done

git_commit="$(git rev-parse HEAD)"
if git diff --quiet && git diff --cached --quiet; then
  git_dirty=false
else
  git_dirty=true
fi

if [[ "${git_dirty}" == "true" && "${ALLOW_DIRTY:-0}" != "1" ]]; then
  echo "Refusing full-scale G-Eval submission from a dirty checkout." >&2
  echo "Commit/stash unrelated changes, or resubmit with --export=ALL,ALLOW_DIRTY=1 if you explicitly accept the risk." >&2
  git status --short >&2
  exit 2
fi

cat > "${LOG_ROOT}/run_metadata.json" <<JSON
{
  "run_tag": "${RUN_TAG}",
  "hostname": "$(hostname)",
  "slurm_job_id": "${SLURM_JOB_ID:-manual}",
  "repo": "${REPO}",
  "git_commit": "${git_commit}",
  "git_dirty": ${git_dirty},
  "output_root": "${OUTPUT_ROOT}",
  "cache_root": "${CACHE_ROOT}",
  "a2s_manifest": "${A2S_MANIFEST}",
  "s2a_manifest": "${S2A_MANIFEST}",
  "packing": "two concurrent tensor_parallel_size=2 processes on disjoint GPU pairs",
  "gpu_memory_utilization": 0.80,
  "metrics_to_run": "${METRICS_TO_RUN}",
  "workload_timeout": "${WORKLOAD_TIMEOUT}"
}
JSON

"${PYTHON}" - configs/eval_metrics.json "${METRICS_CONFIG}" <<'PY'
import json
import sys
from pathlib import Path

source = Path(sys.argv[1])
target = Path(sys.argv[2])
config = json.loads(source.read_text(encoding="utf-8"))
for metric in ("g_eval_prometheus_plausibility", "g_eval_prometheus_reference"):
    config["metrics"][metric]["tensor_parallel_size"] = 2
    config["metrics"][metric]["gpu_memory_utilization"] = 0.80
target.write_text(json.dumps(config, indent=2) + "\n", encoding="utf-8")
print(target)
PY

(
  while true; do
    printf '%s\n' "$(date --iso-8601=seconds)"
    nvidia-smi --query-gpu=index,memory.used,memory.total,utilization.gpu,power.draw \
      --format=csv,noheader,nounits
    sleep "${RESOURCE_INTERVAL:-60}"
  done
) > "${LOG_ROOT}/nvidia_smi_${SLURM_JOB_ID:-manual}.log" 2>&1 &
MONITOR_PID=$!

cleanup() {
  kill "${MONITOR_PID}" 2>/dev/null || true
}
trap cleanup EXIT INT TERM

run_eval() {
  local direction="$1"
  local manifest="$2"
  local metric="$3"
  local gpu_pair="$4"
  local label="${direction}_${metric}"
  local log="${LOG_ROOT}/${label}.log"
  local cache="${CACHE_ROOT}/${label}"

  local args=(
    evaluate.py
    --model "${MODEL}"
    --stage "${STAGE}"
    --direction "${direction}"
    --evaluated-run-id "${EVALUATED_RUN_ID}"
    --dataset "${DATASET}"
    --split "${SPLIT}"
    --manifest "${manifest}"
    --metrics "${metric}"
    --metrics-config "${METRICS_CONFIG}"
    --device cuda
    --cache-dir "${cache}"
    --output-root "${OUTPUT_ROOT}"
  )
  if [[ -n "${MAX_SAMPLES}" ]]; then
    args+=(--max-samples "${MAX_SAMPLES}")
  fi

  echo "START ${label} gpus=${gpu_pair} $(date --iso-8601=seconds)" | tee "${log}"
  printf '%q ' CUDA_VISIBLE_DEVICES="${gpu_pair}" "${PYTHON}" "${args[@]}" | tee -a "${log}"
  printf '\n' | tee -a "${log}"
  CUDA_VISIBLE_DEVICES="${gpu_pair}" timeout --signal=TERM --kill-after=60s "${WORKLOAD_TIMEOUT}" \
    "${PYTHON}" "${args[@]}" >> "${log}" 2>&1
  local status=$?
  echo "END ${label} status=${status} $(date --iso-8601=seconds)" | tee -a "${log}"
  return "${status}"
}

run_wave() {
  local metric="$1"
  local failures=0
  echo "WAVE_START metric=${metric} $(date --iso-8601=seconds)" | tee -a "${LOG_ROOT}/waves.log"
  run_eval a2s "${A2S_MANIFEST}" "${metric}" "0,1" &
  local p_a2s=$!
  run_eval s2a "${S2A_MANIFEST}" "${metric}" "2,3" &
  local p_s2a=$!
  wait "${p_a2s}" || failures=$((failures + 1))
  wait "${p_s2a}" || failures=$((failures + 1))
  echo "WAVE_END metric=${metric} failures=${failures} $(date --iso-8601=seconds)" | tee -a "${LOG_ROOT}/waves.log"
  return "${failures}"
}

failures=0
IFS=',' read -r -a metric_list <<< "${METRICS_TO_RUN}"
for metric in "${metric_list[@]}"; do
  metric="${metric//[[:space:]]/}"
  [[ -z "${metric}" ]] && continue
  case "${metric}" in
    g_eval_prometheus_plausibility|g_eval_prometheus_reference) ;;
    *) echo "Unsupported metric in METRICS_TO_RUN: ${metric}" >&2; exit 2 ;;
  esac
  run_wave "${metric}" || failures=$((failures + $?))
  sleep 30
done

cleanup

"${PYTHON}" - "${OUTPUT_ROOT}" "${LOG_ROOT}/completion_report.json" <<'PY'
import json
import sys
from pathlib import Path

root = Path(sys.argv[1])
out = Path(sys.argv[2])
report = {"output_root": str(root), "metrics": []}
for summary in sorted(root.rglob("summary.json")):
    try:
        payload = json.loads(summary.read_text(encoding="utf-8"))
    except Exception as exc:
        report["metrics"].append({"summary": str(summary), "error": f"{type(exc).__name__}: {exc}"})
        continue
    report["metrics"].append({
        "summary": str(summary),
        "metric": ((payload.get("metric") or {}).get("metric_id")),
        "status": payload.get("status"),
        "samples": payload.get("samples"),
        "scores": payload.get("scores"),
    })
out.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
print(json.dumps(report, indent=2))
PY

echo "geval_v3_full_packed_complete failures=${failures} output_root=${OUTPUT_ROOT} log_root=${LOG_ROOT}"
exit "${failures}"
