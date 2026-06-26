#!/usr/bin/env bash
# Run full A2S + S2A Prometheus plausibility with two TP=2 processes.
# Intended for an allocated 4-GPU node. No Slurm partition is specified here.

set -euo pipefail

REPO="${REPO:-/scratch/l/lovenya/SpeechAudioLM}"
SALM_SCRATCH="${SALM_SCRATCH:-/scratch/l/lovenya/salm_migration_clean}"
PYTHON="${PYTHON:-/home/l/lovenya/.venvs/speechaudiolm/.venv-eval-prometheus-vllm-py312/bin/python}"

DATASET="${DATASET:-ttas}"
SPLIT="${SPLIT:-test}"
STAGE="${STAGE:-t2t}"
EVALUATED_RUN_ID="${EVALUATED_RUN_ID:-20260618_0001}"
MODEL="${MODEL:-cascaded}"
METRIC="${METRIC:-geval_text_plausibility}"
MAX_SAMPLES="${MAX_SAMPLES:-}"
TIMEOUT="${TIMEOUT:-115m}"
GPU_MEMORY_UTILIZATION="${GPU_MEMORY_UTILIZATION:-0.80}"
RESOURCE_INTERVAL="${RESOURCE_INTERVAL:-10}"

A2S_MANIFEST="${A2S_MANIFEST:-${SALM_SCRATCH}/results/cascaded/stages/t2t/a2s/llama3_3_70b_instruct/from-a2t/audio_flamingo_next/ttas/test/20260618_0001/manifest.jsonl}"
S2A_MANIFEST="${S2A_MANIFEST:-${SALM_SCRATCH}/results/cascaded/stages/t2t/s2a/llama3_3_70b_instruct/from-s2t/whisper/ttas/test/20260618_0001/manifest.jsonl}"

RUN_TAG="${RUN_TAG:-$(TZ=America/Toronto date +%Y%m%dT%H%M%S%z)_${METRIC}_tp2_full}"
LOG_ROOT="${LOG_ROOT:-${SALM_SCRATCH}/logs/geval_packed_tp2/${RUN_TAG}}"
CACHE_ROOT="${CACHE_ROOT:-${SALM_SCRATCH}/cache/metrics/geval_packed_tp2/${RUN_TAG}}"
OUTPUT_ROOT="${OUTPUT_ROOT:-${SALM_SCRATCH}/results/evals/geval_packed_tp2/${RUN_TAG}}"
METRICS_CONFIG="${METRICS_CONFIG:-${LOG_ROOT}/eval_metrics_tp2.json}"

mkdir -p "${LOG_ROOT}" "${CACHE_ROOT}" "${OUTPUT_ROOT}"
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

"${PYTHON}" - configs/eval_metrics.json "${METRICS_CONFIG}" "${METRIC}" "${GPU_MEMORY_UTILIZATION}" <<'PY'
import json
import sys
from pathlib import Path

source = Path(sys.argv[1])
target = Path(sys.argv[2])
metric = sys.argv[3]
gpu_memory_utilization = float(sys.argv[4])
config = json.loads(source.read_text(encoding="utf-8"))
config["metrics"][metric]["tensor_parallel_size"] = 2
config["metrics"][metric]["gpu_memory_utilization"] = gpu_memory_utilization
target.write_text(json.dumps(config, indent=2) + "\n", encoding="utf-8")
print(target)
PY

cat > "${LOG_ROOT}/run_metadata.json" <<JSON
{
  "run_tag": "${RUN_TAG}",
  "timestamp": "$(date --iso-8601=seconds)",
  "hostname": "$(hostname)",
  "repo": "${REPO}",
  "git_commit": "$(git rev-parse HEAD)",
  "git_dirty": $(if git diff --quiet && git diff --cached --quiet; then echo false; else echo true; fi),
  "metric": "${METRIC}",
  "max_samples": "${MAX_SAMPLES}",
  "timeout": "${TIMEOUT}",
  "gpu_memory_utilization": "${GPU_MEMORY_UTILIZATION}",
  "a2s_manifest": "${A2S_MANIFEST}",
  "s2a_manifest": "${S2A_MANIFEST}",
  "output_root": "${OUTPUT_ROOT}",
  "cache_root": "${CACHE_ROOT}",
  "log_root": "${LOG_ROOT}"
}
JSON

(
  while true; do
    date --iso-8601=seconds
    nvidia-smi --query-gpu=index,memory.used,memory.total,utilization.gpu,power.draw \
      --format=csv,noheader,nounits
    sleep "${RESOURCE_INTERVAL}"
  done
) > "${LOG_ROOT}/nvidia_smi_${RESOURCE_INTERVAL}s.log" 2>&1 &
MON_PID=$!

cleanup() {
  kill "${MON_PID}" 2>/dev/null || true
}
trap cleanup EXIT INT TERM

run_one() {
  local direction="$1"
  local manifest="$2"
  local gpu_pair="$3"
  local label="${direction}_${METRIC}"
  local log="${LOG_ROOT}/${label}.log"
  local cache_dir="${CACHE_ROOT}/${label}"

  local args=(
    evaluate.py
    --model "${MODEL}"
    --stage "${STAGE}"
    --direction "${direction}"
    --evaluated-run-id "${EVALUATED_RUN_ID}"
    --dataset "${DATASET}"
    --split "${SPLIT}"
    --manifest "${manifest}"
    --metrics "${METRIC}"
    --metrics-config "${METRICS_CONFIG}"
    --device cuda
    --cache-dir "${cache_dir}"
    --output-root "${OUTPUT_ROOT}"
  )
  if [[ -n "${MAX_SAMPLES}" ]]; then
    args+=(--temporary --max-samples "${MAX_SAMPLES}")
  fi

  echo "START ${label} gpus=${gpu_pair} $(date --iso-8601=seconds)" | tee "${log}"
  printf '%q ' CUDA_VISIBLE_DEVICES="${gpu_pair}" "${PYTHON}" "${args[@]}" | tee -a "${log}"
  printf '\n' | tee -a "${log}"
  set +e
  CUDA_VISIBLE_DEVICES="${gpu_pair}" timeout --signal=TERM --kill-after=60s "${TIMEOUT}" \
    "${PYTHON}" "${args[@]}" 2>&1 \
    | awk '{ print strftime("%Y-%m-%dT%H:%M:%S%z"), $0; fflush(); }' >> "${log}"
  local status=${PIPESTATUS[0]}
  set -e
  echo "END ${label} status=${status} $(date --iso-8601=seconds)" | tee -a "${log}"
  return "${status}"
}

run_one a2s "${A2S_MANIFEST}" "0,1" &
P_A2S=$!
run_one s2a "${S2A_MANIFEST}" "2,3" &
P_S2A=$!

set +e
wait "${P_A2S}"
S_A2S=$?
wait "${P_S2A}"
S_S2A=$?
set -e
cleanup

{
  echo "A2S status=${S_A2S}"
  echo "S2A status=${S_S2A}"
  echo "logs=${LOG_ROOT}"
  echo "outputs=${OUTPUT_ROOT}"
} | tee "${LOG_ROOT}/status.txt"

"${PYTHON}" - "${OUTPUT_ROOT}" "${LOG_ROOT}/completion_report.json" <<'PY'
import json
import sys
from pathlib import Path

root = Path(sys.argv[1])
out = Path(sys.argv[2])
report = {"output_root": str(root), "summaries": []}
for path in sorted(root.rglob("summary.json")):
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except Exception as exc:
        report["summaries"].append({"path": str(path), "error": f"{type(exc).__name__}: {exc}"})
        continue
    report["summaries"].append({
        "path": str(path),
        "metric": (payload.get("metric") or {}).get("metric_id"),
        "direction": (payload.get("target") or {}).get("direction"),
        "status": payload.get("status"),
        "samples": payload.get("samples"),
        "timing": payload.get("timing"),
    })
out.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
print(json.dumps(report, indent=2))
PY

tail -80 "${LOG_ROOT}/a2s_${METRIC}.log" || true
tail -80 "${LOG_ROOT}/s2a_${METRIC}.log" || true
tail -60 "${LOG_ROOT}/nvidia_smi_${RESOURCE_INTERVAL}s.log" || true

if [[ "${S_A2S}" -ne 0 || "${S_S2A}" -ne 0 ]]; then
  exit 1
fi
