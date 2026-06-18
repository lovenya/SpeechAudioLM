#!/usr/bin/env bash
# Run current Prometheus G-Eval metrics from an allocated GPU node.
#
# Defaults are a small A2S demo on the merged TTAS test T2T manifest.
# Override with env vars, for example:
#   DIRECTION=s2a MAX_SAMPLES=2 TEMPORARY=1 bash scripts/run_geval_prometheus_v3_eval.sh

set -euo pipefail

REPO="${REPO:-/scratch/l/lovenya/SpeechAudioLM}"
SALM_SCRATCH="${SALM_SCRATCH:-/scratch/l/lovenya/salm_migration_clean}"
PYTHON="${PYTHON:-/home/l/lovenya/.venvs/speechaudiolm/.venv-eval-prometheus-vllm-py312/bin/python}"

DATASET="${DATASET:-ttas}"
SPLIT="${SPLIT:-test}"
STAGE="${STAGE:-t2t}"
DIRECTION="${DIRECTION:-a2s}"
EVALUATED_RUN_ID="${EVALUATED_RUN_ID:-20260618_0001}"
MODEL="${MODEL:-cascaded}"
METRICS="${METRICS:-g_eval_prometheus_plausibility,g_eval_prometheus_reference}"
DEVICE="${DEVICE:-cuda}"
TEMPORARY="${TEMPORARY:-1}"
MAX_SAMPLES="${MAX_SAMPLES:-1}"

A2S_MANIFEST_DEFAULT="${SALM_SCRATCH}/results/cascaded/stages/t2t/a2s/llama3_3_70b_instruct/from-a2t/audio_flamingo_next/ttas/test/20260618_0001/manifest.jsonl"
S2A_MANIFEST_DEFAULT="${SALM_SCRATCH}/results/cascaded/stages/t2t/s2a/llama3_3_70b_instruct/from-s2t/whisper/ttas/test/20260618_0001/manifest.jsonl"

case "${DIRECTION}" in
  a2s) MANIFEST="${MANIFEST:-${A2S_MANIFEST_DEFAULT}}" ;;
  s2a) MANIFEST="${MANIFEST:-${S2A_MANIFEST_DEFAULT}}" ;;
  *) echo "Unsupported DIRECTION=${DIRECTION}; expected a2s or s2a" >&2; exit 2 ;;
esac

RUN_TAG="${RUN_TAG:-$(TZ=America/Toronto date +%Y%m%dT%H%M%S%z)}"
LOG_DIR="${LOG_DIR:-${SALM_SCRATCH}/logs/geval_prometheus_v3/${RUN_TAG}_${DIRECTION}}"
CACHE_DIR="${CACHE_DIR:-${SALM_SCRATCH}/cache/metrics/geval_prometheus_v3/${RUN_TAG}_${DIRECTION}}"
TIMEOUT="${TIMEOUT:-45m}"

mkdir -p "${LOG_DIR}" "${CACHE_DIR}"
cd "${REPO}"

if [[ "${LOAD_MODULES:-1}" == "1" ]]; then
  module --force purge
  module load StdEnv/2023 gcc cuda/12.6 python/3.12.4 scipy-stack
fi

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

if [[ ! -f "${MANIFEST}" ]]; then
  echo "Missing manifest: ${MANIFEST}" >&2
  exit 2
fi

{
  echo "timestamp=$(date --iso-8601=seconds)"
  echo "hostname=$(hostname)"
  echo "repo=${REPO}"
  echo "git_commit=$(git rev-parse HEAD)"
  echo "git_dirty=$(if git diff --quiet && git diff --cached --quiet; then echo false; else echo true; fi)"
  echo "manifest=${MANIFEST}"
  echo "metrics=${METRICS}"
  echo "cache_dir=${CACHE_DIR}"
  echo "log_dir=${LOG_DIR}"
  command -v nvidia-smi >/dev/null 2>&1 && nvidia-smi -L
} | tee "${LOG_DIR}/run_metadata.txt"

args=(
  evaluate.py
  --model "${MODEL}"
  --stage "${STAGE}"
  --direction "${DIRECTION}"
  --evaluated-run-id "${EVALUATED_RUN_ID}"
  --dataset "${DATASET}"
  --split "${SPLIT}"
  --manifest "${MANIFEST}"
  --metrics "${METRICS}"
  --metrics-config configs/eval_metrics.json
  --device "${DEVICE}"
  --cache-dir "${CACHE_DIR}"
)

if [[ "${TEMPORARY}" == "1" ]]; then
  args+=(--temporary)
fi

if [[ -n "${MAX_SAMPLES}" ]]; then
  args+=(--max-samples "${MAX_SAMPLES}")
fi

if [[ -n "${OUTPUT_ROOT:-}" ]]; then
  args+=(--output-root "${OUTPUT_ROOT}")
fi

printf '%q ' "${PYTHON}" "${args[@]}" | tee "${LOG_DIR}/command.txt"
printf '\n' | tee -a "${LOG_DIR}/command.txt"

timeout --signal=TERM --kill-after=30s "${TIMEOUT}" "${PYTHON}" "${args[@]}" 2>&1 | tee "${LOG_DIR}/evaluate.log"

"${PYTHON}" - "${LOG_DIR}/evaluate.log" "${LOG_DIR}/trace_check.json" <<'PY'
import json
import sys
from pathlib import Path

log_path = Path(sys.argv[1])
out_path = Path(sys.argv[2])
judgment_paths = []
for line in log_path.read_text(encoding="utf-8", errors="replace").splitlines():
    if ".manifest=" in line or line.startswith("manifest="):
        manifest = Path(line.split("=", 1)[1].strip())
    else:
        continue
    metric = manifest.parent.parent.name
    candidate = manifest.parent / "artifacts" / "dedicated-script" / metric / "criterion_judgments.jsonl"
    if candidate.is_file():
        judgment_paths.append(candidate)

summary = {"criterion_judgments": []}
for path in judgment_paths:
    rows = []
    with path.open(encoding="utf-8") as handle:
        for index, line in enumerate(handle):
            if index >= 3:
                break
            row = json.loads(line)
            rows.append({
                "sample_id": row.get("sample_id"),
                "criterion_id": row.get("criterion_id"),
                "status": row.get("status"),
                "backend_format": (row.get("prompt") or {}).get("backend_format"),
                "trace_source": (row.get("prompt") or {}).get("trace_source"),
                "has_backend_formatted_prompt": (row.get("prompt") or {}).get("backend_formatted_prompt") is not None,
                "parse_status": ((row.get("output") or {}).get("parse_status")),
                "parse_repair_attempted": ((row.get("output") or {}).get("parse_repair_attempted")),
            })
    summary["criterion_judgments"].append({"path": str(path), "sample_rows": rows})

out_path.write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
print(json.dumps(summary, indent=2))
PY

echo "done log_dir=${LOG_DIR}"
