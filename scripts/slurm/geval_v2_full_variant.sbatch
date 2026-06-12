#!/bin/bash
#SBATCH --job-name=geval-v2-full
#SBATCH --partition=gpubase_bynode_b1
#SBATCH --time=04:00:00
#SBATCH --nodes=1
#SBATCH --gpus-per-node=h100:4
#SBATCH --cpus-per-task=8
#SBATCH --mem=128G
#SBATCH --output=/scratch/l/lovenya/salm_migration_clean/logs/slurm/%x-%j.out

set -euo pipefail

: "${VARIANT:?Submit with --export=ALL,VARIANT=plausibility or reference_grounded}"
if [[ "${VARIANT}" != "plausibility" && "${VARIANT}" != "reference_grounded" ]]; then
    echo "invalid VARIANT=${VARIANT}" >&2
    exit 2
fi

REPO=/scratch/l/lovenya/SpeechAudioLM
PYTHON=/home/l/lovenya/.venvs/speechaudiolm/.venv-eval-prometheus-vllm-py312/bin/python
MODEL=/scratch/l/lovenya/salm_migration_clean/models/eval/prometheus-8x7b-v2.0
A2S_MANIFEST=/scratch/l/lovenya/salm_migration_clean/results/cascaded_full_validation/ttas_val/cascaded/T2T/full_a2s_afn_qwen32_t2t/ttas/val/manifest.jsonl
S2A_MANIFEST=/scratch/l/lovenya/salm_migration_clean/results/cascaded_full_validation/ttas_val/cascaded/T2T/full_s2a_whisper_qwen32_t2t/ttas/val/manifest.jsonl
RUN_LABEL=${RUN_LABEL:-full}
A2S_LIMIT=${A2S_LIMIT:-2258}
S2A_LIMIT=${S2A_LIMIT:-2277}
EVAL_TIMEOUT=${EVAL_TIMEOUT:-170m}
RUN_ID="$(TZ=America/Toronto date +%Y%m%dT%H%M%S%z)_${SLURM_JOB_ID}"
RUN_ROOT=/scratch/l/lovenya/salm_migration_clean/results/geval_v2_${RUN_LABEL}/${VARIANT}/${RUN_ID}
CACHE_ROOT=/scratch/l/lovenya/salm_migration_clean/cache/metrics/g_eval_prometheus_criterion_v2_${VARIANT}
RESOURCE_LOG=/scratch/l/lovenya/salm_migration_clean/logs/resources/geval_v2_full_${VARIANT}_${SLURM_JOB_ID}.jsonl

mkdir -p "${RUN_ROOT}" "${CACHE_ROOT}" "$(dirname "${RESOURCE_LOG}")"

module --force purge
module load StdEnv/2023 gcc cuda/12.6 python/3.12.4 scipy-stack/2026a
source /home/l/lovenya/.venvs/speechaudiolm/.venv-eval-prometheus-vllm-py312/bin/activate
cd "${REPO}"

export OMP_NUM_THREADS=1
export TOKENIZERS_PARALLELISM=false
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export VLLM_CACHE_ROOT=/scratch/l/lovenya/salm_migration_clean/cache/vllm
export TORCHINDUCTOR_CACHE_DIR=/scratch/l/lovenya/salm_migration_clean/cache/vllm/torchinductor
export VLLM_USE_DEEP_GEMM=0
export VLLM_MOE_USE_DEEP_GEMM=0
export VLLM_DEEP_GEMM_WARMUP=skip

echo "run_root=${RUN_ROOT}"
echo "variant=${VARIANT}"
echo "run_label=${RUN_LABEL}"
echo "a2s_limit=${A2S_LIMIT}"
echo "s2a_limit=${S2A_LIMIT}"
echo "eval_timeout=${EVAL_TIMEOUT}"
echo "hostname=$(hostname)"
echo "job_id=${SLURM_JOB_ID}"
echo "git_commit=$(git rev-parse HEAD)"
module list 2>&1
nvidia-smi -L

"${PYTHON}" - <<'PYENV'
import numpy
import scipy
from scipy.linalg import _fblas
import torch
import transformers
import vllm
print("numpy", numpy.__version__, numpy.__file__)
print("scipy", scipy.__version__, scipy.__file__)
print("torch", torch.__version__)
print("transformers", transformers.__version__)
print("vllm", vllm.__version__)
print("gpus", torch.cuda.device_count())
assert torch.cuda.device_count() == 4
PYENV

(
    while true; do
        gpu_csv=$(nvidia-smi --query-gpu=index,memory.used,memory.total,utilization.gpu,power.draw --format=csv,noheader,nounits \
            | "${PYTHON}" -c 'import json,sys; print(json.dumps([line.strip() for line in sys.stdin if line.strip()]))')
        printf '{"timestamp":"%s","job_id":"%s","variant":"%s","gpu_csv":%s}\n' "$(date --iso-8601=seconds)" "${SLURM_JOB_ID}" "${VARIANT}" "${gpu_csv}"
        sleep 60
    done
) > "${RESOURCE_LOG}" 2>&1 &
monitor_pid=$!
trap 'kill "${monitor_pid}" 2>/dev/null || true' EXIT INT TERM

"${PYTHON}" - "${A2S_MANIFEST}" "${S2A_MANIFEST}" "${RUN_ROOT}/input_full.jsonl" "${A2S_LIMIT}" "${S2A_LIMIT}" <<'PYINPUT'
import json
import sys
from pathlib import Path

specs = [
    (Path(sys.argv[1]), "audio_to_speech", "speech_text"),
    (Path(sys.argv[2]), "speech_to_audio", "audio_cap"),
]
output = Path(sys.argv[3])
limits = {"audio_to_speech": int(sys.argv[4]), "speech_to_audio": int(sys.argv[5])}
selected = []
counts = {}

if any(limit < 0 for limit in limits.values()):
    raise ValueError(f"limits must be non-negative: {limits}")

for manifest, direction, reference_field in specs:
    lane_rows = []
    if limits[direction] == 0:
        counts[direction] = 0
        continue
    with manifest.open(encoding="utf-8") as src:
        for line in src:
            wrapped = json.loads(line)
            sample = wrapped.get("sample", wrapped)
            if sample.get("status", "success") != "success":
                continue
            stages = sample.get("stages")
            if not isinstance(stages, list) or not stages or not isinstance(stages[-1], dict):
                continue
            stage = stages[-1]
            input_text = (stage.get("input") or {}).get("text")
            generated_text = (stage.get("output") or {}).get("text")
            reference = sample.get("reference") or wrapped.get("source") or {}
            reference_text = reference.get(reference_field)
            if reference_field == "audio_cap" and not reference_text:
                reference_text = reference.get("audio_caption")
            if not all(isinstance(value, str) and value.strip() for value in (input_text, generated_text, reference_text)):
                continue
            sample_id = str(sample.get("sample_id") or wrapped.get("sample_id") or f"{direction}_{len(lane_rows)}")
            lane_rows.append({
                "sample_id": sample_id,
                "split": sample.get("split") or wrapped.get("split") or "val",
                "direction": direction,
                "primary_audio_class": (reference.get("audio_metadata") or {}).get("primary_audio_class"),
                "input_text": input_text.strip(),
                "generated_text": generated_text.strip(),
                "reference_anchors": [{
                    "anchor_id": sample_id,
                    "text": reference_text.strip(),
                    "source": "paired_dataset_reference",
                }],
            })
            if len(lane_rows) >= limits[direction]:
                break
    counts[direction] = len(lane_rows)
    selected.extend(lane_rows)

if counts != limits:
    raise RuntimeError(f"unexpected usable-row counts: expected={limits} actual={counts}")
with output.open("w", encoding="utf-8") as dst:
    for row_index, row in enumerate(selected):
        dst.write(json.dumps({"row_index": row_index, **row}, ensure_ascii=False) + "\n")
print(f"wrote {output} rows={len(selected)} directions={counts}")
PYINPUT

batch_size=64
if [[ "${VARIANT}" == "reference_grounded" ]]; then
    batch_size=70
fi

set +e
timeout --signal=TERM --kill-after=60s "${EVAL_TIMEOUT}" \
    "${PYTHON}" scripts/run_geval_prometheus_criterion_v2.py \
    --input "${RUN_ROOT}/input_full.jsonl" \
    --output "${RUN_ROOT}/output.jsonl" \
    --summary-output "${RUN_ROOT}/summary.json" \
    --summary-csv-output "${RUN_ROOT}/summary.csv" \
    --variant "${VARIANT}" \
    --backend vllm \
    --model "${MODEL}" \
    --tensor-parallel-size 4 \
    --batch-size "${batch_size}" \
    --temperature 0 \
    --max-tokens 512 \
    --gpu-memory-utilization 0.85 \
    --max-model-len 4096 \
    --enforce-eager \
    --trust-remote-code \
    --cache-dir "${CACHE_ROOT}" \
    > "${RUN_ROOT}/run.log" 2>&1
run_status=$?
set -e

kill "${monitor_pid}" 2>/dev/null || true
wait "${monitor_pid}" 2>/dev/null || true

"${PYTHON}" - "${RUN_ROOT}" "${VARIANT}" "${run_status}" <<'PYREPORT'
import json
import sys
from pathlib import Path
root = Path(sys.argv[1])
variant = sys.argv[2]
process_status = int(sys.argv[3])
rows = [json.loads(line) for line in (root / "output.jsonl").open(encoding="utf-8") if line.strip()] if (root / "output.jsonl").is_file() else []
report = {
    "run_root": str(root),
    "variant": variant,
    "process_status": process_status,
    "rows": len(rows),
    "complete": sum(row.get("status") in {"complete", "complete_with_abstention"} for row in rows),
    "complete_with_abstention": sum(row.get("status") == "complete_with_abstention" for row in rows),
    "incomplete": sum(row.get("status") == "incomplete" for row in rows),
    "pending": sum(row.get("status") == "pending" for row in rows),
    "directions": {direction: sum(row.get("direction") == direction for row in rows) for direction in ("audio_to_speech", "speech_to_audio")},
}
(root / "job_report.json").write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
print(json.dumps(report, indent=2))
PYREPORT

tail -n 100 "${RUN_ROOT}/run.log" || true
echo "completed run_root=${RUN_ROOT} process_status=${run_status}"
if [[ "${run_status}" -ne 0 ]]; then
    exit "${run_status}"
fi

unfinished=$("${PYTHON}" -c 'import json,sys; report=json.load(open(sys.argv[1])); print(report["incomplete"] + report["pending"])' "${RUN_ROOT}/job_report.json")
if [[ "${unfinished}" -ne 0 ]]; then
    echo "completed_with_failed_or_pending_rows=${unfinished}" >&2
    exit 3
fi
exit 0
