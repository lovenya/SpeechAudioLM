#!/bin/bash
#SBATCH --job-name=geval-v2-5x2
#SBATCH --account=aip-ravanelm
#SBATCH --partition=gpubase_bynode_b1
#SBATCH --time=01:00:00
#SBATCH --nodes=1
#SBATCH --gpus-per-node=h100:4
#SBATCH --cpus-per-task=8
#SBATCH --mem=128G
#SBATCH --output=/scratch/l/lovenya/salm_migration_clean/logs/slurm/geval-v2-5x2-%j.out

set -uo pipefail

REPO=/scratch/l/lovenya/SpeechAudioLM
PYTHON=/home/l/lovenya/.venvs/speechaudiolm/.venv-eval-prometheus-vllm-py312/bin/python
MODEL=/scratch/l/lovenya/salm_migration_clean/models/eval/prometheus-8x7b-v2.0
A2S_MANIFEST=/scratch/l/lovenya/salm_migration_clean/results/cascaded_full_validation/ttas_val/cascaded/T2T/full_a2s_afn_qwen32_t2t/ttas/val/manifest.jsonl
S2A_MANIFEST=/scratch/l/lovenya/salm_migration_clean/results/cascaded_full_validation/ttas_val/cascaded/T2T/full_s2a_whisper_qwen32_t2t/ttas/val/manifest.jsonl
RUN_ID="$(TZ=America/Toronto date +%Y%m%dT%H%M%S%z)_${SLURM_JOB_ID}"
RUN_ROOT=/scratch/l/lovenya/salm_migration_clean/results/geval_v2_demo_5x2/${RUN_ID}
LOG_ROOT=/scratch/l/lovenya/salm_migration_clean/logs/resources

mkdir -p "${RUN_ROOT}" "${LOG_ROOT}"

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

cleanup_workers() {
    pkill -TERM -u "${USER}" -f 'EngineCore|Worker_TP|run_geval_prometheus_criterion_v2.py' 2>/dev/null || true
    sleep 5
    pkill -KILL -u "${USER}" -f 'EngineCore|Worker_TP|run_geval_prometheus_criterion_v2.py' 2>/dev/null || true
}
trap cleanup_workers EXIT INT TERM

echo "run_root=${RUN_ROOT}"
echo "hostname=$(hostname)"
echo "job_id=${SLURM_JOB_ID}"
module list 2>&1
nvidia-smi -L

"${PYTHON}" - <<'PY'
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
PY
import_status=$?
if [[ ${import_status} -ne 0 ]]; then
    echo "environment_validation_failed status=${import_status}"
    exit "${import_status}"
fi

(
    while true; do
        printf '{"timestamp":"%s","job_id":"%s","gpu_csv":' "$(date --iso-8601=seconds)" "${SLURM_JOB_ID}"
        nvidia-smi --query-gpu=index,memory.used,memory.total,utilization.gpu,power.draw --format=csv,noheader,nounits \
            | "${PYTHON}" -c 'import json,sys; print(json.dumps([line.strip() for line in sys.stdin if line.strip()]))'
        sleep 30
    done
) > "${LOG_ROOT}/geval_v2_demo_5x2_${SLURM_JOB_ID}.jsonl" 2>&1 &
monitor_pid=$!
trap 'kill "${monitor_pid}" 2>/dev/null || true; cleanup_workers' EXIT INT TERM

"${PYTHON}" - "${A2S_MANIFEST}" "${S2A_MANIFEST}" "${RUN_ROOT}/input_5x2.jsonl" <<'PY'
import json
import sys
from pathlib import Path

specs = [
    (Path(sys.argv[1]), "audio_to_speech", "speech_text"),
    (Path(sys.argv[2]), "speech_to_audio", "audio_cap"),
]
output = Path(sys.argv[3])
selected = []

for manifest, direction, reference_field in specs:
    lane_rows = []
    with manifest.open(encoding="utf-8") as src:
        for line in src:
            wrapped = json.loads(line)
            sample = wrapped.get("sample", wrapped)
            if sample.get("status", "success") != "success":
                continue
            stages = sample.get("stages")
            if not isinstance(stages, list) or not stages:
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
                "direction": direction,
                "input_text": input_text.strip(),
                "generated_text": generated_text.strip(),
                "reference_anchors": [{
                    "anchor_id": sample_id,
                    "text": reference_text.strip(),
                    "source": "paired_dataset_reference",
                }],
            })
            if len(lane_rows) == 5:
                break
    if len(lane_rows) != 5:
        raise RuntimeError(f"{manifest} supplied only {len(lane_rows)}/5 usable {direction} rows")
    selected.extend(lane_rows)

with output.open("w", encoding="utf-8") as dst:
    for row_index, row in enumerate(selected):
        dst.write(json.dumps({"row_index": row_index, **row}, ensure_ascii=False) + "\n")
print(f"wrote {output} rows={len(selected)}")
PY

run_variant() {
    local variant=$1
    local batch_size=$2
    echo "starting_variant=${variant} timestamp=$(date --iso-8601=seconds)"
    set +e
    timeout --signal=TERM --kill-after=30s 25m \
        "${PYTHON}" scripts/run_geval_prometheus_criterion_v2.py \
        --input "${RUN_ROOT}/input_5x2.jsonl" \
        --output "${RUN_ROOT}/${variant}.jsonl" \
        --summary-output "${RUN_ROOT}/${variant}_summary.json" \
        --summary-csv-output "${RUN_ROOT}/${variant}_summary.csv" \
        --variant "${variant}" \
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
        --cache-dir "${RUN_ROOT}/cache_${variant}" \
        > "${RUN_ROOT}/${variant}.log" 2>&1
    local status=$?
    set -e
    echo "variant=${variant} status=${status} timestamp=$(date --iso-8601=seconds)"
    tail -n 80 "${RUN_ROOT}/${variant}.log" || true
    cleanup_workers
    return 0
}

run_variant plausibility 12
run_variant reference_grounded 14

kill "${monitor_pid}" 2>/dev/null || true
wait "${monitor_pid}" 2>/dev/null || true

"${PYTHON}" - "${RUN_ROOT}" <<'PY'
import json
import sys
from pathlib import Path

root = Path(sys.argv[1])
report = {"run_root": str(root), "variants": {}}
for variant in ("plausibility", "reference_grounded"):
    path = root / f"{variant}.jsonl"
    rows = []
    if path.is_file():
        rows = [json.loads(line) for line in path.open(encoding="utf-8") if line.strip()]
    report["variants"][variant] = {
        "rows": len(rows),
        "complete": sum(row.get("status") in {"complete", "complete_with_abstention"} for row in rows),
        "complete_with_abstention": sum(row.get("status") == "complete_with_abstention" for row in rows),
        "incomplete": sum(row.get("status") == "incomplete" for row in rows),
        "pending": sum(row.get("status") == "pending" for row in rows),
        "directions": {
            direction: sum(row.get("direction") == direction for row in rows)
            for direction in ("audio_to_speech", "speech_to_audio")
        },
    }
(root / "job_report.json").write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
print(json.dumps(report, indent=2))
PY

echo "completed run_root=${RUN_ROOT}"
