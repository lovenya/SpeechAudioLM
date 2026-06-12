#!/usr/bin/env bash
#SBATCH --job-name=salm-qwen-spider-retry-v2
#SBATCH --account=aip-ravanelm
#SBATCH --partition=cpubase_bynode_b1
#SBATCH --time=01:00:00
#SBATCH --cpus-per-task=32
#SBATCH --mem=500G
#SBATCH --output=/scratch/l/lovenya/salm_migration_clean/logs/slurm/%x-%j.out

set -euo pipefail

cd /scratch/l/lovenya/SpeechAudioLM
module --force purge
module load StdEnv/2023 gcc opencv/4.13.0 python/3.12.4 scipy-stack java/11.0.22

unset JAVA_TOOL_OPTIONS
export AAC_METRICS_JAVA_PATH="$(command -v java)"
test -x "$AAC_METRICS_JAVA_PATH" || { echo "Java executable unavailable"; exit 1; }

export SALM_SCRATCH="${SALM_SCRATCH:-/scratch/l/lovenya/salm_migration_clean}"
export PYTHONNOUSERSITE=1 PYTHONDONTWRITEBYTECODE=1 TOKENIZERS_PARALLELISM=false
export HF_HOME="$SALM_SCRATCH/cache/metrics/huggingface"
export HF_HUB_CACHE="$HF_HOME" HUGGINGFACE_HUB_CACHE="$HF_HOME" TRANSFORMERS_CACHE="$HF_HOME"
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1
export TORCH_HOME="$SALM_SCRATCH/cache/metrics/torch"
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-32}"
export MKL_NUM_THREADS="${MKL_NUM_THREADS:-32}"
export NUMEXPR_NUM_THREADS="${NUMEXPR_NUM_THREADS:-32}"
export RESOURCE_INTERVAL="${RESOURCE_INTERVAL:-30}"

SCOPE_TAG="${SCOPE_TAG:?Submit with --export=ALL,SCOPE_TAG=<prepared-scope-tag>}"
job_id="${SLURM_JOB_ID:-manual}"
log_root="$SALM_SCRATCH/logs/eval_packed/qwen_spider_retry_v2/$job_id"
mkdir -p "$log_root" "$SALM_SCRATCH/logs/slurm"

source submission_scripts/lib_resource_monitor.sh
start_resource_monitor qwen_spider_retry_v2

spider_config="$log_root/eval_metrics_spider_192g.json"
python - configs/eval_metrics.json "$spider_config" <<'PY'
import json
import sys

source, destination = sys.argv[1:]
with open(source, encoding="utf-8") as handle:
    config = json.load(handle)
config["metrics"]["spider"]["java_max_memory"] = "192G"
with open(destination, "w", encoding="utf-8") as handle:
    json.dump(config, handle, indent=2, sort_keys=True)
    handle.write("\n")
PY

manifest="$SALM_SCRATCH/results/cascaded_full_validation/ttas_val/cascaded/A2T/full_a2t_qwen3_omni_captioner/ttas/val/manifest.jsonl"
scope="$SALM_SCRATCH/results/evaluation_scopes/ttas_val/$SCOPE_TAG/A2T.json"
output="$SALM_SCRATCH/results/evals/final_metrics_v2/$SCOPE_TAG/A2T/full_a2t_qwen3_omni_captioner/spider"
cache="$SALM_SCRATCH/cache/metrics/eval_runtime"

test -f "$manifest" || { echo "missing manifest: $manifest"; exit 2; }
test -f "$scope" || { echo "missing final scope: $scope"; exit 2; }

extra_args=()
if [ "${DRY_RUN:-0}" = "1" ]; then
  extra_args+=(--dry-run)
fi

python evaluate.py \
  --model cascaded \
  --task A2T \
  --dataset ttas \
  --split val \
  --manifest "$manifest" \
  --metrics spider \
  --evaluation-mode final \
  --common-sample-ids "$scope" \
  --metrics-config "$spider_config" \
  --output-root "$output" \
  --cache-dir "$cache" \
  --device cpu \
  "${extra_args[@]}" \
  >"$log_root/spider.log" 2>&1

stop_resource_monitor
echo "qwen_spider_retry_v2_complete log=$log_root/spider.log output=$output"
