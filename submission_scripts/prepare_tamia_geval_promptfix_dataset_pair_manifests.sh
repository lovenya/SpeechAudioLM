#!/usr/bin/env bash
set -euo pipefail

cd /scratch/l/lovenya/SpeechAudioLM

export SALM_SCRATCH="${SALM_SCRATCH:-/scratch/l/lovenya/salm_migration_clean}"
export SALM_RESULTS_ROOT="${SALM_RESULTS_ROOT:-$SALM_SCRATCH/results}"
export SALM_TEMPORARY_RESULTS_ROOT="${SALM_TEMPORARY_RESULTS_ROOT:-$SALM_SCRATCH/temporary/results}"
export SALM_CACHE_ROOT="${SALM_CACHE_ROOT:-$SALM_SCRATCH/cache}"
export SALM_MODEL_ROOT="${SALM_MODEL_ROOT:-$SALM_SCRATCH/models}"
export SALM_ENVIRONMENT_ROOT="${SALM_ENVIRONMENT_ROOT:-$SALM_SCRATCH/.venvs}"

RUN_GROUP="${RUN_GROUP:-dataset_pair_geval_promptfix_$(TZ=America/Toronto date +%Y%m%d_%H%M)}"
MANIFEST_ROOT="${MANIFEST_ROOT:-$SALM_TEMPORARY_RESULTS_ROOT/dataset_pair_eval_manifests/ttas/test/$RUN_GROUP}"

if [ -e "$MANIFEST_ROOT" ]; then
  echo "error: manifest root already exists: $MANIFEST_ROOT" >&2
  echo "Choose a new RUN_GROUP or remove the existing directory after auditing it." >&2
  exit 2
fi

mkdir -p "$MANIFEST_ROOT"

python scripts/build_dataset_pair_eval_manifest.py \
  --split test \
  --stage t2t \
  --direction a2s \
  --output "$MANIFEST_ROOT/t2t_a2s.jsonl"

python scripts/build_dataset_pair_eval_manifest.py \
  --split test \
  --stage t2t \
  --direction s2a \
  --output "$MANIFEST_ROOT/t2t_s2a.jsonl"

python scripts/build_dataset_pair_eval_manifest.py \
  --split test \
  --stage t2s \
  --direction a2s \
  --output "$MANIFEST_ROOT/t2s_a2s.jsonl"

python scripts/build_dataset_pair_eval_manifest.py \
  --split test \
  --stage t2a \
  --direction s2a \
  --output "$MANIFEST_ROOT/t2a_s2a.jsonl"

wc -l "$MANIFEST_ROOT"/t2t_a2s.jsonl "$MANIFEST_ROOT"/t2t_s2a.jsonl "$MANIFEST_ROOT"/t2s_a2s.jsonl "$MANIFEST_ROOT"/t2a_s2a.jsonl

env_file="$MANIFEST_ROOT/run_env.sh"
{
  printf 'export RUN_GROUP=%q\n' "$RUN_GROUP"
  printf 'export MANIFEST_ROOT=%q\n' "$MANIFEST_ROOT"
} > "$env_file"

echo "RUN_GROUP=$RUN_GROUP"
echo "MANIFEST_ROOT=$MANIFEST_ROOT"
echo "env_file=$env_file"
