#!/usr/bin/env bash
set -euo pipefail

# Rebuild only the previously incomplete stable-audio-metrics env under scratch.
# This is for clap_score_t2a and kl_passt. It is intentionally separate from
# the working AAC, Whisper, AudioLDM Eval, KAD, and UTMOSv2 envs.

SCRATCH_ROOT="${SALM_SCRATCH:-/scratch/lovenya/SpeechAudioLM}"
ENV_DIR="$SCRATCH_ROOT/.venvs/metrics/.venv-eval-stable-audio-metrics"
CODE_DIR="$SCRATCH_ROOT/code/metrics/stable-audio-metrics"
CACHE_DIR="$SCRATCH_ROOT/cache/metrics/stable_audio_metrics"

module purge
module load StdEnv/2023 gcc cuda python/3.11

mkdir -p "$SCRATCH_ROOT/.venvs/metrics" "$SCRATCH_ROOT/code/metrics" "$CACHE_DIR/pip" "$CACHE_DIR/load"

if [ "${RECREATE:-0}" = "1" ] && [ -d "$ENV_DIR" ]; then
  rm -rf "$ENV_DIR"
fi

if [ ! -d "$CODE_DIR/.git" ]; then
  git clone https://github.com/Stability-AI/stable-audio-metrics "$CODE_DIR"
fi

if [ ! -d "$ENV_DIR" ]; then
  python -m venv "$ENV_DIR"
fi

source "$ENV_DIR/bin/activate"
python -m pip install --upgrade pip setuptools wheel packaging

# Avoid the upstream requirements.txt as-is on Compute Canada: it pins versions
# such as grpcio==1.59.0 and numpy==1.23.5 that may not exist in the wheelhouse.
python -m pip install \
  requests tqdm librosa pyloudnorm h5py torchlibrosa \
  hear21passt timm laion-clap \
  --cache-dir "$CACHE_DIR/pip"

export PYTHONPATH="$CODE_DIR/src:${PYTHONPATH:-}"
export XDG_CACHE_HOME="$CACHE_DIR"
export HF_HOME="$SCRATCH_ROOT/cache/metrics/huggingface"
export TORCH_HOME="$SCRATCH_ROOT/cache/metrics/torch"

python - <<'PY'
import clap_score
import passt_kld
import laion_clap
import hear21passt

print("clap_score:", clap_score.__file__)
print("passt_kld:", passt_kld.__file__)
print("laion_clap:", laion_clap.__file__)
print("hear21passt:", hear21passt.__file__)
PY

echo "stable-audio-metrics scratch env ready: $ENV_DIR"
