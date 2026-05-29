#!/usr/bin/env bash
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

check_env() {
  local name="$1"
  local python="$2"
  shift 2
  echo "== $name =="
  MODS="$*" "$python" - <<'PY'
import importlib.util
import os
for mod in os.environ["MODS"].split():
    print(mod, bool(importlib.util.find_spec(mod)))
PY
}

check_env asr "$SALM_SCRATCH/.venvs/models/.venv-cascade-asr/bin/python" \
  torch torchaudio transformers qwen_asr soundfile nagisa soynlp
check_env caption "$SALM_SCRATCH/.venvs/models/.venv-cascade-caption/bin/python" \
  torch torchaudio torchvision transformers qwen_omni_utils soundfile
check_env text "$SALM_SCRATCH/.venvs/models/.venv-cascade-text/bin/python" \
  torch transformers accelerate safetensors
check_env f5_tts "$SALM_SCRATCH/.venvs/models/.venv-cascade-f5-tts/bin/python" \
  torch torchaudio soundfile f5_tts cached_path hydra omegaconf vocos
echo "== cosyvoice3 =="
echo "skipped for now; F5-TTS is the active cascaded T2S backend"
