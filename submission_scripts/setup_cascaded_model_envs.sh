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
if [ -f "$HF_HOME/token" ]; then
  export HF_TOKEN="$(cat "$HF_HOME/token")"
fi
export PYTHONDONTWRITEBYTECODE=1

ENV_ROOT="$SALM_SCRATCH/.venvs/models"
mkdir -p "$ENV_ROOT"

create_env() {
  local env="$1"
  if [ ! -x "$env/bin/python" ]; then
    python -m venv "$env"
  fi
  "$env/bin/python" -m pip install --upgrade pip
}

ASR_ENV="$ENV_ROOT/.venv-cascade-asr"
CAPTION_ENV="$ENV_ROOT/.venv-cascade-caption"
TEXT_ENV="$ENV_ROOT/.venv-cascade-text"
F5_ENV="$ENV_ROOT/.venv-cascade-f5-tts"

create_env "$ASR_ENV"
"$ASR_ENV/bin/python" -m pip install --upgrade \
  'torch==2.6.0+computecanada' 'torchaudio==2.6.0+computecanada' \
  'transformers==5.3.0+computecanada' accelerate soundfile librosa sox qwen-omni-utils
"$ASR_ENV/bin/python" -m pip install --no-deps --upgrade qwen-asr

create_env "$CAPTION_ENV"
"$CAPTION_ENV/bin/python" -m pip install --upgrade \
  'torch==2.6.0+computecanada' 'torchaudio==2.6.0+computecanada' \
  'transformers==5.3.0+computecanada' accelerate soundfile qwen-omni-utils

create_env "$TEXT_ENV"
"$TEXT_ENV/bin/python" -m pip install --upgrade \
  'torch==2.6.0+computecanada' 'transformers==5.3.0+computecanada' \
  accelerate sentencepiece protobuf safetensors

create_env "$F5_ENV"
"$F5_ENV/bin/python" -m pip install --upgrade \
  'torch==2.6.0+computecanada' 'torchaudio==2.6.0+computecanada' \
  soundfile cached_path hydra-core omegaconf vocos huggingface_hub matplotlib \
  numpy==1.26.4 librosa pydub pypinyin safetensors tomli torchdiffeq \
  transformers transformers_stream_generator unidecode x_transformers
"$F5_ENV/bin/python" -m pip install --no-deps -e "$SALM_SCRATCH/code/cascaded/F5-TTS"

"$F5_ENV/bin/python" - <<'PY'
from pathlib import Path
init_file = Path("/scratch/l/lovenya/salm_migration_clean/code/cascaded/F5-TTS/src/f5_tts/model/__init__.py")
text = init_file.read_text()
text = text.replace("from f5_tts.model.trainer import Trainer\n", "")
text = text.replace(
    "__all__ = [\"CFM\", \"UNetT\", \"DiT\", \"MMDiT\", \"Trainer\"]",
    "__all__ = [\"CFM\", \"UNetT\", \"DiT\", \"MMDiT\"]",
)
init_file.write_text(text)
print(f"Patched F5-TTS inference init at {init_file}")
PY

"$F5_ENV/bin/python" - <<'PY'
from pathlib import Path
utils_file = Path("/scratch/l/lovenya/salm_migration_clean/code/cascaded/F5-TTS/src/f5_tts/model/utils.py")
text = utils_file.read_text()
text = text.replace(
    "import rjieba\n",
    "try:\n    import rjieba\nexcept ImportError:\n    class _FallbackRjieba:\n        @staticmethod\n        def cut(text):\n            return [text]\n\n    rjieba = _FallbackRjieba()\n",
)
utils_file.write_text(text)
print(f"Patched F5-TTS rjieba fallback at {utils_file}")
PY
"$F5_ENV/bin/python" - <<'PY'
from huggingface_hub import hf_hub_download
from pathlib import Path

target = Path("/scratch/l/lovenya/salm_migration_clean/models/cascaded/tts/vocos-mel-24khz")
target.mkdir(parents=True, exist_ok=True)
for filename in ("config.yaml", "pytorch_model.bin"):
    hf_hub_download(
        repo_id="charactr/vocos-mel-24khz",
        filename=filename,
        local_dir=str(target),
    )
print(f"Vocos cached at {target}")
PY

# CosyVoice3 is intentionally skipped for now.
# Its wetext/kaldifst frontend dependency currently fails to build on TamIA.
# F5-TTS remains the active cascaded T2S backend.

echo "Cascaded model env setup complete:"
printf '  ASR:     %s\n' "$ASR_ENV"
printf '  Caption: %s\n' "$CAPTION_ENV"
printf '  Text:    %s\n' "$TEXT_ENV"
printf '  F5-TTS:  %s\n' "$F5_ENV"
printf '  Cosy:    skipped for now; F5-TTS is active\n'
