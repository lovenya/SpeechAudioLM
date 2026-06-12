#!/usr/bin/env bash
set -euo pipefail

cd /scratch/l/lovenya/SpeechAudioLM

export SALM_SCRATCH="${SALM_SCRATCH:-/scratch/l/lovenya/salm_migration_clean}"
SCOPE_TAG="${SCOPE_TAG:?Set SCOPE_TAG to an immutable run label, for example 20260612_v1}"
base="$SALM_SCRATCH/results/cascaded_full_validation/ttas_val/cascaded"
out="$SALM_SCRATCH/results/evaluation_scopes/ttas_val/$SCOPE_TAG"
mkdir -p "$out"

build_scope() {
  local task="$1"
  shift
  local args=()
  local lane manifest
  for lane in "$@"; do
    manifest="$base/$task/$lane/ttas/val/manifest.jsonl"
    test -f "$manifest" || { echo "missing manifest: $manifest" >&2; return 2; }
    args+=(--manifest "$manifest")
  done
  python scripts/build_final_scope_ids.py "${args[@]}" --output "$out/${task}.json"
}

build_scope A2T \
  full_a2t_audio_flamingo_next \
  full_a2t_qwen3_omni_captioner

build_scope S2T \
  full_s2t_whisper \
  full_s2t_qwen3_asr

build_scope T2A \
  full_s2a_qwen3_asr_llama_audio_omni \
  full_s2a_qwen3_asr_llama_audiogen \
  full_s2a_qwen3_asr_llama_tangoflux \
  full_s2a_qwen3_asr_qwen32_audio_omni \
  full_s2a_qwen3_asr_qwen32_audiogen \
  full_s2a_qwen3_asr_qwen32_tangoflux \
  full_s2a_whisper_llama_audio_omni \
  full_s2a_whisper_llama_audiogen \
  full_s2a_whisper_llama_tangoflux \
  full_s2a_whisper_qwen32_audio_omni \
  full_s2a_whisper_qwen32_audiogen \
  full_s2a_whisper_qwen32_tangoflux

build_scope T2S \
  full_a2s_afn_llama_audio_omni \
  full_a2s_afn_llama_cosyvoice3 \
  full_a2s_afn_llama_f5_tts \
  full_a2s_afn_qwen32_audio_omni \
  full_a2s_afn_qwen32_cosyvoice3 \
  full_a2s_afn_qwen32_f5_tts \
  full_a2s_qwencap_llama_audio_omni \
  full_a2s_qwencap_llama_cosyvoice3 \
  full_a2s_qwencap_llama_f5_tts \
  full_a2s_qwencap_qwen32_audio_omni \
  full_a2s_qwencap_qwen32_cosyvoice3 \
  full_a2s_qwencap_qwen32_f5_tts

echo "SCOPE_ROOT=$out"
