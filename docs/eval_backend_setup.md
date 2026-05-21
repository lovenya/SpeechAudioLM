# Evaluation Backend Setup

This document is the setup checklist for metric backends. It is intentionally
separate from `evaluate.py` implementation: first prepare environments, caches,
and smoke tests; then wire adapters one backend at a time.

Do not commit metric weights, generated outputs, caches, downloaded source
repos, or virtual environments.

## Common Paths

Use separate environments when dependency stacks conflict:

```text
.venvs/metrics/.venv-eval-whisper
.venvs/metrics/.venv-eval-aac
.venvs/metrics/.venv-eval-utmos
.venvs/metrics/.venv-eval-dnsmos
.venvs/metrics/.venv-eval-audioldm-eval
.venvs/metrics/.venv-eval-kadtk
```

Keep backend assets outside git:

```text
/project/rrg-csubakan/lovenya/models/metrics
/project/rrg-csubakan/lovenya/cache/metrics
/project/rrg-csubakan/lovenya/src/metrics
```

Standard module setup:

```bash
cd /lustre09/project/6090520/lovenya/SpeechAudioLM
module load StdEnv/2023 gcc cuda rust arrow python/3.10.13
```

Smoke-test scripts live in:

```text
scripts/eval_backend_smoke/
```

Current evaluator status:

- `evaluate.py` already has `--metrics`, `--dry-run`, `--max-samples`, and
  `--cache-dir`.
- Resume and overwrite are not implemented yet. Add them before full evaluation
  runs if repeated long corpus metrics become likely.
- Offline compute-node runs require every model/checkpoint/cache to be prepared
  before the job starts.

## 1. OpenAI Whisper Large-v3

Supports:

- `asr_wer`
- `dwer`

Install:

```bash
python -m venv .venvs/metrics/.venv-eval-whisper
source .venvs/metrics/.venv-eval-whisper/bin/activate
python -m pip install --upgrade pip
python -m pip install -U openai-whisper
```

Assets to download:

```bash
mkdir -p /project/rrg-csubakan/lovenya/models/metrics/whisper
python - <<'PY'
import whisper
model = whisper.load_model(
    "large-v3",
    download_root="/project/rrg-csubakan/lovenya/models/metrics/whisper",
)
print("loaded", model.dims)
PY
```

Cache directory:

```text
/project/rrg-csubakan/lovenya/models/metrics/whisper
```

Smoke test:

```bash
source .venvs/metrics/.venv-eval-whisper/bin/activate
python scripts/eval_backend_smoke/smoke_whisper_asr.py
python scripts/eval_backend_smoke/smoke_whisper_asr.py --load-model
python scripts/eval_backend_smoke/smoke_whisper_asr.py --load-model --audio /path/to/speech.wav
```

Expected output shape:

- Import-only smoke: JSON with `available: true`.
- Model-load smoke: JSON with `loaded: true`.
- Audio smoke: JSON with one `transcription` string.

Known caveats:

- `large-v3` is heavy; use a compute node for model loading/transcription.
- `ffmpeg` must be available for normal Whisper audio decoding.
- The evaluator must record `large-v3`, cache path, device, package version, and
  decoding policy.
- dWER remains directional:
  `WER(ASR(reference_speech), ASR(generated_speech))`.

Node:

- Import-only: login node is acceptable.
- Model load/transcription: compute node.

Primary references:

- https://github.com/openai/whisper

## 2. aac-metrics

Supports:

- `spider`
- `mace`
- later optional `fense`, `spider_max`

Install:

```bash
python -m venv .venvs/metrics/.venv-eval-aac
source .venvs/metrics/.venv-eval-aac/bin/activate
python -m pip install --upgrade pip
python -m pip install aac-metrics
```

Assets to download:

```bash
mkdir -p /project/rrg-csubakan/lovenya/cache/metrics/aac_metrics
export AAC_METRICS_CACHE=/project/rrg-csubakan/lovenya/cache/metrics/aac_metrics
export XDG_CACHE_HOME=/project/rrg-csubakan/lovenya/cache/metrics/aac_metrics
export TORCH_HOME=/project/rrg-csubakan/lovenya/cache/metrics/torch
export SENTENCE_TRANSFORMERS_HOME=/project/rrg-csubakan/lovenya/cache/metrics/sentence_transformers
aac-metrics-download --cache_path "$AAC_METRICS_CACHE"
aac-metrics-info
```

Cache directory:

```text
/project/rrg-csubakan/lovenya/cache/metrics/aac_metrics
```

Smoke test:

```bash
module load java/11.0.22
source .venvs/metrics/.venv-eval-aac/bin/activate
export AAC_METRICS_CACHE=/project/rrg-csubakan/lovenya/cache/metrics/aac_metrics
unset JAVA_TOOL_OPTIONS
python scripts/eval_backend_smoke/smoke_aac_metrics.py --cache-dir "$AAC_METRICS_CACHE" --metrics spider --device cpu
```

Expected output shape:

- JSON with `corpus_score_keys` and `sample_score_keys`.
- For SPIDEr, expect keys `cider_d`, `spice`, and `spider` for both corpus and
  sample scores. The smoke fixture uses two samples because CIDEr-D rejects a
  one-sample candidate set.
- For MACE, expect one score per candidate once audio paths/assets are wired.

Known caveats:

- SPIDEr/MACE assets may require Java and cached external code/assets.
- On this cluster, load Java before SPIDEr/MACE tokenization:
  `module load java/11.0.22`. Java 17 is visible on the cluster but rejected by
  `aac-metrics` tokenization checks, which expect a major version in `[8, 13]`.
- Unset `JAVA_TOOL_OPTIONS` before running `aac-metrics`. Its Java version
  parser can misread the cluster banner `Picked up JAVA_TOOL_OPTIONS: ...` as a
  version string.
- `aac-metrics-download` defaults to a user cache such as
  `/home/lovenya/.cache`. For offline compute runs, pass `--cache_path` and use
  the same path in the smoke/evaluator config.
- MACE needs candidate caption, reference `source.audio_cap`, and reference
  `source.audio_file`.
- SPIDEr needs candidate caption and reference `source.audio_cap`.
- The current evaluator should fail only when the selected metric needs this
  backend.

Node:

- Install/download: internet-capable node.
- Tiny SPIDEr smoke: login node is acceptable after assets exist.
- MACE/CLAP-backed scoring over many samples: compute node.

Primary references:

- https://github.com/Labbeti/aac-metrics
- https://github.com/satvik-dixit/mace

## 3. CLAP Backend

Supports:

- `clap_similarity`

Decision status:

- Do not install CLAP yet.
- Need one consistent backend/checkpoint choice before implementation.

Options:

- Option 1: use `aac-metrics` CLAPSim / `msclap` for AAC-style caption/audio
  similarity.
- Option 2: use LAION-CLAP or an audio-generation evaluation backend for T2A
  similarity.

Install:

```text
Not selected yet.
```

Assets to download:

```text
Not selected yet.
```

Cache directory:

```text
TBD after backend choice.
```

Smoke test:

```bash
python scripts/eval_backend_smoke/smoke_clap_backend.py
```

Expected output shape:

- JSON showing which candidate packages are currently importable.
- No model scores yet.

Known caveats:

- CLAP scores can differ substantially by backend/checkpoint/sample-rate policy.
- Choose one backend before writing adapters or reports.
- Record model checkpoint, config, sample-rate/resampling policy, cache path, and
  package/git version in `run_config`.

Node:

- Current decision smoke: login node.
- Future model load/scoring: compute node.

Primary references:

- https://github.com/LAION-AI/CLAP

TODO:

- Select one CLAP backend/checkpoint before implementation.

## 4. UTMOS

Supports:

- `utmos`

Install:

```bash
python -m venv .venvs/metrics/.venv-eval-utmos
source .venvs/metrics/.venv-eval-utmos/bin/activate
python -m pip install --upgrade pip
python -m pip install torch torchaudio librosa
```

Assets to download:

```bash
mkdir -p /project/rrg-csubakan/lovenya/models/metrics/torchhub
export TORCH_HOME=/project/rrg-csubakan/lovenya/models/metrics/torchhub
python - <<'PY'
import torch
predictor = torch.hub.load("tarepan/SpeechMOS:v1.2.0", "utmos22_strong", trust_repo=True)
wave = torch.zeros((1, 16000))
print(predictor(wave, sr=16000))
PY
```

Cache directory:

```text
/project/rrg-csubakan/lovenya/models/metrics/torchhub
```

Smoke test:

```bash
source .venvs/metrics/.venv-eval-utmos/bin/activate
export TORCH_HOME=/project/rrg-csubakan/lovenya/models/metrics/torchhub
python scripts/eval_backend_smoke/smoke_utmos.py
python scripts/eval_backend_smoke/smoke_utmos.py --load-model
```

Expected output shape:

- Import-only smoke: JSON with Torch version.
- Model smoke: JSON with `score_shape: [1]`.

Known caveats:

- SpeechMOS loads through `torch.hub`, so the repo/checkpoint must be cached
  before offline compute runs.
- UTMOS is for generated speech quality, not environmental audio quality.
- Record checkpoint/tag `tarepan/SpeechMOS:v1.2.0`, specifier
  `utmos22_strong`, sample rate, and cache path.

Node:

- Import-only: login node.
- Model load/full scoring: compute node.

Primary references:

- https://github.com/tarepan/SpeechMOS

## 5. DNSMOS

Supports:

- `dnsmos`

Install:

```bash
python -m venv .venvs/metrics/.venv-eval-dnsmos
source .venvs/metrics/.venv-eval-dnsmos/bin/activate
python -m pip install --upgrade pip
python -m pip install torchmetrics onnxruntime librosa
```

Assets to download:

```bash
mkdir -p /project/rrg-csubakan/lovenya/models/metrics/dnsmos/DNSMOS
mkdir -p /project/rrg-csubakan/lovenya/models/metrics/dnsmos/pDNSMOS
wget -O /project/rrg-csubakan/lovenya/models/metrics/dnsmos/DNSMOS/model_v8.onnx \
  https://raw.githubusercontent.com/microsoft/DNS-Challenge/master/DNSMOS/DNSMOS/model_v8.onnx
wget -O /project/rrg-csubakan/lovenya/models/metrics/dnsmos/DNSMOS/sig_bak_ovr.onnx \
  https://raw.githubusercontent.com/microsoft/DNS-Challenge/master/DNSMOS/DNSMOS/sig_bak_ovr.onnx
wget -O /project/rrg-csubakan/lovenya/models/metrics/dnsmos/pDNSMOS/sig_bak_ovr.onnx \
  https://raw.githubusercontent.com/microsoft/DNS-Challenge/master/DNSMOS/pDNSMOS/sig_bak_ovr.onnx
```

Cache directory:

```text
/project/rrg-csubakan/lovenya/models/metrics/dnsmos
```

Smoke test:

```bash
source .venvs/metrics/.venv-eval-dnsmos/bin/activate
python scripts/eval_backend_smoke/smoke_dnsmos.py
```

Expected output shape:

- JSON listing DNSMOS functions available in TorchMetrics.
- `asset_status` should be `true` for all three ONNX files.
- Real DNSMOS scoring should return MOS-like scalar scores per audio once the
  adapter is implemented.

Known caveats:

- DNSMOS implementations often resample to 16 kHz and may use fixed-length
  chunks/windows.
- The evaluator must record ONNX asset paths and preprocessing behavior.
- TorchMetrics may try to download assets unless we point it to the prepared
  directory or adapt the loader carefully.

Node:

- Asset presence smoke: login node.
- Full scoring: compute node if many files.

Primary references:

- https://github.com/microsoft/DNS-Challenge
- https://github.com/Lightning-AI/torchmetrics/blob/master/src/torchmetrics/functional/audio/dnsmos.py

## 6. audioldm_eval

Supports:

- `fad`
- `kld`
- `inception_score`

Install:

```bash
python -m venv .venvs/metrics/.venv-eval-audioldm-eval
source .venvs/metrics/.venv-eval-audioldm-eval/bin/activate
python -m pip install --upgrade pip
python -m pip install --upgrade setuptools wheel
python -m pip install torch torchaudio transformers scikit-image torchlibrosa absl-py scipy tqdm librosa soundfile
python -m pip install ssr_eval --no-deps
python -m pip install git+https://github.com/haoheliu/audioldm_eval --no-deps
```

Assets to download:

```bash
mkdir -p /project/rrg-csubakan/lovenya/cache/metrics/audioldm_eval
export XDG_CACHE_HOME=/project/rrg-csubakan/lovenya/cache/metrics
export TORCH_HOME=/project/rrg-csubakan/lovenya/cache/metrics/torch
python - <<'PY'
from audioldm_eval import EvaluationHelper
import torch
helper = EvaluationHelper(16000, torch.device("cpu"))
print(type(helper).__name__)
PY
```

Cache directory:

```text
/project/rrg-csubakan/lovenya/cache/metrics/audioldm_eval
```

Smoke test:

```bash
source .venvs/metrics/.venv-eval-audioldm-eval/bin/activate
python scripts/eval_backend_smoke/smoke_audioldm_eval.py
python scripts/eval_backend_smoke/smoke_audioldm_eval.py \
  --run \
  --reference-dir /path/to/reference_wavs \
  --generated-dir /path/to/generated_wavs \
  --device cuda:0
```

Expected output shape:

- Import-only smoke: JSON with `helper_imported: true`.
- Folder smoke: JSON with a `metrics` dictionary. Expected keys depend on
  backend mode, but should include corpus-level audio-generation metrics such as
  FAD, KL/KLD, and Inception Score when available.

Known caveats:

- The normal `pip install git+https://github.com/haoheliu/audioldm_eval`
  dependency path can pull PyPI `wave`, which pulls Python 2-era
  `MySQL-python` and fails on Python 3 with `No module named 'ConfigParser'`.
  Use the `--no-deps` install above and install Python 3-compatible
  dependencies manually.
- FAD/KLD/Inception Score are corpus-level only. Do not write fake per-sample
  values.
- The toolkit accepts reference/generated folders; paired vs unpaired behavior
  depends on filenames and counts.
- Resampling can affect FAD/KLD/IS; record sample rate and preprocessing policy.
- Cache generated embedding files outside git.

Node:

- Import-only: login node.
- Any folder metric run: compute node.

Primary references:

- https://github.com/haoheliu/audioldm_eval

## 7. kadtk

Supports:

- `kad`

Install:

```bash
python -m venv .venvs/metrics/.venv-eval-kadtk
source .venvs/metrics/.venv-eval-kadtk/bin/activate
python -m pip install --upgrade pip
python -m pip install kadtk
```

Assets to download:

```bash
mkdir -p /project/rrg-csubakan/lovenya/cache/metrics/kadtk
export XDG_CACHE_HOME=/project/rrg-csubakan/lovenya/cache/metrics
export TORCH_HOME=/project/rrg-csubakan/lovenya/cache/metrics/torch
kadtk --help
```

Cache directory:

```text
/project/rrg-csubakan/lovenya/cache/metrics/kadtk
```

Smoke test:

```bash
source .venvs/metrics/.venv-eval-kadtk/bin/activate
python scripts/eval_backend_smoke/smoke_kadtk.py
python scripts/eval_backend_smoke/smoke_kadtk.py \
  --run \
  --reference-dir /path/to/reference_wavs \
  --generated-dir /path/to/generated_wavs
```

Expected output shape:

- Import-only smoke: JSON with a detected `kadtk` CLI path.
- Folder smoke: JSON with `ran: true` and a preview of the KAD CLI output.
- Final adapter should store corpus-level KAD in `summary.json`, not per-sample.

Known caveats:

- Official docs state Python `>=3.9,<3.12`, Torch `>=2.1,<2.6`, and TensorFlow
  `>=2.0`. This may conflict with the current cluster Torch 2.6 baseline env,
  so keep KAD in its own venv.
- KAD direction matters in practice: reference set should be real/ground-truth
  audio and target set should be generated audio.
- Use `--force-emb-encode` only when intentionally ignoring cached embeddings.

Node:

- Import-only/CLI help: login node.
- Corpus KAD run: compute node.

Primary references:

- https://github.com/YoonjinXD/kadtk

## Immediate TODOs Before Backend Adapters

- Choose one CLAP backend/checkpoint and update `configs/eval_metrics.json`.
- Add evaluator `--resume` and `--overwrite` behavior before long corpus runs.
- Decide whether each backend will have a dedicated adapter env or whether some
  can safely share an env.
- After each backend smoke passes, record package versions, git commits,
  checkpoint paths, sample-rate policy, and cache dirs in `configs/eval_metrics.json`.
