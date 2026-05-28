# Evaluation Backend Setup

This document is the setup checklist for metric backends. It is intentionally
separate from `evaluate.py` implementation: first prepare environments, caches,
and smoke tests; then wire adapters one backend at a time.

Do not commit metric weights, generated outputs, caches, downloaded source
repos, or virtual environments.

## Common Paths

Use separate environments when dependency stacks conflict:

```text
.venv-eval-whisper
.venvs/.venv-eval-aac
.venvs/metrics/.venv-eval-audioldm-eval
.venvs/metrics/.venv-eval-kadtk
.venvs/metrics/.venv-eval-utmosv2
/scratch/lovenya/SpeechAudioLM/.venvs/metrics/.venv-eval-stable-audio-metrics
```

Keep backend assets outside git:

```text
/scratch/lovenya/SpeechAudioLM/models/metrics
/scratch/lovenya/SpeechAudioLM/cache/metrics
/scratch/lovenya/SpeechAudioLM/code/metrics
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
- TamIA job `309283` passed the currently wired generated-audio and
  generated-speech metric groups through `evaluate.py`: AudioLDM Eval
  `fad,inception_score`, KADTK `kad`, stable-audio-metrics
  `clap_score_t2a,kl_passt`, Whisper `asr_wer,dwer`, and UTMOSv2 `utmos`.
- Earlier AAC smokes covered SPIDEr and the MS-CLAP-backed AAC path; rerun
  `mace,aac_clapsim,spider` on A2T/audio-captioning manifests once those
  baseline outputs are ready on TamIA.
- DNSMOS remains the main missing generated-speech metric adapter.
- Resume and overwrite are not implemented yet. Add them before full evaluation
  runs if repeated long corpus metrics become likely.
- Offline compute-node runs require every model/checkpoint/cache to be prepared
  before the job starts.
- AudioLDM Eval hard-codes PANN Cnn14 checkpoints under
  `~/.cache/audioldm_eval/ckpt`; TamIA metric smoke scripts set `HOME` to
  `$SALM_SCRATCH/cache/metrics/runtime_home`, so cache `Cnn14_mAP=0.431.pth`
  and `Cnn14_16k_mAP=0.438.pth` under that path before running
  `fad,inception_score`.
- With PyTorch 2.6, those legacy PANN checkpoints must be loaded with
  `weights_only=False`; `scripts/eval_backend_worker.py` applies this inside
  the AudioLDM Eval worker only.
- AudioLDM Eval asserts each generated/reference directory has more than one
  file. For one-sample smoke jobs, the worker duplicates the staged generated
  and reference WAVs inside the temporary work directory before calling the
  backend; use multi-sample runs for meaningful corpus values.
- LAION-CLAP checkpoints used by stable-audio-metrics also need
  `weights_only=False` under PyTorch 2.6; the worker applies this only around
  the trusted local checkpoint load.
- LAION-CLAP default tokenization squeezes the batch dimension for a single
  text input in this environment. The worker passes a batch-preserving tokenizer
  for one-sample `clap_score_t2a` smokes.
- `kl_passt` needs the PaSST checkpoint
  `passt-s-f128-p16-s10-ap.476-swa.pt` under
  `$SALM_SCRATCH/cache/metrics/torch/hub/checkpoints`; otherwise
  `hear21passt` tries GitHub from the compute node.
- On TamIA, UTMOS offline `facebook/wav2vec2-base` loading requires `HF_HOME`,
  `HF_HUB_CACHE`, `HUGGINGFACE_HUB_CACHE`, and `TRANSFORMERS_CACHE` to point
  at `$SALM_SCRATCH/cache/metrics/huggingface`. `HF_HOME` alone was not
  enough in the UTMOS metric environment.
- Current Rorqual config points Whisper at repo-root `.venv-eval-whisper`.
  If the new cluster stores it under `.venvs/metrics/`, update
  `configs/eval_metrics.json`.

## 1. OpenAI Whisper Large-v3

Supports:

- `asr_wer`
- `dwer`

Install:

```bash
python -m venv .venv-eval-whisper
source .venv-eval-whisper/bin/activate
python -m pip install --upgrade pip
python -m pip install -U openai-whisper
```

Assets to download:

```bash
mkdir -p /scratch/lovenya/SpeechAudioLM/models/metrics/whisper
python - <<'PY'
import whisper
model = whisper.load_model(
    "large-v3",
    download_root="/scratch/lovenya/SpeechAudioLM/models/metrics/whisper",
)
print("loaded", model.dims)
PY
```

Cache directory:

```text
/scratch/lovenya/SpeechAudioLM/models/metrics/whisper
```

Smoke test:

```bash
source .venv-eval-whisper/bin/activate
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
mkdir -p /scratch/lovenya/SpeechAudioLM/cache/metrics/aac_metrics
export AAC_METRICS_CACHE=/scratch/lovenya/SpeechAudioLM/cache/metrics/aac_metrics
export XDG_CACHE_HOME=/scratch/lovenya/SpeechAudioLM/cache/metrics/aac_metrics
export TORCH_HOME=/scratch/lovenya/SpeechAudioLM/cache/metrics/torch
export SENTENCE_TRANSFORMERS_HOME=/scratch/lovenya/SpeechAudioLM/cache/metrics/sentence_transformers
aac-metrics-download --cache_path "$AAC_METRICS_CACHE"
aac-metrics-info
```

Cache directory:

```text
/scratch/lovenya/SpeechAudioLM/cache/metrics/aac_metrics
```

Smoke test:

```bash
module load java/11.0.22
source .venvs/metrics/.venv-eval-aac/bin/activate
export AAC_METRICS_CACHE=/scratch/lovenya/SpeechAudioLM/cache/metrics/aac_metrics
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
- `mace` and `aac_clapsim` use MS-CLAP and currently set `batch_size: 1` in
  `configs/eval_metrics.json`; larger batches failed because the backend
  collator stacked variable-length GPT-2 token tensors.
- On Rorqual, AAC metric HF cache variables must point at:
  `/scratch/lovenya/SpeechAudioLM/cache/metrics/aac_metrics/huggingface`.
  The generic metric HF cache did not satisfy offline MS-CLAP loading.
- The current evaluator should fail only when the selected metric needs this
  backend.

Node:

- Install/download: internet-capable node.
- Tiny SPIDEr smoke: login node is acceptable after assets exist.
- MACE/CLAP-backed scoring over many samples: compute node.

Primary references:

- https://github.com/Labbeti/aac-metrics
- https://github.com/satvik-dixit/mace

## 3. stable-audio-metrics

Supports:

- `clap_score_t2a`
- `kl_passt`

Decision status:

- Selected for T2A/S2A-final text-to-audio alignment and KL_PaSST.
- Do not use this backend for A2T AAC CLAPSim; that remains under
  `aac-metrics` as `aac_clapsim`.

Install/current setup:

```bash
/path/to/python -m venv /scratch/lovenya/SpeechAudioLM/.venvs/metrics/.venv-eval-stable-audio-metrics
source /scratch/lovenya/SpeechAudioLM/.venvs/metrics/.venv-eval-stable-audio-metrics/bin/activate
python -m pip install --upgrade pip
python -m pip install requests tqdm numpy scipy librosa soundfile pyloudnorm ftfy regex
python -m pip install timm==0.4.12 --no-deps
python -m pip install laion-clap==1.1.4 --no-deps
python -m pip install git+https://github.com/kkoutini/passt_hear21@3d4e7cf10cd992d15e8e2da4db97c1766e79a93b
```

The upstream `Stability-AI/stable-audio-metrics` repository is not a normal pip
project in the revision used here. Clone/copy it as a source tree and expose
its `src` directory with `PYTHONPATH`.

Assets to download:

```bash
mkdir -p /scratch/lovenya/SpeechAudioLM/cache/metrics/stable_audio_metrics
export XDG_CACHE_HOME=/scratch/lovenya/SpeechAudioLM/cache/metrics/stable_audio_metrics
export TORCH_HOME=/scratch/lovenya/SpeechAudioLM/cache/metrics/torch
export HF_HOME=/scratch/lovenya/SpeechAudioLM/cache/metrics/huggingface
python - <<'PY'
import clap_score
import passt_kld
print(clap_score.__file__)
print(passt_kld.__file__)
PY
```

Cache directory:

```text
/scratch/lovenya/SpeechAudioLM/cache/metrics/stable_audio_metrics
```

Smoke test:

```bash
source .venvs/metrics/.venv-eval-stable-audio-metrics/bin/activate
python scripts/eval_backend_smoke/smoke_clap_backend.py
```

Expected output shape:

- Import smoke should confirm the package imports.
- Through `evaluate.py`, `clap_score_t2a,kl_passt` passed a two-sample
  Audio-Omni T2A smoke after LAION-CLAP/RoBERTa cache setup.

Known caveats:

- `clap_score_t2a` uses LAION-CLAP. Prefer checkpoint
  `630k-audioset-fusion-best` if available through the package.
- `kl_passt` uses a PaSST classifier.
- Stable Audio Metrics documentation indicates GPU support is expected for
  metric execution; use compute nodes for real scoring.
- Record LAION-CLAP checkpoint, PaSST config, package/git version, cache path,
  and backend sample-rate/resampling policy in `run_config`.

Node:

- Import-only: login node is acceptable.
- Model load/scoring: compute node.

Primary references:

- https://github.com/LAION-AI/CLAP

TODO:

- Keep source path, LAION-CLAP checkpoint, RoBERTa/HF cache root, and PaSST
  checkpoint path explicit in `configs/eval_metrics.json` on any new cluster.

## 4. UTMOSv2

Supports:

- `utmos`

Install:

```bash
python -m venv .venvs/metrics/.venv-eval-utmosv2
source .venvs/metrics/.venv-eval-utmosv2/bin/activate
python -m pip install --upgrade pip
python -m pip install git+https://github.com/sarulab-speech/UTMOSv2
```

Assets to download:

```bash
mkdir -p /scratch/lovenya/SpeechAudioLM/cache/metrics/utmosv2
export XDG_CACHE_HOME=/scratch/lovenya/SpeechAudioLM/cache/metrics/utmosv2
```

Cache directory:

```text
/scratch/lovenya/SpeechAudioLM/cache/metrics/utmosv2
```

Smoke test:

```bash
source .venvs/metrics/.venv-eval-utmosv2/bin/activate
python scripts/eval_backend_smoke/smoke_utmos.py
python scripts/eval_backend_smoke/smoke_utmos.py --load-model
```

Expected output shape:

- Import-only smoke: JSON with package/import status.
- Model smoke: JSON with one MOS-like score per test waveform once UTMOSv2
  invocation is wired.

Known caveats:

- This project uses UTMOSv2 from `sarulab-speech/UTMOSv2`, not the older
  SpeechMOS/UTMOS torch.hub backend.
- UTMOS is for generated speech quality, not environmental audio quality.
- The UTMOSv2 checkpoint at
  `/scratch/lovenya/SpeechAudioLM/cache/metrics/utmosv2/models/fusion_stage3/fold0_s42_best_model.pth`
  was once corrupted and had to be redownloaded. Current verified size:
  `818531314` bytes.
- UTMOSv2 also loads `facebook/wav2vec2-base`; run compute-node jobs with
  offline HF flags after caching this model. `configs/eval_metrics.json` sets
  `hf_hub_offline` and `transformers_offline` for `utmos`.
- Record UTMOSv2 package/git commit, checkpoint/cache path, sample rate, and
  preprocessing policy.

Node:

- Import-only: login node.
- Model load/full scoring: compute node.

Primary references:

- https://github.com/sarulab-speech/UTMOSv2

## 5. DNSMOS

Supports:

- `dnsmos`

Decision status:

- Backend direction selected but not wired into `evaluate.py` yet. DNSMOS is
  allowed for generated-speech tasks in `configs/task_metric_map.json`, but
  `configs/eval_metrics.json` still has `backend: null`, and the evaluator
  worker dispatch does not yet include a DNSMOS backend.
- Current TamIA inspection found no prepared DNSMOS model directory under
  `/scratch/l/lovenya/salm_migration_clean/models/metrics/dnsmos`, and the
  checked metric envs did not have `torchmetrics` installed.
- TorchMetrics DNSMOS uses
  `deep_noise_suppression_mean_opinion_score(preds, fs, personalized, ...)`
  and returns four values per sample:
  `[p808_mos, mos_sig, mos_bak, mos_ovr]`.

Install:

```bash
python -m venv .venvs/metrics/.venv-eval-dnsmos
source .venvs/metrics/.venv-eval-dnsmos/bin/activate
python -m pip install --upgrade pip
python -m pip install torchmetrics onnxruntime librosa
```

Assets to download:

```bash
mkdir -p /scratch/lovenya/SpeechAudioLM/models/metrics/dnsmos/DNSMOS
mkdir -p /scratch/lovenya/SpeechAudioLM/models/metrics/dnsmos/pDNSMOS
wget -O /scratch/lovenya/SpeechAudioLM/models/metrics/dnsmos/DNSMOS/model_v8.onnx \
  https://raw.githubusercontent.com/microsoft/DNS-Challenge/master/DNSMOS/DNSMOS/model_v8.onnx
wget -O /scratch/lovenya/SpeechAudioLM/models/metrics/dnsmos/DNSMOS/sig_bak_ovr.onnx \
  https://raw.githubusercontent.com/microsoft/DNS-Challenge/master/DNSMOS/DNSMOS/sig_bak_ovr.onnx
wget -O /scratch/lovenya/SpeechAudioLM/models/metrics/dnsmos/pDNSMOS/sig_bak_ovr.onnx \
  https://raw.githubusercontent.com/microsoft/DNS-Challenge/master/DNSMOS/pDNSMOS/sig_bak_ovr.onnx
```

Cache directory:

```text
/scratch/lovenya/SpeechAudioLM/models/metrics/dnsmos
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

Adapter requirements:

- Add a `torchmetrics-dnsmos` backend entry to `configs/eval_metrics.json`.
- Add DNSMOS to the external metric dispatch in `evaluation/runner.py`.
- Add a `torchmetrics-dnsmos` worker branch in `scripts/eval_backend_worker.py`.
- Point TorchMetrics at the prepared project DNSMOS asset directory before
  calling the functional API, so no compute-node download is attempted.
- Score each generated speech output file and return per-sample `dnsmos`
  values plus sub-scores and backend metadata.
- After the import/asset smoke passes, verify through
  `evaluate.py --metrics dnsmos` on a one-sample generated-speech manifest.

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
mkdir -p /scratch/lovenya/SpeechAudioLM/cache/metrics/audioldm_eval
export XDG_CACHE_HOME=/scratch/lovenya/SpeechAudioLM/cache/metrics
export TORCH_HOME=/scratch/lovenya/SpeechAudioLM/cache/metrics/torch
python - <<'PY'
from audioldm_eval import EvaluationHelper
import torch
helper = EvaluationHelper(16000, torch.device("cpu"))
print(type(helper).__name__)
PY
```

The VGGish torch hub files used by AudioLDM Eval should be present at:

```text
/scratch/lovenya/SpeechAudioLM/cache/metrics/torch/hub/checkpoints/vggish-10086976.pth
/scratch/lovenya/SpeechAudioLM/cache/metrics/torch/hub/checkpoints/vggish_pca_params-970ea276.pth
```

Cache directory:

```text
/scratch/lovenya/SpeechAudioLM/cache/metrics/audioldm_eval
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
  backend mode, but should include corpus-level FAD and Inception Score when
  available.

Known caveats:

- The normal `pip install git+https://github.com/haoheliu/audioldm_eval`
  dependency path can pull PyPI `wave`, which pulls Python 2-era
  `MySQL-python` and fails on Python 3 with `No module named 'ConfigParser'`.
  Use the `--no-deps` install above and install Python 3-compatible
  dependencies manually.
- `vggish_pca_params-970ea276.pth` may require verification with
  `torch.load(..., weights_only=False)` in newer PyTorch because it contains
  older NumPy pickled objects. Only do this for the trusted official torchvggish
  release asset.
- FAD and Inception Score are corpus-level only. Do not write fake per-sample
  values. KL_PaSST is handled by `stable-audio-metrics` as `kl_passt`.
- The toolkit accepts reference/generated folders; paired vs unpaired behavior
  depends on filenames and counts.
- Resampling can affect FAD/IS; record sample rate and preprocessing policy.
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
mkdir -p /scratch/lovenya/SpeechAudioLM/cache/metrics/kadtk
export XDG_CACHE_HOME=/scratch/lovenya/SpeechAudioLM/cache/metrics
export TORCH_HOME=/scratch/lovenya/SpeechAudioLM/cache/metrics/torch
kadtk --help
```

Cache directory:

```text
/scratch/lovenya/SpeechAudioLM/cache/metrics/kadtk
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
- On Rorqual, `kadtk` import first failed because its package imports
  `from turtle import st`, which imports `tkinter`; the selected Python module
  lacked tkinter. A small site-level workaround/stub was used in the env. On a
  new cluster, first check whether Python includes tkinter before applying any
  workaround.
- `kad` passed a two-sample T2A smoke through `evaluate.py`.
- KAD direction matters in practice: reference set should be real/ground-truth
  audio and target set should be generated audio.
- Use `--force-emb-encode` only when intentionally ignoring cached embeddings.

Node:

- Import-only/CLI help: login node.
- Corpus KAD run: compute node.

Primary references:

- https://github.com/YoonjinXD/kadtk

## Immediate TODOs

- Retry `aac_clapsim` and `mace` through `evaluate.py` after the AAC cache-root
  and batch-size fixes.
- Retry `fad,inception_score` through `evaluate.py` after the VGGish PCA
  checkpoint download.
- Defer DNSMOS backend selection.
- Add evaluator `--resume` and `--overwrite` behavior before long corpus runs.
- After each backend smoke passes, record package versions, git commits,
  checkpoint paths, sample-rate policy, and cache dirs in `configs/eval_metrics.json`.
