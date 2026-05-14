# SpeechAudio LM

SpeechAudio LM is a research project for bidirectional generation between environmental audio and speech. The current work is focused on building a clean, reproducible baseline before larger experiments.

## Current Focus

- Dataset cleaning and metadata validation.
- Fixed train/validation/test split.
- AudioCraft/AudioGen setup.
- Baseline inference and evaluation.

## Data

The dataset is stored outside this git repository:

```text
/project/rrg-csubakan/lovenya/datasets/ttas/extracted
```

Current best clean manifest:

```text
/project/rrg-csubakan/lovenya/datasets/ttas/extracted/ttas/22k_samples_clean_manual_meta_clean.jsonl
```

Small metadata artifacts are versioned in this repository:

- `data/manifests/ttas_clean_normalized.jsonl`: normalized canonical manifest with relative `audio_file` and `speech_file` paths.
- `data/splits/ttas_train.jsonl`, `data/splits/ttas_val.jsonl`, `data/splits/ttas_test.jsonl`: fixed `80/10/10` split using seed `1337`.
- `data/reports/ttas_dataset_report.json`: metadata and path validation summary.
- `configs/ttas_dataset.json`: dataset root, manifest, split paths, seed, and stratification settings.

Relative audio and speech paths should be resolved against the configured dataset root, for example:

```text
/project/rrg-csubakan/lovenya/datasets/ttas/extracted/audio/11nhZxx4vr4.wav
/project/rrg-csubakan/lovenya/datasets/ttas/extracted/speech/11nhZxx4vr4.wav
```

Regenerate the metadata report and print a screenshot-friendly CLI summary:

```bash
python utils/generate_metadata_report.py --print --other-labels-limit 8
```

Add `--show-delta` to include actual-minus-expected split-count deltas.

Do not commit audio files, zips, checkpoints, outputs, wandb runs, secrets, or other large generated artifacts.

## AudioCraft / AudioGen Setup

The current AudioCraft environment is:

```bash
module load StdEnv/2023 gcc cuda rust arrow python/3.10.13
source .venv-audiocraft/bin/activate
```

Key package versions currently verified:

```text
torch==2.6.0+computecanada
torchaudio==2.6.0+computecanada
numpy==1.26.4
audiocraft==1.1.0
```

AudioGen weights are stored outside this repository:

```text
/project/rrg-csubakan/lovenya/models/audiogen-medium
```

The local model directory contains `state_dict.bin` and `compression_state_dict.bin`; pass this directory to AudioCraft for offline inference instead of relying on internet access from compute nodes.

AudioGen-medium also requires the separately cached `t5-large` text encoder/tokenizer for prompt conditioning. Cache Hugging Face assets under:

```text
/project/rrg-csubakan/lovenya/models/hf_cache
```

Current setup note: the Torch 2.1 environment is not compatible with Rorqual H100/MIG GPUs because it lacks `sm_90` support. Use the Torch 2.6 environment for H100 inference.

AudioGen has been smoke-tested on a Rorqual H100 20 GB MIG allocation. The model loaded offline from local AudioGen and `t5-large` caches, generated a 10 second mono waveform at 16 kHz, and used under 6 GB reserved VRAM during the smoke run.

## Baseline Model Assets

Additional baseline model assets are stored outside this repository:

```text
/project/rrg-csubakan/lovenya/models/tangoflux
/project/rrg-csubakan/lovenya/models/qwen3-omni-30b-a3b-instruct
/project/rrg-csubakan/lovenya/models/audio-omni
/project/rrg-csubakan/lovenya/models/qwen2.5-omni-3b
```

Current baseline environments:

```text
.venv-tangoflux
.venv-qwen3-omni
.venv-audio-omni
```

These are setup environments only; generated outputs and checkpoints stay outside git.

TangoFlux has been smoke-tested on a Rorqual H100 20 GB MIG allocation. It loaded local model assets plus the cached `google/flan-t5-large` text encoder offline and generated a 5 second 44.1 kHz stereo waveform.

Qwen3-Omni-30B-A3B-Instruct has been smoke-tested for voice output on a full H100 allocation, using local model assets and a venv-local Transformers talker workaround. Audio-Omni has also been smoke-tested with local Qwen2.5-Omni-3B conditioning.

## Baseline Inference CLI

Use `baseline.py` to run a small number of samples from a fixed split. The router loads one model backend and writes generated audio plus a JSONL manifest under the output directory.

Example smoke commands:

```bash
python baseline.py --model audiogen --split test --num-samples 2 --prompt-field speech_text
python baseline.py --model tangoflux --split test --num-samples 2 --prompt-field speech_text
python baseline.py --model qwen3_omni --split test --num-samples 2 --prompt-field speech_text
python baseline.py --model audio_omni --split test --num-samples 2 --prompt-field speech_text --duration 10
```

Use `--dry-run` to inspect selected rows and output paths without loading a model. Use `--num-samples -1` for a full split after the model-specific smoke run is stable. Use `--prompt-field audio_cap` to run caption-conditioned baselines.

Default outputs are organized under:

```text
results/baselines/<model>/<split>/<prompt_field>/
```

Run logs and resource samples are organized similarly under:

```text
logs/baselines/<model>/<split>/<prompt_field>/
```

The `results/` and `logs/` directories are ignored by git.

## Repository Status

This repo is in an early setup phase. The current reproducible dataset state is defined by the normalized manifest, fixed split files, and dataset config.
