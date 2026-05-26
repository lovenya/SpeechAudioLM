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

Use `baseline.py` to run a small number of samples from a fixed split. The router loads one model backend and writes generated outputs plus a JSONL manifest under the output directory. Use `--task` to choose the inference path.

Example smoke commands:

```bash
python baseline.py --model audiogen --task T2A --split test --num-samples 2
python baseline.py --model tangoflux --task T2A --split test --num-samples 2
python baseline.py --model audio_omni --task T2A --split test --num-samples 2 --duration 10
python baseline.py --model audio_omni --task T2S --split test --num-samples 2 --duration 10
python baseline.py --model audio_omni --task A2T --split test --num-samples 2 --duration 10
python baseline.py --model audio_omni --task S2T --split test --num-samples 2 --duration 10
python baseline.py --model audio_omni --task S2A --split test --num-samples 2 --duration 10
python baseline.py --model audio_omni --task A2S --split test --num-samples 2 --duration 10
python baseline.py --model qwen3_omni --task A2S --split test --num-samples 2
```

Use `--dry-run` to inspect selected rows and output paths without loading a model. Use `--num-samples -1` for a full split after the model-specific smoke run is stable. If `--prompt-field` is omitted, `T2A`, `A2T`, and `A2S` default to `audio_cap`; `T2S`, `S2T`, `S2A`, and Qwen3-Omni `A2S` default to `speech_text`. Use `--prompt-template` for one-off prompt experiments.

AudioGen and TangoFlux support native `T2A` only and receive raw captions. Audio-Omni supports native `T2A`/`T2S`, understanding-only `A2T`/`S2T`, and cascaded `S2A`/`A2S`. Qwen3-Omni supports `T2S` and an `A2S` baseline that skips audio understanding and uses the paired `speech_text` as the intermediate transcript before TTS. Any Qwen3-Omni pseudo-T2A paths should be treated as legacy exploratory probes, not active project baselines.

Default baseline outputs are organized under:

```text
results/baselines/<model>/<task>/<dataset>/<split>/
```

Run logs and resource samples are organized similarly under:

```text
logs/baselines/<model>/<task>/<dataset>/<split>/
```

The `results/` and `logs/` directories are ignored by git.

## Evaluation CLI

Use `evaluate.py` to score a baseline manifest. The evaluator resolves the
canonical manifest path from `--model`, `--task`, `--dataset`, and `--split`;
`--manifest` can override this for older or custom runs. Task metric validity is
loaded from `configs/task_metric_map.json`.

Example:

```bash
python evaluate.py --model audio_omni --task S2T --metrics wer --max-samples 2
python evaluate.py --model audio_omni --task T2A --metrics fad,kad,inception_score --dry-run
python evaluate.py --model audio_omni --task T2S --metrics asr_wer,dwer,utmos --dry-run
```

Evaluation outputs are organized under:

```text
results/evals/<task>/<dataset>/<model>/
```

Each run writes timestamped `per_sample`, `summary`, `summary.csv`, `report`,
and `run_config` files plus `latest_*` aliases. Heavy metrics are configurable
through `configs/eval_metrics.json`; missing external packages or model weights
fail only when the corresponding metric is selected.

Current metric backend status:

- Smoke-passed A2T metric: `spider` through `aac-metrics` with Java 11.
- A2T metrics needing retry after cache/batch fixes: `mace`, `aac_clapsim`.
- Smoke-passed T2A/S2A-final metrics: `kad`, `clap_score_t2a`, and
  `kl_passt`.
- T2A/S2A-final metrics needing retry after VGGish PCA download:
  `fad`, `inception_score` through AudioLDM Eval.
- Smoke-passed T2S and A2S-final metrics: `asr_wer`, `dwer`, `utmos` through
  Whisper large-v3 and UTMOSv2.
- Available S2T metric: internal `wer`.

Migration handoff docs for moving this work to a new cluster are maintained
locally in `CLUSTER_MIGRATION_HANDOFF.md` and `NEW_CLUSTER_ENV_SETUP.md`.

Metric envs are intentionally split because dependency stacks conflict:

```text
.venvs/.venv-eval-aac
.venv-eval-whisper
.venvs/metrics/.venv-eval-audioldm-eval
.venvs/metrics/.venv-eval-kadtk
.venvs/metrics/.venv-eval-utmosv2
/scratch/lovenya/SpeechAudioLM/.venvs/metrics/.venv-eval-stable-audio-metrics
```

Metric caches/checkpoints should live on scratch:

```text
/scratch/lovenya/SpeechAudioLM/cache/metrics
/scratch/lovenya/SpeechAudioLM/models/metrics
```

Metric notes:

- `--metrics` is optional; defaults are selected from the task mapping.
- S2A final-output evaluation uses audio-generation metrics, not WER. WER is
  evaluated by running `--task S2T` on the S2T component manifest.
- A2S final-output evaluation uses generated-speech metrics. Caption metrics
  are evaluated by running `--task A2T` on the A2T component manifest.
- Generated-speech `asr_wer` is `WER(reference_transcript, ASR(generated_speech))`.
- Generated-speech `dwer` is directional:
  `WER(ASR(reference_speech), ASR(generated_speech))`.
- For text-to-audio, FAD, KAD, KL_PaSST, and Inception Score are corpus-level
  only. T2A CLAPscore is computed per sample and aggregated when the
  stable-audio-metrics env is available.
- A2T CLAPSim and T2A CLAPscore use different metric names and backends:
  `aac_clapsim` and `clap_score_t2a`.
- FAD is the only default Frechet-style audio metric for v1; generic FD is not
  included unless a later paper-specific comparison requires it.

## Submission Scripts

Compute-node helpers live in `submission_scripts/`.

Interactive metric smoke allocation:

```bash
submission_scripts/interactive_eval_smoke.sh
```

Inside the allocation, validate metric adapters against existing two-sample
baseline outputs:

```bash
submission_scripts/run_eval_smoke_existing_manifests.sh
```

Full baseline submissions:

```bash
sbatch submission_scripts/run_baseline_audiogen_t2a.sbatch
sbatch submission_scripts/run_baseline_tangoflux_t2a.sbatch
sbatch submission_scripts/run_baseline_audio_omni_array.sbatch
sbatch submission_scripts/run_baseline_qwen3_omni_a2s.sbatch
```

Available metric evaluations over completed baseline manifests:

```bash
sbatch submission_scripts/run_eval_available_metrics_array.sbatch
```

Use `NUM_SAMPLES=2 sbatch ...` for a compute-node baseline smoke before full
runs. Use `MAX_SAMPLES=2 sbatch submission_scripts/run_eval_available_metrics_array.sbatch`
for a compute-node evaluation smoke before full metrics. Scratch outputs are
written under `/scratch/lovenya/SpeechAudioLM/results/`.

Slurm resource requests are tracked in
`configs/slurm_resource_profiles.json`. The submission scripts start
`scripts/resource_monitor.py` by default and write node-level CPU, RAM, and GPU
samples plus peak summaries under `/scratch/lovenya/SpeechAudioLM/logs/resources/`.
Use `RESOURCE_INTERVAL=300` for five-minute samples, or `RESOURCE_INTERVAL=0`
to disable monitoring.

Rebuild or refresh the optional stable-audio-metrics env and checkpoint caches
on an internet-capable login node:

```bash
RECREATE=1 submission_scripts/setup_stable_audio_metrics_scratch.sh
```

## Repository Status

This repo is in an early setup phase. The current reproducible dataset state is defined by the normalized manifest, fixed split files, and dataset config.
