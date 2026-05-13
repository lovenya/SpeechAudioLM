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

## Repository Status

This repo is in an early setup phase. The current reproducible dataset state is defined by the normalized manifest, fixed split files, and dataset config.
