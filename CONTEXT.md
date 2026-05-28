# Project Context

## Overview

SpeechAudio LM is a research repository for bidirectional generation between environmental audio and speech. The current phase is setup and baseline preparation, not large-scale model development.

## Current Focus

- Keep the TTAS dataset/splits fixed and reuse them for baseline evaluation.
- Maintain model/task baseline outputs under the canonical `results/baselines/` structure.
- Build out the evaluation scaffold and add external metric backends in controlled environments.
- Preserve metric reproducibility through config files, backend metadata, run configs, and explicit cache/model paths.

## TamIA Migration Status - 2026-05-27

The current TamIA working roots are:

```text
repo: /scratch/l/lovenya/SpeechAudioLM
clean migration root: /scratch/l/lovenya/salm_migration_clean
dataset: /scratch/l/lovenya/salm_migration_clean/datasets/ttas/extracted
models: /scratch/l/lovenya/salm_migration_clean/models
metric cache: /scratch/l/lovenya/salm_migration_clean/cache/metrics
```

Verified migration gates:

- TTAS dataset validates against `data/manifests/ttas_clean_normalized.jsonl`: `22780` rows, no missing audio, no missing speech.
- Metric envs exist under `.venvs/metrics` in the clean root and passed import/smoke checks for Whisper, AAC/SPIDEr, stable-audio-metrics, UTMOSv2, AudioLDM Eval, and KADTK.
- Active configs, model defaults, smoke scripts, and Slurm scripts now point at the clean TamIA roots.
- Unit tests, JSON validation, Python syntax checks, and login-node baseline dry-runs passed.

Compute-node model smoke gate now passes on TamIA. Retry job `309034` completed with `ExitCode 0:0` using one 4-H100 node, `--cpus-per-task=16`, and `--mem=192G`. AudioGen T2A, Qwen3-Omni A2S, Audio-Omni A2S, and Audio-Omni S2A all passed with successful generated manifests. The next migration gate is `evaluate.py` metric scoring smokes on the generated manifests. Use the verified default account `aip-csubakan` for any further Slurm submissions.

Metric-cache follow-up on 2026-05-28: `resampy` was installed for `audioldm_eval`; the KAD PANN checkpoint and stable-audio-metrics LAION-CLAP checkpoint are present under the clean metric cache; `facebook/wav2vec2-base` resolves offline for UTMOS only when `HF_HOME`, `HF_HUB_CACHE`, `HUGGINGFACE_HUB_CACHE`, and `TRANSFORMERS_CACHE` all point at `$SALM_SCRATCH/cache/metrics/huggingface`. The TamIA metric smoke scripts now export those variables.

## Data Locations

Dataset files live outside git:

```text
/project/rrg-csubakan/lovenya/datasets/ttas/extracted
```

Current best clean manifest:

```text
/project/rrg-csubakan/lovenya/datasets/ttas/extracted/ttas/22k_samples_clean_manual_meta_clean.jsonl
```

Do not copy audio, archives, checkpoints, outputs, wandb runs, or secrets into this repository.

## Current Dataset State

- Canonical repo manifest:
  `data/manifests/ttas_clean_normalized.jsonl`
- Dataset config:
  `configs/ttas_dataset.json`
- Fixed split files:
  `data/splits/ttas_train.jsonl`,
  `data/splits/ttas_val.jsonl`,
  `data/splits/ttas_test.jsonl`
- Metadata report:
  `data/reports/ttas_dataset_report.json`
- Metadata CLI:
  `python utils/generate_metadata_report.py --print --other-labels-limit 8`

The manifest and splits store relative `audio_file` and `speech_file` paths such as `audio/<id>.wav` and `speech/<id>.wav`. Downstream code should resolve them against the configured dataset root.

Split policy is fixed at `80/10/10` with global seed `1337`, stratified by `speaker_gender` plus bucketed `speaker_emotion`. Emotion labels with fewer than `20` examples are grouped into `other` for stratification, while raw labels remain available in reports.

## Working Assumptions

- The clean manifest is the preferred starting point until a better one is explicitly chosen.
- Dataset splits are deterministic and should be reused for downstream experiments unless a new split decision is explicitly made.
- Experiment decisions should record both what changed and why.

## Open Decisions

- Which concrete backend/checkpoint to use for each external metric before full evaluation.
- Whether metric backends should share environments or use separate venvs by toolkit.
- Whether xFormers/Triton or other accelerator libraries are worth installing after functional backend checks.
- How cascaded baseline runs should write component-specific manifests in addition to combined final-output manifests.

## HPC Environment Notes

- Load modules before activating project virtual environments.
- Standard baseline module stack:
  `module load StdEnv/2023 gcc cuda rust arrow python/3.10.13`
- Load `opencv` only when needed.
- Use `venv`/`virtualenv`; do not assume conda, anaconda, or uv.
- Rorqual/TamIA compute nodes do not have internet. FIR compute nodes have internet.

## AudioCraft / AudioGen Setup State

- Current Rorqual H100-compatible AudioCraft environment:
  `.venv-audiocraft`
- Activate with:
  `module load StdEnv/2023 gcc cuda rust arrow python/3.10.13`
  then
  `source .venv-audiocraft/bin/activate`
- Core versions in this env:
  `torch==2.6.0+computecanada`, `torchaudio==2.6.0+computecanada`, `numpy==1.26.4+computecanada`, `audiocraft==1.1.0`
- The `.venv-audiocraft-torch21-py310` env has `torch==2.1.1`, which fails on Rorqual H100/MIG with:
  `RuntimeError: CUDA error: no kernel image is available for execution on the device`
  because that build lacks `sm_90` support.
- `.venv-audiocraft` was verified on an H100 20 GB MIG allocation:
  `torch.ones(1, device="cuda")` succeeds.
- `from audiocraft.models import AudioGen` succeeds in `.venv-audiocraft`, with warnings that xFormers C++/CUDA extensions were built for a different Torch version and Triton is missing.
- AudioGen weights are downloaded outside git at:
  `/project/rrg-csubakan/lovenya/models/audiogen-medium`
- Downloaded files include:
  `state_dict.bin` around `3.5G` and `compression_state_dict.bin` around `225M`; total folder size is about `3.7G`.
- AudioCraft's loader supports passing this local model directory to `AudioGen.get_pretrained(...)`; it loads `state_dict.bin` and `compression_state_dict.bin` from the directory.
- AudioGen model load reaches the T5 text conditioner and requires separately cached `t5-large` assets:
  the checkpoint config has `conditioners.description.model=t5`, `name=t5-large`, `finetune=false`.
- `t5-large` is not bundled in `state_dict.bin`; AudioCraft calls `T5Tokenizer.from_pretrained("t5-large")` and `T5EncoderModel.from_pretrained("t5-large")` at model construction.
- `t5-large` is downloaded to `/project/rrg-csubakan/lovenya/models/hf_cache/transformers`; cache size is about `4.9G`.
- The `hf` CLI in `.venv-audiocraft` failed with a `typer.Argument` mismatch, so use the working `huggingface-cli` from `.venv-audiocraft-torch21-py310` or a Python `snapshot_download` for cache downloads.
- Offline cache variables required on no-internet compute nodes:
  `HF_HOME=/project/rrg-csubakan/lovenya/models/hf_cache`,
  `HF_HUB_CACHE=/project/rrg-csubakan/lovenya/models/hf_cache/transformers`,
  `TRANSFORMERS_CACHE=/project/rrg-csubakan/lovenya/models/hf_cache/transformers`,
  `HF_HUB_OFFLINE=1`,
  `TRANSFORMERS_OFFLINE=1`.
- AudioGen smoke test succeeded on a Rorqual H100 20 GB MIG allocation. It generated a 10 second mono waveform at 16 kHz from a weather/thunder prompt and saved:
  `/project/rrg-csubakan/lovenya/outputs/audiogen_smoke/smoke.wav`
- Smoke-test resource observations:
  model load took about `22.73s`; generation took about `13.76s`; CUDA reserved memory after generation was about `5.88G`.
- xFormers/Triton warnings remain optimization warnings for now. They did not block model load or generation.
- Next implementation step: run the baseline inference CLI on compute nodes for the full TTAS test split and inspect generated manifests/audio.

## Additional Baseline Setup State

Current baseline list:

- AudioGen / AudioCraft: working smoke test complete.
- TangoFlux: `declare-lab/TangoFlux`; working smoke test complete.
- Qwen Omni: `Qwen/Qwen3-Omni-30B-A3B-Instruct` instruct model; voice smoke test complete on a full H100.
- Audio-Omni: `HKUSTAudio/Audio-Omni`; working smoke test complete on a 40 GB MIG allocation.

Downloaded model paths:

- `/project/rrg-csubakan/lovenya/models/tangoflux` around `3.8G`
- `/project/rrg-csubakan/lovenya/models/qwen3-omni-30b-a3b-instruct` around `52G`
- `/project/rrg-csubakan/lovenya/models/audio-omni` around `20G`
- `/project/rrg-csubakan/lovenya/models/qwen2.5-omni-3b` around `12G`, required by Audio-Omni's internal Qwen2.5 conditioner

Environment paths:

- `.venv-tangoflux`
- `.venv-qwen3-omni`
- `.venv-audio-omni`

TangoFlux notes:

- Dependencies installed from cluster wheelhouse without pinning versions.
- Active versions include `torch==2.6.0+computecanada`, `torchaudio==2.6.0+computecanada`, `torchvision==0.21.0+computecanada`, `transformers==5.3.0+computecanada`, `diffusers==0.37.1+computecanada`, `accelerate==1.13.0+computecanada`, `librosa==0.11.0+computecanada`, `numpy==1.26.4+computecanada`.
- `datasets` is not installed. TangoFlux imports `datasets.load_dataset` and `datasets.Audio` at top level, but code search shows those names are not used in the text-to-audio inference path.
- `TangoFluxInference` imports successfully when a small runtime `datasets` stub is inserted before import.
- TangoFlux smoke test succeeded on a Rorqual H100 20 GB MIG allocation. It loaded `google/flan-t5-large` offline, generated a 5 second 44.1 kHz stereo waveform, saved `/project/rrg-csubakan/lovenya/outputs/tangoflux_smoke/smoke.wav`, and reserved about `10.71G` CUDA memory.
- The `lm_head.weight` unexpected key warning from `T5EncoderModel` is expected when loading `google/flan-t5-large` as an encoder-only text conditioner.

Qwen3-Omni notes:

- `.venv-qwen3-omni` has core wheelhouse dependencies installed with `torch==2.6.0+computecanada`, `transformers==5.3.0+computecanada`, `accelerate==1.13.0+computecanada`, `librosa==0.11.0+computecanada`, and `soundfile==0.13.1+computecanada`.
- Local `AutoConfig.from_pretrained(..., local_files_only=True, trust_remote_code=True)` works and returns `Qwen3OmniMoeConfig`.
- Local `AutoProcessor.from_pretrained(..., local_files_only=True, trust_remote_code=True)` works and returns `Qwen3OmniMoeProcessor`.
- `qwen-omni-utils` is installed in `.venv-audio-omni` for Audio-Omni cascade understanding.
- `Qwen3OmniMoeForConditionalGeneration.from_pretrained(...)` with the local instruct model path loaded successfully on a full H100. Load time was about `446.83s` in the first run and CUDA reserved memory was about `67.56G`.
- The `rope_parameters` warning about `mrope_section` and `interleaved` is non-fatal with the current Transformers build.
- A first text-only generation attempt used the wrong return handling. With `return_audio=False`, `model.generate(...)` returns only the thinker result; with `return_audio=True`, it returns `(text_ids, audio)`.
- Qwen3-Omni voice generation then hit a local Transformers `5.3.0` talker bug:
  `GenerationMixin.prepare_inputs_for_generation() got multiple values for argument 'next_sequence_length'`.
- A venv-local workaround was applied in:
  `.venv-qwen3-omni/lib/python3.10/site-packages/transformers/models/qwen3_omni_moe/modeling_qwen3_omni_moe.py`
  by changing the talker `super().prepare_inputs_for_generation(...)` call to keyword arguments.
- Deferred TODO: verify this against upstream Transformers and consider a PR if still unfixed.
- After the workaround, Qwen3-Omni voice smoke test succeeded. It generated text:
  `Qwen-Omni voice generation is now working.`
  and saved audio to:
  `/project/rrg-csubakan/lovenya/outputs/qwen3_omni_smoke/smoke_voice.wav`.
- Output audio shape was `(1, 1, 68565)`, about `2.86s` at 24 kHz, with about `67.18G` CUDA memory reserved.

Audio-Omni notes:

- Model files are downloaded under `/project/rrg-csubakan/lovenya/models/audio-omni`.
- Source repo cloned outside git at `/project/rrg-csubakan/lovenya/src/Audio-Omni`.
- `.venv-audio-omni` has core wheelhouse dependencies installed, including `accelerate==1.13.0+computecanada`.
- `from audio_omni import AudioOmni` succeeds with:
  `PYTHONPATH=/project/rrg-csubakan/lovenya/src/Audio-Omni:$PYTHONPATH`.
- Editable install failed from Codex due sandbox read-only behavior in the external source directory; use `PYTHONPATH` for now or install editable from a normal shell if needed.
- OpenCV is available as a module but not needed for the core import path observed so far.
- Additional dependency fixes already applied in `.venv-audio-omni` include `k-diffusion`, `jieba`, `pypinyin`, `x-transformers`, `einops-exts`, `alias-free-torch==0.0.6`, and the Descript Audio Codec package that provides `dac.nn`.
- Audio-Omni smoke test reached model construction but failed when it attempted to load `Qwen/Qwen2.5-Omni-3B` for its internal conditioner. That model has now been downloaded to `/project/rrg-csubakan/lovenya/models/qwen2.5-omni-3b`.
- Audio-Omni needs:
  `QWEN_OMNI_MODEL_PATH=/project/rrg-csubakan/lovenya/models/qwen2.5-omni-3b`
  and currently uses `SKIP_QWEN_WEIGHTS=1`.
- Audio-Omni's source code expects relative assets under `/project/rrg-csubakan/lovenya/src/Audio-Omni`; the baseline backend now patches the released config with an absolute vocab path so the CLI can run from the repo root.
- `flash_attn` is not installed. This is currently acceptable because Audio-Omni detects it missing and disables Flash Attention rather than failing.
- Audio-Omni smoke test succeeded with `seconds_total=10`, generated shape `(2, 483328)`, generation time about `2.65s`, and reserved about `24.9G` CUDA memory. Output:
  `/project/rrg-csubakan/lovenya/outputs/audio_omni_smoke/smoke.wav`.
- The earlier `seconds_total=5` failure was a fixed-shape conditioning mismatch; current Audio-Omni baseline enforces `--duration 10`.
- Audio-Omni `S2A`/`A2S` cascades need `qwen-omni-utils` because they call the Qwen2.5 understanding path. This dependency is listed in Audio-Omni's `requirements.txt` but not in its `setup.py`, so it was missed in the minimal install.
- After installing `qwen-omni-utils`, the cascade understanding path hit a Transformers/Qwen2.5 compatibility issue: upstream Audio-Omni calls `generate(..., return_audio=False)`, but the installed Transformers path still checked the default `Chelsie` speaker even though only the thinker is loaded. The repo runner now uses a local helper that calls the loaded Qwen2.5 thinker directly for cascade understanding.

## Baseline Inference CLI

- Root CLI: `baseline.py`.
- Shared loader: `baselines/dataset.py`.
- Shared run/output writer: `baselines/run.py`.
- Backends:
  `audiogen`,
  `tangoflux`,
  `qwen3_omni`,
  `audio_omni`.
- Task-aware CLI support was added in commit `1c90785 [feat]: add baseline task modes`.
- Supported task paths:
  - AudioGen: `T2A`
  - TangoFlux: `T2A`
  - Audio-Omni: `T2A`, `T2S`, understanding-only `A2T`/`S2T`, cascaded `S2A`, cascaded `A2S`
  - Qwen3-Omni: `T2S`, `pseudo_t2a_direct`, `pseudo_t2a_phoneme`
- Default prompt fields:
  - `T2A`, `A2T`, `A2S`, `pseudo_t2a_direct`, `pseudo_t2a_phoneme`: `audio_cap`
  - `T2S`, `S2T`, `S2A`: `speech_text`
- Default output root is repo-local and git-ignored:
  `results/baselines/<model>/<task>/<dataset>/<split>/`.
- Each run writes generated WAV files under `audio/` and appends metadata to `manifest.jsonl`.
- Logs are repo-local and git-ignored:
  `logs/baselines/<model>/<task>/<split>/<prompt_field>/`.
- Each run writes a human-readable `run_<timestamp>.log` and a `resources_<timestamp>.jsonl`.
- Resource sampling uses `psutil` when available for process/system CPU, RSS memory, and thread count. It also records CUDA allocated/reserved memory when Torch CUDA is available.
- `--num-samples` controls sample count, `--num-samples -1` means full split, and `--dry-run` validates selected rows/output paths without loading models.
- Progress output shows `[current/total]`, elapsed time, and ETA.
- Dry-runs for all committed task modes over `data/splits/ttas_test.jsonl` succeeded from the repo root.
- Five-sample smoke tests for Audio-Omni native `T2A`/`T2S` succeeded. Audio-Omni `A2S` completed 5/5 samples after the thinker-direct cascade patch. `S2A` result is still pending in the current handoff.

## Evaluation Scaffold

- Root CLI: `evaluate.py`.
- Task metric validity is config-driven through `configs/task_metric_map.json`.
- Backend/tool configuration is held in `configs/eval_metrics.json`.
- Evaluation outputs are repo-local and git-ignored:
  `results/evals/<task>/<dataset>/<model>/`.
- Each run writes timestamped `per_sample`, `summary`, `summary.csv`, `report`, and `run_config` files plus `latest_*` aliases.
- `per_sample` keeps actual generated outputs under `predictions` and metric values under `metrics`.
- `summary` includes row counts, successful rows, skipped/error rows, metric aggregate stats where meaningful, and errors grouped by metric.
- `run_config` records command, device, cache path, dataset/metric config paths, task metric map path, backend metadata, and audio preprocessing policy placeholders.
- `--metrics` is optional; defaults come from the task map.
- `--dry-run` validates manifest requirements without computing metrics.
- `--max-samples` limits rows for scaffold checks.
- There is no current `--stage` flag. Intermediate cascade outputs should be evaluated by component manifests directly, such as `--task S2T` for transcript WER or `--task A2T` for caption metrics.
- `S2A + wer`, `A2S + caption metrics`, and `T2A + asr_wer` are rejected as conceptually invalid task/metric pairs.

## Metric Backend Agenda

- Do not install or download heavy metric assets on login nodes.
- Use separate venvs when backend dependencies conflict or are heavy enough to isolate.
- Metric backend venvs belong under `.venvs/metrics/`; future model/module envs
  belong under `.venvs/models/` or another model-specific subfolder.
- Do not move venv directories in place. Recreate them at the intended path.
- Current backend choices remain configurable:
  `aac-metrics` for MACE/SPIDEr/AAC CLAPSim, `stable-audio-metrics` for
  T2A CLAPscore and KL_PaSST, `audioldm_eval` for FAD/Inception Score, `kadtk`
  for KAD, UTMOSv2 for UTMOS, Whisper large-v3 for ASR, and a configurable text
  embedding model for semantic equivalence.
- T2T, A2S, non-final S2A, and DNSMOS backend decisions are deferred.
- External backend adapters should fail only when the selected metric needs that backend, with clear package/checkpoint/config messages.
- Whisper `large-v3` is downloaded and load-tested at:
  `/project/rrg-csubakan/lovenya/models/metrics/whisper`.
- `aac-metrics` assets are downloaded into:
  `/project/rrg-csubakan/lovenya/cache/metrics/aac_metrics`.
  Load `java/11.0.22` before SPIDEr/MACE smoke/evaluation.
  Java 17 is rejected by `aac-metrics` tokenization checks.
  SPIDEr smoke now passes with keys `cider_d`, `spice`, and `spider`.
- `audioldm_eval` imports successfully in its metric env after avoiding the
  broken PyPI `wave` dependency path with the documented `--no-deps` setup.

## Current Eval Smoke Status

Recent compute-node smoke checks against existing Audio-Omni baseline manifests
write outputs under:

```text
/scratch/lovenya/SpeechAudioLM/results/eval_smoke_existing
/scratch/lovenya/SpeechAudioLM/results/eval_smoke
```

Passed through `evaluate.py`:

- `asr_wer`, `dwer` with Whisper large-v3 on `audio_omni` `T2S`.
- `utmos` with UTMOSv2 on `audio_omni` `T2S`.
- Combined `asr_wer,dwer,utmos` on 5 `audio_omni` `T2S` samples after forcing
  UTMOS Hugging Face loads offline.
- `spider` with `aac-metrics` on an `audio_omni` A2T-style manifest; two-sample
  smoke completed in about `15s` with Java 11.
- `kad` with `kadtk` on 2 `audio_omni` `T2A` samples; runtime was about `2m`.
- `clap_score_t2a,kl_passt` with stable-audio-metrics on 2 `audio_omni` `T2A`
  samples; runtime was about `53s`.

Fixes committed after the first eval-smoke report:

- `bdb2065 fix: configure metric cache roots` makes AAC metrics use the
  AAC-specific Hugging Face cache.
- `4241051 fix: set aac clapsim batch size` sets `batch_size: 1` for
  MS-CLAP-backed AAC metrics to avoid variable-length token stack failures.
- `fc7d8b3 fix: run utmos hf models offline` sets backend-specific offline HF
  flags for UTMOS.

Checkpoint/cache note:

- The first UTMOSv2 smoke hung because
  `/scratch/lovenya/SpeechAudioLM/cache/metrics/utmosv2/models/fusion_stage3/fold0_s42_best_model.pth`
  was corrupted. It was redownloaded from `sarulab-speech/UTMOSv2`, now has size
  `818531314` bytes, and `torch.load(..., map_location="cpu")` succeeds.
- AudioLDM Eval needed VGGish torch hub assets. The following files now exist:
  `vggish-10086976.pth` and `vggish_pca_params-970ea276.pth` under
  `/scratch/lovenya/SpeechAudioLM/cache/metrics/torch/hub/checkpoints`.

Still to smoke through `evaluate.py`:

- Retry `aac_clapsim` and `mace` after the cache-root and batch-size fixes.
- Retry `fad`, `inception_score` via `audioldm_eval` after the VGGish PCA
  checkpoint download.

## Cluster Migration

Migration handoff docs were added for moving this project to another cluster:

```text
CLUSTER_MIGRATION_HANDOFF.md
NEW_CLUSTER_ENV_SETUP.md
```

These files are now tracked in git for cross-cluster handoff, along with the
other Markdown context docs. After cloning on the new cluster, verify the docs
are present before asking Codex to make path or environment decisions.
