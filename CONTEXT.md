# Project Context

## Current Objective

SpeechAudio LM is preparing cascaded baselines for bidirectional generation between environmental audio and speech. The immediate target is the TTAS validation split, not the full dataset or test split.

## Active Roots

```text
repo: /scratch/l/lovenya/SpeechAudioLM
scratch root: /scratch/l/lovenya/salm_migration_clean
dataset root: /scratch/l/lovenya/salm_migration_clean/datasets/ttas/extracted
models: /scratch/l/lovenya/salm_migration_clean/models
metric cache: /scratch/l/lovenya/salm_migration_clean/cache/metrics
cascaded validation results: /scratch/l/lovenya/salm_migration_clean/results/cascaded_full_validation/ttas_val
cascaded validation metadata: /scratch/l/lovenya/salm_migration_clean/results/cascaded_full_validation_metadata/ttas_val
```

Keep datasets, generated audio, checkpoints, result manifests, Slurm logs, and caches outside git.

## Dataset State

- Config: `configs/ttas_dataset.json`
- Canonical manifest: `data/manifests/ttas_clean_normalized.jsonl`
- Fixed splits: `data/splits/ttas_train.jsonl`, `data/splits/ttas_val.jsonl`, `data/splits/ttas_test.jsonl`
- Split counts: train `18,226`, val `2,277`, test `2,277`, total `22,780`
- Split policy: fixed `80/10/10`, seed `1337`, stratified by `speaker_gender` plus bucketed `speaker_emotion`

Use `val` for the current cascaded baseline experiments. Do not change splits without an explicit decision.

## Audio Metadata State

PANNs/CNN14 AudioSet tag metadata prep is implemented in `scripts/prepare_audio_tags.py`.
The full TTAS split tag manifests have been generated as enriched copies; the
original split files remain unchanged.

```text
data/manifests/audio_tags/ttas_train_panns_cnn14.jsonl
data/manifests/audio_tags/ttas_val_panns_cnn14.jsonl
data/manifests/audio_tags/ttas_test_panns_cnn14.jsonl
data/reports/audio_tags/ttas_panns_cnn14_tag_frequency_summary.json
```

Tagging settings used for the canonical run: threshold `0.6`, `min_tags=2`,
`fallback_top_k=3`, `fallback_max_score_gap=0.25`, and `store_top_tag_scores=10`.
PANNs model/env paths are recorded in `configs/project_paths.json`.

Primary audio class pooling is implemented in `scripts/assign_audio_classes.py`
from `configs/audio_tag_class_pooling.json`, with
`speechaudio_audio_tag_class_pooling.md` kept as the human-readable rationale.
Canonical assignment now uses `priority_with_evidence_guard`; `priority_first_v1` remains useful only as a diagnostic comparison.

```text
data/manifests/audio_classes/ttas_train_panns_cnn14_classes.jsonl
data/manifests/audio_classes/ttas_val_panns_cnn14_classes.jsonl
data/manifests/audio_classes/ttas_test_panns_cnn14_classes.jsonl
data/reports/audio_classes/ttas_panns_cnn14_primary_audio_class_counts.json
data/reports/audio_classes/ttas_panns_cnn14_unknown_audio_class_samples.jsonl
```

## Cascaded Baseline Plan

The selected full design is the Cartesian product of:

```text
ASR: whisper, qwen3_asr
captioner: audio_flamingo_next, qwen3_omni_captioner
text model: llama3_3_70b_instruct, qwen3_32b
TTS: f5_tts, cosyvoice3
T2A: audiogen, tangoflux
```

That gives `2 x 2 x 2 x 2 x 2 = 32` paired cascade combinations. The canonical metadata file is:

```text
configs/cascaded_baseline_variations.json
```

Run these stage-wise and reuse outputs; do not rerun shared stages 32 times. For paired stages that can run in parallel, report wall/load time as the maximum of the two lanes, but report GPU memory and GPU-hours as sums.

## Cascaded Validation Status

Full validation runs use the stable run id/path `ttas_val`; transient Slurm job ids stay in metadata/logs, not in canonical result paths. The 32 paired configurations are implemented through reusable stage lanes:

- `S2T`: 2 ASR lanes over `2,277` validation rows.
- `A2T`: 2 captioner lanes over `2,277` validation rows, with expected row-level skips carried forward.
- `T2T`: 8 text lanes, reusing ASR/captioner outputs.
- Final `T2A`: 8 unique S2A audio-generation lanes.
- Final `T2S`: 8 unique A2S speech-generation lanes.

Canonical full-run manifests live under:

```text
/scratch/l/lovenya/salm_migration_clean/results/cascaded_full_validation/ttas_val/cascaded/<stage>/<lane>/ttas/val/manifest.jsonl
```

Use these canonical manifests as inputs to evaluation. Shard-specific run directories may remain temporarily when their audio files are still referenced or a repair job has not merged yet.

Current active repair work is limited to completing missing final lanes:

- Llama `AudioGen` S2A lanes are being resumed from the partial `313433` run.
- Qwen-captioner `CosyVoice3` A2S lanes are being resumed from partial manifests.
- Audio-Flamingo-Next A2S TTS lanes are being regenerated with fixed reference-speech paths.

After each job, inspect Slurm accounting, resource-monitor output, manifest row counts, WAV readability/duration/RMS, and row-level errors before treating the lane as complete.

## Smoke Status

One-sample backend smoke job `312281` completed in `8m36s`. It showed all selected backends can run except the first attempt used stale envs for `qwen3_asr` and `cosyvoice3`:

- passed in `312281`: `whisper`, `audio_flamingo_next`, `qwen3_omni_captioner`, `qwen3_32b`, `llama3_3_70b_instruct`, `f5_tts`, `audiogen`, `tangoflux`
- `qwen3_asr` passed after switching to `.venv-cascade-qwen3-asr-py312` in retry job `312299`
- `cosyvoice3` passed after using the Arrow 17 module stack in retry job `312322`

Important: the smoke scripts use `--continue-on-error`; Slurm `ExitCode 0:0` does not guarantee every lane succeeded. Always inspect each lane manifest for `status`.

## Working Model Environments

Use these model envs for the 32-combination baseline work:

```text
whisper S2T:              $SALM_SCRATCH/.venvs/models/.venv-cascade-asr
qwen3_asr S2T:            $SALM_SCRATCH/.venvs/models/.venv-cascade-qwen3-asr-py312
audio_flamingo_next A2T:  $SALM_SCRATCH/.venvs/models/.venv-cascade-caption
qwen3_omni_captioner A2T: $SALM_SCRATCH/.venvs/models/.venv-cascade-caption
qwen3_32b / llama T2T:    $SALM_SCRATCH/.venvs/models/.venv-cascade-text
f5_tts T2S:               $SALM_SCRATCH/.venvs/models/.venv-cascade-f5-tts
cosyvoice3 T2S:           $SALM_SCRATCH/.venvs/models/.venv-cascade-cosyvoice3-py310
audiogen T2A:             $SALM_SCRATCH/.venvs/models/.venv-audiocraft
tangoflux T2A:            $SALM_SCRATCH/.venvs/models/.venv-tangoflux
```

For CosyVoice3, load this exact module stack before activating the env:

```bash
module load StdEnv/2023 gcc cuda/12.6 rust arrow/17.0.0 python/3.10.13
source /scratch/l/lovenya/salm_migration_clean/.venvs/models/.venv-cascade-cosyvoice3-py310/bin/activate
```

This exposes `pyarrow 17.0.0` from the Arrow module and avoids `ModuleNotFoundError: pyarrow`.

For Qwen3-ASR, use the py312 env. Do not install `nagisa`; it pulls `DyNet` and can spawn many C++ compiler processes on login nodes.

## Evaluation Runtime Environments

The current environment map is `configs/runtime_environments.json`.

Validated shared metrics env, currently in home as a temporary quota-safe build:

```text
/home/l/lovenya/.venvs/speechaudiolm/.venv-eval-metrics-shared-py312
```

Use this module stack before activating it:

```bash
module load StdEnv/2023 gcc cuda/12.6 opencv/4.13.0 python/3.12.4 scipy-stack
```

It has passed compute-node smokes for:

```text
asr_wer, dwer, mace, spider, aac_clapsim, fense, dnsmos, utmos, nli_contradiction
```

The shared env replaces the old standalone Whisper, AAC, DNSMOS, and UTMOS metric envs. Whisper, AAC, and DNSMOS standalone envs were deleted; the old standalone UTMOSv2 env is still present under scratch and can be deleted after explicit approval.

Keep these metric envs separate for now:

```text
audioldm_eval:          fad, inception_score
kadtk:                  kad
stable_audio_metrics:   clap_score_t2a, kl_passt
prometheus_vllm:        g_eval_reference_free
```

Do not merge model-specific envs yet. Their source checkouts and binary dependency stacks are riskier to consolidate than the metric envs, and the completed inference outputs do not justify that risk right now.

Current temporary Prometheus env:

```text
/home/l/lovenya/.venvs/speechaudiolm/.venv-eval-prometheus-vllm-py312
```

The env exists and `pip install vllm` succeeded under the cluster-default pip configuration. The remaining gate is a compute-node smoke with the local Prometheus model before any packed eval job.

## Immediate Next Steps

1. Keep using `configs/runtime_environments.json` as the environment-to-workload map.
2. Smoke the home Prometheus/vLLM env on a compute node against a small T2T validation sample before any full judging run.
3. Recreate the validated shared metrics env and the Prometheus env under scratch once quota cleanup is complete, then update configs and re-smoke.
4. Prune stale logs and one-off scratch outputs when they are no longer needed for debugging or audit trails.
5. Run evaluations from stable canonical manifests under `ttas_val`, not transient shard/run-id directories.

## Durable Cluster Rules

- Login nodes are for editing, git, small checks, and job submission only.
- Do not let pip build native packages on login nodes. In particular, avoid `nagisa`/`DyNet` source builds.
- For any unavoidable source build, use an interactive/compute job and cap parallelism with `MAKEFLAGS=-j1`, `CMAKE_BUILD_PARALLEL_LEVEL=1`, and `MAX_JOBS=1`.
- On TamIA, a GPU job reserves a 4-H100 node. Pack independent lanes inside one node when safe, assign `CUDA_VISIBLE_DEVICES` explicitly, and keep one-sample smoke walltimes short.
- Store generated resource JSON, logs, manifests, and audio outside git. Do not commit post-run accounting artifacts.
- If an HPC/toolchain fix fails twice, stop and summarize before trying more alternatives.
