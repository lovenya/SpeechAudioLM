# Changelog

All notable repository-level changes should be recorded here.

## 2026-05-12

- Added concise project guidance files: `AGENTS.md`, `CONTEXT.md`, `HPC_BEST_PRACTICES.md`, `PROGRESS_LOG.md`, `CHANGELOG.md`, and `README.md`.
- Documented current SpeechAudio LM setup focus, external dataset locations, git safety rules, and HPC workflow expectations.

## 2026-05-13

- Added normalized TTAS manifest, deterministic train/validation/test split files, dataset config, and metadata report.
- Added `utils/generate_metadata_report.py` for JSON and CLI metadata summaries.
- Documented relative path resolution and report generation in `README.md`.
- Updated ignore rules so private Markdown handoff docs and Python cache directories remain local.
- Documented AudioCraft/AudioGen setup state, external `facebook/audiogen-medium` weight location, and the current `dora_search` / `treetable` dependency blocker.
- Documented the working Rorqual H100 AudioGen inference setup, offline `t5-large` cache requirement, and successful AudioGen smoke-test generation.
- Documented baseline model downloads and setup checks for TangoFlux, Qwen3-Omni-30B-A3B-Instruct, and Audio-Omni.
- Documented successful TangoFlux smoke-test generation, Qwen3-Omni full-model load on a full H100, and the Audio-Omni dependency on local `Qwen/Qwen2.5-Omni-3B`.
- Documented the venv-local Qwen3-Omni talker workaround, local upstream-PR TODO, and latest Audio-Omni dependency notes.
- Documented successful Qwen3-Omni voice smoke-test generation and the next Audio-Omni dependency batch.
- Added the baseline inference CLI with shared dataset loading, organized git-ignored `results/` outputs, and backends for AudioGen, TangoFlux, Qwen3-Omni, and Audio-Omni.
- Documented successful Audio-Omni smoke-test generation and the fixed 10-second duration requirement for the current Audio-Omni baseline path.
- Added baseline run logging, progress/ETA display, and resource-sampling logs under git-ignored `logs/`.

## 2026-05-14

- Added task-aware baseline inference with `--task`, task-scoped output/log paths, default prompt-field selection, Audio-Omni cascades, and Qwen3-Omni pseudo-T2A task names.
- Documented current baseline task-mode behavior and smoke-test status.
- Documented Audio-Omni cascade runtime dependencies and the Qwen2.5 thinker-direct understanding compatibility workaround.

## 2026-05-21

- Added the evaluation scaffold with task metric validation, metric-specific manifest checks, per-sample/summary/run-config outputs, and latest-result aliases.
- Recorded generated-speech metric semantics, including directional `dwer = WER(ASR(reference_speech), ASR(generated_speech))`.
- Documented that S2A/A2S intermediate metrics should be evaluated through component task manifests, such as `S2T` or `A2T`, rather than a `--stage` flag on combined cascade manifests.
- Added `configs/task_metric_map.json` and `configs/eval_metrics.json` as the current config surfaces before external metric backend installation.
- Documented metric backend setup commands under `.venvs/metrics/` and recorded the successful Whisper `large-v3` download/load check.
- Documented successful AAC SPIDEr smoke setup with project cache, Java 11, and a valid two-sample fixture; documented successful `audioldm_eval` import smoke.
- Split AAC CLAPSim from T2A CLAPscore with explicit metric names, replaced generic KLD with `kl_passt`, and recorded the selected stable-audio-metrics/UTMOSv2 backend direction.

## 2026-05-22

- Added sharded baseline/eval Slurm helpers, baseline shard manifest merging, and node-level Slurm resource monitoring with scratch summaries.
- Recorded compute-node eval smoke status: Whisper `asr_wer,dwer`, UTMOSv2 `utmos`, and AAC `spider` now pass through `evaluate.py`.
- Documented the UTMOSv2 corrupted checkpoint issue and redownloaded scratch checkpoint path.
- Documented the current AAC `mace`/`aac_clapsim` blocker: offline MS-CLAP/text-encoder cache resolution.

## 2026-05-26

- Added local cluster migration handoff docs covering repo, dataset, model,
  metric cache, toolkit, result, and environment locations.
- Added new-cluster setup guidance for recreating model/metric venvs, rewriting
  path configs, preserving offline caches, and validating baseline/eval smokes.
- Updated eval status after backend fixes: AAC metric cache roots are explicit,
  MS-CLAP-backed AAC metrics use `batch_size: 1`, and UTMOS runs Hugging Face
  dependencies in offline mode on compute nodes.
- Recorded that combined `asr_wer,dwer,utmos`, `kad`, and
  `clap_score_t2a,kl_passt` have passed smoke checks; `mace`, `aac_clapsim`,
  and `fad,inception_score` still need final retries.
- Recorded Codex agent knowledge transfer requirements: read tracked `AGENTS.md`
  and copy the custom `update_docs` skill to the new cluster's Codex home.
- Clarified cluster migration docs now that handoff Markdown is tracked: clone
  repo docs, transfer datasets/local assets, recreate venvs, and re-download
  public model/metric assets when practical.
- Vendored the custom `update_docs` Codex skill under
  `codex/skills/update_docs/SKILL.md` so a GitHub clone contains the project
  documentation workflow.
- Added a reusable `cluster-migration` Codex skill under
  `codex/skills/cluster-migration/SKILL.md` for future cross-cluster project
  moves.
