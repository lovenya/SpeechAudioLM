# Progress Log

## 2026-05-12

### Documentation Setup

- Added repository guidance files for agents, project context, HPC workflow, progress tracking, changelog, and public overview.
- Captured the current project focus: dataset cleaning, fixed train/val/test split, AudioCraft/AudioGen setup, and baseline inference/evaluation.

### Current Data Reference

- Dataset root:
  `/project/rrg-csubakan/lovenya/datasets/ttas/extracted`
- Current best clean manifest:
  `/project/rrg-csubakan/lovenya/datasets/ttas/extracted/ttas/22k_samples_clean_manual_meta_clean.jsonl`

### Decisions

- Keep datasets, audio, zips, checkpoints, outputs, wandb runs, and secrets outside git.
- Ask before making structural, split, dataset, or experiment decisions.
- Stop after two failed HPC/toolchain fix attempts and summarize before continuing.

### Roadblocks / Open Items

- Fixed split policy and seed still need confirmation.
- AudioCraft/AudioGen installation path still needs to be validated for the target cluster.
- Baseline metrics and output format still need to be chosen.

## 2026-05-13

### Dataset Preparation

- Copied the clean TTAS manifest into `data/manifests/ttas_clean_normalized.jsonl`.
- Normalized `audio_file` and `speech_file` to relative paths such as `audio/<id>.wav` and `speech/<id>.wav`.
- Added `configs/ttas_dataset.json` with dataset root, source manifest, canonical manifest, seed `1337`, split ratio `80/10/10`, and split output paths.
- Created fixed split files in `data/splits/`:
  `ttas_train.jsonl`, `ttas_val.jsonl`, and `ttas_test.jsonl`.
- Added `utils/generate_metadata_report.py` for JSON report generation and screenshot-friendly CLI summaries.
- Generated `data/reports/ttas_dataset_report.json`.

### Split And Metadata Decisions

- Split is deterministic with seed `1337`.
- Stratification unit is `speaker_gender` plus bucketed `speaker_emotion`.
- Emotion labels with fewer than `20` total examples are grouped into `other` for split stratification; raw rare labels are still reported under the `other` breakdown.
- Relative paths are preferred in manifests/splits because the external dataset root is cluster-specific and belongs in config, not repeated in every row.

### Validation

- Manifest rows: `22,780`.
- Split rows: train `18,226`, validation `2,277`, test `2,277`.
- JSON parse errors: `0`.
- Duplicate sample IDs: `0`.
- Missing audio paths: `0`.
- Missing speech paths: `0`.
- Every manifest row appears in exactly one split.

### Git / Docs Boundary

- Internal Markdown docs remain local and ignored.
- `README.md` is the public Markdown doc intended for GitHub.
- Small metadata artifacts, configs, splits, reports, and utility scripts are versioned.
- Later migration work intentionally changed this boundary for handoff:
  selected Markdown context files are now tracked so a new cluster can recover
  the project state from git.

### HPC Setup Note

- Standard module order for project environments is to load modules first, then activate the venv:
  `module load StdEnv/2023 gcc cuda rust arrow python/3.10.13`
- `opencv` may be loaded as an extra module only when a workflow needs it.

### AudioCraft / AudioGen Setup

- Created/continued with clean AudioCraft environment:
  `.venv-audiocraft-torch21-py310`
- Standard activation for this env:
  `module load StdEnv/2023 gcc cuda rust arrow python/3.10.13`
  followed by
  `source .venv-audiocraft-torch21-py310/bin/activate`
- Verified login-node package versions:
  `torch==2.1.1`, `torchaudio==2.1.1`, `numpy==1.26.4`.
- Confirmed no `torch` module is available through the cluster module tree, so PyTorch is managed inside the venv from available wheels.
- Downloaded `facebook/audiogen-medium` weights outside the repo at:
  `/project/rrg-csubakan/lovenya/models/audiogen-medium`
- Verified downloaded model files:
  `state_dict.bin` around `3.5G`,
  `compression_state_dict.bin` around `225M`,
  total directory size around `3.7G`,
  with no leftover `.incomplete` or `.lock` files.
- Confirmed by code inspection that AudioCraft supports local model directories: when a directory is passed, it loads `state_dict.bin` and `compression_state_dict.bin` from that directory.

### AudioCraft Roadblocks

- Avoided installing full spaCy for now because AudioCraft imports it at module import time, but the expected AudioGen path should use T5 text conditioning rather than spaCy tokenization. A small import stub was enough to move past that top-level import during checks.
- Current blocker is a dependency shape mismatch:
  `dora_search==0.1.12+computecanada` imports `treetable.table._Node`,
  but the installed `treetable==0.1+computecanada` appears as a single `treetable.py` module, not a package with `table.py`.
- Next step is to resolve the `dora_search` / `treetable` mismatch deliberately, preferably by checking the expected upstream package layout or a compatible wheel, before trying multiple versions.
- For first GPU smoke test, request about `20 GB` VRAM if available. `10 GB` may be tight, `40 GB` is comfortable, and `80 GB` is not needed for the initial AudioGen inference check.

### Rorqual H100 AudioCraft Findings

- Requested and received a 20 GB H100 MIG interactive allocation.
- `SLURM_JOB_GPUS` may show a numeric GPU/MIG index such as `5`; the actual visible device is controlled by `CUDA_VISIBLE_DEVICES`, which can contain a MIG UUID. PyTorch correctly reported `cuda device count == 1`.
- `.venv-audiocraft-torch21-py310` is not usable on Rorqual H100 because `torch==2.1.1` lacks `sm_90` support. A minimal CUDA tensor allocation failed with:
  `RuntimeError: CUDA error: no kernel image is available for execution on the device`.
- `.venv-audiocraft` uses `torch==2.6.0+computecanada` and successfully ran:
  `torch.ones(1, device="cuda")`
  on `NVIDIA H100 80GB HBM3 MIG 2g.20gb`.
- In `.venv-audiocraft`, `from audiocraft.models import AudioGen` succeeds. It emits xFormers/Triton warnings because xFormers was built for a different Torch version and Triton is not installed; these are optimization warnings, not import blockers.
- Loading `/project/rrg-csubakan/lovenya/models/audiogen-medium` reached the T5 conditioner and failed only when trying to fetch `t5-large` from Hugging Face on the no-internet compute node.
- Verified from the downloaded checkpoint config:
  `conditioners.description.model=t5`,
  `conditioners.description.t5.name=t5-large`,
  `conditioners.description.t5.finetune=false`.
- Because `finetune=false`, AudioCraft does not bundle T5 weights inside the AudioGen checkpoint. It loads `T5Tokenizer` and `T5EncoderModel` separately from Hugging Face.
- Downloaded `t5-large` into `/project/rrg-csubakan/lovenya/models/hf_cache/transformers` using the working Hugging Face CLI from `.venv-audiocraft-torch21-py310`; cache size is about `4.9G` with no leftover incomplete files.
- The `hf` CLI inside `.venv-audiocraft` failed with a `typer.Argument` mismatch, so avoid that entrypoint for cache downloads unless the env is repaired.

### AudioGen Smoke Test Success

- Verified `t5-large` resolves offline in `.venv-audiocraft` when all cache variables are set:
  `HF_HOME`,
  `HF_HUB_CACHE`,
  `TRANSFORMERS_CACHE`,
  `HF_HUB_OFFLINE=1`,
  and `TRANSFORMERS_OFFLINE=1`.
- Loaded `/project/rrg-csubakan/lovenya/models/audiogen-medium` from local weights on a Rorqual H100 20 GB MIG allocation.
- Model load completed in about `22.73s`.
- Model properties:
  sample rate `16000`,
  channels `1`,
  max duration `10`.
- Generated a smoke-test sample for the prompt:
  `beautiful weather. rain is pouring down. there's thunder every now and then.`
- Generated tensor shape:
  `(1, 1, 160000)`,
  which is 10 seconds of mono audio at 16 kHz.
- Generation completed in about `13.76s`.
- CUDA reserved memory after generation was about `5.88G`, so a 20 GB MIG slice is enough for the current single-sample AudioGen smoke test.
- Output was written outside git:
  `/project/rrg-csubakan/lovenya/outputs/audiogen_smoke/smoke.wav`
- `audio_write` reported a tiny clipping probability during loudness handling; this did not block output creation.

### AudioCraft Setup Decisions

- The official AudioCraft README mentions Python 3.9 and Torch 2.1-era setup, but Rorqual's available module stack uses Python 3.10 and H100 GPUs require an `sm_90`-compatible Torch build.
- Chosen working setup for Rorqual H100 inference:
  Python `3.10.13`,
  `torch==2.6.0+computecanada`,
  `torchaudio==2.6.0+computecanada`,
  `numpy==1.26.4+computecanada`,
  `audiocraft==1.1.0`.
- Training-oriented or nonessential packages were not prioritized during setup. The focus was minimal inference capability: import AudioGen, load local weights, resolve offline text conditioning, and generate audio.
- xFormers was left as a warning rather than a blocker because generation works without its compiled CUDA extension. Revisit xFormers/Triton only if throughput or memory becomes a practical problem.

### Additional Baseline Model Downloads

- Downloaded `declare-lab/TangoFlux` to:
  `/project/rrg-csubakan/lovenya/models/tangoflux`
  size around `3.8G`.
- Downloaded `Qwen/Qwen3-Omni-30B-A3B-Instruct` to:
  `/project/rrg-csubakan/lovenya/models/qwen3-omni-30b-a3b-instruct`
  size around `52G`.
- Downloaded `HKUSTAudio/Audio-Omni` to:
  `/project/rrg-csubakan/lovenya/models/audio-omni`
  size around `20G`.
- Created separate environments:
  `.venv-tangoflux`,
  `.venv-qwen3-omni`,
  `.venv-audio-omni`.
- Did not set up CosyVoice/CosyTTS after the baseline list changed.
- Did not set up AudioFlamingo 3.

### TangoFlux Setup Notes

- Installed inference-focused dependencies into `.venv-tangoflux` from the cluster wheelhouse without version pins.
- Installed versions include:
  `torch==2.6.0+computecanada`,
  `torchaudio==2.6.0+computecanada`,
  `torchvision==0.21.0+computecanada`,
  `transformers==5.3.0+computecanada`,
  `diffusers==0.37.1+computecanada`,
  `accelerate==1.13.0+computecanada`,
  `librosa==0.11.0+computecanada`,
  `numpy==1.26.4+computecanada`.
- First attempted to follow the downloaded model repo's `requirements.txt`, which pinned `torch==2.4.0`; the cluster wheelhouse has `2.4.1+computecanada` but not `2.4.0+computecanada`. The setup then switched to unpinned cluster wheelhouse versions per user preference.
- Skipped `wandb`, `torchlibrosa`, and `datasets` unless a real inference path requires them.
- `datasets` import is a top-level stale dependency in `model.py` and `tangoflux.py`. Code search showed `load_dataset` and `Audio` are not used beyond the import lines.
- `TangoFluxInference` import succeeds with a runtime `datasets` stub before import.
- Smoke test succeeded on a Rorqual H100 20 GB MIG allocation.
- The run loaded `google/flan-t5-large` offline from the shared Hugging Face cache. `T5EncoderModel` reported `lm_head.weight` as unexpected; this is acceptable because TangoFlux uses the encoder, not the text-generation head.
- The run generated a 5 second 44.1 kHz stereo waveform with shape `(2, 220500)`, saved:
  `/project/rrg-csubakan/lovenya/outputs/tangoflux_smoke/smoke.wav`.
- CUDA reserved memory after generation was about `10.71G`, so a 20 GB MIG slice is enough for the current single-sample TangoFlux smoke test.

### Qwen3-Omni Setup Notes

- Confirmed the selected model is the instruct version: `Qwen/Qwen3-Omni-30B-A3B-Instruct`.
- Created `.venv-qwen3-omni` and installed wheelhouse core dependencies:
  `torch==2.6.0+computecanada`,
  `torchaudio==2.6.0+computecanada`,
  `torchvision==0.21.0+computecanada`,
  `transformers==5.3.0+computecanada`,
  `accelerate==1.13.0+computecanada`,
  `librosa==0.11.0+computecanada`,
  `soundfile==0.13.1+computecanada`,
  `safetensors==0.7.0+computecanada`.
- `qwen-omni-utils` is not in the cluster wheelhouse; it has not been installed yet.
- Local config and processor load succeeded:
  `Qwen3OmniMoeConfig` and `Qwen3OmniMoeProcessor`.
- Transformers emitted warnings about unrecognized rope parameter keys and fast image processor defaults, but config/processor loading did not fail.
- The full instruct model loaded successfully with `Qwen3OmniMoeForConditionalGeneration` from:
  `/project/rrg-csubakan/lovenya/models/qwen3-omni-30b-a3b-instruct`.
- First full load on a full H100 took about `446.83s` and reserved about `67.56G` CUDA memory. The full H100 allocation was the right choice for this checkpoint with the talker enabled.
- The `rope_parameters` warning for `mrope_section` and `interleaved` remained non-fatal.
- A text-only generation check used incorrect return handling. For this model, `return_audio=False` returns only the thinker generation result. Voice generation should use `return_audio=True`, which returns `(text_ids, audio)`.
- A voice generation attempt reached the talker path but failed with:
  `GenerationMixin.prepare_inputs_for_generation() got multiple values for argument 'next_sequence_length'`.
- Root cause appears to be a Transformers `5.3.0` Qwen3-Omni implementation mismatch: the talker subclass called `super().prepare_inputs_for_generation(...)` with old positional argument order, while the base `GenerationMixin` now has `next_sequence_length` as the second positional argument.
- Applied a venv-local workaround in:
  `.venv-qwen3-omni/lib/python3.10/site-packages/transformers/models/qwen3_omni_moe/modeling_qwen3_omni_moe.py`
  changing that `super()` call to use keyword arguments.
- Added local `TODO.md` entry to verify this against upstream Transformers later and consider a PR if still unfixed.
- After the venv-local workaround, the single-sample Qwen3-Omni voice smoke test succeeded.
- Generated text:
  `Qwen-Omni voice generation is now working.`
- Generated audio saved:
  `/project/rrg-csubakan/lovenya/outputs/qwen3_omni_smoke/smoke_voice.wav`.
- Output audio shape was `(1, 1, 68565)`, about `2.86s` at 24 kHz. CUDA reserved memory after generation was about `67.18G`.
- The attention-mask warnings were non-fatal for the smoke test, but later inference scripts should pass attention masks explicitly where supported.

### Audio-Omni Setup Notes

- Downloaded model files:
  `Audio-Omni.json`,
  `model.ckpt` around `22.2G`,
  and `synchformer_state_dict.pth` around `950M`.
- Cloned source outside git to:
  `/project/rrg-csubakan/lovenya/src/Audio-Omni`.
- Created `.venv-audio-omni` and installed core wheelhouse dependencies, including `accelerate==1.13.0+computecanada`.
- Editable install from Codex failed because package metadata creation in the external source directory hit sandbox read-only behavior. Current workaround is to set:
  `PYTHONPATH=/project/rrg-csubakan/lovenya/src/Audio-Omni:$PYTHONPATH`.
- Initial `AudioOmni` import failed on missing `k_diffusion`; user installed `k-diffusion` into `.venv-audio-omni`.
- After `k-diffusion` install, `from audio_omni import AudioOmni` succeeds.
- OpenCV is available as `opencv/4.13.0`, but current core import path does not require it. Video/demo paths may require `decord`, `ffmpeg`, or OpenCV later.
- Additional dependency misses surfaced during smoke testing: `jieba`, `pypinyin`, `x_transformers`, `einops_exts`, and `dac.nn`.
- `dac.nn` is provided by `descript-audio-codec`, not the unrelated `dac` package. If the wrong `dac` package is installed, uninstall it and install `descript-audio-codec`.
- Audio-Omni model construction reached its internal Qwen conditioner and failed because `Qwen/Qwen2.5-Omni-3B` was not locally available on the compute node.
- Downloaded `Qwen/Qwen2.5-Omni-3B` to:
  `/project/rrg-csubakan/lovenya/models/qwen2.5-omni-3b`
  size around `12G`.
- Retry should export:
  `QWEN_OMNI_MODEL_PATH=/project/rrg-csubakan/lovenya/models/qwen2.5-omni-3b`
  and initially `SKIP_QWEN_WEIGHTS=1`.
- Audio-Omni's API expects relative assets under `/project/rrg-csubakan/lovenya/src/Audio-Omni`; the baseline backend now handles this internally by temporarily changing directories during load/generation.
- After the Qwen2.5 dependency was resolved, the next missing package was `alias_free_torch`; installing `alias-free-torch==0.0.6` resolved that import.
- After resolving the relative vocab path and `alias-free-torch`, the next missing package was:
  `vector_quantize_pytorch`.
- Declared Audio-Omni model dependencies that are available from the cluster wheelhouse without exact old pins:
  `vector-quantize-pytorch`,
  `ema-pytorch`,
  `pytorch-lightning`,
  `torchmetrics`,
  `webdataset`,
  `pedalboard`.
- Declared dependencies not found in the Compute Canada wheelhouse:
  `encodec`,
  `auraloss`,
  `prefigure`,
  `v-diffusion-pytorch`,
  `laion-clap`.
- `flash_attn` is not currently a blocker. Audio-Omni detects it missing and disables Flash Attention.
- `flash-attn` remains uninstalled by choice for now. It is an optimization dependency and can be difficult to match against the cluster Torch/CUDA stack.
- Audio-Omni smoke test then succeeded with `seconds_total=10`.
- Successful Audio-Omni smoke output:
  `/project/rrg-csubakan/lovenya/outputs/audio_omni_smoke/smoke.wav`.
- Successful Audio-Omni run stats:
  model load about `196.15s`,
  generated shape `(2, 483328)`,
  generation about `2.65s`,
  CUDA reserved about `24.9G`.
- The previous `seconds_total=5` failure was caused by fixed-shape mel conditioning: the config expected the 10-second mel length, but a 5-second dummy prompt produced about half the frames.

### Baseline Inference CLI

- Added `baseline.py` and the `baselines/` package.
- Current backends:
  `audiogen`,
  `tangoflux`,
  `qwen3_omni`,
  `audio_omni`.
- The shared loader reads fixed JSONL splits from `configs/ttas_dataset.json` and supports `--split`, `--split-file`, `--prompt-field`, `--num-samples`, and `--start-index`.
- The shared runner writes organized outputs to:
  `results/baselines/<model>/<split>/<prompt_field>/audio/`
  plus a `manifest.jsonl`.
- `results/` is ignored by git.
- Added organized logs under:
  `logs/baselines/<model>/<split>/<prompt_field>/`.
- Each run writes:
  a human-readable `run_<timestamp>.log`
  and `resources_<timestamp>.jsonl` with process/system CPU, RSS memory, thread count, and CUDA allocated/reserved memory when available.
- Progress lines show `[current/total]`, elapsed time, ETA, row index, sample id, and output path.
- Added `--dry-run` to validate data selection and output paths without loading models. Dry-runs succeeded for all four backends on the TTAS test split.
- Audio-Omni backend sets the local Qwen2.5 path defaults and handles the source-tree working directory internally, so the CLI can be launched from the repo root.

## 2026-05-14

### Baseline Task Modes

- Added task-aware baseline inference in commit `1c90785 [feat]: add baseline task modes`.
- The CLI now uses `--task` and organizes outputs/logs under:
  `results/baselines/<model>/<task>/<split>/<prompt_field>/`
  and
  `logs/baselines/<model>/<task>/<split>/<prompt_field>/`.
- Native T2A models now receive raw `audio_cap` prompts by default:
  AudioGen, TangoFlux, and Audio-Omni `T2A`.
- Audio-Omni supports native `T2A` and `T2S`, plus cascaded `S2A` and `A2S`.
  - `S2A`: speech audio -> understanding text -> likely environmental audio description -> Audio-Omni `T2A`.
  - `A2S`: environmental audio -> understanding text -> likely speech transcript -> Audio-Omni `TTS`.
- Qwen3-Omni pseudo-T2A is intentionally separated from real T2A baselines:
  `pseudo_t2a_direct` and `pseudo_t2a_phoneme`.
  These are talker probes, not native environmental-audio generation.
- Dry-runs passed for the committed task modes.
- Five-sample smoke tests were launched for:
  AudioGen `T2A`, TangoFlux `T2A`, Audio-Omni `T2A`/`T2S`/`S2A`/`A2S`, and Qwen3-Omni pseudo-T2A modes.
- Audio-Omni native `T2S` completed 5/5 samples. The final elapsed time includes model load plus all samples; per-sample `seconds=...` lines are generation time for that sample only.

### Audio-Omni Cascade Roadblocks

- Audio-Omni cascades failed first on:
  `ModuleNotFoundError: No module named 'qwen_omni_utils'`.
- Root cause: cascades call Audio-Omni's understanding path, which imports `qwen_omni_utils` at runtime. Native `T2A`/`T2S` do not use this path, so they worked without it.
- Audio-Omni lists `qwen-omni-utils==0.0.9` in `requirements.txt`, but not in `setup.py` `install_requires`, so it was missed during the minimal inference-focused install.
- Manual fix in `.venv-audio-omni`:
  `python -m pip install --no-index qwen-omni-utils`.
- After that dependency was installed, cascades reached Qwen2.5 understanding but failed with:
  `ValueError: Chelsie is not available, available speakers: dict_keys([])`.
- Root cause: upstream Audio-Omni calls Qwen2.5 with `return_audio=False`, but this installed Transformers version still routes through logic that validates the default audio-output speaker. Audio-Omni intentionally loads only the thinker, so `speaker_map` is empty.
- A first local patch using `generation_mode="text"` still failed because the installed Qwen2.5 wrapper validates the default speaker before reaching text-only generation.
- Final working local patch: Audio-Omni baseline runner now uses an internal `_understand(...)` helper for cascades and calls the loaded Qwen2.5 thinker directly, matching Audio-Omni's Gradio understanding path.
- Audio-Omni `A2S` completed 5/5 samples after this patch. First sample took about `17.37s`; later samples took about `4.7s` to `5.6s`; total elapsed was about `3:46` including model load.
- Do not document manual install commands inside source code. Keep source errors concise; operational commands belong in docs/chat.

## 2026-05-21

### Evaluation Scaffold

- Added `evaluate.py` and the `evaluation/` package for manifest-driven baseline evaluation.
- Added explicit CLI flags for metric selection and lightweight checks:
  `--metrics`, `--dry-run`, `--max-samples`, and `--cache-dir`.
- Evaluation outputs are written under:
  `results/evals/<task>/<dataset>/<model>/`.
- Each run writes timestamped `per_sample`, `summary`, `summary.csv`, `report`, and `run_config` files plus `latest_*` aliases, including `latest_run_config_<split>.json`.
- `summary.json` records row counts, successful rows, skipped/error rows, metric summaries where meaningful, and errors grouped by metric.
- `run_config` records command, device, cache path, metrics config, task metric map, backend metadata, and audio preprocessing policy placeholders.
- Per-sample rows keep generated outputs under `predictions` and metric values under `metrics`; WER predictions use generated text/transcript fields, not metric-named prediction keys.

### Metric Semantics

- Added metric specs for WER, semantic equivalence, MACE, SPIDEr, CLAP similarity, generated-speech ASR-WER/dWER/UTMOS/DNSMOS, and text-to-audio FAD/KAD/KLD/Inception Score.
- `asr_wer` is defined as:
  `WER(reference_transcript, ASR(generated_speech))`.
- `dwer` is directional and defined as:
  `WER(ASR(reference_speech), ASR(generated_speech))`.
- T2A corpus-level metrics are FAD, KAD, KL_PaSST, and Inception Score.
  `clap_score_t2a` is per-sample then aggregated.
- A2T CLAPSim and T2A CLAPscore are separate metrics:
  `aac_clapsim` and `clap_score_t2a`.
- FAD remains the v1 Frechet-style audio metric; generic FD is deferred unless a specific paper/tool comparison requires it.

### Task Metric Mapping

- Added `configs/task_metric_map.json` so task-to-metric validity is config-driven.
- The current design does not use `--stage`. Intermediate cascade outputs should be evaluated as component manifests directly:
  run `--task S2T` for transcript/WER evaluation and `--task A2T` for caption metrics.
- `S2A + wer` is invalid for the combined final-audio manifest.
- `A2S + mace`/caption metrics are invalid for the combined final-speech manifest.
- `T2A + asr_wer` is invalid because ASR-WER is a generated-speech metric.
- Unit tests cover valid/invalid task metric pairs, default metric selection, WER directionality, corpus-level metric marking, and metric-specific validation.

### Next Work

- Before external backend implementation, download or configure the metric toolkits/checkpoints in dedicated environments as needed.
- Current backend choices remain configurable:
  `aac-metrics` for MACE/SPIDEr/AAC CLAPSim, `stable-audio-metrics` for
  T2A CLAPscore and KL_PaSST, `audioldm_eval` for FAD/Inception Score, `kadtk`
  for KAD, UTMOSv2 for UTMOS, Whisper large-v3 for ASR, and a configurable text
  embedding model for semantic equivalence.
- T2T, A2S, non-final S2A, and DNSMOS backend decisions are deferred.
- Cascaded baseline runs still need component-specific manifests so intermediate S2T/A2T outputs can be evaluated without overloading the combined cascade manifest.

### Metric Backend Setup

- Metric evaluation environments should live under `.venvs/metrics/`.
  Model/baseline environments should live under `.venvs/models/` or another
  model-specific subfolder.
- Do not move existing venv directories in place; recreate them at the intended
  path because venvs can contain absolute paths.
- Whisper `large-v3` was downloaded and loaded successfully from:
  `/project/rrg-csubakan/lovenya/models/metrics/whisper`.
- The successful load reported:
  `ModelDimensions(n_mels=128, n_audio_ctx=1500, n_audio_state=1280, n_audio_head=20, n_audio_layer=32, n_vocab=51866, n_text_ctx=448, n_text_state=1280, n_text_head=20, n_text_layer=32)`.
- The downloaded Whisper checkpoint size was about `2.88G`.
- `aac-metrics-download --cache_path` completed into:
  `/project/rrg-csubakan/lovenya/cache/metrics/aac_metrics`.
- AAC SPIDEr smoke initially failed because Java was not on `PATH`, then Java 17
  was rejected by `aac-metrics`. The Stanford CoreNLP jar is present in the
  project cache. Load `java/11.0.22` before the smoke/evaluation command.
- `audioldm_eval` installed successfully with the manual `--no-deps` dependency
  workaround and `smoke_audioldm_eval.py` imported `EvaluationHelper` on CPU.
- AAC SPIDEr smoke passed after switching to Java 11 and using a two-sample
  fixture. Output keys were `cider_d`, `spice`, and `spider` for both corpus and
  sample scores.

## 2026-05-22

### Eval Backend Smoke Checks

- Began compute-node end-to-end metric smoke checks against existing
  Audio-Omni baseline manifests, writing eval outputs to:
  `/scratch/lovenya/SpeechAudioLM/results/eval_smoke_existing`.
- Whisper large-v3 `asr_wer,dwer` passed through `evaluate.py` on
  `audio_omni` `T2S` with `--max-samples 1`; runtime was about `42s`.
- UTMOSv2 initially appeared stuck. Direct backend debugging showed the cached
  checkpoint was corrupted:
  `PytorchStreamReader failed reading zip archive`.
- Redownloaded:
  `/scratch/lovenya/SpeechAudioLM/cache/metrics/utmosv2/models/fusion_stage3/fold0_s42_best_model.pth`.
  The fixed file is `818531314` bytes and loads with `torch.load`.
- Direct UTMOSv2 model check then succeeded on CUDA:
  model creation about `7.47s`, prediction about `23.17s`, score about
  `2.0879` for the first existing Audio-Omni T2S sample.
- UTMOSv2 then passed through `evaluate.py` on `audio_omni` `T2S` with
  `--max-samples 1`; runtime was about `1m38s`.
- `aac-metrics` SPIDEr passed through `evaluate.py` on a two-sample
  `audio_omni` A2T-style manifest; runtime was about `15s`.
- Combined AAC run with `mace,spider,aac_clapsim` failed in the MACE/MS-CLAP
  path because the backend could not resolve a Hugging Face text encoder cache
  while offline. Relevant cache folders for `gpt2`, `microsoft/msclap`,
  `bert-base-uncased`, and `roberta-large` appear to exist on scratch, so the
  next step is to debug cache/env resolution by running `aac_clapsim` and
  `mace` separately.
- Remaining eval backend smoke checks:
  `fad,inception_score` via `audioldm_eval`, `kad` via `kadtk`, and
  `clap_score_t2a,kl_passt` via stable-audio-metrics.

## 2026-05-26

### Eval Backend Fixes After Smoke Debugging

- Fixed AAC worker cache roots so `mace`, `spider`, and `aac_clapsim` can use
  `/scratch/lovenya/SpeechAudioLM/cache/metrics/aac_metrics/huggingface`
  instead of the incomplete generic metric Hugging Face cache.
- Fixed MS-CLAP-backed AAC metrics by configuring `batch_size: 1` for `mace`
  and `aac_clapsim`; this avoids a backend collator error on variable-length
  GPT-2 token tensors.
- Fixed UTMOSv2 compute-node behavior by enabling backend-specific
  `HF_HUB_OFFLINE=1` and `TRANSFORMERS_OFFLINE=1` in `configs/eval_metrics.json`.
- Verified combined `asr_wer,dwer,utmos` on 5 `audio_omni` `T2S` samples.
- `kad` and `clap_score_t2a,kl_passt` passed earlier two-sample T2A smoke
  checks through `evaluate.py`.
- AudioLDM Eval had missing VGGish assets; `vggish-10086976.pth` and
  `vggish_pca_params-970ea276.pth` are now present under the scratch Torch hub
  checkpoint cache, but `fad,inception_score` still need a final retry.
- `aac_clapsim` and `mace` still need final retry after the cache/batch fixes.

### Cluster Migration Handoff

- Added local handoff docs for moving the project to a new cluster:
  `CLUSTER_MIGRATION_HANDOFF.md` and `NEW_CLUSTER_ENV_SETUP.md`.
- Recorded current repo path, dataset/model/cache/toolkit/result locations,
  environment names, eval smoke status, and recommended new-cluster setup order.
- Important policy: transfer checkpoints/caches/datasets/results as external
  resources, but recreate venvs on the new cluster rather than moving them.
- Added Codex agent knowledge transfer notes: tracked `AGENTS.md` should be read
  by the new agent, and the custom local skill at
  `/home/lovenya/.codex/skills/update_docs/SKILL.md` should be copied to the new
  cluster if Codex there does not already have it.
- Clarified migration transfer strategy: clone the repo and tracked handoff
  docs, transfer the TTAS dataset and local/custom assets, recreate venvs, and
  re-download public Hugging Face/GitHub model and metric assets when practical.
- Copied that custom skill into the repository at
  `codex/skills/update_docs/SKILL.md` so the GitHub clone carries the workflow
  without depending on the old cluster home directory.
- Added a reusable `cluster-migration` Codex skill and vendored it at
  `codex/skills/cluster-migration/SKILL.md` so future projects can follow the
  same audit, transfer, re-download, recreate, and validation workflow.
## 2026-05-27

### TamIA Migration Integration

- Completed the clean TamIA migration under `/scratch/l/lovenya/salm_migration_clean` with repo at `/scratch/l/lovenya/SpeechAudioLM`.
- Validated TTAS against `data/manifests/ttas_clean_normalized.jsonl`: `22,780` rows, missing audio `0`, missing speech `0`, and source manifest present.
- Verified model environments for AudioCraft, TangoFlux, Qwen3-Omni, and Audio-Omni with login-node-safe package/import checks.
- Created and verified metric environments for Whisper, AAC metrics, stable-audio-metrics, UTMOSv2, AudioLDM Eval, and KADTK.
- Downloaded AAC metric runtime assets into the clean cache and confirmed the tiny SPIDEr CPU smoke passes with Java 11.
- Rewrote active dataset/model/metric config paths, backend defaults, smoke scripts, and Slurm helpers from old Rorqual/FIR roots to TamIA clean roots.
- Updated Slurm helpers away from old `rrg-*` accounts and MIG GPU requests; TamIA jobs now use whole-node H100-style defaults and use the verified default account `aip-csubakan`.
- Verified JSON configs, Python syntax, `tests/test_evaluation.py`, configured external path existence, and baseline dry-runs for AudioGen, TangoFlux, Qwen3-Omni, and Audio-Omni representative tasks.
- Added `submission_scripts/run_tamia_model_smokes_one_node.sbatch` for a one-hour TamIA smoke allocation that runs four one-sample backend jobs across the four H100s using `CUDA_VISIBLE_DEVICES`.
- Corrected AudioGen and TangoFlux Hugging Face cache variables to point at the actual migrated cache root instead of a nonexistent `hf_cache/transformers` subdirectory.

### Remaining TamIA Gates

- The first one-node, four-GPU model smoke ran. TangoFlux T2A and Audio-Omni T2A passed; AudioGen and Qwen3-Omni A2S exposed retry gates after environment/model-generation fixes.
- Current next model-smoke gate: `submission_scripts/run_tamia_retry_model_smokes_one_node.sbatch`, which runs one-sample AudioGen T2A, Qwen3-Omni A2S, Audio-Omni A2S, and Audio-Omni S2A across one 4-H100 node.
- Retry job `308995` ran with only `cpu=1,mem=2G` because the retry script had no explicit CPU/RAM request. Slurm marked the job `OUT_OF_MEMORY`; Qwen3-Omni A2S passed, while AudioGen and both Audio-Omni lanes were killed before backend conclusions could be drawn.
- Updated the retry script to request `--cpus-per-task=16` and `--mem=192G`, keeping one node and four H100s, and added `--overwrite` to each lane so retries regenerate one-sample outputs instead of silently skipping existing files.
- Retry job `309034` completed with `ExitCode 0:0` using `cpu=16,mem=192G,gres/gpu:h100=4`. AudioGen T2A, Qwen3-Omni A2S, Audio-Omni A2S, and Audio-Omni S2A all passed. Manifests contain only `success` rows; Qwen3-Omni has two success rows because the previous successful retry row remained appended.
- Run `evaluate.py` metric scoring smokes after real generated baseline manifests exist. For tiny metric checks, prefer `submission_scripts/run_eval_smoke_existing_manifests.sh` inside an existing compute allocation before using the broader eval array script.
- Metric smoke job `309048` failed at the first `audioldm_eval` group (`fad,inception_score`) because `torch.hub.load("harritaylor/torchvggish", "vggish")` could not find the Torch Hub source repo under `/scratch/l/lovenya/salm_migration_clean/cache/metrics/torch/hub` and compute nodes have no internet. The VGGish checkpoint files are present, but the source repo cache is missing.
- Updated `submission_scripts/run_tamia_eval_smoke_retry_manifests.sbatch` to split Audioldm Eval, KAD, stable-audio-metrics, and speech metrics into separate backend groups and continue after failures, while still exiting nonzero if any group fails.
- Use the verified default TamIA account `aip-csubakan`; `aip-ravanelm` is also associated with the user.

### Goal Refinement Notes

- `goal_updated.txt` now records the user/agent execution split: the agent prepares exact commands and debugs from evidence, while the user runs quota-burning downloads, env setup, and Slurm submissions unless explicitly instructing otherwise.
- DNSMOS is now documented as the main missing metric adapter: allowed in `configs/task_metric_map.json`, but still `backend: null` in `configs/eval_metrics.json` and not dispatched by `evaluation/runner.py` or `scripts/eval_backend_worker.py`.
- The modular cascade goal is distinct from existing backend-specific cascades: `audio_omni` A2S/S2A and `qwen3_omni` A2S smoke behavior do not yet satisfy the selectable-stage CLI requirement.

## 2026-05-28

### TamIA Metric Smoke Follow-Up

- Confirmed the one-node model-smoke retry job `309034` is the current migration model gate pass: AudioGen T2A, Qwen3-Omni A2S, Audio-Omni A2S, and Audio-Omni S2A all produced successful smoke manifests.
- Metric smoke job `309055` exposed remaining metric-cache/setup blockers: `audioldm_eval` needed `resampy`, KADTK needed the PANN checkpoint, and stable-audio-metrics needed the LAION-CLAP checkpoint. The user completed those setup/download steps.
- Speech metric smoke job `309063` proved Whisper `asr_wer,dwer` runs on generated A2S audio, then failed in UTMOS because `facebook/wav2vec2-base` did not resolve from the offline Hugging Face cache.
- Verified the UTMOS wav2vec files were present and direct local-path loading worked. Repo-id offline loading required exporting `HF_HUB_CACHE` and `HUGGINGFACE_HUB_CACHE` in addition to `HF_HOME` and `TRANSFORMERS_CACHE`; the user confirmed the corrected offline test returned OK.
- Updated the TamIA metric smoke scripts so Slurm jobs inherit the verified Hugging Face cache variables. The speech-only metric smoke now continues to Qwen3-Omni even if Audio-Omni fails, matching the grouped full metric smoke behavior.
- Full metric smoke job `309212` timed out in the first `audioldm_eval` group with near-zero CPU/GPU use. Inspection showed AudioLDM Eval hard-codes `~/.cache/audioldm_eval/ckpt` for Cnn14 PANN checkpoints, so the worker tried Zenodo from a no-internet compute node. The metric smoke scripts now set `HOME=$SALM_SCRATCH/cache/metrics/runtime_home`; the required files are `Cnn14_mAP=0.431.pth` and `Cnn14_16k_mAP=0.438.pth` under `$HOME/.cache/audioldm_eval/ckpt`.
- Metric smoke job `309247` ran all backend groups after the Cnn14 files were cached. KADTK passed for AudioGen T2A and Audio-Omni S2A, although the one-sample corpus KAD summary is `NaN`, so use at least a two-sample check before treating KAD values as meaningful. Remaining blockers: AudioLDM Eval hits the PyTorch 2.6 default `torch.load(weights_only=True)` incompatibility for legacy PANN checkpoints; stable-audio-metrics needs `roberta-base` cached; UTMOS needs `timm/tf_efficientnetv2_s.in21k_ft_in1k` cached.
- Metric smoke job `309252` verified generated-speech metrics now pass for both Audio-Omni A2S and Qwen3-Omni A2S. KADTK still executes for T2A/S2A. Remaining blockers narrowed to AudioLDM Eval rejecting one-file smoke directories and stable-audio-metrics hitting the same PyTorch 2.6 legacy checkpoint load issue for LAION-CLAP. The worker now duplicates staged one-sample AudioLDM files to satisfy its directory check and loads trusted LAION-CLAP checkpoints with `weights_only=False`.
- Refined the cascaded-baseline target model list: the text-to-text Llama option is `meta-llama/Llama-3.3-70B-Instruct`, not an older Llama 3 70B checkpoint.
