# Baseline Runner Conventions

Baseline outputs must use this canonical structure:

```text
results/baselines/<model>/<task>/<dataset>/<split>/
logs/baselines/<model>/<task>/<dataset>/<split>/
```

Each run writes generated audio under `audio/` and appends JSONL records to
`manifest.jsonl`. The manifest must include `model`, `task`, `dataset`,
`split`, `split_file`, `prompt_field`, `row_index`, `sample_id`, `prompt`,
`output_file`, `sample_rate`, `source`, and `model_extra` when available.

Cascaded runs should keep a combined final-output manifest and, when the
intermediate outputs are evaluable, component-specific manifests. For example,
an `S2A` run should preserve an `S2T` component manifest for transcript metrics
and a final `S2A` manifest for generated-audio metrics. Evaluate intermediate
components by running `evaluate.py` on the component task manifest directly,
not by passing stage flags to a combined cascade manifest.

Backend non-negotiables:

- Declare supported tasks through `configs/model_capabilities.json`.
- Keep generated audio, logs, and result manifests out of git.
- Follow task metric validity from `configs/task_metric_map.json`.
- Preserve progress logging with row index, sample id, elapsed time, ETA, and output path.
- Preserve resource logging for long or GPU-backed runs.
- Resolve dataset-relative paths against `configs/ttas_dataset.json`.
- Keep heavyweight or optional dependencies inside backend-specific load paths with clear errors.
- Do not add nested or broad imports unless they are required to keep optional stacks isolated.

Model-specific notes:

- Qwen3-Omni `A2S` is a baseline-only cascade that skips audio understanding
  and uses the paired `speech_text` as the intermediate transcript before TTS.
  It defaults to `speech_text` even though Audio-Omni `A2S` defaults to
  `audio_cap`.
- `S2A`, direct `A2S`, and text-to-text reasoning metrics are deferred. Keep
  those baseline manifests, but evaluate only the final-output metric suites
  that are currently defined in `configs/task_metric_map.json`.

## Modular Cascaded Baseline

`baseline_cascaded.py` is the modular entrypoint for stage-wise and cascaded
runs. It is separate from `baseline.py`, which remains the single-backend runner.
The cascaded runner supports individual pipelines (`T2A`, `T2S`, `A2T`, `S2T`,
`T2T`) and full cascades (`A2S`, `S2A`) with selectors for each stage:

```bash
python baseline_cascaded.py --pipeline S2A --asr reference --text-model template --t2a audiogen --num-samples 1 --dry-run
python baseline_cascaded.py --pipeline A2S --captioner reference --text-model template --tts audio_omni --num-samples 1 --dry-run
```

Backend selectors currently exposed by the modular runner:

- ASR: `audio_omni`, `reference`, `whisper`, `qwen3_asr`
- audio captioning: `audio_omni`, `reference`, `audio_flamingo_next`, `qwen3_omni_captioner`
- text-to-text: `template`, `identity`, `qwen3_32b`, `llama3_3_70b_instruct`
- TTS: `audio_omni`, `qwen3_omni`, `f5_tts`, `cosyvoice3`
- T2A: `audiogen`, `tangoflux`, `audio_omni`

Current adapter smoke status on TamIA:

- one-sample backend smoke coverage exists for all selected 32-combination
  backends: `whisper`, `qwen3_asr`, `audio_flamingo_next`,
  `qwen3_omni_captioner`, `qwen3_32b`, `llama3_3_70b_instruct`, `f5_tts`,
  `cosyvoice3`, `audiogen`, and `tangoflux`;
- Qwen3-ASR must use the upgraded py312 env:
  `$SALM_SCRATCH/.venvs/models/.venv-cascade-qwen3-asr-py312`; do not install
  `nagisa`, because it pulls a DyNet source build and can overload login nodes;
- `audio_flamingo_next` now uses native `MusicFlamingo*` Transformers classes
  in the upgraded caption environment; a local AudioFlamingo3 compatibility
  fallback remains only for older environments without MusicFlamingo support;
- `cosyvoice3` works for simple English TTS with CosyVoice's fallback frontend
  when launched with `cuda/12.6`, `arrow/17.0.0`, and `python/3.10.13`; full
  WeText/ttsfrd text normalization is still deferred.

Large text-to-text models, especially `llama3_3_70b_instruct`, should be smoked
as a whole-node single process using tensor parallel/device-map placement rather
than as one of four independent single-GPU lanes. A packed TamIA smoke has
verified that Llama 3.3 70B can occupy GPUs 1-3 while a light single-GPU lane
uses GPU 0, but do not pack it beside another large model.

Cascaded outputs use:

```text
results/baselines/cascaded/<pipeline>/<variant>/<dataset>/<split>/manifest.jsonl
results/baselines/cascaded/<pipeline>/<variant>/<dataset>/<split>/components/<task>/manifest.jsonl
```

The `<variant>` defaults to the selected stage backends and can be overridden
with `--run-name`. This keeps concurrent variants such as AudioGen S2A and
Audio-Omni S2A from writing the same files. Per-stage model path overrides are
available through `--asr-model-path`, `--captioner-model-path`,
`--text-model-path`, `--tts-model-path`, and `--t2a-model-path`; `--model-path`
remains only a fallback for simple single-backend runs.

Component manifests are intended for direct `evaluate.py --manifest ...` calls
when the component task is evaluable. In cascades, `components/text_to_text`
stores intermediate text prompts, while `components/T2A` and `components/T2S`
point at generated audio files.

The active validation experiment is defined in `configs/cascaded_baseline_variations.json`: `2` ASR choices x `2` captioners x `2` text models x `2` TTS models x `2` T2A models = `32` paired combinations on the `val` split (`2,277` rows). Run stage-wise and reuse shared stage outputs rather than rerunning the same ASR/caption/text stages for every final combination.

A one-node TamIA smoke script is available for current wired modular paths:

```bash
sbatch submission_scripts/run_tamia_cascaded_smokes_one_node.sbatch
```

It runs one-sample `S2A` and `A2S` cascades plus a `T2S` stage across a single
4-H100 node, with per-lane logs under
`$SALM_SCRATCH/logs/cascaded_smokes/<jobid>/`.

Additional adapter smoke scripts:

```bash
sbatch submission_scripts/run_tamia_cascaded_fixed_adapter_retry_one_node.sbatch
CAPTION_CUDA_VISIBLE_DEVICES=0,1,2,3 sbatch submission_scripts/run_tamia_cascaded_qwen3_caption_smoke_one_node.sbatch
ASR_CUDA_VISIBLE_DEVICES=0 sbatch submission_scripts/run_tamia_cascaded_qwen3_asr_smoke_one_node.sbatch
TEXT_MODEL=qwen3_32b TEXT_CUDA_VISIBLE_DEVICES=0 sbatch submission_scripts/run_tamia_cascaded_text_llm_smoke_one_node.sbatch
TEXT_MODEL=llama3_3_70b_instruct TEXT_CUDA_VISIBLE_DEVICES=0,1,2,3 sbatch submission_scripts/run_tamia_cascaded_text_llm_smoke_one_node.sbatch
sbatch submission_scripts/run_tamia_cascaded_packed_smokes_one_node.sbatch
```

`run_tamia_cascaded_packed_smokes_one_node.sbatch` is the intended one-node packing pattern for the current light modular smokes: Whisper `S2T` on GPU 0, F5 `T2S` on GPU 1, and Qwen3-32B `T2T` on GPU 2. Keep `llama3_3_70b_instruct` as a separate whole-node smoke rather than packing it beside other large models.
