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
- TTS: `audio_omni`, `qwen3_omni`, `f5_tts`
- T2A: `audiogen`, `tangoflux`, `audio_omni`

Current adapter smoke status on TamIA:

- verified by one-sample compute smoke: `whisper` for `S2T`, `f5_tts` for `T2S`, `qwen3_32b` for `T2T`, and `qwen3_omni_captioner` for `A2T`;
- retry-ready after repo fixes: `qwen3_asr` now bypasses the installed Transformers `check_model_inputs` mismatch, the unused forced-aligner `nagisa` import path, and the missing `default` RoPE initializer in the installed Transformers build for plain `S2T`;
- deferred: `audio_flamingo_next` needs a Transformers build with `musicflamingo` support; `cosyvoice3` is not exposed until its frontend dependency stack is resolved.

Large text-to-text models, especially `llama3_3_70b_instruct`, should be smoked
as a whole-node single process using tensor parallel/device-map placement rather
than as one of four independent single-GPU lanes.

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
sbatch submission_scripts/run_tamia_cascaded_packed_smokes_one_node.sbatch
```

`run_tamia_cascaded_packed_smokes_one_node.sbatch` is the intended one-node packing pattern for the current light modular smokes: Whisper `S2T` on GPU 0, F5 `T2S` on GPU 1, and Qwen3-32B `T2T` on GPU 2. Keep `llama3_3_70b_instruct` as a separate whole-node smoke rather than packing it beside other large models.
