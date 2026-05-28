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
The cascaded runner supports individual pipelines (`T2A`, `T2S`, `A2T`, `S2T`)
and full cascades (`A2S`, `S2A`) with selectors for each stage:

```bash
python baseline_cascaded.py --pipeline S2A --asr reference --text-model template --t2a audiogen --num-samples 1 --dry-run
python baseline_cascaded.py --pipeline A2S --captioner reference --text-model template --tts audio_omni --num-samples 1 --dry-run
```

Current wired backends are deliberately conservative:

- ASR: `audio_omni`, `reference`
- audio captioning: `audio_omni`, `reference`
- text-to-text: `template`, `identity`
- TTS: `audio_omni`, `qwen3_omni`
- T2A: `audiogen`, `tangoflux`, `audio_omni`

The CLI also names target future backends from the project goal, but they raise
clear unsupported-backend errors until their model assets, environments, and
adapters are added: `whisper`, `qwen3_asr`, `audio_flamingo_next`,
`qwen3_omni_captioner`, `qwen3_32b`, `llama3_3_70b_instruct`, `f5_tts`, and
`cosyvoice3`.

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
