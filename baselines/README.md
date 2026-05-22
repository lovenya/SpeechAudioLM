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
  Run it with `--prompt-field speech_text`.
- `S2A`, direct `A2S`, and text-to-text reasoning metrics are deferred. Keep
  those baseline manifests, but evaluate only the final-output metric suites
  that are currently defined in `configs/task_metric_map.json`.
