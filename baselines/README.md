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

Backend non-negotiables:

- Declare supported tasks through `configs/model_capabilities.json`.
- Keep generated audio, logs, and result manifests out of git.
- Preserve progress logging with row index, sample id, elapsed time, ETA, and output path.
- Preserve resource logging for long or GPU-backed runs.
- Resolve dataset-relative paths against `configs/ttas_dataset.json`.
- Keep heavyweight or optional dependencies inside backend-specific load paths with clear errors.
- Do not add nested or broad imports unless they are required to keep optional stacks isolated.
