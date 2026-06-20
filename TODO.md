# TODO

## Cascaded Manifest Cleanup

- Make downstream cascaded stages consume the upstream stage `manifest.jsonl` directly.
  - The machine-readable `manifest.jsonl` is the contract between stages; `inspect.json` is for human inspection.
  - Remove the current need to convert upstream outputs into derived `stage_inputs/schema_v1/*.jsonl` adapter files before T2T or final stages.
  - Add CLI/config support for selecting the upstream manifest/run directly, then extracting successful rows and required text/media fields internally.
  - Preserve lineage, sample ID checks, and error-row handling without copying full upstream records into every row.
- Fix carried stage metadata so already-completed upstream stages preserve real `checkpoint`, `dtype`, `prompt_template`, `generation`, and `runtime.model_load_seconds` values instead of the current legacy null placeholders.
  - Current gap is in `scripts/build_cascaded_stage_inputs.py` fallback `_legacy_stage_definition(...)`.
- Make stage object field ordering consistent across all stages.
  - `stage` and `task` should appear at the top of every stage object, including carried Stage 1 records.
- Revisit the meaning and usage of `final_output` and `error` in intermediate-stage manifests.
  - `final_output` is intentionally `null` for `T2T` because the final artifact is text, not generated audio.
  - `error` is intentionally present for schema consistency and for partial-failure runs, but the all-success manifests naturally show `null`.
- Confirm whether the current `manifest.inspect.json` should keep `run_config.json`-level metadata separate from per-sample payloads, and avoid repeating invariant model config inside every sample.

## Prompting And Quality Tuning

- Revisit token limits after the end-to-end pipeline is stable.
  - Current captioner generation limits are `128` for Qwen3-Omni Captioner and `256` for Audio Flamingo Next.
  - Current `T2T` max token budget is derived from `--steps`; with the usual `--steps 64`, the actual `max_new_tokens` is `64`.
  - Decide later whether `64` is too restrictive for `T2T` after looking at finalized outputs and metric behavior.

## Metric Follow-Up

- Complete a larger representative Inception Score smoke after the official backend returned non-finite mean/std for two samples. MACE, AAC-CLAPSim, FENSE, FAD, and KAD smoke tests passed; then prepare full-corpus final submissions.
- **Deferred comparison sanity step:** after all model results exist, form the union of per-metric failure sample IDs across compared models and recompute comparisons on the same retained sample set. Do not apply this during individual metric execution.
- **High priority, deferred:** replace Prometheus `non_contradiction` criterion calls in G-Eval v2 with the existing NLI contradiction result once NLI outputs are available before score aggregation.
  - Normalize deterministically onto the G-Eval scale with `non_contradiction_score = 1 + 4 * (1 - contradiction_probability)`.
  - Preserve the raw contradiction probability, NLI model/version, and normalization-policy version in criterion metadata.
  - Reuse the same NLI-derived score for plausibility and reference-grounded variants; references must not affect contradiction.
  - Until this dependency is implemented, keep G-Eval runnable independently rather than blocking it on NLI completion.
- Keep the shared metric environment as the main path for wired metrics.
- Build and validate the separate Prometheus/vLLM environment for `g_eval_reference_free`.
- Extend judge-based evaluation beyond the current text-only LLM-as-judge baseline.
  - Add an audio-language-model-as-judge (`LALM`-as-judge) approach that can inspect generated or reference audio directly.
  - Investigate an additional structured/SQLLM-style judge approach after defining its intended inputs, outputs, and role.
  - Compare judge disagreements as diagnostics; keep NLI, text-only G-Eval, and future audio-aware judges independent rather than feeding one metric's score into another.
- Decide the final metric set for:
  - `T2T` intermediate evaluation
  - `A2S` final speech evaluation
  - `S2A` final audio evaluation
- Add resumability plus explicit overwrite controls for long evaluation runs where still missing.

## Deferred Upstream Issues

- Transformers Qwen3-Omni talker generation bug: consider opening an upstream PR after confirming the issue still exists on the latest Transformers source.
  - Local environment: `.venv-qwen3-omni`, `transformers==5.3.0`.
  - Error seen during `return_audio=True` generation:
    `GenerationMixin.prepare_inputs_for_generation() got multiple values for argument 'next_sequence_length'`.
  - Local workaround applied in the venv file:
    `.venv-qwen3-omni/lib/python3.10/site-packages/transformers/models/qwen3_omni_moe/modeling_qwen3_omni_moe.py`.
  - Suspected fix: in the Qwen3-Omni talker `prepare_inputs_for_generation`, call `super().prepare_inputs_for_generation(...)` with keyword arguments instead of the old positional argument order.
