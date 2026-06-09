# TODO

## Immediate Priorities

- Re-run the cascaded final-generation stages end to end from the cleaned canonical `T2T` manifests only.
  - Treat the current `T2S` and `T2A` outputs as stale for approval purposes.
  - Keep only canonical merged outputs after validation; delete shard directories after successful merge.
- Finish the evaluation pipeline end to end on the finalized cascaded outputs.
  - Run the already-wired metrics on final `T2S` and `T2A` manifests.
  - Verify metric outputs, summaries, and resumability before long full-corpus runs.
- Keep the result tree clean after each merge.
  - Canonical lane directories should retain only the final manifests and minimal run metadata.
  - Remove shard artifacts such as `*_schema_*_s*`, `*_corrected_*_s*`, temporary `text/`, `components/`, and empty `audio/` directories once the canonical lane is validated.

## Cascaded Manifest Cleanup

- Fix carried stage metadata so already-completed upstream stages preserve real `checkpoint`, `dtype`, `prompt_template`, `generation`, and `runtime.model_load_seconds` values instead of the current legacy null placeholders.
  - Current gap is in `scripts/build_cascaded_stage_inputs.py` fallback `_legacy_stage_definition(...)`.
- Make stage object field ordering consistent across all stages.
  - `stage` and `task` should appear at the top of every stage object, including carried Stage 1 records.
- Revisit the meaning and usage of `final_output` and `error` in intermediate-stage manifests.
  - `final_output` is intentionally `null` for `T2T` because the final artifact is text, not generated audio.
  - `error` is intentionally present for schema consistency and for partial-failure runs, but the all-success manifests naturally show `null`.
- Confirm whether the current `manifest.inspect.json` should keep `run_config.json`-level metadata separate from per-sample payloads, and avoid repeating invariant model config inside every sample.

## Prompting And Quality Tuning

- Tighten the Stage 2 text-to-text system prompt.
  - Current prompt is serviceable but still too permissive for some caption-to-speech and speech-to-audio-description outputs.
- Tighten captioning prompts for both Stage 1 captioners.
  - `qwen3_omni_captioner`: too descriptive; needs a stricter brevity constraint and less unnecessary detail.
  - `audio_flamingo_next`: prompt can also be improved even though it is already shorter than Qwen.
- Record and review the exact Stage 1 model input through the `messages` array in carried downstream manifests for every captioner.
- Revisit token limits after the end-to-end pipeline is stable.
  - Current captioner generation limits are `128` for Qwen3-Omni Captioner and `256` for Audio Flamingo Next.
  - Current `T2T` max token budget is derived from `--steps`; with the usual `--steps 64`, the actual `max_new_tokens` is `64`.
  - Decide later whether `64` is too restrictive for `T2T` after looking at finalized outputs and metric behavior.

## Metric Follow-Up

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
