# TODO

## Deferred Upstream Issues

- Transformers Qwen3-Omni talker generation bug: consider opening an upstream PR after confirming the issue still exists on the latest Transformers source.
  - Local environment: `.venv-qwen3-omni`, `transformers==5.3.0`.
  - Error seen during `return_audio=True` generation:
    `GenerationMixin.prepare_inputs_for_generation() got multiple values for argument 'next_sequence_length'`.
  - Local workaround applied in the venv file:
    `.venv-qwen3-omni/lib/python3.10/site-packages/transformers/models/qwen3_omni_moe/modeling_qwen3_omni_moe.py`.
  - Suspected fix: in the Qwen3-Omni talker `prepare_inputs_for_generation`, call `super().prepare_inputs_for_generation(...)` with keyword arguments instead of old positional argument order.

## Next Session

- Request compute allocations and run `baseline.py` for the full TTAS test split for:
  `audiogen`,
  `tangoflux`,
  `qwen3_omni`,
  `audio_omni`.
- Start with `--num-samples 2` per model in the correct environment, then run `--num-samples -1` after checking the first outputs and manifests.
- Inspect `results/baselines/<model>/test/speech_text/manifest.jsonl` and generated WAVs for each model.
- Inspect `logs/baselines/<model>/test/speech_text/resources_*.jsonl` to decide whether future jobs can request fewer CPUs.
- Decide whether to run `speech_text`, `audio_cap`, or both prompt fields for final baseline comparison.
