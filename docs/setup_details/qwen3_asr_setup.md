# Qwen3-ASR Compatibility Shims

Status: historical debugging and migration note. Keep this file because it
records compatibility failures and temporary shims that may matter on another
cluster. Do not treat it as the current active Qwen3-ASR setup without checking
`configs/runtime_environments.json`, `configs/project_paths.json`, and recent
smoke evidence.

This file records temporary runtime shims used to make `Qwen/Qwen3-ASR-1.7B`
work in the TamIA cascaded baseline environment. These are not intended to be a
long-term substitute for a known-good `qwen_asr` / `transformers` version pair.

## Context

Environment path:

```text
/scratch/l/lovenya/salm_migration_clean/.venvs/models/.venv-cascade-asr
```

Model path:

```text
/scratch/l/lovenya/salm_migration_clean/models/cascaded/asr/qwen3-asr-1.7b
```

Adapter path:

```text
baselines/models/speech_asr.py
```

The shims are scoped to `Qwen3ASRRunner.load()` through
`_patch_qwen_asr_transformers_compat()` and run only before importing/loading
`qwen_asr` for the `qwen3_asr` backend.

## Shims

1. `transformers.utils.generic.check_model_inputs`

   Failure seen before model construction:

   ```text
   ImportError: cannot import name 'check_model_inputs' from transformers.utils.generic
   ```

   Temporary behavior: injects a no-op decorator named `check_model_inputs`.

2. `nagisa` import stub

   Failure path: the `qwen_asr` package import reaches forced-aligner code that
   imports `nagisa`, even though plain `S2T` inference does not use the forced
   aligner.

   Temporary behavior: injects a minimal `nagisa` module with `tagging(text)`
   returning an object with `.words`.

3. `transformers.modeling_rope_utils.ROPE_INIT_FUNCTIONS["default"]`

   Failure from job `309391`:

   ```text
   KeyError: 'default'
   ```

   Temporary behavior: registers a local default RoPE initializer compatible
   with the model config fields.

4. `Qwen3ASRThinkerConfig.pad_token_id`

   Failure from job `309456`:

   ```text
   AttributeError: 'Qwen3ASRThinkerConfig' object has no attribute 'pad_token_id'
   ```

   Temporary behavior: patches `Qwen3ASRThinkerConfig.__init__` so missing
   `pad_token_id` is set to `None`, matching the model code's own fallback:
   `self.config.pad_token_id if self.config.pad_token_id is not None else -1`.

5. `Qwen3ASRThinkerTextRotaryEmbedding.compute_default_rope_parameters`

   Failure from packed job `309461` after weights loaded:

   ```text
   AttributeError: 'Qwen3ASRThinkerTextRotaryEmbedding' object has no attribute 'compute_default_rope_parameters'
   ```

   Temporary behavior: adds a static method pointing to the same local default
   RoPE initializer used for `ROPE_INIT_FUNCTIONS["default"]`.

## Latest Failure

Job `309471` was submitted after shim item 5. It advanced through weight
loading and then failed while `qwen_asr` called
`AutoProcessor.from_pretrained(..., fix_mistral_regex=True)`:

```text
TypeError: transformers.tokenization_utils_tokenizers.TokenizersBackend._patch_mistral_regex() got multiple values for keyword argument 'fix_mistral_regex'
```

The error path then raised a protobuf import error from Transformers'
tokenizer fallback handling:

```text
ImportError: requires the protobuf library but it was not found in your environment.
```

## Current Status

As of job `309471`, Qwen3-ASR is deferred. The backend has now hit multiple
unrelated `qwen_asr` / `transformers` API compatibility failures after local
compatibility shims. Use the verified Whisper ASR backend for active modular
`S2T` work.

Job `309461` also verified that Llama 3.3 70B `T2T` passed in the packed
ASR/Llama smoke while Qwen3-ASR failed independently.

## Replacement Plan

Next cleanup path should be one of:

- identify and pin a `qwen_asr` version compatible with the installed
  `transformers` version;
- identify and pin a `transformers` version compatible with the installed
  `qwen_asr` package;
- vendor/fork a small upstream patch if the official package has not caught up.

Do not add more runtime shims for this backend before next week's debugging
pass. The next useful path is dependency-version alignment or a small upstream
patch, then removing as many of the temporary shims above as possible.
