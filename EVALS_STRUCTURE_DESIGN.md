# Evals Structure Design

Status: planning document for evaluation result structure. This doc records
settled decisions for the eval refactor and open questions that still need
discussion. `RESULTS_STRUCTURE_DESIGN.md` remains the reference for the shared
results/logs/temporary roots, run IDs, and generic run-file conventions.

## Goals

- Keep evaluation results organized by the exact result run being evaluated.
- Use metric IDs directly in paths; avoid named metric-suite/group directories.
- Keep temporary backend checks and demos out of canonical results.
- Remove eval sharding, slicing-for-final, default metric bundles, common-ID
  filtering, and hash/fingerprint provenance from the target eval design.
- Store enough metric metadata to support scientific reporting and later website
  presentation without making metadata population a blocker for the refactor.

## Top-Level Placement

Canonical eval results live under:

```text
results/evals/
logs/evals/
```

Temporary eval checks live under:

```text
temporary/results/evals/
temporary/logs/evals/
```

The eval structure follows the same canonical-vs-temporary policy as inference:

- canonical eval: `"temporary": false`
- demo, smoke, backend check, fit test, or any partial/debug eval:
  `"temporary": true`

There is no separate `evaluation_mode` field. Backend smoke is not a metric and
not a mode; it is just a temporary eval run.

## Metric Selection Policy

There are no default or extended metric groups in the target design.

The user must explicitly choose metric IDs or request all available metrics.
If no metric is specified, the CLI should stop and ask for an explicit metric
choice, such as:

```text
Specify one or more metric names, or use "all".
```

This removes implicit default behavior and keeps metric selection visible in
the command and run config.

## Eval Hierarchy

Eval paths are organized around the result object being evaluated and the metric
being run.

### Stage Eval

The eval CLI uses seven explicit stage values: `a2t`, `s2t`, `t2t`, `t2a`,
`t2s`, `s2a`, and `a2s`. The eval path includes the end-to-end direction so
`t2t` runs are unambiguous.

```text
results/evals/stages/<direction>/<stage>/<evaluated_model>/<dataset>/<split>/<evaluated_run_id>/<metric_id>/<eval_run_id>/
logs/evals/stages/<direction>/<stage>/<evaluated_model>/<dataset>/<split>/<evaluated_run_id>/<metric_id>/<eval_run_id>/
```

Direction rules:

- `a2t`, `t2s`, and `a2s` imply `direction=a2s`.
- `s2t`, `t2a`, and `s2a` imply `direction=s2a`.
- `t2t` requires an explicit `--direction` of `a2s` or `s2a`.
- If `--direction` is supplied for an implied stage, it must match the implied
  direction.

Example:

```text
results/evals/stages/s2a/t2a/audiogen/ttas/val/20260616_1500/fad/20260616_1830/
results/evals/stages/a2s/t2s/f5_tts/ttas/val/20260616_1500/utmos/20260616_1842/
results/evals/stages/a2s/t2t/qwen3_32b/ttas/val/20260616_1500/g_eval_reference_free/20260616_1848/
```

There is no separate `pipelines/...` eval hierarchy in the current target
structure. Full-path `a2s` and `s2a` results are evaluated with the `a2s` and
`s2a` stage values under the same `stages/<direction>/<stage>/...` layout.

### Multiple Metrics

Even when a command requests multiple metrics, each metric gets its own metric
directory:

```text
.../<metric_id>/<eval_run_id>/
```

The orchestration layer may spawn separate metric runs from one command, but
the output structure stays metric-specific. There is no `metric_suite` or
group-name path layer.

## Eval Run Directory

Each metric eval run uses stable filenames:

```text
<eval_run_id>/
├── run.json
├── config.json
├── manifest.jsonl
├── artifacts/
├── inspect.json
├── errors.json
├── summary.json
└── report.md
```

- `run.json`: eval identity, lifecycle, target result reference, code state,
  Slurm identity, paths, and `temporary`.
- `config.json`: resolved metric config and backend/runtime settings used for
  this metric run.
- `manifest.jsonl`: compact per-sample metric results and per-sample metric
  errors when the metric is sample-level.
- `artifacts/`: backend inputs/outputs/stdout, raw judge outputs, and other
  metric-specific intermediate files worth preserving.
- `inspect.json`: richer human-readable view, especially for judge-based text
  metrics.
- `errors.json`: structured eval errors.
- `summary.json`: aggregate metric results, counts, and status.
- `report.md`: optional human-readable report.

Empty optional files or directories should not be created unless they are useful
for that metric.

## `run.json`

Stores eval identity and lifecycle metadata. It should stay light: full metric,
backend, and input-contract details belong in `config.json`.

Field order:

```text
schema_version
run_id
temporary
status
metric
target
dataset
split
started_at
completed_at
code
slurm
paths
```

Example:

```json
{
  "schema_version": 1,
  "run_id": "20260616_1830",
  "temporary": false,
  "status": "complete",
  "metric": {
    "metric_id": "fad",
    "level": "corpus",
    "reference_pairing_required": false
  },
  "target": {
    "kind": "stage",
    "category": "cascaded",
    "stage": "t2a",
    "direction": "s2a",
    "model": "audiogen",
    "dataset": "ttas",
    "split": "val",
    "run_id": "20260616_1500",
    "run_path": "results/cascaded/stages/t2a/audiogen/from-t2t/qwen32/ttas/val/20260616_1500",
    "manifest": "results/cascaded/stages/t2a/audiogen/from-t2t/qwen32/ttas/val/20260616_1500/manifest.jsonl",
    "summary": "results/cascaded/stages/t2a/audiogen/from-t2t/qwen32/ttas/val/20260616_1500/summary.json"
  },
  "dataset": "ttas",
  "split": "val",
  "started_at": "2026-06-16T18:30:12-04:00",
  "completed_at": "2026-06-16T18:47:55-04:00",
  "code": {
    "git_commit": "4c49181...",
    "git_dirty": false
  },
  "slurm": {
    "job_id": "321500",
    "array_task_id": null
  },
  "paths": {
    "result": "results/evals/stages/s2a/t2a/audiogen/ttas/val/20260616_1500/fad/20260616_1830",
    "logs": "logs/evals/stages/s2a/t2a/audiogen/ttas/val/20260616_1500/fad/20260616_1830"
  }
}
```

Settled decisions:

- Duplicate `dataset` and `split` at top level for quick filtering and symmetry
  with inference `run.json`.
- Keep `metric` minimal in `run.json`; detailed metric/backend/input metadata
  belongs in `config.json`.
- When one CLI invocation requests multiple metrics, each metric writes to its
  own metric directory, but all created metric runs share the same `eval_run_id`.

Example multi-metric invocation output:

```text
.../fad/20260616_1830/
.../kad/20260616_1830/
.../utmos/20260616_1830/
```

## `config.json`

Stores the exact resolved settings used for this metric run.

Field order:

```text
schema_version
input
metric
backend
execution
```

Example for `utmos`:

```json
{
  "schema_version": 1,
  "input": {
    "target_manifest": "results/cascaded/stages/t2s/f5_tts/from-t2t/qwen32/ttas/val/20260616_1500/manifest.jsonl",
    "target_run": "results/cascaded/stages/t2s/f5_tts/from-t2t/qwen32/ttas/val/20260616_1500",
    "dataset": "ttas",
    "split": "val",
    "start_index": null,
    "max_samples": null
  },
  "metric": {
    "metric_id": "utmos",
    "display_name": "UTMOS",
    "level": "sample",
    "reference_pairing_required": false,
    "higher_is_better": true,
    "input_contract": {
      "prediction_fields": ["generated_speech_file"],
      "reference_fields": [],
      "context_fields": []
    }
  },
  "backend": {
    "name": "utmosv2",
    "implementation": "utmosv2",
    "python": "{environment_root}/metrics/.venv-eval-utmosv2/bin/python",
    "model": {
      "name": "utmosv2",
      "path": "{model_root}/metrics/utmosv2"
    },
    "cache_dir": "{cache_root}/metrics/utmosv2",
    "device": "cuda",
    "parameters": {}
  },
  "execution": {
    "command": [
      "python",
      "evaluate.py",
      "--metric",
      "utmos"
    ],
    "num_samples": 2277
  }
}
```

Canonical full eval runs use `start_index: null` and `max_samples: null`.
Temporary smoke or backend-check runs may set integer values for those fields.
There is no `allow_partial` switch.

`metric.input_contract` is copied from the metric definition at run time so the
run records exactly which prediction, reference, and context fields the metric
consumed.

`backend.parameters` stores metric-specific settings without expanding the
shared top-level config schema.

Internal metrics use `backend.name: "metric_algorithm"` and do not invent model
or cache paths. Example:

```json
{
  "name": "metric_algorithm",
  "implementation": "evaluation.metrics.text.word_error_rate",
  "parameters": {}
}
```

## `manifest.jsonl`

Stores the compact machine-readable metric inputs and outputs, one row per
target sample considered by this metric run. The manifest should make it clear
which eval inputs came from the generated result and which came from the
dataset/reference side.

Field order:

```text
schema_version
row_index
sample_id
metric_id
status
input
output
error
```

Common structure:

```json
{
  "schema_version": 1,
  "row_index": 12,
  "sample_id": "ttas_123",
  "metric_id": "wer",
  "status": "success",
  "input": {
    "source": {},
    "generated": {},
    "reference": {}
  },
  "output": {},
  "error": null
}
```

- `input.source`: original sample or stage input used as context, such as the
  input speech/audio file.
- `input.generated`: generated artifacts, text, media, or intermediate outputs
  from the evaluated inference run that are passed into the metric.
- `input.reference`: dataset ground-truth artifacts or text passed into the
  metric.
- `output`: metric result, including scores, comments, counts, or
  metric-specific payloads.
- `error`: structured row-level metric error, or `null`.

This structure intentionally allows metric-specific keys inside `input` and
`output`. The common requirement is that generated material and dataset
reference material stay visibly separated.

Status values:

```text
success
included
skipped
error
```

- `success`: sample-level metric produced a row-level result.
- `included`: corpus-level metric included this row, but row-level score lives
  in `summary.json` or does not exist.
- `skipped`: row was not evaluated, usually because the target generation row
  was not successful.
- `error`: metric validation or execution failed for this row.

### S2T WER

S2T target metric for now is `wer`. `semantic_equivalence` is out of scope for
the target design.

```json
{
  "schema_version": 1,
  "row_index": 12,
  "sample_id": "ttas_123",
  "metric_id": "wer",
  "status": "success",
  "input": {
    "source": {
      "speech_file": "/project/.../speech.wav"
    },
    "generated": {
      "speech_transcription": "generated transcript here"
    },
    "reference": {
      "speech_transcription": "ground truth transcript here"
    }
  },
  "output": {
    "score": 0.18,
    "score_name": "wer",
    "higher_is_better": false,
    "details": {
      "substitutions": 2,
      "insertions": 1,
      "deletions": 0,
      "reference_words": 17,
      "normalization": "declared_normalizer_name"
    }
  },
  "error": null
}
```

### A2T Caption Metrics

A2T target metrics are:

```text
mace
spider
aac_clapsim
fense
```

`spider_max` is out of scope for the target design.

`spider` example:

```json
{
  "schema_version": 1,
  "row_index": 12,
  "sample_id": "ttas_123",
  "metric_id": "spider",
  "status": "included",
  "input": {
    "source": {
      "audio_file": "/project/.../audio.wav"
    },
    "generated": {
      "audio_caption": "generated caption here"
    },
    "reference": {
      "audio_caption": "ground truth caption here"
    }
  },
  "output": {
    "score": null,
    "score_name": "spider",
    "higher_is_better": true,
    "details": {
      "score_location": "summary.json"
    }
  },
  "error": null
}
```

`aac_clapsim` example:

```json
{
  "schema_version": 1,
  "row_index": 12,
  "sample_id": "ttas_123",
  "metric_id": "aac_clapsim",
  "status": "success",
  "input": {
    "source": {
      "audio_file": "/project/.../audio.wav"
    },
    "generated": {
      "audio_caption": "generated caption here"
    },
    "reference": {
      "audio_file": "/project/.../audio.wav"
    }
  },
  "output": {
    "score": 0.42,
    "score_name": "aac_clapsim",
    "higher_is_better": true,
    "details": {}
  },
  "error": null
}
```

### T2A/T2S Input Shape

For T2A, T2S, A2S, and S2A metrics, the same `input` structure captures metrics
that consume combinations of generated media, generated text, and dataset
references.

Prompt-adherence CLAP example:

```json
{
  "schema_version": 1,
  "row_index": 12,
  "sample_id": "ttas_123",
  "metric_id": "prompt_adherence_clap_score",
  "status": "success",
  "input": {
    "source": {
      "text_source_stage": "t2t"
    },
    "generated": {
      "audio_file": "results/cascaded/stages/t2a/.../outputs/000012_ttas_123.wav",
      "generated_audio_caption": "a dog barking near traffic"
    },
    "reference": {}
  },
  "output": {
    "score": 0.39,
    "score_name": "prompt_adherence_clap_score",
    "higher_is_better": true,
    "details": {}
  },
  "error": null
}
```

Target-caption CLAP example:

```json
{
  "schema_version": 1,
  "row_index": 12,
  "sample_id": "ttas_123",
  "metric_id": "target_caption_clap_score",
  "status": "success",
  "input": {
    "source": {
      "text_source_stage": "t2t"
    },
    "generated": {
      "audio_file": "results/cascaded/stages/t2a/.../outputs/000012_ttas_123.wav"
    },
    "reference": {
      "audio_caption": "ground truth caption here"
    }
  },
  "output": {
    "score": 0.43,
    "score_name": "target_caption_clap_score",
    "higher_is_better": true,
    "details": {}
  },
  "error": null
}
```

Metric-specific verbose context, such as full prompts, system prompts, raw
backend payloads, tokenized variants, and complete backend JSON, belongs in
`inspect.json` or `artifacts/` unless it is directly consumed by the metric.

## `summary.json`

Stores aggregate results for one metric run. It must not contain per-sample
inputs or full backend payloads.

Field order:

```text
schema_version
run_id
status
metric
target
samples
scores
errors
artifacts
timing
```

Sample-level metric example:

```json
{
  "schema_version": 1,
  "run_id": "20260616_1842",
  "status": "complete",
  "metric": {
    "metric_id": "utmos",
    "level": "sample",
    "higher_is_better": true
  },
  "target": {
    "kind": "stage",
    "stage": "t2s",
    "run_id": "20260616_1500"
  },
  "samples": {
    "total": 2277,
    "successful": 2269,
    "included": 2269,
    "skipped": 5,
    "failed": 3
  },
  "scores": {
    "primary": {
      "name": "utmos",
      "value": 3.84,
      "aggregation": "mean",
      "higher_is_better": true
    },
    "statistics": {
      "mean": 3.84,
      "median": 3.91,
      "std": 0.44,
      "min": 1.92,
      "max": 4.71
    }
  },
  "errors": {
    "count": 3,
    "by_type": {
      "MetricInputError": 2,
      "BackendRuntimeError": 1
    }
  },
  "artifacts": {
    "manifest": "manifest.jsonl",
    "errors": "errors.json",
    "inspect": "inspect.json",
    "backend_outputs": "artifacts/"
  },
  "timing": {
    "total_seconds": 812.4,
    "backend_seconds": 790.1,
    "mean_seconds_per_sample": 0.35
  }
}
```

Corpus-level metric example:

```json
{
  "schema_version": 1,
  "run_id": "20260616_1830",
  "status": "complete",
  "metric": {
    "metric_id": "fad",
    "level": "corpus",
    "higher_is_better": false
  },
  "target": {
    "kind": "stage",
    "stage": "t2a",
    "run_id": "20260616_1500"
  },
  "samples": {
    "total": 2277,
    "successful": null,
    "included": 2269,
    "skipped": 5,
    "failed": 3,
    "successful_note": "not_applicable_for_corpus_metric"
  },
  "scores": {
    "primary": {
      "name": "fad",
      "value": 4.82,
      "aggregation": "corpus",
      "higher_is_better": false
    },
    "statistics": null
  },
  "errors": {
    "count": 3,
    "by_type": {
      "MissingGeneratedAudio": 2,
      "InvalidReferenceAudio": 1
    }
  },
  "artifacts": {
    "manifest": "manifest.jsonl",
    "errors": "errors.json",
    "inspect": "inspect.json",
    "backend_outputs": "artifacts/"
  },
  "timing": {
    "total_seconds": 1204.7,
    "backend_seconds": 1188.3,
    "mean_seconds_per_sample": null
  }
}
```

For corpus-level metrics, `samples.successful` is `null` because row-level
success is not applicable. Participating rows are counted under
`samples.included`. Implementations should assert that corpus metrics do not
emit row-level `success` rows in `manifest.jsonl`; included corpus participants
use `status: "included"`.

`timing.total_seconds` is wall-clock runtime for the eval run, including setup,
input validation, backend work, and artifact writing. `timing.backend_seconds`
is the time spent inside the metric backend or metric algorithm when that is
easy to measure. If backend timing is not measured, set it to `null` rather than
estimating it.

## `errors.json`

Stores a compact structured index of non-success rows for quick debugging. It is
derived from `manifest.jsonl` and should not duplicate full backend logs or raw
backend payloads.

Skipped rows are included because they did not receive a metric result. However,
`error_count` counts only rows with `status: "error"`; skipped rows are counted
separately in `skipped_count`.

Field order:

```text
schema_version
run_id
metric_id
error_count
skipped_count
errors
```

Example with a metric error:

```json
{
  "schema_version": 1,
  "run_id": "20260616_1842",
  "metric_id": "utmos",
  "error_count": 2,
  "skipped_count": 5,
  "errors": [
    {
      "row_index": 12,
      "sample_id": "ttas_123",
      "status": "error",
      "error_type": "MetricInputError",
      "message": "generated speech file does not exist",
      "input": {
        "source": {
          "speech_file": "/project/.../speech.wav"
        },
        "generated": {
          "speech_file": "results/cascaded/stages/t2s/.../outputs/000012_ttas_123.wav"
        },
        "reference": {}
      },
      "log_ref": "logs/evals/stages/t2s/f5_tts/ttas/val/20260616_1500/utmos/20260616_1842/run.log",
      "artifact_ref": null
    }
  ]
}
```

Skipped row example:

```json
{
  "row_index": 13,
  "sample_id": "ttas_124",
  "status": "skipped",
  "error_type": "TargetRowNotSuccessful",
  "message": "target generation row status was error",
  "input": {
    "source": {},
    "generated": {},
    "reference": {}
  },
  "log_ref": null,
  "artifact_ref": null
}
```

Minimal empty example:

```json
{
  "schema_version": 1,
  "run_id": "20260616_1842",
  "metric_id": "utmos",
  "error_count": 0,
  "skipped_count": 0,
  "errors": []
}
```

## `inspect.json`

Stores the human-readable/debug view for one metric run. It may be heavier than
`manifest.jsonl`, but it should still be organized and intentional rather than
an uncontrolled dump.

Field order:

```text
schema_version
run_id
metric
target
config_summary
summary
sample_policy
samples
backend_artifacts
```

Rules:

- Sample-level metrics include all rows in `samples` because row-level debugging
  is the purpose of `inspect.json`.
- Corpus-level metrics do not include sample-level `samples`, because there are
  no sample-level metric outputs. Full corpus participation is already recorded
  in `manifest.jsonl`.
- Hyperparameters, prompts, and rendered runtime prompts are included when the
  eval metric uses them.
- Intermediate artifacts produced by the metric, such as ASR transcripts used
  before WER computation, belong in `intermediate`.

Sample-level policy example:

```json
"sample_policy": {
  "mode": "all",
  "count": 2277
}
```

Corpus-level policy example:

```json
"sample_policy": {
  "mode": "omitted",
  "count": 0,
  "reason": "corpus_metric_has_no_sample_level_outputs; manifest.jsonl records included rows"
}
```

Run-level hyperparameters and prompts belong in `config_summary` when
applicable:

```json
"config_summary": {
  "backend": {
    "name": "openai-whisper",
    "implementation": "whisper_large_v3"
  },
  "input_contract": {
    "prediction_fields": ["speech_file"],
    "reference_fields": ["speech_transcription"],
    "context_fields": []
  },
  "parameters": {
    "sample_rate": 16000,
    "batch_size": 8,
    "temperature": 0.0
  },
  "prompts": {
    "system": null,
    "user_template": null
  }
}
```

If prompts are rendered per sample at runtime and consumed by the eval metric,
include them inside the sample:

```json
"runtime_prompt": {
  "system": "...",
  "user": "..."
}
```

Only include prompts used by the eval metric itself. Prompts used by the
original generation run belong in the target run's own `inspect.json` unless the
eval metric directly consumes them.

`asr_wer` sample example:

```json
{
  "row_index": 12,
  "sample_id": "ttas_123",
  "status": "success",
  "input": {
    "source": {
      "text_source_stage": "t2t"
    },
    "generated": {
      "speech_file": "results/cascaded/stages/t2s/.../outputs/000012_ttas_123.wav"
    },
    "reference": {
      "speech_transcription": "ground truth transcript here"
    }
  },
  "intermediate": {
    "generated_speech_asr": {
      "backend": "openai-whisper",
      "model": "large-v3",
      "transcription": "asr transcript of generated speech",
      "language": "en",
      "parameters": {
        "sample_rate": 16000,
        "resampling_policy": "whisper_internal"
      }
    }
  },
  "normalization": {
    "generated_normalized": "asr transcript of generated speech",
    "reference_normalized": "ground truth transcript here",
    "normalizer": "declared_normalizer_name"
  },
  "output": {
    "score": 0.18,
    "score_name": "asr_wer",
    "higher_is_better": false,
    "details": {
      "substitutions": 2,
      "insertions": 1,
      "deletions": 0,
      "reference_words": 17
    }
  },
  "error": null
}
```

`dwer` sample example:

```json
{
  "row_index": 12,
  "sample_id": "ttas_123",
  "status": "success",
  "input": {
    "source": {
      "text_source_stage": "t2t"
    },
    "generated": {
      "speech_file": "results/cascaded/stages/t2s/.../outputs/000012_ttas_123.wav"
    },
    "reference": {
      "speech_file": "/project/.../reference_speech.wav"
    }
  },
  "intermediate": {
    "generated_speech_asr": {
      "backend": "openai-whisper",
      "model": "large-v3",
      "transcription": "asr transcript of generated speech"
    },
    "reference_speech_asr": {
      "backend": "openai-whisper",
      "model": "large-v3",
      "transcription": "asr transcript of reference speech"
    }
  },
  "normalization": {
    "generated_normalized": "asr transcript of generated speech",
    "reference_normalized": "asr transcript of reference speech",
    "normalizer": "declared_normalizer_name"
  },
  "output": {
    "score": 0.22,
    "score_name": "dwer",
    "higher_is_better": false,
    "details": {
      "substitutions": 3,
      "insertions": 1,
      "deletions": 0,
      "reference_words": 18
    }
  },
  "error": null
}
```

`backend_artifacts` points to raw backend files under `artifacts/` when they
exist:

```json
"backend_artifacts": {
  "input": "artifacts/backend_input.json",
  "output": "artifacts/backend_output.json",
  "stdout": "artifacts/stdout.log",
  "stderr": null,
  "directory": "artifacts/"
}
```

## `report.md`

`report.md` is optional and non-authoritative. It may present a concise human
summary of the metric run for quick reading, but it does not replace
`summary.json`, `errors.json`, or `manifest.jsonl`.

Suggested content:

- run identity and metric name
- target result reference
- primary score and key counts
- high-level notes about failures or corpus exclusions
- links or paths to the canonical artifacts

## Target Reference

Each eval run records the exact result run it evaluates. The result run ID and
paths are the provenance source of truth. Hashes and fingerprints are not part
of the target design.

Stage target example:

```json
{
  "kind": "stage",
  "category": "cascaded",
  "stage": "t2a",
  "direction": "s2a",
  "model": "audiogen",
  "dataset": "ttas",
  "split": "val",
  "run_id": "20260616_1500",
  "run_path": "results/cascaded/stages/t2a/audiogen/from-t2t/qwen32/ttas/val/20260616_1500",
  "manifest": "results/cascaded/stages/t2a/audiogen/from-t2t/qwen32/ttas/val/20260616_1500/manifest.jsonl",
  "summary": "results/cascaded/stages/t2a/audiogen/from-t2t/qwen32/ttas/val/20260616_1500/summary.json"
}
```

Full-direction target example:

```json
{
  "kind": "stage",
  "category": "cascaded",
  "stage": "a2s",
  "direction": "a2s",
  "model": "afn_qwen32_f5tts",
  "dataset": "ttas",
  "split": "val",
  "run_id": "20260616_1530",
  "run_path": "results/cascaded/pipelines/a2s/afn_qwen32_f5tts/ttas/val/20260616_1530",
  "manifest": "results/cascaded/pipelines/a2s/afn_qwen32_f5tts/ttas/val/20260616_1530/manifest.jsonl",
  "pipeline": "results/cascaded/pipelines/a2s/afn_qwen32_f5tts/ttas/val/20260616_1530/pipeline.json"
}
```

## Metric Input Contract

Each metric definition must declare the inputs it consumes. The eval runner
should route fields from the target manifest and dataset references using this
contract rather than guessing from the metric name.

Metrics may consume one or more of:

- generated speech/audio media from the evaluated run
- generated text from the evaluated run
- the generation prompt or intermediate generated text that conditioned the
  media output
- reference media from the dataset
- reference text from the dataset, such as ground-truth transcription or audio
  caption

Examples:

- `utmos`: generated speech only.
- `dnsmos`: generated speech only.
- `wer`: generated speech or generated transcription plus reference
  transcription, depending on the metric variant.
- `fad`: generated audio set plus reference audio set.
- `prompt_adherence_clap_score`: generated audio plus the generated or
  persisted prompt text used to condition that audio.
- `target_caption_clap_score`: generated audio plus dataset reference caption.
- `g_eval_*`: generated text plus source/generated context and, for grounded
  variants, dataset reference text.

The metric input contract should be recorded in metric metadata and copied or
referenced by `config.json` for each eval run.

## Common Sample IDs

Common sample-ID filtering is removed from the target design.

Eval runs report simple counts:

- total rows considered
- successful rows
- failed rows
- skipped rows, if applicable

If one model has more failures than another, the difference will be handled by
manual inspection and reporting rather than by forcing all models onto a common
successful-ID intersection.

## No Hashes Or Fingerprints

The eval structure does not include manifest hashes, config hashes, selected-row
hashes, common-ID hashes, or fingerprint blocks.

Run IDs, immutable run directories, target references, and resolved configs are
enough for this academic workflow.

## No Eval Sharding

Eval sharding is removed from the target design.

- No canonical eval shards.
- No eval shard merge scripts.
- No eval `merge_report.json`.
- No sharding fields in eval config.
- No parallel-eval flexibility in the metric runner.

Temporary/debug evals may still evaluate a small subset for quick backend
checks, but they are written under `temporary/` and are not canonical results.

## Metric Metadata

Metric metadata is important for later reporting and website presentation, but
complete population is not required to start the structure refactor.

The target design should keep metric definitions separate from individual eval
runs, with each run copying or referencing the relevant metric metadata.

Useful fields:

```text
metric_id
display_name
description
level
reference_pairing_required
higher_is_better
paper_title
paper_venue
paper_year
paper_url
implementation
constraints
known_limitations
```

`level` records where the metric is computed:

- `sample`: computed independently for each sample, then summarized.
- `corpus`: computed once over the evaluated set.

`reference_pairing_required` records whether the metric needs each prediction to be paired with its corresponding dataset reference.
Reference pairing is separate from level: a metric can use paired prediction/reference inputs but still produce one corpus-level score if it depends on corpus statistics.

Examples:

- `wer`: `level="sample"`, `reference_pairing_required=true`
- `utmos`: `level="sample"`, `reference_pairing_required=false`
- `dnsmos`: `level="sample"`, `reference_pairing_required=false`
- `spider`: `level="corpus"`, `reference_pairing_required=true`
- `fad`: `level="corpus"`, `reference_pairing_required=false`
- `inception_score`: `level="corpus"`, `reference_pairing_required=false`

## Metric Definitions Config

Metric definitions live in:

```text
configs/eval_metric_definitions.json
```

This file stores scientific and display metadata for metrics. It is separate
from `configs/eval_metrics.json`, which stores backend/runtime configuration
such as Python environments, cache directories, model paths, batch sizes, sample
rates, and backend-specific parameters.

Example shape:

```json
{
  "schema_version": 1,
  "metrics": {
    "wer": {
      "metric_id": "wer",
      "display_name": "Word Error Rate",
      "description": "Word-level transcription error between generated and reference transcript.",
      "level": "sample",
      "reference_pairing_required": true,
      "higher_is_better": false,
      "paper": {
        "title": null,
        "venue": null,
        "year": null,
        "url": null
      },
      "implementation": "evaluation.metrics.text.word_error_rate",
      "input_contract": {
        "prediction_fields": ["speech_transcription"],
        "reference_fields": ["speech_transcription"],
        "context_fields": ["speech_file"]
      },
      "constraints": [],
      "known_limitations": []
    }
  }
}
```

Policy decisions:

- `metric_id` must match the path metric ID used under
  `.../<metric_id>/<eval_run_id>/`.
- The definitions file may be partially populated at first. Missing paper
  metadata can use `null` values.
- Each eval run's `config.json` copies the relevant metric definition snapshot
  at run time so old runs remain understandable if the definitions file changes
  later.
- `configs/eval_metric_definitions.json` must not contain backend paths, cache
  paths, environment paths, or batch sizes; those stay in
  `configs/eval_metrics.json`.
- Removed metrics such as `semantic_equivalence` and `spider_max` should not be
  added to the target definitions file.

## Status

The eval structure design is settled at the schema level. Remaining work is
implementation and code wiring, not further path or schema design.
