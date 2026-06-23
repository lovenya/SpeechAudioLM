# Results Structure Design

Status: working design for where inference and evaluation results live. The
current priority is a simple, durable contract that future agents can follow
without rediscovering old demo layouts.

## Goals

- Keep official results separate from temporary runs.
- Prevent demonstrations, smoke tests, caches, and incomplete experiments from bloating or confusing canonical results.
- Keep execution logs organized without mixing them with scientific outputs.
- Centralize configurable path roots and avoid hard-coded absolute paths.
- Preserve enough metadata to identify how each run was produced.

## Runtime Root

The current machine-specific runtime root is:

```text
/scratch/l/lovenya/salm_migration_clean
```

This name is a machine/local migration name, not a scientific concept. Docs and
scripts should refer to named roots such as `results_root`, `logs_root`,
`model_root`, `cache_root`, and `datasets_root` wherever possible so the project
can move cleanly to another cluster.

Target canonical results root:

```text
/scratch/l/lovenya/salm_migration_clean/results
```

Target canonical logs root:

```text
/scratch/l/lovenya/salm_migration_clean/logs
```

The repo-local `results/` and `logs/` directories should be treated as legacy or
local-development locations unless a script is explicitly configured to use
them. New substantial inference and eval outputs should go under the external
runtime root, not into Git working-tree storage.

## Top-Level Structure

```text
{results_root}/
├── cascaded/
└── evals/

{logs_root}/
├── cascaded/
└── evals/

{temporary_results_root}/
├── cascaded/
└── evals/

{temporary_logs_root}/
├── cascaded/
└── evals/

{exports_root}/
```

- `{results_root}` contains canonical scientific results.
- `{logs_root}` contains execution logs for canonical runs.
- `{temporary_results_root}` and `{temporary_logs_root}` contain demo,
  smoke-test, fit-test, and other disposable runs.
- `{exports_root}` contains portable bundles derived from results. It may remain
  empty until needed.
- Generated results, logs, temporary runs, exports, audio, checkpoints, and
  other large artifacts must stay out of Git.

The high-level `cascaded/` structure is defined below. Detailed per-stage manifest fields and the `evals/` structure remain under discussion.

## Configurable Roots

The project configuration should define independent named roots:

```text
results_root
logs_root
temporary_results_root
temporary_logs_root
exports_root
environment_root
model_root
cache_root
```

Paths must be constructed from these roots and configured path extensions. Active scripts should not contain user- or cluster-specific absolute paths.

On the current machine, these roots should resolve under
`/scratch/l/lovenya/salm_migration_clean` for real runs. The old repo-local
defaults may still exist in config while migration/refactor work is in progress;
new work should prefer the external root.

## Canonical And Temporary Runs

Canonical output is the default. No canonical CLI flag is needed.

Passing:

```text
--temporary
```

redirects both result and log output:

```text
{results_root}/...       -> {temporary_results_root}/...
{logs_root}/...          -> {temporary_logs_root}/...
```

Temporary runs must not be consumed as official results automatically. If a temporary result is worth preserving, it must be deliberately copied into the canonical results and logs structure.

Current cleanup policy: previous contents of `temporary/` were demo/smoke tests
and can be deleted. Keep the top-level `temporary/` directory if useful, but do
not preserve its old contents as scientific evidence.

## Legacy Repo-Local Results

The old repo-local eval path:

```text
results/evals/T2S/
```

is a legacy layout. If its contents are from before 2026-06-14, they can be
removed instead of migrated. New eval outputs should follow
`EVALS_STRUCTURE_DESIGN.md` under `{results_root}/evals/stages/...`.

## Logs

The directory hierarchy under `logs/` must mirror the corresponding hierarchy under `results/`.

Example:

```text
results/cascaded/<...>/<run_id>/
logs/cascaded/<...>/<run_id>/
```

For temporary runs:

```text
temporary/results/cascaded/<...>/<run_id>/
temporary/logs/cascaded/<...>/<run_id>/
```

Logs contain execution logs only. Run metadata, resolved configuration, manifests, outputs, inspection records, and summaries stay with the result.

Each run's metadata must record its corresponding logs path.

## Run IDs

Run IDs use Montreal local time:

```text
YYYYMMDD_HHMM
```

Example:

```text
20260615_1437
```

Generate them using the `America/Toronto` timezone rather than a fixed `EDT` offset so daylight-saving changes are handled correctly.

If the selected run directory already exists, the program must fail immediately before model loading, GPU work, or other expensive initialization unless it is an explicit resume of the same run. Resume is allowed only under the rules in [Resume Behavior](#resume-behavior): identical `config.json`, non-terminal run status, no orphan outputs, and no duplicate manifest identity. The program must never overwrite an existing run.

For a non-resumable collision, interactive use may ask the user to wait for the next minute or provide another run ID. Non-interactive Slurm jobs must print a clear error and exit.

## Run Directory

The current agreed generic shape is:

```text
results/<category>/<...>/<run_id>/
├── run.json
├── config.json
├── manifest.jsonl
├── outputs/
├── inspect.json
├── errors.json
└── summary.json
```

Only applicable files and directories should be created. Empty `outputs/` directories are unnecessary for stages that do not generate media. `manifest.jsonl` is machine-oriented and compact; `inspect.json` is human-oriented and may include richer joined stage context.

### `run.json`

Stores run identity and lifecycle metadata. It should not contain generation hyperparameters or per-sample output data.

Field order:

```text
schema_version
run_id
temporary
stage
direction
status
dataset
split
model
started_at
completed_at
code
slurm
paths
lineage
```

Example:

```json
{
  "schema_version": 1,
  "run_id": "20260616_1437",
  "temporary": false,
  "stage": "t2t",
  "direction": "a2s",
  "status": "complete",
  "dataset": "ttas",
  "split": "val",
  "model": "llama3_3b_instruct",
  "started_at": "2026-06-16T14:37:12-04:00",
  "completed_at": "2026-06-16T16:05:43-04:00",
  "code": {
    "git_commit": "4c49181...",
    "git_dirty": false
  },
  "slurm": {
    "job_id": "321500",
    "array_job_id": null,
    "array_task_id": null
  },
  "paths": {
    "result": "results/cascaded/stages/t2t/...",
    "logs": "logs/cascaded/stages/t2t/..."
  },
  "lineage": {
    "direction": "a2s",
    "current_stage_index": 2,
    "stages": [
      {
        "stage_index": 1,
        "stage": "a2t",
        "model": "af_next_aac",
        "run_id": "20260616_1210",
        "run_path": "results/cascaded/stages/a2t/af_next_aac/ttas/val/20260616_1210",
        "manifest": "results/cascaded/stages/a2t/af_next_aac/ttas/val/20260616_1210/manifest.jsonl",
        "config": "results/cascaded/stages/a2t/af_next_aac/ttas/val/20260616_1210/config.json"
      },
      {
        "stage_index": 2,
        "stage": "t2t",
        "model": "llama3_3b_instruct",
        "run_id": "20260616_1437",
        "run_path": "results/cascaded/stages/t2t/a2s/llama3_3b_instruct/from-a2t/af_next_aac/ttas/val/20260616_1437",
        "manifest": "results/cascaded/stages/t2t/a2s/llama3_3b_instruct/from-a2t/af_next_aac/ttas/val/20260616_1437/manifest.jsonl",
        "config": "results/cascaded/stages/t2t/a2s/llama3_3b_instruct/from-a2t/af_next_aac/ttas/val/20260616_1437/config.json"
      }
    ]
  }
}
```

- `temporary`: Whether the run was written under `temporary/`.
- `direction`: `a2s` or `s2a`; omit where irrelevant.
- `code`: Repository state captured at job start, immediately before launching the workload.
- `slurm`: Scheduler identity; values may be `null` outside Slurm.
- `paths`: Root-relative result and log paths.
- `lineage`: Ordered stage lineage known to this run; stage 1 contains only itself, stage 2 contains stages 1-2, and stage 3 contains stages 1-3.

Do not include a `kind` field. The directory path already distinguishes stage and pipeline records. Do not include a repository path inside `code`.

Incomplete or failed runs remain in their intended location with an appropriate status. They are not silently moved into `temporary/`.

### `config.json`

Stores exact resolved settings that affect output generation or the selected input set. It should contain only settings relevant to the current stage, not every possible model family.

Field order:

```text
schema_version
input
model
prompt
generation
execution
```

Example:

```json
{
  "schema_version": 1,
  "input": {
    "manifest": "results/cascaded/stages/a2t/.../manifest.jsonl",
    "input_field": "output_text",
    "start_index": 0,
    "num_samples": 2277
  },
  "model": {
    "name": "llama3_3b_instruct",
    "path": "{model_root}/llama3_3b_instruct",
    "cache_dir": "{cache_root}/huggingface",
    "dtype": "bfloat16",
    "device": "cuda"
  },
  "prompt": {
    "mode": "audio_description_to_speech",
    "template": "...",
    "system_prompt": "..."
  },
  "generation": {
    "seed": 1337,
    "steps": 64,
    "duration_seconds": null
  },
  "execution": {
    "continue_on_error": true,
    "shard_index": null,
    "num_shards": null
  }
}
```

- `input`: Data source consumed by this stage.
- `input_field`: Descriptive/debugging metadata for the field consumed from the input manifest. It is not the only source of truth for stage behavior.
- `model.path`: Checkpoint, weight directory, or model identifier.
- `model.cache_dir`: Optional runtime cache such as Hugging Face, compiled kernel, vLLM, or metric cache.
- `prompt`: Omit for stages that do not use prompts.
- `generation`: Fixed run-wide generation settings. `steps` belongs here, not in every manifest row.
- `duration_seconds`: May be `null` when dynamic or not applicable.
- `execution`: Operational behavior for errors and sharding.

Use named-root paths such as `{model_root}/...` and `{cache_root}/...`, not cluster-specific absolute paths. This snapshot remains stable even if the project's main configuration changes later.

### `manifest.jsonl`

Contains the primary machine-readable results, one row per sample. It should be compact and optimized for inference and evaluation scripts.

A row connects a stable sample ID to its current-stage compact input link, output, status, and error. It should not copy full upstream inputs or previous-stage outputs.

For generated-media stages, the manifest maps each sample to its file under `outputs/`. Per-sample actual media duration belongs in the manifest because it can vary even when requested duration is fixed.

Manifest rows should include minimal input linkage for assertions, such as input sample ID, source stage index, and source status. Full upstream stage data belongs in `inspect.json`, not in every machine manifest row.

Base field order:

```text
schema_version
row_index
sample_id
stage
model
status
input
output
timing
error
```

Example:

```json
{
  "schema_version": 1,
  "row_index": 12,
  "sample_id": "ttas_123",
  "stage": "t2t",
  "model": "llama3_3b_instruct",
  "status": "success",
  "input": {
    "source_stage_index": 1,
    "source_sample_id": "ttas_123",
    "source_status": "success"
  },
  "output": {
    "type": "text",
    "generated_speech_text": "..."
  },
  "timing": {
    "inference_seconds": 2.14
  },
  "error": null
}
```

`output.type` is required so downstream scripts can route rows without guessing from the stage. Text stages use `"text"`; generated media stages use `"audio"` or `"speech"` as appropriate.

### Stage-Specific Manifest Payloads

All stage manifests use the common base row fields. The stage-specific part is the shape of `input` and `output`.

#### A2T

Audio to text:

```json
"input": {
  "audio_file": "..."
},
"output": {
  "type": "text",
  "audio_caption": "..."
}
```

#### S2T

Speech to text:

```json
"input": {
  "speech_file": "..."
},
"output": {
  "type": "text",
  "speech_transcription": "..."
}
```

#### T2T

For A2S:

```json
"input": {
  "source_stage_index": 1,
  "source_sample_id": "ttas_123",
  "source_status": "success"
},
"output": {
  "type": "text",
  "generated_speech_text": "..."
}
```

For S2A:

```json
"input": {
  "source_stage_index": 1,
  "source_sample_id": "ttas_123",
  "source_status": "success"
},
"output": {
  "type": "text",
  "generated_audio_caption": "..."
}
```

#### T2S

Text to speech:

```json
"input": {
  "source_stage_index": 2,
  "source_sample_id": "ttas_123",
  "source_status": "success"
},
"output": {
  "type": "speech",
  "speech_file": "outputs/000012_ttas_123.wav",
  "sample_rate": 24000,
  "duration_seconds": 9.83
}
```

#### T2A

Text to audio:

```json
"input": {
  "source_stage_index": 2,
  "source_sample_id": "ttas_123",
  "source_status": "success"
},
"output": {
  "type": "audio",
  "audio_file": "outputs/000012_ttas_123.wav",
  "sample_rate": 32000,
  "duration_seconds": 10.0
}
```

#### Error Rows

Error rows keep the same base fields, set `status` to `error`, set `output` to `null`, and populate `error`:

```json
"status": "error",
"output": null,
"error": {
  "type": "RuntimeError",
  "message": "..."
}
```

### `outputs/`

Contains generated media or other external output files.

- S2T: no media output; manifest only.
- A2T: no media output; manifest only.
- T2T: no media output; manifest only.
- T2A: generated audio under `outputs/`, referenced by a manifest.
- T2S: generated speech under `outputs/`, referenced by a manifest.
- Evals: normally manifests and summaries only; no generated media.

### `inspect.json`

Contains data organized for human inspection. It may combine prompts, references, intermediate-stage values, generated outputs, model paths, checkpoints, hyperparameters, and useful paths without changing the canonical machine-oriented manifest schema.

`inspect.json` must be actively captured at runtime, not passively reconstructed later. For prompts, store the exact rendered payload produced at the model-call boundary; for chat models, this means the actual `[{"role": ..., "content": ...}, ...]` messages sent or attempted. Rebuilding prompts later from templates and input text is not acceptable for debugging because it can hide template-rendering bugs.

Inspection scope grows with the stage:

- Stage 1 inspect: stage 1 only.
- Stage 2 inspect: stages 1 and 2.
- Stage 3 inspect: stages 1, 2, and 3.

Stage-three T2A and T2S runs should provide an inspection JSON alongside their output manifest and generated media. Pipeline runs should also provide a final end-to-end inspection JSON. This pipeline view is intentionally retained even though some information overlaps with stage-three inspection data.

#### Inspect Structure

Top-level field order:

```text
schema_version
run_id
stage
direction
dataset
split
status
lineage
stage_configs
summary
samples
```

Example skeleton:

```json
{
  "schema_version": 1,
  "run_id": "20260616_1437",
  "stage": "t2t",
  "direction": "a2s",
  "dataset": "ttas",
  "split": "val",
  "status": "complete",
  "lineage": {
    "direction": "a2s",
    "current_stage_index": 2,
    "stages": []
  },
  "stage_configs": [],
  "summary": {
    "requested": 2277,
    "successful": 2274,
    "failed": 3
  },
  "samples": []
}
```

`lineage` stores identity and file linkage, not hyperparameters:

```json
"lineage": {
  "direction": "a2s",
  "current_stage_index": 2,
  "stages": [
    {
      "stage_index": 1,
      "stage": "a2t",
      "model": "af_next_aac",
      "run_id": "20260616_1000",
      "run_path": "results/cascaded/stages/a2t/af_next_aac/ttas/val/20260616_1000",
      "manifest": "results/cascaded/stages/a2t/af_next_aac/ttas/val/20260616_1000/manifest.jsonl",
      "config": "results/cascaded/stages/a2t/af_next_aac/ttas/val/20260616_1000/config.json"
    },
    {
      "stage_index": 2,
      "stage": "t2t",
      "model": "llama3_3b_instruct",
      "run_id": "20260616_1437",
      "run_path": "results/cascaded/stages/t2t/a2s/llama3_3b_instruct/from-a2t/af_next_aac/ttas/val/20260616_1437",
      "manifest": "results/cascaded/stages/t2t/a2s/llama3_3b_instruct/from-a2t/af_next_aac/ttas/val/20260616_1437/manifest.jsonl",
      "config": "results/cascaded/stages/t2t/a2s/llama3_3b_instruct/from-a2t/af_next_aac/ttas/val/20260616_1437/config.json"
    }
  ]
}
```

`stage_configs` stores a compact human-readable resolved config snapshot for each known stage. It should summarize the actual config without copying every full `config.json` verbatim:

```json
"stage_configs": [
  {
    "stage_index": 1,
    "stage": "a2t",
    "model": {
      "name": "af_next_aac",
      "path": "{model_root}/audio_flamingo_next",
      "cache_dir": "{cache_root}/huggingface",
      "device": "cuda",
      "dtype": "bfloat16"
    },
    "input": {
      "manifest": "data/manifests/ttas_val.jsonl",
      "input_field": "audio_file"
    },
    "generation": {
      "seed": 1337
    }
  },
  {
    "stage_index": 2,
    "stage": "t2t",
    "model": {
      "name": "llama3_3b_instruct",
      "path": "{model_root}/llama3_3b_instruct",
      "cache_dir": "{cache_root}/huggingface",
      "device": "cuda",
      "dtype": "bfloat16"
    },
    "input": {
      "manifest": "results/cascaded/stages/a2t/af_next_aac/ttas/val/20260616_1000/manifest.jsonl",
      "input_field": "audio_caption"
    },
    "prompt": {
      "mode": "audio_description_to_speech",
      "template_name": "a2s_default"
    },
    "generation": {
      "seed": 1337,
      "temperature": 0.7,
      "max_new_tokens": 512
    }
  }
]
```

Each item in `samples` is a human-readable joined view for one sample. It includes reference data, stage-by-stage runtime inputs, runtime-captured prompts, outputs, timing, and errors:

```json
{
  "sample_id": "ttas_123",
  "status": "success",
  "reference": {
    "audio_file": "...",
    "speech_file": "...",
    "reference_audio_caption": "...",
    "reference_speech_text": "...",
    "audio_metadata": {}
  },
  "stages": [
    {
      "stage_index": 1,
      "stage": "a2t",
      "model": "af_next_aac",
      "status": "success",
      "input": {
        "audio_file": "..."
      },
      "runtime_prompt": {
        "messages": [
          {
            "role": "user",
            "content": "..."
          }
        ]
      },
      "output": {
        "type": "text",
        "audio_caption": "..."
      },
      "timing": {
        "inference_seconds": 2.14
      },
      "error": null
    }
  ],
  "final_output": {
    "type": "speech",
    "speech_file": "outputs/000012_ttas_123.wav"
  }
}
```

For non-prompt models, set `runtime_prompt` to `null` rather than reconstructing one later.

### `errors.json`

Contains structured debugging information for failed samples. It is derived from the manifest plus optional log references, and it does not replace verbose execution logs.

Field order:

```text
schema_version
run_id
error_count
errors
```

Example:

```json
{
  "schema_version": 1,
  "run_id": "20260616_1437",
  "error_count": 3,
  "errors": [
    {
      "sample_id": "ttas_123",
      "row_index": 12,
      "stage": "t2t",
      "error_type": "RuntimeError",
      "message": "...",
      "log_ref": "logs/cascaded/stages/t2t/.../run.log"
    }
  ]
}
```

Logs retain full verbose details. `errors.json` is the quick structured view for debugging failed rows.

### `summary.json`

Contains aggregate output metadata only. It must not contain per-sample results.

Field order:

```text
schema_version
run_id
status
samples
timing
outputs
errors
```

Example:

```json
{
  "schema_version": 1,
  "run_id": "20260616_1437",
  "status": "complete_with_errors",
  "samples": {
    "requested": 2277,
    "processed": 2277,
    "successful": 2274,
    "failed": 3,
    "skipped": 0
  },
  "timing": {
    "model_load_seconds": 42.18,
    "total_inference_seconds": 5031.44,
    "total_seconds": 5089.72,
    "mean_inference_seconds_per_sample": 2.21,
    "median_inference_seconds_per_sample": 2.04,
    "min_inference_seconds_per_sample": 1.41,
    "max_inference_seconds_per_sample": 8.92
  },
  "outputs": {
    "manifest": "manifest.jsonl",
    "inspect": "inspect.json",
    "generated_media_files": null
  },
  "errors": {
    "count": 3,
    "by_type": {
      "RuntimeError": 2,
      "ValueError": 1
    }
  }
}
```

- `requested`: Number selected for this run.
- `processed`: Number actually attempted.
- `skipped`: Existing or deliberately excluded rows.
- `generated_media_files`: `null` for non-media stages; count for T2A/T2S.
- `timing`: Aggregate load, inference, and per-sample timing statistics.
- `errors.by_type`: Aggregate error classes, not full per-sample messages.

Lifecycle timestamps belong in `run.json`; aggregate durations belong in `summary.json`. Per-sample results belong in manifests. Structured per-run failures are also written to `errors.json`.

## Cascaded Results

The cascaded results hierarchy separates reusable stage runs from lightweight end-to-end pipeline records:

```text
results/cascaded/
├── stages/
│   ├── a2t/
│   ├── s2t/
│   ├── t2t/
│   ├── t2a/
│   └── t2s/
└── pipelines/
    ├── a2s/
    └── s2a/
```

The symmetric stage names are:

- `a2t`: audio to text, replacing the workflow label AAC
- `s2t`: speech to text, replacing the workflow label ASR
- `t2t`: text to text
- `t2a`: text to audio
- `t2s`: text to speech

Stage results are stored independently so that upstream A2T or S2T results can be reused without copying their manifests.

### T2T Hierarchy

T2T paths record the pipeline direction, T2T model, upstream stage type, and upstream model:

```text
stages/t2t/
├── a2s/
│   └── <t2t_model>/
│       └── from-a2t/
│           └── <a2t_model>/
│               └── <dataset>/<split>/<run_id>/
└── s2a/
    └── <t2t_model>/
        └── from-s2t/
            └── <s2t_model>/
                └── <dataset>/<split>/<run_id>/
```

Examples:

```text
stages/t2t/a2s/llama3_3b_instruct/from-a2t/af_next_aac/ttas/val/20260615_1437/
stages/t2t/s2a/llama3_3b_instruct/from-s2t/whisper/ttas/val/20260615_1512/
```

Each T2T run consumes one upstream model run. Supporting arbitrary multi-source input sets is out of scope.

The run metadata references the exact upstream run and manifest. The path describes the relationship; metadata provides precise lineage.

T2T configuration should be selected by direction and previous-stage model, not by manually passing arbitrary prompt fields. Valid previous-stage model choices are direction-specific:

- `a2s`: A2T/captioning models such as Audio Flamingo Next and Qwen3-Omni captioning.
- `s2a`: S2T/ASR models such as Whisper and Qwen ASR.

The CLI/config layer must reject incompatible combinations, such as selecting `s2a` with an A2T captioner as the previous-stage model.

The non-T2T stage hierarchy is:

```text
results/cascaded/stages/
├── a2t/
│   └── <a2t_model>/<dataset>/<split>/<run_id>/
├── s2t/
│   └── <s2t_model>/<dataset>/<split>/<run_id>/
├── t2t/
│   ├── a2s/<t2t_model>/from-a2t/<a2t_model>/<dataset>/<split>/<run_id>/
│   └── s2a/<t2t_model>/from-s2t/<s2t_model>/<dataset>/<split>/<run_id>/
├── t2a/
│   └── <t2a_model>/from-t2t/<t2t_model>/<dataset>/<split>/<run_id>/
└── t2s/
    └── <t2s_model>/from-t2t/<t2t_model>/<dataset>/<split>/<run_id>/
```

`t2a` and `t2s` paths do not include direction because the direction is implied by the stage: `t2a` belongs to S2A and `t2s` belongs to A2S. Exact upstream run IDs still live in `run.json.lineage`.

### Run Identity And Path Lookup

Cascaded stage run IDs are date/time identifiers, for example `20260618_0001` or `20260618_1211`. They are not semantic aliases such as `full_a2s_qwencap_llama_t2t`. Human-readable model and upstream-stage information belongs in the directory path, `config.json`, and `run.json.lineage`; the terminal run directory name remains the run ID.

The stage model identity is the concrete component model segment in the path, such as `llama3_3_70b_instruct`, `audio_flamingo_next`, `whisper`, `cosyvoice3`, or `audio_omni`. `model: cascaded` is intentionally not part of the results schema because it is a category label, not a model. If an eval command or legacy helper still requires `--model cascaded`, treat that as a current CLI compatibility label only and keep the real model identity in the target manifest path and metadata.

Do not discover canonical runs from broad filesystem listings or stale validation directories. The authoritative cascaded stage path pattern is the hierarchy in this document, with the requested dataset split and exact run ID.

### Lineage And Sample Matching

Every stage run records ordered lineage in `run.json`, including the direction, current stage index, and every stage known so far. Each lineage entry includes:

```text
stage_index
stage
model
run_id
run_path
manifest
config
```

Stage 1 lineage contains only itself. Stage 2 lineage contains stages 1 and 2. Stage 3 lineage contains stages 1, 2, and 3.

Downstream stages consume one previous-stage input per output sample. The invariant is:

```text
current_output.sample_id == input_row.sample_id
```

The runner should assert this before writing outputs. Pipeline assembly must join stages by `sample_id`, not by row position, because failures can make row positions diverge across stage manifests.

Each stage manifest should retain all attempted rows for that stage, with `success` or `error` status. A downstream stage normally consumes only successful rows from the previous stage.

### Pipeline Records

Pipeline directories contain end-to-end lineage and inspection records:

```text
pipelines/
├── a2s/
│   └── <pipeline_name>/<dataset>/<split>/<run_id>/
└── s2a/
    └── <pipeline_name>/<dataset>/<split>/<run_id>/
```

An A2S pipeline references:

```text
A2T run -> T2T run -> T2S run
```

An S2A pipeline references:

```text
S2T run -> T2T run -> T2A run
```

A pipeline run gets its own run ID when the end-to-end record is assembled. It does not need to match any stage run ID.

Example:

```text
A2T run:      20260616_1000
T2T run:      20260616_1230
T2S run:      20260616_1500
Pipeline run: 20260616_1530
```

A pipeline run contains:

```text
<run_id>/
├── run.json
├── pipeline.json
├── manifest.jsonl
├── inspect.json
└── summary.json
```

- `run.json` records pipeline identity, status, timestamps, code state, and paths.
- `pipeline.json` records exact stage runs, models, manifests, and the final output location.
- `manifest.jsonl` is a machine-friendly joined manifest, one row per sample, linking all stages by `sample_id`.
- `inspect.json` provides a joined end-to-end view for human inspection.
- `summary.json` records end-to-end counts and completion status.

Pipeline `manifest.jsonl` exists because the stage-three manifest is biased toward stage-three execution. The pipeline manifest is the compact machine-readable end-to-end view. It includes the original pipeline input, such as the input audio file for A2S or input speech file for S2A, plus stage statuses and final output. Pipeline `inspect.json` is the richer human-readable view.

Pipeline manifest example:

```json
{
  "schema_version": 1,
  "sample_id": "ttas_123",
  "status": "success",
  "direction": "a2s",
  "input": {
    "audio_file": "..."
  },
  "stages": [
    {
      "stage_index": 1,
      "stage": "a2t",
      "model": "af_next_aac",
      "run_id": "20260616_1000",
      "status": "success"
    },
    {
      "stage_index": 2,
      "stage": "t2t",
      "model": "llama3_3b_instruct",
      "run_id": "20260616_1230",
      "status": "success"
    },
    {
      "stage_index": 3,
      "stage": "t2s",
      "model": "f5_tts",
      "run_id": "20260616_1500",
      "status": "success"
    }
  ],
  "final_output": {
    "type": "speech",
    "speech_file": "results/cascaded/stages/t2s/.../outputs/000012_ttas_123.wav"
  },
  "error": null
}
```

For S2A, pipeline `input` uses `speech_file`. If a sample fails before the final stage, set pipeline `status` to `incomplete`, set `final_output` to `null`, and populate `error` with the failed stage index, failed stage name, error type, and message.

Pipeline directories do not copy stage manifests or generated media. Canonical outputs remain in their stage-run directories.

### `pipeline.json`

Stores stage-run relationships without copying result data.

Field order:

```text
schema_version
run_id
direction
stages
final_output
```

Example:

```json
{
  "schema_version": 1,
  "run_id": "20260616_1530",
  "direction": "a2s",
  "stages": [
    {
      "position": 1,
      "stage": "a2t",
      "model": "af_next_aac",
      "run_id": "20260616_1000",
      "run_path": "results/cascaded/stages/a2t/.../20260616_1000",
      "manifest": "results/cascaded/stages/a2t/.../manifest.jsonl"
    },
    {
      "position": 2,
      "stage": "t2t",
      "model": "llama3_3b_instruct",
      "run_id": "20260616_1230",
      "run_path": "results/cascaded/stages/t2t/.../20260616_1230",
      "manifest": "results/cascaded/stages/t2t/.../manifest.jsonl"
    },
    {
      "position": 3,
      "stage": "t2s",
      "model": "f5_tts",
      "run_id": "20260616_1500",
      "run_path": "results/cascaded/stages/t2s/.../20260616_1500",
      "manifest": "results/cascaded/stages/t2s/.../manifest.jsonl"
    }
  ],
  "final_output": {
    "stage": "t2s",
    "run_id": "20260616_1500",
    "manifest": "results/cascaded/stages/t2s/.../manifest.jsonl"
  }
}
```

`final_output` is derivable from the last stage, but it is kept as the authoritative convenience entry point for evaluation.


## Sharding And Merge

Sharding applies to one stage run at a time. The new design does not use full per-sample cascaded inference shards that load all three models together. Instead, each stage is run and merged independently:

```text
A2T shards -> merged A2T manifest
T2T shards -> merged T2T manifest
T2S shards -> merged T2S manifest
```

A downstream stage must consume the previous stage's canonical merged manifest. Do not start a downstream full-stage run from an incomplete set of upstream shards.

Shard-local outputs live under the stage run directory, while the canonical merged files live at that same run root.

During execution:

```text
<run_id>/
├── run.json
├── config.json
└── shards/
    ├── shard_000/
    │   ├── manifest.jsonl
    │   ├── errors.json
    │   ├── summary.json
    │   └── outputs/
    ├── shard_001/
    └── ...
```

After a clean merge:

```text
<run_id>/
├── run.json
├── config.json
├── manifest.jsonl
├── inspect.json
├── errors.json
├── summary.json
├── merge_report.json
└── outputs/
```

The `shards/` directory should be deleted after a successful clean merge. Keep `merge_report.json` at the run root. If merge fails, keep `shards/` for debugging and set the run status accordingly.

The merge step must be implemented as a dedicated stage-shard merge script or helper, not ad hoc shell commands. It should be deterministic and reusable across A2T, S2T, T2T, T2A, and T2S stage runs. Historical full-cascade shard merge scripts are legacy references, not the target design.

Merge rules:

- Sort rows by `row_index`.
- Detect duplicate `sample_id`.
- Detect duplicate `row_index`.
- Preserve error rows.
- Count missing expected rows.
- Validate sample ID consistency.
- Merge or copy media files into canonical `outputs/`.
- Fail if a successful generated-media row references a missing media file.
- Write `merge_report.json`.
- Delete shard directories only after the merged manifest, errors, summary, inspect file, media files, and merge report have been written and validated.

`merge_report.json` example:

```json
{
  "schema_version": 1,
  "run_id": "20260616_1437",
  "shards": [
    {
      "shard_id": "shard_000",
      "manifest": "shards/shard_000/manifest.jsonl",
      "requested": 500,
      "successful": 498,
      "failed": 2
    }
  ],
  "merged": {
    "requested": 2277,
    "rows_written": 2277,
    "successful": 2270,
    "failed": 7,
    "missing": 0,
    "duplicate_sample_ids": [],
    "duplicate_row_indices": []
  },
  "outputs": {
    "media_files_copied": 2270,
    "missing_media_files": []
  },
  "status": "complete"
}
```

Temporary sharded runs use the same structure under `temporary/results/.../<run_id>/`.

## Resume Behavior

Resume means the same run continues after interruption, preemption, or partial completion. It uses the same `run_id` and identical `config.json`.

Manual reruns are fresh runs with new run IDs and do not need special schema support.

Resume rules:

- Resume only if `config.json` matches exactly.
- Skip already successful rows.
- Keep existing error rows as errors; do not silently overwrite them.
- Append missing rows.
- Fail if an output file exists without a matching manifest row.
- Fail if the manifest has duplicate `sample_id` or duplicate `row_index`.
- Fail if a successful media row points to a missing media file.
- Fail by default if the run is already `complete`.

`run.json` records lifecycle events:

```json
"events": [
  {
    "type": "started",
    "time": "2026-06-16T14:37:12-04:00",
    "start_row_index": 0
  },
  {
    "type": "resumed",
    "time": "2026-06-16T16:20:00-04:00",
    "start_row_index": 842,
    "existing_successful_rows": 842,
    "existing_error_rows": 3,
    "reason": "manual_resume"
  },
  {
    "type": "completed",
    "time": "2026-06-16T18:45:00-04:00",
    "successful_rows": 2274,
    "error_rows": 3
  }
]
```

## Omitted Legacy Fields

The new schema intentionally omits or moves several legacy fields:

- `lane`: Unclear duplicate of the variant/model combination.
- `variant`: Replaced by structured stage, model, direction, and run ID.
- `model: cascaded`: Not a real model identity.
- `kind`: Redundant with the directory path and file location.
- `task`: Duplicates `stage` or `direction` in the cascaded structure.
- `current_stage`: Derivable from pipeline stage position and inspection scope.
- `config_file`: Replaced by `config.json`, which stores the exact resolved configuration.
- `split_file`: Replaced by `config.input.manifest`.
- `prompt_field`: Renamed to `input_field` under `config.input`.
- Combined `asr`, `captioner`, `text_model`, `tts`, and `t2a`: each stage stores only its own model.
- Top-level `duration` and `steps`: moved to stage-specific `config.generation`.
- Repeated per-row `steps`: run-wide setting, so it belongs once in `config.json`.
- `model_paths`: replaced by the relevant `config.model.path` and optional `config.model.cache_dir`.
- `model_load_times.json`: merged into `summary.timing`.
- Generic `provenance`: avoided because explicit `code`, `lineage`, `input`, and pipeline fields are clearer. Provenance simply means evidence for where a result came from and how it was produced.

## Git And Full-Scale Jobs

Results are not committed to Git. The recorded Git commit identifies the code version used to execute the run.

Full-scale jobs must be submitted only after the intended code has been committed and the checkout is clean. Demo, smoke, and fit runs may use uncommitted code because their purpose is to validate and refine changes.

Slurm does not snapshot the repository at submission. A queued job using the shared checkout sees the code present when it starts.

Immediately before launching the workload, every job must record:

```text
git rev-parse HEAD
git dirty state
SLURM_JOB_ID, SLURM_ARRAY_JOB_ID, and SLURM_ARRAY_TASK_ID when present
```

This uses local Git data and requires no internet access. The commit captured at job start is the authoritative commit recorded for the run.

Changing files after a Python process has imported them normally does not change already loaded code. However, later imports, spawned subprocesses, workers, or runtime file reads can observe newer files. Relevant source files should therefore not be modified while a full-scale job is running.

## Open Design Decisions

The following must be discussed before implementing the full refactor:

1. Internal `evals/` hierarchy and how evaluations reference cascaded runs.

Existing results should not be moved into the new structure until these decisions are complete and the new path handling has been implemented and tested.
