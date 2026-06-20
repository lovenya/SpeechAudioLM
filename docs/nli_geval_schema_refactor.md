# NLI and G-Eval Schema Refactor Notes

Status: merged schema note for `nli_contradiction` and LLM-based G-Eval metrics.
This file replaces the older split between `nli_geval_schema_refactor.md` and
`geval_schema_refactor_plan.md`. It records both the NLI decisions and the
implemented Prometheus LLM G-Eval storage decisions.

For G-Eval prompt wording and rubrics, read
`docs/Geval_prompts/geval_llm_prometheus_prompts.md`. This file is about eval
input/output shape, artifacts, and runner behavior.

## Scope

Metrics in scope:

- `nli_contradiction`
- `g_eval_prometheus_plausibility`
- `g_eval_prometheus_reference`

The stale `g_eval_reference_free` metric is unsupported and removed.

The current G-Eval implementation is LLM-judge based, using Prometheus 8x7B v2.0.
Future audio-judge work should use a distinct `geval_lalm_*` naming family so it
is not confused with these text-boundary LLM judge metrics.

## Shared Boundary Rule

NLI and LLM G-Eval should evaluate the text-to-text stage boundary when it
exists:

- `input.source` records the text that entered the evaluated T2T stage plus
  stage/direction context.
- `input.generated` records the candidate text emitted by that T2T stage.
- `input.reference` is populated only when the metric directly consumes
  reference anchors.

When `model_extra.stages` or equivalent lineage is present, the runner should
extract the final T2T boundary from that structured stage history. Fallbacks to
explicit text-pair rows are acceptable for tests and legacy manifests, but they
should not drive new canonical eval design.

## NLI Decisions

- The NLI stub backend is removed. NLI must use the Hugging Face backend or fail.
- Demo and smoke tests should run on compute allocations when model loading is
  required. Unit tests may mock boundaries, but production CLI/config should not
  expose a fake metric backend.
- `nli_model` belongs in run/config/backend metadata, not in every per-sample
  row.
- The scalar value used by the runner is `max_contradiction_probability`.
- `max_contradiction_probability` is a risk score, not a quality score.
- Higher NLI contradiction is worse.
- NLI score range must be explicit in output/config metadata.
- NLI is not prompt-based. It uses a Hugging Face sequence-pair classifier over
  `(premise, hypothesis)` text pairs.
- Per-sample NLI outputs should record tokenization/truncation metadata for both
  classifier directions, because `truncation=True` can affect either side of the
  pair before scoring.

Compact score metadata:

```json
{
  "name": "max_contradiction_probability",
  "value": 0.73,
  "range": {"min": 0.0, "max": 1.0},
  "higher_is_better": false,
  "interpretation": "contradiction_risk"
}
```

## NLI Canonical Row Shape

Target compact row:

```json
{
  "schema_version": 1,
  "row_index": 0,
  "sample_id": "sample",
  "metric_id": "nli_contradiction",
  "status": "success",
  "input": {
    "source": {
      "direction": "audio_to_speech",
      "stage_input_text": "audio caption or transcript entering T2T"
    },
    "generated": {
      "stage_output_text": "candidate text emitted by T2T"
    },
    "reference": {}
  },
  "output": {
    "score": 0.15,
    "score_name": "max_contradiction_probability",
    "range": {"min": 0.0, "max": 1.0},
    "higher_is_better": false,
    "label": "not_contradiction",
    "threshold": 0.7,
    "details": {
      "source_to_candidate": {
        "entailment_probability": 0.10,
        "neutral_probability": 0.80,
        "contradiction_probability": 0.10,
        "tokenization": {
          "tokenizer_max_length": 512,
          "truncation_enabled": true,
          "truncation_strategy": "longest_first",
          "truncation_side": "right",
          "premise_token_count_raw": 72,
          "hypothesis_token_count_raw": 89,
          "special_token_count_pair": 3,
          "pair_token_count_raw": 164,
          "pair_token_count_after_truncation": 164,
          "was_truncated": false
        }
      },
      "candidate_to_source": {
        "entailment_probability": 0.20,
        "neutral_probability": 0.65,
        "contradiction_probability": 0.15,
        "tokenization": {
          "tokenizer_max_length": 512,
          "truncation_enabled": true,
          "truncation_strategy": "longest_first",
          "truncation_side": "right",
          "premise_token_count_raw": 89,
          "hypothesis_token_count_raw": 72,
          "special_token_count_pair": 3,
          "pair_token_count_raw": 164,
          "pair_token_count_after_truncation": 164,
          "was_truncated": false
        }
      }
    }
  },
  "error": null
}
```

## LLM G-Eval Decisions

Metrics in scope:

- `g_eval_prometheus_plausibility`
- `g_eval_prometheus_reference`

Out of scope for the schema cleanup:

- changing criterion rubrics, weights, prompt wording, or scoring policy;
- running vLLM/Prometheus on login nodes;
- changing NLI behavior except shared helper-boundary utilities if needed;
- adding metric-suite/group behavior.

Implemented decisions:

- The Prometheus helper script may keep an internal resumable JSONL shape with
  full criterion payloads. That file is a backend artifact, not the canonical
  eval manifest schema.
- The canonical eval `manifest.jsonl` row for G-Eval uses the common compact
  shape:

```text
schema_version, row_index, sample_id, metric_id, status, input, output, error
```

- `input.source` records the T2T boundary source context: `direction`,
  `stage_input_text`, and optional `primary_audio_class`.
- `input.generated` records `stage_output_text`.
- `input.reference` is empty for plausibility.
- `input.reference.reference_anchors` is required for
  `g_eval_prometheus_reference`.
- Criterion scores use integer range `1..5`.
- `output.score` is the weighted final score, with range `1.0..5.0` when all
  required criteria are scored.
- `output.details.criteria` contains compact criterion summaries only:
  criterion name, score, score range, higher-is-better flag, weight,
  parse status, feedback, abstention reason, and parse-repair summary fields
  when applicable.
- Compact `manifest.jsonl` and top-level `inspect.json` do not inline raw
  prompts or raw judge responses; they remain readable, per-sample views.
- Missing required reference anchors are row-level metric errors for the
  reference-grounded metric.
- The helper evaluator version is `geval_prometheus_criterion_v3`, so fresh
  runs do not reuse older cache entries that lack exact runtime prompt traces.
- Parse repair is represented explicitly with `parse_repair_attempted`,
  `parse_repair_succeeded`, and `parse_repair_method`.

## LLM G-Eval Artifacts

Runtime traces are stored per sample and per criterion in:

```text
artifacts/dedicated-script/<metric_id>/criterion_judgments.jsonl
```

Each row captures:

- rendered user prompt;
- prompt template;
- chat messages;
- exact backend-formatted prompt from the generation call;
- raw judge response;
- parse status;
- parsed score and feedback;
- repair details if a format repair was attempted.

When a repair call happens, the detailed repair object records the repair prompt
trace and repair raw response.

## Smoke Workflow

Use temporary eval runs for smoke/demo checks. They still use the canonical eval
shape and explicit CLI:

```text
--stage a2t|s2t|t2t
--direction a2s|s2a for t2t
--evaluated-run-id <run-id>
--metrics <metric-id>[,<metric-id>...]
--temporary
```

The interactive helper is:

```text
submission_scripts/run_tamia_eval_text_metric_smoke_interactive.sh
```

It is intended for an active TamIA GPU allocation and writes logs under
`$SALM_SCRATCH/logs/evals/text_metric_smoke/<run_id>/`.

## Verification

CPU-safe G-Eval schema tests:

```text
python -m unittest tests.test_geval_prometheus_criterion_v2 tests.test_geval_prometheus_criterion_v2_runner
```

A Prometheus/vLLM run is not part of CPU-safe refactor verification; it should
be run from an allocated GPU node.
