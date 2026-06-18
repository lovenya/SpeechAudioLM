# NLI and G-Eval Schema Refactor Notes

Status: working design note. This captures the current decisions before the
NLI and Prometheus G-Eval schemas are refactored.

## Scope

Metrics in scope:

- `nli_contradiction`
- `g_eval_prometheus_criterion_v2_plausibility`
- `g_eval_prometheus_criterion_v2_reference_grounded`

The stale `g_eval_reference_free` metric is unsupported and removed.

## Settled NLI Decisions

- The NLI stub backend is removed. NLI must use the Hugging Face backend or fail.
- Demo and smoke tests should run on compute allocations when model loading is
  required. Unit tests may mock boundaries, but production CLI/config should not
  expose a fake metric backend.
- `nli_model` should not be repeated in every per-sample row. It belongs in
  run/config/backend metadata.
- The scalar value currently used by the main runner is
  `max_contradiction_probability`. This is a risk score, not a quality score.
- Higher NLI contradiction is worse.
- NLI score range should be explicit in output/config metadata:

```json
{
  "name": "max_contradiction_probability",
  "value": 0.73,
  "range": {"min": 0.0, "max": 1.0},
  "higher_is_better": false,
  "interpretation": "contradiction_risk"
}
```

## NLI Input Boundary

NLI should evaluate the text-to-text stage boundary when it exists:

- `input.generated` should contain the candidate text emitted by the evaluated
  T2T stage.
- `input.source` should contain the text that entered that T2T stage and the
  stage/direction context.
- `input.reference` should hold dataset reference text only when the metric
  directly consumes it. Plain NLI does not require paired reference anchors.

Current runner behavior should be tightened so NLI uses the same final T2T
stage boundary extraction that G-Eval uses when `model_extra.stages` is present,
then falls back to explicit text-pair rows.

## NLI Per-Sample Shape

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
        "contradiction_probability": 0.10
      },
      "candidate_to_source": {
        "entailment_probability": 0.20,
        "neutral_probability": 0.65,
        "contradiction_probability": 0.15
      }
    }
  },
  "error": null
}
```

## G-Eval Decisions To Revisit

- Both G-Eval variants need explicit score ranges:
  - criterion scores: integer `1..5`
  - weighted final score: real-valued `1.0..5.0` when all required criteria are
    scored
- `judge_model`, prompt versions, scoring policy, tensor parallelism, cache dir,
  and parsing policy belong in config/backend metadata, not repeated as
  uncontrolled per-row clutter.
- `criteria` should remain visible in per-sample rows, but raw prompts and raw
  judge responses should move to `artifacts/` or a dedicated
  `criterion_judgments.jsonl`.
- Reference-grounded G-Eval requires explicit `input.reference.reference_anchors`.
- Plausibility G-Eval should keep `input.reference` empty unless it directly
  consumes reference text.

Open schema choice for G-Eval: whether full criterion feedback remains nested in
`manifest.jsonl` or whether `manifest.jsonl` keeps compact scores and
`criterion_judgments.jsonl` stores full feedback/raw outputs. Current preference:
do both, with compact criterion summaries in `manifest.jsonl`.

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
