# G-Eval Schema Refactor Plan

Status: implemented plan for Prometheus G-Eval schema cleanup.

This note narrows the G-Eval work from `EVALS_STRUCTURE_DESIGN.md` and
`docs/nli_geval_schema_refactor.md` into concrete implementation decisions.

## Scope

Metrics in scope:

- `g_eval_prometheus_plausibility`
- `g_eval_prometheus_reference`

Out of scope for this pass:

- changing criterion rubrics, weights, prompt wording, or scoring policy
- running vLLM/Prometheus on login nodes
- changing NLI behavior except shared helper-boundary utilities if needed
- adding metric-suite/group behavior

## Decisions

- The Prometheus helper script may keep an internal resumable JSONL shape with
  full criterion payloads. That file is a backend artifact, not the canonical
  eval manifest schema.
- The canonical eval `manifest.jsonl` row for G-Eval uses the common compact
  shape:

```text
schema_version, row_index, sample_id, metric_id, status, input, output, error
```

- `input.source` records the T2T boundary source context:
  `direction`, `stage_input_text`, and optional `primary_audio_class`.
- `input.generated` records `stage_output_text`.
- `input.reference` is empty for plausibility.
- `input.reference.reference_anchors` is required for
  `g_eval_prometheus_reference`.
- `output.score` is the weighted final score, with range `1.0..5.0`.
- `output.details.criteria` contains compact criterion summaries only:
  criterion name, score, score range, higher-is-better flag, weight,
  parse status, feedback, abstention reason, and parse-repair summary fields
  when applicable.
- Runtime traces are stored per sample and per criterion in
  `artifacts/dedicated-script/<metric_id>/criterion_judgments.jsonl`. Each row
  captures the rendered user prompt, prompt template, messages, exact
  backend-formatted prompt from the generation call, raw judge response, parse
  status, parsed score/feedback, and repair details if a format repair was
  attempted.
- Compact `manifest.jsonl` and top-level `inspect.json` do not inline raw
  prompts or raw judge responses; they remain readable, per-sample views.
- Missing required reference anchors are row-level metric errors for the
  reference-grounded metric.
- The helper evaluator version is `geval_prometheus_criterion_v3`, so fresh
  runs do not reuse older cache entries that lack exact runtime prompt traces.
- Parse repair is represented explicitly with `parse_repair_attempted`,
  `parse_repair_succeeded`, and `parse_repair_method`. The detailed repair
  object records the repair prompt trace and repair raw response when a repair
  call happens.

## Implementation Steps

1. Add G-Eval-specific canonicalization helpers in `evaluation/runner.py`.
2. Make G-Eval `manifest.jsonl` inputs use the final T2T stage boundary rather
   than generic prediction dumps.
3. Compact G-Eval `output.details` while preserving score range metadata.
4. Preserve full per-criterion helper payloads in `criterion_judgments.jsonl`
   for prompt/raw-response debugging.
5. Update focused G-Eval runner tests to assert compact manifest shape,
   reference handling, exact prompt trace capture, and parse-repair fields.
6. Run targeted unit tests and a CPU stub demo against the merged T2T test
   manifest.

## Verification

Use CPU-safe tests:

```text
python -m unittest tests.test_geval_prometheus_criterion_v2 tests.test_geval_prometheus_criterion_v2_runner
```

A Prometheus/vLLM run is not part of CPU-safe refactor verification; it should
be run from an allocated GPU node.
