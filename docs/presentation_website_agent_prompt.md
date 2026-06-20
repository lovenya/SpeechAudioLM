# SpeechAudioLM Presentation Website Handoff

Status: current implementation prompt for the local presentation website. This
doc is intentionally retained. Use it before changing the frontend; use
`docs/presentation_bundle_developer_handoff.md` only for broader bundle
background.

Build or extend a local presentation website that reads the extracted SpeechAudioLM presentation bundle as its database. Do not redesign the data exporter. Treat the bundle as read-only, use relative media paths, and keep visual styling decisions separate from the data architecture below.

## Project Purpose

SpeechAudioLM studies bidirectional generation between environmental audio and speech through cascaded baselines:

- Audio to speech: environmental audio -> audio caption -> text transformation -> speech audio.
- Speech to audio: speech audio -> speech transcription -> text transformation -> environmental audio.
- Text to text is shared by both directions.

The website exists to inspect individual samples, hear references and generated outputs, compare model combinations stage by stage, and explain aggregate and per-sample evaluation results. It replaces manually assembled slides and must remain scalable as more samples, lanes, models, and judge metrics are added.

## Bundle Contract

Start with `bundle.json`. Schema version is `presentation_bundle_v2`.

- `sample_ids.json`: ordered sample IDs shared across every exported lane.
- `samples.json`: canonical sample metadata and dataset references.
- `media_index.json`: explicit A2S and S2A roles for input media, paired reference output media, input reference text, and target reference text.
- `lanes.json`: lane, pipeline, task, stage definitions, models, and inspect file.
- `inspect/<task>/<lane>.inspect.json`: selected samples with complete intermediate stage history and portable media paths.
- `metric_definitions.json`: metric display metadata and renderer type.
- `metric_runs.jsonl`: one record per metric execution/configuration; join with `metric_run_id`.
- `metric_summaries.jsonl`: aggregate results and provenance for each metric run.
- `metric_values.jsonl`: selected per-sample values and rich details.
- `metric_coverage.json`: stable expected task/lane/metric matrix. Entries are `complete` or `in_progress`; unavailable entries have `summary: null`, `values: null`, and no fake scores.
- `metrics.jsonl`, `metric_sources.json`, `metric_artifacts.jsonl`: compatibility/raw evaluation exports; prefer the normalized v2 files for UI work.
- `media/references/...`: paired dataset audio and speech.
- `media/generated/<task>/<lane>/...`: generated outputs.

Primary joins:

- Join samples, media, inspect rows, and metric values by `sample_id`.
- Join metric values and summaries to metric runs by `metric_run_id`.
- Join metric runs to definitions by `metric_id`.
- Use `metric_coverage.json` to render metrics that are expected but not yet available. Follow `metric_run_ids` only for completed coverage entries.
- Select a cascaded model path using `lane` and `task`.

Do not infer reference/input roles from filenames. Use `media_index.json`. Do not infer G-Eval lanes from judge rows; use the explicit `lane` in `metric_runs.jsonl`.

## Information Architecture

Use four main sections:

1. **Showcase**
   Present 10 handpicked samples, ideally two from each of five highly represented audio pools. Each sample opens an A2S lane and an S2A lane. Show dataset input, dataset paired reference output, reference text, every generated intermediate text, and generated final audio.

2. **Pipeline Explorer**
   Let the user choose direction, sample, and compatible model lane. Render the cascade as ordered stages from the inspect file. Support side-by-side lane comparison without duplicating reference media. Audio controls must be clearly labeled as input, paired dataset reference, or generated output.

3. **Evaluation Lab**
   Use stage tabs: `Audio to Text`, `Speech to Text`, `Text to Text`, `Speech Generation`, and `Audio Generation`. Filter by lane/model combination. Keep corpus summaries separate from individual sample inspection.

4. **Dataset**
   Browse/search the exported samples, audio pool/class, reference text, and paired media. Link into Pipeline Explorer and Evaluation Lab with the selected sample preserved.

## Standard Metric Rendering

For per-sample scalar metrics show count, missing/error count, mean, median, standard deviation, min, max, and available percentiles. Do not emphasize mode for continuous metrics. For corpus-only metrics show the score, evaluated count, scope, completeness, and provenance.

Every metric name should have a tooltip or info panel containing: meaning, stage/task, expected range, whether higher/lower is better, per-sample versus corpus level, reference requirements, backend/model, and caveats. If definition metadata is absent, show `Unknown` rather than inventing it.

Render `in_progress` coverage entries as disabled metric cards with an explicit pending badge. Their summary and per-sample panels must show "Not available yet". Never synthesize placeholder values, distributions, or rankings.

The current authoritative T2A tree exposes a generic `clap_score_t2a`. Distinct target-caption and prompt-adherence CLAPScore entries remain expected/in-progress until distinct authoritative outputs exist; do not display the generic score under both names.

## NLI Renderer

For `renderer == "nli"`, use `metric_values[].details` and display:

- Input text and generated text prominently.
- Forward entailment, neutral, and contradiction probabilities.
- Reverse entailment, neutral, and contradiction probabilities.
- Maximum contradiction probability and predicted label.
- NLI model and sample/pair identifiers in expandable provenance.

Use probability bars or a compact matrix. NLI is directional, so never collapse forward and reverse probabilities into one unlabeled score.

## G-Eval Renderer

There are four logical groups:

- Plausibility / audio-to-speech.
- Plausibility / speech-to-audio.
- Reference-grounded / audio-to-speech.
- Reference-grounded / speech-to-audio.

For `renderer == "geval"`, provide both an aggregate view and a sample judgment view.

Aggregate view:

- Read direction-specific statistics from `metric_summaries[].statistics`.
- Preserve and expose `official_variant_summary`, which is the authoritative schema-v3 combined-variant summary.
- Show scored, abstained, and failed judgments separately.
- Show status counts, null counts, completion, aggregation policy, scoring policy, judge model, and prompt/evaluator versions.
- Visualize final-score and criterion distributions when present in the official summary.
- Never treat abstentions/null scores as zero and never silently exclude them from displayed coverage.

Sample judgment view from `metric_values[].details`:

- Input text and generated text at the top.
- Reference anchors when the variant is `reference_grounded`; hide that block for plausibility.
- Final weighted score and criterion weights.
- One criterion card per criterion with score, feedback, parse status, and repair-applied flag.
- Put raw judge response behind an expandable disclosure.
- Show diagnostics/failure modes and row status visibly.
- Show judge model, evaluator version, prompt version, scoring policy, direction, lane, sample ID, and pair ID in provenance.

Current G-Eval result rows are schema v2 while enriched aggregate summaries are schema v3. Support this intentionally; do not require both files to share a schema number.

## UX Requirements

- Preserve selected sample, direction, lane, and metric in URL/query state where practical.
- Add previous/next sample controls and direct sample-ID search.
- Lazy-load audio and large JSONL-derived records.
- Use accessible native audio controls or an accessible custom wrapper.
- Make missing, failed, abstained, and incomplete states explicit.
- Keep comparison labels fixed while scrolling so the supervisor always knows which reference/model is playing.
- Do not place every metric beside the audio comparison. Pipeline listening and evaluation analysis are separate views connected by sample links.

## Implementation Order

1. Build a typed data loader and indexes for the v2 bundle files.
2. Implement Showcase and Pipeline Explorer using inspect/media data.
3. Implement stage-filtered standard metric summaries.
4. Add dedicated NLI and G-Eval renderers.
5. Add deep links between sample pipelines and sample-level judgments.
6. Add validation tests for joins, missing media, unknown metrics, abstentions, and schema-v2-row/schema-v3-summary G-Eval compatibility.

Before changing visual style, first report the detected framework, current route/component structure, and the proposed mapping from these bundle entities to application types and pages.
