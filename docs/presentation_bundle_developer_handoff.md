# SpeechAudioLM Presentation Website: Developer Handoff

Use this document as the project brief and implementation prompt for the developer agent building the local presentation website. Visual style is intentionally unspecified and should be decided separately.

## Objective

Build a local website for interactively inspecting SpeechAudioLM baseline outputs. The website is not primarily a leaderboard. Its main purpose is to let a researcher choose a dataset sample, follow both generation directions stage by stage, swap model combinations, listen to reference and generated audio, inspect intermediate text, and understand where a pipeline succeeds or fails.

The source data is a portable export bundle copied from the HPC cluster to a Windows machine. The website must read only bundle-relative files and must not depend on cluster paths, Python services, model runtimes, or network access.

## Research Project

SpeechAudioLM studies bidirectional generation between environmental audio and human speech using the paired TTAS dataset.

The two primary directions are:

- `A2S`, audio to speech: environmental audio -> audio caption/description -> generated speech text -> synthesized speech audio.
- `S2A`, speech to audio: speech audio -> speech transcript -> generated environmental audio description -> synthesized environmental audio.

The project uses modular cascades so that individual stages and model choices can be compared independently, rather than treating the whole system as one opaque model.

## Pipeline Stages And Models

### Audio To Speech (`A2S`)

1. `A2T`, environmental audio captioning
   - `audio_flamingo_next`
   - `qwen3_omni_captioner`
2. `T2T`, audio description to plausible spoken utterance
   - `llama3_3_70b_instruct`
   - `qwen3_32b`
3. `T2S`, text-to-speech
   - `f5_tts`
   - `cosyvoice3`
   - Some inspect data may also contain `audio_omni`; treat available lanes as data-driven rather than hard-coding only two TTS models.

### Speech To Audio (`S2A`)

1. `S2T`, automatic speech recognition
   - `whisper`
   - `qwen3_asr`
2. `T2T`, speech transcript to environmental audio description
   - `llama3_3_70b_instruct`
   - `qwen3_32b`
3. `T2A`, text-to-audio
   - `audiogen`
   - `tangoflux`
   - Some inspect data may also contain `audio_omni`.

The canonical validation export currently contains 32 final/intermediate inspect lanes across `T2T`, `T2A`, and `T2S`. Discover lanes from `lanes.json`; do not encode a fixed lane count in the frontend.

## Why The Bundle Exists

The raw cluster output is spread across many manifests, generated-audio directories, evaluation directories, shard outputs, and absolute Linux paths. Manually preparing presentation slides would require repeatedly copying audio, text, and scores and would not scale as models or samples are added.

The bundle converts those outputs into a self-contained database-like directory:

- the same selected validation samples are used across all required inspect lanes;
- every lane keeps its original inspect structure and full stage history;
- reference and generated audio are copied into portable relative locations;
- all available metric rows and run-level summaries are retained with provenance;
- the entire directory can be copied to Windows and used without the HPC cluster.

## Sample Selection

The exporter walks one canonical inspect manifest in dataset order. A candidate sample is selected only if it exists with `status: "success"` in every required inspect manifest. Failed or missing candidates are skipped, and scanning continues until the requested number of samples is reached.

This preserves the intuitive "first successful rows" behavior while ensuring the website does not contain missing model-output cells for the selected samples. The exact selection policy and rejected candidates are recorded in `bundle.json`.

## Bundle Layout

```text
bundle.json
sample_ids.json
samples.json
media_index.json
lanes.json
metrics.jsonl
metric_sources.json
metric_artifacts.jsonl
inspect/<task>/<lane>.inspect.json
media/references/audio/<sample>.wav
media/references/speech/<sample>.wav
media/generated/<task>/<lane>/<output>.wav
```

All paths stored inside the bundle use forward slashes and are relative to the bundle root.

## Data Model

### `bundle.json`

The top-level manifest and entry point. It contains:

- `schema_version`
- creation and source provenance
- sample-selection policy and rejected candidates
- paths to the main data files
- counts for samples, lanes, metric sources, metric rows, and media files
- `metric_sources`, the catalog of all exported evaluation runs

Load this file first. Do not infer directory contents when the catalog already describes them.

### `sample_ids.json`

An ordered array of canonical selected sample IDs. Use this order for default navigation.

### `samples.json`

One object per selected sample:

```json
{
  "sample_id": "mIJ1pJ064L8",
  "row_index": 0,
  "reference": {
    "audio_caption": "A motorcycle is started",
    "audio_file": "media/references/audio/mIJ1pJ064L8.wav",
    "speech_file": "media/references/speech/mIJ1pJ064L8.wav",
    "speech_text": "Let's see if she starts this time...",
    "audio_metadata": {
      "audio_tags": ["Vehicle", "Motorcycle", "Engine"],
      "primary_audio_class": "road_vehicle_traffic"
    }
  }
}
```

This is the canonical sample/reference table. Join everything else using `sample_id`.

### `media_index.json`

This removes ambiguity around the word "reference." TTAS provides a paired environmental-audio file and speech-audio file for every sample. Their roles depend on direction:

- In `A2S`, environmental audio is the pipeline input and paired speech audio is the reference target output.
- In `S2A`, speech audio is the pipeline input and paired environmental audio is the reference target output.

The generated model output is separate and remains under `media/generated/<task>/<lane>/`.

### `lanes.json`

One object per model lane:

```json
{
  "lane": "full_s2a_whisper_qwen32_audiogen",
  "pipeline": "S2A",
  "task": "T2A",
  "current_stage": 3,
  "inspect_file": "inspect/T2A/full_s2a_whisper_qwen32_audiogen.inspect.json",
  "stage_definitions": []
}
```

Use this as the run/model catalog. Model relationships should be derived from `pipeline`, `task`, `stage_definitions`, and the stages inside the inspect file. Lane names are stable identifiers but should not be the only source of model metadata.

### Filtered Inspect Files

Each file under `inspect/` preserves the project `manifest.inspect.json` schema. It contains lane metadata, stage definitions, a filtered summary, and `samples`.

Each inspect sample contains:

- `row_index`
- `sample_id`
- overall `status`
- paired dataset `reference`
- ordered `stages`
- `final_output` when the current lane produces final audio
- error information

Each stage contains its stage number, task, model, input, rendered prompt/messages when available, output, timing, and error. This is the primary source for rendering the pipeline trace.

Do not flatten away prompts, stage metadata, or errors. They are important for understanding model behavior.

### `metrics.jsonl`

A consolidated convenience index. Each line contains metric-source metadata and the original per-sample evaluation row under `result`:

```json
{
  "source_id": "final_validation__T2A__full_s2a_whisper_qwen32_audiogen__clap_score_t2a",
  "evaluation_scope": "final_validation",
  "task": "T2A",
  "lane": "full_s2a_whisper_qwen32_audiogen",
  "shard": null,
  "metric": "clap_score_t2a",
  "result": {
    "sample_id": "mIJ1pJ064L8",
    "status": "success",
    "metrics": {"clap_score_t2a": 0.1216},
    "predictions": {},
    "references": {},
    "errors": {}
  }
}
```

Join per-sample metrics by `sample_id`, then filter by `lane`, `task`, and `metric`.

### `metric_sources.json` And `metric_artifacts.jsonl`

The presentation bundle deliberately flattens the cluster's deeply nested evaluation directories:

- `metric_sources.json` catalogs each source, lane, task, metric, shard, row count, and available artifacts.
- `metric_artifacts.jsonl` stores each source's summary JSON, summary CSV text, run config, and report Markdown.
- `metrics.jsonl` stores the selected per-sample rows.

Some metrics are corpus/distribution metrics and do not have a meaningful independent score for every sample. For those metrics, treat `summary.json` as authoritative and use per-sample rows only for status, references, diagnostics, or generated-file association.

## Metric Categories

The exported data may include:

- `S2T`: WER-style transcription metrics.
- `A2T`: caption metrics such as SPIDEr, MACE, AAC-CLAPSim, and FENSE.
- `T2T`: NLI contradiction and G-Eval/Prometheus judgments when available.
- `T2S`: ASR-WER, pairwise/dWER-style comparison, DNSMOS, and UTMOS.
- `T2A`: CLAPScore, FAD, KAD, Inception Score, and KL-PaSST.

Do not assume all metrics have the same direction, range, scale, or aggregation semantics. Read labels and values from the metric source. The website should display raw values initially; metric normalization or "higher is better" indicators should be introduced only with an explicit metric metadata map.

`evaluation_scope` matters:

- `final_validation`: authoritative complete-run evaluation output.
- `final_validation_shards`: retained shard-level evaluation artifacts. These are useful for per-sample inspection but shard-derived corpus summaries must not be presented as authoritative scientific scores.

## Core Relationships

```text
sample_ids.json
    -> sample_id

samples.json
    sample_id -> reference text/audio and metadata

lanes.json
    lane -> pipeline/task/models -> inspect_file

inspect file
    lane + sample_id -> ordered stage inputs, prompts, outputs, errors, final media

metrics.jsonl
    lane + task + metric + sample_id -> per-sample evaluation result

metric_sources catalog
    source_id -> artifact record and evaluation scope
```

The natural frontend query is: select `sample_id` -> select pipeline -> select stage/model lane -> render inspect stage trace -> attach matching metrics -> resolve audio paths relative to the bundle root.

## Functional Requirements

The first implementation should support:

1. Browse or search the 100 samples.
2. Play paired reference environmental audio and reference speech audio.
3. Switch between `A2S` and `S2A`.
4. Select or compare model combinations at each pipeline stage.
5. Show every stage's input, prompt when useful, generated text, model, timing, and error.
6. Play generated final speech/audio.
7. Display matching per-sample metrics without hiding the raw evaluator payload.
8. Display run-level metric summaries with scope/provenance labels.
9. Handle absent metrics gracefully; missing evaluation data is not the same as a score of zero.
10. Work fully offline from a copied bundle directory.

## Implementation Guidance

- Treat the bundle as immutable input data.
- Build indexes in memory at application startup, or preprocess them into a local browser-friendly database if needed.
- Keep source objects accessible in an expandable raw JSON inspector.
- Use stable keys: `sample_id`, `lane`, `task`, `source_id`, and `metric`.
- Resolve all media URLs relative to the selected bundle root.
- Never depend on original absolute paths recorded in provenance fields.
- Do not silently merge duplicate/sharded metric sources. Preserve scope and shard identity.
- Make the data-loading layer independent from the visual design so the interface can be redesigned without changing bundle semantics.

## Developer Task

Design and implement the local data-loading architecture and website around this bundle. Start by validating `bundle.json`, constructing typed indexes for samples, lanes, inspect records, metric sources, and per-sample metrics, and proving that one selected sample can be traced through both complete pipelines with playable reference/generated media. After the data relationships are correct, develop the visual presentation separately with the user.
