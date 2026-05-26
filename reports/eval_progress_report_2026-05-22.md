# SpeechAudio LM Evaluation Progress Report

Date: 2026-05-22

## 1. Setup Status

The evaluation framework is now set up around the TTAS dataset and baseline model outputs.

Completed setup:

- Standard baseline output structure is in place under scratch.
- Standard evaluation output structure is in place under scratch.
- Baseline outputs are written as manifests with generated file paths, source fields, status, and metadata.
- Evaluation outputs include per-sample JSONL, summary JSON/CSV, Markdown report, and run configuration.
- Metric backends are configured through repo config files rather than hard-coded choices.
- Model capabilities and task-to-metric compatibility are config-driven.
- Metric validation is metric-specific, so a run only requires the fields needed by the selected metrics.
- Sharded baseline inference is supported and shard manifests can be merged into canonical manifests.
- Heavy metric/model assets are stored in scratch rather than the git repo.
- Slurm helper scripts are available for baseline inference and metric evaluation.

Current baseline outputs available:

| Model | Mode | Current status |
|---|---:|---|
| Audio-Omni | T2A | 400 TTAS test rows generated |
| Audio-Omni | T2S | 400 TTAS test rows generated |
| Audio-Omni | A2T | 400 TTAS test rows attempted; a few bad source-audio rows recorded as errors |
| Audio-Omni | S2T | 400 TTAS test rows generated |
| Audio-Omni | S2A | 400 TTAS test rows generated |
| Audio-Omni | A2S | 400 TTAS test rows attempted; a few bad source-audio rows recorded as errors |
| TangoFlux | T2A | 400 TTAS test rows generated |
| AudioGen | T2A | Partial run only; not ready for final comparison |

Important note: these are partial baseline runs on the TTAS test split, not the full TTAS dataset and not the full test split yet.

## 2. Metric Backend Status

Working through `evaluate.py`:

- `asr_wer`
- `dwer`
- `utmos`
- `spider`
- `kad`
- `clap_score_t2a`
- `kl_passt`

Ready for retry:

- `fad`
- `inception_score`

Not ready yet:

- `mace`
- `aac_clapsim`

The issue for `mace` and `aac_clapsim` is specific to the MS-CLAP-backed AAC metric path. SPIDEr itself works, so the AAC Java/CoreNLP setup is functional.

## 3. Mode-Wise Evaluation Plan

### S2T: Speech Understanding / Speech-to-Text

Goal: evaluate generated speech transcripts.

Primary metric:

- `wer`

Why:

- WER directly checks whether the predicted transcript matches the reference speech transcript.

Backend:

- Formula-based WER implementation in the evaluation framework.

Status:

- Framework support is ready.
- Full score table is pending.

### A2T: Audio Understanding / Audio Captioning

Goal: evaluate generated environmental-audio captions.

Metrics:

- `spider`
- `mace`
- `aac_clapsim`

Why:

- SPIDEr is a standard audio-captioning metric combining caption-overlap and semantic matching signals.
- MACE is intended as a stronger reference-aware audio captioning metric.
- AAC CLAPSim measures audio-text alignment between the generated caption and reference audio.

Backends:

- `spider`: `aac-metrics`
- `mace`: planned through `aac-metrics`
- `aac_clapsim`: `aac-metrics` MS-CLAP path

Status:

- `spider` works through `evaluate.py`.
- `mace` and `aac_clapsim` still need offline MS-CLAP cache/debugging.

### T2S: Text-to-Speech / Speech Generation

Goal: evaluate generated speech audio.

Metrics:

- `asr_wer`
- `dwer`
- `utmos`

Why:

- `asr_wer` checks whether generated speech says the intended transcript.
- `dwer` checks whether generated speech preserves ASR-recognized content relative to the paired real speech.
- `utmos` estimates perceptual speech quality.

Definitions:

- `asr_wer = WER(reference_transcript, ASR(generated_speech))`
- `dwer = WER(ASR(reference_speech), ASR(generated_speech))`

Backends:

- ASR backend: OpenAI Whisper large-v3
- UTMOS backend: UTMOSv2

Status:

- All three metrics work through `evaluate.py`.

### T2A: Text-to-Audio / Environmental Audio Generation

Goal: evaluate generated environmental audio from text/audio captions.

Metrics:

- `kad`
- `clap_score_t2a`
- `kl_passt`
- `fad`
- `inception_score`

Why:

- `kad` evaluates audio generation quality through a kernel/audio distribution metric.
- `clap_score_t2a` measures text-audio alignment using LAION-CLAP.
- `kl_passt` compares generated/reference audio class distributions using PaSST.
- `fad` is the main Fréchet-style corpus-level audio generation metric.
- `inception_score` is a secondary corpus-level generation metric.

Backends:

- `kad`: `kadtk`
- `clap_score_t2a`: `stable-audio-metrics` with LAION-CLAP
- `kl_passt`: `stable-audio-metrics` with PaSST
- `fad`: `audioldm_eval`
- `inception_score`: `audioldm_eval`

Status:

- `kad`, `clap_score_t2a`, and `kl_passt` work through `evaluate.py`.
- `fad` and `inception_score` have required checkpoint assets fixed and need one final retry through `evaluate.py`.

Metric-level note:

- `fad`, `kad`, `kl_passt`, and `inception_score` are corpus-level metrics.
- `clap_score_t2a` can be computed per sample and aggregated.

### S2A: Speech-to-Audio

Goal: evaluate speech-conditioned environmental audio generation.

Current plan:

- For final generated audio, use T2A-style audio generation metrics:
  - `kad`
  - `clap_score_t2a`
  - `kl_passt`
  - `fad`
  - `inception_score`

Status:

- Baseline outputs are available for Audio-Omni on 400 TTAS test rows.
- Final-output metrics follow the T2A metric suite.
- Intermediate reasoning/transcript evaluation is still a separate design item.

### A2S: Audio-to-Speech

Goal: evaluate audio-conditioned speech generation.

Current plan:

- For final generated speech, use T2S-style generated speech metrics:
  - `asr_wer`
  - `dwer`
  - `utmos`

Status:

- Baseline outputs are available for Audio-Omni on 400 TTAS test rows.
- A few source-audio rows were too small/corrupt and are recorded as errors.
- Final-output metrics follow the T2S metric suite.

### T2T: Text-to-Text

Goal: evaluate reasoning/transformation between transcript and caption text.

Current plan:

- To be discussed.
- The most likely approach is LLM-as-a-judge, using a strong LLM with a carefully designed rubric.

Status:

- Not implemented yet.

## 4. Metric Choice Notes And Conflicts

CLAP-related metrics are intentionally separated:

- AAC captioning CLAP similarity uses the AAC/MS-CLAP path.
- T2A CLAP score uses stable-audio-metrics with LAION-CLAP.

Fréchet-style audio metric:

- V1 uses FAD, not generic FD.
- Generic FD should only be added later if direct comparison with a specific paper/tool requires it.

KL-style metric:

- Current working T2A KL metric is `kl_passt`, using PaSST through stable-audio-metrics.

Speech quality:

- UTMOSv2 is currently working.
- DNSMOS is deferred for a later backend decision.

## 5. Current Smoke Scores

These are smoke-test results only, not final benchmark results.

### S2T: Speech Understanding

| Metric | Audio-Omni | TangoFlux | AudioGen |
|---|---:|---:|---:|
| WER |  | N/A | N/A |
| Semantic equivalence |  | N/A | N/A |

### A2T: Audio Captioning

| Metric | Audio-Omni | TangoFlux | AudioGen |
|---|---:|---:|---:|
| SPIDEr | 0.0169 | N/A | N/A |
| MACE |  | N/A | N/A |
| AAC CLAPSim |  | N/A | N/A |

### T2S: Speech Generation

| Metric | Audio-Omni | TangoFlux | AudioGen |
|---|---:|---:|---:|
| ASR-WER | 0.1650 | N/A | N/A |
| dWER | 0.0925 | N/A | N/A |
| UTMOS | 1.9223 | N/A | N/A |

### T2A: Environmental Audio Generation

| Metric | Audio-Omni | TangoFlux | AudioGen |
|---|---:|---:|---:|
| KAD |  |  |  |
| CLAPScore T2A |  |  |  |
| KL PaSST |  |  |  |
| FAD |  |  |  |
| Inception Score |  |  |  |

### S2A: Speech-to-Audio

| Metric | Audio-Omni | TangoFlux | AudioGen |
|---|---:|---:|---:|
| KAD |  | N/A | N/A |
| CLAPScore T2A |  | N/A | N/A |
| KL PaSST |  | N/A | N/A |
| FAD |  | N/A | N/A |
| Inception Score |  | N/A | N/A |

### A2S: Audio-to-Speech

| Metric | Audio-Omni | TangoFlux | AudioGen |
|---|---:|---:|---:|
| ASR-WER |  | N/A | N/A |
| dWER |  | N/A | N/A |
| UTMOS |  | N/A | N/A |

## 6. Remaining Work

Immediate next steps:

- Run small-sample eval smoke for all working metrics.
- Retry `fad` and `inception_score`.
- Run full TTAS test-set baseline inference.
- Run full TTAS test-set evaluation after full baseline inference.

Deferred work:

- Fix MS-CLAP-backed AAC metrics for `mace` and `aac_clapsim`.
- Decide T2T evaluation design.
- Decide direct/cascade-specific S2A and A2S reasoning metrics.
- Decide DNSMOS backend if needed.
