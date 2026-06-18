# Metric Scientific-Validity Audit And Fix Plan

## Scope

Audit and correct every evaluation metric except:

- G-Eval / LLM-as-judge
- NLI contradiction

Those metrics are being redesigned separately.

## Core Principles

- Follow the metric's paper and official implementation.
- Do not add workarounds merely to produce a score.
- Never silently truncate, duplicate, skip, substitute, crop, or normalize inputs.
- Any unavoidable metric/model constraint must be recorded and reported.
- Preserve every attempted row with per-sample `passed` or `error` status.
- Aggregate each metric over its passed samples and report attempted, passed, and error counts and IDs.
- Use metric status `passed`, `passed_with_errors`, or `error`; an all-failed metric has no numeric score but does not block other final results.
- Final results with disclosed sample errors are reportable; cross-model differences in failed rows are handled by manual inspection and reporting.
- Corpus metrics must run on the complete corpus.
- Per-sample metrics must preserve per-sample values.
- Canonical eval sharding is removed; temporary partial runs are diagnostics only.
- Reject invalid, non-finite, sentinel, or unaccounted-for outputs; do not silently exclude failed attempts.
- Record resolved metric/backend settings and target references in the eval run artifacts.

## Current Result Status

Current results are provisional and useful for progress reporting.

They must not be treated as final scientific results until the issues below are fixed and affected metrics are rerun.

Status annotations used below:

- **Resolved**: implemented in the active runner/worker path and covered by tests.
- **Partially resolved**: a meaningful subset is implemented, but the complete scientific contract is not.
- **Pending integration**: committed helper/policy code exists, but the active runner/worker does not use it.
- **Pending compute verification**: integrated behavior still requires a representative backend or full-corpus run.
- **Rerun required**: historical scores were produced under superseded or incomplete semantics.

Integration evidence was re-audited against repository `HEAD` on 2026-06-12. Commit `07e778f` integrates the approved final-results-with-errors policy, create-only timestamped artifacts, scientific WER/ASR-WER/dWER aggregation, paper-comparable UTMOS policy, explicit KL-PaSST pair scoring, target-caption CLAPScore semantics, attributable AAC filtering, and valid-pair staging for FAD/KAD/Inception. Structural duplicate/mismatched IDs remain blocking because score attribution would be ambiguous; ordinary sample/metric errors remain in the final manifest and do not block other scores.

Verification at `07e778f`:

- Full local suite: **206 tests passed**.
- Compute smoke passed with complete artifacts: SPIDEr, ASR-WER, dWER, UTMOS, DNSMOS, target-caption CLAPScore, and explicit-pair KL-PaSST.
- Representative compute smoke passed: MACE, AAC-CLAPSim, FENSE, FAD, and KAD. Inception Score processed two pairs but its official backend returned non-finite mean/std; a 50-sample smoke is pending. The updated runner now publishes this as a metric-level `error` artifact with no fabricated score. The updated runner now publishes this as a metric-level `error` artifact with no fabricated score.
- Full-corpus final reruns remain pending. Representative two-sample smoke proves backend integration only, not final scientific stability or corpus adequacy.
- Token-context/truncation coverage and order-sensitive AAC crop provenance remain reporting constraints; official/backend truncation is not changed to force complete-text scoring.

## Critical Findings

### SPIDEr

**Status: Partially resolved; Pending compute verification; Rerun required.**

- SPIDEr currently runs independently on shards.
- CIDEr document-frequency statistics differ between shards.
- Averaging or merging independently sharded SPIDEr results is invalid.
- SPIDEr must run once on the complete caption corpus.
- Current implementation silently truncates candidates to 100 words.
- This truncation affects approximately:
  - Audio Flamingo: 1,283 / 2,277 captions, 56.3%
  - Qwen3-Omni captioner: 2,256 / 2,277 captions, 99.1%
- The 100-word limit is a local workaround, not standard SPIDEr.

Required fix:

- [Resolved 2026-06-10] Removed the local 100-word candidate truncation from the worker and metric config. Stored captions and full candidate text now reach the official `aac-metrics` SPIDEr function after the existing declared single-line PTB-tokenizer preprocessing.
- [Policy decided 2026-06-10] Evaluation sharding will not be supported for scientific results. Final SPIDEr evaluation must run once on the complete manifest; partial runs are temporary backend diagnostics only.
- [Pending compute verification] Run SPIDEr on each complete caption corpus. If SPICE cannot process a caption, preserve that attempted row as an error and aggregate over passed rows; do not restore truncation or fallback behavior. If all rows fail, emit no numeric SPIDEr score and mark only that metric `error`.
- [Rerun required] All existing SPIDEr scores remain provisional because they were computed on shards and/or with the local 100-word cap.

### MACE And AAC-CLAPSim

**Status: Partially resolved; Pending compute verification; Rerun required.** AAC parameter propagation, deterministic invocation seed, sample reconstruction, and policy metadata are integrated in `d8dee62`. Token-overflow coverage and exact crop-offset provenance remain unresolved; the accepted official crop/repeat behavior is order-dependent.

- MS-CLAP's tokenizer specifies a maximum length but omits tokenizer truncation.
- Very long Qwen captions cause a CUDA index-out-of-bounds failure.
- Enabling truncation means only the text fitting MS-CLAP's native context is evaluated.
- Stored captions must never be modified.
- Truncation count, rate, token lengths, and model context limit must be reported.
- MS-CLAP-2023 randomly crops/repeats audio to seven seconds.
- Approximately 2,197 / 2,277 TTAS reference clips exceed seven seconds.
- Crop choice can depend on invocation and shard boundaries.

Required fix:

- Determine the scientifically defensible crop policy from the official implementation/paper.
- Record crop duration and seed.
- Make crop selection deterministic per sample if per-sample reproducibility is required.
- Report caption-token truncation explicitly.
- Pass MACE penalty/error-threshold consistently to corpus and sample calculations.
- Validate reconstructed sample MACE mean against the official corpus score.

### FENSE

**Status: Partially resolved; Pending compute verification; Rerun required.** Raw stored candidates now reach the functional FENSE API through the integrated AAC worker, but finite model contexts still truncate effective information and final truncation coverage has not been measured.

- Current AAC-metrics wrapper applies PTB preprocessing that may differ from original FENSE.
- FENSE's embedding/error models have finite token contexts and silently ignore long-caption tails.

Required fix:

- Use the original functional FENSE behavior on raw captions where appropriate.
- Record all token limits and truncation coverage.
- Decide whether extremely verbose captions are valid inputs for this metric.

### WER, ASR-WER, And dWER

**Status: Resolved in active execution; representative compute smoke passed; full rerun required.** Commit `07e778f` integrates fixed-English Whisper transcription, versioned normalized/raw edit counts, corpus WER from summed counts, macro WER as secondary output, and attributable per-sample failures.

- Current headline scores are macro averages of utterance WER.
- Standard corpus WER is total edits divided by total reference words.
- Current normalization only collapses whitespace.
- Case, punctuation, apostrophes, dashes, and number formatting count as errors.
- Whisper language/task are not explicitly fixed.

Required fix:

- Store substitutions, deletions, insertions, and reference-word counts per sample.
- Report standard corpus WER as primary.
- Keep macro utterance WER as an explicitly named secondary statistic.
- Use a declared, versioned English normalization policy.
- Report both normalized WER and optional raw orthographic WER.
- Explicitly set Whisper language to English and task to transcription.
- Rename dWER to a clearer project-specific name such as pairwise_asr_wer.

Normalization recommendation:

- Use a published/common normalizer such as Whisper EnglishTextNormalizer or a declared jiwer transform.
- Whisper EnglishTextNormalizer is already available with openai-whisper; a new backend is not necessarily required.
- Validate the chosen policy on representative TTAS examples before rerunning.

### FAD, KAD, Inception Score, And Corpus Audio Metrics

**Status: Partially resolved; Pending compute verification; Rerun required.** Input rejection and minimum-size protections are active (`bc0a474`). Reliability thresholds beyond the backend minimum and complete audio preprocessing metadata remain to be established.

- Different models may have different generation failures.
- This changes both generated and reference corpus distributions for passed rows.
- Direct corpus comparisons must disclose attempted, passed, failed, and skipped counts instead of forcing a common retained ID set.
- Invalid values such as FAD sentinel -1 or NaN may be accepted.
- Small-corpus workarounds duplicate files to force metrics to run.
- Such smoke-test scores are not scientifically valid.

Required fix:

- Report model-specific attempted, passed, failed, and skipped counts; do not silently drop rows to force cross-model equality.
- Preserve missing expected outputs as explicit error rows; aggregate the corpus metric over passed rows when its official implementation supports that retained corpus.
- [Partially resolved 2026-06-10] Backend scalar and sample-vector conversion now rejects missing, NaN, and infinity values; FAD also rejects sentinel `-1`. Strict end-to-end backend-result completeness validation remains pending.
- [Resolved 2026-06-10] Removed the FAD/Inception sample-duplication workaround. Undersized corpora now fail explicitly.
- [Resolved 2026-06-10] Removed KAD's silent short-audio padding. Audio below the declared one-second backend constraint now fails explicitly without modifying inputs.
- [Resolved 2026-06-10] Declared a minimum valid corpus size of two for the active FAD, KAD, and Inception adapters. Larger reliability recommendations still need to be reported separately from this mathematical/backend minimum.
- Record actual encoder, checkpoint, preprocessing, sample rate, channels, padding, and normalization.

### KL-PaSST

**Status: Resolved in active execution; representative compute smoke passed; full rerun required.** Commit `07e778f` calls the pinned upstream PaSST probability extractor per reference/generated pair, preserves official preprocessing, reports explicit KL direction and pair errors, and aggregates passed pairs.

- Current implementation computes paired per-example KL values and averages them.
- It is incorrectly treated as a non-composable distributional corpus metric.

Required fix:

- Emit per-pair KL values and successful-pair counts.
- Aggregate using a weighted mean.
- Explicitly label KL direction.
- Preserve every attempted pair, aggregate over passed pairs, and report attempted/passed/error counts and IDs. Partially failed results are reportable as `passed_with_errors`; all-failed KL-PaSST has status `error` and no numeric score.

### CLAPScore For T2A/S2A

**Status: Target-caption and prompt-adherence CLAPScore are now separate active metrics.** `target_caption_clap_score` uses the dataset target caption; `prompt_adherence_clap_score` uses only the exact persisted generation prompt and forbids fallback to target captions or generated text.

- Current score uses source.audio_cap.
- For S2A, that is target-caption alignment, not adherence to the actual speech/text conditioning prompt.

Required fix:

- Separate:
  - target_caption_clap_score
  - prompt_adherence_clap_score
- Persist the exact conditioning text used for generation.

### UTMOS

**Status: Resolved in active execution; representative compute smoke passed; full rerun required.** Commit `07e778f` enforces and records the paper-comparable UTMOS policy with `remove_silent_section=False` and retains attributable sample failures.

- Current backend may silently remove low-amplitude/silent sections.
- This differs from original paper-comparable behavior.

Required fix:

- Explicitly select and record silence-removal behavior.
- Prefer paper-comparable behavior unless a different policy is justified.

### Job And Result Validation

**Status: Resolved in active execution; representative compute verified.** Commit `07e778f` integrates complete-accounting validation, passed-only aggregation, create-only timestamped artifacts, final publication with disclosed errors, and metric-specific backend filtering.

- Some metric errors are silently excluded instead of being retained as attempted error rows.
- External sample metrics may be summarized as one-item corpus scores.

Required fix:

- Preserve all attempted rows and validate that each has `passed` or `error` status before marking a job complete.
- Aggregate over passed samples and assign metric status `passed`, `passed_with_errors`, or `error`.
- Report attempted, passed, and error counts and IDs. An all-failed metric has no numeric score and does not block publication of other final metric results.
- Preserve official corpus score and per-sample distribution separately.
- Never overwrite historical results.
- Store resolved metric/backend settings, target references, row counts, and metric-specific metadata without hash/fingerprint blocks.

## Evaluation Parallelism Policy

**Status: Resolved for runner scope and publication; Pending compute verification.**

Scientific metric evaluation will not be sharded. Final evaluations must run each selected metric once on the complete evaluated manifest.

Partial execution is allowed only when `temporary: true`, writes under temporary eval roots, and is labeled diagnostic/non-authoritative.

Historical shard outputs remain provisional diagnostic artifacts. The legacy eval shard aggregation helper has been removed from the target eval structure; rerun each metric once on the complete evaluated manifest for publication.

## Required Metric Documentation

**Status: Partially resolved.** The runner records backend, package, model, and broad audio-preprocessing metadata, and AAC emits detailed policy metadata. A complete per-metric registry covering every field below is still pending.

For every metric record:

- Academic definition
- Official implementation/package/version
- Metric level: sample, paired-corpus, or distributional-corpus
- Higher/lower-is-better
- Exact evaluated fields
- Reference construction
- Successful-ID policy
- Sample rate and channel policy
- Resampling and normalization
- Cropping, padding, truncation, and token limits
- Random seeds
- Minimum valid corpus size
- Shard composability
- Output validation rules
- Known limitations
- Runtime and resource measurements

## Implementation Plan

1. Keep metric definitions focused on scientific semantics and required inputs; canonical eval sharding and final-scope/common-ID enforcement are removed from the target runner.
2. Remove silent truncation, duplication, fallback, and invalid-value workarounds. [In progress: SPIDEr's local 100-word truncation is removed.] **[Partially resolved: `7460173` and `bc0a474`; remaining backend-native limits must be disclosed, not bypassed.]**
3. Record resolved eval run config and target references without manifest/config/hash fingerprint blocks.
4. Preserve complete attempted-row accounting and report failed/skipped rows directly; common-ID filtering is removed.
5. Correct WER normalization and corpus aggregation. **[Resolved in active execution by `07e778f`; representative compute smoke passed.]**
6. Correct AAC metric constraints and reporting. **[Resolved in execution code: `7b139ad`, `d8dee62`; Pending compute verification.]**
7. Correct T2A/T2S corpus metric validation. **[Integrated by `07e778f`; CLAPScore and KL-PaSST compute smoke passed, FAD/KAD/Inception smoke pending.]**
8. Add complete-accounting job/artifact validation with passed-only aggregation and non-blocking per-metric errors. **[Resolved by `07e778f`.]**
9. Add scientific-validity tests. **[Resolved for local coverage: 206 tests pass; remaining representative and full-corpus compute gates are tracked separately.]**
10. Rerun only affected metrics. **[Rerun required after integration and compute verification gates pass.]**
11. Produce a final metric coverage and validity report. **[Resolved for documentation: see `metric_scientific_validity_report.md`; scientific results remain provisional.]**

## 2026-06-12 Status Addendum

- **Common IDs:** Removed from the target eval design; report attempted, passed, failed, and skipped counts directly.
- **Artifact immutability:** Target eval runs use immutable run directories and stable filenames; mutable `latest_*` aliases are removed from eval output.
- **Result completeness:** Resolved by `07e778f`. Ordinary attributable errors produce `passed_with_errors` or `error` metric status and retain all attempted rows; structural ID corruption remains blocking.
- **Tests:** 203 repository tests passed after integration.
- **Compute status:** SPIDEr, ASR-WER, dWER, UTMOS, DNSMOS, target-caption CLAPScore, and KL-PaSST passed two-sample backend smoke with complete artifacts. MACE, AAC-CLAPSim, FENSE, FAD, and KAD passed representative smoke. Inception Score returned non-finite mean/std on two samples; a 50-sample smoke is pending. The updated runner now publishes this as a metric-level `error` artifact with no fabricated score. The updated runner now publishes this as a metric-level `error` artifact with no fabricated score.
- **Rerun readiness:** Integration is nearly ready, but full scientific reruns wait for remaining smoke checks, truncation/context coverage reporting, explicit target metadata, and full-corpus commands. Representative smoke results are temporary diagnostics.

## Fresh-Chat Agent Plan

**Status: Partially resolved.** The disjoint implementation streams produced the commits cited above. Independent scientific review and execution integration are still required.

Deploy independent implementation agents with disjoint ownership:

- Agent 1: AAC metrics, caption constraints, SPIDEr full-corpus execution
- Agent 2: WER, ASR-WER, dWER normalization and corpus aggregation
- Agent 3: T2A/T2S corpus metrics, invalid-output checks, and attempted/passed/error accounting
- Agent 4: eval run metadata, metric contracts, no-sharding enforcement, and job validation
- Agent 5: independent scientific-validity review after implementation

Do not work on G-Eval or NLI in this goal.

**Out of scope remains unchanged:** G-Eval and NLI are not assessed or planned in this audit beyond this reiteration.
