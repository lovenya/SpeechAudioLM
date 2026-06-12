# Metric Scientific-Validity Report

## Scope And Verdict

This report audits repository `HEAD` on 2026-06-12 using tracked code, tests, configuration, and git history. It covers evaluation metrics except G-Eval and NLI, which remain explicitly out of scope.

The active stack now implements the approved scientific and failure-accounting policies in commit `07e778f`. Every attempted row is retained; metrics aggregate passed samples and report attempted/passed/error counts and IDs; ordinary sample failures do not suppress other final scores. Existing historical results remain provisional, and full-corpus reruns are still required after the remaining backend smoke and reporting gates.

Status meanings:

- **Resolved**: active execution implements the policy and tests cover it.
- **Partially resolved**: only part of the required contract is active.
- **Pending integration**: committed helper/policy exists but is not called by the active path.
- **Pending compute verification**: representative backend/full-corpus execution has not validated the behavior.
- **Rerun required**: historical values used superseded or incomplete semantics.

## Commit Evidence

| Commit | Evidence |
|---|---|
| `7460173` | Removed the local SPIDEr 100-word truncation. |
| `23fe5ce` | Added versioned normalization, edit counts, corpus WER, and macro WER primitives. |
| `7eac6be` | Added fail-closed evaluation scope, common-ID, and provenance contracts. |
| `ef1c39a` | Marked shard aggregation diagnostic-only and invalidated shard-derived SPIDEr. |
| `bc0a474` | Rejected non-finite/sentinel corpus outputs, sample duplication, and KAD short-audio padding; declared minimum corpus sizes. |
| `6d5de6b` | Integrated final/common-ID scope and stronger manifest/config/policy/row/git provenance into the runner. |
| `7b139ad` | Added explicit AAC caption, MS-CLAP, crop/repetition, seed, and reconstruction semantics. |
| `d8dee62` | Integrated AAC semantics into `scripts/eval_backend_worker.py`. |
| `e0b19e2` | Added explicit Whisper/WER backend helpers; not integrated into the active worker. |
| `0f2f532` | Added strict final-result completeness validation; not called by the runner. |
| `85be05f` | Added strict paired KL-PaSST semantics; not integrated into the active worker. |
| `292dea0` | Added separate target-caption and prompt-adherence CLAPScore semantics; not integrated into the active worker. |
| `de44b8c` | Added paper-comparable UTMOS no-silence-removal policy; not integrated into the active worker. |
| `5480f53` | Tests repository-root/backend `PYTHONPATH` construction; this is import coverage, not backend compute verification. |
| `4230dae` | Added atomic create-only artifact writers; the active evaluation runner does not use them. |
| `74b095e` | Added create-only common-successful-ID artifact construction with canonical first-manifest ordering; runner consumption remains operator-managed. |
| `07e778f` | Integrated final-results-with-errors validation, create-only artifacts, WER/UTMOS policies, explicit KL pairs, target-caption CLAPScore, AAC filtering, and valid-pair corpus-audio staging. |

## Current Worktree Integration

**Committed and locally verified.**

Commit `07e778f` contains the active metrics integration. The full repository test suite passes **206 tests**. Representative compute smoke with complete artifacts passes for SPIDEr, ASR-WER, dWER, UTMOS, DNSMOS, target-caption CLAPScore, and explicit-pair KL-PaSST. MACE, AAC-CLAPSim, FENSE, FAD, and KAD passed representative smoke. Inception Score remains under larger-corpus smoke verification after its official backend returned non-finite mean/std for two samples. Full-corpus scientific reruns have not started.

## Cross-Metric Contract

### Evaluation Scope, IDs, And Sharding

**Resolved in the runner; Pending compute verification.**

Final mode requires an explicit evaluation-scope file, no slicing, and exact equality among expected, selected, and declared ID sets. The final manifest must preserve every attempted row with per-sample `passed` or `error` status. Missing required files or fields become explicit row errors rather than silent exclusions. Only final runs may update `latest_*`; `backend_smoke` writes to a separate non-authoritative namespace.

Scientific evaluation must not be sharded. SPIDEr/CIDEr document frequencies, distributional audio metrics, and order-dependent AAC audio crops cannot be reconstructed reliably from independently invoked shards. `ef1c39a` makes historical shard merges diagnostic-only and removes shard-derived SPIDEr values. Partial runs are backend diagnostics, not publishable results.

Final results with disclosed errors are reportable. For strict cross-model comparison, the eventual goal remains recomputation on one common retained set after taking the union of failures across compared models. A sanity/recompute script for that workflow is a later TODO and is not implemented now. `74b095e` provides a create-only common-ID builder, but it does not implement automatic union-of-failures recomputation.

### Provenance

**Resolved for core runner inputs; Partially resolved for prompt-specific semantics.**

`6d5de6b` records exact manifest bytes, canonical combined configuration, evaluation policy, selected rows, selected/common ID sets, git commit, tracked worktree diff, and relevant untracked source/config files. Timestamped outputs preserve history.

Final provenance must also identify metric package/version, model/checkpoint, evaluated fields, reference construction, sample rate, channels, resampling, normalization, duration handling, token limits, seeds, and runtime resources. Backend metadata records several fields, and integrated AAC metrics add detailed policy metadata. Exact conditioning text remains mandatory for prompt-adherence CLAPScore and must be present in the generation manifest.

### Result Completeness

**Resolved in active execution; full-corpus verification pending.**

`0f2f532` validates expected IDs, duplicate/missing rows, row status/errors, sample coverage, corpus metric presence, finite values, and structured corpus counts. Its earlier fail-on-any-error contract is superseded by the approved policy and must be updated during integration. Committed `HEAD` does not call it before writing summaries or updating final aliases. The current uncommitted runner calls it before publication, but that integration is incomplete and presently breaks runner tests, so the scientific status remains Pending integration.

The final gate must require complete accounting, not universal sample success. Every attempted row must have `passed` or `error` status; each metric aggregates only passed samples and reports mandatory attempted, passed, and error counts and IDs. Metric status is `passed`, `passed_with_errors`, or `error`. An all-failed metric has no numeric score but does not block publication of other final metric results. Unaccounted-for rows, malformed structures, non-finite values, sentinels, or misalignment remain invalid.

### Artifact Immutability

**Resolved for timestamped artifacts.**

`4230dae` adds atomic create-only writers that reject existing destinations and non-finite JSON values. `74b095e` uses them for common-ID artifacts. Committed `HEAD` still writes evaluation outputs through `evaluation.io` and overwriteable `latest_*` aliases. The current uncommitted runner has begun using create-only timestamped result writers, but targeted tests fail before this path is valid; mutable aliases must still update only after successful completeness validation.

## Input Transformation Policy

No local workaround may crop, pad, repeat, truncate, duplicate, normalize, resample, downmix, remove silence, or substitute inputs merely to obtain a score. Backend-native behavior is acceptable only when it belongs to the selected official implementation, is scientifically justified and versioned, and is reported with coverage and seed information.

### Truncation And Token Contexts

- **SPIDEr:** local 100-word truncation is removed by `7460173`. Full candidates receive declared single-line whitespace cleanup and official PTB processing. Full-corpus compute verification is pending.
- **MACE/AAC-CLAPSim:** MS-CLAP-2023 has a 77-token text context; `msclap` 1.3.4 does not explicitly request tokenizer truncation. Long-caption behavior and overflow counts must be measured before final use.
- **FENSE:** raw stored captions reach functional FENSE, but its fluency checker uses a 64-token truncating tokenizer and its sentence embedding model has a finite context. Effective complete-caption evaluation is false.
- **CLAPScore:** the committed helper documents a 77-token LAION-CLAP context with truncation. The active worker truncates to 77 but does not report semantic variants or coverage.
- **WER:** empty normalized references must fail rather than be skipped or assigned a sentinel.

Extremely verbose captions may be scored only as explicitly context-limited observations. They must not be described as complete-caption measurements when backend models ignore tails.

### Cropping, Padding, And Repetition

- **MACE/AAC-CLAPSim:** AAC Metrics 0.5.5 with MS-CLAP-2023 resamples to 44.1 kHz, flattens channels, repeats short audio from the start to seven seconds, and selects one contiguous random crop for long audio. The project accepts this official behavior with seed `1337`. The seed is set once per ordered invocation, so crops are order-dependent, not per-sample invariant, and exact offsets are not recorded.
- **KAD:** audio shorter than the declared one-second backend constraint is rejected. `bc0a474` removed silent padding.
- **FAD/Inception:** files are not duplicated to satisfy minimum-size behavior. Undersized corpora fail.
- **UTMOS:** the committed policy preserves official model segment selection but disables silence removal for paper comparability. This is Pending integration.
- **Other metrics:** no project-level crop, pad, repeat, or duplication is authorized without a documented metric-specific contract.

### Normalization, Resampling, And Channels

- **WER:** `english_conservative@1` applies Unicode NFKC, case folding, punctuation-to-space, and whitespace collapse without lexical number, abbreviation, or contraction rewriting. `raw_orthographic@1` preserves orthography except whitespace. Both are Pending integration.
- **SPIDEr:** local preprocessing only collapses Unicode whitespace to one line; official PTB tokenization owns lowercasing and punctuation behavior.
- **FENSE:** raw candidate strings are passed to the functional API; backend normalization/tokenization still applies and must be disclosed.
- **AAC audio metrics:** 44.1 kHz and channel flattening are documented in integrated policy metadata. Flattening is not conventional mono averaging and is a known implementation characteristic.
- **CLAPScore:** the active worker loads mono at 48 kHz, peak-normalizes to -1 dB, and performs int16 round-trip conversion. These transformations are not yet governed by the committed semantic helper in execution.
- **FAD/Inception:** configured at 16 kHz with PANNs/CNN14; exact backend resampling, channel, normalization, and duration behavior requires runtime confirmation.
- **KAD:** configured PANNs Wavegram-Logmel sample rate, channels, and normalization must be captured from the actual environment.
- **KL-PaSST:** configured at 32 kHz with backend-default resampling; exact paired preprocessing must be recorded when strict helpers are integrated.
- **UTMOS:** configured at 16 kHz; channel/resampling and segment behavior must be emitted by the integrated policy.

### Silence And Seeds

Silent or low-amplitude material must not be removed unless the selected policy explicitly requires it. The UTMOS paper-comparable policy sets silence removal to false, but active execution does not yet enforce that choice. Unexpected all-silent or invalid audio must fail rather than disappear from the evaluated set.

AAC audio metrics use seed `1337`, set once per complete ordered invocation. This stabilizes an identical rerun but is not invariant to reordered or partitioned corpora. UTMOS random segment behavior requires a controlled NumPy seed and recorded execution provenance when integrated. Any backend stochasticity must use fixed, reported seeds.

## Metric Findings

### SPIDEr

**Partially resolved; Pending compute verification; Rerun required.**

Historical scores are invalid for final comparison because they were computed on shards and/or with the local 100-word cap, affecting approximately 1,283/2,277 Audio Flamingo captions and 2,256/2,277 Qwen3-Omni captions. The cap is removed and final sharding is disallowed. Rerun once per complete declared caption corpus. Any SPICE failure must be retained as an error row; aggregate over passed rows without truncation or fallback. If all rows fail, SPIDEr has status `error` and no numeric score.

### MACE And AAC-CLAPSim

**Partially resolved; Pending compute verification; Rerun required.**

`d8dee62` integrates penalty/error-threshold propagation, fixed seed, official corpus score preservation, complete sentence-vector validation, per-sample MACE reconstruction, and equality of reconstructed mean with the official scalar. Official seven-second crop/repetition behavior is accepted and disclosed.

Remaining requirements are full-corpus backend verification, observed caption overflow reporting, confirmation of final ordered-ID repeatability, and disclosure that exact crop offsets are unavailable. Historical scores require reruns because prior shard boundaries could alter crops.

### FENSE

**Partially resolved; Pending compute verification; Rerun required.**

The active AAC worker calls functional FENSE with raw candidates and complete sample-vector validation. It records penalty/error threshold and verifies the corpus mean. The metric still has finite embedding and 64-token fluency-checker contexts. Final use requires measured truncation coverage and an explicit limitation that verbose-caption tails may not affect the score.

### WER, ASR-WER, And dWER

**Resolved in active execution; representative compute smoke passed; Rerun required.**

Committed `HEAD` still computes one scalar per utterance with whitespace-only normalization; summaries are macro means and Whisper language/task are not fixed. The current uncommitted worker calls the committed WER backend, emits edit counts plus corpus/macro variants, and fixes `language="en", task="transcribe"`, but its targeted integration test set is not yet clean. Treat this as in-progress integration, not resolved behavior.

Integrate `e0b19e2`, emit per-sample sufficient statistics, make corpus WER primary, retain macro WER under an explicit name, and rename `dwer` to a project-specific paired-ASR label while preserving backward provenance. Validate representative TTAS punctuation, apostrophe, dash, case, number, and empty-reference examples before rerunning.

### FAD, KAD, And Inception Score

**Partially resolved; Pending compute verification; Rerun required.**

`bc0a474` rejects NaN/infinity, FAD `-1`, undersized corpora, and KAD clips below one second without modifying data. The active mathematical/backend minimum is two files for FAD, KAD, and Inception Score. Two is not a reliability recommendation; final reports must state a scientifically justified corpus size and uncertainty/stability limitations.

Final runs require exact encoder/checkpoint/preprocessing provenance and complete attempted-row accounting. Results with disclosed errors are reportable, but direct cross-model claims should later be recomputed on one common retained set using the planned union-of-failures sanity/recompute workflow. Historical smoke scores produced by duplication or silent exclusion remain non-scientific.

### KL-PaSST

**Resolved in active execution; representative compute smoke passed; Rerun required.**

`85be05f` defines paired KL records, explicit direction, complete expected-ID validation, failed-pair rejection, and weighted aggregation. The active worker still returns `passt_kld(..., collect="mean")` as one corpus scalar. Integrate per-pair output and validation, aggregate over passed pairs, and report mandatory attempted/passed/error counts and IDs. Partial failures produce `passed_with_errors`; an all-failed metric produces `error` with no numeric score and does not block other final results.

### CLAPScore

**Target-caption semantics resolved; representative compute smoke passed; prompt-adherence variant pending.**

`292dea0` separates `target_caption_clap_score` from `prompt_adherence_clap_score`, validates exact IDs, and documents model constraints. The active worker exposes `target_caption_clap_score` for `source.audio_cap` target-caption alignment and `prompt_adherence_clap_score` for exact persisted conditioning-prompt adherence. The prompt metric does not fall back to the target caption or generated text.

Integrate both semantics, require the exact stored generation prompt for prompt adherence, retain target-caption alignment only under its own name, and report 77-token truncation coverage plus mono/resampling/peak-normalization/int16 conversion.

### UTMOS

**Resolved in active execution; representative compute smoke passed; Rerun required.**

`de44b8c` selects paper-comparable `silence_removal=False`, documents official random segmentation, and validates complete finite sample records. Committed `HEAD` does not pass this policy to `model.predict`. The current uncommitted worker passes the policy and validates complete records, but integration tests are not clean; controlled runtime RNG provenance and backend compute verification remain required before rerunning.

### Job And Artifact Validation

**Resolved in active execution; full-corpus verification pending.**

Backend nonzero exits already fail, and final artifacts are timestamped. However, the `0f2f532` validator must be revised and integrated before final summaries and `latest_*` updates. Row errors are reportable when every attempted row is retained and counts/IDs are complete. The job must still fail on unaccounted-for rows, unexpected/duplicate IDs, malformed structures, or non-finite values. An all-failed metric has no numeric score but does not block other final results. Historical artifacts must never be overwritten or relabeled as final.

## Minimum Corpus Sizes

The active hard minimum of two passed files for FAD, KAD, and Inception Score only prevents known backend or mathematical failures. It does not establish stable scientific estimation. Before final runs, each distributional metric needs a documented target corpus size, sensitivity or bootstrap analysis where practical, and complete attempted/passed/error accounting. SPIDEr requires one invocation over the entire declared caption corpus because its CIDEr statistics are corpus-dependent. Sample metrics must retain every attempted row, not merely successful samples.

## Tests And Compute Verification

Focused tests exist for:

- SPIDEr no-local-truncation behavior.
- Evaluation modes, common-ID equality, provenance hashes, and final-only publication.
- Diagnostic shard aggregation and SPIDEr invalidation.
- FAD/KAD/Inception invalid-value, minimum-size, duplication, and padding rejection.
- AAC preprocessing, crop/repetition metadata, reconstruction, and active worker integration.
- WER normalization/counting/aggregation and standalone Whisper backend semantics.
- Standalone final-result completeness validation.
- Standalone KL-PaSST, CLAPScore, and UTMOS policies/validators.
- Create-only artifact writers and immutable common-successful-ID construction.
- Repository-root/backend `PYTHONPATH` construction.

After commit `07e778f`, the full repository suite passes 206 tests. These tests cover integration and policy behavior but do not substitute for full-corpus backend execution.

These tests establish local semantics, not backend scientific validity. Required compute verification remains:

1. Run each integrated AAC metric on representative long captions, short/long audio, and then the complete ordered common corpus.
2. Measure token overflow/truncation coverage for MS-CLAP, FENSE, and CLAPScore.
3. Confirm identical-order/seed repeatability and document order sensitivity.
4. Integrate and execute WER, final-result validation, KL-PaSST, CLAPScore, and UTMOS paths.
5. Verify FAD/KAD/Inception preprocessing and stable finite outputs on the final common corpus.
6. Confirm complete attempted-row accounting, passed-only aggregation, correct metric statuses, and isolation of all-failed metrics without blocking other final results.

Current integrated two-sample GPU smoke passes for SPIDEr, ASR-WER, dWER, UTMOS, DNSMOS, target-caption CLAPScore, and KL-PaSST, each with complete backend-smoke artifacts. MACE, AAC-CLAPSim, FENSE, FAD, and KAD now pass representative smoke. Inception Score remains pending at a larger smoke size because the official backend returned non-finite mean/std for two samples. No full-corpus scientific rerun has been performed under `07e778f`.

## Historical Results

All current historical results are provisional diagnostics. They may support progress reporting but not final model claims.

- Sharded SPIDEr and any shard-averaged corpus metric are invalid.
- SPIDEr values produced with the local 100-word cap require replacement.
- MACE/AAC-CLAPSim values may depend on prior invocation order or shard boundaries.
- Legacy WER/ASR-WER/dWER values are macro, whitespace-normalized results rather than the required corpus-WER contract.
- Corpus audio results based on differing successful-ID subsets, duplicated files, padded KAD clips, sentinels, or non-finite values are invalid.
- Legacy KL-PaSST lacks required per-pair completeness and direction evidence.
- Legacy S2A CLAPScore does not establish prompt adherence.
- Legacy UTMOS does not establish paper-comparable silence handling.

Preserve these artifacts with original provenance. Do not overwrite them or promote them to final.

## Rerun Readiness

The repository is ready for final reruns only when all applicable gates pass:

- One explicit, versioned evaluation-scope file exists for each final run.
- Final mode uses the complete unsliced attempted-row manifest in stable ID order.
- Every attempted row has `passed` or `error` status, and every metric reports attempted/passed/error counts and IDs.
- Every attempted ID records either the required output/reference availability and passed result or an explicit error.
- WER backend semantics are integrated.
- Final-result complete-accounting validation runs before publication.
- Timestamped result artifacts use create-only publication; mutable `latest_*` aliases are updated only after all final gates pass.
- KL-PaSST, CLAPScore, and UTMOS helpers are integrated when selected.
- Token-context coverage and every crop, pad, repeat, resample, channel, normalization, and silence policy are emitted.
- Package, checkpoint, config, policy, manifest, row, ID-set, prompt, git commit, worktree, seed, and runtime provenance are retained.
- Representative backend verification passes without fallback, duplication, substitution, sentinel acceptance, or silent exclusion.
- Each metric aggregates finite passed outputs, uses status `passed`, `passed_with_errors`, or `error`, and emits no numeric score when all attempts fail.

After these gates, rerun only affected metrics on the declared evaluation scope. Final results with disclosed errors are reportable. Cross-model common-retained-set recomputation after taking the union of failures remains a later sanity/recompute TODO. Until the applicable gates pass, the correct status is **Rerun required**, not final scientific validity.
