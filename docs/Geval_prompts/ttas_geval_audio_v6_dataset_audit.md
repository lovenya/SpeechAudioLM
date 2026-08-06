# TTAS G-Eval Audio v6 Training-Manifest Audit

Status: final dataset-cleaning decision for model training.

## Source Dataset And Fixed Splits

- Dataset root: `/scratch/l/lovenya/salm_migration_clean/datasets/ttas/extracted`
- Environmental audio: `<dataset_root>/audio/<sample_id>.wav`
- Speech audio: `<dataset_root>/speech/<sample_id>.wav`
- Canonical manifest: `data/manifests/ttas_clean_normalized.jsonl`
- Source splits: `data/splits/ttas_{train,val,test}.jsonl`
- Split counts: train `18,226`, val `2,277`, test `2,277`; total `22,780`
- Split method: seed `1337`, `80/10/10`, stratified by speaker gender and bucketed speaker emotion

The original canonical manifest and split files remain unchanged. Clean-v2
only removes rows within each existing split and preserves source row order.

## Final Direction-Specific Policy

A sample is excluded from a direction if any of that direction's G-Eval audio
v6 results satisfies: final score is null; final score is `<= 3`;
`scenario_plausibility <= 2`; or `target_modality_success == 1`.

Filtering is direction-specific: an A2S failure does not remove the S2A row,
and an S2A failure does not remove the A2S row. Train, validation, and test all
use the same cleaning rule.

## Final Manifests

| Direction | Split | Original | Kept | Excluded | Final manifest |
|---|---:|---:|---:|---:|---|
| A2S | train | 18,226 | 17,596 | 630 | `data/manifests/clean_v2/a2s/ttas_train.jsonl` |
| A2S | val | 2,277 | 2,206 | 71 | `data/manifests/clean_v2/a2s/ttas_val.jsonl` |
| A2S | test | 2,277 | 2,197 | 80 | `data/manifests/clean_v2/a2s/ttas_test.jsonl` |
| S2A | train | 18,226 | 15,033 | 3,193 | `data/manifests/clean_v2/s2a/ttas_train.jsonl` |
| S2A | val | 2,277 | 1,887 | 390 | `data/manifests/clean_v2/s2a/ttas_val.jsonl` |
| S2A | test | 2,277 | 1,878 | 399 | `data/manifests/clean_v2/s2a/ttas_test.jsonl` |

Totals retained are `21,999 / 22,780` (`96.5716%`) for A2S and
`18,798 / 22,780` (`82.5198%`) for S2A. Exclusion-reason counts overlap.

## Score Provenance

- Prompt version: `geval_audio_plausibility_v6`
- Scoring policy: `geval_weighted_mean_v1_no_caps`
- Judge: Qwen3-Omni-30B-A3B-Instruct
- Source run group: `geval_audio_v6_plausibility_20260707_0017`
- Retry/merged group: `geval_audio_v6_plausibility_20260707_0017_resume_20260707_1010`
- Merged results root, relative to `results_root`: `evals/shard_manifests/geval_audio_v6_plausibility/geval_audio_v6_plausibility_20260707_0017_resume_20260707_1010/merged`

Each scope is `dataset_{train,val,test}_{a2s,s2a}/backend_output.jsonl`. The
builder rejects prompt/scoring-version mismatches and requires exact
source/score sample membership.

## Reproducibility And Audit Artifacts

- Policy config: `configs/ttas_geval_audio_v6_clean_v2.json`
- Builder: `scripts/build_clean_v2_manifests.py`
- Eligibility ledger: `data/quality/ttas_geval_audio_v6_eligibility_v2.jsonl`
- Summary and hashes: `data/quality/ttas_geval_audio_v6_clean_v2_summary.json`

The ledger has one row per original sample and both direction decisions. It is
an audit artifact, not a training manifest. There is no unresolved set: null
scores are excluded. Excluded samples remain in the unchanged source splits;
no source data or media was deleted.

```bash
python scripts/build_clean_v2_manifests.py --overwrite
```
