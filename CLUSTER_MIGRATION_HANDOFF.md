# Cluster Migration Handoff

Last updated: 2026-05-26

This handoff is based on the current repo, this Codex thread, and the migration
summary supplied by the user on 2026-05-26. I do not have direct access to other
Codex threads. If another thread has newer inference or metric status, append it
here or reconcile before the move.

## Goal

Move SpeechAudioLM from the current Alliance/Rorqual setup to a new cluster with
enough context for a new coding agent to:

- locate and transfer the repo, data, model assets, metric checkpoints, external
  source trees, and generated outputs;
- recreate model and metric environments under the new cluster's module/Python
  constraints;
- update path configs cleanly instead of hard-coding old Rorqual paths;
- resume baseline inference and evaluation smoke checks.

## Current Repo

Current repo path:

```text
/lustre09/project/6090520/lovenya/SpeechAudioLM
/project/rrg-csubakan/lovenya/SpeechAudioLM
```

Those resolve to the same working tree on this cluster.

Recent important commits:

```text
fc7d8b3 fix: run utmos hf models offline
4241051 fix: set aac clapsim batch size
bdb2065 fix: configure metric cache roots
42d26d4 fix: allow shard merge script direct execution
821139e [docs]: update eval smoke status
8d78d60 chore: add slurm resource monitoring
b1642f1 chore: add sharded eval and baseline helpers
```

Git intentionally ignores venvs, data, audio, checkpoints, generated results,
and logs. Migration handoff/context Markdown files are now tracked in git for
easier migration, even though the older `.gitignore` still has broad Markdown
ignore patterns. After cloning on the new cluster, verify these files are
present:

```text
CLUSTER_MIGRATION_HANDOFF.md
NEW_CLUSTER_ENV_SETUP.md
CONTEXT.md
HPC_BEST_PRACTICES.md
PROGRESS_LOG.md
CHANGELOG.md
reports/eval_progress_report_2026-05-22.md
```

If any of those files are missing after clone, copy them separately from the old
cluster before continuing setup.

## Codex Agent Knowledge Transfer

The repo-level agent instructions are tracked here:

```text
AGENTS.md
```

The most important local Codex skill used for this project has been vendored
into this repository for clone-only migration:

```text
codex/skills/update_docs/SKILL.md
codex/skills/cluster-migration/SKILL.md
```

After cloning on the new cluster, install it into the new Codex home if Codex
does not auto-discover repo-local skills:

```text
~/.codex/skills/update_docs/SKILL.md
```

The old Rorqual source path was:

```text
/home/lovenya/.codex/skills/update_docs/SKILL.md
```

System skills were also present on Rorqual, but they are generic Codex-managed
skills rather than SpeechAudioLM-specific project knowledge:

```text
/home/lovenya/.codex/skills/.system/imagegen/SKILL.md
/home/lovenya/.codex/skills/.system/openai-docs/SKILL.md
/home/lovenya/.codex/skills/.system/plugin-creator/SKILL.md
/home/lovenya/.codex/skills/.system/skill-creator/SKILL.md
/home/lovenya/.codex/skills/.system/skill-installer/SKILL.md
```

If the new Codex install already provides system skills, only the repo-vendored
custom skills need manual installation. The new agent should read `AGENTS.md`,
`CONTEXT.md`, this handoff, `NEW_CLUSTER_ENV_SETUP.md`, and
`docs/eval_backend_setup.md` before changing environments or paths.

## Resource Map

### Dataset

Canonical dataset root on Rorqual/project:

```text
/project/rrg-csubakan/lovenya/datasets/ttas/extracted
```

Current best clean source manifest:

```text
/project/rrg-csubakan/lovenya/datasets/ttas/extracted/ttas/22k_samples_clean_manual_meta_clean.jsonl
```

Repo-tracked normalized metadata and splits:

```text
data/manifests/ttas_clean_normalized.jsonl
data/splits/ttas_train.jsonl
data/splits/ttas_val.jsonl
data/splits/ttas_test.jsonl
configs/ttas_dataset.json
```

Scratch copy observed:

```text
/scratch/lovenya/SpeechAudioLM/datasets/ttas/extracted
```

The manifest/splits use relative `audio_file` and `speech_file` paths. On the
new cluster, update `configs/ttas_dataset.json` so those resolve against the new
dataset root.

### Baseline Model Assets

Current model/checkpoint paths outside git:

```text
/project/rrg-csubakan/lovenya/models/audiogen-medium
/project/rrg-csubakan/lovenya/models/tangoflux
/project/rrg-csubakan/lovenya/models/qwen3-omni-30b-a3b-instruct
/project/rrg-csubakan/lovenya/models/audio-omni
/project/rrg-csubakan/lovenya/models/qwen2.5-omni-3b
/project/rrg-csubakan/lovenya/models/hf_cache
```

Other model folders observed, not all part of the current baseline path:

```text
/project/rrg-csubakan/lovenya/models/cosyvoice-ttsfrd
/project/rrg-csubakan/lovenya/models/cosyvoice2-0.5b
```

External Audio-Omni source tree:

```text
/project/rrg-csubakan/lovenya/src/Audio-Omni
```

Audio-Omni currently uses `PYTHONPATH` rather than a reliable editable install.
The baseline code expects the source tree to be available and uses local
patching/helpers around upstream Audio-Omni and Qwen2.5 behavior.

### Metric Assets And Toolkits

Metric caches/checkpoints are mostly on scratch:

```text
/scratch/lovenya/SpeechAudioLM/models/metrics/whisper
/scratch/lovenya/SpeechAudioLM/cache/metrics/aac_metrics
/scratch/lovenya/SpeechAudioLM/cache/metrics/audioldm_eval
/scratch/lovenya/SpeechAudioLM/cache/metrics/kadtk
/scratch/lovenya/SpeechAudioLM/cache/metrics/stable_audio_metrics
/scratch/lovenya/SpeechAudioLM/cache/metrics/utmosv2
/scratch/lovenya/SpeechAudioLM/cache/metrics/huggingface
/scratch/lovenya/SpeechAudioLM/cache/metrics/torch
/scratch/lovenya/SpeechAudioLM/cache/metrics/sentence_transformers
```

Key files verified:

```text
/scratch/lovenya/SpeechAudioLM/models/metrics/whisper/large-v3.pt
/scratch/lovenya/SpeechAudioLM/cache/metrics/utmosv2/models/fusion_stage3/fold0_s42_best_model.pth
/scratch/lovenya/SpeechAudioLM/cache/metrics/torch/hub/checkpoints/vggish-10086976.pth
/scratch/lovenya/SpeechAudioLM/cache/metrics/torch/hub/checkpoints/vggish_pca_params-970ea276.pth
/scratch/lovenya/SpeechAudioLM/cache/metrics/torch/hub/checkpoints/passt-s-f128-p16-s10-ap.476-swa.pt
```

Stable Audio Metrics source and env were placed on scratch because project quota
became tight:

```text
/scratch/lovenya/SpeechAudioLM/code/metrics/stable-audio-metrics
/scratch/lovenya/SpeechAudioLM/.venvs/metrics/.venv-eval-stable-audio-metrics
```

The source tree should be transferred if stable-audio-metrics remains the chosen
backend, but recreate the venv on the new cluster.

### Environments

Do not rely on transferred venvs. They often contain absolute paths and cluster
ABI assumptions. Recreate them on the new cluster after choosing the module
stack.

Current model/baseline env names:

```text
.venv-audiocraft
.venv-tangoflux
.venv-qwen3-omni
.venv-audio-omni
```

Current metric env names:

```text
.venv-eval-whisper
.venvs/.venv-eval-aac
.venvs/metrics/.venv-eval-audioldm-eval
.venvs/metrics/.venv-eval-kadtk
.venvs/metrics/.venv-eval-utmosv2
/scratch/lovenya/SpeechAudioLM/.venvs/metrics/.venv-eval-stable-audio-metrics
```

On the new cluster, keep model envs and metric envs separated. Metric backends
have conflicting dependencies, Python versions, Java requirements, and toolkit
constraints. The current Whisper env is repo-root `.venv-eval-whisper`; moving
it under `.venvs/metrics/` is fine on the new cluster, but update
`configs/eval_metrics.json` if the path changes.

### Generated Outputs And Logs

Scratch baseline/eval outputs observed:

```text
/scratch/lovenya/SpeechAudioLM/results/baselines
/scratch/lovenya/SpeechAudioLM/results/baselines_shards
/scratch/lovenya/SpeechAudioLM/results/baselines_stage_metadata
/scratch/lovenya/SpeechAudioLM/results/eval_smoke
/scratch/lovenya/SpeechAudioLM/results/eval_smoke_existing
/scratch/lovenya/SpeechAudioLM/logs/baseline_shards
/scratch/lovenya/SpeechAudioLM/logs/eval_smoke
/scratch/lovenya/SpeechAudioLM/logs/resources
/scratch/lovenya/SpeechAudioLM/logs/slurm
```

Older project smoke outputs:

```text
/project/rrg-csubakan/lovenya/outputs/audiogen_smoke
/project/rrg-csubakan/lovenya/outputs/tangoflux_smoke
/project/rrg-csubakan/lovenya/outputs/qwen3_omni_smoke
/project/rrg-csubakan/lovenya/outputs/audio_omni_smoke
```

Transfer generated outputs only if you need to preserve existing smoke/baseline
results. They are not git artifacts.

## Current Evaluation Status

Passed through `evaluate.py`:

- `asr_wer,dwer` with Whisper large-v3 on `audio_omni` `T2S`.
- `utmos` with UTMOSv2 on `audio_omni` `T2S`.
- Combined `asr_wer,dwer,utmos` on 5 `audio_omni` `T2S` samples.
- `spider` with `aac-metrics` on an Audio-Omni A2T-style manifest.
- `kad` through `kadtk` on 2 `audio_omni` `T2A` samples.
- `clap_score_t2a,kl_passt` through stable-audio-metrics on 2 `audio_omni`
  `T2A` samples.

Known fixes already committed:

- AAC metrics now use explicit AAC-specific HF cache roots.
- `mace` and `aac_clapsim` use `batch_size: 1` to avoid an MS-CLAP collator
  failure on variable-length GPT-2 token sequences.
- UTMOSv2 runs with `HF_HUB_OFFLINE=1` and `TRANSFORMERS_OFFLINE=1` so compute
  nodes do not try Hugging Face network checks.

Still needing confirmation from this thread:

- Retry `aac_clapsim` after the batch-size fix.
- Retry `mace` after the batch-size fix.
- Retry `fad,inception_score` after the VGGish PCA checkpoint was downloaded.

## Current Baseline Status

Baseline CLI:

```text
baseline.py
```

Main output convention:

```text
results/baselines/<model>/<task>/<dataset>/<split>/
```

Scratch full/sharded outputs use:

```text
/scratch/lovenya/SpeechAudioLM/results/baselines
/scratch/lovenya/SpeechAudioLM/results/baselines_shards
```

Supported/implemented task paths include:

- AudioGen: `T2A`
- TangoFlux: `T2A`
- Audio-Omni: `T2A`, `T2S`, `A2T`, `S2T`, cascaded `S2A`, cascaded `A2S`
- Qwen3-Omni: `T2S`, `A2S`; any pseudo-T2A task paths should be treated as
  legacy/exploratory and not as project requirements unless explicitly revived

Audio-Omni supports A2T/S2T through its understanding path. The capability
registry should include those modes.

## New Cluster First Steps

1. Copy or clone the repo.
2. Copy external resources listed above to stable project/scratch paths on the
   new cluster.
3. Create a local path policy and update:
   - `configs/ttas_dataset.json`
   - `configs/eval_metrics.json`
   - model backend path defaults/env vars as needed
   - submission scripts if scheduler/module names differ
4. Recreate venvs under the new module stack.
5. Run lightweight validation before full compute:
   - `python utils/generate_metadata_report.py --print --other-labels-limit 8`
   - `python baseline.py --model audio_omni --task T2S --split test --num-samples 1 --dry-run`
   - `python evaluate.py --model audio_omni --task T2S --metrics asr_wer,dwer,utmos --max-samples 1 --dry-run`
6. Use compute nodes for real model loading and metric scoring.

## Transfer Guidance

Prefer transfer tools that preserve symlinks and resume partial copies:

```bash
rsync -aH --info=progress2 SOURCE/ DEST/
```

Do not transfer venvs as the primary setup strategy. Transfer them only as
debugging references if needed.

Prioritize transfer order:

1. Repo clone with tracked handoff docs.
2. Dataset root and repo split/manifest files.
3. Local/custom model assets and toolkit source trees that are not cleanly
   recoverable from public repositories.
4. Existing generated baseline/eval outputs only if needed for continuity.
5. Public Hugging Face/GitHub model weights, metric checkpoints, and caches
   should be re-downloaded on the new cluster when practical; transfer caches
   only if download access, quota, or time makes that necessary.

After transfer, run a path audit before any compute jobs:

```bash
python - <<'PY'
from pathlib import Path
for path in [
    "configs/ttas_dataset.json",
    "configs/eval_metrics.json",
    "configs/model_capabilities.json",
    "configs/task_metric_map.json",
]:
    print(f"\n== {path} ==")
    print(Path(path).read_text()[:2000])
PY
```

Replace old `/project/rrg-csubakan`, `/lustre09/project`, and
`/scratch/lovenya/SpeechAudioLM` paths with the new cluster's project/scratch
roots.

## Notes For The Next Agent

- Do not assume Rorqual module names exist on the new cluster.
- Do not assume compute nodes have internet.
- Do not run full inference/eval on login nodes.
- Keep metric backends lazy: fail only when selected.
- Keep generated audio/results outside git.
- Record package versions, checkpoint paths, cache roots, device, command, and
  preprocessing policy in eval run configs.
