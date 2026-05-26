# New Cluster Environment Setup Notes

Last updated: 2026-05-26

Use this document as a practical checklist for bringing SpeechAudioLM up on a
new cluster. It intentionally describes patterns rather than assuming the new
cluster has Rorqual's modules, accounts, GPU names, or internet policy.

## 1. Establish Path Variables

Choose stable roots on the new cluster before editing configs:

```bash
export SALM_REPO=/path/to/SpeechAudioLM
export SALM_PROJECT=/path/to/persistent/project/storage/SpeechAudioLM
export SALM_SCRATCH=/path/to/scratch/SpeechAudioLM
export SALM_DATA=/path/to/persistent/datasets/ttas/extracted
export SALM_MODELS=/path/to/persistent/models
```

Recommended split:

- repo/code: persistent project or home-backed storage;
- datasets and important model checkpoints: persistent project storage;
- caches, generated audio, eval outputs, temporary shard outputs: scratch;
- venvs: recreate on the new cluster; either repo-local or scratch, depending
  on quota and filesystem policy.

Before setup work, make sure Codex has the same project guidance:

```text
AGENTS.md
CONTEXT.md
CLUSTER_MIGRATION_HANDOFF.md
NEW_CLUSTER_ENV_SETUP.md
docs/eval_backend_setup.md
```

The custom documentation skill is included in the cloned repo:

```text
codex/skills/update_docs/SKILL.md
codex/skills/cluster-migration/SKILL.md
```

Install it into the new cluster's Codex home if repo-local skills are not
auto-discovered:

```text
~/.codex/skills/update_docs/SKILL.md
```

## 2. Transfer Resources

Transfer these categories from the old cluster:

```text
repo working tree
dataset root
local/custom baseline model assets
external toolkit source trees that are not clean public clones
generated outputs/logs only if needed
```

Re-download public Hugging Face/GitHub model weights, metric checkpoints, and
toolkit packages on the new cluster when practical. Transfer existing
model/metric caches only when download access, quota, or time makes
re-downloading worse than copying.

The migration Markdown files are currently tracked in git for convenience, but
verify them after clone. If any are missing, copy them separately before asking
the new Codex agent to make path or environment decisions.

Do not depend on transferred venvs. Recreate them after loading the new
cluster's Python/CUDA/compiler modules.

After transfer, update config paths:

```text
configs/ttas_dataset.json
configs/eval_metrics.json
configs/model_capabilities.json
configs/slurm_resource_profiles.json
submission_scripts/*.sbatch
```

## 3. Module And Python Policy

On Rorqual, working envs used `StdEnv/2023`, `gcc`, `cuda`, and Python
`3.10.13` or `3.11.x` depending on backend. On the new cluster:

1. Inspect available modules.
2. Pick one module stack per environment.
3. Create the venv under the exact stack used later.
4. Prefer cluster wheelhouses for packages available there.
5. Use internet installs only where allowed and deliberate.

Useful probes:

```bash
module avail python
module avail cuda
module avail gcc
python --version
nvidia-smi
```

If compute nodes do not have internet, perform downloads on login/transfer/FIR
nodes, then run compute jobs with offline cache variables.

## 4. Baseline Model Environments

Recreate model envs separately:

```text
.venv-audiocraft
.venv-tangoflux
.venv-qwen3-omni
.venv-audio-omni
```

Rorqual notes that may matter on the new cluster:

- AudioCraft/AudioGen needed an H100-compatible Torch build; Torch 2.1 lacked
  `sm_90` support on H100/MIG, while Torch 2.6 worked.
- AudioGen needs local `audiogen-medium` plus cached `t5-large`.
- TangoFlux needs local TangoFlux assets plus cached `google/flan-t5-large`.
- Audio-Omni needs external source at `src/Audio-Omni`, its checkpoint folder,
  and local Qwen2.5-Omni-3B.
- Qwen3-Omni-30B used a venv-local Transformers talker workaround on Rorqual.
  Re-test on the new cluster before reapplying; upstream may differ.
- Qwen3-Omni pseudo-T2A paths are legacy/exploratory only. Do not prioritize
  them during migration unless the project explicitly revives that direction.
- FlashAttention was treated as an optimization, not a first-pass requirement,
  because fallbacks were functional.

Run dry-runs first:

```bash
python baseline.py --model audiogen --task T2A --split test --num-samples 1 --dry-run
python baseline.py --model tangoflux --task T2A --split test --num-samples 1 --dry-run
python baseline.py --model audio_omni --task T2S --split test --num-samples 1 --dry-run
```

Then run one-sample compute-node smokes before full shards.

## 5. Metric Environments

Keep metric backends isolated:

```text
.venv-eval-whisper
.venvs/.venv-eval-aac
.venvs/metrics/.venv-eval-audioldm-eval
.venvs/metrics/.venv-eval-kadtk
.venvs/metrics/.venv-eval-utmosv2
.venvs/metrics/.venv-eval-stable-audio-metrics
```

Current backend mapping:

| Metric(s) | Backend | Notes |
|---|---|---|
| `asr_wer`, `dwer` | OpenAI Whisper | `large-v3`; cache path in `configs/eval_metrics.json` |
| `spider`, `mace`, `aac_clapsim` | `aac-metrics` | Java 11 worked; Java 17 failed tokenization checks |
| `fad`, `inception_score` | `audioldm_eval` | Needs VGGish and VGGish PCA checkpoints in Torch hub cache |
| `kad` | `kadtk` | Python/toolkit constraints; run corpus-level |
| `clap_score_t2a`, `kl_passt` | stable-audio-metrics | Source tree is not a normal pip project; use PYTHONPATH |
| `utmos` | UTMOSv2 | Needs UTMOS checkpoint and cached `facebook/wav2vec2-base` |

Current Rorqual config points Whisper at repo-root `.venv-eval-whisper`. On the
new cluster, it is reasonable to place it under `.venvs/metrics/`, but update
`configs/eval_metrics.json` before running `evaluate.py`.

Backend-specific lessons from Rorqual:

- `aac-metrics` cache root must point at the AAC-specific HF cache, not a
  generic sibling cache.
- `mace` and `aac_clapsim` are configured with `batch_size: 1` to avoid an
  MS-CLAP collator failure on variable-length GPT-2 token tensors.
- UTMOS uses `HF_HUB_OFFLINE=1` and `TRANSFORMERS_OFFLINE=1` so no-internet
  compute nodes do not attempt Hugging Face HEAD requests.
- `audioldm_eval` needed the VGGish files:
  `vggish-10086976.pth` and `vggish_pca_params-970ea276.pth`.
- `stable-audio-metrics` needed extra LAION-CLAP dependencies and offline
  RoBERTa cache availability before `clap_score_t2a,kl_passt` passed.

After recreating a metric env, use the smoke scripts:

```bash
python scripts/eval_backend_smoke/smoke_whisper_asr.py --help
python scripts/eval_backend_smoke/smoke_aac_metrics.py --help
python scripts/eval_backend_smoke/smoke_audioldm_eval.py --help
python scripts/eval_backend_smoke/smoke_kadtk.py --help
python scripts/eval_backend_smoke/smoke_utmos.py --help
```

Then run through `evaluate.py`, because the actual framework passes cache/env
settings through subprocess workers.

Before full eval, verify the evaluator can see every backend Python path:

```bash
python - <<'PY'
import json
from pathlib import Path
cfg = json.loads(Path("configs/eval_metrics.json").read_text())
for metric, item in cfg["metrics"].items():
    python = item.get("python")
    if python:
        print(metric, python, Path(python).exists())
PY
```

## 6. Offline Cache Variables

Use explicit cache variables in compute jobs, adjusted to the new cluster paths:

```bash
export HF_HOME="$SALM_SCRATCH/cache/metrics/huggingface"
export TRANSFORMERS_CACHE="$HF_HOME"
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export TORCH_HOME="$SALM_SCRATCH/cache/metrics/torch"
export XDG_CACHE_HOME="$SALM_SCRATCH/cache"
```

Backend workers also set metric-specific variables from `configs/eval_metrics.json`.
Update that config after transfer.

## 7. Evaluation Smoke Commands

Once baseline manifests exist on the new cluster:

```bash
python evaluate.py \
  --model audio_omni \
  --task T2S \
  --metrics asr_wer,dwer,utmos \
  --max-samples 5 \
  --baseline-root "$SALM_SCRATCH/results/baselines" \
  --output-root "$SALM_SCRATCH/results/eval_smoke" \
  --cache-dir "$SALM_SCRATCH/cache/metrics/eval_runtime" \
  --device cuda
```

```bash
python evaluate.py \
  --model audio_omni \
  --task T2A \
  --metrics kad,clap_score_t2a,kl_passt \
  --max-samples 2 \
  --baseline-root "$SALM_SCRATCH/results/baselines" \
  --output-root "$SALM_SCRATCH/results/eval_smoke" \
  --cache-dir "$SALM_SCRATCH/cache/metrics/eval_runtime" \
  --device cuda
```

Run `mace`, `aac_clapsim`, and `fad,inception_score` separately while finishing
backend validation. `mace` and `aac_clapsim` are MS-CLAP/AAC-cache sensitive;
`fad` and `inception_score` are AudioLDM Eval/VGGish-cache sensitive.

## 8. Scheduler And Resource Monitoring

The current submission scripts are Slurm-oriented and Rorqual-specific. On a new
cluster:

- rewrite account, partition, GPU, memory, and module lines;
- keep shard ranges for parallel baseline inference;
- keep resource monitoring if Slurm and `nvidia-smi` are available;
- request about 125% of observed walltime for repeat runs after a successful
  smoke, rather than defaulting to long allocations.

Resource monitor:

```text
scripts/resource_monitor.py
submission_scripts/lib_resource_monitor.sh
configs/slurm_resource_profiles.json
```

## 9. Validation Order

Use this order to reduce debugging ambiguity:

1. Repo imports and unit tests:
   `PYTHONDONTWRITEBYTECODE=1 python -m unittest tests/test_evaluation.py`
2. Dataset path/report check.
3. Baseline CLI dry-runs.
4. One model smoke on compute node.
5. One metric backend smoke in its own env.
6. `evaluate.py` smoke using a real manifest.
7. Sharded baseline runs.
8. Full metric runs.

Do not skip from environment creation straight to full-scale inference/eval.
