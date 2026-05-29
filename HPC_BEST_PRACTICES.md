# HPC Best Practices

## Node Usage

Use login nodes for code editing, git, lightweight validation, small metadata scripts, plotting, and job submission only. Do not run heavy compute, training, large preprocessing, aggressive multithreading, or large batch inference on login nodes.

Use compute nodes through interactive jobs or `sbatch` for GPU work, training, large inference, and expensive preprocessing.

## Accounts And Scheduling Priority

Use the project accounts in this priority order:

1. `rrg-ravanelm` for GPU jobs first.
2. `rrg-csubakan` if the first account is not suitable or is not getting accepted.
3. `def-csubakan-ab` only if the RRG accounts are not practical for the job.
4. `def-csubakan-ab_cpu` only for CPU-only jobs.

Do not request GPU nodes for CPU-only work. Over-requesting resources and then using only a small fraction of them can hurt future scheduling priority. Size requests to the experiment:

- CPU-only metadata or path checks: login node if lightweight, otherwise CPU allocation.
- AudioGen smoke test: one GPU, modest CPU count, modest host RAM, short walltime.
- Full inference or larger batches: increase resources only after measuring the smoke-test footprint.

Record both requested resources and observed utilization when possible. This helps right-size later jobs and avoids wasting allocation priority.

On TamIA, a GPU Slurm job reserves a whole node with 4 H100s. Do not submit two separate node-sized jobs when the actual GPU demand fits inside one node. If one workload needs 1 GPU and another needs 2 GPUs, prefer one packed `sbatch` script that launches both processes with explicit `CUDA_VISIBLE_DEVICES` assignments. This uses one 4-H100 node instead of two 4-H100 nodes and avoids wasting mostly idle allocations.

## Interactive Jobs

Use `salloc` for interactive debugging. On Rorqual, local Slurm help shows support for both generic GPU requests and newer GPU flags:

```bash
salloc -A rrg-ravanelm -t 03:00:00 -c 4 --mem=32G --gpus=h100_2g.20gb:1
```

For AudioGen-medium smoke tests, start with a 20 GB MIG slice if available. If that is too small, move to 40 GB MIG; use a full 80 GB H100 only when the workload justifies it.

Rorqual GPU request names from the cluster notes:

| Resource | Slurm request |
|---|---|
| H100 10 GB MIG | `--gpus=h100_1g.10gb:1` |
| H100 20 GB MIG | `--gpus=h100_2g.20gb:1` |
| H100 40 GB MIG | `--gpus=h100_3g.40gb:1` |
| Full H100 80 GB | `--gpus=h100:1` or `--gpus=h100_80gb:1` |

For multiple full H100s on one node, use `--gpus-per-node=h100:<n>` with `<n>` up to `4`. For multiple full H100s spread wherever Slurm can place them, use `--gpus=h100:<n>`.

Check available resources from a normal login terminal with:

```bash
sinfo -o "%P %G %D %t %m %c %f"
sinfo -N -o "%N %P %G %t %m %c %f"
```

If allocation fails, read the Slurm error first. It may require a different account, partition, QOS, time limit, or GPU type.

After allocation, start a shell and initialize the project environment:

```bash
srun --pty bash
module load StdEnv/2023 gcc cuda rust arrow python/3.10.13
source /lustre09/project/6090520/lovenya/SpeechAudioLM/.venv-audiocraft/bin/activate
cd /lustre09/project/6090520/lovenya/SpeechAudioLM
```

Use whole-GPU requests only when the workload can justify the whole device. If smaller VRAM slices are available and sufficient, prefer them for setup, import checks, and short smoke tests.

## Environment Setup

Load cluster modules before activating a Python virtual environment. Standard baseline for this project:

```bash
module load StdEnv/2023 gcc cuda rust arrow python/3.10.13
source .venv/bin/activate
```

Load optional modules only when needed, for example:

```bash
module load opencv
```

Use `venv` or `virtualenv` for Python environments. Do not assume conda, anaconda, or uv are available. Create the environment under the same module stack that will be used later.

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install --no-index -r requirements.txt
```

Prefer `--no-index` for normal package installs so pip uses the cluster wheelhouse instead of the public internet. Use internet installs only deliberately, for packages that are not available in the wheelhouse or for one-time downloads on an internet-capable node.

When setup work has independent parts, split it across terminals to reduce wall-clock time. For example, one terminal can create a venv and install packages while another downloads model weights, ONNX files, or metric checkpoints to the agreed cache path. Only run smoke tests or import checks after both the environment and required assets are complete. Do not parallelize steps that write to the same environment, cache file, or checkpoint path.

For AudioCraft/AudioGen work, the current environment is:

```bash
module load StdEnv/2023 gcc cuda rust arrow python/3.10.13
source .venv-audiocraft/bin/activate
```

Use an H100-compatible PyTorch build on Rorqual. The older Torch 2.1 env fails on H100/MIG with missing `sm_90` kernels. The current working env uses `torch==2.6.0+computecanada`.

Keep Hugging Face models and caches outside the repository. Current AudioGen model location:

```text
/project/rrg-csubakan/lovenya/models/audiogen-medium
```

Use explicit local paths for offline inference on no-internet compute nodes. Avoid relying on implicit home-directory caches. AudioGen-medium also needs the separately cached `t5-large` text encoder/tokenizer because AudioCraft loads it at inference time.

For AudioGen offline inference on Rorqual compute nodes, export all relevant Hugging Face cache variables before loading the model:

```bash
export HF_HOME=/project/rrg-csubakan/lovenya/models/hf_cache
export HF_HUB_CACHE=/project/rrg-csubakan/lovenya/models/hf_cache/transformers
export TRANSFORMERS_CACHE=/project/rrg-csubakan/lovenya/models/hf_cache/transformers
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
```

This is needed because different Hugging Face/Transformers versions may look at different cache variables. Setting all of them avoids accidental internet attempts on compute nodes.

For baseline models with nested Hugging Face dependencies, download every referenced model on an internet-capable node and point the inference code at explicit local paths. Examples encountered so far:

- TangoFlux needs `google/flan-t5-large` in the Transformers cache.
- Audio-Omni needs `Qwen/Qwen2.5-Omni-3B`, currently stored at `/project/rrg-csubakan/lovenya/models/qwen2.5-omni-3b`, with `QWEN_OMNI_MODEL_PATH` exported before loading.
- Qwen3-Omni-30B-A3B-Instruct is large enough that a full H100 is appropriate for local Transformers loading with the talker enabled.

Treat optional accelerator libraries such as FlashAttention as performance work, not first-pass setup requirements. If a model detects missing `flash_attn` and cleanly falls back, keep the fallback until throughput or memory pressure justifies the extra build complexity.

## Evaluation Backend Setup

Metric backends may have incompatible dependency stacks. Prefer separate venvs
when a toolkit brings a large model, strict Python constraints, Java/runtime
requirements, or pinned audio dependencies.

When migrating to a new cluster, recreate venvs instead of moving them in place.
Venvs can contain absolute interpreter paths, compiled extensions, and ABI
assumptions tied to the old cluster. Transfer source trees, caches, wheelhouses,
and checkpoints as references, then create fresh environments under the new
cluster's module stack.

Do not run full metric evaluation, model-weight downloads, or corpus-level audio
embedding extraction on login nodes. Use login nodes for config edits, import
checks, and small dry-runs only. Use an interactive job or `sbatch` for:

- ASR/CLAP/UTMOS/DNSMOS model loading.
- FAD/KLD/Inception Score/KAD corpus embedding extraction.
- Any metric pass over full TTAS test outputs.

Keep metric caches and checkpoints outside git, preferably under a predictable
project path such as:

```text
/project/rrg-csubakan/lovenya/models/metrics
/project/rrg-csubakan/lovenya/cache/metrics
```

The latest Rorqual eval setup moved heavy metric assets to scratch to avoid
project quota pressure:

```text
/scratch/lovenya/SpeechAudioLM/models/metrics
/scratch/lovenya/SpeechAudioLM/cache/metrics
/scratch/lovenya/SpeechAudioLM/code/metrics
```

On a new cluster, choose equivalent persistent/scratch roots first, then update
`configs/eval_metrics.json` rather than patching paths inside Python modules.

Record backend package versions, git commits when applicable, checkpoint paths,
model/config names, device, cache path, sample-rate policy, and resampling or
normalization behavior in evaluation run configs.

For libraries that create user caches during import, prefer node-local temporary cache directories:

```bash
export MPLCONFIGDIR="${SLURM_TMPDIR:-/tmp}/$USER/matplotlib"
mkdir -p "$MPLCONFIGDIR"
```

## Internet Constraints

Rorqual/TamIA compute nodes do not have internet. FIR compute nodes have internet.

For no-internet compute nodes, prefer cached wheels or a wheelhouse:

```bash
python -m pip install --no-index --find-links /path/to/wheelhouse -r requirements.txt
```

Prepare downloads on an internet-capable node when cluster policy allows it.

## Job Hygiene

- Request only the CPUs, memory, GPUs, and walltime needed.
- Write logs to a predictable `logs/` or scratch location.
- Keep checkpoints and generated outputs outside git.
- Record module versions, Python version, command line, dataset manifest, and split file for reproducibility.

## Failure Handling

Do not recursively debug toolchain or cluster errors. If a fix fails twice, stop, summarize the error, list commands tried, and ask before continuing.

For dependency conflicts, inspect the installed package versions and import paths before changing versions. Prefer one deliberate compatibility fix over trying many package combinations.
