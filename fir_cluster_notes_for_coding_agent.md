# Fir Cluster Notes for Coding Agent

Source: Alliance Canada / Calcul Québec documentation PDF uploaded by Lovenya: `Fir - Alliance Doc.pdf`
Purpose: give a coding agent enough cluster-specific context to make sensible setup, transfer, Slurm, storage, and GPU decisions.

> **Important:** This file is a distilled working note, not a replacement for the official Fir documentation. When in doubt, check the official docs or ask Lovenya to provide the latest page/details.

---

## 1. Quick identity / cluster summary

- **Cluster name:** Fir
- **Availability date:** August 11, 2025
- **Host site:** Simon Fraser University, Burnaby, British Columbia
- **Built with:** Lenovo Canada and Data Direct Networks (DDN)
- **General purpose:** heterogeneous scientific computing cluster
- **Cooling / sustainability note:** SFU is transitioning from traditional air cooling to direct-to-chip liquid cooling for better energy efficiency.
- **TOP500:** ranked **#78** on the June 2025 TOP500 list.

Useful public links:

- Official Fir documentation snapshot/source page: `https://docs.alliancecan.ca/mediawiki/index.php?title=Fir&oldid=197500`
- TOP500 June 2025 list: `https://top500.org/lists/top500/list/2025/06/`

---

## 2. Main endpoints and services

| Purpose | Endpoint / URL | Notes |
|---|---|---|
| Login node | `fir.alliancecan.ca` | Use for SSH login and, according to the PDF, for `rsync` / `scp` until a dedicated transfer node is announced. |
| Automation node | `robot.fir.alliancecan.ca` | For automation workflows if needed. |
| Globus collection | `alliancecan#fir-globus` | Use for Globus transfers. |
| Globus URL | `https://globus.alliancecan.ca/file-manager?origin_id=8dec4129-9ab4-451d-a45f-5b4b8471f7a3&two_pane=false` | Official endpoint link from the PDF. |
| JupyterHub | `https://jupyterhub.fir.alliancecan.ca/` | Web-based Jupyter access. |
| Data transfer node | `to be determined` | The PDF says not yet determined. For `rsync`, `scp`, `sftp`, use the login node for now. |
| Portal | `to be determined` | Not yet listed in the PDF. |
| Access request | CCDB → Resources → Access Systems | Select Fir and request access. Likely CCDB URL: `https://ccdb.alliancecan.ca/me/access_systems` |

---

## 3. Access notes

A researcher must request access through CCDB:

1. Go to CCDB.
2. Navigate to **Resources → Access Systems**.
3. Select **Fir** from the list.
4. Select **I request access**.
5. Wait for access to be enabled; the document says this can take up to one hour.

Coding-agent implication:

- Do **not** assume the user has access until login succeeds.
- If SSH fails with permission/access errors, ask Lovenya to verify Fir access in CCDB.
- Do not hard-code project paths unless Lovenya gives the actual project allocation ID.

---

## 4. Site-specific policies

Important cluster policies from the document:

- **Compute nodes have full internet access.**
- `crontab` is **not supported**.
- Each normal job should run for at least **1 hour**.
- Test jobs should run for at least **5 minutes**.
- Maximum job duration is **7 days / 168 hours**.
- For Globus transfers, use the Fir Globus collection.
- For `rsync`, `scp`, and `sftp`, use the **login node** for now because the transfer node is listed as “to be determined.”

Coding-agent implications:

- Since compute nodes have internet, dependency installation during interactive compute sessions may be possible, but do not abuse this. Prefer reproducible `requirements.txt`, lock files, wheelhouse, or documented setup scripts.
- Do not suggest `cron` jobs on Fir.
- For periodic/automated work, investigate scheduler-native approaches or the automation node, but do not invent policies.
- Avoid creating thousands of tiny short jobs; bundle work into reasonable Slurm jobs.

---

## 5. Storage overview

Fir has **51 PB high-performance DDN Lustre storage**, composed of approximately **2 PB NVMe** and **49 PB SAS**. The document says all mounts share the available storage.

| Storage area | Access path | Quota | Backup | Intended use |
|---|---|---|---|---|
| HOME | `$HOME` | Small per-user quota | Daily automatic backup | Configs, scripts, small files. Cannot be increased. Use project space for larger storage. |
| SCRATCH | `$HOME/scratch` | Large per-user quota | No backup | Temporary computation files. Old files are purged automatically. |
| PROJECT | `$HOME/project/${def-project-id}` | Large, adjustable per-project quota | Daily backup | Shared group data and large datasets. |

Coding-agent rules:

- Keep source code, small configs, and environment files in `$HOME` or a Git repo.
- Put large datasets, checkpoints, and shared artifacts under `$HOME/project/${def-project-id}` once the real project ID is known.
- Use `$HOME/scratch` for temporary intermediate files and cache-like artifacts that can be regenerated.
- Do not store important one-off results only in scratch, because scratch has no backup and old files are purged.
- Do not commit large datasets/checkpoints to Git.

Recommended repo hygiene:

```text
repo/
  README.md
  PROJECT_BRIEF.md
  CLUSTER_FIR.md
  requirements.txt / pyproject.toml
  scripts/
  data_utils/
  configs/
  slurm/
  logs/                # small text logs only
  .gitignore
```

Suggested `.gitignore` additions:

```gitignore
# Data / outputs
data/
datasets/
checkpoints/
outputs/
runs/
wandb/
*.pt
*.pth
*.ckpt
*.safetensors
*.tar
*.tar.gz
*.zip

# Local envs
.venv/
venv/
__pycache__/
*.pyc

# Local cluster/session noise
slurm-*.out
*.err
*.log
```

---

## 6. Network / interconnect

Fir uses an **InfiniBand NDR interconnect**.

Additional details from the document:

- CPU node island size: **27:5 blocking factor** over **216 nodes of 192 cores**.
- GPU nodes: **2:1 blocking factor**.
- Storage access is fully non-blocking.
- The new high-speed InfiniBand network is described as delivering more than twice the performance of the previous-generation Cedar cluster.

Coding-agent implication:

- Multi-node distributed training is technically plausible, but should be treated carefully.
- Start with single-GPU or single-node experiments unless Lovenya explicitly asks for multi-node.
- For multi-GPU training, prefer using one GPU node with multiple H100s before scaling across nodes.
- Do not assume network topology details beyond what the official docs say.

---

## 7. Node characteristics

### 7.1 Node table

| Node count | Cores | Available memory | CPU | Local storage | GPU |
|---:|---:|---:|---|---|---|
| 864 | 192 | 750G / 768000M | 2 × AMD EPYC 9655 Zen 5 @ 2.7 GHz, 384 MB L3 cache | 7.84 TB NVMe | none |
| 8 | 192 | 6000G / 6144000M | 2 × AMD EPYC 9654 Zen 4 @ 2.4 GHz, 384 MB L3 cache | 7.84 TB NVMe | none |
| 160 | 48 | 1125G / 1152000M | 1 × AMD EPYC 9454 Zen 4 @ 2.75 GHz, 256 MB L3 cache | 7.84 TB NVMe | 4 × NVIDIA H100 SXM5, 80 GB, connected via NVLink |

Important: the PDF text has one internal inconsistency in the GPU-node section. The node table says each GPU node has **4 × NVIDIA H100 SXM5 80 GB**, but the later GPU-node layout text says “2 NVidia H100 80GB accelerators” and then also says “The 4 node accelerators are interconnected by SXM5.” Treat the table and Slurm examples supporting up to `--gpus-per-node=h100:4` as the more operationally useful reference, but verify on Fir with commands like `sinfo`, `scontrol show node`, or by checking official docs before relying on exact GPU count.

---

## 8. CPU node architecture and tuning

### 8.1 CPU node architecture

The main CPU nodes use:

- **2 × AMD EPYC 9655 Zen 5**
- **192 physical cores total**
- Chiplet-based NUMA architecture
- Each CCD/chiplet behaves as a separate NUMA locality domain
- Performance is sensitive to memory and cache locality

Layout:

- 2 sockets
- Each socket has:
  - 96 cores
  - 4 NUMA nodes
  - 12 CCDs/chiplets per socket
- Each NUMA node has:
  - 3 CCDs/chiplets
  - 3 memory channels
- Each CCD/chiplet has:
  - 8 cores
  - 32 MiB shared L3 cache
- Each core has:
  - 1 MiB L2 cache
  - 32 KiB L1 instruction cache
  - 32 KiB L1 data cache
- Total per node:
  - 8 NUMA nodes
  - 24 CCDs/chiplets
  - 192 cores
  - 768 MiB L3 cache

### 8.2 CPU performance recommendations

The document recommends aligning CPU tasks to CCDs.

Use one task per CCD:

```bash
#SBATCH --cpus-per-task=8
```

To use a full 192-core CPU node cleanly:

```bash
#SBATCH --ntasks-per-node=24
#SBATCH --cpus-per-task=8
```

Reasoning:

- Each CCD has 8 tightly-coupled cores sharing L3 cache.
- Keeping a task’s threads within one CCD avoids unnecessary inter-chiplet communication.
- 24 tasks × 8 CPUs per task = 192 cores.

Coding-agent implication:

- For CPU-only parallel jobs, prefer designing around `--cpus-per-task=8` when appropriate.
- For full-node CPU jobs, consider `--ntasks-per-node=24 --cpus-per-task=8`.
- For Python ML preprocessing, do not blindly request all 192 cores unless the workload truly scales.
- For data loading on GPU jobs, start with modest `num_workers` and increase empirically.

---

## 9. GPU node architecture and tuning

### 9.1 GPU node architecture

GPU node table summary:

- **160 GPU nodes**
- **48 CPU cores per node**
- **1125G / 1152000M system memory**
- **1 × AMD EPYC 9454 Zen 4 @ 2.75 GHz**
- **7.84 TB NVMe local storage**
- **4 × NVIDIA H100 SXM5 80 GB**, connected via NVLink according to the node table

GPU CPU layout:

- 1 socket configured as **NPS=4**
- 4 NUMA nodes
- Each NUMA node has:
  - 2 CCDs
  - 3 memory channels
- Each CCD has:
  - 6 cores
  - 32 MiB shared L3 cache
- Each core has:
  - 1 MiB L2 cache
  - 32 KiB L1 instruction cache
  - 32 KiB L1 data cache
- 12 DDR5 memory channels shared via the I/O die

### 9.2 GPU performance recommendations

The document recommends binding threads to CCDs:

```bash
#SBATCH --cpus-per-task=6
```

Reasoning:

- Each CCD has 6 closely-coupled cores sharing 32 MiB L3 cache.
- Keeping threads within one CCD reduces cross-CCD latency and improves cache usage.

For matching tasks to NUMA domains, the document recommends:

```bash
#SBATCH --ntasks-per-node=4
#SBATCH --cpus-per-task=12
```

Reasoning:

- GPU nodes use NPS=4, giving 4 NUMA nodes per socket.
- Launching 4 tasks per node, or a multiple thereof, helps keep each task within a NUMA domain and improves CPU-GPU data locality.

Coding-agent implication:

- For single-GPU training, start with something simple like one task, one GPU, and enough CPUs for the data loader.
- For full-node 4-GPU training, consider task/GPU mapping carefully.
- Verify with Slurm environment variables and `nvidia-smi`.
- For PyTorch DDP on one node, a common pattern is one process per GPU. On Fir, that likely means 4 tasks/processes per GPU node if using all GPUs.

---

## 10. GPU and MIG instances

### 10.1 Full H100 request examples

Request one full H100:

```bash
#SBATCH --gpus=h100:1
```

Request multiple full H100s on the same node:

```bash
#SBATCH --gpus-per-node=h100:2
#SBATCH --gpus-per-node=h100:3
#SBATCH --gpus-per-node=h100:4
```

Request multiple full H100s spread anywhere:

```bash
#SBATCH --gpus=h100:n
```

Replace `n` with the number of GPUs.

### 10.2 MIG instances

Approximately half of the GPU nodes are configured with MIG technology.

Available MIG sizes:

| MIG size | Approx compute fraction | GPU memory | Slurm request |
|---|---:|---:|---|
| `1g.10gb` | 1/8 | 10 GB | `--gpus=nvidia_h100_80gb_hbm3_1g.10gb:1` |
| `2g.20gb` | 2/8 | 20 GB | `--gpus=nvidia_h100_80gb_hbm3_2g.20gb:1` |
| `3g.40gb` | 3/8 | 40 GB | `--gpus=nvidia_h100_80gb_hbm3_3g.40gb:1` |

Important:

- The document says to request **one and only one GPU instance** for a MIG compute job using the corresponding option.
- Do not assume multiple MIG instances per job unless the official docs or cluster behavior confirms it.

Coding-agent implication:

- Use MIG for lightweight tests, debugging, preprocessing with small neural models, tokenizer experiments, and initial smoke tests.
- Use full H100 for serious training, large batch sizes, large models, or anything requiring >40 GB GPU memory.
- Always make the requested GPU explicit in Slurm scripts.

---

## 11. Practical Slurm templates

These are starter templates. They may need account names, partition names, mail settings, and module names once Lovenya confirms the Fir environment.

### 11.1 MIG 10 GB smoke test

```bash
#!/bin/bash
#SBATCH --job-name=fir_mig_smoke
#SBATCH --time=00:20:00
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --gpus=nvidia_h100_80gb_hbm3_1g.10gb:1
#SBATCH --output=logs/%x-%j.out
#SBATCH --error=logs/%x-%j.err

set -euo pipefail

mkdir -p logs

echo "Host: $(hostname)"
echo "Date: $(date)"
echo "CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-unset}"

nvidia-smi || true
python - <<'PY'
import torch
print("torch:", torch.__version__)
print("cuda available:", torch.cuda.is_available())
if torch.cuda.is_available():
    print("device:", torch.cuda.get_device_name(0))
PY
```

### 11.2 Full H100 single-GPU training starter

```bash
#!/bin/bash
#SBATCH --job-name=fir_h100_train
#SBATCH --time=04:00:00
#SBATCH --cpus-per-task=6
#SBATCH --mem=64G
#SBATCH --gpus=h100:1
#SBATCH --output=logs/%x-%j.out
#SBATCH --error=logs/%x-%j.err

set -euo pipefail

mkdir -p logs

echo "Host: $(hostname)"
echo "Date: $(date)"
echo "CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-unset}"

nvidia-smi

# Example only. Replace with actual environment activation.
# source ~/.venvs/project/bin/activate

python train.py --config configs/train.yaml
```

### 11.3 Full-node 4-GPU PyTorch DDP starter

```bash
#!/bin/bash
#SBATCH --job-name=fir_4gpu_ddp
#SBATCH --time=12:00:00
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=4
#SBATCH --cpus-per-task=12
#SBATCH --gpus-per-node=h100:4
#SBATCH --mem=0
#SBATCH --output=logs/%x-%j.out
#SBATCH --error=logs/%x-%j.err

set -euo pipefail

mkdir -p logs

echo "Host: $(hostname)"
echo "Date: $(date)"
echo "SLURM_JOB_ID=$SLURM_JOB_ID"
echo "SLURM_NTASKS=$SLURM_NTASKS"
echo "CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-unset}"

nvidia-smi

# Example only. Replace with actual environment activation.
# source ~/.venvs/project/bin/activate

srun python -m torch.distributed.run \
  --nproc_per_node=4 \
  train.py \
  --config configs/train.yaml
```

Note: depending on Fir’s Slurm/PyTorch setup, the DDP launch pattern may need adjustment. Validate with a tiny DDP smoke test before running long jobs.

---

## 12. Data transfer guidance

### 12.1 Globus

Use:

- Collection name: `alliancecan#fir-globus`
- Globus link: `https://globus.alliancecan.ca/file-manager?origin_id=8dec4129-9ab4-451d-a45f-5b4b8471f7a3&two_pane=false`

Good for:

- Large files
- Many files
- Reliable resumable transfers
- Transfers between Alliance systems or supported endpoints

### 12.2 SCP / SFTP / rsync

The PDF says the data transfer node is still **to be determined**. For now, use:

```bash
fir.alliancecan.ca
```

Examples:

```bash
scp local_file.zip USER@fir.alliancecan.ca:/path/to/destination/
```

```bash
rsync -avh --progress local_dir/ USER@fir.alliancecan.ca:/path/to/destination/
```

Coding-agent implication:

- Prefer Globus for large dataset movement when possible.
- For small files and scripts, `scp` or `rsync` to the login node is fine.
- Avoid heavy processing on login nodes. Transfer is okay; compute belongs in Slurm jobs.

---

## 13. Environment setup guidance

This is based on typical Alliance-style workflows plus the Fir doc’s note that compute nodes have internet access.

Recommended approach:

1. Keep setup reproducible.
2. Use a Python virtual environment unless the project has a different confirmed standard.
3. Save exact package versions after setup.
4. Keep cluster-specific setup instructions in `CLUSTER_FIR.md`.
5. Do not rely on invisible/manual state.

Example:

```bash
python -m venv ~/.venvs/speechaudio
source ~/.venvs/speechaudio/bin/activate
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
python -m pip freeze > requirements.lock.txt
```

Possible smoke test:

```bash
python - <<'PY'
import sys
print(sys.version)
try:
    import torch
    print("torch", torch.__version__)
    print("cuda", torch.cuda.is_available())
except Exception as e:
    print("torch check failed:", repr(e))
PY
```

Important:

- Verify module names on Fir before writing final setup docs.
- If using CUDA/PyTorch, match PyTorch wheels to the available CUDA/runtime environment.
- Do not assume `conda` exists unless Lovenya confirms it.
- Prefer `venv`/`pip` unless the cluster has a supported module or container workflow for the project.

---

## 14. Checks a coding agent should run on Fir

Once logged in:

```bash
hostname
pwd
whoami
quota -s || true
df -h $HOME || true
ls -lah
```

Slurm discovery:

```bash
sinfo
sinfo -o "%P %a %l %D %G"
scontrol show config | head -80
```

GPU discovery from an allocated GPU job:

```bash
nvidia-smi
echo $CUDA_VISIBLE_DEVICES
```

Node-level details from an allocated job:

```bash
lscpu
numactl --hardware || true
lstopo-no-graphics || lstopo || true
```

Python/PyTorch check:

```bash
python - <<'PY'
import torch
print(torch.__version__)
print(torch.cuda.is_available())
print(torch.cuda.device_count())
for i in range(torch.cuda.device_count()):
    print(i, torch.cuda.get_device_name(i))
PY
```

---

## 15. Coding-agent decision rules

When helping Lovenya on Fir:

1. **Do not run heavy work on login nodes.**
   - Login nodes are for editing, compiling lightly, submitting jobs, and transferring small files.
   - Training, preprocessing, and large extraction jobs should use Slurm.

2. **Use the right storage.**
   - `$HOME`: scripts/configs/small files.
   - `$HOME/project/${def-project-id}`: datasets/checkpoints/shared outputs.
   - `$HOME/scratch`: temporary/regenerable work.

3. **Prefer small tests before large runs.**
   - First: CPU smoke test.
   - Then: small MIG GPU test.
   - Then: full H100 test.
   - Then: long training.

4. **Make GPU request explicit.**
   - Full H100: `--gpus=h100:1`
   - 4 full H100s on one node: `--gpus-per-node=h100:4`
   - MIG 10/20/40 GB: use exact `nvidia_h100_80gb_hbm3_*` names from the docs.

5. **Respect job duration policy.**
   - Normal jobs should be at least 1 hour.
   - Test jobs can be at least 5 minutes.
   - Maximum walltime is 7 days / 168 hours.

6. **For CPU binding, use architecture-aware values.**
   - CPU nodes: `--cpus-per-task=8`; full node: `--ntasks-per-node=24`.
   - GPU nodes: `--cpus-per-task=6` for one CCD; or `--ntasks-per-node=4 --cpus-per-task=12` for NUMA-aligned full-node GPU usage.

7. **Assume compute-node internet is available on Fir, but still document dependencies.**
   - Internet availability does not replace reproducibility.

8. **Ask before deleting or moving large data.**
   - Especially anything under project space.

---

## 16. Links inventory from the Fir PDF

Use these when the coding agent needs to point Lovenya to official docs or ask for more information.

| Topic | Link |
|---|---|
| Fir official documentation snapshot/source | `https://docs.alliancecan.ca/mediawiki/index.php?title=Fir&oldid=197500` |
| Fir Globus file manager endpoint | `https://globus.alliancecan.ca/file-manager?origin_id=8dec4129-9ab4-451d-a45f-5b4b8471f7a3&two_pane=false` |
| Fir JupyterHub | `https://jupyterhub.fir.alliancecan.ca/` |
| June 2025 TOP500 list | `https://top500.org/lists/top500/list/2025/06/` |
| CCDB access systems page | `https://ccdb.alliancecan.ca/me/access_systems` |

---

## 17. Open questions / verify before finalizing scripts

The coding agent should ask Lovenya or check Fir directly for:

- Actual username on Fir.
- Actual project allocation path / `${def-project-id}`.
- Whether a dedicated data transfer node has been announced after this PDF snapshot.
- Whether the portal endpoint has been announced.
- Required Slurm account name, if any.
- Available partitions/QOS names.
- Recommended module stack for Python/CUDA/PyTorch.
- Whether VS Code Remote SSH is allowed on Fir login nodes under current policy.
- Whether containers such as Apptainer/Singularity are preferred.
- Exact GPU node count inconsistency: table says 4 H100s per GPU node; one paragraph says 2 H100s but then says 4 accelerators. Verify operationally.

---

## 18. Minimal first-day Fir checklist

```text
[ ] Confirm CCDB access to Fir.
[ ] SSH into fir.alliancecan.ca.
[ ] Confirm $HOME, $HOME/scratch, and $HOME/project/${def-project-id}.
[ ] Create repo folder.
[ ] Add CLUSTER_FIR.md and PROJECT_BRIEF.md.
[ ] Create Python venv or confirm module/container approach.
[ ] Run CPU smoke test.
[ ] Submit tiny MIG GPU smoke test.
[ ] Submit full H100 smoke test only after MIG works.
[ ] Decide dataset location under project space.
[ ] Use Globus or rsync/scp for data transfer.
[ ] Keep large files out of Git.
```

---

## 19. Compact summary for a coding agent

Fir is an Alliance Canada heterogeneous HPC cluster at SFU. Login is `fir.alliancecan.ca`; compute nodes have internet; `crontab` is not supported. Use Globus collection `alliancecan#fir-globus` for large transfers and the login node for `scp`/`rsync` until a transfer node is announced. Storage is Lustre: `$HOME` is small and backed up, `$HOME/scratch` is large temporary unbacked purged space, and `$HOME/project/${def-project-id}` is backed-up group/project storage for large datasets and checkpoints. CPU nodes have 192 cores; CPU tuning recommendation is `--cpus-per-task=8` and full-node `--ntasks-per-node=24`. GPU nodes have NVIDIA H100 SXM5 80 GB GPUs and support full H100 requests with `--gpus=h100:1` or `--gpus-per-node=h100:4`; MIG options are 10/20/40 GB using exact `nvidia_h100_80gb_hbm3_*` Slurm names. Start with smoke tests before serious training and verify account, partition, module, project path, and GPU topology directly on Fir.
