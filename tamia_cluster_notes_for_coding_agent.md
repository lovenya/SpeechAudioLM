# TamIA Cluster Notes for Coding Agent

Source: Alliance Canada / Calcul Québec documentation PDF uploaded by Lovenya: `TamIA - Alliance Doc.pdf`
Purpose: give a coding agent enough cluster-specific context to make sensible setup, transfer, Slurm, storage, and GPU decisions for TamIA.

> **Important:** This file is a distilled working note, not a replacement for the official TamIA documentation. When in doubt, check the official docs or ask Lovenya to provide the latest page/details.

---

## 1. Quick identity / cluster summary

- **Cluster name:** TamIA / tamIA
- **Availability date:** March 31, 2025
- **Login node:** `tamia.alliancecan.ca`
- **Host site:** Université Laval
- **Co-managed with:** Mila and Calcul Québec
- **Purpose:** AI-focused cluster for the Canadian scientific community
- **PAICE:** TamIA is part of the Pan-Canadian AI Compute Environment.
- **Name origin:** named after the eastern chipmunk / *Tamias*.

Useful public links:

- Official TamIA documentation snapshot/source page: `https://docs.alliancecan.ca/mediawiki/index.php?title=TamIA/en&oldid=199321`
- Université Laval: `http://www.ulaval.ca/`
- Mila: `https://mila.quebec/`
- Calcul Québec: `https://calculquebec.ca/`
- PAICE overview: `https://alliancecan.ca/en/services/advanced-research-computing/pan-canadian-ai-compute-environment-paice`
- Eastern chipmunk / Tamias: `https://en.wikipedia.org/wiki/Tamias`

---

## 2. Main endpoints and services

| Purpose | Endpoint / URL | Notes |
|---|---|---|
| Login node | `tamia.alliancecan.ca` | Main SSH login endpoint. |
| Data transfer node | `tamia.alliancecan.ca` | Official PDF says to use this for `rsync`, `scp`, `sftp`, etc. |
| Automation node | `robot.tamia.ecpia.ca` | For automation workflows if needed. |
| Globus collection | `TamIA's Globus v5 Server` | Official Globus collection name from the PDF. |
| Globus URL | `https://app.globus.org/file-manager?origin_id=72c3bca0-9281-4742-b066-333ba0fdef72` | Use for Globus transfers. |
| Portal | `https://portail.tamia.ecpia.ca/` | Use for job/resource monitoring. |
| CCDB access-services page | `https://ccdb.alliancecan.ca/me/access_services` | Use for requesting access. |
| PAICE declaration form | `https://ccdb.alliancecan.ca/paice/general_access_to_paice_systems` | Required after access request. |

---

## 3. Access notes

The PDF states:

1. Each researcher must complete an access request in CCDB:
   - `https://ccdb.alliancecan.ca/me/access_services`
2. Access can take up to one hour after the request is sent.
3. The user must then submit the **General Access to PAICE Systems** declaration form:
   - `https://ccdb.alliancecan.ca/paice/general_access_to_paice_systems`
4. Eligible principal investigators are members of an **AIP-type RAP**, with prefix `aip-`.
5. The cluster can only be reached from **Canada**.

Sponsoring other researchers, according to the PDF:

1. Go to the CCDB home page:
   - `https://ccdb.alliancecan.ca/`
2. Go to the **Resource Allocation Projects** table.
3. Find the RAPI of the `aip-` project and click it.
4. At the bottom of the RAP management page, click **Manage RAP memberships**.
5. Add a new member under **Add Members** using the user’s CCRI.

Coding-agent implication:

- Do **not** assume TamIA access until SSH succeeds.
- If SSH fails outside Canada, this may be due to the stated Canada-only reachability policy.
- If SSH fails with access/permission errors, ask Lovenya to verify:
  - CCDB access request
  - PAICE declaration form
  - membership in the correct `aip-` RAP/project
- Do not hard-code project paths until Lovenya confirms the actual project allocation.

---

## 4. Site-specific policies

Important cluster policies from the document:

- **Compute nodes cannot access the internet.**
- `crontab` is **not offered** on TamIA.
- **VS Code IDE is forbidden on login nodes** due to its heavy footprint.
- VS Code is **authorized on compute nodes**.
- Each normal job should be at least **1 hour**.
- Test jobs should be at least **5 minutes**.
- A user cannot have more than **1000 jobs**, running plus pending, at the same time.
- Maximum job duration is **1 day / 24 hours**.
- Each job must use all GPUs of the allocated server:
  - **H100 nodes:** use all **4** H100 GPUs.
  - **H200 nodes:** use all **8** H200 GPUs.

Coding-agent implications:

- Do **not** suggest installing dependencies from the internet inside compute jobs unless an exception has been granted.
- Prepare environments ahead of time using login-node-accessible methods, wheelhouses, modules, containers, or cluster-supported workflows.
- Do **not** suggest `cron`.
- Do **not** suggest running VS Code Remote directly on the login node.
- For VS Code workflows, use compute-node allocation if following policy.
- Every GPU job should be designed as a whole-node job:
  - H100 jobs: 4 GPUs.
  - H200 jobs: 8 GPUs.
- TamIA is better for serious multi-GPU training than tiny single-GPU experiments, because policy requires all GPUs on an allocated server.

---

## 5. Storage overview

The PDF lists HOME, SCRATCH, and PROJECT as Lustre filesystems.

| Storage area | Filesystem | Quota / backup | Intended use |
|---|---|---|---|
| HOME | Lustre | Small fixed per-user quota. PDF says there is currently no backup of home directories, with ETA Spring 2026. | Home directories, small configs/scripts only. Use project space for larger needs. |
| SCRATCH | Lustre | Large per-user quota. No backup. Older files are automatically purged. | Temporary files during computations. |
| PROJECT | Lustre | Large, adjustable per-group quota. PDF wording says there is currently no backup of the home directories, ETA Summer 2025; likely verify exact project backup status from current docs/support. | Sharing data among research group members and storing large datasets. |

Important uncertainty:

- The PDF’s storage text has potentially confusing backup wording under PROJECT: it says “There is currently no backup of the home directories. (ETA Summer 2025)” even though this row is about PROJECT. Treat backup status as something to verify before relying on it for irreplaceable results.

Coding-agent rules:

- Keep source code, configs, and small environment files in `$HOME` or a Git repo.
- Put large datasets, checkpoints, and shared artifacts under project space once the real project path is known.
- Use scratch for temporary, regenerable computation artifacts only.
- Do not store important one-off results only in scratch.
- Do not commit large datasets/checkpoints to Git.

Recommended repo hygiene:

```text
repo/
  README.md
  PROJECT_BRIEF.md
  CLUSTER_TAMIA.md
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

## 6. Network / high-performance interconnect

TamIA uses an **InfiniBand NVIDIA NDR** network.

Useful links from the PDF:

- InfiniBand overview: `https://fr.wikipedia.org/wiki/Bus_InfiniBand`
- NVIDIA Quantum-2 networking: `https://www.nvidia.com/en-us/networking/quantum2/`

Interconnect details from the PDF:

- All cluster nodes are linked by the InfiniBand NVIDIA NDR network.
- Each GPU is connected to a single **NDR200** port through an NVIDIA **ConnectX-7 HCA**.
- Each GPU server has **4 or 8 NDR200 ports** connected to the InfiniBand fabric.
- The InfiniBand network is **non-blocking for compute servers**.
- It is composed of two levels of switches in a **fat-tree topology**.
- Storage and compute nodes are connected to the network core via **4 or 8 400 Gb/s connections**.

Coding-agent implication:

- TamIA is designed for serious distributed AI workloads.
- Because jobs are whole-node GPU jobs, prefer training code that can actually use all GPUs on a node.
- For H100 nodes, expect 4-GPU training.
- For H200 nodes, expect 8-GPU training.
- Use single-node multi-GPU DDP first before attempting multi-node scaling.

---

## 7. Node characteristics

### 7.1 Node table

| Node count | Cores | Available memory | CPU | Local storage | GPU |
|---:|---:|---:|---|---|---|
| 12 | 64 | 1024 GB | 2 × Intel Xeon Gold 6448Y, 2.1 GHz, 32 cores each | 1 × 7.68 TB SSD | 8 × NVIDIA HGX H200 SXM, 141 GB HBM3, 700 W, connected via NVLink |
| 53 | 48 | 512 GB | 2 × Intel Xeon Gold 6442Y, 2.6 GHz, 24 cores each | 1 × 7.68 TB SSD | 4 × NVIDIA HGX H100 SXM, 80 GB HBM3, 700 W, connected via NVLink |
| 8 | 64 | 512 GB | 2 × Intel Xeon Gold 6438M, 2.2 GHz, 32 cores / 64 threads each | 1 × 7.68 TB SSD | none |

Useful CPU/GPU links from the PDF:

- Intel Xeon Gold 6448Y specs: `https://www.intel.com/content/www/us/en/products/sku/232384/intel-xeon-gold-6448y-processor-60m-cache-2-10-ghz/specifications.html`
- Intel Xeon Gold 6442Y specs: `https://www.intel.com/content/www/us/en/products/sku/232380/intel-xeon-gold-6442y-processor-60m-cache-2-60-ghz/specifications.html`
- Intel Xeon Gold 6438M specs: `https://www.intel.com/content/www/us/en/products/sku/232398/intel-xeon-gold-6438m-processor-60m-cache-2-20-ghz/specifications.html`
- NVIDIA H200: `https://www.nvidia.com/en-us/data-center/h200/`
- NVIDIA H100: `https://www.nvidia.com/en-us/data-center/h100/`

---

## 8. Software environments

The PDF states:

```bash
StdEnv/2023
```

is the standard environment on TamIA.

Coding-agent implication:

- Start by checking available modules:

```bash
module avail
module spider python
module spider cuda
module spider StdEnv
```

- A likely first module command may be:

```bash
module load StdEnv/2023
```

- Do not assume `conda`, `uv`, or internet-based installation is available on compute nodes.
- Prefer reproducible setup:
  - Python `venv` if supported.
  - Offline wheelhouse if packages need installation without internet.
  - Apptainer/Singularity if supported and appropriate.
  - Exact module list saved to a setup document.

Example module logging:

```bash
module list
module -t list > module_list.txt
```

---

## 9. GPU jobs

TamIA jobs are assigned on **whole nodes**.

### 9.1 H100 node request

For jobs on a node with H100 GPUs:

```bash
#SBATCH --gpus=h100:4
```

Operational meaning:

- Request one H100 node.
- The job must use all 4 H100 GPUs on the allocated server.

### 9.2 H200 node request

For jobs on a node with H200 GPUs:

```bash
#SBATCH --gpus=h200:8
```

Operational meaning:

- Request one H200 node.
- The job must use all 8 H200 GPUs on the allocated server.

### 9.3 Several GPU nodes

The PDF writes the multi-GPU-node form as:

```bash
#SBATCH --gpus-per-nodes=h100:4
#SBATCH --gpus-per-nodes=h200:8
```

Important note:

- In standard Slurm, the common option is usually `--gpus-per-node`, singular `node`, not `--gpus-per-nodes`.
- The PDF text includes `--gpus-per-nodes`. This may be a typo in the documentation or a site-specific accepted alias.
- Verify with TamIA docs/support or `sbatch --help` before finalizing multi-node scripts.

Safer standard Slurm spelling to test/verify:

```bash
#SBATCH --gpus-per-node=h100:4
#SBATCH --gpus-per-node=h200:8
```

Coding-agent implication:

- Do not submit single-GPU jobs on TamIA.
- If the model or experiment cannot use 4 GPUs efficiently, prefer Fir/Rorqual MIG/full-GPU resources for small tests.
- Use TamIA when the training code is ready for whole-node distributed execution.

---

## 10. Practical Slurm templates

These are starter templates. They may need account names, partition names, QoS, mail settings, and module names once Lovenya confirms the TamIA environment.

### 10.1 H100 4-GPU node smoke test

```bash
#!/bin/bash
#SBATCH --job-name=tamia_h100_smoke
#SBATCH --time=00:10:00
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=4
#SBATCH --cpus-per-task=8
#SBATCH --gpus=h100:4
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

module load StdEnv/2023 || true
module list || true

nvidia-smi

python - <<'PY'
import os
print("CUDA_VISIBLE_DEVICES:", os.environ.get("CUDA_VISIBLE_DEVICES"))
try:
    import torch
    print("torch:", torch.__version__)
    print("cuda available:", torch.cuda.is_available())
    print("device count:", torch.cuda.device_count())
    for i in range(torch.cuda.device_count()):
        print(i, torch.cuda.get_device_name(i))
except Exception as e:
    print("torch check failed:", repr(e))
PY
```

### 10.2 H100 4-GPU PyTorch DDP starter

```bash
#!/bin/bash
#SBATCH --job-name=tamia_h100_ddp
#SBATCH --time=12:00:00
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=4
#SBATCH --cpus-per-task=8
#SBATCH --gpus=h100:4
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

module load StdEnv/2023 || true

nvidia-smi

# Example only. Replace with actual environment activation.
# source ~/.venvs/project/bin/activate

srun python -m torch.distributed.run \
  --nproc_per_node=4 \
  train.py \
  --config configs/train.yaml
```

### 10.3 H200 8-GPU PyTorch DDP starter

```bash
#!/bin/bash
#SBATCH --job-name=tamia_h200_ddp
#SBATCH --time=12:00:00
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=8
#SBATCH --cpus-per-task=8
#SBATCH --gpus=h200:8
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

module load StdEnv/2023 || true

nvidia-smi

# Example only. Replace with actual environment activation.
# source ~/.venvs/project/bin/activate

srun python -m torch.distributed.run \
  --nproc_per_node=8 \
  train.py \
  --config configs/train.yaml
```

Notes:

- These templates assume one PyTorch process per GPU.
- Validate with a tiny DDP smoke test before running a real training job.
- Since compute nodes have no internet, all dependencies must already be available.

---

## 11. Data transfer guidance

### 11.1 Globus

Use:

- Collection name: `TamIA's Globus v5 Server`
- Globus link: `https://app.globus.org/file-manager?origin_id=72c3bca0-9281-4742-b066-333ba0fdef72`

Good for:

- Large files
- Many files
- Reliable resumable transfers
- Transfers between Alliance systems or supported endpoints

### 11.2 SCP / SFTP / rsync

Use:

```bash
tamia.alliancecan.ca
```

Examples:

```bash
scp local_file.zip USER@tamia.alliancecan.ca:/path/to/destination/
```

```bash
rsync -avh --progress local_dir/ USER@tamia.alliancecan.ca:/path/to/destination/
```

Coding-agent implication:

- Prefer Globus for large dataset movement.
- For small files/scripts, `scp` or `rsync` to `tamia.alliancecan.ca` is fine.
- Avoid heavy data processing on the login node.
- Use Slurm jobs for unzipping, hashing, preprocessing, manifest building, and training.

---

## 12. Environment setup guidance for no-internet compute nodes

Because TamIA compute nodes cannot access the internet, avoid workflows that require runtime downloads.

### 12.1 Good patterns

- Use cluster modules where possible.
- Build a Python virtual environment ahead of time if packages are available from the login node or a configured package mirror.
- Use a wheelhouse:
  - Download/build wheels somewhere with internet.
  - Transfer wheels to TamIA.
  - Install with `pip --no-index --find-links`.
- Use Apptainer/Singularity if supported.
- Cache model weights/datasets manually in project space before training.

### 12.2 Example wheelhouse install

```bash
python -m venv ~/.venvs/speechaudio
source ~/.venvs/speechaudio/bin/activate

python -m pip install --upgrade pip

python -m pip install \
  --no-index \
  --find-links /path/to/wheelhouse \
  -r requirements.txt
```

### 12.3 Avoid these unless explicitly supported

```bash
pip install some_package
git clone https://...
wget https://...
curl https://...
huggingface-cli download ...
```

inside a compute job, because the compute nodes cannot access the internet.

Coding-agent implication:

- If code needs pretrained weights from Hugging Face, download them elsewhere first, transfer them, and point environment variables/cache paths to project storage.
- Explicitly set cache directories if needed:

```bash
export HF_HOME=/path/to/project/cache/huggingface
export TRANSFORMERS_CACHE=/path/to/project/cache/huggingface/transformers
export TORCH_HOME=/path/to/project/cache/torch
```

---

## 13. Monitoring jobs

TamIA provides a portal:

```text
https://portail.tamia.ecpia.ca/
```

According to the PDF, from the portal the user can:

- Monitor running jobs in real time.
- Examine past jobs.
- Monitor usage of:
  - compute nodes
  - memory
  - GPU

The document emphasizes correcting resource requests when resources are underused. Example: if a job requests 4 CPU cores but uses only 1, adjust the script accordingly.

Coding-agent implication:

- After a job, inspect actual CPU/GPU/memory utilization.
- Reduce overrequested CPU/memory resources when possible.
- Make sure all 4 H100 or all 8 H200 GPUs are actually doing useful work.
- Avoid occupying whole expensive nodes for code that only uses one GPU.

---

## 14. Checks a coding agent should run on TamIA

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
sbatch --help | grep -i gpu -A 5 -B 5 || true
```

Module discovery:

```bash
module avail
module spider StdEnv
module spider python
module spider cuda
module list
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

DDP smoke test idea:

```bash
srun python - <<'PY'
import os, socket
print("host", socket.gethostname(), "rank", os.environ.get("SLURM_PROCID"), "localid", os.environ.get("SLURM_LOCALID"), "cuda", os.environ.get("CUDA_VISIBLE_DEVICES"))
PY
```

---

## 15. Coding-agent decision rules

When helping Lovenya on TamIA:

1. **Never plan internet-dependent compute jobs.**
   - Compute nodes cannot access the internet.
   - Pre-download packages, model weights, and datasets.

2. **Do not use VS Code on login nodes.**
   - The official document says VS Code IDE is forbidden on login nodes.
   - VS Code is allowed on compute nodes.

3. **Use whole GPU nodes.**
   - H100 jobs must use 4 GPUs.
   - H200 jobs must use 8 GPUs.

4. **TamIA is not ideal for tiny single-GPU debugging.**
   - Prefer Fir/Rorqual/MIG resources for tiny tests.
   - Use TamIA when code is ready to use a whole GPU node.

5. **Respect the 24-hour maximum job duration.**
   - Use checkpointing.
   - Make training resumable.
   - Save checkpoints often enough to survive preemption/failures/24h boundaries.

6. **Make training resumable by default.**
   - Required because max walltime is 24h.
   - Save:
     - model state
     - optimizer state
     - scheduler state
     - global step/epoch
     - config
     - random seeds if possible

7. **Use the portal after jobs.**
   - Check GPU utilization.
   - Check memory use.
   - Check CPU use.
   - Adjust Slurm requests.

8. **Avoid heavy login-node work.**
   - Do not preprocess huge datasets on login.
   - Do not launch training on login.
   - Use Slurm jobs.

9. **Ask before deleting/moving large data.**
   - Especially anything in project space.

---

## 16. Links inventory from the TamIA PDF

Use these when the coding agent needs to point Lovenya to official docs or ask for more information.

| Topic | Link |
|---|---|
| TamIA official documentation snapshot/source | `https://docs.alliancecan.ca/mediawiki/index.php?title=TamIA/en&oldid=199321` |
| TamIA Globus endpoint | `https://app.globus.org/file-manager?origin_id=72c3bca0-9281-4742-b066-333ba0fdef72` |
| TamIA portal | `https://portail.tamia.ecpia.ca/` |
| Université Laval | `http://www.ulaval.ca/` |
| Mila | `https://mila.quebec/` |
| Calcul Québec | `https://calculquebec.ca/` |
| PAICE overview | `https://alliancecan.ca/en/services/advanced-research-computing/pan-canadian-ai-compute-environment-paice` |
| CCDB access services | `https://ccdb.alliancecan.ca/me/access_services` |
| CCDB home | `https://ccdb.alliancecan.ca/` |
| PAICE declaration form | `https://ccdb.alliancecan.ca/paice/general_access_to_paice_systems` |
| InfiniBand overview | `https://fr.wikipedia.org/wiki/Bus_InfiniBand` |
| NVIDIA Quantum-2 networking | `https://www.nvidia.com/en-us/networking/quantum2/` |
| Intel Xeon Gold 6448Y specs | `https://www.intel.com/content/www/us/en/products/sku/232384/intel-xeon-gold-6448y-processor-60m-cache-2-10-ghz/specifications.html` |
| Intel Xeon Gold 6442Y specs | `https://www.intel.com/content/www/us/en/products/sku/232380/intel-xeon-gold-6442y-processor-60m-cache-2-60-ghz/specifications.html` |
| Intel Xeon Gold 6438M specs | `https://www.intel.com/content/www/us/en/products/sku/232398/intel-xeon-gold-6438m-processor-60m-cache-2-20-ghz/specifications.html` |
| NVIDIA H200 | `https://www.nvidia.com/en-us/data-center/h200/` |
| NVIDIA H100 | `https://www.nvidia.com/en-us/data-center/h100/` |
| VS Code | `https://code.visualstudio.com/` |

---

## 17. Open questions / verify before finalizing scripts

The coding agent should ask Lovenya or check TamIA directly for:

- Actual username on TamIA.
- Actual project allocation path.
- Correct Slurm account name, if required.
- Available partitions/QoS names for H100 and H200.
- Whether `--gpus-per-nodes` in the PDF is accepted, or whether standard `--gpus-per-node` should be used.
- Recommended module stack for Python/CUDA/PyTorch.
- Whether Apptainer/Singularity is available and preferred.
- How to use VS Code on compute nodes in a policy-compliant way.
- Actual backup status for PROJECT storage, because the PDF wording is confusing.
- Whether there are site-specific examples for PyTorch DDP.
- Best practice for transferring large model checkpoints and Hugging Face caches.

---

## 18. Minimal first-day TamIA checklist

```text
[ ] Confirm CCDB access to TamIA.
[ ] Submit PAICE declaration form.
[ ] Confirm access from inside Canada.
[ ] SSH into tamia.alliancecan.ca.
[ ] Confirm project path and quota.
[ ] Confirm $HOME, scratch, and project storage paths.
[ ] Create repo folder.
[ ] Add CLUSTER_TAMIA.md and PROJECT_BRIEF.md.
[ ] Check available modules: StdEnv/2023, Python, CUDA, PyTorch.
[ ] Decide environment strategy: module, venv + offline wheels, or container.
[ ] Run CPU smoke test.
[ ] Submit H100 4-GPU smoke test only when code can use all 4 GPUs.
[ ] Submit H200 8-GPU smoke test only when code can use all 8 GPUs.
[ ] Verify DDP process/GPU mapping.
[ ] Set up checkpoint/resume before long training.
[ ] Monitor jobs using https://portail.tamia.ecpia.ca/.
[ ] Keep large files out of Git.
```

---

## 19. Compact summary for a coding agent

TamIA is an Alliance/PAICE AI cluster at Université Laval, co-managed with Mila and Calcul Québec. Login and data transfer use `tamia.alliancecan.ca`; Globus uses `TamIA's Globus v5 Server`; the monitoring portal is `https://portail.tamia.ecpia.ca/`. Compute nodes have **no internet**, `crontab` is unavailable, and VS Code is forbidden on login nodes but allowed on compute nodes. Jobs must be at least 1 hour normally, 5 minutes for tests, with a 24-hour maximum. GPU jobs are whole-node: H100 jobs use all 4 H100 GPUs with `--gpus=h100:4`, and H200 jobs use all 8 H200 GPUs with `--gpus=h200:8`. TamIA is best for serious multi-GPU AI training, not tiny single-GPU debugging. Prepare dependencies, datasets, and model weights before compute jobs; make training resumable; monitor actual CPU/GPU/memory usage through the TamIA portal and adjust resource requests accordingly.
