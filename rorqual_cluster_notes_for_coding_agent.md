# Rorqual Cluster Notes for Coding Agent

> Source: `Rorqual - Alliance Doc.pdf`, an Alliance/Calcul Québec Rorqual documentation snapshot.
> Availability listed in the source: **June 19, 2025**.
> Source page last edited: **April 1, 2026, 18:28**.
> Treat these notes as cluster-specific operational context. The live docs may change, so verify against the official links when needed.

---

## 1. Cluster identity and main endpoints

Rorqual is a heterogeneous Alliance/Calcul Québec cluster located at the École de technologie supérieure in Montreal. It is intended for a wide variety of scientific computing workloads.

| Purpose | Host / collection / URL |
|---|---|
| Login node | `rorqual.alliancecan.ca` |
| Data transfer node for `rsync`, `scp`, `sftp`, etc. | `rorqual.alliancecan.ca` |
| Automation node | `robot.rorqual.alliancecan.ca` |
| Globus collection | `alliancecan#rorqual` |
| Globus File Manager link | https://app.globus.org/file-manager?origin_id=f19f13f5-5553-40e3-ba30-6c151b9d35d4 |
| JupyterHub | https://jupyterhub.rorqual.alliancecan.ca/ |
| Portal / metrics | https://metrix.rorqual.alliancecan.ca/ |
| Original retrieved Rorqual docs snapshot | https://docs.alliancecan.ca/mediawiki/index.php?title=Rorqual/en&oldid=201614 |

Practical login/transfer forms:

```bash
ssh <username>@rorqual.alliancecan.ca
scp local_file <username>@rorqual.alliancecan.ca:/path/on/cluster/
rsync -avP local_dir/ <username>@rorqual.alliancecan.ca:/path/on/cluster/
sftp <username>@rorqual.alliancecan.ca
```

---

## 2. Access request

Access must be requested through CCDB:

- CCDB access systems page: https://ccdb.alliancecan.ca/me/access_systems

Steps from the source document:

1. Go to **Resources -> Access Systems** in CCDB.
2. Select **Rorqual** from the list.
3. Select **I request access**.
4. Accept the required agreements:
   - Calcul Québec Consent for the collection and use of personal information
   - Rorqual Service Level Agreement
   - Calcul Québec Terms of Use
5. Access may take up to **one hour** to be enabled.

---

## 3. High-priority operational constraints

These are the constraints a coding agent should assume by default when planning jobs, installs, and workflows.

### 3.1 Compute nodes have no internet

Rorqual compute nodes **cannot access the internet**. If an exception is needed, the user must contact technical support and explain the need.

Implications for coding/setup:

- Do **not** assume `pip install`, `git clone`, `wget`, `curl`, Hugging Face downloads, or model checkpoint downloads will work inside a compute job.
- Prefer preparing dependencies on a login/transfer-accessible environment or using pre-downloaded wheels, source trees, caches, datasets, and model checkpoints.
- For Python workflows, prefer reproducible offline installation patterns, for example:

```bash
# Example pattern only; exact paths/packages should be project-specific.
python -m venv ~/venvs/myenv
source ~/venvs/myenv/bin/activate
pip install --no-index --find-links=/path/to/wheelhouse -r requirements.txt
```

Relevant docs:

- Technical support: https://docs.alliancecan.ca/wiki/Technical_support

### 3.2 No `crontab`

The `crontab` tool is not offered on Rorqual.

Implications:

- Do not design workflows around user-level cron jobs on this cluster.
- Use Slurm jobs, job arrays, external scheduling, or the provided automation node where appropriate.

Relevant docs:

- Automation with multifactor authentication: https://docs.alliancecan.ca/wiki/Automation_in_the_context_of_multifactor_authentication

### 3.3 Job duration and queue limits

Cluster policy from the source:

- Each job should have duration of at least **1 hour**.
- Test jobs may be as short as **5 minutes**.
- The user cannot have more than **1000 jobs**, running or queued, at any given moment.
- Maximum job duration is **7 days / 168 hours**.

Implications:

- Avoid submitting huge numbers of tiny jobs.
- Use job arrays carefully and stay below the 1000 running/queued job limit.
- For quick checks, mark them as test jobs and keep them at least 5 minutes where possible.
- For long training, checkpoint regularly because jobs cannot exceed 168 hours.

---

## 4. Storage layout and intended use

Rorqual has three main storage spaces mentioned in the source.

| Space | Filesystem / size | Access path | Backup | Intended use | Notes |
|---|---:|---|---|---|---|
| `HOME` | Lustre, 116 TB total | `$HOME` | Daily automatic backup | Small personal files, configs, scripts | Small per-user quotas. Cannot be increased. Use `/project` for large storage. |
| `SCRATCH` | Lustre, 6.5 PB | `$HOME/links/scratch` | No backup | Temporary files during computations | Large per-user quotas. Older files are automatically purged. Do not store irreplaceable data here. |
| `PROJECT` | Lustre, 62 PB | `$HOME/links/projects/nom-du-projet` | Daily backup | Shared group data and large datasets | Large and adjustable per-group quotas. Best place for shared datasets/checkpoints that should persist. |

Practical guidance for coding agents:

- Keep source code, small configs, shell scripts, and lightweight logs in `$HOME` or a project repo.
- Keep large datasets, shared assets, and important checkpoints in `/project` / `$HOME/links/projects/<project-name>`.
- Use `$SCRATCH` / `$HOME/links/scratch` for temporary intermediates, extracted temporary files, cache-like data, and disposable job outputs.
- Never assume `$SCRATCH` is permanent.
- Avoid writing huge logs, caches, or datasets into `$HOME`.

Relevant docs:

- Filesystem quotas and policies: https://docs.alliancecan.ca/wiki/Storage_and_file_management#Filesystem_quotas_and_policies
- Scratch purging policy: https://docs.alliancecan.ca/wiki/Scratch_purging_policy
- Storage quotas/policies, French anchor from source: https://docs.alliancecan.ca/wiki/Storage_and_file_management#Quotas_et_politiques

### Data transfer

For data transfer:

- Use the Rorqual Globus endpoint for Globus transfers.
- For `rsync`, `scp`, and `sftp`, use the login node: `rorqual.alliancecan.ca`.

Relevant docs:

- Globus: https://docs.alliancecan.ca/wiki/Globus/fr
- Rsync: https://docs.alliancecan.ca/wiki/Transferring_data#Rsync
- SCP: https://docs.alliancecan.ca/wiki/Transferring_data#SCP

---

## 5. Interconnect

Rorqual uses an InfiniBand interconnect:

- HDR 200 Gb/s
- Maximum blocking factor: `34:6`, or `5.667:1`
- CPU node island size: up to 31 nodes of 192 cores, fully non-blocking

Implications:

- Multi-node CPU or distributed workloads may benefit from the high-performance interconnect.
- For distributed GPU training, still verify the exact Slurm/network/NCCL environment before assuming performance.

---

## 6. Node characteristics

### 6.1 CPU nodes

| Node count | Cores per node | Available memory | Local storage | CPU | GPU |
|---:|---:|---:|---|---|---|
| 670 | 192 | 750G / 768000M | 1 x SATA SSD, 480G, 6 Gbit/s | 2 x AMD EPYC 9654, Zen 4, 2.40 GHz, 384 MB L3 cache | None listed |
| 8 | 192 | 750G / 768000M | 1 x NVMe SSD, 3.84 TB | 2 x AMD EPYC 9654, Zen 4, 2.40 GHz, 384 MB L3 cache | None listed |
| 8 | 192 | 3013G / 3086250M | 1 x SATA SSD, 480G, 6 Gbit/s | 2 x AMD EPYC 9654, Zen 4, 2.40 GHz, 384 MB L3 cache | None listed |

AMD EPYC 9654 reference:
https://www.amd.com/en/support/downloads/drivers.html/processors/epyc/epyc-9004-series/amd-epyc-9654.html

### 6.2 GPU nodes

| Node count | Cores per node | Available memory | Local storage | CPU | GPU |
|---:|---:|---:|---|---|---|
| 93 | 64 | 498G / 510000M | 1 x NVMe SSD, 3.84 TB | 2 x Intel Xeon Gold 6448Y, 2.10 GHz, 60 MB L3 cache | 4 x NVIDIA H100 SXM5, 80 GB, connected via NVLink |

Intel Xeon Gold 6448Y reference:
https://ark.intel.com/content/www/us/en/ark/products/232384/intel-xeon-gold-6448y-processor-60m-cache-2-10-ghz.html

SXM reference:
<https://en.wikipedia.org/wiki/SXM_(socket)>

### 6.3 `$SLURM_TMPDIR` sizing

To request larger `$SLURM_TMPDIR`, a job can be submitted with:

```bash
#SBATCH --tmp=<x>G
```

where `<x>` is between **370** and **3360**.

Example:

```bash
#SBATCH --tmp=1000G
```

---

## 7. CPU topology notes

The 192-core AMD CPU nodes are NUMA/hierarchical. The different cores and memory spaces are not equidistant, causing variable memory access delays.

Topology from source:

- 2 sockets per node
- Each socket has 12 system memory channels
- 4 NUMA nodes per socket
- Each NUMA node is connected to 3 system memory channels
- 3 chiplets per NUMA node
- Each chiplet has its own 32 MiB L3 cache
- 8 cores per chiplet
- Each core has its own 1 MiB L2 cache and 32+32 KiB L1 cache

Summarized:

- Groups of **8 closely spaced cores** share one L3 cache.
- A good pattern for multithreaded programs may be `--cpus-per-task=8`.
- NUMA groups: `3 x 8 = 24` cores sharing a trio of system memory channels.
- Full node: `2 x 4 x 3 x 8 = 192` cores.

For full-node CPU jobs, the source suggests patterns such as:

```bash
#SBATCH --ntasks-per-node=24
#SBATCH --cpus-per-task=8
```

For threaded/OpenMP-style programs, also consider setting thread-related environment variables, for example:

```bash
export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK
```

Actual binding/placement should be controlled explicitly for performance-sensitive programs.

Relevant docs:

- NUMA: https://en.wikipedia.org/wiki/Non-uniform_memory_access
- CPU cache: https://en.wikipedia.org/wiki/CPU_cache
- Threaded / OpenMP jobs: https://docs.alliancecan.ca/wiki/Running_jobs#Threaded_or_OpenMP_job

---

## 8. GPU topology notes

GPU nodes are described as less hierarchical than CPU nodes.

From source:

- 2 sockets per GPU node
- Each socket has:
  - 8 system memory channels
  - 60 MiB L3 cache memory
  - 32 equidistant cores
  - Each core has its own 2 MiB L2 cache and 32+48 KiB L1 cache
  - 2 NVIDIA H100 accelerators
- Each GPU node has 4 NVIDIA H100 SXM5 80 GB accelerators connected via NVLink.

Practical guidance:

- For small debugging or low-memory jobs, consider a MIG slice if appropriate.
- For serious deep learning training, full H100 requests are likely more appropriate than MIG slices.
- For multi-GPU training, be explicit about Slurm GPU requests and verify distributed-training configuration.

---

## 9. GPU instance names and Slurm request options

### 9.1 Available GPU/MIG instance names

| Type | Model or instance | Short name | Without unit | By memory | Long name |
|---|---|---|---|---|---|
| Full GPU | H100-80gb | `h100` | `h100` | `h100_80gb` | `nvidia_h100_80gb_hbm3` |
| MIG | H100-1g.10gb | `h100_1g.10gb` | `h100_1.10` | `h100_10gb` | `nvidia_h100_80gb_hbm3_1g.10gb` |
| MIG | H100-2g.20gb | `h100_2g.20gb` | `h100_2.20` | `h100_20gb` | `nvidia_h100_80gb_hbm3_2g.20gb` |
| MIG | H100-3g.40gb | `h100_3g.40gb` | `h100_3.40` | `h100_40gb` | `nvidia_h100_80gb_hbm3_3g.40gb` |

Relevant docs:

- Multi-Instance GPU: https://docs.alliancecan.ca/wiki/Multi-Instance_GPU
- Ratios in bundles / allocations and scheduling: https://docs.alliancecan.ca/wiki/Allocations_and_compute_scheduling#Ratios_in_bundles

### 9.2 Requesting full H100 GPUs

Request one full H100 80 GB:

```bash
#SBATCH --gpus=h100:1
# or
#SBATCH --gpus=h100_80gb:1
```

Request multiple full H100s on the same node:

```bash
#SBATCH --gpus-per-node=h100:2
#SBATCH --gpus-per-node=h100:3
#SBATCH --gpus-per-node=h100:4
```

Request multiple full H100s spread anywhere:

```bash
#SBATCH --gpus=h100:<n>
```

where `<n>` is the number of GPUs requested.

### 9.3 Requesting MIG GPU instances

Approximately half of the GPU nodes are configured with MIG technology. The source lists only three GPU instance sizes:

| MIG instance | Compute fraction | GPU memory |
|---|---:|---:|
| H100-1g.10gb | 1/8 of H100 compute | 10 GB |
| H100-2g.20gb | 2/8 of H100 compute | 20 GB |
| H100-3g.40gb | 3/8 of H100 compute | 40 GB |

To request one and only one GPU instance for a compute job:

```bash
#SBATCH --gpus=h100_1g.10gb:1   # 10 GB MIG instance
#SBATCH --gpus=h100_2g.20gb:1   # 20 GB MIG instance
#SBATCH --gpus=h100_3g.40gb:1   # 40 GB MIG instance
```

The source mentions that the maximum recommended number of CPU cores and system memory per GPU instance is listed in an external/linked table, but that table is not included in the provided PDF snapshot.

---

## 10. Starter Slurm templates

These are not directly copied from the docs; they are practical templates based on the documented Rorqual constraints. Adjust account, time, memory, CPUs, and paths for the actual project.

### 10.1 Small GPU debug job with a 10 GB MIG slice

```bash
#!/bin/bash
#SBATCH --job-name=debug_mig10
#SBATCH --time=00:10:00
#SBATCH --gpus=h100_1g.10gb:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --tmp=370G
#SBATCH --output=logs/%x-%j.out
#SBATCH --error=logs/%x-%j.err

set -euo pipefail

source ~/venvs/myenv/bin/activate
python -u scripts/smoke_test.py
```

### 10.2 Single full H100 job

```bash
#!/bin/bash
#SBATCH --job-name=train_h100
#SBATCH --time=12:00:00
#SBATCH --gpus=h100:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --tmp=500G
#SBATCH --output=logs/%x-%j.out
#SBATCH --error=logs/%x-%j.err

set -euo pipefail

source ~/venvs/myenv/bin/activate
python -u train.py --config configs/train.yaml
```

### 10.3 Four-H100 single-node job

```bash
#!/bin/bash
#SBATCH --job-name=train_4xh100
#SBATCH --time=24:00:00
#SBATCH --gpus-per-node=h100:4
#SBATCH --cpus-per-task=16
#SBATCH --mem=256G
#SBATCH --tmp=1000G
#SBATCH --output=logs/%x-%j.out
#SBATCH --error=logs/%x-%j.err

set -euo pipefail

source ~/venvs/myenv/bin/activate
python -u train_distributed.py --config configs/train_4gpu.yaml
```

### 10.4 CPU threaded job pattern

```bash
#!/bin/bash
#SBATCH --job-name=cpu_threaded
#SBATCH --time=04:00:00
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --output=logs/%x-%j.out
#SBATCH --error=logs/%x-%j.err

set -euo pipefail

export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK
source ~/venvs/myenv/bin/activate
python -u scripts/cpu_job.py
```

### 10.5 Full CPU node topology-aware pattern

```bash
#!/bin/bash
#SBATCH --job-name=cpu_full_node
#SBATCH --time=06:00:00
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=24
#SBATCH --cpus-per-task=8
#SBATCH --mem=0
#SBATCH --output=logs/%x-%j.out
#SBATCH --error=logs/%x-%j.err

set -euo pipefail

export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK
source ~/venvs/myenv/bin/activate
srun python -u scripts/cpu_distributed_or_hybrid.py
```

---

## 11. Coding-agent checklist for Rorqual

Before suggesting commands or writing scripts for this cluster, check the following:

1. **Internet assumption**: Does the command need internet? If yes, do not put it inside a compute job unless the dependency/data is already staged.
2. **Storage location**: Is the command writing large files? Use `/project` or `$SCRATCH`, not `$HOME`.
3. **Persistence**: Is the output important? Do not leave the only copy in `$SCRATCH`.
4. **Transfer method**: For `scp`, `sftp`, or `rsync`, use `rorqual.alliancecan.ca`. For large transfers, consider Globus.
5. **Job duration**: Avoid jobs shorter than 1 hour except test jobs of at least 5 minutes.
6. **Queue scale**: Do not submit more than 1000 running/queued jobs.
7. **Long runs**: Maximum duration is 168 hours; training code should checkpoint.
8. **GPU choice**: Use MIG slices for small/debug jobs; full H100 for serious training.
9. **CPU topology**: For CPU-heavy threaded jobs, consider `--cpus-per-task=8`; for full-node jobs, consider `--ntasks-per-node=24 --cpus-per-task=8`.
10. **Local temp**: If the job needs substantial local temporary disk, request `--tmp=<x>G` with `370 <= x <= 3360`.

---

## 12. Full link inventory from the source PDF

The following links were present in the PDF annotations or parsed text. Keep this section so the coding agent knows the relevant docs exist, even if it cannot open them directly.

- Automation with multifactor authentication: https://docs.alliancecan.ca/wiki/Automation_in_the_context_of_multifactor_authentication
- Globus File Manager for Rorqual: https://app.globus.org/file-manager?origin_id=f19f13f5-5553-40e3-ba30-6c151b9d35d4
- Rorqual JupyterHub: https://jupyterhub.rorqual.alliancecan.ca/
- Rorqual Portal / metrics: https://metrix.rorqual.alliancecan.ca/
- Rorqual webinar slides: https://docs.google.com/presentation/d/1Ah61BBKZIJcn_AeosgUspxRCX_amubPUetZY68SfyXU
- Rorqual webinar video: https://www.youtube.com/watch?v=lXetzrViI8Q
- École de technologie supérieure: https://www.etsmtl.ca/en/
- Rorqual marine mammal reference: https://en.wikipedia.org/wiki/Rorqual
- CCDB access systems: https://ccdb.alliancecan.ca/me/access_systems
- Alliance technical support: https://docs.alliancecan.ca/wiki/Technical_support
- Storage and file management, filesystem quotas and policies: https://docs.alliancecan.ca/wiki/Storage_and_file_management#Filesystem_quotas_and_policies
- Scratch purging policy: https://docs.alliancecan.ca/wiki/Scratch_purging_policy
- Storage and file management, quotas/policies French anchor from source: https://docs.alliancecan.ca/wiki/Storage_and_file_management#Quotas_et_politiques
- Globus docs: https://docs.alliancecan.ca/wiki/Globus/fr
- Rsync docs: https://docs.alliancecan.ca/wiki/Transferring_data#Rsync
- SCP docs: https://docs.alliancecan.ca/wiki/Transferring_data#SCP
- AMD EPYC 9654 reference: https://www.amd.com/en/support/downloads/drivers.html/processors/epyc/epyc-9004-series/amd-epyc-9654.html
- Intel Xeon Gold 6448Y reference: https://ark.intel.com/content/www/us/en/ark/products/232384/intel-xeon-gold-6448y-processor-60m-cache-2-10-ghz.html
- NUMA reference: https://en.wikipedia.org/wiki/Non-uniform_memory_access
- CPU cache reference: https://en.wikipedia.org/wiki/CPU_cache
- Threaded / OpenMP job docs: https://docs.alliancecan.ca/wiki/Running_jobs#Threaded_or_OpenMP_job
- SXM socket reference: <https://en.wikipedia.org/wiki/SXM_(socket)>
- Multi-Instance GPU docs: https://docs.alliancecan.ca/wiki/Multi-Instance_GPU
- Allocations and compute scheduling, ratios in bundles: https://docs.alliancecan.ca/wiki/Allocations_and_compute_scheduling#Ratios_in_bundles
- Retrieved Rorqual docs snapshot: https://docs.alliancecan.ca/mediawiki/index.php?title=Rorqual/en&oldid=201614
