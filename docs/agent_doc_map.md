# Agent Documentation Map

Status: lightweight guide for future agent sessions. Read this first when
restarting work on results, inference, evaluation, or cluster resource planning.

## Purpose

This repo has several planning and handoff docs. Use this file as the routing
map so future work starts from the right source instead of rediscovering old
layouts or demo scripts.

## Core Docs

- `RESULTS_STRUCTURE_DESIGN.md`: source of truth for result roots, log roots,
  temporary-vs-canonical outputs, run IDs, cascaded stage run directories, and
  what downstream stages should consume.
- `EVALS_STRUCTURE_DESIGN.md`: source of truth for evaluation result layout,
  metric-specific eval directories, target-result references, eval run files,
  and eval simplification decisions.
- `configs/project_paths.json`: machine-local named path registry. On this
  machine, real runs should resolve results, logs, temporary outputs, exports,
  datasets, models, caches, and external sources under
  `/scratch/l/lovenya/salm_migration_clean` where applicable.
- `docs/tamia_resource_usage.json`: measured TamIA resource usage and job audit
  evidence. Use this before requesting new Slurm resources.
- `configs/slurm_resource_profiles.json`: starting resource request profiles.
  Update these after checking measured resource summaries.
- `HPC_BEST_PRACTICES.md`: cluster operating rules, login-vs-compute policy,
  resource monitoring expectations, environment rules, and Slurm hygiene.
- `CONTEXT.md`: current handoff state and active assumptions. Treat it as
  session state, not a permanent design spec.
- `PROGRESS_LOG.md`: chronological history of decisions, experiments, failures,
  and why previous paths were chosen.

## Results And Inference Flow

For inference or generation work, read `RESULTS_STRUCTURE_DESIGN.md` first.
The important contract is:

1. canonical outputs go under `{results_root}` and `{logs_root}`;
2. demo, smoke, fit-test, and partial/debug outputs go under temporary roots;
3. each run writes a stable run directory with metadata, config, manifest,
   optional outputs, inspect data, errors, and summary as applicable;
4. downstream stages should consume the previous stage manifest and lineage,
   not ad hoc output directories;
5. generated media stays outside Git.

Current target roots on this machine:

```text
/scratch/l/lovenya/salm_migration_clean/results
/scratch/l/lovenya/salm_migration_clean/logs
/scratch/l/lovenya/salm_migration_clean/temporary/results
/scratch/l/lovenya/salm_migration_clean/temporary/logs
```

## Eval Flow

For evaluation work, read `EVALS_STRUCTURE_DESIGN.md` after the results doc.
The important contract is:

1. eval outputs live under `{results_root}/evals/...` and `{logs_root}/evals/...`;
2. eval paths are organized by the exact result run being evaluated;
3. metric IDs appear directly in paths;
4. each metric gets its own eval run directory;
5. metric choice must be explicit; avoid hidden default metric bundles;
6. target design removes eval sharding, eval shard merge scripts, and
   parallel-eval flexibility in the metric runner;
7. temporary backend checks are temporary eval runs, not canonical results.

Do not extend legacy repo-local layouts such as `results/evals/T2S/`.

## Resource Planning

Before asking for Slurm resources, check measured evidence first:

- `docs/tamia_resource_usage.json`
- `/scratch/l/lovenya/salm_migration_clean/logs/resources/*_summary.json`
- `/scratch/l/lovenya/salm_migration_clean/logs/resources/*.jsonl`
- Slurm accounting for the relevant job IDs when available

Then compare against `configs/slurm_resource_profiles.json` and update requests
only when the measurements justify it. For TamIA, remember that H100 GPU jobs
reserve a whole 4-GPU node, so packing independent compatible work can be useful
for inference, but canonical eval design should not rely on eval sharding or
parallel-eval flexibility.

## Cleanup Policy

Temporary results from demos, smoke tests, fit tests, and backend checks can be
removed when they are not being promoted to canonical evidence. Keep cleanup
commands explicit and preview paths before deletion.

Repo-local generated result directories are legacy/local-development storage
unless deliberately configured. Do not commit generated results, logs, audio,
checkpoints, caches, or large artifacts.
