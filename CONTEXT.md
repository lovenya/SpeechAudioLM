# Project Context

Status: current handoff context for future agent sessions. Keep this file short,
current, and action-oriented. Durable design belongs in the design docs listed
below; dated history belongs in `PROGRESS_LOG.md`.

## Current Trajectory

The project is being debloated step by step. The immediate work is to make the
repository easier for agents and humans to restart from by documenting:

- where inference results and eval results live;
- how cascaded stage outputs feed later stages;
- which docs are authoritative for results, evals, resources, and cluster rules;
- which old demo, smoke, and repo-local outputs should not guide new work.

Do not add new model/eval implementation complexity until the relevant result
and eval flow is clear in docs.

## Read These First

Start with:

1. `docs/agent_doc_map.md` - routing map for current docs.
2. `RESULTS_STRUCTURE_DESIGN.md` - results roots, temporary policy, inference
   run layout, and cascaded output contracts.
3. `EVALS_STRUCTURE_DESIGN.md` - eval result layout and eval simplification
   decisions.
4. `HPC_BEST_PRACTICES.md` - cluster usage, Slurm hygiene, and resource
   monitoring expectations.
5. `PROGRESS_LOG.md` - dated history when a past decision or failure matters.

Treat this `CONTEXT.md` as session state, not the source of truth for schemas.

## Active Roots

Current repo:

```text
/scratch/l/lovenya/SpeechAudioLM
```

Current machine-specific runtime root:

```text
/scratch/l/lovenya/salm_migration_clean
```

`configs/project_paths.json` is the named path registry. As of the current
results-doc refactor, real runs should use these external roots:

```text
results_root:            /scratch/l/lovenya/salm_migration_clean/results
temporary_results_root:  /scratch/l/lovenya/salm_migration_clean/temporary/results
logs_root:               /scratch/l/lovenya/salm_migration_clean/logs
temporary_logs_root:     /scratch/l/lovenya/salm_migration_clean/temporary/logs
exports_root:            /scratch/l/lovenya/salm_migration_clean/exports
```

Keep datasets, generated audio, checkpoints, result manifests, Slurm logs,
resource logs, caches, and exports outside Git.

## Dataset State

TTAS remains the active dataset.

```text
dataset root:      /scratch/l/lovenya/salm_migration_clean/datasets/ttas/extracted
source manifest:   /scratch/l/lovenya/salm_migration_clean/datasets/ttas/extracted/ttas/22k_samples_clean_manual_meta_clean.jsonl
repo manifest:     data/manifests/ttas_clean_normalized.jsonl
train split:       data/splits/ttas_train.jsonl
val split:         data/splits/ttas_val.jsonl
test split:        data/splits/ttas_test.jsonl
```

Known split counts from prior docs are train `18,226`, val `2,277`, test
`2,277`, total `22,780`. The fixed split uses seed `1337`, an `80/10/10`
ratio, and gender x bucketed-emotion stratification.

The final training manifests are G-Eval audio v6 clean-v2, filtered independently
by direction while preserving original split membership and row order:

```text
A2S: data/manifests/clean_v2/a2s/ttas_{train,val,test}.jsonl
     counts: 17,596 / 2,206 / 2,197
S2A: data/manifests/clean_v2/s2a/ttas_{train,val,test}.jsonl
     counts: 15,033 / 1,887 / 1,878
```

Exclude a direction when its v6 result has a null or `<= 3` final score,
scenario plausibility `<= 2`, or target-modality success `= 1`. See
`docs/Geval_prompts/ttas_geval_audio_v6_dataset_audit.md` for provenance and
audit artifacts. Do not change this policy or the source splits without an
explicit user decision.

## Results And Eval Direction

Current design direction:

- canonical inference results live under `{results_root}`;
- canonical eval results live under `{results_root}/evals`;
- logs mirror results under `{logs_root}`;
- demos, smoke tests, backend checks, fit tests, and partial/debug runs belong
  under temporary roots;
- old repo-local generated directories are legacy/local-development storage;
- `results/evals/T2S/` is a legacy eval layout and should not be extended;
- previous contents of `temporary/` were demo/smoke outputs and can be deleted.

Eval design principles currently agreed:

- metric choice should be explicit;
- metric IDs appear directly in eval paths;
- each metric gets its own eval run directory;
- canonical eval design removes eval sharding, shard merge scripts, and
  parallel-eval flexibility in the metric runner;
- temporary small-subset backend checks are allowed, but they are not canonical
  eval results.

## Resource Planning

Before asking for new Slurm resources, inspect measured evidence:

```text
docs/tamia_resource_usage.json
/scratch/l/lovenya/salm_migration_clean/logs/resources/*_summary.json
/scratch/l/lovenya/salm_migration_clean/logs/resources/*.jsonl
configs/slurm_resource_profiles.json
```

For TamIA, remember that H100 GPU jobs reserve a whole 4-GPU node. Packing
compatible independent inference work can be useful, but eval design should not
reintroduce sharding or implicit parallel metric behavior just to fill a node.

## Current Git/Workspace Notes

A recent commit recorded the results-root and doc-map work:

```text
f8297e3 [docs]: document results roots and agent doc map
```

The worktree still has many unrelated dirty/untracked files from previous work.
Do not stage or revert them unless the user explicitly asks. In particular,
`configs/project_paths.json` may still contain unstaged changes unrelated to the
root-path hunk that was committed.

## What To Avoid

- Do not treat old Slurm job IDs in stale notes as current truth.
- Do not extend old repo-local result/eval layouts.
- Do not commit generated results, logs, audio, checkpoints, caches, Slurm
  outputs, or large artifacts.
- Do not make structural dataset, split, or experiment-design decisions without
  discussing them first.
- Do not over-engineer edge cases while the current objective is documentation
  cleanup and flow clarity.

## Next Useful Docs Work

Likely next steps, subject to user confirmation:

1. Tighten `RESULTS_STRUCTURE_DESIGN.md` around the exact stage-to-stage
   manifest contract for cascaded inference.
2. Tighten `EVALS_STRUCTURE_DESIGN.md` so it references external roots and the
   current no-sharding/no-parallel-eval principle clearly.
3. Add a concise inference-flow doc if the results doc becomes too long.
4. Prune or archive stale operational notes from old docs only after checking
   whether they contain unique decisions not captured elsewhere.
