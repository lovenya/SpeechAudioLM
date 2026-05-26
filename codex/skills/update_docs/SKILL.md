---
name: update_docs
description: Use when updating repository guidance and handoff documentation, especially for the SpeechAudio LM repo. Helps keep AGENTS.md, CONTEXT.md, HPC_BEST_PRACTICES.md, PROGRESS_LOG.md, CHANGELOG.md, README.md, and project briefs synchronized, concise, and accurate.
---

# Update Docs

Use this skill when maintaining repository documentation. It is reusable across projects, with SpeechAudio LM-specific facts included below.

## Core Workflow

1. Inspect the current docs before editing.
2. Check `git status --short` before and after changes.
3. Keep updates concise and specific; avoid duplicating the same details everywhere.
4. Ask before making or documenting structural, dataset, split, or experiment decisions that are not already clear.
5. Respect the repo's git visibility boundary: update ignored/internal docs locally when useful, but do not stage or push them unless the user explicitly asks.
6. After edits, summarize which files changed and why, and distinguish local-only docs from files intended for git.

## File Roles

- `AGENTS.md`: durable agent instructions, safety rules, repo working style, git rules, and when to ask the user.
- `CONTEXT.md`: active handoff state, immediate todo, current assumptions, known paths, and open decisions.
- `HPC_BEST_PRACTICES.md`: reusable cluster workflow, login vs compute node rules, modules, environments, no-internet installs, SLURM hygiene, and failure handling.
- `PROGRESS_LOG.md`: dated chronological progress, decisions, roadblocks, useful commands, and WHY/HOW notes.
- `CHANGELOG.md`: concise repo-level changes to docs, scripts, configs, workflows, or user-visible behavior.
- `README.md`: public-facing overview; keep it stable and avoid internal diary detail. This is the default Markdown doc intended for GitHub.
- Project briefs, such as `docs/project_brief.MD`: source-of-truth research or project background. Read before major doc updates and edit only when the project definition changes. In SpeechAudio LM, these are local/internal unless the user explicitly says otherwise.

## SpeechAudio LM Facts

- Project: SpeechAudio LM, bidirectional generation between environmental audio and speech.
- Current implementation path: copy/import the clean JSONL manifest into the repo, generate dataset metadata, fix audio/speech paths in the copied manifest, create deterministic `80/10/10` train/validation/test splits, then proceed to AudioCraft/AudioGen setup and baseline inference/evaluation.
- Dataset root outside git: `/project/rrg-csubakan/lovenya/datasets/ttas/extracted`.
- Current best clean manifest: `/project/rrg-csubakan/lovenya/datasets/ttas/extracted/ttas/22k_samples_clean_manual_meta_clean.jsonl`.
- A copied clean manifest may be versioned in the working repo if it is small and defines the dataset state.
- Use a fixed global seed for splits and record it in config/docs.

## Safety Rules To Preserve

- Never add audio files, speech files, zips, checkpoints, generated samples, model outputs, wandb runs, secrets, or other large artifacts to git.
- Show relevant `git status` and `git diff` before suggesting commits.
- Keep commit messages professional, concise, and informative. For this repo, prefer bracketed prefixes such as `[add]:`, `[fix]:`, `[feat]:`, `[docs]:`, or `[chore]:` when committing.
- Do not run heavy compute, training, large preprocessing, aggressive multithreading, or large batch inference on login nodes.
- Prefer `module load` where available and `venv`/`virtualenv`; do not assume conda, anaconda, or uv.
- Rorqual/TamIA compute nodes do not have internet. FIR compute nodes have internet.
- If an HPC/toolchain fix fails twice, stop, summarize attempts and errors, then ask before continuing.

## Update Style

- Put durable policy in `AGENTS.md`, current status in `CONTEXT.md`, dated history in `PROGRESS_LOG.md`, and public project description in `README.md`.
- In SpeechAudio LM, update `AGENTS.md`, `CONTEXT.md`, `PROGRESS_LOG.md`, `CHANGELOG.md`, and project briefs locally as needed even though they are ignored; only stage/push `README.md` among Markdown files unless told otherwise.
- Record reasoning for important choices, especially dataset, split, baseline, and HPC decisions.
- Prefer dated bullets in progress logs over long narratives.
- Do not turn every doc into a full project brief; each file should contain only what its role requires.
