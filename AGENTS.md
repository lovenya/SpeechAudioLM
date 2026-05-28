# Repository Guidelines

## Agent Operating Rules

- Work carefully and discuss assumptions before making structural, dataset, split, or experiment decisions.
- Keep explanations concise, but include the reason behind important choices.
- Keep the user in the loop during setup/debugging. Before meaningful environment, dependency, cache, model-weight, or filesystem decisions, explain what will be done, why it is needed, and how it affects the project.
- When preparing commands for independent setup work, split parallelizable steps across terminals where safe. For example, environment creation/package installation and model/checkpoint downloads can often run independently on login or internet-capable nodes; verification commands should wait until the required env and assets both exist.
- For permission prompts, include enough context for informed approval: the target path or command, why escalation/network/write access is needed, and what will or will not be stored in git.
- Trivial package installs or read-only checks can stay brief, but nontrivial dependency/version choices should be discussed before trying alternatives.
- Do not write implementation code unless the user explicitly asks for it.
- If an HPC/toolchain fix fails twice, stop, summarize what was tried, and ask before continuing.

## Project Context

SpeechAudio LM studies bidirectional generation between environmental audio and speech. Immediate work is dataset cleaning, fixed train/val/test split creation, AudioCraft/AudioGen setup, and baseline inference/evaluation.

Dataset root is outside git:

```text
/project/rrg-csubakan/lovenya/datasets/ttas/extracted
```

Current best clean manifest:

```text
/project/rrg-csubakan/lovenya/datasets/ttas/extracted/ttas/22k_samples_clean_manual_meta_clean.jsonl
```

## Git And File Safety

- Never add audio files, zips, checkpoints, outputs, wandb runs, secrets, or large generated files to git.
- Keep audio, checkpoints, outputs, and large generated artifacts out of git.
- Migration and handoff Markdown files may be tracked when explicitly requested for cross-cluster transfer. `README.md` remains the public-facing overview.
- Show `git status` and relevant `git diff` before suggesting commits.
- Keep commit messages professional, concise, and informative.
- Treat datasets, splits, and experiment outputs as decisions that need confirmation unless already documented.

## HPC Constraints

- Use login nodes only for editing, git, lightweight validation, small metadata scripts, plotting, and job submission.
- Use interactive jobs or `sbatch` for heavy preprocessing, training, large inference, and GPU work.
- Prefer `module load` when available, such as `StdEnv/2023`, `gcc`, `cuda`, `arrow`, and Python or Node modules as appropriate.
- Use `venv` or `virtualenv`; do not assume conda, anaconda, or uv.
- Rorqual/TamIA compute nodes do not have internet. FIR compute nodes have internet.
