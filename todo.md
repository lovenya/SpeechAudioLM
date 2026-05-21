# TODO

- For cascaded baseline runs, write component-specific manifests as well as the combined cascade manifest.
  Example: an S2A run should preserve/eject an S2T component manifest for transcript evaluation, a T2A component manifest for generated-audio evaluation when applicable, and a combined S2A manifest for final cascade output tracking.
- Evaluate intermediate components by running `evaluate.py` on their component task manifests directly, such as `--task S2T` for WER or `--task A2T` for caption metrics. Do not use `--stage` on a combined cascade manifest for intermediate metrics.
- Verify `aac-metrics` CLAPSim smoke for `aac_clapsim`.
- Verify `stable-audio-metrics` LAION-CLAP checkpoint name for `clap_score_t2a`; prefer `630k-audioset-fusion-best` if supported.
- Verify `stable-audio-metrics` PaSST setup for `kl_passt`.
- Verify UTMOSv2 setup from `sarulab-speech/UTMOSv2`.
- Decide DNSMOS backend later; do not implement it by default yet.
- Decide T2T, A2S, and non-final S2A metrics later.
- Add evaluator resumability and an explicit `--overwrite` flag before long full-corpus metric runs.
- Recreate/migrate misplaced metric envs when convenient:
  `aac-metrics` currently exists at `.venvs/.venv-eval-aac`; Whisper also remains outside `.venvs/metrics/` for now.
  Do not move venv directories in place; recreate them under `.venvs/metrics/`.
