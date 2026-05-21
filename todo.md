# TODO

- For cascaded baseline runs, write component-specific manifests as well as the combined cascade manifest.
  Example: an S2A run should preserve/eject an S2T component manifest for transcript evaluation, a T2A component manifest for generated-audio evaluation when applicable, and a combined S2A manifest for final cascade output tracking.
- Evaluate intermediate components by running `evaluate.py` on their component task manifests directly, such as `--task S2T` for WER or `--task A2T` for caption metrics. Do not use `--stage` on a combined cascade manifest for intermediate metrics.
