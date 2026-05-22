#!/usr/bin/env bash
set -euo pipefail

# Request a short interactive GPU allocation for metric smoke tests.
# Rorqual/TamIA compute nodes do not have internet, so prepare/download weights
# on a login node first, then run evaluate.py inside the allocation.

ACCOUNT="${ACCOUNT:-rrg-ravanelm}"
TIME="${TIME:-01:00:00}"
CPUS="${CPUS:-4}"
MEM="${MEM:-48G}"
GPUS="${GPUS:-h100_3g.40gb:1}"

salloc -A "$ACCOUNT" -t "$TIME" -c "$CPUS" --mem="$MEM" --gpus="$GPUS"
