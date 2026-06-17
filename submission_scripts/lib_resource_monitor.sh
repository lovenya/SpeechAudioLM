#!/usr/bin/env bash

start_resource_monitor() {
  local label="$1"
  local interval="${RESOURCE_INTERVAL:-300}"
  if [ "$interval" = "0" ]; then
    return 0
  fi

  local scratch="${SALM_SCRATCH:-/scratch/l/lovenya/salm_migration_clean}"
  local resource_dir="$scratch/logs/resources"
  local job_id="${SLURM_JOB_ID:-interactive}"
  local array_id="${SLURM_ARRAY_TASK_ID:-none}"
  local safe_label="${label//[^A-Za-z0-9_.-]/_}"
  local prefix="${safe_label}_${job_id}_${array_id}"

  mkdir -p "$resource_dir"
  python scripts/resource_monitor.py \
    --log "$resource_dir/${prefix}.jsonl" \
    --summary "$resource_dir/${prefix}_summary.json" \
    --interval "$interval" \
    --label "$label" &
  SALM_RESOURCE_MONITOR_PID=$!
  export SALM_RESOURCE_MONITOR_PID
  echo "resource_monitor_pid=$SALM_RESOURCE_MONITOR_PID"
  echo "resource_log=$resource_dir/${prefix}.jsonl"
  echo "resource_summary=$resource_dir/${prefix}_summary.json"
  trap stop_resource_monitor EXIT
}

stop_resource_monitor() {
  if [ -n "${SALM_RESOURCE_MONITOR_PID:-}" ]; then
    kill "$SALM_RESOURCE_MONITOR_PID" 2>/dev/null || true
    wait "$SALM_RESOURCE_MONITOR_PID" 2>/dev/null || true
  fi
}
