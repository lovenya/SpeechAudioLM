#!/usr/bin/env python
"""Lightweight HPC resource monitor for Slurm jobs."""

from __future__ import annotations

import argparse
import csv
import json
import os
import signal
import socket
import subprocess
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


STOP = False


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--log", required=True, help="JSONL sample output path.")
    parser.add_argument("--summary", required=True, help="JSON summary output path.")
    parser.add_argument("--interval", type=float, default=300.0, help="Seconds between samples.")
    parser.add_argument("--label", default=None, help="Human-readable job label.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.interval <= 0:
        return

    signal.signal(signal.SIGTERM, _handle_stop)
    signal.signal(signal.SIGINT, _handle_stop)
    log_path = Path(args.log)
    summary_path = Path(args.summary)
    log_path.parent.mkdir(parents=True, exist_ok=True)
    summary_path.parent.mkdir(parents=True, exist_ok=True)

    peaks: dict[str, Any] = {
        "cpu_percent": None,
        "mem_used_gb": None,
        "mem_used_percent": None,
        "gpu_memory_used_mb": None,
        "gpu_util_percent": None,
    }
    previous_cpu = _read_proc_stat()
    start_time = time.time()

    with log_path.open("a", encoding="utf-8") as handle:
        _write(handle, {"event": "start", "label": args.label, "hostname": socket.gethostname()})
        while not STOP:
            sample, previous_cpu = _sample(previous_cpu)
            _update_peaks(peaks, sample)
            _write(handle, {"event": "sample", **sample})
            handle.flush()
            if STOP:
                break
            time.sleep(args.interval)

        summary = {
            "event": "summary",
            "label": args.label,
            "hostname": socket.gethostname(),
            "started_utc": _utc(start_time),
            "finished_utc": _utc(time.time()),
            "elapsed_seconds": round(time.time() - start_time, 3),
            "peaks": peaks,
        }
        _write(handle, summary)

    summary_path.write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _handle_stop(_signum: int, _frame: object) -> None:
    global STOP
    STOP = True


def _sample(previous_cpu: tuple[int, int] | None) -> tuple[dict[str, Any], tuple[int, int] | None]:
    now = time.time()
    current_cpu = _read_proc_stat()
    cpu_percent = _cpu_percent(previous_cpu, current_cpu)
    mem = _read_meminfo()
    gpu = _read_gpu()
    sample: dict[str, Any] = {
        "time_utc": _utc(now),
        "loadavg": list(os.getloadavg()) if hasattr(os, "getloadavg") else None,
        "cpu_percent": cpu_percent,
        **mem,
        "gpus": gpu,
    }
    return sample, current_cpu


def _read_proc_stat() -> tuple[int, int] | None:
    try:
        fields = Path("/proc/stat").read_text(encoding="utf-8").splitlines()[0].split()[1:]
        values = [int(value) for value in fields]
    except Exception:
        return None
    idle = values[3] + values[4]
    total = sum(values)
    return idle, total


def _cpu_percent(previous: tuple[int, int] | None, current: tuple[int, int] | None) -> float | None:
    if previous is None or current is None:
        return None
    prev_idle, prev_total = previous
    cur_idle, cur_total = current
    total_delta = cur_total - prev_total
    idle_delta = cur_idle - prev_idle
    if total_delta <= 0:
        return None
    return round(100.0 * (1.0 - idle_delta / total_delta), 3)


def _read_meminfo() -> dict[str, float | None]:
    try:
        values = {}
        for line in Path("/proc/meminfo").read_text(encoding="utf-8").splitlines():
            key, value = line.split(":", 1)
            values[key] = int(value.strip().split()[0])
        total = values["MemTotal"] / 1024**2
        available = values.get("MemAvailable", values.get("MemFree", 0)) / 1024**2
        used = total - available
        return {
            "mem_total_gb": round(total, 3),
            "mem_available_gb": round(available, 3),
            "mem_used_gb": round(used, 3),
            "mem_used_percent": round(100.0 * used / total, 3) if total else None,
        }
    except Exception:
        return {
            "mem_total_gb": None,
            "mem_available_gb": None,
            "mem_used_gb": None,
            "mem_used_percent": None,
        }


def _read_gpu() -> list[dict[str, Any]]:
    command = [
        "nvidia-smi",
        "--query-gpu=index,name,memory.used,memory.total,utilization.gpu",
        "--format=csv,noheader,nounits",
    ]
    try:
        output = subprocess.check_output(command, text=True, stderr=subprocess.STDOUT, timeout=10)
    except Exception as exc:
        return [{"error": str(exc)}]
    rows = []
    for row in csv.reader(output.splitlines()):
        if len(row) < 5:
            continue
        rows.append(
            {
                "index": row[0].strip(),
                "name": row[1].strip(),
                "memory_used_mb": _to_float(row[2]),
                "memory_total_mb": _to_float(row[3]),
                "utilization_gpu_percent": _to_float(row[4]),
            }
        )
    return rows


def _to_float(value: str) -> float | None:
    try:
        return float(value.strip())
    except ValueError:
        return None


def _update_peaks(peaks: dict[str, Any], sample: dict[str, Any]) -> None:
    for key in ("cpu_percent", "mem_used_gb", "mem_used_percent"):
        _peak(peaks, key, sample.get(key))
    for gpu in sample.get("gpus", []):
        if not isinstance(gpu, dict):
            continue
        _peak(peaks, "gpu_memory_used_mb", gpu.get("memory_used_mb"))
        _peak(peaks, "gpu_util_percent", gpu.get("utilization_gpu_percent"))


def _peak(peaks: dict[str, Any], key: str, value: Any) -> None:
    if not isinstance(value, (int, float)):
        return
    if peaks[key] is None or value > peaks[key]:
        peaks[key] = value


def _write(handle: Any, record: dict[str, Any]) -> None:
    if "time_utc" not in record:
        record = {"time_utc": _utc(time.time()), **record}
    handle.write(json.dumps(record, sort_keys=True) + "\n")


def _utc(timestamp: float) -> str:
    return datetime.fromtimestamp(timestamp, timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


if __name__ == "__main__":
    main()
