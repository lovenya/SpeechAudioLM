from __future__ import annotations

import json
import os
import platform
import threading
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Iterable

from baselines.dataset import PromptRow
from baselines.models.base import BaselineRunner


def run_inference(
    runner: BaselineRunner,
    rows: Iterable[PromptRow],
    output_root: Path,
    log_root: Path,
    overwrite: bool,
    run_config: dict,
    total_rows: int,
    resource_interval: float,
    continue_on_error: bool = False,
) -> None:
    output_root.mkdir(parents=True, exist_ok=True)
    output_dir = output_root / runner.output_subdir
    output_dir.mkdir(parents=True, exist_ok=True)
    manifest_path = output_root / "manifest.jsonl"
    logger = RunLogger(log_root)
    resource_monitor = ResourceMonitor(
        log_path=logger.resource_path,
        interval_seconds=resource_interval,
        metadata={
            "pid": os.getpid(),
            "host": platform.node(),
            "output_root": str(output_root),
            **run_config,
        },
    )

    logger.log(f"output_root={output_root}")
    logger.log(f"output_dir={output_dir}")
    logger.log(f"manifest={manifest_path}")
    logger.log(f"log_file={logger.log_path}")
    logger.log(f"resource_log={logger.resource_path}")
    logger.log(f"total_rows={total_rows}")
    logger.log("loading model...")
    load_start = time.time()
    processed = 0
    skipped = 0
    failed = 0
    run_start = time.time()
    try:
        resource_monitor.start()
        runner.load()
        logger.log(f"model_loaded_seconds={time.time() - load_start:.2f}")

        with manifest_path.open("a", encoding="utf-8") as manifest:
            for ordinal, row in enumerate(rows, start=1):
                output_path = output_dir / f"{row.row_index:06d}_{row.sample_id}{runner.extension}"
                if output_path.exists() and not overwrite:
                    skipped += 1
                    logger.log(_progress_line("skip", ordinal, total_rows, run_start, row, output_path))
                    continue

                logger.log(_progress_line("generate", ordinal, total_rows, run_start, row, output_path))
                start = time.time()
                try:
                    result = runner.generate(row, output_path)
                except Exception as exc:
                    if not continue_on_error:
                        raise
                    failed += 1
                    elapsed = time.time() - start
                    record = {
                        **run_config,
                        "status": "error",
                        "row_index": row.row_index,
                        "sample_id": row.sample_id,
                        "prompt": row.prompt,
                        "output_file": str(output_path),
                        "generation_seconds": round(elapsed, 3),
                        "source": row.record,
                        "error": {
                            "type": type(exc).__name__,
                            "message": str(exc),
                        },
                    }
                    manifest.write(json.dumps(record, ensure_ascii=False) + "\n")
                    manifest.flush()
                    logger.log(
                        f"error row={row.row_index} sample_id={row.sample_id} "
                        f"type={type(exc).__name__} message={str(exc)!r} seconds={elapsed:.2f}"
                    )
                    continue
                elapsed = time.time() - start
                processed += 1
                record = {
                    **run_config,
                    "status": "success",
                    "row_index": row.row_index,
                    "sample_id": row.sample_id,
                    "prompt": row.prompt,
                    "output_file": str(result.output_path),
                    "sample_rate": result.sample_rate,
                    "generation_seconds": round(elapsed, 3),
                    "source": row.record,
                    "model_extra": result.extra,
                }
                manifest.write(json.dumps(record, ensure_ascii=False) + "\n")
                manifest.flush()
                logger.log(f"wrote {result.output_path} seconds={elapsed:.2f}")
    finally:
        resource_monitor.stop()
        logger.log(
            f"done processed={processed} skipped={skipped} failed={failed} "
            f"total_seen={processed + skipped + failed} "
            f"elapsed={_format_seconds(time.time() - run_start)}"
        )
        logger.close()


def dry_run_inference(
    rows: Iterable[PromptRow],
    output_root: Path,
    output_subdir: str,
    extension: str,
    run_config: dict,
    total_rows: int,
) -> None:
    output_dir = output_root / output_subdir
    print(f"dry_run=true")
    print(f"output_root={output_root}")
    print(f"output_dir={output_dir}")
    print(f"total_rows={total_rows}")
    print(f"run_config={json.dumps(run_config, ensure_ascii=False, sort_keys=True)}")
    count = 0
    for row in rows:
        output_path = output_dir / f"{row.row_index:06d}_{row.sample_id}{extension}"
        print(
            json.dumps(
                {
                    "row_index": row.row_index,
                    "sample_id": row.sample_id,
                    "prompt": row.prompt,
                    "output_file": str(output_path),
                },
                ensure_ascii=False,
            )
        )
        count += 1
    print(f"dry_run_rows={count}")


class RunLogger:
    def __init__(self, log_root: Path) -> None:
        timestamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
        log_root.mkdir(parents=True, exist_ok=True)
        self.log_path = log_root / f"run_{timestamp}.log"
        self.resource_path = log_root / f"resources_{timestamp}.jsonl"
        self._file = self.log_path.open("a", encoding="utf-8")

    def log(self, message: str) -> None:
        line = f"{_utc_now()} {message}"
        print(line)
        self._file.write(line + "\n")
        self._file.flush()

    def close(self) -> None:
        self._file.close()


class ResourceMonitor:
    def __init__(self, log_path: Path, interval_seconds: float, metadata: dict) -> None:
        self.log_path = log_path
        self.interval_seconds = interval_seconds
        self.metadata = metadata
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None
        self._psutil = None
        self._process = None

    def start(self) -> None:
        if self.interval_seconds <= 0:
            return
        self.log_path.parent.mkdir(parents=True, exist_ok=True)
        try:
            import psutil  # type: ignore

            self._psutil = psutil
            self._process = psutil.Process(os.getpid())
            self._process.cpu_percent(interval=None)
            psutil.cpu_percent(interval=None)
        except Exception:
            self._psutil = None
            self._process = None
        self._write({"event": "start", "metadata": self.metadata})
        self._thread = threading.Thread(target=self._loop, daemon=True)
        self._thread.start()

    def stop(self) -> None:
        if self.interval_seconds <= 0:
            return
        self._stop.set()
        if self._thread is not None:
            self._thread.join(timeout=max(1.0, self.interval_seconds + 1.0))
        self._write({"event": "stop"})

    def _loop(self) -> None:
        while not self._stop.wait(self.interval_seconds):
            self._write({"event": "sample", **self._sample()})

    def _sample(self) -> dict:
        sample: dict = {}
        if self._psutil is not None and self._process is not None:
            try:
                memory = self._process.memory_info()
                sample.update(
                    {
                        "process_cpu_percent": self._process.cpu_percent(interval=None),
                        "system_cpu_percent": self._psutil.cpu_percent(interval=None),
                        "process_rss_gb": round(memory.rss / 1024**3, 3),
                        "process_threads": self._process.num_threads(),
                    }
                )
            except Exception as exc:
                sample["psutil_error"] = str(exc)
        try:
            import torch

            if torch.cuda.is_available():
                sample.update(
                    {
                        "cuda_allocated_gb": round(torch.cuda.memory_allocated() / 1024**3, 3),
                        "cuda_reserved_gb": round(torch.cuda.memory_reserved() / 1024**3, 3),
                        "cuda_device": torch.cuda.get_device_name(0),
                    }
                )
        except Exception as exc:
            sample["cuda_error"] = str(exc)
        return sample

    def _write(self, record: dict) -> None:
        record = {"time_utc": _utc_now(), **record}
        with self.log_path.open("a", encoding="utf-8") as f:
            f.write(json.dumps(record, ensure_ascii=False, sort_keys=True) + "\n")


def _progress_line(
    action: str,
    ordinal: int,
    total: int,
    run_start: float,
    row: PromptRow,
    output_path: Path,
) -> str:
    elapsed = time.time() - run_start
    eta = "unknown"
    if ordinal > 1 and total > 0:
        remaining = max(total - ordinal + 1, 0)
        eta = _format_seconds((elapsed / (ordinal - 1)) * remaining)
    return (
        f"{action} [{ordinal}/{total}] row={row.row_index} sample_id={row.sample_id} "
        f"elapsed={_format_seconds(elapsed)} eta={eta} output={output_path}"
    )


def _format_seconds(seconds: float) -> str:
    seconds = max(0, int(round(seconds)))
    hours, remainder = divmod(seconds, 3600)
    minutes, secs = divmod(remainder, 60)
    if hours:
        return f"{hours:d}:{minutes:02d}:{secs:02d}"
    return f"{minutes:d}:{secs:02d}"


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")
