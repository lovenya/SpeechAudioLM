from __future__ import annotations

import errno
import json
import os
import uuid
from collections.abc import Iterable, Mapping
from pathlib import Path
from typing import Any


class ArtifactAlreadyExistsError(FileExistsError):
    """Raised when an immutable artifact destination already exists."""


def write_bytes_create_only(path: Path, content: bytes) -> None:
    """Atomically create an immutable artifact containing bytes."""
    if not isinstance(content, bytes):
        raise TypeError("content must be bytes")
    _publish_bytes(Path(path), content)


def write_text_create_only(path: Path, content: str) -> None:
    """Atomically create an immutable UTF-8 text artifact."""
    if not isinstance(content, str):
        raise TypeError("content must be str")
    _publish_bytes(Path(path), content.encode("utf-8"))


def write_json_create_only(path: Path, value: Any) -> None:
    """Atomically create schema-ordered JSON, rejecting non-finite numbers."""
    content = json.dumps(
        value,
        allow_nan=False,
        ensure_ascii=False,
        indent=2,
        sort_keys=False,
    )
    write_text_create_only(path, content + "\n")


def write_jsonl_create_only(path: Path, rows: Iterable[Mapping[str, Any]]) -> None:
    """Atomically create schema-ordered JSON Lines from mapping rows."""
    lines: list[str] = []
    for index, row in enumerate(rows):
        if not isinstance(row, Mapping):
            raise TypeError(f"JSONL row {index} must be a mapping")
        lines.append(
            json.dumps(
                row,
                allow_nan=False,
                ensure_ascii=False,
                separators=(",", ":"),
                sort_keys=False,
            )
        )
    write_text_create_only(path, "".join(f"{line}\n" for line in lines))


def _publish_bytes(path: Path, content: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary_path = path.parent / f".{path.name}.{uuid.uuid4().hex}.tmp"
    fd: int | None = None

    try:
        fd = os.open(temporary_path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o666)
        with os.fdopen(fd, "wb") as stream:
            fd = None
            stream.write(content)
            stream.flush()
            os.fsync(stream.fileno())

        try:
            os.link(temporary_path, path)
        except OSError as exc:
            if exc.errno == errno.EEXIST:
                raise ArtifactAlreadyExistsError(
                    f"immutable artifact already exists: {path}"
                ) from exc
            raise
    finally:
        if fd is not None:
            os.close(fd)
        try:
            temporary_path.unlink()
        except FileNotFoundError:
            pass
