"""Durable same-directory atomic and immutable file writes."""

from __future__ import annotations

import json
import os
import tempfile
from pathlib import Path
from typing import Any, Union


PathLike = Union[str, os.PathLike]


class ImmutableWriteError(FileExistsError):
    """Raised when an immutable artifact already exists."""


def fsync_directory(path: PathLike) -> None:
    """Persist directory entry changes before returning."""
    directory = Path(path)
    flags = os.O_RDONLY | getattr(os, "O_DIRECTORY", 0)
    fd = os.open(os.fspath(directory), flags)
    try:
        os.fsync(fd)
    finally:
        os.close(fd)


def _json_bytes(value: Any) -> bytes:
    return (
        json.dumps(
            value,
            ensure_ascii=False,
            allow_nan=False,
            sort_keys=True,
            indent=2,
        )
        + "\n"
    ).encode("utf-8")


def _durable_temp(target: Path, data: bytes, mode: int) -> Path:
    target.parent.mkdir(parents=True, exist_ok=True)
    fd, temporary_name = tempfile.mkstemp(
        dir=os.fspath(target.parent),
        prefix=f".{target.name}.",
        suffix=".tmp",
    )
    temporary = Path(temporary_name)
    try:
        os.fchmod(fd, mode)
        with os.fdopen(fd, "wb") as handle:
            fd = -1
            handle.write(data)
            handle.flush()
            os.fsync(handle.fileno())
    except BaseException:
        if fd >= 0:
            os.close(fd)
        try:
            temporary.unlink()
        except FileNotFoundError:
            pass
        raise
    return temporary


def atomic_write_bytes(path: PathLike, data: bytes, mode: int = 0o644) -> Path:
    """Atomically replace ``path`` after flushing a temp in its directory."""
    if not isinstance(data, bytes):
        raise TypeError("atomic_write_bytes data must be bytes")
    target = Path(path)
    temporary = _durable_temp(target, data, mode)
    try:
        os.replace(os.fspath(temporary), os.fspath(target))
        fsync_directory(target.parent)
    except BaseException:
        try:
            temporary.unlink()
        except FileNotFoundError:
            pass
        raise
    return target


def atomic_write_text(
    path: PathLike,
    text: str,
    *,
    encoding: str = "utf-8",
    mode: int = 0o644,
) -> Path:
    if not isinstance(text, str):
        raise TypeError("atomic_write_text text must be str")
    return atomic_write_bytes(path, text.encode(encoding), mode=mode)


def atomic_write_json(path: PathLike, value: Any, mode: int = 0o644) -> Path:
    return atomic_write_bytes(path, _json_bytes(value), mode=mode)


def write_immutable_bytes(path: PathLike, data: bytes, mode: int = 0o644) -> Path:
    """Publish ``path`` exactly once without an overwrite race.

    A fully flushed same-directory temporary file is hard-linked into place.
    ``link`` is atomic and refuses an existing destination, including a symlink.
    """
    if not isinstance(data, bytes):
        raise TypeError("write_immutable_bytes data must be bytes")
    target = Path(path)
    temporary = _durable_temp(target, data, mode)
    try:
        try:
            os.link(os.fspath(temporary), os.fspath(target))
        except FileExistsError as exc:
            raise ImmutableWriteError(f"immutable artifact already exists: {target}") from exc
        temporary.unlink()
        fsync_directory(target.parent)
    except BaseException:
        try:
            temporary.unlink()
        except FileNotFoundError:
            pass
        raise
    return target


def write_immutable_text(
    path: PathLike,
    text: str,
    *,
    encoding: str = "utf-8",
    mode: int = 0o644,
) -> Path:
    if not isinstance(text, str):
        raise TypeError("write_immutable_text text must be str")
    return write_immutable_bytes(path, text.encode(encoding), mode=mode)


def write_immutable_json(path: PathLike, value: Any, mode: int = 0o644) -> Path:
    return write_immutable_bytes(path, _json_bytes(value), mode=mode)

