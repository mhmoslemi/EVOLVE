"""Exclusive EVOLVE run-directory creation and read-only attachment."""

from __future__ import annotations

import fcntl
import os
import re
import secrets
from contextlib import contextmanager
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Iterator, Optional, Union

from .atomic import fsync_directory


RUN_SUBDIRECTORIES = (
    "checkpoints",
    "roles/scout",
    "roles/mechanist",
    "roles/challenger",
    "archive/snapshots",
    "causal_memory/snapshots",
    "best",
    "plots",
    "logs/workers",
    "logs/verifiers",
)


class RunLayoutError(RuntimeError):
    """Base run layout failure."""


class RunCollisionError(RunLayoutError):
    """A fresh run name already exists and must not be attached to."""


class RunAttachmentError(RunLayoutError):
    """An existing run was opened without explicit resume authority."""


def _slug(value: str) -> str:
    slug = re.sub(r"[^A-Za-z0-9._-]+", "_", str(value)).strip("_.-")
    if not slug:
        raise ValueError("run name component cannot be empty")
    return slug


def _model_short(model_name: str) -> str:
    return _slug(str(model_name).rstrip("/").split("/")[-1])


@contextmanager
def _exclusive_creation_lock(root: Path) -> Iterator[None]:
    """Serialize name allocation by locking the runs directory inode."""
    flags = os.O_RDONLY | getattr(os, "O_DIRECTORY", 0)
    fd = os.open(os.fspath(root), flags)
    try:
        fcntl.flock(fd, fcntl.LOCK_EX)
        yield
    finally:
        fcntl.flock(fd, fcntl.LOCK_UN)
        os.close(fd)


@dataclass(frozen=True)
class RunLayout:
    run_dir: Path

    def path(self, relative: str) -> Path:
        if Path(relative).is_absolute() or ".." in Path(relative).parts:
            raise ValueError("run-relative paths must stay inside the run")
        return self.run_dir / relative


def create_fresh_run_layout(
    root: Union[str, os.PathLike],
    *,
    problem: str,
    model_name: str,
    now: Optional[datetime] = None,
    short_random_id: Optional[str] = None,
) -> RunLayout:
    """Create a new, exclusively named EVOLVE run and its fixed directory tree."""
    runs_root = Path(root)
    runs_root.mkdir(parents=True, exist_ok=True)
    timestamp = (now or datetime.now()).strftime("%m%d-%H%M%S")
    random_id = short_random_id or secrets.token_hex(3)
    if not re.fullmatch(r"[A-Za-z0-9]{4,16}", random_id):
        raise ValueError("short_random_id must be 4-16 ASCII letters/digits")
    name = "_".join(
        (_slug(problem), _model_short(model_name), timestamp, random_id)
    )
    run_dir = runs_root / name

    with _exclusive_creation_lock(runs_root):
        try:
            run_dir.mkdir(mode=0o755, exist_ok=False)
        except FileExistsError as exc:
            raise RunCollisionError(
                f"fresh run already exists; use explicit resume instead: {run_dir}"
            ) from exc
        fsync_directory(runs_root)
        created = []
        for relative in RUN_SUBDIRECTORIES:
            directory = run_dir / relative
            directory.mkdir(parents=True, exist_ok=True)
            created.append(directory)
        for directory in reversed(created):
            fsync_directory(directory)
        fsync_directory(run_dir)

    return RunLayout(run_dir=run_dir.resolve())


def open_existing_run_layout(
    run_dir: Union[str, os.PathLike],
    *,
    resume: bool = False,
) -> RunLayout:
    """Attach read-only to an existing directory only under explicit resume."""
    path = Path(run_dir).expanduser()
    if not path.is_dir():
        raise RunAttachmentError(f"run directory does not exist: {path}")
    if not resume:
        detail = "nonempty " if any(path.iterdir()) else ""
        raise RunAttachmentError(
            f"refusing to attach to {detail}existing run without explicit resume: {path}"
        )
    return RunLayout(run_dir=path.resolve())

