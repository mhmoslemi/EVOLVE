"""Shared, read-only helpers for headless EVOLVE plots.

Every plot in :mod:`evolve.viz` reads only already-committed run artifacts
(``stepNN.summary.json``, ``checkpoints/*.json``) -- it never reruns
candidate code and never imports a model or CUDA library, so it works both
after a run and while one is still active.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, List, Optional


def load_epoch_summaries(run_dir: Path) -> List[Dict[str, Any]]:
    summaries = []
    for path in sorted(Path(run_dir).glob("step*.summary.json")):
        try:
            summaries.append(json.loads(path.read_text(encoding="utf-8")))
        except (OSError, json.JSONDecodeError):
            continue
    summaries.sort(key=lambda item: item.get("epoch", 0))
    return summaries


def load_latest_checkpoint(run_dir: Path) -> Optional[Dict[str, Any]]:
    path = Path(run_dir) / "checkpoints" / "latest.json"
    if not path.is_file():
        return None
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return None


def load_status(run_dir: Path) -> Optional[Dict[str, Any]]:
    path = Path(run_dir) / "status.json"
    if not path.is_file():
        return None
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return None


def empty_figure_message(ax, message: str) -> None:
    ax.text(0.5, 0.5, message, ha="center", va="center", transform=ax.transAxes)
    ax.set_xticks([])
    ax.set_yticks([])


__all__ = [
    "empty_figure_message",
    "load_epoch_summaries",
    "load_latest_checkpoint",
    "load_status",
]
