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

from evolve.ids import content_hash


def load_epoch_summaries(run_dir: Path) -> List[Dict[str, Any]]:
    summaries = []
    root = Path(run_dir)
    for path in sorted(root.glob("step*/step*.summary.json")):
        try:
            summary = json.loads(path.read_text(encoding="utf-8"))
            checkpoint_path = root / "checkpoints" / summary["checkpoint"]
            checkpoint = json.loads(
                checkpoint_path.read_text(encoding="utf-8")
            )
            if content_hash(checkpoint) != summary["checkpoint_hash"]:
                continue
            summaries.append(summary)
        except (OSError, KeyError, TypeError, ValueError, json.JSONDecodeError):
            continue
    summaries.sort(key=lambda item: item.get("epoch", 0))
    return summaries


def load_latest_checkpoint(run_dir: Path) -> Optional[Dict[str, Any]]:
    """Load only a checkpoint advertised by a completed barrier marker."""

    root = Path(run_dir)
    marker_paths = list(root.glob("step*/step*.summary.json"))
    bootstrap = root / "bootstrap.summary.json"
    if bootstrap.is_file():
        marker_paths.append(bootstrap)
    candidates = []
    for marker_path in marker_paths:
        try:
            marker = json.loads(marker_path.read_text(encoding="utf-8"))
            epoch = int(marker.get("committed_epoch", marker.get("epoch", -1)))
            checkpoint_path = root / "checkpoints" / marker["checkpoint"]
            checkpoint = json.loads(checkpoint_path.read_text(encoding="utf-8"))
            if content_hash(checkpoint) != marker["checkpoint_hash"]:
                continue
        except (OSError, KeyError, TypeError, ValueError, json.JSONDecodeError):
            continue
        candidates.append((epoch, checkpoint))
    return max(candidates, key=lambda item: item[0])[1] if candidates else None


def load_allocation_plans(run_dir: Path) -> List[Dict[str, Any]]:
    """Load plans only for epochs with durable completion summaries."""

    root = Path(run_dir)
    plans = []
    for summary in load_epoch_summaries(root):
        try:
            epoch = int(summary["epoch"])
            path = root / f"step{epoch:02d}" / "allocation_plan.json"
            plan = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, KeyError, TypeError, ValueError, json.JSONDecodeError):
            continue
        if int(plan.get("epoch", -1)) == epoch:
            plans.append(plan)
    return plans


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
    "load_allocation_plans",
    "load_latest_checkpoint",
    "load_status",
]
