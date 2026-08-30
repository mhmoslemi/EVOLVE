"""Human-readable console reporting so a live run is legible without raw logs.

Live status may label a provisional observation but must never present it as
the committed answer: everything here reads only ``status.json`` and the
committed ``best/`` artifacts, both written exclusively at barrier commit.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Mapping, Union

from evolve.viz._common import load_status

from .best import BestRecord, load_best


def format_progress(
    stage: str,
    *,
    epoch: int | None = None,
    total_epochs: int | None = None,
    completed: int | None = None,
    total: int | None = None,
    unit: str = "items",
    detail: str | None = None,
    bar_width: int = 24,
) -> str:
    """Render one log-friendly progress line without terminal control codes."""

    parts = ["EVOLVE"]
    if epoch is not None:
        epoch_text = f"epoch {epoch + 1}"
        if total_epochs is not None:
            epoch_text += f"/{total_epochs}"
        parts.append(epoch_text)
    parts.append(stage)
    if completed is not None and total is not None:
        safe_total = max(0, int(total))
        safe_completed = min(max(0, int(completed)), safe_total)
        fraction = safe_completed / safe_total if safe_total else 1.0
        filled = int(fraction * max(1, bar_width))
        bar = "█" * filled + "░" * (max(1, bar_width) - filled)
        parts.append(
            f"[{bar}] {safe_completed}/{safe_total} {unit} "
            f"({fraction:.0%}, {safe_total - safe_completed} left)"
        )
    if detail:
        parts.append(detail)
    return " · ".join(parts)


def format_status(status: Mapping[str, Any]) -> str:
    lines = [
        f"EVOLVE run {status.get('run_id', '?')}",
        f"epoch: {status.get('epoch')}",
        f"archive coverage: {float(status.get('archive_coverage', 0.0)):.1%}",
    ]
    record = status.get("confirmed_record", {}) or {}
    if record.get("state_id"):
        lines.append(
            "confirmed record: "
            f"internal_reward={record.get('internal_reward')} "
            f"raw_score={record.get('raw_score')} cell={record.get('cell_id')}"
        )
    else:
        lines.append("confirmed record: none yet (provisional observations do not count)")
    live_epoch = status.get("live_epoch", {}) or {}
    if live_epoch.get("stage"):
        completed = int(live_epoch.get("completed_branches", 0))
        total = int(live_epoch.get("total_branches", 0))
        lines.append(
            format_progress(
                str(live_epoch["stage"]),
                epoch=int(live_epoch.get("epoch", status.get("epoch", 0))),
                total_epochs=int(
                    live_epoch.get("total_epochs", status.get("epoch", 0) + 1)
                ),
                completed=completed,
                total=total,
                unit="branches",
                detail=(
                    f"{int(live_epoch.get('completed_verifications', 0))} "
                    "verifications completed"
                ),
            )
        )
    note = status.get("note")
    if note:
        lines.append(f"note: {note}")
    return "\n".join(lines)


def print_status(run_dir: Union[str, Path]) -> None:
    status = load_status(run_dir)
    if status is None:
        print(f"no status.json in {run_dir}")
        return
    print(format_status(status))


def format_best_answer(record: BestRecord) -> str:
    answer_payload = record.state.to_dict()["answer_payload"]
    lines = [
        "=== EVOLVE best confirmed answer ===",
        f"state_id: {record.state.state_id}",
        f"internal_reward: {record.state.internal_reward}",
        f"raw_score: {record.state.raw_score}",
        "answer_payload:",
        json.dumps(answer_payload, indent=2, sort_keys=True),
    ]
    if record.rendered_paths:
        lines.append("rendered files:")
        lines.extend(f"  {path}" for path in record.rendered_paths)
    return "\n".join(lines)


def print_best_answer(run_dir: Union[str, Path]) -> None:
    record = load_best(run_dir)
    if record is None:
        print("no confirmed best answer yet")
        return
    print(format_best_answer(record))


def should_print_periodically(verifications_done: int, *, every: int) -> bool:
    """Cadence helper for ``reporting.status_every_verifications``."""

    if every <= 0:
        return False
    return verifications_done > 0 and verifications_done % every == 0


__all__ = [
    "format_best_answer",
    "format_progress",
    "format_status",
    "print_best_answer",
    "print_status",
    "should_print_periodically",
]
