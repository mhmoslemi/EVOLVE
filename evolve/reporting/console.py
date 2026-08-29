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
    lines = [
        "=== EVOLVE best confirmed answer ===",
        f"state_id: {record.state.state_id}",
        f"internal_reward: {record.state.internal_reward}",
        f"raw_score: {record.state.raw_score}",
        "answer_payload:",
        json.dumps(record.state.answer_payload, indent=2, sort_keys=True),
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
    "format_status",
    "print_best_answer",
    "print_status",
    "should_print_periodically",
]
