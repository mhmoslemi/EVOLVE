"""Typed, read-only access to the committed ``best/`` artifacts.

Only ever reads what :meth:`evolve.engine.EvolveEngine._publish_best` already
committed at a barrier -- this module never reruns candidate code and never
recomputes a record itself.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Optional, Tuple, Union

from evolve.types import EvidencePacket, VerifiedScientificState


@dataclass(frozen=True)
class BestRecord:
    state: VerifiedScientificState
    evidence: EvidencePacket
    rendered_paths: Tuple[Path, ...]


def load_best(run_dir: Union[str, Path]) -> Optional[BestRecord]:
    best_dir = Path(run_dir) / "best"
    pointer_path = best_dir / "latest.json"
    snapshot_dir = best_dir
    pointer = None
    if pointer_path.is_file():
        pointer = json.loads(pointer_path.read_text(encoding="utf-8"))
        relative = Path(str(pointer.get("snapshot", "")))
        if relative.is_absolute() or ".." in relative.parts or not relative.parts:
            raise ValueError("best/latest.json contains an unsafe snapshot path")
        snapshot_dir = best_dir / relative
    state_path = snapshot_dir / "state.json"
    evidence_path = snapshot_dir / "evidence.json"
    if not state_path.is_file() or not evidence_path.is_file():
        return None
    state = VerifiedScientificState.from_dict(json.loads(state_path.read_text(encoding="utf-8")))
    evidence = EvidencePacket.from_dict(json.loads(evidence_path.read_text(encoding="utf-8")))
    if pointer is not None and (
        pointer.get("state_id") != state.state_id
        or pointer.get("evidence_id") != evidence.evidence_id
    ):
        raise ValueError("best pointer does not match its immutable snapshot")
    rendered = tuple(
        sorted(
            path for path in snapshot_dir.glob("*")
            if path.name not in (
                "state.json",
                "evidence.json",
                "candidate.json",
                "snapshot.manifest.json",
            )
        )
    )
    return BestRecord(state=state, evidence=evidence, rendered_paths=rendered)


__all__ = ["BestRecord", "load_best"]
