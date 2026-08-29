"""Contextual causal-memory retrieval, spending no extra rollout budget.

Retrieval is a pure lookup over already-computed :class:`CausalMemoryRecord`
context keys and promotion status -- it never triggers a rollout, verifier
call, or other resource spend, matching AGENTS.md's "retrieve contextually
without extra rollout budget."
"""

from __future__ import annotations

from dataclasses import dataclass, field, replace
from typing import Any, Mapping, Optional, Sequence, Tuple

from evolve.types import CausalMemoryRecord, MemoryStatus


class MemoryStoreError(ValueError):
    """A causal-memory store update would lose or contradict prior evidence."""


def _context_matches(record_context: Mapping[str, Any], query_context: Mapping[str, Any]) -> bool:
    return all(record_context.get(key) == value for key, value in query_context.items())


def _contraindicated(record: CausalMemoryRecord, query_context: Mapping[str, Any]) -> bool:
    tags = {f"{key}:{value}" for key, value in query_context.items()}
    return bool(tags & set(record.contraindications))


@dataclass(frozen=True)
class MemoryStore:
    """Append/update-only collection of causal memory records, by memory_id."""

    records: Mapping[str, CausalMemoryRecord] = field(default_factory=dict)

    def __post_init__(self) -> None:
        object.__setattr__(self, "records", dict(self.records))
        for memory_id, record in self.records.items():
            if record.memory_id != memory_id:
                raise MemoryStoreError(f"memory_id key mismatch for {memory_id}")

    def get(self, memory_id: str) -> Optional[CausalMemoryRecord]:
        return self.records.get(memory_id)

    def upsert(self, record: CausalMemoryRecord) -> "MemoryStore":
        existing = self.records.get(record.memory_id)
        if existing is not None and record.support < existing.support:
            raise MemoryStoreError(
                f"causal memory {record.memory_id} would regress support "
                f"{existing.support} -> {record.support}"
            )
        records = dict(self.records)
        records[record.memory_id] = record
        return replace(self, records=records)

    def promoted_for_context(
        self,
        *,
        context: Mapping[str, Any],
        intervention_option_id: Optional[str] = None,
    ) -> Tuple[CausalMemoryRecord, ...]:
        """Every promoted record whose context matches, ranked by conservative effect."""

        matches = []
        for record in self.records.values():
            if record.status != MemoryStatus.PROMOTED:
                continue
            if intervention_option_id is not None and record.intervention_option_id != intervention_option_id:
                continue
            if not _context_matches(record.context, context):
                continue
            if _contraindicated(record, context):
                continue
            matches.append(record)
        return tuple(sorted(matches, key=lambda record: (-record.conservative_effect, record.memory_id)))

    def recommended_options(
        self,
        *,
        context: Mapping[str, Any],
        candidate_option_ids: Sequence[str],
    ) -> Tuple[str, ...]:
        """Candidate options with promoted, uncontraindicated positive evidence."""

        ranked = self.promoted_for_context(context=context)
        by_option: Mapping[str, CausalMemoryRecord] = {}
        for record in ranked:
            by_option.setdefault(record.intervention_option_id, record)
        ordered = sorted(
            (option_id for option_id in candidate_option_ids if option_id in by_option),
            key=lambda option_id: (-by_option[option_id].conservative_effect, option_id),
        )
        return tuple(ordered)


__all__ = ["MemoryStore", "MemoryStoreError"]
