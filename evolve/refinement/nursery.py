"""Bounded refinement nursery for near-miss and invalid candidates.

Challenger makes one minimal, diagnostic-targeted change per attempt against
one to three attempts, depth at most two, a fixed cost, a strict epoch TTL,
and exactly one entry per source evidence (no re-entry).  Every revision is a
brand-new candidate that must still pass the same, unbiased ("blinded")
independent verifier -- this module never admits or scores a repair itself.
A failed repair is persisted as a useful diagnostic but is not negative
causal option evidence unless it was placed in a valid refinement-vs-fresh
audit (see :func:`open_refinement_audit`, which reuses
:mod:`evolve.audits.pairing`'s matched-pair machinery over the dedicated
:class:`~evolve.types.Channel.REFINEMENT` channel).
"""

from __future__ import annotations

import math
from dataclasses import dataclass, replace
from typing import Any, Mapping, Optional

from evolve.audits.pairing import AuditPairingError, create_audit_pair
from evolve.ids import content_id
from evolve.types import AuditPair, BranchSpec, EvidencePacket


NURSERY_VERSION = "bounded_refinement_nursery_v1"


class NurseryError(ValueError):
    """A nursery entry, attempt, or refinement audit violates its hard bounds."""


def _positive_int(value: Any, name: str, *, maximum: int) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or not 1 <= value <= maximum:
        raise NurseryError(f"{name} must be an integer in [1, {maximum}]")
    return value


def _resource_map(value: Mapping[str, Any], name: str) -> Mapping[str, float]:
    out = {}
    for resource, amount in value.items():
        if not isinstance(resource, str) or not resource.strip():
            raise NurseryError(f"{name} resource names must be non-empty")
        if isinstance(amount, bool) or not isinstance(amount, (int, float)) or not math.isfinite(amount) or amount < 0.0:
            raise NurseryError(f"{name}.{resource} must be finite and non-negative")
        out[resource] = float(amount)
    return out


@dataclass(frozen=True)
class NurseryPolicy:
    """Hard, method-fixed bounds: at most 3 attempts, depth at most 2."""

    max_attempts: int
    max_depth: int
    fixed_cost: Mapping[str, float]
    ttl_epochs: int

    def __post_init__(self) -> None:
        object.__setattr__(self, "max_attempts", _positive_int(self.max_attempts, "max_attempts", maximum=3))
        object.__setattr__(self, "max_depth", _positive_int(self.max_depth, "max_depth", maximum=2))
        object.__setattr__(self, "fixed_cost", _resource_map(self.fixed_cost, "fixed_cost"))
        if isinstance(self.ttl_epochs, bool) or not isinstance(self.ttl_epochs, int) or self.ttl_epochs < 1:
            raise NurseryError("ttl_epochs must be a positive integer")


@dataclass(frozen=True)
class NurseryEntry:
    """One near-miss/invalid candidate's bounded, non-re-entrant repair slot."""

    entry_id: str
    source_evidence_id: str
    source_proposal_id: str
    branch_id: str
    opened_epoch: int
    ttl_epochs: int
    max_attempts: int
    max_depth: int
    attempts_used: int = 0
    depth: int = 0
    closed: bool = False
    admitted_evidence_id: Optional[str] = None

    @property
    def expiry_epoch(self) -> int:
        return self.opened_epoch + self.ttl_epochs

    def is_expired(self, epoch: int) -> bool:
        return epoch >= self.expiry_epoch

    def attempts_remaining(self) -> int:
        return max(0, self.max_attempts - self.attempts_used)

    def can_attempt(self, epoch: int) -> bool:
        return (
            not self.closed
            and not self.is_expired(epoch)
            and self.attempts_remaining() > 0
            and self.depth < self.max_depth
        )


def open_entry(
    *,
    source_evidence: EvidencePacket,
    branch_id: str,
    epoch: int,
    policy: NurseryPolicy,
) -> NurseryEntry:
    """Admit one non-admitted candidate into the nursery, exactly once."""

    if source_evidence.admitted:
        raise NurseryError("the nursery is only for invalid or near-miss (non-admitted) evidence")
    entry_id = content_id(
        "nursery_entry",
        {
            "source_evidence_id": source_evidence.evidence_id,
            "branch_id": branch_id,
            "opened_epoch": epoch,
        },
    )
    return NurseryEntry(
        entry_id=entry_id,
        source_evidence_id=source_evidence.evidence_id,
        source_proposal_id=source_evidence.proposal_id,
        branch_id=branch_id,
        opened_epoch=epoch,
        ttl_epochs=policy.ttl_epochs,
        max_attempts=policy.max_attempts,
        max_depth=policy.max_depth,
    )


def record_attempt(
    entry: NurseryEntry,
    *,
    repair_evidence: EvidencePacket,
    epoch: int,
) -> NurseryEntry:
    """Fold one blinded-verified repair attempt into the entry's bounded state.

    A successful repair closes the entry immediately.  A failed repair is
    still persisted by the caller as diagnostic evidence, but this entry only
    tracks the bounded counters; it never becomes negative causal evidence
    unless the caller separately places it in a refinement audit.
    """

    if not entry.can_attempt(epoch):
        raise NurseryError(
            f"nursery entry {entry.entry_id} cannot accept another attempt "
            "(closed, expired, or its attempt/depth bound is exhausted)"
        )
    if repair_evidence.parent_state_id is None:
        raise NurseryError("a repair attempt must reference its parent lineage")

    attempts_used = entry.attempts_used + 1
    depth = entry.depth + 1
    closed = repair_evidence.admitted or attempts_used >= entry.max_attempts or depth >= entry.max_depth
    admitted_evidence_id = repair_evidence.evidence_id if repair_evidence.admitted else entry.admitted_evidence_id
    return replace(
        entry,
        attempts_used=attempts_used,
        depth=depth,
        closed=closed,
        admitted_evidence_id=admitted_evidence_id,
    )


def expire_entry(entry: NurseryEntry, *, epoch: int) -> NurseryEntry:
    """Close an entry whose strict TTL has elapsed without a successful repair."""

    if entry.closed or not entry.is_expired(epoch):
        return entry
    return replace(entry, closed=True)


def open_refinement_audit(
    *,
    run_id: str,
    cell_id: str,
    refinement_branch: BranchSpec,
    fresh_continuation_branch: BranchSpec,
    assignment_probability: float,
    assignment_seed: int,
) -> AuditPair:
    """Randomize an eligible case between nursery repair and equal-cost fresh continuation.

    Both branches must already be frozen on the dedicated
    :class:`~evolve.types.Channel.REFINEMENT` channel with identical cost;
    reuses :func:`evolve.audits.pairing.create_audit_pair`'s matched-context
    validation so this comparison closes exactly like any other audit pair.
    """

    try:
        return create_audit_pair(
            run_id=run_id,
            cell_id=cell_id,
            intervention_branch=refinement_branch,
            control_branch=fresh_continuation_branch,
            assignment_probability=assignment_probability,
            assignment_seed=assignment_seed,
        )
    except AuditPairingError as exc:
        raise NurseryError(f"refinement audit pairing failed: {exc}") from exc


__all__ = [
    "NURSERY_VERSION",
    "NurseryEntry",
    "NurseryError",
    "NurseryPolicy",
    "expire_entry",
    "open_entry",
    "open_refinement_audit",
    "record_attempt",
]
