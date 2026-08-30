"""Mandatory exploration/audit/refinement/harness reservations.

AGENTS.md's "Posterior allocation" requires reserving randomized audits,
every role, empty or under-tested cells, harness calibration, and global
exploration *before* the portfolio search allocates the remainder by greedy
marginal record-improvement value.  This module turns the configured
fractions of ``max_inflight_branches`` into concrete slot counts, resolves
the reservations that are plain single-arm selections (every role, empty/
under-tested cells, global exploration) against the candidate pool, and
reports the remaining slot counts that the audit, refinement, and harness
subsystems must fill with their own matched-pair logic.
"""

from __future__ import annotations

import math
import random
from dataclasses import dataclass
from typing import List, Sequence, Tuple

from evolve.ids import derive_seed
from evolve.types import Role

from .arms import ArmCandidate


RESERVATIONS_VERSION = "epoch_reservations_v1"


class ReservationError(ValueError):
    """A reservation computation received invalid inputs."""


def _fraction(value: float, name: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ReservationError(f"{name} must be numeric")
    result = float(value)
    if not 0.0 <= result <= 1.0:
        raise ReservationError(f"{name} must lie in [0, 1]")
    return result


def _slots(inflight: int, fraction: float) -> int:
    return int(math.ceil(inflight * fraction)) if fraction > 0.0 else 0


def _paired_slots(inflight: int, fraction: float) -> int:
    requested = _slots(inflight, fraction)
    if requested == 0:
        return 0
    return requested if requested % 2 == 0 else requested + 1


@dataclass(frozen=True)
class ReservationSlots:
    """Slot counts reserved out of one epoch's ``max_inflight_branches``."""

    total_inflight: int
    audit_branch_slots: int
    no_memory_audit_slots: int
    refinement_slots: int
    harness_trial_slots: int
    empty_cell_slots: int
    global_exploration_slots: int
    role_guaranteed_slots: int
    remaining_production_slots: int

    def __post_init__(self) -> None:
        for name in (
            "total_inflight", "audit_branch_slots", "no_memory_audit_slots",
            "refinement_slots", "harness_trial_slots", "empty_cell_slots",
            "global_exploration_slots", "role_guaranteed_slots",
            "remaining_production_slots",
        ):
            value = getattr(self, name)
            if isinstance(value, bool) or not isinstance(value, int) or value < 0:
                raise ReservationError(f"{name} must be a non-negative integer")
        if self.no_memory_audit_slots > self.audit_branch_slots:
            raise ReservationError("no_memory_audit_slots cannot exceed audit_branch_slots")


def compute_reservation_slots(
    *,
    max_inflight_branches: int,
    audit_fraction: float,
    no_memory_fraction: float,
    refinement_fraction: float,
    harness_trial_fraction: float,
    empty_cell_fraction: float,
    global_exploration_fraction: float,
    roles: Sequence[Role],
) -> ReservationSlots:
    if (
        isinstance(max_inflight_branches, bool)
        or not isinstance(max_inflight_branches, int)
        or max_inflight_branches < 1
    ):
        raise ReservationError("max_inflight_branches must be a positive integer")
    inflight = max_inflight_branches
    requested_audit_slots = _paired_slots(
        inflight, _fraction(audit_fraction, "audit_fraction")
    )
    no_memory_slots = _paired_slots(
        inflight, _fraction(no_memory_fraction, "no_memory_fraction")
    )
    # No-memory evidence is itself a matched audit, so its reservation cannot
    # disappear merely because the general audit fraction is lower or zero.
    audit_slots = max(requested_audit_slots, no_memory_slots)
    refinement_slots = _paired_slots(
        inflight, _fraction(refinement_fraction, "refinement_fraction")
    )
    harness_slots = _paired_slots(
        inflight, _fraction(harness_trial_fraction, "harness_trial_fraction")
    )
    empty_cell_slots = _slots(inflight, _fraction(empty_cell_fraction, "empty_cell_fraction"))
    exploration_slots = _slots(
        inflight, _fraction(global_exploration_fraction, "global_exploration_fraction")
    )
    role_slots = len(set(roles))
    reserved_total = (
        audit_slots + refinement_slots + harness_slots
        + empty_cell_slots + exploration_slots + role_slots
    )
    remaining = max(0, inflight - reserved_total)
    return ReservationSlots(
        total_inflight=inflight,
        audit_branch_slots=audit_slots,
        no_memory_audit_slots=no_memory_slots,
        refinement_slots=refinement_slots,
        harness_trial_slots=harness_slots,
        empty_cell_slots=empty_cell_slots,
        global_exploration_slots=exploration_slots,
        role_guaranteed_slots=role_slots,
        remaining_production_slots=remaining,
    )


@dataclass(frozen=True)
class ResolvedReservations:
    """Concrete single-arm reservations, plus the still-open candidate pool."""

    role_guarantee_arms: Tuple[ArmCandidate, ...]
    empty_cell_arms: Tuple[ArmCandidate, ...]
    global_exploration_arms: Tuple[ArmCandidate, ...]
    remaining_candidates: Tuple[ArmCandidate, ...]
    rng_seed: int


def _select_unique(rng: random.Random, pool: List[ArmCandidate], count: int) -> List[ArmCandidate]:
    if count <= 0 or not pool:
        return []
    shuffled = list(pool)
    rng.shuffle(shuffled)
    return shuffled[:count]


def resolve_single_arm_reservations(
    candidates: Sequence[ArmCandidate],
    *,
    slots: ReservationSlots,
    roles: Sequence[Role],
    seed: int,
) -> ResolvedReservations:
    """Randomly resolve the reservations that need no matched-pair logic."""

    rng_seed = derive_seed("reservation_selection", seed, slots.total_inflight)
    rng = random.Random(rng_seed)
    all_candidates: List[ArmCandidate] = list(candidates)

    def candidate_key(candidate: ArmCandidate):
        return candidate.identity.key()

    role_arms: List[ArmCandidate] = []
    for role in dict.fromkeys(roles):
        role_pool = [
            candidate for candidate in all_candidates
            if candidate.identity.role == role
        ]
        if not role_pool:
            continue
        picked = rng.choice(role_pool)
        role_arms.append(picked)

    empty_pool = [
        candidate for candidate in all_candidates
        if candidate.cell_empty or candidate.cell_under_tested
    ]
    role_keys = {candidate_key(candidate) for candidate in role_arms}
    empty_overlap = [
        candidate for candidate in empty_pool
        if candidate_key(candidate) in role_keys
    ]
    empty_other = [
        candidate for candidate in empty_pool
        if candidate_key(candidate) not in role_keys
    ]
    rng.shuffle(empty_overlap)
    rng.shuffle(empty_other)
    empty_arms = (empty_overlap + empty_other)[: slots.empty_cell_slots]

    # A randomized global-exploration branch may simultaneously satisfy a role
    # or empty-cell reservation. Prefer those already-reserved arms so finite
    # capacity is spent on execution rather than duplicated labels, while the
    # choice within each stratum remains seed-reproducible.
    prior = role_arms + empty_arms
    prior_by_key = {candidate_key(candidate): candidate for candidate in prior}
    exploration_overlap = list(prior_by_key.values())
    exploration_other = [
        candidate for candidate in all_candidates
        if candidate_key(candidate) not in prior_by_key
    ]
    rng.shuffle(exploration_overlap)
    rng.shuffle(exploration_other)
    exploration_arms = (
        exploration_overlap + exploration_other
    )[: slots.global_exploration_slots]

    reserved_keys = {
        candidate_key(candidate)
        for candidate in (*role_arms, *empty_arms, *exploration_arms)
    }
    remaining_pool = [
        candidate for candidate in all_candidates
        if candidate_key(candidate) not in reserved_keys
    ]

    return ResolvedReservations(
        role_guarantee_arms=tuple(role_arms),
        empty_cell_arms=tuple(empty_arms),
        global_exploration_arms=tuple(exploration_arms),
        remaining_candidates=tuple(remaining_pool),
        rng_seed=rng_seed,
    )


__all__ = [
    "RESERVATIONS_VERSION",
    "ReservationError",
    "ReservationSlots",
    "ResolvedReservations",
    "compute_reservation_slots",
    "resolve_single_arm_reservations",
]
