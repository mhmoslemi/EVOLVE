"""Posterior allocation scheduler: arms, posterior, portfolio, reservations.

:func:`plan_epoch` composes the four submodules into one reproducible
per-epoch allocation plan: reservations are resolved first (randomized
audits/refinement/harness slot counts are reported for their own subsystems;
every-role, empty/under-tested-cell, and global-exploration reservations are
resolved directly against the candidate pool), then the portfolio search
allocates whatever inflight-branch and resource budget remains by greedy
marginal joint-max value.  The plan does not itself become a
:class:`~evolve.types.BranchSpec`: freezing seeds, role snapshots, and
verifier references into branches is the composed engine's job (Phase 9).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Mapping, Optional, Sequence, Tuple

from evolve.archive import ScientificArchive
from evolve.harness import HarnessRegistry
from evolve.ids import derive_seed
from evolve.options import OptionRegistry
from evolve.types import AllocationArm, Channel, Role

from .arms import ArmCandidate, ArmIdentity, SchedulerError, arm_id_for, enumerate_candidate_arms, make_allocation_arm
from .posterior import (
    HIERARCHY_LEVELS,
    POSTERIOR_VERSION,
    BetaBinomial,
    LevelStats,
    PosteriorError,
    PosteriorSnapshot,
    PosteriorStore,
    ResourceStats,
    hierarchy_key,
)
from .portfolio import (
    DEFAULT_MONTE_CARLO_SAMPLES,
    PORTFOLIO_VERSION,
    PortfolioError,
    ValuedArm,
    select_portfolio,
    simulate_joint_draws,
)
from .reservations import (
    RESERVATIONS_VERSION,
    ReservationError,
    ReservationSlots,
    ResolvedReservations,
    compute_reservation_slots,
    resolve_single_arm_reservations,
)


@dataclass(frozen=True)
class PlannedArm:
    """One arm chosen for this epoch, with the reason and reproducible seed."""

    arm: AllocationArm
    reservation: Optional[str]
    posterior_level: str
    expected_gain: float
    uncertainty: float
    marginal_gain: float
    rng_seed: int


@dataclass(frozen=True)
class AllocationPlan:
    """The complete, reproducible per-epoch scheduler decision."""

    epoch: int
    posterior_version: str
    portfolio_version: str
    reservations_version: str
    reservation_slots: ReservationSlots
    planned_arms: Tuple[PlannedArm, ...]
    seed: int


def plan_epoch(
    *,
    epoch: int,
    archive: ScientificArchive,
    option_registry: OptionRegistry,
    harness_registry: HarnessRegistry,
    posterior: PosteriorStore,
    roles: Sequence[Role],
    max_inflight_branches: int,
    audit_fraction: float,
    no_memory_fraction: float,
    refinement_fraction: float,
    harness_trial_fraction: float,
    empty_cell_fraction: float,
    global_exploration_fraction: float,
    resource_limits: Mapping[str, float],
    seed: int,
    cost_class: str = "standard",
    monte_carlo_samples: int = DEFAULT_MONTE_CARLO_SAMPLES,
) -> AllocationPlan:
    slots = compute_reservation_slots(
        max_inflight_branches=max_inflight_branches,
        audit_fraction=audit_fraction,
        no_memory_fraction=no_memory_fraction,
        refinement_fraction=refinement_fraction,
        harness_trial_fraction=harness_trial_fraction,
        empty_cell_fraction=empty_cell_fraction,
        global_exploration_fraction=global_exploration_fraction,
        roles=roles,
    )
    candidates = enumerate_candidate_arms(
        archive=archive,
        option_registry=option_registry,
        harness_registry=harness_registry,
        roles=roles,
        cost_class=cost_class,
    )
    if not candidates:
        return AllocationPlan(
            epoch=epoch,
            posterior_version=POSTERIOR_VERSION,
            portfolio_version=PORTFOLIO_VERSION,
            reservations_version=RESERVATIONS_VERSION,
            reservation_slots=slots,
            planned_arms=(),
            seed=seed,
        )

    resolved = resolve_single_arm_reservations(candidates, slots=slots, roles=roles, seed=seed)
    remaining_resources: Dict[str, float] = {resource: float(limit) for resource, limit in resource_limits.items()}
    planned: List[PlannedArm] = []

    def reserve(candidate: ArmCandidate, reservation: str) -> None:
        for resource, amount in candidate.expected_cost.items():
            remaining_resources[resource] = remaining_resources.get(resource, 0.0) - float(amount)
        snapshot = posterior.snapshot(candidate.identity)
        arm = make_allocation_arm(
            candidate.identity,
            channel=Channel.PRODUCTION,
            expected_cost=candidate.expected_cost,
            hard_cost=candidate.hard_cost,
        )
        planned.append(
            PlannedArm(
                arm=arm,
                reservation=reservation,
                posterior_level=snapshot.hierarchy_level,
                expected_gain=snapshot.expected_gain,
                uncertainty=snapshot.uncertainty,
                marginal_gain=0.0,
                rng_seed=derive_seed("reserved_arm_seed", seed, candidate.identity.key()),
            )
        )

    for candidate in resolved.role_guarantee_arms:
        reserve(candidate, "role")
    for candidate in resolved.empty_cell_arms:
        reserve(candidate, "empty_cell")
    for candidate in resolved.global_exploration_arms:
        reserve(candidate, "global_exploration")

    portfolio_max = min(slots.remaining_production_slots, len(resolved.remaining_candidates))
    selected = select_portfolio(
        resolved.remaining_candidates,
        posterior=posterior,
        resource_limits=remaining_resources,
        max_arms=portfolio_max,
        seed=seed,
        samples=monte_carlo_samples,
    )
    for valued in selected:
        spec = option_registry.spec(valued.identity.option_id)
        arm = make_allocation_arm(
            valued.identity,
            channel=Channel.PRODUCTION,
            expected_cost=valued.expected_cost,
            hard_cost=dict(spec.hard_cost),
        )
        planned.append(
            PlannedArm(
                arm=arm,
                reservation=None,
                posterior_level=valued.posterior_level,
                expected_gain=valued.expected_gain,
                uncertainty=valued.uncertainty,
                marginal_gain=valued.marginal_gain,
                rng_seed=valued.rng_seed,
            )
        )

    return AllocationPlan(
        epoch=epoch,
        posterior_version=POSTERIOR_VERSION,
        portfolio_version=PORTFOLIO_VERSION,
        reservations_version=RESERVATIONS_VERSION,
        reservation_slots=slots,
        planned_arms=tuple(planned),
        seed=seed,
    )


__all__ = [
    "DEFAULT_MONTE_CARLO_SAMPLES",
    "HIERARCHY_LEVELS",
    "PORTFOLIO_VERSION",
    "POSTERIOR_VERSION",
    "RESERVATIONS_VERSION",
    "AllocationPlan",
    "ArmCandidate",
    "ArmIdentity",
    "BetaBinomial",
    "LevelStats",
    "PlannedArm",
    "PortfolioError",
    "PosteriorError",
    "PosteriorSnapshot",
    "PosteriorStore",
    "ReservationError",
    "ReservationSlots",
    "ResolvedReservations",
    "ResourceStats",
    "SchedulerError",
    "ValuedArm",
    "arm_id_for",
    "compute_reservation_slots",
    "enumerate_candidate_arms",
    "hierarchy_key",
    "make_allocation_arm",
    "plan_epoch",
    "resolve_single_arm_reservations",
    "select_portfolio",
    "simulate_joint_draws",
]
