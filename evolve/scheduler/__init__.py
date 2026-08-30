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

from dataclasses import dataclass, replace
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
    posterior_support: int = 0
    reliability_probability: float = 0.5
    admission_probability: float = 0.5
    improvement_probability_given_admission: float = 0.5
    mean_positive_gain: float = 0.0
    replicas: int = 1
    reservations: Tuple[str, ...] = ()
    correlation_penalty: float = 0.0
    predicted_cost_uncertainty: Mapping[str, float] = None

    def __post_init__(self) -> None:
        if self.replicas < 1:
            raise SchedulerError("planned arm replicas must be positive")
        labels = tuple(self.reservations) or (
            ((self.reservation,) if self.reservation is not None else ())
        )
        object.__setattr__(self, "reservations", labels)
        object.__setattr__(
            self,
            "predicted_cost_uncertainty",
            dict(self.predicted_cost_uncertainty or {}),
        )


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
    learning_role: Optional[Role] = None,
    group_k: int = 1,
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

    allocation_resource_limits = {
        resource: float(limit) for resource, limit in resource_limits.items()
    }
    if "verifier_calls" in allocation_resource_limits:
        maximum_branch_calls = max(
            float(candidate.hard_cost.get("verifier_calls", 1.0))
            for candidate in candidates
        )
        reserved_calls = (
            (slots.audit_branch_slots + slots.harness_trial_slots)
            * maximum_branch_calls
            + 2.0 * slots.refinement_slots
        )
        allocation_resource_limits["verifier_calls"] = max(
            0.0,
            allocation_resource_limits["verifier_calls"] - reserved_calls,
        )

    # Replace static costs with conservative hierarchical resource estimates
    # whenever the posterior has support. These estimates are logged even for
    # resources without a configured hard global limit.
    adjusted_candidates = []
    cost_uncertainty: Dict[Tuple[object, ...], Dict[str, float]] = {}
    for candidate in candidates:
        expected = dict(candidate.expected_cost)
        uncertainty_by_resource: Dict[str, float] = {}
        resources = set(expected) | set(allocation_resource_limits)
        for resource in resources:
            estimate = posterior.resource_estimate(candidate.identity, resource)
            if estimate.count:
                predicted = max(0.0, estimate.mean + estimate.std)
                if resource in candidate.hard_cost:
                    predicted = min(predicted, float(candidate.hard_cost[resource]))
                expected[resource] = predicted
                uncertainty_by_resource[resource] = estimate.std
            else:
                uncertainty_by_resource[resource] = 0.0
        adjusted = replace(candidate, expected_cost=expected)
        adjusted_candidates.append(adjusted)
        cost_uncertainty[candidate.identity.key()] = uncertainty_by_resource

    resolved = resolve_single_arm_reservations(
        adjusted_candidates, slots=slots, roles=roles, seed=seed
    )
    remaining_resources: Dict[str, float] = {
        resource: float(limit)
        for resource, limit in allocation_resource_limits.items()
    }
    planned: List[PlannedArm] = []
    reserved_index: Dict[str, int] = {}

    def reserve(candidate: ArmCandidate, reservation: str) -> None:
        candidate_arm = make_allocation_arm(
            candidate.identity,
            channel=Channel.PRODUCTION,
            expected_cost=candidate.expected_cost,
            hard_cost=candidate.hard_cost,
        )
        if candidate_arm.arm_id in reserved_index:
            index = reserved_index[candidate_arm.arm_id]
            current = planned[index]
            planned[index] = replace(
                current,
                reservations=tuple(
                    dict.fromkeys((*current.reservations, reservation))
                ),
            )
            return
        for resource in tuple(remaining_resources):
            remaining_resources[resource] = max(
                0.0,
                remaining_resources[resource]
                - float(candidate.expected_cost.get(resource, 0.0)),
            )
        snapshot = posterior.snapshot(candidate.identity)
        arm = candidate_arm
        reserved_index[arm.arm_id] = len(planned)
        planned.append(
            PlannedArm(
                arm=arm,
                reservation=reservation,
                reservations=(reservation,),
                posterior_level=snapshot.hierarchy_level,
                expected_gain=snapshot.expected_gain,
                uncertainty=snapshot.uncertainty,
                marginal_gain=0.0,
                rng_seed=derive_seed("reserved_arm_seed", seed, candidate.identity.key()),
                posterior_support=snapshot.support,
                reliability_probability=snapshot.reliability_probability,
                admission_probability=snapshot.admission_probability,
                improvement_probability_given_admission=(
                    snapshot.improvement_probability_given_admission
                ),
                mean_positive_gain=snapshot.mean_positive_gain,
                predicted_cost_uncertainty=cost_uncertainty[candidate.identity.key()],
            )
        )

    for candidate in resolved.role_guarantee_arms:
        reserve(candidate, "role")
    for candidate in resolved.empty_cell_arms:
        reserve(candidate, "empty_cell")
    for candidate in resolved.global_exploration_arms:
        reserve(candidate, "global_exploration")

    reserved_candidate_keys = {
        candidate.identity.key()
        for candidate in (
            *resolved.role_guarantee_arms,
            *resolved.empty_cell_arms,
            *resolved.global_exploration_arms,
        )
    }
    production_capacity_before_replicas = max(
        0,
        slots.total_inflight
        - slots.audit_branch_slots
        - slots.refinement_slots
        - slots.harness_trial_slots,
    )
    # Role, empty-cell, and global-exploration labels may intentionally share
    # one execution. Spend the released capacity on additional portfolio arms
    # instead of treating overlapping labels as separate consumed branches.
    portfolio_max = min(
        max(0, production_capacity_before_replicas - len(reserved_candidate_keys)),
        len(resolved.remaining_candidates),
    )
    selected = select_portfolio(
        resolved.remaining_candidates,
        posterior=posterior,
        resource_limits=remaining_resources,
        max_arms=portfolio_max,
        seed=seed,
        samples=monte_carlo_samples,
    )
    for valued in selected:
        arm = make_allocation_arm(
            valued.identity,
            channel=Channel.PRODUCTION,
            expected_cost=valued.expected_cost,
            hard_cost=valued.hard_cost,
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
                posterior_support=valued.support,
                reliability_probability=valued.reliability_probability,
                admission_probability=valued.admission_probability,
                improvement_probability_given_admission=(
                    valued.improvement_probability_given_admission
                ),
                mean_positive_gain=valued.mean_positive_gain,
                correlation_penalty=valued.correlation_penalty,
                predicted_cost_uncertainty=cost_uncertainty[valued.identity.key()],
            )
        )

    # The plan, not the executor, owns homogeneous replica counts. Select one
    # learning arm while preserving every-role coverage and make exploration
    # labels multi-purpose when finite capacity requires it.
    normalized_learning_role = (
        learning_role if isinstance(learning_role, Role) else
        (Role(learning_role) if learning_role is not None else None)
    )
    production_capacity = max(
        0,
        slots.total_inflight
        - slots.audit_branch_slots
        - slots.refinement_slots
        - slots.harness_trial_slots,
    )
    selected: List[PlannedArm] = []
    used_ids = set()
    selected_resources: Dict[str, float] = {
        resource: 0.0 for resource in allocation_resource_limits
    }

    def execution_count() -> int:
        return sum(item.replicas for item in selected)

    def add(item: PlannedArm, *, replicas: int, labels: Sequence[str]) -> bool:
        if item.arm.arm_id in used_ids:
            return False
        if execution_count() + replicas > production_capacity:
            return False
        for resource, limit in allocation_resource_limits.items():
            required = replicas * float(
                item.arm.hard_cost.get(
                    resource, item.arm.expected_cost.get(resource, 0.0)
                )
            )
            if selected_resources[resource] + required > float(limit) + 1e-12:
                return False
        selected.append(
            replace(
                item,
                replicas=replicas,
                reservation=(labels[0] if labels else item.reservation),
                reservations=tuple(dict.fromkeys(labels)),
            )
        )
        used_ids.add(item.arm.arm_id)
        for resource in selected_resources:
            selected_resources[resource] += replicas * float(
                item.arm.hard_cost.get(
                    resource, item.arm.expected_cost.get(resource, 0.0)
                )
            )
        return True

    if normalized_learning_role is not None:
        learning_candidates = [
            item for item in planned if item.arm.role == normalized_learning_role
        ]
        learning_candidates.sort(
            key=lambda item: (
                0 if "empty_cell" in item.reservations else 1,
                0 if "global_exploration" in item.reservations else 1,
                item.arm.arm_id,
            )
        )
        if learning_candidates and group_k + max(0, len(roles) - 1) <= production_capacity:
            labels = ["role", "learning_group"]
            labels.extend(learning_candidates[0].reservations)
            add(learning_candidates[0], replicas=max(1, group_k), labels=labels)

    for role in roles:
        if any(item.arm.role == role for item in selected):
            continue
        role_candidates = [item for item in planned if item.arm.role == role]
        role_candidates.sort(
            key=lambda item: (
                0
                if any(
                    label in item.reservations
                    for label in ("empty_cell", "global_exploration", "role")
                )
                else 1,
                item.arm.arm_id,
            )
        )
        if role_candidates:
            labels = ["role"]
            labels.extend(role_candidates[0].reservations)
            add(role_candidates[0], replicas=1, labels=labels)

    for label, required in (
        ("empty_cell", slots.empty_cell_slots),
        ("global_exploration", slots.global_exploration_slots),
    ):
        realized = sum(
            item.replicas for item in selected if label in item.reservations
        )
        for item in planned:
            if realized >= required or execution_count() >= production_capacity:
                break
            if label not in item.reservations or item.arm.arm_id in used_ids:
                continue
            if add(item, replicas=1, labels=(label,)):
                realized += 1

    for item in planned:
        if execution_count() >= production_capacity:
            break
        if item.arm.arm_id not in used_ids:
            add(
                item,
                replicas=1,
                labels=item.reservations,
            )

    planned = selected

    missing_roles = [
        role.value
        for role in roles
        if not any(item.arm.role == role for item in planned)
    ]
    if missing_roles:
        raise SchedulerError(
            "resource limits cannot satisfy mandatory role reservations: "
            + ", ".join(missing_roles)
        )
    for label, resolved_count in (
        ("empty_cell", len(resolved.empty_cell_arms)),
        ("global_exploration", len(resolved.global_exploration_arms)),
    ):
        realized = sum(
            item.replicas for item in planned if label in item.reservations
        )
        if realized < resolved_count:
            raise SchedulerError(
                f"resource limits cannot satisfy mandatory {label} reservation"
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
