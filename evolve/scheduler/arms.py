"""Allocation-arm identity and candidate enumeration.

An allocation arm is ``(cell_id, role, option_id, harness_version, horizon,
cost_class)``.  This module defines that identity, derives a stable
content-addressed :class:`~evolve.types.AllocationArm`, and enumerates the
candidate arms available this epoch from the current archive, option
registry, and harness registry -- without touching the posterior or a
resource budget, which live in :mod:`evolve.scheduler.posterior` and
:mod:`evolve.scheduler.portfolio`/:mod:`evolve.scheduler.reservations`.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping, Optional, Sequence, Tuple

from evolve.archive import ScientificArchive
from evolve.harness import HarnessRegistry
from evolve.ids import content_id, validate_id
from evolve.options import OptionRegistry
from evolve.options import (
    DIAGNOSTIC_REPAIR_STATE_MACHINE,
    FRESH_REFINEMENT_CONTROL_STATE_MACHINE,
    MATCHED_CONTINUATION_STATE_MACHINE,
)
from evolve.types import AllocationArm, ArchiveCell, Channel, Role


class SchedulerError(ValueError):
    """Base error for scheduler-layer failures."""


_AUXILIARY_STATE_MACHINES = {
    DIAGNOSTIC_REPAIR_STATE_MACHINE,
    FRESH_REFINEMENT_CONTROL_STATE_MACHINE,
    MATCHED_CONTINUATION_STATE_MACHINE,
}


@dataclass(frozen=True)
class ArmIdentity:
    """The six frozen dimensions that define one allocation arm's identity."""

    cell_id: str
    role: Role
    option_id: str
    harness_id: str
    horizon: int
    cost_class: str

    def __post_init__(self) -> None:
        validate_id(self.cell_id, "cell")
        owner = self.role if isinstance(self.role, Role) else Role(self.role)
        object.__setattr__(self, "role", owner)
        validate_id(self.option_id, "option")
        validate_id(self.harness_id, "harness")
        if isinstance(self.horizon, bool) or not isinstance(self.horizon, int) or self.horizon < 1:
            raise SchedulerError("horizon must be a positive integer")
        if not isinstance(self.cost_class, str) or not self.cost_class.strip():
            raise SchedulerError("cost_class must be a non-empty string")

    @classmethod
    def from_arm(cls, arm: AllocationArm) -> "ArmIdentity":
        return cls(
            cell_id=arm.cell_id,
            role=arm.role,
            option_id=arm.option_id,
            harness_id=arm.harness_id,
            horizon=arm.horizon,
            cost_class=arm.cost_class,
        )

    def key(self) -> Tuple[str, str, str, str, int, str]:
        return (self.cell_id, self.role.value, self.option_id, self.harness_id, self.horizon, self.cost_class)


def arm_id_for(identity: ArmIdentity, *, channel: Channel) -> str:
    """A stable identity shared by every epoch's instance of the same arm.

    Cost is deliberately excluded: it is a chosen amount for one epoch's plan,
    not part of the arm's logical identity, so hierarchical statistics stay
    keyed on the arm across epochs.
    """

    return content_id(
        "arm",
        {
            "cell_id": identity.cell_id,
            "role": identity.role.value,
            "option_id": identity.option_id,
            "harness_id": identity.harness_id,
            "horizon": identity.horizon,
            "cost_class": identity.cost_class,
            "channel": channel.value,
        },
    )


def make_allocation_arm(
    identity: ArmIdentity,
    *,
    channel: Channel = Channel.PRODUCTION,
    expected_cost: Mapping[str, Any],
    hard_cost: Mapping[str, Any],
) -> AllocationArm:
    return AllocationArm(
        arm_id=arm_id_for(identity, channel=channel),
        cell_id=identity.cell_id,
        role=identity.role,
        option_id=identity.option_id,
        harness_id=identity.harness_id,
        horizon=identity.horizon,
        cost_class=identity.cost_class,
        channel=channel,
        expected_cost=dict(expected_cost),
        hard_cost=dict(hard_cost),
    )


@dataclass(frozen=True)
class ArmCandidate:
    """One eligible arm this epoch, before any posterior or portfolio scoring."""

    identity: ArmIdentity
    expected_cost: Mapping[str, float]
    hard_cost: Mapping[str, float]
    cell_empty: bool
    cell_under_tested: bool
    start_state_id: Optional[str] = None
    fingerprint_family: str = ""
    option_family: str = ""


def enumerate_candidate_arms(
    *,
    archive: ScientificArchive,
    option_registry: OptionRegistry,
    harness_registry: HarnessRegistry,
    roles: Sequence[Role],
    cost_class: str = "standard",
    cell_ids: Optional[Sequence[str]] = None,
) -> Tuple[ArmCandidate, ...]:
    """Every structurally eligible ``(cell, role, option, harness)`` this epoch.

    An option's declared ``prerequisites`` gate it against the cell's tested
    state: an option that requires ``verified_start`` is only offered for a
    cell that already has at least one tested (and therefore verified)
    candidate to branch from.
    """

    cells: Sequence[ArchiveCell] = (
        archive.cells if cell_ids is None else tuple(archive.cell(cid) for cid in cell_ids)
    )
    candidates = []
    normalized_roles = tuple(role if isinstance(role, Role) else Role(role) for role in roles)
    global_launch_state_id: Optional[str] = None
    global_launch_fingerprint = ""
    global_champions = []
    for archive_cell in archive.cells:
        if archive_cell.champion_state_id is None:
            continue
        state = archive.artifacts.representative_state(
            archive_cell.champion_state_id,
            descriptor_id=archive_cell.descriptor_id,
        )
        global_champions.append(
            (float(state.internal_reward), state.state_id, state.fingerprint)
        )
    if global_champions:
        _, global_launch_state_id, global_launch_fingerprint = max(
            global_champions, key=lambda item: (item[0], item[1])
        )
    for cell in cells:
        satisfied = {"verified_start"} if cell.tested_count > 0 else set()
        start_state_id = (
            cell.champion_state_id
            or (cell.promising_state_ids[0] if cell.promising_state_ids else None)
            or (cell.stepping_stone_state_ids[0] if cell.stepping_stone_state_ids else None)
            or global_launch_state_id
        )
        fingerprint_family = (
            global_launch_fingerprint
            if start_state_id == global_launch_state_id
            else ""
        )
        if (
            start_state_id is not None
            and start_state_id != global_launch_state_id
        ):
            try:
                fingerprint_family = archive.artifacts.representative_state(
                    start_state_id, descriptor_id=cell.descriptor_id
                ).fingerprint
            except Exception:
                # Identity is still exact through cell/start; a missing optional
                # family label must not make an otherwise valid arm disappear.
                fingerprint_family = ""
        for role in normalized_roles:
            for harness_id in harness_registry.active_ids:
                for option_id in option_registry.eligible_for(role=role, harness_id=harness_id):
                    spec = option_registry.spec(option_id)
                    # Audit controls and nursery repairs have dedicated
                    # channels. They must never become ordinary production
                    # arms merely because their role/harness is eligible.
                    if spec.state_machine in _AUXILIARY_STATE_MACHINES:
                        continue
                    if not set(spec.prerequisites) <= satisfied:
                        continue
                    for horizon in range(1, spec.max_horizon + 1):
                        scale = float(horizon) / float(spec.max_horizon)
                        identity = ArmIdentity(
                            cell_id=cell.cell_id,
                            role=role,
                            option_id=option_id,
                            harness_id=harness_id,
                            horizon=horizon,
                            cost_class=cost_class,
                        )
                        candidates.append(
                            ArmCandidate(
                                identity=identity,
                                expected_cost={
                                    resource: float(amount) * scale
                                    for resource, amount in spec.expected_cost.items()
                                },
                                hard_cost={
                                    resource: (
                                        float(amount)
                                        * scale
                                        * (2.0 if resource == "verifier_calls" else 1.0)
                                    )
                                    for resource, amount in spec.hard_cost.items()
                                },
                                cell_empty=cell.tested_count == 0,
                                cell_under_tested=cell.under_tested,
                                start_state_id=start_state_id,
                                fingerprint_family=fingerprint_family,
                                option_family=spec.state_machine,
                            )
                        )
    return tuple(candidates)


__all__ = [
    "ArmCandidate",
    "ArmIdentity",
    "SchedulerError",
    "arm_id_for",
    "enumerate_candidate_arms",
    "make_allocation_arm",
]
