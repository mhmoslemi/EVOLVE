"""Barrier-only role-policy training from persisted homogeneous groups.

The actual forward/backward pass through the live backbone is owned by an
injected :data:`GradientStepFn` -- mirroring how :mod:`evolve.verifier.service`
injects a problem adapter and :mod:`evolve.options.branch` injects a branch
step executor.  This module stays pure.

A role receives at most one gradient step per barrier: every persisted
homogeneous group belonging to that role and this barrier's policy snapshot
is concatenated into a single request so one optimizer step sees every
group's contribution together, matching "role-isolated barrier updates."
Each contributing group is still claimed (and therefore logged and rejected
from ever training twice) individually.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Dict, List, Mapping, Optional, Sequence, Tuple

from evolve.roles import RoleRegistry
from evolve.types import LearningGroup, LearningObjective, PolicyTrace, Role


class TrainerError(ValueError):
    """A learning-group barrier update cannot be applied as requested."""


@dataclass(frozen=True)
class GradientStepRequest:
    """Everything the injected gradient step needs to realize one role update."""

    role: Role
    objective: LearningObjective
    objective_version: str
    traces: Tuple[PolicyTrace, ...]
    advantages: Tuple[float, ...]
    group_ids: Tuple[str, ...]
    kl_penalty_coef: float


@dataclass(frozen=True)
class GradientStepResult:
    """The realized update: diagnostics plus the role's new persisted state."""

    loss: float
    kl: float
    gradient_norm: float
    adapter_state: Mapping[str, object]
    optimizer_state: Mapping[str, object]


GradientStepFn = Callable[[GradientStepRequest], GradientStepResult]


@dataclass(frozen=True)
class LearningUpdate:
    """The complete durable product of training one role for one barrier."""

    groups: Tuple[LearningGroup, ...]
    result: GradientStepResult
    role_snapshot_id_before: str
    adapter_hash_before: str
    adapter_hash_after: str
    registry: RoleRegistry


def _zero_advantages(group: LearningGroup) -> bool:
    return all(abs(value) < 1e-12 for value in group.advantages)


def train_role_groups(
    role: Role,
    groups: Sequence[LearningGroup],
    *,
    traces_by_id: Mapping[str, PolicyTrace],
    registry: RoleRegistry,
    epoch: int,
    gradient_step: GradientStepFn,
    kl_penalty_coef: float = 0.0,
    skip_zero_advantage: bool = True,
) -> Optional[LearningUpdate]:
    if not groups:
        return None
    for group in groups:
        if group.role != role:
            raise TrainerError(f"group {group.group_id} does not belong to role {role.value!r}")
        if not (group.homogeneous and group.on_policy and group.persisted_inputs):
            raise TrainerError(
                f"learning group {group.group_id} failed its own homogeneity/on-policy invariants"
            )

    role_state = registry.state(role)
    expected_snapshot = role_state.freeze(epoch)
    for group in groups:
        if group.policy_snapshot_id != expected_snapshot.snapshot_id:
            raise TrainerError(
                f"learning group {group.group_id} policy snapshot is stale for epoch {epoch}"
            )

    objectives = {group.objective for group in groups}
    objective_versions = {group.objective_version for group in groups}
    if len(objectives) > 1 or len(objective_versions) > 1:
        raise TrainerError(f"role {role.value!r} has groups mixing objectives/versions in one barrier")

    if skip_zero_advantage and all(_zero_advantages(group) for group in groups):
        return None

    traces: List[PolicyTrace] = []
    advantages: List[float] = []
    group_ids: List[str] = []
    for group in groups:
        try:
            group_traces = [traces_by_id[trace_id] for trace_id in group.trace_ids]
        except KeyError as exc:
            raise TrainerError(f"missing persisted policy trace for group {group.group_id}") from exc
        for trace, advantage in zip(group_traces, group.advantages):
            if trace.role != role or trace.role_snapshot_id != expected_snapshot.snapshot_id:
                raise TrainerError(
                    f"policy trace {trace.trace_id} does not match group {group.group_id}'s role/snapshot"
                )
            traces.append(trace)
            advantages.append(advantage)
            group_ids.append(group.group_id)

    request = GradientStepRequest(
        role=role,
        objective=next(iter(objectives)),
        objective_version=next(iter(objective_versions)),
        traces=tuple(traces),
        advantages=tuple(advantages),
        group_ids=tuple(group_ids),
        kl_penalty_coef=kl_penalty_coef,
    )
    result = gradient_step(request)
    if not isinstance(result, GradientStepResult):
        raise TrainerError("gradient_step must return GradientStepResult")

    # Claim every contributing group against the *pre-update* snapshot before
    # advancing, so a re-run of the same barrier can never silently retrain.
    claimed = registry
    for group in groups:
        claimed = claimed.claim_learning_group(role, group=group, snapshot=expected_snapshot)
    updated = claimed.advance_role(
        role,
        adapter_state=result.adapter_state,
        optimizer_state=result.optimizer_state,
    )
    return LearningUpdate(
        groups=tuple(groups),
        result=result,
        role_snapshot_id_before=expected_snapshot.snapshot_id,
        adapter_hash_before=expected_snapshot.adapter_hash,
        adapter_hash_after=updated.state(role).adapter.adapter_hash,
        registry=updated,
    )


def train_barrier(
    groups: Sequence[LearningGroup],
    *,
    traces_by_id: Mapping[str, PolicyTrace],
    registry: RoleRegistry,
    epoch: int,
    gradient_step: GradientStepFn,
    kl_penalty_coef: float = 0.0,
    skip_zero_advantage: bool = True,
) -> Tuple[Tuple[LearningUpdate, ...], RoleRegistry]:
    """Apply at most one gradient step per role for this barrier's groups."""

    by_role: Dict[Role, List[LearningGroup]] = {}
    for group in groups:
        by_role.setdefault(group.role, []).append(group)

    updates: List[LearningUpdate] = []
    current = registry
    for role in current.roles:
        role_groups = by_role.get(role, [])
        update = train_role_groups(
            role,
            role_groups,
            traces_by_id=traces_by_id,
            registry=current,
            epoch=epoch,
            gradient_step=gradient_step,
            kl_penalty_coef=kl_penalty_coef,
            skip_zero_advantage=skip_zero_advantage,
        )
        if update is not None:
            updates.append(update)
            current = update.registry
    return tuple(updates), current


__all__ = [
    "GradientStepFn",
    "GradientStepRequest",
    "GradientStepResult",
    "LearningUpdate",
    "TrainerError",
    "train_barrier",
    "train_role_groups",
]
