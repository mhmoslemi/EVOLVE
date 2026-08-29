"""Homogeneous on-policy learning-group construction.

Members sharing role, policy snapshot, start cell, context, option, harness,
horizon, cost class, generation settings, frozen threshold, and channel (and,
within the audit or refinement channel, one audit side or attempt) are
bucketed together and chunked to at most ``group_k`` members; everything
else is rejected by construction rather than mixed, matching
:class:`~evolve.types.LearningGroup`'s own homogeneity invariant.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Mapping, Optional, Sequence, Tuple

from evolve.ids import content_hash, content_id
from evolve.types import (
    AllocationArm,
    AuditSide,
    BranchOutcome,
    BranchSpec,
    Channel,
    LearningGroup,
    LearningObjective,
    PolicyTrace,
)

from .objectives import advantages_for_objective, objective_version as objective_version_for


class LearningGroupError(ValueError):
    """A set of branch observations cannot form a valid learning group."""


@dataclass(frozen=True)
class GroupMember:
    """One closed, admitted-or-not branch with its captured policy trace."""

    branch: BranchSpec
    outcome: BranchOutcome
    trace: PolicyTrace


def context_id_for(*, cell_id: str, memory_view_hash: str) -> str:
    return content_id("context", {"cell_id": cell_id, "memory_view_hash": memory_view_hash})


def _bucket_key(
    member: GroupMember,
    *,
    arm: AllocationArm,
    audit_side: Optional[AuditSide],
    refinement_attempt: Optional[int],
) -> Tuple:
    branch = member.branch
    return (
        arm.role.value,
        branch.role_snapshot_id,
        arm.cell_id,
        branch.memory_view_hash,
        branch.option_id,
        branch.harness_id,
        branch.horizon,
        arm.cost_class,
        content_hash(branch.generation_settings),
        branch.frozen_record_threshold,
        branch.channel.value,
        audit_side.value if audit_side is not None else None,
        refinement_attempt,
    )


def _chunks(members: Sequence[GroupMember], size: int) -> List[List[GroupMember]]:
    return [list(members[index : index + size]) for index in range(0, len(members), size)]


def _group_id_for(
    *,
    role: str,
    policy_snapshot_id: str,
    start_cell_id: str,
    context_id: str,
    option_id: str,
    harness_id: str,
    branch_ids: Sequence[str],
) -> str:
    return content_id(
        "learning_group",
        {
            "role": role,
            "policy_snapshot_id": policy_snapshot_id,
            "start_cell_id": start_cell_id,
            "context_id": context_id,
            "option_id": option_id,
            "harness_id": harness_id,
            "branch_ids": sorted(branch_ids),
        },
    )


def build_learning_groups(
    members: Sequence[GroupMember],
    *,
    arms: Mapping[str, AllocationArm],
    objective: LearningObjective,
    top_m: int,
    group_k: int,
    audit_sides: Mapping[str, AuditSide] = {},
    refinement_attempts: Mapping[str, int] = {},
) -> Tuple[LearningGroup, ...]:
    """Partition members into maximal homogeneous chunks and score each."""

    buckets: Dict[Tuple, List[GroupMember]] = {}
    for member in members:
        branch = member.branch
        arm = arms.get(branch.arm_id)
        if arm is None:
            raise LearningGroupError(f"no allocation arm supplied for branch {branch.branch_id}")
        audit_side = audit_sides.get(branch.branch_id)
        refinement_attempt = refinement_attempts.get(branch.branch_id)
        if branch.channel == Channel.AUDIT and audit_side is None:
            raise LearningGroupError(f"audit-channel branch {branch.branch_id} needs its audit_side")
        if branch.channel == Channel.REFINEMENT and refinement_attempt is None:
            raise LearningGroupError(
                f"refinement-channel branch {branch.branch_id} needs its refinement_attempt"
            )
        key = _bucket_key(member, arm=arm, audit_side=audit_side, refinement_attempt=refinement_attempt)
        buckets.setdefault(key, []).append(member)

    version = objective_version_for(objective)
    groups: List[LearningGroup] = []
    for bucket in buckets.values():
        for chunk in _chunks(bucket, group_k):
            groups.append(
                _build_one_group(
                    chunk,
                    arms=arms,
                    objective=objective,
                    objective_version=version,
                    top_m=min(top_m, len(chunk)),
                    audit_sides=audit_sides,
                    refinement_attempts=refinement_attempts,
                )
            )
    return tuple(groups)


def _build_one_group(
    chunk: Sequence[GroupMember],
    *,
    arms: Mapping[str, AllocationArm],
    objective: LearningObjective,
    objective_version: str,
    top_m: int,
    audit_sides: Mapping[str, AuditSide],
    refinement_attempts: Mapping[str, int],
) -> LearningGroup:
    first = chunk[0]
    arm = arms[first.branch.arm_id]
    for member in chunk:
        if member.trace.branch_id != member.branch.branch_id:
            raise LearningGroupError("policy trace does not belong to its branch")
        if member.outcome.branch_id != member.branch.branch_id:
            raise LearningGroupError("branch outcome does not belong to its branch")

    rewards = [member.outcome.maximum_reward for member in chunk]
    advantages = advantages_for_objective(rewards, objective=objective, top_m=top_m)

    branch_ids = tuple(member.branch.branch_id for member in chunk)
    trace_ids = tuple(member.trace.trace_id for member in chunk)
    outcome_ids = tuple(member.outcome.outcome_id for member in chunk)
    context_id = context_id_for(cell_id=arm.cell_id, memory_view_hash=first.branch.memory_view_hash)
    audit_side = audit_sides.get(first.branch.branch_id) if first.branch.channel == Channel.AUDIT else None
    refinement_attempt = (
        refinement_attempts.get(first.branch.branch_id)
        if first.branch.channel == Channel.REFINEMENT
        else None
    )
    group_id = _group_id_for(
        role=arm.role.value,
        policy_snapshot_id=first.branch.role_snapshot_id,
        start_cell_id=arm.cell_id,
        context_id=context_id,
        option_id=first.branch.option_id,
        harness_id=first.branch.harness_id,
        branch_ids=branch_ids,
    )
    return LearningGroup(
        group_id=group_id,
        role=arm.role,
        policy_snapshot_id=first.branch.role_snapshot_id,
        start_cell_id=arm.cell_id,
        context_id=context_id,
        option_id=first.branch.option_id,
        harness_id=first.branch.harness_id,
        horizon=first.branch.horizon,
        cost_class=arm.cost_class,
        generation_settings=dict(first.branch.generation_settings),
        frozen_record_threshold=first.branch.frozen_record_threshold,
        channel=first.branch.channel,
        branch_ids=branch_ids,
        trace_ids=trace_ids,
        outcome_ids=outcome_ids,
        advantages=advantages,
        objective=objective,
        objective_version=objective_version,
        top_m=top_m,
        audit_side=audit_side,
        refinement_attempt=refinement_attempt,
    )


__all__ = [
    "GroupMember",
    "LearningGroupError",
    "build_learning_groups",
    "context_id_for",
]
