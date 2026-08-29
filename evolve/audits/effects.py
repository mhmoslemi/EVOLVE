"""Closing matched audit pairs and computing their normalized effect.

The pair is closed only once both sides hold a scheduler-eligible, non
infrastructure-aborted :class:`~evolve.types.BranchOutcome`.  The effect
itself is the difference of each side's already problem-normalized gain
(``normalize_gain(reward, threshold)`` from the problem hooks); this module
stays problem-agnostic and only combines two supplied gain values, mirroring
:mod:`evolve.harness.registry`'s harness-trial pattern.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, replace

from evolve.types import AuditPair, AuditStatus, BranchOutcome


class AuditEffectError(ValueError):
    """An audit pair cannot be closed or its effect cannot be computed yet."""


def default_gain(outcome: BranchOutcome, *, frozen_record_threshold: float) -> float:
    """Generic non-negative gain fallback: how far the branch's max cleared the threshold."""

    if outcome.maximum_reward is None:
        return 0.0
    return max(0.0, float(outcome.maximum_reward) - float(frozen_record_threshold))


def close_audit_pair(
    pair: AuditPair,
    *,
    intervention_outcome: BranchOutcome,
    control_outcome: BranchOutcome,
) -> AuditPair:
    if pair.status not in (AuditStatus.PREASSIGNED, AuditStatus.RUNNING):
        raise AuditEffectError(f"audit pair {pair.audit_id} is already {pair.status.value}")
    if intervention_outcome.branch_id != pair.intervention_branch_id:
        raise AuditEffectError("intervention outcome does not belong to this pair")
    if control_outcome.branch_id != pair.control_branch_id:
        raise AuditEffectError("control outcome does not belong to this pair")
    for label, outcome in (("intervention", intervention_outcome), ("control", control_outcome)):
        if not outcome.eligible_for_scheduler:
            raise AuditEffectError(
                f"{label} side is not a closed, non-infrastructure-aborted outcome"
            )
    updated = replace(
        pair,
        status=AuditStatus.CLOSED,
        intervention_outcome_id=intervention_outcome.outcome_id,
        control_outcome_id=control_outcome.outcome_id,
    )
    object.__setattr__(updated, "schema_version", pair.schema_version)
    object.__setattr__(updated, "extensions", pair.extensions)
    return updated


@dataclass(frozen=True)
class AuditEffect:
    """One closed pair's normalized, problem-defined causal effect."""

    audit_id: str
    intervention_gain: float
    control_gain: float
    effect: float


def compute_audit_effect(
    pair: AuditPair,
    *,
    intervention_gain: float,
    control_gain: float,
) -> AuditEffect:
    if pair.status != AuditStatus.CLOSED:
        raise AuditEffectError("effect computation requires a closed audit pair")
    for value, name in ((intervention_gain, "intervention_gain"), (control_gain, "control_gain")):
        if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value):
            raise AuditEffectError(f"{name} must be a finite number")
    return AuditEffect(
        audit_id=pair.audit_id,
        intervention_gain=float(intervention_gain),
        control_gain=float(control_gain),
        effect=float(intervention_gain) - float(control_gain),
    )


__all__ = ["AuditEffect", "AuditEffectError", "close_audit_pair", "compute_audit_effect", "default_gain"]
