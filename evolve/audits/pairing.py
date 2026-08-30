"""Randomized, preassigned matched-option audit pairs.

An audit pair compares an intervention option against a matched control
continuation from the exact same frozen start, cell, role checkpoint,
harness, verifier, horizon, resources, generation settings, and seed --
"common randomness where valid" per AGENTS.md.  Assignment probability and
the pair spec are persisted before either branch executes, matching
:class:`~evolve.types.AuditPair`'s own ``preassigned`` invariant.
"""

from __future__ import annotations

from typing import Any, Mapping, Tuple

from evolve.ids import content_id, derive_seed
from evolve.types import AuditPair, AuditStatus, BranchSpec, Channel


class AuditPairingError(ValueError):
    """Two branches cannot form a valid matched audit pair."""


def _matched_invariants(branch: BranchSpec) -> Mapping[str, Any]:
    return {
        "epoch": branch.epoch,
        "start_state_id": branch.start_state_id,
        "frozen_record_threshold": branch.frozen_record_threshold,
        "role_snapshot_id": branch.role_snapshot_id,
        "harness_id": branch.harness_id,
        "harness_version": branch.harness_version,
        "verifier_id": branch.verifier_id,
        "verifier_version": branch.verifier_version,
        "memory_view_id": branch.memory_view_id,
        "memory_view_hash": branch.memory_view_hash,
        "horizon": branch.horizon,
        "budget": dict(branch.budget),
        "generation_settings": dict(branch.generation_settings),
        "seed": branch.seed,
        "channel": branch.channel.value,
    }


def assign_audit_sides(
    *, option_a: str, option_b: str, seed: int
) -> Tuple[str, str, float]:
    """Freeze treatment semantics and its randomized assignment propensity.

    Returns ``(intervention_option_id, control_option_id, assignment_probability)``.
    The caller uses ``seed`` to randomize the treatment's opaque execution
    slot. The registered continuation remains the control so effect signs and
    causal-memory intervention IDs cannot flip merely because labels swapped.
    """

    if option_a == option_b:
        raise AuditPairingError("an audit needs two distinct options to compare")
    derive_seed("audit_side_assignment", seed, option_a, option_b)
    return option_a, option_b, 0.5


def create_audit_pair(
    *,
    run_id: str,
    cell_id: str,
    intervention_branch: BranchSpec,
    control_branch: BranchSpec,
    assignment_probability: float,
    assignment_seed: int,
) -> AuditPair:
    """Persist one preassigned matched pair before either branch executes."""

    eligible_channels = (Channel.AUDIT, Channel.REFINEMENT)
    if intervention_branch.channel not in eligible_channels or control_branch.channel not in eligible_channels:
        raise AuditPairingError(
            "matched audit branches must use the audit or refinement-audit channel"
        )
    if intervention_branch.branch_id == control_branch.branch_id:
        raise AuditPairingError("audit sides need distinct preassigned branches")
    if intervention_branch.option_id == control_branch.option_id:
        raise AuditPairingError("intervention and matched continuation must differ")

    left = _matched_invariants(intervention_branch)
    right = _matched_invariants(control_branch)
    if left != right:
        differing = sorted(key for key in left if left[key] != right[key])
        raise AuditPairingError(
            "matched audit differs outside its intervention option: " + ", ".join(differing)
        )

    identity = {
        "run_id": run_id,
        "cell_id": cell_id,
        **left,
        "intervention_option_id": intervention_branch.option_id,
        "control_option_id": control_branch.option_id,
        "assignment_probability": float(assignment_probability),
        "assignment_seed": int(assignment_seed),
        "intervention_branch_id": intervention_branch.branch_id,
        "control_branch_id": control_branch.branch_id,
    }
    audit_id = content_id("audit_pair", identity)
    return AuditPair(
        audit_id=audit_id,
        run_id=run_id,
        epoch=intervention_branch.epoch,
        start_state_id=intervention_branch.start_state_id,
        cell_id=cell_id,
        frozen_record_threshold=intervention_branch.frozen_record_threshold,
        role_snapshot_id=intervention_branch.role_snapshot_id,
        harness_id=intervention_branch.harness_id,
        verifier_id=intervention_branch.verifier_id,
        horizon=intervention_branch.horizon,
        resources=dict(intervention_branch.budget),
        generation_settings=dict(intervention_branch.generation_settings),
        intervention_option_id=intervention_branch.option_id,
        control_option_id=control_branch.option_id,
        assignment_probability=float(assignment_probability),
        assignment_seed=int(assignment_seed),
        intervention_branch_id=intervention_branch.branch_id,
        control_branch_id=control_branch.branch_id,
        status=AuditStatus.PREASSIGNED,
        preassigned=True,
    )


__all__ = ["AuditPairingError", "assign_audit_sides", "create_audit_pair"]
