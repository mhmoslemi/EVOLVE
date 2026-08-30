"""Composed EVOLVE engine: fresh/resume run lifecycle and epoch barriers.

``EvolveEngine`` ties every subsystem built in phases 4-8 together into the
epoch-barrier lifecycle AGENTS.md's "Synchronization" section requires:
freeze an :class:`~evolve.types.EpochManifest`, stream branch execution,
then atomically commit evidence, archive, scheduler, memory, learning,
harness, and checkpoint/report state together.  The actual model-touching
work (branch generation+verification, and the role gradient step) is
injected via :class:`EngineWorkers` -- this module never loads a model or
launches a job itself; it is the pure composition and persistence root
around those calls, mirroring the dependency-injection boundary already used
by :mod:`evolve.verifier.service`, :mod:`evolve.options.branch`, and
:mod:`evolve.learning.trainer`.

The production worker boundary loads one HF backbone with three explicitly
named role adapters only after configuration and initial run metadata are
durable. Dry planning and configuration validation remain model-free.
"""

from __future__ import annotations

import hashlib
import json
import math
import os
import platform
import re
import shutil
import socket
import subprocess
import sys
from concurrent.futures import as_completed
from dataclasses import dataclass, field, replace
from pathlib import Path
from typing import Any, Callable, Dict, List, Mapping, Optional, Sequence, Tuple, Union

from evolve.archive import (
    ArchiveAdmissionError,
    ConfirmedRecordTracker,
    ProvenanceStore,
    ScientificArchive,
    validate_stored_evidence,
)
from evolve.audits import (
    AuditEffectError,
    AuditPairingError,
    abort_audit_pair,
    assign_audit_sides,
    close_audit_pair,
    compute_audit_effect,
    create_audit_pair,
)
from evolve.budget import BudgetOverrun, BudgetService
from evolve.causal_memory import (
    MemoryStore,
    add_effect,
    evaluate_promotion,
    memory_id_for,
    new_memory_record,
    stratify_drift,
)
from evolve.config import EvolveConfig
from evolve.harness import (
    HarnessPromotionError,
    HarnessRegistry,
    HarnessTrialRecord,
    MatchedHarnessAuditContext,
    default_harness_registry,
)
from evolve.ids import content_hash, content_id, derive_seed, rollout_seed
from evolve.learning import GroupMember, build_learning_groups
from evolve.learning.trainer import GradientStepFn, train_barrier
from evolve.options import (
    DIAGNOSTIC_REPAIR_STATE_MACHINE,
    FRESH_REFINEMENT_CONTROL_STATE_MACHINE,
    OptionRegistry,
    build_option_context,
    execute_branch,
    production_option_registry,
)
from evolve.options.branch import BranchExecution, BranchStepExecutor
from evolve.refinement import (
    NurseryEntry,
    NurseryPolicy,
    expire_entry,
    open_entry,
    record_attempt,
)
from evolve.roles import RoleRegistry
from evolve.runio import (
    ControllerEventWriter,
    ImmutableWriteError,
    RunLayout,
    atomic_write_bytes,
    atomic_write_json,
    atomic_write_text,
    append_jsonl_records,
    create_fresh_run_layout,
    fsync_directory,
    open_existing_run_layout,
    write_immutable_json,
    write_immutable_text,
    write_initial_run_metadata,
    write_resume_run_metadata,
)
from evolve.scheduler import (
    AllocationPlan,
    ArmIdentity,
    PORTFOLIO_VERSION,
    POSTERIOR_VERSION,
    PlannedArm,
    PosteriorStore,
    RESERVATIONS_VERSION,
    ReservationSlots,
    SchedulerError,
    enumerate_candidate_arms,
    make_allocation_arm,
    plan_epoch,
)
from evolve.types import (
    AllocationArm,
    AuditPair,
    AuditSide,
    AuditStatus,
    BranchSpec,
    BudgetLedger,
    Channel,
    Descriptor,
    EpochManifest,
    EvidencePacket,
    FailureKind,
    Proposal,
    Role,
    VerifiedScientificState,
)
from evolve.verifier.adapters import ProblemScientificAdapter
from evolve.verifier.models import VerificationPolicy


class EngineError(RuntimeError):
    """The composed engine cannot proceed as configured."""


class RecordConfirmationInfrastructureError(EngineError):
    """Both bounded record-confirmation calls failed in infrastructure."""


class _RunAttachInterrupted(Exception):
    """Bootstrap was drained and recorded before model-worker construction."""


def _file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _write_json_once(path: Path, value: Any) -> None:
    try:
        write_immutable_json(path, value)
    except ImmutableWriteError:
        if json.loads(path.read_text(encoding="utf-8")) != value:
            raise EngineError(f"immutable JSON artifact conflict: {path}")


def _json_native_answer_payload(state: VerifiedScientificState) -> Any:
    """Project a frozen scientific answer through its durable schema."""

    return state.to_dict()["answer_payload"]


def _write_text_once(path: Path, value: str) -> None:
    try:
        write_immutable_text(path, value)
    except ImmutableWriteError:
        if path.read_text(encoding="utf-8") != value:
            raise EngineError(f"immutable text artifact conflict: {path}")


def _confirmation_attempt_count(value: Any) -> int:
    """Strictly validate the bounded record-confirmation attempt count."""

    if isinstance(value, bool) or not isinstance(value, int) or value not in (1, 2):
        raise EngineError("record confirmation attempts must be integer 1 or 2")
    return value


def _verification_attempt_index(value: Any, *, maximum: Optional[int] = None) -> int:
    """Reject coerced or out-of-range verifier attempt metadata."""

    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        raise EngineError("verification attempt index must be a nonnegative integer")
    if maximum is not None and value > maximum:
        raise EngineError("verification attempt index exceeds the retry policy")
    return value


def _validate_attempt_count_covers_evidence(
    attempt_flag: Any, attempts: int, *, context: str
) -> None:
    """Ensure total calls include the final durable evidence-producing call.

    A later retry can fail before producing evidence, so the final packet's
    attempt index need not equal the total number of charged calls.  It may
    never be greater, however.
    """

    attempts = _confirmation_attempt_count(attempts)
    if attempt_flag is None:
        return
    evidence_attempt = _verification_attempt_index(attempt_flag, maximum=1) + 1
    if evidence_attempt > attempts:
        raise EngineError(f"{context} attempt count contradicts its evidence")


def _validate_proposal_evidence_binding(
    proposal: Proposal, evidence: EvidencePacket
) -> None:
    """Reject a durable packet that is not the exact proposal observation."""

    try:
        validate_stored_evidence(evidence)
    except Exception as exc:
        raise EngineError(
            f"invalid durable evidence {evidence.evidence_id}: {exc}"
        ) from exc
    expected = {
        "run_id": proposal.run_id,
        "proposal_id": proposal.proposal_id,
        "problem_id": proposal.problem_id,
        "parent_state_id": proposal.parent_state_id,
        "branch_id": proposal.branch_id,
        "source_hash": proposal.source_hash,
    }
    for field_name, value in expected.items():
        if getattr(evidence, field_name) != value:
            raise EngineError(
                f"durable evidence {evidence.evidence_id} has a mismatched "
                f"{field_name} reference"
            )


def _validate_confirmation_binding(
    proposal: Proposal,
    prior_evidence: EvidencePacket,
    confirmation_evidence: EvidencePacket,
) -> None:
    """Validate that a replayed packet confirms precisely the saved payload."""

    _validate_proposal_evidence_binding(proposal, confirmation_evidence)
    expected = {
        "verifier_id": prior_evidence.verifier_id,
        "verifier_version": prior_evidence.verifier_version,
        "harness_id": prior_evidence.harness_id,
        "policy_snapshot_id": prior_evidence.policy_snapshot_id,
        "timeout_is_scientific": prior_evidence.timeout_is_scientific,
    }
    for field_name, value in expected.items():
        if getattr(confirmation_evidence, field_name) != value:
            raise EngineError(
                f"confirmation {confirmation_evidence.evidence_id} has a "
                f"mismatched {field_name} reference"
            )
    if (
        confirmation_evidence.flags.get("confirmation_of_evidence_id")
        != prior_evidence.evidence_id
    ):
        raise EngineError("confirmation references a different source evidence packet")
    if (
        confirmation_evidence.flags.get("confirmation_target_state_id")
        != prior_evidence.scientific_state_id
    ):
        raise EngineError("confirmation references a different scientific state")
    if content_hash(confirmation_evidence.answer_payload) != content_hash(
        prior_evidence.answer_payload
    ):
        raise EngineError("confirmation payload differs from the saved answer payload")
    if confirmation_evidence.admitted:
        if confirmation_evidence.scientific_state_id != prior_evidence.scientific_state_id:
            raise EngineError("confirmation admitted a different scientific state")
        if confirmation_evidence.descriptor_id != prior_evidence.descriptor_id:
            raise EngineError("confirmation changed the scientific descriptor")
        if confirmation_evidence.fingerprint != prior_evidence.fingerprint:
            raise EngineError("confirmation changed the scientific fingerprint")


def _restore_admitted_artifacts(
    evidence: EvidencePacket,
    *,
    adapter: ProblemScientificAdapter,
) -> Tuple[VerifiedScientificState, Descriptor]:
    """Rebuild derived state/descriptor files without rerunning verification."""

    if not evidence.admitted or evidence.scientific_state_id is None:
        raise EngineError("cannot restore scientific artifacts from rejected evidence")

    from evolve.verifier.evidence import build_descriptor
    from evolve.verifier.models import ExecutionCapture, VerificationDecision

    restored_decision = VerificationDecision(
        failure_kind=evidence.failure_kind,
        resolved=evidence.resolved,
        admitted=evidence.admitted,
        internal_reward=evidence.internal_reward,
        raw_score=evidence.raw_score,
        uncertainty=evidence.uncertainty,
        flags=evidence.flags,
        scores=evidence.scores,
        capture=ExecutionCapture(
            diagnostics=evidence.diagnostics,
            resources=evidence.resources,
            started_at=evidence.started_at,
            completed_at=evidence.completed_at,
            attempt_index=_verification_attempt_index(
                evidence.flags.get("verification_attempt_index", 0)
            ),
        ),
    )
    descriptor = build_descriptor(
        problem_id=evidence.problem_id,
        function_version=adapter.descriptor_version,
        dimensions=adapter.describe_scientific_state(
            evidence.answer_payload, restored_decision
        ),
        method_complete=adapter.method_complete,
    )
    if descriptor.descriptor_id != evidence.descriptor_id:
        raise EngineError("scientific descriptor changed while restoring durable evidence")
    fingerprint = adapter.scientific_fingerprint(
        evidence.answer_payload, restored_decision
    )
    if fingerprint != evidence.fingerprint:
        raise EngineError("scientific fingerprint changed while restoring durable evidence")
    state = VerifiedScientificState(
        state_id=evidence.scientific_state_id,
        proposal_id=evidence.proposal_id,
        evidence_id=evidence.evidence_id,
        problem_id=evidence.problem_id,
        answer_payload=evidence.answer_payload,
        resolved=evidence.resolved,
        admitted=evidence.admitted,
        confirmed=evidence.confirmed,
        internal_reward=evidence.internal_reward,
        raw_score=evidence.raw_score,
        descriptor_id=evidence.descriptor_id,
        fingerprint=evidence.fingerprint,
    )
    return state, descriptor


def _normalized_outcome_gain(
    adapter: ProblemScientificAdapter,
    outcome: Any,
    *,
    frozen_record_threshold: float,
) -> float:
    """Apply the problem's frozen-threshold gain normalization exactly once."""

    if outcome.maximum_reward is None or outcome.infrastructure_aborted:
        return 0.0
    value = adapter.problem.normalize_gain(
        float(outcome.maximum_reward), float(frozen_record_threshold)
    )
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise EngineError("problem normalize_gain must return a numeric value")
    return max(0.0, float(value))


def _settle_branch_verifier_budget(
    ledger: BudgetLedger,
    *,
    execution: BranchExecution,
    debit_key: str,
    reserved: float,
) -> BudgetLedger:
    """Reconcile one frozen reservation with durable verifier attempts.

    New plans reserve the bounded retry ceiling up front.  A partially
    executed plan written by an older controller can reserve less, however,
    while its already-durable evidence records more than one infrastructure
    attempt.  Account for that evidence exactly instead of trusting a derived
    ``unused_budget`` value or silently losing the retry from the global
    ledger.
    """

    actual = float(execution.outcome.costs.get("verifier_calls", 0.0))
    reserved = float(reserved)
    if not math.isfinite(actual) or actual < 0.0:
        raise EngineError("branch verifier cost must be finite and nonnegative")
    if not math.isfinite(reserved) or reserved <= 0.0:
        raise EngineError("branch verifier reservation must be finite and positive")
    if actual > reserved + 1e-12:
        try:
            return BudgetService.debit(
                ledger,
                resource="verifier_calls",
                amount=actual - reserved,
                transaction_key=f"{debit_key}:actual-overflow",
            )
        except BudgetOverrun as exc:
            raise EngineError(
                "durable verifier attempts exceed the frozen reservation and "
                "remaining global verifier budget"
            ) from exc
    if reserved > actual + 1e-12:
        return BudgetService.refund(
            ledger,
            resource="verifier_calls",
            amount=reserved - actual,
            transaction_key=f"{debit_key}:refund",
            debit_transaction_key=debit_key,
        )
    return ledger


COMPONENT_SCHEMA_VERSIONS: Mapping[str, int] = {
    "options": 1,
    "harness_registry": 1,
    "scheduler_posterior": 1,
    "scheduler_portfolio": 1,
    "scheduler_reservations": 1,
    "audits": 1,
    "causal_memory": 1,
    "learning_objective": 1,
    "refinement_nursery": 1,
}


@dataclass(frozen=True)
class EngineWorkers:
    """The two injected, model-touching callables the engine never owns."""

    branch_step_executor: BranchStepExecutor
    gradient_step: GradientStepFn
    begin_epoch: Optional[Callable[[Any], None]] = None
    persist_roles: Optional[Callable[[Any], Mapping[str, Any]]] = None
    persist_training_state: Optional[
        Callable[[Any, Mapping[str, Any], Sequence[Path]], None]
    ] = None
    submit_branch: Optional[Callable[[Callable[[], BranchExecution]], Any]] = None
    shutdown: Optional[Callable[[], None]] = None


def build_production_workers(
    config: EvolveConfig,
    *,
    adapter: ProblemScientificAdapter,
    layout: RunLayout,
    state: "EpochState",
) -> EngineWorkers:
    """Wire a live backend for real branch generation and role training.

    Model loading happens only after configuration, metadata, and the initial
    scientific archive are durable. The returned callbacks retain one backbone
    and enforce explicit named-adapter activation for every role operation.
    """

    from evolve.workers.runtime import LiveEvolveRuntime

    runtime = LiveEvolveRuntime(
        config=config, adapter=adapter, layout=layout, state=state
    )
    return EngineWorkers(
        branch_step_executor=runtime.branch_step,
        gradient_step=runtime.gradient_step,
        begin_epoch=runtime.begin_epoch,
        persist_roles=runtime.persist_roles,
        persist_training_state=runtime.persist_training_state,
        submit_branch=runtime.submit_branch,
        shutdown=runtime.shutdown,
    )


@dataclass(frozen=True)
class _PendingBranch:
    branch: BranchSpec
    arm: AllocationArm
    role_snapshot: Any
    ordinal: int
    debit_key: str
    cell_empty: bool = False
    memory_enabled: bool = True


@dataclass(frozen=True)
class _RefinementSource:
    arm: AllocationArm
    proposal: Proposal
    evidence: EvidencePacket
    branch_id: str
    entry: Optional[NurseryEntry] = None


# --------------------------------------------------------------------------
# Snapshot identity helpers
# --------------------------------------------------------------------------


def _archive_snapshot(archive: ScientificArchive) -> Tuple[str, str]:
    payload = {
        "cell_map_version": archive.cell_map_version,
        "descriptors": [d.to_dict() for d in sorted(archive.descriptors, key=lambda d: d.descriptor_id)],
        "cells": [c.to_dict() for c in sorted(archive.cells, key=lambda c: c.cell_id)],
    }
    return content_id("archive_snapshot", payload), content_hash(payload)


def _posterior_snapshot_id(posterior: PosteriorStore) -> str:
    return content_id("scheduler_snapshot", posterior.to_dict())


def _memory_snapshot_id(memory: MemoryStore) -> str:
    payload = {
        memory_id: record.to_dict() for memory_id, record in sorted(memory.records.items())
    }
    return content_id("causal_memory_snapshot", payload)


def _allocation_plan_id(plan: AllocationPlan) -> str:
    payload = {
        "epoch": plan.epoch,
        "seed": plan.seed,
        "arms": [
            {
                "arm_id": planned.arm.arm_id,
                "reservation": planned.reservation,
                "reservations": list(planned.reservations),
                "replicas": planned.replicas,
                "rng_seed": planned.rng_seed,
            }
            for planned in plan.planned_arms
        ],
    }
    return content_id("allocation_plan", payload)


def _plan_document(plan: AllocationPlan) -> Mapping[str, Any]:
    return {
        "schema_version": 1,
        "epoch": plan.epoch,
        "seed": plan.seed,
        "posterior_version": plan.posterior_version,
        "portfolio_version": plan.portfolio_version,
        "reservations_version": plan.reservations_version,
        "reservation_slots": {
            "total_inflight": plan.reservation_slots.total_inflight,
            "audit_branch_slots": plan.reservation_slots.audit_branch_slots,
            "no_memory_audit_slots": plan.reservation_slots.no_memory_audit_slots,
            "refinement_slots": plan.reservation_slots.refinement_slots,
            "harness_trial_slots": plan.reservation_slots.harness_trial_slots,
            "empty_cell_slots": plan.reservation_slots.empty_cell_slots,
            "global_exploration_slots": plan.reservation_slots.global_exploration_slots,
            "role_guaranteed_slots": plan.reservation_slots.role_guaranteed_slots,
            "remaining_production_slots": plan.reservation_slots.remaining_production_slots,
        },
        "planned_arms": [
            {
                "arm": planned.arm.to_dict(),
                "reservation": planned.reservation,
                "posterior_level": planned.posterior_level,
                "expected_gain": planned.expected_gain,
                "uncertainty": planned.uncertainty,
                "posterior_support": planned.posterior_support,
                "reliability_probability": planned.reliability_probability,
                "admission_probability": planned.admission_probability,
                "improvement_probability_given_admission": (
                    planned.improvement_probability_given_admission
                ),
                "mean_positive_gain": planned.mean_positive_gain,
                "marginal_gain": planned.marginal_gain,
                "correlation_penalty": planned.correlation_penalty,
                "predicted_cost_uncertainty": dict(
                    planned.predicted_cost_uncertainty
                ),
                "replicas": planned.replicas,
                "reservations": list(planned.reservations),
                "rng_seed": planned.rng_seed,
            }
            for planned in plan.planned_arms
        ],
    }


def _plan_from_document(document: Mapping[str, Any]) -> AllocationPlan:
    """Strictly restore the authoritative in-epoch allocation decision."""

    if not isinstance(document, Mapping):
        raise EngineError("allocation plan must be a JSON object")
    schema_version = document.get("schema_version", 1)
    if (
        isinstance(schema_version, bool)
        or not isinstance(schema_version, int)
        or schema_version != 1
    ):
        raise EngineError(
            f"unsupported allocation plan schema {schema_version!r}"
        )
    slots_document = document.get("reservation_slots")
    planned_documents = document.get("planned_arms")
    if not isinstance(slots_document, Mapping) or not isinstance(
        planned_documents, list
    ):
        raise EngineError("allocation plan omits reservations or planned arms")

    def plan_number(value: Any, name: str, *, minimum: float = 0.0) -> float:
        if (
            isinstance(value, bool)
            or not isinstance(value, (int, float))
            or not math.isfinite(float(value))
            or float(value) < minimum
        ):
            raise EngineError(f"allocation plan {name} is invalid")
        return float(value)

    def plan_integer(value: Any, name: str, *, minimum: int = 0) -> int:
        if (
            isinstance(value, bool)
            or not isinstance(value, int)
            or value < minimum
        ):
            raise EngineError(f"allocation plan {name} is invalid")
        return value

    def plan_string(value: Any, name: str) -> str:
        if not isinstance(value, str) or not value.strip():
            raise EngineError(f"allocation plan {name} is invalid")
        return value
    try:
        slots = ReservationSlots(
            **{
                name: slots_document[name]
                for name in (
                    "total_inflight",
                    "audit_branch_slots",
                    "no_memory_audit_slots",
                    "refinement_slots",
                    "harness_trial_slots",
                    "empty_cell_slots",
                    "global_exploration_slots",
                    "role_guaranteed_slots",
                    "remaining_production_slots",
                )
            }
        )
        planned = []
        for item in planned_documents:
            if not isinstance(item, Mapping):
                raise EngineError("planned arm entry must be a JSON object")
            arm = AllocationArm.from_dict(item["arm"])
            rebuilt = make_allocation_arm(
                ArmIdentity.from_arm(arm),
                channel=arm.channel,
                expected_cost=arm.expected_cost,
                hard_cost=arm.hard_cost,
            )
            if rebuilt.arm_id != arm.arm_id:
                raise EngineError("persisted allocation arm ID is not canonical")
            reservation = item.get("reservation")
            if reservation is not None:
                reservation = plan_string(reservation, "reservation")
            reservation_values = item.get("reservations", ())
            if not isinstance(reservation_values, (list, tuple)) or any(
                not isinstance(value, str) or not value.strip()
                for value in reservation_values
            ):
                raise EngineError("allocation plan reservations are invalid")
            reservation_values = tuple(reservation_values)
            if len(set(reservation_values)) != len(reservation_values):
                raise EngineError("allocation plan reservation labels must be unique")
            if reservation_values and reservation != reservation_values[0]:
                raise EngineError(
                    "allocation plan primary reservation does not match its labels"
                )
            uncertainty_document = item.get("predicted_cost_uncertainty", {})
            if not isinstance(uncertainty_document, Mapping):
                raise EngineError(
                    "allocation plan predicted cost uncertainty must be a mapping"
                )
            cost_uncertainty = {
                plan_string(resource, "resource uncertainty key"): plan_number(
                    amount, f"predicted_cost_uncertainty.{resource}"
                )
                for resource, amount in uncertainty_document.items()
            }
            reliability = plan_number(
                item.get("reliability_probability", 0.5),
                "reliability_probability",
            )
            admission = plan_number(
                item.get("admission_probability", 0.5),
                "admission_probability",
            )
            improvement = plan_number(
                item.get("improvement_probability_given_admission", 0.5),
                "improvement_probability_given_admission",
            )
            if any(value > 1.0 for value in (reliability, admission, improvement)):
                raise EngineError("allocation plan probabilities must lie in [0, 1]")
            planned.append(
                PlannedArm(
                    arm=arm,
                    reservation=reservation,
                    posterior_level=plan_string(
                        item["posterior_level"], "posterior_level"
                    ),
                    expected_gain=plan_number(item["expected_gain"], "expected_gain"),
                    uncertainty=plan_number(item["uncertainty"], "uncertainty"),
                    marginal_gain=plan_number(item["marginal_gain"], "marginal_gain"),
                    rng_seed=plan_integer(item["rng_seed"], "rng_seed"),
                    posterior_support=plan_integer(
                        item.get("posterior_support", 0), "posterior_support"
                    ),
                    reliability_probability=reliability,
                    admission_probability=admission,
                    improvement_probability_given_admission=improvement,
                    mean_positive_gain=plan_number(
                        item.get("mean_positive_gain", 0.0), "mean_positive_gain"
                    ),
                    replicas=plan_integer(item.get("replicas", 1), "replicas", minimum=1),
                    reservations=reservation_values,
                    correlation_penalty=plan_number(
                        item.get("correlation_penalty", 0.0), "correlation_penalty"
                    ),
                    predicted_cost_uncertainty=cost_uncertainty,
                )
            )
        epoch = document["epoch"]
        seed = document["seed"]
        if (
            isinstance(epoch, bool)
            or not isinstance(epoch, int)
            or epoch < 0
            or isinstance(seed, bool)
            or not isinstance(seed, int)
            or seed < 0
        ):
            raise EngineError("allocation plan epoch and seed must be integers")
        posterior_version = plan_string(
            document["posterior_version"], "posterior_version"
        )
        portfolio_version = plan_string(
            document["portfolio_version"], "portfolio_version"
        )
        reservations_version = plan_string(
            document["reservations_version"], "reservations_version"
        )
        supported_versions = {
            "posterior_version": (posterior_version, POSTERIOR_VERSION),
            "portfolio_version": (portfolio_version, PORTFOLIO_VERSION),
            "reservations_version": (reservations_version, RESERVATIONS_VERSION),
        }
        for name, (persisted, supported) in supported_versions.items():
            if persisted != supported:
                raise EngineError(
                    f"allocation plan {name} {persisted!r} is unsupported; "
                    f"expected {supported!r}"
                )
        arm_ids = [item.arm.arm_id for item in planned]
        if len(set(arm_ids)) != len(arm_ids):
            raise EngineError("allocation plan contains duplicate arm identities")
        production_capacity = max(
            0,
            slots.total_inflight
            - slots.audit_branch_slots
            - slots.refinement_slots
            - slots.harness_trial_slots,
        )
        if sum(item.replicas for item in planned) > production_capacity:
            raise EngineError("allocation plan exceeds its frozen production capacity")
        return AllocationPlan(
            epoch=epoch,
            posterior_version=posterior_version,
            portfolio_version=portfolio_version,
            reservations_version=reservations_version,
            reservation_slots=slots,
            planned_arms=tuple(planned),
            seed=seed,
        )
    except EngineError:
        raise
    except (KeyError, TypeError, ValueError) as exc:
        raise EngineError(f"malformed allocation plan: {exc}") from exc


def _retargeted_arm(
    arm: AllocationArm,
    *,
    option_id: str,
    channel: Channel,
    option_registry: OptionRegistry,
) -> AllocationArm:
    """Rebuild an arm for a different option, keeping its content-addressed identity valid."""

    from evolve.scheduler.arms import ArmIdentity, make_allocation_arm

    if arm.option_id == option_id and arm.channel == channel:
        return arm
    option_registry.spec(option_id)
    identity = ArmIdentity(
        cell_id=arm.cell_id, role=arm.role, option_id=option_id,
        harness_id=arm.harness_id, horizon=arm.horizon, cost_class=arm.cost_class,
    )
    return make_allocation_arm(
        identity,
        channel=channel,
        expected_cost=arm.expected_cost,
        hard_cost=arm.hard_cost,
    )


def _retargeted_harness_arm(
    arm: AllocationArm,
    *,
    harness_id: str,
    channel: Channel,
) -> AllocationArm:
    """Rebuild an arm with one frozen harness and unchanged scientific factors."""

    from evolve.scheduler.arms import ArmIdentity, make_allocation_arm

    identity = ArmIdentity(
        cell_id=arm.cell_id,
        role=arm.role,
        option_id=arm.option_id,
        harness_id=harness_id,
        horizon=arm.horizon,
        cost_class=arm.cost_class,
    )
    return make_allocation_arm(
        identity,
        channel=channel,
        expected_cost=arm.expected_cost,
        hard_cost=arm.hard_cost,
    )


def _special_option_arm(
    base: AllocationArm,
    *,
    role: Role,
    option_id: str,
    channel: Channel,
    option_registry: OptionRegistry,
) -> AllocationArm:
    """Build a dedicated audit/refinement arm from a production context."""

    from evolve.scheduler.arms import ArmIdentity, make_allocation_arm

    spec = option_registry.spec(option_id)
    identity = ArmIdentity(
        cell_id=base.cell_id,
        role=role,
        option_id=option_id,
        harness_id=base.harness_id,
        horizon=spec.max_horizon,
        cost_class="refinement_fixed",
    )
    hard_cost = {
        resource: (
            float(amount) * (2.0 if resource == "verifier_calls" else 1.0)
        )
        for resource, amount in spec.hard_cost.items()
    }
    return make_allocation_arm(
        identity,
        channel=channel,
        expected_cost=spec.expected_cost,
        hard_cost=hard_cost,
    )


def _build_epoch_manifest(
    *,
    run_id: str,
    epoch: int,
    record_threshold: float,
    archive: ScientificArchive,
    posterior: PosteriorStore,
    memory: MemoryStore,
    role_registry: RoleRegistry,
    plan: AllocationPlan,
    option_registry: OptionRegistry,
    harness_registry: HarnessRegistry,
    verifier_id: str,
    verifier_version: str,
    budget_ledger: BudgetLedger,
    seed: int,
) -> EpochManifest:
    archive_snapshot_id, archive_snapshot_hash = _archive_snapshot(archive)
    role_snapshots = role_registry.freeze_epoch(epoch)
    return EpochManifest(
        manifest_id=content_id(
            "epoch_manifest", {"run_id": run_id, "epoch": epoch, "archive_snapshot_id": archive_snapshot_id}
        ),
        run_id=run_id,
        epoch=epoch,
        record_threshold=record_threshold,
        archive_snapshot_id=archive_snapshot_id,
        archive_snapshot_hash=archive_snapshot_hash,
        scheduler_version="zero_inflated_tail_v1",
        scheduler_snapshot_id=_posterior_snapshot_id(posterior),
        role_snapshot_ids={role.value: snapshot.snapshot_id for role, snapshot in role_snapshots.items()},
        causal_memory_snapshot_id=_memory_snapshot_id(memory),
        option_ids=option_registry.option_ids(),
        harness_ids=harness_registry.active_ids,
        verifier_id=verifier_id,
        verifier_version=verifier_version,
        descriptor_version=archive.cell_map_version,
        cell_map_version=archive.cell_map_version,
        fingerprint_version="scientific_fingerprint_v1",
        reporting_schema_version="evolve_reporting_v1",
        budget_ledger_id=budget_ledger.ledger_id,
        allocation_plan_id=_allocation_plan_id(plan),
        seed=seed,
        component_schema_versions=dict(COMPONENT_SCHEMA_VERSIONS),
        method_complete=len(role_snapshots) == 3,
    )


# --------------------------------------------------------------------------
# Epoch state
# --------------------------------------------------------------------------


@dataclass(frozen=True)
class EpochState:
    """Every barrier-synchronized subsystem's current, fully committed state."""

    run_id: str
    epoch: int
    archive: ScientificArchive
    provenance: ProvenanceStore
    posterior: PosteriorStore
    memory: MemoryStore
    role_registry: RoleRegistry
    option_registry: OptionRegistry
    harness_registry: HarnessRegistry
    budget_ledger: BudgetLedger
    record: ConfirmedRecordTracker
    nursery: Mapping[str, NurseryEntry] = field(default_factory=dict)
    role_artifacts: Mapping[str, Mapping[str, Any]] = field(default_factory=dict)


@dataclass(frozen=True)
class EpochReport:
    """Diagnostics returned from one committed epoch, for status/plots."""

    epoch: int
    plan: AllocationPlan
    branch_executions: Tuple[BranchExecution, ...]
    audit_pairs: Tuple[AuditPair, ...]
    record_improved: bool


_ROLE_ADAPTER_DIRECTORY = re.compile(r"adapter_epoch([0-9]{3,})$")
_ROLE_OPTIMIZER_FILE = re.compile(r"optimizer_epoch([0-9]{3,})[.]pt$")


def _artifact_retention_mode(value: Optional[str] = None) -> str:
    """Resolve the operational retention policy without changing run schema."""

    mode = (
        os.environ.get("EVOLVE_ARTIFACT_RETENTION", "all")
        if value is None
        else value
    )
    if mode not in {"all", "latest"}:
        raise EngineError(
            "EVOLVE_ARTIFACT_RETENTION must be 'all' or 'latest'"
        )
    return mode


def _tree_size(path: Path) -> int:
    if path.is_file() and not path.is_symlink():
        return path.stat().st_size
    total = 0
    for item in path.rglob("*"):
        if item.is_file() and not item.is_symlink():
            total += item.stat().st_size
    return total


def _retention_target(layout: RunLayout, relative: str, *, keep_epoch: int) -> Path:
    """Validate one narrowly scoped role-artifact pruning target."""

    parts = Path(relative).parts
    if len(parts) != 3 or parts[0] != "roles":
        raise EngineError(f"unsafe artifact-retention target: {relative}")
    if parts[1] not in {role.value for role in Role}:
        raise EngineError(f"unknown role in artifact-retention target: {relative}")
    match = _ROLE_ADAPTER_DIRECTORY.fullmatch(parts[2])
    if match is None:
        match = _ROLE_OPTIMIZER_FILE.fullmatch(parts[2])
    if match is None or int(match.group(1)) >= keep_epoch:
        raise EngineError(f"unsafe artifact-retention epoch target: {relative}")
    return layout.path(relative)


def _apply_role_artifact_retention(
    layout: RunLayout,
    *,
    keep_epoch: int,
    mode: str,
) -> None:
    """Optionally retain only the newest completed role-training snapshot.

    Scientific evidence, logs, summaries, JSON checkpoints, and their small
    RNG/training companions remain untouched.  Only older immutable LoRA
    directories (including their embedded optimizer ``.pt`` files) and legacy
    standalone role-optimizer files are eligible.  A durable plan is written
    before deletion so an interrupted cleanup can be completed idempotently.
    """

    if _artifact_retention_mode(mode) == "all" or keep_epoch <= 0:
        return
    plan_path = layout.path(f"logs/retention_epoch{keep_epoch:03d}.plan.json")
    result_path = layout.path(f"logs/retention_epoch{keep_epoch:03d}.result.json")
    if result_path.is_file():
        try:
            result = json.loads(result_path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as exc:
            raise EngineError(
                f"invalid artifact-retention result: {result_path}"
            ) from exc
        if (
            result.get("schema_version") != 1
            or result.get("policy") != "latest"
            or result.get("keep_epoch") != keep_epoch
            or not isinstance(result.get("removed"), list)
            or any(not isinstance(item, str) for item in result["removed"])
            or len(set(result["removed"])) != len(result["removed"])
        ):
            raise EngineError(f"invalid artifact-retention result: {result_path}")
        return

    if plan_path.is_file():
        plan = json.loads(plan_path.read_text(encoding="utf-8"))
        if (
            plan.get("schema_version") != 1
            or plan.get("policy") != "latest"
            or plan.get("keep_epoch") != keep_epoch
            or not isinstance(plan.get("targets"), list)
            or any(not isinstance(item, str) for item in plan["targets"])
            or len(set(plan["targets"])) != len(plan["targets"])
        ):
            raise EngineError(f"invalid artifact-retention plan: {plan_path}")
        relative_targets = list(plan["targets"])
    else:
        targets: List[Path] = []
        for role in Role:
            role_dir = layout.path(f"roles/{role.value}")
            if not role_dir.is_dir():
                continue
            for child in role_dir.iterdir():
                match = _ROLE_ADAPTER_DIRECTORY.fullmatch(child.name)
                if match is None:
                    match = _ROLE_OPTIMIZER_FILE.fullmatch(child.name)
                if match is not None and int(match.group(1)) < keep_epoch:
                    if child.is_symlink():
                        raise EngineError(
                            f"refusing symlink artifact-retention target: {child}"
                        )
                    targets.append(child)
        relative_targets = sorted(
            path.relative_to(layout.run_dir).as_posix() for path in targets
        )
        planned_bytes = sum(_tree_size(path) for path in targets)
        _write_json_once(
            plan_path,
            {
                "schema_version": 1,
                "policy": "latest",
                "keep_epoch": keep_epoch,
                "targets": relative_targets,
                "planned_bytes": planned_bytes,
                "preserved": [
                    "scientific evidence and logs",
                    "completed-barrier summaries and JSON checkpoints",
                    "checkpoint training/RNG companions",
                    f"role adapter and optimizer snapshot epoch {keep_epoch}",
                ],
            },
        )

    removed = []
    for relative in relative_targets:
        target = _retention_target(layout, relative, keep_epoch=keep_epoch)
        if not target.exists() and not target.is_symlink():
            removed.append(relative)
            continue
        if target.is_symlink():
            raise EngineError(f"refusing symlink artifact-retention target: {target}")
        if target.is_dir():
            shutil.rmtree(target)
        elif target.is_file():
            target.unlink()
        else:
            raise EngineError(f"unsupported artifact-retention target: {target}")
        fsync_directory(target.parent)
        removed.append(relative)
    _write_json_once(
        result_path,
        {
            "schema_version": 1,
            "policy": "latest",
            "keep_epoch": keep_epoch,
            "removed": removed,
            "latest_role_artifacts_preserved": True,
            "resume_checkpoint_companions_preserved": True,
        },
    )


def _run_guard_document() -> Mapping[str, Any]:
    keys = {
        "cuda_visible_devices": "CUDA_VISIBLE_DEVICES",
        "cpu_cores": "EVOLVE_CPU_CORES",
        "time_limit_hh_mm": "EVOLVE_RUN_TIME_LIMIT",
        "graceful_stop_minutes": "EVOLVE_GRACEFUL_STOP_MINUTES",
        "hard_deadline_epoch": "EVOLVE_HARD_DEADLINE_EPOCH",
        "artifact_retention": "EVOLVE_ARTIFACT_RETENTION",
    }
    return {
        output: os.environ[source]
        for output, source in keys.items()
        if source in os.environ
    }


def _last_committed_epoch(layout: RunLayout) -> Optional[int]:
    """Return only an epoch backed by a fully validated completion marker."""

    try:
        checkpoint_path = _latest_completed_checkpoint(layout)
        checkpoint = json.loads(checkpoint_path.read_text(encoding="utf-8"))
        epoch = checkpoint.get("epoch")
        if isinstance(epoch, bool) or not isinstance(epoch, int) or epoch < 0:
            raise EngineError("completed checkpoint has an invalid epoch")
        return epoch
    except (EngineError, OSError, ValueError, json.JSONDecodeError):
        return None


class EvolveEngine:
    """The composed EVOLVE runtime: fresh/resume lifecycle and epoch barriers."""

    def __init__(
        self,
        *,
        config: EvolveConfig,
        resolved_config: Mapping[str, Any],
        metadata: Mapping[str, Any],
        workers: Optional[EngineWorkers] = None,
        adapter: Optional[ProblemScientificAdapter] = None,
        runs_root: Optional[Union[str, Path]] = None,
    ) -> None:
        self.config = config
        self.resolved_config = dict(resolved_config)
        self.metadata = dict(metadata)
        self._workers = workers
        self._adapter = adapter
        self._runs_root = Path(runs_root) if runs_root is not None else Path.cwd() / "runs"
        self._artifact_retention = _artifact_retention_mode()

    def _print_progress(
        self,
        stage: str,
        *,
        epoch: Optional[int] = None,
        completed: Optional[int] = None,
        total: Optional[int] = None,
        unit: str = "items",
        detail: Optional[str] = None,
    ) -> None:
        from evolve.reporting.console import format_progress

        print(
            format_progress(
                stage,
                epoch=epoch,
                total_epochs=(
                    self.config.evolve.budget.epochs
                    if epoch is not None
                    else None
                ),
                completed=completed,
                total=total,
                unit=unit,
                detail=detail,
            ),
            flush=True,
        )

    @staticmethod
    def _announce_run_directory(layout: RunLayout, *, mode: str) -> None:
        print(
            f"\nEVOLVE · {mode} run directory · {layout.run_dir}\n",
            flush=True,
        )

    # -- run lifecycle -----------------------------------------------------

    def run(self) -> int:
        adapter = self._adapter or self._build_adapter()
        verification_policy = VerificationPolicy.create(
            version="evolve_engine_v1", production=not self.config.method_incomplete
        )
        try:
            layout, state = self._attach(
                adapter=adapter,
                verification_policy=verification_policy,
            )
        except _RunAttachInterrupted:
            return 130
        target_epochs = self.config.evolve.budget.epochs
        completion_reason = "target epochs reached"
        workers = self._workers
        interrupted = False
        try:
            self._print_progress(
                "model runtime",
                detail="initializing backbone, role adapters, and generation workers",
            )
            workers = workers or build_production_workers(
                self.config, adapter=adapter, layout=layout, state=state
            )
            self._print_progress("model runtime", detail="workers ready")
            try:
                _latest_completed_checkpoint(layout)
                has_completed_barrier = True
            except EngineError:
                has_completed_barrier = False
            needs_bootstrap_commit = not has_completed_barrier
            if needs_bootstrap_commit:
                self._print_progress(
                    "bootstrap barrier",
                    completed=0,
                    total=1,
                    unit="barrier",
                    detail="persisting role adapters and checkpoint",
                )
                role_artifacts = (
                    workers.persist_roles(state)
                    if workers.persist_roles is not None
                    else state.role_artifacts
                )
                self._commit_bootstrap(
                    layout,
                    state,
                    role_artifacts=role_artifacts or {},
                    workers=workers,
                    adapter=adapter,
                )
                self._write_status(layout, state, note="bootstrap committed")
                self._print_progress(
                    "bootstrap barrier",
                    completed=1,
                    total=1,
                    unit="barrier",
                    detail="committed",
                )
            else:
                # A checkpoint summary is the completed-barrier authority. Repair
                # non-critical mirrors that may have been interrupted after that
                # marker without replaying any scientific work.
                if state.role_artifacts:
                    self._publish_role_pointers(
                        layout,
                        role_artifacts=state.role_artifacts,
                        epoch=state.epoch,
                    )
                if state.record.evidence_id is not None:
                    best_pointer = layout.path("best/latest.json")
                    best_evidence_id = None
                    if best_pointer.is_file():
                        try:
                            best_evidence_id = json.loads(
                                best_pointer.read_text(encoding="utf-8")
                            ).get("evidence_id")
                        except (OSError, json.JSONDecodeError):
                            best_evidence_id = None
                    if best_evidence_id != state.record.evidence_id:
                        self._publish_best(layout, state, adapter=adapter)
                self._write_status(
                    layout, state, note="resumed from completed barrier"
                )
            while state.epoch < target_epochs:
                if state.budget_ledger.remaining("verifier_calls") <= 2.0:
                    completion_reason = "verifier budget reserve exhausted"
                    break
                if workers.begin_epoch is not None:
                    workers.begin_epoch(state)
                self._print_progress(
                    "planning",
                    epoch=state.epoch,
                    detail="freezing archive, roles, scheduler, and allocation plan",
                )
                try:
                    state, report = self.run_epoch(
                        layout,
                        state,
                        workers=workers,
                        adapter=adapter,
                        verification_policy=verification_policy,
                    )
                except SchedulerError as exc:
                    if not str(exc).startswith(
                        "resource limits cannot satisfy mandatory"
                    ):
                        raise
                    completion_reason = f"verifier budget exhausted: {exc}"
                    break
                self._print_progress(
                    "barrier commit",
                    epoch=report.epoch,
                    completed=0,
                    total=1,
                    unit="barrier",
                    detail="checkpointing adapters, optimizer, RNG, and artifacts",
                )
                self._commit_barrier(
                    layout, state, report, adapter=adapter, workers=workers
                )
                self._print_progress(
                    "barrier commit",
                    epoch=report.epoch,
                    completed=1,
                    total=1,
                    unit="barrier",
                    detail="epoch committed",
                )
        except KeyboardInterrupt:
            interrupted = True
        finally:
            if workers is not None and workers.shutdown is not None:
                workers.shutdown()
        if interrupted:
            committed_epoch = _last_committed_epoch(layout)
            if committed_epoch is not None:
                _apply_role_artifact_retention(
                    layout,
                    keep_epoch=committed_epoch,
                    mode=self._artifact_retention,
                )
            self._write_interrupted_status(
                layout,
                state,
                committed_epoch=committed_epoch,
            )
            return 130
        target_reached = state.epoch >= target_epochs
        self._write_status(
            layout,
            state,
            note=(
                "run complete: target epochs reached"
                if target_reached
                else f"run stopped safely: {completion_reason}"
            ),
        )
        atomic_write_json(
            layout.path("final.summary.json"),
            {
                "schema_version": 1,
                "run_id": state.run_id,
                "epochs_completed": state.epoch,
                "target_epochs": target_epochs,
                "target_epochs_reached": target_reached,
                "completion_reason": completion_reason,
                "confirmed_record": state.record.internal_reward,
                "archive_coverage": state.archive.coverage,
                "budget": state.budget_ledger.to_dict(),
                "checkpoint": json.loads(
                    layout.path("checkpoints/latest.json").read_text(
                        encoding="utf-8"
                    )
                ),
            },
        )
        return 0

    def _build_adapter(self) -> ProblemScientificAdapter:
        try:
            from problems.registry import get_problem
        except ImportError as exc:  # pragma: no cover - packaging failure
            raise EngineError(f"cannot import problems.registry: {exc}") from exc
        problem = get_problem(
            self.config.problem, dict(self.config.problem_runtime_config)
        )
        return ProblemScientificAdapter(problem, problem_id=self.config.problem)

    # -- fresh / resume ------------------------------------------------

    def _attach(
        self, *, adapter: ProblemScientificAdapter, verification_policy: VerificationPolicy
    ) -> Tuple[RunLayout, EpochState]:
        if self.metadata.get("mode") == "resume":
            return self._attach_resume(
                adapter=adapter,
                verification_policy=verification_policy,
            )
        return self._attach_fresh(adapter=adapter, verification_policy=verification_policy)

    def _attach_fresh(
        self, *, adapter: ProblemScientificAdapter, verification_policy: VerificationPolicy
    ) -> Tuple[RunLayout, EpochState]:
        runs_root = self._runs_root
        layout = create_fresh_run_layout(
            runs_root, problem=self.config.problem, model_name=self.config.model_name
        )
        self._announce_run_directory(layout, mode="fresh")
        run_id = content_id("run", {"run_dir": str(layout.run_dir)})
        requested_path = Path(str(self.metadata.get("config_path", "")))
        requested_yaml = (
            requested_path.read_text(encoding="utf-8")
            if requested_path.is_file()
            else json.dumps(self.resolved_config, sort_keys=True, indent=2)
        )
        write_initial_run_metadata(
            layout.run_dir,
            requested_yaml=requested_yaml,
            resolved_config=self.resolved_config,
            command=list(sys.argv),
            environment=_environment_document(),
            git_state=_git_state(),
            model={
                "model_name": self.config.model_name,
                "training_backend": self.config.backend,
                "generation_backend": self.config.generation_backend,
                "training_load_in_4bit": self.config.load_in_4bit,
            },
            package_versions=_package_versions(),
            host=_host_document(),
            gpus=_gpu_manifest(self.config),
            worker_topology=_worker_topology(self.config),
            seeds={"base_seed": self.config.seed},
            versions={"schema_version": self.config.schema_version, "config_hash": self.resolved_config.get("config_hash", "")},
            run_id=run_id,
        )
        state = self._initial_state(run_id)
        try:
            state = self._seed_archive(
                state,
                layout=layout,
                adapter=adapter,
                verification_policy=verification_policy,
            )
        except KeyboardInterrupt:
            self._write_interrupted_status(
                layout,
                state,
                committed_epoch=None,
            )
            raise _RunAttachInterrupted from None
        return layout, state

    def _commit_bootstrap(
        self,
        layout: RunLayout,
        state: EpochState,
        *,
        role_artifacts: Mapping[str, Mapping[str, Any]],
        workers: EngineWorkers,
        adapter: ProblemScientificAdapter,
    ) -> None:
        """Publish epoch zero only after all three role artifacts are durable."""

        self._publish_snapshots(layout, state)
        checkpoint_path = layout.path("checkpoints/checkpoint_epoch000.json")
        checkpoint = _checkpoint_payload(state, role_artifacts=role_artifacts)
        _write_json_once(checkpoint_path, checkpoint)
        self._persist_training_state(
            layout,
            state=state,
            checkpoint=checkpoint,
            workers=workers,
            epoch=state.epoch,
        )
        checkpoint_hash = content_hash(checkpoint)
        training_state_path = layout.path("checkpoints/checkpoint_epoch000.pt")
        pointer = {
            "schema_version": 1,
            "epoch": 0,
            "committed_epoch": 0,
            "checkpoint": checkpoint_path.name,
            "checkpoint_hash": checkpoint_hash,
            "training_state": training_state_path.name,
            "training_state_hash": _file_sha256(training_state_path),
        }
        atomic_write_json(layout.path("checkpoints/latest.json"), pointer)
        _write_json_once(layout.path("bootstrap.summary.json"), pointer)
        self._publish_role_pointers(
            layout, role_artifacts=role_artifacts, epoch=state.epoch
        )
        with ControllerEventWriter(layout.path("events.jsonl")) as event_writer:
            event_writer.append(
                "barrier_committed",
                {
                    "kind": "bootstrap",
                    "epoch": state.epoch,
                    "checkpoint": checkpoint_path.name,
                    "checkpoint_hash": checkpoint_hash,
                },
                idempotency_key="barrier-committed:bootstrap",
            )
        self._publish_best(layout, state, adapter=adapter)

    def _attach_resume(
        self,
        *,
        adapter: ProblemScientificAdapter,
        verification_policy: VerificationPolicy,
    ) -> Tuple[RunLayout, EpochState]:
        resume_dir = Path(self.metadata["resume_dir"])
        layout = open_existing_run_layout(resume_dir, resume=True)
        self._announce_run_directory(layout, mode="resume")
        checkpoint = None
        try:
            checkpoint_path = _latest_completed_checkpoint(layout)
            checkpoint = json.loads(checkpoint_path.read_text(encoding="utf-8"))
        except EngineError:
            # An explicitly resumed fresh run may have crashed before its
            # bootstrap completion marker. Reconstruct the deterministic
            # initial state under the existing run identity and replay only
            # missing seed artifacts; never attach implicitly.
            if layout.path("bootstrap.summary.json").exists() or any(
                layout.run_dir.glob("step*/step*.summary.json")
            ):
                raise EngineError(
                    "completion markers exist but no checkpoint passes hash "
                    "validation; refusing to reinterpret the run as an "
                    "incomplete bootstrap"
                )
            manifest_path = layout.path("manifest.json")
            if not manifest_path.is_file():
                raise
            manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
            run_id = manifest.get("run_id")
            if not isinstance(run_id, str):
                raise EngineError(
                    "incomplete bootstrap manifest has no stable run_id"
                )
            state = self._initial_state(run_id)
            try:
                state = self._seed_archive(
                    state,
                    layout=layout,
                    adapter=adapter,
                    verification_policy=verification_policy,
                )
            except KeyboardInterrupt:
                self._write_interrupted_status(
                    layout,
                    state,
                    committed_epoch=None,
                )
                raise _RunAttachInterrupted from None
            recovery_checkpoint = {
                "record_type": "evolve_bootstrap_recovery_anchor",
                "schema_version": 1,
                "run_id": state.run_id,
                "state": _checkpoint_payload(state),
            }
            _write_json_once(
                layout.path("checkpoints/bootstrap_recovery.json"),
                recovery_checkpoint,
            )
            checkpoint_hash_for_resume = content_hash(recovery_checkpoint)
        else:
            checkpoint_hash_for_resume = content_hash(checkpoint)
        write_resume_run_metadata(
            layout.run_dir,
            resume_index=int(self.metadata.get("effective_resume_index", 0)) + 1,
            resolved_config=self.resolved_config,
            command=list(sys.argv),
            environment=_environment_document(),
            git_state=_git_state(),
            model={
                "model_name": self.config.model_name,
                "training_backend": self.config.backend,
                "generation_backend": self.config.generation_backend,
                "training_load_in_4bit": self.config.load_in_4bit,
            },
            package_versions=_package_versions(),
            host=_host_document(),
            gpus=_gpu_manifest(self.config),
            worker_topology=_worker_topology(self.config),
            seeds={"base_seed": self.config.seed},
            versions={
                "schema_version": self.config.schema_version,
                "config_hash": self.resolved_config.get("config_hash", ""),
            },
            checkpoint_hash=checkpoint_hash_for_resume,
        )
        if checkpoint is not None:
            state = _state_from_checkpoint(checkpoint, config=self.config)
        return layout, state

    def _initial_state(self, run_id: str) -> EpochState:
        from evolve.workers.runtime import backbone_identity_for_config

        backbone = backbone_identity_for_config(self.config)
        roles = self.config.evolve.roles.enabled
        role_registry = RoleRegistry.create_production(
            run_id=run_id,
            backbone_id=backbone.backbone_id,
            backbone_version=backbone.identity_version,
            backbone_hash=backbone.weights_hash,
            base_seed=self.config.seed,
        ) if tuple(roles) == ("scout", "mechanist", "challenger") else RoleRegistry.create_test_fixture(
            run_id=run_id,
            backbone_id=backbone.backbone_id,
            backbone_version=backbone.identity_version,
            backbone_hash=backbone.weights_hash,
            base_seed=self.config.seed,
            roles=[Role(name) for name in roles],
        )
        harness_registry = default_harness_registry(
            active_versions=self.config.evolve.harnesses.active_versions
        )
        option_registry = production_option_registry(
            harness_eligibility=tuple(sorted(harness_registry.specs)),
            max_horizon=self.config.evolve.options.max_horizon,
        )
        budget_ledger = BudgetService.create(
            {"verifier_calls": float(self.config.evolve.budget.verifier_calls)},
            identity=[run_id],
        )
        return EpochState(
            run_id=run_id,
            epoch=0,
            archive=ScientificArchive(
                max_promising_slots=max(
                    1, (self.config.evolve.archive.elites_per_cell - 1) // 2
                ),
                max_stepping_stone_slots=max(
                    1,
                    self.config.evolve.archive.elites_per_cell
                    - 1
                    - max(
                        1,
                        (self.config.evolve.archive.elites_per_cell - 1) // 2,
                    ),
                ),
            ),
            provenance=ProvenanceStore(),
            posterior=PosteriorStore(),
            memory=MemoryStore(),
            role_registry=role_registry,
            option_registry=option_registry,
            harness_registry=harness_registry,
            budget_ledger=budget_ledger,
            record=ConfirmedRecordTracker(),
            nursery={},
        )

    def _seed_archive(
        self,
        state: EpochState,
        *,
        layout: RunLayout,
        adapter: ProblemScientificAdapter,
        verification_policy: VerificationPolicy,
    ) -> EpochState:
        """Bootstrap the archive from the problem's declared seed states.

        Seeds are verified through the same common-verifier pipeline as any
        other candidate; a seed can populate the archive but never becomes
        the confirmed record by itself (that always requires a separate
        confirming re-verification, see :meth:`_confirm`).
        """

        from evolve.verifier.models import PersistedAnswerPayload
        from evolve.verifier.service import verify_persisted_answer
        from evolve.workers.verification import (
            VerificationWorkerError,
            persist_answer_artifact,
            persist_verifier_trace,
            restore_durable_verification_result,
        )

        problem = adapter.problem
        seed_branch_id = content_id("branch", {"kind": "seed", "run_id": state.run_id})
        seed_harness_id = content_id("harness", {"kind": "seed"})
        seed_policy_snapshot_id = content_id(
            "role_snapshot", {"kind": "seed"}
        )
        archive = state.archive
        seeds = tuple(problem.seed_states())
        self._print_progress(
            "bootstrap verification",
            completed=0,
            total=len(seeds),
            unit="seeds",
            detail="validating problem-provided baselines",
        )
        admitted_count = 0
        admitted_results = []
        failures: List[str] = []
        budget_ledger = state.budget_ledger
        bootstrap_dir = layout.path("bootstrap")
        bootstrap_dir.mkdir(parents=True, exist_ok=True)

        def seed_verifier_key(seed_index: int, attempt_index: int) -> str:
            base = f"bootstrap:seed:{seed_index}:verify"
            return base if attempt_index == 0 else f"{base}:retry:{attempt_index}"

        for index, seed in enumerate(seeds):
            self._print_progress(
                "bootstrap verification",
                completed=index,
                total=len(seeds),
                unit="seeds",
                detail=f"verifying seed {index + 1}/{len(seeds)}",
            )
            candidate = seed.construction if seed.construction is not None else seed.code
            source_text = seed.code or json.dumps(candidate, sort_keys=True, default=str)
            source_hash = content_hash(source_text)
            proposal = Proposal(
                proposal_id=content_id(
                    "proposal",
                    {"run_id": state.run_id, "kind": "seed", "index": index, "source_hash": source_hash},
                ),
                run_id=state.run_id,
                problem_id=self.config.problem,
                source_text=source_text,
                source_hash=source_hash,
                parent_state_id=None,
                branch_id=seed_branch_id,
            )
            proposal_path = bootstrap_dir / f"seed{index:03d}.proposal.json"
            evidence_path = bootstrap_dir / f"seed{index:03d}.evidence.json"
            state_path = bootstrap_dir / f"seed{index:03d}.state.json"
            descriptor_path = bootstrap_dir / f"seed{index:03d}.descriptor.json"
            archive_path = bootstrap_dir / f"seed{index:03d}.archive.json"
            error_path = bootstrap_dir / f"seed{index:03d}.error.json"

            _write_json_once(proposal_path, proposal.to_dict())
            if error_path.is_file() and not evidence_path.is_file():
                error_document = json.loads(
                    error_path.read_text(encoding="utf-8")
                )
                error_attempts = error_document.get(
                    "verifier_attempts",
                    1 if error_document.get("verifier_debited") else 0,
                )
                if (
                    isinstance(error_attempts, bool)
                    or not isinstance(error_attempts, int)
                    or not 0 <= error_attempts
                    <= verification_policy.infrastructure_retry_limit + 1
                ):
                    raise EngineError(
                        f"bootstrap seed {index} error has invalid attempt count"
                    )
                for attempt_index in range(error_attempts):
                    budget_ledger = BudgetService.debit(
                        budget_ledger,
                        resource="verifier_calls",
                        amount=1.0,
                        transaction_key=seed_verifier_key(index, attempt_index),
                    )
                failures.append(
                    f"seed {index}: {error_document.get('exception_type')}: "
                    f"{error_document.get('message')}"
                )
                continue

            if evidence_path.is_file():
                if not proposal_path.is_file():
                    raise EngineError(
                        f"bootstrap seed {index} has evidence without its proposal"
                    )
                durable_proposal = Proposal.from_dict(
                    json.loads(proposal_path.read_text(encoding="utf-8"))
                )
                if durable_proposal != proposal:
                    raise EngineError(
                        f"bootstrap seed {index} proposal identity changed"
                    )
                evidence = EvidencePacket.from_dict(
                    json.loads(evidence_path.read_text(encoding="utf-8"))
                )
                _validate_proposal_evidence_binding(durable_proposal, evidence)
                if (
                    evidence.verifier_id != adapter.verifier_id
                    or evidence.verifier_version != adapter.verifier_version
                    or evidence.harness_id != seed_harness_id
                    or evidence.policy_snapshot_id != seed_policy_snapshot_id
                ):
                    raise EngineError(
                        f"bootstrap seed {index} verifier context changed on recovery"
                    )
                attempt_index = _verification_attempt_index(
                    evidence.flags.get("verification_attempt_index"),
                    maximum=verification_policy.infrastructure_retry_limit,
                )
                for replay_index in range(attempt_index + 1):
                    budget_ledger = BudgetService.debit(
                        budget_ledger,
                        resource="verifier_calls",
                        amount=1.0,
                        transaction_key=seed_verifier_key(index, replay_index),
                    )
                if not evidence.admitted:
                    if state_path.exists() or descriptor_path.exists():
                        raise EngineError(
                            f"bootstrap seed {index} failed evidence has scientific state artifacts"
                        )
                    failures.append(
                        f"seed {index}: verifier rejected: "
                        f"{evidence.failure_kind.value}: "
                        f"{evidence.diagnostics.get('message', '')}"
                    )
                    continue
                if evidence.scientific_state_id is None:
                    raise EngineError(
                        f"bootstrap seed {index} admitted evidence has no state"
                    )
                if state_path.is_file() and descriptor_path.is_file():
                    seed_state = VerifiedScientificState.from_dict(
                        json.loads(state_path.read_text(encoding="utf-8"))
                    )
                    seed_descriptor = Descriptor.from_dict(
                        json.loads(descriptor_path.read_text(encoding="utf-8"))
                    )
                else:
                    restored_state, restored_descriptor = (
                        _restore_admitted_artifacts(evidence, adapter=adapter)
                    )
                    if state_path.is_file():
                        seed_state = VerifiedScientificState.from_dict(
                            json.loads(state_path.read_text(encoding="utf-8"))
                        )
                        if seed_state != restored_state:
                            raise EngineError(
                                f"bootstrap seed {index} state changed on recovery"
                            )
                    else:
                        seed_state = restored_state
                        _write_json_once(state_path, seed_state.to_dict())
                    if descriptor_path.is_file():
                        seed_descriptor = Descriptor.from_dict(
                            json.loads(
                                descriptor_path.read_text(encoding="utf-8")
                            )
                        )
                        if seed_descriptor != restored_descriptor:
                            raise EngineError(
                                f"bootstrap seed {index} descriptor changed on recovery"
                            )
                    else:
                        seed_descriptor = restored_descriptor
                        _write_json_once(
                            descriptor_path, seed_descriptor.to_dict()
                        )
                archive = archive.ensure_cell(
                    seed_descriptor, force_empty_sampling=False
                )
                archive, seed_decision = archive.offer(
                    seed_descriptor, durable_proposal, seed_state, evidence
                )
                _write_json_once(
                    archive_path,
                    {"schema_version": 1, **vars(seed_decision)},
                )
                admitted_count += 1
                admitted_results.append((durable_proposal, evidence))
                continue
            if state_path.exists() or descriptor_path.exists() or archive_path.exists():
                raise EngineError(
                    f"bootstrap seed {index} has artifacts without evidence"
                )
            verifier_attempts = 0
            try:
                payload = problem.serialize_answer(candidate)
                artifact_path = persist_answer_artifact(
                    run_dir=layout.run_dir, problem_id=self.config.problem, payload=payload
                )
                persisted = PersistedAnswerPayload.create(
                    problem_id=self.config.problem, artifact_uri=str(artifact_path), payload=payload
                )
                result = None
                for attempt_index in range(
                    verification_policy.infrastructure_retry_limit + 1
                ):
                    attempt_stem = f"seed{index:03d}.attempt{attempt_index:02d}"
                    attempt_evidence_path = (
                        bootstrap_dir / f"{attempt_stem}.evidence.json"
                    )
                    attempt_state_path = bootstrap_dir / f"{attempt_stem}.state.json"
                    attempt_descriptor_path = (
                        bootstrap_dir / f"{attempt_stem}.descriptor.json"
                    )
                    attempt_completed_path = (
                        bootstrap_dir / f"{attempt_stem}.completed.json"
                    )
                    budget_ledger = BudgetService.debit(
                        budget_ledger,
                        resource="verifier_calls",
                        amount=1.0,
                        transaction_key=seed_verifier_key(index, attempt_index),
                    )
                    verifier_attempts = attempt_index + 1
                    if attempt_evidence_path.is_file():
                        attempt_evidence = EvidencePacket.from_dict(
                            json.loads(
                                attempt_evidence_path.read_text(encoding="utf-8")
                            )
                        )
                        _validate_proposal_evidence_binding(
                            proposal, attempt_evidence
                        )
                        if (
                            attempt_evidence.verifier_id != adapter.verifier_id
                            or attempt_evidence.verifier_version
                            != adapter.verifier_version
                            or attempt_evidence.harness_id != seed_harness_id
                            or attempt_evidence.policy_snapshot_id
                            != seed_policy_snapshot_id
                            or attempt_evidence.flags.get(
                                "verification_attempt_index"
                            )
                            != attempt_index
                        ):
                            raise EngineError(
                                f"bootstrap seed {index} attempt context changed"
                            )
                        result = restore_durable_verification_result(
                            evidence=attempt_evidence,
                            adapter=adapter,
                            state_path=attempt_state_path,
                            descriptor_path=attempt_descriptor_path,
                        )
                    else:
                        if (
                            attempt_state_path.exists()
                            or attempt_descriptor_path.exists()
                            or attempt_completed_path.exists()
                        ):
                            raise EngineError(
                                f"bootstrap seed {index} has a partial verifier attempt"
                            )
                        result = verify_persisted_answer(
                            adapter=adapter,
                            proposal=proposal,
                            persisted_answer=persisted,
                            verification_policy=verification_policy,
                            harness_id=seed_harness_id,
                            policy_snapshot_id=seed_policy_snapshot_id,
                            attempt_index=attempt_index,
                        )
                        persist_verifier_trace(
                            run_dir=layout.run_dir,
                            result=result,
                            phase=(
                                "bootstrap_seed_verification_attempt_"
                                f"{attempt_index}"
                            ),
                        )
                        _write_json_once(
                            attempt_evidence_path, result.evidence.to_dict()
                        )
                        if result.state is not None:
                            _write_json_once(
                                attempt_state_path, result.state.to_dict()
                            )
                            _write_json_once(
                                attempt_descriptor_path,
                                result.descriptor.to_dict(),
                            )
                    _write_json_once(
                        attempt_completed_path,
                        {
                            "schema_version": 1,
                            "attempt_index": attempt_index,
                            "evidence_id": result.evidence.evidence_id,
                            "resolved": result.evidence.resolved,
                        },
                    )
                    if result.evidence.resolved:
                        break
                if result is None:
                    raise EngineError(
                        f"bootstrap seed {index} produced no verifier result"
                    )
            except (EngineError, VerificationWorkerError):
                # A durable identity/schema conflict is controller corruption,
                # not a scientifically bad seed and not retryable evidence.
                raise
            except Exception as exc:
                _write_json_once(
                    error_path,
                    {
                        "schema_version": 1,
                        "seed_index": index,
                        "verifier_debited": verifier_attempts > 0,
                        "verifier_attempts": verifier_attempts,
                        "exception_type": type(exc).__name__,
                        "message": str(exc)[:2048],
                    },
                )
                failures.append(f"seed {index}: {type(exc).__name__}: {exc}")
                continue
            _write_json_once(evidence_path, result.evidence.to_dict())
            if not result.evidence.admitted or result.state is None:
                failures.append(
                    f"seed {index}: verifier rejected: "
                    f"{result.evidence.failure_kind.value}: "
                    f"{result.evidence.diagnostics.get('message', '')}"
                )
                continue
            archive = archive.ensure_cell(result.descriptor, force_empty_sampling=False)
            try:
                archive, seed_decision = archive.offer(
                    result.descriptor, proposal, result.state, result.evidence
                )
            except ArchiveAdmissionError as exc:
                failures.append(f"seed {index}: archive rejected: {exc}")
                continue
            admitted_count += 1
            admitted_results.append((proposal, result.evidence))
            _write_json_once(state_path, result.state.to_dict())
            _write_json_once(descriptor_path, result.descriptor.to_dict())
            _write_json_once(
                archive_path, {"schema_version": 1, **vars(seed_decision)}
            )
        self._print_progress(
            "bootstrap verification",
            completed=len(seeds),
            total=len(seeds),
            unit="seeds",
            detail=f"{admitted_count} admitted",
        )
        if admitted_count == 0:
            detail = "; ".join(failures[:3]) or "problem returned no seed states"
            raise EngineError(
                f"{self.config.problem} bootstrap admitted no seeds "
                f"({len(seeds)} declared): {detail}"
            )

        # The finite-budget objective starts from the best independently
        # verified seed, not from a problem failure sentinel. Confirm the saved
        # payload exactly once (plus one bounded infrastructure retry) before
        # publishing the bootstrap barrier.
        seed_proposal, seed_evidence = max(
            admitted_results,
            key=lambda item: (
                float(item[1].internal_reward),
                item[1].evidence_id,
            ),
        )
        self._print_progress(
            "bootstrap confirmation",
            completed=0,
            total=1,
            unit="record",
            detail="reverifying the best saved seed payload",
        )
        confirmation_key = "bootstrap:record-confirmation"
        try:
            budget_ledger = BudgetService.debit(
                budget_ledger,
                resource="verifier_calls",
                amount=2.0,
                transaction_key=confirmation_key,
            )
        except BudgetOverrun as exc:
            raise EngineError(
                "bootstrap verifier budget cannot confirm the best seed"
            ) from exc
        confirmation_evidence_path = (
            bootstrap_dir / "record.confirmation.evidence.json"
        )
        confirmation_state_path = (
            bootstrap_dir / "record.confirmation.state.json"
        )
        confirmation_descriptor_path = (
            bootstrap_dir / "record.confirmation.descriptor.json"
        )
        confirmation_result_path = (
            bootstrap_dir / "record.confirmation.result.json"
        )
        confirmation_aborted_path = (
            bootstrap_dir / "record.confirmation.aborted.json"
        )
        if confirmation_aborted_path.is_file():
            raise EngineError(
                "bootstrap record confirmation was already infrastructure-aborted"
            )
        if confirmation_evidence_path.is_file():
            confirmation_evidence = EvidencePacket.from_dict(
                json.loads(
                    confirmation_evidence_path.read_text(encoding="utf-8")
                )
            )
            _validate_confirmation_binding(
                seed_proposal, seed_evidence, confirmation_evidence
            )
            if not confirmation_evidence.confirmed:
                raise EngineError(
                    "persisted bootstrap confirmation is not confirmed"
                )
            if (
                confirmation_state_path.is_file()
                and confirmation_descriptor_path.is_file()
            ):
                confirmation_state = VerifiedScientificState.from_dict(
                    json.loads(
                        confirmation_state_path.read_text(encoding="utf-8")
                    )
                )
                confirmation_descriptor = Descriptor.from_dict(
                    json.loads(
                        confirmation_descriptor_path.read_text(encoding="utf-8")
                    )
                )
            else:
                restored_state, restored_descriptor = _restore_admitted_artifacts(
                    confirmation_evidence, adapter=adapter
                )
                if confirmation_state_path.is_file():
                    confirmation_state = VerifiedScientificState.from_dict(
                        json.loads(
                            confirmation_state_path.read_text(encoding="utf-8")
                        )
                    )
                    if confirmation_state != restored_state:
                        raise EngineError(
                            "bootstrap confirmation state changed on recovery"
                        )
                else:
                    confirmation_state = restored_state
                    _write_json_once(
                        confirmation_state_path, confirmation_state.to_dict()
                    )
                if confirmation_descriptor_path.is_file():
                    confirmation_descriptor = Descriptor.from_dict(
                        json.loads(
                            confirmation_descriptor_path.read_text(
                                encoding="utf-8"
                            )
                        )
                    )
                    if confirmation_descriptor != restored_descriptor:
                        raise EngineError(
                            "bootstrap confirmation descriptor changed on recovery"
                        )
                else:
                    confirmation_descriptor = restored_descriptor
                    _write_json_once(
                        confirmation_descriptor_path,
                        confirmation_descriptor.to_dict(),
                    )
            attempt_flag = confirmation_evidence.flags.get(
                "verification_attempt_index"
            )
            attempts = (
                _verification_attempt_index(attempt_flag, maximum=1) + 1
                if attempt_flag is not None
                else 2
            )
            if confirmation_result_path.is_file():
                confirmation_result_document = json.loads(
                    confirmation_result_path.read_text(encoding="utf-8")
                )
                durable_attempts = _confirmation_attempt_count(
                    confirmation_result_document["attempts"]
                )
                _validate_attempt_count_covers_evidence(
                    attempt_flag,
                    durable_attempts,
                    context="bootstrap confirmation",
                )
                attempts = durable_attempts
                if (
                    confirmation_result_document.get("evidence_id")
                    != confirmation_evidence.evidence_id
                    or confirmation_result_document.get("state_id")
                    != confirmation_state.state_id
                ):
                    raise EngineError(
                        "bootstrap confirmation result references different artifacts"
                    )
            if attempts < 2:
                budget_ledger = BudgetService.refund(
                    budget_ledger,
                    resource="verifier_calls",
                    amount=float(2 - attempts),
                    transaction_key=f"{confirmation_key}:refund",
                    debit_transaction_key=confirmation_key,
                )
            _write_json_once(
                confirmation_result_path,
                {
                    "schema_version": 1,
                    "evidence_id": confirmation_evidence.evidence_id,
                    "state_id": confirmation_state.state_id,
                    "attempts": attempts,
                },
            )
            archive, confirmation_decision = archive.offer(
                confirmation_descriptor,
                seed_proposal,
                confirmation_state,
                confirmation_evidence,
            )
            _write_json_once(
                bootstrap_dir / "record.confirmation.archive.json",
                {"schema_version": 1, **vars(confirmation_decision)},
            )
            record = state.record.consider(
                confirmation_state,
                confirmation_evidence,
                archive=archive,
            )
            self._print_progress(
                "bootstrap confirmation",
                completed=1,
                total=1,
                unit="record",
                detail="confirmed",
            )
            return replace(
                state,
                archive=archive,
                record=record,
                budget_ledger=budget_ledger,
            )
        try:
            confirmation, attempts = self._confirm(
                seed_proposal,
                seed_evidence,
                adapter=adapter,
                verification_policy=verification_policy,
                run_dir=layout.run_dir,
                attempt_dir=bootstrap_dir / "record.confirmation.attempts",
                phase="bootstrap_record_confirmation",
            )
        except RecordConfirmationInfrastructureError as exc:
            write_immutable_json(
                bootstrap_dir / "record.confirmation.aborted.json",
                {
                    "schema_version": 1,
                    "failure_kind": FailureKind.INFRASTRUCTURE.value,
                    "exception_type": type(exc).__name__,
                    "message": str(exc)[:2048],
                    "attempts": 2,
                },
            )
            raise
        _validate_confirmation_binding(
            seed_proposal, seed_evidence, confirmation.evidence
        )
        attempts = _confirmation_attempt_count(attempts)
        if attempts < 2:
            budget_ledger = BudgetService.refund(
                budget_ledger,
                resource="verifier_calls",
                amount=float(2 - attempts),
                transaction_key=f"{confirmation_key}:refund",
                debit_transaction_key=confirmation_key,
            )
        _write_json_once(
            confirmation_evidence_path, confirmation.evidence.to_dict()
        )
        if not confirmation.evidence.confirmed or confirmation.state is None:
            raise EngineError("the best bootstrap seed could not be confirmed")
        _write_json_once(confirmation_state_path, confirmation.state.to_dict())
        _write_json_once(
            confirmation_descriptor_path, confirmation.descriptor.to_dict()
        )
        _write_json_once(
            confirmation_result_path,
            {
                "schema_version": 1,
                "evidence_id": confirmation.evidence.evidence_id,
                "state_id": confirmation.state.state_id,
                "attempts": attempts,
            },
        )
        archive, confirmation_decision = archive.offer(
            confirmation.descriptor,
            seed_proposal,
            confirmation.state,
            confirmation.evidence,
        )
        write_immutable_json(
            bootstrap_dir / "record.confirmation.archive.json",
            {"schema_version": 1, **vars(confirmation_decision)},
        )
        record = state.record.consider(
            confirmation.state,
            confirmation.evidence,
            archive=archive,
        )
        self._print_progress(
            "bootstrap confirmation",
            completed=1,
            total=1,
            unit="record",
            detail="confirmed",
        )
        return replace(
            state,
            archive=archive,
            record=record,
            budget_ledger=budget_ledger,
        )

    def _confirm(
        self,
        proposal: Proposal,
        evidence: Any,
        *,
        adapter: ProblemScientificAdapter,
        verification_policy: VerificationPolicy,
        run_dir: Path,
        attempt_dir: Path,
        phase: str,
    ):
        """Confirm a saved payload with two durable, replayable attempts."""

        from evolve.verifier.models import PersistedAnswerPayload
        from evolve.verifier.service import confirm_persisted_answer
        from evolve.workers.verification import (
            persist_verifier_trace,
            restore_durable_verification_result,
        )

        persisted = PersistedAnswerPayload.create(
            problem_id=evidence.problem_id,
            artifact_uri=evidence.flags["answer_artifact_uri"],
            payload=evidence.answer_payload,
        )
        _write_json_once(
            attempt_dir / "request.json",
            {
                "schema_version": 1,
                "proposal_id": proposal.proposal_id,
                "prior_evidence_id": evidence.evidence_id,
                "answer_payload_hash": content_hash(evidence.answer_payload),
                "verifier_id": evidence.verifier_id,
                "verifier_version": evidence.verifier_version,
                "harness_id": evidence.harness_id,
                "policy_snapshot_id": evidence.policy_snapshot_id,
                "maximum_attempts": 2,
            },
        )
        last_result = None
        errors = []
        for attempt in range(1, 3):
            attempt_index = attempt - 1
            stem = f"attempt{attempt_index:02d}"
            evidence_path = attempt_dir / f"{stem}.evidence.json"
            state_path = attempt_dir / f"{stem}.state.json"
            descriptor_path = attempt_dir / f"{stem}.descriptor.json"
            completed_path = attempt_dir / f"{stem}.completed.json"
            error_path = attempt_dir / f"{stem}.error.json"
            if evidence_path.is_file() and error_path.is_file():
                raise EngineError(
                    "record confirmation attempt has both evidence and error artifacts"
                )
            if error_path.is_file():
                if state_path.exists() or descriptor_path.exists() or completed_path.exists():
                    raise EngineError(
                        "failed confirmation attempt has contradictory derived artifacts"
                    )
                error_document = json.loads(error_path.read_text(encoding="utf-8"))
                if (
                    error_document.get("schema_version") != 1
                    or error_document.get("attempt_index") != attempt_index
                ):
                    raise EngineError("record confirmation error artifact is malformed")
                errors.append(
                    f"{error_document.get('exception_type')}: "
                    f"{error_document.get('message')}"
                )
                continue
            if evidence_path.is_file():
                attempt_evidence = EvidencePacket.from_dict(
                    json.loads(evidence_path.read_text(encoding="utf-8"))
                )
                _validate_confirmation_binding(proposal, evidence, attempt_evidence)
                if _verification_attempt_index(
                    attempt_evidence.flags.get("verification_attempt_index"),
                    maximum=1,
                ) != attempt_index:
                    raise EngineError("record confirmation attempt index changed")
                result = restore_durable_verification_result(
                    evidence=attempt_evidence,
                    adapter=adapter,
                    state_path=state_path,
                    descriptor_path=descriptor_path,
                )
            else:
                if state_path.exists() or descriptor_path.exists() or completed_path.exists():
                    raise EngineError(
                        "record confirmation has derived artifacts without evidence"
                    )
                try:
                    result = confirm_persisted_answer(
                        adapter=adapter,
                        proposal=proposal,
                        persisted_answer=persisted,
                        prior_evidence=evidence,
                        verification_policy=verification_policy,
                        attempt_index=attempt_index,
                    )
                except Exception as exc:
                    _write_json_once(
                        error_path,
                        {
                            "schema_version": 1,
                            "attempt_index": attempt_index,
                            "exception_type": type(exc).__name__,
                            "message": str(exc)[:2048],
                        },
                    )
                    errors.append(f"{type(exc).__name__}: {exc}")
                    continue
                persist_verifier_trace(
                    run_dir=run_dir,
                    result=result,
                    phase=f"{phase}_attempt_{attempt_index}",
                )
                _write_json_once(evidence_path, result.evidence.to_dict())
                if result.state is not None:
                    _write_json_once(state_path, result.state.to_dict())
                    _write_json_once(
                        descriptor_path, result.descriptor.to_dict()
                    )
            _write_json_once(
                completed_path,
                {
                    "schema_version": 1,
                    "attempt_index": attempt_index,
                    "evidence_id": result.evidence.evidence_id,
                    "resolved": result.evidence.resolved,
                },
            )
            last_result = result
            if result.evidence.failure_kind != FailureKind.INFRASTRUCTURE:
                return result, attempt
        if last_result is not None:
            return last_result, 2
        raise RecordConfirmationInfrastructureError(
            "record confirmation failed as infrastructure on both bounded "
            "attempts: " + " | ".join(errors)
        )

    # -- one epoch -------------------------------------------------------

    def run_epoch(
        self,
        layout: RunLayout,
        state: EpochState,
        *,
        workers: EngineWorkers,
        adapter: ProblemScientificAdapter,
        verification_policy: VerificationPolicy,
    ) -> Tuple[EpochState, EpochReport]:
        settings = self.config.evolve
        epoch = state.epoch
        epoch_seed = derive_seed("epoch_seed", state.run_id, epoch, base_seed=self.config.seed)
        record_threshold = (
            float(state.record.internal_reward)
            if state.record.internal_reward is not None
            else float(getattr(adapter.problem, "fail_score", 0.0))
        )

        resource_limits = {
            # Keep two calls outside allocation for bounded confirmation of a
            # possible new record. Unused reserve remains available next epoch.
            "verifier_calls": max(
                0.0,
                state.budget_ledger.remaining("verifier_calls") - 2.0,
            ),
        }
        enabled_roles = [Role(name) for name in settings.roles.enabled]
        learning_role = enabled_roles[epoch % len(enabled_roles)]
        step_dir = layout.path(f"step{epoch:02d}")
        plan_path = step_dir / "allocation_plan.json"
        recovered_plan = plan_path.is_file()
        if recovered_plan:
            plan = _plan_from_document(
                json.loads(plan_path.read_text(encoding="utf-8"))
            )
            if plan.epoch != epoch or plan.seed != epoch_seed:
                raise EngineError(
                    "durable allocation plan belongs to another epoch or seed"
                )
            enabled_role_set = set(enabled_roles)
            for planned in plan.planned_arms:
                arm = planned.arm
                if arm.channel != Channel.PRODUCTION:
                    raise EngineError(
                        "authoritative epoch plan contains a non-production arm"
                    )
                state.archive.cell(arm.cell_id)
                if arm.role not in enabled_role_set:
                    raise EngineError(
                        "authoritative epoch plan references a disabled role"
                    )
                spec = state.option_registry.spec(arm.option_id)
                state.harness_registry.spec(arm.harness_id)
                if (
                    arm.horizon > spec.max_horizon
                    or arm.option_id
                    not in state.option_registry.eligible_for(
                        role=arm.role, harness_id=arm.harness_id
                    )
                ):
                    raise EngineError(
                        "authoritative epoch plan references an ineligible option"
                    )
        else:
            plan = plan_epoch(
                epoch=epoch,
                archive=state.archive,
                option_registry=state.option_registry,
                harness_registry=state.harness_registry,
                posterior=state.posterior,
                roles=enabled_roles,
                max_inflight_branches=settings.workers.max_inflight_branches,
                audit_fraction=settings.budget.audit_fraction,
                no_memory_fraction=settings.audits.no_memory_fraction,
                refinement_fraction=settings.budget.refinement_fraction,
                harness_trial_fraction=settings.harnesses.trial_fraction,
                empty_cell_fraction=settings.archive.empty_cell_fraction,
                global_exploration_fraction=settings.scheduler.global_exploration_fraction,
                resource_limits=resource_limits,
                seed=epoch_seed,
                learning_role=learning_role,
                group_k=settings.learning.group_k,
            )
        if not plan.planned_arms:
            raise EngineError(
                "scheduler produced no executable production allocation; "
                "refusing to commit an empty epoch"
            )

        role_snapshots = state.role_registry.freeze_epoch(epoch)
        role_registry = state.role_registry
        archive = state.archive
        posterior = state.posterior
        provenance = state.provenance
        budget_ledger = state.budget_ledger
        memory = state.memory
        harness_registry = state.harness_registry
        nursery = {
            entry_id: expire_entry(entry, epoch=epoch)
            for entry_id, entry in state.nursery.items()
        }
        record = state.record
        previous_record = state.record

        manifest = _build_epoch_manifest(
            run_id=state.run_id, epoch=epoch, record_threshold=record_threshold,
            archive=archive, posterior=posterior, memory=memory, role_registry=role_registry,
            plan=plan, option_registry=state.option_registry, harness_registry=state.harness_registry,
            verifier_id=adapter.verifier_id, verifier_version=adapter.verifier_version,
            budget_ledger=budget_ledger, seed=epoch_seed,
        )
        step_dir.mkdir(parents=True, exist_ok=True)
        _write_json_once(step_dir / "epoch.manifest.json", manifest.to_dict())
        if not recovered_plan:
            _write_json_once(plan_path, _plan_document(plan))

        executions: List[BranchExecution] = []
        group_members: List[GroupMember] = []
        arms_by_id: Dict[str, AllocationArm] = {}
        audit_pairs: List[AuditPair] = []
        audit_sides: Dict[str, AuditSide] = {}
        refinement_attempts: Dict[str, int] = {}
        refinement_sources: List[_RefinementSource] = []

        branch_ordinal = 0
        production_pending: List[_PendingBranch] = []
        for allocation_index, planned in enumerate(plan.planned_arms):
            arm = planned.arm
            arms_by_id[arm.arm_id] = arm
            replica_count = planned.replicas
            for replica_index in range(replica_count):
                index = branch_ordinal
                branch_ordinal += 1
                branch, role_snapshot = self._freeze_branch(
                    state=state,
                    arm=arm,
                    role_snapshots=role_snapshots,
                    record_threshold=record_threshold,
                    index=index,
                    epoch_seed=planned.rng_seed,
                    budget=arm.hard_cost,
                    channel=Channel.PRODUCTION,
                    verifier_id=adapter.verifier_id,
                    verifier_version=adapter.verifier_version,
                )
                debit_key = (
                    f"epoch{epoch}:allocation{allocation_index}:"
                    f"replica{replica_index}:verifier_calls"
                )
                try:
                    budget_ledger = BudgetService.debit(
                        budget_ledger,
                        resource="verifier_calls",
                        amount=float(arm.hard_cost.get("verifier_calls", 1.0)),
                        transaction_key=debit_key,
                    )
                except BudgetOverrun as exc:
                    raise EngineError(
                        "the persisted allocation plan exceeds the remaining "
                        "verifier budget; refusing to silently drop an allocation"
                    ) from exc
                production_pending.append(
                    _PendingBranch(
                        branch=branch,
                        arm=arm,
                        role_snapshot=role_snapshot,
                        ordinal=index,
                        debit_key=debit_key,
                        cell_empty=(
                            state.archive.cell(arm.cell_id).tested_count == 0
                        ),
                    )
                )

        for pending, execution in self._dispatch_pending_branches(
            production_pending,
            state=state,
            workers=workers,
            adapter=adapter,
            verification_policy=verification_policy,
            layout=layout,
            stage="production generation + verification",
        ):
            arm = pending.arm
            branch = pending.branch
            index = pending.ordinal
            debit_key = pending.debit_key
            budget_ledger = _settle_branch_verifier_budget(
                budget_ledger,
                execution=execution,
                debit_key=debit_key,
                reserved=float(arm.hard_cost.get("verifier_calls", 1.0)),
            )
            self._persist_branch_execution(
                layout, branch=branch, execution=execution, ordinal=index
            )
            executions.append(execution)
            already_entered = {
                entry.source_evidence_id for entry in nursery.values()
            }
            for observation in execution.observations:
                source_evidence = observation.verification.evidence
                if (
                    not source_evidence.admitted
                    and source_evidence.resolved
                    and source_evidence.evidence_id not in already_entered
                ):
                    refinement_sources.append(
                        _RefinementSource(
                            arm=arm,
                            proposal=observation.proposal,
                            evidence=source_evidence,
                            branch_id=branch.branch_id,
                        )
                    )
                    already_entered.add(source_evidence.evidence_id)
            # Fold each closed branch exactly once. Folding inside the
            # observation loop would multiply posterior observations and
            # provenance work by the option horizon.
            archive, provenance, record, posterior, budget_ledger = self._fold_execution(
                execution=execution,
                arm=arm,
                layout=layout,
                epoch=epoch,
                identity_archive=archive,
                provenance=provenance,
                record=record,
                posterior=posterior,
                record_threshold=record_threshold,
                adapter=adapter,
                verification_policy=verification_policy,
                budget_ledger=budget_ledger,
            )
            if (
                execution.policy_trace is not None
                and execution.outcome.eligible_for_scheduler
                and not execution.outcome.infrastructure_aborted
            ):
                group_members.append(
                    GroupMember(
                        branch=branch,
                        outcome=execution.outcome,
                        trace=execution.policy_trace,
                    )
                )

        # -- randomized audits (matched intervention vs. control option) --
        audit_slots = plan.reservation_slots.audit_branch_slots
        audit_candidates = [
            planned for planned in plan.planned_arms
            if state.option_registry.eligible_for(role=planned.arm.role, harness_id=planned.arm.harness_id)
        ]
        pairs_to_run = audit_slots // 2 if audit_candidates else 0
        for pair_index in range(pairs_to_run):
            planned = audit_candidates[pair_index % len(audit_candidates)]
            arm = planned.arm
            control_options = [
                option_id
                for option_id in state.option_registry.eligible_for(
                    role=arm.role, harness_id=arm.harness_id
                )
                if state.option_registry.spec(option_id).state_machine
                == "matched_continuation_v1"
            ]
            if not control_options:
                continue
            control_option_id = control_options[0]
            if control_option_id == arm.option_id:
                continue
            audit_seed = derive_seed("audit_seed", state.run_id, epoch, pair_index, base_seed=self.config.seed)
            intervention_option, control_option, probability = assign_audit_sides(
                option_a=arm.option_id, option_b=control_option_id, seed=audit_seed
            )
            intervention_arm = _retargeted_arm(
                arm, option_id=intervention_option, channel=Channel.AUDIT,
                option_registry=state.option_registry,
            )
            control_arm = _retargeted_arm(
                arm, option_id=control_option, channel=Channel.AUDIT,
                option_registry=state.option_registry,
            )
            treatment_slot = derive_seed(
                "audit_treatment_slot", audit_seed, pair_index
            ) % 2
            intervention_ordinal = 1000 + pair_index * 2 + treatment_slot
            control_ordinal = 1000 + pair_index * 2 + (1 - treatment_slot)
            intervention_branch, intervention_snapshot = self._freeze_branch(
                state=state, arm=intervention_arm, role_snapshots=role_snapshots,
                record_threshold=record_threshold, index=intervention_ordinal,
                epoch_seed=audit_seed, budget=arm.hard_cost, channel=Channel.AUDIT,
                verifier_id=adapter.verifier_id,
                verifier_version=adapter.verifier_version,
                memory_enabled=False,
            )
            control_branch, _ = self._freeze_branch(
                state=state, arm=control_arm, role_snapshots=role_snapshots,
                record_threshold=record_threshold, index=control_ordinal,
                epoch_seed=audit_seed, budget=arm.hard_cost, channel=Channel.AUDIT,
                shared_seed=intervention_branch.seed,
                verifier_id=adapter.verifier_id,
                verifier_version=adapter.verifier_version,
                memory_enabled=False,
            )
            try:
                pair = create_audit_pair(
                    run_id=state.run_id, cell_id=arm.cell_id,
                    intervention_branch=intervention_branch, control_branch=control_branch,
                    assignment_probability=probability, assignment_seed=audit_seed,
                )
            except AuditPairingError:
                continue

            audit_dir = step_dir / "audits" / pair.audit_id
            audit_dir.mkdir(parents=True, exist_ok=True)
            _write_json_once(audit_dir / "pair.preassigned.json", pair.to_dict())
            with ControllerEventWriter(layout.path("events.jsonl")) as event_writer:
                event_writer.append(
                    "audit_preassigned",
                    pair.to_dict(),
                    idempotency_key=f"audit-preassigned:{pair.audit_id}",
                )

            audit_debits = []
            required_audit_calls = sum(
                float(side_arm.hard_cost.get("verifier_calls", 1.0))
                for side_arm in (intervention_arm, control_arm)
            )
            if required_audit_calls > budget_ledger.remaining("verifier_calls"):
                aborted_pair = abort_audit_pair(pair)
                _write_json_once(
                    audit_dir / "pair.aborted.json",
                    {
                        **aborted_pair.to_dict(),
                        "abort_reason": "insufficient_verifier_budget",
                    },
                )
                audit_pairs.append(aborted_pair)
                continue
            audit_budget_available = True
            for side_name, side_arm in (
                ("intervention", intervention_arm),
                ("control", control_arm),
            ):
                debit_key = f"epoch{epoch}:audit{pair.audit_id}:{side_name}:verifier_calls"
                try:
                    budget_ledger = BudgetService.debit(
                        budget_ledger,
                        resource="verifier_calls",
                        amount=float(side_arm.hard_cost.get("verifier_calls", 1.0)),
                        transaction_key=debit_key,
                    )
                except BudgetOverrun:
                    audit_budget_available = False
                    break
                audit_debits.append((debit_key, side_arm))
            if not audit_budget_available:
                aborted_pair = abort_audit_pair(pair)
                _write_json_once(
                    audit_dir / "pair.aborted.json",
                    {
                        **aborted_pair.to_dict(),
                        "abort_reason": "verifier_budget_debit_failed",
                    },
                )
                audit_pairs.append(aborted_pair)
                continue

            audit_pending = [
                _PendingBranch(
                    branch=intervention_branch,
                    arm=intervention_arm,
                    role_snapshot=intervention_snapshot,
                    ordinal=intervention_ordinal,
                    debit_key=audit_debits[0][0],
                    memory_enabled=False,
                ),
                _PendingBranch(
                    branch=control_branch,
                    arm=control_arm,
                    role_snapshot=intervention_snapshot,
                    ordinal=control_ordinal,
                    debit_key=audit_debits[1][0],
                    memory_enabled=False,
                ),
            ]
            audit_results = self._dispatch_pending_branches(
                audit_pending,
                state=state,
                workers=workers,
                adapter=adapter,
                verification_policy=verification_policy,
                layout=layout,
                stage=f"randomized audit {pair_index + 1}/{pairs_to_run}",
            )
            audit_execution_by_branch = {
                pending.branch.branch_id: execution
                for pending, execution in audit_results
            }
            intervention_execution = audit_execution_by_branch[
                intervention_branch.branch_id
            ]
            control_execution = audit_execution_by_branch[control_branch.branch_id]
            self._persist_branch_execution(
                layout,
                branch=intervention_branch,
                execution=intervention_execution,
                ordinal=intervention_ordinal,
            )
            for (debit_key, side_arm), side_execution in zip(
                audit_debits, (intervention_execution, control_execution)
            ):
                budget_ledger = _settle_branch_verifier_budget(
                    budget_ledger,
                    execution=side_execution,
                    debit_key=debit_key,
                    reserved=float(
                        side_arm.hard_cost.get("verifier_calls", 1.0)
                    ),
                )
            self._persist_branch_execution(
                layout,
                branch=control_branch,
                execution=control_execution,
                ordinal=control_ordinal,
            )
            for execution, side_branch, side in (
                (intervention_execution, intervention_branch, AuditSide.INTERVENTION),
                (control_execution, control_branch, AuditSide.CONTROL),
            ):
                executions.append(execution)
                audit_sides[side_branch.branch_id] = side
                arms_by_id.setdefault(intervention_arm.arm_id, intervention_arm)
                arms_by_id.setdefault(control_arm.arm_id, control_arm)
                archive, provenance, record, posterior, budget_ledger = self._fold_execution(
                    execution=execution, arm=intervention_arm if side == AuditSide.INTERVENTION else control_arm,
                    layout=layout, epoch=epoch,
                    identity_archive=archive, provenance=provenance, record=record, posterior=posterior,
                    record_threshold=record_threshold, adapter=adapter, verification_policy=verification_policy,
                    budget_ledger=budget_ledger,
                )
                if (
                    execution.policy_trace is not None
                    and execution.outcome.eligible_for_scheduler
                    and not execution.outcome.infrastructure_aborted
                ):
                    group_members.append(
                        GroupMember(branch=side_branch, outcome=execution.outcome, trace=execution.policy_trace)
                    )

            try:
                closed_pair = close_audit_pair(
                    pair, intervention_outcome=intervention_execution.outcome,
                    control_outcome=control_execution.outcome,
                )
            except AuditEffectError:
                aborted_pair = abort_audit_pair(pair)
                _write_json_once(
                    audit_dir / "pair.aborted.json",
                    {
                        **aborted_pair.to_dict(),
                        "abort_reason": "ineligible_or_infrastructure_side",
                    },
                )
                audit_pairs.append(aborted_pair)
                continue
            audit_pairs.append(closed_pair)
            _write_json_once(audit_dir / "pair.closed.json", closed_pair.to_dict())

            intervention_gain = _normalized_outcome_gain(
                adapter,
                intervention_execution.outcome,
                frozen_record_threshold=record_threshold,
            )
            control_gain = _normalized_outcome_gain(
                adapter,
                control_execution.outcome,
                frozen_record_threshold=record_threshold,
            )
            effect = compute_audit_effect(
                closed_pair, intervention_gain=intervention_gain, control_gain=control_gain
            )
            _write_json_once(audit_dir / "effect.json", vars(effect))
            context = {"role": arm.role.value, "cell_id": arm.cell_id}
            record_id = memory_id_for(context=context, intervention_option_id=closed_pair.intervention_option_id)
            memory_record = memory.get(record_id) or new_memory_record(
                context=context, intervention_option_id=closed_pair.intervention_option_id,
                scope="cell", recency_epoch=epoch,
                promotion_min_support=settings.audits.min_pairs_for_promotion,
            )
            memory_record = add_effect(memory_record, pair=closed_pair, effect=effect, recency_epoch=epoch)
            memory_record = evaluate_promotion(memory_record)
            memory_record = stratify_drift(memory_record, current_epoch=epoch)
            memory = memory.upsert(memory_record)

        # -- bounded refinement nursery and equal-cost fresh controls -----
        repair_option_id = next(
            (
                option_id
                for option_id in state.option_registry.option_ids()
                if state.option_registry.spec(option_id).state_machine
                == DIAGNOSTIC_REPAIR_STATE_MACHINE
            ),
            None,
        )
        fresh_option_id = next(
            (
                option_id
                for option_id in state.option_registry.option_ids()
                if state.option_registry.spec(option_id).state_machine
                == FRESH_REFINEMENT_CONTROL_STATE_MACHINE
            ),
            None,
        )
        nursery_policy = NurseryPolicy(
            max_attempts=settings.refinement.max_attempts,
            max_depth=settings.refinement.max_depth,
            fixed_cost={"verifier_calls": 1.0},
            ttl_epochs=2,
        )
        queued_evidence_ids = {
            source.evidence.evidence_id for source in refinement_sources
        }
        for entry in sorted(nursery.values(), key=lambda item: item.entry_id):
            if not entry.can_attempt(epoch):
                continue
            evidence_id = entry.latest_evidence_id or entry.source_evidence_id
            proposal_id = entry.latest_proposal_id or entry.source_proposal_id
            if evidence_id in queued_evidence_ids:
                continue
            try:
                source_evidence = archive.artifacts.evidence_packet(evidence_id)
                source_proposal = archive.artifacts.proposal(proposal_id)
                parent_state_id = source_proposal.parent_state_id
                if parent_state_id is None:
                    continue
                parent_state = archive.artifacts.representative_state(parent_state_id)
                source_cell = next(
                    cell
                    for cell in archive.cells
                    if cell.descriptor_id == parent_state.descriptor_id
                )
                source_arm = next(
                    (
                        planned.arm
                        for planned in plan.planned_arms
                        if planned.arm.cell_id == source_cell.cell_id
                    ),
                    None,
                )
                if source_arm is None:
                    fallback_candidates = enumerate_candidate_arms(
                        archive=state.archive,
                        option_registry=state.option_registry,
                        harness_registry=state.harness_registry,
                        roles=(Role.CHALLENGER,),
                        cell_ids=(source_cell.cell_id,),
                    )
                    if not fallback_candidates:
                        continue
                    fallback = min(
                        fallback_candidates,
                        key=lambda item: item.identity.key(),
                    )
                    source_arm = make_allocation_arm(
                        fallback.identity,
                        expected_cost=fallback.expected_cost,
                        hard_cost=fallback.hard_cost,
                    )
            except (KeyError, StopIteration):
                continue
            refinement_sources.append(
                _RefinementSource(
                    arm=source_arm,
                    proposal=source_proposal,
                    evidence=source_evidence,
                    branch_id=entry.branch_id,
                    entry=entry,
                )
            )
            queued_evidence_ids.add(evidence_id)
        refinement_sources.sort(
            key=lambda source: (
                0 if source.entry is not None else 1,
                source.entry.entry_id if source.entry is not None else source.evidence.evidence_id,
            )
        )
        refinement_pairs_to_run = min(
            plan.reservation_slots.refinement_slots // 2,
            len(refinement_sources),
        )
        if repair_option_id is None or fresh_option_id is None:
            refinement_pairs_to_run = 0
        for pair_index in range(refinement_pairs_to_run):
            source = refinement_sources[pair_index]
            source_arm = source.arm
            source_evidence = source.evidence
            entry = source.entry or open_entry(
                source_evidence=source_evidence,
                branch_id=source.branch_id,
                epoch=epoch,
                policy=nursery_policy,
            )
            repair_base = _special_option_arm(
                source_arm,
                role=Role.CHALLENGER,
                option_id=repair_option_id,
                channel=Channel.REFINEMENT,
                option_registry=state.option_registry,
            )
            fresh_base = _special_option_arm(
                source_arm,
                role=Role.CHALLENGER,
                option_id=fresh_option_id,
                channel=Channel.REFINEMENT,
                option_registry=state.option_registry,
            )
            refinement_seed = derive_seed(
                "refinement_audit_seed",
                state.run_id,
                epoch,
                pair_index,
                source_evidence.evidence_id,
                base_seed=self.config.seed,
            )
            intervention_option, control_option, probability = assign_audit_sides(
                option_a=repair_option_id,
                option_b=fresh_option_id,
                seed=refinement_seed,
            )
            arms_by_option = {
                repair_option_id: repair_base,
                fresh_option_id: fresh_base,
            }
            intervention_arm = arms_by_option[intervention_option]
            control_arm = arms_by_option[control_option]
            treatment_slot = derive_seed(
                "refinement_treatment_slot", refinement_seed, entry.entry_id
            ) % 2
            intervention_ordinal = 3000 + pair_index * 2 + treatment_slot
            control_ordinal = 3000 + pair_index * 2 + (1 - treatment_slot)
            frozen_refinement = {
                "refinement_source": source.proposal.source_text,
                "refinement_source_evidence_id": source_evidence.evidence_id,
                "refinement_diagnostics": dict(source_evidence.diagnostics),
            }
            parent_state_id = source.proposal.parent_state_id
            if parent_state_id is None:
                continue
            intervention_branch, challenger_snapshot = self._freeze_branch(
                state=state,
                arm=intervention_arm,
                role_snapshots=role_snapshots,
                record_threshold=record_threshold,
                index=intervention_ordinal,
                epoch_seed=refinement_seed,
                budget=intervention_arm.hard_cost,
                channel=Channel.REFINEMENT,
                verifier_id=adapter.verifier_id,
                verifier_version=adapter.verifier_version,
                memory_enabled=False,
                start_state_id=parent_state_id,
                generation_overrides=frozen_refinement,
            )
            control_branch, _ = self._freeze_branch(
                state=state,
                arm=control_arm,
                role_snapshots=role_snapshots,
                record_threshold=record_threshold,
                index=control_ordinal,
                epoch_seed=refinement_seed,
                budget=control_arm.hard_cost,
                channel=Channel.REFINEMENT,
                verifier_id=adapter.verifier_id,
                verifier_version=adapter.verifier_version,
                shared_seed=intervention_branch.seed,
                memory_enabled=False,
                start_state_id=parent_state_id,
                generation_overrides=frozen_refinement,
            )
            pair = create_audit_pair(
                run_id=state.run_id,
                cell_id=source_arm.cell_id,
                intervention_branch=intervention_branch,
                control_branch=control_branch,
                assignment_probability=probability,
                assignment_seed=refinement_seed,
            )
            refinement_dir = step_dir / "refinement" / entry.entry_id
            refinement_dir.mkdir(parents=True, exist_ok=True)
            _write_json_once(
                refinement_dir / "entry.opened.json", vars(entry)
            )
            _write_json_once(
                refinement_dir / "pair.preassigned.json", pair.to_dict()
            )
            required = sum(
                float(item.hard_cost.get("verifier_calls", 1.0))
                for item in (intervention_arm, control_arm)
            )
            if required > budget_ledger.remaining("verifier_calls"):
                nursery[entry.entry_id] = entry
                aborted_pair = abort_audit_pair(pair)
                _write_json_once(
                    refinement_dir / "pair.aborted.json",
                    {
                        **aborted_pair.to_dict(),
                        "abort_reason": "insufficient_verifier_budget",
                    },
                )
                audit_pairs.append(aborted_pair)
                continue
            debit_records = []
            for side, refinement_arm in (
                ("intervention", intervention_arm),
                ("control", control_arm),
            ):
                key = (
                    f"epoch{epoch}:refinement{pair.audit_id}:"
                    f"{side}:verifier_calls"
                )
                budget_ledger = BudgetService.debit(
                    budget_ledger,
                    resource="verifier_calls",
                    amount=float(
                        refinement_arm.hard_cost.get("verifier_calls", 1.0)
                    ),
                    transaction_key=key,
                )
                debit_records.append(key)
            refinement_pending = [
                _PendingBranch(
                    branch=intervention_branch,
                    arm=intervention_arm,
                    role_snapshot=challenger_snapshot,
                    ordinal=intervention_ordinal,
                    debit_key=debit_records[0],
                    memory_enabled=False,
                ),
                _PendingBranch(
                    branch=control_branch,
                    arm=control_arm,
                    role_snapshot=challenger_snapshot,
                    ordinal=control_ordinal,
                    debit_key=debit_records[1],
                    memory_enabled=False,
                ),
            ]
            refinement_results = self._dispatch_pending_branches(
                refinement_pending,
                state=state,
                workers=workers,
                adapter=adapter,
                verification_policy=verification_policy,
                layout=layout,
                stage=(
                    f"refinement audit {pair_index + 1}/"
                    f"{refinement_pairs_to_run}"
                ),
            )
            refinement_execution_by_branch = {
                pending.branch.branch_id: execution
                for pending, execution in refinement_results
            }
            intervention_execution = refinement_execution_by_branch[
                intervention_branch.branch_id
            ]
            control_execution = refinement_execution_by_branch[
                control_branch.branch_id
            ]
            for refinement_branch, refinement_arm, execution, ordinal in (
                (
                    intervention_branch,
                    intervention_arm,
                    intervention_execution,
                    intervention_ordinal,
                ),
                (
                    control_branch,
                    control_arm,
                    control_execution,
                    control_ordinal,
                )
            ):
                self._persist_branch_execution(
                    layout,
                    branch=refinement_branch,
                    execution=execution,
                    ordinal=ordinal,
                )
                executions.append(execution)
                arms_by_id[refinement_arm.arm_id] = refinement_arm
                archive, provenance, record, posterior, budget_ledger = self._fold_execution(
                    execution=execution,
                    arm=refinement_arm,
                    layout=layout,
                    epoch=epoch,
                    identity_archive=archive,
                    provenance=provenance,
                    record=record,
                    posterior=posterior,
                    record_threshold=record_threshold,
                    adapter=adapter,
                    verification_policy=verification_policy,
                    budget_ledger=budget_ledger,
                )
                if (
                    execution.policy_trace is not None
                    and execution.outcome.eligible_for_scheduler
                    and not execution.outcome.infrastructure_aborted
                ):
                    refinement_attempts[refinement_branch.branch_id] = (
                        entry.attempts_used + 1
                    )
                    group_members.append(
                        GroupMember(
                            branch=refinement_branch,
                            outcome=execution.outcome,
                            trace=execution.policy_trace,
                        )
                    )
            for debit_key, refinement_arm, execution in zip(
                debit_records,
                (intervention_arm, control_arm),
                (intervention_execution, control_execution),
            ):
                budget_ledger = _settle_branch_verifier_budget(
                    budget_ledger,
                    execution=execution,
                    debit_key=debit_key,
                    reserved=float(
                        refinement_arm.hard_cost.get("verifier_calls", 1.0)
                    ),
                )
            repair_execution = (
                intervention_execution
                if intervention_arm.option_id == repair_option_id
                else control_execution
            )
            if (
                repair_execution.observations
                and not repair_execution.outcome.infrastructure_aborted
            ):
                repair_evidence = repair_execution.observations[-1].verification.evidence
                if not repair_evidence.resolved:
                    raise EngineError(
                        "a scheduler-eligible repair produced unresolved evidence"
                    )
                entry = record_attempt(
                    entry,
                    repair_evidence=repair_evidence,
                    epoch=epoch,
                )
            nursery[entry.entry_id] = entry
            _write_json_once(
                refinement_dir / "entry.after_attempt.json", vars(entry)
            )
            try:
                closed_pair = close_audit_pair(
                    pair,
                    intervention_outcome=intervention_execution.outcome,
                    control_outcome=control_execution.outcome,
                )
            except AuditEffectError:
                aborted_pair = abort_audit_pair(pair)
                _write_json_once(
                    refinement_dir / "pair.aborted.json",
                    {
                        **aborted_pair.to_dict(),
                        "abort_reason": "ineligible_or_infrastructure_side",
                    },
                )
                audit_pairs.append(aborted_pair)
                continue
            audit_pairs.append(closed_pair)
            _write_json_once(
                refinement_dir / "pair.closed.json", closed_pair.to_dict()
            )
            effect = compute_audit_effect(
                closed_pair,
                intervention_gain=_normalized_outcome_gain(
                    adapter,
                    intervention_execution.outcome,
                    frozen_record_threshold=record_threshold,
                ),
                control_gain=_normalized_outcome_gain(
                    adapter,
                    control_execution.outcome,
                    frozen_record_threshold=record_threshold,
                ),
            )
            _write_json_once(refinement_dir / "effect.json", vars(effect))
            context = {
                "role": Role.CHALLENGER.value,
                "cell_id": source_arm.cell_id,
                "channel": Channel.REFINEMENT.value,
            }
            memory_record_id = memory_id_for(
                context=context,
                intervention_option_id=closed_pair.intervention_option_id,
            )
            memory_record = memory.get(memory_record_id) or new_memory_record(
                context=context,
                intervention_option_id=closed_pair.intervention_option_id,
                scope="refinement_cell",
                recency_epoch=epoch,
                promotion_min_support=settings.audits.min_pairs_for_promotion,
            )
            memory_record = add_effect(
                memory_record,
                pair=closed_pair,
                effect=effect,
                recency_epoch=epoch,
            )
            memory_record = evaluate_promotion(memory_record)
            memory_record = stratify_drift(
                memory_record, current_epoch=epoch
            )
            memory = memory.upsert(memory_record)

        # -- matched harness calibration ---------------------------------
        inactive_harnesses = tuple(
            harness_id
            for harness_id in sorted(harness_registry.specs)
            if harness_id not in harness_registry.active_ids
        )
        harness_pairs_to_run = min(
            plan.reservation_slots.harness_trial_slots // 2,
            (
                len(plan.planned_arms) * len(inactive_harnesses)
                if plan.planned_arms and inactive_harnesses
                else 0
            ),
        )
        for pair_index in range(harness_pairs_to_run):
            base_arm = plan.planned_arms[pair_index % len(plan.planned_arms)].arm
            candidate_harness_id = inactive_harnesses[
                pair_index % len(inactive_harnesses)
            ]
            incumbent_arm = _retargeted_harness_arm(
                base_arm,
                harness_id=base_arm.harness_id,
                channel=Channel.AUDIT,
            )
            candidate_arm = _retargeted_harness_arm(
                base_arm,
                harness_id=candidate_harness_id,
                channel=Channel.AUDIT,
            )
            trial_seed = derive_seed(
                "harness_trial_seed",
                state.run_id,
                epoch,
                pair_index,
                base_seed=self.config.seed,
            )
            candidate_slot = derive_seed(
                "harness_candidate_slot", trial_seed, candidate_harness_id
            ) % 2
            candidate_ordinal = 2000 + pair_index * 2 + candidate_slot
            incumbent_ordinal = 2000 + pair_index * 2 + (1 - candidate_slot)
            incumbent_branch, snapshot = self._freeze_branch(
                state=state,
                arm=incumbent_arm,
                role_snapshots=role_snapshots,
                record_threshold=record_threshold,
                index=incumbent_ordinal,
                epoch_seed=trial_seed,
                budget=incumbent_arm.hard_cost,
                channel=Channel.AUDIT,
                verifier_id=adapter.verifier_id,
                verifier_version=adapter.verifier_version,
                memory_enabled=False,
            )
            candidate_branch, _ = self._freeze_branch(
                state=state,
                arm=candidate_arm,
                role_snapshots=role_snapshots,
                record_threshold=record_threshold,
                index=candidate_ordinal,
                epoch_seed=trial_seed,
                budget=candidate_arm.hard_cost,
                channel=Channel.AUDIT,
                verifier_id=adapter.verifier_id,
                verifier_version=adapter.verifier_version,
                shared_seed=incumbent_branch.seed,
                memory_enabled=False,
            )
            context = MatchedHarnessAuditContext.create(
                incumbent_branch=incumbent_branch,
                incumbent_arm=incumbent_arm,
                candidate_branch=candidate_branch,
                candidate_arm=candidate_arm,
                assignment_probability=0.5,
                assignment_seed=trial_seed,
            )
            harness_dir = step_dir / "audits" / context.context_id
            harness_dir.mkdir(parents=True, exist_ok=True)
            _write_json_once(
                harness_dir / "harness.preassigned.json", context.to_dict()
            )
            required = sum(
                float(item.hard_cost.get("verifier_calls", 1.0))
                for item in (incumbent_arm, candidate_arm)
            )
            if required > budget_ledger.remaining("verifier_calls"):
                _write_json_once(
                    harness_dir / "harness.aborted.json",
                    {
                        "schema_version": 1,
                        "context_id": context.context_id,
                        "reason": "insufficient_verifier_budget",
                    },
                )
                continue
            debit_records = []
            for label, trial_arm in (
                ("incumbent", incumbent_arm),
                ("candidate", candidate_arm),
            ):
                key = (
                    f"epoch{epoch}:harness{context.context_id}:"
                    f"{label}:verifier_calls"
                )
                budget_ledger = BudgetService.debit(
                    budget_ledger,
                    resource="verifier_calls",
                    amount=float(
                        trial_arm.hard_cost.get("verifier_calls", 1.0)
                    ),
                    transaction_key=key,
                )
                debit_records.append((key, trial_arm))

            harness_pending = [
                _PendingBranch(
                    branch=incumbent_branch,
                    arm=incumbent_arm,
                    role_snapshot=snapshot,
                    ordinal=incumbent_ordinal,
                    debit_key=debit_records[0][0],
                    memory_enabled=False,
                ),
                _PendingBranch(
                    branch=candidate_branch,
                    arm=candidate_arm,
                    role_snapshot=snapshot,
                    ordinal=candidate_ordinal,
                    debit_key=debit_records[1][0],
                    memory_enabled=False,
                ),
            ]
            harness_results = self._dispatch_pending_branches(
                harness_pending,
                state=state,
                workers=workers,
                adapter=adapter,
                verification_policy=verification_policy,
                layout=layout,
                stage=(
                    f"harness calibration {pair_index + 1}/"
                    f"{harness_pairs_to_run}"
                ),
            )
            harness_execution_by_branch = {
                pending.branch.branch_id: execution
                for pending, execution in harness_results
            }
            incumbent_execution = harness_execution_by_branch[
                incumbent_branch.branch_id
            ]
            candidate_execution = harness_execution_by_branch[
                candidate_branch.branch_id
            ]
            for trial_branch, trial_arm, trial_execution, ordinal in (
                (
                    incumbent_branch,
                    incumbent_arm,
                    incumbent_execution,
                    incumbent_ordinal,
                ),
                (
                    candidate_branch,
                    candidate_arm,
                    candidate_execution,
                    candidate_ordinal,
                ),
            ):
                self._persist_branch_execution(
                    layout,
                    branch=trial_branch,
                    execution=trial_execution,
                    ordinal=ordinal,
                )
                executions.append(trial_execution)
                arms_by_id[trial_arm.arm_id] = trial_arm
                archive, provenance, record, posterior, budget_ledger = self._fold_execution(
                    execution=trial_execution,
                    arm=trial_arm,
                    layout=layout,
                    epoch=epoch,
                    identity_archive=archive,
                    provenance=provenance,
                    record=record,
                    posterior=posterior,
                    record_threshold=record_threshold,
                    adapter=adapter,
                    verification_policy=verification_policy,
                    budget_ledger=budget_ledger,
                )
            for (debit_key, trial_arm), trial_execution in zip(
                debit_records, (incumbent_execution, candidate_execution)
            ):
                budget_ledger = _settle_branch_verifier_budget(
                    budget_ledger,
                    execution=trial_execution,
                    debit_key=debit_key,
                    reserved=float(
                        trial_arm.hard_cost.get("verifier_calls", 1.0)
                    ),
                )
            trial = HarnessTrialRecord.from_context(
                context,
                epoch=epoch,
                incumbent_gain=_normalized_outcome_gain(
                    adapter,
                    incumbent_execution.outcome,
                    frozen_record_threshold=record_threshold,
                ),
                candidate_gain=_normalized_outcome_gain(
                    adapter,
                    candidate_execution.outcome,
                    frozen_record_threshold=record_threshold,
                ),
                incumbent_cost=incumbent_execution.outcome.costs,
                candidate_cost=candidate_execution.outcome.costs,
                incumbent_valid=(
                    incumbent_execution.outcome.maximum_state_id is not None
                ),
                candidate_valid=(
                    candidate_execution.outcome.maximum_state_id is not None
                ),
                incumbent_failure_kind=(
                    "infrastructure"
                    if incumbent_execution.outcome.infrastructure_aborted
                    else "none"
                ),
                candidate_failure_kind=(
                    "infrastructure"
                    if candidate_execution.outcome.infrastructure_aborted
                    else "none"
                ),
            )
            _write_json_once(
                harness_dir / "harness.result.json", trial.to_dict()
            )
            harness_registry = harness_registry.record_trial(trial)
            promotion_decision = {
                "schema_version": 1,
                "context_id": context.context_id,
                "candidate_harness_id": candidate_harness_id,
                "min_trials": settings.audits.min_pairs_for_promotion,
                "approved": False,
                "reason": "conservative_evidence_threshold_not_met",
            }
            try:
                harness_registry = harness_registry.promote(
                    candidate_harness_id,
                    min_trials=settings.audits.min_pairs_for_promotion,
                )
                promotion_decision.update(
                    {"approved": True, "reason": "conservative_effect_positive"}
                )
            except HarnessPromotionError as exc:
                promotion_decision["detail"] = str(exc)[:2048]
            _write_json_once(
                harness_dir / "harness.promotion.json", promotion_decision
            )

        # Confirm the best provisional observation only after every scheduled
        # branch has closed. This makes the committed record independent of
        # worker completion order and verifies the saved answer payload rather
        # than rerunning proposal code.
        self._print_progress(
            "record confirmation",
            epoch=epoch,
            detail="checking the best saved provisional payload",
        )
        archive, record, budget_ledger = self._confirm_epoch_record(
            layout=layout,
            epoch=epoch,
            executions=executions,
            archive=archive,
            record=record,
            budget_ledger=budget_ledger,
            frozen_record_threshold=record_threshold,
            adapter=adapter,
            verification_policy=verification_policy,
        )

        # -- role-isolated learning ------------------------------------
        traces_by_id = {member.trace.trace_id: member.trace for member in group_members}
        groups = build_learning_groups(
            group_members,
            arms=arms_by_id,
            objective=_objective_enum(settings.learning.objective),
            top_m=settings.learning.top_m,
            group_k=settings.learning.group_k,
            audit_sides=audit_sides,
            refinement_attempts=refinement_attempts,
            gain_fn=lambda outcome, threshold: _normalized_outcome_gain(
                adapter, outcome, frozen_record_threshold=threshold
            ),
        )
        learning_dir = step_dir / "learning"
        learning_dir.mkdir(parents=True, exist_ok=True)
        for group in groups:
            _write_json_once(learning_dir / f"{group.group_id}.inputs.json", group.to_dict())
        for trace in traces_by_id.values():
            _write_json_once(learning_dir / f"{trace.trace_id}.trace.json", trace.to_dict())
        self._print_progress(
            "role learning",
            epoch=epoch,
            completed=0,
            total=len(groups),
            unit="groups",
            detail="running homogeneous on-policy updates",
        )
        updates, role_registry = train_barrier(
            groups, traces_by_id=traces_by_id, registry=role_registry, epoch=epoch,
            gradient_step=workers.gradient_step, kl_penalty_coef=self.config.kl_penalty_coef,
        )
        for update in updates:
            _write_json_once(
                learning_dir / f"{update.role_snapshot_id_before}.update.json",
                {
                    "schema_version": 1,
                    "role": update.groups[0].role.value,
                    "role_snapshot_id_before": update.role_snapshot_id_before,
                    "adapter_hash_before": update.adapter_hash_before,
                    "adapter_hash_after": update.adapter_hash_after,
                    "objective": update.groups[0].objective.value,
                    "objective_version": update.groups[0].objective_version,
                    "group_ids": [group.group_id for group in update.groups],
                    "group_members": [
                        {
                            "group_id": group.group_id,
                            "branch_ids": list(group.branch_ids),
                            "trace_ids": list(group.trace_ids),
                            "outcome_ids": list(group.outcome_ids),
                            "advantages": list(group.advantages),
                        }
                        for group in update.groups
                    ],
                    "token_masks": {
                        trace_id: [list(mask) for mask in traces_by_id[trace_id].token_masks]
                        for group in update.groups
                        for trace_id in group.trace_ids
                    },
                    "loss": update.result.loss,
                    "kl": update.result.kl,
                    "gradient_norm": update.result.gradient_norm,
                    "optimizer_state": dict(update.result.optimizer_state),
                },
            )
        self._print_progress(
            "role learning",
            epoch=epoch,
            completed=len(groups),
            total=len(groups),
            unit="groups",
            detail=f"{len(updates)} role updates persisted",
        )

        record_improved = (
            record.state_id is not None
            and record.internal_reward is not None
            and (
                previous_record.internal_reward is None
                or record.internal_reward > previous_record.internal_reward
            )
        )
        state = replace(
            state,
            epoch=epoch + 1,
            archive=archive,
            provenance=provenance,
            posterior=posterior,
            memory=memory,
            harness_registry=harness_registry,
            nursery=nursery,
            role_registry=role_registry,
            budget_ledger=budget_ledger,
            record=record,
        )
        report = EpochReport(
            epoch=epoch, plan=plan, branch_executions=tuple(executions),
            audit_pairs=tuple(audit_pairs), record_improved=record_improved,
        )
        return state, report

    def _persist_branch_execution(
        self,
        layout: RunLayout,
        *,
        branch: BranchSpec,
        execution: BranchExecution,
        ordinal: int,
    ) -> None:
        """Durably save every branch input and observation before it is consumed."""

        step_dir = layout.path(f"step{branch.epoch:02d}")
        branch_dir = step_dir / "branches" / branch.branch_id
        branch_dir.mkdir(parents=True, exist_ok=True)
        _write_json_once(branch_dir / "branch.spec.json", branch.to_dict())
        for observation_index, observation in enumerate(execution.observations):
            prefix = f"observation{observation_index:03d}"
            segment = observation.policy_segment
            if segment is not None:
                _write_text_once(branch_dir / f"{prefix}.prompt.txt", segment.prompt)
                _write_text_once(
                    branch_dir / f"{prefix}.response.txt", segment.response_segment
                )
            else:
                _write_text_once(
                    branch_dir / f"{prefix}.response.txt",
                    observation.proposal.source_text,
                )
            _write_json_once(
                branch_dir / f"{prefix}.proposal.json",
                observation.proposal.to_dict(),
            )
            _write_json_once(
                branch_dir / f"{prefix}.evidence.json",
                observation.verification.evidence.to_dict(),
            )
            if observation.verification.state is not None:
                _write_json_once(
                    branch_dir / f"{prefix}.state.json",
                    observation.verification.state.to_dict(),
                )
            flat = f"step{branch.epoch:02d}_group{ordinal:04d}_rollout{observation_index:03d}"
            if segment is not None:
                _write_text_once(step_dir / f"{flat}.prompt.txt", segment.prompt)
            _write_text_once(step_dir / f"{flat}.txt", observation.proposal.source_text)
            _write_json_once(
                step_dir / f"{flat}.meta.json",
                {
                    "branch_id": branch.branch_id,
                    "proposal_id": observation.proposal.proposal_id,
                    "evidence_id": observation.verification.evidence.evidence_id,
                    "failure_kind": observation.verification.evidence.failure_kind.value,
                    "admitted": observation.verification.evidence.admitted,
                    "costs": dict(observation.costs),
                },
            )
        _write_json_once(branch_dir / "branch.outcome.json", execution.outcome.to_dict())
        if execution.policy_trace is not None:
            _write_json_once(
                branch_dir / "policy.trace.json", execution.policy_trace.to_dict()
            )
        with ControllerEventWriter(layout.path("events.jsonl")) as event_writer:
            event_writer.append(
                "branch_closed",
                {
                    "branch": branch.to_dict(),
                    "outcome": execution.outcome.to_dict(),
                },
                idempotency_key=f"branch-closed:{branch.branch_id}",
            )

    def _freeze_branch(
        self,
        *,
        state: EpochState,
        arm: AllocationArm,
        role_snapshots: Mapping[Role, Any],
        record_threshold: float,
        index: int,
        epoch_seed: int,
        budget: Mapping[str, float],
        channel: Channel,
        verifier_id: str,
        verifier_version: str,
        shared_seed: Optional[int] = None,
        memory_enabled: bool = True,
        start_state_id: Optional[str] = None,
        generation_overrides: Optional[Mapping[str, Any]] = None,
    ) -> Tuple[BranchSpec, Any]:
        role_snapshot = role_snapshots[arm.role]
        frozen_start_state_id = start_state_id or self._pick_start_state(
            state.archive, arm.cell_id
        )
        seed = shared_seed if shared_seed is not None else rollout_seed(
            run_id=state.run_id, epoch=state.epoch, allocation_id=arm.arm_id,
            branch_step=index, sample_index=0, role=arm.role.value, base_seed=self.config.seed,
        )
        branch_id = content_id(
            "branch",
            {"run_id": state.run_id, "epoch": state.epoch, "arm_id": arm.arm_id, "index": index, "channel": channel.value},
        )
        memory_context = {
            "role": arm.role.value,
            "cell_id": arm.cell_id,
        }
        memory_records = (
            state.memory.promoted_for_context(
                context=memory_context,
                intervention_option_id=arm.option_id,
            )
            if memory_enabled
            else ()
        )
        memory_payload = [record.to_dict() for record in memory_records]
        memory_view_id = (
            content_id(
                "memory_view",
                {
                    "snapshot": _memory_snapshot_id(state.memory),
                    "records": [record.memory_id for record in memory_records],
                    "context": memory_context,
                },
            )
            if memory_records
            else None
        )
        generation_settings = {
            "role": arm.role.value,
            "temperature": self.config.temperature,
            "top_p": self.config.top_p,
            "max_new_tokens": self.config.max_new_tokens,
            "memory_records": memory_payload,
        }
        if generation_overrides:
            generation_settings.update(dict(generation_overrides))
        branch = BranchSpec(
            branch_id=branch_id,
            arm_id=arm.arm_id,
            epoch=state.epoch,
            start_state_id=frozen_start_state_id,
            frozen_record_threshold=record_threshold,
            role_snapshot_id=role_snapshot.snapshot_id,
            option_id=arm.option_id,
            option_version=state.option_registry.spec(arm.option_id).version,
            harness_id=arm.harness_id,
            harness_version=state.harness_registry.spec(arm.harness_id).version,
            verifier_id=verifier_id,
            verifier_version=verifier_version,
            memory_view_id=memory_view_id,
            memory_view_hash=content_hash(
                {"context": memory_context, "records": memory_payload}
            ),
            horizon=arm.horizon,
            budget=dict(budget),
            seed=seed,
            generation_settings=generation_settings,
            channel=channel,
        )
        return branch, role_snapshot

    def _pick_start_state(self, archive: ScientificArchive, cell_id: str) -> str:
        cell = archive.cell(cell_id)
        if cell.champion_state_id is not None:
            return cell.champion_state_id
        if cell.promising_state_ids:
            return cell.promising_state_ids[0]
        if cell.stepping_stone_state_ids:
            return cell.stepping_stone_state_ids[0]
        # Empty-cell exploration still freezes a real verified source state;
        # the arm's cell remains the target descriptor cell. Prefer the global
        # confirmed champion as the reproducible launch point.
        champions = []
        for candidate_cell in archive.cells:
            if candidate_cell.champion_state_id is None:
                continue
            state = archive.artifacts.representative_state(
                candidate_cell.champion_state_id
            )
            champions.append((float(state.internal_reward), state.state_id))
        if champions:
            return max(champions, key=lambda item: (item[0], item[1]))[1]
        raise EngineError(
            f"cell {cell_id} is empty and the archive has no verified launch state"
        )

    def _execute_one_branch(
        self,
        *,
        branch: BranchSpec,
        arm: AllocationArm,
        role_snapshot: Any,
        option_registry: OptionRegistry,
        workers: EngineWorkers,
        start_verified: bool,
        cell_empty: bool,
        memory_enabled: bool = True,
    ) -> BranchExecution:
        option = option_registry.create(arm.option_id)
        context = build_option_context(
            branch=branch, arm=arm, start_verified=start_verified,
            cell_empty=cell_empty, memory_enabled=memory_enabled,
            satisfied_prerequisites=("verified_start",) if start_verified else (),
        )
        return execute_branch(
            branch=branch, arm=arm, option=option, context=context,
            role_snapshot=role_snapshot, executor=workers.branch_step_executor,
        )

    def _persist_branch_assignment(
        self,
        layout: RunLayout,
        pending: _PendingBranch,
    ) -> None:
        """Make a frozen assignment durable before any worker may execute it."""

        branch_dir = layout.path(
            f"step{pending.branch.epoch:02d}/branches/{pending.branch.branch_id}"
        )
        branch_dir.mkdir(parents=True, exist_ok=True)
        _write_json_once(branch_dir / "branch.spec.json", pending.branch.to_dict())
        _write_json_once(
            branch_dir / "assignment.json",
            {
                "schema_version": 1,
                "branch_id": pending.branch.branch_id,
                "arm": pending.arm.to_dict(),
                "role_snapshot": pending.role_snapshot.to_dict(),
                "ordinal": pending.ordinal,
                "debit_key": pending.debit_key,
                "memory_enabled": pending.memory_enabled,
            },
        )

    def _execute_pending_branch(
        self,
        pending: _PendingBranch,
        *,
        state: EpochState,
        workers: EngineWorkers,
        adapter: ProblemScientificAdapter,
        verification_policy: VerificationPolicy,
        layout: RunLayout,
    ) -> BranchExecution:
        # The production worker boundary converts generation/verifier failures
        # into explicit unresolved evidence. Exceptions escaping that boundary
        # are invariant, schema, or persistence failures and must stop recovery
        # instead of being disguised as a low-quality scientific observation.
        return self._execute_one_branch(
            branch=pending.branch,
            arm=pending.arm,
            role_snapshot=pending.role_snapshot,
            option_registry=state.option_registry,
            workers=workers,
            start_verified=True,
            cell_empty=pending.cell_empty,
            memory_enabled=pending.memory_enabled,
        )

    def _dispatch_pending_branches(
        self,
        pending: List[_PendingBranch],
        *,
        state: EpochState,
        workers: EngineWorkers,
        adapter: ProblemScientificAdapter,
        verification_policy: VerificationPolicy,
        layout: RunLayout,
        stage: str,
    ) -> List[Tuple[_PendingBranch, BranchExecution]]:
        """Run a bounded phase and yield completed branches in arrival order."""

        if not pending:
            return []
        for item in pending:
            self._persist_branch_assignment(layout, item)
        epoch = pending[0].branch.epoch
        self._print_progress(
            stage,
            epoch=epoch,
            completed=0,
            total=len(pending),
            unit="branches",
            detail="generation and verification active",
        )
        self._write_live_progress_status(
            layout,
            state,
            epoch=epoch,
            stage=stage,
            completed_branches=0,
            total_branches=len(pending),
            completed_verifications=0,
            executions=(),
        )
        completed: List[Tuple[_PendingBranch, BranchExecution]] = []
        completed_verifications = 0

        def record_completion(
            item: _PendingBranch, execution: BranchExecution
        ) -> None:
            nonlocal completed_verifications
            completed.append((item, execution))
            completed_verifications += len(execution.observations)
            self._print_progress(
                stage,
                epoch=epoch,
                completed=len(completed),
                total=len(pending),
                unit="branches",
                detail=f"{completed_verifications} verifications completed",
            )
            self._write_live_progress_status(
                layout,
                state,
                epoch=epoch,
                stage=stage,
                completed_branches=len(completed),
                total_branches=len(pending),
                completed_verifications=completed_verifications,
                executions=tuple(result for _, result in completed),
            )

        if workers.submit_branch is None:
            for item in pending:
                execution = self._execute_pending_branch(
                    item,
                    state=state,
                    workers=workers,
                    adapter=adapter,
                    verification_policy=verification_policy,
                    layout=layout,
                )
                record_completion(item, execution)
            return completed

        by_future = {}
        for item in pending:
            future = workers.submit_branch(
                lambda item=item: self._execute_pending_branch(
                    item,
                    state=state,
                    workers=workers,
                    adapter=adapter,
                    verification_policy=verification_policy,
                    layout=layout,
                )
            )
            by_future[future] = item
        for future in as_completed(tuple(by_future)):
            item = by_future[future]
            execution = future.result()
            self._persist_branch_execution(
                layout,
                branch=item.branch,
                execution=execution,
                ordinal=item.ordinal,
            )
            record_completion(item, execution)
        return completed

    def _fold_execution(
        self,
        *,
        execution: BranchExecution,
        arm: AllocationArm,
        layout: RunLayout,
        epoch: int,
        identity_archive: ScientificArchive,
        provenance: ProvenanceStore,
        record: ConfirmedRecordTracker,
        posterior: PosteriorStore,
        record_threshold: float,
        adapter: ProblemScientificAdapter,
        verification_policy: VerificationPolicy,
        budget_ledger: BudgetLedger,
    ) -> Tuple[
        ScientificArchive,
        ProvenanceStore,
        ConfirmedRecordTracker,
        PosteriorStore,
        BudgetLedger,
    ]:
        archive = identity_archive
        provenance = provenance.with_branch(execution.outcome.branch_id)
        for result in execution.observations:
            evidence = result.verification.evidence
            archive = replace(
                archive,
                artifacts=archive.artifacts.add_observation(result.proposal, evidence),
            )
            if not evidence.admitted or result.verification.state is None:
                continue
            descriptor = result.verification.descriptor
            archive = archive.ensure_cell(descriptor, force_empty_sampling=False)
            try:
                archive, decision = archive.offer(
                    descriptor, result.proposal, result.verification.state, evidence
                )
            except ArchiveAdmissionError as exc:
                _write_json_once(
                    layout.path(
                        f"step{epoch:02d}/branches/{execution.outcome.branch_id}/"
                        f"archive/{evidence.evidence_id}.rejected.json"
                    ),
                    {
                        "schema_version": 1,
                        "exception_type": type(exc).__name__,
                        "message": str(exc),
                    },
                )
                raise EngineError(
                    "common-verifier-admitted state violated archive invariants"
                ) from exc
            _write_json_once(
                layout.path(
                    f"step{epoch:02d}/branches/{execution.outcome.branch_id}/"
                    f"archive/{evidence.evidence_id}.decision.json"
                ),
                {
                    "schema_version": 1,
                    **vars(decision),
                },
            )
        if execution.provenance_edges:
            provenance = provenance.with_artifacts(archive.artifacts)
            for edge in execution.provenance_edges:
                try:
                    provenance = provenance.append(edge)
                except Exception as exc:
                    raise EngineError(
                        f"verified provenance edge {edge.edge_id} was rejected"
                    ) from exc

        from evolve.scheduler.arms import ArmIdentity

        identity = ArmIdentity.from_arm(arm)
        maximum_reward = execution.outcome.maximum_reward
        record_improved = maximum_reward is not None and maximum_reward > record_threshold
        gain = _normalized_outcome_gain(
            adapter,
            execution.outcome,
            frozen_record_threshold=record_threshold,
        )
        posterior = posterior.observe(
            identity,
            admitted=execution.outcome.eligible_for_scheduler and execution.outcome.maximum_state_id is not None,
            infrastructure=execution.outcome.infrastructure_aborted,
            record_improved=record_improved,
            gain=gain,
            costs=execution.outcome.costs,
        )
        return archive, provenance, record, posterior, budget_ledger

    def _confirm_epoch_record(
        self,
        *,
        layout: RunLayout,
        epoch: int,
        executions: Sequence[BranchExecution],
        archive: ScientificArchive,
        record: ConfirmedRecordTracker,
        budget_ledger: BudgetLedger,
        frozen_record_threshold: float,
        adapter: ProblemScientificAdapter,
        verification_policy: VerificationPolicy,
    ) -> Tuple[ScientificArchive, ConfirmedRecordTracker, BudgetLedger]:
        """Confirm provisional record candidates in descending reward order.

        A durable result is replayed by ID after a crash. Infrastructure-aborted
        confirmation attempts remain unresolved evidence and are never retried
        under a different implicit sample identity.
        """

        candidates: Dict[str, Tuple[Proposal, EvidencePacket]] = {}
        for execution in executions:
            if execution.outcome.infrastructure_aborted:
                continue
            for observation in execution.observations:
                evidence = observation.verification.evidence
                if (
                    evidence.admitted
                    and evidence.resolved
                    and evidence.internal_reward is not None
                    and float(evidence.internal_reward) > frozen_record_threshold
                ):
                    candidates[evidence.evidence_id] = (
                        observation.proposal,
                        evidence,
                    )
        ordered = sorted(
            candidates.values(),
            key=lambda item: (
                -float(item[1].internal_reward),
                item[1].evidence_id,
            ),
        )
        confirmation_root = layout.path(
            f"step{epoch:02d}/confirmations"
        )
        for proposal, prior_evidence in ordered:
            if budget_ledger.remaining("verifier_calls") < 2.0:
                break
            confirmation_debit = (
                f"epoch-confirm:{epoch}:{prior_evidence.evidence_id}"
            )
            budget_ledger = BudgetService.debit(
                budget_ledger,
                resource="verifier_calls",
                amount=2.0,
                transaction_key=confirmation_debit,
            )
            confirmation_dir = confirmation_root / prior_evidence.evidence_id
            confirmation_dir.mkdir(parents=True, exist_ok=True)
            _write_json_once(
                confirmation_dir / "request.json",
                {
                    "schema_version": 1,
                    "epoch": epoch,
                    "proposal_id": proposal.proposal_id,
                    "prior_evidence_id": prior_evidence.evidence_id,
                    "answer_payload_hash": content_hash(
                        prior_evidence.answer_payload
                    ),
                    "verifier_id": prior_evidence.verifier_id,
                    "verifier_version": prior_evidence.verifier_version,
                },
            )
            result_path = confirmation_dir / "result.json"
            aborted_path = confirmation_dir / "aborted.json"
            evidence_path = confirmation_dir / "evidence.json"
            state_path = confirmation_dir / "state.json"
            descriptor_path = confirmation_dir / "descriptor.json"
            attempts_path = confirmation_dir / "attempts.json"
            if aborted_path.is_file() and evidence_path.is_file():
                raise EngineError(
                    "record confirmation has both aborted and durable evidence artifacts"
                )
            if aborted_path.is_file() and not result_path.is_file():
                continue

            if result_path.is_file():
                result_document = json.loads(
                    result_path.read_text(encoding="utf-8")
                )
                attempts = _confirmation_attempt_count(result_document["attempts"])
                confirmation_evidence = EvidencePacket.from_dict(
                    json.loads(evidence_path.read_text(encoding="utf-8"))
                )
                _validate_confirmation_binding(
                    proposal, prior_evidence, confirmation_evidence
                )
                if result_document.get("evidence_id") != confirmation_evidence.evidence_id:
                    raise EngineError("confirmation result/evidence identity mismatch")
                attempt_flag = confirmation_evidence.flags.get(
                    "verification_attempt_index"
                )
                _validate_attempt_count_covers_evidence(
                    attempt_flag, attempts, context="confirmation result"
                )
                confirmation_state = None
                confirmation_descriptor = None
                if result_document.get("state_id") is not None:
                    confirmation_state = VerifiedScientificState.from_dict(
                        json.loads(state_path.read_text(encoding="utf-8"))
                    )
                    confirmation_descriptor = Descriptor.from_dict(
                        json.loads(descriptor_path.read_text(encoding="utf-8"))
                    )
                    if result_document.get("state_id") != confirmation_state.state_id:
                        raise EngineError("confirmation result/state identity mismatch")
            elif evidence_path.is_file():
                # The common-verifier packet is the durable output. Complete
                # derived files and the marker without invoking the verifier a
                # second time under the same sample identity.
                confirmation_evidence = EvidencePacket.from_dict(
                    json.loads(evidence_path.read_text(encoding="utf-8"))
                )
                _validate_confirmation_binding(
                    proposal, prior_evidence, confirmation_evidence
                )
                attempt_flag = confirmation_evidence.flags.get(
                    "verification_attempt_index"
                )
                attempts = (
                    _verification_attempt_index(attempt_flag, maximum=1) + 1
                    if attempt_flag is not None
                    else 2
                )
                if attempts_path.is_file():
                    durable_attempts = _confirmation_attempt_count(
                        json.loads(attempts_path.read_text(encoding="utf-8"))["attempts"]
                    )
                    _validate_attempt_count_covers_evidence(
                        attempt_flag,
                        durable_attempts,
                        context="confirmation marker",
                    )
                    attempts = durable_attempts
                confirmation_state = None
                confirmation_descriptor = None
                if confirmation_evidence.confirmed:
                    if state_path.is_file() and descriptor_path.is_file():
                        confirmation_state = VerifiedScientificState.from_dict(
                            json.loads(state_path.read_text(encoding="utf-8"))
                        )
                        confirmation_descriptor = Descriptor.from_dict(
                            json.loads(
                                descriptor_path.read_text(encoding="utf-8")
                            )
                        )
                    else:
                        restored_state, restored_descriptor = (
                            _restore_admitted_artifacts(
                                confirmation_evidence, adapter=adapter
                            )
                        )
                        if state_path.is_file():
                            confirmation_state = VerifiedScientificState.from_dict(
                                json.loads(state_path.read_text(encoding="utf-8"))
                            )
                            if confirmation_state != restored_state:
                                raise EngineError(
                                    "partial confirmation state changed on recovery"
                                )
                        else:
                            confirmation_state = restored_state
                            _write_json_once(
                                state_path, confirmation_state.to_dict()
                            )
                        if descriptor_path.is_file():
                            confirmation_descriptor = Descriptor.from_dict(
                                json.loads(
                                    descriptor_path.read_text(encoding="utf-8")
                                )
                            )
                            if confirmation_descriptor != restored_descriptor:
                                raise EngineError(
                                    "partial confirmation descriptor changed on recovery"
                                )
                        else:
                            confirmation_descriptor = restored_descriptor
                            _write_json_once(
                                descriptor_path,
                                confirmation_descriptor.to_dict(),
                            )
                _write_json_once(
                    result_path,
                    {
                        "schema_version": 1,
                        "evidence_id": confirmation_evidence.evidence_id,
                        "state_id": (
                            confirmation_state.state_id
                            if confirmation_state is not None
                            else None
                        ),
                        "confirmed": confirmation_evidence.confirmed,
                        "attempts": attempts,
                    },
                )
            else:
                try:
                    confirmation, attempts = self._confirm(
                        proposal,
                        prior_evidence,
                        adapter=adapter,
                        verification_policy=verification_policy,
                        run_dir=layout.run_dir,
                        attempt_dir=confirmation_dir / "attempts",
                        phase="epoch_record_confirmation",
                    )
                except RecordConfirmationInfrastructureError as exc:
                    _write_json_once(
                        aborted_path,
                        {
                            "schema_version": 1,
                            "failure_kind": FailureKind.INFRASTRUCTURE.value,
                            "exception_type": type(exc).__name__,
                            "message": str(exc)[:2048],
                            "attempts": 2,
                        },
                    )
                    continue
                confirmation_evidence = confirmation.evidence
                confirmation_state = confirmation.state
                confirmation_descriptor = confirmation.descriptor
                attempts = _confirmation_attempt_count(attempts)
                _validate_confirmation_binding(
                    proposal, prior_evidence, confirmation_evidence
                )
                _write_json_once(
                    evidence_path,
                    confirmation_evidence.to_dict(),
                )
                if confirmation_state is not None:
                    _write_json_once(
                        state_path,
                        confirmation_state.to_dict(),
                    )
                    _write_json_once(
                        descriptor_path,
                        confirmation_descriptor.to_dict(),
                    )
                _write_json_once(
                    attempts_path,
                    {"schema_version": 1, "attempts": attempts},
                )
                _write_json_once(
                    result_path,
                    {
                        "schema_version": 1,
                        "evidence_id": confirmation_evidence.evidence_id,
                        "state_id": (
                            confirmation_state.state_id
                            if confirmation_state is not None
                            else None
                        ),
                        "confirmed": confirmation_evidence.confirmed,
                        "attempts": attempts,
                    },
                )

            if attempts < 2:
                budget_ledger = BudgetService.refund(
                    budget_ledger,
                    resource="verifier_calls",
                    amount=float(2 - attempts),
                    transaction_key=f"{confirmation_debit}:refund",
                    debit_transaction_key=confirmation_debit,
                )
            archive = replace(
                archive,
                artifacts=archive.artifacts.add_observation(
                    proposal, confirmation_evidence
                ),
            )
            if confirmation_evidence.confirmed and confirmation_state is not None:
                try:
                    archive, confirmation_decision = archive.offer(
                        confirmation_descriptor,
                        proposal,
                        confirmation_state,
                        confirmation_evidence,
                    )
                except ArchiveAdmissionError as exc:
                    _write_json_once(
                        confirmation_dir / "archive.rejected.json",
                        {
                            "schema_version": 1,
                            "exception_type": type(exc).__name__,
                            "message": str(exc),
                        },
                    )
                    raise EngineError(
                        "confirmed record state violated archive invariants"
                    ) from exc
                _write_json_once(
                    confirmation_dir / "archive.decision.json",
                    {"schema_version": 1, **vars(confirmation_decision)},
                )
                prior_record_evidence_id = record.evidence_id
                record = record.consider(
                    confirmation_state,
                    confirmation_evidence,
                    archive=archive,
                )
                # A noisy confirmation can move below the frozen record even
                # when its provisional observation ranked first. Only stop once
                # confirmation actually advances the record; otherwise another
                # candidate may be tried if refunded/unused budget permits.
                if record.evidence_id != prior_record_evidence_id:
                    break
        return archive, record, budget_ledger

    # -- barrier commit + reporting --------------------------------------

    def _commit_barrier(
        self,
        layout: RunLayout,
        state: EpochState,
        report: EpochReport,
        *,
        adapter: ProblemScientificAdapter,
        workers: EngineWorkers,
    ) -> None:
        self._publish_snapshots(layout, state)
        role_artifacts = (
            workers.persist_roles(state)
            if workers.persist_roles is not None
            else state.role_artifacts
        ) or {}
        checkpoint_dir = layout.path("checkpoints")
        checkpoint_dir.mkdir(parents=True, exist_ok=True)
        checkpoint = _checkpoint_payload(state, role_artifacts=role_artifacts)
        checkpoint_path = checkpoint_dir / f"checkpoint_epoch{state.epoch:03d}.json"
        _write_json_once(checkpoint_path, checkpoint)
        self._persist_training_state(
            layout,
            state=state,
            checkpoint=checkpoint,
            workers=workers,
            epoch=state.epoch,
        )
        checkpoint_hash = content_hash(checkpoint)
        training_state_path = checkpoint_dir / f"checkpoint_epoch{state.epoch:03d}.pt"
        training_state_hash = _file_sha256(training_state_path)
        costs: Dict[str, float] = {}
        for execution in report.branch_executions:
            for resource, amount in execution.outcome.costs.items():
                costs[resource] = costs.get(resource, 0.0) + float(amount)
        summary = {
            "epoch": report.epoch,
            "planned_arms": len(report.plan.planned_arms),
            "audit_pairs": len(report.audit_pairs),
            "record_improved": report.record_improved,
            "confirmed_reward": state.record.internal_reward,
            "confirmed_raw_score": state.record.raw_score,
            "archive_coverage": state.archive.coverage,
            "costs": costs,
            "budget_consumed": {
                resource: state.budget_ledger.consumed(resource)
                for resource in state.budget_ledger.limits
            },
            "admitted_branches": sum(
                1
                for item in report.branch_executions
                if item.outcome.maximum_state_id is not None
            ),
            "infrastructure_aborted": sum(
                1
                for item in report.branch_executions
                if item.outcome.infrastructure_aborted
            ),
            "closed_audit_pairs": sum(
                1 for pair in report.audit_pairs if pair.status == AuditStatus.CLOSED
            ),
            "aborted_audit_pairs": sum(
                1 for pair in report.audit_pairs if pair.status == AuditStatus.ABORTED
            ),
            "reservation_slots": {
                "audit_branch_slots": report.plan.reservation_slots.audit_branch_slots,
                "no_memory_audit_slots": report.plan.reservation_slots.no_memory_audit_slots,
                "refinement_slots": report.plan.reservation_slots.refinement_slots,
                "harness_trial_slots": report.plan.reservation_slots.harness_trial_slots,
                "empty_cell_slots": report.plan.reservation_slots.empty_cell_slots,
                "global_exploration_slots": report.plan.reservation_slots.global_exploration_slots,
                "role_guaranteed_slots": report.plan.reservation_slots.role_guaranteed_slots,
                "remaining_production_slots": report.plan.reservation_slots.remaining_production_slots,
            },
            "arms_by_role": {
                role: sum(
                    planned.replicas
                    for planned in report.plan.planned_arms
                    if planned.arm.role.value == role
                )
                for role in sorted({planned.arm.role.value for planned in report.plan.planned_arms})
            },
        }
        summary.update(
            {
                "schema_version": 1,
                "committed_epoch": state.epoch,
                "checkpoint": checkpoint_path.name,
                "checkpoint_hash": checkpoint_hash,
                "training_state": training_state_path.name,
                "training_state_hash": training_state_hash,
            }
        )
        atomic_write_json(
            checkpoint_dir / "latest.json",
            {
                "schema_version": 1,
                "epoch": state.epoch,
                "checkpoint": checkpoint_path.name,
                "checkpoint_hash": checkpoint_hash,
            },
        )
        step_dir = layout.path(f"step{report.epoch:02d}")
        _write_json_once(
            step_dir / f"step{report.epoch:02d}.summary.json", summary
        )
        self._publish_role_pointers(
            layout, role_artifacts=role_artifacts, epoch=state.epoch
        )
        with ControllerEventWriter(layout.path("events.jsonl")) as event_writer:
            event_writer.append(
                "barrier_committed",
                {
                    "kind": "epoch",
                    "epoch": report.epoch,
                    "next_epoch": state.epoch,
                    "checkpoint": checkpoint_path.name,
                    "checkpoint_hash": checkpoint_hash,
                },
                idempotency_key=f"barrier-committed:epoch:{report.epoch}",
            )
        if report.record_improved:
            self._publish_best(layout, state, adapter=adapter)
        self._write_status(
            layout, state, note=f"epoch {report.epoch} committed", report=report
        )
        if (
            not report.record_improved
            and state.record.evidence_id is not None
            and state.epoch % self.config.evolve.reporting.plots_every_epochs == 0
        ):
            from evolve.reporting import print_best_answer

            print_best_answer(layout.run_dir)
        if state.epoch % self.config.evolve.reporting.plots_every_epochs == 0:
            try:
                from evolve.viz.run import generate_plots

                generate_plots(
                    layout.run_dir,
                    names=(
                        "record",
                        "archive",
                        "provenance",
                        "allocation",
                        "audits",
                        "roles",
                        "posterior",
                        "failures",
                        "resources",
                    ),
                    out_dir=layout.path("plots"),
                )
            except Exception as exc:
                # Barrier durability and discovery never depend on plotting.
                try:
                    _write_json_once(
                        layout.path(
                            f"logs/plot_epoch{report.epoch:03d}.error.json"
                        ),
                        {
                            "schema_version": 1,
                            "epoch": report.epoch,
                            "exception_type": type(exc).__name__,
                            "message": str(exc)[:2048],
                        },
                    )
                except Exception:
                    # Logging the non-critical plotting error is itself
                    # best-effort; the completed barrier remains authoritative.
                    pass
        _apply_role_artifact_retention(
            layout,
            keep_epoch=state.epoch,
            mode=self._artifact_retention,
        )

    def _publish_role_pointers(
        self,
        layout: RunLayout,
        *,
        role_artifacts: Mapping[str, Mapping[str, Any]],
        epoch: int,
    ) -> None:
        """Publish compatibility pointers only after their checkpoint is durable."""

        for role_name, artifact in role_artifacts.items():
            atomic_write_json(
                layout.path(f"roles/{role_name}/latest.json"),
                {"epoch": epoch, **dict(artifact)},
            )

    def _persist_training_state(
        self,
        layout: RunLayout,
        *,
        state: EpochState,
        checkpoint: Mapping[str, Any],
        workers: Optional[EngineWorkers],
        epoch: int,
    ) -> None:
        targets = (
            layout.path(f"checkpoints/checkpoint_epoch{epoch:03d}.pt"),
            layout.path("training_state.pt"),
        )
        callback = workers.persist_training_state if workers is not None else None
        if callback is not None:
            callback(state, checkpoint, targets)
            return
        # Fake-worker fixtures still receive complete compatibility artifacts;
        # the JSON envelope is intentionally marked so it cannot masquerade as
        # a production optimizer/RNG checkpoint.
        fallback = {
            "format": "evolve_json_fixture_training_state_v1",
            "checkpoint": dict(checkpoint),
        }
        for target in targets:
            if target.parent.name == "checkpoints":
                _write_json_once(target, fallback)
            else:
                atomic_write_json(target, fallback)

    def _publish_snapshots(self, layout: RunLayout, state: EpochState) -> None:
        """Publish complete committed subsystem views before the checkpoint."""

        archive_document = {
            "schema_version": 1,
            "epoch": state.epoch,
            "cell_map_version": state.archive.cell_map_version,
            "descriptors": [
                item.to_dict() for item in state.archive.descriptors
            ],
            "cells": [item.to_dict() for item in state.archive.cells],
            "proposals": [
                item.to_dict() for item in state.archive.artifacts.proposals
            ],
            "evidence": [
                item.to_dict() for item in state.archive.artifacts.evidence
            ],
            "states": [
                item.to_dict() for item in state.archive.artifacts.states
            ],
            "provenance": [item.to_dict() for item in state.provenance.edges],
        }
        atomic_write_json(
            layout.path("archive/cells.json"),
            {
                "schema_version": 1,
                "epoch": state.epoch,
                "cells": archive_document["cells"],
            },
        )
        append_jsonl_records(
            layout.path("archive/candidates.jsonl"),
            (item.to_dict() for item in state.archive.artifacts.proposals),
            id_field="proposal_id",
        )
        append_jsonl_records(
            layout.path("archive/evidence.jsonl"),
            (item.to_dict() for item in state.archive.artifacts.evidence),
            id_field="evidence_id",
        )
        append_jsonl_records(
            layout.path("archive/provenance.jsonl"),
            (item.to_dict() for item in state.provenance.edges),
            id_field="edge_id",
        )
        append_jsonl_records(
            layout.path("causal_memory/records.jsonl"),
            (
                {
                    "memory_version_id": content_id(
                        "causal_memory_version", item.to_dict()
                    ),
                    "memory_id": item.memory_id,
                    "record": item.to_dict(),
                }
                for item in sorted(
                    state.memory.records.values(), key=lambda record: record.memory_id
                )
            ),
            id_field="memory_version_id",
        )
        _write_json_once(
            layout.path(f"archive/snapshots/epoch{state.epoch:03d}.json"),
            archive_document,
        )
        _write_json_once(
            layout.path(
                f"causal_memory/snapshots/epoch{state.epoch:03d}.json"
            ),
            {
                "schema_version": 1,
                "epoch": state.epoch,
                "records": [
                    record.to_dict()
                    for record in sorted(
                        state.memory.records.values(),
                        key=lambda item: item.memory_id,
                    )
                ],
            },
        )

    def _publish_best(
        self, layout: RunLayout, state: EpochState, *, adapter: ProblemScientificAdapter
    ) -> None:
        """Atomically republish the confirmed record's answer and renderer output.

        Only ever called for a *confirmed* record at a committed barrier, so
        an in-epoch provisional observation can never reach ``best/``.
        """

        record = state.record
        if record.state_id is None or record.evidence_id is None:
            raise EngineError(
                "best publication was requested without a confirmed record"
            )
        try:
            evidence = state.archive.artifacts.evidence_packet(record.evidence_id)
            verified_state = state.archive.artifacts.state_binding(
                record.state_id, evidence.proposal_id, record.evidence_id
            )
        except Exception as exc:
            raise EngineError(
                "confirmed record cannot be resolved from the committed archive"
            ) from exc
        state_document = verified_state.to_dict()
        evidence_document = evidence.to_dict()
        answer_payload = _json_native_answer_payload(verified_state)
        best_dir = layout.path("best")
        best_dir.mkdir(parents=True, exist_ok=True)
        snapshots_dir = best_dir / "snapshots"
        snapshots_dir.mkdir(parents=True, exist_ok=True)
        snapshot_dir = snapshots_dir / evidence.evidence_id
        if not snapshot_dir.is_dir():
            staging = layout.path(f".best-{evidence.evidence_id}")
            if staging.exists():
                shutil.rmtree(staging)
            staging.mkdir(parents=True, exist_ok=False)
            try:
                atomic_write_json(staging / "state.json", state_document)
                atomic_write_json(staging / "evidence.json", evidence_document)
                atomic_write_json(
                    staging / "candidate.json",
                    {
                        "state_id": verified_state.state_id,
                        "proposal_id": verified_state.proposal_id,
                        "answer_payload": answer_payload,
                    },
                )
                try:
                    adapter.problem.render_best(
                        answer_payload, evidence_document, staging
                    )
                except Exception as exc:
                    atomic_write_json(
                        staging / "answer.json", answer_payload
                    )
                    atomic_write_text(
                        staging / "renderer.error.txt",
                        f"{type(exc).__name__}: {exc}\n",
                    )
                if not (staging / "answer.txt").is_file() and not (
                    staging / "answer.py"
                ).is_file():
                    atomic_write_text(
                        staging / "answer.txt",
                        json.dumps(
                            answer_payload,
                            ensure_ascii=False,
                            sort_keys=True,
                            indent=2,
                        )
                        + "\n",
                    )
                compatibility_files = sorted(
                    path.name for path in staging.iterdir() if path.is_file()
                )
                atomic_write_json(
                    staging / "snapshot.manifest.json",
                    {
                        "schema_version": 1,
                        "state_id": verified_state.state_id,
                        "evidence_id": evidence.evidence_id,
                        "compatibility_files": compatibility_files,
                    },
                )
                os.replace(staging, snapshot_dir)
                fsync_directory(snapshots_dir)
            finally:
                if staging.exists():
                    shutil.rmtree(staging)
        snapshot_manifest = json.loads(
            (snapshot_dir / "snapshot.manifest.json").read_text(encoding="utf-8")
        )
        if (
            snapshot_manifest.get("state_id") != verified_state.state_id
            or snapshot_manifest.get("evidence_id") != evidence.evidence_id
        ):
            raise EngineError("best snapshot identity conflicts with the record")

        old_pointer = {}
        best_pointer = best_dir / "latest.json"
        if best_pointer.is_file():
            try:
                old_pointer = json.loads(best_pointer.read_text(encoding="utf-8"))
            except (OSError, json.JSONDecodeError):
                old_pointer = {}
        compatibility_files = tuple(snapshot_manifest["compatibility_files"])
        for name in compatibility_files:
            source = snapshot_dir / name
            if not source.is_file():
                raise EngineError(f"best snapshot is missing {name}")
            atomic_write_bytes(best_dir / name, source.read_bytes())
        pointer_document = {
            "schema_version": 1,
            "state_id": verified_state.state_id,
            "evidence_id": evidence.evidence_id,
            "snapshot": f"snapshots/{evidence.evidence_id}",
            "compatibility_files": list(compatibility_files),
        }
        atomic_write_json(best_pointer, pointer_document)
        for stale_name in old_pointer.get("compatibility_files", ()):
            if stale_name in compatibility_files:
                continue
            stale_path = best_dir / str(stale_name)
            if stale_path.parent == best_dir and stale_path.is_file():
                stale_path.unlink()
                fsync_directory(best_dir)
        proposal = state.archive.artifacts.proposal(verified_state.proposal_id)
        atomic_write_text(layout.run_dir / "best_code.py", proposal.source_text)
        atomic_write_json(
            layout.run_dir / "best_construction.json",
            {
                "answer_payload": answer_payload,
                "internal_reward": evidence.internal_reward,
            },
        )
        origin = (
            "deterministic problem bootstrap seed"
            if proposal.parent_state_id is None
            else "generated branch proposal"
        )
        print(
            "\nEVOLVE · confirmed record"
            f" · origin={origin}"
            f" · raw_score={verified_state.raw_score}"
            f" · internal_reward={verified_state.internal_reward}\n"
            f"EVOLVE · confirmed record artifacts · {best_dir}\n",
            flush=True,
        )

    def _write_live_progress_status(
        self,
        layout: RunLayout,
        state: EpochState,
        *,
        epoch: int,
        stage: str,
        completed_branches: int,
        total_branches: int,
        completed_verifications: int,
        executions: Sequence[BranchExecution],
    ) -> None:
        """Refresh live progress without publishing an in-epoch record."""

        provisional_best = None
        for execution in executions:
            evidence_id = execution.outcome.maximum_evidence_id
            if evidence_id is None or execution.outcome.maximum_reward is None:
                continue
            matching = next(
                (
                    observation.verification.evidence
                    for observation in execution.observations
                    if observation.verification.evidence.evidence_id == evidence_id
                ),
                None,
            )
            candidate = {
                "state_id": execution.outcome.maximum_state_id,
                "evidence_id": evidence_id,
                "branch_id": execution.outcome.branch_id,
                "internal_reward": execution.outcome.maximum_reward,
                "raw_score": matching.raw_score if matching is not None else None,
                "committed": False,
            }
            if (
                provisional_best is None
                or float(candidate["internal_reward"])
                > float(provisional_best["internal_reward"])
            ):
                provisional_best = candidate
        document = {}
        status_path = layout.path("status.json")
        if status_path.is_file():
            try:
                document = json.loads(status_path.read_text(encoding="utf-8"))
            except (OSError, json.JSONDecodeError):
                document = {}
        previous_live = document.get("live_epoch", {})
        previous_best = (
            previous_live.get("provisional_best")
            if isinstance(previous_live, Mapping)
            and previous_live.get("epoch") == epoch
            else None
        )
        if (
            isinstance(previous_best, Mapping)
            and isinstance(previous_best.get("internal_reward"), (int, float))
            and (
                provisional_best is None
                or float(previous_best["internal_reward"])
                > float(provisional_best["internal_reward"])
            )
        ):
            provisional_best = dict(previous_best)
        document.update(
            {
                "schema_version": 1,
                "run_id": state.run_id,
                "run_directory": str(layout.run_dir),
                "epoch": state.epoch,
                "confirmed_record": {
                    "state_id": state.record.state_id,
                    "cell_id": state.record.cell_id,
                    "internal_reward": state.record.internal_reward,
                    "raw_score": state.record.raw_score,
                },
                "live_epoch": {
                    "epoch": epoch,
                    "stage": stage,
                    "total_epochs": self.config.evolve.budget.epochs,
                    "completed_branches": completed_branches,
                    "total_branches": total_branches,
                    "remaining_branches": max(
                        0, total_branches - completed_branches
                    ),
                    "branch_progress": (
                        completed_branches / total_branches
                        if total_branches
                        else 1.0
                    ),
                    "completed_verifications": completed_verifications,
                    "infrastructure_aborted": sum(
                        1 for item in executions if item.outcome.infrastructure_aborted
                    ),
                    "provisional_observation": (
                        provisional_best["internal_reward"]
                        if provisional_best is not None
                        else None
                    ),
                    "provisional_best": provisional_best,
                    "provisional_is_committed": False,
                },
                "run_guard": dict(_run_guard_document()),
                "note": f"epoch {epoch} active",
            }
        )
        atomic_write_json(status_path, document)

    def _write_interrupted_status(
        self,
        layout: RunLayout,
        state: EpochState,
        *,
        committed_epoch: Optional[int],
    ) -> None:
        """Preserve live evidence while recording a completed graceful drain."""

        status_path = layout.path("status.json")
        try:
            status = json.loads(status_path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            status = {
                "schema_version": 1,
                "run_id": state.run_id,
                "run_directory": str(layout.run_dir),
                "epoch": committed_epoch,
            }
        guard = dict(_run_guard_document())
        interruption = {
            "graceful_worker_shutdown_completed": True,
            "last_committed_epoch": committed_epoch,
            "latest_completed_barrier_found": committed_epoch is not None,
            "partial_epoch_is_committed": False,
            "durable_samples_are_resume_reusable": True,
            "public_best_remains_barrier_confirmed": True,
            "run_guard": guard,
        }
        status.update(
            {
                "note": (
                    "interrupted after graceful worker shutdown; last completed "
                    "barrier preserved"
                ),
                "run_guard": guard,
                "interruption": interruption,
            }
        )
        atomic_write_json(status_path, status)
        with ControllerEventWriter(layout.path("events.jsonl")) as event_writer:
            sequence = event_writer.next_sequence
            event_writer.append(
                "run_interrupted",
                {
                    **interruption,
                    "provisional_best": status.get("live_epoch", {}).get(
                        "provisional_best"
                    ),
                },
                idempotency_key=f"run-interrupted:{sequence}",
            )

    def _write_status(
        self,
        layout: RunLayout,
        state: EpochState,
        *,
        note: str,
        report: Optional[EpochReport] = None,
    ) -> None:
        holder = None
        if state.record.evidence_id is not None:
            try:
                evidence = state.archive.artifacts.evidence_packet(
                    state.record.evidence_id
                )
                holder = {
                    "proposal_id": evidence.proposal_id,
                    "evidence_id": evidence.evidence_id,
                    "branch_id": evidence.branch_id,
                    "harness_id": evidence.harness_id,
                    "policy_snapshot_id": evidence.policy_snapshot_id,
                    "confirmed_at": evidence.completed_at,
                }
                branch_specs = list(
                    layout.run_dir.glob(
                        f"step*/branches/{evidence.branch_id}/branch.spec.json"
                    )
                )
                if branch_specs:
                    branch_document = json.loads(
                        branch_specs[-1].read_text(encoding="utf-8")
                    )
                    holder_role = branch_document.get(
                        "generation_settings", {}
                    ).get("role")
                    if holder_role is None:
                        assignment_path = branch_specs[-1].with_name(
                            "assignment.json"
                        )
                        if assignment_path.is_file():
                            assignment = json.loads(
                                assignment_path.read_text(encoding="utf-8")
                            )
                            holder_role = assignment.get("arm", {}).get("role")
                    holder.update(
                        {
                            "epoch": branch_document.get("epoch"),
                            "role": holder_role,
                            "option_id": branch_document.get("option_id"),
                        }
                    )
            except Exception:
                holder = None
        outcomes = tuple(report.branch_executions) if report is not None else ()
        branch_documents = []
        for execution in outcomes:
            paths = list(
                layout.run_dir.glob(
                    f"step*/branches/{execution.outcome.branch_id}/branch.spec.json"
                )
            )
            if paths:
                try:
                    branch_documents.append(
                        json.loads(paths[-1].read_text(encoding="utf-8"))
                    )
                except (OSError, json.JSONDecodeError):
                    pass
        channels = {}
        roles = {}
        options = {}
        harnesses = {}
        for branch in branch_documents:
            channel = str(branch.get("channel", "unknown"))
            role = str(branch.get("generation_settings", {}).get("role", "unknown"))
            option = str(branch.get("option_id", "unknown"))
            harness = str(branch.get("harness_id", "unknown"))
            channels[channel] = channels.get(channel, 0) + 1
            roles[role] = roles.get(role, 0) + 1
            options[option] = options.get(option, 0) + 1
            harnesses[harness] = harnesses.get(harness, 0) + 1
        closed_audits = (
            sum(1 for pair in report.audit_pairs if pair.status == AuditStatus.CLOSED)
            if report is not None else 0
        )
        aborted_audits = (
            sum(1 for pair in report.audit_pairs if pair.status == AuditStatus.ABORTED)
            if report is not None else 0
        )
        status = {
            "schema_version": 1,
            "run_id": state.run_id,
            "run_directory": str(layout.run_dir),
            "epoch": state.epoch,
            "confirmed_record": {
                "state_id": state.record.state_id,
                "cell_id": state.record.cell_id,
                "internal_reward": state.record.internal_reward,
                "raw_score": state.record.raw_score,
                "holder": holder,
                "age_epochs": (
                    max(0, state.epoch - int(holder["epoch"]))
                    if holder is not None and holder.get("epoch") is not None
                    else None
                ),
            },
            "archive_coverage": state.archive.coverage,
            "archive_cells": len(state.archive.cells),
            "budget": {
                resource: {
                    "limit": float(limit),
                    "consumed": state.budget_ledger.consumed(resource),
                    "remaining": state.budget_ledger.remaining(resource),
                }
                for resource, limit in state.budget_ledger.limits.items()
            },
            "roles": {
                role.value: {
                    "adapter_version": state.role_registry.state(role).adapter.version,
                    "adapter_hash": state.role_registry.state(role).adapter.adapter_hash,
                    "learning_groups": len(
                        state.role_registry.state(role).learning.group_ids
                    ),
                }
                for role in state.role_registry.roles
            },
            "latest_learning": {
                role.value: {
                    "group_id": (
                        state.role_registry.state(role).learning.group_ids[-1]
                        if state.role_registry.state(role).learning.group_ids
                        else None
                    ),
                    "optimizer_step": state.role_registry.state(role).optimizer.step,
                    "adapter_version": state.role_registry.state(role).adapter.version,
                    "adapter_hash": state.role_registry.state(role).adapter.adapter_hash,
                }
                for role in state.role_registry.roles
            },
            "causal_memory": {
                "records": len(state.memory.records),
                "promoted": sum(
                    1
                    for record in state.memory.records.values()
                    if record.status.value == "promoted"
                ),
                "quarantined": sum(
                    1
                    for record in state.memory.records.values()
                    if record.status.value == "quarantined"
                ),
                "rejected": sum(
                    1
                    for record in state.memory.records.values()
                    if record.status.value == "rejected"
                ),
            },
            "allocations": {
                "by_channel": channels,
                "by_role": roles,
                "by_option": options,
                "by_harness": harnesses,
            },
            "latest_epoch": {
                "branches": len(outcomes),
                "admitted": sum(
                    1 for item in outcomes if item.outcome.maximum_state_id is not None
                ),
                "infrastructure_aborted": sum(
                    1 for item in outcomes if item.outcome.infrastructure_aborted
                ),
                "audit_pairs": len(report.audit_pairs) if report is not None else 0,
                "closed_audit_pairs": closed_audits,
                "aborted_audit_pairs": aborted_audits,
                "admission_rate": (
                    sum(1 for item in outcomes if item.outcome.maximum_state_id is not None)
                    / len(outcomes)
                    if outcomes else 0.0
                ),
                "infrastructure_rate": (
                    sum(1 for item in outcomes if item.outcome.infrastructure_aborted)
                    / len(outcomes)
                    if outcomes else 0.0
                ),
            },
            "run_guard": dict(_run_guard_document()),
            "note": note,
        }
        atomic_write_json(layout.path("status.json"), status)


def _objective_enum(value: str):
    from evolve.types import LearningObjective

    return LearningObjective(value)


def _checkpoint_payload(
    state: EpochState,
    *,
    role_artifacts: Optional[Mapping[str, Mapping[str, Any]]] = None,
) -> Mapping[str, Any]:
    return {
        "record_type": "evolve_epoch_checkpoint",
        "schema_version": 1,
        "run_id": state.run_id,
        "epoch": state.epoch,
        "archive": {
            "cell_map_version": state.archive.cell_map_version,
            "max_promising_slots": state.archive.max_promising_slots,
            "max_stepping_stone_slots": state.archive.max_stepping_stone_slots,
            "under_tested_threshold": state.archive.under_tested_threshold,
            "descriptors": [d.to_dict() for d in state.archive.descriptors],
            "cells": [c.to_dict() for c in state.archive.cells],
            "proposals": [p.to_dict() for p in state.archive.artifacts.proposals],
            "evidence": [e.to_dict() for e in state.archive.artifacts.evidence],
            "states": [s.to_dict() for s in state.archive.artifacts.states],
        },
        "provenance": [e.to_dict() for e in state.provenance.edges],
        "posterior": state.posterior.to_dict(),
        "budget_ledger": state.budget_ledger.to_dict(),
        "causal_memory": [record.to_dict() for record in state.memory.records.values()],
        "harness_registry": state.harness_registry.to_dict(),
        "option_ids": list(state.option_registry.option_ids()),
        "nursery": [vars(entry) for entry in state.nursery.values()],
        "record": {
            "state_id": state.record.state_id,
            "evidence_id": state.record.evidence_id,
            "cell_id": state.record.cell_id,
            "internal_reward": state.record.internal_reward,
            "raw_score": state.record.raw_score,
        },
        "roles": state.role_registry.checkpoint_payload(),
        "role_artifacts": {
            role: dict(payload)
            for role, payload in (role_artifacts or state.role_artifacts).items()
        },
        "component_schema_versions": dict(COMPONENT_SCHEMA_VERSIONS),
    }


def _state_from_checkpoint(checkpoint: Mapping[str, Any], *, config: EvolveConfig) -> EpochState:
    from evolve.archive import ScientificArtifactStore
    from evolve.types import ArchiveCell, Descriptor, EvidencePacket, Proposal, ProvenanceEdge, VerifiedScientificState, CausalMemoryRecord

    if checkpoint.get("record_type") != "evolve_epoch_checkpoint":
        raise EngineError("resume checkpoint is not an EVOLVE epoch checkpoint")
    schema_version = checkpoint.get("schema_version")
    if schema_version != 1:
        direction = "future" if isinstance(schema_version, int) and schema_version > 1 else "unsupported"
        raise EngineError(f"{direction} EVOLVE checkpoint schema: {schema_version!r}")
    component_versions = checkpoint.get("component_schema_versions", {})
    for component, saved_version in component_versions.items():
        supported = COMPONENT_SCHEMA_VERSIONS.get(component)
        if supported is None or not isinstance(saved_version, int) or saved_version > supported:
            raise EngineError(
                f"unsupported future checkpoint component {component!r} "
                f"version {saved_version!r}"
            )

    archive_doc = checkpoint["archive"]
    artifacts = ScientificArtifactStore()
    for payload in archive_doc["proposals"]:
        artifacts = artifacts.add_proposal(Proposal.from_dict(payload))
    for payload in archive_doc["evidence"]:
        artifacts = artifacts.add_evidence(EvidencePacket.from_dict(payload))
    for payload in archive_doc["states"]:
        state_record = VerifiedScientificState.from_dict(payload)
        proposal = artifacts.proposal(state_record.proposal_id)
        evidence = artifacts.evidence_packet(state_record.evidence_id)
        artifacts = artifacts.add_verified(proposal, state_record, evidence)
    archive = ScientificArchive(
        cell_map_version=archive_doc["cell_map_version"],
        max_promising_slots=archive_doc["max_promising_slots"],
        max_stepping_stone_slots=archive_doc["max_stepping_stone_slots"],
        under_tested_threshold=archive_doc["under_tested_threshold"],
        descriptors=tuple(Descriptor.from_dict(payload) for payload in archive_doc["descriptors"]),
        cells=tuple(ArchiveCell.from_dict(payload) for payload in archive_doc["cells"]),
        artifacts=artifacts,
    )
    provenance = ProvenanceStore(artifacts=artifacts)
    for branch_id in {payload["branch_id"] for payload in checkpoint["provenance"]}:
        provenance = provenance.with_branch(branch_id)
    for payload in checkpoint["provenance"]:
        provenance = provenance.append(ProvenanceEdge.from_dict(payload))

    memory = MemoryStore()
    for payload in checkpoint["causal_memory"]:
        memory = memory.upsert(CausalMemoryRecord.from_dict(payload))

    budget_ledger = BudgetLedger.from_dict(checkpoint["budget_ledger"])
    role_registry = RoleRegistry.from_checkpoint_payload(checkpoint["roles"])
    record_doc = checkpoint["record"]
    record = ConfirmedRecordTracker(
        state_id=record_doc["state_id"], evidence_id=record_doc["evidence_id"],
        cell_id=record_doc["cell_id"], internal_reward=record_doc["internal_reward"],
        raw_score=record_doc["raw_score"],
    )
    harness_registry = HarnessRegistry.from_dict(checkpoint["harness_registry"])
    option_registry = production_option_registry(
        harness_eligibility=tuple(sorted(harness_registry.specs)),
        max_horizon=config.evolve.options.max_horizon,
    )
    saved_option_ids = tuple(checkpoint.get("option_ids", ()))
    if saved_option_ids and saved_option_ids != option_registry.option_ids():
        raise EngineError(
            "resume option registry differs from the frozen checkpoint; "
            "an explicit supported migration is required"
        )
    if record.state_id is not None:
        record_state = artifacts.representative_state(record.state_id)
        record_evidence = artifacts.evidence_packet(record.evidence_id)
        if not record_evidence.confirmed or record_state.evidence_id != record.evidence_id:
            raise EngineError("checkpoint record is not backed by its confirmed evidence")
    return EpochState(
        run_id=checkpoint["run_id"], epoch=checkpoint["epoch"], archive=archive,
        provenance=provenance,
        posterior=PosteriorStore.from_dict(checkpoint["posterior"]), memory=memory,
        role_registry=role_registry, option_registry=option_registry,
        harness_registry=harness_registry, budget_ledger=budget_ledger, record=record,
        nursery={
            payload["entry_id"]: NurseryEntry(**payload)
            for payload in checkpoint.get("nursery", ())
        },
        role_artifacts={
            role: dict(payload)
            for role, payload in checkpoint.get("role_artifacts", {}).items()
        },
    )


def _latest_completed_checkpoint(layout: RunLayout) -> Path:
    """Resolve the newest checkpoint advertised by a durable completion marker."""

    summaries = []
    bootstrap = layout.path("bootstrap.summary.json")
    if bootstrap.is_file():
        summaries.append(bootstrap)
    summaries.extend(layout.run_dir.glob("step*/step*.summary.json"))
    if not summaries:
        raise EngineError(
            f"no completed-barrier checkpoint found in {layout.run_dir}"
        )
    candidates = []
    for summary_path in summaries:
        try:
            summary = json.loads(summary_path.read_text(encoding="utf-8"))
            if summary.get("schema_version") != 1:
                raise ValueError("unsupported completion-marker schema")
            epoch = summary.get("committed_epoch", summary.get("epoch"))
            if isinstance(epoch, bool) or not isinstance(epoch, int) or epoch < 0:
                raise ValueError("invalid committed epoch")
            filename = summary["checkpoint"]
            expected_hash = summary["checkpoint_hash"]
            if (
                not isinstance(filename, str)
                or Path(filename).name != filename
                or not filename.endswith(".json")
            ):
                raise ValueError("unsafe checkpoint filename")
            if (
                not isinstance(expected_hash, str)
                or len(expected_hash) != 64
                or any(character not in "0123456789abcdef" for character in expected_hash)
            ):
                raise ValueError("invalid checkpoint hash")
            path = layout.path(f"checkpoints/{filename}")
            document = json.loads(path.read_text(encoding="utf-8"))
            if content_hash(document) != expected_hash:
                raise ValueError("checkpoint content hash mismatch")
            training_filename = summary.get("training_state")
            training_hash = summary.get("training_state_hash")
            if (training_filename is None) != (training_hash is None):
                raise ValueError("incomplete training-state companion binding")
            if training_filename is not None:
                if (
                    not isinstance(training_filename, str)
                    or Path(training_filename).name != training_filename
                    or not training_filename.endswith(".pt")
                    or not isinstance(training_hash, str)
                    or len(training_hash) != 64
                    or any(
                        character not in "0123456789abcdef"
                        for character in training_hash
                    )
                ):
                    raise ValueError("invalid training-state companion binding")
                training_path = layout.path(f"checkpoints/{training_filename}")
                if _file_sha256(training_path) != training_hash:
                    raise ValueError("training-state companion hash mismatch")
        except (OSError, ValueError, KeyError, TypeError, json.JSONDecodeError) as exc:
            raise EngineError(
                f"invalid completed-barrier marker {summary_path}: {exc}"
            ) from exc
        candidates.append((epoch, path))
    epochs = [epoch for epoch, _path in candidates]
    if len(set(epochs)) != len(epochs):
        raise EngineError("multiple completed-barrier markers claim the same epoch")
    return max(candidates, key=lambda item: item[0])[1]


# --------------------------------------------------------------------------
# Environment probing for the initial run manifest (no CUDA/model touched)
# --------------------------------------------------------------------------


def _git_state() -> Mapping[str, Any]:
    def _run(*args: str) -> str:
        try:
            return subprocess.check_output(
                ["git", *args], cwd=Path(__file__).resolve().parents[1],
                stderr=subprocess.DEVNULL, text=True, timeout=5,
            ).strip()
        except Exception:
            return ""

    return {
        "commit": _run("rev-parse", "HEAD"),
        "branch": _run("rev-parse", "--abbrev-ref", "HEAD"),
        "dirty": bool(_run("status", "--porcelain")),
    }


def _package_versions() -> Mapping[str, Any]:
    versions: Dict[str, str] = {"python": sys.version.split()[0]}
    try:
        from importlib import metadata as importlib_metadata
    except ImportError:  # pragma: no cover - Python < 3.8
        return versions
    for package in ("torch", "transformers", "peft", "numpy", "pyyaml"):
        try:
            versions[package] = importlib_metadata.version(package)
        except Exception:
            continue
    return versions


def _host_document() -> Mapping[str, Any]:
    return {
        "hostname": socket.gethostname(),
        "platform": platform.platform(),
        "python_implementation": platform.python_implementation(),
    }


def _gpu_manifest(config: EvolveConfig) -> List[Mapping[str, Any]]:
    physical_ids = list(config.runtime_gpu_ids)
    if config.kernel_gpu_id is not None and config.kernel_gpu_id not in physical_ids:
        physical_ids.append(config.kernel_gpu_id)
    entries: List[Mapping[str, Any]] = []
    evaluation_shares_model_gpu = (
        config.kernel_gpu_id is not None
        and (
            config.kernel_gpu_id == config.training_gpu_id
            or config.kernel_gpu_id in config.gpu_ids
        )
    )
    for gpu_id in physical_ids:
        purposes = []
        if config.training_gpu_id == gpu_id or (
            config.training_gpu_id is None and gpu_id in config.gpu_ids
        ):
            purposes.append("barrier_learning")
        if gpu_id in config.gpu_ids:
            purposes.append("tensor_parallel_generation")
        if config.kernel_gpu_id == gpu_id:
            purposes.append(
                "serialized_evaluation"
                if evaluation_shares_model_gpu
                else "exclusive_evaluation"
            )
        entries.append(
            {
                "physical_id": gpu_id,
                "gpu_type": config.gpu_type,
                "purpose": "_and_".join(purposes),
            }
        )
    return entries


def _worker_topology(config: EvolveConfig) -> Mapping[str, Any]:
    evaluation_shares_model_gpu = (
        config.kernel_gpu_id is not None
        and (
            config.kernel_gpu_id == config.training_gpu_id
            or config.kernel_gpu_id in config.gpu_ids
        )
    )
    return {
        "max_inflight_branches": config.evolve.workers.max_inflight_branches,
        "generation_backend": config.generation_backend,
        "tensor_parallel_size": config.vllm_tensor_parallel_size,
        "vllm_quantization": config.vllm_quantization,
        "training_gpu_id": config.training_gpu_id,
        "generation_gpu_ids": list(config.gpu_ids),
        "runtime_visible_gpu_ids": list(config.runtime_gpu_ids),
        "exclusive_evaluation_gpu_id": config.kernel_gpu_id,
        "evaluation_is_serialized_with_model_phase": evaluation_shares_model_gpu,
    }


def _environment_document() -> Mapping[str, Any]:
    return {
        key: os.environ[key]
        for key in (
            "CUDA_VISIBLE_DEVICES",
            "EVOLVE_ARTIFACT_RETENTION",
            "EVOLVE_CPU_CORES",
            "EVOLVE_GRACEFUL_STOP_MINUTES",
            "EVOLVE_HARD_DEADLINE_EPOCH",
            "EVOLVE_RUN_TIME_LIMIT",
            "PYTHONHASHSEED",
            "VLLM_USE_FLASHINFER_SAMPLER",
        )
        if key in os.environ
    }


__all__ = [
    "COMPONENT_SCHEMA_VERSIONS",
    "EngineError",
    "EngineWorkers",
    "EpochReport",
    "EpochState",
    "EvolveEngine",
    "build_production_workers",
]
