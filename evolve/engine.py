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

import json
import os
import platform
import shutil
import socket
import subprocess
import sys
from dataclasses import dataclass, field, replace
from pathlib import Path
from typing import Any, Callable, Dict, List, Mapping, Optional, Tuple, Union

from evolve.archive import (
    ArchiveAdmissionError,
    ConfirmedRecordTracker,
    ProvenanceStore,
    ScientificArchive,
)
from evolve.audits import (
    AuditEffectError,
    AuditPairingError,
    assign_audit_sides,
    close_audit_pair,
    compute_audit_effect,
    create_audit_pair,
    default_gain,
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
    RunLayout,
    atomic_write_json,
    atomic_write_text,
    create_fresh_run_layout,
    open_existing_run_layout,
    write_immutable_json,
    write_immutable_text,
    write_initial_run_metadata,
    write_resume_run_metadata,
)
from evolve.scheduler import AllocationPlan, PosteriorStore, plan_epoch
from evolve.types import (
    AllocationArm,
    AuditPair,
    AuditSide,
    BranchSpec,
    BudgetLedger,
    Channel,
    EpochManifest,
    FailureKind,
    Proposal,
    Role,
)
from evolve.verifier.adapters import ProblemScientificAdapter
from evolve.verifier.models import VerificationPolicy


class EngineError(RuntimeError):
    """The composed engine cannot proceed as configured."""


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
    persist_roles: Optional[Callable[[Any], None]] = None
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
        shutdown=runtime.shutdown,
    )


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
                "rng_seed": planned.rng_seed,
            }
            for planned in plan.planned_arms
        ],
    }
    return content_id("allocation_plan", payload)


def _plan_document(plan: AllocationPlan) -> Mapping[str, Any]:
    return {
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
                "marginal_gain": planned.marginal_gain,
                "rng_seed": planned.rng_seed,
            }
            for planned in plan.planned_arms
        ],
    }


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
    spec = option_registry.spec(option_id)
    identity = ArmIdentity(
        cell_id=arm.cell_id, role=arm.role, option_id=option_id,
        harness_id=arm.harness_id, horizon=arm.horizon, cost_class=arm.cost_class,
    )
    return make_allocation_arm(
        identity, channel=channel, expected_cost=spec.expected_cost, hard_cost=spec.hard_cost
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
    return make_allocation_arm(
        identity,
        channel=channel,
        expected_cost=spec.expected_cost,
        hard_cost=spec.hard_cost,
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


@dataclass(frozen=True)
class EpochReport:
    """Diagnostics returned from one committed epoch, for status/plots."""

    epoch: int
    plan: AllocationPlan
    branch_executions: Tuple[BranchExecution, ...]
    audit_pairs: Tuple[AuditPair, ...]
    record_improved: bool


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

    # -- run lifecycle -----------------------------------------------------

    def run(self) -> int:
        adapter = self._adapter or self._build_adapter()
        verification_policy = VerificationPolicy.create(
            version="evolve_engine_v1", production=not self.config.method_incomplete
        )
        layout, state = self._attach(adapter=adapter, verification_policy=verification_policy)
        workers = self._workers or build_production_workers(
            self.config, adapter=adapter, layout=layout, state=state
        )
        if workers.persist_roles is not None:
            workers.persist_roles(state)
        if self.metadata.get("mode") != "resume":
            self._commit_bootstrap(layout, state)
        target_epochs = self.config.evolve.budget.epochs
        try:
            while state.epoch < target_epochs:
                if workers.begin_epoch is not None:
                    workers.begin_epoch(state)
                state, report = self.run_epoch(
                    layout,
                    state,
                    workers=workers,
                    adapter=adapter,
                    verification_policy=verification_policy,
                )
                self._commit_barrier(
                    layout, state, report, adapter=adapter, workers=workers
                )
        except KeyboardInterrupt:
            self._write_status(layout, state, note="interrupted; last committed epoch preserved")
            return 130
        finally:
            if workers.shutdown is not None:
                workers.shutdown()
        self._write_status(layout, state, note="run complete")
        atomic_write_json(
            layout.path("final.summary.json"),
            {
                "run_id": state.run_id,
                "epochs_completed": state.epoch,
                "confirmed_record": state.record.internal_reward,
                "archive_coverage": state.archive.coverage,
                "budget": state.budget_ledger.to_dict(),
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
            return self._attach_resume()
        return self._attach_fresh(adapter=adapter, verification_policy=verification_policy)

    def _attach_fresh(
        self, *, adapter: ProblemScientificAdapter, verification_policy: VerificationPolicy
    ) -> Tuple[RunLayout, EpochState]:
        runs_root = self._runs_root
        layout = create_fresh_run_layout(
            runs_root, problem=self.config.problem, model_name=self.config.model_name
        )
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
        state = self._seed_archive(
            state, layout=layout, adapter=adapter, verification_policy=verification_policy
        )
        checkpoint = _checkpoint_payload(state)
        checkpoint_path = layout.path("checkpoints/checkpoint_epoch000.json")
        atomic_write_json(checkpoint_path, checkpoint)
        return layout, state

    def _commit_bootstrap(self, layout: RunLayout, state: EpochState) -> None:
        """Publish epoch zero only after all three role artifacts are durable."""

        checkpoint_path = layout.path("checkpoints/checkpoint_epoch000.json")
        checkpoint = json.loads(checkpoint_path.read_text(encoding="utf-8"))
        checkpoint_hash = content_hash(checkpoint)
        pointer = {
            "schema_version": 1,
            "epoch": 0,
            "committed_epoch": 0,
            "checkpoint": checkpoint_path.name,
            "checkpoint_hash": checkpoint_hash,
        }
        atomic_write_json(layout.path("checkpoints/latest.json"), pointer)
        atomic_write_json(layout.path("bootstrap.summary.json"), pointer)

    def _attach_resume(self) -> Tuple[RunLayout, EpochState]:
        resume_dir = Path(self.metadata["resume_dir"])
        layout = open_existing_run_layout(resume_dir, resume=True)
        checkpoint_path = _latest_completed_checkpoint(layout)
        checkpoint = json.loads(checkpoint_path.read_text(encoding="utf-8"))
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
            checkpoint_hash=content_hash(checkpoint),
        )
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
        from evolve.workers.verification import persist_answer_artifact

        problem = adapter.problem
        seed_branch_id = content_id("branch", {"kind": "seed", "run_id": state.run_id})
        archive = state.archive
        seeds = tuple(problem.seed_states())
        admitted_count = 0
        failures: List[str] = []
        for index, seed in enumerate(seeds):
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
            try:
                payload = problem.serialize_answer(candidate)
                artifact_path = persist_answer_artifact(
                    run_dir=layout.run_dir, problem_id=self.config.problem, payload=payload
                )
                persisted = PersistedAnswerPayload.create(
                    problem_id=self.config.problem, artifact_uri=str(artifact_path), payload=payload
                )
                result = verify_persisted_answer(
                    adapter=adapter,
                    proposal=proposal,
                    persisted_answer=persisted,
                    verification_policy=verification_policy,
                    harness_id=content_id("harness", {"kind": "seed"}),
                    policy_snapshot_id=content_id("role_snapshot", {"kind": "seed"}),
                )
            except Exception as exc:
                failures.append(f"seed {index}: {type(exc).__name__}: {exc}")
                continue
            if not result.evidence.admitted or result.state is None:
                failures.append(
                    f"seed {index}: verifier rejected: "
                    f"{result.evidence.failure_kind.value}: "
                    f"{result.evidence.diagnostics.get('message', '')}"
                )
                continue
            archive = archive.ensure_cell(result.descriptor, force_empty_sampling=False)
            try:
                archive, _decision = archive.offer(result.descriptor, proposal, result.state, result.evidence)
            except ArchiveAdmissionError as exc:
                failures.append(f"seed {index}: archive rejected: {exc}")
                continue
            admitted_count += 1
        if admitted_count == 0:
            detail = "; ".join(failures[:3]) or "problem returned no seed states"
            raise EngineError(
                f"{self.config.problem} bootstrap admitted no seeds "
                f"({len(seeds)} declared): {detail}"
            )
        return replace(state, archive=archive)

    def _confirm(
        self,
        proposal: Proposal,
        evidence: Any,
        *,
        adapter: ProblemScientificAdapter,
        verification_policy: VerificationPolicy,
    ):
        """Confirm a possible record by reverifying its saved payload only."""

        from evolve.verifier.models import PersistedAnswerPayload
        from evolve.verifier.service import confirm_persisted_answer

        persisted = PersistedAnswerPayload.create(
            problem_id=evidence.problem_id,
            artifact_uri=evidence.flags["answer_artifact_uri"],
            payload=evidence.answer_payload,
        )
        last_result = None
        errors = []
        for attempt in range(1, 3):
            try:
                result = confirm_persisted_answer(
                    adapter=adapter,
                    proposal=proposal,
                    persisted_answer=persisted,
                    prior_evidence=evidence,
                    verification_policy=verification_policy,
                )
            except Exception as exc:
                errors.append(f"{type(exc).__name__}: {exc}")
                continue
            last_result = result
            if result.evidence.failure_kind != FailureKind.INFRASTRUCTURE:
                return result, attempt
        if last_result is not None:
            return last_result, 2
        raise EngineError(
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
            "verifier_calls": state.budget_ledger.remaining("verifier_calls"),
        }
        plan = plan_epoch(
            epoch=epoch,
            archive=state.archive,
            option_registry=state.option_registry,
            harness_registry=state.harness_registry,
            posterior=state.posterior,
            roles=[Role(name) for name in settings.roles.enabled],
            max_inflight_branches=settings.workers.max_inflight_branches,
            audit_fraction=settings.budget.audit_fraction,
            no_memory_fraction=settings.audits.no_memory_fraction,
            refinement_fraction=settings.budget.refinement_fraction,
            harness_trial_fraction=settings.harnesses.trial_fraction,
            empty_cell_fraction=settings.archive.empty_cell_fraction,
            global_exploration_fraction=settings.scheduler.global_exploration_fraction,
            resource_limits=resource_limits,
            seed=epoch_seed,
        )
        # An allocation arm may execute a homogeneous replica group. This is
        # required for on-policy max-seeking learning: one rollout cannot
        # produce an OrderGrad rank advantage. Every role still receives at
        # least one branch, while one role rotates as the learning role.
        enabled_roles = [Role(name) for name in settings.roles.enabled]
        learning_role = enabled_roles[epoch % len(enabled_roles)]
        selected_plans = []
        projected_branches = 0
        production_capacity = max(
            0,
            settings.workers.max_inflight_branches
            - plan.reservation_slots.audit_branch_slots
            - plan.reservation_slots.refinement_slots
            - plan.reservation_slots.harness_trial_slots,
        )
        for role in enabled_roles:
            candidate = next(
                (item for item in plan.planned_arms if item.arm.role == role),
                None,
            )
            if candidate is None or candidate in selected_plans:
                continue
            replicas = settings.learning.group_k if role == learning_role else 1
            if projected_branches + replicas <= production_capacity:
                selected_plans.append(candidate)
                projected_branches += replicas
        for candidate in plan.planned_arms:
            if candidate in selected_plans:
                continue
            replicas = (
                settings.learning.group_k
                if candidate.arm.role == learning_role
                else 1
            )
            if projected_branches + replicas > production_capacity:
                continue
            selected_plans.append(candidate)
            projected_branches += replicas
        plan = replace(plan, planned_arms=tuple(selected_plans))

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
        step_dir = layout.path(f"step{epoch:02d}")
        step_dir.mkdir(parents=True, exist_ok=True)
        atomic_write_json(step_dir / "epoch.manifest.json", manifest.to_dict())
        atomic_write_json(step_dir / "allocation_plan.json", _plan_document(plan))

        executions: List[BranchExecution] = []
        group_members: List[GroupMember] = []
        arms_by_id: Dict[str, AllocationArm] = {}
        audit_pairs: List[AuditPair] = []
        audit_sides: Dict[str, AuditSide] = {}
        refinement_sources: List[Tuple[AllocationArm, BranchSpec, Any]] = []

        branch_ordinal = 0
        for allocation_index, planned in enumerate(plan.planned_arms):
            arm = planned.arm
            arms_by_id[arm.arm_id] = arm
            replica_count = (
                settings.learning.group_k if arm.role == learning_role else 1
            )
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
                except BudgetOverrun:
                    continue

                execution = self._execute_one_branch(
                    branch=branch,
                    arm=arm,
                    role_snapshot=role_snapshot,
                    option_registry=state.option_registry,
                    workers=workers,
                    start_verified=True,
                    cell_empty=planned.reservation == "empty_cell",
                )
                unused_verifications = float(
                    execution.outcome.unused_budget.get("verifier_calls", 0.0)
                )
                if unused_verifications > 0.0:
                    budget_ledger = BudgetService.refund(
                        budget_ledger,
                        resource="verifier_calls",
                        amount=unused_verifications,
                        transaction_key=f"{debit_key}:refund",
                        debit_transaction_key=debit_key,
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
                        and source_evidence.failure_kind
                        not in (FailureKind.INFRASTRUCTURE, FailureKind.TIMEOUT)
                        and source_evidence.evidence_id not in already_entered
                    ):
                        refinement_sources.append((arm, branch, observation))
                archive, provenance, record, posterior, budget_ledger = self._fold_execution(
                    execution=execution,
                    arm=arm,
                    identity_archive=archive,
                    provenance=provenance,
                    record=record,
                    posterior=posterior,
                    record_threshold=record_threshold,
                    adapter=adapter,
                    verification_policy=verification_policy,
                    budget_ledger=budget_ledger,
                )
                if execution.policy_trace is not None:
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
        pairs_to_run = min(audit_slots // 2, len(audit_candidates))
        for pair_index in range(pairs_to_run):
            planned = audit_candidates[pair_index]
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
            intervention_branch, intervention_snapshot = self._freeze_branch(
                state=state, arm=intervention_arm, role_snapshots=role_snapshots,
                record_threshold=record_threshold, index=1000 + pair_index * 2,
                epoch_seed=audit_seed, budget=arm.hard_cost, channel=Channel.AUDIT,
                verifier_id=adapter.verifier_id,
                verifier_version=adapter.verifier_version,
                memory_enabled=False,
            )
            control_branch, _ = self._freeze_branch(
                state=state, arm=control_arm, role_snapshots=role_snapshots,
                record_threshold=record_threshold, index=1000 + pair_index * 2 + 1,
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
            write_immutable_json(audit_dir / "pair.preassigned.json", pair.to_dict())
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
                continue

            intervention_execution = self._execute_one_branch(
                branch=intervention_branch, arm=intervention_arm, role_snapshot=intervention_snapshot,
                option_registry=state.option_registry, workers=workers,
                start_verified=True, cell_empty=False, memory_enabled=False,
            )
            control_execution = self._execute_one_branch(
                branch=control_branch, arm=control_arm, role_snapshot=intervention_snapshot,
                option_registry=state.option_registry, workers=workers,
                start_verified=True, cell_empty=False, memory_enabled=False,
            )
            self._persist_branch_execution(
                layout,
                branch=intervention_branch,
                execution=intervention_execution,
                ordinal=1000 + pair_index * 2,
            )
            for (debit_key, _side_arm), side_execution in zip(
                audit_debits, (intervention_execution, control_execution)
            ):
                unused = float(
                    side_execution.outcome.unused_budget.get("verifier_calls", 0.0)
                )
                if unused > 0.0:
                    budget_ledger = BudgetService.refund(
                        budget_ledger,
                        resource="verifier_calls",
                        amount=unused,
                        transaction_key=f"{debit_key}:refund",
                        debit_transaction_key=debit_key,
                    )
            self._persist_branch_execution(
                layout,
                branch=control_branch,
                execution=control_execution,
                ordinal=1000 + pair_index * 2 + 1,
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
                    identity_archive=archive, provenance=provenance, record=record, posterior=posterior,
                    record_threshold=record_threshold, adapter=adapter, verification_policy=verification_policy,
                    budget_ledger=budget_ledger,
                )
                if execution.policy_trace is not None:
                    group_members.append(
                        GroupMember(branch=side_branch, outcome=execution.outcome, trace=execution.policy_trace)
                    )

            try:
                closed_pair = close_audit_pair(
                    pair, intervention_outcome=intervention_execution.outcome,
                    control_outcome=control_execution.outcome,
                )
            except AuditEffectError:
                audit_pairs.append(pair)
                continue
            audit_pairs.append(closed_pair)

            intervention_gain = default_gain(intervention_execution.outcome, frozen_record_threshold=record_threshold)
            control_gain = default_gain(control_execution.outcome, frozen_record_threshold=record_threshold)
            effect = compute_audit_effect(
                closed_pair, intervention_gain=intervention_gain, control_gain=control_gain
            )
            context = {"role": arm.role.value, "cell_id": arm.cell_id}
            record_id = memory_id_for(context=context, intervention_option_id=closed_pair.intervention_option_id)
            memory_record = memory.get(record_id) or new_memory_record(
                context=context, intervention_option_id=closed_pair.intervention_option_id,
                scope="cell", recency_epoch=epoch,
                promotion_min_support=settings.audits.min_pairs_for_promotion,
            )
            memory_record = add_effect(memory_record, pair=closed_pair, effect=effect, recency_epoch=epoch)
            memory_record = stratify_drift(memory_record, current_epoch=epoch)
            memory_record = evaluate_promotion(memory_record)
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
        refinement_pairs_to_run = min(
            plan.reservation_slots.refinement_slots // 2,
            len(refinement_sources),
        )
        if repair_option_id is None or fresh_option_id is None:
            refinement_pairs_to_run = 0
        nursery_policy = NurseryPolicy(
            max_attempts=settings.refinement.max_attempts,
            max_depth=settings.refinement.max_depth,
            fixed_cost={"verifier_calls": 1.0},
            ttl_epochs=2,
        )
        for pair_index in range(refinement_pairs_to_run):
            source_arm, source_branch, source_observation = refinement_sources[
                pair_index
            ]
            source_evidence = source_observation.verification.evidence
            entry = open_entry(
                source_evidence=source_evidence,
                branch_id=source_branch.branch_id,
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
            frozen_refinement = {
                "refinement_source": source_observation.proposal.source_text,
                "refinement_source_evidence_id": source_evidence.evidence_id,
                "refinement_diagnostics": dict(source_evidence.diagnostics),
            }
            parent_state_id = source_observation.proposal.parent_state_id
            if parent_state_id is None:
                continue
            intervention_branch, challenger_snapshot = self._freeze_branch(
                state=state,
                arm=intervention_arm,
                role_snapshots=role_snapshots,
                record_threshold=record_threshold,
                index=3000 + pair_index * 2,
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
                index=3000 + pair_index * 2 + 1,
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
            write_immutable_json(
                refinement_dir / "entry.opened.json", vars(entry)
            )
            write_immutable_json(
                refinement_dir / "pair.preassigned.json", pair.to_dict()
            )
            required = sum(
                float(item.hard_cost.get("verifier_calls", 1.0))
                for item in (intervention_arm, control_arm)
            )
            if required > budget_ledger.remaining("verifier_calls"):
                nursery[entry.entry_id] = entry
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
            intervention_execution = self._execute_one_branch(
                branch=intervention_branch,
                arm=intervention_arm,
                role_snapshot=challenger_snapshot,
                option_registry=state.option_registry,
                workers=workers,
                start_verified=True,
                cell_empty=False,
                memory_enabled=False,
            )
            control_execution = self._execute_one_branch(
                branch=control_branch,
                arm=control_arm,
                role_snapshot=challenger_snapshot,
                option_registry=state.option_registry,
                workers=workers,
                start_verified=True,
                cell_empty=False,
                memory_enabled=False,
            )
            for offset, (refinement_branch, refinement_arm, execution) in enumerate(
                (
                    (intervention_branch, intervention_arm, intervention_execution),
                    (control_branch, control_arm, control_execution),
                )
            ):
                self._persist_branch_execution(
                    layout,
                    branch=refinement_branch,
                    execution=execution,
                    ordinal=3000 + pair_index * 2 + offset,
                )
                executions.append(execution)
                arms_by_id[refinement_arm.arm_id] = refinement_arm
                archive, provenance, record, posterior, budget_ledger = self._fold_execution(
                    execution=execution,
                    arm=refinement_arm,
                    identity_archive=archive,
                    provenance=provenance,
                    record=record,
                    posterior=posterior,
                    record_threshold=record_threshold,
                    adapter=adapter,
                    verification_policy=verification_policy,
                    budget_ledger=budget_ledger,
                )
            for debit_key, execution in zip(
                debit_records, (intervention_execution, control_execution)
            ):
                unused = float(
                    execution.outcome.unused_budget.get("verifier_calls", 0.0)
                )
                if unused > 0.0:
                    budget_ledger = BudgetService.refund(
                        budget_ledger,
                        resource="verifier_calls",
                        amount=unused,
                        transaction_key=f"{debit_key}:refund",
                        debit_transaction_key=debit_key,
                    )
            repair_execution = (
                intervention_execution
                if intervention_arm.option_id == repair_option_id
                else control_execution
            )
            if repair_execution.observations:
                entry = record_attempt(
                    entry,
                    repair_evidence=(
                        repair_execution.observations[0].verification.evidence
                    ),
                    epoch=epoch,
                )
            nursery[entry.entry_id] = entry
            write_immutable_json(
                refinement_dir / "entry.closed.json", vars(entry)
            )
            try:
                closed_pair = close_audit_pair(
                    pair,
                    intervention_outcome=intervention_execution.outcome,
                    control_outcome=control_execution.outcome,
                )
            except AuditEffectError:
                audit_pairs.append(pair)
                continue
            audit_pairs.append(closed_pair)
            effect = compute_audit_effect(
                closed_pair,
                intervention_gain=default_gain(
                    intervention_execution.outcome,
                    frozen_record_threshold=record_threshold,
                ),
                control_gain=default_gain(
                    control_execution.outcome,
                    frozen_record_threshold=record_threshold,
                ),
            )
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
            memory = memory.upsert(evaluate_promotion(stratify_drift(
                memory_record, current_epoch=epoch
            )))

        # -- matched harness calibration ---------------------------------
        inactive_harnesses = tuple(
            harness_id
            for harness_id in sorted(harness_registry.specs)
            if harness_id not in harness_registry.active_ids
        )
        harness_pairs_to_run = min(
            plan.reservation_slots.harness_trial_slots // 2,
            1 if plan.planned_arms and inactive_harnesses else 0,
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
            incumbent_branch, snapshot = self._freeze_branch(
                state=state,
                arm=incumbent_arm,
                role_snapshots=role_snapshots,
                record_threshold=record_threshold,
                index=2000 + pair_index * 2,
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
                index=2000 + pair_index * 2 + 1,
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
            write_immutable_json(
                harness_dir / "harness.preassigned.json", context.to_dict()
            )
            required = sum(
                float(item.hard_cost.get("verifier_calls", 1.0))
                for item in (incumbent_arm, candidate_arm)
            )
            if required > budget_ledger.remaining("verifier_calls"):
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

            incumbent_execution = self._execute_one_branch(
                branch=incumbent_branch,
                arm=incumbent_arm,
                role_snapshot=snapshot,
                option_registry=state.option_registry,
                workers=workers,
                start_verified=True,
                cell_empty=False,
                memory_enabled=False,
            )
            candidate_execution = self._execute_one_branch(
                branch=candidate_branch,
                arm=candidate_arm,
                role_snapshot=snapshot,
                option_registry=state.option_registry,
                workers=workers,
                start_verified=True,
                cell_empty=False,
                memory_enabled=False,
            )
            for offset, (trial_branch, trial_arm, trial_execution) in enumerate(
                (
                    (incumbent_branch, incumbent_arm, incumbent_execution),
                    (candidate_branch, candidate_arm, candidate_execution),
                )
            ):
                self._persist_branch_execution(
                    layout,
                    branch=trial_branch,
                    execution=trial_execution,
                    ordinal=2000 + pair_index * 2 + offset,
                )
                executions.append(trial_execution)
                arms_by_id[trial_arm.arm_id] = trial_arm
                archive, provenance, record, posterior, budget_ledger = self._fold_execution(
                    execution=trial_execution,
                    arm=trial_arm,
                    identity_archive=archive,
                    provenance=provenance,
                    record=record,
                    posterior=posterior,
                    record_threshold=record_threshold,
                    adapter=adapter,
                    verification_policy=verification_policy,
                    budget_ledger=budget_ledger,
                )
            for (debit_key, _trial_arm), trial_execution in zip(
                debit_records, (incumbent_execution, candidate_execution)
            ):
                unused = float(
                    trial_execution.outcome.unused_budget.get(
                        "verifier_calls", 0.0
                    )
                )
                if unused > 0.0:
                    budget_ledger = BudgetService.refund(
                        budget_ledger,
                        resource="verifier_calls",
                        amount=unused,
                        transaction_key=f"{debit_key}:refund",
                        debit_transaction_key=debit_key,
                    )
            trial = HarnessTrialRecord.from_context(
                context,
                epoch=epoch,
                incumbent_gain=default_gain(
                    incumbent_execution.outcome,
                    frozen_record_threshold=record_threshold,
                ),
                candidate_gain=default_gain(
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
            harness_registry = harness_registry.record_trial(trial)
            try:
                harness_registry = harness_registry.promote(
                    candidate_harness_id,
                    min_trials=settings.audits.min_pairs_for_promotion,
                )
            except HarnessPromotionError:
                pass

        # -- role-isolated learning ------------------------------------
        traces_by_id = {member.trace.trace_id: member.trace for member in group_members}
        groups = build_learning_groups(
            group_members,
            arms=arms_by_id,
            objective=_objective_enum(settings.learning.objective),
            top_m=settings.learning.top_m,
            group_k=settings.learning.group_k,
            audit_sides=audit_sides,
        )
        learning_dir = step_dir / "learning"
        learning_dir.mkdir(parents=True, exist_ok=True)
        for group in groups:
            write_immutable_json(learning_dir / f"{group.group_id}.inputs.json", group.to_dict())
        for trace in traces_by_id.values():
            write_immutable_json(learning_dir / f"{trace.trace_id}.trace.json", trace.to_dict())
        updates, role_registry = train_barrier(
            groups, traces_by_id=traces_by_id, registry=role_registry, epoch=epoch,
            gradient_step=workers.gradient_step, kl_penalty_coef=self.config.kl_penalty_coef,
        )
        for update in updates:
            write_immutable_json(
                learning_dir / f"{update.role_snapshot_id_before}.update.json",
                {
                    "role_snapshot_id_before": update.role_snapshot_id_before,
                    "adapter_hash_before": update.adapter_hash_before,
                    "adapter_hash_after": update.adapter_hash_after,
                    "group_ids": [group.group_id for group in update.groups],
                    "loss": update.result.loss,
                    "kl": update.result.kl,
                    "gradient_norm": update.result.gradient_norm,
                },
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
        write_immutable_json(branch_dir / "branch.spec.json", branch.to_dict())
        for observation_index, observation in enumerate(execution.observations):
            prefix = f"observation{observation_index:03d}"
            segment = observation.policy_segment
            if segment is not None:
                write_immutable_text(branch_dir / f"{prefix}.prompt.txt", segment.prompt)
                write_immutable_text(
                    branch_dir / f"{prefix}.response.txt", segment.response_segment
                )
            else:
                write_immutable_text(
                    branch_dir / f"{prefix}.response.txt",
                    observation.proposal.source_text,
                )
            write_immutable_json(
                branch_dir / f"{prefix}.proposal.json",
                observation.proposal.to_dict(),
            )
            write_immutable_json(
                branch_dir / f"{prefix}.evidence.json",
                observation.verification.evidence.to_dict(),
            )
            if observation.verification.state is not None:
                write_immutable_json(
                    branch_dir / f"{prefix}.state.json",
                    observation.verification.state.to_dict(),
                )
            flat = f"step{branch.epoch:02d}_group{ordinal:04d}_rollout{observation_index:03d}"
            if segment is not None:
                write_immutable_text(step_dir / f"{flat}.prompt.txt", segment.prompt)
            write_immutable_text(step_dir / f"{flat}.txt", observation.proposal.source_text)
            write_immutable_json(
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
        write_immutable_json(branch_dir / "branch.outcome.json", execution.outcome.to_dict())
        if execution.policy_trace is not None:
            write_immutable_json(
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
        raise EngineError(f"cell {cell_id} has no verified state to branch from")

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

    def _fold_execution(
        self,
        *,
        execution: BranchExecution,
        arm: AllocationArm,
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
            except ArchiveAdmissionError:
                continue
            # A possible new record is confirmed by reverifying its saved
            # payload only -- proposal code is never rerun.
            if (
                evidence.internal_reward is not None
                and evidence.internal_reward > record_threshold
            ):
                confirmation_debit = (
                    f"epoch-confirm:{execution.outcome.branch_id}:"
                    f"{evidence.evidence_id}"
                )
                # Confirmation may make one bounded infrastructure retry. Reserve
                # both calls first so a possible record can never overrun the
                # global verifier ledger, then return the unused retry.
                try:
                    budget_ledger = BudgetService.debit(
                        budget_ledger,
                        resource="verifier_calls",
                        amount=2.0,
                        transaction_key=confirmation_debit,
                    )
                except BudgetOverrun:
                    continue
                confirmation, attempts = self._confirm(
                    result.proposal, evidence, adapter=adapter, verification_policy=verification_policy
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
                        result.proposal, confirmation.evidence
                    ),
                )
                if confirmation.evidence.confirmed and confirmation.state is not None:
                    try:
                        archive, decision = archive.offer(
                            confirmation.descriptor, result.proposal, confirmation.state, confirmation.evidence
                        )
                    except ArchiveAdmissionError:
                        continue
                    record = record.consider(
                        confirmation.state, confirmation.evidence, archive=archive
                    )
        if execution.provenance_edges:
            provenance = provenance.with_artifacts(archive.artifacts)
            for edge in execution.provenance_edges:
                try:
                    provenance = provenance.append(edge)
                except Exception:
                    continue

        from evolve.scheduler.arms import ArmIdentity

        identity = ArmIdentity.from_arm(arm)
        maximum_reward = execution.outcome.maximum_reward
        record_improved = maximum_reward is not None and maximum_reward > record_threshold
        gain = max(0.0, (maximum_reward if maximum_reward is not None else record_threshold) - record_threshold)
        posterior = posterior.observe(
            identity,
            admitted=execution.outcome.eligible_for_scheduler and execution.outcome.maximum_state_id is not None,
            infrastructure=execution.outcome.infrastructure_aborted,
            record_improved=record_improved,
            gain=gain,
            costs=execution.outcome.costs,
        )
        return archive, provenance, record, posterior, budget_ledger

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
        if workers.persist_roles is not None:
            workers.persist_roles(state)
        checkpoint_dir = layout.path("checkpoints")
        checkpoint_dir.mkdir(parents=True, exist_ok=True)
        checkpoint = _checkpoint_payload(state)
        checkpoint_path = checkpoint_dir / f"checkpoint_epoch{state.epoch:03d}.json"
        atomic_write_json(checkpoint_path, checkpoint)
        checkpoint_hash = content_hash(checkpoint)
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
            "reservation_slots": {
                "audit_branch_slots": report.plan.reservation_slots.audit_branch_slots,
                "refinement_slots": report.plan.reservation_slots.refinement_slots,
                "harness_trial_slots": report.plan.reservation_slots.harness_trial_slots,
                "empty_cell_slots": report.plan.reservation_slots.empty_cell_slots,
                "global_exploration_slots": report.plan.reservation_slots.global_exploration_slots,
                "role_guaranteed_slots": report.plan.reservation_slots.role_guaranteed_slots,
                "remaining_production_slots": report.plan.reservation_slots.remaining_production_slots,
            },
            "arms_by_role": {
                role: sum(1 for planned in report.plan.planned_arms if planned.arm.role.value == role)
                for role in sorted({planned.arm.role.value for planned in report.plan.planned_arms})
            },
        }
        summary.update(
            {
                "schema_version": 1,
                "committed_epoch": state.epoch,
                "checkpoint": checkpoint_path.name,
                "checkpoint_hash": checkpoint_hash,
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
        atomic_write_json(step_dir / f"step{report.epoch:02d}.summary.json", summary)
        if report.record_improved:
            self._publish_best(layout, state, adapter=adapter)
        self._write_status(
            layout, state, note=f"epoch {report.epoch} committed", report=report
        )
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
                    ),
                    out_dir=layout.path("plots"),
                )
            except Exception:
                # Barrier durability and discovery never depend on plotting.
                pass

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
        write_immutable_json(
            layout.path(f"archive/snapshots/epoch{state.epoch:03d}.json"),
            archive_document,
        )
        write_immutable_json(
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
            return
        try:
            evidence = state.archive.artifacts.evidence_packet(record.evidence_id)
            verified_state = state.archive.artifacts.state_binding(
                record.state_id, evidence.proposal_id, record.evidence_id
            )
        except Exception:
            return
        best_dir = layout.path("best")
        best_dir.mkdir(parents=True, exist_ok=True)
        staging = layout.path(f".best-{verified_state.state_id}")
        if staging.exists():
            shutil.rmtree(staging)
        staging.mkdir(parents=True, exist_ok=False)
        try:
            atomic_write_json(staging / "state.json", verified_state.to_dict())
            atomic_write_json(staging / "evidence.json", evidence.to_dict())
            atomic_write_json(
                staging / "candidate.json",
                {
                    "state_id": verified_state.state_id,
                    "proposal_id": verified_state.proposal_id,
                    "answer_payload": verified_state.answer_payload,
                },
            )
            try:
                adapter.problem.render_best(
                    verified_state.answer_payload, evidence, staging
                )
            except Exception as exc:
                atomic_write_json(
                    staging / "answer.json", verified_state.answer_payload
                )
                atomic_write_text(
                    staging / "renderer.error.txt",
                    f"{type(exc).__name__}: {exc}\n",
                )
            for path in sorted(staging.iterdir()):
                os.replace(path, best_dir / path.name)
        finally:
            if staging.exists():
                shutil.rmtree(staging)
        proposal = state.archive.artifacts.proposal(verified_state.proposal_id)
        atomic_write_text(layout.run_dir / "best_code.py", proposal.source_text)
        atomic_write_json(
            layout.run_dir / "best_construction.json",
            {"answer_payload": evidence.answer_payload, "internal_reward": evidence.internal_reward},
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
            except Exception:
                holder = None
        outcomes = tuple(report.branch_executions) if report is not None else ()
        status = {
            "schema_version": 1,
            "run_id": state.run_id,
            "epoch": state.epoch,
            "confirmed_record": {
                "state_id": state.record.state_id,
                "cell_id": state.record.cell_id,
                "internal_reward": state.record.internal_reward,
                "raw_score": state.record.raw_score,
                "holder": holder,
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
            "causal_memory": {
                "records": len(state.memory.records),
                "promoted": sum(
                    1
                    for record in state.memory.records.values()
                    if record.status.value == "promoted"
                ),
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
            },
            "note": note,
        }
        atomic_write_json(layout.path("status.json"), status)


def _objective_enum(value: str):
    from evolve.types import LearningObjective

    return LearningObjective(value)


def _checkpoint_payload(state: EpochState) -> Mapping[str, Any]:
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
    }


def _state_from_checkpoint(checkpoint: Mapping[str, Any], *, config: EvolveConfig) -> EpochState:
    from evolve.archive import ScientificArtifactStore
    from evolve.types import ArchiveCell, Descriptor, EvidencePacket, Proposal, ProvenanceEdge, VerifiedScientificState, CausalMemoryRecord

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
    )


def _latest_completed_checkpoint(layout: RunLayout) -> Path:
    """Resolve the newest checkpoint advertised by a durable completion marker."""

    summaries = []
    bootstrap = layout.path("bootstrap.summary.json")
    if bootstrap.is_file():
        summaries.append(bootstrap)
    summaries.extend(layout.run_dir.glob("step*/step*.summary.json"))
    candidates = []
    for summary_path in summaries:
        try:
            summary = json.loads(summary_path.read_text(encoding="utf-8"))
            epoch = int(summary.get("committed_epoch", summary.get("epoch", -1)))
            filename = summary["checkpoint"]
            expected_hash = summary["checkpoint_hash"]
            path = layout.path(f"checkpoints/{filename}")
            document = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, ValueError, KeyError, json.JSONDecodeError):
            continue
        if content_hash(document) == expected_hash:
            candidates.append((epoch, path))
    if not candidates:
        raise EngineError(
            f"no completed-barrier checkpoint found in {layout.run_dir}"
        )
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
                if gpu_id in config.runtime_gpu_ids
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
    return {
        "max_inflight_branches": config.evolve.workers.max_inflight_branches,
        "generation_backend": config.generation_backend,
        "tensor_parallel_size": config.vllm_tensor_parallel_size,
        "vllm_quantization": config.vllm_quantization,
        "training_gpu_id": config.training_gpu_id,
        "generation_gpu_ids": list(config.gpu_ids),
        "runtime_visible_gpu_ids": list(config.runtime_gpu_ids),
        "exclusive_evaluation_gpu_id": config.kernel_gpu_id,
        "evaluation_is_serialized_with_model_phase": (
            config.kernel_gpu_id in config.runtime_gpu_ids
            if config.kernel_gpu_id is not None
            else False
        ),
    }


def _environment_document() -> Mapping[str, Any]:
    return {
        key: os.environ[key]
        for key in (
            "CUDA_VISIBLE_DEVICES",
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
