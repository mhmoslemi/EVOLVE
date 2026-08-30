"""Pure branch execution: drive one frozen option to a :class:`BranchOutcome`.

This module never calls a model, sandbox, or verifier directly.  A caller
supplies a :data:`BranchStepExecutor` callback that turns one planned option
action into an actual generated :class:`~evolve.types.Proposal` and its
:class:`~evolve.verifier.evidence.ScientificVerificationResult`; this module
owns only the deterministic loop, cost bookkeeping, provenance construction,
and :class:`~evolve.types.BranchOutcome`/:class:`~evolve.types.PolicyTrace`
assembly around that callback.  Keeping the model-touching side effect behind
an injected callback is what makes the branch executor itself CPU-testable.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Mapping, Optional, Sequence, Tuple

from evolve.ids import content_hash, content_id
from evolve.types import (
    AllocationArm,
    BranchOutcome,
    BranchSpec,
    BranchStatus,
    FailureKind,
    PolicyTrace,
    Proposal,
    ProvenanceEdge,
    RoleSnapshot,
)
from evolve.archive.provenance import make_provenance_edge
from evolve.verifier.evidence import ScientificVerificationResult

from .base import (
    ExecutableOption,
    OptionContext,
    OptionDecisionKind,
    OptionExecutionError,
    OptionStepInput,
)


class BranchExecutionError(OptionExecutionError):
    """A branch step executor returned a result inconsistent with its branch."""


@dataclass(frozen=True)
class PolicySegment:
    """One role-policy decision captured for later :class:`PolicyTrace` use."""

    prompt: str
    response_segment: str
    token_mask: Tuple[bool, ...]
    log_probabilities: Tuple[float, ...]
    token_ids: Tuple[int, ...] = ()

    def __post_init__(self) -> None:
        if len(self.token_mask) != len(self.log_probabilities):
            raise BranchExecutionError(
                "policy segment token_mask and log_probabilities must align"
            )
        if self.token_ids and len(self.token_ids) != len(self.token_mask):
            raise BranchExecutionError(
                "policy segment token_ids and token_mask must align"
            )


@dataclass(frozen=True)
class BranchStepRequest:
    """Everything the injected executor needs to realize one planned action."""

    branch: BranchSpec
    arm: AllocationArm
    action: str
    capability: str
    prompt_metadata: Mapping[str, Any]
    step_index: int
    parent_state_id: str
    cumulative_cost: Mapping[str, float]


@dataclass(frozen=True)
class BranchStepResult:
    """The executor's realization of one requested branch step."""

    proposal: Proposal
    verification: ScientificVerificationResult
    costs: Mapping[str, float] = field(default_factory=dict)
    policy_segment: Optional[PolicySegment] = None
    novelty: float = 0.0
    uncertainty: float = 0.0

    def __post_init__(self) -> None:
        if not isinstance(self.proposal, Proposal):
            raise BranchExecutionError("step result requires its generating Proposal")
        if not isinstance(self.verification, ScientificVerificationResult):
            raise BranchExecutionError("step result requires a ScientificVerificationResult")
        if self.proposal.proposal_id != self.verification.evidence.proposal_id:
            raise BranchExecutionError("step result proposal does not match its own evidence")
        for resource, amount in self.costs.items():
            if not isinstance(resource, str) or not resource.strip():
                raise BranchExecutionError("step cost resource names must be non-empty strings")
            if (
                isinstance(amount, bool)
                or not isinstance(amount, (int, float))
                or not math.isfinite(float(amount))
                or amount < 0.0
            ):
                raise BranchExecutionError(f"step cost for {resource!r} must be non-negative")
        object.__setattr__(self, "costs", dict(self.costs))


BranchStepExecutor = Callable[[BranchStepRequest], BranchStepResult]


@dataclass(frozen=True)
class BranchExecution:
    """The complete durable product of running one branch to closure."""

    outcome: BranchOutcome
    observations: Tuple[BranchStepResult, ...]
    provenance_edges: Tuple[ProvenanceEdge, ...]
    policy_trace: Optional[PolicyTrace]


def build_option_context(
    *,
    branch: BranchSpec,
    arm: AllocationArm,
    available_capabilities: Sequence[str] = ("propose",),
    satisfied_prerequisites: Sequence[str] = (),
    start_verified: bool = True,
    cell_empty: bool = False,
    diagnostics_available: bool = False,
    matched_control: bool = False,
    memory_enabled: bool = True,
    metadata: Optional[Mapping[str, Any]] = None,
) -> OptionContext:
    """Assemble the frozen structural context checked before an option starts."""

    return OptionContext(
        role=arm.role,
        cell_id=arm.cell_id,
        harness_id=branch.harness_id,
        requested_horizon=branch.horizon,
        available_capabilities=tuple(available_capabilities),
        satisfied_prerequisites=tuple(satisfied_prerequisites),
        budget=dict(branch.budget),
        start_state_id=branch.start_state_id,
        start_verified=start_verified,
        cell_empty=cell_empty,
        diagnostics_available=diagnostics_available,
        matched_control=matched_control,
        memory_enabled=memory_enabled,
        metadata=dict(metadata or {}),
    )


def _branch_outcome_id(
    *,
    branch_id: str,
    branch_spec_hash: str,
    status: BranchStatus,
    descendant_proposal_ids: Tuple[str, ...],
    descendant_state_ids: Tuple[str, ...],
    evidence_ids: Tuple[str, ...],
    maximum_state_id: Optional[str],
    maximum_evidence_id: Optional[str],
    maximum_reward: Optional[float],
    costs: Mapping[str, float],
    unused_budget: Mapping[str, float],
    eligible_for_scheduler: bool,
    infrastructure_aborted: bool,
) -> str:
    return content_id(
        "branch_outcome",
        {
            "branch_id": branch_id,
            "branch_spec_hash": branch_spec_hash,
            "status": status.value,
            "descendant_proposal_ids": list(descendant_proposal_ids),
            "descendant_state_ids": list(descendant_state_ids),
            "evidence_ids": list(evidence_ids),
            "maximum_state_id": maximum_state_id,
            "maximum_evidence_id": maximum_evidence_id,
            "maximum_reward": maximum_reward,
            "costs": dict(costs),
            "unused_budget": dict(unused_budget),
            "eligible_for_scheduler": eligible_for_scheduler,
            "infrastructure_aborted": infrastructure_aborted,
        },
    )


def _accumulate(cost: Dict[str, float], addition: Mapping[str, float]) -> None:
    for resource, amount in addition.items():
        cost[resource] = cost.get(resource, 0.0) + float(amount)


def execute_branch(
    *,
    branch: BranchSpec,
    arm: AllocationArm,
    option: ExecutableOption,
    context: OptionContext,
    role_snapshot: RoleSnapshot,
    executor: BranchStepExecutor,
) -> BranchExecution:
    """Drive one frozen branch's option to closure via an injected executor.

    The loop never inspects model output itself: every admission, reward, and
    diagnostic signal comes from the executor's
    :class:`~evolve.verifier.evidence.ScientificVerificationResult`, so the
    common verifier alone still controls admission and the record.
    """

    if branch.arm_id != arm.arm_id:
        raise BranchExecutionError("branch does not reference its allocation arm")
    if branch.option_id != option.spec.option_id:
        raise BranchExecutionError("branch does not reference its frozen option")
    if branch.role_snapshot_id != role_snapshot.snapshot_id:
        raise BranchExecutionError("branch does not reference its frozen role snapshot")
    if arm.role != role_snapshot.role:
        raise BranchExecutionError("allocation arm role does not match its role snapshot")
    if context.requested_horizon != branch.horizon:
        raise BranchExecutionError("option context horizon does not match the frozen branch")
    if context.cell_id != arm.cell_id or context.harness_id != branch.harness_id:
        raise BranchExecutionError("option context does not match its frozen branch/arm")

    branch_spec_hash = content_hash(branch.to_dict())
    state = option.start(context)
    step_input = OptionStepInput(step_index=0, cumulative_cost={})

    # Actual resource cost includes infrastructure retries and is used for the
    # global ledger/refund. Option hard bounds count logical scientific steps,
    # so a verifier retry cannot silently shorten a branch's frozen horizon.
    cumulative_cost: Dict[str, float] = {}
    option_cost: Dict[str, float] = {}
    observations: List[BranchStepResult] = []
    provenance_edges: List[ProvenanceEdge] = []
    prompts: List[str] = []
    responses: List[str] = []
    masks: List[Tuple[bool, ...]] = []
    log_probabilities: List[Tuple[float, ...]] = []
    token_ids: List[Tuple[int, ...]] = []
    parent_state_id = branch.start_state_id
    stop_reason: Optional[str] = None
    step_count = 0

    while True:
        decision = option.step(state, step_input)
        if decision.kind == OptionDecisionKind.STOP:
            stop_reason = decision.stop_reason
            state = decision.next_state
            break
        step_count += 1
        if step_count > option.spec.max_horizon:
            raise BranchExecutionError(
                "branch executor exceeded the option's max_horizon without stopping"
            )

        request = BranchStepRequest(
            branch=branch,
            arm=arm,
            action=decision.action,
            capability=decision.capability,
            prompt_metadata=dict(decision.prompt_metadata),
            step_index=state.step_index,
            parent_state_id=parent_state_id,
            cumulative_cost=dict(option_cost),
        )
        result = executor(request)
        if not isinstance(result, BranchStepResult):
            raise BranchExecutionError("branch step executor must return BranchStepResult")

        evidence = result.verification.evidence
        if evidence.branch_id != branch.branch_id:
            raise BranchExecutionError("step evidence does not reference this branch")
        if evidence.parent_state_id != parent_state_id:
            raise BranchExecutionError("step evidence does not extend the branch lineage")
        if result.proposal.branch_id != branch.branch_id:
            raise BranchExecutionError("step proposal does not reference this branch")
        if result.proposal.parent_state_id != parent_state_id:
            raise BranchExecutionError("step proposal does not extend the branch lineage")

        _accumulate(cumulative_cost, result.costs)
        logical_cost = dict(result.costs)
        if "verifier_calls" in logical_cost:
            logical_cost["verifier_calls"] = 1.0
        _accumulate(option_cost, logical_cost)
        observations.append(result)
        if result.policy_segment is not None:
            prompts.append(result.policy_segment.prompt)
            responses.append(result.policy_segment.response_segment)
            masks.append(result.policy_segment.token_mask)
            log_probabilities.append(result.policy_segment.log_probabilities)
            token_ids.append(result.policy_segment.token_ids)

        admitted = evidence.admitted
        confirmed = evidence.confirmed
        record_improved = bool(
            admitted
            and evidence.internal_reward is not None
            and float(evidence.internal_reward) > branch.frozen_record_threshold
        )
        if admitted:
            state_obj = result.verification.state
            assert state_obj is not None
            provenance_edges.append(
                make_provenance_edge(
                    parent_state_id=parent_state_id,
                    child_state_id=state_obj.state_id,
                    proposal_id=evidence.proposal_id,
                    evidence_id=evidence.evidence_id,
                    branch_id=branch.branch_id,
                    relation="duplicate" if state_obj.state_id == parent_state_id else "descendant",
                )
            )
            parent_state_id = state_obj.state_id

        if bool(evidence.flags.get("excluded_from_scientific_updates", False)):
            stop_reason = "infrastructure_failure"
            break

        exceeded = [
            resource
            for resource, limit in branch.budget.items()
            if cumulative_cost.get(resource, 0.0) > float(limit) + 1e-9
        ]
        if exceeded:
            stop_reason = "hard_budget_exceeded"
            break

        state = decision.next_state
        step_input = OptionStepInput(
            step_index=state.step_index,
            cumulative_cost=dict(option_cost),
            latest_state_id=result.verification.state.state_id if admitted else None,
            latest_evidence_id=evidence.evidence_id if admitted else None,
            admitted=admitted,
            confirmed=confirmed,
            record_improved=record_improved,
            diagnostics_available=bool(evidence.diagnostics),
            infrastructure_failed=evidence.failure_kind == FailureKind.INFRASTRUCTURE,
            failure_kind=evidence.failure_kind.value,
            novelty=max(0.0, float(result.novelty)),
            uncertainty=max(0.0, float(result.uncertainty)),
        )

    infrastructure_aborted = stop_reason in (
        "infrastructure_failure",
        "hard_budget_exceeded",
    )
    descendant_proposal_ids = tuple(
        result.verification.evidence.proposal_id for result in observations
    )
    evidence_ids = tuple(result.verification.evidence.evidence_id for result in observations)
    admitted_results = [
        result for result in observations if result.verification.evidence.admitted
    ]
    descendant_state_ids: Tuple[str, ...] = tuple(
        dict.fromkeys(result.verification.state.state_id for result in admitted_results)
    )

    maximum_state_id = maximum_evidence_id = None
    maximum_reward: Optional[float] = None
    if admitted_results and not infrastructure_aborted:
        best = max(
            admitted_results,
            key=lambda result: (
                float(result.verification.evidence.internal_reward),
                result.verification.evidence.evidence_id,
            ),
        )
        maximum_state_id = best.verification.state.state_id
        maximum_evidence_id = best.verification.evidence.evidence_id
        maximum_reward = float(best.verification.evidence.internal_reward)

    unused_budget = {
        resource: max(0.0, float(limit) - cumulative_cost.get(resource, 0.0))
        for resource, limit in branch.budget.items()
    }
    status = BranchStatus.ABORTED if infrastructure_aborted else BranchStatus.CLOSED
    eligible_for_scheduler = not infrastructure_aborted

    outcome_id = _branch_outcome_id(
        branch_id=branch.branch_id,
        branch_spec_hash=branch_spec_hash,
        status=status,
        descendant_proposal_ids=descendant_proposal_ids,
        descendant_state_ids=descendant_state_ids,
        evidence_ids=evidence_ids,
        maximum_state_id=maximum_state_id,
        maximum_evidence_id=maximum_evidence_id,
        maximum_reward=maximum_reward,
        costs=cumulative_cost,
        unused_budget=unused_budget,
        eligible_for_scheduler=eligible_for_scheduler,
        infrastructure_aborted=infrastructure_aborted,
    )
    outcome = BranchOutcome(
        outcome_id=outcome_id,
        branch_id=branch.branch_id,
        branch_spec_hash=branch_spec_hash,
        status=status,
        descendant_proposal_ids=descendant_proposal_ids,
        descendant_state_ids=descendant_state_ids,
        evidence_ids=evidence_ids,
        maximum_state_id=maximum_state_id,
        maximum_evidence_id=maximum_evidence_id,
        maximum_reward=maximum_reward,
        costs=cumulative_cost,
        unused_budget=unused_budget,
        eligible_for_scheduler=eligible_for_scheduler,
        infrastructure_aborted=infrastructure_aborted,
    )

    policy_trace: Optional[PolicyTrace] = None
    if prompts and any(any(mask) for mask in masks):
        trace_payload = {
            "branch_id": branch.branch_id,
            "role_snapshot_id": role_snapshot.snapshot_id,
            "role": arm.role.value,
            "adapter_hash": role_snapshot.adapter_hash,
            "prompts": prompts,
            "response_segments": responses,
            "token_masks": [list(mask) for mask in masks],
            "log_probabilities": [list(values) for values in log_probabilities],
        }
        captured_token_ids = bool(token_ids) and all(
            len(ids) == len(mask) and len(ids) > 0
            for ids, mask in zip(token_ids, masks)
        )
        if captured_token_ids:
            trace_payload["token_ids"] = [list(values) for values in token_ids]
        policy_trace = PolicyTrace(
            trace_id=content_id("policy_trace", trace_payload),
            branch_id=branch.branch_id,
            role_snapshot_id=role_snapshot.snapshot_id,
            role=arm.role,
            adapter_hash=role_snapshot.adapter_hash,
            prompts=tuple(prompts),
            response_segments=tuple(responses),
            token_masks=tuple(masks),
            log_probabilities=tuple(log_probabilities),
            token_ids=(tuple(token_ids) if captured_token_ids else ()),
        )

    return BranchExecution(
        outcome=outcome,
        observations=tuple(observations),
        provenance_edges=tuple(provenance_edges),
        policy_trace=policy_trace,
    )


__all__ = [
    "BranchExecution",
    "BranchExecutionError",
    "BranchStepExecutor",
    "BranchStepRequest",
    "BranchStepResult",
    "PolicySegment",
    "build_option_context",
    "execute_branch",
]
