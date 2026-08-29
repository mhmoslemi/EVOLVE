"""Content-addressed harness specifications and matched audit contexts."""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any, Mapping, Sequence, Tuple

from evolve.ids import content_hash, content_id, validate_id
from evolve.types import AllocationArm, BranchSpec, Channel, FrozenDict, HarnessSpec


HARNESS_SUBSYSTEM_SCHEMA_VERSION = 1
HARNESS_SPEC_API_VERSION = "evolve_harness_spec_v1"
HARNESS_AUDIT_CONTEXT_VERSION = "matched_harness_audit_context_v1"
BASELINE_HARNESS_VERSION = "baseline_v1"


class HarnessValidationError(ValueError):
    """A harness artifact changes behavior without a valid immutable identity."""


def _nonempty(value: Any, name: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise HarnessValidationError(f"{name} must be a non-empty string")
    return value


def _nonnegative_int(value: Any, name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        raise HarnessValidationError(f"{name} must be a non-negative integer")
    return value


def _probability(value: Any, name: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise HarnessValidationError(f"{name} must be numeric")
    result = float(value)
    if not math.isfinite(result) or not 0.0 < result < 1.0:
        raise HarnessValidationError(f"{name} must lie strictly in (0, 1)")
    return result


def _require_id(value: Any, namespace: str, name: str) -> str:
    try:
        return validate_id(value, namespace)
    except (TypeError, ValueError) as exc:
        raise HarnessValidationError(f"invalid {name}: {exc}") from exc


def _thaw(value: Any) -> Any:
    if isinstance(value, Mapping):
        return {str(key): _thaw(item) for key, item in value.items()}
    if isinstance(value, tuple):
        return [_thaw(item) for item in value]
    if hasattr(value, "value"):
        return value.value
    return value


def create_harness_spec(
    *,
    version: str,
    instructions: str,
    tools: Sequence[str] = (),
    intermediate_tests: Sequence[str] = (),
    scaffolding: Mapping[str, Any] = FrozenDict(),
    diagnostic_feedback: Mapping[str, Any] = FrozenDict(),
    tool_policy_version: str,
) -> HarnessSpec:
    """Create a spec whose ID covers every behavior-bearing field."""

    spec = HarnessSpec.create(
        version=_nonempty(version, "harness version"),
        instructions=_nonempty(instructions, "harness instructions"),
        tools=tuple(tools),
        intermediate_tests=tuple(intermediate_tests),
        scaffolding=scaffolding,
        diagnostic_feedback=diagnostic_feedback,
        tool_policy_version=_nonempty(tool_policy_version, "tool_policy_version"),
    )
    validate_harness_spec(spec)
    return spec


def validate_harness_spec(spec: HarnessSpec) -> HarnessSpec:
    if not isinstance(spec, HarnessSpec):
        raise HarnessValidationError("expected a typed HarnessSpec")
    _nonempty(spec.version, "harness version")
    _nonempty(spec.instructions, "harness instructions")
    _nonempty(spec.tool_policy_version, "tool_policy_version")
    for name, values in (
        ("tools", spec.tools),
        ("intermediate_tests", spec.intermediate_tests),
    ):
        for index, value in enumerate(values):
            _nonempty(value, f"{name}[{index}]")
        if len(set(values)) != len(values):
            raise HarnessValidationError(f"{name} must not contain duplicates")
    expected_hash = content_hash(spec.identity_payload())
    expected_id = content_id("harness", spec.identity_payload())
    if spec.spec_hash != expected_hash or spec.harness_id != expected_id:
        raise HarnessValidationError(
            "harness ID/hash must cover every instruction, tool, test, scaffold, "
            "diagnostic, policy, and version field"
        )
    return spec


def baseline_harness_spec() -> HarnessSpec:
    """The immutable no-tool baseline used for common comparisons."""

    return create_harness_spec(
        version=BASELINE_HARNESS_VERSION,
        instructions=(
            "Execute the assigned role and option against the frozen verified start. "
            "Respect the branch horizon and hard budget. Harness-local diagnostics "
            "are guidance only; only the independent common verifier may admit a "
            "candidate or change the scientific record."
        ),
        tools=(),
        intermediate_tests=(),
        scaffolding={
            "prompt_sections": [
                "problem",
                "verified_start",
                "role",
                "option",
                "horizon_and_budget",
            ],
            "local_score_authority": False,
            "candidate_admission_authority": "common_verifier_only",
        },
        diagnostic_feedback={"enabled": False, "max_feedback_rounds": 0},
        tool_policy_version="no_harness_tools_v1",
    )


def _validate_branch_arm(branch: BranchSpec, arm: AllocationArm, label: str) -> None:
    if not isinstance(branch, BranchSpec) or not isinstance(arm, AllocationArm):
        raise HarnessValidationError(f"{label} requires typed BranchSpec and AllocationArm")
    if branch.arm_id != arm.arm_id:
        raise HarnessValidationError(f"{label} branch does not reference its allocation arm")
    if branch.harness_id != arm.harness_id:
        raise HarnessValidationError(f"{label} branch/arm harness references differ")
    if branch.option_id != arm.option_id:
        raise HarnessValidationError(f"{label} branch/arm option references differ")
    if branch.horizon != arm.horizon or branch.channel != arm.channel:
        raise HarnessValidationError(f"{label} branch/arm horizon or channel differs")
    if branch.channel != Channel.AUDIT:
        raise HarnessValidationError("matched harness comparisons require the audit channel")


def _matched_invariants(branch: BranchSpec, arm: AllocationArm) -> Mapping[str, Any]:
    return {
        "epoch": branch.epoch,
        "start_state_id": branch.start_state_id,
        "cell_id": arm.cell_id,
        "frozen_record_threshold": branch.frozen_record_threshold,
        "role": arm.role.value,
        "role_snapshot_id": branch.role_snapshot_id,
        "option_id": branch.option_id,
        "option_version": branch.option_version,
        "verifier_id": branch.verifier_id,
        "verifier_version": branch.verifier_version,
        "memory_view_id": branch.memory_view_id,
        "memory_view_hash": branch.memory_view_hash,
        "horizon": branch.horizon,
        "budget": _thaw(branch.budget),
        "cost_class": arm.cost_class,
        "expected_cost": _thaw(arm.expected_cost),
        "hard_cost": _thaw(arm.hard_cost),
        "seed": branch.seed,
        "generation_settings": _thaw(branch.generation_settings),
        "channel": branch.channel.value,
    }


@dataclass(frozen=True)
class MatchedHarnessAuditContext:
    """Preassigned pair that differs in harness identity and nothing scientific."""

    context_id: str
    incumbent_branch_id: str
    candidate_branch_id: str
    incumbent_harness_id: str
    incumbent_harness_version: str
    candidate_harness_id: str
    candidate_harness_version: str
    invariant_hash: str
    assignment_probability: float
    assignment_seed: int
    preassigned: bool = True
    context_version: str = HARNESS_AUDIT_CONTEXT_VERSION
    schema_version: int = HARNESS_SUBSYSTEM_SCHEMA_VERSION

    @classmethod
    def create(
        cls,
        *,
        incumbent_branch: BranchSpec,
        incumbent_arm: AllocationArm,
        candidate_branch: BranchSpec,
        candidate_arm: AllocationArm,
        assignment_probability: float,
        assignment_seed: int,
    ) -> "MatchedHarnessAuditContext":
        _validate_branch_arm(incumbent_branch, incumbent_arm, "incumbent")
        _validate_branch_arm(candidate_branch, candidate_arm, "candidate")
        if incumbent_branch.branch_id == candidate_branch.branch_id:
            raise HarnessValidationError("matched harness audit sides need distinct branches")
        if incumbent_branch.harness_id == candidate_branch.harness_id:
            raise HarnessValidationError("matched harness audit must compare two harness IDs")
        left = _matched_invariants(incumbent_branch, incumbent_arm)
        right = _matched_invariants(candidate_branch, candidate_arm)
        if left != right:
            differing = sorted(key for key in left if left[key] != right[key])
            raise HarnessValidationError(
                "matched harness audit differs outside harness version: "
                + ", ".join(differing)
            )
        probability = _probability(assignment_probability, "assignment_probability")
        seed = _nonnegative_int(assignment_seed, "assignment_seed")
        invariant_hash = content_hash(left)
        identity = {
            "context_version": HARNESS_AUDIT_CONTEXT_VERSION,
            "incumbent_branch_id": incumbent_branch.branch_id,
            "candidate_branch_id": candidate_branch.branch_id,
            "incumbent_harness_id": incumbent_branch.harness_id,
            "incumbent_harness_version": incumbent_branch.harness_version,
            "candidate_harness_id": candidate_branch.harness_id,
            "candidate_harness_version": candidate_branch.harness_version,
            "invariant_hash": invariant_hash,
            "assignment_probability": probability,
            "assignment_seed": seed,
            "preassigned": True,
        }
        return cls(context_id=content_id("harness_audit", identity), **identity)

    def __post_init__(self) -> None:
        _require_id(self.context_id, "harness_audit", "context_id")
        _require_id(self.incumbent_branch_id, "branch", "incumbent_branch_id")
        _require_id(self.candidate_branch_id, "branch", "candidate_branch_id")
        _require_id(self.incumbent_harness_id, "harness", "incumbent_harness_id")
        _require_id(self.candidate_harness_id, "harness", "candidate_harness_id")
        _nonempty(self.incumbent_harness_version, "incumbent_harness_version")
        _nonempty(self.candidate_harness_version, "candidate_harness_version")
        if self.incumbent_branch_id == self.candidate_branch_id:
            raise HarnessValidationError("matched harness audit branches must differ")
        if self.incumbent_harness_id == self.candidate_harness_id:
            raise HarnessValidationError("matched harness audit harnesses must differ")
        if not isinstance(self.invariant_hash, str) or len(self.invariant_hash) != 64:
            raise HarnessValidationError("invariant_hash must be SHA-256")
        _probability(self.assignment_probability, "assignment_probability")
        _nonnegative_int(self.assignment_seed, "assignment_seed")
        if self.preassigned is not True:
            raise HarnessValidationError("harness audits must be preassigned before execution")
        if self.schema_version != HARNESS_SUBSYSTEM_SCHEMA_VERSION:
            raise HarnessValidationError(
                f"unsupported harness audit schema {self.schema_version!r}"
            )
        if self.context_version != HARNESS_AUDIT_CONTEXT_VERSION:
            raise HarnessValidationError(
                f"unsupported harness audit context {self.context_version!r}"
            )
        identity = self.identity_payload()
        if self.context_id != content_id("harness_audit", identity):
            raise HarnessValidationError("harness audit context ID does not match content")

    def identity_payload(self) -> Mapping[str, Any]:
        return {
            "context_version": self.context_version,
            "incumbent_branch_id": self.incumbent_branch_id,
            "candidate_branch_id": self.candidate_branch_id,
            "incumbent_harness_id": self.incumbent_harness_id,
            "incumbent_harness_version": self.incumbent_harness_version,
            "candidate_harness_id": self.candidate_harness_id,
            "candidate_harness_version": self.candidate_harness_version,
            "invariant_hash": self.invariant_hash,
            "assignment_probability": self.assignment_probability,
            "assignment_seed": self.assignment_seed,
            "preassigned": self.preassigned,
        }

    def to_dict(self) -> Mapping[str, Any]:
        return {
            "record_type": "matched_harness_audit_context",
            "schema_version": self.schema_version,
            **self.identity_payload(),
            "context_id": self.context_id,
        }

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> "MatchedHarnessAuditContext":
        expected = {
            "record_type", "schema_version", "context_version", "context_id",
            "incumbent_branch_id", "candidate_branch_id", "incumbent_harness_id",
            "incumbent_harness_version", "candidate_harness_id",
            "candidate_harness_version", "invariant_hash", "assignment_probability",
            "assignment_seed", "preassigned",
        }
        if not isinstance(payload, Mapping) or set(payload) != expected:
            raise HarnessValidationError("invalid matched harness audit context fields")
        if payload["record_type"] != "matched_harness_audit_context":
            raise HarnessValidationError("invalid matched harness audit record_type")
        return cls(**{key: payload[key] for key in expected if key != "record_type"})


__all__ = [
    "BASELINE_HARNESS_VERSION",
    "HARNESS_AUDIT_CONTEXT_VERSION",
    "HARNESS_SPEC_API_VERSION",
    "HARNESS_SUBSYSTEM_SCHEMA_VERSION",
    "HarnessValidationError",
    "MatchedHarnessAuditContext",
    "baseline_harness_spec",
    "create_harness_spec",
    "validate_harness_spec",
]
