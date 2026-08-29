"""Executable, bounded option state machines for EVOLVE branches.

An :class:`OptionSpec` is a content-addressed declaration.  It is not itself
an executable option and must never be interpreted as a prompt label.  This
module binds a validated spec to versioned Python transition logic and exposes
small immutable records that a branch executor can persist around each step.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, ClassVar, Mapping, Optional, Sequence, Tuple

from evolve.ids import canonical_json, content_id, validate_id
from evolve.types import FrozenDict, OptionSpec, Role


OPTION_EXECUTION_SCHEMA_VERSION = 1
OPTION_SPEC_IDENTITY_VERSION = "option_spec_identity_v1"


class OptionError(ValueError):
    """Base error for invalid option specifications or execution state."""


class OptionEligibilityError(OptionError):
    """A frozen branch context cannot initiate a requested option."""


class OptionExecutionError(OptionError):
    """An option transition would violate its executable contract."""


class OptionDecisionKind(str, Enum):
    CONTINUE = "continue"
    STOP = "stop"


def _nonempty(value: Any, name: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise OptionError(f"{name} must be a non-empty string")
    return value


def _nonnegative_int(value: Any, name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        raise OptionError(f"{name} must be a non-negative integer")
    return value


def _positive_int(value: Any, name: str) -> int:
    _nonnegative_int(value, name)
    if value == 0:
        raise OptionError(f"{name} must be positive")
    return value


def _boolean(value: Any, name: str) -> bool:
    if not isinstance(value, bool):
        raise OptionError(f"{name} must be boolean")
    return value


def _identifier(value: Any, namespace: str, name: str) -> str:
    try:
        return validate_id(value, namespace)
    except (TypeError, ValueError) as exc:
        raise OptionError(f"invalid {name}: {exc}") from exc


def _optional_identifier(value: Optional[str], namespace: str, name: str) -> None:
    if value is not None:
        _identifier(value, namespace, name)


def _unique_strings(values: Sequence[str], name: str) -> Tuple[str, ...]:
    frozen = tuple(values)
    for value in frozen:
        _nonempty(value, name)
    if len(set(frozen)) != len(frozen):
        raise OptionError(f"{name} must not contain duplicates")
    return frozen


def _resource_map(value: Mapping[str, Any], name: str) -> FrozenDict:
    if not isinstance(value, Mapping):
        raise OptionError(f"{name} must be a resource mapping")
    frozen = value if isinstance(value, FrozenDict) else FrozenDict(value)
    for resource, amount in frozen.items():
        _nonempty(resource, f"{name} resource")
        if isinstance(amount, bool) or not isinstance(amount, (int, float)):
            raise OptionError(f"{name}.{resource} must be numeric")
        number = float(amount)
        if not math.isfinite(number) or number < 0.0:
            raise OptionError(f"{name}.{resource} must be finite and non-negative")
    return frozen


def _json_mapping(value: Mapping[str, Any], name: str) -> FrozenDict:
    if not isinstance(value, Mapping):
        raise OptionError(f"{name} must be a JSON mapping")
    try:
        frozen = value if isinstance(value, FrozenDict) else FrozenDict(value)
        canonical_json(frozen)
        return frozen
    except (TypeError, ValueError) as exc:
        raise OptionError(f"{name} must be JSON-safe: {exc}") from exc


def _thaw(value: Any) -> Any:
    if isinstance(value, Mapping):
        return {str(key): _thaw(item) for key, item in value.items()}
    if isinstance(value, tuple):
        return [_thaw(item) for item in value]
    if isinstance(value, Enum):
        return value.value
    return value


def option_spec_identity_payload(spec: OptionSpec) -> Mapping[str, Any]:
    """Return every field that can change an option's branch behaviour."""

    return {
        "identity_version": OPTION_SPEC_IDENTITY_VERSION,
        "version": spec.version,
        "state_machine": spec.state_machine,
        "allowed_roles": [role.value for role in spec.allowed_roles],
        "capabilities": list(spec.capabilities),
        "initiation": _thaw(spec.initiation),
        "step_policy": _thaw(spec.step_policy),
        "stop_rule": _thaw(spec.stop_rule),
        "max_horizon": spec.max_horizon,
        "expected_cost": _thaw(spec.expected_cost),
        "hard_cost": _thaw(spec.hard_cost),
        "harness_eligibility": list(spec.harness_eligibility),
        "prerequisites": list(spec.prerequisites),
        "output_contract": _thaw(spec.output_contract),
    }


def create_option_spec(
    *,
    version: str,
    state_machine: str,
    allowed_roles: Sequence[Role],
    capabilities: Sequence[str],
    initiation: Mapping[str, Any],
    step_policy: Mapping[str, Any],
    stop_rule: Mapping[str, Any],
    max_horizon: int,
    expected_cost: Mapping[str, Any],
    hard_cost: Mapping[str, Any],
    harness_eligibility: Sequence[str],
    prerequisites: Sequence[str],
    output_contract: Mapping[str, Any],
) -> OptionSpec:
    """Create an immutable OptionSpec whose ID covers all executable behaviour."""

    normalized_roles = tuple(
        role if isinstance(role, Role) else Role(role) for role in allowed_roles
    )
    placeholder = OptionSpec(
        option_id=content_id("option", {"placeholder": state_machine}),
        version=version,
        state_machine=state_machine,
        allowed_roles=normalized_roles,
        capabilities=tuple(capabilities),
        initiation=initiation,
        step_policy=step_policy,
        stop_rule=stop_rule,
        max_horizon=max_horizon,
        expected_cost=expected_cost,
        hard_cost=hard_cost,
        harness_eligibility=tuple(harness_eligibility),
        prerequisites=tuple(prerequisites),
        output_contract=output_contract,
    )
    return OptionSpec(
        option_id=content_id("option", option_spec_identity_payload(placeholder)),
        version=placeholder.version,
        state_machine=placeholder.state_machine,
        allowed_roles=placeholder.allowed_roles,
        capabilities=placeholder.capabilities,
        initiation=placeholder.initiation,
        step_policy=placeholder.step_policy,
        stop_rule=placeholder.stop_rule,
        max_horizon=placeholder.max_horizon,
        expected_cost=placeholder.expected_cost,
        hard_cost=placeholder.hard_cost,
        harness_eligibility=placeholder.harness_eligibility,
        prerequisites=placeholder.prerequisites,
        output_contract=placeholder.output_contract,
    )


def validate_option_spec_identity(spec: OptionSpec) -> None:
    if not isinstance(spec, OptionSpec):
        raise OptionError("registered option spec must be OptionSpec")
    expected = content_id("option", option_spec_identity_payload(spec))
    if spec.option_id != expected:
        raise OptionError(
            "option_id must be content-addressed from every behaviour field"
        )


@dataclass(frozen=True)
class OptionContext:
    """Frozen structural context checked before an option can start."""

    role: Role
    cell_id: str
    harness_id: str
    requested_horizon: int
    available_capabilities: Tuple[str, ...]
    satisfied_prerequisites: Tuple[str, ...]
    budget: FrozenDict
    start_state_id: Optional[str] = None
    start_verified: bool = False
    cell_empty: bool = False
    diagnostics_available: bool = False
    matched_control: bool = False
    memory_enabled: bool = True
    metadata: FrozenDict = field(default_factory=FrozenDict)

    def __post_init__(self) -> None:
        try:
            owner = self.role if isinstance(self.role, Role) else Role(self.role)
        except (TypeError, ValueError) as exc:
            raise OptionError(f"unknown role {self.role!r}") from exc
        object.__setattr__(self, "role", owner)
        _identifier(self.cell_id, "cell", "cell_id")
        _identifier(self.harness_id, "harness", "harness_id")
        _optional_identifier(self.start_state_id, "state", "start_state_id")
        _positive_int(self.requested_horizon, "requested_horizon")
        object.__setattr__(
            self,
            "available_capabilities",
            _unique_strings(self.available_capabilities, "available_capabilities"),
        )
        object.__setattr__(
            self,
            "satisfied_prerequisites",
            _unique_strings(self.satisfied_prerequisites, "satisfied_prerequisites"),
        )
        object.__setattr__(self, "budget", _resource_map(self.budget, "budget"))
        object.__setattr__(self, "metadata", _json_mapping(self.metadata, "metadata"))
        for field_name in (
            "start_verified",
            "cell_empty",
            "diagnostics_available",
            "matched_control",
            "memory_enabled",
        ):
            _boolean(getattr(self, field_name), field_name)
        if self.start_verified and self.start_state_id is None:
            raise OptionError("start_verified requires a saved start_state_id")

    def to_dict(self) -> Mapping[str, Any]:
        return {
            "role": self.role.value,
            "cell_id": self.cell_id,
            "harness_id": self.harness_id,
            "requested_horizon": self.requested_horizon,
            "available_capabilities": list(self.available_capabilities),
            "satisfied_prerequisites": list(self.satisfied_prerequisites),
            "budget": _thaw(self.budget),
            "start_state_id": self.start_state_id,
            "start_verified": self.start_verified,
            "cell_empty": self.cell_empty,
            "diagnostics_available": self.diagnostics_available,
            "matched_control": self.matched_control,
            "memory_enabled": self.memory_enabled,
            "metadata": _thaw(self.metadata),
        }


@dataclass(frozen=True)
class OptionStepInput:
    """Verifier-backed observation supplied to one state-machine transition."""

    step_index: int
    cumulative_cost: FrozenDict
    latest_state_id: Optional[str] = None
    latest_evidence_id: Optional[str] = None
    admitted: bool = False
    confirmed: bool = False
    record_improved: bool = False
    diagnostics_available: bool = False
    infrastructure_failed: bool = False
    failure_kind: str = "none"
    novelty: float = 0.0
    uncertainty: float = 0.0
    metadata: FrozenDict = field(default_factory=FrozenDict)

    def __post_init__(self) -> None:
        _nonnegative_int(self.step_index, "step_index")
        _optional_identifier(self.latest_state_id, "state", "latest_state_id")
        _optional_identifier(self.latest_evidence_id, "evidence", "latest_evidence_id")
        object.__setattr__(
            self,
            "cumulative_cost",
            _resource_map(self.cumulative_cost, "cumulative_cost"),
        )
        object.__setattr__(self, "metadata", _json_mapping(self.metadata, "metadata"))
        for field_name in (
            "admitted",
            "confirmed",
            "record_improved",
            "diagnostics_available",
            "infrastructure_failed",
        ):
            _boolean(getattr(self, field_name), field_name)
        for field_name in ("novelty", "uncertainty"):
            value = getattr(self, field_name)
            if isinstance(value, bool) or not isinstance(value, (int, float)):
                raise OptionError(f"{field_name} must be numeric")
            if not math.isfinite(float(value)) or float(value) < 0.0:
                raise OptionError(f"{field_name} must be finite and non-negative")
        _nonempty(self.failure_kind, "failure_kind")
        if self.infrastructure_failed and self.failure_kind != "infrastructure":
            raise OptionError(
                "infrastructure_failed requires failure_kind='infrastructure'"
            )
        if self.admitted and (
            self.latest_state_id is None or self.latest_evidence_id is None
        ):
            raise OptionError(
                "an admitted step must reference its scientific state and evidence"
            )
        if self.confirmed and not self.admitted:
            raise OptionError("confirmed observations must be admitted")


@dataclass(frozen=True)
class OptionState:
    """Content-addressed state of one executable option instance."""

    state_id: str
    option_id: str
    option_version: str
    context: OptionContext
    step_index: int
    phase: str
    terminal: bool
    history: Tuple[str, ...]
    data: FrozenDict
    schema_version: int = OPTION_EXECUTION_SCHEMA_VERSION

    @classmethod
    def create(
        cls,
        *,
        option_id: str,
        option_version: str,
        context: OptionContext,
        step_index: int,
        phase: str,
        terminal: bool,
        history: Sequence[str],
        data: Mapping[str, Any],
    ) -> "OptionState":
        payload = {
            "schema_version": OPTION_EXECUTION_SCHEMA_VERSION,
            "option_id": option_id,
            "option_version": option_version,
            "context": context.to_dict(),
            "step_index": step_index,
            "phase": phase,
            "terminal": terminal,
            "history": list(history),
            "data": _thaw(data),
        }
        return cls(
            state_id=content_id("option_state", payload),
            option_id=option_id,
            option_version=option_version,
            context=context,
            step_index=step_index,
            phase=phase,
            terminal=terminal,
            history=tuple(history),
            data=FrozenDict(data),
        )

    def __post_init__(self) -> None:
        if self.schema_version != OPTION_EXECUTION_SCHEMA_VERSION:
            raise OptionError(
                f"unsupported option state schema {self.schema_version}"
            )
        _identifier(self.state_id, "option_state", "state_id")
        _identifier(self.option_id, "option", "option_id")
        _nonempty(self.option_version, "option_version")
        if not isinstance(self.context, OptionContext):
            raise OptionError("option state context must be OptionContext")
        _nonnegative_int(self.step_index, "step_index")
        _nonempty(self.phase, "phase")
        _boolean(self.terminal, "terminal")
        history = _unique_strings(self.history, "history actions")
        object.__setattr__(self, "history", history)
        object.__setattr__(self, "data", _json_mapping(self.data, "state data"))
        payload = {
            "schema_version": self.schema_version,
            "option_id": self.option_id,
            "option_version": self.option_version,
            "context": self.context.to_dict(),
            "step_index": self.step_index,
            "phase": self.phase,
            "terminal": self.terminal,
            "history": list(self.history),
            "data": _thaw(self.data),
        }
        if self.state_id != content_id("option_state", payload):
            raise OptionError("option state ID does not match its transition content")


@dataclass(frozen=True)
class OptionStepDecision:
    """Persistable continue/stop decision returned by an executable option."""

    decision_id: str
    kind: OptionDecisionKind
    option_id: str
    option_version: str
    step_index: int
    action: str
    capability: Optional[str]
    prompt_metadata: FrozenDict
    action_metadata: FrozenDict
    stop_reason: Optional[str]
    next_state: OptionState

    @classmethod
    def create(
        cls,
        *,
        kind: OptionDecisionKind,
        option_id: str,
        option_version: str,
        step_index: int,
        action: str,
        capability: Optional[str],
        prompt_metadata: Mapping[str, Any],
        action_metadata: Mapping[str, Any],
        stop_reason: Optional[str],
        next_state: OptionState,
    ) -> "OptionStepDecision":
        normalized = kind if isinstance(kind, OptionDecisionKind) else OptionDecisionKind(kind)
        payload = {
            "kind": normalized.value,
            "option_id": option_id,
            "option_version": option_version,
            "step_index": step_index,
            "action": action,
            "capability": capability,
            "prompt_metadata": _thaw(prompt_metadata),
            "action_metadata": _thaw(action_metadata),
            "stop_reason": stop_reason,
            "next_state_id": next_state.state_id,
        }
        return cls(
            decision_id=content_id("option_decision", payload),
            kind=normalized,
            option_id=option_id,
            option_version=option_version,
            step_index=step_index,
            action=action,
            capability=capability,
            prompt_metadata=FrozenDict(prompt_metadata),
            action_metadata=FrozenDict(action_metadata),
            stop_reason=stop_reason,
            next_state=next_state,
        )

    def __post_init__(self) -> None:
        try:
            kind = self.kind if isinstance(self.kind, OptionDecisionKind) else OptionDecisionKind(self.kind)
        except (TypeError, ValueError) as exc:
            raise OptionError(f"invalid option decision kind {self.kind!r}") from exc
        object.__setattr__(self, "kind", kind)
        _identifier(self.decision_id, "option_decision", "decision_id")
        _identifier(self.option_id, "option", "option_id")
        _nonempty(self.option_version, "option_version")
        _nonnegative_int(self.step_index, "step_index")
        _nonempty(self.action, "action")
        if self.capability is not None:
            _nonempty(self.capability, "capability")
        object.__setattr__(
            self,
            "prompt_metadata",
            _json_mapping(self.prompt_metadata, "prompt_metadata"),
        )
        object.__setattr__(
            self,
            "action_metadata",
            _json_mapping(self.action_metadata, "action_metadata"),
        )
        if kind == OptionDecisionKind.STOP:
            if self.capability is not None or self.stop_reason is None:
                raise OptionError(
                    "stop decisions require a reason and cannot request a capability"
                )
            if not self.next_state.terminal:
                raise OptionError("stop decisions require a terminal next state")
        else:
            if self.capability is None or self.stop_reason is not None:
                raise OptionError(
                    "continue decisions require a capability and no stop reason"
                )
            if self.next_state.terminal:
                raise OptionError("continue decisions cannot return terminal state")
        payload = {
            "kind": self.kind.value,
            "option_id": self.option_id,
            "option_version": self.option_version,
            "step_index": self.step_index,
            "action": self.action,
            "capability": self.capability,
            "prompt_metadata": _thaw(self.prompt_metadata),
            "action_metadata": _thaw(self.action_metadata),
            "stop_reason": self.stop_reason,
            "next_state_id": self.next_state.state_id,
        }
        if self.decision_id != content_id("option_decision", payload):
            raise OptionError("option decision ID does not match its content")


@dataclass(frozen=True)
class _ActionPlan:
    action: str
    capability: str
    next_phase: str
    prompt_metadata: FrozenDict = field(default_factory=FrozenDict)
    action_metadata: FrozenDict = field(default_factory=FrozenDict)
    next_data: FrozenDict = field(default_factory=FrozenDict)


class ExecutableOption:
    """Base class for one registered, versioned option state machine."""

    STATE_MACHINE: ClassVar[str]
    BEHAVIOR_VERSION: ClassVar[str]
    INITIAL_PHASE: ClassVar[str] = "initial"

    def __init__(self, spec: OptionSpec) -> None:
        validate_option_spec_identity(spec)
        if spec.state_machine != self.STATE_MACHINE:
            raise OptionError(
                f"implementation {self.STATE_MACHINE!r} cannot execute "
                f"state machine {spec.state_machine!r}"
            )
        if spec.version != self.BEHAVIOR_VERSION:
            raise OptionError(
                f"implementation version {self.BEHAVIOR_VERSION!r} cannot execute "
                f"spec version {spec.version!r}"
            )
        self._spec = spec

    @property
    def spec(self) -> OptionSpec:
        return self._spec

    def check_initiation(self, context: OptionContext) -> None:
        if not isinstance(context, OptionContext):
            raise OptionEligibilityError("option context must be OptionContext")
        if context.role not in self.spec.allowed_roles:
            raise OptionEligibilityError(
                f"role {context.role.value!r} cannot execute {self.spec.state_machine}"
            )
        if context.harness_id not in self.spec.harness_eligibility:
            raise OptionEligibilityError(
                "assigned harness is not eligible for this option version"
            )
        if context.requested_horizon > self.spec.max_horizon:
            raise OptionEligibilityError(
                "requested horizon exceeds the option's hard maximum"
            )
        missing_capabilities = sorted(
            set(self.spec.capabilities) - set(context.available_capabilities)
        )
        if missing_capabilities:
            raise OptionEligibilityError(
                "execution environment lacks option capabilities: "
                + ", ".join(missing_capabilities)
            )
        missing_prerequisites = sorted(
            set(self.spec.prerequisites) - set(context.satisfied_prerequisites)
        )
        if missing_prerequisites:
            raise OptionEligibilityError(
                "option prerequisites are not satisfied: "
                + ", ".join(missing_prerequisites)
            )
        for resource, hard_limit in self.spec.hard_cost.items():
            if resource not in context.budget:
                raise OptionEligibilityError(
                    f"branch budget omits hard-bounded resource {resource!r}"
                )
            if float(context.budget[resource]) < float(hard_limit):
                raise OptionEligibilityError(
                    f"branch budget for {resource!r} is below the option hard cost"
                )
        self._check_initiation(context)

    def _check_initiation(self, context: OptionContext) -> None:
        """State-machine-specific initiation guard."""

    def start(self, context: OptionContext) -> OptionState:
        self.check_initiation(context)
        return OptionState.create(
            option_id=self.spec.option_id,
            option_version=self.spec.version,
            context=context,
            step_index=0,
            phase=self.INITIAL_PHASE,
            terminal=False,
            history=(),
            data=self._initial_data(context),
        )

    def _initial_data(self, context: OptionContext) -> Mapping[str, Any]:
        return {}

    def _validate_state_input(
        self, state: OptionState, step_input: OptionStepInput
    ) -> None:
        if not isinstance(state, OptionState):
            raise OptionExecutionError("state must be OptionState")
        if not isinstance(step_input, OptionStepInput):
            raise OptionExecutionError("step input must be OptionStepInput")
        if state.option_id != self.spec.option_id or state.option_version != self.spec.version:
            raise OptionExecutionError("option state belongs to another implementation")
        if state.terminal:
            raise OptionExecutionError("a terminal option state cannot advance")
        if step_input.step_index != state.step_index:
            raise OptionExecutionError(
                "step input index does not match the frozen option state"
            )
        for resource, amount in step_input.cumulative_cost.items():
            if resource in self.spec.hard_cost and float(amount) > float(
                self.spec.hard_cost[resource]
            ):
                raise OptionExecutionError(
                    f"option exceeded its hard {resource!r} cost bound"
                )

    def stop(
        self, state: OptionState, step_input: OptionStepInput
    ) -> Optional[str]:
        """Return a deterministic stop reason, or ``None`` to execute a step."""

        self._validate_state_input(state, step_input)
        if step_input.infrastructure_failed:
            return "infrastructure_failure"
        if step_input.confirmed and bool(self.spec.stop_rule.get("on_confirmed", True)):
            return "confirmed_candidate"
        if step_input.record_improved and bool(
            self.spec.stop_rule.get("on_record_improvement", True)
        ):
            return "record_improvement"
        if state.step_index >= state.context.requested_horizon:
            return "horizon_exhausted"
        for resource, hard_limit in self.spec.hard_cost.items():
            if float(hard_limit) > 0.0 and float(
                step_input.cumulative_cost.get(resource, 0.0)
            ) >= float(hard_limit):
                return f"hard_cost_exhausted:{resource}"
        return self._stop(state, step_input)

    def _stop(
        self, state: OptionState, step_input: OptionStepInput
    ) -> Optional[str]:
        return None

    def step(
        self, state: OptionState, step_input: OptionStepInput
    ) -> OptionStepDecision:
        """Execute one pure transition or emit a terminal stop decision."""

        reason = self.stop(state, step_input)
        if reason is not None:
            terminal = OptionState.create(
                option_id=state.option_id,
                option_version=state.option_version,
                context=state.context,
                step_index=state.step_index,
                phase="stopped",
                terminal=True,
                history=state.history,
                data=state.data,
            )
            return OptionStepDecision.create(
                kind=OptionDecisionKind.STOP,
                option_id=self.spec.option_id,
                option_version=self.spec.version,
                step_index=state.step_index,
                action="stop",
                capability=None,
                prompt_metadata={},
                action_metadata={"last_phase": state.phase},
                stop_reason=reason,
                next_state=terminal,
            )
        plan = self._plan_action(state, step_input)
        if plan.capability not in self.spec.capabilities:
            raise OptionExecutionError(
                f"state machine requested undeclared capability {plan.capability!r}"
            )
        if plan.capability not in state.context.available_capabilities:
            raise OptionExecutionError(
                f"branch context cannot execute capability {plan.capability!r}"
            )
        if plan.action in state.history:
            raise OptionExecutionError(
                "an option action identity cannot repeat within one branch"
            )
        next_state = OptionState.create(
            option_id=state.option_id,
            option_version=state.option_version,
            context=state.context,
            step_index=state.step_index + 1,
            phase=plan.next_phase,
            terminal=False,
            history=state.history + (plan.action,),
            data=plan.next_data,
        )
        return OptionStepDecision.create(
            kind=OptionDecisionKind.CONTINUE,
            option_id=self.spec.option_id,
            option_version=self.spec.version,
            step_index=state.step_index,
            action=plan.action,
            capability=plan.capability,
            prompt_metadata=plan.prompt_metadata,
            action_metadata=plan.action_metadata,
            stop_reason=None,
            next_state=next_state,
        )

    def _plan_action(
        self, state: OptionState, step_input: OptionStepInput
    ) -> _ActionPlan:
        raise NotImplementedError


__all__ = [
    "ExecutableOption",
    "OPTION_EXECUTION_SCHEMA_VERSION",
    "OPTION_SPEC_IDENTITY_VERSION",
    "OptionContext",
    "OptionDecisionKind",
    "OptionEligibilityError",
    "OptionError",
    "OptionExecutionError",
    "OptionState",
    "OptionStepDecision",
    "OptionStepInput",
    "create_option_spec",
    "option_spec_identity_payload",
    "validate_option_spec_identity",
]
