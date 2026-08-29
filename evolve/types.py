"""Schema-v1 immutable records shared across the EVOLVE framework.

The records in this module are deliberately persistence-oriented: every field
round-trips through JSON, reads reject unsupported future schemas, and unknown
fields survive in ``extensions``.  Cross-object services perform deeper store
lookups, while each record enforces every invariant decidable from its own
payload.
"""

from __future__ import annotations

import json
import math
import re
from dataclasses import dataclass, field, fields
from enum import Enum
from types import MappingProxyType
from typing import Any, ClassVar, Mapping, Optional, Tuple, Type, TypeVar

from evolve.ids import canonical_json, content_hash, content_id, validate_id


SCHEMA_VERSION = 1
_SHA256_RE = re.compile(r"^[0-9a-f]{64}$")


class SchemaValidationError(ValueError):
    """A persisted record is malformed or violates a scientific invariant."""


class UnsupportedSchemaVersion(SchemaValidationError):
    """A reader was asked to interpret a schema version it does not support."""


class InvariantViolation(SchemaValidationError):
    """A well-shaped record contains a contradictory state."""


class FrozenDict(Mapping[str, Any]):
    """An actually immutable, recursively frozen JSON mapping.

    This is intentionally *not* a ``dict`` subclass: ``dict.__setitem__`` can
    bypass overridden mutation methods on subclasses.  Mapping consumers still
    get the normal read-only interface, while persistence goes through
    ``SchemaRecord.to_dict``.
    """

    __slots__ = ("_data",)

    def __init__(self, value: Optional[Mapping[str, Any]] = None) -> None:
        source = {} if value is None else dict(value)
        data = {}
        for key, item in source.items():
            if not isinstance(key, str):
                raise SchemaValidationError("FrozenDict keys must be strings")
            data[key] = _freeze_json(item, f"FrozenDict.{key}")
        object.__setattr__(self, "_data", data)

    def __getitem__(self, key: str) -> Any:
        return self._data[key]

    def __iter__(self):
        return iter(self._data)

    def __len__(self) -> int:
        return len(self._data)

    def __repr__(self) -> str:
        return f"FrozenDict({self._data!r})"

    def __eq__(self, other: Any) -> bool:
        if isinstance(other, Mapping):
            return dict(self.items()) == dict(other.items())
        return False

    def __copy__(self) -> "FrozenDict":
        return self

    def __deepcopy__(self, memo: dict) -> "FrozenDict":
        memo[id(self)] = self
        return self

    def __reduce__(self):
        return (FrozenDict, (dict(self.items()),))

    __hash__ = None


JSONValue = Any


class StringEnum(str, Enum):
    def __str__(self) -> str:
        return self.value


class Role(StringEnum):
    SCOUT = "scout"
    MECHANIST = "mechanist"
    CHALLENGER = "challenger"


class Channel(StringEnum):
    PRODUCTION = "production"
    AUDIT = "audit"
    REFINEMENT = "refinement"


class FailureKind(StringEnum):
    NONE = "none"
    PARSE = "parse"
    CODE = "code"
    CONSTRAINT = "constraint"
    SCIENTIFIC = "scientific"
    TIMEOUT = "timeout"
    INFRASTRUCTURE = "infrastructure"


class BranchStatus(StringEnum):
    CLOSED = "closed"
    ABORTED = "aborted"


class AuditStatus(StringEnum):
    PREASSIGNED = "preassigned"
    RUNNING = "running"
    CLOSED = "closed"
    ABORTED = "aborted"


class AuditSide(StringEnum):
    INTERVENTION = "intervention"
    CONTROL = "control"


class MemoryStatus(StringEnum):
    QUARANTINED = "quarantined"
    PROMOTED = "promoted"
    REJECTED = "rejected"


class LearningObjective(StringEnum):
    ORDERGRAD = "ordergrad"
    MAXPO = "maxpo"


class BudgetTransactionKind(StringEnum):
    DEBIT = "debit"
    REFUND = "refund"


def _freeze_json(value: Any, path: str = "value") -> Any:
    if isinstance(value, SchemaRecord):
        return value
    if isinstance(value, Enum):
        return value
    if value is None or isinstance(value, (str, bool, int)):
        return value
    if isinstance(value, float):
        if not math.isfinite(value):
            raise SchemaValidationError(f"{path} must not contain NaN or infinity")
        return 0.0 if value == 0.0 else value
    if isinstance(value, Mapping):
        out = {}
        for key, item in value.items():
            if not isinstance(key, str):
                raise SchemaValidationError(f"{path} mapping keys must be strings")
            out[key] = _freeze_json(item, f"{path}.{key}")
        return FrozenDict(out)
    if isinstance(value, (list, tuple)):
        return tuple(_freeze_json(item, f"{path}[]") for item in value)
    raise SchemaValidationError(
        f"{path} contains non-JSON value of type {type(value).__name__}"
    )


def _thaw_json(value: Any) -> Any:
    if isinstance(value, SchemaRecord):
        return value.to_dict()
    if isinstance(value, Enum):
        return value.value
    if isinstance(value, Mapping):
        return {str(key): _thaw_json(item) for key, item in value.items()}
    if isinstance(value, tuple):
        return [_thaw_json(item) for item in value]
    return value


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise InvariantViolation(message)


def _nonempty(value: Any, field_name: str) -> str:
    _require(isinstance(value, str) and bool(value.strip()), f"{field_name} must be non-empty")
    return value


def _id(value: str, namespace: str, field_name: str) -> None:
    try:
        validate_id(value, namespace)
    except (TypeError, ValueError) as exc:
        raise InvariantViolation(f"invalid {field_name}: {exc}") from exc


def _optional_id(value: Optional[str], namespace: str, field_name: str) -> None:
    if value is not None:
        _id(value, namespace, field_name)


def _any_optional_id(value: Optional[str], field_name: str) -> None:
    if value is not None:
        try:
            validate_id(value)
        except (TypeError, ValueError) as exc:
            raise InvariantViolation(f"invalid {field_name}: {exc}") from exc


def _finite(value: Any, field_name: str, *, minimum: Optional[float] = None) -> float:
    _require(
        isinstance(value, (int, float)) and not isinstance(value, bool),
        f"{field_name} must be numeric",
    )
    number = float(value)
    _require(math.isfinite(number), f"{field_name} must be finite")
    if minimum is not None:
        _require(number >= minimum, f"{field_name} must be >= {minimum}")
    return number


def _nonnegative_int(value: Any, field_name: str) -> int:
    _require(isinstance(value, int) and not isinstance(value, bool), f"{field_name} must be an integer")
    _require(value >= 0, f"{field_name} must be non-negative")
    return value


def _positive_int(value: Any, field_name: str) -> int:
    _nonnegative_int(value, field_name)
    _require(value > 0, f"{field_name} must be positive")
    return value


def _enum_field(record: Any, field_name: str, enum_type: Type[StringEnum]) -> None:
    value = getattr(record, field_name)
    try:
        converted = value if isinstance(value, enum_type) else enum_type(value)
    except (TypeError, ValueError) as exc:
        allowed = ", ".join(item.value for item in enum_type)
        raise InvariantViolation(
            f"{field_name} must be one of: {allowed}; got {value!r}"
        ) from exc
    object.__setattr__(record, field_name, converted)


def _unique(values: Tuple[Any, ...], field_name: str) -> None:
    _require(len(set(values)) == len(values), f"{field_name} must not contain duplicates")


def _numeric_resource_map(value: Mapping[str, Any], field_name: str) -> None:
    for resource, amount in value.items():
        _nonempty(resource, f"{field_name} resource")
        _finite(amount, f"{field_name}.{resource}", minimum=0.0)


def _sha256(value: str, field_name: str, *, optional: bool = False) -> None:
    if optional and value == "":
        return
    _require(isinstance(value, str) and _SHA256_RE.fullmatch(value) is not None,
             f"{field_name} must be a lowercase SHA-256 hex digest")


RecordT = TypeVar("RecordT", bound="SchemaRecord")


@dataclass(frozen=True)
class SchemaRecord:
    """Base persistence protocol for every typed EVOLVE record."""

    # ``init=False`` keeps Python 3.8/3.9 dataclass inheritance compatible with
    # required subclass fields.  ``from_dict`` installs validated persisted
    # values after construction; direct construction always creates schema v1.
    schema_version: int = field(default=SCHEMA_VERSION, init=False)
    extensions: FrozenDict = field(default_factory=FrozenDict, init=False, repr=False)

    RECORD_TYPE: ClassVar[str] = "record"

    def __post_init__(self) -> None:
        if self.schema_version > SCHEMA_VERSION:
            raise UnsupportedSchemaVersion(
                f"{self.RECORD_TYPE} schema {self.schema_version} is newer than supported schema {SCHEMA_VERSION}"
            )
        if self.schema_version != SCHEMA_VERSION:
            raise SchemaValidationError(
                f"unsupported {self.RECORD_TYPE} schema version {self.schema_version}"
            )
        # Deep-freeze every JSON container, not just ``extensions``.  Frozen
        # dataclasses alone do not stop mutation through a list or dict field.
        for item in fields(self):
            value = getattr(self, item.name)
            if isinstance(value, (Mapping, list, tuple)):
                object.__setattr__(self, item.name, _freeze_json(value, item.name))
        canonical_json(self.to_dict())

    def to_dict(self) -> dict:
        payload = {"record_type": self.RECORD_TYPE}
        for item in fields(self):
            payload[item.name] = _thaw_json(getattr(self, item.name))
        return payload

    def to_json(self, *, indent: Optional[int] = None) -> str:
        if indent is None:
            return canonical_json(self.to_dict())
        return json.dumps(
            self.to_dict(), ensure_ascii=False, allow_nan=False,
            sort_keys=True, indent=indent,
        )

    @classmethod
    def from_dict(cls: Type[RecordT], payload: Mapping[str, Any]) -> RecordT:
        if not isinstance(payload, Mapping):
            raise SchemaValidationError(f"{cls.RECORD_TYPE} payload must be a mapping")
        data = dict(payload)
        if "record_type" not in data:
            raise SchemaValidationError("persisted record is missing record_type")
        record_type = data.pop("record_type")
        if record_type != cls.RECORD_TYPE:
            raise SchemaValidationError(
                f"expected record_type={cls.RECORD_TYPE!r}, got {record_type!r}"
            )
        if "schema_version" not in data:
            raise SchemaValidationError("persisted record is missing schema_version")
        version = data.get("schema_version")
        if not isinstance(version, int) or isinstance(version, bool):
            raise SchemaValidationError("schema_version must be an integer")
        if version > SCHEMA_VERSION:
            raise UnsupportedSchemaVersion(
                f"{cls.RECORD_TYPE} schema {version} is newer than supported schema {SCHEMA_VERSION}"
            )
        if version != SCHEMA_VERSION:
            raise SchemaValidationError(
                f"unsupported {cls.RECORD_TYPE} schema version {version}"
            )
        known = {item.name for item in fields(cls)}
        extension_payload = data.pop("extensions", {})
        if not isinstance(extension_payload, Mapping):
            raise SchemaValidationError("extensions must be a mapping")
        extensions = dict(extension_payload)
        conflicting_extension_keys = sorted(
            key for key in extensions if key in known and key != "extensions"
        )
        if conflicting_extension_keys:
            raise SchemaValidationError(
                "extensions cannot shadow current fields: "
                + ", ".join(conflicting_extension_keys)
            )
        for key in tuple(data):
            if key not in known:
                value = data.pop(key)
                if key in extensions and extensions[key] != value:
                    raise SchemaValidationError(
                        f"unknown field {key!r} conflicts with extensions entry"
                    )
                extensions[key] = value
        data.pop("schema_version", None)
        try:
            record = cls(**data)
            object.__setattr__(record, "schema_version", version)
            object.__setattr__(record, "extensions", _freeze_json(extensions, "extensions"))
            canonical_json(record.to_dict())
            return record
        except SchemaValidationError:
            raise
        except (TypeError, ValueError) as exc:
            raise SchemaValidationError(
                f"invalid {cls.RECORD_TYPE} payload: {exc}"
            ) from exc

    @classmethod
    def from_json(cls: Type[RecordT], text: str) -> RecordT:
        try:
            payload = json.loads(text)
        except json.JSONDecodeError as exc:
            raise SchemaValidationError(f"invalid JSON: {exc}") from exc
        return cls.from_dict(payload)


@dataclass(frozen=True)
class Proposal(SchemaRecord):
    RECORD_TYPE: ClassVar[str] = "proposal"

    proposal_id: str
    run_id: str
    problem_id: str
    source_text: str
    source_hash: str
    parent_state_id: Optional[str] = None
    branch_id: Optional[str] = None
    parsed_candidate: JSONValue = None
    created_at: str = ""

    def __post_init__(self) -> None:
        super().__post_init__()
        _id(self.proposal_id, "proposal", "proposal_id")
        _id(self.run_id, "run", "run_id")
        _nonempty(self.problem_id, "problem_id")
        _sha256(self.source_hash, "source_hash")
        _require(self.source_hash == content_hash(self.source_text),
                 "source_hash must match the captured source_text")
        _optional_id(self.parent_state_id, "state", "parent_state_id")
        _optional_id(self.branch_id, "branch", "branch_id")


@dataclass(frozen=True)
class VerifiedScientificState(SchemaRecord):
    RECORD_TYPE: ClassVar[str] = "verified_scientific_state"

    state_id: str
    proposal_id: str
    evidence_id: str
    problem_id: str
    answer_payload: JSONValue
    resolved: bool
    admitted: bool
    confirmed: bool
    internal_reward: Optional[float]
    raw_score: JSONValue = None
    descriptor_id: Optional[str] = None
    fingerprint: str = ""

    def __post_init__(self) -> None:
        super().__post_init__()
        _id(self.state_id, "state", "state_id")
        _id(self.proposal_id, "proposal", "proposal_id")
        _id(self.evidence_id, "evidence", "evidence_id")
        _nonempty(self.problem_id, "problem_id")
        _require(isinstance(self.resolved, bool), "resolved must be boolean")
        _require(isinstance(self.admitted, bool), "admitted must be boolean")
        _require(isinstance(self.confirmed, bool), "confirmed must be boolean")
        if self.internal_reward is not None:
            _finite(self.internal_reward, "internal_reward")
        if self.admitted:
            _require(self.resolved, "an admitted state must be resolved")
            _require(self.internal_reward is not None,
                     "an admitted state needs a higher-is-better internal_reward")
            _require(self.answer_payload is not None,
                     "an admitted state must capture its answer_payload")
        if self.confirmed:
            _require(self.resolved and self.admitted,
                     "confirmed implies resolved and admitted")
        _optional_id(self.descriptor_id, "descriptor", "descriptor_id")


@dataclass(frozen=True)
class EvidencePacket(SchemaRecord):
    RECORD_TYPE: ClassVar[str] = "evidence_packet"

    evidence_id: str
    run_id: str
    proposal_id: str
    scientific_state_id: Optional[str]
    parent_state_id: Optional[str]
    branch_id: str
    problem_id: str
    verifier_id: str
    verifier_version: str
    harness_id: str
    policy_snapshot_id: str
    lineage_ids: Tuple[str, ...]
    resolved: bool
    admitted: bool
    confirmed: bool
    failure_kind: FailureKind
    internal_reward: Optional[float]
    raw_score: JSONValue
    uncertainty: Optional[float]
    descriptor_id: Optional[str]
    fingerprint: str
    source_hash: str
    flags: FrozenDict = field(default_factory=FrozenDict)
    scores: FrozenDict = field(default_factory=FrozenDict)
    diagnostics: FrozenDict = field(default_factory=FrozenDict)
    resources: FrozenDict = field(default_factory=FrozenDict)
    answer_payload: JSONValue = None
    timeout_is_scientific: bool = False
    started_at: str = ""
    completed_at: str = ""

    def __post_init__(self) -> None:
        super().__post_init__()
        _enum_field(self, "failure_kind", FailureKind)
        _id(self.evidence_id, "evidence", "evidence_id")
        _id(self.run_id, "run", "run_id")
        _id(self.proposal_id, "proposal", "proposal_id")
        _optional_id(self.scientific_state_id, "state", "scientific_state_id")
        _optional_id(self.parent_state_id, "state", "parent_state_id")
        _id(self.branch_id, "branch", "branch_id")
        _id(self.verifier_id, "verifier", "verifier_id")
        _id(self.harness_id, "harness", "harness_id")
        _id(self.policy_snapshot_id, "role_snapshot", "policy_snapshot_id")
        _nonempty(self.problem_id, "problem_id")
        _nonempty(self.verifier_version, "verifier_version")
        for lineage_id in self.lineage_ids:
            validate_id(lineage_id)
        _unique(self.lineage_ids, "lineage_ids")
        _require(isinstance(self.resolved, bool), "resolved must be boolean")
        _require(isinstance(self.admitted, bool), "admitted must be boolean")
        _require(isinstance(self.confirmed, bool), "confirmed must be boolean")
        _require(isinstance(self.timeout_is_scientific, bool),
                 "timeout_is_scientific must be boolean")
        if self.internal_reward is not None:
            _finite(self.internal_reward, "internal_reward")
        if self.uncertainty is not None:
            _finite(self.uncertainty, "uncertainty", minimum=0.0)
        if self.confirmed:
            _require(self.resolved and self.admitted,
                     "confirmed implies resolved and admitted")
        if self.admitted:
            _require(self.resolved, "admitted evidence must be resolved")
            _require(self.failure_kind == FailureKind.NONE,
                     "admitted evidence cannot carry a failure")
            _require(self.internal_reward is not None,
                     "admitted evidence needs internal_reward")
            _require(self.scientific_state_id is not None,
                     "admitted evidence must reference its scientific state")
            _require(self.answer_payload is not None,
                     "admitted evidence must capture answer_payload")
            _require(self.descriptor_id is not None,
                     "admitted evidence must reference a descriptor")
            _require(isinstance(self.fingerprint, str) and bool(self.fingerprint.strip()),
                     "admitted evidence must carry a scientific fingerprint")
        else:
            _require(self.failure_kind != FailureKind.NONE,
                     "non-admitted evidence must classify a failure")
            _require(self.scientific_state_id is None,
                     "non-admitted evidence cannot reference a scientific state")
            _require(self.internal_reward is None,
                     "non-admitted evidence cannot carry a scientific reward")
            _require(self.descriptor_id is None,
                     "non-admitted evidence cannot reference a descriptor")
            _require(not self.fingerprint,
                     "non-admitted evidence cannot carry a fingerprint")
        expected_resolved = {
            FailureKind.NONE: True,
            FailureKind.PARSE: True,
            FailureKind.CODE: True,
            FailureKind.CONSTRAINT: True,
            FailureKind.SCIENTIFIC: True,
            FailureKind.TIMEOUT: self.timeout_is_scientific,
            FailureKind.INFRASTRUCTURE: False,
        }[self.failure_kind]
        _require(
            self.resolved == expected_resolved,
            "evidence resolution contradicts its failure/timeout policy",
        )
        if self.failure_kind == FailureKind.INFRASTRUCTURE:
            _require(not self.resolved, "infrastructure failure must remain unresolved")
            _require(not self.admitted and not self.confirmed,
                     "infrastructure failure cannot be admitted or confirmed")
            _require(self.internal_reward is None,
                     "infrastructure failure cannot have a scientific reward")
        if self.failure_kind == FailureKind.TIMEOUT and not self.timeout_is_scientific:
            _require(not self.resolved,
                     "non-scientific timeout must remain unresolved")
            _require(not self.admitted and self.internal_reward is None,
                     "non-scientific timeout cannot affect admission or reward")
        _optional_id(self.descriptor_id, "descriptor", "descriptor_id")
        _sha256(self.source_hash, "source_hash")
        _numeric_resource_map(self.resources, "resources")


@dataclass(frozen=True)
class Descriptor(SchemaRecord):
    RECORD_TYPE: ClassVar[str] = "descriptor"

    descriptor_id: str
    problem_id: str
    function_version: str
    dimensions: FrozenDict
    method_complete: bool = True

    def __post_init__(self) -> None:
        super().__post_init__()
        _id(self.descriptor_id, "descriptor", "descriptor_id")
        _nonempty(self.problem_id, "problem_id")
        _nonempty(self.function_version, "function_version")
        _require(bool(self.dimensions), "descriptor dimensions must not be empty")
        _require(isinstance(self.method_complete, bool), "method_complete must be boolean")


@dataclass(frozen=True)
class ArchiveCell(SchemaRecord):
    RECORD_TYPE: ClassVar[str] = "archive_cell"

    cell_id: str
    descriptor_id: str
    champion_state_id: Optional[str] = None
    champion_evidence_id: Optional[str] = None
    promising_state_ids: Tuple[str, ...] = ()
    stepping_stone_state_ids: Tuple[str, ...] = ()
    tested_count: int = 0
    force_empty_sampling: bool = False
    under_tested: bool = True

    def __post_init__(self) -> None:
        super().__post_init__()
        _id(self.cell_id, "cell", "cell_id")
        _id(self.descriptor_id, "descriptor", "descriptor_id")
        _optional_id(self.champion_state_id, "state", "champion_state_id")
        _optional_id(self.champion_evidence_id, "evidence", "champion_evidence_id")
        _require((self.champion_state_id is None) == (self.champion_evidence_id is None),
                 "champion state and confirmed evidence references must be paired")
        for state_id in self.promising_state_ids + self.stepping_stone_state_ids:
            _id(state_id, "state", "archive slot state_id")
        _unique(self.promising_state_ids, "promising_state_ids")
        _unique(self.stepping_stone_state_ids, "stepping_stone_state_ids")
        _require(not set(self.promising_state_ids).intersection(self.stepping_stone_state_ids),
                 "promising and stepping-stone slots must be distinct")
        _nonnegative_int(self.tested_count, "tested_count")
        _require(isinstance(self.force_empty_sampling, bool),
                 "force_empty_sampling must be boolean")
        _require(isinstance(self.under_tested, bool), "under_tested must be boolean")


@dataclass(frozen=True)
class ProvenanceEdge(SchemaRecord):
    RECORD_TYPE: ClassVar[str] = "provenance_edge"

    edge_id: str
    parent_state_id: str
    child_state_id: str
    proposal_id: str
    evidence_id: str
    branch_id: str
    relation: str = "descendant"
    created_at: str = ""

    def __post_init__(self) -> None:
        super().__post_init__()
        _id(self.edge_id, "provenance", "edge_id")
        _id(self.parent_state_id, "state", "parent_state_id")
        _id(self.child_state_id, "state", "child_state_id")
        endpoints_equal = self.parent_state_id == self.child_state_id
        _require(
            endpoints_equal == (self.relation == "duplicate"),
            "relation='duplicate' is required exactly for equal provenance endpoints",
        )
        _id(self.proposal_id, "proposal", "proposal_id")
        _id(self.evidence_id, "evidence", "evidence_id")
        _id(self.branch_id, "branch", "branch_id")
        _nonempty(self.relation, "relation")


@dataclass(frozen=True)
class RoleSnapshot(SchemaRecord):
    RECORD_TYPE: ClassVar[str] = "role_snapshot"

    snapshot_id: str
    run_id: str
    epoch: int
    role: Role
    adapter_id: str
    adapter_version: str
    adapter_hash: str
    optimizer_state_id: str
    policy_version: str
    rng_seed: int
    frozen: bool = True

    def __post_init__(self) -> None:
        super().__post_init__()
        _enum_field(self, "role", Role)
        _id(self.snapshot_id, "role_snapshot", "snapshot_id")
        _id(self.run_id, "run", "run_id")
        _id(self.adapter_id, "adapter", "adapter_id")
        _id(self.optimizer_state_id, "optimizer", "optimizer_state_id")
        _nonnegative_int(self.epoch, "epoch")
        _nonempty(self.adapter_version, "adapter_version")
        _sha256(self.adapter_hash, "adapter_hash")
        _nonempty(self.policy_version, "policy_version")
        _nonnegative_int(self.rng_seed, "rng_seed")
        _require(self.frozen is True, "role snapshots must be frozen")


@dataclass(frozen=True)
class OptionSpec(SchemaRecord):
    RECORD_TYPE: ClassVar[str] = "option_spec"

    option_id: str
    version: str
    state_machine: str
    allowed_roles: Tuple[Role, ...]
    capabilities: Tuple[str, ...]
    initiation: FrozenDict
    step_policy: FrozenDict
    stop_rule: FrozenDict
    max_horizon: int
    expected_cost: FrozenDict
    hard_cost: FrozenDict
    harness_eligibility: Tuple[str, ...]
    prerequisites: Tuple[str, ...]
    output_contract: FrozenDict

    def __post_init__(self) -> None:
        super().__post_init__()
        object.__setattr__(
            self, "allowed_roles",
            tuple(role if isinstance(role, Role) else Role(role) for role in self.allowed_roles),
        )
        _id(self.option_id, "option", "option_id")
        _nonempty(self.version, "version")
        _nonempty(self.state_machine, "state_machine")
        _require(bool(self.allowed_roles), "allowed_roles must not be empty")
        _unique(self.allowed_roles, "allowed_roles")
        _unique(self.capabilities, "capabilities")
        _positive_int(self.max_horizon, "max_horizon")
        _numeric_resource_map(self.expected_cost, "expected_cost")
        _numeric_resource_map(self.hard_cost, "hard_cost")
        for resource, expected in self.expected_cost.items():
            if resource in self.hard_cost:
                _require(float(self.hard_cost[resource]) >= float(expected),
                         f"hard_cost.{resource} cannot be below expected_cost")
        for harness_id in self.harness_eligibility:
            _id(harness_id, "harness", "harness_eligibility")


@dataclass(frozen=True)
class HarnessSpec(SchemaRecord):
    RECORD_TYPE: ClassVar[str] = "harness_spec"

    harness_id: str
    version: str
    instructions: str
    tools: Tuple[str, ...]
    intermediate_tests: Tuple[str, ...]
    scaffolding: FrozenDict
    diagnostic_feedback: FrozenDict
    tool_policy_version: str
    spec_hash: str

    def identity_payload(self) -> dict:
        return {
            "version": self.version,
            "instructions": self.instructions,
            "tools": list(self.tools),
            "intermediate_tests": list(self.intermediate_tests),
            "scaffolding": _thaw_json(self.scaffolding),
            "diagnostic_feedback": _thaw_json(self.diagnostic_feedback),
            "tool_policy_version": self.tool_policy_version,
        }

    @classmethod
    def create(cls, **kwargs: Any) -> "HarnessSpec":
        payload = {
            "version": kwargs["version"],
            "instructions": kwargs.get("instructions", ""),
            "tools": list(kwargs.get("tools", ())),
            "intermediate_tests": list(kwargs.get("intermediate_tests", ())),
            "scaffolding": kwargs.get("scaffolding", {}),
            "diagnostic_feedback": kwargs.get("diagnostic_feedback", {}),
            "tool_policy_version": kwargs["tool_policy_version"],
        }
        kwargs["spec_hash"] = content_hash(payload)
        kwargs["harness_id"] = content_id("harness", payload)
        return cls(**kwargs)

    def __post_init__(self) -> None:
        super().__post_init__()
        _id(self.harness_id, "harness", "harness_id")
        _nonempty(self.version, "version")
        _nonempty(self.tool_policy_version, "tool_policy_version")
        _unique(self.tools, "tools")
        _sha256(self.spec_hash, "spec_hash")
        expected = content_hash(self.identity_payload())
        _require(self.spec_hash == expected,
                 "spec_hash must cover every harness behavior field")
        _require(self.harness_id == content_id("harness", self.identity_payload()),
                 "harness_id must be content-addressed from its behavior")


@dataclass(frozen=True)
class AllocationArm(SchemaRecord):
    RECORD_TYPE: ClassVar[str] = "allocation_arm"

    arm_id: str
    cell_id: str
    role: Role
    option_id: str
    harness_id: str
    horizon: int
    cost_class: str
    channel: Channel = Channel.PRODUCTION
    expected_cost: FrozenDict = field(default_factory=FrozenDict)
    hard_cost: FrozenDict = field(default_factory=FrozenDict)

    def __post_init__(self) -> None:
        super().__post_init__()
        _enum_field(self, "role", Role)
        _enum_field(self, "channel", Channel)
        _id(self.arm_id, "arm", "arm_id")
        _id(self.cell_id, "cell", "cell_id")
        _id(self.option_id, "option", "option_id")
        _id(self.harness_id, "harness", "harness_id")
        _positive_int(self.horizon, "horizon")
        _nonempty(self.cost_class, "cost_class")
        _numeric_resource_map(self.expected_cost, "expected_cost")
        _numeric_resource_map(self.hard_cost, "hard_cost")
        for resource, expected in self.expected_cost.items():
            if resource in self.hard_cost:
                _require(float(self.hard_cost[resource]) >= float(expected),
                         f"hard_cost.{resource} cannot be below expected_cost")


@dataclass(frozen=True)
class BranchSpec(SchemaRecord):
    RECORD_TYPE: ClassVar[str] = "branch_spec"

    branch_id: str
    arm_id: str
    epoch: int
    start_state_id: str
    frozen_record_threshold: float
    role_snapshot_id: str
    option_id: str
    option_version: str
    harness_id: str
    harness_version: str
    verifier_id: str
    verifier_version: str
    memory_view_id: Optional[str]
    memory_view_hash: str
    horizon: int
    budget: FrozenDict
    seed: int
    generation_settings: FrozenDict
    channel: Channel = Channel.PRODUCTION
    frozen: bool = True

    def __post_init__(self) -> None:
        super().__post_init__()
        _enum_field(self, "channel", Channel)
        _id(self.branch_id, "branch", "branch_id")
        _id(self.arm_id, "arm", "arm_id")
        _id(self.start_state_id, "state", "start_state_id")
        _id(self.role_snapshot_id, "role_snapshot", "role_snapshot_id")
        _id(self.option_id, "option", "option_id")
        _id(self.harness_id, "harness", "harness_id")
        _id(self.verifier_id, "verifier", "verifier_id")
        _any_optional_id(self.memory_view_id, "memory_view_id")
        _nonnegative_int(self.epoch, "epoch")
        _finite(self.frozen_record_threshold, "frozen_record_threshold")
        _nonempty(self.option_version, "option_version")
        _nonempty(self.harness_version, "harness_version")
        _nonempty(self.verifier_version, "verifier_version")
        _sha256(self.memory_view_hash, "memory_view_hash")
        _positive_int(self.horizon, "horizon")
        _require(bool(self.budget), "a frozen branch must reserve a non-empty budget")
        _numeric_resource_map(self.budget, "budget")
        _nonnegative_int(self.seed, "seed")
        _require(self.frozen is True, "a branch must have exactly one frozen BranchSpec")


@dataclass(frozen=True)
class BranchOutcome(SchemaRecord):
    RECORD_TYPE: ClassVar[str] = "branch_outcome"

    outcome_id: str
    branch_id: str
    branch_spec_hash: str
    status: BranchStatus
    descendant_proposal_ids: Tuple[str, ...]
    descendant_state_ids: Tuple[str, ...]
    evidence_ids: Tuple[str, ...]
    maximum_state_id: Optional[str]
    maximum_evidence_id: Optional[str]
    maximum_reward: Optional[float]
    costs: FrozenDict
    unused_budget: FrozenDict
    eligible_for_scheduler: bool
    infrastructure_aborted: bool = False

    def __post_init__(self) -> None:
        super().__post_init__()
        _enum_field(self, "status", BranchStatus)
        _id(self.outcome_id, "branch_outcome", "outcome_id")
        _id(self.branch_id, "branch", "branch_id")
        _sha256(self.branch_spec_hash, "branch_spec_hash")
        for proposal_id in self.descendant_proposal_ids:
            _id(proposal_id, "proposal", "descendant_proposal_ids")
        for state_id in self.descendant_state_ids:
            _id(state_id, "state", "descendant_state_ids")
        for evidence_id in self.evidence_ids:
            _id(evidence_id, "evidence", "evidence_ids")
        _unique(self.descendant_proposal_ids, "descendant_proposal_ids")
        _unique(self.descendant_state_ids, "descendant_state_ids")
        _unique(self.evidence_ids, "evidence_ids")
        _optional_id(self.maximum_state_id, "state", "maximum_state_id")
        _optional_id(self.maximum_evidence_id, "evidence", "maximum_evidence_id")
        _require(
            (self.maximum_state_id is None) == (self.maximum_evidence_id is None),
            "maximum state and evidence references must be present together",
        )
        paired = self.maximum_state_id is not None
        _require(paired == (self.maximum_reward is not None),
                 "maximum state/evidence/reward must be present together")
        if self.maximum_reward is not None:
            _finite(self.maximum_reward, "maximum_reward")
            _require(self.maximum_state_id in self.descendant_state_ids,
                     "maximum_state_id must be a verified descendant")
            _require(self.maximum_evidence_id in self.evidence_ids,
                     "maximum_evidence_id must belong to this branch")
        _numeric_resource_map(self.costs, "costs")
        _numeric_resource_map(self.unused_budget, "unused_budget")
        _require(isinstance(self.eligible_for_scheduler, bool),
                 "eligible_for_scheduler must be boolean")
        _require(isinstance(self.infrastructure_aborted, bool),
                 "infrastructure_aborted must be boolean")
        if self.eligible_for_scheduler:
            _require(self.status == BranchStatus.CLOSED,
                     "scheduler updates require a closed branch")
            _require(not self.infrastructure_aborted,
                     "infrastructure-aborted branch is not scheduler eligible")
        if self.infrastructure_aborted:
            _require(self.status == BranchStatus.ABORTED,
                     "infrastructure_aborted requires aborted status")
            _require(self.maximum_reward is None,
                     "infrastructure-aborted outcome cannot report a scientific maximum")


@dataclass(frozen=True)
class PolicyTrace(SchemaRecord):
    RECORD_TYPE: ClassVar[str] = "policy_trace"

    trace_id: str
    branch_id: str
    role_snapshot_id: str
    role: Role
    adapter_hash: str
    prompts: Tuple[str, ...]
    response_segments: Tuple[str, ...]
    token_masks: Tuple[Tuple[bool, ...], ...]
    log_probabilities: Tuple[Tuple[float, ...], ...]
    persisted: bool = True

    @property
    def branch_log_probability(self) -> float:
        return float(sum(
            logp
            for mask, values in zip(self.token_masks, self.log_probabilities)
            for keep, logp in zip(mask, values)
            if keep
        ))

    def __post_init__(self) -> None:
        super().__post_init__()
        _enum_field(self, "role", Role)
        _id(self.trace_id, "policy_trace", "trace_id")
        _id(self.branch_id, "branch", "branch_id")
        _id(self.role_snapshot_id, "role_snapshot", "role_snapshot_id")
        _sha256(self.adapter_hash, "adapter_hash")
        _require(all(isinstance(prompt, str) for prompt in self.prompts),
                 "PolicyTrace prompts must be strings")
        _require(all(isinstance(segment, str) for segment in self.response_segments),
                 "PolicyTrace response_segments must be strings")
        count = len(self.prompts)
        _require(count > 0, "a PolicyTrace must contain at least one policy decision")
        _require(len(self.response_segments) == count == len(self.token_masks) == len(self.log_probabilities),
                 "prompt, response, mask, and log-probability segments must align")
        for index, (mask, logps) in enumerate(zip(self.token_masks, self.log_probabilities)):
            _require(len(mask) == len(logps),
                     f"token mask/log probabilities differ in segment {index}")
            _require(all(isinstance(item, bool) for item in mask),
                     f"token_masks[{index}] must contain booleans")
            for value in logps:
                _finite(value, f"log_probabilities[{index}]")
        _require(any(keep for mask in self.token_masks for keep in mask),
                 "PolicyTrace must contain at least one role-policy token")
        _require(self.persisted is True,
                 "PolicyTrace must be persisted before it can enter learning")


@dataclass(frozen=True)
class AuditPair(SchemaRecord):
    RECORD_TYPE: ClassVar[str] = "audit_pair"

    audit_id: str
    run_id: str
    epoch: int
    start_state_id: str
    cell_id: str
    frozen_record_threshold: float
    role_snapshot_id: str
    harness_id: str
    verifier_id: str
    horizon: int
    resources: FrozenDict
    generation_settings: FrozenDict
    intervention_option_id: str
    control_option_id: str
    assignment_probability: float
    assignment_seed: int
    intervention_branch_id: str
    control_branch_id: str
    status: AuditStatus = AuditStatus.PREASSIGNED
    intervention_outcome_id: Optional[str] = None
    control_outcome_id: Optional[str] = None
    preassigned: bool = True

    def __post_init__(self) -> None:
        super().__post_init__()
        _enum_field(self, "status", AuditStatus)
        _id(self.audit_id, "audit_pair", "audit_id")
        _id(self.run_id, "run", "run_id")
        _id(self.start_state_id, "state", "start_state_id")
        _id(self.cell_id, "cell", "cell_id")
        _id(self.role_snapshot_id, "role_snapshot", "role_snapshot_id")
        _id(self.harness_id, "harness", "harness_id")
        _id(self.verifier_id, "verifier", "verifier_id")
        _id(self.intervention_option_id, "option", "intervention_option_id")
        _id(self.control_option_id, "option", "control_option_id")
        _id(self.intervention_branch_id, "branch", "intervention_branch_id")
        _id(self.control_branch_id, "branch", "control_branch_id")
        _require(self.intervention_branch_id != self.control_branch_id,
                 "audit sides need distinct preassigned branches")
        _require(self.intervention_option_id != self.control_option_id,
                 "audit intervention and matched continuation must differ")
        _nonnegative_int(self.epoch, "epoch")
        _finite(self.frozen_record_threshold, "frozen_record_threshold")
        _positive_int(self.horizon, "horizon")
        _numeric_resource_map(self.resources, "resources")
        probability = _finite(self.assignment_probability, "assignment_probability")
        _require(0.0 < probability < 1.0,
                 "assignment_probability must be strictly between zero and one")
        _nonnegative_int(self.assignment_seed, "assignment_seed")
        _require(self.preassigned is True,
                 "audit assignment and propensity must be persisted before execution")
        _optional_id(self.intervention_outcome_id, "branch_outcome", "intervention_outcome_id")
        _optional_id(self.control_outcome_id, "branch_outcome", "control_outcome_id")
        if self.status == AuditStatus.CLOSED:
            _require(self.intervention_outcome_id is not None and self.control_outcome_id is not None,
                     "both matched audit sides must close before effect computation")
        else:
            _require(not (self.intervention_outcome_id is not None and self.control_outcome_id is not None),
                     "a pair with both outcomes must be marked closed")


@dataclass(frozen=True)
class CausalMemoryRecord(SchemaRecord):
    RECORD_TYPE: ClassVar[str] = "causal_memory_record"

    memory_id: str
    context: FrozenDict
    intervention_option_id: str
    audit_pair_ids: Tuple[str, ...]
    propensities: Tuple[float, ...]
    effects: Tuple[float, ...]
    effect_mean: float
    uncertainty: float
    support: int
    recency_epoch: int
    scope: str
    contraindications: Tuple[str, ...]
    lineage_ids: Tuple[str, ...]
    status: MemoryStatus = MemoryStatus.QUARANTINED
    promotion_min_support: int = 2

    @property
    def conservative_effect(self) -> float:
        return float(self.effect_mean - self.uncertainty)

    def __post_init__(self) -> None:
        super().__post_init__()
        _enum_field(self, "status", MemoryStatus)
        _id(self.memory_id, "causal_memory", "memory_id")
        _id(self.intervention_option_id, "option", "intervention_option_id")
        for audit_id in self.audit_pair_ids:
            _id(audit_id, "audit_pair", "audit_pair_ids")
        _unique(self.audit_pair_ids, "audit_pair_ids")
        _require(len(self.audit_pair_ids) == len(self.effects) == len(self.propensities),
                 "pair IDs, propensities, and effects must align")
        _require(self.support == len(self.audit_pair_ids),
                 "support must equal the number of matched audit pairs")
        for probability in self.propensities:
            value = _finite(probability, "propensity")
            _require(0.0 < value < 1.0, "propensities must lie strictly in (0, 1)")
        for effect in self.effects:
            _finite(effect, "effect")
        _finite(self.effect_mean, "effect_mean")
        if self.effects:
            observed_mean = sum(float(effect) for effect in self.effects) / len(self.effects)
            _require(math.isclose(float(self.effect_mean), observed_mean,
                                  rel_tol=1e-12, abs_tol=1e-12),
                     "effect_mean must match persisted pair effects")
        _finite(self.uncertainty, "uncertainty", minimum=0.0)
        _nonnegative_int(self.support, "support")
        _nonnegative_int(self.recency_epoch, "recency_epoch")
        _positive_int(self.promotion_min_support, "promotion_min_support")
        _nonempty(self.scope, "scope")
        for lineage_id in self.lineage_ids:
            validate_id(lineage_id)
        if self.status == MemoryStatus.PROMOTED:
            _require(self.support >= self.promotion_min_support,
                     "promoted memory lacks repeated audit support")
            _require(bool(self.audit_pair_ids),
                     "promoted memory must be audit-backed")
            _require(self.conservative_effect > 0.0,
                     "promoted memory needs a positive conservative effect")


@dataclass(frozen=True)
class LearningGroup(SchemaRecord):
    RECORD_TYPE: ClassVar[str] = "learning_group"

    group_id: str
    role: Role
    policy_snapshot_id: str
    start_cell_id: str
    context_id: str
    option_id: str
    harness_id: str
    horizon: int
    cost_class: str
    generation_settings: FrozenDict
    frozen_record_threshold: float
    channel: Channel
    branch_ids: Tuple[str, ...]
    trace_ids: Tuple[str, ...]
    outcome_ids: Tuple[str, ...]
    advantages: Tuple[float, ...]
    objective: LearningObjective
    objective_version: str
    top_m: int
    audit_side: Optional[AuditSide] = None
    refinement_attempt: Optional[int] = None
    on_policy: bool = True
    persisted_inputs: bool = True
    homogeneous: bool = True

    def __post_init__(self) -> None:
        super().__post_init__()
        _enum_field(self, "role", Role)
        _enum_field(self, "channel", Channel)
        _enum_field(self, "objective", LearningObjective)
        if self.audit_side is not None:
            _enum_field(self, "audit_side", AuditSide)
        _id(self.group_id, "learning_group", "group_id")
        _id(self.policy_snapshot_id, "role_snapshot", "policy_snapshot_id")
        _id(self.start_cell_id, "cell", "start_cell_id")
        _id(self.context_id, "context", "context_id")
        _id(self.option_id, "option", "option_id")
        _id(self.harness_id, "harness", "harness_id")
        count = len(self.branch_ids)
        _require(count > 0, "learning group must not be empty")
        _require(count == len(self.trace_ids) == len(self.outcome_ids) == len(self.advantages),
                 "learning members, traces, outcomes, and advantages must align")
        for branch_id in self.branch_ids:
            _id(branch_id, "branch", "branch_ids")
        for trace_id in self.trace_ids:
            _id(trace_id, "policy_trace", "trace_ids")
        for outcome_id in self.outcome_ids:
            _id(outcome_id, "branch_outcome", "outcome_ids")
        _unique(self.branch_ids, "branch_ids")
        _unique(self.trace_ids, "trace_ids")
        _unique(self.outcome_ids, "outcome_ids")
        for advantage in self.advantages:
            _finite(advantage, "advantage")
        _positive_int(self.horizon, "horizon")
        _finite(self.frozen_record_threshold, "frozen_record_threshold")
        _nonempty(self.cost_class, "cost_class")
        _nonempty(self.objective_version, "objective_version")
        _positive_int(self.top_m, "top_m")
        _require(self.top_m <= count, "top_m cannot exceed group size")
        if self.objective == LearningObjective.MAXPO:
            _require(self.top_m == 1, "MaxPO is only valid for pure-max top_m=1")
        _require(self.on_policy is True, "learning groups must be on-policy")
        _require(self.persisted_inputs is True,
                 "learning inputs must be persisted before backward")
        _require(self.homogeneous is True,
                 "mixed roles, versions, contexts, budgets, or channels must be rejected")
        if self.channel == Channel.AUDIT:
            _require(self.audit_side is not None,
                     "audit learning groups must identify one homogeneous audit side")
        else:
            _require(self.audit_side is None,
                     "audit_side is only valid in the audit channel")
        if self.channel == Channel.REFINEMENT:
            _require(self.refinement_attempt is not None,
                     "refinement groups must identify one attempt")
        else:
            _require(self.refinement_attempt is None,
                     "refinement_attempt is only valid in refinement groups")
        if self.refinement_attempt is not None:
            _nonnegative_int(self.refinement_attempt, "refinement_attempt")


@dataclass(frozen=True)
class BudgetTransaction(SchemaRecord):
    RECORD_TYPE: ClassVar[str] = "budget_transaction"

    transaction_id: str
    ledger_id: str
    transaction_key: str
    resource: str
    kind: BudgetTransactionKind
    amount: float
    debit_transaction_key: Optional[str] = None

    def __post_init__(self) -> None:
        super().__post_init__()
        _enum_field(self, "kind", BudgetTransactionKind)
        _id(self.transaction_id, "budget_transaction", "transaction_id")
        _id(self.ledger_id, "budget", "ledger_id")
        _nonempty(self.transaction_key, "transaction_key")
        _nonempty(self.resource, "resource")
        _finite(self.amount, "amount", minimum=0.0)
        _require(float(self.amount) > 0.0, "transaction amount must be positive")
        if self.kind == BudgetTransactionKind.REFUND:
            _nonempty(self.debit_transaction_key, "debit_transaction_key")
            _require(self.debit_transaction_key != self.transaction_key,
                     "refund and debit transaction keys must differ")
        else:
            _require(self.debit_transaction_key is None,
                     "debit transaction cannot reference another debit")
        identity = {
            "ledger_id": self.ledger_id,
            "transaction_key": self.transaction_key,
            "resource": self.resource,
            "kind": self.kind.value,
            "amount": float(self.amount),
        }
        if self.debit_transaction_key is not None:
            identity["debit_transaction_key"] = self.debit_transaction_key
        _require(
            self.transaction_id == content_id("budget_transaction", identity),
            "transaction_id must match the complete budget transaction content",
        )


@dataclass(frozen=True)
class BudgetLedger(SchemaRecord):
    RECORD_TYPE: ClassVar[str] = "budget_ledger"

    ledger_id: str
    limits: FrozenDict
    transactions: Tuple[BudgetTransaction, ...] = ()

    def __post_init__(self) -> None:
        # Decode nested transactions before SchemaRecord validates JSON.
        object.__setattr__(
            self,
            "transactions",
            tuple(
                item if isinstance(item, BudgetTransaction)
                else BudgetTransaction.from_dict(item)
                for item in self.transactions
            ),
        )
        super().__post_init__()
        _id(self.ledger_id, "budget", "ledger_id")
        _require(bool(self.limits), "budget limits must not be empty")
        _numeric_resource_map(self.limits, "limits")
        seen = {}
        debit_by_key = {}
        refunded = {}
        net = {resource: 0.0 for resource in self.limits}
        for transaction in self.transactions:
            _require(
                transaction.ledger_id == self.ledger_id,
                "budget transaction must reference its containing ledger",
            )
            previous = seen.get(transaction.transaction_key)
            if previous is not None:
                _require(previous == transaction,
                         "a transaction key cannot describe two different operations")
                # Exact duplicates would still double-count if retained in the
                # sequence, so canonical ledgers reject them.  The service makes
                # retries idempotent by returning the original ledger instead.
                raise InvariantViolation(
                    f"duplicate transaction key {transaction.transaction_key!r}"
                )
            seen[transaction.transaction_key] = transaction
            _require(transaction.resource in self.limits,
                     f"unknown budget resource {transaction.resource!r}")
            if transaction.kind == BudgetTransactionKind.DEBIT:
                debit_by_key[transaction.transaction_key] = transaction
                net[transaction.resource] += float(transaction.amount)
            else:
                debit = debit_by_key.get(transaction.debit_transaction_key)
                _require(debit is not None,
                         "refund must reference an earlier debit in the same ledger")
                _require(debit.resource == transaction.resource,
                         "refund resource must match its debit")
                refunded[debit.transaction_key] = (
                    refunded.get(debit.transaction_key, 0.0) + float(transaction.amount)
                )
                _require(refunded[debit.transaction_key] <= float(debit.amount) + 1e-12,
                         "refunds cannot exceed their debit")
                net[transaction.resource] -= float(transaction.amount)
            _require(net[transaction.resource] >= -1e-12,
                     "resource balance cannot become negative")
            _require(net[transaction.resource] <= float(self.limits[transaction.resource]) + 1e-12,
                     f"budget overrun for resource {transaction.resource!r}")

    def consumed(self, resource: str) -> float:
        _require(resource in self.limits, f"unknown budget resource {resource!r}")
        amount = 0.0
        for transaction in self.transactions:
            if transaction.resource != resource:
                continue
            sign = 1.0 if transaction.kind == BudgetTransactionKind.DEBIT else -1.0
            amount += sign * float(transaction.amount)
        return float(amount)

    def remaining(self, resource: str) -> float:
        return float(self.limits[resource]) - self.consumed(resource)

    def to_dict(self) -> dict:
        payload = super().to_dict()
        payload["transactions"] = [transaction.to_dict() for transaction in self.transactions]
        return payload


@dataclass(frozen=True)
class EpochManifest(SchemaRecord):
    RECORD_TYPE: ClassVar[str] = "epoch_manifest"

    manifest_id: str
    run_id: str
    epoch: int
    record_threshold: float
    archive_snapshot_id: str
    archive_snapshot_hash: str
    scheduler_version: str
    scheduler_snapshot_id: str
    role_snapshot_ids: FrozenDict
    causal_memory_snapshot_id: str
    option_ids: Tuple[str, ...]
    harness_ids: Tuple[str, ...]
    verifier_id: str
    verifier_version: str
    descriptor_version: str
    cell_map_version: str
    fingerprint_version: str
    reporting_schema_version: str
    budget_ledger_id: str
    allocation_plan_id: str
    seed: int
    component_schema_versions: FrozenDict
    method_complete: bool = True
    frozen: bool = True
    created_at: str = ""

    def __post_init__(self) -> None:
        super().__post_init__()
        _id(self.manifest_id, "epoch_manifest", "manifest_id")
        _id(self.run_id, "run", "run_id")
        _id(self.archive_snapshot_id, "archive_snapshot", "archive_snapshot_id")
        _id(self.scheduler_snapshot_id, "scheduler_snapshot", "scheduler_snapshot_id")
        _id(self.causal_memory_snapshot_id, "causal_memory_snapshot", "causal_memory_snapshot_id")
        _id(self.verifier_id, "verifier", "verifier_id")
        _id(self.budget_ledger_id, "budget", "budget_ledger_id")
        _id(self.allocation_plan_id, "allocation_plan", "allocation_plan_id")
        _nonnegative_int(self.epoch, "epoch")
        _finite(self.record_threshold, "record_threshold")
        _sha256(self.archive_snapshot_hash, "archive_snapshot_hash")
        _nonempty(self.scheduler_version, "scheduler_version")
        _nonempty(self.verifier_version, "verifier_version")
        _nonempty(self.descriptor_version, "descriptor_version")
        _nonempty(self.cell_map_version, "cell_map_version")
        _nonempty(self.fingerprint_version, "fingerprint_version")
        _nonempty(self.reporting_schema_version, "reporting_schema_version")
        _nonnegative_int(self.seed, "seed")
        _require(self.frozen is True, "epoch manifest must be frozen before dispatch")
        _require(isinstance(self.method_complete, bool), "method_complete must be boolean")
        for role_name, snapshot_id in self.role_snapshot_ids.items():
            try:
                Role(role_name)
            except ValueError as exc:
                raise InvariantViolation(f"unknown role in manifest: {role_name!r}") from exc
            _id(snapshot_id, "role_snapshot", f"role_snapshot_ids.{role_name}")
        expected_roles = {role.value for role in Role}
        if self.method_complete:
            _require(set(self.role_snapshot_ids) == expected_roles,
                     "production manifest requires exactly scout, mechanist, and challenger")
        else:
            _require(bool(self.role_snapshot_ids),
                     "method-incomplete test manifest still needs at least one role")
        for option_id in self.option_ids:
            _id(option_id, "option", "option_ids")
        for harness_id in self.harness_ids:
            _id(harness_id, "harness", "harness_ids")
        _unique(self.option_ids, "option_ids")
        _unique(self.harness_ids, "harness_ids")
        _require(bool(self.option_ids), "epoch manifest must freeze at least one option")
        _require(bool(self.harness_ids), "epoch manifest must freeze at least one harness")
        _require(bool(self.component_schema_versions),
                 "component_schema_versions must be persisted")
        for component, version in self.component_schema_versions.items():
            _nonempty(component, "component schema name")
            _positive_int(version, f"component_schema_versions.{component}")


RECORD_TYPES: Mapping[str, Type[SchemaRecord]] = MappingProxyType({
    cls.RECORD_TYPE: cls
    for cls in (
        Proposal, VerifiedScientificState, EvidencePacket, Descriptor,
        ArchiveCell, ProvenanceEdge, RoleSnapshot, OptionSpec, HarnessSpec,
        AllocationArm, BranchSpec, BranchOutcome, PolicyTrace, AuditPair,
        CausalMemoryRecord, LearningGroup, BudgetTransaction, BudgetLedger,
        EpochManifest,
    )
})


def record_from_dict(payload: Mapping[str, Any]) -> SchemaRecord:
    """Decode any registered record using its explicit ``record_type``."""

    if not isinstance(payload, Mapping):
        raise SchemaValidationError("record payload must be a mapping")
    record_type = payload.get("record_type")
    if not isinstance(record_type, str) or record_type not in RECORD_TYPES:
        raise SchemaValidationError(f"unknown record_type: {record_type!r}")
    return RECORD_TYPES[record_type].from_dict(payload)


__all__ = [
    "SCHEMA_VERSION", "JSONValue", "FrozenDict",
    "SchemaValidationError", "UnsupportedSchemaVersion", "InvariantViolation",
    "Role", "Channel", "FailureKind", "BranchStatus", "AuditStatus",
    "AuditSide", "MemoryStatus", "LearningObjective", "BudgetTransactionKind",
    "SchemaRecord", "Proposal", "VerifiedScientificState", "EvidencePacket",
    "Descriptor", "ArchiveCell", "ProvenanceEdge", "RoleSnapshot",
    "OptionSpec", "HarnessSpec", "AllocationArm", "BranchSpec",
    "BranchOutcome", "PolicyTrace", "AuditPair", "CausalMemoryRecord",
    "LearningGroup", "BudgetTransaction", "BudgetLedger", "EpochManifest",
    "RECORD_TYPES", "record_from_dict",
]
