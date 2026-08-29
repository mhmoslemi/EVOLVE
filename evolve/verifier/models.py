"""Frozen, JSON-safe records used at the independent-verifier boundary."""

from __future__ import annotations

import json
import math
import os
from pathlib import Path
import stat
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Mapping, Optional, Tuple

from evolve.ids import canonical_json, content_hash, content_id, validate_id
from evolve.types import FailureKind, FrozenDict


class VerificationValidationError(ValueError):
    """A verifier input or result violates the scientific boundary."""


def _reject_duplicate_json_keys(pairs):
    value = {}
    for key, item in pairs:
        if key in value:
            raise VerificationValidationError(
                f"durable answer artifact contains duplicate JSON key {key!r}"
            )
        value[key] = item
    return value


def _reject_json_constant(value: str):
    raise VerificationValidationError(
        f"durable answer artifact contains non-finite JSON constant {value}"
    )


def _normalize_artifact_path(artifact_uri: str) -> Path:
    _nonempty(artifact_uri, "artifact_uri")
    if "\x00" in artifact_uri:
        raise VerificationValidationError("artifact_uri contains a NUL byte")
    candidate = Path(artifact_uri).expanduser()
    if not candidate.is_absolute():
        candidate = Path.cwd() / candidate
    try:
        # Reject a symlink at the artifact boundary.  Ancestor links are
        # resolved once and the real absolute path is what enters evidence.
        if candidate.is_symlink():
            raise VerificationValidationError(
                "durable answer artifact must not be a symlink"
            )
        normalized = candidate.resolve(strict=True)
    except VerificationValidationError:
        raise
    except (OSError, RuntimeError) as exc:
        raise VerificationValidationError(
            f"durable answer artifact does not exist or cannot be resolved: {exc}"
        ) from exc
    return normalized


def _read_regular_json_artifact(path: Path) -> Any:
    try:
        initial = os.lstat(str(path))
    except OSError as exc:
        raise VerificationValidationError(
            f"durable answer artifact cannot be inspected: {exc}"
        ) from exc
    if stat.S_ISLNK(initial.st_mode):
        raise VerificationValidationError(
            "durable answer artifact must not be a symlink"
        )
    if not stat.S_ISREG(initial.st_mode):
        raise VerificationValidationError(
            "durable answer artifact must be a regular file"
        )
    flags = os.O_RDONLY
    if hasattr(os, "O_NOFOLLOW"):
        flags |= os.O_NOFOLLOW
    try:
        descriptor = os.open(str(path), flags)
    except OSError as exc:
        raise VerificationValidationError(
            f"durable answer artifact cannot be opened safely: {exc}"
        ) from exc
    try:
        metadata = os.fstat(descriptor)
        if not stat.S_ISREG(metadata.st_mode):
            raise VerificationValidationError(
                "durable answer artifact must be a regular file"
            )
        with os.fdopen(descriptor, "rb") as stream:
            descriptor = -1
            encoded = stream.read()
    finally:
        if descriptor >= 0:
            os.close(descriptor)
    try:
        text = encoded.decode("utf-8")
    except UnicodeDecodeError as exc:
        raise VerificationValidationError(
            f"durable answer artifact must be UTF-8 JSON: {exc}"
        ) from exc
    try:
        return json.loads(
            text,
            object_pairs_hook=_reject_duplicate_json_keys,
            parse_constant=_reject_json_constant,
        )
    except VerificationValidationError:
        raise
    except (TypeError, ValueError, json.JSONDecodeError) as exc:
        raise VerificationValidationError(
            f"durable answer artifact is malformed JSON: {exc}"
        ) from exc


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise VerificationValidationError(message)


def _nonempty(value: Any, name: str) -> str:
    _require(isinstance(value, str) and bool(value.strip()), f"{name} must be non-empty")
    return value


def _nonnegative_int(value: Any, name: str) -> int:
    _require(isinstance(value, int) and not isinstance(value, bool), f"{name} must be an integer")
    _require(value >= 0, f"{name} must be non-negative")
    return value


def _finite(value: Any, name: str, *, minimum: Optional[float] = None) -> float:
    _require(
        isinstance(value, (int, float)) and not isinstance(value, bool),
        f"{name} must be numeric",
    )
    number = float(value)
    _require(math.isfinite(number), f"{name} must be finite")
    if minimum is not None:
        _require(number >= minimum, f"{name} must be >= {minimum}")
    return number


def _freeze_json(value: Any, name: str = "value") -> Any:
    if isinstance(value, Enum):
        return value
    if value is None or isinstance(value, (str, bool, int)):
        return value
    if isinstance(value, float):
        _finite(value, name)
        return 0.0 if value == 0.0 else value
    if isinstance(value, Mapping):
        try:
            return FrozenDict(value)
        except (TypeError, ValueError) as exc:
            raise VerificationValidationError(f"{name} must be JSON-safe: {exc}") from exc
    if isinstance(value, (list, tuple)):
        return tuple(_freeze_json(item, f"{name}[]") for item in value)
    raise VerificationValidationError(
        f"{name} contains non-JSON value of type {type(value).__name__}"
    )


def thaw_json(value: Any) -> Any:
    """Return a detached JSON-native copy of a frozen verifier value."""

    if isinstance(value, Enum):
        return value.value
    if isinstance(value, Mapping):
        return {key: thaw_json(item) for key, item in value.items()}
    if isinstance(value, tuple):
        return [thaw_json(item) for item in value]
    return value


@dataclass(frozen=True)
class VerificationPolicy:
    """Content-addressed policy controlling one common-verifier invocation."""

    policy_id: str
    version: str
    production: bool = True
    max_diagnostic_chars: int = 4096
    max_diagnostic_entries: int = 32
    infrastructure_retry_limit: int = 1
    diagnostic_policy_version: str = "bounded_diagnostics_v1"
    resource_limits: FrozenDict = field(default_factory=FrozenDict)

    @classmethod
    def create(
        cls,
        *,
        version: str,
        production: bool = True,
        max_diagnostic_chars: int = 4096,
        max_diagnostic_entries: int = 32,
        infrastructure_retry_limit: int = 1,
        diagnostic_policy_version: str = "bounded_diagnostics_v1",
        resource_limits: Optional[Mapping[str, Any]] = None,
    ) -> "VerificationPolicy":
        values = {
            "version": version,
            "production": production,
            "max_diagnostic_chars": max_diagnostic_chars,
            "max_diagnostic_entries": max_diagnostic_entries,
            "infrastructure_retry_limit": infrastructure_retry_limit,
            "diagnostic_policy_version": diagnostic_policy_version,
            "resource_limits": dict(resource_limits or {}),
        }
        return cls(policy_id=content_id("verification_policy", values), **values)

    def __post_init__(self) -> None:
        _nonempty(self.version, "version")
        _require(isinstance(self.production, bool), "production must be boolean")
        _nonnegative_int(self.max_diagnostic_chars, "max_diagnostic_chars")
        _require(
            self.max_diagnostic_chars >= 256,
            "max_diagnostic_chars must be >= 256 so truncation metadata fits",
        )
        _nonnegative_int(self.max_diagnostic_entries, "max_diagnostic_entries")
        _require(self.max_diagnostic_entries > 0, "max_diagnostic_entries must be positive")
        _nonnegative_int(self.infrastructure_retry_limit, "infrastructure_retry_limit")
        _nonempty(self.diagnostic_policy_version, "diagnostic_policy_version")
        limits = FrozenDict(self.resource_limits)
        for resource, amount in limits.items():
            _nonempty(resource, "resource_limits key")
            _finite(amount, f"resource_limits.{resource}", minimum=0.0)
        object.__setattr__(self, "resource_limits", limits)
        try:
            validate_id(self.policy_id, "verification_policy")
        except (TypeError, ValueError) as exc:
            raise VerificationValidationError(f"invalid policy_id: {exc}") from exc
        expected = content_id("verification_policy", self.identity_payload())
        _require(self.policy_id == expected, "policy_id must match policy content")

    def identity_payload(self) -> Mapping[str, Any]:
        return {
            "version": self.version,
            "production": self.production,
            "max_diagnostic_chars": self.max_diagnostic_chars,
            "max_diagnostic_entries": self.max_diagnostic_entries,
            "infrastructure_retry_limit": self.infrastructure_retry_limit,
            "diagnostic_policy_version": self.diagnostic_policy_version,
            "resource_limits": thaw_json(self.resource_limits),
        }

    def to_dict(self) -> Mapping[str, Any]:
        return {"policy_id": self.policy_id, **self.identity_payload()}


@dataclass(frozen=True)
class PersistedAnswerPayload:
    """A content-addressed payload backed by an exact durable JSON artifact."""

    payload_id: str
    problem_id: str
    artifact_uri: str
    payload_hash: str
    payload: Any

    @classmethod
    def create(
        cls,
        *,
        problem_id: str,
        artifact_uri: str,
        payload: Any,
    ) -> "PersistedAnswerPayload":
        frozen = _freeze_json(payload, "payload")
        payload_hash = content_hash(frozen)
        payload_id = content_id(
            "answer_payload",
            {"problem_id": problem_id, "payload": frozen},
        )
        return cls(
            payload_id=payload_id,
            problem_id=problem_id,
            artifact_uri=artifact_uri,
            payload_hash=payload_hash,
            payload=frozen,
        )

    def __post_init__(self) -> None:
        _nonempty(self.problem_id, "problem_id")
        normalized_path = _normalize_artifact_path(self.artifact_uri)
        object.__setattr__(self, "artifact_uri", str(normalized_path))
        frozen = _freeze_json(self.payload, "payload")
        object.__setattr__(self, "payload", frozen)
        _require(self.payload_hash == content_hash(frozen), "payload_hash must match payload")
        try:
            validate_id(self.payload_id, "answer_payload")
        except (TypeError, ValueError) as exc:
            raise VerificationValidationError(f"invalid payload_id: {exc}") from exc
        expected = content_id(
            "answer_payload",
            {"problem_id": self.problem_id, "payload": frozen},
        )
        _require(self.payload_id == expected, "payload_id must match problem and payload")
        self.validate_durable_artifact()
        canonical_json(self.to_dict())

    def validate_durable_artifact(self) -> str:
        """Re-read and match the artifact immediately before verification.

        Construction proves persistence at capture time.  Revalidation closes
        the deletion/tampering gap before both initial verification and later
        record confirmation.
        """

        normalized_path = _normalize_artifact_path(self.artifact_uri)
        _require(
            str(normalized_path) == self.artifact_uri,
            "artifact_uri must remain the normalized absolute artifact path",
        )
        parsed = _read_regular_json_artifact(normalized_path)
        try:
            artifact_content = canonical_json(parsed)
            expected_content = canonical_json(thaw_json(self.payload))
        except (TypeError, ValueError) as exc:
            raise VerificationValidationError(
                f"durable answer artifact is not canonical JSON: {exc}"
            ) from exc
        _require(
            artifact_content == expected_content,
            "durable answer artifact content does not match the payload",
        )
        return self.artifact_uri

    def to_dict(self) -> Mapping[str, Any]:
        return {
            "payload_id": self.payload_id,
            "problem_id": self.problem_id,
            "artifact_uri": self.artifact_uri,
            "payload_hash": self.payload_hash,
            "payload": thaw_json(self.payload),
        }


@dataclass(frozen=True)
class ExecutionCapture:
    """Bounded-input execution metadata returned by a problem verifier."""

    diagnostics: FrozenDict = field(default_factory=FrozenDict)
    resources: FrozenDict = field(default_factory=FrozenDict)
    started_at: str = ""
    completed_at: str = ""
    attempt_index: int = 0

    def __post_init__(self) -> None:
        diagnostics = FrozenDict(self.diagnostics)
        resources = FrozenDict(self.resources)
        for resource, amount in resources.items():
            _nonempty(resource, "resources key")
            _finite(amount, f"resources.{resource}", minimum=0.0)
        _require(isinstance(self.started_at, str), "started_at must be a string")
        _require(isinstance(self.completed_at, str), "completed_at must be a string")
        _nonnegative_int(self.attempt_index, "attempt_index")
        object.__setattr__(self, "diagnostics", diagnostics)
        object.__setattr__(self, "resources", resources)


@dataclass(frozen=True)
class FailureClassification:
    failure_kind: FailureKind
    resolved: bool
    excluded_from_scientific_updates: bool

    def __post_init__(self) -> None:
        try:
            kind = (
                self.failure_kind
                if isinstance(self.failure_kind, FailureKind)
                else FailureKind(self.failure_kind)
            )
        except (TypeError, ValueError) as exc:
            raise VerificationValidationError(f"unknown failure kind: {self.failure_kind!r}") from exc
        object.__setattr__(self, "failure_kind", kind)
        _require(isinstance(self.resolved, bool), "resolved must be boolean")
        _require(
            isinstance(self.excluded_from_scientific_updates, bool),
            "excluded_from_scientific_updates must be boolean",
        )


def classify_failure(
    *,
    infrastructure_error: bool = False,
    timed_out: bool = False,
    parsed: bool = True,
    executed: bool = True,
    constraints_satisfied: bool = True,
    scientifically_valid: bool = True,
    timeout_is_scientific: bool = False,
) -> FailureClassification:
    """Classify a verifier outcome with explicit, deterministic precedence."""

    values = (
        infrastructure_error,
        timed_out,
        parsed,
        executed,
        constraints_satisfied,
        scientifically_valid,
        timeout_is_scientific,
    )
    _require(all(isinstance(value, bool) for value in values), "classification flags must be boolean")
    if infrastructure_error:
        return FailureClassification(FailureKind.INFRASTRUCTURE, False, True)
    if timed_out:
        return FailureClassification(
            FailureKind.TIMEOUT,
            timeout_is_scientific,
            not timeout_is_scientific,
        )
    if not parsed:
        return FailureClassification(FailureKind.PARSE, True, False)
    if not executed:
        return FailureClassification(FailureKind.CODE, True, False)
    if not constraints_satisfied:
        return FailureClassification(FailureKind.CONSTRAINT, True, False)
    if not scientifically_valid:
        return FailureClassification(FailureKind.SCIENTIFIC, True, False)
    return FailureClassification(FailureKind.NONE, True, False)


@dataclass(frozen=True)
class VerificationDecision:
    """Problem-verifier decision before EVOLVE adds global references and IDs."""

    failure_kind: FailureKind
    resolved: bool
    admitted: bool
    internal_reward: Optional[float]
    raw_score: Any = None
    uncertainty: Optional[float] = None
    flags: FrozenDict = field(default_factory=FrozenDict)
    scores: FrozenDict = field(default_factory=FrozenDict)
    capture: ExecutionCapture = field(default_factory=ExecutionCapture)

    def __post_init__(self) -> None:
        try:
            kind = (
                self.failure_kind
                if isinstance(self.failure_kind, FailureKind)
                else FailureKind(self.failure_kind)
            )
        except (TypeError, ValueError) as exc:
            raise VerificationValidationError(f"unknown failure kind: {self.failure_kind!r}") from exc
        object.__setattr__(self, "failure_kind", kind)
        _require(isinstance(self.resolved, bool), "resolved must be boolean")
        _require(isinstance(self.admitted, bool), "admitted must be boolean")
        if self.internal_reward is not None:
            _finite(self.internal_reward, "internal_reward")
        if self.uncertainty is not None:
            _finite(self.uncertainty, "uncertainty", minimum=0.0)
        object.__setattr__(self, "raw_score", _freeze_json(self.raw_score, "raw_score"))
        object.__setattr__(self, "flags", FrozenDict(self.flags))
        object.__setattr__(self, "scores", FrozenDict(self.scores))
        _require(isinstance(self.capture, ExecutionCapture), "capture must be ExecutionCapture")
        if self.admitted:
            _require(self.resolved, "admitted decision must be resolved")
            _require(kind == FailureKind.NONE, "admitted decision cannot carry a failure")
            _require(self.internal_reward is not None, "admitted decision needs internal_reward")
        else:
            _require(self.internal_reward is None, "non-admitted decision cannot carry internal_reward")
            _require(kind != FailureKind.NONE, "non-admitted decision must classify its failure")
        if kind == FailureKind.INFRASTRUCTURE:
            _require(not self.resolved, "infrastructure decision must remain unresolved")

    @classmethod
    def success(
        cls,
        *,
        internal_reward: float,
        raw_score: Any,
        uncertainty: Optional[float] = None,
        flags: Optional[Mapping[str, Any]] = None,
        scores: Optional[Mapping[str, Any]] = None,
        capture: Optional[ExecutionCapture] = None,
    ) -> "VerificationDecision":
        return cls(
            failure_kind=FailureKind.NONE,
            resolved=True,
            admitted=True,
            internal_reward=internal_reward,
            raw_score=raw_score,
            uncertainty=uncertainty,
            flags=FrozenDict(flags or {}),
            scores=FrozenDict(scores or {}),
            capture=capture or ExecutionCapture(),
        )

    @classmethod
    def failure(
        cls,
        classification: FailureClassification,
        *,
        raw_score: Any = None,
        uncertainty: Optional[float] = None,
        flags: Optional[Mapping[str, Any]] = None,
        scores: Optional[Mapping[str, Any]] = None,
        capture: Optional[ExecutionCapture] = None,
    ) -> "VerificationDecision":
        _require(
            classification.failure_kind != FailureKind.NONE,
            "failure decision requires a failure classification",
        )
        return cls(
            failure_kind=classification.failure_kind,
            resolved=classification.resolved,
            admitted=False,
            internal_reward=None,
            raw_score=raw_score,
            uncertainty=uncertainty,
            flags=FrozenDict(flags or {}),
            scores=FrozenDict(scores or {}),
            capture=capture or ExecutionCapture(),
        )


__all__ = [
    "ExecutionCapture",
    "FailureClassification",
    "PersistedAnswerPayload",
    "VerificationDecision",
    "VerificationPolicy",
    "VerificationValidationError",
    "classify_failure",
    "thaw_json",
]
