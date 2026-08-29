"""Problem-adapter protocol for deterministic saved-payload verification."""

from __future__ import annotations

import json
from typing import Any, Callable, Mapping, Optional, Protocol, runtime_checkable

from evolve.ids import canonical_json, content_hash, content_id
from evolve.types import FailureKind, FrozenDict

from .models import (
    ExecutionCapture,
    VerificationDecision,
    VerificationPolicy,
    VerificationValidationError,
    classify_failure,
    thaw_json,
)


_RESOURCE_IDENTITY_FIELDS = (
    "cpu_cores",
    "memory_mb",
    "timeout_s",
    "gpu_count",
    "exclusive_gpu",
    "network_access",
    "filesystem_policy",
    "timeout_is_scientific",
)


def _canonical_mapping(value: Any, name: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise VerificationValidationError(f"{name} must return a mapping")
    try:
        # Round-trip through the repository's canonical encoder both validates
        # the hook and detaches it from mutable problem-owned containers.
        canonical = json.loads(canonical_json(value))
    except (TypeError, ValueError) as exc:
        raise VerificationValidationError(
            f"{name} must be finite canonical JSON: {exc}"
        ) from exc
    if not isinstance(canonical, Mapping):
        raise VerificationValidationError(f"{name} must be a JSON object")
    return canonical


def _resource_identity(resources: Any) -> Mapping[str, Any]:
    if callable(getattr(resources, "to_dict", None)):
        raw = resources.to_dict()
    elif isinstance(resources, Mapping):
        raw = dict(resources)
    else:
        raw = {
            field: getattr(resources, field)
            for field in _RESOURCE_IDENTITY_FIELDS
            if hasattr(resources, field)
        }
    if not raw:
        raise VerificationValidationError(
            "problem resources must expose a JSON resource declaration"
        )
    return _canonical_mapping(raw, "problem resource identity")


def _problem_identity(problem: Any) -> Mapping[str, Any]:
    hook = getattr(problem, "scientific_verifier_identity", None)
    if callable(hook):
        try:
            raw = hook()
        except Exception as exc:
            raise VerificationValidationError(
                "problem scientific_verifier_identity failed: "
                f"{type(exc).__name__}: {exc}"
            ) from exc
        return _canonical_mapping(raw, "problem scientific_verifier_identity")

    # Duck-typed test/problem adapters predate the additive base hook.  Their
    # fallback remains safe and deterministic: no repr(), object address, or
    # mutable runtime transcript enters identity.
    config = getattr(problem, "cfg", {})
    if config is None:
        config = {}
    if not isinstance(config, Mapping):
        raise VerificationValidationError(
            "duck problem cfg must be a mapping for verifier identity"
        )
    return _canonical_mapping(
        {
            "identity_fallback_version": "duck_problem_identity_v1",
            "declared_problem_name": getattr(problem, "name", ""),
            "problem_config": config,
        },
        "duck problem identity fallback",
    )


def _callable_identity(value: Any) -> Any:
    """Return a stable in-process identity for a function or bound method."""

    return getattr(value, "__func__", value)


@runtime_checkable
class ScientificProblemAdapter(Protocol):
    """The only problem surface visible to the common verifier service."""

    problem_id: str
    verifier_id: str
    verifier_version: str
    descriptor_version: str
    method_complete: bool
    timeout_is_scientific: bool

    def verify_answer_payload(
        self,
        payload: Any,
        policy: VerificationPolicy,
    ) -> VerificationDecision:
        """Verify the captured answer payload without proposal-code replay."""

    def describe_scientific_state(
        self,
        payload: Any,
        decision: VerificationDecision,
    ) -> Mapping[str, Any]:
        """Return deterministic scientific descriptor dimensions."""

    def scientific_fingerprint(
        self,
        payload: Any,
        decision: VerificationDecision,
    ) -> str:
        """Return a deterministic fingerprint derived from verified behavior."""


class LegacyProblemFallbackAdapter:
    """Coarse bridge for legacy/smoke fixtures; never production-complete."""

    method_complete = False
    descriptor_version = "legacy_verified_output_v1"

    def __init__(
        self,
        *,
        problem_id: str,
        verifier_version: str,
        verify_payload: Callable[[Any, VerificationPolicy], VerificationDecision],
        timeout_is_scientific: bool = False,
    ) -> None:
        self.problem_id = str(problem_id)
        self.verifier_version = str(verifier_version)
        self.timeout_is_scientific = bool(timeout_is_scientific)
        self._verify_payload = verify_payload
        self.verifier_id = content_id(
            "verifier",
            {
                "kind": "legacy_problem_fallback",
                "problem_id": self.problem_id,
                "verifier_version": self.verifier_version,
            },
        )

    def verify_answer_payload(
        self,
        payload: Any,
        policy: VerificationPolicy,
    ) -> VerificationDecision:
        return self._verify_payload(payload, policy)

    def describe_scientific_state(
        self,
        payload: Any,
        decision: VerificationDecision,
    ) -> Mapping[str, Any]:
        if isinstance(payload, Mapping):
            shape = "mapping"
            keys = sorted(str(key) for key in payload.keys())
        elif isinstance(payload, (list, tuple)):
            shape = "sequence"
            keys = []
        elif payload is None:
            shape = "null"
            keys = []
        else:
            shape = type(payload).__name__
            keys = []
        return {
            "adapter": "legacy_fallback",
            "payload_shape": shape,
            "payload_keys": keys,
            "admitted": decision.admitted,
        }

    def scientific_fingerprint(
        self,
        payload: Any,
        decision: VerificationDecision,
    ) -> str:
        return "legacy-coarse:" + content_hash(
            {
                "problem_id": self.problem_id,
                "payload": payload,
                "admitted": decision.admitted,
            }
        )


class ProblemScientificAdapter:
    """Duck-typed bridge to the additive hooks on ``problems.base.Problem``.

    The adapter deliberately does not import ``problems.base``: importing the
    neutral problem layer currently pulls in legacy sandbox dependencies.  A
    compatible object is validated by attributes and callables, then only its
    saved-payload scientific hooks are exposed to the verifier service.
    """

    def __init__(
        self,
        problem: Any,
        *,
        problem_id: Optional[str] = None,
        verifier_version: Optional[str] = None,
    ) -> None:
        required_methods = (
            "verify_answer_payload",
            "describe_scientific_state",
            "scientific_fingerprint",
            "resource_requirements",
        )
        missing = [name for name in required_methods if not callable(getattr(problem, name, None))]
        if missing:
            raise VerificationValidationError(
                "problem is missing scientific hooks: " + ", ".join(missing)
            )
        resolved_problem_id = problem_id if problem_id is not None else getattr(problem, "name", "")
        if not isinstance(resolved_problem_id, str) or not resolved_problem_id.strip():
            raise VerificationValidationError("problem_id must be non-empty")
        answer_schema_version = getattr(problem, "answer_schema_version", None)
        if (
            not isinstance(answer_schema_version, int)
            or isinstance(answer_schema_version, bool)
            or answer_schema_version < 1
        ):
            raise VerificationValidationError("problem answer_schema_version must be positive")
        method_complete = getattr(problem, "scientific_method_complete", None)
        if not isinstance(method_complete, bool):
            raise VerificationValidationError("problem scientific_method_complete must be boolean")
        descriptor_version = getattr(problem, "descriptor_function_version", "")
        fingerprint_version = getattr(problem, "fingerprint_function_version", "")
        if not isinstance(descriptor_version, str) or not descriptor_version.strip():
            raise VerificationValidationError("problem descriptor_function_version must be non-empty")
        if not isinstance(fingerprint_version, str) or not fingerprint_version.strip():
            raise VerificationValidationError("problem fingerprint_function_version must be non-empty")
        resources = problem.resource_requirements()
        resource_identity = _resource_identity(resources)
        timeout_is_scientific = getattr(resources, "timeout_is_scientific", None)
        if not isinstance(timeout_is_scientific, bool):
            raise VerificationValidationError(
                "problem resources must declare boolean timeout_is_scientific"
            )
        resolved_verifier_version = verifier_version or f"answer_schema_v{answer_schema_version}"
        if not isinstance(resolved_verifier_version, str) or not resolved_verifier_version.strip():
            raise VerificationValidationError("verifier_version must be non-empty")
        problem_identity = _problem_identity(problem)

        verifier_identity = {
            "adapter": "problems.base.scientific_hooks_v2",
            "problem_id": resolved_problem_id,
            "problem_class": (
                f"{type(problem).__module__}.{type(problem).__qualname__}"
            ),
            "answer_schema_version": answer_schema_version,
            "verifier_version": resolved_verifier_version,
            "descriptor_function_version": descriptor_version,
            "fingerprint_function_version": fingerprint_version,
            "scientific_method_complete": method_complete,
            "resource_requirements": resource_identity,
            "problem_identity": problem_identity,
        }

        self.problem = problem
        frozen_hook_names = list(required_methods)
        if callable(getattr(problem, "scientific_verifier_identity", None)):
            frozen_hook_names.append("scientific_verifier_identity")
        self._frozen_hook_objects = {
            name: _callable_identity(getattr(problem, name))
            for name in frozen_hook_names
        }
        self.problem_id = resolved_problem_id
        self.verifier_version = resolved_verifier_version
        self.descriptor_version = descriptor_version
        self.fingerprint_version = fingerprint_version
        self.method_complete = method_complete
        self.timeout_is_scientific = timeout_is_scientific
        self.answer_schema_version = answer_schema_version
        self.resource_identity = FrozenDict(resource_identity)
        self.problem_identity = FrozenDict(problem_identity)
        self.verifier_identity = FrozenDict(verifier_identity)
        self.verifier_id = content_id("verifier", verifier_identity)

    def validate_frozen_identity(self) -> None:
        """Fail closed if retained problem or adapter behavior has drifted.

        Problems remain ordinary mutable Python objects for legacy
        compatibility.  Recomputing their complete identity before every
        verifier invocation prevents a cfg/resource/version mutation from
        running new behavior under an old content-addressed verifier ID.
        """

        required_methods = (
            "verify_answer_payload",
            "describe_scientific_state",
            "scientific_fingerprint",
            "resource_requirements",
        )
        missing = [
            name for name in required_methods
            if not callable(getattr(self.problem, name, None))
        ]
        if missing:
            raise VerificationValidationError(
                "scientific adapter frozen identity drift: missing hooks "
                + ", ".join(missing)
            )
        changed_hooks = [
            name for name, frozen in self._frozen_hook_objects.items()
            if _callable_identity(getattr(self.problem, name, None)) is not frozen
        ]
        if changed_hooks:
            raise VerificationValidationError(
                "scientific adapter frozen identity drift: replaced hooks "
                + ", ".join(changed_hooks)
            )

        current_schema = getattr(self.problem, "answer_schema_version", None)
        current_descriptor = getattr(
            self.problem, "descriptor_function_version", None
        )
        current_fingerprint = getattr(
            self.problem, "fingerprint_function_version", None
        )
        current_complete = getattr(
            self.problem, "scientific_method_complete", None
        )
        resources = self.problem.resource_requirements()
        current_resources = _resource_identity(resources)
        current_timeout = getattr(resources, "timeout_is_scientific", None)
        current_problem_identity = _problem_identity(self.problem)

        cached_values = {
            "answer_schema_version": (
                self.answer_schema_version, current_schema
            ),
            "descriptor_function_version": (
                self.descriptor_version, current_descriptor
            ),
            "fingerprint_function_version": (
                self.fingerprint_version, current_fingerprint
            ),
            "scientific_method_complete": (
                self.method_complete, current_complete
            ),
            "timeout_is_scientific": (
                self.timeout_is_scientific, current_timeout
            ),
        }
        drifted = [
            name for name, (cached, current) in cached_values.items()
            if cached != current
        ]

        current_identity = {
            "adapter": "problems.base.scientific_hooks_v2",
            "problem_id": self.problem_id,
            "problem_class": (
                f"{type(self.problem).__module__}.{type(self.problem).__qualname__}"
            ),
            "answer_schema_version": current_schema,
            "verifier_version": self.verifier_version,
            "descriptor_function_version": current_descriptor,
            "fingerprint_function_version": current_fingerprint,
            "scientific_method_complete": current_complete,
            "resource_requirements": current_resources,
            "problem_identity": current_problem_identity,
        }
        current_id = content_id("verifier", current_identity)
        if (
            drifted
            or FrozenDict(current_resources) != self.resource_identity
            or FrozenDict(current_problem_identity) != self.problem_identity
            or FrozenDict(current_identity) != self.verifier_identity
            or current_id != self.verifier_id
        ):
            detail = ", ".join(drifted) if drifted else "problem/config/resources"
            raise VerificationValidationError(
                "scientific adapter frozen identity drift: " + detail
            )

    def verify_answer_payload(
        self,
        payload: Any,
        policy: VerificationPolicy,
    ) -> VerificationDecision:
        local = self.problem.verify_answer_payload(
            thaw_json(payload),
            thaw_json(policy.to_dict()),
        )
        required_fields = (
            "resolved",
            "admitted",
            "answer_payload",
            "internal_reward",
            "raw_score",
            "failure_kind",
            "message",
            "uncertainty",
            "scores",
            "flags",
            "diagnostics",
        )
        missing = [name for name in required_fields if not hasattr(local, name)]
        if missing:
            raise VerificationValidationError(
                "problem verification result is missing fields: " + ", ".join(missing)
            )
        if not isinstance(local.resolved, bool) or not isinstance(local.admitted, bool):
            raise VerificationValidationError("problem verification booleans are malformed")
        if local.admitted and local.answer_payload is None:
            raise VerificationValidationError(
                "admitted problem verification must return the exact saved answer payload"
            )
        if local.answer_payload is not None and content_hash(local.answer_payload) != content_hash(payload):
            raise VerificationValidationError(
                "problem verifier changed the persisted answer payload"
            )
        diagnostics = dict(local.diagnostics)
        if local.message:
            diagnostics.setdefault("message", str(local.message))
        flags = dict(local.flags)
        # These declarations are owned by the adapter/service and cannot be
        # self-reported differently by one result.
        reported_incomplete = flags.pop("method_incomplete", None)
        if reported_incomplete is not None:
            if not isinstance(reported_incomplete, bool) or reported_incomplete != (not self.method_complete):
                raise VerificationValidationError("problem result contradicts method completeness")
        reported_complete = flags.pop("method_complete", None)
        if reported_complete is not None:
            if not isinstance(reported_complete, bool) or reported_complete != self.method_complete:
                raise VerificationValidationError("problem result contradicts method completeness")
        capture = ExecutionCapture(diagnostics=FrozenDict(diagnostics))
        if local.admitted:
            if not local.resolved or str(local.failure_kind or "") not in ("", "none"):
                raise VerificationValidationError("admitted problem result has contradictory status")
            return VerificationDecision.success(
                internal_reward=local.internal_reward,
                raw_score=local.raw_score,
                uncertainty=local.uncertainty,
                flags=flags,
                scores=local.scores,
                capture=capture,
            )

        try:
            failure_kind = FailureKind(str(local.failure_kind))
        except (TypeError, ValueError) as exc:
            raise VerificationValidationError(
                f"problem returned unknown failure kind {local.failure_kind!r}"
            ) from exc
        classifications = {
            FailureKind.PARSE: classify_failure(parsed=False),
            FailureKind.CODE: classify_failure(executed=False),
            FailureKind.CONSTRAINT: classify_failure(constraints_satisfied=False),
            FailureKind.SCIENTIFIC: classify_failure(scientifically_valid=False),
            FailureKind.TIMEOUT: classify_failure(
                timed_out=True,
                timeout_is_scientific=self.timeout_is_scientific,
            ),
            FailureKind.INFRASTRUCTURE: classify_failure(infrastructure_error=True),
        }
        if failure_kind not in classifications:
            raise VerificationValidationError("non-admitted problem result must classify a failure")
        classification = classifications[failure_kind]
        if local.resolved != classification.resolved:
            raise VerificationValidationError(
                "problem result resolution contradicts failure/timeout policy"
            )
        return VerificationDecision.failure(
            classification,
            raw_score=local.raw_score,
            uncertainty=local.uncertainty,
            flags=flags,
            scores=local.scores,
            capture=capture,
        )

    @staticmethod
    def _evidence_view(payload: Any, decision: VerificationDecision) -> Mapping[str, Any]:
        return {
            "answer_payload": thaw_json(payload),
            "resolved": decision.resolved,
            "admitted": decision.admitted,
            "internal_reward": decision.internal_reward,
            "raw_score": thaw_json(decision.raw_score),
            "failure_kind": decision.failure_kind.value,
            "uncertainty": decision.uncertainty,
            "scores": thaw_json(decision.scores),
            "flags": thaw_json(decision.flags),
        }

    def describe_scientific_state(
        self,
        payload: Any,
        decision: VerificationDecision,
    ) -> Mapping[str, Any]:
        return self.problem.describe_scientific_state(
            thaw_json(payload),
            self._evidence_view(payload, decision),
        )

    def scientific_fingerprint(
        self,
        payload: Any,
        decision: VerificationDecision,
    ) -> str:
        return self.problem.scientific_fingerprint(
            thaw_json(payload),
            self._evidence_view(payload, decision),
        )


__all__ = [
    "LegacyProblemFallbackAdapter",
    "ProblemScientificAdapter",
    "ScientificProblemAdapter",
]
