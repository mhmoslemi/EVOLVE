"""Independent saved-payload verification and record confirmation service."""

from __future__ import annotations

from dataclasses import replace
from datetime import datetime, timezone
import time
from typing import Any, Mapping, Optional, Tuple

from evolve.ids import content_hash, validate_id
from evolve.types import Descriptor, EvidencePacket, FailureKind, FrozenDict, Proposal

from .adapters import ScientificProblemAdapter
from .evidence import (
    ScientificVerificationResult,
    build_descriptor,
    build_verification_result,
    scientific_state_id,
    validate_evidence_identity,
)
from .models import (
    ExecutionCapture,
    PersistedAnswerPayload,
    VerificationDecision,
    VerificationPolicy,
    VerificationValidationError,
    classify_failure,
    thaw_json,
)


class VerificationServiceError(VerificationValidationError):
    """The requested verification has inconsistent frozen references."""


_RESERVED_EVIDENCE_FLAGS = frozenset(
    {
        "answer_artifact_uri",
        "answer_payload_hash",
        "answer_payload_id",
        "descriptor_version",
        "diagnostic_policy_version",
        "evidence_identity_version",
        "excluded_from_scientific_updates",
        "method_incomplete",
        "verification_attempt_index",
        "verification_policy_id",
        "verification_policy_version",
    }
)


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise VerificationServiceError(message)


def _validate_adapter(
    adapter: ScientificProblemAdapter,
    *,
    problem_id: str,
    policy: VerificationPolicy,
) -> None:
    required_methods = (
        "verify_answer_payload",
        "describe_scientific_state",
        "scientific_fingerprint",
    )
    missing = [name for name in required_methods if not callable(getattr(adapter, name, None))]
    _require(not missing, "scientific adapter is missing: " + ", ".join(missing))
    frozen_validator = getattr(adapter, "validate_frozen_identity", None)
    if callable(frozen_validator):
        try:
            frozen_validator()
        except Exception as exc:
            raise VerificationServiceError(
                "scientific adapter frozen identity validation failed: "
                f"{type(exc).__name__}: {exc}"
            ) from exc
    _require(getattr(adapter, "problem_id", None) == problem_id, "adapter problem mismatch")
    for name in ("verifier_version", "descriptor_version"):
        value = getattr(adapter, name, None)
        _require(isinstance(value, str) and bool(value.strip()), f"adapter {name} must be non-empty")
    try:
        validate_id(getattr(adapter, "verifier_id", ""), "verifier")
    except (TypeError, ValueError) as exc:
        raise VerificationServiceError(f"invalid adapter verifier_id: {exc}") from exc
    method_complete = getattr(adapter, "method_complete", None)
    timeout_is_scientific = getattr(adapter, "timeout_is_scientific", None)
    _require(isinstance(method_complete, bool), "adapter method_complete must be boolean")
    _require(
        isinstance(timeout_is_scientific, bool),
        "adapter timeout_is_scientific must be boolean",
    )
    _require(
        not (policy.production and not method_complete),
        "method-incomplete problem adapters are prohibited in production EVOLVE",
    )


def _validate_context(
    *,
    adapter: ScientificProblemAdapter,
    proposal: Proposal,
    persisted_answer: PersistedAnswerPayload,
    verification_policy: VerificationPolicy,
    harness_id: str,
    policy_snapshot_id: str,
) -> None:
    _require(isinstance(proposal, Proposal), "proposal must be a Proposal")
    _require(
        isinstance(persisted_answer, PersistedAnswerPayload),
        "persisted_answer must be PersistedAnswerPayload",
    )
    try:
        persisted_answer.validate_durable_artifact()
    except VerificationValidationError as exc:
        raise VerificationServiceError(
            f"durable answer artifact validation failed: {exc}"
        ) from exc
    _require(
        isinstance(verification_policy, VerificationPolicy),
        "verification_policy must be VerificationPolicy",
    )
    _require(proposal.branch_id is not None, "proposal must reference a frozen branch")
    _require(
        proposal.problem_id == persisted_answer.problem_id,
        "proposal and persisted payload problem IDs differ",
    )
    try:
        validate_id(harness_id, "harness")
        validate_id(policy_snapshot_id, "role_snapshot")
    except (TypeError, ValueError) as exc:
        raise VerificationServiceError(f"invalid frozen verification reference: {exc}") from exc
    _validate_adapter(adapter, problem_id=proposal.problem_id, policy=verification_policy)


def _validate_decision(
    decision: VerificationDecision,
    adapter: ScientificProblemAdapter,
) -> None:
    if not isinstance(decision, VerificationDecision):
        raise VerificationValidationError("adapter must return VerificationDecision")
    expected_resolved = {
        FailureKind.NONE: True,
        FailureKind.PARSE: True,
        FailureKind.CODE: True,
        FailureKind.CONSTRAINT: True,
        FailureKind.SCIENTIFIC: True,
        FailureKind.TIMEOUT: adapter.timeout_is_scientific,
        FailureKind.INFRASTRUCTURE: False,
    }[decision.failure_kind]
    if decision.resolved != expected_resolved:
        raise VerificationValidationError(
            "decision resolution contradicts failure kind or problem timeout policy"
        )
    collisions = sorted(set(decision.flags) & _RESERVED_EVIDENCE_FLAGS)
    if collisions:
        raise VerificationValidationError(
            "adapter decision overrides service-owned flags: " + ", ".join(collisions)
        )


def _infrastructure_decision(
    exc: Exception,
    *,
    phase: str,
    prior_capture: Optional[ExecutionCapture] = None,
) -> VerificationDecision:
    prior = prior_capture or ExecutionCapture()
    diagnostics = {
        "adapter_error": {
            "exception_type": type(exc).__name__,
            "message": str(exc),
            "phase": phase,
        }
    }
    if prior.diagnostics:
        diagnostics["prior_verifier_diagnostics"] = thaw_json(prior.diagnostics)
    resources = thaw_json(prior.resources)
    resources.setdefault("verifier_calls", 1)
    capture = ExecutionCapture(
        diagnostics=FrozenDict(diagnostics),
        resources=FrozenDict(resources),
        started_at=prior.started_at,
        completed_at=prior.completed_at,
        attempt_index=prior.attempt_index,
    )
    return VerificationDecision.failure(
        classify_failure(infrastructure_error=True),
        flags={"adapter_contract_phase": phase},
        capture=capture,
    )


def _utc_timestamp() -> str:
    return (
        datetime.now(timezone.utc)
        .isoformat(timespec="microseconds")
        .replace("+00:00", "Z")
    )


def _with_service_capture(
    decision: VerificationDecision,
    *,
    service_started_at: str,
    service_completed_at: str,
    verifier_wall_time_s: float,
) -> VerificationDecision:
    """Fill absent capture fields without replacing explicit verifier data."""

    capture = decision.capture
    missing_timestamps = not capture.started_at or not capture.completed_at
    missing_resources = not capture.resources
    if not missing_timestamps and not missing_resources:
        return decision

    resources = thaw_json(capture.resources)
    resources.setdefault("verifier_calls", 1)
    resources.setdefault(
        "verifier_wall_time_s", max(0.0, float(verifier_wall_time_s))
    )
    completed_capture = ExecutionCapture(
        diagnostics=capture.diagnostics,
        resources=FrozenDict(resources),
        started_at=capture.started_at or service_started_at,
        completed_at=capture.completed_at or service_completed_at,
        attempt_index=capture.attempt_index,
    )
    return replace(decision, capture=completed_capture)


def _invoke_adapter(
    *,
    adapter: ScientificProblemAdapter,
    persisted_answer: PersistedAnswerPayload,
    verification_policy: VerificationPolicy,
) -> VerificationDecision:
    service_started_at = _utc_timestamp()
    monotonic_started = time.perf_counter()
    try:
        # A detached copy of the durable answer is the adapter's only candidate
        # input.  Proposal source and parsed code are intentionally unavailable.
        decision = adapter.verify_answer_payload(
            thaw_json(persisted_answer.payload),
            verification_policy,
        )
        _validate_decision(decision, adapter)
        if decision.admitted and persisted_answer.payload is None:
            raise VerificationValidationError(
                "an admitted decision cannot identify a null scientific payload"
            )
    except Exception as exc:
        capture = getattr(locals().get("decision"), "capture", None)
        decision = _infrastructure_decision(
            exc, phase="verify_answer_payload", prior_capture=capture
        )
    service_completed_at = _utc_timestamp()
    elapsed = max(0.0, time.perf_counter() - monotonic_started)
    return _with_service_capture(
        decision,
        service_started_at=service_started_at,
        service_completed_at=service_completed_at,
        verifier_wall_time_s=elapsed,
    )


def _descriptor_and_fingerprint(
    *,
    adapter: ScientificProblemAdapter,
    persisted_answer: PersistedAnswerPayload,
    decision: VerificationDecision,
) -> Tuple[Optional[Descriptor], str, VerificationDecision]:
    if not decision.admitted:
        return None, "", decision
    try:
        dimensions = adapter.describe_scientific_state(
            thaw_json(persisted_answer.payload),
            decision,
        )
        if not isinstance(dimensions, Mapping) or not dimensions:
            raise VerificationValidationError("scientific descriptor must be a non-empty mapping")
        fingerprint = adapter.scientific_fingerprint(
            thaw_json(persisted_answer.payload),
            decision,
        )
        if not isinstance(fingerprint, str) or not fingerprint.strip():
            raise VerificationValidationError("scientific fingerprint must be non-empty")
        descriptor = build_descriptor(
            problem_id=adapter.problem_id,
            function_version=adapter.descriptor_version,
            dimensions=dimensions,
            method_complete=adapter.method_complete,
        )
        return descriptor, fingerprint, decision
    except Exception as exc:
        return (
            None,
            "",
            _infrastructure_decision(
                exc,
                phase="describe_verified_state",
                prior_capture=decision.capture,
            ),
        )


def _verify(
    *,
    adapter: ScientificProblemAdapter,
    proposal: Proposal,
    persisted_answer: PersistedAnswerPayload,
    verification_policy: VerificationPolicy,
    harness_id: str,
    policy_snapshot_id: str,
    confirmation: bool,
    extra_flags: Optional[Mapping[str, Any]] = None,
    expected_descriptor_id: Optional[str] = None,
    expected_fingerprint: Optional[str] = None,
    attempt_index: int = 0,
) -> ScientificVerificationResult:
    if (
        isinstance(attempt_index, bool)
        or not isinstance(attempt_index, int)
        or attempt_index < 0
    ):
        raise VerificationServiceError(
            "verification attempt_index must be a non-negative integer"
        )
    decision = _invoke_adapter(
        adapter=adapter,
        persisted_answer=persisted_answer,
        verification_policy=verification_policy,
    )
    # Preserve an adapter's immutable capture object when the service does not
    # need to add or change anything.  Besides avoiding needless copies, this
    # makes the exact verifier-owned capture available to downstream durable
    # persistence.  Retry orchestration still replaces the capture when it
    # supplies a different durable attempt index.
    if decision.capture.attempt_index != attempt_index:
        decision = replace(
            decision,
            capture=replace(decision.capture, attempt_index=attempt_index),
        )
    descriptor, fingerprint, decision = _descriptor_and_fingerprint(
        adapter=adapter,
        persisted_answer=persisted_answer,
        decision=decision,
    )
    if decision.admitted and expected_descriptor_id is not None:
        if descriptor is None or descriptor.descriptor_id != expected_descriptor_id:
            decision = _infrastructure_decision(
                VerificationValidationError("confirmation descriptor changed for the same payload"),
                phase="confirm_descriptor_identity",
                prior_capture=decision.capture,
            )
            descriptor, fingerprint = None, ""
    if decision.admitted and expected_fingerprint is not None:
        if fingerprint != expected_fingerprint:
            decision = _infrastructure_decision(
                VerificationValidationError("confirmation fingerprint changed for the same payload"),
                phase="confirm_fingerprint_identity",
                prior_capture=decision.capture,
            )
            descriptor, fingerprint = None, ""
    return build_verification_result(
        proposal=proposal,
        persisted_answer=persisted_answer,
        decision=decision,
        verification_policy=verification_policy,
        verifier_id=adapter.verifier_id,
        verifier_version=adapter.verifier_version,
        harness_id=harness_id,
        policy_snapshot_id=policy_snapshot_id,
        timeout_is_scientific=adapter.timeout_is_scientific,
        method_complete=adapter.method_complete,
        descriptor=descriptor,
        fingerprint=fingerprint,
        confirmed=confirmation and decision.admitted,
        extra_flags=extra_flags,
    )


def verify_persisted_answer(
    *,
    adapter: ScientificProblemAdapter,
    proposal: Proposal,
    persisted_answer: PersistedAnswerPayload,
    verification_policy: VerificationPolicy,
    harness_id: str,
    policy_snapshot_id: str,
    attempt_index: int = 0,
) -> ScientificVerificationResult:
    """Verify one saved answer and create observation-specific evidence."""

    _validate_context(
        adapter=adapter,
        proposal=proposal,
        persisted_answer=persisted_answer,
        verification_policy=verification_policy,
        harness_id=harness_id,
        policy_snapshot_id=policy_snapshot_id,
    )
    return _verify(
        adapter=adapter,
        proposal=proposal,
        persisted_answer=persisted_answer,
        verification_policy=verification_policy,
        harness_id=harness_id,
        policy_snapshot_id=policy_snapshot_id,
        confirmation=False,
        attempt_index=attempt_index,
    )


def _validate_confirmation_source(
    *,
    adapter: ScientificProblemAdapter,
    proposal: Proposal,
    persisted_answer: PersistedAnswerPayload,
    prior_evidence: EvidencePacket,
) -> None:
    _require(isinstance(prior_evidence, EvidencePacket), "prior_evidence must be EvidencePacket")
    validate_evidence_identity(prior_evidence)
    exact = {
        "run_id": proposal.run_id,
        "proposal_id": proposal.proposal_id,
        "problem_id": proposal.problem_id,
        "parent_state_id": proposal.parent_state_id,
        "branch_id": proposal.branch_id,
        "source_hash": proposal.source_hash,
        "verifier_id": adapter.verifier_id,
        "verifier_version": adapter.verifier_version,
        "timeout_is_scientific": adapter.timeout_is_scientific,
    }
    for field_name, expected in exact.items():
        _require(
            getattr(prior_evidence, field_name) == expected,
            f"prior evidence {field_name} does not match the frozen confirmation context",
        )
    _require(prior_evidence.resolved, "only resolved evidence can request record confirmation")
    _require(prior_evidence.admitted, "only admitted evidence can request record confirmation")
    _require(
        prior_evidence.failure_kind == FailureKind.NONE,
        "record confirmation source cannot carry a failure",
    )
    expected_state_id = scientific_state_id(proposal.problem_id, persisted_answer.payload)
    _require(
        prior_evidence.scientific_state_id == expected_state_id,
        "prior evidence points to a different scientific state",
    )
    _require(
        content_hash(prior_evidence.answer_payload) == content_hash(persisted_answer.payload),
        "prior evidence payload differs from the persisted confirmation payload",
    )
    expected_flags = {
        "answer_payload_hash": persisted_answer.payload_hash,
        "answer_payload_id": persisted_answer.payload_id,
        "method_incomplete": not adapter.method_complete,
    }
    for flag_name, expected in expected_flags.items():
        _require(
            prior_evidence.flags.get(flag_name) == expected,
            f"prior evidence {flag_name} does not match the persisted payload",
        )
    _require(prior_evidence.descriptor_id is not None, "admitted evidence has no descriptor")
    _require(bool(prior_evidence.fingerprint), "admitted evidence has no fingerprint")


def confirm_persisted_answer(
    *,
    adapter: ScientificProblemAdapter,
    proposal: Proposal,
    persisted_answer: PersistedAnswerPayload,
    prior_evidence: EvidencePacket,
    verification_policy: VerificationPolicy,
    attempt_index: int = 0,
) -> ScientificVerificationResult:
    """Reverify the exact saved payload; proposal code is never replayed."""

    harness_id = getattr(prior_evidence, "harness_id", "")
    policy_snapshot_id = getattr(prior_evidence, "policy_snapshot_id", "")
    _validate_context(
        adapter=adapter,
        proposal=proposal,
        persisted_answer=persisted_answer,
        verification_policy=verification_policy,
        harness_id=harness_id,
        policy_snapshot_id=policy_snapshot_id,
    )
    _validate_confirmation_source(
        adapter=adapter,
        proposal=proposal,
        persisted_answer=persisted_answer,
        prior_evidence=prior_evidence,
    )
    return _verify(
        adapter=adapter,
        proposal=proposal,
        persisted_answer=persisted_answer,
        verification_policy=verification_policy,
        harness_id=prior_evidence.harness_id,
        policy_snapshot_id=prior_evidence.policy_snapshot_id,
        confirmation=True,
        extra_flags={
            "confirmation_of_evidence_id": prior_evidence.evidence_id,
            "confirmation_target_state_id": prior_evidence.scientific_state_id,
        },
        expected_descriptor_id=prior_evidence.descriptor_id,
        expected_fingerprint=prior_evidence.fingerprint,
        attempt_index=attempt_index,
    )


__all__ = [
    "VerificationServiceError",
    "confirm_persisted_answer",
    "verify_persisted_answer",
]
