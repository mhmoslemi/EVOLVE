"""Content-addressed construction and validation for scientific evidence."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping, Optional, Tuple

from evolve.ids import canonical_json, content_hash, content_id
from evolve.types import (
    SCHEMA_VERSION,
    Descriptor,
    EvidencePacket,
    FailureKind,
    FrozenDict,
    Proposal,
    VerifiedScientificState,
)

from .models import (
    PersistedAnswerPayload,
    VerificationDecision,
    VerificationPolicy,
    VerificationValidationError,
    thaw_json,
)


DESCRIPTOR_IDENTITY_VERSION = "scientific_descriptor_v1"
EVIDENCE_IDENTITY_VERSION = "evidence_packet_schema_v1"


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise VerificationValidationError(message)


def scientific_state_id(problem_id: str, answer_payload: Any) -> str:
    """Identify scientific state from canonical problem and answer only.

    Proposal source, proposal ID, branch, evidence, and timestamps are
    deliberately absent.  Equivalent saved answers therefore retain their
    identity across repairs, workers, and resume.
    """

    return content_id(
        "state",
        {
            "problem_id": problem_id,
            "answer_payload": thaw_json(answer_payload),
        },
    )


def descriptor_id(
    *,
    problem_id: str,
    function_version: str,
    dimensions: Mapping[str, Any],
    method_complete: bool,
) -> str:
    return content_id(
        "descriptor",
        {
            "identity_version": DESCRIPTOR_IDENTITY_VERSION,
            "problem_id": problem_id,
            "function_version": function_version,
            "dimensions": thaw_json(dimensions),
            "method_complete": method_complete,
        },
    )


def build_descriptor(
    *,
    problem_id: str,
    function_version: str,
    dimensions: Mapping[str, Any],
    method_complete: bool,
) -> Descriptor:
    frozen_dimensions = FrozenDict(dimensions)
    return Descriptor(
        descriptor_id=descriptor_id(
            problem_id=problem_id,
            function_version=function_version,
            dimensions=frozen_dimensions,
            method_complete=method_complete,
        ),
        problem_id=problem_id,
        function_version=function_version,
        dimensions=frozen_dimensions,
        method_complete=method_complete,
    )


def validate_descriptor_identity(value: Descriptor) -> None:
    expected = descriptor_id(
        problem_id=value.problem_id,
        function_version=value.function_version,
        dimensions=value.dimensions,
        method_complete=value.method_complete,
    )
    _require(value.descriptor_id == expected, "descriptor_id does not match descriptor content")


def _evidence_identity_document_from_fields(fields: Mapping[str, Any]) -> Mapping[str, Any]:
    """Return exactly the persisted packet document except its own ID."""

    document = {
        "record_type": EvidencePacket.RECORD_TYPE,
        "schema_version": SCHEMA_VERSION,
        "extensions": {},
    }
    document.update({key: thaw_json(value) for key, value in fields.items()})
    document.pop("evidence_id", None)
    # Make the identity format explicit even though the schema-v1 record shape
    # is already covered.  It is stored in flags by the builder and therefore
    # remains visible to old schema-aware readers.
    flags = document.get("flags")
    _require(isinstance(flags, Mapping), "evidence flags must be a mapping")
    _require(
        flags.get("evidence_identity_version") == EVIDENCE_IDENTITY_VERSION,
        "evidence identity version is missing or unsupported",
    )
    return document


def evidence_id_from_fields(fields: Mapping[str, Any]) -> str:
    return content_id("evidence", _evidence_identity_document_from_fields(fields))


def validate_evidence_identity(packet: EvidencePacket) -> None:
    document = packet.to_dict()
    actual = document.pop("evidence_id")
    expected = content_id("evidence", _evidence_identity_document_from_fields(document))
    _require(actual == expected, "evidence_id does not match complete packet content")


def validate_state_identity(state: VerifiedScientificState) -> None:
    expected = scientific_state_id(state.problem_id, state.answer_payload)
    _require(state.state_id == expected, "state_id does not match problem and answer payload")


def bounded_diagnostics(
    diagnostics: Mapping[str, Any],
    policy: VerificationPolicy,
) -> FrozenDict:
    """Bound verifier diagnostics while retaining a hash of the full capture."""

    detached = thaw_json(FrozenDict(diagnostics))
    encoded = canonical_json(detached)
    ordered_keys = sorted(detached)
    if (
        len(ordered_keys) <= policy.max_diagnostic_entries
        and len(encoded) <= policy.max_diagnostic_chars
    ):
        return FrozenDict(detached)

    retained = {
        key: detached[key]
        for key in ordered_keys[: policy.max_diagnostic_entries]
    }
    metadata = {
        "truncated": True,
        "original_sha256": content_hash(detached),
        "original_chars": len(encoded),
        "original_entries": len(ordered_keys),
        "retained_entries": len(retained),
    }
    metadata["entries"] = retained
    candidate = {"_bounded": metadata}
    if len(canonical_json(candidate)) <= policy.max_diagnostic_chars:
        return FrozenDict(candidate)

    # Large retained values are replaced by a canonical preview.  Binary
    # search accounts for JSON escaping, so the persisted envelope is always
    # within the exact character bound.
    metadata = dict(metadata)
    metadata["retained_entries"] = 0
    metadata.pop("entries", None)

    def envelope(preview: str) -> Mapping[str, Any]:
        return {"_bounded": {**metadata, "preview": preview}}

    base = envelope("")
    _require(
        len(canonical_json(base)) <= policy.max_diagnostic_chars,
        "max_diagnostic_chars is too small for truncation metadata",
    )
    low, high = 0, len(encoded)
    while low < high:
        midpoint = (low + high + 1) // 2
        if len(canonical_json(envelope(encoded[:midpoint]))) <= policy.max_diagnostic_chars:
            low = midpoint
        else:
            high = midpoint - 1
    return FrozenDict(envelope(encoded[:low]))


def _lineage_ids(proposal: Proposal, state_id: Optional[str]) -> Tuple[str, ...]:
    values = []
    for value in (proposal.parent_state_id, state_id):
        if value is not None and value not in values:
            values.append(value)
    return tuple(values)


@dataclass(frozen=True)
class ScientificVerificationResult:
    """One durable common-verifier result and its admitted scientific state."""

    decision: VerificationDecision
    evidence: EvidencePacket
    state: Optional[VerifiedScientificState]
    descriptor: Optional[Descriptor]

    def __post_init__(self) -> None:
        _require(isinstance(self.decision, VerificationDecision), "decision has the wrong type")
        _require(isinstance(self.evidence, EvidencePacket), "evidence has the wrong type")
        validate_evidence_identity(self.evidence)
        _require(
            self.evidence.failure_kind == self.decision.failure_kind,
            "evidence failure kind must match its decision",
        )
        _require(self.evidence.resolved == self.decision.resolved, "evidence resolution mismatch")
        _require(self.evidence.admitted == self.decision.admitted, "evidence admission mismatch")
        _require(
            self.evidence.internal_reward == self.decision.internal_reward,
            "evidence reward mismatch",
        )
        if self.evidence.admitted:
            _require(self.state is not None, "admitted evidence requires a scientific state")
            _require(self.descriptor is not None, "admitted evidence requires a descriptor")
        else:
            _require(self.state is None, "non-admitted evidence cannot publish a scientific state")
            _require(self.descriptor is None, "non-admitted evidence cannot publish a descriptor")
        if self.state is not None:
            validate_state_identity(self.state)
            _require(self.state.evidence_id == self.evidence.evidence_id, "state/evidence reference mismatch")
            _require(
                self.state.state_id == self.evidence.scientific_state_id,
                "evidence/state identity mismatch",
            )
            _require(self.state.proposal_id == self.evidence.proposal_id, "state proposal mismatch")
            _require(self.state.confirmed == self.evidence.confirmed, "confirmation mismatch")
        if self.descriptor is not None:
            validate_descriptor_identity(self.descriptor)
            _require(
                self.descriptor.descriptor_id == self.evidence.descriptor_id,
                "evidence/descriptor reference mismatch",
            )


def build_verification_result(
    *,
    proposal: Proposal,
    persisted_answer: PersistedAnswerPayload,
    decision: VerificationDecision,
    verification_policy: VerificationPolicy,
    verifier_id: str,
    verifier_version: str,
    harness_id: str,
    policy_snapshot_id: str,
    timeout_is_scientific: bool,
    method_complete: bool,
    descriptor: Optional[Descriptor] = None,
    fingerprint: str = "",
    confirmed: bool = False,
    extra_flags: Optional[Mapping[str, Any]] = None,
) -> ScientificVerificationResult:
    """Construct one packet atomically from already captured verifier output."""

    _require(proposal.branch_id is not None, "proposal must reference a frozen branch")
    _require(proposal.problem_id == persisted_answer.problem_id, "proposal/payload problem mismatch")
    _require(isinstance(confirmed, bool), "confirmed must be boolean")
    _require(not confirmed or decision.admitted, "only admitted evidence can be confirmed")
    if decision.admitted:
        _require(descriptor is not None, "admitted result requires a descriptor")
        validate_descriptor_identity(descriptor)
        _require(descriptor.problem_id == proposal.problem_id, "descriptor problem mismatch")
        _require(bool(fingerprint.strip()), "admitted result requires a scientific fingerprint")
    else:
        _require(descriptor is None, "failed result cannot carry a descriptor")
        _require(fingerprint == "", "failed result cannot carry a fingerprint")

    state_id = (
        scientific_state_id(proposal.problem_id, persisted_answer.payload)
        if decision.admitted
        else None
    )
    reserved_flags = {
        "answer_artifact_uri": persisted_answer.artifact_uri,
        "answer_payload_hash": persisted_answer.payload_hash,
        "answer_payload_id": persisted_answer.payload_id,
        "diagnostic_policy_version": verification_policy.diagnostic_policy_version,
        "evidence_identity_version": EVIDENCE_IDENTITY_VERSION,
        "excluded_from_scientific_updates": (
            decision.failure_kind == FailureKind.INFRASTRUCTURE
            or (
                decision.failure_kind == FailureKind.TIMEOUT
                and not timeout_is_scientific
            )
        ),
        "method_incomplete": not method_complete,
        "verification_attempt_index": decision.capture.attempt_index,
        "verification_policy_id": verification_policy.policy_id,
        "verification_policy_version": verification_policy.version,
    }
    if descriptor is not None:
        reserved_flags["descriptor_version"] = descriptor.function_version
    decision_flags = thaw_json(decision.flags)
    supplied_flags = dict(extra_flags or {})
    collisions = sorted((set(decision_flags) | set(supplied_flags)) & set(reserved_flags))
    _require(not collisions, "reserved evidence flags cannot be overridden: " + ", ".join(collisions))
    overlap = sorted(set(decision_flags) & set(supplied_flags))
    _require(not overlap, "duplicate evidence flags: " + ", ".join(overlap))
    flags = {**decision_flags, **supplied_flags, **reserved_flags}

    resources = thaw_json(decision.capture.resources)
    resources.setdefault("verifier_calls", 1)
    packet_fields = {
        "run_id": proposal.run_id,
        "proposal_id": proposal.proposal_id,
        "scientific_state_id": state_id,
        "parent_state_id": proposal.parent_state_id,
        "branch_id": proposal.branch_id,
        "problem_id": proposal.problem_id,
        "verifier_id": verifier_id,
        "verifier_version": verifier_version,
        "harness_id": harness_id,
        "policy_snapshot_id": policy_snapshot_id,
        "lineage_ids": _lineage_ids(proposal, state_id),
        "resolved": decision.resolved,
        "admitted": decision.admitted,
        "confirmed": confirmed,
        "failure_kind": decision.failure_kind,
        "internal_reward": decision.internal_reward,
        "raw_score": thaw_json(decision.raw_score),
        "uncertainty": decision.uncertainty,
        "descriptor_id": descriptor.descriptor_id if descriptor is not None else None,
        "fingerprint": fingerprint,
        "source_hash": proposal.source_hash,
        "flags": flags,
        "scores": thaw_json(decision.scores),
        "diagnostics": thaw_json(bounded_diagnostics(decision.capture.diagnostics, verification_policy)),
        "resources": resources,
        "answer_payload": thaw_json(persisted_answer.payload),
        "timeout_is_scientific": timeout_is_scientific,
        "started_at": decision.capture.started_at,
        "completed_at": decision.capture.completed_at,
    }
    packet = EvidencePacket(
        evidence_id=evidence_id_from_fields(packet_fields),
        **packet_fields,
    )
    state = None
    if decision.admitted:
        state = VerifiedScientificState(
            state_id=state_id,
            proposal_id=proposal.proposal_id,
            evidence_id=packet.evidence_id,
            problem_id=proposal.problem_id,
            answer_payload=thaw_json(persisted_answer.payload),
            resolved=True,
            admitted=True,
            confirmed=confirmed,
            internal_reward=decision.internal_reward,
            raw_score=thaw_json(decision.raw_score),
            descriptor_id=descriptor.descriptor_id,
            fingerprint=fingerprint,
        )
    return ScientificVerificationResult(
        decision=decision,
        evidence=packet,
        state=state,
        descriptor=descriptor,
    )


__all__ = [
    "DESCRIPTOR_IDENTITY_VERSION",
    "EVIDENCE_IDENTITY_VERSION",
    "ScientificVerificationResult",
    "bounded_diagnostics",
    "build_descriptor",
    "build_verification_result",
    "descriptor_id",
    "evidence_id_from_fields",
    "scientific_state_id",
    "validate_descriptor_identity",
    "validate_evidence_identity",
    "validate_state_identity",
]
