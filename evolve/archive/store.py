"""Collision-safe in-memory scientific artifact registry.

The registry is functional and append-only: every operation returns either the
same value for an idempotent retry or a new registry.  Scientific-state identity
intentionally excludes proposal source, harness, policy, worker, and timing
metadata.  Different programs that serialize to the same independently verified
answer therefore address the same scientific state while retaining both proposal
and evidence artifacts.
"""

from __future__ import annotations

from dataclasses import dataclass, field, replace
from typing import Any, Optional, Tuple

from evolve.ids import content_hash, content_id
from evolve.types import (
    EvidencePacket,
    FailureKind,
    Proposal,
    VerifiedScientificState,
)
from evolve.verifier.evidence import validate_evidence_identity
from evolve.verifier.models import VerificationValidationError


class ArtifactStoreError(ValueError):
    """Base error for malformed or contradictory scientific artifacts."""


class ArtifactCollisionError(ArtifactStoreError):
    """A durable identifier was reused for different immutable content."""


class ArtifactReferenceError(ArtifactStoreError):
    """An artifact references a missing or mismatched durable object."""


class ScientificIdentityError(ArtifactStoreError):
    """A state ID does not match its source-independent scientific content."""


def derive_scientific_state_id(
    *,
    problem_id: str,
    answer_payload: Any,
) -> str:
    """Derive state identity solely from the captured scientific answer.

    Verification observations may change confirmation, reward, uncertainty and
    scores while continuing to refer to this state.  Proposal/source, verifier,
    harness, policy, run/branch, diagnostics, resources and timestamps are also
    deliberately absent.
    """

    if not isinstance(problem_id, str) or not problem_id.strip():
        raise ScientificIdentityError("problem_id must be non-empty")
    payload = {
        "problem_id": problem_id,
        "answer_payload": answer_payload,
    }
    return content_id("state", payload)


def scientific_state_id_from_evidence(evidence: EvidencePacket) -> str:
    if not evidence.resolved or not evidence.admitted:
        raise ScientificIdentityError("scientific states require resolved admitted evidence")
    if evidence.failure_kind != FailureKind.NONE or evidence.internal_reward is None:
        raise ScientificIdentityError("scientific state evidence cannot carry a failure")
    return derive_scientific_state_id(
        problem_id=evidence.problem_id,
        answer_payload=evidence.answer_payload,
    )


def validate_stored_evidence(evidence: EvidencePacket) -> None:
    """Validate content identity and whether a packet may bind a state.

    Every verification attempt is durable evidence, including resolved
    scientific rejection and unresolved infrastructure failure.  Only admitted
    packets address a scientific state or carry descriptor/fingerprint fields.
    """

    try:
        validate_evidence_identity(evidence)
    except VerificationValidationError as exc:
        raise ArtifactCollisionError(str(exc)) from exc
    if evidence.admitted:
        expected_state_id = scientific_state_id_from_evidence(evidence)
        if evidence.scientific_state_id != expected_state_id:
            raise ScientificIdentityError(
                "evidence scientific_state_id is not its verified scientific identity"
            )
        return
    if evidence.scientific_state_id is not None:
        raise ScientificIdentityError(
            "non-admitted evidence cannot address a scientific state"
        )
    if evidence.confirmed:
        raise ArtifactReferenceError("non-admitted evidence cannot be confirmed")
    if evidence.descriptor_id is not None or evidence.fingerprint:
        raise ArtifactReferenceError(
            "non-admitted evidence cannot carry a descriptor or fingerprint"
        )


def _json_equal(left: Any, right: Any) -> bool:
    return content_hash(left) == content_hash(right)


def validate_state_evidence(
    state: VerifiedScientificState,
    evidence: EvidencePacket,
    *,
    require_descriptor: bool = False,
    require_fingerprint: bool = False,
) -> None:
    """Validate exact state/evidence references and scientific values."""

    if state.evidence_id != evidence.evidence_id:
        raise ArtifactReferenceError("state evidence_id does not reference this evidence")
    if state.proposal_id != evidence.proposal_id:
        raise ArtifactReferenceError("state/evidence proposal references disagree")
    if evidence.scientific_state_id != state.state_id:
        raise ArtifactReferenceError("evidence scientific_state_id does not reference this state")
    if state.problem_id != evidence.problem_id:
        raise ArtifactReferenceError("state/evidence problem IDs disagree")
    if not state.resolved or not state.admitted or not evidence.resolved or not evidence.admitted:
        raise ArtifactReferenceError("archive scientific states must be resolved and admitted")
    if (state.resolved, state.admitted, state.confirmed) != (
        evidence.resolved,
        evidence.admitted,
        evidence.confirmed,
    ):
        raise ArtifactReferenceError("state/evidence verification flags disagree")
    if state.internal_reward != evidence.internal_reward:
        raise ArtifactReferenceError("state/evidence internal rewards disagree")
    if not _json_equal(state.raw_score, evidence.raw_score):
        raise ArtifactReferenceError("state/evidence native raw_score values disagree")
    if not _json_equal(state.answer_payload, evidence.answer_payload):
        raise ArtifactReferenceError("state/evidence captured answer payloads disagree")
    if state.descriptor_id != evidence.descriptor_id:
        raise ArtifactReferenceError("state/evidence descriptor references disagree")
    if state.fingerprint != evidence.fingerprint:
        raise ArtifactReferenceError("state/evidence fingerprints disagree")
    if require_descriptor and state.descriptor_id is None:
        raise ArtifactReferenceError("archive admission requires a descriptor")
    if require_fingerprint and not state.fingerprint:
        raise ArtifactReferenceError("archive admission requires a verified fingerprint")
    expected_state_id = scientific_state_id_from_evidence(evidence)
    if state.state_id != expected_state_id:
        raise ScientificIdentityError(
            "state_id is not derived from its source-independent verified content"
        )


@dataclass(frozen=True)
class ScientificArtifactStore:
    """Append-only proposals, evidence, and source-invariant state bindings."""

    proposals: Tuple[Proposal, ...] = field(default_factory=tuple)
    evidence: Tuple[EvidencePacket, ...] = field(default_factory=tuple)
    states: Tuple[VerifiedScientificState, ...] = field(default_factory=tuple)

    def __post_init__(self) -> None:
        proposal_ids = set()
        for proposal in self.proposals:
            if proposal.proposal_id in proposal_ids:
                raise ArtifactCollisionError(
                    f"duplicate proposal_id {proposal.proposal_id} in store state"
                )
            proposal_ids.add(proposal.proposal_id)

        evidence_ids = set()
        for packet in self.evidence:
            if packet.evidence_id in evidence_ids:
                raise ArtifactCollisionError(
                    f"duplicate evidence_id {packet.evidence_id} in store state"
                )
            evidence_ids.add(packet.evidence_id)
            validate_stored_evidence(packet)
            proposal = self.proposal(packet.proposal_id)
            self._validate_proposal_evidence(proposal, packet)

        binding_keys = set()
        identities = {}
        for state in self.states:
            binding_key = (state.state_id, state.proposal_id, state.evidence_id)
            if binding_key in binding_keys:
                raise ArtifactCollisionError(
                    f"duplicate state binding for {state.state_id} in store state"
                )
            binding_keys.add(binding_key)
            identity = (state.problem_id, content_hash(state.answer_payload))
            previous = identities.get(state.state_id)
            if previous is not None and previous != identity:
                raise ArtifactCollisionError(
                    f"scientific state collision for {state.state_id}"
                )
            identities[state.state_id] = identity
            packet = self.evidence_packet(state.evidence_id)
            validate_state_evidence(state, packet)
            proposal = self.proposal(state.proposal_id)
            self._validate_proposal_evidence(proposal, packet)

    @staticmethod
    def _validate_proposal_evidence(
        proposal: Proposal, evidence: EvidencePacket
    ) -> None:
        if proposal.proposal_id != evidence.proposal_id:
            raise ArtifactReferenceError("proposal/evidence references disagree")
        if proposal.run_id != evidence.run_id or proposal.problem_id != evidence.problem_id:
            raise ArtifactReferenceError("proposal/evidence run or problem references disagree")
        if proposal.source_hash != evidence.source_hash:
            raise ArtifactReferenceError("evidence source_hash does not match its proposal")
        if proposal.branch_id != evidence.branch_id:
            raise ArtifactReferenceError("proposal/evidence branch references disagree")
        if proposal.parent_state_id != evidence.parent_state_id:
            raise ArtifactReferenceError("proposal/evidence parent references disagree")

    def proposal(self, proposal_id: str) -> Proposal:
        for proposal in self.proposals:
            if proposal.proposal_id == proposal_id:
                return proposal
        raise ArtifactReferenceError(f"unknown proposal_id {proposal_id!r}")

    def evidence_packet(self, evidence_id: str) -> EvidencePacket:
        for packet in self.evidence:
            if packet.evidence_id == evidence_id:
                return packet
        raise ArtifactReferenceError(f"unknown evidence_id {evidence_id!r}")

    def state_observations(self, state_id: str) -> Tuple[VerifiedScientificState, ...]:
        return tuple(state for state in self.states if state.state_id == state_id)

    def representative_state(
        self,
        state_id: str,
        *,
        descriptor_id: Optional[str] = None,
    ) -> VerifiedScientificState:
        observations = self.state_observations(state_id)
        if descriptor_id is not None:
            observations = tuple(
                state
                for state in observations
                if state.descriptor_id == descriptor_id
            )
        if not observations:
            raise ArtifactReferenceError(f"unknown state_id {state_id!r}")
        return max(
            observations,
            key=lambda state: (
                state.confirmed,
                self.evidence_packet(state.evidence_id).completed_at,
                state.evidence_id,
                state.proposal_id,
            ),
        )

    def state_binding(
        self, state_id: str, proposal_id: str, evidence_id: str
    ) -> VerifiedScientificState:
        for state in self.state_observations(state_id):
            if state.proposal_id == proposal_id and state.evidence_id == evidence_id:
                return state
        raise ArtifactReferenceError(
            f"no state binding for state={state_id!r}, proposal={proposal_id!r}, "
            f"evidence={evidence_id!r}"
        )

    def has_state(self, state_id: str) -> bool:
        return bool(self.state_observations(state_id))

    def add_proposal(self, proposal: Proposal) -> "ScientificArtifactStore":
        for existing in self.proposals:
            if existing.proposal_id != proposal.proposal_id:
                continue
            if existing.to_dict() == proposal.to_dict():
                return self
            raise ArtifactCollisionError(
                f"proposal_id collision for {proposal.proposal_id}"
            )
        proposals = tuple(sorted(self.proposals + (proposal,), key=lambda item: item.proposal_id))
        return replace(self, proposals=proposals)

    def add_evidence(self, evidence: EvidencePacket) -> "ScientificArtifactStore":
        validate_stored_evidence(evidence)
        for existing in self.evidence:
            if existing.evidence_id != evidence.evidence_id:
                continue
            if existing.to_dict() == evidence.to_dict():
                return self
            raise ArtifactCollisionError(
                f"evidence_id collision for {evidence.evidence_id}"
            )
        packets = tuple(sorted(self.evidence + (evidence,), key=lambda item: item.evidence_id))
        return replace(self, evidence=packets)

    def add_observation(
        self,
        proposal: Proposal,
        evidence: EvidencePacket,
    ) -> "ScientificArtifactStore":
        """Retain one proposal/evidence attempt even when it was not admitted."""

        self._validate_proposal_evidence(proposal, evidence)
        updated = self.add_proposal(proposal)
        return updated.add_evidence(evidence)

    def _add_state(self, state: VerifiedScientificState) -> "ScientificArtifactStore":
        for existing in self.state_observations(state.state_id):
            if (
                existing.proposal_id == state.proposal_id
                and existing.evidence_id == state.evidence_id
            ):
                if existing.to_dict() == state.to_dict():
                    return self
                raise ArtifactCollisionError(
                    f"state binding collision for {state.state_id}"
                )
            if (
                existing.problem_id != state.problem_id
                or not _json_equal(existing.answer_payload, state.answer_payload)
            ):
                raise ArtifactCollisionError(
                    f"scientific state collision for {state.state_id}"
                )
        states = tuple(
            sorted(
                self.states + (state,),
                key=lambda item: (item.state_id, item.evidence_id, item.proposal_id),
            )
        )
        return replace(self, states=states)

    def add_verified(
        self,
        proposal: Proposal,
        state: VerifiedScientificState,
        evidence: EvidencePacket,
    ) -> "ScientificArtifactStore":
        validate_state_evidence(state, evidence)
        if proposal.proposal_id != state.proposal_id:
            raise ArtifactReferenceError("proposal does not own the verified state")
        self._validate_proposal_evidence(proposal, evidence)

        updated = self.add_observation(proposal, evidence)
        return updated._add_state(state)


__all__ = [
    "ArtifactStoreError",
    "ArtifactCollisionError",
    "ArtifactReferenceError",
    "ScientificIdentityError",
    "ScientificArtifactStore",
    "derive_scientific_state_id",
    "scientific_state_id_from_evidence",
    "validate_stored_evidence",
    "validate_state_evidence",
]
