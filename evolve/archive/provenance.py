"""Append-only provenance for verified scientific descendants.

Edges are content addressed independently of wall-clock metadata.  The store
accepts an edge only when its branch and every referenced durable artifact are
already known, so a persisted lineage can never contain a dangling endpoint.
"""

from __future__ import annotations

from dataclasses import dataclass, field, replace
from typing import Any, Dict, Mapping, Tuple

from evolve.ids import content_id, validate_id
from evolve.types import ProvenanceEdge

from .store import ArtifactReferenceError, ScientificArtifactStore


PROVENANCE_IDENTITY_VERSION = "verified_provenance_v1"


class ProvenanceError(ValueError):
    """Base error for an invalid provenance transition."""


class ProvenanceCollisionError(ProvenanceError):
    """An edge identifier was reused for a different immutable relation."""


class ProvenanceEndpointError(ProvenanceError):
    """An edge references a missing or contradictory endpoint."""


def provenance_identity_document(
    *,
    parent_state_id: str,
    child_state_id: str,
    proposal_id: str,
    evidence_id: str,
    branch_id: str,
    relation: str = "descendant",
) -> Mapping[str, Any]:
    """Return the durable edge identity document (timestamps are annotations)."""

    return {
        "identity_version": PROVENANCE_IDENTITY_VERSION,
        "parent_state_id": parent_state_id,
        "child_state_id": child_state_id,
        "proposal_id": proposal_id,
        "evidence_id": evidence_id,
        "branch_id": branch_id,
        "relation": relation,
    }


def provenance_edge_id(edge: ProvenanceEdge) -> str:
    return content_id(
        "provenance",
        provenance_identity_document(
            parent_state_id=edge.parent_state_id,
            child_state_id=edge.child_state_id,
            proposal_id=edge.proposal_id,
            evidence_id=edge.evidence_id,
            branch_id=edge.branch_id,
            relation=edge.relation,
        ),
    )


def make_provenance_edge(
    *,
    parent_state_id: str,
    child_state_id: str,
    proposal_id: str,
    evidence_id: str,
    branch_id: str,
    relation: str = "descendant",
    created_at: str = "",
) -> ProvenanceEdge:
    identity = provenance_identity_document(
        parent_state_id=parent_state_id,
        child_state_id=child_state_id,
        proposal_id=proposal_id,
        evidence_id=evidence_id,
        branch_id=branch_id,
        relation=relation,
    )
    return ProvenanceEdge(
        edge_id=content_id("provenance", identity),
        parent_state_id=parent_state_id,
        child_state_id=child_state_id,
        proposal_id=proposal_id,
        evidence_id=evidence_id,
        branch_id=branch_id,
        relation=relation,
        created_at=created_at,
    )


def _same_edge_identity(left: ProvenanceEdge, right: ProvenanceEdge) -> bool:
    return provenance_identity_document(
        parent_state_id=left.parent_state_id,
        child_state_id=left.child_state_id,
        proposal_id=left.proposal_id,
        evidence_id=left.evidence_id,
        branch_id=left.branch_id,
        relation=left.relation,
    ) == provenance_identity_document(
        parent_state_id=right.parent_state_id,
        child_state_id=right.child_state_id,
        proposal_id=right.proposal_id,
        evidence_id=right.evidence_id,
        branch_id=right.branch_id,
        relation=right.relation,
    )


@dataclass(frozen=True)
class ProvenanceStore:
    """Functional, append-only provenance whose endpoints are durable artifacts."""

    artifacts: ScientificArtifactStore = field(default_factory=ScientificArtifactStore)
    branch_ids: Tuple[str, ...] = field(default_factory=tuple)
    edges: Tuple[ProvenanceEdge, ...] = field(default_factory=tuple)

    def __post_init__(self) -> None:
        if len(set(self.branch_ids)) != len(self.branch_ids):
            raise ProvenanceCollisionError("registered branch IDs must be unique")
        for branch_id in self.branch_ids:
            try:
                validate_id(branch_id, "branch")
            except (TypeError, ValueError) as exc:
                raise ProvenanceEndpointError(str(exc)) from exc

        seen: Dict[str, ProvenanceEdge] = {}
        for edge in self.edges:
            previous = seen.get(edge.edge_id)
            if previous is not None:
                if not _same_edge_identity(previous, edge):
                    raise ProvenanceCollisionError(
                        f"provenance edge collision for {edge.edge_id}"
                    )
                raise ProvenanceCollisionError(
                    f"duplicate provenance edge {edge.edge_id} in store state"
                )
            seen[edge.edge_id] = edge
            self._validate_content_address(edge)
            self._validate_endpoints(edge)

    @staticmethod
    def _validate_content_address(edge: ProvenanceEdge) -> None:
        if edge.edge_id != provenance_edge_id(edge):
            raise ProvenanceCollisionError(
                f"provenance edge ID does not match immutable content: {edge.edge_id}"
            )

    def _validate_endpoints(self, edge: ProvenanceEdge) -> None:
        if edge.branch_id not in self.branch_ids:
            raise ProvenanceEndpointError(
                f"unregistered provenance branch {edge.branch_id}"
            )
        if not self.artifacts.has_state(edge.parent_state_id):
            raise ProvenanceEndpointError(
                f"unknown parent scientific state {edge.parent_state_id}"
            )
        if not self.artifacts.has_state(edge.child_state_id):
            raise ProvenanceEndpointError(
                f"unknown child scientific state {edge.child_state_id}"
            )
        try:
            proposal = self.artifacts.proposal(edge.proposal_id)
            evidence = self.artifacts.evidence_packet(edge.evidence_id)
            self.artifacts.state_binding(
                edge.child_state_id, edge.proposal_id, edge.evidence_id
            )
        except ArtifactReferenceError as exc:
            raise ProvenanceEndpointError(str(exc)) from exc

        if proposal.parent_state_id != edge.parent_state_id:
            raise ProvenanceEndpointError(
                "edge parent does not match the descendant proposal"
            )
        if proposal.branch_id != edge.branch_id:
            raise ProvenanceEndpointError(
                "edge branch does not match the descendant proposal"
            )
        if evidence.parent_state_id != edge.parent_state_id:
            raise ProvenanceEndpointError(
                "edge parent does not match the descendant evidence"
            )
        if evidence.branch_id != edge.branch_id:
            raise ProvenanceEndpointError(
                "edge branch does not match the descendant evidence"
            )
        if evidence.proposal_id != edge.proposal_id:
            raise ProvenanceEndpointError(
                "edge proposal does not match the descendant evidence"
            )
        if evidence.scientific_state_id != edge.child_state_id:
            raise ProvenanceEndpointError(
                "edge child does not match the descendant evidence"
            )
        if edge.parent_state_id not in evidence.lineage_ids:
            raise ProvenanceEndpointError(
                "descendant evidence lineage omits the parent state"
            )

    def with_branch(self, branch_id: str) -> "ProvenanceStore":
        try:
            validate_id(branch_id, "branch")
        except (TypeError, ValueError) as exc:
            raise ProvenanceEndpointError(str(exc)) from exc
        if branch_id in self.branch_ids:
            return self
        return replace(self, branch_ids=tuple(sorted(self.branch_ids + (branch_id,))))

    def with_artifacts(self, artifacts: ScientificArtifactStore) -> "ProvenanceStore":
        """Attach a newer append-only artifact snapshot and revalidate all edges."""

        for proposal in self.artifacts.proposals:
            try:
                if artifacts.proposal(proposal.proposal_id).to_dict() != proposal.to_dict():
                    raise ProvenanceCollisionError("proposal history was rewritten")
            except ArtifactReferenceError as exc:
                raise ProvenanceEndpointError("proposal history was removed") from exc
        for evidence in self.artifacts.evidence:
            try:
                if artifacts.evidence_packet(evidence.evidence_id).to_dict() != evidence.to_dict():
                    raise ProvenanceCollisionError("evidence history was rewritten")
            except ArtifactReferenceError as exc:
                raise ProvenanceEndpointError("evidence history was removed") from exc
        for state in self.artifacts.states:
            try:
                retained = artifacts.state_binding(
                    state.state_id, state.proposal_id, state.evidence_id
                )
            except ArtifactReferenceError as exc:
                raise ProvenanceEndpointError("scientific-state history was removed") from exc
            if retained.to_dict() != state.to_dict():
                raise ProvenanceCollisionError("scientific-state history was rewritten")
        return replace(self, artifacts=artifacts)

    def append(self, edge: ProvenanceEdge) -> "ProvenanceStore":
        """Append an edge, treating a timestamp-only retry as idempotent."""

        for existing in self.edges:
            if existing.edge_id != edge.edge_id:
                continue
            if _same_edge_identity(existing, edge):
                return self
            raise ProvenanceCollisionError(
                f"provenance edge collision for {edge.edge_id}"
            )
        self._validate_content_address(edge)
        self._validate_endpoints(edge)
        return replace(
            self,
            edges=tuple(sorted(self.edges + (edge,), key=lambda item: item.edge_id)),
        )


__all__ = [
    "PROVENANCE_IDENTITY_VERSION",
    "ProvenanceCollisionError",
    "ProvenanceEndpointError",
    "ProvenanceError",
    "ProvenanceStore",
    "make_provenance_edge",
    "provenance_edge_id",
    "provenance_identity_document",
]
