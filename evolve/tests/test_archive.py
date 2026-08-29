from __future__ import annotations

from dataclasses import dataclass, replace
from typing import Any, Mapping, Optional

import pytest

from evolve.archive import (
    ArchiveAdmissionError,
    ArtifactCollisionError,
    ConfirmedRecordTracker,
    ProvenanceCollisionError,
    ProvenanceEndpointError,
    ProvenanceStore,
    ScientificArchive,
    ScientificArtifactStore,
    cell_id_for_descriptor,
    create_descriptor,
    derive_scientific_state_id,
    make_provenance_edge,
    provenance_edge_id,
    verified_structure_fingerprint,
)
from evolve.ids import content_hash, content_id
from evolve.types import (
    Descriptor,
    EvidencePacket,
    FailureKind,
    Proposal,
    VerifiedScientificState,
)
from evolve.verifier.evidence import (
    EVIDENCE_IDENTITY_VERSION,
    descriptor_id as verifier_descriptor_id,
    evidence_id_from_fields,
    scientific_state_id as verifier_scientific_state_id,
)


RUN_ID = content_id("run", {"fixture": "archive"})
BRANCH_ID = content_id("branch", {"fixture": "archive"})
VERIFIER_ID = content_id("verifier", {"fixture": "common"})
HARNESS_ID = content_id("harness", {"fixture": "baseline"})
POLICY_ID = content_id("role_snapshot", {"fixture": "scout-v1"})


@dataclass(frozen=True)
class Observation:
    proposal: Proposal
    state: VerifiedScientificState
    evidence: EvidencePacket
    descriptor: Any


def _family_dimensions(view: Any) -> Mapping[str, Any]:
    return {
        "algorithmic_family": view.answer_payload["family"],
        "construction_kind": view.answer_payload.get("construction_kind", "explicit"),
    }


def _observation(
    *,
    source: str,
    answer_payload: Mapping[str, Any],
    internal_reward: float,
    confirmed: bool,
    observation_index: int,
    parent_state_id: Optional[str] = None,
    raw_score: Any = None,
    branch_id: str = BRANCH_ID,
    descriptor_version: str = "archive_toy_descriptor_v1",
    fingerprint_version: str = "verified_structure_v1",
) -> Observation:
    source_hash = content_hash(source)
    state_id = derive_scientific_state_id(
        problem_id="archive_toy", answer_payload=answer_payload
    )
    proposal_id = content_id(
        "proposal",
        {
            "run_id": RUN_ID,
            "problem_id": "archive_toy",
            "source_hash": source_hash,
            "parent_state_id": parent_state_id,
            "branch_id": branch_id,
        },
    )
    proposal = Proposal(
        proposal_id=proposal_id,
        run_id=RUN_ID,
        problem_id="archive_toy",
        source_text=source,
        source_hash=source_hash,
        parent_state_id=parent_state_id,
        branch_id=branch_id,
        parsed_candidate=dict(answer_payload),
        created_at="2026-01-01T00:00:00Z",
    )
    native_score = (
        {"native_metric": -internal_reward} if raw_score is None else raw_score
    )
    dimensions = {
        "algorithmic_family": answer_payload["family"],
        "construction_kind": answer_payload.get("construction_kind", "explicit"),
    }
    initial_descriptor = Descriptor(
        descriptor_id=verifier_descriptor_id(
            problem_id="archive_toy",
            function_version=descriptor_version,
            dimensions=dimensions,
            method_complete=True,
        ),
        problem_id="archive_toy",
        function_version=descriptor_version,
        dimensions=dimensions,
        method_complete=True,
    )
    completed_at = f"2026-01-01T00:00:{observation_index:02d}Z"
    evidence_id = content_id(
        "evidence",
        {
            "proposal_id": proposal_id,
            "state_id": state_id,
            "confirmed": confirmed,
            "internal_reward": internal_reward,
            "raw_score": native_score,
            "completed_at": completed_at,
        },
    )
    lineage = tuple(
        dict.fromkeys(
            (state_id,)
            if parent_state_id is None
            else (parent_state_id, state_id)
        )
    )
    evidence = EvidencePacket(
        evidence_id=evidence_id,
        run_id=RUN_ID,
        proposal_id=proposal_id,
        scientific_state_id=state_id,
        parent_state_id=parent_state_id,
        branch_id=branch_id,
        problem_id="archive_toy",
        verifier_id=VERIFIER_ID,
        verifier_version="common_verifier_v1",
        harness_id=HARNESS_ID,
        policy_snapshot_id=POLICY_ID,
        lineage_ids=lineage,
        resolved=True,
        admitted=True,
        confirmed=confirmed,
        failure_kind=FailureKind.NONE,
        internal_reward=internal_reward,
        raw_score=native_score,
        uncertainty=0.25,
        descriptor_id=initial_descriptor.descriptor_id,
        fingerprint="pending",
        source_hash=source_hash,
        flags={
            "common_verifier": True,
            "evidence_identity_version": EVIDENCE_IDENTITY_VERSION,
        },
        scores={"native_score": native_score},
        diagnostics={"constraint_classes": ["toy"]},
        resources={"verifier_calls": 1, "verifier_seconds": 0.01},
        answer_payload=dict(answer_payload),
        timeout_is_scientific=False,
        started_at="2026-01-01T00:00:00Z",
        completed_at=completed_at,
    )
    state = VerifiedScientificState(
        state_id=state_id,
        proposal_id=proposal_id,
        evidence_id=evidence_id,
        problem_id="archive_toy",
        answer_payload=dict(answer_payload),
        resolved=True,
        admitted=True,
        confirmed=confirmed,
        internal_reward=internal_reward,
        raw_score=native_score,
        descriptor_id=initial_descriptor.descriptor_id,
        fingerprint="pending",
    )
    descriptor = create_descriptor(
        state,
        evidence,
        function_version=descriptor_version,
        extractor=_family_dimensions,
        method_complete=True,
    )
    assert descriptor == initial_descriptor
    fingerprint = verified_structure_fingerprint(
        state, evidence, version=fingerprint_version
    )
    state = replace(state, fingerprint=fingerprint)
    evidence = replace(evidence, fingerprint=fingerprint)
    final_evidence_id = evidence_id_from_fields(evidence.to_dict())
    evidence = replace(evidence, evidence_id=final_evidence_id)
    state = replace(state, evidence_id=final_evidence_id)
    return Observation(proposal, state, evidence, descriptor)


def test_state_descriptor_and_fingerprint_are_source_invariant_and_interoperable() -> None:
    answer = {
        "family": "spectral",
        "construction_kind": "explicit",
        "values": [2, 3, 5],
    }
    initial = _observation(
        source="def candidate(): return [2, 3, 5]",
        answer_payload=answer,
        internal_reward=3.0,
        confirmed=False,
        observation_index=1,
    )
    confirmation = _observation(
        source="result = tuple((2, 3, 5))  # independent source",
        answer_payload=answer,
        internal_reward=3.2,
        confirmed=True,
        observation_index=2,
    )

    expected_state_id = verifier_scientific_state_id("archive_toy", answer)
    assert initial.state.state_id == expected_state_id
    assert confirmation.state.state_id == expected_state_id
    assert initial.descriptor.descriptor_id == confirmation.descriptor.descriptor_id
    assert initial.state.fingerprint == confirmation.state.fingerprint
    assert initial.proposal.source_hash != confirmation.proposal.source_hash
    assert initial.evidence.confirmed is False
    assert confirmation.evidence.confirmed is True
    assert initial.evidence.internal_reward != confirmation.evidence.internal_reward

    assert initial.descriptor.descriptor_id == verifier_descriptor_id(
        problem_id="archive_toy",
        function_version="archive_toy_descriptor_v1",
        dimensions=initial.descriptor.dimensions,
        method_complete=True,
    )

    store = ScientificArtifactStore()
    store = store.add_verified(initial.proposal, initial.state, initial.evidence)
    store = store.add_verified(
        confirmation.proposal, confirmation.state, confirmation.evidence
    )
    assert len(store.state_observations(expected_state_id)) == 2
    assert store.representative_state(expected_state_id) == confirmation.state


def test_same_proposal_confirmation_is_a_new_observation_not_a_state_collision() -> None:
    answer = {
        "family": "matching",
        "construction_kind": "explicit",
        "edges": [[0, 1], [2, 3]],
    }
    initial = _observation(
        source="edges = [(0, 1), (2, 3)]",
        answer_payload=answer,
        internal_reward=1.9,
        confirmed=False,
        observation_index=1,
    )
    confirmation = _observation(
        source="edges = [(0, 1), (2, 3)]",
        answer_payload=answer,
        internal_reward=2.0,
        confirmed=True,
        observation_index=2,
    )
    assert initial.proposal == confirmation.proposal
    assert initial.state.state_id == confirmation.state.state_id
    assert initial.state.evidence_id != confirmation.state.evidence_id

    store = ScientificArtifactStore()
    store = store.add_verified(initial.proposal, initial.state, initial.evidence)
    store = store.add_verified(
        confirmation.proposal, confirmation.state, confirmation.evidence
    )
    assert len(store.proposals) == 1
    assert len(store.evidence) == 2
    assert len(store.states) == 2
    assert store.representative_state(initial.state.state_id).confirmed is True


def test_state_identity_survives_descriptor_and_fingerprint_version_updates() -> None:
    answer = {
        "family": "versioned",
        "construction_kind": "implicit",
        "values": [1, 4, 9],
    }
    first = _observation(
        source="values = [1, 4, 9]",
        answer_payload=answer,
        internal_reward=2.0,
        confirmed=True,
        observation_index=1,
        descriptor_version="archive_toy_descriptor_v1",
        fingerprint_version="verified_structure_v1",
    )
    remapped = _observation(
        source="values = [1, 4, 9]",
        answer_payload=answer,
        internal_reward=2.1,
        confirmed=True,
        observation_index=2,
        descriptor_version="archive_toy_descriptor_v2",
        fingerprint_version="verified_structure_v2",
    )
    assert first.proposal == remapped.proposal
    assert first.state.state_id == remapped.state.state_id
    assert first.state.descriptor_id != remapped.state.descriptor_id
    assert first.state.fingerprint != remapped.state.fingerprint

    store = ScientificArtifactStore()
    store = store.add_verified(first.proposal, first.state, first.evidence)
    store = store.add_verified(remapped.proposal, remapped.state, remapped.evidence)
    assert len(store.state_observations(first.state.state_id)) == 2
    assert store.representative_state(first.state.state_id) == remapped.state
    assert (
        store.representative_state(
            first.state.state_id, descriptor_id=first.descriptor.descriptor_id
        )
        == first.state
    )

    archive = ScientificArchive()
    archive, _ = archive.offer(
        first.descriptor, first.proposal, first.state, first.evidence
    )
    archive, _ = archive.offer(
        remapped.descriptor,
        remapped.proposal,
        remapped.state,
        remapped.evidence,
    )
    assert len(archive.cells) == 2
    assert all(cell.tested_count == 1 for cell in archive.cells)


def test_artifact_identifier_collision_is_rejected_without_mutation() -> None:
    observation = _observation(
        source="answer = 7",
        answer_payload={"family": "direct", "value": 7},
        internal_reward=7.0,
        confirmed=True,
        observation_index=1,
    )
    store = ScientificArtifactStore().add_verified(
        observation.proposal, observation.state, observation.evidence
    )
    conflicting = replace(
        observation.proposal,
        source_text="answer = 8",
        source_hash=content_hash("answer = 8"),
    )
    with pytest.raises(ArtifactCollisionError, match="proposal_id collision"):
        store.add_proposal(conflicting)
    assert store.proposal(observation.proposal.proposal_id) == observation.proposal


@pytest.mark.parametrize(
    ("failure_kind", "resolved"),
    [
        (FailureKind.CONSTRAINT, True),
        (FailureKind.INFRASTRUCTURE, False),
    ],
)
def test_nonadmitted_and_infrastructure_evidence_remain_append_only(
    failure_kind: FailureKind,
    resolved: bool,
) -> None:
    observation = _observation(
        source=f"failed-{failure_kind.value}",
        answer_payload={"family": "failed", "kind": failure_kind.value},
        internal_reward=1.0,
        confirmed=False,
        observation_index=1,
    )
    failed = replace(
        observation.evidence,
        scientific_state_id=None,
        resolved=resolved,
        admitted=False,
        confirmed=False,
        failure_kind=failure_kind,
        internal_reward=None,
        descriptor_id=None,
        fingerprint="",
    )
    failed = replace(
        failed,
        evidence_id=evidence_id_from_fields(failed.to_dict()),
    )

    store = ScientificArtifactStore().add_observation(
        observation.proposal,
        failed,
    )
    retried = store.add_observation(observation.proposal, failed)

    assert retried is store
    assert store.evidence == (failed,)
    assert store.states == ()


def test_artifact_store_rejects_evidence_tampering_under_a_retained_id() -> None:
    observation = _observation(
        source="tamper target",
        answer_payload={"family": "tamper", "value": 1},
        internal_reward=1.0,
        confirmed=True,
        observation_index=1,
    )
    tampered = replace(
        observation.evidence,
        diagnostics={"verdict": "rewritten after verification"},
    )

    with pytest.raises(ArtifactCollisionError, match="evidence_id"):
        ScientificArtifactStore().add_observation(
            observation.proposal,
            tampered,
        )


def test_archive_has_distinct_slots_and_champion_requires_confirmation() -> None:
    a = _observation(
        source="a",
        answer_payload={"family": "graph", "construction_kind": "explicit", "x": 1},
        internal_reward=1.0,
        confirmed=True,
        observation_index=1,
    )
    b = _observation(
        source="b",
        answer_payload={"family": "graph", "construction_kind": "explicit", "x": 2},
        internal_reward=4.5,
        confirmed=False,
        observation_index=2,
    )
    c = _observation(
        source="c",
        answer_payload={"family": "graph", "construction_kind": "explicit", "x": 3},
        internal_reward=2.0,
        confirmed=True,
        observation_index=3,
    )
    d = _observation(
        source="d",
        answer_payload={"family": "graph", "construction_kind": "explicit", "x": 4},
        internal_reward=3.0,
        confirmed=True,
        observation_index=4,
    )
    assert len({item.descriptor.descriptor_id for item in (a, b, c, d)}) == 1

    empty = ScientificArchive(under_tested_threshold=3)
    archive, _ = empty.offer(a.descriptor, a.proposal, a.state, a.evidence)
    assert empty.cells == ()
    archive, decision_b = archive.offer(
        b.descriptor, b.proposal, b.state, b.evidence
    )
    assert decision_b.champion_state_id == a.state.state_id
    assert decision_b.promising_state_ids == (b.state.state_id,)

    archive, _ = archive.offer(c.descriptor, c.proposal, c.state, c.evidence)
    cell = archive.cell(cell_id_for_descriptor(a.descriptor))
    assert cell.champion_state_id == c.state.state_id
    assert cell.champion_evidence_id == c.evidence.evidence_id
    assert archive.artifacts.evidence_packet(cell.champion_evidence_id).confirmed
    assert cell.promising_state_ids == (b.state.state_id,)
    assert cell.stepping_stone_state_ids == (a.state.state_id,)
    assert not (
        {cell.champion_state_id}
        & set(cell.promising_state_ids + cell.stepping_stone_state_ids)
    )

    archive, decision_d = archive.offer(
        d.descriptor, d.proposal, d.state, d.evidence
    )
    cell = archive.cell(cell.cell_id)
    assert cell.champion_state_id == d.state.state_id
    assert decision_d.evicted_slot_state_ids
    assert len(archive.artifacts.states) == 4
    for state_id in decision_d.evicted_slot_state_ids:
        assert archive.artifacts.has_state(state_id)

    # Confirmation re-observes the same saved scientific answer.  It replaces
    # the cell representative but does not count as a fifth scientific state.
    b_confirmed = _observation(
        source="b",
        answer_payload={"family": "graph", "construction_kind": "explicit", "x": 2},
        internal_reward=5.0,
        confirmed=True,
        observation_index=5,
    )
    before_count = cell.tested_count
    archive, confirmation_decision = archive.offer(
        b_confirmed.descriptor,
        b_confirmed.proposal,
        b_confirmed.state,
        b_confirmed.evidence,
    )
    cell = archive.cell(cell.cell_id)
    assert b_confirmed.state.state_id == b.state.state_id
    assert cell.tested_count == before_count
    assert cell.champion_state_id == b.state.state_id
    assert cell.champion_evidence_id == b_confirmed.evidence.evidence_id
    assert confirmation_decision.champion_changed


def test_record_tracker_is_confirmed_only_max_seeking_and_preserves_raw_score() -> None:
    low = _observation(
        source="low",
        answer_payload={"family": "record", "x": 1},
        internal_reward=1.0,
        confirmed=True,
        observation_index=1,
        raw_score={"native_loss": 90.0},
    )
    high_unconfirmed = _observation(
        source="high-u",
        answer_payload={"family": "record", "x": 2},
        internal_reward=8.0,
        confirmed=False,
        observation_index=2,
        raw_score={"native_loss": 20.0},
    )
    high = _observation(
        source="high",
        answer_payload={"family": "record", "x": 3},
        internal_reward=4.0,
        confirmed=True,
        observation_index=3,
        raw_score={"native_loss": 30.0},
    )
    archive = ScientificArchive()
    for observation in (low, high_unconfirmed, high):
        archive, _ = archive.offer(
            observation.descriptor,
            observation.proposal,
            observation.state,
            observation.evidence,
        )
    tracker = ConfirmedRecordTracker()
    with pytest.raises(ArchiveAdmissionError, match="only confirmed"):
        tracker.consider(
            high_unconfirmed.state,
            high_unconfirmed.evidence,
            archive=archive,
        )
    with pytest.raises(ArchiveAdmissionError, match="exact archive state"):
        tracker.consider(low.state, low.evidence, archive=ScientificArchive())
    tracker = tracker.consider(low.state, low.evidence, archive=archive)
    tracker = tracker.consider(high.state, high.evidence, archive=archive)
    unchanged = tracker.consider(low.state, low.evidence, archive=archive)
    assert unchanged is tracker
    assert tracker.internal_reward == 4.0
    assert tracker.raw_score == {"native_loss": 30.0}
    assert tracker.evidence_id == high.evidence.evidence_id


def test_forced_empty_and_under_tested_cells_are_selected_first() -> None:
    occupied = _observation(
        source="occupied",
        answer_payload={"family": "occupied", "x": 1},
        internal_reward=1.0,
        confirmed=True,
        observation_index=1,
    )
    under_tested = _observation(
        source="under",
        answer_payload={"family": "under-tested", "x": 2},
        internal_reward=2.0,
        confirmed=True,
        observation_index=2,
    )
    empty_observation = _observation(
        source="empty",
        answer_payload={"family": "empty", "x": 3},
        internal_reward=3.0,
        confirmed=True,
        observation_index=3,
    )
    archive = ScientificArchive(under_tested_threshold=2)
    archive, _ = archive.offer(
        occupied.descriptor, occupied.proposal, occupied.state, occupied.evidence
    )
    # A duplicate binding does not inflate scientific coverage/test counts.
    archive, duplicate = archive.offer(
        occupied.descriptor, occupied.proposal, occupied.state, occupied.evidence
    )
    assert duplicate.duplicate
    assert archive.cell(cell_id_for_descriptor(occupied.descriptor)).tested_count == 1
    archive, _ = archive.offer(
        under_tested.descriptor,
        under_tested.proposal,
        under_tested.state,
        under_tested.evidence,
    )
    archive = archive.ensure_cell(empty_observation.descriptor)

    selected = archive.sampling_cells()
    assert selected[0].cell_id == cell_id_for_descriptor(empty_observation.descriptor)
    assert selected[0].force_empty_sampling
    assert {cell.cell_id for cell in selected[1:]} == {
        cell_id_for_descriptor(occupied.descriptor),
        cell_id_for_descriptor(under_tested.descriptor),
    }
    assert archive.coverage == pytest.approx(2.0 / 3.0)


def test_provenance_is_append_only_idempotent_and_validates_endpoints() -> None:
    parent = _observation(
        source="parent",
        answer_payload={"family": "seed", "value": 1},
        internal_reward=1.0,
        confirmed=True,
        observation_index=1,
    )
    child = _observation(
        source="child",
        answer_payload={"family": "repair", "value": 2},
        internal_reward=2.0,
        confirmed=True,
        observation_index=2,
        parent_state_id=parent.state.state_id,
    )
    artifacts = ScientificArtifactStore()
    artifacts = artifacts.add_verified(parent.proposal, parent.state, parent.evidence)
    artifacts = artifacts.add_verified(child.proposal, child.state, child.evidence)
    edge = make_provenance_edge(
        parent_state_id=parent.state.state_id,
        child_state_id=child.state.state_id,
        proposal_id=child.proposal.proposal_id,
        evidence_id=child.evidence.evidence_id,
        branch_id=BRANCH_ID,
        relation="descendant",
        created_at="2026-01-01T00:00:02Z",
    )
    assert provenance_edge_id(edge) == edge.edge_id

    with pytest.raises(ProvenanceEndpointError, match="unregistered"):
        ProvenanceStore(artifacts=artifacts).append(edge)
    provenance = ProvenanceStore(artifacts=artifacts).with_branch(BRANCH_ID)
    provenance = provenance.append(edge)
    retried = provenance.append(
        replace(edge, created_at="2026-01-01T00:05:00Z")
    )
    assert retried is provenance
    assert len(provenance.edges) == 1

    collision = replace(edge, relation="minimal_repair")
    with pytest.raises(ProvenanceCollisionError, match="collision"):
        provenance.append(collision)

    missing_parent = content_id("state", {"missing": "parent"})
    dangling = make_provenance_edge(
        parent_state_id=missing_parent,
        child_state_id=child.state.state_id,
        proposal_id=child.proposal.proposal_id,
        evidence_id=child.evidence.evidence_id,
        branch_id=BRANCH_ID,
    )
    with pytest.raises(ProvenanceEndpointError, match="unknown parent"):
        provenance.append(dangling)
    with pytest.raises(ProvenanceEndpointError, match="unknown parent"):
        ProvenanceStore(
            artifacts=artifacts,
            branch_ids=(BRANCH_ID,),
            edges=(dangling,),
        )


def test_provenance_rejects_content_address_and_binding_mismatches() -> None:
    parent = _observation(
        source="p",
        answer_payload={"family": "p", "value": 1},
        internal_reward=1.0,
        confirmed=True,
        observation_index=1,
    )
    child = _observation(
        source="q",
        answer_payload={"family": "q", "value": 2},
        internal_reward=2.0,
        confirmed=True,
        observation_index=2,
        parent_state_id=parent.state.state_id,
    )
    artifacts = ScientificArtifactStore()
    artifacts = artifacts.add_verified(parent.proposal, parent.state, parent.evidence)
    artifacts = artifacts.add_verified(child.proposal, child.state, child.evidence)
    provenance = ProvenanceStore(artifacts=artifacts).with_branch(BRANCH_ID)
    edge = make_provenance_edge(
        parent_state_id=parent.state.state_id,
        child_state_id=child.state.state_id,
        proposal_id=child.proposal.proposal_id,
        evidence_id=child.evidence.evidence_id,
        branch_id=BRANCH_ID,
    )
    wrong_id = replace(
        edge, edge_id=content_id("provenance", {"not": "this edge"})
    )
    with pytest.raises(ProvenanceCollisionError, match="does not match"):
        provenance.append(wrong_id)

    mismatched = make_provenance_edge(
        parent_state_id=parent.state.state_id,
        child_state_id=child.state.state_id,
        proposal_id=parent.proposal.proposal_id,
        evidence_id=parent.evidence.evidence_id,
        branch_id=BRANCH_ID,
    )
    with pytest.raises(ProvenanceEndpointError, match="state binding"):
        provenance.append(mismatched)


def test_provenance_retains_same_state_duplicate_descendants() -> None:
    payload = {"family": "fixed-point", "value": 7}
    parent = _observation(
        source="return seven v1",
        answer_payload=payload,
        internal_reward=7.0,
        confirmed=True,
        observation_index=1,
    )
    duplicate = _observation(
        source="return seven v2",
        answer_payload=payload,
        internal_reward=7.0,
        confirmed=True,
        observation_index=2,
        parent_state_id=parent.state.state_id,
    )
    assert duplicate.state.state_id == parent.state.state_id

    artifacts = ScientificArtifactStore()
    artifacts = artifacts.add_verified(
        parent.proposal, parent.state, parent.evidence
    )
    artifacts = artifacts.add_verified(
        duplicate.proposal, duplicate.state, duplicate.evidence
    )
    edge = make_provenance_edge(
        parent_state_id=parent.state.state_id,
        child_state_id=duplicate.state.state_id,
        proposal_id=duplicate.proposal.proposal_id,
        evidence_id=duplicate.evidence.evidence_id,
        branch_id=BRANCH_ID,
        relation="duplicate",
    )

    provenance = (
        ProvenanceStore(artifacts=artifacts)
        .with_branch(BRANCH_ID)
        .append(edge)
    )
    assert provenance.edges == (edge,)

    with pytest.raises(ValueError, match="relation='duplicate'"):
        make_provenance_edge(
            parent_state_id=parent.state.state_id,
            child_state_id=duplicate.state.state_id,
            proposal_id=duplicate.proposal.proposal_id,
            evidence_id=duplicate.evidence.evidence_id,
            branch_id=BRANCH_ID,
            relation="descendant",
        )

    with pytest.raises(ValueError, match="relation='duplicate'"):
        make_provenance_edge(
            parent_state_id=parent.state.state_id,
            child_state_id=content_id("state", {"different": True}),
            proposal_id=duplicate.proposal.proposal_id,
            evidence_id=duplicate.evidence.evidence_id,
            branch_id=BRANCH_ID,
            relation="duplicate",
        )
