"""Phase-2 integration tests for the saved-payload scientific boundary."""

from pathlib import Path
from typing import Optional

import pytest

from evolve.archive import (
    ArchiveAdmissionError,
    ConfirmedRecordTracker,
    ProvenanceStore,
    ScientificArchive,
    make_provenance_edge,
)
from evolve.ids import content_hash, content_id
from evolve.types import Proposal
from evolve.verifier import (
    PersistedAnswerPayload,
    ProblemScientificAdapter,
    VerificationPolicy,
    confirm_persisted_answer,
    verify_persisted_answer,
)
from problems.evolve_toy import EvolveToyProblem


def _id(namespace: str, label: str) -> str:
    return content_id(namespace, {"label": label})


def _proposal(
    *,
    label: str,
    branch_id: str,
    point: list[int],
    parent_state_id: Optional[str] = None,
) -> Proposal:
    source = f"def run_toy():\n    return {point!r}\n# {label}\n"
    return Proposal(
        proposal_id=_id("proposal", label),
        run_id=_id("run", "phase2-integration"),
        problem_id="evolve_toy",
        source_text=source,
        source_hash=content_hash(source),
        parent_state_id=parent_state_id,
        branch_id=branch_id,
        parsed_candidate=point,
    )


def _persisted(tmp_path: Path, *, label: str, point: list[int]) -> PersistedAnswerPayload:
    path = tmp_path / f"{label}.answer.json"
    path.write_text(f"[{point[0]}, {point[1]}]\n", encoding="utf-8")
    return PersistedAnswerPayload.create(
        problem_id="evolve_toy",
        artifact_uri=str(path),
        payload=point,
    )


def _verify_and_confirm(
    *,
    adapter: ProblemScientificAdapter,
    policy: VerificationPolicy,
    proposal: Proposal,
    persisted: PersistedAnswerPayload,
):
    initial = verify_persisted_answer(
        adapter=adapter,
        proposal=proposal,
        persisted_answer=persisted,
        verification_policy=policy,
        harness_id=_id("harness", "baseline-v1"),
        policy_snapshot_id=_id("role_snapshot", "scout-epoch0"),
    )
    confirmed = confirm_persisted_answer(
        adapter=adapter,
        proposal=proposal,
        persisted_answer=persisted,
        prior_evidence=initial.evidence,
        verification_policy=policy,
    )
    return initial, confirmed


def test_toy_payload_flows_through_confirmation_archive_record_and_provenance(
    tmp_path: Path,
) -> None:
    problem = EvolveToyProblem({"num_seed_states": 8})
    adapter = ProblemScientificAdapter(problem)
    policy = VerificationPolicy.create(version="common-toy-v1", production=True)
    parent_branch = _id("branch", "parent")
    child_branch = _id("branch", "child")

    parent_proposal = _proposal(
        label="parent",
        branch_id=parent_branch,
        point=[-4, -4],
    )
    parent_initial, parent_confirmed = _verify_and_confirm(
        adapter=adapter,
        policy=policy,
        proposal=parent_proposal,
        persisted=_persisted(tmp_path, label="parent", point=[-4, -4]),
    )
    assert parent_initial.state is not None
    assert parent_confirmed.state is not None
    assert parent_initial.state.state_id == parent_confirmed.state.state_id
    assert parent_initial.evidence.evidence_id != parent_confirmed.evidence.evidence_id

    archive = ScientificArchive()
    archive, parent_decision = archive.offer(
        parent_confirmed.descriptor,
        parent_proposal,
        parent_confirmed.state,
        parent_confirmed.evidence,
    )
    record = ConfirmedRecordTracker().consider(
        parent_confirmed.state,
        parent_confirmed.evidence,
        archive=archive,
    )

    child_proposal = _proposal(
        label="child",
        branch_id=child_branch,
        point=[3, -2],
        parent_state_id=parent_confirmed.state.state_id,
    )
    child_initial, child_confirmed = _verify_and_confirm(
        adapter=adapter,
        policy=policy,
        proposal=child_proposal,
        persisted=_persisted(tmp_path, label="child", point=[3, -2]),
    )
    assert child_confirmed.state is not None
    archive, child_decision = archive.offer(
        child_confirmed.descriptor,
        child_proposal,
        child_confirmed.state,
        child_confirmed.evidence,
    )
    record = record.consider(
        child_confirmed.state,
        child_confirmed.evidence,
        archive=archive,
    )
    assert record.state_id == child_confirmed.state.state_id
    assert record.internal_reward == 1.0
    assert archive.coverage == 1.0

    provenance = (
        ProvenanceStore(artifacts=archive.artifacts)
        .with_branch(parent_branch)
        .with_branch(child_branch)
    )
    edge = make_provenance_edge(
        parent_state_id=parent_confirmed.state.state_id,
        child_state_id=child_confirmed.state.state_id,
        proposal_id=child_proposal.proposal_id,
        evidence_id=child_confirmed.evidence.evidence_id,
        branch_id=child_branch,
        created_at="2026-01-01T00:00:00Z",
    )
    provenance = provenance.append(edge)
    assert provenance.append(edge) is provenance
    assert provenance.edges == (edge,)

    with pytest.raises(ArchiveAdmissionError, match="only confirmed"):
        ConfirmedRecordTracker().consider(
            child_initial.state,
            child_initial.evidence,
            archive=archive,
        )


def test_equal_saved_answers_share_state_but_keep_proposal_evidence_lineage(
    tmp_path: Path,
) -> None:
    adapter = ProblemScientificAdapter(EvolveToyProblem({}))
    policy = VerificationPolicy.create(version="common-toy-v1")
    first_proposal = _proposal(
        label="equivalent-a",
        branch_id=_id("branch", "equivalent-a"),
        point=[1, 1],
    )
    second_proposal = _proposal(
        label="equivalent-b",
        branch_id=_id("branch", "equivalent-b"),
        point=[1, 1],
    )
    first, _ = _verify_and_confirm(
        adapter=adapter,
        policy=policy,
        proposal=first_proposal,
        persisted=_persisted(tmp_path, label="equivalent-a", point=[1, 1]),
    )
    second, _ = _verify_and_confirm(
        adapter=adapter,
        policy=policy,
        proposal=second_proposal,
        persisted=_persisted(tmp_path, label="equivalent-b", point=[1, 1]),
    )

    assert first.state.state_id == second.state.state_id
    assert first.state.proposal_id != second.state.proposal_id
    assert first.evidence.evidence_id != second.evidence.evidence_id
    assert first.evidence.source_hash != second.evidence.source_hash
