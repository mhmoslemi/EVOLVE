"""Common-verifier integration for every CPU scientific problem adapter."""

import json
from pathlib import Path
from typing import Any, Callable, Tuple

import numpy as np
import pytest

from evolve.archive import ConfirmedRecordTracker, ScientificArchive
from evolve.ids import content_hash, content_id
from evolve.types import Proposal
from evolve.verifier import (
    PersistedAnswerPayload,
    ProblemScientificAdapter,
    VerificationPolicy,
    confirm_persisted_answer,
    verify_persisted_answer,
)
from problems.ac_inequalities import ACInequalities
from problems.circle_packing import CirclePacking
from problems.denoising import BASELINES, POISSON_NORM_MIN, Denoising
from problems.erdos import ErdosMinOverlap


def _circle_case() -> Tuple[Any, Any]:
    problem = CirclePacking({"num_circles": 2, "sandbox_timeout_s": 2})
    candidate = (
        np.asarray([[0.25, 0.5], [0.75, 0.5]], dtype=float),
        np.asarray([0.25, 0.25], dtype=float),
        999.0,
    )
    return problem, candidate


def _erdos_case() -> Tuple[Any, Any]:
    return (
        ErdosMinOverlap({"budget_s": 1, "sandbox_timeout_s": 2}),
        ([0.2, 0.4, 0.6, 0.8], 999.0, 4),
    )


def _ac1_case() -> Tuple[Any, Any]:
    return ACInequalities({"problem_type": "ac1", "budget_s": 1}), [1.0, 2.0, 3.0]


def _ac2_case() -> Tuple[Any, Any]:
    return ACInequalities({"problem_type": "ac2", "budget_s": 1}), [1.0, 2.0, 3.0]


def _denoising_case() -> Tuple[Any, Any]:
    baseline = BASELINES["pancreas"]
    span = baseline["baseline_poisson"] - baseline["perfect_poisson"]
    poisson = baseline["baseline_poisson"] - POISSON_NORM_MIN * span
    return Denoising({"eval_seed": 42}), (0.2, poisson)


def _identifier(namespace: str, label: str) -> str:
    return content_id(namespace, {"label": label})


@pytest.mark.parametrize(
    "case",
    [_circle_case, _erdos_case, _ac1_case, _ac2_case, _denoising_case],
    ids=["circle", "erdos", "ac1", "ac2", "denoising"],
)
def test_problem_adapter_verifies_confirms_archives_and_records_saved_payload(
    case: Callable[[], Tuple[Any, Any]],
    tmp_path: Path,
) -> None:
    problem, candidate = case()
    payload = problem.serialize_answer(candidate)
    label = getattr(problem, "problem_type", problem.name)
    artifact = tmp_path / f"{label}.answer.json"
    artifact.write_text(
        json.dumps(payload, allow_nan=False, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    persisted = PersistedAnswerPayload.create(
        problem_id=problem.name,
        artifact_uri=str(artifact),
        payload=payload,
    )
    source = f"# proposal source is not scientific identity: {label}\n"
    proposal = Proposal(
        proposal_id=_identifier("proposal", label),
        run_id=_identifier("run", "all-problem-pipeline"),
        problem_id=problem.name,
        source_text=source,
        source_hash=content_hash(source),
        branch_id=_identifier("branch", label),
        parsed_candidate={"captured": True},
    )
    adapter = ProblemScientificAdapter(problem)
    policy = VerificationPolicy.create(version="common-problem-v1")

    initial = verify_persisted_answer(
        adapter=adapter,
        proposal=proposal,
        persisted_answer=persisted,
        verification_policy=policy,
        harness_id=_identifier("harness", "baseline-v1"),
        policy_snapshot_id=_identifier("role_snapshot", "scout-epoch0"),
    )
    confirmed = confirm_persisted_answer(
        adapter=adapter,
        proposal=proposal,
        persisted_answer=persisted,
        prior_evidence=initial.evidence,
        verification_policy=policy,
    )

    assert initial.state is not None
    assert confirmed.state is not None
    assert initial.state.state_id == confirmed.state.state_id
    assert initial.evidence.evidence_id != confirmed.evidence.evidence_id
    assert confirmed.evidence.confirmed is True
    assert confirmed.descriptor.method_complete is True
    assert confirmed.evidence.flags["method_incomplete"] is False
    assert adapter.verifier_id == ProblemScientificAdapter(problem).verifier_id

    archive, _decision = ScientificArchive().offer(
        confirmed.descriptor,
        proposal,
        confirmed.state,
        confirmed.evidence,
    )
    record = ConfirmedRecordTracker().consider(
        confirmed.state,
        confirmed.evidence,
        archive=archive,
    )
    assert record.state_id == confirmed.state.state_id
    assert record.raw_score == confirmed.state.raw_score
