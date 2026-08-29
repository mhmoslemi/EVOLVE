from dataclasses import replace
from datetime import datetime, timezone
from itertools import count
import json
from pathlib import Path
import tempfile
from types import SimpleNamespace

import pytest

from evolve.ids import canonical_json, content_hash, content_id
from evolve.types import EvidencePacket, FailureKind, FrozenDict, Proposal
from evolve.verifier import (
    ExecutionCapture,
    PersistedAnswerPayload,
    ProblemScientificAdapter,
    VerificationDecision,
    VerificationPolicy,
    VerificationServiceError,
    VerificationValidationError,
    classify_failure,
    confirm_persisted_answer,
    scientific_state_id,
    validate_evidence_identity,
    validate_state_identity,
    verify_persisted_answer,
)
from problems.evolve_toy import EvolveToyProblem


def _identifier(namespace, label):
    return content_id(namespace, {"label": label})


def _proposal(
    *,
    source="raise AssertionError('proposal code must not execute')",
    proposal_label="proposal-a",
    branch_label="branch-a",
    run_label="run-a",
    problem_id="toy",
    parent_state_id=None,
):
    return Proposal(
        proposal_id=_identifier("proposal", proposal_label),
        run_id=_identifier("run", run_label),
        problem_id=problem_id,
        source_text=source,
        source_hash=content_hash(source),
        parent_state_id=parent_state_id,
        branch_id=_identifier("branch", branch_label),
        parsed_candidate={"untrusted_source": source},
    )


_DEFAULT_PAYLOAD = object()
_ARTIFACT_DIRECTORY_HANDLE = tempfile.TemporaryDirectory(
    prefix="evolve-verifier-tests-"
)
_ARTIFACT_DIRECTORY = Path(_ARTIFACT_DIRECTORY_HANDLE.name)
_ARTIFACT_INDEX = count()


def _payload(value=_DEFAULT_PAYLOAD, *, problem_id="toy", artifact=None):
    payload = {"point": [3, -2]} if value is _DEFAULT_PAYLOAD else value
    path = (
        _ARTIFACT_DIRECTORY / f"answer-{next(_ARTIFACT_INDEX):05d}.json"
        if artifact is None else Path(artifact)
    )
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(payload, allow_nan=False, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return PersistedAnswerPayload.create(
        problem_id=problem_id,
        artifact_uri=str(path),
        payload=payload,
    )


def _success(*, diagnostics=None, resources=None, reward=1.0, raw_score=0.0):
    return VerificationDecision.success(
        internal_reward=reward,
        raw_score=raw_score,
        uncertainty=0.0,
        flags={"deterministic": True},
        scores={"native": raw_score},
        capture=ExecutionCapture(
            diagnostics=FrozenDict(diagnostics or {"verdict": "valid"}),
            resources=FrozenDict(resources or {"cpu_seconds": 0.01}),
            started_at="2026-01-01T00:00:00Z",
            completed_at="2026-01-01T00:00:01Z",
        ),
    )


class _FakeAdapter:
    problem_id = "toy"
    verifier_version = "fake_verifier_v1"
    descriptor_version = "toy_cells_v1"
    method_complete = True
    timeout_is_scientific = False

    def __init__(self, decisions=None, *, label="fake", descriptor=None, fingerprint=None):
        self.verifier_id = _identifier("verifier", label)
        self.decisions = list(decisions or [_success()])
        self.calls = []
        self.descriptor_calls = []
        self.fingerprint_calls = []
        self._descriptor = descriptor or {"quadrant": "south_east", "band": "inner"}
        self._fingerprint = fingerprint or "toy-structure:3,-2"

    def verify_answer_payload(self, payload, policy):
        self.calls.append((payload, policy.policy_id))
        item = self.decisions[min(len(self.calls) - 1, len(self.decisions) - 1)]
        if isinstance(item, Exception):
            raise item
        return item

    def describe_scientific_state(self, payload, decision):
        self.descriptor_calls.append((payload, decision.internal_reward))
        if isinstance(self._descriptor, Exception):
            raise self._descriptor
        return self._descriptor

    def scientific_fingerprint(self, payload, decision):
        self.fingerprint_calls.append((payload, decision.internal_reward))
        if isinstance(self._fingerprint, Exception):
            raise self._fingerprint
        return self._fingerprint


def _verify(adapter=None, proposal=None, payload=None, policy=None, **kwargs):
    return verify_persisted_answer(
        adapter=adapter or _FakeAdapter(),
        proposal=proposal or _proposal(),
        persisted_answer=payload or _payload(),
        verification_policy=policy or VerificationPolicy.create(version="common_v1"),
        harness_id=kwargs.pop("harness_id", _identifier("harness", "baseline-v1")),
        policy_snapshot_id=kwargs.pop(
            "policy_snapshot_id", _identifier("role_snapshot", "scout-epoch0")
        ),
        **kwargs,
    )


def _count_problem_verifier_hooks(problem):
    calls = {"verify": 0, "describe": 0, "fingerprint": 0}
    original_verify = problem.verify_answer_payload
    original_describe = problem.describe_scientific_state
    original_fingerprint = problem.scientific_fingerprint

    def verify(payload, policy):
        calls["verify"] += 1
        return original_verify(payload, policy)

    def describe(candidate, evidence):
        calls["describe"] += 1
        return original_describe(candidate, evidence)

    def fingerprint(candidate, evidence):
        calls["fingerprint"] += 1
        return original_fingerprint(candidate, evidence)

    problem.verify_answer_payload = verify
    problem.describe_scientific_state = describe
    problem.scientific_fingerprint = fingerprint
    return calls


def _mutate_problem_cfg(problem):
    problem.cfg["seed"] = 999


def _mutate_problem_effective_field(problem):
    problem.target = -123.0


def _mutate_problem_resources(problem):
    changed = replace(problem.resource_requirements(), timeout_s=999.0)
    problem.resource_requirements = lambda: changed


def test_policy_payload_and_capture_are_content_addressed_and_recursively_frozen():
    policy = VerificationPolicy.create(
        version="v1",
        resource_limits={"cpu_seconds": 2.0, "memory_mb": 64},
    )
    same = VerificationPolicy.create(
        version="v1",
        resource_limits={"memory_mb": 64, "cpu_seconds": 2.0},
    )
    payload = _payload({"nested": {"values": [1, 2]}})

    assert policy.policy_id == same.policy_id
    assert payload.payload_id == content_id(
        "answer_payload",
        {"problem_id": "toy", "payload": {"nested": {"values": [1, 2]}}},
    )
    with pytest.raises(TypeError):
        policy.resource_limits["cpu_seconds"] = 3
    with pytest.raises(TypeError):
        payload.payload["nested"]["values"][0] = 9


def test_persisted_payload_requires_exact_regular_json_artifact(tmp_path):
    payload = {"point": [3, -2]}
    valid_path = tmp_path / "nested" / "answer.json"
    valid_path.parent.mkdir()
    valid_path.write_text('{"point": [3, -2]}\n', encoding="utf-8")

    persisted = PersistedAnswerPayload.create(
        problem_id="toy",
        artifact_uri=str(valid_path.parent / ".." / "nested" / "answer.json"),
        payload=payload,
    )

    assert Path(persisted.artifact_uri).is_absolute()
    assert persisted.artifact_uri == str(valid_path.resolve())
    assert persisted.validate_durable_artifact() == str(valid_path.resolve())

    with pytest.raises(VerificationValidationError, match="does not exist"):
        PersistedAnswerPayload.create(
            problem_id="toy",
            artifact_uri=str(tmp_path / "missing.json"),
            payload=payload,
        )

    malformed = tmp_path / "malformed.json"
    malformed.write_text('{"point": [3,}\n', encoding="utf-8")
    with pytest.raises(VerificationValidationError, match="malformed JSON"):
        PersistedAnswerPayload.create(
            problem_id="toy", artifact_uri=str(malformed), payload=payload
        )

    mismatch = tmp_path / "mismatch.json"
    mismatch.write_text('{"point": [0, 0]}\n', encoding="utf-8")
    with pytest.raises(VerificationValidationError, match="does not match"):
        PersistedAnswerPayload.create(
            problem_id="toy", artifact_uri=str(mismatch), payload=payload
        )

    with pytest.raises(VerificationValidationError, match="regular file"):
        PersistedAnswerPayload.create(
            problem_id="toy", artifact_uri=str(tmp_path), payload=payload
        )


def test_persisted_payload_rejects_symlink_and_ambiguous_json(tmp_path):
    payload = {"point": [3, -2]}
    target = tmp_path / "target.json"
    target.write_text(json.dumps(payload), encoding="utf-8")
    link = tmp_path / "answer-link.json"
    try:
        link.symlink_to(target)
    except (NotImplementedError, OSError):
        pytest.skip("symlinks are unavailable on this platform")
    with pytest.raises(VerificationValidationError, match="symlink"):
        PersistedAnswerPayload.create(
            problem_id="toy", artifact_uri=str(link), payload=payload
        )

    duplicate = tmp_path / "duplicate.json"
    duplicate.write_text(
        '{"point": [3, -2], "point": [3, -2]}\n', encoding="utf-8"
    )
    with pytest.raises(VerificationValidationError, match="duplicate JSON key"):
        PersistedAnswerPayload.create(
            problem_id="toy", artifact_uri=str(duplicate), payload=payload
        )


@pytest.mark.parametrize(
    "mutate",
    [_mutate_problem_cfg, _mutate_problem_effective_field,
     _mutate_problem_resources],
    ids=["cfg", "effective-field", "resources"],
)
def test_problem_adapter_identity_drift_fails_before_verifier_hooks(mutate):
    problem = EvolveToyProblem({})
    calls = _count_problem_verifier_hooks(problem)
    adapter = ProblemScientificAdapter(problem)
    mutate(problem)

    with pytest.raises(VerificationServiceError, match="frozen identity"):
        _verify(
            adapter=adapter,
            proposal=_proposal(problem_id="evolve_toy"),
            payload=_payload([3, -2], problem_id="evolve_toy"),
        )
    assert calls == {"verify": 0, "describe": 0, "fingerprint": 0}


@pytest.mark.parametrize(
    ("flags", "kind", "resolved", "excluded"),
    [
        ({"parsed": False}, FailureKind.PARSE, True, False),
        ({"executed": False}, FailureKind.CODE, True, False),
        ({"constraints_satisfied": False}, FailureKind.CONSTRAINT, True, False),
        ({"scientifically_valid": False}, FailureKind.SCIENTIFIC, True, False),
        ({"timed_out": True}, FailureKind.TIMEOUT, False, True),
        (
            {"timed_out": True, "timeout_is_scientific": True},
            FailureKind.TIMEOUT,
            True,
            False,
        ),
        (
            {"infrastructure_error": True, "timed_out": True},
            FailureKind.INFRASTRUCTURE,
            False,
            True,
        ),
    ],
)
def test_failure_classification_is_structured_and_timeout_policy_is_explicit(
    flags, kind, resolved, excluded
):
    classified = classify_failure(**flags)

    assert classified.failure_kind == kind
    assert classified.resolved is resolved
    assert classified.excluded_from_scientific_updates is excluded


def test_success_builds_exact_content_addressed_evidence_state_and_descriptor():
    proposal = _proposal(parent_state_id=_identifier("state", "parent"))
    persisted = _payload()
    adapter = _FakeAdapter()
    result = _verify(adapter=adapter, proposal=proposal, payload=persisted)

    assert result.evidence.run_id == proposal.run_id
    assert result.evidence.proposal_id == proposal.proposal_id
    assert result.evidence.branch_id == proposal.branch_id
    assert result.evidence.parent_state_id == proposal.parent_state_id
    assert result.evidence.source_hash == proposal.source_hash
    assert result.evidence.answer_payload == persisted.payload
    assert result.evidence.flags["answer_payload_id"] == persisted.payload_id
    assert result.evidence.flags["excluded_from_scientific_updates"] is False
    assert result.evidence.confirmed is False
    assert result.state.state_id == content_id(
        "state",
        {"problem_id": "toy", "answer_payload": {"point": [3, -2]}},
    )
    assert result.state.state_id == scientific_state_id("toy", persisted.payload)
    assert result.state.proposal_id == proposal.proposal_id
    assert result.state.evidence_id == result.evidence.evidence_id
    assert result.descriptor.descriptor_id == result.evidence.descriptor_id
    assert tuple(result.evidence.lineage_ids) == (
        proposal.parent_state_id,
        result.state.state_id,
    )
    validate_evidence_identity(result.evidence)
    validate_state_identity(result.state)


def test_state_identity_ignores_source_proposal_evidence_and_confirmation_context():
    payload = _payload()
    first = _verify(
        proposal=_proposal(source="program A", proposal_label="a", branch_label="a"),
        payload=payload,
    )
    second = _verify(
        proposal=_proposal(source="program B", proposal_label="b", branch_label="b"),
        payload=payload,
    )

    assert first.state.state_id == second.state.state_id
    assert first.evidence.evidence_id != second.evidence.evidence_id
    assert first.state.evidence_id != second.state.evidence_id


def test_evidence_identity_covers_every_observation_specific_input():
    base_proposal = _proposal()
    base_payload = _payload()
    base_policy = VerificationPolicy.create(version="common_v1")
    baseline = _verify(
        adapter=_FakeAdapter([_success()]),
        proposal=base_proposal,
        payload=base_payload,
        policy=base_policy,
    )
    variants = [
        _verify(
            adapter=_FakeAdapter([_success()]),
            proposal=_proposal(proposal_label="proposal-b"),
            payload=base_payload,
            policy=base_policy,
        ),
        _verify(
            adapter=_FakeAdapter([_success()]),
            proposal=_proposal(branch_label="branch-b"),
            payload=base_payload,
            policy=base_policy,
        ),
        _verify(
            adapter=_FakeAdapter([_success()], label="verifier-b"),
            proposal=base_proposal,
            payload=base_payload,
            policy=base_policy,
        ),
        _verify(
            adapter=_FakeAdapter([_success()]),
            proposal=base_proposal,
            payload=base_payload,
            policy=base_policy,
            harness_id=_identifier("harness", "diagnostic-v2"),
        ),
        _verify(
            adapter=_FakeAdapter([_success()]),
            proposal=base_proposal,
            payload=base_payload,
            policy=base_policy,
            policy_snapshot_id=_identifier("role_snapshot", "scout-epoch1"),
        ),
        _verify(
            adapter=_FakeAdapter([_success(diagnostics={"variant": 2})]),
            proposal=base_proposal,
            payload=base_payload,
            policy=base_policy,
        ),
        _verify(
            adapter=_FakeAdapter([_success(resources={"cpu_seconds": 0.02})]),
            proposal=base_proposal,
            payload=base_payload,
            policy=base_policy,
        ),
        _verify(
            adapter=_FakeAdapter([_success()]),
            proposal=base_proposal,
            payload=base_payload,
            policy=VerificationPolicy.create(version="common_v2"),
        ),
        _verify(
            adapter=_FakeAdapter([_success()]),
            proposal=base_proposal,
            payload=_payload({"point": [2, -2]}),
            policy=base_policy,
        ),
    ]

    evidence_ids = {baseline.evidence.evidence_id}
    evidence_ids.update(item.evidence.evidence_id for item in variants)
    assert len(evidence_ids) == 1 + len(variants)


def test_diagnostics_are_strictly_bounded_and_retain_full_capture_hash():
    diagnostics = {f"entry_{index}": "x" * 400 for index in range(10)}
    policy = VerificationPolicy.create(
        version="bounded_v1",
        max_diagnostic_chars=300,
        max_diagnostic_entries=2,
    )
    result = _verify(
        adapter=_FakeAdapter([_success(diagnostics=diagnostics)]),
        policy=policy,
    )

    assert len(canonical_json(result.evidence.diagnostics)) <= 300
    assert result.evidence.diagnostics["_bounded"]["truncated"] is True
    assert result.evidence.diagnostics["_bounded"]["original_sha256"] == content_hash(
        diagnostics
    )
    assert result.evidence.resources["verifier_calls"] == 1


def test_service_fills_absent_capture_with_utc_timestamps_and_wall_time():
    decision = VerificationDecision.success(
        internal_reward=1.0,
        raw_score=0.0,
        capture=ExecutionCapture(diagnostics={"verdict": "valid"}),
    )

    result = _verify(adapter=_FakeAdapter([decision]))
    capture = result.decision.capture

    assert capture.started_at.endswith("Z")
    assert capture.completed_at.endswith("Z")
    started = datetime.fromisoformat(capture.started_at.replace("Z", "+00:00"))
    completed = datetime.fromisoformat(
        capture.completed_at.replace("Z", "+00:00")
    )
    assert started.tzinfo == timezone.utc
    assert completed.tzinfo == timezone.utc
    assert completed >= started
    assert capture.resources["verifier_calls"] == 1
    assert capture.resources["verifier_wall_time_s"] >= 0.0
    assert result.evidence.started_at == capture.started_at
    assert result.evidence.completed_at == capture.completed_at
    assert (
        result.evidence.resources["verifier_wall_time_s"]
        == capture.resources["verifier_wall_time_s"]
    )


def test_service_preserves_complete_explicit_capture():
    decision = _success(
        diagnostics={"verdict": "explicit"},
        resources={"cpu_seconds": 0.25, "verifier_calls": 7},
    )

    result = _verify(adapter=_FakeAdapter([decision]))

    assert result.decision.capture is decision.capture
    assert result.decision.capture.started_at == "2026-01-01T00:00:00Z"
    assert result.decision.capture.completed_at == "2026-01-01T00:00:01Z"
    assert dict(result.decision.capture.resources) == {
        "cpu_seconds": 0.25,
        "verifier_calls": 7,
    }


def test_adapter_exception_is_unresolved_infrastructure_evidence_not_low_reward():
    adapter = _FakeAdapter([RuntimeError("worker disappeared")])
    result = _verify(adapter=adapter)

    assert result.evidence.failure_kind == FailureKind.INFRASTRUCTURE
    assert result.evidence.resolved is False
    assert result.evidence.admitted is False
    assert result.evidence.internal_reward is None
    assert result.evidence.confirmed is False
    assert result.evidence.flags["excluded_from_scientific_updates"] is True
    assert result.state is None
    assert result.descriptor is None
    assert result.evidence.diagnostics["adapter_error"]["phase"] == "verify_answer_payload"


def test_service_does_not_hide_retries_inside_one_evidence_attempt():
    adapter = _FakeAdapter([RuntimeError("single durable failure")])
    policy = VerificationPolicy.create(
        version="external-retries-v1", infrastructure_retry_limit=5
    )

    result = _verify(adapter=adapter, policy=policy)

    assert len(adapter.calls) == 1
    assert result.evidence.failure_kind == FailureKind.INFRASTRUCTURE
    assert result.evidence.flags["verification_attempt_index"] == 0


def test_null_payload_cannot_be_admitted_as_a_scientific_state():
    result = _verify(payload=_payload(None), adapter=_FakeAdapter([_success()]))

    assert result.evidence.failure_kind == FailureKind.INFRASTRUCTURE
    assert result.evidence.admitted is False
    assert result.state is None


def test_descriptor_contract_failure_is_infrastructure_and_never_admitted():
    adapter = _FakeAdapter([_success()], descriptor=RuntimeError("descriptor failed"))
    result = _verify(adapter=adapter)

    assert result.evidence.failure_kind == FailureKind.INFRASTRUCTURE
    assert result.evidence.resolved is False
    assert result.state is None
    assert result.evidence.flags["adapter_contract_phase"] == "describe_verified_state"


@pytest.mark.parametrize("timeout_is_scientific", [False, True])
def test_problem_timeout_policy_controls_resolution_and_scientific_exclusion(
    timeout_is_scientific
):
    adapter = _FakeAdapter(
        [
            VerificationDecision.failure(
                classify_failure(
                    timed_out=True,
                    timeout_is_scientific=timeout_is_scientific,
                )
            )
        ]
    )
    adapter.timeout_is_scientific = timeout_is_scientific
    result = _verify(adapter=adapter)

    assert result.evidence.failure_kind == FailureKind.TIMEOUT
    assert result.evidence.resolved is timeout_is_scientific
    assert result.evidence.flags["excluded_from_scientific_updates"] is (
        not timeout_is_scientific
    )
    assert result.evidence.internal_reward is None
    assert result.state is None


def test_timeout_decision_that_contradicts_problem_policy_becomes_infrastructure():
    adapter = _FakeAdapter(
        [VerificationDecision.failure(classify_failure(timed_out=True))]
    )
    adapter.timeout_is_scientific = True
    result = _verify(adapter=adapter)

    assert result.evidence.failure_kind == FailureKind.INFRASTRUCTURE
    assert result.evidence.flags["excluded_from_scientific_updates"] is True


def test_confirmation_reverifies_only_saved_payload_and_preserves_state_identity():
    adapter = _FakeAdapter([_success(reward=0.9), _success(reward=1.0)])
    proposal = _proposal(source="def rerun(): raise AssertionError('never')")
    persisted = _payload()
    policy = VerificationPolicy.create(version="common_v1")
    initial = _verify(
        adapter=adapter,
        proposal=proposal,
        payload=persisted,
        policy=policy,
    )
    confirmed = confirm_persisted_answer(
        adapter=adapter,
        proposal=proposal,
        persisted_answer=persisted,
        prior_evidence=initial.evidence,
        verification_policy=policy,
    )

    assert [call[0] for call in adapter.calls] == [
        {"point": [3, -2]},
        {"point": [3, -2]},
    ]
    assert confirmed.evidence.confirmed is True
    assert confirmed.state.confirmed is True
    assert confirmed.state.state_id == initial.state.state_id
    assert confirmed.evidence.evidence_id != initial.evidence.evidence_id
    assert confirmed.evidence.flags["confirmation_of_evidence_id"] == initial.evidence.evidence_id
    assert confirmed.evidence.harness_id == initial.evidence.harness_id
    assert confirmed.evidence.policy_snapshot_id == initial.evidence.policy_snapshot_id


def test_confirmation_revalidates_frozen_problem_identity_before_hooks():
    problem = EvolveToyProblem({})
    calls = _count_problem_verifier_hooks(problem)
    adapter = ProblemScientificAdapter(problem)
    proposal = _proposal(problem_id="evolve_toy")
    persisted = _payload([3, -2], problem_id="evolve_toy")
    policy = VerificationPolicy.create(version="common_v1")
    initial = _verify(
        adapter=adapter,
        proposal=proposal,
        payload=persisted,
        policy=policy,
    )
    assert calls == {"verify": 1, "describe": 1, "fingerprint": 1}
    problem.target = -123.0

    with pytest.raises(VerificationServiceError, match="frozen identity"):
        confirm_persisted_answer(
            adapter=adapter,
            proposal=proposal,
            persisted_answer=persisted,
            prior_evidence=initial.evidence,
            verification_policy=policy,
        )
    assert calls == {"verify": 1, "describe": 1, "fingerprint": 1}


def test_confirmation_accepts_relocated_exact_artifact(tmp_path):
    adapter = _FakeAdapter([_success(), _success()])
    proposal = _proposal()
    payload = {"point": [3, -2]}
    original = _payload(payload, artifact=tmp_path / "original" / "answer.json")
    policy = VerificationPolicy.create(version="common_v1")
    initial = _verify(
        adapter=adapter,
        proposal=proposal,
        payload=original,
        policy=policy,
    )
    relocated = _payload(payload, artifact=tmp_path / "copied" / "answer.json")
    Path(original.artifact_uri).unlink()

    confirmed = confirm_persisted_answer(
        adapter=adapter,
        proposal=proposal,
        persisted_answer=relocated,
        prior_evidence=initial.evidence,
        verification_policy=policy,
    )

    assert relocated.payload_id == original.payload_id
    assert relocated.payload_hash == original.payload_hash
    assert relocated.artifact_uri != original.artifact_uri
    assert confirmed.state.state_id == initial.state.state_id
    assert confirmed.evidence.flags["answer_artifact_uri"] == relocated.artifact_uri
    assert (
        confirmed.evidence.flags["confirmation_of_evidence_id"]
        == initial.evidence.evidence_id
    )


@pytest.mark.parametrize("mutation", ["delete", "tamper"])
def test_confirmation_rejects_deleted_or_tampered_artifact_before_adapter_call(
    tmp_path, mutation
):
    adapter = _FakeAdapter([_success(), _success()])
    proposal = _proposal()
    persisted = _payload(artifact=tmp_path / "answer.json")
    policy = VerificationPolicy.create(version="common_v1")
    initial = _verify(
        adapter=adapter,
        proposal=proposal,
        payload=persisted,
        policy=policy,
    )
    artifact = Path(persisted.artifact_uri)
    if mutation == "delete":
        artifact.unlink()
    else:
        artifact.write_text('{"point": [0, 0]}\n', encoding="utf-8")

    with pytest.raises(VerificationServiceError, match="durable answer artifact"):
        confirm_persisted_answer(
            adapter=adapter,
            proposal=proposal,
            persisted_answer=persisted,
            prior_evidence=initial.evidence,
            verification_policy=policy,
        )
    assert len(adapter.calls) == 1


def test_initial_verification_rechecks_artifact_before_adapter_call(tmp_path):
    adapter = _FakeAdapter([_success()])
    persisted = _payload(artifact=tmp_path / "answer.json")
    Path(persisted.artifact_uri).write_text(
        '{"point": [99, 99]}\n', encoding="utf-8"
    )

    with pytest.raises(VerificationServiceError, match="does not match"):
        _verify(adapter=adapter, payload=persisted)
    assert adapter.calls == []


def test_confirmation_rejects_payload_or_reference_substitution_before_adapter_call():
    adapter = _FakeAdapter([_success()])
    proposal = _proposal()
    initial = _verify(adapter=adapter, proposal=proposal)

    with pytest.raises(VerificationServiceError, match="different scientific state"):
        confirm_persisted_answer(
            adapter=adapter,
            proposal=proposal,
            persisted_answer=_payload({"point": [0, 0]}),
            prior_evidence=initial.evidence,
            verification_policy=VerificationPolicy.create(version="common_v1"),
        )
    assert len(adapter.calls) == 1

    with pytest.raises(VerificationServiceError, match="proposal_id"):
        confirm_persisted_answer(
            adapter=adapter,
            proposal=_proposal(proposal_label="substitute"),
            persisted_answer=_payload(),
            prior_evidence=initial.evidence,
            verification_policy=VerificationPolicy.create(version="common_v1"),
        )
    assert len(adapter.calls) == 1


def test_failed_confirmation_is_durable_unconfirmed_evidence_without_a_state():
    scientific_failure = VerificationDecision.failure(
        classify_failure(scientifically_valid=False),
        raw_score=99,
        capture=ExecutionCapture(diagnostics={"counterexample": "found"}),
    )
    adapter = _FakeAdapter([_success(), scientific_failure])
    proposal = _proposal()
    persisted = _payload()
    initial = _verify(adapter=adapter, proposal=proposal, payload=persisted)
    failed = confirm_persisted_answer(
        adapter=adapter,
        proposal=proposal,
        persisted_answer=persisted,
        prior_evidence=initial.evidence,
        verification_policy=VerificationPolicy.create(version="common_v1"),
    )

    assert failed.evidence.failure_kind == FailureKind.SCIENTIFIC
    assert failed.evidence.resolved is True
    assert failed.evidence.confirmed is False
    assert failed.evidence.flags["confirmation_target_state_id"] == initial.state.state_id
    assert failed.state is None


def test_problem_hook_adapter_converts_neutral_result_without_controller_imports():
    class DuckProblem:
        name = "duck"
        answer_schema_version = 2
        descriptor_function_version = "duck_descriptor_v3"
        fingerprint_function_version = "duck_fingerprint_v1"
        scientific_method_complete = True

        def __init__(self):
            self.seen = []

        def resource_requirements(self):
            return SimpleNamespace(timeout_is_scientific=False)

        def verify_answer_payload(self, payload, policy):
            self.seen.append((payload, policy))
            return SimpleNamespace(
                resolved=True,
                admitted=True,
                answer_payload=payload,
                internal_reward=0.75,
                raw_score={"distance": 2},
                failure_kind="",
                message="saved payload verified",
                uncertainty=0.0,
                scores={"distance": 2},
                flags={"payload_only": True, "method_complete": True},
                diagnostics={"check": "ok"},
            )

        def describe_scientific_state(self, candidate, evidence):
            assert candidate == evidence["answer_payload"]
            return {"cell": "duck-cell"}

        def scientific_fingerprint(self, candidate, evidence):
            assert candidate == evidence["answer_payload"]
            return "duck:fingerprint"

    problem = DuckProblem()
    adapter = ProblemScientificAdapter(problem, problem_id="toy")
    result = _verify(adapter=adapter)

    assert result.evidence.admitted is True
    assert result.evidence.verifier_version == "answer_schema_v2"
    assert result.descriptor.function_version == "duck_descriptor_v3"
    assert result.evidence.fingerprint == "duck:fingerprint"
    assert problem.seen[0][0] == {"point": [3, -2]}
    assert problem.seen[0][1]["policy_id"].startswith("verification_policy:")


def test_problem_hook_adapter_rejects_admission_without_exact_payload_echo():
    class MissingPayloadProblem:
        name = "toy"
        answer_schema_version = 1
        descriptor_function_version = "missing_payload_descriptor_v1"
        fingerprint_function_version = "missing_payload_fingerprint_v1"
        scientific_method_complete = True

        def resource_requirements(self):
            return SimpleNamespace(timeout_is_scientific=False)

        def verify_answer_payload(self, payload, policy):
            del payload, policy
            return SimpleNamespace(
                resolved=True,
                admitted=True,
                answer_payload=None,
                internal_reward=1.0,
                raw_score=0.0,
                failure_kind="",
                message="",
                uncertainty=0.0,
                scores={},
                flags={"method_complete": True},
                diagnostics={},
            )

        def describe_scientific_state(self, candidate, evidence):
            del candidate, evidence
            return {"cell": "unreachable"}

        def scientific_fingerprint(self, candidate, evidence):
            del candidate, evidence
            return "unreachable"

    result = _verify(adapter=ProblemScientificAdapter(MissingPayloadProblem()))

    assert result.evidence.failure_kind == FailureKind.INFRASTRUCTURE
    assert result.evidence.resolved is False
    assert result.evidence.flags["excluded_from_scientific_updates"] is True
    assert result.state is None
    assert result.evidence.diagnostics["adapter_error"]["phase"] == "verify_answer_payload"


def test_tampered_content_ids_and_state_ids_are_detected_on_read():
    result = _verify()
    packet = result.evidence.to_dict()
    packet["diagnostics"] = {"tampered": True}
    tampered = EvidencePacket.from_dict(packet)

    with pytest.raises(VerificationValidationError, match="evidence_id"):
        validate_evidence_identity(tampered)

    wrong_state = replace(result.state, state_id=_identifier("state", "wrong"))
    with pytest.raises(VerificationValidationError, match="state_id"):
        validate_state_identity(wrong_state)
