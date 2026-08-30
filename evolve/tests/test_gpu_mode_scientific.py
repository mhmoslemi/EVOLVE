"""CPU-only characterization of GPU mode's saved-answer verifier.

Every evaluator entry point is replaced before a test can invoke it.  These
tests validate the scientific contract without importing torch/triton or
creating a CUDA context.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict

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
from evolve.verifier.evidence import scientific_state_id
from problems import gpu_mode
from problems.gpu_mode import GpuMode


_KERNEL = """import triton
import triton.language as tl

@triton.jit
def fused_kernel(x, BLOCK: tl.constexpr):
    offsets = tl.arange(0, BLOCK)
    values = tl.load(x + offsets)
    tl.store(x + offsets, values)

def custom_kernel(data):
    fused_kernel[(1,)](data, BLOCK=128)
    return data
"""


@pytest.fixture(autouse=True)
def _forbid_real_gpu_execution(monkeypatch: pytest.MonkeyPatch) -> None:
    def forbidden(*args: Any, **kwargs: Any) -> Any:
        raise AssertionError("a real GPU evaluator was invoked by a CPU-only test")

    monkeypatch.setattr(gpu_mode, "run_eval_with_timeout", forbidden)
    monkeypatch.setattr(gpu_mode, "_eval_child", forbidden)


@pytest.fixture
def problem_factory(monkeypatch: pytest.MonkeyPatch):
    # The repository task trees are read to content-address the frozen answer,
    # but runtime availability checks are isolated from the machine under test.
    monkeypatch.setattr(GpuMode, "missing_task_files", lambda self: [])

    def make(problem_type: str = "trimul", **overrides: Any) -> GpuMode:
        defaults: Dict[str, Any] = {
            "problem_type": problem_type,
            "gpu_type": "H100" if problem_type == "trimul" else "H200",
            "score_scale": 1500.0 if problem_type == "trimul" else 5000.0,
            "target": 1000.0 if problem_type == "trimul" else 1700.0,
            "kernel_timeout_s": 7.0,
            "kernel_gpu_id": 3,
            "kernel_log_chars": 100,
            "seed_from_reference": False,
        }
        defaults.update(overrides)
        return GpuMode(defaults)

    return make


@pytest.mark.parametrize("problem_type", ["trimul", "mla_decode_nvidia"])
def test_payload_freezes_kernel_task_hardware_and_evaluator(
    problem_factory: Any,
    problem_type: str,
) -> None:
    problem = problem_factory(problem_type)
    first = problem.serialize_answer(_KERNEL)
    second = problem.serialize_answer(_KERNEL)

    assert first == second
    assert json.dumps(first, sort_keys=True, allow_nan=False)
    assert first["kernel_source"] == _KERNEL
    assert first["task"]["problem_type"] == problem_type
    assert len(first["task"]["manifest_sha256"]) == 64
    assert len(first["task"]["bundle_sha256"]) == 64
    assert first["task"]["files"]
    assert first["hardware"] == {
        "declared_gpu_type": "h100" if problem_type == "trimul" else "h200",
        "triton_version": "3.3.1",
        "kernel_gpu_id": 3,
        "exclusive_evaluation": True,
    }
    assert first["evaluator"]["version"] == "libkernelbot_leaderboard_timeout_v1"
    assert first["evaluator"]["submission_mode"] == "leaderboard"
    assert first["evaluator"]["timeout_s"] == 7.0
    assert first["evaluator"]["files"]
    assert len(first["evaluator"]["libkernelbot_bundle_sha256"]) == 64
    assert len(first["evaluator"]["wrapper_protocol_sha256"]) == 64

    assert content_hash(first) == content_hash(second)
    assert (
        scientific_state_id(problem.name, first)
        == scientific_state_id(problem.name, second)
    )


def test_subtypes_have_distinct_scientific_state_identity(problem_factory: Any) -> None:
    trimul = problem_factory("trimul").serialize_answer(_KERNEL)
    mla = problem_factory("mla_decode_nvidia").serialize_answer(_KERNEL)
    assert scientific_state_id("gpu_mode", trimul) != scientific_state_id(
        "gpu_mode", mla
    )


@pytest.mark.parametrize("problem_type", ["trimul", "mla_decode_nvidia"])
def test_saved_payload_replay_uses_only_timeout_runner_and_recomputes_reward(
    monkeypatch: pytest.MonkeyPatch,
    problem_factory: Any,
    problem_type: str,
) -> None:
    problem = problem_factory(problem_type)
    payload = problem.serialize_answer(_KERNEL)
    calls = []

    def fake_runner(*args: Any) -> MappingResult:
        calls.append(args)
        return {
            "ok": True,
            "score_us": 250.0,
            "msg": "runtime_us=250.0",
            "logs": "warning: mocked diagnostic",
        }

    monkeypatch.setattr(gpu_mode, "run_eval_with_timeout", fake_runner)
    # Proposal parsing, preprocessing, and the legacy reward path are forbidden
    # after capture. Confirmation must replay the persisted source directly.
    monkeypatch.setattr(
        gpu_mode, "extract_python_code",
        lambda value: (_ for _ in ()).throw(AssertionError("proposal reparsed")),
    )
    monkeypatch.setattr(
        problem, "compute_reward",
        lambda *args, **kwargs: (_ for _ in ()).throw(
            AssertionError("legacy reward path invoked")
        ),
    )

    result = problem.verify_answer_payload(
        payload, {"max_diagnostic_chars": 1000}
    )
    assert result.resolved is True
    assert result.admitted is True
    assert result.answer_payload == payload
    assert result.raw_score == 250.0
    assert result.internal_reward == pytest.approx(problem.score_scale / 250.0)
    assert result.flags["proposal_replay"] is False
    assert result.flags["noisy_runtime"] is True
    assert result.uncertainty == 0.0
    assert result.scores["verification_repeats"] == 3
    assert len(calls) == 3
    assert calls[0][0] == payload["kernel_source"]
    assert calls[0][1] == payload["evaluator"]["lib_dir"]
    assert calls[0][2] == payload["task"]["task_yaml"]
    assert calls[0][3] == problem_type
    assert calls[0][5] == 7.0
    assert calls[0][6] == 3


# Python 3.9 cannot spell this as a PEP 604 alias; keep tests on the same
# compatibility floor as the production problem module.
MappingResult = Dict[str, Any]


@pytest.mark.parametrize(
    "runner_result, expected_kind, expected_resolved, expected_classification",
    [
        (
            {
                "ok": False,
                "score_us": None,
                "msg": "Failed to pass test cases.",
                "logs": "mismatch at case 2",
            },
            "constraint",
            True,
            "correctness",
        ),
        (
            {
                "ok": False,
                "score_us": None,
                "msg": "kernel_eval_timeout after 7s",
                "logs": "",
            },
            "timeout",
            False,
            "timeout",
        ),
        (
            {
                "ok": False,
                "score_us": None,
                "msg": "Local kernel run failed: CUDA unavailable",
                "logs": "",
            },
            "infrastructure",
            False,
            "infrastructure",
        ),
        (
            {
                "ok": False,
                "score_us": None,
                "msg": "Error: compilation failed",
                "logs": "[test] COMPILE FAILED",
            },
            "code",
            True,
            "compilation_or_execution",
        ),
    ],
)
def test_runner_failures_are_distinct(
    monkeypatch: pytest.MonkeyPatch,
    problem_factory: Any,
    runner_result: MappingResult,
    expected_kind: str,
    expected_resolved: bool,
    expected_classification: str,
) -> None:
    problem = problem_factory()
    payload = problem.serialize_answer(_KERNEL)
    monkeypatch.setattr(
        gpu_mode, "run_eval_with_timeout", lambda *args: runner_result
    )
    result = problem.verify_answer_payload(payload)
    assert result.admitted is False
    assert result.failure_kind == expected_kind
    assert result.resolved is expected_resolved
    assert result.flags["failure_classification"] == expected_classification


def test_syntax_failure_is_detected_without_any_evaluator(
    problem_factory: Any,
) -> None:
    problem = problem_factory()
    payload = problem.serialize_answer("@triton.jit\ndef broken(:\n")
    result = problem.verify_answer_payload(payload)
    assert result.resolved is True
    assert result.admitted is False
    assert result.failure_kind == "parse"
    assert result.flags["failure_classification"] == "syntax"


def test_frozen_configuration_mismatch_is_unresolved_infrastructure(
    problem_factory: Any,
) -> None:
    original = problem_factory(gpu_type="H100", kernel_gpu_id=3)
    payload = original.serialize_answer(_KERNEL)
    different = problem_factory(gpu_type="A100", kernel_gpu_id=4)

    result = different.verify_answer_payload(payload)
    assert result.resolved is False
    assert result.admitted is False
    assert result.failure_kind == "infrastructure"
    assert result.flags["failure_classification"] == "frozen_context_mismatch"
    assert result.answer_payload == payload


def test_malformed_payload_is_a_resolved_parse_failure(problem_factory: Any) -> None:
    problem = problem_factory()
    payload = problem.serialize_answer(_KERNEL)
    payload.pop("task")
    result = problem.verify_answer_payload(payload)
    assert result.resolved is True
    assert result.failure_kind == "parse"
    assert result.answer_payload is None


def test_descriptor_and_fingerprint_are_stable_verified_features(
    monkeypatch: pytest.MonkeyPatch,
    problem_factory: Any,
) -> None:
    problem = problem_factory()
    payload = problem.serialize_answer(_KERNEL)
    monkeypatch.setattr(
        gpu_mode,
        "run_eval_with_timeout",
        lambda *args: {
            "ok": True,
            "score_us": 700.0,
            "msg": "runtime_us=700",
            "logs": "",
        },
    )
    first = problem.verify_answer_payload(payload)
    second = problem.verify_answer_payload(payload)
    descriptor_a = problem.describe_scientific_state(payload, first)
    descriptor_b = problem.describe_scientific_state(payload, second)
    fingerprint_a = problem.scientific_fingerprint(payload, first)
    fingerprint_b = problem.scientific_fingerprint(payload, second)

    assert descriptor_a == descriptor_b
    assert descriptor_a["problem_type"] == "trimul"
    assert descriptor_a["declared_gpu_type"] == "h100"
    assert descriptor_a["kernel_family"] == "triton_memory"
    assert descriptor_a["performance_bin"] == "well_below_target"
    assert descriptor_a["diagnostic_bin"] == "clean"
    assert descriptor_a["task_bundle"] == payload["task"]["bundle_sha256"]
    assert fingerprint_a == fingerprint_b
    assert len(fingerprint_a) == 64


def test_common_verifier_adapter_accepts_production_gpu_contract(
    monkeypatch: pytest.MonkeyPatch,
    problem_factory: Any,
) -> None:
    problem = problem_factory()
    payload = problem.serialize_answer(_KERNEL)
    monkeypatch.setattr(
        gpu_mode,
        "run_eval_with_timeout",
        lambda *args: {
            "ok": True,
            "score_us": 800.0,
            "msg": "runtime_us=800",
            "logs": "",
        },
    )
    adapter = ProblemScientificAdapter(problem)
    policy = VerificationPolicy.create(version="gpu_cpu_fake_v1", production=True)
    decision = adapter.verify_answer_payload(payload, policy)

    assert adapter.method_complete is True
    assert adapter.timeout_is_scientific is False
    assert decision.admitted is True
    assert decision.internal_reward == pytest.approx(1500.0 / 800.0)
    assert adapter.describe_scientific_state(payload, decision)[
        "performance_bin"
    ] == "near_target"
    assert len(adapter.scientific_fingerprint(payload, decision)) == 64


def test_fake_gpu_pipeline_verifies_confirms_archives_and_records_saved_source(
    monkeypatch: pytest.MonkeyPatch,
    problem_factory: Any,
    tmp_path: Path,
) -> None:
    """Exercise the production pipeline without creating a CUDA context."""

    problem = problem_factory()
    payload = problem.serialize_answer(_KERNEL)
    monkeypatch.setattr(
        gpu_mode,
        "run_eval_with_timeout",
        lambda *args: {
            "ok": True,
            "score_us": 750.0,
            "msg": "runtime_us=750",
            "logs": "",
        },
    )
    artifact = tmp_path / "gpu-answer.json"
    artifact.write_text(
        json.dumps(payload, allow_nan=False, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    persisted = PersistedAnswerPayload.create(
        problem_id=problem.name,
        artifact_uri=str(artifact),
        payload=payload,
    )
    source = "# source text is provenance, not scientific identity\n" + _KERNEL
    proposal = Proposal(
        proposal_id=content_id("proposal", {"case": "gpu-fake-pipeline"}),
        run_id=content_id("run", {"case": "gpu-fake-pipeline"}),
        problem_id=problem.name,
        source_text=source,
        source_hash=content_hash(source),
        branch_id=content_id("branch", {"case": "gpu-fake-pipeline"}),
        parsed_candidate={"captured": True},
    )
    adapter = ProblemScientificAdapter(problem)
    policy = VerificationPolicy.create(
        version="gpu-fake-pipeline-v1", production=True
    )
    initial = verify_persisted_answer(
        adapter=adapter,
        proposal=proposal,
        persisted_answer=persisted,
        verification_policy=policy,
        harness_id=content_id("harness", {"version": "baseline-v1"}),
        policy_snapshot_id=content_id(
            "role_snapshot", {"role": "scout", "epoch": 0}
        ),
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
    assert confirmed.state.raw_score == pytest.approx(750.0)

    archive, _ = ScientificArchive().offer(
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
    assert record.raw_score == pytest.approx(750.0)


def test_resources_and_renderer_never_reexecute_kernel(
    monkeypatch: pytest.MonkeyPatch,
    problem_factory: Any,
    tmp_path: Path,
) -> None:
    problem = problem_factory(kernel_timeout_s=11.0)
    payload = problem.serialize_answer(_KERNEL)
    resources = problem.resource_requirements()
    assert resources.gpu_count == 1
    assert resources.exclusive_gpu is True
    assert resources.timeout_s == 11.0
    assert resources.timeout_is_scientific is False
    assert resources.network_access is False

    evidence = {"answer_payload": payload, "raw_score": 321.5}
    files = problem.render_best(None, evidence, tmp_path)
    assert [Path(item).name for item in files] == [
        "answer.py", "answer.json", "answer.txt"
    ]
    assert (tmp_path / "answer.py").read_text(encoding="utf-8") == _KERNEL
    assert json.loads((tmp_path / "answer.json").read_text()) == payload
    assert "runtime_us: 321.5" in (tmp_path / "answer.txt").read_text()


def test_missing_task_files_and_runner_exceptions_remain_infrastructure(
    monkeypatch: pytest.MonkeyPatch,
    problem_factory: Any,
) -> None:
    problem = problem_factory()
    payload = problem.serialize_answer(_KERNEL)
    monkeypatch.setattr(
        problem, "missing_task_files", lambda: ["missing/eval.py"]
    )
    missing = problem.verify_answer_payload(payload)
    assert missing.failure_kind == "infrastructure"
    assert missing.resolved is False

    monkeypatch.setattr(problem, "missing_task_files", lambda: [])
    monkeypatch.setattr(
        gpu_mode,
        "run_eval_with_timeout",
        lambda *args: (_ for _ in ()).throw(RuntimeError("worker vanished")),
    )
    crashed = problem.verify_answer_payload(payload)
    assert crashed.failure_kind == "infrastructure"
    assert crashed.resolved is False
    assert crashed.flags["failure_classification"] == "runner_exception"
