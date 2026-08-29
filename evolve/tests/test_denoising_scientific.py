import copy
import json
import math
from pathlib import Path

import numpy as np
import pytest

from evolve.verifier import ProblemScientificAdapter, VerificationPolicy
from problems.denoising import (
    BASELINES,
    POISSON_NORM_MIN,
    Denoising,
    verify_denoising,
)


def _constraint_boundary() -> float:
    baseline = BASELINES["pancreas"]
    span = baseline["baseline_poisson"] - baseline["perfect_poisson"]
    return baseline["baseline_poisson"] - POISSON_NORM_MIN * span


def _valid_output(mse: float = 0.2):
    return (mse, _constraint_boundary())


def test_denoising_serializes_trusted_evaluator_output_as_canonical_envelope():
    problem = Denoising({"eval_seed": 17})

    envelope = problem.serialize_answer((*_valid_output(), "ignored future tail"))

    assert problem.scientific_method_complete is True
    assert envelope == {
        "schema_version": 1,
        "problem": "denoising",
        "dataset": "pancreas",
        "eval_seed": 17,
        "evaluator_version": "pancreas_holdout_mse_poisson_v1",
        "metrics": {
            "mse": 0.2,
            "poisson": _constraint_boundary(),
        },
    }
    # The persisted representation is strict finite JSON.
    assert json.loads(json.dumps(envelope, allow_nan=False)) == envelope


def test_denoising_saved_envelope_verification_is_deterministic_and_higher_is_better():
    problem = Denoising({"eval_seed": 42})
    payload = problem.serialize_answer(_valid_output(0.2))

    first = problem.verify_answer_payload(payload)
    second = problem.verify_answer_payload(copy.deepcopy(payload))
    better = problem.verify_answer_payload(
        problem.serialize_answer(_valid_output(0.1))
    )

    assert first == second
    assert first.resolved is True
    assert first.admitted is True
    assert first.answer_payload == payload
    assert first.raw_score == 0.2
    assert first.internal_reward == 5.0
    assert first.scores["poisson_norm"] == pytest.approx(POISSON_NORM_MIN)
    assert first.flags["trusted_evaluator_capture_required"] is True
    assert better.raw_score < first.raw_score
    assert better.internal_reward > first.internal_reward


def test_denoising_contract_is_accepted_by_production_scientific_adapter():
    problem = Denoising({})
    payload = problem.serialize_answer(_valid_output())
    adapter = ProblemScientificAdapter(problem)

    decision = adapter.verify_answer_payload(
        payload,
        VerificationPolicy.create(version="denoising_test_v1", production=True),
    )

    assert adapter.method_complete is True
    assert decision.admitted is True
    assert decision.raw_score == 0.2
    assert decision.internal_reward == 5.0


def test_denoising_poisson_constraint_accepts_boundary_and_rejects_either_side():
    problem = Denoising({})
    baseline = BASELINES["pancreas"]
    boundary = _constraint_boundary()

    accepted = problem.verify_answer_payload((0.2, boundary))
    misses_target = problem.verify_answer_payload(
        (0.2, boundary + 1.0e-15)
    )
    below_perfect = problem.verify_answer_payload(
        (0.2, np.nextafter(baseline["perfect_poisson"], -math.inf))
    )

    assert accepted.admitted is True
    assert misses_target.admitted is False
    assert misses_target.failure_kind == "constraint"
    assert "hard constraint" in misses_target.message
    assert below_perfect.admitted is False
    assert below_perfect.failure_kind == "constraint"
    assert "perfect-data baseline" in below_perfect.message


@pytest.mark.parametrize(
    "payload, message_fragment",
    [
        ((0.2,), "evaluator envelope"),
        ((True, _constraint_boundary()), "real number"),
        ((-0.1, _constraint_boundary()), "nonnegative"),
        ((0.1, -0.01), "nonnegative"),
        ((math.inf, _constraint_boundary()), "finite"),
        ((math.nan, _constraint_boundary()), "finite"),
        ({}, "missing"),
    ],
)
def test_denoising_rejects_malformed_or_invalid_payloads(payload,
                                                          message_fragment):
    verified = Denoising({}).verify_answer_payload(payload)

    assert verified.resolved is True
    assert verified.admitted is False
    assert verified.failure_kind == "constraint"
    assert message_fragment in verified.message


@pytest.mark.parametrize(
    "field, bad_value, message_fragment",
    [
        ("schema_version", 2, "schema_version"),
        ("problem", "other", "problem identifier"),
        ("dataset", "pbmc", "dataset identifier"),
        ("eval_seed", 7, "eval_seed"),
        ("evaluator_version", "future_v2", "evaluator_version"),
    ],
)
def test_denoising_rejects_mismatched_evaluator_context(field, bad_value,
                                                         message_fragment):
    problem = Denoising({"eval_seed": 42})
    payload = problem.serialize_answer(_valid_output())
    payload[field] = bad_value

    verified = problem.verify_answer_payload(payload)

    assert verified.admitted is False
    assert message_fragment in verified.message


def test_denoising_rejects_unknown_envelope_and_metric_fields():
    problem = Denoising({})
    envelope_extra = problem.serialize_answer(_valid_output())
    envelope_extra["source"] = "must not enter scientific identity"
    metric_extra = problem.serialize_answer(_valid_output())
    metric_extra["metrics"]["claimed_reward"] = 999.0

    first = problem.verify_answer_payload(envelope_extra)
    second = problem.verify_answer_payload(metric_extra)

    assert first.admitted is False
    assert "unexpected source" in first.message
    assert second.admitted is False
    assert "exactly mse and poisson" in second.message


def test_denoising_zero_mse_has_finite_best_reward():
    problem = Denoising({})

    verified = problem.verify_answer_payload(_valid_output(0.0))

    assert verified.admitted is True
    assert verified.raw_score == 0.0
    assert verified.internal_reward == 1.0e12
    assert math.isfinite(verified.internal_reward)


def test_denoising_descriptor_and_fingerprint_are_source_independent():
    problem = Denoising({})
    payload = problem.serialize_answer(_valid_output())

    descriptor_a = problem.describe_scientific_state(
        None, {"answer_payload": payload, "source_text": "algorithm A"}
    )
    descriptor_b = problem.describe_scientific_state(
        None, {"answer_payload": payload, "source_text": "algorithm B"}
    )
    fingerprint_a = problem.scientific_fingerprint(
        None, {"answer_payload": payload, "source_text": "algorithm A"}
    )
    fingerprint_b = problem.scientific_fingerprint(
        None, {"answer_payload": payload, "source_text": "algorithm B"}
    )

    assert descriptor_a == descriptor_b
    assert descriptor_a == {
        "dataset": "pancreas",
        "mse_gain_bin": "medium",
        "poisson_margin_bin": "low",
        "metric_tradeoff": "poisson_leading",
    }
    assert fingerprint_a == fingerprint_b
    assert len(fingerprint_a) == 64


def test_denoising_confirmation_uses_saved_payload_not_candidate():
    problem = Denoising({})
    payload = problem.serialize_answer(_valid_output())

    confirmed = problem.confirm_record(
        candidate=(math.nan, math.nan),
        evidence={"answer_payload": payload},
    )

    assert confirmed.admitted is True
    assert confirmed.answer_payload == payload


def test_denoising_renderer_and_realistic_resource_declaration(tmp_path):
    problem = Denoising({
        "eval_cpus": 3,
        "eval_memory_mb": 4096,
        "sandbox_timeout_s": 11,
    })
    payload = problem.serialize_answer(_valid_output())

    rendered = problem.render_best(None, {"answer_payload": payload}, tmp_path)
    resources = problem.resource_requirements()

    assert [Path(path).name for path in rendered] == ["answer.json", "answer.txt"]
    assert json.loads((tmp_path / "answer.json").read_text()) == payload
    assert "poisson_norm:" in (tmp_path / "answer.txt").read_text()
    assert resources.cpu_cores == 3
    assert resources.memory_mb == 4096
    assert resources.timeout_s == 11.0
    assert resources.gpu_count == 0
    assert resources.exclusive_gpu is False
    assert resources.network_access is False
    assert resources.filesystem_policy == "read_only_dataset_and_temporary"
    assert resources.timeout_is_scientific is True


def test_denoising_legacy_score_and_threshold_behavior_is_unchanged():
    problem = Denoising({"fail_score": -7.0})
    boundary = _constraint_boundary()

    legacy_valid = problem.score((0.2, boundary), "legacy stdout")
    legacy_invalid = problem.score(
        (0.2, boundary + 1.0e-15), "legacy stdout"
    )

    assert verify_denoising((0.2, boundary)) is True
    assert legacy_valid.valid is True
    assert legacy_valid.raw_score == 0.2
    assert legacy_valid.reward == 5.0
    assert legacy_valid.msg == f"mse=0.2, poisson={boundary}"
    assert legacy_invalid.valid is False
    assert legacy_invalid.reward == -7.0
    assert legacy_invalid.msg == "Invalid solution."
