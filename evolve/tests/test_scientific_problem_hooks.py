import json
import math

import numpy as np
import pytest

from evolve.verifier import ProblemScientificAdapter
from evolve.verifier.evidence import scientific_state_id
from problems.ac_inequalities import (
    ACInequalities,
    evaluate_sequence_ac1,
    evaluate_sequence_ac2,
)
from problems.circle_packing import CirclePacking
from problems.erdos import ErdosMinOverlap


def _raise_if_legacy_score_is_used(*_args, **_kwargs):
    raise AssertionError("scientific payload verification called legacy score")


def _two_circle_candidate(claimed_sum=999.0, *, reversed_order=False):
    centers = np.asarray([[0.25, 0.5], [0.75, 0.5]], dtype=float)
    radii = np.asarray([0.25, 0.25], dtype=float)
    if reversed_order:
        centers = centers[::-1]
        radii = radii[::-1]
    return centers, radii, claimed_sum


def test_circle_payload_is_canonical_and_ignores_claimed_sum():
    problem = CirclePacking({"num_circles": 2})

    first = problem.serialize_answer(_two_circle_candidate(999.0))
    second = problem.serialize_answer(
        _two_circle_candidate(float("nan"), reversed_order=True)
    )

    assert first == second
    assert first == {
        "schema_version": 1,
        "problem": "circle_packing",
        "num_circles": 2,
        "centers": [[0.25, 0.5], [0.75, 0.5]],
        "radii": [0.25, 0.25],
    }
    assert "claimed_sum" not in first


def test_circle_payload_verifier_is_independent_and_recomputes_sum(monkeypatch):
    problem = CirclePacking({"num_circles": 2})
    payload = problem.serialize_answer(_two_circle_candidate())
    monkeypatch.setattr(problem, "score", _raise_if_legacy_score_is_used)

    verified = problem.verify_answer_payload(payload)

    assert problem.scientific_method_complete is True
    assert verified.resolved is True
    assert verified.admitted is True
    assert verified.answer_payload == payload
    assert verified.raw_score == 0.5
    assert verified.internal_reward == 0.5
    assert verified.scores["pair_contacts"] == 1
    assert verified.flags["payload_only"] is True


def test_circle_descriptor_and_fingerprint_use_contact_radius_structure():
    problem = CirclePacking({"num_circles": 2})
    forward = problem.serialize_answer(_two_circle_candidate())
    reverse = problem.serialize_answer(_two_circle_candidate(reversed_order=True))

    descriptor = problem.describe_scientific_state(
        None, {"answer_payload": forward, "source_text": "ignored source A"}
    )
    first_fingerprint = problem.scientific_fingerprint(
        None, {"answer_payload": forward, "source_text": "ignored source A"}
    )
    second_fingerprint = problem.scientific_fingerprint(
        None, {"answer_payload": reverse, "source_text": "ignored source B"}
    )

    assert descriptor == {
        "boundary_contact_bin": "high",
        "pair_contact_bin": "high",
        "radius_dispersion_bin": "uniform",
    }
    assert first_fingerprint == second_fingerprint
    assert len(first_fingerprint) == 64


def test_circle_payload_rejects_invalid_scientific_construction():
    problem = CirclePacking({"num_circles": 2})
    overlapping = (
        np.asarray([[0.5, 0.5], [0.5, 0.5]]),
        np.asarray([0.2, 0.2]),
        0.4,
    )

    verified = problem.verify_answer_payload(problem.serialize_answer(overlapping))

    assert verified.resolved is True
    assert verified.admitted is False
    assert verified.failure_kind == "constraint"
    assert "overlap" in verified.message.lower()


def test_circle_seed_constructions_serialize_and_verify_independently():
    problem = CirclePacking({"num_circles": 26, "num_seed_states": 8})
    seeds = problem.seed_states()
    payloads = []

    assert len(seeds) == 8
    for seed in seeds:
        payload = problem.serialize_answer(seed.construction)
        verified = problem.verify_answer_payload(payload)

        assert "def run_packing():" in seed.code
        assert verified.resolved is True
        assert verified.admitted is True
        assert verified.answer_payload == payload
        assert verified.internal_reward == pytest.approx(seed.value)
        assert verified.raw_score == pytest.approx(seed.raw_score)
        payloads.append(json.dumps(payload, sort_keys=True))

    assert len(set(payloads)) == 8


def test_scientific_payload_hooks_reject_future_or_boolean_schema_versions():
    circle = CirclePacking({"num_circles": 2})
    circle_payload = circle.serialize_answer(_two_circle_candidate())
    erdos = ErdosMinOverlap({"budget_s": 1})
    erdos_payload = erdos.serialize_answer(([0.2, 0.4, 0.6, 0.8], 1.0, 4))
    ac = ACInequalities({"problem_type": "ac1", "budget_s": 1})
    ac_payload = ac.serialize_answer([1.0, 2.0, 3.0])

    for problem, payload in (
        (circle, circle_payload), (erdos, erdos_payload), (ac, ac_payload)
    ):
        for unsupported in (True, 2):
            malformed = {**payload, "schema_version": unsupported}
            verified = problem.verify_answer_payload(malformed)
            assert verified.admitted is False
            assert "schema_version" in verified.message


def test_circle_declares_cpu_only_scientific_timeout():
    resources = CirclePacking({
        "num_circles": 2,
        "sandbox_timeout_s": 7,
    }).resource_requirements()

    assert resources.cpu_cores == 1
    assert resources.gpu_count == 0
    assert resources.timeout_s == 7.0
    assert resources.timeout_is_scientific is True
    assert resources.network_access is False


def test_erdos_payload_is_full_and_ignores_claimed_bound():
    problem = ErdosMinOverlap({"budget_s": 1})
    h_values = [0.2, 0.4, 0.6, 0.8]

    first = problem.serialize_answer((h_values, -12345.0, 4))
    second = problem.serialize_answer((h_values, float("nan"), 4))

    assert first == second
    assert first["h_values"] == h_values
    assert first["n_points"] == 4
    assert "claimed_c5" not in first


def test_erdos_seed_construction_serializes_as_saved_answer():
    problem = ErdosMinOverlap({"budget_s": 1, "num_seed_states": 1})
    construction = problem.seed_states()[0].construction

    payload = problem.serialize_answer(construction)
    verified = problem.verify_answer_payload(payload)

    assert payload["h_values"] == construction
    assert payload["n_points"] == len(construction)
    assert verified.admitted is True


def test_erdos_payload_verifier_recomputes_c5_without_legacy_score(monkeypatch):
    problem = ErdosMinOverlap({"budget_s": 1})
    # Sum one is normalized to sum two for n=4. The raw payload remains intact.
    payload = problem.serialize_answer(([0.1, 0.2, 0.3, 0.4], 0.0, 4))
    monkeypatch.setattr(problem, "score", _raise_if_legacy_score_is_used)

    verified = problem.verify_answer_payload(payload)

    assert problem.scientific_method_complete is True
    assert verified.admitted is True
    assert verified.answer_payload["h_values"] == [0.1, 0.2, 0.3, 0.4]
    assert verified.raw_score == pytest.approx(0.5)
    assert verified.internal_reward == pytest.approx(1.0 / (1e-8 + 0.5))
    assert verified.flags["normalized_for_verification"] is True
    assert verified.flags["payload_only"] is True


def test_erdos_reversal_fingerprint_and_output_descriptor_are_source_free():
    problem = ErdosMinOverlap({"budget_s": 1})
    forward = problem.serialize_answer(([0.2, 0.4, 0.6, 0.8], 9.0, 4))
    reverse = problem.serialize_answer(([0.8, 0.6, 0.4, 0.2], -9.0, 4))

    descriptor = problem.describe_scientific_state(
        None, {"answer_payload": forward, "source_text": "ignored"}
    )
    first = problem.scientific_fingerprint(None, {"answer_payload": forward})
    second = problem.scientific_fingerprint(None, {"answer_payload": reverse})

    assert descriptor["resolution_bin"] == "coarse"
    assert set(descriptor) == {
        "resolution_bin", "binarity_bin", "transition_bin", "symmetry_bin"
    }
    assert first == second
    assert len(first) == 64


@pytest.mark.parametrize(
    "candidate",
    [
        ([0.0, 0.0], 0.0, 2),
        ([0.5, 1.5], 0.0, 2),
        ([0.5, float("inf")], 0.0, 2),
        ([0.5], 0.0, 2),
    ],
)
def test_erdos_payload_rejects_invalid_saved_functions(candidate):
    problem = ErdosMinOverlap({"budget_s": 1})
    try:
        payload = problem.serialize_answer(candidate)
    except ValueError:
        return

    verified = problem.verify_answer_payload(payload)
    assert verified.admitted is False
    assert verified.failure_kind == "constraint"


def test_erdos_confirmation_prefers_saved_payload_and_declares_resources():
    problem = ErdosMinOverlap({
        "budget_s": 1,
        "eval_cpus": 3,
        "sandbox_timeout_s": 7,
    })
    payload = problem.serialize_answer(([0.2, 0.4, 0.6, 0.8], 999.0, 4))

    confirmed = problem.confirm_record(
        candidate="not a scientific answer",
        evidence={"answer_payload": payload},
    )
    resources = problem.resource_requirements()

    assert confirmed.admitted is True
    assert confirmed.raw_score == pytest.approx(0.5)
    assert resources.cpu_cores == 3
    assert resources.gpu_count == 0
    assert resources.timeout_s == 7.0
    assert resources.timeout_is_scientific is True


@pytest.mark.parametrize("problem_type", ["ac1", "ac2"])
def test_ac_payload_captures_full_sequence_and_recomputes_metric(problem_type,
                                                                 monkeypatch):
    problem = ACInequalities({"problem_type": problem_type, "budget_s": 1})
    payload = problem.serialize_answer([2000.0, 2.0, 3.0])
    monkeypatch.setattr(problem, "score", _raise_if_legacy_score_is_used)

    verified = problem.verify_answer_payload(payload)
    expected = (
        evaluate_sequence_ac1(payload["sequence"])
        if problem_type == "ac1"
        else evaluate_sequence_ac2(payload["sequence"])
    )

    assert problem.scientific_method_complete is True
    assert payload["sequence"][0] == 2000.0
    assert payload["problem_type"] == problem_type
    assert verified.admitted is True
    assert verified.raw_score == pytest.approx(expected)
    if problem_type == "ac1":
        assert verified.internal_reward == pytest.approx(1.0 / (1e-8 + expected))
    else:
        assert verified.internal_reward == pytest.approx(expected)
    assert verified.flags["payload_only"] is True


def test_ac_scientific_hook_rejects_negatives_without_changing_legacy_score():
    problem = ACInequalities({"problem_type": "ac1", "budget_s": 1})
    sequence = [-1.0, 1.0, 2.0]

    legacy = problem.score(sequence, "")
    scientific = problem.verify_answer_payload(sequence)

    assert legacy.valid is True  # Legacy evaluator still clips negatives.
    assert scientific.resolved is True
    assert scientific.admitted is False
    assert scientific.failure_kind == "constraint"
    assert "nonnegative" in scientific.message


def test_ac_payload_rejects_subtype_mismatch_and_nonfinite_values():
    ac1 = ACInequalities({"problem_type": "ac1", "budget_s": 1})
    ac2_payload = ACInequalities({
        "problem_type": "ac2", "budget_s": 1
    }).serialize_answer([1.0, 2.0])

    mismatch = ac1.verify_answer_payload(ac2_payload)
    nonfinite = ac1.verify_answer_payload([1.0, math.inf])

    assert mismatch.admitted is False
    assert "subtype" in mismatch.message
    assert nonfinite.admitted is False
    assert "finite" in nonfinite.message


@pytest.mark.parametrize("problem_type", ["ac1", "ac2"])
def test_ac_descriptor_and_fingerprint_are_reversal_canonical(problem_type):
    problem = ACInequalities({"problem_type": problem_type, "budget_s": 1})

    descriptor = problem.describe_scientific_state([1.0, 2.0, 3.0])
    forward = problem.scientific_fingerprint(
        None,
        {"answer_payload": problem.serialize_answer([1.0, 2.0, 3.0]),
         "source_text": "ignored source A"},
    )
    reverse = problem.scientific_fingerprint(
        None,
        {"answer_payload": problem.serialize_answer([3.0, 2.0, 1.0]),
         "source_text": "ignored source B"},
    )

    assert descriptor["problem_type"] == problem_type
    assert descriptor["length_bin"] == "short"
    assert forward == reverse
    assert len(forward) == 64


def test_ac_confirmation_and_cpu_resource_declaration():
    problem = ACInequalities({
        "problem_type": "ac1",
        "budget_s": 1,
        "sandbox_timeout_s": 7,
    })
    payload = problem.serialize_answer([1.0, 2.0, 3.0])

    confirmed = problem.confirm_record(
        candidate=[-1.0], evidence={"answer_payload": payload}
    )
    resources = problem.resource_requirements()

    assert confirmed.admitted is True
    assert resources.cpu_cores == 2
    assert resources.gpu_count == 0
    assert resources.timeout_s == 7.0
    assert resources.timeout_is_scientific is True


@pytest.mark.parametrize(
    ("problem", "candidate"),
    [
        (
            CirclePacking({"num_circles": 2}),
            ([[10 ** 1000, 0.5], [0.75, 0.5]], [0.1, 0.1], 0.2),
        ),
        (
            CirclePacking({"num_circles": 2}),
            ([[True, 0.5], [0.75, 0.5]], [0.1, 0.1], 0.2),
        ),
        (
            ErdosMinOverlap({"budget_s": 1}),
            ([0.5, 10 ** 1000], 0.0, 2),
        ),
        (
            ErdosMinOverlap({"budget_s": 1}),
            ([False, 0.5], 0.0, 2),
        ),
        (
            ACInequalities({"problem_type": "ac1", "budget_s": 1}),
            [1.0, 10 ** 1000],
        ),
        (
            ACInequalities({"problem_type": "ac1", "budget_s": 1}),
            [1.0, True],
        ),
    ],
)
def test_adversarial_numeric_payloads_are_resolved_constraints(problem,
                                                                 candidate):
    verified = problem.verify_answer_payload(candidate)

    assert verified.resolved is True
    assert verified.admitted is False
    assert verified.failure_kind == "constraint"
    assert verified.answer_payload is None


def test_erdos_payload_limit_rejects_before_quadratic_correlation(monkeypatch):
    problem = ErdosMinOverlap({
        "budget_s": 1,
        "scientific_max_points": 4,
    })
    payload = {
        "schema_version": problem.answer_schema_version,
        "problem": problem.name,
        "n_points": 5,
        "h_values": [0.5] * 5,
    }
    monkeypatch.setattr(
        problem,
        "_effective_h",
        lambda *_args: (_ for _ in ()).throw(
            AssertionError("quadratic correlation was reached")
        ),
    )
    monkeypatch.setattr(
        np,
        "asarray",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(
            AssertionError("NumPy conversion was reached")
        ),
    )

    verified = problem.verify_answer_payload(payload)

    assert verified.resolved is True
    assert verified.failure_kind == "constraint"
    assert "scientific_max_points" in verified.message


@pytest.mark.parametrize("problem_type", ["ac1", "ac2"])
def test_ac_payload_limit_rejects_before_quadratic_evaluator(
    monkeypatch,
    problem_type,
):
    problem = ACInequalities({
        "problem_type": problem_type,
        "budget_s": 1,
        "scientific_max_coefficients": 3,
    })
    payload = {
        "schema_version": problem.answer_schema_version,
        "problem": problem.name,
        "problem_type": problem_type,
        "sequence": [1.0, 2.0, 3.0, 4.0],
    }
    monkeypatch.setattr(
        problem,
        "_evaluate",
        lambda *_args: (_ for _ in ()).throw(
            AssertionError("quadratic AC evaluator was reached")
        ),
    )

    verified = problem.verify_answer_payload(payload)

    assert verified.resolved is True
    assert verified.failure_kind == "constraint"
    assert "scientific_max_coefficients" in verified.message


@pytest.mark.parametrize(
    ("problem_class", "config_key", "base_config"),
    [
        (ErdosMinOverlap, "scientific_max_points", {"budget_s": 1}),
        (
            ACInequalities,
            "scientific_max_coefficients",
            {"problem_type": "ac1", "budget_s": 1},
        ),
    ],
)
@pytest.mark.parametrize("bad_limit", [True, 0, -1, 2.5])
def test_scientific_payload_limits_must_be_positive_integers(
    problem_class,
    config_key,
    base_config,
    bad_limit,
):
    with pytest.raises(ValueError, match="positive integer"):
        problem_class({**base_config, config_key: bad_limit})


def test_payload_limit_is_frozen_into_verifier_not_scientific_state_identity():
    low_erdos = ErdosMinOverlap({
        "budget_s": 1,
        "scientific_max_points": 4,
    })
    high_erdos = ErdosMinOverlap({
        "budget_s": 1,
        "scientific_max_points": 8,
    })
    values = ([0.2, 0.4, 0.6, 0.8], 0.0, 4)
    low_payload = low_erdos.serialize_answer(values)
    high_payload = high_erdos.serialize_answer(values)

    assert low_erdos.cfg["scientific_max_points"] == 4
    assert high_erdos.cfg["scientific_max_points"] == 8
    assert low_payload == high_payload
    assert scientific_state_id("erdos", low_payload) == scientific_state_id(
        "erdos", high_payload
    )
    assert (
        ProblemScientificAdapter(low_erdos).verifier_id
        != ProblemScientificAdapter(high_erdos).verifier_id
    )
    default_erdos = ErdosMinOverlap({"budget_s": 1})
    explicit_default_erdos = ErdosMinOverlap({
        "budget_s": 1,
        "scientific_max_points": 4096,
    })
    assert (
        ProblemScientificAdapter(default_erdos).verifier_id
        == ProblemScientificAdapter(explicit_default_erdos).verifier_id
    )

    low_ac = ACInequalities({
        "problem_type": "ac1",
        "budget_s": 1,
        "scientific_max_coefficients": 3,
    })
    high_ac = ACInequalities({
        "problem_type": "ac1",
        "budget_s": 1,
        "scientific_max_coefficients": 6,
    })
    low_ac_payload = low_ac.serialize_answer([1.0, 2.0, 3.0])
    high_ac_payload = high_ac.serialize_answer([1.0, 2.0, 3.0])
    assert low_ac.cfg["scientific_max_coefficients"] == 3
    assert high_ac.cfg["scientific_max_coefficients"] == 6
    assert low_ac_payload == high_ac_payload
    assert scientific_state_id(
        "ac_inequalities", low_ac_payload
    ) == scientific_state_id("ac_inequalities", high_ac_payload)
    assert (
        ProblemScientificAdapter(low_ac).verifier_id
        != ProblemScientificAdapter(high_ac).verifier_id
    )

    default_ac = ACInequalities({"problem_type": "ac1", "budget_s": 1})
    explicit_default_ac = ACInequalities({
        "problem_type": "ac1",
        "budget_s": 1,
        "scientific_max_coefficients": 10_000,
    })
    assert (
        ProblemScientificAdapter(default_ac).verifier_id
        == ProblemScientificAdapter(explicit_default_ac).verifier_id
    )


def test_default_scientific_limits_cover_active_seed_ranges():
    assert ErdosMinOverlap({"budget_s": 1}).scientific_max_points >= 100
    assert ACInequalities({
        "problem_type": "ac1", "budget_s": 1
    }).scientific_max_coefficients >= 8000
