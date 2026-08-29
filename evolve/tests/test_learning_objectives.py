"""Small independent checks for the production OrderGrad transform."""

from itertools import product

import pytest

from evolve.learning.objectives import ObjectiveError, ordergrad_advantages


def test_ordergrad_max_at_k_matches_exact_bernoulli_gradient():
    """Enumerate the sampling distribution and compare to the analytic gradient."""

    n = 4
    k = n - 1
    probability = 0.37
    estimated_gradient = 0.0
    for outcomes in product((0.0, 1.0), repeat=n):
        successes = int(sum(outcomes))
        mass = probability**successes * (1.0 - probability) ** (n - successes)
        advantages = ordergrad_advantages(outcomes, top_m=1)
        score_gradient = sum(
            advantage * (outcome - probability)
            for advantage, outcome in zip(advantages, outcomes)
        ) / n
        estimated_gradient += mass * score_gradient

    # d/d(logit(p)) E[max(X_1,...,X_K)] for Bernoulli rewards.
    analytic_gradient = k * probability * (1.0 - probability) ** k
    assert estimated_gradient == pytest.approx(analytic_gradient, abs=1.0e-12)


def test_ordergrad_advantages_are_centered_for_each_realized_group():
    rewards = (0.1, 0.7, 0.4, 0.9, 0.2)
    for top_m in (1, 2, 3):
        assert sum(ordergrad_advantages(rewards, top_m=top_m)) == pytest.approx(
            0.0, abs=1.0e-12
        )


def test_ordergrad_rejects_incomplete_or_invalid_groups():
    with pytest.raises(ObjectiveError):
        ordergrad_advantages((1.0,), top_m=1)
    with pytest.raises(ObjectiveError):
        ordergrad_advantages((1.0, None), top_m=1)
    with pytest.raises(ObjectiveError):
        ordergrad_advantages((1.0, 2.0), top_m=2)
