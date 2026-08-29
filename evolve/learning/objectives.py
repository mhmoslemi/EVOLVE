"""Exact finite-batch OrderGrad likelihood-ratio advantages.

For a realized on-policy batch of ``N`` rewards this implementation uses the
largest leave-one-out comparison size, ``K = N - 1``.  For each item it
computes the include-one Top-M@K value and subtracts the valid leave-one-out
value.  The returned multiplier includes OrderGrad's ``K`` factor, allowing
the trainer to use its ordinary mean policy-gradient loss directly.

Pure-max MaxPO uses the same canonical finite-batch Max@K advantage with
``top_m = 1``.  It is retained as a separately versioned objective rather than
being relabelled from a binary winner/loser rank heuristic.
"""

from __future__ import annotations

from itertools import combinations
from typing import Optional, Sequence, Tuple

from evolve.types import LearningObjective


ORDERGRAD_VERSION = "ordergrad_lr_top_m_k_n_minus_1_v2"
MAXPO_VERSION = "maxpo_canonical_max_k_n_minus_1_v2"


class ObjectiveError(ValueError):
    """An advantage computation received an invalid group or configuration."""


def rank_order(rewards: Sequence[Optional[float]]) -> Tuple[int, ...]:
    """0-indexed descending rank per member; ties break by (then) input index."""

    if not rewards:
        raise ObjectiveError("cannot rank an empty group")
    indexed = sorted(
        range(len(rewards)),
        key=lambda index: (
            -(rewards[index] if rewards[index] is not None else float("-inf")),
            index,
        ),
    )
    ranks = [0] * len(rewards)
    for order, index in enumerate(indexed):
        ranks[index] = order
    return tuple(ranks)


def ordergrad_advantages(rewards: Sequence[Optional[float]], *, top_m: int) -> Tuple[float, ...]:
    n = len(rewards)
    if n < 2:
        raise ObjectiveError("OrderGrad likelihood-ratio groups need at least two samples")
    k = n - 1
    if isinstance(top_m, bool) or not isinstance(top_m, int) or not 1 <= top_m <= k:
        raise ObjectiveError("top_m must be an integer in [1, group_size - 1]")
    if any(value is None for value in rewards):
        raise ObjectiveError("OrderGrad rewards must be finite normalized gains")
    values = tuple(float(value) for value in rewards)

    def top_m_value(indices: Sequence[int]) -> float:
        selected = sorted((values[index] for index in indices), reverse=True)
        return sum(selected[:top_m]) / top_m

    advantages = []
    all_indices = tuple(range(n))
    for item in all_indices:
        others = tuple(index for index in all_indices if index != item)
        included_values = [
            top_m_value((item,) + subset)
            for subset in combinations(others, k - 1)
        ]
        include_one = sum(included_values) / len(included_values)
        leave_one_out = top_m_value(others)
        advantages.append(k * (include_one - leave_one_out))
    return tuple(advantages)


def maxpo_advantages(rewards: Sequence[Optional[float]]) -> Tuple[float, ...]:
    """The pure-max, exactly centered special case: ``top_m = 1``."""

    return ordergrad_advantages(rewards, top_m=1)


def advantages_for_objective(
    rewards: Sequence[Optional[float]],
    *,
    objective: LearningObjective,
    top_m: int,
) -> Tuple[float, ...]:
    if objective == LearningObjective.MAXPO:
        if top_m != 1:
            raise ObjectiveError("MaxPO is pure-max and requires top_m=1")
        return maxpo_advantages(rewards)
    if objective == LearningObjective.ORDERGRAD:
        return ordergrad_advantages(rewards, top_m=top_m)
    raise ObjectiveError(f"unknown learning objective {objective!r}")


def objective_version(objective: LearningObjective) -> str:
    if objective == LearningObjective.MAXPO:
        return MAXPO_VERSION
    if objective == LearningObjective.ORDERGRAD:
        return ORDERGRAD_VERSION
    raise ObjectiveError(f"unknown learning objective {objective!r}")


__all__ = [
    "MAXPO_VERSION",
    "ORDERGRAD_VERSION",
    "ObjectiveError",
    "advantages_for_objective",
    "maxpo_advantages",
    "objective_version",
    "ordergrad_advantages",
    "rank_order",
]
