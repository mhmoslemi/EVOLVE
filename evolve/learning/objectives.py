"""OrderGrad top-m-at-K advantages, and MaxPO as their pure-max, top_m=1 form.

Every member's advantage is a function only of its rank among the group by
``BranchOutcome.maximum_reward`` (a branch with no admitted descendant ranks
last): the ``top_m`` best-ranked members receive ``1 - top_m/K``, the rest
receive ``-top_m/K``.  This is exactly centered by construction -- the
advantages of any group sum to zero -- which is the "centering identity"
AGENTS.md asks every objective to satisfy.  MaxPO is tracked as its own
named, tested objective (never silently redefined) even though it is
numerically identical to ``top_m=1`` OrderGrad.
"""

from __future__ import annotations

from typing import Optional, Sequence, Tuple

from evolve.types import LearningObjective


ORDERGRAD_VERSION = "ordergrad_top_m_at_k_v1"
MAXPO_VERSION = "maxpo_centered_pure_max_v1"


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
    k = len(rewards)
    if k == 0:
        raise ObjectiveError("cannot compute advantages for an empty group")
    if isinstance(top_m, bool) or not isinstance(top_m, int) or not 1 <= top_m <= k:
        raise ObjectiveError("top_m must be an integer in [1, group_size]")
    ranks = rank_order(rewards)
    fraction = top_m / k
    return tuple((1.0 - fraction) if rank < top_m else -fraction for rank in ranks)


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
