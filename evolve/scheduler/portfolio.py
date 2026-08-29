"""Correlation-aware Monte Carlo joint-portfolio arm selection.

Implements ``E[max(0, max_r Z_r - M)]`` from AGENTS.md's "Posterior
allocation": each candidate arm's simulated per-draw record gain ``Z_r`` is
generated once, up front, from a small Gaussian-copula factor model so arms
sharing a cell or an (role, option) family are positively correlated across
Monte Carlo worlds.  Greedy selection then adds the arm with the largest
marginal joint-max value -- the change in ``E[max(0, ...)]`` from adding it
to the arms already chosen, not a sum of independent expected improvements --
subject to the remaining per-resource budget.
"""

from __future__ import annotations

import math
import random
from dataclasses import dataclass
from typing import Dict, List, Mapping, Sequence, Tuple

from evolve.ids import derive_seed

from .arms import ArmCandidate, ArmIdentity
from .posterior import PosteriorStore


PORTFOLIO_VERSION = "gaussian_copula_joint_max_v1"
DEFAULT_MONTE_CARLO_SAMPLES = 128

# Fixed first-baseline factor loadings: shared cell correlation dominates,
# shared (role, option) family correlation is weaker, and the remainder is
# idiosyncratic.  Chosen so the three variance components sum to one.
_CELL_LOADING = 0.4
_FAMILY_LOADING = 0.15
_IDIOSYNCRATIC_LOADING = 1.0 - _CELL_LOADING - _FAMILY_LOADING


class PortfolioError(ValueError):
    """A portfolio selection received invalid candidates or resources."""


def _norm_cdf(z: float) -> float:
    return 0.5 * (1.0 + math.erf(z / math.sqrt(2.0)))


@dataclass(frozen=True)
class ValuedArm:
    """One arm chosen by the portfolio search, with its reproducible basis."""

    identity: ArmIdentity
    expected_cost: Mapping[str, float]
    posterior_level: str
    expected_gain: float
    uncertainty: float
    marginal_gain: float
    rng_seed: int


def _factor_draws(
    identities: Sequence[ArmIdentity], *, samples: int, seed: int
) -> List[List[float]]:
    """Deterministic Gaussian-copula latents ``z_r`` for every (sample, arm)."""

    rng = random.Random(derive_seed("portfolio_factor_draws", seed, samples, len(identities)))
    cell_ids = sorted({identity.cell_id for identity in identities})
    families = sorted({(identity.role.value, identity.option_id) for identity in identities})
    out: List[List[float]] = [[] for _ in identities]
    for _ in range(samples):
        cell_factor = {cell_id: rng.gauss(0.0, 1.0) for cell_id in cell_ids}
        family_factor = {family: rng.gauss(0.0, 1.0) for family in families}
        for index, identity in enumerate(identities):
            idiosyncratic = rng.gauss(0.0, 1.0)
            z = (
                math.sqrt(_CELL_LOADING) * cell_factor[identity.cell_id]
                + math.sqrt(_FAMILY_LOADING) * family_factor[(identity.role.value, identity.option_id)]
                + math.sqrt(_IDIOSYNCRATIC_LOADING) * idiosyncratic
            )
            out[index].append(z)
    return out


def simulate_joint_draws(
    candidates: Sequence[ArmCandidate],
    *,
    posterior: PosteriorStore,
    samples: int = DEFAULT_MONTE_CARLO_SAMPLES,
    seed: int,
) -> List[List[float]]:
    """Return ``samples`` correlated simulated record-gain draws per candidate.

    Each draw ``Z_r`` is zero unless the copula bernoulli test (driven by the
    arm's posterior point-estimate positive probability) succeeds, in which
    case it is a Bayesian-bootstrap draw of the positive-gain magnitude.
    """

    if samples < 1:
        raise PortfolioError("samples must be positive")
    identities = [candidate.identity for candidate in candidates]
    latents = _factor_draws(identities, samples=samples, seed=seed)
    magnitude_rng = random.Random(derive_seed("portfolio_magnitude_draws", seed, samples, len(identities)))
    draws: List[List[float]] = []
    for candidate, arm_latents in zip(candidates, latents):
        arm_draws = []
        for z in arm_latents:
            # Draw posterior rates inside each Monte Carlo world instead of
            # collapsing sparse evidence to a point estimate.
            p_positive = (
                posterior.sample_admission_rate(candidate.identity, magnitude_rng)
                * posterior.sample_improvement_rate(candidate.identity, magnitude_rng)
            )
            u = _norm_cdf(z)
            if u < p_positive:
                arm_draws.append(posterior.sample_positive_gain(candidate.identity, magnitude_rng))
            else:
                arm_draws.append(0.0)
        draws.append(arm_draws)
    return draws


def _mean_positive(values: Sequence[float]) -> float:
    materialized = list(values)
    return sum(max(0.0, value) for value in materialized) / len(materialized)


def select_portfolio(
    candidates: Sequence[ArmCandidate],
    *,
    posterior: PosteriorStore,
    resource_limits: Mapping[str, float],
    max_arms: int,
    seed: int,
    samples: int = DEFAULT_MONTE_CARLO_SAMPLES,
    min_marginal_gain: float = 1e-9,
) -> Tuple[ValuedArm, ...]:
    """Greedily fill the remainder budget by marginal joint-max value.

    Marginal value is recomputed against the running ``max`` over already
    selected draws at every step -- the joint-max objective's defining
    property -- not summed as if arms were independent.
    """

    if isinstance(max_arms, bool) or not isinstance(max_arms, int) or max_arms < 0:
        raise PortfolioError("max_arms must be a non-negative integer")
    if not candidates or max_arms == 0:
        return ()

    draws = simulate_joint_draws(candidates, posterior=posterior, samples=samples, seed=seed)
    remaining_resources: Dict[str, float] = {resource: float(limit) for resource, limit in resource_limits.items()}
    running_best = [0.0] * samples
    running_value = 0.0
    available = list(range(len(candidates)))
    selected: List[ValuedArm] = []

    while available and len(selected) < max_arms:
        best_index = None
        best_marginal = min_marginal_gain
        best_candidate_value = 0.0
        for index in available:
            candidate = candidates[index]
            if any(
                float(amount) > remaining_resources.get(resource, 0.0) + 1e-9
                for resource, amount in candidate.expected_cost.items()
            ):
                continue
            candidate_draws = draws[index]
            candidate_value = _mean_positive(
                [max(current, draw) for current, draw in zip(running_best, candidate_draws)]
            )
            marginal = candidate_value - running_value
            if marginal > best_marginal:
                best_marginal = marginal
                best_index = index
                best_candidate_value = candidate_value
        if best_index is None:
            break

        candidate = candidates[best_index]
        candidate_draws = draws[best_index]
        running_best = [max(current, draw) for current, draw in zip(running_best, candidate_draws)]
        running_value = best_candidate_value
        for resource, amount in candidate.expected_cost.items():
            remaining_resources[resource] = remaining_resources.get(resource, 0.0) - float(amount)
        snapshot = posterior.snapshot(candidate.identity)
        selected.append(
            ValuedArm(
                identity=candidate.identity,
                expected_cost=dict(candidate.expected_cost),
                posterior_level=snapshot.hierarchy_level,
                expected_gain=snapshot.expected_gain,
                uncertainty=snapshot.uncertainty,
                marginal_gain=best_marginal,
                rng_seed=derive_seed("portfolio_arm_seed", seed, candidate.identity.key()),
            )
        )
        available.remove(best_index)

    return tuple(selected)


__all__ = [
    "DEFAULT_MONTE_CARLO_SAMPLES",
    "PORTFOLIO_VERSION",
    "PortfolioError",
    "ValuedArm",
    "select_portfolio",
    "simulate_joint_draws",
]
