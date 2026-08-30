"""Correlation-aware Monte Carlo joint-portfolio arm selection.

Implements ``E[max(0, max_r Z_r - M)]`` from AGENTS.md's "Posterior
allocation": each candidate arm's simulated per-draw record gain ``Z_r`` is
generated once, up front, from a small Gaussian-copula factor model so arms
sharing a cell, verified start, fingerprint family, or option family are
positively correlated across Monte Carlo worlds. Greedy selection then adds
the arm with the largest
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
_CELL_LOADING = 0.24
_START_LOADING = 0.18
_FINGERPRINT_LOADING = 0.14
_OPTION_FAMILY_LOADING = 0.10
_IDIOSYNCRATIC_LOADING = (
    1.0
    - _CELL_LOADING
    - _START_LOADING
    - _FINGERPRINT_LOADING
    - _OPTION_FAMILY_LOADING
)


class PortfolioError(ValueError):
    """A portfolio selection received invalid candidates or resources."""


def _norm_cdf(z: float) -> float:
    return 0.5 * (1.0 + math.erf(z / math.sqrt(2.0)))


@dataclass(frozen=True)
class ValuedArm:
    """One arm chosen by the portfolio search, with its reproducible basis."""

    identity: ArmIdentity
    expected_cost: Mapping[str, float]
    hard_cost: Mapping[str, float]
    posterior_level: str
    expected_gain: float
    uncertainty: float
    support: int
    reliability_probability: float
    admission_probability: float
    improvement_probability_given_admission: float
    mean_positive_gain: float
    marginal_gain: float
    correlation_penalty: float
    rng_seed: int


def _factor_draws(
    candidates: Sequence[ArmCandidate], *, samples: int, seed: int
) -> List[List[float]]:
    """Deterministic Gaussian-copula latents ``z_r`` for every (sample, arm)."""

    identities = [candidate.identity for candidate in candidates]
    rng = random.Random(derive_seed("portfolio_factor_draws", seed, samples, len(identities)))
    cell_ids = sorted({identity.cell_id for identity in identities})
    start_ids = sorted({candidate.start_state_id or "" for candidate in candidates})
    fingerprints = sorted({candidate.fingerprint_family or "" for candidate in candidates})
    option_families = sorted({candidate.option_family or candidate.identity.option_id for candidate in candidates})
    out: List[List[float]] = [[] for _ in identities]
    for _ in range(samples):
        cell_factor = {cell_id: rng.gauss(0.0, 1.0) for cell_id in cell_ids}
        start_factor = {start_id: rng.gauss(0.0, 1.0) for start_id in start_ids}
        fingerprint_factor = {
            fingerprint: rng.gauss(0.0, 1.0) for fingerprint in fingerprints
        }
        option_factor = {
            family: rng.gauss(0.0, 1.0) for family in option_families
        }
        for index, (identity, candidate) in enumerate(zip(identities, candidates)):
            idiosyncratic = rng.gauss(0.0, 1.0)
            z = (
                math.sqrt(_CELL_LOADING) * cell_factor[identity.cell_id]
                + math.sqrt(_START_LOADING) * start_factor[candidate.start_state_id or ""]
                + math.sqrt(_FINGERPRINT_LOADING)
                * fingerprint_factor[candidate.fingerprint_family or ""]
                + math.sqrt(_OPTION_FAMILY_LOADING)
                * option_factor[candidate.option_family or identity.option_id]
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
    latents = _factor_draws(candidates, samples=samples, seed=seed)
    magnitude_rng = random.Random(derive_seed("portfolio_magnitude_draws", seed, samples, len(identities)))
    draws: List[List[float]] = []
    for candidate, arm_latents in zip(candidates, latents):
        arm_draws = []
        for z in arm_latents:
            # Draw posterior rates inside each Monte Carlo world instead of
            # collapsing sparse evidence to a point estimate.
            p_positive = (
                posterior.sample_reliability_rate(candidate.identity, magnitude_rng)
                * posterior.sample_admission_rate(candidate.identity, magnitude_rng)
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
                float(candidate.expected_cost.get(resource, 0.0)) > remaining + 1e-9
                for resource, remaining in remaining_resources.items()
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
        for resource in tuple(remaining_resources):
            remaining_resources[resource] -= float(
                candidate.expected_cost.get(resource, 0.0)
            )
        snapshot = posterior.snapshot(candidate.identity)
        standalone_value = _mean_positive(candidate_draws)
        selected.append(
            ValuedArm(
                identity=candidate.identity,
                expected_cost=dict(candidate.expected_cost),
                hard_cost=dict(candidate.hard_cost),
                posterior_level=snapshot.hierarchy_level,
                expected_gain=snapshot.expected_gain,
                uncertainty=snapshot.uncertainty,
                support=snapshot.support,
                reliability_probability=snapshot.reliability_probability,
                admission_probability=snapshot.admission_probability,
                improvement_probability_given_admission=(
                    snapshot.improvement_probability_given_admission
                ),
                mean_positive_gain=snapshot.mean_positive_gain,
                marginal_gain=best_marginal,
                correlation_penalty=max(0.0, standalone_value - best_marginal),
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
