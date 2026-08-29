"""Hierarchical sparse-data zero-inflated record-gain posterior.

The first implementable EVOLVE baseline (AGENTS.md, "Posterior allocation"):
a Beta-Binomial admission/improvement model with backoff from the exact arm
through option-role, role, cell, and global levels, and a Bayesian-bootstrap
tail over observed positive record gains at whichever level has support.
Everything here is pure and reproducible from a supplied RNG seed; it never
touches the archive, budget, or portfolio search directly.
"""

from __future__ import annotations

import math
import random
from dataclasses import dataclass, field, replace
from typing import Dict, Mapping, Tuple

from .arms import ArmIdentity


POSTERIOR_VERSION = "zero_inflated_tail_v1"

HIERARCHY_LEVELS: Tuple[str, ...] = ("arm", "option_role", "role", "cell", "global")


class PosteriorError(ValueError):
    """A posterior update or query received an invalid observation or key."""


def hierarchy_key(level: str, identity: ArmIdentity) -> Tuple[object, ...]:
    if level == "arm":
        return identity.key()
    if level == "option_role":
        return (identity.role.value, identity.option_id)
    if level == "role":
        return (identity.role.value,)
    if level == "cell":
        return (identity.cell_id,)
    if level == "global":
        return ()
    raise PosteriorError(f"unknown hierarchy level {level!r}")


@dataclass(frozen=True)
class BetaBinomial:
    """Conjugate Beta-Binomial belief over one Bernoulli rate."""

    prior_alpha: float = 1.0
    prior_beta: float = 1.0
    successes: int = 0
    failures: int = 0

    @property
    def alpha(self) -> float:
        return self.prior_alpha + self.successes

    @property
    def beta(self) -> float:
        return self.prior_beta + self.failures

    @property
    def support(self) -> int:
        return self.successes + self.failures

    @property
    def mean(self) -> float:
        return self.alpha / (self.alpha + self.beta)

    @property
    def variance(self) -> float:
        a, b = self.alpha, self.beta
        return (a * b) / ((a + b) ** 2 * (a + b + 1.0))

    def update(self, success: bool) -> "BetaBinomial":
        if success:
            return replace(self, successes=self.successes + 1)
        return replace(self, failures=self.failures + 1)

    def sample(self, rng: random.Random) -> float:
        return rng.betavariate(max(self.alpha, 1e-6), max(self.beta, 1e-6))


@dataclass(frozen=True)
class ResourceStats:
    """Welford running mean/variance for one predicted resource."""

    count: int = 0
    mean: float = 0.0
    m2: float = 0.0

    @property
    def variance(self) -> float:
        return self.m2 / (self.count - 1) if self.count > 1 else 0.0

    @property
    def std(self) -> float:
        return math.sqrt(self.variance)

    def update(self, value: float) -> "ResourceStats":
        count = self.count + 1
        delta = value - self.mean
        mean = self.mean + delta / count
        m2 = self.m2 + delta * (value - mean)
        return ResourceStats(count=count, mean=mean, m2=m2)


_MAX_RESERVOIR = 256


@dataclass(frozen=True)
class LevelStats:
    """Everything observed so far at one hierarchy level for one key."""

    admission: BetaBinomial = field(default_factory=BetaBinomial)
    improvement_given_admission: BetaBinomial = field(default_factory=BetaBinomial)
    positive_gains: Tuple[float, ...] = ()
    resources: Mapping[str, ResourceStats] = field(default_factory=dict)


@dataclass(frozen=True)
class PosteriorSnapshot:
    """A frozen point-estimate view of one arm's posterior, for logging."""

    hierarchy_level: str
    support: int
    admission_probability: float
    improvement_probability_given_admission: float
    positive_probability: float
    mean_positive_gain: float
    uncertainty: float
    expected_gain: float


@dataclass(frozen=True)
class PosteriorStore:
    """Immutable hierarchical posterior over admission and record gain."""

    min_support: int = 3
    levels: Mapping[str, Mapping[Tuple[object, ...], LevelStats]] = field(
        default_factory=lambda: {level: {} for level in HIERARCHY_LEVELS}
    )

    def __post_init__(self) -> None:
        levels: Dict[str, Dict[Tuple[object, ...], LevelStats]] = {
            level: dict(self.levels.get(level, {})) for level in HIERARCHY_LEVELS
        }
        object.__setattr__(self, "levels", levels)
        if isinstance(self.min_support, bool) or not isinstance(self.min_support, int) or self.min_support < 1:
            raise PosteriorError("min_support must be a positive integer")

    def to_dict(self) -> Mapping[str, object]:
        return {
            "version": POSTERIOR_VERSION,
            "min_support": self.min_support,
            "levels": {
                level: [
                    {
                        "key": list(key),
                        "admission": vars(stats.admission),
                        "improvement_given_admission": vars(
                            stats.improvement_given_admission
                        ),
                        "positive_gains": list(stats.positive_gains),
                        "resources": {
                            name: vars(resource)
                            for name, resource in sorted(stats.resources.items())
                        },
                    }
                    for key, stats in sorted(
                        bucket.items(), key=lambda item: repr(item[0])
                    )
                ]
                for level, bucket in self.levels.items()
            },
        }

    @classmethod
    def from_dict(cls, payload: Mapping[str, object]) -> "PosteriorStore":
        if payload.get("version") != POSTERIOR_VERSION:
            raise PosteriorError("unsupported persisted posterior version")
        raw_levels = payload.get("levels")
        if not isinstance(raw_levels, Mapping):
            raise PosteriorError("persisted posterior levels must be a mapping")
        levels: Dict[str, Dict[Tuple[object, ...], LevelStats]] = {
            level: {} for level in HIERARCHY_LEVELS
        }
        for level in HIERARCHY_LEVELS:
            entries = raw_levels.get(level, ())
            if not isinstance(entries, (list, tuple)):
                raise PosteriorError(f"posterior level {level!r} must be a list")
            for entry in entries:
                if not isinstance(entry, Mapping):
                    raise PosteriorError("posterior entry must be a mapping")
                key = tuple(entry.get("key", ()))
                resources = {
                    str(name): ResourceStats(**dict(value))
                    for name, value in dict(entry.get("resources", {})).items()
                }
                levels[level][key] = LevelStats(
                    admission=BetaBinomial(**dict(entry["admission"])),
                    improvement_given_admission=BetaBinomial(
                        **dict(entry["improvement_given_admission"])
                    ),
                    positive_gains=tuple(float(x) for x in entry.get("positive_gains", ())),
                    resources=resources,
                )
        return cls(min_support=int(payload.get("min_support", 3)), levels=levels)

    def observe(
        self,
        identity: ArmIdentity,
        *,
        admitted: bool,
        infrastructure: bool,
        record_improved: bool,
        gain: float,
        costs: Mapping[str, float],
    ) -> "PosteriorStore":
        """Fold one closed, non-infrastructure branch outcome into every level.

        Infrastructure-excluded observations never touch admission/tail
        statistics but still update resource-cost estimates, matching the
        method requirement that infra failure still informs resource-risk
        models.
        """

        if isinstance(admitted, bool) is False:
            raise PosteriorError("admitted must be boolean")
        levels = {level: dict(bucket) for level, bucket in self.levels.items()}
        for level in HIERARCHY_LEVELS:
            key = hierarchy_key(level, identity)
            stats = levels[level].get(key, LevelStats())
            resources = dict(stats.resources)
            for resource, amount in costs.items():
                resources[resource] = resources.get(resource, ResourceStats()).update(float(amount))
            if infrastructure:
                levels[level][key] = replace(stats, resources=resources)
                continue
            admission = stats.admission.update(admitted)
            improvement = stats.improvement_given_admission
            positive_gains = stats.positive_gains
            if admitted:
                improved = bool(record_improved and float(gain) > 0.0)
                improvement = improvement.update(improved)
                if improved:
                    positive_gains = (stats.positive_gains + (float(gain),))[-_MAX_RESERVOIR:]
            levels[level][key] = LevelStats(
                admission=admission,
                improvement_given_admission=improvement,
                positive_gains=positive_gains,
                resources=resources,
            )
        return replace(self, levels=levels)

    def _backoff(self, identity: ArmIdentity) -> Tuple[str, LevelStats]:
        for level in HIERARCHY_LEVELS:
            key = hierarchy_key(level, identity)
            stats = self.levels.get(level, {}).get(key)
            if stats is not None and stats.admission.support >= self.min_support:
                return level, stats
        # Absolute fallback: an uninformative prior at the coarsest level.
        key = hierarchy_key("global", identity)
        return "global", self.levels.get("global", {}).get(key, LevelStats())

    def _backoff_for_gain(self, identity: ArmIdentity) -> Tuple[str, LevelStats]:
        for level in HIERARCHY_LEVELS:
            key = hierarchy_key(level, identity)
            stats = self.levels.get(level, {}).get(key)
            if stats is not None and len(stats.positive_gains) >= self.min_support:
                return level, stats
        return self._backoff(identity)

    def snapshot(self, identity: ArmIdentity) -> PosteriorSnapshot:
        level, stats = self._backoff(identity)
        gain_level, gain_stats = self._backoff_for_gain(identity)
        p_admit = stats.admission.mean
        p_improve = stats.improvement_given_admission.mean if stats.admission.support else 0.5
        p_positive = p_admit * p_improve
        if gain_stats.positive_gains:
            mean_gain = sum(gain_stats.positive_gains) / len(gain_stats.positive_gains)
            n = len(gain_stats.positive_gains)
            variance = (
                sum((g - mean_gain) ** 2 for g in gain_stats.positive_gains) / (n - 1)
                if n > 1
                else mean_gain ** 2
            )
            uncertainty = math.sqrt(variance / n)
        else:
            mean_gain = 0.0
            uncertainty = 0.0
        return PosteriorSnapshot(
            hierarchy_level=level,
            support=stats.admission.support,
            admission_probability=p_admit,
            improvement_probability_given_admission=p_improve,
            positive_probability=p_positive,
            mean_positive_gain=mean_gain,
            uncertainty=uncertainty,
            expected_gain=p_positive * mean_gain,
        )

    def resource_estimate(self, identity: ArmIdentity, resource: str) -> ResourceStats:
        level, stats = self._backoff(identity)
        return stats.resources.get(resource, ResourceStats())

    def sample_positive_gain(self, identity: ArmIdentity, rng: random.Random) -> float:
        """One Bayesian-bootstrap draw of the positive-gain magnitude.

        Dirichlet(1,...,1) resampling weights are drawn as normalized
        Exponential(1) variates -- the standard Bayesian-bootstrap
        construction -- over whichever backoff level has enough positive
        observations; an empty reservoir returns zero.
        """

        _, stats = self._backoff_for_gain(identity)
        gains = stats.positive_gains
        if not gains:
            return 0.0
        weights = [rng.expovariate(1.0) for _ in gains]
        total = sum(weights)
        if total <= 0.0:
            return sum(gains) / len(gains)
        return sum(w * g for w, g in zip(weights, gains)) / total

    def sample_admission_rate(self, identity: ArmIdentity, rng: random.Random) -> float:
        _, stats = self._backoff(identity)
        return stats.admission.sample(rng)

    def sample_improvement_rate(self, identity: ArmIdentity, rng: random.Random) -> float:
        _, stats = self._backoff(identity)
        if stats.admission.support == 0:
            return 0.5
        return stats.improvement_given_admission.sample(rng)


__all__ = [
    "HIERARCHY_LEVELS",
    "POSTERIOR_VERSION",
    "BetaBinomial",
    "LevelStats",
    "PosteriorError",
    "PosteriorSnapshot",
    "PosteriorStore",
    "ResourceStats",
    "hierarchy_key",
]
