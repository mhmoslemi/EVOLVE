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

    def __post_init__(self) -> None:
        for name in ("prior_alpha", "prior_beta"):
            value = getattr(self, name)
            if (
                isinstance(value, bool)
                or not isinstance(value, (int, float))
                or not math.isfinite(float(value))
                or float(value) <= 0.0
            ):
                raise PosteriorError(f"{name} must be finite and positive")
        for name in ("successes", "failures"):
            value = getattr(self, name)
            if isinstance(value, bool) or not isinstance(value, int) or value < 0:
                raise PosteriorError(f"{name} must be a non-negative integer")

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

    def __post_init__(self) -> None:
        if isinstance(self.count, bool) or not isinstance(self.count, int) or self.count < 0:
            raise PosteriorError("resource count must be a non-negative integer")
        for name in ("mean", "m2"):
            value = getattr(self, name)
            if (
                isinstance(value, bool)
                or not isinstance(value, (int, float))
                or not math.isfinite(float(value))
            ):
                raise PosteriorError(f"resource {name} must be finite")
        if float(self.m2) < 0.0:
            raise PosteriorError("resource m2 must be non-negative")
        if self.count == 0 and (float(self.mean) != 0.0 or float(self.m2) != 0.0):
            raise PosteriorError("an empty resource estimate must have zero moments")

    @property
    def variance(self) -> float:
        return self.m2 / (self.count - 1) if self.count > 1 else 0.0

    @property
    def std(self) -> float:
        return math.sqrt(self.variance)

    def update(self, value: float) -> "ResourceStats":
        if (
            isinstance(value, bool)
            or not isinstance(value, (int, float))
            or not math.isfinite(float(value))
            or float(value) < 0.0
        ):
            raise PosteriorError("resource observations must be finite and non-negative")
        count = self.count + 1
        value = float(value)
        delta = value - self.mean
        mean = self.mean + delta / count
        m2 = max(0.0, self.m2 + delta * (value - mean))
        return ResourceStats(count=count, mean=mean, m2=m2)


_MAX_RESERVOIR = 256


@dataclass(frozen=True)
class LevelStats:
    """Everything observed so far at one hierarchy level for one key."""

    reliability: BetaBinomial = field(default_factory=BetaBinomial)
    admission: BetaBinomial = field(default_factory=BetaBinomial)
    improvement_given_admission: BetaBinomial = field(default_factory=BetaBinomial)
    positive_gains: Tuple[float, ...] = ()
    resources: Mapping[str, ResourceStats] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not isinstance(self.reliability, BetaBinomial):
            raise PosteriorError("reliability must be BetaBinomial statistics")
        if not isinstance(self.admission, BetaBinomial):
            raise PosteriorError("admission must be BetaBinomial statistics")
        if not isinstance(self.improvement_given_admission, BetaBinomial):
            raise PosteriorError(
                "improvement_given_admission must be BetaBinomial statistics"
            )
        gains = tuple(self.positive_gains)
        if len(gains) > _MAX_RESERVOIR:
            raise PosteriorError("positive-gain reservoir exceeds its hard bound")
        for gain in gains:
            if (
                isinstance(gain, bool)
                or not isinstance(gain, (int, float))
                or not math.isfinite(float(gain))
                or float(gain) <= 0.0
            ):
                raise PosteriorError("positive-gain observations must be finite and positive")
        resources = dict(self.resources)
        for name, stats in resources.items():
            if not isinstance(name, str) or not name.strip():
                raise PosteriorError("resource names must be non-empty strings")
            if not isinstance(stats, ResourceStats):
                raise PosteriorError("resource entries must be ResourceStats")
        object.__setattr__(self, "positive_gains", tuple(float(gain) for gain in gains))
        object.__setattr__(self, "resources", resources)


@dataclass(frozen=True)
class PosteriorSnapshot:
    """A frozen point-estimate view of one arm's posterior, for logging."""

    hierarchy_level: str
    support: int
    reliability_probability: float
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
                        "reliability": vars(stats.reliability),
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
                raw_key = entry.get("key", ())
                if not isinstance(raw_key, (list, tuple)):
                    raise PosteriorError("posterior hierarchy key must be a list")
                key = tuple(raw_key)
                if key in levels[level]:
                    raise PosteriorError("persisted posterior contains a duplicate key")
                raw_resources = entry.get("resources", {})
                if not isinstance(raw_resources, Mapping):
                    raise PosteriorError("posterior resources must be a mapping")
                resources = {}
                for name, value in raw_resources.items():
                    if not isinstance(value, Mapping):
                        raise PosteriorError(
                            "posterior resource statistics must be a mapping"
                        )
                    resources[name] = ResourceStats(**dict(value))
                raw_gains = entry.get("positive_gains", ())
                if not isinstance(raw_gains, (list, tuple)):
                    raise PosteriorError("positive_gains must be a list")
                levels[level][key] = LevelStats(
                    reliability=BetaBinomial(
                        **dict(entry.get("reliability", {}))
                    ),
                    admission=BetaBinomial(**dict(entry["admission"])),
                    improvement_given_admission=BetaBinomial(
                        **dict(entry["improvement_given_admission"])
                    ),
                    positive_gains=tuple(raw_gains),
                    resources=resources,
                )
        return cls(min_support=payload.get("min_support", 3), levels=levels)

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

        for name, value in (
            ("admitted", admitted),
            ("infrastructure", infrastructure),
            ("record_improved", record_improved),
        ):
            if not isinstance(value, bool):
                raise PosteriorError(f"{name} must be boolean")
        if (
            isinstance(gain, bool)
            or not isinstance(gain, (int, float))
            or not math.isfinite(float(gain))
            or float(gain) < 0.0
        ):
            raise PosteriorError("gain must be finite and non-negative")
        normalized_costs: Dict[str, float] = {}
        for resource, amount in costs.items():
            if not isinstance(resource, str) or not resource.strip():
                raise PosteriorError("resource names must be non-empty strings")
            if (
                isinstance(amount, bool)
                or not isinstance(amount, (int, float))
                or not math.isfinite(float(amount))
                or float(amount) < 0.0
            ):
                raise PosteriorError(
                    f"resource observation {resource!r} must be finite and non-negative"
                )
            normalized_costs[resource] = float(amount)
        levels = {level: dict(bucket) for level, bucket in self.levels.items()}
        for level in HIERARCHY_LEVELS:
            key = hierarchy_key(level, identity)
            stats = levels[level].get(key, LevelStats())
            resources = dict(stats.resources)
            for resource, amount in normalized_costs.items():
                resources[resource] = resources.get(resource, ResourceStats()).update(amount)
            reliability = stats.reliability.update(not infrastructure)
            if infrastructure:
                levels[level][key] = replace(
                    stats, reliability=reliability, resources=resources
                )
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
                reliability=reliability,
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

    def _backoff_for_reliability(
        self, identity: ArmIdentity
    ) -> Tuple[str, LevelStats]:
        for level in HIERARCHY_LEVELS:
            key = hierarchy_key(level, identity)
            stats = self.levels.get(level, {}).get(key)
            if stats is not None and stats.reliability.support >= self.min_support:
                return level, stats
        key = hierarchy_key("global", identity)
        return "global", self.levels.get("global", {}).get(key, LevelStats())

    def _backoff_for_resource(
        self, identity: ArmIdentity, resource: str
    ) -> Tuple[str, ResourceStats]:
        """Back off on resource support independently of scientific support."""

        if not isinstance(resource, str) or not resource.strip():
            raise PosteriorError("resource must be a non-empty string")
        for level in HIERARCHY_LEVELS:
            key = hierarchy_key(level, identity)
            stats = self.levels.get(level, {}).get(key)
            estimate = stats.resources.get(resource) if stats is not None else None
            if estimate is not None and estimate.count >= self.min_support:
                return level, estimate
        global_key = hierarchy_key("global", identity)
        global_stats = self.levels.get("global", {}).get(global_key)
        if global_stats is not None and resource in global_stats.resources:
            return "global", global_stats.resources[resource]
        return "global", ResourceStats()

    def snapshot(self, identity: ArmIdentity) -> PosteriorSnapshot:
        level, stats = self._backoff(identity)
        _gain_level, gain_stats = self._backoff_for_gain(identity)
        _reliability_level, reliability_stats = self._backoff_for_reliability(
            identity
        )
        reliability = reliability_stats.reliability.mean
        p_admit = stats.admission.mean
        p_improve = stats.improvement_given_admission.mean if stats.admission.support else 0.5
        p_positive = reliability * p_admit * p_improve
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
            reliability_probability=reliability,
            admission_probability=p_admit,
            improvement_probability_given_admission=p_improve,
            positive_probability=p_positive,
            mean_positive_gain=mean_gain,
            uncertainty=uncertainty,
            expected_gain=p_positive * mean_gain,
        )

    def resource_estimate(self, identity: ArmIdentity, resource: str) -> ResourceStats:
        _level, estimate = self._backoff_for_resource(identity, resource)
        return estimate

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

    def sample_reliability_rate(self, identity: ArmIdentity, rng: random.Random) -> float:
        _, stats = self._backoff_for_reliability(identity)
        return stats.reliability.sample(rng)

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
