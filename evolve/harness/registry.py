"""Registered harness versions and conservative-evidence promotion.

The harness is part of the allocation arm, never a hidden constant or final
judge: this registry only tracks which harness versions currently exist and
are eligible for allocation, and accumulates matched-audit trial evidence
that can promote a candidate version once its conservative relative record
gain is repeatedly positive.  Harness-local scores never admit a candidate or
set the record; ``incumbent_gain``/``candidate_gain`` here are the caller's
already-verifier-derived, problem-normalized record gains for each matched
branch, not a harness-local score.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field, replace
from typing import Any, Mapping, Tuple

from evolve.ids import validate_id
from evolve.types import FrozenDict
from .spec import HarnessSpec, HarnessValidationError, MatchedHarnessAuditContext, validate_harness_spec


HARNESS_REGISTRY_SCHEMA_VERSION = 1


class HarnessRegistryError(HarnessValidationError):
    """A harness registry operation is inconsistent with recorded evidence."""


class HarnessPromotionError(HarnessRegistryError):
    """A harness version lacks the repeated, conservative evidence to promote."""


def _nonnegative(value: Any, name: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise HarnessRegistryError(f"{name} must be numeric")
    result = float(value)
    if not math.isfinite(result) or result < 0.0:
        raise HarnessRegistryError(f"{name} must be finite and non-negative")
    return result


def _resource_map(value: Mapping[str, Any], name: str) -> FrozenDict:
    for resource, amount in value.items():
        if not isinstance(resource, str) or not resource.strip():
            raise HarnessRegistryError(f"{name} resource names must be non-empty")
        _nonnegative(amount, f"{name}.{resource}")
    return FrozenDict(value)


@dataclass(frozen=True)
class HarnessTrialRecord:
    """One closed matched-harness audit outcome, in problem-normalized gain."""

    context_id: str
    epoch: int
    incumbent_harness_id: str
    incumbent_harness_version: str
    candidate_harness_id: str
    candidate_harness_version: str
    incumbent_gain: float
    candidate_gain: float
    incumbent_cost: FrozenDict = field(default_factory=FrozenDict)
    candidate_cost: FrozenDict = field(default_factory=FrozenDict)
    incumbent_valid: bool = True
    candidate_valid: bool = True
    incumbent_failure_kind: str = "none"
    candidate_failure_kind: str = "none"

    def __post_init__(self) -> None:
        validate_id(self.context_id, "harness_audit")
        if isinstance(self.epoch, bool) or not isinstance(self.epoch, int) or self.epoch < 0:
            raise HarnessRegistryError("epoch must be a non-negative integer")
        validate_id(self.incumbent_harness_id, "harness")
        validate_id(self.candidate_harness_id, "harness")
        if self.incumbent_harness_id == self.candidate_harness_id:
            raise HarnessRegistryError("a harness trial must compare two distinct harness IDs")
        _nonnegative(self.incumbent_gain, "incumbent_gain")
        _nonnegative(self.candidate_gain, "candidate_gain")
        object.__setattr__(self, "incumbent_cost", _resource_map(self.incumbent_cost, "incumbent_cost"))
        object.__setattr__(self, "candidate_cost", _resource_map(self.candidate_cost, "candidate_cost"))
        for value, name in (
            (self.incumbent_valid, "incumbent_valid"),
            (self.candidate_valid, "candidate_valid"),
        ):
            if not isinstance(value, bool):
                raise HarnessRegistryError(f"{name} must be boolean")

    @classmethod
    def from_context(
        cls,
        context: MatchedHarnessAuditContext,
        *,
        epoch: int,
        incumbent_gain: float,
        candidate_gain: float,
        incumbent_cost: Mapping[str, Any] = FrozenDict(),
        candidate_cost: Mapping[str, Any] = FrozenDict(),
        incumbent_valid: bool = True,
        candidate_valid: bool = True,
        incumbent_failure_kind: str = "none",
        candidate_failure_kind: str = "none",
    ) -> "HarnessTrialRecord":
        if not isinstance(context, MatchedHarnessAuditContext):
            raise HarnessRegistryError("expected a MatchedHarnessAuditContext")
        return cls(
            context_id=context.context_id,
            epoch=epoch,
            incumbent_harness_id=context.incumbent_harness_id,
            incumbent_harness_version=context.incumbent_harness_version,
            candidate_harness_id=context.candidate_harness_id,
            candidate_harness_version=context.candidate_harness_version,
            incumbent_gain=incumbent_gain,
            candidate_gain=candidate_gain,
            incumbent_cost=incumbent_cost,
            candidate_cost=candidate_cost,
            incumbent_valid=incumbent_valid,
            candidate_valid=candidate_valid,
            incumbent_failure_kind=incumbent_failure_kind,
            candidate_failure_kind=candidate_failure_kind,
        )

    @property
    def relative_gain(self) -> float:
        return float(self.candidate_gain - self.incumbent_gain)


@dataclass(frozen=True)
class HarnessEffectSummary:
    """Conservative repeated-evidence summary for one candidate harness."""

    harness_id: str
    trials: int
    mean_relative_gain: float
    uncertainty: float
    validity_rate: float

    @property
    def conservative_gain(self) -> float:
        return float(self.mean_relative_gain - self.uncertainty)


@dataclass(frozen=True)
class HarnessRegistry:
    """Registered harness specs, active versions, and matched-trial evidence."""

    specs: Mapping[str, HarnessSpec] = field(default_factory=dict)
    active_ids: Tuple[str, ...] = field(default_factory=tuple)
    trials: Tuple[HarnessTrialRecord, ...] = field(default_factory=tuple)
    schema_version: int = HARNESS_REGISTRY_SCHEMA_VERSION

    def __post_init__(self) -> None:
        object.__setattr__(self, "specs", dict(self.specs))
        for harness_id, spec in self.specs.items():
            if spec.harness_id != harness_id:
                raise HarnessRegistryError(f"harness_id key mismatch for {harness_id}")
            validate_harness_spec(spec)
        object.__setattr__(self, "active_ids", tuple(dict.fromkeys(self.active_ids)))
        for harness_id in self.active_ids:
            if harness_id not in self.specs:
                raise HarnessRegistryError(f"cannot activate unregistered harness {harness_id!r}")
        seen = set()
        for trial in self.trials:
            if trial.context_id in seen:
                raise HarnessRegistryError(f"duplicate harness trial for {trial.context_id}")
            seen.add(trial.context_id)
            if trial.incumbent_harness_id not in self.specs or trial.candidate_harness_id not in self.specs:
                raise HarnessRegistryError(
                    "harness trial references an unregistered harness"
                )

    def to_dict(self) -> Mapping[str, Any]:
        return {
            "schema_version": self.schema_version,
            "specs": [
                spec.to_dict() for spec in sorted(self.specs.values(), key=lambda item: item.harness_id)
            ],
            "active_ids": list(self.active_ids),
            "trials": [
                {
                    "context_id": trial.context_id,
                    "epoch": trial.epoch,
                    "incumbent_harness_id": trial.incumbent_harness_id,
                    "incumbent_harness_version": trial.incumbent_harness_version,
                    "candidate_harness_id": trial.candidate_harness_id,
                    "candidate_harness_version": trial.candidate_harness_version,
                    "incumbent_gain": trial.incumbent_gain,
                    "candidate_gain": trial.candidate_gain,
                    "incumbent_cost": dict(trial.incumbent_cost),
                    "candidate_cost": dict(trial.candidate_cost),
                    "incumbent_valid": trial.incumbent_valid,
                    "candidate_valid": trial.candidate_valid,
                    "incumbent_failure_kind": trial.incumbent_failure_kind,
                    "candidate_failure_kind": trial.candidate_failure_kind,
                }
                for trial in self.trials
            ],
        }

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> "HarnessRegistry":
        if payload.get("schema_version") != HARNESS_REGISTRY_SCHEMA_VERSION:
            raise HarnessRegistryError("unsupported persisted harness registry schema")
        specs = {
            spec.harness_id: spec
            for spec in (HarnessSpec.from_dict(item) for item in payload.get("specs", ()))
        }
        trials = tuple(
            HarnessTrialRecord(
                **{
                    **dict(item),
                    "incumbent_cost": FrozenDict(item.get("incumbent_cost", {})),
                    "candidate_cost": FrozenDict(item.get("candidate_cost", {})),
                }
            )
            for item in payload.get("trials", ())
        )
        return cls(
            specs=specs,
            active_ids=tuple(payload.get("active_ids", ())),
            trials=trials,
            schema_version=payload["schema_version"],
        )

    def register(self, spec: HarnessSpec) -> "HarnessRegistry":
        validate_harness_spec(spec)
        existing = self.specs.get(spec.harness_id)
        if existing is not None:
            if existing.to_dict() != spec.to_dict():
                raise HarnessRegistryError(f"harness_id collision for {spec.harness_id}")
            return self
        specs = dict(self.specs)
        specs[spec.harness_id] = spec
        return replace(self, specs=specs)

    def activate(self, harness_id: str) -> "HarnessRegistry":
        if harness_id not in self.specs:
            raise HarnessRegistryError(f"cannot activate unregistered harness {harness_id!r}")
        if harness_id in self.active_ids:
            return self
        return replace(self, active_ids=self.active_ids + (harness_id,))

    def spec(self, harness_id: str) -> HarnessSpec:
        try:
            return self.specs[harness_id]
        except KeyError as exc:
            raise HarnessRegistryError(f"unknown harness_id {harness_id!r}") from exc

    def spec_for_version(self, version: str) -> HarnessSpec:
        matches = [spec for spec in self.specs.values() if spec.version == version]
        if not matches:
            raise HarnessRegistryError(f"no registered harness has version {version!r}")
        if len(matches) > 1:
            raise HarnessRegistryError(f"harness version {version!r} is ambiguous across specs")
        return matches[0]

    def is_active(self, harness_id: str) -> bool:
        return harness_id in self.active_ids

    def record_trial(self, trial: HarnessTrialRecord) -> "HarnessRegistry":
        if not isinstance(trial, HarnessTrialRecord):
            raise HarnessRegistryError("expected a HarnessTrialRecord")
        for existing in self.trials:
            if existing.context_id != trial.context_id:
                continue
            if existing == trial:
                return self
            raise HarnessRegistryError(
                f"harness trial collision for context {trial.context_id}"
            )
        if trial.incumbent_harness_id not in self.specs or trial.candidate_harness_id not in self.specs:
            raise HarnessRegistryError("harness trial references an unregistered harness")
        return replace(self, trials=self.trials + (trial,))

    def effect_summary(self, harness_id: str) -> HarnessEffectSummary:
        """Conservative mean-minus-uncertainty gain of ``harness_id`` as candidate."""

        matched = tuple(
            trial
            for trial in self.trials
            if trial.candidate_harness_id == harness_id
            and trial.incumbent_failure_kind != "infrastructure"
            and trial.candidate_failure_kind != "infrastructure"
        )
        if not matched:
            return HarnessEffectSummary(
                harness_id=harness_id, trials=0, mean_relative_gain=0.0,
                uncertainty=0.0, validity_rate=0.0,
            )
        gains = [trial.relative_gain for trial in matched]
        n = len(gains)
        mean = sum(gains) / n
        if n > 1:
            variance = sum((value - mean) ** 2 for value in gains) / (n - 1)
            uncertainty = math.sqrt(variance / n)
        else:
            # A single trial cannot support a confident conservative estimate:
            # its full magnitude is treated as uncertainty, mirroring
            # CausalMemoryRecord's conservative_effect convention.
            uncertainty = abs(mean)
        validity_rate = sum(1 for trial in matched if trial.candidate_valid) / n
        return HarnessEffectSummary(
            harness_id=harness_id,
            trials=n,
            mean_relative_gain=mean,
            uncertainty=uncertainty,
            validity_rate=validity_rate,
        )

    def promote(
        self,
        harness_id: str,
        *,
        min_trials: int,
        min_conservative_gain: float = 0.0,
    ) -> "HarnessRegistry":
        """Activate ``harness_id`` only from repeated, conservative positive evidence."""

        if harness_id not in self.specs:
            raise HarnessRegistryError(f"cannot promote unregistered harness {harness_id!r}")
        if isinstance(min_trials, bool) or not isinstance(min_trials, int) or min_trials < 1:
            raise HarnessRegistryError("min_trials must be a positive integer")
        summary = self.effect_summary(harness_id)
        if summary.trials < min_trials:
            raise HarnessPromotionError(
                f"harness {harness_id!r} has {summary.trials} trial(s); needs >= {min_trials}"
            )
        if summary.conservative_gain <= min_conservative_gain:
            raise HarnessPromotionError(
                f"harness {harness_id!r} conservative gain {summary.conservative_gain:g} "
                f"does not exceed {min_conservative_gain:g}"
            )
        return self.activate(harness_id)


__all__ = [
    "HARNESS_REGISTRY_SCHEMA_VERSION",
    "HarnessEffectSummary",
    "HarnessPromotionError",
    "HarnessRegistry",
    "HarnessRegistryError",
    "HarnessTrialRecord",
]
