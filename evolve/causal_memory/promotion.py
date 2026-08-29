"""Promotion, quarantine, and drift stratification for causal memory.

Promotion is a pure function of repeated support and a positive conservative
(mean-minus-uncertainty) effect -- never of raw production evidence.
Drift stratification re-quarantines an already-promoted record whose most
recent evidence has moved substantially away from its own long-run mean,
protecting against a context whose true effect has shifted underneath it.
"""

from __future__ import annotations

import math
from dataclasses import replace

from evolve.types import CausalMemoryRecord, MemoryStatus


DRIFT_WINDOW = 5
DRIFT_SIGMA = 2.0


def _set_status(record: CausalMemoryRecord, status: MemoryStatus) -> CausalMemoryRecord:
    if record.status == status:
        return record
    updated = replace(record, status=status)
    object.__setattr__(updated, "schema_version", record.schema_version)
    object.__setattr__(updated, "extensions", record.extensions)
    return updated


def evaluate_promotion(record: CausalMemoryRecord) -> CausalMemoryRecord:
    """Promote, reject, or quarantine purely from the record's own evidence."""

    if record.support >= record.promotion_min_support:
        if record.conservative_effect > 0.0:
            return _set_status(record, MemoryStatus.PROMOTED)
        if record.effect_mean + record.uncertainty < 0.0:
            # Confidently negative both directions: no plausible positive
            # effect remains, so further audits would not be a good use of
            # the reserved audit budget.
            return _set_status(record, MemoryStatus.REJECTED)
    return _set_status(record, MemoryStatus.QUARANTINED)


def detect_drift(record: CausalMemoryRecord, *, window: int = DRIFT_WINDOW) -> bool:
    """Flag when the most recent ``window`` effects diverge from the rest."""

    if len(record.effects) < window + 1:
        return False
    recent = record.effects[-window:]
    older = record.effects[:-window]
    recent_mean = sum(recent) / len(recent)
    older_mean = sum(older) / len(older)
    variance = sum((value - record.effect_mean) ** 2 for value in record.effects) / max(
        1, len(record.effects) - 1
    )
    scale = math.sqrt(variance) if variance > 0.0 else max(abs(record.effect_mean), 1e-9)
    return abs(recent_mean - older_mean) > DRIFT_SIGMA * scale


def stratify_drift(
    record: CausalMemoryRecord, *, current_epoch: int, window: int = DRIFT_WINDOW
) -> CausalMemoryRecord:
    """Re-quarantine a promoted record whose recent evidence has drifted.

    Only the most recent window's audit pairs are retained, so the record can
    only re-promote from fresh, undrifted evidence rather than being
    permanently anchored to a stale effect estimate.
    """

    if record.status != MemoryStatus.PROMOTED or not detect_drift(record, window=window):
        return record
    audit_pair_ids = record.audit_pair_ids[-window:]
    propensities = record.propensities[-window:]
    effects = record.effects[-window:]
    n = len(effects)
    effect_mean = sum(effects) / n
    if n > 1:
        variance = sum((value - effect_mean) ** 2 for value in effects) / (n - 1)
        uncertainty = math.sqrt(variance / n)
    else:
        uncertainty = abs(effect_mean)
    updated = replace(
        record,
        status=MemoryStatus.QUARANTINED,
        audit_pair_ids=audit_pair_ids,
        propensities=propensities,
        effects=effects,
        effect_mean=effect_mean,
        uncertainty=uncertainty,
        support=n,
        recency_epoch=current_epoch,
    )
    object.__setattr__(updated, "schema_version", record.schema_version)
    object.__setattr__(updated, "extensions", record.extensions)
    return updated


__all__ = ["DRIFT_SIGMA", "DRIFT_WINDOW", "detect_drift", "evaluate_promotion", "stratify_drift"]
