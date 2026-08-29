"""Building and updating causal option-memory records from audit effects.

A :class:`~evolve.types.CausalMemoryRecord` stores evidence about
interventions, never a summary of successful trajectories: it only grows
from :class:`~evolve.audits.effects.AuditEffect` values tied to a closed,
preassigned :class:`~evolve.types.AuditPair`.
"""

from __future__ import annotations

import math
from dataclasses import replace
from typing import Any, Mapping

from evolve.audits.effects import AuditEffect
from evolve.ids import content_id
from evolve.types import AuditPair, CausalMemoryRecord, MemoryStatus


class CausalMemoryError(ValueError):
    """A causal memory update is inconsistent with its record or evidence."""


def memory_id_for(*, context: Mapping[str, Any], intervention_option_id: str) -> str:
    return content_id(
        "causal_memory",
        {"context": dict(context), "intervention_option_id": intervention_option_id},
    )


def new_memory_record(
    *,
    context: Mapping[str, Any],
    intervention_option_id: str,
    scope: str,
    recency_epoch: int,
    promotion_min_support: int = 2,
) -> CausalMemoryRecord:
    return CausalMemoryRecord(
        memory_id=memory_id_for(context=context, intervention_option_id=intervention_option_id),
        context=dict(context),
        intervention_option_id=intervention_option_id,
        audit_pair_ids=(),
        propensities=(),
        effects=(),
        effect_mean=0.0,
        uncertainty=0.0,
        support=0,
        recency_epoch=recency_epoch,
        scope=scope,
        contraindications=(),
        lineage_ids=(),
        status=MemoryStatus.QUARANTINED,
        promotion_min_support=promotion_min_support,
    )


def _effect_moments(effects):
    n = len(effects)
    mean = sum(effects) / n
    if n > 1:
        variance = sum((value - mean) ** 2 for value in effects) / (n - 1)
        uncertainty = math.sqrt(variance / n)
    else:
        # A single observation cannot support a confident conservative
        # estimate: treat its full magnitude as uncertainty, matching the
        # convention already used for harness trials and posterior gains.
        uncertainty = abs(mean)
    return mean, uncertainty


def add_effect(
    record: CausalMemoryRecord,
    *,
    pair: AuditPair,
    effect: AuditEffect,
    recency_epoch: int,
) -> CausalMemoryRecord:
    """Fold one closed matched-pair effect into a record, idempotently."""

    if pair.audit_id != effect.audit_id:
        raise CausalMemoryError("audit pair and effect reference different audits")
    if pair.intervention_option_id != record.intervention_option_id:
        raise CausalMemoryError("audit pair intervention does not match this memory record")
    if pair.audit_id in record.audit_pair_ids:
        return record

    audit_pair_ids = record.audit_pair_ids + (pair.audit_id,)
    propensities = record.propensities + (pair.assignment_probability,)
    effects = record.effects + (effect.effect,)
    effect_mean, uncertainty = _effect_moments(effects)
    lineage_ids = tuple(dict.fromkeys(record.lineage_ids + (pair.start_state_id,)))

    updated = replace(
        record,
        audit_pair_ids=audit_pair_ids,
        propensities=propensities,
        effects=effects,
        effect_mean=effect_mean,
        uncertainty=uncertainty,
        support=len(effects),
        recency_epoch=max(record.recency_epoch, recency_epoch),
        lineage_ids=lineage_ids,
    )
    object.__setattr__(updated, "schema_version", record.schema_version)
    object.__setattr__(updated, "extensions", record.extensions)
    return updated


def add_contraindication(record: CausalMemoryRecord, tag: str) -> CausalMemoryRecord:
    if not isinstance(tag, str) or not tag.strip():
        raise CausalMemoryError("contraindication tag must be a non-empty string")
    if tag in record.contraindications:
        return record
    updated = replace(record, contraindications=record.contraindications + (tag,))
    object.__setattr__(updated, "schema_version", record.schema_version)
    object.__setattr__(updated, "extensions", record.extensions)
    return updated


__all__ = [
    "CausalMemoryError",
    "add_contraindication",
    "add_effect",
    "memory_id_for",
    "new_memory_record",
]
