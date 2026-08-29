"""Content-addressed harness specs, matched audits, and version registry."""

from .registry import (
    HARNESS_REGISTRY_SCHEMA_VERSION,
    HarnessEffectSummary,
    HarnessPromotionError,
    HarnessRegistry,
    HarnessRegistryError,
    HarnessTrialRecord,
)
from .spec import (
    BASELINE_HARNESS_VERSION,
    HARNESS_AUDIT_CONTEXT_VERSION,
    HARNESS_SPEC_API_VERSION,
    HARNESS_SUBSYSTEM_SCHEMA_VERSION,
    HarnessValidationError,
    MatchedHarnessAuditContext,
    baseline_harness_spec,
    create_harness_spec,
    validate_harness_spec,
)


def default_harness_registry(*, active_versions=(BASELINE_HARNESS_VERSION,)) -> HarnessRegistry:
    """The one production harness registered and active at Phase 4 baseline."""

    baseline = baseline_harness_spec()
    registry = HarnessRegistry().register(baseline)
    if baseline.version in tuple(active_versions):
        registry = registry.activate(baseline.harness_id)
    return registry


__all__ = [
    "BASELINE_HARNESS_VERSION",
    "HARNESS_AUDIT_CONTEXT_VERSION",
    "HARNESS_REGISTRY_SCHEMA_VERSION",
    "HARNESS_SPEC_API_VERSION",
    "HARNESS_SUBSYSTEM_SCHEMA_VERSION",
    "HarnessEffectSummary",
    "HarnessPromotionError",
    "HarnessRegistry",
    "HarnessRegistryError",
    "HarnessTrialRecord",
    "HarnessValidationError",
    "MatchedHarnessAuditContext",
    "baseline_harness_spec",
    "create_harness_spec",
    "default_harness_registry",
    "validate_harness_spec",
]
