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
    DIAGNOSTIC_HARNESS_VERSION,
    HARNESS_AUDIT_CONTEXT_VERSION,
    HARNESS_SPEC_API_VERSION,
    HARNESS_SUBSYSTEM_SCHEMA_VERSION,
    HarnessValidationError,
    MatchedHarnessAuditContext,
    baseline_harness_spec,
    diagnostic_harness_spec,
    create_harness_spec,
    validate_harness_spec,
)


def default_harness_registry(*, active_versions=(BASELINE_HARNESS_VERSION,)) -> HarnessRegistry:
    """Register the baseline and trial candidate; activate requested versions."""

    baseline = baseline_harness_spec()
    diagnostic = diagnostic_harness_spec()
    registry = HarnessRegistry().register(baseline).register(diagnostic)
    requested = tuple(active_versions)
    known = {spec.version for spec in registry.specs.values()}
    unknown = sorted(set(requested) - known)
    if unknown:
        raise HarnessRegistryError(
            f"unknown active harness version(s): {unknown}"
        )
    for version in requested:
        registry = registry.activate(registry.spec_for_version(version).harness_id)
    return registry


__all__ = [
    "BASELINE_HARNESS_VERSION",
    "DIAGNOSTIC_HARNESS_VERSION",
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
    "diagnostic_harness_spec",
    "create_harness_spec",
    "default_harness_registry",
    "validate_harness_spec",
]
