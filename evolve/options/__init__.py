"""Executable option state machines and branch execution for EVOLVE."""

from .base import (
    OPTION_EXECUTION_SCHEMA_VERSION,
    OPTION_SPEC_IDENTITY_VERSION,
    ExecutableOption,
    OptionContext,
    OptionDecisionKind,
    OptionEligibilityError,
    OptionError,
    OptionExecutionError,
    OptionState,
    OptionStepDecision,
    OptionStepInput,
    create_option_spec,
    option_spec_identity_payload,
    validate_option_spec_identity,
)
from .branch import (
    BranchExecution,
    BranchExecutionError,
    BranchStepExecutor,
    BranchStepRequest,
    BranchStepResult,
    PolicySegment,
    build_option_context,
    execute_branch,
)
from .builtins import (
    CHALLENGER_ATTACK_STATE_MACHINE,
    DIAGNOSTIC_REPAIR_STATE_MACHINE,
    EXPLORE_STATE_MACHINE,
    FRESH_REFINEMENT_CONTROL_STATE_MACHINE,
    MECHANIST_DEVELOP_STATE_MACHINE,
    MATCHED_CONTINUATION_STATE_MACHINE,
    PROPOSE_CAPABILITY,
    ChallengerAttackOption,
    DiagnosticRepairOption,
    ExploreOption,
    FreshRefinementControlOption,
    MechanistDevelopOption,
    MatchedContinuationOption,
    create_challenger_attack_option_spec,
    create_diagnostic_repair_option_spec,
    create_explore_option_spec,
    create_fresh_refinement_control_option_spec,
    create_mechanist_develop_option_spec,
    create_matched_continuation_option_spec,
)
from .registry import OptionRegistry, OptionRegistryError


def production_option_registry(
    *,
    harness_eligibility,
    max_horizon: int,
    hard_cost=None,
    expected_cost=None,
) -> OptionRegistry:
    """Register the three method-defined builtin options for one harness set.

    ``hard_cost``/``expected_cost`` default to one verifier call per horizon
    step, which every builtin option consumes at most.  Callers running a
    stricter resource policy may pass explicit per-resource bounds.
    """

    harness_ids = tuple(harness_eligibility)
    bound = dict(hard_cost) if hard_cost is not None else {"verifier_calls": float(max_horizon)}
    expected = dict(expected_cost) if expected_cost is not None else dict(bound)
    registry = OptionRegistry()
    registry = registry.register(
        create_explore_option_spec(
            max_horizon=max_horizon,
            hard_cost=bound,
            expected_cost=expected,
            harness_eligibility=harness_ids,
        ),
        ExploreOption,
    )
    registry = registry.register(
        create_mechanist_develop_option_spec(
            max_horizon=max_horizon,
            hard_cost=bound,
            expected_cost=expected,
            harness_eligibility=harness_ids,
        ),
        MechanistDevelopOption,
    )
    registry = registry.register(
        create_challenger_attack_option_spec(
            max_horizon=max_horizon,
            hard_cost=bound,
            expected_cost=expected,
            harness_eligibility=harness_ids,
        ),
        ChallengerAttackOption,
    )
    registry = registry.register(
        create_matched_continuation_option_spec(
            max_horizon=max_horizon,
            hard_cost=bound,
            expected_cost=expected,
            harness_eligibility=harness_ids,
        ),
        MatchedContinuationOption,
    )
    registry = registry.register(
        create_diagnostic_repair_option_spec(
            max_horizon=max_horizon,
            hard_cost={"verifier_calls": 1.0},
            expected_cost={"verifier_calls": 1.0},
            harness_eligibility=harness_ids,
        ),
        DiagnosticRepairOption,
    )
    registry = registry.register(
        create_fresh_refinement_control_option_spec(
            hard_cost={"verifier_calls": 1.0},
            expected_cost={"verifier_calls": 1.0},
            harness_eligibility=harness_ids,
        ),
        FreshRefinementControlOption,
    )
    return registry


__all__ = [
    "CHALLENGER_ATTACK_STATE_MACHINE",
    "DIAGNOSTIC_REPAIR_STATE_MACHINE",
    "EXPLORE_STATE_MACHINE",
    "FRESH_REFINEMENT_CONTROL_STATE_MACHINE",
    "MECHANIST_DEVELOP_STATE_MACHINE",
    "MATCHED_CONTINUATION_STATE_MACHINE",
    "OPTION_EXECUTION_SCHEMA_VERSION",
    "OPTION_SPEC_IDENTITY_VERSION",
    "PROPOSE_CAPABILITY",
    "BranchExecution",
    "BranchExecutionError",
    "BranchStepExecutor",
    "BranchStepRequest",
    "BranchStepResult",
    "ChallengerAttackOption",
    "DiagnosticRepairOption",
    "ExecutableOption",
    "ExploreOption",
    "FreshRefinementControlOption",
    "MechanistDevelopOption",
    "MatchedContinuationOption",
    "OptionContext",
    "OptionDecisionKind",
    "OptionEligibilityError",
    "OptionError",
    "OptionExecutionError",
    "OptionRegistry",
    "OptionRegistryError",
    "OptionState",
    "OptionStepDecision",
    "OptionStepInput",
    "PolicySegment",
    "build_option_context",
    "create_challenger_attack_option_spec",
    "create_diagnostic_repair_option_spec",
    "create_explore_option_spec",
    "create_fresh_refinement_control_option_spec",
    "create_mechanist_develop_option_spec",
    "create_matched_continuation_option_spec",
    "create_option_spec",
    "execute_branch",
    "option_spec_identity_payload",
    "production_option_registry",
    "validate_option_spec_identity",
]
