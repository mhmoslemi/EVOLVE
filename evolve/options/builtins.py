"""Production executable option state machines for the three EVOLVE roles.

Every branch step ultimately dispatches one role-policy generation through the
``propose`` capability; harness prompt rendering and the actual model call are
owned by the caller (see :mod:`evolve.options.branch`).  These state machines
only decide the deterministic sequence of actions and when to stop, matching
each role's method-defined responsibility:

- :class:`ExploreOption` (scout): broad search, may start in an empty cell.
- :class:`MechanistDevelopOption` (mechanist): focused improvement around an
  already verified start, optionally informed by verifier diagnostics.
- :class:`ChallengerAttackOption` (challenger): construct a counterexample,
  then attempt bounded minimal repairs.
"""

from __future__ import annotations

from typing import Any, Mapping

from evolve.types import OptionSpec, Role

from .base import (
    ExecutableOption,
    OptionContext,
    OptionEligibilityError,
    OptionStepInput,
    _ActionPlan,
    create_option_spec,
)


EXPLORE_STATE_MACHINE = "explore_v1"
MECHANIST_DEVELOP_STATE_MACHINE = "mechanist_develop_v1"
CHALLENGER_ATTACK_STATE_MACHINE = "challenger_attack_v1"
MATCHED_CONTINUATION_STATE_MACHINE = "matched_continuation_v1"
DIAGNOSTIC_REPAIR_STATE_MACHINE = "diagnostic_repair_v1"

PROPOSE_CAPABILITY = "propose"


def _require_started_from_verified(context: OptionContext) -> None:
    if not context.start_verified:
        raise OptionEligibilityError(
            "this option requires an already independently verified start state"
        )


def create_explore_option_spec(
    *,
    max_horizon: int,
    hard_cost: Mapping[str, Any],
    harness_eligibility: Any,
    expected_cost: Mapping[str, Any] = None,
) -> OptionSpec:
    return create_option_spec(
        version=EXPLORE_STATE_MACHINE,
        state_machine=EXPLORE_STATE_MACHINE,
        allowed_roles=(Role.SCOUT,),
        capabilities=(PROPOSE_CAPABILITY,),
        initiation={"allow_empty_cell": True, "allow_verified_start": True},
        step_policy={"structurally_distinct": True},
        stop_rule={"on_confirmed": True, "on_record_improvement": True},
        max_horizon=max_horizon,
        expected_cost=dict(expected_cost or hard_cost),
        hard_cost=dict(hard_cost),
        harness_eligibility=tuple(harness_eligibility),
        prerequisites=(),
        output_contract={"produces": "candidate_answer_payload"},
    )


class ExploreOption(ExecutableOption):
    """Scout: propose structurally different approaches, optionally from empty."""

    STATE_MACHINE = EXPLORE_STATE_MACHINE
    BEHAVIOR_VERSION = EXPLORE_STATE_MACHINE

    def _plan_action(
        self, state, step_input: OptionStepInput
    ) -> _ActionPlan:
        return _ActionPlan(
            action=f"explore_step_{state.step_index}",
            capability=PROPOSE_CAPABILITY,
            next_phase="explored",
            prompt_metadata={
                "mode": "broad_search",
                "cell_empty": state.context.cell_empty,
                "step_index": state.step_index,
            },
        )


def create_mechanist_develop_option_spec(
    *,
    max_horizon: int,
    hard_cost: Mapping[str, Any],
    harness_eligibility: Any,
    expected_cost: Mapping[str, Any] = None,
) -> OptionSpec:
    return create_option_spec(
        version=MECHANIST_DEVELOP_STATE_MACHINE,
        state_machine=MECHANIST_DEVELOP_STATE_MACHINE,
        allowed_roles=(Role.MECHANIST,),
        capabilities=(PROPOSE_CAPABILITY,),
        initiation={"allow_empty_cell": False, "allow_verified_start": True},
        step_policy={"use_diagnostics_when_available": True},
        stop_rule={"on_confirmed": True, "on_record_improvement": True},
        max_horizon=max_horizon,
        expected_cost=dict(expected_cost or hard_cost),
        hard_cost=dict(hard_cost),
        harness_eligibility=tuple(harness_eligibility),
        prerequisites=("verified_start",),
        output_contract={"produces": "candidate_answer_payload"},
    )


class MechanistDevelopOption(ExecutableOption):
    """Mechanist: develop explanations/invariants around a verified start."""

    STATE_MACHINE = MECHANIST_DEVELOP_STATE_MACHINE
    BEHAVIOR_VERSION = MECHANIST_DEVELOP_STATE_MACHINE

    def _check_initiation(self, context: OptionContext) -> None:
        _require_started_from_verified(context)

    def _plan_action(
        self, state, step_input: OptionStepInput
    ) -> _ActionPlan:
        use_diagnostics = bool(
            step_input.diagnostics_available
            and bool(self.spec.step_policy.get("use_diagnostics_when_available", True))
        )
        action_name = "refine_with_diagnostics" if use_diagnostics else "refine_mechanism"
        return _ActionPlan(
            action=f"{action_name}_step_{state.step_index}",
            capability=PROPOSE_CAPABILITY,
            next_phase="refined",
            prompt_metadata={
                "mode": "focused_improvement",
                "use_diagnostics": use_diagnostics,
                "step_index": state.step_index,
            },
        )


def create_challenger_attack_option_spec(
    *,
    max_horizon: int,
    hard_cost: Mapping[str, Any],
    harness_eligibility: Any,
    expected_cost: Mapping[str, Any] = None,
) -> OptionSpec:
    return create_option_spec(
        version=CHALLENGER_ATTACK_STATE_MACHINE,
        state_machine=CHALLENGER_ATTACK_STATE_MACHINE,
        allowed_roles=(Role.CHALLENGER,),
        capabilities=(PROPOSE_CAPABILITY,),
        initiation={"allow_empty_cell": False, "allow_verified_start": True},
        step_policy={"first_step": "counterexample", "then": "minimal_repair"},
        stop_rule={"on_confirmed": True, "on_record_improvement": True},
        max_horizon=max_horizon,
        expected_cost=dict(expected_cost or hard_cost),
        hard_cost=dict(hard_cost),
        harness_eligibility=tuple(harness_eligibility),
        prerequisites=("verified_start",),
        output_contract={"produces": "candidate_answer_payload"},
    )


class ChallengerAttackOption(ExecutableOption):
    """Challenger: attack assumptions, then attempt bounded minimal repairs."""

    STATE_MACHINE = CHALLENGER_ATTACK_STATE_MACHINE
    BEHAVIOR_VERSION = CHALLENGER_ATTACK_STATE_MACHINE

    def _check_initiation(self, context: OptionContext) -> None:
        _require_started_from_verified(context)

    def _plan_action(
        self, state, step_input: OptionStepInput
    ) -> _ActionPlan:
        if state.step_index == 0:
            return _ActionPlan(
                action="construct_counterexample",
                capability=PROPOSE_CAPABILITY,
                next_phase="attacked",
                prompt_metadata={"mode": "counterexample", "step_index": 0},
            )
        return _ActionPlan(
            action=f"minimal_repair_step_{state.step_index}",
            capability=PROPOSE_CAPABILITY,
            next_phase="repaired",
            prompt_metadata={
                "mode": "minimal_repair",
                "diagnostics_available": step_input.diagnostics_available,
                "step_index": state.step_index,
            },
        )


def create_matched_continuation_option_spec(
    *,
    max_horizon: int,
    hard_cost: Mapping[str, Any],
    harness_eligibility: Any,
    expected_cost: Mapping[str, Any] = None,
) -> OptionSpec:
    """Registered neutral continuation used as the matched audit control."""

    return create_option_spec(
        version=MATCHED_CONTINUATION_STATE_MACHINE,
        state_machine=MATCHED_CONTINUATION_STATE_MACHINE,
        allowed_roles=(Role.SCOUT, Role.MECHANIST, Role.CHALLENGER),
        capabilities=(PROPOSE_CAPABILITY,),
        initiation={"allow_empty_cell": False, "allow_verified_start": True},
        step_policy={"change_scope": "single_conservative_continuation"},
        stop_rule={"on_confirmed": True, "on_record_improvement": True},
        max_horizon=max_horizon,
        expected_cost=dict(expected_cost or hard_cost),
        hard_cost=dict(hard_cost),
        harness_eligibility=tuple(harness_eligibility),
        prerequisites=("verified_start",),
        output_contract={"produces": "candidate_answer_payload"},
    )


class MatchedContinuationOption(ExecutableOption):
    STATE_MACHINE = MATCHED_CONTINUATION_STATE_MACHINE
    BEHAVIOR_VERSION = MATCHED_CONTINUATION_STATE_MACHINE

    def _check_initiation(self, context: OptionContext) -> None:
        _require_started_from_verified(context)

    def _plan_action(self, state, step_input: OptionStepInput) -> _ActionPlan:
        return _ActionPlan(
            action=f"matched_continuation_step_{state.step_index}",
            capability=PROPOSE_CAPABILITY,
            next_phase="continued",
            prompt_metadata={
                "mode": "matched_continuation",
                "step_index": state.step_index,
            },
        )


def create_diagnostic_repair_option_spec(
    *,
    max_horizon: int,
    hard_cost: Mapping[str, Any],
    harness_eligibility: Any,
    expected_cost: Mapping[str, Any] = None,
) -> OptionSpec:
    return create_option_spec(
        version=DIAGNOSTIC_REPAIR_STATE_MACHINE,
        state_machine=DIAGNOSTIC_REPAIR_STATE_MACHINE,
        allowed_roles=(Role.CHALLENGER,),
        capabilities=(PROPOSE_CAPABILITY,),
        initiation={"allow_empty_cell": False, "allow_verified_start": True},
        step_policy={"one_diagnostic_target": True, "minimal_change": True},
        stop_rule={"max_attempts_per_branch": 1},
        max_horizon=min(max_horizon, 1),
        expected_cost=dict(expected_cost or hard_cost),
        hard_cost=dict(hard_cost),
        harness_eligibility=tuple(harness_eligibility),
        prerequisites=("verified_start",),
        output_contract={"produces": "single_minimal_repair"},
    )


class DiagnosticRepairOption(ExecutableOption):
    STATE_MACHINE = DIAGNOSTIC_REPAIR_STATE_MACHINE
    BEHAVIOR_VERSION = DIAGNOSTIC_REPAIR_STATE_MACHINE

    def _check_initiation(self, context: OptionContext) -> None:
        _require_started_from_verified(context)

    def _plan_action(self, state, step_input: OptionStepInput) -> _ActionPlan:
        return _ActionPlan(
            action="minimal_diagnostic_repair",
            capability=PROPOSE_CAPABILITY,
            next_phase="repaired",
            prompt_metadata={
                "mode": "minimal_repair",
                "diagnostics_available": True,
                "step_index": state.step_index,
            },
        )


__all__ = [
    "CHALLENGER_ATTACK_STATE_MACHINE",
    "DIAGNOSTIC_REPAIR_STATE_MACHINE",
    "EXPLORE_STATE_MACHINE",
    "MECHANIST_DEVELOP_STATE_MACHINE",
    "MATCHED_CONTINUATION_STATE_MACHINE",
    "PROPOSE_CAPABILITY",
    "ChallengerAttackOption",
    "DiagnosticRepairOption",
    "ExploreOption",
    "MechanistDevelopOption",
    "MatchedContinuationOption",
    "create_challenger_attack_option_spec",
    "create_diagnostic_repair_option_spec",
    "create_explore_option_spec",
    "create_mechanist_develop_option_spec",
    "create_matched_continuation_option_spec",
]
