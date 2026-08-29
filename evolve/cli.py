"""Command-line boundary for the EVOLVE engine.

Configuration validation and dry planning are deliberately pure: neither path
creates a run directory nor imports model/CUDA libraries.  The runtime engine is
loaded only after both operations have completed.
"""

from __future__ import annotations

import importlib
import json
import math
import sys
from typing import Any, Dict, Optional, Sequence

from .config import EvolveConfig, EvolveConfigError, load_evolve_config


def build_dry_plan(config: EvolveConfig, config_hash: str) -> Dict[str, Any]:
    """Return a deterministic, model-free plan summary.

    Phase 5 supplies posterior-ranked concrete arms.  This foundation plan
    exposes the immutable dimensions and minimum reservation counts without
    pretending to have posterior evidence that does not yet exist.
    """

    settings = config.evolve
    inflight = settings.workers.max_inflight_branches
    audit_slots = int(math.ceil(inflight * settings.budget.audit_fraction))
    refinement_slots = int(
        math.ceil(inflight * settings.budget.refinement_fraction)
    )
    harness_slots = int(math.ceil(inflight * settings.harnesses.trial_fraction))
    exploration_slots = int(
        math.ceil(inflight * settings.scheduler.global_exploration_fraction)
    )
    return {
        "schema_version": 1,
        "engine": "evolve",
        "config_hash": config_hash,
        "problem": config.problem,
        "method_complete": not config.method_incomplete,
        "epochs_total": settings.budget.epochs,
        "verifier_calls_total": settings.budget.verifier_calls,
        "frozen_arm_dimensions": [
            "cell_id",
            "role",
            "option_id",
            "harness_version",
            "horizon",
            "cost_class",
        ],
        "roles": list(settings.roles.enabled),
        "active_harness_versions": list(settings.harnesses.active_versions),
        "max_horizon": settings.options.max_horizon,
        "branch_budget": settings.options.branch_budget,
        "max_inflight_branches": inflight,
        "minimum_reservations_per_full_wave": {
            "audits": audit_slots,
            "refinement": refinement_slots,
            "harness_calibration": harness_slots,
            "global_exploration": exploration_slots,
            "every_role": len(settings.roles.enabled),
            "no_memory_audits": (
                1 if settings.audits.no_memory_fraction > 0.0 else 0
            ),
        },
        "posterior": settings.scheduler.posterior,
        "learning_objective": settings.learning.objective,
        "learning_group_k": settings.learning.group_k,
        "top_m": settings.learning.top_m,
        "model_loading": False,
        "writes_run_directory": False,
    }


def _print_json(value: Dict[str, Any]) -> None:
    print(
        json.dumps(
            value,
            ensure_ascii=False,
            allow_nan=False,
            sort_keys=True,
            indent=2,
        )
    )


def _runtime_engine_class():
    """Load the composed runtime without hiding its dependency failures."""

    module_name = f"{__package__}.engine"
    try:
        module = importlib.import_module(".engine", __package__)
    except ModuleNotFoundError as exc:
        if exc.name != module_name:
            raise
        return None
    return module.EvolveEngine


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = list(sys.argv[1:] if argv is None else argv)
    try:
        config, resolved, metadata = load_evolve_config(args)
    except EvolveConfigError as exc:
        print(f"EVOLVE configuration error: {exc}", file=sys.stderr)
        return 2

    if metadata["dry_plan"]:
        _print_json(build_dry_plan(config, metadata["config_hash"]))
        return 0
    if metadata["validate_config"]:
        _print_json(
            {
                "valid": True,
                "engine": "evolve",
                "schema_version": resolved["schema_version"],
                "config_hash": metadata["config_hash"],
                "problem": config.problem,
                "mode": metadata["mode"],
                "method_complete": not metadata["method_incomplete"],
            }
        )
        return 0

    # Imported only for a real run so validation remains CPU/model-free.  Only
    # the absent phase-9 module receives the in-progress message; an ImportError
    # raised *inside* that module is a real dependency/runtime failure and must
    # remain visible.
    EvolveEngine = _runtime_engine_class()
    if EvolveEngine is None:  # removed once the composed engine lands
        print(
            "EVOLVE runtime is not operational yet; use --validate-config or "
            "--dry-plan while implementation phases remain in progress.",
            file=sys.stderr,
        )
        return 3
    engine = EvolveEngine(config=config, resolved_config=resolved, metadata=metadata)
    return int(engine.run())


__all__ = ["build_dry_plan", "main"]
