"""Command-line boundary for the EVOLVE engine.

Configuration validation and dry planning are deliberately pure: neither path
creates a run directory nor imports model/CUDA libraries.  The runtime engine is
loaded only after both operations have completed.
"""

from __future__ import annotations

import importlib
import json
import logging
import math
import os
import sys
import textwrap
import warnings
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

    def slots(fraction: float) -> int:
        return int(math.ceil(inflight * fraction)) if fraction > 0.0 else 0

    def paired_slots(fraction: float) -> int:
        requested = slots(fraction)
        return requested if requested % 2 == 0 else requested + 1

    audit_slots = paired_slots(settings.budget.audit_fraction)
    refinement_slots = paired_slots(settings.budget.refinement_fraction)
    harness_slots = paired_slots(settings.harnesses.trial_fraction)
    empty_cell_slots = slots(settings.archive.empty_cell_fraction)
    exploration_slots = int(
        math.ceil(inflight * settings.scheduler.global_exploration_fraction)
    )
    no_memory_slots = min(
        audit_slots, slots(settings.audits.no_memory_fraction)
    )
    production_slots = inflight - audit_slots - refinement_slots - harness_slots
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
            "empty_or_under_tested_cells": empty_cell_slots,
            "global_exploration": exploration_slots,
            "every_role": len(settings.roles.enabled),
            "no_memory_audits": no_memory_slots,
        },
        "production_branch_slots": production_slots,
        "posterior": settings.scheduler.posterior,
        "learning_objective": settings.learning.objective,
        "learning_group_k": settings.learning.group_k,
        "top_m": settings.learning.top_m,
        "model": {
            "name": config.model_name,
            "training_backend": config.backend,
            "generation_backend": config.generation_backend,
            "training_load_in_4bit": config.load_in_4bit,
            "lora_rank": config.lora_rank,
        },
        "resources": {
            "gpu_type": config.gpu_type,
            "generation_gpu_ids": list(config.gpu_ids),
            "training_gpu_id": config.training_gpu_id,
            "runtime_visible_gpu_ids": list(config.runtime_gpu_ids),
            "vllm_tensor_parallel_size": config.vllm_tensor_parallel_size,
            "vllm_quantization": config.vllm_quantization,
            "vllm_gpu_memory_utilization": config.vllm_gpu_memory_utilization,
            "exclusive_evaluation_gpu_id": config.kernel_gpu_id,
            "kernel_eval_isolation": config.kernel_eval_isolation,
        },
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


def format_startup_banner(
    config: EvolveConfig, metadata: Dict[str, Any], *, width: int = 100
) -> str:
    """Render the resolved real-run hyperparameters before model loading."""

    width = max(72, width)
    inner = width - 2
    label_width = 16
    value_width = inner - label_width - 5
    lines = ["╭" + "─" * inner + "╮"]
    lines.append("│" + " EVOLVE · VERIFIED SCIENTIFIC SEARCH ".center(inner) + "│")

    def section(title: str) -> None:
        marker = f" {title} "
        left = 2
        right = inner - left - len(marker)
        lines.append("├" + "─" * left + marker + "─" * right + "┤")

    def row(label: str, value: Any) -> None:
        rendered = str(value)
        wrapped = textwrap.wrap(
            rendered,
            width=value_width,
            break_long_words=False,
            break_on_hyphens=False,
        ) or [""]
        for index, part in enumerate(wrapped):
            shown_label = label if index == 0 else ""
            body = f"  {shown_label:<{label_width}} {part:<{value_width}}  "
            lines.append("│" + body + "│")

    mode = str(metadata.get("mode", "fresh"))
    if mode == "resume":
        mode = f"resume · {metadata.get('resume_dir', '?')}"
    problem = config.problem
    if config.problem_type:
        problem += f" / {config.problem_type}"
    config_hash = str(metadata.get("config_hash", "?"))
    if len(config_hash) > 16:
        config_hash = config_hash[:16] + "…"

    section("RUN")
    row("mode", mode)
    row("problem", problem)
    row("config", f"schema {config.schema_version} · {config_hash}")

    section("MODEL & SAMPLING")
    row("backbone", config.model_name)
    row(
        "backends",
        f"training={config.backend} · generation={config.generation_backend} · "
        f"HF 4-bit={config.load_in_4bit}",
    )
    row(
        "tokens",
        f"context={config.max_seq_length:,} · max_new={config.max_new_tokens:,} · "
        f"micro_batch={config.gen_micro_batch}",
    )
    row(
        "sampling",
        f"temperature={config.temperature:g} · top_p={config.top_p:g} · "
        f"thinking={config.thinking}",
    )

    section("ROLE ADAPTER LEARNING")
    row("roles", ", ".join(config.evolve.roles.enabled))
    row(
        "LoRA",
        f"rank={config.lora_rank} · alpha={config.lora_alpha} · "
        f"dropout={config.lora_dropout:g} · target_modules={len(config.target_modules)}",
    )
    row(
        "optimizer",
        f"lr={config.learning_rate:g} · KL={config.kl_penalty_coef:g}",
    )
    row(
        "objective",
        f"{config.evolve.learning.objective} · top_m={config.evolve.learning.top_m} · "
        f"group_k={config.evolve.learning.group_k}",
    )

    training = (
        f"GPU {config.training_gpu_id}"
        if config.training_gpu_id is not None
        else f"shared generation pool {list(config.gpu_ids)}"
    )
    if config.kernel_gpu_id is None:
        evaluation = "CPU verifier · no GPU reserved"
    elif config.kernel_gpu_id in config.runtime_gpu_ids:
        evaluation = f"GPU {config.kernel_gpu_id} · serialized model teardown"
    else:
        evaluation = f"GPU {config.kernel_gpu_id} · exclusive"
    gpu_type = (
        "auto · verifier is hardware-independent"
        if config.gpu_type.strip().lower() == "auto"
        else config.gpu_type
    )

    section("RESOURCES")
    row("GPU type", gpu_type)
    row("training", training)
    row(
        "vLLM",
        f"GPUs={list(config.gpu_ids)} · TP={config.vllm_tensor_parallel_size} · "
        f"memory={config.vllm_gpu_memory_utilization:.0%} · "
        f"quantization={config.vllm_quantization}",
    )
    row("evaluation", evaluation)

    section("SEARCH BUDGET")
    row(
        "totals",
        f"epochs={config.evolve.budget.epochs} · "
        f"verifier_calls={config.evolve.budget.verifier_calls:,} · "
        f"seed_states={config.num_seed_states}",
    )
    row(
        "branches",
        f"inflight={config.evolve.workers.max_inflight_branches} · "
        f"horizon≤{config.evolve.options.max_horizon} · "
        f"branch_budget={config.evolve.options.branch_budget}",
    )
    row(
        "reserves",
        f"audit={config.evolve.budget.audit_fraction:.0%} · "
        f"refinement={config.evolve.budget.refinement_fraction:.0%} · "
        f"harness={config.evolve.harnesses.trial_fraction:.0%} · "
        f"exploration={config.evolve.scheduler.global_exploration_fraction:.0%}",
    )
    row("reproducibility", f"seed={config.seed} · deterministic={config.deterministic}")
    lines.append("╰" + "─" * inner + "╯")
    return "\n".join(lines)


def _configure_runtime_noise_filters() -> None:
    """Set safe dependency defaults and hide known non-fatal chatter."""

    os.environ.setdefault("HF_HUB_VERBOSITY", "error")
    os.environ.setdefault("TRANSFORMERS_VERBOSITY", "error")
    # Recent vLLM releases enable FlashInfer sampling by default.  FlashInfer
    # may JIT-compile that sampler during engine warmup and therefore require a
    # full CUDA toolkit (nvcc), which GPU compute nodes do not necessarily
    # expose.  vLLM's native sampler has no such startup requirement.  Keep an
    # explicit environment setting authoritative for users who have nvcc and
    # intentionally want FlashInfer.
    os.environ.setdefault("VLLM_USE_FLASHINFER_SAMPLER", "0")
    logging.getLogger("torch.utils._pytree").setLevel(logging.ERROR)
    warnings.filterwarnings(
        "ignore",
        message=r"Adapter default was active which is now deleted\..*",
        category=UserWarning,
        module=r"peft\.tuners\.tuners_utils",
    )
    warnings.filterwarnings(
        "ignore",
        message=r"You are sending unauthenticated requests to the HF Hub\..*",
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

    print(format_startup_banner(config, metadata), flush=True)
    _configure_runtime_noise_filters()

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


__all__ = ["build_dry_plan", "format_startup_banner", "main"]
