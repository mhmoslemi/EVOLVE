"""Strict, resume-safe configuration for the EVOLVE engine.

The EVOLVE method lives in a closed, schema-versioned ``evolve`` mapping. Resolution
order is:

    dataclass defaults < problem YAML < authoritative resumed config < CLI

An EVOLVE resume is different from a fresh resolution: it must have a complete
``config.resolved.json`` and never fills newly introduced values from today's
defaults or from a possibly changed problem YAML.
"""

from __future__ import annotations

import argparse
import copy
import json
import math
from dataclasses import dataclass, field, fields
from pathlib import Path
from types import MappingProxyType
from typing import Any, Dict, Iterator, List, Mapping, Optional, Sequence, Tuple, Type

from evolve.ids import content_hash
from evolve.runio.schema import (
    RunSchemaError,
    UnsupportedRunSchemaError,
    resolve_effective_run_metadata,
)


CONFIG_SCHEMA_VERSION = 1
ENGINE_NAME = "evolve"
ROLE_NAMES = ("scout", "mechanist", "challenger")


class EvolveConfigError(ValueError):
    """The requested EVOLVE configuration is unsafe or unsupported."""


class UnsupportedEvolveConfigError(EvolveConfigError):
    """The configuration declares a future/unsupported method or schema."""


class _FrozenConfigMapping(Mapping[str, Any]):
    """Recursively immutable JSON mapping for problem compatibility data."""

    __slots__ = ("_data",)

    def __init__(self, value: Mapping[str, Any]) -> None:
        if not isinstance(value, Mapping):
            raise EvolveConfigError("problem_config must be a mapping")
        data: Dict[str, Any] = {}
        for key, item in value.items():
            if not isinstance(key, str):
                raise EvolveConfigError("problem_config mapping keys must be strings")
            data[key] = _freeze_config_value(item)
        object.__setattr__(self, "_data", MappingProxyType(data))

    def __getitem__(self, key: str) -> Any:
        return self._data[key]

    def __iter__(self) -> Iterator[str]:
        return iter(self._data)

    def __len__(self) -> int:
        return len(self._data)

    def __repr__(self) -> str:
        return f"_FrozenConfigMapping({dict(self._data)!r})"

    def __eq__(self, other: Any) -> bool:
        if isinstance(other, Mapping):
            return dict(self.items()) == dict(other.items())
        return False

    def __copy__(self) -> "_FrozenConfigMapping":
        return self

    def __deepcopy__(self, memo: Dict[int, Any]) -> "_FrozenConfigMapping":
        memo[id(self)] = self
        return self

    def __reduce__(self):
        return (_FrozenConfigMapping, (dict(self.items()),))

    __hash__ = None


def _freeze_config_value(value: Any) -> Any:
    if isinstance(value, _FrozenConfigMapping):
        return value
    if isinstance(value, Mapping):
        return _FrozenConfigMapping(value)
    if isinstance(value, (list, tuple)):
        return tuple(_freeze_config_value(item) for item in value)
    if value is None or isinstance(value, (str, int, float, bool)):
        return value
    raise EvolveConfigError(
        "problem_config contains a non-JSON value of type "
        f"{type(value).__name__}"
    )


def _is_bool(value: Any) -> bool:
    return isinstance(value, bool)


def _int(value: Any, name: str, *, minimum: Optional[int] = None) -> int:
    if _is_bool(value) or not isinstance(value, int):
        raise EvolveConfigError(f"{name} must be an integer")
    if minimum is not None and value < minimum:
        raise EvolveConfigError(f"{name} must be >= {minimum}")
    return int(value)


def _number(value: Any, name: str) -> float:
    if _is_bool(value) or not isinstance(value, (int, float)):
        raise EvolveConfigError(f"{name} must be a finite number")
    out = float(value)
    if not math.isfinite(out):
        raise EvolveConfigError(f"{name} must be a finite number")
    return out


def _fraction(value: Any, name: str, *, allow_one: bool = True) -> float:
    out = _number(value, name)
    upper_ok = out <= 1.0 if allow_one else out < 1.0
    if out < 0.0 or not upper_ok:
        bound = "[0, 1]" if allow_one else "[0, 1)"
        raise EvolveConfigError(f"{name} must be in {bound}")
    return out


def _positive_number(value: Any, name: str) -> float:
    out = _number(value, name)
    if out <= 0.0:
        raise EvolveConfigError(f"{name} must be > 0")
    return out


def _bool(value: Any, name: str) -> bool:
    if not _is_bool(value):
        raise EvolveConfigError(f"{name} must be a boolean")
    return value


def _string(value: Any, name: str, *, nonempty: bool = True) -> str:
    if not isinstance(value, str):
        raise EvolveConfigError(f"{name} must be a string")
    out = value.strip() if nonempty else value
    if nonempty and not out:
        raise EvolveConfigError(f"{name} must be a non-empty string")
    return out


def _string_tuple(value: Any, name: str, *, nonempty: bool = True) -> Tuple[str, ...]:
    if isinstance(value, str) or not isinstance(value, (list, tuple)):
        raise EvolveConfigError(f"{name} must be a JSON list of strings")
    out = tuple(_string(item, f"{name}[]") for item in value)
    if nonempty and not out:
        raise EvolveConfigError(f"{name} must not be empty")
    if len(set(out)) != len(out):
        raise EvolveConfigError(f"{name} must not contain duplicates")
    return out


def parse_gpu_ids(value: Any) -> Tuple[int, ...]:
    """Normalize legacy comma strings and JSON arrays to physical GPU IDs."""
    if value is None:
        return ()
    if isinstance(value, str):
        pieces = [piece.strip() for piece in value.split(",") if piece.strip()]
        try:
            values: Sequence[Any] = [int(piece) for piece in pieces]
        except ValueError as exc:
            raise EvolveConfigError(
                f"gpu_ids must be comma-separated integers or a JSON list: {value!r}"
            ) from exc
    elif isinstance(value, (list, tuple)):
        values = value
    else:
        raise EvolveConfigError("gpu_ids must be a JSON list or comma-separated string")

    ids: List[int] = []
    for item in values:
        ids.append(_int(item, "gpu_ids[]", minimum=0))
    if len(set(ids)) != len(ids):
        raise EvolveConfigError("gpu_ids must not contain duplicates")
    return tuple(ids)


@dataclass(frozen=True)
class BudgetConfig:
    epochs: int = 100
    verifier_calls: int = 50_000
    audit_fraction: float = 0.15
    refinement_fraction: float = 0.05

    def __post_init__(self) -> None:
        _int(self.epochs, "evolve.budget.epochs", minimum=1)
        _int(self.verifier_calls, "evolve.budget.verifier_calls", minimum=1)
        if self.verifier_calls < self.epochs:
            raise EvolveConfigError(
                "evolve.budget.verifier_calls must be >= evolve.budget.epochs"
            )
        _fraction(self.audit_fraction, "evolve.budget.audit_fraction", allow_one=False)
        _fraction(
            self.refinement_fraction,
            "evolve.budget.refinement_fraction",
            allow_one=False,
        )
        if self.audit_fraction + self.refinement_fraction > 1.0 + 1e-12:
            raise EvolveConfigError(
                "audit_fraction + refinement_fraction must not exceed 1"
            )


@dataclass(frozen=True)
class ArchiveConfig:
    elites_per_cell: int = 3
    empty_cell_fraction: float = 0.10

    def __post_init__(self) -> None:
        _int(self.elites_per_cell, "evolve.archive.elites_per_cell", minimum=3)
        _fraction(
            self.empty_cell_fraction,
            "evolve.archive.empty_cell_fraction",
            allow_one=False,
        )


@dataclass(frozen=True)
class RolesConfig:
    enabled: Tuple[str, ...] = ROLE_NAMES
    test_mode: bool = False
    method_incomplete: bool = False

    def __post_init__(self) -> None:
        enabled = _string_tuple(self.enabled, "evolve.roles.enabled")
        object.__setattr__(self, "enabled", enabled)
        _bool(self.test_mode, "evolve.roles.test_mode")
        _bool(self.method_incomplete, "evolve.roles.method_incomplete")
        unknown = set(enabled) - set(ROLE_NAMES)
        if unknown:
            raise EvolveConfigError(
                f"unknown EVOLVE role(s): {sorted(unknown)}; expected {list(ROLE_NAMES)}"
            )
        if enabled != ROLE_NAMES:
            if not self.test_mode or not self.method_incomplete:
                raise EvolveConfigError(
                    "role subsets are test-only and require both "
                    "evolve.roles.test_mode=true and "
                    "evolve.roles.method_incomplete=true"
                )
        elif self.test_mode != self.method_incomplete:
            raise EvolveConfigError(
                "test_mode and method_incomplete must be enabled together"
            )


@dataclass(frozen=True)
class OptionsConfig:
    max_horizon: int = 4
    branch_budget: int = 64

    def __post_init__(self) -> None:
        _int(self.max_horizon, "evolve.options.max_horizon", minimum=1)
        _int(self.branch_budget, "evolve.options.branch_budget", minimum=1)
        if self.max_horizon > self.branch_budget:
            raise EvolveConfigError(
                "evolve.options.max_horizon cannot exceed branch_budget"
            )


@dataclass(frozen=True)
class HarnessesConfig:
    trial_fraction: float = 0.05
    active_versions: Tuple[str, ...] = ("baseline_v1",)

    def __post_init__(self) -> None:
        _fraction(
            self.trial_fraction,
            "evolve.harnesses.trial_fraction",
            allow_one=False,
        )
        active = _string_tuple(
            self.active_versions, "evolve.harnesses.active_versions"
        )
        object.__setattr__(self, "active_versions", active)


@dataclass(frozen=True)
class SchedulerConfig:
    posterior: str = "zero_inflated_tail"
    global_exploration_fraction: float = 0.10

    def __post_init__(self) -> None:
        posterior = _string(self.posterior, "evolve.scheduler.posterior")
        object.__setattr__(self, "posterior", posterior)
        if posterior != "zero_inflated_tail":
            raise UnsupportedEvolveConfigError(
                "schema-v1 supports only scheduler.posterior='zero_inflated_tail'"
            )
        _fraction(
            self.global_exploration_fraction,
            "evolve.scheduler.global_exploration_fraction",
            allow_one=False,
        )


@dataclass(frozen=True)
class AuditsConfig:
    no_memory_fraction: float = 0.05
    min_pairs_for_promotion: int = 5

    def __post_init__(self) -> None:
        _fraction(
            self.no_memory_fraction,
            "evolve.audits.no_memory_fraction",
            allow_one=False,
        )
        _int(
            self.min_pairs_for_promotion,
            "evolve.audits.min_pairs_for_promotion",
            minimum=1,
        )


@dataclass(frozen=True)
class LearningConfig:
    objective: str = "ordergrad"
    top_m: int = 1
    group_k: int = 8

    def __post_init__(self) -> None:
        objective = _string(self.objective, "evolve.learning.objective").lower()
        object.__setattr__(self, "objective", objective)
        if objective not in ("ordergrad", "maxpo"):
            raise UnsupportedEvolveConfigError(
                "evolve.learning.objective must be 'ordergrad' or 'maxpo'"
            )
        _int(self.group_k, "evolve.learning.group_k", minimum=2)
        _int(self.top_m, "evolve.learning.top_m", minimum=1)
        if self.top_m > self.group_k:
            raise EvolveConfigError("evolve.learning.top_m cannot exceed group_k")
        if objective == "maxpo" and self.top_m != 1:
            raise EvolveConfigError("MaxPO is pure-max and requires learning.top_m=1")


@dataclass(frozen=True)
class RefinementConfig:
    max_attempts: int = 3
    max_depth: int = 2

    def __post_init__(self) -> None:
        _int(self.max_attempts, "evolve.refinement.max_attempts", minimum=1)
        _int(self.max_depth, "evolve.refinement.max_depth", minimum=1)
        if self.max_attempts > 3:
            raise EvolveConfigError("evolve.refinement.max_attempts must be <= 3")
        if self.max_depth > 2:
            raise EvolveConfigError("evolve.refinement.max_depth must be <= 2")


@dataclass(frozen=True)
class ReportingConfig:
    status_every_verifications: int = 25
    plots_every_epochs: int = 1

    def __post_init__(self) -> None:
        _int(
            self.status_every_verifications,
            "evolve.reporting.status_every_verifications",
            minimum=1,
        )
        _int(
            self.plots_every_epochs,
            "evolve.reporting.plots_every_epochs",
            minimum=1,
        )


@dataclass(frozen=True)
class WorkersConfig:
    max_inflight_branches: int = 16

    def __post_init__(self) -> None:
        _int(
            self.max_inflight_branches,
            "evolve.workers.max_inflight_branches",
            minimum=1,
        )


@dataclass(frozen=True)
class EvolveSettings:
    budget: BudgetConfig = field(default_factory=BudgetConfig)
    archive: ArchiveConfig = field(default_factory=ArchiveConfig)
    roles: RolesConfig = field(default_factory=RolesConfig)
    options: OptionsConfig = field(default_factory=OptionsConfig)
    harnesses: HarnessesConfig = field(default_factory=HarnessesConfig)
    scheduler: SchedulerConfig = field(default_factory=SchedulerConfig)
    audits: AuditsConfig = field(default_factory=AuditsConfig)
    learning: LearningConfig = field(default_factory=LearningConfig)
    refinement: RefinementConfig = field(default_factory=RefinementConfig)
    reporting: ReportingConfig = field(default_factory=ReportingConfig)
    workers: WorkersConfig = field(default_factory=WorkersConfig)


@dataclass(frozen=True)
class EvolveConfig:
    schema_version: int = CONFIG_SCHEMA_VERSION
    engine: str = ENGINE_NAME
    problem: str = "circle_packing"
    problem_type: str = ""
    target: Optional[float] = None
    num_seed_states: int = 8

    model_name: str = "Qwen/Qwen3-8B"
    backend: str = "auto"
    generation_backend: str = "hf"
    max_seq_length: int = 32_000
    max_new_tokens: int = 4_200
    temperature: float = 1.0
    top_p: float = 1.0
    thinking: bool = False
    load_in_4bit: bool = False
    lora_rank: int = 32
    lora_alpha: int = 32
    lora_dropout: float = 0.0
    target_modules: Tuple[str, ...] = (
        "q_proj",
        "k_proj",
        "v_proj",
        "o_proj",
        "gate_proj",
        "up_proj",
        "down_proj",
    )
    learning_rate: float = 4e-5
    kl_penalty_coef: float = 0.1

    sandbox_timeout_s: float = 30.0
    kernel_timeout_s: Optional[float] = None
    reward_workers: int = 0
    gpu_ids: Tuple[int, ...] = (0,)
    num_gpus: int = 1
    kernel_gpu_id: Optional[int] = None
    gen_micro_batch: int = 0
    vllm_gpu_memory_utilization: float = 0.9
    vllm_enforce_eager: bool = False
    vllm_enable_prefix_caching: bool = True
    seed: int = 42
    deterministic: bool = False
    checkpoint_path: Optional[str] = None

    evolve: EvolveSettings = field(default_factory=EvolveSettings)
    problem_config: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        object.__setattr__(
            self,
            "problem_config",
            _FrozenConfigMapping(self.problem_config),
        )

    @property
    def method_incomplete(self) -> bool:
        return self.evolve.roles.method_incomplete

    @property
    def problem_runtime_config(self) -> Mapping[str, Any]:
        """Complete problem-facing projection of the resolved configuration.

        Problem constructors own scientific and resource behavior, so they must
        receive common fields as well as explicitly registered problem fields.
        Keeping this projection in one place prevents the controller from
        silently dropping subtype, seed, timeout, target, or GPU identity.
        """

        values: Dict[str, Any] = dict(self.problem_config)
        values.update(
            {
                "problem": self.problem,
                "problem_type": self.problem_type,
                "target": self.target,
                "num_seed_states": self.num_seed_states,
                "seed": self.seed,
                "sandbox_timeout_s": self.sandbox_timeout_s,
                "kernel_timeout_s": self.kernel_timeout_s,
                "reward_workers": self.reward_workers,
                "gpu_ids": list(self.gpu_ids),
                "num_gpus": self.num_gpus,
                "kernel_gpu_id": self.kernel_gpu_id,
            }
        )
        return _FrozenConfigMapping(values)

    @property
    def num_steps(self) -> int:
        """Compatibility spelling: total target epochs, never additional epochs."""
        return self.evolve.budget.epochs

    def to_dict(self, *, compatibility: bool = True) -> Dict[str, Any]:
        out: Dict[str, Any] = {
            "schema_version": self.schema_version,
            "engine": self.engine,
            "problem": self.problem,
            "problem_type": self.problem_type,
            "target": self.target,
            "num_seed_states": self.num_seed_states,
            "model_name": self.model_name,
            "backend": self.backend,
            "generation_backend": self.generation_backend,
            "max_seq_length": self.max_seq_length,
            "max_new_tokens": self.max_new_tokens,
            "temperature": self.temperature,
            "top_p": self.top_p,
            "thinking": self.thinking,
            "load_in_4bit": self.load_in_4bit,
            "lora_rank": self.lora_rank,
            "lora_alpha": self.lora_alpha,
            "lora_dropout": self.lora_dropout,
            "target_modules": list(self.target_modules),
            "learning_rate": self.learning_rate,
            "kl_penalty_coef": self.kl_penalty_coef,
            "sandbox_timeout_s": self.sandbox_timeout_s,
            "kernel_timeout_s": self.kernel_timeout_s,
            "reward_workers": self.reward_workers,
            "gpu_ids": list(self.gpu_ids),
            "num_gpus": self.num_gpus,
            "kernel_gpu_id": self.kernel_gpu_id,
            "gen_micro_batch": self.gen_micro_batch,
            "vllm_gpu_memory_utilization": self.vllm_gpu_memory_utilization,
            "vllm_enforce_eager": self.vllm_enforce_eager,
            "vllm_enable_prefix_caching": self.vllm_enable_prefix_caching,
            "seed": self.seed,
            "deterministic": self.deterministic,
            "checkpoint_path": self.checkpoint_path,
            "evolve": _settings_dict(self.evolve),
        }
        for key, value in self.problem_config.items():
            if key not in out:
                out[key] = _json_safe(value)
        if compatibility:
            out["num_steps"] = self.evolve.budget.epochs
            out["group_size"] = self.evolve.learning.group_k
        return out


_SECTION_CLASSES: Dict[str, Type[Any]] = {
    "budget": BudgetConfig,
    "archive": ArchiveConfig,
    "roles": RolesConfig,
    "options": OptionsConfig,
    "harnesses": HarnessesConfig,
    "scheduler": SchedulerConfig,
    "audits": AuditsConfig,
    "learning": LearningConfig,
    "refinement": RefinementConfig,
    "reporting": ReportingConfig,
    "workers": WorkersConfig,
}


def _settings_dict(settings: EvolveSettings) -> Dict[str, Any]:
    out: Dict[str, Any] = {}
    for name in _SECTION_CLASSES:
        section = getattr(settings, name)
        values: Dict[str, Any] = {}
        for item in fields(section):
            values[item.name] = _json_safe(getattr(section, item.name))
        out[name] = values
    return out


def _json_safe(value: Any) -> Any:
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, Mapping):
        return {str(key): _json_safe(item) for key, item in value.items()}
    if isinstance(value, (tuple, list)):
        return [_json_safe(item) for item in value]
    if value is None or isinstance(value, (str, int, float, bool)):
        return value
    raise EvolveConfigError(
        f"configuration contains a non-JSON value of type {type(value).__name__}"
    )


def canonical_config_hash(document: Any) -> str:
    """SHA-256 over canonical JSON, excluding the self-referential hash field."""
    if isinstance(document, EvolveConfig):
        value = document.to_dict()
    elif isinstance(document, Mapping):
        value = copy.deepcopy(dict(document))
    else:
        raise EvolveConfigError("canonical_config_hash expects a config mapping")
    value.pop("config_hash", None)
    try:
        return content_hash(_json_safe(value))
    except (TypeError, ValueError) as exc:
        raise EvolveConfigError(f"configuration is not canonical JSON: {exc}") from exc


def _coerce_like_default(value: Any, default: Any, name: str) -> Any:
    if _is_bool(default):
        return _bool(value, name)
    if isinstance(default, int) and not _is_bool(default):
        return _int(value, name)
    if isinstance(default, float):
        return _number(value, name)
    if isinstance(default, str):
        return _string(value, name)
    if isinstance(default, tuple):
        return _string_tuple(value, name)
    return value


def _build_section(
    section_name: str,
    raw: Any,
    section_type: Type[Any],
    *,
    require_complete: bool,
) -> Any:
    if not isinstance(raw, Mapping):
        raise EvolveConfigError(f"evolve.{section_name} must be a mapping")
    allowed = {item.name for item in fields(section_type)}
    unknown = set(raw) - allowed
    if unknown:
        raise EvolveConfigError(
            f"unknown key(s) in evolve.{section_name}: {sorted(unknown)}"
        )
    if require_complete:
        missing = allowed - set(raw)
        if missing:
            raise EvolveConfigError(
                f"resumed config is missing evolve.{section_name} key(s): "
                f"{sorted(missing)}"
            )

    default_instance = section_type()
    values: Dict[str, Any] = {}
    for item in fields(section_type):
        default = getattr(default_instance, item.name)
        value = raw[item.name] if item.name in raw else default
        values[item.name] = _coerce_like_default(
            value, default, f"evolve.{section_name}.{item.name}"
        )
    return section_type(**values)


def _build_settings(raw: Any, *, require_complete: bool) -> EvolveSettings:
    if not isinstance(raw, Mapping):
        raise EvolveConfigError("evolve must be a mapping")
    unknown = set(raw) - set(_SECTION_CLASSES)
    if unknown:
        raise EvolveConfigError(f"unknown section(s) in evolve: {sorted(unknown)}")
    if require_complete:
        missing = set(_SECTION_CLASSES) - set(raw)
        if missing:
            raise EvolveConfigError(
                f"resumed config is missing evolve section(s): {sorted(missing)}"
            )
    defaults = EvolveSettings()
    values: Dict[str, Any] = {}
    for name, section_type in _SECTION_CLASSES.items():
        section_raw = raw.get(name, _settings_dict(defaults)[name])
        values[name] = _build_section(
            name, section_raw, section_type, require_complete=require_complete
        )
    return EvolveSettings(**values)


_COMMON_KEYS = {
    "schema_version",
    "engine",
    "problem",
    "problem_type",
    "target",
    "num_seed_states",
    "model_name",
    "backend",
    "generation_backend",
    "max_seq_length",
    "max_new_tokens",
    "temperature",
    "top_p",
    "thinking",
    "load_in_4bit",
    "lora_rank",
    "lora_alpha",
    "lora_dropout",
    "target_modules",
    "learning_rate",
    "kl_penalty_coef",
    "sandbox_timeout_s",
    "kernel_timeout_s",
    "reward_workers",
    "gpu_ids",
    "num_gpus",
    "kernel_gpu_id",
    "gen_micro_batch",
    "vllm_gpu_memory_utilization",
    "vllm_enforce_eager",
    "vllm_enable_prefix_caching",
    "seed",
    "deterministic",
    "checkpoint_path",
    "evolve",
    "config_hash",
    # Compatibility aliases, validated against the nested source of truth.
    "num_steps",
    "group_size",
}


_ALLOWED_PROBLEM_KEYS = {
    "num_circles",
    "budget_s",
    "eval_cpus",
    "eval_memory_mb",
    "eval_seed",
    "scientific_max_points",
    "scientific_max_coefficients",
    "single_cell",
    "fail_score",
    "degenerate_threshold",
    "score_scale",
    "gpu_type",
    "triton_version",
    "task_yaml",
    "lib_dir",
    "kernel_lib_dir",
    "kernel_log_chars",
    "mla_seed_runtime_us",
    "seed_from_reference",
    "show_launch_note",
}


# Explicit compatibility inventories. These are the fields implemented by the
# active legacy modules and present in the repository's problem YAMLs. Prefix
# matching is intentionally forbidden: a misspelling must fail before model
# loading instead of becoming inert persisted configuration.
_LEGACY_MEMORY_KEYS = {
    "memory",
    "memory_lookup_mode",
    "memory_lookup_max_select",
    "memory_lookup_max_new_tokens",
    "memory_lookup_temperature",
    "memory_lookup_fallback",
    "memory_catalog_max_lessons",
    "memory_catalog_chars",
    "memory_inject_mode",
    "memory_token_budget",
    "memory_grant_context",
    "memory_arm_control_fraction",
    "memory_arm_explore_fraction",
    "memory_arm_max_lessons",
    "memory_arm_exploration_c",
    "memory_outcome_credit",
    "memory_text_reinforce",
    "memory_extract_mode",
    "memory_extract_from",
    "memory_lessons_per_call",
    "memory_require_full_lessons",
    "memory_max_examples_per_call",
    "memory_max_chars_per_example",
    "memory_feedback_chars",
    "memory_reinforce_delta",
    "memory_max_new_tokens",
    "memory_temperature",
    "memory_top_p",
    "memory_use_gen_pool",
    "memory_forbid_constructions",
    "memory_hygiene_profile",
    "memory_max_code_lines",
    "memory_global_scope_allows_code",
    "memory_curate_every",
    "memory_curate_min_bank",
    "memory_curate_max_items",
    "memory_curate_max_new_tokens",
    "memory_curate_min_keep_frac",
    "memory_max_lessons",
    "memory_dedup_jaccard",
    "memory_persist",
}

_LEGACY_FEEDBACK_KEYS = {
    "feedback",
    "feedback_lambda",
    "feedback_anneal_steps",
    "feedback_anneal_shape",
    "feedback_lambda_final",
    "feedback_clip",
    "feedback_chars",
    "feedback_max_per_step",
    "feedback_auto_fraction",
    "feedback_include_constant_groups",
    "feedback_inject_mode",
    "feedback_normalize",
    "feedback_adaptive",
    "feedback_validity_floor",
    "feedback_validity_target",
    "feedback_max_reward_ratio",
    "feedback_reward_scale_floor",
    "feedback_max_per_signature",
    "feedback_auto_signature_fraction",
}

_LEGACY_RERANKER_KEYS = {
    "reranker_enabled",
    "reranker_backend",
    "reranker_model",
    "reranker_base_url",
    "reranker_api_key",
    "reranker_api_key_env",
    "reranker_temperature",
    "reranker_max_tokens",
    "reranker_request_timeout_s",
    "reranker_judge_gpu",
    "reranker_judge_batch_size",
    "reranker_judge_load_in_4bit",
    "reranker_top_p",
    "reranker_max_seq_length",
    "reranker_top_k",
    "reranker_debate",
    "reranker_tournament_mode",
    "reranker_num_random_matches",
    "reranker_both_orders",
    "reranker_judge_concurrency",
    "reranker_max_code_chars",
    "reranker_elo_init",
    "reranker_elo_k",
    "reranker_elo_softmax_temp",
    "reranker_prior_weight",
    "reranker_poll_interval_s",
    "reranker_min_states_to_rank",
    "reranker_goal",
}

def _allowed_problem_key(key: str) -> bool:
    return key in _ALLOWED_PROBLEM_KEYS


def _path_from_config(value: Any, name: str, cwd: Path, *, directory: bool) -> str:
    text = _string(value, name)
    path = Path(text).expanduser()
    if not path.is_absolute():
        path = cwd / path
    path = path.resolve()
    ok = path.is_dir() if directory else path.is_file()
    if not ok:
        kind = "directory" if directory else "file"
        raise EvolveConfigError(f"{name} {kind} does not exist: {path}")
    return str(path)


def _optional_path(value: Any, name: str, cwd: Path) -> Optional[str]:
    if value is None:
        return None
    text = _string(value, name)
    path = Path(text).expanduser()
    if not path.is_absolute():
        path = cwd / path
    path = path.resolve()
    if not path.exists():
        raise EvolveConfigError(f"{name} does not exist: {path}")
    return str(path)


def _validate_cross_section(config: EvolveConfig) -> None:
    e = config.evolve
    reservations = (
        e.budget.audit_fraction
        + e.budget.refinement_fraction
        + e.archive.empty_cell_fraction
        + e.harnesses.trial_fraction
        + e.scheduler.global_exploration_fraction
    )
    if reservations > 1.0 + 1e-12:
        raise EvolveConfigError(
            "reserved audit/refinement/empty-cell/harness/exploration fractions "
            f"sum to {reservations:.6f}, exceeding 1"
        )
    if e.audits.no_memory_fraction > e.budget.audit_fraction + 1e-12:
        raise EvolveConfigError(
            "evolve.audits.no_memory_fraction cannot exceed budget.audit_fraction"
        )

    if config.kernel_gpu_id is not None and config.kernel_gpu_id in config.gpu_ids:
        raise EvolveConfigError(
            "kernel_gpu_id is exclusive and must not appear in authoritative gpu_ids"
        )
    if config.generation_backend == "vllm" and not config.gpu_ids:
        raise EvolveConfigError("vLLM generation requires at least one gpu_id")

    gpu_problem = config.problem.lower() in {
        "gpu_mode",
        "kernel",
        "kernel_engineering",
        "trimul",
        "mla_decode_nvidia",
        "mla",
    }
    if gpu_problem:
        if config.kernel_gpu_id is None:
            raise EvolveConfigError(
                "gpu_mode EVOLVE runs require an exclusive kernel_gpu_id"
            )
        if config.reward_workers != 1:
            raise EvolveConfigError(
                "gpu_mode EVOLVE runs require reward_workers=1 for comparable timings"
            )
        if "task_yaml" not in config.problem_config:
            raise EvolveConfigError("gpu_mode requires problem key task_yaml")
        if not ({"lib_dir", "kernel_lib_dir"} & set(config.problem_config)):
            raise EvolveConfigError("gpu_mode requires problem key lib_dir")

    if config.problem_config.get("reranker_enabled") is True:
        raise UnsupportedEvolveConfigError(
            "the legacy LLM/Elo reranker is not an EVOLVE allocation signal"
        )


def _config_from_mapping(
    raw: Mapping[str, Any],
    *,
    cwd: Path,
    require_complete: bool,
) -> Tuple[EvolveConfig, List[str]]:
    if not isinstance(raw, Mapping):
        raise EvolveConfigError("configuration root must be a mapping")
    raw = copy.deepcopy(dict(raw))

    if require_complete:
        required = _COMMON_KEYS - {"config_hash"}
        missing = required - set(raw)
        if missing:
            raise EvolveConfigError(
                f"resumed config is incomplete; missing key(s): {sorted(missing)}"
            )

    unknown = [key for key in raw if key not in _COMMON_KEYS and not _allowed_problem_key(key)]
    if unknown:
        raise EvolveConfigError(
            f"unknown top-level config key(s): {sorted(unknown)}; "
            "new problem keys must be explicitly registered"
        )

    schema = _int(raw.get("schema_version", CONFIG_SCHEMA_VERSION), "schema_version", minimum=1)
    if schema != CONFIG_SCHEMA_VERSION:
        raise UnsupportedEvolveConfigError(
            f"unsupported EVOLVE schema_version {schema}; supported: {CONFIG_SCHEMA_VERSION}"
        )
    engine = _string(raw.get("engine", ENGINE_NAME), "engine").lower()
    if engine != ENGINE_NAME:
        raise UnsupportedEvolveConfigError(
            f"this repository supports only engine='evolve', got {engine!r}"
        )

    settings = _build_settings(
        raw.get("evolve", _settings_dict(EvolveSettings())),
        require_complete=require_complete,
    )
    if "num_steps" in raw and _int(raw["num_steps"], "num_steps", minimum=1) != settings.budget.epochs:
        raise EvolveConfigError("num_steps must equal evolve.budget.epochs")
    if "group_size" in raw and _int(raw["group_size"], "group_size", minimum=2) != settings.learning.group_k:
        raise EvolveConfigError("group_size must equal evolve.learning.group_k")

    problem = _string(raw.get("problem", "circle_packing"), "problem").lower()
    problem_type = _string(raw.get("problem_type", ""), "problem_type", nonempty=False)
    target_raw = raw.get("target")
    target = None if target_raw is None else _number(target_raw, "target")
    num_seed_states = _int(raw.get("num_seed_states", 8), "num_seed_states", minimum=1)

    model_name = _string(raw.get("model_name", "Qwen/Qwen3-8B"), "model_name")
    backend = _string(raw.get("backend", "auto"), "backend").lower()
    generation_backend = _string(
        raw.get("generation_backend", "hf"), "generation_backend"
    ).lower()
    if backend == "vllm":
        backend, generation_backend = "hf", "vllm"
    if backend not in ("auto", "unsloth", "hf"):
        raise EvolveConfigError("backend must be auto, unsloth, hf, or vllm")
    if generation_backend not in ("hf", "vllm"):
        raise EvolveConfigError("generation_backend must be 'hf' or 'vllm'")

    max_seq_length = _int(raw.get("max_seq_length", 32_000), "max_seq_length", minimum=2)
    max_new_tokens = _int(raw.get("max_new_tokens", 4_200), "max_new_tokens", minimum=1)
    if max_new_tokens >= max_seq_length:
        raise EvolveConfigError("max_new_tokens must be smaller than max_seq_length")
    temperature = _positive_number(raw.get("temperature", 1.0), "temperature")
    top_p = _number(raw.get("top_p", 1.0), "top_p")
    if not 0.0 < top_p <= 1.0:
        raise EvolveConfigError("top_p must be in (0, 1]")
    thinking = _bool(raw.get("thinking", False), "thinking")
    load_in_4bit = _bool(raw.get("load_in_4bit", False), "load_in_4bit")
    lora_rank = _int(raw.get("lora_rank", 32), "lora_rank", minimum=1)
    lora_alpha = _int(raw.get("lora_alpha", 32), "lora_alpha", minimum=1)
    lora_dropout = _fraction(raw.get("lora_dropout", 0.0), "lora_dropout", allow_one=False)
    target_modules = _string_tuple(
        raw.get("target_modules", EvolveConfig().target_modules), "target_modules"
    )
    learning_rate = _positive_number(raw.get("learning_rate", 4e-5), "learning_rate")
    kl_penalty_coef = _number(raw.get("kl_penalty_coef", 0.1), "kl_penalty_coef")
    if kl_penalty_coef < 0.0:
        raise EvolveConfigError("kl_penalty_coef must be >= 0")

    sandbox_timeout_s = _positive_number(
        raw.get("sandbox_timeout_s", 30.0), "sandbox_timeout_s"
    )
    kernel_timeout_raw = raw.get("kernel_timeout_s")
    kernel_timeout_s = (
        None
        if kernel_timeout_raw is None
        else _positive_number(kernel_timeout_raw, "kernel_timeout_s")
    )
    reward_workers = _int(raw.get("reward_workers", 0), "reward_workers", minimum=0)
    gpu_ids = parse_gpu_ids(raw.get("gpu_ids", [0]))
    declared_num_gpus = _int(raw.get("num_gpus", len(gpu_ids)), "num_gpus", minimum=0)
    num_gpus = len(gpu_ids)
    derivations: List[str] = []
    if declared_num_gpus != num_gpus:
        derivations.append(
            f"num_gpus={declared_num_gpus} replaced by authoritative "
            f"len(gpu_ids)={num_gpus}"
        )
    kernel_gpu_raw = raw.get("kernel_gpu_id")
    kernel_gpu_id = (
        None
        if kernel_gpu_raw is None
        else _int(kernel_gpu_raw, "kernel_gpu_id", minimum=0)
    )
    gen_micro_batch = _int(raw.get("gen_micro_batch", 0), "gen_micro_batch", minimum=0)
    vllm_util = _number(
        raw.get("vllm_gpu_memory_utilization", 0.9),
        "vllm_gpu_memory_utilization",
    )
    if not 0.0 < vllm_util <= 1.0:
        raise EvolveConfigError("vllm_gpu_memory_utilization must be in (0, 1]")
    vllm_eager = _bool(raw.get("vllm_enforce_eager", False), "vllm_enforce_eager")
    vllm_prefix = _bool(
        raw.get("vllm_enable_prefix_caching", True),
        "vllm_enable_prefix_caching",
    )
    seed = _int(raw.get("seed", 42), "seed", minimum=0)
    deterministic = _bool(raw.get("deterministic", False), "deterministic")
    checkpoint_path = _optional_path(raw.get("checkpoint_path"), "checkpoint_path", cwd)

    extras: Dict[str, Any] = {}
    for key, value in raw.items():
        if key in _COMMON_KEYS:
            continue
        extras[key] = _json_safe(value)
    # Resource and scientific-complexity controls affect verifier behavior and
    # must fail at the strict configuration boundary, before any model or
    # worker is loaded.  Keep their JSON representation as integers so the
    # resolved manifest and verifier identity agree exactly.
    for key in (
        "eval_cpus",
        "eval_memory_mb",
        "scientific_max_points",
        "scientific_max_coefficients",
    ):
        if key in extras:
            extras[key] = _int(extras[key], key, minimum=1)
    if "eval_seed" in extras:
        extras["eval_seed"] = _int(extras["eval_seed"], "eval_seed", minimum=0)
    if "budget_s" in extras:
        extras["budget_s"] = _positive_number(extras["budget_s"], "budget_s")
    if "task_yaml" in extras:
        extras["task_yaml"] = _path_from_config(
            extras["task_yaml"], "task_yaml", cwd, directory=False
        )
    if "lib_dir" in extras:
        extras["lib_dir"] = _path_from_config(
            extras["lib_dir"], "lib_dir", cwd, directory=True
        )
    if "kernel_lib_dir" in extras:
        extras["kernel_lib_dir"] = _path_from_config(
            extras["kernel_lib_dir"], "kernel_lib_dir", cwd, directory=True
        )

    config = EvolveConfig(
        schema_version=schema,
        engine=engine,
        problem=problem,
        problem_type=problem_type,
        target=target,
        num_seed_states=num_seed_states,
        model_name=model_name,
        backend=backend,
        generation_backend=generation_backend,
        max_seq_length=max_seq_length,
        max_new_tokens=max_new_tokens,
        temperature=temperature,
        top_p=top_p,
        thinking=thinking,
        load_in_4bit=load_in_4bit,
        lora_rank=lora_rank,
        lora_alpha=lora_alpha,
        lora_dropout=lora_dropout,
        target_modules=target_modules,
        learning_rate=learning_rate,
        kl_penalty_coef=kl_penalty_coef,
        sandbox_timeout_s=sandbox_timeout_s,
        kernel_timeout_s=kernel_timeout_s,
        reward_workers=reward_workers,
        gpu_ids=gpu_ids,
        num_gpus=num_gpus,
        kernel_gpu_id=kernel_gpu_id,
        gen_micro_batch=gen_micro_batch,
        vllm_gpu_memory_utilization=vllm_util,
        vllm_enforce_eager=vllm_eager,
        vllm_enable_prefix_caching=vllm_prefix,
        seed=seed,
        deterministic=deterministic,
        checkpoint_path=checkpoint_path,
        evolve=settings,
        problem_config=extras,
    )
    _validate_cross_section(config)
    return config, derivations


def _deep_merge(base: Mapping[str, Any], overlay: Mapping[str, Any]) -> Dict[str, Any]:
    out = copy.deepcopy(dict(base))
    for key, value in overlay.items():
        if isinstance(value, Mapping) and isinstance(out.get(key), Mapping):
            out[key] = _deep_merge(out[key], value)
        else:
            out[key] = copy.deepcopy(value)
    return out


def _read_yaml(path: Path) -> Dict[str, Any]:
    try:
        import yaml
    except ImportError as exc:  # pragma: no cover - repository dependency
        raise EvolveConfigError("PyYAML is required to load problem configurations") from exc
    try:
        value = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    except OSError as exc:
        raise EvolveConfigError(f"cannot read config YAML {path}: {exc}") from exc
    except yaml.YAMLError as exc:
        raise EvolveConfigError(f"invalid YAML in {path}: {exc}") from exc
    if not isinstance(value, dict):
        raise EvolveConfigError(f"config YAML must contain a mapping: {path}")
    return value


def _read_json_object(path: Path) -> Dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except OSError as exc:
        raise EvolveConfigError(f"cannot read {path}: {exc}") from exc
    except json.JSONDecodeError as exc:
        raise EvolveConfigError(f"invalid JSON in {path}: {exc}") from exc
    if not isinstance(value, dict):
        raise EvolveConfigError(f"configuration must be a JSON object: {path}")
    return value


def _apply_compatibility_aliases(layer: Mapping[str, Any]) -> Dict[str, Any]:
    """Translate old flat K/step names only when the nested value is absent."""
    out = copy.deepcopy(dict(layer))
    evolve = out.setdefault("evolve", {})
    if not isinstance(evolve, Mapping):
        return out  # strict builder will produce the useful error
    evolve = copy.deepcopy(dict(evolve))
    out["evolve"] = evolve
    budget = evolve.setdefault("budget", {})
    learning = evolve.setdefault("learning", {})
    if isinstance(budget, Mapping):
        budget = copy.deepcopy(dict(budget))
        evolve["budget"] = budget
        if "num_steps" in out and "epochs" not in budget:
            budget["epochs"] = out["num_steps"]
    if isinstance(learning, Mapping):
        learning = copy.deepcopy(dict(learning))
        evolve["learning"] = learning
        if "group_size" in out and "group_k" not in learning:
            learning["group_k"] = out["group_size"]
    # Preserve the legacy convenience spelling: num_gpus alone expands to
    # physical devices 0..N-1.  When gpu_ids is present, it stays authoritative.
    if "num_gpus" in out and "gpu_ids" not in out:
        count = _int(out["num_gpus"], "num_gpus", minimum=0)
        out["gpu_ids"] = list(range(count))
    # The nested values are authoritative after translation.
    out.pop("num_steps", None)
    out.pop("group_size", None)
    return out


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="EVOLVE verified scientific search",
        allow_abbrev=False,
    )
    parser.add_argument("--engine", choices=("evolve",), default=None)
    parser.add_argument("--schema-version", type=int, default=None)
    parser.add_argument("--problem", default=None)
    parser.add_argument("--problem-type", default=None)
    parser.add_argument("--gpu-type", default=None)
    parser.add_argument("--num-circles", type=int, default=None)
    parser.add_argument("--config", default=None)
    parser.add_argument("--resume", "--resume-from", dest="resume", default=None)
    parser.add_argument("--validate-config", action="store_true")
    parser.add_argument("--dry-plan", action="store_true")

    parser.add_argument("--model-name", default=None)
    parser.add_argument("--backend", choices=("auto", "unsloth", "hf", "vllm"), default=None)
    parser.add_argument("--generation-backend", choices=("hf", "vllm"), default=None)
    parser.add_argument("--max-seq-length", type=int, default=None)
    parser.add_argument("--max-new-tokens", type=int, default=None)
    parser.add_argument("--temperature", type=float, default=None)
    parser.add_argument("--top-p", type=float, default=None)
    parser.add_argument(
        "--thinking", dest="thinking", action="store_const", const=True, default=None
    )
    parser.add_argument(
        "--no-thinking", dest="thinking", action="store_const", const=False
    )
    parser.add_argument("--load-in-4bit", action="store_const", const=True, default=None)
    parser.add_argument("--lora-rank", type=int, default=None)
    parser.add_argument("--lora-alpha", type=int, default=None)
    parser.add_argument("--lora-dropout", type=float, default=None)
    parser.add_argument("--target-modules", default=None, help="comma-separated PEFT modules")
    parser.add_argument("--learning-rate", "--lr", dest="learning_rate", type=float, default=None)
    parser.add_argument("--kl-penalty-coef", type=float, default=None)
    parser.add_argument("--sandbox-timeout-s", type=float, default=None)
    parser.add_argument("--kernel-timeout-s", type=float, default=None)
    parser.add_argument("--reward-workers", type=int, default=None)
    parser.add_argument("--gpu-ids", default=None)
    parser.add_argument("--num-gpus", type=int, default=None)
    parser.add_argument("--kernel-gpu-id", type=int, default=None)
    parser.add_argument("--gen-micro-batch", type=int, default=None)
    parser.add_argument("--vllm-gpu-memory-utilization", type=float, default=None)
    parser.add_argument(
        "--vllm-enforce-eager",
        dest="vllm_enforce_eager",
        action="store_const",
        const=True,
        default=None,
    )
    parser.add_argument(
        "--no-vllm-enforce-eager",
        dest="vllm_enforce_eager",
        action="store_const",
        const=False,
    )
    parser.add_argument(
        "--vllm-prefix-caching",
        dest="vllm_enable_prefix_caching",
        action="store_const",
        const=True,
        default=None,
    )
    parser.add_argument(
        "--no-vllm-prefix-caching",
        dest="vllm_enable_prefix_caching",
        action="store_const",
        const=False,
    )
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument(
        "--deterministic",
        dest="deterministic",
        action="store_const",
        const=True,
        default=None,
    )
    parser.add_argument(
        "--no-deterministic",
        dest="deterministic",
        action="store_const",
        const=False,
    )
    parser.add_argument("--checkpoint-path", default=None)
    parser.add_argument("--target", type=float, default=None)
    parser.add_argument("--num-seed-states", type=int, default=None)

    # Compatibility alias. Both spellings target an absolute total epoch count.
    parser.add_argument("--num-steps", type=int, default=None)
    parser.add_argument("--epochs", type=int, default=None)
    parser.add_argument("--verifier-calls", type=int, default=None)
    parser.add_argument("--audit-fraction", type=float, default=None)
    parser.add_argument("--refinement-fraction", type=float, default=None)
    parser.add_argument("--elites-per-cell", type=int, default=None)
    parser.add_argument("--empty-cell-fraction", type=float, default=None)
    parser.add_argument("--roles", default=None, help="comma-separated role names")
    parser.add_argument("--test-mode", action="store_const", const=True, default=None)
    parser.add_argument("--method-incomplete", action="store_const", const=True, default=None)
    parser.add_argument("--max-horizon", type=int, default=None)
    parser.add_argument("--branch-budget", type=int, default=None)
    parser.add_argument("--harness-trial-fraction", type=float, default=None)
    parser.add_argument("--active-harnesses", default=None)
    parser.add_argument("--posterior", default=None)
    parser.add_argument("--global-exploration-fraction", type=float, default=None)
    parser.add_argument("--no-memory-audit-fraction", type=float, default=None)
    parser.add_argument("--min-pairs-for-promotion", type=int, default=None)
    parser.add_argument("--objective", default=None)
    parser.add_argument("--top-m", type=int, default=None)
    parser.add_argument("--group-k", type=int, default=None)
    parser.add_argument("--group-size", type=int, default=None)
    parser.add_argument("--refinement-max-attempts", type=int, default=None)
    parser.add_argument("--refinement-max-depth", type=int, default=None)
    parser.add_argument("--status-every-verifications", type=int, default=None)
    parser.add_argument("--plots-every-epochs", type=int, default=None)
    parser.add_argument("--max-inflight-branches", type=int, default=None)
    return parser


def parse_evolve_args(argv: Optional[Sequence[str]] = None) -> argparse.Namespace:
    """Parse EVOLVE CLI/config flags without loading models or changing files."""
    return _parser().parse_args(argv)


_TOP_LEVEL_CLI = {
    "engine": "engine",
    "schema_version": "schema_version",
    "problem": "problem",
    "problem_type": "problem_type",
    "gpu_type": "gpu_type",
    "num_circles": "num_circles",
    "model_name": "model_name",
    "backend": "backend",
    "generation_backend": "generation_backend",
    "max_seq_length": "max_seq_length",
    "max_new_tokens": "max_new_tokens",
    "temperature": "temperature",
    "top_p": "top_p",
    "thinking": "thinking",
    "load_in_4bit": "load_in_4bit",
    "lora_rank": "lora_rank",
    "lora_alpha": "lora_alpha",
    "lora_dropout": "lora_dropout",
    "learning_rate": "learning_rate",
    "kl_penalty_coef": "kl_penalty_coef",
    "sandbox_timeout_s": "sandbox_timeout_s",
    "kernel_timeout_s": "kernel_timeout_s",
    "reward_workers": "reward_workers",
    "gpu_ids": "gpu_ids",
    "num_gpus": "num_gpus",
    "kernel_gpu_id": "kernel_gpu_id",
    "gen_micro_batch": "gen_micro_batch",
    "vllm_gpu_memory_utilization": "vllm_gpu_memory_utilization",
    "vllm_enforce_eager": "vllm_enforce_eager",
    "vllm_enable_prefix_caching": "vllm_enable_prefix_caching",
    "seed": "seed",
    "deterministic": "deterministic",
    "checkpoint_path": "checkpoint_path",
    "target": "target",
    "num_seed_states": "num_seed_states",
}


_NESTED_CLI = {
    "verifier_calls": ("budget", "verifier_calls"),
    "audit_fraction": ("budget", "audit_fraction"),
    "refinement_fraction": ("budget", "refinement_fraction"),
    "elites_per_cell": ("archive", "elites_per_cell"),
    "empty_cell_fraction": ("archive", "empty_cell_fraction"),
    "roles": ("roles", "enabled"),
    "test_mode": ("roles", "test_mode"),
    "method_incomplete": ("roles", "method_incomplete"),
    "max_horizon": ("options", "max_horizon"),
    "branch_budget": ("options", "branch_budget"),
    "harness_trial_fraction": ("harnesses", "trial_fraction"),
    "active_harnesses": ("harnesses", "active_versions"),
    "posterior": ("scheduler", "posterior"),
    "global_exploration_fraction": ("scheduler", "global_exploration_fraction"),
    "no_memory_audit_fraction": ("audits", "no_memory_fraction"),
    "min_pairs_for_promotion": ("audits", "min_pairs_for_promotion"),
    "objective": ("learning", "objective"),
    "top_m": ("learning", "top_m"),
    "group_k": ("learning", "group_k"),
    "refinement_max_attempts": ("refinement", "max_attempts"),
    "refinement_max_depth": ("refinement", "max_depth"),
    "status_every_verifications": ("reporting", "status_every_verifications"),
    "plots_every_epochs": ("reporting", "plots_every_epochs"),
    "max_inflight_branches": ("workers", "max_inflight_branches"),
}


def _csv_strings(value: str, name: str) -> List[str]:
    out = [piece.strip() for piece in value.split(",") if piece.strip()]
    if not out:
        raise EvolveConfigError(f"{name} must include at least one value")
    return out


def _cli_layer(args: argparse.Namespace) -> Tuple[Dict[str, Any], Dict[str, Any]]:
    layer: Dict[str, Any] = {}
    recorded: Dict[str, Any] = {}
    for arg_name, config_name in _TOP_LEVEL_CLI.items():
        value = getattr(args, arg_name)
        if value is None:
            continue
        if arg_name == "target_modules":
            value = _csv_strings(value, "target_modules")
        layer[config_name] = value
        recorded[config_name] = _json_safe(value)

    if args.target_modules is not None:
        modules = _csv_strings(args.target_modules, "target_modules")
        layer["target_modules"] = modules
        recorded["target_modules"] = modules

    epochs = args.epochs
    if args.num_steps is not None:
        if epochs is not None and epochs != args.num_steps:
            raise EvolveConfigError("--epochs and --num-steps disagree")
        epochs = args.num_steps
    if epochs is not None:
        layer.setdefault("evolve", {}).setdefault("budget", {})["epochs"] = epochs
        recorded["evolve.budget.epochs"] = epochs

    group_k = args.group_k
    if args.group_size is not None:
        if group_k is not None and group_k != args.group_size:
            raise EvolveConfigError("--group-k and --group-size disagree")
        group_k = args.group_size

    for arg_name, (section, key) in _NESTED_CLI.items():
        value = group_k if arg_name == "group_k" else getattr(args, arg_name)
        if value is None:
            continue
        if arg_name in ("roles", "active_harnesses"):
            value = _csv_strings(value, arg_name)
        layer.setdefault("evolve", {}).setdefault(section, {})[key] = value
        recorded[f"evolve.{section}.{key}"] = _json_safe(value)
    return layer, recorded


_RESUME_MUTABLE_ARGS = {
    "engine",
    "schema_version",
    "gpu_ids",
    "num_gpus",
    "kernel_gpu_id",
    "reward_workers",
    "epochs",
    "num_steps",
    "max_inflight_branches",
}


def _explicit_config_args(args: argparse.Namespace) -> set:
    operational = {"resume", "validate_config", "dry_plan"}
    return {
        key
        for key, value in vars(args).items()
        if key not in operational and value is not None
    }


def _finalize(config: EvolveConfig) -> Tuple[Dict[str, Any], str]:
    resolved = config.to_dict(compatibility=True)
    config_hash = canonical_config_hash(resolved)
    resolved["config_hash"] = config_hash
    return resolved, config_hash


def validate_resolved_config_document(
    document: Mapping[str, Any],
    cwd: Optional[Path] = None,
) -> EvolveConfig:
    """Validate one complete canonical schema-v1 resolved document.

    Unlike fresh YAML resolution, this function applies no defaults and performs
    no compatibility coercions. It is suitable for immutable metadata writers
    and resume readers that must reject a partial or reinterpreted document.
    """

    if not isinstance(document, Mapping):
        raise EvolveConfigError("resolved configuration must be a mapping")
    base_dir = Path(cwd) if cwd is not None else Path.cwd()
    base_dir = base_dir.expanduser().resolve()
    saved = copy.deepcopy(dict(document))
    previous_hash = saved.get("config_hash")
    if not isinstance(previous_hash, str) or not previous_hash:
        raise EvolveConfigError("resolved configuration has no config_hash")
    if previous_hash != canonical_config_hash(saved):
        raise EvolveConfigError("resolved configuration config_hash mismatch")

    without_hash = copy.deepcopy(saved)
    without_hash.pop("config_hash", None)
    config, _ = _config_from_mapping(
        without_hash,
        cwd=base_dir,
        require_complete=True,
    )
    canonical, canonical_hash = _finalize(config)
    if canonical != saved or canonical_hash != previous_hash:
        raise EvolveConfigError(
            "resolved configuration is not complete canonical schema-v1 JSON; "
            "refusing implicit defaults or reinterpretation"
        )
    return config


def load_evolve_config(
    argv: Optional[Sequence[str]] = None,
    cwd: Optional[Path] = None,
) -> Tuple[EvolveConfig, Dict[str, Any], Dict[str, Any]]:
    """Resolve and validate an EVOLVE configuration without runtime side effects.

    Returns ``(config, resolved_dict, metadata)``.  ``resolved_dict`` is ready for
    ``config.resolved.json`` and uses real JSON types.  ``metadata`` describes
    sources/derivations and carries operational flags such as ``dry_plan``.
    """
    args = parse_evolve_args(argv)
    base_dir = Path(cwd) if cwd is not None else Path.cwd()
    base_dir = base_dir.expanduser().resolve()
    cli, cli_record = _cli_layer(args)
    if args.gpu_ids is not None:
        explicit_ids = parse_gpu_ids(args.gpu_ids)
        if args.num_gpus is not None and args.num_gpus != len(explicit_ids):
            raise EvolveConfigError(
                f"--num-gpus={args.num_gpus} disagrees with "
                f"{len(explicit_ids)} id(s) in --gpu-ids"
            )
        cli["gpu_ids"] = list(explicit_ids)
        cli["num_gpus"] = len(explicit_ids)
        cli_record["gpu_ids"] = list(explicit_ids)
        cli_record["num_gpus"] = len(explicit_ids)
    elif args.num_gpus is not None:
        count = _int(args.num_gpus, "num_gpus", minimum=0)
        cli["num_gpus"] = count
        cli_record["num_gpus"] = count
        if args.resume is None:
            cli["gpu_ids"] = list(range(count))
            cli_record["gpu_ids"] = list(range(count))

    if args.resume is not None:
        if args.config is not None:
            raise EvolveConfigError("--config cannot be used with --resume")
        explicit = _explicit_config_args(args)
        unsupported = explicit - _RESUME_MUTABLE_ARGS
        if unsupported:
            raise EvolveConfigError(
                "resume override(s) are not supported because they change frozen "
                f"method/generation settings: {sorted(unsupported)}"
            )
        if args.engine not in (None, ENGINE_NAME):
            raise UnsupportedEvolveConfigError("an EVOLVE run cannot resume as legacy")
        if args.schema_version not in (None, CONFIG_SCHEMA_VERSION):
            raise UnsupportedEvolveConfigError("resume cannot migrate config schema implicitly")

        resume_dir = Path(args.resume).expanduser()
        if not resume_dir.is_absolute():
            resume_dir = base_dir / resume_dir
        resume_dir = resume_dir.resolve()
        if not resume_dir.is_dir():
            raise EvolveConfigError(f"resume directory does not exist: {resume_dir}")

        try:
            effective_metadata = resolve_effective_run_metadata(resume_dir)
        except UnsupportedRunSchemaError as exc:
            raise UnsupportedEvolveConfigError(
                f"invalid EVOLVE resume metadata: {exc}"
            ) from exc
        except RunSchemaError as exc:
            raise EvolveConfigError(f"invalid EVOLVE resume metadata: {exc}") from exc

        compatibility_path = effective_metadata.compatibility_config_path
        resolved_path = effective_metadata.resolved_config_path
        manifest_path = effective_metadata.manifest_path
        saved = copy.deepcopy(effective_metadata.resolved_config)
        if saved.get("engine") != ENGINE_NAME:
            raise UnsupportedEvolveConfigError(
                "authoritative resumed configuration is not engine='evolve'"
            )
        saved_schema = saved.get("schema_version")
        if saved_schema != CONFIG_SCHEMA_VERSION:
            raise UnsupportedEvolveConfigError(
                f"unsupported resumed schema_version {saved_schema!r}"
            )
        previous_hash = saved.get("config_hash")
        if not isinstance(previous_hash, str) or not previous_hash:
            raise EvolveConfigError("resumed config.resolved.json has no config_hash")
        actual_hash = canonical_config_hash(saved)
        if previous_hash != actual_hash:
            raise EvolveConfigError(
                "resumed config.resolved.json hash mismatch; refusing altered evidence"
            )
        manifest = effective_metadata.manifest
        if manifest.get("engine") != ENGINE_NAME:
            raise UnsupportedEvolveConfigError(
                "resumed manifest is not engine='evolve'"
            )
        if manifest.get("config_hash") != previous_hash:
            raise EvolveConfigError(
                "resumed manifest config_hash does not match authoritative config"
            )

        try:
            validate_resolved_config_document(saved, cwd=base_dir)
        except EvolveConfigError as exc:
            raise EvolveConfigError(
                f"authoritative resumed config is invalid: {exc}"
            ) from exc

        if args.num_gpus is not None and args.gpu_ids is None:
            saved_ids = saved["gpu_ids"]
            if args.num_gpus != len(saved_ids):
                raise EvolveConfigError(
                    f"--num-gpus={args.num_gpus} conflicts with authoritative "
                    f"saved gpu_ids containing {len(saved_ids)} device(s); use "
                    "--gpu-ids for an explicit topology change"
                )

        merged = copy.deepcopy(saved)
        merged.pop("config_hash", None)
        merged = _deep_merge(merged, cli)
        # Keep compatibility aliases synchronized after a supported epoch change.
        merged["num_steps"] = merged["evolve"]["budget"]["epochs"]
        merged["group_size"] = merged["evolve"]["learning"]["group_k"]
        config, derivations = _config_from_mapping(
            merged, cwd=base_dir, require_complete=True
        )
        resolved, config_hash = _finalize(config)
        metadata = {
            "mode": "resume",
            "resume_dir": str(resume_dir),
            "config_path": str(resolved_path),
            "manifest_path": str(manifest_path),
            "effective_resume_index": effective_metadata.resume_index,
            "sources": [str(resolved_path), str(manifest_path), "explicit_cli"],
            "cli_overrides": cli_record,
            "derivations": derivations,
            "previous_config_hash": previous_hash,
            "config_hash": config_hash,
            "validate_config": bool(args.validate_config),
            "dry_plan": bool(args.dry_plan),
            "method_incomplete": config.method_incomplete,
        }
        return config, resolved, metadata

    defaults = EvolveConfig().to_dict(compatibility=False)
    selector = args.problem or defaults["problem"]
    if args.config is not None:
        config_path = Path(args.config).expanduser()
        if not config_path.is_absolute():
            config_path = base_dir / config_path
    else:
        config_path = base_dir / "configs" / f"{selector}.yaml"
    config_path = config_path.resolve()
    if not config_path.is_file():
        raise EvolveConfigError(f"problem config YAML does not exist: {config_path}")

    yaml_layer = _apply_compatibility_aliases(_read_yaml(config_path))
    merged = _deep_merge(defaults, yaml_layer)
    merged = _deep_merge(merged, cli)
    # CLI/new nested values are the source of truth for compatibility fields.
    merged["num_steps"] = merged["evolve"]["budget"]["epochs"]
    merged["group_size"] = merged["evolve"]["learning"]["group_k"]
    config, derivations = _config_from_mapping(
        merged, cwd=base_dir, require_complete=False
    )
    resolved, config_hash = _finalize(config)
    metadata = {
        "mode": "fresh",
        "resume_dir": None,
        "config_path": str(config_path),
        "sources": ["dataclass_defaults", str(config_path), "explicit_cli"],
        "cli_overrides": cli_record,
        "derivations": derivations,
        "previous_config_hash": None,
        "config_hash": config_hash,
        "validate_config": bool(args.validate_config),
        "dry_plan": bool(args.dry_plan),
        "method_incomplete": config.method_incomplete,
    }
    return config, resolved, metadata


__all__ = [
    "CONFIG_SCHEMA_VERSION",
    "ENGINE_NAME",
    "ROLE_NAMES",
    "EvolveConfigError",
    "UnsupportedEvolveConfigError",
    "BudgetConfig",
    "ArchiveConfig",
    "RolesConfig",
    "OptionsConfig",
    "HarnessesConfig",
    "SchedulerConfig",
    "AuditsConfig",
    "LearningConfig",
    "RefinementConfig",
    "ReportingConfig",
    "WorkersConfig",
    "EvolveSettings",
    "EvolveConfig",
    "parse_gpu_ids",
    "parse_evolve_args",
    "validate_resolved_config_document",
    "load_evolve_config",
    "canonical_config_hash",
]
