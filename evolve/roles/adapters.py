"""Immutable role-owned adapter, optimizer, and random-stream state.

This module deliberately has no model-framework dependency.  It establishes
the ownership and checkpoint boundary that HF/vLLM adapters must satisfy: one
stable parameter-set identity and optimizer owner per role, plus a complete
role-local random stream.  Backend integrations attach real LoRA tensors to
these identities; they must not reinterpret these CPU-safe records as proof
that a backend supports named adapters.
"""

from __future__ import annotations

from dataclasses import dataclass, replace
from typing import Any, Mapping, Tuple

from evolve.ids import canonical_json, content_hash, content_id, derive_seed, validate_id
from evolve.types import FrozenDict, Role


ROLE_STATE_SCHEMA_VERSION = 1
ADAPTER_STATE_VERSION = "role_adapter_state_v1"
OPTIMIZER_STATE_VERSION = "role_optimizer_state_v1"
RNG_STATE_VERSION = "role_rng_counter_v1"


class RoleIsolationError(ValueError):
    """A role-owned state is malformed, aliased, or advanced inconsistently."""


def coerce_role(value: Any) -> Role:
    try:
        return value if isinstance(value, Role) else Role(value)
    except (TypeError, ValueError) as exc:
        raise RoleIsolationError(f"unknown EVOLVE role: {value!r}") from exc


def require_nonempty(value: Any, name: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise RoleIsolationError(f"{name} must be a non-empty string")
    return value


def require_nonnegative_int(value: Any, name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        raise RoleIsolationError(f"{name} must be a non-negative integer")
    return value


def require_id(value: Any, namespace: str, name: str) -> str:
    try:
        return validate_id(value, namespace)
    except (TypeError, ValueError) as exc:
        raise RoleIsolationError(f"invalid {name}: {exc}") from exc


def frozen_mapping(value: Any, name: str) -> FrozenDict:
    if not isinstance(value, Mapping):
        raise RoleIsolationError(f"{name} must be a JSON mapping")
    try:
        result = FrozenDict(value)
        canonical_json(result)
        return result
    except (TypeError, ValueError) as exc:
        raise RoleIsolationError(f"{name} must be JSON-safe: {exc}") from exc


def thaw_json(value: Any) -> Any:
    if isinstance(value, Mapping):
        return {str(key): thaw_json(item) for key, item in value.items()}
    if isinstance(value, tuple):
        return [thaw_json(item) for item in value]
    if isinstance(value, Role):
        return value.value
    return value


def require_exact_keys(payload: Mapping[str, Any], keys: Tuple[str, ...], name: str) -> None:
    if not isinstance(payload, Mapping):
        raise RoleIsolationError(f"{name} must be a mapping")
    if any(not isinstance(key, str) for key in payload):
        raise RoleIsolationError(f"{name} keys must be strings")
    expected = set(keys)
    actual = set(payload)
    if actual != expected:
        missing = sorted(expected - actual)
        unknown = sorted(actual - expected)
        raise RoleIsolationError(
            f"{name} fields differ (missing={missing}, unknown={unknown})"
        )


def require_schema_version(value: Any, expected: int, name: str) -> None:
    if isinstance(value, bool) or not isinstance(value, int) or value != expected:
        raise RoleIsolationError(f"unsupported {name} schema {value!r}")


def adapter_version(revision: int) -> str:
    require_nonnegative_int(revision, "adapter revision")
    return f"adapter_v{revision:06d}"


@dataclass(frozen=True)
class RoleAdapterState:
    """A JSON-safe manifest for one role's logically disjoint LoRA state."""

    run_id: str
    role: Role
    adapter_id: str
    parameter_set_id: str
    revision: int
    state: FrozenDict
    adapter_hash: str
    state_version: str = ADAPTER_STATE_VERSION
    schema_version: int = ROLE_STATE_SCHEMA_VERSION

    @classmethod
    def create(
        cls,
        *,
        run_id: str,
        role: Any,
        state: Mapping[str, Any],
    ) -> "RoleAdapterState":
        owner = coerce_role(role)
        require_id(run_id, "run", "run_id")
        adapter_id = content_id(
            "adapter",
            {
                "state_version": ADAPTER_STATE_VERSION,
                "run_id": run_id,
                "role": owner.value,
            },
        )
        parameter_set_id = content_id(
            "adapter_parameters",
            {
                "state_version": ADAPTER_STATE_VERSION,
                "run_id": run_id,
                "role": owner.value,
                "adapter_id": adapter_id,
            },
        )
        frozen = frozen_mapping(state, "adapter state")
        digest = cls._state_hash(
            run_id=run_id,
            role=owner,
            adapter_id=adapter_id,
            parameter_set_id=parameter_set_id,
            state=frozen,
        )
        return cls(
            run_id=run_id,
            role=owner,
            adapter_id=adapter_id,
            parameter_set_id=parameter_set_id,
            revision=0,
            state=frozen,
            adapter_hash=digest,
        )

    @staticmethod
    def _state_hash(
        *,
        run_id: str,
        role: Role,
        adapter_id: str,
        parameter_set_id: str,
        state: Mapping[str, Any],
    ) -> str:
        # Ownership metadata is part of the adapter artifact.  Consequently
        # initially equal numerical LoRA weights still cannot alias roles.
        return content_hash(
            {
                "state_version": ADAPTER_STATE_VERSION,
                "run_id": run_id,
                "role": role.value,
                "adapter_id": adapter_id,
                "parameter_set_id": parameter_set_id,
                "state": state,
            }
        )

    @property
    def version(self) -> str:
        return adapter_version(self.revision)

    def __post_init__(self) -> None:
        owner = coerce_role(self.role)
        object.__setattr__(self, "role", owner)
        require_id(self.run_id, "run", "run_id")
        require_id(self.adapter_id, "adapter", "adapter_id")
        require_id(self.parameter_set_id, "adapter_parameters", "parameter_set_id")
        require_nonnegative_int(self.revision, "adapter revision")
        require_schema_version(
            self.schema_version, ROLE_STATE_SCHEMA_VERSION, "role adapter"
        )
        if self.state_version != ADAPTER_STATE_VERSION:
            raise RoleIsolationError(
                f"unsupported adapter state version {self.state_version!r}"
            )
        frozen = frozen_mapping(self.state, "adapter state")
        object.__setattr__(self, "state", frozen)
        expected_adapter_id = content_id(
            "adapter",
            {
                "state_version": ADAPTER_STATE_VERSION,
                "run_id": self.run_id,
                "role": owner.value,
            },
        )
        expected_parameters = content_id(
            "adapter_parameters",
            {
                "state_version": ADAPTER_STATE_VERSION,
                "run_id": self.run_id,
                "role": owner.value,
                "adapter_id": self.adapter_id,
            },
        )
        if self.adapter_id != expected_adapter_id:
            raise RoleIsolationError("adapter_id does not match run/role ownership")
        if self.parameter_set_id != expected_parameters:
            raise RoleIsolationError(
                "parameter_set_id does not match its role adapter"
            )
        expected_hash = self._state_hash(
            run_id=self.run_id,
            role=owner,
            adapter_id=self.adapter_id,
            parameter_set_id=self.parameter_set_id,
            state=frozen,
        )
        if self.adapter_hash != expected_hash:
            raise RoleIsolationError("adapter_hash does not match adapter state")

    def advance(self, state: Mapping[str, Any]) -> "RoleAdapterState":
        frozen = frozen_mapping(state, "adapter state")
        next_hash = self._state_hash(
            run_id=self.run_id,
            role=self.role,
            adapter_id=self.adapter_id,
            parameter_set_id=self.parameter_set_id,
            state=frozen,
        )
        if next_hash == self.adapter_hash:
            raise RoleIsolationError(
                "adapter version may advance only when its persisted state changes"
            )
        return replace(
            self,
            revision=self.revision + 1,
            state=frozen,
            adapter_hash=next_hash,
        )

    def to_dict(self) -> Mapping[str, Any]:
        return {
            "record_type": "role_adapter_state",
            "schema_version": self.schema_version,
            "state_version": self.state_version,
            "run_id": self.run_id,
            "role": self.role.value,
            "adapter_id": self.adapter_id,
            "parameter_set_id": self.parameter_set_id,
            "revision": self.revision,
            "state": thaw_json(self.state),
            "adapter_hash": self.adapter_hash,
        }

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> "RoleAdapterState":
        keys = (
            "record_type", "schema_version", "state_version", "run_id", "role",
            "adapter_id", "parameter_set_id", "revision", "state", "adapter_hash",
        )
        require_exact_keys(payload, keys, "role adapter state")
        if payload["record_type"] != "role_adapter_state":
            raise RoleIsolationError("invalid role adapter record_type")
        return cls(
            schema_version=payload["schema_version"],
            state_version=payload["state_version"],
            run_id=payload["run_id"],
            role=payload["role"],
            adapter_id=payload["adapter_id"],
            parameter_set_id=payload["parameter_set_id"],
            revision=payload["revision"],
            state=payload["state"],
            adapter_hash=payload["adapter_hash"],
        )


@dataclass(frozen=True)
class RoleOptimizerState:
    """One optimizer state bound to exactly one role parameter set."""

    run_id: str
    role: Role
    owner_id: str
    state_id: str
    parameter_set_id: str
    bound_adapter_version: str
    step: int
    state: FrozenDict
    state_hash: str
    state_version: str = OPTIMIZER_STATE_VERSION
    schema_version: int = ROLE_STATE_SCHEMA_VERSION

    @classmethod
    def create(
        cls,
        *,
        adapter: RoleAdapterState,
        state: Mapping[str, Any],
    ) -> "RoleOptimizerState":
        owner_id = content_id(
            "optimizer_owner",
            {
                "state_version": OPTIMIZER_STATE_VERSION,
                "run_id": adapter.run_id,
                "role": adapter.role.value,
                "parameter_set_id": adapter.parameter_set_id,
            },
        )
        frozen = frozen_mapping(state, "optimizer state")
        state_hash = cls._state_hash(
            run_id=adapter.run_id,
            role=adapter.role,
            owner_id=owner_id,
            parameter_set_id=adapter.parameter_set_id,
            state=frozen,
        )
        state_id = cls._state_id(
            owner_id=owner_id,
            step=0,
            bound_adapter_version=adapter.version,
            state_hash=state_hash,
        )
        return cls(
            run_id=adapter.run_id,
            role=adapter.role,
            owner_id=owner_id,
            state_id=state_id,
            parameter_set_id=adapter.parameter_set_id,
            bound_adapter_version=adapter.version,
            step=0,
            state=frozen,
            state_hash=state_hash,
        )

    @staticmethod
    def _state_hash(
        *,
        run_id: str,
        role: Role,
        owner_id: str,
        parameter_set_id: str,
        state: Mapping[str, Any],
    ) -> str:
        return content_hash(
            {
                "state_version": OPTIMIZER_STATE_VERSION,
                "run_id": run_id,
                "role": role.value,
                "owner_id": owner_id,
                "parameter_set_id": parameter_set_id,
                "state": state,
            }
        )

    @staticmethod
    def _state_id(
        *, owner_id: str, step: int, bound_adapter_version: str, state_hash: str
    ) -> str:
        return content_id(
            "optimizer",
            {
                "state_version": OPTIMIZER_STATE_VERSION,
                "owner_id": owner_id,
                "step": step,
                "bound_adapter_version": bound_adapter_version,
                "state_hash": state_hash,
            },
        )

    def __post_init__(self) -> None:
        owner = coerce_role(self.role)
        object.__setattr__(self, "role", owner)
        require_id(self.run_id, "run", "run_id")
        require_id(self.owner_id, "optimizer_owner", "owner_id")
        require_id(self.state_id, "optimizer", "state_id")
        require_id(self.parameter_set_id, "adapter_parameters", "parameter_set_id")
        require_nonempty(self.bound_adapter_version, "bound_adapter_version")
        require_nonnegative_int(self.step, "optimizer step")
        require_schema_version(
            self.schema_version, ROLE_STATE_SCHEMA_VERSION, "role optimizer"
        )
        if self.state_version != OPTIMIZER_STATE_VERSION:
            raise RoleIsolationError(
                f"unsupported optimizer state version {self.state_version!r}"
            )
        frozen = frozen_mapping(self.state, "optimizer state")
        object.__setattr__(self, "state", frozen)
        expected_owner = content_id(
            "optimizer_owner",
            {
                "state_version": OPTIMIZER_STATE_VERSION,
                "run_id": self.run_id,
                "role": owner.value,
                "parameter_set_id": self.parameter_set_id,
            },
        )
        if self.owner_id != expected_owner:
            raise RoleIsolationError("optimizer owner does not match its role parameter set")
        expected_hash = self._state_hash(
            run_id=self.run_id,
            role=owner,
            owner_id=self.owner_id,
            parameter_set_id=self.parameter_set_id,
            state=frozen,
        )
        if self.state_hash != expected_hash:
            raise RoleIsolationError("optimizer state_hash does not match state")
        expected_id = self._state_id(
            owner_id=self.owner_id,
            step=self.step,
            bound_adapter_version=self.bound_adapter_version,
            state_hash=self.state_hash,
        )
        if self.state_id != expected_id:
            raise RoleIsolationError("optimizer state_id does not match state content")

    def advance(
        self,
        *,
        adapter: RoleAdapterState,
        state: Mapping[str, Any],
    ) -> "RoleOptimizerState":
        if adapter.run_id != self.run_id or adapter.role != self.role:
            raise RoleIsolationError("optimizer cannot bind another run or role's adapter")
        if adapter.parameter_set_id != self.parameter_set_id:
            raise RoleIsolationError("optimizer cannot switch role parameter sets")
        frozen = frozen_mapping(state, "optimizer state")
        next_step = self.step + 1
        next_hash = self._state_hash(
            run_id=self.run_id,
            role=self.role,
            owner_id=self.owner_id,
            parameter_set_id=self.parameter_set_id,
            state=frozen,
        )
        next_id = self._state_id(
            owner_id=self.owner_id,
            step=next_step,
            bound_adapter_version=adapter.version,
            state_hash=next_hash,
        )
        return replace(
            self,
            state_id=next_id,
            bound_adapter_version=adapter.version,
            step=next_step,
            state=frozen,
            state_hash=next_hash,
        )

    def to_dict(self) -> Mapping[str, Any]:
        return {
            "record_type": "role_optimizer_state",
            "schema_version": self.schema_version,
            "state_version": self.state_version,
            "run_id": self.run_id,
            "role": self.role.value,
            "owner_id": self.owner_id,
            "state_id": self.state_id,
            "parameter_set_id": self.parameter_set_id,
            "bound_adapter_version": self.bound_adapter_version,
            "step": self.step,
            "state": thaw_json(self.state),
            "state_hash": self.state_hash,
        }

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> "RoleOptimizerState":
        keys = (
            "record_type", "schema_version", "state_version", "run_id", "role",
            "owner_id", "state_id", "parameter_set_id", "bound_adapter_version",
            "step", "state", "state_hash",
        )
        require_exact_keys(payload, keys, "role optimizer state")
        if payload["record_type"] != "role_optimizer_state":
            raise RoleIsolationError("invalid role optimizer record_type")
        return cls(
            schema_version=payload["schema_version"],
            state_version=payload["state_version"],
            run_id=payload["run_id"],
            role=payload["role"],
            owner_id=payload["owner_id"],
            state_id=payload["state_id"],
            parameter_set_id=payload["parameter_set_id"],
            bound_adapter_version=payload["bound_adapter_version"],
            step=payload["step"],
            state=payload["state"],
            state_hash=payload["state_hash"],
        )


@dataclass(frozen=True)
class RoleRNGState:
    """Checkpointable counter-based RNG state owned by one role.

    Hash-derived draws make role decisions independent of worker rank and
    completion order while avoiding unsafe pickle-only ``random.Random`` state.
    Every draw returns a replacement state, so snapshots cannot be mutated by
    an in-flight caller.
    """

    run_id: str
    role: Role
    owner_id: str
    base_seed: int
    counter: int = 0
    state_version: str = RNG_STATE_VERSION
    schema_version: int = ROLE_STATE_SCHEMA_VERSION

    @classmethod
    def create(cls, *, run_id: str, role: Any, base_seed: int) -> "RoleRNGState":
        owner = coerce_role(role)
        require_id(run_id, "run", "run_id")
        require_nonnegative_int(base_seed, "role RNG base_seed")
        owner_id = content_id(
            "role_rng",
            {
                "state_version": RNG_STATE_VERSION,
                "run_id": run_id,
                "role": owner.value,
                "base_seed": base_seed,
            },
        )
        return cls(
            run_id=run_id,
            role=owner,
            owner_id=owner_id,
            base_seed=base_seed,
        )

    def __post_init__(self) -> None:
        owner = coerce_role(self.role)
        object.__setattr__(self, "role", owner)
        require_id(self.run_id, "run", "run_id")
        require_id(self.owner_id, "role_rng", "owner_id")
        require_nonnegative_int(self.base_seed, "role RNG base_seed")
        require_nonnegative_int(self.counter, "role RNG counter")
        require_schema_version(
            self.schema_version, ROLE_STATE_SCHEMA_VERSION, "role RNG"
        )
        if self.state_version != RNG_STATE_VERSION:
            raise RoleIsolationError(f"unsupported role RNG version {self.state_version!r}")
        expected = content_id(
            "role_rng",
            {
                "state_version": RNG_STATE_VERSION,
                "run_id": self.run_id,
                "role": owner.value,
                "base_seed": self.base_seed,
            },
        )
        if self.owner_id != expected:
            raise RoleIsolationError("role RNG owner_id does not match run/role/seed")

    def draw_seed(self, purpose: str) -> Tuple[int, "RoleRNGState"]:
        require_nonempty(purpose, "RNG draw purpose")
        value = derive_seed(
            "role_rng_draw",
            self.run_id,
            self.role.value,
            self.owner_id,
            self.counter,
            purpose,
            base_seed=self.base_seed,
        )
        return value, replace(self, counter=self.counter + 1)

    def epoch_seed(self, epoch: int) -> int:
        require_nonnegative_int(epoch, "epoch")
        # Epoch policy snapshots must remain valid after private role-local
        # choices consume this stream.  The full counter is checkpointed for
        # those choices, while the frozen epoch seed depends only on stable
        # owner identity and epoch.
        return derive_seed(
            "role_policy_snapshot",
            self.run_id,
            self.role.value,
            self.owner_id,
            epoch,
            base_seed=self.base_seed,
        )

    def to_dict(self) -> Mapping[str, Any]:
        return {
            "record_type": "role_rng_state",
            "schema_version": self.schema_version,
            "state_version": self.state_version,
            "run_id": self.run_id,
            "role": self.role.value,
            "owner_id": self.owner_id,
            "base_seed": self.base_seed,
            "counter": self.counter,
        }

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> "RoleRNGState":
        keys = (
            "record_type", "schema_version", "state_version", "run_id", "role",
            "owner_id", "base_seed", "counter",
        )
        require_exact_keys(payload, keys, "role RNG state")
        if payload["record_type"] != "role_rng_state":
            raise RoleIsolationError("invalid role RNG record_type")
        return cls(
            schema_version=payload["schema_version"],
            state_version=payload["state_version"],
            run_id=payload["run_id"],
            role=payload["role"],
            owner_id=payload["owner_id"],
            base_seed=payload["base_seed"],
            counter=payload["counter"],
        )


__all__ = [
    "ADAPTER_STATE_VERSION",
    "OPTIMIZER_STATE_VERSION",
    "RNG_STATE_VERSION",
    "ROLE_STATE_SCHEMA_VERSION",
    "RoleAdapterState",
    "RoleIsolationError",
    "RoleOptimizerState",
    "RoleRNGState",
    "adapter_version",
    "coerce_role",
    "frozen_mapping",
    "require_exact_keys",
    "require_id",
    "require_nonempty",
    "require_nonnegative_int",
    "require_schema_version",
    "thaw_json",
]
