"""Production registry for EVOLVE's three isolated role policies.

The registry is a pure state machine.  Updates return a replacement registry
and preserve the other two role objects byte-for-byte, which makes accidental
cross-role optimizer or RNG mutation observable in tests and checkpoints.
Only epoch snapshots produced here may be attached to generation jobs.
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass, replace
from types import MappingProxyType
from typing import Any, Dict, Mapping, Optional, Sequence, Tuple

from evolve.ids import canonical_json, content_hash, content_id, derive_seed, validate_id
from evolve.types import LearningGroup, Role, RoleSnapshot

from .adapters import (
    ROLE_STATE_SCHEMA_VERSION,
    RoleAdapterState,
    RoleIsolationError,
    RoleOptimizerState,
    RoleRNGState,
    coerce_role,
    require_exact_keys,
    require_id,
    require_nonempty,
    require_nonnegative_int,
    require_schema_version,
)
from .working_memory import (
    RoleLearningOwnership,
    RoleRetrievalView,
    RoleWorkingTranscript,
)


ROLE_REGISTRY_SCHEMA_VERSION = 1
ROLE_REGISTRY_VERSION = "role_registry_v1"
ROLE_POLICY_VERSION = "isolated_role_policy_v1"
PRODUCTION_ROLES = (Role.SCOUT, Role.MECHANIST, Role.CHALLENGER)
_SHA256_RE = re.compile(r"^[0-9a-f]{64}$")


def _sha256(value: Any, name: str) -> str:
    if not isinstance(value, str) or _SHA256_RE.fullmatch(value) is None:
        raise RoleIsolationError(f"{name} must be a lowercase SHA-256 digest")
    return value


def _replace_role(
    states: Tuple["RoleRuntimeState", ...], updated: "RoleRuntimeState"
) -> Tuple["RoleRuntimeState", ...]:
    return tuple(updated if state.role == updated.role else state for state in states)


def _snapshot_identity_fields(
    *,
    run_id: str,
    epoch: int,
    role: Role,
    adapter_id: str,
    adapter_version: str,
    adapter_hash: str,
    optimizer_state_id: str,
    policy_version: str,
    rng_seed: int,
) -> Mapping[str, Any]:
    return {
        "snapshot_version": "role_snapshot_identity_v1",
        "run_id": run_id,
        "epoch": epoch,
        "role": role.value,
        "adapter_id": adapter_id,
        "adapter_version": adapter_version,
        "adapter_hash": adapter_hash,
        "optimizer_state_id": optimizer_state_id,
        "policy_version": policy_version,
        "rng_seed": rng_seed,
        "frozen": True,
    }


def validate_role_snapshot_identity(snapshot: RoleSnapshot) -> None:
    expected = content_id(
        "role_snapshot",
        _snapshot_identity_fields(
            run_id=snapshot.run_id,
            epoch=snapshot.epoch,
            role=snapshot.role,
            adapter_id=snapshot.adapter_id,
            adapter_version=snapshot.adapter_version,
            adapter_hash=snapshot.adapter_hash,
            optimizer_state_id=snapshot.optimizer_state_id,
            policy_version=snapshot.policy_version,
            rng_seed=snapshot.rng_seed,
        ),
    )
    if snapshot.snapshot_id != expected:
        raise RoleIsolationError("role snapshot ID does not match its frozen policy content")


@dataclass(frozen=True)
class RoleRuntimeState:
    """All mutable-over-epochs state owned by exactly one logical role."""

    run_id: str
    role: Role
    adapter: RoleAdapterState
    optimizer: RoleOptimizerState
    rng: RoleRNGState
    transcript: RoleWorkingTranscript
    retrieval: RoleRetrievalView
    learning: RoleLearningOwnership
    policy_version: str = ROLE_POLICY_VERSION
    schema_version: int = ROLE_STATE_SCHEMA_VERSION

    def __post_init__(self) -> None:
        owner = coerce_role(self.role)
        object.__setattr__(self, "role", owner)
        require_id(self.run_id, "run", "run_id")
        require_schema_version(
            self.schema_version, ROLE_STATE_SCHEMA_VERSION, "role runtime"
        )
        require_nonempty(self.policy_version, "policy_version")
        components = (
            self.adapter,
            self.optimizer,
            self.rng,
            self.transcript,
            self.retrieval,
            self.learning,
        )
        for component in components:
            if component.run_id != self.run_id or component.role != owner:
                raise RoleIsolationError(
                    f"{owner.value} runtime contains another role or run's state"
                )
        if self.optimizer.parameter_set_id != self.adapter.parameter_set_id:
            raise RoleIsolationError("optimizer is not bound to its role adapter parameters")
        if self.optimizer.bound_adapter_version != self.adapter.version:
            raise RoleIsolationError(
                "optimizer state is not bound to the current adapter version"
            )

    @classmethod
    def create(
        cls,
        *,
        run_id: str,
        role: Any,
        adapter_state: Mapping[str, Any],
        optimizer_state: Mapping[str, Any],
        rng_seed: int,
        policy_version: str = ROLE_POLICY_VERSION,
    ) -> "RoleRuntimeState":
        owner = coerce_role(role)
        adapter = RoleAdapterState.create(
            run_id=run_id,
            role=owner,
            state=adapter_state,
        )
        optimizer = RoleOptimizerState.create(adapter=adapter, state=optimizer_state)
        return cls(
            run_id=run_id,
            role=owner,
            adapter=adapter,
            optimizer=optimizer,
            rng=RoleRNGState.create(
                run_id=run_id,
                role=owner,
                base_seed=rng_seed,
            ),
            transcript=RoleWorkingTranscript.create(run_id=run_id, role=owner),
            retrieval=RoleRetrievalView.create(run_id=run_id, role=owner),
            learning=RoleLearningOwnership.create(run_id=run_id, role=owner),
            policy_version=policy_version,
        )

    def freeze(self, epoch: int) -> RoleSnapshot:
        require_nonnegative_int(epoch, "epoch")
        rng_seed = self.rng.epoch_seed(epoch)
        identity = _snapshot_identity_fields(
            run_id=self.run_id,
            epoch=epoch,
            role=self.role,
            adapter_id=self.adapter.adapter_id,
            adapter_version=self.adapter.version,
            adapter_hash=self.adapter.adapter_hash,
            optimizer_state_id=self.optimizer.state_id,
            policy_version=self.policy_version,
            rng_seed=rng_seed,
        )
        snapshot = RoleSnapshot(
            snapshot_id=content_id("role_snapshot", identity),
            run_id=self.run_id,
            epoch=epoch,
            role=self.role,
            adapter_id=self.adapter.adapter_id,
            adapter_version=self.adapter.version,
            adapter_hash=self.adapter.adapter_hash,
            optimizer_state_id=self.optimizer.state_id,
            policy_version=self.policy_version,
            rng_seed=rng_seed,
        )
        validate_role_snapshot_identity(snapshot)
        return snapshot

    def advance_policy(
        self,
        *,
        adapter_state: Mapping[str, Any],
        optimizer_state: Mapping[str, Any],
    ) -> "RoleRuntimeState":
        adapter = self.adapter.advance(adapter_state)
        optimizer = self.optimizer.advance(adapter=adapter, state=optimizer_state)
        return replace(self, adapter=adapter, optimizer=optimizer)

    def to_dict(self) -> Mapping[str, Any]:
        return {
            "record_type": "role_runtime_state",
            "schema_version": self.schema_version,
            "run_id": self.run_id,
            "role": self.role.value,
            "policy_version": self.policy_version,
            "adapter": self.adapter.to_dict(),
            "optimizer": self.optimizer.to_dict(),
            "rng": self.rng.to_dict(),
            "transcript": self.transcript.to_dict(),
            "retrieval": self.retrieval.to_dict(),
            "learning": self.learning.to_dict(),
        }

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> "RoleRuntimeState":
        keys = (
            "record_type", "schema_version", "run_id", "role", "policy_version",
            "adapter", "optimizer", "rng", "transcript", "retrieval", "learning",
        )
        require_exact_keys(payload, keys, "role runtime state")
        if payload["record_type"] != "role_runtime_state":
            raise RoleIsolationError("invalid role runtime record_type")
        return cls(
            schema_version=payload["schema_version"],
            run_id=payload["run_id"],
            role=payload["role"],
            policy_version=payload["policy_version"],
            adapter=RoleAdapterState.from_dict(payload["adapter"]),
            optimizer=RoleOptimizerState.from_dict(payload["optimizer"]),
            rng=RoleRNGState.from_dict(payload["rng"]),
            transcript=RoleWorkingTranscript.from_dict(payload["transcript"]),
            retrieval=RoleRetrievalView.from_dict(payload["retrieval"]),
            learning=RoleLearningOwnership.from_dict(payload["learning"]),
        )


@dataclass(frozen=True)
class RoleRegistry:
    """One frozen backbone and isolated runtime state for every enabled role."""

    run_id: str
    backbone_id: str
    backbone_version: str
    backbone_hash: str
    states: Tuple[RoleRuntimeState, ...]
    method_complete: bool = True
    backbone_frozen: bool = True
    registry_version: str = ROLE_REGISTRY_VERSION
    schema_version: int = ROLE_REGISTRY_SCHEMA_VERSION

    @classmethod
    def create_production(
        cls,
        *,
        run_id: str,
        backbone_id: str,
        backbone_version: str,
        backbone_hash: str,
        base_seed: int,
        initial_adapter_states: Optional[Mapping[str, Mapping[str, Any]]] = None,
        initial_optimizer_states: Optional[Mapping[str, Mapping[str, Any]]] = None,
        policy_version: str = ROLE_POLICY_VERSION,
    ) -> "RoleRegistry":
        """Create the only method-complete registry: all three EVOLVE roles."""

        return cls._create(
            run_id=run_id,
            backbone_id=backbone_id,
            backbone_version=backbone_version,
            backbone_hash=backbone_hash,
            base_seed=base_seed,
            roles=PRODUCTION_ROLES,
            initial_adapter_states=initial_adapter_states,
            initial_optimizer_states=initial_optimizer_states,
            policy_version=policy_version,
            method_complete=True,
        )

    @classmethod
    def create_test_fixture(
        cls,
        *,
        run_id: str,
        backbone_id: str,
        backbone_version: str,
        backbone_hash: str,
        base_seed: int,
        roles: Sequence[Any],
        initial_adapter_states: Optional[Mapping[str, Mapping[str, Any]]] = None,
        initial_optimizer_states: Optional[Mapping[str, Mapping[str, Any]]] = None,
        policy_version: str = ROLE_POLICY_VERSION,
    ) -> "RoleRegistry":
        """Create an explicitly method-incomplete role subset for CPU tests."""

        enabled = tuple(coerce_role(role) for role in roles)
        return cls._create(
            run_id=run_id,
            backbone_id=backbone_id,
            backbone_version=backbone_version,
            backbone_hash=backbone_hash,
            base_seed=base_seed,
            roles=enabled,
            initial_adapter_states=initial_adapter_states,
            initial_optimizer_states=initial_optimizer_states,
            policy_version=policy_version,
            method_complete=False,
        )

    @classmethod
    def _create(
        cls,
        *,
        run_id: str,
        backbone_id: str,
        backbone_version: str,
        backbone_hash: str,
        base_seed: int,
        roles: Sequence[Role],
        initial_adapter_states: Optional[Mapping[str, Mapping[str, Any]]],
        initial_optimizer_states: Optional[Mapping[str, Mapping[str, Any]]],
        policy_version: str,
        method_complete: bool,
    ) -> "RoleRegistry":
        require_id(run_id, "run", "run_id")
        require_id(backbone_id, "backbone", "backbone_id")
        require_nonempty(backbone_version, "backbone_version")
        _sha256(backbone_hash, "backbone_hash")
        require_nonnegative_int(base_seed, "base_seed")
        adapter_values = dict(initial_adapter_states or {})
        optimizer_values = dict(initial_optimizer_states or {})
        expected_names = {role.value for role in roles}
        unknown_adapter_roles = set(adapter_values) - expected_names
        unknown_optimizer_roles = set(optimizer_values) - expected_names
        if unknown_adapter_roles or unknown_optimizer_roles:
            raise RoleIsolationError(
                "initial role state includes disabled/unknown role(s): "
                f"adapters={sorted(unknown_adapter_roles)}, "
                f"optimizers={sorted(unknown_optimizer_roles)}"
            )
        states = []
        for role in roles:
            role_seed = derive_seed(
                "role_registry_rng",
                run_id,
                backbone_id,
                role.value,
                base_seed=base_seed,
            )
            states.append(
                RoleRuntimeState.create(
                    run_id=run_id,
                    role=role,
                    adapter_state=adapter_values.get(role.value, {}),
                    optimizer_state=optimizer_values.get(role.value, {}),
                    rng_seed=role_seed,
                    policy_version=policy_version,
                )
            )
        return cls(
            run_id=run_id,
            backbone_id=backbone_id,
            backbone_version=backbone_version,
            backbone_hash=backbone_hash,
            states=tuple(states),
            method_complete=method_complete,
        )

    def __post_init__(self) -> None:
        require_id(self.run_id, "run", "run_id")
        require_id(self.backbone_id, "backbone", "backbone_id")
        require_nonempty(self.backbone_version, "backbone_version")
        _sha256(self.backbone_hash, "backbone_hash")
        require_schema_version(
            self.schema_version, ROLE_REGISTRY_SCHEMA_VERSION, "role registry"
        )
        if self.registry_version != ROLE_REGISTRY_VERSION:
            raise RoleIsolationError(
                f"unsupported role registry version {self.registry_version!r}"
            )
        if self.backbone_frozen is not True:
            raise RoleIsolationError("the shared backbone must remain frozen")
        if not isinstance(self.method_complete, bool):
            raise RoleIsolationError("method_complete must be boolean")
        states = tuple(self.states)
        if not states:
            raise RoleIsolationError("a role registry must contain at least one role")
        roles = tuple(state.role for state in states)
        if len(set(roles)) != len(roles):
            raise RoleIsolationError("role registry contains duplicate role state")
        if self.method_complete:
            if roles != PRODUCTION_ROLES:
                raise RoleIsolationError(
                    "production role registry requires exactly scout, mechanist, challenger"
                )
        else:
            ordered = tuple(role for role in PRODUCTION_ROLES if role in set(roles))
            if roles != ordered:
                raise RoleIsolationError(
                    "test-only role subsets must use canonical production role order"
                )
        for state in states:
            if state.run_id != self.run_id:
                raise RoleIsolationError("role runtime belongs to another run")
        object.__setattr__(self, "states", states)
        ownership_fields = {
            "adapter IDs": [state.adapter.adapter_id for state in states],
            "parameter sets": [state.adapter.parameter_set_id for state in states],
            "adapter hashes": [state.adapter.adapter_hash for state in states],
            "optimizer owners": [state.optimizer.owner_id for state in states],
            "optimizer states": [state.optimizer.state_id for state in states],
            "RNG owners": [state.rng.owner_id for state in states],
            "transcript owners": [state.transcript.owner_id for state in states],
            "retrieval owners": [state.retrieval.owner_id for state in states],
            "learning owners": [state.learning.owner_id for state in states],
        }
        for label, values in ownership_fields.items():
            if len(set(values)) != len(values):
                raise RoleIsolationError(f"role {label} alias across roles")
        claimed_groups = [
            group_id for state in states for group_id in state.learning.group_ids
        ]
        if len(set(claimed_groups)) != len(claimed_groups):
            raise RoleIsolationError("a learning group is owned by multiple roles")

    @property
    def roles(self) -> Tuple[Role, ...]:
        return tuple(state.role for state in self.states)

    def state(self, role: Any) -> RoleRuntimeState:
        owner = coerce_role(role)
        for state in self.states:
            if state.role == owner:
                return state
        raise RoleIsolationError(f"role {owner.value!r} is not enabled")

    def freeze_epoch(self, epoch: int) -> Mapping[Role, RoleSnapshot]:
        require_nonnegative_int(epoch, "epoch")
        snapshots = {state.role: state.freeze(epoch) for state in self.states}
        if len({snapshot.snapshot_id for snapshot in snapshots.values()}) != len(snapshots):
            raise RoleIsolationError("role snapshot IDs alias")
        return MappingProxyType(snapshots)

    def advance_role(
        self,
        role: Any,
        *,
        adapter_state: Mapping[str, Any],
        optimizer_state: Mapping[str, Any],
    ) -> "RoleRegistry":
        current = self.state(role)
        updated = current.advance_policy(
            adapter_state=adapter_state,
            optimizer_state=optimizer_state,
        )
        return replace(self, states=_replace_role(self.states, updated))

    def draw_role_seed(self, role: Any, purpose: str) -> Tuple[int, "RoleRegistry"]:
        current = self.state(role)
        value, rng = current.rng.draw_seed(purpose)
        updated = replace(current, rng=rng)
        return value, replace(self, states=_replace_role(self.states, updated))

    def start_transcript(self, role: Any, branch_id: str) -> "RoleRegistry":
        current = self.state(role)
        updated = replace(current, transcript=current.transcript.start_branch(branch_id))
        return replace(self, states=_replace_role(self.states, updated))

    def append_transcript(
        self,
        role: Any,
        *,
        kind: str,
        content: Mapping[str, Any],
    ) -> "RoleRegistry":
        current = self.state(role)
        updated = replace(
            current,
            transcript=current.transcript.append(kind=kind, content=content),
        )
        return replace(self, states=_replace_role(self.states, updated))

    def clear_transcript(self, role: Any) -> "RoleRegistry":
        current = self.state(role)
        updated = replace(current, transcript=current.transcript.clear())
        return replace(self, states=_replace_role(self.states, updated))

    def advance_retrieval_view(
        self,
        role: Any,
        *,
        memory_snapshot_id: Optional[str],
        memory_ids: Sequence[str],
        scope: Mapping[str, Any],
    ) -> "RoleRegistry":
        current = self.state(role)
        retrieval = current.retrieval.advance(
            memory_snapshot_id=memory_snapshot_id,
            memory_ids=memory_ids,
            scope=scope,
        )
        updated = replace(current, retrieval=retrieval)
        return replace(self, states=_replace_role(self.states, updated))

    def claim_learning_group(
        self,
        role: Any,
        *,
        group: LearningGroup,
        snapshot: RoleSnapshot,
    ) -> "RoleRegistry":
        current = self.state(role)
        expected_snapshot = current.freeze(snapshot.epoch)
        if expected_snapshot != snapshot:
            raise RoleIsolationError(
                "learning group snapshot is stale or belongs to different role state"
            )
        learning = current.learning.claim(group=group, snapshot=snapshot)
        for state in self.states:
            if state.role != current.role and group.group_id in state.learning.group_ids:
                raise RoleIsolationError("learning group is already owned by another role")
        updated = replace(current, learning=learning)
        return replace(self, states=_replace_role(self.states, updated))

    def checkpoint_body(self) -> Mapping[str, Any]:
        """Return the full role state expected inside a barrier checkpoint."""

        return {
            "record_type": "role_registry_checkpoint",
            "schema_version": self.schema_version,
            "registry_version": self.registry_version,
            "run_id": self.run_id,
            "backbone_id": self.backbone_id,
            "backbone_version": self.backbone_version,
            "backbone_hash": self.backbone_hash,
            "backbone_frozen": self.backbone_frozen,
            "method_complete": self.method_complete,
            "states": [state.to_dict() for state in self.states],
        }

    def checkpoint_payload(self) -> Mapping[str, Any]:
        body = self.checkpoint_body()
        return {**body, "checkpoint_hash": content_hash(body)}

    def to_checkpoint_json(self) -> str:
        return canonical_json(self.checkpoint_payload())

    @classmethod
    def from_checkpoint_payload(cls, payload: Mapping[str, Any]) -> "RoleRegistry":
        keys = (
            "record_type", "schema_version", "registry_version", "run_id",
            "backbone_id", "backbone_version", "backbone_hash", "backbone_frozen",
            "method_complete", "states", "checkpoint_hash",
        )
        require_exact_keys(payload, keys, "role registry checkpoint")
        if payload["record_type"] != "role_registry_checkpoint":
            raise RoleIsolationError("invalid role registry checkpoint record_type")
        checkpoint_hash = payload["checkpoint_hash"]
        _sha256(checkpoint_hash, "checkpoint_hash")
        body = {key: payload[key] for key in keys if key != "checkpoint_hash"}
        if content_hash(body) != checkpoint_hash:
            raise RoleIsolationError("role registry checkpoint hash mismatch")
        raw_states = payload["states"]
        if not isinstance(raw_states, (list, tuple)):
            raise RoleIsolationError("role registry states must be a list")
        return cls(
            schema_version=payload["schema_version"],
            registry_version=payload["registry_version"],
            run_id=payload["run_id"],
            backbone_id=payload["backbone_id"],
            backbone_version=payload["backbone_version"],
            backbone_hash=payload["backbone_hash"],
            backbone_frozen=payload["backbone_frozen"],
            method_complete=payload["method_complete"],
            states=tuple(RoleRuntimeState.from_dict(item) for item in raw_states),
        )

    @classmethod
    def from_checkpoint_json(cls, text: str) -> "RoleRegistry":
        if not isinstance(text, str):
            raise RoleIsolationError("role registry checkpoint JSON must be text")

        def reject_duplicate_keys(pairs):
            value = {}
            for key, item in pairs:
                if key in value:
                    raise RoleIsolationError(
                        f"role registry checkpoint contains duplicate key {key!r}"
                    )
                value[key] = item
            return value

        try:
            payload = json.loads(text, object_pairs_hook=reject_duplicate_keys)
        except RoleIsolationError:
            raise
        except (TypeError, ValueError, json.JSONDecodeError) as exc:
            raise RoleIsolationError(f"invalid role registry checkpoint JSON: {exc}") from exc
        return cls.from_checkpoint_payload(payload)


__all__ = [
    "PRODUCTION_ROLES",
    "ROLE_POLICY_VERSION",
    "ROLE_REGISTRY_SCHEMA_VERSION",
    "ROLE_REGISTRY_VERSION",
    "RoleRegistry",
    "RoleRuntimeState",
    "validate_role_snapshot_identity",
]
