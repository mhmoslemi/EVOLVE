"""Capability-checked named-adapter port for the EVOLVE role ecology.

The loader initially exposes one temporary default PEFT adapter. EVOLVE
replaces it with a stricter boundary: one explicitly identified frozen backbone with
exactly three named adapters whose parameters can never be optimized together.
This module supplies that boundary without importing torch, transformers,
PEFT or Transformers. Runtime integrations and CPU fakes
used by the tests satisfy the same small duck-typed protocol.

The port is deliberately limited to an already resolved PEFT-capable ``hf`` or
``unsloth`` backend. Backend resolution is configuration state and is persisted
before this capability is constructed; no automatic backend conversion occurs.
"""

from __future__ import annotations

import contextlib
import hashlib
import json
import os
import shutil
import tempfile
import threading
from dataclasses import dataclass
from pathlib import Path, PurePosixPath
from typing import Any, Dict, Iterator, Mapping, Optional, Sequence, Tuple, Union

from evolve.ids import canonical_json, content_hash, content_id
from evolve.runio.atomic import fsync_directory
from evolve.types import FrozenDict, Role

from .adapters import (
    RoleAdapterState,
    RoleIsolationError,
    coerce_role,
    require_id,
    require_nonempty,
)


ROLE_BACKEND_VERSION = "named_peft_role_backend_v1"
ADAPTER_ARTIFACT_SCHEMA_VERSION = 1
ADAPTER_ARTIFACT_VERSION = "named_adapter_artifact_v1"
ADAPTER_MANIFEST_NAME = "evolve_adapter_manifest.json"
ROLE_ADAPTER_NAMES = FrozenDict(
    {
        Role.SCOUT.value: "evolve_scout",
        Role.MECHANIST.value: "evolve_mechanist",
        Role.CHALLENGER.value: "evolve_challenger",
    }
)
PRODUCTION_ROLES = (Role.SCOUT, Role.MECHANIST, Role.CHALLENGER)


class RoleBackendError(RoleIsolationError):
    """The named-adapter backend violated an EVOLVE ownership invariant."""


class RoleBackendCapabilityError(RoleBackendError):
    """The resolved backend cannot implement exact named-adapter isolation."""


class AdapterArtifactError(RoleBackendError):
    """A persisted adapter artifact is mutable, mismatched, or corrupt."""


def _require_sha256(value: Any, name: str) -> str:
    if (
        not isinstance(value, str)
        or len(value) != 64
        or any(char not in "0123456789abcdef" for char in value)
    ):
        raise RoleBackendError(f"{name} must be a lowercase SHA-256 digest")
    return value


def _strict_json_object(path: Path) -> Mapping[str, Any]:
    def pairs_hook(pairs: Sequence[Tuple[str, Any]]) -> Dict[str, Any]:
        result: Dict[str, Any] = {}
        for key, value in pairs:
            if key in result:
                raise AdapterArtifactError(
                    f"duplicate JSON key {key!r} in adapter manifest"
                )
            result[key] = value
        return result

    try:
        payload = json.loads(
            path.read_text(encoding="utf-8"),
            object_pairs_hook=pairs_hook,
            parse_constant=lambda value: (_ for _ in ()).throw(
                AdapterArtifactError(
                    f"non-finite JSON constant {value!r} in adapter manifest"
                )
            ),
        )
    except AdapterArtifactError:
        raise
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        raise AdapterArtifactError(f"cannot read adapter manifest {path}: {exc}") from exc
    if not isinstance(payload, Mapping):
        raise AdapterArtifactError("adapter manifest must be a JSON object")
    return payload


@dataclass(frozen=True)
class BackboneIdentity:
    """Persistable identity of the one frozen backbone shared by all roles."""

    backbone_id: str
    model_name: str
    revision: str
    weights_hash: str
    config_hash: str
    identity_version: str = ROLE_BACKEND_VERSION

    @classmethod
    def create(
        cls,
        *,
        model_name: str,
        revision: str,
        weights_hash: str,
        config_hash: str,
    ) -> "BackboneIdentity":
        payload = {
            "identity_version": ROLE_BACKEND_VERSION,
            "model_name": require_nonempty(model_name, "model_name"),
            "revision": require_nonempty(revision, "backbone revision"),
            "weights_hash": _require_sha256(weights_hash, "weights_hash"),
            "config_hash": _require_sha256(config_hash, "config_hash"),
        }
        return cls(backbone_id=content_id("backbone", payload), **payload)

    def __post_init__(self) -> None:
        require_id(self.backbone_id, "backbone", "backbone_id")
        require_nonempty(self.model_name, "model_name")
        require_nonempty(self.revision, "backbone revision")
        _require_sha256(self.weights_hash, "weights_hash")
        _require_sha256(self.config_hash, "config_hash")
        if self.identity_version != ROLE_BACKEND_VERSION:
            raise RoleBackendError(
                f"unsupported backbone identity version {self.identity_version!r}"
            )
        expected = content_id(
            "backbone",
            {
                "identity_version": self.identity_version,
                "model_name": self.model_name,
                "revision": self.revision,
                "weights_hash": self.weights_hash,
                "config_hash": self.config_hash,
            },
        )
        if self.backbone_id != expected:
            raise RoleBackendError("backbone_id does not match frozen backbone identity")


@dataclass(frozen=True)
class RoleParameterManifest:
    """Stable parameter-name ownership for one named role adapter."""

    role: Role
    adapter_name: str
    parameter_names: Tuple[str, ...]
    manifest_hash: str

    @classmethod
    def create(
        cls, *, role: Any, adapter_name: str, parameter_names: Sequence[str]
    ) -> "RoleParameterManifest":
        owner = coerce_role(role)
        name = require_nonempty(adapter_name, "adapter_name")
        names = tuple(sorted(parameter_names))
        if not names or any(not isinstance(item, str) or not item for item in names):
            raise RoleBackendError(f"{owner.value} adapter parameter manifest is empty")
        if len(set(names)) != len(names):
            raise RoleBackendError(
                f"{owner.value} adapter parameter manifest contains duplicates"
            )
        digest = content_hash(
            {
                "version": ROLE_BACKEND_VERSION,
                "role": owner.value,
                "adapter_name": name,
                "parameter_names": list(names),
            }
        )
        return cls(owner, name, names, digest)

    def __post_init__(self) -> None:
        owner = coerce_role(self.role)
        object.__setattr__(self, "role", owner)
        require_nonempty(self.adapter_name, "adapter_name")
        names = tuple(sorted(self.parameter_names))
        if not names or any(not isinstance(item, str) or not item for item in names):
            raise RoleBackendError(f"{owner.value} adapter parameter manifest is empty")
        if len(set(names)) != len(names):
            raise RoleBackendError(
                f"{owner.value} adapter parameter manifest contains duplicates"
            )
        if tuple(self.parameter_names) != names:
            raise RoleBackendError("parameter manifest names must be sorted")
        expected_hash = content_hash(
            {
                "version": ROLE_BACKEND_VERSION,
                "role": owner.value,
                "adapter_name": self.adapter_name,
                "parameter_names": list(names),
            }
        )
        if self.manifest_hash != expected_hash:
            raise RoleBackendError("parameter manifest hash does not match its names")


@dataclass(frozen=True)
class RoleBackendBinding:
    """Read-only dispatch binding produced after runtime capability validation."""

    role: Role
    backend_name: str
    backend_version: str
    backbone_id: str
    adapter_name: str
    parameter_manifest_hash: str


@dataclass(frozen=True)
class AdapterArtifact:
    """Content-checked immutable on-disk representation of one role adapter."""

    role: Role
    adapter_id: str
    adapter_version: str
    logical_adapter_hash: str
    backbone_id: str
    backend_version: str
    backend_adapter_name: str
    parameter_manifest_hash: str
    adapter_relative_path: str
    files: FrozenDict
    artifact_hash: str
    artifact_version: str = ADAPTER_ARTIFACT_VERSION
    schema_version: int = ADAPTER_ARTIFACT_SCHEMA_VERSION

    @classmethod
    def create(
        cls,
        *,
        role: Any,
        adapter_id: str,
        adapter_version: str,
        logical_adapter_hash: str,
        backbone_id: str,
        backend_adapter_name: str,
        parameter_manifest_hash: str,
        adapter_relative_path: str,
        files: Mapping[str, str],
    ) -> "AdapterArtifact":
        owner = coerce_role(role)
        frozen_files = FrozenDict(dict(sorted(files.items())))
        identity = {
            "artifact_version": ADAPTER_ARTIFACT_VERSION,
            "schema_version": ADAPTER_ARTIFACT_SCHEMA_VERSION,
            "role": owner.value,
            "adapter_id": adapter_id,
            "adapter_version": adapter_version,
            "logical_adapter_hash": logical_adapter_hash,
            "backbone_id": backbone_id,
            "backend_version": ROLE_BACKEND_VERSION,
            "backend_adapter_name": backend_adapter_name,
            "parameter_manifest_hash": parameter_manifest_hash,
            "adapter_relative_path": adapter_relative_path,
            "files": frozen_files,
        }
        return cls(
            role=owner,
            adapter_id=adapter_id,
            adapter_version=adapter_version,
            logical_adapter_hash=logical_adapter_hash,
            backbone_id=backbone_id,
            backend_version=ROLE_BACKEND_VERSION,
            backend_adapter_name=backend_adapter_name,
            parameter_manifest_hash=parameter_manifest_hash,
            adapter_relative_path=adapter_relative_path,
            files=frozen_files,
            artifact_hash=content_hash(identity),
        )

    def __post_init__(self) -> None:
        owner = coerce_role(self.role)
        object.__setattr__(self, "role", owner)
        require_id(self.adapter_id, "adapter", "adapter_id")
        require_id(self.backbone_id, "backbone", "backbone_id")
        require_nonempty(self.adapter_version, "adapter_version")
        require_nonempty(self.backend_adapter_name, "backend_adapter_name")
        _require_sha256(self.logical_adapter_hash, "logical_adapter_hash")
        _require_sha256(self.parameter_manifest_hash, "parameter_manifest_hash")
        _require_sha256(self.artifact_hash, "artifact_hash")
        if self.schema_version != ADAPTER_ARTIFACT_SCHEMA_VERSION:
            raise AdapterArtifactError(
                f"unsupported adapter artifact schema {self.schema_version}"
            )
        if self.artifact_version != ADAPTER_ARTIFACT_VERSION:
            raise AdapterArtifactError(
                f"unsupported adapter artifact version {self.artifact_version!r}"
            )
        if self.backend_version != ROLE_BACKEND_VERSION:
            raise AdapterArtifactError(
                f"unsupported role backend version {self.backend_version!r}"
            )
        relative = PurePosixPath(self.adapter_relative_path)
        if relative.is_absolute() or ".." in relative.parts:
            raise AdapterArtifactError("adapter_relative_path must stay inside artifact")
        files = FrozenDict(dict(sorted(self.files.items())))
        if not files:
            raise AdapterArtifactError("adapter artifact must contain payload files")
        for name, digest in files.items():
            path = PurePosixPath(name)
            if path.is_absolute() or ".." in path.parts or name == ADAPTER_MANIFEST_NAME:
                raise AdapterArtifactError(f"unsafe adapter artifact file {name!r}")
            _require_sha256(digest, f"files[{name!r}]")
        object.__setattr__(self, "files", files)
        identity = {
            "artifact_version": self.artifact_version,
            "schema_version": self.schema_version,
            "role": owner.value,
            "adapter_id": self.adapter_id,
            "adapter_version": self.adapter_version,
            "logical_adapter_hash": self.logical_adapter_hash,
            "backbone_id": self.backbone_id,
            "backend_version": self.backend_version,
            "backend_adapter_name": self.backend_adapter_name,
            "parameter_manifest_hash": self.parameter_manifest_hash,
            "adapter_relative_path": self.adapter_relative_path,
            "files": files,
        }
        if self.artifact_hash != content_hash(identity):
            raise AdapterArtifactError("artifact_hash does not cover adapter manifest")

    def to_dict(self) -> Mapping[str, Any]:
        return {
            "record_type": "adapter_artifact",
            "schema_version": self.schema_version,
            "artifact_version": self.artifact_version,
            "role": self.role.value,
            "adapter_id": self.adapter_id,
            "adapter_version": self.adapter_version,
            "logical_adapter_hash": self.logical_adapter_hash,
            "backbone_id": self.backbone_id,
            "backend_version": self.backend_version,
            "backend_adapter_name": self.backend_adapter_name,
            "parameter_manifest_hash": self.parameter_manifest_hash,
            "adapter_relative_path": self.adapter_relative_path,
            "files": dict(self.files.items()),
            "artifact_hash": self.artifact_hash,
        }

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> "AdapterArtifact":
        expected = {
            "record_type", "schema_version", "artifact_version", "role",
            "adapter_id", "adapter_version", "logical_adapter_hash",
            "backbone_id", "backend_version", "backend_adapter_name",
            "parameter_manifest_hash", "adapter_relative_path", "files",
            "artifact_hash",
        }
        if set(payload) != expected:
            raise AdapterArtifactError(
                "adapter artifact manifest fields differ "
                f"(missing={sorted(expected - set(payload))}, "
                f"unknown={sorted(set(payload) - expected)})"
            )
        if payload["record_type"] != "adapter_artifact":
            raise AdapterArtifactError("invalid adapter artifact record_type")
        if not isinstance(payload["files"], Mapping):
            raise AdapterArtifactError("adapter artifact files must be a mapping")
        return cls(
            schema_version=payload["schema_version"],
            artifact_version=payload["artifact_version"],
            role=payload["role"],
            adapter_id=payload["adapter_id"],
            adapter_version=payload["adapter_version"],
            logical_adapter_hash=payload["logical_adapter_hash"],
            backbone_id=payload["backbone_id"],
            backend_version=payload["backend_version"],
            backend_adapter_name=payload["backend_adapter_name"],
            parameter_manifest_hash=payload["parameter_manifest_hash"],
            adapter_relative_path=payload["adapter_relative_path"],
            files=FrozenDict(payload["files"]),
            artifact_hash=payload["artifact_hash"],
        )


def _adapter_token_matches(parameter_name: str, adapter_name: str) -> bool:
    return adapter_name in parameter_name.split(".")


def _hash_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        while True:
            chunk = stream.read(1024 * 1024)
            if not chunk:
                break
            digest.update(chunk)
    return digest.hexdigest()


def _artifact_files(root: Path) -> Mapping[str, str]:
    files: Dict[str, str] = {}
    for path in sorted(root.rglob("*")):
        if path.is_symlink():
            raise AdapterArtifactError(f"adapter artifact contains symlink: {path}")
        if path.is_dir():
            continue
        if not path.is_file():
            raise AdapterArtifactError(f"adapter artifact contains non-file: {path}")
        relative = path.relative_to(root).as_posix()
        if relative == ADAPTER_MANIFEST_NAME:
            continue
        files[relative] = _hash_file(path)
    if not files:
        raise AdapterArtifactError("adapter serializer produced no payload files")
    return files


def _locate_adapter_payload(root: Path, adapter_name: str) -> str:
    candidates = []
    for relative in (Path("."), Path(adapter_name)):
        if (root / relative / "adapter_config.json").is_file():
            candidates.append(relative.as_posix())
    if len(candidates) != 1:
        raise AdapterArtifactError(
            "adapter artifact must contain exactly one root or named "
            "adapter_config.json payload"
        )
    return candidates[0]


def inspect_adapter_artifact(
    directory: Union[os.PathLike, str], *, expected_artifact_hash: str
) -> AdapterArtifact:
    """Read and rehash an artifact before it is allowed to touch a model."""

    _require_sha256(expected_artifact_hash, "expected_artifact_hash")
    root = Path(directory)
    if root.is_symlink() or not root.is_dir():
        raise AdapterArtifactError(f"adapter artifact is not a regular directory: {root}")
    manifest_path = root / ADAPTER_MANIFEST_NAME
    if manifest_path.is_symlink() or not manifest_path.is_file():
        raise AdapterArtifactError(f"adapter manifest is missing: {manifest_path}")
    artifact = AdapterArtifact.from_dict(_strict_json_object(manifest_path))
    if artifact.artifact_hash != expected_artifact_hash:
        raise AdapterArtifactError("adapter artifact hash differs from checkpoint binding")
    actual_files = _artifact_files(root)
    if actual_files != dict(artifact.files.items()):
        raise AdapterArtifactError("adapter artifact payload files changed after save")
    payload = root / artifact.adapter_relative_path / "adapter_config.json"
    if payload.is_symlink() or not payload.is_file():
        raise AdapterArtifactError("adapter payload path does not contain adapter_config.json")
    return artifact


class NamedAdapterBackendPort:
    """Exact three-role named-adapter capability over one loaded HF backend.

    Required backend methods are validated before the first adapter is added.
    The object is intentionally not thread-concurrent: a re-entrant lock keeps
    nested role/reference contexts correct and serializes activation on the one
    training backbone.
    """

    _BACKEND_METHODS = (
        "set_inference_mode",
        "set_training_mode",
        "disable_adapter",
    )
    _MODEL_METHODS = (
        "add_adapter",
        "delete_adapter",
        "set_adapter",
        "load_adapter",
        "save_pretrained",
        "named_parameters",
    )

    def __init__(
        self,
        backend: Any,
        *,
        backbone: BackboneIdentity,
        adapter_config: Any,
    ) -> None:
        self._validate_capabilities(backend, adapter_config)
        self._backend = backend
        self._model = backend.model
        self._model_object_id = id(self._model)
        self.backbone = backbone
        self._adapter_config = adapter_config
        self._lock = threading.RLock()
        self._context_depth = 0
        self._active_role: Optional[Role] = None
        self._parameters: Dict[str, Any] = {}
        self._owners: Dict[str, Optional[Role]] = {}
        self._manifests: Dict[Role, RoleParameterManifest] = {}
        self._install_exact_role_adapters()

    @classmethod
    def _validate_capabilities(cls, backend: Any, adapter_config: Any) -> None:
        # Resolution is external state. Never convert an unknown backend here.
        if getattr(backend, "name", None) not in {"hf", "unsloth"}:
            raise RoleBackendCapabilityError(
                "EVOLVE named adapters require an explicitly resolved HF or "
                "Unsloth PEFT backend"
            )
        model = getattr(backend, "model", None)
        if model is None:
            raise RoleBackendCapabilityError("HF backend must be loaded before role setup")
        missing = [name for name in cls._BACKEND_METHODS if not callable(getattr(backend, name, None))]
        missing += [name for name in cls._MODEL_METHODS if not callable(getattr(model, name, None))]
        if missing:
            raise RoleBackendCapabilityError(
                f"HF/PEFT named-adapter capability is missing: {sorted(missing)}"
            )
        if adapter_config is None:
            raise RoleBackendCapabilityError("adapter_config is required for all three roles")
        if not isinstance(getattr(model, "peft_config", None), Mapping):
            raise RoleBackendCapabilityError("model.peft_config must be a named mapping")
        if not isinstance(getattr(model, "training", None), bool):
            raise RoleBackendCapabilityError("model.training must expose its current mode")
        existing = set(model.peft_config)
        if existing not in (set(), {"default"}):
            raise RoleBackendCapabilityError(
                f"expected only the temporary default adapter before EVOLVE setup, got {sorted(existing)}"
            )
        try:
            pairs = list(model.named_parameters())
        except Exception as exc:
            raise RoleBackendCapabilityError(f"cannot enumerate model parameters: {exc}") from exc
        if not pairs or any(
            not isinstance(name, str) or not hasattr(parameter, "requires_grad")
            for name, parameter in pairs
        ):
            raise RoleBackendCapabilityError(
                "named_parameters must yield named objects with requires_grad"
            )

    @property
    def backend_name(self) -> str:
        return self._backend.name

    @property
    def backbone_id(self) -> str:
        return self.backbone.backbone_id

    @property
    def active_role(self) -> Optional[Role]:
        return self._active_role

    @property
    def parameter_manifests(self) -> Mapping[Role, RoleParameterManifest]:
        return dict(self._manifests)

    def _install_exact_role_adapters(self) -> None:
        for role in PRODUCTION_ROLES:
            self._model.add_adapter(ROLE_ADAPTER_NAMES[role.value], self._adapter_config)
        if "default" in self._model.peft_config:
            self._model.delete_adapter("default")
        expected = set(ROLE_ADAPTER_NAMES.values())
        if set(self._model.peft_config) != expected:
            raise RoleBackendCapabilityError(
                "backend did not install exactly scout, mechanist, and challenger adapters"
            )
        self._rebuild_parameter_ownership()
        self._active_role = Role.SCOUT
        self._model.set_adapter(ROLE_ADAPTER_NAMES[Role.SCOUT.value])
        self._set_trainability(None)
        self._backend.set_inference_mode()
        self.assert_isolation(active_role=None, training=False)

    def _rebuild_parameter_ownership(self) -> None:
        pairs = list(self._model.named_parameters())
        parameters: Dict[str, Any] = {}
        object_owners: Dict[int, Optional[Role]] = {}
        owners: Dict[str, Optional[Role]] = {}
        names_by_role: Dict[Role, list[str]] = {role: [] for role in PRODUCTION_ROLES}
        for name, parameter in pairs:
            if name in parameters:
                raise RoleBackendError(f"duplicate named parameter {name!r}")
            matched = [
                role
                for role in PRODUCTION_ROLES
                if _adapter_token_matches(name, ROLE_ADAPTER_NAMES[role.value])
            ]
            if len(matched) > 1:
                raise RoleBackendError(f"parameter {name!r} belongs to multiple roles")
            owner = matched[0] if matched else None
            prior_owner = object_owners.get(id(parameter), owner)
            if id(parameter) in object_owners and prior_owner != owner:
                raise RoleBackendError(
                    f"one parameter object aliases {prior_owner} and {owner}"
                )
            object_owners[id(parameter)] = owner
            parameters[name] = parameter
            owners[name] = owner
            if owner is not None:
                names_by_role[owner].append(name)
        manifests = {
            role: RoleParameterManifest.create(
                role=role,
                adapter_name=ROLE_ADAPTER_NAMES[role.value],
                parameter_names=names_by_role[role],
            )
            for role in PRODUCTION_ROLES
        }
        all_names = [name for manifest in manifests.values() for name in manifest.parameter_names]
        if len(all_names) != len(set(all_names)):
            raise RoleBackendError("role parameter manifests are not disjoint")
        self._parameters = parameters
        self._owners = owners
        self._manifests = manifests

    def _validate_runtime_shape(self) -> None:
        if self._backend.model is not self._model or id(self._model) != self._model_object_id:
            raise RoleBackendError("training backbone object changed after role installation")
        if set(self._model.peft_config) != set(ROLE_ADAPTER_NAMES.values()):
            raise RoleBackendError("named adapter registry changed after installation")
        current = dict(self._model.named_parameters())
        if set(current) != set(self._parameters):
            raise RoleBackendError("model parameter manifest changed after installation")
        for name, parameter in current.items():
            if parameter is not self._parameters[name]:
                raise RoleBackendError(f"parameter object changed without restore: {name}")

    def _set_trainability(self, role: Optional[Role]) -> None:
        for name, parameter in self._parameters.items():
            parameter.requires_grad = role is not None and self._owners[name] == role

    def assert_isolation(self, *, active_role: Optional[Any], training: bool) -> None:
        """Prove the base and inactive adapters are frozen in the current mode."""

        owner = None if active_role is None else coerce_role(active_role)
        self._validate_runtime_shape()
        if not isinstance(training, bool):
            raise RoleBackendError("training must be boolean")
        for name, parameter in self._parameters.items():
            expected = bool(training and owner is not None and self._owners[name] == owner)
            if bool(parameter.requires_grad) != expected:
                label = self._owners[name].value if self._owners[name] else "backbone"
                raise RoleBackendError(
                    f"{label} parameter {name!r} has requires_grad={parameter.requires_grad}; "
                    f"expected {expected}"
                )

    def assert_dispatch_ready(
        self,
        role: Any,
        *,
        expected_backbone_id: Optional[str] = None,
        expected_parameter_manifest_hash: Optional[str] = None,
    ) -> RoleBackendBinding:
        """Validate all immutable bindings immediately before a role dispatch."""

        owner = coerce_role(role)
        with self._lock:
            self._validate_runtime_shape()
            # Dispatch validation must catch accidental base/inactive unfreezing.
            for name, parameter in self._parameters.items():
                if bool(parameter.requires_grad):
                    raise RoleBackendError(
                        f"dispatch attempted while parameter {name!r} is trainable"
                    )
            if expected_backbone_id is not None and expected_backbone_id != self.backbone_id:
                raise RoleBackendError("dispatch backbone identity does not match registry")
            manifest = self._manifests[owner]
            if (
                expected_parameter_manifest_hash is not None
                and expected_parameter_manifest_hash != manifest.manifest_hash
            ):
                raise RoleBackendError("dispatch parameter manifest does not match role")
            return RoleBackendBinding(
                role=owner,
                backend_name=self._backend.name,
                backend_version=ROLE_BACKEND_VERSION,
                backbone_id=self.backbone_id,
                adapter_name=manifest.adapter_name,
                parameter_manifest_hash=manifest.manifest_hash,
            )

    def _restore_mode(self, training: bool) -> None:
        if training:
            self._backend.set_training_mode()
        else:
            self._backend.set_inference_mode()

    @contextlib.contextmanager
    def activate(self, role: Any, *, training: bool = False) -> Iterator[RoleBackendBinding]:
        """Activate one role and restore the exact prior role/mode on exit."""

        owner = coerce_role(role)
        if not isinstance(training, bool):
            raise RoleBackendError("training must be boolean")
        with self._lock:
            self._validate_runtime_shape()
            previous_role = self._active_role
            previous_mode = bool(self._model.training)
            previous_flags = {
                name: bool(parameter.requires_grad)
                for name, parameter in self._parameters.items()
            }
            self._context_depth += 1
            try:
                self._model.set_adapter(ROLE_ADAPTER_NAMES[owner.value])
                self._active_role = owner
                self._set_trainability(owner if training else None)
                self._restore_mode(training)
                self.assert_isolation(active_role=owner if training else None, training=training)
                yield RoleBackendBinding(
                    role=owner,
                    backend_name=self._backend.name,
                    backend_version=ROLE_BACKEND_VERSION,
                    backbone_id=self.backbone_id,
                    adapter_name=ROLE_ADAPTER_NAMES[owner.value],
                    parameter_manifest_hash=self._manifests[owner].manifest_hash,
                )
            finally:
                if previous_role is not None:
                    self._model.set_adapter(ROLE_ADAPTER_NAMES[previous_role.value])
                self._active_role = previous_role
                for name, flag in previous_flags.items():
                    self._parameters[name].requires_grad = flag
                self._restore_mode(previous_mode)
                self._context_depth -= 1

    @contextlib.contextmanager
    def reference_disabled(self) -> Iterator[None]:
        """Disable every adapter for an inference-only frozen-base reference pass."""

        with self._lock:
            self._validate_runtime_shape()
            previous_role = self._active_role
            previous_mode = bool(self._model.training)
            previous_flags = {
                name: bool(parameter.requires_grad)
                for name, parameter in self._parameters.items()
            }
            self._context_depth += 1
            try:
                self._set_trainability(None)
                self._backend.set_inference_mode()
                with self._backend.disable_adapter():
                    self.assert_isolation(active_role=None, training=False)
                    yield
            finally:
                if previous_role is not None:
                    self._model.set_adapter(ROLE_ADAPTER_NAMES[previous_role.value])
                self._active_role = previous_role
                for name, flag in previous_flags.items():
                    self._parameters[name].requires_grad = flag
                self._restore_mode(previous_mode)
                self._context_depth -= 1

    def _validate_adapter_state(self, state: RoleAdapterState, role: Role) -> None:
        if not isinstance(state, RoleAdapterState):
            raise RoleBackendError("adapter persistence requires RoleAdapterState")
        if state.role != role:
            raise RoleBackendError("adapter state belongs to a different role")

    def save_adapter(
        self,
        role: Any,
        *,
        state: RoleAdapterState,
        destination: Union[os.PathLike, str],
        companion_files: Optional[Mapping[str, bytes]] = None,
    ) -> AdapterArtifact:
        """Atomically create a never-overwritten role-training artifact.

        Optional companion files (normally the matching optimizer state) are
        written into the same staging directory before its file manifest is
        hashed and the directory is atomically published. This prevents a
        completed adapter from ever being paired with a recomputed optimizer
        after an interrupted barrier.
        """

        owner = coerce_role(role)
        self._validate_adapter_state(state, owner)
        destination_path = Path(destination)
        with self._lock:
            if self._context_depth:
                raise RoleBackendError("cannot save an adapter inside an activation context")
            self._validate_runtime_shape()
            if destination_path.exists() or destination_path.is_symlink():
                raise AdapterArtifactError(
                    f"immutable adapter destination already exists: {destination_path}"
                )
            destination_path.parent.mkdir(parents=True, exist_ok=True)
            staging = Path(
                tempfile.mkdtemp(
                    prefix=f".{destination_path.name}.tmp-",
                    dir=str(destination_path.parent),
                )
            )
            try:
                adapter_name = ROLE_ADAPTER_NAMES[owner.value]
                self._model.save_pretrained(
                    str(staging),
                    selected_adapters=[adapter_name],
                    safe_serialization=True,
                )
                for relative_name, payload in dict(companion_files or {}).items():
                    if not isinstance(relative_name, str) or not relative_name:
                        raise AdapterArtifactError(
                            "adapter companion filename must be non-empty"
                        )
                    relative_path = PurePosixPath(relative_name)
                    if (
                        relative_path.is_absolute()
                        or ".." in relative_path.parts
                        or relative_path.as_posix() == ADAPTER_MANIFEST_NAME
                    ):
                        raise AdapterArtifactError(
                            "adapter companion filename must stay inside the artifact"
                        )
                    if not isinstance(payload, bytes):
                        raise AdapterArtifactError(
                            "adapter companion payload must be bytes"
                        )
                    companion_path = staging.joinpath(*relative_path.parts)
                    if companion_path.exists():
                        raise AdapterArtifactError(
                            f"adapter companion collides with model payload: {relative_name}"
                        )
                    companion_path.parent.mkdir(parents=True, exist_ok=True)
                    companion_path.write_bytes(payload)
                relative = _locate_adapter_payload(staging, adapter_name)
                files = _artifact_files(staging)
                artifact = AdapterArtifact.create(
                    role=owner,
                    adapter_id=state.adapter_id,
                    adapter_version=state.version,
                    logical_adapter_hash=state.adapter_hash,
                    backbone_id=self.backbone_id,
                    backend_adapter_name=adapter_name,
                    parameter_manifest_hash=self._manifests[owner].manifest_hash,
                    adapter_relative_path=relative,
                    files=files,
                )
                (staging / ADAPTER_MANIFEST_NAME).write_text(
                    canonical_json(artifact.to_dict()) + "\n",
                    encoding="utf-8",
                )
                for file_path in sorted(
                    (path for path in staging.rglob("*") if path.is_file()),
                    key=lambda path: path.as_posix(),
                ):
                    descriptor = os.open(os.fspath(file_path), os.O_RDONLY)
                    try:
                        os.fsync(descriptor)
                    finally:
                        os.close(descriptor)
                for directory in sorted(
                    (path for path in staging.rglob("*") if path.is_dir()),
                    key=lambda path: len(path.parts),
                    reverse=True,
                ):
                    fsync_directory(directory)
                fsync_directory(staging)
                os.replace(staging, destination_path)
                fsync_directory(destination_path.parent)
                return artifact
            except Exception:
                if staging.exists():
                    shutil.rmtree(staging)
                raise

    def load_adapter(
        self,
        role: Any,
        *,
        state: RoleAdapterState,
        directory: Union[os.PathLike, str],
        expected_artifact_hash: str,
    ) -> AdapterArtifact:
        """Validate an artifact fully, then replace only its matching role adapter.

        Callers restore adapters before constructing optimizers: PEFT load may
        replace that role's parameter objects.  The port rebuilds and verifies
        all three disjoint parameter manifests before returning.
        """

        owner = coerce_role(role)
        self._validate_adapter_state(state, owner)
        artifact = inspect_adapter_artifact(
            directory, expected_artifact_hash=expected_artifact_hash
        )
        expected_name = ROLE_ADAPTER_NAMES[owner.value]
        expected_manifest = self._manifests[owner]
        if artifact.role != owner:
            raise AdapterArtifactError("cannot load another role's adapter artifact")
        if artifact.adapter_id != state.adapter_id:
            raise AdapterArtifactError("adapter artifact ID differs from role state")
        if artifact.adapter_version != state.version:
            raise AdapterArtifactError("adapter artifact version differs from role state")
        if artifact.logical_adapter_hash != state.adapter_hash:
            raise AdapterArtifactError("adapter artifact logical hash differs from role state")
        if artifact.backbone_id != self.backbone_id:
            raise AdapterArtifactError("adapter artifact belongs to another backbone")
        if artifact.backend_adapter_name != expected_name:
            raise AdapterArtifactError("adapter artifact backend name differs from role")
        if artifact.parameter_manifest_hash != expected_manifest.manifest_hash:
            raise AdapterArtifactError("adapter artifact parameter manifest differs from role")

        root = Path(directory)
        payload_path = root / artifact.adapter_relative_path
        with self._lock:
            if self._context_depth:
                raise RoleBackendError("cannot load an adapter inside an activation context")
            self._validate_runtime_shape()
            previous_role = self._active_role
            previous_mode = bool(self._model.training)
            fallback = next(role for role in PRODUCTION_ROLES if role != owner)
            self._model.set_adapter(ROLE_ADAPTER_NAMES[fallback.value])
            self._model.delete_adapter(expected_name)
            self._model.load_adapter(
                str(payload_path),
                adapter_name=expected_name,
                is_trainable=False,
            )
            if set(self._model.peft_config) != set(ROLE_ADAPTER_NAMES.values()):
                raise RoleBackendError("adapter restore changed the exact three-role registry")
            self._rebuild_parameter_ownership()
            restored = self._manifests[owner]
            if restored.manifest_hash != artifact.parameter_manifest_hash:
                raise AdapterArtifactError(
                    "restored adapter parameters differ from the persisted manifest"
                )
            self._active_role = previous_role
            if previous_role is not None:
                self._model.set_adapter(ROLE_ADAPTER_NAMES[previous_role.value])
            self._set_trainability(None)
            self._restore_mode(previous_mode)
            self.assert_isolation(active_role=None, training=False)
        return artifact


__all__ = [
    "ADAPTER_ARTIFACT_SCHEMA_VERSION",
    "ADAPTER_ARTIFACT_VERSION",
    "ADAPTER_MANIFEST_NAME",
    "ROLE_ADAPTER_NAMES",
    "ROLE_BACKEND_VERSION",
    "AdapterArtifact",
    "AdapterArtifactError",
    "BackboneIdentity",
    "NamedAdapterBackendPort",
    "RoleBackendBinding",
    "RoleBackendCapabilityError",
    "RoleBackendError",
    "RoleParameterManifest",
    "inspect_adapter_artifact",
]
