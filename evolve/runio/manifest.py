"""Immutable initial metadata and additive resume-version metadata."""

from __future__ import annotations

import json
import os
import re
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Mapping, Optional, Sequence, Union

from evolve.ids import canonical_bytes, content_hash, content_id, validate_id

from .atomic import ImmutableWriteError, write_immutable_json, write_immutable_text
from .schema import (
    CURRENT_CONFIG_SCHEMA_VERSION,
    CURRENT_MANIFEST_SCHEMA_VERSION,
    resolve_effective_run_metadata,
)

_SHA256_RE = re.compile(r"^[0-9a-f]{64}$")


class ManifestValidationError(ValueError):
    """Manifest inputs are incomplete or contradict EVOLVE identity."""


@dataclass(frozen=True)
class MetadataPaths:
    requested_config: Optional[Path]
    resolved_config: Path
    compatibility_config: Path
    manifest: Path
    command: Path
    environment: Path


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")


def _json_copy(value: Any) -> Any:
    try:
        return json.loads(
            json.dumps(value, ensure_ascii=False, allow_nan=False, sort_keys=True)
        )
    except (TypeError, ValueError) as exc:
        raise ManifestValidationError(f"metadata is not JSON-safe: {exc}") from exc


def _object_copy(name: str, value: Mapping[str, Any]) -> Dict[str, Any]:
    if not isinstance(value, Mapping):
        raise ManifestValidationError(f"{name} must be a mapping")
    copied = _json_copy(dict(value))
    if not isinstance(copied, dict):
        raise ManifestValidationError(f"{name} must serialize to an object")
    return copied


def _nonempty_object_copy(name: str, value: Mapping[str, Any]) -> Dict[str, Any]:
    copied = _object_copy(name, value)
    if not copied:
        raise ManifestValidationError(f"{name} must be a non-empty mapping")
    return copied


def _gpu_copy(gpus: Sequence[Any]) -> list:
    if isinstance(gpus, (str, bytes)) or not isinstance(gpus, Sequence):
        raise ManifestValidationError("gpus must be a JSON list of identity objects")
    copied = _json_copy(list(gpus))
    for index, gpu in enumerate(copied):
        if not isinstance(gpu, dict) or not gpu:
            raise ManifestValidationError(
                f"gpus[{index}] must be a non-empty identity mapping"
            )
    return copied


def _nonempty_timestamp(value: Any, field_name: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ManifestValidationError(f"{field_name} must be non-empty text")
    return value


def canonical_config_bytes(config: Mapping[str, Any]) -> bytes:
    copied = _object_copy("resolved_config", config)
    # The resolved document carries its own hash. Identity covers all other
    # fields; including the self field would produce a different, impossible
    # fixed-point digest and disagree with evolve.config.canonical_config_hash.
    copied.pop("config_hash", None)
    return canonical_bytes(copied)


def resolved_config_hash(config: Mapping[str, Any]) -> str:
    copied = _object_copy("resolved_config", config)
    copied.pop("config_hash", None)
    return content_hash(copied)


def _evolve_config(
    config: Mapping[str, Any],
    name: str,
    *,
    expected_hash: Optional[str] = None,
) -> Dict[str, Any]:
    copied = _object_copy(name, config)
    copied.setdefault("engine", "evolve")
    copied.setdefault("schema_version", CURRENT_CONFIG_SCHEMA_VERSION)
    if copied["engine"] != "evolve":
        raise ManifestValidationError(f"{name} engine must be 'evolve'")
    if (
        isinstance(copied["schema_version"], bool)
        or not isinstance(copied["schema_version"], int)
        or copied["schema_version"] != CURRENT_CONFIG_SCHEMA_VERSION
    ):
        raise ManifestValidationError(
            f"{name} schema_version must be {CURRENT_CONFIG_SCHEMA_VERSION}"
        )
    actual_hash = resolved_config_hash(copied)
    if expected_hash is not None and actual_hash != expected_hash:
        raise ManifestValidationError(
            f"{name} must be a canonical copy of the authoritative resolved config"
        )
    canonical_hash = expected_hash or actual_hash
    embedded_hash = copied.get("config_hash")
    if embedded_hash is not None and embedded_hash != canonical_hash:
        raise ManifestValidationError(
            f"{name} config_hash does not match its authoritative identity"
        )
    copied["config_hash"] = canonical_hash
    return copied


def _validated_resolved_config(
    config: Mapping[str, Any], root: Path
) -> Dict[str, Any]:
    """Reject any resolved document that could not be resumed verbatim.

    The import is intentionally local: ``evolve.config`` uses the read-only
    run-schema layer, while this writer is the higher-level publication
    boundary.  Validation must happen before the first immutable artifact is
    written so a partial/default-filled document can never become run identity.
    """

    from evolve.config import EvolveConfigError, validate_resolved_config_document

    copied = _object_copy("resolved_config", config)
    try:
        validate_resolved_config_document(copied, cwd=root)
    except EvolveConfigError as exc:
        raise ManifestValidationError(
            f"resolved_config is not a complete canonical EVOLVE config: {exc}"
        ) from exc
    return _evolve_config(copied, "resolved_config")


def _command_document(command: Sequence[str]) -> Dict[str, Any]:
    if isinstance(command, (str, bytes)) or not isinstance(command, Sequence):
        raise ManifestValidationError("command must be a sequence of argv strings")
    argv = list(command)
    if not argv:
        raise ManifestValidationError("command argv must not be empty")
    if not all(isinstance(item, str) for item in argv):
        raise ManifestValidationError("every command argument must be a string")
    return {"schema_version": 1, "argv": argv}


def _environment_document(
    environment: Mapping[str, Any],
    *,
    host: Mapping[str, Any],
    gpus: Sequence[Any],
    package_versions: Mapping[str, Any],
) -> Dict[str, Any]:
    host_document = _nonempty_object_copy("host", host)
    package_document = _nonempty_object_copy("package_versions", package_versions)
    gpu_document = _gpu_copy(gpus)
    document = {
        "schema_version": 1,
        "variables": _object_copy("environment", environment),
        "host": host_document,
        "gpus": gpu_document,
        "packages": package_document,
    }
    return document


def _manifest_document(
    run_dir: Path,
    resolved: Dict[str, Any],
    *,
    created_at: str,
    git_state: Mapping[str, Any],
    model: Mapping[str, Any],
    package_versions: Mapping[str, Any],
    host: Mapping[str, Any],
    gpus: Sequence[Any],
    worker_topology: Mapping[str, Any],
    seeds: Mapping[str, Any],
    versions: Mapping[str, Any],
    extra: Optional[Mapping[str, Any]],
    run_id: Optional[str] = None,
) -> Dict[str, Any]:
    effective_run_id = run_id or content_id("run", {"run_name": run_dir.name})
    try:
        validate_id(effective_run_id, "run")
    except (TypeError, ValueError) as exc:
        raise ManifestValidationError(f"invalid run_id: {exc}") from exc
    created_at = _nonempty_timestamp(created_at, "created_at")
    document = {
        "schema_version": CURRENT_MANIFEST_SCHEMA_VERSION,
        "engine": "evolve",
        "run_id": effective_run_id,
        "run_name": run_dir.name,
        "created_at": created_at,
        "config_schema_version": resolved["schema_version"],
        "config_hash": resolved_config_hash(resolved),
        "git": _nonempty_object_copy("git_state", git_state),
        "model": _nonempty_object_copy("model", model),
        "packages": _nonempty_object_copy("package_versions", package_versions),
        "host": _nonempty_object_copy("host", host),
        "gpus": _gpu_copy(gpus),
        "worker_topology": _nonempty_object_copy("worker_topology", worker_topology),
        "seeds": _nonempty_object_copy("seeds", seeds),
        "versions": _nonempty_object_copy("versions", versions),
    }
    if extra:
        additions = _object_copy("manifest_extra", extra)
        overlap = sorted(set(document).intersection(additions))
        if overlap:
            raise ManifestValidationError(
                f"manifest_extra cannot replace reserved keys: {', '.join(overlap)}"
            )
        document.update(additions)
    return document


def _preflight_immutable(paths: Sequence[Path]) -> None:
    existing = [path for path in paths if path.exists()]
    if existing:
        raise ImmutableWriteError(
            "immutable run metadata already exists: "
            + ", ".join(str(path) for path in existing)
        )


def _resume_paths(root: Path, resume_index: int) -> MetadataPaths:
    suffix = f"resume{resume_index:03d}.json"
    return MetadataPaths(
        requested_config=None,
        resolved_config=root / f"config.resolved.{suffix}",
        compatibility_config=root / f"config.{suffix}",
        manifest=root / f"manifest.{suffix}",
        command=root / f"command.{suffix}",
        environment=root / f"environment.{suffix}",
    )


def write_initial_run_metadata(
    run_dir: Union[str, os.PathLike],
    *,
    requested_yaml: str,
    resolved_config: Mapping[str, Any],
    command: Sequence[str],
    environment: Mapping[str, Any],
    git_state: Mapping[str, Any],
    model: Mapping[str, Any],
    package_versions: Mapping[str, Any],
    host: Mapping[str, Any],
    gpus: Sequence[Any],
    worker_topology: Mapping[str, Any],
    seeds: Mapping[str, Any],
    versions: Mapping[str, Any],
    compatibility_config: Optional[Mapping[str, Any]] = None,
    created_at: Optional[str] = None,
    manifest_extra: Optional[Mapping[str, Any]] = None,
    run_id: Optional[str] = None,
) -> MetadataPaths:
    """Write the immutable initial run identity, publishing manifest last."""
    root = Path(run_dir)
    if not root.is_dir():
        raise ManifestValidationError(f"run directory does not exist: {root}")
    if not isinstance(requested_yaml, str) or not requested_yaml.strip():
        raise ManifestValidationError("requested_yaml must be non-empty text")
    paths = MetadataPaths(
        requested_config=root / "config.requested.yaml",
        resolved_config=root / "config.resolved.json",
        compatibility_config=root / "config.json",
        manifest=root / "manifest.json",
        command=root / "command.json",
        environment=root / "environment.json",
    )
    _preflight_immutable(
        [
            paths.requested_config,
            paths.resolved_config,
            paths.compatibility_config,
            paths.command,
            paths.environment,
            paths.manifest,
        ]
    )
    resolved = _validated_resolved_config(resolved_config, root)
    compatibility = _evolve_config(
        compatibility_config if compatibility_config is not None else resolved,
        "compatibility_config",
        expected_hash=resolved["config_hash"],
    )
    command_document = _command_document(command)
    environment_document = _environment_document(
        environment,
        host=host,
        gpus=gpus,
        package_versions=package_versions,
    )
    manifest_document = _manifest_document(
        root,
        resolved,
        created_at=_utc_now() if created_at is None else created_at,
        git_state=git_state,
        model=model,
        package_versions=package_versions,
        host=host,
        gpus=gpus,
        worker_topology=worker_topology,
        seeds=seeds,
        versions=versions,
        extra=manifest_extra,
        run_id=run_id,
    )
    write_immutable_text(paths.requested_config, requested_yaml)
    write_immutable_json(paths.resolved_config, resolved)
    write_immutable_json(paths.compatibility_config, compatibility)
    write_immutable_json(paths.command, command_document)
    write_immutable_json(paths.environment, environment_document)
    write_immutable_json(paths.manifest, manifest_document)
    return paths


def write_resume_run_metadata(
    run_dir: Union[str, os.PathLike],
    *,
    resume_index: int,
    resolved_config: Mapping[str, Any],
    command: Sequence[str],
    environment: Mapping[str, Any],
    git_state: Mapping[str, Any],
    model: Mapping[str, Any],
    package_versions: Mapping[str, Any],
    host: Mapping[str, Any],
    gpus: Sequence[Any],
    worker_topology: Mapping[str, Any],
    seeds: Mapping[str, Any],
    versions: Mapping[str, Any],
    checkpoint_hash: str,
    compatibility_config: Optional[Mapping[str, Any]] = None,
    resumed_at: Optional[str] = None,
    manifest_extra: Optional[Mapping[str, Any]] = None,
) -> MetadataPaths:
    """Add a numbered resume metadata set without replacing initial identity."""
    root = Path(run_dir)
    if isinstance(resume_index, bool) or not isinstance(resume_index, int) or resume_index < 1:
        raise ManifestValidationError("resume_index must be a positive integer")
    if not isinstance(checkpoint_hash, str) or not checkpoint_hash:
        raise ManifestValidationError("checkpoint_hash must be a non-empty string")
    if _SHA256_RE.fullmatch(checkpoint_hash) is None:
        raise ManifestValidationError(
            "checkpoint_hash must be a lowercase SHA-256 hex digest"
        )
    paths = _resume_paths(root, resume_index)
    # An exact numbered target owns that index forever, even when another chain
    # problem also exists.  Report the immutable collision before interpreting a
    # duplicate request as a request for some different next index.
    _preflight_immutable(
        [
            paths.resolved_config,
            paths.compatibility_config,
            paths.command,
            paths.environment,
            paths.manifest,
        ]
    )
    try:
        effective = resolve_effective_run_metadata(root)
    except Exception as exc:
        raise ManifestValidationError(
            f"resume metadata requires a complete valid EVOLVE metadata chain: {exc}"
        ) from exc
    expected_resume_index = effective.resume_index + 1
    if resume_index != expected_resume_index:
        raise ManifestValidationError(
            f"resume_index must be the next additive version "
            f"({expected_resume_index}), got {resume_index}"
        )
    initial_manifest = json.loads((root / "manifest.json").read_text(encoding="utf-8"))
    initial_config_hash = initial_manifest["config_hash"]
    previous_manifest_path = effective.manifest_path
    previous_config_hash = effective.manifest["config_hash"]
    if manifest_extra:
        reserved_resume = {
            "resume_index",
            "resumed_at",
            "checkpoint_hash",
            "initial_manifest",
            "initial_config_hash",
            "previous_manifest",
            "previous_config_hash",
        }
        overlap = sorted(reserved_resume.intersection(manifest_extra))
        if overlap:
            raise ManifestValidationError(
                "manifest_extra cannot replace resume keys: " + ", ".join(overlap)
            )

    resolved = _validated_resolved_config(resolved_config, root)
    compatibility = _evolve_config(
        compatibility_config if compatibility_config is not None else resolved,
        "compatibility_config",
        expected_hash=resolved["config_hash"],
    )
    command_document = _command_document(command)
    environment_document = _environment_document(
        environment,
        host=host,
        gpus=gpus,
        package_versions=package_versions,
    )
    manifest_document = _manifest_document(
        root,
        resolved,
        created_at=_utc_now() if resumed_at is None else resumed_at,
        git_state=git_state,
        model=model,
        package_versions=package_versions,
        host=host,
        gpus=gpus,
        worker_topology=worker_topology,
        seeds=seeds,
        versions=versions,
        extra=manifest_extra,
        run_id=initial_manifest.get("run_id"),
    )
    manifest_document.update(
        {
            "resume_index": resume_index,
            "resumed_at": manifest_document.pop("created_at"),
            "checkpoint_hash": checkpoint_hash,
            "initial_manifest": "manifest.json",
            "initial_config_hash": initial_config_hash,
            "previous_manifest": previous_manifest_path.name,
            "previous_config_hash": previous_config_hash,
        }
    )
    write_immutable_json(paths.resolved_config, resolved)
    write_immutable_json(paths.compatibility_config, compatibility)
    write_immutable_json(paths.command, command_document)
    write_immutable_json(paths.environment, environment_document)
    write_immutable_json(paths.manifest, manifest_document)
    return paths
