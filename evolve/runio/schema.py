"""Read-only EVOLVE run schema detection.

Detection is deliberately strict because a resume must never guess which
method or schema produced scientific evidence. A missing or non-EVOLVE engine
is rejected; historical run formats are outside this EVOLVE-only repository.

This module only reads a run directory.  Migration and manifest creation belong
to explicit write paths elsewhere; detection never creates, updates, or
normalizes files.
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Mapping, Optional, Tuple, Union

from evolve.ids import content_hash, validate_id


CURRENT_CONFIG_SCHEMA_VERSION = 1
CURRENT_MANIFEST_SCHEMA_VERSION = 1
SUPPORTED_CONFIG_SCHEMA_VERSIONS = frozenset(
    {CURRENT_CONFIG_SCHEMA_VERSION}
)
SUPPORTED_MANIFEST_SCHEMA_VERSIONS = frozenset(
    {CURRENT_MANIFEST_SCHEMA_VERSION}
)
SUPPORTED_ENGINES = frozenset({"evolve"})
_SHA256_RE = re.compile(r"^[0-9a-f]{64}$")
_RESUME_METADATA_NAMES = {
    "resolved_config": ("config.resolved.resume", re.compile(r"^config\.resolved\.resume(\d+)\.json$")),
    "compatibility_config": ("config.resume", re.compile(r"^config\.resume(\d+)\.json$")),
    "manifest": ("manifest.resume", re.compile(r"^manifest\.resume(\d+)\.json$")),
    "command": ("command.resume", re.compile(r"^command\.resume(\d+)\.json$")),
    "environment": ("environment.resume", re.compile(r"^environment\.resume(\d+)\.json$")),
}
_MANIFEST_IDENTITY_MAPPINGS = (
    "git",
    "model",
    "packages",
    "host",
    "worker_topology",
    "seeds",
    "versions",
)


class RunSchemaError(ValueError):
    """Base class for a run whose method/schema cannot be determined safely."""


class MalformedRunError(RunSchemaError):
    """The run is incomplete, contradictory, or structurally malformed."""


class UnsupportedRunSchemaError(RunSchemaError):
    """The run declares a schema newer or otherwise unsupported here."""


@dataclass(frozen=True)
class RunSchema:
    """The immutable result of inspecting an existing run directory."""

    run_dir: Path
    engine: str
    config_path: Path
    resolved_config_path: Optional[Path]
    manifest_path: Optional[Path]
    config_schema_version: Optional[int]
    manifest_schema_version: Optional[int]

    @property
    def is_evolve(self) -> bool:
        return self.engine == "evolve"


@dataclass(frozen=True)
class EffectiveRunMetadata:
    """Latest committed EVOLVE config/manifest after validating its full chain.

    The helper returning this record is deliberately read-only.  A config loader
    can consume the selected documents without guessing a resume suffix or
    accidentally treating a partial numbered metadata set as authoritative.
    """

    run_dir: Path
    resume_index: int
    resolved_config_path: Path
    compatibility_config_path: Path
    manifest_path: Path
    command_path: Path
    environment_path: Path
    resolved_config: Dict[str, Any]
    manifest: Dict[str, Any]


def _read_json_object(path: Path) -> Dict[str, Any]:
    try:
        text = path.read_text(encoding="utf-8")
    except (OSError, UnicodeError) as exc:
        raise MalformedRunError(f"cannot read run metadata {path}: {exc}") from exc
    try:
        value = json.loads(text)
    except json.JSONDecodeError as exc:
        raise MalformedRunError(f"invalid JSON in run metadata {path}: {exc}") from exc
    if not isinstance(value, dict):
        raise MalformedRunError(f"run metadata must be a JSON object: {path}")
    return value


def _declared_engine(document: Dict[str, Any], path: Path) -> str:
    """Return the explicit EVOLVE engine declaration."""
    if "engine" not in document:
        raise UnsupportedRunSchemaError(
            f"run metadata has no explicit engine='evolve' declaration in {path}"
        )
    value = document["engine"]
    if not isinstance(value, str) or not value:
        raise MalformedRunError(f"engine must be a non-empty string in {path}")
    if value not in SUPPORTED_ENGINES:
        raise UnsupportedRunSchemaError(
            f"unsupported run engine {value!r} in {path}"
        )
    return value


def _schema_version(
    document: Dict[str, Any],
    path: Path,
    *,
    required: bool,
    supported: frozenset,
    kind: str,
) -> Optional[int]:
    if "schema_version" not in document:
        if required:
            raise MalformedRunError(
                f"explicit evolve run is missing {kind} schema_version in {path}"
            )
        return None
    value = document["schema_version"]
    if isinstance(value, bool) or not isinstance(value, int) or value < 1:
        raise MalformedRunError(
            f"{kind} schema_version must be a positive integer in {path}"
        )
    if value not in supported:
        raise UnsupportedRunSchemaError(
            f"unsupported {kind} schema_version {value} in {path}; "
            f"supported: {sorted(supported)}"
        )
    return value


def _strict_sha256(value: Any, field_name: str, path: Path) -> str:
    if not isinstance(value, str) or _SHA256_RE.fullmatch(value) is None:
        raise MalformedRunError(
            f"{field_name} must be a lowercase SHA-256 hex digest in {path}"
        )
    return value


def _nonempty_text(value: Any, field_name: str, path: Path) -> str:
    if not isinstance(value, str) or not value.strip():
        raise MalformedRunError(f"{field_name} must be non-empty in {path}")
    return value


def _nonempty_mapping(value: Any, field_name: str, path: Path) -> Mapping[str, Any]:
    if not isinstance(value, dict) or not value:
        raise MalformedRunError(
            f"manifest identity section {field_name} must be a non-empty object in {path}"
        )
    return value


def _validate_config_document(
    document: Dict[str, Any],
    path: Path,
    *,
    expected_hash: Optional[str] = None,
) -> str:
    if _declared_engine(document, path) != "evolve":
        raise MalformedRunError(f"effective config is not engine='evolve': {path}")
    _schema_version(
        document,
        path,
        required=True,
        supported=SUPPORTED_CONFIG_SCHEMA_VERSIONS,
        kind="config",
    )
    embedded_hash = _strict_sha256(document.get("config_hash"), "config_hash", path)
    hash_input = dict(document)
    hash_input.pop("config_hash", None)
    actual_hash = content_hash(hash_input)
    if actual_hash != embedded_hash:
        raise MalformedRunError(f"config_hash does not match config content in {path}")
    if expected_hash is not None and embedded_hash != expected_hash:
        raise MalformedRunError(
            f"compatibility config does not match authoritative config in {path}"
        )
    return embedded_hash


def _validate_manifest_identity(
    manifest: Dict[str, Any],
    path: Path,
    *,
    resumed: bool,
) -> None:
    if _declared_engine(manifest, path) != "evolve":
        raise MalformedRunError(f"manifest is not engine='evolve': {path}")
    _schema_version(
        manifest,
        path,
        required=True,
        supported=SUPPORTED_MANIFEST_SCHEMA_VERSIONS,
        kind="manifest",
    )
    try:
        validate_id(manifest.get("run_id"), "run")
    except (TypeError, ValueError) as exc:
        raise MalformedRunError(f"manifest has invalid run_id in {path}: {exc}") from exc
    _nonempty_text(manifest.get("run_name"), "run_name", path)
    timestamp_field = "resumed_at" if resumed else "created_at"
    _nonempty_text(manifest.get(timestamp_field), timestamp_field, path)
    config_schema = manifest.get("config_schema_version")
    if (
        isinstance(config_schema, bool)
        or not isinstance(config_schema, int)
        or config_schema != CURRENT_CONFIG_SCHEMA_VERSION
    ):
        raise MalformedRunError(
            f"manifest config_schema_version must be "
            f"{CURRENT_CONFIG_SCHEMA_VERSION} in {path}"
        )
    _strict_sha256(manifest.get("config_hash"), "config_hash", path)
    for field_name in _MANIFEST_IDENTITY_MAPPINGS:
        _nonempty_mapping(manifest.get(field_name), field_name, path)
    gpus = manifest.get("gpus")
    if not isinstance(gpus, list):
        raise MalformedRunError(f"manifest gpus must be a JSON list in {path}")
    for index, gpu in enumerate(gpus):
        if not isinstance(gpu, dict) or not gpu:
            raise MalformedRunError(
                f"manifest gpus[{index}] must be a non-empty object in {path}"
            )


def _validate_sidecars(
    command_path: Path,
    environment_path: Path,
    manifest: Dict[str, Any],
) -> None:
    command = _read_json_object(command_path)
    command_schema = command.get("schema_version")
    if (
        isinstance(command_schema, bool)
        or not isinstance(command_schema, int)
        or command_schema != 1
    ):
        raise MalformedRunError(f"invalid command schema_version in {command_path}")
    argv = command.get("argv")
    if (
        not isinstance(argv, list)
        or not argv
        or not all(isinstance(argument, str) for argument in argv)
    ):
        raise MalformedRunError(f"command argv must be a non-empty string list in {command_path}")

    environment = _read_json_object(environment_path)
    environment_schema = environment.get("schema_version")
    if (
        isinstance(environment_schema, bool)
        or not isinstance(environment_schema, int)
        or environment_schema != 1
    ):
        raise MalformedRunError(
            f"invalid environment schema_version in {environment_path}"
        )
    if not isinstance(environment.get("variables"), dict):
        raise MalformedRunError(
            f"environment variables must be an object in {environment_path}"
        )
    for field_name in ("host", "packages"):
        if not isinstance(environment.get(field_name), dict) or not environment[field_name]:
            raise MalformedRunError(
                f"environment {field_name} must be a non-empty object in {environment_path}"
            )
        if content_hash(environment[field_name]) != content_hash(manifest[field_name]):
            raise MalformedRunError(
                f"environment {field_name} disagrees with manifest in {environment_path}"
            )
    if not isinstance(environment.get("gpus"), list):
        raise MalformedRunError(f"environment gpus must be a list in {environment_path}")
    if content_hash(environment["gpus"]) != content_hash(manifest["gpus"]):
        raise MalformedRunError(
            f"environment gpus disagree with manifest in {environment_path}"
        )


def _resume_metadata_sets(root: Path) -> Tuple[Dict[str, Dict[int, Path]], Tuple[int, ...]]:
    by_kind: Dict[str, Dict[int, Path]] = {
        kind: {} for kind in _RESUME_METADATA_NAMES
    }
    for path in root.iterdir():
        name = path.name
        for kind, (prefix, pattern) in _RESUME_METADATA_NAMES.items():
            if not name.startswith(prefix):
                continue
            match = pattern.fullmatch(name)
            if match is None:
                raise MalformedRunError(f"malformed resume metadata filename: {path}")
            index = int(match.group(1))
            if index < 1 or name != f"{prefix}{index:03d}.json":
                raise MalformedRunError(f"non-canonical resume metadata filename: {path}")
            if not path.is_file():
                raise MalformedRunError(f"resume metadata is not a file: {path}")
            if index in by_kind[kind]:
                raise MalformedRunError(
                    f"duplicate resume metadata index {index} for {kind}"
                )
            by_kind[kind][index] = path
            break

    indices = tuple(sorted({index for values in by_kind.values() for index in values}))
    if indices:
        expected = tuple(range(1, indices[-1] + 1))
        if indices != expected:
            missing = sorted(set(expected) - set(indices))
            raise MalformedRunError(
                f"resume metadata indices must be contiguous from 1; missing {missing}"
            )
        for index in indices:
            missing_kinds = [kind for kind, values in by_kind.items() if index not in values]
            if missing_kinds:
                raise MalformedRunError(
                    f"resume metadata index {index} is partial; missing {missing_kinds}"
                )
    return by_kind, indices


def resolve_effective_run_metadata(
    run_dir: Union[str, Path],
) -> EffectiveRunMetadata:
    """Validate a complete resume chain and return its latest committed metadata.

    Every numbered config, manifest, command, and environment artifact must be
    present, indices must be contiguous, hashes and run identity must link to the
    prior version, and compatibility configs must be true copies of their
    authoritative resolved config.  The function never writes or repairs files.
    """

    detected = detect_run_schema(run_dir)
    if not detected.is_evolve:
        raise UnsupportedRunSchemaError("run is not an EVOLVE run")
    root = detected.run_dir
    requested_path = root / "config.requested.yaml"
    try:
        requested_yaml = requested_path.read_text(encoding="utf-8")
    except (OSError, UnicodeError) as exc:
        raise MalformedRunError(
            f"cannot read config.requested.yaml in {root}: {exc}"
        ) from exc
    if not requested_yaml.strip():
        raise MalformedRunError(
            f"EVOLVE run has no non-empty config.requested.yaml: {root}"
        )

    resolved_path = root / "config.resolved.json"
    compatibility_path = root / "config.json"
    manifest_path = root / "manifest.json"
    command_path = root / "command.json"
    environment_path = root / "environment.json"
    resolved = _read_json_object(resolved_path)
    compatibility = _read_json_object(compatibility_path)
    manifest = _read_json_object(manifest_path)
    initial_hash = _validate_config_document(resolved, resolved_path)
    _validate_config_document(
        compatibility, compatibility_path, expected_hash=initial_hash
    )
    _validate_manifest_identity(manifest, manifest_path, resumed=False)
    forbidden_initial_fields = {
        "resume_index",
        "resumed_at",
        "checkpoint_hash",
        "initial_manifest",
        "initial_config_hash",
        "previous_manifest",
        "previous_config_hash",
    }
    unexpected = sorted(forbidden_initial_fields.intersection(manifest))
    if unexpected:
        raise MalformedRunError(
            f"initial manifest contains resume-only fields {unexpected}: {manifest_path}"
        )
    if manifest.get("config_hash") != initial_hash:
        raise MalformedRunError(
            f"manifest config_hash disagrees with authoritative config in {manifest_path}"
        )
    _validate_sidecars(command_path, environment_path, manifest)

    initial_run_id = manifest["run_id"]
    initial_run_name = manifest["run_name"]
    by_kind, indices = _resume_metadata_sets(root)
    previous_manifest = manifest
    previous_manifest_path = manifest_path
    effective_index = 0
    for index in indices:
        current_resolved_path = by_kind["resolved_config"][index]
        current_compatibility_path = by_kind["compatibility_config"][index]
        current_manifest_path = by_kind["manifest"][index]
        current_command_path = by_kind["command"][index]
        current_environment_path = by_kind["environment"][index]
        current_resolved = _read_json_object(current_resolved_path)
        current_compatibility = _read_json_object(current_compatibility_path)
        current_manifest = _read_json_object(current_manifest_path)
        current_hash = _validate_config_document(
            current_resolved, current_resolved_path
        )
        _validate_config_document(
            current_compatibility,
            current_compatibility_path,
            expected_hash=current_hash,
        )
        _validate_manifest_identity(
            current_manifest, current_manifest_path, resumed=True
        )
        if current_manifest.get("config_hash") != current_hash:
            raise MalformedRunError(
                f"resume manifest config_hash disagrees with config in {current_manifest_path}"
            )
        resume_index = current_manifest.get("resume_index")
        if (
            isinstance(resume_index, bool)
            or not isinstance(resume_index, int)
            or resume_index != index
        ):
            raise MalformedRunError(
                f"resume_index must equal filename index {index} in {current_manifest_path}"
            )
        if current_manifest.get("run_id") != initial_run_id:
            raise MalformedRunError(
                f"resume manifest run_id chain is inconsistent in {current_manifest_path}"
            )
        if current_manifest.get("run_name") != initial_run_name:
            raise MalformedRunError(
                f"resume manifest run_name chain is inconsistent in {current_manifest_path}"
            )
        if current_manifest.get("initial_manifest") != "manifest.json":
            raise MalformedRunError(
                f"resume initial_manifest pointer is invalid in {current_manifest_path}"
            )
        if current_manifest.get("initial_config_hash") != initial_hash:
            raise MalformedRunError(
                f"resume initial_config_hash is invalid in {current_manifest_path}"
            )
        if current_manifest.get("previous_manifest") != previous_manifest_path.name:
            raise MalformedRunError(
                f"resume previous_manifest pointer is invalid in {current_manifest_path}"
            )
        previous_hash = _strict_sha256(
            previous_manifest.get("config_hash"),
            "previous manifest config_hash",
            previous_manifest_path,
        )
        if current_manifest.get("previous_config_hash") != previous_hash:
            raise MalformedRunError(
                f"resume previous_config_hash is invalid in {current_manifest_path}"
            )
        _strict_sha256(
            current_manifest.get("checkpoint_hash"),
            "checkpoint_hash",
            current_manifest_path,
        )
        _validate_sidecars(
            current_command_path, current_environment_path, current_manifest
        )
        resolved_path = current_resolved_path
        compatibility_path = current_compatibility_path
        manifest_path = current_manifest_path
        command_path = current_command_path
        environment_path = current_environment_path
        resolved = current_resolved
        manifest = current_manifest
        previous_manifest = current_manifest
        previous_manifest_path = current_manifest_path
        effective_index = index

    return EffectiveRunMetadata(
        run_dir=root,
        resume_index=effective_index,
        resolved_config_path=resolved_path.resolve(),
        compatibility_config_path=compatibility_path.resolve(),
        manifest_path=manifest_path.resolve(),
        command_path=command_path.resolve(),
        environment_path=environment_path.resolve(),
        resolved_config=resolved,
        manifest=manifest,
    )


def detect_run_schema(run_dir: Union[str, Path]) -> RunSchema:
    """Inspect an existing run without changing it.

    ``config.json`` remains the compatibility identity. The authoritative
    ``config.resolved.json`` and ``manifest.json`` are also required; all three
    documents must agree on ``engine=evolve`` and supported schema versions.
    """
    root = Path(run_dir).expanduser()
    if not root.is_dir():
        raise MalformedRunError(f"run directory does not exist: {root}")

    config_path = root / "config.json"
    if not config_path.is_file():
        raise MalformedRunError(f"run directory has no config.json: {root}")

    resolved_path = root / "config.resolved.json"
    manifest_path = root / "manifest.json"
    config = _read_json_object(config_path)
    resolved = _read_json_object(resolved_path) if resolved_path.is_file() else None
    manifest = _read_json_object(manifest_path) if manifest_path.is_file() else None

    config_engine = _declared_engine(config, config_path)
    resolved_engine = (
        _declared_engine(resolved, resolved_path) if resolved is not None else None
    )
    if resolved_engine is not None and resolved_engine != config_engine:
        raise MalformedRunError(
            "ambiguous run engine: config.json declares "
            f"{config_engine!r} but config.resolved.json declares "
            f"{resolved_engine!r}"
        )

    manifest_engine = None
    if manifest is not None:
        manifest_engine = _declared_engine(manifest, manifest_path)
        if manifest_engine != config_engine:
            raise MalformedRunError(
                "ambiguous run engine: configuration declares "
                f"{config_engine!r} but manifest.json declares "
                f"{manifest_engine!r}"
            )

    is_evolve = config_engine == "evolve"
    if resolved is None:
        raise MalformedRunError(
            "explicit evolve run has no authoritative config.resolved.json"
        )
    if manifest is None:
        raise MalformedRunError("explicit evolve run has no manifest.json")

    assert resolved is not None and manifest is not None
    # Reject future method schemas before interpreting or hashing them with
    # today's rules.
    _schema_version(
        config, config_path, required=True,
        supported=SUPPORTED_CONFIG_SCHEMA_VERSIONS, kind="config",
    )
    _schema_version(
        resolved, resolved_path, required=True,
        supported=SUPPORTED_CONFIG_SCHEMA_VERSIONS, kind="config",
    )
    _schema_version(
        manifest, manifest_path, required=True,
        supported=SUPPORTED_MANIFEST_SCHEMA_VERSIONS, kind="manifest",
    )
    resolved_hash = resolved.get("config_hash")
    if not isinstance(resolved_hash, str) or not _SHA256_RE.fullmatch(resolved_hash):
        raise MalformedRunError(
            "explicit evolve config.resolved.json has no valid config_hash"
        )
    hash_input = dict(resolved)
    hash_input.pop("config_hash", None)
    if content_hash(hash_input) != resolved_hash:
        raise MalformedRunError(
            "config.resolved.json config_hash does not match its content"
        )
    if config.get("config_hash") != resolved_hash:
        raise MalformedRunError(
            "config.json compatibility hash does not match authoritative config"
        )
    if manifest.get("config_hash") != resolved_hash:
        raise MalformedRunError(
            "manifest config_hash does not match authoritative config"
        )
    try:
        validate_id(manifest.get("run_id"), "run")
    except (TypeError, ValueError) as exc:
        raise MalformedRunError(f"manifest has invalid run_id: {exc}") from exc

    config_version = _schema_version(
        config,
        config_path,
        required=is_evolve,
        supported=SUPPORTED_CONFIG_SCHEMA_VERSIONS,
        kind="config",
    )
    resolved_version = None
    if resolved is not None:
        resolved_version = _schema_version(
            resolved,
            resolved_path,
            required=is_evolve,
            supported=SUPPORTED_CONFIG_SCHEMA_VERSIONS,
            kind="config",
        )
        if (
            config_version is not None
            and resolved_version is not None
            and config_version != resolved_version
        ):
            raise MalformedRunError(
                "ambiguous config schema_version between config.json and "
                "config.resolved.json"
            )

    manifest_version = None
    if manifest is not None:
        manifest_version = _schema_version(
            manifest,
            manifest_path,
            required=is_evolve,
            supported=SUPPORTED_MANIFEST_SCHEMA_VERSIONS,
            kind="manifest",
        )
        if is_evolve:
            manifest_config_version = manifest.get("config_schema_version")
            if (
                isinstance(manifest_config_version, bool)
                or not isinstance(manifest_config_version, int)
                or manifest_config_version != resolved_version
            ):
                raise MalformedRunError(
                    "manifest config_schema_version disagrees with authoritative config"
                )

    return RunSchema(
        run_dir=root.resolve(),
        engine=config_engine,
        config_path=config_path.resolve(),
        resolved_config_path=(resolved_path.resolve() if resolved is not None else None),
        manifest_path=(manifest_path.resolve() if manifest is not None else None),
        config_schema_version=(resolved_version or config_version),
        manifest_schema_version=manifest_version,
    )
