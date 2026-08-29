"""Stable identities and deterministic seeds for EVOLVE.

Identity in EVOLVE is derived from canonical JSON, never from Python's
process-randomized :func:`hash`.  The helpers in this module deliberately do
not depend on worker rank, completion order, or process topology.
"""

from __future__ import annotations

import hashlib
import json
import math
import re
from dataclasses import fields, is_dataclass
from enum import Enum
from typing import Any, Mapping, Optional, Sequence


_NAMESPACE_RE = re.compile(r"^[a-z][a-z0-9_.-]*$")
_IDENTIFIER_RE = re.compile(r"^(?P<namespace>[a-z][a-z0-9_.-]*):(?P<digest>[0-9a-f]{64})$")


class CanonicalJSONError(ValueError):
    """Raised when a value cannot be represented as canonical JSON."""


def _canonical_value(value: Any) -> Any:
    """Return a JSON-native, deterministically ordered representation.

    Tuples and other non-text sequences become JSON arrays.  Mapping keys must
    already be strings: silently stringifying a key would make distinct Python
    objects share an identity.  NaN and infinities are rejected because JSON
    encoders disagree about their representation.
    """

    if hasattr(value, "to_dict") and callable(value.to_dict):
        return _canonical_value(value.to_dict())
    if isinstance(value, Enum):
        return _canonical_value(value.value)
    if value is None or isinstance(value, (str, bool, int)):
        return value
    if isinstance(value, float):
        if not math.isfinite(value):
            raise CanonicalJSONError("canonical JSON does not permit NaN or infinity")
        # JSON has only one numeric zero.  Normalizing -0.0 prevents a diagnostic
        # sign bit from changing scientific identity.
        return 0.0 if value == 0.0 else value
    if isinstance(value, Mapping):
        out = {}
        for key in sorted(value):
            if not isinstance(key, str):
                raise CanonicalJSONError(
                    f"canonical JSON mapping keys must be strings, got {type(key).__name__}"
                )
            out[key] = _canonical_value(value[key])
        return out
    if isinstance(value, Sequence) and not isinstance(
        value, (str, bytes, bytearray, memoryview)
    ):
        return [_canonical_value(item) for item in value]
    if is_dataclass(value):
        return {
            field.name: _canonical_value(getattr(value, field.name))
            for field in fields(value)
        }
    raise CanonicalJSONError(
        f"unsupported canonical JSON value of type {type(value).__name__}"
    )


def canonical_json(value: Any) -> str:
    """Serialize *value* with the one canonical encoding used by EVOLVE."""

    try:
        return json.dumps(
            _canonical_value(value),
            ensure_ascii=False,
            allow_nan=False,
            sort_keys=True,
            separators=(",", ":"),
        )
    except (TypeError, ValueError) as exc:
        if isinstance(exc, CanonicalJSONError):
            raise
        raise CanonicalJSONError(str(exc)) from exc


def canonical_bytes(value: Any) -> bytes:
    """Return the UTF-8 bytes hashed for content identity."""

    return canonical_json(value).encode("utf-8")


def content_hash(value: Any) -> str:
    """Return a full SHA-256 hex digest of canonical JSON content."""

    return hashlib.sha256(canonical_bytes(value)).hexdigest()


def _validate_namespace(namespace: str) -> str:
    namespace = str(namespace)
    if not _NAMESPACE_RE.fullmatch(namespace):
        raise ValueError(
            "ID namespace must start with a lowercase letter and contain only "
            "lowercase letters, digits, '.', '_' or '-'"
        )
    return namespace


def content_id(namespace: str, value: Any) -> str:
    """Return ``namespace:<sha256>`` for canonical content."""

    return f"{_validate_namespace(namespace)}:{content_hash(value)}"


def derive_id(namespace: str, *identity_parts: Any) -> str:
    """Derive a namespaced ID from an ordered tuple of identity components."""

    return content_id(namespace, {"identity_parts": list(identity_parts)})


def id_namespace(identifier: str) -> str:
    """Return an identifier's namespace, rejecting malformed identities."""

    match = _IDENTIFIER_RE.fullmatch(str(identifier))
    if match is None:
        raise ValueError(f"invalid EVOLVE identifier: {identifier!r}")
    return match.group("namespace")


def validate_id(identifier: str, namespace: Optional[str] = None) -> str:
    """Validate and return an EVOLVE content identifier.

    When *namespace* is supplied, references cannot accidentally cross record
    kinds (for example an evidence ID in a proposal field).
    """

    actual = id_namespace(identifier)
    if namespace is not None and actual != _validate_namespace(namespace):
        raise ValueError(
            f"expected {namespace!r} identifier, got {actual!r}: {identifier!r}"
        )
    return str(identifier)


def derive_seed(*identity_parts: Any, base_seed: int = 0, bits: int = 63) -> int:
    """Derive a stable non-negative integer seed from logical identity only.

    ``bits=63`` fits Python, NumPy, PyTorch, and common inference APIs.  Worker
    rank and device are intentionally not parameters; callers should pass only
    scientific/job identity components.
    """

    if not isinstance(base_seed, int) or isinstance(base_seed, bool):
        raise TypeError("base_seed must be an integer")
    if not isinstance(bits, int) or not 1 <= bits <= 256:
        raise ValueError("bits must be an integer in [1, 256]")
    digest = hashlib.sha256(
        canonical_bytes(
            {"base_seed": base_seed, "identity_parts": list(identity_parts)}
        )
    ).digest()
    value = int.from_bytes(digest, "big")
    return value & ((1 << bits) - 1)


def rollout_seed(
    *,
    run_id: str,
    epoch: int,
    allocation_id: str,
    branch_step: int,
    sample_index: int,
    role: str,
    base_seed: int = 0,
) -> int:
    """Derive the topology-independent seed mandated for generation jobs."""

    indices = {
        "epoch": epoch,
        "branch_step": branch_step,
        "sample_index": sample_index,
    }
    for name, value in indices.items():
        if isinstance(value, bool) or not isinstance(value, int) or value < 0:
            raise ValueError(f"{name} must be a non-negative integer")
    return derive_seed(
        "rollout",
        str(run_id),
        epoch,
        str(allocation_id),
        branch_step,
        sample_index,
        str(role),
        base_seed=base_seed,
    )


__all__ = [
    "CanonicalJSONError",
    "canonical_bytes",
    "canonical_json",
    "content_hash",
    "content_id",
    "derive_id",
    "derive_seed",
    "id_namespace",
    "rollout_seed",
    "validate_id",
]
