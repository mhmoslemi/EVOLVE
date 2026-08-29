"""Source-independent fingerprints of verified scientific structure."""

from __future__ import annotations

from typing import Any, Dict, Mapping, Optional, Tuple

from evolve.ids import content_id
from evolve.types import EvidencePacket, VerifiedScientificState

from .store import validate_state_evidence


FINGERPRINT_VERSION = "verified_structure_v1"
_FORBIDDEN_SOURCE_KEYS = {
    "proposal",
    "proposal_id",
    "source",
    "source_hash",
    "source_text",
}


def _length_bucket(length: int) -> str:
    if length <= 0:
        return "0"
    lower = 1 << (length.bit_length() - 1)
    upper = (lower << 1) - 1
    return str(lower) if lower == upper else f"{lower}-{upper}"


def json_structure_signature(value: Any) -> Any:
    """Return a JSON-safe shape signature without scalar answer values."""

    if value is None:
        return "null"
    if isinstance(value, bool):
        return "boolean"
    if isinstance(value, (int, float)):
        return "number"
    if isinstance(value, str):
        return {"string_length": _length_bucket(len(value))}
    if isinstance(value, Mapping):
        return {
            "object": [
                [key, json_structure_signature(value[key])]
                for key in sorted(value)
            ]
        }
    if isinstance(value, (list, tuple)):
        item_signatures = tuple(json_structure_signature(item) for item in value)
        # Preserve ordered construction shape while bucketing only the length.
        return {
            "array_length": _length_bucket(len(value)),
            "items": list(item_signatures),
        }
    raise TypeError(f"verified payload contains non-JSON value {type(value).__name__}")


def structure_metrics(value: Any) -> Dict[str, Any]:
    """Compute deterministic complexity metrics from a captured answer payload."""

    counts: Dict[str, int] = {
        "objects": 0,
        "arrays": 0,
        "numbers": 0,
        "strings": 0,
        "booleans": 0,
        "nulls": 0,
        "nodes": 0,
    }
    key_names = set()

    def visit(item: Any, depth: int) -> int:
        counts["nodes"] += 1
        maximum = depth
        if item is None:
            counts["nulls"] += 1
        elif isinstance(item, bool):
            counts["booleans"] += 1
        elif isinstance(item, (int, float)):
            counts["numbers"] += 1
        elif isinstance(item, str):
            counts["strings"] += 1
        elif isinstance(item, Mapping):
            counts["objects"] += 1
            for key in sorted(item):
                key_names.add(key)
                maximum = max(maximum, visit(item[key], depth + 1))
        elif isinstance(item, (list, tuple)):
            counts["arrays"] += 1
            for child in item:
                maximum = max(maximum, visit(child, depth + 1))
        else:
            raise TypeError(
                f"verified payload contains non-JSON value {type(item).__name__}"
            )
        return maximum

    maximum_depth = visit(value, 0)
    return {
        **counts,
        "maximum_depth": maximum_depth,
        "node_count_bucket": _length_bucket(counts["nodes"]),
        "verified_keys": sorted(key_names),
    }


def _reject_source_keys(value: Any, path: str = "verified_features") -> None:
    if isinstance(value, Mapping):
        for key, child in value.items():
            if str(key).lower() in _FORBIDDEN_SOURCE_KEYS:
                raise ValueError(f"{path} exposes forbidden source field {key!r}")
            _reject_source_keys(child, f"{path}.{key}")
    elif isinstance(value, (list, tuple)):
        for index, child in enumerate(value):
            _reject_source_keys(child, f"{path}[{index}]")


def fingerprint_payload(
    state: VerifiedScientificState,
    evidence: EvidencePacket,
    *,
    version: str = FINGERPRINT_VERSION,
    verified_features: Optional[Mapping[str, Any]] = None,
) -> Dict[str, Any]:
    """Build the exact source-free payload hashed for a family fingerprint."""

    validate_state_evidence(state, evidence)
    if not isinstance(version, str) or not version.strip():
        raise ValueError("fingerprint version must be non-empty")
    features = {} if verified_features is None else dict(verified_features)
    _reject_source_keys(features)
    return {
        "version": version,
        "problem_id": state.problem_id,
        "answer_structure": json_structure_signature(state.answer_payload),
        "complexity": structure_metrics(state.answer_payload),
        "verified_behavior_structure": {
            "raw_score": json_structure_signature(evidence.raw_score),
            "scores": json_structure_signature(evidence.scores),
            "flags": json_structure_signature(evidence.flags),
        },
        "verified_diagnostics_structure": json_structure_signature(
            evidence.diagnostics
        ),
        "verified_features": features,
    }


def verified_structure_fingerprint(
    state: VerifiedScientificState,
    evidence: EvidencePacket,
    *,
    version: str = FINGERPRINT_VERSION,
    verified_features: Optional[Mapping[str, Any]] = None,
) -> str:
    """Return a stable family fingerprint that never observes proposal source."""

    return content_id(
        "fingerprint",
        fingerprint_payload(
            state,
            evidence,
            version=version,
            verified_features=verified_features,
        ),
    )


__all__ = [
    "FINGERPRINT_VERSION",
    "fingerprint_payload",
    "json_structure_signature",
    "structure_metrics",
    "verified_structure_fingerprint",
]
