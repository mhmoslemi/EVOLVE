"""Pure descriptor extraction and deterministic archive-cell mapping."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable, Mapping, Optional

from evolve.ids import content_hash, content_id
from evolve.types import (
    ArchiveCell,
    Descriptor,
    EvidencePacket,
    FrozenDict,
    VerifiedScientificState,
)

from .fingerprints import json_structure_signature, structure_metrics
from .store import ArtifactReferenceError, validate_state_evidence


DEFAULT_DESCRIPTOR_VERSION = "coarse_verified_structure_v1"
DEFAULT_CELL_MAP_VERSION = "descriptor_exact_v1"
DESCRIPTOR_IDENTITY_VERSION = "scientific_descriptor_v1"


@dataclass(frozen=True)
class VerifiedDescriptorInput:
    """The complete source-free view made available to a descriptor function."""

    problem_id: str
    answer_payload: Any
    internal_reward: float
    raw_score: Any
    uncertainty: Optional[float]
    scores: FrozenDict
    flags: FrozenDict
    verifier_id: str
    verifier_version: str


DescriptorExtractor = Callable[[VerifiedDescriptorInput], Mapping[str, Any]]
_FORBIDDEN_SOURCE_KEYS = {
    "proposal",
    "proposal_id",
    "source",
    "source_hash",
    "source_text",
}


def _reject_source_keys(value: Any, path: str = "dimensions") -> None:
    if isinstance(value, Mapping):
        for key, child in value.items():
            if str(key).lower() in _FORBIDDEN_SOURCE_KEYS:
                raise ValueError(f"{path} exposes forbidden source field {key!r}")
            _reject_source_keys(child, f"{path}.{key}")
    elif isinstance(value, (list, tuple)):
        for index, child in enumerate(value):
            _reject_source_keys(child, f"{path}[{index}]")


def descriptor_identity_document(
    *,
    problem_id: str,
    function_version: str,
    dimensions: Mapping[str, Any],
    method_complete: bool,
) -> Mapping[str, Any]:
    """Return the descriptor identity shared with the verifier service."""

    return {
        "identity_version": DESCRIPTOR_IDENTITY_VERSION,
        "problem_id": problem_id,
        "function_version": function_version,
        "dimensions": dict(dimensions),
        "method_complete": method_complete,
    }


def derive_descriptor_id(
    *,
    problem_id: str,
    function_version: str,
    dimensions: Mapping[str, Any],
    method_complete: bool,
) -> str:
    return content_id(
        "descriptor",
        descriptor_identity_document(
            problem_id=problem_id,
            function_version=function_version,
            dimensions=dimensions,
            method_complete=method_complete,
        ),
    )


def validate_descriptor_identity(descriptor: Descriptor) -> None:
    expected = derive_descriptor_id(
        problem_id=descriptor.problem_id,
        function_version=descriptor.function_version,
        dimensions=descriptor.dimensions,
        method_complete=descriptor.method_complete,
    )
    if descriptor.descriptor_id != expected:
        raise ValueError("descriptor_id does not match scientific descriptor content")


def _source_free_view(
    state: VerifiedScientificState, evidence: EvidencePacket
) -> VerifiedDescriptorInput:
    # FrozenDict recursively freezes the payload without exposing the EvidencePacket
    # itself (and therefore without exposing source_hash or proposal metadata).
    frozen = FrozenDict(
        {
            "answer_payload": state.answer_payload,
            "raw_score": evidence.raw_score,
            "scores": evidence.scores,
            "flags": evidence.flags,
        }
    )
    assert state.internal_reward is not None
    return VerifiedDescriptorInput(
        problem_id=state.problem_id,
        answer_payload=frozen["answer_payload"],
        internal_reward=state.internal_reward,
        raw_score=frozen["raw_score"],
        uncertainty=evidence.uncertainty,
        scores=FrozenDict(frozen["scores"]),
        flags=FrozenDict(frozen["flags"]),
        verifier_id=evidence.verifier_id,
        verifier_version=evidence.verifier_version,
    )


def coarse_verified_dimensions(view: VerifiedDescriptorInput) -> Mapping[str, Any]:
    """Method-incomplete fallback descriptor for smoke tests and adapters."""

    metrics = structure_metrics(view.answer_payload)
    return {
        "root_structure": json_structure_signature(view.answer_payload),
        "maximum_depth": metrics["maximum_depth"],
        "node_count_bucket": metrics["node_count_bucket"],
        "object_count": metrics["objects"],
        "array_count": metrics["arrays"],
        "verified_key_signature": content_hash(metrics["verified_keys"]),
    }


def create_descriptor(
    state: VerifiedScientificState,
    evidence: EvidencePacket,
    *,
    function_version: str = DEFAULT_DESCRIPTOR_VERSION,
    extractor: Optional[DescriptorExtractor] = None,
    method_complete: bool = False,
) -> Descriptor:
    """Create a content-derived descriptor without accepting a Proposal/source."""

    validate_state_evidence(state, evidence)
    if not isinstance(function_version, str) or not function_version.strip():
        raise ValueError("descriptor function_version must be non-empty")
    if not isinstance(method_complete, bool):
        raise TypeError("method_complete must be boolean")
    if extractor is None:
        if method_complete:
            raise ValueError(
                "the generic coarse descriptor must remain method-incomplete; "
                "provide a problem-defined extractor"
            )
        extractor = coarse_verified_dimensions
    view = _source_free_view(state, evidence)
    dimensions = extractor(view)
    if not isinstance(dimensions, Mapping) or not dimensions:
        raise ValueError("descriptor extractor must return a non-empty mapping")
    _reject_source_keys(dimensions)
    descriptor = Descriptor(
        descriptor_id=derive_descriptor_id(
            problem_id=state.problem_id,
            function_version=function_version,
            dimensions=dimensions,
            method_complete=method_complete,
        ),
        problem_id=state.problem_id,
        function_version=function_version,
        dimensions=dict(dimensions),
        method_complete=method_complete,
    )
    for existing in (state.descriptor_id, evidence.descriptor_id):
        if existing is not None and existing != descriptor.descriptor_id:
            raise ArtifactReferenceError(
                "existing state/evidence descriptor reference disagrees with extraction"
            )
    return descriptor


def cell_id_for_descriptor(
    descriptor: Descriptor,
    *,
    cell_map_version: str = DEFAULT_CELL_MAP_VERSION,
) -> str:
    """Map one scientific descriptor to a stable descriptor-indexed cell."""

    if not isinstance(cell_map_version, str) or not cell_map_version.strip():
        raise ValueError("cell_map_version must be non-empty")
    validate_descriptor_identity(descriptor)
    return content_id(
        "cell",
        {
            "cell_map_version": cell_map_version,
            "problem_id": descriptor.problem_id,
            "descriptor_id": descriptor.descriptor_id,
            "dimensions": descriptor.dimensions,
        },
    )


def empty_cell_for_descriptor(
    descriptor: Descriptor,
    *,
    cell_map_version: str = DEFAULT_CELL_MAP_VERSION,
    force_empty_sampling: bool = True,
) -> ArchiveCell:
    return ArchiveCell(
        cell_id=cell_id_for_descriptor(
            descriptor, cell_map_version=cell_map_version
        ),
        descriptor_id=descriptor.descriptor_id,
        tested_count=0,
        force_empty_sampling=force_empty_sampling,
        under_tested=True,
    )


__all__ = [
    "DEFAULT_CELL_MAP_VERSION",
    "DEFAULT_DESCRIPTOR_VERSION",
    "DESCRIPTOR_IDENTITY_VERSION",
    "DescriptorExtractor",
    "VerifiedDescriptorInput",
    "cell_id_for_descriptor",
    "coarse_verified_dimensions",
    "create_descriptor",
    "derive_descriptor_id",
    "descriptor_identity_document",
    "empty_cell_for_descriptor",
    "validate_descriptor_identity",
]
