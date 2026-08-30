"""Immutable descriptor-indexed quality-diversity archive decisions."""

from __future__ import annotations

from dataclasses import dataclass, field, replace
from typing import Any, Dict, Optional, Tuple

from evolve.ids import validate_id
from evolve.types import (
    ArchiveCell,
    Descriptor,
    EvidencePacket,
    Proposal,
    VerifiedScientificState,
)

from .descriptors import (
    DEFAULT_CELL_MAP_VERSION,
    cell_id_for_descriptor,
    empty_cell_for_descriptor,
    validate_descriptor_identity,
)
from .store import (
    ArtifactReferenceError,
    ScientificArtifactStore,
    validate_state_evidence,
)


class ArchiveError(ValueError):
    """Base archive transition failure."""


class ArchiveCollisionError(ArchiveError):
    """A descriptor/cell identifier was reused for different content."""


class ArchiveAdmissionError(ArchiveError):
    """A candidate is not independently verified archive material."""


@dataclass(frozen=True)
class ArchiveDecision:
    cell_id: str
    state_id: str
    evidence_id: str
    duplicate: bool
    champion_changed: bool
    champion_state_id: Optional[str]
    promising_state_ids: Tuple[str, ...]
    stepping_stone_state_ids: Tuple[str, ...]
    evicted_slot_state_ids: Tuple[str, ...]


@dataclass(frozen=True)
class ScientificArchive:
    """Functional archive; slot eviction never removes retained artifacts."""

    cell_map_version: str = DEFAULT_CELL_MAP_VERSION
    max_promising_slots: int = 1
    max_stepping_stone_slots: int = 1
    under_tested_threshold: int = 2
    descriptors: Tuple[Descriptor, ...] = field(default_factory=tuple)
    cells: Tuple[ArchiveCell, ...] = field(default_factory=tuple)
    artifacts: ScientificArtifactStore = field(default_factory=ScientificArtifactStore)

    def __post_init__(self) -> None:
        if not isinstance(self.cell_map_version, str) or not self.cell_map_version.strip():
            raise ArchiveError("cell_map_version must be non-empty")
        for name, value in (
            ("max_promising_slots", self.max_promising_slots),
            ("max_stepping_stone_slots", self.max_stepping_stone_slots),
            ("under_tested_threshold", self.under_tested_threshold),
        ):
            if isinstance(value, bool) or not isinstance(value, int) or value < 1:
                raise ArchiveError(f"{name} must be a positive integer")
        descriptor_ids = [item.descriptor_id for item in self.descriptors]
        cell_ids = [item.cell_id for item in self.cells]
        if len(set(descriptor_ids)) != len(descriptor_ids):
            raise ArchiveCollisionError("archive descriptor IDs must be unique")
        if len(set(cell_ids)) != len(cell_ids):
            raise ArchiveCollisionError("archive cell IDs must be unique")
        known_descriptors = set(descriptor_ids)
        for descriptor in self.descriptors:
            try:
                validate_descriptor_identity(descriptor)
            except ValueError as exc:
                raise ArchiveCollisionError(str(exc)) from exc
        for cell in self.cells:
            if cell.descriptor_id not in known_descriptors:
                raise ArchiveError("archive cell references an unknown descriptor")
            descriptor = self.descriptor(cell.descriptor_id)
            expected = cell_id_for_descriptor(
                descriptor, cell_map_version=self.cell_map_version
            )
            if cell.cell_id != expected:
                raise ArchiveCollisionError("archive cell ID is not descriptor-derived")
            if cell.under_tested != (cell.tested_count < self.under_tested_threshold):
                raise ArchiveError("cell under_tested flag disagrees with its test count")
            if cell.force_empty_sampling and cell.tested_count != 0:
                raise ArchiveError("only an empty cell can force empty-cell sampling")
            slot_ids = (
                cell.promising_state_ids + cell.stepping_stone_state_ids
            )
            if (
                cell.champion_state_id is not None
                and cell.champion_state_id in slot_ids
            ):
                raise ArchiveError("champion, promising, and stepping slots must be distinct")
            if cell.tested_count == 0 and (
                cell.champion_state_id is not None or slot_ids
            ):
                raise ArchiveError("an untested cell cannot contain archive slots")
            for state_id in slot_ids:
                try:
                    representative = self.artifacts.representative_state(
                        state_id, descriptor_id=cell.descriptor_id
                    )
                except ArtifactReferenceError as exc:
                    raise ArchiveError("archive slot references an unknown state") from exc
                if representative.descriptor_id != cell.descriptor_id:
                    raise ArchiveError("archive slot belongs to a different descriptor")
            if cell.champion_state_id is not None:
                try:
                    champion = self.artifacts.state_binding(
                        cell.champion_state_id,
                        self.artifacts.evidence_packet(
                            cell.champion_evidence_id
                        ).proposal_id,
                        cell.champion_evidence_id,
                    )
                    champion_evidence = self.artifacts.evidence_packet(
                        cell.champion_evidence_id
                    )
                except ArtifactReferenceError as exc:
                    raise ArchiveError("champion references an unknown state binding") from exc
                if champion.descriptor_id != cell.descriptor_id:
                    raise ArchiveError("champion belongs to a different descriptor")
                if not champion.confirmed or not champion_evidence.confirmed:
                    raise ArchiveAdmissionError(
                        "archive champion requires confirmed evidence"
                    )

    def descriptor(self, descriptor_id: str) -> Descriptor:
        for descriptor in self.descriptors:
            if descriptor.descriptor_id == descriptor_id:
                return descriptor
        raise ArchiveError(f"unknown descriptor_id {descriptor_id!r}")

    def cell(self, cell_id: str) -> ArchiveCell:
        for cell in self.cells:
            if cell.cell_id == cell_id:
                return cell
        raise ArchiveError(f"unknown cell_id {cell_id!r}")

    def _replace_cell(self, updated: ArchiveCell) -> "ScientificArchive":
        cells = tuple(
            sorted(
                tuple(cell for cell in self.cells if cell.cell_id != updated.cell_id)
                + (updated,),
                key=lambda item: item.cell_id,
            )
        )
        return replace(self, cells=cells)

    def ensure_cell(
        self,
        descriptor: Descriptor,
        *,
        force_empty_sampling: bool = True,
    ) -> "ScientificArchive":
        existing_descriptor = next(
            (
                item
                for item in self.descriptors
                if item.descriptor_id == descriptor.descriptor_id
            ),
            None,
        )
        if existing_descriptor is not None and existing_descriptor.to_dict() != descriptor.to_dict():
            raise ArchiveCollisionError(
                f"descriptor_id collision for {descriptor.descriptor_id}"
            )
        descriptors = self.descriptors
        if existing_descriptor is None:
            descriptors = tuple(
                sorted(self.descriptors + (descriptor,), key=lambda item: item.descriptor_id)
            )
        cell_id = cell_id_for_descriptor(
            descriptor, cell_map_version=self.cell_map_version
        )
        existing_cell = next(
            (item for item in self.cells if item.cell_id == cell_id), None
        )
        updated_archive = replace(self, descriptors=descriptors)
        if existing_cell is None:
            return updated_archive._replace_cell(
                empty_cell_for_descriptor(
                    descriptor,
                    cell_map_version=self.cell_map_version,
                    force_empty_sampling=force_empty_sampling,
                )
            )
        if existing_cell.descriptor_id != descriptor.descriptor_id:
            raise ArchiveCollisionError(f"cell_id collision for {cell_id}")
        if (
            force_empty_sampling
            and existing_cell.tested_count == 0
            and not existing_cell.force_empty_sampling
        ):
            return updated_archive._replace_cell(
                replace(existing_cell, force_empty_sampling=True)
            )
        return updated_archive

    def _cell_candidates(
        self, descriptor_id: str
    ) -> Tuple[Tuple[VerifiedScientificState, EvidencePacket], ...]:
        state_ids = {
            state.state_id
            for state in self.artifacts.states
            if state.descriptor_id == descriptor_id
        }
        representatives = {
            state_id: self.artifacts.representative_state(
                state_id, descriptor_id=descriptor_id
            )
            for state_id in state_ids
        }
        return tuple(
            (
                representatives[state_id],
                self.artifacts.evidence_packet(representatives[state_id].evidence_id),
            )
            for state_id in sorted(representatives)
        )

    def offer(
        self,
        descriptor: Descriptor,
        proposal: Proposal,
        state: VerifiedScientificState,
        evidence: EvidencePacket,
    ) -> Tuple["ScientificArchive", ArchiveDecision]:
        """Offer one independently verified state and recompute local slots."""

        validate_state_evidence(
            state, evidence, require_descriptor=True, require_fingerprint=True
        )
        if state.descriptor_id != descriptor.descriptor_id:
            raise ArchiveAdmissionError("candidate descriptor reference is not this descriptor")
        if state.problem_id != descriptor.problem_id:
            raise ArchiveAdmissionError("candidate and descriptor problem IDs disagree")

        new_cell_state = not any(
            item.state_id == state.state_id
            and item.descriptor_id == descriptor.descriptor_id
            for item in self.artifacts.states
        )
        with_cell = self.ensure_cell(descriptor, force_empty_sampling=True)
        old_cell = with_cell.cell(
            cell_id_for_descriptor(
                descriptor, cell_map_version=with_cell.cell_map_version
            )
        )
        artifacts = with_cell.artifacts.add_verified(proposal, state, evidence)
        staged = replace(with_cell, artifacts=artifacts)
        candidates = staged._cell_candidates(descriptor.descriptor_id)
        if not candidates:
            raise ArchiveAdmissionError("verified candidate was not retained as an artifact")

        confirmed = sorted(
            (
                (candidate, packet)
                for candidate, packet in candidates
                if candidate.confirmed and packet.confirmed
            ),
            key=lambda item: (
                -float(item[0].internal_reward),
                item[0].state_id,
                item[1].evidence_id,
            ),
        )
        champion_state_id = confirmed[0][0].state_id if confirmed else None
        champion_evidence_id = confirmed[0][1].evidence_id if confirmed else None

        nonchampions = [
            (candidate, packet)
            for candidate, packet in candidates
            if candidate.state_id != champion_state_id
        ]
        promising_ranked = sorted(
            nonchampions,
            key=lambda item: (
                -float(item[0].internal_reward),
                item[0].state_id,
                item[1].evidence_id,
            ),
        )
        promising_ids = tuple(
            item[0].state_id
            for item in promising_ranked[: self.max_promising_slots]
        )

        family_counts: Dict[str, int] = {}
        for candidate, _packet in candidates:
            family_counts[candidate.fingerprint] = (
                family_counts.get(candidate.fingerprint, 0) + 1
            )
        remaining = [
            (candidate, packet)
            for candidate, packet in nonchampions
            if candidate.state_id not in set(promising_ids)
        ]
        stepping_ranked = sorted(
            remaining,
            key=lambda item: (
                family_counts[item[0].fingerprint],
                -float(item[1].uncertainty or 0.0),
                item[0].state_id,
                item[1].evidence_id,
            ),
        )
        stepping_ids = tuple(
            item[0].state_id
            for item in stepping_ranked[: self.max_stepping_stone_slots]
        )

        tested_count = old_cell.tested_count + (1 if new_cell_state else 0)
        updated_cell = ArchiveCell(
            cell_id=old_cell.cell_id,
            descriptor_id=descriptor.descriptor_id,
            champion_state_id=champion_state_id,
            champion_evidence_id=champion_evidence_id,
            promising_state_ids=promising_ids,
            stepping_stone_state_ids=stepping_ids,
            tested_count=tested_count,
            force_empty_sampling=False,
            under_tested=tested_count < self.under_tested_threshold,
        )
        old_slots = set(old_cell.promising_state_ids + old_cell.stepping_stone_state_ids)
        if old_cell.champion_state_id is not None:
            old_slots.add(old_cell.champion_state_id)
        new_slots = set(promising_ids + stepping_ids)
        if champion_state_id is not None:
            new_slots.add(champion_state_id)
        updated = staged._replace_cell(updated_cell)
        decision = ArchiveDecision(
            cell_id=updated_cell.cell_id,
            state_id=state.state_id,
            evidence_id=evidence.evidence_id,
            # Scientific identity belongs to the verified answer payload, not
            # the proposal binding. A new source/proposal that reproduces an
            # existing state is therefore still an explicit duplicate.
            duplicate=not new_cell_state,
            champion_changed=(
                old_cell.champion_state_id,
                old_cell.champion_evidence_id,
            )
            != (champion_state_id, champion_evidence_id),
            champion_state_id=champion_state_id,
            promising_state_ids=promising_ids,
            stepping_stone_state_ids=stepping_ids,
            evicted_slot_state_ids=tuple(sorted(old_slots - new_slots)),
        )
        return updated, decision

    def sampling_cells(self, limit: Optional[int] = None) -> Tuple[ArchiveCell, ...]:
        """Return forced-empty, empty, then under-tested cells deterministically."""

        eligible = [
            cell
            for cell in self.cells
            if cell.tested_count == 0 or cell.under_tested
        ]
        eligible.sort(
            key=lambda cell: (
                0
                if cell.force_empty_sampling and cell.tested_count == 0
                else 1 if cell.tested_count == 0 else 2,
                cell.tested_count,
                cell.cell_id,
            )
        )
        if limit is None:
            return tuple(eligible)
        if isinstance(limit, bool) or not isinstance(limit, int) or limit < 0:
            raise ArchiveError("sampling limit must be a non-negative integer")
        return tuple(eligible[:limit])

    @property
    def coverage(self) -> float:
        if not self.cells:
            return 0.0
        occupied = sum(1 for cell in self.cells if cell.tested_count > 0)
        return float(occupied / len(self.cells))


@dataclass(frozen=True)
class ConfirmedRecordTracker:
    """Best confirmed higher-is-better record with its native score preserved."""

    state_id: Optional[str] = None
    evidence_id: Optional[str] = None
    cell_id: Optional[str] = None
    internal_reward: Optional[float] = None
    raw_score: Any = None

    def __post_init__(self) -> None:
        present = (
            self.state_id is not None,
            self.evidence_id is not None,
            self.cell_id is not None,
            self.internal_reward is not None,
        )
        if any(present) and not all(present):
            raise ArchiveError("record state, evidence, cell, and reward must be present together")
        if self.state_id is not None:
            validate_id(self.state_id, "state")
            validate_id(self.evidence_id, "evidence")
            validate_id(self.cell_id, "cell")

    def consider(
        self,
        state: VerifiedScientificState,
        evidence: EvidencePacket,
        *,
        archive: ScientificArchive,
    ) -> "ConfirmedRecordTracker":
        validate_state_evidence(state, evidence, require_descriptor=True)
        if not state.confirmed or not evidence.confirmed:
            raise ArchiveAdmissionError("only confirmed evidence can update the record")
        if not isinstance(archive, ScientificArchive):
            raise ArchiveAdmissionError("record update requires its scientific archive")
        try:
            descriptor = archive.descriptor(state.descriptor_id)
            cell_id = cell_id_for_descriptor(
                descriptor,
                cell_map_version=archive.cell_map_version,
            )
            cell = archive.cell(cell_id)
            stored_state = archive.artifacts.state_binding(
                state.state_id,
                state.proposal_id,
                state.evidence_id,
            )
            stored_evidence = archive.artifacts.evidence_packet(evidence.evidence_id)
        except (ArchiveError, ArtifactReferenceError) as exc:
            raise ArchiveAdmissionError(
                "record candidate is not an exact archive state binding"
            ) from exc
        if cell.descriptor_id != state.descriptor_id:
            raise ArchiveAdmissionError("record candidate belongs to another cell")
        if stored_state.to_dict() != state.to_dict():
            raise ArchiveAdmissionError("record state differs from its archive binding")
        if stored_evidence.to_dict() != evidence.to_dict():
            raise ArchiveAdmissionError("record evidence differs from its archive packet")
        assert state.internal_reward is not None
        candidate_key = (-float(state.internal_reward), state.state_id, evidence.evidence_id)
        if self.internal_reward is not None:
            current_key = (
                -float(self.internal_reward),
                self.state_id,
                self.evidence_id,
            )
            if candidate_key >= current_key:
                return self
        return ConfirmedRecordTracker(
            state_id=state.state_id,
            evidence_id=evidence.evidence_id,
            cell_id=cell_id,
            internal_reward=float(state.internal_reward),
            raw_score=state.raw_score,
        )


__all__ = [
    "ArchiveAdmissionError",
    "ArchiveCollisionError",
    "ArchiveDecision",
    "ArchiveError",
    "ConfirmedRecordTracker",
    "ScientificArchive",
]
