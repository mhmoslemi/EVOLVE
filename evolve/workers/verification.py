"""Streaming candidate persistence, verification, and bounded backpressure.

Turns one raw generated response into a durably persisted answer payload and
its common-verifier result.  A worker error (an exception outside the
adapter's own verification call) is explicit infrastructure evidence and is
never silently mapped to a low scientific reward; a parse/extraction failure
is instead handed to the adapter as a null-ish payload so the *problem*
(never this plumbing) classifies it.  :class:`BackpressureQueue` provides the
bounded generation-to-verification queue AGENTS.md requires so a slow
verifier/persistence stage throttles generation instead of unbounded memory
growth.
"""

from __future__ import annotations

import queue
import json
import math
import threading
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable, Optional, Tuple, TypeVar, Union

from evolve.ids import content_hash, content_id
from evolve.options.branch import BranchStepResult, PolicySegment
from evolve.runio.atomic import (
    ImmutableWriteError,
    write_immutable_json,
    write_immutable_text,
)
from evolve.types import (
    Descriptor,
    EvidencePacket,
    FrozenDict,
    Proposal,
    VerifiedScientificState,
)
from evolve.verifier.adapters import ScientificProblemAdapter
from evolve.verifier.evidence import ScientificVerificationResult, build_verification_result
from evolve.verifier.models import (
    ExecutionCapture,
    PersistedAnswerPayload,
    VerificationDecision,
    VerificationPolicy,
    classify_failure,
    thaw_json,
)
from evolve.verifier.service import verify_persisted_answer


class VerificationWorkerError(RuntimeError):
    """A genuine infrastructure failure occurred outside adapter verification."""


class DurableVerificationConflict(VerificationWorkerError):
    """Persisted sample evidence is contradictory and must never be retried."""


def _utc_now() -> str:
    return (
        datetime.now(timezone.utc).isoformat(timespec="microseconds").replace("+00:00", "Z")
    )


def persist_answer_artifact(*, run_dir: Union[str, Path], problem_id: str, payload: Any) -> Path:
    """Write one candidate's answer payload as a durable, content-addressed file.

    Identical payloads reuse the same file, matching "content-address
    repeated prompts while retaining readable compatibility files."
    """

    answers_dir = Path(run_dir) / "artifacts" / "answers"
    answers_dir.mkdir(parents=True, exist_ok=True)
    digest = content_hash({"problem_id": problem_id, "payload": payload})
    destination = answers_dir / f"{digest}.json"
    try:
        write_immutable_json(destination, payload)
    except ImmutableWriteError:
        try:
            durable_payload = json.loads(destination.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as exc:
            raise DurableVerificationConflict(
                f"content-addressed answer artifact is unreadable: {destination}"
            ) from exc
        if content_hash(durable_payload) != content_hash(payload):
            raise DurableVerificationConflict(
                f"content-addressed answer artifact conflict: {destination}"
            )
    return destination


@dataclass(frozen=True)
class GenerationOutcome:
    """What a live generation call actually produced for one request."""

    prompt: str
    text: str
    token_ids: Tuple[int, ...] = ()
    log_probabilities: Tuple[float, ...] = ()
    token_mask: Optional[Tuple[bool, ...]] = None
    seed: Optional[int] = None

    def __post_init__(self) -> None:
        if not isinstance(self.prompt, str) or not isinstance(self.text, str):
            raise VerificationWorkerError("generation prompt and response must be text")
        if any(
            isinstance(item, bool) or not isinstance(item, int) or item < 0
            for item in self.token_ids
        ):
            raise VerificationWorkerError(
                "generation token IDs must be non-negative integers"
            )
        if any(
            isinstance(item, bool)
            or not isinstance(item, (int, float))
            or not math.isfinite(float(item))
            for item in self.log_probabilities
        ):
            raise VerificationWorkerError(
                "generation log probabilities must be finite numbers"
            )
        if self.token_ids and self.log_probabilities and (
            len(self.token_ids) != len(self.log_probabilities)
        ):
            raise VerificationWorkerError(
                "generation token IDs and log probabilities must align"
            )
        if self.token_mask is not None:
            if any(not isinstance(item, bool) for item in self.token_mask):
                raise VerificationWorkerError(
                    "generation token mask must contain booleans"
                )
            if len(self.token_mask) != len(self.log_probabilities):
                raise VerificationWorkerError(
                    "generation token mask and log probabilities must align"
                )
        if self.seed is not None and (
            isinstance(self.seed, bool)
            or not isinstance(self.seed, int)
            or self.seed < 0
        ):
            raise VerificationWorkerError(
                "generation seed must be a non-negative integer"
            )


def _immutable_text_or_same(path: Path, value: str) -> None:
    try:
        write_immutable_text(path, value)
    except ImmutableWriteError:
        try:
            durable = path.read_text(encoding="utf-8")
        except OSError as exc:
            raise DurableVerificationConflict(
                f"durable text artifact is unreadable: {path}"
            ) from exc
        if durable != value:
            raise DurableVerificationConflict(
                f"durable artifact identity conflict: {path}"
            )


def _immutable_json_or_same(path: Path, value: Any) -> None:
    try:
        write_immutable_json(path, value)
    except ImmutableWriteError:
        try:
            existing = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as exc:
            raise DurableVerificationConflict(
                f"durable JSON artifact is unreadable: {path}"
            ) from exc
        if existing != value:
            raise DurableVerificationConflict(
                f"durable artifact identity conflict: {path}"
            )


def persist_generation_arrival(
    *,
    run_dir: Union[str, Path],
    request: Any,
    generation: GenerationOutcome,
) -> None:
    """Persist a rendered prompt and raw response before parsing or verifying."""

    branch_dir = (
        Path(run_dir)
        / f"step{request.branch.epoch:02d}"
        / "branches"
        / request.branch.branch_id
    )
    steps_dir = branch_dir / "steps"
    steps_dir.mkdir(parents=True, exist_ok=True)
    prefix = f"step{request.step_index:03d}"
    _immutable_text_or_same(steps_dir / f"{prefix}.prompt.txt", generation.prompt)
    _immutable_text_or_same(steps_dir / f"{prefix}.response.txt", generation.text)
    _immutable_json_or_same(
        steps_dir / f"{prefix}.arrival.json",
        {
            "schema_version": 1,
            "branch_id": request.branch.branch_id,
            "arm_id": request.arm.arm_id,
            "role": request.arm.role.value,
            "role_snapshot_id": request.branch.role_snapshot_id,
            "option_id": request.branch.option_id,
            "harness_id": request.branch.harness_id,
            "step_index": request.step_index,
            "parent_state_id": request.parent_state_id,
            "seed": int(
                generation.seed
                if generation.seed is not None
                else request.branch.seed + request.step_index
            ),
            "prompt_hash": content_hash(generation.prompt),
            "response_hash": content_hash(generation.text),
            "token_ids": list(generation.token_ids),
            "log_probabilities": list(generation.log_probabilities),
            "token_mask": (
                list(generation.token_mask)
                if generation.token_mask is not None
                else None
            ),
        },
    )
    prompt_path = (
        Path(run_dir)
        / "artifacts"
        / "prompts"
        / f"{content_hash(generation.prompt)}.txt"
    )
    _immutable_text_or_same(prompt_path, generation.prompt)


def load_generation_arrival(
    *, run_dir: Union[str, Path], request: Any
) -> Optional[GenerationOutcome]:
    """Reuse an already durable response for the same deterministic sample ID."""

    steps_dir = (
        Path(run_dir)
        / f"step{request.branch.epoch:02d}"
        / "branches"
        / request.branch.branch_id
        / "steps"
    )
    prefix = f"step{request.step_index:03d}"
    arrival_path = steps_dir / f"{prefix}.arrival.json"
    prompt_path = steps_dir / f"{prefix}.prompt.txt"
    response_path = steps_dir / f"{prefix}.response.txt"
    if not arrival_path.is_file():
        return None
    if not prompt_path.is_file() or not response_path.is_file():
        raise DurableVerificationConflict("partial durable generation arrival")
    try:
        document = json.loads(arrival_path.read_text(encoding="utf-8"))
        prompt = prompt_path.read_text(encoding="utf-8")
        response = response_path.read_text(encoding="utf-8")
    except (OSError, json.JSONDecodeError, UnicodeError) as exc:
        raise DurableVerificationConflict(
            "durable generation arrival is unreadable"
        ) from exc
    if document.get("schema_version") != 1:
        raise DurableVerificationConflict("unsupported durable generation schema")
    expected = {
        "branch_id": request.branch.branch_id,
        "arm_id": request.arm.arm_id,
        "role": request.arm.role.value,
        "role_snapshot_id": request.branch.role_snapshot_id,
        "option_id": request.branch.option_id,
        "harness_id": request.branch.harness_id,
        "step_index": request.step_index,
        "parent_state_id": request.parent_state_id,
    }
    if any(document.get(key) != value for key, value in expected.items()):
        raise DurableVerificationConflict(
            "durable generation does not match frozen request"
        )
    if document.get("prompt_hash") != content_hash(prompt):
        raise DurableVerificationConflict("durable prompt hash mismatch")
    if document.get("response_hash") != content_hash(response):
        raise DurableVerificationConflict("durable response hash mismatch")
    mask = document.get("token_mask")
    if mask is not None and (
        not isinstance(mask, list)
        or any(not isinstance(item, bool) for item in mask)
    ):
        raise DurableVerificationConflict("durable response token mask is malformed")
    token_ids = document.get("token_ids", ())
    log_probabilities = document.get("log_probabilities", ())
    seed = document.get("seed")
    if not isinstance(token_ids, list) or any(
        isinstance(item, bool) or not isinstance(item, int) or item < 0
        for item in token_ids
    ):
        raise DurableVerificationConflict("durable response token IDs are malformed")
    if not isinstance(log_probabilities, list) or any(
        isinstance(item, bool)
        or not isinstance(item, (int, float))
        or not math.isfinite(float(item))
        for item in log_probabilities
    ):
        raise DurableVerificationConflict(
            "durable response log probabilities are malformed"
        )
    if isinstance(seed, bool) or not isinstance(seed, int) or seed < 0:
        raise DurableVerificationConflict("durable response seed is malformed")
    return GenerationOutcome(
        prompt=prompt,
        text=response,
        token_ids=tuple(token_ids),
        log_probabilities=tuple(float(item) for item in log_probabilities),
        token_mask=(tuple(mask) if mask is not None else None),
        seed=seed,
    )


def persist_verified_step(
    *,
    run_dir: Union[str, Path],
    epoch: int,
    branch_id: str,
    step_index: int,
    result: BranchStepResult,
) -> None:
    """Persist parsed proposal and common-verifier output before returning it."""

    steps_dir = Path(run_dir) / f"step{epoch:02d}" / "branches" / branch_id / "steps"
    steps_dir.mkdir(parents=True, exist_ok=True)
    prefix = f"step{step_index:03d}"
    _immutable_json_or_same(
        steps_dir / f"{prefix}.proposal.json", result.proposal.to_dict()
    )
    _immutable_json_or_same(
        steps_dir / f"{prefix}.evidence.json",
        result.verification.evidence.to_dict(),
    )
    if result.verification.state is not None:
        _immutable_json_or_same(
            steps_dir / f"{prefix}.state.json",
            result.verification.state.to_dict(),
        )
        _immutable_json_or_same(
            steps_dir / f"{prefix}.descriptor.json",
            result.verification.descriptor.to_dict(),
        )


def persist_verifier_trace(
    *,
    run_dir: Union[str, Path],
    result: ScientificVerificationResult,
    phase: str,
) -> Path:
    """Persist the complete bounded-input verifier capture separately.

    ``EvidencePacket.diagnostics`` stays bounded for routine readers. The
    complete capture returned by the verifier is content-addressed here before
    any scheduler, archive, record, or learning consumer can use the result.
    """

    evidence = result.evidence
    destination = (
        Path(run_dir)
        / "logs"
        / "verifiers"
        / f"{evidence.evidence_id}.trace.json"
    )
    document = {
        "schema_version": 1,
        "evidence_id": evidence.evidence_id,
        "proposal_id": evidence.proposal_id,
        "branch_id": evidence.branch_id,
        "phase": str(phase),
        "diagnostics": thaw_json(result.decision.capture.diagnostics),
        "resources": thaw_json(result.decision.capture.resources),
        "started_at": result.decision.capture.started_at,
        "completed_at": result.decision.capture.completed_at,
        "attempt_index": result.decision.capture.attempt_index,
    }
    document["capture_hash"] = content_hash(
        {key: value for key, value in document.items() if key != "capture_hash"}
    )
    _immutable_json_or_same(destination, document)
    return destination


def restore_durable_verification_result(
    *,
    evidence: EvidencePacket,
    adapter: ScientificProblemAdapter,
    state_path: Path,
    descriptor_path: Path,
) -> ScientificVerificationResult:
    """Validate/rebuild files derived from one already durable verifier packet."""

    attempt_index = evidence.flags.get("verification_attempt_index", 0)
    if (
        isinstance(attempt_index, bool)
        or not isinstance(attempt_index, int)
        or attempt_index < 0
    ):
        raise DurableVerificationConflict(
            "durable evidence has an invalid verifier attempt index"
        )
    decision = VerificationDecision(
        failure_kind=evidence.failure_kind,
        resolved=evidence.resolved,
        admitted=evidence.admitted,
        internal_reward=evidence.internal_reward,
        raw_score=evidence.raw_score,
        uncertainty=evidence.uncertainty,
        flags=evidence.flags,
        scores=evidence.scores,
        capture=ExecutionCapture(
            diagnostics=evidence.diagnostics,
            resources=evidence.resources,
            started_at=evidence.started_at,
            completed_at=evidence.completed_at,
            attempt_index=attempt_index,
        ),
    )
    state = None
    descriptor = None
    if evidence.admitted:
        if evidence.scientific_state_id is None:
            raise DurableVerificationConflict(
                "admitted durable evidence has no scientific state identity"
            )
        expected_state = VerifiedScientificState(
            state_id=evidence.scientific_state_id,
            proposal_id=evidence.proposal_id,
            evidence_id=evidence.evidence_id,
            problem_id=evidence.problem_id,
            answer_payload=evidence.answer_payload,
            resolved=evidence.resolved,
            admitted=evidence.admitted,
            confirmed=evidence.confirmed,
            internal_reward=evidence.internal_reward,
            raw_score=evidence.raw_score,
            descriptor_id=evidence.descriptor_id,
            fingerprint=evidence.fingerprint,
        )
        if state_path.is_file():
            try:
                state = VerifiedScientificState.from_dict(
                    json.loads(state_path.read_text(encoding="utf-8"))
                )
            except Exception as exc:
                raise DurableVerificationConflict(
                    "durable scientific state is unreadable or invalid"
                ) from exc
            if state != expected_state:
                raise DurableVerificationConflict(
                    "durable scientific state differs from its evidence"
                )
        else:
            state = expected_state
            _immutable_json_or_same(state_path, state.to_dict())

        from evolve.verifier.evidence import build_descriptor

        try:
            expected_descriptor = build_descriptor(
                problem_id=evidence.problem_id,
                function_version=adapter.descriptor_version,
                dimensions=adapter.describe_scientific_state(
                    evidence.answer_payload, decision
                ),
                method_complete=adapter.method_complete,
            )
        except Exception as exc:
            raise DurableVerificationConflict(
                "durable evidence cannot reconstruct its scientific descriptor"
            ) from exc
        if expected_descriptor.descriptor_id != evidence.descriptor_id:
            raise DurableVerificationConflict(
                "reconstructed descriptor differs from durable evidence"
            )
        if descriptor_path.is_file():
            try:
                descriptor = Descriptor.from_dict(
                    json.loads(descriptor_path.read_text(encoding="utf-8"))
                )
            except Exception as exc:
                raise DurableVerificationConflict(
                    "durable descriptor is unreadable or invalid"
                ) from exc
            if descriptor != expected_descriptor:
                raise DurableVerificationConflict(
                    "durable descriptor differs from its evidence"
                )
        else:
            descriptor = expected_descriptor
            _immutable_json_or_same(descriptor_path, descriptor.to_dict())
        try:
            recovered_fingerprint = adapter.scientific_fingerprint(
                evidence.answer_payload, decision
            )
        except Exception as exc:
            raise DurableVerificationConflict(
                "durable evidence cannot reconstruct its scientific fingerprint"
            ) from exc
        if recovered_fingerprint != evidence.fingerprint:
            raise DurableVerificationConflict(
                "reconstructed fingerprint differs from durable evidence"
            )
    elif state_path.exists() or descriptor_path.exists():
        raise DurableVerificationConflict(
            "rejected durable evidence has scientific state artifacts"
        )
    return ScientificVerificationResult(
        decision=decision,
        evidence=evidence,
        state=state,
        descriptor=descriptor,
    )


def load_verified_step(
    *,
    run_dir: Union[str, Path],
    epoch: int,
    branch_id: str,
    step_index: int,
    parent_state_id: Optional[str],
    generation: Optional[GenerationOutcome],
    adapter: ScientificProblemAdapter,
    run_id: Optional[str] = None,
    problem_id: Optional[str] = None,
    harness_id: Optional[str] = None,
    policy_snapshot_id: Optional[str] = None,
) -> Optional[BranchStepResult]:
    """Restore a completed verifier result without rerunning candidate code."""

    steps_dir = Path(run_dir) / f"step{epoch:02d}" / "branches" / branch_id / "steps"
    prefix = f"step{step_index:03d}"
    proposal_path = steps_dir / f"{prefix}.proposal.json"
    evidence_path = steps_dir / f"{prefix}.evidence.json"
    if not proposal_path.is_file() and not evidence_path.is_file():
        return None
    if not proposal_path.is_file() or not evidence_path.is_file():
        raise DurableVerificationConflict("partial durable verification result")
    try:
        proposal = Proposal.from_dict(
            json.loads(proposal_path.read_text(encoding="utf-8"))
        )
        evidence = EvidencePacket.from_dict(
            json.loads(evidence_path.read_text(encoding="utf-8"))
        )
    except Exception as exc:
        raise DurableVerificationConflict(
            "durable verification result is unreadable or invalid"
        ) from exc
    proposal_identity = {
        "run_id": proposal.run_id,
        "problem_id": proposal.problem_id,
        "branch_id": branch_id,
        "parent_state_id": parent_state_id,
        "step_index": step_index,
        "source_hash": proposal.source_hash,
    }
    expected_proposal_ids = {content_id("proposal", proposal_identity)}
    infrastructure_phase = evidence.flags.get("worker_failure_phase")
    if isinstance(infrastructure_phase, str) and infrastructure_phase:
        expected_proposal_ids.add(
            content_id(
                "proposal",
                {
                    **proposal_identity,
                    "infrastructure_phase": infrastructure_phase,
                },
            )
        )
    if proposal.proposal_id not in expected_proposal_ids:
        raise DurableVerificationConflict(
            "durable proposal has a different sample identity"
        )
    if proposal.branch_id != branch_id or evidence.branch_id != branch_id:
        raise DurableVerificationConflict(
            "durable verification belongs to another branch"
        )
    if proposal.parent_state_id != parent_state_id or evidence.parent_state_id != parent_state_id:
        raise DurableVerificationConflict(
            "durable verification has a different parent state"
        )
    if proposal.proposal_id != evidence.proposal_id:
        raise DurableVerificationConflict(
            "durable proposal/evidence identity mismatch"
        )
    if (
        proposal.run_id != evidence.run_id
        or proposal.problem_id != evidence.problem_id
        or proposal.source_hash != evidence.source_hash
    ):
        raise DurableVerificationConflict(
            "durable proposal/evidence references disagree"
        )
    if run_id is not None and proposal.run_id != run_id:
        raise DurableVerificationConflict(
            "durable verification belongs to another run"
        )
    if problem_id is not None and proposal.problem_id != problem_id:
        raise DurableVerificationConflict(
            "durable verification belongs to another problem"
        )
    if (
        evidence.verifier_id != adapter.verifier_id
        or evidence.verifier_version != adapter.verifier_version
    ):
        raise DurableVerificationConflict(
            "durable verification uses another verifier"
        )
    if harness_id is not None and evidence.harness_id != harness_id:
        raise DurableVerificationConflict(
            "durable verification uses another harness"
        )
    if (
        policy_snapshot_id is not None
        and evidence.policy_snapshot_id != policy_snapshot_id
    ):
        raise DurableVerificationConflict(
            "durable verification uses another policy snapshot"
        )
    if generation is not None and (
        proposal.source_text != generation.text
        or proposal.source_hash != content_hash(generation.text)
    ):
        raise DurableVerificationConflict(
            "durable verification uses another response"
        )
    verification = restore_durable_verification_result(
        evidence=evidence,
        adapter=adapter,
        state_path=steps_dir / f"{prefix}.state.json",
        descriptor_path=steps_dir / f"{prefix}.descriptor.json",
    )
    costs = {key: float(value) for key, value in evidence.resources.items()}
    costs.setdefault("verifier_calls", 1.0)
    attempt_root = (
        Path(run_dir)
        / "artifacts"
        / "verification_attempts"
        / proposal.proposal_id
    )
    if attempt_root.is_dir():
        request_path = attempt_root / "request.json"
        if not request_path.is_file():
            raise DurableVerificationConflict(
                "durable verifier-attempt directory has no frozen request"
            )
        try:
            attempt_request = json.loads(request_path.read_text(encoding="utf-8"))
        except Exception as exc:
            raise DurableVerificationConflict(
                "durable verifier-attempt request is unreadable"
            ) from exc
        expected_attempt_request = {
            "schema_version": 1,
            "proposal_id": proposal.proposal_id,
            "verifier_id": adapter.verifier_id,
            "verifier_version": adapter.verifier_version,
            "harness_id": evidence.harness_id,
            "policy_snapshot_id": evidence.policy_snapshot_id,
        }
        if any(
            attempt_request.get(name) != value
            for name, value in expected_attempt_request.items()
        ):
            raise DurableVerificationConflict(
                "durable verifier-attempt request has another frozen identity"
            )
        maximum_attempts = attempt_request.get("maximum_attempts")
        if (
            isinstance(maximum_attempts, bool)
            or not isinstance(maximum_attempts, int)
            or maximum_attempts < 1
        ):
            raise DurableVerificationConflict(
                "durable verifier-attempt request has an invalid attempt bound"
            )
        completed_attempts = sorted(attempt_root.glob("attempt*.completed.json"))
        if not completed_attempts:
            raise DurableVerificationConflict(
                "durable verifier-attempt directory has no completion marker"
            )
        final_attempt = None
        for expected_index, attempt_path in enumerate(completed_attempts):
            try:
                attempt_document = json.loads(
                    attempt_path.read_text(encoding="utf-8")
                )
            except Exception as exc:
                raise DurableVerificationConflict(
                    "durable verifier attempt marker is unreadable"
                ) from exc
            if (
                attempt_document.get("schema_version") != 1
                or attempt_document.get("attempt_index") != expected_index
            ):
                raise DurableVerificationConflict(
                    "durable verifier attempt sequence is malformed"
                )
            attempt_evidence_path = (
                attempt_root / f"attempt{expected_index:02d}.evidence.json"
            )
            if not attempt_evidence_path.is_file():
                raise DurableVerificationConflict(
                    "durable verifier attempt marker has no evidence packet"
                )
            try:
                marked_evidence = EvidencePacket.from_dict(
                    json.loads(attempt_evidence_path.read_text(encoding="utf-8"))
                )
            except Exception as exc:
                raise DurableVerificationConflict(
                    "durable verifier attempt evidence is unreadable or invalid"
                ) from exc
            marked_identity = {
                "run_id": proposal.run_id,
                "proposal_id": proposal.proposal_id,
                "problem_id": proposal.problem_id,
                "parent_state_id": proposal.parent_state_id,
                "branch_id": proposal.branch_id,
                "source_hash": proposal.source_hash,
                "verifier_id": adapter.verifier_id,
                "verifier_version": adapter.verifier_version,
                "harness_id": evidence.harness_id,
                "policy_snapshot_id": evidence.policy_snapshot_id,
            }
            if any(
                getattr(marked_evidence, name) != value
                for name, value in marked_identity.items()
            ) or marked_evidence.flags.get(
                "verification_attempt_index"
            ) != expected_index:
                raise DurableVerificationConflict(
                    "durable verifier attempt evidence has another frozen identity"
                )
            if attempt_document.get("evidence_id") != marked_evidence.evidence_id:
                raise DurableVerificationConflict(
                    "durable verifier attempt marker/evidence identity mismatch"
                )
            final_attempt = attempt_document
        if len(completed_attempts) > maximum_attempts:
            raise DurableVerificationConflict(
                "durable verifier attempts exceed their frozen bound"
            )
        if final_attempt.get("evidence_id") != evidence.evidence_id:
            raise DurableVerificationConflict(
                "durable step evidence is not the final verifier attempt"
            )
        costs["verifier_calls"] = float(len(completed_attempts))
    policy_segment = None
    if generation is not None and generation.log_probabilities:
        policy_segment = PolicySegment(
            prompt=generation.prompt,
            response_segment=generation.text,
            token_mask=(
                generation.token_mask
                or tuple(True for _ in generation.log_probabilities)
            ),
            log_probabilities=generation.log_probabilities,
            token_ids=generation.token_ids,
        )
        if generation.token_ids:
            costs["tokens"] = float(len(generation.token_ids))
    return BranchStepResult(
        proposal=proposal,
        verification=verification,
        costs=costs,
        policy_segment=policy_segment,
    )


def infrastructure_step_result(
    *,
    run_id: str,
    problem_id: str,
    branch_id: str,
    epoch: int,
    parent_state_id: Optional[str],
    step_index: int,
    adapter: ScientificProblemAdapter,
    verification_policy: VerificationPolicy,
    harness_id: str,
    policy_snapshot_id: str,
    run_dir: Union[str, Path],
    phase: str,
    error: BaseException,
    generation: Optional[GenerationOutcome] = None,
) -> BranchStepResult:
    """Represent a worker failure as unresolved, scheduler-ineligible evidence."""

    source_text = generation.text if generation is not None else ""
    source_hash = content_hash(source_text)
    proposal = Proposal(
        proposal_id=content_id(
            "proposal",
            {
                "run_id": run_id,
                "problem_id": problem_id,
                "branch_id": branch_id,
                "parent_state_id": parent_state_id,
                "step_index": step_index,
                "source_hash": source_hash,
                "infrastructure_phase": phase,
            },
        ),
        run_id=run_id,
        problem_id=problem_id,
        source_text=source_text,
        source_hash=source_hash,
        parent_state_id=parent_state_id,
        branch_id=branch_id,
        parsed_candidate=None,
    )
    artifact_path = persist_answer_artifact(
        run_dir=run_dir, problem_id=problem_id, payload=None
    )
    persisted = PersistedAnswerPayload.create(
        problem_id=problem_id,
        artifact_uri=str(artifact_path),
        payload=None,
    )
    decision = VerificationDecision.failure(
        classify_failure(infrastructure_error=True),
        flags={"worker_failure_phase": phase},
        capture=ExecutionCapture(
            diagnostics=FrozenDict(
                {
                    "worker_error": {
                        "exception_type": type(error).__name__,
                        "message": str(error)[:2048],
                        "phase": phase,
                    }
                }
            ),
            resources=FrozenDict({"verifier_calls": 0.0}),
            attempt_index=0,
        ),
    )
    verification = build_verification_result(
        proposal=proposal,
        persisted_answer=persisted,
        decision=decision,
        verification_policy=verification_policy,
        verifier_id=adapter.verifier_id,
        verifier_version=adapter.verifier_version,
        harness_id=harness_id,
        policy_snapshot_id=policy_snapshot_id,
        timeout_is_scientific=adapter.timeout_is_scientific,
        method_complete=adapter.method_complete,
    )
    persist_verifier_trace(
        run_dir=run_dir,
        result=verification,
        phase=phase,
    )
    result = BranchStepResult(
        proposal=proposal,
        verification=verification,
        costs={"verifier_calls": 0.0},
    )
    persist_verified_step(
        run_dir=run_dir,
        epoch=epoch,
        branch_id=branch_id,
        step_index=step_index,
        result=result,
    )
    return result


def build_proposal_and_verify(
    *,
    run_id: str,
    problem_id: str,
    branch_id: str,
    parent_state_id: Optional[str],
    step_index: int,
    generation: GenerationOutcome,
    extract_answer: Callable[[str], Any],
    adapter: ScientificProblemAdapter,
    verification_policy: VerificationPolicy,
    harness_id: str,
    policy_snapshot_id: str,
    run_dir: Union[str, Path],
) -> BranchStepResult:
    """Parse, persist, and independently verify one generated response.

    Never raises for a scientific/parse/code/constraint failure -- those
    become typed, resolved, non-admitted evidence via the common verifier.
    An exception raised here (persistence, adapter construction) is a real
    infrastructure error and propagates as :class:`VerificationWorkerError`.
    """

    try:
        source_text = generation.text
        source_hash = content_hash(source_text)
        proposal_id = content_id(
            "proposal",
            {
                "run_id": run_id,
                "problem_id": problem_id,
                "branch_id": branch_id,
                "parent_state_id": parent_state_id,
                "step_index": step_index,
                "source_hash": source_hash,
            },
        )
        # The problem-facing extraction callback returns ``None`` for ordinary
        # parse/code/constraint failures. An exception means the sandbox,
        # evaluator, or serialization boundary itself failed and must propagate
        # as explicit infrastructure evidence rather than being disguised as a
        # low scientific result.
        payload = extract_answer(source_text)
        proposal = Proposal(
            proposal_id=proposal_id,
            run_id=run_id,
            problem_id=problem_id,
            source_text=source_text,
            source_hash=source_hash,
            parent_state_id=parent_state_id,
            branch_id=branch_id,
            parsed_candidate=payload,
            created_at=_utc_now(),
        )
        artifact_path = persist_answer_artifact(run_dir=run_dir, problem_id=problem_id, payload=payload)
        persisted = PersistedAnswerPayload.create(
            problem_id=problem_id,
            artifact_uri=str(artifact_path),
            payload=payload,
        )
        parsed_path = (
            Path(run_dir)
            / "artifacts"
            / "parsed"
            / f"{proposal.proposal_id}.json"
        )
        if parsed_path.is_file():
            try:
                durable_proposal = Proposal.from_dict(
                    json.loads(parsed_path.read_text(encoding="utf-8"))
                )
            except Exception as exc:
                raise DurableVerificationConflict(
                    f"durable parsed proposal is unreadable or invalid: {parsed_path}"
                ) from exc
            comparable_fields = (
                "proposal_id",
                "run_id",
                "problem_id",
                "source_text",
                "source_hash",
                "parent_state_id",
                "branch_id",
                "parsed_candidate",
            )
            if any(
                getattr(durable_proposal, name) != getattr(proposal, name)
                for name in comparable_fields
            ):
                raise DurableVerificationConflict(
                    f"durable parsed proposal identity conflict: {parsed_path}"
                )
            proposal = durable_proposal
        else:
            _immutable_json_or_same(parsed_path, proposal.to_dict())
    except VerificationWorkerError:
        raise
    except Exception as exc:
        raise VerificationWorkerError(
            f"could not persist a verifiable candidate: {type(exc).__name__}: {exc}"
        ) from exc

    attempt_root = (
        Path(run_dir)
        / "artifacts"
        / "verification_attempts"
        / proposal.proposal_id
    )
    _immutable_json_or_same(
        attempt_root / "request.json",
        {
            "schema_version": 1,
            "proposal_id": proposal.proposal_id,
            "verifier_id": adapter.verifier_id,
            "verifier_version": adapter.verifier_version,
            "harness_id": harness_id,
            "policy_snapshot_id": policy_snapshot_id,
            "maximum_attempts": (
                verification_policy.infrastructure_retry_limit + 1
            ),
        },
    )
    result = None
    verifier_calls = 0
    for attempt_index in range(
        verification_policy.infrastructure_retry_limit + 1
    ):
        prefix = f"attempt{attempt_index:02d}"
        evidence_path = attempt_root / f"{prefix}.evidence.json"
        state_path = attempt_root / f"{prefix}.state.json"
        descriptor_path = attempt_root / f"{prefix}.descriptor.json"
        completed_path = attempt_root / f"{prefix}.completed.json"
        if evidence_path.is_file():
            try:
                attempt_evidence = EvidencePacket.from_dict(
                    json.loads(evidence_path.read_text(encoding="utf-8"))
                )
            except Exception as exc:
                raise DurableVerificationConflict(
                    "durable verifier attempt evidence is unreadable or invalid"
                ) from exc
            exact = {
                "run_id": proposal.run_id,
                "proposal_id": proposal.proposal_id,
                "problem_id": proposal.problem_id,
                "parent_state_id": proposal.parent_state_id,
                "branch_id": proposal.branch_id,
                "source_hash": proposal.source_hash,
                "verifier_id": adapter.verifier_id,
                "verifier_version": adapter.verifier_version,
                "harness_id": harness_id,
                "policy_snapshot_id": policy_snapshot_id,
            }
            if any(
                getattr(attempt_evidence, name) != value
                for name, value in exact.items()
            ):
                raise DurableVerificationConflict(
                    "durable verifier attempt has another frozen identity"
                )
            if (
                attempt_evidence.flags.get("verification_attempt_index")
                != attempt_index
            ):
                raise DurableVerificationConflict(
                    "durable verifier attempt index mismatch"
                )
            result = restore_durable_verification_result(
                evidence=attempt_evidence,
                adapter=adapter,
                state_path=state_path,
                descriptor_path=descriptor_path,
            )
        else:
            if state_path.exists() or descriptor_path.exists() or completed_path.exists():
                raise DurableVerificationConflict(
                    "partial verifier attempt exists without durable evidence"
                )
            result = verify_persisted_answer(
                adapter=adapter,
                proposal=proposal,
                persisted_answer=persisted,
                verification_policy=verification_policy,
                harness_id=harness_id,
                policy_snapshot_id=policy_snapshot_id,
                attempt_index=attempt_index,
            )
            persist_verifier_trace(
                run_dir=run_dir,
                result=result,
                phase=f"independent_verification_attempt_{attempt_index}",
            )
            _immutable_json_or_same(evidence_path, result.evidence.to_dict())
            if result.state is not None:
                _immutable_json_or_same(state_path, result.state.to_dict())
                _immutable_json_or_same(
                    descriptor_path, result.descriptor.to_dict()
                )
        verifier_calls = attempt_index + 1
        _immutable_json_or_same(
            completed_path,
            {
                "schema_version": 1,
                "attempt_index": attempt_index,
                "evidence_id": result.evidence.evidence_id,
                "resolved": result.evidence.resolved,
            },
        )
        if result.evidence.resolved:
            break
    if result is None:
        raise VerificationWorkerError("verifier produced no durable attempt")

    costs = {"verifier_calls": float(verifier_calls)}
    if generation.token_ids:
        costs["tokens"] = float(len(generation.token_ids))
    for resource, amount in result.evidence.resources.items():
        if resource == "verifier_calls":
            continue
        costs[resource] = float(amount)

    policy_segment = None
    if generation.log_probabilities:
        mask = generation.token_mask or tuple(True for _ in generation.log_probabilities)
        policy_segment = PolicySegment(
            prompt=generation.prompt,
            response_segment=generation.text,
            token_mask=tuple(mask),
            log_probabilities=tuple(generation.log_probabilities),
            token_ids=tuple(generation.token_ids),
        )

    return BranchStepResult(
        proposal=proposal,
        verification=result,
        costs=costs,
        policy_segment=policy_segment,
    )


T = TypeVar("T")


class BackpressureQueue:
    """A bounded, thread-safe queue that throttles a fast producer.

    Used between generation and verification/persistence: ``put`` blocks (up
    to an optional timeout) once the queue is full instead of buffering
    unboundedly, matching "backpressure generation when verifier or
    persistence queues are full."
    """

    def __init__(self, maxsize: int) -> None:
        if isinstance(maxsize, bool) or not isinstance(maxsize, int) or maxsize < 1:
            raise ValueError("maxsize must be a positive integer")
        self._queue: "queue.Queue" = queue.Queue(maxsize=maxsize)
        self._closed = threading.Event()

    def put(self, item: T, *, timeout: Optional[float] = None) -> None:
        if self._closed.is_set():
            raise VerificationWorkerError("cannot put into a closed BackpressureQueue")
        self._queue.put(item, timeout=timeout)

    def drain(self, handler: Callable[[T], None], *, idle_sleep_s: float = 0.01) -> None:
        """Process items until the queue is closed and empty."""

        while True:
            try:
                item = self._queue.get(timeout=idle_sleep_s)
            except queue.Empty:
                if self._closed.is_set() and self._queue.empty():
                    return
                continue
            try:
                handler(item)
            finally:
                self._queue.task_done()

    def close(self) -> None:
        self._closed.set()

    def join(self) -> None:
        self._queue.join()

    @property
    def approximate_size(self) -> int:
        return self._queue.qsize()


__all__ = [
    "BackpressureQueue",
    "DurableVerificationConflict",
    "GenerationOutcome",
    "VerificationWorkerError",
    "build_proposal_and_verify",
    "infrastructure_step_result",
    "load_generation_arrival",
    "load_verified_step",
    "persist_answer_artifact",
    "persist_generation_arrival",
    "persist_verifier_trace",
    "persist_verified_step",
    "restore_durable_verification_result",
]
