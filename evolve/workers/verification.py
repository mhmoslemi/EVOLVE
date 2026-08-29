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
import threading
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable, Optional, Tuple, TypeVar, Union

from evolve.ids import content_hash, content_id
from evolve.options.branch import BranchStepResult, PolicySegment
from evolve.runio.atomic import ImmutableWriteError, write_immutable_json
from evolve.types import Proposal
from evolve.verifier.adapters import ScientificProblemAdapter
from evolve.verifier.models import PersistedAnswerPayload, VerificationPolicy
from evolve.verifier.service import verify_persisted_answer


class VerificationWorkerError(RuntimeError):
    """A genuine infrastructure failure occurred outside adapter verification."""


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
    if destination.exists():
        return destination
    try:
        write_immutable_json(destination, payload)
    except ImmutableWriteError:
        pass  # a concurrent writer already persisted this exact content
    return destination


@dataclass(frozen=True)
class GenerationOutcome:
    """What a live generation call actually produced for one request."""

    prompt: str
    text: str
    token_ids: Tuple[int, ...] = ()
    log_probabilities: Tuple[float, ...] = ()
    token_mask: Optional[Tuple[bool, ...]] = None


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
        try:
            payload = extract_answer(source_text)
        except Exception:
            # Extraction is a resolved scientific parse/code failure, not an
            # infrastructure failure in the persistence service.
            payload = None
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
    except VerificationWorkerError:
        raise
    except Exception as exc:
        raise VerificationWorkerError(
            f"could not persist a verifiable candidate: {type(exc).__name__}: {exc}"
        ) from exc

    result = verify_persisted_answer(
        adapter=adapter,
        proposal=proposal,
        persisted_answer=persisted,
        verification_policy=verification_policy,
        harness_id=harness_id,
        policy_snapshot_id=policy_snapshot_id,
    )

    costs = {"verifier_calls": 1.0}
    if generation.token_ids:
        costs["tokens"] = float(len(generation.token_ids))
    for resource, amount in result.evidence.resources.items():
        costs[resource] = float(amount)

    policy_segment = None
    if generation.log_probabilities:
        mask = generation.token_mask or tuple(True for _ in generation.log_probabilities)
        policy_segment = PolicySegment(
            prompt=generation.prompt,
            response_segment=generation.text,
            token_mask=tuple(mask),
            log_probabilities=tuple(generation.log_probabilities),
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
    "GenerationOutcome",
    "VerificationWorkerError",
    "build_proposal_and_verify",
    "persist_answer_artifact",
]
