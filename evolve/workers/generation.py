"""Role-isolated, topology-independent generation job contracts.

This module owns EVOLVE's exact-count sharding policy. It deliberately contains no model or GPU
imports.  EVOLVE controllers persist :class:`GenerationJob` objects before
dispatch, expand them into globally unique per-sample requests, and only then
hand homogeneous adapter batches to a backend worker.

The records and executor key every sample by logical identity and surface all
incomplete generation as explicit infrastructure failures.
"""

from __future__ import annotations

import hashlib
import os
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Callable, Dict, Iterable, Mapping, Optional, Sequence, Tuple

from evolve.ids import content_hash, content_id, derive_seed, rollout_seed, validate_id
from evolve.roles import RoleIsolationError, validate_role_snapshot_identity
from evolve.types import FrozenDict, Role, RoleSnapshot


GENERATION_JOB_SCHEMA_VERSION = 1
GENERATION_PARAMETERS_VERSION = "generation_parameters_v1"
MIN_PRODUCTION_ROLE_CAPACITY = 3
_MAX_SEED = (1 << 63) - 1


def _distribute_jobs(prompts, counts, workers):
    shards = [[] for _ in range(workers)]
    for group_index, (prompt, count) in enumerate(zip(prompts, counts)):
        base, remainder = divmod(int(count), workers)
        for worker in range(workers):
            assigned = base + (1 if worker < remainder else 0)
            if assigned:
                shards[worker].append((group_index, prompt, assigned))
    return shards


class GenerationContractError(ValueError):
    """A generation job or worker result violates the frozen contract."""


class GenerationOutOfMemory(RuntimeError):
    """A backend reports a retryable generation OOM for the current chunk."""


class GenerationBackend(str, Enum):
    HF = "hf"
    VLLM = "vllm"


def _positive_int(value: Any, field_name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
        raise GenerationContractError(f"{field_name} must be a positive integer")
    return value


def _nonnegative_int(value: Any, field_name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        raise GenerationContractError(
            f"{field_name} must be a non-negative integer"
        )
    return value


def _finite_float(value: Any, field_name: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise GenerationContractError(f"{field_name} must be numeric")
    number = float(value)
    if number != number or number in (float("inf"), float("-inf")):
        raise GenerationContractError(f"{field_name} must be finite")
    return number


def _normalized_adapter_path(value: Any) -> str:
    try:
        raw = os.fspath(value)
    except TypeError as exc:
        raise GenerationContractError("adapter_path must be path-like") from exc
    if not isinstance(raw, str) or not raw.strip():
        raise GenerationContractError("adapter_path must be a non-empty path")
    if not os.path.isabs(raw):
        raise GenerationContractError("adapter_path must be absolute")
    normalized = os.path.normpath(raw)
    if normalized != raw:
        raise GenerationContractError("adapter_path must already be normalized")
    return normalized


@dataclass(frozen=True)
class GenerationParameters:
    """Frozen model sampling parameters for one logical generation job."""

    max_new_tokens: int
    temperature: float
    top_p: float
    micro_batch: int = 0
    stop: Tuple[str, ...] = ()
    extras: FrozenDict = field(default_factory=FrozenDict)
    version: str = GENERATION_PARAMETERS_VERSION
    schema_version: int = GENERATION_JOB_SCHEMA_VERSION
    extensions: FrozenDict = field(default_factory=FrozenDict)

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> "GenerationParameters":
        data, extensions = _read_persisted_mapping(
            payload,
            record_type="generation_parameters",
            known_fields={
                "max_new_tokens",
                "temperature",
                "top_p",
                "micro_batch",
                "stop",
                "extras",
                "version",
                "schema_version",
                "extensions",
            },
        )
        data.pop("schema_version")
        data.pop("record_type")
        data.pop("extensions", None)
        data["stop"] = tuple(data.get("stop", ()))
        data["extensions"] = extensions
        try:
            return cls(**data)
        except (TypeError, ValueError) as exc:
            if isinstance(exc, GenerationContractError):
                raise
            raise GenerationContractError(
                f"invalid persisted generation parameters: {exc}"
            ) from exc

    def __post_init__(self) -> None:
        if self.schema_version != GENERATION_JOB_SCHEMA_VERSION:
            _reject_schema(self.schema_version, "generation parameters")
        _positive_int(self.max_new_tokens, "max_new_tokens")
        temperature = _finite_float(self.temperature, "temperature")
        if temperature < 0.0:
            raise GenerationContractError("temperature must be non-negative")
        top_p = _finite_float(self.top_p, "top_p")
        if not 0.0 < top_p <= 1.0:
            raise GenerationContractError("top_p must be in (0, 1]")
        _nonnegative_int(self.micro_batch, "micro_batch")
        if not isinstance(self.version, str) or not self.version.strip():
            raise GenerationContractError("generation parameter version is required")
        if not isinstance(self.extras, FrozenDict):
            object.__setattr__(self, "extras", FrozenDict(self.extras))
        if not isinstance(self.extensions, FrozenDict):
            object.__setattr__(self, "extensions", FrozenDict(self.extensions))
        stop = tuple(self.stop)
        if any(not isinstance(item, str) or not item for item in stop):
            raise GenerationContractError("stop entries must be non-empty strings")
        if len(set(stop)) != len(stop):
            raise GenerationContractError("stop entries must be unique")
        object.__setattr__(self, "stop", stop)
        reserved = {
            "max_new_tokens", "temperature", "top_p", "micro_batch", "stop",
        }
        overlap = reserved.intersection(self.extras)
        if overlap:
            raise GenerationContractError(
                "generation extras cannot shadow typed fields: "
                + ", ".join(sorted(overlap))
            )

    def identity_payload(self) -> dict:
        return {
            "schema_version": self.schema_version,
            "version": self.version,
            "max_new_tokens": self.max_new_tokens,
            "temperature": float(self.temperature),
            "top_p": float(self.top_p),
            "micro_batch": self.micro_batch,
            "stop": list(self.stop),
            "extras": dict(self.extras),
        }

    def to_dict(self) -> dict:
        payload = self.identity_payload()
        payload.update(
            {
                "record_type": "generation_parameters",
                "extensions": dict(self.extensions),
            }
        )
        return payload


@dataclass(frozen=True)
class GenerationJob:
    """One persisted logical policy decision requesting exact samples.

    ``adapter_path`` is execution metadata and is intentionally excluded from
    ``job_id``.  The frozen snapshot's content hash defines adapter behaviour,
    so relocating an otherwise identical run does not change logical rollout
    identity.  :class:`AdapterRequestRegistry` prevents a live process from
    binding one snapshot to multiple paths or one path to multiple snapshots.
    """

    job_id: str
    run_id: str
    epoch: int
    allocation_id: str
    branch_id: str
    branch_step: int
    role: Role
    adapter_path: str
    adapter_id: str
    adapter_version: str
    policy_snapshot: RoleSnapshot
    option_id: str
    harness_id: str
    prompt: str
    sample_index_start: int
    sample_count: int
    generation_parameters: GenerationParameters
    seed: int
    schema_version: int = GENERATION_JOB_SCHEMA_VERSION
    extensions: FrozenDict = field(default_factory=FrozenDict)

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> "GenerationJob":
        known = {
            "job_id",
            "run_id",
            "epoch",
            "allocation_id",
            "branch_id",
            "branch_step",
            "role",
            "adapter_path",
            "adapter_id",
            "adapter_version",
            "policy_snapshot",
            "option_id",
            "harness_id",
            "prompt",
            "sample_index_start",
            "sample_count",
            "generation_parameters",
            "seed",
            "schema_version",
            "extensions",
        }
        data, extensions = _read_persisted_mapping(
            payload,
            record_type="generation_job",
            known_fields=known,
        )
        data.pop("record_type")
        data.pop("extensions", None)
        try:
            data["role"] = Role(data["role"])
            data["policy_snapshot"] = RoleSnapshot.from_dict(data["policy_snapshot"])
            data["generation_parameters"] = GenerationParameters.from_dict(
                data["generation_parameters"]
            )
            data["extensions"] = extensions
            return cls(**data)
        except (KeyError, TypeError, ValueError) as exc:
            if isinstance(exc, GenerationContractError):
                raise
            raise GenerationContractError(f"invalid persisted generation job: {exc}") from exc

    @classmethod
    def create(
        cls,
        *,
        run_id: str,
        epoch: int,
        allocation_id: str,
        branch_id: str,
        branch_step: int,
        role: Role,
        adapter_path: str,
        adapter_id: str,
        adapter_version: str,
        policy_snapshot: RoleSnapshot,
        option_id: str,
        harness_id: str,
        prompt: str,
        sample_index_start: int,
        sample_count: int,
        generation_parameters: GenerationParameters,
    ) -> "GenerationJob":
        normalized_role = role if isinstance(role, Role) else Role(role)
        fields = {
            "schema_version": GENERATION_JOB_SCHEMA_VERSION,
            "run_id": run_id,
            "epoch": epoch,
            "allocation_id": allocation_id,
            "branch_id": branch_id,
            "branch_step": branch_step,
            "role": normalized_role.value,
            "adapter_id": adapter_id,
            "adapter_version": adapter_version,
            "policy_snapshot": policy_snapshot.to_dict(),
            "option_id": option_id,
            "harness_id": harness_id,
            "prompt": prompt,
            "sample_index_start": sample_index_start,
            "sample_count": sample_count,
            "generation_parameters": generation_parameters.identity_payload(),
        }
        job_id = content_id("generation_job", fields)
        seed = derive_seed(
            "generation_job_seed_v1",
            job_id,
            base_seed=policy_snapshot.rng_seed,
        )
        return cls(
            job_id=job_id,
            run_id=run_id,
            epoch=epoch,
            allocation_id=allocation_id,
            branch_id=branch_id,
            branch_step=branch_step,
            role=normalized_role,
            adapter_path=adapter_path,
            adapter_id=adapter_id,
            adapter_version=adapter_version,
            policy_snapshot=policy_snapshot,
            option_id=option_id,
            harness_id=harness_id,
            prompt=prompt,
            sample_index_start=sample_index_start,
            sample_count=sample_count,
            generation_parameters=generation_parameters,
            seed=seed,
        )

    def identity_payload(self) -> dict:
        return {
            "schema_version": self.schema_version,
            "run_id": self.run_id,
            "epoch": self.epoch,
            "allocation_id": self.allocation_id,
            "branch_id": self.branch_id,
            "branch_step": self.branch_step,
            "role": self.role.value,
            "adapter_id": self.adapter_id,
            "adapter_version": self.adapter_version,
            "policy_snapshot": self.policy_snapshot.to_dict(),
            "option_id": self.option_id,
            "harness_id": self.harness_id,
            "prompt": self.prompt,
            "sample_index_start": self.sample_index_start,
            "sample_count": self.sample_count,
            "generation_parameters": self.generation_parameters.identity_payload(),
        }

    def __post_init__(self) -> None:
        if self.schema_version != GENERATION_JOB_SCHEMA_VERSION:
            _reject_schema(self.schema_version, "generation job")
        try:
            validate_id(self.job_id, "generation_job")
            validate_id(self.run_id, "run")
            validate_id(self.allocation_id, "arm")
            validate_id(self.branch_id, "branch")
            validate_id(self.adapter_id, "adapter")
            validate_id(self.option_id, "option")
            validate_id(self.harness_id, "harness")
        except (TypeError, ValueError) as exc:
            raise GenerationContractError(str(exc)) from exc
        _nonnegative_int(self.epoch, "epoch")
        _nonnegative_int(self.branch_step, "branch_step")
        _nonnegative_int(self.sample_index_start, "sample_index_start")
        _positive_int(self.sample_count, "sample_count")
        _nonnegative_int(self.seed, "seed")
        if self.seed > _MAX_SEED:
            raise GenerationContractError("seed must fit a signed 63-bit integer")
        normalized_role = self.role if isinstance(self.role, Role) else Role(self.role)
        object.__setattr__(self, "role", normalized_role)
        _normalized_adapter_path(self.adapter_path)
        if not isinstance(self.adapter_version, str) or not self.adapter_version.strip():
            raise GenerationContractError("adapter_version must be non-empty")
        if not isinstance(self.prompt, str) or not self.prompt:
            raise GenerationContractError("prompt must be a non-empty rendered prompt")
        if not isinstance(self.generation_parameters, GenerationParameters):
            raise GenerationContractError(
                "generation_parameters must be GenerationParameters"
            )
        if not isinstance(self.extensions, FrozenDict):
            object.__setattr__(self, "extensions", FrozenDict(self.extensions))
        snapshot = self.policy_snapshot
        if not isinstance(snapshot, RoleSnapshot):
            raise GenerationContractError("policy_snapshot must be RoleSnapshot")
        try:
            validate_role_snapshot_identity(snapshot)
        except RoleIsolationError as exc:
            raise GenerationContractError(
                f"invalid frozen policy snapshot: {exc}"
            ) from exc
        if snapshot.run_id != self.run_id or snapshot.epoch != self.epoch:
            raise GenerationContractError(
                "policy snapshot run and epoch must match the generation job"
            )
        if snapshot.role != self.role:
            raise GenerationContractError("job role must match policy snapshot role")
        if snapshot.adapter_id != self.adapter_id:
            raise GenerationContractError(
                "job adapter_id must match policy snapshot adapter_id"
            )
        if snapshot.adapter_version != self.adapter_version:
            raise GenerationContractError(
                "job adapter_version must match policy snapshot adapter_version"
            )
        expected_id = content_id("generation_job", self.identity_payload())
        if self.job_id != expected_id:
            raise GenerationContractError(
                "job_id must cover the complete frozen generation identity"
            )
        expected_seed = derive_seed(
            "generation_job_seed_v1",
            self.job_id,
            base_seed=snapshot.rng_seed,
        )
        if self.seed != expected_seed:
            raise GenerationContractError(
                "generation job seed must be derived from its frozen identity"
            )

    @property
    def policy_snapshot_id(self) -> str:
        return self.policy_snapshot.snapshot_id

    def to_dict(self) -> dict:
        data = self.identity_payload()
        data.update(
            {
                "record_type": "generation_job",
                "job_id": self.job_id,
                "adapter_path": self.adapter_path,
                "generation_parameters": self.generation_parameters.to_dict(),
                "seed": self.seed,
                "extensions": dict(self.extensions),
            }
        )
        return data


def _reject_schema(version: Any, record_name: str) -> None:
    if isinstance(version, bool) or not isinstance(version, int):
        raise GenerationContractError(f"{record_name} schema_version must be an integer")
    if version > GENERATION_JOB_SCHEMA_VERSION:
        raise GenerationContractError(
            f"{record_name} schema {version} is newer than supported schema "
            f"{GENERATION_JOB_SCHEMA_VERSION}"
        )
    raise GenerationContractError(f"unsupported {record_name} schema {version}")


def _read_persisted_mapping(
    payload: Mapping[str, Any], *, record_type: str, known_fields: set
) -> Tuple[dict, FrozenDict]:
    if not isinstance(payload, Mapping):
        raise GenerationContractError(f"{record_type} payload must be a mapping")
    data = dict(payload)
    if data.get("record_type") != record_type:
        raise GenerationContractError(
            f"expected record_type={record_type!r}, got {data.get('record_type')!r}"
        )
    if "schema_version" not in data:
        raise GenerationContractError("persisted record is missing schema_version")
    version = data["schema_version"]
    if version != GENERATION_JOB_SCHEMA_VERSION:
        _reject_schema(version, record_type.replace("_", " "))
    extension_payload = data.get("extensions", {})
    if not isinstance(extension_payload, Mapping):
        raise GenerationContractError("extensions must be a mapping")
    extensions = dict(extension_payload)
    conflicts = sorted(key for key in extensions if key in known_fields)
    if conflicts:
        raise GenerationContractError(
            "extensions cannot shadow current fields: " + ", ".join(conflicts)
        )
    for key in tuple(data):
        if key not in known_fields and key != "record_type":
            value = data.pop(key)
            if key in extensions and extensions[key] != value:
                raise GenerationContractError(
                    f"unknown field {key!r} conflicts with extensions entry"
                )
            extensions[key] = value
    return data, FrozenDict(extensions)


@dataclass(frozen=True)
class GenerationRequest:
    """One topology-independent sample request expanded from a job."""

    request_id: str
    hf_request_id: str
    vllm_request_id: str
    job: GenerationJob
    sample_index: int
    seed: int

    @classmethod
    def from_job(cls, job: GenerationJob, sample_index: int) -> "GenerationRequest":
        fields = _generation_request_fields(job, sample_index)
        return cls(job=job, sample_index=sample_index, **fields)

    def __post_init__(self) -> None:
        try:
            validate_id(self.request_id, "generation_request")
            validate_id(self.hf_request_id, "hf_request")
            validate_id(self.vllm_request_id, "vllm_request")
        except (TypeError, ValueError) as exc:
            raise GenerationContractError(str(exc)) from exc
        _nonnegative_int(self.sample_index, "sample_index")
        _nonnegative_int(self.seed, "seed")
        expected = _generation_request_fields(self.job, self.sample_index)
        for field_name in ("request_id", "hf_request_id", "vllm_request_id", "seed"):
            if getattr(self, field_name) != expected[field_name]:
                raise GenerationContractError(
                    f"{field_name} does not match the logical sample identity"
                )

    def backend_request_id(self, backend: GenerationBackend) -> str:
        normalized = backend if isinstance(backend, GenerationBackend) else GenerationBackend(backend)
        return self.hf_request_id if normalized == GenerationBackend.HF else self.vllm_request_id


def _generation_request_fields(job: GenerationJob, sample_index: int) -> dict:
    if not (
        job.sample_index_start
        <= sample_index
        < job.sample_index_start + job.sample_count
    ):
        raise GenerationContractError("sample_index is outside the job range")
    payload = {"job_id": job.job_id, "sample_index": sample_index}
    request_id = content_id("generation_request", payload)
    seed = rollout_seed(
        run_id=job.run_id,
        epoch=job.epoch,
        allocation_id=job.allocation_id,
        branch_step=job.branch_step,
        sample_index=sample_index,
        role=job.role.value,
        base_seed=job.policy_snapshot.rng_seed,
    )
    return {
        "request_id": request_id,
        "hf_request_id": content_id(
            "hf_request", {"generation_request_id": request_id}
        ),
        "vllm_request_id": content_id(
            "vllm_request", {"generation_request_id": request_id}
        ),
        "seed": seed,
    }


def requests_for_job(job: GenerationJob) -> Tuple[GenerationRequest, ...]:
    return tuple(
        GenerationRequest.from_job(job, sample_index)
        for sample_index in range(
            job.sample_index_start, job.sample_index_start + job.sample_count
        )
    )


@dataclass(frozen=True)
class GenerationBatch:
    """A backend batch proven homogeneous in role and policy snapshot."""

    batch_id: str
    jobs: Tuple[GenerationJob, ...]

    @classmethod
    def create(cls, jobs: Sequence[GenerationJob]) -> "GenerationBatch":
        frozen_jobs = tuple(jobs)
        if not frozen_jobs:
            raise GenerationContractError("a generation batch cannot be empty")
        return cls(batch_id=_generation_batch_id(frozen_jobs), jobs=frozen_jobs)

    def __post_init__(self) -> None:
        try:
            validate_id(self.batch_id, "generation_batch")
        except (TypeError, ValueError) as exc:
            raise GenerationContractError(str(exc)) from exc
        if not self.jobs:
            raise GenerationContractError("a generation batch cannot be empty")
        first = self.jobs[0]
        binding = _batch_binding_key(first)
        job_ids = set()
        logical_samples = set()
        for job in self.jobs:
            if _batch_binding_key(job) != binding:
                raise GenerationContractError(
                    "generation batches cannot mix roles, snapshots, or adapters"
                )
            if job.job_id in job_ids:
                raise GenerationContractError("duplicate generation job in batch")
            job_ids.add(job.job_id)
            for sample_index in range(
                job.sample_index_start, job.sample_index_start + job.sample_count
            ):
                logical_position = (
                    job.run_id,
                    job.epoch,
                    job.allocation_id,
                    job.branch_step,
                    job.role.value,
                    sample_index,
                )
                if logical_position in logical_samples:
                    raise GenerationContractError(
                        "generation jobs overlap a logical rollout sample"
                    )
                logical_samples.add(logical_position)
        expected = _generation_batch_id(self.jobs)
        if self.batch_id != expected:
            raise GenerationContractError("batch_id does not cover all batch jobs")

    @property
    def role(self) -> Role:
        return self.jobs[0].role

    @property
    def policy_snapshot(self) -> RoleSnapshot:
        return self.jobs[0].policy_snapshot

    @property
    def adapter_path(self) -> str:
        return self.jobs[0].adapter_path

    @property
    def sample_count(self) -> int:
        return sum(job.sample_count for job in self.jobs)


def _generation_batch_id(jobs: Sequence[GenerationJob]) -> str:
    first = jobs[0]
    payload = {
        "policy_snapshot_id": first.policy_snapshot_id,
        "adapter_id": first.adapter_id,
        "adapter_version": first.adapter_version,
        "role": first.role.value,
        "job_ids": sorted(job.job_id for job in jobs),
    }
    return content_id("generation_batch", payload)


def _batch_binding_key(job: GenerationJob) -> tuple:
    return (
        job.role.value,
        job.policy_snapshot_id,
        job.adapter_id,
        job.adapter_version,
        job.policy_snapshot.adapter_hash,
        os.path.realpath(job.adapter_path),
    )


def partition_generation_jobs(
    jobs: Iterable[GenerationJob],
) -> Tuple[GenerationBatch, ...]:
    """Deterministically partition jobs without ever mixing role policies."""

    groups: Dict[tuple, list] = {}
    seen_job_ids = set()
    seen_positions = set()
    for job in jobs:
        if not isinstance(job, GenerationJob):
            raise GenerationContractError("all generation jobs must be typed")
        if job.job_id in seen_job_ids:
            raise GenerationContractError("generation jobs must be globally unique")
        seen_job_ids.add(job.job_id)
        for sample_index in range(
            job.sample_index_start, job.sample_index_start + job.sample_count
        ):
            position = (
                job.run_id,
                job.epoch,
                job.allocation_id,
                job.branch_step,
                job.role.value,
                sample_index,
            )
            if position in seen_positions:
                raise GenerationContractError(
                    "generation jobs overlap a logical rollout sample"
                )
            seen_positions.add(position)
        groups.setdefault(_batch_binding_key(job), []).append(job)
    return tuple(
        GenerationBatch.create(tuple(sorted(groups[key], key=lambda item: item.job_id)))
        for key in sorted(groups)
    )


@dataclass(frozen=True)
class WorkerShard:
    """One worker's exact subset of a homogeneous batch."""

    worker_slot: int
    batch: GenerationBatch
    requests: Tuple[GenerationRequest, ...]

    def __post_init__(self) -> None:
        _nonnegative_int(self.worker_slot, "worker_slot")
        allowed = {job.job_id for job in self.batch.jobs}
        request_ids = set()
        for request in self.requests:
            if request.job.job_id not in allowed:
                raise GenerationContractError("worker shard request is outside its batch")
            if request.request_id in request_ids:
                raise GenerationContractError("duplicate request in worker shard")
            request_ids.add(request.request_id)


def distribute_generation_jobs(
    batch: GenerationBatch, num_workers: int
) -> Tuple[WorkerShard, ...]:
    """Apply exact-count sharding while preserving logical identities.

    The physical split may change when worker topology changes.  Request IDs
    and seeds do not: they are assigned before this function and never include
    the worker slot.
    """

    _positive_int(num_workers, "num_workers")
    jobs = tuple(sorted(batch.jobs, key=lambda item: item.job_id))
    prompts = [job.prompt for job in jobs]
    counts = [job.sample_count for job in jobs]
    worker_shards = _distribute_jobs(prompts, counts, num_workers)
    requests = [requests_for_job(job) for job in jobs]
    cursors = [0] * len(jobs)
    shards = []
    assigned_ids = set()
    for worker_slot, worker_jobs in enumerate(worker_shards):
        assigned = []
        for group_index, prompt, count in worker_jobs:
            if prompt != jobs[group_index].prompt:
                raise GenerationContractError("sharding changed a rendered prompt")
            start = cursors[group_index]
            end = start + count
            selected = requests[group_index][start:end]
            if len(selected) != count:
                raise GenerationContractError("sharding exceeded requested samples")
            cursors[group_index] = end
            assigned.extend(selected)
            for request in selected:
                if request.request_id in assigned_ids:
                    raise GenerationContractError("a request was assigned more than once")
                assigned_ids.add(request.request_id)
        shards.append(
            WorkerShard(
                worker_slot=worker_slot,
                batch=batch,
                requests=tuple(assigned),
            )
        )
    expected = {request.request_id for group in requests for request in group}
    if assigned_ids != expected or cursors != counts:
        raise GenerationContractError(
            "generation distribution did not preserve the exact requested count"
        )
    return tuple(shards)


def _adapter_identity_payload(batch: GenerationBatch) -> dict:
    snapshot = batch.policy_snapshot
    return {
        "snapshot_id": snapshot.snapshot_id,
        "run_id": snapshot.run_id,
        "epoch": snapshot.epoch,
        "role": snapshot.role.value,
        "adapter_id": snapshot.adapter_id,
        "adapter_version": snapshot.adapter_version,
        "adapter_hash": snapshot.adapter_hash,
        "policy_version": snapshot.policy_version,
    }


def _stable_positive_lora_id(adapter_key: str) -> int:
    digest = hashlib.sha256(adapter_key.encode("utf-8")).digest()
    value = int.from_bytes(digest[:8], "big") & _MAX_SEED
    return value or 1


@dataclass(frozen=True)
class AdapterRequest:
    """Stable HF/vLLM adapter binding for a frozen role snapshot."""

    adapter_key: str
    role: Role
    snapshot_id: str
    adapter_id: str
    adapter_version: str
    expected_adapter_hash: str
    adapter_path: str
    resolved_adapter_path: str
    hf_adapter_name: str
    vllm_lora_name: str
    vllm_lora_id: int


class AdapterRequestRegistry:
    """Reject live adapter cache aliases across roles, epochs, and paths."""

    def __init__(
        self,
        *,
        artifact_hash_resolver: Optional[Callable[[str], str]] = None,
        numeric_id_factory: Callable[[str], int] = _stable_positive_lora_id,
        test_mode: bool = False,
    ) -> None:
        if not isinstance(test_mode, bool):
            raise GenerationContractError("test_mode must be boolean")
        self._artifact_hash_resolver = artifact_hash_resolver
        self._numeric_id_factory = numeric_id_factory
        self._test_mode = test_mode
        self._by_snapshot: Dict[str, AdapterRequest] = {}
        self._by_realpath: Dict[str, str] = {}
        self._by_numeric_id: Dict[int, str] = {}

    def register(self, batch: GenerationBatch) -> AdapterRequest:
        snapshot = batch.policy_snapshot
        realpath = os.path.realpath(batch.adapter_path)
        if self._artifact_hash_resolver is None and not self._test_mode:
            raise GenerationContractError(
                "production adapter registration requires artifact hash validation"
            )
        payload = _adapter_identity_payload(batch)
        adapter_key = content_id("adapter_request", payload)
        digest = content_hash(payload)
        numeric_id = self._numeric_id_factory(adapter_key)
        if isinstance(numeric_id, bool) or not isinstance(numeric_id, int) or numeric_id <= 0:
            raise GenerationContractError(
                "vLLM LoRA numeric ID factory must return a positive integer"
            )
        existing_key = self._by_numeric_id.get(numeric_id)
        if existing_key is not None and existing_key != adapter_key:
            raise GenerationContractError(
                "vLLM LoRA numeric ID collision; refusing an adapter alias"
            )
        path_key = self._by_realpath.get(realpath)
        if path_key is not None and path_key != adapter_key:
            raise GenerationContractError(
                "one adapter path cannot serve different roles or snapshots"
            )
        previous = self._by_snapshot.get(snapshot.snapshot_id)
        if self._artifact_hash_resolver is not None:
            observed_hash = self._artifact_hash_resolver(batch.adapter_path)
            if observed_hash != snapshot.adapter_hash:
                raise GenerationContractError(
                    "adapter artifact hash does not match the frozen role snapshot"
                )
        if previous is not None:
            if previous.adapter_key != adapter_key or previous.resolved_adapter_path != realpath:
                raise GenerationContractError(
                    "one policy snapshot cannot be rebound to another adapter"
                )
            return previous
        request = AdapterRequest(
            adapter_key=adapter_key,
            role=snapshot.role,
            snapshot_id=snapshot.snapshot_id,
            adapter_id=snapshot.adapter_id,
            adapter_version=snapshot.adapter_version,
            expected_adapter_hash=snapshot.adapter_hash,
            adapter_path=batch.adapter_path,
            resolved_adapter_path=realpath,
            hf_adapter_name=f"ev_{snapshot.role.value}_{digest}",
            vllm_lora_name=f"ev_{snapshot.role.value}_{digest}",
            vllm_lora_id=numeric_id,
        )
        self._by_snapshot[snapshot.snapshot_id] = request
        self._by_realpath[realpath] = adapter_key
        self._by_numeric_id[numeric_id] = adapter_key
        return request


def vllm_lora_capacity(active_snapshot_count: int) -> Mapping[str, int]:
    """Return capacity sufficient for the three production role adapters."""

    _nonnegative_int(active_snapshot_count, "active_snapshot_count")
    capacity = max(MIN_PRODUCTION_ROLE_CAPACITY, active_snapshot_count)
    return {
        "max_loras": capacity,
        "max_cpu_loras": max(2 * capacity, MIN_PRODUCTION_ROLE_CAPACITY),
    }


@dataclass(frozen=True)
class GeneratedSample:
    """A successful backend result bound to exactly one request."""

    request_id: str
    backend_request_id: str
    text: str
    token_ids: Tuple[int, ...]

    def __post_init__(self) -> None:
        try:
            validate_id(self.request_id, "generation_request")
            validate_id(self.backend_request_id)
        except (TypeError, ValueError) as exc:
            raise GenerationContractError(str(exc)) from exc
        if not isinstance(self.text, str):
            raise GenerationContractError("generated text must be a string")
        token_ids = tuple(self.token_ids)
        if any(
            isinstance(token, bool) or not isinstance(token, int) or token < 0
            for token in token_ids
        ):
            raise GenerationContractError(
                "generated token IDs must be non-negative integers"
            )
        object.__setattr__(self, "token_ids", token_ids)


@dataclass(frozen=True)
class GenerationFailure:
    """Explicit infrastructure outcome; never a low scientific candidate."""

    failure_id: str
    worker_slot: int
    batch_id: str
    backend: GenerationBackend
    request_ids: Tuple[str, ...]
    failure_kind: str
    detail: str

    def __post_init__(self) -> None:
        normalized = (
            self.backend
            if isinstance(self.backend, GenerationBackend)
            else GenerationBackend(self.backend)
        )
        object.__setattr__(self, "backend", normalized)
        request_ids = tuple(self.request_ids)
        object.__setattr__(self, "request_ids", request_ids)
        try:
            validate_id(self.failure_id, "generation_failure")
            validate_id(self.batch_id, "generation_batch")
            for request_id in request_ids:
                validate_id(request_id, "generation_request")
        except (TypeError, ValueError) as exc:
            raise GenerationContractError(str(exc)) from exc
        _nonnegative_int(self.worker_slot, "worker_slot")
        if not request_ids:
            raise GenerationContractError(
                "a generation failure must identify affected requests"
            )
        if len(set(request_ids)) != len(request_ids):
            raise GenerationContractError(
                "generation failure request IDs must be unique"
            )
        if self.failure_kind != "infrastructure":
            raise GenerationContractError(
                "generation worker failures are infrastructure outcomes"
            )
        if not isinstance(self.detail, str) or not self.detail:
            raise GenerationContractError("generation failure detail is required")
        if len(self.detail) > 4000:
            raise GenerationContractError(
                "generation failure detail exceeds the persisted bound"
            )
        expected = content_id("generation_failure", self.identity_payload())
        if self.failure_id != expected:
            raise GenerationContractError(
                "generation failure ID must cover the complete failure observation"
            )

    def identity_payload(self) -> dict:
        return {
            "worker_slot": self.worker_slot,
            "batch_id": self.batch_id,
            "backend": self.backend.value,
            "request_ids": list(self.request_ids),
            "failure_kind": self.failure_kind,
            "detail": self.detail,
        }

    @classmethod
    def create(
        cls,
        *,
        worker_slot: int,
        batch_id: str,
        backend: GenerationBackend,
        request_ids: Sequence[str],
        detail: str,
    ) -> "GenerationFailure":
        normalized = backend if isinstance(backend, GenerationBackend) else GenerationBackend(backend)
        bounded_detail = str(detail)[:4000] or "unspecified generation infrastructure failure"
        payload = {
            "worker_slot": worker_slot,
            "batch_id": batch_id,
            "backend": normalized.value,
            "request_ids": list(request_ids),
            "failure_kind": "infrastructure",
            "detail": bounded_detail,
        }
        return cls(
            failure_id=content_id("generation_failure", payload),
            worker_slot=worker_slot,
            batch_id=batch_id,
            backend=normalized,
            request_ids=tuple(request_ids),
            failure_kind="infrastructure",
            detail=bounded_detail,
        )


class GenerationInfrastructureError(RuntimeError):
    """Raised with a persistable infrastructure failure record."""

    def __init__(self, failure: GenerationFailure) -> None:
        self.failure = failure
        super().__init__(failure.detail)


class StickyMicrobatch:
    """Worker-local OOM ceiling that can shrink but never drop requests."""

    def __init__(self, configured_limit: int = 0) -> None:
        _nonnegative_int(configured_limit, "configured_limit")
        self.configured_limit = configured_limit
        self.learned_limit = 0

    def next_size(self, remaining: int) -> int:
        _positive_int(remaining, "remaining")
        limit = remaining
        if self.configured_limit > 0:
            limit = min(limit, self.configured_limit)
        if self.learned_limit > 0:
            limit = min(limit, self.learned_limit)
        return limit

    def record_oom(self, attempted_size: int) -> None:
        _positive_int(attempted_size, "attempted_size")
        if attempted_size <= 1:
            raise GenerationOutOfMemory(
                "generation OOM persisted at one logical sample"
            )
        reduced = max(1, attempted_size // 2)
        if self.learned_limit == 0:
            self.learned_limit = reduced
        else:
            self.learned_limit = min(self.learned_limit, reduced)


GenerateChunk = Callable[
    [AdapterRequest, Tuple[GenerationRequest, ...]],
    Sequence[GeneratedSample],
]


def execute_worker_shard(
    shard: WorkerShard,
    *,
    backend: GenerationBackend,
    adapter_registry: AdapterRequestRegistry,
    microbatch: StickyMicrobatch,
    generate_chunk: GenerateChunk,
) -> Tuple[GeneratedSample, ...]:
    """Execute every request exactly once or raise explicit infrastructure.

    OOM retries repeat the same request identities and seeds under a smaller
    sticky chunk size.  Missing, duplicate, cross-backend, or extra outputs
    fail the whole worker shard; they are never padded with candidate text.
    """

    normalized_backend = (
        backend if isinstance(backend, GenerationBackend) else GenerationBackend(backend)
    )
    adapter_request = adapter_registry.register(shard.batch)
    pending = shard.requests
    completed = []
    cursor = 0
    while cursor < len(pending):
        chunk_size = microbatch.next_size(len(pending) - cursor)
        chunk = pending[cursor:cursor + chunk_size]
        try:
            raw_results = tuple(generate_chunk(adapter_request, chunk))
        except GenerationOutOfMemory as exc:
            try:
                microbatch.record_oom(chunk_size)
            except GenerationOutOfMemory as terminal:
                failure = GenerationFailure.create(
                    worker_slot=shard.worker_slot,
                    batch_id=shard.batch.batch_id,
                    backend=normalized_backend,
                    request_ids=[request.request_id for request in chunk],
                    detail=str(terminal),
                )
                raise GenerationInfrastructureError(failure) from exc
            continue
        except GenerationInfrastructureError:
            raise
        except Exception as exc:
            failure = GenerationFailure.create(
                worker_slot=shard.worker_slot,
                batch_id=shard.batch.batch_id,
                backend=normalized_backend,
                request_ids=[request.request_id for request in chunk],
                detail=f"generation backend failed: {type(exc).__name__}: {exc}",
            )
            raise GenerationInfrastructureError(failure) from exc
        expected = {request.request_id: request for request in chunk}
        actual: Dict[str, GeneratedSample] = {}
        contract_error = None
        for result in raw_results:
            if not isinstance(result, GeneratedSample):
                contract_error = "backend returned an untyped generation result"
                break
            if result.request_id in actual:
                contract_error = "backend returned a duplicate generation result"
                break
            request = expected.get(result.request_id)
            if request is None:
                contract_error = "backend returned a result for an unknown request"
                break
            if result.backend_request_id != request.backend_request_id(normalized_backend):
                contract_error = "backend request ID does not match the frozen request"
                break
            actual[result.request_id] = result
        if contract_error is None and set(actual) != set(expected):
            contract_error = "backend returned fewer generation results than requested"
        if contract_error is not None:
            failure = GenerationFailure.create(
                worker_slot=shard.worker_slot,
                batch_id=shard.batch.batch_id,
                backend=normalized_backend,
                request_ids=[request.request_id for request in chunk],
                detail=contract_error,
            )
            raise GenerationInfrastructureError(failure)
        completed.extend(actual[request.request_id] for request in chunk)
        cursor += chunk_size
    if len(completed) != len(shard.requests):
        failure = GenerationFailure.create(
            worker_slot=shard.worker_slot,
            batch_id=shard.batch.batch_id,
            backend=normalized_backend,
            request_ids=[request.request_id for request in shard.requests],
            detail="worker did not complete the exact logical rollout count",
        )
        raise GenerationInfrastructureError(failure)
    return tuple(completed)


__all__ = [
    "AdapterRequest",
    "AdapterRequestRegistry",
    "GeneratedSample",
    "GenerationBackend",
    "GenerationBatch",
    "GenerationContractError",
    "GenerationFailure",
    "GenerationInfrastructureError",
    "GenerationJob",
    "GenerationOutOfMemory",
    "GenerationParameters",
    "GenerationRequest",
    "StickyMicrobatch",
    "WorkerShard",
    "distribute_generation_jobs",
    "execute_worker_shard",
    "partition_generation_jobs",
    "requests_for_job",
    "vllm_lora_capacity",
]
