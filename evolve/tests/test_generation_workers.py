from __future__ import annotations

from copy import deepcopy
from dataclasses import FrozenInstanceError, replace
from typing import Optional

import pytest

from evolve.ids import content_hash, content_id
from evolve.types import Role, RoleSnapshot
from evolve.workers import (
    AdapterRequestRegistry,
    GeneratedSample,
    GenerationBackend,
    GenerationContractError,
    GenerationInfrastructureError,
    GenerationJob,
    GenerationOutOfMemory,
    GenerationParameters,
    StickyMicrobatch,
    distribute_generation_jobs,
    execute_worker_shard,
    partition_generation_jobs,
    requests_for_job,
    vllm_lora_capacity,
)


def _id(namespace: str, label: str) -> str:
    return content_id(namespace, {"label": label})


def _snapshot(
    *,
    role: Role = Role.SCOUT,
    epoch: int = 3,
    version: str = "epoch003",
    run_label: str = "run",
    snapshot_label: str = "snapshot",
    adapter_hash_label: str = "weights",
    rng_seed: int = 71,
) -> RoleSnapshot:
    run_id = _id("run", run_label)
    adapter_id = _id("adapter", f"{role.value}-{epoch}-{version}")
    adapter_hash = content_hash({"weights": adapter_hash_label})
    optimizer_state_id = _id("optimizer", f"{role.value}-{epoch}")
    policy_version = f"policy-{version}"
    identity = {
        "snapshot_version": "role_snapshot_identity_v1",
        "run_id": run_id,
        "epoch": epoch,
        "role": role.value,
        "adapter_id": adapter_id,
        "adapter_version": version,
        "adapter_hash": adapter_hash,
        "optimizer_state_id": optimizer_state_id,
        "policy_version": policy_version,
        "rng_seed": rng_seed,
        "frozen": True,
    }
    return RoleSnapshot(
        snapshot_id=content_id("role_snapshot", identity),
        run_id=run_id,
        epoch=epoch,
        role=role,
        adapter_id=adapter_id,
        adapter_version=version,
        adapter_hash=adapter_hash,
        optimizer_state_id=optimizer_state_id,
        policy_version=policy_version,
        rng_seed=rng_seed,
    )


def _job(
    tmp_path,
    *,
    snapshot: Optional[RoleSnapshot] = None,
    path_name: Optional[str] = None,
    sample_count: int = 5,
    sample_index_start: int = 0,
    prompt: str = "rendered prompt",
    allocation_label: str = "allocation",
    branch_label: str = "branch",
    branch_step: int = 2,
) -> GenerationJob:
    snapshot = snapshot or _snapshot()
    path_name = path_name or f"{snapshot.role.value}-{snapshot.adapter_version}"
    adapter_path = tmp_path / path_name
    adapter_path.mkdir(exist_ok=True)
    return GenerationJob.create(
        run_id=snapshot.run_id,
        epoch=snapshot.epoch,
        allocation_id=_id("arm", allocation_label),
        branch_id=_id("branch", branch_label),
        branch_step=branch_step,
        role=snapshot.role,
        adapter_path=str(adapter_path),
        adapter_id=snapshot.adapter_id,
        adapter_version=snapshot.adapter_version,
        policy_snapshot=snapshot,
        option_id=_id("option", "bounded-option"),
        harness_id=_id("harness", "baseline-v1"),
        prompt=prompt,
        sample_index_start=sample_index_start,
        sample_count=sample_count,
        generation_parameters=GenerationParameters(
            max_new_tokens=64,
            temperature=0.8,
            top_p=0.95,
            micro_batch=4,
            extras={"min_p": 0.05},
        ),
    )


def _successful_chunk(backend, seen=None):
    def generate(adapter_request, requests):
        if seen is not None:
            seen.append((adapter_request, requests))
        return [
            GeneratedSample(
                request_id=request.request_id,
                backend_request_id=request.backend_request_id(backend),
                text=f"{request.job.role.value}-{request.sample_index}",
                token_ids=(request.sample_index + 1,),
            )
            for request in requests
        ]

    return generate


def test_generation_job_is_frozen_complete_and_content_addressed(tmp_path):
    job = _job(tmp_path, sample_count=3, sample_index_start=11)

    assert job.policy_snapshot_id == job.policy_snapshot.snapshot_id
    assert job.adapter_id == job.policy_snapshot.adapter_id
    assert job.adapter_version == job.policy_snapshot.adapter_version
    assert job.role == Role.SCOUT
    assert job.sample_count == 3
    assert job.to_dict()["generation_parameters"]["extras"] == {"min_p": 0.05}
    with pytest.raises(FrozenInstanceError):
        job.sample_count = 8
    with pytest.raises(GenerationContractError, match="job_id must cover"):
        replace(job, prompt="mutated after identity assignment")
    with pytest.raises(GenerationContractError, match="must match policy snapshot"):
        replace(job, adapter_version="wrong-version")


def test_job_identity_survives_adapter_artifact_relocation(tmp_path):
    snapshot = _snapshot()
    first = _job(tmp_path, snapshot=snapshot, path_name="first-copy")
    second = _job(tmp_path, snapshot=snapshot, path_name="relocated-copy")

    assert first.job_id == second.job_id
    assert first.seed == second.seed
    assert first.adapter_path != second.adapter_path


def test_job_and_parameters_roundtrip_strict_persisted_schema(tmp_path):
    job = _job(tmp_path)

    loaded_parameters = GenerationParameters.from_dict(
        job.generation_parameters.to_dict()
    )
    loaded_job = GenerationJob.from_dict(job.to_dict())

    assert loaded_parameters == job.generation_parameters
    assert loaded_job == job
    assert loaded_job.to_dict() == job.to_dict()

    extended = deepcopy(job.to_dict())
    extended["reader_hint"] = {"safe": True}
    loaded_extended = GenerationJob.from_dict(extended)
    assert loaded_extended.job_id == job.job_id
    assert loaded_extended.extensions["reader_hint"] == {"safe": True}
    assert loaded_extended.to_dict()["extensions"]["reader_hint"] == {"safe": True}

    future_job = deepcopy(job.to_dict())
    future_job["schema_version"] = 2
    with pytest.raises(GenerationContractError, match="newer than supported"):
        GenerationJob.from_dict(future_job)
    future_parameters = deepcopy(job.generation_parameters.to_dict())
    future_parameters["schema_version"] = 2
    with pytest.raises(GenerationContractError, match="newer than supported"):
        GenerationParameters.from_dict(future_parameters)


def test_per_sample_ids_and_seeds_ignore_worker_topology_and_completion_order(tmp_path):
    job = _job(tmp_path, sample_count=13, sample_index_start=9)
    batch = partition_generation_jobs([job])[0]

    one_worker = distribute_generation_jobs(batch, 1)
    four_workers = distribute_generation_jobs(batch, 4)
    expected = {
        request.request_id: (
            request.seed,
            request.hf_request_id,
            request.vllm_request_id,
        )
        for request in requests_for_job(job)
    }
    observed_one = {
        request.request_id: (
            request.seed,
            request.hf_request_id,
            request.vllm_request_id,
        )
        for shard in reversed(one_worker)
        for request in reversed(shard.requests)
    }
    observed_four = {
        request.request_id: (
            request.seed,
            request.hf_request_id,
            request.vllm_request_id,
        )
        for shard in reversed(four_workers)
        for request in reversed(shard.requests)
    }

    assert observed_one == expected
    assert observed_four == expected
    assert sum(len(shard.requests) for shard in four_workers) == 13
    assert len(set(expected)) == 13
    assert len({values[0] for values in expected.values()}) == 13
    assert all(values[1] != values[2] for values in expected.values())


def test_distribution_preserves_each_job_count_through_legacy_adapter(tmp_path):
    snapshot = _snapshot()
    jobs = [
        _job(
            tmp_path,
            snapshot=snapshot,
            sample_count=count,
            sample_index_start=start,
            prompt=f"prompt-{index}",
            branch_label=f"branch-{index}",
        )
        for index, (start, count) in enumerate(((0, 1), (1, 8), (9, 13)))
    ]
    batch = partition_generation_jobs(jobs)[0]

    shards = distribute_generation_jobs(batch, 5)
    by_job = {job.job_id: 0 for job in jobs}
    for shard in shards:
        for request in shard.requests:
            by_job[request.job.job_id] += 1

    assert len(shards) == 5
    assert by_job == {job.job_id: job.sample_count for job in jobs}
    assert sum(by_job.values()) == batch.sample_count == 22


def test_partition_never_mixes_role_or_policy_snapshot(tmp_path):
    scout = _snapshot(role=Role.SCOUT, snapshot_label="scout")
    mechanist = _snapshot(role=Role.MECHANIST, snapshot_label="mechanist")
    newer_scout = _snapshot(
        role=Role.SCOUT,
        version="epoch003b",
        snapshot_label="scout-b",
    )
    jobs = [
        _job(tmp_path, snapshot=scout, allocation_label="scout"),
        _job(tmp_path, snapshot=mechanist, allocation_label="mechanist"),
        _job(tmp_path, snapshot=newer_scout, allocation_label="scout-b"),
    ]

    batches = partition_generation_jobs(reversed(jobs))

    assert len(batches) == 3
    assert {batch.policy_snapshot.snapshot_id for batch in batches} == {
        job.policy_snapshot.snapshot_id for job in jobs
    }
    for batch in batches:
        assert len({job.role for job in batch.jobs}) == 1
        assert len({job.policy_snapshot_id for job in batch.jobs}) == 1
        assert len({job.adapter_id for job in batch.jobs}) == 1


def test_overlapping_logical_sample_is_rejected_even_with_different_prompt(tmp_path):
    first = _job(tmp_path, sample_count=3, prompt="first")
    second = _job(tmp_path, sample_count=3, prompt="different")

    with pytest.raises(GenerationContractError, match="overlap"):
        partition_generation_jobs([first, second])


def test_adapter_ids_are_stable_and_separate_roles_and_epochs(tmp_path):
    snapshots = [
        _snapshot(role=Role.SCOUT, snapshot_label="scout"),
        _snapshot(role=Role.MECHANIST, snapshot_label="mechanist"),
        _snapshot(role=Role.CHALLENGER, snapshot_label="challenger"),
        _snapshot(
            role=Role.SCOUT,
            epoch=4,
            version="epoch004",
            snapshot_label="scout-next",
        ),
    ]
    batches = [
        partition_generation_jobs(
            [
                _job(
                    tmp_path,
                    snapshot=snapshot,
                    allocation_label=f"arm-{index}",
                    branch_label=f"branch-{index}",
                )
            ]
        )[0]
        for index, snapshot in enumerate(snapshots)
    ]
    forward_registry = AdapterRequestRegistry(test_mode=True)
    reverse_registry = AdapterRequestRegistry(test_mode=True)

    forward = {
        batch.policy_snapshot.snapshot_id: forward_registry.register(batch)
        for batch in batches
    }
    reverse = {
        batch.policy_snapshot.snapshot_id: reverse_registry.register(batch)
        for batch in reversed(batches)
    }

    assert {key: value.adapter_key for key, value in forward.items()} == {
        key: value.adapter_key for key, value in reverse.items()
    }
    assert {key: value.vllm_lora_id for key, value in forward.items()} == {
        key: value.vllm_lora_id for key, value in reverse.items()
    }
    assert len({value.adapter_key for value in forward.values()}) == 4
    assert len({value.hf_adapter_name for value in forward.values()}) == 4
    assert len({value.vllm_lora_name for value in forward.values()}) == 4
    assert len({value.vllm_lora_id for value in forward.values()}) == 4
    assert all(
        1 <= value.vllm_lora_id <= (1 << 31) - 1
        for value in forward.values()
    )


def test_adapter_registry_rejects_path_and_numeric_aliases(tmp_path):
    shared_path = "shared-adapter"
    scout = _job(
        tmp_path,
        snapshot=_snapshot(role=Role.SCOUT, snapshot_label="scout"),
        path_name=shared_path,
        allocation_label="scout",
    )
    mechanist = _job(
        tmp_path,
        snapshot=_snapshot(role=Role.MECHANIST, snapshot_label="mechanist"),
        path_name=shared_path,
        allocation_label="mechanist",
    )
    scout_batch = partition_generation_jobs([scout])[0]
    mechanist_batch = partition_generation_jobs([mechanist])[0]

    path_registry = AdapterRequestRegistry(test_mode=True)
    path_registry.register(scout_batch)
    with pytest.raises(GenerationContractError, match="one adapter path"):
        path_registry.register(mechanist_batch)

    collision_registry = AdapterRequestRegistry(
        numeric_id_factory=lambda _key: 7,
        test_mode=True,
    )
    collision_registry.register(scout_batch)
    with pytest.raises(GenerationContractError, match="numeric ID collision"):
        collision_registry.register(mechanist_batch)


def test_adapter_registry_can_verify_frozen_artifact_hash(tmp_path):
    job = _job(tmp_path)
    batch = partition_generation_jobs([job])[0]
    registry = AdapterRequestRegistry(
        artifact_hash_resolver=lambda _path: content_hash({"not": "the weights"})
    )

    with pytest.raises(GenerationContractError, match="artifact hash"):
        registry.register(batch)


def test_production_adapter_registry_never_silently_skips_hash_validation(tmp_path):
    batch = partition_generation_jobs([_job(tmp_path)])[0]

    with pytest.raises(GenerationContractError, match="requires artifact hash"):
        AdapterRequestRegistry().register(batch)


def test_adapter_registry_revalidates_overwrite_and_symlink_retarget(tmp_path):
    snapshot = _snapshot()
    expected_hash = snapshot.adapter_hash
    state = {"observed_hash": expected_hash}
    direct_job = _job(tmp_path, snapshot=snapshot, path_name="direct")
    direct_batch = partition_generation_jobs([direct_job])[0]
    registry = AdapterRequestRegistry(
        artifact_hash_resolver=lambda _path: state["observed_hash"]
    )
    registry.register(direct_batch)
    state["observed_hash"] = content_hash({"overwritten": True})
    with pytest.raises(GenerationContractError, match="artifact hash"):
        registry.register(direct_batch)

    target_one = tmp_path / "target-one"
    target_two = tmp_path / "target-two"
    target_one.mkdir()
    target_two.mkdir()
    symlink = tmp_path / "adapter-link"
    symlink.symlink_to(target_one, target_is_directory=True)
    linked_job = replace(direct_job, adapter_path=str(symlink))
    linked_batch = partition_generation_jobs([linked_job])[0]
    symlink_registry = AdapterRequestRegistry(
        artifact_hash_resolver=lambda _path: expected_hash
    )
    symlink_registry.register(linked_batch)
    symlink.unlink()
    symlink.symlink_to(target_two, target_is_directory=True)
    with pytest.raises(GenerationContractError, match="cannot be rebound"):
        symlink_registry.register(linked_batch)


def test_vllm_capacity_keeps_all_three_production_roles_available():
    assert vllm_lora_capacity(0) == {"max_loras": 3, "max_cpu_loras": 6}
    assert vllm_lora_capacity(5) == {"max_loras": 5, "max_cpu_loras": 10}
    with pytest.raises(GenerationContractError):
        vllm_lora_capacity(-1)


@pytest.mark.parametrize("backend", [GenerationBackend.HF, GenerationBackend.VLLM])
def test_cpu_fake_execution_preserves_role_adapter_nonleakage(tmp_path, backend):
    jobs = [
        _job(
            tmp_path,
            snapshot=_snapshot(role=role, snapshot_label=role.value),
            allocation_label=role.value,
            branch_label=role.value,
            sample_count=2,
        )
        for role in Role
    ]
    registry = AdapterRequestRegistry(test_mode=True)
    seen = []
    results = []
    for batch in partition_generation_jobs(jobs):
        shard = distribute_generation_jobs(batch, 1)[0]
        results.extend(
            execute_worker_shard(
                shard,
                backend=backend,
                adapter_registry=registry,
                microbatch=StickyMicrobatch(configured_limit=2),
                generate_chunk=_successful_chunk(backend, seen),
            )
        )

    assert len(results) == 6
    assert len({result.request_id for result in results}) == 6
    for adapter_request, requests in seen:
        assert {request.job.role for request in requests} == {adapter_request.role}
        assert {request.job.policy_snapshot_id for request in requests} == {
            adapter_request.snapshot_id
        }
        assert all(
            result.backend_request_id
            == next(
                request.backend_request_id(backend)
                for request in requests
                if request.request_id == result.request_id
            )
            for result in results
            if result.request_id in {request.request_id for request in requests}
        )


def test_sticky_oom_reduction_retries_without_changing_logical_count(tmp_path):
    job = _job(tmp_path, sample_count=4)
    shard = distribute_generation_jobs(partition_generation_jobs([job])[0], 1)[0]
    microbatch = StickyMicrobatch()
    attempted = []

    def generate(adapter_request, requests):
        attempted.append(tuple(request.request_id for request in requests))
        if len(requests) > 2:
            raise GenerationOutOfMemory("fake oom")
        return _successful_chunk(GenerationBackend.HF)(adapter_request, requests)

    results = execute_worker_shard(
        shard,
        backend=GenerationBackend.HF,
        adapter_registry=AdapterRequestRegistry(test_mode=True),
        microbatch=microbatch,
        generate_chunk=generate,
    )

    assert [len(chunk) for chunk in attempted] == [4, 2, 2]
    assert microbatch.learned_limit == 2
    assert [result.request_id for result in results] == [
        request.request_id for request in shard.requests
    ]
    assert len(results) == job.sample_count


def test_terminal_oom_is_explicit_infrastructure_not_empty_candidate(tmp_path):
    job = _job(tmp_path, sample_count=1)
    shard = distribute_generation_jobs(partition_generation_jobs([job])[0], 1)[0]

    with pytest.raises(GenerationInfrastructureError) as raised:
        execute_worker_shard(
            shard,
            backend=GenerationBackend.VLLM,
            adapter_registry=AdapterRequestRegistry(test_mode=True),
            microbatch=StickyMicrobatch(configured_limit=1),
            generate_chunk=lambda _adapter, _requests: (_ for _ in ()).throw(
                GenerationOutOfMemory("fake terminal oom")
            ),
        )

    assert raised.value.failure.failure_kind == "infrastructure"
    assert raised.value.failure.request_ids == (shard.requests[0].request_id,)
    assert "one logical sample" in raised.value.failure.detail


def test_missing_backend_result_is_explicit_infrastructure(tmp_path):
    job = _job(tmp_path, sample_count=2)
    shard = distribute_generation_jobs(partition_generation_jobs([job])[0], 1)[0]

    with pytest.raises(GenerationInfrastructureError) as raised:
        execute_worker_shard(
            shard,
            backend=GenerationBackend.HF,
            adapter_registry=AdapterRequestRegistry(test_mode=True),
            microbatch=StickyMicrobatch(configured_limit=2),
            generate_chunk=lambda adapter, requests: _successful_chunk(
                GenerationBackend.HF
            )(adapter, requests[:1]),
        )

    assert raised.value.failure.failure_kind == "infrastructure"
    assert "fewer generation results" in raised.value.failure.detail
    assert len(raised.value.failure.request_ids) == 2

    with pytest.raises(GenerationContractError, match="infrastructure outcomes"):
        replace(raised.value.failure, failure_kind="scientific")
    with pytest.raises(GenerationContractError, match="complete failure observation"):
        replace(raised.value.failure, detail="tampered detail")


def test_reversed_backend_results_route_by_request_id_not_completion_order(tmp_path):
    job = _job(tmp_path, sample_count=4)
    shard = distribute_generation_jobs(partition_generation_jobs([job])[0], 1)[0]

    def reverse_results(adapter, requests):
        return tuple(
            reversed(_successful_chunk(GenerationBackend.VLLM)(adapter, requests))
        )

    results = execute_worker_shard(
        shard,
        backend=GenerationBackend.VLLM,
        adapter_registry=AdapterRequestRegistry(test_mode=True),
        microbatch=StickyMicrobatch(configured_limit=4),
        generate_chunk=reverse_results,
    )

    assert [result.request_id for result in results] == [
        request.request_id for request in shard.requests
    ]


@pytest.mark.parametrize(
    "kwargs",
    [
        {"max_new_tokens": 0, "temperature": 1.0, "top_p": 1.0},
        {"max_new_tokens": 1, "temperature": -0.1, "top_p": 1.0},
        {"max_new_tokens": 1, "temperature": 1.0, "top_p": 0.0},
        {
            "max_new_tokens": 1,
            "temperature": 1.0,
            "top_p": 1.0,
            "extras": {"temperature": 0.5},
        },
    ],
)
def test_generation_parameters_reject_invalid_or_shadowed_values(kwargs):
    with pytest.raises(GenerationContractError):
        GenerationParameters(**kwargs)
