from __future__ import annotations

import copy
from dataclasses import replace

import pytest

from evolve.ids import content_hash, content_id
from evolve.roles import (
    PRODUCTION_ROLES,
    RoleIsolationError,
    RoleRegistry,
    validate_role_snapshot_identity,
)
from evolve.types import Channel, LearningGroup, LearningObjective, Role


RUN_ID = content_id("run", {"fixture": "role-isolation"})
BACKBONE_ID = content_id("backbone", {"fixture": "one-frozen-backbone"})
BACKBONE_HASH = content_hash({"fake_backbone_weights": "immutable"})


def _registry() -> RoleRegistry:
    # Equal numerical initial state is intentional: ownership metadata still
    # makes all three adapter artifacts and parameter sets non-aliasing.
    return RoleRegistry.create_production(
        run_id=RUN_ID,
        backbone_id=BACKBONE_ID,
        backbone_version="fake_backbone_v1",
        backbone_hash=BACKBONE_HASH,
        base_seed=991,
        initial_adapter_states={
            role.value: {"lora_A": [0.0, 0.0], "lora_B": [0.0]}
            for role in PRODUCTION_ROLES
        },
        initial_optimizer_states={
            role.value: {"step": 0, "momentum": [0.0]}
            for role in PRODUCTION_ROLES
        },
    )


def _learning_group(role: Role, snapshot_id: str, label: str) -> LearningGroup:
    return LearningGroup(
        group_id=content_id("learning_group", {"fixture": label}),
        role=role,
        policy_snapshot_id=snapshot_id,
        start_cell_id=content_id("cell", {"fixture": label}),
        context_id=content_id("context", {"fixture": label}),
        option_id=content_id("option", {"fixture": label}),
        harness_id=content_id("harness", {"fixture": label}),
        horizon=1,
        cost_class="tiny",
        generation_settings={"temperature": 0.5},
        frozen_record_threshold=1.0,
        channel=Channel.PRODUCTION,
        branch_ids=(content_id("branch", {"fixture": label}),),
        trace_ids=(content_id("policy_trace", {"fixture": label}),),
        outcome_ids=(content_id("branch_outcome", {"fixture": label}),),
        advantages=(1.0,),
        objective=LearningObjective.ORDERGRAD,
        objective_version="ordergrad_exact_v1",
        top_m=1,
    )


def _all_unique(values) -> bool:
    values = tuple(values)
    return len(set(values)) == len(values)


def test_production_registry_has_exactly_three_non_aliasing_role_owners() -> None:
    registry = _registry()

    assert registry.roles == (
        Role.SCOUT,
        Role.MECHANIST,
        Role.CHALLENGER,
    )
    assert registry.method_complete is True
    assert registry.backbone_frozen is True
    assert {state.run_id for state in registry.states} == {RUN_ID}
    assert _all_unique(state.adapter.adapter_id for state in registry.states)
    assert _all_unique(state.adapter.parameter_set_id for state in registry.states)
    assert _all_unique(state.adapter.adapter_hash for state in registry.states)
    assert _all_unique(state.optimizer.owner_id for state in registry.states)
    assert _all_unique(state.optimizer.state_id for state in registry.states)
    assert _all_unique(state.rng.owner_id for state in registry.states)
    assert _all_unique(state.transcript.owner_id for state in registry.states)
    assert _all_unique(state.retrieval.owner_id for state in registry.states)
    assert _all_unique(state.learning.owner_id for state in registry.states)
    for state in registry.states:
        assert state.optimizer.parameter_set_id == state.adapter.parameter_set_id
        assert state.optimizer.bound_adapter_version == state.adapter.version
        assert state.adapter.version == "adapter_v000000"


def test_epoch_snapshots_are_stable_content_addressed_and_immutable() -> None:
    registry = _registry()
    first = registry.freeze_epoch(4)
    second = registry.freeze_epoch(4)

    assert first == second
    assert tuple(first) == PRODUCTION_ROLES
    assert _all_unique(snapshot.snapshot_id for snapshot in first.values())
    assert _all_unique(snapshot.rng_seed for snapshot in first.values())
    for role, snapshot in first.items():
        validate_role_snapshot_identity(snapshot)
        state = registry.state(role)
        assert snapshot.adapter_id == state.adapter.adapter_id
        assert snapshot.adapter_hash == state.adapter.adapter_hash
        assert snapshot.optimizer_state_id == state.optimizer.state_id
        assert snapshot.frozen is True

    with pytest.raises(TypeError):
        first[Role.SCOUT] = first[Role.MECHANIST]
    with pytest.raises(Exception):
        first[Role.SCOUT].adapter_version = "tampered"

    forged = replace(
        first[Role.SCOUT],
        snapshot_id=content_id("role_snapshot", {"forged": True}),
    )
    with pytest.raises(RoleIsolationError, match="snapshot ID"):
        validate_role_snapshot_identity(forged)


def test_one_role_policy_advance_cannot_mutate_other_roles_or_old_snapshot() -> None:
    registry = _registry()
    frozen_before = registry.freeze_epoch(2)
    scout_before = registry.state(Role.SCOUT)
    mechanist_before = registry.state(Role.MECHANIST)
    challenger_before = registry.state(Role.CHALLENGER)

    updated = registry.advance_role(
        Role.SCOUT,
        adapter_state={"lora_A": [1.0, 0.0], "lora_B": [0.25]},
        optimizer_state={"step": 1, "momentum": [0.75]},
    )

    scout_after = updated.state(Role.SCOUT)
    assert scout_after.adapter.adapter_id == scout_before.adapter.adapter_id
    assert scout_after.adapter.parameter_set_id == scout_before.adapter.parameter_set_id
    assert scout_after.adapter.version == "adapter_v000001"
    assert scout_after.adapter.adapter_hash != scout_before.adapter.adapter_hash
    assert scout_after.optimizer.owner_id == scout_before.optimizer.owner_id
    assert scout_after.optimizer.state_id != scout_before.optimizer.state_id
    assert scout_after.optimizer.step == 1
    assert scout_after.optimizer.bound_adapter_version == scout_after.adapter.version

    # Exact object identity makes incidental copy/mutation of other roles fail
    # this test, not merely obvious value changes.
    assert updated.state(Role.MECHANIST) is mechanist_before
    assert updated.state(Role.CHALLENGER) is challenger_before
    assert registry.state(Role.SCOUT) is scout_before
    assert frozen_before == registry.freeze_epoch(2)
    assert updated.freeze_epoch(2)[Role.SCOUT] != frozen_before[Role.SCOUT]
    assert updated.freeze_epoch(2)[Role.MECHANIST] == frozen_before[Role.MECHANIST]

    with pytest.raises(TypeError):
        scout_after.adapter.state["lora_A"] = [99.0]
    with pytest.raises(RoleIsolationError, match="only when.*changes"):
        scout_before.adapter.advance(scout_before.adapter.state)
    with pytest.raises(RoleIsolationError, match="another run or role"):
        scout_before.optimizer.advance(
            adapter=mechanist_before.adapter,
            state={"step": 1},
        )


def test_role_rng_state_is_independent_checkpointed_and_resume_deterministic() -> None:
    registry = _registry()
    first_seed, after_scout_draw = registry.draw_role_seed(Role.SCOUT, "cell-choice")

    assert after_scout_draw.state(Role.SCOUT).rng.counter == 1
    assert after_scout_draw.state(Role.MECHANIST).rng is registry.state(Role.MECHANIST).rng
    assert after_scout_draw.state(Role.CHALLENGER).rng is registry.state(Role.CHALLENGER).rng
    assert first_seed >= 0
    # Consuming private RNG choices cannot move an already-frozen epoch policy.
    assert after_scout_draw.freeze_epoch(7) == registry.freeze_epoch(7)

    restored = RoleRegistry.from_checkpoint_json(after_scout_draw.to_checkpoint_json())
    expected_seed, expected_state = after_scout_draw.draw_role_seed(
        Role.SCOUT, "option-choice"
    )
    actual_seed, actual_state = restored.draw_role_seed(Role.SCOUT, "option-choice")
    assert actual_seed == expected_seed
    assert actual_state == expected_state

    mechanist_seed, _ = registry.draw_role_seed(Role.MECHANIST, "cell-choice")
    assert mechanist_seed != first_seed


def test_private_transcript_and_retrieval_view_updates_are_role_local() -> None:
    registry = _registry()
    branch_id = content_id("branch", {"fixture": "scout-private"})
    with_transcript = registry.start_transcript(Role.SCOUT, branch_id)
    with_transcript = with_transcript.append_transcript(
        Role.SCOUT,
        kind="proposal",
        content={"summary": "private working hypothesis", "candidate_ids": [1, 2]},
    )

    transcript = with_transcript.state(Role.SCOUT).transcript
    assert transcript.branch_id == branch_id
    assert transcript.entries[0]["content"]["summary"] == "private working hypothesis"
    assert with_transcript.state(Role.MECHANIST) is registry.state(Role.MECHANIST)
    assert with_transcript.state(Role.CHALLENGER) is registry.state(Role.CHALLENGER)
    assert not hasattr(transcript, "promote")
    with pytest.raises(TypeError):
        transcript.entries[0]["content"]["summary"] = "leaked"

    memory_snapshot_id = content_id(
        "causal_memory_snapshot", {"fixture": "barrier-3"}
    )
    memory_id = content_id("causal_memory", {"fixture": "audit-backed"})
    with_view = with_transcript.advance_retrieval_view(
        Role.MECHANIST,
        memory_snapshot_id=memory_snapshot_id,
        memory_ids=(memory_id,),
        scope={"cell_region": "dense", "no_extra_rollouts": True},
    )
    assert with_view.state(Role.MECHANIST).retrieval.memory_ids == (memory_id,)
    assert with_view.state(Role.SCOUT) is with_transcript.state(Role.SCOUT)
    assert with_view.state(Role.CHALLENGER) is with_transcript.state(Role.CHALLENGER)
    with pytest.raises(RoleIsolationError, match="only when.*changes"):
        with_view.advance_retrieval_view(
            Role.MECHANIST,
            memory_snapshot_id=memory_snapshot_id,
            memory_ids=(memory_id,),
            scope={"cell_region": "dense", "no_extra_rollouts": True},
        )


def test_learning_groups_are_claimed_only_by_exact_generating_role_snapshot() -> None:
    registry = _registry()
    snapshots = registry.freeze_epoch(0)
    scout_group = _learning_group(
        Role.SCOUT, snapshots[Role.SCOUT].snapshot_id, "scout-group"
    )

    claimed = registry.claim_learning_group(
        Role.SCOUT,
        group=scout_group,
        snapshot=snapshots[Role.SCOUT],
    )
    assert claimed.state(Role.SCOUT).learning.group_ids == (scout_group.group_id,)
    assert claimed.state(Role.MECHANIST).learning.group_ids == ()
    assert claimed.state(Role.CHALLENGER).learning.group_ids == ()
    # Idempotent durable replay does not duplicate ownership.
    assert claimed.claim_learning_group(
        Role.SCOUT,
        group=scout_group,
        snapshot=snapshots[Role.SCOUT],
    ) == claimed

    with pytest.raises(RoleIsolationError, match="another role"):
        registry.claim_learning_group(
            Role.MECHANIST,
            group=scout_group,
            snapshot=snapshots[Role.MECHANIST],
        )

    advanced = registry.advance_role(
        Role.SCOUT,
        adapter_state={"lora_A": [3.0], "lora_B": [4.0]},
        optimizer_state={"step": 1},
    )
    with pytest.raises(RoleIsolationError, match="stale"):
        advanced.claim_learning_group(
            Role.SCOUT,
            group=scout_group,
            snapshot=snapshots[Role.SCOUT],
        )


def test_full_cpu_checkpoint_round_trip_restores_every_role_owner() -> None:
    registry = _registry()
    branch_id = content_id("branch", {"fixture": "restore-transcript"})
    registry = registry.start_transcript(Role.CHALLENGER, branch_id)
    registry = registry.append_transcript(
        Role.CHALLENGER,
        kind="counterexample",
        content={"claim": "minimal repair failed", "attempt": 1},
    )
    _, registry = registry.draw_role_seed(Role.MECHANIST, "mechanism-choice")
    registry = registry.advance_role(
        Role.CHALLENGER,
        adapter_state={"lora_A": [0.0], "lora_B": [-1.0]},
        optimizer_state={"step": 1, "momentum": [-0.2]},
    )
    snapshots = registry.freeze_epoch(5)
    group = _learning_group(
        Role.MECHANIST,
        snapshots[Role.MECHANIST].snapshot_id,
        "restore-group",
    )
    registry = registry.claim_learning_group(
        Role.MECHANIST,
        group=group,
        snapshot=snapshots[Role.MECHANIST],
    )

    encoded = registry.to_checkpoint_json()
    restored = RoleRegistry.from_checkpoint_json(encoded)
    assert restored == registry
    assert restored.to_checkpoint_json() == encoded
    assert restored.freeze_epoch(5) == snapshots
    assert restored.state(Role.CHALLENGER).transcript.entries == (
        registry.state(Role.CHALLENGER).transcript.entries
    )
    assert restored.state(Role.MECHANIST).learning.group_ids == (group.group_id,)


def test_checkpoint_tamper_future_schema_and_aliases_are_rejected() -> None:
    registry = _registry()
    payload = copy.deepcopy(registry.checkpoint_payload())
    payload["states"][0]["adapter"]["state"]["lora_A"] = [999.0]
    body = {key: value for key, value in payload.items() if key != "checkpoint_hash"}
    payload["checkpoint_hash"] = content_hash(body)
    with pytest.raises(RoleIsolationError, match="adapter_hash"):
        RoleRegistry.from_checkpoint_payload(payload)

    future = copy.deepcopy(registry.checkpoint_payload())
    future["schema_version"] = 2
    body = {key: value for key, value in future.items() if key != "checkpoint_hash"}
    future["checkpoint_hash"] = content_hash(body)
    with pytest.raises(RoleIsolationError, match="unsupported role registry schema"):
        RoleRegistry.from_checkpoint_payload(future)

    boolean_schema = copy.deepcopy(registry.checkpoint_payload())
    boolean_schema["schema_version"] = True
    body = {
        key: value
        for key, value in boolean_schema.items()
        if key != "checkpoint_hash"
    }
    boolean_schema["checkpoint_hash"] = content_hash(body)
    with pytest.raises(RoleIsolationError, match="unsupported role registry schema"):
        RoleRegistry.from_checkpoint_payload(boolean_schema)

    with pytest.raises(RoleIsolationError, match="duplicate key"):
        RoleRegistry.from_checkpoint_json('{"record_type":"x","record_type":"y"}')

    with pytest.raises(RoleIsolationError, match="backbone must remain frozen"):
        replace(registry, backbone_frozen=False)

    with pytest.raises(RoleIsolationError, match="another role or run"):
        replace(
            registry.state(Role.MECHANIST),
            adapter=registry.state(Role.SCOUT).adapter,
        )


def test_role_subsets_are_explicitly_method_incomplete_test_fixtures() -> None:
    fixture = RoleRegistry.create_test_fixture(
        run_id=RUN_ID,
        backbone_id=BACKBONE_ID,
        backbone_version="fake_backbone_v1",
        backbone_hash=BACKBONE_HASH,
        base_seed=4,
        roles=(Role.SCOUT,),
    )
    assert fixture.roles == (Role.SCOUT,)
    assert fixture.method_complete is False
    with pytest.raises(RoleIsolationError, match="production role registry requires exactly"):
        replace(fixture, method_complete=True)
    with pytest.raises(RoleIsolationError, match="canonical production role order"):
        RoleRegistry.create_test_fixture(
            run_id=RUN_ID,
            backbone_id=BACKBONE_ID,
            backbone_version="fake_backbone_v1",
            backbone_hash=BACKBONE_HASH,
            base_seed=4,
            roles=(Role.CHALLENGER, Role.SCOUT),
        )
