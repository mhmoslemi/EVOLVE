import copy
import dataclasses
import json
import pickle

import pytest

from evolve.ids import (
    CanonicalJSONError,
    canonical_json,
    content_hash,
    content_id,
    derive_id,
    rollout_seed,
)
from evolve.types import (
    AllocationArm,
    ArchiveCell,
    AuditPair,
    AuditStatus,
    BranchOutcome,
    BranchSpec,
    BranchStatus,
    BudgetLedger,
    CausalMemoryRecord,
    Channel,
    Descriptor,
    EpochManifest,
    EvidencePacket,
    FailureKind,
    HarnessSpec,
    InvariantViolation,
    LearningGroup,
    MemoryStatus,
    OptionSpec,
    PolicyTrace,
    Proposal,
    ProvenanceEdge,
    Role,
    RoleSnapshot,
    SchemaValidationError,
    UnsupportedSchemaVersion,
    VerifiedScientificState,
    record_from_dict,
)


def ident(namespace, label):
    return content_id(namespace, {"label": label})


def digest(label):
    return content_hash({"label": label})


def harness():
    return HarnessSpec.create(
        version="baseline_v1",
        instructions="Use the declared tools only.",
        tools=("python",),
        intermediate_tests=("smoke",),
        scaffolding={"template": "v1"},
        diagnostic_feedback={"chars": 1000},
        tool_policy_version="sandbox_v1",
    )


def all_records():
    run_id = ident("run", "run")
    proposal_id = ident("proposal", "proposal")
    state_id = ident("state", "state")
    parent_state_id = ident("state", "parent")
    evidence_id = ident("evidence", "evidence")
    branch_id = ident("branch", "branch")
    branch_two = ident("branch", "branch-two")
    verifier_id = ident("verifier", "common")
    role_snapshot_id = ident("role_snapshot", "scout-0")
    descriptor_id = ident("descriptor", "descriptor")
    cell_id = ident("cell", "cell")
    option_id = ident("option", "improve")
    control_option_id = ident("option", "continue")
    harness_spec = harness()
    source_text = "def answer():\n    return 3\n"
    proposal = Proposal(
        proposal_id=proposal_id,
        run_id=run_id,
        problem_id="toy",
        source_text=source_text,
        source_hash=content_hash(source_text),
        parent_state_id=parent_state_id,
        branch_id=branch_id,
        parsed_candidate={"entrypoint": "answer"},
    )
    verified = VerifiedScientificState(
        state_id=state_id,
        proposal_id=proposal_id,
        evidence_id=evidence_id,
        problem_id="toy",
        answer_payload={"answer": 3},
        resolved=True,
        admitted=True,
        confirmed=True,
        internal_reward=3.0,
        raw_score=3,
        descriptor_id=descriptor_id,
        fingerprint="family:constant",
    )
    evidence = EvidencePacket(
        evidence_id=evidence_id,
        run_id=run_id,
        proposal_id=proposal_id,
        scientific_state_id=state_id,
        parent_state_id=parent_state_id,
        branch_id=branch_id,
        problem_id="toy",
        verifier_id=verifier_id,
        verifier_version="toy_v1",
        harness_id=harness_spec.harness_id,
        policy_snapshot_id=role_snapshot_id,
        lineage_ids=(parent_state_id, state_id),
        resolved=True,
        admitted=True,
        confirmed=True,
        failure_kind=FailureKind.NONE,
        internal_reward=3.0,
        raw_score=3,
        uncertainty=0.0,
        descriptor_id=descriptor_id,
        fingerprint="family:constant",
        source_hash=content_hash(source_text),
        flags={"deterministic": True},
        scores={"native": 3},
        diagnostics={"stdout": "ok"},
        resources={"verifier_calls": 1, "wall_time_s": 0.01},
        answer_payload={"answer": 3},
    )
    descriptor = Descriptor(
        descriptor_id=descriptor_id,
        problem_id="toy",
        function_version="descriptor_v1",
        dimensions={"family": "constant", "complexity": 1},
    )
    archive_cell = ArchiveCell(
        cell_id=cell_id,
        descriptor_id=descriptor_id,
        champion_state_id=state_id,
        champion_evidence_id=evidence_id,
        promising_state_ids=(ident("state", "promising"),),
        stepping_stone_state_ids=(ident("state", "stepping"),),
        tested_count=3,
        under_tested=False,
    )
    provenance = ProvenanceEdge(
        edge_id=ident("provenance", "edge"),
        parent_state_id=parent_state_id,
        child_state_id=state_id,
        proposal_id=proposal_id,
        evidence_id=evidence_id,
        branch_id=branch_id,
    )
    role_snapshot = RoleSnapshot(
        snapshot_id=role_snapshot_id,
        run_id=run_id,
        epoch=0,
        role=Role.SCOUT,
        adapter_id=ident("adapter", "scout-0"),
        adapter_version="epoch000",
        adapter_hash=digest("adapter"),
        optimizer_state_id=ident("optimizer", "scout-0"),
        policy_version="policy_v1",
        rng_seed=123,
    )
    option = OptionSpec(
        option_id=option_id,
        version="v1",
        state_machine="bounded_improvement",
        allowed_roles=(Role.SCOUT, Role.MECHANIST),
        capabilities=("propose", "test"),
        initiation={"requires_state": True},
        step_policy={"action": "continue"},
        stop_rule={"on_record": True},
        max_horizon=2,
        expected_cost={"tokens": 100},
        hard_cost={"tokens": 200},
        harness_eligibility=(harness_spec.harness_id,),
        prerequisites=(),
        output_contract={"candidate": True},
    )
    arm = AllocationArm(
        arm_id=ident("arm", "arm"),
        cell_id=cell_id,
        role=Role.SCOUT,
        option_id=option_id,
        harness_id=harness_spec.harness_id,
        horizon=2,
        cost_class="small",
        expected_cost={"tokens": 100},
        hard_cost={"tokens": 200},
    )
    branch = BranchSpec(
        branch_id=branch_id,
        arm_id=arm.arm_id,
        epoch=0,
        start_state_id=parent_state_id,
        frozen_record_threshold=2.5,
        role_snapshot_id=role_snapshot_id,
        option_id=option_id,
        option_version="v1",
        harness_id=harness_spec.harness_id,
        harness_version="baseline_v1",
        verifier_id=verifier_id,
        verifier_version="toy_v1",
        memory_view_id=None,
        memory_view_hash=digest("empty-memory"),
        horizon=2,
        budget={"tokens": 200, "verifier_calls": 2},
        seed=1234,
        generation_settings={"temperature": 1.0},
    )
    outcome_id = ident("branch_outcome", "outcome")
    outcome = BranchOutcome(
        outcome_id=outcome_id,
        branch_id=branch_id,
        branch_spec_hash=content_hash(branch.to_dict()),
        status=BranchStatus.CLOSED,
        descendant_proposal_ids=(proposal_id,),
        descendant_state_ids=(state_id,),
        evidence_ids=(evidence_id,),
        maximum_state_id=state_id,
        maximum_evidence_id=evidence_id,
        maximum_reward=3.0,
        costs={"tokens": 150},
        unused_budget={"tokens": 50},
        eligible_for_scheduler=True,
    )
    trace_id = ident("policy_trace", "trace")
    trace = PolicyTrace(
        trace_id=trace_id,
        branch_id=branch_id,
        role_snapshot_id=role_snapshot_id,
        role=Role.SCOUT,
        adapter_hash=role_snapshot.adapter_hash,
        prompts=("p0", "p1"),
        response_segments=("a", "bc"),
        token_masks=((True,), (True, False)),
        log_probabilities=((-0.5,), (-0.2, -0.3)),
    )
    control_outcome_id = ident("branch_outcome", "control-outcome")
    audit = AuditPair(
        audit_id=ident("audit_pair", "pair"),
        run_id=run_id,
        epoch=0,
        start_state_id=parent_state_id,
        cell_id=cell_id,
        frozen_record_threshold=2.5,
        role_snapshot_id=role_snapshot_id,
        harness_id=harness_spec.harness_id,
        verifier_id=verifier_id,
        horizon=2,
        resources={"tokens": 400},
        generation_settings={"temperature": 1.0},
        intervention_option_id=option_id,
        control_option_id=control_option_id,
        assignment_probability=0.5,
        assignment_seed=55,
        intervention_branch_id=branch_id,
        control_branch_id=branch_two,
        status=AuditStatus.CLOSED,
        intervention_outcome_id=outcome_id,
        control_outcome_id=control_outcome_id,
    )
    memory = CausalMemoryRecord(
        memory_id=ident("causal_memory", "record"),
        context={"cell_region": "constants"},
        intervention_option_id=option_id,
        audit_pair_ids=(audit.audit_id, ident("audit_pair", "pair-two")),
        propensities=(0.5, 0.5),
        effects=(1.0, 0.8),
        effect_mean=0.9,
        uncertainty=0.1,
        support=2,
        recency_epoch=2,
        scope="cell_region",
        contraindications=(),
        lineage_ids=(state_id,),
        status=MemoryStatus.PROMOTED,
        promotion_min_support=2,
    )
    second_trace_id = ident("policy_trace", "trace-two")
    learning = LearningGroup(
        group_id=ident("learning_group", "group"),
        role=Role.SCOUT,
        policy_snapshot_id=role_snapshot_id,
        start_cell_id=cell_id,
        context_id=ident("context", "context"),
        option_id=option_id,
        harness_id=harness_spec.harness_id,
        horizon=2,
        cost_class="small",
        generation_settings={"temperature": 1.0},
        frozen_record_threshold=2.5,
        channel=Channel.PRODUCTION,
        branch_ids=(branch_id, branch_two),
        trace_ids=(trace_id, second_trace_id),
        outcome_ids=(outcome_id, control_outcome_id),
        advantages=(0.5, -0.5),
        objective="ordergrad",
        objective_version="ordergrad_v1",
        top_m=1,
    )
    ledger = BudgetLedger(
        ledger_id=ident("budget", "ledger"),
        limits={"tokens": 1000, "verifier_calls": 10},
    )
    manifest = EpochManifest(
        manifest_id=ident("epoch_manifest", "epoch-0"),
        run_id=run_id,
        epoch=0,
        record_threshold=2.5,
        archive_snapshot_id=ident("archive_snapshot", "epoch-0"),
        archive_snapshot_hash=digest("archive"),
        scheduler_version="zero_inflated_tail_v1",
        scheduler_snapshot_id=ident("scheduler_snapshot", "epoch-0"),
        role_snapshot_ids={
            "scout": role_snapshot_id,
            "mechanist": ident("role_snapshot", "mechanist-0"),
            "challenger": ident("role_snapshot", "challenger-0"),
        },
        causal_memory_snapshot_id=ident("causal_memory_snapshot", "epoch-0"),
        option_ids=(option_id,),
        harness_ids=(harness_spec.harness_id,),
        verifier_id=verifier_id,
        verifier_version="toy_v1",
        descriptor_version="descriptor_v1",
        cell_map_version="cells_v1",
        fingerprint_version="fingerprint_v1",
        reporting_schema_version="report_v1",
        budget_ledger_id=ledger.ledger_id,
        allocation_plan_id=ident("allocation_plan", "epoch-0"),
        seed=999,
        component_schema_versions={"records": 1, "events": 1},
    )
    return (
        proposal, verified, evidence, descriptor, archive_cell, provenance,
        role_snapshot, option, harness_spec, arm, branch, outcome, trace, audit,
        memory, learning, ledger, manifest,
    )


def test_canonical_ids_are_order_stable_and_reject_non_json_numbers():
    left = {"b": [2, 3], "a": {"x": "é"}}
    right = {"a": {"x": "é"}, "b": (2, 3)}
    assert canonical_json(left) == canonical_json(right)
    assert content_hash(left) == content_hash(right)
    assert derive_id("proposal", left) == derive_id("proposal", right)
    with pytest.raises(CanonicalJSONError):
        canonical_json({"bad": float("nan")})


def test_rollout_seed_is_stable_and_logical_sample_specific():
    kwargs = dict(
        run_id=ident("run", "seeded"),
        epoch=4,
        allocation_id=ident("arm", "seeded"),
        branch_step=2,
        sample_index=7,
        role="scout",
        base_seed=42,
    )
    seed = rollout_seed(**kwargs)
    assert seed == rollout_seed(**kwargs)
    assert 0 <= seed < 2 ** 63
    assert seed != rollout_seed(**{**kwargs, "sample_index": 8})
    assert seed != rollout_seed(**{**kwargs, "role": "mechanist"})


@pytest.mark.parametrize("record", all_records(), ids=lambda item: item.RECORD_TYPE)
def test_every_required_record_round_trips_as_json(record):
    payload = record.to_dict()
    assert json.loads(record.to_json()) == payload
    restored = type(record).from_dict(payload)
    assert restored == record
    assert record_from_dict(payload) == record


def test_unknown_fields_are_preserved_in_immutable_extensions():
    record = all_records()[0]
    payload = record.to_dict()
    payload["future_diagnostic"] = {"values": [1, 2]}
    restored = Proposal.from_dict(payload)
    assert restored.extensions["future_diagnostic"]["values"] == (1, 2)
    assert restored.to_dict()["extensions"]["future_diagnostic"] == {"values": [1, 2]}
    with pytest.raises(TypeError):
        restored.extensions["new"] = True
    with pytest.raises(dataclasses.FrozenInstanceError):
        restored.problem_id = "changed"


def test_future_schema_is_rejected_before_interpretation():
    payload = all_records()[0].to_dict()
    payload["schema_version"] = 2
    with pytest.raises(UnsupportedSchemaVersion):
        Proposal.from_dict(payload)


def test_evidence_confirmation_and_infrastructure_invariants():
    valid = all_records()[2]
    payload = valid.to_dict()
    payload.update({"confirmed": True, "admitted": False})
    with pytest.raises(InvariantViolation, match="confirmed implies"):
        EvidencePacket.from_dict(payload)

    payload = valid.to_dict()
    payload.update({
        "failure_kind": "infrastructure",
        "resolved": True,
        "admitted": False,
        "confirmed": False,
        "internal_reward": None,
        "scientific_state_id": None,
        "descriptor_id": None,
        "fingerprint": "",
        "answer_payload": None,
    })
    with pytest.raises(InvariantViolation, match="resolution contradicts"):
        EvidencePacket.from_dict(payload)


@pytest.mark.parametrize(
    ("changes", "message"),
    [
        (
            {
                "admitted": False,
                "confirmed": False,
                "failure_kind": "constraint",
            },
            "scientific state",
        ),
        (
            {
                "admitted": False,
                "confirmed": False,
                "failure_kind": "constraint",
                "scientific_state_id": None,
            },
            "scientific reward",
        ),
        (
            {
                "admitted": False,
                "confirmed": False,
                "failure_kind": "constraint",
                "scientific_state_id": None,
                "internal_reward": None,
            },
            "descriptor",
        ),
        (
            {"descriptor_id": None},
            "admitted evidence must reference a descriptor",
        ),
        (
            {"fingerprint": ""},
            "scientific fingerprint",
        ),
    ],
)
def test_evidence_read_rejects_semantically_inconsistent_admission_fields(
    changes, message
):
    payload = all_records()[2].to_dict()
    payload.update(changes)

    with pytest.raises(InvariantViolation, match=message):
        EvidencePacket.from_dict(payload)


def test_wrong_reference_namespace_and_unfrozen_branch_are_rejected():
    branch = all_records()[10]
    payload = branch.to_dict()
    payload["role_snapshot_id"] = ident("evidence", "wrong-kind")
    with pytest.raises(InvariantViolation, match="role_snapshot_id"):
        BranchSpec.from_dict(payload)
    payload = branch.to_dict()
    payload["frozen"] = False
    with pytest.raises(InvariantViolation, match="frozen BranchSpec"):
        BranchSpec.from_dict(payload)


def test_policy_masks_audits_and_learning_inputs_are_strict():
    trace = all_records()[12]
    payload = trace.to_dict()
    payload["token_masks"][1] = [True]
    with pytest.raises(InvariantViolation, match="mask/log probabilities"):
        PolicyTrace.from_dict(payload)

    audit = all_records()[13]
    payload = audit.to_dict()
    payload["preassigned"] = False
    with pytest.raises(InvariantViolation, match="persisted before execution"):
        AuditPair.from_dict(payload)

    learning = all_records()[15]
    payload = learning.to_dict()
    payload["persisted_inputs"] = False
    with pytest.raises(InvariantViolation, match="persisted before backward"):
        LearningGroup.from_dict(payload)


def test_memory_promotion_requires_repeated_audit_backed_positive_effect():
    memory = all_records()[14]
    payload = memory.to_dict()
    payload.update({"effects": [0.1, 0.0], "effect_mean": 0.05, "uncertainty": 0.1})
    with pytest.raises(InvariantViolation, match="positive conservative effect"):
        CausalMemoryRecord.from_dict(payload)
    payload = memory.to_dict()
    payload.update({"promotion_min_support": 3})
    with pytest.raises(InvariantViolation, match="repeated audit support"):
        CausalMemoryRecord.from_dict(payload)


def test_scheduler_cannot_consume_aborted_outcome():
    outcome = all_records()[11]
    payload = outcome.to_dict()
    payload.update({
        "status": "aborted",
        "eligible_for_scheduler": True,
        "maximum_state_id": None,
        "maximum_evidence_id": None,
        "maximum_reward": None,
    })
    with pytest.raises(InvariantViolation, match="closed branch"):
        BranchOutcome.from_dict(payload)


def test_nested_record_mappings_cannot_be_mutated_or_bypassed():
    branch = all_records()[10]
    with pytest.raises(TypeError):
        branch.generation_settings["temperature"] = 99
    with pytest.raises(TypeError):
        dict.__setitem__(branch.generation_settings, "temperature", 99)
    assert copy.deepcopy(branch) == branch
    assert pickle.loads(pickle.dumps(branch)) == branch


@pytest.mark.parametrize("bad", [True, 1.9, "1"])
def test_rollout_seed_rejects_non_integer_logical_indices(bad):
    with pytest.raises(ValueError, match="epoch"):
        rollout_seed(
            run_id="run",
            epoch=bad,
            allocation_id="allocation",
            branch_step=0,
            sample_index=0,
            role="scout",
        )


def test_persisted_records_require_type_schema_and_clean_extensions():
    proposal = all_records()[0]
    payload = proposal.to_dict()
    payload.pop("record_type")
    with pytest.raises(SchemaValidationError, match="record_type"):
        Proposal.from_dict(payload)
    payload = proposal.to_dict()
    payload.pop("schema_version")
    with pytest.raises(SchemaValidationError, match="schema_version"):
        Proposal.from_dict(payload)
    payload = proposal.to_dict()
    payload["extensions"] = {"source_text": "shadow"}
    with pytest.raises(SchemaValidationError, match="shadow"):
        Proposal.from_dict(payload)


def test_branch_maximum_references_are_all_or_none():
    outcome = all_records()[11]
    payload = outcome.to_dict()
    payload.update(
        maximum_state_id=outcome.maximum_state_id,
        maximum_evidence_id=None,
        maximum_reward=None,
    )
    with pytest.raises(InvariantViolation, match="present together"):
        BranchOutcome.from_dict(payload)


def test_policy_trace_rejects_non_text_segments():
    trace = all_records()[12]
    payload = trace.to_dict()
    payload["prompts"][0] = 7
    with pytest.raises(InvariantViolation, match="prompts must be strings"):
        PolicyTrace.from_dict(payload)


def test_harness_identity_changes_with_any_behavior_change():
    original = harness()
    changed = HarnessSpec.create(
        version=original.version,
        instructions=original.instructions + " Be concise.",
        tools=original.tools,
        intermediate_tests=original.intermediate_tests,
        scaffolding=original.scaffolding,
        diagnostic_feedback=original.diagnostic_feedback,
        tool_policy_version=original.tool_policy_version,
    )
    assert original.harness_id != changed.harness_id
    assert original.spec_hash != changed.spec_hash
