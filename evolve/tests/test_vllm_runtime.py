from __future__ import annotations

import json
from types import SimpleNamespace

import pytest

from evolve.ids import content_hash, content_id
from evolve.options import (
    ExploreOption,
    BranchStepResult,
    build_option_context,
    create_explore_option_spec,
    execute_branch,
)
from evolve.types import AllocationArm, BranchSpec, FrozenDict, Proposal, Role, RoleSnapshot
from evolve.verifier import (
    PersistedAnswerPayload,
    ProblemScientificAdapter,
    VerificationPolicy,
    verify_persisted_answer,
)
from evolve.workers.runtime import LiveEvolveRuntime, _apply_chat_template, _prompt_json
from evolve.workers.vllm_runtime import (
    VLLMRuntimeError,
    _build_engine_options,
    _configure_vllm_file_logging,
    _engine_arg_names,
    _logging_config_document,
    _positive_lora_id,
)
from problems.evolve_toy import EvolveToyProblem


def _config(*, split: bool = False):
    generation_ids = (1,) if split else (0,)
    runtime_ids = (0, 1) if split else generation_ids
    return SimpleNamespace(
        model_name="org/model",
        vllm_tensor_parallel_size=1,
        vllm_gpu_memory_utilization=0.85,
        max_seq_length=4096,
        vllm_max_num_seqs=4,
        vllm_max_num_batched_tokens=2048,
        vllm_enable_prefix_caching=True,
        vllm_enforce_eager=True,
        vllm_cpu_offload_gb=0.0,
        lora_rank=32,
        vllm_fully_sharded_loras=True,
        seed=42,
        vllm_quantization="auto",
        vllm_swap_space_gb=4.0,
        runtime_gpu_ids=runtime_ids,
        gpu_ids=generation_ids,
        vllm_device_indices=(1,) if split else (0,),
    )


def test_vllm_lora_id_is_stable_positive_int32():
    first = _positive_lora_id("role_snapshot:scout-epoch-1")

    assert first == _positive_lora_id("role_snapshot:scout-epoch-1")
    assert 1 <= first <= (1 << 31) - 1


def test_vllm_028_silently_omits_removed_swap_space():
    config = _config()
    legacy = _build_engine_options(config, supported_engine_args=None)
    supported = set(legacy) - {"swap_space"}

    current = _build_engine_options(config, supported_engine_args=supported)

    assert "swap_space" not in current
    assert current["cpu_offload_gb"] == 0.0


def test_vllm_logging_is_routed_to_run_local_append_only_file(
    tmp_path, monkeypatch
):
    log_path = tmp_path / "logs" / "workers" / "vllm.log"
    document = _logging_config_document(log_path)

    handler = document["handlers"]["vllm_file"]
    assert handler["class"] == "logging.FileHandler"
    assert handler["filename"] == str(log_path.resolve())
    assert handler["mode"] == "a"
    assert document["loggers"]["vllm"]["propagate"] is False

    monkeypatch.setenv("VLLM_CONFIGURE_LOGGING", "0")
    monkeypatch.setenv("VLLM_LOGGING_CONFIG_PATH", "previous.json")
    monkeypatch.setenv("VLLM_LOGGING_COLOR", "1")
    config_path = _configure_vllm_file_logging(log_path)
    assert config_path == log_path.resolve().with_suffix(".logging.json")
    assert json.loads(config_path.read_text(encoding="utf-8")) == document
    assert config_path.parent == log_path.resolve().parent


def test_engine_argument_discovery_falls_back_to_annotations():
    class FakeEngineArgs:
        model: str
        device_ids: list[int]

    assert _engine_arg_names(FakeEngineArgs) == frozenset({"model", "device_ids"})


def test_split_topology_passes_only_vllm_logical_device_ids():
    config = _config(split=True)
    legacy = _build_engine_options(_config(), supported_engine_args=None)
    supported = (set(legacy) - {"swap_space"}) | {"device_ids"}

    options = _build_engine_options(config, supported_engine_args=supported)

    assert options["device_ids"] == [1]
    assert options["tensor_parallel_size"] == 1


def test_split_topology_rejects_vllm_without_device_selection_support():
    config = _config(split=True)
    legacy = _build_engine_options(_config(), supported_engine_args=None)
    supported = set(legacy) - {"swap_space"}

    with pytest.raises(VLLMRuntimeError, match="EngineArgs.device_ids"):
        _build_engine_options(config, supported_engine_args=supported)


@pytest.mark.parametrize("thinking", [False, True])
def test_chat_template_receives_resolved_thinking_mode(thinking):
    class Tokenizer:
        def apply_chat_template(self, messages, **kwargs):
            self.messages = messages
            self.kwargs = kwargs
            return "rendered"

    tokenizer = Tokenizer()
    messages = [{"role": "user", "content": "prompt"}]

    assert _apply_chat_template(tokenizer, messages, thinking=thinking) == "rendered"
    assert tokenizer.messages == messages
    assert tokenizer.kwargs == {
        "tokenize": False,
        "add_generation_prompt": True,
        "enable_thinking": thinking,
    }


def test_prompt_json_thaws_nested_runtime_mappings():
    value = FrozenDict(
        {
            "records": [
                {"context": {"cell": "north_east"}, "effect": 0.25}
            ]
        }
    )

    assert json.loads(_prompt_json(value)) == {
        "records": [
            {"context": {"cell": "north_east"}, "effect": 0.25}
        ]
    }


def test_multistep_branch_uses_newly_admitted_parent_before_barrier(tmp_path):
    """A local descendant must not require visibility in the frozen archive."""

    def ident(namespace, label):
        return content_id(namespace, {"label": label})

    run_id = ident("run", "local-parent")
    branch_id = ident("branch", "local-parent")
    cell_id = ident("cell", "local-parent")
    harness_id = ident("harness", "local-parent")
    snapshot = RoleSnapshot(
        snapshot_id=ident("role_snapshot", "local-parent"),
        run_id=run_id,
        epoch=0,
        role=Role.SCOUT,
        adapter_id=ident("adapter", "local-parent"),
        adapter_version="adapter_v000000",
        adapter_hash=content_hash({"adapter": "local-parent"}),
        optimizer_state_id=ident("optimizer", "local-parent"),
        policy_version="policy_v1",
        rng_seed=7,
    )
    spec = create_explore_option_spec(
        max_horizon=2,
        hard_cost={"verifier_calls": 2.0},
        harness_eligibility=(harness_id,),
    )
    option = ExploreOption(spec)
    arm = AllocationArm(
        arm_id=ident("arm", "local-parent"),
        cell_id=cell_id,
        role=Role.SCOUT,
        option_id=spec.option_id,
        harness_id=harness_id,
        horizon=2,
        cost_class="small",
        expected_cost={"verifier_calls": 2.0},
        hard_cost={"verifier_calls": 2.0},
    )
    branch = BranchSpec(
        branch_id=branch_id,
        arm_id=arm.arm_id,
        epoch=0,
        start_state_id=ident("state", "frozen-parent"),
        frozen_record_threshold=2.0,
        role_snapshot_id=snapshot.snapshot_id,
        option_id=spec.option_id,
        option_version=spec.version,
        harness_id=harness_id,
        harness_version="baseline_v1",
        verifier_id=ident("verifier", "toy"),
        verifier_version="answer_schema_v1",
        memory_view_id=None,
        memory_view_hash=content_hash({"memory": []}),
        horizon=2,
        budget={"verifier_calls": 2.0},
        seed=7,
        generation_settings={"max_new_tokens": 16},
    )
    context = build_option_context(branch=branch, arm=arm)
    adapter = ProblemScientificAdapter(EvolveToyProblem({}))
    policy = VerificationPolicy.create(version="local-parent-v1")
    results = []

    class FrozenArtifacts:
        def representative_state(self, state_id):
            raise AssertionError(
                f"transient state {state_id} was incorrectly read from the frozen archive"
            )

    runtime = LiveEvolveRuntime.__new__(LiveEvolveRuntime)
    runtime.state = SimpleNamespace(
        archive=SimpleNamespace(artifacts=FrozenArtifacts())
    )

    def executor(request):
        if request.step_index == 0:
            assert request.parent_proposal is None
            assert request.parent_state is None
        else:
            assert request.parent_proposal == results[-1].proposal
            assert request.parent_state == results[-1].verification.state
            parent = runtime._parent_context(request)
            assert parent.code == results[-1].proposal.source_text
            assert tuple(parent.construction) == (2, -2)

        point = [2 - request.step_index, -2]
        source = f"def run_toy():\n    return {point!r}\n"
        proposal = Proposal(
            proposal_id=ident("proposal", f"step-{request.step_index}"),
            run_id=run_id,
            problem_id="evolve_toy",
            source_text=source,
            source_hash=content_hash(source),
            parent_state_id=request.parent_state_id,
            branch_id=branch_id,
            parsed_candidate=point,
        )
        answer_path = tmp_path / f"step-{request.step_index}.json"
        answer_path.write_text(json.dumps(point), encoding="utf-8")
        persisted = PersistedAnswerPayload.create(
            problem_id="evolve_toy",
            artifact_uri=str(answer_path),
            payload=point,
        )
        verification = verify_persisted_answer(
            adapter=adapter,
            proposal=proposal,
            persisted_answer=persisted,
            verification_policy=policy,
            harness_id=harness_id,
            policy_snapshot_id=snapshot.snapshot_id,
        )
        result = BranchStepResult(
            proposal=proposal,
            verification=verification,
            costs={"verifier_calls": 1.0},
        )
        results.append(result)
        return result

    execution = execute_branch(
        branch=branch,
        arm=arm,
        option=option,
        context=context,
        role_snapshot=snapshot,
        executor=executor,
    )

    assert len(execution.observations) == 2
    assert execution.outcome.infrastructure_aborted is False
