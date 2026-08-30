from __future__ import annotations

import json
from types import SimpleNamespace

import pytest

from evolve.types import FrozenDict
from evolve.workers.runtime import _apply_chat_template, _prompt_json
from evolve.workers.vllm_runtime import (
    VLLMRuntimeError,
    _build_engine_options,
    _engine_arg_names,
    _positive_lora_id,
)


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
