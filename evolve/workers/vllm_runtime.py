"""Tensor-parallel vLLM generation for EVOLVE's frozen role adapters.

The owning runtime never keeps this engine resident beside the HF training
backbone.  It phase-switches at epoch barriers so the same physical generation
GPUs hold one model implementation at a time.
"""

from __future__ import annotations

import gc
import hashlib
from pathlib import Path
from typing import Any, Mapping, Tuple

from evolve.types import Role


class VLLMRuntimeError(RuntimeError):
    """The vLLM engine or one of its frozen LoRA requests is invalid."""


def _positive_lora_id(snapshot_id: str) -> int:
    digest = hashlib.sha256(snapshot_id.encode("utf-8")).digest()
    value = int.from_bytes(digest[:8], "big") & ((1 << 63) - 1)
    return value or 1


class TensorParallelVLLM:
    """One offline vLLM engine sharded across every configured generation GPU."""

    def __init__(self, *, config: Any, adapter_paths: Mapping[Role, Path]) -> None:
        if config.vllm_tensor_parallel_size != len(config.gpu_ids):
            raise VLLMRuntimeError(
                "vLLM tensor parallel size must equal the authoritative "
                "generation GPU count"
            )
        paths = {role: Path(path).resolve() for role, path in adapter_paths.items()}
        if set(paths) != {Role.SCOUT, Role.MECHANIST, Role.CHALLENGER}:
            raise VLLMRuntimeError("vLLM requires one artifact for each EVOLVE role")
        missing = [str(path) for path in paths.values() if not path.is_dir()]
        if missing:
            raise VLLMRuntimeError(
                "vLLM role adapter artifact(s) are missing: " + ", ".join(missing)
            )
        if len({str(path) for path in paths.values()}) != len(paths):
            raise VLLMRuntimeError("role adapters cannot alias one filesystem path")

        try:
            from vllm import LLM
        except ImportError as exc:
            raise VLLMRuntimeError(
                "generation_backend=vllm requires the vllm package"
            ) from exc

        # `load_in_4bit` belongs exclusively to the HF/Unsloth training
        # loader. Never forward it as vLLM in-flight BitsAndBytes quantization:
        # doing so changes the inference base and can invalidate role adapters.
        # A pre-quantized BnB repository is also incompatible with tensor
        # parallel vLLM, so reject that topology before allocating model memory.
        try:
            from transformers import AutoConfig

            hf_config = AutoConfig.from_pretrained(
                config.model_name, trust_remote_code=True
            )
            quantization = getattr(hf_config, "quantization_config", None) or {}
            quant_method = str(quantization.get("quant_method", "")).lower()
            if quant_method == "bitsandbytes" and config.vllm_tensor_parallel_size > 1:
                raise VLLMRuntimeError(
                    "pre-quantized BitsAndBytes checkpoints cannot use vLLM "
                    "tensor parallelism; use a native/MXFP4 inference base or "
                    "one GPU. Training load_in_4bit does not quantize vLLM."
                )
        except VLLMRuntimeError:
            raise
        except Exception as exc:
            raise VLLMRuntimeError(
                "could not inspect the inference model quantization before vLLM startup"
            ) from exc

        self.config = config
        self.adapter_paths = paths
        engine_options = dict(
            model=config.model_name,
            trust_remote_code=True,
            tensor_parallel_size=config.vllm_tensor_parallel_size,
            distributed_executor_backend="mp",
            gpu_memory_utilization=config.vllm_gpu_memory_utilization,
            max_model_len=config.max_seq_length,
            max_num_seqs=config.vllm_max_num_seqs,
            max_num_batched_tokens=config.vllm_max_num_batched_tokens,
            enable_chunked_prefill=True,
            enable_prefix_caching=config.vllm_enable_prefix_caching,
            enforce_eager=config.vllm_enforce_eager,
            cpu_offload_gb=config.vllm_cpu_offload_gb,
            swap_space=config.vllm_swap_space_gb,
            enable_lora=True,
            max_lora_rank=config.lora_rank,
            max_loras=3,
            max_cpu_loras=6,
            fully_sharded_loras=config.vllm_fully_sharded_loras,
            seed=config.seed,
        )
        if config.vllm_quantization != "auto":
            engine_options["quantization"] = config.vllm_quantization
        self.engine = LLM(**engine_options)

    def generate(
        self,
        *,
        prompt: str,
        role: Role,
        role_snapshot_id: str,
        seed: int,
        max_new_tokens: int,
        temperature: float,
        top_p: float,
    ) -> Tuple[str, Tuple[int, ...], Tuple[float, ...]]:
        try:
            from vllm import SamplingParams
            from vllm.lora.request import LoRARequest
        except ImportError as exc:
            raise VLLMRuntimeError("installed vLLM has no LoRA request API") from exc

        if role not in self.adapter_paths:
            raise VLLMRuntimeError(f"no vLLM adapter registered for role {role.value}")
        lora_request = LoRARequest(
            f"evolve_{role.value}_{role_snapshot_id[-12:]}",
            _positive_lora_id(role_snapshot_id),
            str(self.adapter_paths[role]),
        )
        sampling = SamplingParams(
            max_tokens=int(max_new_tokens),
            temperature=float(temperature),
            top_p=float(top_p),
            seed=int(seed),
            logprobs=1,
        )
        outputs = self.engine.generate(
            [prompt],
            sampling_params=sampling,
            lora_request=lora_request,
            use_tqdm=False,
        )
        if len(outputs) != 1 or len(outputs[0].outputs) != 1:
            raise VLLMRuntimeError("vLLM returned an unexpected output cardinality")
        completion = outputs[0].outputs[0]
        token_ids = tuple(int(token_id) for token_id in completion.token_ids)
        token_logprobs = []
        raw_logprobs = completion.logprobs or ()
        if len(raw_logprobs) != len(token_ids):
            raise VLLMRuntimeError("vLLM did not return one log probability per token")
        for token_id, alternatives in zip(token_ids, raw_logprobs):
            selected = alternatives.get(token_id)
            if selected is None:
                raise VLLMRuntimeError(
                    "vLLM logprobs omitted the generated token despite logprobs=1"
                )
            value = getattr(selected, "logprob", selected)
            token_logprobs.append(float(value))
        return completion.text, token_ids, tuple(token_logprobs)

    def shutdown(self) -> None:
        shutdown = getattr(self.engine, "shutdown", None)
        if not callable(shutdown):
            engine = getattr(self.engine, "llm_engine", None)
            shutdown = getattr(engine, "shutdown", None)
        if callable(shutdown):
            shutdown()
        del self.engine
        gc.collect()
        try:
            import torch

            if torch.cuda.is_available():
                torch.cuda.empty_cache()
                torch.cuda.ipc_collect()
        except (ImportError, RuntimeError):
            pass


__all__ = ["TensorParallelVLLM", "VLLMRuntimeError"]
