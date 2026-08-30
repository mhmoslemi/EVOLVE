"""Tensor-parallel vLLM generation for EVOLVE's frozen role adapters.

The owning runtime never keeps this engine resident beside the HF training
backbone.  It phase-switches at epoch barriers so the same physical generation
GPUs hold one model implementation at a time.
"""

from __future__ import annotations

import gc
import hashlib
from importlib.metadata import PackageNotFoundError, version
from pathlib import Path
from typing import AbstractSet, Any, Dict, Mapping, Optional, Tuple

from evolve.types import Role


class VLLMRuntimeError(RuntimeError):
    """The vLLM engine or one of its frozen LoRA requests is invalid."""


def _positive_lora_id(snapshot_id: str) -> int:
    digest = hashlib.sha256(snapshot_id.encode("utf-8")).digest()
    # vLLM 0.28 stores request LoRA IDs in a NumPy int32 array even though
    # LoRARequest annotates the field as an unconstrained Python int.
    value = int.from_bytes(digest[:4], "big") & ((1 << 31) - 1)
    return value or 1


def _build_engine_options(
    config: Any, *, supported_engine_args: Optional[AbstractSet[str]]
) -> Dict[str, Any]:
    """Translate EVOLVE's frozen settings across supported vLLM APIs."""

    options: Dict[str, Any] = dict(
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
        enable_lora=True,
        max_lora_rank=config.lora_rank,
        max_loras=3,
        max_cpu_loras=6,
        fully_sharded_loras=config.vllm_fully_sharded_loras,
        seed=config.seed,
    )
    if config.vllm_quantization != "auto":
        options["quantization"] = config.vllm_quantization

    # ``vllm_swap_space_gb`` remains in the resolved schema so old EVOLVE
    # configs and resumes remain readable, but vLLM 0.28 removed the public
    # EngineArgs field. Do not forward it, even when an older install happens
    # to expose a similarly named argument: runtime behavior is version-frozen.

    split_devices = tuple(config.runtime_gpu_ids) != tuple(config.gpu_ids)
    if split_devices:
        if (
            supported_engine_args is not None
            and "device_ids" not in supported_engine_args
        ):
            raise VLLMRuntimeError(
                "installed vLLM cannot enforce split training/generation GPUs: "
                "EngineArgs.device_ids is unavailable"
            )
        options["device_ids"] = list(config.vllm_device_indices)

    if supported_engine_args is not None:
        unsupported = set(options) - set(supported_engine_args)
        if unsupported:
            raise VLLMRuntimeError(
                "installed vLLM does not support required engine option(s): "
                + ", ".join(sorted(unsupported))
            )
    return options


def _engine_arg_names(engine_args_type: type) -> frozenset[str]:
    fields = getattr(engine_args_type, "__dataclass_fields__", None)
    if fields:
        return frozenset(fields)
    names = set()
    for base in reversed(engine_args_type.__mro__):
        names.update(getattr(base, "__annotations__", {}))
    if not names:
        raise VLLMRuntimeError(
            "could not inspect installed vLLM EngineArgs compatibility"
        )
    return frozenset(names)


def _require_vllm_openai_schema() -> None:
    """Validate vLLM's local OpenAI-compatible schema dependency."""

    try:
        from openai.types.responses import NamespaceTool  # noqa: F401
    except (ImportError, ModuleNotFoundError) as exc:
        try:
            installed = version("openai")
        except PackageNotFoundError:
            installed = "not installed"
        raise VLLMRuntimeError(
            "vLLM 0.28 offline startup requires openai>=2.25.0 for the local "
            f"NamespaceTool schema (installed: {installed}). This dependency "
            "does not select or call an OpenAI model; EVOLVE still uses the "
            "configured Qwen backbone. Install with: python -m pip install "
            "'openai>=2.25.0,<3'"
        ) from exc


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

        _require_vllm_openai_schema()
        try:
            from vllm import LLM
            from vllm.engine.arg_utils import EngineArgs
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
        self._lora_id_owners: Dict[int, str] = {}
        self._request_ids = set()
        engine_options = _build_engine_options(
            config, supported_engine_args=_engine_arg_names(EngineArgs)
        )
        self.engine = LLM(**engine_options)

    def generate(
        self,
        *,
        prompt: str,
        request_id: str,
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
        if not isinstance(request_id, str) or not request_id:
            raise VLLMRuntimeError("vLLM generation requires a stable request ID")
        if request_id in self._request_ids:
            raise VLLMRuntimeError(
                f"vLLM request ID alias detected: {request_id}"
            )
        self._request_ids.add(request_id)
        lora_id = _positive_lora_id(role_snapshot_id)
        existing_owner = self._lora_id_owners.get(lora_id)
        if existing_owner is not None and existing_owner != role_snapshot_id:
            raise VLLMRuntimeError(
                "distinct role snapshots collided on the bounded vLLM LoRA ID"
            )
        self._lora_id_owners[lora_id] = role_snapshot_id
        lora_request = LoRARequest(
            f"evolve_{role.value}_{role_snapshot_id[-12:]}",
            lora_id,
            str(self.adapter_paths[role]),
        )
        sampling = SamplingParams(
            max_tokens=int(max_new_tokens),
            temperature=float(temperature),
            top_p=float(top_p),
            seed=int(seed),
            logprobs=1,
        )
        try:
            outputs = self.engine.generate(
                [prompt],
                sampling_params=sampling,
                lora_request=lora_request,
                use_tqdm=False,
            )
        except BaseException:
            # A failed call may be retried by the controller under the same
            # logical sample identity after an engine restart, but never
            # aliased to another live request in this engine instance.
            self._request_ids.discard(request_id)
            raise
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


__all__ = [
    "TensorParallelVLLM",
    "VLLMRuntimeError",
    "_build_engine_options",
    "_engine_arg_names",
    "_require_vllm_openai_schema",
]
