"""Live EVOLVE worker wiring over one model identity and three role LoRAs.

HF generation keeps the training backbone resident. vLLM generation instead
phase-switches at barriers: persisted HF adapters are released before one
tensor-parallel vLLM engine starts, and vLLM is fully shut down before HF role
learning resumes. The two full model implementations are never resident at the
same time.
"""

from __future__ import annotations

import gc
import io
import json
import os
import random
import threading
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Any, Mapping, Optional, Sequence

from evolve.ids import content_hash
from evolve.learning.trainer import GradientStepRequest, GradientStepResult
from evolve.options.branch import BranchStepRequest, BranchStepResult
from evolve.roles.adapters import thaw_json
from evolve.roles.backend import BackboneIdentity, NamedAdapterBackendPort
from evolve.runio import (
    ImmutableWriteError,
    atomic_write_bytes,
    atomic_write_json,
    write_immutable_json,
)
from evolve.types import Role
from evolve.verifier.models import VerificationPolicy
from evolve.workers.verification import (
    DurableVerificationConflict,
    GenerationOutcome,
    build_proposal_and_verify,
    infrastructure_step_result,
    load_generation_arrival,
    load_verified_step,
    persist_generation_arrival,
    persist_verified_step,
)
from evolve.workers.resources import ResourceLeaseManager
from evolve.workers.generation import (
    GenerationJob,
    GenerationParameters,
    requests_for_job,
)
from problems.base import ParentContext, build_problem_prompt


ROLE_INSTRUCTIONS = {
    Role.SCOUT: "Search broadly and propose a structurally different approach.",
    Role.MECHANIST: "Develop the mechanism and make one focused, justified improvement.",
    Role.CHALLENGER: "Attack assumptions and make only a bounded minimal repair.",
}


class LiveRuntimeContractError(RuntimeError):
    """Frozen runtime identity or durable role/generation state changed."""


def _prompt_json(value: Any) -> str:
    """Serialize recursively frozen runtime values for model-visible prompts."""

    return json.dumps(
        thaw_json(value), ensure_ascii=False, sort_keys=True
    )


def _apply_chat_template(
    tokenizer: Any,
    messages: Sequence[Mapping[str, Any]],
    *,
    thinking: bool,
) -> str:
    """Render the configured Qwen thinking mode into the frozen prompt."""

    return tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True,
        enable_thinking=bool(thinking),
    )


def backbone_identity_for_config(config: Any) -> BackboneIdentity:
    revision = str(config.problem_config.get("model_revision", "local-or-default"))
    identity = {
        "model_name": config.model_name,
        "revision": revision,
        "training_backend": config.backend,
        "max_seq_length": config.max_seq_length,
        "load_in_4bit": config.load_in_4bit,
    }
    return BackboneIdentity.create(
        model_name=config.model_name,
        revision=revision,
        weights_hash=content_hash({"model": config.model_name, "revision": revision}),
        config_hash=content_hash(identity),
    )


class LiveEvolveRuntime:
    """Model-owning implementation of branch and barrier worker callbacks."""

    def __init__(self, *, config: Any, adapter: Any, layout: Any, state: Any) -> None:
        if config.backend not in {"hf", "unsloth"}:
            raise RuntimeError(
                "production EVOLVE requires backend: hf or unsloth so exact "
                "named-adapter isolation can be enforced"
            )
        if config.num_gpus < 1:
            raise RuntimeError("a live EVOLVE run requires at least one generation GPU")

        requested_mask = ",".join(str(item) for item in config.runtime_gpu_ids)
        existing_mask = os.environ.get("CUDA_VISIBLE_DEVICES")
        if existing_mask is not None and existing_mask != requested_mask:
            raise RuntimeError(
                "CUDA_VISIBLE_DEVICES disagrees with the resolved training/generation "
                "GPU topology: "
                f"{existing_mask!r} != {requested_mask!r}"
            )
        os.environ["CUDA_VISIBLE_DEVICES"] = requested_mask

        self.config = config
        self.adapter = adapter
        self.problem = adapter.problem
        self.layout = layout
        self.state = state
        self.verification_policy = VerificationPolicy.create(
            version="evolve_engine_v1", production=not config.method_incomplete
        )
        self.leases = ResourceLeaseManager()
        self.backend = None
        self.model = None
        self.tokenizer = None
        self.port = None
        self.optimizers = None
        self.vllm = None
        self._vllm_log_announced = False
        self._rng_restore_checked = False
        self._role_artifacts = {
            role: dict(payload)
            for role, payload in getattr(state, "role_artifacts", {}).items()
        }
        self._generation_lock = threading.RLock()
        self._branch_pool = ThreadPoolExecutor(
            max_workers=int(config.evolve.workers.max_inflight_branches),
            thread_name_prefix="evolve-branch",
        )
        try:
            self._load_hf()
        except BaseException:
            # Deadline/user interrupts may arrive while the initial backbone is
            # loading, before the controller receives an EngineWorkers handle.
            # Release any partially constructed model and executor resources at
            # the ownership boundary, then preserve the original exception.
            self.shutdown()
            raise

    def submit_branch(self, callback: Any):
        """Submit one frozen branch to the persistent bounded controller pool."""

        return self._branch_pool.submit(callback)

    def _load_hf(self) -> None:
        if self.model is not None:
            return
        if self.vllm is not None:
            raise RuntimeError("cannot load HF while vLLM is resident")
        from evolve.roles.model import HFBackbone, UnslothBackbone

        backend_type = HFBackbone if self.config.backend == "hf" else UnslothBackbone
        self.backend = backend_type(self.config)
        self.model, loaded_tokenizer = self.backend.load()
        self.tokenizer = loaded_tokenizer
        default_config = self.model.peft_config["default"]
        self.port = NamedAdapterBackendPort(
            self.backend,
            backbone=backbone_identity_for_config(self.config),
            adapter_config=default_config,
        )
        # PEFT adapter loading can replace parameter objects. Restore all role
        # adapters first, then bind optimizers to the final role parameters.
        self._restore_adapters_if_present(self.state)
        self.optimizers = self._create_optimizers()
        self._restore_optimizers_if_present(self.state)
        self._restore_rng_if_present(self.state)

    def _release_cuda(self) -> None:
        gc.collect()
        try:
            import torch

            if torch.cuda.is_available():
                torch.cuda.synchronize()
                torch.cuda.empty_cache()
                torch.cuda.ipc_collect()
        except (ImportError, RuntimeError):
            pass

    def _unload_hf(self) -> None:
        if self.model is None:
            # A signal or loader exception can occur after the backend object
            # is created but before ``load()`` returns the model. Drop those
            # partial references and still flush allocator state.
            self.optimizers = None
            self.port = None
            self.tokenizer = None
            self.backend = None
            self._release_cuda()
            return
        self.optimizers = None
        self.port = None
        self.model = None
        self.tokenizer = None
        self.backend = None
        self._release_cuda()

    def _latest_adapter_paths(self) -> Mapping[Role, Path]:
        paths = {}
        for role in self.state.role_registry.roles:
            document = self._artifact_document(self.state, role)
            if document is None:
                raise RuntimeError(
                    f"missing persisted {role.value} adapter before vLLM startup"
                )
            artifact_root = self.layout.path(
                f"roles/{role.value}/{document['adapter']}"
            )
            manifest_path = artifact_root / "evolve_adapter_manifest.json"
            if not manifest_path.is_file():
                raise RuntimeError(
                    f"missing immutable adapter manifest for role {role.value}"
                )
            manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
            paths[role] = artifact_root / manifest["adapter_relative_path"]
        return paths

    def _load_vllm(self) -> None:
        if self.vllm is not None:
            return
        if self.model is not None:
            raise RuntimeError("HF must be released before vLLM startup")
        from evolve.workers.vllm_runtime import TensorParallelVLLM

        log_path = self.layout.path("logs/workers/vllm.log")
        if not self._vllm_log_announced:
            print(f"EVOLVE · vLLM diagnostics · {log_path}", flush=True)
            self._vllm_log_announced = True
        self.vllm = TensorParallelVLLM(
            config=self.config,
            adapter_paths=self._latest_adapter_paths(),
            log_path=log_path,
        )

    def _unload_vllm(self) -> None:
        if self.vllm is None:
            return
        engine = self.vllm
        self.vllm = None
        engine.shutdown()
        self._release_cuda()

    def _ensure_hf(self) -> None:
        if self.model is not None:
            return
        self._unload_vllm()
        self._load_hf()

    def _ensure_vllm(self) -> None:
        if self.vllm is not None:
            return
        if self.model is not None:
            raise RuntimeError(
                "HF adapters must be persisted and released before vLLM generation"
            )
        self._load_vllm()

    def _create_optimizers(self) -> Mapping[Role, Any]:
        import torch

        named = dict(self.model.named_parameters())
        optimizers = {}
        for role, manifest in self.port.parameter_manifests.items():
            parameters = [named[name] for name in manifest.parameter_names]
            optimizers[role] = torch.optim.AdamW(
                parameters, lr=float(self.config.learning_rate)
            )
        return optimizers

    def begin_epoch(self, state: Any) -> None:
        self.state = state
        if self.config.generation_backend == "vllm":
            # Engine barrier order persists all role adapters before begin_epoch.
            self._unload_hf()
            self._load_vllm()

    def _parent_context(self, request: BranchStepRequest) -> ParentContext:
        # Descendants admitted earlier in this branch are intentionally absent
        # from the frozen epoch archive until the barrier. The pure branch
        # executor carries that exact verified binding forward so only this
        # branch can use it as its next local parent. Initial branch parents
        # continue to resolve from the frozen archive.
        if request.parent_state is not None:
            state = request.parent_state
            proposal = request.parent_proposal
            assert proposal is not None
        else:
            state = self.state.archive.artifacts.representative_state(
                request.parent_state_id
            )
            proposal = self.state.archive.artifacts.proposal(state.proposal_id)
        answer = state.answer_payload
        construction = answer
        if isinstance(answer, Mapping):
            for key in ("h_values", "sequence", "point"):
                if key in answer:
                    construction = answer[key]
                    break
        return ParentContext(
            code=proposal.source_text,
            value=float(state.internal_reward),
            raw_score=state.raw_score,
            construction=construction,
        )

    def _render_prompt(self, request: BranchStepRequest, parent: ParentContext) -> str:
        memory_records = request.branch.generation_settings.get("memory_records", ())
        memory = ""
        if memory_records:
            memory = _prompt_json(memory_records)
        messages = [dict(item) for item in build_problem_prompt(self.problem, parent, memory)]
        harness = self.state.harness_registry.spec(request.branch.harness_id)
        instruction = (
            f"\n\n## Frozen EVOLVE assignment\n"
            f"Role: {request.arm.role.value}. {ROLE_INSTRUCTIONS[request.arm.role]}\n"
            f"Option action: {request.action}.\n"
            f"Harness: {harness.instructions}\n"
            f"Horizon step: {request.step_index + 1}/{request.branch.horizon}.\n"
            "Return the complete candidate in the problem's required format."
        )
        target_cell = self.state.archive.cell(request.arm.cell_id)
        target_descriptor = self.state.archive.descriptor(
            target_cell.descriptor_id
        )
        instruction += (
            "\nTarget archive cell descriptor: "
            + _prompt_json(target_descriptor.dimensions)
        )
        failed_candidate = request.branch.generation_settings.get(
            "refinement_source"
        )
        if failed_candidate:
            if request.action == "minimal_diagnostic_repair":
                instruction += (
                    "\n\nRefinement nursery source (make exactly one minimal "
                    "diagnostic-targeted change):\n" + str(failed_candidate)
                )
                diagnostics = request.branch.generation_settings.get(
                    "refinement_diagnostics", {}
                )
                if diagnostics:
                    instruction += (
                        "\n\nIndependent verifier diagnostics:\n"
                        + _prompt_json(diagnostics)
                    )
            else:
                instruction += (
                    "\n\nThis is the equal-cost fresh-continuation control. "
                    "Do not copy or repair the nursery source; propose a fresh "
                    "candidate from the verified start."
                )
        for index in range(len(messages) - 1, -1, -1):
            if messages[index].get("role") == "user":
                messages[index]["content"] = str(messages[index].get("content", "")) + instruction
                break
        return _apply_chat_template(
            self.tokenizer,
            messages,
            thinking=self.config.thinking,
        )

    def _generation_job_path(self, request: BranchStepRequest) -> Path:
        return self.layout.path(
            f"step{request.branch.epoch:02d}/branches/"
            f"{request.branch.branch_id}/steps/"
            f"step{request.step_index:03d}.generation-job.json"
        )

    def _persist_generation_contract(
        self, request: BranchStepRequest, prompt: str
    ):
        snapshot = self.state.role_registry.freeze_epoch(request.branch.epoch)[
            request.arm.role
        ]
        if snapshot.snapshot_id != request.branch.role_snapshot_id:
            raise LiveRuntimeContractError(
                "generation snapshot differs from frozen branch"
            )
        artifact = self._artifact_document(self.state, request.arm.role)
        if artifact is None:
            raise LiveRuntimeContractError(
                "generation role adapter has no durable artifact"
            )
        adapter_root = self.layout.path(
            f"roles/{request.arm.role.value}/{artifact['adapter']}"
        ).resolve()
        adapter_manifest = json.loads(
            (adapter_root / "evolve_adapter_manifest.json").read_text(
                encoding="utf-8"
            )
        )
        adapter_path = (adapter_root / adapter_manifest["adapter_relative_path"]).resolve()
        parameters = GenerationParameters(
            max_new_tokens=int(request.branch.generation_settings["max_new_tokens"]),
            temperature=float(request.branch.generation_settings["temperature"]),
            top_p=float(request.branch.generation_settings["top_p"]),
            micro_batch=int(self.config.gen_micro_batch),
        )
        job = GenerationJob.create(
            run_id=self.state.run_id,
            epoch=request.branch.epoch,
            allocation_id=request.arm.arm_id,
            branch_id=request.branch.branch_id,
            branch_step=request.step_index,
            role=request.arm.role,
            adapter_path=str(adapter_path),
            adapter_id=snapshot.adapter_id,
            adapter_version=snapshot.adapter_version,
            policy_snapshot=snapshot,
            option_id=request.branch.option_id,
            harness_id=request.branch.harness_id,
            prompt=prompt,
            sample_index_start=0,
            sample_count=1,
            generation_parameters=parameters,
            common_random_seed=request.branch.seed,
        )
        job_path = self._generation_job_path(request)
        try:
            write_immutable_json(job_path, job.to_dict())
        except ImmutableWriteError:
            durable = GenerationJob.from_dict(
                json.loads(job_path.read_text(encoding="utf-8"))
            )
            if durable.to_dict() != job.to_dict():
                raise LiveRuntimeContractError(
                    f"generation job identity conflict: {job_path}"
                )
            job = durable
        return requests_for_job(job)[0]

    def branch_step(self, request: BranchStepRequest) -> BranchStepResult:
        resources = self.problem.resource_requirements()
        evaluation_shares_model_gpu = bool(
            resources.exclusive_gpu
            and (
                self.config.kernel_gpu_id == self.config.training_gpu_id
                or self.config.kernel_gpu_id in self.config.gpu_ids
            )
        )
        generation_seed = None
        max_new_tokens = int(
            request.branch.generation_settings["max_new_tokens"]
        )
        temperature = float(request.branch.generation_settings["temperature"])
        top_p = float(request.branch.generation_settings["top_p"])
        generation = load_generation_arrival(
            run_dir=self.layout.run_dir, request=request
        )
        if generation is not None:
            generation_job_path = self._generation_job_path(request)
            if generation_job_path.is_file():
                generation_request = self._persist_generation_contract(
                    request, generation.prompt
                )
                generation_seed = generation_request.seed
                if generation.seed != generation_seed:
                    raise RuntimeError(
                        "durable response seed differs from its generation job"
                    )
            else:
                # Schema-v1 runs could persist an arrival before live
                # GenerationJob integration. Reuse that response under its
                # original deterministic seed instead of retroactively
                # assigning it a different sample identity.
                generation_seed = int(generation.seed)
                legacy_seed = request.branch.seed + request.step_index
                if generation_seed != legacy_seed:
                    raise RuntimeError(
                        "legacy durable response seed differs from its frozen branch"
                    )
                legacy_binding = {
                    "schema_version": 1,
                    "kind": "legacy_generation_arrival_binding",
                    "branch_id": request.branch.branch_id,
                    "step_index": request.step_index,
                    "seed": generation_seed,
                    "prompt_hash": content_hash(generation.prompt),
                }
                legacy_path = generation_job_path.with_name(
                    generation_job_path.name.replace(
                        ".generation-job.json", ".legacy-generation.json"
                    )
                )
                try:
                    write_immutable_json(legacy_path, legacy_binding)
                except ImmutableWriteError:
                    if json.loads(legacy_path.read_text(encoding="utf-8")) != legacy_binding:
                        raise RuntimeError(
                            f"legacy generation binding conflict: {legacy_path}"
                        )
        cached = load_verified_step(
            run_dir=self.layout.run_dir,
            epoch=request.branch.epoch,
            branch_id=request.branch.branch_id,
            step_index=request.step_index,
            parent_state_id=request.parent_state_id,
            generation=generation,
            adapter=self.adapter,
            run_id=self.state.run_id,
            problem_id=self.config.problem,
            harness_id=request.branch.harness_id,
            policy_snapshot_id=request.branch.role_snapshot_id,
        )
        if cached is not None:
            return cached
        reused_generation = generation is not None
        parent = None
        generation_lock_held = False
        generation_errors = []
        for attempt in (
            range(self.verification_policy.infrastructure_retry_limit + 1)
            if generation is None
            else ()
        ):
            failure_dir = self.layout.path(
                f"step{request.branch.epoch:02d}/branches/"
                f"{request.branch.branch_id}/steps"
            )
            failure_path = (
                failure_dir
                / f"step{request.step_index:03d}.generation-attempt{attempt:02d}.json"
            )
            if failure_path.is_file():
                failure_document = json.loads(
                    failure_path.read_text(encoding="utf-8")
                )
                if (
                    failure_document.get("schema_version") != 1
                    or failure_document.get("attempt") != attempt
                    or failure_document.get(
                        "branch_id", request.branch.branch_id
                    )
                    != request.branch.branch_id
                    or failure_document.get("step_index", request.step_index)
                    != request.step_index
                ):
                    raise RuntimeError(
                        f"generation retry artifact has another identity: {failure_path}"
                    )
                generation_errors.append(
                    RuntimeError(
                        f"durable {failure_document.get('exception_type', 'generation')}"
                        f": {failure_document.get('message', '')}"
                    )
                )
                continue
            try:
                self._generation_lock.acquire()
                generation_lock_held = True
                parent = self._parent_context(request)
                prompt = self._render_prompt(request, parent)
                generation_request = self._persist_generation_contract(
                    request, prompt
                )
                generation_seed = generation_request.seed
                if self.config.generation_backend == "vllm":
                    self._ensure_vllm()
                    text, token_ids, log_probabilities = self.vllm.generate(
                        prompt=prompt,
                        request_id=generation_request.vllm_request_id,
                        role=request.arm.role,
                        role_snapshot_id=request.branch.role_snapshot_id,
                        seed=generation_seed,
                        max_new_tokens=max_new_tokens,
                        temperature=temperature,
                        top_p=top_p,
                    )
                else:
                    self._ensure_hf()
                    import torch

                    # The chat template already renders all required special
                    # tokens. Use the same no-extra-special-token convention
                    # later used to recompute persisted policy likelihoods.
                    encoded = self.tokenizer(
                        prompt,
                        return_tensors="pt",
                        add_special_tokens=False,
                    )
                    device = next(self.model.parameters()).device
                    encoded = {
                        name: value.to(device) for name, value in encoded.items()
                    }
                    generator = torch.Generator(device=device)
                    generator.manual_seed(generation_seed)
                    with self.port.activate(
                        request.arm.role, training=False
                    ), torch.no_grad():
                        output = self.model.generate(
                            **encoded,
                            max_new_tokens=max_new_tokens,
                            temperature=temperature,
                            top_p=top_p,
                            do_sample=temperature > 0.0,
                            return_dict_in_generate=True,
                            output_scores=True,
                            generator=generator,
                            pad_token_id=self.tokenizer.pad_token_id,
                        )
                    prompt_length = encoded["input_ids"].shape[1]
                    generated_ids = output.sequences[0, prompt_length:]
                    token_ids = tuple(
                        int(value) for value in generated_ids.tolist()
                    )
                    text = self.tokenizer.decode(
                        generated_ids, skip_special_tokens=True
                    )
                    log_probabilities = tuple(
                        float(
                            torch.log_softmax(scores[0], dim=-1)[token_id].item()
                        )
                        for scores, token_id in zip(output.scores, generated_ids)
                    )
                generation = GenerationOutcome(
                    prompt=prompt,
                    text=text,
                    token_ids=token_ids,
                    log_probabilities=log_probabilities,
                    seed=generation_seed,
                )
                persist_generation_arrival(
                    run_dir=self.layout.run_dir,
                    request=request,
                    generation=generation,
                )

                # A shared evaluation GPU is exclusive in time. Keep the
                # generation lock through verification so another branch cannot
                # reload a model while the benchmark lease is active.
                if evaluation_shares_model_gpu:
                    if self.config.kernel_gpu_id in self.config.gpu_ids:
                        self._unload_vllm()
                    if self.config.kernel_gpu_id == self.config.training_gpu_id:
                        self._unload_hf()
                else:
                    self._generation_lock.release()
                    generation_lock_held = False
                break
            except (LiveRuntimeContractError, DurableVerificationConflict):
                if generation_lock_held:
                    self._generation_lock.release()
                    generation_lock_held = False
                raise
            except Exception as exc:
                generation_errors.append(exc)
                retryable = True
                if self.config.generation_backend == "vllm" and self.vllm is not None:
                    try:
                        # A synchronous engine failure may leave request/LoRA
                        # bookkeeping or worker processes unusable. Retry the
                        # same logical sample only in a fresh engine instance.
                        self._unload_vllm()
                    except Exception as cleanup_exc:
                        generation_errors.append(cleanup_exc)
                        retryable = False
                if generation_lock_held:
                    self._generation_lock.release()
                    generation_lock_held = False
                failure_dir.mkdir(parents=True, exist_ok=True)
                failure_document = {
                    "schema_version": 1,
                    "branch_id": request.branch.branch_id,
                    "step_index": request.step_index,
                    "attempt": attempt,
                    "seed": generation_seed,
                    "exception_type": type(exc).__name__,
                    "message": str(exc)[:2048],
                }
                try:
                    write_immutable_json(failure_path, failure_document)
                except ImmutableWriteError:
                    if json.loads(failure_path.read_text(encoding="utf-8")) != failure_document:
                        raise RuntimeError(
                            f"generation retry artifact conflict: {failure_path}"
                        ) from exc
                if not retryable:
                    break

        if generation is None:
            error = generation_errors[-1]
            return infrastructure_step_result(
                run_id=self.state.run_id,
                problem_id=self.config.problem,
                branch_id=request.branch.branch_id,
                epoch=request.branch.epoch,
                parent_state_id=request.parent_state_id,
                step_index=request.step_index,
                adapter=self.adapter,
                verification_policy=self.verification_policy,
                harness_id=request.branch.harness_id,
                policy_snapshot_id=request.branch.role_snapshot_id,
                run_dir=self.layout.run_dir,
                phase="generation",
                error=error,
            )

        if parent is None:
            parent = self._parent_context(request)
        if reused_generation and evaluation_shares_model_gpu:
            self._generation_lock.acquire()
            generation_lock_held = True
            if self.config.kernel_gpu_id in self.config.gpu_ids:
                self._unload_vllm()
            if self.config.kernel_gpu_id == self.config.training_gpu_id:
                self._unload_hf()
        evaluated: dict[str, Any] = {}

        def extract_answer(response_text: str) -> Any:
            if "payload" not in evaluated:
                if resources.exclusive_gpu:
                    # GPU scientific verification benchmarks the saved kernel
                    # payload below. Do not run the older reward path first.
                    evaluated["payload"] = self.problem.serialize_answer(
                        response_text
                    )
                    return evaluated["payload"]
                else:
                    reward = self.problem.compute_reward(
                        response_text,
                        parent,
                        timeout_s=float(self.config.sandbox_timeout_s),
                    )
                if not reward.valid:
                    evaluated["payload"] = None
                else:
                    candidate = reward.construction if reward.construction is not None else reward
                    evaluated["payload"] = self.problem.serialize_answer(candidate)
            return evaluated["payload"]

        def verify_call() -> BranchStepResult:
            return build_proposal_and_verify(
                run_id=self.state.run_id,
                problem_id=self.config.problem,
                branch_id=request.branch.branch_id,
                parent_state_id=request.parent_state_id,
                step_index=request.step_index,
                generation=generation,
                extract_answer=extract_answer,
                adapter=self.adapter,
                verification_policy=self.verification_policy,
                harness_id=request.branch.harness_id,
                policy_snapshot_id=request.branch.role_snapshot_id,
                run_dir=self.layout.run_dir,
            )

        try:
            if resources.exclusive_gpu:
                with self.leases.lease(
                    f"gpu:{self.config.kernel_gpu_id}",
                    holder=request.branch.branch_id,
                    timeout=float(resources.timeout_s) * 3.0,
                ):
                    result = verify_call()
            else:
                result = verify_call()
            persist_verified_step(
                run_dir=self.layout.run_dir,
                epoch=request.branch.epoch,
                branch_id=request.branch.branch_id,
                step_index=request.step_index,
                result=result,
            )
            return result
        except DurableVerificationConflict:
            raise
        except Exception as exc:
            return infrastructure_step_result(
                run_id=self.state.run_id,
                problem_id=self.config.problem,
                branch_id=request.branch.branch_id,
                epoch=request.branch.epoch,
                parent_state_id=request.parent_state_id,
                step_index=request.step_index,
                adapter=self.adapter,
                verification_policy=self.verification_policy,
                harness_id=request.branch.harness_id,
                policy_snapshot_id=request.branch.role_snapshot_id,
                run_dir=self.layout.run_dir,
                phase="persistence_or_verification",
                error=exc,
                generation=generation,
            )
        finally:
            if generation_lock_held:
                self._generation_lock.release()

    def _sequence_log_probability(
        self,
        prompt: str,
        response: str,
        token_mask: Sequence[bool],
        captured_token_ids: Sequence[int],
    ) -> Any:
        import torch

        prompt_ids = self.tokenizer(prompt, add_special_tokens=False).input_ids
        response_ids = (
            list(captured_token_ids)
            if captured_token_ids
            else self.tokenizer(response, add_special_tokens=False).input_ids
        )
        if not response_ids:
            return torch.zeros((), device=next(self.model.parameters()).device)
        if not prompt_ids:
            raise RuntimeError("policy prompt tokenization produced no tokens")
        if len(token_mask) != len(response_ids):
            raise RuntimeError(
                "persisted policy token mask does not match retokenized response; "
                "rejecting the learning input instead of training on different tokens"
            )
        ids = torch.tensor([prompt_ids + response_ids], device=next(self.model.parameters()).device)
        logits = self.model(input_ids=ids).logits[:, :-1, :]
        targets = ids[:, 1:]
        logps = torch.log_softmax(logits, dim=-1).gather(-1, targets.unsqueeze(-1)).squeeze(-1)
        response_logps = logps[:, len(prompt_ids) - 1 :]
        mask = torch.tensor(tuple(token_mask), dtype=torch.bool, device=ids.device)
        return response_logps[:, mask].sum()

    def gradient_step(self, request: GradientStepRequest) -> GradientStepResult:
        self._ensure_hf()
        import torch

        optimizer = self.optimizers[request.role]
        optimizer.zero_grad(set_to_none=True)
        branches = [
            (
                tuple(
                    zip(
                        trace.prompts,
                        trace.response_segments,
                        trace.token_masks,
                        (
                            trace.token_ids
                            if trace.token_ids
                            else tuple(() for _ in trace.prompts)
                        ),
                    )
                ),
                float(advantage),
            )
            for trace, advantage in zip(request.traces, request.advantages)
        ]
        if not branches or any(not segments for segments, _ in branches):
            raise RuntimeError("gradient step received an empty policy trace")

        # A branch's policy statistic is the sum of all same-role response
        # decisions in its trace. Keeping the outer mean at branch level avoids
        # giving longer multi-step branches accidental extra batch weight.
        reference_logps = []
        for segments, _advantage in branches:
            with self.port.reference_disabled(), torch.no_grad():
                reference_logps.append(
                    torch.stack(
                        [
                            self._sequence_log_probability(
                                prompt, response, mask, token_ids
                            )
                            for prompt, response, mask, token_ids in segments
                        ]
                    ).sum().detach()
                )
        losses = []
        kls = []
        with self.port.activate(request.role, training=True):
            for (segments, advantage), reference in zip(branches, reference_logps):
                current = torch.stack(
                    [
                        self._sequence_log_probability(
                            prompt, response, mask, token_ids
                        )
                        for prompt, response, mask, token_ids in segments
                    ]
                ).sum()
                losses.append(-advantage * current)
                kls.append(current - reference)
            policy_loss = torch.stack(losses).mean()
            kl = torch.stack(kls).mean()
            loss = policy_loss + float(request.kl_penalty_coef) * kl
            loss.backward()
            manifest = self.port.parameter_manifests[request.role]
            named = dict(self.model.named_parameters())
            parameters = [named[name] for name in manifest.parameter_names]
            gradient_norm = torch.nn.utils.clip_grad_norm_(parameters, 1.0)
            optimizer.step()
        step = int(self.state.role_registry.state(request.role).adapter.revision) + 1
        return GradientStepResult(
            loss=float(loss.detach().item()),
            kl=float(kl.detach().item()),
            gradient_norm=float(gradient_norm),
            adapter_state={
                "backend": f"{self.config.backend}_named_lora",
                "role": request.role.value,
                "optimizer_step": step,
                "parameter_manifest_hash": manifest.manifest_hash,
            },
            optimizer_state={"backend": "adamw", "step": step},
        )

    def persist_roles(self, state: Any) -> Mapping[str, Mapping[str, Any]]:
        self.state = state
        self._ensure_hf()
        import torch

        roles = tuple(state.role_registry.roles)
        destinations = {
            role: self.layout.path(
                f"roles/{role.value}/adapter_epoch{state.epoch:03d}"
            )
            for role in roles
        }
        optimizer_paths = {
            role: self.layout.path(
                f"roles/{role.value}/optimizer_epoch{state.epoch:03d}.pt"
            )
            for role in roles
        }
        replayed_artifact = any(path.exists() for path in destinations.values())
        prior_optimizer_states = {
            role: self.optimizers[role].state_dict() for role in roles
        }
        for role in roles:
            destination = destinations[role]
            if destination.exists():
                role_state = state.role_registry.state(role)
                durable_manifest = json.loads(
                    (destination / "evolve_adapter_manifest.json").read_text(
                        encoding="utf-8"
                    )
                )
                # A crash may have made the immutable role artifact durable
                # before the barrier pointer. Adopt that exact output on replay
                # instead of allowing a second backward pass to diverge from it.
                self.port.load_adapter(
                    role,
                    state=role_state.adapter,
                    directory=destination,
                    expected_artifact_hash=durable_manifest["artifact_hash"],
                )
        if replayed_artifact:
            # Adapter loading may replace PEFT Parameter objects. Rebind every
            # optimizer only after all durable role adapters are adopted, then
            # restore the optimizer that was atomically saved with that exact
            # artifact. Never pair durable weights with a replayed gradient.
            self.optimizers = self._create_optimizers()
            for role in roles:
                destination = destinations[role]
                if not destination.exists():
                    self.optimizers[role].load_state_dict(
                        prior_optimizer_states[role]
                    )
                    continue
                embedded_optimizer = destination / "optimizer_state.pt"
                legacy_optimizer = optimizer_paths[role]
                optimizer_path = (
                    embedded_optimizer
                    if embedded_optimizer.is_file()
                    else legacy_optimizer
                )
                if not optimizer_path.is_file():
                    raise RuntimeError(
                        f"durable {role.value} adapter has no matching optimizer state; "
                        "refusing nondeterministic barrier recovery"
                    )
                try:
                    optimizer_state = torch.load(
                        optimizer_path, map_location="cpu", weights_only=False
                    )
                except TypeError:
                    optimizer_state = torch.load(optimizer_path, map_location="cpu")
                self.optimizers[role].load_state_dict(optimizer_state)

        artifacts = {}
        for role in roles:
            role_state = state.role_registry.state(role)
            destination = destinations[role]
            if not destination.exists():
                optimizer_buffer = io.BytesIO()
                torch.save(self.optimizers[role].state_dict(), optimizer_buffer)
                self.port.save_adapter(
                    role,
                    state=role_state.adapter,
                    destination=destination,
                    companion_files={
                        "optimizer_state.pt": optimizer_buffer.getvalue()
                    },
                )
            embedded_optimizer = destination / "optimizer_state.pt"
            legacy_optimizer = optimizer_paths[role]
            optimizer_path = (
                embedded_optimizer
                if embedded_optimizer.is_file()
                else legacy_optimizer
            )
            if not optimizer_path.is_file():
                raise RuntimeError(
                    f"durable {role.value} adapter has no matching optimizer state"
                )
            if optimizer_path == legacy_optimizer:
                try:
                    optimizer_state = torch.load(
                        optimizer_path, map_location="cpu", weights_only=False
                    )
                except TypeError:
                    optimizer_state = torch.load(optimizer_path, map_location="cpu")
                self.optimizers[role].load_state_dict(optimizer_state)
            manifest = json.loads(
                (destination / "evolve_adapter_manifest.json").read_text(
                    encoding="utf-8"
                )
            )
            artifacts[role.value] = {
                "adapter": destination.name,
                "optimizer": str(
                    optimizer_path.relative_to(destination.parent)
                ),
                "logical_adapter_hash": role_state.adapter.adapter_hash,
                "artifact_hash": manifest["artifact_hash"],
            }
        self._role_artifacts = artifacts
        return artifacts

    def persist_training_state(
        self,
        state: Any,
        checkpoint: Mapping[str, Any],
        targets: Sequence[Path],
    ) -> None:
        """Atomically publish the compatibility training checkpoint.

        The JSON checkpoint remains the schema-aware controller source of
        truth. This companion carries the concrete optimizer and RNG tensors
        expected by legacy tooling without weakening completed-barrier resume.
        """

        self._ensure_hf()
        import torch

        payload = {
            "format": "evolve_torch_training_state_v1",
            "checkpoint": dict(checkpoint),
            "optimizers": {
                role.value: self.optimizers[role].state_dict()
                for role in state.role_registry.roles
            },
            "python_rng_state": random.getstate(),
            "torch_rng_state": torch.get_rng_state(),
            "training_cuda_rng_state": (
                torch.cuda.get_rng_state(
                    self.config.training_device_index
                    if self.config.training_device_index is not None
                    else 0
                )
                if torch.cuda.is_available()
                else None
            ),
            "cuda_rng_states": (
                torch.cuda.get_rng_state_all() if torch.cuda.is_available() else []
            ),
        }
        buffer = io.BytesIO()
        torch.save(payload, buffer)
        serialized = buffer.getvalue()
        for target in targets:
            if target.parent.name == "checkpoints" and target.is_file():
                try:
                    durable = torch.load(
                        target, map_location="cpu", weights_only=False
                    )
                except TypeError:
                    durable = torch.load(target, map_location="cpu")
                if (
                    not isinstance(durable, Mapping)
                    or durable.get("format") != "evolve_torch_training_state_v1"
                    or durable.get("checkpoint") != dict(checkpoint)
                ):
                    raise RuntimeError(
                        f"immutable training-state checkpoint conflict: {target}"
                    )
                continue
            atomic_write_bytes(target, serialized)

    def _artifact_document(self, state: Any, role: Role) -> Optional[Mapping[str, Any]]:
        # During a live process, barrier persistence advances this mapping
        # before the controller's immutable EpochState is checkpointed. It is
        # therefore the newest authoritative artifact for the next phase.
        in_process = self._role_artifacts.get(role.value)
        if in_process:
            return in_process
        checkpoint_artifact = getattr(state, "role_artifacts", {}).get(role.value)
        if checkpoint_artifact:
            return checkpoint_artifact
        # Compatibility fallback for checkpoints written before adapter
        # artifacts became checkpoint-owned.
        latest = self.layout.path(f"roles/{role.value}/latest.json")
        if latest.is_file():
            return json.loads(latest.read_text(encoding="utf-8"))
        return None

    def _restore_adapters_if_present(self, state: Any) -> None:
        for role in state.role_registry.roles:
            document = self._artifact_document(state, role)
            if document is None:
                continue
            adapter_dir = self.layout.path(
                f"roles/{role.value}/{document['adapter']}"
            )
            manifest = json.loads(
                (adapter_dir / "evolve_adapter_manifest.json").read_text(encoding="utf-8")
            )
            self.port.load_adapter(
                role,
                state=state.role_registry.state(role).adapter,
                directory=adapter_dir,
                expected_artifact_hash=document.get(
                    "artifact_hash", manifest["artifact_hash"]
                ),
            )

    def _restore_optimizers_if_present(self, state: Any) -> None:
        import torch

        for role in state.role_registry.roles:
            document = self._artifact_document(state, role)
            if document is None:
                continue
            optimizer_path = self.layout.path(
                f"roles/{role.value}/{document['optimizer']}"
            )
            if optimizer_path.is_file():
                try:
                    optimizer_state = torch.load(
                        optimizer_path, map_location="cpu", weights_only=False
                    )
                except TypeError:
                    optimizer_state = torch.load(optimizer_path, map_location="cpu")
                self.optimizers[role].load_state_dict(optimizer_state)

    def _restore_rng_if_present(self, state: Any) -> None:
        """Restore the RNG state owned by the last completed barrier once.

        Model phase switches within a live epoch must not rewind randomness,
        so this hook is deliberately one-shot. Fresh runs have no completed
        training-state companion yet and simply mark the check complete.
        """

        if self._rng_restore_checked:
            return
        self._rng_restore_checked = True
        training_state_path = self.layout.path(
            f"checkpoints/checkpoint_epoch{state.epoch:03d}.pt"
        )
        if not training_state_path.is_file():
            return
        import torch

        try:
            payload = torch.load(
                training_state_path, map_location="cpu", weights_only=False
            )
        except TypeError:
            payload = torch.load(training_state_path, map_location="cpu")
        if not isinstance(payload, Mapping):
            raise RuntimeError("training-state companion must be a mapping")
        if payload.get("format") != "evolve_torch_training_state_v1":
            # JSON fake-worker companions are never consumed by a live model
            # runtime and cannot carry optimizer/RNG tensors.
            return
        checkpoint = payload.get("checkpoint")
        if not isinstance(checkpoint, Mapping):
            raise RuntimeError("training-state companion is missing its checkpoint")
        if checkpoint.get("run_id") != state.run_id or checkpoint.get("epoch") != state.epoch:
            raise RuntimeError(
                "training-state companion does not match the completed barrier"
            )
        python_state = payload.get("python_rng_state")
        torch_state = payload.get("torch_rng_state")
        if python_state is None or torch_state is None:
            raise RuntimeError("training-state companion is missing RNG state")
        random.setstate(python_state)
        torch.set_rng_state(torch_state)
        training_cuda_state = payload.get("training_cuda_rng_state")
        cuda_states = payload.get("cuda_rng_states", ())
        if torch.cuda.is_available() and training_cuda_state is not None:
            # Resource-only resume overrides may move the learning backbone or
            # change the number of visible GPUs. Restore the role-learning RNG
            # onto the new explicit training device; rollout seeds themselves
            # are topology-independent and do not use worker-rank CUDA state.
            training_device = (
                self.config.training_device_index
                if self.config.training_device_index is not None
                else 0
            )
            torch.cuda.set_rng_state(training_cuda_state, device=training_device)
        elif torch.cuda.is_available() and cuda_states:
            # Backward compatibility for companions written before the
            # training-device RNG field existed.
            if len(cuda_states) != torch.cuda.device_count():
                raise RuntimeError(
                    "legacy saved CUDA RNG topology differs from the resumed runtime"
                )
            torch.cuda.set_rng_state_all(list(cuda_states))

    def shutdown(self) -> None:
        try:
            self._branch_pool.shutdown(wait=True, cancel_futures=False)
        finally:
            try:
                self._unload_vllm()
            finally:
                self._unload_hf()


__all__ = ["LiveEvolveRuntime", "backbone_identity_for_config"]
