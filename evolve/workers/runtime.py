"""Live EVOLVE worker wiring over one model identity and three role LoRAs.

HF generation keeps the training backbone resident. vLLM generation instead
phase-switches at barriers: persisted HF adapters are released before one
tensor-parallel vLLM engine starts, and vLLM is fully shut down before HF role
learning resumes. The two full model implementations are never resident at the
same time.
"""

from __future__ import annotations

import gc
import json
import os
import tempfile
from pathlib import Path
from typing import Any, Mapping, Optional

from evolve.ids import content_hash
from evolve.learning.trainer import GradientStepRequest, GradientStepResult
from evolve.options.branch import BranchStepRequest, BranchStepResult
from evolve.roles.backend import BackboneIdentity, NamedAdapterBackendPort
from evolve.runio import atomic_write_json
from evolve.types import Role
from evolve.verifier.models import VerificationPolicy
from evolve.workers.verification import GenerationOutcome, build_proposal_and_verify
from evolve.workers.resources import ResourceLeaseManager
from problems.base import ParentContext, build_problem_prompt


ROLE_INSTRUCTIONS = {
    Role.SCOUT: "Search broadly and propose a structurally different approach.",
    Role.MECHANIST: "Develop the mechanism and make one focused, justified improvement.",
    Role.CHALLENGER: "Attack assumptions and make only a bounded minimal repair.",
}


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
        self._load_hf()

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
            return
        self.optimizers = None
        self.port = None
        self.model = None
        self.backend = None
        self._release_cuda()

    def _latest_adapter_paths(self) -> Mapping[Role, Path]:
        paths = {}
        for role in self.state.role_registry.roles:
            latest_path = self.layout.path(f"roles/{role.value}/latest.json")
            if not latest_path.is_file():
                raise RuntimeError(
                    f"missing persisted {role.value} adapter before vLLM startup"
                )
            document = json.loads(latest_path.read_text(encoding="utf-8"))
            artifact_root = latest_path.parent / document["adapter"]
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

        self.vllm = TensorParallelVLLM(
            config=self.config,
            adapter_paths=self._latest_adapter_paths(),
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

    def _parent_context(self, state_id: str) -> ParentContext:
        state = self.state.archive.artifacts.representative_state(state_id)
        evidence = self.state.archive.artifacts.evidence_packet(state.evidence_id)
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
            memory = json.dumps(memory_records, ensure_ascii=False, sort_keys=True)
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
                        + json.dumps(
                            diagnostics,
                            ensure_ascii=False,
                            sort_keys=True,
                        )
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
        return self.tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )

    def branch_step(self, request: BranchStepRequest) -> BranchStepResult:
        parent = self._parent_context(request.parent_state_id)
        prompt = self._render_prompt(request, parent)
        generation_seed = int(request.branch.seed + request.step_index)
        max_new_tokens = int(
            request.branch.generation_settings["max_new_tokens"]
        )
        temperature = float(request.branch.generation_settings["temperature"])
        top_p = float(request.branch.generation_settings["top_p"])
        if self.config.generation_backend == "vllm":
            self._ensure_vllm()
            text, token_ids, log_probabilities = self.vllm.generate(
                prompt=prompt,
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

            encoded = self.tokenizer(prompt, return_tensors="pt")
            device = next(self.model.parameters()).device
            encoded = {name: value.to(device) for name, value in encoded.items()}
            generator = torch.Generator(device=device)
            generator.manual_seed(generation_seed)
            with self.port.activate(request.arm.role, training=False), torch.no_grad():
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
            token_ids = tuple(int(value) for value in generated_ids.tolist())
            text = self.tokenizer.decode(generated_ids, skip_special_tokens=True)
            log_probabilities = tuple(
                float(torch.log_softmax(scores[0], dim=-1)[token_id].item())
                for scores, token_id in zip(output.scores, generated_ids)
            )

        evaluated: dict[str, Any] = {}
        resources = self.problem.resource_requirements()

        # A one-GPU kernel run is exclusive in time rather than by a dedicated
        # physical device. Tear down the proposing model before the spawned
        # benchmark process starts; the next branch lazily restores its phase.
        if resources.exclusive_gpu:
            if self.config.kernel_gpu_id in self.config.gpu_ids:
                self._unload_vllm()
            if self.config.kernel_gpu_id == self.config.training_gpu_id:
                self._unload_hf()

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
                generation=GenerationOutcome(
                    prompt=prompt,
                    text=text,
                    token_ids=token_ids,
                    log_probabilities=log_probabilities,
                ),
                extract_answer=extract_answer,
                adapter=self.adapter,
                verification_policy=self.verification_policy,
                harness_id=request.branch.harness_id,
                policy_snapshot_id=request.branch.role_snapshot_id,
                run_dir=self.layout.run_dir,
            )

        if resources.exclusive_gpu:
            with self.leases.lease(
                f"gpu:{self.config.kernel_gpu_id}",
                holder=request.branch.branch_id,
                timeout=float(resources.timeout_s) * 3.0,
            ):
                return verify_call()
        return verify_call()

    def _sequence_log_probability(self, prompt: str, response: str) -> Any:
        import torch

        prompt_ids = self.tokenizer(prompt, add_special_tokens=False).input_ids
        response_ids = self.tokenizer(response, add_special_tokens=False).input_ids
        if not response_ids:
            return torch.zeros((), device=next(self.model.parameters()).device)
        ids = torch.tensor([prompt_ids + response_ids], device=next(self.model.parameters()).device)
        logits = self.model(input_ids=ids).logits[:, :-1, :]
        targets = ids[:, 1:]
        logps = torch.log_softmax(logits, dim=-1).gather(-1, targets.unsqueeze(-1)).squeeze(-1)
        start = max(0, len(prompt_ids) - 1)
        return logps[:, start:].sum()

    def gradient_step(self, request: GradientStepRequest) -> GradientStepResult:
        self._ensure_hf()
        import torch

        optimizer = self.optimizers[request.role]
        optimizer.zero_grad(set_to_none=True)
        branches = [
            (
                tuple(zip(trace.prompts, trace.response_segments)),
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
                            self._sequence_log_probability(prompt, response)
                            for prompt, response in segments
                        ]
                    ).sum().detach()
                )
        losses = []
        kls = []
        with self.port.activate(request.role, training=True):
            for (segments, advantage), reference in zip(branches, reference_logps):
                current = torch.stack(
                    [
                        self._sequence_log_probability(prompt, response)
                        for prompt, response in segments
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

    def persist_roles(self, state: Any) -> None:
        self.state = state
        self._ensure_hf()
        import torch

        for role in state.role_registry.roles:
            role_state = state.role_registry.state(role)
            destination = self.layout.path(
                f"roles/{role.value}/adapter_epoch{state.epoch:03d}"
            )
            if not destination.exists():
                self.port.save_adapter(
                    role, state=role_state.adapter, destination=destination
                )
            optimizer_path = self.layout.path(
                f"roles/{role.value}/optimizer_epoch{state.epoch:03d}.pt"
            )
            if not optimizer_path.exists():
                optimizer_path.parent.mkdir(parents=True, exist_ok=True)
                handle, temporary_name = tempfile.mkstemp(
                    prefix=f".{optimizer_path.name}.", dir=str(optimizer_path.parent)
                )
                os.close(handle)
                try:
                    torch.save(self.optimizers[role].state_dict(), temporary_name)
                    os.replace(temporary_name, optimizer_path)
                finally:
                    if os.path.exists(temporary_name):
                        os.unlink(temporary_name)
            atomic_write_json(
                self.layout.path(f"roles/{role.value}/latest.json"),
                {
                    "epoch": state.epoch,
                    "adapter": destination.name,
                    "optimizer": optimizer_path.name,
                    "logical_adapter_hash": role_state.adapter.adapter_hash,
                },
            )

    def _restore_adapters_if_present(self, state: Any) -> None:
        for role in state.role_registry.roles:
            latest = self.layout.path(f"roles/{role.value}/latest.json")
            if not latest.is_file():
                continue
            document = json.loads(latest.read_text(encoding="utf-8"))
            adapter_dir = latest.parent / document["adapter"]
            manifest = json.loads(
                (adapter_dir / "evolve_adapter_manifest.json").read_text(encoding="utf-8")
            )
            self.port.load_adapter(
                role,
                state=state.role_registry.state(role).adapter,
                directory=adapter_dir,
                expected_artifact_hash=manifest["artifact_hash"],
            )

    def _restore_optimizers_if_present(self, state: Any) -> None:
        import torch

        for role in state.role_registry.roles:
            latest = self.layout.path(f"roles/{role.value}/latest.json")
            if not latest.is_file():
                continue
            document = json.loads(latest.read_text(encoding="utf-8"))
            optimizer_path = latest.parent / document["optimizer"]
            if optimizer_path.is_file():
                self.optimizers[role].load_state_dict(
                    torch.load(optimizer_path, map_location="cpu")
                )

    def shutdown(self) -> None:
        self._unload_vllm()
        self._unload_hf()


__all__ = ["LiveEvolveRuntime", "backbone_identity_for_config"]
