"""Live EVOLVE worker wiring over one frozen HF backbone and three LoRAs.

The controller owns this object. It keeps exactly one model resident, switches
named role adapters explicitly, captures token log probabilities, verifies the
saved answer payload, and applies one role-isolated optimizer step at barriers.
"""

from __future__ import annotations

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
        if config.backend != "hf":
            raise RuntimeError(
                "production EVOLVE currently requires backend: hf so exact named "
                "adapter isolation can be enforced"
            )
        if config.num_gpus < 1:
            raise RuntimeError("a live EVOLVE run requires at least one generation GPU")

        requested_mask = ",".join(str(item) for item in config.gpu_ids)
        existing_mask = os.environ.get("CUDA_VISIBLE_DEVICES")
        if existing_mask is not None and existing_mask != requested_mask:
            raise RuntimeError(
                "CUDA_VISIBLE_DEVICES disagrees with authoritative config gpu_ids: "
                f"{existing_mask!r} != {requested_mask!r}"
            )
        os.environ["CUDA_VISIBLE_DEVICES"] = requested_mask

        from evolve.roles.model import HFBackbone

        self.config = config
        self.adapter = adapter
        self.problem = adapter.problem
        self.layout = layout
        self.state = state
        self.verification_policy = VerificationPolicy.create(
            version="evolve_engine_v1", production=not config.method_incomplete
        )
        self.leases = ResourceLeaseManager()
        self.backend = HFBackbone(config)
        self.model, self.tokenizer = self.backend.load()
        default_config = self.model.peft_config["default"]
        self.port = NamedAdapterBackendPort(
            self.backend,
            backbone=backbone_identity_for_config(config),
            adapter_config=default_config,
        )
        self.optimizers = self._create_optimizers()
        self._restore_if_present(state)

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
        for index in range(len(messages) - 1, -1, -1):
            if messages[index].get("role") == "user":
                messages[index]["content"] = str(messages[index].get("content", "")) + instruction
                break
        return self.tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )

    def branch_step(self, request: BranchStepRequest) -> BranchStepResult:
        import torch

        parent = self._parent_context(request.parent_state_id)
        prompt = self._render_prompt(request, parent)
        encoded = self.tokenizer(prompt, return_tensors="pt")
        device = next(self.model.parameters()).device
        encoded = {name: value.to(device) for name, value in encoded.items()}
        generator = torch.Generator(device=device)
        generator.manual_seed(int(request.branch.seed + request.step_index))
        with self.port.activate(request.arm.role, training=False), torch.no_grad():
            output = self.model.generate(
                **encoded,
                max_new_tokens=int(request.branch.generation_settings["max_new_tokens"]),
                temperature=float(request.branch.generation_settings["temperature"]),
                top_p=float(request.branch.generation_settings["top_p"]),
                do_sample=float(request.branch.generation_settings["temperature"]) > 0.0,
                return_dict_in_generate=True,
                output_scores=True,
                generator=generator,
                pad_token_id=self.tokenizer.pad_token_id,
            )
        prompt_length = encoded["input_ids"].shape[1]
        generated_ids = output.sequences[0, prompt_length:]
        text = self.tokenizer.decode(generated_ids, skip_special_tokens=True)
        log_probabilities = tuple(
            float(torch.log_softmax(scores[0], dim=-1)[token_id].item())
            for scores, token_id in zip(output.scores, generated_ids)
        )

        evaluated: dict[str, Any] = {}
        resources = self.problem.resource_requirements()

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
                    token_ids=tuple(int(value) for value in generated_ids.tolist()),
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
        import torch

        optimizer = self.optimizers[request.role]
        optimizer.zero_grad(set_to_none=True)
        segments = [
            (prompt, response, float(advantage))
            for trace, advantage in zip(request.traces, request.advantages)
            for prompt, response in zip(trace.prompts, trace.response_segments)
        ]
        references = []
        for prompt, response, _advantage in segments:
            with self.port.reference_disabled(), torch.no_grad():
                references.append(
                    self._sequence_log_probability(prompt, response).detach()
                )
        losses = []
        kls = []
        if not segments:
            raise RuntimeError("gradient step received no policy response segments")
        with self.port.activate(request.role, training=True):
            for (prompt, response, advantage), reference in zip(segments, references):
                current = self._sequence_log_probability(prompt, response)
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
                "backend": "hf_named_lora",
                "role": request.role.value,
                "optimizer_step": step,
                "parameter_manifest_hash": manifest.manifest_hash,
            },
            optimizer_state={"backend": "adamw", "step": step},
        )

    def persist_roles(self, state: Any) -> None:
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

    def _restore_if_present(self, state: Any) -> None:
        import torch

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
            optimizer_path = latest.parent / document["optimizer"]
            if optimizer_path.is_file():
                self.optimizers[role].load_state_dict(
                    torch.load(optimizer_path, map_location="cpu")
                )

    def shutdown(self) -> None:
        return None


__all__ = ["LiveEvolveRuntime", "backbone_identity_for_config"]
