"""CPU-only contract tests for the one-backbone named-adapter port."""

from __future__ import annotations

import contextlib
import json
import shutil
from pathlib import Path

import pytest

from evolve.ids import content_hash, content_id
from evolve.roles.adapters import RoleAdapterState
from evolve.roles.backend import (
    ADAPTER_MANIFEST_NAME,
    ROLE_ADAPTER_NAMES,
    AdapterArtifactError,
    BackboneIdentity,
    NamedAdapterBackendPort,
    RoleBackendCapabilityError,
    RoleBackendError,
    inspect_adapter_artifact,
)
from evolve.types import Role


class _FakeParameter:
    def __init__(self, value: str, *, requires_grad: bool = False):
        self.value = value
        self.requires_grad = requires_grad


class _FakePeftModel:
    """Small PEFT-shaped model with no torch dependency."""

    def __init__(self, *, unexpected_adapter: bool = False, alias_roles: bool = False):
        self.calls = []
        self.training = True
        self.active_adapter = "default"
        self.adapters_disabled = False
        self.alias_roles = alias_roles
        self.peft_config = {"default": {"rank": 2}}
        if unexpected_adapter:
            self.peft_config["foreign"] = {"rank": 2}
        self.parameters = {
            "base_model.layer.weight": _FakeParameter("base", requires_grad=False),
            "base_model.layer.lora_A.default.weight": _FakeParameter(
                "default-a", requires_grad=True
            ),
            "base_model.layer.lora_B.default.weight": _FakeParameter(
                "default-b", requires_grad=True
            ),
        }

    def named_parameters(self):
        return list(self.parameters.items())

    def add_adapter(self, adapter_name, adapter_config):
        self.calls.append(("add_adapter", adapter_name))
        if adapter_name in self.peft_config:
            raise ValueError("duplicate adapter")
        self.peft_config[adapter_name] = dict(adapter_config)
        first = _FakeParameter(f"{adapter_name}-a")
        if self.alias_roles and adapter_name == ROLE_ADAPTER_NAMES[Role.MECHANIST.value]:
            first = self.parameters[
                "base_model.layer.lora_A.evolve_scout.weight"
            ]
        self.parameters[
            f"base_model.layer.lora_A.{adapter_name}.weight"
        ] = first
        self.parameters[
            f"base_model.layer.lora_B.{adapter_name}.weight"
        ] = _FakeParameter(f"{adapter_name}-b")

    def delete_adapter(self, adapter_name):
        self.calls.append(("delete_adapter", adapter_name))
        self.peft_config.pop(adapter_name)
        self.parameters = {
            name: parameter
            for name, parameter in self.parameters.items()
            if adapter_name not in name.split(".")
        }
        if self.active_adapter == adapter_name:
            self.active_adapter = None

    def set_adapter(self, adapter_name):
        self.calls.append(("set_adapter", adapter_name))
        if adapter_name not in self.peft_config:
            raise ValueError(f"unknown adapter {adapter_name}")
        self.active_adapter = adapter_name

    def train(self):
        self.calls.append(("train",))
        self.training = True

    def eval(self):
        self.calls.append(("eval",))
        self.training = False

    @contextlib.contextmanager
    def disable_adapter(self):
        self.calls.append(("disable_enter",))
        assert not self.adapters_disabled
        self.adapters_disabled = True
        try:
            yield
        finally:
            self.adapters_disabled = False
            self.calls.append(("disable_exit",))

    def save_pretrained(self, directory, *, selected_adapters, safe_serialization):
        self.calls.append(("save_pretrained", tuple(selected_adapters)))
        assert safe_serialization is True
        assert len(selected_adapters) == 1
        adapter_name = selected_adapters[0]
        root = Path(directory) / adapter_name
        root.mkdir(parents=True)
        (root / "adapter_config.json").write_text(
            json.dumps(self.peft_config[adapter_name]), encoding="utf-8"
        )
        selected = {
            name: parameter.value
            for name, parameter in self.parameters.items()
            if adapter_name in name.split(".")
        }
        (root / "adapter_model.safetensors").write_text(
            json.dumps(selected, sort_keys=True), encoding="utf-8"
        )

    def load_adapter(self, directory, *, adapter_name, is_trainable):
        self.calls.append(("load_adapter", adapter_name, is_trainable))
        assert is_trainable is False
        root = Path(directory)
        config = json.loads((root / "adapter_config.json").read_text(encoding="utf-8"))
        selected = json.loads(
            (root / "adapter_model.safetensors").read_text(encoding="utf-8")
        )
        self.peft_config[adapter_name] = config
        for name, value in selected.items():
            self.parameters[name] = _FakeParameter(value, requires_grad=False)


class _FakeHFBackend:
    name = "hf"

    def __init__(self, model=None):
        self.model = model or _FakePeftModel()

    def set_inference_mode(self):
        self.model.eval()

    def set_training_mode(self):
        self.model.train()

    def disable_adapter(self):
        return self.model.disable_adapter()


def _backbone(label="one"):
    return BackboneIdentity.create(
        model_name="org/frozen-model",
        revision=f"commit-{label}",
        weights_hash=content_hash({"weights": label}),
        config_hash=content_hash({"config": label}),
    )


def _port(*, model=None, backbone=None):
    backend = _FakeHFBackend(model=model)
    port = NamedAdapterBackendPort(
        backend,
        backbone=backbone or _backbone(),
        adapter_config={"rank": 2, "alpha": 4},
    )
    return backend, port


def _run_id():
    return content_id("run", {"fixture": "role-backend"})


def _state(role, marker="initial"):
    return RoleAdapterState.create(
        run_id=_run_id(),
        role=role,
        state={"tensor_digest": content_hash({"role": role.value, "marker": marker})},
    )


def _owned_flags(port, model):
    manifests = port.parameter_manifests
    by_name = {
        name: role
        for role, manifest in manifests.items()
        for name in manifest.parameter_names
    }
    return {
        name: (by_name.get(name), parameter.requires_grad)
        for name, parameter in model.named_parameters()
    }


@pytest.mark.parametrize("backend_name", ["auto", "unsloth", "vllm"])
def test_backend_must_be_explicitly_resolved_hf_before_any_adapter_mutation(backend_name):
    backend = _FakeHFBackend()
    backend.name = backend_name

    with pytest.raises(RoleBackendCapabilityError, match="explicitly resolved HF"):
        NamedAdapterBackendPort(
            backend, backbone=_backbone(), adapter_config={"rank": 2}
        )

    assert backend.model.calls == []
    assert set(backend.model.peft_config) == {"default"}


def test_missing_capability_and_foreign_adapter_fail_before_installation():
    missing = _FakeHFBackend()
    missing.model.load_adapter = None
    with pytest.raises(RoleBackendCapabilityError, match="load_adapter"):
        NamedAdapterBackendPort(
            missing, backbone=_backbone(), adapter_config={"rank": 2}
        )
    assert missing.model.calls == []

    foreign = _FakeHFBackend(_FakePeftModel(unexpected_adapter=True))
    with pytest.raises(RoleBackendCapabilityError, match="foreign"):
        NamedAdapterBackendPort(
            foreign, backbone=_backbone(), adapter_config={"rank": 2}
        )
    assert foreign.model.calls == []


def test_installs_exact_three_disjoint_named_adapters_on_one_frozen_backbone():
    backend, port = _port()
    model = backend.model

    assert set(model.peft_config) == set(ROLE_ADAPTER_NAMES.values())
    assert port.active_role is Role.SCOUT
    assert model.active_adapter == ROLE_ADAPTER_NAMES[Role.SCOUT.value]
    assert model.training is False
    manifests = port.parameter_manifests
    assert set(manifests) == {Role.SCOUT, Role.MECHANIST, Role.CHALLENGER}
    owned = [set(manifest.parameter_names) for manifest in manifests.values()]
    assert all(owned)
    assert not (owned[0] & owned[1] or owned[0] & owned[2] or owned[1] & owned[2])
    assert all(not parameter.requires_grad for _, parameter in model.named_parameters())
    assert port.backbone_id == _backbone().backbone_id

    binding = port.assert_dispatch_ready(
        Role.MECHANIST,
        expected_backbone_id=port.backbone_id,
        expected_parameter_manifest_hash=manifests[Role.MECHANIST].manifest_hash,
    )
    assert binding.role is Role.MECHANIST
    assert binding.adapter_name == ROLE_ADAPTER_NAMES[Role.MECHANIST.value]

    with pytest.raises(RoleBackendError, match="backbone identity"):
        port.assert_dispatch_ready(Role.SCOUT, expected_backbone_id=_backbone("two").backbone_id)


def test_role_activation_is_disjoint_nested_and_exception_safe():
    backend, port = _port()
    model = backend.model

    with pytest.raises(RuntimeError, match="stop"):
        with port.activate(Role.MECHANIST, training=True) as binding:
            assert binding.role is Role.MECHANIST
            assert port.active_role is Role.MECHANIST
            assert model.training is True
            flags = _owned_flags(port, model)
            assert all(
                flag is (owner is Role.MECHANIST)
                for owner, flag in flags.values()
            )
            port.assert_isolation(active_role=Role.MECHANIST, training=True)

            with port.activate(Role.CHALLENGER, training=False):
                assert port.active_role is Role.CHALLENGER
                assert model.training is False
                assert all(not parameter.requires_grad for _, parameter in model.named_parameters())

            assert port.active_role is Role.MECHANIST
            assert model.training is True
            port.assert_isolation(active_role=Role.MECHANIST, training=True)
            raise RuntimeError("stop")

    assert port.active_role is Role.SCOUT
    assert model.active_adapter == ROLE_ADAPTER_NAMES[Role.SCOUT.value]
    assert model.training is False
    assert all(not parameter.requires_grad for _, parameter in model.named_parameters())


def test_reference_disable_is_inference_only_and_restores_training_role():
    backend, port = _port()
    model = backend.model

    with port.activate(Role.CHALLENGER, training=True):
        with port.reference_disabled():
            assert model.adapters_disabled is True
            assert model.training is False
            assert all(not parameter.requires_grad for _, parameter in model.named_parameters())
        assert model.adapters_disabled is False
        assert model.training is True
        assert port.active_role is Role.CHALLENGER
        port.assert_isolation(active_role=Role.CHALLENGER, training=True)

    assert port.active_role is Role.SCOUT
    assert model.training is False


def test_dispatch_rejects_base_or_inactive_unfreezing_and_runtime_drift():
    backend, port = _port()
    model = backend.model

    model.parameters["base_model.layer.weight"].requires_grad = True
    with pytest.raises(RoleBackendError, match="dispatch attempted"):
        port.assert_dispatch_ready(Role.SCOUT)
    model.parameters["base_model.layer.weight"].requires_grad = False

    inactive = port.parameter_manifests[Role.MECHANIST].parameter_names[0]
    model.parameters[inactive].requires_grad = True
    with pytest.raises(RoleBackendError, match="dispatch attempted"):
        port.assert_dispatch_ready(Role.SCOUT)
    model.parameters[inactive].requires_grad = False

    model.peft_config["rogue"] = {}
    with pytest.raises(RoleBackendError, match="registry changed"):
        port.assert_dispatch_ready(Role.SCOUT)


def test_parameter_object_alias_across_roles_is_rejected():
    with pytest.raises(RoleBackendError, match="parameter object aliases"):
        _port(model=_FakePeftModel(alias_roles=True))


def test_adapter_artifact_is_atomic_immutable_relocatable_and_loadable(tmp_path):
    backend, port = _port()
    scout = _state(Role.SCOUT)
    destination = tmp_path / "roles" / "scout" / "adapter_epoch000"

    artifact = port.save_adapter(
        Role.SCOUT, state=scout, destination=destination
    )
    assert (destination / ADAPTER_MANIFEST_NAME).is_file()
    assert not list(destination.parent.glob(".adapter_epoch000.tmp-*"))
    assert inspect_adapter_artifact(
        destination, expected_artifact_hash=artifact.artifact_hash
    ) == artifact
    with pytest.raises(AdapterArtifactError, match="already exists"):
        port.save_adapter(Role.SCOUT, state=scout, destination=destination)

    relocated = tmp_path / "relocated" / "scout"
    shutil.copytree(destination, relocated)
    assert inspect_adapter_artifact(
        relocated, expected_artifact_hash=artifact.artifact_hash
    ) == artifact

    scout_names = port.parameter_manifests[Role.SCOUT].parameter_names
    for name in scout_names:
        backend.model.parameters[name].value = "mutated"
    loaded = port.load_adapter(
        Role.SCOUT,
        state=scout,
        directory=relocated,
        expected_artifact_hash=artifact.artifact_hash,
    )
    assert loaded == artifact
    assert all(backend.model.parameters[name].value != "mutated" for name in scout_names)
    assert set(backend.model.peft_config) == set(ROLE_ADAPTER_NAMES.values())
    assert all(not parameter.requires_grad for _, parameter in backend.model.named_parameters())


def test_adapter_load_validates_role_backbone_and_bytes_before_model_mutation(tmp_path):
    backend, port = _port()
    scout = _state(Role.SCOUT)
    destination = tmp_path / "scout"
    artifact = port.save_adapter(Role.SCOUT, state=scout, destination=destination)
    before_calls = list(backend.model.calls)

    with pytest.raises(RoleBackendError, match="different role"):
        port.load_adapter(
            Role.MECHANIST,
            state=scout,
            directory=destination,
            expected_artifact_hash=artifact.artifact_hash,
        )
    assert backend.model.calls == before_calls

    other_backend, other_port = _port(backbone=_backbone("other"))
    with pytest.raises(AdapterArtifactError, match="another backbone"):
        other_port.load_adapter(
            Role.SCOUT,
            state=scout,
            directory=destination,
            expected_artifact_hash=artifact.artifact_hash,
        )
    assert not any(call[0] == "load_adapter" for call in other_backend.model.calls)

    payload = next(destination.rglob("adapter_model.safetensors"))
    payload.write_text("tampered", encoding="utf-8")
    before_calls = list(backend.model.calls)
    with pytest.raises(AdapterArtifactError, match="payload files changed"):
        port.load_adapter(
            Role.SCOUT,
            state=scout,
            directory=destination,
            expected_artifact_hash=artifact.artifact_hash,
        )
    assert backend.model.calls == before_calls

