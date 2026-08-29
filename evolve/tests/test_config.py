import copy
import json
from pathlib import Path

import pytest

from evolve.ids import content_hash, content_id
from evolve.runio.manifest import (
    write_initial_run_metadata,
    write_resume_run_metadata,
)

from evolve.config import (
    EvolveConfig,
    EvolveConfigError,
    UnsupportedEvolveConfigError,
    canonical_config_hash,
    load_evolve_config,
    parse_evolve_args,
    validate_resolved_config_document,
)


def _write_yaml(path: Path, body: str) -> Path:
    path.write_text(body, encoding="utf-8")
    return path


def _minimal_yaml(tmp_path: Path, extra: str = "") -> Path:
    return _write_yaml(
        tmp_path / "problem.yaml",
        "problem: erdos\n"
        "model_name: yaml/model\n"
        "gpu_ids: '0,2'\n"
        "num_gpus: 99\n"
        "max_seq_length: 4096\n"
        "max_new_tokens: 512\n"
        "num_steps: 12\n"
        "group_size: 6\n"
        "budget_s: 2\n"
        + extra,
    )


def _fresh(tmp_path: Path, extra: str = "", cli=()):
    path = _minimal_yaml(tmp_path, extra)
    return load_evolve_config(["--config", str(path), *cli], cwd=tmp_path)


def _write_resume(run_dir: Path, resolved: dict) -> None:
    run_dir.mkdir()
    host = {"hostname": "config-fixture"}
    packages = {"python": "test"}
    gpus = []
    (run_dir / "config.requested.yaml").write_text(
        "engine: evolve\n", encoding="utf-8"
    )
    (run_dir / "config.resolved.json").write_text(
        json.dumps(resolved), encoding="utf-8"
    )
    (run_dir / "config.json").write_text(
        json.dumps(resolved), encoding="utf-8"
    )
    (run_dir / "manifest.json").write_text(
        json.dumps(
            {
                "engine": "evolve",
                "schema_version": 1,
                "config_hash": resolved.get("config_hash"),
                "run_id": content_id("run", {"fixture": resolved.get("config_hash")}),
                "run_name": run_dir.name,
                "created_at": "2026-08-28T00:00:00Z",
                "config_schema_version": resolved["schema_version"],
                "git": {"commit": "fixture"},
                "model": {"name": resolved.get("model_name", "fixture")},
                "packages": packages,
                "host": host,
                "gpus": gpus,
                "worker_topology": {"generation_workers": 0},
                "seeds": {"run": resolved.get("seed", 0)},
                "versions": {"config": 1},
            }
        ),
        encoding="utf-8",
    )
    (run_dir / "command.json").write_text(
        json.dumps({"schema_version": 1, "argv": ["train_evolve.py"]}),
        encoding="utf-8",
    )
    (run_dir / "environment.json").write_text(
        json.dumps(
            {
                "schema_version": 1,
                "variables": {},
                "host": host,
                "gpus": gpus,
                "packages": packages,
            }
        ),
        encoding="utf-8",
    )


def _metadata_arguments():
    return {
        "command": ["python", "train_evolve.py", "--problem", "erdos"],
        "environment": {},
        "git_state": {"commit": "fixture"},
        "model": {"name": "fixture"},
        "package_versions": {"python": "test"},
        "host": {"hostname": "fixture"},
        "gpus": [],
        "worker_topology": {"generation_workers": 0},
        "seeds": {"run": 42},
        "versions": {"config": 1},
    }


def test_parse_api_accepts_operational_flags():
    args = parse_evolve_args(
        ["--engine", "evolve", "--validate-config", "--dry-plan"]
    )
    assert args.engine == "evolve"
    assert args.validate_config is True
    assert args.dry_plan is True


def test_common_cli_aliases_and_symmetric_boolean_overrides(tmp_path):
    cfg, resolved, metadata = _fresh(
        tmp_path,
        "thinking: true\n"
        "deterministic: true\n",
        cli=(
            "--no-thinking",
            "--no-deterministic",
            "--lr",
            "0.002",
            "--gpu-type",
            "H100",
            "--num-circles",
            "31",
            "--group-size",
            "4",
        ),
    )

    assert cfg.thinking is False
    assert cfg.deterministic is False
    assert cfg.learning_rate == pytest.approx(0.002)
    assert cfg.evolve.learning.group_k == 4
    assert cfg.problem_config["gpu_type"] == "H100"
    assert cfg.problem_config["num_circles"] == 31
    assert resolved["group_size"] == 4
    assert metadata["cli_overrides"]["thinking"] is False

    enabled, _, _ = _fresh(
        tmp_path,
        "thinking: false\n"
        "deterministic: false\n",
        cli=(
            "--thinking",
            "--deterministic",
        ),
    )
    assert enabled.thinking is True
    assert enabled.deterministic is True


def test_group_size_alias_rejects_disagreement(tmp_path):
    with pytest.raises(EvolveConfigError, match="group-k.*group-size.*disagree"):
        _fresh(tmp_path, cli=("--group-k", "4", "--group-size", "5"))


def test_defaults_yaml_aliases_and_cli_precedence(tmp_path):
    cfg, resolved, metadata = _fresh(
        tmp_path,
        "evolve:\n"
        "  budget:\n"
        "    epochs: 13\n",
        cli=("--model-name", "cli/model", "--num-steps", "20"),
    )

    assert isinstance(cfg, EvolveConfig)
    assert cfg.model_name == "cli/model"
    assert cfg.evolve.budget.epochs == 20
    assert cfg.evolve.learning.group_k == 6
    assert cfg.evolve.archive.elites_per_cell == 3
    assert cfg.problem_config["budget_s"] == 2
    assert cfg.gpu_ids == (0, 2)
    assert cfg.num_gpus == 2
    assert resolved["gpu_ids"] == [0, 2]
    assert resolved["num_gpus"] == 2
    assert resolved["num_steps"] == 20
    assert resolved["group_size"] == 6
    assert metadata["mode"] == "fresh"
    assert metadata["config_hash"] == resolved["config_hash"]
    assert canonical_config_hash(cfg) == resolved["config_hash"]
    assert "authoritative" in metadata["derivations"][0]


def test_num_steps_is_total_epoch_alias_and_disagreement_is_rejected(tmp_path):
    cfg, _, _ = _fresh(tmp_path, cli=("--num-steps", "150"))
    assert cfg.num_steps == 150
    with pytest.raises(EvolveConfigError, match="disagree"):
        _fresh(tmp_path, cli=("--num-steps", "10", "--epochs", "11"))


def test_canonical_hash_is_order_stable_and_value_sensitive():
    left = {"b": [2, 3], "a": {"x": 1}, "config_hash": "ignored"}
    right = {"a": {"x": 1}, "b": [2, 3]}
    assert canonical_config_hash(left) == canonical_config_hash(right)
    right["b"][1] = 4
    assert canonical_config_hash(left) != canonical_config_hash(right)


@pytest.mark.parametrize(
    "extra, message",
    [
        ("gpu_ids: '0,0'\n", "duplicates"),
        ("gpu_ids: '0,nope'\n", "gpu_ids"),
        ("gpu_ids: [-1]\n", "non-negative|>= 0"),
        ("kernel_gpu_id: 2\n", "exclusive"),
    ],
)
def test_bad_gpu_lists_and_exclusive_conflicts(tmp_path, extra, message):
    with pytest.raises(EvolveConfigError, match=message):
        _fresh(tmp_path, extra)


def test_gpu_ids_are_authoritative_over_num_gpus(tmp_path):
    cfg, resolved, metadata = _fresh(tmp_path)
    assert cfg.gpu_ids == (0, 2)
    assert cfg.num_gpus == resolved["num_gpus"] == 2
    assert metadata["derivations"]

    with pytest.raises(EvolveConfigError, match="disagrees"):
        _fresh(
            tmp_path,
            cli=("--gpu-ids", "0,2", "--num-gpus", "3"),
        )


def test_num_gpus_cli_alone_expands_authoritative_ids(tmp_path):
    cfg, resolved, _ = _fresh(tmp_path, cli=("--num-gpus", "3"))
    assert cfg.gpu_ids == (0, 1, 2)
    assert resolved["gpu_ids"] == [0, 1, 2]


def test_unknown_nested_evolve_keys_are_rejected(tmp_path):
    with pytest.raises(EvolveConfigError, match="unknown key"):
        _fresh(
            tmp_path,
            "evolve:\n"
            "  budget:\n"
            "    epochs: 5\n"
            "    surprise: 1\n",
        )
    with pytest.raises(EvolveConfigError, match="unknown section"):
        _fresh(tmp_path, "evolve:\n  puct: {}\n")


def test_unknown_top_level_typo_is_rejected_but_problem_key_is_preserved(tmp_path):
    with pytest.raises(EvolveConfigError, match="unknown top-level"):
        _fresh(tmp_path, "modle_name: typo\n")
    cfg, resolved, _ = _fresh(tmp_path, "num_circles: 26\n")
    assert cfg.problem_config["num_circles"] == 26
    assert resolved["num_circles"] == 26


def test_scientific_resource_limits_are_typed_before_runtime(tmp_path):
    cfg, resolved, _ = _fresh(
        tmp_path,
        "eval_cpus: 3\n"
        "eval_memory_mb: 2048\n"
        "eval_seed: 7\n"
        "scientific_max_points: 128\n"
        "scientific_max_coefficients: 512\n",
    )
    assert cfg.problem_config["eval_cpus"] == 3
    assert cfg.problem_config["eval_memory_mb"] == 2048
    assert cfg.problem_config["eval_seed"] == 7
    assert cfg.problem_config["scientific_max_points"] == 128
    assert cfg.problem_config["scientific_max_coefficients"] == 512
    assert resolved["scientific_max_points"] == 128


@pytest.mark.parametrize(
    ("key", "value"),
    [
        ("eval_cpus", "0"),
        ("eval_memory_mb", "false"),
        ("eval_seed", "-1"),
        ("scientific_max_points", "0"),
        ("scientific_max_coefficients", "1.5"),
        ("budget_s", "0"),
    ],
)
def test_invalid_scientific_resource_limits_fail_config_loading(
    tmp_path, key, value
):
    with pytest.raises(EvolveConfigError, match=key):
        _fresh(tmp_path, f"{key}: {value}\n")


@pytest.mark.parametrize(
    "typo",
    [
        "memory_lookup_max_selec",
        "feedback_validity_targt",
        "reranker_poll_intervl_s",
        "_max_seq_length_unknown_marker",
    ],
)
def test_prefixed_compatibility_typos_are_rejected(tmp_path, typo):
    with pytest.raises(EvolveConfigError, match="unknown top-level"):
        _fresh(tmp_path, f"{typo}: 1\n")


@pytest.mark.parametrize(
    "config_name",
    [
        "ac1.yaml",
        "ac2.yaml",
        "circle_packing.yaml",
        "denoising.yaml",
        "erdos.yaml",
        "evolve_toy.yaml",
        "gpu_mode_mla.yaml",
        "gpu_mode_trimul.yaml",
    ],
)
def test_current_problem_yamls_use_only_registered_compatibility_keys(config_name):
    repository = Path(__file__).resolve().parents[2]
    config_path = repository / "configs" / config_name
    config, _, _ = load_evolve_config(
        ["--config", str(config_path)],
        cwd=repository,
    )
    assert isinstance(config, EvolveConfig)


def test_problem_config_is_recursively_immutable_and_detached():
    source = {"outer": {"items": [1, {"answer": 2}]}}
    config = EvolveConfig(problem_config=source)
    source["outer"]["items"].append(3)

    assert len(config.problem_config["outer"]["items"]) == 2
    assert config.problem_config["outer"]["items"][0] == 1
    assert config.problem_config["outer"]["items"][1]["answer"] == 2
    with pytest.raises(TypeError):
        config.problem_config["new"] = 1
    with pytest.raises(TypeError):
        config.problem_config["outer"]["answer"] = 3
    with pytest.raises(AttributeError):
        config.problem_config["outer"]["items"].append(4)
    with pytest.raises(TypeError):
        config.problem_config._data["new"] = 1
    assert copy.deepcopy(config.problem_config) is config.problem_config


@pytest.mark.parametrize(
    "extra, message",
    [
        (
            "evolve:\n  budget:\n    audit_fraction: -0.1\n",
            "audit_fraction",
        ),
        (
            "evolve:\n"
            "  budget:\n"
            "    audit_fraction: 0.4\n"
            "    refinement_fraction: 0.2\n"
            "  archive:\n"
            "    empty_cell_fraction: 0.2\n"
            "  harnesses:\n"
            "    trial_fraction: 0.1\n"
            "  scheduler:\n"
            "    global_exploration_fraction: 0.2\n",
            "reserv.*exceeding",
        ),
        (
            "evolve:\n"
            "  budget:\n"
            "    audit_fraction: 0.01\n"
            "  audits:\n"
            "    no_memory_fraction: 0.02\n",
            "no_memory_fraction",
        ),
        ("max_seq_length: 512\nmax_new_tokens: 512\n", "smaller"),
        ("top_p: 0\n", "top_p"),
    ],
)
def test_invalid_ranges_context_and_reservations(tmp_path, extra, message):
    with pytest.raises(EvolveConfigError, match=message):
        _fresh(tmp_path, extra)


def test_production_roles_and_test_only_subsets(tmp_path):
    with pytest.raises(EvolveConfigError, match="test-only"):
        _fresh(tmp_path, "evolve:\n  roles:\n    enabled: [scout]\n")

    cfg, _, metadata = _fresh(
        tmp_path,
        "evolve:\n"
        "  roles:\n"
        "    enabled: [scout]\n"
        "    test_mode: true\n"
        "    method_incomplete: true\n",
    )
    assert cfg.evolve.roles.enabled == ("scout",)
    assert metadata["method_incomplete"] is True


@pytest.mark.parametrize(
    "extra, message",
    [
        ("evolve:\n  learning:\n    objective: entropy\n", "objective"),
        (
            "evolve:\n  learning:\n    group_k: 4\n    top_m: 5\n",
            "top_m",
        ),
        (
            "evolve:\n  learning:\n    objective: maxpo\n    top_m: 2\n",
            "MaxPO",
        ),
    ],
)
def test_learning_objective_group_and_top_m_validation(tmp_path, extra, message):
    with pytest.raises(EvolveConfigError, match=message):
        _fresh(tmp_path, extra)


def test_gpu_problem_requires_real_paths_serial_verifier_and_exclusive_gpu(tmp_path):
    task = tmp_path / "task.yml"
    task.write_text("name: fake\n", encoding="utf-8")
    lib = tmp_path / "lib"
    lib.mkdir()
    yaml_path = _write_yaml(
        tmp_path / "gpu.yaml",
        "problem: gpu_mode\n"
        "problem_type: trimul\n"
        f"task_yaml: {task}\n"
        f"lib_dir: {lib}\n"
        "gpu_ids: [0]\n"
        "kernel_gpu_id: 1\n"
        "reward_workers: 1\n"
        "max_seq_length: 2048\n"
        "max_new_tokens: 256\n",
    )
    cfg, resolved, _ = load_evolve_config(
        ["--config", str(yaml_path)], cwd=tmp_path
    )
    assert cfg.kernel_gpu_id == 1
    assert resolved["task_yaml"] == str(task.resolve())

    task.unlink()
    with pytest.raises(EvolveConfigError, match="task_yaml.*does not exist"):
        load_evolve_config(["--config", str(yaml_path)], cwd=tmp_path)


def test_gpu_problem_rejects_missing_resource_declarations(tmp_path):
    path = _write_yaml(
        tmp_path / "gpu.yaml",
        "problem: gpu_mode\n"
        "gpu_ids: [0]\n"
        "max_seq_length: 2048\n"
        "max_new_tokens: 256\n",
    )
    with pytest.raises(EvolveConfigError, match="kernel_gpu_id"):
        load_evolve_config(["--config", str(path)], cwd=tmp_path)


def test_resume_uses_authoritative_resolved_then_supported_cli(tmp_path):
    _, resolved, _ = _fresh(tmp_path, cli=("--model-name", "saved/model"))
    run_dir = tmp_path / "run"
    _write_resume(run_dir, resolved)

    # No YAML is consulted on resume, and supported explicit fields win.
    cfg, resumed, metadata = load_evolve_config(
        [
            "--resume",
            str(run_dir),
            "--num-steps",
            "25",
            "--gpu-ids",
            "4,5",
            "--num-gpus",
            "2",
        ],
        cwd=tmp_path,
    )
    assert cfg.model_name == "saved/model"
    assert cfg.evolve.budget.epochs == 25
    assert cfg.gpu_ids == (4, 5)
    assert cfg.num_gpus == 2
    assert resumed["num_steps"] == 25
    assert metadata["mode"] == "resume"
    assert metadata["previous_config_hash"] == resolved["config_hash"]
    assert metadata["config_hash"] != resolved["config_hash"]


def test_resume_selects_latest_complete_additive_config_version(tmp_path):
    _, initial, _ = _fresh(tmp_path, cli=("--model-name", "saved/model"))
    run_dir = tmp_path / "versioned-run"
    run_dir.mkdir()
    arguments = _metadata_arguments()
    write_initial_run_metadata(
        run_dir,
        requested_yaml="engine: evolve\nproblem: erdos\n",
        resolved_config=initial,
        **arguments,
    )
    latest = copy.deepcopy(initial)
    latest["gpu_ids"] = [7, 9]
    latest["num_gpus"] = 2
    latest["config_hash"] = canonical_config_hash(latest)
    paths = write_resume_run_metadata(
        run_dir,
        resume_index=1,
        resolved_config=latest,
        checkpoint_hash=content_hash({"checkpoint": 1}),
        **arguments,
    )

    config, resumed, metadata = load_evolve_config(
        ["--resume", str(run_dir)], cwd=tmp_path
    )

    assert config.gpu_ids == (7, 9)
    assert resumed["config_hash"] == latest["config_hash"]
    assert metadata["previous_config_hash"] == latest["config_hash"]
    assert metadata["effective_resume_index"] == 1
    assert metadata["config_path"] == str(paths.resolved_config.resolve())


def test_resume_num_gpus_alone_preserves_authoritative_ids_or_rejects(tmp_path):
    _, resolved, _ = _fresh(tmp_path)
    run_dir = tmp_path / "run"
    _write_resume(run_dir, resolved)

    cfg, unchanged, metadata = load_evolve_config(
        ["--resume", str(run_dir), "--num-gpus", "2"],
        cwd=tmp_path,
    )
    assert cfg.gpu_ids == (0, 2)
    assert unchanged["gpu_ids"] == [0, 2]
    assert unchanged["config_hash"] == resolved["config_hash"]
    assert metadata["cli_overrides"] == {"num_gpus": 2}

    with pytest.raises(EvolveConfigError, match="conflicts.*saved gpu_ids"):
        load_evolve_config(
            ["--resume", str(run_dir), "--num-gpus", "1"],
            cwd=tmp_path,
        )


def test_validate_resolved_document_requires_complete_canonical_shape(tmp_path):
    _, resolved, _ = _fresh(tmp_path)
    validated = validate_resolved_config_document(resolved, cwd=tmp_path)
    assert isinstance(validated, EvolveConfig)
    assert validated.to_dict()["gpu_ids"] == [0, 2]

    missing = copy.deepcopy(resolved)
    del missing["evolve"]["workers"]["max_inflight_branches"]
    missing["config_hash"] = canonical_config_hash(missing)
    with pytest.raises(EvolveConfigError, match="missing"):
        validate_resolved_config_document(missing, cwd=tmp_path)

    noncanonical = copy.deepcopy(resolved)
    noncanonical["gpu_ids"] = "0,2"
    noncanonical["config_hash"] = canonical_config_hash(noncanonical)
    with pytest.raises(EvolveConfigError, match="canonical"):
        validate_resolved_config_document(noncanonical, cwd=tmp_path)

    no_hash = copy.deepcopy(resolved)
    no_hash.pop("config_hash")
    with pytest.raises(EvolveConfigError, match="config_hash"):
        validate_resolved_config_document(no_hash, cwd=tmp_path)


def test_resume_rejects_method_changes_and_implicit_config_overlay(tmp_path):
    _, resolved, _ = _fresh(tmp_path)
    run_dir = tmp_path / "run"
    _write_resume(run_dir, resolved)
    with pytest.raises(EvolveConfigError, match="not supported"):
        load_evolve_config(
            ["--resume", str(run_dir), "--model-name", "other/model"],
            cwd=tmp_path,
        )
    with pytest.raises(EvolveConfigError, match="--config"):
        load_evolve_config(
            ["--resume", str(run_dir), "--config", "new.yaml"], cwd=tmp_path
        )


def test_resume_detects_non_evolve_missing_resolved_future_schema_and_tampering(tmp_path):
    non_evolve = tmp_path / "non-evolve"
    non_evolve.mkdir()
    (non_evolve / "config.json").write_text(
        json.dumps({"problem": "erdos"}), encoding="utf-8"
    )
    with pytest.raises(UnsupportedEvolveConfigError, match="EVOLVE"):
        load_evolve_config(["--resume", str(non_evolve)], cwd=tmp_path)

    missing = tmp_path / "missing"
    missing.mkdir()
    (missing / "config.json").write_text(
        json.dumps({"engine": "evolve", "schema_version": 1}), encoding="utf-8"
    )
    with pytest.raises(EvolveConfigError, match="config.resolved"):
        load_evolve_config(["--resume", str(missing)], cwd=tmp_path)

    _, resolved, _ = _fresh(tmp_path)
    future = tmp_path / "future"
    future_doc = dict(resolved)
    future_doc["schema_version"] = 99
    future_doc["config_hash"] = canonical_config_hash(future_doc)
    _write_resume(future, future_doc)
    with pytest.raises(UnsupportedEvolveConfigError, match="schema_version"):
        load_evolve_config(["--resume", str(future)], cwd=tmp_path)

    tampered = tmp_path / "tampered"
    _write_resume(tampered, resolved)
    doc = json.loads((tampered / "config.resolved.json").read_text())
    doc["model_name"] = "altered/model"
    (tampered / "config.resolved.json").write_text(json.dumps(doc))
    with pytest.raises(EvolveConfigError, match="config_hash"):
        load_evolve_config(["--resume", str(tampered)], cwd=tmp_path)


def test_resume_rejects_incomplete_schema_instead_of_filling_new_defaults(tmp_path):
    _, resolved, _ = _fresh(tmp_path)
    del resolved["evolve"]["workers"]["max_inflight_branches"]
    resolved["config_hash"] = canonical_config_hash(resolved)
    run_dir = tmp_path / "incomplete"
    _write_resume(run_dir, resolved)
    with pytest.raises(EvolveConfigError, match="resumed config is missing"):
        load_evolve_config(["--resume", str(run_dir)], cwd=tmp_path)


def test_resume_requires_matching_immutable_manifest(tmp_path):
    _, resolved, _ = _fresh(tmp_path)
    missing = tmp_path / "missing-manifest"
    _write_resume(missing, resolved)
    (missing / "manifest.json").unlink()
    with pytest.raises(EvolveConfigError, match="manifest"):
        load_evolve_config(["--resume", str(missing)], cwd=tmp_path)

    mismatched = tmp_path / "mismatched-manifest"
    _write_resume(mismatched, resolved)
    manifest_path = mismatched / "manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    manifest["config_hash"] = "wrong"
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")
    with pytest.raises(EvolveConfigError, match="config_hash"):
        load_evolve_config(["--resume", str(mismatched)], cwd=tmp_path)


def test_resume_rejects_noncanonical_gpu_string_in_authoritative_json(tmp_path):
    _, resolved, _ = _fresh(tmp_path)
    resolved["gpu_ids"] = "0,2"
    resolved["num_gpus"] = 2
    resolved["config_hash"] = canonical_config_hash(resolved)
    run_dir = tmp_path / "noncanonical-gpus"
    _write_resume(run_dir, resolved)

    with pytest.raises(EvolveConfigError, match="canonical"):
        load_evolve_config(["--resume", str(run_dir)], cwd=tmp_path)
