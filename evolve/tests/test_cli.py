import json

import pytest

import evolve.cli as cli_module
from evolve.cli import build_dry_plan, main
from evolve.config import load_evolve_config


def _config(path):
    path.write_text(
        """\
engine: evolve
problem: erdos
model_name: fake/model
gpu_ids: []
num_gpus: 0
evolve:
  budget:
    epochs: 2
    verifier_calls: 20
  workers:
    max_inflight_branches: 10
""",
        encoding="utf-8",
    )


def test_validate_config_is_read_only_and_model_free(tmp_path, capsys, monkeypatch):
    config_path = tmp_path / "config.yaml"
    _config(config_path)
    monkeypatch.chdir(tmp_path)

    result = main(["--config", str(config_path), "--validate-config"])

    assert result == 0
    output = json.loads(capsys.readouterr().out)
    assert output["valid"] is True
    assert output["engine"] == "evolve"
    assert output["problem"] == "erdos"
    assert list(tmp_path.iterdir()) == [config_path]


def test_dry_plan_reports_frozen_dimensions_and_does_not_create_run(
    tmp_path, capsys, monkeypatch
):
    config_path = tmp_path / "config.yaml"
    _config(config_path)
    monkeypatch.chdir(tmp_path)

    result = main(["--config", str(config_path), "--dry-plan"])

    assert result == 0
    plan = json.loads(capsys.readouterr().out)
    assert plan["epochs_total"] == 2
    assert plan["model_loading"] is False
    assert plan["writes_run_directory"] is False
    assert plan["frozen_arm_dimensions"] == [
        "cell_id",
        "role",
        "option_id",
        "harness_version",
        "horizon",
        "cost_class",
    ]
    assert plan["minimum_reservations_per_full_wave"]["every_role"] == 3
    assert list(tmp_path.iterdir()) == [config_path]


def test_num_steps_is_total_epoch_alias_in_dry_plan(tmp_path):
    config_path = tmp_path / "config.yaml"
    _config(config_path)
    config, resolved, metadata = load_evolve_config(
        ["--config", str(config_path), "--num-steps", "7"], cwd=tmp_path
    )

    plan = build_dry_plan(config, metadata["config_hash"])
    assert plan["epochs_total"] == 7
    assert resolved["num_steps"] == 7


def test_invalid_config_exits_before_runtime_import(tmp_path, capsys):
    config_path = tmp_path / "bad.yaml"
    config_path.write_text(
        "engine: evolve\nevolve:\n  scheduler:\n    typo: true\n",
        encoding="utf-8",
    )

    result = main(["--config", str(config_path), "--validate-config"])

    assert result == 2
    assert "unknown" in capsys.readouterr().err.lower()


def test_only_absent_engine_module_gets_in_progress_message(
    tmp_path, capsys, monkeypatch
):
    config_path = tmp_path / "config.yaml"
    _config(config_path)

    def absent_engine(_name, _package):
        raise ModuleNotFoundError(
            "No module named 'evolve.engine'", name="evolve.engine"
        )

    monkeypatch.setattr(cli_module.importlib, "import_module", absent_engine)
    result = main(["--config", str(config_path)])

    assert result == 3
    assert "not operational yet" in capsys.readouterr().err


def test_runtime_dependency_module_error_is_not_mislabeled(tmp_path, monkeypatch):
    config_path = tmp_path / "config.yaml"
    _config(config_path)

    def missing_dependency(_name, _package):
        raise ModuleNotFoundError(
            "No module named 'runtime_dependency'", name="runtime_dependency"
        )

    monkeypatch.setattr(cli_module.importlib, "import_module", missing_dependency)
    with pytest.raises(ModuleNotFoundError, match="runtime_dependency"):
        main(["--config", str(config_path)])
