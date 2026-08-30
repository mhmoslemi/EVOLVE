import json
import os
from pathlib import Path
import shlex
import subprocess
import sys

import pytest

import evolve.cli as cli_module
from evolve.cli import build_dry_plan, format_startup_banner, main
from evolve.config import load_evolve_config, parse_evolve_args


def _config(path):
    path.write_text(
        """\
engine: evolve
problem: erdos
model_name: fake/model
gpu_ids: []
num_gpus: 0
num_seed_states: 4
evolve:
  budget:
    epochs: 2
    verifier_calls: 20
  workers:
    max_inflight_branches: 20
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


def test_startup_banner_separates_sections_and_preserves_width(tmp_path):
    config_path = tmp_path / "config.yaml"
    _config(config_path)
    config, _, metadata = load_evolve_config(
        ["--config", str(config_path)], cwd=tmp_path
    )

    banner = format_startup_banner(config, metadata, width=100)
    lines = banner.splitlines()
    spacer = "│" + " " * 98 + "│"

    assert all(len(line) == 100 for line in lines)
    assert lines.count(spacer) == 4
    assert "4 deterministic problem seeds" in banner
    assert "independently verified" in banner
    for title in (
        "MODEL & SAMPLING",
        "ROLE ADAPTER LEARNING",
        "RESOURCES",
        "SEARCH BUDGET",
    ):
        section_index = next(
            index for index, line in enumerate(lines) if f" {title} " in line
        )
        assert lines[section_index - 1] == spacer


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


def test_documented_run_commands_use_supported_cli_flags():
    repository = Path(__file__).resolve().parents[2]
    commands = [
        line.strip()
        for line in (repository / "README.md").read_text(
            encoding="utf-8"
        ).splitlines()
        if line.strip().startswith("sh run.sh")
    ]

    assert commands
    for command in commands:
        words = shlex.split(command)
        assert words[:2] == ["sh", "run.sh"]
        parse_evolve_args(words[2:])


def test_run_sh_forwards_explicit_validation_without_creating_a_run():
    repository = Path(__file__).resolve().parents[2]
    runs_before = sorted(
        path.name for path in (repository / "runs").iterdir()
    ) if (repository / "runs").is_dir() else []
    environment = dict(os.environ)
    environment.pop("CUDA_VISIBLE_DEVICES", None)
    environment["EVOLVE_PYTHON"] = sys.executable

    completed = subprocess.run(
        [
            "sh",
            "run.sh",
            "--config",
            "configs/erdos.yaml",
            "--validate-config",
        ],
        cwd=repository,
        env=environment,
        check=False,
        capture_output=True,
        text=True,
        timeout=10.0,
    )

    assert completed.returncode == 0, completed.stderr
    assert json.loads(completed.stdout)["valid"] is True
    runs_after = sorted(
        path.name for path in (repository / "runs").iterdir()
    ) if (repository / "runs").is_dir() else []
    assert runs_after == runs_before
