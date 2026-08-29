import json
from pathlib import Path

import pytest

from evolve.runio.schema import (
    MalformedRunError,
    UnsupportedRunSchemaError,
    detect_run_schema,
)
from evolve.ids import content_hash, content_id


FIXTURES = Path(__file__).parent / "fixtures" / "run_schemas"


def _write_json(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value), encoding="utf-8")


def _write_evolve_run(run_dir, config_schema=1, manifest_schema=1):
    config = {
        "engine": "evolve",
        "schema_version": config_schema,
        "problem": "fixture_problem",
    }
    config["config_hash"] = content_hash(config)
    _write_json(run_dir / "config.json", config)
    _write_json(run_dir / "config.resolved.json", config)
    _write_json(
        run_dir / "manifest.json",
        {
            "engine": "evolve",
            "schema_version": manifest_schema,
            "run_id": content_id("run", {"fixture": True}),
            "config_hash": config["config_hash"],
            "config_schema_version": config_schema,
        },
    )


def test_missing_engine_detects_committed_legacy_fixture_without_writes():
    run_dir = FIXTURES / "legacy_v0"
    before = {
        path.relative_to(run_dir): path.read_bytes()
        for path in run_dir.rglob("*")
        if path.is_file()
    }

    detected = detect_run_schema(run_dir)

    after = {
        path.relative_to(run_dir): path.read_bytes()
        for path in run_dir.rglob("*")
        if path.is_file()
    }
    assert detected.engine == "legacy"
    assert detected.is_legacy and not detected.is_evolve
    assert detected.config_schema_version is None
    assert detected.manifest_path is None
    assert after == before


def test_explicit_evolve_requires_and_accepts_supported_schemas(tmp_path):
    run_dir = tmp_path / "evolve-run"
    run_dir.mkdir()
    _write_evolve_run(run_dir)

    detected = detect_run_schema(run_dir)

    assert detected.engine == "evolve"
    assert detected.is_evolve and not detected.is_legacy
    assert detected.config_schema_version == 1
    assert detected.manifest_schema_version == 1
    assert detected.resolved_config_path == (run_dir / "config.resolved.json").resolve()


@pytest.mark.parametrize(
    "missing_name",
    ["config.resolved.json", "manifest.json"],
)
def test_explicit_evolve_rejects_missing_required_metadata(tmp_path, missing_name):
    run_dir = tmp_path / "evolve-run"
    run_dir.mkdir()
    _write_evolve_run(run_dir)
    (run_dir / missing_name).unlink()

    with pytest.raises(MalformedRunError):
        detect_run_schema(run_dir)


@pytest.mark.parametrize("location", ["config", "resolved", "manifest"])
def test_future_schema_is_rejected(tmp_path, location):
    run_dir = tmp_path / "evolve-run"
    run_dir.mkdir()
    _write_evolve_run(run_dir)
    filenames = {
        "config": "config.json",
        "resolved": "config.resolved.json",
        "manifest": "manifest.json",
    }
    path = run_dir / filenames[location]
    value = json.loads(path.read_text(encoding="utf-8"))
    value["schema_version"] = 999
    _write_json(path, value)

    with pytest.raises(UnsupportedRunSchemaError):
        detect_run_schema(run_dir)


def test_missing_schema_on_explicit_evolve_is_malformed(tmp_path):
    run_dir = tmp_path / "evolve-run"
    run_dir.mkdir()
    _write_evolve_run(run_dir)
    config_path = run_dir / "config.json"
    config = json.loads(config_path.read_text(encoding="utf-8"))
    del config["schema_version"]
    _write_json(config_path, config)

    with pytest.raises(MalformedRunError):
        detect_run_schema(run_dir)


def test_conflicting_engine_documents_are_rejected_as_ambiguous(tmp_path):
    run_dir = tmp_path / "ambiguous-run"
    run_dir.mkdir()
    _write_evolve_run(run_dir)
    _write_json(run_dir / "config.json", {"problem": "old-style-legacy"})

    with pytest.raises(MalformedRunError, match="ambiguous run engine"):
        detect_run_schema(run_dir)


@pytest.mark.parametrize(
    "filename,content",
    [
        ("config.json", "not json"),
        ("config.json", "[]"),
    ],
)
def test_malformed_config_is_rejected(tmp_path, filename, content):
    run_dir = tmp_path / "bad-run"
    run_dir.mkdir()
    (run_dir / filename).write_text(content, encoding="utf-8")

    with pytest.raises(MalformedRunError):
        detect_run_schema(run_dir)


@pytest.mark.parametrize(
    "field,value,message",
    [
        ("run_id", "fabricated-run", "run_id"),
        ("config_hash", "not-a-hash", "config_hash"),
        ("config_schema_version", None, "config_schema_version"),
    ],
)
def test_evolve_manifest_identity_fields_are_strict(tmp_path, field, value, message):
    run_dir = tmp_path / "evolve-run"
    run_dir.mkdir()
    _write_evolve_run(run_dir)
    manifest_path = run_dir / "manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if value is None:
        manifest.pop(field)
    else:
        manifest[field] = value
    _write_json(manifest_path, manifest)

    with pytest.raises(MalformedRunError, match=message):
        detect_run_schema(run_dir)


def test_evolve_resolved_hash_is_recomputed_from_content(tmp_path):
    run_dir = tmp_path / "evolve-run"
    run_dir.mkdir()
    _write_evolve_run(run_dir)
    resolved_path = run_dir / "config.resolved.json"
    resolved = json.loads(resolved_path.read_text(encoding="utf-8"))
    resolved["problem"] = "fabricated-after-hash"
    _write_json(resolved_path, resolved)

    with pytest.raises(MalformedRunError, match="config_hash.*content"):
        detect_run_schema(run_dir)
