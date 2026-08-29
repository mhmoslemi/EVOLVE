import json
from datetime import datetime

import pytest

from evolve.config import EvolveConfig, canonical_config_hash
from evolve.ids import content_hash

from evolve.runio import atomic
from evolve.runio.atomic import (
    ImmutableWriteError,
    atomic_write_text,
    write_immutable_text,
)
from evolve.runio.events import (
    ControllerEventWriter,
    EventLogCorruptionError,
    EventWriterOwnershipError,
    IdempotencyConflictError,
)
from evolve.runio.layout import (
    RUN_SUBDIRECTORIES,
    RunAttachmentError,
    RunCollisionError,
    create_fresh_run_layout,
    open_existing_run_layout,
)
from evolve.runio.manifest import (
    ManifestValidationError,
    resolved_config_hash,
    write_initial_run_metadata,
    write_resume_run_metadata,
)
from evolve.runio.schema import (
    MalformedRunError,
    detect_run_schema,
    resolve_effective_run_metadata,
)


def test_atomic_replace_uses_same_directory_temp_and_fsync(tmp_path, monkeypatch):
    target = tmp_path / "state.json"
    target.write_text("old", encoding="utf-8")
    real_replace = atomic.os.replace
    real_fsync = atomic.os.fsync
    replacements = []
    fsync_calls = []

    def recording_replace(source, destination):
        source_path = atomic.Path(source)
        destination_path = atomic.Path(destination)
        assert source_path.parent == destination_path.parent
        assert source_path.read_text(encoding="utf-8") == "new"
        replacements.append((source_path, destination_path))
        return real_replace(source, destination)

    def recording_fsync(fd):
        fsync_calls.append(fd)
        return real_fsync(fd)

    monkeypatch.setattr(atomic.os, "replace", recording_replace)
    monkeypatch.setattr(atomic.os, "fsync", recording_fsync)

    atomic_write_text(target, "new")

    assert target.read_text(encoding="utf-8") == "new"
    assert len(replacements) == 1
    assert len(fsync_calls) >= 2  # durable temp file and containing directory
    assert not list(tmp_path.glob(".state.json.*.tmp"))


def test_immutable_write_never_replaces_existing_content(tmp_path):
    target = tmp_path / "manifest.json"
    write_immutable_text(target, "first")

    with pytest.raises(ImmutableWriteError):
        write_immutable_text(target, "second")

    assert target.read_text(encoding="utf-8") == "first"
    assert not list(tmp_path.glob(".manifest.json.*.tmp"))


def test_fresh_layout_is_exclusive_named_and_complete(tmp_path):
    now = datetime(2026, 8, 28, 14, 5, 9)
    layout = create_fresh_run_layout(
        tmp_path,
        problem="erdos",
        model_name="org/Model Name",
        now=now,
        short_random_id="a1b2c3",
    )

    assert layout.run_dir.name == "erdos_Model_Name_0828-140509_a1b2c3"
    assert all((layout.run_dir / relative).is_dir() for relative in RUN_SUBDIRECTORIES)
    before = sorted(path.relative_to(layout.run_dir) for path in layout.run_dir.rglob("*"))

    with pytest.raises(RunCollisionError):
        create_fresh_run_layout(
            tmp_path,
            problem="erdos",
            model_name="org/Model Name",
            now=now,
            short_random_id="a1b2c3",
        )
    with pytest.raises(RunAttachmentError):
        open_existing_run_layout(layout.run_dir)

    resumed = open_existing_run_layout(layout.run_dir, resume=True)
    after = sorted(path.relative_to(layout.run_dir) for path in layout.run_dir.rglob("*"))
    assert resumed.run_dir == layout.run_dir
    assert after == before


def test_event_sequences_idempotency_and_reopen_are_durable(tmp_path):
    path = tmp_path / "events.jsonl"
    with ControllerEventWriter(path) as writer:
        first = writer.append(
            "allocation.created",
            {"allocation_id": "alloc_1", "roles": ["scout"]},
            idempotency_key="allocation:1",
            timestamp="2026-08-28T14:00:00Z",
        )
        # The durable event owns a JSON-normalized snapshot, not the caller's
        # mutable object graph.
        mutable_payload = {"items": [1]}
        snapshot = writer.append(
            "snapshot.saved",
            mutable_payload,
            idempotency_key="snapshot:1",
            timestamp="2026-08-28T14:00:00Z",
        )
        mutable_payload["items"].append(2)
        assert snapshot["payload"] == {"items": [1]}
        retried = writer.append(
            "allocation.created",
            {"allocation_id": "alloc_1", "roles": ["scout"]},
            idempotency_key="allocation:1",
            timestamp="ignored-on-idempotent-retry",
        )
        second = writer.append(
            "evidence.saved",
            {"evidence_id": "evidence_1"},
            idempotency_key="evidence:1",
            timestamp="2026-08-28T14:00:01Z",
        )
        with pytest.raises(IdempotencyConflictError):
            writer.append(
                "allocation.created",
                {"allocation_id": "different"},
                idempotency_key="allocation:1",
            )

    assert first == retried
    assert first["sequence"] == 1
    assert second["sequence"] == 3
    assert len(path.read_text(encoding="utf-8").splitlines()) == 3

    with ControllerEventWriter(path) as reopened:
        assert reopened.next_sequence == 4
        third = reopened.append(
            "barrier.closed",
            {"epoch": 0},
            idempotency_key="barrier:0",
            timestamp="2026-08-28T14:00:02Z",
        )
    assert third["sequence"] == 4


def test_event_writer_rejects_worker_process_and_torn_tail(tmp_path, monkeypatch):
    path = tmp_path / "events.jsonl"
    writer = ControllerEventWriter(path)
    owner_pid = writer._owner_pid
    monkeypatch.setattr("evolve.runio.events.os.getpid", lambda: owner_pid + 1)
    with pytest.raises(EventWriterOwnershipError):
        writer.append("worker.bad", {}, idempotency_key="bad")
    monkeypatch.undo()
    writer.close()

    with path.open("ab") as handle:
        handle.write(b'{"schema_version":1')
        handle.flush()
    with pytest.raises(EventLogCorruptionError, match="torn event-log tail"):
        ControllerEventWriter(path)

    malformed = tmp_path / "malformed-events.jsonl"
    malformed.write_bytes(b'{"schema_version":1}\n')
    with pytest.raises(EventLogCorruptionError):
        ControllerEventWriter(malformed)


def test_event_idempotency_distinguishes_json_boolean_from_integer(tmp_path):
    path = tmp_path / "typed-events.jsonl"
    with ControllerEventWriter(path) as writer:
        writer.append("typed", {"value": True}, idempotency_key="same")
        with pytest.raises(IdempotencyConflictError):
            writer.append("typed", {"value": 1}, idempotency_key="same")


def test_event_rejects_invalid_timestamp_before_publishing(tmp_path):
    path = tmp_path / "timestamp-events.jsonl"
    with ControllerEventWriter(path) as writer:
        with pytest.raises(ValueError, match="timestamp"):
            writer.append("bad", {}, idempotency_key="bad", timestamp=7)
    assert path.read_bytes() == b""


def test_event_recovery_quarantines_only_partial_tail(tmp_path):
    path = tmp_path / "recover-events.jsonl"
    with ControllerEventWriter(path) as writer:
        expected = writer.append("durable", {"x": 1}, idempotency_key="one")
    with path.open("ab") as handle:
        handle.write(b'{"partial":')

    with pytest.raises(EventLogCorruptionError, match="torn"):
        ControllerEventWriter(path)
    with ControllerEventWriter(path, recover_torn_tail=True) as recovered:
        assert recovered.events == [expected]
        assert recovered.recovered_tail_path is not None
        assert recovered.recovered_tail_path.read_bytes() == b'{"partial":'
    with ControllerEventWriter(path) as reopened:
        assert reopened.events == [expected]


def test_event_reader_rejects_float_schema_and_sequence(tmp_path):
    for name, field in (("schema", "schema_version"), ("sequence", "sequence")):
        path = tmp_path / f"{name}.jsonl"
        event = {
            "schema_version": 1,
            "sequence": 1,
            "idempotency_key": "key",
            "event_type": "type",
            "payload": {},
            "timestamp": "now",
        }
        event[field] = 1.0
        path.write_text(json.dumps(event) + "\n", encoding="utf-8")
        with pytest.raises(EventLogCorruptionError):
            ControllerEventWriter(path)


def _metadata_arguments():
    return {
        "command": ["python", "train_evolve.py", "--problem", "toy"],
        "environment": {"CUDA_VISIBLE_DEVICES": "0,1"},
        "git_state": {"commit": "abc123", "dirty": False},
        "model": {"name": "fake/model", "revision": "test"},
        "package_versions": {"python": "3.11", "torch": "fake"},
        "host": {"hostname": "cpu-fixture", "platform": "test"},
        "gpus": [{"physical_id": 0}, {"physical_id": 1}],
        "worker_topology": {"generation_workers": 2, "reward_workers": 1},
        "seeds": {"run": 42, "scheduler": 43},
        "versions": {
            "harnesses": ["baseline_v1"],
            "verifier": "toy_v1",
            "descriptor": "toy_v1",
            "reporting": 1,
        },
    }


def _resolved_config(gpu_ids=(0,)):
    config = EvolveConfig(
        problem="toy",
        model_name="fake/model",
        gpu_ids=tuple(gpu_ids),
        num_gpus=len(gpu_ids),
    )
    document = config.to_dict(compatibility=True)
    document["config_hash"] = canonical_config_hash(document)
    return document


def _with_gpu_ids(document, gpu_ids):
    updated = json.loads(json.dumps(document))
    updated["gpu_ids"] = list(gpu_ids)
    updated["num_gpus"] = len(gpu_ids)
    updated["config_hash"] = canonical_config_hash(updated)
    return updated


def test_initial_manifest_is_immutable_complete_and_json_typed(tmp_path):
    layout = create_fresh_run_layout(
        tmp_path,
        problem="toy",
        model_name="fake/model",
        now=datetime(2026, 8, 28, 14, 0, 0),
        short_random_id="abc123",
    )
    resolved = _resolved_config((0, 1))
    compatibility = json.loads(json.dumps(resolved))
    arguments = _metadata_arguments()
    paths = write_initial_run_metadata(
        layout.run_dir,
        requested_yaml="engine: evolve\nproblem: toy\n",
        resolved_config=resolved,
        compatibility_config=compatibility,
        created_at="2026-08-28T14:00:00Z",
        **arguments,
    )

    saved_resolved = json.loads(paths.resolved_config.read_text(encoding="utf-8"))
    saved_compat = json.loads(paths.compatibility_config.read_text(encoding="utf-8"))
    manifest = json.loads(paths.manifest.read_text(encoding="utf-8"))
    environment = json.loads(paths.environment.read_text(encoding="utf-8"))
    command = json.loads(paths.command.read_text(encoding="utf-8"))
    assert paths.requested_config.read_text(encoding="utf-8") == (
        "engine: evolve\nproblem: toy\n"
    )
    assert saved_resolved == resolved
    assert saved_resolved["config_hash"] == manifest["config_hash"]
    assert saved_compat["engine"] == "evolve"
    assert saved_compat["schema_version"] == 1
    assert saved_compat["gpu_ids"] == [0, 1]
    assert isinstance(saved_compat["gpu_ids"], list)
    assert manifest["config_hash"] == resolved["config_hash"]
    assert manifest["run_id"].startswith("run:")
    for key in (
        "git",
        "model",
        "packages",
        "host",
        "gpus",
        "worker_topology",
        "seeds",
        "versions",
    ):
        assert key in manifest
    assert environment["variables"]["CUDA_VISIBLE_DEVICES"] == "0,1"
    assert command["argv"] == arguments["command"]
    assert detect_run_schema(layout.run_dir).is_evolve

    initial_bytes = {
        path.name: path.read_bytes()
        for path in (
            paths.requested_config,
            paths.resolved_config,
            paths.compatibility_config,
            paths.manifest,
            paths.command,
            paths.environment,
        )
    }
    with pytest.raises(ImmutableWriteError):
        write_initial_run_metadata(
            layout.run_dir,
            requested_yaml="different: true\n",
            resolved_config=resolved,
            compatibility_config=compatibility,
            **arguments,
        )
    assert all((layout.run_dir / name).read_bytes() == content
               for name, content in initial_bytes.items())


def test_manifest_hash_matches_authoritative_embedded_config_hash(tmp_path):
    layout = create_fresh_run_layout(
        tmp_path,
        problem="toy",
        model_name="fake/model",
        now=datetime(2026, 8, 28, 14, 0, 0),
        short_random_id="abc124",
    )
    resolved = _resolved_config(())

    paths = write_initial_run_metadata(
        layout.run_dir,
        requested_yaml="engine: evolve\n",
        resolved_config=resolved,
        **_metadata_arguments(),
    )

    saved = json.loads(paths.resolved_config.read_text(encoding="utf-8"))
    manifest = json.loads(paths.manifest.read_text(encoding="utf-8"))
    assert manifest["config_hash"] == saved["config_hash"]


def test_runio_and_config_hashes_agree_on_negative_zero():
    negative = {
        "engine": "evolve",
        "schema_version": 1,
        "temperature": -0.0,
    }
    positive = dict(negative, temperature=0.0)

    assert resolved_config_hash(negative) == resolved_config_hash(positive)
    assert resolved_config_hash(negative) == canonical_config_hash(negative)


def test_compatibility_config_must_be_a_canonical_authoritative_copy(tmp_path):
    layout = create_fresh_run_layout(
        tmp_path,
        problem="toy",
        model_name="fake/model",
        now=datetime(2026, 8, 28, 14, 0, 0),
        short_random_id="abc126",
    )
    resolved = _resolved_config(())
    projection = json.loads(json.dumps(resolved))
    projection["problem"] = "different"

    with pytest.raises(ManifestValidationError, match="canonical copy"):
        write_initial_run_metadata(
            layout.run_dir,
            requested_yaml="engine: evolve\n",
            resolved_config=resolved,
            compatibility_config=projection,
            **_metadata_arguments(),
        )
    assert not (layout.run_dir / "manifest.json").exists()
    assert not (layout.run_dir / "config.resolved.json").exists()


def test_manifest_identity_sections_are_required_and_cpu_gpus_may_be_empty(tmp_path):
    valid_layout = create_fresh_run_layout(
        tmp_path,
        problem="toy",
        model_name="fake/model",
        now=datetime(2026, 8, 28, 14, 0, 0),
        short_random_id="abc127",
    )
    arguments = _metadata_arguments()
    arguments["gpus"] = []
    paths = write_initial_run_metadata(
        valid_layout.run_dir,
        requested_yaml="engine: evolve\n",
        resolved_config=_resolved_config(()),
        **arguments,
    )
    assert json.loads(paths.manifest.read_text(encoding="utf-8"))["gpus"] == []

    invalid_layout = create_fresh_run_layout(
        tmp_path,
        problem="toy",
        model_name="fake/model",
        now=datetime(2026, 8, 28, 14, 0, 1),
        short_random_id="abc128",
    )
    invalid_arguments = _metadata_arguments()
    invalid_arguments["versions"] = {}
    with pytest.raises(ManifestValidationError, match="versions.*non-empty"):
        write_initial_run_metadata(
            invalid_layout.run_dir,
            requested_yaml="engine: evolve\n",
            resolved_config=_resolved_config(()),
            **invalid_arguments,
        )
    assert not (invalid_layout.run_dir / "config.resolved.json").exists()


def test_manifest_rejects_incorrect_embedded_config_hash(tmp_path):
    layout = create_fresh_run_layout(
        tmp_path,
        problem="toy",
        model_name="fake/model",
        now=datetime(2026, 8, 28, 14, 0, 0),
        short_random_id="abc125",
    )
    invalid = _resolved_config(())
    invalid["config_hash"] = "definitely-wrong"
    with pytest.raises(ManifestValidationError, match="config_hash"):
        write_initial_run_metadata(
            layout.run_dir,
            requested_yaml="engine: evolve\n",
            resolved_config=invalid,
            **_metadata_arguments(),
        )


def test_manifest_rejects_partial_resolved_config_before_any_publication(tmp_path):
    layout = create_fresh_run_layout(
        tmp_path,
        problem="toy",
        model_name="fake/model",
        now=datetime(2026, 8, 28, 14, 0, 0),
        short_random_id="abc133",
    )
    partial = {"engine": "evolve", "schema_version": 1, "problem": "toy"}
    partial["config_hash"] = canonical_config_hash(partial)

    with pytest.raises(ManifestValidationError, match="complete canonical"):
        write_initial_run_metadata(
            layout.run_dir,
            requested_yaml="engine: evolve\n",
            resolved_config=partial,
            **_metadata_arguments(),
        )

    for name in (
        "config.requested.yaml",
        "config.resolved.json",
        "config.json",
        "command.json",
        "environment.json",
        "manifest.json",
    ):
        assert not (layout.run_dir / name).exists()


def test_resume_metadata_is_additive_and_versions_checkpoint_hash(tmp_path):
    layout = create_fresh_run_layout(
        tmp_path,
        problem="toy",
        model_name="fake/model",
        now=datetime(2026, 8, 28, 14, 0, 0),
        short_random_id="abc123",
    )
    initial = _resolved_config((0,))
    arguments = _metadata_arguments()
    initial_paths = write_initial_run_metadata(
        layout.run_dir,
        requested_yaml="engine: evolve\n",
        resolved_config=initial,
        **arguments,
    )
    immutable_before = {
        path: path.read_bytes()
        for path in (
            initial_paths.requested_config,
            initial_paths.resolved_config,
            initial_paths.compatibility_config,
            initial_paths.manifest,
            initial_paths.command,
            initial_paths.environment,
        )
    }
    resumed = _with_gpu_ids(initial, (2, 3))

    resume_paths = write_resume_run_metadata(
        layout.run_dir,
        resume_index=1,
        resolved_config=resumed,
        checkpoint_hash=content_hash({"checkpoint": 0}),
        resumed_at="2026-08-28T15:00:00Z",
        **arguments,
    )

    resume_manifest = json.loads(resume_paths.manifest.read_text(encoding="utf-8"))
    assert resume_paths.manifest.name == "manifest.resume001.json"
    assert resume_manifest["resume_index"] == 1
    assert resume_manifest["checkpoint_hash"] == content_hash({"checkpoint": 0})
    assert resume_manifest["config_hash"] == resolved_config_hash(resumed)
    assert resume_manifest["run_id"] == json.loads(
        initial_paths.manifest.read_text(encoding="utf-8")
    )["run_id"]
    assert all(path.read_bytes() == content for path, content in immutable_before.items())
    with pytest.raises(ImmutableWriteError):
        write_resume_run_metadata(
            layout.run_dir,
            resume_index=1,
            resolved_config=resumed,
            checkpoint_hash=content_hash({"checkpoint": "other"}),
            **arguments,
        )


def test_effective_metadata_resolves_latest_contiguous_resume(tmp_path):
    layout = create_fresh_run_layout(
        tmp_path,
        problem="toy",
        model_name="fake/model",
        now=datetime(2026, 8, 28, 14, 0, 0),
        short_random_id="abc129",
    )
    arguments = _metadata_arguments()
    initial = _resolved_config((0,))
    write_initial_run_metadata(
        layout.run_dir,
        requested_yaml="engine: evolve\n",
        resolved_config=initial,
        **arguments,
    )
    first = _with_gpu_ids(initial, (1,))
    write_resume_run_metadata(
        layout.run_dir,
        resume_index=1,
        resolved_config=first,
        checkpoint_hash=content_hash({"checkpoint": 1}),
        **arguments,
    )
    second = _with_gpu_ids(initial, (2,))
    second_paths = write_resume_run_metadata(
        layout.run_dir,
        resume_index=2,
        resolved_config=second,
        checkpoint_hash=content_hash({"checkpoint": 2}),
        **arguments,
    )

    effective = resolve_effective_run_metadata(layout.run_dir)

    assert effective.resume_index == 2
    assert effective.resolved_config_path == second_paths.resolved_config.resolve()
    assert effective.manifest_path == second_paths.manifest.resolve()
    assert effective.resolved_config["gpu_ids"] == [2]
    assert effective.manifest["previous_manifest"] == "manifest.resume001.json"


def test_effective_metadata_rejects_partial_resume_and_writer_adds_nothing(tmp_path):
    layout = create_fresh_run_layout(
        tmp_path,
        problem="toy",
        model_name="fake/model",
        now=datetime(2026, 8, 28, 14, 0, 0),
        short_random_id="abc130",
    )
    arguments = _metadata_arguments()
    config = _resolved_config((0,))
    write_initial_run_metadata(
        layout.run_dir,
        requested_yaml="engine: evolve\n",
        resolved_config=config,
        **arguments,
    )
    first_paths = write_resume_run_metadata(
        layout.run_dir,
        resume_index=1,
        resolved_config=config,
        checkpoint_hash=content_hash({"checkpoint": 1}),
        **arguments,
    )
    first_paths.command.unlink()

    with pytest.raises(MalformedRunError, match="partial.*command"):
        resolve_effective_run_metadata(layout.run_dir)
    with pytest.raises(ManifestValidationError, match="complete valid.*chain"):
        write_resume_run_metadata(
            layout.run_dir,
            resume_index=2,
            resolved_config=config,
            checkpoint_hash=content_hash({"checkpoint": 2}),
            **arguments,
        )
    assert not (layout.run_dir / "manifest.resume002.json").exists()
    assert not (layout.run_dir / "config.resolved.resume002.json").exists()


def test_effective_metadata_rejects_gap_and_bad_checkpoint_hash(tmp_path):
    gap_layout = create_fresh_run_layout(
        tmp_path,
        problem="toy",
        model_name="fake/model",
        now=datetime(2026, 8, 28, 14, 0, 0),
        short_random_id="abc131",
    )
    arguments = _metadata_arguments()
    config = _resolved_config((0,))
    write_initial_run_metadata(
        gap_layout.run_dir,
        requested_yaml="engine: evolve\n",
        resolved_config=config,
        **arguments,
    )
    (gap_layout.run_dir / "manifest.resume002.json").write_text("{}\n", encoding="utf-8")
    with pytest.raises(MalformedRunError, match="contiguous.*missing.*1"):
        resolve_effective_run_metadata(gap_layout.run_dir)

    hash_layout = create_fresh_run_layout(
        tmp_path,
        problem="toy",
        model_name="fake/model",
        now=datetime(2026, 8, 28, 14, 0, 1),
        short_random_id="abc132",
    )
    write_initial_run_metadata(
        hash_layout.run_dir,
        requested_yaml="engine: evolve\n",
        resolved_config=config,
        **arguments,
    )
    resume_paths = write_resume_run_metadata(
        hash_layout.run_dir,
        resume_index=1,
        resolved_config=config,
        checkpoint_hash=content_hash({"checkpoint": 1}),
        **arguments,
    )
    resume_manifest = json.loads(resume_paths.manifest.read_text(encoding="utf-8"))
    resume_manifest["checkpoint_hash"] = "not-a-hash"
    resume_paths.manifest.write_text(json.dumps(resume_manifest), encoding="utf-8")
    with pytest.raises(MalformedRunError, match="checkpoint_hash"):
        resolve_effective_run_metadata(hash_layout.run_dir)
