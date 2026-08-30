"""CPU-only composed-engine, completed-resume, and crash-recovery gates."""

from __future__ import annotations

import hashlib
import json
import threading
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from types import SimpleNamespace
from typing import Any, Dict, List, Optional, Tuple

import pytest

from evolve.config import EvolveConfig, load_evolve_config
from evolve.engine import (
    EngineError,
    EngineWorkers,
    EvolveEngine,
    _apply_role_artifact_retention,
    _json_native_answer_payload,
    _latest_completed_checkpoint,
)
from evolve.ids import content_hash, content_id
from evolve.learning.trainer import (
    GradientStepRequest,
    GradientStepResult,
)
from evolve.runio import create_fresh_run_layout, open_existing_run_layout
from evolve.reporting.console import format_best_answer, format_progress, format_status
from evolve.types import Channel, FrozenDict, VerifiedScientificState
from evolve.viz.run import main as viz_main
from evolve.verifier import ProblemScientificAdapter, VerificationPolicy
from evolve.workers.verification import (
    GenerationOutcome,
    build_proposal_and_verify,
    persist_generation_arrival,
)
from problems.evolve_toy import EvolveToyProblem


_FAKE_POLICY = VerificationPolicy.create(
    version="evolve_engine_v1", production=True
)


def _write_config(path: Path, *, epochs: int = 1) -> Path:
    path.write_text(
        f"""\
engine: evolve
schema_version: 1
problem: evolve_toy
target: 0.0
num_seed_states: 4
sandbox_timeout_s: 2
reward_workers: 2
model_name: fixture/evolve-engine
backend: hf
generation_backend: hf
max_seq_length: 512
max_new_tokens: 64
load_in_4bit: false
gpu_ids: []
num_gpus: 0
vllm_tensor_parallel_size: 0
seed: 17
deterministic: true
evolve:
  budget:
    epochs: {epochs}
    verifier_calls: 4096
    audit_fraction: 0.15
    refinement_fraction: 0.05
  archive:
    elites_per_cell: 3
    empty_cell_fraction: 0.10
  roles:
    enabled: [scout, mechanist, challenger]
  options:
    max_horizon: 2
    branch_budget: 8
  harnesses:
    trial_fraction: 0.05
    active_versions: [baseline_v1]
  scheduler:
    posterior: zero_inflated_tail
    global_exploration_fraction: 0.10
  audits:
    no_memory_fraction: 0.05
    min_pairs_for_promotion: 2
  learning:
    objective: ordergrad
    top_m: 1
    group_k: 4
  refinement:
    max_attempts: 2
    max_depth: 2
  reporting:
    status_every_verifications: 4
    plots_every_epochs: 1
  workers:
    max_inflight_branches: 12
""",
        encoding="utf-8",
    )
    return path


def _load_fresh(config_path: Path) -> Tuple[EvolveConfig, Dict[str, Any], Dict[str, Any]]:
    config, resolved, metadata = load_evolve_config(
        ["--config", str(config_path)], cwd=config_path.parent
    )
    return config, resolved, metadata


def _load_resume(
    run_dir: Path, *, target_epochs: int
) -> Tuple[EvolveConfig, Dict[str, Any], Dict[str, Any]]:
    config, resolved, metadata = load_evolve_config(
        ["--resume", str(run_dir), "--num-steps", str(target_epochs)],
        cwd=run_dir.parent,
    )
    return config, resolved, metadata


def _adapter(config: EvolveConfig) -> ProblemScientificAdapter:
    return ProblemScientificAdapter(
        EvolveToyProblem(dict(config.problem_runtime_config)),
        problem_id=config.problem,
    )


class _FakeWorkers:
    """Persisting fake worker boundary with optional deterministic crash."""

    def __init__(
        self,
        *,
        runs_root: Path,
        adapter: ProblemScientificAdapter,
        fail_on_call: Optional[int] = None,
        concurrent: bool = False,
    ) -> None:
        self.runs_root = runs_root
        self.adapter = adapter
        self.fail_on_call = fail_on_call
        self.concurrent = concurrent
        self.branch_calls = 0
        self.gradient_requests: List[GradientStepRequest] = []
        self.shutdown_calls = 0
        self.max_active_branches = 0
        self._active_branches = 0
        self._submitted_branches = 0
        self._lock = threading.Lock()
        self._start_barrier = threading.Barrier(3) if concurrent else None
        self._pool = ThreadPoolExecutor(max_workers=3) if concurrent else None

    def _run_dir(self) -> Path:
        candidates = [path for path in self.runs_root.iterdir() if path.is_dir()]
        assert len(candidates) == 1
        return candidates[0]

    def branch_step(self, request):
        with self._lock:
            self.branch_calls += 1
            call_number = self.branch_calls
        if self.fail_on_call == call_number:
            raise RuntimeError("injected fake-worker crash")

        point_options = (
            (3, -2),
            (2, -2),
            (3, -1),
            (1, -2),
            (3, 0),
            (0, -2),
        )
        if self.concurrent:
            # Completion order must not affect rollout content in the
            # multi-worker gate.
            point_index = (
                request.branch.seed + request.step_index
            ) % len(point_options)
        elif request.branch.channel == Channel.PRODUCTION:
            # The sequential fake guarantees variation inside the homogeneous
            # production group, so the learning path is exercised on every
            # platform and random run ID. A restarted worker repeats the same
            # call prefix and therefore preserves partial-epoch replay.
            point_index = (call_number - 1) % len(point_options)
        else:
            # Matched audit/refinement sides share a seed; preserve their
            # common randomness instead of assigning by execution order.
            point_index = (
                request.branch.seed + request.step_index
            ) % len(point_options)
        point = point_options[point_index]
        prompt = (
            f"fake:{request.branch.branch_id}:{request.step_index}:"
            f"{request.action}"
        )
        response = (
            "```python\n"
            "def run_toy():\n"
            f"    return [{point[0]}, {point[1]}]\n"
            "```"
        )
        generation = GenerationOutcome(
            prompt=prompt,
            text=response,
            token_ids=(100 + point_index, 200 + request.step_index),
            log_probabilities=(-0.25, -0.5),
            token_mask=(True, True),
            seed=request.branch.seed + request.step_index,
        )
        run_dir = self._run_dir()
        persist_generation_arrival(
            run_dir=run_dir, request=request, generation=generation
        )
        return build_proposal_and_verify(
            run_id=self._run_id(run_dir),
            problem_id=self.adapter.problem_id,
            branch_id=request.branch.branch_id,
            parent_state_id=request.parent_state_id,
            step_index=request.step_index,
            generation=generation,
            extract_answer=lambda _source: list(point),
            adapter=self.adapter,
            verification_policy=_FAKE_POLICY,
            harness_id=request.branch.harness_id,
            policy_snapshot_id=request.branch.role_snapshot_id,
            run_dir=run_dir,
        )

    @staticmethod
    def _run_id(run_dir: Path) -> str:
        return json.loads(
            (run_dir / "manifest.json").read_text(encoding="utf-8")
        )["run_id"]

    def gradient_step(self, request: GradientStepRequest) -> GradientStepResult:
        self.gradient_requests.append(request)
        assert request.traces
        assert {trace.role for trace in request.traces} == {request.role}
        assert {
            trace.role_snapshot_id for trace in request.traces
        } == {request.traces[0].role_snapshot_id}
        return GradientStepResult(
            loss=-sum(request.advantages),
            kl=0.0,
            gradient_norm=1.0,
            adapter_state={
                "role": request.role.value,
                "groups": list(request.group_ids),
            },
            optimizer_state={"step": len(self.gradient_requests)},
        )

    def shutdown(self) -> None:
        if self._pool is not None:
            self._pool.shutdown(wait=True)
        self.shutdown_calls += 1

    def submit_branch(self, callback):
        assert self._pool is not None
        with self._lock:
            self._submitted_branches += 1
            submitted = self._submitted_branches

        def run():
            with self._lock:
                self._active_branches += 1
                self.max_active_branches = max(
                    self.max_active_branches, self._active_branches
                )
            try:
                if submitted <= 3:
                    assert self._start_barrier is not None
                    self._start_barrier.wait(timeout=2.0)
                return callback()
            finally:
                with self._lock:
                    self._active_branches -= 1

        return self._pool.submit(run)

    def boundary(self) -> EngineWorkers:
        return EngineWorkers(
            branch_step_executor=self.branch_step,
            gradient_step=self.gradient_step,
            submit_branch=(self.submit_branch if self.concurrent else None),
            shutdown=self.shutdown,
        )


def _run_engine(
    *,
    config: EvolveConfig,
    resolved: Dict[str, Any],
    metadata: Dict[str, Any],
    runs_root: Path,
    workers: _FakeWorkers,
) -> int:
    return EvolveEngine(
        config=config,
        resolved_config=resolved,
        metadata=metadata,
        workers=workers.boundary(),
        adapter=workers.adapter,
        runs_root=runs_root,
    ).run()


def _events(run_dir: Path) -> List[Dict[str, Any]]:
    return [
        json.loads(line)
        for line in (run_dir / "events.jsonl").read_text(
            encoding="utf-8"
        ).splitlines()
    ]


def _assert_machine_readable_artifacts(run_dir: Path) -> None:
    json_paths = list(run_dir.rglob("*.json"))
    jsonl_paths = list(run_dir.rglob("*.jsonl"))
    assert json_paths
    assert jsonl_paths
    for path in json_paths:
        json.loads(path.read_text(encoding="utf-8"))
    for path in jsonl_paths:
        for line_number, line in enumerate(
            path.read_text(encoding="utf-8").splitlines(), start=1
        ):
            assert line.strip(), f"blank JSONL record at {path}:{line_number}"
            json.loads(line)
    assert not list(run_dir.rglob("*.tmp"))


def test_best_answer_projection_thaws_erdos_payload_for_json() -> None:
    answer_payload = {
        "schema_version": 1,
        "problem": "erdos",
        "n_points": 4,
        "h_values": [0.0, 1.0, 1.0, 0.0],
    }
    state = VerifiedScientificState(
        state_id=content_id("state", {"case": "erdos-best"}),
        proposal_id=content_id("proposal", {"case": "erdos-best"}),
        evidence_id=content_id("evidence", {"case": "erdos-best"}),
        problem_id="erdos",
        answer_payload=answer_payload,
        resolved=True,
        admitted=True,
        confirmed=True,
        internal_reward=1.0,
    )

    assert isinstance(state.answer_payload, FrozenDict)
    projected = _json_native_answer_payload(state)
    assert projected == answer_payload
    assert isinstance(projected, dict)
    assert isinstance(projected["h_values"], list)
    json.dumps({"answer_payload": projected}, allow_nan=False)
    rendered = format_best_answer(
        SimpleNamespace(state=state, rendered_paths=())
    )
    assert '"problem": "erdos"' in rendered
    assert '"h_values": [' in rendered


def test_progress_format_and_live_status_expose_stage_and_remaining_work() -> None:
    rendered = format_progress(
        "production generation + verification",
        epoch=1,
        total_epochs=30,
        completed=3,
        total=8,
        unit="branches",
        detail="5 verifications completed",
        bar_width=8,
    )
    assert "epoch 2/30" in rendered
    assert "3/8 branches (38%, 5 left)" in rendered
    assert "5 verifications completed" in rendered

    status = format_status(
        {
            "run_id": "run:test",
            "epoch": 1,
            "archive_coverage": 0.5,
            "live_epoch": {
                "epoch": 1,
                "total_epochs": 30,
                "stage": "production generation + verification",
                "completed_branches": 3,
                "total_branches": 8,
                "completed_verifications": 5,
            },
        }
    )
    assert "production generation + verification" in status
    assert "3/8 branches" in status


def test_fake_epoch_and_completed_barrier_resume(tmp_path: Path, capsys) -> None:
    config_path = _write_config(tmp_path / "toy.yaml", epochs=1)
    runs_root = tmp_path / "runs"
    config, resolved, metadata = _load_fresh(config_path)
    first_workers = _FakeWorkers(
        runs_root=runs_root, adapter=_adapter(config)
    )

    assert _run_engine(
        config=config,
        resolved=resolved,
        metadata=metadata,
        runs_root=runs_root,
        workers=first_workers,
    ) == 0
    assert first_workers.shutdown_calls == 1
    run_dir = first_workers._run_dir()
    fresh_output = capsys.readouterr().out
    assert f"EVOLVE · fresh run directory · {run_dir.resolve()}" in fresh_output
    assert "EVOLVE · epoch 1/1 · production generation + verification" in fresh_output
    assert "EVOLVE · epoch 1/1 · role learning" in fresh_output
    assert "origin=deterministic problem bootstrap seed" in fresh_output
    bootstrap_before = (run_dir / "bootstrap.summary.json").read_bytes()
    epoch_zero_before = (run_dir / "step00/step00.summary.json").read_bytes()

    final = json.loads((run_dir / "final.summary.json").read_text(encoding="utf-8"))
    status = json.loads((run_dir / "status.json").read_text(encoding="utf-8"))
    bootstrap_checkpoint = json.loads(
        (run_dir / "checkpoints/checkpoint_epoch000.json").read_text(
            encoding="utf-8"
        )
    )
    best_evidence = json.loads(
        (run_dir / "best/evidence.json").read_text(encoding="utf-8")
    )
    assert final["epochs_completed"] == 1
    assert final["target_epochs_reached"] is True
    assert status["confirmed_record"]["internal_reward"] == final["confirmed_record"]
    assert (
        status["confirmed_record"]["internal_reward"]
        >= bootstrap_checkpoint["record"]["internal_reward"]
    )
    assert best_evidence["confirmed"] is True
    assert (
        best_evidence["evidence_id"]
        == status["confirmed_record"]["holder"]["evidence_id"]
    )
    assert (run_dir / "best/latest.json").is_file()
    assert list((run_dir / "step00/branches").glob("*/steps/*.arrival.json"))
    assert list((run_dir / "artifacts/verification_attempts").glob("*/*.evidence.json"))
    assert first_workers.gradient_requests

    resumed_config, resumed_resolved, resumed_metadata = _load_resume(
        run_dir, target_epochs=2
    )
    resumed_workers = _FakeWorkers(
        runs_root=runs_root, adapter=_adapter(resumed_config)
    )
    assert _run_engine(
        config=resumed_config,
        resolved=resumed_resolved,
        metadata=resumed_metadata,
        runs_root=runs_root,
        workers=resumed_workers,
    ) == 0
    assert resumed_workers.shutdown_calls == 1
    assert (run_dir / "bootstrap.summary.json").read_bytes() == bootstrap_before
    assert (run_dir / "step00/step00.summary.json").read_bytes() == epoch_zero_before
    assert (run_dir / "step01/step01.summary.json").is_file()

    resumed_final = json.loads(
        (run_dir / "final.summary.json").read_text(encoding="utf-8")
    )
    assert resumed_final["epochs_completed"] == 2
    committed = [
        event for event in _events(run_dir)
        if event["event_type"] == "barrier_committed"
    ]
    assert [event["payload"]["kind"] for event in committed] == [
        "bootstrap", "epoch", "epoch"
    ]
    assert len({event["idempotency_key"] for event in committed}) == 3
    assert list(run_dir.glob("config.resume*.json"))
    _assert_machine_readable_artifacts(run_dir)
    cli_plot_dir = run_dir / "plots_cli"
    assert viz_main([str(run_dir), "--all", "--out", str(cli_plot_dir)]) == 0
    assert {path.stem for path in cli_plot_dir.glob("*.png")} == {
        "record",
        "archive",
        "provenance",
        "allocation",
        "audits",
        "roles",
        "posterior",
        "failures",
        "resources",
    }


def test_keyboard_interrupt_drains_workers_and_preserves_live_status(
    tmp_path: Path,
    monkeypatch,
) -> None:
    config_path = _write_config(tmp_path / "toy.yaml", epochs=1)
    runs_root = tmp_path / "runs"
    config, resolved, metadata = _load_fresh(config_path)
    workers = _FakeWorkers(runs_root=runs_root, adapter=_adapter(config))

    def interrupt_epoch(self, layout, state, **_kwargs):
        live = {
            "schema_version": 1,
            "run_id": state.run_id,
            "run_directory": str(layout.run_dir),
            "epoch": state.epoch,
            "live_epoch": {
                "epoch": state.epoch,
                "provisional_best": {
                    "internal_reward": 3.5,
                    "raw_score": 0.285,
                    "committed": False,
                },
                "provisional_is_committed": False,
            },
        }
        layout.path("status.json").write_text(
            json.dumps(live, allow_nan=False, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        raise KeyboardInterrupt

    monkeypatch.setattr(EvolveEngine, "run_epoch", interrupt_epoch)
    assert _run_engine(
        config=config,
        resolved=resolved,
        metadata=metadata,
        runs_root=runs_root,
        workers=workers,
    ) == 130

    run_dir = workers._run_dir()
    status = json.loads((run_dir / "status.json").read_text(encoding="utf-8"))
    assert workers.shutdown_calls == 1
    assert status["live_epoch"]["provisional_best"]["raw_score"] == 0.285
    assert status["interruption"]["graceful_worker_shutdown_completed"] is True
    assert status["interruption"]["public_best_remains_barrier_confirmed"] is True
    interrupted = [
        event for event in _events(run_dir)
        if event["event_type"] == "run_interrupted"
    ]
    assert len(interrupted) == 1
    assert interrupted[0]["payload"]["provisional_best"]["internal_reward"] == 3.5


def test_keyboard_interrupt_during_bootstrap_is_recorded_without_traceback(
    tmp_path: Path,
    monkeypatch,
) -> None:
    config_path = _write_config(tmp_path / "toy.yaml", epochs=1)
    runs_root = tmp_path / "runs"
    config, resolved, metadata = _load_fresh(config_path)
    workers = _FakeWorkers(runs_root=runs_root, adapter=_adapter(config))

    def interrupt_bootstrap(self, state, **_kwargs):
        raise KeyboardInterrupt

    monkeypatch.setattr(EvolveEngine, "_seed_archive", interrupt_bootstrap)
    assert _run_engine(
        config=config,
        resolved=resolved,
        metadata=metadata,
        runs_root=runs_root,
        workers=workers,
    ) == 130

    run_dir = workers._run_dir()
    status = json.loads((run_dir / "status.json").read_text(encoding="utf-8"))
    assert workers.shutdown_calls == 0
    assert status["interruption"]["last_committed_epoch"] is None
    assert status["interruption"]["latest_completed_barrier_found"] is False
    assert [
        event for event in _events(run_dir)
        if event["event_type"] == "run_interrupted"
    ]


def test_partial_epoch_replays_plan_and_fails_closed_on_companion_corruption(
    tmp_path: Path,
) -> None:
    config_path = _write_config(tmp_path / "toy.yaml", epochs=1)
    runs_root = tmp_path / "runs"
    config, resolved, metadata = _load_fresh(config_path)
    crashing_workers = _FakeWorkers(
        runs_root=runs_root,
        adapter=_adapter(config),
        fail_on_call=2,
    )

    with pytest.raises(RuntimeError, match="injected fake-worker crash"):
        _run_engine(
            config=config,
            resolved=resolved,
            metadata=metadata,
            runs_root=runs_root,
            workers=crashing_workers,
        )
    assert crashing_workers.shutdown_calls == 1
    run_dir = crashing_workers._run_dir()
    plan_path = run_dir / "step00/allocation_plan.json"
    plan_before = plan_path.read_bytes()
    arrivals = list(run_dir.glob("step00/branches/*/steps/*.arrival.json"))
    assert len(arrivals) == 1
    arrival_before = arrivals[0].read_bytes()
    assert not (run_dir / "step00/step00.summary.json").exists()
    active_plot_dir = run_dir / "plots_active_fixture"
    assert viz_main([str(run_dir), "--all", "--out", str(active_plot_dir)]) == 0
    assert {path.stem for path in active_plot_dir.glob("*.png")} == {
        "record",
        "archive",
        "provenance",
        "allocation",
        "audits",
        "roles",
        "posterior",
        "failures",
        "resources",
    }

    resumed_config, resumed_resolved, resumed_metadata = _load_resume(
        run_dir, target_epochs=1
    )
    recovery_workers = _FakeWorkers(
        runs_root=runs_root, adapter=_adapter(resumed_config)
    )
    assert _run_engine(
        config=resumed_config,
        resolved=resumed_resolved,
        metadata=resumed_metadata,
        runs_root=runs_root,
        workers=recovery_workers,
    ) == 0
    assert plan_path.read_bytes() == plan_before
    assert arrivals[0].read_bytes() == arrival_before
    assert (run_dir / "step00/step00.summary.json").is_file()

    layout = open_existing_run_layout(run_dir, resume=True)
    assert _latest_completed_checkpoint(layout).name == "checkpoint_epoch001.json"
    companion = run_dir / "checkpoints/checkpoint_epoch001.pt"
    companion.write_bytes(companion.read_bytes() + b"corrupt")
    with pytest.raises(EngineError, match="training-state companion hash mismatch"):
        _latest_completed_checkpoint(layout)


def test_fake_engine_streams_three_concurrent_branch_workers(tmp_path: Path) -> None:
    config_path = _write_config(tmp_path / "toy.yaml", epochs=1)
    runs_root = tmp_path / "runs"
    config, resolved, metadata = _load_fresh(config_path)
    workers = _FakeWorkers(
        runs_root=runs_root,
        adapter=_adapter(config),
        concurrent=True,
    )

    assert _run_engine(
        config=config,
        resolved=resolved,
        metadata=metadata,
        runs_root=runs_root,
        workers=workers,
    ) == 0
    assert workers.max_active_branches >= 3
    assert workers.shutdown_calls == 1
    run_dir = workers._run_dir()
    final = json.loads(
        (run_dir / "final.summary.json").read_text(encoding="utf-8")
    )
    assert final["epochs_completed"] == 1
    branch_events = [
        event for event in _events(run_dir)
        if event["event_type"] == "branch_closed"
    ]
    assert branch_events
    assert len({event["idempotency_key"] for event in branch_events}) == len(
        branch_events
    )


def _minimal_completed_layout(root: Path):
    layout = create_fresh_run_layout(
        root,
        problem="evolve_toy",
        model_name="fixture/model",
        short_random_id="recovery",
    )
    checkpoint = {"schema_version": 1, "record_type": "fixture", "epoch": 0}
    checkpoint_path = layout.path("checkpoints/checkpoint_epoch000.json")
    checkpoint_path.write_text(
        json.dumps(checkpoint, allow_nan=False, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    summary = {
        "schema_version": 1,
        "committed_epoch": 0,
        "checkpoint": checkpoint_path.name,
        "checkpoint_hash": content_hash(checkpoint),
    }
    layout.path("bootstrap.summary.json").write_text(
        json.dumps(summary, allow_nan=False, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return layout, summary


def test_latest_retention_prunes_only_old_role_training_artifacts(
    tmp_path: Path,
) -> None:
    layout = create_fresh_run_layout(
        tmp_path,
        problem="evolve_toy",
        model_name="fixture/model",
        short_random_id="retention",
    )
    for role in ("scout", "mechanist", "challenger"):
        role_dir = layout.path(f"roles/{role}")
        old_adapter = role_dir / "adapter_epoch000"
        latest_adapter = role_dir / "adapter_epoch001"
        old_adapter.mkdir()
        latest_adapter.mkdir()
        (old_adapter / "adapter_model.safetensors").write_bytes(b"old" * 20)
        (old_adapter / "optimizer_state.pt").write_bytes(b"optimizer")
        (latest_adapter / "adapter_model.safetensors").write_bytes(b"latest")
        (latest_adapter / "optimizer_state.pt").write_bytes(b"optimizer")
        (role_dir / "optimizer_epoch000.pt").write_bytes(b"legacy")

    checkpoint = layout.path("checkpoints/checkpoint_epoch000.pt")
    checkpoint.write_bytes(b"required resume companion")
    evidence = layout.path("logs/verifiers/evidence.json")
    evidence.write_bytes(b"scientific evidence")

    _apply_role_artifact_retention(
        layout,
        keep_epoch=1,
        mode="all",
    )
    assert layout.path("roles/scout/adapter_epoch000").is_dir()

    _apply_role_artifact_retention(
        layout,
        keep_epoch=1,
        mode="latest",
    )
    # Reapplication after an interrupted/controller retry is idempotent.
    _apply_role_artifact_retention(
        layout,
        keep_epoch=1,
        mode="latest",
    )

    for role in ("scout", "mechanist", "challenger"):
        assert not layout.path(f"roles/{role}/adapter_epoch000").exists()
        assert not layout.path(f"roles/{role}/optimizer_epoch000.pt").exists()
        assert layout.path(f"roles/{role}/adapter_epoch001").is_dir()
    assert checkpoint.read_bytes() == b"required resume companion"
    assert evidence.read_bytes() == b"scientific evidence"
    plan = json.loads(
        layout.path("logs/retention_epoch001.plan.json").read_text(
            encoding="utf-8"
        )
    )
    result = json.loads(
        layout.path("logs/retention_epoch001.result.json").read_text(
            encoding="utf-8"
        )
    )
    assert plan["policy"] == result["policy"] == "latest"
    assert len(plan["targets"]) == len(result["removed"]) == 6
    assert result["resume_checkpoint_companions_preserved"] is True


@pytest.mark.parametrize(
    ("corruption", "message"),
    [
        ("malformed_marker", "invalid completed-barrier marker"),
        ("future_marker", "unsupported completion-marker schema"),
        ("unsafe_checkpoint", "unsafe checkpoint filename"),
        ("checkpoint_hash", "checkpoint content hash mismatch"),
        ("incomplete_companion", "incomplete training-state companion binding"),
        ("missing_companion", "invalid completed-barrier marker"),
        ("duplicate_epoch", "multiple completed-barrier markers claim the same epoch"),
    ],
)
def test_completed_barrier_corruption_matrix_fails_closed(
    tmp_path: Path, corruption: str, message: str
) -> None:
    layout, summary = _minimal_completed_layout(tmp_path / corruption)
    marker_path = layout.path("bootstrap.summary.json")

    if corruption == "malformed_marker":
        marker_path.write_text("{", encoding="utf-8")
    elif corruption == "future_marker":
        summary["schema_version"] = 2
        marker_path.write_text(json.dumps(summary), encoding="utf-8")
    elif corruption == "unsafe_checkpoint":
        summary["checkpoint"] = "../checkpoint_epoch000.json"
        marker_path.write_text(json.dumps(summary), encoding="utf-8")
    elif corruption == "checkpoint_hash":
        summary["checkpoint_hash"] = "0" * 64
        marker_path.write_text(json.dumps(summary), encoding="utf-8")
    elif corruption == "incomplete_companion":
        summary["training_state"] = "checkpoint_epoch000.pt"
        marker_path.write_text(json.dumps(summary), encoding="utf-8")
    elif corruption == "missing_companion":
        summary["training_state"] = "checkpoint_epoch000.pt"
        summary["training_state_hash"] = hashlib.sha256(b"missing").hexdigest()
        marker_path.write_text(json.dumps(summary), encoding="utf-8")
    elif corruption == "duplicate_epoch":
        step_dir = layout.path("step00")
        step_dir.mkdir()
        (step_dir / "step00.summary.json").write_text(
            json.dumps(summary), encoding="utf-8"
        )
    else:  # pragma: no cover - exhaustive parameter guard
        raise AssertionError(corruption)

    with pytest.raises(EngineError, match=message):
        _latest_completed_checkpoint(layout)
