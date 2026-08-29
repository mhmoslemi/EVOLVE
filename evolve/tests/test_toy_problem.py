import json
from pathlib import Path

import pytest

from evolve.config import load_evolve_config
from problems.base import (
    ParentContext,
    Problem,
    ResourceRequirements,
    RewardResult,
    SeedState,
    build_problem_prompt,
)
from problems.evolve_toy import EvolveToyProblem
from problems.registry import available_problems, get_problem


class _PromptCompatibilityProblem(Problem):
    """Scientific problem whose prompt hook does not consume causal memory."""

    def __init__(self):
        super().__init__({})
        self.messages = [{"role": "user", "content": "base prompt"}]

    def build_prompt(self, parent):
        del parent
        return self.messages

    def preprocess(self, code, parent):
        del parent
        return code

    def score(self, output, stdout):
        del stdout
        valid = isinstance(output, list) and len(output) == 2
        return RewardResult(
            reward=float(sum(output)) if valid else 0.0,
            raw_score=float(sum(output)) if valid else None,
            valid=valid,
            construction=output if valid else None,
            msg="ok" if valid else "bad",
        )

    def seed_states(self):
        return [SeedState()]

    def serialize_answer(self, candidate, evidence=None):
        del evidence
        return candidate

    def verify_answer_payload(self, payload, policy=None):
        del payload, policy
        raise NotImplementedError

    def describe_scientific_state(self, candidate, evidence=None):
        del candidate, evidence
        return {"fixture": True}

    def scientific_fingerprint(self, candidate, evidence=None):
        del candidate, evidence
        return "0" * 64

    def resource_requirements(self):
        return ResourceRequirements(timeout_s=1.0)


def test_prompt_adapter_preserves_old_signature_and_does_not_mutate_messages():
    problem = _PromptCompatibilityProblem()
    parent = ParentContext()

    assert build_problem_prompt(problem, parent) is problem.messages
    rendered = build_problem_prompt(problem, parent, memory="tested lesson")

    assert rendered is not problem.messages
    assert rendered[0] is not problem.messages[0]
    assert "tested lesson" in rendered[0]["content"]
    assert problem.messages == [{"role": "user", "content": "base prompt"}]


def test_resource_declarations_reject_ambiguous_or_impossible_values():
    with pytest.raises(ValueError, match="cpu_cores"):
        ResourceRequirements(cpu_cores=1.5)
    with pytest.raises(ValueError, match="exclusive_gpu"):
        ResourceRequirements(exclusive_gpu=True, gpu_count=0)
    with pytest.raises(ValueError, match="timeout_s"):
        ResourceRequirements(timeout_s=float("inf"))


def test_toy_payload_verification_recomputes_direction_and_reward():
    problem = EvolveToyProblem({})

    optimum = problem.verify_answer_payload([3, -2])
    other = problem.verify_answer_payload([0, 0])

    assert problem.scientific_method_complete is True
    assert problem.maximize is False
    assert optimum.admitted is True
    assert optimum.raw_score == 0.0
    assert optimum.internal_reward == 1.0
    assert optimum.uncertainty == 0.0
    assert optimum.flags["payload_only"] is True
    assert other.raw_score == 13.0
    assert other.internal_reward == pytest.approx(1.0 / 14.0)
    assert problem.record_key(optimum) > problem.record_key(other)
    assert problem.normalize_gain(1.0, 0.5) == 0.5
    assert problem.normalize_gain(0.25, 0.5) == 0.0


@pytest.mark.parametrize(
    "payload",
    [None, [1], [1, 2, 3], [3.0, -2], [True, 0], [9, 0], [0, -9]],
)
def test_toy_rejects_malformed_or_out_of_range_payloads(payload):
    problem = EvolveToyProblem({})
    verified = problem.verify_answer_payload(payload)

    assert verified.resolved is True
    assert verified.admitted is False
    assert verified.internal_reward is None
    assert verified.failure_kind == "constraint"


def test_toy_confirmation_uses_saved_payload():
    problem = EvolveToyProblem({})
    # Candidate is deliberately invalid. Confirmation must prefer the saved
    # answer in evidence and must not execute or serialize candidate source.
    confirmed = problem.confirm_record(
        candidate="def stochastic_proposal(): raise AssertionError",
        evidence={"answer_payload": [3, -2]},
    )

    assert confirmed.admitted is True
    assert confirmed.answer_payload == [3, -2]
    assert confirmed.internal_reward == 1.0

    with pytest.raises(ValueError, match="persisted answer_payload"):
        problem.confirm_record([3, -2], evidence={})


def test_toy_descriptor_has_multiple_cells_and_fingerprint_ignores_source():
    problem = EvolveToyProblem({})

    assert problem.describe_scientific_state([1, 1]) == {
        "quadrant": "north_east",
        "radial_band": "inner",
    }
    assert problem.describe_scientific_state([-4, 4])["quadrant"] == "north_west"
    assert problem.describe_scientific_state([-4, -4])["quadrant"] == "south_west"
    assert problem.describe_scientific_state([4, -4])["quadrant"] == "south_east"
    assert problem.describe_scientific_state([4, 4])["radial_band"] == "middle"
    assert problem.describe_scientific_state([8, 8])["radial_band"] == "outer"

    first = problem.scientific_fingerprint(
        candidate=None,
        evidence={"answer_payload": [4, -4], "source_text": "program A"},
    )
    second = problem.scientific_fingerprint(
        candidate=None,
        evidence={"answer_payload": [4, -4], "source_text": "program B"},
    )
    assert first == second
    assert len(first) == 64


def test_toy_seeds_are_deterministic_and_cover_every_quadrant():
    left = EvolveToyProblem({"num_seed_states": 8, "seed": 123}).seed_states()
    right = EvolveToyProblem({"num_seed_states": 8, "seed": 123}).seed_states()

    assert left == right
    assert len(left) == 8
    problem = EvolveToyProblem({})
    first_four_cells = {
        problem.describe_scientific_state(seed.construction)["quadrant"]
        for seed in left[:4]
    }
    assert first_four_cells == {
        "north_east", "north_west", "south_east", "south_west"
    }
    for seed in left:
        verified = problem.verify_answer_payload(seed.construction)
        assert verified.raw_score == seed.raw_score
        assert verified.internal_reward == seed.value


def test_toy_sandbox_compute_path_and_memory_prompt_are_operational():
    problem = EvolveToyProblem({})
    response = """```python
def run_toy():
    return [3, -2]
```"""
    result = problem.compute_reward(response, ParentContext(), timeout_s=2.0)
    prompt = build_problem_prompt(
        problem, ParentContext(construction=[-4, 4]), memory="prefer the southeast"
    )

    assert result.valid is True
    assert result.parsed is True
    assert result.ran is True
    assert result.raw_score == 0.0
    assert result.reward == 1.0
    assert result.construction == [3, -2]
    assert "prefer the southeast" in prompt[0]["content"]
    assert "initial_point" in prompt[0]["content"]


def test_toy_renderer_and_resource_declaration(tmp_path):
    problem = EvolveToyProblem({})
    rendered = problem.render_best(
        None, {"answer_payload": [3, -2]}, tmp_path / "best"
    )
    resources = problem.resource_requirements()

    assert [Path(item).name for item in rendered] == ["candidate.json", "answer.txt"]
    candidate = json.loads(Path(rendered[0]).read_text(encoding="utf-8"))
    assert candidate["point"] == [3, -2]
    assert candidate["raw_score"] == 0.0
    assert resources == ResourceRequirements(
        cpu_cores=1,
        memory_mb=64,
        timeout_s=2.0,
        gpu_count=0,
        exclusive_gpu=False,
        network_access=False,
        filesystem_policy="none",
        timeout_is_scientific=True,
    )


def test_registry_adds_toy_aliases_and_rejects_subtype_conflicts():
    assert "evolve_toy" in available_problems()
    assert isinstance(get_problem("toy", {}), EvolveToyProblem)
    assert get_problem("ac1", {}).problem_type == "ac1"

    with pytest.raises(ValueError, match="requires problem_type='ac1'"):
        get_problem("ac1", {"problem_type": "ac2"})
    with pytest.raises(ValueError, match="does not accept problem_type"):
        get_problem("evolve_toy", {"problem_type": "ac1"})


def test_committed_toy_config_is_cpu_only_and_strictly_validated():
    repository = Path(__file__).resolve().parents[2]
    config_path = repository / "configs" / "evolve_toy.yaml"
    config, resolved, _metadata = load_evolve_config(
        ["--config", str(config_path)], cwd=repository
    )

    assert config.problem == "evolve_toy"
    assert config.gpu_ids == ()
    assert config.num_gpus == 0
    assert config.deterministic is True
    assert config.evolve.budget.epochs == 2
    assert config.evolve.learning.group_k == 4
    assert resolved["gpu_ids"] == []
