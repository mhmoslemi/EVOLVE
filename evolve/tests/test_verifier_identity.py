from types import SimpleNamespace

from evolve.verifier import ProblemScientificAdapter
from problems.ac_inequalities import ACInequalities
from problems.circle_packing import CirclePacking


def test_problem_verifier_id_separates_subtypes_and_is_stable_for_equal_config():
    ac1_config = {"problem_type": "ac1", "budget_s": 1}
    ac2_config = {"problem_type": "ac2", "budget_s": 1}
    ac1_first = ProblemScientificAdapter(ACInequalities(dict(ac1_config)))
    ac1_second = ProblemScientificAdapter(ACInequalities(dict(ac1_config)))
    ac2 = ProblemScientificAdapter(ACInequalities(dict(ac2_config)))

    circle2_config = {"num_circles": 2, "sandbox_timeout_s": 7}
    circle26_config = {"num_circles": 26, "sandbox_timeout_s": 7}
    circle2_first = ProblemScientificAdapter(CirclePacking(dict(circle2_config)))
    circle2_second = ProblemScientificAdapter(CirclePacking(dict(circle2_config)))
    circle26 = ProblemScientificAdapter(CirclePacking(dict(circle26_config)))

    assert ac1_first.verifier_id == ac1_second.verifier_id
    assert ac1_first.verifier_id != ac2.verifier_id
    assert circle2_first.verifier_id == circle2_second.verifier_id
    assert circle2_first.verifier_id != circle26.verifier_id

    assert ac1_first.problem_identity["problem_config"]["problem_type"] == "ac1"
    assert ac2.problem_identity["problem_config"]["problem_type"] == "ac2"
    assert circle2_first.problem_identity["problem_config"]["num_circles"] == 2
    assert circle26.problem_identity["problem_config"]["num_circles"] == 26


class _IdentityDuck:
    name = "identity_duck"
    answer_schema_version = 1

    def __init__(
        self,
        *,
        config=None,
        descriptor="descriptor_v1",
        fingerprint="fingerprint_v1",
        complete=True,
        timeout=3.0,
    ):
        self.cfg = dict(config or {"variant": "a"})
        self.descriptor_function_version = descriptor
        self.fingerprint_function_version = fingerprint
        self.scientific_method_complete = complete
        self.timeout = timeout

    def resource_requirements(self):
        return SimpleNamespace(
            cpu_cores=1,
            memory_mb=64,
            timeout_s=self.timeout,
            gpu_count=0,
            exclusive_gpu=False,
            network_access=False,
            filesystem_policy="none",
            timeout_is_scientific=False,
        )

    def verify_answer_payload(self, payload, policy):
        raise AssertionError("identity construction must not verify payloads")

    def describe_scientific_state(self, payload, decision):
        raise AssertionError("identity construction must not describe payloads")

    def scientific_fingerprint(self, payload, decision):
        raise AssertionError("identity construction must not fingerprint payloads")


def test_verifier_id_covers_versions_completeness_resources_and_duck_config():
    baseline = ProblemScientificAdapter(_IdentityDuck())
    equal = ProblemScientificAdapter(_IdentityDuck())
    variants = [
        ProblemScientificAdapter(_IdentityDuck(config={"variant": "b"})),
        ProblemScientificAdapter(_IdentityDuck(descriptor="descriptor_v2")),
        ProblemScientificAdapter(_IdentityDuck(fingerprint="fingerprint_v2")),
        ProblemScientificAdapter(_IdentityDuck(complete=False)),
        ProblemScientificAdapter(_IdentityDuck(timeout=4.0)),
    ]

    assert baseline.verifier_id == equal.verifier_id
    assert len({baseline.verifier_id, *(item.verifier_id for item in variants)}) == 6
    assert baseline.verifier_identity["descriptor_function_version"] == "descriptor_v1"
    assert baseline.verifier_identity["fingerprint_function_version"] == "fingerprint_v1"
    assert baseline.verifier_identity["scientific_method_complete"] is True
    assert baseline.verifier_identity["resource_requirements"]["timeout_s"] == 3.0

