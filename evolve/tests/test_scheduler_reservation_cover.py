from evolve.ids import content_id
from evolve.scheduler import PlannedArm, _select_role_cover
from evolve.types import AllocationArm, Role


def _planned(role: Role, label: str, reservations):
    arm = AllocationArm(
        arm_id=content_id("arm", {"label": label}),
        cell_id=content_id("cell", {"label": label}),
        role=role,
        option_id=content_id("option", {"label": label}),
        harness_id=content_id("harness", {"label": "baseline"}),
        horizon=1,
        cost_class="small",
        expected_cost={"verifier_calls": 1.0},
        hard_cost={"verifier_calls": 1.0},
    )
    labels = tuple(reservations)
    return PlannedArm(
        arm=arm,
        reservation=labels[0] if labels else None,
        reservations=labels,
        posterior_level="global",
        expected_gain=0.0,
        uncertainty=1.0,
        marginal_gain=0.0,
        rng_seed=1,
    )


def test_learning_role_choice_covers_all_mandatory_reservations():
    candidates = (
        _planned(Role.SCOUT, "scout-empty", ("role", "empty_cell")),
        _planned(
            Role.SCOUT,
            "scout-exploration",
            ("role", "global_exploration"),
        ),
        _planned(Role.MECHANIST, "mechanist-empty", ("role", "empty_cell")),
        _planned(Role.CHALLENGER, "challenger-empty", ("role", "empty_cell")),
    )

    selected = _select_role_cover(
        candidates,
        roles=(Role.SCOUT, Role.MECHANIST, Role.CHALLENGER),
        learning_role=Role.SCOUT,
        group_k=4,
        production_capacity=6,
        resource_limits={"verifier_calls": 100.0},
        required_reservations={
            "empty_cell": 2,
            "global_exploration": 2,
        },
    )

    by_role = {item.arm.role: (item, replicas, labels) for item, replicas, labels in selected}
    scout, scout_replicas, scout_labels = by_role[Role.SCOUT]
    assert scout.arm.option_id == content_id(
        "option", {"label": "scout-exploration"}
    )
    assert scout_replicas == 4
    assert "learning_group" in scout_labels
    assert "global_exploration" in scout_labels
    assert sum(
        replicas
        for _, replicas, labels in selected
        if "empty_cell" in labels
    ) == 2

