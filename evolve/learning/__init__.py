"""Homogeneous learning groups, OrderGrad/MaxPO objectives, and the trainer."""

from .groups import GroupMember, LearningGroupError, build_learning_groups, context_id_for
from .objectives import (
    MAXPO_VERSION,
    ORDERGRAD_VERSION,
    ObjectiveError,
    advantages_for_objective,
    maxpo_advantages,
    objective_version,
    ordergrad_advantages,
    rank_order,
)
from .trainer import (
    GradientStepFn,
    GradientStepRequest,
    GradientStepResult,
    LearningUpdate,
    TrainerError,
    train_barrier,
    train_role_groups,
)

__all__ = [
    "MAXPO_VERSION",
    "ORDERGRAD_VERSION",
    "GradientStepFn",
    "GradientStepRequest",
    "GradientStepResult",
    "GroupMember",
    "LearningGroupError",
    "LearningUpdate",
    "ObjectiveError",
    "TrainerError",
    "advantages_for_objective",
    "build_learning_groups",
    "context_id_for",
    "maxpo_advantages",
    "objective_version",
    "ordergrad_advantages",
    "rank_order",
    "train_barrier",
    "train_role_groups",
]
