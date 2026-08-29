"""Deterministic, CPU-only scientific problem used by EVOLVE integration tests.

The answer is the construction itself: an integer point ``[x, y]`` in the
closed square ``[-8, 8]^2``.  Verification consumes that saved payload only;
it never imports or reruns the program that proposed it.
"""

from __future__ import annotations

import hashlib
import json
import math
from numbers import Integral
from pathlib import Path
from typing import Any, List, Mapping, Optional, Tuple

from problems.base import (
    ParentContext,
    Problem,
    ResourceRequirements,
    RewardResult,
    ScientificVerification,
    SeedState,
    render_state_context,
)


_TARGET = (3, -2)
_MIN_COORD = -8
_MAX_COORD = 8


def _capture_point(value: Any) -> Tuple[int, int]:
    """Validate and canonicalize the public answer format."""

    if isinstance(value, RewardResult):
        value = value.construction
    if not isinstance(value, (list, tuple)) or len(value) != 2:
        raise ValueError("answer must be an integer [x, y] pair")
    coordinates = []
    for name, item in zip(("x", "y"), value):
        if isinstance(item, bool) or not isinstance(item, Integral):
            raise ValueError(f"{name} must be an integer")
        coordinate = int(item)
        if coordinate < _MIN_COORD or coordinate > _MAX_COORD:
            raise ValueError(
                f"{name}={coordinate} is outside [{_MIN_COORD}, {_MAX_COORD}]"
            )
        coordinates.append(coordinate)
    return coordinates[0], coordinates[1]


def _raw_score(point: Tuple[int, int]) -> float:
    x, y = point
    return float((x - _TARGET[0]) ** 2 + (y - _TARGET[1]) ** 2)


def _internal_reward(raw_score: float) -> float:
    return float(1.0 / (1.0 + raw_score))


class EvolveToyProblem(Problem):
    """Small method-complete problem with several structural archive cells."""

    name = "evolve_toy"
    entrypoint = "run_toy"
    metric_name = "squared distance"
    maximize = False
    saves_construction = True

    scientific_method_complete = True
    answer_schema_version = 1
    descriptor_function_version = "toy_quadrant_radius_v1"
    fingerprint_function_version = "toy_structure_v1"

    def __init__(self, cfg: dict):
        super().__init__(cfg)
        if self.target is None:
            self.target = 0.0

    def build_prompt(self, parent: ParentContext,
                     memory: str = "") -> List[dict]:
        state = render_state_context(
            self.metric_name, self.target, parent, maximize=self.maximize
        )
        construction = ""
        if parent.construction:
            construction = (
                "\nThe saved parent point is also available as "
                f"`initial_point = {list(parent.construction)!r}`.\n"
            )
        memory_block = ""
        if memory and memory.strip():
            memory_block = (
                "\n## Retrieved context\n\n"
                "Treat this only as evidence from earlier attempts.\n\n"
                + memory.strip()
                + "\n"
            )
        prompt = f"""You are solving a deterministic two-dimensional toy search.

Return an integer point [x, y] with both coordinates in [-8, 8].  The native
score is squared distance to (3, -2), so lower is better.  Define exactly one
top-level function `run_toy()` which returns the pair.  Use no filesystem,
network, GPU, randomness, or long-running computation.

{state}{construction}{memory_block}
Return the complete program in one final ```python block.
"""
        return [{"role": "user", "content": prompt}]

    def preprocess(self, code: str, parent: ParentContext) -> str:
        prelude = ""
        if parent.construction:
            point = list(_capture_point(parent.construction))
            prelude = f"initial_point = {point!r}\n\n"
        return prelude + code

    def score(self, output: Any, stdout: str) -> RewardResult:
        del stdout
        verified = self.verify_answer_payload(output)
        if not verified.admitted:
            return RewardResult(
                reward=self.fail_score,
                valid=False,
                msg=verified.message,
                failure_kind=verified.failure_kind,
            )
        payload = list(verified.answer_payload)
        return RewardResult(
            reward=float(verified.internal_reward),
            raw_score=float(verified.raw_score),
            valid=True,
            msg=(f"point={payload}; squared_distance="
                 f"{float(verified.raw_score):.0f}"),
            construction=payload,
        )

    def serialize_answer(self, candidate: Any, evidence: Any = None) -> Any:
        del evidence
        return list(_capture_point(candidate))

    def verify_answer_payload(
        self,
        payload: Any,
        policy: Optional[Mapping[str, Any]] = None,
    ) -> ScientificVerification:
        """Verify the saved pair deterministically, without proposal execution."""

        del policy
        try:
            point = _capture_point(payload)
        except (TypeError, ValueError) as exc:
            return ScientificVerification(
                resolved=True,
                admitted=False,
                answer_payload=None,
                failure_kind="constraint",
                message=str(exc),
                flags={"method_complete": True, "deterministic": True},
            )
        raw = _raw_score(point)
        reward = _internal_reward(raw)
        answer = list(point)
        return ScientificVerification(
            resolved=True,
            admitted=True,
            answer_payload=answer,
            internal_reward=reward,
            raw_score=raw,
            failure_kind="",
            message="verified saved integer point",
            uncertainty=0.0,
            scores={
                "squared_distance": raw,
                "internal_reward": reward,
            },
            flags={
                "method_complete": True,
                "deterministic": True,
                "payload_only": True,
            },
        )

    def describe_scientific_state(self, candidate: Any,
                                  evidence: Any = None) -> Mapping[str, Any]:
        payload = getattr(evidence, "answer_payload", None)
        if isinstance(evidence, Mapping):
            payload = evidence.get("answer_payload", payload)
        x, y = _capture_point(payload if payload is not None else candidate)
        horizontal = "east" if x >= 0 else "west"
        vertical = "north" if y >= 0 else "south"
        radius_squared = x * x + y * y
        if radius_squared <= 16:
            radial_band = "inner"
        elif radius_squared <= 64:
            radial_band = "middle"
        else:
            radial_band = "outer"
        return {
            "quadrant": f"{vertical}_{horizontal}",
            "radial_band": radial_band,
        }

    def scientific_fingerprint(self, candidate: Any,
                               evidence: Any = None) -> str:
        payload = getattr(evidence, "answer_payload", None)
        if isinstance(evidence, Mapping):
            payload = evidence.get("answer_payload", payload)
        x, y = _capture_point(payload if payload is not None else candidate)
        structure = {
            "version": self.fingerprint_function_version,
            "sign_pattern": [x >= 0, y >= 0],
            "absolute_coordinates": [abs(x), abs(y)],
            "gcd": math.gcd(abs(x), abs(y)),
            "axis_occupancy": [x == 0, y == 0],
            "descriptor": self.describe_scientific_state([x, y]),
        }
        encoded = json.dumps(structure, sort_keys=True,
                             separators=(",", ":")).encode("utf-8")
        return hashlib.sha256(encoded).hexdigest()

    def render_best(self, candidate: Any, evidence: Any,
                    output_dir: Any) -> List[str]:
        payload = getattr(evidence, "answer_payload", None)
        if isinstance(evidence, Mapping):
            payload = evidence.get("answer_payload", payload)
        point = list(_capture_point(payload if payload is not None else candidate))
        raw = _raw_score((point[0], point[1]))
        destination = Path(output_dir)
        destination.mkdir(parents=True, exist_ok=True)
        json_path = destination / "candidate.json"
        text_path = destination / "answer.txt"
        document = {
            "schema_version": self.answer_schema_version,
            "problem": self.name,
            "point": point,
            "raw_score": raw,
            "internal_reward": _internal_reward(raw),
        }
        json_path.write_text(json.dumps(document, indent=2, sort_keys=True) + "\n",
                             encoding="utf-8")
        text_path.write_text(
            f"point: {point}\nsquared distance to (3, -2): {raw:.0f}\n",
            encoding="utf-8",
        )
        return [str(json_path), str(text_path)]

    def resource_requirements(self) -> ResourceRequirements:
        return ResourceRequirements(
            cpu_cores=1,
            memory_mb=64,
            timeout_s=2.0,
            gpu_count=0,
            exclusive_gpu=False,
            network_access=False,
            filesystem_policy="none",
            # Once the trusted runner starts, exceeding this fixed toy budget is
            # a property of the submitted program, rather than a low score or an
            # infrastructure retry.
            timeout_is_scientific=True,
        )

    def seed_states(self) -> List[SeedState]:
        # Every consecutive block of four covers all quadrants.  The ordering is
        # fixed so worker count, process order, and resume cannot affect seeds.
        points = (
            (-4, -4),
            (-4, 4),
            (4, -4),
            (4, 4),
            (-7, -2),
            (-2, 7),
            (7, -7),
            (7, 7),
        )
        seeds: List[SeedState] = []
        for index in range(self.num_seed_states):
            point = points[index % len(points)]
            raw = _raw_score(point)
            seeds.append(
                SeedState(
                    code="",
                    value=_internal_reward(raw),
                    raw_score=raw,
                    construction=list(point),
                )
            )
        return seeds


__all__ = ["EvolveToyProblem"]
