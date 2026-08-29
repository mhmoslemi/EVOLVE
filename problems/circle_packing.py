"""
Circle packing.

  - entrypoint:  run_packing
  - validator:   validate_packing (byte-identical to the paper's)
  - reward:      sum of radii if valid else 0   (maximize)

The prompt is memory-aware. build_prompt takes an optional `memory` block and
places it BETWEEN the parent state and the instruction, per Fig. 1's ordering,
rather than having the trainer staple it onto the end. Two things change when
memory is present:

  - the fixed "Consider:" hint list is replaced. Those four hints are the same
    every step and compete with the retrieved lessons for the model's attention;
    with memory on, the analysis step is to consult the lessons instead.
  - the prompt asks the model to note where the lessons do NOT apply. That is a
    pressure valve: v1's failure was a lesson reaching 99% adoption, and a prompt
    that only ever asks "how do I use this" has no way to reject it.
"""

from __future__ import annotations
import hashlib
import inspect
import json
from numbers import Real
from typing import Any, List, Mapping, Optional, Tuple
import numpy as np

from problems.base import (
    Problem, ParentContext, ResourceRequirements, RewardResult,
    ScientificVerification, SeedState, render_state_context,
)


# ----------------------------------------------------------------------
# Validator (verbatim copy of the paper's / examples/circle_packing/env.py)
# ----------------------------------------------------------------------
def validate_packing(centers, radii):
    n = centers.shape[0]

    if np.isnan(centers).any() or np.isnan(radii).any():
        return False, "NaN values present"

    for i in range(n):
        if radii[i] < 0:
            return False, f"Circle {i} has negative radius {radii[i]}"

    for i in range(n):
        x, y = centers[i]
        r = radii[i]
        if (x - r < -1e-12 or x + r > 1 + 1e-12
                or y - r < -1e-12 or y + r > 1 + 1e-12):
            return False, f"Circle {i} at ({x},{y}) r={r} outside unit square"

    for i in range(n):
        for j in range(i + 1, n):
            dist = np.sqrt(np.sum((centers[i] - centers[j]) ** 2))
            if dist < radii[i] + radii[j] - 1e-12:
                return False, f"Circles {i} and {j} overlap"

    return True, "ok"


_VALIDATOR_SRC = inspect.getsource(validate_packing)


# ----------------------------------------------------------------------
# Prompt sections
# ----------------------------------------------------------------------
_ANALYSIS_NO_MEMORY = """Reason about how you could further improve this packing. Consider:
- Are circles placed optimally near boundaries and corners?
- Could a different arrangement (hexagonal, nested, hybrid) yield better results?
- Are there gaps that could be filled with repositioned or resized circles?
- Could optimization parameters or methods be improved?"""

_ANALYSIS_WITH_MEMORY = """## 1. Analysis and strategy

Work through the recorded lessons above before you write anything:
- Which of them bear on the program you were given, and what would each change?
- Which do NOT apply here, and why? Say so explicitly. The lessons are evidence
  from earlier attempts, not requirements, and some of them will be wrong or
  irrelevant for this state.
- Is anything the lessons recommend already present in the program above and
  still not working? If so, the lesson has been tried and the improvement lies
  somewhere it does not cover.

Then decide what to change. A lesson tells you an idea; you decide the
implementation. Do not copy any expression from a lesson verbatim, and do not let
a lesson choose your overall arrangement for you.

If none of the lessons is useful here, ignore them and reason from first
principles about the packing itself: boundary and corner occupancy, whether a
different arrangement family would do better, gaps that could absorb a
repositioned circle, and whether the optimization formulation itself is the
limit."""

_MEMORY_HEADER = """## Lessons from earlier attempts at this task

Extracted from programs already generated and evaluated in this same search.
They are empirical findings, not part of the specification above, and they do
not override any rule stated in it."""


class CirclePacking(Problem):
    name = "circle_packing"
    entrypoint = "run_packing"
    metric_name = "sum of radii"
    maximize = True
    scientific_method_complete = True
    answer_schema_version = 1
    descriptor_function_version = "circle_contact_radius_v1"
    fingerprint_function_version = "circle_contact_radius_v1"

    def __init__(self, cfg: dict):
        super().__init__(cfg)
        self.num_circles = int(cfg.get("num_circles", 26))
        if self.target is None:
            self.target = 2.636 if self.num_circles == 26 else 2.940

    # ------------------------------------------------------------------
    def build_prompt(self, parent: ParentContext, memory: str = "") -> List[dict]:
        state_ctx = render_state_context(self.metric_name, self.target, parent,
                                         maximize=self.maximize)
        n = self.num_circles

        memory_section = ""
        if memory and memory.strip():
            memory_section = f"\n{_MEMORY_HEADER}\n\n{memory.strip()}\n"
        analysis = _ANALYSIS_WITH_MEMORY if memory_section else _ANALYSIS_NO_MEMORY

        user = f"""You are an expert mathematician specializing in circle packing problems and computational geometry.

Your task is to pack {n} circles in a unit square [0,1]×[0,1] to maximize the sum of radii.

We will run the below validation function (read-only, do not modify this):
```python
{_VALIDATOR_SRC}
```

{state_ctx}
{memory_section}
{analysis}

Rules:
- You must define the run_packing function: def run_packing() -> tuple[np.ndarray, np.ndarray, float]
- Returns (centers, radii, sum_radii) where centers has shape ({n}, 2) and radii has shape ({n},).
- You can use scientific libraries like scipy, numpy, cvxpy, math.
- Centers must lie within [0,1]^2 and radii must be nonnegative.
- The pair (centers, radii) must satisfy non-overlap and boundary constraints.
- Make all helper functions top level and have no closures from function nesting. Don't use any lambda functions.
- No filesystem or network IO.
- Do not catch exceptions in order to return a degenerate packing. A program that
  returns all-zero or near-zero radii when something goes wrong scores the same as
  one that crashes, and it hides the error that would have told you what to fix.
  Let it fail loudly instead.
- You need to get really creative and think from first principles.

Make sure to /think step by step, first give your strategy between <strategy> and </strategy> tags, then finally return the final program between ```python and ```.
"""

        return [{"role": "user", "content": user}]

    # ------------------------------------------------------------------
    def preprocess(self, code: str, parent: ParentContext) -> str:
        prelude = (
            "import numpy as np\n"
            "import math\n"
            "try:\n"
            "    from scipy.optimize import minimize\n"
            "except ImportError:\n"
            "    minimize = None\n\n"
            + _VALIDATOR_SRC + "\n"
        )
        return prelude + "\n# ---- model code below ----\n" + code

    # ------------------------------------------------------------------
    def score(self, output: Any, stdout: str) -> RewardResult:
        res = RewardResult(reward=self.fail_score)
        if not (isinstance(output, tuple) and len(output) == 3):
            res.msg = "bad_return_shape"
            res.failure_kind = "code"
            return res
        centers, radii, _ = output
        try:
            centers = np.asarray(centers, dtype=float)
            radii = np.asarray(radii, dtype=float).ravel()
        except (ValueError, TypeError) as e:
            res.msg = f"bad_array: {e}"
            res.failure_kind = "code"
            return res

        if centers.ndim != 2 or centers.shape[1] != 2 or centers.shape[0] != self.num_circles:
            res.msg = f"bad_centers_shape: {centers.shape}"
            res.failure_kind = "code"
            return res
        if radii.shape != (self.num_circles,):
            res.msg = f"bad_radii_shape: {radii.shape}"
            res.failure_kind = "code"
            return res

        valid, msg = validate_packing(centers, radii)

        # A packing the validator accepts but whose radii are all (near) zero is
        # not a solution: it is a program that detected its own failure and
        # returned something harmless. Accepting it at reward 0 makes defensive
        # coding free, inflates the valid rate, and gives the memory extractor a
        # pile of "returned zeros" rollouts to learn from. Reject it instead.
        if valid:
            s = float(np.sum(radii))
            if s <= self.degenerate_threshold:
                res.valid = False
                res.msg = (f"degenerate_packing: sum of radii {s:.3e} <= "
                           f"{self.degenerate_threshold:.3e}")
                return res
            res.valid = True
            res.msg = msg
            res.reward = s
            res.raw_score = s
            return res

        res.valid = False
        res.msg = msg
        return res

    @property
    def degenerate_threshold(self) -> float:
        """
        Below this the packing is treated as invalid. Default is deliberately
        tiny: it should catch all-zero and numerically-collapsed returns without
        rejecting a genuinely poor but real packing. Override with
        `degenerate_threshold` in the problem config.
        """
        return float(self.cfg.get("degenerate_threshold", 1e-6))

    # ------------------------------------------------------------------
    def seed_states(self) -> List[SeedState]:
        return [SeedState(code="", value=0.0, raw_score=0.0)
                for _ in range(self.num_seed_states)]

    # ------------------------------------------------------------------
    # EVOLVE saved-answer hooks. These do not participate in legacy scoring.
    def _answer_arrays(self, candidate: Any) -> Tuple[np.ndarray, np.ndarray]:
        if isinstance(candidate, Mapping):
            schema_version = candidate.get("schema_version")
            if (type(schema_version) is not int
                    or schema_version != self.answer_schema_version):
                raise ValueError("unsupported circle answer schema_version")
            if candidate.get("problem") != self.name:
                raise ValueError("circle answer problem identifier mismatch")
            declared_count = candidate.get("num_circles")
            if (type(declared_count) is not int
                    or declared_count != self.num_circles):
                raise ValueError("circle answer num_circles mismatch")
            centers_value = candidate.get("centers")
            radii_value = candidate.get("radii")
        elif isinstance(candidate, (tuple, list)) and len(candidate) == 3:
            # The third value is an untrusted claimed sum. The scientific
            # payload intentionally ignores it and verification recomputes it.
            centers_value, radii_value, _claimed_sum = candidate
        else:
            raise ValueError(
                "circle answer must be a payload or (centers, radii, claimed_sum)"
            )
        if not isinstance(centers_value, (list, tuple, np.ndarray)):
            raise ValueError("centers must be a finite numeric sequence")
        if not isinstance(radii_value, (list, tuple, np.ndarray)):
            raise ValueError("radii must be a finite numeric sequence")
        if len(centers_value) != self.num_circles:
            raise ValueError(
                f"centers must have {self.num_circles} rows, "
                f"got {len(centers_value)}"
            )
        if len(radii_value) != self.num_circles:
            raise ValueError(
                f"radii must have length {self.num_circles}, "
                f"got {len(radii_value)}"
            )
        finite_centers = []
        finite_radii = []
        for index, center in enumerate(centers_value):
            if (not isinstance(center, (list, tuple, np.ndarray))
                    or len(center) != 2):
                raise ValueError(f"centers[{index}] must contain exactly x and y")
            row = []
            for coordinate, value in zip(("x", "y"), center):
                if (isinstance(value, (bool, np.bool_))
                        or not isinstance(value, Real)):
                    raise ValueError(
                        f"centers[{index}].{coordinate} must be a real number"
                    )
                try:
                    number = float(value)
                except (TypeError, ValueError, OverflowError) as exc:
                    raise ValueError(
                        f"centers[{index}].{coordinate} cannot be represented "
                        f"as float64: {exc}"
                    ) from exc
                if not np.isfinite(number):
                    raise ValueError(
                        f"centers[{index}].{coordinate} must be finite"
                    )
                row.append(number)
            finite_centers.append(row)
        for index, value in enumerate(radii_value):
            if (isinstance(value, (bool, np.bool_))
                    or not isinstance(value, Real)):
                raise ValueError(f"radii[{index}] must be a real number")
            try:
                number = float(value)
            except (TypeError, ValueError, OverflowError) as exc:
                raise ValueError(
                    f"radii[{index}] cannot be represented as float64: {exc}"
                ) from exc
            if not np.isfinite(number):
                raise ValueError(f"radii[{index}] must be finite")
            finite_radii.append(number)
        try:
            centers = np.asarray(finite_centers, dtype=np.float64)
            radii = np.asarray(finite_radii, dtype=np.float64)
        except (TypeError, ValueError, OverflowError) as exc:
            raise ValueError(f"circle answer arrays are not numeric: {exc}") from exc
        if centers.shape != (self.num_circles, 2):
            raise ValueError(
                f"centers must have shape ({self.num_circles}, 2), got {centers.shape}"
            )
        if radii.shape != (self.num_circles,):
            raise ValueError(
                f"radii must have shape ({self.num_circles},), got {radii.shape}"
            )
        if not np.all(np.isfinite(centers)) or not np.all(np.isfinite(radii)):
            raise ValueError("circle answer contains NaN or infinity")
        return centers, radii

    def serialize_answer(self, candidate: Any, evidence: Any = None) -> Any:
        del evidence
        centers, radii = self._answer_arrays(candidate)
        # Circle ordering has no scientific meaning. Sort complete triples so
        # permutations of the same packing have one canonical saved payload.
        def canonical_float(value: Any) -> float:
            number = float(value)
            return 0.0 if number == 0.0 else number

        circles = sorted(
            (
                canonical_float(center[0]),
                canonical_float(center[1]),
                canonical_float(radius),
            )
            for center, radius in zip(centers, radii)
        )
        return {
            "schema_version": self.answer_schema_version,
            "problem": self.name,
            "num_circles": self.num_circles,
            "centers": [[x, y] for x, y, _radius in circles],
            "radii": [radius for _x, _y, radius in circles],
        }

    def verify_answer_payload(
        self,
        payload: Any,
        policy: Optional[Mapping[str, Any]] = None,
    ) -> ScientificVerification:
        del policy
        try:
            answer = self.serialize_answer(payload)
            centers, radii = self._answer_arrays(answer)
            valid, message = validate_packing(centers, radii)
            total = float(np.sum(radii))
            if valid and total <= self.degenerate_threshold:
                valid = False
                message = (
                    f"degenerate_packing: sum of radii {total:.3e} <= "
                    f"{self.degenerate_threshold:.3e}"
                )
        except (TypeError, ValueError, OverflowError) as exc:
            return ScientificVerification(
                resolved=True,
                admitted=False,
                answer_payload=None,
                failure_kind="constraint",
                message=str(exc),
                flags={"method_complete": True, "payload_only": True},
            )
        if not valid:
            return ScientificVerification(
                resolved=True,
                admitted=False,
                answer_payload=answer,
                failure_kind="constraint",
                message=message,
                flags={"method_complete": True, "payload_only": True},
            )
        features = self._contact_radius_features(centers, radii)
        return ScientificVerification(
            resolved=True,
            admitted=True,
            answer_payload=answer,
            internal_reward=total,
            raw_score=total,
            uncertainty=0.0,
            message="verified saved circle packing",
            scores={"sum_radii": total, **features["scores"]},
            flags={
                "method_complete": True,
                "payload_only": True,
                "deterministic": True,
            },
        )

    def _contact_radius_features(self, centers: np.ndarray,
                                 radii: np.ndarray) -> Mapping[str, Any]:
        contact_tolerance = 1e-6
        boundary_masks = []
        for (x, y), radius in zip(centers, radii):
            mask = 0
            if abs(float(x - radius)) <= contact_tolerance:
                mask |= 1
            if abs(float((1.0 - x) - radius)) <= contact_tolerance:
                mask |= 2
            if abs(float(y - radius)) <= contact_tolerance:
                mask |= 4
            if abs(float((1.0 - y) - radius)) <= contact_tolerance:
                mask |= 8
            boundary_masks.append(mask)

        degrees = [0] * len(radii)
        gap_bins = [0, 0, 0, 0]
        contacts = 0
        for i in range(len(radii)):
            for j in range(i + 1, len(radii)):
                distance = float(np.linalg.norm(centers[i] - centers[j]))
                gap = max(0.0, distance - float(radii[i] + radii[j]))
                if gap <= contact_tolerance:
                    contacts += 1
                    degrees[i] += 1
                    degrees[j] += 1
                    gap_bins[0] += 1
                elif gap <= 0.02:
                    gap_bins[1] += 1
                elif gap <= 0.10:
                    gap_bins[2] += 1
                else:
                    gap_bins[3] += 1

        pair_count = len(radii) * (len(radii) - 1) // 2
        contact_density = contacts / pair_count if pair_count else 0.0
        boundary_fraction = (
            sum(mask != 0 for mask in boundary_masks) / len(radii)
            if len(radii) else 0.0
        )
        mean_radius = float(np.mean(radii)) if len(radii) else 0.0
        radius_cv = (
            float(np.std(radii) / mean_radius) if mean_radius > 0.0 else 0.0
        )
        return {
            "boundary_masks": boundary_masks,
            "degrees": degrees,
            "gap_bins": gap_bins,
            "scores": {
                "pair_contacts": contacts,
                "contact_density": contact_density,
                "boundary_fraction": boundary_fraction,
                "mean_radius": mean_radius,
                "radius_cv": radius_cv,
            },
        }

    @staticmethod
    def _fraction_bin(value: float) -> str:
        if value < 0.25:
            return "low"
        if value < 0.60:
            return "medium"
        return "high"

    def describe_scientific_state(self, candidate: Any,
                                  evidence: Any = None) -> Mapping[str, Any]:
        payload = evidence.get("answer_payload") if isinstance(evidence, Mapping) else None
        if payload is None and evidence is not None:
            payload = getattr(evidence, "answer_payload", None)
        answer = self.serialize_answer(payload if payload is not None else candidate)
        verified = self.verify_answer_payload(answer)
        if not verified.admitted:
            raise ValueError(f"cannot describe invalid packing: {verified.message}")
        centers, radii = self._answer_arrays(answer)
        features = self._contact_radius_features(centers, radii)["scores"]
        radius_cv = float(features["radius_cv"])
        if radius_cv < 0.10:
            dispersion = "uniform"
        elif radius_cv < 0.35:
            dispersion = "mixed"
        else:
            dispersion = "hierarchical"
        return {
            "boundary_contact_bin": self._fraction_bin(
                float(features["boundary_fraction"])
            ),
            "pair_contact_bin": self._fraction_bin(
                float(features["contact_density"])
            ),
            "radius_dispersion_bin": dispersion,
        }

    def scientific_fingerprint(self, candidate: Any,
                               evidence: Any = None) -> str:
        payload = evidence.get("answer_payload") if isinstance(evidence, Mapping) else None
        if payload is None and evidence is not None:
            payload = getattr(evidence, "answer_payload", None)
        answer = self.serialize_answer(payload if payload is not None else candidate)
        verified = self.verify_answer_payload(answer)
        if not verified.admitted:
            raise ValueError(f"cannot fingerprint invalid packing: {verified.message}")
        centers, radii = self._answer_arrays(answer)
        features = self._contact_radius_features(centers, radii)
        per_circle = sorted(
            (
                round(float(radius), 10),
                int(degree),
                int(mask),
            )
            for radius, degree, mask in zip(
                radii, features["degrees"], features["boundary_masks"]
            )
        )
        structure = {
            "version": self.fingerprint_function_version,
            "descriptor": self.describe_scientific_state(answer),
            "circle_structure": per_circle,
            "gap_histogram": features["gap_bins"],
        }
        encoded = json.dumps(structure, sort_keys=True,
                             separators=(",", ":")).encode("utf-8")
        return hashlib.sha256(encoded).hexdigest()

    def resource_requirements(self) -> ResourceRequirements:
        return ResourceRequirements(
            cpu_cores=1,
            memory_mb=512,
            timeout_s=float(self.cfg.get("sandbox_timeout_s", 60.0)),
            gpu_count=0,
            exclusive_gpu=False,
            network_access=False,
            filesystem_policy="none",
            timeout_is_scientific=True,
        )
