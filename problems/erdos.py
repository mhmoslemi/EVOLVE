"""
Erdos' Minimum Overlap Problem.

Two changes from the original:

  The prompt now specifies an output format. It previously ended with "Write
  code to optimize this construction." and said nothing about structure, so the
  model had no block to fill and no fence to reach and simply ran toward
  max_new_tokens, and extract_python_code (which looks for ```python fences)
  had nothing reliable to find.

  build_prompt takes `memory` and places the retrieved lessons between the
  parent state and the instruction, and adapts the instruction when they are
  present, rather than having the trainer staple the block onto the end.

  The compute budget is config-driven. The prompt used to hardcode
  "budget_s=1000", and the sandbox calls run() with NO arguments, so that
  default is what actually executes: every rollout may burn 1000 seconds of
  optimization. At 512 rollouts a step that is 4 to 70 hours depending on how
  many evaluations run in parallel. `budget_s` in the problem config now sets
  both the number in the prompt and the sandbox ceiling, so the two cannot
  drift apart.
"""

from __future__ import annotations
import hashlib
import inspect
import json
from numbers import Integral, Real
from typing import Any, List, Mapping, Optional, Tuple
import numpy as np
from problems.base import (
    Problem, ParentContext, ResourceRequirements, RewardResult,
    ScientificVerification, SeedState, render_state_context,
)


def verify_c5_solution(h_values: np.ndarray, c5_achieved: float, n_points: int):
    if not isinstance(h_values, np.ndarray):
        try:
            h_values = np.array(h_values, dtype=np.float64)
        except (ValueError, TypeError) as e:
            raise ValueError(f"Cannot convert h_values to numpy array: {e}")

    if len(h_values.shape) != 1:
        raise ValueError(f"h_values must be 1D array, got shape {h_values.shape}")

    if h_values.shape[0] != n_points:
        raise ValueError(f"Expected h shape ({n_points},), got {h_values.shape}")

    if not np.all(np.isfinite(h_values)):
        raise ValueError("h_values contain NaN or inf values")

    if np.any(h_values < 0) or np.any(h_values > 1):
        raise ValueError(f"h(x) is not in [0, 1]. Range: [{h_values.min()}, {h_values.max()}]")

    n = n_points
    target_sum = n / 2.0
    current_sum = np.sum(h_values)

    if current_sum != target_sum:
        # Invalid candidates can have an all-zero or numerically tiny sum. The
        # verifier rejects the resulting non-finite normalization below; keep
        # NumPy from also emitting a noisy RuntimeWarning for that candidate.
        with np.errstate(divide="ignore", over="ignore", invalid="ignore"):
            h_values = h_values * (target_sum / current_sum)
        if not np.all(np.isfinite(h_values)):
            raise ValueError("Normalization produced NaN or inf values")
        if np.any(h_values < 0) or np.any(h_values > 1):
            raise ValueError(f"After normalization, h(x) is not in [0, 1]. Range: [{h_values.min()}, {h_values.max()}]")

    dx = 2.0 / n_points

    j_values = 1.0 - h_values
    correlation = np.correlate(h_values, j_values, mode="full") * dx
    computed_c5 = np.max(correlation)

    if not np.isfinite(computed_c5):
        raise ValueError(f"Computed C5 is not finite: {computed_c5}")

    if not np.isclose(computed_c5, c5_achieved, atol=1e-4):
        raise ValueError(f"C5 mismatch: reported {c5_achieved:.6f}, computed {computed_c5:.6f}")

    return computed_c5


def evaluate_erdos_solution(h_values: np.ndarray, c5_bound: float, n_points: int) -> float:
    verify_c5_solution(h_values, c5_bound, n_points)
    return float(c5_bound)


def verify_erdos_solution(result) -> bool:
    try:
        h_values, c5_bound, n_points = result
        c5_bound = evaluate_erdos_solution(h_values, c5_bound, n_points)
        if c5_bound <= 0 or np.isnan(c5_bound) or np.isinf(c5_bound):
            return False
    except Exception:
        return False
    return True


_VERIFIER_SRC = (
    "import numpy as np\n\n"
    + inspect.getsource(verify_c5_solution) + "\n\n"
    + inspect.getsource(evaluate_erdos_solution) + "\n\n"
)


class ErdosMinOverlap(Problem):
    name = "erdos"
    entrypoint = "run"
    metric_name = "C\u2085 bound"
    maximize = False
    scientific_method_complete = True
    answer_schema_version = 1
    descriptor_function_version = "erdos_output_structure_v1"
    fingerprint_function_version = "erdos_reversal_structure_v1"
    # The h array is the solution and cannot be recovered by replay: the
    # optimizer is stochastic and budget-bounded, and a mid-run rollout's parent
    # array is not otherwise stored either. Save both.
    saves_construction = True

    def __init__(self, cfg: dict):
        super().__init__(cfg)
        if self.target is None:
            self.target = 0.3808
        # What the model is told it has, and what the sandbox actually allows.
        # The sandbox gets headroom so a program that respects its budget is
        # killed by its own clock rather than by the harness, which produces a
        # returned best-so-far instead of a lost rollout.
        self.budget_s = float(cfg.get("budget_s", 60.0))
        self.n_cpus = int(cfg.get("eval_cpus", 2))
        limit = cfg.get("scientific_max_points", 4096)
        if type(limit) is not int or limit < 1:
            raise ValueError("scientific_max_points must be a positive integer")
        # Verification uses an O(n^2) full correlation.  This limit applies
        # only to the saved-answer path and comfortably covers all registered
        # seeds (40--99 points).
        self.scientific_max_points = limit
        # Materialize the default in the copied problem cfg. The common
        # verifier content-addresses scientific_verifier_identity(), which
        # includes this resolved cfg, so implicit and explicit defaults share
        # one verifier ID while a changed bound cannot alias it.
        self.cfg["scientific_max_points"] = limit

    # ------------------------------------------------------------------
    def build_prompt(self, parent: ParentContext, memory: str = "") -> List[dict]:
        state_ctx = render_state_context(self.metric_name, self.target, parent,
                                         maximize=self.maximize)

        construction_section = ""
        if parent.construction is not None and len(parent.construction) > 0:
            construction_section = f"""
You may want to start your search from the current construction, which you can access through the `initial_h_values` global variable (n={len(parent.construction)} samples).
You are encouraged to explore solutions that use other starting points to prevent getting stuck in a local optimum.
"""

        memory_section = ""
        if memory and memory.strip():
            memory_section = f"""
## Lessons from earlier attempts at this problem

Extracted from programs already generated and evaluated in this same search.
Empirical findings, not part of the specification above, and they do not
override any constraint stated in it.

{memory.strip()}
"""

        if memory_section:
            code_section = '''Work through the lessons above before writing anything:
- Which bear on the construction you were given, and what would each change?
- Which do NOT apply here, and why? Say so explicitly. Some will be wrong or
  irrelevant for this state.
- Is anything they recommend already in the algorithm above and still not
  improving the bound? Then that avenue is spent and the gain is elsewhere.

Then reason about how to improve the construction. Aim for something different
from the algorithm above: a different algorithmic idea, different heuristics, a
different parameterization or sweep. A lesson gives you an idea; you choose the
implementation, and you should not copy any expression from one verbatim.
Unless you make a meaningful improvement, you will not be rewarded.'''
        elif parent.code and parent.code.strip():
            code_section = '''Reason about how you could further improve this construction.
Ideally, try to do something different than the above algorithm. Could be using different algorithmic ideas, adjusting your heuristics, adjusting / sweeping your hyperparemeters, etc. 
Unless you make a meaningful improvement, you will not be rewarded.'''
        else:
            code_section = '''Write code to optimize this construction.'''

        user = f'''You are an expert in harmonic analysis, numerical optimization, and mathematical discovery.
Your task is to find an improved upper bound for the Erdős minimum overlap problem constant C₅.

## Problem

Find a step function h: [0, 2] → [0, 1] that **minimizes** the overlap integral:

$$C_5 = \\max_k \\int h(x)(1 - h(x+k)) dx$$

**Constraints**:
1. h(x) ∈ [0, 1] for all x
2. ∫₀² h(x) dx = 1

**Discretization**: Represent h as n_points samples over [0, 2].
With dx = 2.0 / n_points:
- 0 ≤ h[i] ≤ 1 for all i
- sum(h) * dx = 1 (equivalently: sum(h) == n_points / 2 exactly)

The evaluation computes: C₅ = max(np.correlate(h, 1-h, mode="full") * dx)

Smaller sequences with less than 1k samples are preferred - they are faster to optimize and evaluate.

**Lower C₅ values are better** - they provide tighter upper bounds on the Erdős constant.

## Budget & Resources
- **Time budget**: {self.budget_s:.0f}s for your code to run
- **CPUs**: {self.n_cpus} available

## Rules
- Define `run(seed=42, budget_s={self.budget_s:.0f}, **kwargs)` that returns `(h_values, c5_bound, n_points)`
- It is called with NO arguments, so your default for `budget_s` is the one that
  runs. Respect it: track elapsed time and return your best solution before it
  expires, rather than being killed with nothing to show
- Use scipy, numpy, cvxpy[CBC,CVXOPT,GLOP,GLPK,GUROBI,MOSEK,PDLP,SCIP,XPRESS,ECOS], math
- Make all helper functions top level, no closures or lambdas
- No filesystem or network IO
- `evaluate_erdos_solution()` and `initial_h_values` (an initial construction, if available) are pre-imported
- Your function must complete within budget_s seconds and return the best solution found

**Lower is better**. Current record: C₅ ≤ 0.38092. Our goal is to find a construction that shows C₅ ≤ 0.38080.

{state_ctx}
{construction_section}{memory_section}
{code_section}

## Output format


Make sure to /think step by step, first give your strategy between <strategy> and </strategy> tags, then finally return the final program between ```python and ```.


- Exactly ONE ```python block, containing the complete program. It is extracted
  verbatim and executed as written.
- No prose, notes, explanation, or example usage after the closing fence.
- No second code block. No partial snippets earlier in the response.
- The block must define `run` at top level and be runnable on its own.
'''

# Make sure to /think step by step, first give your strategy between <strategy> and </strategy> tags (under 100 words / 3-4 sentences maximum), then finally return the final program between ```python and ```.

        return [{"role": "user", "content": user}]

    # ------------------------------------------------------------------
    def preprocess(self, code: str, parent: ParentContext) -> str:
        prelude = _VERIFIER_SRC
        if parent.construction is not None:
            prelude += f"initial_h_values = np.array({list(parent.construction)!r})\n\n"
        return prelude + "# ---- model code below ----\n" + code

    # ------------------------------------------------------------------
    def score(self, output: Any, stdout: str) -> RewardResult:
        res = RewardResult(reward=self.fail_score)
        if not (isinstance(output, (tuple, list)) and len(output) == 3):
            res.msg = "bad_return_shape"
            res.failure_kind = "code"
            return res
        if not verify_erdos_solution(output):
            res.msg = "Invalid solution."
            return res
        h_values, c5_bound, n_points = output
        c5_bound = evaluate_erdos_solution(h_values, c5_bound, n_points)
        res.valid = True
        res.raw_score = float(c5_bound)
        res.reward = float(1.0 / (1e-8 + c5_bound))
        res.construction = list(np.asarray(h_values).ravel())
        res.msg = f"C5 bound: {c5_bound}"
        return res

    # ------------------------------------------------------------------
    def seed_states(self) -> List[SeedState]:
        seeds: List[SeedState] = []
        for i in range(self.num_seed_states):
            rng = np.random.default_rng(self.seed + i)
            n_points = int(rng.integers(40, 100))
            construction = np.ones(n_points) * 0.5
            perturbation = rng.uniform(-0.4, 0.4, n_points)
            perturbation = perturbation - np.mean(perturbation)
            construction = construction + perturbation
            dx = 2.0 / n_points
            correlation = np.correlate(construction, 1 - construction, mode="full") * dx
            c5_bound = float(np.max(correlation))
            seeds.append(SeedState(
                code="",
                value=float(1.0 / (1e-8 + c5_bound)),
                raw_score=c5_bound,
                construction=list(construction),
            ))
        return seeds

    # ------------------------------------------------------------------
    # EVOLVE saved-answer hooks. The claimed bound is never trusted here.
    def _answer_values(self, candidate: Any) -> Tuple[np.ndarray, int]:
        if isinstance(candidate, Mapping):
            schema_version = candidate.get("schema_version")
            if (type(schema_version) is not int
                    or schema_version != self.answer_schema_version):
                raise ValueError("unsupported Erdos answer schema_version")
            if candidate.get("problem") != self.name:
                raise ValueError("Erdos answer problem identifier mismatch")
            h_value = candidate.get("h_values")
            n_value = candidate.get("n_points")
        elif isinstance(candidate, (tuple, list)) and len(candidate) == 3:
            # candidate[1] is the proposal's claimed C5. It is deliberately
            # ignored; common verification recomputes C5 from h_values.
            h_value, _claimed_bound, n_value = candidate
        else:
            raise ValueError(
                "Erdos answer must be a payload or (h_values, claimed_c5, n_points)"
            )
        if isinstance(n_value, bool) or not isinstance(n_value, Integral):
            raise ValueError("n_points must be an integer")
        n_points = int(n_value)
        if n_points < 1:
            raise ValueError("n_points must be positive")
        if n_points > self.scientific_max_points:
            raise ValueError(
                f"n_points exceeds scientific_max_points "
                f"({n_points} > {self.scientific_max_points})"
            )
        if not isinstance(h_value, (list, tuple, np.ndarray)):
            raise ValueError("h_values must be a finite numeric sequence")
        if len(h_value) > self.scientific_max_points:
            raise ValueError(
                f"h_values length exceeds scientific_max_points "
                f"({len(h_value)} > {self.scientific_max_points})"
            )
        if len(h_value) != n_points:
            raise ValueError(
                f"h_values must have length {n_points}, got {len(h_value)}"
            )
        converted = []
        for index, value in enumerate(h_value):
            if isinstance(value, (bool, np.bool_)) or not isinstance(value, Real):
                raise ValueError(f"h_values[{index}] must be a real number")
            try:
                number = float(value)
            except (TypeError, ValueError, OverflowError) as exc:
                raise ValueError(
                    f"h_values[{index}] cannot be represented as float64: {exc}"
                ) from exc
            if not np.isfinite(number):
                raise ValueError(f"h_values[{index}] must be finite")
            converted.append(number)
        try:
            h_values = np.asarray(converted, dtype=np.float64)
        except (TypeError, ValueError, OverflowError) as exc:
            raise ValueError(f"h_values are not numeric: {exc}") from exc
        if h_values.ndim != 1 or h_values.shape != (n_points,):
            raise ValueError(
                f"h_values must have shape ({n_points},), got {h_values.shape}"
            )
        if not np.all(np.isfinite(h_values)):
            raise ValueError("h_values contain NaN or infinity")
        return h_values, n_points

    @staticmethod
    def _effective_h(h_values: np.ndarray,
                     n_points: int) -> Tuple[np.ndarray, float]:
        if np.any(h_values < 0.0) or np.any(h_values > 1.0):
            raise ValueError("h_values must lie in [0, 1]")
        effective = np.asarray(h_values, dtype=np.float64).copy()
        target_sum = n_points / 2.0
        current_sum = float(np.sum(effective))
        if current_sum != target_sum:
            if current_sum == 0.0:
                raise ValueError("h_values cannot be normalized from zero sum")
            effective *= target_sum / current_sum
        if not np.all(np.isfinite(effective)):
            raise ValueError("normalization produced NaN or infinity")
        if np.any(effective < 0.0) or np.any(effective > 1.0):
            raise ValueError("normalized h_values must lie in [0, 1]")
        dx = 2.0 / n_points
        correlation = np.correlate(effective, 1.0 - effective, mode="full") * dx
        computed_c5 = float(np.max(correlation))
        if not np.isfinite(computed_c5) or computed_c5 <= 0.0:
            raise ValueError("computed C5 must be positive and finite")
        return effective, computed_c5

    def serialize_answer(self, candidate: Any, evidence: Any = None) -> Any:
        del evidence
        h_values, n_points = self._answer_values(candidate)
        return {
            "schema_version": self.answer_schema_version,
            "problem": self.name,
            "n_points": n_points,
            # Confirmation needs the full saved function. Legacy's optional
            # max_saved_construction truncation is not applied here.
            "h_values": [
                0.0 if float(value) == 0.0 else float(value)
                for value in h_values
            ],
        }

    def verify_answer_payload(
        self,
        payload: Any,
        policy: Optional[Mapping[str, Any]] = None,
    ) -> ScientificVerification:
        del policy
        try:
            answer = self.serialize_answer(payload)
            h_values, n_points = self._answer_values(answer)
            effective, computed_c5 = self._effective_h(h_values, n_points)
        except (TypeError, ValueError, OverflowError) as exc:
            return ScientificVerification(
                resolved=True,
                admitted=False,
                answer_payload=None,
                failure_kind="constraint",
                message=str(exc),
                flags={"method_complete": True, "payload_only": True},
            )
        reward = float(1.0 / (1e-8 + computed_c5))
        features = self._structure_features(effective)
        return ScientificVerification(
            resolved=True,
            admitted=True,
            answer_payload=answer,
            internal_reward=reward,
            raw_score=computed_c5,
            uncertainty=0.0,
            message="verified saved h_values and recomputed C5",
            scores={"computed_c5": computed_c5, **features["scores"]},
            flags={
                "method_complete": True,
                "payload_only": True,
                "deterministic": True,
                "normalized_for_verification": bool(
                    not np.array_equal(h_values, effective)
                ),
            },
        )

    @staticmethod
    def _structure_features(effective: np.ndarray) -> Mapping[str, Any]:
        n_points = len(effective)
        differences = np.abs(np.diff(effective))
        transition_density = (
            float(np.mean(differences > 0.10)) if len(differences) else 0.0
        )
        total_variation = float(np.sum(differences))
        binarity = float(np.mean((effective <= 0.05) | (effective >= 0.95)))
        symmetry_error = float(np.mean(np.abs(effective - effective[::-1])))
        correlation = np.correlate(effective, 1.0 - effective, mode="full")
        peak_index = int(np.argmax(correlation))
        peak_lag = peak_index - (n_points - 1)
        return {
            "peak_lag": peak_lag,
            "scores": {
                "binarity": binarity,
                "transition_density": transition_density,
                "total_variation": total_variation,
                "symmetry_error": symmetry_error,
                "absolute_peak_lag_fraction": abs(peak_lag) / n_points,
            },
        }

    @staticmethod
    def _thirds_bin(value: float) -> str:
        if value < 1.0 / 3.0:
            return "low"
        if value < 2.0 / 3.0:
            return "medium"
        return "high"

    def describe_scientific_state(self, candidate: Any,
                                  evidence: Any = None) -> Mapping[str, Any]:
        payload = evidence.get("answer_payload") if isinstance(evidence, Mapping) else None
        if payload is None and evidence is not None:
            payload = getattr(evidence, "answer_payload", None)
        answer = self.serialize_answer(payload if payload is not None else candidate)
        h_values, n_points = self._answer_values(answer)
        effective, _computed_c5 = self._effective_h(h_values, n_points)
        scores = self._structure_features(effective)["scores"]
        if n_points <= 64:
            resolution = "coarse"
        elif n_points <= 256:
            resolution = "medium"
        else:
            resolution = "fine"
        symmetry_error = float(scores["symmetry_error"])
        symmetry = (
            "symmetric" if symmetry_error < 0.05
            else ("mixed" if symmetry_error < 0.20 else "asymmetric")
        )
        return {
            "resolution_bin": resolution,
            "binarity_bin": self._thirds_bin(float(scores["binarity"])),
            "transition_bin": self._thirds_bin(
                float(scores["transition_density"])
            ),
            "symmetry_bin": symmetry,
        }

    def scientific_fingerprint(self, candidate: Any,
                               evidence: Any = None) -> str:
        payload = evidence.get("answer_payload") if isinstance(evidence, Mapping) else None
        if payload is None and evidence is not None:
            payload = getattr(evidence, "answer_payload", None)
        answer = self.serialize_answer(payload if payload is not None else candidate)
        h_values, n_points = self._answer_values(answer)
        effective, _computed_c5 = self._effective_h(h_values, n_points)
        forward = tuple(round(float(value), 10) for value in effective)
        reverse = tuple(reversed(forward))
        canonical = min(forward, reverse)
        features = self._structure_features(effective)
        structure = {
            "version": self.fingerprint_function_version,
            "n_points": n_points,
            "reversal_canonical_h": canonical,
            "descriptor": self.describe_scientific_state(answer),
            "absolute_peak_lag": abs(int(features["peak_lag"])),
        }
        encoded = json.dumps(structure, sort_keys=True,
                             separators=(",", ":")).encode("utf-8")
        return hashlib.sha256(encoded).hexdigest()

    def resource_requirements(self) -> ResourceRequirements:
        return ResourceRequirements(
            cpu_cores=max(1, int(self.n_cpus)),
            memory_mb=1024,
            timeout_s=float(self.cfg.get(
                "sandbox_timeout_s", max(1.0, self.budget_s + 20.0)
            )),
            gpu_count=0,
            exclusive_gpu=False,
            network_access=False,
            filesystem_policy="none",
            timeout_is_scientific=True,
        )
