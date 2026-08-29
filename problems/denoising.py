"""
Single-Cell Analysis (scRNA-seq denoising).

  - entrypoint:  run_denoising
  - reward:      1 / mse   (Poisson is a HARD CONSTRAINT: rejected if poisson_norm < 0.97)
  - preprocess injects the bio imports + the exact evaluate_mse / evaluate_poisson /
    run_denoising_eval sources + a wrapper, then the model's magic_denoise.
  - get_question reproduced verbatim (SYSTEM_PROMPT with placeholders filled).

 !!!!!  REQUIRMENTS ---- (scanpy, anndata, scprep, graphtools, magic-impute, molecular-cross-validation, openproblems, pancreas dataset ---- REQUIRMENTS!!!!!

 """

from __future__ import annotations
import hashlib
import json
import math
from numbers import Integral, Real
from pathlib import Path
from typing import Any, List, Mapping, Optional, Tuple
import numpy as np
from problems.base import (
    Problem, ParentContext, ResourceRequirements, RewardResult,
    ScientificVerification, SeedState,
)


BASELINES = {
    "pancreas": {
        "baseline_mse": 0.304721,
        "baseline_poisson": 0.257575,
        "perfect_mse": 0.000000,
        "perfect_poisson": 0.031739,
    },
}


POISSON_NORM_MIN = 0.97
_DENOISING_DATASET = "pancreas"
_DENOISING_EVALUATOR_VERSION = "pancreas_holdout_mse_poisson_v1"
_MAX_RECIPROCAL_REWARD = 1.0e12


def verify_denoising(result) -> bool:
    if not isinstance(result, (list, tuple)) or len(result) < 2:
        return False
    mse, poisson = result[0], result[1]
    if not np.isfinite(mse) or not np.isfinite(poisson):
        return False
    baseline = BASELINES["pancreas"]
    if poisson < baseline["perfect_poisson"]:
        return False
    poisson_range = baseline["baseline_poisson"] - baseline["perfect_poisson"]
    poisson_norm = (baseline["baseline_poisson"] - poisson) / poisson_range if poisson_range > 0 else 0
    if poisson_norm < 0.97:
        return False
    return True


class Denoising(Problem):
    name = "denoising"
    entrypoint = "run_denoising"
    metric_name = "MSE"
    maximize = False   # minimize MSE; reward = 1/mse keeps higher-is-better
    # The trusted sandbox wrapper is the evaluator boundary for this benchmark.
    # EVOLVE persists its complete scalar result together with the frozen data
    # split and evaluator identities.  Confirmation validates that immutable
    # evaluator envelope; it never reruns a candidate's potentially stochastic
    # denoising code.
    scientific_method_complete = True
    answer_schema_version = 1
    descriptor_function_version = "denoising_metric_tradeoff_v1"
    fingerprint_function_version = "denoising_evaluator_behavior_v1"

    def __init__(self, cfg: dict):
        super().__init__(cfg)
        if self.target is None:
            self.target = 0.97  # poisson_norm constraint threshold (informational)
        self.eval_seed = int(cfg.get("eval_seed", 42))

    # ------------------------------------------------------------------
    def build_prompt(self, parent: ParentContext) -> List[dict]:
        from examples.denoising.prompt import SYSTEM_PROMPT
        from examples.denoising.utils import EVALUATE_MSE_FUNC, EVALUATE_POISSON_FUNC

        prompt = SYSTEM_PROMPT
        prompt = prompt.replace("<<<EVALUATE_MSE_FUNC>>>", EVALUATE_MSE_FUNC)
        prompt = prompt.replace("<<<EVALUATE_POISSON_FUNC>>>", EVALUATE_POISSON_FUNC)

        has_code = bool(parent.code and parent.code.strip())
        value_ctx = ""
        if parent.raw_score is not None:
            value_ctx = f"\nCurrent metrics (lower is better): MSE: {parent.raw_score:.6f}"

        if has_code:
            clean_code = parent.code.strip()
            if clean_code.startswith("```python"):
                clean_code = clean_code[len("```python"):].strip()
            if clean_code.startswith("```"):
                clean_code = clean_code[3:].strip()
            if clean_code.endswith("```"):
                clean_code = clean_code[:-3].strip()
            code_section = f"""
Here is the current implementation:
```python
{clean_code}
```

You are iteratively improving the denoising algorithm.{value_ctx}

Reason about how you could improve this approach.
"""
        else:
            code_section = f"""
{value_ctx}

Write code to implement a denoising algorithm.
"""

        user = f"""{prompt}
{code_section}
Write your improved `magic_denoise` function."""
        return [{"role": "user", "content": user}]

    # ------------------------------------------------------------------
    def preprocess(self, code: str, parent: ParentContext) -> str:
        import inspect
        from examples.denoising.utils import (
            evaluate_mse, evaluate_poisson, run_denoising_eval,
        )

        imports = f"""import numpy as np
import scipy
import scipy.sparse
from scipy import linalg
from scipy.spatial.distance import cdist, pdist, squareform
from scipy.sparse import csr_matrix, issparse
from sklearn.neighbors import NearestNeighbors
from sklearn.decomposition import PCA, TruncatedSVD
from sklearn.cluster import KMeans
import graphtools
import scprep
import anndata
import scanpy as sc
import sklearn.metrics
import math
import random
from molecular_cross_validation.mcv_sweep import poisson_nll_loss

_SEED = {self.eval_seed}
"""
        wrapper = """
def run_denoising():
    return run_denoising_eval(magic_denoise, seed=_SEED)
"""
        return (
            imports + "\n\n"
            + code + "\n\n"
            + inspect.getsource(evaluate_mse) + "\n\n"
            + inspect.getsource(evaluate_poisson) + "\n\n"
            + inspect.getsource(run_denoising_eval) + "\n\n"
            + wrapper
        )

    # ------------------------------------------------------------------
    def score(self, output: Any, stdout: str) -> RewardResult:
        res = RewardResult(reward=self.fail_score)
        if not isinstance(output, (list, tuple)) or len(output) < 2:
            res.msg = "bad_return_shape"
            res.failure_kind = "code"
            return res
        try:
            finite_metrics = np.isfinite(output[0]) and np.isfinite(output[1])
        except (TypeError, ValueError):
            finite_metrics = False
        if not finite_metrics:
            res.msg = "bad_return_values"
            res.failure_kind = "code"
            return res
        if not verify_denoising(output):
            res.msg = "Invalid solution."
            return res
        mse, poisson = output[0], output[1]
        current_mse = mse if mse is not None else float("inf")
        res.valid = True
        res.raw_score = float(current_mse)
        res.reward = float(1.0 / current_mse) if current_mse > 0 else self.fail_score
        # Preserve the controller-observed evaluator result. The common
        # verifier serializes this exact envelope; it never trusts metrics
        # parsed from model text.
        res.construction = [float(mse), float(poisson)]
        res.msg = f"mse={current_mse}, poisson={poisson}"
        return res

    # ------------------------------------------------------------------
    def seed_states(self) -> List[SeedState]:
        # Initial state mirrors create_initial_state: MAGIC baseline.
        try:
            from examples.denoising.utils import MAGIC_FUNC
            code = MAGIC_FUNC
        except Exception:
            code = ""  # bio stack absent; model will write from scratch
        baseline_mse = 0.2316  # value used in create_initial_state
        value = float(1.0 / baseline_mse)
        baseline_poisson = float(BASELINES[_DENOISING_DATASET]["baseline_poisson"])
        return [SeedState(
                    code=code,
                    value=value,
                    raw_score=baseline_mse,
                    construction=[baseline_mse, baseline_poisson],
                )
                for _ in range(self.num_seed_states)]

    # ------------------------------------------------------------------
    # EVOLVE saved-answer hooks.  The scientific state for this benchmark is
    # the trusted evaluator observation, not candidate source or a replay of a
    # stochastic proposal.  The controller must therefore serialize the raw
    # run_denoising_eval return value before discarding it.
    @staticmethod
    def _metric_value(value: Any, name: str) -> float:
        if isinstance(value, bool) or not isinstance(value, Real):
            raise ValueError(f"{name} must be a real number")
        metric = float(value)
        if not math.isfinite(metric):
            raise ValueError(f"{name} must be finite")
        if metric < 0.0:
            raise ValueError(f"{name} must be nonnegative")
        return metric

    def _answer_metrics(self, candidate: Any) -> Tuple[float, float]:
        if isinstance(candidate, Mapping):
            expected_keys = {
                "schema_version", "problem", "dataset", "eval_seed",
                "evaluator_version", "metrics",
            }
            actual_keys = set(candidate)
            if actual_keys != expected_keys:
                missing = sorted(str(key) for key in expected_keys - actual_keys)
                extra = sorted(str(key) for key in actual_keys - expected_keys)
                details = []
                if missing:
                    details.append("missing " + ", ".join(missing))
                if extra:
                    details.append("unexpected " + ", ".join(extra))
                raise ValueError(
                    "denoising evaluator envelope has " + "; ".join(details)
                )
            schema_version = candidate.get("schema_version")
            if (isinstance(schema_version, bool)
                    or not isinstance(schema_version, Integral)
                    or int(schema_version) != self.answer_schema_version):
                raise ValueError("unsupported denoising answer schema_version")
            if candidate.get("problem") != self.name:
                raise ValueError("denoising answer problem identifier mismatch")
            if candidate.get("dataset") != _DENOISING_DATASET:
                raise ValueError("denoising answer dataset identifier mismatch")
            eval_seed = candidate.get("eval_seed")
            if (isinstance(eval_seed, bool)
                    or not isinstance(eval_seed, Integral)
                    or int(eval_seed) != self.eval_seed):
                raise ValueError("denoising answer eval_seed mismatch")
            if candidate.get("evaluator_version") != _DENOISING_EVALUATOR_VERSION:
                raise ValueError("denoising answer evaluator_version mismatch")
            metrics = candidate.get("metrics")
            if not isinstance(metrics, Mapping) or set(metrics) != {"mse", "poisson"}:
                raise ValueError(
                    "denoising answer metrics must contain exactly mse and poisson"
                )
            mse_value = metrics.get("mse")
            poisson_value = metrics.get("poisson")
        elif isinstance(candidate, (list, tuple)) and len(candidate) >= 2:
            # The trusted evaluator currently returns exactly two metrics.  Any
            # later diagnostic tail is deliberately not part of scientific
            # identity until a new evaluator/schema version defines it.
            mse_value, poisson_value = candidate[0], candidate[1]
        else:
            raise ValueError(
                "denoising answer must be an evaluator envelope or "
                "(mse, poisson, ...) output"
            )
        mse = self._metric_value(mse_value, "mse")
        poisson = self._metric_value(poisson_value, "poisson")
        return mse, poisson

    def serialize_answer(self, candidate: Any, evidence: Any = None) -> Any:
        del evidence
        mse, poisson = self._answer_metrics(candidate)
        return {
            "schema_version": self.answer_schema_version,
            "problem": self.name,
            "dataset": _DENOISING_DATASET,
            "eval_seed": self.eval_seed,
            "evaluator_version": _DENOISING_EVALUATOR_VERSION,
            "metrics": {
                "mse": mse,
                "poisson": poisson,
            },
        }

    @staticmethod
    def _poisson_norm(poisson: float) -> float:
        baseline = BASELINES[_DENOISING_DATASET]
        denominator = (
            baseline["baseline_poisson"] - baseline["perfect_poisson"]
        )
        if denominator <= 0.0:
            raise ValueError("invalid versioned Poisson normalization baselines")
        return float((baseline["baseline_poisson"] - poisson) / denominator)

    def verify_answer_payload(
        self,
        payload: Any,
        policy: Optional[Mapping[str, Any]] = None,
    ) -> ScientificVerification:
        del policy
        try:
            answer = self.serialize_answer(payload)
            mse, poisson = self._answer_metrics(answer)
            poisson_norm = self._poisson_norm(poisson)
        except (TypeError, ValueError) as exc:
            return ScientificVerification(
                resolved=True,
                admitted=False,
                answer_payload=None,
                failure_kind="constraint",
                message=str(exc),
                flags={"method_complete": True, "payload_only": True},
            )

        baseline = BASELINES[_DENOISING_DATASET]
        if poisson < baseline["perfect_poisson"]:
            return ScientificVerification(
                resolved=True,
                admitted=False,
                answer_payload=answer,
                raw_score=mse,
                failure_kind="constraint",
                message=(
                    "poisson is below the versioned perfect-data baseline: "
                    f"{poisson} < {baseline['perfect_poisson']}"
                ),
                scores={"mse": mse, "poisson": poisson,
                        "poisson_norm": poisson_norm},
                flags={"method_complete": True, "payload_only": True},
            )
        if poisson_norm < POISSON_NORM_MIN:
            return ScientificVerification(
                resolved=True,
                admitted=False,
                answer_payload=answer,
                raw_score=mse,
                failure_kind="constraint",
                message=(
                    "poisson_norm does not meet the hard constraint: "
                    f"{poisson_norm} < {POISSON_NORM_MIN}"
                ),
                scores={"mse": mse, "poisson": poisson,
                        "poisson_norm": poisson_norm},
                flags={"method_complete": True, "payload_only": True},
            )

        reciprocal = math.inf if mse == 0.0 else 1.0 / mse
        internal_reward = float(min(reciprocal, _MAX_RECIPROCAL_REWARD))
        return ScientificVerification(
            resolved=True,
            admitted=True,
            answer_payload=answer,
            internal_reward=internal_reward,
            raw_score=mse,
            uncertainty=0.0,
            message="verified saved trusted denoising evaluator envelope",
            scores={
                "mse": mse,
                "poisson": poisson,
                "poisson_norm": poisson_norm,
            },
            flags={
                "method_complete": True,
                "payload_only": True,
                "deterministic": True,
                "trusted_evaluator_capture_required": True,
                "reward_clamped": reciprocal > _MAX_RECIPROCAL_REWARD,
            },
        )

    @staticmethod
    def _level(value: float, low: float, high: float) -> str:
        if value < low:
            return "low"
        if value < high:
            return "medium"
        return "high"

    def _metric_features(self, answer: Mapping[str, Any]) -> Mapping[str, float]:
        mse, poisson = self._answer_metrics(answer)
        baseline = BASELINES[_DENOISING_DATASET]
        mse_gain = (
            (baseline["baseline_mse"] - mse) / baseline["baseline_mse"]
        )
        poisson_norm = self._poisson_norm(poisson)
        return {
            "mse_gain": float(mse_gain),
            "poisson_norm": poisson_norm,
            "poisson_margin": float(poisson_norm - POISSON_NORM_MIN),
        }

    def describe_scientific_state(self, candidate: Any,
                                  evidence: Any = None) -> Mapping[str, Any]:
        payload = evidence.get("answer_payload") if isinstance(evidence, Mapping) else None
        if payload is None and evidence is not None:
            payload = getattr(evidence, "answer_payload", None)
        answer = self.serialize_answer(payload if payload is not None else candidate)
        verification = self.verify_answer_payload(answer)
        if not verification.admitted:
            raise ValueError(
                "cannot describe invalid denoising result: "
                + verification.message
            )
        features = self._metric_features(answer)
        difference = features["mse_gain"] - features["poisson_norm"]
        if difference > 0.15:
            tradeoff = "mse_leading"
        elif difference < -0.15:
            tradeoff = "poisson_leading"
        else:
            tradeoff = "balanced"
        return {
            "dataset": _DENOISING_DATASET,
            "mse_gain_bin": self._level(features["mse_gain"], 0.25, 0.75),
            "poisson_margin_bin": self._level(
                features["poisson_margin"], 0.005, 0.02
            ),
            "metric_tradeoff": tradeoff,
        }

    def scientific_fingerprint(self, candidate: Any,
                               evidence: Any = None) -> str:
        payload = evidence.get("answer_payload") if isinstance(evidence, Mapping) else None
        if payload is None and evidence is not None:
            payload = getattr(evidence, "answer_payload", None)
        answer = self.serialize_answer(payload if payload is not None else candidate)
        verification = self.verify_answer_payload(answer)
        if not verification.admitted:
            raise ValueError(
                "cannot fingerprint invalid denoising result: "
                + verification.message
            )
        mse, poisson = self._answer_metrics(answer)
        behavior = {
            "version": self.fingerprint_function_version,
            "dataset": _DENOISING_DATASET,
            "eval_seed": self.eval_seed,
            "evaluator_version": _DENOISING_EVALUATOR_VERSION,
            "metrics": {
                "mse": round(mse, 12),
                "poisson": round(poisson, 12),
            },
            "descriptor": self.describe_scientific_state(answer),
        }
        encoded = json.dumps(
            behavior, allow_nan=False, sort_keys=True, separators=(",", ":")
        ).encode("utf-8")
        return hashlib.sha256(encoded).hexdigest()

    def render_best(self, candidate: Any, evidence: Any,
                    output_dir: Any) -> List[str]:
        payload = evidence.get("answer_payload") if isinstance(evidence, Mapping) else None
        if payload is None and evidence is not None:
            payload = getattr(evidence, "answer_payload", None)
        answer = self.serialize_answer(payload if payload is not None else candidate)
        verified = self.verify_answer_payload(answer)
        if not verified.admitted:
            raise ValueError("cannot render invalid denoising answer")
        destination = Path(output_dir)
        destination.mkdir(parents=True, exist_ok=True)
        json_path = destination / "answer.json"
        text_path = destination / "answer.txt"
        json_path.write_text(
            json.dumps(answer, allow_nan=False, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        text_path.write_text(
            "\n".join([
                "Denoising evaluator result",
                f"dataset: {answer['dataset']}",
                f"eval_seed: {answer['eval_seed']}",
                f"evaluator_version: {answer['evaluator_version']}",
                f"mse: {answer['metrics']['mse']}",
                f"poisson: {answer['metrics']['poisson']}",
                f"poisson_norm: {verified.scores['poisson_norm']}",
                f"internal_reward: {verified.internal_reward}",
            ]) + "\n",
            encoding="utf-8",
        )
        return [str(json_path), str(text_path)]

    def resource_requirements(self) -> ResourceRequirements:
        return ResourceRequirements(
            cpu_cores=max(1, int(self.cfg.get("eval_cpus", 2))),
            memory_mb=max(1, int(self.cfg.get("eval_memory_mb", 8192))),
            timeout_s=float(self.cfg.get("sandbox_timeout_s", 530.0)),
            gpu_count=0,
            exclusive_gpu=False,
            network_access=False,
            filesystem_policy="read_only_dataset_and_temporary",
            timeout_is_scientific=True,
        )
