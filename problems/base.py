"""Typed scientific-problem contract for EVOLVE.

Every problem exposes a higher-is-better internal reward while preserving its
native ``raw_score``. Admission and record changes are controlled only by
deterministic verification of a persisted answer payload.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
import inspect
import json
import math
from pathlib import Path
from typing import Any, List, Mapping, Optional

from evolve.verifier.parsing import extract_python_code
from evolve.verifier.sandbox import run_code


# ----------------------------------------------------------------------
# Data carried between the engine and the problems
# ----------------------------------------------------------------------
@dataclass
class SeedState:
    """One initial archive entry."""
    code: str = ""
    value: float = 0.0                 # internal reward (higher is better)
    raw_score: Optional[float] = None  # true metric, shown in the prompt
    construction: Optional[list] = None  # injected global (height_sequence_1 / initial_h_values)


@dataclass
class ParentContext:
    """Everything a prompt/preprocess needs from the parent state."""
    code: str = ""
    value: float = 0.0
    raw_score: Optional[float] = None
    construction: Optional[list] = None


@dataclass
class RewardResult:
    reward: float = 0.0
    raw_score: Optional[float] = None
    valid: bool = False
    parsed: bool = False
    ran: bool = False
    msg: str = ""
    stdout: str = ""
    code: str = ""
    construction: Optional[list] = None  
    # Problem-local failure stage. The independent verifier later classifies it
    # into scientific, code, timeout, or infrastructure evidence.
    failure_kind: str = ""


@dataclass(frozen=True)
class ScientificVerification:
    """Problem-local result of verifying an already captured answer payload.

    This intentionally contains no run, branch, policy, or harness identifiers.
    The EVOLVE verifier service adds those fields when it constructs the
    immutable EvidencePacket. Keeping this record independent of the engine
    prevents problem code from assigning archive or scheduler authority.
    """

    resolved: bool
    admitted: bool
    answer_payload: Any = None
    internal_reward: Optional[float] = None
    raw_score: Any = None
    failure_kind: str = ""
    message: str = ""
    uncertainty: Optional[float] = None
    scores: Mapping[str, Any] = field(default_factory=dict)
    flags: Mapping[str, Any] = field(default_factory=dict)
    diagnostics: Mapping[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class ResourceRequirements:
    """Declared verifier resources and timeout semantics for one problem."""

    cpu_cores: int = 1
    memory_mb: int = 512
    timeout_s: float = 30.0
    gpu_count: int = 0
    exclusive_gpu: bool = False
    network_access: bool = False
    filesystem_policy: str = "temporary_only"
    timeout_is_scientific: bool = False

    def __post_init__(self) -> None:
        if (not isinstance(self.cpu_cores, int)
                or isinstance(self.cpu_cores, bool) or self.cpu_cores < 1):
            raise ValueError("cpu_cores must be a positive integer")
        if (not isinstance(self.memory_mb, int)
                or isinstance(self.memory_mb, bool) or self.memory_mb < 1):
            raise ValueError("memory_mb must be a positive integer")
        if (isinstance(self.timeout_s, bool)
                or not isinstance(self.timeout_s, (int, float))
                or not math.isfinite(float(self.timeout_s))
                or self.timeout_s <= 0):
            raise ValueError("timeout_s must be positive and finite")
        if (not isinstance(self.gpu_count, int)
                or isinstance(self.gpu_count, bool) or self.gpu_count < 0):
            raise ValueError("gpu_count must be a non-negative integer")
        for field_name in (
            "exclusive_gpu", "network_access", "timeout_is_scientific"
        ):
            if not isinstance(getattr(self, field_name), bool):
                raise ValueError(f"{field_name} must be boolean")
        if self.exclusive_gpu and self.gpu_count < 1:
            raise ValueError("exclusive_gpu requires at least one GPU")
        if (not isinstance(self.filesystem_policy, str)
                or not self.filesystem_policy.strip()):
            raise ValueError("filesystem_policy must be non-empty")

    def to_dict(self) -> Mapping[str, Any]:
        """Return the complete JSON identity of the verifier resource policy."""

        return {
            "cpu_cores": self.cpu_cores,
            "memory_mb": self.memory_mb,
            "timeout_s": float(self.timeout_s),
            "gpu_count": self.gpu_count,
            "exclusive_gpu": self.exclusive_gpu,
            "network_access": self.network_access,
            "filesystem_policy": self.filesystem_policy,
            "timeout_is_scientific": self.timeout_is_scientific,
        }


# ----------------------------------------------------------------------
# Prompt helper
# ----------------------------------------------------------------------
def render_state_context(metric_name: str, target, parent: ParentContext,
                         maximize: bool = True) -> str:
    direction = "higher is better" if maximize else "lower is better"
    if parent.code and parent.code.strip():
        shown = parent.raw_score if parent.raw_score is not None else parent.value
        return (
            f"Target {metric_name}: {target} ({direction}).\n"
            f"Your previous program achieved {metric_name} = {shown:.6f}.\n"
            f"Here is the previous program:\n"
            f"```python\n{parent.code}\n```\n"
        )
    return (
        f"Target {metric_name}: {target} ({direction}).\n"
        f"No previous program. Write one from scratch.\n"
    )


def build_problem_prompt(problem: "Problem", parent: ParentContext,
                         memory: str = "") -> List[dict]:
    """Normalize problem prompts and append retrieved causal context safely."""

    method = problem.build_prompt
    parameters = inspect.signature(method).parameters.values()
    accepts_memory = any(
        item.name == "memory" or item.kind == inspect.Parameter.VAR_KEYWORD
        for item in parameters
    )
    if accepts_memory:
        return method(parent, memory=memory)

    messages = method(parent)
    block = str(memory or "").strip()
    if not block:
        return messages
    copied = [dict(message) for message in messages]
    suffix = (
        "\n\n## Retrieved context\n\n"
        "This context is evidence from earlier attempts, not part of the "
        "problem specification.\n\n" + block
    )
    for index in range(len(copied) - 1, -1, -1):
        if copied[index].get("role") == "user":
            copied[index]["content"] = str(copied[index].get("content", "")) + suffix
            break
    else:
        copied.append({"role": "user", "content": suffix.lstrip()})
    return copied


def _json_payload(value: Any) -> Any:
    """Return a conservative JSON-safe representation or raise ValueError."""

    if hasattr(value, "tolist"):
        value = value.tolist()
    elif hasattr(value, "item"):
        value = value.item()
    if isinstance(value, Mapping):
        if any(not isinstance(key, str) for key in value):
            raise ValueError("answer payload mapping keys must be strings")
        value = {key: _json_payload(item) for key, item in value.items()}
    elif isinstance(value, (list, tuple)):
        value = [_json_payload(item) for item in value]
    try:
        # Round-tripping also rejects custom containers and normalizes tuples.
        return json.loads(json.dumps(value, allow_nan=False, sort_keys=True))
    except (TypeError, ValueError) as exc:
        raise ValueError(f"answer payload is not finite JSON: {exc}") from exc


def _verification_field(evidence: Any, name: str, default: Any = None) -> Any:
    if isinstance(evidence, Mapping):
        return evidence.get(name, default)
    return getattr(evidence, name, default)


# ----------------------------------------------------------------------
# Problem ABC
# ----------------------------------------------------------------------
class Problem(ABC):
    name: str = "base"
    entrypoint: str = "run"         
    metric_name: str = "score"
    maximize: bool = True
    # Concrete EVOLVE problems must opt in and implement every scientific hook.
    scientific_method_complete: bool = False
    answer_schema_version: int = 1
    descriptor_function_version: str = "unimplemented"
    fingerprint_function_version: str = "unimplemented"

    # Whether RewardResult.construction is the actual solution object and
    # therefore worth writing into every rollout's meta. True only where the
    # construction cannot be recovered by re-running the program: Erdos returns
    # an h array produced by a stochastic, wall-clock-bounded optimizer, so a
    # replay does not reproduce it. Circle packing and gpu_mode carry no
    # construction at all, and their programs are the artifact.
    saves_construction: bool = False

    def __init__(self, cfg: dict):
        self.cfg = dict(cfg or {})
        self.target = self.cfg.get("target")
        self.fail_score = float(self.cfg.get("fail_score", 0.0))
        self.num_seed_states = int(self.cfg.get("num_seed_states", 8))
        self.seed = int(self.cfg.get("seed", 42))

    # ---- prompt / sandbox program / scoring (subclasses implement) ----
    @abstractmethod
    def build_prompt(self, parent: ParentContext, memory: str = "") -> List[dict]:
        """
        Build the chat messages for one parent.

        `memory` is a pre-rendered block of retrieved lessons, or "" when memory
        is off or nothing was selected. A problem that accepts it should place it
        between the parent state and the instruction, and adapt the instruction
        to it; a problem that ignores it still works, and the trainer falls back
        to appending the block itself.
        """
        ...

    @abstractmethod
    def preprocess(self, code: str, parent: ParentContext) -> str:
        """Return the full program to execute (prelude + verifier + construction + code)."""
        ...

    @abstractmethod
    def score(self, output: Any, stdout: str) -> RewardResult:
        """Validate the sandbox return value and turn it into a RewardResult."""
        ...

    @abstractmethod
    def seed_states(self) -> List[SeedState]:
        ...

    # ---- additive EVOLVE scientific-answer API ----------------------
    def scientific_verifier_identity(self) -> Mapping[str, Any]:
        """Return every problem-owned input that defines verifier behavior.

        The adapter adds its class and protocol versions around this payload.
        Including the resolved problem configuration and resource declaration
        prevents subtypes of one class from accidentally sharing an identity.
        A specialized problem should override this when verifier behavior is
        derived from state not represented by ``cfg`` or the effective fields.
        """

        resources = self.resource_requirements()
        resource_identity = (
            resources.to_dict()
            if callable(getattr(resources, "to_dict", None))
            else {
                "cpu_cores": resources.cpu_cores,
                "memory_mb": resources.memory_mb,
                "timeout_s": float(resources.timeout_s),
                "gpu_count": resources.gpu_count,
                "exclusive_gpu": resources.exclusive_gpu,
                "network_access": resources.network_access,
                "filesystem_policy": resources.filesystem_policy,
                "timeout_is_scientific": resources.timeout_is_scientific,
            }
        )
        return {
            "problem_config": _json_payload(self.cfg),
            "effective_problem": {
                "name": self.name,
                "entrypoint": self.entrypoint,
                "metric_name": self.metric_name,
                "maximize": self.maximize,
                "target": _json_payload(self.target),
                "fail_score": self.fail_score,
                "num_seed_states": self.num_seed_states,
                "seed": self.seed,
            },
            "answer_schema_version": self.answer_schema_version,
            "descriptor_function_version": self.descriptor_function_version,
            "fingerprint_function_version": self.fingerprint_function_version,
            "scientific_method_complete": self.scientific_method_complete,
            "resource_requirements": resource_identity,
        }

    @abstractmethod
    def serialize_answer(self, candidate: Any, evidence: Any = None) -> Any:
        """Capture the complete finite JSON answer used for later verification."""
        ...

    @abstractmethod
    def verify_answer_payload(self, payload: Any,
                              policy: Optional[Mapping[str, Any]] = None
                              ) -> ScientificVerification:
        """Verify a saved payload without rerunning stochastic proposal code."""
        ...

    @abstractmethod
    def describe_scientific_state(self, candidate: Any,
                                  evidence: Any = None) -> Mapping[str, Any]:
        """Return source-free scientific descriptor dimensions."""
        ...

    @abstractmethod
    def scientific_fingerprint(self, candidate: Any,
                               evidence: Any = None) -> str:
        """Return a stable verified-behavior fingerprint, never source identity."""
        ...

    def record_key(self, evidence: Any) -> float:
        """Return the higher-is-better internal reward from verified evidence."""

        value = _verification_field(evidence, "internal_reward")
        if value is None:
            value = _verification_field(evidence, "reward")
        if isinstance(value, bool) or not isinstance(value, (int, float)):
            raise ValueError("verified evidence has no numeric internal reward")
        value = float(value)
        if not math.isfinite(value):
            raise ValueError("verified evidence internal reward must be finite")
        return value

    def confirm_record(self, candidate: Any, evidence: Any = None,
                       policy: Optional[Mapping[str, Any]] = None
                       ) -> ScientificVerification:
        """Confirm by verifying the captured payload, never by rerunning code."""

        payload = _verification_field(evidence, "answer_payload")
        if payload is None:
            if self.scientific_method_complete:
                raise ValueError(
                    "production record confirmation requires a persisted "
                    "answer_payload in evidence"
                )
            payload = self.serialize_answer(candidate, evidence)
        return self.verify_answer_payload(payload, policy)

    def normalize_gain(self, new_reward: float, threshold: float) -> float:
        new_value = float(new_reward)
        old_value = float(threshold)
        if not math.isfinite(new_value) or not math.isfinite(old_value):
            raise ValueError("gain inputs must be finite")
        return max(0.0, new_value - old_value)

    def render_best(self, candidate: Any, evidence: Any,
                    output_dir: Any) -> List[str]:
        """Portable text fallback; specialized problems may add richer files."""

        payload = _verification_field(evidence, "answer_payload")
        if payload is None:
            payload = self.serialize_answer(candidate, evidence)
        destination = Path(output_dir)
        destination.mkdir(parents=True, exist_ok=True)
        path = destination / "answer.txt"
        path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n",
                        encoding="utf-8")
        return [str(path)]

    def harness_specs(self) -> List[Any]:
        return []

    @abstractmethod
    def resource_requirements(self) -> ResourceRequirements:
        """Declare actual verifier resource and timeout semantics."""
        ...

    # ---- default reward path (subprocess sandbox) --------------------
    def compute_reward(self, response_text: str, parent: ParentContext,
                       timeout_s: float) -> RewardResult:
        res = RewardResult(reward=self.fail_score)
        code = extract_python_code(response_text)
        if code is None:
            res.msg = "no_code_block"
            res.failure_kind = "code"
            return res
        res.parsed = True
        res.code = code

        full_code = self.preprocess(code, parent)
        resources = self.resource_requirements()
        out = run_code(
            full_code,
            entrypoint=self.entrypoint,
            timeout_s=min(float(timeout_s), float(resources.timeout_s)),
            max_cpus=resources.cpu_cores,
            memory_mb=resources.memory_mb,
            network_access=resources.network_access,
            filesystem_policy=resources.filesystem_policy,
        )
        diagnostics = [out.get("stdout", ""), out.get("traceback", ""),
                       out.get("stderr", "")]
        res.stdout = "\n".join(str(x).strip() for x in diagnostics if x).strip()
        if not out.get("ok"):
            error = str(out.get("error", "unknown"))
            res.msg = f"run_failed: {error}"
            res.failure_kind = ("timeout" if "timeout" in error.lower()
                                else "code")
            return res
        res.ran = True

        scored = self.score(out.get("value"), res.stdout)
        # carry engine-level fields the scorer does not set
        scored.parsed = True
        scored.ran = True
        scored.code = code
        if not scored.stdout:
            scored.stdout = res.stdout
        if not scored.valid and not scored.msg:
            scored.msg = "invalid"
        if not scored.valid and not scored.failure_kind:
            scored.failure_kind = "constraint"
        return scored
