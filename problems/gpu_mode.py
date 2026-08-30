"""
Kernel Engineering (GPU Mode: trimul / mla_decode_nvidia).

  - problem_type "trimul":  score_scale 1500, target ~1000 us
  - problem_type "mla_decode_nvidia": score_scale 5000, target ~1700 us
  - reward = score_scale / runtime_us   (minimize runtime, reward higher=better)

Needs:
  * a CUDA GPU + triton (and the packages in requirements/requirements-gpumode.txt)
  * the examples/gpu_mode/lib tree present (task.yml, reference.py, eval.py, utils.py)
  * run from the repo root so `examples` and `libkernelbot` resolve

Two changes from the original, both for the memory and feedback components:

  build_prompt takes `memory` and places the retrieved lessons between the
  parent kernel and the rules, adapting the instruction when they are present.

  compute_reward now captures compiler and test output into res.stdout. The
  original discarded it and returned only a one-line msg, which meant the
  feedback signal's f_i was "Failed to pass test cases." with nothing in it.
  Triton compile errors and correctness mismatches are the richest textual
  feedback this problem produces, and they were being thrown away.

  compute_reward now also honours a timeout. The original ignored
  sandbox_timeout_s entirely and called run_config inline, so a kernel that hung
  (an unterminated loop, a deadlocked launch) blocked forever. With
  reward_workers=1, which this problem requires, that stalls the whole step. The
  evaluation now runs in a spawned child that is terminated on timeout, and the
  child returns a plain dict so nothing has to pickle across.
"""

from __future__ import annotations
import ast
import hashlib
import inspect
import json
import math
import multiprocessing as mp
import os
import sys
from collections import Counter
from pathlib import Path
from statistics import mean, stdev
from typing import Any, Dict, List, Mapping, Optional, Tuple
from problems.base import (Problem, ParentContext, ResourceRequirements,
                           RewardResult, ScientificVerification, SeedState,
                           render_state_context)
from evolve.verifier.parsing import extract_python_code

from examples.gpu_mode.prompt import (
    TRIMUL_PROMPT,
    MLA_DECODE_PROMPT,
    MLA_DECODE_PROMPT_END,
    MLA_DECODE_INITIAL_STATE,
    MLA_DECODE_INITIAL_VALUE,
)

# Runtime scaling relative to H100, from memory bandwidth, because trimul and
# mla_decode are both bandwidth-bound. ESTIMATES: they set the reward scale, not
# correctness, and the right move is to benchmark one kernel and pin `target`
# and `score_scale` in the YAML. Until you do, this at least keeps reward
# magnitudes in the same range across hardware instead of silently 4x off.
#   H100 3.35 TB/s | H200 4.8 | A100 2.0 | L40S 0.864 | L4 0.3 | RTX4090 1.01
_ARCH_RUNTIME_FACTOR = {
    "h100": 1.0,
    "h200": 0.70,
    "a100": 1.68,
    "l40s": 3.88,
    "l4": 11.2,
    "rtx4090": 3.32,
    "a6000": 4.30,
}


def arch_factor(gpu_type: str) -> float:
    return _ARCH_RUNTIME_FACTOR.get((gpu_type or "").strip().lower(), 1.0)


_DEFAULTS = {
    "trimul": {
        "score_scale": 1500.0,
        "target": 1000.0,
        "gpu_type": "H100",
        "task_yaml": "examples/gpu_mode/lib/bioml/trimul/task.yml",
    },
    "mla_decode_nvidia": {
        "score_scale": 5000.0,
        "target": 1700.0,
        "gpu_type": "H200",
        "task_yaml": "examples/gpu_mode/lib/mla-decode/task.yml",
    },
}

_MEMORY_HEADER = """## Lessons from earlier kernels in this search

Extracted from kernels already generated and benchmarked here. Empirical
findings, not part of the specification above, and they do not override any rule
stated in it."""

_ARCH_NOTES = {
    "l40s": """Target hardware: NVIDIA L40S (Ada Lovelace, sm_89, 48 GB GDDR6, no NVLink).
Ada is not Hopper. The following do NOT exist on this device and will fail to
compile or silently fall back:
- TMA / `tl.make_tensor_descriptor` and descriptor-based async copies
- warpgroup MMA (`wgmma`), and anything assuming a 128-thread warpgroup
- thread-block clusters and distributed shared memory
- the 228 KB shared-memory-per-SM budget; Ada gives you 100 KB per SM
What Ada does have and is worth using: 142 SMs' worth of 4th-gen tensor cores
(bf16 and fp8), `cp.async` style pipelining, and a large 96 MB L2. Memory
bandwidth is 864 GB/s, roughly a quarter of an H100's, so this part is far more
bandwidth-bound than the same kernel on Hopper: favour fewer passes over the
data, fusion, and reuse through L2 over strategies that assume you can stream.""",
    "a100": """Target hardware: NVIDIA A100 (Ampere, sm_80). No TMA, no wgmma, no
thread-block clusters. 164 KB shared memory per SM, 1.5-2.0 TB/s HBM.""",
    "h100": """Target hardware: NVIDIA H100 (Hopper, sm_90). TMA, wgmma, and
thread-block clusters are available. 228 KB shared memory per SM, ~3.3 TB/s HBM.""",
    "h200": """Target hardware: NVIDIA H200 (Hopper, sm_90). As H100 but with 141 GB
HBM3e at ~4.8 TB/s.""",
}


def arch_notes(gpu_type: str) -> str:
    return _ARCH_NOTES.get((gpu_type or "").strip().lower(), "")


_LAUNCH_NOTE = """## Triton launch convention

Every failure mode below has appeared in this search. Read this before writing.

A `@triton.jit` function is NOT a Python function. It is launched with a grid
subscript, and its `tl.constexpr` parameters exist only inside it:

```python
@triton.jit
def my_kernel(x_ptr, y_ptr, n, BLOCK: tl.constexpr):
    pid = tl.program_id(0)
    offs = pid * BLOCK + tl.arange(0, BLOCK)
    m = offs < n
    tl.store(y_ptr + offs, tl.load(x_ptr + offs, mask=m, other=0.0), mask=m)

def custom_kernel(data):
    ...
    BLOCK = 128                                   # a host variable YOU define
    grid = (triton.cdiv(n, BLOCK),)               # host-side, use triton.cdiv
    my_kernel[grid](x, y, n, BLOCK=BLOCK)         # note the [grid] subscript
```

- `my_kernel(...)` without `[grid]` raises "Cannot call @triton.jit'd outside
  the scope of a kernel".
- `BLOCK`, `BLOCK_M`, `BLOCK_N` are not defined on the host unless you define
  them there. Referencing one you only declared as a `tl.constexpr` parameter
  raises NameError.
- `tl.cdiv` is device-side. On the host use `triton.cdiv`.
- Define the kernel at module top level, never nested inside `custom_kernel`.

You do not have to move everything into Triton. A correct submission that keeps
most of the computation in PyTorch and moves one hot region into a kernel scores;
one that fails to launch scores zero."""


_ANALYSIS_WITH_MEMORY = """## Analysis

Work through the lessons above before writing anything:
- Which apply to the kernel you were given, and what would each change?
- Which do NOT apply here, and why? Say so explicitly. They are evidence from
  earlier attempts, and some will be wrong or irrelevant for this kernel.
- Is anything they recommend already present above and still not fast enough?
  Then that avenue is exhausted and the win is somewhere they do not cover.

A lesson gives you an idea; you choose the implementation. Do not copy a kernel
body or an autotune configuration verbatim, and do not let a lesson fix your
tiling or memory-access strategy for you."""


_SCIENTIFIC_EVALUATOR_VERSION = "libkernelbot_leaderboard_timeout_v1"
_SCIENTIFIC_DESCRIPTOR_VERSION = "gpu_verified_kernel_cells_v1"
_SCIENTIFIC_FINGERPRINT_VERSION = "gpu_verified_kernel_family_v1"
_GPU_PAYLOAD_KEYS = frozenset({
    "schema_version", "problem", "kernel_source", "task", "hardware",
    "evaluator",
})


class _PayloadFormatError(ValueError):
    """The saved kernel answer is not a complete canonical payload."""


class _ScientificContextMismatch(ValueError):
    """The saved evaluator context differs from the active verifier."""


def _sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def _canonical_digest(value: Any) -> str:
    encoded = json.dumps(
        value, allow_nan=False, sort_keys=True, separators=(",", ":")
    ).encode("utf-8")
    return _sha256_bytes(encoded)


def _plain_json(value: Any) -> Any:
    try:
        return json.loads(json.dumps(value, allow_nan=False, sort_keys=True))
    except (TypeError, ValueError) as exc:
        raise _PayloadFormatError(
            "GPU answer payload must contain only finite JSON values: " + str(exc)
        ) from exc


def _call_name(node: ast.AST) -> str:
    if isinstance(node, ast.Name):
        return node.id
    if isinstance(node, ast.Attribute):
        prefix = _call_name(node.value)
        return (prefix + "." if prefix else "") + node.attr
    return ""


def _clip(s: str, n: int) -> str:
    s = s or ""
    return s if len(s) <= n else s[:n // 3] + "\n...[truncated]...\n" + s[-(2 * n // 3):]


def collect_logs(result, limit: int = 4000) -> str:
    """
    Everything the runner said, as f_i for the feedback signal and as failure
    evidence for the memory maker.

    Ordered worst-first on purpose: a compile error explains a failure and a
    benchmark log does not, and the feedback reprompt keeps the tail.
    """
    parts = []
    err = getattr(result, "error", "") or ""
    if err:
        parts.append("runner error:\n" + _clip(err, limit))

    for name, ev in (getattr(result, "runs", {}) or {}).items():
        comp = getattr(ev, "compilation", None)
        if comp is not None and not getattr(comp, "success", True):
            parts.append(f"[{name}] COMPILE FAILED\n"
                         + _clip(getattr(comp, "stderr", "") or
                                 getattr(comp, "stdout", ""), limit))
        run = getattr(ev, "run", None)
        if run is None:
            continue
        if not getattr(run, "passed", False):
            body = (getattr(run, "stderr", "") or "") + "\n" + (getattr(run, "stdout", "") or "")
            parts.append(f"[{name}] RUN FAILED\n" + _clip(body.strip(), limit))
        elif getattr(run, "stderr", ""):
            parts.append(f"[{name}] warnings\n" + _clip(run.stderr, limit // 2))

    return "\n\n".join(p for p in parts if p.strip())[: limit * 2]


def ensure_python3_on_path() -> Optional[str]:
    """
    libkernelbot runs the submission with subprocess.run(["python3", ...]).

    That name is hardcoded, so it must resolve to the SAME interpreter the run
    is using: a venv whose bin is not on PATH, a conda env exposing only
    `python`, or a container without /usr/bin/python3 all produce a bare
    FileNotFoundError from subprocess with no indication of what was missing.
    Worse, if some other python3 resolves first, the submission runs against a
    different torch than the one you installed.

    So: put the directory of sys.executable first on PATH, and if that still
    does not give a `python3`, symlink one into a temp dir and prepend that.
    Returns the resolved path, or None if it could not be arranged.
    """
    import shutil
    import tempfile

    exe_dir = os.path.dirname(os.path.abspath(sys.executable))
    os.environ["PATH"] = exe_dir + os.pathsep + os.environ.get("PATH", "")

    found = shutil.which("python3")
    if found and os.path.realpath(found) == os.path.realpath(sys.executable):
        return found
    if os.path.exists(os.path.join(exe_dir, "python3")):
        return os.path.join(exe_dir, "python3")

    try:
        shim = tempfile.mkdtemp(prefix="evolve-py3-")
        link = os.path.join(shim, "python3")
        os.symlink(os.path.abspath(sys.executable), link)
        os.environ["PATH"] = shim + os.pathsep + os.environ["PATH"]
        return link
    except Exception:
        return found        # whatever was there, possibly the wrong interpreter


def _eval_child(conn, code: str, lib_dir: str, task_yaml: str,
                problem_type: str, log_chars: int,
                gpu_id: Optional[int] = None) -> None:
    """
    Runs in a spawned process. Does the whole evaluation and sends back a plain
    dict: no dataclasses, no datetimes, nothing that has to pickle.

    gpu_id pins the benchmark to one device, set BEFORE torch is imported so the
    driver honours it. This matters more than it looks: the reward here IS a
    measured runtime, so benchmarking on a device that is concurrently running
    generation or a training backward pass produces a number that reflects the
    contention rather than the kernel. Reserve a device that nothing else uses.
    """
    out = {"ok": False, "msg": "unknown", "logs": "", "score_us": None}
    if gpu_id is not None:
        # The spawned evaluator must not inherit distributed-generation state.
        # It sees exactly one physical GPU, which becomes logical cuda:0.
        for key in list(os.environ):
            if (
                key in {
                    "LOCAL_RANK",
                    "RANK",
                    "WORLD_SIZE",
                    "MASTER_ADDR",
                    "MASTER_PORT",
                }
                or key.startswith("VLLM_")
                or key.startswith("RAY_")
            ):
                os.environ.pop(key, None)
        os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
        os.environ["CUDA_VISIBLE_DEVICES"] = str(gpu_id)
    py3 = ensure_python3_on_path()
    try:
        if lib_dir not in sys.path:
            sys.path.insert(0, lib_dir)
        from libkernelbot.task import make_task_definition, build_task_config
        from libkernelbot.run_eval import run_config
        from libkernelbot.submission import compute_score
        from libkernelbot.consts import SubmissionMode

        task = make_task_definition(task_yaml).task
        config = build_task_config(task=task, submission_content=code,
                                   arch=None, mode=SubmissionMode.LEADERBOARD)
        result = run_config(config)
        out["logs"] = collect_logs(result, log_chars)

        if not getattr(result, "success", False):
            out["msg"] = f"Error: {getattr(result, 'error', 'run failed')}"
        else:
            runs = result.runs
            if "test" in runs and (not runs["test"].run or not runs["test"].run.passed):
                out["msg"] = "Failed to pass test cases."
            elif ("leaderboard" not in runs or not runs["leaderboard"].run
                  or not runs["leaderboard"].run.passed):
                out["msg"] = "No passing leaderboard run."
            else:
                score_seconds = compute_score(result, task, submission_id=-1)
                out["score_us"] = float(score_seconds) * 1_000_000.0
                out["ok"] = True
                out["msg"] = f"runtime_us={out['score_us']}"
    except FileNotFoundError as e:
        # subprocess raises this with the executable in .filename and nothing
        # in the repr, which is how this arrived as an unreadable
        # "FileNotFoundError(2, 'No such file or directory')".
        missing = getattr(e, "filename", None) or "(unknown)"
        out["msg"] = (f"Local kernel run failed: could not find {missing!r}. "
                      f"python3 resolved to {py3!r}, sys.executable is "
                      f"{sys.executable!r}, cwd is {os.getcwd()!r}.")
    except Exception as e:
        out["msg"] = (f"Local kernel run failed: {type(e).__name__}: {e} "
                      f"(cwd={os.getcwd()!r})")
    try:
        conn.send(out)
    except Exception:
        pass
    finally:
        conn.close()


def run_eval_with_timeout(code: str, lib_dir: str, task_yaml: str,
                          problem_type: str, log_chars: int,
                          timeout_s: float, gpu_id: Optional[int] = None) -> Dict:
    """
    Spawn, wait, and kill on timeout.

    A thread cannot be killed, so a thread-based timeout would leave the hung
    kernel running in the background and leak a GPU context every time. A child
    process can be terminated, which is the point.
    """
    ctx = mp.get_context("spawn")
    parent_conn, child_conn = ctx.Pipe(duplex=False)
    proc = ctx.Process(target=_eval_child,
                       args=(child_conn, code, lib_dir, task_yaml,
                             problem_type, log_chars, gpu_id), daemon=True)
    proc.start()
    child_conn.close()

    out = None
    try:
        if parent_conn.poll(timeout_s):
            out = parent_conn.recv()
    except Exception:
        out = None
    finally:
        parent_conn.close()

    if proc.is_alive():
        proc.terminate()
        proc.join(10)
        if proc.is_alive():
            proc.kill()
            proc.join(5)
    else:
        proc.join(5)

    if out is None:
        return {"ok": False, "score_us": None, "logs": "",
                "msg": f"kernel_eval_timeout after {timeout_s:.0f}s "
                       f"(process killed)"}
    return out


# Retain the loaded protocol functions for content-addressing even when a
# CPU-only test replaces the public evaluator callables.  Execution still goes
# through ``run_eval_with_timeout`` below and is therefore safely patchable.
_SCIENTIFIC_COLLECT_LOGS = collect_logs
_SCIENTIFIC_EVAL_CHILD = _eval_child
_SCIENTIFIC_TIMEOUT_RUNNER = run_eval_with_timeout


class GpuMode(Problem):
    name = "gpu_mode"
    metric_name = "runtime (microseconds)"
    maximize = False
    scientific_method_complete = True
    answer_schema_version = 1
    descriptor_function_version = _SCIENTIFIC_DESCRIPTOR_VERSION
    fingerprint_function_version = _SCIENTIFIC_FINGERPRINT_VERSION

    def __init__(self, cfg: dict):
        super().__init__(cfg)
        self.problem_type = str(cfg.get("problem_type", "trimul")).lower()
        if self.problem_type not in _DEFAULTS:
            raise ValueError(f"problem_type must be one of {list(_DEFAULTS)}, "
                             f"got {self.problem_type}")
        d = _DEFAULTS[self.problem_type]
        self.gpu_type = str(cfg.get("gpu_type", d["gpu_type"]))
        self.triton_version = str(cfg.get("triton_version", "3.3.1"))

        # target and score_scale default to the H100 numbers scaled to this
        # device, so setting gpu_type is enough. An explicit value in the YAML
        # or on the CLI still wins.
        f = arch_factor(self.gpu_type)
        self._arch_factor = f
        self.score_scale = float(cfg.get("score_scale", d["score_scale"] * f))
        self.task_yaml = str(cfg.get("task_yaml", d["task_yaml"]))
        self.lib_dir = str(cfg.get("kernel_lib_dir",
                                   cfg.get("lib_dir", "examples/gpu_mode/lib")))
        if self.target is None:
            self.target = float(d["target"]) * f
        if abs(f - 1.0) > 1e-9 and ("target" not in cfg or "score_scale" not in cfg):
            print(f"[gpu_mode] {self.gpu_type}: scaled H100 defaults by {f:.2f}x "
                  f"-> target={self.target:.0f}us, score_scale={self.score_scale:.0f} "
                  f"(reward {self.score_scale / self.target:.2f} at target). "
                  f"ESTIMATE from memory bandwidth; benchmark one kernel and pin "
                  f"both in the YAML.")
        missing = self.missing_task_files()
        if missing:
            raise FileNotFoundError(
                "gpu_mode cannot start: task.yml references files that are not "
                "present:\n  " + "\n  ".join(missing)
                + "\nFetch them from github.com/gpu-mode/reference-kernels "
                  "(problems/bioml/trimul and problems/amd/mla-decode) into the "
                  "directory holding task.yml.")

        self.log_chars = int(cfg.get("kernel_log_chars", 4000))
        # 0 falls back to sandbox_timeout_s; set explicitly to override it.
        self.show_launch_note = bool(cfg.get("show_launch_note", True))
        # Seed the tree with the task's reference submission instead of empty
        # code. With code="" the model writes a kernel from nothing at step 0,
        # and the whole batch fails on launch syntax rather than on anything
        # about the kernel. The reference is correct, complete, and slow, which
        # is exactly what a search wants as a starting point.
        self.seed_from_reference = bool(cfg.get("seed_from_reference", True))
        self.kernel_timeout_s = float(cfg.get("kernel_timeout_s", 0.0))
        # Physical device the benchmark runs on. None inherits the parent's,
        # which is the training GPU: correct only if nothing else is on it.
        gid = cfg.get("kernel_gpu_id", None)
        self.kernel_gpu_id = None if gid is None else int(gid)
        self.kernel_eval_isolation = bool(
            cfg.get("kernel_eval_isolation", True)
        )
        # entrypoint is implicit (custom_kernel inside submission.py)
        self.entrypoint = "custom_kernel"

    # ------------------------------------------------------------------
    def build_prompt(self, parent: ParentContext, memory: str = "") -> List[dict]:
        state_ctx = render_state_context(self.metric_name, self.target, parent,
                                         maximize=self.maximize)

        memory_section = ""
        analysis = ""
        if memory and memory.strip():
            memory_section = f"\n{_MEMORY_HEADER}\n\n{memory.strip()}\n"
            analysis = f"\n{_ANALYSIS_WITH_MEMORY}\n"

        notes = arch_notes(self.gpu_type)
        arch_section = f"\n{notes}\n" if notes else ""
        launch_section = f"\n{_LAUNCH_NOTE}\n" if self.show_launch_note else ""

        if self.problem_type == "trimul":
            user = f"""{TRIMUL_PROMPT}
{arch_section}{launch_section}
{state_ctx}
{memory_section}{analysis}
Rules:
- The tensors arguments passed in will be already on your cuda device.
- Define all of your code in one final ```python ``` block.
- We will test the correctness of your kernel on multiple input shapes, make sure to support different potential test cases.
- You are allowed to use mixed precision computations, but make sure your final output is in float32.
- You must use triton {self.triton_version} and these kernels will be run on an {self.gpu_type}.
- You do not have to implement everything in triton, you may choose to have some of the operations done in pytorch. However, you must implement at least part of the operations in a kernel.
- Include a short docstring at the top summarizing your algorithm.
- Do not wrap the kernel in a try/except that falls back to a slow reference path.
  A fallback that silently produces correct-but-slow output hides the failure and
  scores worse than letting the kernel fail loudly.
"""
        else:
            user = f"""{MLA_DECODE_PROMPT}
{arch_section}{launch_section}
{state_ctx}
{memory_section}{analysis}
{MLA_DECODE_PROMPT_END}
"""
        return [{"role": "user", "content": user}]

    def preprocess(self, code: str, parent: ParentContext) -> str:
        return code

    def score(self, output: Any, stdout: str) -> RewardResult:
        return RewardResult(reward=self.fail_score, msg="unused")

    def missing_task_files(self) -> List[str]:
        """
        task.yml lists source files by relative path and libkernelbot reads them
        with (root / source).read_text(), so an absent one surfaces as a bare
        FileNotFoundError from deep inside the runner, once per rollout. Check
        up front and say what is missing.
        """
        import yaml as _yaml
        from pathlib import Path
        p = Path(self.task_yaml)
        if not p.exists():
            return [f"{self.task_yaml} (task.yml itself; run from the repo root)"]
        try:
            raw = _yaml.safe_load(p.read_text())
        except Exception as e:
            return [f"{self.task_yaml} (unparsable: {e})"]
        out = []
        # `files:` are the sources copied into the run directory.
        for spec in raw.get("files", []):
            src = spec.get("source")
            if not src or src == "@SUBMISSION@":
                continue
            if not (p.parent / src).exists():
                out.append(str(p.parent / src))
        # `templates:` are read too, by make_task_definition, and are easy to
        # miss because the same filename also appears in `files:` as the
        # @SUBMISSION@ placeholder. Different thing, still has to exist.
        for src in (raw.get("templates", {}) or {}).values():
            if src and not (p.parent / src).exists():
                out.append(f"{p.parent / src} (templates:)")
        return out

    # ------------------------------------------------------------------
    # EVOLVE scientific-answer contract. Proposal execution and independent
    # saved-kernel verification remain separate paths.
    def _scientific_timeout_s(self) -> float:
        timeout = float(self.kernel_timeout_s or 0.0)
        if timeout <= 0.0:
            timeout = float(self.cfg.get("sandbox_timeout_s", 0.0) or 0.0)
        # EVOLVE never permits an unbounded verifier. A config that supplied
        # neither timeout gets a conservative finite boundary.
        return timeout if timeout > 0.0 and math.isfinite(timeout) else 900.0

    @staticmethod
    def _file_record(path: Path, logical_path: str,
                     purpose: str) -> Mapping[str, Any]:
        try:
            data = path.read_bytes()
        except OSError as exc:
            raise _ScientificContextMismatch(
                f"cannot freeze {purpose} file {path}: {exc}"
            ) from exc
        return {
            "logical_path": str(logical_path).replace(os.sep, "/"),
            "purpose": purpose,
            "sha256": _sha256_bytes(data),
        }

    def _task_snapshot(self) -> Mapping[str, Any]:
        import yaml as _yaml

        task_path = Path(self.task_yaml)
        try:
            task_bytes = task_path.read_bytes()
            raw = _yaml.safe_load(task_bytes.decode("utf-8"))
        except (OSError, UnicodeError, ValueError) as exc:
            raise _ScientificContextMismatch(
                f"cannot freeze task manifest {self.task_yaml}: {exc}"
            ) from exc
        if not isinstance(raw, Mapping):
            raise _ScientificContextMismatch(
                f"task manifest {self.task_yaml} is not a mapping"
            )

        records = []
        files = raw.get("files", [])
        if not isinstance(files, list):
            raise _ScientificContextMismatch("task manifest files must be a list")
        for spec in files:
            if not isinstance(spec, Mapping):
                raise _ScientificContextMismatch(
                    "task manifest file entries must be mappings"
                )
            source = spec.get("source")
            if not source or source == "@SUBMISSION@":
                continue
            source_text = str(source)
            records.append(self._file_record(
                task_path.parent / source_text,
                source_text,
                "task_source",
            ))

        templates = raw.get("templates", {}) or {}
        if not isinstance(templates, Mapping):
            raise _ScientificContextMismatch(
                "task manifest templates must be a mapping"
            )
        for language, source in templates.items():
            if not source:
                continue
            source_text = str(source)
            records.append(self._file_record(
                task_path.parent / source_text,
                source_text,
                "template_" + str(language),
            ))
        records = sorted(
            records,
            key=lambda item: (
                str(item["logical_path"]), str(item["purpose"]),
                str(item["sha256"]),
            ),
        )
        identity = {
            "problem_type": self.problem_type,
            "task_yaml": str(self.task_yaml).replace(os.sep, "/"),
            "manifest_sha256": _sha256_bytes(task_bytes),
            "files": records,
        }
        return {**identity, "bundle_sha256": _canonical_digest(identity)}

    def _evaluator_snapshot(self) -> Mapping[str, Any]:
        root = Path(self.lib_dir)
        package_root = root / "libkernelbot"
        evaluator_files = []
        if package_root.is_dir():
            for path in sorted(package_root.rglob("*.py")):
                logical = path.relative_to(root).as_posix()
                evaluator_files.append(
                    self._file_record(path, logical, "evaluator_source")
                )
        if not evaluator_files:
            raise _ScientificContextMismatch(
                f"cannot freeze evaluator: no Python files under {package_root}"
            )
        bundle_identity = {"files": evaluator_files}
        protocol_sources = {
            "collect_logs": inspect.getsource(_SCIENTIFIC_COLLECT_LOGS),
            "eval_child": inspect.getsource(_SCIENTIFIC_EVAL_CHILD),
            "run_eval_with_timeout": inspect.getsource(
                _SCIENTIFIC_TIMEOUT_RUNNER
            ),
            "failure_classification": inspect.getsource(
                type(self)._failure_classification
            ),
            "verify_answer_payload": inspect.getsource(
                type(self).verify_answer_payload
            ),
        }
        return {
            "version": _SCIENTIFIC_EVALUATOR_VERSION,
            "submission_mode": "leaderboard",
            "lib_dir": str(self.lib_dir).replace(os.sep, "/"),
            "libkernelbot_bundle_sha256": _canonical_digest(bundle_identity),
            "wrapper_protocol_sha256": _canonical_digest(protocol_sources),
            "files": evaluator_files,
            "score_scale": float(self.score_scale),
            "target_runtime_us": float(self.target),
            "timeout_s": float(self._scientific_timeout_s()),
            "log_chars": int(self.log_chars),
            "timeout_is_scientific": False,
        }

    def _scientific_context(self) -> Mapping[str, Any]:
        return {
            "task": self._task_snapshot(),
            "hardware": {
                "declared_gpu_type": self.gpu_type.strip().lower(),
                "triton_version": self.triton_version.strip(),
                # Device index is frozen because it identifies the exclusive
                # benchmark lease used for a measured record.  It is not used
                # as a proxy for architecture.
                "kernel_gpu_id": self.kernel_gpu_id,
                "exclusive_evaluation": self.kernel_eval_isolation,
            },
            "evaluator": self._evaluator_snapshot(),
        }

    @staticmethod
    def _validate_context_section(value: Any, expected: Mapping[str, Any],
                                  section: str) -> None:
        if not isinstance(value, Mapping):
            raise _PayloadFormatError(f"GPU payload {section} must be a mapping")
        if set(value) != set(expected):
            raise _PayloadFormatError(
                f"GPU payload {section} fields must be exactly "
                f"{sorted(expected)}"
            )
        for list_name in ("files",):
            if list_name not in expected:
                continue
            entries = value.get(list_name)
            if not isinstance(entries, (list, tuple)):
                raise _PayloadFormatError(
                    f"GPU payload {section}.{list_name} must be a list"
                )
            for entry in entries:
                if (not isinstance(entry, Mapping)
                        or set(entry) != {"logical_path", "purpose", "sha256"}):
                    raise _PayloadFormatError(
                        f"GPU payload {section}.{list_name} has a malformed entry"
                    )

    def _validated_payload(self, payload: Any) -> Mapping[str, Any]:
        plain = _plain_json(payload)
        if not isinstance(plain, Mapping):
            raise _PayloadFormatError("GPU answer payload must be a mapping")
        if set(plain) != _GPU_PAYLOAD_KEYS:
            raise _PayloadFormatError(
                "GPU answer payload fields must be exactly "
                + str(sorted(_GPU_PAYLOAD_KEYS))
            )
        if (type(plain.get("schema_version")) is not int
                or plain["schema_version"] != self.answer_schema_version):
            raise _PayloadFormatError(
                f"unsupported GPU answer schema {plain.get('schema_version')!r}"
            )
        if plain.get("problem") != self.name:
            raise _PayloadFormatError("GPU answer problem must be 'gpu_mode'")
        source = plain.get("kernel_source")
        if not isinstance(source, str) or not source.strip():
            raise _PayloadFormatError("GPU answer kernel_source must be non-empty")

        current = self._scientific_context()
        for section in ("task", "hardware", "evaluator"):
            self._validate_context_section(
                plain.get(section), current[section], section
            )
            if plain[section] != current[section]:
                raise _ScientificContextMismatch(
                    f"saved GPU {section} does not match the active verifier"
                )
        return plain

    def serialize_answer(self, candidate: Any, evidence: Any = None) -> Any:
        del evidence
        if isinstance(candidate, Mapping) and set(candidate) == _GPU_PAYLOAD_KEYS:
            return self._validated_payload(candidate)

        source: Any = candidate
        if isinstance(candidate, RewardResult):
            source = candidate.code
        elif isinstance(candidate, Mapping):
            source = candidate.get(
                "kernel_source", candidate.get("code", candidate.get("source"))
            )
        if not isinstance(source, str):
            raise ValueError(
                "GPU answer must be kernel source, a RewardResult, or a saved payload"
            )
        # Generation artifacts may still contain the surrounding markdown at
        # capture time.  Extraction happens once here; verification receives
        # only the resulting persisted kernel and never revisits the proposal.
        extracted = extract_python_code(source) if "```" in source else None
        kernel_source = extracted if extracted is not None else source
        if not kernel_source.strip():
            raise ValueError("GPU kernel submission must be non-empty")

        context = self._scientific_context()
        payload = {
            "schema_version": self.answer_schema_version,
            "problem": self.name,
            "kernel_source": kernel_source,
            "task": context["task"],
            "hardware": context["hardware"],
            "evaluator": context["evaluator"],
        }
        return self._validated_payload(payload)

    @staticmethod
    def _kernel_structure(source: str) -> Tuple[ast.AST, Mapping[str, Any]]:
        tree = ast.parse(source, filename="saved_submission.py", mode="exec")
        node_counts = Counter(type(node).__name__ for node in ast.walk(tree))
        call_counts = Counter()
        triton_kernel_count = 0
        constexpr_parameters = 0
        launch_count = 0
        for node in ast.walk(tree):
            if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
                decorators = {_call_name(item) for item in node.decorator_list}
                if "triton.jit" in decorators:
                    triton_kernel_count += 1
                for argument in node.args.args + node.args.kwonlyargs:
                    annotation = _call_name(argument.annotation) if argument.annotation else ""
                    if annotation == "tl.constexpr" or annotation == "triton.language.constexpr":
                        constexpr_parameters += 1
            if isinstance(node, ast.Call):
                name = _call_name(node.func)
                if name:
                    call_counts[name] += 1
                if isinstance(node.func, ast.Subscript):
                    launch_count += 1

        tl_primitives = {
            name: count for name, count in sorted(call_counts.items())
            if name.startswith("tl.") or name.startswith("triton.language.")
        }
        torch_calls = sum(
            count for name, count in call_counts.items()
            if name.startswith("torch.") or name.startswith("F.")
        )
        dot_calls = sum(
            count for name, count in call_counts.items()
            if name.endswith(".dot") or name.endswith(".matmul")
        )
        memory_calls = sum(
            count for name, count in call_counts.items()
            if name.endswith(".load") or name.endswith(".store")
        )
        if dot_calls:
            family = "dot_or_matmul"
        elif memory_calls and torch_calls:
            family = "hybrid_memory"
        elif memory_calls:
            family = "triton_memory"
        elif triton_kernel_count:
            family = "other_triton"
        else:
            family = "no_triton_kernel"

        branch_nodes = sum(node_counts[name] for name in (
            "If", "For", "While", "Try", "BoolOp", "IfExp",
        ))
        source_lines = len(source.splitlines())
        if branch_nodes <= 2:
            control_flow_bin = "low"
        elif branch_nodes <= 10:
            control_flow_bin = "medium"
        else:
            control_flow_bin = "high"
        if source_lines <= 80:
            source_size_bin = "small"
        elif source_lines <= 240:
            source_size_bin = "medium"
        else:
            source_size_bin = "large"
        return tree, {
            "node_counts": dict(sorted(node_counts.items())),
            "call_counts": dict(sorted(call_counts.items())),
            "tl_primitives": tl_primitives,
            "triton_kernel_count": triton_kernel_count,
            "constexpr_parameters": constexpr_parameters,
            "launch_count": launch_count,
            "torch_call_count": torch_calls,
            "branch_nodes": branch_nodes,
            "control_flow_bin": control_flow_bin,
            "source_lines": source_lines,
            "source_size_bin": source_size_bin,
            "kernel_family": family,
        }

    @staticmethod
    def _failure_classification(out: Mapping[str, Any]) -> Tuple[str, str]:
        explicit = str(out.get("failure_kind", "") or "").lower()
        explicit_map = {
            "parse": ("parse", "syntax"),
            "syntax": ("parse", "syntax"),
            "code": ("code", "compilation_or_execution"),
            "compile": ("code", "compilation_or_execution"),
            "constraint": ("constraint", "correctness"),
            "correctness": ("constraint", "correctness"),
            "timeout": ("timeout", "timeout"),
            "infrastructure": ("infrastructure", "infrastructure"),
        }
        if explicit in explicit_map:
            return explicit_map[explicit]

        message = str(out.get("msg", "") or "")
        logs = str(out.get("logs", "") or "")
        combined = (message + "\n" + logs).lower()
        if "timeout" in combined or "timed out" in combined:
            return "timeout", "timeout"
        if ("failed to pass test cases" in combined
                or "no passing leaderboard run" in combined
                or "correctness" in combined or "mismatch" in combined):
            return "constraint", "correctness"
        if ("compile failed" in combined or "syntaxerror" in combined
                or "compilation error" in combined):
            return "code", "compilation_or_execution"
        infrastructure_markers = (
            "could not find", "no such file", "task files missing",
            "cuda unavailable", "no cuda", "driver", "out of memory",
            "local kernel run failed", "broken pipe", "worker died",
        )
        if any(marker in combined for marker in infrastructure_markers):
            return "infrastructure", "infrastructure"
        # An unclassified runner failure cannot safely be interpreted as a bad
        # scientific candidate.  It remains unresolved and retry-eligible.
        return "infrastructure", "infrastructure_unknown"

    @staticmethod
    def _evidence_value(evidence: Any, name: str,
                        default: Any = None) -> Any:
        if isinstance(evidence, Mapping):
            return evidence.get(name, default)
        return getattr(evidence, name, default)

    @classmethod
    def _answer_from_evidence(cls, candidate: Any,
                              evidence: Any) -> Any:
        payload = cls._evidence_value(evidence, "answer_payload")
        return payload if payload is not None else candidate

    def _verification_failure(self, *, answer: Any, failure_kind: str,
                              classification: str, message: str,
                              logs: str = "") -> ScientificVerification:
        resolved = failure_kind not in ("timeout", "infrastructure")
        diagnostics = {
            "classification": classification,
            "runner_message": message,
        }
        if logs:
            diagnostics["runner_logs"] = logs
        return ScientificVerification(
            resolved=resolved,
            admitted=False,
            answer_payload=answer,
            failure_kind=failure_kind,
            message=message,
            flags={
                "method_complete": True,
                "payload_only": True,
                "proposal_replay": False,
                "failure_classification": classification,
            },
            scores={"diagnostic_chars": len(logs)},
            diagnostics=diagnostics,
        )

    def verify_answer_payload(
        self,
        payload: Any,
        policy: Optional[Mapping[str, Any]] = None,
    ) -> ScientificVerification:
        try:
            answer = self._validated_payload(payload)
        except _PayloadFormatError as exc:
            return self._verification_failure(
                answer=None,
                failure_kind="parse",
                classification="payload_format",
                message=str(exc),
            )
        except _ScientificContextMismatch as exc:
            return self._verification_failure(
                answer=_plain_json(payload),
                failure_kind="infrastructure",
                classification="frozen_context_mismatch",
                message=str(exc),
            )

        source = str(answer["kernel_source"])
        try:
            _tree, structure = self._kernel_structure(source)
        except (SyntaxError, ValueError, TypeError) as exc:
            return self._verification_failure(
                answer=answer,
                failure_kind="parse",
                classification="syntax",
                message=f"saved kernel syntax error: {exc}",
            )
        if int(structure["triton_kernel_count"]) < 1:
            return self._verification_failure(
                answer=answer,
                failure_kind="constraint",
                classification="kernel_requirement",
                message="Code must define at least one @triton.jit kernel.",
            )
        if self.problem_type == "trimul" and "identity" in source:
            return self._verification_failure(
                answer=answer,
                failure_kind="constraint",
                classification="kernel_requirement",
                message="Identity kernel is not allowed.",
            )

        missing = self.missing_task_files()
        if missing:
            return self._verification_failure(
                answer=answer,
                failure_kind="infrastructure",
                classification="task_files_missing",
                message="task files missing: " + ", ".join(missing),
            )

        evaluator = answer["evaluator"]
        hardware = answer["hardware"]
        diagnostic_limit = int(evaluator["log_chars"]) * 2
        if isinstance(policy, Mapping):
            requested = policy.get("max_diagnostic_chars")
            if (isinstance(requested, int) and not isinstance(requested, bool)
                    and requested >= 0):
                diagnostic_limit = min(diagnostic_limit, requested)
        measurements = []
        log_parts = []
        message = ""
        for repeat_index in range(3):
            try:
                out = run_eval_with_timeout(
                    source,
                    str(evaluator["lib_dir"]),
                    str(answer["task"]["task_yaml"]),
                    str(answer["task"]["problem_type"]),
                    int(evaluator["log_chars"]),
                    float(evaluator["timeout_s"]),
                    hardware["kernel_gpu_id"],
                )
            except Exception as exc:
                return self._verification_failure(
                    answer=answer,
                    failure_kind="infrastructure",
                    classification="runner_exception",
                    message=f"kernel evaluator repeat {repeat_index} raised {type(exc).__name__}: {exc}",
                )
            if not isinstance(out, Mapping):
                return self._verification_failure(
                    answer=answer,
                    failure_kind="infrastructure",
                    classification="runner_protocol",
                    message="kernel evaluator returned a non-mapping result",
                )
            message = str(out.get("msg", "run failed") or "run failed")
            log_parts.append(str(out.get("logs", "") or ""))
            if not out.get("ok"):
                kind, classification = self._failure_classification(out)
                return self._verification_failure(
                    answer=answer,
                    failure_kind=kind,
                    classification=classification,
                    message=message,
                    logs="\n".join(log_parts)[:diagnostic_limit],
                )
            try:
                measured = float(out["score_us"])
            except (KeyError, TypeError, ValueError) as exc:
                return self._verification_failure(
                    answer=answer,
                    failure_kind="infrastructure",
                    classification="invalid_measurement",
                    message=f"kernel evaluator returned invalid score_us: {exc}",
                )
            if not math.isfinite(measured) or measured <= 0.0:
                return self._verification_failure(
                    answer=answer,
                    failure_kind="infrastructure",
                    classification="invalid_measurement",
                    message=f"kernel evaluator returned invalid runtime {measured!r}",
                )
            measurements.append(measured)

        # Joining three empty repeat logs produces ``"\n\n"`` and would
        # falsely classify a clean verification as diagnostic-bearing.
        logs = "\n".join(part for part in log_parts if part)[:diagnostic_limit]
        runtime_us = float(mean(measurements))
        standard_error = float(stdev(measurements) / math.sqrt(len(measurements)))
        conservative_runtime = runtime_us + 1.96 * standard_error
        reward = float(evaluator["score_scale"]) / conservative_runtime
        warning_count = logs.lower().count("warning")
        return ScientificVerification(
            resolved=True,
            admitted=True,
            answer_payload=answer,
            internal_reward=reward,
            raw_score=runtime_us,
            # Repeated independently bounded evaluator calls provide the
            # uncertainty estimate used for conservative record comparison.
            uncertainty=standard_error,
            message=f"verified saved kernel; runtime_us={runtime_us}",
            scores={
                "runtime_us": runtime_us,
                "runtime_standard_error_us": standard_error,
                "conservative_runtime_us": conservative_runtime,
                "verification_repeats": len(measurements),
                "record_reward": reward,
                "runtime_to_target": (
                    runtime_us / float(evaluator["target_runtime_us"])
                ),
                "diagnostic_chars": len(logs),
                "warning_count": warning_count,
                "triton_kernel_count": int(structure["triton_kernel_count"]),
                "launch_count": int(structure["launch_count"]),
            },
            flags={
                "method_complete": True,
                "payload_only": True,
                "proposal_replay": False,
                "noisy_runtime": True,
                "requires_independent_record_confirmation": True,
            },
            diagnostics={
                "classification": "admitted",
                "runner_message": message,
                "runner_logs": logs,
            },
        )

    @staticmethod
    def _performance_bin(runtime_us: float, target_us: float) -> str:
        ratio = runtime_us / target_us
        if ratio <= 0.75:
            return "well_below_target"
        if ratio <= 1.05:
            return "near_target"
        if ratio <= 2.0:
            return "within_2x_target"
        return "above_2x_target"

    def describe_scientific_state(self, candidate: Any,
                                  evidence: Any = None) -> Mapping[str, Any]:
        answer = self._validated_payload(
            self._answer_from_evidence(candidate, evidence)
        )
        _tree, structure = self._kernel_structure(str(answer["kernel_source"]))
        raw_score = self._evidence_value(evidence, "raw_score")
        if isinstance(raw_score, bool) or not isinstance(raw_score, (int, float)):
            scores = self._evidence_value(evidence, "scores", {})
            raw_score = scores.get("runtime_us") if isinstance(scores, Mapping) else None
        if (isinstance(raw_score, bool)
                or not isinstance(raw_score, (int, float))
                or not math.isfinite(float(raw_score)) or float(raw_score) <= 0.0):
            raise ValueError("verified GPU evidence has no positive runtime_us")
        scores = self._evidence_value(evidence, "scores", {})
        warnings = scores.get("warning_count", 0) if isinstance(scores, Mapping) else 0
        diagnostic_chars = (
            scores.get("diagnostic_chars", 0) if isinstance(scores, Mapping) else 0
        )
        if int(warnings) > 0:
            diagnostic_bin = "warnings"
        elif int(diagnostic_chars) > 0:
            diagnostic_bin = "logged_clean"
        else:
            diagnostic_bin = "clean"
        target_us = float(answer["evaluator"]["target_runtime_us"])
        return {
            "problem_type": str(answer["task"]["problem_type"]),
            "declared_gpu_type": str(answer["hardware"]["declared_gpu_type"]),
            "task_bundle": str(answer["task"]["bundle_sha256"]),
            "kernel_family": str(structure["kernel_family"]),
            "kernel_count_bin": (
                "single" if int(structure["triton_kernel_count"]) == 1 else "multiple"
            ),
            "control_flow_bin": str(structure["control_flow_bin"]),
            "source_size_bin": str(structure["source_size_bin"]),
            "performance_bin": self._performance_bin(float(raw_score), target_us),
            "diagnostic_bin": diagnostic_bin,
        }

    def scientific_fingerprint(self, candidate: Any,
                               evidence: Any = None) -> str:
        answer = self._validated_payload(
            self._answer_from_evidence(candidate, evidence)
        )
        _tree, structure = self._kernel_structure(str(answer["kernel_source"]))
        descriptor = self.describe_scientific_state(answer, evidence)
        # Deliberately no source digest: this is a verified structural and
        # behavioral family fingerprint, not the scientific-state identity.
        fingerprint_payload = {
            "version": self.fingerprint_function_version,
            "task_identity": {
                "problem_type": answer["task"]["problem_type"],
                "task_bundle_sha256": answer["task"]["bundle_sha256"],
                "declared_gpu_type": answer["hardware"]["declared_gpu_type"],
                "triton_version": answer["hardware"]["triton_version"],
                "evaluator_bundle_sha256": answer["evaluator"][
                    "libkernelbot_bundle_sha256"
                ],
            },
            "descriptor": descriptor,
            "ast_node_histogram": structure["node_counts"],
            "call_histogram": structure["call_counts"],
            "triton_primitives": structure["tl_primitives"],
            "constexpr_parameters": structure["constexpr_parameters"],
            "launch_count": structure["launch_count"],
            "branch_nodes": structure["branch_nodes"],
        }
        return _canonical_digest(fingerprint_payload)

    def resource_requirements(self) -> ResourceRequirements:
        memory_mb = self.cfg.get("kernel_memory_mb", 4096)
        return ResourceRequirements(
            cpu_cores=1,
            memory_mb=int(memory_mb),
            timeout_s=float(self._scientific_timeout_s()),
            gpu_count=1,
            exclusive_gpu=True,
            network_access=False,
            filesystem_policy="isolated_kernel_workspace_v1",
            # A benchmark timeout can be caused by driver/resource state as
            # well as the kernel.  Keep it unresolved and retry-eligible.
            timeout_is_scientific=False,
        )

    def render_best(self, candidate: Any, evidence: Any,
                    output_dir: Any) -> List[str]:
        answer = self._validated_payload(
            self._answer_from_evidence(candidate, evidence)
        )
        destination = Path(output_dir)
        destination.mkdir(parents=True, exist_ok=True)
        kernel_path = destination / "answer.py"
        payload_path = destination / "answer.json"
        summary_path = destination / "answer.txt"
        source = str(answer["kernel_source"])
        kernel_path.write_text(
            source if source.endswith("\n") else source + "\n",
            encoding="utf-8",
        )
        payload_path.write_text(
            json.dumps(answer, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        raw_score = self._evidence_value(evidence, "raw_score", "unknown")
        summary_path.write_text(
            "GPU kernel scientific answer\n"
            f"problem_type: {answer['task']['problem_type']}\n"
            f"declared_gpu_type: {answer['hardware']['declared_gpu_type']}\n"
            f"runtime_us: {raw_score}\n"
            f"task_bundle_sha256: {answer['task']['bundle_sha256']}\n",
            encoding="utf-8",
        )
        return [str(kernel_path), str(payload_path), str(summary_path)]

    # ------------------------------------------------------------------
    def _fail(self, msg: str, logs: str = "",
              failure_kind: str = "code") -> RewardResult:
        return RewardResult(reward=self.fail_score, raw_score=None, valid=False,
                            parsed=True, ran=False, msg=msg, stdout=logs,
                            construction=[], failure_kind=failure_kind)

    def compute_reward(self, response_text: str, parent: ParentContext,
                       timeout_s: float) -> RewardResult:
        from evolve.verifier.parsing import extract_python_code
        code = extract_python_code(response_text)
        if code is None:
            return RewardResult(reward=self.fail_score, parsed=False,
                                msg="no_code_block", failure_kind="code")

        if "@triton.jit" not in code:
            return self._fail("Code must contain @triton.jit.")
        if self.problem_type == "trimul" and "identity" in code:
            return self._fail("Identity kernel is not allowed.",
                              failure_kind="constraint")

        missing = self.missing_task_files()
        if missing:
            return self._fail(
                f"task files missing: {', '.join(missing)}. task.yml lists them "
                f"but they are not in the repo. Fetch them from "
                f"github.com/gpu-mode/reference-kernels "
                f"(problems/bioml/trimul, problems/amd/mla-decode) into the "
                f"same directory as task.yml.", failure_kind="infrastructure")

        timeout = float(self.kernel_timeout_s or 0.0) or float(timeout_s or 0.0)
        if (
            not self.kernel_eval_isolation
            or self.kernel_gpu_id is None
            or timeout <= 0.0
        ):
            return self._fail(
                "kernel evaluation requires an exclusive kernel_gpu_id, "
                "kernel_eval_isolation=true, and a positive timeout",
                failure_kind="infrastructure",
            )
        out = run_eval_with_timeout(
            code,
            self.lib_dir,
            self.task_yaml,
            self.problem_type,
            self.log_chars,
            timeout,
            self.kernel_gpu_id,
        )

        if not out.get("ok"):
            msg = str(out.get("msg", "run failed"))
            msg_lower = msg.lower()
            if "timeout" in msg_lower:
                kind = "timeout"
            elif ("failed to pass test cases" in msg_lower
                  or "no passing leaderboard run" in msg_lower):
                kind = "constraint"
            elif ("could not find" in msg_lower
                  or "task files missing" in msg_lower):
                kind = "infrastructure"
            else:
                kind = "code"
            return self._fail(msg, out.get("logs", ""), failure_kind=kind)

        score_us = float(out["score_us"])
        res = RewardResult()
        res.valid = True
        res.parsed = True
        res.ran = True
        res.code = code
        res.stdout = out.get("logs", "")      # warnings survive on success too
        res.raw_score = score_us
        res.reward = (float(self.score_scale / score_us) if score_us > 0
                      else self.fail_score)
        res.construction = []
        res.msg = out.get("msg", f"runtime_us={score_us}")
        return res

    # ------------------------------------------------------------------
    def seed_states(self) -> List[SeedState]:
        if self.problem_type == "mla_decode_nvidia":
            # MLA_DECODE_INITIAL_VALUE is a measured H200 runtime. On other
            # hardware it must never be treated as evidence. If no measured
            # hint was configured, use an unscored seed: EVOLVE immediately
            # verifies the saved kernel on the isolated evaluation GPU before
            # it can enter the archive.
            if self.gpu_type.strip().lower() not in ("h200",):
                seed_us = self.cfg.get("mla_seed_runtime_us")
                if seed_us is None:
                    return [SeedState(code=MLA_DECODE_INITIAL_STATE, value=0.0,
                                      raw_score=None, construction=[])
                            for _ in range(self.num_seed_states)]
                us = abs(float(seed_us))
                reward = float(self.score_scale / us) if us > 0 else 0.0
                return [SeedState(code=MLA_DECODE_INITIAL_STATE, value=reward,
                                  raw_score=us, construction=[])
                        for _ in range(self.num_seed_states)]
            us = abs(float(MLA_DECODE_INITIAL_VALUE))
            reward = float(self.score_scale / us) if us > 0 else 0.0
            return [SeedState(code=MLA_DECODE_INITIAL_STATE, value=reward,
                              raw_score=us, construction=[])
                    for _ in range(self.num_seed_states)]
        code = ""
        if self.seed_from_reference:
            from pathlib import Path
            ref = Path(self.task_yaml).parent / "submission.py"
            if ref.exists():
                code = ref.read_text()
            else:
                print(f"[gpu_mode] seed_from_reference set but {ref} is missing; "
                      f"seeding with empty code")
        # value stays 0: the reference has not been benchmarked here, and a made
        # up number would seed the tree with a reward nothing can reproduce.
        return [SeedState(code=code, value=0.0, raw_score=None, construction=[])
                for _ in range(self.num_seed_states)]
