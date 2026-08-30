"""Bounded subprocess execution for generated Python candidates."""

from __future__ import annotations

import json
import os
import resource
import shutil
import signal
import subprocess
import sys
import tempfile
import time
from pathlib import Path
from typing import Any, Mapping, Optional


_RUNNER = r'''
import importlib.util
import json
import os
import socket
import sys
import traceback

PROGRAM_PATH = os.environ["EVOLVE_PROGRAM_PATH"]
FUNCTION_NAME = os.environ["EVOLVE_ENTRYPOINT"]
RESULTS_PATH = os.environ["EVOLVE_RESULTS_PATH"]
NETWORK_ACCESS = os.environ.get("EVOLVE_NETWORK_ACCESS") == "1"

if not NETWORK_ACCESS:
    class _BlockedSocket:
        def __init__(self, *args, **kwargs):
            raise PermissionError("network access is disabled by the EVOLVE verifier")
    socket.socket = _BlockedSocket
    socket.create_connection = _BlockedSocket

def jsonable(value):
    if hasattr(value, "tolist"):
        return jsonable(value.tolist())
    if hasattr(value, "item"):
        return jsonable(value.item())
    if isinstance(value, dict):
        return {str(key): jsonable(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [jsonable(item) for item in value]
    if value is None or isinstance(value, (str, int, float, bool)):
        return value
    raise TypeError(f"non-JSON answer value: {type(value).__name__}")

try:
    spec = importlib.util.spec_from_file_location("candidate", PROGRAM_PATH)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    answer = getattr(module, FUNCTION_NAME)()
    payload = {"ok": True, "value": jsonable(answer)}
except BaseException as exc:
    payload = {
        "ok": False,
        "error": f"{type(exc).__name__}: {exc}",
        "traceback": traceback.format_exc(),
    }
with open(RESULTS_PATH, "w", encoding="utf-8") as stream:
    json.dump(payload, stream, allow_nan=False, sort_keys=True)
'''


def _kill_process_group(process: subprocess.Popen, *, hard: bool) -> None:
    try:
        os.killpg(process.pid, signal.SIGKILL if hard else signal.SIGTERM)
    except (ProcessLookupError, PermissionError):
        pass


def _resource_limiter(*, timeout_s: float, memory_mb: int, max_file_mb: int):
    # Darwin cannot reliably lower RLIMIT_AS in a forked Python preexec hook:
    # the inherited controller mappings can already exceed the requested
    # candidate limit, causing exec itself to fail. Darwin memory is enforced
    # by the parent-side process-group RSS watchdog below instead.
    apply_address_space_limit = sys.platform != "darwin"

    def apply_limits() -> None:
        cpu_seconds = max(1, int(timeout_s) + 1)
        resource.setrlimit(resource.RLIMIT_CPU, (cpu_seconds, cpu_seconds + 1))
        if apply_address_space_limit:
            memory_bytes = int(memory_mb) * 1024 * 1024
            resource.setrlimit(resource.RLIMIT_AS, (memory_bytes, memory_bytes))
        file_bytes = int(max_file_mb) * 1024 * 1024
        resource.setrlimit(resource.RLIMIT_FSIZE, (file_bytes, file_bytes))
        resource.setrlimit(resource.RLIMIT_NOFILE, (64, 64))
    return apply_limits


def _darwin_process_group_rss_bytes(process_group_id: int) -> Optional[int]:
    """Best-effort total RSS for a spawned candidate process group on macOS."""

    ps = shutil.which("ps")
    if ps is None:
        return None
    try:
        observed = subprocess.run(
            [ps, "-axo", "pgid=,rss="],
            check=False,
            capture_output=True,
            text=True,
            timeout=1.0,
        )
    except (OSError, subprocess.SubprocessError):
        return None
    if observed.returncode != 0:
        return None
    total_kib = 0
    for line in observed.stdout.splitlines():
        fields = line.split()
        if len(fields) != 2:
            continue
        try:
            group_id, rss_kib = (int(fields[0]), int(fields[1]))
        except ValueError:
            continue
        if group_id == process_group_id:
            total_kib += max(0, rss_kib)
    return total_kib * 1024


def run_code(
    code: str,
    entrypoint: str,
    timeout_s: float,
    max_cpus: int = 1,
    *,
    memory_mb: int = 2048,
    network_access: bool = False,
    filesystem_policy: str = "temporary_only",
    diagnostics_chars: int = 16000,
) -> Mapping[str, Any]:
    """Execute a candidate without changing the controller working directory."""

    # ``none`` is the legacy problem spelling for a candidate with no mounted
    # problem data.  It executes in the same fresh temporary work directory as
    # ``temporary_only``; keeping the spelling preserves frozen verifier
    # identities while making the declared no-input policy operational.
    if filesystem_policy not in {"none", "temporary_only"}:
        raise ValueError(
            "filesystem_policy must be 'none' or 'temporary_only' for the "
            "generic subprocess sandbox"
        )
    if not isinstance(code, str) or not code.strip():
        return {"ok": False, "error": "empty candidate source", "stdout": ""}
    with tempfile.TemporaryDirectory(prefix="evolve-verifier-") as directory_name:
        directory = Path(directory_name)
        program = directory / "candidate.py"
        runner = directory / "runner.py"
        results = directory / "results.json"
        stdout_path = directory / "stdout.txt"
        stderr_path = directory / "stderr.txt"
        program.write_text(code, encoding="utf-8")
        runner.write_text(_RUNNER, encoding="utf-8")
        environment = {
            key: value
            for key, value in os.environ.items()
            if key not in {"PYTHONPATH", "PYTHONSTARTUP"}
        }
        threads = str(max(1, int(max_cpus)))
        for name in (
            "OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS",
            "NUMEXPR_NUM_THREADS", "VECLIB_MAXIMUM_THREADS", "BLIS_NUM_THREADS",
        ):
            environment[name] = threads
        environment.update(
            {
                "EVOLVE_PROGRAM_PATH": str(program),
                "EVOLVE_ENTRYPOINT": str(entrypoint),
                "EVOLVE_RESULTS_PATH": str(results),
                "EVOLVE_NETWORK_ACCESS": "1" if network_access else "0",
            }
        )
        with stdout_path.open("wb") as stdout, stderr_path.open("wb") as stderr:
            process = subprocess.Popen(
                [sys.executable, "-I", str(runner)],
                cwd=str(directory),
                env=environment,
                stdout=stdout,
                stderr=stderr,
                start_new_session=True,
                preexec_fn=_resource_limiter(
                    timeout_s=float(timeout_s), memory_mb=memory_mb, max_file_mb=16
                ),
            )
            timed_out = False
            memory_exceeded = False
            try:
                try:
                    if sys.platform != "darwin":
                        process.wait(timeout=float(timeout_s))
                    else:
                        deadline = time.monotonic() + float(timeout_s)
                        memory_limit = int(memory_mb) * 1024 * 1024
                        while process.poll() is None:
                            remaining = deadline - time.monotonic()
                            if remaining <= 0.0:
                                timed_out = True
                                break
                            rss_bytes = _darwin_process_group_rss_bytes(
                                process.pid
                            )
                            if (
                                rss_bytes is not None
                                and rss_bytes > memory_limit
                            ):
                                memory_exceeded = True
                                break
                            try:
                                process.wait(timeout=min(0.05, remaining))
                            except subprocess.TimeoutExpired:
                                pass
                except subprocess.TimeoutExpired:
                    timed_out = True
                if timed_out or memory_exceeded:
                    _kill_process_group(process, hard=False)
                    try:
                        process.wait(timeout=1.0)
                    except subprocess.TimeoutExpired:
                        _kill_process_group(process, hard=True)
                        process.wait(timeout=1.0)
            finally:
                _kill_process_group(process, hard=True)
        stdout_text = stdout_path.read_text(encoding="utf-8", errors="replace")[-diagnostics_chars:]
        stderr_text = stderr_path.read_text(encoding="utf-8", errors="replace")[-diagnostics_chars:]
        if timed_out:
            return {
                "ok": False,
                "error": f"Timeout after {timeout_s}s",
                "stdout": stdout_text,
                "stderr": stderr_text,
            }
        if memory_exceeded:
            return {
                "ok": False,
                "error": f"Memory limit exceeded ({memory_mb} MiB)",
                "stdout": stdout_text,
                "stderr": stderr_text,
            }
        if not results.is_file():
            return {
                "ok": False,
                "error": f"Process exited with code {process.returncode}",
                "stdout": stdout_text,
                "stderr": stderr_text,
            }
        try:
            payload = json.loads(results.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as exc:
            return {
                "ok": False,
                "error": f"Invalid result envelope: {exc}",
                "stdout": stdout_text,
                "stderr": stderr_text,
            }
        payload["stdout"] = stdout_text
        payload["stderr"] = stderr_text
        return payload


__all__ = ["run_code"]
