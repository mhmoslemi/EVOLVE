"""CPU-only enforcement checks for the generated-code subprocess boundary."""

from __future__ import annotations

import os
import signal

from evolve.verifier.sandbox import run_code


def test_sandbox_uses_disposable_working_directory_and_legacy_none_policy():
    controller_cwd = os.getcwd()
    result = run_code(
        "def answer():\n"
        "    import os\n"
        "    return {'cwd': os.getcwd(), 'threads': os.environ['OMP_NUM_THREADS']}\n",
        "answer",
        timeout_s=1.0,
        max_cpus=2,
        memory_mb=128,
        filesystem_policy="none",
    )

    assert result["ok"] is True
    assert result["value"]["cwd"] != controller_cwd
    assert result["value"]["threads"] == "2"
    assert not os.path.exists(result["value"]["cwd"])
    assert os.getcwd() == controller_cwd


def test_sandbox_denies_python_socket_creation_when_network_is_disabled():
    result = run_code(
        "def answer():\n"
        "    import socket\n"
        "    try:\n"
        "        socket.socket()\n"
        "    except Exception as exc:\n"
        "        return type(exc).__name__\n"
        "    return 'network-was-not-blocked'\n",
        "answer",
        timeout_s=1.0,
        memory_mb=128,
        network_access=False,
    )

    assert result["ok"] is True
    assert result["value"] == "PermissionError"


def test_sandbox_timeout_terminates_spawned_process_group():
    result = run_code(
        "def answer():\n"
        "    import subprocess, sys, time\n"
        "    child = subprocess.Popen(\n"
        "        [sys.executable, '-c', 'import time; time.sleep(30)']\n"
        "    )\n"
        "    print(child.pid, flush=True)\n"
        "    time.sleep(30)\n",
        "answer",
        timeout_s=0.25,
        memory_mb=128,
    )

    assert result["ok"] is False
    assert result["error"] == "Timeout after 0.25s"
    child_pid = int(result["stdout"].strip())
    try:
        os.kill(child_pid, 0)
    except ProcessLookupError:
        pass
    else:  # Ensure a failed assertion never leaves the fixture process alive.
        os.kill(child_pid, signal.SIGKILL)
        raise AssertionError("sandbox timeout left a descendant process alive")


def test_sandbox_bounds_memory_and_diagnostic_capture():
    memory_result = run_code(
        "def answer():\n"
        "    blocks = []\n"
        "    while True:\n"
        "        blocks.append(bytearray(16 * 1024 * 1024))\n",
        "answer",
        timeout_s=3.0,
        memory_mb=96,
    )
    diagnostics_result = run_code(
        "def answer():\n"
        "    print('x' * 5000)\n"
        "    return 1\n",
        "answer",
        timeout_s=1.0,
        memory_mb=128,
        diagnostics_chars=64,
    )

    assert memory_result["ok"] is False
    assert any(
        marker in memory_result["error"]
        for marker in ("Memory limit exceeded", "MemoryError", "Process exited")
    )
    assert diagnostics_result["ok"] is True
    assert diagnostics_result["value"] == 1
    assert len(diagnostics_result["stdout"]) == 64
    assert diagnostics_result["stdout"].endswith("\n")
