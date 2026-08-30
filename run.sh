#!/bin/sh
# ---------------------------------------------------------------------------
# USER SETTINGS — edit these values for the allocation before each run.
# ---------------------------------------------------------------------------
PROBLEM="erdos"       # ac1, ac2, circle_packing, denoising, erdos,
                      # evolve_toy, gpu_mode_mla, or gpu_mode_trimul
ACTION="run"          # dry-plan = inspect safely, run = start the real run,
                      # validate = check configuration without loading a model
CUDA_DEVICES="${EVOLVE_CUDA_DEVICES:-0}"
                      # Ordered physical IDs, separated by spaces or commas.
                      # Also becomes CUDA_VISIBLE_DEVICES; keep it consistent
                      # with the GPUs actually assigned by Slurm.
                      # Non-GPU problems: first trains; all others run vLLM.
                      # GPU mode: first trains, last evaluates, middle run vLLM.
                      # With <=2 GPUs, GPU-mode evaluation shares training.
CPU_CORES="${EVOLVE_CPU_CORES:-40}"
                      # Fresh runs override problem eval_cpus with this value.
                      # Resumes retain their immutable saved verifier setting.
TIME_LIMIT="${EVOLVE_RUN_TIME_LIMIT:-00:10}"
                      # Hard wall-clock limit from launch, exactly HH:MM.
GRACEFUL_STOP_MINUTES="${EVOLVE_GRACEFUL_STOP_MINUTES:-5}"
                      # Send SIGINT this many minutes before the hard limit.
ARTIFACT_RETENTION="${EVOLVE_ARTIFACT_RETENTION:-latest}"
                      # all = retain every role snapshot (recommended).
                      # latest = irreversibly prune older role LoRA/optimizer
                      # snapshots after a newer barrier is durable; scientific
                      # evidence and resumability stay intact.
# ---------------------------------------------------------------------------
# Examples:
#   1. Safely inspect Erdos:       PROBLEM="erdos"; ACTION="dry-plan"
#   2. Actually run Erdos:         PROBLEM="erdos"; ACTION="run"
#   3. Run MLA kernel search:      PROBLEM="gpu_mode_mla"; ACTION="run"
#   4. Then execute:               sh run.sh
#
# Advanced commands still work and override the settings above:
#   sh run.sh --config configs/ac1.yaml --dry-plan
#   sh run.sh --resume /absolute/path/to/runs/RUN_NAME
#   sh run.sh --resume /absolute/path/to/runs/RUN_NAME --num-steps 150
#
# vLLM settings, LoRA settings, and budgets live in configs/<problem>.yaml.
# CUDA_DEVICES overrides that YAML's device topology only for the simple
# no-argument command and always pins CUDA visibility. Set
# EVOLVE_PYTHON=/path/to/python if needed.
set -eu

project_dir=$(CDPATH= cd -- "$(dirname -- "$0")" && pwd)
cd "$project_dir"

case "$CPU_CORES" in
    ''|*[!0-9]*|0)
        echo "CPU_CORES must be a positive integer." >&2
        exit 2
        ;;
esac
if [ -n "${SLURM_CPUS_PER_TASK:-}" ]; then
    case "$SLURM_CPUS_PER_TASK" in
        ''|*[!0-9]*|0)
            echo "SLURM_CPUS_PER_TASK must be a positive integer when set." >&2
            exit 2
            ;;
    esac
    if [ "$CPU_CORES" -gt "$SLURM_CPUS_PER_TASK" ]; then
        echo "CPU_CORES=$CPU_CORES exceeds the Slurm allocation SLURM_CPUS_PER_TASK=$SLURM_CPUS_PER_TASK." >&2
        exit 2
    fi
fi
case "$GRACEFUL_STOP_MINUTES" in
    ''|*[!0-9]*)
        echo "GRACEFUL_STOP_MINUTES must be a non-negative integer." >&2
        exit 2
        ;;
esac
case "$ARTIFACT_RETENTION" in
    all|latest) ;;
    *)
        echo "ARTIFACT_RETENTION must be all or latest." >&2
        exit 2
        ;;
esac
case "$TIME_LIMIT" in
    [0-9][0-9]:[0-5][0-9]) ;;
    *)
        echo "TIME_LIMIT must use exactly HH:MM (for example 00:50)." >&2
        exit 2
        ;;
esac

time_hours=${TIME_LIMIT%:*}
time_minutes=${TIME_LIMIT#*:}
case "$time_hours" in
    00) time_hours_value=0 ;;
    0*) time_hours_value=${time_hours#0} ;;
    *) time_hours_value=$time_hours ;;
esac
case "$time_minutes" in
    00) time_minutes_value=0 ;;
    0*) time_minutes_value=${time_minutes#0} ;;
    *) time_minutes_value=$time_minutes ;;
esac
time_limit_seconds=$((time_hours_value * 3600 + time_minutes_value * 60))
grace_seconds=$((GRACEFUL_STOP_MINUTES * 60))
if [ "$time_limit_seconds" -le "$grace_seconds" ]; then
    echo "TIME_LIMIT must be longer than GRACEFUL_STOP_MINUTES." >&2
    exit 2
fi
soft_stop_seconds=$((time_limit_seconds - grace_seconds))

cuda_words=$(printf '%s\n' "$CUDA_DEVICES" | tr ',' ' ')
gpu_count=0
gpu_seen=""
cuda_mask=""
training_gpu=""
evaluation_gpu=""
for gpu_id in $cuda_words; do
    case "$gpu_id" in
        ''|*[!0-9]*)
            echo "CUDA_DEVICES must contain non-negative integer IDs." >&2
            exit 2
            ;;
    esac
    case " $gpu_seen " in
        *" $gpu_id "*)
            echo "CUDA_DEVICES contains duplicate GPU $gpu_id." >&2
            exit 2
            ;;
    esac
    gpu_seen="$gpu_seen $gpu_id"
    if [ -n "$cuda_mask" ]; then
        cuda_mask="$cuda_mask,$gpu_id"
    else
        cuda_mask=$gpu_id
    fi
    gpu_count=$((gpu_count + 1))
    if [ "$gpu_count" -eq 1 ]; then
        training_gpu=$gpu_id
    fi
    evaluation_gpu=$gpu_id
done
if [ "$gpu_count" -eq 0 ]; then
    echo "CUDA_DEVICES must name at least one GPU." >&2
    exit 2
fi

inherited_cuda_mask=${CUDA_VISIBLE_DEVICES:-}
if [ -n "$inherited_cuda_mask" ]; then
    case "$inherited_cuda_mask" in
        *[!0-9,]*)
            echo "Inherited CUDA_VISIBLE_DEVICES=$inherited_cuda_mask is nonnumeric; refusing to override an allocation UUID." >&2
            exit 2
            ;;
    esac
    if [ "$inherited_cuda_mask" != "$cuda_mask" ]; then
        echo "CUDA_DEVICES=$cuda_mask disagrees with inherited CUDA_VISIBLE_DEVICES=$inherited_cuda_mask; edit CUDA_DEVICES or explicitly unset the stale mask." >&2
        exit 2
    fi
fi

export CUDA_VISIBLE_DEVICES=$cuda_mask
export EVOLVE_CPU_CORES=$CPU_CORES
export EVOLVE_RUN_TIME_LIMIT=$TIME_LIMIT
export EVOLVE_GRACEFUL_STOP_MINUTES=$GRACEFUL_STOP_MINUTES
export EVOLVE_ARTIFACT_RETENTION=$ARTIFACT_RETENTION

if [ -n "${EVOLVE_PYTHON:-}" ]; then
    python_bin=$EVOLVE_PYTHON
elif [ -n "${VIRTUAL_ENV:-}" ] && [ -x "$VIRTUAL_ENV/bin/python" ]; then
    # Respect an explicitly activated virtual environment. In particular, do
    # not let a stale project-local .venv shadow the environment whose pip the
    # user just invoked.
    python_bin=$VIRTUAL_ENV/bin/python
elif [ -n "${CONDA_PREFIX:-}" ] && [ -x "$CONDA_PREFIX/bin/python" ]; then
    python_bin=$CONDA_PREFIX/bin/python
elif [ -x "$project_dir/.venv/bin/python" ]; then
    python_bin=$project_dir/.venv/bin/python
else
    # macOS /usr/bin/python3 often lacks project packages. Prefer a newer
    # user-installed Python that can at least read the YAML configuration.
    python_bin=
    for candidate in python3.12 python3.11 python3; do
        candidate_path=$(command -v "$candidate" 2>/dev/null || true)
        if [ -n "$candidate_path" ] \
            && "$candidate_path" -c 'import yaml' >/dev/null 2>&1; then
            python_bin=$candidate_path
            break
        fi
    done
    if [ -z "$python_bin" ]; then
        echo "EVOLVE needs Python 3 with PyYAML; create .venv or set EVOLVE_PYTHON." >&2
        exit 2
    fi
fi

# With no command-line arguments, use the simple settings at the top. Explicit
# command-line arguments bypass problem/topology selection, while the CUDA,
# CPU, deadline, and retention safety settings above still apply.
resume_mode=false
model_free=false
for argument in "$@"; do
    case "$argument" in
        --resume|--resume=*|--resume-from|--resume-from=*) resume_mode=true ;;
        --dry-plan|--validate-config) model_free=true ;;
    esac
done
if [ "$#" -eq 0 ]; then
    config_path="configs/${PROBLEM}.yaml"
    if [ ! -f "$config_path" ]; then
        echo "Unknown PROBLEM=$PROBLEM (missing $config_path)." >&2
        exit 2
    fi

    gpu_mode=false
    case "$PROBLEM" in
        gpu_mode|gpu_mode_*) gpu_mode=true ;;
    esac

    if [ "$gpu_count" -eq 1 ]; then
        vllm_gpus=$training_gpu
        vllm_gpu_count=1
        evaluation_gpu=$training_gpu
    elif [ "$gpu_mode" = false ]; then
        vllm_gpus=""
        vllm_gpu_count=0
        gpu_index=0
        for gpu_id in $gpu_words; do
            gpu_index=$((gpu_index + 1))
            if [ "$gpu_index" -gt 1 ]; then
                if [ -n "$vllm_gpus" ]; then
                    vllm_gpus="$vllm_gpus,$gpu_id"
                else
                    vllm_gpus=$gpu_id
                fi
                vllm_gpu_count=$((vllm_gpu_count + 1))
            fi
        done
    elif [ "$gpu_count" -eq 2 ]; then
        vllm_gpus=$evaluation_gpu
        vllm_gpu_count=1
        evaluation_gpu=$training_gpu
    else
        vllm_gpus=""
        vllm_gpu_count=0
        gpu_index=0
        for gpu_id in $gpu_words; do
            gpu_index=$((gpu_index + 1))
            if [ "$gpu_index" -gt 1 ] && [ "$gpu_index" -lt "$gpu_count" ]; then
                if [ -n "$vllm_gpus" ]; then
                    vllm_gpus="$vllm_gpus,$gpu_id"
                else
                    vllm_gpus=$gpu_id
                fi
                vllm_gpu_count=$((vllm_gpu_count + 1))
            fi
        done
    fi

    set -- \
        --config "$config_path" \
        --training-gpu-id "$training_gpu" \
        --gpu-ids "$vllm_gpus" \
        --vllm-tensor-parallel-size "$vllm_gpu_count"
    case "$PROBLEM" in
        gpu_mode|gpu_mode_*)
            set -- "$@" --kernel-gpu-id "$evaluation_gpu"
            ;;
    esac
    case "$ACTION" in
        run)
            ;;
        dry-plan)
            set -- "$@" --dry-plan
            model_free=true
            ;;
        validate)
            set -- "$@" --validate-config
            model_free=true
            ;;
        *)
            echo "Unknown ACTION=$ACTION; use dry-plan, validate, or run." >&2
            exit 2
            ;;
    esac
fi

if [ "$resume_mode" = false ]; then
    set -- "$@" --eval-cpus "$CPU_CORES"
fi

if [ "$model_free" = true ]; then
    exec "$python_bin" train_evolve.py "$@"
fi

launch_epoch=$(date +%s)
hard_deadline_epoch=$((launch_epoch + time_limit_seconds))
export EVOLVE_HARD_DEADLINE_EPOCH=$hard_deadline_epoch

echo "EVOLVE · run guard · CUDA_VISIBLE_DEVICES=$CUDA_VISIBLE_DEVICES · CPU cores=$CPU_CORES · graceful stop=T-${GRACEFUL_STOP_MINUTES}m · hard limit=$TIME_LIMIT · retention=$ARTIFACT_RETENTION"

# Start EVOLVE in a new session so the emergency hard stop can target this run
# and all of its descendants without touching unrelated jobs or processes.
"$python_bin" -c '
import os
import sys
os.setsid()
os.execv(sys.argv[1], sys.argv[1:])
' "$python_bin" train_evolve.py "$@" &
run_pid=$!

# The controller gets the full grace window to drain branches, persist status
# and logs, and unload vLLM/HF. At the absolute deadline, only this run session
# is force-stopped if any descendant remains.
"$python_bin" -c '
import os
import signal
import sys
import time

pid = int(sys.argv[1])
soft_seconds = int(sys.argv[2])
grace_seconds = int(sys.argv[3])
limit = sys.argv[4]
time.sleep(soft_seconds)
try:
    os.kill(pid, signal.SIGINT)
except ProcessLookupError:
    raise SystemExit(0)
print(
    f"EVOLVE · run guard · graceful stop requested; {grace_seconds // 60} minutes remain before hard limit {limit}",
    file=sys.stderr,
    flush=True,
)
time.sleep(grace_seconds)
try:
    os.killpg(pid, 0)
except ProcessLookupError:
    raise SystemExit(0)
print(
    f"EVOLVE · run guard · hard limit {limit} reached; force-stopping this run process group",
    file=sys.stderr,
    flush=True,
)
os.killpg(pid, signal.SIGKILL)
' "$run_pid" "$soft_stop_seconds" "$grace_seconds" "$TIME_LIMIT" &
watchdog_pid=$!

forward_interrupt() {
    kill -INT "$run_pid" 2>/dev/null || true
}

terminate_run() {
    kill "$watchdog_pid" 2>/dev/null || true
    kill -TERM -- "-$run_pid" 2>/dev/null || true
}

trap forward_interrupt INT
trap terminate_run HUP TERM

# A signal can interrupt wait(1) while Python is still draining. Keep waiting
# until the exact controller PID exits so the allocation is not left with live
# model or verifier children.
while :; do
    if wait "$run_pid"; then
        run_status=0
        break
    else
        run_status=$?
    fi
    if kill -0 "$run_pid" 2>/dev/null; then
        continue
    fi
    break
done

kill "$watchdog_pid" 2>/dev/null || true
if wait "$watchdog_pid" 2>/dev/null; then
    :
fi
trap - INT HUP TERM
exit "$run_status"
