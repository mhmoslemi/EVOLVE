#!/bin/sh
# ---------------------------------------------------------------------------
# USER SETTINGS — these are the only three lines you normally need to edit.
# ---------------------------------------------------------------------------
PROBLEM="erdos"       # ac1, ac2, circle_packing, denoising, erdos,
                      # evolve_toy, gpu_mode_mla, or gpu_mode_trimul
ACTION="run"          # dry-plan = inspect safely, run = start the real run,
                      # validate = check configuration without loading a model
AVAILABLE_GPUS="0"    # Ordered physical IDs, separated by spaces or commas.
                      # Non-GPU problems: first trains; all others run vLLM.
                      # GPU mode: first trains, last evaluates, middle run vLLM.
                      # With <=2 GPUs, GPU-mode evaluation shares training.
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
# AVAILABLE_GPUS overrides that YAML's device topology only for the simple
# no-argument command. Set EVOLVE_PYTHON=/path/to/python if needed.
set -eu

project_dir=$(CDPATH= cd -- "$(dirname -- "$0")" && pwd)
cd "$project_dir"

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
# command-line arguments bypass this block and are forwarded unchanged.
if [ "$#" -eq 0 ]; then
    config_path="configs/${PROBLEM}.yaml"
    if [ ! -f "$config_path" ]; then
        echo "Unknown PROBLEM=$PROBLEM (missing $config_path)." >&2
        exit 2
    fi

    gpu_words=$(printf '%s\n' "$AVAILABLE_GPUS" | tr ',' ' ')
    gpu_count=0
    gpu_seen=""
    training_gpu=""
    evaluation_gpu=""
    for gpu_id in $gpu_words; do
        case "$gpu_id" in
            ''|*[!0-9]*)
                echo "AVAILABLE_GPUS must contain non-negative integer IDs." >&2
                exit 2
                ;;
        esac
        case " $gpu_seen " in
            *" $gpu_id "*)
                echo "AVAILABLE_GPUS contains duplicate GPU $gpu_id." >&2
                exit 2
                ;;
        esac
        gpu_seen="$gpu_seen $gpu_id"
        gpu_count=$((gpu_count + 1))
        if [ "$gpu_count" -eq 1 ]; then
            training_gpu=$gpu_id
        fi
        evaluation_gpu=$gpu_id
    done
    if [ "$gpu_count" -eq 0 ]; then
        echo "AVAILABLE_GPUS must name at least one GPU." >&2
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
            ;;
        validate)
            set -- "$@" --validate-config
            ;;
        *)
            echo "Unknown ACTION=$ACTION; use dry-plan, validate, or run." >&2
            exit 2
            ;;
    esac
fi

exec "$python_bin" train_evolve.py "$@"
