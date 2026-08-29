#!/bin/sh
# ---------------------------------------------------------------------------
# USER SETTINGS — these are the only two lines you normally need to edit.
# ---------------------------------------------------------------------------
PROBLEM="erdos"       # ac1, ac2, circle_packing, denoising, erdos,
                      # evolve_toy, gpu_mode_mla, or gpu_mode_trimul
ACTION="dry-plan"     # dry-plan = inspect safely, run = start the real run,
                      # validate = check configuration without loading a model
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
# GPU assignments, vLLM settings, LoRA settings, and budgets live in the chosen
# configs/<problem>.yaml file. Set EVOLVE_PYTHON=/path/to/python if needed.
set -eu

project_dir=$(CDPATH= cd -- "$(dirname -- "$0")" && pwd)
cd "$project_dir"

if [ -n "${EVOLVE_PYTHON:-}" ]; then
    python_bin=$EVOLVE_PYTHON
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
    case "$ACTION" in
        run)
            set -- --config "$config_path"
            ;;
        dry-plan)
            set -- --config "$config_path" --dry-plan
            ;;
        validate)
            set -- --config "$config_path" --validate-config
            ;;
        *)
            echo "Unknown ACTION=$ACTION; use dry-plan, validate, or run." >&2
            exit 2
            ;;
    esac
fi

exec "$python_bin" train_evolve.py "$@"
