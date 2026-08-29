#!/bin/sh
# EVOLVE launcher. Configuration lives in configs/*.yaml; explicit command-line
# options override YAML values. No CUDA mask or model choice is hidden here.
set -eu

project_dir=$(CDPATH= cd -- "$(dirname -- "$0")" && pwd)
cd "$project_dir"

if [ -n "${EVOLVE_PYTHON:-}" ]; then
    python_bin=$EVOLVE_PYTHON
elif [ -x "$project_dir/.venv/bin/python" ]; then
    python_bin=$project_dir/.venv/bin/python
else
    python_bin=python3
fi

# With no arguments, print a model-free plan for the small deterministic
# configuration. Remove --dry-plan when supplying a real model configuration.
# Common examples:
#   sh run.sh --config configs/erdos.yaml
#   sh run.sh --resume /absolute/path/to/runs/RUN_NAME
#   sh run.sh --resume /absolute/path/to/runs/RUN_NAME --num-steps 150
if [ "$#" -eq 0 ]; then
    set -- --config configs/evolve_toy.yaml --dry-plan
fi

exec "$python_bin" train_evolve.py "$@"
