# EVOLVE

EVOLVE searches independently verified scientific states under a finite compute
budget. It uses a quality-diversity archive, posterior allocation for expected
record improvement, three isolated role adapters, matched causal audits,
bounded refinement, and barrier-only max-seeking learning.

## Start here

`run.sh` contains no hidden model or GPU settings. With no arguments it prints
a model-free plan for the CPU toy configuration:

```sh
sh run.sh
```

Validate or inspect a real problem configuration without loading a model:

```sh
sh run.sh --config configs/erdos.yaml --validate-config
sh run.sh --config configs/gpu_mode_trimul.yaml --dry-plan
```

Start a configured run or resume the last completed barrier:

```sh
sh run.sh --config configs/erdos.yaml
sh run.sh --resume /absolute/path/to/runs/RUN_NAME
sh run.sh --resume /absolute/path/to/runs/RUN_NAME --num-steps 150
```

`--num-steps` is a compatibility spelling for the total target epoch count.
Use `EVOLVE_PYTHON=/path/to/python` to choose an interpreter. GPU visibility is
controlled explicitly by the YAML `gpu_ids` field and the caller's environment.

## Configuration

Every file in `configs/` is an EVOLVE configuration. The resolution order is:

1. typed defaults;
2. problem YAML;
3. the resumed run's immutable resolved configuration;
4. explicit CLI overrides.

The nested `evolve` section controls budgets, archive capacity, roles, options,
harnesses, posterior allocation, audits, learning, refinement, reporting, and
worker backpressure. Problem-specific fields stay at the top level.

## Runtime artifacts

Runs are created under `runs/`. Raw responses, parsed proposals, verification
evidence, failures, branch outcomes, audit assignments, policy traces, learning
inputs, archive state, causal memory, role checkpoints, status, best answers,
and completion markers are retained. Resume selects only a checkpoint referenced
by a durable completed-barrier marker.

Existing `runs/` directories are user evidence and are never modified unless
the user explicitly resumes one.

## Repository map

- `train_evolve.py`: command-line entrypoint
- `evolve/engine.py`: epoch lifecycle and atomic barriers
- `evolve/archive/`: scientific states, cells, and provenance
- `evolve/scheduler/`: posterior record-value allocation
- `evolve/roles/`: isolated role adapters and backbone ownership
- `evolve/options/`, `evolve/harness/`: frozen executable branches
- `evolve/verifier/`: saved-payload verification and sandboxing
- `evolve/audits/`, `evolve/causal_memory/`: randomized causal evidence
- `evolve/learning/`: homogeneous barrier updates
- `evolve/refinement/`: bounded challenger nursery
- `evolve/runio/`, `evolve/reporting/`, `evolve/viz/`: artifacts and reporting
- `problems/`: one typed scientific problem implementation per problem
- `configs/`: one readable EVOLVE YAML per problem or subtype

The authoritative method and implementation contract is [AGENTS.md](AGENTS.md).
