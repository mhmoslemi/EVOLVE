# EVOLVE

EVOLVE searches independently verified scientific states under a finite compute
budget. It uses a quality-diversity archive, posterior allocation for expected
record improvement, three isolated role adapters, matched causal audits,
bounded refinement, and barrier-only max-seeking learning.

## Start here

Set `PROBLEM`, `ACTION`, `CUDA_DEVICES`, `CPU_CORES`, `TIME_LIMIT`, and the
retention policy at the top of `run.sh`. With no arguments, the launcher
applies those settings:

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
Use `EVOLVE_PYTHON=/path/to/python` to choose an interpreter. Explicit command
line arguments bypass simple problem/topology selection, while the launcher
guard settings still apply. Advanced runs use their explicit topology; resumes
use saved topology, which must agree with `CUDA_DEVICES`.

`CUDA_DEVICES="0"` both exports `CUDA_VISIBLE_DEVICES=0` and shares that GPU
sequentially. For ordinary scientific
problems, the first ID is used for HF/Unsloth learning and every remaining ID
forms the vLLM tensor-parallel set; their CPU verifiers reserve no GPU. Only
GPU-mode problems reserve an evaluation device: the last ID when three or more
are available, or the training device with serialized evaluation when one or
two are available. The launcher rejects nonnumeric or duplicate CUDA IDs, and
the runtime fails closed if the exported mask disagrees with resolved topology.

`CPU_CORES` overrides `eval_cpus` for fresh runs and also appears in the saved
run-guard metadata. A resume keeps its immutable saved verifier core count.
`TIME_LIMIT` uses exact `HH:MM` wall-clock syntax. Five minutes before the hard
deadline (configurable with `GRACEFUL_STOP_MINUTES`), the launcher sends a
graceful interrupt so EVOLVE can drain workers, save live status and logs, and
release vLLM/HF. At the deadline it force-stops only that run's isolated process
group if cleanup has not finished.

`ARTIFACT_RETENTION="all"` is the recommended evidence-complete default.
`ARTIFACT_RETENTION="latest"` prunes older role LoRA directories and their
optimizer files only after a newer completed barrier is durable. It preserves
the latest resumable role snapshot, all scientific evidence and logs, immutable
summaries, JSON checkpoints, and checkpoint training/RNG companions. Cleanup
plans and results are recorded under the run's `logs/` directory.

Before a real run imports the model runtime, EVOLVE prints a resolved startup
banner covering the model, sampling, role-LoRA learning, GPU placement, search
budget, reservations, and reproducibility settings. Validation and dry-plan
commands remain machine-readable JSON without the banner.

For vLLM, EVOLVE starts one model sharded across all generation `gpu_ids`, and
requires `vllm_tensor_parallel_size == len(gpu_ids)`. An explicit
`training_gpu_id` pins the HF/Unsloth backbone to that one device. HF/Unsloth is
released before vLLM starts, and vLLM is shut down before learning resumes.
`load_in_4bit` applies only to the HF/Unsloth training loader and is never
forwarded to vLLM. A pre-quantized BitsAndBytes checkpoint is rejected with
multi-GPU tensor parallelism; use a native/MXFP4 base there.
`vllm_quantization` is the independent inference setting; the gpt-oss examples
use `auto` so vLLM reads checkpoint-native MXFP4.

Kernel configurations normally reserve the last GPU in the user's ordered list
as `kernel_gpu_id`; physical CUDA numbers need not be numerically increasing.
With two GPUs evaluation shares the training GPU while vLLM uses the other
device. With one GPU, EVOLVE shuts down the model phase and releases its
CUDA allocations before each isolated, spawned benchmark; this is safe but
incurs model-reload overhead. GPU-mode YAMLs must declare the real `gpu_type`
because benchmark targets and scientific evidence depend on the hardware.
Other problems default to `auto`; their CPU verifiers are hardware-independent.

Install the GPU runtime overlay only inside a CUDA/PyTorch environment that
matches the machine:

```sh
python -m pip install -r requirements/requirements-evolve-gpu.txt
```

For Unsloth, follow its CUDA/PyTorch-specific installer when it recommends a
more specific command than the generic overlay.

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
