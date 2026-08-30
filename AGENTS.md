# EVOLVE server handoff and active repository contract

## Authority and required reading

This is the active root instruction file for the repository. It is both the
implementation handoff and the remaining-readiness procedure.

Before changing code or launching anything, read these files completely:

1. `AGENTS.md` — this active handoff;
2. `AGENTS-old.md` — the original 397-line EVOLVE method and implementation
   contract, preserved verbatim;
3. `docs/EVOLVE_IMPLEMENTATION.md` — the live phase checklist, decisions,
   schema versions, validation record, and remaining gaps;
4. `README.md` and the selected `configs/*.yaml` before using `run.sh`.

Every scientific, persistence, configuration, resource, and compatibility
invariant in `AGENTS-old.md` remains binding. This handoff records what is now
implemented and gives the exact remaining GPU procedure; it does not relax or
replace the method. If anything is ambiguous, preserve scientific validity,
append-only evidence, deterministic resume, and user-owned runs.

## Current completion statement

EVOLVE phases 0 through 10 are implemented and pass the complete CPU/fake
end-to-end suite. Phase 11 and the Definition of Done remain open only because
the real Qwen/vLLM hardware smoke has not been run.

Do not claim the project fully complete until the GPU smoke in this file passes
and its evidence is recorded in `docs/EVOLVE_IMPLEMENTATION.md`. Do not claim
that EVOLVE improves scientific discovery from architecture, unit tests, fake
workers, or a runtime smoke test.

The implementation baseline immediately before this handoff document was:

- branch: `master`;
- commit: `aac8a5b`;
- subject: `feat(evolve): harden runtime and complete CPU readiness gates`;
- date of recorded validation: 2026-08-29;
- validation host: macOS, Python 3.11, CPU-only fixtures and fake workers.

The commit containing this handoff will naturally be newer. On the server,
confirm that both `AGENTS.md` and `AGENTS-old.md` arrived in the same pulled
commit before using the instructions below.

## Repository contract that must remain intact

- This repository is EVOLVE-only. Never introduce a hidden legacy, PUCT, Elo,
  LLM-judge, or self-likelihood fallback.
- `train_evolve.py` is the active entrypoint. `run.sh` must keep forwarding its
  final `"$@"` unchanged.
- Configuration precedence is defaults, problem YAML, resumed resolved config,
  then explicit CLI flags. The final layer wins. `gpu_ids` is authoritative.
- Keep the typed problem API in `problems/base.py`, registration in
  `problems/registry.py`, and the eight existing YAMLs in `configs/`.
- Never edit, normalize, rename, delete, or backfill an existing directory in
  `runs/`. A new smoke must create a new run. Resume only a run explicitly
  named with `--resume`.
- Preserve all legacy compatibility files and readers, existing problem
  implementations, multi-worker generation, concurrent verification,
  completed-barrier resume, status, best-answer mirrors, and plotting tools.
- Persist rendered prompts, raw responses, parsed proposals, verifier attempts,
  invalids, duplicates, refinements, audit assignments, and learning inputs
  before any consumer or gradient update uses them.
- Never launch a long model run, multi-GPU job, GPU scientific benchmark, or
  expensive experiment without a new explicit user authorization and agreed
  topology. The procedure below authorizes only a deliberately tiny smoke when
  the user explicitly asks the server agent to execute it.

## Non-negotiable scientific invariants

- The objective is the best independently verified admissible record under a
  finite budget, not mean reward, likelihood, archive rank, novelty, or visits.
- A scientific state is identified from its canonical saved answer payload and
  independent evidence. Proposal source hashes are diagnostics only.
- The global archive is descriptor-indexed quality diversity with append-only
  provenance. Local option trees never replace the global scheduler.
- There is one frozen backbone and exactly three isolated role LoRAs:
  `scout`, `mechanist`, and `challenger`. They have separate adapter state,
  optimizer state, RNG, retrieval view, transcripts, and learning groups.
- An allocation arm freezes cell, role, executable option, harness version,
  horizon, and cost class. Harnesses are versioned arms, never hidden judges.
- Only the common verifier admits candidates or changes the record. Confirm a
  possible record from the saved answer payload; never rerun stochastic source.
- Infrastructure failure stays unresolved and is excluded from scientific
  admission, gain tails, causal effects, and learning.
- Allocation uses the hierarchical zero-inflated record-gain posterior and a
  correlation-aware joint-maximum portfolio with mandatory audit, role,
  empty/under-tested-cell, harness, and exploration reservations.
- Production evidence may update the archive and scheduler but never causal
  memory. Causal option memory is promoted only from closed, preassigned,
  randomized matched audit pairs with conservative repeated support.
- Barrier learning uses persisted homogeneous on-policy groups and exact
  OrderGrad top-m-at-K, or the explicitly selected centered MaxPO objective.
  Never mix roles, snapshots, contexts, options, harnesses, horizons, budgets,
  generation settings, audit halves, production, or refinement channels.
- Refinement is Challenger-only, minimal, bounded, blinded, TTL-limited, and
  separated from production learning.
- Every epoch freezes archive, threshold, scheduler, role adapters, memory,
  options, harnesses, verifier, descriptors, fingerprints, and schemas. New
  state becomes visible only after the atomic completed barrier.
- `allocation_plan.json` is authoritative for partial-epoch replay. Resume from
  the newest valid completed marker, reuse durable sample IDs, and never double
  count. Malformed or conflicting durable evidence fails closed.

## Implemented system map

The implementation is additive through `train_evolve.py` and `evolve/`:

- `evolve/config.py`, `types.py`, `ids.py`, `budget.py`, and `cli.py` provide
  strict schema-v1 configuration, canonical IDs, immutable typed records,
  idempotent budgets, validation, dry planning, and startup reporting.
- `evolve/archive/` implements descriptors, fingerprints, cells, champions,
  promising descendants, stepping stones, record tracking, and provenance.
- `evolve/roles/` owns the one backbone, three named LoRAs, optimizers, RNGs,
  role snapshots, adapter manifests, restore logic, and private working memory.
- `evolve/options/` contains executable bounded state machines and frozen
  branches; `evolve/harness/` contains immutable content-addressed harnesses.
- `evolve/scheduler/` implements hierarchical admission/reliability/tail/cost
  posteriors, correlation-aware Monte Carlo portfolio selection, horizon arms,
  resource limits, and mandatory reservations.
- `evolve/audits/` and `evolve/causal_memory/` implement persisted matched
  assignments, effects, promotion, quarantine, drift, and no-memory controls.
- `evolve/learning/` implements strict group construction, independently tested
  OrderGrad/MaxPO objectives, per-role updates, masks, KL, and optimizer logs.
- `evolve/refinement/` implements the bounded Challenger nursery and its
  separate randomized audit/learning channel.
- `evolve/workers/` implements deterministic generation jobs, exact sample
  counts, bounded concurrency, generation-verification overlap, backpressure,
  resource leases, HF OOM microbatch fallback, live HF/vLLM phase switching,
  and graceful shutdown.
- `evolve/verifier/` implements saved-payload verification, confirmation,
  evidence identity, retries, bounded captures, subprocess isolation, and
  failure classification.
- `evolve/runio/` implements exclusive layouts, atomic writes, monotonic events,
  immutable metadata, checkpoints, schema detection, and recovery.
- `evolve/engine.py` composes bootstrap, frozen epoch planning, streamed branch
  execution, audits, harness trials, refinement, record confirmation, learning,
  atomic barriers, recovery, and compatibility artifacts.
- `evolve/reporting/` and `evolve/viz/` provide live status, best answers, and
  nine headless artifact-only plots. They never rerun candidate code by default.
- `problems/evolve_toy.py` is the deterministic method-complete CPU fixture.
  The original scientific problem implementations remain registered and intact.

## Completed A-to-Z phases

### Phase 0 — evidence compatibility

Strict EVOLVE schema detection, configuration precedence, immutable run
attachment, rollout compatibility files, score direction, best tracking,
checkpoint pointers, and the deterministic toy fixture are operational.

### Phase 1 — foundations

Schema-v1 typed records, canonical content IDs, strict nested configuration,
idempotent budgets/events, atomic persistence, initial/resume manifests,
`--validate-config`, `--dry-plan`, and engine dispatch are operational.

### Phase 2 — scientific evidence

Saved-payload verification, confirmation, descriptors, fingerprints,
descriptor cells, local competition, confirmed record tracking, complete
captures, and global provenance are operational.

### Phase 3 — role isolation

Exactly three role adapters, independent logical optimizers/RNGs, stable policy
snapshots, immutable adapter artifacts, restore, vLLM request separation,
bounded signed-int32 LoRA IDs, and collision rejection are operational.

### Phase 4 — options and harnesses

Registered executable option state machines, initiation/stop rules,
horizon-scaled hard costs, immutable branch specifications, intermediate
evidence, content-addressed harnesses, and matched harness trials are
operational.

### Phase 5 — record allocation

Hierarchical admission, reliability, positive-gain tail and resource models,
backoff, uncertain costs, correlated joint-max selection, homogeneous replicas,
mandatory reservations, and reproducible allocation logs are operational.

### Phase 6 — causal memory

Persisted randomization, matched option audits, common randomness, closed and
aborted pairs, normalized effects, support/uncertainty, promotion, quarantine,
drift, contextual retrieval, and permanent no-memory audits are operational.

### Phase 7 — role learning

Persisted exact-K homogeneous groups, complete same-role policy traces, token
IDs/masks/log probabilities, exact OrderGrad and optional MaxPO, per-role
gradient steps, KL, gradient norms, adapter hashes, and optimizer state are
operational.

### Phase 8 — refinement

Challenger-only minimal repairs, fresh equal-cost controls, blinded verification,
no re-entry, attempts/depth/TTL/cost bounds, failed-repair persistence, and
separate refinement groups are operational.

### Phase 9 — composition and recovery

Frozen manifests, bounded asynchronous branch execution, durable arrival and
verification records, authoritative partial plans, exact retry settlement,
deterministic record confirmation, atomic barriers, role optimizer/RNG
companions, graceful draining, and completed-marker recovery are operational.

### Phase 10 — reporting

Atomic live status, immutable best snapshots and pointer, compatibility best
files, answer rendering, append-only archive/memory streams, and artifact-only
record/archive/provenance/allocation/audit/role/posterior/failure/resource plots
are operational for active and completed runs.

### Phase 11 — readiness

All CPU, fake end-to-end, concurrency, resume, corruption, artifact, plotting,
and static gates pass. Only the real model/runtime boundary below remains.

## Important compatibility and correctness fixes already present

- `run.sh` accepts an ordered GPU list. The first GPU trains; remaining GPUs
  run vLLM. GPU-mode evaluation uses the last GPU with three or more devices.
  With one or two devices, sharing is serialized safely.
- `run.sh` selects the explicitly requested/activated Python environment before
  falling back to a project or discoverable Python, preventing the earlier
  interpreter/NumPy mismatch.
- vLLM 0.28 compatibility is implemented: unsupported `swap_space` is not sent;
  the OpenAI schema requirement is pinned; FlashInfer sampling is disabled by
  default when `nvcc` is unavailable; LoRA IDs fit signed int32; and collisions
  fail instead of aliasing adapters.
- The startup banner prints model, sampling, LoRA, topology, budgets, reserves,
  and reproducibility settings before a real model import.
- Ordinary CPU-verified problems default `gpu_type` to `auto`; GPU benchmark
  problems require explicit real hardware identity.
- Erdos bare seed constructions serialize correctly, eight seeds can populate
  the archive, and total bootstrap rejection fails clearly instead of allowing
  empty epochs.
- Complete verifier-owned captures retain identity unless a retry index changes.
- GPU-mode verification uses three bounded repeats, standard error, conservative
  runtime, clean empty-log handling, and persisted exclusive-GPU identity.
- The generic sandbox accepts legacy `none` as a fresh no-input temporary
  workspace. Linux keeps `RLIMIT_AS`; macOS uses a process-group RSS watchdog.
- Harness trial records have a JSON-safe projection.
- Scheduler reservation satisfaction considers every label on an arm, so role,
  learning, empty-cell, and exploration reservations can overlap correctly.
- Allocation capacity validation now reports explicit reservation overflow.
- Artifact-only allocation plotting suppresses only the known harmless
  tight-layout warning.
- The composed resume fixture uses enough verifier budget to avoid mistaking a
  legitimate run-derived budget stop for a recovery failure.

## Completed CPU and model-free validation

These results were obtained after the implementation and fixes above:

- complete suite: **349 passed in 12.10 seconds**;
- skipped tests: **0**;
- warnings: **0**;
- compilation: passed;
- POSIX `run.sh` syntax: passed;
- `git diff --check`: passed;
- all eight shipped YAMLs: **16/16** validate/dry-plan invocations passed;
- problem registry audit: all YAMLs instantiate the expected problem/subtype,
  expose every required scientific hook, and declare the expected resources;
- expanded CLI/engine/sandbox boundary slice: **22 passed**;
- completed-barrier resume was repeated three additional consecutive times in
  5.34, 5.39, and 5.92 seconds.

The complete suite command was:

```sh
PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 /opt/homebrew/bin/python3.11 \
  -m pytest evolve/tests -q -p no:cacheprovider
```

The bare `python` on the validation Mac was Python 2.7. Python 3.11 was selected
explicitly. Pytest plugin auto-loading was disabled because an unrelated
globally installed Hydra plugin crashed pytest startup; the EVOLVE suite does
not require that plugin.

Coverage includes:

- config precedence, hashing, strict unknown-key/range rejection, resume
  overrides, and model-free CLI behavior;
- schema round-trips, canonical IDs, immutable typed state, unknown-field
  preservation, and future-schema rejection;
- idempotent budget/event persistence, torn tails, ownership, atomic writes,
  exclusive run creation, and immutable metadata;
- common-verifier identity, saved payloads, confirmation, timeouts,
  infrastructure classification, descriptors, fingerprints, archive cells,
  records, and provenance;
- three-role isolation, adapter relocation/restore, optimizer/RNG state,
  request non-aliasing, exact sample counts, worker-topology-independent seeds,
  reversed completion routing, and HF/vLLM mocked non-leakage;
- executable bounded options, immutable harnesses, record-gain posteriors,
  reservations, correlation-aware portfolios, and reproducible plans;
- randomized option/harness/refinement audits, causal effects, promotion,
  quarantine, drift, no-memory controls, and separate refinement groups;
- exact OrderGrad identities and homogeneous on-policy learning constraints;
- fresh fake bootstrap/epoch, confirmed best publication, three concurrent
  branch workers, homogeneous learning, audits/harness trials, graceful
  shutdown, completed resume, and immutable old summaries;
- partial-epoch crash and exact durable replay from the unchanged allocation
  plan without changing the already durable sample;
- malformed/future completion markers, unsafe checkpoint paths, checkpoint
  hash mismatch, incomplete/missing/corrupt training companions, and duplicate
  committed epochs all failing closed;
- every JSON and nonblank JSONL artifact parsing, no leftover temporary files,
  and all nine plots from both active-interrupted and completed-resumed runs;
- disposable sandbox working directories, legacy filesystem policy, Python
  socket denial, timeout, spawned-process-group cleanup, memory enforcement,
  and bounded diagnostic capture;
- live `run.sh` validation forwarding and every documented README command
  parsing against the current CLI without creating or changing a run.

No model was loaded, no vLLM server started, no CUDA kernel ran, no real
training occurred, and no user-owned run was modified during these gates.

## Remaining GPU readiness gate

### Purpose

The remaining gate must exercise the boundary that fakes cannot prove:

1. installed CUDA/PyTorch/Transformers/PEFT/bitsandbytes/vLLM/OpenAI-schema
   compatibility;
2. actual ordered GPU topology and `CUDA_VISIBLE_DEVICES` handling;
3. creation and persistence of all three real role LoRAs;
4. vLLM loading the one backbone and the correct role adapter per request;
5. bounded signed-int32 vLLM LoRA IDs with no collision or role leakage;
6. real generation followed by full vLLM shutdown and HF reload;
7. at least one homogeneous OrderGrad role update with a durable optimizer;
8. barrier checkpoint, status, best artifacts, and artifact-only plots;
9. real adapter/optimizer/RNG restore on one short completed-barrier resume.

This is a runtime compatibility test, not a scientific experiment or benchmark.
Use `evolve_toy`, `Qwen/Qwen3-0.6B`, one physical GPU, one fresh epoch, and one
resumed epoch. Do not use `erdos`, `gpu_mode_*`, multiple GPUs, or a large model
for this gate.

### Authorization boundary

Do not execute the GPU commands merely because this file exists. The server
agent may run model-free preflight and dry-plan commands, but must wait until
the user explicitly says to run the tiny GPU smoke. That permission does not
authorize a multi-GPU run, a benchmark problem, a longer run, or dependency
changes in a shared environment.

### Server preflight

Use the exact Python environment intended for the real run. If the environment
is shared and packages need installation or upgrades, report that and obtain
permission before mutating it. The overlay is
`requirements/requirements-evolve-gpu.txt` and currently expects vLLM 0.28.x
and `openai>=2.25.0,<3`.

```sh
cd /absolute/path/to/TTT-small
git status --short
nvidia-smi
/absolute/path/to/python -m pip check
/absolute/path/to/python - <<'PY'
import importlib.metadata as metadata
import torch

for package in (
    "torch", "transformers", "peft", "accelerate", "bitsandbytes",
    "vllm", "openai", "pyyaml",
):
    print(f"{package}: {metadata.version(package)}")
print(f"cuda_available: {torch.cuda.is_available()}")
print(f"cuda_device_count: {torch.cuda.device_count()}")
for index in range(torch.cuda.device_count()):
    print(f"gpu[{index}]: {torch.cuda.get_device_name(index)}")
PY
```

Stop before the smoke if `pip check` fails, CUDA is unavailable, no device is
visible, the selected Python differs from the environment where packages were
installed, vLLM is not 0.28.x, or OpenAI is older than 2.25. The OpenAI package
is a schema dependency used internally by vLLM; the backbone remains Qwen.

Optionally repeat the already-passing CPU gate on the Linux server before using
the GPU:

```sh
PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 /absolute/path/to/python \
  -m pytest evolve/tests -q -p no:cacheprovider
/absolute/path/to/python -m compileall -q evolve problems train_evolve.py
sh -n run.sh
```

### Model-free validation of the exact smoke configuration

The command below was dry-planned successfully on the development machine. It
does not load a model and must report `model_loading: false`,
`writes_run_directory: false`, one generation GPU, training GPU 0, three roles,
one epoch, horizon 1, group K=4, and Qwen3-0.6B.

```sh
EVOLVE_PYTHON=/absolute/path/to/python sh run.sh \
  --config configs/evolve_toy.yaml \
  --model-name Qwen/Qwen3-0.6B \
  --backend hf \
  --generation-backend vllm \
  --load-in-4bit \
  --training-gpu-id 0 \
  --gpu-ids 0 \
  --num-gpus 1 \
  --vllm-tensor-parallel-size 1 \
  --vllm-max-num-seqs 4 \
  --vllm-max-num-batched-tokens 2048 \
  --max-seq-length 2048 \
  --max-new-tokens 128 \
  --num-seed-states 4 \
  --epochs 1 \
  --verifier-calls 4096 \
  --max-horizon 1 \
  --branch-budget 8 \
  --group-k 4 \
  --max-inflight-branches 12 \
  --dry-plan
```

If physical GPU 0 is not the intended device, replace every topology occurrence
of `0` consistently. Do not leave a contradictory `CUDA_VISIBLE_DEVICES`; unset
it or make it exactly match EVOLVE's resolved ordered physical list.

### Fresh one-epoch GPU smoke

After explicit user authorization, run the same command with only
`--dry-plan` removed:

```sh
EVOLVE_PYTHON=/absolute/path/to/python sh run.sh \
  --config configs/evolve_toy.yaml \
  --model-name Qwen/Qwen3-0.6B \
  --backend hf \
  --generation-backend vllm \
  --load-in-4bit \
  --training-gpu-id 0 \
  --gpu-ids 0 \
  --num-gpus 1 \
  --vllm-tensor-parallel-size 1 \
  --vllm-max-num-seqs 4 \
  --vllm-max-num-batched-tokens 2048 \
  --max-seq-length 2048 \
  --max-new-tokens 128 \
  --num-seed-states 4 \
  --epochs 1 \
  --verifier-calls 4096 \
  --max-horizon 1 \
  --branch-budget 8 \
  --group-k 4 \
  --max-inflight-branches 12
```

The high verifier limit is a ceiling, not a requirement to spend 4096 calls.
Epoch count, horizon, and inflight capacity keep this smoke small while ensuring
that mandatory reservations and a K=4 learning group cannot be blocked merely
by an artificially tight fixture budget.

Do not reuse or delete a failed smoke run. Preserve it, diagnose from its
artifacts, and use a new run for a corrected retry.

### Fresh-run pass criteria

Record the new absolute run directory printed by EVOLVE as `SMOKE_RUN_DIR`.
The first half passes only if all of the following are true:

- startup shows Qwen3-0.6B, HF training, vLLM generation, three isolated roles,
  one ordered generation GPU, training GPU 0, and the expected budgets;
- the run exits normally and `final.summary.json` says one epoch completed and
  target epochs reached;
- `bootstrap.summary.json` and `step00/step00.summary.json` exist;
- `checkpoints/latest.json`, `checkpoint_epoch001.json`,
  `checkpoint_epoch001.pt`, and compatibility `training_state.pt` exist;
- every role has immutable `adapter_epoch000` and `adapter_epoch001` artifacts,
  an adapter manifest, and matching optimizer state;
- branch arrivals, raw responses, parsed proposals, verifier attempts,
  evidence, policy traces, allocation plan, and events are durable;
- at least one `step00/learning/*.inputs.json`, `*.trace.json`, and
  `*.update.json` exists, with finite loss/KL/gradient norm, one role only,
  group K=4, and different before/after adapter hashes for an updated role;
- `events.jsonl` contains bootstrap and epoch `barrier_committed` events with
  unique monotonic sequences and idempotency keys;
- `status.json`, `best/`, compatibility best files, and plots are readable;
- no unsupported `swap_space`, missing `NamespaceTool`, signed-int32 LoRA
  overflow, LoRA collision, cross-role adapter, CUDA mask, or unreleased-model
  error occurred.

Useful artifact-only inspection commands are:

```sh
SMOKE_RUN_DIR=/absolute/path/to/the/new/run
test -f "$SMOKE_RUN_DIR/final.summary.json"
test -f "$SMOKE_RUN_DIR/step00/step00.summary.json"
test -f "$SMOKE_RUN_DIR/checkpoints/checkpoint_epoch001.pt"
find "$SMOKE_RUN_DIR/roles" -maxdepth 3 -type f | sort
find "$SMOKE_RUN_DIR/step00/learning" -maxdepth 1 -type f | sort
/absolute/path/to/python -m evolve.viz.run "$SMOKE_RUN_DIR" --all
```

Plotting is artifact-only and may be run after completion or against committed
state while active. A plot failure must not invalidate scientific artifacts,
but it must be reported and fixed before closing Phase 11.

### One-epoch completed-barrier resume

After the fresh half passes, resume the same new smoke run to a total target of
two epochs. `--num-steps 2` means total epochs, not two additional epochs. The
saved resolved topology is authoritative, so do not restate fresh defaults.

```sh
EVOLVE_PYTHON=/absolute/path/to/python sh run.sh \
  --resume /absolute/path/to/the/new/run \
  --num-steps 2
```

The resume half passes only if:

- the original manifest, bootstrap summary, and step00 summary are unchanged;
- `config.resume001.json`, `config.resolved.resume001.json`,
  `manifest.resume001.json`, `command.resume001.json`, and
  `environment.resume001.json` are added rather than replacing initial files;
- the real role adapters, optimizers, Python/Torch/CUDA RNG state, archive,
  posterior, ledger, memory, harness registry, and next epoch restore without
  alias or topology errors;
- already durable sample IDs are not regenerated or counted twice;
- `step01/step01.summary.json`, `checkpoint_epoch002.json`, and its exact
  training companion are committed;
- `final.summary.json` reports two completed target epochs;
- all three roles again have versioned epoch-2 adapter/optimizer artifacts;
- the event log contains exactly one additional epoch barrier commit and stays
  monotonic/idempotent;
- artifact-only plots work after resume.

### Known failure triage

- A package appears installed but import fails: confirm `EVOLVE_PYTHON` is the
  exact interpreter whose `-m pip` installed it. Do not mix system, venv, and
  Conda interpreters.
- `NamespaceTool` or an OpenAI schema import is missing: require
  `openai>=2.25.0,<3`; this is vLLM plumbing, not an OpenAI model selection.
- vLLM rejects `swap_space`: the server is not running the current code or the
  expected vLLM 0.28 path. Current EVOLVE intentionally omits that argument.
- FlashInfer compilation fails with no `nvcc`: current startup should select
  the safe non-FlashInfer sampling path. Capture the exact environment if it
  does not.
- LoRA ID overflow or collision: treat it as a correctness bug and stop. Never
  coerce, reuse, or silently remap an adapter ID.
- CUDA mask disagreement: unset the preexisting mask or make it exactly equal
  to the resolved ordered list. Do not guess logical versus physical IDs.
- OOM with Qwen3-0.6B: preserve the failed run, first lower
  `--vllm-gpu-memory-utilization` to 0.60 or reduce context/batching, and never
  reduce the logical rollout count silently.
- No learning update file: the runtime boundary is not fully tested. Inspect
  validity, group construction, and policy traces; do not declare success just
  because generation completed.
- Worker/runtime failure is infrastructure evidence, never a scientific score.
  Preserve it unresolved and do not edit artifacts to make the run pass.

## Closing Phase 11 after the server smoke

Only after both fresh and resume halves pass:

1. update `docs/EVOLVE_IMPLEMENTATION.md` Phase 11 to checked;
2. record server OS, GPU model/count, driver, CUDA, Python, PyTorch,
   Transformers, PEFT, bitsandbytes, vLLM, and OpenAI package versions;
3. record the exact fresh and resume commands, run directory, elapsed time,
   peak GPU memory if available, adapter versions/hashes, learning update IDs,
   checkpoint hashes, event/barrier counts, plots generated, and any warnings;
4. state explicitly that the smoke used Qwen3-0.6B and `evolve_toy`, not a
   scientific benchmark;
5. preserve the smoke run permanently as evidence and do not normalize it;
6. run `git diff --check` and the relevant CPU/static checks after any fix;
7. commit only source/docs changes, never the generated `runs/` directory.

If the smoke fails, leave Phase 11 open, preserve the run, document the exact
failure and environment, make the smallest evidence-preserving fix, run the
relevant CPU regression tests, and retry in a new run only with user approval.

## Definition of Done

Done means the original contract in `AGENTS-old.md` remains satisfied; phases
0–10 and all CPU/crash gates still pass; the fresh and resumed real Qwen/vLLM
smoke above passes on the documented server topology; complete artifacts,
adapters, optimizer/RNG companions, live status, best answers, and plots are
present; and `docs/EVOLVE_IMPLEMENTATION.md` contains the evidence.

Until then, describe EVOLVE as an implementation in progress with CPU readiness
established and the external model/runtime boundary pending.
