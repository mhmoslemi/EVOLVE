# EVOLVE implementation status

This repository is EVOLVE-only. Historical TTT/PUCT training, self-likelihood
feedback, observational memory, and Elo reranking are not runtime dependencies.
Existing `runs/` evidence remains untouched.

## Implemented integration

- Strict nested EVOLVE configuration and complete problem-facing resolved
  configuration, including subtype, seed, resources, timeouts, and GPU identity.
- Content-addressed proposals, scientific states, evidence, descriptors,
  fingerprints, archive cells, provenance, branches, audits, and checkpoints.
- One HF backbone with named `scout`, `mechanist`, and `challenger` LoRAs,
  separate optimizers and RNG identities, explicit activation, and role artifact
  persistence.
- Executable role options plus a registered matched-continuation audit control.
- Baseline and structured-diagnostic harness specs, matched harness trials, and
  conservative barrier-only promotion.
- Hierarchical zero-inflated posterior, posterior-rate Monte Carlo draws,
  reservations, and joint maximum portfolio selection.
- Homogeneous rollout replicas for non-degenerate barrier learning groups.
- Preassigned option audits, causal effect storage and promotion, contextual
  promoted-memory retrieval, and permanent no-memory reservation accounting.
- Challenger-only bounded refinement with a randomized equal-cost fresh-control
  branch, attempt/depth/TTL limits, and persisted failed repairs.
- Durable prompt/response/proposal/evidence/state/outcome/policy artifacts before
  learning, plus idempotent controller events.
- Branch and record-confirmation verifier-call accounting, bounded
  infrastructure retry, and unused-budget refunds.
- Checkpoints containing archive, provenance, posterior, budget, causal memory,
  harness trials, nursery state, role state, and confirmed record.
- Resume selects only checkpoints referenced by completed barrier markers.
- Atomic status, final summary, best-state/evidence/candidate artifacts, rendered
  answers, and compatibility best source.
- Bounded subprocess verification with unique temporary directories, process
  group cleanup, CPU/memory/file limits, bounded diagnostics, explicit network
  policy, and JSON-only result envelopes.
- Exact finite-batch OrderGrad Top-M@K likelihood-ratio advantages, with each
  branch's same-role policy decisions summed before the branch-level loss.
- Human-readable, framework-annotated problem YAMLs and an EVOLVE-only launcher
  with model-free validation/dry-plan commands.
- HF and Unsloth training backends with optional training-only 4-bit loading,
  barrier phase-switching to a single tensor-parallel vLLM generation engine,
  and preflight rejection of unsupported multi-GPU pre-quantized BnB loading.
- Ordered launcher GPU topology with explicit training placement, vLLM device
  selection, GPU-mode-only dedicated evaluation on the last device when
  available, and serialized model teardown for a one-GPU kernel benchmark.
- vLLM EngineArgs capability validation, including the vLLM 0.28 removal of
  `swap_space` and its explicit `device_ids` split-placement interface.

## Earlier validation (before the backend/resource changes)

- `python3 -m compileall -q evolve problems train_evolve.py`
- `sh -n run.sh`
- model-free `--validate-config` for every committed YAML
- default `sh run.sh` toy dry plan
- 18 focused CPU tests covering YAML/config/CLI behavior, scientific toy
  verification and identity, budget idempotency, role learning ownership, and
  enumerated OrderGrad correctness

Per the user's instruction, no test, compile check, config validation, dry plan,
model load, live training, multi-GPU run, vLLM run, or benchmark run was executed
for the HF/Unsloth/vLLM and GPU-isolation changes. The implementation therefore
remains in progress until those readiness gates pass, and it makes no empirical
claim that EVOLVE improves scientific discovery.

The current live HF boundary owns one backbone and exact sequential role-adapter
activation. Persistent streamed multi-worker generation and concurrent verifier
execution still require a dedicated validation/integration pass before a costly
production launch.

## Schema versions

- resolved configuration: 1
- manifest: 1
- typed scientific records: 1
- event stream: 1
- checkpoint: 1
- role registry/backend: 1
- posterior: `zero_inflated_tail_v1`
- harness registry: 1

## Remaining readiness commands

```sh
python -m compileall evolve problems train_evolve.py
python -m pytest evolve/tests -q -p no:cacheprovider
sh -n run.sh
```

`compileall`, shell syntax, configuration validation, dry plans for one/two/four
GPU layouts, and installed-vLLM-0.28 argument compatibility have been checked.
The full pytest gate remains outstanding because neither available local Python
environment currently contains pytest. No live model, training, vLLM, or GPU
benchmark smoke run was launched.
