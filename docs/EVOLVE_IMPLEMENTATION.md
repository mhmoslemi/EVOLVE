# EVOLVE implementation status

This is the live implementation checklist for the root `AGENTS.md` contract.
The repository is EVOLVE-only, and existing `runs/` directories are immutable
user evidence unless a user explicitly resumes one. No run directory was
modified during the 2026-08-29 source-audit pass.

## Current status

All implementation phases have concrete runtime code, persistence, and
reporting paths. The complete deterministic CPU suite, composed fake engine,
completed-barrier resume, partial-epoch replay, corruption, compilation, and
shell gates pass as of 2026-08-29. Readiness is still **in progress** only at
the external-runtime boundary: the required tiny real-model/vLLM smoke needs
explicit user authorization and was not launched.

## A-to-Z phase checklist

- [x] **0 — Protect evidence behavior.** Strict EVOLVE schema detection,
  configuration precedence, immutable run attachment, compatibility rollout
  files, score direction, checkpoint pointers, and the deterministic toy
  fixture are implemented.
- [x] **1 — Foundations.** Schema-v1 typed records, canonical IDs, strict nested
  configuration, idempotent budgets/events, atomic writes, manifests,
  `train_evolve.py`, engine dispatch, `--validate-config`, and `--dry-plan` are
  implemented.
- [x] **2 — Scientific evidence.** Saved-payload common verification,
  confirmations, descriptors, fingerprints, multi-cell archive competition,
  record tracking, provenance, complete verifier captures, and the toy problem
  are implemented.
- [x] **3 — Isolated roles.** Exactly three named role LoRAs, independent logical
  optimizer/RNG state, stable role snapshots, concrete adapter artifacts,
  collision-bounded vLLM LoRA IDs, request-ID alias detection, and checkpoint
  restore are implemented.
- [x] **4 — Options and harnesses.** Registered executable state machines,
  horizon-scaled hard bounds, immutable branch specs, intermediate evidence,
  content-addressed harness specs, and matched harness trials are implemented.
- [x] **5 — Record allocation.** Hierarchical reliability/admission/improvement
  posteriors, independently validated resource statistics/backoff,
  positive-gain tails, uncertain costs, horizon arms, mandatory reservations,
  correlated Monte Carlo joint-max selection, homogeneous replicas, and
  reproducible allocation logs are implemented.
- [x] **6 — Causal memory.** Preassigned matched option audits, fixed treatment
  semantics, common randomness, explicit aborted pairs, normalized effects,
  support/uncertainty, quarantine/promotion/drift, contextual retrieval, and
  no-memory reservations are implemented.
- [x] **7 — Role learning.** Persisted exact-K homogeneous groups, OrderGrad and
  MaxPO objective selection, complete same-role policy traces, captured token
  masks/IDs, per-role gradient steps, KL logging, and before/after adapter
  identities are implemented.
- [x] **8 — Refinement.** Challenger-only one-change repairs, equal-cost fresh
  controls, blinded verification, no re-entry, attempts/depth/TTL/cost bounds,
  persisted failed repairs, and separate refinement grouping are implemented.
- [x] **9 — Compose and recover.** Frozen epoch manifests, bounded concurrent
  branch execution, generation/verification overlap, durable sample replay,
  authoritative in-epoch allocation-plan replay, durable bounded verifier and
  confirmation attempts, exact global retry settlement, deterministic record
  confirmation, atomic barriers, optimizer/RNG companions, completion-marker
  recovery, whole-lifecycle graceful draining, and compatibility artifacts are
  implemented.
- [x] **10 — Reporting.** Atomic live status, immutable best snapshots and
  pointer, compatibility best mirrors, periodic answer output, append-only
  archive/memory streams, and artifact-only record/archive/provenance/allocation/
  audit/role/posterior/failure/resource plots are implemented.
- [ ] **11 — Readiness gates.** The full CPU suite, repeated composed fake
  end-to-end run, completed-barrier resume, partial-epoch durable replay, and
  corrupted training-companion rejection pass. The explicitly authorized tiny
  real-model/vLLM smoke remains unexecuted, so this phase and the Definition of
  Done remain open.

## Decisions and invariants

- The root `AGENTS.md` is the sole method specification. No PUCT, Elo,
  self-likelihood feedback, observational success memory, or alternate engine
  is used as an EVOLVE fallback.
- Scientific state identity comes from the canonical saved answer payload.
  Proposal source hashes are diagnostics and never assign archive cells.
- Every raw response, parsed proposal, evidence packet, full verifier capture,
  branch outcome, audit assignment, and learning input is durable before the
  corresponding consumer runs.
- Infrastructure outcomes remain unresolved and affect reliability/resource
  beliefs only. They cannot enter archive quality, record confirmation, causal
  effects, tail gains, or policy learning.
- Production observations update the archive and scheduler only. Causal memory
  receives effects solely from closed, preassigned matched audits.
- Records are selected after all epoch branches close, ordered independently of
  worker completion, and confirmed only by reverifying the saved payload.
- Short-horizon arms reserve proportional option hard cost; execution and
  scheduler use the same logical bound. The retry ceiling is reserved at the
  allocation-arm layer so registered option IDs and completed-resume identities
  remain stable. Actual durable verifier attempts are reconciled against that
  reserve in the global ledger. A branch is folded into posterior/provenance
  exactly once regardless of its number of intermediate observations.
- An existing `allocation_plan.json` is the authoritative partial-epoch
  decision. Recovery validates its scheduler versions, canonical arm IDs,
  reservation labels, probabilities, and frozen production capacity instead of
  replanning against changed completion order or posterior state.
- Contradictory or unreadable durable prompts, responses, proposals, verifier
  attempts, states, descriptors, and evidence abort recovery as integrity
  failures. They are never reclassified as low scientific scores or replaced
  by retry evidence.
- Infrastructure-only observations update reliability and resource models but
  resource backoff is independent of scientific-admission support. Persisted
  posterior moments, counts, costs, and positive-gain reservoirs are validated
  before use.
- Adapter artifacts contain their exact optimizer companion before hashing and
  publication; role pointers are published only after the completed barrier.
- A noisy confirmation stops candidate scanning only when the confirmed record
  actually advances, allowing the next provisional candidate to be tried when
  budget remains.
- The immutable completed summary plus checkpoint hash is the barrier
  authority. New markers also bind the exact optimizer/RNG training companion
  by SHA-256, while older schema-v1 markers remain readable. Recovery fails
  closed on any malformed completed marker instead of silently rolling back to
  an older epoch. Role pointers, best mirrors, status, and plots are recoverable
  publications.
- `gpu_ids` remains the authoritative ordered vLLM set. Physical CUDA numbers
  are opaque labels: the training, generation, and evaluation assignments are
  explicit and do not depend on numeric ordering.
- vLLM 0.28 does not receive `swap_space`; the legacy resolved-config field is
  retained only so existing configs and resumes remain readable.
- Resource-only resume topology changes restore the saved training-device CUDA
  RNG onto the new training device; rollout identities remain independent of
  worker rank, completion order, and physical GPU number.
- Complete verifier captures are preserved by identity when the service has no
  missing fields to fill; a retry only replaces the capture when its durable
  attempt index actually changes.
- GPU-mode payload verification performs three bounded repeats using
  standard-library statistics, records the standard error and conservative
  runtime, treats empty repeat logs as clean, and persists exclusive evaluation
  as part of the frozen hardware identity.
- The generic subprocess sandbox accepts the legacy `none` policy as no mounted
  problem data in a fresh temporary directory. Linux retains `RLIMIT_AS`;
  macOS uses a parent-side process-group RSS watchdog because lowering
  `RLIMIT_AS` in Python's forked pre-exec child fails before `exec`.
- Harness trial records have an explicit JSON-safe durable projection; barrier
  persistence never sends nested `FrozenDict` values directly to `json.dumps`.
- Scheduler reservation overlap is evaluated from the complete reservation
  label set, not only the primary label, so an arm can simultaneously satisfy
  role, learning-group, empty-cell, and global-exploration requirements under
  finite production capacity.
- The composed fake-engine fixture reserves 4096 verifier calls. A smaller
  512-call fixture could legitimately exhaust its remaining budget after one
  run-ID-seeded portfolio and therefore did not deterministically guarantee
  that the second-epoch resume path would execute.

## Artifact and schema versions

- resolved configuration: `1`
- manifest and checkpoint envelope: `1`
- typed scientific records: `1`
- event and artifact JSONL streams: `1`
- generation job/parameters: `1`
- role registry/backend: `1`
- posterior: `zero_inflated_tail_v1` (additive reliability statistics remain
  readable from earlier v1 snapshots with an uninformative prior)
- harness registry: `1`
- reporting artifacts: `1`

## Validation record

Executed on 2026-08-29 with CPU-only fixtures and fake workers. The host's bare
`python` is Python 2.7, so the repository gates used the installed Python 3.11
interpreter explicitly. Third-party pytest plugin auto-loading was disabled
because an unrelated globally installed Hydra plugin crashes during pytest
startup; no project plugin is required by the suite.

```sh
/opt/homebrew/bin/python3.11 -m compileall -q evolve problems train_evolve.py
PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 /opt/homebrew/bin/python3.11 \
  -m pytest evolve/tests -q -p no:cacheprovider
sh -n run.sh
```

Results:

- compilation: passed;
- `run.sh` POSIX shell syntax: passed;
- `git diff --check`: passed;
- complete EVOLVE CPU suite: **349 passed in 12.10s**, no skips and no warnings;
- focused pre-integration suite: **139 passed**;
- affected config/GPU/toy/vLLM regression slice: **95 passed**;
- all eight shipped YAMLs: **16/16** validation and dry-plan invocations passed,
  with `model_loading: false` and `writes_run_directory: false` in every plan;
- problem registry audit: all eight YAMLs instantiate the expected typed
  problem/subtype, expose every required scientific hook, and declare the
  expected CPU or exclusive-GPU verifier requirement;
- launcher/docs audit: explicit `run.sh` validation and GPU-mode dry-plan pass,
  every README `run.sh` command parses against the current CLI, and validation
  leaves the existing run-directory name set unchanged;
- expanded CLI/engine/sandbox boundary slice: **22 passed**;
- composed fake engine coverage includes fresh bootstrap/epoch, confirmed best
  publication, homogeneous role learning, matched audits/harness trials,
  three-worker concurrent branch overlap and completion-order handling,
  target-epoch resume, immutable old summaries, durable partial-epoch replay
  from the identical allocation plan, shutdown, and fail-closed training-state
  corruption. The completed-barrier resume test passed three additional
  consecutive runs in **5.34s**, **5.39s**, and **5.92s**;
- every JSON and nonblank JSONL record produced by the composed run parses, no
  temporary artifact remains, and all nine artifact-only plots generate from
  both an interrupted active-run fixture and a completed resumed run;
- subprocess isolation: disposable working directory, legacy `none` policy,
  Python socket denial, timeout and spawned-process-group cleanup, memory
  enforcement, and bounded diagnostics pass on CPU;
- recovery corruption matrix: malformed/future completion markers, unsafe
  checkpoint paths, checkpoint hash mismatch, incomplete/missing training-state
  companions, duplicate committed epochs, and the existing companion-content
  corruption all fail closed.

Failures found and fixed during these gates:

- needless replacement of a complete verifier-owned execution capture;
- top-level `gpu_type` fixture expectations lagging the problem-runtime
  projection and explicit GPU benchmark hardware contract;
- removed vLLM 0.28 `swap_space` warnings still expected by old tests;
- GPU saved-payload verification referenced removed NumPy symbols;
- empty repeated evaluator logs were misclassified as non-empty diagnostics;
- macOS rejected the generic sandbox's memory limit inside `preexec_fn`;
- harness audit barrier persistence was not JSON-safe;
- overlapping scheduler reservations could select a role-only arm and then
  falsely report the empty-cell reservation as unsatisfied;
- the original 512-call fake integration budget could validly stop before the
  second resumed epoch for some run-derived scheduler seeds; the fixture now
  uses enough CPU-only verifier budget to make the recovery assertion stable.

No model was loaded, no vLLM server was started, no CUDA/GPU benchmark ran, and
no user-owned `runs/` directory was modified.

## Remaining gates

The CPU, fake end-to-end, completed-barrier resume, and targeted crash/corruption
gates are complete. The remaining readiness gate is an explicitly authorized
tiny Qwen/vLLM smoke run exercising the selected installed versions, adapter
load/generation/training phase switch, and actual GPU topology. No real model,
multi-GPU job, GPU benchmark, or scientific experiment is authorized by this
checklist.
