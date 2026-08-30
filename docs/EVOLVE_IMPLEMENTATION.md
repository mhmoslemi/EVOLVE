# EVOLVE implementation status

This is the live implementation checklist for the root `AGENTS.md` contract.
The repository is EVOLVE-only, and existing `runs/` directories are immutable
user evidence unless a user explicitly resumes one. No run directory was
modified during the 2026-08-29 source-audit pass.

## Current status

All implementation phases have concrete runtime code, persistence, and
reporting paths. Readiness is still **in progress** because the user explicitly
requested that no tests or validation commands be run during the latest pass.
The Definition of Done cannot be claimed until the required CPU, crash/recovery,
and authorized model gates execute successfully.

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
- [ ] **11 — Readiness gates.** Code paths are present, but the required CPU
  suite, crash suite, full fake end-to-end/resume suite, and authorized tiny
  model smoke have not all been executed after the latest changes.

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

No pytest invocation, compile check, shell syntax check, config validation, dry
plan, model load, vLLM startup, training job, GPU benchmark, or plotting command
was run during the latest source-audit pass, per the user's instruction. Any
older validation notes predate these changes and are not treated as current
evidence.

## Remaining gates

When testing is authorized, the minimum readiness sequence is:

```sh
python -m compileall evolve problems train_evolve.py
python -m pytest evolve/tests -q -p no:cacheprovider
sh -n run.sh
```

Then run the CPU fake end-to-end and completed-barrier resume/crash scenarios.
Only after those pass should an explicitly authorized tiny Qwen/vLLM smoke run
exercise the selected installed versions and GPU topology. No expensive model,
multi-GPU, or scientific experiment is authorized by this checklist.
