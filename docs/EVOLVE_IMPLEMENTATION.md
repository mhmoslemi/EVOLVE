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
- Hierarchical zero-inflated posterior, posterior-rate Monte Carlo draws,
  reservations, and joint maximum portfolio selection.
- Homogeneous rollout replicas for non-degenerate barrier learning groups.
- Preassigned option audits, causal effect storage and promotion, contextual
  promoted-memory retrieval, and permanent no-memory reservation accounting.
- Durable prompt/response/proposal/evidence/state/outcome/policy artifacts before
  learning, plus idempotent controller events.
- Actual branch verifier-call accounting and unused-budget refunds.
- Checkpoints containing archive, provenance, posterior, budget, causal memory,
  harness trials, nursery state, role state, and confirmed record.
- Resume selects only checkpoints referenced by completed barrier markers.
- Atomic status, final summary, best-state/evidence/candidate artifacts, rendered
  answers, and compatibility best source.
- Bounded subprocess verification with unique temporary directories, process
  group cleanup, CPU/memory/file limits, bounded diagnostics, explicit network
  policy, and JSON-only result envelopes.

## Deliberately unclaimed

No test suite, compile command, dry run, model run, or GPU run was executed during
the EVOLVE-only migration, at the user's explicit request. Consequently this
document does not claim readiness or empirical scientific improvement. The
source must receive a separate validation pass before a costly run is launched.

## Schema versions

- resolved configuration: 1
- manifest: 1
- typed scientific records: 1
- event stream: 1
- checkpoint: 1
- role registry/backend: 1
- posterior: `zero_inflated_tail_v1`
- harness registry: 1

## Safe validation commands for a later authorized pass

```sh
python -m compileall evolve problems train_evolve.py
python -m pytest tests evolve/tests -q -p no:cacheprovider
python -m pytest dpuct/tests -q
sh -n run.sh
```

These commands are documentation only; they were not run in this migration.
