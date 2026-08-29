# EVOLVE implementation contract

## Mission

Build EVOLVE end to end while preserving problem YAMLs, streamed multi-worker generation, concurrent verification, complete artifacts, completed-barrier legacy resume, reporting, and plots. This is not a PUCT facelift. Search verified scientific states, allocate by posterior record value, isolate three roles, treat harnesses as arms, learn memory from matched audits, and train homogeneous on-policy max-seeking groups. Migrate additively, keep `legacy` runnable, and never reinterpret existing runs.

## Scope and source of truth

This file governs the repository and is the method source of truth. Local instructions may add rules but not weaken these invariants. The current README, memory design, sampler, advantage, feedback, and memory code specify only the legacy path.

`legacy` is global PUCT, one LoRA, the entropic objective, one-step memory credit, and self-likelihood feedback. `evolve` is this specification. An arm chooses cell, role, option, harness, horizon, and cost. An epoch is a frozen cycle, a branch executes one arm, an audit pair is randomized and matched, and the record is the best confirmed admissible reward.

## Repository facts that must be preserved

Read the relevant implementation before editing it. The active legacy path is rooted at `train_multy.py`.

- Preserve `run.sh`, argument forwarding, and `train_multy.py` as the legacy entrypoint.
- Reuse `gen_workers.py`, `model_backend.py`, `experiment_io.py`, `sandbox.py`, and the typed problem contract in `problems/base.py` through tested adapters.
- Keep problem registration in `problems/registry.py` and one YAML per problem or subtype in `configs/`.
- `dpuct/` is a standalone tested search library. Reuse it behind an adapter when useful for branch-local tree decisions. Do not create a third generic PUCT implementation and do not use D-PUCT as the global EVOLVE allocation rule.
- Preserve old plot and analysis readers.
- `_old_stuff_/`, `sampler-puct.py`, and `memory/train_multy.py` are historical. Never make them production dependencies.
- `runs/` contains user-owned evidence. Never edit, normalize, rename, or delete an existing run.

The configuration precedence is a compatibility contract.

1. Dataclass defaults
2. The problem YAML
3. The resumed run's resolved `config.json`
4. Explicit CLI flags

The last layer wins. `gpu_ids` remains authoritative. A resume must use the saved method and schema unless the user explicitly requests a supported migration.

The current run invariant also remains in force. Save every rendered prompt, raw response, parsed candidate, verifier result, invalid attempt, duplicate, repair, and pruned proposal before any gradient update that consumes it.

## Non-negotiable EVOLVE method

### Search objective

Optimize the expected final verified record under a finite compute budget. Average reward, token likelihood, archive rank, visit count, and novelty are supporting signals. None is the terminal objective.

At allocation time, freeze the current record threshold. Estimate the probability and magnitude of a descendant maximum exceeding that threshold over the arm's horizon and cost. Plan a portfolio of arms that maximizes expected record improvement while satisfying exploration, audit, role, harness, and resource reservations.

### Global archive and local trees

There is no global source-code tree. Separate an immutable source `Proposal` from the `VerifiedScientificState` identified by its captured answer payload and evidence. Maintain a descriptor-indexed quality-diversity archive and append-only provenance. Source hashes are diagnostics only. Each cell has distinct slots for confirmed quality champion, promising verified descendant value, and stepping stones chosen by novelty or uncertainty, plus forced empty and under-tested sampling. Fingerprints use verified structure, behavior, diagnostics, complexity, construction, and algorithmic family. LLM labels never define identity.

An option may build a horizon- and budget-bounded local tree rooted at its assigned archive candidate. Link every verified descendant into global provenance and offer it to the archive. Archive eviction never removes artifacts or lineage, and local tree statistics never replace the global scheduler.

### Agent ecology and model ownership

Use one frozen LLM backbone and exactly three isolated role adapters.

- `scout` searches broadly, enters empty cells, and proposes structurally different approaches.
- `mechanist` develops explanations, invariants, and focused improvements around promising mechanisms.
- `challenger` attacks assumptions, constructs counterexamples, and performs bounded minimal repairs.

Each owns a LoRA, optimizer, RNG, working transcript, retrieval view, and learning groups. Never merge adapters or gradients, train the base, or mix roles in a batch. The verifier is independent. An adapter registry gives every generation job a stable role, adapter ID and version, policy snapshot, and seed. vLLM request IDs cannot alias. Sequential role activation is allowed if isolation is exact. Do not add multiple backbones to the baseline.

### Options and branches

An option is a registered executable state machine, not a prompt label. Its versioned spec defines allowed roles and capabilities, initiation, per-step policy, stop rule, horizon, expected and hard costs, harness eligibility, prerequisites, and output contract. Before dispatch, a branch freezes its start candidate, record threshold, role and option versions, harness, verifier, memory view, horizon, budget, and seed. Its outcome is its maximum independently verified admissible descendant plus every intermediate candidate and cost. Return genuinely unused budget after early stop.

### Adaptive harness

The harness is part of the allocation arm, not a hidden constant or final judge. A content-addressed `HarnessSpec` versions instructions, tools, intermediate tests, scaffolding, and diagnostic feedback. Freeze each branch's assigned version. Option audits hold it fixed. Harness audits differ only in harness version while holding start, role, option, horizon, cost, policy, verifier, and generation settings fixed. Store harness effects and promote versions by conservative repeated audit evidence. Any behavior change creates a new version.

Every candidate passes the same independent problem verifier. Harness-local scores may guide a branch but cannot admit a candidate or set the record. Do not pool incomparable harness scores without common-verifier calibration. Give harness trials a bounded budget, compare matched contexts where possible, log cost, validity, record gain, and failure mix, and promote only through an explicit barrier decision.

### Independent verification

The common verifier alone controls admission and the record. Its immutable, content-hashed `EvidencePacket` records run, proposal, scientific-state, parent, branch, problem, verifier, harness, policy, and lineage IDs, flags and scores, uncertainty, descriptors and fingerprint, source hash, full bounded diagnostics, failure kind, resources, serialized answer payload, and timestamps.

Infrastructure failure is not a low scientific score. Retry under a small budget, persist it unresolved, and exclude it from scientific admission and tail estimates, causal credit, and learning while still updating reliability and resource-risk models. Problems declare whether timeouts are scientific. Confirm possible records by verifying the saved answer payload, never by rerunning stochastic proposal code. Noisy problems use a lower confidence bound or robust repeats.

### Posterior allocation

An allocation arm is

```text
(cell_id, role, option_id, harness_version, horizon, cost_class)
```

Use a hierarchical sparse-data posterior. The first implementable baseline is a zero-inflated record-gain model.

- Use Beta-Binomial admission probability with backoff from exact arm through option-role, role, cell region, and global levels.
- Given admission, model positive descendant-max gain over the frozen record by bounded empirical, Bayesian-bootstrap, threshold, or QoMax-style tails.
- Predict tokens, wall time, verifier time, and scarce-device cost separately and with uncertainty.
- Monte Carlo the joint portfolio objective `E[max(0, max_r Z_r - M)]`. Marginal value is the change in this joint maximum after adding an arm, not a sum of independent expected improvements. Model dependence using shared cell, fingerprint family, start, option family, and observed correlation.

Reserve randomized audits, every role, empty or under-tested cells, harness calibration, and global exploration first. Allocate the remainder by greedy or small-knapsack marginal expected record improvement subject to each resource. Recompute only at barriers.

Log sufficient statistics, hierarchy fallback, RNG seed, expected gain and cost, uncertainty, correlation penalty, reservations, and final choice so the plan reproduces exactly.

Do not use raw historical maximum PUCT, archive-wide rank alone, or an LLM judge score as the EVOLVE scheduler. D-PUCT may select actions within a bounded option tree after the global arm has been assigned.

### Causal option memory

Memory stores evidence about interventions, not attractive summaries of successful trajectories.

Production updates the scheduler and archive but never establishes causal memory. Credit comes only from randomized matched audits sharing start candidate and cell, frozen threshold, role checkpoint, harness and verifier, horizon, resources, generation settings, and common randomness where valid. Persist assignment probability and the pair spec before execution. Compare the intervention with a registered matched continuation and close both sides before computing their normalized descendant-max difference.

Store pair effects. A memory record includes context, intervention, pair IDs, propensity, effect and uncertainty, support, recency, scope, contraindications, and lineage. Promote only after repeated support and positive conservative effect. Quarantine conflicting or sparse records, stratify drift, retrieve contextually without extra rollout budget, and permanently reserve no-memory audits.

Private role working memory may summarize the current branch, but it is ephemeral and cannot be promoted into causal option memory without audits.

### Test-time learning

Train at barriers only from persisted homogeneous on-policy groups matching role and policy snapshot, start cell or context, option, harness, horizon, cost, generation settings, frozen threshold, and production, audit, or refinement channel. Reject rather than mix roles, versions, budgets, audit halves, or refinement attempts. A `PolicyTrace` stores every role-policy prompt, response segment, token mask, log probability, and adapter hash. Branch log probability sums all same-role policy decisions, and descendant gain credits that complete trace while excluding tools, verifier text, and other-role tokens.

Learn from branch descendant maxima using exact, independently tested OrderGrad for top-m-at-K. Pure-max mode may use exactly centered MaxPO. KL or entropy is explicit and secondary. Never relabel the entropic legacy surrogate or improvise equations. Check the primary papers, enumerated distributions, centering identities, and finite-difference or Monte Carlo tests.

Update only the generating role. Record objective version, group members, advantages, masks, KL, gradient norm, optimizer step, and before and after adapter hashes. Save inputs before backward.

The legacy self-likelihood feedback update is not part of EVOLVE. Verifier diagnostics can guide allocation or enter bounded refinement, but learning from a repair requires an executed and independently verified improvement under the refinement protocol.

### Bounded refinement nursery

Invalid or nearly admissible candidates may enter a separate nursery. Challenger makes one minimal diagnostic-targeted change per attempt. Allow one to three attempts, depth at most two, fixed cost, strict TTL, one entry, and a separate cap. Every revision is a new candidate with blinded verification. Randomize eligible cases between refinement and equal-cost fresh continuation in a dedicated refinement-audit channel. Its paired effect may update causal evidence. Confirmed refinement traces use separate homogeneous role-learning groups and never enter production groups.

Persist failed repairs. They are useful diagnostics but not negative causal option evidence unless placed in a valid audit.

### Synchronization

Use epoch barriers to prevent moving-target evidence.

At epoch start, freeze archive, record threshold, scheduler, role adapters, causal memory, options, harnesses, verifier, descriptor function, cell map, fingerprint function, and reporting schema versions. Persist them in the manifest, branch specs, evidence, and checkpoint before dispatch. New mappings take effect only after a barrier.

During the epoch, stream branch generation and verification with bounded backpressure. Append artifacts as they arrive, but do not expose new archive entries or learned weights to in-flight branches.

Wait for or explicitly close every allocation, then atomically perform

1. evidence finalization and record confirmation
2. archive and provenance updates
3. scheduler posterior updates
4. completed audit effect updates and memory promotion decisions
5. role-specific learning updates
6. approved harness-state updates
7. checkpoint and report publication

Resume from the last barrier. Reuse already durable responses and evidence by sample ID, replay only missing IDs with identical versions and seeds, or mark them infrastructure-aborted. Never double count.

## Target code organization

Add a cohesive `evolve/` package. Keep root entrypoints and compatibility helpers small.

```text
train_evolve.py
evolve/
  config.py, types.py, ids.py, budget.py, engine.py, cli.py
  archive/        cells, descriptors, fingerprints, provenance
  roles/          registry, adapters, working_memory
  options/        base, registry, builtins, branch
  harness/        spec, registry
  scheduler/      arms, posterior, portfolio, reservations
  verifier/       evidence, service, confirmation
  audits/         pairing, effects
  causal_memory/  records, promotion, retrieval
  learning/       groups, objectives, trainer
  refinement/     nursery
  workers/        generation, verification, resources
  runio/          events, checkpoint, recovery, status
  reporting/      console, best
  viz/            record, archive, provenance, allocation, audits, roles
```

Use small typed services and pure decisions, not another monolithic step. Adapt the problem verifier, worker stream, sandbox, run I/O, and old readers under compatibility tests. Use D-PUCT only locally. Extract tested neutral helpers rather than importing `train_multy.py` as a library.

## Required typed records and invariants

Define schema-versioned typed records for `Proposal`, `VerifiedScientificState`, `EvidencePacket`, `Descriptor`, `ArchiveCell`, `ProvenanceEdge`, `RoleSnapshot`, `OptionSpec`, `HarnessSpec`, `AllocationArm`, `BranchSpec`, `BranchOutcome`, `PolicyTrace`, `AuditPair`, `CausalMemoryRecord`, `LearningGroup`, `BudgetLedger`, and `EpochManifest`.

IDs are stable and namespaced, using canonical content when identity survives resume. Never persist Python `hash()`.

Runtime validation must enforce immutability after ID assignment, exact candidate and verifier references, confirmed-only record updates, higher-is-better internal reward with native `raw_score`, valid provenance endpoints, one frozen branch spec, idempotent resource-specific budget debits, scheduler updates only from closed eligible branches, paired preassigned audits, audit-backed promoted memory, persisted on-policy learning inputs, and append-only artifacts except documented pointers, status, plots, and compatibility best mirrors.

Use JSON-safe enums and versions, validate reads, preserve unknown migrated fields, and reject unsupported future schemas.

## Problem API evolution

Do not break existing problem files in the first migration. Normalize `build_prompt(parent, memory="")` through an adapter, then add hooks such as

```python
describe_scientific_state(candidate, evidence) -> Descriptor
scientific_fingerprint(candidate, evidence) -> str
serialize_answer(candidate, evidence) -> JSONValue
verify_answer_payload(payload, policy) -> EvidencePacket
record_key(evidence) -> float
confirm_record(candidate, evidence, policy) -> EvidencePacket
normalize_gain(new_reward, threshold) -> float
render_best(candidate, evidence, output_dir) -> list[str]
harness_specs() -> list[HarnessSpec]
resource_requirements() -> ResourceRequirements
```

Smoke and legacy adapters may derive a coarse verified-output descriptor, but must be marked method-incomplete. Production EVOLVE requires problem-defined scientific descriptors, fingerprints, answer serialization, and deterministic payload verification. Source text never assigns a cell.

Adding a problem requires `problems/<name>.py`, registry aliases, a matching problem or subtype YAML, deterministic seed and verifier-direction tests, descriptor and fingerprint tests, a best-answer renderer or text fallback, and an explicit resource declaration.

Never change process-wide working directories in a verifier. Pass declared CPU, memory, timeout, and GPU limits into actual execution and record observed resources. Prompt text is not sandbox enforcement. Keep isolation, unique temporary paths, process-group cleanup, and explicit network or filesystem policy versioned. Runtime benchmarks retain exclusive evaluation resources and generally `reward_workers: 1`.

## Configuration and launch contract

Keep one human-readable YAML per problem or problem subtype. Do not introduce Hydra or a second unrelated configuration stack. Add a strictly validated nested `evolve` section while keeping current problem keys and common CLI flags.

An indicative configuration is

```yaml
engine: evolve

evolve:
  budget:
    epochs: 100
    verifier_calls: 50000
    audit_fraction: 0.15
    refinement_fraction: 0.05
  archive:
    elites_per_cell: 3
    empty_cell_fraction: 0.10
  roles:
    enabled: [scout, mechanist, challenger]
  options:
    max_horizon: 4
    branch_budget: 64
  harnesses:
    trial_fraction: 0.05
    active_versions: [baseline_v1]
  scheduler:
    posterior: zero_inflated_tail
    global_exploration_fraction: 0.10
  audits:
    no_memory_fraction: 0.05
    min_pairs_for_promotion: 5
  learning:
    objective: ordergrad
    top_m: 1
    group_k: 8
  refinement:
    max_attempts: 3
    max_depth: 2
  reporting:
    status_every_verifications: 25
    plots_every_epochs: 1
  workers:
    max_inflight_branches: 16
```

This is a shape contract, not permission to hard-code values. Production baseline validates exactly `scout`, `mechanist`, and `challenger`; role subsets are test-only and method-incomplete. Reject unknown keys and invalid ranges before model loading, including sampling, budgets, group dimensions, GPU conflicts, paths, context, disk, and checkpoints.

Write the requested config and an immutable resolved manifest after every default and runtime derivation. Include command, Git state, model and package versions, host, GPUs, worker topology, seeds, harness and verifier versions, and config hash. Runtime must match it. Preserve real JSON types rather than stringifying lists.

Keep `run.sh` working for both fresh and resumed runs. Required interfaces are

```bash
sh run.sh
sh run.sh --engine evolve --problem erdos --config configs/erdos.yaml
sh run.sh --resume /absolute/path/to/runs/RUN_NAME
sh run.sh --resume /absolute/path/to/runs/RUN_NAME --num-steps 150
```

Preserve final `"$@"` forwarding. `run.sh` resolves and removes dispatcher-only flags before calling legacy. Runs without an engine field are legacy. Fresh defaults never become implicit resume overrides, especially backend, GPU IDs, or CUDA masks. A resume derives engine and topology from its versioned effective config unless the user explicitly overrides a supported resource field, then writes a new config version and checkpoint hash without replacing the initial manifest. In EVOLVE, `--num-steps N` is a compatibility alias for total target epochs, not N additional epochs. Keep `train_multy.py` for legacy and add `train_evolve.py`; change the fresh default only after all gates and user agreement.

Do not launch a real multi-GPU or long model run as part of ordinary implementation or testing. Such a run requires explicit user authorization and an agreed config.

## Multi-worker and resource behavior

Preserve persistent spawned generation workers and generation-verification overlap. Add per-job adapter selection, partition batches by role snapshot when required, raise vLLM LoRA capacity, and prove HF and vLLM adapter non-leakage. Do not create one base-model pool per role.

- Every job includes allocation ID, branch step, role, adapter path and version, option, harness, policy snapshot, exact sample count, generation parameters, and deterministic seed.
- `distribute_jobs` must still return exactly the requested number of samples across workers.
- Stream results into verification immediately with bounded queues.
- Backpressure generation when verifier or persistence queues are full.
- A worker error is explicit infrastructure evidence and never silently mapped to a low scientific reward.
- HF OOM recovery may reduce a worker's sticky microbatch, but may not change the logical rollout count.
- vLLM adapter IDs and paths must not alias across roles or epochs.
- Do not hold archive or scheduler locks while generating, verifying, plotting, judging, or writing large files.
- Resource leases prevent concurrent use of an exclusive benchmark GPU.
- Graceful shutdown drains durable writes, closes assignments, stops workers, and preserves the last valid checkpoint.

Derive and persist seeds from run, epoch, allocation, branch step, sample, and role identity, independent of worker rank, completion order, and rescheduling. Log worker and microbatch topology as execution metadata.

## Run directory and persistence contract

Keep the recognizable top-level naming convention, but use seconds plus a short random run ID and an exclusive creation lock. Refuse to attach to a nonempty run unless `--resume` was explicit.

```text
runs/<problem>_<model>_<MMDD-HHMMSS>_<run-id>/
```

Evolve the contents without creating a disconnected second logging system.

```text
config.requested.yaml
config.resolved.json
config.json                 compatibility copy of resolved config
manifest.json, command.json, environment.json
status.json, events.jsonl, training_state.pt
checkpoints/{latest.json, checkpoint_epochNNN.pt}
roles/{scout,mechanist,challenger}/adapter_epochNNN/
archive/{cells.json,candidates.jsonl,evidence.jsonl,provenance.jsonl,snapshots/}
causal_memory/{records.jsonl,snapshots/}
stepNN/
  stepNN.parents.json, epoch.manifest.json, allocation_plan.json
  stepNN_groupGG_rolloutRRR.prompt.txt
  stepNN_groupGG_rolloutRRR.txt
  stepNN_groupGG_rolloutRRR.meta.json
  branches/, audits/, learning/, refinement/
  stepNN.summary.json        completion marker
best/{candidate.json,evidence.json,answer.py,answer.txt}
plots/, logs/controller.log, logs/workers/, logs/verifiers/
final.summary.json
best_code.py, best_construction.json
```

Treat `stepNN` as EVOLVE epoch NN and `allocation_plan.json` as authoritative. `stepNN.parents.json` is a compatibility view. Give every allocation hop, audit side, refinement attempt, retry, group, and rollout a deterministic globally unique epoch index so flat files never overwrite. Preserve legacy fields and add EVOLVE IDs. `config.json` is an explicit top-level compatibility projection tested with existing plotters, while `config.resolved.json` is authoritative.

One controller-owned serialized writer assigns monotonic event sequences and idempotency keys. Workers never append JSONL directly. Persist each response on arrival and evidence after verification, flush assignments and evidence before use, keep complete bounded verifier traces separately, and content-address repeated prompts while retaining readable compatibility files.

Atomic commit order at a completed barrier is

1. all raw branch, audit, and refinement artifacts
2. finalized evidence and closed allocation events
3. archive, scheduler, memory, and learning snapshots
4. all three versioned role adapters and optimizer states
5. the versioned checkpoint
6. atomic replacement of `checkpoints/latest.json` and compatibility `training_state.pt`
7. `stepNN.summary.json` as completion marker
8. committed best, status, and non-critical plots

Never let a summary advertise an uncommitted checkpoint. Use temporary files in the destination directory followed by atomic replace. Checkpoint role-to-adapter mapping, three optimizer states, RNG states, archive, provenance, scheduler posterior, budget ledger, causal memory, harness registry, next epoch, and schema versions.

An in-epoch record is provisional and cannot affect the frozen threshold or public `best/`. Publish only confirmed records in the barrier commit. Old runs remain read-only and auto-detected. Never backfill them.

## Periodic status, best answer, and plots

The user must be able to understand a live run without opening raw logs.

Refresh `status.json` atomically at the verification cadence and every barrier. Report confirmed internal and native records, holder and its cell, role, option, and harness, resource ledger and record age, archive coverage, allocations by channel and role, outcome and infrastructure rates, audit and memory state, and latest learning group and adapter versions.

At every barrier that commits a confirmed new record, atomically update `best/`, compatibility best files, and the problem renderer. Print the full answer then when configured and periodically otherwise. Live status may label a provisional observation but never present it as the committed answer.

Generate headless plots from stored artifacts only. Plot record against tokens, verifier calls, and time, archive coverage and quality, provenance with record lineage, allocation by cell, role, option, and harness, posterior calibration, audit effects and memory state, role learning, failures, and resources. Plot failures never abort discovery.

Retain old plotting commands. Add one schema-aware command such as

```bash
python -m evolve.viz.run RUN_DIR --all
```

It must work after a run and while a run is active by reading committed snapshots. It must never rerun candidate code by default. Problem-specific rendering may replay only when no saved construction exists and the user explicitly requests it.

## A-to-Z implementation sequence

Work in vertical slices. Keep the repository runnable after every phase. Maintain `docs/EVOLVE_IMPLEMENTATION.md` with this checklist, decisions, schema versions, tests, and remaining gaps. Do not mark a phase complete from interfaces alone.

0. **Protect legacy behavior.** Characterize config precedence, rollout counts, artifact order, score direction, best tracking, and checkpoint pointers. Add schema detection, a tiny fixture, and correct test ignore rules.
1. **Build foundations.** Add typed records, strict config, IDs, budget, events, atomic writes, manifest, `train_evolve.py --validate-config --dry-plan`, and engine dispatch.
2. **Make evidence scientific.** Adapt verifier output, confirmation, descriptors, fingerprints, cells, local competition, record, and provenance. Add a deterministic multi-cell toy problem.
3. **Isolate roles.** Add three adapters, optimizers, RNGs, role-aware HF and vLLM jobs, non-aliasing IDs, and full restore tests.
4. **Execute options.** Add immutable harnesses, option state machines, frozen branches, hard bounds, intermediate evidence, and optional local D-PUCT.
5. **Allocate for records.** Add hierarchical admission and tail models, horizon-aware expected improvement, costs, reservations, correlation-aware portfolio choice, and reproducible logs.
6. **Learn causal memory.** Add persisted randomization, matched pair closure, effects, promotion, quarantine, drift, contextual retrieval, and no-memory audits.
7. **Learn role policies.** Add strict groups, tested OrderGrad and optional MaxPO, one-role barrier updates, objective logs, and isolation checks.
8. **Bound refinement.** Add minimal Challenger repairs, TTL, depth, attempts, cost, blinded verification, no re-entry, and separate groups.
9. **Compose and recover.** Build `EvolveEngine`, frozen manifests, bounded asynchronous execution, barriers, shutdown, idempotent crash recovery, and compatibility artifacts.
10. **Report.** Add live status, periodic best artifacts, generic plots, renderers, and active-run fixture tests.
11. **Establish readiness.** Run legacy, D-PUCT, EVOLVE, fake end-to-end, and resume suites. After CPU gates, run only an explicitly authorized tiny model smoke test. Changing the fresh-run default needs all gates and user agreement. Old resumes always retain their saved engine.

## Test requirements

Most tests must be CPU-only, deterministic, and fast. Use fakes and skip marked GPU or vLLM tests. Add root pytest configuration with explicit active test paths so `_old_stuff_` is never collected, and fix `.gitignore` before trusting new tests.

Cover config precedence and hashing, IDs and schema round-trips, event and budget idempotency, verifier classification, archive cells and provenance, role isolation, exact worker counts and seeds, frozen harnesses and bounded options, posterior allocation and reservations, randomized audit effects and no-memory persistence, group and objective correctness, nursery limits, barrier crash recovery, legacy readers, and active and complete run plots.

Common lightweight checks should remain available.

```bash
python -m compileall evolve problems train_evolve.py
python -m pytest tests evolve/tests -q -p no:cacheprovider
python -m pytest dpuct/tests -q
sh -n run.sh
```

If the full suite is too slow, document and run the smallest relevant set during iteration, then run the complete CPU suite before declaring a phase complete. Report skipped tests and why.

## Coding and change discipline

- Inspect and characterize before changing interfaces. Prefer additive typed adapters and pure policy services over a rewrite or controller monolith.
- Use atomic, content-hashed persistence. Preserve user work, unrelated files, and generated runs. Never use destructive Git operations or edit evidence to pass a test.
- Prefer existing or standard-library code. Pin and explain necessary dependencies.
- Never hide a fallback to legacy, infrastructure error, schema mismatch, partial checkpoint, adapter alias, or budget overrun.
- Prefer deterministic verification over LLM judgment. Never launch jobs or reserve GPUs without user authorization. Keep public commands documented.

When a design question is ambiguous, protect scientific validity first. Persist enough information to reconstruct the decision, separate observational production evidence from randomized causal evidence, and keep final verification independent from the mechanism that proposed the candidate.

## Definition of done

Done means both engines start and resume to their documented guarantees, problem compatibility holds, role isolation, archive, provenance, bounded branches, harnesses, posterior allocation, common verification, audit-only memory, homogeneous top-rank learning, refinement, deterministic EVOLVE recovery, complete artifacts, live answers, plots, and all CPU compatibility and crash tests work together.

Until these conditions hold, describe the work as an implementation in progress. Do not claim that EVOLVE improves scientific discovery from architecture or mock tests alone.
