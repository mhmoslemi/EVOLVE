# EVOLVE implementation status

This is the live implementation checklist for the active root `AGENTS.md`
handoff and the original contract preserved verbatim as `AGENTS-old.md`.
The repository is EVOLVE-only, and existing `runs/` directories are immutable
user evidence unless a user explicitly resumes one. No run directory was
modified during the 2026-08-29 source-audit pass.

## Current status

All implementation phases have concrete runtime code, persistence, and
reporting paths. The complete deterministic CPU suite, composed fake engine,
completed-barrier resume, partial-epoch replay, corruption, compilation, and
shell gates pass as of 2026-08-29. The user-authorized real Qwen/vLLM fresh
smoke and completed-barrier resume also pass on one NVIDIA L40S. EVOLVE's
implementation and runtime readiness gates are complete; this runtime smoke is
not evidence that EVOLVE improves scientific discovery.

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
- [x] **11 — Readiness gates.** The full CPU suite, repeated composed fake
  end-to-end run, completed-barrier resume, partial-epoch durable replay,
  corrupted training-companion rejection, and the explicitly authorized tiny
  real Qwen/vLLM fresh-and-resume smoke pass. The permanent server evidence and
  environment record are below.

## Decisions and invariants

- The active root `AGENTS.md` requires the complete original method contract in
  `AGENTS-old.md`; neither may be weakened. No PUCT, Elo, self-likelihood
  feedback, observational success memory, or alternate engine is used as an
  EVOLVE fallback.
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
  corruption all fail closed;
- exact server-smoke dry plan: Qwen/Qwen3-0.6B, real vLLM generation, HF 4-bit
  learning, three roles, one shared ordered GPU, one epoch, horizon 1, K=4,
  4096-call ceiling, and 12 inflight branches validated with config hash
  `a2629cd0d88025db9e4cce2a687a33a335ce2b18ead790adf97bf0e1d78837a8`;
  it reported `model_loading: false` and `writes_run_directory: false`.

Failures found and fixed during these gates:

- normal HF-to-vLLM phase switching released the shared CPU tokenizer before
  prompt rendering, so every branch became a zero-cost infrastructure failure;
  the tokenizer now survives model phase switches, all-infrastructure epochs
  fail before an empty barrier, and progress distinguishes verifier calls from
  infrastructure-aborted branches;
- the guarded launcher's background controller could inherit ignored SIGINT
  handling from the POSIX shell, allowing work to continue after the announced
  graceful deadline; the wrapper now restores default interrupt handling
  before executing Python so EVOLVE can drain and checkpoint normally;
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
  uses enough CPU-only verifier budget to make the recovery assertion stable;
- recursively frozen runtime mappings were sent directly to `json.dumps` while
  rendering one live prompt, causing a persisted infrastructure retry instead
  of generation;
- the resolved `thinking: false` setting was not forwarded to Qwen's chat
  template, allowing hidden reasoning to consume the complete 128-token smoke
  allowance without a final proposal or nonzero learning advantages;
- `python -m evolve.viz.run` eagerly imported its own entrypoint through the
  package initializer and emitted a `runpy` warning. Plot generation now uses a
  lazy package-level wrapper and passes with `RuntimeWarning` treated as an
  error.

For the CPU validation record above, no model was loaded, no vLLM server was
started, no CUDA/GPU benchmark ran, and no user-owned `runs/` directory was
modified.

## Real Qwen/vLLM GPU readiness record

Executed on 2026-08-29/30 on compute host `kn104` under Slurm job `5093564`.
The successful gate used one NVIDIA L40S with 46,068 MiB, driver `580.159.03`,
CUDA `13.0`, and `CUDA_VISIBLE_DEVICES=0`. The host was Ubuntu 24.04.4 LTS,
Linux 6.8.0-136, and Python 3.11.4. The activated interpreter was
`/scratch/mmoslem3/TTT-small/unsloth_env/bin/python`.

Installed runtime versions:

- PyTorch 2.13.0, Transformers 5.16.1, PEFT 0.20.0, Accelerate 1.14.0;
- bitsandbytes 0.50.1, vLLM 0.28.0, OpenAI 2.54.0;
- PyYAML 6.0.3 and NumPy 2.3.5.

The exact fresh command was:

```sh
EVOLVE_PYTHON=/scratch/mmoslem3/TTT-small/unsloth_env/bin/python sh run.sh \
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

The exact completed-barrier resume command was:

```sh
EVOLVE_PYTHON=/scratch/mmoslem3/TTT-small/unsloth_env/bin/python sh run.sh \
  --resume /scratch/mmoslem3/EVOLVE/runs/evolve_toy_Qwen3-0.6B_0829-225234_1d0769 \
  --num-steps 2
```

The permanent passing evidence directory is
`/scratch/mmoslem3/EVOLVE/runs/evolve_toy_Qwen3-0.6B_0829-225234_1d0769`.
The fresh epoch took 3m22.624s and the resumed epoch took 2m32.048s. Peak GPU
memory was not sampled externally; vLLM reported a 36.34 GiB KV cache under the
configured 85% utilization ceiling. The model fully shut down after generation
before each HF reload and learning/barrier phase.

Fresh-run evidence:

- all three roles persisted separate `adapter_epoch000` and
  `adapter_epoch001` trees, manifests, safetensors, and optimizer companions;
- scout learning group
  `learning_group:1497286472571b1b7359491133e4b6078ab86677632222f8dd9d9c708737c0ac`
  was homogeneous, on-policy, role-local, OrderGrad top-1-at-K=4;
- update
  `role_snapshot:b2f0c2493ad8bd58dbd3757b3fde8275050ac38914998646b4058e23f0ae53ec`
  had loss 9.593084335327148, KL 0.0, gradient norm 73.15972900390625,
  optimizer step 1, and changed the scout logical adapter hash from
  `df9b269f6a5d5e6432cf634cb725d2819e05ef5300e0359c5fb86d128eed4b62`
  to `58241a0a82465681335a0608b58618e33f72faed3de0447ac0b7ceda3b2a07d1`;
- mechanist and challenger remained isolated at logical hashes
  `3dac8eee926b525273bb251ff2b89b7fce848ef4a3d8d8dfd7fb4427a5cb48ef`
  and `245c28d57faa5ad37d45a4375b70e55498c53f660b7b87227425bbbef6947a5f`;
- checkpoint epoch 1 had barrier hash
  `038e7a952fd83bfdab50b350b1b88394a07adee8e4b19aa096b2d968d42c3c1b`
  and training-state SHA-256
  `04cafc15676c54d41b2cfa7975a9c387b30f1ff395064f4c84ac1f18c14dff6a`;
- 372 JSON artifacts and 57 nonblank JSONL rows parsed before resume, 12
  durable branch IDs were unique, and events 1-15 contained exactly the
  bootstrap and epoch-0 barrier commits.

Resume evidence:

- `config.resume001.json`, `config.resolved.resume001.json`,
  `manifest.resume001.json`, `command.resume001.json`, and
  `environment.resume001.json` were appended;
- SHA-256 values for the original manifest, bootstrap summary, step-0 summary,
  and step-0 allocation plan remained respectively
  `e0457a8d9c1b225a8c6f7400c241a975e54c05f0bde277b8a90dd05da567e5c1`,
  `61e74ed3df03fd6e804494cc830dce75c60a78d717a6b82abb8e29522655f246`,
  `01f966d75f15ed733fef86514728ce8f4eb3895f13055597d09199d651a7bd60`,
  and `7ec5d84403e3aa7cd98be70f14fa81fa406a0c391ba3603514204f9df58e25a5`;
- the epoch-2 training companion restored and persisted three optimizer states,
  Python RNG, Torch RNG, training-device CUDA RNG, and one visible CUDA RNG;
  the checkpoint also retained the archive, posterior, budget ledger, causal
  memory, harness registry, nursery, provenance, scheduler component versions,
  and role registry;
- each role has a distinct `adapter_epoch002` artifact and optimizer companion;
  the final epoch-2 artifact hashes are scout `48749a67b637b7a45ee078b80f336ed7c37b22140487d15b16bee467fa15ecb0`,
  mechanist `d2a973de2275f9051b07486b2afcaa80cd444511d72e37f89a33eb26645abfbe`,
  and challenger `0d40d268e9f5c607cad01a04010f0cdc10a03ff5b0aaa1da4737498475d3018c`;
- checkpoint epoch 2 had barrier hash
  `917b146e3698fa607e2de977f0887f5cf2e00732f3d0ff66c7d93bc21103dd65`
  and training-state SHA-256
  `b07301f0cb0a52dc62e93795766982243713d0ecdfc6355c651672db09373999`;
- all 12 epoch-1 durable branch IDs were unique and disjoint from the 12 old
  IDs; all 56 budget transaction keys and IDs were unique; events 1-29 were
  contiguous and idempotent with exactly one added epoch-1 barrier commit;
- 622 JSON artifacts and 100 nonblank JSONL rows parsed after resume;
  `final.summary.json` reports two of two target epochs completed; status,
  best-answer mirrors, and all checkpoints are readable.

All nine artifact-only plots regenerated after resume: record, archive,
provenance, allocation, audits, roles, posterior, failures, and resources.
Post-fix server validation passed `compileall`, `sh -n run.sh`,
`git diff --check`, the exact smoke dry plan, focused cached-Qwen tokenizer and
prompt-serialization assertions, and the complete CPU suite: **352 passed in
28.63s** with plugin autoload disabled. Pytest was loaded from a disposable
`/tmp` target because it is not installed in the shared environment.

Warnings and preserved failure evidence:

- the shared environment's `pip check` reports unrelated `inspect-ai`,
  `kernels`, and `ortools` metadata conflicts plus Unsloth/Unsloth-Zoo upper
  bounds that disagree with the installed PyTorch/Transformers versions. The
  user explicitly directed use of this activated environment; every required
  import and the complete real HF/bitsandbytes/vLLM/PEFT boundary passed, and no
  shared package was installed, upgraded, or removed;
- vLLM logged an optional `deep_gemm` import fallback because `CUDA_HOME` was
  unavailable and one post-worker EngineCore force-kill warning on the fresh
  run. Generation, graceful worker exit, HF reload, learning, and both barriers
  still completed; no `swap_space`, `NamespaceTool`, LoRA-ID collision/overflow,
  cross-role adapter, or CUDA-mask error occurred;
- the first authorized H100 attempt is preserved unchanged at
  `/scratch/mmoslem3/EVOLVE/runs/evolve_toy_Qwen3-0.6B_0829-222835_a592cf`.
  It reached one completed barrier but correctly failed the gate because no
  nonzero real learning update was persisted. Its evidence exposed the frozen
  prompt-mapping and Qwen thinking-mode defects fixed before the new L40S run.

This was a runtime compatibility smoke using `Qwen/Qwen3-0.6B` and the
deterministic CPU-verified `evolve_toy` fixture. It was not Erdos, a GPU-mode
problem, a benchmark, a scientific experiment, or evidence of improved
scientific discovery.

## Readiness result

The CPU, fake end-to-end, completed-barrier resume, targeted crash/corruption,
and real single-GPU Qwen/vLLM gates are complete. The EVOLVE implementation
Definition of Done is satisfied for the documented runtime topology and
versions. Scientific performance claims still require separately authorized,
properly designed benchmark experiments.
