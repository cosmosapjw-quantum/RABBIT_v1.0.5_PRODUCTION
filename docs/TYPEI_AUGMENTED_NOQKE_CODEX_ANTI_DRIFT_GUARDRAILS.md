# Type-I Augmented No-QKE Codex Anti-Drift Guardrails

Date: 2026-05-20

This document is mandatory context before implementing any further augmented
Type-I PSTF no-QKE PR.  It records the two May 2026 audits that found a drift
pattern: the branch was adding many diagnostic gates, manifests, claim ledgers,
and figure wrappers while the physical endpoint blocker remained almost
unchanged.  Future Codex sessions must read this document before editing code in
this programme.

The cost-effectiveness companion policy is
`bbn_codex_anti_drift_cost_effective_policy.md`.  Treat it as active for every
PR: record line-cost, token-counter availability, blocker movement ratio, and a
cost-effectiveness verdict.  Exact token counts must not be fabricated when the
harness does not expose them.

## Hard Rule

Do not create another PR whose main contribution is a new readiness gate,
manifest wrapper, figure-package wrapper, hash relay, or claim-boundary relay
unless it removes or consolidates at least one older gate and directly changes a
runtime physics, solver, or performance blocker.

BD397 adds a second hard rule: do not keep harvesting cheap local wins while the
measured activation/cold endpoint blocker remains.  If profiling shows that a
window is already cheap, or that a component is not an endpoint bottleneck, a PR
may touch it only as a parity scaffold for the larger blocker or as part of a
net deflation.  Segment-only speedups must be labeled segment-only and cannot be
claimed as endpoint progress.  New knobs, wrappers, telemetry fields, and audit
surfaces are forbidden unless the same PR either moves the measured
activation/cold wall, fixes a physics readout/calibration/parity blocker, or
deletes/consolidates more obsolete surface than it adds.

Allowed exceptions:

- A one-PR consolidation that deletes or replaces duplicated gates.
- A failing regression that exposes a concrete physics or solver defect and
  feeds the next implementation PR.
- A doc-only discipline update like this one.

Forbidden shortcuts:

- Claiming progress because a new artifact schema exists.
- Claiming endpoint progress from a benchmark that only covers a cheap tail
  window or a previously non-dominant component.
- Treating README, module names, generated registry prose, or "landed" labels
  as evidence of physical maturity.
- Adding another phase-2, payload, Jacobian, publication, or readiness knob
  without a before/after result on the measured blocker or a net reduction in
  active code surface.
- Hiding negative abundances or negative `Y_p` by output truncation.
- Adding more public-dispatch, SMC, or publication packaging before a real
  continuous full-BBN endpoint exists.
- Expanding slow Python/JAX twin implementations instead of moving an
  endpoint-consumed Rust vertical slice. Rust AOT is the active implementation
  target; SciPy/BDF is the temporary number-of-record and JAX is a frozen
  local parity/AD/Jacobian oracle.
- Claiming public production support, QKE support, or publication-ready
  all-freedom support.

## What Went Wrong

The recent AP/FB branch made real infrastructure progress, but the audit found
four recurring failure modes.

1. Evidence plumbing outgrew physics implementation.

   The AP4/AP65/FB chain accumulated many wrappers around AP68/AP72/AP75/AP79,
   figure manifests, hash propagation, and readiness ledgers.  These preserved
   claim honesty but did not move the continuous AP65 solver toward
   `T_gamma <= 0.01 MeV`.

2. Gate granularity became the unit of progress.

   Each tiny distinction became a module, script, test file, audit note, and
   generated registry entry.  This produced readable local contracts but poor
   global signal.  FB79 through FB89 show the pattern most clearly: many
   hot-endpoint span/gate refinements, still ending near `0.8 MeV`.

3. Publication and SMC surfaces arrived too early.

   Figure renderers and SMC/readiness attachments were built around diagnostic
   current artifacts before the continuous physical runner could produce a
   stable full-BBN history.  These surfaces are useful only after the physics
   runner reaches endpoint.

4. Solver and collision costs were described more often than reduced.

   Profiling identified the dominant multiplier: full finite-difference
   Jacobian probes call the expensive AP65 collision payload repeatedly.  Yet
   later PRs mostly added telemetry around the cost instead of replacing the
   hot-loop structure.

5. Local minima displaced endpoint work.

   BD397 found that recent PRs improved the already-cheap `[4.5, 4.75]` tail
   window and host-Jacobian details while the activation window
   `N in [2.5, 3.0]`, cold-endpoint phase-2 wall, asymptotic `N_eff_3T`
   readout, and bridge/table energy-transfer mismatch remained decisive.  The
   next PR must name the measured blocker it moves; otherwise it is drift even
   if the local benchmark improves.

## What Is Actually Needed

The AP/FB implementation inventory in this 2026-05-20 section is historical
after later deflation.  Preserve the anti-wrapper, raw-state, endpoint, and
measured-blocker principles; use the unified publication-code plan and current
tree for executable work selection.

Keep as bounded migration evidence or reference code:

- Continuous AP65 RHS prototype code that evaluates current-state sources.
- the minimum CPU-JAX/Rodas5P replay and chain machinery needed for frozen
  local parity, AD, Jacobian, and solver-method falsification;
- physical conventions, compact-state algorithms, finite-mass EOS methods,
  characteristic maps, failure semantics, and verified fixtures that the Rust
  implementation can redesign rather than discard.
- One raw full-BBN baseline ladder for LRS/no-collision and freedom expansion.
- One trace-positivity regression that proves the network evolves inside a
  physical or numerical invariant treatment rather than truncating outputs.
- One residual/AP65 equivalence witness until it is replaced by continuous
  source coupling.
- Claim firewall checks that prevent public dispatch, production SMC, and QKE
  overclaim.

Consolidate or stop extending:

- FB79/FB80/FB81/FB88/FB89 style span probes into one parameterized
  continuous-span experiment.
- FB82/FB83/FB86 failure-probe documents into one Yp/He4 failure-debug artifact.
- FB73/FB74/FB75/FB76/FB77 publication/SMC/readiness gates until a real
  endpoint artifact exists.
- Repeated JSON/hash/no-public/no-QKE helpers into a shared validation utility.
- Giant generated prose that lists every FB witness in one paragraph.

Delete or defer:

- New standalone policy-budget atlases that still end at the same hot endpoint.
- Current-artifact publication figure expansion before endpoint success.
- Standalone internal dispatch decision gates that merely re-state "not ready".

## Required Codex Context Prompt

Before implementation, paste or read this checklist into the working context:

```text
Read docs/TYPEI_AUGMENTED_NOQKE_CODEX_ANTI_DRIFT_GUARDRAILS.md first.

Goal: retire a named physics, solver, or performance blocker toward augmented
Type-I PSTF no-QKE full e2e BBN. QKE is out of scope. Public production support
must not be claimed. Rust AOT is the active implementation/repeated-run design
target, SciPy/BDF is the temporary number-of-record, and JAX is a frozen local
oracle. Reuse proven physics and numerical knowledge in Rust-optimized form;
do not grow another Python/JAX production twin.

Forbidden PR: a new gate, manifest, hash relay, readiness wrapper, or figure
wrapper that does not remove/consolidate an older gate and does not move the
runtime physics/solver/performance state.

Required evidence for every PR:
- What blocker is retired or measurably reduced?
- Is the benchmark an endpoint/activation/cold-blocker benchmark, or only a
  segment-only local result?
- Which old gate/wrapper is deleted, merged, or no longer extended?
- What raw physics state is preserved?
- What focused test or benchmark proves the changed executable path?
- Did self-review try to disprove the progress claim?
```

## Required Harness Prompts

Use these exact prompts when starting a new Codex PR loop in this programme.

### Implementation Harness

```text
Implement the next augmented Type-I no-QKE PR only if it retires a named
physics, solver, or performance blocker from
docs/TYPEI_AUGMENTED_NOQKE_CODEX_ANTI_DRIFT_GUARDRAILS.md. Do not add another
standalone readiness/manifest/figure/hash gate. Prefer deleting or consolidating
old gates. Do not optimize a cheap segment or non-dominant component unless it
is a scaffold for the activation/cold endpoint blocker and is labeled as such.
Keep public dispatch closed and QKE out of scope. Run focused tests and commit
only after self-review.
```

### Review Harness

```text
/review this PR adversarially. First decide whether it is real progress or
gate-only inflation. Reject it if it adds claim plumbing without changing a
runtime physics, solver, or performance blocker, or if it reports a segment-only
speedup as endpoint progress. Check for output truncation of
negative abundances, public-production overclaim, QKE overclaim, unverified
publication/SMC claims, missing raw-state preservation, and absent before/after
test or benchmark evidence.
```

### Profiling Harness

```text
Profile before porting or optimizing. Attribute endpoint time to collision
payload construction, EOS/thermo, weak/network work, solver stages, Python
control, and frozen JAX compile/runtime separately. Keep a Rust slice only if
an endpoint-consumed before/after benchmark shows a real win and the scoped
parity/positivity gates still pass. Kernel-only timing is not endpoint progress.
```

### Physics Debug Harness

```text
Debug in this order: LRS no neutrino collision, non-LRS no neutrino collision,
LRS full collision, non-LRS full collision. Report the MeV temperature region,
active freedoms, raw abundances, raw RHS signs, conservation residual, solver
diagnostics, and collision source budget. Do not repair by truncating final
observables.
```

## Historical Breakthrough WBS (superseded ordering)

The BD0-BD9 ordering below records the 2026-05-20 intervention and is retained
for provenance.  Its anti-drift principles remain controlling, but PUB-00
supersedes its PR ordering with
`docs/TYPEI_AUGMENTED_NOQKE_UNIFIED_FUTURE_PLAN.md`.

This WBS supersedes adding FB90-style standalone gates.  Each PR must include a
short self-audit table with `real_blocker_moved`, `gate_removed_or_consolidated`,
`raw_state_preserved`, `verification`, and `remaining_blocker`.

### DAG

```text
BD0 anti-drift guardrails
  -> BD1 gate consolidation and shared validation helpers
  -> BD2 continuous AP65 performance-mode RHS
  -> BD3 structured/block Jacobian prototype
  -> BD4 LRS continuous collision-coupled endpoint run
  -> BD5 non-LRS no-collision endpoint parity and geometry ladder
  -> BD6 LRS full-collision endpoint run
  -> BD7 non-LRS full-collision private endpoint run
  -> BD8 pairwise freedom composition ladder
  -> BD9 endpoint-backed figures and SMC smoke
```

BD2 and BD3 can proceed in parallel after BD1 if their write sets stay
separate.  BD9 must not start until at least BD4, BD5, and BD6 have endpoint
artifacts with raw histories below `0.01 MeV`.

### BD0: Anti-Drift Guardrails

Scope:

- Add this document.
- Add a Codex memory note with the same anti-drift rule.
- Link the guardrail from `AGENTS.md`, `ROADMAP_INDEX.md`, the state of record,
  and the current full e2e plan.
- Remove uncommitted FB90-style standalone policy-budget atlas files.

Pass:

- New PR loops have a mandatory pre-read target.
- No public production, QKE, or publication-ready support is claimed.
- Uncommitted gate-only byproducts are gone.

Fail:

- The change adds another readiness wrapper.
- The documents still say the next step is merely another span scout.

Self-audit:

- Would this have prevented FB90 from being opened as a standalone PR?  It must.

### BD1: Gate Consolidation And Shared Helpers

Scope:

- Replace FB79/FB80/FB81/FB88/FB89 wrappers with one parameterized
  continuous-span experiment surface.
- Keep old public function names only as thin compatibility wrappers if tests
  require them.
- Move repeated no-public/no-production/no-QKE, JSON-safe writing, and artifact
  hash helpers into one validation utility.

Pass:

- Net line count drops in `src/rabbit/validation`, `scripts`, and tests.
- Existing focused tests still pass through compatibility wrappers.
- One table-driven test covers span, hmax, trace-policy, and Jacobian-policy
  rows.

Fail:

- A new gate is added without deleting or folding older gate code.
- Generated capability prose gets longer.

Self-audit:

- Show the deleted or consolidated file/function list.

### BD2: Continuous AP65 Performance-Mode RHS

Scope:

- Add a hot-loop mode to the continuous AP65 RHS that avoids per-probe
  JSON-safe conversion, fingerprinting, and large trace row construction.
- Preserve raw row-boundary provenance outside the RHS/Jacobian inner loop.

Pass:

- Same tiny-span physical outputs as the diagnostic mode within configured
  tolerance.
- Fewer allocations or lower wall time in a focused benchmark.
- Raw state and claim-boundary metadata remain available at accepted steps.

Fail:

- Benchmark improvement is within noise.
- The mode hides failed raw states or changes physics silently.

Self-audit:

- Report before/after wall time, source evaluations, stage evaluations, and
  final raw observable deltas.
- A later BD75-compliant instance is
  `stage_collision_payload_policy=auto_small_collision_reuse`: it changed the
  existing FB69/FB70 runtime surface, records observed auto-reuse versus
  current-state fallback counts, and compares q4/mu5 wall-budget progress
  against exact `current_state` and fixed `step_base_reuse` without adding a
  standalone gate.

### BD3: Structured Or Block Jacobian

Scope:

- Replace full finite-difference Jacobian over the whole state with a structured
  approximation: block finite differences, sparse coloring, or analytic blocks
  for known network/thermo components.
- Keep finite-difference full Jacobian as reference for tiny grids.

Pass:

- Source evaluation count per accepted step drops materially.
- Tiny-grid reference parity passes.
- Rodas5P rejection/error telemetry does not regress.

Fail:

- Zero-Jacobian fallback is used as the main solver policy.
- Stability comes from larger tolerances rather than a defensible Jacobian.

Self-audit:

- Explain which blocks are exact, approximated, or ignored and why.

### BD4: LRS Continuous Collision-Coupled Endpoint

Scope:

- Prove the continuous architecture can reach `T_gamma <= 0.01 MeV` in the
  safest collision-coupled LRS setting before non-LRS promotion work.
- Preserve raw abundances and raw collision source diagnostics.

Pass:

- Endpoint artifact reaches below `0.01 MeV`.
- Raw `Y_p`, D/H, and mass-fraction residual are finite and physical without
  output truncation.
- Focused regression locks the endpoint contract.

Fail:

- Endpoint success depends on clipping final observables.
- The run is only a sidecar/window-map replay rather than continuous current
  RHS evolution.

Self-audit:

- Compare against LRS no-collision baseline and report MeV regions of maximum
  source/abundance stress.

### BD5: Non-LRS No-Collision Endpoint Parity

Scope:

- Expand geometry without neutrino collision terms to isolate non-LRS transport
  and weak/network effects.

Pass:

- Endpoint below `0.01 MeV`.
- LRS limit agrees with BD4 or the canonical LRS baseline when non-LRS shear is
  set to zero.
- Non-LRS metadata remains private and public dispatch stays closed.

Fail:

- Collision terms are accidentally active.
- Public `jax_characteristic_nonlrs` collision guard is weakened.

Self-audit:

- Show active freedom flags and the LRS-limit parity result.

### BD6: LRS Full-Collision Endpoint

Scope:

- Add full no-QKE collision terms in LRS and reach endpoint with the optimized
  continuous RHS/Jacobian stack.
- Current status: the private FB70 LRS collision-coupled run reaches
  `T_gamma=0.009144759667062704 MeV` with `h_max=0.1`,
  `h_refinement_factors=(1.0,0.5)`, trace-boundary abundances, and
  `frozen_source_jax`.  This is endpoint evidence for the private LRS
  continuous-AP65 path only; it is not public dispatch, production SMC, QKE, or
  non-LRS full-collision support.

Pass:

- Endpoint below `0.01 MeV`.
- Collision source moments satisfy number/energy closure diagnostics.
- Observable shifts relative to no-collision baseline are recorded from raw
  histories.

Fail:

- The run only evaluates terminal collision payloads.
- Closure diagnostics are missing or only schema-checked.

Self-audit:

- Identify the dominant collision channel and runtime cost after optimization.

### BD7: Non-LRS Full-Collision Private Endpoint

Scope:

- Enable private non-LRS full collision in the continuous path, still with
  public dispatch closed.

Pass:

- Endpoint below `0.01 MeV` in a smoke grid.
- LRS limit matches BD6 within tolerance.
- Non-LRS anisotropy observables and raw collision moments are finite.

Fail:

- The residual S2 shortcut is treated as physically equivalent without
  trajectory-level agreement.
- Any public production/canonical support is claimed.

Self-audit:

- State whether residual/AP65 equivalence is proven, disproven, or still a
  private approximation.

### BD8: Pairwise Freedom Composition Ladder

Scope:

- Re-enable weak-rate corrections, collision terms, and non-LRS geometry in
  single, pairwise, then all-freedom combinations using endpoint-capable paths.

Pass:

- Every row reaches endpoint or reports a classified physical/solver failure.
- Pairwise deltas are computed from raw histories over the same span.
- No row is hidden because it failed.

Fail:

- Rows are skipped without fail-closed metadata.
- Figure/SMC packaging is expanded before row outcomes are known.

Self-audit:

- Explain which freedom is least stable and why.

### BD9: Endpoint-Backed Figures And SMC Smoke

Scope:

- Only after endpoint artifacts exist, rebuild figures and guarded SMC smoke
  from the endpoint-backed run family.

Pass:

- Figure manifests point to endpoint raw histories and hashes.
- Plots have physical meaning: temperature histories, raw abundance histories,
  freedom deltas, collision closure, solver effort, and convergence.
- SMC remains guarded and smoke-scale unless real-data validation is separately
  completed.

Fail:

- Any plot uses hot-endpoint or legacy figure data as if it were publication
  evidence.
- Public production support is claimed.

Self-audit:

- For each figure, name the exact endpoint artifact and physical question it
  answers.

## Global Pass/Fail Standard

A PR passes only when it makes a measurable move on one of these blockers:

- endpoint span,
- positivity without output truncation,
- continuous current-state source coupling,
- solver/Jacobian cost,
- collision source cost,
- LRS/non-LRS freedom expansion,
- raw endpoint-backed figures/SMC.

A PR fails review when its strongest claim would still be true if all runtime
physics code were deleted and only schemas/manifests remained.
