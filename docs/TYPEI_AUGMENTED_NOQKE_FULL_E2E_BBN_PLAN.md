# Type-I Augmented PSTF No-QKE Full E2E BBN Plan

Date: 2026-05-20

> **Historical planning surface after BD186.**  The active future-order plan is
> now
> [TYPEI_AUGMENTED_NOQKE_UNIFIED_FUTURE_PLAN.md](TYPEI_AUGMENTED_NOQKE_UNIFIED_FUTURE_PLAN.md).
> Keep this file as provenance for landed stages and older reasoning, but do
> not use it to choose the next PR when it conflicts with the unified future
> plan.

This document originally consolidated the remaining implementation plan for the
augmented Type-I PSTF no-QKE line after FB63.  It is based on the roadmap docs,
capability registries, validation modules, forward/inference adapters, and JAX
collision/replay code available at that point.  It is now historical
provenance; use the unified future plan for active next-step ordering.

Mandatory companion:
[TYPEI_AUGMENTED_NOQKE_CODEX_ANTI_DRIFT_GUARDRAILS.md](TYPEI_AUGMENTED_NOQKE_CODEX_ANTI_DRIFT_GUARDRAILS.md).
Read it before implementing another AP/FB increment.  It supersedes the old
habit of adding standalone readiness/manifest/figure/hash gates when the
physical endpoint blocker is unchanged.

## Non-Negotiable Scope

- QKE remains out of scope.
- Do not claim public production support for augmented Type-I PSTF no-QKE.
- CPU-JAX with the in-tree Rosenbrock/Rodas5P path is the repeated-run and
  backend target.
- SciPy may remain a source-generation/reference path where still needed, but
  new production-oriented execution should not expand slow SciPy-first surfaces.
- A result is not publication-level merely because a figure exists.  A
  publication-level physics figure must consume full-BBN physical histories from
  the same validated run family that supports the corresponding claim.
- Positivity fixes must be physical or numerical-stability fixes.  Do not make
  negative BBN yields disappear by silently truncating observables.

## Current Completed State

The following surfaces are implemented as staged or diagnostic evidence:

- AP4/AP65 source-chain infrastructure, including live-source RHS sidecars,
  terminal RHS metadata, RHS deltas, restart handoff kwargs, and CPU-JAX
  Rodas5P live-source chain CLI support.
- FB50 LRS no-collision full-BBN diagnostic baseline for the current augmented
  stack.
- FB51 progressive freedom ladder for LRS/non-LRS, weak-rate correction, and
  collision toggles.
- FB52 private non-LRS residual full-BBN route, explicitly outside public
  `jax_characteristic_nonlrs` dispatch.
- FB53 residual full-BBN resolution ladder over q-grid, angular grid, and
  residual relaxation controls.
- FB54-FB56 residual/AP65/same-state and terminal-payload comparison gates.
- FB58 full-BBN diagnostic physics figures.
- FB60 full-BBN diagnostic suite bundle.
- FB61-FB63 evidence propagation from FB58/FB60 into FB23/AP75/AP79 and the
  FB45/FB46/FB48 figure-input chain.
- AP68 guarded candidate forward model plumbing.
- AP70-AP72 guarded SMC/smoke validation surfaces.
- AP73-AP80 publication artifact, readiness, weak-rate, and production-candidate
  diagnostic gates.

These completions are useful, but they do not promote the augmented no-QKE
programme to a public backend and do not yet establish publication-ready
all-freedom physics.

## Unimplemented Physics And Blockers

### Continuous Collision-Coupled RHS

Current full-chain execution is still source-refresh, window-map, or sidecar
based.  CPU-JAX replay can consume AP4/AP65 source products, but it does not yet
own a continuous live collision-coupled RHS where transport, collision source,
weak rates, thermodynamics, and BBN network evolve as one physical state.

Required work:

- Build a CPU-JAX Rodas5P RHS that recomputes the augmented AP65 combined source
  from the current state rather than only replaying or freezing terminal/source
  payloads.
- Compare continuous-live RHS results against piecewise/window-map chain
  results on identical spans.
- Track source budget, RHS delta, final BBN observable delta, and positivity
  diagnostics per window.
- Only after the continuous path is stable, scale the span toward true full-BBN
  histories below 0.01 MeV.

### Non-LRS Collision Coupling

The public JAX characteristic non-LRS route remains collision-closed:
`enable_collisions=True` is rejected in canonical forward dispatch.  The private
residual full-BBN route is useful diagnostic evidence, but it is not a physical
promotion of non-LRS collision-coupled transport.

Required work:

- Keep `jax_characteristic_nonlrs` public dispatch guarded until collision
  coupling is actually implemented and validated.
- Resolve whether the residual S2 relaxation state can be made physically
  equivalent to the AP65/PSTF source state, or replace it with a continuous
  source-coupled state that is not q-flat by construction.
- Add trajectory-level residual/AP65 agreement gates, not only terminal
  comparisons.
- Separate LRS with collision, non-LRS without collision, LRS with full
  collision, and non-LRS with full collision in every diagnostic artifact.

### Weak-Rate Corrections Over Full BBN

Coupled weak-rate gates exist, but current AP80-style convergence remains
smoke/tiny-span diagnostic evidence.  It does not yet certify weak-rate
correction stability over full-BBN evolution.

Required work:

- Extend weak-rate correction gates from tiny spans to actual BBN spans.
- Track p-to-n/n-to-p rate deltas, weak correction contributions, and BBN
  observable sensitivity along the full temperature history.
- Keep weak-rate correction, collision, and non-LRS freedoms isolated before
  testing pairwise and all-freedom combinations.

### Solver Positivity And Stiffness

Negative Y_p or D/H indicates either physical wiring error, numerical
instability, stiffness/Jacobian failure, or invalid observable extraction.  It
cannot be treated as an output-format issue.

Required work:

- Preserve raw states and raw observables in artifacts before any display-level
  filtering.
- Classify failures by MeV-scale temperature region, active freedoms, solver
  diagnostics, RHS norm/delta, Jacobian condition proxy where available, and
  source-budget sign.
- Build an ordered debug ladder:
  LRS no neutrino collision, non-LRS no neutrino collision, LRS full collision,
  non-LRS full collision.
- Fix the first physical or solver-level cause before broadening freedoms.

### Production-Scale Statistical Path

AP68/AP70/AP72 provide guarded inference and SMC smoke paths.  They are not a
production/real-data statistical validation surface.

Required work:

- Connect validated full-BBN physical run products into guarded SMC only after
  the physics runner has stable full-span histories.
- Keep synthetic/smoke and real-data validation clearly separated.
- Record model family, active freedoms, solver mode, source policy, grid
  settings, and claim scope in every likelihood/SMC artifact.

## Implemented But Not Yet Connected

The following code exists but is not yet connected to the final physical path:

- FB58/FB60 full-BBN diagnostic figures and suite are now discoverable through
  FB61-FB63, but no publication renderer consumes them as final paper figures.
- The private non-LRS residual full-BBN runner is not selectable through
  `canonical_forward_solver`, AP68 full-chain inference, or AP70/AP72 SMC as a
  validated physical model.  This is correct for now; it is diagnostic-only.
- AP65 combined source and FB56 terminal-payload gates compare source products,
  but they are not yet the continuous RHS source inside the repeated-run solver.
- FB21/FB36 live-source repeated-run diagnostics have not become the default
  full-BBN suite policy.
- FB32-FB35 vectorization/cache/reference-dispatch work exists, but current
  full-chain runtime still has Python-loop and source-refresh bottlenecks.
- AP80 weak-rate convergence evidence is not yet consumed by the full-BBN
  publication suite or figure renderer.
- AP73-AP79 readiness/publication surfaces attach evidence, but they should
  remain fail-closed until the underlying physics run products improve.

## Performance Plan

Optimization should follow current profile evidence and retain changes only
when measured wins survive focused tests.

Near-term targets:

- Profile the current full-BBN ladder with LRS/no-collision, non-LRS/no-collision,
  LRS/collision, and private residual all-freedom cases.
- Replace Python-loop finite-difference Jacobian materialization and transport
  gather/scatter construction with static sparse structures, JAX `vmap`, or
  pretabulation where the grid is fixed.
- Batch q-grid/angular-grid/source-policy sweeps where the RHS and source
  payload shapes are static.
- Cache collision kernels and source bundles by physical grid signature, not by
  ad hoc artifact path.
- Record before/after wall time, compile time, step count, failure mode, and BBN
  observable deltas for every optimization PR.

Stop optimizing a path when:

- the speedup is within benchmark noise,
- the optimized path fails parity or positivity gates,
- the path is still dominated by a larger unimplemented physics gap,
- or the optimization makes the solver less inspectable before the physics is
  stable.

## PR Roadmap

### FB64: Consolidated Plan And Audit Ledger

Create this document from actual docs/code scans and review it once before
committing.  No physics claim changes.

Exit gates:

- Document separates blockers from implemented-but-not-connected surfaces.
- No public-production or QKE claims are introduced.
- Focused docs/registry tests still pass.

### FB65: Full-BBN Figure Input Index

Status: landed in current workspace as a diagnostic, stage-scoped index.

Create a single machine-readable index that points from FB48/FB63 to the current
full-BBN diagnostic suite, figures, source manifests, and readiness metadata.
This should make figure generation reproducible without treating current plots
as publication-ready.

Exit gates:

- Index records active freedoms, source policy, solver mode, temperature span,
  and claim scope.
- It includes hashes for suite manifests and figure inputs.
- It rejects missing full-BBN endpoint evidence below 0.01 MeV.

### FB66: Freedom-Ladder Full-BBN Sweep Runner

Status: landed in current workspace as a diagnostic sweep index over FB51/FB52
artifacts.

Promote the current one-off diagnostic ladder into a repeatable runner for:

- LRS no weak correction / no collision,
- LRS weak correction only,
- LRS collision only,
- non-LRS no collision,
- LRS weak correction plus collision,
- non-LRS weak correction,
- non-LRS private residual collision,
- private residual all-freedom diagnostic.

Exit gates:

- Every run records MeV-scale failure region or full completion.
- No observable is silently truncated.
- Pairwise freedom combinations are compared to single-freedom baselines.

### FB67: Trajectory-Level Residual/AP65 Closure

Status: landed in current workspace as a diagnostic checkpoint artifact over
FB54 same-state residual/AP65 source probes.

Move beyond terminal residual comparison by evaluating AP65/source/residual
agreement along the trajectory.

Exit gates:

- Per-window source budget agreement is recorded.
- q-flat projections are labeled when present.
- A failure separates physics mismatch from solver instability.

### FB68: Collision Hot-Path Profiling And First JAX Optimization

Landed as the first profiler in this block: `augmented_nonlrs_dynamic_collision_payload_hotpath_profile_fb68_v1`
isolates the dynamic AP65 collision payload refresh used by the
CPU-JAX/Rodas5P live-source chain, comparing cache-disabled cold execution,
shared-cache cold miss, and shared-cache warm hits.  The CPU smoke in
`docs/audit/fb68_dynamic_collision_hotpath_profile.md` measured a 4-species
shared-cache cold-miss median of `1.6804822400445119 s` and warm-hit median of
`0.00948077195789665 s`, with cProfile pointing at the AP65/AP6
`pstf_radial` source factory/radial-grid construction as the next optimization
target.  Do not land a speculative optimization until the profiler shows a
before/after gain on this surface.

BD7 follow-up folded the first non-LRS full-collision hot-loop optimization into
the existing AP6/AP65 runtime path: universal PSTF angular-geometry tables are
now reused across radial-provider rebuilds with the same validated geometry,
with both outer geometry entries and per-geometry table entries bounded.
The focused CPU-JAX/Rodas5P smoke reduced a `T_nu_e_MeV`-changed radial rebuild
to `0.037700 s` and shortened the private FB70 non-LRS full-collision
`N_span_end=0.4` probe to `15.338262253906578 s` with `passed=true` and
`best_T_final_MeV=0.5419225510211213`.  This remains hot-endpoint evidence;
full endpoint coverage below `0.01 MeV` is still open.

Exit gates:

- Hot-path profile benchmark is committed.
- Cache-hit semantics, effective radial-source diagnostics, payload contract, and
  no-public/no-QKE boundaries pass.
- Slow SciPy-first production surfaces are not expanded.

### FB69: Continuous Live AP65 Source RHS Prototype

Landed as a private host-stepped Rodas5P-tableau prototype:
`augmented_nonlrs_continuous_ap65_source_rhs_prototype_fb69_v1` recomputes the
AP65 combined angular+`pstf_radial` source from the current RHS/stage state on
micro-windows, records source-evaluation traces and cache reuse, compares
adjacent step caps, and compares against the existing dynamic restart-state
payload path that freezes the payload within the JAX Rodas5P window.  The first
real CPU-JAX smoke in `diagnostic_outputs/fb69_continuous_ap65_rhs_prototype.json`
passed two step-cap rows with 178 current-state source evaluations, 17
source-factory cache entries, 162 radial-grid cache entries, finite
`Y_p`/D/H/`N_eff_3T`, and finite reference deltas.  The public jitted
CPU-JAX/Rodas5P replay and chain paths are not rerouted by this stage.

Exit gates:

- Micro-window parity against piecewise/source-refresh chain is recorded.
- RHS delta and BBN observable delta are stable under smaller step caps.
- Failure artifacts preserve raw state, source payload traces, and solver
  diagnostics.
- No public dispatch, production SMC validation, QKE, or jitted production
  continuous-AP65 claim is made.

### FB70: Physical Full-BBN Span Expansion

Landed as a private continuous-AP65 span-expansion artifact:
`augmented_continuous_ap65_full_bbn_span_ladder_fb70_v1` runs the FB69
current-state AP65 RHS prototype over increasing `N_span_end` rungs, records
the MeV endpoint reached by each rung, classifies hot endpoints and failures by
active freedom set, and checks raw `Y_p`/D/H observables without truncation.  The
first finite-difference CPU-JAX smoke reached only
`T_gamma=0.7999999999214282--0.7999999999607141 MeV`, so
`physical_full_bbn_span_ready=false`.  The current LRS collision-coupled
private run now uses FB70 `h_refinement_factors` to retry a failed late-time
window at smaller `h_max`; with `h_max=0.1`, `h_refinement_factors=(1.0,0.5)`,
chained windows through `N_span_end=4.8`, trace-boundary abundances, and
`frozen_source_jax`, it reached `T_gamma=0.009144759667062704 MeV` with
positive raw `Y_p=0.1631801917360858` and nonnegative
`D/H=2.0942876300288725e-05`.  Later BD14/BD15/BD67 private endpoint probes
extend this same FB69/FB70 surface to non-LRS no-collision, non-LRS collision,
and all-three weak+non-LRS+collision rows.  These are private no-QKE
continuous-AP65 evidence only; public dispatch, production SMC validation, and
publication-ready support remain unclaimed.

BD8 adds an FB70 composition mode rather than a new gate: the same span-ladder
builder now runs requested single-, pairwise-, and all-freedom cases, embeds
each nested raw span history, and reports pairwise terminal-row deltas against
same-ladder single-freedom controls.  A CPU-JAX/Rodas5P smoke over weak,
non-LRS, weak+non-LRS, and all-three cases with trace-boundary abundances and
`frozen_source_jax` classified all four rows but remained hot-endpoint evidence
only, with `T_gamma=0.7252382210315019--0.7252912365246144 MeV` and
`physical_full_bbn_span_ready=false`.

BD9 folds stage-domain retry, same-state base RHS/Jacobian retry caching, and a
cooperative per-row wall-time budget checked between host-step attempts into
the same FB69/FB70 private surface.  A CPU-JAX two-window non-LRS+collision
chained smoke through `N_span_end=(0.8,1.6)` passed the first hot endpoint at
`T_gamma=0.3708909114623228 MeV` and failed closed in the second window at the
60 s row budget with
`selected_step_domain_rejection_count_total=4`,
`selected_host_step_base_cache_hit_total=82`, and
`selected_frozen_source_jax_jacobian_evaluations_total=100`.  This preserves
the raw stage-domain failure evidence and confirms that non-LRS full-collision
endpoint coverage is still blocked by both endpoint completion and hot-loop
payload/Jacobian plus tiny-step cost.

BD10 adds a post-rejection step-growth cap inside that same private solver
surface.  After an accepted retry that followed a rejected attempt, the next
proposed attempted step is capped at `1.5x`, reducing stage-domain ping-pong
while preserving raw accepted states.
The same two-window smoke still fails closed at the row wall-time budget, but
stage-domain rejects drop to `1` and
`selected_post_rejection_growth_cap_count_total=21`; the remaining blocker is
now dominated by hot-loop payload/Jacobian work across many tiny accepted
steps.

BD11 adds an opt-in `stage_collision_payload_policy=step_base_reuse` on the
same FB69/FB70 private surface.  It keeps the host-step base collision payload
for unaccepted Rodas stage collision terms while still evaluating the RHS at
each stage state, records dynamic-payload-build versus stage-reuse counts, and
marks the policy as a private performance approximation.  The default remains
`current_state`.  On the same two-window non-LRS+collision CPU-JAX smoke, the
60 s row-budget run now passes both hot-endpoint rows with
`selected_dynamic_collision_payload_builds_total=237`,
`selected_stage_collision_payload_reuse_total=3071`, and terminal
`T_gamma=0.18315223284690685 MeV`.  This removes the immediate row-budget
failure but does not claim full current-state stage-payload evidence or full
BBN readiness; endpoint expansion below `0.01 MeV` and long-span solver cost
remain open.

BD12 keeps the same private solver surface and reuses one factorization of the
Rodas host-step linear system across all stage right-hand sides.  The
three-window CPU-JAX smoke with `step_base_reuse` now records
`selected_linear_system_factorizations_total=480` for
`selected_linear_system_solves_total=3838`, passes all three hot-endpoint rows,
and reaches terminal `T_gamma=0.09515938503015112 MeV`.  This is a hot-loop
linear algebra reduction only; it does not move the terminal endpoint below
`0.01 MeV`, and the remaining blocker is still endpoint expansion plus
source/Jacobian cost in the `[0.8, 1.6]` window.

BD13 adds an opt-in chain h-max policy on FB70,
`chain_h_max_policy=first_rejection_half_ceiling`, that caps later chained
windows from the first rejected step in a successful previous window.  With
base `h_max=0.1`, `step_base_reuse`, and the same three-window CPU-JAX smoke,
the policy caps two later rows, reduces selected source evaluations from
`3632` to `521`, removes the `[0.8, 1.6]` rejected attempts (`205` to `0`), and
reaches terminal `T_gamma=0.09515938504584408 MeV` in
`selected_wall_seconds_total=16.805868397117592`.  This is still private
solver policy evidence only; it does not move the endpoint below `0.01 MeV`.

BD14 keeps that policy opt-in and adds
`chain_h_max_policy=first_rejection_half_ceiling_once`, which applies the first
rejection-derived half-ceiling once and then avoids later successful-row
over-tightening.  A six-window non-LRS+collision CPU-JAX/Rodas5P smoke with
base `h_max=0.1`, `step_base_reuse`, `rhs_trace_policy=boundary`, and
`N_span_end_ladder=(0.8,1.6,2.4,3.2,4.0,4.8)` passed with
`physical_full_bbn_span_ready=true`, `terminal_completion_class=full_bbn_completed`,
terminal `T_gamma=0.009144759664500419 MeV`, and
`selected_wall_seconds_total=52.717273119837046`.  This moves the continuous
AP65 endpoint blocker for the private backend target, but it is still not a
public production-support claim; the remaining blocker is weak-rate/
convergence/statistics evidence and the `[2.4,3.2]` hot-loop cost.

BD15 connects the endpoint-capable FB70 freedom-composition rows to the existing
FB71 weak-rate convergence index instead of creating a new gate.  Scoped FB70
composition runs now preserve missing single-reference metadata without
failing the row evidence, and FB71 can require a specified weak/control context.
The non-LRS+collision endpoint pair with `weak_correction_level=3`,
`chain_h_max_policy=first_rejection_half_ceiling_once`, and the same six-window
CPU-JAX/Rodas5P ladder passes both control and weak rows below `0.01 MeV`.
FB71 reports `full_bbn_weak_rate_pairs_ready=true` for the scoped context,
`max_abs_weak_delta_Yp=0.0016890594392898195`, and
`max_abs_weak_delta_DH=7.310612378372874e-08`, while
`ap80_to_full_bbn_bridge_ready=false`.  This moves endpoint-backed weak-pair
evidence for the private non-LRS+collision context only; AP80 bridge evidence,
resolution/tolerance ladders, and statistics remain open.

BD16 passes that scoped FB70 endpoint pair through the existing FB72 AP80-FB71
bridge path.  FB72 now forwards `required_contexts` to the nested FB71 build
and accepts FB70 freedom-composition artifacts through the compatibility
`progressive_freedom_artifact` input.  The real bridge probe combines AP80
smoke `smoke_q34` (`total_nfev=7596`,
`applied_rate_q_relative_delta=0.0024445680701901517`) with the BD15
non-LRS+collision endpoint pair.  FB71 reports
`ap80_to_full_bbn_bridge_ready=true`, and FB72 reports
`ap80_fb71_bridge_ready=true`, with the same endpoint weak deltas
`max_abs_weak_delta_Yp=0.0016890594392898195` and
`max_abs_weak_delta_DH=7.310612378372874e-08`.  FB72 marks this as
`fb71_required_context_scope=scoped_subset` and keeps
`blocking_next_step=extend_scoped_bridge_to_default_context_matrix_or_resolution_tolerance_ladder`.
FB73 and FB75 reject scoped FB72 bridges by default, and FB74 rejects scoped
FB73 figure manifests by default, so this is still private scoped bridge
evidence: AP80 remains profile-level, and resolution/tolerance ladders,
publication figures, statistics, public dispatch, production SMC validation,
and QKE remain open/out of scope.

BD35 folds the missing default LRS no-collision control into the existing FB70
freedom-composition matrix.  The default matrix now has eight rows, including
`lrs_no_collision`, while direct `enabled_freedoms=()` remains rejected outside
the internal composition runner.  A real CPU-JAX/Rodas5P FB70 default run
completed all eight rows below `0.01 MeV`, and the existing FB72 bridge over
that artifact passed without scoped `required_contexts` with
`fb71_required_context_scope=default_all_contexts`,
`fb71_passed_pair_count=4`, and
`fb71_rows_reaching_full_bbn_endpoint=8`.  This clears the scoped-to-default
bridge blocker on the existing private surfaces; any figure follow-up is only a
diagnostic input refresh.  AP80 is still profile-level, and
resolution/tolerance, figure refresh, statistics, public dispatch, production
SMC validation, and QKE remain open/out of scope.

BD36 carries that default matrix into the existing FB70
`resolution_ladder_cases` mode.  Nested freedom-composition rows are now
compared by `freedom_key` across adjacent resolution cases, and nested
composition runtime/source telemetry is summed instead of being dropped.  The
real same-q tolerance smoke over the default eight-row matrix reported
`composition_resolution_tolerance_ready=true`,
`composition_resolution_comparison_count=8`,
`composition_resolution_delta_violations=[]`,
`max_abs_composition_delta_Yp=2.118909769865951e-09`, and
`selected_source_evaluations_total=56728`.  This clears the immediate
same-q tolerance check for the default private matrix; q/angular-grid
convergence, hot-loop cost, statistics, public dispatch, production SMC
validation, and QKE remain open/out of scope.  Figure refresh remains a later
diagnostic consumer of these blocker-moving checks.

BD37 keeps that same FB70 resolution surface and records the actual comparison
axes as `axis_delta_kinds`.  A real CPU-JAX/Rodas5P default-matrix q/angular
smoke comparing q3/N_mu3/N_phi5 to q4/N_mu4/N_phi6 passed with
`resolution_axis_delta_kinds=["angular_grid","q_grid"]`,
`composition_resolution_axis_delta_kinds=["angular_grid","q_grid"]`,
`composition_resolution_comparison_count=8`,
`composition_resolution_delta_violations=[]`,
`max_abs_composition_delta_Yp=4.220789247222356e-06`,
`max_abs_composition_delta_DH=2.246835889453539e-08`, and
`selected_wall_seconds_total=169.38022271264344`.  This clears a one-step
private q/angular smoke for the default matrix and moves the next blocker to
`reduce_hot_loop_payload_jacobian_cost_or_extend_grid_ladder`.  It is not a
publication-grade convergence ladder, public dispatch, production SMC
validation, or QKE support.

BD38 removes redundant metadata work from the existing FB69/FB70 hot loop.
Internal `jacobian_base` RHS calls now request RHS-only evaluation while
`rhs_initial`/`rhs_final` still preserve endpoint metadata.  One observed FB69
CPU-JAX/Rodas5P `N_span=(0,0.8)` smoke with boundary trace and
`step_base_reuse` moved from `wall_seconds_total=7.293221939005889`
(`/tmp/fb69_bd38_before_metadata_opt.json`) to
`7.103321349015459`; source and structured-Jacobian counts stayed fixed while
`rhs_only_jax_evaluations_total` rose from `112` to `126`
(`/tmp/fb69_bd38_after_metadata_opt.json`).  This is a bounded runtime code
reduction with single-smoke timing evidence only.  The AP65 dynamic
payload/Jacobian and linear-solve cost remains the main blocker for larger
q/angular ladders and statistics.

BD39 removes per-stage `dict(...)` copies around the already explicit
`step_base_reuse` payload/provenance overrides.  On the same single observed
FB69 smoke, the BD39 artifact
`/tmp/fb69_bd39_after_payload_override_nocopy.json` passed with
`wall_seconds_total=7.030492526013404`,
`source_evaluations_total=128`, `rhs_only_jax_evaluations_total=126`,
`structured_jacobian_evaluations_total=15`, and
`stage_collision_payload_reuse_total=112`.  This is hot-loop cleanup only; it
does not change physics scope, public dispatch, production SMC validation, or
QKE boundaries.

BD40 disables duplicate SciPy LU `check_finite` scans inside the existing FB69
host Rodas5P linear solve after the RHS/Jacobian/state finite checks have
already run.  On the same single observed FB69 smoke,
`/tmp/fb69_bd40_after_lu_no_check_finite.json` passed with
`wall_seconds_total=6.980848156963475` and unchanged source/RHS/Jacobian count
telemetry.  This is a bounded solver hot-loop optimization only; it does not
weaken raw-state failure preservation because FB69 now keeps explicit
non-finite linear-system matrix/RHS rejection, and it does not alter
public/QKE/production scope.

BD41 lets the same FB69 SciPy LU path use overwrite hints for the freshly formed
linear-system matrix and per-stage RHS arrays while preserving the explicit
non-finite matrix/RHS checks introduced in BD40.  The single observed FB69
smoke `/tmp/fb69_bd41_after_lu_overwrite.json` passed with
`wall_seconds_total=7.060200556064956` and unchanged source/RHS/Jacobian count
telemetry; this is copy-pressure cleanup only, not timing-improvement evidence
or a public/QKE/production scope change.

BD42 keeps `jacobian_policy=frozen_source_jax` but coevaluates the accepted
host-step base RHS and frozen-source dense Jacobian through one
`jax.jacfwd(..., has_aux=True)` function.  On the same single observed FB69
smoke, `/tmp/fb69_bd42_after_combined_only_jacfwd_aux.json` passed with
`wall_seconds_total=6.956021819030866`,
`rhs_only_jax_evaluations_total=112` instead of BD41's 126, and
`frozen_source_jax_rhs_and_jacobian_evaluations_total=15`.  This remains a
private hot-loop optimization only: the dynamic AP65 payload derivative is still
ignored by this policy, and public dispatch, QKE, and production validation
remain out of scope.

BD43 threads the existing AP6 `pstf_radial` disk radial-grid cache through FB69
and FB70 source refresh via `--source-refresh-radial-grid-cache-dir`.  The replay
path now forwards `pstf_radial_grid_cache_dir` into both non-LRS S2 and LRS
dynamic payload rebuilds without changing the exact finite-mass temperature
cache keys.  On the same FB69 smoke, a populated cache-dir repeated process
`/tmp/fb69_bd43_cache_second.json` passed with
`wall_seconds_total=6.026476241997443` versus
`/tmp/fb69_bd43_nocache_compare.json` at `7.262460570083931`; source,
RHS/Jacobian, and stage-payload counts stayed fixed.  The first population run
is slower because it writes NPZ grids, so this is repeated-process cache
evidence only, not a public dispatch, QKE, production, or publication claim.

BD44 adds an opt-in JAX persistent compilation cache control to the same FB69
and FB70 private CLIs instead of adding a new gate.  The setting is recorded in
dry-run and artifact `runtime_cache` metadata and applied before the artifact
run when `--jax-compilation-cache-dir` is supplied; the helper lives in
`augmented_validation_utils.py` with the existing shared JSON/hash/no-public
validation helpers.  The metadata records that JAX cache-hit status is not
directly measured by these artifacts.  On the same FB69 smoke with the BD43
radial cache already
populated, the first compilation-cache population run
`/tmp/fb69_bd44_jaxcache_metadata_first.json` passed at `6.252862777910195 s`; the next
separate process using the populated cache
`/tmp/fb69_bd44_jaxcache_metadata_second.json` passed at `3.1601327889366075 s`, with
source, RHS/Jacobian, and stage-payload counts unchanged from BD43.  This is
private repeated CPU-JAX/Rodas5P runtime evidence only.  It does not change AP65
physics, solver tolerances, raw-state preservation, public dispatch status, QKE
scope, production SMC validation, or publication readiness.

BD45 carries the same repeated-run cache controls through the consolidated
continuous-span surface that already backs FB79, FB80, FB81, FB88, and FB89.
`augmented_continuous_ap65_span_cli.py` now exposes
`--jax-compilation-cache-dir`, records `runtime_cache` in dry runs, configures
JAX before the existing span writer, and forwards the same metadata into nested
FB70/FB88 builders and top-level span artifacts.  The consolidated source
refresh CLI also forwards `--source-refresh-radial-grid-cache-dir` through
FB79/80/81/88 wrappers so the BD43 AP6 radial-grid cache is available to these
larger private profiles.  Focused tests lock dry-run surfacing, real-writer
ordering, and nested builder propagation.  This is existing-runner backend
plumbing only: no standalone gate, public dispatch, QKE, production validation,
raw-output repair, or full-span endpoint claim is added.

BD46 keeps that same consolidated FB88/FB89 span surface and removes a blocker
that would have forced another wrapper before real endpoint growth: the
trace-boundary extension/growth builders now accept an opt-in
`target_T_gamma_MeV` mode.  With no target supplied, FB88/FB89 preserve their
old hot-span scope and still reject nested full-BBN endpoint rows as out of
scope.  With `target_T_gamma_MeV=0.01`, endpoint-capable nested FB70/FB88 rows
are allowed inside the existing private artifacts, classified as
`trace_boundary_extension_endpoint_target_reached` or
`trace_span_growth_endpoint_target_reached`, and still keep public dispatch,
production SMC validation, publication readiness, and QKE closed.  Targets above
the full-BBN threshold are rejected, and failed or negative-observable endpoint
rows keep top-level pass closed rather than being treated as target success.  A
small real
CPU-JAX/Rodas5P target-mode smoke failed closed at the hot endpoint with
`artifact_payload_sha256=e166fcfb27df5648fd4ed880f24399efe79cdbd54d02c8caa9da82d64d1abfcb`,
`best_T_final_MeV=0.7999999921428074`,
`classification=trace_span_growth_endpoint_target_not_reached`,
`endpoint_target_reached=false`, `violations=[fb89_endpoint_target_not_reached]`,
`source_evaluations_total=81`, and conservation max `6.284872348663924e-18`.
This verifies the mode without claiming endpoint completion.  The next blocker
remains running the endpoint-target ladder far enough to reach
`T_gamma <= 0.01 MeV` and then feeding the raw endpoint history into
resolution/weak-rate/statistical consumers.

BD47 keeps endpoint targeting on that same FB88/FB89 surface and moves the
next runtime blocker instead of adding a gate.  FB70 now accepts private
`stop_at_T_gamma_MeV` row-loop early stop metadata: it stops only after a clean
row reaches the requested temperature, keeps failed/negative rows raw, and
refuses to early-stop after any prior failed physical row.  FB88 passes the
target through to FB70 in endpoint-target mode, and FB89 geometric mode adds
`target_max_span_rows` so a target run can extend the multiplicative ladder by
row budget without hand-writing another explicit ladder.  The artifact records
requested versus executed rows, target row budget, and whether the target stop
truncated execution.  This reduces wasted post-endpoint rows and manual
endpoint-ladder plumbing, but it still does not claim public dispatch,
production validation, QKE, or endpoint completion unless the raw rows actually
reach `T_gamma <= target_T_gamma_MeV` without physical-observable failures.

BD48 threads FB70's existing `enabled_freedoms` control through the consolidated
FB79/FB80/FB81/FB88/FB89 runner and CLI.  This keeps freedom-subset endpoint
probes on the same private span surface instead of adding another wrapper, and
records the active subset in dry-run and artifact inputs.  A weak-rate-only
FB89 endpoint-target smoke over the existing explicit ten-row ladder reached
`T_gamma <= 0.01 MeV` with
`artifact_payload_sha256=61eec8da223b8053bc99a1c3c02d822f5b7383f5d89fbd400f39da06aaa6bbd5`,
`classification=trace_span_growth_endpoint_target_reached`,
`best_T_final_MeV=0.00960777013815278`, and
`rows_reaching_endpoint=1`.  This is private weak-only endpoint evidence; it is
not non-LRS, not collision-term, not all-freedom, not production SMC, not QKE,
and not a public support claim.

BD49 adds a freedom endpoint matrix axis to that same consolidated FB89
surface.  `freedom_subset_cases` executes the existing trace-span growth
runner once per private freedom subset, embeds each nested FB89 artifact and
raw nested rows, and fails the parent artifact if any case fails, misses the
endpoint, or opens a forbidden claim boundary.  A real singleton matrix smoke
over weak-rate corrections only, non-LRS geometry only, and neutrino collision
terms only reached `T_gamma <= 0.01 MeV` in all three cases with
`artifact_payload_sha256=77adadce64d9726da8fe142f42d6095dbef4de2b1adbaf85182e5eebc3ae91db`,
`classification=trace_span_growth_freedom_endpoint_matrix_passed`,
`endpoint_cases_reached=3`, `best_T_final_MeV=0.009607770331335521`,
`largest_passing_N_span_end=4.75`, `source_evaluations_total=10099`, and
`stage_source_evaluations_total=8848`.  This folds singleton freedom
comparison into the existing runtime surface; pairwise/all-freedom composition,
convergence, public dispatch, production SMC validation, QKE, and publication
support remain unclaimed.

BD50 tightens the weak-rate interpretation for that matrix surface and connects
endpoint-capable histories to the existing FB70 resolution/tolerance runner.
Weak-containing `freedom_subset_cases` now require `weak_correction_level > 0`;
earlier weak-labeled smoke artifacts should be read as selector plumbing unless
rerun with a positive weak level.  A weak-level-3 matrix over weak-only,
weak+non-LRS, weak+collision, and all-freedom cases passed with
`artifact_payload_sha256=f38b5979fc54e203e90e95dd943c4a8abbc6357212201aeb2d0c643d33ccad2a`,
`endpoint_cases_reached=4`, `best_T_final_MeV=0.009607770336848207`, and
`source_evaluations_total=13414`.  FB70 `resolution_ladder_cases` also now
forward `stop_at_T_gamma_MeV` into nested ladders, so endpoint resolution
probes can stop after a clean target row instead of running post-endpoint rows.
The scoped all-freedom weak-level-3 solver-tolerance smoke passed with
`artifact_payload_sha256=d109ce304c985312b4557bb8a73a981ae746290f937057db98727d13e4b5af62`,
`resolution_tolerance_ready=true`, both nested cases truncating `3` requested
rows to `2`, and adjacent deltas
`max_abs_delta_Yp=2.107925639593944e-09`,
`max_abs_delta_DH=6.259600207486897e-11`, and
`max_abs_delta_T_final_MeV=1.3748550908854185e-13`.  This remains private
endpoint resolution evidence only; q/angular convergence, public dispatch,
production SMC validation, QKE, and publication support remain open or out of
scope.

BD51 continues the runtime-blocker line on the existing FB69/FB70 private
surface.  The first host step can already reuse the serialized `rhs_initial`
boundary RHS from BD18; BD51 now pairs that seed with a Jacobian-only
`frozen_source_jax` replay rather than calling the combined RHS+Jacobian JAX
function and discarding its RHS output.  A focused one-row FB69 before/after
smoke moved
`frozen_source_jax_rhs_and_jacobian_evaluations_total` from `1` to `0` while
keeping `source_evaluations_total=9`,
`dynamic_collision_payload_builds_total=2`, and
`initial_boundary_rhs_seed_reuse_total=1`.  The endpoint-capable six-window
all-freedom weak-level-3 FB70 smoke
`/tmp/fb70_bd51_after_seed_jacobian_only_endpoint_smoke.json` passed with
`artifact_payload_sha256=4b063a34c589f56e04322f20154b2e69ec7ca30ec354b941607a0eae73e4a984`,
`physical_full_bbn_span_ready=true`, `T_final_MeV_min=0.00914475966407264`,
`selected_step_count_total=390`,
`selected_initial_boundary_rhs_seed_reuse_total=6`,
`selected_frozen_source_jax_jacobian_evaluations_total=390`, and
`selected_frozen_source_jax_rhs_and_jacobian_evaluations_total=384`.  This is
bounded CPU-JAX/Rodas5P hot-loop cost removal only; it does not claim public
dispatch, production SMC validation, QKE, q/angular convergence, or
publication-ready support.

BD52 removes another host-loop overhead source from the same private runtime
surface by replacing the solver retry-cache key's JSON state fingerprint with
an exact contiguous-byte state key.  Emitted trace rows still use JSON-safe
fingerprints for artifact provenance; only the internal `_HostStepBase` cache
key changes.  A local 20,000-call microbenchmark over a 41-component state
measured the old JSON fingerprint at `0.8712652230169624 s` versus the byte key
at `0.06128845305647701 s` (`14.215813576075998x` faster for this isolated
function).  The focused FB69 smoke
`/tmp/fb69_bd52_after_byte_retry_key.json` passed with
`artifact_payload_sha256=b62cf848c121eb23009141f5c64cf06085f11730d0bda0b26f02216f4b5259ac`,
unchanged `source_evaluations_total=9`,
`dynamic_collision_payload_builds_total=2`, and boundary-trace provenance.
This is internal retry-cache overhead reduction only; it does not alter
physics, raw-state reporting, public dispatch, production SMC validation, QKE,
or publication readiness.

BD53 removes the JAX restart-serializer/device-transfer detour from FB69's
current-state dynamic-payload build.  The hot-loop
`_state_to_current_restart_kwargs(...)` path now unpacks the accepted RHS vector
directly from the replay layout's host NumPy slices while preserving the same
restart contract, layout metadata, raw `A_modes`, and raw `X0` fields consumed
by the existing AP65 payload builder.  A local 20,000-call benchmark over the
41-component FB69 smoke state measured the old serializer path at
`15.329722063965164 s` and the host-slice path at `1.2295497520826757 s`
(`12.467752555761878x` faster for this isolated conversion).  The focused FB69
smoke `/tmp/fb69_bd53_after_host_restart_slice.json` passed with
`artifact_payload_sha256=3f2946a8b76e32d479584eea39bb06d83696bc58d23a8c6fb99086c2251939a6`,
`source_evaluations_total=9`,
`current_restart_payload_builds_total=2`,
`dynamic_collision_payload_builds_total=2`, and boundary-trace provenance.
This is private hot-loop conversion work only; raw states, fail-closed negative
abundance behavior, public dispatch status, production SMC validation, QKE
scope, and publication readiness are unchanged.

BD54 removes another redundant metadata cost inside the existing continuous
AP65 CPU-JAX/Rodas5P hot loop.  Dynamic payloads built for suppressed inner-loop
RHS/Jacobian calls now use `payload_metadata_policy="hot_loop_minimal"` so they
avoid restart-state, refresh-config, q-grid, and diagnostics fingerprint
construction.  Boundary `rhs_initial` and `rhs_final` payloads still emit full
metadata and fingerprints.  A 5,000-call fake-source benchmark measured full
metadata at `1.2704691931139678 s` versus hot-loop minimal metadata at
`0.5435653830645606 s` (`2.3372886366515933x` local function-level speedup).
The relevant focused tests passed with `83 passed`.  This is private hot-loop
metadata cost removal only; public dispatch, production SMC validation, QKE,
publication readiness, raw-state preservation, and negative-output fail-closed
behavior are unchanged.

BD55 clears the tiny-grid full-endpoint step-budget blocker inside the existing
FB70 span-ladder surface.  The formerly failing `[2.4,3.2]` window was traced
to `continuous AP65 host Rodas5P prototype exceeded max_steps`, so FB70 now
accepts `max_step_retry_factors`, preserves the failed low-budget attempt, and
retries the same row with a larger `max_steps` budget before handing off the
recovered restart state.  The real baseline retry artifact reached
`T_final_MeV=0.009139193822551734`, and the existing eight-case
freedom-composition artifact
`diagnostic_outputs/bd55_blocker_debug/fb70_case_matrix_maxstep_retry.json`
reported `physical_full_bbn_span_ready=true`, `rows_full_bbn_completed=8`,
`failed_or_exception_rows=0`, and
`artifact_payload_sha256=986c64c2296de5115ab44a6f93e233dc423cf7915b63249b0803cbfb0c013cf4`.
The generated endpoint comparison tables and plots are diagnostic-only.  This
does not add a gate, does not hide raw failed attempts or negative outputs, and
does not change the no-public/no-production/no-QKE boundary.  The next blockers
are q/angular and tolerance convergence at stronger grids, hot-loop
AP65/Jacobian/Rodas cost, and statistics-pipeline evidence.

BD17 folds endpoint tolerance comparison into the existing FB70 private surface
rather than adding a new gate.  The new `resolution_ladder_cases` mode runs
multiple nested FB70 span ladders, preserves each raw span history, and compares
adjacent terminal `Y_p`, D/H, and `T_gamma` values against configured
tolerances.  A real two-case non-LRS+collision CPU-JAX/Rodas5P endpoint probe
over the BD14 six-window ladder compared `rtol/atol=(1e-8,1e-10)` with
`(5e-9,5e-11)`.  Both rows completed below `0.01 MeV`, and FB70 reports
`resolution_tolerance_ready=true` with
`max_abs_delta_Yp=3.2406321515132674e-09`,
`max_abs_delta_DH=5.394384165228962e-11`, and
`max_abs_delta_T_final_MeV=8.066776413517829e-13`.  This is private scoped
endpoint tolerance evidence only; q-grid/angular-grid convergence,
publication figures, public dispatch, production SMC validation, and QKE remain
open/out of scope, and the middle-window payload/Jacobian hot loop remains the
dominant runtime blocker.

BD18 removes one repeated same-state hot-loop AP65 payload/RHS evaluation per
FB69 row inside the existing FB69/FB70 private surfaces.  The already-recorded
`rhs_initial` boundary RHS now seeds the first host-step base instead of
rebuilding an immediate `jacobian_base` at the same `(N, y)`.  A focused FB69
probe records `initial_boundary_rhs_seed_reuse_count=1`,
`dynamic_collision_payload_build_count=2`, and `source_evaluation_count=9` for
the one-step boundary-trace case.  The six-window non-LRS+collision endpoint
run reports `selected_source_evaluations_total=3307`,
`selected_dynamic_collision_payload_builds_total=409`, and
`selected_initial_boundary_rhs_seed_reuse_total=6`; a paired no-seed control
keeps the same terminal `T_gamma`, `Yp`, and D/H while requiring
`3313` source evaluations and `415` dynamic payload builds.  This is a
hot-loop cost reduction only: public dispatch, production SMC validation, QKE,
q/angular convergence, and publication-ready claims remain closed.

BD19 keeps the same existing FB69/FB70 surfaces and makes restart-state payload
construction lazy.  Under `stage_collision_payload_policy=step_base_reuse`,
stage RHS calls already receive the host-step base collision payload, so they
skip `_state_to_current_restart_kwargs` unless a fresh dynamic AP65 payload is
actually needed.  A focused one-row probe records
`current_restart_payload_build_count=2` for `source_evaluation_count=9`; the
same six-window endpoint run records
`selected_current_restart_payload_builds_total=409`,
`selected_dynamic_collision_payload_builds_total=409`, and the unchanged
terminal `T_gamma`, `Yp`, and D/H from BD18.  This moves hot-loop payload
construction cost only; public dispatch, production SMC validation, QKE,
q/angular convergence, and publication-ready claims remain closed.

BD20 keeps the default adaptive Rodas controller unchanged but exposes
`adaptive_step_safety` as a validated opt-in solver-control axis on FB69 and
FB70.  On the same six-window private non-LRS+collision endpoint configuration,
`adaptive_step_safety=0.93` preserved `passed=true`,
`physical_full_bbn_span_ready=true`, and raw positive endpoint observables while
reducing selected host steps `403 -> 389`, source evaluations `3307 -> 3258`,
dynamic/restart payload builds `409 -> 395`, and frozen-source JAX Jacobians
`403 -> 389` versus the default `0.90` run.  This is private performance-mode
runtime evidence only; it does not claim a portable wall-time guarantee, public
dispatch, production SMC validation, QKE, q/angular convergence, or
publication-ready support.

BD21 removes unused metadata construction from the existing frozen-source JAX
Jacobian autodiff closure by allowing replay RHS calls to request
`return_metadata=False`.  Boundary and final RHS calls still preserve metadata
and raw states; the Jacobian closure differentiates only the RHS vector it
consumes.  FB69/FB70 CLIs now expose the existing
`abundance_positivity_policy` axis so the `trace_boundary` endpoint-evolution
policy can be reproduced from scripts without truncating reported outputs.  On
a six-window private endpoint reproducibility run with
`adaptive_step_safety=0.93` and `trace_boundary`, the local CPU-JAX/Rodas5P run
preserved `passed=true`, `physical_full_bbn_span_ready=true`, raw positive
terminal observables, and the endpoint below `0.01 MeV`.  The supported BD21
claim is the removal of unused Jacobian replay metadata work on the existing
private surface; those trace-boundary endpoint counts are not isolated
step-count evidence against BD20's default-positivity run.  Public dispatch,
production SMC validation, QKE, q/angular convergence, wall-time portability,
and publication-ready support remain unclaimed.

BD22 extends the RHS-only metadata policy to unaccepted Rodas stage RHS calls
and finite-difference Jacobian probes, which also consume only the RHS vector.
The one-row boundary-trace regression records `source_evaluation_count=9` and
`boundary_source_evaluation_count=2` while `_live_source_metadata_payload` is
called only for the two boundary rows.  The six-window private endpoint
reproducibility run with `adaptive_step_safety=0.93` and `trace_boundary`
preserved `passed=true`, `physical_full_bbn_span_ready=true`, endpoint
observables, `selected_step_count_total=381`, and
`selected_source_evaluations_total=3201`; local wall time was
`47.029263105010614 s`.  This is bounded hot-loop work removal inside the
existing private surface, not a portable speedup, public dispatch, production
SMC validation, QKE, q/angular convergence, or publication-ready support.

BD23 adds a bounded structural cache for frozen-source JAX Jacobian functions
across equivalent FB69 contexts, avoiding repeated `jax.jacfwd`/`jax.jit`
closure construction across FB70 windows when the replay layout, source grid,
rate-table identity, and scalar RHS controls match.  The six-window private
endpoint reproducibility run with `adaptive_step_safety=0.93` and
`trace_boundary` preserved `passed=true`, `physical_full_bbn_span_ready=true`,
endpoint observables, `selected_step_count_total=381`, and
`selected_source_evaluations_total=3201`; local wall time was
`40.10995108494535 s`.  This is backend setup reuse inside the existing
private CPU-JAX/Rodas5P surface, not a portable speedup, public dispatch,
production SMC validation, QKE, q/angular convergence, or publication-ready
support.

BD25 adds `jacobian_policy=frozen_source_finite_difference`, a structured
finite-difference fallback/reference policy that reuses the host-step base AP65
collision payload for Jacobian probes instead of rebuilding Python/NumPy
payloads for each perturbed column.  Full finite-difference remains the
tiny-grid reference, and `frozen_source_jax` remains the repeated-run backend
target.  A one-row CPU-JAX/Rodas5P smoke with boundary traces and
`stage_collision_payload_policy=step_base_reuse` preserved `passed=true` while
reducing dynamic/restart payload builds from `43` to `2` versus full
finite-difference; both runs still performed `41` RHS-column probes.  This
moves AP65 payload-build cost for finite-difference Jacobian probes only; it
does not change RHS physics, truncate raw states, claim public dispatch,
production SMC validation, QKE, or publication-ready support.

BD26 carries that opt-in policy through the existing FB70 endpoint ladder
rather than leaving it CLI-only: the FB70 builder allowlist, terminal rows,
h-refinement attempts, nested resolution/freedom rows, and selected/total
summary telemetry now preserve
`frozen_source_finite_difference_jacobian_evaluation_count`.  A one-rung FB70
CPU-JAX/Rodas5P smoke with boundary traces and
`stage_collision_payload_policy=step_base_reuse` preserved `passed=true` while
reporting `selected_frozen_source_finite_difference_jacobian_evaluations_total=1`,
`selected_jacobian_probe_source_evaluations_total=41`, and
dynamic/restart payload builds of `2`.  This fixes runtime-policy propagation
inside the existing endpoint ladder; it is not a new gate and does not change
the repeated-run target, public dispatch status, QKE scope, or raw-observable
policy.

BD27 adds a cached RHS-only JAX execution path for inner-loop FB69 calls that
request `return_metadata=False` under the repeated-run
`jacobian_policy=frozen_source_jax` target.  Boundary/final metadata calls are
unchanged, and finite-difference reference/fallback policies keep the eager RHS
path to avoid first-compile overhead on tiny grids.  A one-row FB69 smoke
reported `rhs_only_jax_evaluations_total=7`, `source_evaluations_total=9`, and
`passed=true`; the paired frozen-source finite-difference fallback preserved
`rhs_only_jax_evaluations_total=0` and `jacobian_probe_source_evaluations_total=41`.
The FB70 one-rung smoke propagated the same count as
`selected_rhs_only_jax_evaluations_total=7`.  This moves RHS-only stage/probe
execution cost inside the existing private CPU-JAX/Rodas5P path; it is not a
new gate and does not alter RHS physics, raw-state preservation, QKE scope, or
public-production support.

BD28 removes repeated setup work from the existing dynamic AP65 payload-refresh
hot loop.  Current-state restart mappings used only for collision-source
payload construction now unpack directly with NumPy instead of going through a
host CPU-JAX replay-vector pack/unpack cycle, while actual CPU-JAX/Rodas5P solve
replay contracts remain unchanged.  The same PR also reuses same-geometry
unit-direction PSTF radial momentum-delta weights through the existing
bounded byte-budgeted static-delta cache; temperature-dependent radial grids and source values
remain rebuilt from the current RHS state, and radial-grid cache counters remain
grid-only.  The six-window private FB70 endpoint smoke still reported
`passed=true`, `physical_full_bbn_span_ready=true`,
`T_final_MeV_min=0.009144759664395794`, `selected_step_count_total=389`, and
`selected_dynamic_collision_payload_builds_total=395`, with local wall evidence
`21.18 s` before BD28 versus `21.05 s` after BD28.  This is hot-loop runtime
work on the existing private surface, not a new gate, public dispatch claim,
production SMC validation claim, QKE support, or output truncation.

BD29 continues the same performance-blocker path by caching AP6 radial
moment-weight/projection bundles by their true dependencies rather than by
descriptor-specific radial kernel grids.  The bounded in-module cache keeps
`radial_grid_cache` grid-only while reducing the six-window private FB70
cProfile `build_pstf_process_radial_moment_weights(...)` calls from BD28 `7110`
to `1`; the non-profiled endpoint smoke preserved `passed=true`,
`physical_full_bbn_span_ready=true`, `T_final_MeV_min=0.009144759664395794`,
and `selected_step_count_total=389`, with local wall evidence
`21.05 s` before BD29 versus `19.97 s` after BD29.  Remaining runtime blockers
are radial grid/source rebuild cost, JAX compile/cache misses, and Rodas5P
Jacobian/LU work.  This does not alter collision physics, public dispatch,
production SMC validation, QKE scope, publication readiness, or raw-state
preservation.

Exit gates:

- Temperature history and endpoint readiness are recorded for every rung.
- BBN observables remain physical without truncation for passing rows.
- Failures and hot endpoints are classified by temperature region and active
  freedoms.
- Stage-domain rejects and row wall-time budget failures remain private
  solver/runtime telemetry, not public readiness gates.
- No public dispatch, production SMC validation, QKE, or publication-ready
  all-freedom full-BBN claim is made.

### FB71: Full-BBN Weak-Rate Convergence

Landed as `augmented_full_bbn_weak_rate_convergence_fb71_v1`, a private
diagnostic index over the FB52 full-BBN freedom ladder.  It pairs weak-off and
weak-on rows for LRS/no-collision, non-LRS/no-collision, LRS/collision, and
private residual non-LRS/collision contexts; requires full-BBN endpoint coverage
and raw `Y_p`/D/H bounds; records observable deltas and optional AP80
profile-level weak-rate convergence evidence.  The first real index over the
FB52 artifact passed all four full-BBN weak pairs, but
`ap80_to_full_bbn_bridge_ready=false` because no AP80 JSON artifact was supplied
for that run.

Exit gates:

- Weak-rate correction sensitivity is tracked over full-BBN endpoint rows.
- Weak-only and pairwise contexts are indexed before downstream convergence
  promotion.
- Observable deltas are compared to weak-off controls in the same active-freedom
  context.
- AP80 remains diagnostic evidence unless explicitly supplied and validated; no
  public dispatch, production SMC validation, QKE, or publication-ready claim is
  made.

### FB72: AP80-FB71 Full-BBN Weak-Rate Bridge

Landed as `augmented_full_bbn_weak_rate_bridge_fb72_v1`, a private diagnostic
bridge between AP80 profile-level coupled weak-rate convergence evidence and
the FB71 full-BBN weak-control pair index.  The bridge can generate an AP80
smoke or extended profile artifact, build a nested FB71 index with that AP80
artifact supplied, and fail closed unless both sources agree on AP80 profile
count, profile names, and the applied-rate q-ladder delta.  A real CPU smoke
using the FB52 full-BBN freedom ladder passed with
`ap80_fb71_bridge_ready=true`, `ap80_profile_count=1`,
`ap80_total_nfev=7596`, `ap80_applied_rate_q_relative_delta_abs_max=0.0024445680701901517`,
`fb71_passed_pair_count=4`, and `fb71_rows_reaching_full_bbn_endpoint=8`.

Exit gates:

- AP80 profile evidence is present, passed, violation-free, and still
  diagnostic.
- The nested FB71 index reports all required full-BBN weak/control pairs passed.
- AP80 and FB71 profile metadata agree exactly before bridge readiness is true.
- No public dispatch, production SMC validation, QKE, promotion-grade weak-rate
  convergence, or publication-ready all-freedom full-BBN claim is made.

### FB73: Publication Figure Renderer V2

Landed as `augmented_publication_figure_renderer_v2_fb73_v1`, a new
current-artifact renderer that consumes FB60, FB66, FB70, and FB72 artifacts
directly and does not call the legacy plotting modules.  The first real render
writes four diagnostic PNG panels: full-BBN endpoint coverage,
freedom-ladder terminal yields, AP80-FB71 weak-rate bridge deltas, and the
continuous-AP65 span boundary.  The manifest records source artifact hashes,
plot hashes, captions, `legacy_plot_code_reused=false`,
`current_artifact_inputs_only=true`, and `publication_figure_ready=false`.

Real current render evidence:

- `artifact_payload_sha256=6a62a214b0d13e38e5d76bac652c8ce02caf4ec93880157b49908a75a567ceae`
- `plot_count=4`
- `full_bbn_T_final_MeV=0.004996944944314105--0.005000010963688484`
- `freedom_sweep_completed_rows=8`
- `weak_rate_bridge_passed_pair_count=4`
- `continuous_ap65_physical_span_ready=false`
- `publication_readiness_blocker=continuous_ap65_full_bbn_span_not_ready`

Remaining figure families stay future work until the corresponding current
artifacts exist: time histories through BBN termination, source-budget
trajectories, residual/AP65 checkpoint closure, solver/stiffness
failure-region diagnostics, and performance scaling panels.

Exit gates:

- Every figure has provenance, input hashes, claim scope, and caption metadata.
- Diagnostic figures are labeled diagnostic.
- Publication-ready is false until the corresponding full-span physics gate
  passes.

### FB74: Publication Bundle QA Gate

Landed as `augmented_publication_figure_bundle_qa_fb74_v1`, a QA/copy gate
over the FB73 current-artifact figure manifest.  It validates the FB73
contract, recomputes the embedded FB73 payload hash, verifies source artifact
and PNG hashes, checks diagnostic claim labels and captions, copies the four
PNG files into a clean QA bundle, and writes an FB74 manifest with explicit
QA check rows.

Real current QA evidence:

- stable rerun `artifact_payload_sha256=e22a8f6ea68b24e376b1ded12b6bb531199005bead8b4b7ef6d187f76f645e45`
- manifest file SHA256 `d609ba75756bbb9be0c7dd1fa256b6ad167eda1974a005527f8a08354c664cd5`
- `plot_count=4`
- `copied_plot_count=4`
- `qa_checks=10`
- source FB73 payload SHA256 `6a62a214b0d13e38e5d76bac652c8ce02caf4ec93880157b49908a75a567ceae`
- `publication_figure_ready=false`

Exit gates:

- The bundle can be regenerated from committed scripts and recorded artifacts.
- Captions do not overclaim public production or QKE.
- Missing physical-span or freedom-ladder evidence fails the gate.

### FB75: Guarded SMC Pilot On Validated Full-BBN Products

Landed as `augmented_guarded_smc_pilot_gate_fb75_v1`, a fail-closed diagnostic
pilot gate over the AP72 full-chain physical-smoke validation artifact, FB60
full-BBN diagnostic suite, FB66 freedom-ladder sweep, FB70 continuous-AP65
span ladder, FB72 AP80-to-FB71 weak-rate bridge, and FB74 figure QA bundle.
The gate requires file-backed source hashes, recomputes embedded payload hashes
for hashed sources, verifies closed public/production-SMC/QKE claim boundaries,
rejects missing AP72 physical-smoke evidence, rejects non-full-BBN
FB60/FB66/FB72 products, rejects inconsistent FB70 physical-span claims, and
records an AP69 SMC schema snapshot for guarded statistical pilot input wiring.
It does not run a new SMC sampler.

Real current gate evidence:

- `artifact_payload_sha256=18841d947067979eb5cdfddeef1a4c55656fbc62e92257a6e63197820bfea352`
- manifest file SHA256 `6087af94215ff25628c18e7a5fa3fd9a22ae2981166ec0ae467d6c036e661922`
- `guarded_smc_pilot_input_ready=true`
- `validated_full_bbn_product_inputs_ready=true`
- `statistical_pilot_input_ready=true`
- `source_hashes_checked=true`
- `runs_new_smc_sampler=false`
- `fb66_completed_rows=8`
- `fb72_rows_reaching_full_bbn_endpoint=8`
- `pilot_blockers=[continuous_ap65_full_bbn_span_not_ready]`
- FB70 remains hot-endpoint evidence with `physical_full_bbn_span_ready=false`,
  `rows_reaching_endpoint=0`, and
  `T_final_MeV=0.7999999999214282--0.7999999999607141`

Exit gates:

- Synthetic/smoke and real-data modes are separate.
- The gate remains a manifest over validated diagnostic products, not a sampler
  run or a public dispatch path.
- No public production inference, production SMC validation, or QKE claim is
  made.

### FB76: Internal Candidate Dispatch Decision

Landed as `augmented_internal_candidate_dispatch_decision_fb76_v1`, a
fail-closed decision record over the FB75 guarded SMC pilot-input gate.  The
artifact consumes FB75 as a file-backed, hash-checked input; rechecks every
FB75 nested source file SHA; rejects legacy sampler-readiness overclaim keys;
verifies that the AP68 callable symbols are present without running solves;
and snapshots the registry state showing
`jax_typeI_augmented_pstf_noqke_staging` remains absent from
`CAPABILITY_BY_BACKEND`.

Real current decision evidence:

- `artifact_payload_sha256=d361d9dab63dde7b54fed1656b6d7f61f5a10ec2ffdb2e6467dbb4d3f3b09518`
- manifest file SHA256 `2e871e1ec920371779bb22570e69ec4925369e0ef51d3a4f005cbb466604eac3`
- `internal_candidate_dispatch_decision=defer`
- `internal_candidate_dispatch_warranted=false`
- `registers_dispatch=false`
- `canonical_forward_solver_registered=false`
- `candidate_dispatch_registered=false`
- `decision_blockers=[continuous_ap65_full_bbn_span_not_ready]`

Exit gates:

- The decision artifact is reproducible from the committed CLI and FB75 input.
- Registry metadata names every blocker that remains.
- Public canonical dispatch remains unchanged unless separate promotion gates
  are passed.
- No public production inference, production SMC validation, sampler execution,
  or QKE claim is made.

### FB77: Claim Readiness Review

Landed as `augmented_claim_readiness_review_fb77_v1`, a final claim-readiness
ledger over the FB76 internal candidate-dispatch decision.  The artifact
hash-checks the FB76 payload, requires FB76 nested source-hash verification,
hashes the current roadmap documents with FB77 self-reference artifact-hash
lines redacted, and states the strongest defensible claim without changing
dispatch or running any solver/sampler path.

Real current review evidence:

- `artifact_payload_sha256=e38abd1c8de1b7f61755fff396c9465a3679ba0f657676fa2b934b931da06f95`
- manifest file SHA256 `13c83abdb5fa0f66f61f2158f5f4e50b9c89d1dbf9a178ac8c41ad2babdddd13`
- `claim_readiness_level=diagnostic_evidence_chain_ready`
- `strongest_defensible_claim_key=guarded_internal_diagnostic_evidence_chain`
- `public_dispatch_ready=false`
- `production_smc_validation_ready=false`
- `publication_ready_all_freedom_full_bbn=false`
- `qke_scope=out_of_scope`
- `registers_dispatch=false`
- `recommended_next_physics_pr=extend_continuous_ap65_full_bbn_span_to_0p01_MeV`
- `remaining_blockers=[continuous_ap65_full_bbn_span_not_ready, public canonical dispatch remains unregistered, production SMC validation remains absent, QKE remains out of scope, publication-ready all-freedom full-BBN support remains unclaimed]`

Exit gates:

- The strongest defensible claim is explicitly stated.
- Remaining blockers are listed with evidence.
- Public production and QKE boundaries remain closed unless future work truly
  changes them.

### FB78: Continuous AP65 Chained Span Handoff

Status: landed in current workspace as a private chained-window span
diagnostic.  FB78 extends FB69/FB70 so terminal restart state from one
continuous-AP65 micro-window can seed the next.  FB69 accepts supplied restart
kwargs and emits terminal restart kwargs for finite rows.  FB70 can run the
span ladder in `chain_restart_handoff` mode, turning `N_span_end` rungs into
consecutive windows rather than independent replays from the initial state.

Real current evidence:

- `artifact_payload_sha256=463418cba619ef8199b642debcd3425f54a3fd21f24b62038a85ecba5f1e46b9`
- manifest file SHA256 `f3c22071c252c990041aea33471db0b52ecb2da59cddf59872300ba84bdc36fa`
- `span_rows=4`
- `restart_handoff_ready_rows=4`
- `source_evaluations_total=588`
- `step_count_total=10`
- `T_final_MeV_min=0.799999999607141`
- `T_final_MeV_max=0.7999999999607141`
- `physical_full_bbn_span_ready=false`
- `rows_reaching_endpoint=0`

Exit gates:

- Supplied restart kwargs are accepted by FB69 and fingerprinted in artifact
  inputs.
- FB69 terminal rows emit JSON-safe restart kwargs from the actual final state.
- FB70 chained mode propagates only passing physical rows and fails closed if a
  previous window lacks a valid restart.
- Real chained smoke evidence records consecutive spans and remains explicit
  about the hot-endpoint blocker.
- No public dispatch, production SMC validation, QKE, or publication-ready
  all-freedom full-BBN claim is made.

### FB79: Continuous AP65 Span Bracket

Status: landed in current workspace as a private profile-level stability
bracket.  FB79 runs multiple chained FB70 profiles with the same physics and
solver controls, keeps nested FB70 failure regions, and reports the last
passing profile plus the first observed failing endpoint.

Real current evidence:

- `artifact_payload_sha256=e1c73bdae84d013a3ac0551bff404716f78bf3fdcd37a337326f3f5740e8df35`
- manifest file SHA256 `2cb311b2779809db259d0574b26c1def21057ebbe152ddb97711fdf82d260557`
- `bracket_status=pass_fail_bracketed`
- `largest_passing_N_span_end=5e-10`
- `first_failing_N_span_end=1e-09`
- `best_passing_T_final_MeV=0.799999999607141`
- `first_failing_T_final_MeV=0.7999999992142808`
- `physical_full_bbn_span_ready=false`

Exit gates:

- Profile-level pass/fail bracket is computed from nested FB70 artifacts.
- Nested FB70 public/production/QKE boundaries are checked fail-closed.
- First observed failing endpoint and failure-region evidence are retained.
- No public dispatch, production SMC validation, QKE, or publication-ready
  all-freedom full-BBN claim is made.

### FB80: Continuous AP65 h_max Sensitivity

Status: landed in current workspace as a private step-size sensitivity
diagnostic.  FB80 fixes the target continuous-AP65 span at `N_span_end=1e-9`
and sweeps `h_max=(1e-9,5e-10,2.5e-10)` through nested FB70 runs to classify
whether the FB79 first failure is recovered by internal step refinement.

Real current evidence:

- `artifact_payload_sha256=84d6aac41fc673889320ebc5802fa78049977917da193ad9154fc487048558e4`
- manifest file SHA256 `93a3dc65aa573f0217063fc947084caa97f6c5409e6d592cf8a0d8005a927912`
- `classification=h_max_refinement_recovers_observable_failure`
- `target_N_span_end=1e-09`
- `largest_failing_h_max=1e-09`
- `first_passing_h_max_after_failure=5e-10`
- `smallest_passing_h_max=2.5e-10`
- `rows_failed=1`
- `rows_passed=2`
- `physical_full_bbn_span_ready=false`

Exit gates:

- The h_max ladder is strictly decreasing and holds the span/physics controls fixed.
- Nested FB70 public/production/QKE boundaries are checked fail-closed.
- Unexpected nested failure classes make the FB80 artifact fail closed.
- No public dispatch, production SMC validation, QKE, or publication-ready
  all-freedom full-BBN claim is made.

### FB81: Continuous AP65 Refined Span Bracket

Status: landed in current workspace as a private refined-step span bracket.
FB81 holds `h_max=2.5e-10`, runs chained FB70 endpoints
`(5e-10,1e-9,1.5e-9,2e-9)`, and records the largest passing endpoint plus the
first refined-hmax failing endpoint.

Real current evidence:

- `artifact_payload_sha256=76bf833035a9a23f7b444786d19924d7d676d23d2f79c086703faeb0ae3f212e`
- manifest file SHA256 `243e1170947e2ce33271c0410169be55f73fa5383de5ac305bb9005e051ab2f9`
- `classification=refined_span_pass_fail_bracketed`
- `h_max=2.5e-10`
- `largest_passing_N_span_end=1e-09`
- `first_failing_N_span_end=1.5e-09`
- `first_failing_T_final_MeV=0.7999999988214215`
- `rows_passed=2`
- `rows_failed=2`
- `physical_full_bbn_span_ready=false`

Exit gates:

- The refined h_max and span ladder are recorded in the artifact inputs.
- Nested FB70 public/production/QKE boundaries are checked fail-closed.
- The first failing endpoint must carry concrete first-failure evidence.
- No public dispatch, production SMC validation, QKE, or publication-ready
  all-freedom full-BBN claim is made.

### FB82: Continuous AP65 First-Failure Triage

Status: landed in current workspace as a private strict-`Y_p` failure triage.
FB82 reruns the FB81 refined-span bracket, extracts the first failing row, and
records strict `Y_p` positivity, abundance-bound tolerance, BBN observables,
restart-handoff state, and source-evaluation counts as separate fields.

Real current evidence:

- `artifact_payload_sha256=c64bf7175a6935b39859ae521a05fadd6548fcfa5e2326d3faff9da1e9f9a783`
- manifest file SHA256 `6eed5354131696519b92f3e7ba4c2132cf5f37f9b88e4911ebcabb7012649b0b`
- `classification=strict_y_p_sign_failure_within_abundance_tolerance`
- `h_max=2.5e-10`
- `largest_passing_N_span_end=1e-09`
- `first_failing_N_span_end=1.5e-09`
- `first_failing_T_final_MeV=0.7999999988214215`
- `Yp=-1.2294890184644955e-30`
- `abs_Yp=1.2294890184644955e-30`
- `abundance_bound_tolerance=1e-18`
- `abundance_bounds_ok=true`
- `bound_tolerance_masks_strict_sign=true`
- `DH=2.5844839174694797e-13`
- `Xn=0.1300000000856175`
- `Xp=0.869999999913933`
- `N_eff_3T=11.084874967851695`
- `Sigma_H=0.015620499328388281`
- `physical_full_bbn_span_ready=false`

Exit gates:

- Nested FB81 public/production/QKE boundaries are checked fail-closed.
- The first failure must expose BBN observables including `Yp` and abundance
  tolerance.
- Strict `Y_p > 0` remains a blocker even when abundance tolerance accepts the
  tiny negative value.
- No public dispatch, production SMC validation, QKE, abundance repair, or
  publication-ready all-freedom full-BBN claim is made.

### FB83: Continuous AP65 Yp Source Probe

Status: landed in current workspace as a private source-localization probe.
FB83 consumes the FB82 first-failure triage, reads `X_phase2[5]` from the
packed FB69 `last_attempted_state_vector[-9:]` tail, and compares that
last-attempted `He4` value against the terminal BBN `Yp` readout.

Real current evidence:

- `artifact_payload_sha256=bf8c39c5947c063cb800c2f2b34f75bb3a70ac311aa3a388d6578a07a6692bb1`
- manifest file SHA256 `234d89567922ddad92d0c3750c35522f0c205616bfb3dbb3f47f8885e936d3d5`
- `classification=terminal_y_p_sign_crossing_below_tolerance_after_positive_last_stage_he4`
- `first_failing_N_span_end=1.5e-09`
- `first_failing_terminal_Yp=-1.2294890184644955e-30`
- `last_passing_terminal_Yp=8.116150311829752e-31`
- `terminal_Yp_delta=-2.0411040496474707e-30`
- `abundance_bound_tolerance=1e-18`
- `first_failing_last_attempted_He4=2.2765668298302704e-32`
- `last_passing_last_attempted_He4=1.2963142013342297e-30`
- `last_attempted_He4_delta=-1.273548533035927e-30`
- `terminal_sign_transition=positive_to_nonpositive`
- `x_phase2_tail_start=41`
- `he4_tail_index=46`
- `physical_scale_assessment=sub_tolerance_terminal_sign_crossing`
- `physical_full_bbn_span_ready=false`

Exit gates:

- Nested FB82 public/production/QKE boundaries are checked fail-closed.
- The probe uses the live-source replay `X_phase2_shape=(9,)` contract, not
  the sparse observable-index mapping, to locate `He4`.
- Missing first-failure rows, missing last-passing rows, and missing state
  vectors fail closed.
- No public dispatch, production SMC validation, QKE, abundance repair, or
  publication-ready all-freedom full-BBN claim is made.

### FB84: Continuous AP65 Terminal Final-State Probe

Status: landed in current workspace as FB70 row provenance enrichment.  FB84
records the terminal FB69 `final_state_vector[-9:]` `X_phase2` tail in FB70
span rows and compares `X_phase2[5]` against the terminal BBN `Yp` observable.

Real current evidence:

- nested FB82 `artifact_payload_sha256=47efcd214cc16b0810797d19d59baca5ab0a1e965ab169416ac2cdb3fe486609`
- manifest file SHA256 `81cfb5fc61419c14306f703326d333135cc34d0a4d172bc545cb27195d065acb`
- `classification=strict_y_p_sign_failure_within_abundance_tolerance`
- `first_failing_N_span_end=1.5e-09`
- `terminal_final_state_probe.available=true`
- `terminal_final_state_probe.x_phase2_tail_start=41`
- `terminal_final_state_probe.he4_tail_index=46`
- `terminal_final_state_probe.he4_from_final_state_vector=-1.2294890184644955e-30`
- `terminal_final_state_probe.terminal_observable_Yp=-1.2294890184644955e-30`
- `terminal_final_state_probe.terminal_y_p_minus_final_state_he4=0.0`
- `terminal_final_state_probe.terminal_y_p_matches_final_state_tail=true`

Exit gates:

- FB70 rows preserve terminal final-state tail metadata without changing state
  evolution, positivity policy, or public dispatch.
- Missing `final_state_vector` yields an unavailable probe instead of a crash.
- No public dispatch, production SMC validation, QKE, abundance repair, or
  publication-ready all-freedom full-BBN claim is made.

### FB85: Continuous AP65 Adaptive Step Acceptance

Status: landed in current workspace as private FB69/FB70 solver-control
hardening.  FB85 makes the host-stepped continuous AP65 Rodas5P prototype
reject `err_norm > 1` attempts, shrink `h`, and retry without advancing `N`
or `y`.  It also preserves accept/reject telemetry through FB70 span rows.

Real current evidence:

- nested FB82 `artifact_payload_sha256=9a0e0fe58cf8e318777b6b2a3cadae4cc367dd3424b6df75da178ba4a41b04dd`
- manifest file SHA256 `9a5d0cd620a4036ed1dc65c20842f6efc068d6de18fcd4091baffb9fad4ebee5`
- `classification=strict_y_p_sign_failure_within_abundance_tolerance`
- `first_failing_N_span_end=1.5e-09`
- `first_failure_row.step_count=2`
- `first_failure_row.attempt_count=2`
- `first_failure_row.n_rejected=0`
- `first_failure_row.error_norm_max=3.021584391530104e-14`
- `first_failure_row.rejected_error_norm_max=0.0`
- `first_failure_row.terminal_final_state_probe.he4_from_final_state_vector=-1.2294890184644955e-30`
- `first_failure_row.terminal_final_state_probe.terminal_y_p_minus_final_state_he4=0.0`

Interpretation:

- The current tiny finite-difference ladder's first strict-`Y_p` failure is not
  explained by accepting a step with `err_norm > 1`.
- The blocker remains the sub-tolerance final-state `He4` sign crossing, but
  future span expansion now has row-level accept/reject telemetry.

Exit gates:

- `err_norm > 1` attempts do not advance state.
- Accepted and rejected step telemetry is preserved through FB70 rows.
- No public dispatch, production SMC validation, QKE, abundance repair, or
  publication-ready all-freedom full-BBN claim is made.

### FB86: Continuous AP65 He4 RHS Boundary Probe

Status: landed in current workspace as private phase-2 network RHS boundary
evidence.  FB86 consumes the FB82/FB85 first-failure artifact and evaluates
the JAX phase-2 network RHS at raw terminal/last-attempted `X_phase2` tails,
plus diagnostic-only `He4=0`, `He4=1e-30`, and nonnegative trace-species
counterfactuals.  The probe fails closed if the terminal observable `Y_p` does
not match the final-state `He4` tail, and the nonnegative-trace counterfactual
floors only trace-species indices rather than repairing core abundances.

Real current evidence:

- `artifact_payload_sha256=ebd16b1fa3b6d4b673e33c2cff07855a075d3ca7f288230ccf2b9fe24b275fdf`
- manifest file SHA256 `b170d117620b40dcf413e94b865774b3a6f0c4dc12b2e52d8568130cf3baf201`
- `classification=he4_boundary_negative_due_to_negative_trace_intermediates`
- `first_failure_Yp=-1.2294890184644955e-30`
- `first_failure_N_span=[1e-09,1.5e-09]`
- `first_failure_T_final_MeV=0.7999999988214215`
- `first_failure_negative_trace_indices=[3,4,6,7]`
- `first_failure_negative_core_non_he4_indices=[]`
- `first_failure_terminal_dHe4_network_rhs=-2.618301171321943e-21`
- `first_failure_he4_zero_dHe4_network_rhs=-2.618301171321943e-21`
- `first_failure_nonnegative_trace_he4_zero_dHe4_network_rhs=7.273403769914826e-286`

Interpretation:

- The first strict-`Y_p` failure is fed by negative trace intermediates in the
  raw phase-2 abundance vector.
- The next implementation target is positivity-preserving phase-2 network
  evolution for trace species, not output truncation.

Exit gates:

- Raw terminal and last-attempted `X_phase2` tails are preserved.
- Diagnostic nonnegative-trace counterfactuals are labeled as diagnostics and
  are not used to repair evolution.
- No public dispatch, production SMC validation, QKE, abundance repair, or
  publication-ready all-freedom full-BBN claim is made.

### FB87: Continuous AP65 Trace-Boundary Positivity Gate

Status: landed in current workspace as private evolution-policy evidence.
FB87 keeps the default raw phase-2 network RHS available and adds an opt-in
`abundance_positivity_policy=trace_boundary` for private continuous-AP65 RHS
evolution.  The policy constrains trace/`He4` activities and active lower-bound
derivatives inside the RHS.  Terminal `Y_p` is still a raw solver-state readout,
not a truncated observable.  The gate records raw-vs-policy phase-2
mass-fraction sum residuals and fails closed if the trace-boundary residual
exceeds the configured conservation limit.

Real current evidence:

- `artifact_payload_sha256=dcdae7615088893f2bfbbece52620b8d81e60b1e775cb7ca8059c9d65a755276`
- manifest file SHA256 `99333c8747f006758fe9da2f0a2c8e633584a3e44a9d111999f8880d149f759d`
- `classification=trace_boundary_resolves_smoke_y_p_sign_failure_with_conservation_gate`
- raw first failure `N_span=[0.0,1.5e-09]`
- raw first-failure `T_final_MeV=0.7999999988214215`
- raw first-failure `Yp=-1.2294890184644993e-30`
- raw `Yp` failure rows: `2`
- trace-boundary `Yp` failure rows: `0`
- raw conservation max: `6.284872348663924e-18`
- trace-boundary conservation max: `8.110492019931864e-18`
- trace-boundary conservation limit: `1e-16`
- raw largest passing endpoint: `1e-09`
- trace-boundary largest passing endpoint: `2e-09`

Interpretation:

- The FB86 negative-trace diagnosis is now connected to an evolution-level
  private policy, not an output repair.
- The current smoke-scale `Y_p` sign failure is resolved through
  `N_span_end=2e-9` with smoke-scale conservation residuals gated below
  `1e-16`.
- The next blocker is span expansion toward `T_gamma <= 0.01 MeV` while
  conservation/stiffness/effort diagnostics remain controlled.

Exit gates:

- Raw-vs-trace-boundary comparison is recorded over the same smoke ladder.
- Trace-boundary mass-fraction conservation residuals are recorded and gated.
- Default raw network RHS and public dispatch are not rerouted.
- No public dispatch, production SMC validation, QKE, terminal abundance repair,
  or publication-ready all-freedom full-BBN claim is made.

### FB88: Trace-Boundary Span Extension

Status: landed in current workspace as private hot-endpoint span-extension
evidence.  FB88 runs FB70 with `abundance_positivity_policy=trace_boundary`,
chained restart handoff, and the same no-public/no-production/no-QKE boundary.

Real current evidence:

- `artifact_payload_sha256=49b2e9e858ffb87fece72c0ea2a031ed174eae9a0934db2e086c74c9997ba251`
- manifest file SHA256 `01be70a682384b58296b85a712ba4f05b9898fa95810804b8ba17ec8ca8507fb`
- `classification=trace_boundary_extension_all_requested_spans_passed`
- largest passing endpoint: `5e-09`
- rows passed/failed: `3` / `0`
- best `T_final_MeV=0.7999999960714048`
- conservation max: `8.746901892447222e-18`
- conservation limit: `1e-16`
- complete conservation/solver/stiffness rows: `3` / `3` / `3`
- step/attempt totals: `20` / `20`
- rejected steps: `0`
- `error_norm_max=0.0006360385926131681`
- source/stage source evaluations: `1166` / `140`

Interpretation:

- The trace-boundary policy remains sign-stable through `N_span_end=5e-9`.
- The run is still at the hot `~0.8 MeV` scale, so it is not a full-BBN result.
- The next blocker is a larger multiplicative span ladder that drives
  `T_gamma_MeV` down measurably while watching conservation, stiffness, source
  evaluation cost, and first-failure physics if a bracket appears.

### FB89: Trace-Boundary Span Growth Scout

Status: landed in current workspace as private hot-endpoint geometric
span-growth evidence.  FB89 runs FB88 over a multiplicative ladder from the
FB88 baseline, re-checks nested no-public/no-production/no-QKE/no-full-BBN-claim
boundaries, and preserves row-complete conservation/solver/stiffness telemetry.
In endpoint-target mode, the same surface can extend the geometric row budget
with `target_max_span_rows` and lets nested FB70 stop cleanly at
`stop_at_T_gamma_MeV`, without treating failed or negative-observable rows as a
successful endpoint.

Real current evidence:

- `artifact_payload_sha256=77a9d8a0dab4ef5b140622fb26e87860877059eab9ea3acb25e0ef068b1ab057`
- manifest file SHA256 `bc352e714afee26c01bfbc71298719dfd2fe91c063a27c11b03ce2427e53f9b2`
- `classification=trace_span_growth_all_requested_spans_passed`
- nested FB88 classification: `trace_boundary_extension_all_requested_spans_passed`
- largest passing endpoint: `4e-08`
- requested span rows: `3`
- best `T_final_MeV=0.7999999685712307`
- conservation max: `7.782547616453054e-18`
- conservation limit: `1e-16`
- complete conservation/solver/stiffness rows: `3` / `3` / `3`
- step/attempt totals: `40` / `40`
- rejected steps: `0`
- `error_norm_max=0.0006361033936367059`
- source/stage source evaluations: `2326` / `280`

Interpretation:

- The trace-boundary policy remains sign-stable through `N_span_end=4e-8`.
- The run is still at the hot `~0.8 MeV` scale, so it is not a full-BBN result.
- The next blocker is span growth large enough to move `T_gamma_MeV` visibly,
  with runtime budgeting because finite-difference source evaluations already
  exceed two thousand calls for this tiny scout.

### BD60: Nonlinear Logit Transport In Live Rodas RHS

Status: landed in current workspace as a runtime-physics upgrade on the
private CPU-JAX/Rodas5P live-source RHS.  The live-source sidecar no longer
uses a source-only non-LRS stress projection for the hierarchy RHS: it now
reconstructs the current S2 distribution, evaluates the AP62 nonlinear
collisionless transport derivatives in `q`, `mu`, and periodic `phi`, converts
the nodal `df/dN` through the augmented-logit chain rule, and projects the
result into the staged A-mode basis before adding any frozen collision
`dA_modes` payload.

Focused verification:

- `tests/test_augmented_nonlrs_nonlinear_transport.py::test_nonlrs_nonlinear_transport_projects_logit_rhs`
  locks the Python AP62 operator against direct logit-chain projection.
- `tests/test_jax_augmented_typeI_replay.py::test_live_source_rhs_uses_nonlinear_logit_transport_operator`
  locks the CPU-JAX live-source RHS against the Python nonlinear coevolution RHS
  and rejects the old source-only projection.
- The focused nonlinear/AP64/AP65/JAX replay bundle passed `66 passed`, and the
  registry/WBS bundle passed `33 passed, 3 skipped`.

Interpretation:

- This retires the source-only transport shortcut inside the repeated-run
  CPU-JAX/Rodas5P live-source path without adding a new gate.
- The work remains private/staged/no-QKE and does not claim public production
  support or full-BBN readiness.
- Raw states and failed-state observables remain untruncated; negative
  abundances or `Y_p` are not hidden by output clamping.
- Remaining blockers are q/angular/tolerance convergence at stronger grids,
  hot-loop AP65/Jacobian/Rodas cost, and endpoint-backed statistics/plot
  evidence.

### BD61: Precompute Nonlinear Transport Drift Factors

Status: landed in current workspace as a hot-loop reduction on the AP62/BD60
nonlinear transport path.  `build_nonlrs_nonlinear_transport_grid(...)` now
precomputes the angular energy-shift and drift basis factors, and the
CPU-JAX/Rodas5P live-source grid additionally precomputes the q-weighted
energy-shift factors.  The Python AP62 operator and JAX live-source RHS compose
those arrays linearly with current `Sigma_+`/`Sigma_-` instead of rebuilding the
same trigonometric and q-by-angle products on every RHS call.

Focused verification:

- RED checks first failed on missing precomputed factor fields in the Python and
  JAX live-source grids.
- The affected Python/JAX/AP65 RHS bundle passed `99 passed`.
- `scripts/sync_test_counts.py` refreshed the generated test-count blocks to
  `4189 total`.

Interpretation:

- This is a runtime-path optimization, not a new diagnostic/readiness gate.
- The nonlinear transport equations, raw state outputs, no-QKE boundary, and
  non-public claim boundary are unchanged.
- Remaining blockers are still larger-span endpoint growth, stronger
  q/angular/tolerance convergence, and AP65 Jacobian/Rodas linear-solve cost.

### BD62: Cache AP62 q-Differentiation Matrices

Status: landed in current workspace as another nonlinear-transport RHS
hot-loop reduction.  The AP62/BD60 non-LRS nonlinear transport path now keeps a
bounded in-module cache for q-differentiation matrices keyed by the exact q-grid
bytes.  Python AP62/AP63 RHS calls reuse that matrix across repeated RHS
evaluations on the same grid, and the CPU-JAX/Rodas5P live-source grid builder
uses the same cache helper before converting the matrix to a JAX array.

Focused verification:

- RED checks first showed repeated Python `build_q_diff_matrix(q)` calls on an
  unchanged q-grid and no shared-cache use in the JAX live-source grid builder.
- The affected nonlinear/JAX/AP65 RHS bundle passed `101 passed`.
- `scripts/sync_test_counts.py` refreshed generated test-count blocks to
  `4191 total`.

Interpretation:

- This removes repeated setup cost from existing private runtime paths and does
  not add a standalone gate.
- The cache is exact-grid keyed and bounded; it does not change q-grid physics,
  raw outputs, solver tolerances, no-QKE scope, or public dispatch status.
- Remaining blockers are still larger-span endpoint growth, stronger
  q/angular/tolerance convergence, and AP65 Jacobian/Rodas linear-solve cost.

### BD63: Cache Nonlinear S2 Transport Geometry

Status: landed in current workspace as setup-cost reduction for the same
AP62/BD60 nonlinear transport path.  `build_nonlrs_nonlinear_transport_grid(...)`
now reuses bounded `(N_mu, N_phi)` grid objects containing the S2 quadrature,
angular derivative matrices, projection matrices, and BD61 drift factors.  AP65
wrappers and the CPU-JAX/Rodas5P live-source grid builder therefore avoid
reconstructing identical angular geometry for repeated smoke/window runs.

Focused verification:

- The RED check first showed two same-resolution builds producing distinct
  geometry objects and calling `build_non_lrs_s2_grid(...)` twice.
- The affected nonlinear/JAX/AP65 RHS bundle passed `102 passed`.
- `scripts/sync_test_counts.py` refreshed generated test-count blocks to
  `4192 total`.

Interpretation:

- This is bounded runtime setup reuse on an existing private path, not a new
  diagnostic/readiness gate.
- It does not change the nonlinear transport equations, raw output handling,
  no-QKE scope, public dispatch status, or solver tolerances.
- Remaining blockers are still larger-span endpoint growth, stronger
  q/angular/tolerance convergence, and AP65 Jacobian/Rodas linear-solve cost.

### BD64: Reuse Live-Source JAX Grid Across Chain Windows

Status: landed in current workspace as a repeated-run CPU-JAX/Rodas5P setup
reduction.  `run_augmented_nonlrs_rodas5p_live_source_rhs_chain(...)` now owns
one validated `AugmentedNonLRSSourceGridJax` for the chain and passes it into
each restarted live-source RHS window.  The per-window replay still accepts
standalone calls, but supplied grids are checked against the q-grid and
`N_mu`/`N_phi` before use.

Focused verification:

- The RED check first showed a multi-window chain building no shared grid and
  passing no grid into the window replay boundary.
- The targeted chain grid-reuse regression passed, and the real restarted
  window plus dynamic-payload chain checks passed.

Interpretation:

- This removes repeated JAX q/S2 grid construction from the existing private
  live-source chain path; it does not add a diagnostic/readiness gate.
- It does not change transport equations, collision payload semantics, raw
  state handling, no-QKE scope, public dispatch status, or solver tolerances.
- Remaining blockers are endpoint growth, stronger q/angular/tolerance
  convergence, and AP65 Jacobian/Rodas linear-solve cost.

### BD65: Share AP65 Runtime Caches Across FB69/FB70 Windows

Status: landed in current workspace as a continuous-AP65 runtime-cache
reduction.  FB69 now takes shared in-memory AP65 radial-grid and source-factory
caches from the existing `runtime_cache` channel, and FB70 supplies one internal
child runtime cache across nested FB69 span/window calls even when the caller
does not configure a JAX persistent compilation cache.

Focused verification:

- The RED checks first showed the second FB69 artifact starting with cold
  source/radial caches and FB70 passing `None` as the nested runtime cache by
  default.
- The targeted FB69 cache-reuse and FB70 nested-runtime-cache regressions
  passed after the change.

Interpretation:

- This folds repeated AP65 setup work across chained private span windows; it
  does not add a diagnostic/readiness gate.
- It does not change collision physics, transport equations, raw state
  reporting, no-QKE scope, public dispatch status, or solver tolerances.
- Remaining blockers are endpoint growth, stronger q/angular/tolerance
  convergence, and AP65 Jacobian/Rodas linear-solve cost.

### BD66: Cache FB69 Live-Source JAX Grid In Runtime Cache

Status: landed in current workspace as a repeated-FB69 setup-cost reduction.
FB69 now reuses the same `AugmentedNonLRSSourceGridJax` from the existing
`runtime_cache` channel for identical q-grid, q-weight, `N_mu`, and `N_phi`
inputs.  This composes with BD65 so FB70 nested windows share both AP65
dynamic-payload caches and the fixed live-source JAX grid.

Focused verification:

- The RED check first showed two same-input FB69 artifacts calling
  `_build_live_source_grid_jax(...)` twice even with a shared runtime cache.
- The targeted grid-reuse regression passed after keying the grid cache by
  q-grid, q weights, and angular resolution.

Interpretation:

- This removes repeated fixed JAX source-grid construction across private
  FB69/FB70 runs; it does not add a diagnostic/readiness gate.
- It does not change transport equations, collision payload semantics, raw
  state reporting, no-QKE scope, public dispatch status, or solver tolerances.
- Remaining blockers are endpoint growth, stronger q/angular/tolerance
  convergence, and AP65 Jacobian/Rodas linear-solve cost.

### BD67: Refresh Private Endpoint Probes After Cache Reuse

Status: completed in current workspace as runtime evidence on the existing
FB69/FB70 private surface after BD64-BD66 cache reuse.  No new gate or public
dispatch was added.

Observed CPU-JAX/Rodas5P probes with `chain_restart_handoff`,
`chain_h_max_policy=first_rejection_half_ceiling_once`,
`jacobian_policy=frozen_source_jax`, `rhs_trace_policy=boundary`,
`abundance_positivity_policy=trace_boundary`, `h_max=0.2`, and
`N_span_end_ladder=(0.8,1.6,2.4,3.2,4.0,4.8)`:

- Non-LRS no-collision reached `T_gamma=0.00913919409378424 MeV`, with
  `physical_full_bbn_span_ready=true`, `rows_reaching_endpoint=1`, no
  violations, and `selected_wall_seconds_total=10.089954509865493`.
- Non-LRS collision with `stage_collision_payload_policy=step_base_reuse`
  reached `T_gamma=0.009144759673024236 MeV`, with
  `physical_full_bbn_span_ready=true`, `rows_reaching_endpoint=1`, no
  violations, `selected_dynamic_collision_payload_builds_total=778`, and
  `selected_wall_seconds_total=23.59657490684185`.
- All-three weak+non-LRS+collision with `weak_correction_level=3` reached
  `T_gamma=0.009144759672723686 MeV`, with
  `physical_full_bbn_span_ready=true`, `rows_reaching_endpoint=1`, no
  violations, `selected_dynamic_collision_payload_builds_total=502`, and
  `selected_wall_seconds_total=19.1935998670524`.

Interpretation:

- The immediate private endpoint blocker is retired for smoke-scale
  continuous-AP65 all-three runs under the bounded `step_base_reuse`
  performance approximation.
- This does not prove q/angular/tolerance convergence, production SMC
  validation, public dispatch, QKE support, or publication-ready support.
- Remaining blockers move to endpoint-backed resolution/tolerance ladders,
  weak-rate bridge/profile extension, figure regeneration from current
  endpoint artifacts, and guarded statistical-pipeline input refresh.

### BD68: Optimize Exact Current-State AP65 Stage Payload Hot Path

Status: completed in current workspace as runtime physics/performance work on
existing FB68/FB69/FB70 surfaces.  No new readiness, manifest, hash, or claim
gate was added.

Implemented changes:

- The public dynamic payload function keeps JSON-safe output by default, but
  FB69 suppressed non-LRS stage payloads now use a packed current-state NumPy
  restart record and raw ndarray `dA_modes` when `rhs_trace_policy=boundary`
  suppresses stage payload rows.  Boundary/full-trace roles still request
  JSON-safe payloads.
- `build_pstf_radial_channel_kernel_grid(...)` now vectorizes the exact static
  momentum-delta AP6 radial channel-grid assembly across all radial tuples.
  Callable p-dependent momentum-delta providers keep the tuple-dependent slow
  path.  A focused parity test compares the new static vectorized path against
  the callable path at `rtol=1e-13`.

Observed CPU evidence:

- Existing FB68 hotpath profile, standard-3T radial normalization:
  shared-cache cold-miss median improved from `0.05483742000069469 s` to
  `0.030677367001771927 s` (`1.787x`, about `44%` lower).  Warm-hit median
  stayed effectively unchanged (`0.00871060648933053 s` to
  `0.008741265046410263 s`), as expected because the optimized path targets
  radial-grid construction on misses.
- Existing FB70 all-three `current_state` two-window probe
  (`N_span_end_ladder=(0.8,1.6)`, `h_max=0.2`, `frozen_source_jax`,
  `trace_boundary`) remained passing and improved selected wall time from
  `19.31860326998867 s` to `16.103335389052518 s` (`1.20x`, about `16.6%`
  lower).  Total wall time improved from `64.39050275902264 s` to
  `61.14263989403844 s`; the total is still dominated by failed refinement
  attempts and exact stage payload builds.

Interpretation:

- This reduces gate sprawl by using existing profiler/span surfaces and moving
  actual collision-source runtime cost.
- Exact `current_state` stage payloads are faster but still not endpoint-ready
  for the full `0.01 MeV` all-three ladder; the smoke-scale endpoint evidence
  remains on the bounded `step_base_reuse` performance approximation.
- Remaining blockers are exact-current-state continuous AP65 endpoint below
  `0.01 MeV`, endpoint-backed q/angular/tolerance ladders, weak-rate bridge
  extension, and figure/statistical refresh from endpoint-backed artifacts.

### BD69: Feed Recovered h-Refinement Into Exact AP65 Chain Policy

Status: completed in current workspace as solver-policy/runtime work on the
existing FB70 span ladder.  No new diagnostic or readiness gate was added.

Implemented changes:

- FB70 now accepts `chain_h_max_policy=first_rejection_or_recovered_h_ceiling`.
  It preserves the existing first-rejection half-ceiling behavior, but when a
  chained window fails at the coarse `h_max` and passes only after
  `h_refinement_factors`, the selected recovered `h_max` becomes the next
  chained-window ceiling.
- The policy is private to the existing FB70 surface and records the source as
  `previous_window_recovered_selected_h_max` in `restart_handoff`.

Observed CPU evidence:

- Exact all-three `stage_collision_payload_policy=current_state` with
  `chain_restart_handoff`, `frozen_source_jax`, `rhs_trace_policy=boundary`,
  `h_max=0.2`, `h_refinement_factors=(1.0,0.5)`, and
  `N_span_end_ladder=(0.8,1.6,2.4,3.2,4.0,4.8,5.6)` reached the full-BBN
  endpoint and stopped at `N_span_end=4.8`.
- The run passed with `physical_full_bbn_span_ready=true`,
  `terminal_completion_class=full_bbn_completed`,
  `T_gamma=0.009144759663514114 MeV`, raw `Y_p=0.16318746257583355`,
  raw `D/H=2.0957653972447094e-05`, `rows_reaching_endpoint=1`,
  `failed_rows=0`, and `physical_observable_failure_rows=0`.
- Selected exact-current-state work was `selected_wall_seconds_total=
  65.46224005485419`, `selected_source_evaluations_total=4017`,
  `selected_current_restart_payload_builds_total=4017`, and
  `selected_stage_collision_payload_current_state_builds_total=3521`.

Interpretation:

- The exact-current-state all-three private FB70 endpoint blocker is now moved
  below `0.01 MeV` on the CPU-JAX/Rodas5P target without using the
  `step_base_reuse` approximation.
- This is still private no-QKE staging evidence, not public dispatch,
  production SMC validation, or publication-ready support.
- Remaining blockers move to endpoint-backed q/angular/tolerance ladders,
  weak-rate/profile extension on the exact endpoint artifact, figure refresh,
  and guarded statistical-pipeline input refresh.

### BD70: Carry Exact AP65 Weak-Pair Provenance Through FB71/FB72

Status: completed in current workspace as a correctness and provenance fix on
the existing FB70/FB71/FB72 path.  No new diagnostic, readiness, manifest,
hash, or figure gate was added.

Implemented changes:

- FB70 freedom-composition rows now thread the requested
  `stop_at_T_gamma_MeV` into each nested continuous-span child run.  This fixes
  the exact-current-state weak/control composition case that had reached the
  `0.01 MeV` endpoint but then continued to the next requested span and failed
  on a later nonfinite row.
- FB70 composition rows now preserve the continuous-AP65 solver controls that
  make a weak/control pair comparable: span ladder, q/angular grids, tolerance,
  Jacobian policy, RHS trace policy, abundance positivity policy,
  stage-collision payload policy, chain h-max policy, refinement factors, and
  endpoint stop target.
- FB71 now rejects FB70-derived weak/control pairs when those solver controls
  differ, and FB72 carries the FB71 match result plus exact policy labels into
  its bridge summary.

Observed CPU evidence:

- Exact all-three weak/control FB70 freedom-composition rerun with
  `stage_collision_payload_policy=current_state`,
  `chain_h_max_policy=first_rejection_or_recovered_h_ceiling`,
  `h_refinement_factors=(1.0,0.5)`, `frozen_source_jax`,
  `rhs_trace_policy=boundary`, and `stop_at_T_gamma_MeV=0.01` passed with both
  rows stopping at the endpoint.
- The non-LRS+collision control row reached
  `T_gamma=0.009144759663514114 MeV`, raw
  `Y_p=0.16318746257583355`, raw
  `D/H=2.0957653972447094e-05`, and
  `selected_source_evaluations_total=4017`.
- The weak-level-3 non-LRS+collision row reached
  `T_gamma=0.009144759663283382 MeV`, raw
  `Y_p=0.16487647774832928`, raw
  `D/H=2.1030704630003705e-05`, and
  `selected_source_evaluations_total=3988`.
- FB71 over that exact FB70 artifact reported
  `full_bbn_weak_rate_pairs_ready=true`,
  `fb70_continuous_ap65_pair_solver_policy_matched=true`,
  `fb70_stage_collision_payload_policies=["current_state"]`,
  `fb70_chain_h_max_policies=["first_rejection_or_recovered_h_ceiling"]`,
  `max_abs_weak_delta_Yp=0.001689015172495728`, and
  `max_abs_weak_delta_DH=7.305065755661101e-08`.
- FB72 using the same exact FB70 pair plus the existing AP80 smoke artifact
  reported `ap80_fb71_bridge_ready=true` and propagated the same matched exact
  FB70 policy labels while keeping
  `fb71_required_context_scope=scoped_subset`.

Interpretation:

- This folds the exact endpoint weak/control path back through the existing
  FB70/FB71/FB72 surfaces instead of adding a new gate.
- The exact-current-state weak-rate delta is now endpoint-backed for the scoped
  non-LRS+collision pair, but AP80 remains profile-level evidence and the
  bridge remains private diagnostic evidence.
- Remaining blockers are endpoint-backed q/angular/tolerance/default-context
  expansion, figure refresh from exact endpoint artifacts, guarded
  statistical-pipeline input refresh, and publication-grade convergence.

### BD71: Push Exact Endpoint Evidence Into Figure And SMC Inputs

Status: completed in current workspace as stale-blocker removal on the existing
FB73/FB74/FB75 figure/statistical-input path.  No new standalone gate was
added.

Implemented changes:

- FB73, FB74, and FB75 now expose explicit scoped-weak-bridge override flags.
  The default still rejects scoped FB72/FB73 weak-rate evidence, but an
  explicitly scoped diagnostic run can be rendered, QA-packaged, and passed
  into FB75 while preserving the scoped claim boundary.
- FB75 no longer treats a consistent endpoint-ready FB70 span ladder as an
  unexpected condition.  Its FB70 consistency check now understands that
  multi-window span ladders can have early hot rows, so endpoint readiness is
  checked by completed endpoint rows and `T_final_MeV_min <= 0.01` rather than
  requiring every row's `T_final_MeV_max` to be below the endpoint.
- FB75 blocker selection is no longer hard-coded to
  `continuous_ap65_full_bbn_span_not_ready`.  It reports that blocker only when
  FB70 is not endpoint-ready, reports
  `weak_rate_bridge_default_context_matrix_not_ready` when scoped weak-rate
  evidence is explicitly allowed, and otherwise falls back to
  `diagnostic_scope_not_promoted`.

Observed CPU/file-backed evidence:

- FB73 rendered the BD69 exact-current-state endpoint FB70 artifact plus the
  BD70 exact scoped FB72 bridge with
  `continuous_ap65_physical_span_ready=true`,
  `continuous_ap65_rows_reaching_endpoint=1`,
  `weak_rate_bridge_passed_pair_count=1`,
  `weak_rate_bridge_required_context_scope=scoped_subset`, and
  `publication_readiness_blocker=diagnostic_scope_not_promoted`.
- FB74 QA-packaged that FB73 figure bundle with the scoped override and
  preserved `weak_rate_bridge_required_context_scope=scoped_subset`.
- FB75 consumed the same file-backed FB60, FB66, exact FB70, exact scoped FB72,
  and scoped FB74 inputs with `guarded_smc_pilot_input_ready=true`,
  `validated_full_bbn_product_inputs_ready=true`,
  `statistical_pilot_input_ready=true`, `violations=[]`,
  `fb70_physical_full_bbn_span_ready=true`,
  `fb70_rows_reaching_endpoint=1`, and
  `pilot_blockers=["weak_rate_bridge_default_context_matrix_not_ready"]`.

Interpretation:

- The old statistical-input blocker
  `continuous_ap65_full_bbn_span_not_ready` is retired for the exact
  current-state endpoint artifact.
- The remaining statistical/figure blocker is now correctly narrowed to
  default-context weak-rate bridge expansion plus q/angular/tolerance
  convergence.  Public dispatch, production SMC validation, QKE support, and
  publication-ready claims remain unclaimed.

### BD72: Complete Exact Default-Context Weak Bridge Evidence

Status: completed in current workspace as an execution/evidence refresh through
the existing FB70/FB72/FB73/FB74/FB75 path.  No code surface or standalone gate
was added.

Observed CPU/file-backed evidence:

- The exact `stage_collision_payload_policy=current_state` FB70
  freedom-composition default matrix was run with
  `chain_h_max_policy=first_rejection_or_recovered_h_ceiling`,
  `h_refinement_factors=(1.0,0.5)`, `frozen_source_jax`,
  `rhs_trace_policy=boundary`, and `stop_at_T_gamma_MeV=0.01`.
- FB70 passed all eight default context rows with
  `physical_full_bbn_span_ready=true`, `rows_full_bbn_completed=8`,
  `failed_or_exception_rows=0`, `T_final_MeV_min=0.00913919385022409`,
  `T_final_MeV_max=0.009144759665756305`, and
  `blocking_next_step=compare_pairwise_freedom_deltas_against_resolution_ladder`.
- FB72 over that exact default FB70 artifact and the existing AP80 smoke
  artifact passed with `ap80_fb71_bridge_ready=true`,
  `fb71_required_context_scope=default_all_contexts`,
  `fb71_passed_pair_count=4`, `fb71_rows_reaching_full_bbn_endpoint=8`,
  `fb71_fb70_continuous_ap65_pair_solver_policy_matched=true`,
  `fb71_fb70_stage_collision_payload_policies=["current_state"]`, and
  `fb71_fb70_chain_h_max_policies=["first_rejection_or_recovered_h_ceiling"]`.
- FB73/FB74/FB75 then ran without scoped overrides.  FB75 reported
  `guarded_smc_pilot_input_ready=true`,
  `validated_full_bbn_product_inputs_ready=true`,
  `statistical_pilot_input_ready=true`, `violations=[]`,
  `fb72_required_context_scope=default_all_contexts`,
  `fb72_passed_pair_count=4`, `fb72_rows_reaching_full_bbn_endpoint=8`, and
  `pilot_blockers=["diagnostic_scope_not_promoted"]`.

Interpretation:

- The BD71 `weak_rate_bridge_default_context_matrix_not_ready` blocker is now
  retired for the exact current-state endpoint path.
- The remaining blockers are not more evidence plumbing: they are
  endpoint-backed q/angular/tolerance convergence, AP80/profile extension
  beyond smoke, and eventually real production-statistical validation.
  Public dispatch, production SMC validation, QKE support, and publication-ready
  claims remain unclaimed.

### BD73: Block-JVP Rodas5P Policy For q4 Memory Triage

Status: landed in current workspace as a private CPU-JAX/Rodas5P solver-policy
optimization inside the existing FB69/FB70 continuous-AP65 surfaces.  No new
standalone gate was added, and no public dispatch or production support is
claimed.

Implemented changes:

- FB69 now exposes `jacobian_policy=frozen_source_jax_block_jvp`, a private
  Rodas5P W-method policy that builds exact frozen-source JVP columns for the
  geometry/thermo and phase-2 network blocks while omitting the high-resolution
  `A_modes_flat` columns from the implicit linear solve.
- Accepted stages and adaptive error estimates still evaluate the live
  current-state RHS.  This does not truncate negative abundances or alter raw
  terminal states; it changes only the linearization used inside the implicit
  solve.
- The JVP helper allocates one unit tangent at a time rather than a full dense
  identity, and FB70 carries block-JVP Jacobian telemetry through selected-row
  and h-refinement summaries.

Observed CPU-JAX/Rodas5P evidence:

- A q4 all-freedom exact-current-state probe with the existing full
  `frozen_source_jax` Jacobian exceeded practical memory before writing an
  artifact.
- The early full column-JVP triage variant was not kept in BD73.  BD77 later
  reintroduced a one-column full-state JVP policy after BD76 showed that
  omitting the A-mode columns caused the q4 step-size limiter.
- The retained `frozen_source_jax_block_jvp` policy wrote a raw failed artifact
  instead of disappearing into OOM: the first q4 all-freedom row stopped at the
  explicit `wall_time_budget_seconds` limit with `step_count=54`,
  `attempt_count=61`, `n_rejected=7`, and
  `frozen_source_jax_block_jvp_jacobian_evaluation_count=54`.

Interpretation:

- BD73 reduces the immediate q4 blocker from no-artifact memory failure to an
  explicit Rodas5P step-size/wall-budget failure while preserving raw output and
  claim boundaries.
- It does not establish q4/angular convergence or a publication-ready endpoint
  ladder.  Remaining blockers are the tiny accepted q4 step sizes under the
  block linearization, exact high-resolution Jacobian cost, AP80/profile
  extension beyond smoke, production-statistical validation, public dispatch,
  and QKE, which remains out of scope.

### BD74: Trace-Abundance Domain Guard In Rodas5P Acceptance

Status: landed in current workspace as a private solver/evolution hardening
inside the existing FB69/FB70 continuous-AP65 surfaces.  It adds no standalone
gate, keeps public dispatch closed, and does not claim production support.

Implemented changes:

- FB69 now rejects an unaccepted Rodas5P internal stage or step candidate if
  the phase-2 trace abundance block moves below the live-source safe tolerance
  (`1e-14`) on indices `[3,4,5,6,7,8]`.  The raw state is not truncated or
  repaired; the state is treated like an adaptive domain rejection and the
  step is retried at the existing domain-reject factor.
- The existing FB70 span surface now preserves the trace-abundance domain
  rejection count alongside the previous adaptive, stage-domain, and
  h-refinement telemetry, so the consolidated runner records whether a failure
  came from accepted-state physics/domain control or from hot-loop cost.
- The default raw RHS policy remains available.  The opt-in `trace_boundary`
  evolution policy remains an evolution-level trace-species boundary treatment,
  not an output-level abundance clamp.

Observed CPU-JAX/Rodas5P evidence:

- Focused regression tests cover the helper, the host Rodas5P stage path, and
  the actual `_run_step_cap_row` acceptance path: a candidate with negative
  trace abundance and small error norm is rejected, then retried.
- A q4/mu5 all-freedom raw local probe with
  `frozen_source_jax_block_jvp`, `current_state` stage payloads, and a 60 second
  wall budget still failed before endpoint by wall time (`step_count=9`,
  `attempt_count=22`, `n_rejected=13`), with
  `trace_abundance_domain_rejection_count_total=6`.
- The same q4/mu5 probe with `abundance_positivity_policy=trace_boundary`
  remained wall-budget limited but advanced farther under the same budget
  (`step_count=8`, `attempt_count=15`, `n_rejected=7`) while preserving the
  raw-state/no-output-truncation claim boundary.

Interpretation:

- BD74 closes the specific solver bug class where a negative trace-abundance
  internal stage can evaluate a source RHS, or a negative trace-abundance
  candidate can be accepted, only because the scalar error norm is small enough.
- The endpoint blocker is not retired.  The active blockers remain continuous
  AP65 endpoint progress below `0.01 MeV` and hot-loop payload/Jacobian cost,
  with q4 all-freedom current-state payload rebuilds still dominating wall time.

### BD75: Auto Small-Collision Stage Payload Reuse

Status: landed in current workspace as a private CPU-JAX/Rodas5P hot-loop
performance policy inside the existing FB69/FB70 continuous-AP65 surfaces.  It
adds no standalone gate, keeps public dispatch closed, and does not claim
production support.

Implemented changes:

- FB69 now accepts
  `stage_collision_payload_policy=auto_small_collision_reuse`.  The policy
  reuses the host-step base AP65 collision payload inside unaccepted Rodas5P
  stage RHS calls only when the base collision `dA_modes` amplitude is finite
  and below `1e-6`; otherwise it falls back to exact current-state stage
  payload rebuilds.
- FB69 records separate auto-reuse and auto-current-state counts.  FB70 carries
  those counters through selected-row, h-refinement, and summary telemetry, and
  the CLI exposes the same policy on the existing span-ladder runner.
- The existing `current_state` and `step_base_reuse` policies remain available.
  The auto policy is explicitly a private performance approximation when reuse
  is observed, not full current-state AP65 stage-source evidence.

Observed CPU-JAX/Rodas5P evidence:

- The BD74 q4/mu5 `current_state` trace-boundary probe with
  `frozen_source_jax_block_jvp` advanced only `step_count=8` within the 60
  second wall budget, with stage payloads rebuilt from the current state.
- The fixed `step_base_reuse` comparison advanced `step_count=83`,
  `attempt_count=90`, `n_rejected=7`, with
  `selected_stage_collision_payload_reuse_total=630` and
  `selected_stage_collision_payload_current_state_builds_total=0`.
- The new `auto_small_collision_reuse` q4/mu5 probe wrote
  `/tmp/rabbit_bd75_auto_small_collision_reuse_trace_boundary_q4_mu5_probe.json`
  with `artifact_payload_sha256=2a9ea8b9d192693c932b0e607ffa182b420f038d36d6e5efc5a340a2462dd6da`.
  It remained wall-budget limited but advanced to `step_count=85`,
  `attempt_count=92`, `n_rejected=7`,
  `selected_stage_collision_payload_auto_reuse_total=644`,
  `selected_stage_collision_payload_auto_current_state_total=0`, and
  `rhs_stress_collision_dA_abs_max_max=1.5255268474667314e-07`.

Interpretation:

- BD75 converts the q4 all-freedom collision-stage hot loop from repeated
  current-state payload rebuilds to a physics-budgeted reuse mode when the AP65
  collision correction is small compared with the transport/network RHS scale.
- This is a runtime blocker reduction only.  The full-BBN endpoint below
  `0.01 MeV` is still not reached, exact current-state stage payload evidence
  remains expensive, and q/angular convergence, production SMC validation,
  public dispatch, publication-ready support, and QKE remain unclaimed.

### BD76: Rodas Error-Block Attribution For q4 Step-Size Limiter

Status: landed in current workspace as solver telemetry on the existing FB69
and FB70 continuous-AP65 surfaces.  It adds no standalone gate and does not
change the accepted state, error norm, raw observables, public dispatch, or QKE
scope.

Implemented changes:

- FB69 can now return Rodas5P error-estimator diagnostics split by state block:
  `geometry_thermo`, `A_modes_flat`, and `X_phase2`.
- FB69 rows record capped samples of accepted/rejected error-block diagnostics
  plus dominant-block counts.  FB70 preserves the selected-row and
  h-refinement dominant-block counts in span-ladder summaries.
- This is attribution for the existing adaptive global error norm, not a
  species-specific or block-specific tolerance policy.

Observed CPU-JAX/Rodas5P evidence:

- The q4/mu5 all-freedom `auto_small_collision_reuse` probe wrote
  `/tmp/rabbit_bd76_error_block_q4_mu5_auto_probe.json` with
  `artifact_payload_sha256=990b640c86e920ad035e575837dcf6ad3be8ac997c5cd629b8aa57b6c7c9e1d5`.
  It remained wall-budget limited with `step_count=87`, `attempt_count=94`,
  `n_rejected=7`, and
  `selected_stage_collision_payload_auto_reuse_total=658`.
- The dominant Rodas error block was `A_modes_flat` for all 94 attempts:
  `error_norm_dominant_block_counts={"A_modes_flat":94}`.  Early rejected
  samples show the A-mode scaled-error norm dominating geometry/thermo and
  `X_phase2`, even though RHS stress still reports a large network
  `dX_dN_abs_max`.

Interpretation:

- BD76 rules out the immediate guess that the q4 tiny-step limiter is primarily
  the phase-2 network tolerance.  The next blocker-moving PR should target the
  kinetic hierarchy block: A-mode linearization, A-mode transport/collision RHS
  scaling, or a physically justified implicit/block treatment for those modes.
- It does not claim endpoint readiness, convergence, public dispatch,
  production SMC validation, publication-ready support, or QKE.

### BD77: Full-State One-Column JVP Rodas Policy For q4 A-Mode Linearization

Status: landed in current workspace as a private CPU-JAX/Rodas5P solver-policy
extension on the existing FB69/FB70 continuous-AP65 surfaces.  It adds no
standalone gate, keeps public dispatch closed, and does not claim production
support.

Implemented changes:

- FB69 now exposes `jacobian_policy=frozen_source_jax_full_jvp`, a private
  Rodas5P W-method policy that differentiates every packed state column at the
  frozen AP65 collision payload with one-column-at-a-time JVPs.
- Unlike BD73 `frozen_source_jax_block_jvp`, the full-JVP policy restores the
  `A_modes_flat` columns to the implicit linear solve while still avoiding a
  dense tangent identity allocation.  Accepted stages and adaptive error
  estimates continue to evaluate the live current-state RHS.
- FB70 and the FB69/FB70 CLIs carry the policy and its Jacobian-evaluation
  counters through selected-row, h-refinement, resolution, and summary
  telemetry.

Observed CPU-JAX/Rodas5P evidence:

- The BD76 q4/mu5 `auto_small_collision_reuse` probe with block-JVP remained
  wall-budget limited at `step_count=87`, `attempt_count=94`, and
  `error_norm_dominant_block_counts={"A_modes_flat":94}`.
- The BD77 q4/mu5 probe with `frozen_source_jax_full_jvp`,
  `rhs_trace_policy=boundary`, `trace_boundary` abundances, and
  `stage_collision_payload_policy=auto_small_collision_reuse` wrote
  `/tmp/rabbit_bd77_full_jvp_q4_mu5_auto_probe.json` with
  `artifact_payload_sha256=374e880b63c209aa1bc02b0f453014ac40c77d71b111fde044d82b2d164df711`.
  It passed the `N_span_end=0.8` row in `47.47854742605705 s` with
  `step_count=20`, `attempt_count=31`, `n_rejected=11`,
  `selected_frozen_source_jax_full_jvp_jacobian_evaluations_total=20`, and
  terminal `T_gamma=0.3708908727961177 MeV`.
- A longer `N_span_end=2.0` probe wrote
  `artifact_payload_sha256=a291288b9d17648628b30fbd85d63e0152670dba407e0cd6edd57459c5df714c`
  but failed closed after `46` accepted steps at the trace-abundance/domain
  boundary.  The raw failure payload preserved the last attempted state and
  stage exception, with first late failures around `T_gamma~0.247 MeV` and a
  final recorded stage failure near `T_gamma~0.177 MeV`; no final observable
  truncation or sign repair was applied.

Interpretation:

- BD77 retires the immediate q4 A-mode linearization blocker exposed by BD76:
  the same q4/mu5 all-freedom run moves from 80+ tiny accepted steps without
  completing the `0.8` span to 20 accepted steps and a completed hot-endpoint
  row.
- The remaining blocker has shifted to long-span trace-abundance/domain
  evolution stability below roughly `0.25--0.18 MeV`, plus endpoint completion
  below `0.01 MeV`, q/angular convergence, production SMC validation, public
  dispatch, publication-ready support, and QKE, which remains out of scope.

### BD78: Stage-Only Trace RHS Projection For Reused-Payload Rodas Stages

Status: landed in current workspace as a private CPU-JAX/Rodas5P solver-domain
policy inside the existing FB69/FB70 continuous-AP65 surfaces.  It adds no
standalone gate, keeps public dispatch closed, and does not claim production
support.

Implemented changes:

- FB69 now permits a bounded projection only for unaccepted Rodas internal-stage
  RHS inputs when `abundance_positivity_policy=trace_boundary` and the stage is
  using a reused/frozen AP65 source payload.  Current-state source refreshes,
  accepted states, candidate states, final states, and reported BBN observables
  remain raw and strict.
- The projection is limited to phase-2 trace species whose accepted step-base
  abundance is within the solver absolute trace layer.  The allowed stage
  undershoot is tied to solver `atol` and is recorded with role, index,
  reference/projection tolerances, raw minimum, and projected absolute maximum.
- FB70 carries the new selected-row, h-refinement, and summary telemetry:
  `stage_trace_abundance_projection_count`,
  `stage_trace_abundance_projection_event_count`, and
  `stage_trace_abundance_projected_abs_max`.

Observed CPU-JAX/Rodas5P evidence:

- The BD77 baseline-grid q4/mu5 all-freedom `N_span_end=2.0` probe wrote
  `artifact_payload_sha256=a291288b9d17648628b30fbd85d63e0152670dba407e0cd6edd57459c5df714c`
  and failed closed after `46` accepted steps with
  `trace_abundance_domain_rejection_count_total=9`.
- With BD78 on the same baseline q-grid/weights, `N_mu=5`, `N_phi=7`,
  `max_steps=256`, and `wall_time_budget_seconds=90`, the FB70 probe wrote
  `/tmp/rabbit_bd78_stage_trace_projection_q4_mu5_N2_baseline_grid_probe.json`
  with
  `artifact_payload_sha256=3218f17c7c0431722281f4a882dd8ad9fb415102fc329ca737162d7249dda060`.
  It passed the hot-endpoint `N_span_end=2.0` row with terminal
  `T_gamma=0.13208063389700056 MeV`, `step_count=39`,
  `attempt_count=60`, `n_rejected=21`,
  `stage_trace_abundance_projection_count_total=24`,
  `stage_trace_abundance_projection_event_count_total=19`, and
  `stage_trace_abundance_projected_abs_max=6.076688171653506e-09`.

Interpretation:

- BD78 moves the trace-domain stage-instability blocker that stopped the BD77
  baseline-grid `N_span_end=2.0` run.  It does so by changing only unaccepted
  stage RHS inputs under reused source payloads, not by truncating output
  abundances or terminal `Y_p`.
- The remaining blocker is still physical endpoint completion below `0.01 MeV`
  on the private continuous-AP65 path, plus q/angular convergence, exact
  current-state stage-source cost, production SMC validation, public dispatch,
  publication-ready support, and QKE, which remains out of scope.

### BD79: Direct-Logit Transport And Trace-Domain Recovery Triage

Status: landed in current workspace as a private CPU-JAX/Rodas5P solver and
transport hardening step on the existing AP62/FB69/FB70 continuous-AP65 path.
It adds no standalone gate, keeps QKE out of scope, keeps public dispatch
closed, and does not claim production support.

Implemented changes:

- The non-LRS AP62 transport RHS now evaluates the augmented-logit transport
  equation directly in A-mode coordinates on both the Python and JAX
  live-source paths.  This removes the old low-occupancy tail amplification
  caused by projecting nodal `df/dN` and dividing by `f(1-f)` in the Rodas
  RHS.
- FB69 now treats recoverable trace-domain failures as a consecutive rejection
  budget rather than a lifetime row budget, and candidate trace-domain
  rejections use a boundary-crossing-limited retry step before falling back to
  the generic domain shrink.
- The stage-only trace projection remains limited to unaccepted stage RHS
  inputs, but the depleted-trace allowance now has an explicit small
  diagnostic cap.  Accepted states, candidates, final states, and reported
  observables remain raw and strict.
- A trace-log abundance coordinate pilot exists only behind the explicit
  `trace_log_solver_coordinates_enabled=True` flag.  The first pilot targeted
  trace species `[T, He3, He4, Li7, Be7, Li6]`, converts RHS/Jacobian/error
  estimates between solver and physical abundance units, and is accompanied by
  `solver_audit_english.md`; BD81 narrows the active solver transform to the
  subset above the active abundance floor.  It is not enabled by default because
  endpoint probes showed overflow or early `h_min` failures before the
  network/controller work is complete.

Observed CPU-JAX/Rodas5P evidence:

- A late local direct-logit probe reduced the A-mode transport RHS stress from
  the previous tail-amplified `dA_abs_max~1.24e5` scale to
  `dA_abs_max=0.54545`, with the network block becoming the dominant late
  stress instead.
- The q4/mu5 baseline-grid all-freedom boundary-limited run wrote
  `artifact_payload_sha256=a893993230412112079a3c494d2661c45c66bf11c0d337404f41d7b40ffbdc1c`
  and failed closed by `max_steps` near
  `T_gamma~0.07990 MeV` with `368` accepted steps, `144` rejected steps, and
  `trace_abundance_domain_rejection_count_total=135`.  Raw rejected
  trace-domain evidence was preserved; no accepted abundance or `Y_p` was
  truncated.
- Trace-log pilot probes wrote failed diagnostic artifacts rather than changing
  default behavior.  Pure/all-trace variants overflowed or failed before
  progress; hybrid active-floor variants accepted only `17--18` steps before
  `h_min`.  This keeps the next implementation target on network positivity
  and flux/Jacobian structure, not on another wrapper.

Interpretation:

- BD79 removes a real direct-logit transport hot-path pathology and narrows the
  stronger-grid all-freedom blocker from mixed A-mode/network suspicion to
  phase-2 trace-network positivity/stiffness plus the remaining AP65/Jacobian
  hot-loop cost.
- The remaining direct implementation targets are a production/destruction
  split audit for Li/Be/Li6 trace reactions, a stable positivity-preserving
  network coordinate or constrained implicit network block, and endpoint-backed
  q/angular/tolerance convergence below `0.01 MeV`.
- Public dispatch, production SMC validation, publication-ready support, and
  QKE remain unclaimed.

### BD80: Phase-2 Trace Network Production/Destruction Split

Status: landed in current workspace as network/solver diagnostics on the
existing FB69 continuous-AP65 row surface.  It adds no standalone gate and does
not alter the RHS, accepted states, terminal abundances, public dispatch, or
QKE boundary.

Implemented changes:

- `abundances_standard.py` now exposes directional forward and reverse reaction
  flux components, and a trace-species production/destruction split for
  `7Li`, `7Be`, and `6Li`.
- The split treats each forward reaction with the normal stoichiometry and each
  reverse reaction with the opposite stoichiometry, so for nonnegative physical
  abundances the per-species production/destruction totals are nonnegative and
  reconstruct the nuclear part of `abundance_rhs_phase2`.
- FB69 rows now attach `phase2_trace_flux_split_summary` at the accepted row
  state.  If the raw accepted diagnostic state already contains negative trace
  abundances under raw-policy smoke settings, the row preserves those raw
  negative indices and adds a clearly labeled nonnegative counterfactual split
  instead of hiding or repairing the raw state.

Focused verification:

- `tests/test_standard_network.py::TestTraceFluxSplit::test_trace_species_production_destruction_split_reconstructs_network_rhs`
  locks the directional split against the standard network RHS.
- `tests/test_augmented_continuous_ap65_rhs.py::test_fb69_step_cap_ladder_records_finite_rhs_and_bbn_deltas`
  locks the FB69 row payload and raw-negative/counterfactual boundary.

Interpretation:

- BD80 moves the Li/Be/Li6 blocker from a generic trace-stiffness suspicion to
  an inspectable production/destruction ledger that can identify whether large
  `dlogX` stress is real destruction, production over tiny abundance, reverse
  flux cancellation, or a flux-form bug.
- The next implementation target remains a stable positivity-preserving
  network coordinate or constrained implicit network block when this split
  shows the stiff direction that dominates near the failed AP65 endpoint.
- Public dispatch, production SMC validation, publication-ready support, and
  QKE remain unclaimed.

### BD81: Selective Active Trace-Log Solver Coordinate Pilot

Status: historical opt-in solver-coordinate step on the existing FB69
continuous-AP65 row surface.  BD95 supersedes the active-floor-only transform
with an all-finite-nonnegative trace transform plus zero-floor accounting.  The
same no-standalone-gate, no-QKE, no-public-dispatch, and no-production-support
boundaries apply.

Implemented changes:

- The original BD81 pilot encoded only trace abundance entries whose physical
  abundance was above the active floor.  BD95 replaces this with encoding of
  every finite nonnegative constrained trace abundance; exactly zero trace
  entries use `_TRACE_LOG_SOLVER_X_FLOOR`, while negative or nonfinite accepted
  trace entries remain direct-X evidence.
- The transformed path still decodes active trace species back to physical
  abundance units for RHS evaluation, Jacobian conversion, embedded error
  scaling, stage diagnostics, and candidate output.  The row metadata records
  active coordinate steps, inactive-floor counts, and that output truncation is
  not applied.
- `build_augmented_continuous_ap65_source_rhs_prototype_artifact(...)` exposes
  `trace_log_solver_coordinates_enabled=False` by default.  When the caller opts
  in under `abundance_positivity_policy="trace_boundary"`, current FB69 metadata
  reports the BD95 all-nonnegative trace-log coordinate scope directly on the
  same continuous-span artifact surface.

Motivation and bounded evidence:

- BD80 split evidence at the late q4/mu5 failure point showed `7Be` carries a
  real destruction rate of order `2.5e3` per `N`, while inactive `6Li` at
  `X~1e-30` has finite production but `P/X~7e17`.  A naive all-trace log
  transform therefore moves the Li6 floor into an artificial `dlogX` hot spot.
- BD81 fixes that coordinate-level pathology without repairing accepted
  abundances, suppressing raw negative diagnostics, or changing the physical
  network fluxes.

Interpretation:

- BD81 makes the log-coordinate path usable for active Li/Be/He trace states
  while preserving a direct-X fallback for inactive floor species.  It is still
  a pilot and not endpoint completion below `0.01 MeV`.
- The remaining implementation target is to run the selective pilot on the
  failed endpoint replay and then decide whether the next blocker is active
  `7Be` destruction stiffness, frozen-source Jacobian quality, or the need for
  a constrained implicit production/destruction network block.
- Public dispatch, production SMC validation, publication-ready support, and
  QKE remain unclaimed.

### BD82: Positive 3T Solver Coordinates And Recoverable Nonfinite Stages

Status: implemented in current workspace as an opt-in solver-coordinate and
recoverable-domain handling change on the existing FB69 continuous-AP65 row
surface.  It adds no standalone gate, keeps QKE out of scope, keeps public
dispatch closed, and does not claim production support.

Implemented changes:

- FB69 now exposes `temperature_log_solver_coordinates_enabled=False` by
  default.  When explicitly enabled, the packed 3T temperatures are evolved as
  positive log solver coordinates inside the host Rodas5P stage/candidate
  algebra, then decoded back to physical MeV units for RHS evaluation,
  Jacobian conversion, embedded-error scaling, diagnostics, and output.
- The temperature coordinate shares the same physical-unit RHS/Jacobian/error
  conversion path as BD81 trace coordinates but reports separate row metadata:
  `temperature_log_solver_coordinate_step_count`,
  `temperature_log_solver_inactive_floor_count`, and
  `temperature_log_solver_output_truncation_applied=false`.
- Nonfinite linear-system RHS/matrix values produced by transformed stage
  algebra are now classified as recoverable solver-domain rejections instead
  of fatal `ValueError` exits.  The row records
  `linear_system_rhs_nonfinite_rejection_count` and
  `linear_system_matrix_nonfinite_rejection_count`.
- `scripts/run_augmented_continuous_ap65_source_rhs_prototype.py` forwards
  `--trace-log-solver-coordinates` and
  `--temperature-log-solver-coordinates` into the existing FB69 artifact
  builder, so repeat probes no longer require one-off Python calls.

Measured endpoint probe evidence:

- BD81 selective trace-log q4/mu5 all-freedom probe on FB69 reached
  `N_final=2.5829709354498807` with `step_count=390`,
  `attempt_count=517`, `trace_abundance_domain_rejection_count=56`, and
  `stage_temperature_domain_rejection_count=43`.  This reduced trace
  rejections relative to the BD79 baseline but exposed large negative/overflow
  3T stage excursions.  The local artifact was
  `/tmp/rabbit_bd82_selective_trace_log_endpoint_q4_mu5_fb69.json`, SHA
  `b360e1db50023990ab612c2362f260ea68c9f3c63852cf45dcd0082c084af311`.
- BD82 trace+temperature log probe with recoverable nonfinite linear RHS
  handling reached `N_final=2.5831949741940643`, with `step_count=392`,
  `attempt_count=512`, `trace_abundance_domain_rejection_count=27`,
  `stage_temperature_domain_rejection_count=61`, and
  `linear_system_rhs_nonfinite_rejection_count=9`.  The fatal nonfinite-RHS
  exit became an adaptive max-step closure instead.  The local artifact was
  `/tmp/rabbit_bd82_trace_temperature_log_recoverable_rhs_endpoint_q4_mu5_fb69.json`,
  SHA `dd2e5545339fc2d797034f0d7d204766ee68cdfd2648022a3989bba394513416`.

Interpretation:

- BD82 is a concrete solver-domain improvement: it reduces trace-domain
  rejection pressure and keeps transformed nonfinite stage algebra inside the
  recoverable adaptive controller rather than aborting the row.
- It is not endpoint completion below `0.01 MeV`.  The q4/mu5 all-freedom path
  still closes by max-step budget around `T_gamma~0.081 MeV`, with the dominant
  accepted-error block still `X_phase2`.  The next runtime blocker remains
  active trace-network stiffness/overshoot and step-count pressure, not public
  readiness or QKE.
- If five further BD/PR commits fail to produce another concrete runtime,
  physics, solver, or performance breakthrough, pause implementation and write
  a self-contained external-audit prompt describing the solver failure mode.

### BD83: Trace-Tail Patankar Candidate Corrector For Continuous AP65

Status: implemented in current workspace as an opt-in private solver corrector
on the existing FB69 continuous-AP65 row surface.  It adds no standalone gate,
keeps QKE out of scope, keeps public dispatch closed, and does not claim
production support.

Implemented changes:

- FB69 now exposes `trace_tail_patankar_corrector_enabled=False` by default.
  When explicitly enabled, each trial Rodas candidate is updated before
  acceptance for the Li7/Be7/Li6 tail only with a production/destruction
  formula `(X_n + h P)/(1 + h D)` built from the in-tree phase-2 network split.
- The raw Rodas candidate values, Patankar candidate values, per-N production
  and destruction rates, correction magnitude, and mass-fraction delta are
  recorded in row telemetry.  This is a solver candidate update, not output
  truncation; `observable_y_p_truncated=false` and public support remains
  closed.
- `scripts/run_augmented_continuous_ap65_source_rhs_prototype.py` forwards
  `--trace-tail-patankar-corrector` into the same FB69 artifact builder, so
  repeat probes use the existing continuous-span surface instead of adding a
  new readiness gate.

Measured endpoint probe evidence:

- The BD82 trace+temperature coordinate q4/mu5 all-freedom probe reached only
  `N_final=2.5831949741940643` with `step_count=392`,
  `attempt_count=512`, `n_rejected=120`,
  `trace_abundance_domain_rejection_count=27`,
  `stage_temperature_domain_rejection_count=61`, and no finite endpoint
  observables.
- The reviewed BD83 trace+temperature coordinate plus trace-tail Patankar probe
  reached the full requested `N_final=4.8` with `passed=true`,
  `violations=[]`, `step_count=373`, `attempt_count=397`, `n_rejected=24`,
  `trace_abundance_domain_rejection_count=1`,
  `stage_temperature_domain_rejection_count=4`, no nonfinite linear-system
  rejections, and finite staging observables
  (`T_gamma_MeV=0.00914475830896589`, `Yp=0.16318718435928586`,
  `D/H=2.0956470941271044e-05`, `N_eff_3T=2.978160043133659`).  The local
  artifact was
  `/tmp/rabbit_bd83_trace_tail_patankar_accepted_telemetry_q4_mu5_fb69.json`,
  SHA
  `8d98c96b102d9ea35343f3d054449b2066d9085267b04cb379044dc42f6a688e`.
  It recorded `trace_tail_patankar_corrector_attempt_count=392`,
  `trace_tail_patankar_corrector_accepted_count=373`,
  `trace_tail_patankar_corrector_rejected_count=19`, and a dedicated raw
  negative Rodas-candidate sample for the one negative trace-tail event.

Interpretation:

- BD83 opens an endpoint-reaching private q4/mu5 all-freedom candidate: the row
  now reaches below `0.01 MeV` and produces finite BBN readouts only when the
  opt-in, not-yet-converged trace-tail Patankar corrector is enabled.  The
  reviewed implementation also counts the corrector displacement in acceptance
  error diagnostics, separates attempted/accepted/rejected corrector telemetry,
  and preserves raw negative Rodas-candidate evidence outside the general
  sample cap.
- The remaining blocker is no longer "cannot reach the endpoint"; it is
  convergence/physics validation of the new non-conservative trace-tail
  candidate corrector, including h-refinement, mass/charge residual audits,
  and comparison against longer micro-window or network-only references.

### BD84: Trace-Tail Patankar H-Refinement And Conservation Audit Telemetry

Status: implemented on the existing FB69 continuous-AP65 row surface.  No new
readiness gate is added; QKE, public dispatch, and production support remain
out of scope.

Implemented changes:

- Each row now records phase-2 abundance diagnostics for the initial and final
  states: mass-fraction sum, mass-sum residual from unity, charge-weighted
  abundance sum, trace-tail mass/charge sums, and negative abundance count.
  The charge-weighted sum is explicitly an audit quantity, not a conserved
  invariant across weak reactions.
- The trace-tail Patankar telemetry now records accepted/rejected/attempted
  mass-fraction and charge-fraction correction delta maxima in addition to the
  existing correction magnitude and raw negative candidate samples.
- Existing adjacent `h_max` comparisons now include phase-2 abundance deltas
  and Patankar attempt/accepted/rejected telemetry deltas, so h-refinement
  evidence is attached to the same runtime rows instead of a separate gate.

Self-audit:

| field | value |
| --- | --- |
| `real_blocker_moved` | Added same-setup h-refinement and mass/charge audit telemetry for the endpoint-reaching private Patankar candidate. |
| `gate_removed_or_consolidated` | Consolidated into the existing FB69 continuous-span row surface; no standalone BD84 readiness gate was added. |
| `raw_state_preserved` | Final state vectors, raw Rodas candidate negative samples, and untruncated BBN observables remain recorded. |
| `verification` | Focused tests, `sync_test_counts.py`, `git diff --check`, and a real q4/mu5 `h_max=(0.1,0.05)` endpoint probe. |
| `remaining_blocker` | Micro-window/network-only validation of the non-conservative trace-tail corrector, then q/angular/tolerance convergence and statistical-pipeline refresh. |

Measured q4/mu5 h-refinement evidence:

- Artifact:
  `/tmp/rabbit_bd84_trace_tail_patankar_hmax_refinement_q4_mu5_fb69.json`,
  `artifact_payload_sha256=3d451b6448a779756da01b66cff145150b61d1e8817b3ae76ba33859a7f93f56`;
  the written JSON file SHA256 was
  `c5c966472f79c9d2be9f67d0629359553b8bd3c87dadc4cd526b909fa713f64c`.
- Inputs matched the BD83 endpoint probe except `h_max_ladder=(0.1,0.05)` and
  `max_steps=1024`.  Both rows passed and reached `N_final=4.8`.
- Row `h_max=0.1`: `step_count=373`, `n_rejected=24`,
  `trace_tail_patankar_corrector_accepted_count=373`,
  accepted mass-delta max `1.4110550437461175e-10`, accepted charge-delta max
  `6.047383504857119e-11`, final `Yp=0.16318718435928586`,
  `D/H=2.0956470941271044e-05`, and final mass-sum residual
  `-5.191262975046129e-10`.
- Row `h_max=0.05`: `step_count=395`, `n_rejected=13`,
  `trace_tail_patankar_corrector_accepted_count=395`,
  accepted mass-delta max `1.388034567545877e-10`, accepted charge-delta max
  `5.94872422295282e-11`, final `Yp=0.163187193211788`,
  `D/H=2.0956288836728712e-05`, and final mass-sum residual
  `4.099456329953455e-10`.
- Adjacent row deltas: `Yp=8.85250214799349e-09`,
  `D/H=1.8210454233222166e-10`, `N_eff_3T=2.9474269291895894e-07`,
  phase-2 final mass-fraction-sum delta `9.290719304999584e-10`, and
  trace-tail mass-fraction-sum delta `1.0218661236994238e-11`.

Interpretation:

- BD84 moves the candidate from endpoint-only evidence to first same-setup
  h-refinement and conservation-audit evidence.  The deltas are small on this
  two-row ladder, but this remains private diagnostic evidence only.
- The remaining blocker is now micro-window or network-only validation of the
  non-conservative trace-tail corrector, followed by q/angular/tolerance
  convergence and statistical-pipeline refresh.  Public production support and
  QKE remain unclaimed.

### BD85: Trace-Tail Network-Reference Corrector Audit

Status: implemented inside the same FB69 private continuous-AP65 row telemetry.
No standalone gate is added, QKE remains out of scope, and public dispatch
remains closed.

Implemented changes:

- Each applied trace-tail Patankar attempt now computes a frozen-background
  network-only reference for Li7/Be7/Li6 using four production/destruction
  substeps over the same `h`.  The reference refreshes the in-tree phase-2
  production/destruction split at each substep while holding the non-tail
  species and thermodynamic/background state fixed.
- Row telemetry records Patankar-minus-reference absolute, raw relative,
  active-floor relative, mass-fraction, and charge-fraction deltas for
  attempted, accepted, and rejected corrector applications, plus explicit
  reference available/unavailable counts and unavailable-reason counts.  The
  raw relative number is retained for audit but can be dominated by inactive
  trace species; the active-floor relative value is the interpretable scale for
  solver staging.
- This is still a local frozen-background reference, not a full independent
  network-only integration proof and not a publication-grade convergence gate.

Self-audit:

| field | value |
| --- | --- |
| `real_blocker_moved` | Added local network-reference deltas for the non-conservative trace-tail corrector inside the endpoint-reaching solver row. |
| `gate_removed_or_consolidated` | Consolidated into existing FB69 telemetry; no BD85 standalone gate was added. |
| `raw_state_preserved` | Raw candidate samples, untruncated final vectors, and Patankar-minus-reference deltas are recorded. |
| `verification` | Focused BD83/BD84 tests, py_compile, and a real q4/mu5 endpoint probe with network-reference telemetry. |
| `remaining_blocker` | Longer micro-window or full network-only comparison, then q/angular/tolerance convergence and statistical-pipeline refresh. |

Measured q4/mu5 reference evidence:

- Artifact:
  `/tmp/rabbit_bd85_trace_tail_network_reference_q4_mu5_fb69.json`,
  `artifact_payload_sha256=5053a09ef63b86a4889f8be9a5f573fa7484b430240b0874ea588da563d759b6`;
  written JSON file SHA256
  `c5ed880c0ac164ce44fbcf83519140fb71bb920cf42c00dcdebbc0ba83336739`.
- The row passed with `N_final=4.8`, `step_count=373`,
  `attempt_count=397`, `n_rejected=24`,
  `trace_tail_patankar_corrector_accepted_count=373`, and
  `trace_tail_patankar_corrector_rejected_count=19`.
- Reference availability counts were explicit and complete:
  attempts `392/392` available, accepted `373/373` available, rejected
  `19/19` available, with zero unavailable reference payloads.
- Accepted Patankar-vs-reference maxima:
  `abs_delta_max=3.2818022951510465e-11`,
  `mass_delta_abs_max=3.281802379621345e-11`,
  `charge_delta_abs_max=1.4064867461906192e-11`, and
  `active_floor_rel_delta_max=0.037286123183448115`.
- Endpoint observables were unchanged from the BD83/BD84 `h_max=0.1` row:
  `T_gamma_MeV=0.00914475830896589`, `Yp=0.16318718435928586`,
  `D/H=2.0956470941271044e-05`, and
  `N_eff_3T=2.978160043133659`.

Interpretation:

- BD85 does not prove that the trace-tail corrector is globally converged.
  It does show that the accepted one-step corrector is now continuously audited
  against a refreshed local network reference, with absolute local deltas below
  the accepted correction mass-delta scale recorded in BD84.
- The next blocker is a longer micro-window or network-only comparison that
  evolves the full phase-2 network over a real interval rather than only
  auditing individual solver attempts.

### BD86: Trace-Tail Accepted-Window Replay Audit

Status: implemented inside the same FB69 private continuous-AP65 row telemetry.
No standalone gate is added, QKE remains out of scope, and public dispatch
remains closed.

Implemented changes:

- Each accepted trace-tail Patankar step now records a compact in-memory
  start/end phase-2 sample.  After the row finishes, FB69 replays Li7/Be7/Li6
  through the same refreshed production/destruction split across the accepted
  step sequence, using the accepted solver thermodynamic and non-tail phase-2
  background at each step.
- The replay reports completion, unavailable reasons, final
  solver-minus-reference tail values, final absolute/active-floor-relative
  deltas, stepwise maximum deltas, and mass/charge deltas.  It does not alter
  the solver state or truncate outputs.
- This is a multi-step trace-tail replay along the accepted solver background,
  not a full independent phase-2 network integration proof.

Self-audit:

| field | value |
| --- | --- |
| `real_blocker_moved` | Upgraded BD85 from one-step local Patankar references to a 373-step accepted-window trace-tail replay over the endpoint row. |
| `gate_removed_or_consolidated` | Consolidated into existing FB69 telemetry; no BD86 standalone gate was added. |
| `raw_state_preserved` | Raw final vectors, raw Rodas negative samples, and solver-vs-window-reference deltas are recorded separately. |
| `verification` | Red-first helper test, focused BD83-BD86 tests, py_compile, and a real q4/mu5 endpoint probe with accepted-window replay telemetry. |
| `remaining_blocker` | Full phase-2 network-only comparison, then q/angular/tolerance convergence and statistical-pipeline refresh. |

Measured q4/mu5 accepted-window replay evidence:

- Artifact:
  `/tmp/rabbit_bd86_trace_tail_window_reference_q4_mu5_fb69.json`,
  `artifact_payload_sha256=31fca47c82ada46655f088c070a560fae8f07e207d43ab2196b671f4736e7bf6`;
  written JSON file SHA256
  `6348c01618304f800f57e456ceccbfa1ba63d520e0263593bf845c807be49714`.
- The row passed with `N_final=4.8`, `step_count=373`,
  `attempt_count=397`, `n_rejected=24`, and a completed window replay over
  `373` accepted trace-tail steps from `N=0.0` to `N=4.8`.
- Window replay final deltas:
  `final_abs_delta_max=3.853295547106437e-11`,
  `final_active_floor_rel_delta_max=0.01710025512934746`,
  `step_abs_delta_max=4.897935231531469e-11`,
  `mass_fraction_delta=-3.979172293096205e-11`, and
  `charge_fraction_delta=-2.2558303466278645e-11`.
- Endpoint observables remained unchanged from the BD83-BD85 `h_max=0.1` row:
  `T_gamma_MeV=0.00914475830896589`, `Yp=0.16318718435928586`,
  `D/H=2.0956470941271044e-05`, and
  `N_eff_3T=2.978160043133659`.

### BD87: Full Phase-2 Accepted-Background Replay Audit

Status: implemented inside the same FB69 private continuous-AP65 row telemetry.
No standalone gate is added, QKE remains out of scope, public dispatch remains
closed, and the replay does not alter solver states or truncate observables.

Implemented changes:

- FB69 now records accepted-background live weak rates alongside the accepted
  phase-2 start/end samples already used by BD86.
- After each row, FB69 attempts a full nine-species phase-2
  production/destruction replay over the accepted solver thermodynamic and weak
  backgrounds.  It evolves `n`, `p`, `D`, `T`, `He3`, `He4`, `Li7`, `Be7`,
  and `Li6` with the in-tree phase-2 split plus the accepted live
  `lambda_np`/`lambda_pn` weak-rate pair.
- The replay fails closed on nonfinite values or abundance blow-up and reports
  requested steps, completed steps, unavailable steps, completed fraction,
  replay `N_end`, last reference abundance scale, and solver-minus-reference
  deltas.  Deltas from an incomplete replay are explicitly tied to the last
  completed replay step, not the final BBN endpoint.

Self-audit:

| field | value |
| --- | --- |
| `real_blocker_moved` | Upgraded BD86 from trace-tail-only replay to the first full nine-species phase-2 accepted-background replay and exposed its early blow-up point. |
| `gate_removed_or_consolidated` | Consolidated into existing FB69 telemetry; no BD87 standalone gate was added. |
| `raw_state_preserved` | Raw solver final vectors, raw Rodas negative samples, accepted start/end samples, and partial replay deltas are recorded separately. |
| `verification` | Red-first helper tests, review-driven partial-step accounting regression, focused BD84/BD86/BD87 tests, py_compile, and a real q4/mu5 endpoint probe with full-network replay telemetry. |
| `remaining_blocker` | Stabilize the full phase-2 network replay or replace it with a constrained implicit/network-only reference before q/angular/tolerance convergence and statistical-pipeline refresh. |

Measured q4/mu5 full-network replay evidence:

- Artifact:
  `/tmp/rabbit_bd87_phase2_full_network_window_reference_q4_mu5_fb69.json`,
  `artifact_payload_sha256=6b32405c453b03a39ac92e7139f937468b63f0f735f40ac2784e81a917059319`;
  written JSON file SHA256
  `2365ef81ca418f41610ae19501bb8ad35cce6dfb5f51cd44cb904b11b50e8b36`.
- The FB69 row still passed with `N_final=4.8`, `step_count=373`,
  `attempt_count=397`, `n_rejected=24`, and endpoint readouts
  `T_gamma_MeV=0.00914475830896589`, `Yp=0.16318718435928586`,
  `D/H=2.0956470941271044e-05`, and `N_eff_3T=2.978160043133659`.
- The trace-tail BD86 replay still completed all `373` accepted steps with
  unchanged final deltas:
  `final_abs_delta_max=3.853295547106437e-11`,
  `final_active_floor_rel_delta_max=0.01710025512934746`, and
  `step_abs_delta_max=4.897935231531469e-11`.
- The full phase-2 replay did not complete.  It completed `2/373` requested
  accepted steps (`completed_fraction=0.005361930294906166`) and stopped at
  `N_end=0.08719626309362064` with
  `unavailable_reason_counts={"full phase2 reference abundance blow-up limit exceeded": 1}`.
  The recorded blow-up scale was
  `last_reference_abs_max=346823686802.07214` against
  `reference_abs_limit=1000000.0`.
- Incomplete-replay deltas at the last completed replay point were
  `final_abs_delta_max=368454.79995315574`,
  `final_active_floor_rel_delta_max=30.10177348438217`, and
  `step_abs_delta_max=368454.79995315574`; they are not endpoint convergence
  evidence.

### BD88: Adaptive Full Phase-2 Replay Substep Retry

Status: implemented inside the same FB69 private continuous-AP65 row telemetry.
No standalone gate is added, QKE remains out of scope, public dispatch remains
closed, and the replay remains diagnostic accepted-background evidence rather
than a coupled independent BBN solve.

Implemented changes:

- The BD87 full nine-species replay now retries a failed accepted solver step
  with doubled internal production/destruction substeps up to
  `max_substeps_per_solver_step=4096` before reporting that step unavailable.
- Failed solver-sample substeps are not committed into the replay reference
  state.  The row reports the maximum attempted substep count, failed substep
  count, retry reason counts, completed substep total, and completed solver
  step fraction.
- The trace-tail BD86 replay remains applied-step-only; the full phase-2 replay
  uses all accepted solver step samples when the trace-tail corrector is
  enabled, so its requested-step denominator tracks accepted solver coverage.

Self-audit:

| field | value |
| --- | --- |
| `real_blocker_moved` | Tested and ruled out simple internal substep refinement up to 4096 substeps as sufficient for the full phase-2 accepted-background replay. |
| `gate_removed_or_consolidated` | Consolidated into existing FB69 telemetry; no BD88 standalone gate was added. |
| `raw_state_preserved` | Raw solver vectors, raw Rodas negative samples, accepted start/end samples, failed-retry reason counts, and partial replay deltas are recorded separately. |
| `verification` | BD87/BD88 helper tests, partial-substep regression, py_compile, and a real q4/mu5 endpoint probe with adaptive replay telemetry. |
| `remaining_blocker` | Replace the independent per-species production/destruction replay with a conservative coupled implicit/network-only reference, then rerun q/angular/tolerance convergence and statistical-pipeline refresh. |

Measured q4/mu5 adaptive-replay evidence:

- Artifact:
  `/tmp/rabbit_bd88_phase2_full_network_adaptive_replay_q4_mu5_fb69.json`,
  `artifact_payload_sha256=e7301496d6fd050d43da6000d1da35092502be8683076e98a2b17583fb90c03a`;
  written JSON file SHA256
  `6cac028a246aa6ae5cf86bd6024117fd0f543a2e5c40d6f1e8f03175b3dbc4ce`.
- The main FB69 endpoint row still passed with `N_final=4.8`,
  `step_count=373`, `attempt_count=397`, and `n_rejected=24`.
- Adaptive replay did not improve full phase-2 coverage: it still completed
  only `2/373` requested accepted steps
  (`completed_fraction=0.005361930294906166`) and stopped at
  `N_end=0.08719626309362064`.
- The failed step was retried up to `attempted_substep_count_max=4096` with
  `adaptive_substep_retry_count=10`,
  `failed_substeps=4096`, and
  `adaptive_substep_retry_reason_counts={"full phase2 reference abundance blow-up limit exceeded": 10}`.
- The negative result shows that the current independent per-species
  production/destruction replay is not stabilized by smaller internal substeps
  alone; the next implementation should target a coupled conservative implicit
  network-only reference rather than another retry-budget increase.

### BD89: Conservative Directional-Extent Full Phase-2 Replay

Status: implemented inside the same FB69 private continuous-AP65 row telemetry.
No standalone gate is added, QKE remains out of scope, public dispatch remains
closed, and the replay remains a diagnostic accepted-background network-only
reference rather than a coupled independent BBN solve.

Implemented changes:

- FB69 now attaches a second full nine-species replay that applies each
  forward and reverse nuclear reaction as a coupled directional extent in
  abundance-per-baryon space (`Y_i=X_i/A_i`) using the in-tree stoichiometry and
  PRIMAT rate fluxes.  Each extent is limited by the currently available
  reactant abundances before applying the full stoichiometric update.
- Weak `n<->p` conversion is applied as bounded directional extents over the
  accepted-background live `lambda_np`/`lambda_pn` rates.
- The conservative replay records completion fraction, extent-limited counts,
  final solver-minus-reference deltas, and mass/charge deltas next to the BD87
  independent per-species replay.  It does not alter solver state or hide raw
  Rodas candidates.

Self-audit:

| field | value |
| --- | --- |
| `real_blocker_moved` | Replaced the failing independent per-species full-network replay with a coupled stoichiometric extent reference that completes the full 373-step q4/mu5 endpoint row. |
| `gate_removed_or_consolidated` | Consolidated into existing FB69 telemetry; no BD89 standalone gate was added. |
| `raw_state_preserved` | Raw solver vectors, raw Rodas negative samples, accepted start/end samples, independent replay failure telemetry, and conservative replay deltas are recorded separately. |
| `verification` | Conservative extent unit test, focused BD84/BD87/BD88/BD89 tests, py_compile, and a real q4/mu5 endpoint probe with conservative replay telemetry. |
| `remaining_blocker` | Conservative replay and solver-corrected path disagree at endpoint scale, so compare q/angular/tolerance ladders and consider moving the solver corrector from trace-tail Patankar to conservative directional extents. |

Measured q4/mu5 conservative-replay evidence:

- Artifact:
  `/tmp/rabbit_bd89_phase2_conservative_extent_replay_q4_mu5_fb69.json`,
  `artifact_payload_sha256=99e8acdafdae36018a50373824a2c5015d87a253997b59437e696b655dee3ac3`;
  written JSON file SHA256
  `f7a7e96b90ee476bc5db8f266f87f8195fd6f95d68e89aa924bff15494146022`.
- The main FB69 endpoint row still passed with `N_final=4.8`,
  `step_count=373`, `attempt_count=397`, `n_rejected=24`, and endpoint readouts
  `T_gamma_MeV=0.00914475830896589`, `Yp=0.16318718435928586`,
  `D/H=2.0956470941271044e-05`, and `N_eff_3T=2.978160043133659`.
- The older independent full-network replay still failed at `2/373` steps, but
  the conservative directional-extent replay completed `373/373` requested
  accepted steps (`completed_fraction=1.0`) through `N_end=4.8`.
- The conservative replay recorded `extent_limited_count=2438`,
  `weak_extent_limited_count=0`, `mass_fraction_delta=-5.191330698650631e-10`,
  `charge_fraction_delta=0.09192104724730205`,
  `final_abs_delta_max=0.18387853358706596`,
  `final_active_floor_rel_delta_max=11.434752559453116`, and
  `step_abs_delta_max=0.5079428702787598`.
- The completed conservative reference endpoint was `He4=0.3470657179463518`
  versus the solver endpoint `He4=0.16318718435928586`; this is not convergence
  evidence, but it replaces the previous replay blow-up with a finite,
  conservative network-only comparison target.

### BD90: Full Phase-2 Conservative Candidate Corrector Probe

Status: implemented as an opt-in private FB69 solver candidate corrector and
kept closed as a diagnostic/rescue path.  No public dispatch, production SMC,
QKE, or convergence claim is added.

Implemented changes:

- FB69 exposes `phase2_conservative_extent_corrector_enabled` and the CLI flag
  `--phase2-conservative-extent-corrector`.  It reuses the BD89 conservative
  directional-extent update as a candidate corrector for all nine phase-2
  species while preserving raw Rodas candidate values and raw negative samples.
- The conservative corrector is now `domain_rescue_only`: it skips candidates
  that are already inside the phase-2 trace domain and only attempts a full
  conservative replacement when the raw candidate would otherwise be rejected
  for trace-domain violation.
- Existing trace-tail Patankar behavior remains backward-compatible and
  separate.  The new full conservative candidate path records attempt,
  accepted/rejected, raw-negative, extent-limited, mass-delta, and charge-delta
  telemetry on the existing FB69 row.

Self-audit:

| field | value |
| --- | --- |
| `real_blocker_moved` | Tested moving the conservative extent update from replay telemetry into the solver candidate path; the unconditional form was ruled out, and the landed form is restricted to domain rescue. |
| `gate_removed_or_consolidated` | Consolidated into existing FB69 telemetry and CLI; no BD90 standalone gate was added. |
| `raw_state_preserved` | Raw Rodas candidate values, raw negative candidate samples, corrected values, and rejected correction telemetry are recorded separately. |
| `verification` | BD90 unit/integration tests, py_compile, and real q4/mu5 probes of unconditional and rescue-only conservative corrector behavior. |
| `remaining_blocker` | The full conservative candidate corrector does not yet clear the full endpoint; the next implementation should use conservative replay as a convergence/step-policy diagnostic while keeping the trace-tail solver path as the currently passing q4/mu5 route. |

Measured q4/mu5 conservative-corrector evidence:

- Unconditional prototype artifact:
  `/tmp/rabbit_bd90_phase2_conservative_extent_corrector_q4_mu5_fb69.json`,
  `artifact_payload_sha256=3368dca01eed79f78d9232bc8be67e0bb631b95df84764e3aca3e330edcf9e59`;
  written JSON file SHA256
  `43eec085d21742d56ccd53e6c9a5bcc09279cad247eb133cfd04a7fe003690cb`.
  It accepted only `1` step (`N_final=2.2122913581559576e-07`) before
  fail-closing; conservative correction attempts reached
  `correction_abs_max=0.43162265524573284`, so unconditional all-species
  replacement is too intrusive for the existing error controller.
- Rescue-only artifact:
  `/tmp/rabbit_bd90_phase2_conservative_extent_rescue_q4_mu5_fb69.json`,
  `artifact_payload_sha256=e7287f18d1d9ea54bf64ad1457c06dc618e58de6d36e04dabd89d0de1766956a`;
  written JSON file SHA256
  `f54c5b507a6e524280b7526d7f43ba94cb2364b14b2681bba1578b4dba5f2b0c`.
  It advanced to `N_final=2.952937223169137` with `512` accepted steps and
  `98` rejected attempts before fail-closing on the existing `max_steps`
  budget.  The conservative rescue attempted `43` raw-negative candidates, all
  rejected by the correction/error machinery; the accepted solver path therefore
  remained unmodified by the full conservative corrector.
- The rescue-only run still produced a completed conservative replay through
  the reached window (`512/512`, `N_end=2.952937223169137`) with
  `final_abs_delta_max=0.1841069053305247`, while the independent per-species
  replay again failed after `2/512` steps.  This keeps the next blocker focused
  on step-policy/convergence interpretation rather than another retry budget.

### BD91: Phase-2 Telemetry Folded Into FB70

BD91 keeps the existing FB69/FB70 surfaces and carries the phase-2
replay/corrector diagnostics through FB70 rows, h-refinement attempt rows,
resolution summaries, and adjacent diagnostic deltas.  The folded fields include
full-network replay status, conservative extent replay status, trace-tail
Patankar counters, conservative-corrector counters, and raw candidate
negative/domain/corrector samples.  This is a consolidation of solver telemetry
into the full-span comparison surface, not a new readiness gate or public
promotion.

Self-audit:

| field | value |
| --- | --- |
| `real_blocker_moved` | FB70 can now compare phase-2 replay and candidate-corrector diagnostics across span, h-refinement, and resolution rows. |
| `gate_removed_or_consolidated` | Existing FB69 telemetry is folded into the existing FB70 span ladder; no standalone gate was added. |
| `raw_state_preserved` | Raw negative/domain/corrector samples are preserved through FB70 selected rows and h-refinement attempts. |
| `verification` | FB70 focused tests, FB69/FB70/WBS/registry focused suite, sync_test_counts.py, py_compile, and git diff --check. |
| `remaining_blocker` | Endpoint-backed q/angular/tolerance convergence and reconciliation against conservative phase-2 replay/corrector diagnostics. |

### BD92: Positive Solver Coordinates Routed Through FB70

BD92 forwards the existing FB69 private positive-coordinate solver options
`trace_log_solver_coordinates_enabled` and
`temperature_log_solver_coordinates_enabled` through FB70 span-ladder,
resolution-ladder, freedom-composition, and CLI paths.  FB70 rows and summaries
now preserve positive log-coordinate step counters, and claim boundaries state
that these are solver-coordinate controls rather than output truncation.

A real CPU-JAX/Rodas5P FB70 smoke with `frozen_source_jax`, boundary trace,
trace-boundary abundance policy, trace+temperature log solver coordinates, and
a tiny weak-only span passed as hot-endpoint evidence with
`T_gamma=0.7999999999606833 MeV`,
`positive_log_solver_coordinate_step_count_total=1`,
`temperature_log_solver_coordinate_step_count_total=1`, and
`trace_log_solver_coordinate_step_count_total=0` because no trace abundance was
above the active log-coordinate floor in that initial tiny span.  The artifact
payload hash was
`e5fd3d462617cb096a0acf3b96405475a2342b4c5d2fba804f40b756c44a1aef`.
This does not claim endpoint completion, public production support, QKE
support, or publication-ready convergence.

Self-audit:

| field | value |
| --- | --- |
| `real_blocker_moved` | Endpoint-capable positive solver coordinates can now be exercised by the existing FB70 full-span and resolution surfaces. |
| `gate_removed_or_consolidated` | The existing FB70 surface was extended; no standalone gate was added. |
| `raw_state_preserved` | Raw FB69 rows, h-refinement attempts, and untruncated observables remain embedded in FB70. |
| `verification` | Red-first FB70 forwarding regression, CLI dry-run coverage, full FB70 focused tests, and real CPU-JAX/Rodas5P smoke. |
| `remaining_blocker` | q/angular/tolerance convergence of the endpoint-reaching trace-tail route and reconciliation against conservative phase-2 replay/corrector diagnostics. |

### BD93: Phase-2 Replay Mass/Charge Residuals In FB70 Resolution Deltas

BD93 keeps the existing FB69/FB70 private surfaces and carries the phase-2
window-reference mass/charge residuals into the same comparison rows that
already hold completed-fraction and final-abundance deltas.  FB69 now exposes
top-level trace-tail window mass/charge residuals and summary maxima for
trace-tail, full-network, and conservative directional-extent window
references.  FB70 selected rows and adjacent resolution comparisons now include
those signed-residual absolute maxima, so q/angular/tolerance ladders can
compare the conservative replay discrepancy without opening a new diagnostic
gate.

This remains solver telemetry on the CPU-JAX/Rodas5P private path.  Raw solver
states and raw negative candidate samples are still preserved separately, and
the residual summaries do not truncate or repair output abundances, do not
claim public production support, and do not change QKE scope.

Self-audit:

| field | value |
| --- | --- |
| `real_blocker_moved` | Conservative replay disagreement can now be compared across FB70 resolution axes using mass/charge residuals, not only final abundance deltas. |
| `gate_removed_or_consolidated` | Existing FB69 telemetry is folded into the existing FB70 resolution comparison surface; no standalone gate was added. |
| `raw_state_preserved` | Raw rows, raw candidate samples, and untruncated observables remain embedded; BD93 adds residual summaries only. |
| `verification` | Red-first FB69/FB70 telemetry regressions, focused RHS/FB70 tests, py_compile, and git diff --check. |
| `remaining_blocker` | Endpoint-backed q/angular/tolerance convergence of the trace-tail route and reducing the conservative replay versus solver endpoint discrepancy. |

### BD94: Block-Max Rodas Error Acceptance Norm

BD94 changes the private FB69 host Rodas5P adaptive acceptance norm from a
single scalar RMS over the whole packed state to the maximum RMS over the
existing state blocks: geometry/thermo, A-modes, and phase-2 abundances.  The
component scaling remains `atol + rtol * |y|` in physical units; BD94 does not
introduce per-species error weights or species-specific tolerances.  It prevents
large `X_phase2` embedded errors from being hidden by the many A-mode entries,
while preserving the scalar RMS as diagnostic telemetry.

The returned error diagnostics now record `scalar_rms_norm`,
`acceptance_norm`, and `acceptance_norm_policy=block_max_rms_over_state_blocks`.
This is a runtime solver-control change on the CPU-JAX/Rodas5P private path,
not a gate, output truncation, public production support, or QKE support.

A real tiny CPU-JAX/Rodas5P FB69 smoke with `frozen_source_jax`, boundary trace,
trace-boundary abundance policy, trace+temperature log solver coordinates, and
no collision payload passed with
`artifact_payload_sha256=2b7bb44362ebf966da27f2fc0ed96378f3393ae239b485c066cc349bd4e02311`.
The accepted row recorded `dominant_block=X_phase2`,
`scalar_rms_norm=0.0006356929888458115`, and
`acceptance_norm=0.0014983427438853923`, showing that the controller used the
block-max norm rather than the diluted scalar RMS.

Self-audit:

| field | value |
| --- | --- |
| `real_blocker_moved` | Adaptive acceptance now directly controls the phase-2 block instead of allowing full-state scalar RMS dilution. |
| `gate_removed_or_consolidated` | Existing FB69 Rodas5P step control changed in place; no standalone gate was added. |
| `raw_state_preserved` | Error diagnostics preserve both scalar RMS and block-max norms; solver states and observables remain untruncated. |
| `verification` | Red-first error-norm regressions, focused RHS/FB70/WBS/registry tests, py_compile, and git diff --check. |
| `remaining_blocker` | Rerun endpoint q/angular/tolerance ladders with trace-tail solver coordinates and compare conservative replay residuals under the stricter block-max controller. |

### BD95: All Nonnegative Trace Log Solver Coordinates

BD95 supersedes the BD81 active-floor trace-coordinate pilot inside the same
private FB69 host Rodas5P path.  When trace log coordinates are explicitly
enabled with `abundance_positivity_policy=trace_boundary`, every finite
nonnegative constrained trace species `[T, He3, He4, Li7, Be7, Li6]` is now
encoded as a positive solver coordinate.  Positive sub-active abundances such
as `1e-30` retain their physical value through the encode/decode map; exactly
zero trace entries are encoded at `_TRACE_LOG_SOLVER_X_FLOOR` with
`trace_log_solver_encoded_floor_count` recording the event.  Negative or
nonfinite accepted trace entries are not silently repaired: they remain outside
the transform so the existing raw domain evidence and rejection path can still
surface them.

The artifact metadata now reports
`trace_log_solver_coordinate_scope=all_nonnegative_trace_abundance_solver_state_with_zero_floor`
and
`trace_log_solver_inactive_species_policy=negative_or_nonfinite_trace_state_remains_direct_X_for_domain_evidence`.
The host Rodas5P base RHS/Jacobian is evaluated at the decoded solver-coordinate
base state, so exactly-zero trace species use the same floor in the base vector,
Jacobian, stages, and candidate; initial RHS seed reuse is used only when that
decoded base state is identical to the supplied raw state.
This is an invariant-domain solver-coordinate change only.  It is not output
truncation, a standalone gate, public dispatch, production support, or QKE.

A real tiny CPU-JAX/Rodas5P FB69 smoke with `frozen_source_jax`, boundary trace,
trace-boundary abundance policy, trace+temperature log solver coordinates, and
no collision payload passed with
`artifact_payload_sha256=0b836a3cadda293cbc5d35054c2aedf4985f3a04ed129ec350de47e124f7c0dc`.
The row recorded `trace_log_solver_coordinate_step_count=40`,
`trace_log_solver_inactive_floor_count=0`,
`temperature_log_solver_coordinate_step_count=40`, and
`error_norm_max=4.626783530044199e-06` under the BD94 block-max norm.

Self-audit:

| field | value |
| --- | --- |
| `real_blocker_moved` | The trace-coordinate fix now covers zero and sub-active trace species consistently across base RHS/Jacobian, stages, and candidates instead of leaving them in direct-X stage algebra. |
| `gate_removed_or_consolidated` | Existing FB69 solver coordinate transform changed in place; no new wrapper or gate was added. |
| `raw_state_preserved` | Positive sub-active values decode back to their raw values; zero-floor encodes are counted; negative/nonfinite accepted trace states remain direct-X evidence rather than being repaired. |
| `verification` | Red-first trace-transform regression, focused transform/host-step/builder tests, real tiny FB69 CPU-JAX/Rodas5P smoke, focused RHS/FB70/WBS/registry tests, py_compile, and git diff --check. |
| `remaining_blocker` | Rerun endpoint q/angular/tolerance ladders and inspect whether remaining trace-domain rejections are overflow/nonfinite solver-stage failures, network stiffness, or conservative replay mismatch. |

### BD96: Sub-Active Trace Solver Scale

BD96 fixes the BD95 all-trace coordinate regression without reverting the
all-nonnegative trace encoding.  The failed q4/mu5 endpoint replay showed that
positive but sub-active trace seeds at `X=1e-30` generated `dX/X` stage
increments of order `1e13-1e14`; the resulting Rodas stage vector overflowed
birth trace coordinates and underflowed depleted trace coordinates before any
accepted step.  FB69 now keeps those species in the positive solver-coordinate
path with the internally consistent coordinate
`z = log1p(X / _TRACE_LOG_SOLVER_ACTIVE_FLOOR)`, so
`dX/dz = X + _TRACE_LOG_SOLVER_ACTIVE_FLOOR`.  Trace
RHS/Jacobian/error conversion uses that derivative scale.  This preserves the
physical ODE in transformed coordinates while avoiding the artificial `dX/X`
singularity for birth species below the active floor.  Trace decode maps
extreme positive coordinates to `inf` and extreme negative coordinates to a
negative active-floor abundance without NumPy exp overflow/underflow warnings,
so recoverable stage-domain rejection evidence remains explicit.

This is a solver-coordinate change on the existing FB69 private host Rodas5P
path.  It does not truncate public output, repair negative/nonfinite accepted
trace evidence, add a gate, claim public dispatch, claim production support, or
introduce QKE.

The q4/mu5 CPU-JAX/Rodas5P endpoint replay that failed immediately after BD95
with `N_final=0`, `step_count=0`, and source payload exceptions now reaches
`N_final=4.8` with no row violations under the same q/angular grid and
`frozen_source_jax_full_jvp`, `auto_small_collision_reuse`, trace+temperature
solver coordinates, and trace-tail Patankar candidate correction when
`max_steps=640`.  The artifact hash was
`ded80431a849cdf85c3dfb9854af30e4b33302949e378665dc35232cfebd8ac4`; the row
recorded `step_count=486`, `attempt_count=512`, `n_rejected=26`,
`trace_tail_patankar_corrector_accepted_count=486`, and
`phase2_conservative_extent_window_reference_replay_N_end=4.8`.

Self-audit:

| field | value |
| --- | --- |
| `real_blocker_moved` | The all-trace positive coordinate path no longer stalls at the first q4/mu5 endpoint step; it reaches `N=4.8` on CPU-JAX/Rodas5P with the private full-JVP frozen-source policy. |
| `gate_removed_or_consolidated` | Existing FB69 transform/Rodas conversion changed in place; no standalone wrapper, readiness gate, hash gate, or claim gate was added. |
| `raw_state_preserved` | Accepted states remain untruncated; negative/nonfinite accepted trace states stay outside the transform; rejected stage overflow/underflow still surfaces through recoverable domain evidence. |
| `verification` | Red-first sub-active log1p solver-scale, finite-difference transformed-Jacobian, and safe-decode regressions plus real q4/mu5 endpoint CPU-JAX/Rodas5P replay. |
| `remaining_blocker` | Step budget and physics fidelity remain blockers: the q4/mu5 endpoint now needs 486 accepted steps/512 attempts, and the conservative full-network replay still reports a large mismatch after early replay failure. |

### BD97: Conservative Extent Candidate Activation

BD97 turns the existing private
`--phase2-conservative-extent-corrector` path from a trace-domain rescue-only
no-op into an actual full nine-species conservative directional-extent
candidate update whenever the flag is enabled.  The corrected candidate now
uses an 8-substep refined extent replay, records the 4-substep coarse replay,
and evaluates acceptance against the refined-minus-coarse conservative local
error rather than the raw Rodas candidate displacement.  This keeps raw Rodas
candidate values in telemetry while making the opt-in conservative candidate
path internally consistent.

The same change also stops classifying `z=-inf` trace log coordinates as a
domain failure when they decode to the safe-atol boundary value
`X=-_TRACE_LOG_SOLVER_ACTIVE_FLOOR`; nonfinite decoded abundances and decoded
values below the safe-atol boundary still fail closed.

The q4/mu5 CPU-JAX/Rodas5P conservative-candidate probe now applies the
corrector (`phase2_conservative_extent_corrector_accepted_count_total=1`,
`rejected_count_total=11`) and the conservative extent replay matches the
accepted corrected step to `final_abs_delta_max=2.372171364812073e-10` with
mass delta `4.930823790466995e-16`.  The artifact hash was
`b0cf136ef28e731d8b2daa0398b7a2b953e278ef46e21675c5af47518da6bfef`.
The row still fails at `N_final=2.22122087094942e-07` because the next Rodas
stage enters nonfinite trace log coordinates after the conservative accepted
state.  This is not endpoint support.

Self-audit:

| field | value |
| --- | --- |
| `real_blocker_moved` | The conservative extent candidate flag now performs a real full-network candidate update and exposes the next blocker: Rodas trace-coordinate stage instability after the first conservative accepted step. |
| `gate_removed_or_consolidated` | Existing FB69 conservative-candidate plumbing changed in place; no standalone readiness, hash, figure, or claim gate was added. |
| `raw_state_preserved` | Raw Rodas candidate values, raw negative candidate values, coarse/refined conservative values, and local-error values are recorded separately; accepted output is not truncated. |
| `verification` | Red-first conservative always-apply regression, conservative coarse/refined local-error regressions, safe boundary decode regression, and real q4/mu5 CPU-JAX/Rodas5P conservative-candidate probe. |
| `remaining_blocker` | Full conservative extent candidate does not yet reach the endpoint; after one accepted tiny step the next Rodas stage still produces nonfinite trace log coordinates. |

### BD98: Conservative Extent Operator Split

BD98 removes the full phase-2 abundance block from the Rodas stage algebra when
the existing private `--phase2-conservative-extent-corrector` flag is enabled.
In that mode FB69 now zeros the `X_phase2` RHS block and the `X_phase2`
Jacobian rows/columns for the host Rodas5P solve, then applies the conservative
directional-extent candidate as the accepted-step network update.  This changes
the existing private conservative-corrector path in place; it does not add a new
gate, public dispatch, production support, output truncation, or QKE.

The q4/mu5 CPU-JAX/Rodas5P conservative-candidate probe now gets past the BD97
post-corrector trace-coordinate stage failure.  It records
`phase2_conservative_extent_corrector_accepted_count_total=400`,
`rejected_count_total=240`, `trace_abundance_domain_rejection_count=0`,
`stage_temperature_domain_rejection_count=0`, and
`N_final=2.9920208477777853e-06` before exhausting the `max_steps=640` attempt
budget.  Artifact hash:
`5fdb6428def859cf5420649d1c0d83dfa0b8d834ed0366ca8676c436a489ca58`.
This is not endpoint support.  The dominant remaining error block is
`phase2_conservative_extent_corrector`; its refined-minus-coarse local error
shrinks the global step rather than allowing Rodas stage-domain rejection.  A
follow-up review fix also makes the operator split fail closed if the
conservative update is unavailable; this probe recorded
`phase2_conservative_extent_corrector_missing_update_count=0`.

Self-audit:

| field | value |
| --- | --- |
| `real_blocker_moved` | Conservative extent no longer feeds stiff `X_phase2` stages into Rodas; the BD97 nonfinite trace-coordinate stage failure is replaced by a conservative-network local-error/max-steps blocker. |
| `gate_removed_or_consolidated` | Existing FB69 conservative-corrector/Rodas paths changed in place; no standalone readiness, manifest, hash, figure, or claim gate was added. |
| `raw_state_preserved` | Raw Rodas candidate values, corrected extent values, and local-error values remain separate telemetry; accepted output is not clipped to hide negative or nonfinite abundance evidence. |
| `verification` | Operator-split RHS/Jacobian unit regressions, RHS-only hot-path regression, full-JVP Jacobian regression, fail-closed missing-update regression, and real q4/mu5 CPU-JAX/Rodas5P conservative-candidate probe. |
| `remaining_blocker` | Conservative network substep/local-error control is now the blocker: the run reaches only `N=2.9920208477777853e-06` before max-step exhaustion, with `phase2_conservative_extent_corrector` dominating rejected-error norms. |

### BD99: Phase-Split Hot Weak Candidate

BD99 changes the existing private FB69
`phase2_conservative_extent_corrector_enabled` path in place so the conservative
candidate no longer activates the full nuclear network at the hot
`T_gamma=0.8 MeV` start.  For proposed steps whose start and candidate
temperatures both remain above the configurable private
`phase2_network_activation_T_gamma_MeV` threshold, the candidate applies an
exact frozen-background `n <-> p` weak update and leaves D/T/He/Li/Be/Li6 at
the accepted-step start values.  At, below, or crossing into the activation
threshold, the existing full phase-2 conservative directional-extent candidate
remains the network update.  The default threshold remains `0.08 MeV` only as a
private standard-BBN staging default; the artifact explicitly records that it
is not a validated universal constant for strong Bianchi anisotropy, dynamic
collision payloads, or distorted neutrino thermodynamics.  This is a private
phase-split solver correction, not a standalone gate, not public dispatch, not
production support, not QKE, and not output truncation.

The q4/mu5 CPU-JAX/Rodas5P conservative probe now moves the blocker from the
hot-start conservative local-error collapse to the lower-temperature network
activation region.  It records
`phase2_conservative_extent_corrector_accepted_count_total=640`,
`phase2_conservative_extent_corrector_rejected_count_total=18`,
`trace_abundance_domain_rejection_count=0`, `stage_temperature_domain_rejection_count=0`,
and `N_final=2.6010666700970373` before exhausting `max_steps=640`.  Artifact
hash: `07bb2d6c252f8d1903b3e81bb4fb2a94409c64bb3acb289fc163f52434c7637e`.
This is not endpoint support.  The last attempted state has
`T_gamma=0.07999656235483887 MeV`, finite nonnegative accepted phase-2
abundances, and a conserved mass-fraction sum within
`6.38378239159465e-14`; the remaining failure is step-size collapse at the
activation boundary/full-network onset, with
`phase2_conservative_extent_corrector` still dominating most error norms.  The
same artifact records
`phase2_network_activation_threshold_validation.validation_status=requires_model_specific_validation`,
`anisotropy_detected=true`, `dynamic_collision_payload_active=true`, and
`neutrino_temperature_distortion_detected=true`.

Self-audit:

| field | value |
| --- | --- |
| `real_blocker_moved` | The run progresses from `N=2.9920208477777853e-06` to `N=2.6010666700970373`; hot phase-2 full-network activation is replaced by exact weak-only phase-1 evolution for steps that remain above the configurable activation threshold. |
| `gate_removed_or_consolidated` | Existing FB69 conservative-corrector plumbing changed in place; no standalone readiness, manifest, hash, figure, or claim gate was added. |
| `raw_state_preserved` | Raw Rodas candidate values remain recorded separately from corrected weak-only/full-network candidate values and local-error values; accepted output is not clipped to hide negative abundance evidence. |
| `verification` | Red-first hot weak-only candidate regression, activation-crossing regression, configurable-threshold/claim-bound regression, conservative-corrector focused subset, py_compile, CLI dry-run, and real q4/mu5 CPU-JAX/Rodas5P conservative probe. |
| `remaining_blocker` | Activation-boundary event handling and a real full-network implicit/embedded estimator are still missing; endpoint below `0.01 MeV` remains unsupported. |

### BD100: Activation Event Step Limiter

BD100 changes the existing private FB69
`phase2_conservative_extent_corrector_enabled` path in place so a raw Rodas
candidate that crosses the private phase-2 activation temperature no longer
feeds the full conservative network corrector across that switch surface.  If
the accepted step base is still above `phase2_network_activation_T_gamma_MeV`
and the raw candidate is at or below it, FB69 records a
`private_phase2_activation_event_step_limiter` payload, rejects that attempt,
and retries with `h` linearly limited to just before the activation crossing.
Once the start state is within the activation-event tolerance, the existing
full-network candidate path is allowed to proceed.  This is a solver-control
change inside the existing FB69 surface, not a new gate, not a public-dispatch
claim, not production support, not QKE, and not output abundance truncation.

The q4/mu5 CPU-JAX/Rodas5P conservative probe records
`phase2_activation_event_limiter_count_total=5`,
`phase2_conservative_extent_corrector_accepted_count_total=640`,
`phase2_conservative_extent_corrector_rejected_count_total=13`,
`step_count=640`, `attempt_count=658`, `n_rejected=18`, and
`N_final=2.6010680162198607` before exhausting `max_steps=640`.  Artifact hash:
`ff0310a516524f4fa419d3f4c7807d905309a409ca30237ee4ba5fbbab4b5882`.
The dominant-error counts are
`phase2_activation_event_step_limiter=5`,
`phase2_conservative_extent_corrector=624`, `geometry_thermo=17`, and
`A_modes_flat=12`.  The conservative accepted-background replay reaches the
same `N_end=2.6010680162198607`, while the full phase-2 reference remains
unstable (`completed_fraction_min=0.00625`,
`replay_N_end_max=0.3072889717271672`,
`last_reference_abs_max=376427566798.0082`).  This is not endpoint support; it
removes the activation switch from the corrector input and exposes that the
remaining blocker is the full-network conservative/implicit estimator.

Self-audit:

| field | value |
| --- | --- |
| `real_blocker_moved` | Activation-crossing attempts are now handled as event-limited rejections before the full phase-2 corrector runs; the measured conservative-corrector rejection count dropped from BD99's `18` to `13` on the same q4/mu5 probe. |
| `gate_removed_or_consolidated` | Existing FB69 step-control/corrector plumbing changed in place; no standalone readiness, manifest, hash, figure, or claim gate was added. |
| `raw_state_preserved` | Raw Rodas crossing candidates are recorded in activation-event limiter samples and are not clipped into accepted abundances. |
| `verification` | Red-first activation-event helper regression, step-loop no-corrector-before-limiter regression, conservative-corrector focused subset, and real q4/mu5 CPU-JAX/Rodas5P conservative probe. |
| `remaining_blocker` | Endpoint below `0.01 MeV` remains blocked by the full phase-2 network update/error estimator after activation; an NSE handoff plus MPRK/backward-Euler-style implicit network candidate is still required. |

### BD101: Deuterium Bottleneck Handoff Seed

BD101 changes the existing private FB69 conservative phase-2 candidate in place
so the first post-activation network attempts no longer start from a pure
floor-state D abundance.  When the full-network branch is active and the
non-n/p light species are still below the private handoff seed threshold, FB69
computes a post-consumption deuterium target from the existing PRIMAT R0
`n+p <-> D+gamma` forward/reverse flux balance at the local `T_gamma` and
`eta`.  It solves the scalar balance after the equal `X_n`/`X_p` donor draw,
then raises only `X_D` toward that detailed-balance target, preserving mass fraction and
charge-fraction diagnostics to roundoff.  No He4/Li/Be species are invented,
negative inputs fail closed without repair, and no accepted observable is
truncated.

The q4/mu5 CPU-JAX/Rodas5P conservative probe records
`phase2_deuterium_handoff_seed_applied_count_total=12`,
`phase2_deuterium_handoff_seed_delta_abs_max=0.0007592006995322019`,
`phase2_activation_event_limiter_count_total=5`,
`phase2_conservative_extent_corrector_accepted_count_total=640`,
`phase2_conservative_extent_corrector_rejected_count_total=13`,
`step_count=640`, `attempt_count=658`, `n_rejected=18`, and
`N_final=2.6023722966786704` before exhausting `max_steps=640`.  Artifact hash:
`6b62fa643550233b3139fe07f127fd3287ea8b47cd7ef7cd1f7059c7a1b0a63c`.
The conservative accepted-background replay reaches
`N_end=2.6023722966786704`, but the full phase-2 reference remains unstable
(`completed_fraction_min=0.00625`, `replay_N_end_max=0.3072889717271672`,
`last_reference_abs_max=376427566798.0082`).  This is not endpoint support; it
removes the floor-state deuterium handoff as the immediate activation input,
while leaving the implicit/full-network estimator blocker exposed.

Self-audit:

| field | value |
| --- | --- |
| `real_blocker_moved` | Full-network activation now receives a mass/charge-conserving R0 deuterium bottleneck seed instead of pure D floor state; the measured `N_final` moved from BD100's `2.6010680162198607` to `2.6023722966786704`. |
| `gate_removed_or_consolidated` | Existing FB69 conservative-candidate plumbing changed in place; no standalone readiness, manifest, hash, figure, or claim gate was added. |
| `raw_state_preserved` | Raw candidate values and handoff-seed deltas are recorded separately; negative abundance inputs are not hidden or clipped. |
| `verification` | Red-first handoff-seed conservation regression, conservative-candidate seed-use regression, focused conservative-corrector tests, and real q4/mu5 CPU-JAX/Rodas5P conservative probe. |
| `remaining_blocker` | Endpoint below `0.01 MeV` still requires replacing the directional-extent update/error estimate with a positive implicit/MPRK-style network candidate. |

### BD102: Positive Implicit Directional Extents

BD102 changes the same private FB69 conservative phase-2 candidate in place:
the full-network branch now uses a positive implicit directional-extent solve
for each forward/reverse reaction direction instead of hard-clipping an explicit
reaction extent to the remaining donor pool.  The helper solves the end-point
reactant equation for common one-reactant and two-reactant mass-action cases
with analytic roots, falls back to a bounded scalar bisection for higher-order
reverse directions, applies weak `n<->p` evolution as the exact frozen-background
two-state update inside each substep, and records `hard_extent_clipping_applied`
separately.  Raw Rodas candidates remain recorded; negative/nonfinite initial
network states fail closed rather than being repaired.

The optimized q4/mu5 CPU-JAX/Rodas5P conservative probe records
`artifact_payload_sha256=e6ffba6a04a3b47c2e9214266e14eab4094196307cbfb46b7df9a6cfbd78d3d0`,
`phase2_conservative_extent_corrector_accepted_count_total=640`,
`phase2_conservative_extent_corrector_implicit_extent_solved_count_total=119808`,
`phase2_conservative_extent_corrector_hard_clip_applied_count_total=0`,
`phase2_conservative_extent_corrector_weak_exact_update_count_total=4992`,
`phase2_deuterium_handoff_seed_applied_count_total=13`,
`step_count=640`, `attempt_count=658`, `n_rejected=18`, and
`N_final=2.6010419227122354` before exhausting `max_steps=640`.
The all-bisection variant reached only 565 accepted corrector steps in the
same 180-second wall budget, so the analytic roots are required for the CPU-JAX
target.  BD102 does not claim endpoint support: the full phase-2 reference still
fails (`completed_fraction_min=0.00625`, `replay_N_end_max=0.3072889717271672`,
`last_reference_abs_max=376427566798.0082`), and the accepted span endpoint is
slightly below BD101's `2.6023722966786704`.

Self-audit:

| field | value |
| --- | --- |
| `real_blocker_moved` | The accepted full-network branch no longer relies on hard donor clipping for reaction extents; it uses positive implicit directional extents plus real coarse/refined local-error differences. |
| `gate_removed_or_consolidated` | Existing FB69 corrector plumbing changed in place; no standalone readiness, manifest, hash, figure, or claim gate was added. |
| `raw_state_preserved` | Raw Rodas candidate values remain separate from corrected values; negative or nonfinite initial network states fail closed. |
| `verification` | Red-first positive implicit extent regression, updated conservative-candidate integration regressions, telemetry regression, focused conservative-corrector subset, and real q4/mu5 CPU-JAX/Rodas5P probe. |
| `remaining_blocker` | Endpoint below `0.01 MeV` still requires a coupled implicit/full-network solve or MPRK-grade estimator that improves physical burn and reference stability without hiding failed states. |

### BD103: Adaptive Internal Positive-Implicit Pair Budget

BD103 changes the existing FB69 conservative phase-2 candidate in place again:
the full-network positive-implicit branch now builds an adaptive coarse/refined
internal substep pair instead of hard-wiring one `4/8` pair.  The internal
criterion uses the same physical-X RMS scaling family as the Rodas phase-2
corrector block, records the max scaled component separately, and caps the
accepted-path refinement at `16` substeps after cap-64 probes showed target-miss
cost inflation without endpoint progress.  This is not a new gate; it is a
runtime solver/error-estimator change on the existing private corrector.

The q4/mu5 CPU-JAX/Rodas5P cap-16 probe records
`artifact_payload_sha256=3b2f078bda56c50b0007b68402fcbfb191e0b2854f70468f59e59d94a30c825c`,
`phase2_conservative_extent_corrector_accepted_count_total=640`,
`phase2_conservative_extent_corrector_implicit_extent_solved_count_total=239616`,
`phase2_conservative_extent_corrector_internal_attempt_count_total=1872`,
`phase2_conservative_extent_corrector_internal_attempted_substeps_total=17472`,
`phase2_conservative_extent_corrector_internal_max_refined_substeps_reached_count_total=624`,
`phase2_conservative_extent_corrector_hard_clip_applied_count_total=0`,
`N_final=2.601051816017013`, and raw final `X_He4=3.10302777320292e-08`.
Relative to BD102, this roughly doubles the tiny accepted He4 burn and moves
`N_final` upward by about `9.9e-06`, but it also doubles implicit extent solves
and still does not reach the post-BBN endpoint.  Cap-64 probes reached only
`548--550` accepted steps in the 180-second budget and never met the internal
target, so BD103 keeps the cheaper cap-16 runtime budget.

Self-audit:

| field | value |
| --- | --- |
| `real_blocker_moved` | The full-network candidate no longer has a fixed `4/8` local-error pair; it can refine internally and now exposes the corrector substep budget as measured runtime telemetry. |
| `gate_removed_or_consolidated` | Existing FB69 corrector code changed in place; no readiness, manifest, hash, figure, publication, public-dispatch, production-support, or QKE gate was added. |
| `raw_state_preserved` | Raw Rodas candidates, corrected values, local-error values, and max-refinement target-miss telemetry are recorded separately; no output truncation is introduced. |
| `verification` | Red-first adaptive-pair regression, updated conservative-candidate regressions, focused conservative-corrector subset, and real q4/mu5 CPU-JAX/Rodas5P before/after probe. |
| `remaining_blocker` | Endpoint below `0.01 MeV` remains blocked by both hot-loop positive-implicit extent cost and insufficient physical full-network burn; a coupled implicit/MPRK-grade network update is still required. |

### BD104: Coupled Backward-Euler/Newton Network Corrector

BD104 stops using the positive implicit directional-extent pair as the accepted
activated full-network candidate.  The existing FB69 conservative phase-2
corrector now keeps the BD98 operator split in place and replaces the activated
network branch with a network-only coupled backward-Euler/Newton solve in
abundance-per-baryon variables.  A full-versus-two-half adaptive pair uses
block-max physical-X error over n/p, burn, and trace species; if the cap is
reached without meeting the target, the corrector fails closed rather than
accepting a target-miss candidate.  The old directional-extent solver remains
available as a positivity/conservative diagnostic and reference helper, but it
is no longer the default accepted full-network update.

The new corrector records Newton convergence, residual-evaluation, Jacobian,
linear-solve, positivity-line-search, and raw negative Newton-trial
count/minimum/first-vector telemetry in the existing row and aggregate
surfaces.  This preserves raw evidence and makes the next q4/mu5 probe
comparable against the BD103
`239616` scalar directional-extent solve budget.  No output truncation, public
dispatch, production-support, publication, or QKE claim is added.

Self-audit:

| field | value |
| --- | --- |
| `real_blocker_moved` | The accepted activated full-network candidate now solves the nine-species network as one coupled implicit BE/Newton block instead of iterating the directional-extent pair that BD103 showed was target-miss/cost dominated. |
| `gate_removed_or_consolidated` | Existing FB69 corrector code changed in place; the old directional-extent path is retained only as diagnostic/reference code, and no standalone gate or wrapper was added. |
| `raw_state_preserved` | Raw Rodas candidates, corrected values, BE local-error vectors, raw negative Newton-trial counts/minima/first-vector samples, line-search telemetry, and mass/charge deltas remain surfaced; final observables are not truncated. |
| `verification` | Red-first BE/Newton step and cap-target-miss regressions, updated conservative-candidate routing tests, full focused RHS test file, and py_compile.  Long q4/mu5 endpoint/burn probe remains to run after this code step. |
| `remaining_blocker` | The next blocker is empirical: run the q4/mu5 CPU-JAX/Rodas5P activation probe to measure whether BE/Newton improves He4 burn, target-hit count, and endpoint progress without excessive Newton/Jacobian cost. |

### BD105: BDF2/Newton Network Subcycling

BD105 keeps the BD98 operator split and BD104 coupled network solve, but raises
the accepted activated network substep from first-order BE to a BE-started
BDF2/Newton sequence.  The adaptive full-versus-half pair is still network-only
and still uses physical-X block-max error over n/p, burn, and trace species.
The old directional extent solver remains diagnostic/reference-only, and the
BD104 BE/Newton helper remains tested as the lower-order control.  No output
truncation, public dispatch, production-support, publication, or QKE claim is
added.

Measured q4/mu5 CPU-JAX/Rodas5P evidence:

- BD103 directional extent cap-16 baseline:
  `N_final=2.601051816017013`, raw final
  `X_He4=3.10302777320292e-08`,
  `phase2_conservative_extent_corrector_internal_error_target_met_count_total=0`,
  `phase2_conservative_extent_corrector_internal_max_refined_substeps_reached_count_total=624`,
  `phase2_conservative_extent_corrector_implicit_extent_solved_count_total=239616`,
  `wall_seconds_total=158.35451461200137`.
- BD104 BE/Newton baseline:
  `artifact_payload_sha256=dc0904e23ef647e049e3c58ab01bb3df664b2e75b9540ceeb6019b99d24814fa`,
  `N_final=2.601415834334738`, raw final
  `X_He4=4.0819020417956984e-06`,
  `phase2_conservative_extent_corrector_internal_error_target_met_count_total=480`,
  `phase2_conservative_extent_corrector_newton_converged_count_total=4808`,
  `phase2_conservative_extent_corrector_missing_update_count=128`,
  `wall_seconds_total=131.5501189000206`.
- BD105 BDF2/Newton:
  `artifact_payload_sha256=c4843cb29e7d70171ac270e4c6c536ae7cc1e0f4bc21eccb69cc3a33f34a58de`,
  `N_final=2.6044412715585126`, raw final
  `X_He4=0.00013650657657444313`,
  `phase2_conservative_extent_corrector_internal_error_target_met_count_total=481`,
  `phase2_conservative_extent_corrector_refined_substeps_max=64`,
  `phase2_conservative_extent_corrector_newton_converged_count_total=23168`,
  `phase2_conservative_extent_corrector_missing_update_count=126`,
  `wall_seconds_total=176.59483786707278`.

BD105 is a real burn-order breakthrough, not endpoint support: relative to
BD104 it moves `N_final` by `~0.0030` and increases accepted raw He4 by about
`33x`, while preserving mass and positivity in the accepted candidate.  The
remaining blocker is now the expensive BDF2 finite-difference Jacobian hot loop
and the activation-onset host retry pattern; endpoint below `0.01 MeV` still
requires either analytic/sparse network Jacobians, a higher-order/less expensive
network error estimator, a model-specific activation diagnostic, or some
combination of those.

Self-audit:

| field | value |
| --- | --- |
| `real_blocker_moved` | The accepted activated network update now uses a second-order BDF2/Newton network subcycle that increases measured q4/mu5 He4 burn by about `33x` over BD104. |
| `gate_removed_or_consolidated` | Existing FB69 corrector code changed in place; no standalone readiness, manifest, hash, figure, publication, public-dispatch, production-support, or QKE gate was added. |
| `raw_state_preserved` | Raw Rodas candidates, BDF2 corrected values, local-error vectors, Newton trial negativity telemetry, and mass/charge residuals remain surfaced; final observables are not truncated. |
| `verification` | Red-first BDF2 activation-error regression, BDF2 negative-linear-base/positive-Newton-guess regression, BDF2 candidate target-miss telemetry regression, routing regression, focused BD99-BD105 corrector subset, py_compile, and real q4/mu5 CPU-JAX/Rodas5P comparison probe. |
| `remaining_blocker` | Endpoint below `0.01 MeV` remains blocked by BDF2 finite-difference Newton cost and activation-onset host retries, not by directional-extent target-miss behavior. |

### BD106: Analytic Network Jacobian For BDF2/Newton

BD106 removes the finite-difference network-Jacobian hot loop from the
coupled BE/BDF2 Newton solve when the standard PRIMAT-backed network flux
function is active.  The residual Jacobian is assembled analytically in
abundance-per-baryon variables from the same directional forward/reverse flux
products used by `compute_flux_components`; non-standard flux functions still
fall back to finite differences with explicit telemetry.  This is runtime
solver work on the existing FB69 path: it does not add a gate, reroute public
dispatch, truncate outputs, claim production support, or touch QKE.

Measured q4/mu5 CPU-JAX/Rodas5P evidence:

- BD105 BDF2/Newton finite-difference baseline:
  `N_final=2.6044412715585126`,
  `phase2_conservative_extent_corrector_newton_converged_count_total=23168`,
  `phase2_conservative_extent_corrector_newton_residual_evaluation_count_total=278016`,
  `phase2_conservative_extent_corrector_newton_jacobian_evaluation_count_total=23168`,
  `phase2_conservative_extent_corrector_missing_update_count=126`, and
  `wall_seconds_total=176.59483786707278`.
- BD106 analytic network Jacobian:
  `artifact_payload_sha256=000c434cf108be4d7c51713f64887e466c60e0c2180d2da69a0c85281f30277b`,
  `N_final=2.6044412762134264`,
  `phase2_conservative_extent_corrector_newton_converged_count_total=23168`,
  `phase2_conservative_extent_corrector_newton_residual_evaluation_count_total=69504`,
  `phase2_conservative_extent_corrector_newton_finite_difference_residual_evaluation_count_total=0`,
  `phase2_conservative_extent_corrector_newton_jacobian_evaluation_count_total=23168`,
  `phase2_conservative_extent_corrector_missing_update_count=126`, and
  `wall_seconds_total=144.12645068601705`.

BD106 is a cost breakthrough, not an endpoint or physics-promotion claim: it
cuts Newton residual evaluations by exactly `4x` on the q4/mu5 probe and
removes all finite-difference residual evaluations from the network Jacobian
path, while leaving the same missing-update/host-retry blocker and endpoint
failure below `0.01 MeV`.

Self-audit:

| field | value |
| --- | --- |
| `real_blocker_moved` | The BDF2/Newton network corrector no longer pays nine residual evaluations for each 9x9 finite-difference Jacobian on the standard network path. |
| `gate_removed_or_consolidated` | Existing FB69 solver internals changed in place; no standalone readiness, manifest, hash, figure, publication, public-dispatch, production-support, or QKE gate was added. |
| `raw_state_preserved` | The positivity line search and raw Newton trial negativity telemetry remain unchanged; analytic Jacobian fallback telemetry records whether finite differences were used. |
| `verification` | Red-first analytic-vs-finite-difference Jacobian regression, analytic Newton residual-evaluation regression, focused BD105/BD106 subset, py_compile, and q4/mu5 CPU-JAX/Rodas5P comparison probe. |
| `remaining_blocker` | Endpoint below `0.01 MeV` is now blocked mainly by activation-onset host retries/missing updates and the 64-substep BDF2 controller pattern, not by finite-difference network-Jacobian cost. |

### BD107: BDF2 Activation-Onset Substep Cap

BD107 keeps the BD98 operator split, BD105 BDF2/Newton network subcycling, and
BD106 analytic standard-network Jacobian intact, but raises the accepted
activated network subcontroller cap from 64 to 128 refined BDF2 substeps.  This
is not a new gate and does not restore the positive implicit directional-extent
pair as the accepted solver.  It is a measured runtime solver change against a
captured activation-onset target-miss sample: the 64-substep analytic-Jacobian
pair reached local-error norm `0.5889751083490837`, while the 128-substep pair
reaches `0.1456371126806533` under the same target `0.2` without
finite-difference Jacobian residual evaluations.

Measured q4/mu5 CPU-JAX/Rodas5P evidence:

- BD106 analytic-Jacobian cap-64 baseline:
  `artifact_payload_sha256=000c434cf108be4d7c51713f64887e466c60e0c2180d2da69a0c85281f30277b`,
  `N_final=2.6044412762134264`,
  `phase2_conservative_extent_corrector_newton_residual_evaluation_count_total=69504`,
  `phase2_conservative_extent_corrector_newton_finite_difference_residual_evaluation_count_total=0`,
  `phase2_conservative_extent_corrector_missing_update_count=126`, and
  `wall_seconds_total=144.12645068601705`.
- BD107 analytic-Jacobian cap-128:
  `artifact_payload_sha256=62de50f337f277e715249b730c6ecfc1882c4961f32c65f2927561b6b4f04a7a`,
  `N_final=2.6188010100730494`,
  `phase2_conservative_extent_corrector_newton_residual_evaluation_count_total=168576`,
  `phase2_conservative_extent_corrector_newton_finite_difference_residual_evaluation_count_total=0`,
  `phase2_conservative_extent_corrector_missing_update_count=124`,
  `phase2_conservative_extent_corrector_refined_substeps_max=128`, and
  `wall_seconds_total=178.12946439301595`.

BD107 is progress with an explicit cost tradeoff, not endpoint support: relative
to BD106 it moves `N_final` forward by about `0.01436` and eliminates one
captured activation-onset target miss at the network-subcontroller level, but
it raises Newton residual evaluations by about `2.4x` and returns close to the
180-second staging budget.  Endpoint below `0.01 MeV` remains blocked.

Self-audit:

| field | value |
| --- | --- |
| `real_blocker_moved` | A captured activation-onset BDF2 target miss that failed at the 64-substep cap now accepts at 128 substeps, and the q4/mu5 run advances from `N_final=2.6044412762134264` to `2.6188010100730494`. |
| `gate_removed_or_consolidated` | Existing FB69 BDF2/Newton solver internals changed in place; no standalone readiness, manifest, hash, figure, publication, public-dispatch, production-support, or QKE gate was added. |
| `raw_state_preserved` | Raw Rodas candidates, BDF2 one-full/two-half candidates, Newton trial negativity telemetry, and mass/charge residuals remain surfaced; final observables are not truncated. |
| `verification` | Red-first activation-onset cap regression, focused BD106/BD107 subset, py_compile, and q4/mu5 CPU-JAX/Rodas5P comparison probe. |
| `remaining_blocker` | Endpoint below `0.01 MeV` is now blocked by the high cost of starting every network pair from low substep counts near activation, remaining host retries/missing updates, and the absence of a model-specific activation diagnostic. |

### BD108: BDF2 Error-Predicted Substep Jump

BD108 keeps the accepted activated network branch on the BD104-BD107 coupled
BE/BDF2/Newton path and adds an error-predicted jump inside the existing
network-only step-doubling subcontroller.  When a coarse/refined pair misses the
internal target by a large margin, the next pair estimates the refined substep
count from the observed local-error ratio and can skip obviously
under-resolved intermediate powers of two.  The acceptance rule is unchanged:
the corrected network candidate is still accepted only when the physical-X
block-max local-error norm meets the same target.  No positive implicit
directional extent path is restored, and no standalone gate, public dispatch,
production-support, publication, or QKE claim is added.

Focused evidence:

- Synthetic under-resolved pair: the adaptive sequence changes from
  `4,8,16,32,64,128` to `4,8,64,128`, while still accepting only on the
  `64/128` pair after the local-error target is met.
- Captured BD107 activation-onset sample:
  `refined_substeps=128`,
  `local_error_internal_norm=0.1456371126806533`,
  `adaptive_internal_substep_retry_count=1`,
  `adaptive_internal_substep_jump_count=1`,
  `internal_attempt_count=4`, and
  `internal_attempted_substeps_total=204`; the previous no-jump path required
  six internal attempts and `252` attempted substeps for the same accepted
  `128`-substep result.
- q4/mu5 CPU-JAX/Rodas5P probe:
  `/tmp/rabbit_bd108_bdf2_jump_q4_mu5.json`,
  `artifact_payload_sha256=267ded2b902cd016e1bbc5d7056b9907637073517540bc8a3642b0a0b4a0c464`,
  `selected_wall_seconds_total=167.95700480300002`,
  `selected_step_count_total=508`,
  `selected_adaptive_attempt_count_total=643`,
  `selected_phase2_conservative_extent_window_reference_replay_N_end_max=2.6198945026118343`,
  and `passed=false`.

BD108 is a cost/startup-controller optimization, not endpoint support.  Relative
to BD107's q4/mu5 probe (`wall_seconds_total=178.12946439301595`,
`phase2_conservative_extent_window_reference_replay_N_end=2.6188010100730494`),
the wall time improves by about `10.2` seconds and the conservative reference
N-end moves slightly forward, but the row still fails with missing final BBN
observables and endpoint below `0.01 MeV` remains blocked.

Self-audit:

| field | value |
| --- | --- |
| `real_blocker_moved` | Repeated low-substep startup in the BDF2 network subcontroller is reduced for measured under-resolved pairs without changing the acceptance target or re-enabling the directional extent solver. |
| `gate_removed_or_consolidated` | Existing FB69 BDF2/Newton solver internals changed in place; no standalone readiness, manifest, hash, figure, publication, public-dispatch, production-support, or QKE gate was added. |
| `raw_state_preserved` | Raw Rodas candidates, BDF2 one-full/two-half candidates, Newton trial negativity telemetry, and mass/charge residuals remain surfaced; final observables are not truncated. |
| `verification` | Red-first predicted-jump regression, focused BD104-BD108 subset, py_compile, direct captured-onset probe, and q4/mu5 CPU-JAX/Rodas5P comparison probe. |
| `remaining_blocker` | Endpoint below `0.01 MeV` remains blocked by host retries/max-step exhaustion, a hard private activation threshold requiring model-specific replacement, and physical burn/network-controller convergence past the activation window. |

### BD109: Local Activation Diagnostic Payload

BD109 keeps the private `0.08 MeV` activation temperature as a fallback, but no
longer leaves the model-specific threshold problem as prose only.  The existing
activation validation payload now computes a local diagnostic at the runtime
decision point from the current phase-2 abundances, photon temperature, `eta`,
network truncation, and optional accepted-background `H_rate_s`/host step:

- R0 `n+p -> D+gamma` forward/reverse fluxes and balance ratio.
- The existing PRIMAT R0 detailed-balance deuterium target from the BD101
  handoff seed helper.
- Downstream D-consuming / T-He3-He4-producing reaction indices and flux scale.
- A dimensionless downstream-burn change estimate when `H_rate_s` and `h` are
  available.

The threshold role is now
`private_staging_default_with_local_diagnostics`.  The diagnostic status is
`computed_not_yet_activation_trigger`: BD109 does not yet replace the event
surface and does not claim that `0.08 MeV` is valid under strong anisotropy,
dynamic collision payloads, or distorted neutrino temperatures.  It does not
add a standalone gate, public dispatch, production-support, publication, output
truncation, or QKE path.

Self-audit:

| field | value |
| --- | --- |
| `real_blocker_moved` | The activation decision point now exposes local bottleneck/burn diagnostics needed to replace the hard private temperature fallback in a following runtime PR. |
| `gate_removed_or_consolidated` | Existing FB69 activation-validation payload changed in place; no standalone readiness, manifest, hash, figure, publication, public-dispatch, production-support, or QKE gate was added. |
| `raw_state_preserved` | The diagnostic reads current raw phase-2 abundances and fails closed on nonfinite or negative inputs; it does not repair or truncate abundances or observables. |
| `verification` | Red-first activation-diagnostic payload regression, updated activation-threshold role regression, focused BD99-BD101/BD109 activation-corrector subset, and py_compile. |
| `remaining_blocker` | Endpoint below `0.01 MeV` still requires using the local diagnostic to step-limit/activate the full network, plus resolving host retries/max-step exhaustion and post-activation burn convergence. |

### BD110: Model-Specific Activation Requires Local Bottleneck

BD110 changes the activated phase-2 branch selection in place.  The `0.08 MeV`
temperature remains a private fallback guard, but in model-specific states
where the activation validation status is `requires_model_specific_validation`
and the local diagnostic is available, the full network is no longer activated
by the temperature fallback alone.  The candidate must also satisfy the local
deuterium-bottleneck policy, currently a private `deuterium_equilibrium_X_D >=
1e-3` threshold with a small event-signal tolerance and downstream-burn
Damkohler telemetry.  Near-FLRW/equilibrium rows or rows without available
local diagnostics keep the previous temperature-or-local fallback behavior.

Two rejected BD110 variants are recorded here because they are useful solver
evidence, not because they are landed behavior:

- A local-only activation event limiter repeatedly narrowed host steps at the
  diagnostic surface and failed by `h_min` around `T_gamma ~= 0.08133 MeV`.
- A local diagnostic early-trigger policy avoided that `h_min` loop but moved
  the q4/mu5 probe into an earlier expensive burn window and regressed the
  180-second comparison.

Measured q4/mu5 CPU-JAX/Rodas5P evidence for the landed policy:

- BD108 baseline:
  `/tmp/rabbit_bd108_bdf2_jump_q4_mu5.json`,
  `artifact_payload_sha256=267ded2b902cd016e1bbc5d7056b9907637073517540bc8a3642b0a0b4a0c464`,
  `selected_wall_seconds_total=167.95700480300002`,
  `selected_step_count_total=508`,
  `selected_adaptive_attempt_count_total=643`,
  `selected_phase2_conservative_extent_window_reference_replay_N_end_max=2.6198945026118343`,
  and `passed=false`.
- BD110 model-specific local-required policy:
  `/tmp/rabbit_bd110_model_specific_local_required_q4_mu5.json`,
  `artifact_payload_sha256=eba9c86ded2347f1d8e9bca7ef96b2a48d9f0ec9609f835c547aae6de3b47c33`,
  file SHA256
  `b5aa09a382949de6e30e562d89efd77cdb597ac603f678f20f773bd28dc46c88`,
  `selected_wall_seconds_total=180.05199413001537`,
  `selected_step_count_total=433`,
  `selected_adaptive_attempt_count_total=549`,
  `selected_phase2_conservative_extent_window_reference_replay_N_end_max=2.623157403134011`,
  and `passed=false`.

BD110 is a small runtime-policy improvement, not endpoint support.  It retires
unconditional hard-`0.08 MeV` activation for anisotropic/dynamic/distorted rows
with available local diagnostics and moves the measured conservative reference
window slightly forward, but it still exhausts the 180-second staging budget and
does not produce final BBN observables.

Self-audit:

| field | value |
| --- | --- |
| `real_blocker_moved` | Model-specific rows no longer treat the hard private `0.08 MeV` fallback as sufficient for activating the full network; activation must pass the local bottleneck diagnostic when available. |
| `gate_removed_or_consolidated` | Existing FB69 activation/corrector runtime code changed in place; no standalone readiness, manifest, hash, figure, publication, public-dispatch, production-support, or QKE gate was added. |
| `raw_state_preserved` | Raw phase-2 states remain read without repair, failed diagnostic inputs fail closed, raw Rodas candidates and Newton negativity telemetry remain surfaced, and final observables are not truncated. |
| `verification` | Red-first model-specific activation regressions, focused BD99-BD101/BD109/BD110 subset, full RHS/WBS focused suite, py_compile, and q4/mu5 CPU-JAX/Rodas5P comparison probe. |
| `remaining_blocker` | Endpoint below `0.01 MeV` remains blocked by phase2-corrector-dominated host attempts, post-activation burn convergence, and the need for a robust continuous-extension local event solve before local-only event limiting can be reintroduced. |

### BD111: Split Accepted Network-Corrector Error From Host Rodas Control

BD111 changes the accepted phase-2 conservative-corrector control path in the
FB69 host loop.  The coupled BE/BDF2/Newton network subcontroller remains the
authority for accepting or failing a phase-2 network candidate; when it reports
success, its physical-X pair error is preserved as telemetry but is no longer
folded back into the host Rodas5P error norm.  Failed or missing network
updates still reject the host attempt.  The older trace-tail Patankar corrector
keeps the previous host-error-control behavior.

This directly addresses the BD108-BD110 symptom where accepted network
candidate local-error estimates repeatedly became the dominant host rejection
block even after the network subcontroller had met its own target.  BD111 does
not loosen the network subcontroller target and does not repair or truncate
abundances; it only removes the second host-level gate on an already accepted
network update.

Measured q4/mu5 CPU-JAX/Rodas5P evidence for the same 180-second staging row:

- BD110 model-specific local-required policy:
  `/tmp/rabbit_bd110_model_specific_local_required_q4_mu5.json`,
  `selected_wall_seconds_total=180.05199413001537`,
  `selected_step_count_total=433`,
  `selected_adaptive_attempt_count_total=549`,
  `selected_phase2_conservative_extent_window_reference_replay_N_end_max=2.623157403134011`,
  and `passed=false`.
- BD111 host-control split:
  `/tmp/rabbit_bd111_host_split_q4_mu5.json`,
  `artifact_payload_sha256=17c01101f8ca0be576a4c3ebcd26a5224e15384da2876cc0f05da7007bf1ef36`,
  file SHA256
  `170e8d8754cb93cf4c7c2eba05695338d24f1db98e951b54c6f284e5b27212f3`,
  `wall_seconds_total=180.08855348592624`,
  `N_final=2.630433999972621`,
  `step_count=379`,
  `attempt_count=610`,
  `n_rejected=231`,
  `phase2_conservative_extent_corrector_accepted_count_total=379`,
  `phase2_conservative_extent_corrector_internal_error_target_met_count_total=337`,
  `phase2_conservative_extent_corrector_internal_attempt_count_total=1349`,
  `phase2_conservative_extent_corrector_refined_substeps_max=128`,
  and `passed=false`.

BD111 moves the measured conservative-corrector window forward from BD110's
`2.623157403134011` to `2.630433999972621` under the same budget.  It also
exposes the next blocker more clearly: remaining phase-2-dominant rejections
are now mostly missing network updates where the BDF2/Newton subcontroller
misses the target at the 128-substep cap, not accepted-update telemetry being
double-counted by the host controller.

Self-audit:

| field | value |
| --- | --- |
| `real_blocker_moved` | Accepted phase-2 network corrections are no longer double-gated by the host Rodas error norm after the network subcontroller has met its own target. |
| `gate_removed_or_consolidated` | Existing FB69 runtime controller behavior changed in place; no standalone readiness, manifest, hash, figure, publication, public-dispatch, production-support, or QKE gate was added. |
| `raw_state_preserved` | The correction telemetry keeps raw network pair-error magnitude, raw candidate values, Newton negativity telemetry, and `output_truncation_applied=False`; failed network updates remain fail-closed. |
| `verification` | Red-first BD111 host-control regression, full RHS/WBS focused suite, py_compile, and q4/mu5 CPU-JAX/Rodas5P comparison probe. |
| `remaining_blocker` | Endpoint below `0.01 MeV` remains blocked by BDF2/Newton subcontroller target misses at the 128-substep cap, post-activation burn stiffness, and the legacy full-network window reference blow-up. |

### BD112: BDF2/Newton Refined-Substep Capacity

BD112 keeps the BD98 operator split, BD105 coupled BE/BDF2/Newton activated
network branch, BD106 analytic standard-network Jacobian, BD108 error-predicted
substep jump, BD110 model-specific local-bottleneck activation policy, and
BD111 host/network controller split.  It changes the existing BDF2/Newton
network subcontroller capacity in place from `128` to `1024` refined substeps.
This is not a standalone gate, does not restore the directional-extent solver
as the accepted activated-network method, and does not claim endpoint or public
production support.

The immediate trigger was the BD111 q4/mu5 evidence: accepted network-pair
errors were no longer double-gated by the host controller, so the remaining
phase-2 rejections concentrated in missing BDF2/Newton updates at the 128
refined-substep cap.  A network-specific host retry limiter was tested and
discarded because it regressed the measured q4/mu5 row.  The discarded artifact,
`/tmp/rabbit_bd112_network_retry_limiter_q4_mu5.json`, had
`artifact_payload_sha256=92b30993d8e9ed3f896d28eabd15c6e9ff7d3689d9a771a37dc375b31ca28d84`,
`wall_seconds_total=180.13437562505715`,
`N_final=2.629757148845655`, and `passed=false`, which is below the BD111
`N_final=2.630433999972621`.

Measured q4/mu5 CPU-JAX/Rodas5P evidence for the same 180-second staging row:

- BD111 host/network controller split:
  `/tmp/rabbit_bd111_host_split_q4_mu5.json`,
  `N_final=2.630433999972621`,
  `phase2_conservative_extent_corrector_refined_substeps_max=128`,
  and `passed=false`.
- Actual-code BD112 cap `1024`:
  `/tmp/rabbit_bd112_bdf2_cap1024_q4_mu5.json`,
  `artifact_payload_sha256=b635f58048f1465363758ef25bc24444e1ade2bc83228472102a857d2e479b7d`,
  file SHA256
  `2de0f1f23e77598491b263c604692eb498918d61e7a3641572c6ef951a21fff4`,
  `wall_seconds_total=180.06371211295482`,
  `N_final=2.7137160704223873`,
  `step_count=132`,
  `attempt_count=202`,
  `n_rejected=70`,
  `phase2_conservative_extent_corrector_accepted_count_total=132`,
  `phase2_conservative_extent_corrector_internal_error_target_met_count_total=103`,
  `phase2_conservative_extent_corrector_internal_max_refined_substeps_reached_count_total=0`,
  `phase2_conservative_extent_corrector_internal_attempted_substeps_total=157140`,
  `phase2_conservative_extent_corrector_refined_substeps_max=1024`,
  and `passed=false`.
- Actual-code cap `2048` comparison:
  `/tmp/rabbit_bd112_bdf2_cap2048_q4_mu5.json`,
  `artifact_payload_sha256=9951758022844761d9d5bc06bc6e6d57b12b117735ea70c79b1c49cf325808b8`,
  file SHA256
  `ca9e66d7ed5bd4b1a6c511ad33e2ed18d603f1722a7f0b0a6910d05cbc692741`,
  `wall_seconds_total=180.3589538380038`,
  `N_final=2.7059926846321596`,
  `phase2_conservative_extent_corrector_refined_substeps_max=2048`,
  and `passed=false`.

BD112 therefore keeps `1024`: it removes max-refined-substeps hits from the
accepted-corrector summary telemetry and advances the conservative-corrector
window from BD111's `2.630433999972621` to `2.7137160704223873`, while cap
`2048` spends more budget per accepted correction and regresses the 180-second
N progression.  Later BD114 inspection showed that rejected missing-update
attempts could still reach the `1024` cap under the stricter `0.2` target; the
legacy full-network window reference is also still not repaired at BD112:
`phase2_full_network_window_reference_replay_N_end=0.3072889717271672` and
`phase2_full_network_window_reference_charge_delta_abs_max=364198.46994798543`.

Self-audit:

| field | value |
| --- | --- |
| `real_blocker_moved` | The measured BDF2/Newton cap-miss blocker exposed by BD111 is reduced in the live q4/mu5 row: accepted-corrector cap-hit telemetry drops to zero and the conservative-corrector window advances by about `0.0832820704507657` in `N` under the same wall budget. |
| `gate_removed_or_consolidated` | Existing FB69 BDF2/Newton solver capacity changed in place; no standalone readiness, manifest, hash, figure, publication, public-dispatch, production-support, or QKE gate was added. |
| `raw_state_preserved` | Raw Rodas candidates, BDF2 one-full/two-half candidates, Newton trial negativity telemetry, mass/charge residuals, and `output_truncation_applied=False` remain surfaced; failed final observables stay unavailable. |
| `verification` | Focused BDF2 cap regression, full RHS/WBS focused suite, py_compile, sync-test-count check, q4/mu5 CPU-JAX/Rodas5P cap comparison probes, and diff check. |
| `remaining_blocker` | Endpoint below `0.01 MeV` remains blocked by post-activation burn stiffness/cost and the legacy full-network window reference blow-up, not by the 128-substep BDF2 cap. |

### BD113: Fold Legacy Full-Network Window Reference Into BDF2 Replay

BD113 removes a redundant post-run evidence path from the active FB69 artifact:
the legacy `phase2_full_network_window_reference` slot is now populated from
the accepted phase-2 BE/BDF2/Newton corrector replay instead of separately
running the older production-destruction full-network reference.  The legacy
helper remains available for focused historical unit tests, but the q4/mu5
runtime artifact no longer spends a second post-run reference path that reports
a contradictory blow-up after the accepted BDF2 replay has already completed.

This is a consolidation of older reference plumbing, not a new gate.  It does
not change the host Rodas solve, does not change the accepted network update,
does not truncate outputs, and does not claim endpoint support.  The nested
payload records `legacy_production_destruction_reference_replayed=false` and
`consolidated_from_scope=accepted_solver_background_phase2_corrector_replay` so
downstream readers can distinguish the folded reference from the older replay.

Measured q4/mu5 CPU-JAX/Rodas5P evidence for the same 180-second staging row:

- BD112 legacy full-network reference:
  `/tmp/rabbit_bd112_bdf2_cap1024_q4_mu5.json`,
  `phase2_full_network_window_reference_replay_N_end_max=0.3072889717271672`,
  `phase2_full_network_window_reference_completed_fraction_min=0.030303030303030304`,
  `phase2_full_network_window_reference_failed_substeps_max=4096`,
  `phase2_full_network_window_reference_charge_delta_abs_max=364198.46994798543`,
  and `phase2_full_network_window_reference_mass_delta_abs_max=422141.1580566204`.
- BD113 consolidated reference:
  `/tmp/rabbit_bd113_consolidated_fullref_q4_mu5.json`,
  `artifact_payload_sha256=94ddae4adf7804ff6e1130c1a7c214c7f8a61a5b01d0f9eaf2b516f2f5d23e0f`,
  file SHA256
  `a0164451b9555c75637824040f5f1128bb9700a21b45f88cbfb97aa8dac9b32b`,
  `wall_seconds_total=180.2209013379179`,
  `phase2_full_network_window_reference_replay_N_end_max=2.7137160704223873`,
  `phase2_full_network_window_reference_completed_fraction_min=1.0`,
  `phase2_full_network_window_reference_failed_substeps_max=0`,
  `phase2_full_network_window_reference_unavailable_step_count_total=0`,
  `phase2_full_network_window_reference_charge_delta_abs_max=2.2655005245807524e-06`,
  `phase2_full_network_window_reference_mass_delta_abs_max=9.11933235089174e-12`,
  and `passed=false`.

Self-audit:

| field | value |
| --- | --- |
| `real_blocker_moved` | The artifact no longer reports the accepted BDF2 replay and a contradictory legacy P/D full-network blow-up for the same accepted window; the full-network reference slot now reaches the same `N=2.7137160704223873` replay endpoint as the accepted corrector reference. |
| `gate_removed_or_consolidated` | Older full-network reference plumbing is consolidated into the accepted BDF2/Newton replay surface; no standalone readiness, manifest, hash, figure, publication, public-dispatch, production-support, or QKE gate was added. |
| `raw_state_preserved` | Raw Rodas candidates, BDF2 one-full/two-half candidates, Newton negativity telemetry, mass/charge residuals, and `output_truncation_applied=False` remain surfaced; failed final observables stay unavailable. |
| `verification` | Focused window-reference regression, q4/mu5 CPU-JAX/Rodas5P artifact comparison, full RHS/WBS focused suite, py_compile, sync-test-count check, review, and diff check. |
| `remaining_blocker` | Endpoint below `0.01 MeV` remains blocked by the live solver's post-activation burn stiffness/cost and final-state nonfinite failure, not by the now-folded legacy full-network reference replay. |

### BD114: Unit Network Pair Acceptance Target

BD114 changes the accepted activated phase-2 BE/BDF2/Newton network-pair target
from `0.2` to `1.0`.  The target is a block-max scaled step-doubling acceptance
norm, so `1.0` is the natural "within tolerance" threshold for the network-only
subcontroller.  The previous `0.2` target was an overly conservative staging
safety factor: with the BD112 `1024` cap it still rejected oversized
post-activation attempts whose pair error was below unity but above `0.2`,
forcing expensive host retries until the wall-time budget expired.  BD114 does
not change `atol`, `rtol`, the positivity line-search policy, raw candidate
telemetry, Rodas5P host control, QKE scope, or public support claims.

Measured q4/mu5 CPU-JAX/Rodas5P evidence for the same 180-second staging row:

- BD113 consolidated reference with target `0.2`:
  `/tmp/rabbit_bd113_consolidated_fullref_q4_mu5.json`,
  `artifact_payload_sha256=94ddae4adf7804ff6e1130c1a7c214c7f8a61a5b01d0f9eaf2b516f2f5d23e0f`,
  `N_final=2.7137160704223873`,
  `passed=false`,
  and violations
  `continuous_ap65_bbn_observables_unavailable`,
  `continuous_ap65_boundary_provenance_missing`,
  `continuous_ap65_final_state_nonfinite`,
  `continuous_ap65_source_payload_exception`.
- BD114 actual-code target `1.0`:
  `/tmp/rabbit_bd114_target1_actual_q4_mu5.json`,
  `artifact_payload_sha256=8c7b705587c866b0060fb7f5e45ea9effae0d2c766684c6e735f7400a65172be`,
  file SHA256
  `3491173a785674c380f562d6c0dac65799222012ad41a9d4e7b60b9f38ec1630`,
  `wall_seconds_total=178.70334624999668`,
  `passed=true`,
  `violations=[]`,
  `N_final=4.8`,
  `T_gamma_MeV=0.00914475962100043`,
  `phase2_conservative_extent_corrector_accepted_count_total=155`,
  `phase2_conservative_extent_corrector_rejected_count_total=1`,
  `phase2_conservative_extent_corrector_internal_error_target_met_count_total=127`,
  `phase2_conservative_extent_corrector_local_error_internal_norm_max=0.9887712594583102`,
  `phase2_conservative_extent_corrector_local_error_internal_max_scaled_max=1.941659386974815`,
  `phase2_conservative_extent_corrector_raw_newton_trial_negative_count_total=0`,
  `phase2_conservative_extent_window_reference_replay_N_end_max=4.8`,
  `phase2_full_network_window_reference_replay_N_end_max=4.8`,
  and `phase2_full_network_window_reference_charge_delta_abs_max=3.0258961909360453e-05`.

The endpoint-backed BBN readouts for this private q4/mu5, `n_reactions=12`
backbone row are diagnostic only: `Yp=0.16971384189564434`,
`D/H=1.8870272438777542e-05`, `N_eff_3T=2.9781565403951245`, and
`Sigma_H=0.11432002150599589`.  These values are not a public BBN claim, are not
31-reaction validation, and are not a statistical pipeline result.

Self-audit:

| field | value |
| --- | --- |
| `real_blocker_moved` | The first q4/mu5 CPU-JAX/Rodas5P private continuous AP65 row reaches `N=4.8` and `T_gamma=0.00914475962100043 MeV` with endpoint observables and no artifact violations. |
| `gate_removed_or_consolidated` | Existing BE/BDF2/Newton network subcontroller acceptance scaling changed in place; no standalone readiness, manifest, hash, figure, publication, public-dispatch, production-support, or QKE gate was added. |
| `raw_state_preserved` | Raw Rodas candidates, BDF2 one-full/two-half candidates, local-error vectors, Newton trial negativity telemetry, mass/charge residuals, and `output_truncation_applied=False` remain surfaced; no final abundance truncation is used. |
| `verification` | Red-path target propagation regression, q4/mu5 CPU-JAX/Rodas5P actual-code endpoint artifact, focused RHS/WBS suite, py_compile, sync-test-count check, review, and diff check. |
| `remaining_blocker` | Broaden from one private q4/mu5 12-reaction endpoint row to q/angular/tolerance ladders, no-collision/collision comparisons, 31-reaction checks, plot generation, and statistical pipeline inputs before any stronger claim. |

### BD115: Frozen Standard-Network Kinetic Cache

BD115 keeps the BD98 operator split and the accepted BE/BDF2/Newton activated
network subcontroller from BD105-BD114, but removes redundant frozen-background
standard-network rate and density-factor work from the Newton hot loop.  For
the standard PRIMAT-backed 9-species network path, the network substep now
precomputes the temperature-, eta-, density-, stoichiometry-, and rate-dependent
kinetic factors once per accepted-background network problem, then reuses those
factors in both the residual and analytic Jacobian evaluations.  This changes
no acceptance tolerance, no positivity line search, no raw candidate telemetry,
no Rodas5P host policy, no QKE scope, and no public support claim.

Measured q4/mu5 CPU-JAX/Rodas5P evidence for the same private 31-reaction
staging row:

- BD115 pre-cache 31-reaction baseline:
  `/tmp/rabbit_bd115_n31_q4_mu5.json`,
  `artifact_payload_sha256=dd25bf0565f026e2627a71819b95f4d00821dfb583efa5702312ad242cd15628`,
  file SHA256
  `02f87154ef7d988b8f43237bee352c9f48232b7b4ee17a2718551083e819d48b`,
  `wall_seconds_total=180.1826181249926`,
  `passed=false`,
  `N_final=2.8310100852364353`,
  `phase2_conservative_extent_corrector_step_count_total=123`,
  `phase2_conservative_extent_corrector_newton_residual_evaluation_count_total=255756`,
  `phase2_conservative_extent_corrector_newton_jacobian_evaluation_count_total=85254`,
  and `source_evaluations=1418`.
- BD115 cached 31-reaction path:
  `/tmp/rabbit_bd115_cached_n31_q4_mu5.json`,
  `artifact_payload_sha256=59f08367c0cc6093e31178a47372832ba084ee3a7293f516922b2e9121d76a91`,
  file SHA256
  `f64106736099f71a26fa6c803ab7fcf2b7592a60a6a4b8ef4d2d65948c53566c`,
  `wall_seconds_total=163.8800985190319`,
  `passed=true`,
  `violations=[]`,
  `N_final=4.8`,
  `phase2_conservative_extent_corrector_step_count_total=155`,
  `phase2_conservative_extent_corrector_newton_residual_evaluation_count_total=292068`,
  `phase2_conservative_extent_corrector_newton_jacobian_evaluation_count_total=97358`,
  and `source_evaluations=1710`.

The endpoint-backed BBN readouts for this private q4/mu5, `n_reactions=31`
diagnostic row are `Yp=0.1697138421195527`,
`D/H=1.8871137096284503e-05`, `N_eff_3T=2.9781565403951245`, and
`Sigma_H=0.11432002150599589`.  These are not a public BBN claim, not
production support, not QKE evidence, and not a statistical pipeline result.

Self-audit:

| field | value |
| --- | --- |
| `real_blocker_moved` | The same private q4/mu5 CPU-JAX/Rodas5P 31-reaction row moves from a wall-time failure at `N=2.8310100852364353` to an endpoint-backed `N=4.8` artifact within the 180-second staging budget. |
| `gate_removed_or_consolidated` | Existing BE/BDF2/Newton standard-network residual and analytic-Jacobian internals changed in place; no standalone readiness, manifest, hash, figure, publication, public-dispatch, production-support, or QKE gate was added. |
| `raw_state_preserved` | Raw Rodas candidates, BDF2 one-full/two-half candidates, local-error vectors, Newton trial negativity telemetry, mass/charge residuals, and `output_truncation_applied=False` remain surfaced; no final abundance truncation is used. |
| `verification` | Cached-vs-uncached residual/Jacobian regression, q4/mu5 CPU-JAX/Rodas5P 31-reaction endpoint artifact, focused RHS/WBS suite, py_compile, sync-test-count check, review, and diff check. |
| `remaining_blocker` | Broaden from one private q4/mu5 31-reaction endpoint row to no-collision/collision comparisons, q/angular/tolerance ladders, plot generation, and statistical pipeline inputs before any stronger claim. |

### BD116: Network-Pair-Scoped Kinetic Cache

BD116 lifts the BD115 frozen standard-network kinetic cache from individual
coarse/refined network step attempts to the whole BE/BDF2/Newton adaptive
network pair.  The full-vs-half step-doubling comparison is still performed
under one frozen accepted background, so sharing the same precomputed
temperature-, eta-, density-, stoichiometry-, and rate-dependent factors across
coarse, refined, and jumped retry attempts is the intended operator-split
semantics.  The accepted candidate, tolerance, positivity line search, raw
Newton trial telemetry, Rodas5P host policy, QKE scope, and public-support
boundary are unchanged.

Micro-regression evidence:

- `test_bd116_network_pair_shares_kinetic_cache_across_attempts` runs a
  standard-network BDF2 adaptive pair with one coarse and one refined attempt,
  verifies both attempts report the cache, and asserts
  `evaluate_nuclear_rates(...)` is called exactly once for the whole pair.

Measured q4/mu5 CPU-JAX/Rodas5P 31-reaction endpoint check:

- BD115 step-attempt-scoped cache:
  `/tmp/rabbit_bd115_cached_n31_q4_mu5.json`,
  `artifact_payload_sha256=59f08367c0cc6093e31178a47372832ba084ee3a7293f516922b2e9121d76a91`,
  `wall_seconds_total=163.8800985190319`,
  `passed=true`, `violations=[]`, and `N_final=4.8`.
- BD116 pair-scoped cache:
  `/tmp/rabbit_bd116_paircache_n31_q4_mu5.json`,
  `artifact_payload_sha256=a093111eb3dd3d18bf2beb9780cd2e2b6ae3626a45326f732b0224f3a60a4c51`,
  file SHA256
  `8c6074be7b62f021e71e9b786a66ca29701daeb97e7cd8a55de925c6af2dfe57`,
  `wall_seconds_total=164.2399338139221`,
  `passed=true`,
  `violations=[]`,
  `N_final=4.8`,
  `phase2_conservative_extent_corrector_internal_attempt_count_total=495`,
  `phase2_conservative_extent_corrector_newton_residual_evaluation_count_total=292068`,
  and
  `phase2_conservative_extent_corrector_newton_jacobian_evaluation_count_total=97358`.

The endpoint-backed BBN readouts remain the same private q4/mu5,
`n_reactions=31` diagnostic values: `Yp=0.1697138421195527`,
`D/H=1.8871137096284503e-05`, `N_eff_3T=2.9781565403951245`, and
`Sigma_H=0.11432002150599589`.  The wall-time comparison is effectively
neutral, so BD116 should be read as a hot-loop cleanup that removes repeated
rate evaluation work, not as a full-run speed breakthrough.  The measured
dominant cost remains the BDF2/Newton residual/Jacobian/linear-solve volume.

Self-audit:

| field | value |
| --- | --- |
| `real_blocker_moved` | Repeated standard-network rate evaluation is now eliminated across coarse/refined/jumped attempts inside one frozen-background network adaptive pair; this is verified directly by a rate-call-count regression. |
| `gate_removed_or_consolidated` | Existing BE/BDF2/Newton standard-network cache internals changed in place; no standalone readiness, manifest, hash, figure, publication, public-dispatch, production-support, or QKE gate was added. |
| `raw_state_preserved` | Raw Rodas candidates, BDF2 one-full/two-half candidates, local-error vectors, Newton trial negativity telemetry, mass/charge residuals, and `output_truncation_applied=False` remain surfaced; no final abundance truncation is used. |
| `verification` | Pair-scope rate-call regression, q4/mu5 CPU-JAX/Rodas5P 31-reaction endpoint artifact, focused RHS/WBS suite, py_compile, sync-test-count check, review, and diff check. |
| `remaining_blocker` | The full-run hotspot is still BDF2/Newton residual/Jacobian/linear-solve volume, plus the broader no-collision/collision comparison, q/angular/tolerance ladder, plot-input, and statistical-pipeline evidence gaps. |

### BD117: Reuse Accepted Newton Line-Search Residuals

BD117 removes a duplicate residual evaluation in the coupled BE/BDF2/Newton
network solve.  Before BD117, each accepted Newton line-search trial evaluated
the residual once to decide whether the trial reduced the norm, then the next
Newton iteration immediately recomputed the same residual at the same accepted
state.  The solver now carries the accepted trial residual/payload/norm into
the next iteration.  This changes no Newton update, no line-search positivity
policy, no residual tolerance, no network cache, no Rodas5P host policy, no raw
negative telemetry, no QKE scope, and no public-support claim.

Micro-regression evidence:

- `test_bd117_newton_reuses_accepted_line_search_residual` checks a coupled
  BE/Newton burn solve and asserts the residual evaluation count is the
  non-duplicated formula
  `1 + finite_difference_residual_evaluation_count + linear_solve_count`.
- `test_bd116_network_pair_shares_kinetic_cache_across_attempts` now also
  asserts that the BDF2 adaptive-pair payload reports nonzero line-search
  residual reuse.

Measured q4/mu5 CPU-JAX/Rodas5P 31-reaction endpoint evidence:

- No-collision pre-BD117 comparison row:
  `/tmp/rabbit_bd117_nocollision_n31_q4_mu5.json`,
  `artifact_payload_sha256=7e67fff47fc9ae3f50b5dea44bb1c3c3509c1c2b29505c7efb643ba3800ad12f`,
  file SHA256
  `d6b849933196d76873105833582dd9934e4f8df8786e9480f83c80f07bcdbf97`,
  `passed=true`, `violations=[]`, `N_final=4.8`,
  `wall_seconds_total=103.17029682197608`, and
  `phase2_conservative_extent_corrector_newton_residual_evaluation_count_total=292066`.
- No-collision BD117 row:
  `/tmp/rabbit_bd117_resreuse_nocollision_n31_q4_mu5.json`,
  `artifact_payload_sha256=987499e09d652232473fc9dc2a1aa4a503189c649203479c6a36ac5aba89fbbe`,
  file SHA256
  `95a2d446632b15ddc6439c6674317778e12cf92e4ba4e77cf5ca2815e1052070`,
  `passed=true`, `violations=[]`, `N_final=4.8`,
  `wall_seconds_total=94.74744493700564`, and
  `phase2_conservative_extent_corrector_newton_residual_evaluation_count_total=194709`.
- Dynamic-collision BD116 row:
  `/tmp/rabbit_bd116_paircache_n31_q4_mu5.json`,
  `artifact_payload_sha256=a093111eb3dd3d18bf2beb9780cd2e2b6ae3626a45326f732b0224f3a60a4c51`,
  file SHA256
  `8c6074be7b62f021e71e9b786a66ca29701daeb97e7cd8a55de925c6af2dfe57`,
  `passed=true`, `violations=[]`, `N_final=4.8`,
  `wall_seconds_total=164.2399338139221`, and
  `phase2_conservative_extent_corrector_newton_residual_evaluation_count_total=292068`.
- Dynamic-collision BD117 row:
  `/tmp/rabbit_bd117_resreuse_dynamic_n31_q4_mu5.json`,
  `artifact_payload_sha256=459543b0e69cffcf939717c0addfba7c300debda451d646cf6787ce86bd2ff73`,
  file SHA256
  `68d85fb1aec3d5cb573535d541542928acec7ff4049aaf72610c5f13a071ffe1`,
  `passed=true`, `violations=[]`, `N_final=4.8`,
  `wall_seconds_total=154.66314127494115`, and
  `phase2_conservative_extent_corrector_newton_residual_evaluation_count_total=194710`.

The residual-evaluation reduction is `97357` in the no-collision row and
`97358` in the dynamic-collision row, about one third of the previous network
Newton residual evaluations.  The endpoint readouts remain stable within the
same run family.  The BD117 no-collision row gives `Yp=0.16972783156110663`,
`D/H=1.8904206719153318e-05`, `N_eff_3T=2.993397912593453`, and
`Sigma_H=0.11432002114948502`; the BD117 dynamic-collision row gives
`Yp=0.1697138421195527`, `D/H=1.8871137096284503e-05`,
`N_eff_3T=2.9781565403951245`, and `Sigma_H=0.11432002150599589`.  The
dynamic-minus-no-collision endpoint deltas for this private row are
`Delta Yp=-1.3989441553924342e-05`, `Delta D/H=-3.306962286881511e-08`,
`Delta N_eff_3T=-0.015241372198328662`, and
`Delta Sigma_H=3.5651087637589285e-10`.

Self-audit:

| field | value |
| --- | --- |
| `real_blocker_moved` | The coupled network Newton hot loop no longer recomputes accepted line-search residuals; measured 31-reaction q4/mu5 no-collision and dynamic-collision rows reduce network residual evaluations by about one third while preserving endpoint success. |
| `gate_removed_or_consolidated` | Existing BE/BDF2/Newton residual reuse internals changed in place; no standalone readiness, manifest, hash, figure, publication, public-dispatch, production-support, or QKE gate was added. |
| `raw_state_preserved` | Raw Rodas candidates, BDF2 one-full/two-half candidates, local-error vectors, Newton trial negativity telemetry, mass/charge residuals, and `output_truncation_applied=False` remain surfaced; no final abundance truncation is used. |
| `verification` | Newton residual-reuse regression, pair-payload telemetry regression, q4/mu5 CPU-JAX/Rodas5P no-collision and dynamic-collision 31-reaction endpoint artifacts, focused RHS/WBS suite, py_compile, sync-test-count check, review, and diff check. |
| `remaining_blocker` | Dynamic collision remains about 60 seconds slower than no-collision for this row because current-state collision payload builds/reuse still add real cost; broader q/angular/tolerance ladders, plot inputs, and statistical pipeline artifacts remain unpromoted. |

### BD118: Vectorized Analytic Network Jacobian Assembly

BD118 keeps the BD98 operator split and the accepted BE/BDF2/Newton activated
network subcontroller unchanged, but removes Python scalar-loop work from the
standard-network analytic Jacobian.  Forward directional mass-action
derivatives now use indexed vector accumulation, and reverse
photodissociation derivatives use the positive-state identity
`d(prod_i Y_i**nu_i)/dY_j = prod_i Y_i**nu_i * nu_j / Y_j`.  The previous
reverse loop remains as the fallback for nonpositive trial states.  This
changes no Newton update, no BDF2 pair target, no positivity line-search
policy, no kinetic cache semantics, no Rodas5P host policy, no raw negative
telemetry, no QKE scope, and no public-support claim.

Micro-regression evidence:

- `test_bd118_full_standard_network_analytic_jacobian_matches_finite_difference`
  checks the full 31-reaction standard-network analytic Jacobian against the
  finite-difference residual Jacobian on the activation sample.
- The existing BD106/BD115-BD117 focused subset still covers the smaller
  12-reaction activation sample, kinetic-cache sharing, legacy hook
  compatibility, and Newton residual reuse.

Measured q4/mu5 CPU-JAX/Rodas5P 31-reaction endpoint evidence:

- No-collision BD118 row:
  `/tmp/rabbit_bd118_vecjac_nocollision_n31_q4_mu5.json`,
  `artifact_payload_sha256=d542413e503e1236a160b1d92b7277043ab801c2688750a1dd853edead0d1a7c`,
  file SHA256
  `46081b5c344e4244647ff5e5d135b4cf93d8735726aaf6c1f5cfc2f8a6d80a02`,
  `passed=true`, `violations=[]`, `N_final=4.8`,
  `wall_seconds_total=69.74548954004422`,
  `phase2_conservative_extent_corrector_newton_residual_evaluation_count_total=194709`,
  and
  `phase2_conservative_extent_corrector_newton_jacobian_evaluation_count_total=97357`.
- Dynamic-collision BD118 row:
  `/tmp/rabbit_bd118_vecjac_dynamic_n31_q4_mu5.json`,
  `artifact_payload_sha256=49273ad03bf195881c8dd32ad41ac2cb4ee40c8bbf79979fb84edbf9f4f2e14d`,
  file SHA256
  `7625b204bada0d1bab0b5bba2534a59e03700b58d20fbfd20fac99a586f5d4f8`,
  `passed=true`, `violations=[]`, `N_final=4.8`,
  `wall_seconds_total=130.993984925095`,
  `phase2_conservative_extent_corrector_newton_residual_evaluation_count_total=194710`,
  and
  `phase2_conservative_extent_corrector_newton_jacobian_evaluation_count_total=97358`.

Relative to BD117, the no-collision row wall time drops from
`94.74744493700564` to `69.74548954004422`, and the dynamic-collision row wall
time drops from `154.66314127494115` to `130.993984925095`, with the same
Newton residual/Jacobian/linear-solve counts.  Endpoint readouts remain stable:
BD118 no-collision gives `Yp=0.16972783156110663`,
`D/H=1.8904206719153318e-05`, `N_eff_3T=2.993397912593453`, and
`Sigma_H=0.11432002114948502`; BD118 dynamic-collision gives
`Yp=0.1697138421195527`, `D/H=1.8871137096284503e-05`,
`N_eff_3T=2.9781565403951245`, and `Sigma_H=0.11432002150599589`.

Self-audit:

| field | value |
| --- | --- |
| `real_blocker_moved` | The coupled network Newton hot loop no longer spends scalar Python loops assembling the standard-network analytic Jacobian; measured 31-reaction q4/mu5 no-collision and dynamic-collision endpoint rows reduce wall time while preserving endpoint success and Newton counts. |
| `gate_removed_or_consolidated` | Existing BE/BDF2/Newton analytic-Jacobian internals changed in place; no standalone readiness, manifest, hash, figure, publication, public-dispatch, production-support, or QKE gate was added. |
| `raw_state_preserved` | Raw Rodas candidates, BDF2 one-full/two-half candidates, local-error vectors, Newton trial negativity telemetry, mass/charge residuals, and `output_truncation_applied=False` remain surfaced; no final abundance truncation is used. |
| `verification` | Full 31-reaction analytic-vs-finite-difference Jacobian regression, q4/mu5 CPU-JAX/Rodas5P no-collision and dynamic-collision endpoint artifacts, focused RHS/WBS suite, py_compile, sync-test-count check, review, and diff check. |
| `remaining_blocker` | Dynamic collision still carries real current-state source/payload cost, and the broader q/angular/tolerance ladder, plot-input, and statistical-pipeline evidence remain unpromoted. |

### BD119: Full Standard-Network Defaults For Continuous Endpoint Surfaces

BD119 changes the private continuous-AP65 endpoint surfaces from the old
12-reaction backbone default to the full 31-reaction in-tree standard network
default.  This applies to the FB69 source-RHS prototype builder/CLI, the FB70
full-BBN span-ladder builder/CLI, and the consolidated span-experiment CLI.
Explicit `n_reactions=12` remains supported for old smoke/backbone comparisons,
but new default endpoint probes no longer silently run the reduced backbone
while being discussed as full-network evidence.

This is not a public promotion: the surfaces still emit private claim scopes,
`qke_scope=out_of_scope`, `public_dispatch_ready=false`, and
`production_smc_validation_ready=false`.  It also does not add a new
standalone gate; it changes existing execution defaults after BD115-BD118
showed that the 31-reaction q4/mu5 CPU-JAX/Rodas5P endpoint row reaches
`N_final=4.8` for both no-collision and dynamic-collision probes.

Focused default-surface checks:

- `scripts/run_augmented_continuous_ap65_source_rhs_prototype.py --dry-run
  --skip-reference` reports `inputs.n_reactions=31` with
  `claim_scope=private_continuous_ap65_rhs_prototype_only`,
  `public_dispatch_ready=false`, and `qke_scope=out_of_scope`.
- `scripts/run_augmented_continuous_ap65_full_bbn_span_ladder.py --dry-run`
  reports `inputs.n_reactions=31` with
  `claim_scope=private_continuous_ap65_full_bbn_span_ladder_only`,
  `public_dispatch_ready=false`, and `qke_scope=out_of_scope`.
- CLI regressions now assert the default 31-reaction network on the FB69 and
  consolidated span surfaces.

Self-audit:

| field | value |
| --- | --- |
| `real_blocker_moved` | Default continuous endpoint runs now use the full 31-reaction standard network that BD115-BD118 measured at endpoint, instead of accidentally producing new backbone-network rows unless the caller opts into `n_reactions=12`. |
| `gate_removed_or_consolidated` | Existing FB69/FB70/span CLI and builder defaults changed in place; no standalone readiness, manifest, hash, figure, publication, public-dispatch, production-support, or QKE gate was added. |
| `raw_state_preserved` | No solver state, positivity policy, raw candidate telemetry, BDF2/Newton pair evidence, mass/charge residual, or `output_truncation_applied=False` behavior changes. |
| `verification` | FB69/FB70 dry-run default checks, FB69 and consolidated span CLI regressions, focused RHS/WBS suite, py_compile, sync-test-count check, review, and diff check. |
| `remaining_blocker` | Default scope is fixed, but q/angular/tolerance ladder evidence, plot inputs, statistical-pipeline wiring, and dynamic current-state source/payload cost remain open. |

### BD120: Endpoint Resolution Ladder Observable Deltas

BD120 extends the existing FB70 resolution/tolerance ladder so endpoint
comparisons can use the augmented observables that are now emitted by the
31-reaction BE/BDF2/Newton rows: `Sigma_H` and `N_eff_3T`.  This changes the
current resolution ladder in place; it does not add a standalone readiness,
manifest, hash, figure, public-dispatch, production-support, or QKE gate.  The
old `Yp`, `D/H`, and `T_final_MeV` comparisons remain unchanged, and the
default tolerance set remains backward-compatible unless a caller explicitly
requests the new observable tolerances.

The first no-collision q4/mu5 31-reaction h-ladder with explicit
`Sigma_H`/`N_eff_3T` tolerances reached the endpoint in both rows but failed
the strict abundance-delta tolerance:

- Artifact:
  `/tmp/rabbit_bd120_h_ladder_nocollision_n31_q4_mu5.json`,
  `artifact_payload_sha256=3ba5b11a0fbca767d1c2a3864e280c9cff8240fc1d45701d809ce7e09fe19dda`,
  file SHA256
  `bff0d14f43737f232150c2770e55a4d4c6f597865807e26a1cf55258d10adf80`,
  `passed=false`, `physical_full_bbn_span_ready=true`,
  `violations=["resolution_ladder_terminal_delta_tolerance_failed"]`.
- `h_max=0.1` row:
  `T_final_MeV=0.009139194142233497`,
  `Yp=0.16972783156110663`, `D/H=1.8904206719153318e-05`,
  `Sigma_H=0.11432002114948502`, `N_eff_3T=2.993397912593453`,
  `selected_step_count_total=155`, and
  `selected_wall_seconds_total=71.92629868595395`.
- `h_max=0.05` row:
  `T_final_MeV=0.00913919407430807`,
  `Yp=0.16773888104198983`, `D/H=1.9916578619100963e-05`,
  `Sigma_H=0.11432002106208976`, `N_eff_3T=2.99339813347606`,
  `selected_step_count_total=211`, and
  `selected_wall_seconds_total=78.09296064195223`.
- Adjacent deltas:
  `abs_delta_Yp=0.0019889505191167944`,
  `abs_delta_DH=1.0123718999476457e-06`,
  `abs_delta_T_final_MeV=6.792542743550012e-11`,
  `abs_delta_Sigma_H=8.739525769740908e-11`, and
  `abs_delta_N_eff_3T=2.2088260687169736e-07`.

The useful conclusion is narrow: the BE/BDF2/Newton full-network path reaches
the private full-BBN endpoint for both h values, and background/thermo
observables are stable under this h refinement, but `Yp` and `D/H` are not yet
converged to the strict ladder tolerances.  This points the next solver work
at abundance accuracy across the activated network subcontroller and
operator-split background treatment, not at the fixed `0.08 MeV` threshold or
background endpoint drift for this no-collision row.

Self-audit:

| field | value |
| --- | --- |
| `real_blocker_moved` | Existing endpoint resolution ladders can now compare `Sigma_H` and `N_eff_3T` in addition to `Yp`, `D/H`, and `T_final_MeV`, and a real q4/mu5 no-collision h-ladder exposes that abundance convergence, not background endpoint drift, is the current failing tolerance. |
| `gate_removed_or_consolidated` | Existing FB70 resolution-ladder comparison support changed in place; no standalone readiness, manifest, hash, figure, publication, public-dispatch, production-support, or QKE gate was added. |
| `raw_state_preserved` | Raw child span rows, terminal observables, adjacent deltas, phase-2 diagnostic deltas, failed-tolerance fields, and `output_truncation_applied=false` row policies remain embedded; no final abundance repair or display truncation was introduced. |
| `verification` | Red-first resolution-ladder/CLI tests, full resolution-ladder focused slice, real q4/mu5 CPU-JAX/Rodas5P no-collision 31-reaction h-ladder artifact, focused FB70/WBS suite, py_compile, sync-test-count check, review, and diff check. |
| `remaining_blocker` | Endpoint abundance convergence under h/tolerance refinement remains open: `Yp` and `D/H` move by `1.99e-3` and `1.01e-6` between `h_max=0.1` and `0.05`; dynamic-collision cost and broader q/angular/freedom ladders also remain open. |

### BD121: Weak-Only Hot-Phase Step-Averaged Rates

BD121 changes the existing private phase-2 corrector path in place so the
pre-activation weak-only hot phase no longer evolves `n <-> p` with only the
host-step start weak-rate sample.  When both start and raw candidate
accepted-background weak-rate payloads are finite, the exact two-state weak
update now uses trapezoid-averaged per-`N` coefficients:
`0.5 * (lambda/H)_start + 0.5 * (lambda/H)_candidate`.  If the candidate weak
payload or candidate Hubble rate is unavailable, the payload records a
`start_only_candidate_unavailable` policy and preserves the old start-only
behavior.  The activated network branch remains the existing network-only
BDF2/Newton subcontroller; no phase-2 burn variable is reintroduced into the
host Rodas stage algebra.

The same q4/mu5 no-collision 31-reaction h-ladder used in BD120 reached the
endpoint in both rows after this change:

- Artifact:
  `/tmp/rabbit_bd121_full_h_ladder_nocollision_n31_q4_mu5.json`,
  `artifact_payload_sha256=598c6f449505cf070fd15cc90f3c21e8e3a77010a6486fb9cb5e73e7f4013342`,
  file SHA256
  `6afcdc4edf116dc8ffca257e4a719f295615162b3c4f841f5aee70a76614c4f7`,
  `passed=false`, `physical_full_bbn_span_ready=true`,
  `violations=["resolution_ladder_terminal_delta_tolerance_failed"]`.
- `h_max=0.1` row:
  `T_final_MeV=0.009139194142233497`,
  `Yp=0.16327817895368615`, `D/H=1.8648952626716096e-05`,
  `Sigma_H=0.11432002114948502`, `N_eff_3T=2.993397912593453`,
  `selected_step_count_total=155`, and
  `selected_wall_seconds_total=71.56073158606887`.
- `h_max=0.05` row:
  `T_final_MeV=0.009139194074258984`,
  `Yp=0.16313718642544403`, `D/H=1.9729450411965964e-05`,
  `Sigma_H=0.11432002106200953`, `N_eff_3T=2.9933981336732316`,
  `selected_step_count_total=207`, and
  `selected_wall_seconds_total=76.07932914199773`.
- Adjacent deltas:
  `abs_delta_Yp=0.00014099252824212316`,
  `abs_delta_DH=1.0804977852498688e-06`,
  `abs_delta_T_final_MeV=6.797451317097636e-11`,
  `abs_delta_Sigma_H=8.747548518872605e-11`, and
  `abs_delta_N_eff_3T=2.210797784840679e-07`.

This materially narrows the BD120 abundance blocker: the weak-only rate
freezing error was the dominant source of the previous `Yp` h-sensitivity, and
`Yp` now satisfies the configured `5e-4` adjacent tolerance.  `D/H` remains
outside the strict `5e-8` tolerance, so the next solver work should target the
activated deuterium burn/network subcontroller or operator-split background
treatment rather than another weak-only freeze-out patch.

Self-audit:

| field | value |
| --- | --- |
| `real_blocker_moved` | The BD120 `Yp` h-delta fell from `1.9889505191167944e-3` to `1.4099252824212316e-4` on the same q4/mu5 no-collision endpoint ladder, while endpoint/background deltas stayed stable. |
| `gate_removed_or_consolidated` | Existing FB69 phase-2 corrector internals changed in place; no standalone readiness, manifest, hash, figure, publication, public-dispatch, production-support, or QKE gate was added. |
| `raw_state_preserved` | Raw Rodas candidates, raw phase-2 corrector samples, start/candidate weak payloads, per-`N` weak coefficients, mass/charge residuals, and no-output-truncation policies remain embedded. |
| `verification` | Red-first weak-only hot-phase rate-average regression, focused BD99/BD101/BD105/BD110/BD111/BD121 RHS slice, real q4/mu5 CPU-JAX/Rodas5P no-collision 31-reaction h-ladder artifact, py_compile, sync-test-count check, review, and diff check. |
| `remaining_blocker` | Endpoint `D/H` h-convergence remains open: the same ladder still has `abs_delta_DH=1.0804977852498688e-6` against the configured `5e-8` tolerance; dynamic-collision cost and broader q/angular/freedom ladders also remain open. |

### BD122: Activated Network Step-Averaged Background

BD122 changes the existing activated phase-2 network corrector path in place.
The network-only BDF2/Newton subcontroller still owns the burn update and the
phase-2 block remains removed from the host Rodas stage algebra, but the frozen
background used by each accepted network correction is now compressed from both
the host-step start and raw candidate:

- `T_gamma` uses a positive geometric midpoint.
- The nuclear `dN/H` factor uses a harmonic effective Hubble rate.
- Weak `n <-> p` terms use the same start/candidate trapezoid per-`N`
  coefficients introduced in BD121, mapped back through the effective Hubble
  rate passed to the network solver.
- If any candidate quantity is unavailable, the payload records the fallback
  policy and keeps the previous start-only behavior for that component.

The same q4/mu5 no-collision 31-reaction h-ladder used in BD121 now passes the
strict endpoint resolution tolerances:

- Artifact:
  `/tmp/rabbit_bd122_full_h_ladder_nocollision_n31_q4_mu5.json`,
  `artifact_payload_sha256=0a17d0f614d16b34919c8a7fda1ce88f6e8ddf2e9efdd55f386ec5fdf9327216`,
  file SHA256
  `e25ef34d3d73b73bffd503fad0d05abd6372c60333fd70818c7eaa9664114dc2`,
  `passed=true`, `physical_full_bbn_span_ready=true`, and `violations=[]`.
- `h_max=0.1` row:
  `T_final_MeV=0.009139193906524163`,
  `Yp=0.1633988563067751`, `D/H=2.0938002394713553e-05`,
  `Sigma_H=0.1188007682257251`, `N_eff_3T=2.9933985593496453`,
  `selected_step_count_total=169`, and
  `selected_wall_seconds_total=66.98154121695552`.
- `h_max=0.05` row:
  `T_final_MeV=0.009139193846884783`,
  `Yp=0.16319714057186888`, `D/H=2.0969952095959142e-05`,
  `Sigma_H=0.11880076804530677`, `N_eff_3T=2.9933987567053344`,
  `selected_step_count_total=216`, and
  `selected_wall_seconds_total=62.05258181807585`.
- Adjacent deltas:
  `abs_delta_Yp=0.00020171573490621042`,
  `abs_delta_DH=3.194970124558984e-08`,
  `abs_delta_T_final_MeV=5.963938023989535e-11`,
  `abs_delta_Sigma_H=1.8041833305115773e-10`, and
  `abs_delta_N_eff_3T=1.9735568912437884e-07`.

Compared with BD121, the `D/H` h-delta fell from
`1.0804977852498688e-6` to `3.194970124558984e-8`, crossing the configured
`5e-8` tolerance while preserving endpoint/background stability.  This is the
first q4/mu5 no-collision 31-reaction CPU-JAX/Rodas5P endpoint ladder in this
line to satisfy the configured `Yp` and `D/H` adjacent h-ladder tolerances.

Self-audit:

| field | value |
| --- | --- |
| `real_blocker_moved` | The BD121 `D/H` h-delta fell from `1.0804977852498688e-6` to `3.194970124558984e-8` on the same q4/mu5 no-collision 31-reaction endpoint ladder, making the existing strict resolution ladder pass. |
| `gate_removed_or_consolidated` | Existing FB69 phase-2 corrector internals and existing FB70 resolution-ladder evidence changed in place; no standalone readiness, manifest, hash, figure, publication, public-dispatch, production-support, or QKE gate was added. |
| `raw_state_preserved` | Raw Rodas candidates, phase-2 local errors, mass/charge residuals, and `output_truncation_applied=false` artifact policies remain embedded; the start/candidate/effective background contract is recorded by the corrector payload and locked by focused regression. |
| `verification` | Red-first activated-network background regression, fallback-policy regression, focused RHS/WBS suite, real q4/mu5 CPU-JAX/Rodas5P no-collision 31-reaction h-ladder artifact, py_compile, sync-test-count check, review, and diff check. |
| `remaining_blocker` | Carry this endpoint-resolution result into broader q/angular/freedom matrices, then revisit dynamic-collision/non-LRS payload cost and full freedom composition; this is still private staging evidence, not public production support. |

### BD123: Activated Network Substep-Varying Background

BD123 changes the same existing FB69 activated phase-2 corrector path in place
by adding an explicit activated-network background policy.  The default
`effective_midpoint` policy preserves BD122's start/candidate compressed
sample.  The opt-in `substep_loglinear_midpoint` policy feeds midpoint
background nodes into the network-only BE/BDF2/Newton subcontroller so BD122's
one-background approximation can be stress-tested without replacing the
CPU-JAX/Rodas5P host solver or moving `X_phase2` back into the host stage
algebra.  When the opt-in policy is selected:

- `T_gamma` is log-linearly interpolated from the host-step start and raw
  candidate values at substep midpoint `theta=(k+1/2)/m`.
- The nuclear `dN/H` factor uses the corresponding linear-in-`1/H` midpoint.
- Weak `n <-> p` terms linearly interpolate BD121 per-`N` weak coefficients
  and map them back to per-second lambdas with the substep Hubble rate.
- Standard-network kinetic factors are cached per substep background node and
  passed directly into the BDF2/Newton residual/Jacobian path.

This directly targets the external-audit concern that BD122's one-full/two-half
network error was mostly estimating an autonomous frozen-background network
problem, not the nonautonomous operator-split error from changing
`T_gamma`, `H`, and weak rates.  The change is an executable solver-control
path inside the existing FB69/FB70 runtime, not a new gate, figure, manifest,
hash relay, public dispatch path, production-support claim, or QKE path.

Self-audit:

| field | value |
| --- | --- |
| `real_blocker_moved` | The existing activated-network corrector and FB70 CLI now expose a runtime policy that can compare BD122 effective-midpoint updates against substep-varying background-node updates, directly attacking the operator-split background-compression blocker without changing the host Rodas5P solver. |
| `gate_removed_or_consolidated` | Existing FB69 corrector and BE/BDF2/Newton subcontroller internals changed in place; no standalone readiness, manifest, hash, figure, publication, public-dispatch, production-support, or QKE gate was added. |
| `raw_state_preserved` | Raw Rodas candidates, BDF2 one-full/two-half candidates, local errors, Newton trial negativity, mass/charge residuals, and `output_truncation_applied=false` policies remain preserved. |
| `verification` | Red-first BD123 regressions lock substep-node routing into BDF2/Newton residuals, the opt-in activated candidate background factory, default effective-midpoint preservation, and FB70 CLI/builder pass-through; bounded q4/mu5 31-reaction no-collision activation-window smokes passed for default `effective_midpoint` to `T_gamma=0.07653738276516263 MeV` and opt-in `substep_loglinear_midpoint` to `T_gamma=0.0796750449846388 MeV`, both still hot-endpoint private diagnostics. |
| `remaining_blocker` | Quantify effective-midpoint versus `substep_loglinear_midpoint` on no-collision, dynamic-collision, and non-LRS rows; if the opt-in policy keeps causing tiny host steps near activation, add Rodas dense/stage background or AB-predictor Newton initialization before promoting it beyond stress/comparison mode. |

### BD124: AB2 Predictor Newton Initializer

BD124 keeps the BD98 operator split, the CPU-JAX/Rodas5P host step, and the
accepted BE/BDF2/Newton activated-network corrector.  It adds an opt-in
`phase2_network_newton_initial_guess_policy=ab2_rhs_predictor` that uses an
AB2 RHS predictor only as the Newton initial guess on post-startup BDF2
network substeps.  The default `current_state` policy preserves BD123 behavior.

This follows the external-review recommendation to reuse predictor information
without promoting ABM to the accepted stiff network solver.  Accepted values
still come from the coupled implicit network solve and its step-doubling
physical-X error estimate.  Raw AB2 predictor negativity is preserved in the
payload and rejected before the Newton initial guess falls back to the current
state; no output abundance or observable is truncated.  Nonnegative AB2 guesses
must also pass a displacement guard and residual preflight, so AB2 cannot
silently move the accepted update away from the existing BE/BDF2/Newton solve.

Self-audit:

| field | value |
| --- | --- |
| `real_blocker_moved` | The existing activated-network corrector now exposes an opt-in, displacement-guarded AB2 Newton initializer that can test initial-residual/line-search stress without changing the host Rodas5P solver or accepted BE/BDF2/Newton update. |
| `gate_removed_or_consolidated` | Existing FB69 corrector, BE/BDF2/Newton adaptive pair, and FB70 runtime parameter plumbing changed in place; no standalone readiness, manifest, hash, figure, publication, public-dispatch, production-support, or QKE gate was added. |
| `raw_state_preserved` | Raw Rodas candidates, BDF2 one-full/two-half candidates, local errors, Newton trial negativity, AB2 predictor negative values, mass/charge residuals, and `output_truncation_applied=false` policies remain preserved. |
| `verification` | Red-first BD124 regressions lock AB2 use as a Newton initializer, raw-negative predictor rejection, residual/displacement preflight rejection, conservative-candidate pass-through, and FB70 pass-through; focused RHS/FB70/WBS tests, py_compile, sync-test-count check, review, and diff check are required before commit. |
| `runtime_evidence` | q4/mu5 no-collision activation smoke with `ab2_rhs_predictor` now completes to `T_final_MeV=0.07653738338285641` with 45 accepted steps and 14 rejected attempts; 5431 AB2 guesses were used and 2492 were rejected by displacement guard. It was not faster than the `current_state` control on this smoke, so it remains opt-in. |
| `remaining_blocker` | Measure whether `ab2_rhs_predictor` helps dynamic-collision/non-LRS activation rows; endpoint below `0.01 MeV`, dynamic-collision/non-LRS cost, plot inputs, and statistical-pipeline evidence remain unresolved. |

### BD125: Dynamic-Collision Background Policy Resolution

BD125 keeps the BD98 operator split, the CPU-JAX/Rodas5P host step, and the
accepted BE/BDF2/Newton activated-network corrector.  It adds the requested
policy `phase2_network_background_policy=auto_dynamic_effective_midpoint` and
records its effective runtime policy.  The auto policy resolves to
`effective_midpoint` when dynamic AP65 collision payloads are active, and to
`substep_loglinear_midpoint` otherwise.

This is not another readiness/manifest/claim gate.  It changes the executable
activated-network background policy selected inside the existing FB69/FB70
runtime after BD125 probes showed that `substep_loglinear_midpoint` is not yet
a safe dynamic-collision default.  In the q4/mu5 31-reaction dynamic-collision
activation window, `substep_loglinear_midpoint` stalled near `N=2.6149` even
with `max_steps=96`, while `effective_midpoint` plus
`stage_collision_payload_policy=auto_small_collision_reuse` passed to
`T_final_MeV=0.07657924201846118` with 46 dynamic payload builds and 413
payload reuses.  The landed auto-policy smoke reproduced that result with
`artifact_payload_sha256=f7dff35b4f4caaab83653b727ce3ce9be7688c2655f8a9d973f46e98f690beac`
and `selected_wall_seconds_total=25.1181877059862`.  The explicit substep
policy remains available for operator-split stress comparisons.

Self-audit:

| field | value |
| --- | --- |
| `real_blocker_moved` | Dynamic-collision activated-network rows can request a single auto policy that avoids the measured substep-background max-step collapse while preserving the no-collision substep-background comparison path. |
| `gate_removed_or_consolidated` | Existing FB69/FB70 runtime policy plumbing changed in place; no standalone readiness, manifest, hash, figure, publication, public-dispatch, production-support, or QKE gate was added. |
| `raw_state_preserved` | Raw Rodas candidates, network one-full/two-half candidates, local errors, Newton trial negativity, mass/charge residuals, and `output_truncation_applied=false` policies remain preserved. |
| `verification` | Red-first BD125 regressions lock dynamic/non-dynamic auto-policy resolution, FB70 metadata propagation, and CLI dry-run acceptance; focused RHS/FB70/WBS tests, py_compile, sync-test-count check, review, and diff check are required before commit. |
| `remaining_blocker` | Full endpoint below `0.01 MeV`, dynamic-collision/non-LRS resolution evidence, plot generation inputs, and statistical-pipeline evidence remain unresolved; BD125 does not claim public production support. |

### BD126: Chain Max-Steps Retry Budget Handoff

BD126 keeps the BD98 operator split, the CPU-JAX/Rodas5P host step, and the
accepted BE/BDF2/Newton activated-network corrector.  It adds the opt-in FB70
chain policy `chain_max_steps_policy=recovered_max_steps_floor`, which carries
a recovered max-step retry cap forward as the floor for later chained windows.

This is not another readiness/manifest/claim gate.  It changes the executable
chain runtime policy after BD125 follow-up probes showed that the dynamic
all-freedom q4/mu5 row to `N_span_end=3.2` can fail at `max_steps=160` and then
pass under `max_step_retry_factors=1.0,4.0`.  The recovered run reached
`T_final_MeV=0.045260353575832024` with
`artifact_payload_sha256=92d91d35e22d28372547d1fbd8824156bb9d0770156bb3871f8f7a208b2aebb1`,
`max_step_retry_rows_recovered=1`, selected trace-domain rejection count zero,
and selected stage projection count zero.  BD126 prevents later chain windows
from rediscovering that same low-budget failure from scratch.

The landed policy reached the first private q4/mu5 all-freedom endpoint smoke
for this dynamic-collision path:
`artifact_payload_sha256=35b1be98847dfcc19ef7fb37807313b71c9c7271e18d2f2a195cc54ef2568683`,
`physical_full_bbn_span_ready=true`, `rows_reaching_endpoint=1`,
terminal `T_final_MeV=0.009144759648108285`, terminal raw
`Yp=0.1633688431175918`, terminal raw `D/H=2.095575624748707e-05`,
`selected_wall_seconds_total=85.66187122091651`, and zero selected trace-domain,
stage-projection, or raw-candidate-negative events.  This remains private
smoke evidence only; it is not a public-production, publication, SMC, or QKE
claim.

Self-audit:

| field | value |
| --- | --- |
| `real_blocker_moved` | Post-activation chain windows can reuse a recovered host max-step budget instead of repeatedly spending failed low-budget attempts after restart handoff; the private all-freedom dynamic q4/mu5 smoke now reaches `T_gamma < 0.01 MeV`. |
| `gate_removed_or_consolidated` | Existing FB70 chain/retry runtime policy changed in place; no standalone readiness, manifest, hash, figure, publication, public-dispatch, production-support, or QKE gate was added. |
| `raw_state_preserved` | Raw Rodas candidates, network one-full/two-half candidates, local errors, Newton trial negativity, mass/charge residuals, and `output_truncation_applied=false` policies remain preserved. |
| `verification` | Red-first BD126 chain max-step handoff regression plus existing fixed-policy retry regression and CLI dry-run coverage; focused RHS/FB70/WBS tests, py_compile, sync-test-count check, subagent review, diff check, and the q4/mu5 all-freedom endpoint smoke above. |
| `remaining_blocker` | Dynamic-collision/non-LRS resolution evidence, plot generation inputs, and statistical-pipeline evidence remain unresolved; BD126 does not claim public production support. |

### BD127: Resolution-Ladder Chain Max-Steps Provenance

BD127 fixes a narrow reproducibility bug in the existing FB70
resolution-ladder artifact.  Per-case `chain_max_steps_policy` already affected
the nested runtime, but `inputs.resolution_ladder_cases[*]` did not preserve
that case-local policy.  The omission was exposed by the first private
all-freedom dynamic q4/mu5 endpoint h-resolution probe, which passed with
`resolution_tolerance_ready=true`,
`artifact_payload_sha256=f49dbc4d3e569e4d7bf44bb102d36bd2ddc8bf90baedd0484cb58312b1a0c608`,
`max_abs_delta_Yp=0.00019559700227750332`, and
`max_abs_delta_DH=1.2282171793308707e-08`.

Self-audit:

| field | value |
| --- | --- |
| `real_blocker_moved` | Endpoint h-resolution evidence can now be reproduced from the artifact input block because case-local chain max-step policy is preserved alongside the existing chain h-max policy. |
| `gate_removed_or_consolidated` | Existing FB70 resolution-ladder metadata changed in place; no standalone readiness, manifest, hash, figure, publication, public-dispatch, production-support, or QKE gate was added. |
| `raw_state_preserved` | Raw nested span rows and raw observables remain embedded; no output truncation or sign repair is introduced. |
| `verification` | Red-first resolution-ladder provenance regression plus focused FB70/WBS tests, py_compile, and diff check are required before commit. |
| `remaining_blocker` | Extend dynamic endpoint evidence to q/angular/default-context ladders and endpoint-backed plot inputs before statistical claims. |

### BD128: Geometry Endpoint Defaults For Resolution Readiness

BD128 fixes a fail-open classification in the existing FB70 resolution-ladder
readiness logic.  FB70 already computes adjacent terminal deltas for `Yp`,
`D/H`, `T_final_MeV`, `Sigma_H`, and `N_eff_3T`, but the default
`resolution_terminal_tolerances` only checked `Yp`, `D/H`, and `T_final_MeV`.
The private all-freedom dynamic q4/q5 endpoint probe exposed the gap:
`artifact_payload_sha256=edebfdc4093d329a48ccd0130e87b3b8b90a41ec0e5eb068d04083e7a5e58d2d`
completed both q cases below `0.01 MeV` and reported
`resolution_tolerance_ready=true`, while `max_abs_delta_Sigma_H` was
`0.024309377733043078`.

BD128 widens the default tolerance map in place to include `Sigma_H` and
`N_eff_3T`.  This keeps raw rows and raw adjacent deltas visible, and makes the
existing artifact fail closed when geometry/effective-neutrino endpoints drift
even if `Yp`, `D/H`, and `T_final_MeV` are within tolerance.

Self-audit:

| field | value |
| --- | --- |
| `real_blocker_moved` | Resolution readiness can no longer classify abundance-only q-grid convergence as full endpoint convergence when geometry endpoints drift. |
| `gate_removed_or_consolidated` | Existing FB70 resolution-ladder tolerance defaults changed in place; no standalone readiness, manifest, hash, figure, publication, public-dispatch, production-support, or QKE gate was added. |
| `raw_state_preserved` | Raw nested span rows, raw terminal observables, adjacent deltas, and no-output-truncation policies remain embedded. |
| `verification` | Red-first BD128 regression reproduces default tolerance omission with large `Sigma_H/N_eff_3T` deltas; focused FB70/WBS tests, py_compile, diff check, and review are required before commit. |
| `remaining_blocker` | Resolve q-grid geometry convergence for dynamic all-freedom endpoint runs, then continue Rodas-informed network background and plot-input/statistical follow-ups. |

### BD129: Gauss-Laguerre Q-Grid Resolution Inputs

BD129 moves the q-grid geometry blocker from hand-weighted resolution cases
toward a reproducible quadrature-family comparison.  FB70
`resolution_ladder_cases` may now specify `q_laguerre_order`; the existing
surface then generates Gauss-Laguerre `q_nodes` and AP65 energy weights
`w exp(q) q^3`, records `q_energy_weight_source`, and rejects mixed
`q_laguerre_order` plus explicit `q_nodes`/`q_energy_weights` inputs.
The automatic path finite-checks generated nodes, base weights, and energy
weights; overflowing or nonfinite grids fail closed instead of being clipped
under the same source label.

This is not another gate.  It changes the executable resolution inputs that
feed the CPU-JAX/Rodas5P continuous AP65 path, so q-grid ladders can compare
consistent quadrature orders rather than unrelated manual energy weights.

Focused smoke evidence:
`artifact_payload_sha256=bd521fa521ab4459801a14f013dae1d52dccd251b2d6b550468393a6eca72555`
used `q_laguerre_order=3,4`, non-LRS geometry only, no dynamic collision,
boundary trace policy, and the phase-2 conservative corrector over
`N_span_end_ladder=0.005`.  Both rows reached `completed_hot_endpoint` with
`resolution_terminal_delta_violations=[]`,
`max_abs_delta_Sigma_H=4.761351568224881e-09`, and
`max_abs_delta_N_eff_3T=2.2932766796657233e-12`.  The artifact remains above
the full-BBN endpoint and is only an executable-path smoke, not endpoint
validation.

Self-audit:

| field | value |
| --- | --- |
| `real_blocker_moved` | q-grid resolution cases can now use a consistent Gauss-Laguerre energy quadrature family instead of manually changing the shear/source energy normalization between cases. |
| `gate_removed_or_consolidated` | Existing FB70 resolution case input handling changed in place; no standalone readiness, manifest, hash, figure, publication, public-dispatch, production-support, or QKE gate was added. |
| `raw_state_preserved` | Raw nested span rows, q nodes, q energy weights, terminal observables, adjacent deltas, and no-output-truncation policies remain embedded. |
| `verification` | Red-first BD129 regression locks `q_laguerre_order` node/weight generation; mixed automatic/explicit q-grid inputs fail closed; focused FB70 tests, py_compile, diff check, and review are required before commit. |
| `remaining_blocker` | Re-run dynamic all-freedom endpoint q/angular ladders with Gauss-Laguerre q orders and resolve any remaining `Sigma_H` convergence gap. |

### BD131: Laguerre Raw-Weight Contract Preservation

BD131 responds to the q-Laguerre high-q source/Jacobian audit by preserving the
raw Gauss-Laguerre quadrature weights alongside AP65 energy weights in the
existing FB70 resolution artifact.  `q_laguerre_order` rows now record
`q_laguerre_weights` with `q_laguerre_weight_source="gauss_laguerre_raw_w"` and
continue to record `q_energy_weights` with
`q_energy_weight_source="gauss_laguerre_w_exp_q_q3"`.

This does not add a standalone gate.  It changes the executable q-grid contract
so downstream source-budget and dynamic-collision probes can distinguish
physical high-q stiffness from an accidental raw-vs-energy-weight quadrature
contract mismatch.  A focused FD moment regression verifies that the recorded
energy weights reproduce `7 pi^4 / 120` for a thermal FD energy moment, and
that the raw weights are recoverable from the recorded energy weights.

Self-audit:

| field | value |
| --- | --- |
| `real_blocker_moved` | The q-grid resolution artifact now preserves enough quadrature information to audit dynamic high-q source budgets without guessing whether a downstream builder consumed raw or energy weights. |
| `gate_removed_or_consolidated` | Existing FB70 resolution metadata changed in place; no standalone readiness, manifest, hash, figure, publication, public-dispatch, production-support, or QKE gate was added. |
| `raw_state_preserved` | Raw nested span rows, q nodes, raw Laguerre weights, AP65 energy weights, terminal observables, and adjacent deltas remain embedded. |
| `verification` | Red-first BD131 regressions lock raw/energy q-weight preservation and FD energy-moment reconstruction; focused FB70/WBS tests, diff check, and review are required before commit. |
| `remaining_blocker` | Add q-node source-budget concentration diagnostics, then run dynamic q-Laguerre all-freedom endpoint ladders under the existing structured/JVP Jacobian policies. |

### BD132: Live RHS Q-Node Energy Concentration Diagnostics

BD132 adds the first q-node source-budget diagnostic to the current CPU-JAX
live-source RHS.  `_live_source_rhs_vector` now computes, only when metadata is
requested, the reconstructed per-q energy-density contribution and records:
`q_energy_density_total`, `q_energy_density_max_fraction`,
`q_energy_density_argmax`, `q_energy_density_highest_q_fraction`, and
`q_energy_density_weighted_q_mean`.

This does not alter the accepted state, the Rodas5P stage algebra, or public
claim boundaries.  It gives the existing FB69/FB70 trace/stress payloads enough
runtime evidence to decide whether q-Laguerre dynamic rows are being dominated
by a high-q node before changing Jacobian or collision-source policy.

Self-audit:

| field | value |
| --- | --- |
| `real_blocker_moved` | Dynamic q-grid failures can now report whether AP65 energy density is concentrated in one q node or in the highest-q node, instead of inferring that from endpoint failure alone. |
| `gate_removed_or_consolidated` | Existing live-source RHS metadata changed in place; no standalone readiness, manifest, hash, figure, publication, public-dispatch, production-support, or QKE gate was added. |
| `raw_state_preserved` | Raw RHS state, raw reconstructed distribution, q nodes, q weights, and untruncated BBN observables remain unchanged; the new fields are diagnostic reductions only. |
| `verification` | Red-first JAX replay regression locks q-node energy concentration against an independent reconstruction; focused JAX/FB70/WBS tests, diff check, and review are required before commit. |
| `remaining_blocker` | Run dynamic q-Laguerre endpoint rows with `frozen_source_jax_block_jvp`/`full_jvp` and use these q concentration fields to choose between high-q physics stiffness and Jacobian/source-cost fixes. |

### BD133: Q-Node Concentration Summary Surfacing

BD133 threads the BD132 q-node energy concentration metadata through the
existing FB69/FB70 stress summary path.  FB69 `rhs_stress_summary` now includes
the finite q concentration fields, and FB70 summary rows expose
`rhs_stress_q_energy_density_max_fraction_max`,
`rhs_stress_q_energy_density_argmax_max`,
`rhs_stress_q_energy_density_highest_q_fraction_max`, and
`rhs_stress_q_energy_density_weighted_q_mean_max`.

This is intentionally not a new artifact surface.  It makes the current
continuous AP65 ladder report the q-node concentration signal needed to
interpret dynamic q-Laguerre endpoint failures.

Self-audit:

| field | value |
| --- | --- |
| `real_blocker_moved` | The existing endpoint/resolution artifact can now report whether q-Laguerre dynamic rows are high-q dominated without opening raw trace payloads manually. |
| `gate_removed_or_consolidated` | Existing FB69/FB70 stress summaries changed in place; no standalone readiness, manifest, hash, figure, publication, public-dispatch, production-support, or QKE gate was added. |
| `raw_state_preserved` | Raw RHS metadata and nested rows remain embedded; the new top-level summary fields are reductions of recorded raw metadata. |
| `verification` | Red-first FB69 and FB70 regressions lock q concentration stress-summary propagation; focused FB69/FB70/WBS tests, diff check, and review are required before commit. |
| `remaining_blocker` | Run dynamic q-Laguerre endpoint rows under structured/JVP policies and decide whether to change q-grid source weighting, source/Jacobian policy, or both. |

### BD134: Dynamic Laguerre Full-JVP Auto Policy

BD134 turns the BD132/BD133 q-node concentration probe results into an existing
runtime policy rather than another artifact surface.  FB70 resolution-ladder
cases that use generated Gauss-Laguerre q grids and active dynamic collision
terms now auto-resolve the effective `jacobian_policy` from
`frozen_source_jax` to `frozen_source_jax_full_jvp` unless the individual case
explicitly overrides the policy.  Rows and input-case metadata preserve both
the requested and effective policies plus the auto-resolution reason.

The local q4 dynamic short-span probes showed full-JVP taking 2 selected steps
and 18 dynamic collision payload builds in about 52.2 s, while block-JVP took
24 selected steps and 264 payload builds in about 66.9 s on the same q4/N_mu5/
N_phi7 setup.  The q-node concentration maximum was about 0.634 at q-index 2,
with the highest-q fraction about 0.079, so the immediate blocker is runtime
source/Jacobian and step behavior before any high-q-tail reweighting change.
Mixed freedom-composition ladders preserve the requested case-wide policy
because the nested artifact currently receives one Jacobian policy for both
collision and no-collision controls.

Self-audit:

| field | value |
| --- | --- |
| `real_blocker_moved` | Default q-Laguerre dynamic resolution rows now use the measured faster full-JVP structured path instead of silently using the slower block-JVP/default path. |
| `gate_removed_or_consolidated` | Existing FB70 resolution-ladder policy and metadata changed in place; no standalone readiness, manifest, hash, figure, publication, public-dispatch, production-support, or QKE gate was added. |
| `raw_state_preserved` | The artifact records requested policy, effective policy, auto flag, reason, q grids, raw Laguerre weights, AP65 energy weights, and raw nested rows. |
| `verification` | Red-first FB70 regressions lock auto-default and explicit-override behavior; focused FB70/WBS tests, diff check, and review are required before commit. |
| `remaining_blocker` | Run endpoint dynamic q-Laguerre all-freedom rows under full-JVP and use q concentration plus payload/Jacobian counts to decide whether source weighting, background quadrature, or Jacobian/source hot-loop work is next. |

### BD135: Raw Q-Weight Runtime Plumbing

BD135 carries the BD131 raw Laguerre q-weight contract into the runtime path.
FB70 generated q-Laguerre resolution cases now pass `q_laguerre_weights` into
FB69.  FB69 forwards the raw weights into the CPU-JAX live-source replay, and
the JAX replay layer stores them on the static live-source grid and forwards
them through dynamic non-LRS/LRS collision payload refresh.  Dynamic collision
payload provenance records `source_q_laguerre_weights` next to
`source_q_energy_weights`.

The accepted interpretation is unchanged: AP65 energy weights still drive
energy-density/stress/weak-moment reductions, while raw q weights drive the
radial/collision quadrature builders when supplied.  If raw weights are not
supplied, the old reconstructed raw-equivalent weights remain the fallback.
No high-q damping, clipping, or source reweighting was added.

Self-audit:

| field | value |
| --- | --- |
| `real_blocker_moved` | Dynamic q-Laguerre source refresh now consumes preserved raw q weights directly instead of reconstructing them from energy weights in the hot source path. |
| `gate_removed_or_consolidated` | Existing FB70/FB69/JAX replay runtime plumbing changed in place; no standalone readiness, manifest, hash, figure, publication, public-dispatch, production-support, or QKE gate was added. |
| `raw_state_preserved` | q nodes, AP65 energy weights, raw q weights, dynamic collision payload provenance, and nested raw rows remain recorded. |
| `verification` | Red-first JAX replay and FB70 regressions lock raw-weight grid storage, dynamic payload forwarding, and generated Laguerre child pass-through; focused JAX/FB70/WBS tests, diff check, and review are required before commit. |
| `remaining_blocker` | Re-run endpoint dynamic q-Laguerre all-freedom rows under full-JVP and classify whether the remaining blocker is source/Jacobian cost, background quadrature, or true q-grid physics stiffness. |

## Immediate Execution Order

1. Keep provenance and JSON/hash work out of the continuous AP65 RHS/Jacobian
   hot loop; BD21-BD23 cover metadata/cache removal, BD25 adds a frozen-source
   finite-difference Jacobian fallback that avoids AP65 payload rebuilds in
   probe columns, and BD68 keeps suppressed non-LRS stage payloads off the
   JSON-safe artifact path.
2. Keep CPU-JAX/Rodas5P as the repeated-run target, retain
   `frozen_source_jax` and full finite-difference as tiny-grid reference
   evidence, and use `frozen_source_jax_full_jvp` for q4 all-freedom
   A-mode-stiff spans where dense/autodiff memory and block-JVP step-size
   limits are the measured blocker.
3. Treat the BD69/BD70 exact-current-state all-three FB70 endpoint and
   weak/control artifacts as the preferred continuous-AP65 inputs, and keep
   `step_base_reuse` only as a labeled performance approximation/control.
4. Replace hot-endpoint figure/statistics inputs with endpoint-backed FB70
   artifacts from LRS, non-LRS no-collision, non-LRS collision, and exact
   all-three private rows.
5. Extend endpoint-backed resolution/tolerance ladders over q, angular grid,
   weak-rate level/profile evidence, h-policy, and `stage_collision_payload_policy`
   before any publication-ready claim.
6. Treat the BD72 FB73/FB74/FB75 path as endpoint-backed and default-context:
   continuous AP65 endpoint and default weak bridge inputs are no longer the
   blockers; convergence and production-statistical validation remain blockers.
7. For stronger q/angular rows, prefer solver policies that directly reduce
   high-resolution Jacobian/Rodas5P cost and preserve raw failure artifacts over
   adding another wrapper or evidence gate.
8. Keep public dispatch, production SMC validation, QKE, and publication-ready
   support unclaimed until convergence/statistical evidence is endpoint-backed.

## Review Checklist For Future PRs

- Does the PR move a real physics or runtime blocker, rather than only attaching
  claim metadata?
- Does it delete or consolidate older gate/manifest/hash/readiness plumbing if
  it adds any new diagnostic surface?
- Does the artifact preserve raw failed states and untruncated observables?
- Are active freedoms and disabled freedoms explicit?
- Is QKE still out of scope?
- Is public production support still unclaimed?
- Is CPU-JAX/Rodas5P the repeated-run target?
- Are SciPy paths restricted to reference/source-generation unless justified?
- Does a focused test or benchmark directly exercise the changed contract?
- Did generated capability/docs/test-count sections get refreshed when registry
  or test metadata changed?
