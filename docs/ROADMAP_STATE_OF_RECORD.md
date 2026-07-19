# RABBIT State of Record

> **Historical snapshot (PUB-00, 2026-07-12).**  This large document preserves
> pre-deflation architecture and execution history.  It is not the current
> implementation inventory and its linked WBS is not a forward plan.  Use
> `docs/harness/PROJECT_STATE.md` for the current bounded status and
> `TYPEI_AUGMENTED_NOQKE_UNIFIED_FUTURE_PLAN.md` for future ordering.

This document captures a historical design state of the RABBIT codebase at
the baseline from which the old PR roadmap in
[ROADMAP_PR_WBS.md](ROADMAP_PR_WBS.md) began.  A reader with no prior
exposure to this project should be able to answer the following from
this document alone:

- What physics does RABBIT compute and what are the design constraints?
- Which components are production-grade vs candidate vs deferred?
- What are the numerical parity targets and where have they been met?
- What explicit decisions were made to reach the current state, and
  what was their rationale?
- Which tests exist, which are green, and which are known-red with
  justification?

Companion documents: [ROADMAP_INDEX.md](ROADMAP_INDEX.md) (navigation),
[ROADMAP_PR_WBS.md](ROADMAP_PR_WBS.md) (forward work),
[ROADMAP_SELF_AUDIT.md](ROADMAP_SELF_AUDIT.md) (audit template),
[ROADMAP_PR_CATALOG.md](ROADMAP_PR_CATALOG.md) (completed-PR log), and
[TYPEI_AUGMENTED_NOQKE_CODEX_ANTI_DRIFT_GUARDRAILS.md](TYPEI_AUGMENTED_NOQKE_CODEX_ANTI_DRIFT_GUARDRAILS.md)
(mandatory pre-read for any further augmented Type-I no-QKE work).

### 2026-05-20 augmented anti-drift correction

Two focused audits found that the AP/FB branch had started treating diagnostic
gate creation as progress.  That was the wrong development direction.  The
branch gained useful claim firewalls and evidence plumbing, but many recent
increments added manifests, readiness gates, figure wrappers, and hot-endpoint
span scouts while the continuous AP65 full-BBN endpoint blocker stayed open.

The controlling rule is now in
[TYPEI_AUGMENTED_NOQKE_CODEX_ANTI_DRIFT_GUARDRAILS.md](TYPEI_AUGMENTED_NOQKE_CODEX_ANTI_DRIFT_GUARDRAILS.md):
no new standalone gate/manifest/hash/figure/readiness PR is acceptable unless
it also removes or consolidates older gate plumbing and directly moves a
runtime physics, solver, or performance blocker.  The next work must therefore
consolidate redundant span gates, optimize the continuous AP65 hot loop and
Jacobian, then drive LRS and non-LRS endpoint runs below `0.01 MeV` before
publication or SMC packaging expands again.

### 2026-05-17 augmented full-BBN DAG refresh

The current augmented Type-I PSTF no-QKE scan keeps the programme in
diagnostic/staged scope.  The strongest executable path is the AP4/AP65
combined angular+`pstf_radial` nonlinear 3T solve with
`piecewise_frozen` source refresh, finite-mass electron-bath controls,
charge-neutral state handoff, exact scalar-QED routing, AP6 radial
number/pair-energy closure diagnostics, and AP66-AP76 publication/SMC
artifact surfaces already available downstream.

The refreshed WBS therefore does not add a new public backend or change
AP0-AP81 status.  It decomposes the AP4 programme-partial blocker into
FB-01 through FB-52: restartable terminal-state export, chained
physical-window execution, adaptive source refresh, full-chain BBN
observables, live-RHS micro-window comparison, collision ledgers,
weak-rate and electron/QED chained controls, chained resolution ladders,
AP66/AP67 full-chain rows, AP68/AP69 full-chain inference controls,
AP70/AP72 smoke SMC on real full-chain calls, AP73/AP74 Schramm and
publication artifacts, AP75 reproducibility packaging, AP76/AP79
readiness audit, optional slow production-candidate gates, and a fail-closed
live-source repeated-run diagnostic gate plus deterministic downstream
evidence-chain witness, multi-row live-source repeated-run profile gate, and
optional profile-evidence attachment to the downstream witness, plus opt-in
dynamic collision-payload refresh at CPU-JAX live-source chain window
boundaries, a diagnostic increasing-span dynamic live-source profile with
PNG plot outputs, a preset bundle runner for reproducible smoke/extended
profile plus plot artifacts, and a longer diagnostic span preset that records
BBN physical-bound metadata and keeps out-of-bound final live-source readouts
as diagnostic gate failures rather than promotion evidence, plus chain-local
AP6 radial-grid cache reuse for dynamic restart-state collision payloads and
AP41 deterministic collision-reference vectorization for cache-hit payload
refreshes, plus AP41 angular batch dispatch for those deterministic
references, plus same-geometry AP41+AP6 source-factory cache reuse, plus
dynamic payload cache-hit overhead removal, plus a single-command smoke and
extended dynamic E2E BBN span suite that emits the FB41 comparison artifact,
plus a current manifest-driven figure bundle over the FB37 PNG outputs already
generated inside that span suite, and a current figure-generation pipeline that
runs FB42 then FB43 without using legacy report/paper plotting scripts, plus an
AP75/AP79 input-bundle resolver and one-shot run wrapper for that current
pipeline, a paper/report-intent current publication-plot layer over the
FB46/FB42 artifacts, and a one-shot AP75/AP79-to-FB47 publication-intent figure
runner, followed by MeV-coverage and `Y_p` sign-stability physics diagnostics
over the current profile evidence, an LRS CL0 no-collision full-BBN
baseline artifact that compares canonical full-BBN observables with standalone
extended-LRS raw final-abundance readouts through `T_gamma ~= 0.005 MeV`, and
a progressive freedom full-BBN ladder that switches on weak-rate corrections,
non-LRS geometry, and LRS collision terms one at a time before testing the
currently supported two-freedom combinations and recording the guarded
non-LRS-collision blocker, then a private CPU-JAX residual-state full-BBN
row that replaces that guard with diagnostic non-LRS collision and all-three
evidence while leaving public dispatch closed.
FB-36 pivots that optimized path back into E2E surfaces: the FB-04 chained
runner, FB-21 live-source repeated-run gate, and AP68 guarded full-chain
forward model now accept the opt-in dynamic restart-state collision payload
refresh mode and carry dynamic payload request/build/cache counters as
diagnostic evidence without public dispatch, production SMC, or QKE promotion.
FB-37 turns those guarded AP68/AP72 dynamic E2E BBN metadata rows into
`augmented_dynamic_e2e_bbn_readout_profile_fb37_v1` and
`augmented_dynamic_e2e_bbn_readout_plots_fb37_v1`, requiring finite and
physically bounded BBN readouts, dynamic payload provenance counts, unique
fingerprints, and no-public/no-production/no-QKE scope before rendering three
diagnostic PNGs and optionally attaching the profile/plot pair to the FB-23
evidence-chain witness.
FB-38 lifts that optional FB37 profile/plot pair into the AP75/AP79
publication/readiness packaging surfaces as diagnostic attachment evidence:
AP75 now validates and copies the FB37 profile, plot manifest, and PNG files
only when AP72 full-chain physical smoke is present, while AP79 rechecks the
same no-public/no-production/no-QKE, dynamic-payload, repeated-run readout, and
span/plot consistency metadata before recording it in the readiness audit.
FB-39 makes that evidence path executable: a deterministic chain writer now
packages supplied FB37 profile/plot artifacts through AP75, immediately runs the
AP79 readiness audit with AP77 evidence, records the copied FB37 attachment and
audit-check names under `augmented_dynamic_e2e_bbn_readiness_chain_fb39_v1`, and
adds an adapter that converts passed FB27 dynamic live-source span profiles into
the AP68-style metadata rows consumed by FB37.  It remains diagnostic evidence
orchestration only, not public dispatch, production SMC validation, QKE support,
or a production-calibrated full-span BBN claim.
FB-40 turns that composed path into a single smoke-bundle command:
`augmented_dynamic_e2e_bbn_smoke_bundle_fb40_v1` generates the FB27 dynamic
span profile, converts it through the FB39 adapter, writes the FB37 profile plus
three diagnostic PNGs, then runs the FB39 AP75/AP79 readiness chain with AP77
evidence while recording hashes, paths, span/payload summaries, and retained
claim boundaries.  It remains smoke-scale diagnostic orchestration only, not
public dispatch, production SMC validation, QKE support, or a
production-calibrated full-span BBN claim.
FB-41 adds a smoke-vs-extended comparison artifact over those FB40 manifests:
`augmented_dynamic_e2e_bbn_smoke_comparison_fb41_v1` requires both bundles to
pass, requires the extended bundle to cover a strictly larger `max_N_end`,
records the span ratio, MeV `T_gamma_final` range, BBN readout ranges, dynamic
payload provenance class, and six FB37 diagnostic PNG paths, while keeping the
result diagnostic-only and not promotion evidence.
FB-42 makes that comparison reproducible as one command:
`augmented_dynamic_e2e_bbn_span_suite_fb42_v1` runs the FB40 smoke bundle and
FB40 extended bundle into separate output directories, then writes the FB41
comparison and a top-level manifest with the artifact hashes, span ratio, MeV
temperature range, BBN readout ranges, six-plot inventory, and retained
no-public/no-production/no-QKE checks.  It is still diagnostic span-suite
orchestration only, not public dispatch, production SMC validation, QKE support,
or publication-ready full-span BBN support.
FB-43 starts the plotting-script reset for the current augmented dynamic E2E
path: `augmented_dynamic_e2e_bbn_figure_bundle_fb43_v1` consumes the FB42
manifest, revalidates the FB40/FB41/FB37 artifact and PNG hashes, then copies
the six current FB37 diagnostic PNGs into a clean figure bundle with explicit
provenance and no-public/no-production/no-QKE boundaries.  It does not route
through the legacy report/paper figure generators and does not upgrade the
figures to publication-ready full-span BBN evidence.
FB-44 makes that reset executable from a clean figures directory:
`augmented_dynamic_e2e_bbn_current_figures_fb44_v1` first runs the FB42
smoke+extended span suite, rejects open top-level or nested claim boundaries,
cleans the current figure output directory, then runs the FB43 manifest-driven
bundle writer.  It is a diagnostic current-figure pipeline only, not public
dispatch, production SMC validation, QKE support, or publication-ready full-span
BBN evidence.
FB-45 makes the pipeline reachable from existing diagnostic evidence:
`augmented_dynamic_e2e_bbn_current_figure_inputs_fb45_v1` validates AP75/AP79
artifacts, copies verified AP66/AP67/AP72/AP74 inputs into a stable directory,
copies a full AP77 gate when AP79 retained a path, and otherwise records the
missing AP77 input plus the exact regeneration command or opt-in AP77 rebuild.
The FB44 CLI can consume that input bundle directly while preserving the same
diagnostic-only/no-public/no-production/no-QKE boundary.
FB-46 collapses that handoff into one command:
`augmented_dynamic_e2e_bbn_current_figure_run_fb46_v1` runs FB45, requires
`fb44_ready=true`, then runs FB44 and records the generated current figure
inventory plus `legacy_plot_generators_used=false` under one manifest.
FB-47 translates the paper/report figure meanings into current artifact-backed
plots without reusing the legacy report/paper plotting code:
`augmented_dynamic_e2e_bbn_publication_current_plots_fb47_v1` reads the FB46
run and its FB42/FB37 profile evidence, renders current observable span
response, thermo/shear span context, and dynamic payload stability-audit PNGs,
and records figure-intent references such as `paper:fig:constraint`,
`paper:fig:dynamics`, `paper:fig:ablation`, `report:fig:observable_response`,
`report:fig:story_background`, and `report:fig:convergence`.  The manifest
keeps `legacy_plot_code_reused=false` and `publication_figure_ready=false`; it
is diagnostic current-figure evidence only, not public dispatch, production SMC
validation, QKE support, or publication-ready full-span BBN support.
FB-48 collapses the same path into one reproducible command from AP75/AP79
evidence:
`augmented_dynamic_e2e_bbn_publication_figure_run_fb48_v1` runs the FB46
current-figure pipeline and then the FB47 publication-intent plot renderer,
recording FB46/FB47 artifact summaries, copied current figures, paper/report
intent plot records, FB37 source-profile hashes, and retained
`legacy_plot_generators_used=false` / `legacy_plot_code_reused=false`
boundaries.  It remains diagnostic figure orchestration only, not public
dispatch, production SMC validation, QKE support, or publication-ready
full-span BBN support.
FB-49 adds the first physics-coverage figure audit above that publication-intent
pipeline:
`augmented_dynamic_e2e_bbn_publication_physics_figures_fb49_v1` reads the FB48
source profiles plus optional long-span and excluded historical diagnostic
profiles, renders terminal `Y_p` sign/span-stability and MeV
temperature-coverage PNGs, and records that the included current evidence has
zero negative-`Y_p` rows and touches the 0.7--1.0 MeV freeze-out window while
still failing full nucleosynthesis coverage (`T_gamma <= 0.07 MeV`).  Historical
negative-`Y_p` rows can be plotted only as excluded diagnostic failure context.
The manifest keeps `publication_figure_ready=false`, no public dispatch, no
production SMC validation, and QKE out of scope.
FB-50 adds a direct baseline check for the debugging path where weak-rate
corrections and neutrino collision terms are disabled in the LRS model:
`augmented_lrs_no_collision_full_bbn_baseline_fb50_v1` compares canonical
SciPy, canonical JAX characteristic, and the standalone extended LRS runner
over `Sigma_H=(0,0.01,0.05)`, records
`Y_p=0.2423494053--0.2423927149` and
`D/H=2.4887625269e-5--2.4890647584e-5`, and keeps
`positivity_policy=raw_solver_abundances_no_observable_truncation`.  The
standalone extended-LRS rows expose raw final `X_phase2` readouts with
`T_final_MeV_min=0.004996944944314105` and zero raw-readout delta, while the
canonical SciPy/JAX rows are explicitly marked as solver-observable rows because
`canonical_forward_solver` does not expose the terminal abundance vector.  The
artifact passes with `max_abs_reference_delta_Yp=2.6441819643313602e-05` and
`max_abs_reference_delta_DH=1.3893467615459332e-09`; it is diagnostic baseline
evidence only, not public dispatch, production SMC validation, QKE support, or
collision-coupled full-BBN support.
FB-51 then re-expands the degrees of freedom from that baseline using
`augmented_progressive_freedom_full_bbn_fb51_v1`: weak-CL3 LRS,
collisionless non-LRS, and LRS collision rows each pass as single toggles;
weak+non-LRS and weak+LRS-collision pass as supported pairs; and
non-LRS+collision plus all-three rows fail closed as unsupported guarded rows
because `jax_characteristic_nonlrs` still rejects `enable_collisions=True`.
The real CPU ladder reaches `T_final_MeV_min=0.0049999999964358745` on all six
supported rows with `Y_p=0.24103996070074077--0.24854536776259234`,
`D/H=2.4792648090988874e-5--2.5233496342152163e-5`,
`single_toggle_supported_passed=3`, `pair_toggle_supported_passed=2`, and
`unsupported_guarded_rows=2`.  This is diagnostic freedom-ladder evidence
only; collision-coupled non-LRS full-BBN remains the next implementation
blocker.
FB-52 implements that blocker on the private diagnostic surface, not public
dispatch.  `JAXNonLRSResidualFullBBNConfig` phase-splits the existing S2
per-species residual-state closure into weak-freeze-out and PRIMAT-network
full-BBN phases, and the staged ladder mode
`nonlrs_collision_mode="staged_residual"` passes all eight rows through
`T_final_MeV_min=0.004999999959857061`.  The new diagnostic rows give
`nonlrs_collision_residual: Y_p=0.24177934397238576,
D/H=2.48084394432579e-5` and `all_three_residual:
Y_p=0.2479422750286894, D/H=2.5135990615031494e-5`; public
`jax_characteristic_nonlrs` collision dispatch remains guarded, and the next
blocker is resolution and physics validation of the residual closure.
FB-53 now closes the first validation blocker for that private path by running
`augmented_nonlrs_residual_full_bbn_resolution_fb53_v1` over the private
CPU-JAX/Rodas5P residual full-BBN surface.  The artifact covers `N_q=(12,16)`,
angular grids `(4,6),(6,8)`, and `residual_relax=(0.5,1.0)`, reuses duplicate
baseline solve points, and records row-level `Y_p`, `D/H`, `N_eff`,
`T_final_MeV`, residual amplitudes, residual weighted-mean closure, phase-event
diagnostics, and adjacent resolution deltas.  The real CPU run passed all six
rows with `unique_solve_points=4`, `prediction_cache_hits=2`,
`T_final_MeV=0.004999999968892998--0.005000000053785873`,
`max_abs_adjacent_delta_Yp=2.056706482561621e-05`,
`max_abs_adjacent_delta_DH=1.4852111311029742e-08`,
`max_abs_adjacent_delta_N_eff=0.00015827049378192015`, and
`residual_weighted_mean_abs_max=0.0`.  This upgrades FB52 from a single private
smoke row to a stage-scoped landed diagnostic resolution surface only; public
dispatch, production SMC validation, QKE, and publication-ready all-freedom
support remain closed.  The next blocker is same-state comparison against AP65
deterministic collision sources.
FB-54 implements that comparison as a private diagnostic source probe.  The
residual full-BBN driver now exports
`nonlrs_residual_full_bbn_terminal_ap65_same_state_projection_v1`, a JSON-safe
terminal payload that projects the residual S2 intensity state onto the
current-ray `{monopole,W_plus,W_minus}` basis and repeats it q-flat for AP65's
PSTF q-state input contract while explicitly recording
`residual_and_ap65_states_not_isomorphic=true`.  The artifact
`augmented_nonlrs_residual_ap65_same_state_comparator_fb54_v1` then calls the
existing AP65 combined angular+`pstf_radial` source evaluator at the same
scalar/shear/thermo state and records AP65 closure, component/effective `dA`
norms, residual-state norms, and cache counts.  A real CPU run over `q=(4,6)`,
angular grid `(4,6)`, and `residual_relax=1.0` passed both rows with
`ap65_dA_abs_max=1.989939293353786e-05`,
`ap65_effective_dA_over_residual_state_abs_max=0.0002918186524464677`,
`source_factory_cache_entries=2`, `radial_grid_cache_entries=36`, and
`stage_scoped_landed_surface_ready=true`.  This does not promote public
dispatch, production SMC validation, QKE, or an isomorphism claim between the
two state representations.
FB-55 turns that same-state AP65 source probe into an AP4 terminal-payload
compatibility artifact without adding a physics-equivalence claim.  The helper
`ap65_same_state_source_to_terminal_source_payload` serializes the AP65 source
as an AP4-style terminal source payload with full finite `dA_modes`, q grid,
q-energy weights, source diagnostics, and explicit
`physical_equivalence_claimed=false` provenance.  The artifact
`augmented_nonlrs_residual_ap65_terminal_payload_comparator_fb55_v1` compares
that payload shape and source contract against real AP4/AP65
`piecewise_frozen` terminal-state payloads, fails closed on missing/nonfinite
terminal `dA`, q-grid/q-weight/A-shape mismatches, missing terminal source
moments, missing AP4 `piecewise_frozen` source-update/subspan provenance,
public/production/QKE leakage, or any attempted equivalence claim, and records
only diagnostic scale ratios.  A
real CPU run against a matched `N_q=4`, angular `(4,6)` AP4/AP65 piecewise
terminal artifact passed one compatibility row with
`dA_abs_max_scale_ratio=0.014786221202355652`.  This is a stage-scoped payload
bridge for downstream diagnostic surfaces only; AP4-vs-residual physical
equality, residual/AP65 state isomorphism, public dispatch, production SMC
validation, and QKE remain out of scope.
FB-56 makes that evidence path reproducible as one command rather than a
manual AP4-build-then-FB55-compare sequence.  The artifact
`augmented_nonlrs_residual_ap65_terminal_payload_gate_fb56_v1` builds an
AP4/AP65 `piecewise_frozen` terminal payload for each requested FB54/FB55
same-state row, extracts the generated terminal states, runs the FB55
comparator over those states, and requires both the AP4 terminal artifacts and
the comparator to pass while preserving no-public/no-production/no-QKE claim
boundaries.  A real CPU smoke over `N_q=4`, angular `(4,6)`, and
`residual_relax=1.0` passed with one generated AP4 terminal artifact, one AP4
terminal state, one compatible FB55 row, and
`stage_scoped_landed_surface_ready=true`.  The next blocker is attaching this
single-command gate as optional downstream diagnostic evidence without
promoting public dispatch or a physical-equivalence claim.
FB-57 performs that attachment.  The FB-23 downstream evidence-chain writer now
accepts an optional `residual_ap65_terminal_payload_gate_artifact`, validates
the FB56 contract, pass status, empty violations, diagnostic claim scope,
no-public/no-production/no-QKE boundary, `not_promoted` decision, AP4
row/state counts, nested FB55 comparator contract/pass status, and
`physical_equivalence_claimed=false`, then records a compact
`fb56_residual_ap65_terminal_payload_gate` summary beside the existing
FB21/FB24/FB27/FB37 summaries.  The CLI dry-run and real run expose
`--residual-ap65-terminal-payload-gate`.  This is passive diagnostic evidence
only; FB56 is not required for FB23 completion and does not change readiness,
production-candidate, public dispatch, production SMC, QKE, or physical
equivalence semantics.
FB-58 adds the corresponding full-BBN physics figure layer over the artifacts
that actually reach the post-BBN temperature range.  The new
`augmented_full_bbn_physics_figures_fb58_v1` renderer consumes FB50, FB51/FB52,
FB53, and optional FB56 artifacts, rejects promoted/public/QKE-tampered inputs,
requires positive included `Y_p` and non-negative D/H, requires
`T_gamma_final <= 0.01 MeV` full-BBN temperature evidence, rejects FB56
physical-equivalence claims, and renders non-legacy PNGs for progressive
freedom yields, terminal-temperature coverage, and residual-resolution plus
terminal-payload provenance.  A real run over the current FB50/FB52/FB53/FB56
diagnostic artifacts produced three PNGs with 23 included rows,
17 terminal-temperature rows, `T_final_MeV=0.004996944944314105--0.005000010963688484`,
and `Y_p=0.24103996070074077--0.24854536776259234`.  This upgrades figure
inputs from short-span dynamic profiles to current full-BBN diagnostic
evidence, but it remains not-public, not production-SMC, not QKE, and not
publication-ready all-freedom support.
FB-59 carries that FB58 manifest into the FB-23 downstream evidence-chain
witness as optional passive evidence.  FB-23 now revalidates the FB58
contract/schema/stage, diagnostic-only claim boundary, no-public/no-production
/no-QKE/not-promoted flags, `publication_figure_ready=false`, legacy-plot
exclusion, full-BBN temperature coverage, non-negative included abundance
sign checks, required source-artifact refs, optional FB56 provenance
consistency, three PNG plot records, plot file/hash integrity, and
`physical_equivalence_claimed=false` before recording
`fb58_full_bbn_physics_figures`.  This does not rerun the renderer, affect
`chain_complete`, or promote public dispatch, production SMC validation, QKE,
AP4/residual physical equality, or publication-ready all-freedom support.
FB-60 packages the current full-BBN diagnostic layer into a single reproducible
suite manifest.  `augmented_full_bbn_diagnostic_suite_fb60_v1` consumes FB50,
FB52, FB53, and optional FB56 artifacts, rechecks row-level full-BBN terminal
temperatures and sign safety, requires the FB52 staged-residual all-three row
and FB53 residual-resolution readiness, rejects FB56 physical-equivalence
leaks, renders a nested FB58 figure manifest, verifies its PNG files and
hashes, and records the suite as diagnostic-only.  The real current artifact
run passed with `T_final_MeV=0.004996944944314105--0.005000010963688484`,
`all_three_residual_Yp=0.2479422750286894`, and
`all_three_residual_DH=2.5135990615031494e-05`.  This is still private
diagnostic suite evidence only, not public dispatch, production SMC validation,
QKE support, AP4/residual physical equality, or publication-ready all-freedom
support.
FB-61 carries that FB60 suite manifest into the FB23 downstream evidence-chain
witness as optional passive diagnostic evidence.  The FB23 writer and CLI now
accept a supplied `augmented_full_bbn_diagnostic_suite_fb60_v1` manifest, verify
its contract/schema/stage, no-public/no-production/no-QKE/not-promoted
boundary, full-BBN terminal-temperature coverage, non-negative included yields,
FB52/FB53 readiness flags, optional FB56 non-equivalence provenance, nested FB58
manifest content, nested FB58 manifest hash, manifest-relative paths, and nested PNG hashes before recording
`fb60_full_bbn_diagnostic_suite`.  This remains an evidence attachment only: it
does not affect `chain_complete`, rerun FB60 from FB23, or promote public
dispatch, production SMC validation, QKE support, AP4/residual physical
equality, or publication-ready all-freedom support.
FB-62 carries the same FB60 suite through the publication bundle/readiness
surfaces as optional diagnostic evidence.  AP75 now accepts a supplied
`augmented_full_bbn_diagnostic_suite_fb60_v1` manifest only when AP72
full-chain physical-smoke evidence is already present, revalidates the FB60 and
nested FB58 no-public/no-production/no-QKE/not-promoted boundaries, requires
full-BBN temperature/sign-safety coverage, source-relative nested paths,
FB60-vs-FB58 plot path/hash continuity, and copies the FB60 manifest, FB58
manifest, and three PNGs into the bundle.  AP79 then rechecks copied FB60/FB58
manifest hashes and plot hashes before recording
`source_bundle.full_bbn_diagnostic_suite_evidence`.  This remains optional
diagnostic attachment evidence only; it is not public dispatch, production SMC
validation, QKE support, AP4/residual physical equality, or publication-ready
all-freedom support.
FB-63 carries those AP75/AP79 full-BBN diagnostic figure attachments into the
current figure/publication run surfaces.  FB45 now validates the optional AP75
FB60 evidence against AP79 `source_bundle.full_bbn_diagnostic_suite_evidence`,
requires the same no-public/no-production/no-QKE/not-promoted and
not-publication-ready boundary, hash-checks the AP75-bundled FB60 manifest,
nested FB58 manifest, and three FB58 PNGs, then copies them into a stable
`full_bbn_diagnostic_suite` input directory while keeping them outside
`fb44_inputs`.  FB46 and FB48 propagate the resulting
`full_bbn_diagnostic_figure_inputs` block.  This is passive figure-input
indexing only: it does not rerender FB58, call legacy plot scripts, promote
public dispatch or production SMC validation, add QKE, prove AP4/residual
physical equality, or make all-freedom full-BBN plots publication-ready.
FB-64 adds a single consolidated remaining-work plan at
`docs/TYPEI_AUGMENTED_NOQKE_FULL_E2E_BBN_PLAN.md`, separating completed
diagnostic surfaces from unimplemented physics blockers, implemented-but-not-yet
connected surfaces, JAX-native optimization targets, and the FB65-FB76 execution
order.  It is a planning surface only and does not change capability claims.
FB-65 makes the FB63 full-BBN diagnostic figure inputs discoverable from one
machine-readable artifact.  `augmented_full_bbn_figure_input_index_fb65_v1`
validates an FB48 publication-figure-run artifact, requires the carried
`full_bbn_diagnostic_figure_inputs` block, rechecks no-public/no-production/no-QKE
and not-promoted/not-publication-ready boundaries, requires `T_final_MeV_max <
0.01`, verifies sign-safety fields, hash-checks the copied FB60 manifest, nested
FB58 manifest, and three FB58 PNGs, and writes role/path/hash rows for future
figure-code consumption.  This remains diagnostic indexing only: it does not
rerender FB58, call legacy plot scripts, promote public dispatch, claim
production SMC validation, add QKE, prove AP4/residual physical equality, or make
all-freedom full-BBN plots publication-ready.
FB-66 turns the existing FB51/FB52 progressive freedom ladder outputs into a
single sweep index.  `augmented_freedom_ladder_full_bbn_sweep_fb66_v1` consumes
an FB51 or FB52 artifact, keeps raw progressive-ladder observables without
index-level truncation, classifies every row as full-BBN completed,
guarded-not-supported, or failed with a MeV-region label, records the guarded
non-LRS collision blocker when only FB51 evidence is supplied, computes pairwise
interaction residuals against single-freedom baseline deltas, records all-freedom
readiness for FB52 residual evidence, and keeps the artifact diagnostic-only,
not-promoted, no-public-dispatch, no-production-SMC, and no-QKE.  It is a sweep
index over existing full-BBN rows; it does not run a new solver or make the
all-freedom path publication-ready.
FB-67 turns the terminal FB54 residual/AP65 same-state source probe into a
trajectory-checkpoint diagnostic.  `augmented_nonlrs_residual_ap65_trajectory_closure_fb67_v1`
wraps the existing FB54 comparator over explicit decreasing temperature
checkpoints, records per-window MeV spans, AP65 source-budget agreement,
q-flat projection scope/model labels, compact residual/AP65 closure rows, and
failure-kind counts that separate solver instability, projection-contract
mismatch, and source-physics mismatch.  It remains diagnostic-only,
not-promoted, no-public-dispatch, no-production-SMC, and no-QKE; it does not
implement a continuous AP65 live-source RHS or prove residual/AP65 state
isomorphism.
FB-68 profiles the exact dynamic AP65 collision payload refresh used by the
CPU-JAX/Rodas5P live-source chain before landing any further collision-kernel
optimization.  `augmented_nonlrs_dynamic_collision_payload_hotpath_profile_fb68_v1`
compares cache-disabled cold execution, shared-cache cold miss, and shared-cache
warm hits for `_dynamic_collision_source_payload_from_restart_state(...)`,
recording payload/closure contracts, `dA_modes` shape, effective `pstf_radial`
source and cache diagnostics, cProfile top rows, and cold/warm timings.  The first CPU-JAX smoke
in `docs/audit/fb68_dynamic_collision_hotpath_profile.md` passed on the
4-species chain-default shape with shared-cache cold-miss median
`1.6804822400445119 s`, warm-hit median `0.00948077195789665 s`, speedup factor
`177.2516254485815`, one source-factory cache entry, first-warm cache hit, and
18 radial-grid cache entries.  The next measured
target is AP65/AP6 `pstf_radial` source-factory or radial-grid pretabulation
for first-use cost; FB-68 remains diagnostic-only and does not promote public
dispatch, production SMC validation, QKE, or continuous AP65 collision
evaluation inside the JAX RHS.
FB-69 adds the first private current-state AP65 source-RHS prototype.
`augmented_nonlrs_continuous_ap65_source_rhs_prototype_fb69_v1` uses a
host-stepped Rodas5P-tableau micro-window path to rebuild the AP65 combined
angular+`pstf_radial` payload from current RHS/stage states, records
payload/state fingerprints, cache diagnostics, finite-difference Jacobian
policy, adjacent step-cap deltas, and frozen-window dynamic reference deltas.
A real CPU-JAX smoke over `q=(0.5,1.5,3.0)` and
`h_max=(5e-11,2.5e-11)` passed two rows with 178 source evaluations, 17
source-factory cache entries, 162 radial-grid cache entries, finite BBN
readouts, step-cap BBN delta abs max `1.1127229005893926e-16`, and reference
BBN delta abs max `3.019806626980426e-14`.  FB-69 remains private and
not-promoted: it does not reroute the public jitted CPU-JAX/Rodas5P chain,
register public dispatch, claim production SMC validation, add QKE, or make a
publication-ready all-freedom full-BBN claim.
FB-70 turns that prototype into a private increasing-span classifier.
`augmented_continuous_ap65_full_bbn_span_ladder_fb70_v1` runs FB69 over
explicit `N_span_end` ladders, records each rung's `T_gamma_final` in MeV,
labels endpoint coverage against the `0.01 MeV` full-BBN convention, preserves
the active freedom set, and fails closed on raw `Y_p`/D/H sign or bound
violations without truncation.  The first finite-difference CPU-JAX smoke over
`N_span_end=(5e-11,1e-10)` passed as a diagnostic execution with
`physical_full_bbn_span_ready=false`, two `completed_hot_endpoint` rows,
`T_gamma_final=0.7999999999214282--0.7999999999607141 MeV`, 178 source
evaluations, 21 stage-source evaluations, and zero endpoint-reaching rows.  A
zero-Jacobian probe over the same spans failed raw BBN bounds (`Y_p`/D/H) and
is retained only as failure-region evidence; the finite-difference policy
remains the default for the FB70 smoke.  FB-70 is still private, not promoted,
not public dispatch, not production SMC validation, not QKE, and not
publication-ready all-freedom full-BBN support.
FB-71 adds a private diagnostic full-BBN weak-rate convergence index.
`augmented_full_bbn_weak_rate_convergence_fb71_v1` consumes the FB51/FB52
progressive freedom full-BBN ladder, pairs weak-off and weak-on rows in the
same active-freedom context, checks `T_gamma <= 0.01 MeV`, preserves raw
positive `Y_p` and nonnegative D/H checks, records same-context weak deltas,
and can link AP80 profile-level weak-rate convergence evidence when supplied.
The first real CPU index over the FB52 artifact passed four full-BBN weak
pairs with `rows_reaching_full_bbn_endpoint=8`,
`max_abs_weak_delta_Yp=0.006193828014246616`, and
`max_abs_weak_delta_DH=3.452364587852889e-07`; it deliberately reports
`ap80_to_full_bbn_bridge_ready=false` because no AP80 JSON artifact was
supplied for that run.  FB-71 does not run a new solver, register public
dispatch, claim production SMC validation, add QKE, prove all-freedom
publication readiness, or remove the promotion-grade weak-rate convergence
blocker.
FB-72 closes the next diagnostic evidence composition step between AP80 and
FB71.  `augmented_full_bbn_weak_rate_bridge_fb72_v1` generates or consumes
AP80 profile-level weak-rate convergence evidence, builds a nested FB71 index
with that AP80 artifact supplied, and sets `ap80_fb71_bridge_ready=true` only
when AP80 and FB71 agree on profile count, profile names, and the applied-rate
q-ladder delta.  A real CPU smoke over the FB52 full-BBN freedom ladder passed
with `ap80_profile_count=1`, `ap80_total_nfev=7596`,
`ap80_applied_rate_q_relative_delta_abs_max=0.0024445680701901517`,
`fb71_passed_pair_count=4`, `fb71_rows_reaching_full_bbn_endpoint=8`, and
`ap80_fb71_bridge_ready=true`.  FB-72 remains private diagnostic bridge
evidence: AP80 is still profile/tiny-span evidence rather than a full-BBN
weak-rate convergence proof, and public dispatch, production SMC validation,
QKE, promotion-grade weak convergence, and publication-ready all-freedom
claims remain out of scope.
FB-73 rewrites the current full-BBN figure surface around current artifacts.
`augmented_publication_figure_renderer_v2_fb73_v1` consumes FB60 full-BBN
suite, FB66 freedom-ladder sweep, FB70 continuous-AP65 span ladder, and FB72
weak-rate bridge artifacts without calling legacy plotting modules, then writes
a hashed manifest and four PNG panels for endpoint coverage, freedom-ladder
terminal yields, weak-rate bridge deltas, and the continuous-AP65 span
boundary.  A real render passed with artifact payload SHA256
`6a62a214b0d13e38e5d76bac652c8ce02caf4ec93880157b49908a75a567ceae`,
`plot_count=4`, `full_bbn_T_final_MeV=0.004996944944314105--0.005000010963688484`,
`freedom_sweep_completed_rows=8`, `weak_rate_bridge_passed_pair_count=4`,
`continuous_ap65_physical_span_ready=false`, and
`publication_readiness_blocker=continuous_ap65_full_bbn_span_not_ready`.
FB-73 is diagnostic current-artifact figure evidence only: it does not run a
new solver, register public dispatch, claim production SMC validation, add QKE,
reuse legacy plotting code, or make the all-freedom full-BBN path
publication-ready.
FB-74 adds the corresponding reproducibility and QA bundle for those current
figures.  `augmented_publication_figure_bundle_qa_fb74_v1` consumes the FB73
manifest, recomputes the embedded FB73 payload hash, verifies the FB73 file
hash plus every referenced source-artifact and PNG hash, checks diagnostic
claim labels and captions, copies the four PNG files into a clean bundle, and
writes explicit QA check rows.  A real QA run passed with
stable rerun hashes
`artifact_payload_sha256=e22a8f6ea68b24e376b1ded12b6bb531199005bead8b4b7ef6d187f76f645e45`
and manifest file SHA256
`d609ba75756bbb9be0c7dd1fa256b6ad167eda1974a005527f8a08354c664cd5`,
source FB73 payload SHA256
`6a62a214b0d13e38e5d76bac652c8ce02caf4ec93880157b49908a75a567ceae`,
`plot_count=4`, `copied_plot_count=4`, and `qa_checks=10`.  FB-74 is
diagnostic QA evidence only: it does not rerender figures, call legacy plotting
scripts, run a solver, register public dispatch, claim production SMC
validation, add QKE, or make the all-freedom full-BBN path publication-ready.
FB-75 then adds a guarded statistical-pilot readiness gate over the validated
diagnostic products.  `augmented_guarded_smc_pilot_gate_fb75_v1` consumes the
AP72 full-chain physical-smoke validation artifact, FB60 full-BBN diagnostic
suite, FB66 freedom-ladder sweep, FB70 continuous-AP65 span ladder, FB72
AP80-to-FB71 weak-rate bridge, and FB74 figure QA bundle, verifies their
source hashes and closed claim boundaries, rejects stale hashed inputs or
inconsistent physical-span claims, and emits an AP69 SMC schema snapshot for a
guarded diagnostic pilot input handoff.  A real current-artifact run passed with
`artifact_payload_sha256=18841d947067979eb5cdfddeef1a4c55656fbc62e92257a6e63197820bfea352`,
manifest file SHA256
`6087af94215ff25628c18e7a5fa3fd9a22ae2981166ec0ae467d6c036e661922`,
`guarded_smc_pilot_input_ready=true`,
`statistical_pilot_input_ready=true`, `source_hashes_checked=true`,
`fb66_completed_rows=8`, `fb72_rows_reaching_full_bbn_endpoint=8`, and
`pilot_blockers=[continuous_ap65_full_bbn_span_not_ready]`.  FB-75 does not
run SMC, register candidate or public dispatch, claim production SMC
validation, claim sampler readiness, add QKE, or remove the FB70
continuous-AP65 hot-endpoint blocker.
FB-76 records the internal candidate-dispatch decision implied by that gate
without adding a dispatch route.  `augmented_internal_candidate_dispatch_decision_fb76_v1`
consumes the file-backed FB75 gate, validates its input-readiness and closed
claim boundary, rechecks every FB75 nested source file SHA, records the AP68
internal callable entrypoints as symbol-only evidence, and snapshots the
registry state showing the augmented staging capability remains in
`CAPABILITY_BY_KEY` but not `CAPABILITY_BY_BACKEND`.
The real current decision passed with
`artifact_payload_sha256=d361d9dab63dde7b54fed1656b6d7f61f5a10ec2ffdb2e6467dbb4d3f3b09518`,
manifest file SHA256
`2e871e1ec920371779bb22570e69ec4925369e0ef51d3a4f005cbb466604eac3`,
`internal_candidate_dispatch_decision=defer`,
`internal_candidate_dispatch_warranted=false`, `registers_dispatch=false`,
`canonical_forward_solver_registered=false`, and
`decision_blockers=[continuous_ap65_full_bbn_span_not_ready]`.  FB-76 does
not add a backend alias, alter `canonical_forward_solver`, run SMC, claim
production SMC validation, claim public dispatch, add QKE, or remove the FB70
continuous-AP65 hot-endpoint blocker.
FB-77 closes the FB65-FB76 diagnostic-claim ledger with
`augmented_claim_readiness_review_fb77_v1`.  It consumes the file-backed FB76
decision artifact, validates the FB76 payload hash and nested-source-hash
review status, hashes the current roadmap documents with FB77 self-reference
artifact-hash lines redacted, and records
`claim_readiness_level=diagnostic_evidence_chain_ready` with
`strongest_defensible_claim_key=guarded_internal_diagnostic_evidence_chain`.
The real current review passed with
`artifact_payload_sha256=e38abd1c8de1b7f61755fff396c9465a3679ba0f657676fa2b934b931da06f95`,
manifest file SHA256
`13c83abdb5fa0f66f61f2158f5f4e50b9c89d1dbf9a178ac8c41ad2babdddd13`,
`public_dispatch_ready=false`, `production_smc_validation_ready=false`,
`publication_ready_all_freedom_full_bbn=false`, `qke_scope=out_of_scope`,
`registers_dispatch=false`, and
`recommended_next_physics_pr=extend_continuous_ap65_full_bbn_span_to_0p01_MeV`.
FB-77 does not run a solver or sampler, register dispatch, claim production
SMC validation, add QKE, or make the all-freedom full-BBN path
publication-ready.
FB-78 opens the first consecutive-window restart handoff on the private
continuous-AP65 span-expansion surface.  FB69 now accepts supplied restart
kwargs and emits terminal restart kwargs from finite current-state AP65
micro-window rows; FB70 can set `chain_restart_handoff=true` so each
`N_span_end` rung is executed as `(previous_end,current_end)` using the previous
window's terminal state.  A real CPU-JAX finite-difference smoke over
`N_span_end=(5e-11,1e-10,2e-10,5e-10)` passed four chained windows with
`artifact_payload_sha256=463418cba619ef8199b642debcd3425f54a3fd21f24b62038a85ecba5f1e46b9`,
manifest file SHA256
`f3c22071c252c990041aea33471db0b52ecb2da59cddf59872300ba84bdc36fa`,
`restart_handoff_ready_rows=4`, `source_evaluations_total=588`,
`step_count_total=10`, and
`T_gamma_final=0.799999999607141--0.7999999999607141 MeV`.  The run remains a
hot-endpoint diagnostic with `physical_full_bbn_span_ready=false`,
`rows_reaching_endpoint=0`, and `terminal_completion_class=completed_hot_endpoint`;
it does not reach the `0.01 MeV` full-BBN endpoint or promote public dispatch,
production SMC validation, QKE, or publication-ready all-freedom support.
FB-79 turns that span-handoff surface into a repeatable private stability
bracket.  `augmented_continuous_ap65_span_bracket_fb79_v1` runs multiple
chained FB70 profiles, preserves nested failure regions, and records the last
passing profile plus the first observed failing endpoint.  The real CPU-JAX
finite-difference bracket passed with
`artifact_payload_sha256=e1c73bdae84d013a3ac0551bff404716f78bf3fdcd37a337326f3f5740e8df35`,
manifest file SHA256
`2cb311b2779809db259d0574b26c1def21057ebbe152ddb97711fdf82d260557`,
`bracket_status=pass_fail_bracketed`, `largest_passing_N_span_end=5e-10`,
`first_failing_N_span_end=1e-09`,
`best_passing_T_final_MeV=0.799999999607141`, and
`first_failing_T_final_MeV=0.7999999992142808`.  The first failing profile
still fails above the full-BBN endpoint with nested raw `Yp_nonpositive`
evidence, so FB79 records the next solver/physics blocker rather than removing
the continuous-AP65 full-BBN span blocker.
FB-80 immediately isolates that first observed failure as an `h_max`
sensitivity.  `augmented_continuous_ap65_hmax_sensitivity_fb80_v1` holds the
target span at `N_span_end=1e-9` and sweeps `h_max=(1e-9,5e-10,2.5e-10)` through
nested FB70 runs.  The real CPU-JAX finite-difference diagnostic passed with
`artifact_payload_sha256=84d6aac41fc673889320ebc5802fa78049977917da193ad9154fc487048558e4`,
manifest file SHA256
`93a3dc65aa573f0217063fc947084caa97f6c5409e6d592cf8a0d8005a927912`,
`classification=h_max_refinement_recovers_observable_failure`,
`largest_failing_h_max=1e-09`, `first_passing_h_max_after_failure=5e-10`,
`smallest_passing_h_max=2.5e-10`, `rows_failed=1`, and `rows_passed=2`.
The failing coarse row remains above the `0.01 MeV` endpoint with raw
`Yp_nonpositive`, but the same endpoint passes under smaller `h_max`, so the
next blocker is refined-step/adaptive-step span extension rather than a claim
that the endpoint is physically unreachable.
FB-81 applies that refined-step policy to the span ladder itself.  The private
`augmented_continuous_ap65_refined_span_bracket_fb81_v1` artifact holds
`h_max=2.5e-10` and runs chained FB70 endpoints
`(5e-10,1e-9,1.5e-9,2e-9)`.  The real CPU-JAX finite-difference diagnostic
passed with
`artifact_payload_sha256=76bf833035a9a23f7b444786d19924d7d676d23d2f79c086703faeb0ae3f212e`,
manifest file SHA256
`243e1170947e2ce33271c0410169be55f73fa5383de5ac305bb9005e051ab2f9`,
`classification=refined_span_pass_fail_bracketed`,
`largest_passing_N_span_end=1e-09`,
`first_failing_N_span_end=1.5e-09`,
`first_failing_T_final_MeV=0.7999999988214215`, `rows_passed=2`,
`rows_failed=2`, and `physical_full_bbn_span_ready=false`.  The new primary
blocker is the refined-hmax `1.5e-9` row's raw `Yp_nonpositive` failure, still
far above the full-BBN endpoint.
FB-82 triages that first refined-span failure without relaxing the gate.  The
private `augmented_continuous_ap65_failure_triage_fb82_v1` artifact passed with
`artifact_payload_sha256=c64bf7175a6935b39859ae521a05fadd6548fcfa5e2326d3faff9da1e9f9a783`,
manifest file SHA256
`6eed5354131696519b92f3e7ba4c2132cf5f37f9b88e4911ebcabb7012649b0b`,
`classification=strict_y_p_sign_failure_within_abundance_tolerance`,
`Yp=-1.2294890184644955e-30`, `abs_Yp=1.2294890184644955e-30`,
`abundance_bound_tolerance=1e-18`, `abundance_bounds_ok=true`,
`bound_tolerance_masks_strict_sign=true`, `DH=2.5844839174694797e-13`,
`Xn=0.1300000000856175`, `Xp=0.869999999913933`,
`N_eff_3T=11.084874967851695`, and `Sigma_H=0.015620499328388281`.  This keeps
the strict positivity failure as the next physics/debugging blocker rather
than treating abundance tolerance as a sign repair or promotion criterion.
FB-83 localizes the FB82 failure to the packed replay-state readout.  The
private `augmented_continuous_ap65_y_p_source_probe_fb83_v1` artifact passed
with
`artifact_payload_sha256=bf8c39c5947c063cb800c2f2b34f75bb3a70ac311aa3a388d6578a07a6692bb1`,
manifest file SHA256
`234d89567922ddad92d0c3750c35522f0c205616bfb3dbb3f47f8885e936d3d5`,
`classification=terminal_y_p_sign_crossing_below_tolerance_after_positive_last_stage_he4`,
`first_failing_terminal_Yp=-1.2294890184644955e-30`,
`last_passing_terminal_Yp=8.116150311829752e-31`,
`first_failing_last_attempted_He4=2.2765668298302704e-32`,
`last_passing_last_attempted_He4=1.2963142013342297e-30`,
`x_phase2_tail_start=41`, `he4_tail_index=46`, and
`physical_scale_assessment=sub_tolerance_terminal_sign_crossing`.  This narrows
the next solver/debugging target to the final update versus strict `Y_p`
positivity, not to a resolved macroscopic helium excursion.
FB-84 adds that missing terminal final-state evidence directly to FB70 rows.
The private refresh run through FB82 passed with nested
`artifact_payload_sha256=47efcd214cc16b0810797d19d59baca5ab0a1e965ab169416ac2cdb3fe486609`,
manifest file SHA256
`81cfb5fc61419c14306f703326d333135cc34d0a4d172bc545cb27195d065acb`,
`terminal_final_state_probe.x_phase2_tail_start=41`,
`terminal_final_state_probe.he4_tail_index=46`,
`terminal_final_state_probe.he4_from_final_state_vector=-1.2294890184644955e-30`,
`terminal_final_state_probe.terminal_observable_Yp=-1.2294890184644955e-30`,
and `terminal_final_state_probe.terminal_y_p_minus_final_state_he4=0.0`.
This rules out a terminal observable extraction mismatch for the current first
failure and keeps the next target on the final update/strict-sign interaction.
FB-85 then fixes the private FB69 continuous host Rodas5P prototype so
`err_norm > 1` attempts are rejected instead of merely recorded, and FB70 rows
preserve `attempt_count`, `n_rejected`, accepted/rejected step-size samples,
and adaptive-controller metadata.  The current finite-difference refresh
through FB82 passed with
`artifact_payload_sha256=9a0e0fe58cf8e318777b6b2a3cadae4cc367dd3424b6df75da178ba4a41b04dd`,
manifest file SHA256
`9a5d0cd620a4036ed1dc65c20842f6efc068d6de18fcd4091baffb9fad4ebee5`,
`first_failure_row.attempt_count=2`, `first_failure_row.n_rejected=0`,
`first_failure_row.error_norm_max=3.021584391530104e-14`, and the same
`Yp=-1.2294890184644955e-30`.  This rules out accepted `err_norm > 1` steps
as the observed smoke-ladder explanation while preserving the correct
accept/reject control for subsequent span expansion.
FB-86 then probes the phase-2 network RHS boundary directly.  The private
artifact `augmented_continuous_ap65_he4_rhs_probe_fb86_v1` passed with
`artifact_payload_sha256=ebd16b1fa3b6d4b673e33c2cff07855a075d3ca7f288230ccf2b9fe24b275fdf`,
manifest file SHA256
`b170d117620b40dcf413e94b865774b3a6f0c4dc12b2e52d8568130cf3baf201`,
`classification=he4_boundary_negative_due_to_negative_trace_intermediates`,
`first_failure_negative_trace_indices=[3,4,6,7]`,
`first_failure_negative_core_non_he4_indices=[]`,
`first_failure_he4_zero_dHe4_network_rhs=-2.618301171321943e-21`, and
`first_failure_nonnegative_trace_he4_zero_dHe4_network_rhs=7.273403769914826e-286`.
The FB86 contract fails closed if terminal observable `Y_p` mismatches the
final-state `He4` tail, and its nonnegative-trace counterfactual floors only
trace-species indices.  This identifies the next implementation target as
positivity-preserving phase-2 network evolution for trace species, not post-hoc
`Y_p` truncation.
FB-87 implements that next step as an opt-in private continuous-AP65 evolution
policy rather than a readout repair.  The default raw network RHS remains
available, while `abundance_positivity_policy=trace_boundary` constrains
trace/`He4` activities and active lower-bound derivatives inside the private
RHS, records raw-vs-policy phase-2 mass-fraction sum residuals, and gates the
trace-boundary residual.  The comparison artifact
`augmented_continuous_ap65_trace_positivity_gate_fb87_v1` passed with
`artifact_payload_sha256=dcdae7615088893f2bfbbece52620b8d81e60b1e775cb7ca8059c9d65a755276`,
manifest file SHA256
`99333c8747f006758fe9da2f0a2c8e633584a3e44a9d111999f8880d149f759d`,
`classification=trace_boundary_resolves_smoke_y_p_sign_failure_with_conservation_gate`, raw first
failure `N_span=[0.0,1.5e-09]`, raw `Yp=-1.2294890184644993e-30`, raw `Yp`
failure rows `2`, trace-boundary failure rows `0`, and trace-boundary largest
passing endpoint `2e-09`, with raw conservation max
`6.284872348663924e-18`, trace-boundary conservation max
`8.110492019931864e-18`, and conservation limit `1e-16`.  This moves the
immediate blocker from the hot `Y_p` sign crossing to extending the
trace-boundary ladder toward the real `0.01 MeV` endpoint while monitoring
whether conservation, stiffness, and solver effort remain controlled at longer
spans.
FB-88 extends that private trace-boundary ladder beyond the FB87 smoke endpoint
without opening public dispatch or QKE scope.  The new
`augmented_continuous_ap65_trace_span_extension_fb88_v1` artifact runs FB70 with
`abundance_positivity_policy=trace_boundary` and chained restart handoff,
requires complete row-level conservation/stiffness/solver-effort telemetry, and passed a real
finite-difference CPU-JAX diagnostic over `N_span_end=(2e-9,3e-9,5e-9)` with
`artifact_payload_sha256=49b2e9e858ffb87fece72c0ea2a031ed174eae9a0934db2e086c74c9997ba251`,
manifest file SHA256
`01be70a682384b58296b85a712ba4f05b9898fa95810804b8ba17ec8ca8507fb`,
`classification=trace_boundary_extension_all_requested_spans_passed`, largest
passing endpoint `5e-09`, rows passed/failed `3/0`, best
`T_final_MeV=0.7999999960714048`, conservation max
`8.746901892447222e-18`, conservation limit `1e-16`, complete
conservation/solver/stiffness rows `3/3/3`, step/attempt totals `20/20`,
rejected steps `0`, `error_norm_max=0.0006360385926131681`, source
evaluations `1166`, and stage source evaluations `140`.  This is still
hot-endpoint span-extension evidence, not full-BBN completion below `0.01 MeV`.
FB-89 turns that clean FB88 endpoint into a private multiplicative
trace-boundary span-growth scout.  The new
`augmented_continuous_ap65_trace_span_growth_fb89_v1` artifact runs the FB88
gate over a geometric ladder from `5e-9` through `4e-8`, re-checks nested
no-public/no-production/no-QKE/no-full-BBN-claim boundaries, and requires the
row-complete conservation/solver/stiffness telemetry inherited from FB88.  The
real finite-difference CPU-JAX diagnostic passed with
`artifact_payload_sha256=77a9d8a0dab4ef5b140622fb26e87860877059eab9ea3acb25e0ef068b1ab057`,
manifest file SHA256
`bc352e714afee26c01bfbc71298719dfd2fe91c063a27c11b03ce2427e53f9b2`,
`classification=trace_span_growth_all_requested_spans_passed`, nested FB88
classification `trace_boundary_extension_all_requested_spans_passed`, largest
passing endpoint `4e-08`, requested rows `3`, best
`T_final_MeV=0.7999999685712307`, conservation max
`7.782547616453054e-18`, conservation limit `1e-16`, complete
conservation/solver/stiffness rows `3/3/3`, step/attempt totals `40/40`,
rejected steps `0`, `error_norm_max=0.0006361033936367059`, source
evaluations `2326`, and stage source evaluations `280`.  This is still
hot-endpoint span-growth evidence, not full-BBN completion below `0.01 MeV`.
The backend target is also fixed: SciPy is retained as the current
physics-reference/source-generation shell for AP4/AP65, but repeated
full-chain and SMC execution should move to CPU-first JAX through the in-tree
Rosenbrock/Rodas5P solver once the restart/replay state layout exists.

FB-01 now exports the restartable terminal-state payload needed by every
higher node.  FB-02 now chains multiple AP4/AP65 piecewise-frozen physical
windows with full `Sigma/A/T/X/electron` handoff, deterministic restart/resume
equivalence, and one CPU-JAX/Rodas5P replay-state vector per successful
window.  FB-03 now adds drift-budgeted adaptive source-refresh scheduling with
uniform-schedule compatibility, per-window driver diagnostics, and explicit
budget caps.  The current FB-04 follow-up now executes a CPU-JAX/Rodas5P
pretabulated window-map replay solve for every successful chained window and
records replay pass/error metadata.  FB-04 also now has an opt-in CPU-JAX
live-source RHS sidecar for each chained window: the Rodas5P RHS reconstructs
the current S2 distribution and evaluates source-only non-LRS stress/transport,
3T/Hubble thermo, live weak monopole rates, and the PRIMAT phase-2 network in
JAX, with optional frozen collision source moments still explicitly bounded.
The same artifact extracts finite terminal `Yp`, `D/H`, `N_eff_3T`, `Sigma_H`,
and Schramm coordinates from the chained phase-2 terminal state, and when the
live-source sidecar is enabled it now records the sidecar's own terminal
`Yp`/`D/H`/`N_eff_3T`/`Sigma_H`/Schramm readout plus sidecar-vs-terminal
observable deltas as separate summary metadata, and exports JSON-safe
`restart_kwargs` from the sidecar final state so a subsequent live-source
window can start from the previous Rodas5P result.  A dedicated CPU-JAX
live-source RHS chain runner now exercises that restart handoff across
consecutive smoke windows without a pretabulated window-map replay in between,
with a JSON artifact CLI for dry-run and smoke execution.  The FB-04 chained
artifact can now optionally attach that live-source chain as finite-delta
diagnostic comparison evidence against the same piecewise/window-map chained
rows, including frozen per-window terminal collision payloads with
supplied/applied payload counts and provenance fingerprints when requested.
It can also select `rodas5p_repeated_run_source="live_source_rhs_chain"` so the
CPU-JAX live-source chain becomes the staged repeated-run evidence/readout
source while still reporting `public_dispatch_ready=false`.
A CPU smoke over two `(1e-10)` windows with that repeated-run source enabled
passed with `rodas5p_live_source_rhs_chain_completed_windows=2`,
`rodas5p_live_source_rhs_chain_collision_payloads=2`,
`rodas5p_live_source_rhs_chain_bbn_delta_abs_max=1.195155086008981e-13`,
`rodas5p_live_source_rhs_chain_state_vector_delta_abs_max=0.3975283023938289`,
`rodas5p_repeated_run_source_ready=true`,
and `public_dispatch_ready=false`.
FB-21 now wraps that opt-in source in
`augmented_nonlrs_live_source_repeated_run_gate_fb21_v1`, requiring the
live-source chain as the repeated-run source, finite `Yp`/`D/H`/`N_eff_3T`/
`Sigma_H` readouts, finite live-source-vs-piecewise/window-map comparison
deltas over the same tiny spans, and supplied/applied frozen terminal collision
payloads plus provenance fingerprints by default.  A CPU-JAX/Rodas5P smoke over
two `(1e-10)` windows passed that gate with two completed live-source chain
windows and two supplied/applied/provenance-fingerprinted collision payloads.
FB-36 now also lets the same gate accept `dynamic_restart_state` payload
refresh evidence without requiring terminal payload provenance, because the
payloads are rebuilt from each restart state and counted separately.
FB-22 now propagates those frozen-payload provenance fingerprints through AP68,
AP72, AP73, AP74, AP75, AP76/AP79, and FB-20, requiring exact completed-window
payload counts and terminal-source fingerprint evidence before accepting
live-source repeated-run BBN readout metadata.  FB-23 now adds
`augmented_live_source_repeated_run_evidence_chain_fb23_v1`, a deterministic
composition witness that writes the AP73 publication tables, AP74 plot manifest,
AP75 bundle, AP79 readiness audit, and FB-20 candidate gate under one manifest
and records the FB-21 payload-provenance summary while keeping public dispatch,
production SMC validation, and QKE disabled.
FB-24 now adds `augmented_nonlrs_live_source_repeated_run_profile_fb24_v1`, a
deterministic diagnostic profile over multiple tiny live-source repeated-run
span layouts.  It wraps FB-21 rows, requires finite repeated-run BBN readouts,
finite live-source-vs-piecewise/window-map comparison deltas, exact frozen
terminal collision-payload provenance per completed window, and preserves
no-public/no-production/no-QKE boundaries.
FB-25 now attaches that FB-24 profile artifact to the FB-23 downstream
evidence-chain manifest as optional passive evidence.  The attachment validates
the FB-24 contract, all-pass row summary, finite BBN/delta readouts, exact
per-row frozen terminal collision-payload counts, terminal-source provenance,
and unique fingerprints, but it does not make FB-24 required for FB-23
completion and does not alter AP79/FB20 readiness or candidate-gate semantics.
FB-26 now adds `collision_payload_refresh_mode="dynamic_restart_state"` to the
CPU-JAX/Rodas5P live-source RHS chain.  In that opt-in mode, each window starts
by evaluating the existing AP65 combined angular+`pstf_radial` no-QKE collision
source on the current restart state, records finite collision moments,
`dA_modes`, restart/config fingerprints, and provenance source
`dynamic_restart_state`, then freezes that payload for the Rodas5P window.  A
two-window CPU-JAX smoke built and applied two dynamic payloads with finite
BBN readouts; this remains window-boundary payload refresh, not intra-step
collision-kernel evaluation or public production support.
FB-27 now adds `augmented_nonlrs_live_source_dynamic_span_profile_fb27_v1`, a
diagnostic profile that runs the opt-in dynamic live-source chain over
increasing smoke-to-extended span ends and records finite BBN/shear readouts
plus per-window dynamic `dQ`/`dA` payload summaries and provenance
fingerprints.  Its companion
plot manifest `augmented_nonlrs_live_source_dynamic_span_profile_plots_fb27_v1`
renders generated diagnostic PNGs after cleaning only prior FB27 manifest-listed
or exact-prefix outputs.  A CPU-JAX profile through
`N_end=(1e-10,2e-10,5e-10)` and an extended profile through
`N_end=(1e-9,3e-9,1e-8)` both passed with three rows and six dynamic payloads
built/applied; this remains diagnostic and not public production support.
FB-28 now lets `augmented_live_source_repeated_run_evidence_chain_fb23_v1`
accept the FB-27 dynamic span profile and plot manifest as an optional passive
evidence pair.  The witness validates the FB-27 profile contract, dynamic
restart-state payload provenance, plot source hash, three PNG plot records, and
matched span/summary metadata before recording `fb27_live_source_dynamic_span_profile`;
this does not affect `chain_complete`, readiness promotion, public dispatch, or
production SMC validation.
FB-29 now adds
`augmented_nonlrs_live_source_dynamic_span_profile_bundle_fb29_v1`, a
diagnostic bundle writer/CLI that composes the FB-27 profile artifact and
FB-27 plot manifest under one output directory.  The smoke and extended presets
fail closed if their span ends, window count, or max-step budgets drift, and
the bundle manifest records profile/plot hashes, commands, and the same
no-public/no-production/no-QKE boundary.
FB-30 now adds the `diagnostic_long` preset and
`augmented_nonlrs_live_source_dynamic_span_long_probe_bundle_fb30_v1` for a
larger CPU-JAX/Rodas5P dynamic-span stress probe through
`N_end=(3e-8,1e-7,3e-7)`.  The live-source replay now caps the effective
Rodas5P absolute tolerance for this diagnostic path at `1e-14`, records
requested/effective solver tolerances, and fails closed when final BBN readouts
leave simple physical bounds.  The rerun over the same long-probe MeV range
has zero BBN-bound warning rows, with positive trace `Yp` readouts, while
public dispatch, production SMC validation, and QKE remain disabled.
FB-31 now threads one chain-local AP6 radial-grid cache through dynamic
restart-state collision payload refreshes in the CPU-JAX/Rodas5P live-source
chain.  The chain and dynamic span-profile surfaces record cache enabled/entry
metadata, and a focused two-window payload probe measured
`3.889020878006704 s` without a shared cache versus
`1.8382543009938672 s` with the shared cache (`2.1156054828236077x`) while
preserving the closure contract, source model, and `dA_modes` shape.  This is
runtime/cache plumbing only; it is not public dispatch or production support.
FB-32 now removes the remaining scalar Python quadrature loops from the AP41
deterministic `nu-e`, pair-annihilation, and diagonal `nu-nu` no-QKE
references by evaluating the fixed quadrature sums with NumPy broadcast
contractions.  Scalar-loop parity tests preserve the old loop outputs, and a
smoke `N_q=5` benchmark measured `8.68x`, `10.89x`, and `5.53x` speedups for
the three references; the FB31 cache-hit payload probe improved from
`0.08017252199351788 s` to `0.04857580701354891 s`.  This is still runtime
plumbing only, not public dispatch or production support.
FB-33 now batches those deterministic AP41 references across angular nodes and
species in the angular bridge, preserving the same no-QKE quadrature algebra
and closure projections while removing the per-angle scalar reference dispatch
layer.  A `B=15`, `N_q=5` micro-benchmark measured `6.88x` to `8.29x`
batch-dispatch speedups across `nu-e`, pair, and diagonal `nu-nu`, and the
same cache-hit dynamic payload probe improved from `0.07198696094565094 s` to
`0.02897743700305 s`.  The remaining cache-hit hot path is now AP6
radial-grid/provider work.
FB-34 now threads a chain-local source-factory cache beside the FB-31
radial-grid cache for `dynamic_restart_state` payload refresh.  The combined
AP41 angular plus AP6 radial source closures are reused only when the source
geometry/configuration cache key matches, and replay/chain summaries record
factory-cache enabled/entry metadata plus per-payload hit diagnostics.  A
smoke `N_q=5` cache-hit payload probe improved from radial-cache-only median
`0.026258694007992744 s` to source-factory-cache median
`0.01560014404822141 s` (`1.6832340731486084x`) while preserving the closure
contract and `dA_modes` shape.  This is still runtime/cache plumbing only,
not public dispatch or production support.
FB-35 now trims the remaining dynamic payload cache-hit overhead: source-factory
cache entries retain the factory S2 grid, angular source validation skips
redundant `allclose` checks for the identical factory grid, external-q dynamic
refreshes rebuild fixed pair-leg quadrature only on source-factory cache miss,
and AP6 radial moment projection reuses precomputed bases plus the 2x2
pseudo-inverse.  The same cache-hit payload probe improved from the FB34
median `0.01560014404822141 s` to `0.007451459008734673 s` over 25 repetitions.
This is still runtime/cache-hit overhead reduction only, not public dispatch or
production support.
Default repeated-run replacement, full collision-coupled
JAX RHS validation, and production-calibrated full-span BBN yield validation
remain open.  FB-05 now adds a staged
live-RHS micro-window comparator over the same chained restart states; the
smoke CPU run over two `(1e-14)` windows passes with no violations and records
live-minus-piecewise thermo/network/source deltas while keeping `live_rhs` as
comparison evidence rather than default policy.  FB-06 now adds a per-window
collision ledger over the same chained artifact rows, recording finite-mass
electromagnetic source counts, standard-3T plasma energy-closure residuals,
all-nine diagonal `nu-nu` number projection, identical-bank number/energy
projection, and off-diagonal unordered-pair energy closure.  The smoke CPU run
over two `(1e-8)` windows passes with no violations and residual maxima below
`1.1e-19` for the `nu-nu` pair-energy ledger and below `8.5e-19` for the
electromagnetic energy-closure ledger.  FB-07 now adds paired chained weak-rate
evidence on those real rows: terminal states preserve `weak_rates_final` and
CL3 rate-application metadata, and a same-CL3 metadata-only control chain is
compared against the applied non-LRS S2 CL3 weak-rate chain.  The smoke CPU run
over two `(1e-8)` windows passes with no violations,
`lambda_np_relative_delta_abs_max=6.910814616395567e-05`,
`lambda_pn_relative_delta_abs_max=4.6906512614639065e-05`, and
`Xn_final_delta_abs_max=4.3565151486291143e-13`.  FB-08 now runs the
fixed/charge-neutral electron-bath and finite/exact scalar-QED cross-product
over the same chained windows.  The smoke CPU run over two `(1e-8)` windows
passes with no violations, `combination_count=4`, `row_count=8`,
`charge_neutrality_evolved_rows=4`, and `exact_scalar_qed_rows=4`.  FB-09 now
runs smoke chained resolution ladders over `N_q=(3,4)`, `N_mu=(3,4)`, and
`N_phi=(5,6)`, recording terminal `Yp`, `D/H`, `N_eff_3T`, and `Sigma_H`
adjacent deltas plus source-evaluation budgets.  The smoke CPU run over two
`(1e-8)` windows passes with no violations, `row_count=6`,
`converged_ladders=3`, `source_evaluations_total=12.0`, and
`terminal_observable_delta_abs_max=1.679316997614903e-22`.
Public canonical dispatch, QKE/flavour coherence,
anisotropic/tensor QED response, and production SMC validation remain
blocked until the full-chain evidence passes the documented gates.

---

## 1. Physics scope and solver architecture

### 1.1 Scientific target

RABBIT computes Big Bang Nucleosynthesis (BBN) observables
— ⁴He mass fraction `Y_p`, deuterium-to-hydrogen ratio `D/H`,
⁷Li, ⁶Li, `N_eff` — on an **anisotropic Bianchi Type I background**
instead of the standard FLRW cosmology.  The anisotropy is parametrised
by two Hubble-normalised shear amplitudes `Σ_+, Σ_-`; for axisymmetric
(LRS) Type I, `Σ_- = 0`.

Physics foundation is the RABBIT Pedagogical Report v2.0
(`RABBIT_report_H1_revision_typo_pass2.pdf`).  Every equation referenced
in the source tree is traceable to a numbered equation in that report.

### 1.2 Three transport methods in the codebase

| Method | State DOF (N_μ=12, N_q=20) | Fidelity |
|---|---|---|
| Linearised PSTF hierarchy (ℓ=2) | 240 + scalars | Reference model; captures `σ∝a^{-5/2}` oscillatory decay but **misses** the 27–36 %% nonlinear characteristic correction to `Y_p` (paper §6.8, Fig. 5). |
| Characteristic-ray (Paper I scope) | N_μ+1 + scalars ≈ 25 at tier-1 phase-2; 27 at tier-2 | **Exact collisionless** Bianchi I solution by the method of characteristics (paper §6, eq 41–58). Publication path. |
| Full phase-space ray (Paper II scope) | N_μ·N_q + scalars ≈ 975 | Per-ray, per-momentum distribution `f_j(q_k)`; enables full collision-aware incomplete decoupling. **Private candidate only**: collisionless shell, audit-only spectral-relaxation collision preflight, and bounded tier-2 3T hook are landed; no physical collision operator or public backend yet. |

### 1.3 Tier hierarchy for incomplete decoupling

Paper §11.2, Table 2:

| Tier | Thermodynamics | Neutrino collision coupling |
|---|---|---|
| 1 | Plasma entropy conservation, single helper `T_ν` below decoupling. | None (collisionless). |
| 2 | Coupled 3-temperature system `(T_γ, T_νₑ, T_νₓ)` with Mangano-style momentum-averaged energy-transfer source (paper eq 161–163). | Momentum-averaged ν–e and ν–ν energy-transfer scalar per species. |
| 3 | Tier-2 thermo + full per-momentum, per-direction collision operator. Adds diagonal ν–ν scattering. | Full Boltzmann with Hannestad–Madsen ν–e kernel + pair process + diagonal ν–ν. |
| 4 | Tier-3 + flavour oscillations via QKE density-matrix formalism. | Tier-3 + off-diagonal ν–ν terms. |

### 1.4 Solver choice and AD strategy

The production integrator is the 8-stage order-5(4) Rosenbrock–Wanner
method **Rodas5P** (Steinebach 2023, BIT 63:27; paper §16.1, eq 129).
This choice is **load-bearing** — Rosenbrock methods handle the
stiffness around weak freeze-out (where Γ_ν/H ≫ 1 transitions through
unity) materially better than diffrax's current explicit or
ESDIRK-class stiff tableaux.  AD support through the adaptive Rodas5P
step controller is not needed at the solver level — it is provided at
the outer layer via `jax.custom_vjp` in
[`src/rabbit/jax/gradient_bridge.py`](../src/rabbit/jax/gradient_bridge.py)
with a finite-difference fallback when native forward-mode AD through
the event-triggered `while_loop` is unavailable.

**Consequence for the roadmap:** any PR that proposes switching to
diffrax for GPU compatibility or AD must be rejected on stiffness
grounds.  See [JAX_CHAR_GPU_OPTIMIZATION_PLAN.md §4](JAX_CHAR_GPU_OPTIMIZATION_PLAN.md#4-what-not-to-do)
for the full argument.

---

## 2. Current component status

### 2.1 SciPy reference driver
[`src/rabbit/drivers/full_coupled_typeI.py`](../src/rabbit/drivers/full_coupled_typeI.py)

| Capability | Status | Notes |
|---|---|---|
| Bianchi Type I geometry (Σ_+, Σ_-) | Production | Paper §2.1–§2.3. |
| Characteristic transport (LRS, tier-1) | Production | Paper §6. |
| Characteristic transport (LRS, tier-2 collisions=OFF) | **Fixed** | Was broken with IndexError (fall-through bug); hoisted char pack/return inside `if use_char:` scope. See PR-S1 in [ROADMAP_PR_CATALOG.md](ROADMAP_PR_CATALOG.md). |
| Linearised PSTF (tier-2 collisions=OFF) | **Fixed** | Was broken with UnboundLocalError on `T_gamma_for_rates`; added explicit assignment. |
| Characteristic transport (per-species, tier-2 collisions=ON) | Candidate | Isotropic-decoupling backbone + anisotropic residual relaxation; bounded transitional collision surface, not a broad production-ready claim. |
| CL0/CL1/CL2 weak corrections | Production | Paper §13.4. |
| CL3 (finite-mass + weak magnetism) | Candidate | Retained; parity with external codes limited. |
| PRIMAT AC2024 network | Production | 12-reaction backbone / 31 with ⁶Li. |
| Bennett QED EoS (O(e³)) | Production | Paper §14. |

### 2.2 JAX linearised-PSTF driver
[`src/rabbit/jax/driver_typeI.py`](../src/rabbit/jax/driver_typeI.py)

Historical JAX path.  At tier-1 and tier-2, 240-DOF transport; Schur
block-sparse Jacobian is *beneficial* at this size (the JAX solver's
`active_indices=...` path exploits this).  Weak rates via live
monopoles from the linearised Ψ₀ (defaults to equilibrium FD since
collisionless Ψ₀ = 0 exactly).  Retains only the reduced-model
shear-to-quadrupole feedback sector; **does not capture the
characteristic-ray nonlinear monopole distortion**.

Now dispatches to the characteristic driver when
`transport_mode="characteristic"` — see §2.3.

### 2.3 JAX characteristic-ray driver
[`src/rabbit/jax/driver_typeI_char.py`](../src/rabbit/jax/driver_typeI_char.py)

Publication-grade JAX Bianchi I BBN path (LRS, tier-1 and tier-2).
Mirrors the SciPy characteristic production path 1-to-1.

**Tier-1:** |ΔY_p| ≤ 4 × 10⁻⁸ across Σ_H ∈ [0, 0.5], CL0–CL2.
**Tier-2:** |ΔY_p| ≤ 7 × 10⁻⁸ across Σ_H ∈ [0, 0.5], CL0–CL2
(directly against the now-fixed SciPy tier-2 char reference).

Key design features:
- Rodas5P integrator (`src/rabbit/jax/solver_jax_rodas5p.py`).
- State layout `[Σ₊, Σ₋, I₁..I_Nμ, S, T_γ, (T_νₑ, T_νₓ if tier≥2), X_i]`
  with analytic `J_j(S)` reconstruction; the explicit `J_j` state
  slots were removed in PR-A.
- Analytic `μ_j(S)` reconstruction via the LRS forward map
  `μ_j = sign(μ₀)·√(X₀ e^{6S}/(1+X₀ e^{6S}))`.
- Closed-form angular Jacobian in the production weight convention
  `J_j = dμ_j/dμ_{j,0}` reconstructed from `(X₀, S, μ_j)` on every RHS
  call rather than numerically evolved.
- Live weak rates at CL0/CL1/CL2/CL3 from the transported monopole
  `f̃₀(q) = ½ Σ_j w_j J_j f_FD(q e^{2 I_j})` (paper eq 58).
- Tier-1 N_eff ≈ 3.011 (entropy-cascade helper),
  tier-2 N_eff ≈ 3.034 (Mangano momentum-averaged source;
  gap to SM 3.044 closed only by tier-3 full collision).

### 2.4 Other dispatch backends

| Backend | Status | Scope |
|---|---|---|
| `auto` | Canonical | Routes to the bounded JAX characteristic default (`jax_characteristic`; tier-2 via `jax_thermo_tier=2` inside the same bounded scope) |
| `scipy` | Canonical | SciPy production driver |
| `jax` | Canonical | JAX linearised PSTF live-weak tier-1 |
| `jax_advanced` | Canonical | JAX linearised PSTF live-weak tier-3 (CL0–CL3, 3T thermo). |
| `jax_characteristic` | Canonical | JAX characteristic-ray tier-1 |
| `jax_characteristic_tier2` | Canonical | JAX characteristic-ray tier-2 |
| `jax_characteristic_nonlrs` | Candidate | JAX characteristic-ray generic Type I tier-1/tier-2 |
| `jax_classA` | Candidate | All Class A Bianchi types |
| `jax_classB` | Candidate | Class B layered |
| `jax_tilted` | Candidate | Tilted Type I (v0 ≤ 10⁻³) |

Registered in [`src/rabbit/config/backend_capabilities.py`](../src/rabbit/config/backend_capabilities.py).

---

## 3. Design decisions and their rationale

These decisions were made during development and are **not captured**
in the paper or in older implementation notes.  Each is load-bearing
for subsequent work.

### 3.1 CPU-preferred default for the characteristic driver

The compact characteristic state is now 13 DOF at tier-1 and 15 DOF at
tier-2.  Single solves remain kernel-launch-bound on GPU; JAX's
default `XLA_PYTHON_CLIENT_MEM_FRACTION=0.75` preallocates ~12 GB of
VRAM unconditionally.  Default `runtime_device_policy="cpu_preferred"`
therefore:
- Wraps the driver body in `jax.default_device(cpu_devices[0])`.
- Avoids GPU preallocation entirely.
- Is strictly faster on single solves and still faster for small batch
  grids.

PR-G added explicit batched entrypoints
`run_char_batch_tier1(...)` / `run_char_batch_tier2(...)` on top of an
event-masked pure-JAX Rodas5P core.  On the local RX 6950 XT audit
rig, GPU throughput only begins to beat CPU around batch size
`N ≈ 128`; at `N = 256` the measured warm tier-1 throughput is
`7.44 ms/solve` on GPU versus `11.83 ms/solve` on CPU.  GPU is
therefore recommended only for medium/large batch inference with an
explicit `runtime_device_policy="gpu_then_cpu_retry"` override plus
appropriate XLA env vars
([JAX_CHAR_GPU_OPTIMIZATION_PLAN.md §2.1](JAX_CHAR_GPU_OPTIMIZATION_PLAN.md#21-vram-preallocation-zero-risk-zero-physics-impact)).

### 3.2 Stable-identity RHS cache

Rodas5P's internal solver-runner cache keys on `id(rhs_fn)` of the
traced RHS callable.  Rebuilding a `jax.jit(rhs)` closure per call
produces a fresh id and forces the Rodas5P `while_loop` to re-trace.
The fix is a host-side cache
(`_CHAR_RHS_CACHE` in `driver_typeI_char.py`) keyed by
`(phase, correction_level, thermo_tier, N_μ, N_q, n_species, n_reactions, τ_n, η, N_eff, f_ν)`.
Repeated solves with identical parameters reuse the same compiled
kernel.  This single change took warm single-solve time from ~6 s to
~1.3 s.

### 3.3 Dense Jacobian over block-sparse at the current characteristic size

The JAX solver exposes a block-sparse Schur path via `active_indices`
([solver_jax_rodas5p.py:83](../src/rabbit/jax/solver_jax_rodas5p.py#L83)).
After the analytic characteristic compaction, the canonical LRS
driver no longer carries an explicit passive transport block: the
state is `[Σ₊, Σ₋, S, T_γ, (T_νₑ, T_νₓ), X_i]`.  The public
`jacobian_mode='block_sparse'` switch is therefore normalized back to
`'dense'`, and the production path treats the characteristic system as
fully dense at the current 13/15-DOF size.

The accepted benchmark trend is now:
- dense beats the legacy block-sparse path on warm scalar solves
- PR-G's GPU batch pay-off comes from amortising launches across the
  batch, not from reviving Schur complement structure
- any future block-sparse revisit belongs to the tier-3
  full-phase-space surface where a genuinely large passive sector
  reappears

Block-sparse is retained as a config opt-in because:
- At genuinely larger state (for example the future per-ray
  full-phase-space tier-3 surface with ~975 DOF) the calculus flips.
- On GPU the extra matrix products are cheap (one big GEMM), and
  smaller LU launches fewer kernels.

### 3.4 Tier-2 via Mangano momentum-averaged source (not full collision)

The existing JAX primitive `coupled_3T_rhs_jax` already contains the
Mangano-style `(T_γ − T_ν)·T_γ⁴·T_ν⁴·F(m_e/T)` ν–e energy-transfer
term (see `src/rabbit/jax/nudec_coupled_jax.py:69–88`).  Wiring this
into the characteristic driver yielded tier-2 `N_eff ≈ 3.034`, within
0.3 %% of the Standard-Model benchmark 3.044.  The residual 0.01 gap is
**not** a bug — it is the accuracy ceiling of the momentum-averaged
collision model and is closed only by the full-collision tier-3
upgrade.

### 3.5 SciPy char tier-2 fall-through + PSTF UnboundLocalError fixes

The SciPy `coupled_rhs` had two pre-existing latent bugs:

1. `transport_mode=CHARACTERISTIC, tier=2, enable_collisions=False` fell
   through the `if use_char:` block without returning, then crashed
   with `IndexError` because the subsequent LINEARIZED_PSTF layout used
   indices incompatible with the char-sized state.
2. `transport_mode=LINEARIZED_PSTF, tier=2` never assigned
   `T_gamma_for_rates` in its tier-2 branch, triggering
   `UnboundLocalError` at the weak-rate call.

Both were fixed in the same SciPy revision that unlocked SciPy-as-
reference for the JAX tier-2 parity tests.  See
[ROADMAP_PR_CATALOG.md#pr-s1](ROADMAP_PR_CATALOG.md) for the diff.
This is the only SciPy-source fix required to date; no other SciPy
physics changes were made.

### 3.6 Characteristic scope split: LRS canonical, non-LRS candidate

The publication characteristic path remains the LRS canonical surface:
`transport_mode="characteristic"` still requires `Σ_- = 0` and keeps
the single-direction cosine map `μ = cos θ ∈ [-1, 1]` with the
analytic forward integral `X = μ²/(1-μ²) → X_0 e^{6S}`.

PR-N1 delivered the additive non-LRS S² primitives in
[`src/rabbit/jax/characteristic_rays_nonlrs_jax.py`](../src/rabbit/jax/characteristic_rays_nonlrs_jax.py),
and PR-N2 wired them into the explicit candidate backend
`jax_characteristic_nonlrs`.  The landed non-LRS driver is still
compact: it evolves `(Σ_+, Σ_-, S_+, S_-, T_γ, X_i)` at tier-1 and
reconstructs `(μ, φ, I, J)` analytically from `(S_+, S_-)` on every
RHS call.  Tier-2 3T thermodynamics is available as an explicit
candidate opt-in.  PR-N3 adds a private CPU-JAX/Rodas5P residual-state
staging helper, `run_nonlrs_tier2_residual_state_jax(...)`, that carries
the explicit S² per-species `R_I/R_J` state required by anisotropic
residual relaxation smoke tests.  That helper is diagnostic evidence, not
public collision dispatch.  Current guardrails:

- explicit opt-in only; never selected by `backend="auto"`
- tier-1 by default; tier-2 3T thermodynamics is explicit opt-in via
  `jax_thermo_tier=2`
- public `jax_characteristic_nonlrs` dispatch remains collisionless and
  still rejects `enable_collisions=True`
- residual-state non-LRS collision closure is private/staged only, with
  `public_dispatch_ready=False` and QKE out of scope
- `N_phi=1` reserved for the exact LRS reduction slice

SciPy characteristic remains LRS-only.

### 3.7 Tier-3 deferred behind the three dependency PRs

Per [IMPLEMENTATION_GUIDE_FULL_COLLISION_NOQKE.md §3](IMPLEMENTATION_GUIDE_FULL_COLLISION_NOQKE.md#3-implementation-plan),
tier-3 needs (i) q-advection, (ii) per-ray full-phase-space state, and
(iii) a new diagonal ν–ν scattering operator.  The first two landed in
**partial** form in PR-T3A as a private CPU-only collisionless shell:
`src/rabbit/jax/q_advection_jax.py` provides continuous upwind
semidiscrete transport plus the exact-remap PCHIP oracle, and
`src/rabbit/jax/driver_typeI_full_boltzmann.py` wires a tier-1
collisionless `(Σ_+, Σ_-, f_{species,ray,q}, S, T_γ, X_i)` driver
through the experimental factorized Jacobian hook.  This surface is not
registered in the backend capability table and is not inference-exposed
yet.  An analytic/factorized Jacobian remains *mandatory* for the
future full-collision 3T driver because `jacfwd` on the ~975-DOF state
at every Rodas5P step is computationally prohibitive without it.  A
bounded host-side Stage-B preflight has now also landed in
`src/rabbit/jax/full_boltzmann_collision_preflight.py`: it projects the
explicit ray state onto the three isotropic species banks
`(ν_e, ν̄_e, ν_x)` and reuses the existing species-resolved collision
backbone to materialize a small moment-space collision core.  The same
module now also lifts that core back onto the explicit `(species, ray, q)`
state with a factorized `U C V` transport update.  Current finding: the
production-bounded source-only backbone has zero state Jacobian, while
an audit-only spectral-relaxation closure produces a finite
block-diagonal moment-core Jacobian whose lifted explicit-state
factorization matches dense finite differences in bounded tests.  A
follow-up private runtime mode now reuses the same spectral-relaxation
closure inside `src/rabbit/jax/driver_typeI_full_boltzmann.py`, so the
explicit shell can run bounded CPU solves with collision transport and a
matching factorized Jacobian payload, but this path remains
non-canonical and non-inference-exposed.  A second private runtime mode
now also exists behind `collision_mode="projected_physical_preflight"`:
it replaces the spectral surrogate with a projected physical
state-dependent source+damping closure on the species banks and restores
exact dense-AD parity for the bounded tier-2 regression surface.  This
is still only a preflight step, not the final Hannestad–Madsen + pair
Stage-B operator.  Separately, the host-side preflight now also exposes
`closure_mode="direct_kernel"`, which evaluates the existing
`NuEScatteringOperator` and `PairProcessOperator` on the isotropic
species banks through an explicit `T_bank/T_gamma` remap.  That direct
operator-backed bank surface is regression-locked, now includes the
required `1/H` conversion to `d/dN`, and also materializes a bounded
augmented Jacobian on `[f_bank, T_gamma, T_nu_e, T_nu_x, H]`.  It is
now also wired into a bounded private runtime candidate behind
`collision_mode="direct_kernel_preflight"`, where the primal RHS stays
on a host callback and the Jacobian is supplied explicitly through the
low-rank Rodas5P hook.  This has only been locked at the `rhs/jacobian`
evaluability level so far; full solve smoke and bounded physics parity
are still outstanding.

**PR-T3B canonical milestone (2026-04 cumulative):** the
``ap_unified_preflight`` collision mode in
`rabbit.jax.driver_typeI_full_boltzmann` simultaneously satisfies
2 of 3 canonical PR-T3D §5 gates while keeping Rodas5P + JAX/GPU
friendly architecture (no IMEX, no operator splitting):

- FLRW ``N_eff = 3.0345`` (gap ``+0.0095`` to Mangano 3.044 —
  the documented AP-form model-approximation limit per
  PR-T3B-PF #15 scope reframing).
- Grid-converged across ``(N_mu, N_q) ∈ {(4,6), (8,12), (12,20)}``
  with spread ``< 1e-4``.
- **Anisotropy spread ``~7e-5`` across ``Σ_H ∈ {0, 0.05, 0.10}``
  passes the canonical ``< 1e-3`` gate by two orders of magnitude.**

The canonical-track candidate is registered as
``JAX_TYPEI_AP_UNIFIED_TIER3_CANDIDATE`` in ``CAPABILITY_BY_KEY``
**and** wired into ``CAPABILITY_BY_BACKEND`` under
``backend="jax_ap_unified_tier3"`` (PR-T3D canonical #2).
``canonical_forward_solver(backend="jax_ap_unified_tier3", ...)``
returns a ``BBNPrediction`` whose metadata surfaces
``flrw_mangano_gap_documented = 0.0095`` plus the PR-T3D §5
canonical gate verdicts (``anisotropy_canonical_gate_passed``,
``grid_canonical_gate_passed``).  The dispatch is LRS-only and
rejects the candidate Teff opt-in.  See
``docs/audit/PR-T3B_jax_kernel_runtime.md``,
``tests/test_pr_t3b_ap_unified.py`` (calibration milestone), and
``tests/test_pr_t3d_ap_unified_dispatch.py`` (dispatch contract).

A separate additive preflight,
`src/rabbit/jax/collisions_jax.py`, now provides JIT-compatible
pure-JAX ports of `NuEScatteringOperator._evaluate_vectorized` and
`PairProcessOperator.evaluate` so the future GCS cycle does not require
host callbacks.  The ports mirror the SciPy reference structurally
(matrix element, statistical factor, prefactor, divisor, near-zero
`y1` skip and quadrature schemes); on the shared Laguerre grid
(`N_q == N_int` for ν-e or `N_q == N_quad` for the pair process) they
match the SciPy reference element-wise to better than `1e-30` absolute
on both `nue` and `nux` couplings, and detailed balance at
`f = f_FD` collapses to the same tolerance.  Off the matched grid,
the pair-process port uses the existing PCHIP cubic Hermite from
`q_advection_jax` rather than scipy `interp1d` natural cubic spline,
and a measured `~8.3%` relative gap is locked at `< 10%` pending a
JAX-native natural cubic spline replacement.

A follow-up preflight has now wired those pure-JAX kernels into the
private full-Boltzmann driver as
`collision_mode="jax_kernel_preflight"` (tier-2 only), routed through
the existing `_collision_bank_core_jax` dispatcher alongside
`spectral_relaxation_preflight` and `projected_physical_preflight`.
The bank-core internally uses Laguerre quadrature grids matched to
`q_nodes` so PCHIP interpolation collapses to identity at the input
nodes; algebraic detailed balance at `T_nu = T_gamma` is preserved on
the bank state, and the residual on the explicit transport rays at
FLRW (`Σ = 0`, `T = 10 MeV`) is measured at `6.7e-14` (locked at
`< 1e-10`, the natural noise floor after dividing kernel-level DB
residuals `~1e-30` by `H_MeV ~ 1e-20`).  The new mode is private,
non-canonical, and not exposed via inference dispatch; it remains an
audit-only runtime path until end-to-end Rodas5P solve smoke and
bounded FLRW `N_eff` parity close.

The cumulative tier-3 preflight surface is now feature-registered
as `TIER3_FULL_COLLISION_PREFLIGHT` (candidate, diagnostic) in
`rabbit.config.feature_capabilities`.  Calibration measurements
locked across 122 regression tests:

- AP-form (`projected_physical_preflight`) FLRW `N_eff = 3.030738`,
  fully grid-converged across `(N_mu, N_q) ∈ {(4, 6), (8, 12),
  (12, 20)}` (spread `< 3e-5`); gap to Mangano 2005's 3.044 is
  `+0.0133` — the canonical PR-T3B path needs an AP/IMEX hybrid
  or higher-order asymptotic expansion to close it.
- jax_kernel (full Hannestad-Madsen) FLRW `N_eff = 2.993427`
  (anti-heating from the `T_e = T_ν` approximation embedded in
  the SciPy reference; q-grid remap fix exposes a stiff
  ``∂C / ∂T`` manifold that Rodas5P cannot handle without
  IMEX/AP splitting).
- Anisotropic stability (PR-T3D §5 gate `< 1e-3`): currently
  `~0.54` across `Σ_H ∈ {0, 0.05, 0.10}` — `~500x` above the
  canonical target.  Σ = 0.30 exceeds the bounded preflight
  solver budget.
- Pair-process off-grid SciPy parity tightened from `8.3% rel` to
  `6.5e-16 rel` (14 OOM) after the PCHIP -> JAX-native cubic
  spline swap.
- Diagonal ν-ν detailed balance at FD: `~3e-23` after cubic
  spline swap (`~5x` tighter than PCHIP); energy conservation
  rel: `~0.23%` (`~8x` tighter).

Building-block modules landed for the canonical work:
`rabbit.jax.cubic_spline_jax` (not-a-knot natural cubic, 1e-12
SciPy parity), `rabbit.jax.collision_rates_jax` (Mangano /
Hannestad-Madsen total rate, 1e-14 SciPy parity).  Cross-code
fixture (`tests/fixtures/tier3_cross_code.json`) carries per-mode
preflight measurements + grid-resolved AP-form entries +
anisotropic sweep block, all version-locked.

Four canonical blockers explicitly enumerated in the registry:
(i) AP-form 0.013 N_eff gap to Mangano at FLRW; (ii) JAX-kernel
q-grid remap stiff Jacobian manifold; (iii) anisotropic N_eff
stability ~0.54 spread; (iv) Dolgov-Hansen-Semikoz ν-ν
coefficient calibration deferred.

---

### 3.8 Augmented PSTF no-QKE programme ledger

The forward path for a nonperturbative augmented-PSTF Type I solver is
now tracked in
[IMPLEMENTATION_GUIDE_TYPEI_AUGMENTED_PSTF_NOQKE_WBS.md](IMPLEMENTATION_GUIDE_TYPEI_AUGMENTED_PSTF_NOQKE_WBS.md).
This is a new active SDD/WBS ledger, not a promoted runtime capability.
It covers:

- `(non)-LRS Bianchi Type I` only;
- classical scalar Boltzmann transport only, with QKE explicitly out of
  scope;
- positivity-preserving augmented distribution coefficients rather than
  the legacy linearised `f=f0(1+Psi)` PSTF state;
- S_N angular quadrature plus PSTF projection as the reference angular
  method;
- mandatory `ell_max`, angular-grid, and momentum-grid convergence
  gates before any capability promotion;
- SciPy reference implementation before any JAX/XLA port.

The first scaffold module is
[`src/rabbit/transport/angular_decomposition.py`](../src/rabbit/transport/angular_decomposition.py),
which defines the angular-mode, convergence-ladder, and deterministic
collision-quadrature contracts without changing solver dispatch.

Current landed pieces now also include the LRS collisionless SciPy shell
and its validation harness: augmented distribution reconstruction,
stress-moment projection, solver-backed `ell_max`, `N_q`, and `N_mu`
convergence runners, deterministic collision-quadrature primitives,
matched-node monopole fixed-quadrature `nu-e` and pair-process reference
kernels, explicit-rate pointwise diagonal no-QKE `nu-nu` redistribution
diagnostics, an augmented-distribution `nu-e` monopole projection bridge,
an LRS collisionless stability-envelope reporter, and the weak-rate
monopole adapter.  A replay-stable Sobol/QMC control-variate validation
scaffold is also landed, and its augmented nodal-source projection report
now routes sampled angular `df/dN` sources through the generic nodal
collision projection bridge; it is still not a full collision-physics
evaluator.  The
first CPU-JAX augmented distribution core parity module is landed for
fixed-grid reconstruction/projection tests, and a CPU-JAX nodal
collision-source projection bridge now mirrors the SciPy AP6 bridge's
array-level projection semantics for LRS/non-LRS fixed grids.  A
fixed-grid CPU-JAX LRS collisionless RHS
factory now tracks the SciPy reference through the existing JAX Rodas5P
pure core on short staging solves, and a bounded static fixed-source
projected-source RHS plus Rodas5P wrapper now exercises that projection
bridge inside the solver loop.  Fixed-grid
inputs are validated statically and dynamic traced grids are rejected in
this staging surface.  A SciPy non-LRS S2 quadrupole source projection
gate is also landed for the diagonal `{monopole, W_+, W_-}` coefficient
staging basis, including the `Sigma_-=0` reduction check against the LRS
linearized quadrupole source.  Source-only non-LRS `N_mu` and `N_phi`
angular-grid convergence runners now report coefficient-source RMS
values and expected-mode residuals for that projection gate.  The
collision bridge now also includes a generic deterministic nodal
`df/dN(q,n)` source-to-augmented projection primitive; this is the
projection surface a future S_N collision kernel can feed, not a landed
angular collision kernel.  The augmented weak bridge records explicit
CL3 angular metadata with distribution quadrupole proxies when the
caller supplies angular kernels and q-energy weights.  AP25 wires the
current LRS `Sigma_+` into the existing bounded `Sigma_+ K_2` CL3
multiplier for the SciPy augmented weak/network RHS, so `lambda_np` and
`lambda_pn` receive that staged LRS angular weak-rate correction when
`correction_level=3` and the metadata is available.  AP58-AP60 then add
live moment-input angular weak-rate modes: LRS
`lrs_cl3_quadrupole_input` and non-LRS S2
`nonlrs_s2_cl3_quadrupole_input` compute species-aware
`lambda_np`/`lambda_pn` factors from current per-q angular moments, with
the angular-mode resolver now reporting the corresponding rate
application as ready when the required moments are present.  LRS keeps
the bounded legacy multiplier as its default; staged non-LRS
correction-level-3 configs now default to the current S2 moment-input
application while explicit `metadata_only` remains available for control
rows.
AP26 adds the corresponding smoke-scale AP8 weak-rate candidate
sub-gate: it sweeps `Sigma_+`, records the multiplier plus
`lambda_np`/`lambda_pn` deltas against the `Sigma_+=0` reference, and
enforces explicit factor/rate limits without promoting collision-coupled
or full-BBN candidate status.  AP27 adds the corresponding
collision-feedback candidate sub-gate around the AP22/AP23/AP36
source-variant artifact: it checks selected standard-relative
thermo/network deltas and collision source moments for deterministic
source variants, forwards AP36 frozen/live source-update policy and
solver controls into the AP35 angular wrapper route, and still does not
promote a full physical angular collision kernel or full-BBN candidate.
AP28 adds a staged 3T span-ladder candidate gate over that same artifact
path, recording thermo/H/network/source limits across explicit `N_span`
ladders with smoke defaults and optional longer spans; its LRS angular
cases forward the same AP36 source-update policy.
AP29 adds deterministic collision-feedback 3T convergence runners over
`ell_max`, `N_q`, and `N_mu` ladders for selected AP22/AP23/AP36 source
variants.  The LRS convergence runners now forward the AP36 angular
source-update policy into the artifact path, and a smoke q-ladder exercises
the real AP35/AP36 angular wrapper.
AP30 adds replay-stable QMC collision source-moment vector reports for
named moments such as `dQ_nue_pair_N` and `dQ_nux_bank_N`.
AP31 adds the first live angular collision-kernel subpath: deterministic
`nu-e` scattering can now be evaluated at each LRS angular node from
the reconstructed augmented distribution `f(q,mu)`, projected onto a
number-conserving elastic-scattering source that preserves the energy
moment, and projected back into PSTF modes.
AP32 extends that live LRS angular projection path to the electromagnetic
pair-process reference: `nu_e`, `anti-nu_e`, and the effective `nu_x`
bank now evaluate angular-node scattering plus pair-process sources with
number-closed scattering component diagnostics and pair-process moment
diagnostics before projection back into PSTF modes.
AP33 extends the same live LRS angular projection path to the diagonal
no-QKE `nu-nu` redistribution reference, including weighted bank-energy
diagnostics before projection back into PSTF modes.
AP34 lifts the AP31-AP33 bridges to generic angular-node v2 closure
contracts and locks the same collision-reference projection on the staged
non-LRS S2 `{monopole, W_plus, W_minus}` basis.  This closes AP6 as a
stage-scoped deterministic collision-reference deliverable without
promoting runtime coupling.
AP35 connects those deterministic angular collision references to the
existing AP18 thermo-feedback interface by adding an explicit angular
collision `Augmented3TCollisionThermoSource` factory.  The factory
composes the live angular electromagnetic bridge and the live angular
diagonal no-QKE `nu-nu` bridge into `dQ_nue_pair_N` and `dQ_nux_bank_N`
moments with numeric component diagnostics, and the 3T shell source
evaluator accepts it as an opt-in callback.  A smoke-scale
`run_augmented_lrs_angular_collision_weak_network_3T_solve(...)` wrapper
now composes that source directly into the LRS AP15/AP18 solve path with
explicit frozen-initial-state or live-RHS source-update policy.  This is
not a default or promoted collision-coupled solve.
AP36 makes that callback visible in the deterministic validation surface:
the collision-feedback artifact path, candidate gate, span-ladder gate,
and convergence runners now accept an explicit `angular` source variant
with AP35 diagnostics.  The LRS artifact route now calls the AP35
`run_augmented_lrs_angular_collision_weak_network_3T_solve(...)` direct
wrapper with explicit frozen-initial-state or live-RHS source-update
policy, and the smoke-scale artifact path records real wrapper output.
Default/promoted full-span angular feedback remains a blocker.
AP37 advances the AP4 non-LRS side from static source projection to a
source-only coevolution shell: `Sigma_+`, `Sigma_-`, and the staged S2
`{monopole, W_+, W_-}` augmented modes are packed into a SciPy
`solve_ivp` state, the live distribution is reconstructed on every RHS
call, and `Pi_+`/`Pi_-` stress moments feed the diagonal shear equations.
This is still source-only transport; nonlinear non-LRS angular advection,
q-cascade transport, collisions, weak/network coupling, and public
dispatch remain outside the shell.
AP38 adds the corresponding fixed-thermo weak/network shell for that
source-only non-LRS path.  It extracts live `nu_e` and `anti-nu_e`
angular monopoles from the S2 distribution, records CL3 plus/minus
metadata, computes live weak rates, and feeds the PRIMAT network
derivative into the same `d/dN` state.  AP38 initially kept the non-LRS
angular weak-rate terms metadata-only; AP60 later promotes the S2
moment-input correction to the staged non-LRS correction-level-3 default,
with explicit `metadata_only` retained for controls.  No public dispatch,
promotion-grade full anisotropic weak-rate integration, or
collision-sourced thermodynamics is promoted.
AP39 extends that non-LRS source-only path to the staged 3T
thermo/Hubble shell: `T_gamma`, `T_nu_e`, and `T_nu_x` are packed into
the state, `H(T_gamma,T_nu_e,T_nu_x,Sigma_+^2+Sigma_-^2)` is recomputed
each RHS call, and the standard 3T table thermo RHS is coevolved with
the source-only transport and live weak/network blocks.  Collision-moment
thermodynamics and nonlinear non-LRS transport remain blockers.
AP40 adds the matching opt-in collision-moment thermo feedback hook for
that non-LRS 3T shell.  A caller-supplied source callback receives the
live S2 state, `Sigma_+`, `Sigma_-`, q-grid weights, the S2 grid object,
temperatures, and Hubble rate, and must return
`Augmented3TCollisionThermoSource`; only then does the shell use
`coupled_3T_rhs_from_collision_moments(...)`.  If the same source carries a
projected `dA_modes` collision term, the LRS, source-only non-LRS, and
nonlinear non-LRS 3T shells add that term to the augmented hierarchy RHS.
No non-LRS collision source is default or promoted yet.
AP41 provides the first non-LRS S2 angular collision thermo-source
factory for that hook by wrapping the generic AP35 angular source on the
`{monopole, W_+, W_-}` grid.  It composes the AP32/AP33 angular
electromagnetic and diagonal no-QKE `nu-nu` bridges, preserves their
component diagnostics, and records `Sigma_-` plus S2 grid context for
the callback path.  The source remains opt-in.
AP42 wires collision sources into a deterministic non-LRS 3T
collision-feedback artifact runner.  The artifact compares the AP39
standard table RHS with the AP41 `angular` source variant and the AP6
descriptor-driven `pstf_radial` source variant through the AP40 callback
hook, records S2 grid metadata, `Sigma_-` diagnostics, standard-relative
observable deltas, and source contracts, and remains JSON-ready for smoke
reports.  The smoke default freezes the initial collision moments for
runtime stability; explicit `live_rhs` source re-evaluation is available
under the existing source-policy/budget controls for experiments and is
not a promoted default.
AP43 adds non-LRS collision-feedback convergence runners over `N_q`,
`N_mu`, and `N_phi` ladders by extracting AP42 artifact observables for a
selected source variant.  These runners record plus/minus stress
observables, thermo/network outputs, selected standard-relative deltas,
source moments, and solve effort for smoke-scale convergence reports.
They remain diagnostic convergence surfaces, not promotion-tolerance
full physical collision/BBN convergence.
AP44 adds the matching non-LRS collision-feedback candidate gate around
the AP42 artifact.  It checks the standard/angular source variants for
selected standard-relative thermo/network deltas, plus/minus shear and
stress limits, source-moment magnitude, mode RMS values, and solve
effort.  This is an AP8 diagnostic gate only; it does not make non-LRS
collision feedback default and does not promote nonlinear non-LRS
transport, full anisotropic weak rates, or public dispatch.
AP45 adds a deterministic non-LRS source-update policy artifact around
the same AP42 path.  It compares `frozen_initial_state` against
`live_rhs`, records per-variant solve effort and observable deltas
relative to the frozen reference, and uses a very short LSODA smoke span
by default so live source re-evaluation is actually exercised without
turning it into a promoted full-span runtime.
AP46 adds the corresponding source-update policy candidate gate.  It
consumes the AP45 artifact, evaluates frozen/live policies and
standard/angular variants, and checks selected live-vs-frozen
thermo/network deltas, collision source-moment deltas, absolute source
moments, and solve effort.  It is a smoke gate for the opt-in policy,
not a default-policy or full-span promotion.
AP47 adds a direct opt-in transport wrapper,
`run_augmented_nonlrs_angular_collision_weak_network_3T_solve(...)`, that
wires the AP41 S2 angular collision source through the AP40 non-LRS 3T
collision-moment hook with explicit `live_rhs` or `frozen_initial_state`
source update policy selection.  This makes the staged angular
collision-feedback solve path callable outside validation artifacts while
preserving the no-public-dispatch and no-default-source boundary.
AP48 moves the non-LRS collision-feedback artifact's `angular` variant onto
that wrapper, so AP42/AP45/AP46 artifact and gate surfaces exercise the same
transport-level AP41/AP40 integration path rather than a duplicate
validation-local source construction path.
AP49 adds a direct deterministic JSON artifact and CLI runner for the AP47
wrapper itself, with smoke-scale LSODA/live-RHS defaults and explicit
source-contract, observable, source-diagnostic, and solve-effort reporting.
AP50 adds a direct candidate gate over that AP49 artifact, checking source
moments, plus/minus shear/stress, A-mode RMS, temperature/Hubble/network
bounds, and solve effort while staying smoke-scale and opt-in.
AP51 adds the first direct-wrapper outcome policy over AP49/AP50: smoke,
medium, and production-span diagnostic presets, wall-time metadata, explicit
success/stiffness/bound/timeout/invalid-output classifications, a JSON writer,
and a CLI artifact runner for downstream convergence, plotting, and SMC
filters.
AP52 adds the pre-budget direct-wrapper solver matrix over AP51 outcomes,
including LSODA/Radau method comparisons, tolerance/span/source-policy
metadata, selected terminal-observable deltas, classified failure propagation,
and a JSON/CLI artifact surface for choosing a candidate solver policy before
AP55 source-budget closure.  AP53 adds the source-update policy promotion
study that consumes AP42/AP45 artifacts and AP51 direct-wrapper outcomes,
keeps `frozen_initial_state` classified as a diagnostic fallback, and only
marks `live_rhs` as AP56-eligible when all selected surfaces stay within
limits.  AP54 adds pre-budget direct-wrapper convergence ladders over
`N_q`, `N_mu`, and `N_phi`, reusing the AP51 outcome path and the existing
resolution convergence report contract while keeping every label diagnostic
until AP55 source-budget closure.  AP55 closes that source-budget blocker for
the staged deterministic no-QKE moment source by adding a JSON/CLI artifact
over FLRW quiet, LRS electron-pair heating, LRS `nu-nu` redistribution,
LRS fixed-`mu_e`, LRS charge-neutral, non-LRS fixed-`mu_e`, and non-LRS charge-neutral AP6 `pstf_radial` process-budget cases, non-LRS S2
LRS-limit, non-LRS minus-mode, and non-LRS quiet cases with finite source moments,
component-sum closure, weighted-energy/number residuals, sign,
finite-mass radial process markers, kinetic `dA` hierarchy amplitude,
algebraic charge-neutral finite-mass e-/e+ bath diagnostics,
non-LRS radial S2 context markers, and source-context checks.  AP56 aggregates AP42/AP49/AP51-AP55 evidence
into one diagnostic candidate bundle with source routing, solver/source-policy
decision, direct convergence, and source-budget summaries, including the AP55
LRS/non-LRS fixed-`mu_e` and charge-neutral radial case lists, source contracts, e-/e+ charge-neutral context, S2 context markers, and selected AP6 radial observables.  The current smoke
bundle is completed and ready for AP57 input but still records
`diagnostic_fallback_only` source-policy status when `live_rhs` is not
AP56-eligible.  AP57 adds the corresponding physical sanity matrix over
AP56/AP55 evidence, checking bundle readiness, source-policy boundary,
terminal thermo/network/source bounds, plus/minus bounded response, FLRW
quietness, LRS sign/energy budget, AP6 radial source-budget markers and
amplitudes, LRS/non-LRS charge-neutral AP6 radial source-budget markers and finite-mass e-/e+ positivity, non-LRS AP6 radial source-budget markers and amplitudes, and non-LRS S2 reduction/context.  AP58
upgrades the existing weak bridge with explicit LRS/non-LRS per-q angular
weak-rate input objects, approximation/source metadata, and fail-closed
angular-mode resolution.  AP59 now wires the LRS side of that contract into an
opt-in `lrs_cl3_quadrupole_input` rate mode with species-aware `lambda_np` and
`lambda_pn` factors from current per-q plus moments, while preserving the
legacy `Sigma_+ K_2` multiplier as the default bounded subcase.  AP60 extends
the same moment-input rate path to the non-LRS S2 plus/minus basis through
`nonlrs_s2_cl3_quadrupole_input`, with zero-minus LRS reduction locked.  The
staged non-LRS correction-level-3 weak/network config now defaults to that
current S2 moment-input application; explicit `metadata_only` remains the
same-CL3 control/baseline mode.
AP61 adds the deterministic rate-only weak-rate convergence/candidate gate for
those LRS/non-LRS moment-input factors, including q/angular ladder deltas,
factor sign/range checks, JSON report output, and unsupported-mode diagnostics.
AP62 adds the first non-LRS S2 nonlinear collisionless transport RHS scaffold,
using the landed LRS `jax/nonlinear_transport.py` equation as the reduction
oracle and projecting the q/mu/phi nodal RHS back into the staged
`{monopole, W_+, W_-}` mode basis.  It is not yet a solve shell or coupled
BBN path.
AP63 adds the SciPy nonlinear transport solve shell around that RHS with live
shear-stress feedback, finite short-span trajectories, FLRW quietness, and
`Sigma_-=0` reduction, while still excluding weak/network and collision
feedback.
AP64 couples that nonlinear non-LRS transport solve to the existing live
weak/network and 3T thermo/Hubble shell under an explicit
`transport_model` routing flag, locking dynamic Hubble, AP60 moment-input
weak-rate application, abundance normalization, plus/minus stress metadata,
and source-only versus nonlinear routing without collision-sourced thermo
feedback.
AP65 adds the opt-in nonlinear angular collision-feedback 3T wrapper, reusing
the AP41 angular source builder on the AP64 nonlinear grid and recording
collision-moment thermo feedback metadata while preserving live/frozen
source-update policy controls.  The angular source's projected `dA_modes`
collision term now also enters the AP64 nonlinear hierarchy RHS when present,
so the nonlinear angular candidate is no longer thermo-only.  The AP65
radial follow-up routes the AP6 descriptor-driven `pstf_radial` source
through the same nonlinear S2 shell with budgeted live-RHS diagnostics
and finite smoke-scale source moments.  Because the radial AP18 adapter now
returns `dA_modes`, the same wrapper receives radial `C_modes` hierarchy
feedback in addition to thermo moments.  The AP65 combined-source follow-up
then runs the AP41 angular source and AP6 `pstf_radial` source together in one
nonlinear RHS callback without double-counting the same no-QKE collision
family: AP41 angular moments remain diagnostic component observables, while
the AP6 finite-mass radial source supplies the effective 3T thermo moments and
compatible `dA_modes` payloads, preserving radial source-evaluation budgets.  The
radial and combined wrappers also forward the AP6 momentum-delta controls, so
the default normalized `unit_direction_gaussian` closure and opt-in
`radial_gaussian` p-dependent closure can both run inside the AP65 nonlinear
solve path.  A
smoke combined-source run at `N_span=(0, 1e-14)`, `N_q=3`, `N_mu=3`,
`N_phi=5`, `standard_3t_plasma`, and `RK23` returned `success=true`, `nfev=5`,
`source_contract=combined_nonlrs_angular_pstf_radial_collision_thermo_source_v2`,
`collision_dQ_nue_pair_N_final=1.0831195293549981e-4`,
`collision_dQ_nux_bank_N_final=2.8944175873527294e-4`,
`collision_dA_abs_max_final=3.3626741471031303e-4`,
`combined_angular_dA_abs_max=3.5318831933493763e-22`, and
`combined_pstf_radial_dA_abs_max=3.3626741471031303e-4`.
The AP4/AP65 combined full-span candidate follow-up now consumes that combined
artifact through `run_augmented_nonlrs_combined_full_span_3t_candidate_gate(...)`
and a JSON/CLI writer over explicit span ladders.  The first smoke gate at
`N_span=(0, 1e-14)`, `frozen_initial_state`, `standard_3t_plasma`, and `RK23`
returned `success=true`, `T_gamma_final=0.7999999999999922`,
`H_rate_s_final=0.4315487123652324`, `Xn_final=0.13000000000065723`,
`collision_dA_abs_max_final=2.8419082353054516e-4`, component radial/angular
`dA` diagnostics, and `nfev=5`.
The same gate also has real `live_rhs` two-span evidence over
`N_span=(0, 1e-14)` and `(0, 1e-12)` with a `256` radial source-evaluation
budget.  Both rows passed with `source_evaluations=7`, budget diagnostics
present and passing, and the longer row recorded
`T_gamma_final=0.7999999999992143`,
`H_rate_s_final=0.43154871236438125`,
`Xn_final=0.13000002470684513`, and
`collision_dA_abs_max_final=2.841908235303566e-4`.
The gate now also has a deterministic `warm` preset for the longer diagnostic
live-RHS ladder `N_span=(0, 1e-12)`, `(0, 1e-10)`, and `(0, 1e-8)` with
`max_pstf_radial_source_evaluations=2048` and `max_nfev=50000`.  A real
warm-preset CLI run passed with `span_count=3`, `max_span_length=1e-8`,
`source_evaluation_max=70`, and
`collision_dA_abs_max_final=3.362674147101354e-4`.
The gate now also requires AP6 conserved-moment closure diagnostics from the
combined-source payload itself.  A real `live_rhs` two-span RK23 smoke over
`N_span=(0, 1e-14)` and `(0, 1e-12)` passed with all-nine diagonal `nu-nu`
radial number projection enabled, `9` projected number sources, `6`
off-diagonal projected number sources, `6` off-diagonal pair-energy projected
sources, `3` unordered projected pairs, and residuals
`radial_nunu_max_abs_number_moment_final=1.0508502501873664e-20` and
`radial_offdiagonal_nunu_pair_max_abs_energy_residual_final=7.707999820014133e-20`.
The gate fails closed if the off-diagonal pair-energy closure marker is
missing.  This records the landed AP6 radial closure inside the AP4/AP65
full-span solve surface; it is still diagnostic and does not promote QKE,
public dispatch, production SMC, or promotion-grade full-BBN support.
The same gate now exposes an opt-in `physical_preview` preset for a longer
frozen-source diagnostic ladder using SciPy `Radau`.  The preset runs
`N_span=(0, 1e-6)`, `(0, 1e-4)`, and `(0, 1e-3)` with
`source_update_policy=frozen_initial_state`,
`max_pstf_radial_source_evaluations=64`, and `max_nfev=200000`.  A real
isolated CLI run passed in `elapsed_s=29.111393796047196`: the `1e-3` row
reported
`T_gamma_final=0.7992146754386976`, `H_rate_s_final=0.4306897631857426`,
`Xn_final=0.1927605271823484`, `nfev=10634`, and off-diagonal `nu-nu`
pair-energy residual `9.571472303973594e-20`.  The preset label is
fail-closed: CLI overrides of preset-defining fields are relabeled `custom`,
and direct spec construction rejects mismatched `physical_preview` contracts.
A frozen `1e-2` Radau probe failed with overflow/non-success terminal values,
and the frozen-source `1e-3` row was later found order-sensitive after a
source-refresh solve in the same process.  Routine frozen-source Radau numeric
coverage is therefore kept to the stable `1e-6` and `1e-4` rows, with artifact
inputs and CLI dry-runs explicitly separating `routine_numeric_gate_spans` from
`isolated_diagnostic_spans` for the preset, while CPU live-RHS evolution at
`1e-6` remains timeout-level in smoke settings.  This
remains a diagnostic physical-preview surface rather than promoted live-RHS
full-BBN evidence.
The same gate now also supports `source_update_policy=piecewise_frozen` with
explicit `source_update_subspan_ends`.  This recomputes the combined AP41
angular plus AP6 `pstf_radial` source at subspan boundaries from the current
state, freezes it within each subspan, and hands off Sigma/A/T/X to the next
subspan.  A real custom CLI run over `N_span=(0, 1e-4)` with subspan ends
`(5e-5, 1e-4)`, `Radau`, `max_pstf_radial_source_evaluations=8`, and
`max_nfev=5000` passed with `source_update_subspan_count=2`,
`source_evaluations=2`, `source_diagnostic_evaluations=1`, terminal source
diagnostics at `N=1e-4` after the last refresh at `N=5e-5`, `nfev=1397`,
`T_gamma_final=0.7999214320646801`, `H_rate_s_final=0.43146274191619854`,
`Xn_final=0.1300096609235235`,
`collision_dA_abs_max_final=0.00026527543966857903`, and AP6 pair-energy
residual `1.1784345878675453e-19`.  A longer nonuniform run over
`N_span=(0, 1e-3)` with subspan ends `(1e-6, 1e-4, 1e-3)` also passed with
`source_update_subspan_count=3`, `source_evaluations=3`,
`source_diagnostic_evaluations=1`, terminal source diagnostics at `N=1e-3`
after the last refresh at `N=1e-4`, `nfev=47`,
`T_gamma_final=0.7992146796753832`, `H_rate_s_final=0.43068978083203713`,
`Xn_final=0.13005888045355307`,
`collision_dA_abs_max_final=0.000265067738217371`, and AP6 pair-energy
residual `7.326834993749698e-20`.  This is a concrete source-refresh
diagnostic between frozen and fully live RHS, not promoted full-BBN support.
The same nonuniform source-refresh path is now exposed as the named
`piecewise_physical_preview` preset.  The preset resolves to
`N_span=(0,1e-4),(0,1e-3)`,
`source_update_policy=piecewise_frozen`,
`source_update_subspan_ends=(1e-6,1e-4,1e-3)`, `method=Radau`,
`max_pstf_radial_source_evaluations=8`, and `max_nfev=10000`, with dry-run
metadata separating routine `N_span=(0,1e-4)` from isolated diagnostic
`N_span=(0,1e-3)` and declaring supported electron-bath modes
`[fixed, charge_neutrality]` plus scalar-QED models
`[finite_mu_scaled, exact_finite_mu_scalar]`.  A real named-preset artifact run passed with
`span_count=2`, `source_evaluation_max=3`, `radial_grid_cache_entries=45`, and
no violations; the `1e-4` row reported
`T_gamma_final=0.7999214320680265`, `Xn_final=0.13000591435767977`, and
`nfev=31`, while the `1e-3` row reported
`T_gamma_final=0.7992146796753832`, `Xn_final=0.13005888045355307`, and
`nfev=47`.  The same named preset also passed with
`electron_chemical_potential_mode=charge_neutrality` through `N_span=1e-3`,
recording `electron_chemical_potential_MeV_final=3.295370971985368e-10`,
`electron_charge_asymmetry_density_MeV3_final=6.59880533199606e-11`, and
`source_update_charge_asymmetry_state_handoff=1`.  With
`qed_correction_model=exact_finite_mu_scalar`, the preset passed through
`N_span=1e-3` with `qed_correction_model_exact_finite_mu_scalar=1`,
`T_gamma_final=0.7992145483449656`, `Xn_final=0.13005893883614722`, and
`nfev=47`.  The combined charge-neutral plus exact scalar-QED row also passed
through `N_span=1e-3` with
`electron_chemical_potential_MeV_final=3.295370300138031e-10`,
`electron_charge_asymmetry_density_MeV3_final=6.598801808206643e-11`, and
`qed_correction_model_exact_finite_mu_scalar=1`.  This keeps the
source-refresh preview reproducible without
promoting public dispatch, production SMC, QKE, or full-BBN support.
The source-refresh path also has a diagnostic refinement artifact comparing
coarse `(1e-6,1e-4,1e-3)` and refined `(1e-6,1e-5,1e-4,1e-3)` subspan
schedules over `N_span=(0,1e-3)`.  A real refinement run passed with
`source_evaluation_max=4`, `nfev_max=56`, `radial_grid_cache_entries=72`, and
refined-minus-coarse deltas `T_gamma_final=-1.1280976153216216e-12`,
`Xn_final=2.3314683517128287e-15`, and
`collision_dA_abs_max_final=4.4915406029882865e-14`; this is still
operator-split refinement evidence, not continuous live-RHS collision
coupling.
The `piecewise_frozen` terminal source re-evaluation now forwards the selected
`qed_correction_model` and records finite/exact scalar-QED one-hot diagnostics
on the returned combined source.  A real charge-neutral plus
`exact_finite_mu_scalar` named-preset run still passed with
`source_evaluation_max=3`,
`electron_chemical_potential_abs_max=3.298436883301101e-10`, and no
violations.  This closes terminal scalar-QED diagnostic forwarding only; it is
not anisotropic/tensor QED response or promotion-grade full-BBN support.
The same piecewise path now accepts charge-neutral finite-mass e-/e+ evolution
by handing off the evolved `electron_charge_asymmetry_density_MeV3` state
between subspans and into terminal source diagnostics.  A real charge-neutral
`N_span=(0, 1e-4)` run with subspan ends `(5e-5, 1e-4)`, `Radau`,
`max_pstf_radial_source_evaluations=8`, and `max_nfev=10000` passed with
`source_update_charge_asymmetry_state_handoff=1`,
`source_evaluations=2`, `source_diagnostic_evaluations=1`, `nfev=8256`,
`electron_chemical_potential_MeV_final=3.298132792363573e-10`,
`electron_charge_asymmetry_density_MeV3_final=6.61672994731945e-11`,
`T_gamma_final=0.7999214313698554`, `Xn_final=0.13000851100280264`,
`collision_dA_abs_max_final=0.00026527544839372085`, and AP6 pair-energy
residual `1.1784345878675453e-19`.  This closes the previous fixed-only
piecewise limitation at smoke scale, while keeping the full-span claim
operator-split and diagnostic.
The companion combined-source source-policy span profile now compares
frozen-initial-state and `live_rhs` rows over the same spans in one artifact.
The first real profile passed all four rows with frozen/live `nfev=5/5`,
frozen/live source evaluations `1/7`, no failed rows, maximum
`collision_dA_abs_max_final=2.8419082353054516e-4`, and only roundoff-scale
live-minus-frozen thermo/network deltas.
The same AP4/AP65 full-span gate/profile surface now forwards
`electron_chemical_potential_MeV` and `electron_chemical_potential_mode` into
the AP65 combined nonlinear artifact.  This exposes the already landed
finite-mass e-/e+ fixed-`mu_e` and charge-neutrality radial bath route from the
combined full-span diagnostic surface.  A charge-neutral live-RHS gate smoke
run at `N_span=(0, 1e-14)` passed with `source_evaluation_max=7`,
`collision_dA_abs_max_final=2.8419082353048944e-4`, and no violations; the
charge-neutral frozen/live source-policy profile over `(0, 1e-14)` and
`(0, 1e-12)` passed all four rows with frozen/live source evaluations `1/7`
and no failed rows.
The AP65 combined artifact and AP4/AP65 gate/profile now also propagate
terminal electron-bath observables from the nonlinear solve.  A charge-neutral
AP65 combined smoke artifact reported
`electron_chemical_potential_MeV_final=3.3006028673558046e-10` and
`electron_charge_asymmetry_density_MeV3_final=6.623064974048602e-11`; the
AP4/AP65 full-span gate/profile summaries over the same charge-neutral surface
reported `electron_chemical_potential_abs_max=3.298439955909406e-10` and
`electron_charge_asymmetry_density_abs_max_MeV3=6.618724826621509e-11`.
The AP17/AP36 LRS 3T convergence/collision-feedback surfaces, AP42/AP45
source-only non-LRS surfaces, the AP49 direct angular artifact,
direct/nonlinear radial AP6/AP65 artifacts, AP65 combined artifact, and
AP4/AP65 full-span gate/profile now route the scalar
`qed_correction_model` control through the staged 3T thermo shells.
The first exact-scalar-QED AP4/AP65 tiny-span gate smoke at
`N_span=(0, 1e-14)` passed with `T_gamma_final=0.7999999999999922`,
`H_rate_s_final=0.4311207244202148`, and explicit exact-model observables.
This is diagnostic scalar EOS routing only; anisotropic/tensor QED and
promotion-grade full-span exact-QED validation remain blockers.
The same AP4/AP65 full-span gate/profile now reuses a shared AP6 radial-grid
cache across span rows and frozen/live policy rows.  The live collision source
is still evaluated on the fly from the current augmented distribution; only
invariant descriptor/radial-grid construction is reused.  A two-span live-RHS
comparison over `(0, 1e-14)` and `(0, 1e-12)` measured
`separate_cache_s=5.539025628997479`,
`shared_cache_s=3.3420192470075563`, `speedup=1.6573889076064183`, and
`shared_cache_entries=18`.
AP66 adds the deterministic publication-candidate convergence matrix around
that AP65 path, reusing the existing resolution-convergence report contract
over `N_q`, `N_mu`, `N_phi`, span, solver tolerance, source-model,
source-policy, weak-rate-mode, scalar QED model, AP6 radial momentum-delta
model/sigma, and fixed/charge-neutral electron-bath rows while recording terminal
thermo/network/source observables, the final collision-source `dA_modes`
amplitude, combined angular/radial component `dA` amplitudes, radial
source-budget observables and momentum-delta controls/provenance, terminal
electron-mu/charge-asymmetry observables, QED model markers, first-converged candidate settings, and residual
risks.  A smoke AP66 combined-source matrix at `N_span=(0, 1e-14)`,
`source_model=combined_angular_pstf_radial`, `live_rhs`,
`nonlrs_s2_cl3_quadrupole_input`, `standard_3t_plasma`, and `RK23` converged
the `N_q=(3,4)`, `N_mu=(3,4)`, and `N_phi=(5,6)` ladders with candidate
`N_q=4`, `N_mu=4`, `N_phi=6`; the `N_q=4` row recorded
`collision_dQ_nue_pair_N_final=1.0774167274668752e-4`,
`collision_dQ_nux_bank_N_final=2.1548334549332648e-4`, and
`collision_dA_abs_max_final=9.013121610643922e-4`.
The charge-neutral variant of the same AP66 matrix converged with the same
candidate settings and row observables including
`electron_chemical_potential_MeV_final=3.298439955909307e-10`,
`electron_charge_asymmetry_density_MeV3_final=6.61872482662131e-11`, and
`electron_charge_asymmetry_state_evolved=1.0`.
The AP66 matrix can also now run `qed_correction_model=exact_finite_mu_scalar`
and records row-level exact-scalar versus finite-mu-scaled model markers.  This
extends the exact scalar QED control into the publication-candidate convergence
matrix but remains diagnostic; anisotropic/tensor QED and promotion-grade
full-span exact-QED validation are still blockers.
The AP66 matrix can now also consume the AP4/AP65 `piecewise_frozen`
combined-source gate for `combined_angular_pstf_radial` rows, forwarding
explicit subspan ends and recording source-refresh observables in the same
convergence-report contract.  The AP4 full-span spec now preserves `Xn0` and
`weak_rate_mode` plus radial momentum-delta model/sigma controls in those
actual solve rows.  A real AP66 q-ladder smoke over
`N_span=(0,1e-14)`, subspan ends `(5e-15,1e-14)`, `metadata_only`, `RK23`,
and `standard_3t_plasma` passed with `source_update_subspan_count=2`,
`pstf_radial_source_evaluations=2`, positive terminal `T_gamma`, physical
`Xn_final`, and nonzero `collision_dA_abs_max_final`.  A full short-span
piecewise AP66 matrix over q/`N_mu`/`N_phi` produced candidate `(4,4,6)` with
`collision_dA_abs_max_final=0.0010936512219095985`,
`T_gamma_final=0.7999999999999923`, and
`Xn_final=0.13000000000019393`.  This is still operator-split diagnostic
evidence, not public dispatch.  FB-10 extends the same AP66 artifact family so
it can consume the FB-09 chained full-BBN resolution artifact directly:
`--chained-resolution-artifact` loads the chained artifact, AP66 validates the
FB-09 contract plus passed/no-QKE/not-public-dispatch/not-production-SMC scope,
rejects failed or malformed rows, and records `full_chain_evidence.row_links`
with source artifact path/contract, row index/key, terminal `Yp`, `D/H`,
`N_eff_3T`, `Sigma_H`, source-evaluation and nfev totals, and CPU-JAX/Rodas5P
replay metadata for each chained `N_q`/`N_mu`/`N_phi` row.  These links make
AP66 a real consumer of chained-window evidence while preserving the staged
claim boundary; the combined AP66 evidence remains publication-candidate
evidence, not live-RHS full-BBN or public dispatch.
AP67 adds the deterministic known-limit validation atlas, reusing AP65 solves
and AP66 convergence evidence across FLRW/null, LRS, non-LRS, injected-moment,
and short-span stress rows with explicit output-ready, convergence-ready, and
inference-ready labels.  AP67 now preserves the AP65 source model for each
case, can run combined angular+`pstf_radial` atlas rows with radial
energy-normalization, source-budget, and momentum-delta controls, links the
AP66 candidate source model and radial momentum-delta settings in reused
evidence metadata, forwards fixed/charge-neutral electron-bath controls into
both AP65 case solves and nested AP66 evidence, forwards scalar-QED model
selection including `exact_finite_mu_scalar` into both AP65 case solves and
nested AP66 evidence,
records terminal electron-mu/charge-asymmetry observables plus scalar-QED
markers for atlas rows, uses
the same CL3 weak-rate control basis as the AP66 matrix rows, preserves AP66
`source_update_policy`, `source_update_policies`, and
`source_update_subspan_ends` evidence-link provenance for AP4-backed
`piecewise_frozen` source-refresh rows, records AP66/FB-09 full-chain artifact
contract/path/row-count/row-key/no-QKE/not-public provenance when a chained
resolution artifact is supplied, and records budget exhaustion as failed rows
rather than aborting the atlas.  FB-11 adds a fail-closed
`require_full_chain_evidence` mode plus CLI `--chained-resolution-artifact`;
a CPU smoke with the FB-09 artifact produced six passing AP67 known-limit cases,
`limit_violations=[]`, and `readiness_summary.full_chain_evidence_ready=true`.
A real AP67 combined-source charge-neutral atlas run over the six default
cases with `standard_3t_plasma` and nested AP66 skipped reported
`limit_violations=[]`, `output_ready=true`,
`electron_chemical_potential_MeV_final=3.2984399559038986e-10`,
`electron_charge_asymmetry_density_MeV3_final=6.618724826601604e-11`, and
`electron_charge_asymmetry_state_evolved=1.0`.
AP68 adds a guarded inference-facing adapter around the AP65 candidate solve,
mapping the staged result into the existing `ForwardModel`/`BBNLikelihood`
surface with failure metadata and without canonical/public dispatch.  AP68 now
also exposes the AP65 source model to inference calls, so guarded predictions
can run the angular-only or combined angular+`pstf_radial` solve path while
recording radial energy-normalization, source-budget controls, AP6 radial
momentum-delta model/sigma, fixed/charge-neutral electron-bath controls,
scalar-QED model selection including `exact_finite_mu_scalar`, AP4-style
`piecewise_frozen` operator-split source refresh over explicit subspan ends,
aggregate subspan/source-evaluation metadata, budget failures, terminal
collision-source `dA` metadata, terminal electron-mu/charge-asymmetry
observables, scalar-QED model/contract markers, and the mode-specific
collision-feedback contract.  FB-12 extends AP68 with
`execution_mode="full_chain"`, which builds the FB-04 chained runner spec from
the guarded AP68 parameter/config surface, supports cached chained artifacts,
validates passed/no-QKE/not-public/not-production-SMC artifact scope, and maps
finite chained terminal `Yp`/`D/H` plus window/replay/source metadata into the
same `BBNPrediction` interface.  AP68 can also forward the optional FB-04
live-source RHS chain diagnostic switch, can opt into the live-source chain as
the repeated-run readout source, and preserves the resulting full-chain summary
metadata without registering public dispatch.  AP68 can now also build or
consume the FB-21 live-source repeated-run gate as optional diagnostic evidence,
recording the gate contract/path/pass status, collision-payload counts and
provenance fingerprints, finite BBN readouts, and
live-source-vs-piecewise/window-map deltas in prediction and SMC metadata while
leaving the terminal AP68 `Yp`/`D/H` values unchanged.  A real
short-span combined-source charge-neutral AP68 prediction reported
`success=True`, `electron_chemical_potential_MeV_final=3.300602867355805e-10`,
`electron_charge_asymmetry_density_MeV3_final=6.623064974048603e-11`,
`electron_charge_asymmetry_state_evolved=True`,
`collision_dA_abs_max_final=2.2350365034067905e-4`, and
`pstf_radial_source_evaluations=7`.
A real AP68 piecewise-frozen combined-source smoke over
`N_span=(0,1e-14)` and `source_update_subspan_ends=(5e-15,1e-14)` reported
`success=True`, `source_update_subspan_count=2`, aggregate `nfev=10`,
`pstf_radial_source_evaluations=2`,
`collision_dA_abs_max_final=0.0002652986559886625`,
`T_gamma_final_MeV=0.7999999999999923`, and
`Sigma_plus_final=0.0099999999999999`.
A real AP68 full-chain smoke over two `(1e-8)` windows reported
`success=True`, finite `Yp`/`D/H`, `full_chain_completed_windows=2`,
`nfev=18`, `full_chain_bbn_observables_present=True`,
`public_dispatch_ready=False`, and `qke_scope=out_of_scope`.
AP69 adds the augmented SMC likelihood schema, including continuous
`Sigma_H`/`Sigma_H_minus`/`eta`/`tau_n` parameter specs, observational priors,
fixed source-policy/source-model/radial-normalization/radial-momentum-delta/
weak-rate/solver controls plus AP68 `piecewise_frozen` subspan ends,
electron-bath mode, scalar-QED model, AP68 `execution_mode`, and full-chain
window/cache/source-refresh/replay/restart controls, AP68
forward-config provenance including radial source-budget, momentum-delta
settings, source-refresh subspan controls, electron-bath controls, scalar-QED
controls, full-chain controls, and the FB-36 live-source collision-payload
refresh selector, vector/dict coercion, and AP68 failure propagation.
The schema can now drive the combined
angular+`pstf_radial` AP68 path at smoke scale while retaining diagnostic-only
status.  A real short-span combined-source charge-neutral likelihood reported
`log_likelihood=-5731.092605376938`,
`electron_chemical_potential_MeV_final=3.3006028673558036e-10`,
`electron_charge_asymmetry_density_MeV3_final=6.623064974048599e-11`,
`electron_charge_asymmetry_state_evolved=True`,
`collision_dA_abs_max_final=2.2350365034067905e-4`, and
`pstf_radial_source_evaluations=35`.
A real AP69 piecewise-frozen combined-source likelihood over
`N_span=(0,1e-14)` and `source_update_subspan_ends=(5e-15,1e-14)` reported
finite `log_likelihood=-5731.092605444093`,
`source_update_subspan_count=2`, aggregate `nfev=18`,
`pstf_radial_source_evaluations=2`, and
`collision_dA_abs_max_final=0.0002652986559886625`.
A real AP69 full-chain likelihood smoke through AP68
`execution_mode="full_chain"` reported finite
`log_likelihood=-5731.092605318141`, `full_chain_completed_windows=2`,
`public_dispatch_ready=False`, and `qke_scope=out_of_scope`.
AP70 adds a smoke-scale tempered-SMC runner over the AP69 schema/likelihood,
with replayable seeds, explicit temperature schedules, ESS resampling,
optional random-walk rejuvenation, initial-particle support, and failure
accounting.  AP70 result metadata now carries the AP69
source-model/source-refresh/radial/electron-bath/scalar-QED control values and AP68
forward-config provenance, including radial momentum-delta model/sigma settings
and `piecewise_frozen` source-refresh subspan ends plus full-chain
execution/window/cache/source-refresh and live-source collision-payload
refresh-mode controls,
so downstream validation and artifact builders do not lose the selected
collision-source model, source-refresh schedule, radial closure,
electron-bath mode, scalar-QED model, or full-chain mode.
A tiny real AP70 run
with two supplied particles and the combined-source charge-neutral likelihood
reported `complete=True`, `electron_chemical_potential_mode=charge_neutrality`,
`qed_correction_model=exact_finite_mu_scalar`, `finite_loglike_count=2`,
`forward_failures=0`, normalized weights
`[0.5000000000059117, 0.4999999999940883]`, and terminal
`collision_dA_abs_max_final=7.360025983455233e-05`.
A real full-chain CLI SMC smoke with two particles and temperatures `(0,1)`
reported `complete=True`, `execution_mode=full_chain`,
`finite_loglike_count=2`, `forward_failures=0`, `cache_misses=2`,
`full_chain_window_edges=[0,1e-8,2e-8]`, `public_dispatch_ready=false`, and
`qke_scope=out_of_scope`.
AP71 adds runtime/cache controls for that runner: duplicate-call caching,
batched likelihood hooks, stage checkpoint/resume, incompatible-manifest
rejection, portable run manifests, failure-metadata preservation, source-model
and source-refresh/radial/electron-bath-control-aware cache keys,
scalar-QED/full-chain-aware cache context separation, explicit source-refresh,
electron-bath, scalar-QED, and full-chain cache context/runtime payload fields, and a
diagnostic CLI runner that forwards AP69 source/source-refresh/radial/electron-bath
controls plus `N_span` end, scalar-QED, radial momentum-delta, and
`piecewise_frozen` subspan controls plus full-chain execution/window/cache/source-refresh/replay/restart and live-source collision-payload refresh controls in dry-run metadata.  The runtime manifest also preserves the
last successful AP68 prediction metadata, so smoke SMC artifacts can expose
concrete source-refresh diagnostics such as subspan counts, radial source
evaluation counts, and terminal collision-source amplitudes without rerunning
the likelihood.
AP72 adds a synthetic SMC validation suite over deterministic FLRW/null and
non-LRS injection targets, recording posterior summaries, ESS and acceptance
diagnostics, a logZ-error proxy, forward-success accounting, and pass/fail
thresholds.  AP72 can now carry AP69
source-model/source-refresh/`N_span`/radial/electron-bath-control provenance
plus scalar-QED provenance and radial momentum-delta provenance via
schema overrides and CLI dry-run schema payloads, while keeping the default
artifact analytic synthetic sampler validation.  FB-15 adds an opt-in physical
full-chain smoke row that calls the AP68 `execution_mode="full_chain"`
likelihood from AP70/AP71 SMC, records AP68 terminal `Yp`/`D/H`, completed
chained windows, and CPU-JAX/Rodas5P replay status.  It can now request the
FB-04 live-source RHS chain as the repeated-run BBN readout and records the
readout source, ready flag, and replay target in physical-smoke diagnostics.
When the FB-21 live-source repeated-run gate is requested or present, AP72 also
requires and preserves the current gate contract, diagnostic-only claim scope,
no-public/no-production/no-QKE flags, finite repeated-run BBN readouts,
same-window gate counts, finite comparison deltas, and supplied/applied
frozen-collision payload counts/provenance fingerprints or
`dynamic_restart_state` payload request/build/provenance fingerprints.
The row remains diagnostic rather than production SMC or public dispatch
evidence.
AP73 adds versioned figure-ready publication artifact tables over AP66/AP67/AP72
and existing Schramm/likelihood cache rows, with provenance, units, registry
keys, stale/mixed-contract rejection, and
source-model/source-refresh/`N_span`/radial/electron-bath-control context plus
radial momentum-delta model/sigma provenance in convergence, validation, and
SMC-derived rows.  AP72-derived SMC posterior and temperature-trace rows now
also preserve `piecewise_frozen` source-refresh subspan context for downstream
plot inputs.  AP66
convergence rows and AP67 validation rows now also preserve categorical
`qed_correction_model` provenance and finite-mu-scaled versus
exact-scalar-QED markers for downstream plot inputs without promoting
exact-QED full-BBN evidence.  FB-16 extends AP73 so passed AP72 physical
full-chain smoke artifacts can generate diagnostic Schramm rows with AP68
terminal `Yp`/`D/H`, `Sigma_H`, `eta10`, `N_eff_3T`, completed-window count,
CPU-JAX/Rodas5P replay status, and optional live-source repeated-run BBN readout
provenance.  When the AP72 smoke requests
`full_chain_rodas5p_repeated_run_source="live_source_rhs_chain"`, AP73 requires
the AP72 `live_source_repeated_run_readout` check to pass before it builds the
Schramm row.  If AP72 carries FB-21 gate evidence, AP73 validates the current
gate contract, diagnostic-only claim scope, no-public/no-production/no-QKE
flags, finite repeated-run BBN readouts, gate counts matching the AP72 smoke
windows, finite live-source-vs-piecewise/window-map deltas, and
supplied/applied/provenance-fingerprinted frozen-collision payloads before copying the gate
provenance into Schramm rows; non-synthetic AP72 artifacts without the passed
smoke row remain rejected.
AP74 adds diagnostic publication plot rendering over those AP73 tables,
including convergence, validation-atlas, Schramm `Y_p`/`D/H`, synthetic
posterior, and SMC temperature-trace panels plus PNG hashes,
source-model/source-refresh/`N_span`/radial momentum-delta/electron-bath/
scalar-QED provenance, and plot manifests.  FB-17 extends AP74 so AP73
diagnostic full-chain physical-smoke Schramm rows render through the Schramm
panel path and plot records/manifest preserve completed-window counts and
CPU-JAX/Rodas5P replay status plus optional live-source repeated-run BBN readout
provenance.  AP74 now also fail-closes stale or malformed FB-21 gate rows and
preserves the gate contract, diagnostic claim scope, no-public/no-production/
no-QKE flags, repeated-run BBN readouts, same-window counts, collision-payload
counts that exactly match completed windows, and live-source-vs-piecewise/
window-map deltas in plot records and the manifest.
AP75 adds diagnostic reproducibility-bundle packaging for AP66/AP67/AP72/AP74
artifacts, including required artifact provenance, AP74 source-model,
source-refresh/`N_span`, radial momentum-delta, electron-bath, and scalar-QED
consistency checks, plot hash verification, copied artifact/figure manifests,
environment metadata, command records, and claim-boundary notes.  FB-18 extends
AP75 so a passed AP72 full-chain physical-smoke row and AP74 full-chain Schramm
plot can be bundled together with finite AP68 terminal `Yp`/`D/H`, zero forward
failures, completed-window counts, CPU-JAX/Rodas5P replay status, and optional
live-source repeated-run BBN readout provenance.  AP75 now validates and copies
the same FB-21 gate provenance across AP72 summaries, AP74 plot records, the
AP74 manifest aggregate, the bundle top-level summary, and copied-plot records,
including exact completed-window/payload-count matching, while remaining
diagnostic/non-promoted rather than production SMC evidence.
AP76 adds an executable final readiness audit over the AP75 bundle; it rechecks
AP66/AP67/AP72/AP74 summaries, plot claim labels, no-QKE scope, and dispatch
status, validates source-model/source-refresh/`N_span`/radial
momentum-delta/electron-bath/scalar-QED provenance consistency, then records
the current decision as `not_promoted`.  FB-19 extends that audit so the AP75
full-chain physical-smoke bundle path is accepted only when the AP72 smoke
summary and AP74 full-chain Schramm plot provenance agree on finite terminal
`Yp`/`D/H`, completed-window counts, and CPU-JAX/Rodas5P replay status; if the
bundle carries live-source repeated-run BBN readout provenance, AP76 now
requires matching FB-21 gate contract/claim/window/payload/delta provenance,
including frozen-payload fingerprints, with every full-chain plot covered.  It
still records diagnostic smoke evidence
rather than production SMC validation.
AP77 adds a coupled weak-rate smoke gate that runs AP60
`nonlrs_s2_cl3_quadrupole_input` inside the AP65 nonlinear angular
collision-feedback 3T solve against same-CL3 `metadata_only` controls, with
finite solve checks, exact rate-application metadata, fixed/charge-neutral
electron-bath control forwarding into the coupled solve and case ledger,
bounded nonzero weak-rate deltas, and q-ladder drift diagnostics.
AP78 applies the same-CL3 control rule back to the AP66 publication-candidate
matrix: `metadata_only` rows now use the same CL3 weak kernel as
`nonlrs_s2_cl3_quadrupole_input` rows, so AP66 weak-rate-mode ladder evidence
does not mix AP60 angular application with CL0-vs-CL3 base weak-rate-kernel
changes.
AP79 links that coupled weak-rate evidence into the readiness ledger: the
AP76/AP79 audit now requires an AP77 coupled weak-rate gate artifact in addition
to the AP75 reproducibility bundle and fails closed on missing pass status,
public dispatch, production SMC, QKE, non-same-CL3 controls, or missing AP60
rate-application metadata, and it now records and checks AP77 electron-bath
control provenance from gate inputs and case rows.
AP80 adds a profile-level coupled weak-rate convergence artifact around AP77:
the smoke preset keeps q=(3,4) while the explicit extended preset adds
q=(3,4,5), records profile-level pass/fail status, and rejects AP77 reports
whose observed q ladders or comparison metadata do not match the requested
profile.  AP80 remains diagnostic and does not promote public dispatch,
production SMC, full-span weak-rate convergence, or QKE.
FB-20 now adds the optional production-candidate gate over those downstream
artifacts: the gate consumes AP79 full-chain physical-smoke readiness evidence
and AP80 extended weak-rate convergence, requires finite AP72/AP75 terminal
`Yp`/`D/H`, completed-window counts, zero forward failures, and CPU-JAX/Rodas5P
replay status, requires matching AP72/AP74 route summaries plus FB-21 gate
provenance for live-source repeated-run BBN readout evidence, and
records a pass/fail candidate ledger while keeping `promotion_decision=not_promoted`,
public dispatch disabled, production SMC validation disabled, and QKE out of
scope.
FB-23 adds a deterministic downstream evidence-chain witness over AP73/AP74/
AP75/AP79/FB20 outputs so the live-source repeated-run path can be checked as a
single composition artifact without promoting the result to public dispatch or
production SMC validation.
FB-24 adds a deterministic multi-row live-source repeated-run profile gate over
FB-21 rows, preserving the diagnostic-only boundary while reducing reliance on a
single tiny smoke layout.
FB-25 attaches that profile to the FB-23 witness as optional passive evidence
with contract, row, finite-readout, and payload-provenance checks, without
changing FB-20/AP79 promotion semantics.
FB-26 adds an opt-in dynamic restart-state payload refresh to the CPU-JAX
live-source RHS chain, refreshing the no-QKE combined collision payload at
window boundaries and then freezing it within each Rodas5P window.
FB-27 profiles that dynamic refresh over increasing tiny live-source spans and
renders generated diagnostic BBN/shear/payload plots, and FB-28 passively
attaches those profile/plot summaries to the downstream evidence-chain witness
without changing dispatch.  FB-29 packages those profile and plot outputs into
smoke/extended diagnostic bundles with preset drift checks and manifest hashes,
and FB-30 adds a longer diagnostic preset with explicit BBN-bound metadata and
fail-closed final-readout checks rather than promotion claims, FB-31 adds
chain-local AP6 radial-grid cache reuse for dynamic payload refresh cost, and
FB-32 vectorizes the deterministic AP41 collision-reference contractions that
remain hot after cache hits, FB-33 batches AP41 angular bridge dispatch over
those references, FB-34 reuses same-geometry AP41/AP6 source factories, and
FB-35 removes remaining cache-hit rebuild/validation overhead before the next
E2E BBN integration pass.  FB-36 performs that integration pass by exposing
`dynamic_restart_state` collision-payload refresh through the FB-04 chained
runner, FB-21 repeated-run gate, AP68 full-chain forward model, and the AP69
through AP72 SMC diagnostic schema/runtime/validation surfaces, with bounded
AP68 E2E smoke evidence and no public-production promotion.
FB-37 then adds the plot-ready diagnostic readout layer for those dynamic E2E
rows: AP68/AP72 metadata are normalized into an FB37 profile only when the
live-source RHS chain supplies the repeated-run BBN readout, dynamic payload
counts/provenance/fingerprints match completed windows, BBN readouts are finite
and within basic physical bounds, and public dispatch, production SMC, and QKE
remain disabled.  The plot renderer writes three generated PNG diagnostics with
manifest-hash cleanup discipline and FB-23 can carry the pair as optional
passive evidence.
FB-38 then makes that pair consumable by the existing publication bundle and
readiness audit: AP75 treats FB37 artifacts as copied diagnostic attachments,
not required AP73/AP74 publication plots or promotion evidence, and AP79
records the validated dynamic E2E BBN readout evidence only when full-chain
physical-smoke provenance is already present.
FB-39 closes the manual-composition gap by adding a runnable diagnostic chain
writer/CLI for FB37 -> AP75 -> AP79, plus a converter from passed FB27 dynamic
live-source span profiles into FB37 source rows.  The chain manifest records
the AP75 bundle, AP79 audit, FB37 copied artifacts/plots, MeV temperature
coverage when available, dynamic payload totals, and retained claim boundaries
while leaving promotion and production dispatch closed.
FB-40 adds the next executable surface above that chain: one smoke-bundle writer
and CLI now build the FB27 dynamic span profile, FB37 dynamic E2E BBN readout
profile/plots, and FB39 readiness chain in order, then record all generated
artifact hashes and claim-boundary checks in a single manifest.  This removes
the remaining manual smoke-run assembly step without upgrading the evidence to
public dispatch, production SMC validation, QKE, or production-calibrated
full-span BBN support.
FB-41 records the first reusable larger-span comparison layer: smoke and
extended FB40 manifests are validated side by side, with the extended
`N_end=(1e-9,3e-9,1e-8)` ladder required to exceed the smoke
`N_end=(1e-10,2e-10,5e-10)` ladder, and the comparison artifact captures the
MeV temperature range, BBN readout ranges, generated diagnostic figure
inventory, and retained claim boundaries.
FB-42 removes the final manual step in that diagnostic ladder: one suite writer
and CLI now run both FB40 presets, call the FB41 comparison writer, and record a
single top-level manifest with smoke/extended bundle hashes, comparison hash,
span-ratio metadata, BBN readout ranges, MeV temperature range, diagnostic plot
inventory, and explicit no-public/no-production/no-QKE boundaries.
FB-43 adds the current plotting bridge above that suite: instead of invoking the
legacy report/paper plotting scripts, the new bundle writer packages only the
FB37 PNGs referenced by FB42/FB41 after rechecking every source hash and claim
boundary, so downstream figure work can start from current dynamic E2E BBN
artifacts.
FB-44 adds the executable current figure pipeline above FB43: the new CLI runs
FB42, deletes stale current-figure outputs, runs FB43, and records a top-level
manifest proving that the path avoided the legacy report/paper plotting scripts
while retaining diagnostic-only/no-public/no-production/no-QKE boundaries.
FB-45 closes the input handoff for that path: AP75/AP79 evidence can now be
converted into an FB44-ready input bundle, and the current figure CLI accepts
that bundle directly; missing full AP77 evidence is either reported honestly or
rebuilt by explicit opt-in from the AP79 summary.  When AP75/AP79 carry FB60
full-BBN diagnostic figure attachments, FB45 also hash-checks and copies the
FB60/FB58 manifests plus three FB58 PNGs into a passive sidecar input block
outside `fb44_inputs`.
FB-46 makes the current figure regeneration repeatable in one command from
AP75/AP79 evidence by composing FB45 and FB44 and recording the final figure
inventory in a dedicated run manifest without touching legacy plotting scripts;
it now preserves FB45's optional full-BBN diagnostic figure-input block.
FB-47 starts the replacement plotting layer above that run manifest: the new
renderer uses the paper/report figure registry only as intent, consumes the
current FB46/FB42/FB37 artifacts directly, and writes three new PNG panels plus
a manifest with explicit no-legacy, diagnostic-only, and not-publication-ready
claim boundaries.
FB-48 makes that replacement plotting layer a single AP75/AP79-to-figures
command by composing FB46 and FB47 with the same no-public/no-production/no-QKE
boundaries and profile-hash provenance, including the optional FB60/FB58
full-BBN figure-input block when FB46 provides one.
FB-49 adds a physics-readiness figure audit for the same profile evidence,
separating included non-negative current `Y_p` rows from excluded negative
historical probes and making the current MeV coverage gap explicit.
FB-50 records the corresponding LRS no-collision CL0 full-BBN baseline through
`T_gamma ~= 0.005 MeV` in the standalone extended-LRS rows, showing positive
standard mass-fraction readouts and SciPy/JAX-characteristic/standalone LRS
agreement at the diagnostic tolerance without observable positivity clipping.
FB-51 records the next progressive freedom ladder through the same full-BBN
temperature range: weak corrections, non-LRS geometry, and LRS collision terms
each pass in isolation; weak+non-LRS and weak+LRS-collision pass as supported
pairs; and non-LRS+collision/all-three remain guarded until collision-coupled
non-LRS transport is implemented instead of approximated by an LRS fallback.
FB-52 adds that first implementation as private diagnostic evidence:
the non-LRS residual-state JAX surface now runs a phase-split full-BBN solve,
and the progressive ladder's staged residual mode passes the
non-LRS+collision and all-three rows without changing public dispatch status.
AP81 lands a collision-term algebra upgrade from
`neutrino_collision_term_PSTF.md`: the diagonal no-QKE fermionic 2-to-2
statistical factor is now the explicit quartic-cancelled six-monomial
polynomial (`34`, `12`, `123`, `124`, `134`, `234`) in the deterministic
`nu-e`/pair references, a new deterministic pairwise diagonal `nu-nu` 2-to-2
reference, legacy SciPy `nu-e`/pair operators, JAX `nu-e`/pair kernels, and JAX
diagonal `nu-nu` kernels.  Focused tests lock the monomial signs, absence of a
quartic `1234` term, FD detailed balance, legacy operator sharing, JAX
parity/preflight behavior, and concrete replay-stable non-equilibrium numeric
values for fixed `N_q=4` `nu-e`, pair, and pairwise diagonal `nu-nu` references.
The staged NumPy AP19/AP33/AP35/AP41 diagonal `nu-nu` source bridge now routes
through that pairwise reference by default and applies an explicit
per-bank number closure plus an effective-`nu_x` weighted-energy closure
projection; the older fixed-point redistribution helper remains legacy
comparison plumbing.  The AP6 radial source geometry now uses normalized
unit-direction momentum-delta weights by default instead of a uniform four-angle
factor, so deterministic smoke-scale `pstf_radial` sources favor vector-closed
angular quadrature tuples while keeping a normalized angular integral; an
opt-in `radial_gaussian` model evaluates the full
`p1 e1 + p2 e2 - p3 e3 - p4 e4` residual for p-dependent smoke studies.  AP81 is executable collision-factor algebra in the listed
kernels/references and the staged source route.  The AP6 follow-up adds the
first reusable local PSTF six-monomial angular contraction table, projecting
`K34`, `K12`, `K123`, `K124`, `K134`, and `K234` scalar occupation products
into mode space on deterministic angular grids.  The same AP6 line now also
adds universal `G0`, `G_mu`, and `G_mumu` four-angle geometric tables for
caller-supplied deterministic momentum-delta weights, plus channel-specific
`K` tensor assembly from HM-style `Pi_ij` and `Pi_ij Pi_kl` descriptors.
The next AP6 follow-up adds the radial-grid `p2,p3` contraction with linear
`p4 = E1 + E2 - E3` energy-conservation interpolation and invalid-kinematic
zeroing.  The following AP6 process-grid step assembles radial channel
kernel grids from invariant prefactors, supplied or normalized unit-direction
momentum-delta weights, and
HM-style channel descriptors.  The AP6 process-catalog step maps the existing UR pair annihilation/creation and pairwise diagonal no-QKE nu-nu matrix-element formulas into physical HM `Pi_ij Pi_kl` descriptors, adds charge-split finite-mass HM elastic `nu-e_minus`/`nu-e_plus` descriptors with `m_e^2 Pi_ij` interference terms, adds finite-mass HM pair-annihilation descriptors with the crossed `Pi_ij Pi_kl`, `m_e^2 Pi_ij`, and `m_e^4` terms, and threads those descriptors into the radial channel-grid builder.  The follow-up
AP6 radial source evaluator now composes that catalog, process-grid assembly,
and the six-monomial radial contraction into concrete process-specific
`C_modes` values for deterministic smoke-scale inputs.  The next AP6 step
integrates those returned source modes into raw-quadrature number and energy
moments, `sum_i w_i E_i^2 C_i` and `sum_i w_i E_i^3 C_i`, while leaving the
caller responsible for the specific p1 weight convention at the AP35/AP36
runtime bridge boundary.  A follow-up AP18 adapter now maps those concrete
radial moments into the existing `Augmented3TCollisionThermoSource` shape,
including `dQ_nue_pair_N`/`dQ_nux_bank_N` bank bookkeeping and per-species
diagnostics for opt-in callback use.  The radial provider now applies
conserved-moment projections to all nine default diagonal `nu-nu` radial
sources before AP18 thermo/hierarchy feedback: same-bank sources remain
number/energy-neutral, and the six off-diagonal sources are number-neutral
with complete unordered pairs made energy-neutral while preserving relative
raw species energy-transfer differences.  A real AP55 LRS source-budget smoke
reported `n_radial_nunu_sources=9`,
`n_radial_nunu_number_projected_sources=9`,
`n_radial_offdiagonal_nunu_number_projected_sources=6`,
`n_radial_offdiagonal_nunu_pair_energy_projected_sources=6`,
`n_radial_offdiagonal_nunu_pair_energy_projected_pairs=3`,
`radial_nunu_max_abs_number_moment=4.828087799349512e-20`,
`radial_offdiagonal_nunu_max_abs_number_moment=2.498747194400186e-20`, and
`radial_offdiagonal_nunu_pair_max_abs_energy_residual=9.317362419797304e-20`.
The same adapter now maps returned
process `C_modes(q, mode)` blocks into species-indexed `dA_modes` hierarchy
payloads, so the existing 3T shells can apply radial kinetic collision feedback
when the opt-in source is selected.  The next live-provider step reconstructs
occupation modes from the current `A_modes`/`q_nodes` callback payload and
evaluates configured radial process descriptors into those AP18 source moments.
The LRS and non-LRS AP18 source evaluators now have focused regressions
accepting that live radial bridge and reporting finite 3T `dQ` bank outputs
plus nonzero radial `C_modes` diagnostics, while deliberately avoiding a long
default live-RHS Radau solve.  The newest AP6 follow-up threads that same
radial provider into the LRS collision-feedback artifact/candidate surface and
the AP42 non-LRS S2 collision-feedback artifact as an explicit `pstf_radial`
source variant.  The default smoke routes evaluate the radial source at the
initial state and inject the resulting AP18 3T source through the existing
callback; the non-LRS route uses the S2 basis/direction vectors and records
finite radial source moments.  The unbudgeted `pstf_radial` `live_rhs` policy
is kept diagnostic because repeated radial contractions in the stiff RHS are
not yet promotion-stable; that live-radial coupling remains a budgeted runtime
gate.  The first such gate is now landed as a tiny-span diagnostic live-RHS
artifact with an explicit source-evaluation budget, concrete radial moment
diagnostics, and fail-closed budget-exceeded metadata.  A follow-up LRS
source-policy artifact now runs the frozen and budgeted live-RHS radial routes
at matched smoke settings and records live-minus-frozen observable and
source-diagnostic deltas; full-span or default live-RHS radial coupling is still
unpromoted.
The direct fixed-`mu_e` sensitivity and charge-neutrality radial source
artifacts also expose the evaluated `collision_dA_abs_max` amplitude from
`source.dA_modes`, with delta maps covering that kinetic source alongside the
thermo `dQ` moments.
The AP6 radial AP18 thermo-source bridge now also has an opt-in
`standard_3t_plasma` energy-normalization mode.  In that mode the concrete
radial source still supplies the kinetic `dA_modes` payload, while the
electromagnetic radial energy moments receive a number-neutral monopole
correction so the 3T callback matches the canonical plasma energy-transfer
table per e-fold at the current temperatures and Hubble rate.  The direct
source-only and nonlinear non-LRS frozen-source wrappers now seed the frozen
radial source with the actual initial 3T Hubble rate instead of a unit
placeholder.  The direct non-LRS AP6 artifact CLI and LRS/non-LRS
collision-feedback artifact CLIs expose this mode and explicit 3T initial
temperatures; a non-equilibrium smoke run at `0.8/0.79/0.78 MeV` reported
`radial_em_energy_target_nue_pair_N=0.002439621160259491`,
`radial_em_energy_target_nux_bank_N=0.0019820818977550445`, and maximum
closure residual `8.673617379884035e-19`.  The same mode is now threaded into
the AP6 LRS live-RHS budget/source-policy artifacts and the AP45 non-LRS
source-policy artifact; a non-LRS frozen/live `pstf_radial` source-policy
smoke run at the same non-equilibrium temperatures used the standard-3T radial
source contract in both rows with live source evaluations `7/16`.  A direct
non-LRS standard-3T radial span-profile artifact now records explicit
`N_span_end` ladders; the first LSODA diagnostic run over
`1e-14, 1e-12, 1e-8` succeeded with nfev `7, 635, 5577` and maximum displayed
closure residual `8.673617379884035e-19`.  A companion source-policy
span-profile artifact now runs the same standard-3T radial closure under both
frozen-initial-state and live-RHS source updates; the first LSODA diagnostic
run over `1e-14, 1e-12` succeeded with frozen/live nfev `7, 635`, live source
evaluations `9, 637` under a `2048` budget, and maximum displayed closure
residual `8.673617379884035e-19`.  Live-RHS source-evaluation budget
exhaustion on this direct radial path is now a typed/structured artifact
outcome rather than an unhandled traceback.  A diagnostic run over
`1e-14, 1e-12, 1e-10` with budget `4096` preserved the two shorter live-RHS
successes and recorded `source_evaluation_budget_exceeded` for the `1e-10`
live row, while the frozen-source `1e-10` reference completed with nfev
`11728`.
The signed-`mu_e` plasma EOS now also evaluates the staged isotropic QED
correction with the finite-mass e-/e+ bath at the supplied chemical potential:
`qed_delta_rho_with_electron_mu(0.8, 0.2)` gives
`0.0013502932161548037 MeV^4` versus the zero-`mu_e`
`0.001314159794007091 MeV^4`.  The exact scalar QED mode is also available as
an opt-in thermo entrypoint: `exact_finite_mu_scalar` gives
`delta_rho = -0.001545790523714566 MeV^4` and
`delta_P = -0.00045775883349570345 MeV^4` at `T=0.8 MeV`, `mu_e=0.2 MeV`.
The same opt-in model is now threaded through the LRS, source-only non-LRS,
and nonlinear non-LRS 3T Hubble/RHS solve shells with recorded
`qed_correction_model` and `qed_correction_contract` metadata.  Anisotropic/
tensor QED and promotion-grade exact-scalar-QED full-span coupled-solver
validation remain outside the staged claim.
AP55 now includes that AP6 descriptor-driven radial source in the deterministic
collision-source budget report as `lrs_pstf_radial_process_budget`,
`lrs_pstf_radial_charge_neutrality_budget`, and
`nonlrs_pstf_radial_process_budget`, and
`nonlrs_pstf_radial_charge_neutrality_budget`, recording finite radial moments,
finite-mass process markers, algebraic charge-neutral e-/e+ diagnostics,
`radial_max_abs_C_mode`, and `collision_dA_abs_max` without promoting the
radial source to default runtime.
Default/promoted runtime coupling and complete process coverage beyond this
staged UR catalog remain future work.
The queue is reuse-first rather than greenfield.  Nuclear evolution should keep
using the landed PRIMAT AC2024 `9`-species/`31`-reaction network and the
augmented weak/network shells that already call `abundance_rhs_phase2(...)`.
Weak-rate APs should extend the landed `AugmentedWeakInputs`,
`AngularWeakRateMomentInputs`, and `WeakAngularCorrectionMetadata` bridge
rather than creating a second input model, and non-LRS nonlinear transport APs should use the existing LRS
`jax/nonlinear_transport.py` discrete-ordinate RHS as an oracle/reference.
Inference APs should adapt the existing `ForwardModel`/`BBNLikelihood` and
vector/batch likelihood contracts; SMC APs should extend the current
`jax/smc_pipeline.py` and non-LRS publication/adaptive SMC runners with their
checkpoint, ESS, proposal, and failure-metadata machinery; plotting APs should
extend the existing Schramm/report figure generator, figure registry, and cache
schema rather than creating a disconnected plotting stack.
The stability harness now includes a source-only non-LRS `Sigma_-`
projection envelope over `N_mu` and `N_phi`, with plus/minus
coefficient-source limits and expected-mode residuals; it is not a
non-LRS coevolution stability claim.  AP13 adds the first SciPy
weak/network RHS bridge for this staging surface: each RHS evaluation
reconstructs the current augmented LRS distribution, extracts live
`nu_e`/`anti-nu_e` monopoles, calls the existing live weak-rate
functional, and feeds the resulting `lambda_np`/`lambda_pn` into the
PRIMAT standard network derivative.  A combined helper now returns the
collisionless transport derivative and the weak/network derivative from
the same augmented state; it still does not integrate the photon/plasma
thermodynamics, BBN network, and Einstein background as one promoted
solve.
AP14 adds a narrow SciPy `solve_ivp` shell around those same blocks:
`Sigma_+`, augmented LRS modes, and PRIMAT abundances are packed into a
single `d/dN` state with externally supplied fixed `T_gamma`, `T_nu`,
and Hubble rate.  The shell supports `method="LSODA"` for stiffness
experiments, but this is still a staging solve with fixed thermo/H
inputs rather than a promoted full thermo/background BBN driver.
AP15 adds a separate 3T thermo/Hubble variant of that shell.  It packs
`T_gamma`, `T_nu_e`, and `T_nu_x` into the same SciPy state, recomputes
`H(T_gamma,T_nu_e,T_nu_x,Sigma^2)` on every RHS call with the existing
3T helper, and feeds the resulting Hubble rate into the weak/network
block.  This still does not source thermodynamics from the live
augmented collision moments and is not a collision-coupled promoted BBN
driver.
AP16 adds convergence runners for the AP15 3T shell over `ell_max`,
`N_q`, and `N_mu`, reusing the existing convergence-report contracts
and recording thermo/network observables plus `ell_max` tail norms.
AP17 adds a deterministic JSON artifact runner around those AP16
ladders, with a smoke-scale default and an optional extended preset
that includes `ell_max = 2,4,6,8` when runtime permits.
AP18 adds an opt-in collision-moment thermo feedback hook for the same
SciPy 3T shell: when an explicit source callback supplies
`dQ_nue_pair_N` and `dQ_nux_bank_N`, the temperature RHS is evaluated
from those moments instead of the standard 3T table source.  This is
feedback plumbing only; it is not a built-in full angular collision
kernel.
AP19 supplies the first deterministic source callback for that hook:
current augmented LRS monopoles can feed the AP81 fixed-quadrature pairwise
diagonal no-QKE `nu-nu` 2-to-2 reference with the six-monomial Pauli factor,
then produce energy-closed `dQ_nue_pair_N`/`dQ_nux_bank_N` moments through an
explicit per-bank number closure and effective-`nu_x` closure projection.  This
remains a monopole diagonal `nu-nu` source and does not implement the full
angular collision kernel.
AP20 adds the corresponding electromagnetic-bath source factory: current
augmented LRS monopoles can feed the fixed-quadrature `nu-e` scattering
and pair-process references to produce `dQ_nue_pair_N` and
`dQ_nux_bank_N` moments.  This remains an angle-independent monopole
source and does not implement the full angular collision kernel.
AP21 adds an explicit combined source factory that sums the AP19
diagonal `nu-nu` and AP20 electromagnetic-bath moments into one callback
for the AP18 hook while preserving numeric component diagnostics.  This
is still opt-in source composition, not a default physical angular
collision kernel.
AP22 adds a deterministic smoke-scale collision-feedback source-variant
artifact runner comparing the standard 3T table RHS with the AP19
`nu-nu`, AP20 electromagnetic-bath, and AP21 combined source callbacks.
This is diagnostic reporting only, not promoted/default collision
coupling.
AP23 adds standard-relative observable deltas to that artifact, so each
opt-in source variant can be read against the standard 3T table-RHS
baseline in the JSON report.  This is report metadata only and does not
change the solve path.
AP24 normalizes the WBS status ledger: AP10-AP23 are now marked as
stage-scoped landed where their named deliverables and exit gates are
implemented.  AP25 then promotes AP7 from metadata-only to the bounded
LRS `Sigma_+ K_2` CL3 weak-rate multiplier described above.  AP26 lands
the AP8 weak-rate candidate sub-gate, and AP27 lands the AP8
collision-feedback source-variant candidate sub-gate.  AP28 lands the
3T span-ladder candidate gate and marks AP8 stage-scoped landed.  AP29
lands collision-feedback 3T convergence runners and marks AP5
stage-scoped landed.  AP30 lands QMC collision source-moment reports and
marks AP9 stage-scoped landed.  AP31 lands the LRS angular `nu-e`
scattering projection bridge with elastic-scattering number closure, and
AP32 lands the LRS angular electromagnetic pair-process bridge with
number-closed scattering component diagnostics.  AP33 lands the LRS angular pairwise
diagonal `nu-nu` bridge.  AP34 lands non-LRS S2 angular projection coverage for
those bridges and marks AP6 stage-scoped landed while AP4 remains partial
because its row text still names unresolved programme blockers.  AP35
lands the opt-in angular collision thermo-source callback for AP18 while
leaving default/promoted collision-feedback runtime and full-span gates
outside the staged surface.  AP36 threads that callback through the
staged artifact/gate/convergence variant machinery as `angular`.  AP37
lands the source-only non-LRS plus/minus coevolution shell, and AP38
lands its fixed-thermo weak/network companion.  AP39 lands the dynamic 3T
thermo/Hubble companion, and AP40 lands the opt-in non-LRS collision-moment
thermo feedback hook while AP4 remains partial for nonlinear transport,
default/promoted collision sources, and collision-sourced full-BBN
blockers.  AP41 lands the opt-in non-LRS S2 angular source factory for
that hook.  AP42 lands the non-LRS collision-feedback artifact runner
with angular and AP6 `pstf_radial` source variants under a smoke-default
frozen source-moment policy.  AP43 lands the
non-LRS collision-feedback q/`N_mu`/`N_phi` convergence runners.  AP44 lands
the non-LRS collision-feedback candidate gate over that artifact path including
the AP6 `pstf_radial` source-evaluation budget.  AP45 lands the live-vs-frozen
source-update policy artifact with radial budget forwarding.  AP46 lands the
source-update policy candidate gate with radial budget observables.  AP47 lands the direct opt-in angular
collision-feedback 3T solve wrapper.  AP48 lands artifact reuse of that
wrapper for the angular source variant.  AP49 lands the direct wrapper
artifact runner.  AP50 lands the direct wrapper candidate gate.  AP51 lands the
direct-wrapper outcome policy/report runner, AP52 lands the pre-budget direct
solver matrix, AP53 lands the source-update policy promotion study including
the AP42 `pstf_radial` budgeted surface, AP54 lands
the pre-budget direct-wrapper convergence artifact, AP55 lands collision-source
budget closure, AP56 lands the candidate evidence bundle, AP57 lands the
physical sanity matrix, AP58 lands the angular weak-rate input model, AP59
lands the LRS moment-input angular weak-rate mode, AP60 lands the non-LRS S2
moment-input angular weak-rate mode, and AP61 lands the rate-only weak-rate
candidate/convergence gate.  AP62 lands the non-LRS S2 nonlinear transport RHS
scaffold with LRS-oracle reduction.  AP63 lands the nonlinear transport solve
shell without weak/network or collision feedback.  AP64 lands the nonlinear
non-LRS 3T weak/network candidate shell with explicit source-only/nonlinear
transport routing.  AP65 lands the opt-in nonlinear angular collision-feedback
3T wrapper, an opt-in nonlinear `pstf_radial` wrapper, and an opt-in combined
angular+`pstf_radial` nonlinear wrapper/artifact.  AP66 lands the publication-candidate convergence matrix over angular-only
and combined angular+`pstf_radial` AP65 source-model rows.  AP67 lands the known-limit validation atlas over the AP65/AP66
evidence surface.  AP68 lands guarded inference access, AP69 lands the
augmented SMC likelihood schema, AP70 lands a smoke tempered-SMC runner, AP71
lands SMC runtime/cache controls, AP72 lands synthetic SMC validation, AP73
lands figure-ready publication artifact tables, AP74 lands diagnostic plot
rendering, AP75 lands reproducibility packaging, AP76/AP79 land the
not-promoted readiness audit with AP77 coupled weak-rate evidence, AP78 lands
same-CL3 weak-rate control hardening, AP80 lands profile-level weak-rate
convergence diagnostics, FB-20 lands the optional production-candidate
evidence gate over AP79 full-chain readiness and AP80 extended weak-rate
profiles without promotion, AP81 lands the shared six-monomial 2-to-2 Pauli
collision statistical factor plus staged pairwise diagonal `nu-nu` source
routing with number/energy closure, and the AP6 collision-reference follow-up
lands local six-monomial PSTF angular contraction plus universal geometric
kernel and channel `K` assembly tables, a supported-species
`{nue,nuebar,nux}` process descriptor catalog whose default `nu-nu` entries
are all nine ordered pairs, including identical-bank self-scattering with
Fierz factor `2`, same-bank number/energy-neutral radial projection, and
off-diagonal number-neutral projection plus unordered-pair energy-neutral
closure, radial
process/source/moment helpers, descriptor-to-mode-label mapping, fixed,
dynamic ultra-relativistic FD, dynamic finite-mass FD electron/positron
bath support for the radial provider, zero-chemical-potential default,
signed-`mu_e` pair-process Pauli blocking in the staged electromagnetic
source bridge with AP18/AP40 collision callback forwarding, explicit
fixed-`mu_e` route-level `e_minus`/`e_plus` bath splitting, direct
fixed-`mu_e` and charge-neutrality radial source artifacts with concrete
moment deltas, finite-mu-scaled isotropic QED feedback in the signed-`mu_e`
plasma EOS, opt-in exact_finite_mu_scalar QED pressure/energy corrections
through scalar 3T thermo entrypoints and LRS/source-only non-LRS/nonlinear
non-LRS 3T Hubble/RHS solve shells, charge-neutral network-derivative
electron energy feedback plus fast charge-susceptibility `mu_e` solves in the
3T shells, evolved LRS/source-only non-LRS/nonlinear non-LRS charge-neutral
electron charge-asymmetry density states/histories, and descriptor-label-aware
total-energy radial grids with neutrino `p=q` and electron/positron
`p=sqrt(max(q^2-(m_e/T_nu_e0)^2,0))` on the frozen dimensionless solver grid.  The default-catalog LRS
`pstf_radial` artifact route now includes all 18 default radial
moment sources, finite-mass elastic/pair and diagonal nu-nu source-count
diagnostics, plus all-nine diagonal nu-nu number- and pair-energy-projection diagnostics, live radial bridge acceptance at the AP18
source-evaluator boundary, opt-in standard-3T electromagnetic energy closure
for the AP18 radial thermo-source bridge, and an explicit LRS `pstf_radial`
collision-feedback artifact/candidate variant, non-LRS AP42/AP44/AP45/AP46/AP53
budgeted `pstf_radial` artifact/gate/source-policy coverage with a dedicated
non-LRS radial live-vs-frozen artifact/writer, an opt-in direct non-LRS
`pstf_radial` collision-feedback 3T wrapper plus JSON/CLI artifact with
budgeted live-RHS diagnostics and concrete finite source moments, an opt-in
nonlinear `pstf_radial` collision-feedback 3T wrapper with concrete finite
source moments on the AP65 shell, an AP65 combined angular+`pstf_radial`
nonlinear collision wrapper/artifact, plus a tiny-span budgeted
live-RHS radial diagnostic artifact, an AP4/AP65 combined full-span candidate
gate over the combined nonlinear artifact, and matched live-vs-frozen source-policy
artifact.  This does not
promote the registered augmented-PSTF no-QKE feature beyond diagnostic
substrate.
These remain scaffold/reference building blocks only; promotion-grade full-span angular kinetic+thermo collision feedback using the AP35/AP36 source path and
JAX/XLA public promotion are still outside the promoted runtime surface.

AP12 adds registry visibility without dispatch promotion:
`jax_typeI_augmented_pstf_noqke_staging` is registered in
`CAPABILITY_BY_KEY` as a diagnostic substrate, and
`typeI_augmented_pstf_noqke_staging` is registered in `FEATURE_BY_KEY`.
There is deliberately no `CAPABILITY_BY_BACKEND` entry and no
`canonical_forward_solver` route.  The registered contract records the
landed AP0-AP81 SciPy/JAX fixed-grid reconstruction, projection,
convergence, QMC replay, CL3 metadata, Rodas5P projected-source staging,
bounded LRS `Sigma_+ K_2` CL3 weak-rate multiplier, weak-rate candidate sub-gate, collision-feedback source-variant candidate sub-gate, staged 3T span-ladder candidate gate, collision-feedback 3T convergence runners, QMC collision source-moment reports, angular `nu-e` scattering projection, angular electromagnetic pair-process projection, angular pairwise diagonal `nu-nu` projection, optional angular collision `dA_modes` hierarchy RHS feedback in the LRS, source-only non-LRS, and nonlinear non-LRS 3T shells, non-LRS S2 angular projection coverage, the local PSTF six-monomial angular contraction, universal geometric, channel-kernel assembly, radial `p4` contraction, radial channel-grid tables, UR physical process descriptors, finite-mass HM elastic `nu-e_minus`/`nu-e_plus` descriptors, finite-mass HM pair-annihilation descriptors, the default supported-species finite-mass electromagnetic plus UR diagonal-`nu-nu` descriptor catalog, descriptor-label-aware radial kinematics, descriptor-driven radial source evaluation, signed-`mu_e` `nu-e` scattering and pair-process blocking with AP18/AP40 callback forwarding, fixed-`mu_e` and charge-neutrality radial source artifacts, evolved charge-neutral electron charge-asymmetry states in the LRS/source-only non-LRS/nonlinear non-LRS 3T shells, opt-in exact_finite_mu_scalar QED pressure/energy corrections through scalar 3T thermo entrypoints and LRS/source-only non-LRS/nonlinear non-LRS 3T Hubble/RHS solve shells and LRS/source-only non-LRS/nonlinear non-LRS 3T Hubble/RHS solve shells, LRS/non-LRS live radial AP18 bridge acceptance, LRS and non-LRS `pstf_radial` artifact/candidate source variants, direct and nonlinear non-LRS `pstf_radial` collision-feedback 3T wrappers plus the direct-wrapper artifact, a budgeted tiny-span live-RHS radial diagnostic artifact, and an LRS live-vs-frozen radial source-policy artifact, the AP35 opt-in angular collision thermo-source callback, AP36 `angular` source-variant artifact/gate/convergence plumbing, AP37 source-only non-LRS plus/minus coevolution shell, AP38 source-only non-LRS fixed-thermo weak/network shell, AP39 source-only non-LRS 3T thermo/Hubble shell, AP40 opt-in non-LRS collision-moment thermo feedback hook, AP41 non-LRS S2 angular collision thermo-source factory, AP42 non-LRS collision-feedback artifact runner with angular and AP6 `pstf_radial` variants, AP43 non-LRS collision-feedback convergence runners, AP44 non-LRS collision-feedback candidate gate, AP45 source-update policy artifact, AP46 source-update policy candidate gate, AP47 direct angular collision-feedback 3T solve wrapper, AP48 AP42 artifact wrapper routing, AP49 direct wrapper artifact runner, AP50 direct wrapper candidate gate, AP51 direct-wrapper outcome policy/report runner, AP52 pre-budget direct solver matrix, AP53 source-update policy promotion study, AP54 pre-budget direct-wrapper convergence artifact, AP55 collision-source budget closure artifact with pairwise `nu-nu` diagnostics, AP56 candidate evidence bundle, AP57 physical sanity matrix, AP58 angular weak-rate input model, AP59 LRS moment-input angular weak-rate mode, AP60 non-LRS S2 moment-input angular weak-rate mode, AP61 rate-only weak-rate candidate/convergence gate, AP62 non-LRS S2 nonlinear transport RHS scaffold, AP63 nonlinear transport solve shell, AP64 nonlinear non-LRS 3T weak/network candidate shell, AP65 opt-in nonlinear angular and `pstf_radial` collision-feedback 3T wrappers, AP66 publication-candidate convergence matrix, AP67 known-limit validation atlas, AP68 guarded inference adapter, AP69 augmented SMC likelihood schema, AP70 smoke tempered-SMC runner, AP71 runtime/cache controls, AP72 synthetic SMC validation, AP73 figure-ready publication artifact tables, AP74 diagnostic publication plot rendering, AP75 reproducibility-bundle packaging, AP76/AP79 final readiness audit retaining diagnostic staging and requiring AP77 coupled weak-rate evidence, AP77 coupled weak-rate smoke gate, AP78 AP66 same-CL3 weak-rate matrix control hardening, AP80 profile-level weak-rate convergence diagnostics, and SciPy live weak/network RHS/fixed-thermo/3T solve-shell plus convergence-runner/artifact, diagonal pairwise `nu-nu`/electromagnetic-bath collision-moment thermo-feedback hook pieces, explicit source composition, source-variant artifact reporting, standard-relative artifact deltas, and WBS status normalization while preserving the blockers:
FB-20 is also recorded as a landed optional production-candidate evidence gate
over AP79/AP80 artifacts; it is not a promotion decision and does not reduce
the blockers below.  FB-21 is also recorded as a landed live-source
repeated-run diagnostic gate over the FB-04 CPU-JAX/Rodas5P chain; it is not a
promotion decision and does not make the live-source chain public dispatch.
FB-22/FB-23/FB-24/FB-25/FB-26/FB-27/FB-28/FB-29/FB-30/FB-31/FB-32/FB-33 are also recorded as landed diagnostic provenance,
composition, profile, passive profile-handoff, window-boundary dynamic
payload-refresh/span-profile, passive span-profile handoff, preset bundle, long-probe, cache-reuse, reference-vectorization, and angular-batch-dispatch
witnesses for that gate; they
do not change the promotion decision or dispatch surface.
default/promoted full-span angular kinetic+thermo collision feedback using the AP35/AP36 source path,
weak-rate convergence promotion in coupled full-span solves beyond AP80 profile-level diagnostic gates, promotion-grade PSTF collision-kernel solver/runtime coupling beyond the LRS frozen-smoke `pstf_radial` artifact/candidate route, the non-LRS AP42/AP44/AP45/AP46/AP53 diagnostic `pstf_radial` route, the direct and nonlinear non-LRS `pstf_radial` 3T wrappers, the AP4/AP65 combined full-span candidate gate, the direct-wrapper artifact, tiny-span budgeted live-RHS radial artifact, and live-vs-frozen radial source-policy artifact, process families outside the supported no-QKE HM finite-mass electromagnetic plus all-nine diagonal-`nu-nu` catalog plus promotion-grade coupled-solver use of that catalog, promotion-grade physical full-BBN
span integration, anisotropic/tensor QED response, promotion-grade exact-scalar-QED full-span coupled-solver validation beyond the diagnostic exact-scalar routing/gate smoke now landed, real-data/production SMC evidence, convergence promotion
ladders, and any GPU/XLA promotion.
The AP76/AP79 readiness audit, FB-20 optional candidate gate, FB-21
live-source repeated-run diagnostic gate, and FB-22/FB-23/FB-24/FB-25/FB-26/FB-27/FB-28/FB-29/FB-30/FB-31/FB-32/FB-33
provenance/composition/profile/profile-handoff/payload-refresh/span-profile/span-profile-handoff/bundle/long-probe/cache-reuse/reference-vectorization/angular-batch-dispatch witnesses are landed and
record `not_promoted`: they do not change the registered capability, canonical
dispatch, public production support, or the current AP0-AP81 diagnostic evidence
ledger.

## 4. Current numerical parity

Measured at `N_q=20, N_mu=12, n_reactions=12` on CPU (no JIT
recompilation — warm runs).

### 4.1 Tier-1 characteristic: SciPy ↔ JAX

| Σ_H | CL0 |ΔY_p| | CL1 |ΔY_p| | CL2 |ΔY_p| | |ΔD/H|/DH max |
|---|---|---|---|---|
| 0.0 | 1.4 × 10⁻⁸ | 3.8 × 10⁻⁸ | 6.0 × 10⁻⁹ | 2.8 × 10⁻⁵ |
| 0.1 | 7.7 × 10⁻⁹ | 9.9 × 10⁻⁹ | 1.8 × 10⁻⁹ | 3.0 × 10⁻⁵ |
| 0.3 | 2.5 × 10⁻⁸ | 5.7 × 10⁻⁹ | 1.8 × 10⁻⁸ | 1.5 × 10⁻⁵ |
| 0.5 | 7.3 × 10⁻⁹ | 1.6 × 10⁻⁹ | 1.2 × 10⁻⁸ | 7.4 × 10⁻⁶ |

Publication tolerance: 5 × 10⁻⁵.  Headroom: 3–4 orders of magnitude.

### 4.2 Tier-2 characteristic: SciPy ↔ JAX

| Σ_H | CL0 |ΔY_p| | CL1 |ΔY_p| | CL2 |ΔY_p| | |ΔN_eff| max |
|---|---|---|---|---|
| 0.0 | 2.9 × 10⁻⁸ | 4.3 × 10⁻⁸ | 2.3 × 10⁻⁸ | 6.2 × 10⁻⁶ |
| 0.1 | 6.0 × 10⁻⁸ | 3.6 × 10⁻⁸ | 8.1 × 10⁻⁹ | 6.2 × 10⁻⁶ |
| 0.3 | 1.4 × 10⁻⁸ | 4.7 × 10⁻⁹ | 2.9 × 10⁻⁸ | 4.1 × 10⁻⁶ |
| 0.5 | 1.0 × 10⁻⁸ | 3.1 × 10⁻⁹ | 2.0 × 10⁻⁸ | 2.5 × 10⁻⁶ |

Publication tolerance: 5 × 10⁻⁷.

### 4.3 FLRW cross-code target (tier-3 candidate)

| Reference | Y_p (standard η, CL2/3) | N_eff |
|---|---|---|
| LASAGNA (Escudero 2019) | 0.2470 | 3.044 |
| FortEPiaNO (Froustey 2020) | 0.2470 ± 2×10⁻⁴ | 3.043 ± 0.001 |
| PRIMAT AC2024 (Pitrou 2024) | 0.24703 ± 6×10⁻⁵ | 3.044 |
| Mangano 2005 (SM benchmark) | — | 3.044 |
| **RABBIT AP-unified candidate** CL0 (`backend='jax_ap_unified_tier3'`) | **0.24139** (FLRW) | **3.0345** (gap **+0.0095** documented) |
| **RABBIT AP-unified candidate** CL1 (Born) | **0.24529** (FLRW) | **3.0345** |
| **RABBIT AP-unified candidate** CL2 (Born+Coulomb+Sirlin) | **0.24699** (FLRW) | **3.0345** |
| **RABBIT AP-unified candidate** CL3 (+finite-mass) | **0.24750** (FLRW) | **3.0345** |
| RABBIT tier-3 canonical target (deferred Option E) | 0.2470 ± 5×10⁻⁴ | 3.044 ± 0.005 |

Current RABBIT tier-2 lands at Y_p = 0.24736 (CL2 FLRW),
N_eff = 3.034 — consistent with tier-2 scope.

The AP-unified candidate row records the PR-T3B canonical
milestone reachable through ``canonical_forward_solver(backend=
'jax_ap_unified_tier3', ...)`` after PR-T3D canonical #2.  The
Mangano gap is exposed in ``BBNPrediction.metadata`` as
``flrw_mangano_gap_documented = 0.0095``; closing it to the
``5×10⁻³`` canonical-tier target requires Option E (in-RHS
analytic relaxation pre-conditioner via Jacobian augmentation,
deferred to post-canonical, see
``docs/research/PR-T3B_option_E_canonical_post_enhancement.md``).
The AP-unified anisotropic stability spread (~5.4×10⁻⁵ across
``Σ_H ∈ {0, 0.05, 0.10}``) and grid scaling spread (~8.7×10⁻⁵)
both pass the canonical PR-T3D §5 gates by >10× headroom; it is
*only* the FLRW Mangano gap that keeps the surface in candidate
tier rather than canonical tier.

---

## 5. File inventory (load-bearing files only)

### 5.1 Source

- [`src/rabbit/jax/driver_typeI_char.py`](../src/rabbit/jax/driver_typeI_char.py)
  — JAX characteristic-ray driver (tier-1 + tier-2).
- [`src/rabbit/jax/driver_typeI.py`](../src/rabbit/jax/driver_typeI.py)
  — JAX linearised PSTF driver + dispatch into char driver.
- [`src/rabbit/jax/solver_jax_rodas5p.py`](../src/rabbit/jax/solver_jax_rodas5p.py)
  — JAX Rodas5P solver with event detection + block-sparse option.
- [`src/rabbit/jax/q_advection_jax.py`](../src/rabbit/jax/q_advection_jax.py)
  — continuous q-advection operators plus PCHIP exact-remap oracle for
  the private tier-3 collisionless shell.
- [`src/rabbit/jax/driver_typeI_full_boltzmann.py`](../src/rabbit/jax/driver_typeI_full_boltzmann.py)
  — private CPU-only full phase-space Type-I shell
  (collisionless canonical path plus audit-only
  `collision_mode="spectral_relaxation_preflight"` with bounded
  `thermo_tier ∈ {1,2}` and projected low-rank Jacobian payloads).
- [`src/rabbit/jax/full_boltzmann_collision_preflight.py`](../src/rabbit/jax/full_boltzmann_collision_preflight.py)
  — host-side species-resolved isotropic collision-core preflight for
  the private full phase-space shell.
- [`src/rabbit/jax/nudec_coupled_jax.py`](../src/rabbit/jax/nudec_coupled_jax.py)
  — 3T thermo primitives (`hubble_3T_jax`, `coupled_3T_rhs_jax`, `N_eff_from_3T_jax`).
- [`src/rabbit/jax/thermo_provider_jax.py`](../src/rabbit/jax/thermo_provider_jax.py)
  — tier-1 thermo primitives.
- [`src/rabbit/jax/characteristic_rays_jax.py`](../src/rabbit/jax/characteristic_rays_jax.py)
  — ray map, analytic angular Jacobian, stress, monopole extractors.
- [`src/rabbit/jax/characteristic_rays_nonlrs_jax.py`](../src/rabbit/jax/characteristic_rays_nonlrs_jax.py)
  — additive non-LRS S² quadrature, forward map, analytic intensity/Jacobian,
  and stress/monopole primitives.
- [`src/rabbit/jax/weak_live_jax.py`](../src/rabbit/jax/weak_live_jax.py)
  — CL0–CL3 live weak kernels.
- [`src/rabbit/drivers/full_coupled_typeI.py`](../src/rabbit/drivers/full_coupled_typeI.py)
  — SciPy reference driver (tier-2 fall-through fix applied).
- [`src/rabbit/config/backend_capabilities.py`](../src/rabbit/config/backend_capabilities.py)
  — backend registry (adds `jax_characteristic`, `jax_characteristic_tier2`,
  `jax_characteristic_nonlrs`).
- [`src/rabbit/inference/forward_likelihood.py`](../src/rabbit/inference/forward_likelihood.py)
  — `canonical_forward_solver` dispatch.

### 5.2 Tests

- [`tests/test_jax_typeI_characteristic_parity.py`](../tests/test_jax_typeI_characteristic_parity.py)
  — 18 tests; tier-1 parity grid.
- [`tests/test_jax_typeI_characteristic_tier2.py`](../tests/test_jax_typeI_characteristic_tier2.py)
  — 28 tests; tier-2 parity + heating asymmetry + capability registry.
- [`tests/test_pr_a_analytic_jacobian.py`](../tests/test_pr_a_analytic_jacobian.py)
  — 4 tests; locks the analytic `J_j(S)` reconstruction against the
  numerically integrated transport ODE.
- [`tests/test_pr_n1_nonlrs_primitives.py`](../tests/test_pr_n1_nonlrs_primitives.py)
  — non-LRS primitive lock: S² weight sum, constant-shear forward-map
  ODE cross-check, LRS reduction, `Π_- = 0` at `Σ_- = 0`, and x↔y symmetry.
- [`tests/test_pr_n2_nonlrs_driver.py`](../tests/test_pr_n2_nonlrs_driver.py)
  — non-LRS driver integration: analytic Jacobian audit, LRS reduction,
  generic sign audit, large-shear smoke, and inference dispatch.
- [`tests/test_cross_backend_regression.py`](../tests/test_cross_backend_regression.py)
  — matched-physics cross-backend (now pinned at characteristic).
- [`tests/test_jax_typeI_publication_parity.py`](../tests/test_jax_typeI_publication_parity.py)
  — previously `@xfail`ed anisotropic test, now passes via `backend='jax_characteristic'`.
- [`tests/test_pr_t3a_collisionless_driver.py`](../tests/test_pr_t3a_collisionless_driver.py)
  — PR-T3A lock: q-advection inflow boundaries, PCHIP oracle bounds,
  FLRW reduction, bounded anisotropic reduction, tier-1 state-dim
  contract, and private tier-2 collision-preflight Jacobian/smoke locks
  for the full phase-space shell.
- [`tests/test_pr_t3b_collision_preflight.py`](../tests/test_pr_t3b_collision_preflight.py)
  — host-side collision-core preflight lock: bank gather, equilibrium
  zero, species hierarchy, source-only vs state-dependent moment-core
  Jacobian structure, and lifted explicit-shell factorization parity.
- [`tests/test_production_gates.py`](../tests/test_production_gates.py)
  — P0–P15 release gates.
- [`tests/test_registry_sync.py`](../tests/test_registry_sync.py)
  — registry ↔ docs drift detector.

### 5.3 Topic guides (referenced from this document)

- [JAX_CHAR_GPU_OPTIMIZATION_PLAN.md](JAX_CHAR_GPU_OPTIMIZATION_PLAN.md)
- [IMPLEMENTATION_GUIDE_NON_LRS_TYPEI.md](IMPLEMENTATION_GUIDE_NON_LRS_TYPEI.md)
- [IMPLEMENTATION_GUIDE_3T_THERMO.md](IMPLEMENTATION_GUIDE_3T_THERMO.md)
- [IMPLEMENTATION_GUIDE_FULL_COLLISION_NOQKE.md](IMPLEMENTATION_GUIDE_FULL_COLLISION_NOQKE.md)

---

## 6. Test-suite status

- Total characteristic-roadmap regression tests locked: 76
  (tier-1 char + tier-2 char + analytic-J lock + cross-backend +
  publication parity + registry + production gates).
- Previously-xfail `test_jax_matches_scipy_publication_scope_anisotropic_backbone`
  **now passes** at `backend='jax_characteristic'`.

### 6.1 Known pre-existing red tests (NOT caused by recent work)

Surfaced during regression but traceable to earlier tiers, not to any
PR in this roadmap:

1. `test_registry_sync.py::test_supported_capabilities_mentions_features`
   — `SUPPORTED_CAPABILITIES.md` was originally missing the literal
   string "Inference".  **Resolved**: the doc has since been
   refreshed and the test is green at the time of the PR-R-PF
   release-gate preflight pass.
2. `test_production_gates.py::test_classB_typeV_bbn_gold`
   — Class B Type V Y_p fixture drift (rel 9 × 10⁻⁴).  Out of scope
   for Type I roadmap; owned by the Class A/B effort.  **Formally
   `xfail`** in PR-R-PF with `strict=False` and an inline rationale
   pointing to `docs/CLASSB_PROMOTION_PACKET.md`; no Type I code
   change.
3. `test_production_gates.py::test_jax_flrw_gold`
   — The fixture entry `jax_flrw` stores the `live_f0_cl0` gold value
   (0.26125) but the test runs with `use_live_weak_monopoles=False`
   (equilibrium FD) whose correct gold is 0.2423504 (`jax_flrw_equilibrium`
   entry).  **Test-side bug**, not code drift.  **Resolved** in
   PR-R-PF: the test now reads `gold["jax_flrw_equilibrium"]["Yp"]`.
4. `test_production_gates.py::test_anisotropy_signal_parity`
   — Compares SciPy char (nonperturbative) against JAX linearised PSTF
   (perturbative) which capture only ~21 %% of the anisotropic Y_p shift.
   This was a genuine physics-gap red per paper §6.8, not a regression.
   **Resolved** by PR-D: the gate was rewritten to compare the
   SciPy reference backend against the promoted bounded `backend='auto'`
   characteristic surface.

After PR-R-PF, three of the four pre-existing reds are formally
green and the Class B Type V drift is `xfail`-ed with a documented
deferral.  None of these resolutions tag a release: the full PR-R
release tag is gated on the upstream tier-3 promotions (PR-T3B/C/D
canonical) and is intentionally not attempted in PR-R-PF.

The historical four-test red baseline is documented in
[ROADMAP_PR_CATALOG.md](ROADMAP_PR_CATALOG.md) so the status of every
dispatch/test decision survives as code and tests change.

---

## 7. What is explicitly out of scope for the Type I roadmap

- Bianchi Types II, VI₀, VII₀, VIII, IX (Class A curved kernels) —
  handled by the separate Class A effort, `src/rabbit/jax/driver_classA.py`.
- Class B (layered) Bianchi types — same.
- Tilted cosmologies (`v0 > 0`) — candidate surface
  `jax_tilted`, low priority.
- CMB-joint analyses.
- Radiative corrections beyond CL3.

These remain out of scope regardless of tier-3 completion because they
introduce orthogonal physics (curved space, tilt, CMB) that does not
belong in the Type I incomplete-decoupling line.
