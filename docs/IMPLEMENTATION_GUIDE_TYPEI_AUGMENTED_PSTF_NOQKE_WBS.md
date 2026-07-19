# Implementation Guide — Type I Augmented PSTF Full Boltzmann, No QKE

> **Historical WBS ledger after BD186.**  This document remains the landed-work
> and provenance ledger.  The active future-order plan is now
> [TYPEI_AUGMENTED_NOQKE_UNIFIED_FUTURE_PLAN.md](TYPEI_AUGMENTED_NOQKE_UNIFIED_FUTURE_PLAN.md).
> When this ledger conflicts with that file, follow the unified future plan.

> **Post-deflation note (BD612, 2026-07-08).** The augmented-PSTF AP0–AP81 code
> surface was largely deleted in PR-D1..D3 (−183.7K LOC). The per-row statuses
> below are a HISTORICAL provenance record, NOT current capability, and some
> named functions/modules no longer exist in-tree. The only augmented modules
> that survive are the collisionless substrate:
> `transport/augmented_pstf_distribution.py`, `augmented_nonlrs_transport.py`,
> `augmented_typeI_nonlrs_collisionless.py`, `augmented_typeI_observables.py`,
> and `jax/characteristic_rays_nonlrs_jax.py` (candidate backend
> `jax_characteristic_nonlrs`). For current capability, treat
> `src/rabbit/config/backend_capabilities.py` / `feature_capabilities.py` as
> source of truth.

**Status:** historical SDD/WBS ledger after BD186.  This document preserves the
landed work-breakdown record for the nonperturbative augmented-PSTF Type I
transport programme; active future ordering now lives in the unified future
plan linked above.

**Scope:** `(non)-LRS Bianchi Type I`, classical scalar neutrino
Boltzmann transport, Einstein-background coevolution, full weak-rate
functionals from the live distribution, and no quantum kinetic equation
or flavour-coherence terms.

**Execution order:** SciPy reference first, stability/convergence gates
second, JAX/XLA port third.

**Mandatory anti-drift pre-read:** before adding any further AP/FB work in this
programme, read
`docs/TYPEI_AUGMENTED_NOQKE_CODEX_ANTI_DRIFT_GUARDRAILS.md`.  Future PRs must
retire or measurably reduce a named physics, solver, or performance blocker.
Do not add another standalone diagnostic/readiness/manifest/hash/figure gate
unless the same PR deletes or consolidates older gate plumbing.

---

## 1. Contract

The target model evolves the Einstein background and neutrino
Boltzmann distribution in one coupled RHS.  The neutrino distribution
is not represented as a linear perturbation around equilibrium.
Instead, each effective species carries an augmented logit/PSTF
distribution:

```text
f_s(q, n, N) = sigmoid(-(q + A_s(q, n, N)))
A_s(q, n, N) = sum_{ell,m} A_{s,ell,m}(q,N) Y^real_{ell,m}(n)
```

For LRS Type I, the angular basis reduces to even Legendre modes
`A_ell(q) P_ell(mu)`.  For generic diagonal non-LRS Type I, the basis
uses real PSTF/spherical-harmonic modes in the principal-axis frame.

This is a finite `ell_max` truncation, not an exact infinite hierarchy.
Every physics promotion therefore requires an explicit `ell_max`
ladder such as `(2, 4, 6, 8, 10)`, plus separate angular-grid and
momentum-grid convergence evidence.

QKE is excluded by construction:

- no density matrices,
- no off-diagonal flavour coherence,
- no oscillation Hamiltonian,
- no off-diagonal neutrino-neutrino collision terms.

The species split is:

```text
nu_e, anti_nu_e, nu_x, anti_nu_x
```

where `nu_x` represents the mu/tau sector when symmetry permits.

---

## 2. Physics References

Every formula added under this programme must cite one of the local
report equations or a committed derivation note.  External references
used to establish the formalism:

- Thorne, *Relativistic radiative transfer: moment formalisms*,
  MNRAS 194, 439 (1981), DOI `10.1093/mnras/194.2.439`.
- Ellis, Treciokas, and Matravers, *Anisotropic solutions of the
  Einstein-Boltzmann equations*, Annals of Physics 150, 455 (1983),
  DOI `10.1016/0003-4916(83)90023-4`.
- Challinor and Lasenby, covariant PSTF CMB hierarchy work,
  Annals of Physics 280, 301 (2000), DOI `10.1006/aphy.2000.6033`.
- Repository report sections:
  - `docs/RABBIT_report/sections/03_neutrino_transport_linearised_pstf_hierarchy.tex`
  - `docs/RABBIT_report/sections/06_exact_characteristic_ray_transport.tex`
  - `docs/RABBIT_report/sections/07_the_collision_integral_general_structure.tex`
  - `docs/RABBIT_report/sections/08_the_hannestad_madsen_collision_kernel.tex`
  - `docs/RABBIT_report/sections/13_weak_n_p_interconversion_rates.tex`
  - `docs/RABBIT_report/sections/A05_full_phase_space_ray_boltzmann_equation.tex`

Transport-method policy:

- `S_N` / discrete ordinates is the reference angular transport method,
  because it evaluates angular integrals directly and supports
  independent angular-grid convergence.
- PSTF projection is used to store and audit angular multipoles.
- M1 is diagnostic only.  It is a useful two-moment closure comparator
  but cannot replace the required `ell_max` ladder.
- DSMC or stochastic sampling is not used inside the stiff RHS as live
  randomness.  Any future sampling accelerator must use fixed samples
  per solve, deterministic replay, and convergence to the deterministic
  quadrature reference.

Web/literature survey notes for the transport-method decision:

- Modern S_N reviews describe the method as direct angular
  discretization of the Boltzmann/transport equation and emphasize that
  angular quadrature choice controls accuracy and ray effects; this
  matches the need for independent angular-grid convergence.
- S_N adaptive collision-source work suggests an acceleration path:
  high angular order for strongly anisotropic early collision/source
  pieces, then reduced angular order as scattering isotropizes the
  source.
- M1 neutrino-transport closure literature reports that no analytic
  closure is uniformly best across regimes; this supports keeping M1 as
  a comparator rather than the reference solver.
- Monte Carlo/control-variate literature supports variance reduction
  by subtracting a deterministic neighbouring problem; this motivates
  any future QMC accelerator being a control-variate correction to the
  deterministic quadrature reference.

Useful external URLs:

- S_N review: `https://doi.org/10.3390/en18112880`
- Adaptive collision source S_N method:
  `https://doi.org/10.1016/j.anucene.2017.02.013`
- Analytic M1 neutrino closures:
  `https://academic.oup.com/mnras/article/469/2/1725/3752461`
- Deterministic-computation control variates:
  `https://doi.org/10.1016/S0378-3758(99)00078-6`

---

## 3. Architecture

### 3.1 Angular Decomposition Contracts

The initial API scaffold lives in
`src/rabbit/transport/angular_decomposition.py` and is intentionally
separate from the legacy linearised `MultipoleSpec`.

Core contracts:

- `AngularMode(ell, m, parity)`
- `AngularDecompositionSpec`
- `EllMaxConvergenceSpec`
- `CollisionQuadratureSpec`
- `CollisionMomentResult`

The LRS default mode ladder is even `ell`, `m=0`.  The non-LRS diagonal
default is even `ell` and even cosine `m` modes in the principal-axis
frame.  Sine partners are explicit opt-in diagnostics.

### 3.2 RHS Data Flow

Each RHS evaluation must follow this order:

1. Unpack geometry, augmented distribution coefficients, thermo state,
   and BBN network state.
2. Reconstruct `f_s(q,n)` on the chosen S_N grid.
3. Compute stress-energy integrals:
   `rho_nu`, `P_nu`, `Pi_+`, `Pi_-`.
4. Compute weak-rate inputs from the current `nu_e` and `anti_nu_e`
   distribution.  Born/CL0-CL2 rates consume the exact monopole;
   CL3 angular corrections must be explicit metadata-bearing terms.
5. Evaluate collision moments from the current distribution if
   collisions are enabled.
6. Project nodal `df/dN` back to augmented coefficients using the
   logit relation with a documented `f(1-f)` floor.
7. Pack one coupled derivative vector for `solve_ivp`.

### 3.3 Collision Evaluation

The first implementation target is deterministic fixed quadrature.
Collision output shape must match the input angular/momentum moment
state and must carry diagnostics:

- detailed-balance residual,
- number residual,
- energy-transfer residual by species,
- quadrature contract string.

Required collision processes:

- `nu + e^\pm -> nu + e^\pm`,
- `nu + anti_nu <-> e^+ + e^-`,
- diagonal no-QKE `nu + nu -> nu + nu`.

### 3.4 Weak-Rate Evaluation

Weak rates are algebraic functionals of the current distribution and
plasma state.  They are not independent ODE variables.

The weak bridge must not silently fall back to a temperature-compressed
distribution on this path.  Any approximation must be exposed in result
metadata and in the SDD status table.

### 3.5 Reuse-First Completion Assets

AP51-AP81 must reuse existing production/candidate infrastructure where it
already exists instead of creating parallel implementations:

- Nuclear network evolution is already landed through the PRIMAT AC2024
  `9`-species, `31`-reaction network in
  `src/rabbit/network/abundances_standard.py`; augmented RHS code must keep
  using `abundance_rhs_phase2(...)`, `evaluate_nuclear_rates(...)`,
  `compute_fluxes(...)`, and the existing `n_reactions=12/31` contracts rather
  than adding another network.
- The staged augmented weak/network shells in
  `src/rabbit/transport/augmented_typeI_weak_network.py` already reconstruct
  the current distribution, compute live weak rates, and feed the PRIMAT
  abundance derivative into SciPy solves.  Later APs should optimize and extend
  those shells, not bypass them.
- The weak angular input contract is represented by
  `src/rabbit/weak/augmented_bridge.py` through `AugmentedWeakInputs`,
  `AngularWeakRateMomentInputs`, and `WeakAngularCorrectionMetadata`.  AP59-AP61
  should extend that contract into live rate application and convergence gates,
  not introduce a second angular weak-rate input model.
- Existing nonlinear transport math in `src/rabbit/jax/nonlinear_transport.py`
  provides the LRS discrete-ordinate Boltzmann RHS and should be used as an
  oracle/reference when extending to the non-LRS S2 basis in AP62-AP64.
- Direct-wrapper artifacts and gates already exist:
  `build_augmented_nonlrs_angular_collision_3t_solve_artifact(...)`,
  `run_augmented_nonlrs_angular_collision_direct_candidate_gate(...)`, and the
  AP45 frozen/live source-policy artifact.  AP51-AP57 already extend these
  report contracts; later validation runners should keep that reuse pattern
  rather than create new validation result semantics.
- Existing convergence report types in `src/rabbit/validation/augmented_convergence.py`
  (`EllMaxConvergenceReport`, `ResolutionConvergenceReport`) are the report
  surface for AP54/AP66; later runners add observables and ladders, not new
  convergence schemas.
- Inference plumbing already exists in `src/rabbit/inference/forward_likelihood.py`
  and `src/rabbit/inference/sampler.py`: `ForwardModel`, `BBNLikelihood`,
  vector log-likelihood adapters, canonical batch likelihoods, and sampler
  config/result contracts.  Augmented inference APs should add guarded
  candidate adapters on top of these contracts.
- Tempered SMC is not a blank slate.  `scripts/run_nonlrs_typeI_publication_smc.py`,
  `scripts/run_nonlrs_typeI_blackjax_adaptive_smc.py`, and
  `src/rabbit/jax/smc_pipeline.py` already provide priors, non-LRS parameter
  summaries, adaptive or scheduled tempering, ESS resampling, proposal
  adaptation, checkpoint/restart, batched likelihood evaluation, and failure
  metadata.  AP70/AP71 should generalize these runners to the augmented
  candidate forward model instead of reimplementing SMC mechanics.
- Schramm-style and publication figure surfaces already exist in
  `scripts/generate_paper_figures.py`, `scripts/figure_registry.py`,
  `scripts/figure_cache_schema.py`, and the report figure regeneration scripts.
  AP73/AP74 should add augmented artifact schemas and panels to that registry
  and cache discipline, including Schramm `Y_p` and `D/H` panels, rather than
  creating a disconnected plotting stack.
- Local report PDFs and guides document the current publication surface:
  `docs/RABBIT_report/RABBIT_report.pdf` includes the production Type-I
  likelihood, Schramm-style `Y_p`/`D/H` panels, and PRIMAT network discussion;
  figure cache/report conventions live in `scripts/figure_cache_schema.py`,
  `scripts/figure_registry.py`, and the report regeneration scripts;
  `docs/IMPLEMENTATION_GUIDE_FULL_COLLISION_NOQKE.md`
  keeps QKE/flavour-coherence exclusions and no-QKE collision-source validation
  gates explicit.  AP51-AP81 documentation must preserve those boundaries.

---

## 4. PR/WBS Ledger

### 4.0 2026-05-17 refreshed full-BBN DAG/WBS

This refresh is based on a current-code scan of the AP4/AP65
`piecewise_frozen` combined-source path, the AP66-AP76 publication and SMC
surfaces, and the AP77/AP80 coupled weak-rate diagnostics.  It does not
renumber AP0-AP81 or promote any public backend.  AP4 remains
programme-partial until collision-coupled full-BBN execution is shown with
weak-rate coupling and publication/SMC artifacts that consume the same
physical run products.  FB-01/FB-03 now close the restart, chained-window, and
adaptive source-refresh infrastructure layer, including CPU-JAX/Rodas5P
replay-state payloads, and the FB-04 follow-up now starts executable CPU-JAX
Rodas5P window-map replay solves from those payloads, adds an opt-in
live-source RHS sidecar over nonlinear non-LRS S2 logit transport, 3T/Hubble,
live weak monopoles, and PRIMAT network blocks, and extracts finite terminal
`Yp`, `D/H`, `N_eff_3T`, and Schramm coordinates from both the chained
phase-2 terminal states and the enabled live-source sidecar's own final state,
including sidecar-vs-terminal observable deltas and restart kwargs that can
seed the next live-source window; a dedicated CPU-JAX live-source chain runner
now uses that handoff across consecutive smoke windows, exposes a JSON
artifact CLI, and can be attached to the FB-04 chained artifact as optional
diagnostic comparison evidence against the same piecewise/window-map rows,
using frozen per-window terminal collision payloads with provenance
fingerprints when requested,
and can be selected as the staged CPU-JAX/Rodas5P repeated-run evidence/readout
source while remaining non-public,
and FB-21 now wraps that same live-source repeated-run path in a fail-closed
diagnostic gate requiring finite BBN readouts, finite live-source-vs-piecewise
comparison deltas over the same tiny spans, and supplied/applied frozen
terminal collision payloads with provenance fingerprints; FB-24 now profiles
that FB-21 gate over multiple tiny chained span layouts as a diagnostic ladder,
and FB-25 now lets the FB-23 downstream witness attach and validate that FB-24
profile as optional passive evidence,
and FB-26 now adds an opt-in dynamic restart-state collision-payload refresh at
CPU-JAX/Rodas5P live-source chain window boundaries,
and FB-27 now profiles that dynamic refresh over increasing smoke-to-extended
live-source spans and renders diagnostic PNG/manifest plots,
and FB-28 now lets the FB-23 downstream witness passively attach that FB-27
dynamic span profile plus plot manifest after fail-closed provenance checks,
and FB-29 now packages the FB-27 profile plus generated plots into a single
smoke/extended diagnostic bundle runner with preset metadata and manifest
hashes,
and FB-30 now adds a longer diagnostic preset through `N_end=3e-7` with
finite-run plot output, fail-closed BBN observable-bound checks, and
CPU-JAX live-source abundance-safe Rodas5P tolerance metadata,
and FB-31 now reuses AP6 radial-grid/pretabulation cache entries across
dynamic restart-state collision payload refreshes within a live-source chain,
and FB-32 now vectorizes the AP41 deterministic `nu-e`, pair-annihilation,
and diagonal `nu-nu` collision-reference contractions used by dynamic payload
refresh,
and FB-33 now batches those AP41 deterministic references across angular
nodes/species in the angular bridge, removing the cache-hit per-angle scalar
reference dispatch layer,
and FB-34 now reuses AP41+AP6 source factory closures across dynamic
restart-state payload refreshes when the source geometry/configuration is
unchanged,
and FB-35 now removes remaining cache-hit rebuild/validation overhead by
reusing the factory S2 grid, lazily rebuilding external-q pair-leg quadrature
only on source-factory cache misses, and precomputing AP6 radial moment
projection matrices,
but does not yet promote that sidecar to a full
collision-coupled backend or production-calibrated full-span BBN yield.
FB-05 now adds a
tractable live-RHS micro-window comparator that reuses chained restart states
and classifies live-vs-piecewise deltas without making `live_rhs` the default.
FB-06 now promotes the AP6/AP65 electromagnetic and all-nine diagonal `nu-nu`
conserved-moment diagnostics into a per-window and cumulative ledger over the
same chained artifact family.  FB-07 now carries the AP60 non-LRS S2 CL3
weak-rate mode onto the same chained rows, preserving terminal
`lambda_np`/`lambda_pn` and CL3 rate-application metadata and comparing
same-CL3 metadata-only controls against applied rows.  FB-08 now runs the
fixed/charge-neutral electron-bath and finite/exact scalar-QED cross-product
over the same chained windows, checking evolved charge-asymmetry handoff and
QED one-hot route diagnostics on every row.
SciPy remains the physics-reference and source-generation shell for this
stage; the repeated full-chain/SMC executor target is CPU-first JAX using the
in-tree Rosenbrock/Rodas5P path, not long SciPy production runs.

The dependency DAG is:

```text
D0 current AP4/AP65 piecewise physical-preview baseline
  -> FB-01 restartable terminal-state payloads
  -> FB-02 chained physical-window runner plus CPU-JAX/Rodas5P replay layout
  -> FB-03 adaptive source-refresh scheduler
  -> FB-04 physical BBN span calibration and observables
  -> FB-05 live-RHS micro-window comparator
  -> FB-06 collision number/energy ledger over chained windows
  -> FB-07 coupled weak-rate convergence on chained windows
  -> FB-08 electron-bath and scalar-QED cross-product chain evidence
  -> FB-09 q/angular/PSTF-resolution chain ladders
  -> FB-10 AP66 matrix rows backed by chained artifacts
  -> FB-11 AP67 known-limit atlas rows backed by chained artifacts
  -> FB-12 AP68 forward adapter full-chain mode
  -> FB-13 AP69 likelihood schema for full-chain controls
  -> FB-14 AP70 tempered-SMC warm run using the full-chain adapter
  -> FB-15 AP72 synthetic recovery using the full-chain adapter
  -> FB-16 AP73 Schramm/publication artifact tables from real rows
  -> FB-17 AP74 diagnostic publication plots from real rows
  -> FB-18 AP75 reproducibility bundle from real rows
  -> FB-19 AP76/AP79 readiness audit with full-chain evidence
  -> FB-20 optional slow production-candidate gate, still no public dispatch
  -> FB-21 live-source repeated-run evidence gate, still no public dispatch
  -> FB-22 downstream live-source payload provenance propagation
  -> FB-23 downstream evidence-chain witness
  -> FB-24 multi-row live-source repeated-run profile gate
  -> FB-25 optional FB-24 profile evidence attachment to FB-23 witness
  -> FB-26 dynamic restart-state collision-payload refresh at live-source window boundaries
  -> FB-27 increasing-span dynamic live-source profile plus diagnostic plots
  -> FB-28 optional FB-27 dynamic span-profile evidence attachment to FB-23 witness
  -> FB-29 preset diagnostic bundle for FB-27 profile plus plots
  -> FB-30 diagnostic-long span preset with BBN-bound checks
  -> FB-31 chain-local dynamic collision radial-grid cache reuse
  -> FB-32 deterministic collision-reference vectorization
  -> FB-33 batched angular deterministic collision-reference dispatch
  -> FB-34 dynamic source-factory cache/pretabulation reuse
  -> FB-35 dynamic cache-hit overhead removal
  -> FB-36 dynamic live-source full-chain/AP68 E2E BBN surface wiring
  -> FB-37 dynamic E2E BBN readout profile plus diagnostic plots
  -> FB-38 optional FB37 attachment through AP75/AP79 publication readiness surfaces
  -> FB-39 executable FB37 -> AP75 -> AP79 dynamic E2E BBN readiness chain
  -> FB-40 single-command FB27 -> FB37 -> FB39 dynamic E2E BBN smoke bundle
  -> FB-41 smoke-vs-extended FB40 dynamic E2E BBN comparison artifact
  -> FB-42 single-command smoke+extended FB40 span suite plus FB41 comparison
  -> FB-43 current dynamic E2E BBN figure bundle from FB42/FB37 manifests
  -> FB-44 current FB42 -> FB43 figure generation pipeline, no legacy plotting route
  -> FB-45 AP75/AP79-to-FB44 input bundle resolver with optional AP77 rebuild
  -> FB-46 one-shot AP75/AP79 -> FB45 -> FB44 current figure run
  -> FB-47 paper/report-intent current publication figure set over FB46 artifacts
  -> FB-48 one-shot AP75/AP79 -> FB46 -> FB47 current publication-intent figure run
  -> FB-49 MeV coverage and Yp-sign physics diagnostic figures over current profiles
  -> FB-50 LRS CL0 no-collision full-BBN baseline through T_gamma ~= 0.005 MeV
  -> FB-51 progressive freedom full-BBN ladder: weak, non-LRS, collision single/pair rows plus non-LRS collision guard
  -> FB-52 private S2 residual non-LRS collision full-BBN ladder row
  -> FB-53 private residual full-BBN q/angular/relaxation resolution ladder
  -> FB-54 private residual terminal-state vs AP65 same-state source comparator
  -> FB-55 private AP65 same-state source to AP4 terminal-payload comparator
  -> FB-56 single-command AP4 terminal-payload gate plus FB55 comparator
  -> FB-57 optional FB56 gate attachment to the FB23 downstream witness
  -> FB-58 full-BBN physics figure bundle from FB50/FB52/FB53/FB56 artifacts
  -> FB-59 optional FB58 figure-manifest attachment to the FB23 downstream witness
  -> FB-60 full-BBN diagnostic suite bundle over FB50/FB52/FB53/optional FB56/FB58
  -> FB-61 optional FB60 suite-manifest attachment to the FB23 downstream witness
  -> FB-62 optional FB60 suite attachment through AP75/AP79 publication-readiness surfaces
  -> FB-63 optional FB60/FB58 figure-input propagation through FB45/FB46/FB48
  -> FB-64 consolidated full E2E BBN remaining-work plan
  -> FB-65 hash-checked full-BBN figure-input index over FB48/FB63 evidence
  -> FB-66 full-BBN freedom-ladder sweep index over FB51/FB52 rows
  -> FB-67 trajectory-level residual/AP65 closure checkpoint artifact
  -> FB-68 dynamic AP65 collision-payload hot-path profile
```

| WBS package | Existing AP owner | Depends on | Implementation focus | Exit evidence |
|---|---|---|---|---|
| FB-01 | AP4/AP65 | D0 | Export compact terminal-state payloads from AP4/AP65 candidate cases, including `Sigma_+`, `Sigma_-`, `A_modes`, temperatures, network abundances, electron charge-asymmetry state, terminal source diagnostics, and restart metadata. | Landed in current workspace (stage-scoped): focused tests prove JSON-safe payloads are finite, reconstruct a restart config, and preserve fixed/charge-neutral plus scalar-QED controls. |
| FB-02 | AP4/AP65 plus JAX/Rodas5P | FB-01 | Add a checkpointed chained-window runner that advances through multiple physical windows by feeding each terminal payload into the next window, while fixing a CPU-JAX/Rodas5P replay state layout for piecewise/pretabulated source payloads. | Landed in current workspace (stage-scoped): the FB-02 artifact chains multiple windows with full `Sigma/A/T/X/electron` handoff, verifies deterministic restart/resume equivalence, emits one CPU-JAX/Rodas5P replay vector per successful window, and includes a smoke CLI. |
| FB-03 | AP4/AP65 | FB-02 | Replace hand-written subspan schedules with an adaptive source-refresh scheduler driven by temperature, collision `dA`, abundance drift, and source-evaluation budgets. | Landed in current workspace (stage-scoped): the chained runner now supports `source_refresh_strategy="adaptive_budget"`, records driver estimates/counts and budget caps per window, preserves uniform mode, and exposes CLI controls for adaptive thresholds. |
| FB-04 | AP4 plus JAX/Rodas5P | FB-02, FB-03 | Tie the chained runner to CPU-JAX/Rodas5P repeated execution and the production BBN thermal-span conventions, then extract BBN observables such as `Yp`, `D/H`, `N_eff`-style radiation summaries, terminal network rows, and Schramm coordinates. | Landed in current workspace (stage-scoped): every chained window now runs an executable CPU-JAX/Rodas5P pretabulated window-map replay solve and records replay pass/error metadata, the runner now has an opt-in CPU-JAX/Rodas5P live-source RHS sidecar that evaluates nonlinear non-LRS S2 logit transport, 3T/Hubble thermo, live weak monopole rates, and the PRIMAT phase-2 network inside the RHS, and the artifact emits finite terminal `Yp`, `D/H`, `N_eff_3T`, `Sigma_H`, and Schramm coordinates from both the chained phase-2 terminal state and the enabled live-source sidecar final state, including sidecar-vs-terminal observable deltas and restart kwargs that can seed the next live-source window.  A dedicated CPU-JAX live-source chain runner and JSON artifact CLI now verify that handoff across consecutive smoke windows without inserting a pretabulated window-map replay between them, the FB-04 artifact can optionally attach that chain as finite-delta diagnostic comparison evidence against the same piecewise/window-map rows with frozen terminal collision payloads or opt-in `dynamic_restart_state` payload rebuilds, supplied/applied/dynamic payload counts, and per-window provenance fingerprints, and `rodas5p_repeated_run_source="live_source_rhs_chain"` can use the live-source chain as the staged repeated-run evidence/readout source.  FB-21 now hardens that opt-in source as a diagnostic repeated-run gate, and FB-36 carries the dynamic refresh evidence into this full-chain surface; default repeated-run replacement, full collision-coupled JAX RHS validation, and production-calibrated full-span BBN yield validation remain open. |
| FB-05 | AP4/AP65 | FB-02 | Run continuous `live_rhs` only on micro-windows where it is tractable, then compare terminal source, thermo, and network deltas against `piecewise_frozen`. | Landed in current workspace (stage-scoped): `augmented_nonlrs_live_rhs_micro_window_comparator_ap4_fb05_v1` rebuilds each chained restart as a live-RHS AP65 micro-window, records live-minus-piecewise thermo/network/source/kinetic deltas, classifies pass/live-RHS/budget/non-finite/delta failures, and includes a smoke CLI.  A CPU smoke over two `(1e-14)` windows passed with zero `T_gamma` delta, `8.326672684688674e-17` `Xn` delta, max live source evaluations `7`, and no violations. |
| FB-06 | AP6/AP4 | FB-02 | Promote the current radial number/pair-energy closure checks from terminal-row diagnostics to a per-window ledger. | Landed in current workspace (stage-scoped): `augmented_nonlrs_collision_ledger_over_chain_ap4_fb06_v1` records per-window and cumulative closure residuals for finite-mass electromagnetic and all-nine diagonal `nu-nu` rows, with fail-closed source-count/projection/residual classifications and a smoke CLI.  A CPU smoke over two `(1e-8)` windows passed with no violations, `radial_em_energy_closure_residual_abs_max=8.445326839929337e-19`, `radial_nunu_max_abs_number_moment_max=9.846758011831241e-21`, and `radial_offdiagonal_nunu_pair_max_abs_energy_residual_max=1.0482032722271967e-19`. |
| FB-07 | AP7/AP77/AP80 | FB-02 | Apply the current non-LRS S2 CL3 weak-rate mode throughout the chained windows and run weak-rate convergence/profile diagnostics on the same rows. | Landed in current workspace (stage-scoped): terminal chained states now preserve live weak-rate and CL3 rate-application metadata, and `augmented_nonlrs_chained_weak_rate_convergence_ap4_fb07_v1` runs paired same-CL3 `metadata_only` and `nonlrs_s2_cl3_quadrupole_input` chains with bounded per-window lambda and abundance deltas.  A CPU smoke over two `(1e-8)` windows passed with no violations, `lambda_np_relative_delta_abs_max=6.910814616395567e-05`, `lambda_pn_relative_delta_abs_max=4.6906512614639065e-05`, and `Xn_final_delta_abs_max=4.3565151486291143e-13`. |
| FB-08 | AP4/AP6 | FB-02 | Run the fixed/charge-neutral electron-bath and finite/exact scalar-QED cross-product on chained windows. | Landed in current workspace (stage-scoped): `augmented_nonlrs_electron_qed_chain_cross_product_ap4_fb08_v1` runs the four `fixed/charge_neutrality x finite_mu_scaled/exact_finite_mu_scalar` control combinations over chained AP4/AP65 rows, records per-window evolved charge-asymmetry state handoff and scalar-QED one-hot diagnostics, and includes a smoke CLI.  A CPU smoke over two `(1e-8)` windows passed with no violations, `combination_count=4`, `row_count=8`, `charge_neutrality_evolved_rows=4`, and `exact_scalar_qed_rows=4`. |
| FB-09 | AP5/AP66 | FB-02, FB-03 | Extend `N_q`, `N_mu`, `N_phi`, and available PSTF/angular ladders from single-window smoke to chained-window diagnostics. | Landed in current workspace (stage-scoped): `augmented_nonlrs_chained_resolution_ladders_ap4_fb09_v1` runs chained AP4/AP65 rows over `N_q`, `N_mu`, and `N_phi` ladders, records terminal `Yp`, `D/H`, `N_eff_3T`, and `Sigma_H` adjacent deltas plus source-evaluation/replay budgets, and includes a smoke CLI.  A CPU smoke over `q=(3,4)`, `N_mu=(3,4)`, `N_phi=(5,6)`, and two `(1e-8)` windows passed with no violations, `row_count=6`, `converged_ladders=3`, `source_evaluations_total=12.0`, and `terminal_observable_delta_abs_max=1.679316997614903e-22`. |
| FB-10 | AP66 | FB-09 | Make AP66 publication-candidate matrix rows consume chained-window artifacts rather than only smoke/single-span rows. | Landed in current workspace (stage-scoped): AP66 now accepts an optional FB-09 `augmented_nonlrs_chained_resolution_ladders_ap4_fb09_v1` artifact, validates its passed/no-QKE/not-public-dispatch/not-production-SMC scope, rejects failed or malformed chained rows, and records compact `full_chain_evidence.row_links` that point every chained `N_q`/`N_mu`/`N_phi` row back to the source artifact path, contract, row index, row key, terminal BBN observables, and replay/source-budget metadata. |
| FB-11 | AP67 | FB-10 | Add known-limit atlas rows for FLRW/null, LRS limit, injected stress, and electron/QED controls using full-chain artifacts. | Landed in current workspace (stage-scoped): AP67 now forwards optional FB-09 chained resolution artifacts into nested AP66 evidence, can require AP66 full-chain provenance, records AP66 full-chain artifact contract/path/row-count/row-keys/no-QKE/not-public flags in `reused_evidence_links`, and exposes `readiness_summary.full_chain_evidence_ready`.  A CPU smoke with the FB-09 artifact passed all six AP67 known-limit cases with `full_chain_evidence_ready=true` and no limit violations. |
| FB-12 | AP68 | FB-04, FB-10 | Add a guarded forward-model mode that calls the chained runner with artifact-cache controls. | Landed in current workspace (stage-scoped): AP68 now exposes `execution_mode="full_chain"` for guarded inference calls, constructs FB-04 chained-runner specs from the AP68 parameter/config surface, can consume a cached chained artifact without rerunning the builder, validates passed/no-QKE/not-public/not-production-SMC artifact scope, and maps finite terminal `Yp`/`D/H` plus full-chain metadata into `BBNPrediction` without canonical registration.  AP68 can also opt into `full_chain_rodas5p_repeated_run_source="live_source_rhs_chain"`, which requires the FB-04 live-source chain readout and records `full_chain_bbn_readout_source="rodas5p_repeated_run"` while keeping public dispatch disabled.  The same guarded surface can now build or consume the FB-21 live-source repeated-run gate as optional diagnostic evidence, recording the gate contract/path/pass status, collision-payload counts/provenance, finite BBN readouts, live-source-vs-piecewise/window-map deltas, and FB-36 dynamic payload-refresh counters in prediction/SMC metadata without changing the AP68 terminal `Yp`/`D/H` readout or promoting public dispatch.  A CPU smoke over two `(1e-8)` chained windows returned `success=True`, finite `Yp`/`D/H`, `full_chain_completed_windows=2`, and `public_dispatch_ready=False`; an FB-36 AP68 dynamic live-source repeated-run smoke over `(0,5e-11,1e-10)` passed with `dynamic_restart_state`, two built dynamic payloads, finite `Yp`/`D/H`, and `public_dispatch_ready=False` when using the Radau source-generation shell with tighter tolerances. |
| FB-13 | AP69 | FB-12 | Extend the augmented likelihood schema with full-chain solver controls, artifact cache keys, and restart provenance. | Landed in current workspace (stage-scoped): AP69 solver controls now record `execution_mode`, full-chain source-refresh strategy, chained replay/restart toggles, and the FB-36 live-source collision-payload refresh selector; schema metadata and AP71 cache/runtime payloads expose full-chain window edges, cache keys, scheduler controls, execution mode, and `dynamic_restart_state` versus frozen-payload mode, so direct AP65 and full-chain likelihood records do not collide.  A real AP69 likelihood smoke using AP68 `execution_mode="full_chain"` returned finite log-likelihood, `full_chain_completed_windows=2`, and preserved no-QKE/not-public metadata. |
| FB-14 | AP70/AP71 | FB-13 | Run a smoke tempered-SMC warm phase with the full-chain adapter, small particles, checkpoint/restart, and failure-aware likelihoods. | Landed in current workspace (stage-scoped): AP70/AP71 result metadata and the diagnostic CLI now preserve `execution_mode="full_chain"`, full-chain window/cache/source-refresh controls, live-source collision-payload refresh mode, and cached-artifact identities through dry-run and real SMC paths.  A real CLI smoke with two particles, temperatures `(0,1)`, and AP68 full-chain forward calls completed with `finite_loglike_count=2`, `forward_failures=0`, `cache_misses=2`, `full_chain_window_edges=[0,1e-8,2e-8]`, and no public/QKE promotion. |
| FB-15 | AP72 | FB-14 | Re-run synthetic null/recovery SMC with the full-chain adapter and compare to analytic or cached truth rows. | Landed in current workspace (stage-scoped): AP72 keeps synthetic null/injection validation as the default AP73-compatible artifact, and now adds an opt-in `full_chain_physical_forward_smoke` row that runs the AP69/AP70/AP71 SMC path through the AP68 `execution_mode="full_chain"` likelihood.  A real CPU smoke with two physical particles, two `(1e-8)` chained windows, and CPU-JAX/Rodas5P window-map replay verification passed with `finite_loglike_count=2`, `forward_failures=0`, `full_chain_completed_windows=2`, `full_chain_rodas5p_window_map_replay_passed=true`, `Yp=6.824762169972173e-23`, and `D/H=5.174943664962658e-13`.  A follow-up CPU smoke over two `(1e-10)` windows with `full_chain_rodas5p_repeated_run_source="live_source_rhs_chain"` passed with `live_source_repeated_run_readout=true`, `full_chain_bbn_readout_source="rodas5p_repeated_run"`, `full_chain_rodas5p_repeated_run_source_ready=true`, `Yp=4.133815059150155e-25`, and `D/H=5.174943482288938e-13`.  AP72 physical-smoke diagnostics can now require and preserve the FB-21 live-source repeated-run gate contract, diagnostic-only claim scope, no-public/no-production/no-QKE flags, finite repeated-run BBN readouts, same-window gate counts, comparison deltas, and either frozen terminal payload provenance or `dynamic_restart_state` payload request/build/provenance fingerprints; this remains diagnostic and not production SMC. |
| FB-16 | AP73 | FB-10, FB-14 | Populate Schramm/publication artifact tables from real full-chain rows instead of placeholder or synthetic-only rows. | Landed in current workspace (stage-scoped): AP73 now accepts AP72 non-synthetic artifacts only when they include a passed `full_chain_physical_forward_smoke` row, validates no-QKE/no-public/finite-`Yp`/finite-`D/H`/Rodas5P replay checks fail-closed, requires the live-source repeated-run readout check when the AP72 smoke requests `full_chain_rodas5p_repeated_run_source="live_source_rhs_chain"`, and emits a diagnostic `augmented_schramm_response` row with terminal `Yp`, `D/H`, `Sigma_H`, `eta10`, `N_eff_3T`, completed-window count, CPU-JAX/Rodas5P replay status, and optional live-source repeated-run BBN readout provenance.  When FB-21 gate evidence is present or requested, AP73 also requires the current FB-21 contract, diagnostic-only claim scope, no-public/no-production/no-QKE flags, finite repeated-run BBN readouts, same-window gate counts, finite comparison deltas, and supplied/applied/provenance-fingerprinted frozen-collision payloads before carrying the gate provenance into Schramm rows.  Existing cache-only Schramm rows keep their previous `existing_cache_reformatted_only` claim label. |
| FB-17 | AP74 | FB-16 | Render diagnostic publication plots from AP73 tables, including Schramm panels and convergence/source-budget panels. | Landed in current workspace (stage-scoped): AP74 now accepts AP73 Schramm tables labeled `diagnostic_full_chain_physical_smoke_only` or `diagnostic_schramm_rows_mixed_sources` in addition to existing cache rows, renders the full-chain physical-smoke Schramm row through the existing PNG path, and records `full_chain_physical_smoke_present`, completed-window counts, CPU-JAX/Rodas5P replay status, optional live-source repeated-run BBN readout provenance, and FB-21 gate contract/claim/readout/window/payload/delta provenance in plot records and the manifest, with exact completed-window/payload-count/provenance matching.  Labels remain diagnostic/not-promoted. |
| FB-18 | AP75 | FB-17 | Package chained artifacts, plots, configs, provenance, and focused verification outputs into the reproducibility bundle. | Landed in current workspace (stage-scoped): AP75 now accepts the passed AP72 full-chain physical-smoke row together with AP74 full-chain Schramm provenance, validates completed-window, CPU-JAX/Rodas5P replay, optional live-source repeated-run BBN readout fields, and FB-21 gate contract/claim/readout/window/payload/delta provenance across AP72/AP74 summaries, preserves finite terminal `Yp`/`D/H` summaries plus copied-plot FB-21 fields, and rejects orphan, stale, mismatched, AP72/AP74-divergent, or malformed non-synthetic evidence while keeping the bundle diagnostic. |
| FB-19 | AP76/AP79 | FB-18 | Re-run readiness audit against full-chain artifacts and close claim language for every still-blocked surface. | Landed in current workspace (stage-scoped): AP76/AP79 now accepts AP75 bundles with AP72 full-chain physical-smoke evidence and AP74 full-chain Schramm provenance, records that evidence as diagnostic/not-promoted readiness metadata, validates completed-window/replay fields, requires matching FB-21 gate contract/claim/window/payload/delta provenance for any live-source repeated-run BBN readout with every full-chain plot covered, and rejects public dispatch, production-SMC, QKE, or unmatched full-chain provenance. |
| FB-20 | AP4/AP66/AP70/AP76/AP80 | FB-19 | Optional production-candidate gate over already-produced evidence: AP79 full-chain physical-smoke readiness plus AP80 extended coupled weak-rate convergence. | Landed in current workspace (stage-scoped): `augmented_production_candidate_gate_fb20_v1` validates AP79 full-chain readiness, AP72/AP75 completed-window and CPU-JAX/Rodas5P replay evidence, requires and preserves FB-21 gate provenance for live-source repeated-run BBN readout evidence, and validates AP80 extended q-profile pass status, then emits an optional candidate-gate artifact while preserving `not_promoted`, no public dispatch, no production SMC validation, and QKE out of scope. |
| FB-21 | AP4 plus JAX/Rodas5P | FB-04 | Harden the opt-in live-source RHS chain as an explicit repeated-run diagnostic evidence gate over the same tiny chained spans. | Landed in current workspace (stage-scoped): `augmented_nonlrs_live_source_repeated_run_gate_fb21_v1` forces `rodas5p_repeated_run_source="live_source_rhs_chain"`, requires the live-source chain diagnostic, compares finite live-source-chain deltas against the same piecewise/window-map chained rows, requires finite repeated-run `Yp`, `D/H`, `N_eff_3T`, and `Sigma_H`, and by default requires per-window frozen terminal collision payloads to be supplied, applied, and provenance-fingerprinted.  A CPU-JAX/Rodas5P smoke over two `(1e-10)` windows passed with two completed live-source chain windows, two supplied/applied/provenance-fingerprinted collision payloads, finite BBN/comparison deltas, and `public_dispatch_ready=false`. |
| FB-22 | AP4/AP68/AP72/AP73/AP74/AP75/AP76/FB20 | FB-21 | Promote frozen collision-payload provenance from the live-source RHS chain into every downstream diagnostic/readiness surface that consumes FB-21 evidence. | Landed in current workspace (stage-scoped): the live-source RHS chain now fingerprints the full applied `dA_modes` payload, records payload source/provenance per window, and AP68/AP72/AP73/AP74/AP75/AP76/FB20 validators require exact completed-window payload counts, terminal-source provenance, and non-duplicated fingerprints before accepting live-source repeated-run BBN readout evidence.  The FB-21 CLI smoke passed with two provenance-fingerprinted payloads while keeping public dispatch, production SMC, and QKE disabled. |
| FB-23 | AP73/AP74/AP75/AP79/FB20 | FB-22 | Compose the already-landed AP73->AP74->AP75->AP79->FB20 live-source repeated-run evidence path into one deterministic witness artifact. | Landed in current workspace (stage-scoped): `augmented_live_source_repeated_run_evidence_chain_fb23_v1` writes AP73 publication tables, AP74 plots, AP75 bundle, AP79 readiness audit, and FB20 candidate-gate artifacts under one manifest, records the FB-21 payload-provenance summary, and fails closed if any stage loses the no-public/no-production/no-QKE boundary.  This is a diagnostic composition witness, not public dispatch or production-calibrated BBN support. |
| FB-24 | AP4 plus JAX/Rodas5P | FB-21 | Run a deterministic multi-row profile gate over the live-source repeated-run diagnostic path instead of relying on a single tiny smoke layout. | Landed in current workspace (stage-scoped): `augmented_nonlrs_live_source_repeated_run_profile_fb24_v1` wraps the FB-21 gate over multiple tiny `N_window_edges` layouts, requires every row to keep `rodas5p_repeated_run_source="live_source_rhs_chain"`, finite repeated-run `Yp`/`D/H`/`N_eff_3T`/`Sigma_H`, finite BBN/state comparison deltas, exact per-window frozen terminal collision payload provenance, no public dispatch, no production SMC, and QKE out of scope, then records per-row readouts plus observable ranges in one profile manifest/CLI.  This is diagnostic profile evidence only, not default repeated-run replacement or public production support. |
| FB-25 | FB-23/FB-24 | FB-23, FB-24 | Carry the new multi-row profile evidence into the downstream evidence-chain witness without changing readiness or production-candidate semantics. | Landed in current workspace (stage-scoped): `augmented_live_source_repeated_run_evidence_chain_fb23_v1` now accepts an optional FB-24 profile artifact/path, validates the profile contract, diagnostic claim scope, no-public/no-production/no-QKE boundary, all-pass row summary, finite BBN/delta readouts, exact per-row frozen terminal collision-payload provenance, unique fingerprints, and terminal-source payload provenance, then records a passive `fb24_live_source_repeated_run_profile` summary beside the FB-21 gate summary.  FB-25 does not make FB-24 required for FB-23 completion and does not promote public dispatch or production SMC validation. |
| FB-26 | AP4 plus JAX/Rodas5P/AP65 source evaluator | FB-04, FB-21 | Replace stale terminal-only live-source chain collision payloads with an opt-in payload refresh built from the current CPU-JAX/Rodas5P restart state at each window boundary. | Landed in current workspace (stage-scoped): `run_augmented_nonlrs_rodas5p_live_source_rhs_chain(..., collision_payload_refresh_mode="dynamic_restart_state")` now evaluates the existing AP65 combined angular+`pstf_radial` no-QKE collision source at the live-source chain window-start restart state, serializes finite `dQ_nue_pair_N`, `dQ_nux_bank_N`, `dA_modes`, source diagnostics, restart/config fingerprints, and per-window provenance as `dynamic_restart_state`, then freezes that refreshed payload only inside the Rodas5P window.  A CPU-JAX smoke over two `(1e-10)` windows built/applied/provenance-fingerprinted two dynamic payloads with finite BBN readouts while preserving diagnostic-only/no-public/no-production/no-QKE boundaries.  This is window-boundary payload refresh, not intra-step collision-kernel evaluation or public production support. |
| FB-27 | AP4 plus JAX/Rodas5P/AP65 source evaluator | FB-26 | Move beyond one smoke by running the dynamic restart-state live-source chain over increasing smoke-to-extended spans and rendering concrete diagnostic figures. | Landed in current workspace (stage-scoped): `augmented_nonlrs_live_source_dynamic_span_profile_fb27_v1` runs opt-in `dynamic_restart_state` live-source RHS chains over increasing `N_span_end` rows, requires dynamic payloads built/applied/provenance-fingerprinted for every completed window, records finite BBN/shear readouts and per-window `dQ`/`dA` payload summaries, and writes `augmented_nonlrs_live_source_dynamic_span_profile_plots_fb27_v1` PNG/manifest outputs under generated diagnostic output paths.  CPU-JAX profiles through `N_end=(1e-10,2e-10,5e-10)` and an extended `N_end=(1e-9,3e-9,1e-8)` both passed with three rows, six dynamic payloads built/applied, finite BBN readouts, and no public/no-production/no-QKE promotion. |
| FB-28 | FB-23/FB-27 | FB-23, FB-27 | Carry the dynamic span-profile and plot evidence into the downstream evidence-chain witness without changing readiness or production-candidate semantics. | Landed in current workspace (stage-scoped): `augmented_live_source_repeated_run_evidence_chain_fb23_v1` now accepts optional FB-27 profile and plot-manifest inputs as a required pair, validates the dynamic-span contract, diagnostic claim scope, no-public/no-production/no-QKE boundary, increasing span inputs, all-pass rows, dynamic restart-state payload counts/provenance/fingerprints, plot source hash, three PNG plot records, and matched span/summary metadata, then records a passive `fb27_live_source_dynamic_span_profile` summary beside the FB-21/FB-24 evidence.  FB-28 does not make FB-27 required for FB-23 completion and does not promote public dispatch or production SMC validation. |
| FB-29 | FB-27/FB-28 | FB-27, FB-28 | Make longer-span diagnostic runs reproducible by bundling the FB-27 dynamic profile and plots behind named smoke/extended presets. | Landed in current workspace (stage-scoped): `augmented_nonlrs_live_source_dynamic_span_profile_bundle_fb29_v1` writes a profile artifact, plot manifest, and bundle manifest under one output directory, records preset-defining span ends/max-step settings, hashes the profile and plot manifest, validates preset drift fail-closed, and keeps the bundle diagnostic-only/no-public/no-production/no-QKE.  The default smoke preset remains tiny for repeated CI use while the extended preset records the larger `N_end=(1e-9,3e-9,1e-8)` diagnostic span ladder. |
| FB-30 | FB-29 | FB-29 | Push the dynamic live-source span-profile ladder to a longer diagnostic stress preset while preserving claim honesty when finite BBN readouts leave physical bounds. | Landed in current workspace (stage-scoped): `span_profile_preset="diagnostic_long"` runs the same CPU-JAX/Rodas5P dynamic restart-state chain through `N_end=(3e-8,1e-7,3e-7)` with three windows per row and `max_steps=8000`, labels the bundle as `augmented_nonlrs_live_source_dynamic_span_long_probe_bundle_fb30_v1`, records BBN observable-bound metadata, makes live-source chain readouts fail closed on out-of-bound final `Yp`/`D/H`/`Xn`/`Xp`/`N_eff_3T`, and keeps the output diagnostic-only/no-public/no-production/no-QKE.  The long-probe rerun after the CPU-JAX abundance-safe Rodas5P tolerance fix passed with `Yp_final_min=2.475236220294135e-25`, `Yp_final_max=3.7947976780826716e-25`, and zero BBN-bound warning rows over `T_gamma ~= 0.79999976-0.79999998 MeV`. |
| FB-31 | FB-26/FB-30 | FB-26, FB-30 | Reduce repeated dynamic collision-source construction cost without changing collision physics or promoting the live-source path. | Landed in current workspace (stage-scoped): `run_augmented_nonlrs_rodas5p_live_source_rhs_chain(..., collision_payload_refresh_mode="dynamic_restart_state")` now threads one chain-local AP6 radial-grid cache through each dynamic restart-state payload build, records `dynamic_collision_radial_grid_cache_enabled` and `dynamic_collision_radial_grid_cache_entries`, and carries those fields through the dynamic span-profile rows and summary.  A two-window CPU payload probe measured `3.889020878006704 s` without a shared cache versus `1.8382543009938672 s` with the shared cache (`2.1156054828236077x`) while preserving closure contract, source model, and `dA_modes` shape.  This is CPU runtime/cache plumbing only: no new QKE, no public dispatch, no production SMC validation, and no production-calibrated full-span BBN support. |
| FB-32 | AP41/FB-31 | AP41, FB-31 | Reduce cache-hit dynamic payload cost by removing Python scalar loops from the deterministic AP41 collision references. | Landed in current workspace (stage-scoped): `evaluate_nue_scattering_reference`, `evaluate_pair_annihilation_reference`, and `evaluate_nunu_diagonal_twoto2_reference` now evaluate their fixed quadrature sums with NumPy broadcast contractions instead of nested scalar Python loops, while preserving the same Pauli polynomial, interpolation convention, matrix elements, prefactors, moment extraction, and no-QKE diagnostic contracts.  A smoke `N_q=5` benchmark measured `8.68x`, `10.89x`, and `5.53x` speedups for `nu-e`, pair, and diagonal `nu-nu` references, and the FB31 cache-hit dynamic payload probe improved from `0.08017252199351788 s` to `0.04857580701354891 s`.  Scalar-loop parity tests lock all three vectorized kernels against legacy loop outputs; no public dispatch, QKE, production SMC, or full-span BBN claim changes. |
| FB-33 | AP41/FB-32 | AP41, FB-32 | Reduce cache-hit dynamic payload cost by batching AP41 angular collision-reference dispatch across angular nodes/species. | Landed in current workspace (stage-scoped): `evaluate_nue_scattering_reference_batch`, `evaluate_pair_annihilation_reference_batch`, and `evaluate_nunu_diagonal_twoto2_reference_batch` now evaluate batches of fixed-quadrature no-QKE references over shared `q` grids, and the AP41 angular electromagnetic plus pairwise diagonal `nu-nu` bridges use those batch helpers instead of calling scalar deterministic references once per angular node.  A `B=15`, `N_q=5` micro-benchmark measured per-call batch times of `0.00019041859020944686 s`, `0.00020314766035880893 s`, and `0.00019346193003002554 s` versus scalar-loop dispatch times of `0.0015221935498993843 s`, `0.0016838777303928509 s`, and `0.001331207490293309 s` for `nu-e`, pair, and diagonal `nu-nu`.  The same FB31 cache-hit dynamic payload probe improved from the FB33 pre-patch `0.07198696094565094 s` to `0.02897743700305 s`, shifting the remaining hot path toward AP6 radial-grid/provider work.  Batch-vs-single parity and scalar-dispatch guard tests lock the staged bridge behavior; no public dispatch, QKE, production SMC, or full-span BBN claim changes. |
| FB-34 | AP41/FB-33 | AP41, FB-33 | Reduce repeated dynamic payload setup cost by caching AP41 angular plus AP6 radial source factories across same-geometry refreshes. | Landed in current workspace (stage-scoped): `evaluate_augmented_nonlrs_nonlinear_combined_collision_3T_source(..., source_factory_cache=...)` now reuses the AP41 angular source closure and AP6 PSTF radial source closure when the angular grid, species labels, q grid/weights, pair-leg quadrature, gamma, neutrino temperature, radial builder settings, electron chemical-potential mode, scalar-QED/radial normalization controls, eta, and shared radial cache identity match.  The CPU-JAX live-source replay and chain runners thread one chain-local source-factory cache beside the existing radial-grid cache and report enabled/entry metadata plus per-payload hit diagnostics.  A smoke `N_q=5` cache-hit dynamic payload probe improved from radial-cache-only median `0.026258694007992744 s` to source-factory-cache median `0.01560014404822141 s` (`1.6832340731486084x`) while preserving the closure contract and `dA_modes` shape.  This is pretabulation/cache plumbing only; no public dispatch, QKE, production SMC, collision formula, weak-rate, or full-span BBN claim changes. |
| FB-35 | AP41/FB-34 | AP41, FB-34 | Reduce dynamic payload cache-hit overhead further without changing collision physics before pivoting back to full E2E BBN integration. | Landed in current workspace (stage-scoped): source-factory cache entries now retain the factory S2 grid so cache hits can pass the identical grid object through angular/radial closures; the angular bridge skips expensive allclose validation for that identity case; external-q dynamic refreshes defer fixed pair-leg quadrature construction until source-factory cache miss while the cache key records the deterministic pair-leg contract/order; AP6 radial exact diagnostics use exact `array_equal` checks; and AP6 moment weights precompute the number/energy projection bases plus pseudo-inverse.  A cache-hit dynamic payload probe improved from the FB34 median `0.01560014404822141 s` to `0.007451459008734673 s` over 25 repetitions with one source-factory cache entry.  This is cache-hit overhead reduction only; no public dispatch, QKE, production SMC, collision formula, weak-rate, or full-span BBN claim changes. |
| FB-36 | AP4/AP68/FB21 | FB-12, FB-21, FB-26, FB-35 | Consolidate the optimized dynamic live-source RHS chain back into the E2E BBN surfaces instead of continuing low-level optimization only. | Landed in current workspace (stage-scoped): `rodas5p_live_source_rhs_chain_collision_payload_refresh_mode="dynamic_restart_state"` is now forwarded through the FB-04 chained runner CLI, the FB-21 live-source repeated-run gate CLI, and AP68 `execution_mode="full_chain"` via `full_chain_rodas5p_live_source_rhs_chain_collision_payload_refresh_mode`.  Dynamic mode disables terminal collision-payload requirements, records dynamic payload request/build/cache counters in chained artifacts, FB-21 gate artifacts, and AP68 prediction metadata, and preserves diagnostic-only/no-public/no-production/no-QKE boundaries.  A tiny AP68 E2E smoke over `(0,1e-12,2e-12)` passed with the live-source RHS chain as the repeated-run BBN readout and two dynamic payloads built; a larger `(0,5e-11,1e-10)` smoke passed with Radau/tighter source-generation tolerances, while the coarse RK23 version failed before dynamic refresh because the SciPy source-generation shell overshot the deuterium abundance below its handoff bound. |
| FB-37 | AP68/AP72/FB23 | FB-23, FB-36 | Convert dynamic live-source AP68/AP72 E2E BBN readout metadata into plot-ready diagnostic evidence without promoting dispatch. | Landed in current workspace (stage-scoped): `augmented_dynamic_e2e_bbn_readout_profile_fb37_v1` normalizes AP68/AP72 `execution_mode="full_chain"` rows whose readout source is the CPU-JAX/Rodas5P live-source RHS chain with `dynamic_restart_state` payload refresh, requires no-public/no-production/no-QKE metadata, finite `Yp`/`D/H`/`N_eff_3T`/`Sigma_H`, physical BBN readout bounds, finite live-source-vs-window-map deltas, exact dynamic payload request/build/applied/provenance counts, and unique payload fingerprints.  `augmented_dynamic_e2e_bbn_readout_plots_fb37_v1` renders three PNG diagnostics after cleaning only prior FB37 manifest-listed files or exact FB37 basenames, and the FB-23 evidence-chain witness can passively attach the profile/plot pair.  This is diagnostic plot evidence only, not publication readiness, public dispatch, production SMC validation, QKE, or production-calibrated full-span BBN support. |
| FB-38 | AP75/AP79/FB23 | FB-37 | Make the FB37 profile/plot pair consumable by reproducibility-bundle and readiness-audit surfaces as optional diagnostic attachment evidence. | Landed in current workspace (stage-scoped): AP75 now accepts a complete FB37 dynamic E2E BBN readout profile plus plot manifest only when AP72 full-chain physical-smoke evidence is present, validates the profile contract, no-public/no-production/no-QKE metadata, repeated-run readout value source, dynamic restart-state payload provenance, increasing span ends, unique fingerprints, and exactly three PNG plots, then copies the profile, plot manifest, and PNGs as diagnostic attachments.  AP79 revalidates that attachment from the AP75 bundle and records `dynamic_e2e_bbn_readout_evidence` in `source_bundle` with a dedicated audit check while preserving `not_promoted`.  FB-23 forwards supplied FB37 evidence into AP75/AP79 before attaching its own passive witness summary. |
| FB-39 | FB27/FB37/AP75/AP79 | FB-27, FB-37, FB-38 | Make the dynamic E2E BBN evidence path executable instead of requiring manual artifact composition. | Landed in current workspace (stage-scoped): `augmented_dynamic_e2e_bbn_readout_source_fb39_v1` converts passed FB27 dynamic live-source span-profile rows into the AP68-style metadata consumed by FB37 while preserving dynamic restart-state payload counts, finite BBN/shear readouts, MeV temperature readouts when available, no-public/no-production/no-QKE boundaries, and the diagnostic source contract.  `augmented_dynamic_e2e_bbn_readiness_chain_fb39_v1` writes an AP75 bundle with FB37 profile/plot attachments, runs the AP79 readiness audit with AP77 evidence, and records AP75/AP79 artifact hashes, copied FB37 artifact/plot counts, audit-check names, dynamic payload totals, and retained claim boundaries under one manifest/CLI.  This is diagnostic orchestration only; it does not register public dispatch, promote production SMC, add QKE, or claim production-calibrated full-span BBN support. |
| FB-40 | FB27/FB37/FB39 | FB-27, FB-37, FB-39 | Collapse the dynamic E2E BBN smoke path into one reproducible command that actually generates the FB27 profile, FB37 profile/plots, and FB39 readiness chain in order. | Landed in current workspace (stage-scoped): `augmented_dynamic_e2e_bbn_smoke_bundle_fb40_v1` runs the CPU-JAX/Rodas5P FB27 dynamic span-profile preset, converts the resulting artifact through the FB39 FB27-to-FB37 adapter, writes the FB37 dynamic E2E BBN readout profile plus three PNG diagnostics, and then runs the FB39 AP75/AP79 readiness chain with AP77 evidence under one manifest/CLI.  The manifest records hashes and paths for the generated FB27, FB37, and FB39 artifacts, span/payload summaries, retained no-public/no-production/no-QKE boundaries, and explicit chain checks.  This is still diagnostic smoke-scale orchestration only, not public dispatch, production SMC validation, QKE support, or production-calibrated full-span BBN support. |
| FB-41 | FB40 | FB-40 | Compare actual smoke and extended FB40 bundle manifests so larger-span CPU-JAX/Rodas5P runs leave a reusable diagnostic evidence artifact. | Landed in current workspace (stage-scoped): `augmented_dynamic_e2e_bbn_smoke_comparison_fb41_v1` validates two passed FB40 manifests, requires the extended bundle to cover a strictly larger `max_N_end`, preserves the smoke/extended span ladders, MeV `T_gamma_final` range, BBN readout ranges, dynamic restart-state payload/readout provenance, and six generated FB37 diagnostic PNG paths, and records fail-closed chain checks plus no-public/no-production/no-QKE boundaries.  This is diagnostic smoke-vs-extended comparison evidence only; it is not publication-ready full-span BBN support, public dispatch, production SMC validation, or QKE support. |
| FB-42 | FB40/FB41 | FB-40, FB-41 | Collapse the smoke-plus-extended diagnostic ladder into one reproducible command that runs both FB40 presets and immediately writes the FB41 comparison. | Landed in current workspace (stage-scoped): `augmented_dynamic_e2e_bbn_span_suite_fb42_v1` runs the FB40 smoke bundle and FB40 extended bundle into separate output directories, validates their passed manifests and retained claim boundaries, writes the FB41 smoke-vs-extended comparison, and records the span ratio, MeV temperature range, BBN readout ranges, six-plot inventory, artifact hashes, and fail-closed suite checks in one top-level manifest/CLI.  This is diagnostic span-suite orchestration only; it is not public dispatch, production SMC validation, QKE support, or publication-ready full-span BBN support. |
| FB-43 | FB42/FB37 | FB-37, FB-40, FB-41, FB-42 | Replace legacy report/paper plotting entrypoints for the current dynamic E2E BBN path with a manifest-driven figure bundle over the plots already generated by FB37 inside FB40/FB42. | Landed in current workspace (stage-scoped): `augmented_dynamic_e2e_bbn_figure_bundle_fb43_v1` consumes an FB42 span-suite manifest, validates the FB40 smoke/extended bundle hashes, FB41 comparison hash, FB37 plot-manifest hashes, six PNG plot hashes, zero BBN-bound warnings, and retained no-public/no-production/no-QKE claim boundary, then copies the current FB37 diagnostic PNGs into a clean output bundle with a top-level manifest.  It deliberately does not call the legacy report/paper plotting scripts and remains diagnostic figure inventory only, not public dispatch, production SMC validation, QKE support, or publication-ready full-span BBN evidence. |
| FB-44 | FB42/FB43 | FB-42, FB-43 | Replace ad hoc figure regeneration with one current pipeline command that first produces the FB42 smoke+extended span suite and then packages the FB43 figure bundle after deleting stale generated figures. | Landed in current workspace (stage-scoped): `augmented_dynamic_e2e_bbn_current_figures_fb44_v1` runs the FB42 span-suite writer, validates the top-level and nested diagnostic-only/no-public/no-production/no-QKE boundaries, cleans the target current figure directory, then runs the FB43 bundle writer and records both artifacts under one manifest/CLI.  It deliberately does not invoke legacy report/paper plotting scripts and remains a current diagnostic figure-generation pipeline only, not public dispatch, production SMC validation, QKE support, or publication-ready full-span BBN evidence. |
| FB-45 | AP75/AP79/FB44 | AP75, AP79, FB-44 | Make existing AP75/AP79 diagnostic evidence directly consumable by the current FB44 figure pipeline, including the common case where AP79 retained only an AP77 summary. | Landed in current workspace (stage-scoped): `augmented_dynamic_e2e_bbn_current_figure_inputs_fb45_v1` validates an AP75 bundle plus AP79 readiness audit, copies verified AP66/AP67/AP72/AP74 artifacts into a stable FB44 input directory, copies a full AP77 gate when AP79 references one, otherwise records `fb44_ready=false` with a recommended AP77 regeneration command or opt-in rebuilds AP77 from the AP79 summary.  The FB44 CLI can now consume this input bundle directly.  This remains diagnostic input preparation only, not public dispatch, production SMC validation, QKE support, or publication-ready full-span BBN evidence. |
| FB-46 | AP75/AP79/FB45/FB44 | AP75, AP79, FB-45, FB-44 | Collapse current dynamic E2E BBN figure regeneration from AP75/AP79 evidence into one reproducible command. | Landed in current workspace (stage-scoped): `augmented_dynamic_e2e_bbn_current_figure_run_fb46_v1` runs FB45 input preparation, requires `fb44_ready=true`, then runs the FB44 current figure pipeline and records the FB43 figure inventory, copied figures, retained no-public/no-production/no-QKE boundaries, and `legacy_plot_generators_used=false` in one manifest/CLI.  This is diagnostic current figure orchestration only, not public dispatch, production SMC validation, QKE support, or publication-ready full-span BBN evidence. |
| FB-47 | FB46/FB42 plus paper/report figure intent | FB-46, FB-42 | Start the new plotting layer by translating paper/report figure meaning into current artifact-backed plots without reusing legacy plotting code. | Landed in current workspace (stage-scoped): `augmented_dynamic_e2e_bbn_publication_current_plots_fb47_v1` reads an FB46 run and its FB42/FB37 profile artifacts, renders three new PNGs for current observable span response, thermo/shear span context, and dynamic payload stability audit, and records paper/report intent references such as `paper:fig:constraint`, `report:fig:observable_response`, `paper:fig:dynamics`, `report:fig:story_background`, `paper:fig:ablation`, and `report:fig:convergence`.  The manifest explicitly records `legacy_plot_code_reused=false` and `publication_figure_ready=false`; this remains diagnostic current-figure evidence only, not public dispatch, production SMC validation, QKE support, or publication-ready full-span BBN evidence. |
| FB-48 | AP75/AP79/FB46/FB47 | AP75, AP79, FB-46, FB-47 | Collapse the current publication-intent figure path into one reproducible command from AP75/AP79 evidence. | Landed in current workspace (stage-scoped): `augmented_dynamic_e2e_bbn_publication_figure_run_fb48_v1` runs FB46 current figure regeneration, then runs FB47 publication-intent current plots, preserving FB46/FB47 provenance, FB37 profile hashes, paper/report intent metadata, and retained `legacy_plot_generators_used=false` / `legacy_plot_code_reused=false` boundaries.  This is diagnostic figure orchestration only, not public dispatch, production SMC validation, QKE support, or publication-ready full-span BBN evidence. |
| FB-49 | FB48 plus FB27/FB30 profiles | FB-48, FB-47, FB-30 | Add physics-coverage figures that answer whether the current diagnostic evidence touches the MeV freeze-out window, keeps included `Y_p` non-negative, and reaches the full nucleosynthesis temperature track. | Landed in current workspace (stage-scoped): `augmented_dynamic_e2e_bbn_publication_physics_figures_fb49_v1` reads the FB48 source profiles plus optional long-span and excluded historical diagnostic profiles, renders a `Y_p` sign/span-stability figure and a MeV temperature-coverage figure, records included/excluded row counts, included/excluded `Y_p` sign violations, `T_gamma` MeV range, freeze-out-window coverage, and `full_nucleosynthesis_mev_coverage_ready=false`.  Historical negative-`Y_p` probes can be shown only as excluded diagnostic failure context; included current rows must remain finite and non-negative.  This remains diagnostic physics coverage only, not public dispatch, production SMC validation, QKE support, or publication-ready full-span BBN evidence. |
| FB-50 | LRS baseline/debugging | D0, canonical LRS BBN | Establish the no-truncation baseline requested by the `Y_p` sign debugging loop: weak-rate corrections off, neutrino collision terms off, LRS model, full BBN down to the real post-BBN temperature range. | Landed in current workspace (stage-scoped): `augmented_lrs_no_collision_full_bbn_baseline_fb50_v1` and `scripts/run_augmented_lrs_no_collision_full_bbn_baseline.py` compare canonical SciPy, canonical JAX characteristic, and standalone extended LRS full-BBN rows over `Sigma_H=(0,0.01,0.05)`, require `correction_level=0`, `enable_collisions=false`, standard mass-fraction readouts, and `positivity_policy=raw_solver_abundances_no_observable_truncation`.  Canonical rows are recorded as `canonical_forward_solver_observables` because that surface does not expose a terminal phase-2 vector; standalone extended-LRS rows preserve raw final `X_phase2` readouts, zero raw-readout deltas, and `T_final_MeV` evidence through `0.005 MeV`.  The real CPU run passed with `Y_p=0.2423494053--0.2423927149`, `D/H=2.4887625269e-5--2.4890647584e-5`, `T_final_MeV_min=0.004996944944314105`, `raw_abundance_evidence_rows=3`, all `Y_p>0`, all `D/H>=0`, `max_abs_reference_delta_Yp=2.6441819643313602e-05`, and `max_abs_reference_delta_DH=1.3893467615459332e-09`.  This is diagnostic baseline evidence only, not public dispatch, production SMC validation, QKE support, weak-corrected support, or collision-coupled full-BBN support. |
| FB-51 | Progressive freedom debugging | FB-50, canonical LRS/non-LRS/collision surfaces | Re-expand the physical degrees of freedom one at a time after the clean FB50 baseline: weak-rate corrections, non-LRS geometry, and LRS neutrino collision terms, then test supported two-freedom combinations before attempting the full all-three path. | Landed in current workspace (stage-scoped): `augmented_progressive_freedom_full_bbn_fb51_v1` and `scripts/run_augmented_progressive_freedom_full_bbn_ladder.py` run eight rows: LRS CL0/no-collision baseline, weak-CL3 LRS, collisionless non-LRS, LRS collision, weak+non-LRS, weak+LRS-collision, and guarded non-LRS+collision/all-three rows.  The real CPU run passed all six supported rows through `T_final_MeV_min=0.0049999999964358745` with `Y_p=0.24103996070074077--0.24854536776259234`, `D/H=2.4792648090988874e-5--2.5233496342152163e-5`, `single_toggle_supported_passed=3`, `pair_toggle_supported_passed=2`, `unsupported_guarded_rows=2`, and all supported `Y_p>0`, `D/H>=0`.  The canonical guard remains explicit: `jax_characteristic_nonlrs` rejects `enable_collisions=True`, so non-LRS collision-coupled and all-three full-BBN support remain blocked on implementing collision-coupled non-LRS transport rather than silently falling back to an LRS SciPy row. |
| FB-52 | Private non-LRS collision full-BBN | FB-51, private non-LRS residual-state JAX surface | Replace the guarded non-LRS+collision/all-three ladder rows with a real private CPU-JAX phase-split full-BBN residual-state solve while keeping canonical/public dispatch closed. | Landed in current workspace (stage-scoped): `JAXNonLRSResidualFullBBNConfig` and `run_nonlrs_tier2_residual_full_bbn_jax` extend the existing explicit S2 per-species residual collision state from smoke-span evidence to a phase-split full-BBN solve.  Phase 1 evolves weak freeze-out from `10 MeV` to `0.08 MeV`; phase 2 hands off to the PRIMAT network and reaches `T_gamma ~= 0.005 MeV`.  `scripts/run_augmented_progressive_freedom_full_bbn_ladder.py --nonlrs-collision-mode staged_residual` emits `augmented_progressive_freedom_full_bbn_fb52_v1`; the real CPU run passed all eight rows with `nonlrs_collision_residual: Y_p=0.24177934397238576, D/H=2.48084394432579e-5`, `all_three_residual: Y_p=0.2479422750286894, D/H=2.5135990615031494e-5`, `T_final_MeV_min=0.004999999959857061`, `supported_passed_rows=8`, `unsupported_guarded_rows=0`, and `final_all_three_supported=true`.  This is private diagnostic residual-state evidence only: public `canonical_forward_solver(backend="jax_characteristic_nonlrs", enable_collisions=True)` remains guarded, production SMC validation and QKE remain out of scope, and the next blocker is resolution/physics validation of the residual closure. |
| FB-53 | Private residual full-BBN resolution ladder | FB-52 | Convert the FB52 single private residual all-three smoke row into a fail-closed q/angular-grid/residual-relaxation full-BBN resolution artifact before any public dispatch, production SMC, or publication-level plot claim. | Landed in current workspace (stage-scoped): `augmented_nonlrs_residual_full_bbn_resolution_fb53_v1` and `scripts/run_augmented_residual_full_bbn_resolution_ladder.py` run private CPU-JAX/Rodas5P full-BBN residual rows over `N_q`, `(N_theta,N_phi)`, and `residual_relax` ladders, reuse duplicate baseline solve points, record per-row `Y_p`, `D/H`, `N_eff`, `T_final_MeV`, residual-state amplitudes, residual weighted-mean closure, phase event diagnostics, adjacent observable/residual deltas, and diagnostic-only claim boundaries.  The real CPU run over `q=(12,16)`, angular grids `(4,6),(6,8)`, and `residual_relax=(0.5,1.0)` passed all six rows with `unique_solve_points=4`, `prediction_cache_hits=2`, `T_final_MeV=0.004999999968892998--0.005000000053785873`, `max_abs_adjacent_delta_Yp=2.056706482561621e-05`, `max_abs_adjacent_delta_DH=1.4852111311029742e-08`, `max_abs_adjacent_delta_N_eff=0.00015827049378192015`, `residual_weighted_mean_abs_max=0.0`, and `stage_scoped_landed_surface_ready=true`.  A tiny event-refinement tolerance accepts `T_final` roundoff at the `1e-10 MeV` level without truncating abundances.  This remains private diagnostic residual-resolution evidence only; public dispatch, production SMC validation, QKE, and publication-ready all-freedom support remain out of scope, and the next blocker is same-state comparison of the residual closure against AP65 deterministic collision sources. |
| FB-54 | Private residual/AP65 same-state source comparator | FB-53, AP65 combined source evaluator | Compare the terminal private residual full-BBN state against the existing AP65 deterministic combined angular+`pstf_radial` source evaluator at the same thermodynamic and shear state, without claiming the residual S2 state and AP65 PSTF q-state are isomorphic. | Landed in current workspace (stage-scoped): `augmented_nonlrs_residual_ap65_same_state_comparator_fb54_v1` and `scripts/run_augmented_residual_ap65_same_state_comparator.py` run private CPU-JAX/Rodas5P residual full-BBN rows, export a JSON-safe terminal projection payload from `run_nonlrs_tier2_residual_full_bbn_jax`, project residual S2 intensities onto the current-ray `{monopole,W_plus,W_minus}` basis with a q-flat diagnostic repeat, call `evaluate_augmented_nonlrs_nonlinear_combined_collision_3T_source` on that same scalar state, and record AP65 closure contract, component/effective `dA` norms, source-factory/radial-grid cache counts, residual-state norms, and explicit non-isomorphism/diagnostic-only boundaries.  The real CPU run over `q=(4,6)`, angular grid `(4,6)`, and `residual_relax=1.0` passed both rows with `ap65_dA_abs_max=1.989939293353786e-05`, `ap65_effective_dA_over_residual_state_abs_max=0.0002918186524464677`, `source_factory_cache_entries=2`, `radial_grid_cache_entries=36`, and `stage_scoped_landed_surface_ready=true`.  This remains a same-state diagnostic source probe only; public dispatch, production SMC validation, QKE, and a proof of residual/AP65 state isomorphism remain out of scope. |
| FB-55 | Private AP65 terminal-payload compatibility comparator | FB-54, AP4/AP65 piecewise terminal-state payloads | Serialize the FB54 same-state AP65 source into the existing AP4 terminal-source payload shape and compare it against real AP4/AP65 `piecewise_frozen` terminal payloads over matched tiny spans, while explicitly forbidding a physics-equivalence claim. | Landed in current workspace (stage-scoped): `augmented_nonlrs_residual_ap65_terminal_payload_comparator_fb55_v1` and `scripts/run_augmented_residual_ap65_terminal_payload_comparator.py` validate the same-state AP65 source's full `terminal_source_dA_modes`, source contract, q-grid/q-weight/A-shape metadata, terminal source moments, AP4 `piecewise_frozen` source-update/subspan provenance, and diagnostic-only claim boundary against AP4 terminal-state payloads produced by the existing piecewise source-refresh surface.  A real CPU run against a matched `N_q=4`, angular `(4,6)`, `residual_relax=1.0` piecewise AP4/AP65 terminal artifact passed one compatibility row with `physical_equivalence_claimed=false` and `dA_abs_max_scale_ratio=0.014786221202355652`.  This closes a payload-contract bridge for downstream diagnostics only; public dispatch, production SMC validation, QKE, residual/AP65 state isomorphism, and AP4-vs-residual physical equality remain out of scope. |
| FB-56 | AP4 terminal-payload gate wrapper | FB-55, AP4/AP65 candidate gate | Make the AP4 `piecewise_frozen` terminal-payload generation plus FB55 comparator reproducible as one diagnostic command instead of a manual two-step sequence. | Landed in current workspace (stage-scoped): `augmented_nonlrs_residual_ap65_terminal_payload_gate_fb56_v1` and `scripts/run_augmented_residual_ap65_terminal_payload_gate.py` build AP4/AP65 `piecewise_frozen` terminal payloads for each requested same-state comparator row, extract terminal states, run the FB55 comparator over the generated states, require all AP4 terminal artifacts and comparator rows to pass, and preserve no-public/no-production/no-QKE boundaries.  A real CPU smoke over `N_q=4`, angular `(4,6)`, and `residual_relax=1.0` passed with one AP4 terminal artifact, one AP4 terminal state, one compatible comparator row, and `stage_scoped_landed_surface_ready=true`.  This is a reproducibility gate for diagnostic payload compatibility only; public dispatch, production SMC validation, QKE, residual/AP65 state isomorphism, and AP4-vs-residual physical equality remain out of scope. |
| FB-57 | Optional FB56 downstream evidence-chain attachment | FB-23, FB-56 | Carry the FB56 single-command AP4 terminal-payload gate into the downstream FB23 diagnostic witness as optional passive evidence, without changing readiness or production-candidate semantics. | Landed in current workspace (stage-scoped): `write_augmented_live_source_repeated_run_evidence_chain` now accepts `residual_ap65_terminal_payload_gate_artifact`, validates the FB56 contract, pass status, empty violations, diagnostic claim scope, no-public/no-production/no-QKE boundary, `not_promoted` decision, AP4 row/state counts, nested FB55 comparator contract/pass status, and no physical-equivalence claim, then records a compact `fb56_residual_ap65_terminal_payload_gate` summary beside the existing FB21/FB24/FB27/FB37 summaries.  The CLI exposes `--residual-ap65-terminal-payload-gate` and dry-run reports the supplied path.  This is passive diagnostic evidence only; it does not make FB56 required for FB23 completion or promote public dispatch, production SMC validation, QKE, residual/AP65 state isomorphism, or AP4-vs-residual physical equality. |
| FB-58 | Full-BBN diagnostic physics figures | FB-50, FB-52, FB-53, optional FB-56 | Move beyond current short-span dynamic profile plots by rendering figures from artifacts that actually reached the post-BBN temperature range. | Landed in current workspace (stage-scoped): `augmented_full_bbn_physics_figures_fb58_v1` and `scripts/plot_augmented_full_bbn_physics_figures.py` consume FB50 LRS no-collision full-BBN, FB51/FB52 progressive freedom full-BBN, FB53 residual full-BBN resolution, and optional FB56 terminal-payload gate artifacts; fail closed on public/production/QKE promotion, negative included `Y_p`/D/H, missing `T_gamma<=0.01 MeV` full-BBN temperature coverage, failed residual resolution, and FB56 physical-equivalence claims; then render three non-legacy PNGs for progressive yields, terminal-temperature coverage, and residual-resolution plus terminal-payload provenance.  A real render over the current FB50/FB52/FB53/FB56 diagnostic artifacts passed with 23 included rows, 17 terminal-temperature rows, `T_final_MeV=0.004996944944314105--0.005000010963688484`, `Y_p=0.24103996070074077--0.24854536776259234`, positive D/H, and `terminal_payload_gate_attached=true`.  This is diagnostic full-BBN physics figure evidence only; it is not public dispatch, production SMC validation, QKE support, AP4-vs-residual physical equality, or publication-ready all-freedom support. |
| FB-59 | Optional FB58 figure-manifest evidence-chain attachment | FB-23, FB-58 | Make the current full-BBN diagnostic figure artifact visible in the higher repeated-run/full-chain evidence surface without rerendering figures or promoting public dispatch. | Landed in current workspace (stage-scoped): the FB23 downstream evidence-chain writer and CLI now accept an optional FB58 full-BBN physics figure manifest, validate the FB58 contract/schema/stage, diagnostic claim scope, no-public/no-production/no-QKE/not-promoted boundary, `publication_figure_ready=false`, `legacy_plot_code_reused=false`, row-level full-BBN temperature coverage, zero included negative-yield rows, residual-resolution coverage, required FB50/FB52/FB53 source refs, optional FB56 provenance consistency, three unique PNG plot records, plot file existence/hash integrity, and `physical_equivalence_claimed=false`, then attach a compact `fb58_full_bbn_physics_figures` summary to the FB23 manifest.  FB58 remains passive optional evidence: it does not affect `chain_complete`, call the renderer from FB23, register public dispatch, claim production SMC validation, add QKE, prove AP4/residual physical equality, or make the all-freedom path publication-ready. |
| FB-60 | Full-BBN diagnostic suite bundle | FB-50, FB-52, FB-53, optional FB-56, FB-58 | Collapse the current staged full-BBN diagnostic evidence into one reproducible suite manifest that validates the existing artifacts and regenerates the current FB58 figures without promoting public dispatch. | Landed in current workspace (stage-scoped): `augmented_full_bbn_diagnostic_suite_fb60_v1` and `scripts/run_augmented_full_bbn_diagnostic_suite.py` consume FB50 LRS no-collision baseline, FB52 staged-residual progressive freedom ladder, FB53 residual resolution ladder, optional FB56 terminal-payload gate, render a nested FB58 full-BBN physics figure manifest, and fail closed on stale/hot terminal temperatures, guarded FB51 input, failed residual-resolution readiness, negative included yields, FB56 physical-equivalence leaks, missing plot files, and stale plot hashes.  A real run over current FB50/FB52/FB53/FB56 artifacts passed with `T_final_MeV=0.004996944944314105--0.005000010963688484`, `all_three_residual_Yp=0.2479422750286894`, `all_three_residual_DH=2.5135990615031494e-05`, zero included sign violations, `fb52_all_three_residual_supported=true`, `fb53_residual_resolution_ready=true`, `fb58_diagnostic_figures_ready=true`, and three nested FB58 PNGs.  This is a diagnostic suite only: no public dispatch, production SMC validation, QKE support, AP4-vs-residual physical equality, or publication-ready all-freedom support. |
| FB-61 | Optional FB60 suite evidence-chain attachment | FB-23, FB-60 | Carry the current full-BBN diagnostic suite manifest into the higher repeated-run/full-chain evidence witness without making it a readiness or production gate. | Landed in current workspace (stage-scoped): the FB23 downstream evidence-chain writer and CLI now accept an optional FB60 full-BBN diagnostic suite manifest, validate the FB60 contract/schema/stage, diagnostic claim scope, no-public/no-production/no-QKE/not-promoted boundary, `passed=true`, empty violations, full-BBN terminal temperature coverage through `T_gamma<=0.01 MeV`, non-negative included `Y_p` and D/H, FB52 all-three support, FB53 residual-resolution readiness, optional FB56 non-equivalence provenance, nested FB58 manifest content plus manifest hash, three nested FB58 PNG records, manifest-relative path resolution, plot file/hash integrity, and `physical_equivalence_claimed=false`, then attach a compact `fb60_full_bbn_diagnostic_suite` summary to the FB23 manifest.  FB60 remains passive optional evidence: it does not affect `chain_complete`, rerun FB60 from FB23, register public dispatch, claim production SMC validation, add QKE, prove AP4/residual physical equality, or make the all-freedom path publication-ready. |
| FB-62 | Optional FB60 publication-bundle/readiness attachment | AP75, AP79, FB-60 | Carry the current full-BBN diagnostic suite manifest through the higher reproducibility-bundle and readiness-audit surfaces as optional diagnostic evidence, without promoting public dispatch or readiness. | Landed in current workspace (stage-scoped): AP75 now accepts an optional FB60 full-BBN diagnostic suite manifest only when AP72 full-chain physical-smoke evidence is present, validates the FB60 contract/schema/stage, no-public/no-production/no-QKE/not-promoted boundary, full-BBN temperature and sign-safety coverage, FB52 all-three support, FB53 residual readiness, optional FB56 non-equivalence provenance, nested FB58 manifest content, nested FB58 manifest hash, source-relative paths, and compact-vs-full FB58 plot path/hash equality, then copies the FB60 manifest, FB58 manifest, and three FB58 PNGs as diagnostic attachments.  AP79 revalidates the AP75 attachment, requires explicit FB60/FB58 evidence hashes and copied-file hash continuity, records `full_bbn_diagnostic_suite_evidence` in `source_bundle`, and adds a dedicated audit check while preserving `not_promoted`.  FB62 remains optional diagnostic evidence only: it does not register public dispatch, claim production SMC validation, add QKE, prove AP4/residual physical equality, or make the all-freedom full-BBN path publication-ready. |
| FB-63 | Optional FB60/FB58 figure-input propagation | FB-45, FB-46, FB-48, FB-62 | Make the AP75/AP79-carried full-BBN diagnostic suite figures directly discoverable by the current figure/publication run manifests without rerendering or promoting claims. | Landed in current workspace (stage-scoped): FB45 now detects optional AP75 `full_bbn_diagnostic_suite_evidence` only when AP79 `source_bundle.full_bbn_diagnostic_suite_evidence` confirms the same compact summary, rechecks the no-public/no-production/no-QKE/not-promoted/publication-not-ready/physical-non-equivalence boundary, verifies full-BBN temperature and sign-safety fields, hash-checks and copies the AP75-bundled FB60 manifest, nested FB58 manifest, and three FB58 PNGs into a stable `full_bbn_diagnostic_suite` input directory, and records `full_bbn_diagnostic_figure_inputs` outside `fb44_inputs`.  FB46 and FB48 propagate that optional block so higher figure surfaces can consume the real full-BBN PNGs without parsing AP75 directly.  FB63 remains passive diagnostic figure-input attachment only: it does not rerender FB58, call legacy plotting scripts, register public dispatch, claim production SMC validation, add QKE, prove AP4/residual physical equality, or make the all-freedom full-BBN path publication-ready. |
| FB-64 | Consolidated full E2E BBN remaining-work plan | FB-63, roadmap docs, code scan | Collapse scattered partial/stage-scoped/missing-physics notes into one executable plan before continuing physical implementation. | Landed in current workspace (stage-scoped): `docs/TYPEI_AUGMENTED_NOQKE_FULL_E2E_BBN_PLAN.md` now separates completed diagnostic surfaces, unimplemented physics blockers, implemented-but-not-connected surfaces, optimization targets, and the FB65-FB76 execution order.  It preserves no-QKE/no-public-production boundaries and keeps CPU-JAX/Rodas5P as the repeated-run target. |
| FB-65 | Full-BBN figure-input index | FB-48, FB-63 | Create a single machine-readable index over the current FB48-carried full-BBN diagnostic suite inputs so future figure code can discover FB60/FB58 manifests and PNGs without parsing AP75/AP79 directly. | Landed in current workspace (stage-scoped): `augmented_full_bbn_figure_input_index_fb65_v1` validates an FB48 publication-figure-run artifact carrying FB63 `full_bbn_diagnostic_figure_inputs`, requires no-public/no-production/no-QKE/not-promoted and `publication_figure_ready=false` boundaries, requires full-BBN endpoint evidence below `0.01 MeV`, rejects negative included yields/sign-violation rows, hash-checks the copied FB60 manifest, nested FB58 manifest, and three FB58 PNGs, and writes a compact `input_index` with role/path/hash provenance.  The CLI `scripts/build_augmented_full_bbn_figure_input_index.py` exposes dry-run and manifest-writing modes.  This is diagnostic input indexing only: it does not rerender FB58, call legacy plotting scripts, register public dispatch, claim production SMC validation, add QKE, prove AP4/residual physical equality, or make the all-freedom full-BBN path publication-ready. |
| FB-66 | Freedom-ladder full-BBN sweep index | FB-51, FB-52 | Promote the existing full-BBN freedom ladder rows into one reusable sweep manifest with completion/failure-region classification and single/pair/all-freedom comparisons. | Landed in current workspace (stage-scoped): `augmented_freedom_ladder_full_bbn_sweep_fb66_v1` consumes an FB51 or FB52 progressive full-BBN ladder artifact, preserves no-public/no-production/no-QKE/not-promoted boundaries, classifies every row as full-BBN completed, guarded-not-supported, or failed with a MeV-region label, records guarded non-LRS collision blockers, computes pairwise interaction residuals against single-freedom baseline deltas, records all-freedom readiness, and exposes `scripts/build_augmented_freedom_ladder_full_bbn_sweep.py` for dry-run and manifest-writing modes.  This is diagnostic sweep indexing only: it does not run a new solver, register public dispatch, claim production SMC validation, add QKE, prove AP4/residual physical equality, or make the all-freedom full-BBN path publication-ready. |
| FB-67 | Trajectory-level residual/AP65 closure checkpoints | FB-54, FB-66 | Move beyond a terminal same-state comparison by evaluating the existing FB54 residual/AP65 source probe at explicit temperature checkpoints and classifying failures by solver instability, projection contract mismatch, or source-physics mismatch. | Landed in current workspace (stage-scoped): `augmented_nonlrs_residual_ap65_trajectory_closure_fb67_v1` wraps the FB54 same-state comparator over decreasing `T_end` checkpoints, records per-window temperature spans, AP65 source-budget agreement, q-flat projection scope/model labels, compact residual/AP65 closure rows, and failure-kind counts.  The CLI `scripts/run_augmented_residual_ap65_trajectory_closure.py` exposes dry-run and artifact-writing modes.  This remains diagnostic checkpoint evidence only: it does not run a continuous AP65 live-source RHS, register public dispatch, claim production SMC validation, add QKE, prove residual/AP65 state isomorphism, or make the all-freedom full-BBN path publication-ready. |
| FB-68 | Dynamic AP65 collision-payload hot-path profile | FB-67, FB-36 | Profile the exact dynamic restart-state collision payload refresh used by the CPU-JAX/Rodas5P live-source chain before attempting a continuous AP65 RHS or radial-kernel optimization. | Landed in current workspace (stage-scoped): `augmented_nonlrs_dynamic_collision_payload_hotpath_profile_fb68_v1` and `scripts/profile_augmented_dynamic_collision_hotpath.py` compare cache-disabled cold execution, shared-cache cold miss, and shared-cache warm hits for `_dynamic_collision_source_payload_from_restart_state(...)`, recording payload contract, closure contract, `dA_modes` shape, effective `pstf_radial` source diagnostics, source-factory/radial-grid cache entries, first-warm cache-hit status, cProfile top rows, and cold/warm timings with cProfile excluded from medians.  A 4-species CPU-JAX smoke with `q=(0.5,1.5,3.0)` passed with shared-cache cold-miss median `1.6804822400445119 s`, warm-hit median `0.00948077195789665 s`, speedup factor `177.2516254485815`, one source-factory cache entry, and 18 radial-grid cache entries.  BD68 follow-up landed exact runtime optimizations rather than a new gate: suppressed non-LRS stage payloads can use packed current-state arrays with raw ndarray `dA_modes`, and static momentum-delta AP6 radial channel grids now assemble all radial tuples through a vectorized exact path while callable p-dependent momentum-delta grids keep the tuple path.  The same standard-3T profile's shared-cache cold miss improved from `0.05483742000069469 s` to `0.030677367001771927 s`; the all-three `current_state` two-window FB70 probe improved selected wall time from `19.31860326998867 s` to `16.103335389052518 s` while preserving no-QKE/no-public/no-production boundaries.  This still does not run collision evaluation inside the JAX RHS, register public dispatch, claim production SMC validation, add QKE, or make the all-freedom full-BBN path publication-ready. |
| FB-69 | Private continuous AP65 source RHS prototype | FB-68, FB-36 | Recompute the AP65 combined angular+`pstf_radial` source from the current RHS/stage state on tiny CPU-JAX-target micro-windows, while making the Python/NumPy AP65 boundary explicit before any jitted public RHS promotion. | Landed in current workspace (stage-scoped): `augmented_nonlrs_continuous_ap65_source_rhs_prototype_fb69_v1` and `scripts/run_augmented_continuous_ap65_source_rhs_prototype.py` run a private host-stepped Rodas5P-tableau prototype with per-RHS current-state AP65 payload rebuilds, source-evaluation trace fingerprints, shared source-factory/radial-grid caches, finite-difference or zero Jacobian diagnostic policy, adjacent step-cap comparisons, frozen-window dynamic-reference deltas, and fail-closed raw state/source trace artifacts.  A real CPU-JAX smoke over `q=(0.5,1.5,3.0)` and `h_max=(5e-11,2.5e-11)` passed two finite-difference rows with 178 source evaluations, 17 source-factory cache entries, 162 radial-grid cache entries, finite positive-bound BBN readouts, step-cap BBN delta abs max `1.1127229005893926e-16`, and reference BBN delta abs max `3.019806626980426e-14`.  This is a private prototype only: it does not reroute the public jitted CPU-JAX/Rodas5P chain, register public dispatch, claim production SMC validation, add QKE, or make the all-freedom full-BBN path publication-ready. |
| FB-70 | Private continuous-AP65 full-BBN span ladder | FB-69, FB-66 | Expand the current-state AP65 RHS prototype over increasing private spans while recording MeV endpoint coverage, raw BBN observable bounds, and freedom/failure classifications before any publication or public-dispatch claim. | Landed in current workspace (stage-scoped): `augmented_continuous_ap65_full_bbn_span_ladder_fb70_v1` and `scripts/run_augmented_continuous_ap65_full_bbn_span_ladder.py` run FB69 over an increasing `N_span_end` ladder, classify `full_bbn_completed`, `completed_hot_endpoint`, and failed rows using the `0.01 MeV` endpoint convention, preserve active freedoms, and fail closed on untruncated negative/out-of-range `Y_p` or D/H.  FB70 now also accepts `h_refinement_factors` so a failed span window can be retried at smaller `h_max` while preserving the failed attempt telemetry and raw failure artifacts inside the same row.  It also has a `freedom_composition_cases` mode that folds single-, pairwise-, and all-freedom comparisons into the same FB70 artifact, embedding each nested raw span history and terminal-row pairwise deltas instead of adding a standalone composition gate.  The current LRS collision-coupled CPU-JAX/Rodas5P private run with `h_max=0.1`, `h_refinement_factors=(1.0,0.5)`, chained windows through `N_span_end=4.8`, trace-boundary abundances, and `frozen_source_jax` passed with `physical_full_bbn_span_ready=true`, terminal `T_gamma=0.009144759667062704 MeV`, positive raw `Y_p=0.1631801917360858`, nonnegative `D/H=2.0942876300288725e-05`, `h_refinement_rows_recovered=1`, and total source evaluations `3657` including the failed refinement attempt.  A composition smoke over weak, non-LRS, weak+non-LRS, and all-three cases remained hot-endpoint only with `T_gamma=0.7252382210315019--0.7252912365246144 MeV` and `physical_full_bbn_span_ready=false`.  FB70 now also preserves stage-domain reject counts, post-rejection step-growth cap telemetry, same-state host-step base RHS/Jacobian cache hit telemetry, and cooperative per-row wall-time budget failures checked between host-step attempts; the current non-LRS+collision two-window smoke reaches the first hot endpoint and then fails closed in the second window at the 60 s row budget with one stage-domain rejection and 21 post-rejection growth caps, confirming that non-LRS full-collision endpoint coverage is still blocked by both endpoint completion and hot-loop payload/Jacobian plus tiny-step cost.  This is private no-QKE continuous-AP65 endpoint/composition/runtime evidence only; FB70 remains not promoted and makes no public dispatch, production SMC validation, QKE, non-LRS full-collision endpoint, or publication-ready all-freedom full-BBN support claim. |

BD69 follow-up keeps FB70 as the single span-ladder surface and adds the private solver policy `chain_h_max_policy=first_rejection_or_recovered_h_ceiling`, which forwards a recovered refined `h_max` after a failed coarse attempt to subsequent chained windows instead of repeating the same failed coarse solve.  A real exact all-three `stage_collision_payload_policy=current_state` CPU-JAX/Rodas5P run with `h_max=0.2`, `h_refinement_factors=(1.0,0.5)`, `frozen_source_jax`, boundary tracing, chain restart handoff, and the BD69 policy passed six selected windows, stopped at `N_span_end=4.8`, reached `T_gamma=0.009144759663514114 MeV`, raw `Y_p=0.16318746257583355`, raw `D/H=2.0957653972447094e-05`, `rows_reaching_endpoint=1`, `physical_full_bbn_span_ready=true`, `selected_source_evaluations_total=4017`, and `selected_wall_seconds_total=65.46224005485419`.  This moves the exact-current-state private all-three endpoint blocker below `0.01 MeV` without using `step_base_reuse`, while public dispatch, production SMC validation, QKE, and publication-ready claims remain unclaimed.
| FB-71 | Private full-BBN weak-rate convergence index | FB-52, AP80 | Index weak-off/weak-on full-BBN rows by active freedom context, preserve raw endpoint and observable checks, and optionally link AP80 profile-level weak-rate convergence evidence without promoting public dispatch. | Landed in current workspace (stage-scoped): `augmented_full_bbn_weak_rate_convergence_fb71_v1` and `scripts/build_augmented_full_bbn_weak_rate_convergence.py` consume an FB51/FB52 progressive freedom full-BBN ladder, build same-context weak-control pairs for LRS/no-collision, non-LRS/no-collision, LRS/collision, and residual non-LRS/collision, require `T_gamma <= 0.01 MeV` plus positive `Y_p` and nonnegative D/H, and report weak-on minus weak-off `Y_p`/D/H deltas.  The first real CPU index over the FB52 artifact passed all four full-BBN weak pairs with `rows_reaching_full_bbn_endpoint=8`, `max_abs_weak_delta_Yp=0.006193828014246616`, and `max_abs_weak_delta_DH=3.452364587852889e-07`; `ap80_to_full_bbn_bridge_ready=false` because no AP80 JSON artifact was supplied.  FB71 is diagnostic index evidence only: it does not run a new solver, register public dispatch, claim production SMC validation, add QKE, prove all-freedom publication readiness, or remove promotion-grade weak-rate blockers. |
| FB-72 | AP80-FB71 full-BBN weak-rate bridge | FB-71, AP80 | Compose AP80 profile-level weak-rate convergence evidence with the FB71 full-BBN weak-pair index before V2 figures consume weak-rate claims. | Landed in current workspace (stage-scoped): `augmented_full_bbn_weak_rate_bridge_fb72_v1` and `scripts/run_augmented_full_bbn_weak_rate_bridge.py` generate or consume AP80 smoke/extended profile evidence, build a nested FB71 index with AP80 supplied, and require AP80/FB71 agreement on profile count, profile names, and applied-rate q-ladder delta before `ap80_fb71_bridge_ready=true`.  A real CPU smoke over the FB52 full-BBN freedom ladder passed with `ap80_profile_count=1`, `ap80_total_nfev=7596`, `ap80_applied_rate_q_relative_delta_abs_max=0.0024445680701901517`, `fb71_passed_pair_count=4`, `fb71_rows_reaching_full_bbn_endpoint=8`, and `ap80_fb71_bridge_ready=true`.  FB72 remains a private diagnostic bridge only: AP80 is still profile/tiny-span evidence, not a full-BBN convergence proof, and public dispatch, production SMC validation, QKE, promotion-grade weak convergence, and publication-ready claims remain out of scope. |
| FB-73 | Publication figure renderer V2 | FB-60, FB-66, FB-70, FB-72 | Rewrite the current full-BBN figure path around current artifacts instead of legacy plot modules, with provenance hashes and explicit diagnostic/publication claim labels. | Landed in current workspace (stage-scoped): `augmented_publication_figure_renderer_v2_fb73_v1` and `scripts/render_augmented_publication_figures_v2.py` consume FB60 full-BBN suite, FB66 freedom-ladder sweep, FB70 continuous-AP65 span ladder, and FB72 weak-rate bridge artifacts; validate no-public/no-production/no-QKE/not-promoted boundaries; and render four hashed PNG panels for endpoint coverage, freedom-ladder terminal yields, weak-rate bridge deltas, and continuous-AP65 span boundary.  A real render passed with `artifact_payload_sha256=6a62a214b0d13e38e5d76bac652c8ce02caf4ec93880157b49908a75a567ceae`, `plot_count=4`, `full_bbn_T_final_MeV=0.004996944944314105--0.005000010963688484`, `freedom_sweep_completed_rows=8`, `weak_rate_bridge_passed_pair_count=4`, and `publication_readiness_blocker=continuous_ap65_full_bbn_span_not_ready`.  FB73 is diagnostic current-artifact figure evidence only: it does not run a solver, reuse legacy plotting code, register public dispatch, claim production SMC validation, add QKE, or make the all-freedom full-BBN path publication-ready. |
| FB-74 | Publication figure bundle QA | FB-73 | Validate and package the FB73 current-artifact figures as a reproducible QA bundle without rerendering or promoting publication readiness. | Landed in current workspace (stage-scoped): `augmented_publication_figure_bundle_qa_fb74_v1` and `scripts/package_augmented_publication_figure_bundle_qa.py` consume an FB73 manifest, recompute its embedded payload hash, verify source artifact and PNG hashes, reject caption/claim-label overclaims, reject output directories overlapping FB73 manifest, plot, or source-artifact directories, copy the four FB73 PNGs into a clean QA bundle, and write explicit QA check rows.  A real QA run passed with stable rerun hashes `artifact_payload_sha256=e22a8f6ea68b24e376b1ded12b6bb531199005bead8b4b7ef6d187f76f645e45` and manifest file SHA256 `d609ba75756bbb9be0c7dd1fa256b6ad167eda1974a005527f8a08354c664cd5`, source FB73 payload SHA256 `6a62a214b0d13e38e5d76bac652c8ce02caf4ec93880157b49908a75a567ceae`, `plot_count=4`, `copied_plot_count=4`, and `qa_checks=10`.  FB74 is diagnostic QA evidence only: it does not rerender figures, call legacy plotting scripts, run a solver, register public dispatch, claim production SMC validation, add QKE, or make the all-freedom full-BBN path publication-ready. |
| FB-75 | Guarded SMC pilot gate | AP-72, FB-60, FB-66, FB-70, FB-72, FB-74 | Connect validated full-BBN diagnostic products into a fail-closed statistical-pilot input-readiness manifest without running a new sampler or promoting public dispatch. | Landed in current workspace (stage-scoped): `augmented_guarded_smc_pilot_gate_fb75_v1` and `scripts/build_augmented_guarded_smc_pilot_gate.py` consume the AP72 full-chain physical-smoke validation artifact, FB60 full-BBN diagnostic suite, FB66 freedom-ladder sweep, FB70 continuous-AP65 span ladder, FB72 AP80-to-FB71 weak-rate bridge, and FB74 figure QA bundle; require file-backed source hashes; recompute embedded payload hashes for hashed inputs; reject stale source artifacts, over-open public/production/QKE boundaries, missing AP72 physical-smoke evidence, non-full-BBN FB60/FB66/FB72 products, non-integral count fields, mapping-only unhashable sources, and inconsistent FB70 physical-span claims; and emit an AP69 SMC schema snapshot plus source hashes for a guarded diagnostic pilot handoff.  A real current-artifact run passed with `artifact_payload_sha256=18841d947067979eb5cdfddeef1a4c55656fbc62e92257a6e63197820bfea352`, manifest file SHA256 `6087af94215ff25628c18e7a5fa3fd9a22ae2981166ec0ae467d6c036e661922`, `guarded_smc_pilot_input_ready=true`, `validated_full_bbn_product_inputs_ready=true`, `statistical_pilot_input_ready=true`, `runs_new_smc_sampler=false`, `source_hashes_checked=true`, `ap72_full_chain_completed_windows=2`, `fb66_completed_rows=8`, `fb72_rows_reaching_full_bbn_endpoint=8`, and `pilot_blockers=[continuous_ap65_full_bbn_span_not_ready]` because FB70 remains a hot-endpoint continuous-AP65 span diagnostic with `physical_full_bbn_span_ready=false`, `rows_reaching_endpoint=0`, and `T_final_MeV=0.7999999999214282--0.7999999999607141`.  FB75 is a diagnostic pilot-input gate only: it does not run SMC, register candidate or public dispatch, claim sampler readiness, claim production SMC validation, add QKE, or remove the continuous AP65 full-BBN span blocker. |
| FB-76 | Internal candidate dispatch decision | FB-75, AP-68, AP-69, registry | Decide whether an internal fail-closed candidate-dispatch surface is warranted while keeping public/canonical dispatch unchanged. | Landed in current workspace (stage-scoped): `augmented_internal_candidate_dispatch_decision_fb76_v1` and `scripts/build_augmented_internal_candidate_dispatch_decision.py` consume the file-backed FB75 guarded SMC pilot-input gate, hash-check its payload and every FB75 nested source file SHA, verify that the augmented staging capability is present only in `CAPABILITY_BY_KEY` and not in `CAPABILITY_BY_BACKEND`, symbol-check the AP68 internal callable entrypoints without running solves, and record an internal decision plus blockers.  The real current decision artifact passed with `artifact_payload_sha256=d361d9dab63dde7b54fed1656b6d7f61f5a10ec2ffdb2e6467dbb4d3f3b09518`, manifest file SHA256 `2e871e1ec920371779bb22570e69ec4925369e0ef51d3a4f005cbb466604eac3`, `internal_candidate_dispatch_decision=defer`, `internal_candidate_dispatch_warranted=false`, `registers_dispatch=false`, `canonical_forward_solver_registered=false`, `candidate_dispatch_registered=false`, and `decision_blockers=[continuous_ap65_full_bbn_span_not_ready]`.  FB76 is a decision record only: it does not add a backend alias, change `canonical_forward_solver`, run SMC, claim production SMC validation, claim public dispatch, add QKE, or remove the continuous AP65 full-BBN span blocker. |
| FB-77 | Claim readiness review | FB-76, roadmap docs | Record the strongest defensible current claim and remaining blockers without opening public/canonical dispatch or running new physics. | Landed in current workspace (stage-scoped): `augmented_claim_readiness_review_fb77_v1` and `scripts/build_augmented_claim_readiness_review.py` consume the file-backed FB76 internal candidate-dispatch decision, hash-check its payload, require FB76 nested source-hash verification, hash the roadmap claim-boundary docs with FB77 self-reference artifact-hash lines redacted, and record `claim_readiness_level=diagnostic_evidence_chain_ready` with `strongest_defensible_claim_key=guarded_internal_diagnostic_evidence_chain`.  The real current review artifact passed with `artifact_payload_sha256=e38abd1c8de1b7f61755fff396c9465a3679ba0f657676fa2b934b931da06f95`, manifest file SHA256 `13c83abdb5fa0f66f61f2158f5f4e50b9c89d1dbf9a178ac8c41ad2babdddd13`, `public_dispatch_ready=false`, `production_smc_validation_ready=false`, `publication_ready_all_freedom_full_bbn=false`, `qke_scope=out_of_scope`, `registers_dispatch=false`, and `recommended_next_physics_pr=extend_continuous_ap65_full_bbn_span_to_0p01_MeV`.  FB77 is a claim-review ledger only: it does not add a backend alias, change `canonical_forward_solver`, run SMC, run a solver, claim production SMC validation, claim public dispatch, add QKE, or remove the continuous AP65 full-BBN span blocker. |
| FB-78 | Continuous AP65 chained span ladder | FB-69, FB-70 | Turn the private continuous-AP65 span ladder into a consecutive-window diagnostic by handing each passing FB69 terminal restart state to the next window. | Landed in current workspace (stage-scoped): FB69 now accepts supplied restart kwargs, records their fingerprint/source, and emits terminal restart kwargs from finite final states; FB70 adds `chain_restart_handoff` mode so `N_span_end` rungs run as `(previous_end,current_end)` instead of independent starts.  The CLI exposes `--chain-restart-handoff`, and tests lock supplied restart use, terminal restart emission, consecutive window spans, and fail-closed behavior when a failed row has a restart payload.  A real finite-difference CPU-JAX smoke over `N_span_end=(5e-11,1e-10,2e-10,5e-10)` passed four chained windows with `artifact_payload_sha256=463418cba619ef8199b642debcd3425f54a3fd21f24b62038a85ecba5f1e46b9`, manifest file SHA256 `f3c22071c252c990041aea33471db0b52ecb2da59cddf59872300ba84bdc36fa`, `restart_handoff_ready_rows=4`, `source_evaluations_total=588`, `step_count_total=10`, and `T_gamma_final=0.799999999607141--0.7999999999607141 MeV`.  FB78 remains a private hot-endpoint diagnostic with `physical_full_bbn_span_ready=false` and `rows_reaching_endpoint=0`: it does not register public dispatch, claim production SMC validation, add QKE, or make the all-freedom full-BBN path publication-ready. |
| FB-79 | Continuous AP65 span bracket | FB-78, FB-70 | Record a repeatable pass/fail bracket for chained continuous-AP65 span expansion before attempting larger full-BBN endpoint runs. | Landed in current workspace (stage-scoped): `augmented_continuous_ap65_span_bracket_fb79_v1` and `scripts/run_augmented_continuous_ap65_span_bracket.py` run multiple chained FB70 span profiles, require nested no-public/no-production/no-QKE boundaries, preserve nested failure-region evidence, and summarize the last passing profile plus first observed failing endpoint.  A real finite-difference CPU-JAX bracket passed with `artifact_payload_sha256=e1c73bdae84d013a3ac0551bff404716f78bf3fdcd37a337326f3f5740e8df35`, manifest file SHA256 `2cb311b2779809db259d0574b26c1def21057ebbe152ddb97711fdf82d260557`, `bracket_status=pass_fail_bracketed`, `largest_passing_N_span_end=5e-10`, `first_failing_N_span_end=1e-09`, `best_passing_T_final_MeV=0.799999999607141`, and `first_failing_T_final_MeV=0.7999999992142808`.  FB79 remains private diagnostic bracket evidence: the failing profile is still above the `0.01 MeV` endpoint and carries raw nonpositive `Y_p`; no public dispatch, production SMC validation, QKE, or publication-ready all-freedom full-BBN support is claimed. |
| FB-80 | Continuous AP65 h_max sensitivity | FB-79, FB-70 | Determine whether the FB79 first failing endpoint is recovered by internal step refinement before extending the span ladder. | Landed in current workspace (stage-scoped): `augmented_continuous_ap65_hmax_sensitivity_fb80_v1` and `scripts/run_augmented_continuous_ap65_hmax_sensitivity.py` hold the FB70 target span fixed at `N_span_end=1e-9`, sweep strictly decreasing `h_max=(1e-9,5e-10,2.5e-10)`, check nested no-public/no-production/no-QKE boundaries, and fail closed on unexpected nested failure classes.  A real finite-difference CPU-JAX diagnostic passed with `artifact_payload_sha256=84d6aac41fc673889320ebc5802fa78049977917da193ad9154fc487048558e4`, manifest file SHA256 `93a3dc65aa573f0217063fc947084caa97f6c5409e6d592cf8a0d8005a927912`, `classification=h_max_refinement_recovers_observable_failure`, `largest_failing_h_max=1e-09`, `first_passing_h_max_after_failure=5e-10`, `smallest_passing_h_max=2.5e-10`, `rows_failed=1`, and `rows_passed=2`.  FB80 remains private sensitivity evidence: it classifies a hot-endpoint coarse-step failure and does not claim full-BBN completion, public dispatch, production SMC validation, QKE, or publication-ready all-freedom support. |
| FB-81 | Continuous AP65 refined span bracket | FB-80, FB-70 | Apply the refined h_max policy to the span ladder and record the next pass/fail endpoint bracket. | Landed in current workspace (stage-scoped): `augmented_continuous_ap65_refined_span_bracket_fb81_v1` and `scripts/run_augmented_continuous_ap65_refined_span_bracket.py` hold `h_max=2.5e-10`, run chained FB70 endpoints `(5e-10,1e-9,1.5e-9,2e-9)`, check nested no-public/no-production/no-QKE boundaries, and require concrete first-failure evidence.  A real finite-difference CPU-JAX diagnostic passed with `artifact_payload_sha256=76bf833035a9a23f7b444786d19924d7d676d23d2f79c086703faeb0ae3f212e`, manifest file SHA256 `243e1170947e2ce33271c0410169be55f73fa5383de5ac305bb9005e051ab2f9`, `classification=refined_span_pass_fail_bracketed`, `largest_passing_N_span_end=1e-09`, `first_failing_N_span_end=1.5e-09`, `rows_passed=2`, and `rows_failed=2`.  FB81 remains private refined-span bracket evidence: it classifies a hot-endpoint failure and does not claim full-BBN completion, public dispatch, production SMC validation, QKE, or publication-ready all-freedom support. |
| FB-82 | Continuous AP65 first-failure triage | FB-81 | Split the first refined-span `Y_p` failure into strict-sign, abundance-tolerance, observable, restart-handoff, and source-evaluation evidence without relaxing the physical gate. | Landed in current workspace (stage-scoped): `augmented_continuous_ap65_failure_triage_fb82_v1` and `scripts/run_augmented_continuous_ap65_failure_triage.py` rerun FB81, check nested no-public/no-production/no-QKE boundaries, require first-failure BBN observables, and preserve strict `Y_p > 0` as a blocker rather than truncating or repairing abundances.  A real finite-difference CPU-JAX diagnostic passed with `artifact_payload_sha256=c64bf7175a6935b39859ae521a05fadd6548fcfa5e2326d3faff9da1e9f9a783`, manifest file SHA256 `6eed5354131696519b92f3e7ba4c2132cf5f37f9b88e4911ebcabb7012649b0b`, `classification=strict_y_p_sign_failure_within_abundance_tolerance`, `first_failing_N_span_end=1.5e-09`, `Yp=-1.2294890184644955e-30`, `abundance_bound_tolerance=1e-18`, `abundance_bounds_ok=true`, and `bound_tolerance_masks_strict_sign=true`.  FB82 remains private triage evidence: it does not claim full-BBN completion, public dispatch, production SMC validation, QKE, abundance repair, or publication-ready all-freedom support. |
| FB-83 | Continuous AP65 Yp source probe | FB-82 | Localize the first strict-`Y_p` failure by comparing the terminal BBN readout against `He4=X_phase2[5]` in the packed last-attempted FB69 replay-state tail. | Landed in current workspace (stage-scoped): `augmented_continuous_ap65_y_p_source_probe_fb83_v1` and `scripts/run_augmented_continuous_ap65_y_p_source_probe.py` consume FB82, check nested no-public/no-production/no-QKE boundaries, use the live-source replay `X_phase2_shape=(9,)` tail contract rather than sparse observable indices, and fail closed on missing first-failure rows, last-passing rows, or state vectors.  A real finite-difference CPU-JAX diagnostic passed with `artifact_payload_sha256=bf8c39c5947c063cb800c2f2b34f75bb3a70ac311aa3a388d6578a07a6692bb1`, manifest file SHA256 `234d89567922ddad92d0c3750c35522f0c205616bfb3dbb3f47f8885e936d3d5`, `classification=terminal_y_p_sign_crossing_below_tolerance_after_positive_last_stage_he4`, `first_failing_terminal_Yp=-1.2294890184644955e-30`, `first_failing_last_attempted_He4=2.2765668298302704e-32`, `x_phase2_tail_start=41`, and `he4_tail_index=46`.  FB83 remains private source-localization evidence: it does not claim full-BBN completion, public dispatch, production SMC validation, QKE, abundance repair, or publication-ready all-freedom support. |
| FB-84 | Continuous AP65 terminal final-state probe | FB-70, FB-83 | Preserve terminal FB69 final-state `X_phase2` tail evidence in FB70 rows so terminal `Y_p` can be compared directly with accepted final-state `He4`. | Landed in current workspace (stage-scoped): FB70 rows now include `terminal_final_state_probe` with the live-source replay `X_phase2_shape=(9,)` tail, fixed `He4` index 5, terminal observable `Yp`, and their delta; missing final-state vectors produce an unavailable probe.  A real finite-difference CPU-JAX refresh through FB82 passed with nested `artifact_payload_sha256=47efcd214cc16b0810797d19d59baca5ab0a1e965ab169416ac2cdb3fe486609`, manifest file SHA256 `81cfb5fc61419c14306f703326d333135cc34d0a4d172bc545cb27195d065acb`, `terminal_final_state_probe.he4_tail_index=46`, `terminal_final_state_probe.he4_from_final_state_vector=-1.2294890184644955e-30`, `terminal_final_state_probe.terminal_observable_Yp=-1.2294890184644955e-30`, and `terminal_final_state_probe.terminal_y_p_minus_final_state_he4=0.0`.  FB84 remains private provenance evidence: it does not claim full-BBN completion, public dispatch, production SMC validation, QKE, abundance repair, or publication-ready all-freedom support. |
| FB-85 | Continuous AP65 adaptive step acceptance | FB-69, FB-70, FB-84 | Make the private host-stepped continuous AP65 Rodas5P prototype use its embedded error estimate as a real accept/reject gate and preserve retry telemetry through FB70 rows. | Landed in current workspace (stage-scoped): `_run_step_cap_row` now rejects `err_norm > 1` attempts without advancing `N` or `y`, shrinks `h`, retries within the diagnostic budget, records `attempt_count`, `n_rejected`, accepted/rejected `h` samples, rejected error samples, and adaptive-controller metadata, and FB70 span rows preserve those fields.  A real finite-difference CPU-JAX refresh through FB82 passed with nested `artifact_payload_sha256=9a0e0fe58cf8e318777b6b2a3cadae4cc367dd3424b6df75da178ba4a41b04dd`, manifest file SHA256 `9a5d0cd620a4036ed1dc65c20842f6efc068d6de18fcd4091baffb9fad4ebee5`, `first_failure_row.attempt_count=2`, `first_failure_row.n_rejected=0`, `first_failure_row.error_norm_max=3.021584391530104e-14`, and unchanged `Yp=-1.2294890184644955e-30`, ruling out accepted `err_norm > 1` steps as the current smoke-ladder explanation.  FB85 remains private solver-control evidence: it does not claim full-BBN completion, public dispatch, production SMC validation, QKE, abundance repair, or publication-ready all-freedom support. |
| FB-86 | Continuous AP65 He4 RHS boundary probe | FB-82, FB-84, FB-85 | Probe whether the first strict-`Y_p` failure is already present in the phase-2 network RHS at `He4=0` and whether negative trace intermediates drive the boundary sign. | Landed in current workspace (stage-scoped): `augmented_continuous_ap65_he4_rhs_probe_fb86_v1` and `scripts/run_augmented_continuous_ap65_he4_rhs_probe.py` evaluate the JAX phase-2 network RHS at raw terminal/last-attempted `X_phase2` tails plus diagnostic-only `He4=0`, `He4=1e-30`, and nonnegative trace-species counterfactual points.  The contract fails closed on terminal `Y_p` versus final-state `He4` mismatch and floors only trace-species indices in the diagnostic counterfactual.  A real finite-difference CPU-JAX diagnostic passed with `artifact_payload_sha256=ebd16b1fa3b6d4b673e33c2cff07855a075d3ca7f288230ccf2b9fe24b275fdf`, manifest file SHA256 `b170d117620b40dcf413e94b865774b3a6f0c4dc12b2e52d8568130cf3baf201`, `classification=he4_boundary_negative_due_to_negative_trace_intermediates`, `first_failure_negative_trace_indices=[3,4,6,7]`, `first_failure_negative_core_non_he4_indices=[]`, `first_failure_he4_zero_dHe4_network_rhs=-2.618301171321943e-21`, and `first_failure_nonnegative_trace_he4_zero_dHe4_network_rhs=7.273403769914826e-286`.  FB86 remains private RHS-localization evidence: it does not claim full-BBN completion, public dispatch, production SMC validation, QKE, abundance repair, or publication-ready all-freedom support; the next implementation target is positivity-preserving phase-2 network evolution for trace species. |
| FB-87 | Continuous AP65 trace-boundary positivity gate | FB-86, FB-70 | Add an opt-in private phase-2 trace/`He4` lower-bound RHS policy and compare it against the raw ladder over the same smoke spans. | Landed in current workspace (stage-scoped): `abundance_positivity_policy=trace_boundary` keeps the default raw network RHS available, applies only inside private continuous-AP65 RHS evolution, constrains trace/`He4` activities and active lower-bound derivatives, and records metadata without truncating terminal `Y_p`.  FB70 rows now propagate live RHS metadata so FB87 can gate raw-vs-policy phase-2 mass-fraction sum residuals.  The new `augmented_continuous_ap65_trace_positivity_gate_fb87_v1` artifact and `scripts/run_augmented_continuous_ap65_trace_positivity_gate.py` compare raw vs trace-boundary FB70 ladders.  A real finite-difference CPU-JAX diagnostic passed with `artifact_payload_sha256=dcdae7615088893f2bfbbece52620b8d81e60b1e775cb7ca8059c9d65a755276`, manifest file SHA256 `99333c8747f006758fe9da2f0a2c8e633584a3e44a9d111999f8880d149f759d`, `classification=trace_boundary_resolves_smoke_y_p_sign_failure_with_conservation_gate`, raw first failure `N_span=[0.0,1.5e-09]`, raw `Yp=-1.2294890184644993e-30`, raw `Yp` failure rows `2`, trace-boundary failure rows `0`, trace-boundary largest passing endpoint `2e-09`, raw conservation max `6.284872348663924e-18`, trace-boundary conservation max `8.110492019931864e-18`, and conservation limit `1e-16`.  FB87 remains private evolution-policy evidence: it does not claim full-BBN completion, public dispatch, production SMC validation, QKE, terminal abundance repair, or publication-ready all-freedom support. |
| FB-88 | Continuous AP65 trace-boundary span extension | FB-87, FB-70 | Extend the private trace-boundary continuous-AP65 ladder beyond the FB87 smoke endpoint while preserving conservation, stiffness, and solver-effort telemetry. | Landed in current workspace (stage-scoped): `augmented_continuous_ap65_trace_span_extension_fb88_v1` and `scripts/run_augmented_continuous_ap65_trace_span_extension.py` run FB70 with `abundance_positivity_policy=trace_boundary` and chained restart handoff, summarize clean extension versus first-failure bracket, and fail closed on opened public/QKE/full-BBN-readiness boundaries, missing per-row telemetry, or conservation residuals above the configured limit.  A real finite-difference CPU-JAX diagnostic passed with `artifact_payload_sha256=49b2e9e858ffb87fece72c0ea2a031ed174eae9a0934db2e086c74c9997ba251`, manifest file SHA256 `01be70a682384b58296b85a712ba4f05b9898fa95810804b8ba17ec8ca8507fb`, `classification=trace_boundary_extension_all_requested_spans_passed`, largest passing endpoint `5e-09`, rows passed/failed `3/0`, best `T_final_MeV=0.7999999960714048`, conservation max `8.746901892447222e-18`, conservation limit `1e-16`, complete conservation/solver/stiffness rows `3/3/3`, step/attempt totals `20/20`, rejected steps `0`, `error_norm_max=0.0006360385926131681`, source evaluations `1166`, and stage source evaluations `140`.  FB88 remains hot-endpoint private span-extension evidence: it does not claim full-BBN completion below `0.01 MeV`, public dispatch, production SMC validation, QKE, or publication-ready all-freedom support. |
| FB-89 | Continuous AP65 trace-boundary multiplicative span growth | FB-88 | Turn the clean FB88 endpoint into a private geometric span-growth scout that keeps the same trace-boundary, conservation, stiffness, solver-effort, no-public, no-production, and no-QKE gates. | Landed in current workspace (stage-scoped): `augmented_continuous_ap65_trace_span_growth_fb89_v1` and `scripts/run_augmented_continuous_ap65_trace_span_growth.py` generate a monotone multiplicative ladder from the FB88 baseline, run the FB88 gate, and fail closed on nested FB88 boundary, ladder/row/telemetry mismatch, nested gate, or not-beyond-baseline failures.  A real finite-difference CPU-JAX diagnostic passed with `artifact_payload_sha256=77a9d8a0dab4ef5b140622fb26e87860877059eab9ea3acb25e0ef068b1ab057`, manifest file SHA256 `bc352e714afee26c01bfbc71298719dfd2fe91c063a27c11b03ce2427e53f9b2`, `classification=trace_span_growth_all_requested_spans_passed`, nested FB88 classification `trace_boundary_extension_all_requested_spans_passed`, largest passing endpoint `4e-08`, requested span rows `3`, best `T_final_MeV=0.7999999685712307`, conservation max `7.782547616453054e-18`, conservation limit `1e-16`, complete conservation/solver/stiffness rows `3/3/3`, step/attempt totals `40/40`, rejected steps `0`, `error_norm_max=0.0006361033936367059`, source evaluations `2326`, and stage source evaluations `280`.  FB89 remains hot-endpoint private span-growth evidence: it does not claim full-BBN completion below `0.01 MeV`, public dispatch, production SMC validation, QKE, or publication-ready all-freedom support. |

BD24 consolidation note: FB79/FB80/FB81/FB88/FB89 now share the
`augmented_continuous_ap65_span_experiment` builder/writer dispatch and the
`augmented_continuous_ap65_span_cli` parser/dry-run surface.  The new
`scripts/run_augmented_continuous_ap65_span_experiment.py --experiment ...`
entrypoint is the consolidated private span-experiment CLI; the five legacy
script names are compatibility wrappers over the same helper.  This folds
duplicated parser, JSON-safe serialization, artifact-hash writing, and
no-public/no-production/no-QKE claim-boundary plumbing without adding public
dispatch, production SMC validation, QKE, terminal abundance repair, or a new
standalone readiness gate.

BD11 WBS update: FB-69/FB-70 now expose opt-in
`stage_collision_payload_policy=step_base_reuse`, which reuses the host-step
base collision payload for unaccepted Rodas stage collision terms while still
evaluating the RHS at each stage state.  FB69 and FB70 record dynamic payload
build counts separately from stage payload reuse counts and mark this as a
private performance approximation.  A real CPU-JAX two-window non-LRS+collision
smoke with trace-boundary abundances, `frozen_source_jax`, chained handoff, and
a 60 s row budget passed both hot-endpoint rows with terminal
`T_gamma=0.18315223284690685 MeV`, `selected_dynamic_collision_payload_builds_total=237`,
and `selected_stage_collision_payload_reuse_total=3071`.  This removes the
immediate second-window row-budget failure but does not claim full current-state
stage-payload evidence, public dispatch, production SMC validation, QKE, or
full-BBN endpoint support below `0.01 MeV`.

| PR | Status | Purpose | Exit gate |
|---|---|---|---|
| AP0 | stage-recorded (historical; not current capability) | Create this SDD/WBS ledger and add the first API scaffold. | Document exists, roadmap index links it, API tests pass. |
| AP1 | stage-recorded (historical; not current capability) | Commit derivation notes and literature provenance for 1+3 PSTF, stress projections, weak bridge, and collision moments. | `docs/audit/v4_derivations/typeI_augmented_pstf_noqke.md` records the local and external equation provenance. |
| AP2 | stage-recorded (historical; not current capability) | Angular decomposition API for LRS/non-LRS mode ladders and convergence specs. | LRS/non-LRS mode tests pass; `MultipoleSpec` remains locked to legacy Type I `ell_max=2`. |
| AP3 | stage-recorded (historical; not current capability) | Augmented distribution reconstruction and `df -> dA` projection core. | Positivity, FD equilibrium, Pauli-floor, species-axis, and projection round-trip tests pass. |
| AP4 | partial | SciPy collisionless coevolution with augmented PSTF state.  LRS collisionless `solve_ivp` reference shell, live-distribution stress primitives, a source-only non-LRS S2 quadrupole projection gate, a source-only non-LRS plus/minus coevolution solve shell, a source-only non-LRS fixed-thermo weak/network solve shell, a source-only non-LRS 3T thermo/Hubble solve shell, an opt-in fixed-`mu_e` and charge-neutrality finite-mass e-/e+ EOS feedback path in the LRS, non-LRS source-only, and non-LRS nonlinear 3T Hubble/RHS shells with recorded `mu_e` histories, evolved LRS/source-only non-LRS/nonlinear non-LRS charge-neutral electron charge-asymmetry density states and histories, network-derivative electron charge-asymmetry energy feedback, fast finite-mass charge-susceptibility `mu_e` solves, and opt-in exact_finite_mu_scalar QED pressure/energy corrections through the scalar 3T thermo entrypoints plus LRS/source-only non-LRS/nonlinear non-LRS 3T Hubble/RHS solve shells with recorded QED-model metadata, an opt-in non-LRS collision-moment thermo feedback hook, optional projected angular-collision `dA_modes` hierarchy feedback in the LRS, source-only non-LRS, and nonlinear non-LRS 3T RHS shells, a non-LRS S2 angular collision thermo-source factory, deterministic non-LRS collision-feedback artifact/candidate/source-policy surfaces, an AP42 non-LRS `pstf_radial` source-variant route for the AP6 descriptor-driven radial moment provider on the S2 basis, an opt-in angular collision-feedback 3T solve wrapper, an opt-in direct non-LRS `pstf_radial` collision-feedback 3T wrapper and JSON/CLI artifact, artifact routing through the angular wrapper, a direct wrapper artifact runner, a direct wrapper candidate gate, a direct wrapper outcome policy, a direct solver matrix, a source-policy promotion study, a direct-wrapper convergence artifact, a collision-source budget closure artifact, a candidate evidence-bundle artifact, a physical sanity-matrix artifact, a non-LRS S2 nonlinear transport logit-RHS operator, a nonlinear transport solve shell without weak/network/collision feedback, a nonlinear non-LRS 3T weak/network candidate solve with explicit source-only/nonlinear routing, opt-in nonlinear angular and `pstf_radial` collision-feedback 3T wrappers, an AP4/AP65 combined angular+`pstf_radial` full-span candidate gate with JSON artifact/CLI output over explicit span ladders, a frozen-source Radau physical-preview preset with stable routine short-span evidence plus isolated `N_span=1e-3` diagnostic evidence, a `piecewise_frozen` nonuniform source-refresh/state-handoff gate through `N_span=1e-3` plus charge-neutral `N_span=1e-4` evolved charge-asymmetry handoff evidence, and AP6 radial number/pair-energy closure observables, an AP66 publication-candidate convergence matrix, an AP67 known-limit validation atlas, an AP68 guarded inference adapter, an AP69 augmented SMC likelihood schema, an AP70 smoke tempered-SMC runner, AP71 SMC runtime/cache controls, AP72 synthetic SMC validation, AP73 figure-ready publication artifact tables, AP74 diagnostic publication plots, AP75 diagnostic reproducibility-bundle packaging, AP76/AP79 readiness audit with a `not_promoted` diagnostic decision, AP77 coupled weak-rate smoke gate, AP78 same-CL3 AP66 weak-rate matrix control hardening, AP79 AP77-gate readiness linkage, AP80 profile-level coupled weak-rate convergence diagnostics, AP81 shared six-monomial Pauli collision statistical-factor plus staged pairwise diagonal `nu-nu` source-bridge wiring, and AP6 local PSTF six-monomial angular contraction, universal geometric kernel, channel `K` assembly, radial `p4` contraction, radial channel-grid tables, UR physical process descriptors, finite-mass HM elastic `nu-e` descriptors, finite-mass HM pair-annihilation descriptors, plus a default supported-species `{nue,nuebar,nux}` descriptor catalog whose electromagnetic entries use finite-mass HM terms and whose default `nu-nu` entries are all nine ordered pairs including identical-bank self-scattering with Fierz factor 2, same-bank number/energy-neutral projection and off-diagonal number-neutral projection plus unordered-pair energy-neutral closure, descriptor-to-mode-label mapping, fixed, dynamic ultra-relativistic FD, dynamic finite-mass FD electron/positron bath support with zero-chemical-potential default, explicit fixed-`mu_e` route-level `e_minus`/`e_plus` bath splitting, direct fixed-`mu_e` and charge-neutrality radial source artifacts with concrete moment deltas, and descriptor-label-aware total-energy radial grids with neutrino `p=q` and electron/positron `p=sqrt(max(q^2-(m_e/T_nu_e0)^2,0))` in the radial moment provider, descriptor-driven radial source evaluation, radial source number/energy moment extraction, live augmented-state radial moment provider, default-catalog LRS and non-LRS `pstf_radial` artifact routing with 18 radial moment sources plus all-nine diagonal `nu-nu` number- and pair-energy-projection diagnostics, an opt-in radial-moment AP18 thermo-source bridge, live radial bridge acceptance at the AP18 source-evaluator boundary, an explicit LRS `pstf_radial` collision-feedback artifact/candidate source variant with frozen default and budgeted live-RHS execution, an LRS tiny-span budgeted live-RHS radial artifact, and an LRS live-vs-frozen radial source-policy comparison artifact are landed; default collision-sourced full-BBN driver, anisotropic/tensor QED response and promotion-grade exact-scalar-QED full-span coupled-solver validation, promoted full-span live-RHS PSTF collision-kernel coupling beyond this no-QKE HM catalog, and public-production promotion gates remain planned. | FLRW fixed-point, LRS shear-to-distribution smoke gates, the non-LRS `Sigma_-=0` source reduction gate, live `Pi_-` feedback gate, short source-only non-LRS solve gate, fixed-thermo non-LRS weak/network gate, dynamic-H non-LRS 3T gate, explicit non-LRS collision-source callback gate, collision kinetic `dA_modes` RHS gates for LRS/source-only non-LRS/nonlinear non-LRS 3T shells, non-LRS angular source-factory gate, non-LRS collision-feedback artifact gate, non-LRS `pstf_radial` collision-feedback artifact smoke gate, direct and nonlinear non-LRS `pstf_radial` wrapper smoke gates, non-LRS collision-feedback candidate gate, live-vs-frozen source-policy smoke artifact, source-policy candidate gate, direct angular collision-feedback 3T wrapper smoke, AP42 artifact wrapper-routing gates, direct wrapper artifact JSON runner gates, direct-wrapper candidate-gate smoke, direct wrapper outcome classification gates, direct solver-matrix gates, source-policy promotion-study gates, direct-wrapper convergence smoke gates, source-budget closure gates, AP56 evidence-bundle smoke gates, AP57 physical sanity gates, AP62 nonlinear S2 RHS reduction gates, AP63 nonlinear solve-shell gates, AP64 nonlinear 3T weak/network routing gates, AP65 nonlinear collision-source wiring gates, AP4/AP65 combined full-span candidate gate artifact/script, physical-preview, piecewise-frozen source-refresh, and radial closure gates, AP66 publication matrix gates, AP67 validation-atlas gates, AP68 guarded forward-model adapter gates, AP69 likelihood-schema gates, AP70 smoke tempered-SMC gates, AP71 runtime/cache gates, AP72 synthetic-validation gates, AP73 publication-artifact schema gates, AP74 plotting gates, AP75 bundle gates, AP76/AP79 readiness-audit gates, AP77 coupled weak-rate gates, AP78 same-CL3 AP66 weak-rate matrix control gates, AP80 weak-rate convergence diagnostics, AP81 deterministic/JAX polynomial-factor numeric gates, local PSTF six-monomial projection, universal-geometric, channel-kernel, radial `p4` contraction, radial channel-grid, supported-species finite-mass electromagnetic process-catalog, default-catalog radial provider, dynamic finite-mass/signed-chemical-potential FD radial electron bath, fixed-`mu_e`/charge-neutrality source artifacts, fixed-`mu_e` and charge-neutrality LRS/non-LRS 3T EOS feedback with evolved LRS/source-only non-LRS/nonlinear non-LRS charge-neutral electron charge-asymmetry states, network-derivative electron charge-asymmetry feedback, fast finite-mass charge-susceptibility `mu_e` solves, finite-mu-scaled isotropic QED, and exact_finite_mu_scalar thermo-entrypoint gates and exact_finite_mu_scalar 3T solve-shell routing gates, descriptor-aware radial kinematics, process-radial-source, radial source moment, live radial moment provider, radial moment thermo-source, live radial AP18 source-evaluator, LRS/non-LRS `pstf_radial` artifact/candidate gates including budgeted LRS live-RHS execution, LRS budgeted live-RHS radial smoke artifact gates, LRS radial source-policy comparison gates, AP19/AP33/AP35/AP41 pairwise number/energy source-routing gates, and AP6 all-nine diagonal `nu-nu` radial number- and pair-energy-projection gates pass; collision-coupled full-BBN gates remain planned. |
| AP5 | stage-recorded (historical; not current capability) | `ell_max`, angular-grid, and `q`-grid convergence harness.  Generic report builder plus LRS collisionless, 3T weak/network, deterministic collision-feedback, source-only non-LRS, and non-LRS collision-feedback convergence runners are landed for the staged augmented-PSTF surfaces.  Promotion-tolerance full physical collision/BBN convergence remains outside this AP. | Observable deltas are recorded for each ladder point; `ell_max` tail/source summaries are recorded for multipole ladders; 3T thermo/H/network/source observables are recorded for collision-feedback ladders; source-only non-LRS reports record coefficient-source RMS values and expected-mode residuals; non-LRS collision-feedback reports record q, `N_mu`, and `N_phi` ladder deltas. |
| AP6 | stage-recorded (historical; not current capability) | Deterministic full collision reference.  Fixed collision quadrature, detailed-balance statistical-factor primitives, monopole fixed-quadrature `nu-e` plus pair-process reference kernels, explicit-rate pointwise diagonal no-QKE `nu-nu` redistribution diagnostics, an AP81 pairwise diagonal no-QKE `nu-nu` 2-to-2 reference, an augmented-distribution `nu-e` monopole projection bridge with elastic-scattering number closure, live angular-node electromagnetic `nu-e` scattering plus pair-process projection bridges, a live angular-node pairwise diagonal no-QKE `nu-nu` projection bridge, a generic deterministic nodal source-to-augmented projection bridge, a local PSTF six-monomial angular contraction table for the `K34`, `K12`, `K123`, `K124`, `K134`, and `K234` scalar occupation products, a universal `G0`/`G_mu`/`G_mumu` geometric kernel table for supplied deterministic momentum-delta angular weights, channel `K` assembly from HM-style `Pi_ij`/`Pi_ij Pi_kl` descriptors, a radial-grid `p2,p3` contraction with linear `p4 = E1 + E2 - E3` interpolation, radial channel kernel-grid assembly with invariant prefactors, a physical UR HM process descriptor catalog, finite-mass HM elastic `nu-e` descriptors, finite-mass HM pair-annihilation descriptors with the required `m_e^4` term, and a default supported-species `{nue,nuebar,nux}` descriptor catalog whose electromagnetic entries use finite-mass HM terms while all nine ordered pairwise diagonal no-QKE `nu-nu` channels are supported by default, including identical-bank self-scattering descriptors with Fierz factor `2`, same-bank number/energy-neutral projection, and off-diagonal number-neutral projection plus unordered-pair energy-neutral closure before thermo/hierarchy feedback.  Descriptor-to-mode-label mapping, fixed, dynamic ultra-relativistic FD, dynamic finite-mass FD electron/positron bath mode support with zero-chemical-potential default, signed-`mu_e` `nu-e` scattering and pair-process Pauli blocking in the staged electromagnetic source bridge with AP18/AP40 collision callback forwarding, explicit fixed-`mu_e` route-level `e_minus`/`e_plus` bath splitting, direct fixed-`mu_e` and charge-neutrality radial source artifacts with concrete moment deltas and finite-mass signed-`mu_e` e-/e+ energy/pressure diagnostics, corrected single-charge number-density normalization, opt-in fixed-`mu_e` and charge-neutrality finite-mass e-/e+ EOS feedback through the LRS, non-LRS source-only, and non-LRS nonlinear 3T Hubble/RHS shells with recorded `mu_e` histories, evolved charge-asymmetry state support, network-derivative electron charge-asymmetry energy feedback, fast finite-mass charge-susceptibility `mu_e` solves, and opt-in exact_finite_mu_scalar QED pressure/energy corrections through scalar 3T solve-shell Hubble/RHS routing, descriptor-label-aware total-energy radial grids with neutrino `p=q` and electron/positron `p=sqrt(max(q^2-(m_e/T_nu_e0)^2,0))`, static radial-grid precomputation, fast internal channel-grid assembly, precomputed radial quadrature weights, optional NPZ pretabulated radial-grid reuse, scalar contraction hot-path reuse of validated radial grids, mode-thresholded process-batched radial contraction within an explicit memory budget, a descriptor-driven radial collision source evaluator returning concrete `C_modes` values, raw-quadrature number/energy moment extraction from those source modes, a live augmented-state radial moment provider, default-catalog LRS and non-LRS `pstf_radial` artifact routing with 18 radial moment sources plus all-nine diagonal `nu-nu` number- and pair-energy-projection diagnostics, an opt-in AP18 3T thermo-source bridge that maps radial moments into `dQ_nue_pair_N`/`dQ_nux_bank_N`, an opt-in `standard_3t_plasma` radial energy-normalization mode that applies number-neutral monopole corrections so electromagnetic radial moments close to the canonical 3T plasma-transfer table per e-fold, LRS/non-LRS AP18 source-evaluator acceptance for that live radial bridge, LRS collision-feedback artifact/candidate/full-span routing via `source_variant="pstf_radial"` with frozen default and budgeted `live_rhs` execution under `max_pstf_radial_source_evaluations` including charge-neutrality `mu_e` payload diagnostics plus candidate/full-span budget observables, non-LRS AP42 collision-feedback artifact routing via `source_variant="pstf_radial"` on the S2 basis with concrete finite radial source moments, an opt-in direct non-LRS `pstf_radial` collision-feedback 3T wrapper plus JSON/CLI artifact with budgeted live-RHS diagnostics and concrete finite source moments, an opt-in nonlinear non-LRS `pstf_radial` collision-feedback 3T wrapper with concrete finite source moments, an LRS-only budgeted tiny-span live-RHS radial artifact/writer, and an LRS live-vs-frozen radial source-policy comparison artifact/writer are landed.  Public/default runtime coupling, anisotropic/tensor QED response, promotion-grade exact-scalar-QED full-span coupled-solver validation, promotion-grade full-span live-RHS PSTF collision-kernel coupling beyond this no-QKE HM catalog, and promotion gates remain planned outside this AP. | FD detailed-balance gates and finite residual diagnostics pass for landed kernels; elastic `nu-e` scattering number residuals close in the augmented source projection; signed-`mu_e` `nu-e` scattering and pair-process Pauli blocking preserve FD detailed balance and change distorted source terms; AP18/AP40 collision callbacks receive current electron chemical potential and mode; diagonal `nu-nu` per-bank number/energy moments and weighted moment residuals are locked; all nine default radial diagonal `nu-nu` rows are projected for particle-number conservation before thermo/hierarchy feedback, with identical-bank rows number/energy-neutral and six off-diagonal rows number-neutral plus unordered-pair energy-neutral while preserving relative raw species energy-transfer differences; monopole, angular electromagnetic, angular diagonal pairwise `nu-nu`, nodal-source bridge, local six-monomial PSTF projection, universal-geometric, channel-kernel descriptor, radial `p4` contraction, radial channel-grid, supported-species finite-mass electromagnetic process-catalog, default-catalog radial provider, dynamic finite-mass/signed-chemical-potential FD radial electron bath, fixed-`mu_e`/charge-neutrality source artifacts with finite-mass signed-`mu_e` energy/pressure diagnostics, fixed-`mu_e` and charge-neutrality LRS/non-LRS 3T EOS feedback with network-derivative electron charge-asymmetry feedback, fast finite-mass charge-susceptibility `mu_e` solves, finite-mu-scaled isotropic QED, exact_finite_mu_scalar thermo-entrypoint gates and exact_finite_mu_scalar 3T solve-shell routing gates, descriptor-aware radial kinematics, process-radial-source, radial moment, live radial moment provider, radial moment thermo-source, radial standard-3T electromagnetic energy-closure, radial-grid fast assembly, NPZ pretabulation, scalar contraction hot-path, mode-thresholded batched-contraction gates, all-nine diagonal `nu-nu` radial number- and pair-energy-projection gates, LRS/non-LRS live radial AP18 source-evaluator gates, LRS and non-LRS `pstf_radial` artifact gates including budgeted LRS live-RHS execution and charge-neutrality `mu_e` payload diagnostics, direct non-LRS `pstf_radial` wrapper/artifact gates with budgeted live-RHS diagnostics, nonlinear non-LRS `pstf_radial` wrapper gates, LRS budgeted live-RHS radial artifact/writer gates, and LRS radial source-policy comparison artifact/writer gates pass. |
| AP7 | stage-recorded (historical; not current capability) | Live weak-rate bridge from augmented distributions.  The augmented-distribution-to-monopole input adapter, explicit CL3 angular metadata wiring, bounded LRS `Sigma_+ K_2` CL3 rate multiplier, and AP58/AP59/AP60 LRS/non-LRS moment-input angular weak-rate modes are landed for the SciPy augmented weak/network RHS.  LRS keeps the bounded legacy multiplier as its default; staged non-LRS correction-level-3 configs now apply the AP60 current S2 moment-input path by default, with explicit `metadata_only` retained for control rows.  Public production dispatch remains outside this AP. | Born monopole contract, CL3 angular metadata, applied LRS `Sigma_+ K_2` multiplier, application-ready angular weak-rate mode resolution, species-aware LRS moment-input factors, default non-LRS S2 plus/minus moment-input factors, metadata-only control isolation, and coupled AP60 smoke evidence are covered by focused tests. |
| AP8 | stage-recorded (historical; not current capability) | SciPy stability envelope and candidate gate.  LRS collisionless shell stability-envelope reports over `Sigma_+`, `ell_max`, `N_q`, and `N_mu` are landed; source-only non-LRS `Sigma_-` projection envelope is landed; bounded LRS CL3 weak-rate, deterministic collision-feedback, staged 3T span-ladder, non-LRS collision-feedback, and non-LRS source-policy candidate gates are landed.  This closes the AP8 staged candidate-gate deliverable without promoting a full physical angular collision kernel or public full-BBN runtime. | LRS bounded shell tests check trajectory `Sigma_+` maxima plus final `Pi_+`/mode amplitudes; non-LRS source-only envelope checks plus/minus coefficient-source limits and expected-mode residuals; weak-rate, collision-feedback, 3T span-ladder, non-LRS collision-feedback, and source-policy candidate gates check bounded multiplier/source-variant deltas, thermo/H/network finiteness, plus/minus source moments/stress limits, live-vs-frozen policy deltas, solve effort, and limit-failure reporting without promoting a runtime capability. |
| AP9 | stage-recorded (historical; not current capability) | Optional deterministic QMC control-variate accelerator.  Replay-stable fixed Sobol samples, scalar control-variate convergence reports, augmented nodal `df/dN` source projection reports, and collision source-moment vector reports are landed for the staged no-QKE validation surfaces.  Full collision-kernel evaluation remains planned under AP6. | Same config is replay-stable; sample ladders converge to deterministic references for scalar, nodal projection, and `dQ_nue_pair_N`/`dQ_nux_bank_N` moment reports, including fixed converged sample counts and tail-convergence checks. |
| AP10 | stage-recorded (historical; not current capability) | JAX/XLA core parity port.  A CPU-JAX augmented distribution core mirrors the SciPy reconstruction, nodal projection, Gram solve, `df -> dA` projection, and generic nodal collision-source projection bridge on fixed tiny grids; solver integration is tracked separately in AP11. | Focused CPU-JAX parity tests pass for reconstruction, Gram projection, RHS projection, `jax.jit` on the core function, and LRS/non-LRS nodal collision-source projection parity. |
| AP11 | stage-recorded (historical; not current capability) | JAX solver integration.  A static fixed-grid CPU-JAX LRS collisionless RHS factory, static fixed-source projected nodal-source RHS factory, and existing JAX Rodas5P pure-core solve wrappers are landed for parity staging; physical collision coupling and public backend dispatch remain planned. | Focused CPU-JAX tests match the SciPy LRS RHS, reject dynamic traced grids/sources, track a short SciPy reference solve through the in-tree Rodas5P core, and track a bounded projected-source Rodas5P solve before any GPU/XLA promotion. |
| AP12 | stage-recorded (historical; not current capability) | Capability registry and generated-doc visibility.  The staged augmented-PSTF surface is registered in `CAPABILITY_BY_KEY` and `FEATURE_BY_KEY` only; public forward-solver dispatch remains deferred. | Registry-driven docs are regenerated, the capability is labelled as a diagnostic substrate, and tests lock that no `CAPABILITY_BY_BACKEND`/`canonical_forward_solver` route exists. |
| AP13 | stage-recorded (historical; not current capability) | SciPy live weak/network RHS bridge.  The current augmented LRS modes are reconstructed on the angular grid, `nu_e`/`anti-nu_e` monopoles are extracted, live weak rates are computed, and the PRIMAT standard network derivative is evaluated in the same RHS block; later solve-shell integration is tracked in AP14/AP15. | Focused tests lock FD monopole extraction, current-monopole rate sensitivity, PRIMAT network RHS parity, CL3 metadata threading without rate application, and a combined collisionless-transport plus weak/network RHS block. |
| AP14 | stage-recorded (historical; not current capability) | SciPy LRS collisionless plus weak/network solve shell.  `Sigma_+`, augmented LRS modes, and PRIMAT abundances are packed into one `solve_ivp` `d/dN` state with externally supplied fixed thermo/H values; LSODA is supported through `method=`, while the smoke default stays aligned with the existing SciPy shell. | Focused tests lock FLRW transport preservation, shear-to-mode coupling, finite abundance evolution, custom q-grid threading, and input/config rejection. |
| AP15 | stage-recorded (historical; not current capability) | SciPy 3T thermo/Hubble staging shell.  The AP14 solve loop has a separate 3T variant that packs `T_gamma`, `T_nu_e`, and `T_nu_x` into the same state, recomputes `H(T_gamma,T_nu_e,T_nu_x,Sigma^2)` every RHS call, and feeds the dynamic Hubble rate into the weak/network block. | Focused tests lock dynamic temperature evolution, dynamic Hubble history, shear-to-mode coupling, LSODA override support, and invalid-temperature/config rejection. |
| AP16 | stage-recorded (historical; not current capability) | 3T shell convergence runners.  The AP15 3T solve can be swept over `ell_max`, `N_q`, and `N_mu` using the existing convergence-report contracts, including tail norms for the `ell_max` ladder. | Focused tests lock 3T thermo/network observables, dynamic-H summaries, tail norms, and q/angular resolution labels. |
| AP17 | stage-recorded (historical; not current capability) | Deterministic 3T convergence artifact runner.  The AP16 `ell_max`, `N_q`, and `N_mu` 3T convergence ladders can be emitted as a JSON-ready diagnostic artifact with a smoke preset and an optional extended preset including `ell_max = 2,4,6,8`. | Focused tests lock the artifact contract, serialized ladder labels/values, solver metadata, thermo/network observable names, and explicit no-QKE/no-public-dispatch limitations. |
| AP18 | stage-recorded (historical; not current capability) | Collision-moment thermo feedback hook for the SciPy 3T shell.  The AP15 solve can switch from the standard 3T table RHS to direct `dQ_nue_pair_N`/`dQ_nux_bank_N` collision moments when an explicit source callback is supplied.  The AP6 radial-moment bridge can now supply that callback shape from concrete PSTF radial source moments, including moments computed from live `A_modes` through a configured radial-process provider, and the AP18 source evaluator accepts that live bridge with concrete nonzero radial-source diagnostics.  This is plumbing for deterministic collision references, not a built-in full angular collision kernel. | Focused tests lock the opt-in source contract, source-driven `T_nu_e`/`T_nu_x` split, default table-RHS metadata, rejection of invalid source callbacks, radial-moment-to-3T source bookkeeping, live augmented-state radial moment provider wiring, and live radial bridge acceptance by the AP18 source evaluator. |
| AP19 | stage-recorded (historical; not current capability) | Deterministic pairwise diagonal no-QKE `nu-nu` thermo source factory.  Current augmented LRS monopoles feed the AP18 hook through the AP81 fixed-quadrature pairwise diagonal `nu-nu` 2-to-2 reference using the six-monomial Pauli factor, with explicit per-bank number closure and effective-`nu_x` weighted-energy closure projection for `dQ_nue_pair_N`/`dQ_nux_bank_N` moments.  The older fixed-point redistribution helper remains legacy comparison plumbing.  This remains a monopole diagonal source, not the full angular collision kernel. | Focused tests lock source sign, pairwise reference metadata, six-monomial diagnostics, number residual closure, raw/closed weighted-energy residual diagnostics, and use of the factory inside the 3T solve callback path. |
| AP20 | stage-recorded (historical; not current capability) | Deterministic monopole `nu-e` plus pair-process thermo source factory.  Current augmented LRS monopoles can feed the AP18 hook through the existing fixed-quadrature electron-scattering and pair references, producing `dQ_nue_pair_N`/`dQ_nux_bank_N` moments against the electromagnetic bath.  This remains angle-independent and does not implement the full angular collision kernel. | Focused tests lock FD quietness, heating sign for under-populated neutrino monopoles, total neutrino energy-gain diagnostics, and detailed-balance residual metadata. |
| AP21 | stage-recorded (historical; not current capability) | Combined collision-moment thermo source factory.  The AP19 pairwise diagonal `nu-nu` and AP20 electromagnetic-bath monopole sources can be summed into one explicit callback for the AP18 3T hook, with component diagnostics preserved.  This remains opt-in and still does not implement the full angular collision kernel. | Focused tests lock exact component-moment summation, component diagnostic propagation, and acceptance by the 3T solve callback path with an extra unused species bank. |
| AP22 | stage-recorded (historical; not current capability) | Deterministic collision-feedback source-variant artifact runner.  A smoke-scale JSON report compares the standard 3T table RHS against the opt-in AP19 `nu-nu`, AP20 electromagnetic-bath, and AP21 combined collision-moment source variants.  This is a diagnostic report surface only, not a promoted/default collision-coupled solve. | Focused tests lock the JSON artifact contract, variant source contracts, source diagnostics, known limitations, and JSON serializability; the companion script emits the report artifact. |
| AP23 | stage-recorded (historical; not current capability) | Standard-relative collision-feedback artifact deltas.  The AP22 source-variant artifact records per-observable deltas against the standard 3T table-RHS baseline whenever that baseline is included.  This is report metadata only and does not change solver behavior. | Focused tests lock the `delta_reference` field and exact standard-relative observable delta arithmetic for the combined source variant. |
| AP24 | stage-recorded (historical; not current capability) | Stage-scoped WBS status normalization.  AP10-AP23 are now marked as landed where their named deliverables and exit gates are implemented, while AP4-AP9 remain partial because their row text still names unresolved programme blockers at that stage. | Focused tests parse this ledger and lock the split between stage-scoped landed AP rows and remaining partial programme blockers; the note below preserves the distinction between scoped landing and overall feature promotion. |
| AP25 | stage-recorded (historical; not current capability) | Stage-scoped AP7 promotion from metadata-only to an applied bounded LRS CL3 weak-rate correction.  The augmented combined RHS now passes the current `Sigma_+` into the existing `sigma_plus_K2_correction_factor(...)` path and records the applied multiplier in the weak-network result metadata.  Full anisotropic weak-rate integration remains outside this AP. | Focused tests lock exact multiplier propagation into `lambda_np`/`lambda_pn`, metadata `rate_application`, and the updated WBS ledger at that stage: AP4-AP6/AP8-AP9 remain partial while AP7/AP10-AP25 are stage-scoped landed. |
| AP26 | stage-recorded (historical; not current capability) | Weak-rate candidate sub-gate for AP8.  A smoke-scale stability report now sweeps `Sigma_+` for the bounded LRS CL3 weak-rate multiplier, records `lambda_np`/`lambda_pn` deltas against the `Sigma_+=0` reference, and enforces explicit factor/rate limits.  This does not add collision-coupled or full BBN candidate promotion. | Focused tests lock the report contract, rate-application metadata, accepted small-shear ladder, limit-failure reporting, invalid-spec rejection, and the WBS ledger with AP8 still partial at that stage. |
| AP27 | stage-recorded (historical; not current capability) | Collision-feedback candidate sub-gate for AP8.  A smoke-scale gate now wraps the AP22/AP23/AP36 source-variant artifact, checks selected standard-relative thermo/network deltas and collision source moments, and forwards the AP36 frozen-initial-state/live-RHS source-update policy plus solver controls into the LRS angular wrapper path.  This is not a full physical angular collision-kernel or full BBN candidate gate. | Focused tests lock accepted standard/combined/angular variants, real AP35/AP36 angular-wrapper smoke output through the candidate gate, source-update/solver-control propagation, limit-failure reporting, invalid-spec rejection, public case validation, updated artifact limitations, and the WBS ledger with AP8 still partial at that stage. |
| AP28 | stage-recorded (historical; not current capability) | Staged 3T span-ladder candidate gate for AP8.  A new stability report repeatedly runs the AP22/AP23/AP36 collision-feedback artifact path over an explicit `N_span` ladder, including smoke defaults and optional longer spans, records thermo/H/network/source observables for each source variant, and forwards the AP36 LRS angular source-update policy into span cases.  This closes AP8 as a staged candidate-gate deliverable while preserving the full angular-kernel, full anisotropic weak-rate, non-LRS nonlinear, production-span convergence, and public-dispatch blockers. | Focused tests lock span-ladder validation, accepted standard/combined/angular variants, source-update propagation, source-moment and thermo/H/network limit checks, invalid-spec rejection, public case validation, and the WBS ledger with AP8 stage-scoped landed; AP4 through AP6 and AP9 were still partial at that stage. |
| AP29 | stage-recorded (historical; not current capability) | Deterministic collision-feedback 3T convergence runners for AP5.  The AP22/AP23/AP36 source-variant artifact can now be swept over `ell_max`, `N_q`, and `N_mu` ladders, extracting standard-relative thermo/network deltas, source moments, solve effort, and the AP36 LRS angular source-update policy for a selected source variant.  This closes AP5 as a staged convergence-harness deliverable without claiming promotion-tolerance full physical collision/BBN convergence. | Focused tests lock `N_q_3T_collision_feedback`, `N_mu_3T_collision_feedback`, and `ell_max` collision-feedback report contracts, selected source-variant observables, AP36 source-update propagation, real LRS angular-wrapper q-convergence smoke output, bad-source rejection, and the WBS ledger with AP5/AP8 stage-scoped landed while AP4/AP6/AP9 remain partial. |
| AP30 | stage-recorded (historical; not current capability) | QMC collision source-moment control-variate reports for AP9.  The deterministic QMC validation utilities now estimate named collision source moments such as `dQ_nue_pair_N` and `dQ_nux_bank_N` with replay-stable Sobol samples, explicit control integrals, reference-moment errors, adjacent deltas, and tail-convergence selection.  This closes AP9 as a staged accelerator/report deliverable; AP6 angular collision-reference closure is handled by AP31-AP34. | Focused tests lock multi-moment convergence to deterministic references, replay identity, missing-key rejection, report validation, and the WBS ledger with AP5/AP8/AP9 stage-scoped landed at that stage; AP6 is stage-scoped landed by AP34. |
| AP31 | stage-recorded (historical; not current capability) | LRS angular-node `nu-e` scattering projection bridge for AP6.  The deterministic `nu-e` scattering reference can now be evaluated separately at each LRS angular node from the live reconstructed augmented distribution `f(q,mu)`, projected onto a number-conserving elastic-scattering source that preserves the energy moment, and projected back into PSTF modes.  This is the first landed live angular collision-kernel evaluation path, but angular pair-process, angular `nu-nu`, non-LRS angular kernels, and public runtime coupling remain planned. | Focused tests lock FD quietness, live quadrupole source projection into the `A2` mode, elastic-scattering number closure, input validation, public result validation, and unchanged existing monopole/nunu/electron-pair/nodal bridge behavior. |
| AP32 | stage-recorded (historical; not current capability) | LRS angular-node electromagnetic pair-process projection bridge for AP6.  The deterministic `nu-e` scattering and pair-process references can now be evaluated at each LRS angular node from the live reconstructed `f(q,mu)` for `nu_e`, `anti-nu_e`, and the effective `nu_x` bank, with number-closed elastic-scattering component diagnostics, pair-process diagnostics, and projection back into PSTF modes.  Angular `nu-nu`, non-LRS angular kernels, and public runtime coupling remain planned. | Focused tests lock FD quietness, component energy-moment summation, elastic-scattering number closure, live pair-process quadrupole projection for all three supported species banks, required-species validation, and unchanged existing bridge behavior. |
| AP33 | stage-recorded (historical; not current capability) | LRS angular-node diagonal no-QKE `nu-nu` projection bridge for AP6.  The AP81 pairwise diagonal `nu-nu` 2-to-2 reference is evaluated separately at each LRS angular node from live reconstructed `f(q,mu)` for `nu_e`, `anti-nu_e`, and the effective `nu_x` bank, with six-monomial Pauli diagnostics, per-bank number closure, weighted energy-closure projection, and projection back into PSTF modes.  Non-LRS angular kernels and public runtime coupling remain planned. | Focused tests lock FD quietness at deterministic-reference precision, live quadrupole redistribution projection, per-bank number closure, weighted bank-energy conservation, pairwise/statistical diagnostics, required-species validation, and unchanged existing bridge behavior. |
| AP34 | stage-recorded (historical; not current capability) | Non-LRS S2 angular collision-reference closure for AP6.  The AP31-AP33 angular bridges now carry generic angular-node v2 closure contracts and are tested on the staged non-LRS S2 `{monopole, W_plus, W_minus}` basis, including live minus-mode projection for number-closed `nu-e`, electromagnetic pair-process, and pairwise diagonal no-QKE `nu-nu` sources.  This closes AP6 as a deterministic collision-reference deliverable without promoting runtime coupling. | Focused tests lock non-LRS S2 minus-mode projection through all three angular bridges, generic v2 closure contracts, number/energy closure diagnostics on supported components, and unchanged LRS bridge behavior. |
| AP35 | stage-recorded (historical; not current capability) | Angular collision thermo-source callback for AP18.  The AP32 electromagnetic angular bridge and AP33 pairwise diagonal no-QKE `nu-nu` angular bridge can now be composed into an explicit `Augmented3TCollisionThermoSource` callback, producing `dQ_nue_pair_N` and `dQ_nux_bank_N` from the live augmented angular distribution while preserving numeric component diagnostics.  A smoke-scale opt-in LRS angular-collision 3T wrapper now wires that source directly into the AP15/AP18 solve path with frozen-initial-state or live-RHS source policy.  This is opt-in coupling only, not a default/promoted collision-coupled solve. | Focused tests lock isotropic agreement with the existing combined monopole source, finite angular-source moments, pairwise six-monomial diagnostics, numeric closure diagnostics, acceptance by the AP18 3T shell source evaluator with an unused species bank, and an actual LRS angular-collision 3T wrapper smoke run. |
| AP36 | stage-recorded (historical; not current capability) | Angular collision-feedback artifact/gate variant.  The AP35 angular thermo-source callback is now available as the explicit `angular` source variant in the AP22/AP23 collision-feedback artifact path, AP27 candidate gate, AP28 span-ladder gate, and AP29 convergence runners, and the LRS artifact route now calls the AP35 direct wrapper with explicit frozen-initial-state/live-RHS source-update policy instead of rebuilding the callback locally.  Full default/promoted angular-feedback solves remain outside promotion claims. | Focused tests lock `angular` source-variant validation, artifact contract metadata, AP35 direct-wrapper routing, real LRS angular-wrapper artifact smoke output, AP35 source diagnostics in the artifact payload, candidate/span gate acceptance, and q-convergence extraction through the existing report contracts. |
| AP37 | stage-recorded (historical; not current capability) | Source-only non-LRS coevolution shell for AP4.  The AP4 non-LRS S2 quadrupole source projection now has a SciPy `solve_ivp` shell that packs `Sigma_+`, `Sigma_-`, and `{monopole, W_+, W_-}` augmented modes, reconstructs the live distribution each RHS call, and feeds `Pi_+`/`Pi_-` stress moments back into the diagonal shear equations.  Nonlinear non-LRS angular advection, q-cascade transport, collisions, weak/network coupling, and public dispatch remain outside this AP. | Focused tests lock RHS/source-projection agreement, live `Pi_-` feedback from a minus-mode distortion, short plus/minus solve evolution, and the AP4 partial boundary for unresolved nonlinear/full-BBN blockers. |
| AP38 | stage-recorded (historical; not current capability) | Source-only non-LRS weak/network shell for AP4.  The AP37 plus/minus source-only shell now has a fixed-thermo weak/network companion that reconstructs the live S2 distribution, extracts `nu_e`/`anti-nu_e` angular monopoles, records CL3 plus/minus metadata without applying a non-LRS anisotropic weak-rate correction, computes live weak rates, and feeds the PRIMAT network derivative into the same `d/dN` state.  Nonlinear non-LRS transport, collision-sourced thermodynamics, full anisotropic weak rates, and public dispatch remain outside this AP. | Focused tests lock live S2 monopole extraction, plus/minus CL3 metadata, metadata-only rate application, source-transport agreement, short fixed-thermo weak/network solve evolution, and missing `nu_e`/`anti-nu_e` species rejection. |
| AP39 | stage-recorded (historical; not current capability) | Source-only non-LRS 3T thermo/Hubble shell for AP4.  The AP38 non-LRS weak/network shell now has a dynamic 3T companion that packs `T_gamma`, `T_nu_e`, and `T_nu_x` into the same state, recomputes `H(T_gamma,T_nu_e,T_nu_x,Sigma_+^2+Sigma_-^2)` each RHS call, and feeds the standard 3T table thermo RHS alongside the source-only non-LRS transport and live weak/network blocks.  Collision-moment thermodynamics, nonlinear non-LRS transport, full anisotropic weak rates, and public dispatch remain outside this AP. | Focused tests lock dynamic temperature evolution, dynamic Hubble history with `Sigma_+^2+Sigma_-^2`, plus/minus mode evolution, finite `Pi_+`/`Pi_-`, live weak rates, and abundance normalization for the staged 3T solve. |
| AP40 | stage-recorded (historical; not current capability) | Opt-in non-LRS collision-moment thermo feedback hook for AP4.  The AP39 3T shell now accepts an explicit collision source callback returning `Augmented3TCollisionThermoSource`, passes `Sigma_-` and the S2 grid to that callback, and switches the temperature RHS to `coupled_3T_rhs_from_collision_moments(...)` when supplied.  This is hook plumbing only: no default non-LRS collision source, nonlinear transport, full anisotropic weak rates, or public dispatch is promoted. | Focused tests lock callback invocation with `Sigma_-` and S2 grid metadata, collision-source diagnostics propagation, `collision_moment_thermo_feedback_v1` metadata, bad-source rejection, and unchanged standard table-RHS behavior when no callback is supplied. |
| AP41 | stage-recorded (historical; not current capability) | Non-LRS S2 angular collision thermo-source factory for AP4.  The generic AP35 angular thermo source is now exposed through a non-LRS S2 wrapper that fixes the `{monopole, W_+, W_-}` grid, composes the AP32/AP33 angular electromagnetic and pairwise diagonal no-QKE `nu-nu` bridges, and records `Sigma_-` plus S2 diagnostics for the AP40 callback path.  It remains opt-in and is not a default collision-coupled solve. | Focused tests lock equivalence to the generic angular source on the same S2 grid, non-LRS closure metadata, S2 angle-count diagnostics, `Sigma_-` propagation, and component contract diagnostics. |
| AP42 | stage-recorded (historical; not current capability) | Deterministic non-LRS 3T collision-feedback artifact runner for AP4.  The AP41 S2 angular source can now be emitted through the AP40 non-LRS 3T callback hook as a JSON-ready smoke artifact comparing `standard` and `angular` source variants.  The smoke default freezes the initial angular collision moments for runtime stability, with explicit `live_rhs` re-evaluation available for longer experiments. | Focused tests lock artifact JSON serializability, S2 grid metadata, `Sigma_-` diagnostics, source-update policy metadata, AP41 source contract propagation, standard-relative observable deltas, writer output, and explicit no-QKE/no-public-dispatch limitations. |
| AP43 | stage-recorded (historical; not current capability) | Non-LRS 3T collision-feedback convergence runners for AP5.  The AP42 artifact observables can now be swept over `N_q`, `N_mu`, and `N_phi` ladders for the selected non-LRS source variant, extracting plus/minus stress, thermo/network, selected standard-relative deltas, source moments, and solve effort. | Focused tests lock q/S2 angular ladder labels, standard-relative source-moment observables, invalid variant and resolution rejection, and the AP5 stage-scoped convergence boundary without promoting full physical collision/BBN convergence. |
| AP44 | stage-recorded (historical; not current capability) | Non-LRS collision-feedback candidate gate for AP8.  A smoke-scale gate now wraps the AP42 artifact path, including the AP41 `angular` source variant through the AP40 3T hook and the AP6 `pstf_radial` route under an explicit source-evaluation budget, and records plus/minus shear/stress, source moments, selected standard-relative thermo/network deltas, solve effort, and radial budget observables. | Focused tests lock accepted standard/angular/`pstf_radial` variants, AP41 and radial source-contract propagation, plus/minus observable extraction, live-RHS source-evaluation budget extraction, limit-failure reporting, invalid-spec rejection, public case validation, and the AP8 stage-scoped gate boundary without promoting default collision feedback, nonlinear non-LRS transport, full anisotropic weak rates, or public dispatch. |
| AP45 | stage-recorded (historical; not current capability) | Non-LRS collision-feedback source-update policy artifact for AP4.  The AP42 artifact can now be emitted as a deterministic frozen-initial-state versus `live_rhs` comparison, using a very short smoke span by default and recording per-variant observables, standard-relative deltas, reference-policy deltas, solve effort, and AP6 `pstf_radial` live-RHS source-evaluation budget metadata when that variant is selected.  A dedicated non-LRS `pstf_radial` source-policy artifact/writer now wraps that real AP42 path and emits compact live-minus-frozen radial observables and diagnostics. | Focused tests lock source-policy validation, live-vs-frozen observable deltas, nfev reporting, `pstf_radial` budget forwarding, dedicated non-LRS radial artifact/writer output, JSON writer output, and explicit diagnostic/no-QKE/no-public-dispatch scope; actual smoke runs exercise the live RHS at short `N_span`. |
| AP46 | stage-recorded (historical; not current capability) | Non-LRS source-update policy candidate gate for AP8.  The AP45 artifact now has a pass/fail stability gate over frozen/live policies and standard/angular/`pstf_radial` variants, checking selected live-vs-frozen thermo/network deltas, collision source-moment deltas, absolute source moments, solve effort, and live radial source-evaluation budgets. | Focused tests lock accepted `live_rhs` policy cases including `pstf_radial`, source-evaluation budget observables, nfev limit-failure reporting, invalid spec rejection, public case validation, and an actual AP45-backed LSODA smoke gate without promoting live RHS to the default source policy. |
| AP47 | stage-recorded (historical; not current capability) | Opt-in non-LRS angular collision-feedback 3T solve wrapper for AP4.  `run_augmented_nonlrs_angular_collision_weak_network_3T_solve(...)` now directly wires the AP41 S2 angular source through the AP40 non-LRS 3T collision-moment hook, with explicit `live_rhs` or `frozen_initial_state` source-update policy selection. | Focused tests lock wrapper source construction, S2 grid/q quadrature threading, live versus frozen source-policy behavior, export wiring, invalid-policy rejection, and an actual LSODA smoke run through the AP41/AP40 path without public dispatch or default source promotion. |
| AP48 | stage-recorded (historical; not current capability) | Non-LRS collision-feedback artifact reuse of the AP47 wrapper for AP4.  The AP42 artifact now routes its `angular` source variant through `run_augmented_nonlrs_angular_collision_weak_network_3T_solve(...)` instead of duplicating AP41 source construction and AP40 hook wiring locally. | Focused tests lock that only the `standard` variant uses the low-level source-only 3T shell while the `angular` variant forwards `N_q`, `N_mu`, `N_phi`, source policy, solver controls, initial modes, and thermo inputs to the AP47 wrapper; source-policy artifacts inherit this routing without promoting public dispatch or default collision feedback. |
| AP49 | stage-recorded (historical; not current capability) | Direct non-LRS angular collision-feedback 3T solve artifact runner for AP4.  `build_augmented_nonlrs_angular_collision_3t_solve_artifact(...)`, its JSON writer, and `scripts/run_augmented_nonlrs_angular_collision_3t_solve_artifact.py` now emit a deterministic smoke artifact for the AP47 wrapper itself. | Focused tests lock artifact schema, wrapper argument forwarding, result/source diagnostics, JSON writer output, and CLI summary output; an actual LSODA smoke run records the AP41 source contract through the AP47 wrapper with `live_rhs` without promoting public dispatch or default collision feedback. |
| AP50 | stage-recorded (historical; not current capability) | Direct non-LRS angular collision-feedback candidate gate for AP4.  `run_augmented_nonlrs_angular_collision_direct_candidate_gate(...)` now wraps the AP49 artifact and checks source moments, plus/minus shear/stress, A-mode RMS, temperature/Hubble/network bounds, and solve effort. | Focused tests lock accepted live-RHS cases, source-moment/nfev failure reporting, spec validation, case validation, and an actual AP49-backed LSODA smoke gate without promoting public dispatch, default collision feedback, nonlinear non-LRS transport, or QKE. |
| AP51 | stage-recorded (historical; not current capability) | Direct-wrapper outcome classifier over the existing AP49/AP50 artifact and candidate-gate path.  `src/rabbit/validation/augmented_outcomes.py` defines smoke, medium, and production-span diagnostic presets, hard solver timeouts, failure classes, JSON-ready report metadata, and `scripts/run_augmented_nonlrs_angular_collision_direct_outcomes.py` for AP51 artifacts; the smoke default uses the runtime-stable frozen source policy, with `live_rhs` kept as an explicit option. | Focused tests classify success, stiffness failure, bound violation, timeout, and invalid-output cases on existing AP49/AP50 reports; a real medium-span LSODA smoke run with explicit frozen source policy emits bounded metadata without changing public dispatch, default collision feedback, nonlinear non-LRS transport, or QKE scope. |
| AP52 | stage-recorded (historical; not current capability) | Pre-budget diagnostic solver tolerance and method matrix for the direct wrapper.  `src/rabbit/validation/augmented_solver_matrix.py` compares LSODA and Radau over span/tolerance/source-policy ladders by reusing AP51 classified outcomes, records runtime, nfev, terminal observables, source moments, selected deltas, and pre-budget candidate policy metadata, and `scripts/run_augmented_direct_solver_matrix.py` emits the JSON artifact. | Focused tests lock accepted LSODA/Radau agreement, method-disagreement failures, classified timeout propagation, spec validation, JSON writer/CLI summaries, and a real frozen-source LSODA/Radau smoke matrix; every result is labelled `pre_budget_diagnostic_ap52` and no physics claim advances before AP55 source-budget closure. |
| AP53 | stage-recorded (historical; not current capability) | Source-update policy promotion study extending the existing AP45 frozen-initial-state versus `live_rhs` artifact into a higher-level AP42/direct-wrapper decision surface.  `src/rabbit/validation/augmented_source_policy.py` consumes AP42 source-policy artifacts, including the AP6 `pstf_radial` budgeted surface, and AP51 direct-wrapper outcomes across span ladders, classifies `live_rhs` AP56 eligibility, records frozen moments as diagnostic fallback only, and `scripts/run_augmented_source_policy_study.py` emits the JSON artifact. | Focused tests lock live-candidate promotion when AP42 and direct-wrapper surfaces pass, rejection when direct `live_rhs` is classified as timeout, AP42 policy-delta limit failures, AP42 `pstf_radial` budget propagation, spec validation, JSON writer/CLI summaries, and a real AP42 frozen/live smoke study; no public dispatch, default collision feedback, AP55 source-budget closure, nonlinear non-LRS transport, or QKE scope advances. |
| AP54 | stage-recorded (historical; not current capability) | Pre-budget direct-wrapper convergence ladders over `N_q`, `N_mu`, and `N_phi` using the AP51 classified direct outcome path.  `src/rabbit/validation/augmented_direct_convergence.py` reuses the existing `ResolutionConvergenceReport` contract, records terminal thermo/network/source observables plus classified outcome flags, and `scripts/run_augmented_direct_convergence.py` emits the AP54 JSON artifact. | Focused tests lock q/angular ladder observables, classified timeout rows, invalid input rejection, JSON writer/CLI summaries, and a real frozen-source q-ladder smoke run; the AP54 artifact smoke converges q, `N_mu`, and `N_phi` ladders at smoke tolerances while remaining pre-budget diagnostic pending AP55 source-moment/energy closure. |
| AP55 | stage-recorded (historical; not current capability) | Collision source energy/moment budget closure for AP41/AP47/AP6.  `src/rabbit/validation/augmented_collision_source_budget.py` and `scripts/run_augmented_collision_source_budget.py` evaluate the AP41/AP47 deterministic no-QKE angular source over FLRW quiet, LRS electron-pair heating, LRS pairwise `nu-nu` energy redistribution, LRS fixed-`mu_e` and charge-neutral AP6 `pstf_radial` process budgets, non-LRS fixed-`mu_e` and charge-neutral AP6 `pstf_radial` process budgets, non-LRS S2 LRS-limit, non-LRS minus-mode, and non-LRS quiet cases. | Focused tests prove finite source moments, component-sum closure, bounded no-QKE pairwise `nu-nu` weighted-energy and number residuals, pairwise source metadata, six-monomial diagnostics, explicit number/energy closure projection diagnostics, correct source-contract diagnostics, FLRW quietness, LRS component sign behavior, LRS/non-LRS AP6 radial source moments, finite-mass process markers, algebraic charge-neutral finite-mass e-/e+ bath diagnostics, kinetic `dA` hierarchy amplitude, non-LRS radial S2 context propagation, JSON writer/CLI summaries, and a real default AP55 artifact pass.  This closes the source-budget blocker only for the staged deterministic no-QKE moment source and does not promote public dispatch, nonlinear transport, full anisotropic weak rates, or QKE. |
| AP56 | stage-recorded (historical; not current capability) | Full-span diagnostic candidate angular collision-feedback artifact aggregator.  `src/rabbit/validation/augmented_candidate_artifact.py` and `scripts/run_augmented_collision_feedback_candidate_artifact.py` aggregate AP42/AP49/AP51/AP52/AP53/AP54/AP55 summaries into one evidence bundle for standard versus angular direct-wrapper collision-feedback comparisons while preserving the AP55 `lrs_pstf_radial_process_budget`, `lrs_pstf_radial_charge_neutrality_budget`, `nonlrs_pstf_radial_process_budget`, and `nonlrs_pstf_radial_charge_neutrality_budget` case lists, source contracts, charge-neutral e-/e+ context, S2 context markers, and selected radial observables. | Focused tests lock evidence contracts, source routing, solver/source-update policy decisions, direct convergence status, AP55 source-budget status including the LRS/non-LRS fixed-`mu_e` and charge-neutral AP6 radial budget summaries, failure propagation, JSON writer output, CLI summaries, and a real smoke artifact with `completed=true`; the smoke artifact records `diagnostic_fallback_only` source-policy status when `live_rhs` is not AP56-eligible.  No public dispatch, default collision feedback, nonlinear transport, full anisotropic weak rates, publication plot, SMC, or QKE is promoted. |
| AP57 | stage-recorded (historical; not current capability) | Collision-sourced thermo feedback physical sanity matrix over AP56/AP55 evidence.  `src/rabbit/validation/augmented_physics_gates.py` and `scripts/run_augmented_collision_thermo_sanity_matrix.py` check AP56 bundle readiness, source-policy claim boundary, terminal thermo/network/source bounds, plus/minus bounded response, FLRW quiet source limits, LRS sign/energy budget, LRS/non-LRS fixed-`mu_e` and charge-neutral AP6 radial source-budget markers and amplitudes, charge-neutral e-/e+ bath positivity, and non-LRS S2 reduction/context. | Focused tests lock pass/fail behavior for candidate-not-ready, unphysical temperature, quiet-source limit, spec validation, default artifact collection, LRS/non-LRS fixed-`mu_e` and charge-neutral AP6 radial source-budget sanity, JSON writer output, and CLI summary; a real smoke artifact passes with eleven sanity cases.  This is bounded diagnostic sanity evidence only and does not promote public dispatch, nonlinear transport, full anisotropic weak rates, publication plots, SMC, or QKE. |
| AP58 | stage-recorded (historical; not current capability) | Full angular weak-rate SDD and existing input-model upgrade.  `AugmentedWeakInputs` now carries explicit `AngularWeakRateMomentInputs` with per-q LRS plus moments or non-LRS S2 plus/minus moments, approximation/source labels, metadata linkage through `WeakAngularCorrectionMetadata`, and fail-closed angular weak-rate mode resolution.  This is input-model-only and does not apply the AP59/AP60 full anisotropic weak-rate correction. | Focused tests show the existing weak-rate input objects carry plus/minus angular data, reject missing S2 minus moments and temperature-compressed fallback sources, preserve approximation labels, thread through the non-LRS weak/network RHS, and fail closed on unsupported angular weak-rate modes. |
| AP59 | stage-recorded (historical; not current capability) | LRS live anisotropic weak-rate integration beyond the bounded `Sigma_+ K_2` multiplier.  The AP58 per-q LRS plus-moment input model now drives an opt-in `lrs_cl3_quadrupole_input` rate mode that computes species-aware `lambda_np`/`lambda_pn` correction factors from the current reconstructed distribution, while preserving the existing `legacy_sigma_plus_k2_multiplier` as the default bounded subcase. | Focused tests cover species-specific CL3 factors, opt-in LRS weak/network rate application, lambda multiplier propagation, unsupported-mode rejection, and unchanged legacy multiplier behavior. |
| AP60 | stage-recorded (historical; not current capability) | Non-LRS live anisotropic weak-rate integration.  The AP59 moment-input rate path now extends to the staged S2 plus/minus basis through `nonlrs_s2_cl3_quadrupole_input`; staged non-LRS correction-level-3 weak/network configs apply that current S2 moment-input mode by default, with explicit `metadata_only` retained for same-CL3 controls and baselines. | Focused tests lock plus/minus sensitivity, zero-minus LRS reduction against the AP59 helper, live non-LRS weak/network multiplier propagation, default current-moment application, metadata-only baseline isolation, and fail-closed behavior for unsupported LRS/non-LRS mode routing. |
| AP61 | stage-recorded (historical; not current capability) | Weak-rate convergence and candidate gate.  `src/rabbit/validation/augmented_weak_rate_gates.py` and `scripts/run_augmented_weak_rate_gate.py` sweep q and S2 angular smoke ladders for the AP59/AP60 opt-in LRS/non-LRS moment-input rate factors, record lambda_np/lambda_pn factors and deltas, and enforce physical sign/range and unsupported-mode diagnostics before those modes are used in later coupled full-span solves. | Focused tests lock the report contract, bounded six-case LRS/non-LRS pass, excessive delta failure, unsupported angular-mode diagnostics, spec validation, JSON writer output, and CLI summary output.  This closes the rate-only candidate/convergence gate while preserving the full coupled-solve, publication plot, SMC, public dispatch, and QKE blockers. |
| AP62 | stage-recorded (historical; not current capability) | Non-LRS nonlinear transport logit-RHS operator using the existing LRS nonlinear transport as oracle.  `src/rabbit/transport/augmented_nonlrs_transport.py` adds the diagonal S2 nonlinear collisionless nodal RHS with q, mu, and periodic phi derivatives, the shear scalar `2*(Sigma_+ W_+ + Sigma_- W_-)`, angular-gradient drift terms, and logit-chain projection into the staged `{monopole, W_+, W_-}` A-mode basis while preserving the AP37 source-only shell. | Focused tests lock agreement with `src/rabbit/jax/nonlinear_transport.py` in the LRS limit, zero-shear/FLRW quietness, `Sigma_-=0` minus-mode reduction, `Sigma_-` sign reversal, mode reconstruction/projection parity, invalid-input rejection, and explicit `nonlrs_s2_nonlinear_collisionless_transport_rhs_v1` metadata.  This is a private transport operator, not public dispatch or an AP65 coupled full-BBN path. |
| AP63 | stage-recorded (historical; not current capability) | Non-LRS nonlinear transport solve shell without weak/network/collision feedback.  `run_augmented_nonlrs_nonlinear_transport_solve(...)` integrates the AP62 operator in SciPy with live shear-stress feedback, FD monopole initialization by default, LSODA/Radau method selection, and the same staged q/S2 grid controls used by AP62. | Focused tests pass finite short-span trajectories, FLRW quietness, `Sigma_-=0` reduction, finite stress feedback, invalid span/input rejection, and explicit `nonlrs_s2_nonlinear_transport_coevolution_v1` metadata before any 3T, weak/network, collision feedback, public dispatch, or QKE coupling. |
| AP64 | stage-recorded (historical; not current capability) | Non-LRS nonlinear transport plus live weak/network and 3T thermo/Hubble coupling.  `run_augmented_nonlrs_nonlinear_collisionless_weak_network_3T_solve(...)` replaces the source-only AP39 transport block with the AP63 nonlinear S2 transport option, and `run_augmented_nonlrs_candidate_weak_network_3T_solve(..., transport_model=...)` provides explicit source-only versus nonlinear routing. | Focused tests lock dynamic Hubble, live AP60 non-LRS moment-input weak-rate application, abundance normalization, plus/minus stress, solver metadata, shared-initial-state source-only versus nonlinear routing, and fail-closed transport-model selection.  This is still no collision-sourced thermo feedback, no public dispatch, and no QKE. |
| AP65 | stage-recorded (historical; not current capability) | Collision-coupled nonlinear non-LRS 3T integration.  The AP64 nonlinear solve now accepts an explicit `Augmented3TCollisionThermoSource` callback, `run_augmented_nonlrs_nonlinear_angular_collision_weak_network_3T_solve(...)` composes the AP41 angular source with AP64 nonlinear transport and AP60 weak rates under an opt-in wrapper, the angular source's projected `dA_modes` collision term is added to the nonlinear hierarchy RHS when present, `run_augmented_nonlrs_nonlinear_pstf_radial_collision_weak_network_3T_solve(...)` routes the AP6 descriptor-driven `pstf_radial` source through the same nonlinear S2 transport shell with budgeted live-RHS diagnostics, and `run_augmented_nonlrs_nonlinear_combined_collision_weak_network_3T_solve(...)` now evaluates AP41 angular and AP6 `pstf_radial` sources together without double counting: angular moments are retained as diagnostics, while the finite-mass AP6 radial source supplies the effective thermo moments and kinetic `dA_modes` in the nonlinear RHS callback.  The radial and combined wrappers now forward default `unit_direction_gaussian` and opt-in `radial_gaussian` momentum-delta controls into the AP6 provider, so the p-dependent closure can run inside the AP65 solve path.  `build_augmented_nonlrs_nonlinear_combined_collision_3t_solve_artifact(...)` and its CLI emit deterministic AP65 combined-source JSON evidence with the v2 no-double-count contract. | Focused tests lock collision-moment thermo feedback metadata, collision kinetic `dA_modes` RHS application, live `Sigma_-`/S2 grid source context, temperature splitting from source moments, nonlinear-grid wrapper wiring, live-source update behavior, AP6 `pstf_radial` radial source moments on the nonlinear shell, angular diagnostic plus radial-effective source selection, radial momentum-delta forwarding, radial `dA_modes` injection, budgeted source-evaluation diagnostics, real `radial_gaussian` combined-source smoke output, JSON writer/CLI summaries, and fail-closed source-update policy while keeping public dispatch disabled.  Publication validation atlas, guarded inference access, likelihood schema, smoke SMC plumbing, runtime/cache controls, synthetic SMC validation, figure-ready artifact tables, diagnostic plots, reproducibility packaging, final readiness audit, coupled weak-rate smoke evidence, same-CL3 AP66 weak-rate matrix control hardening, AP77 readiness-audit linkage, AP80 profile-level weak-rate convergence diagnostics, and AP81 six-monomial collision-factor wiring are now AP67-AP81; production SMC evidence and public-production promotion remain blocked after the AP76/AP79 `not_promoted` decision. |
| AP66 | stage-recorded (historical; not current capability) | Publication-candidate convergence matrix built from the existing convergence report contracts.  `src/rabbit/validation/augmented_publication_matrix.py` and `scripts/run_augmented_publication_convergence_matrix.py` run AP65/AP4 combined-source evidence over `N_q`, `N_mu`, `N_phi`, span, solver tolerance, source model, source policy, same-CL3 `metadata_only`/`nonlrs_s2_cl3_quadrupole_input` weak-rate ladders, radial energy-normalization/source-budget/momentum-delta controls, fixed/charge-neutral electron-bath controls, and scalar QED-model selection with JSON artifacts.  The matrix supports both `angular` and `combined_angular_pstf_radial` source-model rows, including AP4 `piecewise_frozen` combined-source rows that recompute source moments at explicit subspan ends and record `source_update_subspan_*` plus handoff observables.  It records the final collision-source `dA_modes` amplitude as `collision_dA_abs_max_final`, component `combined_angular_dA_abs_max_final`/`combined_pstf_radial_dA_abs_max_final`, `pstf_radial` source-budget observables, radial momentum-delta provenance, terminal electron-mu/charge-asymmetry observables, and row-level `finite_mu_scaled` versus `exact_finite_mu_scalar` scalar-QED markers.  FB-10 extends AP66 so a supplied FB-09 chained resolution artifact is validated as passed/no-QKE/not-public-dispatch/not-production-SMC and compactly linked through `full_chain_evidence.row_links` for every chained `N_q`/`N_mu`/`N_phi` row. | Focused tests lock AP65 observable extraction, AP4 piecewise gate routing, terminal kinetic-source amplitude extraction, terminal electron-bath observable extraction, q/`N_mu`/`N_phi` ladder routing, source-model selection, radial-control and momentum-delta forwarding, electron-bath control forwarding, scalar QED-model forwarding, weak-rate/source-policy matrix coverage, same-CL3 metadata controls after AP78, FB-09 chained-artifact contract/scope rejection and row-level provenance links, first-converged candidate settings including piecewise subspan ends, no-candidate residual risks, JSON writer/CLI summaries, invalid input rejection, and real smoke evidence for the angular default plus short-span combined-source fixed, charge-neutral, and piecewise-frozen rows.  This is diagnostic publication-candidate evidence only; AP67 consumes it, AP68 wraps it for guarded inference, AP69 adds schema-level likelihood access, AP70 adds smoke tempered-SMC plumbing, AP71 adds runtime/cache controls, AP72 adds synthetic validation, AP73 normalizes figure-ready tables, AP74 renders diagnostic plots, AP75 packages reproducibility evidence, AP76 final audit is landed, AP77 adds smoke coupled weak-rate evidence, AP78 removes CL0-vs-CL3 confounding from AP66 weak-rate matrix controls, and public dispatch/QKE remain out of scope. |
| AP67 | stage-recorded (historical; not current capability) | Physics validation atlas reusing existing AP55/AP57/AP66 evidence links and the AP65 candidate solve.  `src/rabbit/validation/augmented_validation_atlas.py` and `scripts/run_augmented_validation_atlas.py` build deterministic FLRW/null, LRS `Sigma_-=0`, small non-LRS, finite non-LRS, injected-moment, and short-span stress cases with output/convergence/full-chain/inference readiness labels.  The atlas now records AP65 source-model provenance, can run `combined_angular_pstf_radial` rows with radial energy-normalization/source-budget/momentum-delta controls, forwards fixed/charge-neutral electron-bath controls into AP65 cases and nested AP66 evidence, forwards scalar-QED model selection including `exact_finite_mu_scalar` into AP65 cases and nested AP66 evidence, records terminal electron-mu/charge-asymmetry observables and scalar-QED markers, links AP66 candidate source-model, radial momentum-delta, scalar-QED metadata, AP66 `source_update_policy`/`source_update_subspan_ends` provenance for AP4-backed `piecewise_frozen` rows, and AP66/FB-09 full-chain artifact contract/path/row-count/row-key/no-QKE/not-public provenance, uses same-CL3 weak-rate controls, and records radial source-budget exhaustion as failed rows. | Focused tests lock known-limit case coverage, AP66 evidence-link reuse including nested AP66 piecewise source-refresh policy/subspan provenance and full-chain provenance, source-model dispatch, radial-control, momentum-delta, electron-bath, and scalar-QED forwarding, terminal electron-bath observable extraction, exact-scalar-QED atlas markers, required-full-chain fail-closed checks, budget-exhaustion failure rows, output/convergence/full-chain readiness separation, failed-solve rejection, JSON writer/CLI summaries including nested AP66 source-update and chained-artifact forwarding, case-spec validation, a real angular single-case smoke, and a real combined-source single-case smoke.  This is a diagnostic validation atlas only; AP68/AP69 add guarded inference and likelihood-schema access, AP70 adds smoke tempered-SMC plumbing, AP71 adds runtime/cache controls, AP72 adds synthetic validation, AP73 normalizes figure-ready artifacts, AP74 renders diagnostic plots, AP75 packages reproducibility evidence, AP76 final audit is landed, AP77 adds smoke coupled weak-rate evidence, and public dispatch/QKE remain out of scope. |
| AP68 | stage-recorded (historical; not current capability) | Candidate forward-model adapter for inference.  `src/rabbit/inference/augmented_forward_model.py` wraps the AP65 candidate solve in the existing `ForwardModel`/`BBNLikelihood` style API without adding canonical/public default dispatch.  The guarded config now exposes AP65 source-model selection, so inference calls can run either the angular-only wrapper or the combined angular+`pstf_radial` wrapper with radial energy-normalization, source-budget controls, radial momentum-delta model/sigma controls, fixed/charge-neutral electron-bath controls, scalar-QED model selection including `exact_finite_mu_scalar`, and the AP4-style `piecewise_frozen` combined-source operator-split source-refresh path over explicit subspan ends.  FB-12 extends AP68 with `execution_mode="full_chain"`, which builds the FB-04 chained full-BBN runner spec from the guarded AP68 parameters, or consumes a cached chained artifact, validates no-QKE/not-public/not-production-SMC scope, and maps terminal chained `Yp`/`D/H` plus window/replay/source metadata into `BBNPrediction`.  Prediction metadata carries radial momentum-delta provenance, source-refresh subspan/count metadata, terminal electron-mu/charge-asymmetry observables, scalar-QED model/contract markers, full-chain artifact provenance, and the mode-specific collision-feedback contract. | Focused tests lock API inputs, AP65 parameter forwarding, source-model dispatch, radial-control, momentum-delta, electron-bath, scalar-QED forwarding, AP68 piecewise-frozen subspan handoff into repeated AP65 frozen-source solves, aggregate `nfev`/source-evaluation metadata, full-chain builder/cached-artifact routing, full-chain artifact scope rejection, radial budget-failure metadata, guarded config validation, terminal electron-bath metadata, exact-scalar-QED prediction metadata, `Yp`/`D/H` extraction from the PRIMAT abundance vector and chained terminal observables, collision-source `dA` metadata, no accidental public dispatch registration, compatibility with the existing Gaussian likelihood wrapper, real short-span combined-source fixed/charge-neutral/piecewise smokes, and a real two-window full-chain smoke.  This is a guarded candidate adapter only; AP69 adds likelihood schema, AP70 adds smoke SMC plumbing, AP71 adds runtime/cache controls, AP72 adds synthetic validation, AP73 adds figure-ready artifacts, AP74 adds diagnostic plots, AP75 adds a reproducibility bundle, AP76 final audit is landed, AP77 adds smoke coupled weak-rate evidence, and public dispatch/QKE remain out of scope. |
| AP69 | stage-recorded (historical; not current capability) | Likelihood, priors, and parameter schema for augmented SMC using existing inference contracts.  `src/rabbit/inference/augmented_smc.py` defines continuous `{Sigma_H, Sigma_H_minus, eta, tau_n}` parameter specs, fixed source-policy/source-model/radial-normalization/radial-momentum-delta/electron-bath/scalar-QED/weak-rate/solver controls, vector/dict coercion, priors, AP68 forward-config provenance including radial source budgets, radial momentum-delta settings, AP68 piecewise source-refresh subspan ends, electron-bath mode, and scalar-QED model, and an AP68-backed failure-aware likelihood wrapper.  FB-13 extends the same schema/runtime surface with AP68 `execution_mode`, full-chain source-refresh controls, replay/restart verification toggles, full-chain window edges, and cache-key provenance so AP69 and AP71 cache contexts can distinguish direct AP65 calls from AP68 full-chain calls. | Focused tests reject invalid priors and solver controls, preserve units/labels, round-trip vectors, produce stable log-likelihood values for deterministic AP68 outputs, propagate AP68 classified solver failures, verify source-model, source-update/subspan, radial, radial momentum-delta, electron-bath, scalar-QED, and full-chain controls in schema metadata, verify AP71 full-chain cache-context separation, run real angular and combined-source fixed/charge-neutral/piecewise AP68-backed smoke likelihoods, run a real AP68 full-chain likelihood smoke, and lock compatibility with the existing sampler vector-loglike adapter.  This is schema and likelihood plumbing only; AP70 adds a smoke SMC runner, AP71 adds cache/restart controls, AP72 adds synthetic validation, AP73 adds figure-ready artifacts, AP74 adds diagnostic plots, AP75 adds a reproducibility bundle, AP76 final audit is landed, AP77 adds smoke coupled weak-rate evidence, and public dispatch/QKE remain out of scope. |
| AP70 | stage-recorded (historical; not current capability) | Smoke-scale tempered SMC runner for the augmented candidate.  `run_augmented_tempered_smc(...)` reuses the AP69 schema/likelihood, supports replayable seeds, explicit temperature schedules, ESS-triggered resampling, optional random-walk rejuvenation, caller-provided initial particles, failure accounting, and direct result-metadata preservation of AP69 source-model/source-refresh/radial/electron-bath/scalar-QED controls plus radial momentum-delta, `piecewise_frozen` subspan AP68 forward-config provenance, and FB-14 full-chain execution/window/cache/source-refresh controls.  `scripts/run_augmented_tempered_smc.py` now accepts full-chain CLI controls and cached chained artifacts. | Focused tests reject bad temperature schedules, lock replayability, final beta-one normalization, finite weights, ESS/resampling metadata, initial-particle handling, non-finite forward-result accounting, and source-model/source-refresh/radial/electron-bath/scalar-QED/full-chain-control metadata preservation with radial momentum-delta and `piecewise_frozen` subspan provenance.  A real two-particle CLI SMC smoke completed through AP68 full-chain calls with finite log-likelihoods and zero forward failures.  This is smoke-scale diagnostic SMC plumbing only; AP71 adds checkpoint/cache controls, AP72 adds synthetic validation, AP73 adds figure-ready artifacts, AP74 adds diagnostic plots, AP75 adds a reproducibility bundle, AP76 final audit is landed, AP77 adds smoke coupled weak-rate evidence, and public dispatch/QKE remain out of scope. |
| AP71 | stage-recorded (historical; not current capability) | SMC runtime controls and artifact cache for expensive augmented AP68 calls.  `AugmentedSMCRuntimeConfig`, `AugmentedSMCCachedLikelihood`, and `scripts/run_augmented_tempered_smc.py` add checkpoint/resume, duplicate-call avoidance, source-model/source-refresh/radial/electron-bath-control-aware cache keys plus scalar-QED/full-chain-aware cache context and runtime payloads, batched likelihood hooks, run manifests, CLI dry-run forwarding for AP69 source/source-refresh/radial/electron-bath controls plus `N_span` end, scalar-QED, radial momentum-delta, `piecewise_frozen` subspan controls, and full-chain execution/window/cache/source-refresh/replay/restart controls, last successful AP68 prediction metadata preservation, and failure-metadata preservation around the AP70 runner. | Focused tests lock manifest/checkpoint emission, duplicate forward-call avoidance, source-model/source-refresh/radial/electron-bath-control cache separation, scalar-QED and full-chain cache separation/cache-context provenance, CLI dry-run electron-bath, `N_span` end, scalar-QED, radial momentum-delta, `piecewise_frozen` subspan, and full-chain control forwarding, successful AP68 prediction-metadata preservation, non-finite failure metadata preservation through cache records, resume from checkpoint without repeating checkpointed calls, incompatible-manifest rejection, and batched uncached likelihood evaluation.  This remains diagnostic SMC runtime plumbing; AP72 adds synthetic validation, AP73 adds figure-ready artifact tables, AP74 adds diagnostic plots, AP75 adds a reproducibility bundle, AP76 final audit is landed, AP77 adds smoke coupled weak-rate evidence, and public dispatch/QKE remain out of scope. |
| AP72 | stage-recorded (historical; not current capability) | SMC validation suite using the existing null/recovery and non-LRS SMC conventions.  `src/rabbit/validation/augmented_smc_validation.py` and `scripts/run_augmented_smc_validation.py` run deterministic synthetic FLRW/null and non-LRS injection cases through the AP70/AP71 SMC path.  The synthetic likelihood can now accept AP69 schema overrides and records source-model/source-refresh/`N_span`/radial/electron-bath controls, scalar-QED model selection, and radial momentum-delta controls in validation metadata and CLI dry-run schema payloads without claiming those analytic tests validate the physical combined-source solve.  FB-15 adds an opt-in physical full-chain smoke row that calls the AP68 full-chain likelihood from AP70/AP71 SMC, records AP68 terminal BBN observables and CPU-JAX/Rodas5P replay evidence, and keeps the default AP73 artifact synthetic-only. | Focused tests lock case-spec validation, null no-spurious-shear checks, injection/recovery pass/fail reporting, posterior summaries, ESS trace, acceptance diagnostics, logZ-error proxy, forward-success accounting, source-model/source-refresh/`N_span`/radial/electron-bath-control provenance, scalar-QED provenance, radial momentum-delta provenance, JSON artifact writing, CLI dry-run/validation output, full-chain AP72 schema control forwarding, and the opt-in full-chain physical smoke artifact.  A real CPU smoke with two physical full-chain SMC particles passed with zero forward failures and successful Rodas5P window-map replay checks.  This remains stage-scoped diagnostic validation; AP73 consumes the default synthetic-only AP72 artifacts, AP74 renders diagnostic plots, AP75 adds a reproducibility bundle, AP76 final audit is landed, AP77 adds smoke coupled weak-rate evidence, and public dispatch/QKE remain out of scope. |
| AP73 | stage-recorded (historical; not current capability) | Publication artifact schema and figure data builder on the existing figure cache discipline.  `src/rabbit/validation/augmented_publication_artifacts.py`, `scripts/build_augmented_publication_artifacts.py`, `scripts/figure_cache_schema.py`, and `scripts/figure_registry.py` convert AP66 convergence, AP67 validation-atlas, AP72 SMC-validation, and existing Schramm/likelihood cache rows into versioned figure-ready tables with provenance, carrying `collision_dA_abs_max_final` with explicit per-e-fold units, AP66 electron-mu/charge-asymmetry columns with MeV/MeV^3 units, plus source-model/source-refresh/`N_span`/radial-control/radial-momentum-delta/electron-bath-control/scalar-QED context for AP66/AP67/AP72-derived AP74 plots.  AP73 now preserves AP66 and AP67 `qed_correction_model` categorical row context plus finite-mu-scaled versus exact-scalar-QED markers where present and AP72 `piecewise_frozen` source-refresh subspan context in SMC posterior/temperature rows.  FB-16 extends AP73 so a passed AP72 physical full-chain smoke artifact can produce diagnostic Schramm rows with real AP68 terminal `Yp`/`D/H` and CPU-JAX/Rodas5P replay status, while malformed non-synthetic AP72 artifacts still fail closed. | Focused tests lock schema versions, required columns, units including terminal kinetic-source amplitude and electron-bath observables, source-model/source-refresh/`N_span`/radial/radial-momentum-delta/electron-bath/scalar-QED provenance in convergence/validation/SMC tables where present, AP72 physical full-chain Schramm row extraction, commit provenance, registry keys, JSON writer output, Schramm cache loading, CLI summary output, and rejection of stale/mixed-contract or invalid non-synthetic SMC artifacts.  This creates AP74 plot inputs only; it does not generate publication plots, add production SMC evidence, promote public dispatch, or change the no-QKE boundary. |
| AP74 | stage-recorded (historical; not current capability) | Publication plot generator integrated with the existing paper/report figure stack.  `src/rabbit/figures/augmented_publication_plots.py`, `scripts/plot_augmented_publication_figures.py`, `scripts/generate_paper_figures.py`, and `scripts/regenerate_all_figures.sh` render AP73 convergence, validation-atlas, Schramm `Y_p`/`D/H`, posterior, and SMC temperature panels with a plot manifest, source-model/source-refresh/`N_span`/radial-momentum-delta/electron-bath/scalar-QED provenance, and PNG hashes.  FB-17 extends the Schramm plot contract to accept AP73 diagnostic full-chain physical-smoke rows and preserve completed-window/Rodas5P replay provenance. | Focused tests verify expected files, nonempty PNG layers, axis metadata, registry/cache metadata, source-model/source-refresh/`N_span`/radial-momentum-delta/electron-bath/scalar-QED manifest provenance, AP72 physical-smoke Schramm plot provenance, reproducible hashes, CLI summary output, paper-generator option wiring, regeneration-script hook wiring, and rejection of unsupported or incomplete artifacts.  This is diagnostic AP73-derived plotting only; it does not add production SMC evidence, public dispatch, or QKE. |
| AP75 | stage-recorded (historical; not current capability) | Reproducibility bundle packaging for the AP66/AP67/AP72/AP74 publication artifacts.  `src/rabbit/validation/augmented_publication_bundle.py` and `scripts/package_augmented_publication_bundle.py` validate required artifact contracts, common provenance, AP74 source-model/source-refresh/`N_span`/radial-momentum-delta/electron-bath/scalar-QED consistency, AP74 plot hashes/registry keys, copied artifact and figure manifests, environment metadata, command records, and claim-boundary notes.  FB-18 extends the same AP75 bundle so a passed AP72 full-chain physical-smoke artifact and AP74 full-chain Schramm plot can be packaged together with finite AP68 terminal `Yp`/`D/H`, completed-window counts, and CPU-JAX/Rodas5P replay status. | Focused tests lock diagnostic-only bundle metadata, required AP66/AP67/AP72/AP74 artifact summaries, common git-commit checks, fail-closed rejection of non-synthetic AP72 artifacts without a passed full-chain physical-smoke row, public-dispatch AP74 rejection, public-production plot-label rejection, source-model/source-refresh/`N_span`/radial-momentum-delta/electron-bath/scalar-QED bundle provenance, full-chain physical-smoke summary/provenance preservation, manifest writing, plot copy/hash validation including full-chain Schramm provenance, legacy AP74 top-level provenance backfill from plot records, mismatch rejection, and CLI dry-run output.  This is diagnostic packaging only; full-chain physical-smoke evidence is not production SMC evidence and does not add public dispatch, QKE, or promotion approval. |
| AP76 | stage-recorded (historical; not current capability) | Final publication-readiness audit and registry promotion decision.  `src/rabbit/validation/augmented_publication_readiness.py` and `scripts/run_augmented_publication_readiness_audit.py` consume the AP75 bundle, re-check AP66/AP67/AP72/AP74 summary contracts, reject promoted/public/QKE-tampered bundles, validate AP75/AP74/plot source-model/source-refresh/`N_span`/radial-momentum-delta/electron-bath/scalar-QED consistency, record dispatch-surface status, and write the final claim ledger.  FB-19 extends the audit so AP75 bundles containing the AP72 full-chain physical-smoke row plus AP74 full-chain Schramm provenance are accepted, audited, and recorded as diagnostic/not-promoted evidence with finite `Yp`/`D/H`, completed-window counts, CPU-JAX/Rodas5P replay status, and live-source repeated-run BBN readout provenance only when matching FB-21 gate contract/claim/window/payload/delta provenance is present.  AP79 extends the same audit to require the AP77 coupled weak-rate gate artifact. | Focused tests lock the `not_promoted` diagnostic decision, source-model/source-refresh/`N_span`/radial-momentum-delta/electron-bath/scalar-QED readiness-ledger provenance, residual blockers, forbidden-claim wording, AP72 synthetic-only evidence, AP72 full-chain physical-smoke diagnostic evidence including live-source repeated-run BBN readout only with FB-21 gate provenance, AP74 diagnostic/cache and full-chain Schramm plot labels, AP75 source-refresh, electron-bath, scalar-QED, radial momentum-delta, full-chain-smoke mismatch/invalid-mode rejection, FB-21 missing, stale-contract, source, fractional-window, mismatch, and partial-plot-coverage rejection, no canonical dispatch registration, JSON writer output, CLI dry-run output, AP77 pass status, AP77 no-QKE/no-public-dispatch status, same-CL3 control basis, paired q rows, empty nested AP77 violations, q-ladder rows, and AP60 rate-application metadata.  AP76/AP79 deliberately keep public production support forbidden because production SMC evidence, promoted full-span angular collision feedback, promotion-grade coupled weak-rate convergence, full physical BBN span, promotion-tolerance convergence ladders, and public/GPU dispatch gates remain incomplete. |
| AP77 | stage-recorded (historical; not current capability) | Coupled weak-rate smoke gate beyond AP61 rate-only evidence.  `src/rabbit/validation/augmented_coupled_weak_rate_gate.py` and `scripts/run_augmented_coupled_weak_rate_gate.py` run the AP65 nonlinear angular collision-feedback 3T wrapper with injected non-LRS S2 moments in same-CL3 `metadata_only` control rows and `nonlrs_s2_cl3_quadrupole_input` applied rows, forward fixed/charge-neutral electron-bath controls into the coupled solve, then compare live weak-rate deltas and all adjacent q-ladder drift pairs. | Focused tests lock exact AP60 rate-application metadata inside the coupled solve, same-CL3 control isolation, electron-bath spec/CLI forwarding and case-row provenance, bounded nonzero `lambda_np`/`lambda_pn` deltas against metadata-only controls, default-injection nonnegativity, finite solve outputs, solve-effort checks, JSON writer output, CLI dry-run output, invalid-spec rejection, fail-closed missing-application behavior, every-adjacent-pair q drift checks, and a real AP65 smoke solve.  This moves beyond AP61 but remains smoke-scale; it does not provide promotion-grade full-span weak-rate convergence, public dispatch, production SMC evidence, or QKE. |
| AP78 | stage-recorded (historical; not current capability) | Same-CL3 weak-rate control hardening for the AP66 publication matrix.  `src/rabbit/validation/augmented_publication_matrix.py` now builds both `metadata_only` and `nonlrs_s2_cl3_quadrupole_input` weak-rate-mode rows with `correction_level=3`, so AP66 weak-rate ladder comparisons no longer mix AP60 angular application with CL0-vs-CL3 base weak-rate-kernel changes. | Focused tests lock same-CL3 `_weak_config_for_mode(...)` behavior, AP66 observable metadata reporting `weak_rate_correction_level=3` for metadata-only rows, and the existing AP66 q/`N_mu`/`N_phi`, JSON, CLI, invalid-input, candidate-selection, failed-solve, and real frozen-source smoke gates.  This is a diagnostic matrix-control upgrade only; it does not provide promotion-grade full-span weak-rate convergence, public dispatch, production SMC evidence, or QKE. |
| AP79 | stage-recorded (historical; not current capability) | Readiness-audit coupled weak-rate linkage.  `src/rabbit/validation/augmented_publication_readiness.py` and `scripts/run_augmented_publication_readiness_audit.py` now require an AP77 coupled weak-rate gate artifact in addition to the AP75 reproducibility bundle, and record that AP77 summary, including electron-bath controls, in the readiness ledger. | Focused tests lock AP77 contract/schema/stage validation, `passed=True`, empty top-level/comparison/convergence violations, no-QKE scope, no public dispatch, no production SMC, same-CL3 control basis, paired metadata-only/AP60 rows for every q value, exact comparison metadata tying each row to metadata-only controls and AP60 applied rows, metadata-only no-rate-correction rows, AP60 applied rows, AP77 top-level/input/case-row electron-bath provenance consistency, q-ladder convergence rows for every adjacent q pair, writer output, and CLI dry-run input surfacing.  This links AP77 to the final audit but remains diagnostic; it does not provide promotion-grade full-span weak-rate convergence, public dispatch, production SMC evidence, or QKE. |
| AP80 | stage-recorded (historical; not current capability) | Profile-level coupled weak-rate convergence artifact.  `src/rabbit/validation/augmented_coupled_weak_rate_convergence.py` and `scripts/run_augmented_coupled_weak_rate_convergence.py` wrap AP77 coupled weak-rate gate reports into smoke and explicit extended q-ladder profiles, including the default q=(3,4) profile and an optional q=(3,4,5) profile for slower gates. | Focused tests lock AP80 contract/schema/stage metadata, no-QKE/no-public-dispatch/no-production-SMC boundaries, successful multi-profile aggregation, failed-profile violation recording, requested-vs-observed q-ladder rejection, duplicate case-row rejection, contradictory nested AP77 comparison/convergence pass-status rejection, unique profile validation, JSON writer output, and CLI dry-run profile surfacing.  This moves the coupled weak-rate blocker beyond a single AP77 smoke gate, but remains diagnostic and SciPy-first; it does not provide full-span promotion-grade weak-rate convergence, public dispatch, production SMC evidence, or QKE. |
| AP81 | stage-recorded (historical; not current capability) | Six-monomial Pauli collision statistical factor.  `collision_statistical_factor(...)` now evaluates the quartic-cancelled fermionic 2-to-2 polynomial with signed monomials `34`, `12`, `123`, `124`, `134`, and `234`; deterministic `nu-e`/pair references, the deterministic pairwise diagonal `nu-nu` 2-to-2 reference, the default staged NumPy AP19/AP33/AP35/AP41 diagonal `nu-nu` source bridge, legacy SciPy `NuEScatteringOperator`/`PairProcessOperator`, JAX nu-e/pair kernels, and JAX diagonal no-QKE nu-nu kernels use that shared algebra.  The staged NumPy source bridge now evaluates all nine ordered `{nue,nuebar,nux}` bank pairs by default, including identical-bank self-scattering with Fierz factor 2 and same-bank number/energy-neutral projection, while retaining an explicit off-diagonal-only legacy comparison switch.  The AP6 radial follow-up now also projects all nine default descriptor-driven radial diagonal `nu-nu` source rows for particle-number conservation before AP18 thermo/hierarchy feedback: identical-bank rows are number/energy-neutral, and six off-diagonal rows are number-neutral plus unordered-pair energy-neutral while preserving relative raw species energy-transfer differences.  The earlier AP6 radial geometry update replaces the default uniform angular factor with normalized unit-direction momentum-delta weights, so the deterministic angular kernel favors vector-closed `e1+e2=e3+e4` quadrature tuples while preserving the smoke-scale radial source normalization; an opt-in `radial_gaussian` model evaluates the full `p1 e1 + p2 e2 - p3 e3 - p4 e4` residual for small p-dependent studies.  The CPU hot-path follow-up reuses static angular geometry inside radial-grid assembly when the momentum-delta tensor is radial-independent, leaving callable p-dependent closures unchanged. | Focused tests lock the monomial set/signs, absence of a quartic `1234` term, FD detailed balance, legacy operator sharing, replay-stable non-equilibrium numeric values for fixed `N_q=4` nu-e, pair, and pairwise diagonal `nu-nu` references, all-nine/default and off-diagonal legacy pair-count diagnostics, identical-bank number/energy projection diagnostics, off-diagonal radial `nu-nu` number and pair-energy projection diagnostics, pairwise bridge/source metadata, number/energy closure projections, normalized unit-direction and opt-in p-dependent momentum-delta weighting, static angular-geometry reuse without an external cache, real nonlinear non-LRS `pstf_radial` smoke diagnostics, and JAX polynomial/parity/preflight behavior.  This lands executable scalar collision-factor algebra from `neutrino_collision_term_PSTF.md` into the staged diagonal `nu-nu` source route with per-bank number closure and an explicit effective-`nu_x` weighted-energy closure projection.  AP81/AP6 still does not provide public dispatch, production SMC evidence, full anisotropic weak-rate integration, exact Dirac-delta angular convergence at promotion tolerances, or QKE. |

`stage-recorded (historical; not current capability)` means the named AP
deliverable and its exit gate are implemented and tested in this
checkout.  It does not promote the overall augmented-PSTF no-QKE
programme beyond its registry status; blockers named in the Purpose or
Exit gate columns still apply until a later AP explicitly closes them.

2026-05-14 AP4/AP6/AP18 update: explicit collision source callbacks can now
carry kinetic hierarchy payloads as well as thermo moments.  AP35/AP41 angular
sources return projected `dA_modes`, and the AP6 `pstf_radial` adapter maps
process `C_modes(q, mode)` into species-indexed `dA_modes`; the LRS,
source-only non-LRS, and nonlinear non-LRS 3T shells add those payloads to the
augmented hierarchy RHS when the opt-in source is selected.  This remains
staged/no-QKE and does not promote public production dispatch.

2026-05-14 AP6 charge-neutral radial update: the algebraic charge-neutral
`mu_e` wrapper for the AP6 `pstf_radial` source now preserves the underlying
radial `dA_modes` hierarchy payload instead of returning only thermo moments
and diagnostics.  The charge-neutral e-/e+ bath split can therefore feed both
the 3T energy-source terms and the opt-in kinetic hierarchy RHS on the staged
radial source path.

2026-05-14 AP4/AP6 artifact update: LRS and non-LRS collision-feedback
artifact observables now record the final collision source hierarchy amplitude
as `collision_dA_abs_max_final`, so real `pstf_radial` smoke artifacts expose a
concrete kinetic-source value alongside the existing thermo `dQ` moments.

2026-05-15 AP6 direct source artifact update: the fixed-`mu_e` sensitivity and
charge-neutrality source artifacts now record `collision_dA_abs_max` directly
from the evaluated AP6 `source.dA_modes` payload, and their delta maps include
that kinetic-source amplitude alongside the `dQ_nue_pair_N`/`dQ_nux_bank_N`
thermo moments.  This closes the direct artifact observability gap without
promoting public/default collision-coupled full-BBN dispatch.

2026-05-15 AP55/AP6 source-budget update: the AP55 deterministic
collision-source budget report now includes an executable
`lrs_pstf_radial_process_budget` case for the AP6 descriptor-driven radial
source.  That case records concrete radial moments, finite-mass process
markers, `radial_max_abs_C_mode`, and `collision_dA_abs_max` from the returned
kinetic hierarchy payload while preserving the no-QKE/staged boundary.

2026-05-15 AP4 charge-neutral evolved-state update: the LRS 3T charge-neutral
finite-mass e-/e+ EOS path now carries an evolved
`electron_charge_asymmetry_density_MeV3` state seeded from the network charge
density, uses that state as the `mu_e` target in the RHS, and records the
state history/final value with contract
`charge_neutral_positive_charge_density_evolved_v1`.  This initially closed
the LRS piece of the independent charge-asymmetry blocker while leaving
non-LRS evolved charge-asymmetry states for the follow-up below.

2026-05-15 AP4 non-LRS charge-neutral evolved-state update: the source-only
and nonlinear non-LRS 3T charge-neutral finite-mass e-/e+ EOS paths now carry
the same evolved `electron_charge_asymmetry_density_MeV3` state.  Both shells
seed the state from `eta * n_gamma(T_gamma) * sum_i Z_i X_i/A_i`, pass it as
the charge-neutral `mu_e` target at each RHS evaluation, evolve it with the
network and photon-temperature derivatives, and expose the state history,
final value, and `charge_neutral_positive_charge_density_evolved_v1` contract.
This closes the staged LRS/source-only non-LRS/nonlinear non-LRS independent
charge-asymmetry state blocker.  Exact finite-`mu_e`/tensor QED, public
dispatch, production SMC, promotion-grade full-BBN, and QKE remain out of
scope.

2026-05-15 AP4 exact finite-mu scalar QED update: the thermo layer now exposes
an opt-in `exact_finite_mu_scalar` QED correction mode.  The exact QED EOS
integrals use the signed occupation sum `f_e(E, mu_e) + f_pos(E, mu_e)`, reduce
to the zero-`mu_e` exact correction, and feed
`qed_delta_rho_with_electron_mu(...)`,
`qed_delta_pressure_with_electron_mu(...)`, signed-`mu_e` plasma
energy/pressure, fixed-`mu_e` `d rho / dT`, `hubble_3T(...)`, and
`coupled_3T_rhs(...)` when selected.  The LRS, source-only non-LRS, and
nonlinear non-LRS 3T solve shells now expose the same
`qed_correction_model` control, pass it through every staged Hubble and thermo
RHS evaluation, and record `qed_correction_model` plus
`qed_correction_contract` in result metadata.  At `T=0.8 MeV`,
`mu_e=0.2 MeV`, the mode gives
`delta_rho = -0.001545790523714566 MeV^4` and
`delta_P = -0.00045775883349570345 MeV^4`.  This is scalar plasma-frame QED
thermodynamics only: anisotropic/tensor QED response, promotion-grade
exact-scalar-QED full-span coupled-solver validation, public dispatch, and QKE
remain out of scope.

2026-05-15 AP55/AP56/AP57 charge-neutral radial budget update: the AP55
source-budget report now also evaluates
`lrs_pstf_radial_charge_neutrality_budget` and
`nonlrs_pstf_radial_charge_neutrality_budget` with algebraic charge-neutral
finite-mass e-/e+ bath diagnostics from `phase1_to_phase2(Xn0)`.  AP56
preserves both rows, and AP57 adds LRS and non-LRS charge-neutral radial
source-budget cases to the sanity matrix.
The smoke artifact remains staged/no-QKE and does not promote default or
public collision-coupled full-BBN dispatch.

2026-05-15 AP4/AP65 combined full-span candidate update: the AP65 combined
angular+`pstf_radial` nonlinear 3T artifact is now consumed by a deterministic
span-ladder candidate gate in `rabbit.validation.augmented_stability`.  The
gate writes a JSON artifact/CLI and records terminal `Sigma_\pm`, `Pi_\pm`,
`A_\pm` RMS, 3T temperatures, Hubble rate, `Xn`, combined/radial/angular
`dA` amplitudes, source moments, radial source-evaluation budgets, and solver
effort.  A smoke run at `N_span=(0, 1e-14)`, `N_q=3`, `N_mu=3`, `N_phi=5`,
`standard_3t_plasma`, `frozen_initial_state`, and `RK23` returned
`T_gamma_final=0.7999999999999922`, `H_rate_s_final=0.4315487123652324`,
`Xn_final=0.13000000000065723`,
`collision_dA_abs_max_final=2.8419082353054516e-4`, and `nfev=5`.  This closes a concrete AP4 physical
candidate-gate gap but remains staged: promotion-grade full-BBN span,
public dispatch, production SMC validation, QKE, and promotion-tolerance
coupled convergence remain blocked.
The same gate now has real `live_rhs` two-span evidence at
`N_span=(0, 1e-14)` and `(0, 1e-12)` with a `256` radial source-evaluation
budget: both rows passed, `source_evaluations=7`, budget diagnostics were
present and passed, `max_span_length=1e-12`, and the longer row recorded
`T_gamma_final=0.7999999999992143`,
`H_rate_s_final=0.43154871236438125`,
`Xn_final=0.13000002470684513`, and
`collision_dA_abs_max_final=2.841908235303566e-4`.
The same gate now exposes a deterministic `warm` preset over
`N_span=(0, 1e-12)`, `(0, 1e-10)`, and `(0, 1e-8)` with `live_rhs`,
`max_pstf_radial_source_evaluations=2048`, and `max_nfev=50000`; a real CLI
run passed with `source_evaluation_max=70`, `max_span_length=1e-8`, and
`collision_dA_abs_max_final=3.362674147101354e-4`.
The companion AP4/AP65 combined-source source-policy span profile now runs
matched frozen-initial-state and `live_rhs` rows over the same two spans.  The
first real profile passed all four rows with frozen/live `nfev=5/5`,
frozen/live source evaluations `1/7`, `failure_rows=0`, and maximum
`collision_dA_abs_max_final=2.8419082353054516e-4`; live-minus-frozen thermo
and network deltas stayed at roundoff scale on both spans.
The AP4/AP65 combined full-span gate and source-policy span profile now also
forward `electron_chemical_potential_MeV` and
`electron_chemical_potential_mode` into the AP65 combined nonlinear artifact,
so the existing finite-mass e-/e+ fixed-`mu_e` and charge-neutrality radial
bath paths are executable from the full-span diagnostic surface.  A
charge-neutral live-RHS gate smoke run at `N_span=(0, 1e-14)` passed with
`source_evaluation_max=7`,
`collision_dA_abs_max_final=2.8419082353048944e-4`, and no violations.  The
charge-neutral frozen/live source-policy profile over `(0, 1e-14)` and
`(0, 1e-12)` passed all four rows with `failure_rows=0`,
frozen/live source evaluations `1/7`, and maximum
`collision_dA_abs_max_final=2.841908235304908e-4`.  This is still a
diagnostic full-span surface, not public dispatch or production SMC evidence.
The AP65 combined artifact and the AP4/AP65 gate/profile now propagate
terminal electron-bath observables from the nonlinear solve, including
`electron_chemical_potential_abs_max`,
`electron_chemical_potential_MeV_final`,
`electron_charge_asymmetry_density_MeV3_final`, and the evolved-state marker.
A charge-neutral AP65 combined smoke run reported
`electron_chemical_potential_MeV_final=3.3006028673558046e-10` and
`electron_charge_asymmetry_density_MeV3_final=6.623064974048602e-11`; the
AP4/AP65 full-span gate/profile summaries over the same charge-neutral surface
reported `electron_chemical_potential_abs_max=3.298439955909406e-10` and
`electron_charge_asymmetry_density_abs_max_MeV3=6.618724826621509e-11`.
The same AP17/AP36/AP42/AP45/AP49/AP6/AP65/AP4 diagnostic 3T surfaces now
accept, validate, record, and forward `qed_correction_model`, including the opt-in
`exact_finite_mu_scalar` scalar EOS mode already landed in the 3T thermo shells.
A real AP4/AP65 exact-scalar-QED tiny-span smoke gate at `N_span=(0, 1e-14)`,
`frozen_initial_state`, and `RK23` passed with
`T_gamma_final=0.7999999999999922`, `H_rate_s_final=0.4311207244202148`, and
the gate-level `qed_correction_model_exact_finite_mu_scalar=1.0` marker.  This
closes diagnostic scalar-QED routing through the current combined full-span
surface, but still does not implement anisotropic/tensor QED response, public
dispatch, production SMC validation, QKE, or promotion-grade full-BBN exact-QED
validation.
The same AP4/AP65 combined full-span gate and frozen/live source-policy span
profile now reuse a shared AP6 radial-grid cache across matched span rows and
policy rows.  This keeps the live Boltzmann source evaluation on-the-fly while
avoiding repeated construction of invariant descriptor/radial grids.  A
two-span AP65 comparison at `N_span=(0, 1e-14)` and `(0, 1e-12)`,
`live_rhs`, `N_q=3`, `N_mu=3`, `N_phi=5`, and `standard_3t_plasma` measured
`separate_cache_s=5.539025628997479`, `shared_cache_s=3.3420192470075563`,
`speedup=1.6573889076064183`, `shared_cache_entries=18`, and passing source
rows with `source_evaluations=7/7`.  This is CPU runtime/cache reuse only; it
does not pretabulate evolving distributions, alter the collision operator,
promote public dispatch, or change the no-QKE boundary.
The same AP4/AP65 combined full-span gate now requires AP6 conserved-moment
closure diagnostics from the actual combined-source payload.  A real
`live_rhs` two-span RK23 smoke over `(0, 1e-14)` and `(0, 1e-12)` passed with
all-nine diagonal `nu-nu` radial number projection enabled, `9` projected
number sources, `6` off-diagonal projected number sources, `6` off-diagonal
pair-energy projected sources, `3` unordered projected pairs,
`radial_nunu_max_abs_number_moment_final=1.0508502501873664e-20`, and
`radial_offdiagonal_nunu_pair_max_abs_energy_residual_final=7.707999820014133e-20`.
Rows missing the pair-energy closure marker now fail closed.  This upgrades the
diagnostic AP4 full-span surface to enforce the landed AP6 radial closure
contract, but still does not promote public dispatch, QKE, production SMC
validation, or a promotion-grade full-BBN span.
2026-05-17 AP4/AP65 physical-preview preset update: the same combined
full-span gate now exposes an opt-in `physical_preview` preset for longer
diagnostic evolution with frozen collision sources and a stiff SciPy solver.
The preset runs `N_span=(0, 1e-6)`, `(0, 1e-4)`, and `(0, 1e-3)` with
`source_update_policy=frozen_initial_state`, `method=Radau`,
`max_pstf_radial_source_evaluations=64`, and `max_nfev=200000`.  A real
isolated CLI run passed in `elapsed_s=29.111393796047196`; the three rows
reported
`T_gamma_final=0.7999992142768074`, `H_rate_s_final=0.4315478525579576`,
`Xn_final=0.13000006723216642`, and `nfev=22` at `1e-6`;
`T_gamma_final=0.7999214316323131`, `H_rate_s_final=0.4314627401457326`,
`Xn_final=0.1300112257070123`, and `nfev=420` at `1e-4`; and
`T_gamma_final=0.7992146754386976`, `H_rate_s_final=0.4306897631857426`,
`Xn_final=0.1927605271823484`, and `nfev=10634` at `1e-3`.  All rows retained
the AP6 off-diagonal `nu-nu` pair-energy closure residual
`9.571472303973594e-20`.  CLI overrides of preset-defining fields are relabeled
`custom`, and direct spec construction rejects mismatched `physical_preview`
contracts.  Artifact inputs and CLI dry-runs now expose
`routine_numeric_gate_spans=[(0,1e-6),(0,1e-4)]`,
`isolated_diagnostic_spans=[(0,1e-3)]`, and an explicit isolated-process
marker for that diagnostic long row.  The routine frozen-source Radau numeric
regression is now limited to the stable `1e-6` and `1e-4` rows because the
isolated `1e-3` frozen row was later found order-sensitive after source-refresh
solves in the same process.  A
`N_span=1e-2` frozen Radau probe returned a non-success result with overflow
warning and unusable terminal values, and a live-RHS probe at `N_span=1e-6` was
timeout-level in CPU smoke settings.  This is therefore a longer frozen-source
physical diagnostic, not a promoted live-RHS full-BBN gate.
2026-05-17 AP4/AP65 piecewise-frozen source-refresh update: the combined
full-span gate now supports `source_update_policy=piecewise_frozen` with
explicit `source_update_subspan_ends`.  This is a state-handoff operator-split
diagnostic: each subspan recomputes the AP41 angular plus AP6 `pstf_radial`
combined source from the current Sigma/A/T/X state, then integrates that
subspan with frozen source moments before handing the terminal state to the
next subspan.  A real custom CLI run over `N_span=(0, 1e-4)`,
`source_update_subspan_ends=(5e-5, 1e-4)`, `Radau`,
`max_pstf_radial_source_evaluations=8`, and `max_nfev=5000` passed with
`source_update_subspan_count=2`, `source_evaluations=2`,
`source_diagnostic_evaluations=1`, terminal source diagnostics at `N=1e-4`
after the last refresh at `N=5e-5`, `nfev=1397`,
`T_gamma_final=0.7999214320646801`,
`H_rate_s_final=0.43146274191619854`, `Xn_final=0.1300096609235235`,
`collision_dA_abs_max_final=0.00026527543966857903`, and AP6 pair-energy
residual `1.1784345878675453e-19`.  A longer nonuniform source-refresh run over
`N_span=(0, 1e-3)` with subspan ends `(1e-6, 1e-4, 1e-3)` also passed with
`source_update_subspan_count=3`, `source_evaluations=3`,
`source_diagnostic_evaluations=1`, terminal source diagnostics at `N=1e-3`
after the last refresh at `N=1e-4`, `nfev=47`,
`T_gamma_final=0.7992146796753832`, `H_rate_s_final=0.43068978083203713`,
`Xn_final=0.13005888045355307`,
`collision_dA_abs_max_final=0.000265067738217371`, and AP6 pair-energy
residual `7.326834993749698e-20`.  This removes the all-or-nothing jump from
fully frozen initial sources for this diagnostic span, but it remains
operator-split and is not live-RHS full-BBN or public dispatch evidence.
2026-05-17 AP4/AP65 piecewise physical-preview preset update: the nonuniform
source-refresh path now has a named `piecewise_physical_preview` preset rather
than only a custom CLI recipe.  It resolves to `N_span=(0,1e-4),(0,1e-3)`,
`source_update_policy=piecewise_frozen`,
`source_update_subspan_ends=(1e-6,1e-4,1e-3)`, `method=Radau`,
`max_pstf_radial_source_evaluations=8`, and `max_nfev=10000`, while artifact
inputs separate routine `N_span=(0,1e-4)` from isolated diagnostic
`N_span=(0,1e-3)` and declare supported electron-bath modes
`[fixed, charge_neutrality]` plus scalar-QED models
`[finite_mu_scaled, exact_finite_mu_scalar]`.  A real named-preset run passed with `span_count=2`,
`source_evaluation_max=3`, `radial_grid_cache_entries=45`, and no violations.
The `1e-4` row reported `T_gamma_final=0.7999214320680265`,
`H_rate_s_final=0.43146274191619854`, `Xn_final=0.13000591435767977`,
`nfev=31`, and AP6 pair-energy residual `8.205631676526035e-20`; the `1e-3`
row reported `T_gamma_final=0.7992146796753832`,
`H_rate_s_final=0.43068978083203713`, `Xn_final=0.13005888045355307`,
`nfev=47`, and AP6 pair-energy residual `7.326834993749698e-20`.  The same
named preset passed with `electron_chemical_potential_mode=charge_neutrality`
through `N_span=1e-3`, recording
`electron_chemical_potential_MeV_final=3.295370971985368e-10`,
`electron_charge_asymmetry_density_MeV3_final=6.59880533199606e-11`, and
`source_update_charge_asymmetry_state_handoff=1`.  With
`qed_correction_model=exact_finite_mu_scalar`, the preset also passed through
`N_span=1e-3` with `qed_correction_model_exact_finite_mu_scalar=1`,
`T_gamma_final=0.7992145483449656`, `H_rate_s_final=0.4302626189334139`,
`Xn_final=0.13005893883614722`, `nfev=47`, and AP6 pair-energy residual
`9.634999775017666e-20`.  The combined
`electron_chemical_potential_mode=charge_neutrality` plus
`qed_correction_model=exact_finite_mu_scalar` control row also passed through
`N_span=1e-3`, recording
`electron_chemical_potential_MeV_final=3.295370300138031e-10`,
`electron_charge_asymmetry_density_MeV3_final=6.598801808206643e-11`,
`qed_correction_model_exact_finite_mu_scalar=1`, and
`source_update_charge_asymmetry_state_handoff=1`.  This standardizes the
current source-refresh preview evidence, but does not promote public dispatch,
production SMC, QKE, or full-BBN support.
2026-05-17 AP4/AP65 piecewise refinement update: a deterministic diagnostic
artifact/CLI now compares the coarse source-refresh schedule
`(1e-6,1e-4,1e-3)` with the refined schedule `(1e-6,1e-5,1e-4,1e-3)` over the
same `N_span=(0,1e-3)` combined-source solve using a shared AP6 radial-grid
cache.  A real run passed with `schedule_count=2`, `source_evaluation_max=4`,
`nfev_max=56`, `radial_grid_cache_entries=72`, and no violations.  The
refined-minus-coarse deltas were `T_gamma_final=-1.1280976153216216e-12`,
`Xn_final=2.3314683517128287e-15`,
`collision_dA_abs_max_final=4.4915406029882865e-14`, and AP6 pair-energy
residual delta `2.0117032497289633e-21`.  This is source-refresh refinement
evidence only; continuous live-RHS collision coupling and full-BBN promotion
remain blocked.
2026-05-17 AP4/AP65 terminal scalar-QED forwarding update: the
`piecewise_frozen` terminal source re-evaluation now forwards
`qed_correction_model` and records finite/exact scalar-QED one-hot diagnostics
on the returned combined source.  This closes a diagnostic consistency gap for
charge-neutral plus `exact_finite_mu_scalar` rows; a real named-preset
charge-neutral exact-QED run still passed with `span_count=2`,
`source_evaluation_max=3`, `electron_chemical_potential_abs_max=3.298436883301101e-10`,
`electron_charge_asymmetry_density_abs_max_MeV3=6.618704871052225e-11`, and no
violations.  This does not add anisotropic/tensor QED, QKE, public dispatch, or
promotion-grade full-BBN support.
2026-05-17 AP4/AP65 charge-neutral piecewise handoff update: the same
`piecewise_frozen` source-refresh path now supports
`electron_chemical_potential_mode="charge_neutrality"` by carrying the evolved
finite-mass e-/e+ charge-asymmetry density from one subspan into the next
subspan's source construction and Hubble/RHS initialization.  The radial
electron/positron bath therefore uses the handoff charge state rather than
falling back to a fixed or algebraically recomputed initial value at each
source refresh.  A real charge-neutral run over `N_span=(0, 1e-4)` with
subspan ends `(5e-5, 1e-4)`, `Radau`, `max_pstf_radial_source_evaluations=8`,
and `max_nfev=10000` passed with `source_update_subspan_count=2`,
`source_update_charge_asymmetry_state_handoff=1`,
`source_evaluations=2`, `source_diagnostic_evaluations=1`, `nfev=8256`,
`electron_chemical_potential_MeV_final=3.298132792363573e-10`,
`electron_charge_asymmetry_density_MeV3_final=6.61672994731945e-11`,
`T_gamma_final=0.7999214313698554`,
`H_rate_s_final=0.43146274153219083`, `Xn_final=0.13000851100280264`,
`collision_dA_abs_max_final=0.00026527544839372085`, and AP6 pair-energy
residual `1.1784345878675453e-19`.  This closes the fixed-only piecewise
handoff limitation for the current diagnostic span; it is still
operator-split, smoke-scale, and not a promotion-grade live-RHS full-BBN
claim.

2026-05-17 AP66 piecewise publication-matrix update: the AP66
publication-candidate convergence matrix can now consume the AP4/AP65
`piecewise_frozen` combined-source gate instead of being restricted to AP65
direct frozen/live source-policy rows.  `piecewise_frozen` is accepted only for
`source_model="combined_angular_pstf_radial"` and records
`source_policy_piecewise_frozen`, `source_update_subspan_count`,
`source_update_subspan_max_length`,
`source_update_charge_asymmetry_state_handoff`, and
`source_diagnostic_evaluations` in the same q/`N_mu`/`N_phi` report contract.
AP4 gate specs now carry `Xn0` and `weak_rate_mode` into the actual solve rows
plus radial momentum-delta model/sigma controls into the actual solve rows so
AP66 piecewise rows do not silently fall back to hard-coded neutron fraction,
weak-rate mode, or radial closure settings.  A real AP66 q-ladder smoke over
`N_span=(0, 1e-14)`, `source_update_subspan_ends=(5e-15, 1e-14)`,
`metadata_only`, `RK23`, and `standard_3t_plasma` passed with converged
`N_q=(3,4)`, `source_update_subspan_count=2`,
`pstf_radial_source_evaluations=2`, `collision_dA_abs_max_final>0`,
positive `T_gamma_final`, and physical `0<=Xn_final<=1`.  A full short-span
AP66 piecewise matrix over q/`N_mu`/`N_phi` also produced candidate
`(N_q,N_mu,N_phi)=(4,4,6)` with
`collision_dA_abs_max_final=0.0010936512219095985`,
`T_gamma_final=0.7999999999999923`, and
`Xn_final=0.13000000000019393`.  This promotes the AP4 operator-split
source-refresh diagnostic into the publication-candidate evidence surface but
still does not claim live-RHS full-BBN, public dispatch, production SMC, or
QKE.

2026-05-17 AP67 nested AP66 piecewise-evidence update: the validation atlas
now preserves AP66 source-refresh provenance instead of collapsing nested
convergence evidence to source-model/radial/QED metadata only.  AP67
`reused_evidence_links["AP66"]` records AP66 `source_update_policy`,
`source_update_policies`, `source_update_subspan_ends`, and candidate
`source_update_policy`/`source_update_subspan_ends` for AP4-backed
`piecewise_frozen` publication-matrix rows.  The AP67 CLI also accepts
`--source-update-policy` and `--source-update-subspan-ends`, forwarding those
values only to the nested AP66 convergence matrix while keeping AP67 direct
atlas cases on their existing frozen/live source-policy contract.  This makes
AP67 a faithful consumer of AP66 piecewise source-refresh evidence without
promoting `piecewise_frozen` to live-RHS full-BBN, public dispatch, production
SMC, or QKE.

2026-05-15 AP66 exact-scalar-QED matrix update: the publication-candidate
convergence matrix now accepts, validates, records, and forwards
`qed_correction_model` into every AP65 q/`N_mu`/`N_phi` solve row.  Matrix
inputs, matrix-case settings, first-candidate settings, CLI summaries, and
ladder observables now distinguish `finite_mu_scaled` from
`exact_finite_mu_scalar` with one-hot row markers.  This makes the already
landed scalar 3T QED thermo mode executable from the AP66 publication-candidate
surface, but does not add anisotropic/tensor QED response, public dispatch,
production SMC validation, QKE, or promotion-grade full-BBN exact-QED
validation.

2026-05-15 AP6 standard-3T radial energy-closure update: the AP6
`pstf_radial` AP18 thermo-source bridge now exposes
`energy_normalization="standard_3t_plasma"`.  That mode keeps the raw radial
`dA_modes` hierarchy payload, then applies a number-neutral monopole correction
to the electromagnetic radial source so `dQ_nue_pair_N` and `dQ_nux_bank_N`
match the canonical 3T plasma-transfer table per e-fold at the current
`T_gamma`, `T_nu_e`, `T_nu_x`, and Hubble rate.  Direct source-only and
nonlinear non-LRS frozen-source wrappers seed that source with the actual
initial 3T Hubble rate instead of a unit placeholder.  This closes a concrete
radial thermo-normalization gap while preserving opt-in/staged/no-QKE status.
The non-LRS direct artifact CLI and LRS/non-LRS collision-feedback artifact
CLIs now expose that normalization mode and explicit 3T initial temperatures.
A smoke command with `T_gamma0=0.8 MeV`, `T_nu_e0=0.79 MeV`,
`T_nu_x0=0.78 MeV`, and `N_span_end=1e-14` reported
`radial_em_energy_target_nue_pair_N=0.002439621160259491`,
`radial_em_energy_target_nux_bank_N=0.0019820818977550445`, and maximum
closure residual `8.673617379884035e-19`.
The AP6 LRS live-RHS budget/source-policy artifacts and the AP45 non-LRS
source-policy artifact now forward the same mode, so frozen-vs-live radial
policy rows can be generated under canonical 3T electromagnetic
energy-closure rather than falling back to raw radial thermo moments only.

2026-05-21 BD12 continuous AP65 Rodas linear-system reuse: the private FB69
host-stepped Rodas5P loop now factorizes `W = I / (gamma h) - J` once per host
step and reuses that factorization for all stage right-hand sides.  FB69 rows
and FB70 summaries record linear-system backend, factorization counts, and
solve counts.  The three-window non-LRS+collision CPU-JAX smoke with
`stage_collision_payload_policy=step_base_reuse` passed all hot-endpoint rows,
recorded `480` factorized linear systems for `3838` stage solves, and reached
terminal `T_gamma=0.09515938503015112 MeV`.  This is a private hot-loop solver
optimization only; public dispatch, production SMC validation, QKE, and
full-BBN readiness remain unclaimed.

2026-05-21 BD13 continuous AP65 chained h-max policy: FB70 now exposes the
private opt-in `chain_h_max_policy=first_rejection_half_ceiling`, which uses the
first rejected step in a successful chained window to cap later-window `h_max`.
The three-window non-LRS+collision CPU-JAX smoke with base `h_max=0.1`,
`stage_collision_payload_policy=step_base_reuse`, and the BD12 linear-solve
reuse passed all hot-endpoint rows, capped two later rows, reduced selected
source evaluations to `521`, removed the `[0.8, 1.6]` rejection thrash, and
reached terminal `T_gamma=0.09515938504584408 MeV`.  This is still private
solver-policy evidence above the full-BBN endpoint; public dispatch,
production SMC validation, QKE, and full-BBN readiness remain unclaimed.

2026-05-21 BD14 continuous AP65 once-capped full-BBN endpoint: FB70 now also
exposes the private opt-in
`chain_h_max_policy=first_rejection_half_ceiling_once`, which preserves the
first rejection-derived chain cap but avoids late-window over-tightening.  The
six-window non-LRS+collision CPU-JAX/Rodas5P smoke with base `h_max=0.1`,
`stage_collision_payload_policy=step_base_reuse`, `rhs_trace_policy=boundary`,
and `N_span_end_ladder=(0.8,1.6,2.4,3.2,4.0,4.8)` passed with
`physical_full_bbn_span_ready=true`, terminal
`T_gamma=0.009144759664500419 MeV`, `selected_source_evaluations_total=3313`,
and `selected_wall_seconds_total=52.717273119837046`.  This moves the private
continuous-AP65 endpoint blocker below `0.01 MeV`, while public dispatch,
production SMC validation, QKE, and publication-ready claims remain unclaimed;
the next blockers are weak-rate/convergence/statistics evidence and the
dominant `[2.4,3.2]` hot-loop cost.

2026-05-21 BD15 endpoint-backed weak/control pair indexing: FB70 subset
freedom-composition runs now keep missing single-reference comparisons as
metadata instead of treating deliberately scoped pairs as row failures, and
FB71 now accepts FB70 freedom-composition artifacts plus scoped
`required_contexts` while preserving its FB51/FB52 default behavior.  The
non-LRS+collision endpoint pair over the BD14 six-window CPU-JAX/Rodas5P ladder
with `weak_correction_level=3` passed both control and weak rows below
`0.01 MeV`; FB71 reported `full_bbn_weak_rate_pairs_ready=true`,
`max_abs_weak_delta_Yp=0.0016890594392898195`, and
`max_abs_weak_delta_DH=7.310612378372874e-08`, with
`ap80_to_full_bbn_bridge_ready=false`.  This is private scoped weak/control
evidence only; AP80 bridge evidence, resolution/tolerance ladders, public
dispatch, production SMC validation, QKE, and publication-ready claims remain
unclaimed.

2026-05-21 BD16 scoped AP80-FB71 bridge: FB72 now forwards scoped
`required_contexts` into its nested FB71 build and accepts FB70
freedom-composition sources through the compatibility
`progressive_freedom_artifact` input.  A real bridge probe combined AP80 smoke
`smoke_q34` (`total_nfev=7596`,
`applied_rate_q_relative_delta=0.0024445680701901517`) with the BD15
non-LRS+collision endpoint pair; FB71 reported
`ap80_to_full_bbn_bridge_ready=true`, and FB72 reported
`ap80_fb71_bridge_ready=true`, while retaining
`fb71_max_abs_weak_delta_Yp=0.0016890594392898195` and
`fb71_max_abs_weak_delta_DH=7.310612378372874e-08`.  FB72 marks the bridge
`fb71_required_context_scope=scoped_subset` and keeps the next blocker at
scoped-to-default matrix or resolution/tolerance extension; FB73 and FB75
reject scoped FB72 bridges by default, and FB74 rejects scoped FB73 figure
manifests by default.  This is private scoped bridge evidence only; AP80 remains
profile-level, and resolution/tolerance ladders, publication figures,
statistics, public dispatch, production SMC validation, QKE, and
publication-ready claims remain unclaimed.

2026-05-21 BD35 default AP80-FB71 bridge baseline: FB70's default
freedom-composition matrix now includes an explicit empty-freedom
`lrs_no_collision` baseline row while keeping direct `enabled_freedoms=()`
rejected.  A real CPU-JAX/Rodas5P default eight-case FB70 smoke passed with
`physical_full_bbn_span_ready=true`, `rows_full_bbn_completed=8`,
`baseline_freedom_rows=1`, `failed_or_exception_rows=0`,
`T_final_MeV_min=0.009139193835665199`, and
`T_final_MeV_max=0.009144759664527284`.  Feeding that artifact into the
existing FB72 bridge without scoped `required_contexts` passed with
`fb71_required_context_scope=default_all_contexts`,
`fb71_required_pair_count=4`, `fb71_passed_pair_count=4`, and
`fb71_rows_reaching_full_bbn_endpoint=8`.  BD36 supersedes the bridge artifact's
older diagnostic figure-input handoff by keeping the next blocker on
q/angular-grid convergence and hot-loop cost.
This folds the missing default-matrix control row into existing FB70/FB72
contracts rather than adding a new gate.  It remains private diagnostic
evidence only: no public dispatch, production SMC validation, QKE, or
publication-ready support is claimed.

2026-05-21 BD36 default-matrix resolution/tolerance ladder: FB70's existing
`resolution_ladder_cases` mode now accepts nested `freedom_composition_cases`
and compares matching `freedom_key` terminal rows across adjacent resolution
cases.  A real CPU-JAX/Rodas5P default eight-row matrix tolerance smoke over
`rtol/atol=(1e-8,1e-10)` versus `(5e-9,5e-11)` passed with
`resolution_tolerance_ready=true`,
`composition_resolution_tolerance_ready=true`,
`nested_freedom_composition_case_count=16`,
`composition_resolution_comparison_count=8`,
`composition_resolution_delta_violations=[]`,
`max_abs_composition_delta_Yp=2.118909769865951e-09`,
`max_abs_composition_delta_DH=1.366312742639823e-10`,
`max_abs_composition_delta_T_final_MeV=8.73974156934132e-12`,
`selected_source_evaluations_total=56728`, and
`selected_wall_seconds_total=135.17470189894084`.  The next blocker remains
q/angular-grid convergence beyond this same-q tolerance smoke plus hot-loop
cost, not figure plumbing.  This remains private smoke-scale diagnostic
evidence: no public dispatch, production SMC validation, QKE, or
publication-ready support is claimed.

2026-05-21 BD37 default-matrix q/angular axis classification: FB70's existing
`resolution_ladder_cases` comparisons now record `axis_delta_kinds` and summary
counts for q-grid and angular-grid deltas.  A real CPU-JAX/Rodas5P default
eight-row matrix smoke comparing q3/N_mu3/N_phi5 to q4/N_mu4/N_phi6 passed with
`resolution_axis_delta_kinds=["angular_grid","q_grid"]`,
`composition_resolution_axis_delta_kinds=["angular_grid","q_grid"]`,
`composition_q_grid_comparison_count=8`,
`composition_angular_grid_comparison_count=8`,
`composition_resolution_delta_violations=[]`,
`max_abs_composition_delta_Yp=4.220789247222356e-06`,
`max_abs_composition_delta_DH=2.246835889453539e-08`,
`selected_source_evaluations_total=51265`, and
`selected_wall_seconds_total=169.38022271264344`.  The next blocker is now
`reduce_hot_loop_payload_jacobian_cost_or_extend_grid_ladder`: one q/angular
smoke is clear, but broader grid ladders, statistics, public dispatch,
production SMC validation, QKE, and publication-ready support remain unclaimed.

2026-05-21 BD38 internal base RHS metadata bypass: FB69/FB70 now keep endpoint
metadata on `rhs_initial`/`rhs_final` while requesting RHS-only evaluation for
internal `jacobian_base` calls, and rejected-attempt cache hits reuse cached
RHS/Jacobian arrays without copying.  One observed CPU-JAX/Rodas5P FB69
`N_span=(0,0.8)` smoke with boundary traces and `step_base_reuse` moved from
`wall_seconds_total=7.293221939005889` to `7.103321349015459`, with
`source_evaluations_total=128` and `structured_jacobian_evaluations_total=15`
unchanged while `rhs_only_jax_evaluations_total` rose from `112` to `126`.
This is private runtime work with single-smoke timing evidence on existing
surfaces only; public dispatch, production SMC validation, QKE, and
publication-ready support remain unclaimed.

2026-05-21 BD39 reused stage payload no-copy: FB69 now passes the host-step
base payload/provenance mapping directly into `step_base_reuse` stage RHS calls
instead of copying it with `dict(...)` for every Rodas stage.  On the same
single observed CPU-JAX/Rodas5P `N_span=(0,0.8)` smoke, the BD39 artifact
`/tmp/fb69_bd39_after_payload_override_nocopy.json` passed with
`wall_seconds_total=7.030492526013404`,
`source_evaluations_total=128`, `rhs_only_jax_evaluations_total=126`,
`structured_jacobian_evaluations_total=15`, and
`stage_collision_payload_reuse_total=112`.  This is private hot-loop cleanup
only; public dispatch, production SMC validation, QKE, and publication-ready
support remain unclaimed.

2026-05-21 BD40 LU finite-check bypass: FB69's host Rodas5P SciPy LU path now
calls `lu_factor`/`lu_solve` with `check_finite=False` while preserving
explicit non-finite linear-system matrix/RHS rejection in FB69.  On the same
single observed CPU-JAX/Rodas5P `N_span=(0,0.8)` smoke,
`/tmp/fb69_bd40_after_lu_no_check_finite.json` passed with
`wall_seconds_total=6.980848156963475`,
`source_evaluations_total=128`, `rhs_only_jax_evaluations_total=126`,
`structured_jacobian_evaluations_total=15`, and
`stage_collision_payload_reuse_total=112`.  This is private solver hot-loop
work only; public dispatch, production SMC validation, QKE, and
publication-ready support remain unclaimed.

2026-05-21 BD41 LU overwrite buffer hints: FB69 now passes SciPy overwrite hints
for the freshly formed LU matrix and per-stage RHS arrays while keeping the BD40
explicit non-finite matrix/RHS checks.  On the same single observed
CPU-JAX/Rodas5P `N_span=(0,0.8)` smoke,
`/tmp/fb69_bd41_after_lu_overwrite.json` passed with
`wall_seconds_total=7.060200556064956`,
`source_evaluations_total=128`, `rhs_only_jax_evaluations_total=126`,
`structured_jacobian_evaluations_total=15`, and
`stage_collision_payload_reuse_total=112`.  This is private copy-pressure
cleanup only, not timing-improvement evidence; public dispatch, production SMC
validation, QKE, and publication-ready support remain unclaimed.

2026-05-21 BD42 frozen-source RHS/Jacobian coevaluation: FB69 now computes the
accepted host-step base RHS and frozen-source dense Jacobian through one
`jax.jacfwd(..., has_aux=True)` function inside the existing
`frozen_source_jax` policy.  On the same single observed CPU-JAX/Rodas5P
`N_span=(0,0.8)` smoke,
`/tmp/fb69_bd42_after_combined_only_jacfwd_aux.json` passed with
`wall_seconds_total=6.956021819030866`,
`rhs_only_jax_evaluations_total=112`,
`frozen_source_jax_rhs_and_jacobian_evaluations_total=15`,
`source_evaluations_total=128`, `structured_jacobian_evaluations_total=15`,
and `stage_collision_payload_reuse_total=112`.  This is private hot-loop
optimization only; public dispatch, production SMC validation, QKE, and
publication-ready support remain unclaimed.

2026-05-21 BD43 AP6 radial grid cache-dir threading: FB69/FB70 now forward an
opt-in `pstf_radial_grid_cache_dir` source-refresh setting through non-LRS S2 and
LRS dynamic AP65 payload rebuilds, exposing the existing AP6 NPZ radial-grid
cache to repeated CPU-JAX/Rodas5P runs.  This does not quantize, truncate, or
drop temperature-dependent finite-mass radial-grid keys.  On the same
`N_span=(0,0.8)` FB69 smoke, the no-cache comparison
`/tmp/fb69_bd43_nocache_compare.json` passed at
`wall_seconds_total=7.262460570083931`, the first cache population run passed at
`7.948269885964692`, and the repeated process using the populated cache
`/tmp/fb69_bd43_cache_second.json` passed at `6.026476241997443`.  Counts stayed
fixed: `source_evaluations_total=128`,
`dynamic_collision_payload_builds_total=16`, `rhs_only_jax_evaluations_total=112`,
`frozen_source_jax_rhs_and_jacobian_evaluations_total=15`, and
`stage_collision_payload_reuse_total=112`.  This is private repeated-process
runtime evidence only; public dispatch, production SMC validation, QKE, and
publication-ready support remain unclaimed.

2026-05-21 BD44 JAX persistent compilation cache control: FB69/FB70 now expose
an opt-in `--jax-compilation-cache-dir` private CLI control plus shared
`augmented_validation_utils.py` helpers for recording and applying JAX persistent
compilation-cache settings.  Dry runs and real artifacts include a
`runtime_cache` block with cache-hit status explicitly marked `not_measured`,
and real runs configure the cache before the artifact builder starts.  On the same
`N_span=(0,0.8)` FB69 smoke with BD43's radial-grid cache already populated, the
first JAX-cache population process `/tmp/fb69_bd44_jaxcache_metadata_first.json` passed at
`wall_seconds_total=6.252862777910195`; the next separate process
`/tmp/fb69_bd44_jaxcache_metadata_second.json` passed at `3.1601327889366075`.  Counts
stayed fixed: `source_evaluations_total=128`,
`dynamic_collision_payload_builds_total=16`, `rhs_only_jax_evaluations_total=112`,
`frozen_source_jax_rhs_and_jacobian_evaluations_total=15`, and
`stage_collision_payload_reuse_total=112`.  This folds repeated-run backend
setup into existing private surfaces only; it is not a new readiness gate and
does not change AP65 physics, raw output handling, public dispatch, production
SMC validation, QKE scope, or publication-ready support.

2026-05-21 BD45 consolidated span cache-control propagation: the consolidated
FB79/FB80/FB81/FB88/FB89 span CLI now forwards the existing AP6
`--source-refresh-radial-grid-cache-dir` setting and the BD44
`--jax-compilation-cache-dir` runtime cache controls through nested FB70/FB88
runners.  Span dry runs and span artifacts record `runtime_cache` with cache-hit
status explicitly `not_measured`, and real CLI execution configures JAX before
calling the existing writer.  Focused regressions cover legacy and consolidated
dry-run metadata, nested builder propagation, and real-writer ordering.  This
keeps the older span wrappers folded into the existing consolidated surface and
moves repeated-run backend setup for larger private profiles; it does not add a
standalone gate, change AP65 physics, hide raw outputs, open public dispatch,
claim production SMC validation, add QKE, or provide publication-ready support.

2026-05-21 BD46 consolidated span endpoint-target mode: the same consolidated
FB88/FB89 span surface now accepts opt-in `target_T_gamma_MeV`, so endpoint
growth can stay on the folded runner instead of spawning another gate.  With no
target supplied, FB88/FB89 preserve their old hot-span scope and reject nested
endpoint-ready rows as out of scope.  With `target_T_gamma_MeV=0.01`, nested
FB70/FB88 endpoint rows are allowed and classified as endpoint-target evidence
while public dispatch, production SMC validation, publication readiness, and
QKE remain closed.  Targets above `0.01 MeV` are rejected, and failed or
negative-observable endpoint rows keep top-level pass closed.  The small real
CPU-JAX/Rodas5P target-mode smoke
`/tmp/fb89_bd46_target_mode_smoke.json` failed closed as hot evidence with
`artifact_payload_sha256=e166fcfb27df5648fd4ed880f24399efe79cdbd54d02c8caa9da82d64d1abfcb`,
`classification=trace_span_growth_endpoint_target_not_reached`,
`endpoint_target_reached=false`, `violations=[fb89_endpoint_target_not_reached]`,
`best_T_final_MeV=0.7999999921428074`, `source_evaluations_total=81`, and
conservation max `6.284872348663924e-18`.  This removes the old surface-level
endpoint rejection but does not claim endpoint completion; the next blocker is
driving the target-mode ladder to `T_gamma <= 0.01 MeV` and then extending the
raw endpoint history into resolution, weak-rate, and statistical evidence.

2026-05-21 BD47 endpoint-target row budget and early stop: FB70 now accepts
private `stop_at_T_gamma_MeV` metadata and stops the ladder only after a clean
row reaches the requested temperature.  Failed or negative-observable rows are
preserved and prevent early-stop success, so raw bad states are not hidden.
FB88 forwards endpoint targets into that FB70 stop control, and FB89 geometric
mode now accepts `target_max_span_rows` to extend the existing multiplicative
ladder budget without hand-writing another explicit ladder or adding a new
gate.  Artifacts record requested/executed span rows, `early_stop_reason`,
`target_stop_T_gamma_MeV`, `target_max_span_rows`, and
`target_effective_span_rows`.  This folds the endpoint-growth optimization into
the consolidated span runner; it still does not claim public dispatch,
production SMC validation, QKE, publication readiness, or endpoint completion
unless raw rows cleanly reach the target.

2026-05-21 BD48 freedom-subset span runner threading: FB70's existing
`enabled_freedoms` control is now exposed through the consolidated
FB79/FB80/FB81/FB88/FB89 span runner and CLI.  Focused regressions cover
nested builder forwarding and dry-run metadata.  A real weak-rate-only
CPU-JAX/Rodas5P FB89 endpoint smoke reached the private endpoint target with
`artifact_payload_sha256=61eec8da223b8053bc99a1c3c02d822f5b7383f5d89fbd400f39da06aaa6bbd5`,
`classification=trace_span_growth_endpoint_target_reached`,
`best_T_final_MeV=0.00960777013815278`, `rows_reaching_endpoint=1`, and
`source_evaluations_total=3364`.  The result is weak-only private endpoint
evidence and does not claim non-LRS, collision-term, all-freedom, public
dispatch, production SMC validation, QKE, or publication-ready support.

2026-05-21 BD49 freedom endpoint matrix: FB89 now accepts
`freedom_subset_cases` so singleton, pairwise, or later all-freedom endpoint
comparisons can run on the existing consolidated trace-span growth surface
rather than through another standalone gate.  Each matrix case embeds its
nested FB89 artifact and raw nested rows, records case-local violations, and
keeps the parent fail-closed when any case fails or misses the endpoint.  The
CLI exposes the same axis via `--freedom-subset-cases` on the consolidated span
experiment runner.  A real singleton CPU-JAX/Rodas5P endpoint matrix over
weak-rate corrections only, non-LRS geometry only, and neutrino collision terms
only passed with
`artifact_payload_sha256=77adadce64d9726da8fe142f42d6095dbef4de2b1adbaf85182e5eebc3ae91db`,
`classification=trace_span_growth_freedom_endpoint_matrix_passed`,
`endpoint_cases_reached=3`, `freedom_cases_failed=0`,
`best_T_final_MeV=0.009607770331335521`,
`largest_passing_N_span_end=4.75`, and `source_evaluations_total=10099`.
This is private singleton endpoint evidence only; pairwise/all-freedom
composition, q/angular-grid convergence, public dispatch, production SMC
validation, QKE, and publication-ready support remain unclaimed.

2026-05-21 BD50 resolution endpoint stop and weak-level guard: FB70
`resolution_ladder_cases` now forwards `stop_at_T_gamma_MeV` into nested span
ladders, records the target in parent and case inputs, and lets resolution
probes stop after a clean endpoint row without truncating failed or
negative-observable rows.  The FB70 CLI exposes this through
`--stop-at-T-gamma-MeV`.  FB89 freedom matrices now require positive
`weak_correction_level` whenever a case includes `weak_rate_corrections`, so
CL0 selector plumbing cannot be reported as weak-rate correction evidence.  A
weak-level-3 endpoint matrix over weak-only, weak+non-LRS, weak+collision, and
all-freedom cases passed with
`artifact_payload_sha256=f38b5979fc54e203e90e95dd943c4a8abbc6357212201aeb2d0c643d33ccad2a`,
`endpoint_cases_reached=4`, `best_T_final_MeV=0.009607770336848207`, and
`source_evaluations_total=13414`.  The scoped all-freedom solver-tolerance
resolution smoke passed with
`artifact_payload_sha256=d109ce304c985312b4557bb8a73a981ae746290f937057db98727d13e4b5af62`,
`physical_full_bbn_span_ready=true`, `resolution_tolerance_ready=true`, both
nested cases truncating `3` requested rows to `2`, and adjacent terminal deltas
`Yp=2.107925639593944e-09`, `D/H=6.259600207486897e-11`, and
`T_gamma=1.3748550908854185e-13`.  This remains private endpoint resolution
evidence and runtime-row reduction only; q/angular convergence, public
dispatch, production SMC validation, QKE, and publication-ready support remain
unclaimed.

2026-05-21 BD51 seeded frozen-source Jacobian replay: FB69 now uses the
existing `rhs_initial` boundary RHS seed with a Jacobian-only
`frozen_source_jax` replay on the first host step, instead of running the
combined RHS+Jacobian JAX function and discarding its RHS output.  Unseeded host
steps keep the combined replay path.  The focused one-row FB69 before/after
smoke moved `frozen_source_jax_rhs_and_jacobian_evaluations_total` from `1` to
`0` while preserving `source_evaluations_total=9`,
`dynamic_collision_payload_builds_total=2`, and
`initial_boundary_rhs_seed_reuse_total=1`.  The six-window all-freedom
weak-level-3 endpoint smoke passed with
`artifact_payload_sha256=4b063a34c589f56e04322f20154b2e69ec7ca30ec354b941607a0eae73e4a984`,
`physical_full_bbn_span_ready=true`,
`T_final_MeV_min=0.00914475966407264`,
`selected_initial_boundary_rhs_seed_reuse_total=6`,
`selected_frozen_source_jax_jacobian_evaluations_total=390`, and
`selected_frozen_source_jax_rhs_and_jacobian_evaluations_total=384`.  This is
private CPU-JAX/Rodas5P hot-loop work removal on the existing runtime surface,
not a public dispatch, production SMC validation, QKE, q/angular convergence,
raw-output repair, or publication-ready support claim.

2026-05-21 BD52 exact byte retry cache keys: FB69 now uses an exact contiguous
float-state byte key for the internal host-step retry cache instead of the
JSON-safe trace fingerprint used in artifact rows.  A focused 20,000-call
microbenchmark over a 41-component state measured the old JSON fingerprint at
`0.8712652230169624 s` and the byte key at `0.06128845305647701 s`
(`14.215813576075998x` local function-level speedup).  The FB69 one-row
CPU-JAX/Rodas5P smoke passed with
`artifact_payload_sha256=b62cf848c121eb23009141f5c64cf06085f11730d0bda0b26f02216f4b5259ac`,
`source_evaluations_total=9`, `dynamic_collision_payload_builds_total=2`,
`initial_boundary_rhs_seed_reuse_total=1`, and
`trace_rows_emitted_total=2`.  This is private solver retry-cache overhead
reduction only; raw trace provenance, physics, public dispatch status,
production SMC status, QKE scope, and publication readiness are unchanged.

2026-05-21 BD53 host-slice current restart payloads: FB69 now builds hot-loop
current-state restart kwargs directly from the replay layout's host NumPy
slices instead of calling the JAX final-state restart serializer and moving
state fields through device-get conversions.  The emitted mapping keeps the
same restart contract, layout payload, raw `initial_A_modes`, raw `X0`, and
`source_N` fields consumed by the AP65 dynamic payload builder.  A focused
20,000-call microbenchmark over the 41-component FB69 smoke state measured the
previous JAX serializer path at `15.329722063965164 s` and the host-slice path
at `1.2295497520826757 s` (`12.467752555761878x` local function-level speedup).
The FB69 one-row CPU-JAX/Rodas5P smoke passed with
`artifact_payload_sha256=3f2946a8b76e32d479584eea39bb06d83696bc58d23a8c6fb99086c2251939a6`,
`source_evaluations_total=9`, `current_restart_payload_builds_total=2`,
`dynamic_collision_payload_builds_total=2`, and `trace_rows_emitted_total=2`.
This is private hot-loop conversion overhead reduction only; raw state
preservation, negative-output fail-closed behavior, public dispatch status,
production SMC status, QKE scope, and publication readiness are unchanged.

2026-05-21 BD54 hot-loop minimal payload metadata: FB69 now keeps full dynamic
collision-payload metadata for boundary `rhs_initial`/`rhs_final` provenance
but passes `payload_metadata_policy="hot_loop_minimal"` into suppressed
inner-loop RHS/Jacobian dynamic payload builds.  The replay payload builder
uses that policy to skip restart-state, refresh-config, q-grid, and diagnostics
fingerprint construction while preserving the actual collision increments and
raw `dA_modes`.  A local 5,000-call fake-source benchmark measured full
metadata at `1.2704691931139678 s` versus hot-loop minimal metadata at
`0.5435653830645606 s` (`2.3372886366515933x` local function-level speedup),
and focused FB69/replay tests passed with `83 passed`.  This is existing
private CPU-JAX/Rodas5P hot-loop metadata cost reduction, not a public dispatch,
production SMC, QKE, publication-ready, or output-truncation claim.  The next
blocker remains the continuous-span `[2.4,3.2]` nonfinite endpoint failure plus
remaining payload/Jacobian cost.

2026-05-21 BD55 FB70 max-step budget retry: FB70 now accepts
`max_step_retry_factors` on the existing full-BBN span ladder and CLI.  Failed
row attempts caused by a low `max_steps` budget remain embedded in
`h_refinement_attempts`, while the row can retry with a larger effective
`max_steps` and continue chain restart handoff if the retry passes.  The
baseline retry artifact recovered the previously blocked `[2.4,3.2]` window
from `max_steps=128` to `max_steps=1024`, then reached
`T_final_MeV=0.009139193822551734` in the extended run.  The existing
eight-case freedom-composition artifact
`diagnostic_outputs/bd55_blocker_debug/fb70_case_matrix_maxstep_retry.json`
reported `physical_full_bbn_span_ready=true`, `rows_full_bbn_completed=8`,
`failed_or_exception_rows=0`, `all_freedom_full_bbn_ready=true`, and
`artifact_payload_sha256=986c64c2296de5115ab44a6f93e233dc423cf7915b63249b0803cbfb0c013cf4`.
This folds solver-budget recovery into FB70 rather than adding a gate, keeps
raw failed attempts visible, and does not alter public dispatch, production
SMC, QKE, publication-readiness, or negative-output fail-closed boundaries.
The tiny-grid endpoint-below-`0.01 MeV` blocker is now cleared; remaining
blockers are q/angular and tolerance convergence at stronger grids, hot-loop
AP65/Jacobian/Rodas cost, and statistical-pipeline evidence.

2026-05-21 BD17 continuous AP65 endpoint tolerance ladder: FB70 now includes an
in-surface `resolution_ladder_cases` mode instead of a standalone gate.  Each
case runs a nested raw FB70 span ladder with case-local grid/tolerance/solver
policy settings, embeds the raw span rows, and compares adjacent terminal
`Y_p`, D/H, and `T_gamma` deltas while keeping public dispatch, production SMC,
and QKE closed.  The real non-LRS+collision CPU-JAX/Rodas5P endpoint probe over
the BD14 six-window ladder compared `rtol/atol=(1e-8,1e-10)` with
`(5e-9,5e-11)` and reported `passed=true`,
`physical_full_bbn_span_ready=true`, `resolution_tolerance_ready=true`,
`max_abs_delta_Yp=3.2406321515132674e-09`,
`max_abs_delta_DH=5.394384165228962e-11`,
`max_abs_delta_T_final_MeV=8.066776413517829e-13`,
`selected_wall_seconds_total=107.78340663993731`,
`selected_source_evaluations_total=7233`,
`selected_dynamic_collision_payload_builds_total=905`, and
`selected_frozen_source_jax_jacobian_evaluations_total=881`.  This is private
scoped endpoint tolerance evidence only; q-grid/angular-grid convergence,
statistics, publication figures, public dispatch, production SMC validation,
QKE, and publication-ready claims remain unclaimed, and the endpoint hot-loop
payload/Jacobian cost remains the next runtime blocker.
2026-05-21 BD18 initial-boundary RHS seed reuse: FB69 now reuses the recorded
`rhs_initial` boundary RHS as the first host-step base, eliminating the
immediate duplicate same-state `jacobian_base` payload/RHS evaluation inside
each existing FB69/FB70 row.  FB69 records
`initial_boundary_rhs_seed_reuse_count`, and FB70 propagates selected and
h-refinement seed-reuse totals through normal span and resolution summaries.
A one-step FB69 boundary-trace probe reports payload calls `2`,
`source_evaluation_count=9`, `dynamic_collision_payload_build_count=2`, and
`initial_boundary_rhs_seed_reuse_count=1`.  The same six-window
non-LRS+collision endpoint run reports `passed=true`,
`physical_full_bbn_span_ready=true`, terminal
`T_gamma=0.009144759664502618 MeV`, `Yp=0.16318833305226624`,
`D/H=2.09662848233949e-05`, `selected_source_evaluations_total=3307`,
`selected_dynamic_collision_payload_builds_total=409`,
`selected_initial_boundary_rhs_seed_reuse_total=6`, and
`selected_frozen_source_jax_jacobian_evaluations_total=403`; a paired no-seed
control keeps the same terminal observables but requires `3313` source
evaluations and `415` dynamic payload builds.  This is a private hot-loop
runtime reduction only; public dispatch, production SMC validation, QKE,
q/angular convergence, and publication-ready claims remain unclaimed.

2026-05-21 BD19 lazy restart payload builds: FB69 now builds restart-state
payloads only when a fresh dynamic AP65 payload refresh is required.  Stage RHS
calls under `stage_collision_payload_policy=step_base_reuse` receive the
host-step base collision payload and skip `_state_to_current_restart_kwargs`;
FB69 records `current_restart_payload_build_count`, and FB70 propagates
selected, h-refinement, and resolution-ladder totals.  A one-step FB69
boundary-trace probe reports `source_evaluation_count=9`,
`dynamic_collision_payload_build_count=2`,
`current_restart_payload_build_count=2`, and
`stage_collision_payload_reuse_count=7`.  The same six-window
non-LRS+collision endpoint run reports `passed=true`,
`physical_full_bbn_span_ready=true`, terminal
`T_gamma=0.009144759664502618 MeV`, `Yp=0.16318833305226624`,
`D/H=2.09662848233949e-05`, `selected_source_evaluations_total=3307`,
`selected_current_restart_payload_builds_total=409`,
`selected_dynamic_collision_payload_builds_total=409`, and
`selected_frozen_source_jax_jacobian_evaluations_total=403`.  This is a
private hot-loop restart-payload cost reduction only; public dispatch,
production SMC validation, QKE, q/angular convergence, and publication-ready
claims remain unclaimed.

2026-05-21 BD20 adaptive step-safety control: FB69 now exposes the private
adaptive Rodas next-step safety factor as `adaptive_step_safety`, with the
default `0.9` preserved and finite `(0, 1]` validation.  FB70 forwards the same
solver-control axis through normal span rows, resolution-ladder cases, and the
CLI.  On the six-window non-LRS+collision endpoint configuration, the opt-in
`0.93` setting preserved `passed=true`, `physical_full_bbn_span_ready=true`,
and raw positive endpoint observables while reducing selected host steps
`403 -> 389`, source evaluations `3307 -> 3258`, dynamic/restart payload
builds `409 -> 395`, and frozen-source JAX Jacobians `403 -> 389`.  This is a
private performance-mode runtime reduction only; public dispatch, production
SMC validation, QKE, q/angular convergence, wall-time portability, and
publication-ready claims remain unclaimed.

2026-05-21 BD21 RHS-only frozen-source Jacobian replay: the existing
frozen-source JAX Jacobian path now calls the replay RHS helper with
`return_metadata=False`, so autodiff skips metadata construction that the
Jacobian never consumes while boundary/final RHS calls still serialize their
metadata.  FB69/FB70 CLIs also expose the existing
`abundance_positivity_policy` axis for reproducible trace-boundary endpoint
runs without truncating reported outputs.  On the six-window private
non-LRS+collision endpoint configuration with `adaptive_step_safety=0.93` and
`trace_boundary`, the local CPU-JAX/Rodas5P reproducibility run preserved
`passed=true`, `physical_full_bbn_span_ready=true`, raw positive endpoint
observables, and the endpoint below `0.01 MeV`.  The supported BD21 claim is
the removal of unused Jacobian replay metadata work on the existing private
surface; those trace-boundary endpoint counts are not isolated step-count
evidence against BD20's default-positivity run.  Public dispatch, production
SMC validation, QKE, q/angular convergence, wall-time portability, and
publication-ready claims remain unclaimed.

2026-05-21 BD22 stage/probe RHS-only metadata policy: FB69 now also passes
`return_metadata=False` for unaccepted Rodas stage RHS calls and
finite-difference Jacobian probes, which only consume the RHS vector.  Boundary
and final RHS metadata remain serialized, payload trace stats remain recorded,
and raw states remain untruncated.  The focused one-row regression records
`source_evaluation_count=9` and `boundary_source_evaluation_count=2` while
calling `_live_source_metadata_payload` only for the two boundary rows.  The
six-window private endpoint reproducibility run with
`adaptive_step_safety=0.93` and `trace_boundary` preserved `passed=true`,
`physical_full_bbn_span_ready=true`, endpoint observables,
`selected_step_count_total=381`, and `selected_source_evaluations_total=3201`;
local wall time was `47.029263105010614 s`.  This is bounded hot-loop work
removal inside the existing private surface, not a portable speedup, public
dispatch, production SMC validation, QKE, q/angular convergence, or
publication-ready support.

2026-05-21 BD23 shared frozen-source Jacobian function cache: FB69 now uses a
bounded structural cache for frozen-source JAX Jacobian functions across
equivalent contexts, avoiding repeated `jax.jacfwd`/`jax.jit` closure
construction across FB70 windows when the replay layout, source grid,
rate-table identity, and scalar RHS controls match.  The focused regression
builds two equivalent contexts and verifies one `jacfwd`/`jit` construction.
The six-window private endpoint reproducibility run with
`adaptive_step_safety=0.93` and `trace_boundary` preserved `passed=true`,
`physical_full_bbn_span_ready=true`, endpoint observables,
`selected_step_count_total=381`, and `selected_source_evaluations_total=3201`;
local wall time was `40.10995108494535 s`.  This is backend setup reuse inside
the existing private CPU-JAX/Rodas5P surface, not a portable speedup, public
dispatch, production SMC validation, QKE, q/angular convergence, or
publication-ready support.

2026-05-21 BD25 frozen-source finite-difference Jacobian policy: FB69 now
accepts `jacobian_policy=frozen_source_finite_difference`, which keeps full
finite-difference as the tiny-grid reference while reusing the host-step base
AP65 collision payload inside Jacobian probe RHS calls.  The policy records
structured-Jacobian and frozen-source finite-difference counters and is exposed
through the FB69/FB70/FB82/FB83/FB86/FB87 and consolidated span CLIs.  A
one-row CPU-JAX/Rodas5P smoke with boundary traces and
`stage_collision_payload_policy=step_base_reuse` preserved `passed=true` while
reducing dynamic/restart payload builds from `43` to `2` versus full
finite-difference; both rows still used `41` RHS-column probes.  This is a
private finite-difference fallback/runtime policy only; `frozen_source_jax`
remains the repeated-run backend target, raw states are not truncated, and
public dispatch, production SMC validation, QKE, q/angular convergence, and
publication-ready support remain unclaimed.

2026-05-21 BD26 FB70 frozen-source finite-difference propagation: FB70 now
accepts `jacobian_policy=frozen_source_finite_difference` in its builder,
forwards it to nested FB69 endpoint rungs, and preserves the new
frozen-source finite-difference Jacobian counter in terminal rows,
h-refinement attempts, nested resolution/freedom rows, and selected/total
summary telemetry.  A one-rung FB70 CPU-JAX/Rodas5P smoke with boundary traces
and `stage_collision_payload_policy=step_base_reuse` passed as a hot-endpoint
row and reported `selected_frozen_source_finite_difference_jacobian_evaluations_total=1`,
`selected_jacobian_probe_source_evaluations_total=41`, and dynamic/restart
payload builds of `2`.  This corrects endpoint-ladder runtime-policy
propagation from BD25; it is not a standalone gate, keeps `frozen_source_jax`
as the repeated-run target, preserves raw states, and does not claim public
dispatch, production SMC validation, QKE, q/angular convergence, or
publication-ready support.

2026-05-21 BD27 RHS-only JAX stage path: FB69 now uses a cached RHS-only JAX
function for inner-loop `return_metadata=False` evaluations when
`jacobian_policy=frozen_source_jax`, the repeated-run CPU-JAX/Rodas5P target.
Boundary/final metadata calls are unchanged, and finite-difference
reference/fallback policies keep the eager RHS path to avoid first-compile
overhead on tiny-grid checks.  A one-row FB69 smoke with boundary traces and
`stage_collision_payload_policy=step_base_reuse` reported
`rhs_only_jax_evaluations_total=7`, `source_evaluations_total=9`, and
`passed=true`; the paired frozen-source finite-difference fallback preserved
`rhs_only_jax_evaluations_total=0` and `jacobian_probe_source_evaluations_total=41`.
The same count is propagated through FB70 as
`selected_rhs_only_jax_evaluations_total=7`.  This is private hot-loop
execution-path work, not a new readiness gate, and it does not claim public
dispatch, production SMC validation, QKE, q/angular convergence, raw-output
repair, or publication-ready support.

2026-05-21 BD28 dynamic AP65 payload setup cache: dynamic restart-state AP65
payload refreshes now unpack the current-state restart mapping directly with
NumPy for collision-source construction instead of using a host CPU-JAX replay
vector round trip.  AP6 unit-direction PSTF radial momentum-delta weights are
also cached through a bounded byte-budgeted static-delta cache for same-geometry rebuilds,
while temperature-dependent radial grids/source values remain current-state
dependent and radial-grid cache counters remain grid-only.  The six-window
private FB70 CPU-JAX/Rodas5P endpoint smoke preserved `passed=true`,
`physical_full_bbn_span_ready=true`, `T_final_MeV_min=0.009144759664395794`,
`selected_step_count_total=389`, and `selected_dynamic_collision_payload_builds_total=395`;
local wall evidence changed from BD27 `21.18 s` to BD28 `21.05 s`, and the
paired cProfile dynamic payload path dropped from about `13.10 s` to `11.76 s`.
This is runtime hot-loop work on existing private surfaces, not a standalone
gate, public dispatch, production SMC validation, QKE, q/angular convergence,
raw-output repair, or publication-ready support.

2026-05-21 BD29 AP6 radial moment-weight cache: AP6 radial moment-weight and
neutral-projection bundles are now reused through a bounded byte-budgeted
in-module cache keyed by `p1_energies`, radial mode count, `p1_weights`,
`mode_index`, and moment powers.  The cache is separate from `radial_grid_cache`,
so radial-grid counters remain grid-only.  The six-window private FB70
CPU-JAX/Rodas5P endpoint smoke preserved `passed=true`,
`physical_full_bbn_span_ready=true`, `T_final_MeV_min=0.009144759664395794`,
`selected_step_count_total=389`, `selected_source_evaluations_total=3258`, and
`selected_dynamic_collision_payload_builds_total=395`; local wall evidence
changed from BD28 `21.05 s` to BD29 `19.97 s`.  The paired cProfile run reduced
`build_pstf_process_radial_moment_weights(...)` from `7110` calls to `1`.
Remaining blocker work is radial grid/source rebuild cost, JAX compile/cache
misses, and Rodas5P Jacobian/LU cost, not another standalone gate.

2026-05-22 BD64-BD66 continuous AP65 runtime-cache reuse: the CPU-JAX/Rodas5P
private live-source chain and FB69/FB70 continuous-AP65 path now reuse fixed
live-source grids and AP65 runtime caches across repeated windows/artifacts.
BD64 moves one `AugmentedNonLRSSourceGridJax` to the live-source chain boundary;
BD65 shares AP65 radial-grid and source-factory caches through the existing
`runtime_cache`; BD66 keys the FB69 live-source grid in that same runtime cache.
The focused regressions passed without changing physics equations, raw-state
reporting, public dispatch status, production SMC status, or no-QKE scope.

2026-05-22 BD67 endpoint probe refresh after cache reuse: existing FB69/FB70
private endpoint probes were rerun with CPU-JAX/Rodas5P,
`chain_restart_handoff`, `chain_h_max_policy=first_rejection_half_ceiling_once`,
`jacobian_policy=frozen_source_jax`, `rhs_trace_policy=boundary`,
`abundance_positivity_policy=trace_boundary`, `h_max=0.2`, and
`N_span_end_ladder=(0.8,1.6,2.4,3.2,4.0,4.8)`.  Non-LRS no-collision reached
`T_gamma=0.00913919409378424 MeV` in `10.089954509865493 s`; non-LRS
collision with `stage_collision_payload_policy=step_base_reuse` reached
`T_gamma=0.009144759673024236 MeV` in `23.59657490684185 s`; all-three
weak+non-LRS+collision with `weak_correction_level=3` reached
`T_gamma=0.009144759672723686 MeV` in `19.1935998670524 s`.  All three rows
reported `physical_full_bbn_span_ready=true`, `rows_reaching_endpoint=1`, and
no violations.  This retires the smoke-scale private continuous-AP65 endpoint
blocker, but it remains diagnostic only: q/angular/tolerance convergence,
weak-rate profile extension, endpoint-backed figures/statistics, public
dispatch, production SMC validation, QKE, and publication-ready support remain
unclaimed.

2026-05-22 BD70 exact-current-state weak-pair provenance: FB70
freedom-composition rows now thread `stop_at_T_gamma_MeV` into each nested
continuous-span child run and preserve the continuous-AP65 solver controls used
to compare weak/control rows.  FB71 rejects FB70-derived weak/control pairs
when those controls differ, and FB72 propagates the matched exact policy labels
through its bridge summary.  A real exact all-three CPU-JAX/Rodas5P
freedom-composition rerun with `stage_collision_payload_policy=current_state`,
`chain_h_max_policy=first_rejection_or_recovered_h_ceiling`,
`h_refinement_factors=(1.0,0.5)`, `frozen_source_jax`,
`rhs_trace_policy=boundary`, and `stop_at_T_gamma_MeV=0.01` passed both the
non-LRS+collision control and weak-level-3 rows below `0.01 MeV`.  FB71 over
that artifact reported `full_bbn_weak_rate_pairs_ready=true`,
`fb70_continuous_ap65_pair_solver_policy_matched=true`,
`max_abs_weak_delta_Yp=0.001689015172495728`, and
`max_abs_weak_delta_DH=7.305065755661101e-08`; FB72 with the existing AP80
smoke artifact reported `ap80_fb71_bridge_ready=true` while keeping the bridge
private and scoped.  This folds exact endpoint weak/control evidence into the
existing FB70/FB71/FB72 surfaces; default-context resolution/tolerance
expansion, endpoint-backed figures/statistics, public dispatch, production SMC
validation, QKE, and publication-ready support remain unclaimed.

2026-05-22 BD71 endpoint-backed figure/statistical input refresh: FB73, FB74,
and FB75 now expose explicit scoped weak-rate bridge override controls while
keeping the default fail-closed behavior for scoped FB72/FB73 evidence.  FB75
also accepts a consistent endpoint-ready FB70 span ladder and no longer
requires every multi-window `T_final_MeV_max` to be below `0.01 MeV`; the
consistency check uses completed endpoint rows plus `T_final_MeV_min <= 0.01`.
The actual file-backed BD69/BD70 endpoint artifacts now flow through the
existing figure and guarded-SMC input path: FB73 rendered four diagnostic
figures with `continuous_ap65_physical_span_ready=true` and scoped FB72 weak
bridge coverage; FB74 QA-packaged them with
`weak_rate_bridge_required_context_scope=scoped_subset`; FB75 consumed the
file-backed FB60, FB66, exact FB70, exact scoped FB72, and scoped FB74 inputs
with `guarded_smc_pilot_input_ready=true`, `validated_full_bbn_product_inputs_ready=true`,
`statistical_pilot_input_ready=true`, no violations, `fb70_rows_reaching_endpoint=1`,
and `pilot_blockers=["weak_rate_bridge_default_context_matrix_not_ready"]`.
This removes the stale `continuous_ap65_full_bbn_span_not_ready` blocker from
the exact current-state statistical-input path and leaves the real blocker at
default-context weak-rate expansion plus q/angular/tolerance convergence.  No
public dispatch, production SMC validation, QKE, or publication-ready support
is claimed.

2026-05-22 BD72 exact default-context weak bridge evidence refresh: the existing
FB70/FB72/FB73/FB74/FB75 path now has real exact-current-state default-context
evidence without adding code or a new gate.  The CPU-JAX/Rodas5P FB70
freedom-composition default matrix with
`stage_collision_payload_policy=current_state`,
`chain_h_max_policy=first_rejection_or_recovered_h_ceiling`,
`h_refinement_factors=(1.0,0.5)`, `frozen_source_jax`,
`rhs_trace_policy=boundary`, and `stop_at_T_gamma_MeV=0.01` passed all eight
default rows with `physical_full_bbn_span_ready=true`,
`rows_full_bbn_completed=8`, `failed_or_exception_rows=0`,
`T_final_MeV_min=0.00913919385022409`, and
`T_final_MeV_max=0.009144759665756305`.  FB72 over that exact default matrix
and the existing AP80 smoke artifact passed with
`fb71_required_context_scope=default_all_contexts`, `fb71_passed_pair_count=4`,
`fb71_rows_reaching_full_bbn_endpoint=8`, and matched FB70 solver policy
provenance for `current_state` plus the recovered-h chain policy.  FB73/FB74/FB75
then ran without scoped overrides, and FB75 reported
`guarded_smc_pilot_input_ready=true`, `statistical_pilot_input_ready=true`, no
violations, `fb72_required_context_scope=default_all_contexts`, and
`pilot_blockers=["diagnostic_scope_not_promoted"]`.  This retires the
default-context weak-rate bridge blocker for the exact current-state endpoint
path; endpoint-backed q/angular/tolerance convergence, AP80/profile extension
beyond smoke, production SMC validation, public dispatch, QKE, and
publication-ready support remain unclaimed.

2026-05-22 BD73 block-JVP Rodas5P q4 memory triage: FB69/FB70 now expose the
private `jacobian_policy=frozen_source_jax_block_jvp` policy on the existing
continuous-AP65 surfaces.  The policy keeps exact frozen-source JVP columns for
the geometry/thermo and phase-2 network blocks, omits the high-resolution
`A_modes_flat` columns from the implicit linear solve, and still evaluates the
live current-state RHS for accepted stages and adaptive error estimates.  The
JVP helper now forms one unit tangent at a time instead of allocating a dense
identity, and FB70 preserves block-JVP Jacobian counts through selected-row and
h-refinement summaries.  The q4 all-freedom exact-current-state probe that
previously hit memory/no-artifact failure now writes a raw failed artifact under
the block-JVP policy: the first row fails at `wall_time_budget_seconds` with
`step_count=54`, `attempt_count=61`, `n_rejected=7`, and
`frozen_source_jax_block_jvp_jacobian_evaluation_count=54`.  The early full
column-JVP triage was not retained in BD73; BD77 later reintroduced a
one-column full-state JVP policy after BD76 showed that omitting the A-mode
columns was the active q4 step-size limiter.  BD73 reduces the blocker from
OOM/no-artifact to an explicit q4 Rodas5P step-size/wall-budget failure; it
does not establish q/angular convergence, public dispatch, production SMC
validation, publication-ready support, or QKE.

2026-05-22 BD74 trace-abundance Rodas5P domain guard: FB69 now checks the
phase-2 trace abundance block before evaluating a host Rodas5P internal stage
RHS and before accepting a step candidate.  If indices `[3,4,5,6,7,8]` move
below the live-source safe tolerance `1e-14`, the state is rejected through the
existing adaptive stage-domain retry path rather than being used in a source RHS
or accepted and later exposed as a negative raw abundance or negative `Y_p`.
This is not an output clamp:
accepted states and terminal observables remain raw, and the default raw RHS
policy remains available.  FB70 now preserves the trace-abundance domain
rejection counter through selected-row, h-refinement, and summary telemetry.
Focused tests cover the helper, the host Rodas5P stage path, and the
`_run_step_cap_row` acceptance path.  q4/mu5 all-freedom local probes with
`frozen_source_jax_block_jvp` and
current-state AP65 stage payloads still hit the 60 second wall budget before
endpoint: raw reached only `step_count=9`, `attempt_count=22`, `n_rejected=13`,
with `trace_abundance_domain_rejection_count_total=6`, while `trace_boundary`
reached `step_count=8`, `attempt_count=15`, `n_rejected=7`.  The remaining
blocker is therefore not another claim/gate
surface but the continuous AP65 endpoint below `0.01 MeV` plus hot-loop
payload/Jacobian cost.

2026-05-22 BD75 auto small-collision stage payload reuse: FB69 now exposes the
private `stage_collision_payload_policy=auto_small_collision_reuse` policy on
the existing continuous-AP65 RHS surface.  The host Rodas5P stage RHS reuses
the host-step base AP65 collision payload only when the base `dA_modes`
amplitude is finite and no larger than `1e-6`; otherwise it falls back to exact
current-state stage payload construction.  FB69 records auto reuse versus
auto-current-state fallback counts, FB70 preserves those counters through
selected-row, h-refinement, and summary telemetry, and the FB69/FB70 CLIs
accept the same policy without adding a new gate.  On the q4/mu5 all-freedom
trace-boundary probe with `frozen_source_jax_block_jvp`, the BD74 exact
`current_state` path advanced only `step_count=8` in a 60 second wall budget.
The fixed `step_base_reuse` comparison advanced `step_count=83` with
`selected_stage_collision_payload_reuse_total=630`; the BD75 auto policy
advanced `step_count=85`, `attempt_count=92`, `n_rejected=7`,
`selected_stage_collision_payload_auto_reuse_total=644`,
`selected_stage_collision_payload_auto_current_state_total=0`, and
`rhs_stress_collision_dA_abs_max_max=1.5255268474667314e-07`
(`artifact_payload_sha256=2a9ea8b9d192693c932b0e607ffa182b420f038d36d6e5efc5a340a2462dd6da`).
This is a private CPU-JAX/Rodas5P hot-loop runtime reduction only: exact
current-state stage payloads remain expensive, the full-BBN endpoint below
`0.01 MeV` is still not reached, and public dispatch, production SMC
validation, publication-ready support, and QKE remain unclaimed.

2026-05-22 BD76 Rodas error-block attribution: FB69 now records the Rodas5P
error-estimator contribution by state block (`geometry_thermo`, `A_modes_flat`,
`X_phase2`) on the existing continuous-AP65 rows, and FB70 carries selected-row
and h-refinement dominant-block counts through the span-ladder summary.  This
does not alter the global adaptive error norm, accepted states, raw
observables, or any tolerance policy; it is solver attribution on the existing
runtime surface, not a standalone gate.  The q4/mu5 all-freedom
`auto_small_collision_reuse` probe wrote
`artifact_payload_sha256=990b640c86e920ad035e575837dcf6ad3be8ac997c5cd629b8aa57b6c7c9e1d5`,
remained wall-budget limited at `step_count=87`, `attempt_count=94`,
`n_rejected=7`, and recorded
`error_norm_dominant_block_counts={"A_modes_flat":94}` with
`selected_stage_collision_payload_auto_reuse_total=658`.  This narrows the next
solver/physics blocker from generic network/tolerance suspicion to the kinetic
hierarchy block: A-mode linearization, A-mode transport/collision RHS scaling,
or a justified implicit/block treatment for those modes.  Endpoint readiness,
q/angular convergence, public dispatch, production SMC validation,
publication-ready support, and QKE remain unclaimed.

2026-05-22 BD77 full-state one-column JVP Rodas policy: FB69 now exposes the
private `jacobian_policy=frozen_source_jax_full_jvp` policy on the existing
continuous-AP65 RHS surface, and FB70 plus the FB69/FB70 CLIs carry that policy
and its evaluation counters through selected-row, h-refinement, resolution, and
summary telemetry.  The policy differentiates every packed state column at the
frozen AP65 collision payload with one-column JVPs, so the A-mode columns are
restored to the implicit linear solve without allocating a dense tangent
identity; accepted stages and adaptive error estimates still evaluate the live
current-state RHS.  The q4/mu5 all-freedom probe with
`stage_collision_payload_policy=auto_small_collision_reuse`,
`rhs_trace_policy=boundary`, and `trace_boundary` abundances passed
`N_span_end=0.8` with
`artifact_payload_sha256=374e880b63c209aa1bc02b0f453014ac40c77d71b111fde044d82b2d164df711`,
`step_count=20`, `attempt_count=31`, `n_rejected=11`,
`selected_frozen_source_jax_full_jvp_jacobian_evaluations_total=20`,
`selected_stage_collision_payload_auto_reuse_total=217`, and terminal
`T_gamma=0.3708908727961177 MeV`.  A longer `N_span_end=2.0` probe failed
closed after `46` accepted steps with
`artifact_payload_sha256=a291288b9d17648628b30fbd85d63e0152670dba407e0cd6edd57459c5df714c`,
`trace_abundance_domain_rejection_count_total=9`, and raw preserved
stage-domain exceptions beginning near `T_gamma~0.247 MeV` and recurring near
`T_gamma~0.177 MeV`.  This retires the immediate q4 A-mode linearization
blocker but shifts the remaining direct implementation target to long-span
trace-abundance/domain evolution stability and endpoint completion below
`0.01 MeV`; q/angular convergence, public dispatch, production SMC validation,
publication-ready support, and QKE remain unclaimed.

2026-05-22 BD78 stage-only trace RHS projection for reused-payload Rodas
stages: FB69 now applies a bounded projection only to unaccepted internal-stage
RHS inputs when `abundance_positivity_policy=trace_boundary` and the stage uses
a reused/frozen AP65 source payload.  Current-state source refreshes, accepted
states, candidate states, final states, and terminal observables remain raw and
strict; this is not final-state truncation or `Y_p` repair.  The projection is
limited to phase-2 trace species whose accepted step-base abundance is within
the solver absolute trace layer, with the permitted stage undershoot tied to
solver `atol`.  FB69 records role/index/raw-min/projection-tolerance telemetry,
and FB70 preserves selected-row, h-refinement, and summary totals for
`stage_trace_abundance_projection_count`,
`stage_trace_abundance_projection_event_count`, and
`stage_trace_abundance_projected_abs_max`.  The same baseline-grid q4/mu5
all-freedom `N_span_end=2.0` probe that failed in BD77 now passed with
`artifact_payload_sha256=3218f17c7c0431722281f4a882dd8ad9fb415102fc329ca737162d7249dda060`,
terminal `T_gamma=0.13208063389700056 MeV`, `step_count=39`,
`attempt_count=60`, `n_rejected=21`,
`stage_trace_abundance_projection_count_total=24`,
`stage_trace_abundance_projection_event_count_total=19`, and
`stage_trace_abundance_projected_abs_max=6.076688171653506e-09`.  This moves the
long-span trace-domain stage-instability blocker on the existing FB69/FB70
private surface without adding a gate; endpoint completion below `0.01 MeV`,
q/angular convergence, exact current-state stage-source cost, public dispatch,
production SMC validation, publication-ready support, and QKE remain unclaimed.

2026-05-22 BD79 direct-logit transport and trace-domain recovery triage:
the AP62 non-LRS transport RHS now evaluates the augmented-logit A-mode
transport equation directly in both Python and CPU-JAX live-source paths,
removing the old low-occupancy tail amplification from `df/dN` projection
followed by division through `f(1-f)`.  FB69 also changes recoverable
trace-domain failures to a consecutive rejection budget, applies a
boundary-crossing-limited retry step for negative trace candidates, and extends
the unaccepted-stage trace projection only for depleted trace references under
an explicit diagnostic cap.  Accepted states, candidate/final states, terminal
observables, and raw failure artifacts remain untruncated.  An opt-in
trace-log coordinate pilot converts RHS/Jacobian/error estimates back to
physical abundance units; BD81 narrows the actual solver transform to active
trace species only.  The pilot remains disabled by default because local
endpoint probes still overflowed or reached early `h_min`.

The q4/mu5 baseline-grid all-freedom boundary-limited run progressed to
`T_gamma~0.07990 MeV` and then failed closed by `max_steps`, with
`368` accepted steps, `144` rejected steps, and
`trace_abundance_domain_rejection_count_total=135`; this preserves the raw
trace-domain evidence and does not claim endpoint completion.  The late local
direct-logit probe reduced the A-mode transport RHS stress from the previous
tail-amplified `dA_abs_max~1.24e5` scale to `dA_abs_max=0.54545`, moving the
remaining implementation target to the phase-2 Li/Be/Li6 positivity/stiffness
problem, production/destruction flux auditing, and eventually a stable
positivity-preserving network coordinate or constrained implicit network block.
`solver_audit_english.md` records the external-audit prompt and recommended
follow-up path.  This is private CPU-JAX/Rodas5P solver/transport work on the
existing continuous-AP65 surface, not a new gate, public dispatch, production
support, publication-ready evidence, or QKE support.

2026-05-22 BD80 phase-2 trace network production/destruction split:
the standard PRIMAT network now exposes directional forward and reverse flux
components plus a trace-species production/destruction split for `7Li`, `7Be`,
and `6Li`.  The split applies normal stoichiometry to forward reactions and
opposite stoichiometry to reverse reactions, so for nonnegative physical
abundances the per-species production/destruction totals are nonnegative and
reconstruct the nuclear part of `abundance_rhs_phase2`.  FB69 rows now attach
`phase2_trace_flux_split_summary` at the accepted row state.  Raw-policy rows
that already contain negative trace abundances preserve
`raw_negative_species_indices` and add only a labeled
`diagnostic_nonnegative_counterfactual`; no accepted state, candidate state,
terminal abundance, or `Y_p` is repaired or truncated.

Focused tests lock the network split against the standard RHS and lock the FB69
row payload boundary.  This is an existing-surface diagnostic that directly
supports the next solver/physics implementation step: deciding whether the
Li/Be/Li6 endpoint blocker is real production/destruction stiffness, reverse
flux cancellation, or a flux-form bug before landing a stable
positivity-preserving network coordinate or constrained implicit network block.
It is not a standalone gate, public dispatch, production support,
publication-ready evidence, or QKE support.

2026-05-22 BD81 selective active trace-log solver coordinate pilot
(historical, superseded by BD95): FB69 exposed
`trace_log_solver_coordinates_enabled=False` on the existing continuous-AP65
artifact builder.  When explicitly enabled together with
`abundance_positivity_policy="trace_boundary"`, that pilot encoded only trace
species above the active abundance floor as log solver variables.  BD95 replaces
that active-floor-only transform with encoding of every finite nonnegative
constrained trace abundance, exact-zero floor accounting, and direct-X
preservation only for negative or nonfinite accepted trace-state evidence.

The row and prototype metadata still record whether the transform is enabled,
zero-floor counts, inactive direct-X evidence counts, and that output
truncation is not applied.  This stays on the existing FB69 surface rather than
adding a new gate.  It does not alter network fluxes, terminal observables,
public dispatch, production support, publication-ready evidence, or QKE scope.
The next runtime blocker remains endpoint progress below `0.01 MeV`, with
trace-network stiffness, frozen-source Jacobian quality, and a possible
constrained implicit production/destruction network block as live candidates.

2026-05-22 BD82 positive 3T solver coordinates and recoverable nonfinite
Rodas stages:
FB69 now exposes `temperature_log_solver_coordinates_enabled=False` on the same
continuous-AP65 artifact builder.  When explicitly enabled, the packed
`T_gamma`, `T_nu_e`, and `T_nu_x` state entries are represented internally as
positive log solver coordinates inside the host Rodas5P stage algebra, then
decoded back to physical MeV units for RHS evaluation, Jacobian conversion,
embedded-error scaling, diagnostics, and output.  This is an invariant-domain
solver variable change, not display/output truncation.

The same PR also reclassifies nonfinite transformed linear-system RHS/matrix
values as recoverable solver-domain rejections, with explicit row counters
`linear_system_rhs_nonfinite_rejection_count` and
`linear_system_matrix_nonfinite_rejection_count`.  The source-RHS CLI now
forwards `--trace-log-solver-coordinates` and
`--temperature-log-solver-coordinates` into the existing FB69 builder for
repeatable CPU-JAX/Rodas5P probes.

Measured q4/mu5 all-freedom endpoint probes on the existing FB69 surface:
BD81 trace-only coordinates reached `N_final=2.5829709354498807` with
`trace_abundance_domain_rejection_count=56` and
`stage_temperature_domain_rejection_count=43` in local artifact
`/tmp/rabbit_bd82_selective_trace_log_endpoint_q4_mu5_fb69.json`
(`b360e1db50023990ab612c2362f260ea68c9f3c63852cf45dcd0082c084af311`).
BD82 trace+temperature coordinates plus recoverable nonfinite linear RHS handling reached
`N_final=2.5831949741940643` with
`trace_abundance_domain_rejection_count=27`,
`stage_temperature_domain_rejection_count=61`, and
`linear_system_rhs_nonfinite_rejection_count=9` in local artifact
`/tmp/rabbit_bd82_trace_temperature_log_recoverable_rhs_endpoint_q4_mu5_fb69.json`
(`dd2e5545339fc2d797034f0d7d204766ee68cdfd2648022a3989bba394513416`).
The fatal nonfinite-RHS failure became an adaptive max-step closure, but full
endpoint completion below `0.01 MeV` remains unclaimed.  The remaining blocker is active
phase-2 trace-network stiffness/overshoot and step-count pressure.  If five
more BD/PR commits produce no concrete physics, solver, or performance
breakthrough, stop and write a self-contained external-audit prompt.

2026-05-22 BD83 trace-tail Patankar candidate corrector: FB69 now exposes an
opt-in `trace_tail_patankar_corrector_enabled` private solver path and the CLI
flag `--trace-tail-patankar-corrector`.  The corrector applies only to the
Li7/Be7/Li6 trace tail after a Rodas candidate is formed, using the in-tree
phase-2 production/destruction split as `(X_n + h P)/(1 + h D)`.  It records
raw Rodas candidate values, corrected values, per-N production/destruction
rates, correction magnitude, and mass-fraction delta; it is a private solver
candidate update, not an output truncation or public dispatch claim.

The reviewed q4/mu5 all-freedom endpoint probe with BD82 trace+temperature
coordinates and BD83 trace-tail Patankar correction reached the full
`N_final=4.8`, below `0.01 MeV`, with `passed=true`, `violations=[]`,
`step_count=373`, `attempt_count=397`, `n_rejected=24`,
`trace_abundance_domain_rejection_count=1`,
`stage_temperature_domain_rejection_count=4`, no nonfinite linear-system
rejections, finite staging BBN readouts
(`T_gamma_MeV=0.00914475830896589`, `Yp=0.16318718435928586`,
`D/H=2.0956470941271044e-05`, `N_eff_3T=2.978160043133659`), and artifact
`/tmp/rabbit_bd83_trace_tail_patankar_accepted_telemetry_q4_mu5_fb69.json`
(`8d98c96b102d9ea35343f3d054449b2066d9085267b04cb379044dc42f6a688e`).
This is the first continuous-AP65 q4/mu5 private row in this line to produce an
endpoint-reaching candidate with finite BBN observables while counting the
Patankar correction in the acceptance error diagnostics, splitting
`trace_tail_patankar_corrector_attempt_count=392`,
`accepted_count=373`, and `rejected_count=19`, and preserving the one raw
negative Rodas-candidate event in a dedicated audit sample.  The candidate
depends on the opt-in, not-yet-converged, non-conservative trace-tail corrector.
The next blocker is convergence and physics validation of that corrector,
 especially h-refinement, mass/charge residual audits, and comparison against
micro-window or network-only references; public production support and QKE
remain unclaimed.

2026-05-22 BD84 trace-tail Patankar convergence telemetry: FB69 now records
phase-2 mass-fraction sum and charge-weighted abundance diagnostics for each
row, Patankar attempt/accepted/rejected mass- and charge-delta maxima, and
adjacent `h_max` comparison deltas for those abundance diagnostics on the
existing continuous-span row surface.  This folds the next validation work into
FB69 instead of adding a new readiness gate.

The q4/mu5 all-freedom full-span refinement probe with
`h_max_ladder=(0.1,0.05)` passed both rows with the BD82 trace+temperature
coordinates and BD83 trace-tail corrector enabled.  The artifact
`/tmp/rabbit_bd84_trace_tail_patankar_hmax_refinement_q4_mu5_fb69.json`
(`artifact_payload_sha256=3d451b6448a779756da01b66cff145150b61d1e8817b3ae76ba33859a7f93f56`,
file SHA256 `c5c966472f79c9d2be9f67d0629359553b8bd3c87dadc4cd526b909fa713f64c`)
reported row-0/row-1 step counts `373/395`, rejected attempts `24/13`,
accepted Patankar counts `373/395`, accepted Patankar mass-delta maxima
`1.4110550437461175e-10` and `1.388034567545877e-10`, accepted
charge-delta maxima `6.047383504857119e-11` and
`5.94872422295282e-11`, final `Yp` values `0.16318718435928586` and
`0.163187193211788`, final `D/H` values `2.0956470941271044e-05` and
`2.0956288836728712e-05`, adjacent `Yp` delta `8.85250214799349e-09`,
adjacent `D/H` delta `1.8210454233222166e-10`, and adjacent final
mass-fraction-sum delta `9.290719304999584e-10`.  This is encouraging
same-configuration h-refinement evidence for the private endpoint candidate,
not a publication-grade convergence or production-support claim.  Remaining
blockers are micro-window/network-only comparison, q/angular convergence,
statistical-pipeline refresh, and public dispatch remaining closed.

BD84 self-audit: `real_blocker_moved=same-setup h-refinement and mass/charge
audit telemetry for the endpoint-reaching private Patankar candidate`;
`gate_removed_or_consolidated=existing FB69 continuous-span row surface, no new
standalone gate`; `raw_state_preserved=final vectors, raw negative Rodas
candidate samples, and untruncated observables`; `verification=focused tests,
sync_test_counts.py, git diff --check, and q4/mu5 h_max=(0.1,0.05) endpoint
probe`; `remaining_blocker=micro-window/network-only validation of the
non-conservative trace-tail corrector, then q/angular/tolerance convergence and
statistical-pipeline refresh`.

2026-05-22 BD85 trace-tail network-reference corrector audit: FB69 now computes
a frozen-background four-substep network-only reference for each trace-tail
Patankar attempt and records Patankar-minus-reference absolute, raw-relative,
active-floor-relative, mass-fraction, and charge-fraction deltas for attempted,
accepted, and rejected applications, with explicit available/unavailable
reference counts and unavailable-reason counts.  This remains local solver
telemetry, not a standalone readiness gate or a global network-only convergence
proof.

The q4/mu5 one-row endpoint probe
`/tmp/rabbit_bd85_trace_tail_network_reference_q4_mu5_fb69.json`
(`artifact_payload_sha256=5053a09ef63b86a4889f8be9a5f573fa7484b430240b0874ea588da563d759b6`,
file SHA256 `c5ed880c0ac164ce44fbcf83519140fb71bb920cf42c00dcdebbc0ba83336739`)
passed with `N_final=4.8`, `step_count=373`, `attempt_count=397`,
`n_rejected=24`, `accepted Patankar count=373`, `rejected Patankar count=19`,
reference availability counts `attempt=392/392`, `accepted=373/373`,
`rejected=19/19`, accepted Patankar-vs-reference
`abs_delta_max=3.2818022951510465e-11`,
`active_floor_rel_delta_max=0.037286123183448115`,
`mass_delta_abs_max=3.281802379621345e-11`, and
`charge_delta_abs_max=1.4064867461906192e-11`.  Endpoint readouts remained
finite (`Yp=0.16318718435928586`, `D/H=2.0956470941271044e-05`).  BD85
self-audit: `real_blocker_moved=local network-reference deltas for the
non-conservative trace-tail corrector`; `gate_removed_or_consolidated=existing
FB69 telemetry, no standalone gate`; `raw_state_preserved=raw candidates,
untruncated vectors, and Patankar-minus-reference deltas`; `verification=focused
tests, py_compile, and real q4/mu5 endpoint probe`; `remaining_blocker=longer
micro-window or full network-only comparison, then q/angular/tolerance
convergence and statistical-pipeline refresh`.

2026-05-22 BD86 trace-tail accepted-window replay audit: FB69 now records
accepted trace-tail Patankar start/end phase-2 samples in memory and replays
Li7/Be7/Li6 through the refreshed production/destruction split over the accepted
solver step sequence.  The replay follows the accepted solver thermodynamic and
non-tail phase-2 background, reports completion/unavailable reasons, final
solver-minus-reference tail values, final and stepwise deltas, and mass/charge
deltas, and does not alter solver states or truncate outputs.  This remains
accepted-background trace-tail replay telemetry, not a standalone readiness gate
or a full independent phase-2 network integration proof.

The q4/mu5 one-row endpoint probe
`/tmp/rabbit_bd86_trace_tail_window_reference_q4_mu5_fb69.json`
(`artifact_payload_sha256=31fca47c82ada46655f088c070a560fae8f07e207d43ab2196b671f4736e7bf6`,
file SHA256 `6348c01618304f800f57e456ceccbfa1ba63d520e0263593bf845c807be49714`)
passed with `N_final=4.8`, `step_count=373`, `attempt_count=397`,
`n_rejected=24`, and a completed accepted-window replay over `373` trace-tail
steps.  Window replay deltas were
`final_abs_delta_max=3.853295547106437e-11`,
`final_active_floor_rel_delta_max=0.01710025512934746`,
`step_abs_delta_max=4.897935231531469e-11`,
`mass_fraction_delta=-3.979172293096205e-11`, and
`charge_fraction_delta=-2.2558303466278645e-11`.  Endpoint readouts remained
finite (`Yp=0.16318718435928586`, `D/H=2.0956470941271044e-05`).  BD86
self-audit: `real_blocker_moved=multi-step accepted-window trace-tail replay
over the endpoint row`; `gate_removed_or_consolidated=existing FB69 telemetry,
no standalone gate`; `raw_state_preserved=raw candidates, untruncated vectors,
and solver-vs-window-reference deltas`; `verification=red-first helper test,
focused tests, py_compile, and real q4/mu5 endpoint probe`;
remaining blocker is full phase-2 network-only comparison, then
q-angular-tolerance convergence and statistical-pipeline refresh.

2026-05-22 BD87 full phase-2 accepted-background replay audit: FB69 now records
accepted-background live weak rates with the accepted phase-2 start/end samples
and attempts a full nine-species phase-2 production/destruction replay over the
accepted solver thermodynamic and weak-rate backgrounds.  The replay covers
`n`, `p`, `D`, `T`, `He3`, `He4`, `Li7`, `Be7`, and `Li6`, uses the in-tree
phase-2 split plus accepted `lambda_np`/`lambda_pn`, and fails closed on
nonfinite values or abundance blow-up.  Row and summary telemetry now report
requested steps, completed steps, unavailable steps, completed fraction,
replay `N_end`, last reference abundance scale, and solver-minus-reference
deltas; incomplete-replay deltas are tied to the last completed replay step,
not the final BBN endpoint.  This remains existing FB69 telemetry, not a
standalone readiness gate, public backend, QKE path, output truncation, or
full independent BBN proof.

The q4/mu5 one-row endpoint probe
`/tmp/rabbit_bd87_phase2_full_network_window_reference_q4_mu5_fb69.json`
(`artifact_payload_sha256=6b32405c453b03a39ac92e7139f937468b63f0f735f40ac2784e81a917059319`,
file SHA256 `2365ef81ca418f41610ae19501bb8ad35cce6dfb5f51cd44cb904b11b50e8b36`)
passed the main FB69 endpoint row with `N_final=4.8`, `step_count=373`,
`attempt_count=397`, `n_rejected=24`, and finite endpoint readouts
(`Yp=0.16318718435928586`, `D/H=2.0956470941271044e-05`,
`N_eff_3T=2.978160043133659`).  The BD86 trace-tail window replay still
completed all `373` accepted steps, but the new full phase-2 replay completed
only `2/373` requested steps (`completed_fraction=0.005361930294906166`) and
stopped at `N_end=0.08719626309362064` with
`unavailable_reason_counts={"full phase2 reference abundance blow-up limit exceeded": 1}`,
`last_reference_abs_max=346823686802.07214`, and
`reference_abs_limit=1000000.0`.  BD87 self-audit:
`real_blocker_moved=first full nine-species accepted-background replay exposed
an early full-network blow-up point`; `gate_removed_or_consolidated=existing
FB69 telemetry, no standalone gate`; `raw_state_preserved=raw final vectors,
raw negative Rodas samples, accepted start/end samples, and partial replay
deltas`; `verification=red-first helper tests, review-driven partial-step
accounting regression, focused tests, py_compile, and real q4/mu5 endpoint
probe`; remaining blocker is stabilizing the full phase-2
network replay or replacing it with a constrained implicit/network-only
reference before q-angular-tolerance convergence and statistical-pipeline
refresh.

2026-05-22 BD88 adaptive full phase-2 replay substep retry: the BD87
full-network accepted-background replay now retries failed accepted solver
steps with doubled production/destruction substeps up to
`max_substeps_per_solver_step=4096`, records attempted and failed substep
counts, retry reason counts, completed substep totals, and keeps failed
solver-sample substeps out of the committed replay reference state.  The
trace-tail replay remains applied-step-only, while the full-network replay uses
all accepted solver step samples when the trace-tail corrector is enabled.
This remains existing FB69 telemetry, not a standalone readiness gate, public
backend, QKE path, output truncation, or full independent BBN proof.

The q4/mu5 adaptive-replay endpoint probe
`/tmp/rabbit_bd88_phase2_full_network_adaptive_replay_q4_mu5_fb69.json`
(`artifact_payload_sha256=e7301496d6fd050d43da6000d1da35092502be8683076e98a2b17583fb90c03a`,
file SHA256 `6cac028a246aa6ae5cf86bd6024117fd0f543a2e5c40d6f1e8f03175b3dbc4ce`)
passed the main FB69 endpoint row with `N_final=4.8`, `step_count=373`,
`attempt_count=397`, and `n_rejected=24`, but did not improve the full phase-2
replay coverage: it still completed only `2/373` requested accepted steps and
stopped at `N_end=0.08719626309362064`.  The failed replay step was retried to
`attempted_substep_count_max=4096` with
`adaptive_substep_retry_count=10`, `failed_substeps=4096`, and
`adaptive_substep_retry_reason_counts={"full phase2 reference abundance blow-up limit exceeded": 10}`.
BD88 self-audit: `real_blocker_moved=substep refinement to 4096 was tested and
ruled out as sufficient for the independent per-species replay`;
`gate_removed_or_consolidated=existing FB69 telemetry, no standalone gate`;
`raw_state_preserved=raw vectors, raw negative Rodas samples, accepted
start/end samples, retry reason counts, and partial replay deltas`;
`verification=BD87/BD88 helper tests, partial-substep regression, py_compile,
and real q4/mu5 endpoint probe`; remaining blocker is a coupled conservative
implicit/network-only reference, not another retry-budget increase.

2026-05-22 BD89 conservative directional-extent full phase-2 replay: FB69 now
adds a second full nine-species accepted-background replay that applies each
forward and reverse nuclear reaction as a coupled directional extent in
`Y_i=X_i/A_i` abundance-per-baryon space using the in-tree stoichiometry and
PRIMAT fluxes.  Each extent is limited by currently available reactants before
the stoichiometric update is applied, and weak `n<->p` conversion is applied as
bounded directional extents from accepted-background live weak rates.  The
reference records completion fraction, extent-limited counts, final
solver-minus-reference deltas, and mass/charge deltas alongside the independent
per-species replay failure.  This remains existing FB69 telemetry, not a
standalone readiness gate, public backend, QKE path, output truncation, or full
independent BBN proof.

The q4/mu5 conservative-replay endpoint probe
`/tmp/rabbit_bd89_phase2_conservative_extent_replay_q4_mu5_fb69.json`
(`artifact_payload_sha256=99e8acdafdae36018a50373824a2c5015d87a253997b59437e696b655dee3ac3`,
file SHA256 `f7a7e96b90ee476bc5db8f266f87f8195fd6f95d68e89aa924bff15494146022`)
passed the main FB69 endpoint row with `N_final=4.8`, `step_count=373`,
`attempt_count=397`, `n_rejected=24`, and finite endpoint readouts
(`Yp=0.16318718435928586`, `D/H=2.0956470941271044e-05`,
`N_eff_3T=2.978160043133659`).  The old independent full-network replay still
failed at `2/373` steps, but the conservative directional-extent replay
completed `373/373` requested accepted steps through `N_end=4.8`, with
`extent_limited_count=2438`, `weak_extent_limited_count=0`,
`mass_fraction_delta=-5.191330698650631e-10`,
`charge_fraction_delta=0.09192104724730205`,
`final_abs_delta_max=0.18387853358706596`, and
`step_abs_delta_max=0.5079428702787598`.  BD89 self-audit:
`real_blocker_moved=coupled conservative stoichiometric extent reference
completed the full 373-step accepted-background endpoint row`;
`gate_removed_or_consolidated=existing FB69 telemetry, no standalone gate`;
`raw_state_preserved=raw solver vectors, raw negative Rodas samples, accepted
start/end samples, independent replay failure telemetry, and conservative
replay deltas`; `verification=conservative extent unit test, focused tests,
py_compile, and real q4/mu5 endpoint probe`; remaining blocker is the endpoint
scale disagreement between the solver-corrected path and conservative replay,
so the next stage should run q/angular/tolerance comparisons and consider a
conservative directional-extent solver corrector rather than the current
trace-tail-only Patankar corrector.

2026-05-22 BD90 full phase-2 conservative candidate-corrector probe: FB69 now
exposes `phase2_conservative_extent_corrector_enabled` and
`--phase2-conservative-extent-corrector` as an opt-in private candidate
corrector for all nine phase-2 species.  It reuses the BD89 conservative
directional-extent update, preserves raw Rodas candidate values and raw
negative samples, and records attempt/accepted/rejected/skip counts plus
mass/charge/extent-limited telemetry on the existing FB69 row.  The landed
corrector is `domain_rescue_only`: candidates already inside the phase-2 trace
domain are left as the high-order Rodas candidate, and the conservative
replacement is attempted only when the raw candidate would otherwise violate
the trace-domain guard.  Existing trace-tail Patankar behavior remains
backward-compatible and separate.

The q4/mu5 unconditional conservative-corrector probe
`/tmp/rabbit_bd90_phase2_conservative_extent_corrector_q4_mu5_fb69.json`
(`artifact_payload_sha256=3368dca01eed79f78d9232bc8be67e0bb631b95df84764e3aca3e330edcf9e59`,
file SHA256 `43eec085d21742d56ccd53e6c9a5bcc09279cad247eb133cfd04a7fe003690cb`)
accepted only one tiny step (`N_final=2.2122913581559576e-07`) before
fail-closing, with attempted conservative corrections as large as
`0.43162265524573284`.  That ruled out unconditional all-species replacement
under the current correction-error controller.  The rescue-only probe
`/tmp/rabbit_bd90_phase2_conservative_extent_rescue_q4_mu5_fb69.json`
(`artifact_payload_sha256=e7287f18d1d9ea54bf64ad1457c06dc618e58de6d36e04dabd89d0de1766956a`,
file SHA256 `f54c5b507a6e524280b7526d7f43ba94cb2364b14b2681bba1578b4dba5f2b0c`)
advanced to `N_final=2.952937223169137` with `512` accepted steps before
fail-closing on `max_steps`; it attempted `43` conservative raw-negative
rescues, all rejected by the correction/error machinery.  BD90 self-audit:
`real_blocker_moved=conservative extent update was tested inside the solver
candidate path and restricted to domain rescue after unconditional replacement
proved too intrusive`; `gate_removed_or_consolidated=existing FB69 telemetry
and CLI, no standalone gate`; `raw_state_preserved=raw Rodas candidates, raw
negative samples, corrected values, and rejected corrections`; `verification=
BD90 unit/integration tests, py_compile, q4/mu5 unconditional and rescue-only
probes`; remaining blocker is step-policy/convergence interpretation, with the
trace-tail solver path still the currently passing q4/mu5 route and
conservative replay retained as a finite comparison diagnostic.

2026-05-22 BD91 FB70 phase-2 telemetry fold-in: the existing FB70 full-BBN
span ladder now carries the FB69 phase-2 replay/corrector telemetry instead of
requiring a separate FB69-only inspection path.  Selected rows, h-refinement
attempt rows, resolution summaries, and adjacent resolution diagnostic deltas
preserve full phase-2 replay status, conservative extent replay status,
trace-tail Patankar counters, conservative-corrector counters, and raw
candidate negative/domain/corrector samples.  This did not add a standalone
gate; it consolidated the already existing phase-2 solver evidence into the
full-span comparison surface while keeping raw negative evidence and
untruncated observables.

BD91 self-audit: `real_blocker_moved=FB70 can now compare phase-2 replay and
candidate-corrector diagnostics across span/resolution rows`;
`gate_removed_or_consolidated=existing FB69 telemetry folded into FB70, no new
gate`; `raw_state_preserved=raw negative/domain/corrector samples are copied
through FB70 rows and h-refinement attempts`; `verification=FB70 focused tests,
FB69/FB70/WBS/registry focused suite, sync_test_counts.py, py_compile, and
git diff --check`; remaining blocker is moving from telemetry comparison to
endpoint-backed q/angular/tolerance convergence and reducing the phase-2
solver/corrector discrepancy.

2026-05-22 BD92 FB70 positive solver-coordinate routing: FB70 now forwards the
existing FB69 `trace_log_solver_coordinates_enabled` and
`temperature_log_solver_coordinates_enabled` private solver options through the
span ladder, resolution ladder, freedom-composition rows, and CLI.  Rows,
h-refinement attempts, summaries, and claim boundaries preserve whether the
positive-coordinate solver path was requested and how many Rodas5P steps used
positive log solver coordinates.  The flags remain private solver-coordinate
controls; they do not truncate reported abundances or temperatures, do not
open public dispatch, and do not change QKE scope.

A real CPU-JAX/Rodas5P FB70 smoke with `frozen_source_jax`, boundary trace,
trace-boundary abundance policy, trace+temperature log solver coordinates, and
a tiny weak-only span passed as hot-endpoint evidence at
`T_gamma=0.7999999999606833 MeV` with
`positive_log_solver_coordinate_step_count_total=1`,
`temperature_log_solver_coordinate_step_count_total=1`,
`trace_log_solver_coordinate_step_count_total=0` because no trace abundances
were above the active log-coordinate floor in that tiny initial span, and
`artifact_payload_sha256=e5fd3d462617cb096a0acf3b96405475a2342b4c5d2fba804f40b756c44a1aef`.
BD92 self-audit: `real_blocker_moved=endpoint-capable positive solver
coordinates can now be exercised by FB70 full-span and resolution surfaces`;
`gate_removed_or_consolidated=existing FB70 surface extended, no standalone
gate`; `raw_state_preserved=raw FB69 rows, h-refinement attempts, and
untruncated observables remain embedded`; `verification=red-first FB70
forwarding regression, FB70 CLI dry-run coverage, full FB70 focused tests, and
real CPU-JAX/Rodas5P smoke`; remaining blocker is q/angular/tolerance
convergence of the endpoint-reaching trace-tail solver route and reconciliation
against conservative phase-2 replay/corrector diagnostics.

2026-05-22 BD93 phase-2 replay residual comparison: FB69 now surfaces
trace-tail, independent full-network, and conservative directional-extent
window-reference mass/charge residual maxima in row summaries, and FB70 folds
those selected residuals into resolution rows and adjacent phase-2 diagnostic
deltas.  This keeps the conservative replay discrepancy on the existing
full-span/resolution comparison surface instead of adding another gate; raw
solver rows, raw negative candidate samples, and untruncated endpoint
observables remain the source of truth.

BD93 self-audit: `real_blocker_moved=conservative replay disagreement is now
visible across FB70 q/angular/tolerance axes as mass/charge residual deltas`;
`gate_removed_or_consolidated=existing FB69 telemetry folded into existing FB70
resolution comparisons, no standalone gate`; `raw_state_preserved=raw rows and
candidate samples remain embedded, no output truncation or sign repair`;
`verification=red-first FB69/FB70 telemetry regressions, focused RHS/FB70 tests,
py_compile, and git diff --check`; remaining blocker is endpoint-backed
q/angular/tolerance convergence of the trace-tail route and reducing the
conservative replay versus solver endpoint discrepancy.

2026-05-22 BD94 Rodas5P block-max adaptive error norm: FB69 now uses the
maximum RMS over existing state blocks (`geometry_thermo`, `A_modes_flat`, and
`X_phase2`) as the host Rodas5P acceptance norm instead of the full packed-state
scalar RMS.  The component scale remains physical `atol + rtol*|y|`; no
species-specific error weights or per-species tolerances are introduced.  Error
diagnostics preserve both `scalar_rms_norm` and
numeric `acceptance_norm`, with the policy string recorded separately as
`acceptance_norm_policy=block_max_rms_over_state_blocks`, so old dilution
evidence is not hidden.

A real tiny CPU-JAX/Rodas5P FB69 smoke with `frozen_source_jax`, boundary trace,
trace-boundary abundance policy, trace+temperature log solver coordinates, and
no collision payload passed with
`artifact_payload_sha256=2b7bb44362ebf966da27f2fc0ed96378f3393ae239b485c066cc349bd4e02311`.
The accepted diagnostic sample had `dominant_block=X_phase2`,
`scalar_rms_norm=0.0006356929888458115`, and
`acceptance_norm=0.0014983427438853923`.

BD94 self-audit: `real_blocker_moved=phase-2 embedded error can no longer be
diluted by high-dimensional A-mode entries in adaptive acceptance`;
`gate_removed_or_consolidated=existing FB69 Rodas5P controller changed in place,
no standalone gate`; `raw_state_preserved=state vectors, raw candidates, and
untruncated observables remain unchanged while scalar-vs-block norms are
recorded`; `verification=red-first block-max norm regressions, focused
RHS/FB70/WBS/registry tests, py_compile, and git diff --check`; remaining
blocker is rerunning endpoint q/angular/tolerance ladders and conservative
replay residual comparisons under the stricter controller.

2026-05-22 BD95 all nonnegative trace-log solver coordinates: FB69 now encodes
every finite nonnegative constrained trace species `[T, He3, He4, Li7, Be7,
Li6]` as a positive log solver coordinate when the private trace-log option is
enabled with `trace_boundary`.  This supersedes the earlier active-floor-only
pilot: positive sub-active values such as `1e-30` decode back to their raw
physical abundances, exactly zero trace values are encoded at
`_TRACE_LOG_SOLVER_X_FLOOR` and counted by
`trace_log_solver_encoded_floor_count`, and negative/nonfinite accepted trace
states remain outside the transform so the raw domain evidence path is not
repaired away.

The prototype metadata now records
`trace_log_solver_coordinate_scope=all_nonnegative_trace_abundance_solver_state_with_zero_floor`
and
`trace_log_solver_inactive_species_policy=negative_or_nonfinite_trace_state_remains_direct_X_for_domain_evidence`.
The host Rodas5P base RHS/Jacobian is evaluated at the decoded solver-coordinate
base state, and initial RHS seed reuse is skipped when zero-floor encoding
changes that base state.
A real tiny CPU-JAX/Rodas5P FB69 smoke with `frozen_source_jax`, boundary trace,
trace-boundary abundance policy, trace+temperature log solver coordinates, and
no collision payload passed with
`artifact_payload_sha256=0b836a3cadda293cbc5d35054c2aedf4985f3a04ed129ec350de47e124f7c0dc`.
The row recorded `trace_log_solver_coordinate_step_count=40`,
`trace_log_solver_inactive_floor_count=0`,
`temperature_log_solver_coordinate_step_count=40`, and
`error_norm_max=4.626783530044199e-06`.

BD95 self-audit: `real_blocker_moved=zero and sub-active trace species no longer
remain in direct-X Rodas base/stage algebra`; `gate_removed_or_consolidated=existing
FB69 transform changed in place, no standalone gate`; `raw_state_preserved=raw
positive sub-active values decode unchanged, zero floors are counted, and
negative/nonfinite trace states remain evidence`; `verification=red-first
trace-transform/base-RHS regression, focused transform/host-step/builder tests,
real tiny FB69 smoke, focused RHS/FB70/WBS/registry tests, py_compile, and git
diff --check`; remaining blocker is endpoint
q/angular/tolerance rerun under the all-trace coordinate and block-max
controller.

2026-05-22 BD96 sub-active trace solver scale: FB69 keeps the BD95
all-nonnegative trace coordinate scope but stops positive `X=1e-30` seeds from
driving `dX/X` Rodas stage increments of order `1e13-1e14`.  Trace coordinates
now use `z = log1p(X / _TRACE_LOG_SOLVER_ACTIVE_FLOOR)`, with
`dX/dz = X + _TRACE_LOG_SOLVER_ACTIVE_FLOOR`, and the
RHS/Jacobian/error conversion uses that same derivative scale.  This keeps the
transformed ODE internally consistent while avoiding the artificial birth-trace
log singularity.  Decode maps extreme negative trace coordinates to a negative
active-floor abundance and extreme positive coordinates to `inf` without NumPy
exp overflow/underflow warnings, preserving recoverable stage-domain evidence
rather than silently truncating accepted output.

The concrete blocker moved: the q4/mu5 endpoint replay that failed immediately
after BD95 with `N_final=0` now reaches `N_final=4.8` on CPU-JAX/Rodas5P with
`frozen_source_jax_full_jvp`, `auto_small_collision_reuse`, trace+temperature
solver coordinates, trace-tail Patankar correction, and `max_steps=640`.
Artifact hash:
`ded80431a849cdf85c3dfb9854af30e4b33302949e378665dc35232cfebd8ac4`.
The row recorded `step_count=486`, `attempt_count=512`, `n_rejected=26`, no
violations, and conservative extent replay coverage to `N=4.8`.

BD96 self-audit: `real_blocker_moved=all-trace coordinate endpoint no longer
fails at the first step`; `gate_removed_or_consolidated=existing FB69
solver-coordinate conversion changed in place, no standalone gate`;
`raw_state_preserved=accepted states remain untruncated and rejected
overflow/underflow remains domain evidence`; `verification=red-first
sub-active log1p solver-scale, finite-difference transformed-Jacobian, and
safe-decode regressions plus real q4/mu5 endpoint
CPU-JAX/Rodas5P replay`; remaining blockers are step budget
(`486/512` accepted/attempted with `max_steps=640`) and the large full-network
conservative replay mismatch after early replay failure.

2026-05-22 BD97 conservative extent candidate activation: the existing private
FB69 `phase2_conservative_extent_corrector_enabled` path now applies a real
full nine-species conservative directional-extent candidate update instead of
skipping whenever the raw Rodas candidate is already inside the trace domain.
The candidate uses an 8-substep refined extent replay, records the 4-substep
coarse replay, and feeds the refined-minus-coarse local error into the adaptive
acceptance diagnostics under the `phase2_conservative_extent_corrector` block.
It also permits `z=-inf` trace log coordinates when they decode to the
safe-atol boundary value, while decoded nonfinite or below-boundary abundances
still fail closed.

BD97 evidence: the q4/mu5 CPU-JAX/Rodas5P conservative-candidate probe applies
the corrector once and rejects 11 conservative attempts before failing at
`N_final=2.22122087094942e-07`.  The accepted conservative step's extent replay
matches to `final_abs_delta_max=2.372171364812073e-10` and mass delta
`4.930823790466995e-16`; artifact hash
`b0cf136ef28e731d8b2daa0398b7a2b953e278ef46e21675c5af47518da6bfef`.
BD97 self-audit: `real_blocker_moved=conservative extent flag no longer no-ops
and exposes post-corrector Rodas trace-coordinate stage instability`;
`gate_removed_or_consolidated=existing FB69 corrector plumbing changed in
place`; `raw_state_preserved=raw Rodas, coarse extent, refined extent, and
local-error values are recorded separately`; no QKE, public dispatch, or
production support claim is made.

2026-05-22 BD98 conservative extent operator split: when the existing private
FB69 `phase2_conservative_extent_corrector_enabled` flag is active, the host
Rodas5P path now removes the full `X_phase2` abundance block from the Rodas
stage RHS and Jacobian, zeroing both the abundance RHS and the abundance
Jacobian rows/columns.  The conservative directional-extent candidate remains
the accepted-step abundance update and records raw Rodas candidate values,
coarse/refined extent values, and local-error values separately.

BD98 evidence: the q4/mu5 CPU-JAX/Rodas5P conservative probe no longer fails
after a single accepted step with nonfinite trace-log stage coordinates.  It
records `phase2_conservative_extent_corrector_accepted_count_total=400`,
`phase2_conservative_extent_corrector_rejected_count_total=240`,
`trace_abundance_domain_rejection_count=0`, and
`stage_temperature_domain_rejection_count=0` before exhausting the
`max_steps=640` attempt budget at `N_final=2.9920208477777853e-06`.  Artifact
hash: `5fdb6428def859cf5420649d1c0d83dfa0b8d834ed0366ca8676c436a489ca58`.
The same patch fails closed when operator split is enabled but the conservative
update is unavailable; the q4/mu5 probe recorded
`phase2_conservative_extent_corrector_missing_update_count=0`.
BD98 self-audit: `real_blocker_moved=Rodas no longer carries stiff X_phase2
stages in conservative mode`; `gate_removed_or_consolidated=existing FB69
corrector/Jacobian paths changed in place, no standalone gate`;
`raw_state_preserved=raw Rodas candidate and conservative local-error telemetry
remain visible`; remaining blocker is conservative network subcycle/local-error
control, not trace-domain stage rejection.

2026-05-22 BD99 phase-split hot weak candidate: the existing private FB69
`phase2_conservative_extent_corrector_enabled` path now keeps the full nuclear
network inactive only for proposed steps whose start and candidate temperatures
both remain above the configurable private
`phase2_network_activation_T_gamma_MeV` threshold; those hot steps apply an
exact frozen-background `n <-> p` weak update for the phase-1 part of the same
continuous run.  At, below, or crossing into the activation threshold, the
existing full phase-2 conservative directional-extent candidate remains active.
The default threshold remains `0.08 MeV` only as a private standard-BBN staging
default; the artifact records that strong anisotropy, dynamic collision
payloads, or distorted neutrino thermodynamics require model-specific
validation.  This changes the existing corrector in place and does not add a
readiness, manifest, hash, figure, claim, public-dispatch, production-support,
or QKE gate.

BD99 evidence: the q4/mu5 CPU-JAX/Rodas5P conservative probe advanced from the
BD98 `N_final=2.9920208477777853e-06` max-step exhaustion to
`N_final=2.6010666700970373` before exhausting `max_steps=640`.  It recorded
`phase2_conservative_extent_corrector_accepted_count_total=640`,
`phase2_conservative_extent_corrector_rejected_count_total=18`,
`trace_abundance_domain_rejection_count=0`,
`stage_temperature_domain_rejection_count=0`, and artifact hash
`07bb2d6c252f8d1903b3e81bb4fb2a94409c64bb3acb289fc163f52434c7637e`.
The last attempted state is at the activation boundary
(`T_gamma=0.07999656235483887 MeV`), with finite nonnegative accepted
abundances and mass-fraction residual `-6.38378239159465e-14`.  The artifact
also records
`phase2_network_activation_threshold_validation.validation_status=requires_model_specific_validation`,
`anisotropy_detected=true`, `dynamic_collision_payload_active=true`, and
`neutrino_temperature_distortion_detected=true`.  This is not endpoint support;
the remaining blocker is activation-boundary event handling plus a real
full-network implicit/embedded estimator before the `0.01 MeV` endpoint.
BD99 self-audit: `real_blocker_moved=hot-start full-network activation removed
from the conservative path`; `gate_removed_or_consolidated=existing FB69
corrector changed in place, no standalone gate`;
`raw_state_preserved=raw Rodas candidate and corrected weak/full-network
candidate telemetry remain separate`.

2026-05-23 BD100 activation-event step limiter: the existing private FB69
`phase2_conservative_extent_corrector_enabled` path now rejects and step-limits
a raw Rodas candidate that crosses the private
`phase2_network_activation_T_gamma_MeV` threshold before the full conservative
phase-2 network corrector is called.  The limiter records
`private_phase2_activation_event_step_limiter` samples with the raw start and
candidate temperatures, crossing fraction, threshold-validation payload, and
limited `h`; once the step base is within the event tolerance, the existing
full-network candidate path proceeds.  This is in-place solver-control work,
not a readiness, manifest, hash, figure, claim, public-dispatch,
production-support, or QKE gate.

BD100 evidence: the q4/mu5 CPU-JAX/Rodas5P conservative probe recorded
`artifact_payload_sha256=ff0310a516524f4fa419d3f4c7807d905309a409ca30237ee4ba5fbbab4b5882`,
`phase2_activation_event_limiter_count_total=5`,
`phase2_conservative_extent_corrector_accepted_count_total=640`,
`phase2_conservative_extent_corrector_rejected_count_total=13`,
`step_count=640`, `attempt_count=658`, `n_rejected=18`, and
`N_final=2.6010680162198607` before max-step exhaustion.  The dominant-error
counts are `phase2_activation_event_step_limiter=5`,
`phase2_conservative_extent_corrector=624`, `geometry_thermo=17`, and
`A_modes_flat=12`.  The full phase-2 reference is still unstable
(`completed_fraction_min=0.00625`, `replay_N_end_max=0.3072889717271672`,
`last_reference_abs_max=376427566798.0082`), so the remaining blocker is still
the post-activation full-network update/error estimator, not public endpoint
support.
BD100 self-audit: `real_blocker_moved=activation-crossing attempts are split
out before the full-network corrector, reducing conservative-corrector
rejections from BD99's 18 to 13 on the same probe`;
`gate_removed_or_consolidated=existing FB69 step-control/corrector path changed
in place, no standalone gate`; `raw_state_preserved=raw crossing candidates are
recorded in limiter samples and never hidden by accepted-output truncation`;
`next_blocker=NSE handoff plus MPRK/backward-Euler-style implicit network
candidate with a real embedded estimator`.

2026-05-23 BD101 deuterium bottleneck handoff seed: the existing private FB69
conservative phase-2 candidate now computes a post-consumption D handoff target
from the local PRIMAT R0 `n+p <-> D+gamma` forward/reverse flux balance once
the full-network branch is active and the non-n/p light species are still below
the private handoff seed threshold.  The seed solves the scalar R0 balance
after the equal `X_n`/`X_p` donor draw, raises only `X_D`, and preserves mass
and charge diagnostics; it does
not seed He4/Li/Be, does not repair negative inputs, and does not truncate
accepted outputs.  This is in-place solver/physics handoff work, not a
readiness, manifest, hash, figure, claim, public-dispatch, production-support,
or QKE gate.

BD101 evidence: the q4/mu5 CPU-JAX/Rodas5P conservative probe recorded
`artifact_payload_sha256=6b62fa643550233b3139fe07f127fd3287ea8b47cd7ef7cd1f7059c7a1b0a63c`,
`phase2_deuterium_handoff_seed_applied_count_total=12`,
`phase2_deuterium_handoff_seed_delta_abs_max=0.0007592006995322019`,
`phase2_activation_event_limiter_count_total=5`,
`phase2_conservative_extent_corrector_accepted_count_total=640`,
`phase2_conservative_extent_corrector_rejected_count_total=13`, and
`N_final=2.6023722966786704` before max-step exhaustion.  The full phase-2
reference remains unstable (`completed_fraction_min=0.00625`,
`replay_N_end_max=0.3072889717271672`,
`last_reference_abs_max=376427566798.0082`), so this is not endpoint support.
BD101 self-audit: `real_blocker_moved=activation handoff no longer starts from
pure D floor state, with N_final moving from BD100's 2.6010680162198607 to
2.6023722966786704`; `gate_removed_or_consolidated=existing FB69
conservative-candidate path changed in place, no standalone gate`;
`raw_state_preserved=raw candidate values and handoff seed deltas are recorded
separately, negative abundance inputs fail closed`; `next_blocker=positive
implicit/MPRK-style network candidate with a real embedded estimator`.

2026-05-23 BD102 positive implicit directional extents: the same private FB69
conservative phase-2 candidate now uses a positive implicit directional-extent
solve for each forward/reverse reaction direction instead of hard-clipping an
explicit reaction extent to the donor pool.  Common one-reactant and
two-reactant mass-action cases use analytic roots, higher-order reverse
directions use bounded scalar bisection, and the weak `n<->p` substep uses the
exact frozen-background two-state update.  The old hard-clipped helper remains
available for existing reference/test plumbing, but the accepted full-network
candidate path records `network_update_scheme=positive_implicit_directional_extent`
and `hard_extent_clipping_applied=false`.  This is in-place solver work, not a
readiness, manifest, hash, figure, claim, public-dispatch, production-support,
or QKE gate.

BD102 evidence: the optimized q4/mu5 CPU-JAX/Rodas5P conservative probe recorded
`artifact_payload_sha256=e6ffba6a04a3b47c2e9214266e14eab4094196307cbfb46b7df9a6cfbd78d3d0`,
`phase2_conservative_extent_corrector_accepted_count_total=640`,
`phase2_conservative_extent_corrector_implicit_extent_solved_count_total=119808`,
`phase2_conservative_extent_corrector_hard_clip_applied_count_total=0`,
`phase2_conservative_extent_corrector_weak_exact_update_count_total=4992`,
`phase2_deuterium_handoff_seed_applied_count_total=13`, and
`N_final=2.6010419227122354` before max-step exhaustion.  The all-bisection
variant reached only 565 accepted corrector steps in the same 180-second wall
budget, so analytic roots are part of the CPU-JAX/Rodas5P target.  The full
phase-2 reference remains unstable (`completed_fraction_min=0.00625`,
`replay_N_end_max=0.3072889717271672`,
`last_reference_abs_max=376427566798.0082`), so this is not endpoint support.
BD102 self-audit: `real_blocker_moved=hard donor clipping removed from the
accepted full-network candidate, with real coarse/refined local-error
differences`; `gate_removed_or_consolidated=existing FB69 corrector path changed
in place, no standalone gate`; `raw_state_preserved=raw candidate values remain
separate, negative/nonfinite network inputs fail closed`; `next_blocker=coupled
implicit/full-network solve or MPRK-grade estimator that improves physical burn
and reference stability`.

2026-05-23 BD103 adaptive internal positive-implicit pair budget: the same
private FB69 conservative phase-2 candidate now replaces the fixed `4/8`
positive-implicit coarse/refined pair with an adaptive internal pair that can
retry to a capped refined-substep budget.  The candidate uses physical-X RMS
scaling aligned with the Rodas phase-2 corrector block, records max scaled
component diagnostics separately, and caps the accepted path at `16` refined
substeps because cap-64 probes inflated runtime without meeting the target.
This changes the existing runtime solver/error-estimator path in place; it is
not a new readiness, manifest, hash, figure, publication, public-dispatch,
production-support, or QKE gate.

BD103 evidence: the cap-16 q4/mu5 CPU-JAX/Rodas5P conservative probe recorded
`artifact_payload_sha256=3b2f078bda56c50b0007b68402fcbfb191e0b2854f70468f59e59d94a30c825c`,
`phase2_conservative_extent_corrector_accepted_count_total=640`,
`phase2_conservative_extent_corrector_implicit_extent_solved_count_total=239616`,
`phase2_conservative_extent_corrector_internal_attempt_count_total=1872`,
`phase2_conservative_extent_corrector_internal_attempted_substeps_total=17472`,
`phase2_conservative_extent_corrector_internal_max_refined_substeps_reached_count_total=624`,
`phase2_conservative_extent_corrector_hard_clip_applied_count_total=0`,
`N_final=2.601051816017013`, and raw final
`X_He4=3.10302777320292e-08`.  This moves the accepted endpoint only
`~9.9e-06` beyond BD102 and roughly doubles the tiny He4 burn, while increasing
implicit extent work; cap-64 probes reached only `548--550` accepted steps in
the same 180-second budget.  BD103 therefore documents the next real blocker as
both the positive-implicit hot-loop cost and insufficient full-network burn,
not endpoint support.
BD103 self-audit: `real_blocker_moved=fixed full-network local-error pair is
replaced by an adaptive capped pair with measured runtime telemetry`;
`gate_removed_or_consolidated=existing FB69 corrector path changed in place, no
standalone gate`; `raw_state_preserved=raw candidates, corrected values,
local-error values, and max-refinement target misses remain visible`;
`next_blocker=coupled implicit or MPRK-grade network update that improves
physical burn without exploding the CPU-JAX/Rodas5P hot loop`.

2026-05-23 BD104 coupled BE/Newton phase-2 corrector: the same private FB69
conservative phase-2 candidate now stops using the positive implicit
directional-extent pair as the accepted activated network update.  The BD98
operator split remains in place, while the activated branch solves the
nine-species network as one backward-Euler/Newton block in abundance-per-baryon
variables and compares one full network substep against two half substeps.  The
internal error norm is block-max physical-X over n/p, burn, and trace species;
if the cap is reached without meeting target, the corrector returns a retryable
failure instead of accepting a target-miss candidate.  The old directional
extent routine remains available for diagnostic/reference paths only.

BD104 evidence so far: red-first BE/Newton unit regressions verify that a toy
coupled burn is solved without directional extents and that target-miss at cap
fails closed; conservative-candidate routing tests now assert the accepted
scheme is `coupled_backward_euler_newton`; the focused RHS test file passes.
The row/aggregate telemetry now includes Newton convergence, residual
evaluations, finite-difference Jacobian evaluations, linear solves, positivity
line-search counts, raw negative Newton-trial counts/minima/first-vector
samples, and max residual norms.  A long q4/mu5 CPU-JAX/Rodas5P endpoint/burn
probe has not yet been run for this code step.

BD104 self-audit: `real_blocker_moved=accepted activated network update is now
a coupled implicit solve instead of the BD103 target-miss directional-extent
pair`; `gate_removed_or_consolidated=existing FB69 corrector path changed in
place, no standalone gate`; `raw_state_preserved=raw Rodas candidate,
corrected values, BE local-error vectors, raw negative Newton-trial telemetry,
and mass/charge residuals remain visible`; `next_blocker=run q4/mu5 CPU-JAX
Rodas5P probe and tune Newton/subcycling from measured He4 burn, endpoint
progress, and Newton/Jacobian cost`.

2026-05-23 BD105 BDF2/Newton network subcycling: the accepted activated branch
keeps the BD104 coupled nine-species implicit solve but advances the network
substeps with a BE-started BDF2/Newton sequence instead of pure first-order
BE/Newton.  The same physical-X block-max full-versus-half error criterion is
used, but the accepted active-network corrector now reports
`network_update_scheme=coupled_bdf2_newton` and
`network_subcontroller=network_only_bdf2_newton_step_doubling`.  The old
directional extent code remains diagnostic/reference-only, and the BE/Newton
helper remains covered as the lower-order control.  This changes existing FB69
runtime solver code in place and adds no standalone gate, public dispatch,
production-support, publication, or QKE claim.

BD105 evidence: the q4/mu5 CPU-JAX/Rodas5P probe wrote
`/tmp/rabbit_bd105_bdf2_newton_q4_mu5.json` with
`artifact_payload_sha256=c4843cb29e7d70171ac270e4c6c536ae7cc1e0f4bc21eccb69cc3a33f34a58de`,
`N_final=2.6044412715585126`, raw final
`X_He4=0.00013650657657444313`,
`phase2_conservative_extent_corrector_internal_error_target_met_count_total=481`,
`phase2_conservative_extent_corrector_refined_substeps_max=64`,
`phase2_conservative_extent_corrector_newton_converged_count_total=23168`,
`phase2_conservative_extent_corrector_missing_update_count=126`, and
`wall_seconds_total=176.59483786707278`.  Relative to BD104's BE/Newton probe
(`N_final=2.601415834334738`, raw final `X_He4=4.0819020417956984e-06`,
`newton_converged_count_total=4808`, `wall_seconds_total=131.5501189000206`),
BD105 raises accepted He4 by about `33x` and moves the endpoint forward by
`~0.0030`, but at substantially higher Newton/Jacobian cost.  Endpoint below
`0.01 MeV` remains blocked.
BD105 self-audit: `real_blocker_moved=accepted activated network subcycling is
now second-order BDF2/Newton and produces materially larger burn on q4/mu5`;
`gate_removed_or_consolidated=existing FB69 corrector path changed in place, no
standalone gate`; `raw_state_preserved=raw candidates, BDF2 corrected values,
local-error vectors, Newton trial negativity telemetry, and mass/charge
residuals remain visible, including target-miss local-error telemetry`;
`next_blocker=reduce BDF2 finite-difference Newton
cost and activation-onset host retries through analytic/sparse Jacobian,
improved network estimator, or model-specific activation diagnostic`.

2026-05-23 BD106 analytic BDF2/Newton network Jacobian: the standard-network
BE/BDF2 Newton residual now uses an analytic 9x9 network Jacobian assembled in
abundance-per-baryon variables from the same directional forward/reverse flux
products as `compute_flux_components`, with finite-difference fallback telemetry
for non-standard flux functions.  This is an in-place runtime solver change on
FB69, not a standalone gate, public dispatch path, production-support claim,
publication claim, output truncation, or QKE path.

BD106 evidence: the q4/mu5 CPU-JAX/Rodas5P probe wrote
`/tmp/rabbit_bd106_analytic_jac_q4_mu5.json` with
`artifact_payload_sha256=000c434cf108be4d7c51713f64887e466c60e0c2180d2da69a0c85281f30277b`,
`N_final=2.6044412762134264`,
`phase2_conservative_extent_corrector_newton_residual_evaluation_count_total=69504`,
`phase2_conservative_extent_corrector_newton_finite_difference_residual_evaluation_count_total=0`,
`phase2_conservative_extent_corrector_newton_jacobian_evaluation_count_total=23168`,
`phase2_conservative_extent_corrector_missing_update_count=126`, and
`wall_seconds_total=144.12645068601705`.  Relative to BD105
(`newton_residual_evaluation_count_total=278016`,
`wall_seconds_total=176.59483786707278`), BD106 removes the network
finite-difference residual-evaluation cost and cuts Newton residual evaluations
by `4x`, but endpoint below `0.01 MeV` remains blocked.
BD106 self-audit: `real_blocker_moved=finite-difference 9x9 network Jacobian
hot loop removed for the standard network path`; `gate_removed_or_consolidated=
existing FB69 solver path changed in place, no standalone gate`; `raw_state_preserved=
raw Newton trial negativity telemetry and no-output-truncation policy unchanged`;
`next_blocker=activation-onset host retries/missing updates and 64-substep BDF2
controller pattern`.

2026-05-23 BD107 BDF2 activation-onset substep cap: the accepted activated
network branch keeps the BD98 operator split, BD105 BDF2/Newton update, and
BD106 analytic standard-network Jacobian, but raises the BDF2 refined-substep
cap from `64` to `128`.  This is an in-place solver-budget change against a
captured activation-onset target-miss sample, not a return to the directional
extent pair and not a standalone readiness/manifest/hash/figure gate.

BD107 evidence: the captured onset sample at
`h=0.00004710865069811873`, `T_gamma_MeV=0.08000000000135893`, and
`H_rate_s=0.0025614496214830976` missed target at cap `64`
(`local_error_internal_norm=0.5889751083490837`) but accepted at cap `128`
(`local_error_internal_norm=0.1456371126806533`) with zero
finite-difference Jacobian residual evaluations.  The q4/mu5 CPU-JAX/Rodas5P
probe wrote `/tmp/rabbit_bd107_bdf2_cap128_q4_mu5.json` with
`artifact_payload_sha256=62de50f337f277e715249b730c6ecfc1882c4961f32c65f2927561b6b4f04a7a`,
`N_final=2.6188010100730494`,
`phase2_conservative_extent_corrector_newton_residual_evaluation_count_total=168576`,
`phase2_conservative_extent_corrector_newton_finite_difference_residual_evaluation_count_total=0`,
`phase2_conservative_extent_corrector_missing_update_count=124`,
`phase2_conservative_extent_corrector_refined_substeps_max=128`, and
`wall_seconds_total=178.12946439301595`.  Relative to BD106, this moves
`N_final` forward by about `0.01436` but increases Newton residual evaluations
by about `2.4x` and runs close to the 180-second staging budget.
BD107 self-audit: `real_blocker_moved=one measured activation-onset BDF2 target
miss now accepts and q4/mu5 advances materially beyond BD106`; `gate_removed_or_consolidated=
existing FB69 solver path changed in place, no standalone gate`; `raw_state_preserved=
raw BDF2 one-full/two-half candidates, Newton negativity telemetry, and
mass/charge residuals remain surfaced with no output truncation`;
`next_blocker=reduce high network-substep startup cost near activation and add
model-specific activation diagnostics before endpoint support can be claimed`.

2026-05-23 BD108 BDF2 error-predicted substep jump: the accepted activated
network branch remains the coupled BE/BDF2/Newton network-only corrector, but
the adaptive pair now estimates the next refined substep count from the
observed physical-X block-max local-error ratio.  Large target misses can skip
obviously under-resolved intermediate powers of two while preserving the same
acceptance target.  This does not restore the directional-extent pair, does not
truncate raw abundance evidence, and does not add a standalone gate.

BD108 evidence: a red-first synthetic pair now follows `4,8,64,128` instead of
`4,8,16,32,64,128` and still accepts only on the `64/128` pair.  The captured
BD107 activation-onset sample accepts with `refined_substeps=128`,
`local_error_internal_norm=0.1456371126806533`,
`adaptive_internal_substep_jump_count=1`, `internal_attempt_count=4`, and
`internal_attempted_substeps_total=204`, down from six attempts and `252`
attempted substeps on the no-jump path.  The q4/mu5 CPU-JAX/Rodas5P probe wrote
`/tmp/rabbit_bd108_bdf2_jump_q4_mu5.json` with
`artifact_payload_sha256=267ded2b902cd016e1bbc5d7056b9907637073517540bc8a3642b0a0b4a0c464`,
`selected_wall_seconds_total=167.95700480300002`,
`selected_step_count_total=508`,
`selected_adaptive_attempt_count_total=643`,
`selected_phase2_conservative_extent_window_reference_replay_N_end_max=2.6198945026118343`,
and `passed=false`.  Relative to BD107, this saves about `10.2` seconds and
moves the conservative reference N-end slightly forward, but final BBN
observables remain missing and endpoint support is still blocked.
BD108 self-audit: `real_blocker_moved=BDF2 network-pair startup no longer walks
every low substep count when the observed error ratio predicts a much finer
pair`; `gate_removed_or_consolidated=existing FB69 solver path changed in
place, no standalone gate`; `raw_state_preserved=raw candidates, Newton
negativity telemetry, and no-output-truncation policy unchanged`;
`next_blocker=host retries/max-step exhaustion plus replacement of the hard
private activation threshold with a model-specific local activation diagnostic`.

2026-05-23 BD109 local activation diagnostic payload: the existing FB69
activation-validation payload now computes model-specific local bottleneck/burn
diagnostics at the runtime activation decision point.  The private `0.08 MeV`
temperature remains a fallback, but the payload role is now
`private_staging_default_with_local_diagnostics` and includes R0
`n+p -> D+gamma` forward/reverse fluxes, the R0 balance ratio, the BD101
detailed-balance deuterium target, downstream D-consuming / T-He3-He4-producing
reaction indices, downstream flux scale, and a dimensionless downstream-burn
change estimate when accepted-background `H_rate_s` and host step `h` are
available.  This is in-place runtime payload work on the activation blocker,
not a standalone gate or public-support claim.

BD109 evidence: the red-first activation validation regression now verifies
`local_activation_diagnostic.available=true`,
`scope=private_phase2_local_activation_diagnostic`, positive R0 forward/reverse
fluxes, finite nonnegative R0 balance and downstream burn measures, and
`uses_hardcoded_temperature_as_fallback=true` under anisotropy, dynamic AP65
collision payloads, and distorted neutrino temperatures.  The focused
BD99-BD101/BD109 activation-corrector subset passed.  BD109 self-audit:
`real_blocker_moved=local activation diagnostics are computed where the hard
temperature fallback is currently evaluated`; `gate_removed_or_consolidated=
existing FB69 activation payload changed in place, no standalone gate`;
`raw_state_preserved=diagnostic reads raw phase-2 abundances and fails closed
on nonfinite or negative inputs without repair`; `next_blocker=use the local
diagnostic to drive event step-limiting/full-network activation instead of the
hard fallback threshold`.

2026-05-23 BD110 model-specific activation policy: FB69 now uses the BD109 local
bottleneck diagnostic in the activated phase-2 branch decision.  The private
`0.08 MeV` temperature remains a fallback guard, but model-specific states with
`validation_status=requires_model_specific_validation` and an available local
diagnostic no longer activate the full network from that hard temperature alone;
the local deuterium-bottleneck policy must also be active.  The landed private
threshold is `deuterium_equilibrium_X_D >= 1e-3`, with downstream-burn
Damkohler telemetry retained.  Local-only event limiting and local early
activation were tested and rejected in this stage because they caused h-min
collapse or a worse early expensive burn window.

BD110 evidence: focused BD99-BD101/BD109/BD110 tests passed, and the q4/mu5
CPU-JAX/Rodas5P probe wrote
`/tmp/rabbit_bd110_model_specific_local_required_q4_mu5.json` with
`artifact_payload_sha256=eba9c86ded2347f1d8e9bca7ef96b2a48d9f0ec9609f835c547aae6de3b47c33`,
`selected_wall_seconds_total=180.05199413001537`,
`selected_step_count_total=433`,
`selected_adaptive_attempt_count_total=549`, and
`selected_phase2_conservative_extent_window_reference_replay_N_end_max=2.623157403134011`.
This is a small runtime-policy improvement over BD108's
`2.6198945026118343` reference-window N-end, not endpoint support.  BD110
self-audit: `real_blocker_moved=unconditional hard-0.08 activation is retired
for model-specific rows with local diagnostics`; `gate_removed_or_consolidated=
existing FB69 runtime activation/corrector code changed in place, no standalone
gate`; `raw_state_preserved=no abundance repair or observable truncation`;
`next_blocker=phase2-corrector-dominated host attempts, post-activation burn
convergence, and a robust continuous-extension local event solve`.

2026-05-23 BD111 host/network controller split: FB69 now treats successful
coupled BE/BDF2/Newton phase-2 network-corrector local error as host telemetry
instead of folding that already accepted network pair error back into the host
Rodas5P error norm.  Missing or failed network updates still reject the host
attempt, and the older trace-tail Patankar path keeps its previous host-error
control behavior.  The q4/mu5 CPU-JAX/Rodas5P probe wrote
`/tmp/rabbit_bd111_host_split_q4_mu5.json` with
`artifact_payload_sha256=17c01101f8ca0be576a4c3ebcd26a5224e15384da2876cc0f05da7007bf1ef36`,
file SHA256
`170e8d8754cb93cf4c7c2eba05695338d24f1db98e951b54c6f284e5b27212f3`,
`wall_seconds_total=180.08855348592624`, `N_final=2.630433999972621`,
`step_count=379`, `attempt_count=610`, and
`phase2_conservative_extent_corrector_internal_error_target_met_count_total=337`.
This moves the measured conservative-corrector window beyond BD110's
`2.623157403134011`, but still fails endpoint support; the next blocker is now
the BDF2/Newton subcontroller missing its target at the 128-substep cap on
post-activation burn-onset attempts.  BD111 self-audit:
`real_blocker_moved=accepted network-corrector local error is no longer
double-gated by host Rodas control`; `gate_removed_or_consolidated=existing FB69
runtime controller changed in place, no standalone gate`; `raw_state_preserved=
raw pair-error, raw candidate, and Newton negativity telemetry retained`;
`next_blocker=BDF2/Newton cap misses and post-activation burn stiffness`.

2026-05-23 BD112 BDF2/Newton refined-substep capacity: the accepted activated
network branch still uses the BD98 operator split, BD105 coupled BE/BDF2/Newton
network-only corrector, BD106 analytic standard-network Jacobian, BD108
error-predicted substep jump, BD110 model-specific local-bottleneck activation,
and BD111 host/network controller split.  BD112 changes the existing
BDF2/Newton network subcontroller capacity in place from `128` to `1024`
refined substeps.  It does not add a readiness, manifest, hash, figure,
publication, public-dispatch, production-support, or QKE gate, and it does not
restore the positive implicit directional-extent pair as the accepted activated
network solver.

BD112 evidence: the actual-code q4/mu5 CPU-JAX/Rodas5P probe wrote
`/tmp/rabbit_bd112_bdf2_cap1024_q4_mu5.json` with
`artifact_payload_sha256=b635f58048f1465363758ef25bc24444e1ade2bc83228472102a857d2e479b7d`,
file SHA256
`2de0f1f23e77598491b263c604692eb498918d61e7a3641572c6ef951a21fff4`,
`wall_seconds_total=180.06371211295482`, `N_final=2.7137160704223873`,
`step_count=132`, `attempt_count=202`, `n_rejected=70`,
`phase2_conservative_extent_corrector_internal_error_target_met_count_total=103`,
`phase2_conservative_extent_corrector_internal_max_refined_substeps_reached_count_total=0`,
and `phase2_conservative_extent_corrector_refined_substeps_max=1024`.  This is
still `passed=false`, but it moves the measured conservative-corrector window
beyond BD111's `2.630433999972621`.  A tested host retry-limiter variant was
discarded because its q4/mu5 artifact stopped at `N_final=2.629757148845655`,
and an actual-code cap-2048 comparison stopped lower than cap-1024 at
`N_final=2.7059926846321596`.  Later BD114 inspection showed rejected
missing-update attempts could still reach the `1024` cap under the stricter
`0.2` target, so the BD112 movement is bounded to accepted-corrector cap-hit
telemetry and N-window progress.  BD112 self-audit:
`real_blocker_moved=accepted-corrector BDF2 cap-hit telemetry is removed for the
measured q4/mu5 row and the live window advances`; `gate_removed_or_consolidated=existing
FB69 solver internals changed in place, no standalone gate`; `raw_state_preserved=
raw Rodas candidates, BDF2 one-full/two-half candidates, Newton negativity
telemetry, mass/charge residuals, and no output truncation retained`;
`next_blocker=post-activation burn stiffness/cost and legacy full-network window
reference blow-up`.

2026-05-23 BD113 full-network reference consolidation: the active FB69 artifact
now folds the legacy `phase2_full_network_window_reference` slot into the
accepted phase-2 BE/BDF2/Newton corrector replay instead of separately running
the older production-destruction full-network reference.  The older helper
remains available for focused historical unit tests, but the runtime artifact no
longer reports a contradictory legacy P/D blow-up for the same accepted window.
The nested payload records
`legacy_production_destruction_reference_replayed=false` and
`consolidated_from_scope=accepted_solver_background_phase2_corrector_replay`.

BD113 evidence: the q4/mu5 CPU-JAX/Rodas5P probe wrote
`/tmp/rabbit_bd113_consolidated_fullref_q4_mu5.json` with
`artifact_payload_sha256=94ddae4adf7804ff6e1130c1a7c214c7f8a61a5b01d0f9eaf2b516f2f5d23e0f`,
file SHA256
`a0164451b9555c75637824040f5f1128bb9700a21b45f88cbfb97aa8dac9b32b`,
`wall_seconds_total=180.2209013379179`,
`phase2_full_network_window_reference_replay_N_end_max=2.7137160704223873`,
`phase2_full_network_window_reference_completed_fraction_min=1.0`,
`phase2_full_network_window_reference_failed_substeps_max=0`,
`phase2_full_network_window_reference_unavailable_step_count_total=0`,
`phase2_full_network_window_reference_charge_delta_abs_max=2.2655005245807524e-06`,
and `phase2_full_network_window_reference_mass_delta_abs_max=9.11933235089174e-12`.
The artifact remains `passed=false`; endpoint support is not claimed.  BD113
self-audit: `real_blocker_moved=redundant legacy full-network reference blow-up
is folded into the accepted BDF2 replay surface`; `gate_removed_or_consolidated=
older reference plumbing consolidated, no standalone gate`; `raw_state_preserved=
raw candidate, BDF2 pair, Newton negativity, mass/charge, and no-output-
truncation telemetry retained`; `next_blocker=live solver post-activation burn
stiffness/cost and final-state nonfinite failure`.

2026-05-23 BD114 unit network-pair acceptance target: the accepted activated
phase-2 BE/BDF2/Newton network-only subcontroller now uses a block-max scaled
step-doubling target of `1.0` instead of the earlier staging safety factor
`0.2`.  The tolerance scales (`atol=1e-10`, `rtol=1e-8`) and positivity/Newton
policies are unchanged; this only changes the acceptance threshold to the
usual "within scaled tolerance" value for the network pair.  The q4/mu5
CPU-JAX/Rodas5P actual-code probe wrote
`/tmp/rabbit_bd114_target1_actual_q4_mu5.json` with
`artifact_payload_sha256=8c7b705587c866b0060fb7f5e45ea9effae0d2c766684c6e735f7400a65172be`,
file SHA256
`3491173a785674c380f562d6c0dac65799222012ad41a9d4e7b60b9f38ec1630`,
`wall_seconds_total=178.70334624999668`, `passed=true`, `violations=[]`,
`N_final=4.8`, `T_gamma_MeV=0.00914475962100043`,
`phase2_conservative_extent_corrector_local_error_internal_norm_max=0.9887712594583102`,
`phase2_conservative_extent_corrector_raw_newton_trial_negative_count_total=0`,
`phase2_conservative_extent_window_reference_replay_N_end_max=4.8`, and
`phase2_full_network_window_reference_replay_N_end_max=4.8`.  The private
backbone-network endpoint readouts were `Yp=0.16971384189564434`,
`D/H=1.8870272438777542e-05`, `N_eff_3T=2.9781565403951245`, and
`Sigma_H=0.11432002150599589`.  These are diagnostic staging readouts for one
q4/mu5 `n_reactions=12` row, not public production support, not QKE, not a
31-reaction claim, and not a statistical-pipeline result.  BD114 self-audit:
`real_blocker_moved=first private q4/mu5 CPU-JAX/Rodas5P continuous AP65 row
reaches N=4.8 below 0.01 MeV with endpoint observables`; `gate_removed_or_consolidated=
existing BDF2/Newton subcontroller acceptance scaling changed in place, no
standalone gate`; `raw_state_preserved=raw candidates, BDF2 pair, local-error,
Newton negativity, mass/charge, and no-output-truncation telemetry retained`;
`next_blocker=q/angular/tolerance ladders, no-collision/collision comparisons,
31-reaction checks, plot inputs, and statistical pipeline artifacts`.

2026-05-23 BD115 frozen standard-network kinetic cache: the accepted activated
phase-2 BE/BDF2/Newton network-only subcontroller now precomputes frozen
standard-network kinetic factors once per accepted-background network problem
and reuses them in the coupled Newton residual and analytic Jacobian.  The
cache covers the standard flux-components path, `T_gamma`, `eta`, the density
factor, stoichiometry/product powers, forward/reverse rates, and reaction
symmetry factors.  It does not change the acceptance target, tolerances,
positivity line search, Rodas5P host policy, raw state telemetry, QKE scope, or
public-support claim.  The pre-cache private q4/mu5 `n_reactions=31`
CPU-JAX/Rodas5P row wrote `/tmp/rabbit_bd115_n31_q4_mu5.json` with
`artifact_payload_sha256=dd25bf0565f026e2627a71819b95f4d00821dfb583efa5702312ad242cd15628`,
file SHA256
`02f87154ef7d988b8f43237bee352c9f48232b7b4ee17a2718551083e819d48b`,
`wall_seconds_total=180.1826181249926`, `passed=false`, and
`N_final=2.8310100852364353`.  The cached row wrote
`/tmp/rabbit_bd115_cached_n31_q4_mu5.json` with
`artifact_payload_sha256=59f08367c0cc6093e31178a47372832ba084ee3a7293f516922b2e9121d76a91`,
file SHA256
`f64106736099f71a26fa6c803ab7fcf2b7592a60a6a4b8ef4d2d65948c53566c`,
`wall_seconds_total=163.8800985190319`, `passed=true`, `violations=[]`, and
`N_final=4.8`.  The private 31-reaction endpoint readouts were
`Yp=0.1697138421195527`, `D/H=1.8871137096284503e-05`,
`N_eff_3T=2.9781565403951245`, and `Sigma_H=0.11432002150599589`.  These are
diagnostic staging readouts for one q4/mu5 31-reaction row, not public
production support, not QKE, and not a statistical-pipeline result.  BD115
self-audit: `real_blocker_moved=the same private q4/mu5 31-reaction row moves
from wall-time failure at N=2.8310100852364353 to endpoint-backed N=4.8 within
the 180-second staging budget`; `gate_removed_or_consolidated=existing
BE/BDF2/Newton standard-network residual and analytic-Jacobian internals changed
in place, no standalone gate`; `raw_state_preserved=raw candidates, BDF2 pair,
local-error, Newton negativity, mass/charge, and no-output-truncation telemetry
retained`; `next_blocker=no-collision/collision comparisons, q/angular/tolerance
ladders, plot inputs, and statistical pipeline artifacts`.

2026-05-23 BD116 network-pair-scoped kinetic cache: the frozen standard-network
kinetic cache now lives at the BE/BDF2/Newton adaptive network-pair scope rather
than the individual coarse/refined step-attempt scope.  One accepted-background
full-vs-half network pair therefore shares the same precomputed `T_gamma`,
`eta`, density, stoichiometry, symmetry, and forward/reverse rate factors across
coarse, refined, and jumped retry attempts.  The accepted candidate, tolerance,
positivity line search, raw Newton telemetry, Rodas5P host policy, QKE scope,
and public-support boundary are unchanged.  The micro-regression
`test_bd116_network_pair_shares_kinetic_cache_across_attempts` verifies a
standard-network BDF2 adaptive pair performs two attempts while calling
`evaluate_nuclear_rates(...)` exactly once.  The q4/mu5 31-reaction
CPU-JAX/Rodas5P endpoint check wrote
`/tmp/rabbit_bd116_paircache_n31_q4_mu5.json` with
`artifact_payload_sha256=a093111eb3dd3d18bf2beb9780cd2e2b6ae3626a45326f732b0224f3a60a4c51`,
file SHA256
`8c6074be7b62f021e71e9b786a66ca29701daeb97e7cd8a55de925c6af2dfe57`,
`wall_seconds_total=164.2399338139221`, `passed=true`, `violations=[]`, and
`N_final=4.8`.  This is wall-time neutral relative to BD115
`163.8800985190319`, so the honest conclusion is that repeated rate evaluation
was removed but the dominant hotspot remains BDF2/Newton residual, Jacobian, and
linear-solve volume.  The private 31-reaction endpoint readouts remain
`Yp=0.1697138421195527`, `D/H=1.8871137096284503e-05`,
`N_eff_3T=2.9781565403951245`, and `Sigma_H=0.11432002150599589`.  BD116
self-audit: `real_blocker_moved=standard-network rate evaluation is now shared
across coarse/refined/jumped attempts inside one frozen-background network
adaptive pair`; `gate_removed_or_consolidated=existing BE/BDF2/Newton
standard-network cache internals changed in place, no standalone gate`;
`raw_state_preserved=raw candidates, BDF2 pair, local-error, Newton negativity,
mass/charge, and no-output-truncation telemetry retained`; `next_blocker=
BDF2/Newton residual/Jacobian/linear-solve volume plus no-collision/collision
comparisons, q/angular/tolerance ladders, plot inputs, and statistical pipeline
artifacts`.

2026-05-23 BD117 accepted Newton line-search residual reuse: the coupled
BE/BDF2/Newton network solve now carries an accepted line-search
residual/payload/norm into the next Newton iteration instead of recomputing the
same residual at the same accepted trial state.  This changes no Newton update,
line-search positivity policy, residual tolerance, network cache, Rodas5P host
policy, raw negative telemetry, QKE scope, or public-support claim.  The
regression `test_bd117_newton_reuses_accepted_line_search_residual` checks the
non-duplicated residual-count formula, and the pair-cache regression now checks
nonzero BDF2 pair-level residual reuse telemetry.  The no-collision pre-BD117
row `/tmp/rabbit_bd117_nocollision_n31_q4_mu5.json` wrote
`artifact_payload_sha256=7e67fff47fc9ae3f50b5dea44bb1c3c3509c1c2b29505c7efb643ba3800ad12f`,
file SHA256
`d6b849933196d76873105833582dd9934e4f8df8786e9480f83c80f07bcdbf97`,
`passed=true`, `N_final=4.8`, `wall_seconds_total=103.17029682197608`, and
`phase2_conservative_extent_corrector_newton_residual_evaluation_count_total=292066`.
The BD117 no-collision row `/tmp/rabbit_bd117_resreuse_nocollision_n31_q4_mu5.json`
wrote
`artifact_payload_sha256=987499e09d652232473fc9dc2a1aa4a503189c649203479c6a36ac5aba89fbbe`,
file SHA256
`95a2d446632b15ddc6439c6674317778e12cf92e4ba4e77cf5ca2815e1052070`,
`passed=true`, `N_final=4.8`, `wall_seconds_total=94.74744493700564`, and
`phase2_conservative_extent_corrector_newton_residual_evaluation_count_total=194709`.
The BD117 dynamic-collision row `/tmp/rabbit_bd117_resreuse_dynamic_n31_q4_mu5.json`
wrote
`artifact_payload_sha256=459543b0e69cffcf939717c0addfba7c300debda451d646cf6787ce86bd2ff73`,
file SHA256
`68d85fb1aec3d5cb573535d541542928acec7ff4049aaf72610c5f13a071ffe1`,
`passed=true`, `N_final=4.8`, `wall_seconds_total=154.66314127494115`, and
`phase2_conservative_extent_corrector_newton_residual_evaluation_count_total=194710`.
The private q4/mu5 31-reaction endpoint comparison after BD117 is:
no-collision `Yp=0.16972783156110663`,
`D/H=1.8904206719153318e-05`, `N_eff_3T=2.993397912593453`,
`Sigma_H=0.11432002114948502`; dynamic collision
`Yp=0.1697138421195527`, `D/H=1.8871137096284503e-05`,
`N_eff_3T=2.9781565403951245`, `Sigma_H=0.11432002150599589`.  BD117
self-audit: `real_blocker_moved=network residual evaluations drop by about
one third in both no-collision and dynamic-collision endpoint rows`;
`gate_removed_or_consolidated=existing BE/BDF2/Newton residual reuse internals
changed in place, no standalone gate`; `raw_state_preserved=raw candidates,
BDF2 pair, local-error, Newton negativity, mass/charge, and no-output-
truncation telemetry retained`; `next_blocker=dynamic collision payload cost,
q/angular/tolerance ladders, plot inputs, and statistical pipeline artifacts`.

2026-05-23 BD118 vectorized analytic standard-network Jacobian assembly: the
accepted activated BE/BDF2/Newton network branch keeps the same BDF2 pair
target, positivity line search, kinetic cache, Rodas5P host policy, raw
negative telemetry, QKE exclusion, and public-support boundary, but replaces
scalar Python derivative loops inside the standard-network analytic Jacobian.
Forward mass-action derivatives use indexed vector accumulation, while reverse
photodissociation derivatives use the positive-state
`prod(Y**nu) * nu_i / Y_i` identity; nonpositive trials retain the previous
fallback loop.  The new regression
`test_bd118_full_standard_network_analytic_jacobian_matches_finite_difference`
checks the full 31-reaction analytic Jacobian against the finite-difference
residual Jacobian on the activation sample.  The no-collision BD118 row
`/tmp/rabbit_bd118_vecjac_nocollision_n31_q4_mu5.json` wrote
`artifact_payload_sha256=d542413e503e1236a160b1d92b7277043ab801c2688750a1dd853edead0d1a7c`,
file SHA256
`46081b5c344e4244647ff5e5d135b4cf93d8735726aaf6c1f5cfc2f8a6d80a02`,
`passed=true`, `N_final=4.8`, `wall_seconds_total=69.74548954004422`,
`phase2_conservative_extent_corrector_newton_residual_evaluation_count_total=194709`,
and
`phase2_conservative_extent_corrector_newton_jacobian_evaluation_count_total=97357`.
The dynamic-collision BD118 row
`/tmp/rabbit_bd118_vecjac_dynamic_n31_q4_mu5.json` wrote
`artifact_payload_sha256=49273ad03bf195881c8dd32ad41ac2cb4ee40c8bbf79979fb84edbf9f4f2e14d`,
file SHA256
`7625b204bada0d1bab0b5bba2534a59e03700b58d20fbfd20fac99a586f5d4f8`,
`passed=true`, `N_final=4.8`, `wall_seconds_total=130.993984925095`,
`phase2_conservative_extent_corrector_newton_residual_evaluation_count_total=194710`,
and
`phase2_conservative_extent_corrector_newton_jacobian_evaluation_count_total=97358`.
Relative to BD117, the no-collision row wall time drops from
`94.74744493700564` to `69.74548954004422`, and the dynamic-collision row wall
time drops from `154.66314127494115` to `130.993984925095`, with unchanged
Newton counts and stable endpoint readouts (`Yp=0.16972783156110663`,
`D/H=1.8904206719153318e-05`, `N_eff_3T=2.993397912593453`,
`Sigma_H=0.11432002114948502` for no collision; `Yp=0.1697138421195527`,
`D/H=1.8871137096284503e-05`, `N_eff_3T=2.9781565403951245`,
`Sigma_H=0.11432002150599589` for dynamic collision).  BD118 self-audit:
`real_blocker_moved=standard-network analytic Jacobian assembly no longer pays
scalar Python-loop cost in the Newton hot loop`; `gate_removed_or_consolidated=
existing BE/BDF2/Newton analytic-Jacobian internals changed in place, no
standalone gate`; `raw_state_preserved=raw candidates, BDF2 pair, local-error,
Newton negativity, mass/charge, and no-output-truncation telemetry retained`;
`next_blocker=dynamic collision current-state source/payload cost plus
q/angular/tolerance ladders, plot inputs, and statistical pipeline artifacts`.

2026-05-23 BD119 full standard-network defaults for continuous endpoint
surfaces: after BD115-BD118 showed that the 31-reaction q4/mu5
CPU-JAX/Rodas5P endpoint row reaches `N_final=4.8` for both no-collision and
dynamic-collision probes, the private continuous-AP65 endpoint surfaces now
default to the full 31-reaction in-tree standard network instead of the older
12-reaction backbone.  The changed surfaces are the FB69 source-RHS prototype
builder/CLI, the FB70 full-BBN span-ladder builder/CLI, and the consolidated
span-experiment CLI.  Explicit `n_reactions=12` remains available for
backbone smoke/reference comparisons, but default endpoint probes no longer
silently emit backbone-network rows.  Dry-run checks show
`scripts/run_augmented_continuous_ap65_source_rhs_prototype.py --dry-run
--skip-reference` reports `inputs.n_reactions=31`,
`claim_scope=private_continuous_ap65_rhs_prototype_only`,
`public_dispatch_ready=false`, and `qke_scope=out_of_scope`, while
`scripts/run_augmented_continuous_ap65_full_bbn_span_ladder.py --dry-run`
reports `inputs.n_reactions=31`,
`claim_scope=private_continuous_ap65_full_bbn_span_ladder_only`,
`public_dispatch_ready=false`, and `qke_scope=out_of_scope`.  BD119
self-audit: `real_blocker_moved=default continuous endpoint rows now use the
full 31-reaction standard network unless the caller explicitly opts into
12-reaction backbone mode`; `gate_removed_or_consolidated=existing FB69/FB70
and span CLI/builder defaults changed in place, no standalone gate`;
`raw_state_preserved=no solver state, positivity policy, raw-candidate
telemetry, BDF2/Newton pair evidence, mass/charge residual, or no-output-
truncation behavior changed`; `next_blocker=q/angular/tolerance ladder
evidence, plot inputs, statistical-pipeline wiring, and dynamic current-state
source/payload cost`.

2026-05-23 BD120 endpoint resolution-ladder observable deltas: the existing
FB70 resolution/tolerance ladder now accepts and propagates `Sigma_H` and
`N_eff_3T` terminal tolerances alongside `Yp`, `D/H`, and `T_final_MeV`.
This changed the current ladder comparison surface in place rather than adding
a standalone gate.  A real q4/mu5 CPU-JAX/Rodas5P no-collision 31-reaction
h-ladder artifact
`/tmp/rabbit_bd120_h_ladder_nocollision_n31_q4_mu5.json`
(`artifact_payload_sha256=3ba5b11a0fbca767d1c2a3864e280c9cff8240fc1d45701d809ce7e09fe19dda`,
file SHA256
`bff0d14f43737f232150c2770e55a4d4c6f597865807e26a1cf55258d10adf80`)
reached the full-BBN endpoint in both rows but failed strict abundance
tolerance: `passed=false`, `physical_full_bbn_span_ready=true`,
`abs_delta_Yp=0.0019889505191167944`, `abs_delta_D/H=1.0123718999476457e-06`,
`abs_delta_T_final_MeV=6.792542743550012e-11`,
`abs_delta_Sigma_H=8.739525769740908e-11`, and
`abs_delta_N_eff_3T=2.2088260687169736e-07`.  BD120 self-audit:
`real_blocker_moved=endpoint h-ladder can now separate abundance convergence
failure from stable background/Sigma/N_eff deltas`;
`gate_removed_or_consolidated=existing FB70 resolution ladder changed in
place, no standalone gate`; `raw_state_preserved=raw child span rows,
terminal observables, adjacent deltas, phase-2 diagnostic deltas, and
no-output-truncation policies remain embedded`; `next_blocker=abundance
convergence under h/tolerance refinement, especially Yp and D/H across
h_max=0.1 to 0.05, plus dynamic-collision cost and broader q/angular ladders`.

2026-05-23 BD121 weak-only hot-phase step-averaged rates: the existing FB69
private phase-2 corrector now evolves the pre-activation weak-only hot branch
with start/candidate trapezoid-averaged per-`N` weak coefficients,
`0.5 * (lambda/H)_start + 0.5 * (lambda/H)_candidate`, while preserving the
old start-only policy as recorded fallback when candidate weak-rate or Hubble
payloads are unavailable.  The activated branch remains the network-only
BDF2/Newton subcontroller; no burn variable was moved back into the host
Rodas stage algebra.  On the same q4/mu5 CPU-JAX/Rodas5P no-collision
31-reaction h-ladder as BD120, artifact
`/tmp/rabbit_bd121_full_h_ladder_nocollision_n31_q4_mu5.json`
(`artifact_payload_sha256=598c6f449505cf070fd15cc90f3c21e8e3a77010a6486fb9cb5e73e7f4013342`,
file SHA256
`6afcdc4edf116dc8ffca257e4a719f295615162b3c4f841f5aee70a76614c4f7`)
reached endpoint in both rows with `physical_full_bbn_span_ready=true`,
`rows_full_bbn_completed=2`, and `failed_or_exception_rows=0`.  The ladder
still failed strict terminal tolerance only through `D/H`:
`abs_delta_Yp=0.00014099252824212316`,
`abs_delta_D/H=1.0804977852498688e-06`,
`abs_delta_T_final_MeV=6.797451317097636e-11`,
`abs_delta_Sigma_H=8.747548518872605e-11`, and
`abs_delta_N_eff_3T=2.210797784840679e-07`.  BD121 self-audit:
`real_blocker_moved=BD120 Yp h-delta reduced from 1.9889505191167944e-3 to
1.4099252824212316e-4 on the same endpoint ladder`;
`gate_removed_or_consolidated=existing FB69 corrector internals changed in
place, no standalone gate`; `raw_state_preserved=raw Rodas candidates,
start/candidate weak payloads, per-N weak coefficients, phase-2 samples,
mass/charge residuals, and no-output-truncation policies remain embedded`;
`next_blocker=activated deuterium burn/network subcontroller D/H convergence,
dynamic-collision cost, and broader q/angular/freedom ladders`.

2026-05-23 BD122 activated network step-averaged background: the existing FB69
activated phase-2 corrector now feeds the network-only BDF2/Newton
subcontroller a start/candidate compressed frozen background instead of the
host-step start sample alone.  The network sample uses geometric-midpoint
`T_gamma`, a harmonic effective Hubble rate for the nuclear `dN/H` factor, and
BD121-style trapezoid weak rates per `N`, with recorded fallback policies when
candidate data are unavailable.  On the same q4/mu5 CPU-JAX/Rodas5P
no-collision 31-reaction h-ladder as BD121, artifact
`/tmp/rabbit_bd122_full_h_ladder_nocollision_n31_q4_mu5.json`
(`artifact_payload_sha256=0a17d0f614d16b34919c8a7fda1ce88f6e8ddf2e9efdd55f386ec5fdf9327216`,
file SHA256
`e25ef34d3d73b73bffd503fad0d05abd6372c60333fd70818c7eaa9664114dc2`)
passed with `physical_full_bbn_span_ready=true`, `rows_full_bbn_completed=2`,
`failed_or_exception_rows=0`, and `violations=[]`.  Adjacent deltas were
`abs_delta_Yp=0.00020171573490621042`,
`abs_delta_DH=3.194970124558984e-08`,
`abs_delta_T_final_MeV=5.963938023989535e-11`,
`abs_delta_Sigma_H=1.8041833305115773e-10`, and
`abs_delta_N_eff_3T=1.9735568912437884e-07`.  BD122 self-audit:
`real_blocker_moved=BD121 D/H h-delta reduced from 1.0804977852498688e-6 to
3.194970124558984e-8 and the existing strict q4/mu5 no-collision endpoint
ladder now passes`; `gate_removed_or_consolidated=existing FB69 corrector and
FB70 resolution-ladder surfaces changed in place, no standalone gate`;
`raw_state_preserved=raw Rodas candidates, phase-2 local errors, mass/charge
residuals, and no-output-truncation artifact policies remain embedded; the
start/candidate/effective background contract is recorded by the corrector
payload and locked by focused regressions`; `next_blocker=carry endpoint
resolution through broader q/angular/freedom matrices, then address
dynamic-collision/non-LRS payload cost and validation`.

2026-05-23 BD123 activated network background policy: the same FB69 activated
phase-2 corrector and BE/BDF2/Newton network-only subcontroller now accept an
explicit `phase2_network_background_policy`.  The default
`effective_midpoint` policy preserves BD122's geometric-`T`/harmonic-`H`/
trapezoid-weak sample; opt-in `substep_loglinear_midpoint` evaluates
log-linear `T_gamma(theta)`, linear-in-`1/H` `H(theta)`, and per-`N` weak-rate
interpolation at `theta=(k+1/2)/m` inside each attempted network substep, with
node-local standard-network kinetic factors fed into the existing analytic
BDF2/Newton residual/Jacobian.  Focused red-first tests verify that BDF2
substeps consume node-specific `T/H/weak/kinetic` values, that the activated
candidate passes a loglinear-background factory only when the opt-in policy is
selected, and that FB70 forwards the policy through its CLI/builder.  BD123
self-smokes passed bounded q4/mu5 31-reaction no-collision activation windows:
default `effective_midpoint` reached `T_gamma=0.07653738276516263 MeV` at
`N_span_end=2.65`, and opt-in `substep_loglinear_midpoint` reached
`T_gamma=0.0796750449846388 MeV` at `N_span_end=2.605`; both remain
hot-endpoint private diagnostics, not endpoint/full-BBN readiness.  BD123
self-audit:
`real_blocker_moved=the activated network now has an executable runtime policy
to compare BD122 one-background updates against substep-varying background-node
updates, directly targeting the operator-split background-compression issue
identified after BD122`;
`gate_removed_or_consolidated=existing FB69 corrector internals changed in
place, no standalone gate`; `raw_state_preserved=raw Rodas candidates, BDF2
pair candidates, local errors, Newton negativity, mass/charge residuals, and
no-output-truncation telemetry retained`; `next_blocker=measure
effective-midpoint versus substep-loglinear background deltas on no-collision,
dynamic-collision, and non-LRS rows; if substep-loglinear remains tiny-step
sensitive near activation, feed it Rodas dense/stage background or add an AB
predictor only as Newton initialization before promotion beyond stress mode`.

2026-05-23 BD124 AB2 predictor Newton initializer: the accepted activated
network branch keeps the BD98 operator split, CPU-JAX/Rodas5P host step, and
BE/BDF2/Newton network-only corrector, but adds opt-in
`phase2_network_newton_initial_guess_policy=ab2_rhs_predictor`.  The AB2
formula is used only to prepare the Newton initial guess on post-startup BDF2
network substeps; accepted values still come from the coupled implicit
BE/BDF2/Newton solve and step-doubling local-error check.  Raw negative AB2
predictor values are recorded and rejected before the initial guess falls back
to the current state, and nonnegative AB2 guesses must pass displacement and
residual preflight guards; this is not output truncation and not an ABM accepted
solver.  Focused tests lock AB2 initializer use, raw-negative predictor
rejection, residual/displacement guard rejection, conservative-candidate policy
pass-through, and FB70 pass-through.  BD124
self-audit:
`real_blocker_moved=existing BDF2/Newton corrector can now test guarded AB2
predictor initialization without moving X_phase2 back into host Rodas algebra`;
`gate_removed_or_consolidated=existing FB69/FB70 runtime path changed in
place, no standalone gate`; `raw_state_preserved=raw Rodas candidates, BDF2
pair candidates, Newton trial negativity, AB2 predictor negative values,
mass/charge residuals, and no-output-truncation policies remain preserved`;
`runtime_evidence=q4/mu5 no-collision activation smoke completed with
ab2_rhs_predictor after displacement guarding but was not faster than the
current_state control`; `next_blocker=benchmark dynamic-collision activation
rows, then return to endpoint/full-run blocker reduction`.

2026-05-23 BD125 dynamic-collision background-policy resolution: FB69/FB70 now
accept `phase2_network_background_policy=auto_dynamic_effective_midpoint`.
The requested policy resolves to `effective_midpoint` when dynamic AP65
collision payloads are active, and to `substep_loglinear_midpoint` otherwise.
This preserves BD123's no-collision operator-split stress comparison while
avoiding the measured dynamic-collision tiny-step collapse from using
substep-loglinear background nodes in the active AP65 payload row.  The patch
records requested/effective/auto-resolution fields in FB69 and FB70 inputs,
prototype metadata, row metadata, and claim boundaries, while leaving the
explicit `substep_loglinear_midpoint` policy available for stress tests.
Focused tests lock dynamic and non-dynamic auto resolution, FB70 propagation,
and CLI dry-run acceptance.  Probe evidence before landing showed
`substep_loglinear_midpoint` stalling near `N=2.6149` even with `max_steps=96`,
whereas `effective_midpoint` plus
`stage_collision_payload_policy=auto_small_collision_reuse` passed the same
q4/mu5 31-reaction dynamic activation window to
`T_final_MeV=0.07657924201846118` with 46 payload builds and 413 reuses; the
landed auto-policy smoke reproduced that with
`artifact_payload_sha256=f7dff35b4f4caaab83653b727ce3ce9be7688c2655f8a9d973f46e98f690beac`.
BD125 self-audit:
`real_blocker_moved=dynamic-collision activation rows now have an auto policy
that selects the measured stable background policy without hiding the
substep-loglinear stress mode`;
`gate_removed_or_consolidated=existing FB69/FB70 runtime policy path changed in
place, no standalone gate`; `raw_state_preserved=raw Rodas candidates, network
pair candidates, Newton trial negativity, mass/charge residuals, and
no-output-truncation policies remain preserved`; `next_blocker=extend the
dynamic-collision hot endpoint toward full endpoint and compare resolution
ladders before plot/statistical claims`.

2026-05-23 BD126 chain max-step retry budget handoff: FB70 now accepts
`chain_max_steps_policy=recovered_max_steps_floor`.  When chain restart handoff
is enabled and a window passes only after a larger `max_step_retry_factor`, the
selected `max_steps` cap is recorded as the floor for later chained windows.
The default remains `fixed`, so older retry behavior is unchanged unless the
caller opts in.  The policy records `base_max_steps`, `effective_max_steps`,
`input_max_steps_floor`, `input_max_steps_floor_source`,
`output_max_steps_floor`, and `output_max_steps_floor_source` in the existing
restart-handoff payload.

BD126 is motivated by the BD125 follow-up all-freedom dynamic chain probe:
with `max_steps=160` and no retry the row to `N_span_end=3.2` hit the host
Rodas5P max-step cap, while the same row with
`max_step_retry_factors=1.0,4.0` passed to
`T_final_MeV=0.045260353575832024`
(`artifact_payload_sha256=92d91d35e22d28372547d1fbd8824156bb9d0770156bb3871f8f7a208b2aebb1`,
`max_step_retry_rows_recovered=1`, selected trace-domain rejection count zero).
With the landed `recovered_max_steps_floor` policy, the same private
all-freedom dynamic setup extended to `N_span_end_ladder=2.65,3.2,3.8,4.4,4.8`
passed with
`artifact_payload_sha256=35b1be98847dfcc19ef7fb37807313b71c9c7271e18d2f2a195cc54ef2568683`,
`physical_full_bbn_span_ready=true`, terminal
`T_final_MeV=0.009144759648108285`, raw `Yp=0.1633688431175918`, raw
`D/H=2.095575624748707e-05`, and zero selected trace-domain/stage-projection/
raw-candidate-negative events.  This is private q4/mu5 endpoint smoke evidence
only, not public-production, publication, statistical-pipeline, or QKE support.
BD126 self-audit:
`real_blocker_moved=post-activation chained windows can inherit a recovered
host max-step budget instead of repeating the same low-budget failure, and the
private all-freedom dynamic q4/mu5 smoke reaches below 0.01 MeV`;
`gate_removed_or_consolidated=existing FB70 chain/retry runtime policy changed
in place, no standalone gate`; `raw_state_preserved=raw Rodas candidates,
network pair candidates, Newton trial negativity, mass/charge residuals, and
no-output-truncation policies remain preserved`; `next_blocker=compare
dynamic-collision/non-LRS resolution ladders and build endpoint-backed plot
inputs before statistical claims`.

2026-05-23 BD127 resolution-ladder chain max-step provenance: the existing
FB70 resolution-ladder artifact now preserves per-case
`chain_max_steps_policy` in `inputs.resolution_ladder_cases[*]`.  The nested
runtime already honored the policy, but the artifact input block only recorded
`chain_h_max_policy`, making endpoint resolution probes less reproducible from
their own metadata.  The issue was exposed by the private all-freedom dynamic
q4/mu5 endpoint h-resolution probe
(`artifact_payload_sha256=f49dbc4d3e569e4d7bf44bb102d36bd2ddc8bf90baedd0484cb58312b1a0c608`),
which completed both h cases below `0.01 MeV` with
`resolution_tolerance_ready=true`, `max_abs_delta_Yp=0.00019559700227750332`,
and `max_abs_delta_DH=1.2282171793308707e-08`.  BD127 self-audit:
`real_blocker_moved=endpoint h-resolution evidence is now reproducible from the
artifact input block`; `gate_removed_or_consolidated=existing FB70
resolution-ladder metadata changed in place, no standalone gate`;
`raw_state_preserved=raw nested span rows and raw observables remain embedded,
with no output truncation`; `next_blocker=extend dynamic endpoint evidence to
q/angular/default-context ladders and endpoint-backed plot inputs`.

2026-05-23 BD128 resolution-readiness geometry endpoint defaults: FB70 already
computed adjacent resolution deltas for `Yp`, `D/H`, `T_final_MeV`, `Sigma_H`,
and `N_eff_3T`, but its default `resolution_terminal_tolerances` only checked
the abundance and temperature fields.  The private dynamic all-freedom q4/q5
endpoint probe exposed this fail-open default:
`artifact_payload_sha256=edebfdc4093d329a48ccd0130e87b3b8b90a41ec0e5eb068d04083e7a5e58d2d`
completed both cases below `0.01 MeV` and reported
`resolution_tolerance_ready=true` while `max_abs_delta_Sigma_H` was
`0.024309377733043078`.  BD128 widens the existing default tolerance map in
place to include `Sigma_H` and `N_eff_3T`, so future default q/angular
resolution artifacts fail closed when geometry or effective-neutrino endpoints
do not converge even if `Yp`, `D/H`, and `T_final_MeV` do.  BD128 self-audit:
`real_blocker_moved=default endpoint-resolution readiness now covers geometry
and N_eff endpoints, preventing abundance-only convergence overclaim`;
`gate_removed_or_consolidated=existing FB70 tolerance defaults changed in
place, no standalone gate`; `raw_state_preserved=raw nested span rows,
terminal observables, and adjacent deltas remain embedded with no truncation`;
`next_blocker=resolve dynamic q-grid geometry convergence before publication,
statistics, public dispatch, production SMC validation, or QKE claims`.

2026-05-23 BD129 Gauss-Laguerre q-grid resolution inputs: FB70
`resolution_ladder_cases` now accept `q_laguerre_order` as an alternative to
explicit `q_nodes` and `q_energy_weights`.  The existing resolution surface
generates Gauss-Laguerre nodes plus AP65 energy weights `w exp(q) q^3`, records
`q_energy_weight_source`, and rejects mixed automatic/explicit q-grid inputs.
Generated nodes, base weights, and energy weights are finite-checked, so
nonfinite or overflowing automatic q-grid inputs fail closed instead of being
clipped under the same source label.
This moves the q-grid geometry blocker exposed by BD128 away from unrelated
hand-weighted grids and toward comparable quadrature-order ladders.  A tiny
CPU-JAX/Rodas5P smoke above the endpoint
(`artifact_payload_sha256=bd521fa521ab4459801a14f013dae1d52dccd251b2d6b550468393a6eca72555`)
used non-LRS geometry only, no dynamic collision, boundary trace, phase-2
conservative corrector, and `q_laguerre_order=3,4` over
`N_span_end_ladder=0.005`; both rows reached `completed_hot_endpoint`, with
`resolution_terminal_delta_violations=[]`,
`max_abs_delta_Sigma_H=4.761351568224881e-09`, and
`max_abs_delta_N_eff_3T=2.2932766796657233e-12`.  This is not full endpoint
validation.  BD129 self-audit:
`real_blocker_moved=q-grid resolution cases can now use a consistent
Gauss-Laguerre energy quadrature family`; `gate_removed_or_consolidated=existing
FB70 resolution input handling changed in place, no standalone gate`;
`raw_state_preserved=raw q nodes, q energy weights, nested rows, and adjacent
deltas remain embedded`; `next_blocker=re-run dynamic all-freedom endpoint
q/angular ladders with Gauss-Laguerre q orders and resolve any remaining
Sigma_H convergence gap`.

2026-05-23 BD131 Laguerre raw-weight contract preservation: the external audit
identified q-Laguerre dynamic rows as a likely source/Jacobian stiffness
blocker and noted that a raw-vs-energy quadrature mismatch must be falsified
before treating high-q behavior as physical.  FB70 `q_laguerre_order` rows now
preserve raw Gauss-Laguerre weights as `q_laguerre_weights` with
`q_laguerre_weight_source=gauss_laguerre_raw_w`, while keeping AP65 energy
weights as `q_energy_weights` with
`q_energy_weight_source=gauss_laguerre_w_exp_q_q3`.  A focused FD moment
regression checks that the energy weights reconstruct
`7*pi^4/120` for `int q^3/(exp(q)+1) dq`, and adjacent q-grid comparisons treat
raw weight changes as q-grid changes.  BD131 self-audit:
`real_blocker_moved=q-grid artifacts now preserve enough raw/energy quadrature
data to audit dynamic high-q source budgets`; `gate_removed_or_consolidated=the
existing FB70 resolution artifact changed in place, no standalone gate`;
`raw_state_preserved=raw nested rows, q nodes, raw Laguerre weights, AP65 energy
weights, terminal observables, and adjacent deltas remain embedded`;
`next_blocker=add q-node source-budget concentration diagnostics and run dynamic
q-Laguerre endpoint ladders under existing structured/JVP Jacobian policies`.

2026-05-23 BD132 live RHS q-node energy concentration diagnostics:
`_live_source_rhs_vector` now records q-node energy-density concentration in
metadata payloads when metadata is requested:
`q_energy_density_total`, `q_energy_density_max_fraction`,
`q_energy_density_argmax`, `q_energy_density_highest_q_fraction`, and
`q_energy_density_weighted_q_mean`.  The calculation uses the reconstructed
distribution, AP65 energy weights, and angular weights; it is skipped on
`return_metadata=False` hot-loop calls and does not change Rodas5P stages or
accepted states.  BD132 self-audit:
`real_blocker_moved=dynamic q-grid artifacts can now report whether source
stress is concentrated in one q node or the highest-q node`;
`gate_removed_or_consolidated=existing live-source RHS metadata changed in
place, no standalone gate`; `raw_state_preserved=raw RHS state, q grids,
weights, and untruncated observables remain unchanged`;
`next_blocker=run dynamic q-Laguerre endpoint rows under structured/JVP
Jacobian policies and classify high-q concentration versus source/Jacobian cost`.

2026-05-23 BD133 q-node concentration summary surfacing: FB69
`rhs_stress_summary` now includes the BD132 q energy-density concentration
fields, and FB70 selected-row summaries expose
`rhs_stress_q_energy_density_max_fraction_max`,
`rhs_stress_q_energy_density_argmax_max`,
`rhs_stress_q_energy_density_highest_q_fraction_max`, and
`rhs_stress_q_energy_density_weighted_q_mean_max`.  BD133 self-audit:
`real_blocker_moved=endpoint/resolution artifacts can directly report high-q
energy-budget concentration`; `gate_removed_or_consolidated=existing FB69/FB70
stress summaries changed in place, no standalone gate`;
`raw_state_preserved=raw RHS metadata and nested rows remain embedded`;
`next_blocker=run dynamic q-Laguerre endpoint rows under structured/JVP
Jacobian policies and choose source-weighting versus source/Jacobian fixes from
the recorded concentration signal`.

2026-05-23 BD134 dynamic Laguerre full-JVP auto policy: FB70
resolution-ladder cases now auto-resolve generated Gauss-Laguerre q-grid rows
with active dynamic collision terms from requested `frozen_source_jax` to
effective `frozen_source_jax_full_jvp`, unless the individual case explicitly
sets `jacobian_policy`.  Rows and input cases record
`jacobian_policy_requested`, effective `jacobian_policy`,
`jacobian_policy_auto_resolved`, and `jacobian_policy_auto_reason`.  The local
short-span q4 dynamic probes motivating this policy showed full-JVP at 2
selected steps and 18 dynamic payload builds in about 52.2 s versus block-JVP
at 24 selected steps and 264 payload builds in about 66.9 s, with q energy
concentration peaking at q-index 2 rather than the highest node.  Mixed
freedom-composition ladders preserve the requested case-wide policy because the
nested artifact currently receives one Jacobian policy for collision and
no-collision controls.  BD134
self-audit: `real_blocker_moved=default dynamic q-Laguerre resolution rows now
use the measured faster full-JVP policy`; `gate_removed_or_consolidated=existing
FB70 resolution policy changed in place, no standalone gate`;
`raw_state_preserved=requested/effective Jacobian policy, q grids, raw Laguerre
weights, AP65 energy weights, and nested rows remain recorded`;
`next_blocker=run endpoint dynamic q-Laguerre all-freedom rows under full-JVP
and classify source weighting, background quadrature, or Jacobian/source
hot-loop work from the resulting q concentration and payload/Jacobian counts`.

2026-05-23 BD135 raw q-weight runtime plumbing: generated q-Laguerre
resolution rows now pass `q_laguerre_weights` into FB69, FB69 forwards those
raw weights into the CPU-JAX live-source replay, and the replay layer stores
them on `AugmentedNonLRSSourceGridJax` while forwarding them through dynamic
non-LRS/LRS collision payload refresh.  AP65 energy weights remain the weights
for energy-density/stress/weak-moment reductions; raw q weights are used for
radial/collision quadrature builders when supplied, with legacy reconstruction
from `q_energy_weights / (exp(q) q^3)` retained only as fallback.  BD135
self-audit: `real_blocker_moved=dynamic q-Laguerre source refresh now consumes
preserved raw q weights directly`; `gate_removed_or_consolidated=existing
FB70/FB69/JAX replay plumbing changed in place, no standalone gate`;
`raw_state_preserved=q nodes, AP65 energy weights, raw q weights, dynamic
collision payload provenance, and nested rows remain recorded`;
`next_blocker=re-run endpoint dynamic q-Laguerre all-freedom rows under full-JVP
and classify source/Jacobian cost versus background quadrature versus true
q-grid physics stiffness`.

AP6 also has a direct non-LRS standard-3T radial span-profile artifact and CLI.
The first longer diagnostic run over `N_span_end=(1e-14,1e-12,1e-8)` at
`0.8/0.79/0.78 MeV` with LSODA reported `all_success=true`, nfev
`7/635/5577`, and the same `8.673617379884035e-19` maximum closure residual.
The adjacent AP6 source-policy span-profile artifact now runs the same
standard-3T closure under both frozen-initial-state and live-RHS source update
policies.  The first LSODA run over `N_span_end=(1e-14,1e-12)` reported
matching frozen/live nfev `7/635`, live source evaluations `9/637` under a
`2048` evaluation budget, and closure residual
`8.673617379884035e-19`, while thermo/network terminal deltas stayed at
roundoff scale.
The same direct and source-policy artifacts now classify live-RHS source-budget
exhaustion as structured JSON rather than aborting with a traceback.  A
diagnostic `N_span_end=(1e-14,1e-12,1e-10)` run with budget `4096` preserved the
two shorter live-RHS successes, recorded one `source_evaluation_budget_exceeded`
row at `1e-10`, and kept the frozen-source `1e-10` row as a measured reference
with nfev `11728`.

### Status Update Rule

Every PR in this ledger must update its row in the same PR that changes
code.  A PR is incomplete if code changes land without:

- this table updated,
- `ROADMAP_STATE_OF_RECORD.md` updated when capability/status changes,
- `ROADMAP_PR_CATALOG.md` appended at merge/close,
- generated capability docs refreshed through the registry scripts when
  a public capability changes.

---

## 5. Validation Gates

### 5.1 Decomposition

- LRS modes are even `ell`, `m=0`.
- non-LRS diagonal modes include both `Pi_+` and `Pi_-` sectors.
- `Sigma_-=0` non-LRS output matches the LRS path.

### 5.2 Distribution

- `0 <= f <= 1` by construction.
- FD equilibrium is represented exactly at `A=0`.
- Projection/reconstruction error is measured for every `ell_max`.

### 5.3 Convergence

Run at minimum:

```text
ell_max = 2, 4, 6, 8
```

and extend to `10` where the `8 -> 10` delta is not below tolerance.
Report:

- `Y_p`,
- `D/H`,
- `N_eff`,
- `Pi_+`,
- `Pi_-`,
- weak rates,
- collision residual norms,
- angular tail norms.

### 5.4 Collision

- `C[f_FD] = 0` at common temperature.
- Energy and number residuals remain within documented tolerances.
- Fixed quadrature is deterministic.
- Sampling, if enabled later, converges to deterministic quadrature and
  is exactly replayable for the same seed/config.

### 5.5 Backend

- SciPy reference must pass before JAX is wired.
- JAX CPU parity must pass before XLA/GPU runtime gates are considered.
- Any public dispatch remains `candidate` until all convergence and
  conservation gates are recorded.

---

## 6. Current Implementation Notes

Landed augmented-PSTF substrate modules and tests:

- `src/rabbit/transport/angular_decomposition.py`
- `src/rabbit/transport/augmented_pstf_distribution.py`
- `src/rabbit/transport/augmented_typeI_observables.py`
- `src/rabbit/transport/augmented_typeI_collisionless.py`
- `src/rabbit/transport/augmented_typeI_nonlrs_collisionless.py`
- `src/rabbit/transport/augmented_typeI_weak_network.py`
- `src/rabbit/transport/augmented_collision_bridge.py`
- `src/rabbit/validation/augmented_convergence.py`
- `scripts/run_augmented_3t_convergence_artifact.py`
- `src/rabbit/collisions/deterministic_reference.py`
- `src/rabbit/weak/augmented_bridge.py`
- `tests/test_angular_decomposition.py`
- `tests/test_augmented_pstf_distribution.py`
- `tests/test_augmented_typeI_observables.py`
- `tests/test_augmented_typeI_collisionless.py`
- `tests/test_augmented_typeI_nonlrs_collisionless.py`
- `tests/test_augmented_typeI_weak_network_bridge.py`
- `tests/test_augmented_typeI_weak_network_solve.py`
- `tests/test_augmented_typeI_weak_network_3t_solve.py`
- `tests/test_augmented_convergence.py`
- `tests/test_deterministic_collision_reference.py`
- `tests/test_augmented_weak_bridge.py`

AP1 provenance note added:

- `docs/audit/v4_derivations/typeI_augmented_pstf_noqke.md`

The scaffold is deliberately non-invasive.  It introduces the data
contracts required by the new surface without changing existing solver
dispatch, capability claims, or generated documentation.
