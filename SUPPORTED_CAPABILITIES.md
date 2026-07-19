<!-- BEGIN:SC_HEADER -->
# RABBIT v1.0.0: Supported Capabilities

**Single sources of truth** (two registries):
- **Backend dispatch**: `src/rabbit/config/backend_capabilities.py`
- **Feature maturity**: `src/rabbit/config/feature_capabilities.py`

All maturity claims in this file MUST match the `tier` field in code.

**Generated scope**: This entire document is registry-generated via `render_capability_tables.py --apply` and self-heals on each release. If content disagrees with code, the code is correct and this file is wrong.

## Maturity Tiers

| Tier | Meaning | Gate requirement |
|---|---|---|
| **canonical** | Runtime-regression reference path | Runtime gold; publication gate separate |
| **candidate** | Functional, partially validated | Component + dispatch + partial BBN |
| **substrate** | Module-level validated, not wired end-to-end | Import/component tests only |

## Backend Maturity (from `backend_capabilities.py`)
<!-- END:SC_HEADER -->

<!-- BEGIN:BACKEND_TABLE -->
| Backend | Tier | Surface class | Description |
|---|---|---|---|
| `auto` | **canonical** | canonical | Regression-locked SciPy Type-I reference path |
| `scipy` | **canonical** | canonical | Regression-locked SciPy Type-I reference path |
<!-- END:BACKEND_TABLE -->

<!-- BEGIN:SC_CANONICAL_CORE -->
## Active Runtime Core and Frozen Oracles (NOT publication validation)

| Capability | Backend | Regime |
|---|---|---|
| FLRW BBN baseline | SciPy Radau | CL0-CL3 |
| Type I anisotropic BBN | SciPy reference | historical regression envelope; publication validation open |
| Weak rates (Born-Sirlin) | Python | CL0-CL2 |
| 9-species network | PRIMAT AC2024 | full + backbone (31/12) |

## Feature Registry (by surface class)

Generated from `feature_capabilities.py`.
<!-- END:SC_CANONICAL_CORE -->

<!-- BEGIN:FEATURE_TABLE -->
### Canonical

| Feature | Evidence | Blockers |
|---|---|---|
| FLRW BBN baseline | Runtime regression evidence only: Tier-1 FLRW gold and historical cros... | — |
| Type I anisotropic BBN | Historical SciPy/JAX Type-I regression evidence exists, but the finite... | — |
| Weak rates (Born-Sirlin-FM) | Runtime weak-rate regressions exist; Stage-I electron collision public... | — |
| 9-species nuclear network | PRIMAT AC2024 runtime network and backbone regression; BBN publication... | — |

### Candidate-Strong

| Feature | Evidence | Blockers |
|---|---|---|
| Tilted scalar BBN | Type I BBN-verified; v0=0 recovery 6.9e-6; 6-point v0 envelope (0, 1e-... | — |

### Candidate-Layered

| Feature | Evidence | Blockers |
|---|---|---|
| Class A curved transport | 6-type geometry (I, II, VI0, VII0, VIII, IX); flat Type I may route to... | — |
| Class B BBN | Reduced-mask Class B geometry registered for 6 labels; full-BBN valida... | — |

### Diagnostic

| Feature | Evidence | Blockers |
|---|---|---|
| Teff spectral hardening legacy kernel | Deprecated legacy Channel 2 closure. Low-level NumPy/JAX kernels are k... | public runtime removed; use characteristic transport / full transport paths |
| AD diagnostic gradients | Diagnostic: custom_vjp/FD parity 8.2e-8 on Type I canonical; parameter... | — |
| Tier-3 full-collision preflight surface (classical Boltzmann; no QKE) | Cumulative PR-T3A/B/C/D preflight surface, scoped to classical Boltzma... | AP-form unification (combine spectral_relaxation anisotropy stability with projected_physical grid scaling): RESOLVED in PR-T3B canonical milestone via collision_mode='ap_unified_preflight' (anisotropy spread ~7e-5, grid spread <1e-4); Dolgov-Hansen-Semikoz nu-nu coefficient calibration: PARTIAL (total_rate_nu_nu_diagonal_jax helper landed; energy-conserving 3T nu-nu source wired in ap_unified_nu_nu_preflight; moment-projected spectral bank source plus number/energy-neutral shape damping wired in ap_unified_nu_nu_spectral_preflight; calibrated no-QKE AP energy-transfer accuracy candidate wired in ap_unified_nu_nu_spectral_accuracy_preflight; full DH-S coefficient-table runtime kernel remains unpromoted); Tier-3 AP energy-moment contract: LOCKED for the no-QKE surface (single Mangano C-rate source, sign-safe positive scaling, bidirectional heating/cooling AP moment parity with 3T rates, bidirectional nu-nu spectral conservation tests, and number/energy-neutral spectral self-thermalization tests); Public diagonal nu-nu dispatch: LOCKED as explicit opt-in for jax_ap_unified_tier3 (off default; 3t/spectral/accuracy modes routed to full forward collision modes; spectral public solve reduces flavour-temperature split; accuracy candidate closes the bounded smoke-grid FLRW N_eff gap to <5e-3 without claiming QKE; public 3T-vs-spectral same-limit comparison locked); FLRW N_eff gap to Mangano 2005 (~0.0095) remains the default AP-form baseline limitation; jax_tier3_nu_nu='accuracy' is a no-QKE calibrated candidate path, not a full DH-S/QKE promotion |
| Type I augmented-PSTF no-QKE staging | HISTORICAL AP0-AP81 stage-record (backing code largely deleted in PR-D... | default/promoted full-span angular kinetic+thermo collision feedback solve using the AP35/AP36 source path; promotion-grade coupled-solve weak-rate convergence beyond the AP80/FB-07 diagnostic gates; promotion-grade PSTF collision-kernel solver/runtime coupling beyond the LRS frozen-smoke pstf_radial artifact/candidate/full-span route, the non-LRS AP42/AP44/AP45/AP46/AP53 diagnostic pstf_radial route, the direct and nonlinear non-LRS pstf_radial 3T wrappers, the AP65 combined angular+pstf_radial nonlinear wrapper/artifact, the AP4/AP65 combined full-span candidate gate/source-policy profile, the direct-wrapper artifact, plus tiny-span budgeted live-RHS radial artifact and live-vs-frozen radial source-policy artifact; anisotropic/tensor QED response and promotion-grade exact-scalar-QED full-span coupled-solver validation beyond the diagnostic exact_finite_mu_scalar scalar 3T routing/gate smoke and FB-08 chained scalar-QED control cross-product rows; process families outside the supported no-QKE HM finite-mass electromagnetic plus all-nine diagonal-nu-nu catalog and promotion-grade coupled-solver use of that catalog; promotion-grade physical full-BBN span with collision-coupled thermodynamics, network, and background variables; real-data/production SMC evidence; AP76/AP79 readiness audit retained diagnostic staging with no promotion approval and AP77 electron-bath provenance; ell_max/angular/q convergence ladder at promotion tolerances beyond FB-09 smoke chained ladders; public forward-solver dispatch and GPU/XLA promotion gates |

### Exploratory

| Feature | Evidence | Blockers |
|---|---|---|
| Inference / PE framework / model comparison | ForwardModel.predict and scalar log-likelihood wrappers exercised on T... | — |

<!-- END:FEATURE_TABLE -->

<!-- BEGIN:CLASSB_LAYERS -->
### Class B validation layers (registry-generated)

| Layer | Count | Types | Evidence |
|---|---|---|---|
| Geometry substrate | 6 types | V, IV, III, VI_h, VII_h, VI_1/9 | Unit tests |
| Family envelope | 6 types | V, IV, III, VI_h, VII_h, VI_1/9 | Representative candidate envelope |
| Full-BBN smoke | 6 types | V, IV, III, VI_h, VII_h, VI_1/9 | Physical range + success |
| BBN-verified gold | 6 types | V/IV/III/VIH/VIIH/VI_M19 only | V A-frame, IV A+N1, canonical III, h-locked VI_h/VII_h representatives, and canonical VI_{-1/9} c=15/4 yield-shift gates |
<!-- END:CLASSB_LAYERS -->

<!-- BEGIN:CLAIMS -->
## Forbidden Claims

- "full Bianchi-BBN package" (only Type I canonical)
- "full differentiable BBN solver" (AD is diagnostic custom_vjp/FD bridge only, not native reverse-mode)
- "Bayes factor headline" (no sampler ran on real BBN likelihood; grid-scan only)
- "gradient-based inference ready" (AD is diagnostic-tier, not production HMC/NUTS)
- "Teff is active on public forward-solver paths" (deprecated legacy; characteristic/full transport supersedes it)
- "Type I validated to Sigma_H = 0.95" (the 0.75 cutoff is only a legacy guarded runtime/implementation range; the publication/science shear domain is NOT VALIDATED pending B-03/B-05)
- "CL3 is fully canonical or publication validated" (historical JAX tiers are frozen)
- "publication-grade model comparison / evidence" (inference is exploratory PE framework only)
- "full neutrino-decoupling self-consistent solver" (canonical default is tier-1 thermo)
- "augmented Type I PSTF no-QKE publication-ready / public production" (AP51-AP81 are only direct-wrapper outcome, solver-matrix, source-policy, convergence, source-budget, evidence-bundle, sanity-matrix, angular input-model, and LRS/non-LRS angular-rate diagnostics plus rate-only and AP65-coupled weak-rate candidate gates, nonlinear transport RHS/solve scaffolds, nonlinear 3T weak/network candidate coupling, opt-in nonlinear collision-feedback wiring, a publication-candidate convergence matrix, a known-limit validation atlas, a guarded inference adapter, an augmented SMC likelihood schema, a smoke tempered-SMC runner, AP71 runtime/cache controls, AP72 synthetic SMC validation, AP73 figure-ready artifact tables, AP74 diagnostic plots, AP75 reproducibility-bundle packaging, AP76/AP79 readiness audit, AP77 smoke coupled weak-rate gate, AP78 same-CL3 weak-rate controls, AP80 profile-level weak-rate convergence diagnostics, FB-20 optional production-candidate evidence gate, and AP81 six-monomial collision statistical-factor plus staged pairwise diagonal-nu-nu source wiring; no public dispatch or production SMC validation yet)
- "Teff is a promotion target" (deprecated legacy; retained only as low-level kernel diagnostics)
- "Tilted production-ready outside Type I scalar slice" (documented candidate scope only: Type I, scalar, v0 in [0, 1e-3])
- "Class A exact curved PSTF production-ready" (exact hierarchy remains substrate; only reduced kappa-cascade is a documented candidate path)
- "Class B production-ready beyond documented candidate slices" (all 6 reduced-mask labels have representative gold cells; continuous h-family, anisotropic K_+/K_-, frame-vector momentum sources, and full Class B remain unpromoted)

## Permitted Claims

Only the following runtime-scoped statements are permitted here. None is publication validation or public-production authority.

- "Type I runtime regression surface (finite-shear publication domain NOT VALIDATED; B-01--B-06 open)"
- "PRIMAT-backed network and FLRW gold are runtime regressions, not matched publication anchors"
- "JAX Type I parity is frozen historical regression evidence; PUB-02C grants no repeated-run authority; G-01R remains open"
- "Tilted scalar BBN documented runtime/diagnostic scope (principal-axis tilted candidate slice, v0 in [0, 1e-3], CL0-CL3; runtime-guarded); not publication evidence"
- "Class A curved transport documented runtime/diagnostic scope (6-type; Type I exact characteristic option, curved cells reduced transport κ-cascade; CL0-CL3); not publication evidence"
- "Class B BBN documented runtime/diagnostic scope (6-label reduced-mask h-locked representative gold candidate); not publication evidence"
- "AD diagnostic gradients documented runtime/diagnostic scope (Diagnostic gradients (custom_vjp/FD parity 8.2e-8); Type I, {eta, tau_n}; not full differentiable solver); not publication evidence"
- "Inference / PE framework / model comparison documented runtime/diagnostic scope (PE framework on Type I {Yp, D/H}; full-forward null scan locked; evidence exploratory); not publication evidence"
- "Bayes factor / evidence computation remains exploratory"
<!-- END:CLAIMS -->
