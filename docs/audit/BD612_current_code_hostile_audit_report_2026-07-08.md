# BD612 Current-Code Hostile Audit — Report

Date: 2026-07-08
Repository: `/home/cosmosapjw/Dropbox/rabbit/RABBIT_v1.0.5_PRODUCTION`
Branch: `feature/bianchi-i-full-nonperturbative`, HEAD `2ed77742c6342d3261dc1fd2168b0044443adb63`
Environment: `venv/bin/python` = Python 3.12.3 (symlink to system python), jax 0.10.0, pytest 9.0.3, CPU JAX (`JAX_PLATFORMS=cpu`, `JAX_ENABLE_X64=1` forced by `tests/conftest.py`).
Prompt: `docs/audit/BD612_current_code_hostile_audit_prompt_2026-07-08.md`.

Method note (disclosed): evidence gathered by direct file inspection at HEAD, executed
test batches (all commands listed in §10), executable probes, and four parallel
read-only section auditors of which one (physics/numerics) was lost to an external
resource limit and was re-performed inline by the lead auditor. Adversarial
verification (§7) was performed against primary sources; two independent auditors
(contract/mapping and lead-probe) reached the top finding separately before
cross-comparison. No repository file other than this report was created or modified.

Claim-status vocabulary used exactly as specified. Calibration law applied
throughout: parity ≠ physical validation; conserved-by-construction ≠ correct
transfer rate; N_eff ≈ 3.044 ≠ validation; endpoint completion ≠ Bianchi-I
full-BBN validation.

---

## 1. Current target reconstruction

"Reaches production path" = reachable at runtime from public inference dispatch
(`forward_likelihood.canonical_forward_solver` / `canonical_batch_forward_solver` /
`BBNLikelihood`), not merely import-loadable. All cells verified by fresh `rg`
consumer traces at HEAD this session.

| # | Track | Claimed purpose | Live files | Tests/gates | Reaches production path? | Claim status | Source of truth |
|---|---|---|---|---|---|---|---|
| 1 | Clean FLRW decoupling core | Standalone FLRW ν-decoupling integrator (clean-core seed) | `src/rabbit/collisions/dynamic_collision_core.py` (285 L), `dynamic_collision_driver.py` (`integrate_flrw_decoupling` :236) | `tests/test_dynamic_collision_core.py`, `tests/test_dynamic_collision_driver.py` — executed, all green incl. slow Gate B | **NO — deliberate island.** `rg integrate_flrw_decoupling` → definition + its test only; zero consumers in `src/rabbit/{inference,jax,drivers}` | IMPLEMENTED (FLRW-only). Endpoint ≠ full-BBN validation; module says so itself (:47-53) | executed batch1 + slow run; fresh rg trace |
| 2 | Deterministic collision reference | First-principles weak matrix-element gain–loss oracle | `deterministic_reference.py` (811 L) + `kernels.py` | `tests/test_deterministic_collision_reference.py` (33 tests, executed green) | **NO runtime reach.** Imported by clean core, `nu_e_scattering.py`, `pair_processes.py`, `pstf_*`; those feed tier-3 full-Boltzmann preflight which has no `CAPABILITY_BY_BACKEND` key (`backend_capabilities.py:1001-1013`) | IMPLEMENTED as oracle; rate normalization **NOT VALIDATED** (Finding F-1) | executed batch1; rg trace; probe §7-Q10 |
| 3 | B4 calibrated-RTA collision twins | Parity-locked SciPy↔JAX calibrated-RTA gather-scatter bridge | `projected_operator.py`, `transport/teff_collision_bridge.py`, `jax/collision_operator_jax.py`, `jax/teff_collision_bridge_jax.py` | batch2 executed, 58 green | **SciPy twin: YES, opt-in only** (`drivers/full_coupled_typeI.py:772-773` under `enable_collisions and tier>=2`, default `False` :1039). **JAX twin: NO** (blocked in dispatch) | IMPLEMENTED (calibrated RTA, `_C_RATE=210` tuned to N_eff≈3.044). Parity is engineering fidelity, NOT physics validation — files say so verbatim | `full_coupled_typeI.py:772-773,941,1039`; executed probe §7-Q7 |
| 4 | JAX char LRS tier-2 collision candidate wiring | Candidate collisional I_coll DOF on JAX characteristic driver | `jax/driver_typeI_char.py` (default OFF :1615; layout :367-371; RHS gate :751) | `tests/test_b4_jax_char_collision_wiring.py` (executed green), unsound firewall test | **NO.** Triple fence verified + probed live: config default OFF; `__post_init__` :1680-1696 rejects non-LRS/tier<2 (probe: raised); `forward_likelihood.py` :1087-1092, :1721-1726, :2243-2247 raise (probe: raised) | IMPLEMENTED candidate; public dispatch FORBIDDEN and fail-closed | executed probes; batch2 |
| 5 | B5 non-LRS operator substrate | Augmented-PSTF distribution + non-LRS transport substrate | `transport/augmented_pstf_distribution.py`, `augmented_nonlrs_transport.py`, `augmented_typeI_nonlrs_collisionless.py`, `augmented_typeI_observables.py`; `jax/characteristic_rays_nonlrs_jax.py` | `test_augmented_*`, `test_b5_*`, `test_pr_n1/n2_*`, `test_nonlrs_char_observable_oracle.py` (not in required batches) | **4 of 5 dead islands** (consumers: `transport/__init__.py` re-export + each other + tests only). `characteristic_rays_nonlrs_jax` IS reachable via candidate backend `jax_characteristic_nonlrs` (tier="candidate", `backend_capabilities.py:273`, guard `forward_likelihood.py:2231-2236`) | quartet: IMPLEMENTED island; rays: IMPLEMENTED candidate-reachable | fresh per-module rg traces |
| 6 | Canonical/public forward solver + inference dispatch | Single public forward-BBN + likelihood entry | `inference/forward_likelihood.py` (:1049, :1294, :1499) → `jax/driver_typeI.py`, `jax/driver_typeI_char.py` batch tiers (:4202/:4247), `drivers/full_coupled_typeI.py` | `test_likelihood_rejects_surrogate.py`, `test_production_gates.py`, gold suite | **YES — this is the production path.** Guards verified verbatim: surrogate rejection :401-408; batch guards :1082-1097; JAX collision guard :1721-1726; characteristic guards :2237-2252 | IMPLEMENTED with fail-closed guards (guards = internal consistency, not physics validation) | file reads + live probe |
| 7 | Claim gates + capability registries | Fail-closed Forbidden→Permitted machinery + capability truth tables | `config/claim_gates.py` (14 gates), `backend_capabilities.py` (25 backends, 12 public dispatch keys), `feature_capabilities.py` (**12** features, not 11), `scripts/promotion_check.py` | batch3 executed: 32 green + finite-shear anchor FAILED closed by design; `promotion_check.py --status` executed (ledger §5/§7) | YES (dispatch reads `CAPABILITY_BY_BACKEND` :1718) | IMPLEMENTED. Gates gate claims; they do not validate physics | executed batch3 + promotion run |
| 8 | Post-deflation script/test hygiene | Script/test population after PR-D1..D3 | `scripts/`: 168 top-level .py (169 with `scripts/cas/`); `tests/`: 308 top-level `test_*.py` + conftest (332 repo-wide) | n/a | **6 scripts broken** (ModuleNotFoundError demonstrated live): imports of deleted `rabbit.debug.{grid_resolver,modal_cluster_policy,qprofile_canonical,replay_dump}`; `src/rabbit/debug/` retains only 5 modules + `__init__` | 6 scripts DEPRECATED (un-runnable); no `src/`/`tests/` breakage | executed import demo; ls + rg |

---

## 2. Contract/interface audit

### Contract 1 — Clean driver frame (`dynamic_collision_driver.py`)

| Clause | Evidence | Status |
|---|---|---|
| State = fixed comoving f(Y) + T_γ; Y grid = reinterpreted Laguerre nodes | :11-14, `Y = q_nodes.copy()` :163, state pack :227-231 | IMPLEMENTED |
| z = a·T_γ, z_init = 1 | :21-22, :176-177, `a_init = 1/T_gamma_init` :264-266 | IMPLEMENTED |
| df/dN\|_Y = C/H, no redshift advection | :17; collision branch :198-202; collisionless zeros :213-217 | IMPLEMENTED |
| Collisionless endpoint recovers analytic entropy limit "within stated EOS convention" | readout :339-344, `_NEFF_PREFAC=(11/4)^{4/3}` :80; Gate A executed green; n_q-convergence pin 2.9934±1e-3 | IMPLEMENTED, internal-consistency only. **Ambiguity**: the EOS convention is never stated — the 0.0066 offset from "analytic 3" is silently inherited from `nudec_coupled` defaults (`qed_correction_model="finite_mu_scaled"`, `nudec_coupled.py:238`); driver docstring still says "N_eff = 3.00 exactly" (Finding F-8) |
| Fail-closed endpoint | RuntimeError :311-321 (no event / solver fail), :326-334 (negative endpoint moment, explicitly refusing to floor) | IMPLEMENTED |

### Contract 2 — Energy transfer (driver + core)

| Clause | Evidence | Status |
|---|---|---|
| G = ∫Y³ (df/dN) dY from SAME evolved df | :206-208; pinned through real rhs by `test_energy_conserving_plasma_coupling` (executed green, rel 1e-10) | IMPLEMENTED |
| Frame factor (T_γ⁴/z⁴)/(2π²) | inline :209 (NOT `plasma_prefactor` :83 — that function has zero src callers; disconnected-but-pinned) | IMPLEMENTED |
| Degeneracy 2 (ν_e+ν̄_e) / 4 (ν_x bank) | hard-coded literals :210-211, consistent with `species.py:12-16` and N_eff readout weights (1 + 2·ν_x, :342). **Counted once and only once — no double-count found** (independent trace, §3 item 5) | IMPLEMENTED |
| Plasma loss ≡ −(ν gain) | :206-225 → `nudec_coupled.py:297-299`. Conservation is BY CONSTRUCTION — per calibration law this does not validate the rate (Finding F-1) | IMPLEMENTED |
| Core twin `collision_bank_energy_sources_per_efold` (core :217-253) | **Normalization inconsistent with driver**: omits the T_γ⁴/(2π²) frame factor the driver's coupling includes (driver docstring :39-40 equates its coupling to "PHYS·(module dQ)"); fed bare into `coupled_3T_rhs_from_collision_moments` (÷ MeV³ heat capacities, `nudec_coupled.py:301-305`) units do not close; docstring "[MeV⁵]" (core :117) overstates. Test-only path, not marked DEPRECATED (Finding F-3) | IMPLEMENTED but inconsistent |

### Contract 3 — Deterministic collision reference (`deterministic_reference.py` + `kernels.py`)

| Clause | Evidence | Status |
|---|---|---|
| Pauli six-monomial factor | :18-26, :70-78; hand-expansion of f₃f₄(1−f₁)(1−f₂)−f₁f₂(1−f₃)(1−f₄) reproduces the 6 monomials exactly | IMPLEMENTED, algebraically exact |
| Detailed-balance null | energy-conserving y₄=y₁+y₂−y₃ :399, :519-522 ⟹ S[FD]=0; executed tests on the FULL callable green (`max|C|<1e-28`) | IMPLEMENTED |
| Heating sign for colder ν | executed green (dQ>0, monotone; ν_x < ν_e) | IMPLEMENTED |
| Rate dimensions & T-scaling | prefactor `G_F_MEV**2 * T**4 / (4π³)` :394, :451, :515, :578, :711, :766; `G_F_MEV` = MeV⁻² (`kernels.py:31`); internal y-integrals dimensionless; **no electron mass anywhere in the module** (rg: zero hits for `M_E|0.511`) | **NOT VALIDATED — Finding F-1** (dimensionally inconsistent as a df/dt in MeV; one power of T below the repo's own canonical G_F²T⁵ weak-rate scaling; probe §7-Q10 confirms dQ ∝ T⁴ exactly and inverted Γ/H(T)) |
| Couplings | `kernels.py:28-56` (sin²θ_W=0.23122, V−A G_L/G_R, Notzold-Raffelt A_TOTAL) | IMPLEMENTED |

### Contract 4 — Calibrated RTA / gather-scatter bridge

| Clause | Evidence | Status |
|---|---|---|
| numpy reference = calibrated damping, NOT physical | `projected_operator.py:1-38` (`_C_RATE=210` calibrated to N_eff≈3.044 :33); `PhysicalCollisionOperator` limitations :253-259 (linearized ~7% at ΔT/T=1%, q-independent Γ, no ν-ν) | IMPLEMENTED, honestly labelled |
| JAX parity = engineering fidelity only | both twins carry verbatim "PARITY IS NOT PHYSICAL VALIDATION" (:22-27); constants imported not forked (:37-49) | IMPLEMENTED |
| δI / I_coll semantics | δI_j = −δρ/(8·ρ_ref) uniform (`teff_collision_bridge.py:233-245`); I_eff = I_recon + I_coll (`driver_typeI_char.py:741-755`); dI_coll/dN = δI[0] :795-801. **Ambiguities**: (a) docstring says ÷ρ_ν but code divides by fixed equilibrium `rho_ref` (:207 vs :226,:239); (b) numpy env knobs `RABBIT_COLLISION_QMAX/BRIDGE_RELAX` (:43-59) vs JAX static defaults — parity silently diverges under non-default env; (c) `delta_rho_nu` never consumed; plasma never debited; tier-2 3T runs an independent averaged source (:774) — undocumented one-sided energy bookkeeping on the candidate surface (Finding F-6) | IMPLEMENTED with 3 under-documented approximations |
| Tangency diagnostic | declared DEGENERATE, report-only (:12-17, :142-185); JAX mirror :68-78 | IMPLEMENTED (honest) |
| LRS tier-2 candidate scope; non-LRS closure unpromoted | layout gate :367-371; `__post_init__` :1685-1696; `_char_layout_nonlrs` takes no `enable_collisions` (:420-424) | IMPLEMENTED / FORBIDDEN surface enforced |

### Contract 5 — Claim gates

| Clause | Evidence | Status |
|---|---|---|
| Missing tests are red | `_check_gate` puts non-collectible nodes in `missing` (`promotion_check.py:130-143`); `is_green` requires none missing (:85-91); collect timeout ⟹ all missing (:111-113) | IMPLEMENTED |
| Skipped ≠ pass | JUnit parse rejects skipped/failure/error/empty (:56-70); `PYTEST_ADDOPTS` cleared (:49-53, :103-104); per-subprocess timeouts fail closed (:45-46, :163-164); locked end-to-end by `test_promotion_check_skip_is_not_pass.py` (executed green incl. real pytest-subprocess skip) | IMPLEMENTED |
| External finite-shear anchor fail-closed | `pytest.fail` not skip (:46-60) — executed: FAILED closed by design; wired into `GATE_SIGMA_H_TO_0P95` (`claim_gates.py:119`); loader rejects FLRW-only / sourceless / RABBIT-self-run benchmarks (:146-201) | IMPLEMENTED. **Stale docstring**: test module :9-13 still describes pre-BD601 returncode semantics (F-8) |
| No silent green on stale/deleted nodes | deleted node ⟹ missing ⟹ red; verified against ledger (§5). **Gap**: no test detects gate-node staleness itself; `test_test_node_ids_are_well_formed` checks syntax only, and 17 required nodes across 9 deleted test files exist in `claim_gates.py` today (Finding F-4) | IMPLEMENTED, fail-safe; staleness undetected |
| Public-dispatch nuance | "candidate collisions out of public inference" is exact for JAX/batch surfaces only; SciPy calibrated-RTA collisions are publicly reachable via explicit opt-in (`forward_likelihood.py:2588-2620`, default off, tier forced ≥2 :2592). Guard messages direct users there by design (:1090-91, :2246) (Finding F-7) | IMPLEMENTED, documented boundary |

---

## 3. Physics/math audit ledger

| # | Item | Status | Evidence | Residual risk |
|---|---|---|---|---|
| 1 | Does collisionless N_eff test the intended frame, or can it pass while a collisional frame/interpolation error remains? | IMPLEMENTED (frame bookkeeping); the risk is real and acknowledged in-repo | Gate A never exercises the comoving↔thermal resample — the test file's own FINDING-4 comment concedes it (`test_dynamic_collision_driver.py:106-107`); resample only runs with collisions on, where slow `test_collision_on_frame_scale_is_tight` (executed green, T_ν/T_γ abs 0.01) constrains it coarsely | A collisional-only interpolation bias below the 0.01/band tolerance passes; interp interior error is tens of % (item 3) |
| 2 | Does energy conservation by construction hide an inaccurate continuum transfer? | **YES — by design it cannot see rate errors** | Driver :204-211 defines plasma loss as minus the moment of the SAME df; conservation test asserts rel 1e-10 "regardless of interpolation quality" (its own docstring) | The transfer MAGNITUDE has no independent check anywhere (§6-Q1); combined with F-1 this is the load-bearing blind spot |
| 3 | Is the deterministic reference rate dimensionally and physically plausible? | **NOT VALIDATED — dimensionally inconsistent (F-1)** | Prefactor `G_F²T⁴/(4π³)` is dimensionless (G_F² = MeV⁻⁴); consumed as df/dt in MeV by driver :194,:201. Repo's own canonical scalings disagree: `rate_prefactors.py:9,16` (Γ=(7π/12)G_F²T⁵), `projected_operator.py:70` (G_F²T_γ⁴T_ν, MeV), `nudec_tables.py:95` (dQ ∝ T⁸ΔT, MeV⁵). Executed probe: dQ ∝ T⁴ exactly (ratio 16.0000 for T 10→5); Γ_relax = dQ/Δρ = 3.04e-21 MeV CONSTANT in T; Γ/H crosses 1 at ≈2.5 MeV **from below** (0.07 at 10 MeV → 29 at 0.5 MeV) — inverted vs the standard Γ/H ∝ T³ falling profile. Also: electrons are massless in the module (no m_e anywhere) — undocumented ~10-30% pair-phase-space effect at T ~ m_e | Clean-core collisional results (Gate B N_eff ∈ [3.00,3.15]) cannot be interpreted as physical decoupling; magnitude error ≈ (T/MeV)⁻¹ across the window; BD611 already lists this exact check as open (":262: If no, freeze N_eff claims and fix prefactor/dimension") |
| 4 | Does detailed balance hold for the actual callable field? | IMPLEMENTED, executed | Three layers, all executed green: algebraic factor (`max|S|<1e-15`), full `evaluate_*_reference` at FD (`max|C|<1e-28`), full `neutrino_collision_energy_transfer` at equilibrium (db_res<1e-12) | DB nulls are insensitive to overall normalization — they cannot catch F-1 |
| 5 | Degeneracy factors counted once and only once? | IMPLEMENTED — **no double-count found** | Driver: literals 2.0/4.0 at :210-211 matching `BANK_DEGENERACY` (`species.py:12-16`); N_eff readout weights (1 + 2·ν_x) :342 consistent (4 Weyl = 2 N_eff units); frame 1/(2π²) is per-species with degeneracy applied exactly once; core path applies 4 once (:252). Executed `test_nux_bank_applies_degeneracy_factor` green | Literals linked to registry by comment only (F-9); a registry change would not propagate |
| 6 | Are clip/H_safe/max_clip_excursion diagnostics strong enough to prevent silent repair? | IMPLEMENTED on the clean core; one candidate-path exception | Raw min-Hubble recorded before the 1e-100 floor (:187-191); clip excursion on accepted samples (:299-306) asserted <1e-6 (executed); negative endpoint moment refuses to floor, RuntimeError :326-334. Exception: `teff_collision_bridge.py:229-231` `nan_to_num`s non-finite C and returns zeros on degenerate `rho_ref` — silent sanitize on the candidate path (F-6) | Clip tracking excludes Radau probe states by documented rationale (:156-160) — acceptable; the bridge exception is the one true silent-repair site |
| 7 | Does B4 I_coll represent only isotropic energy shift; what does it miss? | IMPLEMENTED as documented candidate | δI uniform across rays (:233-245), q-independent Γ, linearized source (~7% at ΔT/T=1%), no ν-ν; anisotropic collision damping of shear-induced multipoles (ℓ≥2 ray structure) is structurally absent; `delta_rho_nu` unconsumed / plasma undebited (F-6) | Finite-shear collisional physics unrepresented — consistent with candidate tier; would poison any Σ_H≠0 collisional claim if promoted (it is not) |
| 8 | Is the B5 non-LRS substrate an active driver capability or an operator island? | IMPLEMENTED: 4 modules island, 1 candidate-reachable | Per-module rg traces (§1 track 5): augmented quartet has zero production call sites (import-time re-export only); `characteristic_rays_nonlrs_jax` serves candidate backend `jax_characteristic_nonlrs` (tiers 1-2 only, :2231-2236) | Island modules imply maintenance surface without capability; registry prose still narrates them as part of a "landed AP0-AP81 surface" (F-2) |
| 9 | Do any docs/registries upgrade internal consistency to validation? | Mostly NO; two degraded spots | Live claim surfaces hedge correctly (README clean; forbidden blocks negated; drivers carry "NOT a physics-validated collision path" `driver_typeI_char.py:1614`). Degraded: `src/rabbit/weak/corrections.py:86` "validated against PRIMAT" — affirmative external claim whose named gate file (`tests/test_cl3_canonical_promotion.py`) was deleted (live `tests/test_primat_parity.py` exists, so PLAUSIBLE but evidence chain degraded); WBS/registry "landed" prose (F-2) narrates deleted capability as present | Wording drift, not active overclaim; fix via F-2 deflation |

---

## 4. Equation-to-code mapping

| # | Equation / object | Location | Verdict |
|---|---|---|---|
| 1a | Comoving→thermal resample f_th(q) = f(q·z), tail→0, left-hold, clip [0,1] | `dynamic_collision_driver.py : _resample_comoving_to_thermal : 132-137` | approximate-with-documented-regime (linear interp; tens-of-% interior error pinned by `test_interp_roundtrip_bounds_error`, executed) |
| 1b | Thermal→comoving field map C(Y) = C_th(Y/z) | `: _map_thermal_field_to_comoving : 140-142` | approximate-with-documented-regime (low-Y left-hold boundary stated in code only — minor) |
| 2 | ∫q³ f dq Gauss-Laguerre (FD ⟹ 7π⁴/120) | `dynamic_collision_core.py : spectral_energy_moment : 84-88` (+`_laguerre_dq_weights` :76-81) | exact (executed against closed form) |
| 3 | PHYS = T_γ⁴/(2π²) | `dynamic_collision_driver.py : plasma_prefactor : 83-94` | **disconnected** (zero src callers; live rhs uses inline frame :209; pinned by one test as documentation) |
| 4 | C[f] = C_scatt + C_pair | `dynamic_collision_core.py : _collision_field : 153-161` (kernels `deterministic_reference.py:366-425, 485-548`) | **approximate-under-documented** (composition exact; underlying G_F²T⁴/(4π³) prefactor dimensionally inconsistent as a MeV rate — F-1; deviation from canonical G_F²T⁵ documented nowhere in module or tests) |
| 5 | (dQ_νe-pair, dQ_νx-bank)/dN with degeneracy 4 | `dynamic_collision_core.py : collision_bank_energy_sources_per_efold : 217-253` | **approximate-under-documented** (frame factor absent vs driver's conserving coupling; units do not close into `nudec_coupled` MeV³ heat capacities — F-3; test-only path) |
| 6 | FLRW comoving decoupling integration; N_eff = (11/4)^{4/3} z⁻⁴(I_νe + 2I_νx)/I_FD | `dynamic_collision_driver.py : integrate_flrw_decoupling : 236-360` (readout :339-344; fail-closed :311-334) | approximate-with-documented-regime (frame/entropy algebra exact; EOS offset and interp error test-documented; collision magnitude inherits row 4) |
| 7 | JAX gather-scatter (Θ_j=e^{−2I_j}; RTA collide; δI=−δρ/(8ρ_ref)) | `jax/teff_collision_bridge_jax.py : apply_gather_scatter_collision_jax : 81-137`; twin `jax/collision_operator_jax.py : physical_collision_rhs_jax : 92-140` | approximate-with-documented-regime (calibrated RTA loudly labelled; ρ_ref-vs-ρ_ν + env-knob caveats under-documented — F-6) |
| 8 | I_coll construction + handoff | `jax/driver_typeI_char.py : _rhs_core : 741-755, 788-803, 840-841; _char_layout : 332-396; _carry_collision_accumulator_handoff : 399-417` | approximate-with-documented-regime (superposition argument structurally sound; handoff fails loud on one-sided slot; `delta_rho_nu` unconsumed — F-6) |
| 9 | Canonical dispatch guards | `inference/forward_likelihood.py` :401-408 (surrogate), :1082-1097 (batch), :1324-1328 (batch-3d), :1721-1726 (any-JAX), :2237-2252 (characteristic), :2084-2093 (ap_unified) | exact (every guard a hard ValueError, verified in situ and probed live; scipy opt-in nuance F-7) |
| 10 | Claim-gate definitions + promotion behavior | `config/claim_gates.py:38-46, 245-260` (anchor gate :108-130); `scripts/promotion_check.py:56-70, 98-121, 124-174, 85-91` | exact (skip≠pass, missing⇒red, deleted-node⇒red, timeout⇒red — executed) |
| 11 | BD205 root-cause references | `docs/audit/BD205_qlaguerre_collision_root_cause_2026-07-01.md:38` cites `transport/augmented_collision_bridge.py:4388-4400` | **stale-deleted** (cited file deleted in PR-D; forensic history, needs a deleted-file annotation) |

---

## 5. Numerical/pipeline audit

| # | Item | Assessment |
|---|---|---|
| 1 | Stiff solver suitability + endpoint fail-closed | Radau (implicit, stiff-appropriate) with terminal event; non-endpoint and solver-fail ⟹ RuntimeError :311-321; negative endpoint moment ⟹ RuntimeError refusing to floor :326-334. Executed: slow Gate B trio green in 45 s. IMPLEMENTED |
| 2 | Tolerance / n_q convergence meaning | Collisionless N_eff = 2.9934 stable ±1e-3 across n_q ∈ {16,24,32,48} (executed). Establishes quadrature/frame self-convergence to the code's own EOS convention — NOT convergence to continuum truth, and the "analytic 3.00 exactly" docstring overstates it (F-8) |
| 3 | Interpolation/quadrature error | Linear `np.interp` between comoving/thermal Laguerre grids; interior relative error band [0.05, 2.0] documented and pinned by `test_interp_roundtrip_bounds_error` (executed). Conservation is immune by construction — which is exactly why transfer accuracy is unchecked (§3 item 2) |
| 4 | Rate normalization & decoupling-window plausibility | Executed probe (§7-Q10): Γ_relax/H crosses 1 at ≈2.5 MeV (ν_e) / ≈0.9 MeV (ν_x) — right order of magnitude at the window, **wrong sign of T-dependence** (rises as T falls). The plausible-endpoint N_eff is therefore not evidence of a correct rate (F-1) |
| 5 | Tail underflow/overflow/cancellation | `exp(min(q,500))` clip in Laguerre weights (:79-81); `fnn_floor=1e-12` where-guard (:205-206); `q²` floor 1e-30 (`deterministic_reference.py:414`); logit form makes f∈(0,1) structural. No unguarded tail found in the audited files |
| 6 | Raw state preservation vs clipping | Clean core: preserved and instrumented (§3 item 6). Candidate bridge: `nan_to_num` + zeros-on-degenerate-ρ_ref silent sanitize (F-6) |
| 7 | JAX x64 assumptions | `tests/conftest.py` forces `JAX_ENABLE_X64=1` pre-import; the JAX twins contain no x64 guard of their own (rg: zero hits) — a library consumer outside pytest could silently run float32. Parity tolerances (1e-12) would fail loudly under tests; unguarded outside them. P3-level |
| 8 | Cache/state leaks | `teff_collision_bridge.py:35-36` module-level `_COLLISION_OPERATOR_CACHE` (keyed by N_q) and `_BRIDGE_QUAD_CACHE`; env knobs `RABBIT_COLLISION_QMAX/BRIDGE_RELAX` read per call (:43-59) are NOT part of cache keys' construction path for the operator — combined with the JAX static defaults this is the F-6(b) parity-divergence surface. Clean core: no module-level mutable caches found |
| 9 | Default-off / candidate fences | Verified at all three layers and probed live (§1 track 4). Fences intact |
| 10 | Runtime impact of remaining scripts/tests post-AP65 | Fast batches: batch1 3.95 s, batch2 17.4 s, batch3 0.94 s. Slow: Gate B trio 45.0 s (104 MB RSS), B4 unsound 16.1 s (1.16 GB RSS). `promotion_check.py --status` dominates wall time (pytest collect of 3172 nodes + per-gate runs). 6 broken scripts fail at import (no runtime cost unless invoked); 2 packet scripts emit commands referencing deleted files (fail when an external auditor replays them) |

`promotion_check.py --status` executed ledger (fail-closed states as designed):

```
Collected 3172 node(s).

Gate                                      State           Tests
------------------------------------------------------------------------------------------
full_bianchi_bbn                          RED-fail        0/22
full_differentiable_bbn_solver            GREEN           3/3
bayes_factor_headline                     RED-missing     0/3
gradient_based_inference_ready            RED-missing     2/3
teff_public_runtime_deprecated            doc-only        0/0
sigma_h_validated_to_0p95                 RED-fail        0/3
cl3_fully_canonical                       RED-missing     0/3
publication_grade_model_comparison        RED-missing     0/3
full_neutrino_decoupling_selfconsistent   RED-missing     0/3
augmented_typei_public_production         doc-only        0/0
teff_promotion_target_deprecated          doc-only        0/0
tilted_production_outside_type_i_scalar   RED-fail        0/11
class_a_exact_curved_pstf_production      RED-missing     3/5
class_b_beyond_documented_slices          RED-missing     6/8
```

Reading: `sigma_h_validated_to_0p95` is RED-fail because the external anchor was
actually executed and failed closed (correct); the six RED-missing gates are red
because required nodes are non-collectible — including the deleted-file references
of Finding F-4 (note `gradient_based_inference_ready` 2/3 and
`class_b_beyond_documented_slices` 6/8: passing majorities cannot green a gate with
a missing node — fail-closed working as designed). The single GREEN gate
(`full_differentiable_bbn_solver`) rests on live, executed AD-parity/Fisher nodes;
`--diff` promotion remains human-in-the-loop and nothing auto-writes.

---

## 6. Test/docs/registry honesty audit

### Classification

| File | Dominant classes |
|---|---|
| `test_dynamic_collision_core.py` | unit algebra; internal consistency (cross-model check :242-255 is **sign-only** by its own docstring) |
| `test_dynamic_collision_driver.py` | internal consistency; analytic-limit anchor (Gate A); numerical regression (2.9934 pin); slow-gate; module says "No external anchor exists… INTERNAL consistency numbers only" (:14-16) |
| `test_deterministic_collision_reference.py` | unit algebra; internal consistency; numerical regression (replay pins are self-generated — catch drift, not wrong-from-birth normalization) |
| `test_physical_collision_operator_reference.py` | unit algebra of its OWN model; parity oracle |
| `test_jax_collision_operator_parity.py`, `test_jax_teff_collision_bridge_parity.py` | parity-to-reference; import/smoke (jit/grad) |
| `test_b4_jax_char_collision_wiring.py` | numerical regression (pinned pre-wiring baseline); parity; claim firewall; "Magnitude is NOT validated here" (:158-176) |
| `test_b4_collisional_char_reference_unsound.py` | claim firewall; inverted internal consistency (asserts the SciPy collisional reference IS anomalous and diverges — executed green) |
| `test_claim_gates.py` | claim firewall (schema/text only; executes no physics) |
| `test_promotion_check_skip_is_not_pass.py` | gate-tooling hygiene (unit + real-subprocess end-to-end) |
| `test_finite_shear_external_anchor.py` | **the only genuine external-validation surface for finite shear — fail-closed, currently RED by design** (executed: FAILED as designed) |
| `test_augmented_wbs_status_ledger.py` | **stale-wording enforcement lock** (asserts deleted AP5-AP80 = "landed in current workspace (stage-scoped)"; :22-36) |
| `test_augmented_pstf_capability_registry.py` | claim firewall (honesty tiers) + **stale-prose enforcement lock** (:101, :154) |

### The six questions

1. **Wrong collision-rate normalization** — **no test would catch it.** Sign-only cross-model check (`test_dynamic_collision_core.py:244-245`: "Magnitudes are NOT compared"); replay pins self-generated; Gate B's [3.00, 3.15] band is coarse and N_eff-closeness is not validation; `test_collision_hm_full.py` closed-form checks use the same constants they verify. This is the direct test-side counterpart of F-1.
2. **Wrong comoving/thermal transfer** — slow `test_collision_on_frame_scale_is_tight` (abs 0.01) and Gate A (which "never exercises the frame resample", its own comment :106-107); `test_interp_roundtrip_bounds_error` documents rather than bounds the error; conservation tests cannot catch a wrong rate by construction.
3. **Stale capability registry after deletion** — **none; two tests enforce staleness instead** (`test_augmented_wbs_status_ledger.py:22-37`, `test_augmented_pstf_capability_registry.py:101,154` — correcting the stale docs/registry is a CI failure today). Additionally 17 gate-required nodes across 9 deleted test files sit in `claim_gates.py` undetected (fail closed for promotion, invisible to tests) — F-4.
4. **Public-dispatch leak of candidate collisions** — `test_b4_collisional_char_reference_unsound.py::test_inference_forward_path_still_blocks_jax_char_collisions`, `test_surface_scope_honesty.py::test_jax_backends_reject_enable_collisions`, `test_b4_jax_char_collision_wiring.py::test_collisions_rejected_outside_lrs_tier2`, `test_scipy_characteristic_production_fence.py`. All executed or verified in-file. The SciPy opt-in path is deliberately outside these locks (F-7).
5. **False-green / schema-only / self-consistency locks** — worst: the two stale-wording enforcement tests (green precisely because deleted code is still ledgered "landed"). Schema-only: all of `test_claim_gates.py`, the B4 line-scan firewall. Self-consistency: replay pins, pinned pre-wiring Yp/DH baseline, `jax_bbn_gold.json` family, all JAX↔numpy parity, conservation-by-construction.
6. **Slow tests required before stronger physics claims** — the Gate B trio and the B4 unsound diagnostic (both executed green this audit); `test_finite_shear_matches_external_oracle` (required and currently impossible — no external benchmark exists); `cross_code`-marked live FLRW parity (env-gated; skips never count as passes).

### Stale-wording verification (exact quotes, all confirmed on disk)

- `docs/IMPLEMENTATION_GUIDE_TYPEI_AUGMENTED_PSTF_NOQKE_WBS.md:599`: AP65 row "landed in current workspace (stage-scoped)… `run_augmented_nonlrs_nonlinear_angular_collision_weak_network_3T_solve(...)` composes…" — that function is defined **nowhere** in the repo (rg over `src/` and repo-wide: zero definitions; `out_of_scope/` only imports it). 82 rows carry the "landed" status.
- `tests/test_augmented_wbs_status_ledger.py:22-36` locks those statuses verbatim; the second test's NAME says "planned" while its body asserts "landed" — a name/body honesty defect in the lock itself.
- `src/rabbit/config/backend_capabilities.py:557-574` + :644-647 and `feature_capabilities.py:371-378`: honesty tiers intact (`tier="substrate"`, `validated_default=False`, "Catalog-only… no public dispatch") but prose narrates "The landed AP0-AP81 surface includes … AP65 opt-in nonlinear angular collision-feedback 3T wrapper… AP65 3T solve…" for deleted modules (`src/rabbit/validation/` contains no `augmented_*` today).
- `SUPPORTED_CAPABILITIES.md:86` (registry-generated): "AP0-AP81 substrate pieces are landed for the no-QKE augmented-PSTF pro…" — the stale prose reaches a headline claim surface.
- `scripts/make_external_reaudit_packet.py:420-474, 676` and `scripts/make_external_performance_optimization_audit_packet.py:527, 616, 1121` embed pytest commands/repo-map rows naming deleted `tests/test_augmented_collision_bridge.py` and `src/rabbit/transport/augmented_collision_bridge.py` (both confirmed absent) — generated audit packets are non-replayable.
- `bbn_codex_anti_drift_cost_effective_policy.md:16-17`: "Target repeated-run backend remains CPU-JAX plus the in-tree Rodas5P/AP65 host…" — the AP65 host is no longer in-tree.
- `docs/audit/BD205_…:38` cites deleted bridge line ranges (forensic history; annotate).
- `docs/harness/VALIDATION_LEDGER.md`, `docs/ROADMAP_STATE_OF_RECORD.md:46-52`, `docs/harness/PROJECT_STATE.md:5`: dated-but-presented-as-current descriptions of deleted AP65/FB surfaces; the VALIDATION_LEDGER's recorded pytest commands are no longer replayable.
- `src/rabbit` docstring sweep: no live upgrade of internal consistency to validation; hedges dominate. One degraded external claim: `weak/corrections.py:86` "validated against PRIMAT" (gate file deleted; live `test_primat_parity.py` exists — PLAUSIBLE, evidence chain degraded).

---

## 7. Adversarial verification questions and answers

Method: each question answered from primary sources or executed commands; the top
finding (F-1) was reached independently by two auditors (contract auditor via
dimensional analysis and repo-internal contrast; lead auditor via executable probe)
before cross-comparison.

**Q1. First positive claim that breaks on code-only reading?**
"The clean core couples a *first-principles* collision integral." The composition
(matrix elements, Pauli factor, DB null) is genuinely first-principles, but the rate
the driver consumes is not a dimensionally consistent df/dt: `G_F²T⁴/(4π³)` is
dimensionless, one power of T below the canonical weak rate, with no electron mass
(probe: dQ ∝ T⁴ exact; Γ/H inverted). The words "first-principles" survive; the
implied "physically normalized rate" does not. Everything else claimed of the clean
core (endpoint, fail-closed, conditioning fix) held under execution.

**Q2. Strongest surviving engineering contribution?**
The BD205 conditioning repair (evolve bank energy, never divide by c_v ∝ T³ —
`dynamic_collision_core.py:67-73`) plus the fail-closed driver endpoint machinery
(:311-334), executed to a green endpoint; and the promotion firewall (skip≠pass
JUnit parsing + fail-closed external anchor), executed. These survive a hostile read
intact.

**Q3. Parity described as physics validation anywhere?**
No live surface does. Both JAX twins carry "PARITY IS NOT PHYSICAL VALIDATION"
verbatim (:22-27); `test_claim_gates.py::TestNoValidatedCollisionPhysicsClaim`
(executed green) forbids affirmative validat*+collision+physic* co-location on the
B4 surface.

**Q4. Internal consistency described as external validation?**
No live upgrade found. FLRW external anchors (PRIMAT/Mangano/NUDEC_BSM) are scoped
FLRW-only; the finite-shear anchor is RED by design and `GATE_SIGMA_H_TO_0P95`
structurally depends on it (`claim_gates.py:119`, executed anchor failure).
Degraded spot: `weak/corrections.py:86` "validated against PRIMAT" with its named
gate file deleted (PLAUSIBLE; live `test_primat_parity.py` exists).

**Q5. Deleted capability still advertised?**
Yes — narrowly but on real surfaces: WBS "landed" ×82 rows (naming a function that
exists nowhere), registry AP0-AP81 "landed" prose, and `SUPPORTED_CAPABILITIES.md:86`.
Honesty tiers (substrate/diagnostic/no-dispatch/`validated_default=False`) and the
Forbidden-claims framing are intact, which keeps this at P1 (reader-misleading
capability narrative) rather than P0 (false promoted claim). Two tests LOCK the
stale wording (correcting it breaks CI) — the inversion of a firewall.

**Q6. Does `promotion_check.py --status` fail closed where it should?**
Yes — executed. Skip≠pass locked end-to-end (real pytest-subprocess skip rejected,
executed green); missing/deleted nodes ⟹ RED-missing; collect/run timeouts ⟹ red;
`PYTEST_ADDOPTS` neutralized. Ledger in §5.

**Q7. Does `enable_collisions=True` leak into canonical inference/public dispatch?**
JAX/batch: no — probed live, ValueError at config (`__post_init__`) and at all three
dispatch sites. SciPy: `canonical_forward_solver(0.0, enable_collisions=True)` with
the default backend **runs to completion** (probed live), wiring the calibrated-RTA
bridge into the SciPy driver (`full_coupled_typeI.py:772-773`, default off, tier
forced ≥2). Documented boundary (guard messages direct users there), and the slow
unsound test (executed green) machine-records that this path's collisional
characteristic reference is anomalous. Not a silent leak; it is an opt-in,
documented, unvalidated path with no runtime "candidate/unsound" marking on its
output — kept at P2 (F-7).

**Q8. Does N_eff-closeness influence claims beyond what tests justify?**
Contained, with wording leaks: `_C_RATE=210` is honestly labelled calibrated; but the
driver docstring's "N_eff = 3.00 exactly" and Gate B's test name
(`…physical_neff…`) overstate what the executed 2.9934 pin and coarse [3.00, 3.15]
band justify — especially given F-1 makes the collisional band physically
uninterpretable. No registry/doc upgrades N_eff-closeness to validation.

**Q9. Are raw failed states preserved?**
Clean core: yes (raw min-Hubble before floor, clip excursion on accepted samples,
refuse-to-floor RuntimeError, anchor red not skipped — all executed or verified).
One exception: candidate-path bridge `nan_to_num` + zeros-on-degenerate-ρ_ref
(`teff_collision_bridge.py:229-231`) silently repairs non-finite collision fields
(F-6).

**Q10. Cheapest executable test most likely to overturn the conclusion?**
Ran it: the Γ/H(T) crossing probe (scratchpad-only, ~20 lines, seconds). It did not
overturn — it *produced* the top finding (constant Γ_relax, inverted crossing). The
next decisive command is equally cheap: recompute the probe with the prefactor
multiplied by T (G_F²T⁵-class) and check the crossing flips to the standard
falling-through-1-near-2-MeV profile; then re-run Gate B.

**Q11 (self-generated). Do any tests lock stale narratives so that honesty fixes break CI?**
Yes — `test_augmented_wbs_status_ledger.py` (all 6 tests) and
`test_augmented_pstf_capability_registry.py` (:101, :154). Any patch deflating the
stale WBS/registry prose must update these two locks in the same commit.

**Q12 (self-generated). Is detailed balance tested on the full callable or only algebraic factors?**
Full callable, three layers, all executed green (§3 item 4) — but DB nulls are
normalization-blind and cannot substitute for a rate-magnitude check.

---

## 8. Ranked findings

### F-1 — Deterministic-reference collision rate is dimensionally inconsistent; T-scaling of the coupled decoupling dynamics is inverted
- **Severity:** P1
- **Status:** IMPLEMENTED code; rate normalization NOT VALIDATED
- **Symptom:** `dQ ∝ T⁴` exactly at fixed spectral shape; Γ_relax = dQ/Δρ constant in T (3.04e-21 MeV for ν_e); Γ/H rises as T falls, crossing 1 near 2.5 MeV from below — standard weak physics has Γ ∝ G_F²T⁵, Γ/H ∝ T³ falling.
- **Source:** `src/rabbit/collisions/deterministic_reference.py:394, 451, 515, 578, 711, 766` (prefactor `G_F_MEV**2 * T**4/(4π³)`, G_F² = MeV⁻⁴ ⟹ dimensionless); consumed as MeV rate at `dynamic_collision_driver.py:194, 201`; no electron mass anywhere in the module.
- **Evidence:** executed probe (§7-Q10); independent dimensional audit; repo-internal contrast: `rate_prefactors.py:9,16` (Γ=(7π/12)G_F²T⁵), `projected_operator.py:70`, `nudec_tables.py:95` — the deterministic reference is the outlier. No test checks magnitude or T-scaling (§6-Q1). BD611 lists this exact check as open (`docs/audit/BD611_…:99-101, :262`).
- **Consequence:** clean-core collisional results (Gate B N_eff ∈ [3.00, 3.15], T_νe>T_νx split, frame-scale band) cannot be physically interpreted; the rate magnitude is right-ordered only near ~2 MeV by coincidence of the constant crossing; energy conservation-by-construction structurally hides it.
- **Minimal remedy:** patch PR-1 (§9) — T-scaling falsification probe/test + prefactor correction; freeze any N_eff wording until it lands (per BD611's own instruction).
- **Proof needed:** corrected-prefactor probe shows standard falling Γ/H ∝ T³ crossing near 1.5-2 MeV; Gate B re-run; a literature-anchored magnitude check at one temperature.

### F-2 — Stale-after-deletion claim narrative, actively LOCKED by two tests
- **Severity:** P1
- **Status:** DEPRECATED content advertised as current; locks IMPLEMENTED
- **Symptom:** deleted AP65-class capability narrated as "landed"; correcting the docs/registry breaks CI.
- **Source/Evidence:** WBS `docs/IMPLEMENTATION_GUIDE_TYPEI_AUGMENTED_PSTF_NOQKE_WBS.md:599` (+81 more rows; named solve function defined nowhere — rg verified); locks `tests/test_augmented_wbs_status_ledger.py:22-36` (with a "planned"-vs-"landed" name/body mismatch) and `tests/test_augmented_pstf_capability_registry.py:101,154`; registry prose `backend_capabilities.py:557-574, 644-647`, `feature_capabilities.py:371-378`; reaches headline surface at `SUPPORTED_CAPABILITIES.md:86`.
- **Consequence:** a careful reader of WBS/STATUS/SUPPORTED substrate prose would believe deleted non-LRS collision/weak-network capability exists in-tree; the firewall tests enforce the false narrative rather than detect it.
- **Minimal remedy:** patch PR-2 — deflate wording to "deleted in PR-D1..D3 (stage record)", regenerate rendered surfaces, update the two locks in the same commit.
- **Proof needed:** rg zero hits for "landed in current workspace" against deleted APs; regenerated SUPPORTED/STATUS; both locks green on corrected wording.

### F-3 — Core 3T energy path normalization inconsistent with driver (frame factor absent)
- **Severity:** P2
- **Status:** IMPLEMENTED, inconsistent; test-only
- **Symptom:** `collision_bank_energy_sources_per_efold` feeds bare spectral moments (docstring claims "[MeV⁵]") into `coupled_3T_rhs_from_collision_moments`, which divides by MeV³ heat capacities; the driver's conserving coupling applies the T_γ⁴/(2π²) frame factor the core path lacks (driver docstring :39-40 "PHYS·(module dQ)").
- **Source:** `dynamic_collision_core.py:217-253`; `nudec_coupled.py:301-305`.
- **Consequence:** two mutually inconsistent energy conventions inside the clean core package; anyone promoting `flrw_dynamic_collision_rhs` would inherit a wrong normalization silently.
- **Minimal remedy:** patch PR-5 — apply the frame factor or mark the path DEPRECATED/test-only in docstrings and fix the "[MeV⁵]"/"exact inputs" claims.
- **Proof needed:** a units-closure assertion test on the core path, or the DEPRECATED marking.

### F-4 — 17 gate-required test nodes across 9 deleted files; staleness undetectable by any test
- **Severity:** P2
- **Status:** IMPLEMENTED fail-safe; hygiene gap
- **Symptom:** `claim_gates.py` references `tests/test_cl3_canonical_promotion.py`, `test_cross_code_cluster.py`, `test_curvature_full.py`, `test_curved_pstf_convergence.py`, `test_flrw_external_parity.py`, `test_h_family_continuity.py`, `test_sampler_end_to_end_synthetic.py`, `test_tier3_decoupled_thermo.py`, `test_tilt_vector_scalar_parity.py` — none on disk; `test_test_node_ids_are_well_formed` checks syntax only.
- **Evidence:** ledger `--status` RED-missing rows (§5); rg/ls verification.
- **Consequence:** promotion fails closed (good) but the gate registry silently rots; future gate edits can't distinguish intentional forward references from post-deletion staleness.
- **Minimal remedy:** inside patch PR-2: annotate each affected gate's docstring/comment as forward-reference-by-design or update node ids to live equivalents (e.g. `test_flrw_external_parity` → `test_cross_code_live`/`test_primat_parity` where semantics match). No new gate surface.
- **Proof needed:** ledger unchanged (still red where intended) + annotations present.

### F-5 — Post-deletion tooling breakage: 6 dead scripts + 2 non-replayable packet builders + stale policy line
- **Severity:** P2
- **Status:** DEPRECATED (dead)
- **Symptom:** `ModuleNotFoundError` at import (demonstrated live for `scripts/assert_negative_disabled.py:3`); generated external-audit packets embed commands naming deleted files.
- **Source:** six scripts (`probe_reduced_modal_direct_from_dump.py:11`, `split_manifest_by_cluster.py:5`, `assert_negative_disabled.py:3`, `evaluate_macro_basis_screen_windowed.py:11`, `build_macro_mode_bank_windowed.py:4`, `dump_scipy_phase1_accepted_states.py:22`); `make_external_reaudit_packet.py:420-474,676`; `make_external_performance_optimization_audit_packet.py:527,616,1121`; `bbn_codex_anti_drift_cost_effective_policy.md:16-17` ("in-tree Rodas5P/AP65 host").
- **Consequence:** broken tooling; an external auditor replaying a generated packet hits collection errors — reputational risk for exactly the audience these scripts serve.
- **Minimal remedy:** patch PR-3 — delete the 6 dead scripts; strip deleted-file references from the 2 packet builders; correct the policy line. Net-negative LOC.
- **Proof needed:** every remaining `scripts/*.py` imports cleanly (compile-only sweep).

### F-6 — Candidate-bridge silent sanitize + one-sided energy bookkeeping + env-knob parity divergence
- **Severity:** P2 (candidate path only; escalates if the path is ever promoted)
- **Status:** IMPLEMENTED, under-documented
- **Symptom:** (a) `nan_to_num` on non-finite C and zeros on degenerate ρ_ref (`teff_collision_bridge.py:229-231`) — the repo's one true silent-repair site; (b) δI normalized by fixed equilibrium `rho_ref` while the docstring says ρ_ν (:207 vs :226,:239); (c) `delta_rho_nu` never consumed and the tier-2 3T plasma channel never debited (`driver_typeI_char.py:774, 801`) — undocumented double-representation risk; (d) numpy env knobs `RABBIT_COLLISION_QMAX/BRIDGE_RELAX` vs JAX static defaults — "parity-locked" claim silently breaks under non-default env.
- **Minimal remedy:** patch PR-4 — replace the sanitize with a fail-closed non-finite check on the candidate path; one docstring paragraph stating (b)/(c)/(d) explicitly.
- **Proof needed:** a non-finite-C injection test raising instead of zeroing.

### F-7 — Public SciPy opt-in collisional dispatch runs an admittedly anomalous model without output marking
- **Severity:** P2
- **Status:** IMPLEMENTED, documented boundary
- **Symptom:** `canonical_forward_solver(0.0, enable_collisions=True)` (default backend) runs to completion (probed live) on the SciPy calibrated-RTA path whose collisional characteristic reference the repo's own executed test records as anomalous (`test_b4_collisional_char_reference_unsound.py`, N_eff far from 3.044).
- **Evidence:** `forward_likelihood.py:2588-2620` (tier forced ≥2 :2592); `full_coupled_typeI.py:772-773, 1039` (default False); live probe output.
- **Consequence:** a user can obtain unvalidated collisional Y_p/N_eff from the public entrypoint with one flag and no "candidate/diagnostic" marking on the result.
- **Minimal remedy:** within patch PR-4: attach a metadata field (`collision_model="calibrated_rta_candidate"`) or a single warning at the dispatch site — no new gate/wrapper surface.
- **Proof needed:** probe re-run shows marked output; JAX/batch fences unchanged.

### F-8 — Wording overstatements on the clean core and gate-test docstrings
- **Severity:** P3
- **Status:** IMPLEMENTED code, stale/overstated words
- **Symptom:** "N_eff = 3.00 exactly"/"analytic 3" (driver docstring :6-9, :51) vs the executed 2.9934±1e-3 pin with an unstated EOS convention inherited from `nudec_coupled` defaults; anchor-test docstring :9-13 describes pre-BD601 returncode semantics; ledger-lock test name says "planned" while asserting "landed"; Gate B test name `…physical_neff…` overstates given F-1.
- **Minimal remedy:** fold one-line docstring corrections into PR-2/PR-5.

### F-9 — Dead-island and literal-constant hygiene
- **Severity:** P3
- **Status:** IMPLEMENTED, deliberate islands
- **Symptom:** clean core (`integrate_flrw_decoupling`) has zero non-test consumers; the augmented quartet is re-exported by `transport/__init__.py:33-71` with zero production call sites (import-reachable ≠ call-graph-reachable); degeneracy literals 2.0/4.0 hard-coded at driver :210-211 linked to `BANK_DEGENERACY` by comment only; JAX twins have no x64 guard outside pytest.
- **Minimal remedy:** no action now beyond PR-2's registry wording ("operator-level island, no driver capability"); do not delete the substrate (it is the B5 seed).

---

## 9. Minimal patch plan

| Patch | Files | Net LOC est. | Blocker moved | Why now | Test/command | Risk | Cost-effectiveness verdict |
|---|---|---:|---|---|---|---|---|
| PR-1 Collision-rate dimension / Γ/H scaling probe + prefactor repair | `src/rabbit/collisions/deterministic_reference.py` (+1 new test file) | +80 | THE physics-normalization blocker (F-1; BD611 open item ":262") | Every collisional clean-core number is uninterpretable until fixed; probe already written and seconds-cheap | new `test_deterministic_reference_rate_scaling.py`: assert dQ(2T)/dQ(T) ≈ 2⁵-class at fixed shape + Γ/H falling profile; re-run Gate B trio | Prefactor change shifts Gate B band — expected and desired; requires a short reduction derivation to fix the constant too (4π³ vs 2π³ unverified) | ACCEPT — highest blocker-movement per line in the repo |
| PR-2 Stale claim-surface deflation (F-2, F-4, F-8 wording) | WBS doc, `backend_capabilities.py`, `feature_capabilities.py`, `tests/test_augmented_wbs_status_ledger.py`, `tests/test_augmented_pstf_capability_registry.py`, `claim_gates.py` annotations, regenerate README/STATUS/SUPPORTED/PROMOTION_GATES, driver docstring one-liners | −150 | Claim-surface honesty after −183.7K-LOC deflation | Two tests currently make honesty fixes a CI failure; headline surface carries "landed AP0-AP81" | `render_capability_tables.py --apply`; both lock tests green on corrected wording; `promotion_check.py --status` unchanged | Lock-test updates must land atomically with doc changes | ACCEPT — deletes stale surface, no new gates |
| PR-3 Deletion/lazy-import hygiene (F-5) | 6 dead scripts (delete), 2 packet builders, policy doc line | −900 | Post-deflation tooling correctness; external-packet replayability | Demonstrated live ModuleNotFoundError; packets mislead external auditors | compile-only sweep: `python -m py_compile scripts/*.py`; regenerate one packet and grep for deleted paths | Low — dead code | ACCEPT |
| PR-4 Candidate-path fail-closed hardening + output marking (F-6, F-7) | `transport/teff_collision_bridge.py`, `jax/teff_collision_bridge_jax.py` (mirror), `inference/forward_likelihood.py` (metadata tag at :2588-2620), docstring paragraph | +40 | Preserve-raw-failed-states rule on the candidate surface; public-output honesty | The one silent-repair site; unmarked anomalous-model output from public entry | non-finite-C injection test raises; probe re-run shows `collision_model` metadata; JAX parity suite green | Guard-order parity with JAX twin must be preserved (mirror both) | ACCEPT — small, fences candidate path tighter without new surfaces |
| PR-5 Core 3T path units closure (F-3) | `dynamic_collision_core.py` (docstrings + either frame factor or DEPRECATED marking) | +15 | Internal consistency of the clean core package | Prevents silent inheritance of the wrong convention by any future promotion of `flrw_dynamic_collision_rhs` | units-closure assertion test on core path vs driver path at one state | Low | ACCEPT |

Disfavored and not proposed: new readiness/manifest/figure surfaces, claim ledgers,
broad rewrites, segment optimization.

---

## 10. Final verdict

**PARTIAL_ENGINEERING_PASS_PHYSICS_BLOCKED**

1. **One-line reason:** the clean core's engineering (conditioning fix, fail-closed endpoint, fences, promotion firewall) executed green under hostile testing, but its collision rate is dimensionally inconsistent with an inverted Γ/H(T) profile and no magnitude test — the physics interpretation of every collisional result is blocked (with a P1 stale-claim narrative cluster riding second).
2. **Strongest surviving contribution:** the BD205 energy-variable conditioning repair plus the fail-closed driver endpoint and the skip≠pass/red-anchor promotion machinery — all executed, none refuted.
3. **Most dangerous overclaim:** "landed AP0-AP81" WBS/registry narrative for deleted capability, enforced by two locking tests (F-2) — closely followed by the implicit "physically normalized first-principles collision rate" reading of the clean core (F-1).
4. **Next single patch:** PR-1 (collision-rate dimension/Γ/H scaling probe + prefactor repair).
5. **Must not be touched yet:** the `enable_collisions` fences and the RED finite-shear external anchor (`test_finite_shear_external_anchor.py` + `GATE_SIGMA_H_TO_0P95`) — they are the honesty spine; no fixture, no skip conversion, no gate relaxation.
6. **Commands run** (all with `env -u RABBIT_NUDEC_BSM_PATH PYTHONPATH=src JAX_PLATFORMS=cpu venv/bin/python`):
   - `git status --short; git log --oneline -30; git rev-parse HEAD; --version checks`
   - `find src/rabbit scripts tests -name '*.py' | xargs wc -l | sort -n | tail -40`
   - required `rg` stale-wording sweep (3513 hits; per-file counts recorded)
   - `-m pytest -q -p no:cacheprovider -m "not slow"` batch1 (65 passed, 3.95 s), batch2 (58 passed, 17.4 s), batch3 (32 passed + designed-RED anchor failure, 0.94 s)
   - `-m pytest … -m slow tests/test_dynamic_collision_driver.py` (3 passed, 45.0 s) and `tests/test_b4_collisional_char_reference_unsound.py` (1 passed, 16.1 s)
   - `scripts/promotion_check.py --status` (ledger §5)
   - live probes: Γ/H crossing probe; `JAXTypeICharConfig(enable_collisions=True)` fence probe; `canonical_forward_solver` dispatch probes; `scripts/assert_negative_disabled.py` ModuleNotFoundError demonstration
7. **Commands skipped and why:**
   - full test suite (~3172 nodes): out of BD612 scope; required batches + slow set executed instead
   - `cross_code`-marked external parity (`test_cross_code_live.py`): requires external backends (NUDEC_BSM env) not configured; skips would not count as passes anyway
   - `test_finite_shear_matches_external_oracle` with a benchmark: impossible by design — no external finite-shear benchmark exists; the designed failure was executed and recorded instead
   - B5 substrate test files (`test_augmented_*`, `test_b5_*`): not in the BD612 required command set; classified statically
   - independent multi-agent adversarial verification passes: external resource limit; §7 was performed inline against primary sources with two-auditor independence only on F-1

---

## 11. Changed-files risk note

This audit changed no repository file except creating this report
(`docs/audit/BD612_current_code_hostile_audit_report_2026-07-08.md`, untracked).
The patch plan (§9) proposes future edits with these risks:

- **PR-1** changes physical output of the clean core (intended); Gate B pins and the
  2.9934 collisionless pin must be re-derived deliberately, never adjusted to
  preserve green.
- **PR-2** must update `tests/test_augmented_wbs_status_ledger.py` and
  `tests/test_augmented_pstf_capability_registry.py` in the same commit as the
  doc/registry wording or CI breaks; regenerated README/STATUS/SUPPORTED must come
  from `render_capability_tables.py --apply`, not hand edits.
- **PR-3** deletes scripts; verify none is referenced by external runbooks before
  removal (rg over docs/ first).
- **PR-4** touches the numpy/JAX bridge pair; any guard-order change must be
  mirrored in both twins or the parity suite (executed green today) will fail.
- **PR-5** must not alter the driver's live conserving coupling — only the core's
  test-only path and its docstrings.
