# RABBIT PR Completion Catalogue

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

> **PUB-00 continuation rule (2026-07-12).**  Keep this file as historical
> landed-work provenance.  New publication-code PRs are specified only in
> `TYPEI_AUGMENTED_NOQKE_UNIFIED_FUTURE_PLAN.md`; their detailed evidence stays
> in the PR body plus the existing claim/validation ledgers rather than growing
> this catalogue.

Historical append-only log of completed PRs.  Entries below retain their
original chronology and terminology; they do not establish current capability
or dependency order.

Companion documents: [ROADMAP_INDEX.md](ROADMAP_INDEX.md),
[ROADMAP_STATE_OF_RECORD.md](ROADMAP_STATE_OF_RECORD.md),
[ROADMAP_PR_WBS.md](ROADMAP_PR_WBS.md),
[ROADMAP_SELF_AUDIT.md](ROADMAP_SELF_AUDIT.md).

---

## 0.  Entry template

```markdown
### PR-<ID>  <one-line summary>

- **Status:** merged / reverted / partial
- **Scope:** component(s) changed
- **Key files:** `path/to/file.py:Lx-Ly` (deltas only)
- **Physics added/changed:** paper equation refs, new primitives
- **Parity before / after:** worst-case numbers from audit §2.2
- **Performance before / after:** timing + VRAM figures
- **Known red tests:** any test the PR flipped or left in a pre-existing state
- **Docs updated:** list of files touched for doc sync
- **Self-audit verdict:** pass / conditional / fail with link to adversarial review
```

---

## PR-CHAR  JAX characteristic-ray Type I tier-1 driver

- **Status:** merged
- **Scope:** new JAX driver
  (`src/rabbit/jax/driver_typeI_char.py`), dispatch hook in
  `driver_typeI.py`, capability registration, parity tests.
- **Key files:**
  - `src/rabbit/jax/driver_typeI_char.py` — new file, ~700 LOC.
  - `src/rabbit/jax/driver_typeI.py` — adds `transport_mode`,
    `N_mu`, and the early-dispatch branch to the new driver.
  - `src/rabbit/config/backend_capabilities.py` — registers
    `JAX_TYPEI_CHARACTERISTIC_TIER1` and the `jax_characteristic`
    backend key.
  - `src/rabbit/inference/forward_likelihood.py` — adds a
    `backend="jax_characteristic"` dispatch branch.
  - `tests/test_jax_typeI_characteristic_parity.py` — 18 tests.
- **Physics added/changed:**  First JAX implementation of the paper-§6
  characteristic-ray method (eq 41–58); state layout matches SciPy
  production 1-to-1; live weak rates at CL0–CL3 from the transported
  monopole `f̃₀(q)`.
- **Parity before / after:**  Before — JAX Bianchi I publication parity
  was `@xfail` (linearised PSTF captured only ~21 %% of the anisotropic
  Y_p shift).  After — |ΔY_p| ≤ 4 × 10⁻⁸ across Σ_H ∈ [0, 0.5], CL0–CL2;
  `test_jax_matches_scipy_publication_scope_anisotropic_backbone` flips
  to green.
- **Performance before / after:**  Warm 1-solve: 5–6 s → 1.3 s after the
  stable-identity RHS cache; CPU-preferred default avoids the 12 GB
  VRAM preallocation.
- **Known red tests:**  Unchanged — the four pre-existing reds
  (see [STATE_OF_RECORD.md §6.1](ROADMAP_STATE_OF_RECORD.md#61-known-pre-existing-red-tests-not-caused-by-recent-work))
  remained red at the time.  One of them (`test_anisotropy_signal_parity`)
  has since been closed by PR-D, which promoted the bounded
  `backend='auto'` surface onto JAX characteristic transport.
- **Docs updated:**
  - `STATE_OF_RECORD.md` — filed under §2.3, §3.1–§3.6, §4.1, §5.1, §5.2.
  - `JAX_CHAR_GPU_OPTIMIZATION_PLAN.md` — references the stable-identity
    cache fix and the CPU-preferred policy.
- **Self-audit verdict:**  Pass.  Third-party auditor report:
  "No hallucinated physics, no mock surrogates, no off-by-one, no
  unit mismatches.  The file is a faithful JAX port of the SciPy
  tier-1 shared characteristic path."

---

## PR-T3T  Tier-2 three-temperature thermodynamics (JAX characteristic)

- **Status:** merged
- **Scope:** tier-2 upgrade for the JAX characteristic driver;
  dedicated backend key; parity tests.
- **Key files:**
  - `src/rabbit/jax/driver_typeI_char.py` — adds
    `thermo_tier` to the config, extends `_char_layout` with
    `i_tne, i_tnx`, branches `_rhs_core` between tier-1
    (`tier1_dT_gamma_dN_jax`, `tier1_hubble_aniso_jax`) and tier-2
    (`coupled_3T_rhs_jax`, `hubble_3T_jax`), updates the cache key
    with `thermo_tier`, extends initial-condition packing and
    handoff, reports `T_nu_e_final` and `T_nu_x_final` in the result.
  - `src/rabbit/jax/driver_typeI.py` — accepts `thermo_tier ∈ {1, 2}`
    for `transport_mode="characteristic"` and forwards via the
    dispatch branch.
  - `src/rabbit/config/backend_capabilities.py` — registers
    `JAX_TYPEI_CHARACTERISTIC_TIER2` and the backend key
    `jax_characteristic_tier2`.
  - `src/rabbit/inference/forward_likelihood.py` — accepts
    `backend="jax_characteristic_tier2"` and forwards `thermo_tier=2`
    to the driver.
  - `tests/test_jax_typeI_characteristic_tier2.py` — 28 tests.
- **Physics added/changed:**  Couples the characteristic-ray transport
  to the paper-eq-(161–163) 3T thermodynamic sector; weak rates now
  use `T_νₑ` rather than the tier-1 entropy helper; Mangano
  momentum-averaged ν–e source term (`nudec_coupled_jax.py`) supplies
  the reheating, producing `N_eff ≈ 3.034` at FLRW and the expected
  heating asymmetry `T_νₑ > T_νₓ` (paper §8.2 a_e/a_x ≈ 4.68).
- **Parity before / after:**  Before — no tier-2 characteristic JAX
  path existed.  After — |ΔY_p| ≤ 7 × 10⁻⁸ across Σ_H ∈ [0, 0.5],
  CL0–CL2 against the (newly-fixed) SciPy char tier-2 reference; 3e-7
  headroom on `N_eff`.
- **Performance before / after:**  Warm 1-solve: 1.3 s (unchanged
  relative to tier-1 — the extra 2 DOF has negligible Jacobian
  impact).
- **Known red tests:**  Unchanged vs the stable baseline.
- **Docs updated:**
  - `STATE_OF_RECORD.md` — §2.1 (SciPy status), §2.3, §2.4, §3.4,
    §4.2, §5.
  - `IMPLEMENTATION_GUIDE_3T_THERMO.md` — status transitioned from
    "planned" to "delivered in PR-T3T".
- **Self-audit verdict:**  Pass.  Direct SciPy-JAX parity test
  `test_scipy_jax_tier2_parity` added and locked at `|ΔY_p| < 5 × 10⁻⁷`.

---

## PR-S1  SciPy tier-2 fall-through + PSTF UnboundLocalError fix

- **Status:** merged
- **Scope:** SciPy reference driver correctness fixes.
- **Key files:**
  - `src/rabbit/drivers/full_coupled_typeI.py` —
    1. Hoisted the monopole → weak → network → pack → return block
       out of the tier-1 `else:` sub-branch and back inside the
       `if use_char:` scope so that `tier ≥ 2` no longer falls through
       to the LINEARIZED_PSTF code.  Comment on lines 784–793 retained
       as a permanent record.
    2. Added `T_gamma_for_rates = T_gamma` in the tier-2 branch of the
       LINEARIZED_PSTF path so the subsequent weak-rate call does not
       hit `UnboundLocalError`.
  - `tests/test_jax_typeI_characteristic_tier2.py` — extended to pin
    direct SciPy ↔ JAX parity (12 new cases across Σ × CL).
- **Physics added/changed:**  None.  The SciPy driver is now
  structurally correct for `transport_mode=CHARACTERISTIC, tier=2,
  enable_collisions=False` and for
  `transport_mode=LINEARIZED_PSTF, tier=2`; previously both raised at
  the first RHS evaluation.
- **Parity before / after:**  Before — IndexError and
  UnboundLocalError prevented SciPy from being used as a tier-2
  reference; JAX tier-2 numbers were cross-checked only against the
  PSTF path indirectly.  After — direct SciPy ↔ JAX char-tier-2
  parity at |ΔY_p| ≤ 7 × 10⁻⁸ across the full grid.
- **Performance before / after:**  SciPy char-tier-1 values
  unchanged (regression-lock confirmed against the pre-fix numbers).
- **Known red tests:**  Unchanged vs the stable baseline; the four
  pre-existing reds remain.
- **Docs updated:**
  - `STATE_OF_RECORD.md §2.1` (SciPy status row for tier-2
    collisions=OFF flipped from "broken" to "fixed"), `§3.5` (design
    decision added).
- **Self-audit verdict:**  Pass.  Regression-lock confirmed on all
  tier-1 Σ × CL combinations; tier-2 scope directly validated with
  the 12-case grid.

---

## PR-OPT  Characteristic-driver performance lock + VRAM policy

- **Status:** merged (grouped with PR-CHAR / PR-T3T; recorded here for
  provenance of the individual optimisation decisions)
- **Scope:** performance + VRAM behaviour of the JAX characteristic
  driver.
- **Key files:**
  - `src/rabbit/jax/driver_typeI_char.py` — adds
    `runtime_device_policy` with default `"cpu_preferred"`,
    `jacobian_mode` (default `"dense"`, `"block_sparse"` opt-in),
    `_CHAR_RHS_CACHE` host-side stable-identity cache, `_active_indices`
    for Schur Jacobian.
- **Physics added/changed:**  None.  Pure performance / policy work.
- **Parity before / after:**  Unchanged (regression-locked by the
  existing parity suite).
- **Performance before / after:**
  - Warm 1-solve at Σ = 0.1, CL0, N_q=20: 5–6 s → **1.3 s** (4.6 ×).
  - 10-Σ parameter scan: 51 s → **13 s** (4 ×).
  - Full parity test suite runtime: 476 s → **74 s** (6.4 ×).
  - VRAM (CPU-preferred default): ~12 GB → **0 GB**.
- **Known red tests:**  Unchanged.
- **Docs updated:**
  - `JAX_CHAR_GPU_OPTIMIZATION_PLAN.md` — §2 lists the optimisations
    actually applied and the ones deferred.
  - `STATE_OF_RECORD.md §3.1–§3.3` records the design decisions and
    their rationale.
- **Self-audit verdict:**  Pass.  Dense-Jacobian beats block-sparse at
  37 DOF by ~1.4 × on warm runs (block-sparse is kept as opt-in
  because it is expected to flip at larger state / on GPU).

---

## PR-DOCS  Roadmap documentation set

- **Status:** merged
- **Scope:** documentation only.  Establishes the roadmap/SDD/WBS
  structure used from this point onward.
- **Key files:**
  - `docs/ROADMAP_INDEX.md` — master index.
  - `docs/ROADMAP_STATE_OF_RECORD.md` — project state of record.
  - `docs/ROADMAP_PR_WBS.md` — forward PR roadmap.
  - `docs/ROADMAP_SELF_AUDIT.md` — audit + doc-update protocol.
  - `docs/ROADMAP_PR_CATALOG.md` — this file.
  - `docs/JAX_CHAR_GPU_OPTIMIZATION_PLAN.md` — referenced.
  - `docs/IMPLEMENTATION_GUIDE_NON_LRS_TYPEI.md` — referenced.
  - `docs/IMPLEMENTATION_GUIDE_3T_THERMO.md` — referenced.
  - `docs/IMPLEMENTATION_GUIDE_FULL_COLLISION_NOQKE.md` — referenced.
- **Physics added/changed:**  None.
- **Parity before / after:**  Unchanged.
- **Performance before / after:**  Unchanged.
- **Known red tests:**  Unchanged.
- **Docs updated:**  The five new `ROADMAP_*.md` files constitute the
  update.  Every topic guide now points back into the roadmap set via
  cross-references.
- **Self-audit verdict:**  Pass.  The five documents form a closed
  reference set: a reader starting from

---

## PR-T3A  Private collisionless full phase-space shell

- **Status:** partial
- **Scope:** private tier-3 preflight surface only.  Adds
  `src/rabbit/jax/q_advection_jax.py`,
  `src/rabbit/jax/driver_typeI_full_boltzmann.py`, and the dedicated
  regression file `tests/test_pr_t3a_collisionless_driver.py`.
- **Key files:**
  - `src/rabbit/jax/q_advection_jax.py` — continuous upwind
    semidiscrete q-advection operators on Laguerre nodes, physical
    inflow boundary rows, and the exact-remap PCHIP oracle.
  - `src/rabbit/jax/driver_typeI_full_boltzmann.py` — CPU-only
    collisionless `(Σ_+, Σ_-, f_{species,ray,q}, S, T_γ, X_i)` driver
    with phase-1/phase-2 handoff, transported-monopole weak rates, and
    projected low-rank Jacobian payload emission through the experimental
    `linear_solver_prepare_fn` / `linear_solver_solve_fn` hook.
  - `tests/test_pr_t3a_collisionless_driver.py` — inflow-boundary lock,
    PCHIP oracle bound check, FLRW reduction, bounded anisotropic
    reduction, and tier-1 state-dimension contract.
- **Physics added/changed:**  First landed per-ray per-momentum
  collisionless Type-I shell.  Transport is now carried explicitly on
  the `4 × N_μ × N_q` ray/q state rather than reconstructed from the
  analytic intensity shift.  The driver remains LRS-only, tier-1 only,
  and collisionless-only; ν–e / pair / ν–ν operators are still future
  PRs.
- **Parity before / after:**  Before — no explicit full phase-space
  JAX shell existed.  After — FLRW reduction at `(N_μ,N_q)=(4,6)`
  gives `|ΔY_p| ≈ 6.0×10⁻⁷`, `|ΔD/H| ≈ 9.5×10⁻⁹`, and
  `|ΔX_n| ≈ 3.2×10⁻⁷` against the characteristic driver.  At
  `(N_μ,N_q)=(12,20), Σ_H=0.1` the bounded collisionless drift is
  `|ΔY_p| ≈ 6.5×10⁻⁴`, `|ΔD/H| ≈ 5.4×10⁻⁸`, `|ΔX_n| ≈ 3.4×10⁻⁴`.
- **Performance before / after:**  The full shell is materially heavier
  than compact characteristics.  At `N_μ=12, N_q=20` the landed phase-2
  state is `973` DOF and requires the custom low-rank linear-solver
  hook to remain usable.  A bounded `Σ_H=0.3` CPU audit exceeded the
  audit budget and was aborted; this is not a production-ready path.
- **Known red tests:**  None introduced in the bounded regression set.
  High-shear coarse-grid reduction remains intentionally loose and is
  documented in the audit rather than promoted to a strict parity gate.
- **Docs updated:**
  - `docs/ROADMAP_STATE_OF_RECORD.md`
  - `docs/IMPLEMENTATION_GUIDE_FULL_COLLISION_NOQKE.md`
  - `docs/audit/PR-T3A.md`
- **Self-audit verdict:**  Conditional pass.  The collisionless shell,
  oracle, and solver-hook wiring are landed and regression-locked, but
  the path remains private/candidate until collision operators, tier-2
  thermo, and a bounded high-shear audit close the remaining tier-3
  gaps.

---

## PR-T3B-PF  Species-resolved isotropic collision-core preflight

- **Status:** partial
- **Scope:** host-side preflight plus a private runtime hook.  Adds
  `src/rabbit/jax/full_boltzmann_collision_preflight.py` and
  `tests/test_pr_t3b_collision_preflight.py`, then reuses the same
  closure inside `src/rabbit/jax/driver_typeI_full_boltzmann.py`.
- **Key files:**
  - `src/rabbit/jax/full_boltzmann_collision_preflight.py` — extracts
    bank monopoles from the explicit `(species, ray, q)` state, reuses
    the existing species-resolved isotropic collision backbone,
    materializes a finite-difference moment-core Jacobian on the
    stacked bank state `[f_νe, f_ν̄e, f_νx]`, and lifts that core back to
    the explicit ray state as a factorized `U C V` update.
  - `tests/test_pr_t3b_collision_preflight.py` — bank gather lock,
    equal-temperature equilibrium zero, ν_e vs ν_x coupling hierarchy,
    source-only vs state-dependent Jacobian structure, and lifted
    factorization parity against dense finite differences.
- **Physics added/changed:**  No runtime physics surface changed.  This
  patch first quantified what the current isotropic collision backbone
  can and cannot provide to the future full-collision JAX shell, then
  landed a private `collision_mode="spectral_relaxation_preflight"`
  runtime path on the explicit shell for bounded CPU smoke/audit use.
  A follow-up bounded extension now also allows `thermo_tier=2` on that
  private surface, feeding the lifted collision moments into the JAX 3T
  thermo primitives while keeping the path non-canonical.  The latest
  bounded extension adds `collision_mode="projected_physical_preflight"`
  on the same shell: a state-dependent projected-physical bank closure
  with exact dense-AD parity on the bounded tier-2 regression surface.
  A further host-side-only extension now adds `closure_mode="direct_kernel"`
  in the preflight module, evaluating the existing Hannestad–Madsen
  `ν-e` and pair operators on the species banks through an explicit
  thermal-variable remap.  That same bounded preflight now also restores
  the required `1/H` scaling and materializes an augmented active-scalar
  Jacobian on `[f_bank, T_gamma, T_nu_e, T_nu_x, H]`.  A follow-up
  private runtime candidate now threads the same direct operator through
  the full-Boltzmann RHS via host callbacks while supplying an explicit
  mixed low-rank Jacobian payload to Rodas5P.
- **Parity before / after:**  Before — no explicit moment-core
  preflight existed.  After — the host-side bank core is regression
  locked; equal-temperature FD inputs return zero collision source, and
  hotter-plasma cases show the expected `ν_e > ν_x` hierarchy.  On the
  bounded private tier-2 runtime surface, the factorized Jacobian now
  matches dense AD exactly on all non-thermo rows, with only the three
  thermo rows left at bounded preflight accuracy
  (`max |ΔJ| < 3 × 10⁻³`) for the spectral-relaxation mode.  The newer
  projected-physical runtime mode closes that remaining gap on the
  bounded regression point and matches dense AD on all rows.  The
  operator-backed host preflight now additionally locks detailed balance
  and flavour hierarchy for the direct `ν-e` + pair bank response.
- **Performance before / after:**  Host-side only and bounded.  The
  finite-difference bank Jacobian is small (`(3N_q+3) × 3N_q`) and the
  lifted explicit-state factorization avoids materializing the transport
  matrix except during audit/design checks.  The private runtime hook
  preserves the same factorized payload shape and keeps the default
  public path collisionless.  At the bounded `(N_μ, N_q) = (4, 6)`
  tier-2 regression point, the phase-state sizes are `104` and `111`
  with low-rank factor shapes `(state,26) · (26,37) · (37,state)`.
- **Known red tests:**  None.  The main finding is conceptual: the
  production-bounded source-only closure has zero state Jacobian.  A
  nontrivial moment-core appears only when an audit-only state-dependent
  spectral-relaxation term is included.  The new tier-2 runtime hook is
  still bounded-preflight only on the thermo rows and must not be
  mistaken for a physical Stage-B collision closure.  The projected-
  physical follow-up removes that thermo-row parity gap at bounded
  `(N_μ, N_q) = (4, 6)`, but it remains a projected preflight rather
  than the final ν-e / pair operator.  The direct-kernel follow-up is
  no longer host-only: a bounded private runtime candidate now exists at
  the `rhs/jacobian` level.  The remaining gap is end-to-end solve
  smoke and bounded parity on that callback-plus-explicit-Jacobian path.
- **Docs updated:**
  - `docs/ROADMAP_STATE_OF_RECORD.md`
  - `docs/IMPLEMENTATION_GUIDE_FULL_COLLISION_NOQKE.md`
  - `docs/audit/PR-T3B_preflight.md`
- **Self-audit verdict:**  Conditional pass.  This closes the bounded
  design question for Stage B and opens a private runtime hook, but it
  still does not provide a physical Stage-B collision closure or upgrade
  the path to a public backend.
  `ROADMAP_INDEX.md` can reach every other roadmap document and every
  topic guide in at most two hops.

---

## PR-T3B-PF  JAX-native ν-e + pair operator port (preflight)

- **Status:** partial
- **Scope:** strictly additive preflight.  Adds
  `src/rabbit/jax/collisions_jax.py` and
  `tests/test_pr_t3b_jax_operator_parity.py`.  No driver, public
  backend, inference dispatch, SciPy reference, or `coupled_3T_rhs_jax`
  is touched.
- **Key files:**
  - `src/rabbit/jax/collisions_jax.py` — JIT-compatible ν-e and pair
    collision kernels.  Algorithmic mirror of
    `NuEScatteringOperator._evaluate_vectorized` and
    `PairProcessOperator.evaluate`: identical matrix element,
    statistical factor, prefactor, `1/y1²` divisor, near-zero `y1`
    skip, and quadrature schemes.  Off-node interpolation reuses the
    existing PCHIP cubic Hermite from `q_advection_jax`.
  - `tests/test_pr_t3b_jax_operator_parity.py` — element-wise SciPy ↔
    JAX parity on the shared Laguerre grid, NuE / pair detailed balance
    locks, off-grid PCHIP-vs-scipy-cubic gap measurement, and JIT
    determinism.
- **Physics added/changed:**  No new physics.  The patch lifts the
  existing SciPy collision operators onto a pure-JAX, JIT-compatible
  surface so the future full-Boltzmann GCS cycle does not require host
  callbacks.  Coupling constants and matrix elements are reused
  verbatim from `rabbit.collisions.kernels` /
  `nu_e_scattering._matrix_element` /
  `pair_processes._matrix_element_ann`.
- **Parity before / after:**  Before — no JAX-native port existed; the
  preflight bridge invoked SciPy via host callbacks.  After:
  - NuE shared-Laguerre (`N_q ∈ {12, 20, 24}`, both species):
    `|Δ| < 1e-30` absolute element-wise vs SciPy.
  - NuE detailed balance at `f = f_FD`: `max|C| < 1e-30` (both species).
  - Pair shared-Laguerre (`N_q == N_quad == 24`, both species):
    `|Δ| < 1e-30` absolute or `< 1e-12` relative element-wise vs SciPy
    on a 50% above-FD perturbation.
  - Pair detailed balance at matched grid (`N_q == N_quad == 24`):
    `max|C| < 1e-30` (both species).
  - Pair off-grid (`N_q = 20`, `N_quad = 24`): measured PCHIP-vs-cubic
    relative gap `~8.3%`, locked at `< 10%`.
- **Performance before / after:**  No production runtime change; the
  pure-JAX kernels are not yet wired into the driver.
- **Known red tests:**  None.  Regression bundle
  (`test_pr_t3a_collisionless_driver.py +
  test_pr_t3b_collision_preflight.py +
  test_pr_t3b_jax_operator_parity.py +
  test_jax_typeI_characteristic_parity.py`):
  `59 passed in 225.7 s` (43 prior + 16 new).
- **Docs updated:**
  - `docs/audit/PR-T3B_jax_operator_port.md`
  - `docs/ROADMAP_PR_CATALOG.md`
- **Self-audit verdict:**  Conditional pass.  Phase-prompt items (1)
  and (2) of PR-T3B (operator parity and detailed balance on the
  shared Laguerre grid) are now closed in additive form.  Items
  (3)–(6) — FLRW `N_eff` lock, `dQ_α` sign convention, Rodas5P step
  rejection, and `coupled_3T_rhs_jax` backward compatibility — remain
  open until the GCS wiring step lands.  The off-grid pair-process
  parity gap should be revisited with a JAX-native natural cubic
  spline before promoting the path to the public driver surface.

---

## PR-T3B-PF  JAX-kernel runtime mode (preflight)

- **Status:** partial
- **Scope:** wires the pure-JAX `nu_e_collision_jax` /
  `pair_collision_jax` from the prior preflight slice into the
  full-Boltzmann driver as a new private `collision_mode=
  "jax_kernel_preflight"`.  Strictly bounded: no public backend, no
  inference dispatch, no capability-registry promotion.  Refactors the
  driver's nested-ternary metadata contract resolution into helper
  functions for readability.
- **Key files:**
  - `src/rabbit/jax/driver_typeI_full_boltzmann.py` — adds
    `_collision_jax_kernel_bank_core_jax` (pure-JAX bank-core that
    calls the JAX collision kernels at the matched Laguerre grid),
    routes `"jax_kernel_preflight"` through `_collision_bank_core_jax`,
    and refactors the metadata contracts via
    `_resolve_jacobian_payload_contract` /
    `_resolve_jacobian_transport_projector_contract` /
    `_resolve_jacobian_low_rank_moment_dim` /
    `_resolve_jacobian_low_rank_apply_dim` (grouped by
    `_PURE_JAX_BANK_CORE_MODES`).  Validation extended in
    `JAXFullBoltzmannConfig.__post_init__`.
  - `tests/test_pr_t3a_collisionless_driver.py` — adds two new tests:
    rhs/jacobian smoke and FLRW detailed-balance regression.
- **Physics added/changed:**  No new physics.  The new mode lifts the
  existing JAX collision kernels from a host-callback path
  (`direct_kernel_preflight`) onto a fully JIT-compatible
  bank-core path.  Quadrature grids inside the bank-core are matched
  to `q_nodes` so the algebraic detailed balance at `T_nu = T_gamma`
  collapses without interpolation noise.
- **Parity before / after:**  Before — the only runtime path that
  exercised the actual ν-e / pair operators was
  `direct_kernel_preflight`, which uses host callbacks per RHS call.
  After — `jax_kernel_preflight` is a parallel pure-JAX runtime mode
  with detailed balance on transport rays measured at `6.7e-14` at
  FLRW, T=10 MeV, locked at `< 1e-10`.
- **Performance before / after:**  No production runtime change; the
  new mode is private and audit-only.  The pure-JAX path eliminates
  per-RHS host callbacks but is currently used only inside smoke
  tests at `(N_mu, N_q) = (4, 6)`.
- **Known red tests:**  None.  Targeted regression bundle
  (`test_pr_t3a_collisionless_driver.py +
  test_pr_t3b_collision_preflight.py +
  test_pr_t3b_jax_operator_parity.py +
  test_jax_typeI_characteristic_parity.py`):
  `61 passed in 220.6 s` after the wiring step (Phase 2);
  `44 passed in 189.8 s` on a follow-up Phase 3 pass that adds the
  end-to-end Rodas5P solve smoke
  (`test_full_boltzmann_private_tier2_jax_kernel_smoke`) on the new
  mode at `Σ=0.05`, `(N_mu, N_q) = (4, 6)`, exercising phase-1 +
  phase-2 with the expected `T_νe / T_νx` heating asymmetry.
- **Docs updated:**
  - `docs/audit/PR-T3B_jax_kernel_runtime.md`
  - `docs/ROADMAP_PR_CATALOG.md`
  - `docs/ROADMAP_STATE_OF_RECORD.md`
- **Self-audit verdict:**  Conditional pass.  The pure-JAX nu-e +
  pair collision operators are now reachable at runtime without host
  callbacks via the standard bank-core dispatcher, and an end-to-end
  Rodas5P solve through phase-1 + phase-2 succeeds at bounded shear.
  The mode remains private and audit-only.  Items remaining for the
  full PR-T3B runtime patch: FLRW `N_eff = 3.044 ± 0.01` lock against
  Mangano 2005, explicit `dQ_α` sign-convention check (plasma hotter
  → ν receives energy), and Rodas5P step-rejection diagnostics near
  freeze-out.

---

## PR-T3C-PF  Diagonal ν-ν skeleton (preflight)

- **Status:** partial
- **Scope:** strictly additive preflight.  Adds
  `src/rabbit/jax/nu_nu_scattering_jax.py` and
  `tests/test_pr_t3c_nu_nu_preflight.py`.  No driver wiring, no
  capability-registry promotion, no replacement of the existing
  SciPy placeholder relaxation operator
  (`rabbit.collisions.nu_nu_scattering`, which is **not** the
  Dolgov-Hansen-Semikoz kernel).
- **Key files:**
  - `src/rabbit/jax/nu_nu_scattering_jax.py` — JIT-compatible
    diagonal ``ν_α + ν_β -> ν_α + ν_β`` Fierz-diagonal collision
    integral with the Hannestad-Madsen-style symmetric matrix
    element and a Fierz-aware ``epsilon_alpha_beta`` prefactor
    (= 2 for identical species, 1 for distinguishable).  PCHIP
    cubic Hermite is used for ``f_β`` evaluation; the partner-
    momentum integral is set up so the matched-grid configuration
    (``q_nodes == y2_nodes == y3_nodes``) collapses to identity at
    the input nodes for ``y_2`` and ``y_3``.
  - `tests/test_pr_t3c_nu_nu_preflight.py` — 8 tests: detailed
    balance at ``f = f_FD`` (locked at ``< 1e-20``, measured
    ``~7e-23``); energy conservation on a 50%-above-FD probe
    (locked at ``< 5% rel``, measured ``~1.8%``); Fierz factor
    scaling ``ε=2 vs ε=1`` (locked at ``< 1e-12 rel``); JIT
    determinism.
- **Physics added/changed:**  No new physics surface.  The kernel
  is a structural skeleton: the matrix-element prefactor uses the
  ν-e ``(y1 y2)^2 + (y3 y4)^2`` form rather than the
  Dolgov-Hansen-Semikoz appendix-A coefficient table, so the
  absolute ``N_eff`` shift is **not** calibrated.
- **Parity before / after:**  Before — no JAX-native ν-ν kernel
  existed (only the SciPy placeholder relaxation in
  `rabbit.collisions.nu_nu_scattering`).  After — a JIT-compatible
  skeleton with locked detailed balance and energy conservation
  invariants on the matched Laguerre grid.  Element-wise SciPy
  parity is **not** measured because no SciPy
  Dolgov-Hansen-Semikoz reference exists in the codebase yet.
- **Performance before / after:**  No production runtime change.
- **Known red tests:**  None.  Targeted regression bundle
  (`test_pr_t3a_collisionless_driver.py +
  test_pr_t3b_collision_preflight.py +
  test_pr_t3b_jax_operator_parity.py +
  test_pr_t3c_nu_nu_preflight.py`):
  `52 passed in 188.8 s` (44 prior + 8 new).
- **Docs updated:**
  - `docs/audit/PR-T3C_preflight.md`
  - `docs/ROADMAP_PR_CATALOG.md`
- **Self-audit verdict:**  Conditional pass.  The structural
  skeleton with documented algebraic invariants is in place and
  audit-only.  Open items for the full PR-T3C runtime patch:
  (1) implement a SciPy Dolgov-Hansen-Semikoz reference and lock
  element-wise JAX↔SciPy parity at 1e-12; (2) replace the
  placeholder matrix-element prefactor with the Dolgov-Hansen-
  Semikoz appendix-A coefficients; (3) tighten DB / energy
  conservation by removing the off-grid ``f_β(y_4)`` PCHIP
  interpolation; (4) wire through bank-core as a new
  ``nu_nu_preflight`` collision mode; (5) lock FLRW
  ``|N_eff - 3.044| < 0.005``.

---

## PR-T3D-PF  Tier-3 capability skeleton (preflight)

- **Status:** partial
- **Scope:** strictly additive registry skeleton.  Adds the
  `JAX_TYPEI_FULL_BOLTZMANN_TIER3_PREFLIGHT` capability and
  registers it in `CAPABILITY_BY_KEY` only.  No
  `CAPABILITY_BY_BACKEND` entry, no `canonical_forward_solver`
  dispatch branch, no public backend, no promotion to canonical.
  Follows the existing `JAX_EXTENDED_PSTF` candidate-without-
  dispatch pattern.
- **Key files:**
  - `src/rabbit/config/backend_capabilities.py` — adds the new
    capability constant and a `CAPABILITY_BY_KEY` entry.
  - `tests/test_inference_hierarchy_lock.py` — adds
    ``"jax_typeI_full_boltzmann_tier3_preflight"`` to
    `EXPECTED_CATALOG_KEYS`.
- **Physics added/changed:**  No new physics.  The capability
  metadata aggregates the existing PR-T3A (private collisionless
  shell), PR-T3B (`jax_kernel_preflight` runtime mode), and
  PR-T3C (diagonal ν-ν skeleton) preflight slices under a single
  introspection-level identity.  Notes string explicitly documents
  the open gates (DH-S calibration, FLRW `N_eff` lock, cross-code
  parity).
- **Parity before / after:**  No parity surface change; the
  capability is introspection-only.
- **Performance before / after:**  No production runtime change.
- **Known red tests:**  None.  Targeted regression bundle
  (`test_pr_t3a_collisionless_driver.py +
  test_pr_t3b_collision_preflight.py +
  test_pr_t3b_jax_operator_parity.py +
  test_pr_t3c_nu_nu_preflight.py +
  test_inference_hierarchy_lock.py`):
  `135 passed in 186.4 s`.  Catalog-size lock flips
  ``len(CAPABILITY_BY_KEY) == 20 -> 21`` cleanly;
  ``len(CAPABILITY_BY_BACKEND) == 10`` is unchanged.
- **Docs updated:**
  - `docs/audit/PR-T3D_preflight.md`
  - `docs/ROADMAP_PR_CATALOG.md`
- **Self-audit verdict:**  Conditional pass.  Surface-class /
  tier-alignment / catalog-completeness locks all green.  No
  silent promotion: no dispatch entry, candidate tier explicitly
  documented.  Items remaining for the full PR-T3D promotion:
  close the upstream PR-T3B FLRW `N_eff` lock and PR-T3C DH-S
  calibration, then build the cross-code fixture
  (LASAGNA / FortEPiaNO / PRIMAT-AC2024) and the
  ``|ΔY_p| < 5e-4`` parity tests, register a public dispatch key
  ``"jax_full_collision_tier3"``, and promote the capability
  ``tier`` from ``"candidate"`` to ``"canonical"`` with refreshed
  ``_canonical_v1`` contract strings.

---

## PR-R-PF  Release-gate preflight (pre-existing red snapshot)

- **Status:** partial
- **Scope:** doc-side release-gate preflight.  Closes the trivial
  pre-existing red-test items from the original PR-R phase prompt
  (fixture swap, formal `xfail` with rationale) and snapshots the
  current roadmap state.  Does **not** tag a release; the release
  tag is gated on the upstream PR-T3B/T3C/T3D canonical
  promotions.
- **Key files:**
  - `tests/test_production_gates.py` — fixture swap in
    `test_jax_flrw_gold` (now reads
    `gold["jax_flrw_equilibrium"]["Yp"]`); `pytest.mark.xfail`
    with `strict=False` on `test_classB_typeV_bbn_gold` linking to
    `docs/CLASSB_PROMOTION_PACKET.md`.
  - `docs/ROADMAP_STATE_OF_RECORD.md §6.1` — refreshed pre-existing
    red status table; three of the four originally-red tests are
    now formally green and one is formally `xfail` with linked
    rationale.
  - `docs/audit/PR-R_preflight.md` — preflight audit doc.
- **Physics added/changed:**  None.  Doc-side patch; one test fix
  is a fixture-key swap and the other is a deferral marker.
- **Parity before / after:**  No parity surface change.  The
  fixture swap on `test_jax_flrw_gold` exposes the
  already-correct equilibrium-FD gold (Yp = 0.2423504); no code
  change.
- **Performance before / after:**  No change.
- **Known red tests:**  None new.  Baseline four reds at the time
  of `PR-DOCS`: now reduced to one formally `xfail`-ed Class B
  drift (deferred to the Class A/B roadmap) and three formally
  green.
- **Docs updated:**
  - `docs/audit/PR-R_preflight.md`
  - `docs/ROADMAP_PR_CATALOG.md`
  - `docs/ROADMAP_STATE_OF_RECORD.md`
- **Self-audit verdict:**  Conditional pass.  Pre-existing reds
  resolved without any Type I code change; release tag explicitly
  deferred until the upstream tier-3 canonical promotions close.
  Targeted regression bundle (`test_production_gates.py +
  test_registry_sync.py + test_inference_hierarchy_lock.py`):
  `166 passed, 3 skipped, 1 xfailed in 70.8 s`.

---

## PR-T3B canonical milestone  AP-form unified mode passes anisotropy + grid gates

- **Status:** partial (FLRW Mangano gap accepted as documented limit;
  anisotropy + grid gates passed)
- **Scope:** lands the canonical AP-form unified bank-core in
  ``rabbit.jax.driver_typeI_full_boltzmann`` as a new
  ``collision_mode='ap_unified_preflight'`` runtime mode.
  Combines the three mechanisms from spectral_relaxation +
  projected_physical that previously had complementary
  strengths but were never simultaneously satisfied:
  1. Full Fermi-Dirac ``psi_target = (f_target - f_eq) / f_eq``
     (anisotropy-aware target shape, from spectral_relaxation).
  2. Decomposition into ``source_raw`` and ``damping_raw`` with
     the energy-neutral damping projection (from spectral).
  3. Soft total-rate enforcement on ``source_raw`` with clip
     ``[0.1, 5]`` and 1% floor (vs spectral's ``[0, 100]``
     aggressive clip that breaks at larger grids).
- **Canonical milestone metrics** (locked in
  ``tests/test_pr_t3b_ap_unified.py``):

  | ``(N_mu, N_q)`` | ``Σ_H`` | ``N_eff``  | gap to Mangano 3.044 |
  | --- | --- | --- | --- |
  | (4, 6)          | 0.00    | 3.034483   | +0.0095              |
  | (4, 6)          | 0.05    | 3.034530   | +0.0095              |
  | (4, 6)          | 0.10    | 3.034476   | +0.0095              |
  | (8, 12)         | 0.00    | 3.034568   | +0.0094              |
  | (12, 20)        | 0.00    | 3.034481   | +0.0095              |

  - FLRW gap ``+0.0095`` matches spectral_relaxation (the
    AP-form ceiling).
  - Grid spread ``<1e-4`` across 3 grids.
  - **Anisotropy spread ``~7e-5`` PASSES the canonical PR-T3D
    §5 gate of ``<1e-3`` by 2 orders of magnitude.**

- **Registry impact:**
  - New ``BackendCapability`` ``JAX_TYPEI_AP_UNIFIED_TIER3_CANDIDATE``
    in ``backend_capabilities`` (candidate tier; in
    ``CAPABILITY_BY_KEY`` only, not in
    ``CAPABILITY_BY_BACKEND``).
  - Catalog-size lock flips ``len(CAPABILITY_BY_KEY) == 21 -> 22``.
  - New ``rabbit.jax.collision_rates_jax``: Mangano /
    Hannestad-Madsen total rate helper for both ν-e and
    diagonal ν-ν (PR-T3C canonical #1).
- **Test surface added:**
  - ``tests/test_pr_t3b_ap_unified.py``: 5 tests locking FLRW
    baseline, grid convergence, anisotropy gate, metadata
    contracts, and spectral parity.
  - ``tests/test_pr_t3c_collision_rates_parity.py``: 4 new
    tests for ``total_rate_nu_nu_diagonal_jax`` (T^5 scaling,
    Fierz factor, dimensional consistency, JIT compatibility).
- **Docs updated:**
  - ``docs/audit/PR-T3B_jax_kernel_runtime.md`` — canonical
    milestone section.
  - ``docs/ROADMAP_PR_CATALOG.md`` (this entry).
  - ``STATUS.md`` and ``BACKEND_CAPABILITY_MATRIX.md`` re-rendered
    via ``render_capability_tables.py --apply``.
- **Remaining canonical work:**
  - Public dispatch wiring in ``canonical_forward_solver`` —
    deferred until either (a) the FLRW Mangano gap is closed
    via post-canonical Option E, or (b) the gap is formally
    accepted and the candidate is promoted to canonical with
    explicit Mangano-fidelity disclaimer in the dispatch
    docstring.
  - Dolgov-Hansen-Semikoz appendix-A coefficient table
    refinement on top of the leading-order ``(7π/12) G_F^2 T^5
    a_αβ`` form.
- **Self-audit verdict:**  Canonical milestone reached on 2 of
  3 PR-T3D §5 gates (anisotropy + grid scaling); the FLRW
  Mangano gap remains at the documented AP-form model-
  approximation limit per PR-T3B-PF #15 scope reframing.

---

## PR-T3B-PF #15  Scope reframing: AP-form canonical, no IMEX/jax_kernel pursuit

- **Status:** partial
- **Scope:** documentation + registry-only narrowing of the
  canonical destination after the cumulative
  PR-T3B-PF #1 - #14 calibration trail.  Confirms classical
  Boltzmann tier-3 (no QKE), keeps Rodas5P + JAX/GPU friendly
  architecture, drops the unreachable Mangano 5e-3 N_eff target.
- **Out of canonical scope** (deferred indefinitely):
  - ``jax_kernel_preflight`` (full Hannestad-Madsen kernel) —
    incompatible with the Rodas5P invariant due to the stiff
    ``∂C/∂T`` Jacobian manifold.  Closing requires either an
    IMEX/operator-split solver (violating the invariant) or a
    JAX-native AP-Rosenbrock (research-grade).
  - IMEX splitting on top of Rodas5P — Rosenbrock-Wanner
    methods do not natively support IMEX.
  - Mangano 5e-3 N_eff precision — out of reach without the
    above.
- **Retained canonical scope (Rodas5P + JAX/GPU compatible):**
  - AP-form unification (combine spectral_relaxation anisotropy
    stability with projected_physical grid scaling).
  - Dolgov-Hansen-Semikoz appendix-A coefficient table for
    diagonal ν-ν.
  - Documented ``~0.013`` FLRW gap to Mangano 2005 as accepted
    AP-form model fidelity limit.
- **Registry impact:** narrows
  ``feature_capabilities.TIER3_FULL_COLLISION_PREFLIGHT.blockers``
  from 5 entries (mode-attributed) to 3 (canonical-target-
  attributed).  Test expectations updated (4 affected tests in
  ``test_pr_t3c_feature_registration.py`` and
  ``test_pr_t3d_cross_registry_sync.py``).
- **Docs updated:**
  - ``docs/audit/PR-T3B_jax_kernel_runtime.md`` (PR-T3B-PF #15
    section with full rationale)
  - ``docs/audit/PR-R_preflight.md`` (3-blocker reformulation
    + scope-reframing reference)
  - ``docs/ROADMAP_PR_CATALOG.md`` (this entry)
  - ``src/rabbit/config/feature_capabilities.py``
    (``TIER3_FULL_COLLISION_PREFLIGHT`` notes, blockers, and
    evidence_summary all reframed)
  - ``tests/test_pr_t3c_feature_registration.py`` and
    ``tests/test_pr_t3d_cross_registry_sync.py`` (test
    expectations follow the narrowed scope)
- **Self-audit verdict:**  Pass.  The canonical PR-T3B/C/D work
  shrinks from "match Mangano 2005 to 5e-3" (research-grade,
  multi-quarter) to "AP-form unification + DH-S coefficient
  with documented Mangano gap" (engineering-grade, bounded).
  Both registries and the audit trail consistently reflect the
  narrowed scope.

---

## PR-T3C canonical #1  Dolgov-Hansen-Semikoz ν-ν total-rate helper

- **Status:** merged
- **Scope:** lands the leading-order Mangano /
  Hannestad-Madsen diagonal ν-ν total rate as a JAX building
  block for the AP-unified canonical track.  No live coupling
  into the runtime collision RHS yet; the rate is reachable via
  ``rabbit.jax.collision_rates_jax`` for downstream calibration
  and post-canonical Option E pre-conditioning.
- **Key files:**
  - ``src/rabbit/jax/collision_rates_jax.py`` — adds
    ``total_rate_nu_nu_diagonal_jax(T, alpha_eq_beta)`` in the
    ``(7π/12) G_F² T^5 a_αβ`` form and a
    ``gamma_nu_nu_over_H_jax`` helper.
- **Physics added/changed:** none in canonical RHS (additive
  building block only).  Same diagonal-coefficient convention as
  the existing pair-process / ν-e total rates; appendix-A
  ``O(1)`` running-coupling correction is an explicit research-
  track refinement and is *not* applied here.
- **Parity before / after:** none changed.  New helper matches
  SciPy reference to ``1e-14`` absolute on the diagonal-rate
  closed form.
- **Performance before / after:** no change (additive helper).
- **Known red tests:** none introduced.
- **Docs updated:**
  - ``docs/audit/PR-T3C_canonical_no1.md``
  - ``docs/ROADMAP_STATE_OF_RECORD.md`` (canonical milestone
    block)
- **Self-audit verdict:** pass; helper is regression-locked at
  the closed-form rate and waits on Option E for live wiring.

---

## PR-T3D canonical #1  Register AP-unified canonical-track candidate

- **Status:** merged
- **Scope:** registers
  ``JAX_TYPEI_AP_UNIFIED_TIER3_CANDIDATE`` in
  ``rabbit.config.backend_capabilities.CAPABILITY_BY_KEY`` for
  introspection and downstream tooling.  The capability backs
  the AP-unified ``collision_mode='ap_unified_preflight'`` slice
  of ``rabbit.jax.driver_typeI_full_boltzmann`` and is the
  first-class metadata anchor for the canonical milestone in
  PR-T3B.
- **Key files:**
  - ``src/rabbit/config/backend_capabilities.py`` — new
    ``JAX_TYPEI_AP_UNIFIED_TIER3_CANDIDATE`` capability + key
    catalog entry; collision-scope contract
    ``ap_unified_preflight_v1``.
- **Physics added/changed:** none.
- **Parity before / after:** unchanged.
- **Known red tests:** none introduced.
- **Docs updated:** registry-rendered docs picked up the new
  key (``docs/BACKEND_CAPABILITY_MATRIX.md``, etc.).
- **Self-audit verdict:** pass; introspection-only by design,
  not yet exposed via public dispatch (that lands in PR-T3D
  canonical #2).

---

## PR-T3D canonical #2  ``jax_ap_unified_tier3`` public dispatch + Mangano gap disclaimer

- **Status:** merged
- **Scope:** wires the AP-unified tier-3 candidate into
  ``canonical_forward_solver`` so callers can request it
  through ``backend="jax_ap_unified_tier3"``.  Surfaces the
  documented FLRW Mangano gap and the PR-T3D §5 canonical gate
  verdicts as first-class fields in the returned
  ``BBNPrediction.metadata``.
- **Key files:**
  - ``src/rabbit/inference/forward_likelihood.py`` — dispatch
    branch routes through ``run_full_boltzmann_jax`` with
    ``collision_mode='ap_unified_preflight'`` and
    ``thermo_tier=2``; rejects ``enable_teff=True``
    (``ValueError``); rejects ``Σ_- ≠ 0``
    (``NotImplementedError``: LRS-only); rejects
    ``enable_collisions=True``.
  - ``src/rabbit/config/backend_capabilities.py`` —
    ``CAPABILITY_BY_BACKEND["jax_ap_unified_tier3"]`` entry.
  - ``tests/test_pr_t3d_ap_unified_dispatch.py`` — 9 dispatch
    contract tests (success, FLRW ``N_eff`` baseline ±5e-3,
    metadata Mangano-gap exposure ``0.0095``, gate verdicts,
    contract metadata, anisotropic solve, LRS rejection, Teff
    rejection, unknown-backend message).
  - ``tests/test_inference_hierarchy_lock.py``,
    ``tests/test_advanced_envelope_lock.py``,
    ``tests/test_production_gates.py``,
    ``tests/test_registry_sync.py`` — backend enum + count
    locks updated 10 → 11.
  - ``scripts/render_capability_tables.py`` — dispatch_order
    list extended (3 sites) so registry-driven tables include
    the new backend.
- **Physics added/changed:** none in the RHS; all physics is
  carried by the existing ``ap_unified_preflight`` collision
  mode.  The PR adds the *contract surface* and disclaimer
  exposure.
- **Parity before / after:** dispatched FLRW
  ``N_eff = 3.0345`` matches the direct
  ``run_full_boltzmann_jax`` call to better than ``5e-3`` (the
  AP-form model approximation gap of ``+0.0095`` to Mangano
  3.044 is the documented model fidelity limit per
  PR-T3B-PF #15).  Anisotropic solves at ``Σ_H = 0.05`` keep
  ``|ΔN_eff| < 5e-3``.
- **Performance before / after:** unchanged.
- **Known red tests:** none introduced.  The canonical regression
  bundle (registry sync + dispatch + envelope lock + production
  gates) runs 186 / 3 (passed / skipped).
- **Docs updated:**
  - ``README.md``, ``SUPPORTED_CAPABILITIES.md``,
    ``STATUS.md``, ``PROMOTION_GATES.md``,
    ``docs/BACKEND_CAPABILITY_MATRIX.md`` — re-rendered by
    ``render_capability_tables.py`` with the new dispatch
    entry.
  - ``docs/ROADMAP_STATE_OF_RECORD.md`` — PR-T3B canonical
    milestone block updated to note the dispatch landing and
    the metadata-exposed Mangano gap.
  - ``docs/audit/PR-T3B_jax_kernel_runtime.md`` — same.
  - ``src/rabbit/config/feature_capabilities.py`` —
    ``TIER3_FULL_COLLISION_PREFLIGHT`` evidence / blockers /
    notes refreshed; only blocker remaining is the documented
    Mangano gap (deferred Option E).
- **Self-audit verdict:** pass.  PR-T3D canonical #2 closes the
  public-dispatch loop on the AP-unified canonical milestone:
  callers see one clean ``backend=`` selector and one metadata
  field that names the gap.  Canonical-tier promotion remains
  gated on Option E (post-canonical, ``~1-2 weeks``,
  Rodas5P/JAX-compatible).

---

## PR-R-PF #2  Cumulative tier-3 preflight status snapshot

- **Status:** partial
- **Scope:** doc-side cumulative snapshot of the cumulative
  tier-3 preflight surface across PR-T3A / PR-T3B / PR-T3C /
  PR-T3D.  Pure documentation update; no code or test changes.
- **Cumulative preflight slice inventory (28 commits across the
  PR-T3 chain since `PR-T3B preflight` baseline):**
  - PR-T3A: private collisionless full-phase-space shell
  - PR-T3B-PF: JAX nu-e + pair operator port; jax_kernel_preflight
    runtime mode; end-to-end smoke; FLRW N_eff baseline lock
    (2.993427); relaxation sign convention; q-grid remap attempts
    (×2, both reverted with documented stiffness regression);
    cache-leak fix (`_cached_*` numpy isolation); cross-mode
    comparison; AP-form grid convergence (3.030729 fully
    grid-converged at all (4,6) / (8,12) / (12,20) grids);
    anisotropic stability sweep (Σ_H spread ~0.54)
  - PR-T3C-PF: diagonal nu-nu skeleton with detailed-balance and
    energy-conservation locks; JAX-native not-a-knot cubic spline
    (1e-12 SciPy parity); PCHIP -> cubic swap on pair (8e-2 ->
    6e-16 rel parity, 14 OOM tightening); PCHIP -> cubic swap on
    nu-nu (DB ~5x, energy conservation ~8x); collisions_jax cache
    pattern alignment; Mangano/Hannestad-Madsen total-rate JAX
    helper (1e-14 SciPy parity); feature-level registration
    (TIER3_FULL_COLLISION_PREFLIGHT)
  - PR-T3D-PF: candidate capability registration; cross-code
    fixture skeleton (LASAGNA / FortEPiaNO / PRIMAT-AC2024 /
    Mangano 2005); per-cross-code parametrized diagnostics;
    fixture metadata extension with all 4 mode measurements +
    grid-resolved projected_physical entries + anisotropic sweep
- **Test surface added:** ``+86`` parametrized tests across the
  T3 preflight bundle (the September 2025 regression baseline
  was 36 tests; the cumulative bundle is now 122 tests).
- **Calibration measurements locked:**
  - AP-form FLRW ``N_eff = 3.030738``, gap to Mangano 3.044 is
    ``+0.0133`` (fully grid-converged across 3 grids).
  - jax_kernel FLRW ``N_eff = 2.993427`` (anti-heating bug;
    canonical needs IMEX/AP hybrid).
  - Anisotropic ``N_eff`` spread across ``Σ_H ∈ {0, 0.05, 0.10}``
    is ``~0.54`` (canonical target ``< 1e-3``).
  - Pair off-grid SciPy parity tightened from ``8.3% rel`` to
    ``6.5e-16 rel`` after the JAX cubic spline swap.
- **Open canonical blockers** (carried over to PR-T3B/C/D
  canonical):
  1. AP-form 0.013 N_eff gap to Mangano at FLRW
  2. JAX-kernel q-grid remap exposes stiff Jacobian manifold
  3. Anisotropic N_eff stability spread ~0.54
  4. Dolgov-Hansen-Semikoz ν-ν coefficient calibration deferred
- **Docs updated:**
  - `docs/ROADMAP_PR_CATALOG.md` (this entry)
  - All `docs/audit/PR-T3{B,C,D}_*.md` files carry per-PF audit
    sections with measurements, locks, and references back to
    the canonical phase-prompt targets.
- **Self-audit verdict:**  Cumulative preflight surface is
  measurable, regression-locked, and self-documenting.  Release
  tag remains gated on the four canonical blockers above.

---

## PR-A  Analytic J_j elimination in the JAX characteristic driver

- **Status:** merged
- **Scope:** JAX characteristic-driver state-layout refactor; removes
  the explicit `J_j` transport state and reconstructs the quadrature
  weight analytically from `(X0, S, μ_j)`.
- **Key files:**
  - `src/rabbit/jax/characteristic_rays_jax.py:23-69` —
    adds `jacobian_jax`, shrinks `characteristic_rhs_jax` to `(dI, dS)`.
  - `src/rabbit/jax/driver_typeI_char.py:124-156,203-320,595-612,656-761`
    — drops `i_J` layout slots, inlines analytic `J_vals`, updates the
    block-sparse partition, initial packing, and phase handoff.
  - `tests/test_pr_a_analytic_jacobian.py:1-57` — new regression lock
    against the numerically integrated `dJ/dN` ODE.
  - `tests/test_advanced_envelope_lock.py:209-225`,
    `tests/test_inference_hierarchy_lock.py:1-69` — stale 7-backend
    registry locks updated to the already-merged 9-backend registry so
    the fast regression gate reflects current dispatch reality.
- **Physics added/changed:**  No new physics.  The JAX driver now uses
  the exact closed-form forward Jacobian
  `J_j = dμ_j/dμ_{j,0}` implied by differentiating the analytic ray map;
  the audit locked the sign convention against the carried ODE
  `dJ_j/dN = 3 Σ₊ (1 - 3 μ_j²) J_j`.  The OCR text around paper eq (51)
  was found to be ambiguous, so the production convention is fixed by
  eq (55) plus direct numerical ODE parity.
- **Parity before / after:**  Unchanged at roadmap tolerance:
  tier-1 remains |ΔY_p| ≤ 4 × 10⁻⁸, tier-2 remains
  |ΔY_p| ≤ 7 × 10⁻⁸.  New analytic-J regression test passes at
  max |ΔJ| < 1 × 10⁻⁹ versus the numerically integrated ODE.
- **Performance before / after:**  State/Jacobian size falls
  37 → 25 at tier-1 phase-2.  On the local sandbox with
  `JAX_PLATFORMS=cpu` and `JAX_COMPILATION_CACHE_DIR=/tmp/rabbit_jax_cache`,
  warm single-solve timing is 1.66 s min / 1.70 s mean.  Stable-identity
  caching remains intact (`id(rhs_fn)` unchanged across repeated
  `_get_char_rhs(...)` calls).

- **Known red tests:**  The stable baseline reds remain tracked in the
  appendix.  The fast-sweep audit also surfaced two stale backend-count
  locks that still expected 7 registered backends even though the
  characteristic backends were already merged; those tests were updated
  as sync fixes in the same PR.
- **Docs updated:**
  - `ROADMAP_STATE_OF_RECORD.md` — characteristic state size, design
    notes, file inventory, regression counts.
  - `JAX_CHAR_GPU_OPTIMIZATION_PLAN.md` — Phase 1 flipped from planned
    to delivered, and the `J_j` formula corrected to the production
    forward-measure convention.
  - `docs/audit/PR-A*.md` — stage-by-stage audit trail.
- **Self-audit verdict:**  Pass.  See `docs/audit/PR-A.md` and
  `docs/audit/PR-A_stage{1,2,3}.md`.

### PR-N1  Non-LRS Bianchi Type I characteristic primitives

- **Status:** merged
- **Scope:** additive non-LRS characteristic infrastructure only; no
  driver wiring yet.
- **Key files:**
  - `src/rabbit/jax/characteristic_rays_nonlrs_jax.py` — new S²
    quadrature, generic-Type-I forward map, and `Π_+` / `Π_-` /
    monopole extractors.
  - `tests/test_pr_n1_nonlrs_primitives.py` — 9 regression tests.
- **Physics added/changed:** adds the pure angular primitives needed
  for generic orthogonal Type I.  The forward map uses the exact
  commuting diagonal stretch in the Wainwright-Hsu basis with
  integrated shear eigenvalues
  `diag(S_+ + √3 S_-, S_+ - √3 S_-, -2 S_+)`, which reduces to the
  existing LRS logistic map at `Σ_- = 0`.  The real `|m| = 2`
  stress kernel now follows the landed PR-N2 sign convention
  `-(√3/2) sin²θ cos 2φ`.
- **Parity before / after:** existing driver parity unchanged; new
  primitive locks pass at:
  - LRS `Π_+` reduction: absolute error `< 1e-12`
  - LRS monopole reduction: absolute error `< 1e-12`
  - constant-shear forward-map ODE cross-check: vector error `< 3e-11`
  - `Π_-` at `Σ_- = 0`: `< 1e-12`
- **Performance before / after:** not a driver PR; no production
  runtime change claimed.
- **Known red tests:** unchanged vs the stable baseline.
- **Docs updated:**
  - `ROADMAP_STATE_OF_RECORD.md`
  - `ROADMAP_PR_CATALOG.md`
  - `IMPLEMENTATION_GUIDE_NON_LRS_TYPEI.md`
  - `docs/audit/PR-N1*.md`
- **Self-audit verdict:** pass.  The audit corrected one prompt-level
  symmetry statement: the exact x↔y exchange carried by
  `Σ_- -> -Σ_-` is `φ -> π/2 - φ`, not `φ -> π - φ`.

### PR-N2  Non-LRS Bianchi Type I characteristic driver integration

- **Status:** merged
- **Scope:** wires the non-LRS S² primitives into the JAX
  characteristic driver as an explicit candidate backend; adds
  inference dispatch and registry/doc sync.
- **Key files:**
  - `src/rabbit/jax/driver_typeI_char.py` — adds
    `transport_mode="characteristic_nonlrs"`, compact non-LRS state
    layout, `_rhs_core_nonlrs`, and candidate metadata surface.
  - `src/rabbit/jax/characteristic_rays_nonlrs_jax.py` — adds
    analytic `intensity_shift_nonlrs_jax(...)` and
    `jacobian_nonlrs_jax(...)`.
  - `src/rabbit/jax/driver_typeI.py` — public JAX dispatch now accepts
    `transport_mode="characteristic_nonlrs"`.
  - `src/rabbit/config/backend_capabilities.py` — registers
    `jax_typeI_characteristic_nonlrs_tier1` /
    `backend="jax_characteristic_nonlrs"`.
  - `src/rabbit/inference/forward_likelihood.py` — explicit inference
    dispatch for the candidate non-LRS backend.
  - `tests/test_pr_n2_nonlrs_driver.py` — non-LRS driver integration
    tests.
- **Physics added/changed:** generic orthogonal Type I exact
  characteristic transport at tier-1 / collisionless scope.  The
  landed design keeps the state compact: `(μ, φ, I, J)` are
  reconstructed analytically from accumulated shear integrals
  `(S_+, S_-)` instead of being carried as explicit ray-state ODE
  blocks.
- **Parity before / after:** no canonical-surface parity changed.
  Non-LRS candidate locks at:
  - analytic S² Jacobian vs `jax.jacfwd`: `< 1e-11`
  - LRS reduction (`N_phi=1`, `Σ_-=0`): `|ΔY_p| ~ 1e-8`,
    `|ΔD/H| ~ 1e-10`
  - generic small-shear sign agreement with linearized PSTF; exact
    characteristic magnitude is about `4x` larger on the locked cell,
    so the surface remains candidate-only.
- **Performance before / after:** no change to canonical LRS
  characteristic runtime.  The non-LRS candidate path stays compact
  rather than inflating to an explicit `N_θ × N_φ` ray state.
- **Known red tests:** none introduced in the regression bundle.
- **Docs updated:**
  - `ROADMAP_STATE_OF_RECORD.md`
  - `IMPLEMENTATION_GUIDE_NON_LRS_TYPEI.md`
  - `docs/audit/PR-N2.md`
  - registry-generated capability docs via `render_capability_tables.py --apply`
- **Self-audit verdict:** pass, with explicit candidate-scope caveat.
  Tier-1 by default with explicit tier-2 opt-in, collisionless public
  dispatch only, and never selected by `backend="auto"`.

### PR-N3  Non-LRS explicit residual-state staging surface

- **Status:** stage-recorded (historical; not current capability)
- **Scope:** add the missing S² per-species residual state required to apply
  anisotropic residual relaxation on the non-LRS characteristic ray bundle,
  while keeping public `jax_characteristic_nonlrs` collision dispatch closed.
- **Key files:**
  - `src/rabbit/jax/driver_typeI_char.py` — adds
    `JAXNonLRSResidualStateConfig`,
    `JAXNonLRSResidualStateResult`, and
    `run_nonlrs_tier2_residual_state_jax(...)`.
  - `tests/test_pr_n2_nonlrs_driver.py` — locks the private residual-state
    smoke contract and the public fail-closed collision guard.
  - `docs/IMPLEMENTATION_GUIDE_NON_LRS_TYPEI.md` and
    `docs/ROADMAP_STATE_OF_RECORD.md` — split the compact public path from
    the private residual-state staging path.
- **Physics added/changed:** the compact public non-LRS state still uses the
  analytic `(S_+, S_-)` ray map.  The new private helper adds explicit
  residual variables `R_I[species, ray]` and `R_J[species, ray]` so the
  residual closure can evolve raywise `delta_I` and `delta_J` without
  pretending those corrections fit into the compact state.  The closure is
  mean-preserving in `I`, species-tagged (`ν_e`, `ν̄_e`, `ν_x`), and runs on
  CPU-JAX/Rodas5P with tier-2 3T thermodynamics and live weak monopoles.
- **Scope boundary:** no public dispatch promotion, no
  `enable_collisions=True` support for `jax_characteristic_nonlrs`, no
  production SMC validation, and QKE remains out of scope.
- **Verification:** `PYTHONPATH=src JAX_PLATFORMS=cpu venv/bin/python -m pytest -q tests/test_pr_n2_nonlrs_driver.py -k "residual_state_surface or public_dispatch_still_rejects_collision_promotion"` passed locally.
- **Self-audit verdict:** pass.  This lands the physical residual state
  needed by future non-LRS collision coupling but keeps the public candidate
  backend claim bounded to collisionless transport.

### PR-G  GPU vmap batched characteristic solve

- **Status:** merged
- **Scope:** additive batched characteristic infrastructure for the LRS
  JAX driver; scalar entry points and CPU-first policy remain intact.
- **Key files:**
  - `src/rabbit/jax/solver_jax_rodas5p.py` — adds
    `_solve_core_event_masked(...)` plus a cached compiled runner for
    per-element event-aware batched Rodas5P solves.
  - `src/rabbit/jax/driver_typeI_char.py` — adds
    `run_char_batch_tier1(...)` / `run_char_batch_tier2(...)`,
    batched runtime-device metadata, batch initial-condition/handoff
    helpers, and GPU usage notes.
  - `tests/test_pr_g_vmap_batch.py` — locks scalar-vs-batch parity,
    runtime-policy semantics, and frozen-finished-lane behaviour.
- **Physics added/changed:** none.  This PR is pure infrastructure; it
  batches independent characteristic solves without changing the carried
  equations or solver family.
- **Parity before / after:** canonical scalar parity unchanged.  New
  batch locks pass with:
  - tier-1 batch vs sequential scalar: `|ΔY_p| < 1e-10`
  - tier-2 batch vs sequential scalar: `|ΔY_p| < 1e-10`,
    `|ΔN_eff| < 1e-10`
  - event-masked solver freeze test: terminal-state drift `< 5e-6`
    after later lanes continue stepping
- **Performance before / after:** local tier-1 warm throughput on the
  RX 6950 XT audit rig:
  - CPU batch helper: `48.23 ms/solve` at `N=1`,
    `10.48 ms/solve` at `N=64`, `11.83 ms/solve` at `N=256`
  - GPU batch helper: `1156.18 ms/solve` at `N=1`,
    `19.12 ms/solve` at `N=64`, `9.91 ms/solve` at `N=128`,
    `7.44 ms/solve` at `N=256`
  The practical breakeven is therefore around `N≈128` on this compact
  state, later than the original roadmap estimate of `N=64`.
- **Known red tests:** none introduced in the targeted regression
  bundle.
- **Docs updated:**
  - `ROADMAP_STATE_OF_RECORD.md`
  - `JAX_CHAR_GPU_OPTIMIZATION_PLAN.md`
  - `docs/audit/PR-G*.md`
- **Self-audit verdict:** pass.  GPU batching is now a real,
  measured candidate path for medium/large inference grids, while the
  scalar driver remains CPU-first.

### AP4-NLRS  Non-LRS augmented quadrupole source reduction gate

- **Status:** merged into the active augmented-PSTF WBS branch
- **Scope:** additive SciPy-side reduction gate for the diagonal
  non-LRS augmented Type I staging basis; no public backend dispatch
  and no nonlinear non-LRS transport solver claim.
- **Key files:**
  - `src/rabbit/transport/augmented_typeI_nonlrs_collisionless.py`
    builds a tensor-product S2 grid and projects the source-only
    quadrupole RHS onto `{monopole, W_+, W_-}` coefficient modes.
  - `tests/test_augmented_typeI_nonlrs_collisionless.py` locks S2
    weight normalization, `Sigma_-=0` LRS reduction, `Sigma_-`
    population of the minus sector, and invalid-grid/input guards.
- **Physics added/changed:** source-only diagonal Type I quadrupole
  projection,
  `dA/dN = -2 q (Sigma_+ W_+ + Sigma_- W_-)`, on a deterministic S2
  angular grid.  Nonlinear angular advection, q-cascade terms,
  collisions, and BBN-network coevolution remain outside this gate.
- **Parity before / after:** no promoted runtime parity changed.  The
  new reduction test matches the LRS linearized quadrupole source when
  `Sigma_-=0`.
- **Known red tests:** none introduced in the focused augmented suite.
- **Docs updated:**
  - `ROADMAP_STATE_OF_RECORD.md`
  - `IMPLEMENTATION_GUIDE_TYPEI_AUGMENTED_PSTF_NOQKE_WBS.md`
- **Self-audit verdict:** pass, with the explicit source-only scope
  caveat above.

### AP5-NLRS  Non-LRS source angular-grid convergence runners

- **Status:** merged into the active augmented-PSTF WBS branch
- **Scope:** additive validation harness for the AP4-NLRS source-only
  projection gate; no full non-LRS transport solver or BBN-driver
  promotion.
- **Key files:**
  - `src/rabbit/validation/augmented_convergence.py` adds
    `run_augmented_nonlrs_source_nmu_convergence(...)` and
    `run_augmented_nonlrs_source_nphi_convergence(...)`.
  - `tests/test_augmented_convergence.py` locks the reported
    coefficient-source RMS values, expected-mode residuals, labels, and
    invalid-grid/q-grid rejection.
- **Physics added/changed:** none beyond AP4-NLRS.  This stage only
  records angular-grid convergence diagnostics for
  `dA/dN = -2 q (Sigma_+ W_+ + Sigma_- W_-)`.
- **Parity before / after:** no promoted runtime parity changed.  The
  source projection remains deterministic across the tested `N_mu` and
  `N_phi` ladders.
- **Known red tests:** none introduced in the focused augmented suite.
- **Docs updated:**
  - `ROADMAP_STATE_OF_RECORD.md`
  - `IMPLEMENTATION_GUIDE_TYPEI_AUGMENTED_PSTF_NOQKE_WBS.md`
- **Self-audit verdict:** pass, with explicit source-only scope.

### AP6-PROJ  Deterministic nodal collision-source projection bridge

- **Status:** merged into the active augmented-PSTF WBS branch
- **Scope:** additive bridge that projects a caller-supplied
  deterministic nodal `df/dN(q,n)` collision source into augmented
  coefficients.  It does not evaluate the angular collision kernel.
- **Key files:**
  - `src/rabbit/transport/augmented_collision_bridge.py` adds
    `project_augmented_nodal_collision_source(...)` and
    `AugmentedNodalCollisionProjectionResult`.
  - `tests/test_augmented_collision_bridge.py` locks LRS monopole-only
    projection, non-LRS `W_-` projection, source-shape validation, and
    closure metadata.
- **Physics added/changed:** no new collision physics.  This stage
  hardens the source-projection interface needed once deterministic
  S_N collision kernels produce angular `df/dN` values.
- **Parity before / after:** no promoted runtime parity changed.
- **Known red tests:** none introduced in the focused augmented suite.
- **Docs updated:**
  - `ROADMAP_STATE_OF_RECORD.md`
  - `IMPLEMENTATION_GUIDE_TYPEI_AUGMENTED_PSTF_NOQKE_WBS.md`
- **Self-audit verdict:** pass, with explicit projection-only scope.

### AP7-META  Augmented weak CL3 angular metadata wiring

- **Status:** merged into the active augmented-PSTF WBS branch
- **Scope:** additive metadata surface for CL3 angular terms in the
  augmented weak bridge; weak-rate evaluation still consumes live
  monopoles only on this original AP7 staged path.  AP58-AP60 later
  extend the same bridge with explicit moment-input angular weak-rate
  application modes without changing public dispatch.
- **Key files:**
  - `src/rabbit/weak/augmented_bridge.py` adds
    `WeakAngularCorrectionMetadata` and
    `build_cl3_angular_metadata(...)`.
  - `tests/test_augmented_weak_bridge.py` locks Born-monopole
    invariance, CL3 metadata availability, explicit missing-input
    metadata, and CL2 inactive metadata.
- **Physics added/changed:** no rate correction is applied.  The
  metadata records distribution quadrupole proxies for audit/future
  wiring when q-energy weights and angular kernels are supplied.
- **Parity before / after:** no promoted runtime parity changed.
- **Known red tests:** none introduced in the focused augmented suite.
- **Docs updated:**
  - `ROADMAP_STATE_OF_RECORD.md`
  - `IMPLEMENTATION_GUIDE_TYPEI_AUGMENTED_PSTF_NOQKE_WBS.md`
- **Self-audit verdict:** pass, with explicit metadata-only scope.
  Superseded for angular weak-rate application by AP58/AP59/AP60; LRS keeps the
  bounded legacy default, while staged non-LRS correction-level-3 configs now
  default to the AP60 current S2 moment-input application with explicit
  `metadata_only` controls.

### AP8-NLRS  Non-LRS source stability envelope

- **Status:** merged into the active augmented-PSTF WBS branch
- **Scope:** source-only non-LRS stability envelope for the AP4-NLRS
  quadrupole projection gate; no non-LRS coevolution solver or BBN
  candidate promotion.
- **Key files:**
  - `src/rabbit/validation/augmented_stability.py` adds
    `AugmentedNonLRSSourceStabilityEnvelopeSpec` and
    `run_augmented_nonlrs_source_stability_envelope(...)`.
  - `tests/test_augmented_stability_envelope.py` locks accepted
    `Sigma_-` cases, limit-failure reporting, and invalid spec guards.
- **Physics added/changed:** no new dynamics.  The envelope checks
  deterministic source projection
  `dA/dN = -2 q (Sigma_+ W_+ + Sigma_- W_-)`.
- **Parity before / after:** no promoted runtime parity changed.
- **Known red tests:** none introduced in the focused augmented suite.
- **Docs updated:**
  - `ROADMAP_STATE_OF_RECORD.md`
  - `IMPLEMENTATION_GUIDE_TYPEI_AUGMENTED_PSTF_NOQKE_WBS.md`
- **Self-audit verdict:** pass, with explicit source-only scope.

### AP9-QMC-BRIDGE  QMC augmented nodal projection bridge

- **Status:** merged into the active augmented-PSTF WBS branch
- **Scope:** replay-stable Sobol/QMC validation report for sampled
  augmented nodal `df/dN` source projection through the existing AP6 bridge.
  This does not evaluate the physical collision kernel.
- **Key files:**
  - `src/rabbit/validation/qmc_control_variate.py` adds
    `QMCAugmentedNodalProjectionReport` and
    `build_qmc_augmented_nodal_projection_report(...)`.
  - `tests/test_qmc_control_variate.py` locks LRS sampled projection
    convergence, replay identity, and geometry/dimension validation.
- **Physics added/changed:** no new collision physics.  The report
  consumes a deterministic nodal `df/dN` source on each Sobol angular
  grid and projects it with
  `project_augmented_nodal_collision_source(...)`.
- **Parity before / after:** no promoted runtime parity changed.
- **Known red tests:** none introduced in the focused QMC suite.
- **Docs updated:**
  - `ROADMAP_STATE_OF_RECORD.md`
  - `IMPLEMENTATION_GUIDE_TYPEI_AUGMENTED_PSTF_NOQKE_WBS.md`
- **Self-audit verdict:** pass, with explicit validation-only scope.

### AP10-JAX-COLLISION-BRIDGE  JAX augmented nodal projection parity

- **Status:** merged into the active augmented-PSTF WBS branch
- **Scope:** CPU-JAX array-level parity helper for the AP6 generic nodal
  `df/dN` source-to-augmented projection bridge.  This is a fixed-grid
  parity surface, not public backend dispatch and not a collision-kernel
  evaluator.
- **Key files:**
  - `src/rabbit/jax/augmented_collision_bridge_jax.py` adds
    `project_augmented_nodal_collision_source_jax(...)`.
  - `tests/test_jax_augmented_collision_bridge.py` locks LRS closure-JIT parity,
    non-LRS `W_-` parity, saturated-bin Pauli-floor parity, and static
    fixed-grid/eager input guards.
- **Physics added/changed:** no new collision physics.  The JAX bridge
  consumes deterministic nodal `df/dN` sources and mirrors the SciPy
  augmented logit projection.
- **Parity before / after:** JAX AP10 parity now includes the generic
  AP6 nodal source projection arrays in addition to distribution
  reconstruction/projection primitives.
- **Known red tests:** none introduced in the focused JAX augmented suite.
- **Docs updated:**
  - `ROADMAP_STATE_OF_RECORD.md`
  - `IMPLEMENTATION_GUIDE_TYPEI_AUGMENTED_PSTF_NOQKE_WBS.md`
- **Self-audit verdict:** pass, with explicit parity-only scope.

### AP11-JAX-PROJECTED-SOURCE-SOLVE  JAX Rodas5P projected-source staging

- **Status:** merged into the active augmented-PSTF WBS branch
- **Scope:** CPU-JAX fixed-grid LRS RHS and Rodas5P wrapper that inject
  a static deterministic modal source through the AP10 nodal projection
  bridge at each RHS evaluation.  This is a solver-staging gate, not a
  physical collision-kernel coupling or public backend.
- **Key files:**
  - `src/rabbit/jax/augmented_typeI_collisionless_jax.py` adds
    `make_augmented_lrs_collisionless_projected_source_rhs_jax(...)`
    and
    `solve_augmented_lrs_collisionless_projected_source_rodas5p_jax(...)`.
  - `tests/test_jax_augmented_typeI_collisionless.py` locks additive
    RHS parity, a bounded Rodas5P projected-source solve, and static
    source shape/tracer validation.
- **Physics added/changed:** no new collision physics.  The source is a
  deterministic modal `dA/dN` staging input converted to nodal `df/dN`
  from the live distribution and reprojected inside the RHS.
- **Parity before / after:** AP11 now verifies the existing JAX Rodas5P
  core can carry an augmented projected-source term through the solve
  loop without introducing a new solver implementation.
- **Known red tests:** none introduced in the focused JAX augmented suite.
- **Docs updated:**
  - `ROADMAP_STATE_OF_RECORD.md`
  - `IMPLEMENTATION_GUIDE_TYPEI_AUGMENTED_PSTF_NOQKE_WBS.md`
- **Self-audit verdict:** pass, with explicit solver-staging scope.

### AP12-REGISTRY  Augmented-PSTF staging capability registry

- **Status:** merged into the active augmented-PSTF WBS branch
- **Scope:** registry and generated-doc visibility for the AP0-AP11
  augmented-PSTF no-QKE staging surface.  This is catalog-only and does
  not add public forward-solver dispatch.
- **Key files:**
  - `src/rabbit/config/backend_capabilities.py` adds
    `JAX_TYPEI_AUGMENTED_PSTF_NOQKE_STAGING` under `CAPABILITY_BY_KEY`
    only.
  - `src/rabbit/config/feature_capabilities.py` adds
    `TYPEI_AUGMENTED_PSTF_NOQKE_STAGING`.
  - `tests/test_augmented_pstf_capability_registry.py` locks the
    diagnostic-substrate scope, scope-contract strings, and absence from
    `CAPABILITY_BY_BACKEND`.
- **Physics added/changed:** none.  This records the landed AP0-AP11
  reconstruction/projection, convergence, QMC replay, CL3 metadata, and
  Rodas5P projected-source staging pieces without promoting collision
  physics, weak-rate angular corrections, full BBN coevolution, non-LRS
  nonlinear transport, or GPU/XLA.
- **Parity before / after:** no promoted runtime parity changed.
- **Known red tests:** none introduced in the focused registry suite.
- **Docs updated:**
  - `README.md`
  - `STATUS.md`
  - `SUPPORTED_CAPABILITIES.md`
  - `docs/BACKEND_CAPABILITY_MATRIX.md`
  - `docs/ROADMAP_STATE_OF_RECORD.md`
  - `IMPLEMENTATION_GUIDE_TYPEI_AUGMENTED_PSTF_NOQKE_WBS.md`
- **Self-audit verdict:** pass, with explicit catalog-only scope.

### AP13-WEAK-NETWORK-RHS  SciPy augmented live weak/network RHS bridge

- **Status:** merged into the active augmented-PSTF WBS branch
- **Scope:** SciPy-side RHS bridge that evaluates live weak rates and
  PRIMAT network derivatives from the current LRS augmented distribution.
  This is not a public BBN driver and not a full thermo/background
  coevolution solve.
- **Key files:**
  - `src/rabbit/transport/augmented_typeI_weak_network.py` adds
    `augmented_lrs_weak_network_rhs(...)` and
    `augmented_lrs_collisionless_weak_network_rhs(...)`.
  - `tests/test_augmented_typeI_weak_network_bridge.py` locks FD
    monopole extraction, current-monopole rate sensitivity, network RHS
    parity, CL3 metadata threading, and combined transport+network RHS
    output.
- **Physics added/changed:** weak rates remain algebraic functionals,
  not ODE state variables.  Each RHS call reconstructs `f_s(q,n)`,
  extracts live `nu_e`/`anti-nu_e` monopoles, computes
  `lambda_np`/`lambda_pn` with the existing live weak-rate functional,
  and injects them into `abundance_rhs_phase2(...)`.
- **Parity before / after:** no promoted runtime parity changed.  AP13
  verifies the network derivative equals the PRIMAT standard RHS when fed
  the bridge-computed weak rates.
- **Known red tests:** none introduced in the focused AP13 suite.
- **Docs updated:**
  - `README.md`
  - `STATUS.md`
  - `SUPPORTED_CAPABILITIES.md`
  - `docs/ROADMAP_STATE_OF_RECORD.md`
  - `IMPLEMENTATION_GUIDE_TYPEI_AUGMENTED_PSTF_NOQKE_WBS.md`
- **Self-audit verdict:** pass, with explicit RHS-bridge-only scope.

### AP14-WEAK-NETWORK-SOLVE  SciPy augmented weak/network solve shell

- **Status:** merged into the active augmented-PSTF WBS branch
- **Scope:** SciPy-side LRS staging solve that packs `Sigma_+`,
  augmented LRS modes, and PRIMAT abundances into one `solve_ivp`
  `d/dN` state using externally supplied fixed `T_gamma`, `T_nu`, and
  Hubble rate.  This is not public dispatch and not full
  thermo/background BBN coevolution.
- **Key files:**
  - `src/rabbit/transport/augmented_typeI_weak_network.py` adds
    `run_augmented_lrs_collisionless_weak_network_solve(...)` and the
    corresponding solve-result dataclass.
  - `tests/test_augmented_typeI_weak_network_solve.py` locks the FLRW
    transport fixed point, shear-to-mode coupling, finite abundance
    evolution, custom q-grid threading, and validation failures.
- **Physics added/changed:** no new collision or weak correction
  physics.  The AP13 RHS blocks are now exercised inside one SciPy
  solver loop with explicit `dX_dt` to `dX_dN` Hubble scaling.
- **Solver note:** the helper keeps a `method=` escape hatch for
  `LSODA` stiffness experiments.  The smoke default remains aligned
  with the existing SciPy staging shell; JAX promotion must reuse the
  in-tree Rodas5P/Rosenbrock solver core.
- **Parity before / after:** no promoted runtime parity changed.
- **Known red tests:** none introduced in the focused AP14 suite.
- **Docs updated:**
  - `README.md`
  - `STATUS.md`
  - `SUPPORTED_CAPABILITIES.md`
  - `docs/ROADMAP_STATE_OF_RECORD.md`
  - `IMPLEMENTATION_GUIDE_TYPEI_AUGMENTED_PSTF_NOQKE_WBS.md`
- **Self-audit verdict:** pass, with explicit fixed-thermo/H
  solve-shell-only scope.

### AP15-WEAK-NETWORK-3T-SOLVE  SciPy augmented 3T thermo/Hubble shell

- **Status:** merged into the active augmented-PSTF WBS branch
- **Scope:** SciPy-side LRS staging solve that extends AP14 by packing
  `T_gamma`, `T_nu_e`, and `T_nu_x` into the same `solve_ivp` state.
  The RHS recomputes `H(T_gamma,T_nu_e,T_nu_x,Sigma^2)` every call and
  uses that dynamic Hubble rate for the weak/network `dX_dt -> dX_dN`
  conversion.  This is not public dispatch and not a promoted
  collision-coupled BBN driver.
- **Key files:**
  - `src/rabbit/transport/augmented_typeI_weak_network.py` adds
    `run_augmented_lrs_collisionless_weak_network_3T_solve(...)` and the
    corresponding solve-result dataclass.
  - `tests/test_augmented_typeI_weak_network_3t_solve.py` locks dynamic
    temperature evolution, dynamic Hubble history, shear-to-mode
    coupling, LSODA override support, and validation failures.
- **Physics added/changed:** no new angular collision kernel or weak
  correction physics.  AP15 wires the existing 3T thermo/Hubble helpers
  into the staged augmented-PSTF solver loop.
- **Solver note:** `method="LSODA"` remains available for stiffness
  experiments.  JAX promotion must still reuse the in-tree
  Rodas5P/Rosenbrock solver core.
- **Parity before / after:** no promoted runtime parity changed.
- **Known red tests:** none introduced in the focused AP15 suite.
- **Docs updated:**
  - `README.md`
  - `STATUS.md`
  - `SUPPORTED_CAPABILITIES.md`
  - `docs/ROADMAP_STATE_OF_RECORD.md`
  - `IMPLEMENTATION_GUIDE_TYPEI_AUGMENTED_PSTF_NOQKE_WBS.md`
- **Self-audit verdict:** pass, with explicit 3T-thermo/Hubble
  solve-shell-only scope.

### AP16-WEAK-NETWORK-3T-CONVERGENCE  SciPy 3T shell convergence runners

- **Status:** merged into the active augmented-PSTF WBS branch
- **Scope:** SciPy-side convergence runners for the AP15 LRS 3T
  thermo/Hubble staging shell over `ell_max`, `N_q`, and `N_mu`.
  This is not public dispatch and not a promotion gate by itself.
- **Key files:**
  - `src/rabbit/validation/augmented_convergence.py` adds
    `run_augmented_lrs_3t_ell_convergence(...)`,
    `run_augmented_lrs_3t_q_convergence(...)`, and
    `run_augmented_lrs_3t_angular_convergence(...)`.
  - `tests/test_augmented_convergence.py` locks thermo/network
    observables, dynamic-H summaries, `ell_max` tail norms, and
    q/angular resolution labels.
- **Physics added/changed:** none.  AP16 reports convergence of the
  AP15 staging shell using existing report contracts.
- **Parity before / after:** no promoted runtime parity changed.
- **Known red tests:** none introduced in the focused AP16 suite.
- **Docs updated:**
  - `README.md`
  - `STATUS.md`
  - `SUPPORTED_CAPABILITIES.md`
  - `docs/ROADMAP_STATE_OF_RECORD.md`
  - `IMPLEMENTATION_GUIDE_TYPEI_AUGMENTED_PSTF_NOQKE_WBS.md`
- **Self-audit verdict:** pass, with explicit convergence-runner-only
  scope.

### AP17-WEAK-NETWORK-3T-CONVERGENCE-ARTIFACT  Deterministic SciPy 3T convergence report

- **Status:** merged into the active augmented-PSTF WBS branch
- **Scope:** deterministic JSON artifact runner around the AP16 SciPy
  3T `ell_max`, `N_q`, and `N_mu` convergence ladders.  The default is
  smoke-scale; the optional extended preset includes `ell_max = 2,4,6,8`
  when runtime permits.  This is not public dispatch and not a
  promotion gate by itself.
- **Key files:**
  - `src/rabbit/validation/augmented_convergence.py` adds
    `build_augmented_lrs_3t_convergence_artifact(...)` and
    `write_augmented_lrs_3t_convergence_artifact(...)`.
  - `scripts/run_augmented_3t_convergence_artifact.py` emits the
    deterministic JSON artifact from the same validation code path.
  - `tests/test_augmented_convergence.py` locks the serialized artifact
    contract, ladder labels, solver metadata, observables, and explicit
    limitations.
- **Physics added/changed:** none.  AP17 records the existing AP16
  staged 3T convergence evidence in a deterministic artifact format.
- **Parity before / after:** no promoted runtime parity changed.
- **Known red tests:** none introduced in the focused AP17 suite.
- **Docs updated:**
  - `README.md`
  - `STATUS.md`
  - `SUPPORTED_CAPABILITIES.md`
  - `docs/ROADMAP_STATE_OF_RECORD.md`
  - `IMPLEMENTATION_GUIDE_TYPEI_AUGMENTED_PSTF_NOQKE_WBS.md`
- **Self-audit verdict:** pass, with explicit diagnostic-artifact-only
  scope.

### AP18-WEAK-NETWORK-3T-COLLISION-THERMO-HOOK  Opt-in collision-moment thermo feedback

- **Status:** merged into the active augmented-PSTF WBS branch
- **Scope:** optional collision-moment thermodynamic feedback hook for
  the SciPy 3T weak/network staging shell.  If a caller supplies an
  explicit `collision_thermo_source` callback returning
  `dQ_nue_pair_N` and `dQ_nux_bank_N`, the shell evaluates the 3T
  temperature RHS from those moments.  With no callback, the AP15 table
  RHS remains the default.  This is not a full angular collision kernel
  and not public dispatch.
- **Key files:**
  - `src/rabbit/transport/augmented_typeI_weak_network.py` adds
    `Augmented3TCollisionThermoSource` and the opt-in
    `collision_thermo_source=` path for
    `run_augmented_lrs_collisionless_weak_network_3T_solve(...)`.
  - `tests/test_augmented_typeI_weak_network_3t_solve.py` locks the
    source-driven temperature split, default table-RHS metadata, and
    callback validation.
  - `src/rabbit/validation/augmented_convergence.py` updates AP17
    artifact limitations to distinguish the smoke artifact default from
    the AP18 opt-in feedback hook.
- **Physics added/changed:** collision-moment energy feedback can now
  drive the staged 3T thermo RHS when supplied explicitly.  No physical
  angular collision kernel is introduced.
- **Parity before / after:** default 3T table-RHS behavior remains the
  same when no callback is supplied.
- **Known red tests:** none introduced in the focused AP18 suite.
- **Docs updated:**
  - `README.md`
  - `STATUS.md`
  - `SUPPORTED_CAPABILITIES.md`
  - `docs/ROADMAP_STATE_OF_RECORD.md`
  - `IMPLEMENTATION_GUIDE_TYPEI_AUGMENTED_PSTF_NOQKE_WBS.md`
- **Self-audit verdict:** pass, with explicit opt-in hook-only scope.

### AP19-WEAK-NETWORK-3T-NUNU-THERMO-SOURCE  Diagonal no-QKE nu-nu source factory

- **Status:** merged into the active augmented-PSTF WBS branch
- **Scope:** deterministic diagonal no-QKE `nu-nu` thermo source
  factory for the AP18 collision-moment feedback hook.  The source
  reconstructs current augmented LRS distributions, takes angular
  monopoles, evaluates the AP81 fixed-quadrature pairwise diagonal `nu-nu`
  2-to-2 reference with the six-monomial Pauli factor, applies explicit
  per-bank number closure and effective-`nu_x` weighted-energy closure
  projection, and returns
  `dQ_nue_pair_N`/`dQ_nux_bank_N` moments for the 3T shell.  This remains a
  monopole diagonal source and not a full angular collision kernel.
- **Key files:**
  - `src/rabbit/transport/augmented_collision_bridge.py` adds
    `build_augmented_lrs_nunu_collision_thermo_source(...)`.
  - `tests/test_augmented_collision_bridge.py` locks source sign,
    pairwise source metadata, six-monomial diagnostics, number residual
    closure, weighted energy residual diagnostics, and finite source metadata.
  - `tests/test_augmented_typeI_weak_network_3t_solve.py` verifies the
    source factory can drive the AP18 3T solve callback path.
- **Physics added/changed:** bounded pairwise diagonal no-QKE `nu-nu`
  monopole-moment feedback can now source the staged 3T thermo RHS using the
  same six-monomial scalar occupation-number algebra as AP81.  The older
  fixed-point redistribution helper remains available only as legacy
  comparison plumbing.  The full angular collision kernel remains
  unimplemented on this augmented path.
- **Parity before / after:** default 3T table-RHS behavior remains the
  same unless the AP19 source is explicitly supplied.
- **Known red tests:** none introduced in the focused AP19 suite.
- **Docs updated:**
  - `README.md`
  - `STATUS.md`
  - `SUPPORTED_CAPABILITIES.md`
  - `docs/ROADMAP_STATE_OF_RECORD.md`
  - `IMPLEMENTATION_GUIDE_TYPEI_AUGMENTED_PSTF_NOQKE_WBS.md`
- **Self-audit verdict:** pass, with explicit diagonal-monopole-source
  scope.

### AP20-WEAK-NETWORK-3T-EM-THERMO-SOURCE  Monopole nu-e plus pair source factory

- **Status:** merged into the active augmented-PSTF WBS branch
- **Scope:** deterministic electromagnetic-bath thermo source factory
  for the AP18 collision-moment feedback hook.  The source reconstructs
  current augmented LRS distributions, takes angular monopoles, evaluates
  the existing fixed-quadrature `nu-e` scattering and pair-process
  references, and returns `dQ_nue_pair_N`/`dQ_nux_bank_N` moments for
  the 3T shell.  This remains angle-independent and not a full angular
  collision kernel.
- **Key files:**
  - `src/rabbit/transport/augmented_collision_bridge.py` adds
    `build_augmented_lrs_electron_pair_collision_thermo_source(...)`.
  - `tests/test_augmented_collision_bridge.py` locks FD quietness,
    heating sign for under-populated monopoles, total energy-gain
    diagnostics, and detailed-balance residual metadata.
- **Physics added/changed:** bounded monopole `nu-e` scattering and
  pair-process energy feedback can now source the staged 3T thermo RHS
  when supplied explicitly.
- **Parity before / after:** default 3T table-RHS behavior remains the
  same unless the AP20 source is explicitly supplied.
- **Known red tests:** none introduced in the focused AP20 suite.
- **Docs updated:**
  - `README.md`
  - `STATUS.md`
  - `SUPPORTED_CAPABILITIES.md`
  - `docs/ROADMAP_STATE_OF_RECORD.md`
  - `IMPLEMENTATION_GUIDE_TYPEI_AUGMENTED_PSTF_NOQKE_WBS.md`
- **Self-audit verdict:** pass, with explicit monopole-source-only
  scope.

### AP21-WEAK-NETWORK-3T-COMBINED-THERMO-SOURCE  Combined collision source factory

- **Status:** merged into the active augmented-PSTF WBS branch
- **Scope:** explicit source-composition factory for the AP18
  collision-moment feedback hook.  The factory builds the AP19
  diagonal no-QKE `nu-nu` source and the AP20 electromagnetic-bath
  source, evaluates both on the same current augmented LRS state, and
  returns the summed `dQ_nue_pair_N`/`dQ_nux_bank_N` moments with
  component diagnostics.  This remains opt-in and is not a full angular
  collision kernel.
- **Key files:**
  - `src/rabbit/transport/augmented_collision_bridge.py` adds
    `build_augmented_lrs_combined_collision_thermo_source(...)`.
  - `tests/test_augmented_collision_bridge.py` locks exact component
    moment summation and diagnostic propagation.
  - `tests/test_augmented_typeI_weak_network_3t_solve.py` verifies the
    combined source factory can drive the AP18 3T solve callback path
    with an extra unused species bank.
- **Physics added/changed:** bounded deterministic monopole collision
  moment sources can now be composed for the staged 3T thermo RHS when
  supplied explicitly.
- **Parity before / after:** default 3T table-RHS behavior remains the
  same unless the AP21 source is explicitly supplied.
- **Known red tests:** none introduced in the focused AP21 suite.
- **Docs updated:**
  - `README.md`
  - `STATUS.md`
  - `SUPPORTED_CAPABILITIES.md`
  - `docs/ROADMAP_STATE_OF_RECORD.md`
  - `IMPLEMENTATION_GUIDE_TYPEI_AUGMENTED_PSTF_NOQKE_WBS.md`
- **Self-audit verdict:** pass, with explicit source-composition-only
  scope.

### AP22-WEAK-NETWORK-3T-COLLISION-FEEDBACK-ARTIFACT  Source-variant report

- **Status:** merged into the active augmented-PSTF WBS branch
- **Scope:** deterministic smoke-scale JSON artifact runner comparing
  the standard 3T table RHS with the AP19 diagonal `nu-nu`, AP20
  electromagnetic-bath, and AP21 combined collision-moment source
  variants.  This is a diagnostic report surface only; it does not
  promote the collision-feedback path to a default physical solve.
- **Key files:**
  - `src/rabbit/validation/augmented_convergence.py` adds
    `build_augmented_lrs_3t_collision_feedback_artifact(...)` and
    `write_augmented_lrs_3t_collision_feedback_artifact(...)`.
  - `scripts/run_augmented_3t_collision_feedback_artifact.py` emits the
    AP22 JSON report.
  - `tests/test_augmented_convergence.py` locks the artifact contract,
    source-variant contracts, source diagnostics, limitations, and JSON
    serializability.
- **Physics added/changed:** no new physics kernel; the existing
  monopole source callbacks are exercised in one deterministic report.
- **Parity before / after:** default 3T table-RHS behavior remains the
  same unless an opt-in source callback is supplied.
- **Known red tests:** none introduced in the focused AP22 suite.
- **Docs updated:**
  - `README.md`
  - `STATUS.md`
  - `SUPPORTED_CAPABILITIES.md`
  - `docs/ROADMAP_STATE_OF_RECORD.md`
  - `IMPLEMENTATION_GUIDE_TYPEI_AUGMENTED_PSTF_NOQKE_WBS.md`
- **Self-audit verdict:** pass, with explicit diagnostic-artifact-only
  scope.

### AP23-WEAK-NETWORK-3T-COLLISION-FEEDBACK-DELTAS  Standard-relative artifact deltas

- **Status:** merged into the active augmented-PSTF WBS branch
- **Scope:** report-metadata enhancement for the AP22 collision-feedback
  source-variant artifact.  When the standard 3T table RHS baseline is
  included, the artifact now records per-observable deltas for each
  opt-in source variant relative to that baseline.
- **Key files:**
  - `src/rabbit/validation/augmented_convergence.py` adds
    standard-relative observable delta construction.
  - `tests/test_augmented_convergence.py` locks the `delta_reference`
    field and exact delta arithmetic for the combined source variant.
- **Physics added/changed:** none; report metadata only.
- **Parity before / after:** solve behavior and default 3T table-RHS
  behavior are unchanged.
- **Known red tests:** none introduced in the focused AP23 suite.
- **Docs updated:**
  - `README.md`
  - `STATUS.md`
  - `SUPPORTED_CAPABILITIES.md`
  - `docs/ROADMAP_STATE_OF_RECORD.md`
  - `IMPLEMENTATION_GUIDE_TYPEI_AUGMENTED_PSTF_NOQKE_WBS.md`
- **Self-audit verdict:** pass, with explicit report-metadata-only
  scope.

### AP24-WBS-STAGE-SCOPED-LANDING-AUDIT  Stage-scoped WBS status normalization

- **Status:** merged into the active augmented-PSTF WBS branch
- **Scope:** claim-boundary/status cleanup for the Type-I augmented-PSTF
  no-QKE WBS ledger.  AP10-AP23 are now marked as `landed in current
  workspace (stage-scoped)` because their named deliverables and exit
  gates are implemented and tested.  AP4-AP9 remain `partial` because
  their row text still names unresolved programme blockers.
- **Key files:**
  - `docs/IMPLEMENTATION_GUIDE_TYPEI_AUGMENTED_PSTF_NOQKE_WBS.md`
    updates the WBS status column and defines the stage-scoped meaning.
  - `tests/test_augmented_wbs_status_ledger.py` parses the WBS table
    and locks AP4-AP9 as partial and AP10-AP24 as stage-scoped landed.
- **Physics added/changed:** none; status-ledger cleanup only.
- **Parity before / after:** no runtime behavior changed.
- **Known red tests:** none introduced in the focused AP24 suite.
- **Docs updated:**
  - `README.md`
  - `STATUS.md`
  - `SUPPORTED_CAPABILITIES.md`
  - `docs/ROADMAP_STATE_OF_RECORD.md`
  - `IMPLEMENTATION_GUIDE_TYPEI_AUGMENTED_PSTF_NOQKE_WBS.md`
- **Self-audit verdict:** pass, with explicit no-promotion status
  scope.

### AP25-AUGMENTED-LRS-CL3-WEAK-RATE-MULTIPLIER  Applied bounded LRS angular weak correction

- **Status:** merged into the active augmented-PSTF WBS branch
- **Scope:** stage-scoped AP7 completion for the SciPy augmented
  weak/network RHS.  When `correction_level=3`, the combined RHS now
  passes the current LRS `Sigma_+` into the existing
  `sigma_plus_K2_correction_factor(...)` multiplier and records the
  applied rate contract in the weak-input metadata.
- **Key files:**
  - `src/rabbit/transport/augmented_typeI_weak_network.py` adds the
    current-shear CL3 multiplier handoff, a result-level correction
    factor, and opt-out config for the staged weak/network RHS.
  - `tests/test_augmented_typeI_weak_network_bridge.py` locks exact
    `lambda_np`/`lambda_pn` multiplier propagation and metadata
    `rate_application`.
  - `tests/test_augmented_wbs_status_ledger.py` updates the WBS ledger
    contract so AP7/AP10-AP25 are stage-scoped landed while AP4-AP6 and
    AP8-AP9 remain partial.
- **Physics added/changed:** bounded LRS `Sigma_+ K_2` CL3 multiplier
  is applied to live weak rates in the augmented SciPy combined RHS.
  This is not a full anisotropic weak-rate integration and does not
  promote public dispatch.
- **Parity before / after:** no promoted runtime parity changed.
- **Known red tests:** none introduced in the focused AP25 suite.
- **Docs updated:**
  - `README.md`
  - `STATUS.md`
  - `SUPPORTED_CAPABILITIES.md`
  - `docs/ROADMAP_STATE_OF_RECORD.md`
  - `IMPLEMENTATION_GUIDE_TYPEI_AUGMENTED_PSTF_NOQKE_WBS.md`
- **Self-audit verdict:** pass, with explicit bounded-LRS/no-promotion
  scope.

### AP26-AUGMENTED-LRS-WEAK-RATE-CANDIDATE-GATE  Bounded LRS weak-rate gate

- **Status:** merged into the active augmented-PSTF WBS branch
- **Scope:** AP8 weak-rate candidate sub-gate for the AP25 bounded LRS
  CL3 multiplier.  The gate sweeps `Sigma_+` at smoke scale, records
  `cl3_angular_rate_correction_factor`, `lambda_np`, `lambda_pn`, and
  reference-relative deltas, and enforces explicit factor/rate limits.
- **Key files:**
  - `src/rabbit/validation/augmented_stability.py` adds
    `AugmentedLRSWeakRateCandidateGateSpec`,
    `AugmentedLRSWeakRateCandidateGateCase`, and
    `run_augmented_lrs_cl3_weak_rate_candidate_gate(...)`.
  - `tests/test_augmented_stability_envelope.py` locks accepted
    small-shear cases, limit-failure reporting, invalid-spec rejection,
    and public case validation.
  - `docs/IMPLEMENTATION_GUIDE_TYPEI_AUGMENTED_PSTF_NOQKE_WBS.md`
    records AP26 while keeping AP8 partial.
- **Physics added/changed:** no new rate kernel beyond AP25.  This is a
  deterministic candidate gate around the applied bounded LRS
  `Sigma_+ K_2` multiplier; collision-coupled and full-BBN candidate
  gates remain planned.
- **Parity before / after:** no promoted runtime parity changed.
- **Known red tests:** none introduced in the focused AP26 suite.
- **Docs updated:**
  - `README.md`
  - `STATUS.md`
  - `SUPPORTED_CAPABILITIES.md`
  - `docs/ROADMAP_STATE_OF_RECORD.md`
  - `IMPLEMENTATION_GUIDE_TYPEI_AUGMENTED_PSTF_NOQKE_WBS.md`
- **Self-audit verdict:** pass, with explicit weak-rate-sub-gate scope.

### AP27-AUGMENTED-LRS-COLLISION-FEEDBACK-CANDIDATE-GATE  Source-variant gate

- **Status:** merged into the active augmented-PSTF WBS branch
- **Scope:** AP8 collision-feedback candidate sub-gate around the
  AP22/AP23/AP36 source-variant artifact.  The gate checks selected
  standard-relative thermo/network deltas and collision source moments
  for deterministic source variants and forwards the AP36 LRS angular
  source-update policy plus solver controls into the AP35 wrapper route.
- **Key files:**
  - `src/rabbit/validation/augmented_stability.py` adds
    `AugmentedLRSCollisionFeedbackCandidateGateSpec`,
    `AugmentedLRSCollisionFeedbackCandidateGateCase`, and
    `run_augmented_lrs_collision_feedback_candidate_gate(...)`.
  - `tests/test_augmented_stability_envelope.py` locks accepted
    standard/combined/angular variants, source-update and solver-control
    propagation, a real AP35/AP36 angular-wrapper smoke gate, limit-failure
    reporting, invalid-spec rejection, and public case validation.
  - `src/rabbit/validation/augmented_convergence.py` updates artifact
    limitations to reflect the bounded LRS weak-rate multiplier while
    preserving the full-anisotropic-integration blocker.
- **Physics added/changed:** no new collision kernel.  This is a
  deterministic candidate gate around existing source variants and the AP36
  angular-wrapper route, not a full physical angular collision-kernel or
  full-BBN candidate gate.
- **Parity before / after:** no promoted runtime parity changed.
- **Known red tests:** AP27 routing upgrade tests first failed because the LRS
  candidate spec did not accept `source_update_policy` or solver controls.
- **Docs updated:**
  - `README.md`
  - `STATUS.md`
  - `SUPPORTED_CAPABILITIES.md`
  - `docs/ROADMAP_STATE_OF_RECORD.md`
  - `IMPLEMENTATION_GUIDE_TYPEI_AUGMENTED_PSTF_NOQKE_WBS.md`
- **Self-audit verdict:** pass, with explicit source-variant-sub-gate
  scope.

### AP28-AUGMENTED-LRS-3T-SPAN-CANDIDATE-GATE  Span-ladder gate

- **Status:** merged into the active augmented-PSTF WBS branch
- **Scope:** AP8 staged 3T candidate gate over explicit `N_span`
  ladders.  The gate repeatedly runs the AP22/AP23 collision-feedback
  artifact path for standard and opt-in source variants, then records
  thermo/H/network/source observables and limit checks for every
  span/variant case.  The LRS angular span route forwards the AP36
  frozen/live source-update policy.  The smoke default remains short; longer
  spans are explicit optional gates because the current SciPy-first
  source-variant path is expensive.
- **Key files:**
  - `src/rabbit/validation/augmented_stability.py` adds
    `AugmentedLRSFullSpan3TCandidateGateSpec`,
    `AugmentedLRSFullSpan3TCandidateGateCase`, and
    `run_augmented_lrs_full_span_3t_candidate_gate(...)`.
  - `tests/test_augmented_stability_envelope.py` locks span-ladder
    validation, accepted standard/combined/angular variants, source-update
    propagation, source-moment and thermo/H/network limit-failure reporting,
    invalid-spec rejection, and public case validation.
  - `docs/IMPLEMENTATION_GUIDE_TYPEI_AUGMENTED_PSTF_NOQKE_WBS.md`
    marks AP8 stage-scoped landed; AP4 through AP6 and AP9 were still
    partial at that stage.
- **Physics added/changed:** no new collision kernel.  This is a
  staged 3T full-state candidate gate around existing deterministic source
  variants and the AP36 angular-wrapper route, not a full physical angular
  collision-kernel, full anisotropic weak-rate integration, or public full-BBN
  dispatch.
- **Parity before / after:** no promoted runtime parity changed.
- **Known red tests:** the new focused AP28 tests first failed on the
  missing public gate classes/functions, then passed after implementation.
- **Docs updated:**
  - `README.md`
  - `STATUS.md`
  - `SUPPORTED_CAPABILITIES.md`
  - `docs/ROADMAP_STATE_OF_RECORD.md`
  - `IMPLEMENTATION_GUIDE_TYPEI_AUGMENTED_PSTF_NOQKE_WBS.md`
- **Self-audit verdict:** pass, with explicit span-ladder/no-promotion
  scope.

### AP29-AUGMENTED-LRS-COLLISION-FEEDBACK-CONVERGENCE  3T convergence runners

- **Status:** merged into the active augmented-PSTF WBS branch
- **Scope:** AP5 deterministic convergence runners for the AP22/AP23/AP36
  collision-feedback source-variant artifact path.  The new runners
  sweep `ell_max`, `N_q`, and `N_mu`, then extract selected
  standard-relative thermo/network deltas, source moments, and solve
  effort for a selected source variant while forwarding AP36 LRS angular
  source-update policy.
- **Key files:**
  - `src/rabbit/validation/augmented_convergence.py` adds
    `run_augmented_lrs_3t_collision_feedback_ell_convergence(...)`,
    `run_augmented_lrs_3t_collision_feedback_q_convergence(...)`, and
    `run_augmented_lrs_3t_collision_feedback_angular_convergence(...)`.
  - `tests/test_augmented_convergence.py` locks the three convergence
    report labels, source-variant observable extraction, AP36 source-update
    propagation, a real AP35/AP36 angular-wrapper q-convergence smoke path,
    bad-source rejection, and JSON-artifact compatibility.
  - `docs/IMPLEMENTATION_GUIDE_TYPEI_AUGMENTED_PSTF_NOQKE_WBS.md`
    marks AP5 stage-scoped landed while keeping AP4/AP6/AP9 partial.
- **Physics added/changed:** no new collision kernel.  This adds
  deterministic convergence-report coverage around the existing staged
  3T source-variant path and the AP36 angular-wrapper route; it is not
  promotion-tolerance full physical collision/BBN convergence.
- **Parity before / after:** no promoted runtime parity changed.
- **Known red tests:** the new focused AP29 tests first failed on the
  missing convergence runner imports; the AP36 routing upgrade tests later
  failed because the LRS convergence runners did not accept
  `source_update_policy`.
- **Docs updated:**
  - `README.md`
  - `STATUS.md`
  - `SUPPORTED_CAPABILITIES.md`
  - `docs/ROADMAP_STATE_OF_RECORD.md`
  - `IMPLEMENTATION_GUIDE_TYPEI_AUGMENTED_PSTF_NOQKE_WBS.md`
- **Self-audit verdict:** pass, with explicit convergence-runner/no-promotion
  scope.

### AP30-AUGMENTED-QMC-COLLISION-MOMENT-REPORTS  Source-moment QMC reports

- **Status:** merged into the active augmented-PSTF WBS branch
- **Scope:** AP9 deterministic QMC control-variate reporting for named
  collision source moments.  The report estimates moment vectors such
  as `dQ_nue_pair_N` and `dQ_nux_bank_N` using replay-stable Sobol
  samples, explicit control integrals, deterministic reference moments,
  per-moment errors, adjacent deltas, and tail-convergence selection.
- **Key files:**
  - `src/rabbit/validation/qmc_control_variate.py` adds
    `QMCCollisionMomentConvergenceReport` and
    `build_qmc_collision_moment_convergence_report(...)`.
  - `tests/test_qmc_control_variate.py` locks multi-moment convergence,
    replay identity, missing-key rejection, and report validation.
  - `docs/IMPLEMENTATION_GUIDE_TYPEI_AUGMENTED_PSTF_NOQKE_WBS.md`
    marks AP9 stage-scoped landed while keeping AP4/AP6 partial.
- **Physics added/changed:** no new collision kernel.  This is a QMC
  validation/report accelerator surface for named source moments; AP6
  remains the home for full angular collision-kernel evaluation.
- **Parity before / after:** no promoted runtime parity changed.
- **Known red tests:** the new focused AP30 tests first failed on the
  missing report class/function imports, then passed after implementation.
- **Docs updated:**
  - `README.md`
  - `STATUS.md`
  - `SUPPORTED_CAPABILITIES.md`
  - `docs/ROADMAP_STATE_OF_RECORD.md`
  - `IMPLEMENTATION_GUIDE_TYPEI_AUGMENTED_PSTF_NOQKE_WBS.md`
- **Self-audit verdict:** pass, with explicit QMC-report/no-kernel
  scope.

### AP31-AUGMENTED-LRS-ANGULAR-NUE-SCATTERING  Live angular bridge

- **Status:** merged into the active augmented-PSTF WBS branch
- **Scope:** AP6 live angular collision-kernel subpath for LRS `nu-e`
  scattering.  The deterministic `nu-e` scattering reference is
  evaluated separately at each LRS angular node from the reconstructed
  augmented distribution `f(q,mu)`, projected onto a number-conserving
  elastic-scattering source that preserves the energy moment, then
  projected back into PSTF modes.
- **Key files:**
  - `src/rabbit/transport/augmented_collision_bridge.py` adds
    `AugmentedAngularCollisionBridgeResult` and
    `evaluate_augmented_nue_scattering_angular_bridge(...)`.
  - `tests/test_augmented_collision_bridge.py` locks FD quietness,
    live quadrupole source projection into the `A2` mode, input
    validation, elastic-scattering number closure, result validation,
    and unchanged existing bridge behavior.
  - `docs/IMPLEMENTATION_GUIDE_TYPEI_AUGMENTED_PSTF_NOQKE_WBS.md`
    records AP31 while keeping AP6 partial for angular pair-process,
    angular `nu-nu`, and non-LRS angular kernels.
- **Physics added/changed:** real LRS angular-node `nu-e` scattering
  evaluation from live anisotropic distributions with a deterministic
  two-moment correction that closes the elastic number moment while
  preserving the energy moment.  This is not yet the full AP6 collision
  kernel, and it is not public runtime coupling.
- **Parity before / after:** no promoted runtime parity changed.
- **Known red tests:** the new focused AP31 tests first failed on the
  missing angular bridge class/function imports, then passed after
  implementation.
- **Docs updated:**
  - `README.md`
  - `STATUS.md`
  - `SUPPORTED_CAPABILITIES.md`
  - `docs/ROADMAP_STATE_OF_RECORD.md`
  - `IMPLEMENTATION_GUIDE_TYPEI_AUGMENTED_PSTF_NOQKE_WBS.md`
- **Self-audit verdict:** pass, with explicit angular-subpath/no-public
  runtime scope and elastic-scattering number closure locked by focused
  tests.

### AP32-AUGMENTED-LRS-ANGULAR-ELECTRON-PAIR  Pair-process bridge

- **Status:** merged into the active augmented-PSTF WBS branch
- **Scope:** AP6 live LRS angular electromagnetic collision subpath.
  The deterministic `nu-e` scattering and pair-process references are
  evaluated at each LRS angular node from reconstructed `f(q,mu)` for
  `nu_e`, `anti-nu_e`, and the effective `nu_x` bank, then projected
  back into PSTF modes with number-closed elastic-scattering component
  diagnostics.
- **Key files:**
  - `src/rabbit/transport/augmented_collision_bridge.py` adds
    `evaluate_augmented_electron_pair_angular_bridge(...)`, component
    moment diagnostics on `AugmentedAngularCollisionBridgeResult`, and
    shared angular bridge input validation.
  - `tests/test_augmented_collision_bridge.py` locks FD quietness,
    component moment summation, elastic-scattering number closure, live
    pair-process quadrupole projection for all supported species banks,
    required-species validation, and unchanged existing bridge behavior.
  - `docs/IMPLEMENTATION_GUIDE_TYPEI_AUGMENTED_PSTF_NOQKE_WBS.md`
    records AP32 while keeping AP6 partial for angular `nu-nu`, non-LRS
    angular kernels, and public runtime coupling.
- **Physics added/changed:** live LRS angular-node pair-process
  evaluation from anisotropic augmented distributions, composed with
  number-closed elastic `nu-e` scattering components.  This is still a
  staged diagnostic bridge, not a promoted collision-coupled BBN runtime.
- **Parity before / after:** no promoted runtime parity changed.
- **Known red tests:** the focused AP32 tests first failed on the
  missing angular electron/pair bridge import, then passed after
  implementation.
- **Docs updated:**
  - `README.md`
  - `STATUS.md`
  - `SUPPORTED_CAPABILITIES.md`
  - `docs/ROADMAP_STATE_OF_RECORD.md`
  - `IMPLEMENTATION_GUIDE_TYPEI_AUGMENTED_PSTF_NOQKE_WBS.md`
- **Self-audit verdict:** pass, with explicit angular electromagnetic
  subpath/no-public-runtime scope and elastic-scattering number closure
  locked by focused tests.

### AP33-AUGMENTED-LRS-ANGULAR-NUNU  Diagonal nu-nu bridge

- **Status:** merged into the active augmented-PSTF WBS branch
- **Scope:** AP6 live LRS angular diagonal no-QKE `nu-nu` collision
  subpath.  The AP81 pairwise diagonal `nu-nu` 2-to-2 reference is evaluated at
  each LRS angular node from reconstructed `f(q,mu)` for `nu_e`,
  `anti-nu_e`, and the effective `nu_x` bank, then projected back into PSTF
  modes after explicit per-bank number closure and effective-`nu_x`
  weighted-energy closure projection.
- **Key files:**
  - `src/rabbit/transport/augmented_collision_bridge.py` adds
    `evaluate_augmented_nunu_angular_bridge(...)` using the AP81 pairwise
    deterministic diagonal no-QKE `nu-nu` reference and the shared angular
    bridge validation/projection path.
  - `tests/test_augmented_collision_bridge.py` locks FD quietness at
    deterministic-reference precision, live quadrupole redistribution
    projection, per-bank number closure, weighted bank-energy conservation,
    pairwise/statistical diagnostics, required-species validation, and unchanged existing bridge
    behavior.
  - `docs/IMPLEMENTATION_GUIDE_TYPEI_AUGMENTED_PSTF_NOQKE_WBS.md`
    records AP33 while keeping AP6 partial for non-LRS angular kernels
    and public runtime coupling.
- **Physics added/changed:** live LRS angular-node pairwise diagonal `nu-nu`
  redistribution from anisotropic augmented distributions using the executable
  six-monomial Pauli factor.  This remains classical no-QKE diagnostic
  collision physics, not off-diagonal QKE or promoted collision-coupled BBN
  runtime.
- **Parity before / after:** no promoted runtime parity changed.
- **Known red tests:** the focused AP33 tests first failed on the
  missing angular `nu-nu` bridge import; after implementation the FD
  quietness tolerance was aligned to the existing deterministic-reference
  roundoff floor.
- **Docs updated:**
  - `README.md`
  - `STATUS.md`
  - `SUPPORTED_CAPABILITIES.md`
  - `docs/ROADMAP_STATE_OF_RECORD.md`
  - `IMPLEMENTATION_GUIDE_TYPEI_AUGMENTED_PSTF_NOQKE_WBS.md`
- **Self-audit verdict:** pass, with explicit diagonal no-QKE/no-public-runtime
  scope.

### AP34-AUGMENTED-NONLRS-ANGULAR-COLLISION  Generic angular bridges

- **Status:** merged into the active augmented-PSTF WBS branch
- **Scope:** AP6 non-LRS S2 angular collision-reference closure.  The
  AP31-AP33 bridges now use generic angular-node v2 closure contracts
  and are tested on the staged non-LRS S2 `{monopole, W_plus, W_minus}`
  basis.
- **Key files:**
  - `src/rabbit/transport/augmented_collision_bridge.py` updates the
    angular bridge closure contracts from LRS-only v1 names to generic
    angular-node v2 names and documents LRS/non-LRS support.
  - `tests/test_augmented_collision_bridge.py` locks non-LRS minus-mode
    projection through `nu-e`, electromagnetic pair-process, and diagonal
    no-QKE `nu-nu` angular bridges.
  - `docs/IMPLEMENTATION_GUIDE_TYPEI_AUGMENTED_PSTF_NOQKE_WBS.md`
    marks AP6 stage-scoped landed while keeping runtime coupling outside
    this AP.
- **Physics added/changed:** the deterministic angular collision
  references are now verified beyond LRS on the staged non-LRS S2 basis.
  This is still diagnostic no-QKE collision-reference plumbing, not a
  promoted collision-coupled solver.
- **Parity before / after:** no promoted runtime parity changed.
- **Known red tests:** focused AP34 tests first failed on the old
  LRS-only closure contracts, then passed after generic v2 contracts and
  non-LRS S2 coverage were added.
- **Docs updated:**
  - `README.md`
  - `STATUS.md`
  - `SUPPORTED_CAPABILITIES.md`
  - `docs/ROADMAP_STATE_OF_RECORD.md`
  - `IMPLEMENTATION_GUIDE_TYPEI_AUGMENTED_PSTF_NOQKE_WBS.md`
- **Self-audit verdict:** pass, with explicit non-LRS collision-reference
  coverage/no-public-runtime scope.

### AP35-AUGMENTED-ANGULAR-COLLISION-THERMO-SOURCE  Opt-in 3T source

- **Status:** merged into the active augmented-PSTF WBS branch
- **Scope:** AP18 thermo-feedback source integration for the landed AP31-AP34
  angular collision bridges.  The new source factory composes the live
  angular electromagnetic bridge and the live angular diagonal no-QKE `nu-nu`
  bridge into `dQ_nue_pair_N` and `dQ_nux_bank_N` moments and now preserves
  the projected angular-collision `dA_modes` hierarchy source for 3T RHS
  coupling.
- **Key files:**
  - `src/rabbit/transport/augmented_collision_bridge.py` adds
    `build_augmented_angular_collision_thermo_source(...)`, returning an
    explicit `Augmented3TCollisionThermoSource` callback with numeric
    component diagnostics.
  - `tests/test_augmented_collision_bridge.py` locks isotropic agreement
    against the existing combined monopole source and the new closure
    diagnostics.
  - `tests/test_augmented_typeI_weak_network_3t_solve.py` locks acceptance by
    the AP18 3T shell source evaluator and a smoke-scale direct LRS
    angular-collision 3T wrapper run without paying a long live-source solve
    loop in the smoke suite.
- **Physics added/changed:** live angular collision-reference moments can now
  feed the existing 3T thermo feedback interface through an explicit callback,
  the same callback can carry the projected collision term for the augmented
  hierarchy, and an opt-in LRS direct wrapper wires that source into the
  AP15/AP18 solve path with explicit source-update policy.  This is still
  staged coupling, not a default or promoted collision-coupled BBN solve.
- **Parity before / after:** no public dispatch or promoted runtime parity
  changed.
- **Known red tests:** the first focused 3T test attempted a full solve and was
  too expensive for the smoke path; it was narrowed to the AP18 source-evaluator
  contract after the bridge-level source test passed.
- **Docs updated:**
  - `README.md`
  - `STATUS.md`
  - `SUPPORTED_CAPABILITIES.md`
  - `docs/ROADMAP_STATE_OF_RECORD.md`
  - `IMPLEMENTATION_GUIDE_TYPEI_AUGMENTED_PSTF_NOQKE_WBS.md`
- **Self-audit verdict:** pass, with explicit opt-in/no-public-runtime scope.

### AP36-AUGMENTED-ANGULAR-COLLISION-FEEDBACK-VARIANT  Artifact/gate wiring

- **Status:** merged into the active augmented-PSTF WBS branch
- **Scope:** validation/report plumbing for the AP35 angular thermo-source
  callback.  The collision-feedback artifact path, candidate gate,
  span-ladder gate, and convergence runners now accept an explicit `angular`
  source variant.  The LRS artifact route now calls the AP35 direct wrapper
  with explicit frozen-initial-state/live-RHS source-update policy.
- **Key files:**
  - `src/rabbit/validation/augmented_convergence.py` routes the `angular`
    source variant through
    `run_augmented_lrs_angular_collision_weak_network_3T_solve(...)`, records
    source-update policy and routing metadata, updates the implementation stage
    to AP36, and keeps the existing q/angular/ell convergence wrappers.
  - `scripts/run_augmented_3t_collision_feedback_artifact.py` exposes the
    AP35 angular source-update policy for deterministic artifact runs.
  - `src/rabbit/validation/augmented_stability.py` admits `angular` in the
    smoke-scale candidate and span-ladder gate specs.
  - `tests/test_augmented_convergence.py` and
    `tests/test_augmented_stability_envelope.py` lock artifact diagnostics,
    direct-wrapper routing, real LRS angular-wrapper smoke output, variant
    validation, gate acceptance, and q-convergence extraction.
- **Physics added/changed:** no new collision kernel beyond AP35.  This moves
  the AP36 LRS artifact from validation-local callback construction to the
  same AP35 direct solve wrapper that users can call, making frozen/live source
  update policy part of the deterministic artifact contract without making the
  path default.
- **Parity before / after:** no public dispatch or promoted runtime parity
  changed.
- **Known red tests:** focused AP36 tests first failed because `angular` was an
  unknown source variant in the artifact/convergence/gate validators; the
  routing upgrade test later failed because the LRS artifact still rebuilt a
  callback locally and did not accept `source_update_policy`.
- **Docs updated:**
  - `README.md`
  - `STATUS.md`
  - `SUPPORTED_CAPABILITIES.md`
  - `docs/ROADMAP_STATE_OF_RECORD.md`
  - `IMPLEMENTATION_GUIDE_TYPEI_AUGMENTED_PSTF_NOQKE_WBS.md`
- **Self-audit verdict:** pass, with explicit staged-variant/no-public-runtime
  scope.

### AP37-NLRS-SOURCE-COEVOLUTION-SHELL  Non-LRS source-only solve wiring

- **Status:** merged into the active augmented-PSTF WBS branch
- **Scope:** AP4 non-LRS source-only coevolution shell.  The staged S2
  quadrupole source projection now sits inside a SciPy `solve_ivp` shell that
  evolves `Sigma_+`, `Sigma_-`, and `{monopole, W_+, W_-}` augmented modes.
  This is not nonlinear non-LRS transport and has no public dispatch.
- **Key files:**
  - `src/rabbit/transport/augmented_typeI_nonlrs_collisionless.py` adds
    `AugmentedNonLRSSourceCollisionlessConfig`,
    `augmented_nonlrs_source_collisionless_rhs(...)`, and
    `run_augmented_nonlrs_source_collisionless_solve(...)`.
  - `src/rabbit/transport/__init__.py` exports the staged AP37 API.
  - `tests/test_augmented_typeI_nonlrs_collisionless.py` locks source
    projection agreement, live `Pi_-` feedback, and a short plus/minus solve.
- **Physics added/changed:** every RHS call reconstructs the current non-LRS
  augmented distribution on the S2 grid, computes live `Pi_+` and `Pi_-`
  stress moments, and feeds those moments into the diagonal shear equations
  while keeping the transport derivative at the source-only AP4 projection.
- **Parity before / after:** no promoted runtime parity changed.  This only
  promotes the AP4 staged non-LRS block from static projection to a bounded
  reference solve shell.
- **Known red tests:** focused AP37 tests first failed because the new
  non-LRS source-coevolution dataclasses/RHS/solve API did not exist.
- **Docs updated:**
  - `README.md`
  - `STATUS.md`
  - `SUPPORTED_CAPABILITIES.md`
  - `docs/ROADMAP_STATE_OF_RECORD.md`
  - `IMPLEMENTATION_GUIDE_TYPEI_AUGMENTED_PSTF_NOQKE_WBS.md`
- **Self-audit verdict:** pass, with explicit source-only/non-public-runtime
  scope.

### AP38-NLRS-SOURCE-WEAK-NETWORK-SHELL  Non-LRS fixed-thermo network wiring

- **Status:** merged into the active augmented-PSTF WBS branch
- **Scope:** AP4 non-LRS source-only weak/network shell.  The AP37
  plus/minus source-only solve path now has a fixed-thermo PRIMAT network
  companion that uses live S2 angular monopoles.
- **Key files:**
  - `src/rabbit/transport/augmented_typeI_weak_network.py` adds
    `AugmentedNonLRSWeakNetworkConfig`,
    `augmented_nonlrs_source_collisionless_weak_network_rhs(...)`, and
    `run_augmented_nonlrs_source_collisionless_weak_network_solve(...)`.
  - `tests/test_augmented_typeI_nonlrs_weak_network_solve.py` locks live S2
    weak monopoles, CL3 plus/minus metadata, fixed-thermo solve evolution,
    and missing-species rejection.
- **Physics added/changed:** the non-LRS S2 distribution is reconstructed
  inside the weak/network RHS, `nu_e`/`anti_nu_e` monopoles are extracted for
  live weak rates, and the PRIMAT network derivative is packed into the same
  `d/dN` state as `Sigma_+`, `Sigma_-`, and the augmented modes.
- **Scope boundary:** CL3 non-LRS angular weak information is metadata-only.
  The LRS `Sigma_+ K_2` multiplier is not applied to the non-LRS path, and
  full anisotropic weak-rate integration, collision-sourced thermodynamics,
  nonlinear non-LRS transport, and public dispatch remain outside this AP.
- **Known red tests:** focused AP38 tests first failed because the non-LRS
  weak/network config, RHS, and solve API did not exist.
- **Docs updated:**
  - `README.md`
  - `STATUS.md`
  - `SUPPORTED_CAPABILITIES.md`
  - `docs/ROADMAP_STATE_OF_RECORD.md`
  - `IMPLEMENTATION_GUIDE_TYPEI_AUGMENTED_PSTF_NOQKE_WBS.md`
- **Self-audit verdict:** pass, with explicit fixed-thermo/metadata-only
  non-LRS weak-rate scope.

### AP39-NLRS-SOURCE-3T-SHELL  Non-LRS source-only 3T thermo/Hubble wiring

- **Status:** merged into the active augmented-PSTF WBS branch
- **Scope:** AP4 non-LRS source-only 3T thermo/Hubble shell.  The AP38
  fixed-thermo weak/network path now has a dynamic 3T companion that evolves
  temperatures and recomputes Hubble with both diagonal shear components.
- **Key files:**
  - `src/rabbit/transport/augmented_typeI_weak_network.py` adds
    `AugmentedNonLRSSourceCollisionlessWeakNetwork3TSolveResult` and
    `run_augmented_nonlrs_source_collisionless_weak_network_3T_solve(...)`.
  - `tests/test_augmented_typeI_nonlrs_weak_network_solve.py` locks dynamic
    temperature evolution, dynamic Hubble with `Sigma_+^2+Sigma_-^2`,
    plus/minus mode evolution, live weak rates, and abundance normalization.
- **Physics added/changed:** `T_gamma`, `T_nu_e`, and `T_nu_x` are packed
  into the same SciPy state as `Sigma_+`, `Sigma_-`, augmented modes, and the
  PRIMAT abundances.  Each RHS call evaluates
  `H(T_gamma,T_nu_e,T_nu_x,Sigma_+^2+Sigma_-^2)` and the standard 3T table
  thermo RHS.
- **Scope boundary:** this is still source-only non-LRS transport and
  standard 3T table thermodynamics.  Collision-moment thermo feedback,
  nonlinear non-LRS transport, full anisotropic weak-rate integration, and
  public dispatch remain outside this AP.
- **Known red tests:** focused AP39 tests first failed because the non-LRS
  source-only 3T solve API did not exist.
- **Docs updated:**
  - `README.md`
  - `STATUS.md`
  - `SUPPORTED_CAPABILITIES.md`
  - `docs/ROADMAP_STATE_OF_RECORD.md`
  - `IMPLEMENTATION_GUIDE_TYPEI_AUGMENTED_PSTF_NOQKE_WBS.md`
- **Self-audit verdict:** pass, with explicit source-only/non-public-runtime
  scope.

### AP40-NLRS-COLLISION-THERMO-HOOK  Non-LRS opt-in 3T source callback

- **Status:** merged into the active augmented-PSTF WBS branch
- **Scope:** opt-in collision-moment thermo feedback hook for the AP39
  non-LRS 3T shell.  This mirrors the LRS AP18 hook shape while adding
  `Sigma_-` and the S2 grid to the callback context.  The hook also accepts
  an optional projected collision `dA_modes` payload and adds it to the
  augmented hierarchy RHS when supplied.
- **Key files:**
  - `src/rabbit/transport/augmented_typeI_weak_network.py` adds a non-LRS
    collision-source evaluator and a `collision_thermo_source=` option on
    `run_augmented_nonlrs_source_collisionless_weak_network_3T_solve(...)`.
  - `tests/test_augmented_typeI_nonlrs_weak_network_solve.py` locks callback
    invocation, `Sigma_-`/grid propagation, diagnostics, feedback metadata,
    and bad-source rejection.
- **Physics added/changed:** when an explicit callback returns
  `Augmented3TCollisionThermoSource`, the non-LRS 3T shell evaluates
  `coupled_3T_rhs_from_collision_moments(...)` instead of the standard table
  thermo RHS; if that source includes `dA_modes`, the same RHS call also adds
  the collision term to the staged `{monopole, W_+, W_-}` hierarchy.  Without
  a callback, the AP39 standard 3T table path is unchanged.
- **Scope boundary:** this is hook plumbing only.  No default non-LRS
  collision source, nonlinear non-LRS transport, full anisotropic weak-rate
  integration, or public dispatch is promoted.
- **Known red tests:** focused AP40 tests first failed because
  `collision_thermo_source` was not accepted by the non-LRS 3T solve API.
- **Docs updated:**
  - `README.md`
  - `STATUS.md`
  - `SUPPORTED_CAPABILITIES.md`
  - `docs/ROADMAP_STATE_OF_RECORD.md`
  - `IMPLEMENTATION_GUIDE_TYPEI_AUGMENTED_PSTF_NOQKE_WBS.md`
- **Self-audit verdict:** pass, with explicit opt-in/no-default-source scope.

### AP41-NLRS-ANGULAR-COLLISION-SOURCE  Non-LRS S2 angular thermo source

- **Status:** merged into the active augmented-PSTF WBS branch
- **Scope:** opt-in non-LRS S2 angular collision thermo-source factory for the
  AP40 3T callback hook.  It wraps the generic AP35 angular source on the
  staged non-LRS `{monopole, W_+, W_-}` grid.
- **Key files:**
  - `src/rabbit/transport/augmented_collision_bridge.py` adds
    `build_augmented_nonlrs_angular_collision_thermo_source(...)`.
  - `tests/test_augmented_collision_bridge.py` locks S2-grid diagnostics,
    `Sigma_-` propagation, and equivalence to the generic angular source on
    the same grid.
- **Physics added/changed:** the non-LRS factory composes the existing angular
  electromagnetic pair-process bridge and angular diagonal no-QKE `nu-nu`
  bridge for a live S2 augmented state, returning
  `Augmented3TCollisionThermoSource` for the AP40 hook.
- **Scope boundary:** this source remains explicit and opt-in.  It does not
  make non-LRS collision feedback default, does not add nonlinear non-LRS
  transport, and does not promote public full-BBN dispatch.
- **Known red tests:** focused AP41 tests first failed because the non-LRS
  angular thermo-source factory was not exported by the collision bridge.
- **Docs updated:**
  - `README.md`
  - `STATUS.md`
  - `SUPPORTED_CAPABILITIES.md`
  - `docs/ROADMAP_STATE_OF_RECORD.md`
  - `IMPLEMENTATION_GUIDE_TYPEI_AUGMENTED_PSTF_NOQKE_WBS.md`
- **Self-audit verdict:** pass, with explicit opt-in/no-default-source scope.

### AP42-NLRS-COLLISION-FEEDBACK-ARTIFACT  Non-LRS 3T source-variant report

- **Status:** merged into the active augmented-PSTF WBS branch
- **Scope:** deterministic non-LRS 3T collision-feedback artifact runner for
  the AP40 callback hook and AP41 S2 angular source.
- **Key files:**
  - `src/rabbit/validation/augmented_convergence.py` adds
    `build_augmented_nonlrs_3t_collision_feedback_artifact(...)` and
    `write_augmented_nonlrs_3t_collision_feedback_artifact(...)`.
  - `scripts/run_augmented_nonlrs_3t_collision_feedback_artifact.py` emits
    the JSON artifact from the command line.
  - `tests/test_augmented_convergence.py` locks the artifact contract and
    writer.
- **Physics added/changed:** the artifact compares the standard AP39
  non-LRS 3T table RHS against the AP41 `angular` collision source variant
  through the AP40 collision-moment thermo feedback hook.  It records S2
  grid metadata, `Sigma_-` source diagnostics, AP41 source contracts,
  non-LRS plus/minus observables, source moments, and standard-relative
  observable deltas.
- **Runtime policy:** smoke defaults use
  `source_update_policy="frozen_initial_state"` so the AP41 angular source
  is evaluated once and then injected through the AP40 hook as fixed source
  moments.  `source_update_policy="live_rhs"` remains available for longer
  experiments but is not the smoke default.
- **Scope boundary:** this is a diagnostic artifact only.  It does not make
  non-LRS collision feedback default, does not add nonlinear non-LRS
  transport, does not add full anisotropic weak-rate integration, and does
  not promote public full-BBN dispatch.
- **Known red tests:** focused AP42 tests first failed because the non-LRS
  collision-feedback artifact builder and writer did not exist.
- **Docs updated:**
  - `README.md`
  - `STATUS.md`
  - `SUPPORTED_CAPABILITIES.md`
  - `docs/ROADMAP_STATE_OF_RECORD.md`
  - `IMPLEMENTATION_GUIDE_TYPEI_AUGMENTED_PSTF_NOQKE_WBS.md`
- **Self-audit verdict:** pass, with explicit smoke-default frozen-source and
  no-public-dispatch scope.

### AP43-NLRS-COLLISION-FEEDBACK-CONVERGENCE  Non-LRS q/S2 ladder reports

- **Status:** merged into the active augmented-PSTF WBS branch
- **Scope:** convergence runners over the AP42 non-LRS collision-feedback
  artifact surface.
- **Key files:**
  - `src/rabbit/validation/augmented_convergence.py` adds
    `run_augmented_nonlrs_3t_collision_feedback_q_convergence(...)`,
    `run_augmented_nonlrs_3t_collision_feedback_nmu_convergence(...)`,
    and `run_augmented_nonlrs_3t_collision_feedback_nphi_convergence(...)`.
  - `tests/test_augmented_convergence.py` locks runner labels, selected
    observables, and bad-input rejection using a deterministic fake artifact.
- **Physics added/changed:** AP43 does not add new physics kernels.  It makes
  the AP42 standard-relative non-LRS collision-feedback observables
  convergence-reportable over q and S2 angular resolution ladders, including
  `Sigma_-`, `Pi_-`, `Aminus_rms`, thermo/network outputs, source moments,
  selected deltas, and solve effort.
- **Scope boundary:** the runners are diagnostic convergence surfaces.  They
  do not make AP41/AP42 collision feedback default, do not add nonlinear
  non-LRS transport, and do not claim promotion-tolerance full physical
  collision/BBN convergence.
- **Known red tests:** focused AP43 tests first failed because the non-LRS
  collision-feedback convergence runners were not exported.
- **Docs updated:**
  - `README.md`
  - `STATUS.md`
  - `SUPPORTED_CAPABILITIES.md`
  - `docs/ROADMAP_STATE_OF_RECORD.md`
  - `IMPLEMENTATION_GUIDE_TYPEI_AUGMENTED_PSTF_NOQKE_WBS.md`
- **Self-audit verdict:** pass, with AP5 stage-scoped convergence-only
  boundary.

### AP44-NLRS-COLLISION-FEEDBACK-CANDIDATE-GATE  Non-LRS AP8 diagnostic gate

- **Status:** merged into the active augmented-PSTF WBS branch
- **Scope:** candidate-gate wrapper over the AP42 non-LRS
  collision-feedback artifact surface.
- **Key files:**
  - `src/rabbit/validation/augmented_stability.py` adds
    `AugmentedNonLRSCollisionFeedbackCandidateGateSpec`, case/report
    dataclasses, and
    `run_augmented_nonlrs_collision_feedback_candidate_gate(...)`.
  - `tests/test_augmented_stability_envelope.py` locks accepted
    standard/angular/`pstf_radial` variants, live-RHS source-evaluation budget
    reporting, limit-failure reporting, invalid-spec rejection, and public case
    validation using a deterministic fake artifact.
- **Physics added/changed:** AP44 does not add new collision kernels.  It
  turns the AP42 non-LRS standard/angular/`pstf_radial` collision-feedback
  artifact into an AP8 candidate-gate surface by checking selected
  standard-relative thermo/network deltas, plus/minus shear and stress, source
  moments, mode RMS values, solve effort, and explicit radial source-evaluation
  budgets for `pstf_radial live_rhs`.
- **Scope boundary:** the gate remains diagnostic/staged.  It does not make
  AP41/AP42 collision feedback default, does not add nonlinear non-LRS
  transport, does not add full anisotropic weak-rate integration, and does
  not promote public dispatch or full physical collision-coupled BBN.
- **Known red tests:** focused AP44 tests first failed because the non-LRS
  collision-feedback candidate-gate dataclasses and runner were not
  exported.
- **Docs updated:**
  - `README.md`
  - `STATUS.md`
  - `SUPPORTED_CAPABILITIES.md`
  - `docs/ROADMAP_STATE_OF_RECORD.md`
  - `IMPLEMENTATION_GUIDE_TYPEI_AUGMENTED_PSTF_NOQKE_WBS.md`
- **Self-audit verdict:** pass, with AP8 stage-scoped gate-only boundary.

### AP45-NLRS-COLLISION-FEEDBACK-SOURCE-POLICY  Live-vs-frozen source update artifact

- **Status:** merged into the active augmented-PSTF WBS branch
- **Scope:** deterministic artifact comparing AP42 non-LRS collision-feedback
  source update policies.
- **Key files:**
  - `src/rabbit/validation/augmented_convergence.py` adds
    `build_augmented_nonlrs_3t_collision_feedback_source_policy_artifact(...)`
    and writer support.
  - `scripts/run_augmented_nonlrs_3t_collision_feedback_source_policy_artifact.py`
    emits the JSON artifact from the command line.
  - `tests/test_augmented_convergence.py` locks live-vs-frozen observable
    deltas, solve-effort metadata, `pstf_radial` budget forwarding, policy
    validation, and JSON writer output.
- **Physics added/changed:** AP45 does not add a new collision kernel.  It
  makes the AP42 `live_rhs` source re-evaluation path a deterministic report
  surface by comparing it against the frozen-initial-state policy for the
  same standard/angular/`pstf_radial` source variants and forwarding the
  explicit radial source-evaluation budget into each AP42 artifact run.  The
  AP45 runner now also forwards the AP6 `standard_3t_plasma`
  electromagnetic radial energy-closure mode, explicit 3T initial
  temperatures, and the radial source-evaluation budget, so frozen/live
  `pstf_radial` source-policy rows can be compared under the same canonical
  3T plasma-transfer normalization.
- **Scope boundary:** the default artifact is intentionally very short-span
  smoke scale.  It does not promote `live_rhs` to the default source policy,
  does not add nonlinear non-LRS transport, does not add full anisotropic
  weak rates, and does not claim public dispatch or full physical
  collision-coupled BBN.
- **Known red tests:** focused AP45 tests first failed because the
  source-policy artifact builder and writer were not exported.
- **Docs updated:**
  - `README.md`
  - `STATUS.md`
  - `SUPPORTED_CAPABILITIES.md`
  - `docs/ROADMAP_STATE_OF_RECORD.md`
  - `IMPLEMENTATION_GUIDE_TYPEI_AUGMENTED_PSTF_NOQKE_WBS.md`
- **Self-audit verdict:** pass, with `live_rhs` validated only as a
  diagnostic smoke artifact.
- **Follow-up evidence:** `PYTHONPATH=src python scripts/run_augmented_nonlrs_3t_collision_feedback_source_policy_artifact.py --output /tmp/rabbit_ap45_nonlrs_standard_3t_radial_source_policy.json --source-variants standard,pstf_radial --pstf-radial-energy-normalization standard_3t_plasma --T-gamma0-MeV 0.8 --T-nu-e0-MeV 0.79 --T-nu-x0-MeV 0.78 --N-span-end 1e-14 --N-q 3 --N-mu 3 --N-phi 5 --method RK23 --rtol 1e-4 --atol 1e-7 --max-pstf-radial-source-evaluations 16`
  emitted frozen/live `pstf_radial` rows with
  `augmented_pstf_radial_moment_standard_3t_plasma_energy_closed_v1`,
  live source evaluations `7/16`, and sub-`1e-18` displayed EM closure
  residual.

### AP46-NLRS-SOURCE-POLICY-CANDIDATE-GATE  Live-vs-frozen policy gate

- **Status:** merged into the active augmented-PSTF WBS branch
- **Scope:** stability/candidate gate over the AP45 source-update policy
  artifact.
- **Key files:**
  - `src/rabbit/validation/augmented_stability.py` adds
    `AugmentedNonLRSSourcePolicyCandidateGateSpec`, case/report dataclasses,
    and `run_augmented_nonlrs_source_policy_candidate_gate(...)`.
  - `tests/test_augmented_stability_envelope.py` locks accepted `live_rhs`
    cases, `pstf_radial` source-evaluation budget reporting, nfev
    limit-failure reporting, invalid spec rejection, and public case
    validation.
- **Physics added/changed:** AP46 does not add a new physics kernel.  It
  converts the AP45 live-vs-frozen policy artifact into a pass/fail gate over
  selected thermo/network deltas, collision source-moment deltas, absolute
  source moments, solve effort, and explicit radial source-evaluation budgets.
- **Scope boundary:** the gate remains smoke-scale and diagnostic.  It does
  not promote `live_rhs` to the default source update policy, does not add
  nonlinear non-LRS transport, does not add full anisotropic weak rates, and
  does not claim public dispatch or full physical collision-coupled BBN.
- **Known red tests:** focused AP46 tests first failed because the
  source-policy candidate-gate dataclasses and runner were not exported.
- **Docs updated:**
  - `README.md`
  - `STATUS.md`
  - `SUPPORTED_CAPABILITIES.md`
  - `docs/ROADMAP_STATE_OF_RECORD.md`
  - `IMPLEMENTATION_GUIDE_TYPEI_AUGMENTED_PSTF_NOQKE_WBS.md`
- **Self-audit verdict:** pass, with the live source-update policy still
  bounded to a diagnostic smoke gate.

### AP47-NLRS-ANGULAR-COLLISION-SOLVE-WRAPPER  Direct opt-in 3T path

- **Status:** merged into the active augmented-PSTF WBS branch
- **Scope:** transport-level wrapper that wires the AP41 non-LRS S2 angular
  source through the AP40 non-LRS 3T collision-moment hook.
- **Key files:**
  - `src/rabbit/transport/augmented_typeI_weak_network.py` adds
    `run_augmented_nonlrs_angular_collision_weak_network_3T_solve(...)`.
  - `src/rabbit/transport/__init__.py` exports the wrapper.
  - `tests/test_augmented_typeI_nonlrs_weak_network_solve.py` locks live and
    frozen source-policy behavior, source/grid/q threading, and invalid policy
    rejection.
- **Physics added/changed:** AP47 does not add a new collision kernel.  It
  makes the existing AP41 angular source plus AP40 collision-feedback hook a
  reusable opt-in solve API rather than validation-only wiring.
- **Scope boundary:** this remains the source-only non-LRS shell with an
  explicit source policy.  It does not add nonlinear non-LRS transport, does
  not promote collision feedback to the default, does not add full anisotropic
  weak-rate integration, and does not create public dispatch.
- **Known red tests:** focused AP47 tests first failed because the direct
  wrapper was not exported.
- **Docs updated:**
  - `README.md`
  - `STATUS.md`
  - `SUPPORTED_CAPABILITIES.md`
  - `docs/ROADMAP_STATE_OF_RECORD.md`
  - `IMPLEMENTATION_GUIDE_TYPEI_AUGMENTED_PSTF_NOQKE_WBS.md`
- **Self-audit verdict:** pass, with transport API landed but still opt-in and
  diagnostic.

### AP48-NLRS-ARTIFACT-WRAPPER-ROUTING  Reuse direct 3T path

- **Status:** merged into the active augmented-PSTF WBS branch
- **Scope:** refactor the AP42 non-LRS collision-feedback artifact so its
  `angular` source variant calls the AP47 transport wrapper instead of
  carrying duplicate AP41 source construction and AP40 hook wiring in
  validation code.
- **Key files:**
  - `src/rabbit/validation/augmented_convergence.py` routes `angular` through
    `run_augmented_nonlrs_angular_collision_weak_network_3T_solve(...)`.
  - `tests/test_augmented_convergence.py` locks that only `standard` uses the
    low-level source-only 3T shell and that the angular variant forwards
    resolution, source-policy, initial-state, and solver controls to the
    wrapper.
- **Physics added/changed:** AP48 adds no new kernel and no new default path.
  It makes the artifact/gate/convergence stack consume the same opt-in
  AP41/AP40 transport API that AP47 landed.
- **Scope boundary:** this remains diagnostic staging.  It does not promote
  public dispatch, default angular collision feedback, nonlinear non-LRS
  transport, full anisotropic weak rates, or QKE.
- **Known red tests:** focused AP48 regression first failed because both
  `standard` and `angular` variants still used the lower-level source-only
  shell.
- **Docs updated:**
  - `README.md`
  - `STATUS.md`
  - `SUPPORTED_CAPABILITIES.md`
  - `docs/ROADMAP_STATE_OF_RECORD.md`
  - `IMPLEMENTATION_GUIDE_TYPEI_AUGMENTED_PSTF_NOQKE_WBS.md`
- **Self-audit verdict:** pass, with artifact routing consolidated onto the
  landed opt-in transport wrapper.

### AP49-NLRS-DIRECT-WRAPPER-ARTIFACT  Standalone AP47 evidence

- **Status:** merged into the active augmented-PSTF WBS branch
- **Scope:** deterministic JSON artifact and CLI runner for the AP47 direct
  non-LRS angular collision-feedback 3T wrapper.
- **Key files:**
  - `src/rabbit/validation/augmented_convergence.py` adds
    `build_augmented_nonlrs_angular_collision_3t_solve_artifact(...)` and its
    JSON writer.
  - `scripts/run_augmented_nonlrs_angular_collision_3t_solve_artifact.py`
    emits the AP49 smoke artifact.
  - `tests/test_augmented_convergence.py` locks wrapper argument forwarding,
    result/source diagnostics, JSON writer behavior, and CLI summary output.
- **Physics added/changed:** AP49 adds no new collision kernel.  It gives the
  AP47 wrapper a direct deterministic evidence artifact with source contract,
  observables, diagnostics, and solve effort.
- **Scope boundary:** still smoke-scale, SciPy-first, opt-in, and diagnostic.
  It does not add public dispatch, default angular collision feedback,
  nonlinear non-LRS transport, full anisotropic weak rates, or QKE.
- **Known red tests:** the first AP49 test failed because the direct artifact
  builder/writer were absent; the CLI test then failed because the script was
  absent.
- **Docs updated:**
  - `README.md`
  - `STATUS.md`
  - `SUPPORTED_CAPABILITIES.md`
  - `docs/ROADMAP_STATE_OF_RECORD.md`
  - `IMPLEMENTATION_GUIDE_TYPEI_AUGMENTED_PSTF_NOQKE_WBS.md`
- **Self-audit verdict:** pass, with standalone wrapper evidence but no
  promotion beyond staged diagnostic support.

### AP50-NLRS-DIRECT-WRAPPER-CANDIDATE-GATE  AP49 pass/fail gate

- **Status:** merged into the active augmented-PSTF WBS branch
- **Scope:** candidate gate over the AP49 direct non-LRS angular
  collision-feedback 3T solve artifact.
- **Key files:**
  - `src/rabbit/validation/augmented_stability.py` adds
    `AugmentedNonLRSAngularCollisionDirectCandidateGateSpec`,
    `AugmentedNonLRSAngularCollisionDirectCandidateGateCase`, and
    `run_augmented_nonlrs_angular_collision_direct_candidate_gate(...)`.
  - `tests/test_augmented_stability_envelope.py` locks accepted live-RHS
    cases, limit-failure reporting, spec validation, and case validation.
- **Physics added/changed:** AP50 adds no new kernel and no dispatch route.  It
  adds a smoke-scale pass/fail stability surface for the AP47/AP49 direct
  wrapper path.
- **Scope boundary:** checks source moments, plus/minus shear/stress, A-mode
  RMS, temperature/Hubble/network bounds, and solve effort only for the
  staged source-only non-LRS shell.  No nonlinear non-LRS transport, default
  collision feedback, full anisotropic weak-rate integration, or QKE is
  promoted.
- **Known red tests:** focused AP50 tests first failed because the gate API was
  absent.
- **Docs updated:**
  - `README.md`
  - `STATUS.md`
  - `SUPPORTED_CAPABILITIES.md`
  - `docs/ROADMAP_STATE_OF_RECORD.md`
  - `IMPLEMENTATION_GUIDE_TYPEI_AUGMENTED_PSTF_NOQKE_WBS.md`
- **Self-audit verdict:** pass, with direct wrapper gate staged and bounded.

### AP51-FULL-SPAN-OUTCOME-POLICY  Existing direct-wrapper classifier

- **Status:** stage-recorded (historical; not current capability)
- **Scope:** AP51 outcome policy and classifier over the existing
  AP49/AP50 direct-wrapper artifact and candidate-gate path.
- **Files changed:** `src/rabbit/validation/augmented_outcomes.py`,
  `scripts/run_augmented_nonlrs_angular_collision_direct_outcomes.py`,
  `tests/test_augmented_direct_outcomes.py`,
  `tests/test_augmented_wbs_status_ledger.py`,
  `docs/IMPLEMENTATION_GUIDE_TYPEI_AUGMENTED_PSTF_NOQKE_WBS.md`,
  generated capability/status docs, and this catalogue.
- **Physics added/changed:** no new kernel.  This PR makes solver outcomes,
  timeouts, bound failures, stiffness failures, and invalid outputs explicit so
  later convergence, plot, and SMC paths can consume only classified AP49/AP50
  runs.  The AP51 smoke default uses the runtime-stable frozen source policy;
  `live_rhs` remains an explicit diagnostic option and hard-timeout failures are
  classified rather than allowed to hang.
- **Scope boundary:** SciPy-first, opt-in, and diagnostic.  It does not promote
  public dispatch, default angular collision feedback, nonlinear non-LRS
  transport, full anisotropic weak rates, or QKE.
- **Exit gate:** focused tests classify success, stiffness failure, bound
  violation, timeout, and invalid output on existing AP49/AP50 reports; a real
  medium-span LSODA smoke run with explicit frozen source policy emits bounded
  metadata.
- **Tests run:** `PYTHONPATH=src pytest -q tests/test_augmented_direct_outcomes.py`
  passes with `10 passed`.
- **Artifact smoke:** `PYTHONPATH=src venv/bin/python
  scripts/run_augmented_nonlrs_angular_collision_direct_outcomes.py --output
  /tmp/rabbit_ap51_outcome_smoke.json` emits a passing smoke outcome report.
- **Known red tests:** none introduced in the focused AP51 suite.

### AP52-DIRECT-SOLVER-TOLERANCE-MATRIX  LSODA/Radau policy scan

- **Status:** stage-recorded (historical; not current capability)
- **Scope:** AP52 pre-budget diagnostic solver method and tolerance matrix
  for the AP47/AP49 direct wrapper over smoke-scale spans, using AP51
  classified outcomes as the input surface.
- **Files changed:** `src/rabbit/validation/augmented_solver_matrix.py`,
  `scripts/run_augmented_direct_solver_matrix.py`,
  `tests/test_augmented_direct_solver_matrix.py`, WBS/catalog/state docs,
  generated capability/status docs, and registry lock tests.
- **Physics added/changed:** no new physics.  The artifact measures runtime,
  nfev, terminal observables, source moments, and classified solver failures
  so later production gates have a defensible solver-policy candidate.
- **Scope boundary:** AP52 runs before AP55 source-moment/energy-budget closure,
  so every result remains pre-budget diagnostic.  A stable numerical policy is
  not a physical promotion, and any method/tolerance disagreement above
  configured limits keeps the path diagnostic.
- **Exit gate:** artifact and gate identify the smallest stable solver policy
  candidate or fail closed with explicit method/tolerance deltas and
  pre-budget labels.
- **Tests run:** `PYTHONPATH=src pytest -q
  tests/test_augmented_direct_solver_matrix.py` passes with `8 passed`.
- **Artifact smoke:** `PYTHONPATH=src venv/bin/python
  scripts/run_augmented_direct_solver_matrix.py --output
  /tmp/rabbit_ap52_solver_matrix_smoke.json --methods LSODA Radau --rtol 1e-4
  --atol 1e-7 --N-span-end 1e-12` emits a passing pre-budget matrix with an
  LSODA candidate policy.
- **Known red tests:** none introduced in the focused AP52 suite.

### AP53-SOURCE-UPDATE-PROMOTION-STUDY  Frozen versus live source policy

- **Status:** stage-recorded (historical; not current capability)
- **Scope:** source-update promotion study over the existing AP45
  frozen-initial-state versus `live_rhs` source-policy artifact and the AP51
  direct-wrapper outcome path.
- **Files changed:** `src/rabbit/validation/augmented_source_policy.py`,
  `scripts/run_augmented_source_policy_study.py`,
  `tests/test_augmented_source_policy_study.py`, WBS/catalog/state docs,
  generated capability/status docs, and registry lock tests.
- **Physics added/changed:** no new collision kernel.  The PR turns existing
  AP42/AP45 and AP51 executable evidence into a decision artifact that records
  when `live_rhs` is stable enough to be an AP56 candidate and when
  `frozen_initial_state` remains only a diagnostic fallback.  The AP42 surface
  now also accepts the AP6 `pstf_radial` variant and carries its explicit
  source-evaluation budget into the study rows.
- **Scope boundary:** even if `live_rhs` becomes the candidate policy, default
  collision feedback and public dispatch remain disabled.  AP55 source-budget
  closure, AP56 aggregation, full anisotropic weak rates, nonlinear non-LRS
  transport, and QKE remain outside this PR.
- **Exit gate:** report records live/frozen deltas, nfev deltas, source-moment
  deltas, AP42 `pstf_radial` budget observables, classified direct-wrapper
  failures, and whether any policy is eligible for AP56.
- **Tests run:** `PYTHONPATH=src pytest -q
  tests/test_augmented_source_policy_study.py` passes with `9 passed`.
- **Artifact smoke:** `PYTHONPATH=src venv/bin/python
  scripts/run_augmented_source_policy_study.py --output
  /tmp/rabbit_ap53_source_policy_smoke.json --surface ap42_artifact
  --N-span-end 1e-12 --policy-delta-abs-limit 1e-6
  --source-moment-abs-limit 20` emits an AP42 frozen/live study with
  `live_rhs` eligible on the AP42 surface.  The same CLI with
  `--surface direct_wrapper --direct-timeout-s 5` emits one frozen success,
  one classified `live_rhs` timeout, and no AP56 candidate policy on the
  direct-wrapper surface.
- **Known red tests:** the `pstf_radial` AP42-surface regression first failed
  because the AP53 spec rejected the variant and had no radial source-budget
  control; a follow-up fail-closed regression first failed because a missing
  `pstf_radial live_rhs` budget was ignored by the AP53 study.

### AP54-DIRECT-PROMOTION-CONVERGENCE  Direct-wrapper convergence ladder

- **Status:** stage-recorded (historical; not current capability)
- **Scope:** pre-budget convergence ladders for the AP47/AP49 direct wrapper
  over `N_q`, `N_mu`, and `N_phi`, using the AP51 classified outcome path and
  the AP53 frozen direct-wrapper fallback by default.
- **Files changed:** `src/rabbit/validation/augmented_direct_convergence.py`,
  `scripts/run_augmented_direct_convergence.py`,
  `tests/test_augmented_direct_convergence.py`, WBS/catalog/state docs,
  generated capability/status docs, and registry lock tests.
- **Physics added/changed:** no new kernel.  This PR establishes whether the
  staged direct wrapper has smoke-scale convergence over the three active
  non-LRS direct-wrapper resolution axes before source-budget closure.
- **Scope boundary:** convergence at one ladder point does not close weak-rate,
  source-budget, nonlinear transport, or inference-readiness blockers.  Labels
  remain pre-budget diagnostic until AP55 passes.
- **Exit gate:** reports include adjacent deltas, terminal thermo/network/source
  observables, first-converged settings, and explicit unconverged residual
  risks, all marked pre-budget.
- **Tests run:** `PYTHONPATH=src pytest -q
  tests/test_augmented_direct_convergence.py` passes with `8 passed`.
- **Artifact smoke:** `PYTHONPATH=src venv/bin/python
  scripts/run_augmented_direct_convergence.py --output
  /tmp/rabbit_ap54_direct_convergence_smoke.json --q-values 3,4
  --N-mu-values 3,4 --N-phi-values 5,6 --N-span-end 1e-12
  --relative-tolerance 1 --absolute-tolerance 1` emits converged smoke
  q, `N_mu`, and `N_phi` ladders with `pre_budget_direct_convergence_ap54`.
- **Known red tests:** none introduced in the focused AP54 suite.

### AP55-COLLISION-SOURCE-BUDGET-CLOSURE  Source moment budget audit

- **Status:** stage-recorded (historical; not current capability)
- **Scope:** deterministic AP41/AP47/AP6 collision source-moment budget artifact.
  `src/rabbit/validation/augmented_collision_source_budget.py` evaluates the
  existing no-QKE angular source path over FLRW quiet, LRS electron-pair
  heating, LRS `nu-nu` energy redistribution, non-LRS S2 LRS-limit, non-LRS
  minus-mode, and non-LRS quiet cases, plus the AP6 descriptor-driven LRS
  `pstf_radial` process-source budget case, the LRS charge-neutral
  `pstf_radial` process-source budget case, the non-LRS
  `pstf_radial` process-source budget case, and the non-LRS charge-neutral
  `pstf_radial` process-source budget case;
  `scripts/run_augmented_collision_source_budget.py`
  emits the JSON report.
- **Expected key files:** `src/rabbit/validation/augmented_collision_source_budget.py`,
  `scripts/run_augmented_collision_source_budget.py`,
  `tests/test_augmented_collision_source_budget.py`.
- **Physics added/changed:** closes the first load-bearing collision-feedback
  source-budget blocker by locking finite moments, component summation, source
  contracts, signs, FLRW quietness, non-LRS S2 context, and no-QKE `nu-nu`
  weighted-energy/number residuals in the deterministic angular source path.
  It now also runs the AP6 finite-mass electromagnetic plus staged diagonal
  `nu-nu` `pstf_radial` source through the same report for both the LRS and
  non-LRS S2 bases, evaluates separate LRS and non-LRS charge-neutral rows with
  algebraic finite-mass e-/e+ bath diagnostics from `phase1_to_phase2(Xn0)`,
  and records concrete radial moments, process markers,
  `radial_max_abs_C_mode`, `collision_dA_abs_max`, charge-neutral `mu_e`,
  and non-LRS radial S2 context from the returned kinetic hierarchy payload.
  The AP55 CLI now exposes `--eta` and `--Xn0` so the charge-neutral bath
  context can be reproduced in standalone artifact runs.
- **Scope boundary:** this is still moment-sourced thermo feedback, not a full
  Boltzmann/QKE collision operator.  It does not promote public dispatch,
  nonlinear non-LRS transport, full anisotropic weak rates, a publication plot,
  or SMC evidence.
- **Exit gate:** focused tests prove finite source moments, bounded energy
  residuals, correct source-contract diagnostics, FLRW/LRS/non-LRS limits,
  no sign-inverted heating or cooling, LRS fixed-`mu_e`, LRS charge-neutral,
  non-LRS fixed-`mu_e`, and non-LRS charge-neutral AP6 `pstf_radial` source moments and kinetic `dA` amplitude,
  charge-neutral e-/e+ positivity, non-LRS radial S2 context markers, JSON
  writer output, and CLI summary output with charge-neutral `eta`/`Xn0`
  controls.
- **Verification:** `PYTHONPATH=src pytest -q tests/test_augmented_collision_source_budget.py`
  passed locally.  The non-LRS radial follow-up emitted
  `/tmp/rabbit_ap55_source_budget_nonlrs_radial.json` with `case_count == 8`,
  no limit violations, `n_angles == 15`, `n_radial_moment_sources == 15`,
  `collision_dA_abs_max = 6.718332084185485e-06`, and
  `radial_max_abs_C_mode = 6.596385366105848e-06`.  The charge-neutral radial
  follow-up emitted `/tmp/rabbit_ap55_source_budget_charge_neutral_radial.json`
  with `case_count == 9`, no limit violations, `electron_mu_charge_neutrality_MeV =
  3.2984392994706013e-10`, `collision_dA_abs_max =
  5.1543738900744335e-09`, and `radial_max_abs_C_mode =
  5.061988581870585e-09`.  The non-LRS charge-neutral radial follow-up emitted
  `/tmp/rabbit_ap55_source_budget_nonlrs_charge_neutral_radial.json` with
  `case_count == 10`, no limit violations, `n_angles == 15`,
  `electron_mu_charge_neutrality_MeV = 3.2984392994706013e-10`,
  `collision_dA_abs_max = 6.7183320841851475e-06`, and
  `radial_max_abs_C_mode = 6.596385366111102e-06`.
- **Known red tests:** the AP6 radial budget extension first failed because the
  AP55 report had only 6 angular/S2 cases and no
  `lrs_pstf_radial_process_budget` row.  The non-LRS radial follow-up first
  failed because AP55 still had no `nonlrs_pstf_radial_process_budget` row.
  The charge-neutral radial follow-up first failed because AP55 did not
  preserve a source-budget row for the executable charge-neutral radial bath
  route; the non-LRS charge-neutral follow-up then failed until that same
  charge-neutral budget evidence was evaluated on the S2 radial basis.

### AP56-DIAGNOSTIC-COLLISION-FEEDBACK-CANDIDATE-ARTIFACT  Full-span candidate artifact

- **Status:** stage-recorded (historical; not current capability)
- **Scope:** consolidate AP42/AP49 outputs plus AP51/AP52/AP53/AP54/AP55 summaries into
  one diagnostic candidate evidence bundle for standard versus angular
  direct-wrapper collision-feedback comparisons.  The bundle can consume
  prebuilt artifacts or run the smoke-scale staged builders directly.
- **Expected key files:** `src/rabbit/validation/augmented_candidate_artifact.py`,
  `scripts/run_augmented_collision_feedback_candidate_artifact.py`,
  `tests/test_augmented_candidate_artifact.py`.
- **Physics added/changed:** packages the AP47/AP55 collision-feedback path with
  explicit source routing, solver policy, source-update policy, full observable
  deltas, AP54 convergence status, AP55 source-budget status, the AP55
  `lrs_pstf_radial_process_budget`,
  `lrs_pstf_radial_charge_neutrality_budget`, and
  `nonlrs_pstf_radial_process_budget`, and
  `nonlrs_pstf_radial_charge_neutrality_budget`
  case lists/source contracts/charge-neutral context/S2 context markers/selected radial observables,
  and failure classification while preserving the existing JSON artifact
  contracts.
- **Scope boundary:** still no canonical dispatch, no default collision
  feedback, no nonlinear non-LRS transport, no full anisotropic weak rates,
  no publication plot, no SMC evidence, and no QKE.  The smoke result records
  `diagnostic_fallback_only` source-policy status when `live_rhs` is not
  AP56-eligible.
- **Exit gate:** artifact passes schema, source-routing, bounded observable,
  AP54/AP55 evidence-link, LRS/non-LRS fixed-`mu_e` and charge-neutral AP6 radial budget summaries, and
  stale-artifact tests on smoke-scale configurations; optional longer spans
  remain explicit runner inputs.
- **Verification:** `PYTHONPATH=src pytest -q tests/test_augmented_candidate_artifact.py`
  passed locally; `PYTHONPATH=src venv/bin/python scripts/run_augmented_collision_feedback_candidate_artifact.py --output /tmp/rabbit_ap56_candidate_smoke.json --timeout-s 5`
  emitted a completed AP56 smoke artifact.  The radial-summary follow-up
  emitted `/tmp/rabbit_ap56_candidate_radial_strict.json` with `case_count == 7`,
  `has_lrs_pstf_radial_process_budget == true`, `n_radial_moment_sources ==
  15`, `collision_dA_abs_max = 3.259532061098667e-09`, and
  `radial_max_abs_C_mode = 3.19546213603058e-09`.  The non-LRS radial follow-up
  emitted `/tmp/rabbit_ap56_candidate_nonlrs_radial.json` with
  `case_count == 8`, `has_nonlrs_pstf_radial_process_budget == true`,
  `n_angles == 15`, `collision_dA_abs_max = 4.290513064347994e-06`, and
  `radial_max_abs_C_mode = 4.205592784837029e-06`.  The charge-neutral radial
  follow-up emitted `/tmp/rabbit_ap56_candidate_charge_neutral_radial.json`
  with `case_count == 9`,
  `has_lrs_pstf_radial_charge_neutrality_budget == true`,
  `electron_mu_charge_neutrality_MeV = 3.2984392994706013e-10`,
  `collision_dA_abs_max = 3.2595320610887296e-09`, and
  `radial_max_abs_C_mode = 3.195462136023764e-09`.  The non-LRS
  charge-neutral radial follow-up emitted
  `/tmp/rabbit_ap56_candidate_nonlrs_charge_neutral_radial.json` with
  `case_count == 10`,
  `has_nonlrs_pstf_radial_charge_neutrality_budget == true`,
  `n_angles == 15`, `electron_mu_charge_neutrality_MeV =
  3.2984392994706013e-10`, `collision_dA_abs_max =
  4.290513064334856e-06`, and `radial_max_abs_C_mode =
  4.205592784828058e-06`.
- **Known red tests:** the radial-summary follow-up first failed because AP56
  only retained AP55 `case_count` and did not expose the
  `lrs_pstf_radial_process_budget` row.  The self-review RED then failed
  because AP56 still trusted the AP55 case `passed` flag without requiring
  nonzero radial `C_modes`/`dA_modes` and finite-mass process markers in the
  summary.  The non-LRS radial follow-up first failed because AP56 had no
  `nonlrs_pstf_radial_process_budget` summary row.  The charge-neutral radial
  follow-up first failed because AP56 did not require the AP55
  `lrs_pstf_radial_charge_neutrality_budget` row, charge-neutral marker, and
  positive e-/e+ diagnostics.  The non-LRS charge-neutral follow-up first
  failed because AP56 did not require the AP55
  `nonlrs_pstf_radial_charge_neutrality_budget` row, S2 marker,
  charge-neutral marker, and positive e-/e+ diagnostics.

### AP57-COLLISION-THERMO-SANITY-MATRIX  Physical sanity matrix

- **Status:** stage-recorded (historical; not current capability)
- **Scope:** deterministic physical sanity matrix over the AP56 candidate
  evidence bundle plus AP55 source-budget evidence.
- **Expected key files:** `src/rabbit/validation/augmented_physics_gates.py`,
  `scripts/run_augmented_collision_thermo_sanity_matrix.py`,
  `tests/test_augmented_collision_thermo_sanity.py`.
- **Physics added/changed:** no new kernel.  The PR turns known-limit physics
  behavior into a fail-closed gate: FLRW quietness, LRS reduction,
  source-policy boundary preservation, plus/minus bounded response,
  thermo/network bounds, and AP6 radial source-budget markers/amplitudes from
  the AP55 `lrs_pstf_radial_process_budget`,
  `lrs_pstf_radial_charge_neutrality_budget`, and
  `nonlrs_pstf_radial_process_budget`, and
  `nonlrs_pstf_radial_charge_neutrality_budget` cases.
- **Scope boundary:** passing this matrix supports bounded diagnostic
  collision-feedback evidence only; it does not establish inference readiness,
  public dispatch, nonlinear transport, full anisotropic weak rates,
  publication plots, SMC, or QKE.
- **Exit gate:** gate fails on unphysical temperature, Hubble, abundance,
  source-moment, missing LRS/non-LRS radial source markers, missing finite-mass
  radial process markers, missing charge-neutral e-/e+ bath diagnostics,
  missing non-LRS radial S2 context, zero radial
  `C_modes`/`dA_modes`, or symmetry behavior and passes the deterministic
  sanity cases.
- **Verification:** `PYTHONPATH=src pytest -q tests/test_augmented_candidate_artifact.py tests/test_augmented_collision_thermo_sanity.py`
  passed locally; `PYTHONPATH=src venv/bin/python scripts/run_augmented_collision_thermo_sanity_matrix.py --output /tmp/rabbit_ap57_sanity_smoke.json`
  emitted a passing AP57 smoke artifact.  The radial sanity follow-up
  emitted `/tmp/rabbit_ap57_sanity_radial_strict.json` with `case_count == 8`, no
  limit violations, `n_radial_moment_sources == 15`,
  `n_radial_moment_processes == 4`, `collision_dA_abs_max =
  5.154373890075132e-09`, and `radial_max_abs_C_mode =
  5.061988581867105e-09`.  The non-LRS radial follow-up emitted
  `/tmp/rabbit_ap57_sanity_nonlrs_radial.json` with `case_count == 9`, no limit
  violations, `n_angles == 15`, `collision_dA_abs_max =
  6.718332084185485e-06`, and `radial_max_abs_C_mode =
  6.596385366105848e-06`.  The charge-neutral radial follow-up emitted
  `/tmp/rabbit_ap57_sanity_charge_neutral_radial.json` with `case_count ==
  10`, no limit violations, `electron_mu_charge_neutrality_MeV =
  3.2984392994706013e-10`, `collision_dA_abs_max =
  5.1543738900744335e-09`, and `radial_max_abs_C_mode =
  5.061988581870585e-09`.  The non-LRS charge-neutral radial follow-up emitted
  `/tmp/rabbit_ap57_sanity_nonlrs_charge_neutral_radial.json` with
  `case_count == 11`, no limit violations, `n_angles == 15`,
  `electron_mu_charge_neutrality_MeV = 3.2984392994706013e-10`,
  `collision_dA_abs_max = 6.7183320841851475e-06`, and
  `radial_max_abs_C_mode = 6.596385366111102e-06`.
- **Known red tests:** the radial sanity follow-up first failed because AP57
  still emitted seven cases and had no `lrs_pstf_radial_source_budget` sanity
  row.  The non-LRS radial follow-up first failed because AP57 still had no
  `nonlrs_pstf_radial_source_budget` sanity row.  The charge-neutral radial
  follow-up first failed because AP57 did not turn the AP55 charge-neutral
  radial row into a fail-closed sanity case.  The non-LRS charge-neutral
  follow-up first failed because AP57 did not turn the AP55 S2 charge-neutral
  radial row into a fail-closed sanity case.

### AP58-WEAK-RATE-ANGULAR-DATA-MODEL  Existing angular weak-rate model upgrade

- **Status:** stage-recorded (historical; not current capability)
- **Scope:** extend the existing `AugmentedWeakInputs` and
  `WeakAngularCorrectionMetadata` contract beyond the former metadata-only
  non-LRS weak-rate angular plumbing with explicit angular moment inputs,
  approximation metadata, and fail-closed unsupported-mode behavior.
- **Expected key files:** `src/rabbit/weak/augmented_bridge.py`,
  `src/rabbit/weak/augmented_angular_rates.py`,
  `src/rabbit/transport/augmented_typeI_weak_network.py`,
  `tests/test_augmented_angular_rate_inputs.py`.
- **Physics added/changed:** creates the data contract needed for live
  anisotropic weak-rate integration in LRS and non-LRS cases by upgrading the
  landed weak bridge rather than adding a parallel input model.  `AugmentedWeakInputs`
  now carries `AngularWeakRateMomentInputs` with per-q plus moments for LRS and
  plus/minus moments for the staged non-LRS S2 basis, with approximation/source
  labels copied into `WeakAngularCorrectionMetadata`.
- **Scope boundary:** input-model upgrade only; live rate application lands in
  AP59/AP60.  The fail-closed mode resolver now reports those AP59/AP60 modes
  as application-ready only when the required current angular moments are
  supplied.
- **Exit gate:** tests show the existing rate input objects carry plus/minus
  angular data, reject missing species/moments, preserve approximation labels,
  report application-ready mode resolution for valid AP59/AP60 modes, and never
  silently fall back to temperature-compressed distributions.
- **Verification:** `PYTHONPATH=src pytest -q tests/test_augmented_angular_rate_inputs.py tests/test_augmented_weak_bridge.py tests/test_augmented_typeI_weak_network_bridge.py tests/test_augmented_typeI_nonlrs_weak_network_solve.py`
  passed locally.
- **Known red tests:** none introduced in the focused AP58 suite.

### AP59-LRS-ANISOTROPIC-WEAK-RATES  LRS live angular rates

- **Status:** stage-recorded (historical; not current capability)
- **Scope:** wire AP58 into an opt-in LRS live anisotropic weak-rate functional
  beyond the bounded `Sigma_+ K_2` multiplier.
- **Expected key files:** `src/rabbit/weak/augmented_angular_rates.py`,
  `src/rabbit/transport/augmented_typeI_weak_network.py`,
  `tests/test_augmented_angular_rate_inputs.py`.
- **Physics added/changed:** weak rates become sensitive to current LRS angular
  distribution moments.  The opt-in `lrs_cl3_quadrupole_input` mode computes
  species-aware `lambda_np` and `lambda_pn` factors from AP58 per-q LRS plus
  moments, with the existing multiplier preserved as the default bounded
  `legacy_sigma_plus_k2_multiplier` approximation.
- **Scope boundary:** LRS only; no non-LRS plus/minus weak-rate promotion until
  AP60.
- **Exit gate:** Born/CL3 rate tests cover distribution sensitivity,
  FD/isotropic recovery, bounded correction metadata, and agreement with the
  existing multiplier where intentionally selected; mode resolution reports
  `applied_lrs_cl3_moment_input_quadrupole_multiplier_v1` when valid LRS
  moment inputs are supplied.
- **Verification:** `PYTHONPATH=src pytest -q tests/test_augmented_angular_rate_inputs.py tests/test_augmented_typeI_weak_network_bridge.py tests/test_typeI_anisotropic_weak_kernel.py`
  passed locally.
- **Known red tests:** none introduced in the focused AP59 suite.

### AP60-NLRS-ANISOTROPIC-WEAK-RATES  Non-LRS live angular rates

- **Status:** stage-recorded (historical; not current capability)
- **Scope:** extend AP59 to the staged S2 plus/minus basis so weak rates consume
  current non-LRS angular moment data rather than metadata-only tags.
- **Expected key files:** `src/rabbit/weak/augmented_angular_rates.py`,
  `src/rabbit/transport/augmented_typeI_weak_network.py`,
  `tests/test_augmented_angular_rate_inputs.py`.
- **Physics added/changed:** adds plus/minus angular sensitivity, `Sigma_-`
  dependence, and LRS reduction behavior to live weak-rate correction paths.
  The `nonlrs_s2_cl3_quadrupole_input` mode computes species-aware `lambda_np`
  and `lambda_pn` factors from the AP58 S2 plus/minus moment profiles.  The
  staged non-LRS correction-level-3 weak/network config now applies that current
  S2 moment-input mode by default, with explicit `metadata_only` retained for
  same-CL3 controls and baselines.
- **Scope boundary:** remains staged and SciPy-first; unsupported correction
  levels and incompatible LRS/non-LRS mode routing must fail closed.  This is
  not public dispatch and not promotion-grade full anisotropic weak-rate
  convergence.
- **Exit gate:** tests lock plus/minus sensitivity, `Sigma_-=0` LRS reduction,
  species validation, finite rates, unsupported-mode rejection, and mode
  resolution reporting
  `applied_nonlrs_s2_cl3_moment_input_quadrupole_multiplier_v1` when valid S2
  plus/minus moment inputs are supplied.
- **Verification:** `PYTHONPATH=src pytest -q tests/test_augmented_angular_rate_inputs.py tests/test_augmented_typeI_nonlrs_weak_network_solve.py`
  passed locally.
- **Known red tests:** none introduced in the focused AP60 suite.

### AP61-WEAK-RATE-CANDIDATE-GATE  Angular weak-rate convergence gate

- **Status:** stage-recorded (historical; not current capability)
- **Scope:** sweep q and S2 angular smoke ladders for the AP59/AP60 opt-in
  LRS/non-LRS moment-input rate corrections and enforce physical sign/range
  limits before coupling to full-span solves.
- **Key files:** `src/rabbit/validation/augmented_weak_rate_gates.py`,
  `scripts/run_augmented_weak_rate_gate.py`,
  `tests/test_augmented_weak_rate_gate.py`.
- **Physics added/changed:** no new rate formula.  The gate decides whether
  live angular weak-rate factors are numerically stable enough to enter later
  AP65-style coupled candidate solves.
- **Scope boundary:** a passing rate-only gate is not a full BBN solve claim.
- **Exit gate:** bounded LRS/non-LRS cases pass; excessive
  `lambda_np`/`lambda_pn` deltas, nonfinite rates, or unsupported angular modes
  fail with explicit diagnostics.
- **Verification:** `PYTHONPATH=src pytest -q tests/test_augmented_weak_rate_gate.py`
  passed locally; `python scripts/run_augmented_weak_rate_gate.py --output /tmp/ap61_weak_rate_gate.json`
  emitted a six-case passing JSON summary.
- **Known red tests:** none introduced in the focused AP61 suite.

### AP62-NLRS-NONLINEAR-TRANSPORT-SDD-RHS  Nonlinear transport operator

- **Status:** stage-recorded (historical; not current capability)
- **Scope:** SDD plus first RHS implementation for missing nonlinear angular
  advection/streaming terms in the S2 non-LRS basis, using the existing LRS
  nonlinear discrete-ordinate transport module as an oracle/reference while
  preserving the AP37 source-only shell.
- **Key files:** `src/rabbit/transport/augmented_nonlrs_transport.py`,
  `src/rabbit/jax/nonlinear_transport.py`,
  `tests/test_augmented_nonlrs_nonlinear_transport.py`.
- **Physics added/changed:** starts closing the source-only AP37 blocker by
  adding q, mu, and periodic phi nonlinear collisionless transport terms
  projected back into the staged `{monopole, W_+, W_-}` basis, without
  weak/network/collision coupling.
- **Scope boundary:** AP37 source-only transport remains available and clearly
  labeled; this PR does not run full 3T BBN.
- **Exit gate:** unit tests lock agreement with `nonlinear_transport.py` in the
  LRS limit, zero-shear/FLRW quietness, `Sigma_-=0` reduction, plus/minus
  coupling signs, and metadata distinguishing nonlinear transport from
  source-only transport.
- **Verification:** `PYTHONPATH=src pytest -q tests/test_augmented_nonlrs_nonlinear_transport.py`
  passed locally.
- **Known red tests:** none introduced in the focused AP62 suite.

### AP63-NLRS-NONLINEAR-TRANSPORT-SOLVE-SHELL  Nonlinear solve shell

- **Status:** stage-recorded (historical; not current capability)
- **Scope:** integrate the AP62 operator in a SciPy-first solve shell without
  weak/network/collision feedback, with live shear-stress feedback and
  short-span finite trajectory checks.
- **Key files:** `src/rabbit/transport/augmented_nonlrs_transport.py`,
  `tests/test_augmented_nonlrs_nonlinear_transport.py`.
- **Physics added/changed:** makes nonlinear non-LRS transport executable with
  stress diagnostics and controlled LSODA/Radau solve metadata.
- **Scope boundary:** no 3T thermo/Hubble, no live weak rates, no collision
  feedback, no public dispatch.
- **Exit gate:** finite trajectories, stress feedback, reductions, failure
  classification, and conservation/sanity checks pass on short spans.
- **Verification:** `PYTHONPATH=src pytest -q tests/test_augmented_nonlrs_nonlinear_transport.py`
  passed locally.
- **Known red tests:** none introduced in the focused AP63 suite.

### AP64-NLRS-NONLINEAR-3T-WEAK-NETWORK  Nonlinear transport with 3T weak/network coupling

- **Status:** stage-recorded (historical; not current capability)
- **Scope:** add dynamic 3T thermo/Hubble and live weak/network coupling to the
  AP63 nonlinear non-LRS transport option under an explicit candidate flag.
- **Key files:** `src/rabbit/transport/augmented_typeI_weak_network.py`,
  `src/rabbit/transport/augmented_nonlrs_transport.py`,
  `tests/test_augmented_typeI_nonlrs_nonlinear_weak_network_3t.py`.
- **Physics added/changed:** replaces the source-only AP39 transport block with
  a nonlinear-transport candidate in the 3T weak/network solve path, while
  reusing the AP60 non-LRS moment-input weak-rate mode and existing PRIMAT
  abundance RHS.
- **Scope boundary:** no angular collision feedback or live anisotropic weak
  rates unless explicitly enabled by the staged AP60 mode; no collision-sourced
  thermo feedback, public dispatch, or QKE.
- **Exit gate:** tests lock dynamic Hubble, live rates, abundance
  normalization, plus/minus stress, solver metadata, and source-only versus
  nonlinear routing.
- **Verification:** `PYTHONPATH=src pytest -q tests/test_augmented_typeI_nonlrs_nonlinear_weak_network_3t.py`
  passed locally.
- **Known red tests:** none introduced in the focused AP64 suite.

### AP65-COLLISION-COUPLED-NLRS-CANDIDATE  Combined nonlinear no-QKE candidate

- **Status:** stage-recorded (historical; not current capability)
- **Scope:** combine the AP41/AP47 angular collision source, AP60 live angular
  weak rates, and AP64 nonlinear non-LRS 3T transport under one opt-in
  candidate solve path.
- **Key files:** `src/rabbit/transport/augmented_typeI_weak_network.py`,
  `tests/test_augmented_typeI_nonlrs_nonlinear_collision_feedback_3t.py`.
- **Physics added/changed:** first integrated no-QKE candidate containing the
  major staged physics pieces needed before publication-scale convergence:
  nonlinear S2 transport, live weak/network, dynamic 3T/Hubble, and explicit
  angular collision-moment thermo feedback.  The angular source now also
  propagates its projected `dA_modes` collision term into the AP64 nonlinear
  hierarchy RHS, so the candidate is no longer thermo-only for the angular
  collision feedback path.
- **Scope boundary:** still candidate-only, SciPy-first, no QKE, and no public
  canonical dispatch; promotion-grade span/convergence gates remain AP66+.
- **Exit gate:** focused tests pass with finite thermo/network/source
  observables, nonlinear-grid wrapper wiring, live-source update behavior,
  kinetic `dA_modes` RHS application in the nonlinear shell, and unsupported
  source-update policies failing closed.
- **Verification:** `PYTHONPATH=src pytest -q tests/test_augmented_typeI_nonlrs_nonlinear_collision_feedback_3t.py`
  passed locally.
- **Known red tests:** none introduced in the focused AP65 suite.

### AP65-PSTF-RADIAL-NONLINEAR-WRAPPER  Nonlinear radial collision 3T solve

- **Status:** stage-recorded (historical; not current capability)
- **Scope:** route the AP6 descriptor-driven `pstf_radial` collision source
  through the AP65 nonlinear non-LRS 3T shell, using the same explicit
  collision-moment thermo-feedback callback contract as the angular AP65
  wrapper.
- **Key files:** `src/rabbit/transport/augmented_typeI_weak_network.py`,
  `src/rabbit/transport/__init__.py`,
  `tests/test_augmented_typeI_nonlrs_nonlinear_collision_feedback_3t.py`.
- **Physics added/changed:** `run_augmented_nonlrs_nonlinear_pstf_radial_collision_weak_network_3T_solve(...)`
  builds the finite-mass electromagnetic plus staged UR diagonal-`nu-nu`
  radial source on the nonlinear S2 grid, initializes the nonlinear transport
  monopoles with FD occupations by default, forwards fixed or charge-neutral
  electron-chemical-potential modes, records `pstf_radial` live-RHS
  source-evaluation budget diagnostics, and receives the AP6 radial
  `C_modes -> dA_modes` hierarchy payload through the shared AP65 collision
  source callback.
- **Smoke evidence:** `N_q=3`, `N_mu=3`, `N_phi=5`, `N_span=(0, 1e-14)`,
  `method=RK23` returned `nfev = 5`, `dQ_nue_pair_N =
  1.7901492532165454e-3`, `dQ_nux_bank_N = 1.050609070529733e-3`,
  `radial_max_abs_C_mode = 4.04099948362707e-6`, and
  `pstf_radial_source_evaluations = 7` under the default budget of 64.
- **Scope boundary:** this is still opt-in diagnostic staging.  It does not
  make `pstf_radial` default/public dispatch, does not promote full-span
  live-RHS radial collision coupling, does not add an independent evolved
  electron charge-asymmetry state, and does not change the no-QKE boundary.
- **Verification:** `PYTHONPATH=src pytest -q tests/test_augmented_typeI_nonlrs_nonlinear_collision_feedback_3t.py -k pstf_radial`
  passed locally.

### AP65-COMBINED-ANGULAR-PSTF-RADIAL-NONLINEAR-WRAPPER  Combined collision term

- **Status:** stage-recorded (historical; not current capability)
- **Scope:** compose the AP41 angular source and AP6 descriptor-driven
  `pstf_radial` source into one AP65 nonlinear non-LRS 3T RHS callback.
- **Key files:** `src/rabbit/transport/augmented_typeI_weak_network.py`,
  `src/rabbit/validation/augmented_convergence.py`,
  `src/rabbit/validation/augmented_stability.py`,
  `scripts/run_augmented_nonlrs_nonlinear_combined_collision_3t_solve_artifact.py`,
  `scripts/run_augmented_nonlrs_combined_full_span_3t_candidate_gate.py`,
  `tests/test_augmented_typeI_nonlrs_nonlinear_collision_feedback_3t.py`, and
  `tests/test_augmented_convergence.py`, and
  `tests/test_augmented_stability_envelope.py`.
- **Physics added/changed:** `run_augmented_nonlrs_nonlinear_combined_collision_weak_network_3T_solve(...)`
  builds both live no-QKE collision sources on the same nonlinear S2 grid,
  records AP41 angular diagnostics, and forwards the AP6 finite-mass
  `pstf_radial` source as the effective thermo and kinetic RHS payload.  This
  v2 composition prevents double-counting the same no-QKE collision family:
  the angular source remains a diagnostic comparator, while
  `dQ_nue_pair_N`, `dQ_nux_bank_N`, and the hierarchy `dA_modes` are taken
  from the radial source.  The radial component keeps the explicit live-RHS
  source-evaluation budget, optional `standard_3t_plasma` energy closure, and forwarded
  `unit_direction_gaussian`/`radial_gaussian` momentum-delta controls.  The AP4/AP65
  follow-up candidate gate now consumes that combined artifact over explicit
  span ladders, records terminal shear/stress/3T/network/source values,
  component angular/radial `dA` amplitudes, radial source-evaluation budgets,
  radial momentum-delta provenance, and writes a JSON artifact/CLI summary
  without promoting public dispatch.
- **Smoke evidence:** the AP65 combined-source CLI at `N_q=3`, `N_mu=3`,
  `N_phi=5`, `N_span=(0, 1e-14)`, `source_update_policy=live_rhs`,
  `pstf_radial_energy_normalization=standard_3t_plasma`, and `method=RK23`
  returned `success=true`, `nfev=5`,
  `source_contract=combined_nonlrs_angular_pstf_radial_collision_thermo_source_v2`,
  `collision_dQ_nue_pair_N_final=1.0831195293549981e-4`,
  `collision_dQ_nux_bank_N_final=2.8944175873527294e-4`,
  `collision_dA_abs_max_final=3.3626741471031303e-4`,
  `combined_angular_dA_abs_max=3.5318831933493763e-22`,
  `combined_pstf_radial_dA_abs_max=3.3626741471031303e-4`, and
  `pstf_radial_source_evaluations=7`.  The artifact diagnostics include
  `combined_collision_no_double_count_v1=1`,
  `combined_angular_source_diagnostic_only=1`, and
  `combined_effective_source_pstf_radial=1`.
  The AP4/AP65 full-span candidate gate smoke at `N_span=(0, 1e-14)`,
  `source_update_policy=frozen_initial_state`,
  `pstf_radial_energy_normalization=standard_3t_plasma`, and `method=RK23`
  returned `success=true`, `T_gamma_final=0.7999999999999922`,
  `H_rate_s_final=0.4315487123652324`, `Xn_final=0.13000000000065723`,
  `collision_dA_abs_max_final=2.8419082353054516e-4`,
  `combined_pstf_radial_dA_abs_max_final=2.8419082353054516e-4`,
  `source_evaluations=1`, and `nfev=5`.
  The real `live_rhs` two-span candidate gate over `N_span=(0, 1e-14)` and
  `(0, 1e-12)` with `max_pstf_radial_source_evaluations=256` also passed;
  both rows recorded `source_evaluations=7`, present/passing source-budget
  diagnostics, and the longer row recorded `T_gamma_final=0.7999999999992143`,
  `H_rate_s_final=0.43154871236438125`, `Xn_final=0.13000002470684513`,
  and `collision_dA_abs_max_final=2.841908235303566e-4`.
  AP4/AP65 now also exposes a deterministic `--preset warm` ladder using
  `live_rhs`, `max_pstf_radial_source_evaluations=2048`, and
  `max_nfev=50000` over `N_span=(0, 1e-12)`, `(0, 1e-10)`, and
  `(0, 1e-8)`.  A real warm-preset CLI run passed with `span_count=3`,
  `max_span_length=1e-8`, `source_evaluation_max=70`,
  `collision_dA_abs_max_final=3.362674147101354e-4`, and no violations.
  The companion combined-source source-policy span profile over the same two
  spans passed all four frozen/live rows with frozen/live `nfev=5/5`,
  frozen/live source evaluations `1/7`, `failure_rows=0`, maximum
  `collision_dA_abs_max_final=2.8419082353054516e-4`, and roundoff-scale
  live-minus-frozen thermo/network deltas.
- **Scope boundary:** still opt-in diagnostic staging.  This does not make the
  combined source default, does not promote public dispatch or production SMC,
  does not remove source-evaluation budgets, and keeps QKE out of scope.
- **Exit gate:** focused tests lock no-double-count effective source selection,
  component diagnostics, radial `dA_modes` payloads, radial momentum-delta forwarding, real
  `radial_gaussian` smoke output, JSON writer, CLI summary,
  and the AP4/AP65 full-span candidate gate artifact/script surface including
  real `live_rhs` budget-diagnostic evidence plus the frozen/live
  source-policy span-profile artifact/CLI.
- **Verification:** `PYTHONPATH=src pytest -q tests/test_augmented_typeI_nonlrs_nonlinear_collision_feedback_3t.py -k "combined_collision" tests/test_augmented_convergence.py -k "nonlinear_combined_collision"`
  and `PYTHONPATH=src pytest -q tests/test_augmented_stability_envelope.py -k "combined_full_span"`
  plus the smoke CLI command above passed locally.
- **Known red tests:** focused tests first failed because the combined wrapper
  and artifact builder were not exported; the AP4/AP65 gate writer tests first
  failed because the candidate-gate artifact builder and CLI did not exist.

### AP66-PUBLICATION-CONVERGENCE-MATRIX  Publication-tolerance matrix

- **Status:** stage-recorded (historical; not current capability)
- **Scope:** run AP65 over `N_q`, `N_mu`, `N_phi`, span, solver tolerance,
  source model, source policy, weak-rate correction, scalar QED model,
  radial momentum-delta, and fixed/charge-neutral electron-bath control ladders with cached artifacts, reusing the existing
  augmented convergence report contracts.
- **Key files:** `src/rabbit/validation/augmented_publication_matrix.py`,
  `scripts/run_augmented_publication_convergence_matrix.py`,
  `tests/test_augmented_publication_convergence_matrix.py`.
- **Physics added/changed:** no new kernel; AP66 promotes the AP65 coupled
  candidate sources into a deterministic convergence-evidence surface.  The matrix
  now selects the `angular` AP65 wrapper, the direct combined
  `combined_angular_pstf_radial` AP65 wrapper, or the AP4/AP65
  `piecewise_frozen` combined-source gate per row, forwards radial
  energy-normalization, source-evaluation-budget, and momentum-delta controls for combined rows,
  records terminal AP65 thermo/network/source observables, outcome flags,
  source-model rows, source-policy rows, weak-rate-mode rows, radial momentum-delta rows, the final
  collision-source `dA_modes` amplitude as `collision_dA_abs_max_final`,
  component angular/radial `dA` amplitudes, and radial source-budget observables plus momentum-delta controls/provenance
  while reusing `ResolutionConvergenceReport`.  Charge-neutral AP66 rows now
  forward the evolved electron-bath state through AP65 and record terminal
  `electron_chemical_potential_MeV_final`,
  `electron_charge_asymmetry_density_MeV3_final`, evolved-state, and
  thermo-feedback correction observables.  AP66 rows also forward
  `qed_correction_model` into AP65 and record finite-mu-scaled versus
  exact-scalar-QED row markers.  Piecewise AP66 rows are allowed only for
  `combined_angular_pstf_radial`, forward explicit
  `source_update_subspan_ends`, and record `source_policy_piecewise_frozen`,
  `source_update_subspan_count`, `source_update_subspan_max_length`,
  `source_update_charge_asymmetry_state_handoff`, and
  `source_diagnostic_evaluations`; the AP4 gate spec now forwards `Xn0` and
  `weak_rate_mode` plus radial momentum-delta model/sigma controls into the
  underlying solve rows used by this matrix path.
- **Scope boundary:** this is SciPy-first, smoke-scale by default, opt-in for
  longer span/tolerance rows, no QKE, and no public canonical dispatch.
  Failure to converge keeps later plot/SMC work diagnostic-only.
- **Exit gate:** matrix identifies first-converged publication-candidate
  settings, including source model, scalar QED model, radial momentum-delta, and electron-bath controls, and records
  residual risks where convergence is not achieved.
- **Verification:** `PYTHONPATH=src pytest -q tests/test_augmented_publication_convergence_matrix.py`
  passed locally; a short-span combined-source matrix with `source_model =
  combined_angular_pstf_radial`, `source_update_policy = live_rhs`,
  `weak_rate_mode = nonlrs_s2_cl3_quadrupole_input`, and
  `standard_3t_plasma` converged `N_q=(3,4)`, `N_mu=(3,4)`, and
  `N_phi=(5,6)` with candidate `(4,4,6)`.  The charge-neutral variant of the
  same smoke matrix reported `electron_chemical_potential_MeV_final =
  3.298439955909307e-10`,
  `electron_charge_asymmetry_density_MeV3_final =
  6.61872482662131e-11`, and evolved-state marker `1.0` in AP66 row
  observables.  A real piecewise q-ladder smoke with
  `source_update_policy=piecewise_frozen`,
  `source_update_subspan_ends=(5e-15,1e-14)`, `metadata_only`, `RK23`, and
  `standard_3t_plasma` passed, and a full short-span piecewise AP66 matrix
  over q/`N_mu`/`N_phi` produced candidate `(4,4,6)` with
  `source_update_subspan_count=2`, `pstf_radial_source_evaluations=2`,
  `collision_dA_abs_max_final=0.0010936512219095985`,
  `T_gamma_final=0.7999999999999923`, and
  `Xn_final=0.13000000000019393`.  Focused exact-scalar-QED AP66 routing
  tests reported `18 passed` before the later piecewise extension.
- **Known red tests:** the kinetic-source observable regression first failed
  because AP66 only exposed thermo `dQ` source amplitudes.  The electron-bath
  regression first failed because AP66 did not accept or forward
  `electron_chemical_potential_mode`.  The radial-Gaussian forwarding
  regression first failed because AP66 had no momentum-delta model/sigma
  surface.  The exact scalar QED regression first failed because AP66 did not
  expose `qed_correction_model` or row-level QED model markers.  The piecewise
  regression first failed because AP66 had no AP4 full-span gate import,
  rejected `piecewise_frozen`, and the CLI did not accept subspan ends.

### AP67-PHYSICS-VALIDATION-ATLAS  Known-limit benchmark atlas

- **Status:** stage-recorded (historical; not current capability)
- **Scope:** deterministic FLRW, LRS, `Sigma_-=0`, small non-LRS, and finite
  non-LRS benchmark atlas for AP65, now with explicit source-model coverage
  for angular-only and combined angular+`pstf_radial` rows, reusing PRIMAT
  parity, cross-code, convergence, and QMC/control-variate hooks where
  applicable.
- **Key files:** `src/rabbit/validation/augmented_validation_atlas.py`,
  `scripts/run_augmented_validation_atlas.py`,
  `tests/test_augmented_validation_atlas.py`.
- **Physics added/changed:** converts known-limit reductions, null cases,
  injections, stress tests, and reused parity/QMC evidence links into versioned
  validation artifacts.  AP67 does not add a new kernel; it organizes AP65
  candidate solves and AP66 convergence evidence into pass/fail atlas rows,
  preserves source-model provenance, forwards radial source controls to
  combined rows, forwards radial momentum-delta controls into AP65 case solves
  and nested AP66 evidence, forwards fixed/charge-neutral electron-bath controls
  into AP65 case solves and nested AP66 evidence, forwards scalar-QED model
  selection including `exact_finite_mu_scalar` into AP65 case solves and nested
  AP66 evidence, records terminal electron-mu/charge-asymmetry observables and
  scalar-QED markers for atlas rows, uses same-CL3
  weak-rate configs for atlas rows, preserves AP66 `source_update_policy`,
  `source_update_policies`, and `source_update_subspan_ends` evidence-link
  provenance for AP4-backed `piecewise_frozen` rows, and records fail-closed
  radial source-budget cases rather than aborting the artifact.
- **Scope boundary:** atlas artifacts distinguish output-ready, convergence-
  ready, and inference-ready cases rather than collapsing them into one claim;
  AP68/AP69 consume the atlas through guarded inference and likelihood schema
  adapters, AP70 adds smoke tempered-SMC plumbing, AP71 adds runtime/cache
  controls, AP72 adds synthetic SMC validation, AP73 adds figure-ready
  artifact tables, AP74 renders diagnostic publication plots, AP75 packages
  reproducibility evidence, and public dispatch plus production SMC evidence
  remain blocked after the AP76 `not_promoted` readiness audit.
- **Exit gate:** JSON artifacts include known-limit assertions, expected
  reductions, reused evidence links, pass/fail labels, explicit residual
  blockers, source-model labels, AP66 candidate source-model, AP66
  source-update policy/subspan provenance, radial momentum-delta, and
  scalar-QED links, terminal electron-bath observables, and separate
  output-ready/convergence-ready/inference-ready labels.
- **Verification:** `PYTHONPATH=src pytest -q tests/test_augmented_validation_atlas.py`
  passed locally, including a real short-span combined-source single-case smoke.
  A real AP67 combined-source charge-neutral atlas CLI over the six default
  cases with `standard_3t_plasma`, `radial_gaussian`, and
  `--skip-convergence-matrix` reported
  `limit_violations=[]`, `output_ready=true`,
  `electron_chemical_potential_MeV_final=3.2984399559038986e-10`,
  `electron_charge_asymmetry_density_MeV3_final=6.618724826601604e-11`,
  and `electron_charge_asymmetry_state_evolved=1.0`.
  The scalar-QED follow-up focused targets
  `PYTHONPATH=src pytest -q tests/test_augmented_validation_atlas.py::test_validation_atlas_forwards_exact_scalar_qed_to_cases_and_ap66 tests/test_augmented_publication_artifacts.py::test_augmented_publication_artifacts_build_figure_ready_tables`
  and `PYTHONPATH=src pytest -q tests/test_augmented_validation_atlas.py tests/test_augmented_publication_artifacts.py`
  passed after AP67 accepted `qed_correction_model=exact_finite_mu_scalar`,
  forwarded it to AP65/AP66, and AP73 preserved AP67 scalar-QED rows.  A real
  single-case combined-source AP67 smoke with `exact_finite_mu_scalar`,
  charge-neutral electron bath, `radial_gaussian`, and `standard_3t_plasma`
  reported `status=pass`, `limit_violations=[]`, exact-QED marker `1.0`,
  `collision_dA_abs_max_final=7.360025983446734e-05`, and
  `pstf_radial_source_budget_passed=1.0`.
  AP67 source-refresh follow-up tests now lock nested AP66
  `piecewise_frozen` policy/subspan forwarding through
  `convergence_matrix_kwargs` and preserve those values under
  `reused_evidence_links["AP66"]`.
- **Known red tests:** focused AP67 tests first failed because the atlas had no
  combined AP65 wrapper import, no `source_model` build surface, no
  fail-closed radial budget handling, and later no
  `electron_chemical_potential_mode` dataclass/build/CLI surface, and radial
  Gaussian forwarding tests later failed because AP67 had no momentum-delta
  dataclass/build/CLI surface.  The scalar-QED follow-up first failed because
  AP67 had no `qed_correction_model` build/dataclass/CLI surface and AP73
  validation-atlas rows dropped `qed_correction_model`.  The AP66
  source-refresh provenance regression first failed because AP67 accepted
  nested AP66 `piecewise_frozen` kwargs but did not retain AP66
  source-update policy/subspan fields in reused evidence links, and the AP67
  CLI rejected `--source-update-policy`/`--source-update-subspan-ends`.

### AP68-CANDIDATE-FORWARD-MODEL-API  Guarded inference-facing API

- **Status:** stage-recorded (historical; not current capability)
- **Scope:** expose a guarded augmented no-QKE candidate forward-model adapter
  callable by publication plotting and SMC code without canonical/public default
  dispatch.
- **Key files:** `src/rabbit/inference/augmented_forward_model.py`,
  `tests/test_augmented_candidate_forward_model.py`.
- **Physics added/changed:** no new collision kernel.  This PR stabilizes the
  API contract, deterministic configuration, source-model selection, and
  failure metadata by wrapping the AP65 candidate solve in the existing
  `ForwardModel`/`BBNLikelihood` style rather than inventing a second inference
  interface.  The guarded config can now select either the angular-only AP65
  wrapper or the combined angular+`pstf_radial` AP65 wrapper, forwarding radial
  energy-normalization, source-budget controls, radial momentum-delta controls,
  fixed/charge-neutral electron-bath controls, and scalar-QED model selection
  including `exact_finite_mu_scalar` into the actual solve path.  The guarded
  adapter now also accepts `source_update_policy="piecewise_frozen"` for the
  combined source model, partitions the AP68 span by explicit
  `source_update_subspan_ends`, runs each subspan as a frozen-source AP65 solve,
  and hands Sigma/A/T/X plus charge-neutral electron state into the next
  subspan.  FB-12 extends this same guarded adapter with
  `execution_mode="full_chain"`: AP68 can now build and run the FB-04 chained
  full-BBN runner from the inference parameter/config surface, or consume a
  cached chained artifact, then validate passed/no-QKE/not-public/not-production-SMC
  artifact scope before returning a `BBNPrediction`.  The full-chain surface can
  also build or consume the FB-21 live-source repeated-run gate as optional
  diagnostic evidence, preserving the gate contract/path/pass status,
  collision-payload counts/provenance fingerprints, finite BBN readouts, and live-source-vs-
  piecewise/window-map deltas without changing the terminal AP68 `Yp`/`D/H`
  readout.  Prediction metadata now
  carries source-refresh subspan/count and aggregate source-evaluation metadata,
  full-chain artifact/window/replay/source provenance, terminal electron-mu/charge-asymmetry observables,
  scalar-QED model/contract markers, and the mode-specific collision-feedback
  contract.
- **Scope boundary:** no accidental `canonical_forward_solver` promotion, no
  public canonical dispatch, no SMC-run/plot promotion, and no public production
  support; AP76 later audited the evidence and retained diagnostic staging.
- **Exit gate:** tests lock API inputs, AP65 parameter forwarding, source-model
  dispatch, radial-control, momentum-delta, and scalar-QED forwarding, AP68
  piecewise-frozen subspan handoff, aggregate source-evaluation metadata, radial budget-failure metadata, guarded
  deterministic config, electron-bath control forwarding, terminal
  electron-bath metadata, exact-scalar-QED prediction metadata, `Yp`/`D/H` extraction from the PRIMAT abundance
  vector, collision-source `dA` metadata, existing likelihood-wrapper
  compatibility, and absence of default dispatch or registry promotion.
- **Verification:** `PYTHONPATH=src pytest -q tests/test_augmented_candidate_forward_model.py`
  passed locally, including a real short-span combined-source smoke.
  A real short-span combined-source charge-neutral AP68 prediction reported
  `success=True`, `electron_chemical_potential_MeV_final=3.300602867355805e-10`,
  `electron_charge_asymmetry_density_MeV3_final=6.623064974048603e-11`,
  `electron_charge_asymmetry_state_evolved=True`,
  `collision_dA_abs_max_final=2.2350365034067905e-4`, and
  `pstf_radial_source_evaluations=7`.
  The scalar-QED forwarding focused target
  `PYTHONPATH=src pytest -q tests/test_augmented_candidate_forward_model.py::test_augmented_candidate_forward_prediction_forwards_exact_scalar_qed tests/test_augmented_candidate_forward_model.py::test_augmented_candidate_forward_config_rejects_bad_inputs`
  passed after AP68 accepted `qed_correction_model=exact_finite_mu_scalar`,
  forwarded it to AP65, and recorded prediction metadata markers.  A real
  short-span combined-source AP68 prediction with charge-neutral electron bath,
  `radial_gaussian`, `standard_3t_plasma`, and `exact_finite_mu_scalar`
  reported `success=True`, `qed_correction_contract=exact_finite_mu_scalar_3t_thermo_v1`,
  exact-QED marker `1.0`, `collision_dA_abs_max_final=7.360025983455074e-05`,
  and `pstf_radial_source_budget_passed=True`.
  A real AP68 piecewise-frozen combined-source smoke over `N_span=(0,1e-14)`
  and `source_update_subspan_ends=(5e-15,1e-14)` reported `success=True`,
  `source_update_subspan_count=2`, aggregate `nfev=10`,
  `pstf_radial_source_evaluations=2`,
  `collision_dA_abs_max_final=0.0002652986559886625`,
  `T_gamma_final_MeV=0.7999999999999923`, and
  `Sigma_plus_final=0.0099999999999999`.
  A real AP68 full-chain smoke over two `(1e-8)` windows reported
  `success=True`, finite `Yp`/`D/H`,
  `full_chain_completed_windows=2`, `nfev=18`,
  `full_chain_bbn_observables_present=True`, `public_dispatch_ready=False`,
  and `qke_scope=out_of_scope`.
- **Known red tests:** focused AP68 tests first failed on the missing
  `rabbit.inference.augmented_forward_model` module; AP68 source-model upgrade
  tests later failed on the missing combined wrapper import and missing
  `source_model` config surface, and the electron-bath upgrade tests later
  failed on the missing `electron_chemical_potential_mode` config surface,
  then passed after implementation.  The scalar-QED follow-up first failed
  because AP68 did not accept `qed_correction_model` as a guarded config option.
  The piecewise-frozen follow-up first failed because AP68 rejected
  `source_update_subspan_ends` as an unknown config key and limited
  `source_update_policy` to frozen/live values.  The FB-12 full-chain follow-up
  first failed because AP68 had no chained runner import and rejected
  `execution_mode`/`full_chain_artifact` config keys.

### AP69-AUGMENTED-SMC-LIKELIHOOD-SCHEMA  Priors and likelihood contract

- **Status:** stage-recorded (historical; not current capability)
- **Scope:** define likelihood, priors, units, labels, and parameter schema for
  augmented no-QKE SMC runs by extending existing inference contracts.
- **Key files:** `src/rabbit/inference/augmented_smc.py`,
  `tests/test_augmented_smc_schema.py`.
- **Physics added/changed:** no solver physics.  The PR defines how baryon
  density, nuisance terms, shear, non-LRS amplitudes, source policy,
  source model, radial energy-normalization/source-budget/momentum-delta controls,
  fixed/charge-neutral electron-bath controls, scalar-QED model selection,
  AP68 piecewise source-refresh subspan ends, AP68 `execution_mode`, full-chain
  window/cache/source-refresh/replay/restart controls, weak-rate mode, and solver policy enter inference while reusing observation loading, `BBNLikelihood`,
  and vector-adapter conventions.
- **Scope boundary:** schema and likelihood plumbing only; no production SMC
  run, publication plot, public dispatch, or QKE claim.
- **Exit gate:** tests reject invalid priors and solver controls, preserve
  units/labels, round-trip vector parameters, record AP68 source-model/radial
  momentum-delta, source-refresh subspan, electron-bath, scalar-QED, and full-chain controls in schema metadata, produce stable log-likelihood
  values for deterministic AP68 outputs, run real angular and combined-source
  fixed/charge-neutral/piecewise/full-chain AP68-backed smoke likelihoods, verify existing vector-adapter compatibility, and
  propagate AP68 classified solver failures.
- **Verification:** `PYTHONPATH=src pytest -q tests/test_augmented_smc_schema.py`
  passed locally, including a real short-span combined-source likelihood smoke.
  A real short-span combined-source charge-neutral likelihood reported
  `log_likelihood=-5731.092605376938`,
  `electron_chemical_potential_MeV_final=3.3006028673558036e-10`,
  `electron_charge_asymmetry_density_MeV3_final=6.623064974048599e-11`,
  `electron_charge_asymmetry_state_evolved=True`,
  `collision_dA_abs_max_final=2.2350365034067905e-4`, and
  `pstf_radial_source_evaluations=35`.
  The scalar-QED schema follow-up focused targets passed after AP69 recorded
  `qed_correction_model=exact_finite_mu_scalar` in solver controls,
  AP68 forward-config metadata, and AP68 likelihood-call kwargs.  A real
  short-span combined-source AP69 likelihood with charge-neutral electron bath,
  `radial_gaussian`, `standard_3t_plasma`, and `exact_finite_mu_scalar`
  reported finite `log_likelihood=-5731.092605376489`, exact-QED marker `1.0`,
  and `collision_dA_abs_max_final=7.360025983455233e-05`.
  A real AP69 piecewise-frozen combined-source likelihood over
  `N_span=(0,1e-14)` and `source_update_subspan_ends=(5e-15,1e-14)`
  reported finite `log_likelihood=-5731.092605444093`,
  `source_update_subspan_count=2`, aggregate `nfev=18`,
  `pstf_radial_source_evaluations=2`, and
  `collision_dA_abs_max_final=0.0002652986559886625`.
  A real AP69 full-chain likelihood smoke over AP68
  `execution_mode="full_chain"` returned finite
  `log_likelihood=-5731.092605318141`,
  `full_chain_completed_windows=2`, `public_dispatch_ready=False`, and
  `qke_scope=out_of_scope`.
- **Known red tests:** focused AP69 tests first failed on the missing
  `rabbit.inference.augmented_smc` module; AP69 source-model provenance tests
  later failed on the missing `source_model` solver-control row, and
  electron-bath provenance tests later failed on the missing
  `electron_chemical_potential_mode` solver-control row, and radial-Gaussian
  provenance tests later failed on the missing momentum-delta solver-control
  row, then passed after implementation.  The scalar-QED follow-up first failed
  because AP69 had no schema-level `qed_correction_model` solver-control row.
  The piecewise follow-up first failed because the AP69 source-update policy
  solver-control allowlist still rejected `piecewise_frozen`.  The FB-13
  full-chain follow-up first failed because AP69 had no `execution_mode`
  solver-control row and AP71 cache contexts did not expose full-chain keys.

### AP70-TEMPERED-SMC-RUNNER  Adaptive tempered SMC

- **Status:** stage-recorded (historical; not current capability)
- **Scope:** add a smoke-scale tempered-SMC runner for the augmented candidate
  forward model.
- **Key files:** `src/rabbit/inference/augmented_smc.py`,
  `tests/test_augmented_tempered_smc.py`.
- **Physics added/changed:** no new physics.  Adaptive temperature scheduling,
  ESS resampling, proposal moves, deterministic seeds, and failure accounting
  are wired to the AP69 schema/likelihood.  The result metadata now preserves
  AP69 source-model/source-refresh/radial/electron-bath/scalar-QED controls and the AP68
  forward-config provenance, including radial momentum-delta settings and
  `piecewise_frozen` source-refresh subspan ends and FB-14 full-chain execution
  controls, for
  downstream artifacts.  Checkpoint/restart and runtime cache controls are AP71.
- **Scope boundary:** smoke-scale by default; production particle counts,
  public dispatch, and long spans remain blocked after AP76; AP74 adds diagnostic
  AP73-derived plots and AP75 packages reproducibility evidence only.
- **Exit gate:** unit tests reject invalid schedules, lock replayable seeds,
  beta-one normalization, finite log weights, ESS/resampling metadata,
  caller-provided initial particles, failure-aware particle accounting, and
  source-model/source-refresh/radial/electron-bath/scalar-QED-control metadata
  preservation with radial momentum-delta, full-chain controls, and `piecewise_frozen` subspan
  forward-config provenance.
- **Verification:** `PYTHONPATH=src pytest -q tests/test_augmented_tempered_smc.py`
  passed locally.
  A tiny real AP70 run with two supplied particles and the AP69 combined-source
  charge-neutral likelihood reported `complete=True`,
  `qed_correction_model=exact_finite_mu_scalar`,
  `electron_chemical_potential_mode=charge_neutrality`,
  `finite_loglike_count=2`, `forward_failures=0`, normalized weights
  `[0.5000000000059117, 0.4999999999940883]`, and terminal
  `collision_dA_abs_max_final=7.360025983455233e-05`.
  A real full-chain CLI SMC smoke with two particles and temperatures `(0,1)`
  completed with `execution_mode=full_chain`, `finite_loglike_count=2`,
  `forward_failures=0`, `cache_misses=2`, `full_chain_window_edges=[0,1e-8,2e-8]`,
  `public_dispatch_ready=false`, and `qke_scope=out_of_scope`.
- **Known red tests:** focused AP70 tests first failed on the missing
  `AUGMENTED_TEMPERED_SMC_CONTRACT`/runner symbols; AP70 source-model metadata
  tests later failed because result metadata did not expose the AP69
  source-model/radial controls, and electron-bath metadata tests later failed
  because top-level result metadata did not expose
  `electron_chemical_potential_mode`; scalar-QED metadata tests then failed
  because top-level result metadata did not expose `qed_correction_model`, and
  source-refresh metadata tests later failed because top-level result metadata
  did not expose `source_update_policy`, then passed after implementation.
  The FB-14 CLI follow-up first failed because
  `scripts/run_augmented_tempered_smc.py` did not accept full-chain controls.

### AP71-SMC-RUNTIME-CACHE  Resumable expensive-run controls

- **Status:** stage-recorded (historical; not current capability)
- **Scope:** extend existing checkpoint, batched likelihood, duplicate-call
  avoidance, run-manifest, and failure-metadata surfaces for expensive AP68
  calls.
- **Key files:** `src/rabbit/inference/augmented_smc.py`,
  `scripts/run_augmented_tempered_smc.py`,
  `tests/test_augmented_smc_runtime.py`.
- **Physics added/changed:** no new physics.  This PR makes long SMC runs
  restartable and auditable by reusing the current checkpoint and manifest
  discipline.  The AP71 cache key now includes AP69 solver controls and AP68
  forward-config provenance, with first-class source-model/source-refresh/radial/electron-bath
  cache-context fields plus scalar-QED/full-chain-aware cache context, so angular,
  combined angular+`pstf_radial`, fixed `mu_e`, charge-neutral, and
  finite-mu-scaled versus exact scalar-QED likelihood records cannot collide for
  the same parameter vector, and different `piecewise_frozen` subspan schedules
  or full-chain window/cache controls cannot collide.  The diagnostic CLI dry-run now forwards and records the same
  AP69 source/source-refresh/radial/electron-bath controls plus `N_span` end,
  scalar-QED, radial momentum-delta, `piecewise_frozen` subspan controls, and
  full-chain execution/window/cache/source-refresh/replay/restart controls.  Runtime
  result metadata and manifests also preserve the last successful AP68
  prediction metadata, so source-refresh subspan counts, radial source
  evaluations, and terminal collision-source amplitudes remain visible in SMC
  artifacts.
- **Scope boundary:** cache hits must preserve all solver failure metadata and
  cannot silently mix configurations; validation suites, publication plots,
  public dispatch, and QKE remain out of scope.
- **Exit gate:** smoke SMC runs resume from checkpoint, avoid duplicate forward
  calls, keep source-model/radial/electron-bath-control cache records separated,
  keep scalar-QED, source-refresh, and full-chain cache records separated, preserve last
  successful prediction metadata, reject incompatible manifests, and emit
  portable run metadata.
- **Verification:** `PYTHONPATH=src pytest -q tests/test_augmented_smc_runtime.py`
  passed locally, and `PYTHONPATH=src python scripts/run_augmented_tempered_smc.py --dry-run --particles 4 --outdir /tmp/rabbit_augmented_smc_ap71_dry_run`
  emitted the staged no-QKE dry-run manifest payload.  The AP71 electron-bath
  follow-up focused bundle
  `PYTHONPATH=src pytest -q tests/test_augmented_smc_runtime.py tests/test_augmented_tempered_smc.py tests/test_augmented_smc_schema.py tests/test_augmented_pstf_capability_registry.py tests/test_registry_sync.py`
  passed with `48 passed, 3 skipped`, and
  `PYTHONPATH=src python scripts/run_augmented_tempered_smc.py --dry-run --particles 4 --outdir /tmp/rabbit_augmented_smc_ap71_electron_dry_run --source-model combined_angular_pstf_radial --source-update-policy live_rhs --weak-rate-mode nonlrs_s2_cl3_quadrupole_input --pstf-radial-energy-normalization standard_3t_plasma --pstf-radial-momentum-delta-model radial_gaussian --pstf-radial-momentum-delta-sigma 0.2 --max-pstf-radial-source-evaluations 11 --electron-chemical-potential-MeV 2.5e-7 --electron-chemical-potential-mode fixed`
  reported `source_model=combined_angular_pstf_radial`,
  `pstf_radial_momentum_delta_model=radial_gaussian`,
  `electron_chemical_potential_MeV=2.5e-7`, and
  `electron_chemical_potential_mode=fixed` in the dry-run schema payload.  A
  scalar-QED cache-context follow-up passed after verifying that
  `finite_mu_scaled` and `exact_finite_mu_scalar` records do not collide and
  that cache records expose top-level `qed_correction_model`.
  AP71 manifests now record `qed_correction_model=exact_finite_mu_scalar` in
  both metadata and runtime payloads.
  A
  real two-particle AP71 runtime smoke with duplicate supplied particles and
  the same fixed electron-bath mode reported `complete=True`, `cache_hits=1`,
  `cache_misses=1`, `cache_entries=1`, `finite_loglike_count=2`,
  `forward_failures=0`, normalized weights, and
  `electron_chemical_potential_mode=fixed` in the written manifest payload.
- **Known red tests:** focused AP71 tests first failed on the missing runtime
  symbols and then on fingerprint serialization expecting nonexistent metadata
  properties; AP71 source-model cache tests later failed because the duplicate
  likelihood key omitted source-model/radial controls, then passed after
  implementation.  AP71 electron-bath cache-context and CLI dry-run tests later
  failed because electron controls were only nested in the forward config and
  absent from the script surface, then passed after implementation.  AP71
  scalar-QED cache-context tests later failed because QED controls were only
  nested in solver controls/forward config and absent from top-level cache
  context, the piecewise source-refresh follow-up first failed because
  subspan controls were absent from the CLI and top-level cache context, and the
  full-chain CLI follow-up first failed because execution/window/cache controls
  were absent from the CLI surface, and the
  successful-prediction metadata follow-up first failed because AP68 prediction
  metadata was not copied into result/manifest metadata, then passed after
  implementation.

### AP72-SMC-VALIDATION-SUITE  Synthetic SMC validation plus physical full-chain smoke

- **Status:** stage-recorded (historical; not current capability)
- **Scope:** run synthetic null and injection/recovery tests, posterior
  contraction checks, logZ uncertainty reports, and no-spurious-shear checks on
  FLRW-like data using the existing full-forward null/recovery and non-LRS SMC
  conventions.
- **Key files:** `src/rabbit/validation/augmented_smc_validation.py`,
  `scripts/run_augmented_smc_validation.py`,
  `tests/test_augmented_smc_validation.py`.
- **Physics added/changed:** the default AP72 artifact is still synthetic and
  analytic.  The PR determines whether the AP70/AP71 SMC path is statistically
  sane before publication plotting, and the synthetic AP69-compatible
  likelihood can now accept schema overrides so AP72 validation metadata and
  CLI dry-run schema payloads preserve
  source-model/source-refresh/`N_span`/radial/electron-bath controls plus
  scalar-QED and radial momentum-delta provenance without claiming the analytic
  synthetic tests validate the physical combined-source solve.  FB-15 adds an
  opt-in physical full-chain smoke row that calls the AP68
  `execution_mode="full_chain"` likelihood from AP70/AP71 SMC and records AP68
  terminal BBN observables plus CPU-JAX/Rodas5P replay evidence.  It can also
  request the FB-04 live-source RHS chain as the repeated-run BBN readout and
  records that readout source in AP72 diagnostics.  When FB-21 gate evidence is
  requested or present, AP72 preserves the live-source repeated-run gate
  contract, diagnostic-only claim scope, no-public/no-production/no-QKE flags,
  finite repeated-run BBN readouts, same-window gate counts, finite comparison
  deltas, and supplied/applied/provenance-fingerprinted frozen-collision payloads in
  physical-smoke diagnostics and metadata.
- **Scope boundary:** synthetic validation and the opt-in physical smoke do not
  equal real-data production inference; AP73 still consumes synthetic-only AP72
  artifacts, while plots, public dispatch, production SMC, and QKE remain out of
  scope.
- **Exit gate:** validation artifacts include posterior summaries, ESS trace,
  rejuvenation diagnostics, logZ error, forward-success accounting, and
  explicit pass/fail thresholds, plus AP69
  source-model/source-refresh/`N_span`/radial/electron-bath control provenance,
  scalar-QED provenance, and radial momentum-delta
  provenance when schema overrides are supplied.  The opt-in physical smoke
  gate additionally requires complete full-chain SMC execution, finite
  log-likelihoods, zero forward failures, finite AP68 `Yp`/`D/H`, no
  public/QKE promotion, and CPU-JAX/Rodas5P replay-target metadata; when the
  live-source repeated-run source is requested, it also requires the
  `rodas5p_repeated_run` BBN readout source and ready flag.  If FB-21 gate
  evidence is requested or present, the smoke gate also requires the current
  gate contract, diagnostic-only claim scope, no-public/no-production/no-QKE
  flags, finite repeated-run BBN readouts, gate counts matching the AP72 smoke
  windows, finite comparison deltas, and frozen-collision payload counts/provenance
  covering every completed chain window.
- **Verification:** `PYTHONPATH=src pytest -q tests/test_augmented_smc_validation.py`
  passed locally; `PYTHONPATH=src python scripts/run_augmented_smc_validation.py --particles 48 --outdir /tmp/rabbit_augmented_smc_ap72_validation`
  produced a passing two-case synthetic validation artifact.  The AP72
  electron-bath follow-up bundle
  `PYTHONPATH=src pytest -q tests/test_augmented_smc_validation.py tests/test_augmented_smc_runtime.py tests/test_augmented_tempered_smc.py tests/test_augmented_smc_schema.py tests/test_augmented_pstf_capability_registry.py tests/test_registry_sync.py`
  passed with `54 passed, 3 skipped`; the CLI command
  `PYTHONPATH=src python scripts/run_augmented_smc_validation.py --particles 48 --temperatures smoke --seed 20260514 --outdir /tmp/rabbit_augmented_smc_ap72_electron_validation --source-model combined_angular_pstf_radial --source-update-policy live_rhs --weak-rate-mode nonlrs_s2_cl3_quadrupole_input --pstf-radial-energy-normalization standard_3t_plasma --pstf-radial-momentum-delta-model radial_gaussian --pstf-radial-momentum-delta-sigma 0.2 --max-pstf-radial-source-evaluations 11 --electron-chemical-potential-MeV 2.5e-7 --electron-chemical-potential-mode fixed`
  produced `status=pass` for both synthetic cases with
  `electron_chemical_potential_mode=fixed`, `source_model=combined_angular_pstf_radial`,
  `pstf_radial_momentum_delta_model=radial_gaussian`,
  `min_ess_fraction=0.25072332439029354` / `0.29439317936882914`,
  and `logz_error_estimate=0.05744447703772728` /
  `0.05565714430192415`.  The scalar-QED follow-up CLI run
  `PYTHONPATH=src python scripts/run_augmented_smc_validation.py --particles 24 --temperatures 0,1 --mcmc-steps 0 --seed 20260515 --outdir /tmp/rabbit_augmented_smc_ap72_qed_validation --source-model combined_angular_pstf_radial --source-update-policy live_rhs --weak-rate-mode nonlrs_s2_cl3_quadrupole_input --pstf-radial-energy-normalization standard_3t_plasma --pstf-radial-momentum-delta-model radial_gaussian --pstf-radial-momentum-delta-sigma 0.2 --max-pstf-radial-source-evaluations 11 --electron-chemical-potential-MeV 2.5e-7 --electron-chemical-potential-mode fixed --qed-correction-model exact_finite_mu_scalar`
  produced `status=pass` for both synthetic cases with
  `qed_correction_model=exact_finite_mu_scalar`,
  `min_ess_fraction=0.3806873652949445` / `0.3835355515447813`, and
  `logz_error_estimate=0.0675312092815847` / `0.06727999413169596`.
  The piecewise source-refresh follow-up CLI run with
  `source_update_policy=piecewise_frozen`, `N_span=(0,1e-14)`, and
  `source_update_subspan_ends=(5e-15,1e-14)` produced `status=pass` for both
  synthetic cases, preserving `source_update_policy`, `source_update_subspan_ends`,
  `N_span`, `qed_correction_model=exact_finite_mu_scalar`, and
  `pstf_radial_momentum_delta_model=radial_gaussian` in each case row.  The
  FB-15 physical-smoke follow-up
  `PYTHONPATH=src JAX_PLATFORMS=cpu python scripts/run_augmented_smc_validation.py --outdir /tmp/rabbit_ap72_full_chain_physical_smoke --particles 48 --temperatures 0,0.5,1 --mcmc-steps 0 --include-full-chain-physical-smoke --physical-smoke-particles 2 --physical-smoke-temperatures 0,1 --physical-smoke-mcmc-steps 0 --execution-mode full_chain --source-model combined_angular_pstf_radial --source-update-policy piecewise_frozen --source-update-subspan-ends 1e-8 --N-span-end 1e-8 --method Radau --full-chain-window-edges 0,1e-8,2e-8 --full-chain-cache-key ap72-physical-smoke --full-chain-source-refresh-strategy adaptive_budget --full-chain-subspans-per-window 1 --full-chain-verify-rodas5p-window-map-replay`
  produced top-level `status=pass`; the physical smoke row reported
  `finite_loglike_count=2`, `forward_failures=0`,
  `full_chain_completed_windows=2`,
  `full_chain_rodas5p_window_map_replay_passed=true`,
  `last_Yp_final=6.824762169972173e-23`, and
  `last_DH_final=5.174943664962658e-13`.  A live-source repeated-run follow-up
  `PYTHONPATH=src JAX_PLATFORMS=cpu venv/bin/python scripts/run_augmented_smc_validation.py --outdir /tmp/rabbit_ap72_live_repeated_smoke --particles 48 --temperatures 0,0.5,1 --mcmc-steps 0 --include-full-chain-physical-smoke --physical-smoke-particles 2 --physical-smoke-temperatures 0,1 --physical-smoke-mcmc-steps 0 --execution-mode full_chain --source-model combined_angular_pstf_radial --source-update-policy piecewise_frozen --source-update-subspan-ends 1e-10 --N-span-end 1e-10 --method Radau --full-chain-window-edges 0,1e-10,2e-10 --full-chain-cache-key ap72-live-repeated-smoke --full-chain-source-refresh-strategy adaptive_budget --full-chain-rodas5p-repeated-run-source live_source_rhs_chain --full-chain-subspans-per-window 1`
  produced `status=pass` with `live_source_repeated_run_readout=true`,
  `full_chain_bbn_readout_source=rodas5p_repeated_run`,
  `full_chain_rodas5p_repeated_run_source_ready=true`,
  `last_Yp_final=4.133815059150155e-25`, and
  `last_DH_final=5.174943482288938e-13`.
- **Known red tests:** focused AP72 tests first failed on missing
  `rabbit.validation.augmented_smc_validation`; AP72 source-model provenance
  tests later failed because validation cases could not accept AP69 schema
  overrides, then passed after implementation.  AP72 electron-bath provenance
  and CLI dry-run tests later failed because metadata only exposed source-model
  controls and the script had no AP69 control surface, then passed after
  implementation.  Radial-Gaussian provenance tests later failed because AP72
  CLI and top-level case metadata did not expose momentum-delta controls.
  AP72 scalar-QED provenance tests later failed because validation case metadata
  did not expose `qed_correction_model` and the script had no
  `--qed-correction-model` surface, then passed after implementation.  The
  source-refresh follow-up first failed because validation metadata did not
  expose `source_update_policy` / `source_update_subspan_ends` / `N_span`, and
  the CLI rejected `--N-span-end` / `--source-update-subspan-ends`.  FB-15 first
  failed because AP72 had no physical full-chain validation function and the
  validation CLI rejected full-chain execution/window/replay controls.  The
  live-source repeated-run follow-up first failed because AP72 physical-smoke
  payloads dropped `full_chain_rodas5p_repeated_run_source` and
  `full_chain_bbn_readout_source`, then passed after the provenance was wired
  into schema kwargs, diagnostics, checks, and metadata.

### AP73-PUBLICATION-ARTIFACT-SCHEMA  Figure-ready artifact builder

- **Status:** stage-recorded (historical; not current capability)
- **Scope:** convert convergence, validation-atlas, Schramm/likelihood-cache,
  and SMC-validation outputs into versioned figure-ready tables with provenance
  on the existing figure-cache discipline.
- **Key files:** `src/rabbit/validation/augmented_publication_artifacts.py`,
  `scripts/figure_cache_schema.py`,
  `scripts/figure_registry.py`,
  `scripts/build_augmented_publication_artifacts.py`,
  `tests/test_augmented_publication_artifacts.py`.
- **Physics added/changed:** no new physics.  This PR prevents publication plots
  from consuming stale, mixed-contract, or unsupported artifacts and keeps
  augmented outputs compatible with existing report/figure cache tooling.  The
  AP73 convergence table schema now carries AP66 terminal kinetic-source
  amplitude rows via `collision_dA_abs_max_final` with explicit per-e-fold
  units plus AP66 electron-mu/charge-asymmetry rows with MeV/MeV^3 units for
  downstream figure panels, and AP67/AP72-derived rows now preserve
  source-model/source-refresh/`N_span`/radial-control/radial-momentum-delta/
  electron-bath-control context for downstream plot provenance, including
  AP72 `piecewise_frozen` subspan context in SMC posterior and temperature
  rows, and AP67 validation rows can now carry scalar-QED context.
  AP66-derived convergence rows and AP67-derived
  validation rows now also preserve categorical `qed_correction_model` provenance plus
  finite-mu-scaled versus exact-scalar-QED markers.  FB-16 adds a guarded
  non-synthetic AP72 intake for passed `full_chain_physical_forward_smoke`
  artifacts and maps their AP68 terminal `Yp`/`D/H`, `Sigma_H`, `eta10`,
  `N_eff_3T`, completed windows, CPU-JAX/Rodas5P replay status, and optional
  live-source repeated-run BBN readout provenance into diagnostic Schramm rows.
  Live-source AP72 smoke rows fail closed unless the AP72
  `live_source_repeated_run_readout` check passed.  FB-21 gate evidence now
  also fails closed unless the current gate contract, diagnostic-only claim
  scope, no-public/no-production/no-QKE flags, finite repeated-run BBN
  readouts, same-window gate counts, finite comparison deltas, and
  supplied/applied/provenance-fingerprinted frozen-collision payloads are present, and those gate
  fields are copied into the Schramm row.
- **Scope boundary:** artifact readiness is separate from physics promotion.
- **Scope boundary:** this is a figure-data artifact builder only.  It does not
  render publication plots, run real-data/production SMC, promote public
  dispatch, or alter the no-QKE boundary.
- **Exit gate:** tests lock schema versions, required columns, units including
  the AP66 terminal kinetic-source amplitude,
  source-model/source-refresh/`N_span`/radial/radial-momentum-delta/electron-bath/scalar-QED provenance in convergence, validation, and SMC rows where present, commit provenance,
  registry keys, JSON writer output, Schramm cache loading, CLI summary output,
  and rejection of stale/mixed-contract or invalid non-synthetic SMC artifacts.
  FB-16 also locks AP72 live-source repeated-run source, BBN readout source, and
  ready-flag propagation into the Schramm row.
  Existing Schramm cache-only rows keep the
  `existing_cache_reformatted_only` claim label, while AP72 physical-smoke-only
  Schramm rows use `diagnostic_full_chain_physical_smoke_only`.
- **Verification:** `PYTHONPATH=src pytest -q tests/test_augmented_publication_artifacts.py`
  passed.  The AP73 electron-bath follow-up bundle
  `PYTHONPATH=src pytest -q tests/test_augmented_publication_artifacts.py tests/test_augmented_smc_validation.py tests/test_augmented_publication_plots.py tests/test_augmented_pstf_capability_registry.py tests/test_registry_sync.py`
  passed with `47 passed, 3 skipped`; a concrete AP73 JSON build wrote
  `/tmp/rabbit_augmented_ap73_electron_artifacts.json` with `table_count=5`,
  `row_count=11`, `validation_electron_mode=fixed`,
  `posterior_electron_MeV=2.5e-7`, `trace_electron_mode=fixed`, and
  `electron_chemical_potential_MeV` present in SMC posterior columns.
- **Known red tests:** focused AP73 tests first failed on missing
  `rabbit.validation.augmented_publication_artifacts`; AP73 source-model
  provenance tests later failed because AP67/AP72 rows dropped the source model,
  then passed after implementation.  AP73 electron-bath row-context tests later
  failed because AP67/AP72 figure rows omitted
  `electron_chemical_potential_MeV` and `electron_chemical_potential_mode`,
  then passed after implementation.  AP73 radial momentum-delta row-context
  tests later failed because AP66 rows omitted
  `pstf_radial_momentum_delta_model`, then passed after implementation.
  AP73 source-refresh row-context tests later failed because AP72-derived SMC
  artifact rows omitted `source_update_subspan_ends` and `N_span`, then passed
  after implementation.  FB-16 first failed because AP73 rejected all
  non-synthetic AP72 artifacts, then passed after adding fail-closed
  full-chain physical smoke validation and Schramm-row extraction.  The
  live-source repeated-run follow-up first failed because AP73 Schramm rows
  dropped `live_source_repeated_run_readout` and BBN readout provenance, then
  passed after the AP72 diagnostics/checks were propagated and validated.

### AP74-PUBLICATION-PLOT-GENERATOR  Publication-level figures

- **Status:** stage-recorded (historical; not current capability)
- **Scope:** integrate augmented publication panels with the existing
  paper/report figure stack using AP73 figure-ready tables.
- **Key files:** `src/rabbit/figures/augmented_publication_plots.py`,
  `scripts/generate_paper_figures.py`,
  `scripts/regenerate_all_figures.sh`,
  `scripts/plot_augmented_publication_figures.py`,
  `tests/test_augmented_publication_plots.py`.
- **Physics added/changed:** no new physics.  This PR generates figures only
  from AP73-approved artifacts and records plot/source-model/source-refresh/
  `N_span`/radial momentum-delta/electron-bath/scalar-QED provenance.  The
  stage-scoped panel set includes augmented convergence, validation-atlas,
  Schramm `Y_p`/`D/H`, synthetic posterior, and SMC temperature-trace panels.
  FB-17 extends AP74 so Schramm tables labeled
  `diagnostic_full_chain_physical_smoke_only` or
  `diagnostic_schramm_rows_mixed_sources` render through the existing Schramm
  panel path, with plot/manifest records carrying full-chain physical-smoke
  presence, completed-window count, CPU-JAX/Rodas5P replay status, optional
  live-source repeated-run BBN readout provenance, and FB-21 gate contract/
  claim/readout/window/payload/delta provenance with exact completed-window
  payload-count matching.
- **Scope boundary:** figures must carry diagnostic/candidate labels from the
  source artifacts and cannot imply unsupported production support.  AP74 does
  not add real-data/production SMC evidence, publication-bundle promotion,
  canonical dispatch, or QKE.
- **Exit gate:** tests verify expected files, nonempty plot layers, labels and
  units, registry/cache metadata, source-model/source-refresh/`N_span`/radial momentum-delta/electron-bath/scalar-QED
  manifest provenance, reproducible hashes or metadata, and rejection of
  unsupported artifacts, plus full-chain physical-smoke Schramm plot provenance.
- **Verification:** `PYTHONPATH=src pytest -q tests/test_augmented_publication_plots.py`
  passed locally.  The AP74 electron-bath follow-up bundle
  `PYTHONPATH=src pytest -q tests/test_augmented_publication_plots.py tests/test_augmented_publication_artifacts.py tests/test_augmented_pstf_capability_registry.py tests/test_registry_sync.py`
  passed with `41 passed, 3 skipped`; a concrete render wrote
  `/tmp/rabbit_augmented_ap74_electron_plots/augmented_publication_plot_manifest.json`
  with `plot_count=5`, `source_models=[combined_angular_pstf_radial]`,
  `electron_chemical_potential_modes=[fixed]`,
  `electron_chemical_potential_MeV_values=[2.5e-7]`, and the validation plot
  record carrying `electron_chemical_potential_modes=[fixed]`.  The AP74 radial
  momentum-delta follow-up focused test
  `PYTHONPATH=src pytest -q tests/test_augmented_publication_plots.py::test_augmented_publication_plots_render_expected_files`
  passed after adding manifest and plot-record
  `pstf_radial_momentum_delta_models=[radial_gaussian]` and
  `pstf_radial_momentum_delta_sigmas=[0.42]` provenance.  The AP74 scalar-QED
  provenance follow-up focused test passed after adding manifest and plot-record
  `qed_correction_models=[exact_finite_mu_scalar]` provenance for AP66-derived
  convergence rows.  The AP74 source-refresh follow-up focused tests passed
  after adding manifest, plot-record, and CLI-summary
  `source_update_policies=[piecewise_frozen]`,
  `source_update_subspan_ends=[(5e-15,1e-14)]`, and
  `N_span=(0,1e-14)` provenance for AP72-derived SMC rows.
- **Known red tests:** focused AP74 tests first failed on missing
  `rabbit.figures.augmented_publication_plots`; the regeneration-hook label
  assertion then failed on the stale `1/5` step count before the script label
  was refreshed.  The radial momentum-delta follow-up first failed on missing
  `pstf_radial_momentum_delta_models` in the AP74 manifest before per-plot and
  aggregate provenance was added.
  fix; the source-model provenance regression then failed until AP73 table
  source models were carried into plot records and the manifest.  AP74
  electron-bath provenance tests later failed because plot records and manifest
  omitted `electron_chemical_potential_mode` and
  `electron_chemical_potential_MeV`, then passed after implementation.  The
  scalar-QED provenance regression first failed because the AP74 manifest had
  no `qed_correction_models` field.  AP74 source-refresh provenance tests
  later failed because the AP74 manifest and CLI summary omitted
  `source_update_policies`, `source_update_subspan_ends`, and `N_spans`, then
  passed after implementation.  FB-17 first failed because AP74 rejected the
  AP73 `diagnostic_full_chain_physical_smoke_only` Schramm claim label.

### AP75-PRODUCTION-REPRO-BUNDLE  Reproducibility bundle

- **Status:** stage-recorded (historical; not current capability)
- **Scope:** package AP66/AP67/AP72/AP74 artifacts with existing figure-cache
  metadata, environment metadata, manifests, commands, and claim-boundary notes,
  including the FB-18 path that packages a passed AP72 full-chain physical smoke
  row together with AP74 full-chain Schramm plot provenance.
- **Key files:** `scripts/package_augmented_publication_bundle.py`,
  `src/rabbit/validation/augmented_publication_bundle.py`,
  `tests/test_augmented_publication_bundle.py`.
- **Physics added/changed:** no new collision physics.  This PR makes the
  candidate evidence bundle reproducible and reviewable while preserving AP74
  source-model/source-refresh/`N_span`/radial momentum-delta/electron-bath/
  scalar-QED provenance.  The FB-18 extension also preserves AP72 real
  full-chain physical-smoke diagnostics: finite AP68 terminal `Yp`/`D/H`,
  completed-window counts, zero forward failures, and CPU-JAX/Rodas5P
  window-map replay status, plus optional live-source repeated-run BBN readout
  and FB-21 gate contract/claim/readout/window/payload/delta provenance.
- **Scope boundary:** bundle packaging is not a promotion decision.  AP75 does
  not add real-data/production SMC evidence, canonical dispatch, QKE, or
  promotion approval; AP76 consumes the bundle for a separate not-promoted audit.
  The AP72 full-chain physical-smoke row remains diagnostic smoke evidence, not
  production SMC validation.
- **Exit gate:** bundle gate fails if required artifacts are missing, stale,
  produced from different commits/configs, absent from the registry/cache
  manifest, marked diagnostic-only where publication-candidate evidence is
  required, or carrying AP74 top-level source-model labels inconsistent with
  plot records, or carrying AP74 top-level electron-bath labels/values
  inconsistent with plot records, or carrying AP74 top-level radial
  momentum-delta models/sigmas inconsistent with plot records, or carrying AP74
  top-level scalar-QED model labels inconsistent with plot records, or carrying
  AP74 top-level source-update policy/subspan/`N_span` values inconsistent
  with plot records, or carrying AP74 full-chain physical-smoke plot records
  without a matching passed AP72 full-chain physical-smoke row, or carrying AP74
  live-source repeated-run/BBN-readout/FB-21 gate summaries inconsistent with
  plot records, including collision-payload counts/provenance that do not exactly match
  completed windows.
- **Verification:** `PYTHONPATH=src pytest -q tests/test_augmented_publication_bundle.py`
  passed locally.  The AP75 radial momentum-delta follow-up bundle
  `PYTHONPATH=src pytest -q tests/test_augmented_publication_bundle.py`
  passed after adding top-level, required-summary, verified-plot, and
  copied-plot `pstf_radial_momentum_delta_models=[radial_gaussian]` plus
  `pstf_radial_momentum_delta_sigmas=[0.42]` provenance.  The AP75 scalar-QED
  provenance follow-up passed after adding top-level, required-summary,
  verified-plot, and copied-plot
  `qed_correction_models=[exact_finite_mu_scalar]` provenance.  The AP75
  source-refresh follow-up passed after adding top-level, required-summary,
  verified-plot, and copied-plot
  `source_update_policies=[piecewise_frozen]`,
  `source_update_subspan_ends=[(5e-15,1e-14)]`, and `N_span=(0,1e-14)`
  provenance.  The FB-18 follow-up
  `PYTHONPATH=src pytest -q tests/test_augmented_publication_bundle.py -m "not slow"`
  passed after AP75 accepted AP72 full-chain physical-smoke artifacts only when
  their checks, finite `Yp`/`D/H`, zero forward failures, completed-window count,
  and CPU-JAX/Rodas5P replay status were present, and preserved the matching AP74
  Schramm plot provenance in bundle and copied-plot manifests.  The live-source
  repeated-run follow-up passed after AP75 preserved
  `full_chain_rodas5p_repeated_run_sources=[live_source_rhs_chain]` and
  `full_chain_bbn_readout_sources=[rodas5p_repeated_run]` at bundle, AP74
  summary, and verified-plot levels.
- **Known red tests:** focused AP75 tests first failed on missing
  `rabbit.validation.augmented_publication_bundle`; a negative test then
  caught the promoted-manifest error-message contract before the message was
  normalized; the AP74 source-model/electron-bath provenance regressions failed
  until bundle summaries and copied plot records preserved the source-model,
  electron chemical-potential mode, and electron chemical-potential value lists.
  The radial momentum-delta follow-up first failed until AP75 normalized,
  backfilled, and mismatch-checked AP74 radial momentum-delta model/sigma lists.
  The scalar-QED provenance regression first failed because AP75 did not expose
  `qed_correction_models` from the AP74 manifest.  AP75 source-refresh
  provenance tests later failed because bundle summaries and copied plot
  records omitted `source_update_policies`, `source_update_subspan_ends`, and
  `N_spans`, then passed after implementation.  FB-18 first failed because AP75
  still hard-rejected every non-synthetic AP72 artifact as non-synthetic instead
  of checking for a passed full-chain physical-smoke row.  The live-source
  repeated-run follow-up first failed because AP75 dropped
  `full_chain_rodas5p_repeated_run_sources` and
  `full_chain_bbn_readout_sources`, then passed after bundle summaries validated
  and propagated those fields.

### AP76-FINAL-PUBLICATION-READINESS-AUDIT  Promotion decision

- **Status:** stage-recorded (historical; not current capability)
- **Scope:** final audit of AP51-AP75 evidence, capability registry wording, and
  the exact publication/SMC claim surface, with the current decision recorded as
  `not_promoted`.
- **Key files:** `scripts/run_augmented_publication_readiness_audit.py`,
  `src/rabbit/validation/augmented_publication_readiness.py`,
  `tests/test_augmented_publication_readiness_audit.py`,
  `docs/ROADMAP_STATE_OF_RECORD.md`, `STATUS.md`, `SUPPORTED_CAPABILITIES.md`,
  `src/rabbit/config/backend_capabilities.py`,
  `src/rabbit/config/feature_capabilities.py`.
- **Physics added/changed:** no new physics.  The PR turns the final decision
  into an executable artifact audit, preserves AP75/AP74 source-model,
  source-refresh/`N_span`, radial momentum-delta, electron-bath, and
  scalar-QED provenance in the final ledger, accepts the FB-18 AP75 full-chain
  physical-smoke bundle path with finite terminal `Yp`/`D/H`, completed-window
  counts, CPU-JAX/Rodas5P replay status, and live-source repeated-run BBN
  readout only with matching FB-21 gate contract/claim/window/payload/delta
  provenance,
  and explicitly leaves the programme diagnostic.
- **Scope boundary:** no public production support is claimed.  The AP76 audit
  preserves the forbidden public-production claim because production SMC
  evidence, public dispatch, QKE, and full collision-coupled BBN blockers remain
  unresolved.
- **Exit gate:** final claim ledger includes commands, pass/fail summaries,
  source-model/source-refresh/`N_span`/radial momentum-delta/electron-bath/scalar-QED provenance checks, supported
  parameter ranges, residual blockers, forbidden claims, and exact wording for
  publication plots and tempered SMC evidence, including diagnostic full-chain
  physical-smoke evidence and FB-21 live-source repeated-run gate evidence when
  present.
- **Verification:** `PYTHONPATH=src pytest -q tests/test_augmented_publication_readiness_audit.py`
  passed locally.  The AP76 radial momentum-delta follow-up focused target
  `PYTHONPATH=src pytest -q tests/test_augmented_publication_readiness_audit.py::test_augmented_publication_readiness_audit_retains_diagnostic_surface tests/test_augmented_publication_readiness_audit.py::test_augmented_publication_readiness_audit_rejects_tampered_required_summaries`
  passed after adding source-bundle ledger fields and mismatch rejection.
  The scalar-QED provenance follow-up focused target passed after adding
  source-bundle `qed_correction_models=[exact_finite_mu_scalar]` ledger fields
  and mismatch/invalid-label rejection.  The source-refresh provenance
  follow-up focused target passed after adding source-bundle
  `source_update_policies=[piecewise_frozen]`,
  `source_update_subspan_ends=[(5e-15,1e-14)]`, and `N_span=(0,1e-14)`
  ledger fields plus mismatch rejection.  The FB-19 follow-up
  `PYTHONPATH=src pytest -q tests/test_augmented_publication_readiness_audit.py -m "not slow"`
  passed after AP76 accepted AP75 bundles that carry the AP72 full-chain
  physical-smoke summary and AP74 full-chain Schramm plot provenance while
  keeping the readiness decision `not_promoted`.  The live-source repeated-run
  follow-up passed after AP76 carried
  `full_chain_rodas5p_repeated_run_sources=[live_source_rhs_chain]` and
  `full_chain_bbn_readout_sources=[rodas5p_repeated_run]` into the final
  source-bundle ledger and rejected mismatched AP75/AP74 summaries.
  The FB-21 readiness follow-up passed after AP76 carried
  `full_chain_live_source_repeated_run_gate_provenance` into the ledger,
  validated contract/claim/window/payload/delta consistency, and rejected
  stale, fractional, source-missing, mismatched, or partial-plot-coverage gate
  evidence.
- **Known red tests:** focused AP76 tests first failed on the missing
  `scripts/run_augmented_publication_readiness_audit.py` CLI and the
  production-SMC error-message contract, then passed after the script and
  stricter AP75-bundle validation were added; source-model/electron-bath
  provenance regressions later failed until AP76 reported and validated the
  AP75/AP74/plot source-model and electron-bath chains.  The radial
  momentum-delta follow-up first failed until AP76 reported and validated the
  AP75/AP74/plot radial momentum-delta model/sigma chain.  The scalar-QED
  provenance regression first failed because the AP76 `source_bundle` ledger had
  no `qed_correction_models` field.  AP76 source-refresh provenance tests later
  failed because the `source_bundle` ledger omitted `source_update_policies`,
  `source_update_subspan_ends`, and `N_spans`, then passed after implementation.
  FB-19 first failed because AP76 still required AP72 synthetic-only evidence
  and rejected the AP75 full-chain physical-smoke bundle.  The live-source
  repeated-run follow-up first failed because the AP76 `source_bundle` ledger
  dropped `full_chain_rodas5p_repeated_run_sources`, then passed after those
  fields were validated and propagated.  The FB-21 gate provenance follow-up
  first failed because the AP76 `source_bundle` ledger dropped the gate block
  and because partial verified-plot coverage was accepted, then passed after
  AP76 required matching AP72/AP74/AP75/plot gate provenance for every
  full-chain plot.

### AP77-COUPLED-WEAK-RATE-GATE  Coupled AP60 weak-rate evidence

- **Status:** stage-recorded (historical; not current capability)
- **Scope:** move beyond AP61 rate-only evidence by exercising the AP60
  `nonlrs_s2_cl3_quadrupole_input` weak-rate mode inside the AP65 nonlinear
  angular collision-feedback 3T solve and comparing it with same-CL3
  `metadata_only` controls while forwarding fixed/charge-neutral electron-bath
  controls into the coupled solve.
- **Key files:** `scripts/run_augmented_coupled_weak_rate_gate.py`,
  `src/rabbit/validation/augmented_coupled_weak_rate_gate.py`,
  `tests/test_augmented_coupled_weak_rate_gate.py`.
- **Physics added/changed:** no new weak kernel.  The PR wires existing AP60
  angular weak-rate factors into an executable coupled-solve gate with injected
  non-LRS S2 moments, exact rate-application metadata checks, bounded nonzero
  `lambda_np`/`lambda_pn` deltas, every-adjacent-pair q-ladder drift checks,
  default-injection nonnegativity, fixed/charge-neutral electron-bath control
  forwarding and case provenance, finite solve outputs, and solve-effort limits.
- **Scope boundary:** AP77 is smoke-scale coupled evidence only.  It does not
  provide promotion-grade full-span weak-rate convergence, public dispatch,
  production SMC validation, default collision feedback, or QKE.
- **Exit gate:** report artifacts fail closed on missing AP60 application
  metadata, zero/oversized weak-rate response, solve failure, non-finite output,
  invalid electron-bath controls, q-ladder drift, or excessive solve effort.
- **Verification:** `PYTHONPATH=src pytest -q tests/test_augmented_coupled_weak_rate_gate.py`
  passed locally.
- **Known red tests:** focused AP77 tests first failed on the missing
  `rabbit.validation.augmented_coupled_weak_rate_gate` module, then passed after
  the module, CLI, writer, fake-runner checks, same-CL3 control isolation,
  every-adjacent-pair q-drift checks, default-injection nonnegativity, and real
  AP65 smoke gate were added.  Electron-bath forwarding tests later failed until
  AP77 carried fixed/charge-neutral controls through the spec, default runner,
  CLI dry-run payload, report inputs, and case rows.

### AP78-PUBLICATION-MATRIX-SAME-CL3-WEAK-RATE-CONTROL  AP66 control hardening

- **Status:** stage-recorded (historical; not current capability)
- **Scope:** harden the AP66 publication-candidate convergence matrix so
  `metadata_only` weak-rate rows use the same CL3 weak kernel as
  `nonlrs_s2_cl3_quadrupole_input` rows, isolating AP60 angular application
  from CL0-vs-CL3 base weak-rate-kernel differences.
- **Key files:** `src/rabbit/validation/augmented_publication_matrix.py`,
  `tests/test_augmented_publication_convergence_matrix.py`, registry-backed
  roadmap/capability docs.
- **Physics added/changed:** no new weak kernel.  AP78 changes the AP66
  diagnostic control basis so weak-rate-mode matrix rows differ by angular
  weak-rate application only while preserving the AP65 SciPy-first candidate
  solve path.
- **Scope boundary:** AP78 is a diagnostic matrix-control hardening PR only.
  It does not provide promotion-grade full-span weak-rate convergence, public
  dispatch, production SMC validation, default collision feedback, or QKE.
- **Exit gate:** focused AP66 tests must show same-CL3 configuration for both
  metadata-only and applied weak-rate modes, preserve AP66 observable metadata,
  and keep existing q/`N_mu`/`N_phi`, JSON, CLI, invalid-input,
  candidate-selection, failed-solve, and real frozen-source smoke gates green.
- **Verification:** `PYTHONPATH=src pytest -q tests/test_augmented_publication_convergence_matrix.py`
  passed locally.
- **Known red tests:** focused AP66 tests first failed because metadata-only
  rows still reported/used correction level 0; they passed after
  `_weak_config_for_mode(...)` and observable metadata were moved to same-CL3
  controls.

### AP79-READINESS-AUDIT-COUPLED-WEAK-RATE-LINK  AP77 evidence in readiness audit

- **Status:** stage-recorded (historical; not current capability)
- **Scope:** extend the AP76 readiness audit so it requires the AP77 coupled
  weak-rate gate artifact alongside the AP75 reproducibility bundle, then records
  the AP77 gate summary, including electron-bath controls, in the readiness
  ledger.
- **Key files:** `src/rabbit/validation/augmented_publication_readiness.py`,
  `scripts/run_augmented_publication_readiness_audit.py`,
  `tests/test_augmented_publication_readiness_audit.py`, registry-backed
  roadmap/capability docs.
- **Physics added/changed:** no new physics.  AP79 hardens the evidence chain:
  the final readiness audit now fails closed unless AP77 passed with no-QKE
  scope, no public dispatch, no production SMC, same-CL3 metadata-only controls,
  paired metadata-only/AP60 rows for every q value, metadata-only
  no-rate-correction rows, AP60 applied rows, exact comparison mode metadata,
  AP77 gate/input/case-row electron-bath provenance consistency, empty
  comparison/convergence violations, and q-ladder convergence rows for every
  adjacent q pair.
- **Scope boundary:** AP79 is readiness-ledger hardening only.  It does not
  provide promotion-grade full-span weak-rate convergence, public dispatch,
  production SMC validation, default collision feedback, or QKE.
- **Exit gate:** focused readiness tests reject tampered AP77 artifacts,
  promoted/QKE AP75/AP77 inputs, unknown plot claims, dispatch promotion, and
  stale bundle contracts, including mismatched AP77 electron-bath controls,
  while preserving JSON writer and CLI dry-run coverage.
- **Verification:** `PYTHONPATH=src pytest -q tests/test_augmented_publication_readiness_audit.py`
  passed locally.
- **Known red tests:** focused readiness tests first failed because the audit
  still accepted AP75-only inputs; they passed after the required
  `coupled_weak_rate_gate` input and AP77 fail-closed checks were added.  The
  AP79 self-review then found contradictory AP77 payloads could still pass when
  `passed=True` was paired with nonempty nested violations or missing q pairs;
  the validator now rejects those tampered artifacts.  AP77 electron-bath
  summary tests later failed until AP79 validated and reported AP77
  top-level/input/case-row electron controls.

---

### AP80-COUPLED-WEAK-RATE-CONVERGENCE  Profile-level AP77 weak-rate convergence

- **Status:** stage-recorded (historical; not current capability)
- **Scope:** add a deterministic AP80 artifact that wraps AP77 coupled
  weak-rate gate reports into named convergence profiles.  The smoke preset
  keeps q=(3,4) as the default; the explicit extended preset adds q=(3,4,5)
  for slower gates.
- **Key files:** `src/rabbit/validation/augmented_coupled_weak_rate_convergence.py`,
  `scripts/run_augmented_coupled_weak_rate_convergence.py`,
  `tests/test_augmented_coupled_weak_rate_convergence.py`, registry-backed
  roadmap/capability docs.
- **Physics added/changed:** no new weak-rate formula.  AP80 reuses the AP77
  AP65 coupled solve path and checks requested-vs-observed q ladders,
  metadata-only/AP60 paired rows, exact comparison mode metadata, adjacent
  q-ladder rows, and aggregate profile pass/fail status.
- **Scope boundary:** AP80 moves beyond a single smoke AP77 gate, but remains a
  diagnostic SciPy-first convergence artifact.  It does not provide
  promotion-grade full-span weak-rate convergence, public dispatch, production
  SMC validation, default collision feedback, or QKE.
- **Exit gate:** focused AP80 tests cover successful multi-profile aggregation,
  failed-profile violation recording, q-ladder mismatch rejection, duplicate
  case-row rejection, contradictory nested AP77 comparison/convergence
  pass-status rejection, profile validation, JSON writer output, and CLI dry-run
  profile surfacing.
- **Verification:** `PYTHONPATH=src pytest -q tests/test_augmented_coupled_weak_rate_convergence.py`
  passed locally.
- **Known red tests:** the new AP80 focused test first failed on missing module
  import, then passed after the AP80 module and CLI were added.

---

### AP81-SIX-MONOMIAL-COLLISION-FACTOR  Pauli-blocked 2-to-2 algebra landing

- **Status:** stage-recorded (historical; not current capability)
- **Scope:** replace the executable diagonal no-QKE 2-to-2 statistical factor
  with the quartic-cancelled six-monomial Pauli polynomial described in
  `neutrino_collision_term_PSTF.md`.  The signed monomials are `34`, `12`,
  `123`, `124`, `134`, and `234`; no quartic `1234` term is retained.
- **Key files:** `src/rabbit/collisions/deterministic_reference.py`,
  `src/rabbit/collisions/nu_e_scattering.py`,
  `src/rabbit/collisions/pair_processes.py`,
  `src/rabbit/jax/collisions_jax.py`,
  `src/rabbit/jax/nu_nu_scattering_jax.py`,
  `tests/test_deterministic_collision_reference.py`,
  `tests/test_pr_t3b_jax_operator_parity.py`, and
  `docs/audit/PR-AP81_six_monomial_collision_factor_2026-05-14.md`, plus
  `docs/audit/PR-AP81_all_nine_pairwise_nunu_bridge_2026-05-15.md` and
  `docs/audit/PR-AP6_offdiagonal_nunu_number_projection_2026-05-15.md`.
- **Physics added/changed:** deterministic `nu-e`/pair references, the
  deterministic pairwise diagonal `nu-nu` 2-to-2 reference, the staged NumPy
  AP19/AP33/AP35/AP41 diagonal `nu-nu` source bridge, legacy SciPy `nu-e`/pair
  operators, JAX `nu-e`/pair kernels, and JAX diagonal `nu-nu` kernels now
  share the same scalar occupation-number Pauli polynomial.  The staged source
  bridge uses that pairwise reference by default over all nine ordered
  `{nue,nuebar,nux}` bank pairs, including identical-bank self-scattering with
  Fierz factor 2 and same-bank number/energy-neutral projection before
  accumulation, plus explicit per-bank number closure and effective-`nu_x`
  weighted-energy closure projection; the older fixed-point redistribution
  helper plus an off-diagonal-only pairwise switch remain legacy comparison
  plumbing.  The AP6 radial follow-up now also projects all nine default radial
  diagonal `nu-nu` sources for particle-number conservation before AP18
  thermo/hierarchy feedback: identical-bank rows remain number/energy-neutral,
  while six off-diagonal rows are number-neutral and complete unordered pairs
  are energy-neutral while preserving their relative raw species
  energy-transfer differences.  A real AP55 LRS source-budget smoke reported
  `n_radial_nunu_sources=9`,
  `n_radial_nunu_number_projected_sources=9`,
  `n_radial_offdiagonal_nunu_number_projected_sources=6`,
  `n_radial_offdiagonal_nunu_pair_energy_projected_sources=6`,
  `n_radial_offdiagonal_nunu_pair_energy_projected_pairs=3`,
  `radial_nunu_max_abs_number_moment=4.828087799349512e-20`,
  `radial_offdiagonal_nunu_max_abs_number_moment=2.498747194400186e-20`, and
  `radial_offdiagonal_nunu_pair_max_abs_energy_residual=9.317362419797304e-20`.
  This is the
  collision statistical-factor algebra needed before the full PSTF angular
  kernel contraction table can be landed.
- **Scope boundary:** AP81 is executable scalar collision-factor algebra in
  existing kernels, but it is not a public-dispatch promotion and does not land
  full PSTF angular collision kernels, full anisotropic weak-rate integration,
  production SMC validation, or QKE.
- **Exit gate:** focused tests lock the monomial set/signs, absence of the
  quartic term, FD detailed balance, legacy operator sharing, JAX polynomial
  parity, JAX nu-e/pair parity, JAX diagonal `nu-nu` preflight behavior, and
  replay-stable non-equilibrium numeric values for fixed-quadrature `nu-e`,
  pair, pairwise diagonal `nu-nu` references, all-nine default pair-count
  diagnostics, identical-bank number/energy projection diagnostics, and the
  off-diagonal-only legacy switch, plus AP6 radial off-diagonal `nu-nu`
  number-neutral and unordered-pair energy-neutral projection diagnostics.
- **Verification:** `PYTHONPATH=src pytest -q tests/test_deterministic_collision_reference.py tests/test_pr_t3b_jax_operator_parity.py tests/test_pr_t3c_nu_nu_preflight.py`
  passed locally.
- **Known red tests:** focused deterministic tests first failed on missing
  six-monomial exports, then failed on placeholder numerical values; they
  passed after the shared polynomial and replay-stable values were landed.

---

### AP6-PSTF-LOCAL-SIX-MONOMIAL-CONTRACTION  Local angular contraction table

- **Status:** stage-recorded (historical; not current capability)
- **Scope:** add the reusable local PSTF angular contraction table needed
  between the AP81 scalar Pauli polynomial and future full HM angular
  kernels.  The table projects the `34`, `12`, `123`, `124`, `134`, and
  `234` scalar occupation products into mode space on deterministic angular
  grids.
- **Key files:** `src/rabbit/collisions/pstf_contractions.py`,
  `src/rabbit/collisions/__init__.py`, and
  `tests/test_pstf_collision_contractions.py`.
- **Physics added/changed:** the executable collision algebra now includes a
  mode-space local angular contraction surface matching
  `neutrino_collision_term_PSTF.md`: `K34`, `K12`, `K123`, `K124`, `K134`,
  and `K234` tensors are built from a weighted PSTF/S_N projection matrix and
  contracted against scalar occupation-number multipoles without introducing a
  quartic term.  This is a deterministic no-QKE scalar occupation contraction,
  not a density-matrix/QKE operator.
- **Scope boundary:** this lands the local angular projection/contraction
  table only.  Later AP6 entries add universal geometry, channel assembly,
  and the radial `p4` contraction; process-specific table generation and
  public collision-coupled full-BBN runtime promotion remain blocked.
- **Exit gate:** focused tests lock direct nodal-projection parity, FD
  isotropic detailed balance, absence of the `1234` quartic term, quadrupole
  projection without odd leakage, and rejection of unphysical nodal
  occupations.
- **Verification:** `PYTHONPATH=src pytest -q tests/test_pstf_collision_contractions.py`
  passed locally.
- **Known red tests:** focused AP6 contraction tests first failed on the
  missing `rabbit.collisions.pstf_contractions` module, then passed after the
  local contraction table and exports were landed.

---

### AP6-PSTF-UNIVERSAL-GEOMETRIC-KERNELS  Universal angular geometry table

- **Status:** stage-recorded (historical; not current capability)
- **Scope:** add the deterministic four-angle geometric tables described in
  `neutrino_collision_term_PSTF.md` for `Q=1`, `Q=mu_ij`, and
  `Q=mu_ij mu_kl`, using caller-supplied momentum-delta angular weights.
- **Key files:** `src/rabbit/collisions/pstf_contractions.py`,
  `src/rabbit/collisions/__init__.py`, and
  `tests/test_pstf_collision_contractions.py`.
- **Physics added/changed:** the AP6 collision-reference substrate now has
  executable universal `G0`, `G_mu`, and `G_mumu` tensors for the same
  `34`, `12`, `123`, `124`, `134`, and `234` monomial basis as the AP81
  Pauli polynomial.  These tensors carry the weighted PSTF projection over
  particle-1 angle and the remaining angular quadrature over particles 2-4.
- **Scope boundary:** channel-specific HM matrix-element coefficients and
  radial contraction are handled by later AP6 entries.  Promoted
  collision-coupled solve paths remain outside this landed substep.
- **Exit gate:** focused tests compare `G0`, `G_mu`, and `G_mumu` tensors
  against direct four-angle quadrature and reject malformed
  momentum-delta-weight tensors.
- **Verification:** `PYTHONPATH=src pytest -q tests/test_pstf_collision_contractions.py`
  passed locally.
- **Known red tests:** focused universal-kernel tests first failed on missing
  `build_universal_pstf_geometric_kernel_table`, then passed after the
  four-angle geometric table builder and exports were landed.

---

### AP6-PSTF-CHANNEL-KERNEL-ASSEMBLY  HM descriptor to K tensors

- **Status:** stage-recorded (historical; not current capability)
- **Scope:** combine universal `G0`/`G_mu`/`G_mumu` geometry with HM-style
  matrix-element descriptors for `Pi_ij` and `Pi_ij Pi_kl`, producing
  channel-specific `K` tensors for the six AP81 monomials.
- **Key files:** `src/rabbit/collisions/pstf_contractions.py`,
  `src/rabbit/collisions/__init__.py`, and
  `tests/test_pstf_collision_contractions.py`.
- **Physics added/changed:** implements the compact formulas
  `D^(ij) = E_i E_j G0 / c^2 - p_i p_j G^(ij)` and
  `D^((ij)(kl)) = E_iE_jE_kE_l G0/c^4 - E_iE_j p_kp_l G^(kl)/c^2
  - p_ip_jE_kE_l G^(ij)/c^2 + p_ip_jp_kp_l G^(ij,kl)`, then combines
  `eta` bilinear and `zeta m_e^2 c^2` mass terms into replay-stable `K`
  tensors.
- **Scope boundary:** radial prefactors are accepted only as caller-provided
  scalars.  The follow-up AP6 radial entry handles `p4` interpolation and
  deterministic `p2,p3` summation; process-specific table generation and
  promoted runtime coupling remain outside this stage.
- **Exit gate:** focused tests compare a descriptor-built channel kernel
  against direct `D` formula assembly and reject missing `G_mumu` inputs for
  requested bilinear terms.
- **Verification:** `PYTHONPATH=src pytest -q tests/test_pstf_collision_contractions.py`
  passed locally.
- **Known red tests:** focused channel-kernel tests first failed on missing
  `PSTFBilinearMatrixElementTerm`, then passed after descriptor dataclasses,
  `build_pstf_channel_kernel_table`, and exports were landed.

---

### AP6-PSTF-RADIAL-P4-CONTRACTION  Radial energy-conservation contraction

- **Status:** stage-recorded (historical; not current capability)
- **Scope:** add the radial-grid contraction layer that consumes
  channel-specific AP6 `K` tensors, sums over deterministic `p2,p3`
  quadrature weights, and evaluates the particle-4 occupation by linear
  interpolation at `p4 = E1 + E2 - E3`.
- **Key files:** `src/rabbit/collisions/pstf_contractions.py`,
  `src/rabbit/collisions/__init__.py`, and
  `tests/test_pstf_collision_contractions.py`.
- **Physics added/changed:** the PSTF collision-reference substrate now has
  an executable radial contraction for the six AP81 monomials after HM-style
  channel assembly.  The contraction applies the signed
  `34 - 12 + 123 + 124 - 134 - 234` source algebra, interpolates the
  particle-4 mode vector on the supplied energy grid, records the `p4`
  interpolation metadata, and zeros kinematically invalid radial tuples.
- **Scope boundary:** this is still an explicit no-QKE deterministic
  contraction table.  The following AP6 process-grid entry assembles radial
  channel grids from supplied process descriptors; adaptive quadrature design,
  promoted collision-coupled runtime wiring, and public full-BBN support remain
  outside this stage.
- **Exit gate:** focused tests lock the `p4` interpolation weight and index,
  all six signed monomial contractions against explicit einsum arithmetic,
  invalid-radial zeroing, a three-mode S2-style radial ladder, and result
  metadata/export contracts.
- **Verification:** `PYTHONPATH=src pytest -q tests/test_pstf_collision_contractions.py`
  passed locally.
- **Known red tests:** focused radial-grid tests first failed on missing
  `contract_pstf_channel_radial_grid`, then passed after the radial
  contraction result type, interpolation logic, contraction loop, and exports
  were landed.

---

### AP6-PSTF-RADIAL-CHANNEL-KERNEL-GRID  Process-grid channel assembly

- **Status:** stage-recorded (historical; not current capability)
- **Scope:** assemble radial channel `K` grids from supplied or normalized
  unit-direction momentum-delta angular weights, HM-style `Pi_ij`/`Pi_ij Pi_kl`
  descriptors, invariant radial-measure prefactors, and the AP6
  universal-geometry/channel builders.
- **Key files:** `src/rabbit/collisions/pstf_contractions.py`,
  `src/rabbit/collisions/__init__.py`, and
  `tests/test_pstf_collision_contractions.py`.
- **Physics added/changed:** adds the executable precomputation layer between
  one-tuple channel kernels and runtime radial contraction.  For each valid
  `(p1,p2,p3)` tuple it computes `p4 = E1 + E2 - E3`, interpolates the
  `p4` momentum grid, applies the invariant radial prefactor from
  `neutrino_collision_term_PSTF.md`, builds the needed `G_mu`/`G_mumu`
  tables, and stores the six signed monomial `K` tensors.  Invalid radial
  tuples are kept as zero kernels with explicit metadata.
- **Scope boundary:** this remains a caller-supplied process-grid substrate,
  not a physical process catalog or promoted collision-coupled solver path.
  QKE, density matrices, adaptive production quadrature, and public full-BBN
  dispatch remain out of scope.
- **Exit gate:** focused tests lock invariant-prefactor arithmetic,
  geometry/channel parity against direct one-tuple assembly, wrapped radial
- **Follow-up physics:** the descriptor-driven `pstf_radial` source builder now
  uses normalized unit-direction momentum-delta weights by default instead of a
  uniform four-angle factor, favoring vector-closed `e1+e2=e3+e4` angular
  quadrature tuples while keeping the angular integral normalized for
  smoke-scale stability.  The same builder exposes an opt-in
  `radial_gaussian` model that evaluates the full
  `p1 e1 + p2 e2 - p3 e3 - p4 e4` residual for p-dependent smoke studies.
- **Exit gate:** focused tests lock invariant-prefactor arithmetic,
  geometry/channel parity against direct one-tuple assembly, wrapped radial
  contraction parity, invalid-energy zeroing, normalized unit-direction
  momentum-delta weighting, static angular-geometry reuse for radial-independent
  momentum-delta tensors, opt-in p-dependent radial Gaussian weighting, real
  nonlinear `pstf_radial` smoke diagnostics, and
  export contracts.
- **Verification:** `PYTHONPATH=src pytest -q tests/test_pstf_collision_contractions.py`
  passed locally.
- **Known red tests:** focused process-grid tests first failed on missing
  `PSTFRadialInvariantPrefactorConfig`, then passed after the prefactor
  config, radial kernel-grid builder, metadata result type, contraction
  wrapper, and exports were landed.

---

### AP6-PSTF-PROCESS-CATALOG  Physical UR HM process descriptors

- **Status:** stage-recorded (historical; not current capability)
- **Scope:** map the already-landed scalar ultra-relativistic weak-process
  formulas into physical HM `Pi_ij Pi_kl` descriptors and feed those
  descriptors directly into the AP6 radial channel-grid builder.
- **Key files:** `src/rabbit/collisions/pstf_process_catalog.py`,
  `src/rabbit/collisions/__init__.py`, and
  `tests/test_pstf_process_catalog.py`.
- **Physics added/changed:** `build_ur_nue_scattering_process_descriptor(...)`
  reproduces the existing `nu-e` scattering matrix-element coefficients
  `(G_L^2+G_R^2)[(12)^2+(34)^2] + (G_L^2-G_R^2)[(14)^2+(23)^2]`;
  `build_ur_pair_annihilation_process_descriptor(...)` reproduces the pair
  coefficients `(G_L^2+G_R^2)[(13)^2+(24)^2] + (G_L^2-G_R^2)[(14)^2+(23)^2]`;
  and `build_ur_nunu_diagonal_process_descriptor(...)` reproduces the AP81
  pairwise diagonal no-QKE `nu-nu` kernel
  `epsilon_alpha_beta * [(12)^2 + (34)^2]`.  The catalog can optionally carry
  the `G_F^2` prefactor and provides a one-tuple descriptor evaluator for
  replay checks before radial-grid assembly.
- **Scope boundary:** this lands the physical UR process catalog needed by the
  staged PSTF radial kernel path.  It does not promote a default
  collision-coupled solver route, add QKE/density matrices, or close complete
  process coverage beyond the staged UR `nu-e`, pair, and pairwise diagonal
  `nu-nu` descriptors.
- **Exit gate:** focused tests compare catalog descriptors against the existing
  deterministic scalar HM `nu-e` and pair-process formulas, lock the AP81
  pairwise diagonal `nu-nu` descriptor coefficient, validate optional
  `G_F^2` prefactor handling, and check that descriptor-driven radial-grid
  assembly matches manual term assembly.
- **Verification:** `PYTHONPATH=src pytest -q tests/test_pstf_process_catalog.py`
  passed locally.
- **Known red tests:** focused catalog tests first failed on missing
  `rabbit.collisions.pstf_process_catalog`, then passed after the descriptor
  dataclass, process builders, one-tuple evaluator, radial-grid wrapper, and
  exports were landed.

---

### AP6-PSTF-PROCESS-CATALOG-SUPPORTED-SPECIES  Default supported-species catalog

- **Status:** stage-recorded (historical; not current capability)
- **Scope:** add a default AP6 weak-process descriptor catalog covering the
  supported species banks, with the UR catalog retained as an explicit
  compatibility/reference helper.
- **Key files:** `src/rabbit/collisions/pstf_process_catalog.py`,
  `src/rabbit/collisions/__init__.py`, and
  `tests/test_pstf_process_catalog.py`.
- **Physics added/changed:** `build_default_ur_weak_process_catalog(...)`
  enumerates `nu-e` elastic scattering, pair annihilation/creation, and every
  ordered no-QKE pairwise diagonal `nu-nu` target/partner descriptor for
  `{nue, nuebar, nux}`, including identical-bank self-scattering entries.
  The staged pairwise bridge now matches that all-nine coverage by default,
  while retaining an explicit off-diagonal-only legacy comparison switch.
  `build_ur_nunu_pairwise_process_descriptor(...)` assigns the explicit
  identical-species reference Fierz factor `2` and off-diagonal factor `1`,
  while `pstf_process_descriptor_key(...)` exposes stable catalog keys for
  artifact/report surfaces.  The newer
  `build_default_supported_weak_process_catalog(...)` keeps the same supported
  process/species enumeration but uses separate finite-mass HM elastic
  `nu-e_minus`/`nu-e_plus` descriptors and finite-mass HM pair-annihilation
  descriptors by default while leaving pairwise diagonal no-QKE `nu-nu` on the
  staged UR descriptors.
- **Scope boundary:** this closes supported-bank enumeration for the staged
  finite-mass electromagnetic plus UR diagonal-`nu-nu` descriptor catalog.  It
  does not add charged muon/tau processes, oscillation/QKE density matrices, a
  promoted solver route, or complete physical process coverage beyond the
  supported staged catalog.
- **Exit gate:** focused tests lock `nuebar` target labeling for pair
  annihilation, identical/off-diagonal `nu-nu` Fierz factors, default
  all-nine catalog alignment with the staged pairwise bridge plus the
  off-diagonal-only legacy switch,
  charge-split finite-mass elastic and finite-mass pair entries in the default
  supported catalog, optional identical-reference descriptors, stable keys and
  coverage count for all default descriptors, finite AP6 radial-source
  construction for every default catalog descriptor, package exports,
  Fermi-prefactor catalog normalization, and invalid species-list/model
  rejection.
- **Verification:** `PYTHONPATH=src pytest -q tests/test_pstf_process_catalog.py`
  passed locally.
- **Known red tests:** focused tests first failed because
  `build_default_ur_weak_process_catalog` was not exported.

---

### AP6-PSTF-FINITE-MASS-PAIR-DESCRIPTOR  Pair HM crossing mass terms

- **Status:** stage-recorded (historical; not current capability)
- **Scope:** add the finite-mass Hannestad-Madsen
  `nu + nubar <-> e+ + e-` process descriptor and route it into the current
  supported AP6 no-QKE catalog.
- **Key files:** `src/rabbit/collisions/pstf_contractions.py`,
  `src/rabbit/collisions/pstf_process_catalog.py`,
  `src/rabbit/collisions/__init__.py`,
  `tests/test_pstf_collision_contractions.py`, and
  `tests/test_pstf_process_catalog.py`.
- **Physics added/changed:** `PSTFMassQuarticMatrixElementTerm` extends the
  AP6 descriptor algebra with the `zeta m_e^4` term required by finite-mass
  crossed HM pair annihilation.  `build_finite_mass_pair_annihilation_process_descriptor(...)`
  implements the in-tree HM crossing convention
  `M2_nu_nubar_to_ee(s,t,u) = M2_nu_e_elastic(u,t,s)`: for a neutrino target
  it uses `s = 2 Pi_12` and `u-m_e^2 = -2 Pi_14`, while for a `nuebar`
  target particle 2 is the incoming neutrino and the crossed term uses
  `Pi_24`.  The default supported catalog now uses finite-mass pair
  descriptors for `{nue,nuebar,nux}` while retaining the explicit UR catalog
  mode for compatibility/reference checks.
- **Scope boundary:** this lands finite-mass pair matrix-element descriptors
  and their radial-kernel algebra only.  QED-corrected EOS feedback, live
  chemical-potential/electron thermodynamics evolution, promoted full-span
  live-RHS coupling, public dispatch, charged muon/tau channels, and QKE
  remain outside this step.
- **Exit gate:** focused tests first fail on the missing mass-quartic term and
  finite-mass pair descriptor, then compare one-tuple descriptor evaluation
  against the HM crossing closed form for neutrino-target and antineutrino-target
  cases, lock the default supported catalog's finite-mass pair entries, and
  verify public package exports.
- **Verification:** `PYTHONPATH=src pytest -q tests/test_pstf_collision_contractions.py`
  and `PYTHONPATH=src pytest -q tests/test_pstf_process_catalog.py` passed locally.
- **Known red tests:** focused tests first failed on the missing
  `PSTFMassQuarticMatrixElementTerm` /
  `build_finite_mass_pair_annihilation_process_descriptor` APIs.

---

### AP6-PSTF-FINITE-MASS-NUE-SCATTERING-DESCRIPTOR  Elastic HM mass terms

- **Status:** stage-recorded (historical; not current capability)
- **Scope:** add the finite-mass Hannestad-Madsen elastic `nu-e` process
  descriptor and route it into the current supported AP6 no-QKE catalog.
- **Key files:** `src/rabbit/collisions/pstf_process_catalog.py`,
  `src/rabbit/collisions/__init__.py`,
  `src/rabbit/validation/augmented_convergence.py`,
  `src/rabbit/transport/augmented_collision_bridge.py`,
  `tests/test_pstf_process_catalog.py`, and
  `tests/test_augmented_convergence.py`.
- **Physics added/changed:** `build_finite_mass_nue_scattering_process_descriptor(...)`
  implements the finite-mass elastic HM form
  `8[G_L^2(s-m_e^2)^2 + G_R^2(u-m_e^2)^2 + G_L G_R m_e^2(s+u-2m_e^2)]`
  with `s-m_e^2 = 2 Pi_12` and `u-m_e^2 = -2 Pi_14`, giving descriptor
  terms `32 G_L^2 Pi_12^2`, `32 G_R^2 Pi_14^2`, and
  `16 G_L G_R m_e^2(Pi_12 - Pi_14)`.  `nuebar` and `e_plus` each cross the squared chiral terms, while `nuebar + e_plus` returns to the uncrossed ordering.  The LRS `pstf_radial` route now builds its 15-process supported catalog with separate `e_minus` and `e_plus` finite-mass elastic entries, and the source diagnostics expose the
  active process labels.
- **Scope boundary:** this lands elastic `nu-e` finite-mass matrix-element
  terms only; finite-mass pair annihilation is landed separately in
  `AP6-PSTF-FINITE-MASS-PAIR-DESCRIPTOR`.  QED-corrected EOS feedback,
  live chemical-potential/electron thermodynamics evolution, production
  quadrature, public dispatch, and full-span live-RHS promotion remain
  outside this step.
- **Exit gate:** focused tests compare the one-tuple descriptor evaluator
  against the finite-mass HM closed form for uncrossed, crossed, and double-crossed charge/species cases,
  check finite radial-source construction for the new descriptor, lock the
  default supported catalog's finite-mass elastic entries plus UR compatibility
  mode, and check the LRS `pstf_radial` artifact reports the finite-mass
  elastic process labels.
- **Verification:** `PYTHONPATH=src pytest -q tests/test_pstf_process_catalog.py`
  passed locally; the focused LRS `pstf_radial` artifact test was run for this
  route update.
- **Known red tests:** focused tests first failed on the missing
  `build_finite_mass_nue_scattering_process_descriptor` /
  `build_default_supported_weak_process_catalog` exports.

---

### AP6-PSTF-RADIAL-CATALOG-PROVIDER  Default catalog radial provider route

- **Status:** stage-recorded (historical; not current capability)
- **Scope:** wire the default supported AP6 process descriptor catalog into
  the live PSTF radial moment provider and LRS `pstf_radial` validation route.
- **Key files:** `src/rabbit/collisions/pstf_process_catalog.py`,
  `src/rabbit/transport/augmented_collision_bridge.py`,
  `src/rabbit/validation/augmented_convergence.py`,
  `tests/test_pstf_process_catalog.py`,
  `tests/test_augmented_collision_bridge.py`, and
  `tests/test_augmented_convergence.py`.
- **Physics added/changed:** `pstf_process_particle_mode_labels(...)` maps
  default supported descriptors, including charge-split finite-mass elastic
  `nu-e_minus`/`nu-e_plus` and finite-mass pair entries,
  to the live neutrino and fixed or callback-provided electron/positron
  distribution labels required by the radial contraction.
  The radial moment provider now accepts fixed `F_modes` labels and dynamic
  callback-provided `F_modes` labels for electron-bath particles, precomputes
  the static radial channel grids once per configured process, and evaluates
  the full default 15-descriptor supported catalog in the LRS `pstf_radial`
  artifact route with charge-split finite-mass elastic `nu-e_minus`/`nu-e_plus`
  plus finite-mass pair matrix elements and a
  temperature-updated finite-mass zero-chemical-potential Fermi-Dirac
  electron/positron bath built from the live AP18 `T_gamma_MeV` payload and
  the `T_nu_e_MeV` energy scale.  The LRS route can now pass an explicit fixed
  `electron_chemical_potential_MeV` into charge-split `e_minus`/`e_plus`
  bath providers with opposite signs for opt-in charge-asymmetric experiments,
  while leaving live charge-asymmetry evolution out of scope.  The follow-up descriptor-aware radial
  kinematics path replaces the older smoke-only momentum fractions with
  label-derived momenta on the frozen dimensionless solver grid: neutrino legs
  use `p=q`, electron/positron legs use
  `p=sqrt(max(q^2-(m_e/T_nu_e0)^2,0))`, and the finite-mass bath is evaluated
  on the corresponding total-energy grid used by the radial kernel.
- **Scope boundary:** this is still a smoke-scale deterministic no-QKE radial
  source route.  The default LRS route uses zero chemical potential, and the
  opt-in chemical-potential provider primitive is not a live charge-asymmetry
  evolution model.  QED-corrected EOS feedback, nonzero chemical-potential
  evolution/coupling, and a full electron thermodynamics solve remain out of
  scope;
  full-span live-RHS promotion beyond this no-QKE HM catalog and public
  dispatch remain out of scope.
- **Exit gate:** focused tests lock descriptor-to-mode-label mapping,
  package exports, fixed, dynamic ultra-relativistic, and finite-mass
  electron-bath provider evaluation for default catalog descriptors, direct
  signed chemical-potential provider primitive evaluation, fail-closed
  static-grid reuse, finite-mass route sensitivity, and LRS `pstf_radial`
  artifacts reporting 18 radial moment sources plus finite-mass elastic/pair
  and diagonal `nu-nu` source-count diagnostics, the finite-mass `e_minus`,
  `e_plus`, pair-process diagnostics, and identical-bank `nu-nu`
  number/energy-neutral projection diagnostics.
- **Verification:** `PYTHONPATH=src pytest -q tests/test_pstf_process_catalog.py`
  and focused `tests/test_augmented_collision_bridge.py` /
  `tests/test_augmented_convergence.py` radial-catalog routes passed locally.
- **Known red tests:** the new provider test first failed because
  `pstf_process_particle_mode_labels`, electron-bath mode support, and the
  static-grid fail-closed check were not implemented.

---

### AP6-PSTF-CHARGE-SPLIT-RADIAL-BATH  Fixed-mu e-/e+ radial bath routing

- **Status:** stage-recorded (historical; not current capability)
- **Scope:** promote the already-landed signed chemical-potential FD bath
  primitive into the AP6 LRS `pstf_radial` source factory as an explicit
  fixed-`mu_e` route option.
- **Key files:** `src/rabbit/validation/augmented_convergence.py`,
  `src/rabbit/thermo/eos_photon_electron.py`,
  `tests/test_augmented_convergence.py`, WBS/state docs,
  `tests/test_eos_photon_electron_charge_neutrality.py`,
  `neutrino_collision_term_PSTF.md`, and generated capability docs.
- **Physics added/changed:** `_build_lrs_pstf_radial_moment_thermo_source(...)`
  now accepts `electron_chemical_potential_MeV`.  For nonzero values it keeps
  the neutral `e_pm` bath at zero chemical potential while routing `e_minus`
  through a `+mu_e` FD provider and `e_plus` through a `-mu_e` FD provider on
  the same total-energy radial grid.  The deterministic LRS artifact,
  budgeted live-RHS artifact, and live-vs-frozen source-policy artifact record
  and forward the same fixed `mu_e` value.  The same AP6 path now also has a
  deterministic direct-source sensitivity artifact,
  `build_augmented_lrs_pstf_radial_electron_mu_sensitivity_artifact(...)`,
  that evaluates the concrete radial source at fixed `mu_e` ladder values,
  records the returned `dQ_nue_pair_N`/`dQ_nux_bank_N` moments and radial
  diagnostics, records the projected kinetic-source amplitude
  `collision_dA_abs_max`, and reports both thermo-moment and kinetic-source
  amplitude deltas relative to the zero-chemical-potential row.  The
  finite-mass EOS module now also exposes
  `electron_number_density(...)`, `positron_number_density(...)`,
  `electron_charge_asymmetry_density(...)`, and
  `charge_neutral_electron_chemical_potential(...)`; the AP6 radial route can
  opt into `electron_chemical_potential_mode="charge_neutrality"` to derive
  the e-/e+ bath split from the current network mass fractions, `T_gamma`, and
  `eta`.  A direct charge-neutrality artifact compares that row against the
  zero-`mu_e` source evaluation, records the solved `mu_e`, and carries the
  same `collision_dA_abs_max` kinetic-source row/delta surface.  The
  charge-neutrality wrapper now preserves the radial `dA_modes` hierarchy
  payload returned by the underlying AP6 source, so the solved e-/e+ split can
  feed the staged kinetic RHS as well as the thermo `dQ` moments.
- **Scope boundary:** this is a fixed external bath parameter for deterministic
  staged experiments, plus a bounded charge-neutrality algebraic closure for
  direct source evaluation.  It is not live charge-asymmetry evolution, not a
  QED-corrected electron thermodynamics solve, not public dispatch, and not
  QKE.
- **Exit gate:** focused route-construction tests verify that nonzero fixed
  `mu_e` splits `e_minus` and `e_plus` FD occupations while leaving the neutral
  `e_pm` bath at the zero-chemical-potential value.  Source-sensitivity tests
  verify that the real AP6 radial callback returns finite source moments,
  nonzero fixed-`mu_e` deltas, concrete kinetic-source amplitudes, and the
  `collision_dA_abs_max` delta surface, plus JSON writer output.  Charge-neutrality
  tests verify finite-mass e-/e+ number-density splitting, chemical-potential
  root solving, AP6 radial-source diagnostics for the solved `mu_e`, radial
  `dA_modes` payload preservation, direct artifact kinetic-source amplitudes,
  and
  fail-closed rejection of an ambiguous fixed-`mu_e` offset in
  charge-neutrality mode.
- **Verification:** focused `tests/test_augmented_convergence.py` radial route,
  source-sensitivity, and non-live `pstf_radial` subset tests passed locally
  for this routing update.
- **Known red tests:** the route test first failed on the missing
  `electron_chemical_potential_MeV` argument, and the subset then caught the
  missing artifact-surface signature.  The charge-neutral kinetic-payload
  regression first failed because the wrapper rebuilt
  `Augmented3TCollisionThermoSource` without forwarding `dA_modes`.  The direct
  source kinetic-observable regression first failed with missing
  `kinetic_source` rows before `collision_dA_abs_max` was surfaced from
  `source.dA_modes`.

---

### AP6-SIGNED-MU-PAIR-BLOCKING  Staged electromagnetic pair-process mu_e blocking

- **Status:** stage-recorded (historical; not current capability)
- **Scope:** thread signed electron chemical potential into the staged
  ultra-relativistic electromagnetic pair-process reference and into the
  AP18/AP40 collision-source callback contract.
- **Key files:** `src/rabbit/collisions/deterministic_reference.py`,
  `src/rabbit/transport/augmented_collision_bridge.py`,
  `src/rabbit/transport/augmented_typeI_weak_network.py`,
  `tests/test_deterministic_collision_reference.py`,
  `tests/test_augmented_collision_bridge.py`,
  `tests/test_augmented_typeI_weak_network_3t_solve.py`,
  `tests/test_augmented_typeI_nonlrs_weak_network_solve.py`,
  WBS/state docs, and generated capability docs.
- **Physics added/changed:** pair annihilation blocking now evaluates the
  final-state positron/electron occupations as `f_e+(y, -mu_e/T)` and
  `f_e-(y, +mu_e/T)`.  The LRS electron-pair thermo source, combined source,
  and angular electromagnetic bridge accept the current
  `electron_chemical_potential_MeV` and report explicit signed-mu blocking
  diagnostics.  The LRS/source-only non-LRS/nonlinear non-LRS 3T collision
  source evaluator now forwards both the current electron chemical potential
  and mode into explicit source callbacks.
- **Scope boundary:** this lands signed-mu Pauli blocking for the staged
  ultra-relativistic pair-process source bridge.  It is not the finite-mass
  angular electromagnetic kernel, does not promote public dispatch, and keeps
  QKE out of scope.  AP6-SIGNED-MU-SCATTERING-BLOCKING follows by applying the
  same signed bath convention to the staged ultra-relativistic scattering
  bridge.
- **Exit gate:** focused deterministic tests verify FD detailed balance at
  nonzero `mu_e`, finite input rejection, and a nonzero distorted-source change
  relative to zero `mu_e`.  Bridge tests lock diagnostics on electron-pair,
  combined, and angular source surfaces.  3T solve tests lock forwarding of
  `electron_chemical_potential_MeV` and `electron_chemical_potential_mode` into
  collision-source callbacks.
- **Verification:** focused deterministic, bridge, AP18 LRS callback-forwarding,
  and AP40 non-LRS callback-forwarding tests passed locally before the final
  focused bundle.
- **Known red tests:** the deterministic regression first failed because
  `evaluate_pair_annihilation_reference(...)` did not accept
  `electron_chemical_potential_MeV`; the 3T callback regression first failed
  because collision source callbacks did not receive the electron-mu payload.

---

### AP6-SIGNED-MU-SCATTERING-BLOCKING  Staged electromagnetic nu-e mu_e blocking

- **Status:** stage-recorded (historical; not current capability)
- **Scope:** thread signed electron chemical potential into the staged
  ultra-relativistic `nu-e` scattering reference and the electromagnetic
  scattering portions of the monopole, angular, electron-pair, and combined
  source bridges.
- **Key files:** `src/rabbit/collisions/deterministic_reference.py`,
  `src/rabbit/transport/augmented_collision_bridge.py`,
  `tests/test_deterministic_collision_reference.py`,
  `tests/test_augmented_collision_bridge.py`, WBS/state docs, and generated
  capability docs.
- **Physics added/changed:** `evaluate_nue_scattering_reference(...)` now
  evaluates the initial and final electron bath occupations as
  `f_e(y, +mu_e/T)`.  The staged electromagnetic source bridge forwards the
  current `electron_chemical_potential_MeV` into scattering and pair-process
  references and records separate scattering/pair signed-mu diagnostics.
- **Scope boundary:** this is still the staged ultra-relativistic
  electromagnetic scattering reference.  It is not the finite-mass angular
  electromagnetic kernel, does not add QED/tensor response, does not make the
  source default, does not promote public dispatch, and keeps QKE out of scope.
- **Exit gate:** focused deterministic tests verify FD detailed balance at
  nonzero `mu_e`, finite input rejection, positive/negative `mu_e` source
  changes, and antineutrino label preservation.  Bridge tests lock signed-mu forwarding/effect on the standalone scattering bridge and diagnostics on electron-pair,
  combined, and angular source surfaces.
- **Verification:** focused deterministic and bridge tests passed locally before
  the final focused bundle.
- **Known red tests:** deterministic scattering first rejected
  `electron_chemical_potential_MeV`; bridge tests then failed on missing
  scattering signed-mu diagnostics before forwarding and diagnostics were
  wired.

---

### AP6-PSTF-SIGNED-MU-EOS-THERMO-FEEDBACK  Fixed-mu e-/e+ 3T EOS coupling

- **Status:** stage-recorded (historical; not current capability)
- **Scope:** promote the fixed signed-`mu_e` finite-mass e-/e+ EOS from a
  radial-source diagnostic into an opt-in LRS 3T thermodynamics/Hubble
  feedback path.
- **Key files:** `src/rabbit/thermo/eos_photon_electron.py`,
  `src/rabbit/thermo/nudec_coupled.py`,
  `src/rabbit/transport/augmented_typeI_weak_network.py`,
  `tests/test_eos_photon_electron_charge_neutrality.py`, and
  `tests/test_augmented_typeI_weak_network_3t_solve.py`.
- **Physics added/changed:** the finite-mass electron/positron number-density
  split now uses the single-charge spin degeneracy normalization, locked
  against the high-temperature massless Fermi-Dirac limit.  The photon/electron
  EOS exposes signed-`mu_e` plasma energy density, pressure, and fixed-`mu_e`
  `d rho / dT` helpers.  `hubble_3T(...)`, `coupled_3T_rhs(...)`, and
  `coupled_3T_rhs_from_collision_moments(...)` accept an explicit
  `electron_chemical_potential_MeV`; for nonzero values the electromagnetic
  bath uses finite-mass signed-`mu_e` e-/e+ energy/pressure and the LRS 3T
  solve forwards that value into the Hubble and photon-temperature RHS.
  The default zero-chemical-potential path remains canonical.
- **Scope boundary:** this is fixed-`mu_e` finite-mass e-/e+ thermodynamics
  feedback in the staged LRS 3T helper.  The signed-`mu_e` helper now uses the
  finite-mu-scaled isotropic QED correction from
  `qed_delta_rho_with_electron_mu(...)`; later AP4/AP6 follow-ups add evolved
  charge-asymmetry states and an opt-in exact finite-mu scalar QED thermo mode.
  Anisotropic/tensor QED, public dispatch, promotion-grade full-span
  collision-coupled BBN, and QKE remain out of scope.
- **Exit gate:** focused EOS tests lock the corrected number-density
  normalization, signed-`mu_e` EOS symmetry and zero-`mu_e` canonical
  reduction.  Focused 3T tests lock signed-`mu_e` Hubble response,
  photon-temperature RHS response, and the LRS 3T solve forwarding the fixed
  `electron_chemical_potential_MeV` into every Hubble evaluation.
- **Verification:** `PYTHONPATH=src pytest -q tests/test_eos_photon_electron_charge_neutrality.py` and
  focused signed-`mu_e` rows in
  `tests/test_augmented_typeI_weak_network_3t_solve.py` passed locally.
- **Known red tests:** the EOS test first failed on missing signed-`mu_e`
  plasma helpers and the corrected number-density normalization; the 3T tests
  first failed because `hubble_3T(...)` and the LRS 3T solve did not accept
  `electron_chemical_potential_MeV`.

---

### AP6-PSTF-FINITE-MU-QED-EOS-SCALING  Signed-mu isotropic QED EOS feedback

- **Status:** stage-recorded (historical; not current capability)
- **Scope:** connect the already-landed signed-`mu_e` finite-mass e-/e+ EOS to
  the repository's isotropic O(e^2)+O(e^3) QED plasma correction instead of
  reusing the zero-`mu_e` suppression factor.
- **Key files:** `src/rabbit/thermo/eos_photon_electron.py`,
  `src/rabbit/config/backend_capabilities.py`,
  `src/rabbit/config/feature_capabilities.py`, and
  `tests/test_eos_photon_electron_charge_neutrality.py`.
- **Physics added/changed:** `qed_delta_rho_with_electron_mu(...)` evaluates the
  existing isotropic thermal-QED correction with the finite-mass
  `rho(e-) + rho(e+)` bath at signed chemical potential.  The zero-`mu_e` path
  remains exactly compatible with `qed_delta_rho(...)`, the correction is even
  in `mu_e`, and `rho_plasma_with_electron_mu(...)` plus
  `pressure_plasma_with_electron_mu(...)` now use that finite-mu QED value.
  At `T = 0.8 MeV`, `mu_e = 0.2 MeV`, the concrete smoke value is
  `qed_delta_rho_with_electron_mu = 0.0013502932161548037 MeV^4` versus
  `qed_delta_rho = 0.001314159794007091 MeV^4`.
- **Scope boundary:** this is an isotropic finite-mu scaling of the existing QED
  correction.  The opt-in exact scalar QED follow-up below adds scalar
  occupation-level exact pressure/energy corrections.  This stage is not an
  anisotropic/tensor QED EOS response, not public dispatch, not
  promotion-grade full-span BBN, and not QKE.
- **Exit gate:** focused EOS tests lock zero-`mu_e` canonical reduction,
  sign-even finite-`mu_e` behavior, a nonzero finite-mu QED increment, and
  use of that increment in signed-`mu_e` plasma energy density and pressure.
- **Verification:** `PYTHONPATH=src pytest -q tests/test_eos_photon_electron_charge_neutrality.py`
  passed locally with 8 tests.
- **Known red tests:** the finite-mu QED test first failed because
  `qed_delta_rho_with_electron_mu(...)` did not exist and the signed-`mu_e`
  plasma EOS still used the zero-`mu_e` QED correction.

---

### AP4-EXACT-FINITE-MU-SCALAR-QED-THERMO  Opt-in scalar QED EOS/3T solve-shell mode

- **Status:** stage-recorded (historical; not current capability)
- **Scope:** add an opt-in scalar finite-`mu_e` QED pressure/energy correction
  mode for the electromagnetic plasma thermo entrypoints and staged 3T
  Hubble/RHS solve shells.
- **Key files:** `src/rabbit/thermo/qed_eos_exact.py`,
  `src/rabbit/thermo/eos_photon_electron.py`,
  `src/rabbit/thermo/nudec_coupled.py`,
  `src/rabbit/transport/augmented_typeI_weak_network.py`,
  `tests/test_qed_eos_exact_finite_mu.py`,
  `tests/test_augmented_typeI_weak_network_3t_solve.py`,
  `tests/test_augmented_typeI_nonlrs_weak_network_solve.py`,
  `tests/test_augmented_typeI_nonlrs_nonlinear_weak_network_3t.py`, and
  `docs/audit/PR-AP4_exact_finite_mu_scalar_qed_2026-05-15.md`.
- **Physics added/changed:** the exact QED EOS integrals now accept a signed
  electron chemical potential by replacing the zero-`mu_e` summed occupation
  with `f_e(E, mu_e) + f_pos(E, mu_e)`.  The `exact_finite_mu_scalar` mode
  exposes exact O(e^2)+O(e^3) scalar QED pressure and energy corrections
  through `qed_delta_rho_with_electron_mu(...)`,
  `qed_delta_pressure_with_electron_mu(...)`,
  `rho_plasma_with_electron_mu(...)`,
  `pressure_plasma_with_electron_mu(...)`, `drho_dT_plasma_with_electron_mu(...)`,
  `hubble_3T(...)`, and `coupled_3T_rhs(...)`.  The LRS, source-only
  non-LRS, and nonlinear non-LRS 3T solve shells expose
  `qed_correction_model`, pass the selected model through every staged Hubble,
  standard thermo RHS, collision-moment thermo RHS, and charge-neutral plasma
  derivative override, and record `qed_correction_model` plus
  `qed_correction_contract` in result metadata.  The default remains the
  previous finite-mu-scaled isotropic correction.
- **Scope boundary:** this is an isotropic plasma-frame scalar QED EOS mode in
  the staged scalar 3T shells.  It does not implement anisotropic/tensor QED
  response, does not promote the exact scalar mode into public/default full
  BBN dispatch, does not provide promotion-grade full-span coupled-solver
  validation, and does not change QKE scope.
- **Numeric smoke evidence:** at `T = 0.8 MeV`, `mu_e = 0.2 MeV`,
  `delta_rho_qed_exact_with_electron_mu = -0.001545790523714566 MeV^4` and
  `delta_P_qed_exact_with_electron_mu = -0.00045775883349570345 MeV^4`.
- **Exit gate:** focused tests lock zero-`mu_e` reduction to the exact QED
  helper, evenness under `mu_e -> -mu_e`, finite nonzero signed-`mu_e`
  corrections, selected EOS pressure/energy assembly, positive fixed-`mu_e`
  `d rho / dT`, direct 3T thermo entrypoint acceptance, LRS/source-only
  non-LRS/nonlinear non-LRS 3T solve-shell forwarding into Hubble/RHS
  evaluations, result metadata, and unknown-model rejection.
- **Verification:** `PYTHONPATH=src pytest -q tests/test_qed_eos_exact_finite_mu.py`,
  `PYTHONPATH=src pytest -q tests/test_eos_photon_electron_charge_neutrality.py tests/test_qed_eos_exact_finite_mu.py`, and
  `PYTHONPATH=src pytest -q tests/test_nu_nu_3t_equilibration.py tests/test_jax_thermo_provider.py tests/test_species_boltzmann_bridge.py tests/test_isotropic_decoupling_skeleton.py tests/test_qed_eos_exact_finite_mu.py tests/test_eos_photon_electron_charge_neutrality.py`, and
  `PYTHONPATH=src pytest -q tests/test_augmented_typeI_weak_network_3t_solve.py tests/test_augmented_typeI_nonlrs_weak_network_solve.py tests/test_augmented_typeI_nonlrs_nonlinear_weak_network_3t.py -k "qed_correction_model or exact_scalar_qed"`
  passed locally.
- **Known red tests:** the focused exact-scalar QED tests first failed at
  collection because `qed_delta_pressure_with_electron_mu(...)` and the
  finite-`mu_e` exact QED helper functions did not exist.  The solve-shell
  forwarding tests then failed because the staged 3T solvers did not accept
  `qed_correction_model`.

---

### AP6-PSTF-CHARGE-NEUTRAL-3T-EOS-FEEDBACK  Algebraic charge-neutral e-/e+ 3T coupling

- **Status:** stage-recorded (historical; not current capability)
- **Scope:** extend the LRS 3T thermodynamics/Hubble helper from fixed
  signed-`mu_e` feedback to an opt-in algebraic charge-neutrality mode that
  solves `n_e- - n_e+ = n_b Z/A` from the current network mass fractions,
  `T_gamma`, and `eta` inside the SciPy RHS.
- **Key files:** `src/rabbit/transport/augmented_typeI_weak_network.py`,
  `src/rabbit/thermo/nudec_coupled.py`,
  `tests/test_augmented_typeI_weak_network_3t_solve.py`,
  `src/rabbit/config/backend_capabilities.py`, and
  `src/rabbit/config/feature_capabilities.py`.
- **Physics added/changed:** `run_augmented_lrs_collisionless_weak_network_3T_solve(...)`
  now accepts `electron_chemical_potential_mode="charge_neutrality"` and
  rejects ambiguous simultaneous fixed `mu_e` input.  In that mode it computes
  the baryonic positive charge density from the current PRIMAT mass fractions,
  solves the finite-mass e-/e+ charge-neutral chemical potential, feeds that
  `mu_e` into `hubble_3T(...)`, and passes charge-neutral-path plasma
  `d rho / dT` and base-cooling overrides into the 3T photon-temperature RHS.
  The result object records `electron_chemical_potential_mode`,
  `electron_chemical_potential_MeV`, and
  `electron_chemical_potential_MeV_final` so concrete solved values are
  available for validation artifacts.
- **Scope boundary:** this is algebraic charge-neutral finite-mass e-/e+ EOS
  feedback in the staged LRS 3T helper using the finite-mu-scaled isotropic QED
  correction.  It is not an independent evolved electron charge-asymmetry
  state, not an exact
  finite-`mu_e`/tensor thermal-QED correction, not public dispatch, not a
  promoted collision-coupled full-BBN solve, and not QKE.
- **Exit gate:** focused 3T tests lock charge-neutral mode validation,
  forwarding of solved `mu_e` into Hubble, finite positive charge-density
  targets from the current abundances, the charge-neutral contract string, and
  result-level `mu_e` history/final metadata.
- **Verification:** `PYTHONPATH=src pytest -q tests/test_augmented_typeI_weak_network_3t_solve.py -k "electron_mu or charge_neutral"`
  passed locally.
- **Known red tests:** the first charge-neutral path test was too expensive
  when run through a real LSODA solve; it was narrowed to a deterministic
  RHS-path smoke by monkeypatching `solve_ivp`.  A later metadata extension
  first failed because the new fields were attached to the wrong result
  dataclass before being moved to the LRS 3T result.

---

### AP4-NLRS-ELECTRON-EOS-3T-FEEDBACK  Non-LRS fixed/charge-neutral e-/e+ 3T coupling

- **Status:** stage-recorded (historical; not current capability)
- **Scope:** extend the staged finite-mass e-/e+ 3T EOS feedback path from the
  LRS helper into the source-only non-LRS 3T solve, nonlinear non-LRS 3T solve,
  and LRS/non-LRS angular-collision wrapper surfaces.
- **Key files:** `src/rabbit/transport/augmented_typeI_weak_network.py`,
  `tests/test_augmented_typeI_nonlrs_weak_network_solve.py`,
  `tests/test_augmented_typeI_nonlrs_nonlinear_weak_network_3t.py`, and
  `tests/test_augmented_typeI_weak_network_3t_solve.py`.
- **Physics added/changed:** `run_augmented_nonlrs_source_collisionless_weak_network_3T_solve(...)`
  and `run_augmented_nonlrs_nonlinear_collisionless_weak_network_3T_solve(...)`
  now accept `electron_chemical_potential_MeV` and
  `electron_chemical_potential_mode="fixed" | "charge_neutrality"`.  Both
  solvers compute the per-RHS finite-mass e-/e+ `mu_e`, feed it into
  `hubble_3T(...)`, pass fixed- or charge-neutral-path plasma overrides into
  the 3T photon-temperature RHS, and record result-level `mu_e` histories and
  final values.  The LRS, non-LRS source-only, and nonlinear non-LRS
  angular-collision 3T wrappers forward the same controls to their underlying
  SciPy solves.
- **Scope boundary:** this closes the staged electron-EOS routing gap for the
  currently implemented AP4/AP64/AP65 3T shells using the finite-mu-scaled
  isotropic QED correction.  It is not a non-LRS independent evolved electron
  charge-asymmetry state, not an exact finite-`mu_e`/tensor thermal-QED
  correction, not public dispatch, not promotion-grade full-span
  collision-coupled BBN, and not QKE.
- **Exit gate:** focused tests lock fixed-`mu_e` and algebraic
  charge-neutrality forwarding into non-LRS Hubble evaluations, finite positive
  charge-density targets from current abundances, contract-string selection,
  result-level `mu_e` history/final metadata, nonlinear non-LRS signed-`mu_e`
  Hubble response, and wrapper-level electron-mode pass-through.
- **Verification:** `PYTHONPATH=src pytest -q tests/test_augmented_typeI_nonlrs_weak_network_solve.py -k "electron_mu or charge_neutral"`,
  `PYTHONPATH=src pytest -q tests/test_augmented_typeI_nonlrs_nonlinear_weak_network_3t.py -k "couples_transport_thermo_network_and_rates or nonlinear_angular_collision_wrapper_forwards_electron_mu_mode"`,
  and `PYTHONPATH=src pytest -q tests/test_augmented_typeI_weak_network_3t_solve.py -k "angular_collision_3t_wrapper_forwards_electron_mu_mode or electron_mu or charge_neutral"`
  passed locally.
- **Known red tests:** the first source-only fixed-`mu_e` route test was too
  expensive through a real LSODA solve and was narrowed to a deterministic
  RHS-path smoke by monkeypatching `solve_ivp`; the wrapper pass-through test
  first failed because the LRS angular wrapper did not expose electron-mode
  arguments.

---

### AP4-CHARGE-NEUTRAL-TOTAL-DERIVATIVE-ELECTRON-THERMO  Network-coupled charge-neutral e-/e+ energy feedback

- **Status:** stage-recorded (historical; not current capability)
- **Scope:** upgrade the opt-in charge-neutral finite-mass e-/e+ 3T EOS path
  from algebraic `mu_e(T_gamma, X)` evaluation only to a total-derivative
  photon-bath energy update that includes the current network derivative.
- **Key files:** `src/rabbit/transport/augmented_typeI_weak_network.py`,
  `tests/test_augmented_typeI_weak_network_3t_solve.py`,
  `tests/test_augmented_typeI_nonlrs_weak_network_solve.py`,
  `src/rabbit/config/backend_capabilities.py`,
  `src/rabbit/config/feature_capabilities.py`, and
  `docs/audit/PR-AP4_charge_neutral_total_derivative_electron_thermo_2026-05-15.md`.
- **Physics added/changed:** the LRS, source-only non-LRS, and nonlinear
  non-LRS 3T RHS shells now evaluate the weak/network derivative before the
  photon-temperature RHS when charge-neutral electron thermodynamics is active.
  The charge-neutral plasma override adds
  `d rho_em / d(n_e- - n_e+) * d[n_b sum_i Z_i X_i/A_i]/dN` to the
  photon-bath energy equation, using the finite-mass signed-`mu_e` plasma EOS
  and the current PRIMAT abundance derivative.  Result metadata records the
  final energy-derivative contribution and its equivalent temperature-RHS
  correction.
- **Scope boundary:** this stage was still an algebraic charge-neutrality
  closure.  Later AP4 evolved-state follow-ups close the LRS/source-only
  non-LRS/nonlinear non-LRS state-variable pieces; exact finite-`mu_e`/tensor
  thermal-QED calculation, public forward dispatch, promotion-grade full-span
  collision-coupled BBN, and QKE remain out of scope.
- **Numeric smoke evidence:** a direct helper smoke at `T_gamma = 0.8 MeV`,
  `eta = 1e-4`, and a neutron-to-proton test derivative gives
  `mu_e = 5.407278613347443e-05 MeV`, static
  `plasma_dT_base_dN = -0.7857261820041459`, total-derivative
  `plasma_dT_base_dN = -0.78572618257828`,
  `plasma_charge_asymmetry_drho_dN_MeV4 = 2.1108534095619534e-09`, and
  `plasma_charge_asymmetry_dT_correction_dN = -5.741341845659509e-10`.
- **Exit gate:** focused tests lock the nonzero network-derivative energy
  contribution, the updated charge-neutral contract strings, and zero
  correction under deterministic fake-constant-`mu_e` solve-path smokes.
- **Verification:** targeted AP4 charge-neutral total-derivative tests passed
  locally.  A real short physical charge-neutral solve was intentionally not
  used as the default gate because the existing per-RHS charge-neutral root
  solve remains too expensive for smoke-scale CI.
- **Known red tests:** the focused tests first failed because the helper did
  not accept `dX_dN` and the charge-neutral contract strings still described
  the algebraic-only path.

---

### AP4-CHARGE-NEUTRAL-MU-LINEAR-RESPONSE  Finite-mass charge-susceptibility mu_e solve

- **Status:** stage-recorded (historical; not current capability)
- **Scope:** remove the BBN-scale charge-neutral `mu_e` bracket-solve hot path
  from the staged charge-neutral e-/e+ 3T thermodynamics route.
- **Key files:** `src/rabbit/thermo/eos_photon_electron.py`,
  `tests/test_eos_photon_electron_charge_neutrality.py`, and
  `docs/audit/PR-AP4_charge_neutral_mu_linear_response_2026-05-15.md`.
- **Physics added/changed:** the finite-mass e-/e+ EOS now exposes
  `electron_charge_asymmetry_susceptibility(T_MeV) =
  d(n_e- - n_e+)/dmu_e |_{mu_e=0}` from the same Gauss-Laguerre quadrature used
  for the electron and positron densities.  For BBN-scale charge targets whose
  inferred `mu_e` lies in the linear-response regime, the charge-neutral
  chemical-potential solve returns `target / susceptibility`; larger targets
  keep the existing finite-mass bracketed bisection fallback.
- **Scope boundary:** this is a physics-based fast path for the algebraic
  charge-neutrality solve.  Later AP4 evolved-state follow-ups close the
  LRS/source-only non-LRS/nonlinear non-LRS state-variable pieces; exact
  finite-`mu_e`/tensor QED, public dispatch, promotion-grade full-span BBN, and
  QKE remain out of scope.
- **Numeric smoke evidence:** an actual LRS charge-neutral 3T solve with
  `N_span=(0, 1e-7)`, `N_q=3`, `N_mu=8`, `WeakQuadrature(4, 4)`,
  `method=RK23`, `rtol=1e-4`, and `atol=1e-8` completed with `success=True`,
  `nfev=10598`, elapsed `42.4977612849907 s`,
  `mu_final=3.2984396489273045e-10 MeV`,
  `electron_charge_asymmetry_drho_dN_MeV4_final=-1.542004531086532e-20`,
  and `electron_charge_asymmetry_dT_correction_dN_final=4.1941224989310315e-21`.
- **Exit gate:** focused EOS tests lock susceptibility against a finite
  difference derivative, the no-residual-call linear-response path for a
  BBN-scale target, and the existing charge-neutral solve target residual.
- **Known red tests:** the focused tests first failed because
  `electron_charge_asymmetry_susceptibility` did not exist.

---

### AP4-CHARGE-NEUTRAL-EVOLVED-LRS-STATE  LRS charge-asymmetry state variable

- **Status:** stage-recorded (historical; not current capability)
- **Scope:** upgrade the staged LRS charge-neutral finite-mass e-/e+ 3T EOS
  path so charge neutrality is carried as an evolved
  `electron_charge_asymmetry_density_MeV3` state rather than only recomputed as
  algebraic metadata.
- **Key files:** `src/rabbit/transport/augmented_typeI_weak_network.py`,
  `tests/test_augmented_typeI_weak_network_3t_solve.py`, and
  `docs/audit/PR-AP4_charge_neutral_evolved_lrs_state_2026-05-15.md`.
- **Physics added/changed:** charge-neutral LRS 3T solves now seed an electron
  charge-asymmetry density from
  `eta * n_gamma(T_gamma) * sum_i Z_i X_i / A_i`, append it to the SciPy state
  vector, use the evolved state as the target passed to
  `charge_neutral_electron_chemical_potential(...)`, and evolve it with
  `3 n_Q dT_gamma/(T_gamma dN) + eta n_gamma d(sum_i Z_i X_i/A_i)/dN`.
  Result metadata records the state history, final value, and
  `charge_neutral_positive_charge_density_evolved_v1`.
- **Scope boundary:** this initially closed the LRS piece of the independent
  charge-asymmetry blocker.  The follow-up below closes the source-only
  non-LRS and nonlinear non-LRS state pieces; exact finite-`mu_e`/tensor QED,
  public dispatch, production SMC, promotion-grade full-BBN, and QKE remain out
  of scope.
- **Numeric smoke evidence:** at `T_gamma = 0.8 MeV`, `eta = 1e-4`,
  `X = phase1_to_phase2(0.13)`, `dT_gamma/dN = -0.02`, and a neutron-to-proton
  test derivative, the helper returns
  `electron_charge_asymmetry_density_MeV3 = 1.0850368568231983e-05`,
  `d(electron_charge_asymmetry_density_MeV3)/dN = 2.304144359748114e-06`, and
  `n_gamma = 0.1247168800946205`.
- **Exit gate:** focused tests lock the charge-density derivative formula,
  result state-history/final metadata, LRS charge-neutral state contract, and
  the fact that the charge-neutral `mu_e` callback sees the evolved state
  target.
- **Verification:** `PYTHONPATH=src pytest -q tests/test_augmented_typeI_weak_network_3t_solve.py -k "charge_asymmetry_state"`,
  `PYTHONPATH=src pytest -q tests/test_augmented_typeI_weak_network_3t_solve.py -k "charge_neutral or electron_mu or fixed_electron"`,
  and `PYTHONPATH=src pytest -q tests/test_augmented_typeI_weak_network_3t_solve.py`
  passed locally.
- **Known red tests:** the focused tests first failed because the derivative
  helper and result state fields did not exist.  A real finite-mass
  charge-neutral short solve exceeded the interactive smoke budget and remains
  a runtime caveat rather than a gate for this stage.

---

### AP4-CHARGE-NEUTRAL-EVOLVED-NONLRS-STATES  Non-LRS charge-asymmetry state variables

- **Status:** stage-recorded (historical; not current capability)
- **Scope:** upgrade the staged source-only non-LRS and nonlinear non-LRS
  charge-neutral finite-mass e-/e+ 3T EOS paths so charge neutrality is carried
  as an evolved `electron_charge_asymmetry_density_MeV3` ODE state rather than
  only recomputed as algebraic metadata.
- **Key files:** `src/rabbit/transport/augmented_typeI_weak_network.py`,
  `tests/test_augmented_typeI_nonlrs_weak_network_solve.py`,
  `tests/test_augmented_typeI_nonlrs_nonlinear_weak_network_3t.py`, and
  `docs/audit/PR-AP4_nonlrs_charge_neutral_evolved_state_2026-05-15.md`.
- **Physics added/changed:** source-only and nonlinear non-LRS 3T
  charge-neutral solves now seed an electron charge-asymmetry density from
  `eta * n_gamma(T_gamma) * sum_i Z_i X_i / A_i`, append it to the SciPy state
  vector only in charge-neutral mode, use the evolved state as the target
  passed to `charge_neutral_electron_chemical_potential(...)`, and evolve it
  with `3 n_Q dT_gamma/(T_gamma dN) + eta n_gamma d(sum_i Z_i X_i/A_i)/dN`.
  Result metadata records the state history, final value, and
  `charge_neutral_positive_charge_density_evolved_v1`.
- **Scope boundary:** this closes the staged independent charge-asymmetry state
  blocker across LRS, source-only non-LRS, and nonlinear non-LRS 3T shells.
  Exact finite-`mu_e`/tensor QED, public dispatch, production SMC,
  promotion-grade full-BBN, and QKE remain out of scope.
- **Exit gate:** focused tests lock source-only and nonlinear non-LRS
  state-history/final metadata, charge-neutral state contracts, initial state
  seeding, nonzero evolved-state motion under the fake one-step solve, and the
  fact that the charge-neutral `mu_e` callback sees both initial and final
  evolved state targets.
- **Verification:** `PYTHONPATH=src pytest -q tests/test_augmented_typeI_nonlrs_weak_network_solve.py tests/test_augmented_typeI_nonlrs_nonlinear_weak_network_3t.py -k "charge_asymmetry_state"`,
  `PYTHONPATH=src pytest -q tests/test_augmented_typeI_nonlrs_weak_network_solve.py tests/test_augmented_typeI_nonlrs_nonlinear_weak_network_3t.py -k "charge_neutral or electron_mu or charge_asymmetry_state"`,
  and `PYTHONPATH=src pytest -q tests/test_augmented_typeI_nonlrs_weak_network_solve.py tests/test_augmented_typeI_nonlrs_nonlinear_weak_network_3t.py`
  passed locally.
- **Known red tests:** the focused source-only and nonlinear non-LRS tests first
  failed because both result contracts remained `not_evolved` and the state
  vector did not yet include the charge-asymmetry slot.

---

### AP6-PSTF-DESCRIPTOR-AWARE-RADIAL-KINEMATICS  Label-aware radial momentum grids

- **Status:** stage-recorded (historical; not current capability)
- **Scope:** upgrade the AP6 LRS `pstf_radial` route so radial-grid
  kinematics are derived from each process descriptor's particle labels instead
  of validation-local placeholder momentum fractions.
- **Key files:** `src/rabbit/transport/augmented_collision_bridge.py`,
  `src/rabbit/validation/augmented_convergence.py`,
  `tests/test_augmented_collision_bridge.py`, and
  `tests/test_augmented_convergence.py`.
- **Physics added/changed:** `build_pstf_radial_grid_kwargs_for_particle_labels(...)`
  builds shared total-energy grids with label-aware momenta.  The LRS
  `pstf_radial` source factory uses the frozen dimensionless solver grid with
  neutrino `p=q` and electron/positron
  `p=sqrt(max(q^2-(m_e/T_nu_e0)^2,0))` from the descriptor electron mass and
  initial neutrino-energy scale.  It now builds per-process radial grids from
  `pstf_process_particle_mode_labels(...)` and evaluates the electron/positron FD bath with
  `build_energy_grid_finite_mass_fermi_dirac_radial_bath_mode_provider(...)`,
  which treats the radial node as total electron/positron energy rather than
  electron momentum.
- **Scope boundary:** this closes the placeholder radial-kinematics gap for
  the staged LRS smoke route.  It is compatible with the staged finite-mu-scaled
  isotropic QED EOS feedback but does not add live electron chemical-potential
  evolution, production quadrature, public dispatch, QKE, or promotion-grade
  full-span live-RHS radial coupling.
- **Exit gate:** focused bridge tests lock total-energy finite-mass FD bath
  evaluation and descriptor-label radial momenta, while the LRS `pstf_radial`
  smoke tests lock finite execution, 15-source catalog routing, budgeted
  live-RHS execution, and live-vs-frozen source-policy reporting.
- **Verification:** focused `tests/test_augmented_collision_bridge.py` radial
  kinematics/provider tests and the LRS `pstf_radial` smoke artifact tests
  passed locally for this route update.
- **Known red tests:** the descriptor-aware tests first failed on the missing
  total-energy-grid bath provider and label-aware radial-grid helper.

---

### AP6-PSTF-PROCESS-RADIAL-SOURCE  Descriptor-driven radial source evaluation

- **Status:** stage-recorded (historical; not current capability)
- **Scope:** compose the AP6 physical process descriptor catalog, radial
  channel-grid assembly, and six-monomial radial contraction into one
  executable process-specific source evaluator.
- **Key files:** `src/rabbit/collisions/pstf_process_catalog.py`,
  `src/rabbit/collisions/__init__.py`, and
  `tests/test_pstf_process_catalog.py`.
- **Physics added/changed:** `evaluate_pstf_process_radial_collision_source(...)`
  now builds a descriptor-driven radial channel grid and contracts supplied
  mode-space occupation arrays into concrete finite `C_modes` values.  The
  returned `PSTFProcessRadialCollisionResult` preserves the physical process
  descriptor, the precomputed radial grid, the lower-level radial contraction
  metadata, and a replay-stable process contract string.
- **Scope boundary:** this is an executable deterministic collision-reference
  source evaluator.  It does not yet wire the PSTF radial source into the AP35
  angular thermo callback, promote default collision feedback, add adaptive
  production quadrature, or change the no-QKE/public-dispatch boundary.
- **Exit gate:** focused tests compare the one-call process radial source
  against manual descriptor-grid construction plus radial contraction and lock
  a nonzero finite smoke `C_modes` value for a physical `nu-e` descriptor.
- **Verification:** `PYTHONPATH=src pytest -q tests/test_pstf_process_catalog.py`
  passed locally.
- **Known red tests:** the evaluator test first failed on missing
  `evaluate_pstf_process_radial_collision_source`, then passed after the result
  dataclass, source evaluator, and exports were landed.

---

### AP6-PSTF-PROCESS-RADIAL-MOMENTS  Radial source moment extraction

- **Status:** stage-recorded (historical; not current capability)
- **Scope:** integrate a concrete process-specific radial PSTF source mode into
  number and energy moments suitable for later thermo-source callback wiring.
- **Key files:** `src/rabbit/collisions/pstf_process_catalog.py`,
  `src/rabbit/collisions/__init__.py`, and
  `tests/test_pstf_process_catalog.py`.
- **Physics added/changed:** `compute_pstf_process_radial_moments(...)`
  consumes a `PSTFProcessRadialCollisionResult`, selects a concrete output
  mode, and returns raw-quadrature `sum_i w_i E_i^2 C_i` and
  `sum_i w_i E_i^3 C_i` moments with the source descriptor, p1 grid, selected
  `C_mode`, and max-amplitude metadata preserved.  The helper deliberately
  takes raw particle-1 quadrature weights from the caller so AP35/AP36 can apply
  Laguerre plain-weight conversion or finite-volume bin weights explicitly at
  the runtime bridge boundary.
- **Scope boundary:** this closes the next numerical step after `C_modes`; it
  does not yet promote the PSTF radial source as the default AP35 angular
  thermo callback, add adaptive production quadrature, or change the no-QKE and
  no-public-dispatch boundaries.
- **Exit gate:** focused tests evaluate a physical `nu-e` descriptor-driven
  radial source and check exact number/energy moment arithmetic against the
  concrete returned monopole `C_mode`.
- **Verification:** `PYTHONPATH=src pytest -q tests/test_pstf_process_catalog.py`
  passed locally.
- **Known red tests:** the moment test first failed on missing
  `compute_pstf_process_radial_moments`, then passed after the moment result
  dataclass, raw-quadrature integration helper, and exports were landed.

---

### AP6-PSTF-RADIAL-MOMENT-THERMO-BRIDGE  AP18 source adapter

- **Status:** stage-recorded (historical; not current capability)
- **Scope:** expose concrete PSTF radial source moments through the existing
  AP18 `Augmented3TCollisionThermoSource` callback shape.
- **Key files:** `src/rabbit/transport/augmented_collision_bridge.py` and
  `tests/test_augmented_collision_bridge.py`.
- **Physics added/changed:** `build_augmented_pstf_radial_moment_thermo_source(...)`
  evaluates an explicit runtime moment provider, accepts returned
  `PSTFProcessRadialMomentResult` objects, groups them by `nue`, `nuebar`, and
  `nux`, and maps their energy moments into `dQ_nue_pair_N =
  dQ_nue + dQ_nuebar` and `dQ_nux_bank_N = g_nux dQ_nux`.  Diagnostics retain
  per-species number/energy moments, max source-mode amplitude, source count,
  and a no-QKE radial-moment bridge marker.  The same adapter now also maps
  every returned process source `C_modes(q, mode)` block into a species-indexed
  `dA_modes(species, mode, q)` hierarchy payload for AP18/AP40/AP65 RHS
  coupling.
- **Scope boundary:** this is an opt-in AP18 callback adapter.  It does not make
  PSTF radial collisions the default AP35/AP36 source, does not choose a
  production quadrature policy, and does not promote public full-BBN dispatch.
- **Exit gate:** focused tests build physical descriptor-driven `nu-e` radial
  moments for `nue`, `nuebar`, and `nux`, run the callback, and check exact
  `dQ_nue_pair_N`/`dQ_nux_bank_N` bank bookkeeping, `C_modes -> dA_modes`
  hierarchy mapping, and diagnostics.
- **Verification:** `PYTHONPATH=src pytest -q tests/test_augmented_collision_bridge.py -k pstf_radial_moment_thermo_source`
  passed locally.
- **Known red tests:** the bridge test first failed on missing
  `build_augmented_pstf_radial_moment_thermo_source`, then passed after the
  opt-in callback builder, moment-output validation, and export were landed.

---

### AP6-PSTF-LIVE-RADIAL-MOMENT-PROVIDER  Live augmented-state provider

- **Status:** stage-recorded (historical; not current capability)
- **Scope:** evaluate configured AP6 radial process moments from the live AP18
  callback payload rather than only from externally precomputed moments.
- **Key files:** `src/rabbit/transport/augmented_collision_bridge.py` and
  `tests/test_augmented_collision_bridge.py`.
- **Physics added/changed:** `AugmentedPSTFRadialMomentProcessConfig` records a
  physical process descriptor, the four species slots supplying
  `F1`/`F2`/`F3`/`F4`, raw p1 moment weights, radial-grid inputs, and moment
  powers.  `build_augmented_pstf_radial_moment_provider(...)` reconstructs the
  current occupation distribution from `A_modes`/`q_nodes`, projects nodal
  occupations back to PSTF mode coefficients, evaluates the configured radial
  process source, and returns concrete `PSTFProcessRadialMomentResult` objects
  for the AP18 radial-moment thermo bridge.
- **Scope boundary:** this is still an explicit configured provider.  It does
  not choose the production process catalog, default quadrature, source-update
  policy, or public forward-solver dispatch.
- **Exit gate:** focused tests build live `nue`/`nux` pairwise diagonal no-QKE
  radial process configs, feed non-equilibrium `A_modes`, and check finite
  live radial moments plus exact AP18 `dQ` bank bookkeeping through the existing
  radial-moment thermo-source bridge.
- **Verification:** `PYTHONPATH=src pytest -q tests/test_augmented_collision_bridge.py -k pstf_radial_moment_provider`
  passed locally.
- **Known red tests:** the provider test first failed on missing
  `build_augmented_pstf_radial_moment_provider`, then passed after the config
  dataclass, live distribution-to-mode provider, and exports were landed.

---

### AP6-PSTF-LIVE-RADIAL-MOMENT-AP18-EVALUATOR  Source evaluator acceptance

- **Status:** stage-recorded (historical; not current capability)
- **Scope:** thread the live AP6 radial-moment provider and thermo-source
  adapter through the existing LRS and non-LRS AP18 source-evaluator
  boundaries, using current `A_modes`/`q_nodes` rather than frozen precomputed
  source moments.
- **Key files:** `tests/test_augmented_typeI_weak_network_3t_solve.py`.
  The exercised source boundaries are the existing
  `src/rabbit/transport/augmented_typeI_weak_network.py` evaluators and the
  existing radial bridge in `src/rabbit/transport/augmented_collision_bridge.py`.
- **Physics added/changed:** the AP18 source evaluators now have regressions
  showing that configured pairwise diagonal no-QKE PSTF radial process sources
  can reconstruct live `nue`/`nux` occupation modes, evaluate concrete radial
  `C_modes`, integrate them into number/energy moments, and return finite
  `dQ_nue_pair_N`/`dQ_nux_bank_N` values with nonzero radial-source diagnostics
  on both the LRS basis and the non-LRS S2 `{monopole, W_+, W_-}` grid.
- **Scope boundary:** this does not make the radial PSTF collision source the
  default AP35/AP36 source, does not add public dispatch, and does not run a
  long live-RHS stiff solve in the smoke suite.  Full solve/update-policy
  promotion for the radial PSTF source remains a separate runtime and
  convergence problem.
- **Exit gate:** focused tests call `_evaluate_collision_thermo_source(...)`
  and `_evaluate_nonlrs_collision_thermo_source(...)` with live radial
  providers/adapters, configured diagonal no-QKE `nu-nu` radial process
  descriptors, and non-equilibrium `A_modes`; they check the AP18 closure
  contract, source count, nonzero `radial_max_abs_C_mode`, bridge marker, and
  finite 3T `dQ` bank outputs.
- **Verification:** `PYTHONPATH=src pytest -q tests/test_augmented_typeI_weak_network_3t_solve.py -k pstf_radial_moment_source`
  passed locally.
- **Known red tests:** the first focused regression attempted to run the full
  Radau solve with the live radial callback and was too expensive for the
  default smoke suite; root cause was repeated expensive radial-source
  evaluation inside the stiff RHS loop, so the regression was narrowed to the
  AP18 source-evaluator boundary.

---

### AP6-PSTF-RADIAL-COLLISION-FEEDBACK-VARIANT  LRS artifact/gate route

- **Status:** stage-recorded (historical; not current capability)
- **Scope:** add an explicit LRS `pstf_radial` source variant to the existing
  3T collision-feedback artifact and candidate-gate surfaces.
- **Key files:** `src/rabbit/validation/augmented_convergence.py`,
  `src/rabbit/validation/augmented_stability.py`,
  `tests/test_augmented_convergence.py`, and
  `tests/test_augmented_stability_envelope.py`.
- **Physics added/changed:** the LRS artifact now builds a deterministic
  pairwise diagonal no-QKE `nu-nu` AP6 radial process configuration on the
  current LRS angular/q grids, reconstructs the radial source moments through
  the live AP6 provider, maps them into the AP18 3T thermo-source contract, and
  feeds the resulting `dQ_nue_pair_N`/`dQ_nux_bank_N` source into the existing
  LRS 3T solve shell.  The route freezes the evaluated source at the initial
  state by default so it returns stable concrete numerical outputs without
  repeated expensive radial contractions inside the stiff RHS loop.  After the
  AP6 CPU hot-path pass, the same standard artifact route also accepts explicit
  `source_update_policy="live_rhs"` under `max_pstf_radial_source_evaluations`
  and records the radial source-evaluation budget in the `pstf_radial` variant
  payload.  The budgeted live-RHS path also exercises the existing
  charge-neutrality electron chemical-potential callback from the current RHS
  `X` and `T_gamma_MeV` payload when
  `electron_chemical_potential_mode="charge_neutrality"`.  The LRS
  candidate-gate and span-ladder gate specs now forward
  `electron_chemical_potential_mode`,
  `electron_chemical_potential_MeV`, and
  `max_pstf_radial_source_evaluations` into that same artifact route, so
  smoke-scale `pstf_radial live_rhs` candidate runs can use the live
  charge-neutrality electron/positron bath split without the old frozen-only
  gate restriction.  The charge-neutrality diagnostics now also expose
  finite-mass signed-`mu_e` e-/e+ energy and pressure densities from the local
  photon/electron EOS module.  The artifact observables now also record the
  final projected kinetic source amplitude as `collision_dA_abs_max_final`.
- **Scope boundary:** this does not make `pstf_radial` a default source, does
  not add public/canonical dispatch, does not promote full-span live-RHS radial
  collision feedback, and does not claim physical process families outside the
  supported no-QKE HM finite-mass electromagnetic plus all-nine diagonal-`nu-nu`
  descriptor catalog.  `source_update_policy="live_rhs"` remains diagnostic and
  requires the explicit source-evaluation budget.
- **Exit gate:** the artifact records
  `source_variant_routing["pstf_radial"] ==
  "lrs_pstf_radial_moment_3t_shell_callback"`, source contract
  `augmented_pstf_radial_moment_thermo_source_v1`, nonzero radial
  source-moment diagnostics, finite 3T source observables, candidate-gate
  pass/fail handling for the new variant, final `dA_modes` amplitude
  extraction, and budgeted live-RHS execution with
  concrete source-evaluation accounting.  The charge-neutrality regression
  records one solved `mu_e` payload per live source evaluation and checks the
  final radial source diagnostics; the LRS candidate gate also records
  `source_evaluations` and `source_evaluation_budget_passed` as gate
  observables for the budgeted live-RHS radial route.  The EOS regression
  locks the zero-`mu_e` e-/e+ energy/pressure split against the existing
  total `rho_electron`/`pressure_electron` functions and checks positive
  `mu_e` makes the electron branch exceed the positron branch.
- **Verification:** `PYTHONPATH=src pytest -q tests/test_augmented_convergence.py::test_build_augmented_lrs_3t_collision_feedback_artifact_runs_real_pstf_radial_smoke_path tests/test_augmented_stability_envelope.py::test_augmented_lrs_collision_feedback_candidate_gate_accepts_pstf_radial_variant`
  passed locally.
- **Known red tests:** the focused artifact and candidate-gate regressions
  first failed because `pstf_radial` was rejected by the source-variant
  allowlists and no LRS radial source factory was wired into the artifact.

---

### AP6-PSTF-RADIAL-NONLRS-COLLISION-FEEDBACK-VARIANT  Non-LRS artifact route

- **Status:** stage-recorded (historical; not current capability)
- **Scope:** add an explicit non-LRS `pstf_radial` source variant to the AP42
  3T collision-feedback artifact so the AP6 descriptor-driven radial moment
  provider can run on the staged S2 `{monopole, W_+, W_-}` basis.
- **Key files:** `src/rabbit/validation/augmented_convergence.py` and
  `tests/test_augmented_convergence.py`.
- **Physics added/changed:** the AP42 artifact now builds S2 direction vectors
  from the non-LRS quadrature grid, reuses the AP6 finite-mass electromagnetic
  plus UR diagonal no-QKE `nu-nu` process catalog, reconstructs live
  `A_modes` through the radial moment provider, maps the resulting number and
  energy source moments into the AP18 3T thermo-source contract, and feeds that
  callback into the AP40 non-LRS source-only 3T shell.  The smoke default
  freezes the initial radial source; `live_rhs` remains explicit and budgeted
  through `max_pstf_radial_source_evaluations`.  The artifact observables
  record `collision_dA_abs_max_final` so the staged non-LRS route exposes the
  final kinetic collision source amplitude as well as thermo moments.
- **Scope boundary:** this does not make `pstf_radial` a default source, does
  not add public/canonical dispatch, does not promote nonlinear non-LRS
  transport or full-span live-RHS radial collision feedback, and does not claim
  complete process coverage beyond the staged finite-mass electromagnetic plus
  UR diagonal no-QKE descriptor catalog.
- **Exit gate:** focused tests lock public artifact routing through
  `nonlrs_pstf_radial_moment_3t_shell_callback`, forwarding of fixed
  `mu_e` controls to the AP40 shell, finite real smoke output for
  `collision_dQ_nue_pair_N_final`/`collision_dQ_nux_bank_N_final`,
  nonzero `collision_dA_abs_max_final`, and radial
  diagnostics including `pstf_radial_moment_bridge_v1`,
  `n_radial_moment_sources`, finite-mass elastic/pair and diagonal `nu-nu`
  source counts, and the finite-mass HM process markers.
- **Verification:** `PYTHONPATH=src pytest -q tests/test_augmented_convergence.py -k "nonlrs_3t_collision_feedback_artifact_routes_pstf_radial_source or nonlrs_3t_collision_feedback_artifact_runs_real_pstf_radial_smoke_path"`
  passed locally.
- **Known red tests:** the routing regression first failed because the AP42
  artifact did not accept `electron_chemical_potential_MeV` and the non-LRS
  source-variant allowlist had no `pstf_radial` entry.

---

### AP6-PSTF-RADIAL-NONLRS-SOURCE-POLICY  Non-LRS live-vs-frozen artifact

- **Status:** stage-recorded (historical; not current capability)
- **Scope:** add a dedicated non-LRS S2 `pstf_radial` source-policy artifact
  and JSON writer around the AP45 generic frozen/live source-policy runner.
- **Key files:** `src/rabbit/validation/augmented_convergence.py` and
  `tests/test_augmented_convergence.py`.
- **Physics added/changed:** no new kernel is introduced.  The artifact runs
  the real AP42 `pstf_radial` route twice, once with frozen initial radial
  moments and once with budgeted live-RHS source re-evaluation, then reports
  terminal thermo/network/source observables, radial source diagnostics,
  live-minus-frozen deltas, and the live source-evaluation budget.  It now
  forwards the AP6 `standard_3t_plasma` electromagnetic energy-closure mode
  so the policy comparison can use the same standard 3T plasma-transfer
  normalization as the direct AP6 radial artifact.
- **Scope boundary:** this is a smoke-scale diagnostic source-policy artifact.
  It does not make the radial source default, does not promote full-span
  live-RHS collision coupling, does not add nonlinear non-LRS collision
  transport, and does not change public dispatch.
- **Exit gate:** focused tests run the real non-LRS `pstf_radial`
  frozen/live smoke path, require finite nonzero source moments, require the
  AP42 source-policy contract, and check the writer emits JSON.
- **Verification:** `PYTHONPATH=src pytest -q tests/test_augmented_convergence.py -k "nonlrs_pstf_radial_source_policy_artifact"`
  passed locally.
- **Known red tests:** focused tests first failed because the dedicated
  non-LRS radial source-policy artifact and writer were not exported.

---

### AP6-PSTF-RADIAL-LIVE-RHS-BUDGET  Bounded live source probe

- **Status:** stage-recorded (historical; not current capability)
- **Scope:** add a deterministic tiny-span live-RHS budget artifact and JSON
  writer for the LRS `pstf_radial` AP6 source route.
- **Key files:** `src/rabbit/validation/augmented_convergence.py` and
  `tests/test_augmented_convergence.py`.
- **Physics added/changed:** the artifact builds the same AP6 radial
  pairwise diagonal no-QKE `nu-nu` process provider as the frozen
  `pstf_radial` route, but wires it directly into the AP15/AP18 LRS 3T solve
  as a live RHS callback under an explicit source-evaluation budget.  The
  default smoke run uses `N_q=3`, `N_mu=4`, `N_span=(0, 1e-14)`, and `RK23`,
  records concrete terminal 3T observables plus radial source diagnostics, and
  returns a fail-closed artifact if the source evaluation budget is exceeded.
  The live budget path now accepts the AP6 `standard_3t_plasma` radial energy
  normalization and records that mode in artifact inputs.
- **Scope boundary:** this is a diagnostic budget probe only.  It does not
  make `pstf_radial` live-RHS available on the standard collision-feedback
  artifact, does not promote full-span stiff radial coupling, does not add
  public/canonical dispatch, and does not claim QKE or complete process
  coverage.
- **Exit gate:** focused tests run the real budgeted live-RHS smoke artifact,
  check finite nonzero radial source output and source-evaluation counts, check
  budget-exceeded failure metadata, and check JSON writer round-trip.
- **Verification:** `PYTHONPATH=src pytest -q tests/test_augmented_convergence.py::test_build_augmented_lrs_pstf_radial_live_rhs_budget_artifact_runs_real_smoke_path tests/test_augmented_convergence.py::test_build_augmented_lrs_pstf_radial_live_rhs_budget_artifact_reports_source_budget_failure tests/test_augmented_convergence.py::test_write_augmented_lrs_pstf_radial_live_rhs_budget_artifact_writes_json`
  passed locally.
- **Known red tests:** focused tests first failed because
  `build_augmented_lrs_pstf_radial_live_rhs_budget_artifact` did not exist.

---

### AP6-PSTF-RADIAL-SOURCE-POLICY  Live-vs-frozen radial policy comparison

- **Status:** stage-recorded (historical; not current capability)
- **Scope:** add a deterministic LRS `pstf_radial` source-policy comparison
  artifact and JSON writer.
- **Key files:** `src/rabbit/validation/augmented_convergence.py` and
  `tests/test_augmented_convergence.py`.
- **Physics added/changed:** the artifact runs the frozen-initial-state
  `pstf_radial` collision-feedback route and the budgeted live-RHS radial route
  at the same smoke-scale quadrature/span/settings, then records
  live-minus-frozen deltas for common 3T/network observables and radial
  source diagnostics.  Both rows use the AP6 radial provider and AP18 3T
  thermo-source bridge, so the report compares concrete coupled-solve numbers
  rather than only source contracts.  The comparison now forwards
  `pstf_radial_energy_normalization`, allowing both policies to run with the
  standard 3T electromagnetic energy-closure mode.
- **Scope boundary:** this remains a diagnostic comparison.  It does not make
  live-RHS radial feedback default in the standard artifact, does not promote
  full-span stiff radial coupling, does not add public/canonical dispatch, and
  does not claim QKE or complete process coverage.
- **Exit gate:** focused tests run the real live-vs-frozen artifact, check
  nonzero radial source observables in both rows, check the live RHS
  source-evaluation accounting, check live-minus-frozen diagnostic deltas, and
  check JSON writer round-trip.
- **Verification:** `PYTHONPATH=src pytest -q tests/test_augmented_convergence.py::test_build_augmented_lrs_pstf_radial_source_policy_artifact_compares_live_and_frozen tests/test_augmented_convergence.py::test_write_augmented_lrs_pstf_radial_source_policy_artifact_writes_json`
  passed locally.
- **Known red tests:** focused tests first failed because
  `build_augmented_lrs_pstf_radial_source_policy_artifact` did not exist.

---

### AP6-PSTF-RADIAL-CPU-PRETAB  CPU warm-cache optimisation

- **Status:** stage-recorded (historical; not current capability)
- **Scope:** reduce AP6 radial-kernel warm-build overhead and add an explicit
  pretabulation path for static radial channel grids.
- **Key files:** `src/rabbit/collisions/pstf_contractions.py`,
  `src/rabbit/transport/augmented_collision_bridge.py`,
  `tests/test_pstf_collision_contractions.py`, and
  `tests/test_augmented_typeI_weak_network_3t_solve.py`.
- **Physics added/changed:** none.  The live neutrino distribution still
  enters the radial source contraction on the RHS.  Static pieces of the
  collision operator are now more clearly pretabulated: `p4` interpolation
  metadata, radial quadrature weights, invariant prefactors, in-builder static
  angular-geometry reuse for radial-independent momentum-delta tensors,
  geometric table lookup for static normalized unit-direction momentum-delta
  weights, and descriptor-specific radial channel `K` tensors.  The radial
  grid builder now uses an internal fast channel assembly path instead of
  allocating public channel-table objects for every valid radial tuple.
- **Runtime/cache controls:** radial channel grids can be round-tripped through
  an NPZ cache, and `build_augmented_pstf_radial_moment_provider(...,
  radial_grid_cache_dir=...)` reuses those pretabulated grids across provider
  construction when the deterministic cache key matches.
- **Scope boundary:** this is a CPU/runtime optimisation only.  It does not
  promote `pstf_radial` to default/public dispatch, does not change the
  no-QKE boundary, does not add GPU support, and does not replace live RHS
  Boltzmann evolution with table lookup.
- **Exit gate:** focused tests lock internal fast radial-grid assembly, static
  angular-geometry reuse without an external cache, radial quadrature-weight
  precomputation, NPZ grid round-trip, and provider-level disk-cache reuse
  without rebuilding the grid.
- **Verification:** `PYTHONPATH=src pytest -q tests/test_pstf_collision_contractions.py tests/test_augmented_typeI_weak_network_3t_solve.py -k "pstf_radial or radial_channel_kernel_grid"`
  passed locally.
- **Known red tests:** focused tests first failed because the radial grid
  builder still called the public channel-table constructor per radial tuple,
  rebuilt the same static angular geometry once per valid radial tuple when no
  external cache was supplied, and no NPZ radial-grid cache API/provider hook
  existed.

---

### AP6-PSTF-RADIAL-CPU-CONTRACTION-HOTPATH  CPU live-RHS contraction optimisation

- **Status:** stage-recorded (historical; not current capability)
- **Scope:** reduce warm live-RHS AP6 radial contraction overhead without
  changing the collision physics or promoting the `pstf_radial` route.
- **Key files:** `src/rabbit/collisions/pstf_contractions.py`,
  `src/rabbit/transport/augmented_collision_bridge.py`,
  `tests/test_pstf_collision_contractions.py`, and
  `tests/test_augmented_typeI_weak_network_3t_solve.py`.
- **Physics added/changed:** none.  The live augmented distribution is still
  reconstructed on every source evaluation, and the same six-monomial PSTF
  radial kernel is contracted against current `F1`/`F2`/`F3`/`F4` modes.
- **Runtime/cache controls:** scalar radial contractions now reuse the already
  validated kernel-grid tensors instead of revalidating the full `K` mapping on
  every RHS call.  A process-leading batch contraction API is available for
  compatible grids, guarded by an explicit `max_kernel_nbytes` budget; provider
  auto-dispatch only enables it by default for mode ladders with at least three
  angular modes, while two-mode smoke paths remain on the scalar hot path.
- **Performance before / after:** AP6 radial live-RHS smoke warm-cache profile
  at `ell_max=2, N_q=8, N_mu=12, max_source_evaluations=64` previously measured
  about `0.088 s` elapsed / `0.082 s` source wall.  After this change, five
  warm repetitions measured mean `0.075 s` elapsed / `0.071 s` source wall.
  An `ell_max=4` smoke check kept the default process-batched route available
  with mean wall time about `0.186 s` versus `0.199 s` when batch construction
  was forced off in the same local run.
- **Scope boundary:** this is CPU-first runtime plumbing only.  It does not add
  GPU support, does not replace on-the-fly Boltzmann source evaluation with
  pretabulated distributions, does not make live-RHS full-span coupling a
  promoted path, and keeps QKE out of scope.
- **Exit gate:** focused tests lock batch-vs-scalar numerical parity, memory
  budget fallback, provider batch dispatch when explicitly allowed, the default
  two-mode scalar policy, and the surrounding radial/provider regression bundle.
- **Verification:** `PYTHONPATH=src pytest -q tests/test_pstf_collision_contractions.py tests/test_augmented_typeI_weak_network_3t_solve.py`
  passed locally.
- **Known red tests:** focused policy test first failed because the provider
  attempted to build a batch for the two-mode smoke path by default.

---

### AP6-PSTF-RADIAL-MOMENT-WEIGHT-HOTPATH  CPU moment-extraction optimisation

- **Status:** stage-recorded (historical; not current capability)
- **Scope:** finish the AP6 CPU hot-path pass by moving invariant p1 moment
  weights out of repeated live-RHS source evaluations.
- **Key files:** `src/rabbit/collisions/pstf_process_catalog.py`,
  `src/rabbit/collisions/__init__.py`,
  `src/rabbit/transport/augmented_collision_bridge.py`,
  `tests/test_pstf_process_catalog.py`, and
  `tests/test_augmented_typeI_weak_network_3t_solve.py`.
- **Physics added/changed:** none.  The provider still evaluates live
  distribution-dependent radial `C_modes`; only `p1_weights * E^power`
  factors used to integrate number and energy moments are precomputed per
  configured process.
- **Performance before / after:** after AP6-PSTF-RADIAL-CPU-CONTRACTION-HOTPATH,
  a representative warm profile measured about `0.081 s` elapsed / `0.075 s`
  source wall for `ell_max=2, N_q=8, N_mu=12`.  With precomputed moment
  weights, five warm repetitions measured mean `0.069 s` elapsed / `0.064 s`
  source wall on the same smoke artifact.
- **Scope boundary:** this is a CPU/runtime cleanup only.  It does not
  precompute the evolving distribution, does not change the collision operator,
  does not promote live-RHS full-span coupling, and keeps QKE out of scope.
- **Exit gate:** focused tests lock preweighted moment parity against the
  public helper and provider reuse of construction-time moment weights.
- **Known red tests:** focused tests first failed because the preweighted moment
  API did not exist.

---

### AP6-NONLRS-PSTF-RADIAL-DIRECT-WRAPPER  Direct non-LRS radial collision 3T solve

- **Status:** stage-recorded (historical; not current capability)
- **Scope:** expose the AP6 descriptor-driven `pstf_radial` collision source as
  an opt-in direct non-LRS 3T solve path instead of only through AP42/AP45
  artifact policy wrappers.
- **Key files:** `src/rabbit/transport/augmented_collision_bridge.py`,
  `src/rabbit/transport/augmented_typeI_weak_network.py`,
  `src/rabbit/validation/augmented_convergence.py`,
  `scripts/run_augmented_nonlrs_pstf_radial_collision_3t_solve_artifact.py`,
  `tests/test_augmented_typeI_nonlrs_weak_network_solve.py`, and
  `tests/test_augmented_convergence.py`.
- **Physics added/changed:** the reusable radial source construction now lives
  in the transport collision bridge for LRS and non-LRS geometries, including
  finite-mass electron/positron bath handling and algebraic charge-neutral
  `mu_e` diagnostics.  `run_augmented_nonlrs_pstf_radial_collision_weak_network_3T_solve(...)`
  composes that source with the AP40 non-LRS 3T collision-moment hook and the
  source-only S2 transport shell.  The live-RHS route records explicit
  `pstf_radial_source_evaluations` and budget-pass diagnostics.
- **Numeric smoke evidence:** the CLI smoke run with `N_q=3, N_mu=3, N_phi=5`,
  `N_span=(0, 1e-14)`, `method=RK23` emitted
  `collision_dQ_nue_pair_N_final = 2.394037037096395e-3`,
  `collision_dQ_nux_bank_N_final = 6.159604620583849e-4`,
  `radial_max_abs_C_mode = 4.205596736353273e-6`, `nfev = 5`, and
  7 source evaluations under the default budget of 64.
- **Scope boundary:** this is an explicit diagnostic wrapper and JSON/CLI
  artifact.  It does not make `pstf_radial` default/public dispatch, does not
  replace AP37 source-only non-LRS transport with the nonlinear operator, does
  not promote full-span live-RHS collision feedback, and keeps QKE out of
  scope.
- **Verification:** `PYTHONPATH=src pytest -q tests/test_augmented_typeI_nonlrs_weak_network_solve.py -k pstf_radial`,
  `PYTHONPATH=src pytest -q tests/test_augmented_convergence.py -k "pstf_radial_collision_3t_solve_artifact"`, and
  `PYTHONPATH=src python scripts/run_augmented_nonlrs_pstf_radial_collision_3t_solve_artifact.py --output /tmp/nonlrs_pstf_radial_direct.json --N-span-end 1e-14 --method RK23 --rtol 1e-4 --atol 1e-7`
  passed locally.
- **Known red tests:** focused tests first failed because the direct wrapper,
  artifact builder, and source-evaluation budget diagnostics did not exist.

---

### AP6-PSTF-RADIAL-STANDARD-3T-ENERGY-CLOSURE  Standard 3T radial energy normalization

- **Status:** stage-recorded (historical; not current capability)
- **Scope:** add an opt-in standard-3T plasma energy-normalization mode to the
  AP6 radial-moment AP18 thermo-source bridge, and thread it through the
  direct source-only and nonlinear non-LRS `pstf_radial` 3T wrappers plus
  collision-feedback artifact builders.
- **Key files:** `src/rabbit/transport/augmented_collision_bridge.py`,
  `src/rabbit/transport/augmented_typeI_weak_network.py`,
  `src/rabbit/validation/augmented_convergence.py`,
  `scripts/run_augmented_nonlrs_pstf_radial_collision_3t_solve_artifact.py`,
  `scripts/run_augmented_nonlrs_3t_collision_feedback_artifact.py`,
  `scripts/run_augmented_3t_collision_feedback_artifact.py`,
  `tests/test_augmented_collision_bridge.py`,
  `tests/test_augmented_convergence.py`,
  `tests/test_augmented_typeI_nonlrs_weak_network_solve.py`, and
  `tests/test_augmented_typeI_nonlrs_nonlinear_collision_feedback_3t.py`.
- **Physics added/changed:** `energy_normalization="standard_3t_plasma"`
  keeps the concrete radial `dA_modes` hierarchy payload, then applies a
  number-neutral monopole correction to the electromagnetic radial source so
  `dQ_nue_pair_N` and `dQ_nux_bank_N` match the canonical 3T
  plasma-transfer table per e-fold at the current `T_gamma`, `T_nu_e`,
  `T_nu_x`, and Hubble rate.  Raw radial source moments remain available via
  the default `energy_normalization="raw"`.  The direct AP6 artifact CLI and
  LRS/non-LRS collision-feedback artifact CLIs expose the normalization mode
  and initial 3T temperatures; the direct CLI summary reports the nonzero
  `nue+nuebar` and `nux` electromagnetic energy targets plus the maximum
  closure residual.
- **Scope boundary:** this closes a radial thermo-normalization gap for an
  opt-in staged AP6 path.  It does not promote default/public collision-coupled
  full-BBN dispatch, does not remove live-RHS source budgets, and keeps QKE out
  of scope.
- **Exit gate:** focused tests lock exact closure to
  `total_energy_transfer(...) / H_MeV`, zero electromagnetic closure residuals,
  preserved nonzero `dA_modes`, wrapper forwarding of the normalization mode,
  and frozen-source initialization with the actual initial 3T Hubble rate
  instead of a unit placeholder.
- **Verification:** `PYTHONPATH=src pytest -q tests/test_augmented_collision_bridge.py -k rate_closes_em_energy_to_3t_table`,
  `PYTHONPATH=src pytest -q tests/test_augmented_typeI_nonlrs_weak_network_solve.py -k "pstf_radial_collision_3t_wrapper"`, and
  `PYTHONPATH=src pytest -q tests/test_augmented_typeI_nonlrs_nonlinear_collision_feedback_3t.py -k "pstf_radial_collision_3t_wrapper"`
  passed locally.  Follow-up CLI/artifact evidence used
  `PYTHONPATH=src pytest -q tests/test_augmented_convergence.py -k "standard_3t_radial_mode or energy_normalized_smoke_path or pstf_radial_collision_3t_solve_artifact_script"`
  and
  `PYTHONPATH=src python scripts/run_augmented_nonlrs_pstf_radial_collision_3t_solve_artifact.py --output /tmp/rabbit_ap6_standard_3t_radial_artifact.json --source-update-policy frozen_initial_state --pstf-radial-energy-normalization standard_3t_plasma --T-gamma0-MeV 0.8 --T-nu-e0-MeV 0.79 --T-nu-x0-MeV 0.78 --N-span-end 1e-14 --N-q 3 --N-mu 3 --N-phi 5 --method RK23 --rtol 1e-4 --atol 1e-7`,
  which reported `radial_em_energy_target_nue_pair_N=0.002439621160259491`,
  `radial_em_energy_target_nux_bank_N=0.0019820818977550445`, and
  `radial_em_energy_closure_residual_abs_max=8.673617379884035e-19`.
- **Known red tests:** focused wrapper regressions first failed because the
  test expected the physical initial Hubble rate to exceed 1 s^-1; the actual
  `T=0.8 MeV` smoke value is about 0.432 s^-1.  The assertions now compare
  directly to `hubble_3T(...) * _MEV_TO_S`.

---

### AP6-PSTF-RADIAL-STANDARD-3T-SPAN-PROFILE  Standard 3T radial span profile

- **Status:** stage-recorded (historical; not current capability)
- **Scope:** add a deterministic span-profile artifact and CLI around the
  direct non-LRS AP6 `pstf_radial` standard-3T radial closure wrapper.
- **Key files:** `src/rabbit/validation/augmented_convergence.py`,
  `scripts/run_augmented_nonlrs_pstf_radial_standard_3t_span_profile_artifact.py`,
  and `tests/test_augmented_convergence.py`.
- **Physics added/changed:** no new collision kernel is introduced.  The
  artifact runs the existing AP6 direct non-LRS radial 3T wrapper over
  explicit `N_span_end` ladder points with
  `source_update_policy="frozen_initial_state"` and
  `pstf_radial_energy_normalization="standard_3t_plasma"`, then records
  terminal 3T/network/source observables, source contract, `nfev`,
  electromagnetic energy targets, and maximum closure residual by span.
- **Scope boundary:** this is diagnostic span evidence only.  It does not
  promote full-span live-RHS radial coupling, does not add public dispatch,
  does not create production full-BBN evidence, and keeps QKE out of scope.
- **Exit gate:** focused tests lock the artifact schema, span-end validation,
  direct-wrapper forwarding, JSON writer, and CLI summary.
- **Verification:** `PYTHONPATH=src pytest -q tests/test_augmented_convergence.py -k "standard_3t_span_profile"`
  passed locally.  A real CLI run with `--N-span-ends 1e-14,1e-12,1e-8`,
  `T_gamma/T_nu_e/T_nu_x=0.8/0.79/0.78 MeV`, `method=LSODA`,
  `rtol=1e-5`, and `atol=1e-8` reported `all_success=true`,
  `nfev_by_span_end={"1e-14": 7, "1e-12": 635, "1e-08": 5577}`, and
  `max_radial_em_energy_closure_residual_abs=8.673617379884035e-19`.
- **Known red tests:** focused tests first failed because the span-profile
  artifact and CLI did not exist.

---

### AP6-PSTF-RADIAL-STANDARD-3T-SOURCE-POLICY-SPAN-PROFILE  Standard 3T radial source-policy span profile

- **Status:** stage-recorded (historical; not current capability)
- **Scope:** add a deterministic source-policy span-profile artifact and CLI
  around the direct non-LRS AP6 `pstf_radial` standard-3T radial closure
  wrapper.
- **Key files:** `src/rabbit/validation/augmented_convergence.py`,
  `scripts/run_augmented_nonlrs_pstf_radial_standard_3t_source_policy_span_profile_artifact.py`,
  and `tests/test_augmented_convergence.py`.
- **Physics added/changed:** no new collision kernel is introduced.  The
  artifact executes the existing direct AP6 radial 3T wrapper for the same
  explicit `N_span_end` ladder under both `frozen_initial_state` and
  `live_rhs`, always with
  `pstf_radial_energy_normalization="standard_3t_plasma"`.  It records
  terminal observables, live-minus-frozen observable deltas, source contract,
  `nfev`, source-evaluation counts, electromagnetic energy targets, and
  fail-closed maximum closure residuals by policy and span.
- **Scope boundary:** this is diagnostic live-RHS span evidence only.  It does
  not promote full-span radial coupling, does not add public dispatch, does
  not create production full-BBN evidence, and keeps QKE out of scope.
- **Exit gate:** focused tests lock policy/span forwarding, fail-closed
  standard-3T diagnostics, a real frozen/live smoke path, JSON writer, and CLI
  summary.
- **Verification:** `PYTHONPATH=src pytest -q tests/test_augmented_convergence.py -k "source_policy_span_profile"`
  passed locally.  A real CLI run with `--N-span-ends 1e-14,1e-12`,
  `--source-update-policies frozen_initial_state,live_rhs`,
  `--max-pstf-radial-source-evaluations 2048`,
  `T_gamma/T_nu_e/T_nu_x=0.8/0.79/0.78 MeV`, `method=LSODA`,
  `rtol=1e-5`, and `atol=1e-8` reported `all_success=true`, frozen/live
  `nfev_by_policy_and_span_end={"1e-14": 7, "1e-12": 635}`, live source
  evaluations `{"1e-14": 9.0, "1e-12": 637.0}`, and
  `max_radial_em_energy_closure_residual_abs=8.673617379884035e-19`.
- **Known red tests:** focused tests first failed because the source-policy
  span-profile artifact and CLI did not exist.

---

### AP6-PSTF-RADIAL-LIVE-RHS-BUDGET-OUTCOMES  Structured live-RHS radial budget outcomes

- **Status:** stage-recorded (historical; not current capability)
- **Scope:** make direct non-LRS AP6 `pstf_radial` live-RHS source-evaluation
  budget exhaustion a structured artifact outcome instead of an unhandled
  runtime traceback.
- **Key files:** `src/rabbit/transport/augmented_typeI_weak_network.py`,
  `src/rabbit/validation/augmented_convergence.py`,
  `scripts/run_augmented_nonlrs_pstf_radial_collision_3t_solve_artifact.py`,
  `scripts/run_augmented_nonlrs_pstf_radial_standard_3t_source_policy_span_profile_artifact.py`,
  and `tests/test_augmented_convergence.py`.
- **Physics added/changed:** no new collision kernel is introduced.  The
  live-RHS path still evaluates the AP6 descriptor-driven radial source on the
  RHS; when the explicit budget is exhausted, the transport wrapper raises a
  typed budget exception carrying used/max source evaluations.  The direct
  artifact records `success=false`, `failure.reason=source_evaluation_budget_exceeded`,
  budget diagnostics, and no closure target claim.  The source-policy span
  profile keeps successful shorter rows and records failed live rows in-place.
- **Scope boundary:** this is diagnostic outcome classification for live-RHS
  radial coupling.  It does not make live-RHS default, does not promote
  full-span radial coupling, does not add public dispatch, and keeps QKE out of
  scope.
- **Exit gate:** focused tests lock direct-artifact budget-failure reporting,
  source-policy span-profile failure-row recording, CLI summaries, and the
  existing successful standard-3T frozen/live smoke rows.
- **Verification:** `PYTHONPATH=src pytest -q tests/test_augmented_convergence.py -k "source_budget_failure or source_policy_span_profile or pstf_radial_collision_3t_solve_artifact_script"`
  passed locally.  A real direct CLI run at `N_span_end=1e-10` with live-RHS,
  `standard_3t_plasma`, LSODA, and budget `4096` returned `success=false` with
  `failure_reason=source_evaluation_budget_exceeded`.  A real source-policy
  span-profile run over `1e-14,1e-12,1e-10` returned `failure_rows=1`, preserved
  the shorter live-RHS successes, and recorded live source evaluations
  `9, 637, 4096`.
- **Known red tests:** focused tests first failed because the typed budget
  exception and structured artifact rows did not exist.

---

### AP4-AP65-COMBINED-ELECTRON-MU-FULL-SPAN  Combined full-span electron-mu controls

- **Status:** stage-recorded (historical; not current capability)
- **Scope:** expose the already landed finite-mass e-/e+ fixed-`mu_e` and
  charge-neutrality radial bath route through the AP4/AP65 combined
  angular+`pstf_radial` full-span candidate gate and frozen/live
  source-policy span-profile artifact.
- **Key files:** `src/rabbit/validation/augmented_stability.py`,
  `scripts/run_augmented_nonlrs_combined_full_span_3t_candidate_gate.py`,
  `scripts/run_augmented_nonlrs_combined_full_span_source_policy_profile.py`,
  `src/rabbit/config/backend_capabilities.py`,
  `src/rabbit/config/feature_capabilities.py`, and
  `tests/test_augmented_stability_envelope.py`.
- **Physics added/changed:** no new collision kernel is introduced.  The
  AP4/AP65 diagnostic surfaces now forward
  `electron_chemical_potential_MeV` and `electron_chemical_potential_mode`
  into the AP65 combined nonlinear artifact, so charge-neutral e-/e+ bath
  reconstruction and fixed-`mu_e` splitting are available in the same
  full-span combined angular+radial source path used by the live-RHS budget
  evidence.
- **Scope boundary:** diagnostic full-span artifact/gate/profile only.  This
  does not promote public dispatch, production SMC validation, QKE, or a
  promotion-grade full-BBN span.
- **Exit gate:** focused tests lock spec validation, artifact argument
  forwarding, CLI argument parsing, source-policy profile forwarding, and
  registry claim text.
- **Verification:** the focused combined full-span/profile slice passed
  locally.  A real charge-neutral live-RHS gate smoke run at
  `N_span=(0, 1e-14)` reported `passed=true`,
  `source_evaluation_max=7.0`, and
  `collision_dA_abs_max_final=2.8419082353048944e-4`.  A real charge-neutral
  frozen/live source-policy profile over `1e-14,1e-12` reported
  `passed=true`, `row_count=4`, `failure_rows=0`, frozen/live source
  evaluations `1/7`, and
  `collision_dA_abs_max_final=2.841908235304908e-4`.
- **Known red tests:** focused tests first failed because the combined
  full-span spec/profile builders and both CLIs did not accept
  `electron_chemical_potential_mode`.

---

### AP4-AP65-COMBINED-ELECTRON-MU-OBSERVABLES  Combined full-span electron-bath observables

- **Status:** stage-recorded (historical; not current capability)
- **Scope:** propagate the electron chemical-potential and charge-asymmetry
  state already evolved by the nonlinear 3T solve into the AP65 combined
  artifact and AP4/AP65 full-span gate/profile summaries.
- **Key files:** `src/rabbit/validation/augmented_convergence.py`,
  `src/rabbit/validation/augmented_stability.py`,
  `scripts/run_augmented_nonlrs_nonlinear_combined_collision_3t_solve_artifact.py`,
  `scripts/run_augmented_nonlrs_combined_full_span_3t_candidate_gate.py`,
  `scripts/run_augmented_nonlrs_combined_full_span_source_policy_profile.py`,
  and focused augmented tests.
- **Physics added/changed:** no new collision kernel is introduced.  The
  AP65 combined nonlinear artifact now records terminal electron-mu,
  electron-mu/T, charge-asymmetry density, evolved-state marker, and
  charge-asymmetry thermo-feedback correction observables.  The AP4/AP65
  gate/profile summaries aggregate electron-mu and charge-asymmetry maxima
  over the same diagnostic combined full-span rows.
- **Scope boundary:** diagnostic artifact and gate/profile observables only.
  This does not promote public dispatch, production SMC validation, QKE, or a
  promotion-grade full-BBN span.
- **Verification:** a real charge-neutral AP65 combined smoke artifact reported
  `electron_chemical_potential_MeV_final=3.3006028673558046e-10` and
  `electron_charge_asymmetry_density_MeV3_final=6.623064974048602e-11`.  The
  AP4/AP65 charge-neutral full-span gate/profile summaries reported
  `electron_chemical_potential_abs_max=3.298439955909406e-10` and
  `electron_charge_asymmetry_density_abs_max_MeV3=6.618724826621509e-11`.
- **Known red tests:** focused tests first failed because AP65 artifact result
  metadata and AP4/AP65 gate/profile summaries did not expose electron-bath
  observables.

---

### AP4-AP65-COMBINED-EXACT-QED-FULL-SPAN  Combined full-span scalar QED routing

- **Status:** stage-recorded (historical; not current capability)
- **Scope:** route the existing opt-in `exact_finite_mu_scalar` scalar QED EOS
  mode through the AP17/AP36 LRS 3T convergence/collision-feedback surfaces,
  AP42/AP45 source-only non-LRS artifact/profile surfaces, the AP49 direct
  angular artifact, the direct and nonlinear non-LRS `pstf_radial`/combined
  AP6/AP65 artifacts, and the AP4/AP65 combined full-span candidate gate/source-policy profile CLIs.
- **Key files:** `src/rabbit/validation/augmented_convergence.py`,
  `src/rabbit/validation/augmented_stability.py`,
  `scripts/run_augmented_3t_collision_feedback_artifact.py`,
  `scripts/run_augmented_nonlrs_3t_collision_feedback_artifact.py`,
  `scripts/run_augmented_nonlrs_3t_collision_feedback_source_policy_artifact.py`,
  `scripts/run_augmented_nonlrs_pstf_radial_collision_3t_solve_artifact.py`,
  `scripts/run_augmented_nonlrs_nonlinear_combined_collision_3t_solve_artifact.py`,
  full-span gate/profile CLIs, and focused augmented tests.
- **Physics added/changed:** no new tensor or anisotropic QED physics is added.
  The landed scalar finite-`mu_e` exact QED thermo mode is now executable from
  the combined full-span diagnostic surfaces and recorded in artifact inputs,
  result metadata where available, CLI summaries, and gate observables.
- **Scope boundary:** diagnostic scalar EOS routing only.  This does not promote
  public dispatch, production SMC validation, QKE, anisotropic/tensor QED, or a
  promotion-grade physical full-BBN exact-QED span.
- **Exit gate:** API and CLI tests lock model validation/forwarding; the
  AP4/AP65 gate records one-hot QED model observables and a real exact-scalar
  tiny-span smoke row.
- **Verification:** `PYTHONPATH=src pytest -q tests/test_augmented_stability_envelope.py -k "combined_full_span_gate"`
  reported `10 passed, 52 deselected`; the focused convergence/CLI slice
  reported `30 passed, 76 deselected`; registry sync tests reported
  `27 passed, 3 skipped`; `python -m py_compile` and `git diff --check` passed
  for the touched validation modules, scripts, and tests.  A real AP4/AP65 exact-scalar
  smoke gate reported `T_gamma_final=0.7999999999999922`,
  `H_rate_s_final=0.4311207244202148`, and
  `qed_correction_model_exact_finite_mu_scalar=1.0`.
- **Known red tests:** the first focused tests failed because the AP4/AP65 spec
  and CLIs did not accept `qed_correction_model`; a one-off real exact-scalar
  smoke run then failed until the AP65 artifact surface accepted and forwarded
  the same model key.  Follow-up review regressions first failed on AP36/AP42,
  AP45, and AP49 silent no-op forwarding paths before those wrappers were
  aligned with the same validated model key.

---

### AP4-AP65-COMBINED-SPAN-CACHE-REUSE  Combined full-span radial-grid cache reuse

- **Status:** stage-recorded (historical; not current capability)
- **Scope:** finish the current CPU optimisation pass for the AP4/AP65
  combined angular+`pstf_radial` full-span gate and frozen/live source-policy
  profile by sharing the AP6 radial-grid cache across span rows and matched
  policy rows.
- **Key files:** `src/rabbit/validation/augmented_stability.py`,
  `scripts/run_augmented_nonlrs_combined_full_span_3t_candidate_gate.py`,
  `scripts/run_augmented_nonlrs_combined_full_span_source_policy_profile.py`,
  `src/rabbit/config/backend_capabilities.py`,
  `src/rabbit/config/feature_capabilities.py`, and
  `tests/test_augmented_stability_envelope.py`.
- **Physics added/changed:** none.  The nonlinear 3T solve still evaluates the
  AP6 collision source from the current live augmented distribution.  Only
  invariant descriptor/radial-grid construction is cached across rows with the
  same geometry and electron-bath settings.
- **Runtime/cache controls:** the candidate gate now creates one radial-grid
  cache per report and forwards it to every nested AP65 combined solve
  artifact.  The source-policy profile creates one cache shared across both
  frozen-initial-state and `live_rhs` policy reports.  Artifact and CLI
  summaries expose `radial_grid_cache_entries`.
- **Performance before / after:** a two-span AP65 comparison at
  `N_span=(0, 1e-14)` and `(0, 1e-12)`, `live_rhs`, `N_q=3`, `N_mu=3`,
  `N_phi=5`, and `standard_3t_plasma` measured
  `separate_cache_s=5.539025628997479`, `shared_cache_s=3.3420192470075563`,
  `speedup=1.6573889076064183`, `shared_cache_entries=18`, and passing rows
  with source evaluations `7/7`.
- **Scope boundary:** CPU runtime/cache reuse only.  This does not pretabulate
  evolving distributions, does not alter the AP6 collision operator, does not
  promote full-span live-RHS coupling, does not add public dispatch or
  production SMC evidence, and keeps QKE out of scope.
- **Exit gate:** focused tests lock shared cache object forwarding through the
  span ladder and source-policy profile, JSON summary cache-entry reporting,
  and the existing real combined full-span smoke paths.
- **Known red tests:** focused tests first failed in the prototype because
  profile monkeypatch fakes did not accept the new `radial_grid_cache` keyword.

---

### AP4-AP65-COMBINED-RADIAL-CLOSURE-OBSERVABLES  Combined full-span radial conserved-moment closure

- **Status:** stage-recorded (historical; not current capability)
- **Scope:** require the AP4/AP65 combined angular+`pstf_radial` full-span
  candidate gate to consume and record the AP6 radial conserved-moment closure
  diagnostics from the actual combined-source payload, rather than treating the
  AP6 source-budget contract as a separate artifact-only proof.
- **Key files:** `src/rabbit/validation/augmented_stability.py`,
  `tests/test_augmented_stability_envelope.py`,
  `src/rabbit/config/feature_capabilities.py`,
  `src/rabbit/config/backend_capabilities.py`, and this roadmap/WBS surface.
- **Physics added/changed:** no new collision process family is introduced in
  this PR.  The combined full-span solve now enforces the already-landed AP6
  radial closure physics at the live solve boundary: all-nine diagonal
  `nu-nu` radial number projection is required, all six off-diagonal ordered
  `nu-nu` rows must be number projected, and the three unordered off-diagonal
  `nu-nu` bank pairs must carry the pair-energy neutral projection residual.
- **Scope boundary:** this closes a concrete AP4 diagnostic full-span claim for
  the current no-QKE HM catalog.  It does not add QKE, public dispatch,
  production SMC validation, process families outside the supported finite-mass
  electromagnetic plus diagonal `nu-nu` catalog, or promotion-grade full-BBN
  support.
- **Exit gate:** the AP4/AP65 candidate gate now fails closed on missing
  radial number or off-diagonal pair-energy projection markers and records the
  closure observables in each case and artifact summary.
- **Verification:** a real `live_rhs` RK23 two-span smoke over
  `N_span=(0, 1e-14)` and `(0, 1e-12)`, `N_q=3`, `N_mu=3`, `N_phi=5`,
  `standard_3t_plasma`, and `max_pstf_radial_source_evaluations=256` passed
  with `source_evaluations=7`, all-nine radial number projection enabled,
  `n_radial_nunu_number_projected_sources_final=9`,
  `n_radial_offdiagonal_nunu_number_projected_sources_final=6`,
  `n_radial_offdiagonal_nunu_pair_energy_projected_sources_final=6`,
  `n_radial_offdiagonal_nunu_pair_energy_projected_pairs_final=3`,
  `radial_nunu_max_abs_number_moment_final=1.0508502501873664e-20`, and
  `radial_offdiagonal_nunu_pair_max_abs_energy_residual_final=7.707999820014133e-20`.
- **Known red tests:** the new focused regression first failed because the
  gate accepted cases missing the off-diagonal pair-energy projection marker
  and did not surface the closure observables in the report.

---

### AP4-AP65-COMBINED-PHYSICAL-PREVIEW-PRESET  Combined full-span physical-preview ladder

- **Status:** stage-recorded (historical; not current capability)
- **Scope:** add an opt-in longer-span diagnostic preset to the AP4/AP65
  combined angular+`pstf_radial` full-span candidate gate, so the already
  landed combined 3T/network path can be run beyond the smoke and warm
  live-RHS spans without claiming promoted full-BBN support.
- **Key files:** `scripts/run_augmented_nonlrs_combined_full_span_3t_candidate_gate.py`,
  `src/rabbit/validation/augmented_stability.py`,
  `tests/test_augmented_stability_envelope.py`,
  `src/rabbit/config/feature_capabilities.py`,
  `src/rabbit/config/backend_capabilities.py`, this roadmap/WBS surface, and
  `docs/audit/PR-AP4_combined_full_span_physical_preview_2026-05-17.md`.
- **Physics added/changed:** no new collision process family is introduced.
  The preset uses the existing combined AP65 nonlinear 3T/network solve surface
  and freezes the collision source at the initial augmented state while
  allowing the thermo/Hubble and weak network states to evolve over longer
  spans.  AP6 radial number and pair-energy closure diagnostics remain active
  on the combined source payload.
- **Preset contract:** `--preset physical_preview` resolves to
  `N_span=(0, 1e-6)`, `(0, 1e-4)`, and `(0, 1e-3)`,
  `source_update_policy=frozen_initial_state`, `method=Radau`,
  `max_pstf_radial_source_evaluations=64`, and `max_nfev=200000`.  Dry-run
  JSON records the resolved method and separates
  `routine_numeric_gate_spans=[(0,1e-6),(0,1e-4)]` from
  `isolated_diagnostic_spans=[(0,1e-3)]` with an explicit isolated-process
  marker for the long diagnostic row.  CLI overrides of preset-defining fields
  are relabeled `custom`, and direct spec construction with
  `span_ladder_preset="physical_preview"` rejects any mismatch with this
  frozen-source Radau contract.
- **Scope boundary:** this is frozen-source physical-preview evidence, not
  live-RHS full-BBN evidence.  It does not promote public dispatch, production
  SMC validation, QKE, or promotion-tolerance coupled convergence.
- **Exit gate:** focused tests lock the dry-run preset contract, override
  relabeling, direct-spec fail-closed validation, and registry wording, while a
  slow numeric gate records finite terminal thermo/network values and AP6
  off-diagonal `nu-nu` pair-energy closure on the stable `1e-6` and `1e-4`
  frozen-source preview rows.  The isolated `1e-3` row remains recorded
  diagnostic evidence, not a routine pass/fail stability claim.
- **Verification:** a real isolated preset CLI run passed in
  `elapsed_s=29.111393796047196`.  The rows reported
  `T_gamma_final=0.7999992142768074`, `H_rate_s_final=0.4315478525579576`,
  `Xn_final=0.13000006723216642`, and `nfev=22` at `N_span=1e-6`;
  `T_gamma_final=0.7999214316323131`, `H_rate_s_final=0.4314627401457326`,
  `Xn_final=0.1300112257070123`, and `nfev=420` at `N_span=1e-4`; and
  `T_gamma_final=0.7992146754386976`, `H_rate_s_final=0.4306897631857426`,
  `Xn_final=0.1927605271823484`, and `nfev=10634` at `N_span=1e-3`.  Every
  row retained
  `radial_offdiagonal_nunu_pair_max_abs_energy_residual_final=9.571472303973594e-20`.
- **Known red tests / blockers:** a frozen-source `Radau` probe at
  `N_span=1e-2` returned a non-success result with overflow warning and
  unusable terminal values.  After a `piecewise_frozen` source-refresh solve in
  the same process, the frozen-source `Radau` `N_span=1e-3` row can fail with
  non-finite network values, so long routine evidence moved to the nonuniform
  `piecewise_frozen` gate.  CPU live-RHS evolution at `N_span=1e-6` was
  timeout-level in smoke settings.

---

### AP4-AP65-COMBINED-PIECEWISE-FROZEN-SOURCE-REFRESH  Combined full-span state-handoff source refresh

- **Status:** stage-recorded (historical; not current capability)
- **Scope:** add a `piecewise_frozen` source-update policy to the AP4/AP65
  combined full-span candidate gate, bridging the gap between fully frozen
  initial-source diagnostics and prohibitively slow full live-RHS updates.
- **Key files:** `src/rabbit/validation/augmented_stability.py`,
  `scripts/run_augmented_nonlrs_combined_full_span_3t_candidate_gate.py`,
  `tests/test_augmented_stability_envelope.py`,
  `src/rabbit/config/feature_capabilities.py`,
  `src/rabbit/config/backend_capabilities.py`, this roadmap/WBS surface, and
  `docs/audit/PR-AP4_combined_full_span_piecewise_frozen_2026-05-17.md`.
- **Physics added/changed:** the policy recomputes the AP41 angular plus AP6
  `pstf_radial` combined source at each explicit subspan boundary from the
  current Sigma/A/T/X state, integrates that subspan with the recomputed source
  frozen, then hands the terminal state to the next subspan.  AP6 radial
  number and pair-energy closure diagnostics remain active on the terminal
  combined source payload.
- **Scope boundary:** this is an operator-split source-refresh diagnostic.  It
  is not full live-RHS collision coupling, not QKE, not public dispatch, not
  production SMC validation, and not promotion-tolerance full-BBN evidence.
- **Exit gate:** focused tests lock CLI dry-run forwarding of
  `source_update_subspan_ends`, direct numeric state-handoff execution,
  source-evaluation budget accounting, finite terminal thermo/network values,
  and AP6 off-diagonal pair-energy closure.
- **Verification:** a real custom CLI run over `N_span=(0, 1e-4)`,
  `source_update_subspan_ends=(5e-5, 1e-4)`, `Radau`,
  `max_pstf_radial_source_evaluations=8`, and `max_nfev=5000` passed with
  `source_update_subspan_count=2`, `source_evaluations=2`,
  `source_diagnostic_evaluations=1`, terminal source diagnostics at `N=1e-4`
  after the last refresh at `N=5e-5`, `nfev=1397`,
  `T_gamma_final=0.7999214320646801`, `H_rate_s_final=0.43146274191619854`,
  `Xn_final=0.1300096609235235`,
  `collision_dA_abs_max_final=0.00026527543966857903`, and
  `radial_offdiagonal_nunu_pair_max_abs_energy_residual_final=1.1784345878675453e-19`.
  A longer nonuniform CLI run over `N_span=(0, 1e-3)`,
  `source_update_subspan_ends=(1e-6, 1e-4, 1e-3)`, `Radau`,
  `max_pstf_radial_source_evaluations=8`, and `max_nfev=10000` passed with
  `source_update_subspan_count=3`, `source_evaluations=3`,
  `source_diagnostic_evaluations=1`, terminal source diagnostics at `N=1e-3`
  after the last refresh at `N=1e-4`, `nfev=47`,
  `T_gamma_final=0.7992146796753832`, `H_rate_s_final=0.43068978083203713`,
  `Xn_final=0.13005888045355307`,
  `collision_dA_abs_max_final=0.000265067738217371`, and
  `radial_offdiagonal_nunu_pair_max_abs_energy_residual_final=7.326834993749698e-20`.
- **Known red tests / blockers:** the TDD red tests first failed because the
  CLI rejected `piecewise_frozen` and the spec did not accept
  `source_update_subspan_ends`.  Equal-width `N_span=1e-3` chunks were unstable
  with this Radau setup, so longer piecewise ladders require explicit
  nonuniform subspan control; the `(1e-6, 1e-4, 1e-3)` row is smoke-scale
  diagnostic evidence, not promotion-grade full-BBN stability.

---

### AP4-AP65-COMBINED-PIECEWISE-PHYSICAL-PREVIEW-PRESET  Named piecewise source-refresh preview

- **Status:** stage-recorded (historical; not current capability)
- **Scope:** make the successful nonuniform AP4/AP65 `piecewise_frozen`
  source-refresh ladder a named `piecewise_physical_preview` preset rather
  than only a custom CLI recipe.
- **Key files:** `src/rabbit/validation/augmented_stability.py`,
  `scripts/run_augmented_nonlrs_combined_full_span_3t_candidate_gate.py`,
  `tests/test_augmented_stability_envelope.py`,
  `src/rabbit/config/feature_capabilities.py`,
  `src/rabbit/config/backend_capabilities.py`, this roadmap/WBS surface, and
  `docs/audit/PR-AP4_combined_full_span_piecewise_physical_preview_2026-05-17.md`.
- **Physics added/changed:** no new collision process family is added.  The
  preset uses the already landed AP4/AP65 combined angular+`pstf_radial`
  operator-split path: recompute the source at explicit nonuniform subspan
  boundaries, freeze it within each subspan, and hand off Sigma/A/T/X state.
  AP6 radial number and off-diagonal pair-energy closure diagnostics remain
  enforced on the terminal combined source payload.
- **Preset contract:** `--preset piecewise_physical_preview` resolves to
  `N_span=(0,1e-4),(0,1e-3)`,
  `source_update_policy=piecewise_frozen`,
  `source_update_subspan_ends=(1e-6,1e-4,1e-3)`, `method=Radau`,
  `max_pstf_radial_source_evaluations=8`, and `max_nfev=10000`.  Dry-run
  JSON records routine `N_span=(0,1e-4)` separately from isolated diagnostic
  `N_span=(0,1e-3)` and declares supported electron-bath modes
  `[fixed, charge_neutrality]` plus scalar-QED models
  `[finite_mu_scaled, exact_finite_mu_scalar]`.
- **Exit gate:** focused tests lock the CLI dry-run preset contract and the
  existing piecewise subspan partitioning.  A real artifact run over the named
  preset passed with `span_count=2`, `max_span_length=0.001`,
  `source_evaluation_max=3`, `radial_grid_cache_entries=45`, and no
  violations.
- **Verification:** `PYTHONPATH=src python scripts/run_augmented_nonlrs_combined_full_span_3t_candidate_gate.py --preset piecewise_physical_preview --output /tmp/rabbit_ap4_piecewise_physical_preview.json`
  returned `passed=true`.  The `N_span=(0,1e-4)` row reported
  `T_gamma_final=0.7999214320680265`, `H_rate_s_final=0.43146274191619854`,
  `Xn_final=0.13000591435767977`, `nfev=31`, and off-diagonal pair-energy
  residual `8.205631676526035e-20`; the `N_span=(0,1e-3)` row reported
  `T_gamma_final=0.7992146796753832`, `H_rate_s_final=0.43068978083203713`,
  `Xn_final=0.13005888045355307`, `nfev=47`, and off-diagonal pair-energy
  residual `7.326834993749698e-20`.  The same named preset with
  `electron_chemical_potential_mode=charge_neutrality` also passed through
  `N_span=(0,1e-3)` with
  `electron_chemical_potential_MeV_final=3.295370971985368e-10`,
  `electron_charge_asymmetry_density_MeV3_final=6.59880533199606e-11`,
  `source_update_charge_asymmetry_state_handoff=1`, and no violations.
  With `qed_correction_model=exact_finite_mu_scalar`, the same preset passed
  through `N_span=(0,1e-3)` with
  `qed_correction_model_exact_finite_mu_scalar=1`,
  `T_gamma_final=0.7992145483449656`,
  `H_rate_s_final=0.4302626189334139`, `Xn_final=0.13005893883614722`,
  `nfev=47`, and off-diagonal pair-energy residual
  `9.634999775017666e-20`.
  The combined `electron_chemical_potential_mode=charge_neutrality` plus
  `qed_correction_model=exact_finite_mu_scalar` control row also passed
  through `N_span=(0,1e-3)`, recording
  `electron_chemical_potential_MeV_final=3.295370300138031e-10`,
  `electron_charge_asymmetry_density_MeV3_final=6.598801808206643e-11`,
  `qed_correction_model_exact_finite_mu_scalar=1`,
  `source_update_charge_asymmetry_state_handoff=1`, and no violations.
- **Scope boundary:** this is named operator-split source-refresh preview
  evidence for the existing AP4/AP65 path.  It is not fully live-RHS collision
  coupling, public dispatch, production SMC validation, QKE, or
  promotion-grade full-BBN support.

---

### AP4-AP65-COMBINED-PIECEWISE-REFINEMENT  Piecewise source-refresh refinement artifact

- **Status:** stage-recorded (historical; not current capability).
- **Files:** `src/rabbit/validation/augmented_stability.py`,
  `scripts/run_augmented_nonlrs_combined_full_span_piecewise_refinement.py`,
  `tests/test_augmented_stability_envelope.py`,
  `docs/audit/PR-AP4_combined_full_span_piecewise_refinement_2026-05-17.md`.
- **Scope:** compare nonuniform `piecewise_frozen` source-refresh schedules for
  the AP4/AP65 combined angular+`pstf_radial` nonlinear non-LRS 3T solve over
  the same physical-preview `N_span=(0,1e-3)` span, using a shared AP6
  radial-grid cache and recording refined-minus-reference observable deltas.
- **Exit gate:** focused tests lock the artifact builder and CLI summary.  A
  real artifact run comparing coarse `(1e-6,1e-4,1e-3)` against refined
  `(1e-6,1e-5,1e-4,1e-3)` passed with `schedule_count=2`,
  `source_evaluation_max=4`, `nfev_max=56`, `radial_grid_cache_entries=72`,
  and no violations.
- **Verification:** `PYTHONPATH=src python scripts/run_augmented_nonlrs_combined_full_span_piecewise_refinement.py --output /tmp/rabbit_ap4_piecewise_refinement.json`
  returned `passed=true`.  The refined-minus-coarse deltas included
  `T_gamma_final=-1.1280976153216216e-12`,
  `Xn_final=2.3314683517128287e-15`,
  `collision_dA_abs_max_final=4.4915406029882865e-14`,
  off-diagonal pair-energy residual `2.0117032497289633e-21`,
  `source_evaluations=1`, and `nfev=9`.
- **Scope boundary:** this is operator-split source-refresh refinement
  evidence.  It is not continuous live-RHS collision coupling, public dispatch,
  production SMC validation, QKE, or promotion-grade full-BBN support.

---

### AP4-AP65-PIECEWISE-TERMINAL-QED-FORWARDING  Terminal source scalar-QED forwarding

- **Status:** stage-recorded (historical; not current capability).
- **Files:** `src/rabbit/transport/augmented_typeI_weak_network.py`,
  `src/rabbit/validation/augmented_stability.py`,
  `tests/test_augmented_stability_envelope.py`,
  `docs/audit/PR-AP4_piecewise_terminal_qed_forwarding_2026-05-17.md`.
- **Scope:** forward the selected `qed_correction_model` into the
  `piecewise_frozen` terminal source re-evaluation and record finite/exact
  scalar-QED one-hot diagnostics on that combined source.
- **Exit gate:** focused regression first failed because terminal source kwargs
  omitted `qed_correction_model`, then passed after forwarding
  `spec.qed_correction_model`.
- **Verification:** a real charge-neutral plus `exact_finite_mu_scalar`
  `piecewise_physical_preview` run still passed with `span_count=2`,
  `source_evaluation_max=3`,
  `electron_chemical_potential_abs_max=3.298436883301101e-10`,
  `electron_charge_asymmetry_density_abs_max_MeV3=6.618704871052225e-11`,
  and no violations.
- **Scope boundary:** this is terminal diagnostic forwarding for an already
  selected scalar-QED control.  It is not anisotropic/tensor QED response,
  QKE, public dispatch, production SMC validation, or promotion-grade full-BBN
  support.

---

### AP4-AP65-COMBINED-CHARGE-NEUTRAL-PIECEWISE-HANDOFF  Piecewise charge-neutral e-/e+ state handoff

- **Status:** stage-recorded (historical; not current capability)
- **Scope:** remove the fixed-electron-mode limitation from the AP4/AP65
  `piecewise_frozen` combined full-span source-refresh path.
- **Key files:** `src/rabbit/transport/augmented_typeI_weak_network.py`,
  `src/rabbit/transport/augmented_collision_bridge.py`,
  `src/rabbit/validation/augmented_stability.py`,
  `tests/test_augmented_typeI_weak_network_3t_solve.py`,
  `tests/test_augmented_stability_envelope.py`,
  `src/rabbit/config/feature_capabilities.py`,
  `src/rabbit/config/backend_capabilities.py`, this roadmap/WBS surface, and
  `docs/audit/PR-AP4_combined_full_span_piecewise_charge_neutrality_2026-05-17.md`.
- **Physics added/changed:** charge-neutral finite-mass e-/e+ 3T solves can
  now accept an initial evolved charge-asymmetry density for a subspan.  The
  AP4/AP65 piecewise gate carries
  `electron_charge_asymmetry_density_MeV3_final` from one subspan into the next
  subspan's Hubble/RHS initialization and radial electron/positron bath source
  payload, rather than recomputing each chunk from only the initial abundances
  or falling back to fixed `mu_e`.
- **Scope boundary:** this closes a concrete operator-split handoff gap for the
  existing charge-neutral 3T electron bath.  It is not full live-RHS collision
  coupling, not QKE, not public dispatch, not production SMC validation, and
  not promotion-grade full-BBN span evidence.
- **Exit gate:** focused tests lock lower-solver acceptance of an initial
  charge-asymmetry state, piecewise subspan handoff into the second chunk,
  terminal source diagnostics with charge-neutral mode, and a real
  charge-neutral two-subspan smoke run.
- **Verification:** a real custom CLI run over `N_span=(0, 1e-4)`,
  `source_update_subspan_ends=(5e-5, 1e-4)`, `Radau`,
  `electron_chemical_potential_mode=charge_neutrality`,
  `max_pstf_radial_source_evaluations=8`, and `max_nfev=10000` passed with
  `source_update_subspan_count=2`,
  `source_update_charge_asymmetry_state_handoff=1`,
  `source_evaluations=2`, `source_diagnostic_evaluations=1`, `nfev=8256`,
  `electron_chemical_potential_MeV_final=3.298132792363573e-10`,
  `electron_charge_asymmetry_density_MeV3_final=6.61672994731945e-11`,
  `T_gamma_final=0.7999214313698554`,
  `H_rate_s_final=0.43146274153219083`,
  `Xn_final=0.13000851100280264`,
  `collision_dA_abs_max_final=0.00026527544839372085`, and
  `radial_offdiagonal_nunu_pair_max_abs_energy_residual_final=1.1784345878675453e-19`.
- **Known red tests / blockers:** the focused handoff regression first failed
  because the nonlinear non-LRS 3T solve did not accept an initial
  charge-asymmetry state.  The piecewise stability regression first failed
  because the gate rejected `piecewise_frozen` unless the electron mode was
  fixed.  A charge-neutral `max_nfev=5000` run reached the valid terminal source
  path but failed only the effort limit with `nfev=8256`, so the smoke gate uses
  `max_nfev=10000`.

---

## Augmented Type-I anti-drift correction (2026-05-20)

The May 2026 audits found that recent AP/FB work had drifted toward
gate-only progress: many diagnostic wrappers, figure/readiness manifests, hash
relays, and hot-endpoint span scouts were added while the continuous AP65
full-BBN endpoint blocker remained open.  The controlling guardrail is now
`docs/TYPEI_AUGMENTED_NOQKE_CODEX_ANTI_DRIFT_GUARDRAILS.md`.

Forward catalogue entries for this line must now satisfy a stricter standard:
each PR must retire or measurably reduce a named physics, solver, or performance
blocker, or consolidate/delete older gate plumbing.  Standalone FB90-style
solver-policy atlases, publication wrappers, internal dispatch decision gates,
and readiness relays are not acceptable next steps unless folded into that
consolidation.  The breakthrough DAG is BD0 through BD9 in the guardrail doc:
consolidate gates, move provenance out of the continuous AP65 hot loop, replace
full finite-difference Jacobian probes, reach endpoint in LRS first, then expand
non-LRS/collision freedoms, and only then rebuild endpoint-backed figures/SMC.

## Refreshed AP4 full-BBN close-out DAG (2026-05-17)

The current scan shows that the lowest unblocked path to a concrete
diagnostic full-BBN run is not another pass/fail wrapper.  It is to reuse the
landed AP4/AP65 `piecewise_frozen` combined angular+`pstf_radial` path,
export restartable state, then chain physical windows until the already
landed AP66-AP76 publication and SMC surfaces can consume real full-chain
artifacts.  This section is a planning catalog for AP4 close-out packages; it
does not change AP0-AP81 statuses and does not create public canonical
dispatch.  Backend policy for the close-out is explicit: SciPy is the
reference/source-generation shell for current AP4/AP65 physics, while repeated
full-chain and SMC execution must move toward CPU-first JAX with the in-tree
Rosenbrock/Rodas5P solver as soon as the restart/replay state layout is
available.

### FB-01-TERMINAL-STATE-RESTART-PAYLOAD  Restartable AP4/AP65 state export

- **Status:** stage-recorded (historical; not current capability).
- **Depends on:** AP4/AP65 piecewise physical-preview and charge-neutral
  handoff.
- **Scope:** export compact terminal-state payloads from
  `AugmentedNonLRSCombinedFullSpan3TCandidateGateCase` rows: shear,
  temperatures, phase-2 network abundances, final augmented modes, electron
  charge-asymmetry state, selected scalar-QED model, terminal source
  diagnostics, and source-refresh metadata.
- **Scope boundary:** restart payloads are infrastructure for chained
  diagnostic runs and CPU-JAX/Rodas5P replay handoff, not a new physics model
  and not public dispatch.
- **Exit gate:** focused tests prove payloads are finite, JSON-safe, and can be
  converted back into the next AP4/AP65 solve inputs for fixed,
  charge-neutral, finite-mu-scaled, and exact-scalar-QED controls.
- **Verification:** focused FB-01 regressions passed, including piecewise
  charge-neutral/exact-QED handoff, JSON artifact terminal-state persistence,
  and the existing combined full-span JSON artifact contract.

### FB-02-CHAINED-PIECEWISE-FULL-BBN-RUNNER  Multi-window physical-chain runner

- **Status:** stage-recorded (historical; not current capability).
- **Depends on:** FB-01.
- **Scope:** add a checkpointed runner that advances through a sequence of
  physical windows, feeding each terminal payload into the next window while
  preserving the AP6 radial-grid cache and source-refresh metadata.  This PR
  must also freeze the CPU-JAX/Rodas5P replay state layout for
  piecewise/pretabulated source payloads so later warm/full SMC does not depend
  on long SciPy repeated solves.
- **Scope boundary:** this remains operator-split `piecewise_frozen` evidence,
  not continuous live-RHS collision coupling; SciPy remains reference
  execution, not the intended high-throughput backend.
- **Exit gate:** a smoke artifact chains at least two windows beyond the
  current single-window preview, records restart/resume equivalence, finite
  terminal thermodynamics/network values, cumulative source-budget accounting,
  and a Rodas5P-ready replay bundle.
- **Verification:** `tests/test_augmented_stability_envelope.py` locks full
  `Sigma/A/T/X/electron` handoff, JSON persistence, restart/resume equality,
  and CLI dry-run metadata; `tests/test_jax_augmented_typeI_replay.py` locks
  the CPU-JAX pack/unpack state layout.  A real smoke command over
  `N_window_edges=(0,1e-8,2e-8)` produced two successful windows, two replay
  payloads, zero restart/resume delta, and no violations.

### FB-03-ADAPTIVE-SOURCE-REFRESH-SCHEDULER  Drift-budgeted source refresh

- **Status:** stage-recorded (historical; not current capability).
- **Depends on:** FB-02.
- **Scope:** replace fixed subspan recipes with an adaptive scheduler keyed to
  temperature drift, collision `dA`, abundance drift, and source-evaluation
  budget while preserving the fixed `uniform` schedule for comparisons.
- **Scope boundary:** the scheduler controls operator splitting only; it does
  not claim a promoted live collision RHS and does not make the SciPy
  source-generation shell the intended high-throughput backend.
- **Exit gate:** chained-runner artifacts record `source_refresh_strategy`,
  per-window subspan counts, driver estimates, and budget-limited windows with
  explicit source-evaluation counts.
- **Verification:** focused FB-03 regressions lock the adaptive scheduler cap,
  the chained-runner summary fields, and CLI dry-run controls.  The next
  physical stage should use these schedules to drive CPU-JAX/Rodas5P replay
  RHS implementation instead of adding more long repeated SciPy solves.

### FB-04-PHYSICAL-BBN-SPAN-OBSERVABLES  Full-chain BBN observable extraction

- **Status:** stage-recorded (historical; not current capability).
- **Depends on:** FB-02 and FB-03.
- **Scope:** tie the chain to CPU-JAX/Rodas5P repeated execution and the
  production BBN thermal-span conventions, then extract `Yp`, `D/H`,
  `N_eff`-style radiation summaries, terminal network rows, and Schramm
  coordinates from the same run.
- **Scope boundary:** observable extraction is diagnostic until convergence,
  weak-rate, and SMC gates consume the same artifact family.
- **Current evidence:** the chained artifact now executes an in-tree
  CPU-JAX/Rodas5P pretabulated window-map replay solve for each successful
  SciPy source-generation window and records replay pass/error metadata.  It
  also exposes an opt-in CPU-JAX/Rodas5P live-source RHS sidecar whose RHS
  reconstructs the current S2 distribution and evaluates source-only non-LRS
  stress/transport, 3T/Hubble thermo, live weak monopole rates, and the PRIMAT
  phase-2 network in JAX, with any collision moments still frozen payload
  terms.  The artifact emits finite terminal `Yp`, `D/H`, `N_eff_3T`,
  `Sigma_H`, and Schramm coordinates from the same phase-2 terminal state, and
  records the enabled live-source RHS sidecar's own final-state
  `Yp`/`D/H`/`N_eff_3T`/`Sigma_H`/Schramm readout and sidecar-vs-terminal
  observable deltas as separate summary metadata.  The sidecar final state also
  exports restart kwargs that can seed the next CPU-JAX/Rodas5P live-source
  window, and a dedicated CPU-JAX live-source RHS chain runner exercises that
  handoff across consecutive smoke windows with a JSON artifact CLI.  The FB-04
  chained artifact can optionally attach that live-source chain as finite-delta
  diagnostic comparison evidence against the same piecewise/window-map rows and
  can feed it frozen per-window terminal collision payloads with supplied/applied
  payload counts and provenance fingerprints.  The same chain can now be selected as the staged
  CPU-JAX/Rodas5P repeated-run evidence/readout source through
  `rodas5p_repeated_run_source="live_source_rhs_chain"`.  A CPU
  smoke over two `(1e-10)` windows with that repeated-run source enabled passed
  with two completed live-source RHS chain windows, two frozen collision
  payloads with provenance fingerprints, finite BBN deltas, `rodas5p_repeated_run_source_ready=true`, and
  `public_dispatch_ready=false`.
  This is executable Rodas5P live-source sidecar evidence plus staged repeated-run
  observable readout, not yet a full collision-coupled live JAX replacement or
  production-calibrated full-span BBN yield.
- **Exit gate:** one diagnostic full-chain artifact emits finite BBN
  observables with explicit `qke_scope=out_of_scope`,
  `public_dispatch_ready=false`, replay metadata from the same windows, and
  opt-in live-source RHS sidecar metadata when requested.

### FB-05-LIVE-RHS-MICRO-WINDOW-COMPARATOR  Tractable live-vs-piecewise physics check

- **Status:** stage-recorded (historical; not current capability).
- **Depends on:** FB-02.
- **Scope:** run `live_rhs` only on short windows where it is numerically
  tractable, comparing terminal source, thermo, weak-rate, and network deltas
  against the piecewise chain.
- **Scope boundary:** live RHS remains comparison evidence, not default policy.
- **Current evidence:** `augmented_nonlrs_live_rhs_micro_window_comparator_ap4_fb05_v1`
  rebuilds each chained restart state as a live-RHS AP65 micro-window and
  records live-minus-piecewise thermo, network, source-moment, and kinetic
  `dA` deltas with pass/live-RHS/budget/non-finite/delta classifications.  A
  smoke CPU artifact over two `(1e-14)` windows passed with no violations,
  max live source evaluations `7`, zero `T_gamma` delta, and
  `8.326672684688674e-17` `Xn` delta.
- **Exit gate:** satisfied for smoke scale.  Longer windows and promotion
  gates remain future work; this does not make `live_rhs` the default policy.

### FB-06-COLLISION-LEDGER-OVER-CHAIN  Per-window number/energy closure

- **Status:** stage-recorded (historical; not current capability).
- **Depends on:** FB-02.
- **Scope:** promote existing terminal radial closure observables into a
  per-window and cumulative ledger for electromagnetic and all-nine diagonal
  no-QKE `nu-nu` process rows.
- **Scope boundary:** ledger closure is necessary evidence, not sufficient for
  production support.
- **Current evidence:** `augmented_nonlrs_collision_ledger_over_chain_ap4_fb06_v1`
  runs over the chained full-BBN artifact rows, extracts AP6/AP65 terminal
  `pstf_radial` diagnostics for finite-mass electromagnetic source counts,
  standard-3T plasma energy-closure residuals, all-nine diagonal `nu-nu`
  source counts, identical-bank number/energy-neutral projection, and
  off-diagonal unordered-pair energy closure, then records per-window and
  cumulative maxima with fail-closed classifications.
- **Exit gate:** satisfied for smoke scale.  A CPU smoke over
  `N_window_edges=(0,1e-8,2e-8)`, `(N_q,N_mu,N_phi)=(3,3,5)`, no replay
  verification, and default diagnostic limits passed with no violations,
  `ledger_rows=2`,
  `radial_em_energy_closure_residual_abs_max=8.445326839929337e-19`,
  `radial_nunu_max_abs_number_moment_max=9.846758011831241e-21`, and
  `radial_offdiagonal_nunu_pair_max_abs_energy_residual_max=1.0482032722271967e-19`.

### FB-07-CHAINED-WEAK-RATE-CONVERGENCE  Coupled weak-rate evidence on real rows

- **Status:** stage-recorded (historical; not current capability).
- **Depends on:** FB-02.
- **Scope:** run the AP77/AP80 weak-rate diagnostics on chained rows using the
  same non-LRS S2 CL3 mode and same-CL3 controls.  Terminal AP4/AP65 chained
  states now preserve `weak_rates_final` and `weak_angular_metadata`, and the
  FB-07 artifact runs paired `metadata_only` and
  `nonlrs_s2_cl3_quadrupole_input` chains with identical injected CL3 state
  controls.
- **Scope boundary:** this closes the metadata-only weak-rate gap only for the
  tested chained rows.  It remains diagnostic, SciPy source-generation first,
  and not public dispatch or production SMC evidence.
- **Exit gate:** artifact records weak-rate mode, same-CL3 controls, finite
  lambda deltas, and bounded abundance deltas.
- **Artifact/CLI:** `augmented_nonlrs_chained_weak_rate_convergence_ap4_fb07_v1`
  via
  `scripts/run_augmented_nonlrs_chained_weak_rate_convergence.py`.
- **Verification:** CPU smoke with
  `N_window_edges=(0,1e-8,2e-8)`, `(N_q,N_mu,N_phi)=(3,3,5)`, no replay
  verification, and bounded diagnostic limits passed with no violations,
  `comparison_rows=2`,
  `lambda_np_relative_delta_abs_max=6.910814616395567e-05`,
  `lambda_pn_relative_delta_abs_max=4.6906512614639065e-05`, and
  `Xn_final_delta_abs_max=4.3565151486291143e-13`.

### FB-08-ELECTRON-QED-CHAIN-CROSS-PRODUCT  e-/e+ and scalar-QED chained controls

- **Status:** stage-recorded (historical; not current capability).
- **Depends on:** FB-02.
- **Scope:** carry the current fixed/charge-neutral electron-bath and
  finite/exact scalar-QED control cross-product through chained windows.  The
  FB-08 artifact runs four AP4/AP65 chained artifacts over the same restart
  windows: `fixed/finite_mu_scaled`, `fixed/exact_finite_mu_scalar`,
  `charge_neutrality/finite_mu_scaled`, and
  `charge_neutrality/exact_finite_mu_scalar`.
- **Scope boundary:** this remains scalar QED only; anisotropic/tensor QED
  response and promotion-grade exact-scalar-QED full-span coupled-solver
  validation remain blocked.  It is still diagnostic, SciPy source-generation
  first, and not public dispatch or production SMC evidence.
- **Exit gate:** cross-product rows record evolved charge-asymmetry handoff and
  exact-scalar-QED diagnostics for every window.
- **Artifact/CLI:**
  `augmented_nonlrs_electron_qed_chain_cross_product_ap4_fb08_v1` via
  `scripts/run_augmented_nonlrs_electron_qed_chain_cross_product.py`.
- **Verification:** CPU smoke with
  `N_window_edges=(0,1e-8,2e-8)`, `(N_q,N_mu,N_phi)=(3,3,5)`, no replay
  verification, and bounded chained limits passed with no violations,
  `combination_count=4`, `row_count=8`, `failed_rows=0`,
  `charge_neutrality_evolved_rows=4`, and `exact_scalar_qed_rows=4`.

### FB-09-CHAINED-RESOLUTION-LADDERS  q/angular/PSTF resolution on chains

- **Status:** stage-recorded (historical; not current capability).
- **Depends on:** FB-03 and FB-04.
- **Scope:** extend `N_q`, `N_mu`, `N_phi`, and available PSTF/angular ladders
  from single-window smoke to chained-window diagnostics.  The FB-09 artifact
  runs the chained AP4/AP65 full-BBN runner for the `N_q`, `N_mu`, and `N_phi`
  ladders and records terminal BBN observable deltas plus solver/source budget
  metadata for every ladder point.
- **Scope boundary:** smoke-scale convergence rows are not promotion
  tolerances.  The artifact is diagnostic, SciPy source-generation first, and
  not public dispatch or production SMC evidence.
- **Exit gate:** ladder artifacts report terminal observable deltas, source
  budgets, and failure rows.
- **Artifact/CLI:** `augmented_nonlrs_chained_resolution_ladders_ap4_fb09_v1`
  via `scripts/run_augmented_nonlrs_chained_resolution_ladders.py`.
- **Verification:** CPU smoke with
  `N_window_edges=(0,1e-8,2e-8)`, `q=(3,4)`, `N_mu=(3,4)`, `N_phi=(5,6)`,
  no replay verification, and bounded chained limits passed with no violations,
  `row_count=6`, `converged_ladders=3`, `source_evaluations_total=12.0`, and
  `terminal_observable_delta_abs_max=1.679316997614903e-22`.

### FB-10-AP66-FULL-CHAIN-MATRIX  Publication matrix consumes chained rows

- **Status:** stage-recorded (historical; not current capability).
- **Depends on:** FB-09.
- **Scope:** AP66 now accepts an optional FB-09 chained resolution artifact and
  records compact full-chain provenance next to the existing AP66
  publication-candidate matrix rows.
- **Physics added/changed:** no new collision kernel.  The change makes AP66
  consume real chained-window resolution evidence instead of treating the
  publication matrix as only single-span/smoke evidence: it validates the
  `augmented_nonlrs_chained_resolution_ladders_ap4_fb09_v1` contract, requires
  passed/no-QKE/not-public-dispatch/not-production-SMC scope, rejects failed or
  malformed chained rows, and emits `full_chain_evidence.row_links` with the
  source artifact path, contract, row index/key, ladder value, terminal
  `Yp`/`D/H`/`N_eff_3T`/`Sigma_H`, source-evaluation totals, nfev totals, and
  CPU-JAX/Rodas5P replay metadata for every chained `N_q`/`N_mu`/`N_phi` row.
- **Scope boundary:** AP66 remains diagnostic publication-candidate evidence.
  Full-chain provenance does not promote public dispatch, production SMC
  validation, or QKE support.
- **Exit gate:** AP66 JSON links every supplied full-chain row back to its
  source artifact and fails closed on wrong contract, public-dispatch-ready
  artifacts, failed artifacts, failed row links, nonfinite terminal
  observables, or row-count mismatches.
- **Artifact/CLI:** `scripts/run_augmented_publication_convergence_matrix.py`
  accepts `--chained-resolution-artifact`.
- **Verification:** focused AP66 tests first failed because the builder and CLI
  did not accept `chained_resolution_artifact`; after the implementation, the
  focused provenance/CLI tests passed.

### FB-11-AP67-FULL-CHAIN-ATLAS  Known-limit atlas consumes chained rows

- **Status:** stage-recorded (historical; not current capability).
- **Depends on:** FB-10.
- **Scope:** AP67 now consumes the AP66/FB-09 full-chain provenance emitted by
  FB-10 and records it in the validation atlas readiness/evidence ledger.
- **Physics added/changed:** no new collision kernel.  AP67 forwards optional
  FB-09 chained resolution artifacts into nested AP66 convergence evidence,
  can require AP66 full-chain provenance, records AP66 full-chain artifact
  contract/path/row-count/row-keys/no-QKE/not-public/not-production-SMC flags
  in `reused_evidence_links["AP66"]`, and adds
  `readiness_summary.full_chain_evidence_ready`.
- **Scope boundary:** atlas rows validate limits; they do not create public
  dispatch, production SMC evidence, or QKE support.
- **Exit gate:** AP67 rejects AP66 evidence links that are missing full-chain
  provenance when `require_full_chain_evidence=True`, and rejects wrong
  artifact contracts, QKE scope drift, public-dispatch-ready evidence,
  production-SMC-ready evidence, or empty row links.
- **Artifact/CLI:** `scripts/run_augmented_validation_atlas.py` accepts
  `--chained-resolution-artifact` and then requires nested AP66 full-chain
  provenance.
- **Verification:** focused tests first failed because AP67 had no
  `require_full_chain_evidence` or AP66 full-chain link extraction.  After
  implementation, AP67 full-chain focused tests and non-slow AP67 tests passed.
  A CPU smoke with `/tmp/rabbit_fb09_chained_resolution_ladders_smoke.json`
  produced six passing AP67 cases, no limit violations, and
  `full_chain_evidence_ready=true`.

### FB-12-AP68-FULL-CHAIN-FORWARD-MODEL  Guarded full-chain adapter

- **Status:** stage-recorded (historical; not current capability).
- **Depends on:** FB-04 and FB-10.
- **Scope:** add an AP68 guarded forward-model mode that calls the chained
  runner with artifact cache and restart controls.  The landed adapter exposes
  `execution_mode="full_chain"`, builds
  `AugmentedNonLRSChainedFullBBNRunnerSpec` from AP68 parameters and guarded
  solver controls, supports cached artifact consumption, rejects malformed or
  overclaiming chained artifacts, forwards the optional FB-04 live-source RHS
  chain diagnostic/repeated-run source switch, and maps finite chained terminal
  or opt-in live-source repeated-run `Yp`/`D/H` into the existing
  `ForwardModel`/`BBNLikelihood` prediction contract.  It can also validate a
  supplied FB-21 gate artifact or build one on demand as metadata-only evidence
  for the same tiny chained spans; the FB-21 gate does not alter AP68 terminal
  readouts and does not widen dispatch support.
- **Scope boundary:** no `canonical_forward_solver` registration and no public
  default backend; the returned full-chain yields are smoke-scale terminal
  readouts, not production-calibrated BBN yield claims.
- **Exit gate:** `ForwardModel` and `BBNLikelihood` smoke tests return finite
  predictions or structured failure metadata.  Focused AP68 tests lock direct
  builder routing, cached-artifact routing, and fail-closed artifact-scope
  validation; a CPU smoke over two `(1e-8)` windows returned
  `success=True`, finite `Yp`/`D/H`, `full_chain_completed_windows=2`, and
  `public_dispatch_ready=False`.

### FB-13-AP69-FULL-CHAIN-LIKELIHOOD-SCHEMA  Full-chain solver controls

- **Status:** stage-recorded (historical; not current capability).
- **Depends on:** FB-12.
- **Scope:** extend AP69 schema rows with full-chain source policy,
  checkpoint/cache, adaptive-scheduler, electron-bath, and scalar-QED controls.
  The landed schema records `execution_mode`, full-chain source-refresh
  strategy, and replay/restart toggles as solver controls, while metadata and
  AP71 cache/runtime payloads expose full-chain window edges and cache keys.
- **Scope boundary:** schema support is not production SMC evidence by itself.
- **Exit gate:** schema tests lock vector adapters, priors, fixed controls, and
  cache provenance.  A real AP69 likelihood smoke through AP68 full-chain mode
  returned finite log-likelihood with two completed chained windows and
  no-QKE/not-public metadata preserved.

### FB-14-AP70-FULL-CHAIN-SMC-WARM-RUN  Smoke tempered SMC on full-chain adapter

- **Status:** stage-recorded (historical; not current capability).
- **Depends on:** FB-13.
- **Scope:** run a small-particle tempered-SMC warm phase through the
  full-chain adapter with checkpoint/restart and failure-aware likelihoods.
  The landed CLI accepts full-chain execution mode, window edges, cache key,
  cached chained artifacts, adaptive scheduler controls, and replay/restart
  toggles, and the AP70/AP71 result metadata preserves those controls.
- **Scope boundary:** smoke SMC is not production SMC validation.
- **Exit gate:** artifact records real forward-call count, ESS, acceptance,
  temperatures, failures, checkpoints, and wall time.  A real two-particle
  full-chain CLI smoke completed with finite log-likelihoods, zero forward
  failures, two cache misses, and no public/QKE promotion.

### FB-15-AP72-FULL-CHAIN-SYNTHETIC-RECOVERY  Physical smoke on full-chain adapter

- **Status:** stage-recorded (historical; not current capability).
- **Depends on:** FB-14.
- **Scope:** keep AP72 synthetic null/recovery as the default downstream
  artifact, and add an opt-in physical smoke row that runs AP70/AP71 SMC through
  AP68 full-chain candidate forward calls.
- **Scope boundary:** the physical row is a tiny diagnostic full-chain smoke,
  not observational production SMC and not public dispatch.
- **Exit gate:** the physical row reports posterior moments, finite
  log-likelihood counts, forward failures, AP68 `Yp`/`D/H`, completed chained
  windows, CPU-JAX/Rodas5P replay status, and optional live-source repeated-run
  BBN readout plus FB-21 gate provenance from real full-chain calls.

### FB-16-AP73-SCHRAMM-ARTIFACT-ROWS  Publication tables from real rows

- **Status:** stage-recorded (historical; not current capability).
- **Depends on:** FB-10 and FB-14.
- **Scope:** populate AP73 Schramm and publication tables from the AP72
  full-chain physical smoke artifact, not placeholders or synthetic-only rows.
- **Scope boundary:** artifact tables remain diagnostic unless AP76/AP79
  readiness passes; malformed or overclaiming non-synthetic AP72 artifacts fail
  closed.
- **Exit gate:** tables include real AP68 full-chain `Yp`, `D/H`, `eta10`,
  `Sigma_H`, `N_eff_3T`, source policy, completed-window count, CPU-JAX/Rodas5P
  replay status, optional live-source repeated-run BBN readout plus FB-21 gate
  provenance, and artifact fingerprints.

### FB-17-AP74-SCHRAMM-PUBLICATION-PLOTS  Diagnostic plots from real rows

- **Status:** stage-recorded (historical; not current capability).
- **Depends on:** FB-16.
- **Scope:** render diagnostic Schramm, convergence, source-budget, and SMC
  panels from AP73 full-chain tables.  AP74 now accepts AP73 full-chain
  physical-smoke Schramm rows and records physical-smoke provenance in plot
  records and the manifest, including optional live-source repeated-run BBN
  readout plus FB-21 gate contract/claim/window/payload/delta provenance when
  AP73 supplied it.
- **Scope boundary:** plots must label diagnostic/not-promoted scope.
- **Exit gate:** nonempty PNG outputs have provenance, source-artifact
  fingerprints, completed-window counts, and CPU-JAX/Rodas5P replay status for
  full-chain physical-smoke rows, plus optional live-source repeated-run BBN
  readout and FB-21 gate contract/claim/window/payload/delta provenance.

### FB-18-AP75-FULL-CHAIN-BUNDLE  Reproducibility bundle from real rows

- **Status:** stage-recorded (historical; not current capability).
- **Depends on:** FB-17.
- **Scope:** package the AP72 full-chain physical-smoke row and AP74 full-chain
  Schramm plot provenance inside the existing AP75 reproducibility-bundle
  contract, including finite terminal `Yp`/`D/H`, zero forward failures,
  completed-window counts, CPU-JAX/Rodas5P replay status, and optional
  live-source repeated-run BBN readout plus FB-21 gate contract/claim/window/
  payload/delta provenance.
- **Scope boundary:** bundle reproducibility is not readiness approval; this is
  diagnostic physical-smoke evidence and not production SMC validation.
- **Exit gate:** manifest rejects mixed commits, missing source artifacts,
  non-synthetic AP72 artifacts without a passed full-chain physical-smoke row,
  AP74 full-chain Schramm plots without the matching AP72 row, and malformed
  completed-window, replay, live-source BBN readout, or FB-21 gate provenance.

### FB-19-AP76-AP79-FULL-CHAIN-READINESS  Readiness audit with full-chain evidence

- **Status:** stage-recorded (historical; not current capability).
- **Depends on:** FB-18.
- **Scope:** extend the existing AP76/AP79 readiness audit so AP75 bundles that
  carry a passed AP72 full-chain physical-smoke row plus AP74 full-chain Schramm
  provenance are accepted and recorded in the readiness ledger, including
  live-source repeated-run BBN readout only when matching FB-21 gate
  contract/claim/window/payload/delta provenance is present.
- **Scope boundary:** readiness remains `not_promoted`; full-chain physical
  smoke is diagnostic evidence, not production SMC validation.
- **Exit gate:** audit rejects non-synthetic AP72 summaries without a passed
  full-chain physical-smoke summary, AP74 full-chain Schramm provenance without
  the matching AP72 row, malformed completed-window/replay/live-source BBN
  readout fields, stale or partial FB-21 gate provenance, public dispatch,
  production-SMC flags, or QKE scope.

### FB-20-SLOW-PRODUCTION-CANDIDATE-GATE  Optional longer slow gate

- **Status:** stage-recorded (historical; not current capability).
- **Depends on:** FB-19 and AP80.
- **Scope:** add `augmented_production_candidate_gate_fb20_v1`, an optional
  production-candidate evidence gate that consumes an AP79 readiness audit with
  full-chain physical-smoke provenance plus an AP80 extended coupled weak-rate
  convergence artifact.
- **Physics added/changed:** no new collision term.  FB-20 hardens the evidence
  chain by requiring finite AP72/AP75 full-chain `Yp`/`D/H`, completed-window
  counts, CPU-JAX/Rodas5P replay status, zero full-chain forward failures,
  matching AP72/AP74 route summaries plus FB-21 gate provenance for live-source
  repeated-run BBN readout evidence, and an AP80 q=(3,4,5)-class profile
  before a candidate pass can be recorded.
- **Scope boundary:** the gate can pass while `promotion_decision` remains
  `not_promoted`; it does not enable public canonical dispatch, production SMC
  validation, QKE, or a real-data production likelihood.
- **Exit gate:** focused tests lock acceptance of matching AP79/AP80 evidence,
  rejection of missing full-chain readiness, rejection of missing/failed AP80
  extended convergence, rejection of stale AP72/AP74 live-source route summaries,
  rejection of missing or mismatched FB-21 live-source repeated-run gate
  provenance, JSON writing, and CLI dry-run claim-boundary output.

### FB-21-LIVE-SOURCE-REPEATED-RUN-GATE  Fail-closed live-source readout evidence

- **Status:** stage-recorded (historical; not current capability).
- **Depends on:** FB-04.
- **Scope:** add `augmented_nonlrs_live_source_repeated_run_gate_fb21_v1`, a
  fail-closed diagnostic gate that forces the FB-04 chained runner to use
  `rodas5p_repeated_run_source="live_source_rhs_chain"` and attaches the
  live-source chain comparison over the same tiny piecewise/window-map spans.
- **Physics added/changed:** no QKE and no public dispatch.  The gate hardens
  the repeated-run evidence surface by requiring finite live-source-chain
  `Yp`, `D/H`, `N_eff_3T`, and `Sigma_H` readouts, finite live-source-vs-
  piecewise/window-map state and BBN comparison deltas, and by default one
  supplied, applied, and provenance-fingerprinted frozen terminal collision
  payload per smoke window.
- **Evidence:** a CPU-JAX/Rodas5P CLI smoke over `N_window_edges=0,1e-10,2e-10`
  passed with `rodas5p_live_source_rhs_chain_completed_windows=2`,
  `rodas5p_live_source_rhs_chain_collision_payloads_supplied=2`,
  `rodas5p_live_source_rhs_chain_collision_payloads_applied=2`,
  `rodas5p_live_source_rhs_chain_collision_payloads_with_provenance=2`,
  `rodas5p_live_source_rhs_chain_bbn_delta_abs_max=1.195155086008981e-13`,
  `rodas5p_live_source_rhs_chain_state_vector_delta_abs_max=0.3975283023938289`,
  `rodas5p_repeated_run_source_ready=true`, and `public_dispatch_ready=false`.
- **Scope boundary:** this makes the live-source chain a landed diagnostic
  repeated-run gate, not the default replay replacement, not a full live
  collision-coupled JAX RHS, and not production-calibrated full-span BBN
  support.  AP68/SMC can now carry this gate as optional full-chain diagnostic
  metadata, but that handoff is still not public dispatch or production SMC
  validation.
- **Exit gate:** focused tests lock the builder contract, fail-closed collision
  payload requirements, and CLI dry-run claim-boundary output; the CPU smoke
  above exercises the real in-tree Rodas5P path.

### FB-22-LIVE-SOURCE-PAYLOAD-PROVENANCE  Frozen-payload provenance propagation

- **Status:** stage-recorded (historical; not current capability).
- **Depends on:** FB-21.
- **Scope:** make frozen terminal collision payload provenance a first-class
  diagnostic contract from the CPU-JAX/Rodas5P live-source RHS chain through
  AP68, AP72, AP73, AP74, AP75, AP76/AP79, and FB-20.
- **Physics added/changed:** no new collision term and no QKE.  The live-source
  chain now fingerprints the full applied `dA_modes` payload rather than only a
  scalar amplitude, and downstream gates require exact completed-window
  terminal/supplied/applied/provenance counts, terminal-source metadata, and
  unique fingerprints before accepting live-source repeated-run BBN readout
  evidence.
- **Scope boundary:** this hardens diagnostic evidence only.  It does not make
  the live-source chain public dispatch, production SMC evidence, or a
  production-calibrated full-span BBN yield.
- **Exit gate:** focused JAX replay, FB21 gate, AP68/AP72/AP73/AP74/AP75/AP76,
  FB20, registry-sync, CLI smoke, py_compile, and diff-check gates passed; ROCm
  plugin warnings from CPU-JAX initialization are non-fatal and ignored.

### FB-23-LIVE-SOURCE-EVIDENCE-CHAIN  Downstream composition witness

- **Status:** stage-recorded (historical; not current capability).
- **Depends on:** FB-22.
- **Scope:** add `augmented_live_source_repeated_run_evidence_chain_fb23_v1`, a
  deterministic orchestrator/witness that writes AP73 publication artifacts,
  AP74 plot manifest, AP75 bundle, AP79 readiness audit, and FB20 candidate-gate
  outputs under one manifest for supplied AP66/AP67/AP72/AP77/AP80 inputs.
- **Physics added/changed:** no new physics kernel.  FB-23 composes already
  fail-closed diagnostic surfaces and records the FB-21 payload-provenance
  summary so the live-source repeated-run path is checked as one artifact chain.
- **Scope boundary:** the witness keeps `promotion_decision=not_promoted`,
  `public_dispatch_ready=false`, `production_smc_validation_ready=false`, and
  `qke_scope=out_of_scope`.  It is not public production support.
- **Exit gate:** focused tests lock manifest writing, fail-closed rejection of
  missing payload fingerprints, CLI dry-run claim-boundary output, and the AP74
  renderer/AP75 bundle compatibility path for non-full-chain plot records.

### FB-24-LIVE-SOURCE-REPEATED-RUN-PROFILE  Multi-row diagnostic profile gate

- **Status:** stage-recorded (historical; not current capability).
- **Depends on:** FB-21.
- **Scope:** add `augmented_nonlrs_live_source_repeated_run_profile_fb24_v1`, a
  deterministic profile artifact/CLI that runs FB-21 live-source repeated-run
  gates over multiple tiny `N_window_edges` layouts.
- **Physics added/changed:** no new collision term and no QKE.  FB-24 broadens
  the diagnostic repeated-run evidence from one smoke layout to a profile over
  multiple chained span layouts, requiring the CPU-JAX/Rodas5P live-source RHS
  chain readout, finite BBN observables, finite live-source-vs-piecewise/window
  comparison deltas, and per-window frozen terminal collision payload
  provenance on every row.
- **Scope boundary:** this remains diagnostic profile evidence.  It does not
  make the live-source chain the default repeated-run replacement, does not
  promote public dispatch, does not provide production SMC validation, and does
  not claim production-calibrated full-span BBN support.
- **Exit gate:** focused tests lock multi-row aggregation, fail-closed promoted
  nested-row rejection, CLI dry-run claim-boundary output, py_compile, and
  registry/WBS sync.

### FB-25-LIVE-SOURCE-PROFILE-EVIDENCE-HANDOFF  Optional profile attachment to witness

- **Status:** stage-recorded (historical; not current capability).
- **Depends on:** FB-23, FB-24.
- **Scope:** extend the FB-23 downstream evidence-chain witness so it can accept
  an optional FB-24 live-source repeated-run profile artifact/path and record a
  compact passive summary beside the FB-21 repeated-run gate summary.
- **Physics added/changed:** no new collision term and no QKE.  FB-25 carries
  the already-landed multi-row CPU-JAX/Rodas5P live-source RHS profile evidence
  into the downstream manifest, validating the FB-24 contract, diagnostic claim
  scope, no-public/no-production/no-QKE boundary, all-pass row ledger, finite
  BBN/delta readouts, exact per-row frozen terminal collision-payload counts,
  terminal-source provenance, and unique payload fingerprints.
- **Scope boundary:** FB-24 remains optional for FB-23 completion.  The
  attachment is not an AP79 readiness promotion, not an FB20 production-candidate
  strengthening, not public dispatch, and not production SMC validation.
- **Exit gate:** focused tests lock optional profile acceptance, fail-closed
  rejection of promoted FB24 evidence, CLI dry-run path surfacing, py_compile,
  and registry/WBS sync.

### FB-26-LIVE-SOURCE-DYNAMIC-PAYLOAD-REFRESH  Window-boundary collision payload refresh

- **Status:** stage-recorded (historical; not current capability).
- **Depends on:** FB-04, FB-21.
- **Scope:** add an opt-in `collision_payload_refresh_mode="dynamic_restart_state"`
  to the CPU-JAX/Rodas5P live-source RHS chain so each window can build its
  collision payload from the current restart state instead of only consuming
  stale terminal-source payloads.
- **Physics added/changed:** FB-26 uses the existing AP65 combined angular plus
  `pstf_radial` no-QKE collision source evaluator at the live-source chain
  window start, serializes finite `dQ_nue_pair_N`, `dQ_nux_bank_N`, `dA_modes`,
  source diagnostics, restart/config fingerprints, and per-window provenance
  source `dynamic_restart_state`, then passes that payload into the Rodas5P
  live-source RHS.
- **Scope boundary:** this is a window-boundary payload refresh.  The refreshed
  payload is frozen inside each Rodas5P window; FB-26 does not evaluate the
  collision kernel inside every JAX RHS call, does not add QKE, does not promote
  public dispatch, and does not provide production SMC validation.
- **Exit gate:** focused tests lock single-window dynamic selection over stale
  terminal payloads, two-window dynamic payload build/apply/provenance counts,
  CLI dry-run controls, a real CPU-JAX dynamic smoke artifact, py_compile, and
  registry/WBS sync.

### FB-27-LIVE-SOURCE-DYNAMIC-SPAN-PROFILE  Increasing-span dynamic profile and plots

- **Status:** stage-recorded (historical; not current capability).
- **Depends on:** FB-26.
- **Scope:** run the opt-in CPU-JAX/Rodas5P live-source RHS chain with
  `collision_payload_refresh_mode="dynamic_restart_state"` over increasing
  smoke-to-extended `N_span_end` rows, and render generated diagnostic plots
  from the result.
- **Physics added/changed:** no new QKE or intra-step collision kernel.  FB-27
  exercises the existing FB-26 window-boundary dynamic collision-payload refresh
  over longer smoke spans, records finite BBN/shear readouts, per-window
  `dQ_nue_pair_N`, `dQ_nux_bank_N`, `dA_abs_max`, restart/config/q-grid
  provenance fingerprints, and dynamic built/applied/provenance counts.
- **Plot output:** `augmented_nonlrs_live_source_dynamic_span_profile_plots_fb27_v1`
  renders BBN, shear/solver, and payload-moment PNGs plus a SHA-256 manifest
  under generated diagnostic output paths.  Cleanup is restricted to prior FB27
  manifest-listed plot paths or exact FB27 basenames in the selected output
  directory.
- **Scope boundary:** this is diagnostic evidence only.  FB-27 does not promote
  the live-source chain into public dispatch, does not provide production SMC
  validation, and does not claim production-calibrated full-span BBN support.
- **Exit gate:** focused mock tests cover dynamic-row acceptance, fail-closed
  missing payloads, spoofed provenance rejection, JSON-safety rejection,
  public-claim rejection, failed-artifact plot rejection, PNG/manifest cleanup,
  and CLI dry-runs; real CPU-JAX profiles through
  `N_end=(1e-10,2e-10,5e-10)` and extended `N_end=(1e-9,3e-9,1e-8)` both passed
  with three rows and six dynamic payloads built/applied.

### FB-28-LIVE-SOURCE-DYNAMIC-SPAN-EVIDENCE-HANDOFF  Optional dynamic span-profile attachment

- **Status:** stage-recorded (historical; not current capability).
- **Depends on:** FB-23, FB-27.
- **Scope:** extend the FB-23 downstream evidence-chain witness so it can accept
  the FB-27 dynamic span-profile artifact and FB-27 plot manifest as an optional
  passive evidence pair.
- **Physics added/changed:** no new QKE, no new collision term, and no dispatch
  promotion.  FB-28 only carries the already-landed CPU-JAX/Rodas5P dynamic
  restart-state span-profile evidence into the downstream manifest.
- **Validation:** the witness requires both FB-27 inputs together, validates the
  dynamic-span contract, diagnostic claim scope, no-public/no-production/no-QKE
  boundary, increasing span inputs, all-pass rows, dynamic restart-state payload
  counts/provenance/fingerprints, plot source hash, exactly three PNG plot rows,
  and matched span/summary metadata.
- **Scope boundary:** FB-27 remains optional for FB-23 completion.  The attachment
  is not an AP79 readiness promotion, not an FB20 production-candidate
  strengthening, not public dispatch, and not production SMC validation.
- **Exit gate:** focused tests lock optional FB27 profile/plot acceptance,
  dangling pair rejection, promoted-profile rejection, spoofed dynamic provenance
  rejection, plot source-hash rejection, plot-count rejection, CLI dry-run path
  surfacing, py_compile, and registry/WBS sync.

### FB-29-LIVE-SOURCE-DYNAMIC-SPAN-DIAGNOSTIC-BUNDLE  Preset profile-plus-plot bundle

- **Status:** stage-recorded (historical; not current capability).
- **Depends on:** FB-27, FB-28.
- **Scope:** add a reproducible bundle runner for the FB-27 dynamic span profile
  and generated plot manifest, with named `smoke` and `extended` presets plus a
  `custom` escape hatch for exploratory diagnostic runs.
- **Physics added/changed:** no new QKE, no new collision term, and no dispatch
  promotion.  FB-29 reruns the existing FB-26/FB-27 window-boundary dynamic
  restart-state live-source chain and records the resulting profile plus plots
  under one manifest.
- **Validation:** preset builds fail closed if preset-defining span ends,
  windows-per-span, or max-step budgets drift; the bundle records profile and
  plot manifest hashes, plot count, commands, known limitations, and the
  no-public/no-production/no-QKE boundary.  FB-28 evidence-chain tests consume
  FB-29 bundle outputs through the existing FB-27 profile/plot attachment path.
- **Scope boundary:** FB-29 makes longer diagnostic runs easier to reproduce and
  inspect, but it is not public dispatch, not an FB20/AP79 readiness promotion,
  not production SMC validation, and not production-calibrated full-span BBN
  support.
- **Exit gate:** focused tests lock preset contracts, bundle writing, extended
  preset metadata, failed-profile rejection before plotting, bundle CLI dry-run
  output, FB-28 compatibility with bundle outputs, py_compile, real smoke and
  extended CPU-JAX bundle execution, and registry/WBS sync.

### FB-30-LIVE-SOURCE-DYNAMIC-LONG-PROBE  Longer diagnostic span preset

- **Status:** stage-recorded (historical; not current capability).
- **Depends on:** FB-29.
- **Scope:** add a named `diagnostic_long` preset for the FB-27/FB-29 dynamic
  span-profile bundle path, extending the CPU-JAX/Rodas5P dynamic
  restart-state live-source chain to `N_end=(3e-8,1e-7,3e-7)` with three
  windows per row and `max_steps=8000`.
- **Physics added/changed:** no new QKE, no new collision term, and no dispatch
  promotion.  FB-30 reruns the existing window-boundary dynamic collision
  payload refresh for longer diagnostic spans and records the resulting BBN,
  shear, and payload plots.
- **Validation:** FB-30 records BBN observable-bound metadata and the
  live-source chain fails closed when final readouts leave simple physical
  bounds such as `0 <= Yp <= 1`.  The CPU-JAX live-source replay also records
  requested/effective Rodas5P absolute tolerances after capping the effective
  diagnostic-path tolerance at `1e-14` to avoid trace-abundance overshoot below
  the requested scalar tolerance.
- **Scope boundary:** the long probe is diagnostic evidence only.  It is not
  public dispatch, not an FB20/AP79 readiness promotion, not production SMC
  validation, and not production-calibrated full-span BBN support.
- **Exit gate:** focused tests lock the diagnostic-long preset contract,
  bound metadata, FB30 bundle contract/stage, CLI dry-run surface, trace
  `Yp` nonnegativity at `T_gamma ~= 0.79999998 MeV`, py_compile, real CPU-JAX
  long-probe execution with zero BBN-bound warning rows, and registry/WBS sync.

### FB-31-LIVE-SOURCE-DYNAMIC-COLLISION-RADIAL-CACHE  Window-chain AP6 cache reuse

- **Status:** stage-recorded (historical; not current capability).
- **Depends on:** FB-26, FB-30.
- **Scope:** thread one chain-local AP6 radial-grid cache through the
  CPU-JAX/Rodas5P live-source RHS chain when
  `collision_payload_refresh_mode="dynamic_restart_state"`.
- **Physics added/changed:** no new QKE, no new collision term, no weak/network
  change, and no dispatch promotion.  FB-31 only reuses invariant AP6
  `pstf_radial` kernel-grid/pretabulation objects across dynamic
  restart-state payload builds in consecutive live-source windows.
- **Validation:** the live-source chain now reports
  `dynamic_collision_radial_grid_cache_enabled` and
  `dynamic_collision_radial_grid_cache_entries`, and the FB-27/FB-30 span
  profile carries those cache fields into row and summary metadata.  A focused
  two-window payload probe measured `3.889020878006704 s` without a shared
  cache versus `1.8382543009938672 s` with the shared cache (`2.1156054828236077x`)
  while preserving the closure contract, source model, and `dA_modes` shape.
- **Scope boundary:** this is CPU runtime/cache plumbing for diagnostic
  live-source chains.  It does not make dynamic collision refresh public
  dispatch, production SMC validation, or production-calibrated full-span BBN
  support.
- **Exit gate:** focused tests lock chain-local cache reuse, real dynamic
  q-grid/closure behavior, span-profile cache metadata propagation, py_compile,
  benchmark evidence, and registry/WBS sync.

### FB-32-DETERMINISTIC-COLLISION-REFERENCE-VECTORIZATION  AP41 scalar-loop removal

- **Status:** stage-recorded (historical; not current capability).
- **Depends on:** AP41, FB-31.
- **Scope:** replace nested scalar Python quadrature loops in the deterministic
  no-QKE `nu-e`, pair-annihilation, and diagonal `nu-nu` 2-to-2 references
  with NumPy broadcast contractions.
- **Physics added/changed:** no new QKE, no collision formula change, no weak
  or network change, and no dispatch promotion.  The same fixed quadrature,
  interpolation convention, six-monomial Pauli polynomial, matrix elements,
  prefactors, and moment extraction are preserved.
- **Validation:** a scalar-loop parity regression compares all three
  vectorized kernels against test-local legacy loops on distorted input
  distributions.  Existing deterministic replay-stable numerical tests still
  pass.  A smoke `N_q=5` benchmark measured `8.68x` speedup for `nu-e`
  scattering, `10.89x` for pair annihilation, and `5.53x` for diagonal
  `nu-nu` 2-to-2.  The FB31 cache-hit dynamic payload probe improved from
  `0.08017252199351788 s` to `0.04857580701354891 s`.
- **Scope boundary:** this is CPU runtime/vectorization plumbing for staged
  diagnostic live-source chains.  It does not make dynamic collision refresh
  public dispatch, production SMC validation, or production-calibrated
  full-span BBN support.
- **Exit gate:** scalar-loop parity, deterministic reference tests, focused
  augmented collision/live-source tests, py_compile, registry/WBS sync, and
  `git diff --check`.

### FB-33-BATCHED-ANGULAR-COLLISION-REFERENCE-DISPATCH  AP41 angular batch path

- **Status:** stage-recorded (historical; not current capability).
- **Depends on:** AP41, FB-32.
- **Scope:** batch the deterministic AP41 no-QKE collision-reference calls
  across angular nodes/species in the angular bridge so dynamic payload
  refresh no longer dispatches one scalar reference per angular node.
- **Physics added/changed:** no new QKE, no collision formula change, no weak
  or network change, and no dispatch promotion.  The same fixed quadrature,
  interpolation convention, six-monomial Pauli polynomial, matrix elements,
  prefactors, number/energy moment extraction, and closure projections are
  preserved.  The change is runtime/vectorization plumbing for the staged
  diagnostic AP41 bridge.
- **Validation:** batch-vs-single parity tests compare the new
  `evaluate_nue_scattering_reference_batch`,
  `evaluate_pair_annihilation_reference_batch`, and
  `evaluate_nunu_diagonal_twoto2_reference_batch` helpers against the existing
  single-reference APIs on distorted inputs.  Angular bridge tests monkeypatch
  the scalar deterministic references to fail, proving the electromagnetic and
  pairwise diagonal `nu-nu` angular paths use the batch helpers, and full
  bridge-level parity tests compare `df_dN_nodal`, `dA_modes`, component
  energy transfers, and number residuals against explicit scalar per-angle
  loops.  A `B=15`, `N_q=5` micro-benchmark measured per-call batch times of
  `0.00019041859020944686 s`, `0.00020314766035880893 s`, and
  `0.00019346193003002554 s` versus scalar-loop dispatch times of
  `0.0015221935498993843 s`, `0.0016838777303928509 s`, and
  `0.001331207490293309 s` for `nu-e`, pair, and diagonal `nu-nu`.  The same
  FB31 cache-hit dynamic payload probe improved from the FB33 pre-patch
  `0.07198696094565094 s` to `0.02897743700305 s`.
- **Scope boundary:** this is CPU runtime/batch plumbing for diagnostic
  live-source chains.  It does not make dynamic collision refresh public
  dispatch, production SMC validation, or production-calibrated full-span BBN
  support.  The post-FB33 profile now points the next optimization target at
  AP6 radial-grid/provider work rather than AP41 scalar reference dispatch.
- **Exit gate:** batch-vs-single parity, bridge-level scalar-loop parity,
  scalar-dispatch guard tests, deterministic reference tests, focused
  augmented collision/live-source tests, py_compile, registry/WBS sync, and
  `git diff --check`.

### FB-34-DYNAMIC-SOURCE-FACTORY-CACHE  AP41/AP6 source-factory pretabulation reuse

- **Status:** stage-recorded (historical; not current capability).
- **Depends on:** AP41, FB-33.
- **Scope:** cache the combined AP41 angular collision source factory and AP6
  PSTF radial source factory across dynamic restart-state payload refreshes
  when the angular/source geometry and source-refresh controls are unchanged.
- **Physics added/changed:** no new QKE, no collision formula change, no weak
  or network change, and no dispatch promotion.  The cache reuses source
  closures only for matching `N_mu`, `N_phi`, species labels, q grid/weights,
  pair-leg quadrature, gamma, neutrino temperature, radial builder controls,
  electron chemical-potential mode, scalar-QED/radial normalization controls,
  eta, radial-cache identity, and optional radial-cache directory.
- **Validation:** dynamic payload tests prove the source-factory cache fills on
  the first payload, hits on the second payload, preserves the closure
  contract/source model/`dA_modes` payload values, and keeps the cache entry
  count stable.  The live-source RHS chain test proves one chain-local
  source-factory cache object is reused across consecutive windows and that
  chain summaries report factory-cache enabled/entry metadata.  A smoke
  `N_q=5` cache-hit dynamic payload probe improved from radial-cache-only
  median `0.026258694007992744 s` to source-factory-cache median
  `0.01560014404822141 s` (`1.6832340731486084x`).
- **Scope boundary:** this is CPU runtime/source-factory cache plumbing for
  diagnostic dynamic live-source chains.  It does not make dynamic collision
  refresh public dispatch, production SMC validation, or
  production-calibrated full-span BBN support.
- **Exit gate:** focused dynamic payload cache tests, focused CPU-JAX/Rodas5P
  replay/nonlinear collision-feedback tests, py_compile, registry/WBS sync,
  internal review, and `git diff --check`.

### FB-35-DYNAMIC-SOURCE-CACHE-HIT-OVERHEAD  Dynamic payload cache-hit trim

- **Status:** stage-recorded (historical; not current capability).
- **Depends on:** AP41, FB-34.
- **Scope:** remove remaining Python-side rebuild and validation overhead from
  source-factory cache hits before pivoting back to E2E BBN path consolidation.
- **Physics added/changed:** no new QKE, no collision formula change, no weak
  or network change, and no dispatch promotion.  Source-factory cache entries
  now retain the S2 factory grid; angular source validation skips redundant
  `allclose` checks when the runtime grid is that exact object; external-q-grid
  dynamic refreshes build fixed pair-leg quadrature only on source-factory
  cache miss; AP6 radial exact checks use exact `array_equal`; and AP6 radial
  moment weights precompute the number/energy projection bases plus the 2x2
  pseudo-inverse.
- **Validation:** focused tests prove cache hits reuse the factory grid, skip
  fixed quadrature rebuild, keep actual transport-grid dimensions in the cache
  key, avoid `allclose` on exact radial diagnostics, and reuse the precomputed
  moment pseudo-inverse.  A cache-hit dynamic payload probe improved from the
  FB34 median `0.01560014404822141 s` to `0.007451459008734673 s` over 25
  repetitions with one source-factory cache entry.
- **Scope boundary:** this is CPU runtime/cache-hit overhead reduction for
  diagnostic dynamic live-source chains.  It does not make dynamic collision
  refresh public dispatch, production SMC validation, or
  production-calibrated full-span BBN support.
- **Exit gate:** focused dynamic payload cache-hit tests, focused CPU-JAX/Rodas5P
  replay/nonlinear collision-feedback tests, py_compile, registry/WBS sync,
  internal review, and `git diff --check`.

### FB-36-DYNAMIC-LIVE-SOURCE-E2E-BBN-SURFACE  Dynamic live-source E2E wiring

- **Status:** stage-recorded (historical; not current capability).
- **Depends on:** FB-12, FB-21, FB-26, and FB-35.
- **Scope:** pivot from low-level dynamic collision-payload optimization back
  into the actual E2E BBN path by exposing
  `collision_payload_refresh_mode="dynamic_restart_state"` through the FB-04
  chained full-BBN runner, the FB-21 live-source repeated-run gate, AP68
  `execution_mode="full_chain"`, and the AP69/AP70/AP71/AP72 diagnostic
  SMC schema/runtime/validation surfaces.
- **Physics added/changed:** no QKE, no collision formula change, and no
  public dispatch.  The change wires already-landed restart-state no-QKE
  payload rebuilds into higher diagnostic surfaces, disables terminal-payload
  requirements for that mode, and records dynamic payload request/build/cache
  counters in chained artifacts, FB-21 gate artifacts, and AP68 prediction
  metadata.  AP69 solver controls, AP70/AP71 result and runtime manifests, and
  AP72 physical-smoke dry-run/diagnostic metadata now preserve the same selector
  without promoting public dispatch.
- **Validation:** focused tests lock AP68 config/spec forwarding, invalid-mode
  rejection, FB21 dynamic gate behavior without terminal payload provenance,
  AP69/AP71/AP72 dry-run and manifest propagation, and CLI dry-run inputs.
  A tiny AP68 E2E smoke over `(0,1e-12,2e-12)` passed
  with the live-source RHS chain as the repeated-run BBN readout and two
  dynamic payloads built.  A larger `(0,5e-11,1e-10)` AP68 smoke passed with
  Radau/tighter source-generation tolerances; the same span with coarse RK23
  failed before dynamic refresh because the SciPy source-generation shell
  overshot deuterium below the restart handoff abundance bound.
- **Scope boundary:** this is guarded diagnostic E2E wiring and bounded smoke
  evidence only.  It does not claim production-calibrated full-span BBN support,
  public dispatch, production SMC validation, or intra-step JAX collision-kernel
  evaluation.
- **Exit gate:** focused AP68/FB21/CLI tests, two bounded AP68 CPU-JAX dynamic
  E2E smoke probes, py_compile, registry/WBS sync, internal review, and
  `git diff --check`.

### FB-37-DYNAMIC-E2E-BBN-READOUT-PLOTS  Plot-ready dynamic E2E BBN evidence

- **Status:** stage-recorded (historical; not current capability).
- **Depends on:** FB-23 and FB-36.
- **Scope:** add a deterministic diagnostic profile and plot-manifest layer
  over guarded AP68/AP72 `execution_mode="full_chain"` rows that use the
  CPU-JAX/Rodas5P live-source RHS chain as the repeated-run BBN readout with
  `dynamic_restart_state` collision-payload refresh.
- **Key files:** `src/rabbit/validation/augmented_dynamic_e2e_bbn_plots.py`,
  `src/rabbit/figures/augmented_live_source_profiles.py`,
  `scripts/plot_augmented_dynamic_e2e_bbn_readout.py`,
  `src/rabbit/validation/augmented_live_source_evidence_chain.py`,
  `scripts/run_augmented_live_source_repeated_run_evidence_chain.py`,
  `tests/test_augmented_dynamic_e2e_bbn_plots.py`, and
  `tests/test_augmented_live_source_evidence_chain.py`.
- **Physics added/changed:** no new collision formula, no QKE, no public
  dispatch, and no production SMC validation.  The PR converts existing
  dynamic AP68/AP72 E2E metadata into
  `augmented_dynamic_e2e_bbn_readout_profile_fb37_v1`, requiring finite
  `Yp`/`D/H`/`N_eff_3T`/`Sigma_H`, physical BBN readout bounds, finite
  live-source-vs-window-map deltas, completed-window dynamic payload
  request/build/applied/provenance counts, and unique dynamic payload
  fingerprints.  The renderer writes
  `augmented_dynamic_e2e_bbn_readout_plots_fb37_v1` with three PNG records and
  cleanup restricted to prior FB37 manifest-listed outputs or exact FB37
  basenames.
- **Validation:** focused tests lock AP68-row normalization, PNG/manifest
  generation, manifest cleanup, negative-`Yp` fail-closed behavior, spoofed
  dynamic-payload rejection, CLI dry-run metadata, and optional FB37 profile
  plus plot attachment to the FB-23 evidence-chain witness.
- **Scope boundary:** FB37 is diagnostic plot-ready evidence only.  It does not
  claim publication-level figures, production-calibrated full-span BBN support,
  canonical/public dispatch, production SMC validation, or QKE.
- **Exit gate:** focused FB37 plot/evidence-chain tests, py_compile,
  registry/WBS sync, internal review, and `git diff --check`.

### FB-38-DYNAMIC-E2E-BBN-READOUT-PUBLICATION-ATTACHMENT  AP75/AP79 FB37 attachment

- **Status:** stage-recorded (historical; not current capability).
- **Depends on:** FB-37.
- **Scope:** let AP75 reproducibility bundles and AP79 readiness audits consume
  a complete FB37 dynamic E2E BBN readout profile plus plot manifest as optional
  diagnostic attachment evidence, without making it an AP74 publication plot,
  promotion decision, public dispatch surface, production SMC validation, or QKE
  claim.
- **Key files:** `src/rabbit/validation/augmented_publication_bundle.py`,
  `src/rabbit/validation/augmented_publication_readiness.py`,
  `src/rabbit/validation/augmented_live_source_evidence_chain.py`,
  `tests/test_augmented_publication_bundle.py`,
  `tests/test_augmented_publication_readiness_audit.py`, and
  `tests/test_augmented_live_source_evidence_chain.py`.
- **Physics added/changed:** no new physics equations.  AP75 now accepts FB37
  evidence only when AP72 full-chain physical-smoke evidence is already present,
  revalidates the diagnostic contracts, repeated-run readout source, dynamic
  restart-state collision-payload provenance, increasing span ends, unique
  payload fingerprints, and exactly three hashed PNG plots, then copies the
  profile, manifest, and plots as diagnostic attachments.  AP79 revalidates the
  AP75 attachment and records it in `source_bundle.dynamic_e2e_bbn_readout_evidence`
  with a dedicated audit check.
- **Validation:** focused tests lock AP75 acceptance, complete-pair rejection,
  full-chain-smoke requirement, write-time artifact/plot copying, AP79
  propagation, AP79 full-chain-smoke requirement, and FB23 propagation into
  AP75/AP79.
- **Scope boundary:** FB38 keeps the evidence diagnostic and not-promoted.  It
  does not upgrade FB37 plots into publication-ready figures or full-span
  production-calibrated BBN support.
- **Exit gate:** focused AP75/AP79/FB23 tests, py_compile, registry/WBS sync,
  internal review, and `git diff --check`.

---

### FB-39-DYNAMIC-E2E-BBN-READINESS-CHAIN  Executable FB37/AP75/AP79 chain

- **Status:** stage-recorded (historical; not current capability).
- **Depends on:** FB-27, FB-37, FB-38.
- **Scope:** make the dynamic E2E BBN evidence path runnable as one diagnostic
  artifact chain instead of requiring manually composed FB37/AP75/AP79 inputs.
- **Key files:** `src/rabbit/validation/augmented_dynamic_e2e_bbn_plots.py`,
  `src/rabbit/validation/augmented_dynamic_e2e_readiness_chain.py`,
  `scripts/run_augmented_dynamic_e2e_bbn_readiness_chain.py`, and
  `tests/test_augmented_dynamic_e2e_readiness_chain.py`.
- **Physics added/changed:** no new physics equations.  FB39 adds an adapter
  from passed FB27 CPU-JAX/Rodas5P dynamic live-source span-profile rows into
  the AP68-style metadata FB37 already validates, preserving repeated-run BBN
  readouts, dynamic restart-state payload counts/fingerprints, MeV temperature
  readouts when present, and no-public/no-production/no-QKE claim boundaries.
  It also adds a chain writer that packages FB37 artifacts through AP75 and
  immediately runs AP79 with AP77 evidence.
- **Validation:** focused tests lock FB27-to-FB37 conversion, fail-closed public
  or failed FB27 inputs, AP75/AP79 chain output, AP72 full-chain-smoke
  requirement, and CLI dry-run claim-boundary metadata.
- **Scope boundary:** FB39 is diagnostic evidence orchestration.  It does not
  register public dispatch, claim production SMC validation, add QKE, or promote
  production-calibrated full-span BBN support.
- **Exit gate:** focused FB39/FB37/AP75/AP79/FB23 tests, py_compile, registry/WBS
  sync, internal review, and `git diff --check`.

---

### FB-40-DYNAMIC-E2E-BBN-SMOKE-BUNDLE  Single-command dynamic E2E smoke path

- **Status:** stage-recorded (historical; not current capability).
- **Depends on:** FB-27, FB-37, FB-39.
- **Scope:** make the dynamic E2E BBN smoke path executable as one diagnostic
  command that generates the FB27 dynamic span profile, FB37 readout profile
  and diagnostic plots, and FB39 AP75/AP79 readiness chain in order.
- **Key files:** `src/rabbit/validation/augmented_dynamic_e2e_smoke_bundle.py`,
  `scripts/run_augmented_dynamic_e2e_bbn_smoke_bundle.py`, and
  `tests/test_augmented_dynamic_e2e_smoke_bundle.py`.
- **Physics added/changed:** no new physics equations.  FB40 composes the
  existing CPU-JAX/Rodas5P dynamic restart-state span-profile runner, the FB39
  FB27-to-FB37 adapter, the FB37 readout/plot renderer, and the FB39 readiness
  chain into one smoke-scale artifact writer.  It records generated artifact
  paths/hashes, span and payload summaries, three FB37 PNG diagnostics, and
  claim-boundary checks under
  `augmented_dynamic_e2e_bbn_smoke_bundle_fb40_v1`.
- **Validation:** focused tests lock the generated FB27->FB37->FB39 sequence,
  fail-closed behavior for failed FB27 span profiles, CLI dry-run
  claim-boundary metadata, and artifact path/hash recording.
- **Scope boundary:** FB40 is diagnostic smoke orchestration only.  It does not
  register public dispatch, claim production SMC validation, add QKE, or promote
  production-calibrated full-span BBN support.
- **Exit gate:** focused FB40/FB39/FB37/FB27 tests, py_compile, registry/WBS
  sync, internal review, and `git diff --check`.

---

### FB-41-DYNAMIC-E2E-BBN-SMOKE-COMPARISON  Smoke-vs-extended FB40 comparison

- **Status:** stage-recorded (historical; not current capability).
- **Depends on:** FB-40.
- **Scope:** compare passed smoke and extended FB40 bundle manifests so larger
  CPU-JAX/Rodas5P dynamic E2E BBN span runs leave a reusable diagnostic evidence
  artifact rather than an informal terminal transcript.
- **Key files:** `src/rabbit/validation/augmented_dynamic_e2e_smoke_comparison.py`,
  `scripts/compare_augmented_dynamic_e2e_bbn_smoke_bundles.py`, and
  `tests/test_augmented_dynamic_e2e_smoke_comparison.py`.
- **Physics added/changed:** no new physics equations.  FB41 validates two FB40
  manifests, requires the extended bundle to cover a strictly larger
  `max_N_end`, and records the smoke/extended span ladders, span ratio, MeV
  `T_gamma_final` range, BBN readout ranges, dynamic restart-state payload and
  Rodas5P repeated-run readout provenance, and six FB37 diagnostic PNG paths.
- **Validation:** focused tests lock accepted smoke-vs-extended comparison,
  fail-closed rejection when the extended bundle does not increase `max_N_end`,
  and CLI dry-run claim-boundary metadata.  A manual CPU-JAX/Rodas5P run in
  `/tmp` generated smoke and extended FB40 bundles and an FB41 comparison with
  extended-to-smoke `max_N_end` ratio `20.0`, zero BBN-bound warning rows, and
  six diagnostic PNG records.
- **Scope boundary:** FB41 is diagnostic smoke-vs-extended comparison evidence.
  It does not register public dispatch, claim production SMC validation, add
  QKE, or promote publication-ready full-span BBN support.
- **Exit gate:** focused FB41/FB40 tests, actual `/tmp` smoke+extended
  comparison run, py_compile, registry/WBS sync, internal review, and
  `git diff --check`.

---

### FB-42-DYNAMIC-E2E-BBN-SPAN-SUITE  Single-command smoke+extended ladder

- **Status:** stage-recorded (historical; not current capability).
- **Depends on:** FB-40, FB-41.
- **Scope:** run the smoke and extended FB40 dynamic E2E BBN bundles under one
  command, then immediately produce the FB41 smoke-vs-extended comparison and a
  top-level manifest for reproducible diagnostic span evidence.
- **Key files:** `src/rabbit/validation/augmented_dynamic_e2e_span_suite.py`,
  `scripts/run_augmented_dynamic_e2e_bbn_span_suite.py`, and
  `tests/test_augmented_dynamic_e2e_span_suite.py`.
- **Physics added/changed:** no new physics equations.  FB42 composes existing
  FB40 and FB41 evidence writers, preserving the CPU-JAX/Rodas5P dynamic
  restart-state smoke/extended presets, MeV temperature readouts, BBN readout
  ranges, dynamic payload provenance, and generated FB37 diagnostic PNG
  inventory under `augmented_dynamic_e2e_bbn_span_suite_fb42_v1`.
- **Validation:** focused tests lock the smoke-then-extended FB40 call order,
  FB41 comparison attachment, top-level artifact/hash metadata, and CLI dry-run
  claim-boundary metadata.
- **Scope boundary:** FB42 is diagnostic span-suite orchestration only.  It
  does not register public dispatch, claim production SMC validation, add QKE,
  or promote publication-ready full-span BBN support.
- **Exit gate:** focused FB42/FB41/FB40 tests, py_compile, registry/WBS sync,
  internal review, and `git diff --check`.

---

### FB-43-DYNAMIC-E2E-BBN-FIGURE-BUNDLE  Current manifest-driven figure bundle

- **Status:** stage-recorded (historical; not current capability).
- **Depends on:** FB-37, FB-40, FB-41, FB-42.
- **Scope:** replace the legacy report/paper plotting route for the current
  dynamic E2E BBN path with a manifest-driven figure bundle that consumes an
  FB42 span-suite manifest and packages the existing FB37 PNG diagnostics.
- **Key files:** `src/rabbit/validation/augmented_dynamic_e2e_bbn_figure_bundle.py`,
  `scripts/package_augmented_dynamic_e2e_bbn_figure_bundle.py`, and
  `tests/test_augmented_dynamic_e2e_bbn_figure_bundle.py`.
- **Physics added/changed:** no new physics equations and no new renderer.
  FB43 validates FB42/FB40/FB41 contracts, passed states, artifact hashes, FB37
  plot-manifest hashes, six PNG plot hashes, zero BBN-bound warning rows, and
  retained no-public/no-production/no-QKE boundaries before copying those
  current FB37 PNGs into a clean diagnostic figure bundle.
- **Validation:** focused tests lock successful packaging, stale/open claim
  rejection, changed source-plot hash rejection, and CLI dry-run metadata.
- **Scope boundary:** FB43 is diagnostic figure inventory only.  It deliberately
  does not run the legacy report/paper plotting scripts, register public
  dispatch, claim production SMC validation, add QKE, or promote
  publication-ready full-span BBN support.
- **Exit gate:** focused FB43/FB42 tests, py_compile, registry/WBS sync,
  internal review, and `git diff --check`.

---

### FB-44-DYNAMIC-E2E-BBN-CURRENT-FIGURES  Current figure-generation pipeline

- **Status:** stage-recorded (historical; not current capability).
- **Depends on:** FB-42, FB-43.
- **Scope:** replace ad hoc current dynamic E2E BBN figure regeneration with a
  single current pipeline command that runs FB42 and then packages FB43 from a
  clean current figure directory.
- **Key files:** `src/rabbit/validation/augmented_dynamic_e2e_bbn_current_figures.py`,
  `scripts/generate_current_augmented_dynamic_e2e_bbn_figures.py`, and
  `tests/test_augmented_dynamic_e2e_bbn_current_figures.py`.
- **Physics added/changed:** no new physics equations and no legacy plotting
  renderer is called.  FB44 validates FB42 top-level and nested claim
  boundaries, deletes stale files from the selected current figure output
  directory, then validates the FB43 figure bundle contract and retained
  no-public/no-production/no-QKE boundaries.
- **Validation:** focused tests lock successful FB42->FB43 sequencing, stale
  figure cleanup, nested claim-boundary rejection, FB43 public-claim rejection,
  and CLI dry-run metadata.
- **Scope boundary:** FB44 is a diagnostic current-figure pipeline only.  It
  does not run legacy report/paper plotting scripts, register public dispatch,
  claim production SMC validation, add QKE, or promote publication-ready
  full-span BBN support.
- **Exit gate:** focused FB44/FB43/FB42 tests, py_compile, registry/WBS sync,
  internal review, and `git diff --check`.

---

### FB-45-DYNAMIC-E2E-BBN-CURRENT-FIGURE-INPUTS  FB44 input bundle resolver

- **Status:** stage-recorded (historical; not current capability).
- **Depends on:** AP75, AP79, FB-44.
- **Scope:** make existing AP75/AP79 diagnostic evidence directly consumable by
  the FB44 current figure pipeline.
- **Key files:** `src/rabbit/validation/augmented_dynamic_e2e_bbn_current_figure_inputs.py`,
  `scripts/prepare_current_augmented_dynamic_e2e_bbn_figure_inputs.py`,
  `scripts/generate_current_augmented_dynamic_e2e_bbn_figures.py`, and
  `tests/test_augmented_dynamic_e2e_bbn_current_figure_inputs.py`.
- **Physics added/changed:** no new physics equations.  FB45 validates the AP75
  bundle and AP79 readiness audit, copies verified AP66/AP67/AP72/AP74 inputs
  into a stable FB44 input directory, copies a full AP77 gate when AP79 retained
  a path, and otherwise records a missing AP77 input plus a concrete regeneration
  command or opt-in rebuilds AP77 from the AP79 summary.
- **Validation:** focused tests lock AP75/AP79 input resolution, missing-AP77
  no-false-ready behavior, opt-in AP77 rebuild plumbing, FB45 CLI dry-run
  metadata, and FB44 CLI consumption of an FB45 input bundle.
- **Scope boundary:** FB45 is diagnostic input preparation only.  It does not
  run legacy report/paper plotting scripts, register public dispatch, claim
  production SMC validation, add QKE, or promote publication-ready full-span BBN
  support.
- **Exit gate:** focused FB45/FB44 tests, py_compile, registry/WBS sync,
  internal review, and `git diff --check`.

---

### FB-46-DYNAMIC-E2E-BBN-CURRENT-FIGURE-RUN  One-shot current figure run

- **Status:** stage-recorded (historical; not current capability).
- **Depends on:** AP75, AP79, FB-45, FB-44.
- **Scope:** collapse current dynamic E2E BBN figure regeneration from AP75/AP79
  evidence into one reproducible command.
- **Key files:** `src/rabbit/validation/augmented_dynamic_e2e_bbn_current_figure_run.py`,
  `scripts/run_current_augmented_dynamic_e2e_bbn_figures.py`, and
  `tests/test_augmented_dynamic_e2e_bbn_current_figure_run.py`.
- **Physics added/changed:** no new physics equations.  FB46 runs FB45 input
  preparation, requires `fb44_ready=true`, runs FB44 current figure generation,
  and records the FB43 figure inventory, copied figures, and
  `legacy_plot_generators_used=false` in one manifest.
- **Validation:** focused tests lock successful FB45->FB44 sequencing, fail-closed
  handling of not-ready FB45 input bundles, and CLI dry-run metadata.
- **Scope boundary:** FB46 is diagnostic current figure orchestration only.  It
  does not run legacy report/paper plotting scripts, register public dispatch,
  claim production SMC validation, add QKE, or promote publication-ready
  full-span BBN support.
- **Exit gate:** focused FB46/FB45/FB44 tests, py_compile, registry/WBS sync,
  internal review, and `git diff --check`.

---

### FB-47-DYNAMIC-E2E-BBN-PUBLICATION-CURRENT-PLOTS  Paper/report-intent current plots

- **Status:** stage-recorded (historical; not current capability).
- **Depends on:** FB-46, FB-42, FB-37, paper/report figure intent.
- **Scope:** start the new plotting layer for the current augmented dynamic E2E
  BBN path by translating the paper/report figure meanings into artifact-backed
  plots without reusing the legacy report/paper plotting scripts.
- **Key files:** `src/rabbit/validation/augmented_dynamic_e2e_bbn_publication_current_plots.py`,
  `scripts/plot_current_augmented_dynamic_e2e_bbn_publication_figures.py`, and
  `tests/test_augmented_dynamic_e2e_bbn_publication_current_plots.py`.
- **Physics added/changed:** no new physics equations.  FB47 consumes an FB46
  current figure run, resolves the FB42 span-suite and FB37 profile evidence,
  and renders three current diagnostic PNGs: observable span response
  (`Y_p`, `D/H` versus `N_end`), thermo/shear span context
  (`Sigma_H`, `N_eff_3T`, `T_gamma`), and dynamic payload stability audit
  (payload build/application counts and readout deltas).  Plot records carry
  intent references such as `paper:fig:constraint`, `paper:fig:dynamics`,
  `paper:fig:ablation`, `report:fig:observable_response`,
  `report:fig:story_background`, and `report:fig:convergence`.
- **Validation:** focused tests lock dry-run intent metadata, manifest contract
  and fail-closed FB46 input handling, nonempty PNG generation, source-artifact
  provenance, axis metadata, claim boundaries, and the
  `legacy_plot_code_reused=false` / `publication_figure_ready=false` contract.
- **Scope boundary:** FB47 is diagnostic current plotting only.  It does not
  call legacy report/paper plotting scripts, register public dispatch, claim
  production SMC validation, add QKE, or promote publication-ready full-span BBN
  support.
- **Exit gate:** focused FB47/FB46/FB45/FB44 tests, py_compile,
  registry/WBS sync, internal review, and `git diff --check`.

---

### FB-48-DYNAMIC-E2E-BBN-PUBLICATION-FIGURE-RUN  One-shot publication-intent figure run

- **Status:** stage-recorded (historical; not current capability).
- **Depends on:** AP75, AP79, FB-46, FB-47.
- **Scope:** collapse the current publication-intent figure path from AP75/AP79
  evidence into one reproducible command.
- **Key files:** `src/rabbit/validation/augmented_dynamic_e2e_bbn_publication_figure_run.py`,
  `scripts/run_current_augmented_dynamic_e2e_bbn_publication_figures.py`, and
  `tests/test_augmented_dynamic_e2e_bbn_publication_figure_run.py`.
- **Physics added/changed:** no new physics equations.  FB48 runs the FB46
  AP75/AP79 -> FB45 -> FB44 current figure pipeline, then runs the FB47
  paper/report-intent current plot renderer, carrying copied-current-figure
  inventory, FB47 plot records, FB37 profile hashes, and claim-boundary
  metadata into one manifest.
- **Validation:** focused tests lock FB46->FB47 sequencing, fail-closed rejection
  of legacy FB46 plotting provenance, CLI dry-run metadata, no-public/no-production
  no-QKE boundaries, and retained `publication_figure_ready=false`.
- **Scope boundary:** FB48 is diagnostic figure orchestration only.  It does not
  call legacy report/paper plotting scripts, register public dispatch, claim
  production SMC validation, add QKE, or promote publication-ready full-span BBN
  support.
- **Exit gate:** focused FB48/FB47/FB46/FB45/FB44 tests, py_compile,
  registry/WBS sync, internal review, and `git diff --check`.

---

### FB-49-DYNAMIC-E2E-BBN-PUBLICATION-PHYSICS-FIGURES  MeV coverage and yield-sign diagnostics

- **Status:** stage-recorded (historical; not current capability).
- **Depends on:** FB-48, FB-47, FB-30.
- **Scope:** add physics-coverage figures that make the current publication-intent
  evidence more useful for paper/report review by separating what is physically
  demonstrated from what remains missing.
- **Key files:** `src/rabbit/validation/augmented_dynamic_e2e_bbn_publication_physics_figures.py`,
  `scripts/plot_current_augmented_dynamic_e2e_bbn_physics_figures.py`, and
  `tests/test_augmented_dynamic_e2e_bbn_publication_physics_figures.py`.
- **Physics added/changed:** no new physics equations.  FB49 consumes an FB48
  publication-figure run, reuses the FB48 source-profile hashes, optionally adds
  a current long-span profile, and can render excluded historical negative-`Y_p`
  probes as failure context only.  It writes two current PNGs: terminal
  `Y_p`/`D/H` sign and span stability, and `T_gamma` MeV coverage with the
  0.7--1.0 MeV freeze-out band and 0.07 MeV nucleosynthesis-track marker.
- **Validation:** focused tests lock nonempty plot generation, included negative
  `Y_p` rejection, excluded negative-probe accounting, freeze-out-window
  coverage, full-nucleosynthesis coverage remaining false, CLI dry-run metadata,
  and no-public/no-production/no-QKE boundaries.
- **Scope boundary:** FB49 is diagnostic physics-coverage plotting only.  It does
  not call legacy report/paper plotting scripts, register public dispatch, claim
  production SMC validation, add QKE, or promote publication-ready full-span BBN
  support.
- **Exit gate:** focused FB49/FB48/FB47 current-figure tests, py_compile,
  registry/WBS sync, internal review, and `git diff --check`.

---

### FB-50-LRS-NO-COLLISION-FULL-BBN-BASELINE  Raw CL0 LRS full-BBN baseline

- **Status:** stage-recorded (historical; not current capability).
- **Depends on:** canonical LRS full-BBN paths and FB49 debugging evidence.
- **Scope:** run the clean debugging baseline requested for the negative-`Y_p`
  investigation: LRS model, weak-rate corrections disabled, neutrino collision
  terms disabled, no QKE, and full BBN down below `0.01 MeV`.
- **Key files:** `src/rabbit/validation/augmented_lrs_no_collision_full_bbn.py`,
  `scripts/run_augmented_lrs_no_collision_full_bbn_baseline.py`,
  `src/rabbit/jax/augmented_typeI_replay.py`, and
  `tests/test_augmented_lrs_no_collision_full_bbn.py`.
- **Physics added/changed:** no new physics equations.  FB50 records the
  baseline where existing canonical SciPy, canonical JAX characteristic, and
  standalone extended LRS BBN paths are run with `correction_level=0`,
  `enable_collisions=false`, `Sigma_H_minus=0`, and standard phase-2 abundance
  readouts.  Canonical rows are tagged as solver-observable rows because
  `canonical_forward_solver` does not expose a terminal phase-2 abundance
  vector; standalone extended-LRS rows preserve the raw final `X_phase2`
  readout payload.  The live-source replay BBN readout convention is corrected
  to the same standard mass-fraction convention (`Y_p=X[5]`,
  `D/H=X[2]/(2 X_p)`) rather than the previous over-scaled
  `4*X[5]`/`X[2]/X_p` diagnostic convention, and the replay `D/H` readout no
  longer floors `X_p`.
- **Validation:** focused tests lock the standard readout convention, fake-solver
  artifact comparison, out-of-tolerance failure behavior, CLI dry-run metadata,
  and explicit `positivity_policy=raw_solver_abundances_no_observable_truncation`.
  A real CPU run wrote
  `diagnostic_outputs/fb50_lrs_no_collision_full_bbn_baseline/fb50_lrs_no_collision_full_bbn_baseline.json`
  and passed over `Sigma_H=(0,0.01,0.05)` with
  `T_final_MeV_min=0.004996944944314105`,
  `Y_p=0.2423494053--0.2423927149`,
  `D/H=2.4887625269e-5--2.4890647584e-5`,
  `raw_abundance_evidence_rows=3`,
  `max_abs_reference_delta_Yp=2.6441819643313602e-05`, and
  `max_abs_reference_delta_DH=1.3893467615459332e-09`.
- **Scope boundary:** FB50 is diagnostic LRS baseline evidence only.  It does not
  register public dispatch, claim production SMC validation, add QKE, enable
  neutrino collision terms, enable weak-rate corrections, or promote the
  augmented non-LRS live-source chain to publication-ready full-BBN support.
- **Exit gate:** focused FB50/readout tests, real CPU baseline artifact run,
  py_compile, registry/WBS sync, internal review, and `git diff --check`.

---

### FB-51-PROGRESSIVE-FREEDOM-FULL-BBN-LADDER  Weak/non-LRS/collision staged re-expansion

- **Status:** stage-recorded (historical; not current capability).
- **Depends on:** FB50, canonical JAX characteristic LRS/non-LRS dispatch, and
  canonical SciPy characteristic collision rows.
- **Scope:** re-enable the requested degrees of freedom in the safest order
  after the clean LRS CL0/no-collision full-BBN baseline: weak-rate
  corrections, non-LRS geometry, and LRS neutrino collision terms one at a
  time, then the supported two-freedom combinations.
- **Key files:** `src/rabbit/validation/augmented_progressive_freedom_full_bbn.py`,
  `scripts/run_augmented_progressive_freedom_full_bbn_ladder.py`,
  `src/rabbit/inference/forward_likelihood.py`, and
  `tests/test_augmented_progressive_freedom_full_bbn.py`.
- **Physics added/changed:** no new collision equation is promoted.  FB51 adds a
  fail-closed diagnostic artifact over the existing canonical surfaces:
  `jax_characteristic` for LRS weak rows, `jax_characteristic_nonlrs` for
  collisionless non-LRS rows, and SciPy characteristic tier-2 for LRS collision
  rows.  The SciPy prediction metadata now exposes `T_final` so full-BBN
  temperature reach can be checked uniformly with JAX rows.  Non-LRS+collision
  and all-three rows are deliberately run through the guarded JAX non-LRS
  request; the artifact records the guard rather than using the LRS SciPy
  collision path as a hidden substitute.
- **Validation:** focused tests lock row order, supported/guarded status counts,
  effective weak/non-LRS/collision metadata, non-positive-observable,
  terminal-temperature, and unexpected-guard-exception fail-closed behavior, and
  CLI dry-run claim flags.  A real CPU run wrote
  `diagnostic_outputs/fb51_progressive_freedom_full_bbn/fb51_progressive_freedom_full_bbn_ladder.json`
  and passed all six supported rows with
  `T_final_MeV_min=0.0049999999964358745`,
  `Y_p=0.24103996070074077--0.24854536776259234`,
  `D/H=2.4792648090988874e-5--2.5233496342152163e-5`,
  `single_toggle_supported_passed=3`, `pair_toggle_supported_passed=2`,
  `unsupported_guarded_rows=2`, all supported `Y_p>0`, and all supported
  `D/H>=0`.
- **Scope boundary:** FB51 is diagnostic progressive-freedom evidence only.  It
  does not register public dispatch, claim production SMC validation, add QKE,
  promote non-LRS collision-coupled full-BBN support, or make the all-three
  path publication-ready.
- **Exit gate:** focused FB51/FB50 tests, real CPU ladder artifact run,
  py_compile, registry/WBS sync, internal review, and `git diff --check`.

---

### FB-52-NONLRS-RESIDUAL-COLLISION-FULL-BBN  Private S2 residual full-BBN row

- **Status:** stage-recorded (historical; not current capability).
- **Depends on:** FB51 and the private non-LRS residual-state JAX surface.
- **Scope:** turn the FB51 guarded non-LRS+collision/all-three rows into real
  private diagnostic full-BBN rows without changing canonical/public dispatch.
- **Key files:** `src/rabbit/jax/driver_typeI_char.py`,
  `src/rabbit/validation/augmented_progressive_freedom_full_bbn.py`,
  `scripts/run_augmented_progressive_freedom_full_bbn_ladder.py`,
  `tests/test_pr_n2_nonlrs_driver.py`, and
  `tests/test_augmented_progressive_freedom_full_bbn.py`.
- **Physics added/changed:** `JAXNonLRSResidualFullBBNConfig` and
  `run_nonlrs_tier2_residual_full_bbn_jax` extend the existing private S2
  per-species residual collision state into a phase-split full-BBN solve.  The
  new path evolves weak freeze-out to `T_handoff=0.08 MeV`, hands off `X_n` into
  the PRIMAT phase-2 network, carries the explicit residual `R_I/R_J` state
  across the handoff, and reaches `T_end=0.005 MeV` with
  `collision_closure_mode="nonlrs_s2_residual_relaxation_v1"`.  The progressive
  ladder adds `--nonlrs-collision-mode staged_residual`, which emits the FB52
  contract while keeping the FB51 guarded mode available.
- **Validation:** focused tests lock the private full-BBN runner, public
  `jax_characteristic_nonlrs` collision guard, staged residual ladder counts,
  effective weak/non-LRS/collision metadata, and diagnostic-only claim boundary.
  A real CPU run wrote
  `diagnostic_outputs/fb52_progressive_freedom_full_bbn/fb52_progressive_freedom_full_bbn_ladder.json`
  and passed all eight rows with
  `nonlrs_collision_residual: Y_p=0.24177934397238576,
  D/H=2.48084394432579e-5`,
  `all_three_residual: Y_p=0.2479422750286894,
  D/H=2.5135990615031494e-5`,
  `T_final_MeV_min=0.004999999959857061`,
  `supported_passed_rows=8`, `unsupported_guarded_rows=0`, and
  `final_all_three_supported=true`.
- **Scope boundary:** FB52 is private diagnostic residual-state evidence only.
  It does not register public dispatch, claim production SMC validation, add
  QKE, or make the all-three path publication-ready.  Public
  `canonical_forward_solver(backend="jax_characteristic_nonlrs",
  enable_collisions=True)` remains guarded.
- **Exit gate:** focused FB52/FB51/non-LRS tests, real CPU staged residual ladder
  artifact run, py_compile, registry/WBS sync, internal review, and
  `git diff --check`.

---

### FB-53-RESIDUAL-FULL-BBN-RESOLUTION  Private residual full-BBN q/angular/relaxation ladder

- **Status:** stage-recorded (historical; not current capability).
- **Depends on:** FB52 private residual full-BBN row.
- **Scope:** turn the FB52 single private all-three residual full-BBN smoke row
  into a fail-closed q/angular/residual-relaxation resolution artifact while
  keeping canonical/public dispatch closed.
- **Key files:** `src/rabbit/validation/augmented_residual_full_bbn_resolution.py`,
  `scripts/run_augmented_residual_full_bbn_resolution_ladder.py`,
  `tests/test_augmented_residual_full_bbn_resolution.py`,
  `src/rabbit/jax/driver_typeI_char.py`, and
  `tests/test_pr_n2_nonlrs_driver.py`.
- **Physics added/changed:** the FB53 artifact calls the private
  `run_nonlrs_tier2_residual_full_bbn_jax` CPU-JAX/Rodas5P surface over
  `N_q`, `(N_theta,N_phi)`, and `residual_relax` ladders, records row-level
  `Y_p`, `D/H`, `N_eff`, `T_final_MeV`, phase event diagnostics,
  residual-state amplitudes, residual weighted-mean closure, and adjacent
  observable/residual deltas.  Duplicate physical solve points are cached so
  repeated baseline rows do not re-run Rodas5P.  The private residual full-BBN
  driver now accepts event-refinement terminal-temperature roundoff at the
  `1e-10 MeV` level; this is a terminal event classification tolerance, not an
  abundance truncation or positivity clamp.
- **Validation:** focused tests lock ladder row construction, diagnostic-only
  claim boundaries, physical sanity failures, adjacent-delta failures,
  duplicate baseline solve caching, CLI dry-run output, and terminal
  event-refinement roundoff.  A real CPU run wrote
  `diagnostic_outputs/fb53_residual_full_bbn_resolution/fb53_residual_full_bbn_resolution_ladder.json`
  and passed all six q/angular/relaxation rows with `unique_solve_points=4`,
  `prediction_cache_hits=2`, `T_final_MeV=0.004999999968892998--0.005000000053785873`,
  `max_abs_adjacent_delta_Yp=2.056706482561621e-05`,
  `max_abs_adjacent_delta_DH=1.4852111311029742e-08`,
  `max_abs_adjacent_delta_N_eff=0.00015827049378192015`,
  `max_abs_adjacent_delta_residual_state=0.009762656690099653`,
  `residual_weighted_mean_abs_max=0.0`, and
  `stage_scoped_landed_surface_ready=true`.
- **Scope boundary:** FB53 is private diagnostic residual-resolution evidence
  only.  It does not register public dispatch, claim production SMC validation,
  add QKE, or make the all-three path publication-ready.  Public
  `canonical_forward_solver(backend="jax_characteristic_nonlrs",
  enable_collisions=True)` remains guarded.
- **Exit gate:** focused FB53/non-LRS tests, real CPU residual-resolution
  artifact run, py_compile, registry/WBS sync, internal review, and
  `git diff --check`.

---

### FB-54-RESIDUAL-AP65-SAME-STATE  Private residual terminal-state source comparator

- **Status:** stage-recorded (historical; not current capability).
- **Depends on:** FB53 private residual full-BBN resolution ladder and the AP65
  combined angular+`pstf_radial` same-state source evaluator.
- **Scope:** evaluate the existing AP65 deterministic combined source on the
  terminal private residual full-BBN state, while keeping the comparison
  diagnostic and explicitly rejecting any claim that the residual S2 state is
  isomorphic to AP65's PSTF q-state.
- **Key files:** `src/rabbit/validation/augmented_residual_ap65_same_state.py`,
  `scripts/run_augmented_residual_ap65_same_state_comparator.py`,
  `tests/test_augmented_residual_ap65_same_state.py`,
  `src/rabbit/jax/driver_typeI_char.py`, and
  `tests/test_pr_n2_nonlrs_driver.py`.
- **Physics added/changed:** `run_nonlrs_tier2_residual_full_bbn_jax` now emits
  `nonlrs_residual_full_bbn_terminal_ap65_same_state_projection_v1`, a compact
  terminal payload containing the residual solve's scalar/shear/thermo/network
  state, Laguerre q grid, q-energy weights, and a diagnostic current-ray
  `{monopole,W_plus,W_minus}` projection of the residual S2 intensity state
  repeated q-flat into AP65's `A_modes` input shape.  The FB54 artifact calls
  `evaluate_augmented_nonlrs_nonlinear_combined_collision_3T_source` with that
  state, records the AP65 combined source contract, angular and `pstf_radial`
  component `dA` norms, effective source `dA` norm, source-factory and
  radial-grid cache counts, and ratios against residual-state norms where
  defined.
- **Validation:** focused tests lock terminal payload shape/boundaries, CLI
  dry-run output, AP65 `dA_modes` presence/shape/finite gates,
  production-SMC/no-QKE/no-public claim gates, and the explicit non-isomorphism
  boundary.  A real CPU run wrote
  `diagnostic_outputs/fb54_residual_ap65_same_state/fb54_residual_ap65_same_state_q4_q6.json`
  and passed both `N_q=(4,6)` rows at angular grid `(4,6)` and
  `residual_relax=1.0` with `ap65_dA_abs_max=1.989939293353786e-05`,
  `ap65_effective_dA_over_residual_state_abs_max=0.0002918186524464677`,
  `source_factory_cache_entries=2`, `radial_grid_cache_entries=36`, and
  `stage_scoped_landed_surface_ready=true`.
- **Scope boundary:** FB54 is a same-state diagnostic source probe only.  It
  does not register public dispatch, claim production SMC validation, add QKE,
  prove residual/AP65 state isomorphism, or make the all-three path
  publication-ready.
- **Exit gate:** focused FB54/non-LRS tests, real CPU same-state artifact run,
  py_compile, registry/WBS sync, internal review, and `git diff --check`.

---

### FB-55-RESIDUAL-AP65-TERMINAL-PAYLOAD  Private AP65-to-AP4 terminal-payload comparator

- **Status:** stage-recorded (historical; not current capability).
- **Depends on:** FB54 private residual/AP65 same-state source comparator and
  AP4/AP65 `piecewise_frozen` terminal-state payload exports.
- **Scope:** serialize the FB54 same-state AP65 source as an AP4 terminal-source
  payload and compare the payload contract against real AP4/AP65 terminal
  payloads over matched tiny spans.  The comparison is a schema/provenance
  bridge only: it deliberately forbids AP4-vs-residual physical-equivalence
  claims.
- **Key files:** `src/rabbit/validation/augmented_residual_ap65_same_state.py`,
  `scripts/run_augmented_residual_ap65_terminal_payload_comparator.py`,
  `tests/test_augmented_residual_ap65_same_state.py`, and
  `tests/test_augmented_stability_envelope.py`.
- **Physics added/changed:** no collision formula, weak-rate, QKE, or public
  dispatch path changes.  `ap65_same_state_source_to_terminal_source_payload`
  records the AP65 source contract, finite full `terminal_source_dA_modes`,
  source moments/diagnostics, q nodes, q-energy weights, A-mode shape, and
  same-state projection provenance in an AP4 terminal-source-compatible JSON
  payload.  The FB55 artifact compares that payload to supplied AP4 terminal
  states, requiring source-contract, q-grid, q-weight, A-shape, finite-`dA`,
  finite terminal source moments, AP4 `piecewise_frozen` source-update/subspan
  provenance, no-public, no-production, and no-QKE compatibility while
  recording only diagnostic scale ratios.
- **Validation:** focused tests lock JSON serialization, AP4 terminal-state
  compatibility, missing/nonfinite terminal-source failure behavior, q-grid and
  q-weight value/shape mismatch rejection, missing source-moment rejection,
  AP4 piecewise-provenance rejection, CLI dry-run output, empty-terminal-state rejection,
  AP4 terminal-source export fields, and the forbidden
  `physical_equivalence_claimed` gate.  A real CPU run first generated a
  matched AP4/AP65 `piecewise_frozen` terminal artifact at `N_q=4`, angular
  grid `(4,6)`, then wrote
  `diagnostic_outputs/fb55_residual_ap65_terminal_payload/fb55_residual_ap65_terminal_payload_q4.json`
  with one passing compatibility row, `physical_equivalence_claimed=false`,
  and `dA_abs_max_scale_ratio=0.014786221202355652`.
- **Scope boundary:** FB55 is a private terminal-payload compatibility
  comparator only.  It does not prove residual/AP65 state isomorphism, claim
  AP4-vs-residual physical equality, register public dispatch, claim
  production SMC validation, add QKE, or make the all-three full-BBN path
  publication-ready.
- **Exit gate:** focused FB55/stability tests, real CPU AP4 terminal artifact
  plus comparator run, py_compile, registry/WBS sync, internal review, and
  `git diff --check`.

---

### FB-56-RESIDUAL-AP65-TERMINAL-PAYLOAD-GATE  Single-command AP4 terminal-payload gate

- **Status:** stage-recorded (historical; not current capability).
- **Depends on:** FB55 terminal-payload comparator and the AP4/AP65
  `piecewise_frozen` candidate-gate terminal-state producer.
- **Scope:** collapse the manual AP4 terminal-artifact generation plus FB55
  comparator sequence into one diagnostic artifact and CLI.  The gate builds an
  AP4/AP65 `piecewise_frozen` terminal payload for each requested same-state
  comparator row, extracts the generated terminal states, runs the FB55
  comparator over those states, and requires both the AP4 artifacts and
  comparator rows to pass.
- **Key files:** `src/rabbit/validation/augmented_residual_ap65_same_state.py`,
  `scripts/run_augmented_residual_ap65_terminal_payload_gate.py`, and
  `tests/test_augmented_residual_ap65_same_state.py`.
- **Physics added/changed:** no new collision formulas, weak-rate corrections,
  public backend dispatch, production SMC path, or QKE path.  The change wires
  existing AP4/AP65 terminal payload generation to the FB55 diagnostic
  comparator with explicit no-public/no-production/no-QKE and no
  physical-equivalence boundaries.
- **Validation:** focused tests lock the gate pass path, AP4-artifact failure
  propagation, and CLI dry-run contract.  A real CPU smoke wrote
  `diagnostic_outputs/fb56_residual_ap65_terminal_payload_gate/fb56_residual_ap65_terminal_payload_gate_q4.json`
  over `N_q=4`, angular grid `(4,6)`, and `residual_relax=1.0`; it passed with
  one AP4 terminal artifact, one AP4 terminal state, one compatible comparator
  row, and `stage_scoped_landed_surface_ready=true`.
- **Scope boundary:** FB56 is a reproducibility gate for diagnostic payload
  compatibility only.  It does not prove residual/AP65 state isomorphism, claim
  AP4-vs-residual physical equality, register public dispatch, claim
  production SMC validation, add QKE, or make the all-three full-BBN path
  publication-ready.
- **Exit gate:** focused FB56/FB55 tests, real CPU gate artifact run,
  py_compile, registry/WBS sync, internal review, and `git diff --check`.

---

### FB-57-RESIDUAL-AP65-TERMINAL-PAYLOAD-EVIDENCE  Optional FB23 attachment

- **Status:** stage-recorded (historical; not current capability).
- **Depends on:** FB23 downstream evidence-chain witness and FB56
  single-command AP4 terminal-payload gate.
- **Scope:** carry the FB56 terminal-payload gate into the FB23 live-source
  repeated-run evidence-chain manifest as optional passive diagnostic evidence.
  FB57 does not make FB56 required for FB23 completion and does not alter FB20
  production-candidate semantics.
- **Key files:** `src/rabbit/validation/augmented_live_source_evidence_chain.py`,
  `scripts/run_augmented_live_source_repeated_run_evidence_chain.py`, and
  `tests/test_augmented_live_source_evidence_chain.py`.
- **Physics added/changed:** no new physics kernel, no public dispatch route,
  no production SMC path, and no QKE path.  The change validates an already
  generated FB56 artifact before recording its compact row/count/comparator
  summary in the FB23 manifest.
- **Validation:** focused tests lock acceptance of a passed FB56 artifact,
  rejection of a failed FB56 artifact, and CLI dry-run forwarding of
  `--residual-ap65-terminal-payload-gate`.
- **Scope boundary:** FB57 is passive diagnostic evidence attachment only.  It
  does not prove residual/AP65 state isomorphism, claim AP4-vs-residual
  physical equality, register public dispatch, claim production SMC validation,
  add QKE, or make the all-three full-BBN path publication-ready.
- **Exit gate:** focused FB57 evidence-chain tests, py_compile, registry/WBS
  sync, internal review, and `git diff --check`.

---

### FB-58-FULL-BBN-PHYSICS-FIGURES  Full-BBN diagnostic physics figure bundle

- **Status:** stage-recorded (historical; not current capability).
- **Depends on:** FB50 LRS no-collision full-BBN baseline, FB51/FB52
  progressive freedom ladder, FB53 residual full-BBN resolution ladder, and
  optional FB56 terminal-payload gate.
- **Scope:** render non-legacy diagnostic physics figures from current
  artifacts that actually reach the post-BBN temperature range, rather than
  from short-span dynamic E2E profile rows alone.
- **Key files:** `src/rabbit/validation/augmented_full_bbn_physics_figures.py`,
  `scripts/plot_augmented_full_bbn_physics_figures.py`, and
  `tests/test_augmented_full_bbn_physics_figures.py`.
- **Physics added/changed:** no new solver kernel and no public dispatch route.
  FB58 validates existing full-BBN diagnostic artifacts, requires positive
  included `Y_p` and non-negative D/H, requires terminal-temperature evidence
  at or below `0.01 MeV`, and renders three PNGs: progressive freedom yields,
  full-BBN temperature coverage, and residual-resolution plus terminal-payload
  provenance.  Optional FB56 data is plotted as payload compatibility
  provenance only.
- **Validation:** focused tests cover the happy path, optional missing FB56,
  negative included `Y_p` rejection, insufficient FB50 terminal-temperature
  coverage rejection, FB56 physical-equivalence-claim rejection, and CLI
  dry-run.  A real render over current FB50/FB52/FB53/FB56 diagnostic
  artifacts produced `diagnostic_outputs/fb58_full_bbn_physics_figures/`
  with three PNGs, 23 included rows, 17 terminal-temperature rows,
  `T_final_MeV=0.004996944944314105--0.005000010963688484`, positive D/H, and
  `Y_p=0.24103996070074077--0.24854536776259234`.
- **Scope boundary:** FB58 is diagnostic figure evidence only.  It does not
  register public dispatch, claim production SMC validation, add QKE, prove
  AP4-vs-residual physical equality, or make the all-freedom full-BBN path
  publication-ready.
- **Exit gate:** focused FB58 figure tests, real-artifact smoke render,
  py_compile, registry/WBS sync, internal review, and `git diff --check`.

---

### FB-59-FULL-BBN-FIGURE-EVIDENCE-CHAIN-HANDOFF  Optional FB23 attachment

- **Status:** stage-recorded (historical; not current capability).
- **Depends on:** FB23 downstream evidence-chain witness and FB58 full-BBN
  diagnostic physics figure manifest.
- **Scope:** carry the FB58 full-BBN figure manifest into the FB23 live-source
  repeated-run evidence-chain manifest as optional passive diagnostic evidence.
  FB59 does not call the FB58 renderer, does not make FB58 required for FB23
  completion, and does not alter FB20 production-candidate semantics.
- **Key files:** `src/rabbit/validation/augmented_live_source_evidence_chain.py`,
  `scripts/run_augmented_live_source_repeated_run_evidence_chain.py`, and
  `tests/test_augmented_live_source_evidence_chain.py`.
- **Physics added/changed:** no new solver kernel, no public dispatch route,
  no production SMC path, and no QKE path.  The change validates an already
  generated FB58 manifest before recording its compact plot/coverage/claim
  summary in the FB23 manifest.
- **Validation:** focused tests lock acceptance of a passed FB58 manifest,
  rejection of publication-ready claims, stale full-BBN temperature coverage,
  bad plot hashes, missing plot files, physical-equivalence claims, and CLI
  dry-run forwarding of `--full-bbn-physics-figures`.
- **Scope boundary:** FB59 is passive diagnostic evidence attachment only.  It
  does not prove AP4-vs-residual physical equality, register public dispatch,
  claim production SMC validation, add QKE, affect `chain_complete`, or make
  the all-freedom full-BBN path publication-ready.
- **Exit gate:** focused FB59 evidence-chain tests, py_compile, registry/WBS
  sync, internal review, and `git diff --check`.

---

### FB-60-FULL-BBN-DIAGNOSTIC-SUITE  Current full-BBN diagnostic evidence suite

- **Status:** stage-recorded (historical; not current capability).
- **Depends on:** FB50 LRS no-collision baseline, FB52 staged-residual
  progressive freedom ladder, FB53 residual full-BBN resolution ladder,
  optional FB56 terminal-payload gate, and FB58 physics figures.
- **Scope:** collapse the current staged full-BBN diagnostic evidence into one
  reproducible manifest and regenerate the nested FB58 full-BBN physics figures
  from the supplied artifacts.
- **Key files:** `src/rabbit/validation/augmented_full_bbn_diagnostic_suite.py`,
  `scripts/run_augmented_full_bbn_diagnostic_suite.py`, and
  `tests/test_augmented_full_bbn_diagnostic_suite.py`.
- **Physics added/changed:** no new solver kernel and no public dispatch route.
  FB60 validates already generated diagnostic full-BBN artifacts, requires the
  FB52 private staged-residual all-three row, requires FB53 residual-resolution
  readiness, rechecks terminal temperatures and abundance signs, verifies FB56
  remains compatibility provenance only, renders FB58 PNGs, and records the
  suite summary.
- **Validation:** focused tests lock the happy path, FB51 guarded-input
  rejection, stale/hot FB50 row rejection, FB56 nested physical-equivalence
  rejection, and CLI dry-run.  A real run over current FB50/FB52/FB53/FB56
  artifacts wrote `diagnostic_outputs/fb60_full_bbn_diagnostic_suite/` with
  `T_final_MeV=0.004996944944314105--0.005000010963688484`,
  `all_three_residual_Yp=0.2479422750286894`,
  `all_three_residual_DH=2.5135990615031494e-05`, zero sign violations, and
  three nested FB58 PNGs.
- **Scope boundary:** FB60 is diagnostic suite evidence only.  It does not
  register public dispatch, claim production SMC validation, add QKE, prove
  AP4-vs-residual physical equality, or make the all-freedom full-BBN path
  publication-ready.
- **Exit gate:** focused FB60 suite tests, real-artifact smoke run, py_compile,
  registry/WBS sync, internal review, and `git diff --check`.

---

### FB-61-FULL-BBN-SUITE-EVIDENCE-CHAIN-HANDOFF  Optional FB60 suite evidence-chain attachment

- **Status:** stage-recorded (historical; not current capability).
- **Depends on:** FB23 downstream evidence-chain witness and FB60 full-BBN
  diagnostic suite.
- **Scope:** make the current full-BBN diagnostic suite visible in the higher
  repeated-run/full-chain evidence surface without rerunning FB60 or changing
  readiness semantics.
- **Key files:** `src/rabbit/validation/augmented_live_source_evidence_chain.py`,
  `scripts/run_augmented_live_source_repeated_run_evidence_chain.py`, and
  `tests/test_augmented_live_source_evidence_chain.py`.
- **Physics added/changed:** no new solver kernel and no public dispatch route.
  FB61 validates a supplied FB60 suite manifest, requires its no-public,
  no-production-SMC, no-QKE, `not_promoted`, passed/empty-violation boundary,
  rechecks full-BBN terminal-temperature coverage, abundance sign safety,
  FB52 all-three support, FB53 residual-resolution readiness, optional FB56
  non-equivalence provenance, nested FB58 manifest content, nested FB58
  manifest hash, and nested FB58 PNG plot hashes, then records a compact passive
  `fb60_full_bbn_diagnostic_suite` summary in the FB23 manifest.
- **Validation:** focused tests lock accepted FB60 handoff, accepted FB60
  without optional FB56, failed-suite rejection, stale/hot terminal-temperature
  rejection, publication-claim rejection, physical-equivalence rejection,
  FB56-equivalence leakage rejection, manifest-relative nested-path resolution
  with CWD conflicts, nested FB58 publication-claim rejection, nested FB58
  missing-plot rejection, nested FB58 plot-hash rejection, nested FB58
  manifest-hash rejection, and CLI dry-run path propagation.
- **Scope boundary:** FB61 is optional passive diagnostic evidence only.  It
  does not affect `chain_complete`, rerun FB60 from FB23, register public
  dispatch, claim production SMC validation, add QKE, prove AP4/residual
  physical equality, or make the all-freedom full-BBN path publication-ready.
- **Exit gate:** focused FB23/FB58/FB60 tests, CLI dry-run over the real FB60
  path, py_compile, registry/WBS sync, internal review, and `git diff --check`.

---

### FB-62-FULL-BBN-SUITE-PUBLICATION-ATTACHMENT  Optional FB60 AP75/AP79 attachment

- **Status:** stage-recorded (historical; not current capability).
- **Depends on:** AP75 reproducibility bundle, AP79 readiness audit, and FB60
  full-BBN diagnostic suite.
- **Scope:** make the current FB60 diagnostic full-BBN suite visible in the
  higher publication-bundle/readiness surfaces without rerunning FB60 or
  changing readiness/promotion semantics.
- **Key files:** `src/rabbit/validation/augmented_full_bbn_diagnostic_suite.py`,
  `src/rabbit/validation/augmented_publication_bundle.py`,
  `src/rabbit/validation/augmented_publication_readiness.py`,
  `scripts/package_augmented_publication_bundle.py`,
  `tests/test_augmented_publication_bundle.py`, and
  `tests/test_augmented_publication_readiness_audit.py`.
- **Physics added/changed:** no new solver kernel and no public dispatch route.
  FB62 adds a shared FB60 suite summarizer, lets AP75 attach a supplied FB60
  manifest only when AP72 full-chain physical-smoke evidence is present, copies
  the FB60 manifest, nested FB58 manifest, and three FB58 PNGs, and lets AP79
  revalidate the copied evidence as `full_bbn_diagnostic_suite_evidence`.
- **Validation:** focused tests lock AP75 acceptance, full-chain-smoke
  requirement, public/production/QKE/promotion/publication/physical-equivalence
  rejection, compact-vs-full FB58 plot path/hash mismatch rejection,
  source-relative path handling with CWD conflicts, AP75 write-time attachment
  copying, AP79 propagation, missing-copy rejection, copied hash mismatch
  rejection, and missing evidence-hash rejection.
- **Scope boundary:** FB62 is optional diagnostic evidence only.  It does not
  register public dispatch, claim production SMC validation, add QKE, prove
  AP4-vs-residual physical equality, or make the all-freedom full-BBN path
  publication-ready.
- **Exit gate:** focused AP75/AP79/FB60 tests, CLI dry-run, py_compile,
  registry/WBS sync, internal review, and `git diff --check`.

---

### FB-63-FULL-BBN-FIGURE-INPUT-PROPAGATION  Optional FB60/FB58 figure attachments

- **Status:** stage-recorded (historical; not current capability).
- **Depends on:** FB45 current figure input resolver, FB46 current figure run,
  FB48 publication figure run, and FB62 AP75/AP79 FB60 attachment.
- **Scope:** make the full-BBN diagnostic figures already carried by AP75/AP79
  directly discoverable by the current figure/publication run manifests without
  rerendering FB58 or changing readiness/promotion semantics.
- **Key files:** `src/rabbit/validation/augmented_dynamic_e2e_bbn_current_figure_inputs.py`,
  `src/rabbit/validation/augmented_dynamic_e2e_bbn_current_figure_run.py`,
  `src/rabbit/validation/augmented_dynamic_e2e_bbn_publication_figure_run.py`,
  `scripts/prepare_current_augmented_dynamic_e2e_bbn_figure_inputs.py`,
  `scripts/run_current_augmented_dynamic_e2e_bbn_figures.py`,
  `scripts/run_current_augmented_dynamic_e2e_bbn_publication_figures.py`,
  and the matching FB45/FB46/FB48 tests.
- **Physics added/changed:** no new solver kernel and no public dispatch route.
  FB63 adds a passive `full_bbn_diagnostic_figure_inputs` sidecar to FB45:
  when AP75 carries `full_bbn_diagnostic_suite_evidence`, FB45 requires AP79
  `source_bundle.full_bbn_diagnostic_suite_evidence`, compares the compact
  claim/physics summary, rechecks full-BBN temperature and sign-safety fields,
  verifies no-public/no-production/no-QKE/not-promoted and
  `publication_figure_ready=false`, hash-checks the AP75 copied FB60 manifest,
  nested FB58 manifest, and three FB58 PNGs, then copies them into a stable
  FB45 input workspace outside `fb44_inputs`.  FB46 and FB48 propagate the
  sidecar summary.
- **Validation:** focused tests lock accepted AP75/AP79 FB60 sidecar copying,
  missing AP79 evidence rejection, stale copied-plot hash rejection, FB46
  propagation, FB48 propagation, and unchanged absent-sidecar behavior.
- **Scope boundary:** FB63 is passive diagnostic figure-input indexing only. It
  does not rerender FB58, call legacy plotting scripts, register public
  dispatch, claim production SMC validation, add QKE, prove AP4-vs-residual
  physical equality, or make the all-freedom full-BBN path publication-ready.
- **Exit gate:** focused FB45/FB46/FB48 tests, py_compile, registry/WBS sync,
  internal review, and `git diff --check`.

---

### FB-64-FULL-E2E-BBN-REMAINING-PLAN  Consolidated remaining-work plan

- **Status:** stage-recorded (historical; not current capability).
- **Depends on:** FB63 and the current roadmap/capability/code scan.
- **Scope:** replace scattered partial/stage-scoped notes with one explicit
  plan for the remaining full E2E BBN and publication-figure path.
- **Key files:** `docs/TYPEI_AUGMENTED_NOQKE_FULL_E2E_BBN_PLAN.md`.
- **Physics added/changed:** no new equations and no runtime dispatch change.
  FB64 separates completed diagnostic evidence, unimplemented physics blockers,
  implemented-but-not-connected surfaces, JAX-native optimization targets, and
  the FB65-FB76 PR order.
- **Validation:** internal docs/code scan plus subagent review; focused
  docs/registry tests and `git diff --check` passed.
- **Scope boundary:** FB64 is planning only.  It does not register public
  dispatch, claim production SMC validation, add QKE, prove AP4-vs-residual
  physical equality, or make the all-freedom full-BBN path publication-ready.
- **Exit gate:** plan document committed after review and focused tests.

---

### FB-65-FULL-BBN-FIGURE-INPUT-INDEX  Hash-checked full-BBN figure input index

- **Status:** stage-recorded (historical; not current capability).
- **Depends on:** FB48 publication figure run and FB63 full-BBN diagnostic
  figure-input propagation.
- **Scope:** provide one machine-readable index over the FB48-carried FB60/FB58
  full-BBN diagnostic figure inputs so the next figure pipeline can discover the
  current full-BBN manifests and PNGs without parsing AP75/AP79 directly.
- **Key files:** `src/rabbit/validation/augmented_full_bbn_figure_input_index.py`,
  `scripts/build_augmented_full_bbn_figure_input_index.py`, and
  `tests/test_augmented_full_bbn_figure_input_index.py`.
- **Physics added/changed:** no new solver kernel and no public dispatch route.
  FB65 validates an FB48 publication-figure-run artifact carrying
  `full_bbn_diagnostic_figure_inputs`, requires the no-public/no-production/no-QKE
  and not-promoted/not-publication-ready boundaries, requires full-BBN endpoint
  evidence below `0.01 MeV`, rejects negative included yields or sign-violation
  rows, hash-checks the copied FB60 manifest, nested FB58 manifest, and three
  FB58 PNGs, and writes compact role/path/hash rows under `input_index`.
- **Validation:** focused FB65 tests cover the accepted hash-checked index,
  missing endpoint evidence rejection, cwd-relative FB48/FB63 path resolution,
  stale copied-plot hash rejection, nested FB58 manifest plot-hash rejection,
  and CLI dry run.
- **Scope boundary:** FB65 is diagnostic input indexing only.  It does not
  rerender FB58, call legacy plotting scripts, register public dispatch, claim
  production SMC validation, add QKE, prove AP4-vs-residual physical equality,
  or make the all-freedom full-BBN path publication-ready.
- **Exit gate:** focused FB65 tests, current FB45/FB46/FB48 propagation tests,
  py_compile, registry/WBS sync, internal review, and `git diff --check`.

---

### FB-66-FREEDOM-LADDER-FULL-BBN-SWEEP  Completion and interaction index

- **Status:** stage-recorded (historical; not current capability).
- **Depends on:** FB51 progressive freedom ladder and FB52 private residual
  non-LRS collision full-BBN ladder.
- **Scope:** make existing freedom-ladder full-BBN rows reusable as one sweep
  artifact that future diagnostics and figure code can consume without parsing
  the FB51/FB52 row schema directly.
- **Key files:** `src/rabbit/validation/augmented_freedom_ladder_full_bbn_sweep.py`,
  `scripts/build_augmented_freedom_ladder_full_bbn_sweep.py`, and
  `tests/test_augmented_freedom_ladder_full_bbn_sweep.py`.
- **Physics added/changed:** no new solver kernel and no public dispatch route.
  FB66 consumes an FB51 or FB52 artifact, classifies every row as full-BBN
  completed, guarded-not-supported, or failed with a MeV-region label, records
  guarded non-LRS collision blockers, computes pairwise interaction residuals
  against single-freedom baseline deltas, records all-freedom readiness, and
  preserves raw progressive-ladder observables without index-level truncation.
- **Validation:** focused FB66 tests cover a passed FB52 residual sweep with
  pairwise comparisons, a real FB52 producer-output consumer path, a guarded
  FB51 non-LRS collision blocker, hot failed rows with MeV failure-region
  metadata, pass rows with nonpositive `Y_p` or negative D/H, failed-source
  all-freedom readiness blocking, failed-baseline pairwise residual blocking,
  and CLI dry run.
- **Scope boundary:** FB66 is diagnostic sweep indexing only.  It does not run a
  new solver, register public dispatch, claim production SMC validation, add
  QKE, prove AP4-vs-residual physical equality, or make the all-freedom
  full-BBN path publication-ready.
- **Exit gate:** focused FB66 tests, progressive-ladder tests, py_compile,
  registry/WBS sync, internal review, and `git diff --check`.

---

### FB-67-RESIDUAL-AP65-TRAJECTORY-CLOSURE  Checkpoint closure diagnostics

- **Status:** stage-recorded (historical; not current capability).
- **Depends on:** FB54 residual/AP65 same-state source comparator and FB66
  freedom-ladder sweep evidence.
- **Scope:** evaluate the existing FB54 residual/AP65 source probe at explicit
  decreasing temperature checkpoints so trajectory-level closure can be audited
  before attempting a continuous AP65 live-source RHS.
- **Key files:** `src/rabbit/validation/augmented_residual_ap65_trajectory_closure.py`,
  `scripts/run_augmented_residual_ap65_trajectory_closure.py`, and
  `tests/test_augmented_residual_ap65_trajectory_closure.py`.
- **Physics added/changed:** no new collision formula, weak-rate path, public
  dispatch route, production SMC path, or QKE path.  FB67 wraps the FB54
  same-state comparator over `checkpoint_T_end_values`, records MeV temperature
  windows, q-flat projection scope/model labels, AP65 source-budget agreement,
  compact residual/AP65 closure rows, and failure-kind counts for
  solver-instability, projection-contract, and source-physics failures.
- **Validation:** focused FB67 tests cover passing multi-checkpoint trajectory
  rows, AP65 source-budget failure as source-physics mismatch, residual solve
  exception as solver instability, missing q-flat projection labeling as
  projection-contract mismatch, and CLI dry run.
- **Scope boundary:** FB67 is diagnostic checkpoint evidence only.  It does not
  run a continuous AP65 live-source RHS, register public dispatch, claim
  production SMC validation, add QKE, prove residual/AP65 state isomorphism, or
  make the all-freedom full-BBN path publication-ready.
- **Exit gate:** focused FB67/FB54 tests, py_compile, registry/WBS sync,
  internal review, and `git diff --check`.

---

### FB-68-DYNAMIC-COLLISION-HOTPATH-PROFILE  AP65 payload profiler

- **Status:** stage-recorded (historical; not current capability).
- **Depends on:** FB67 residual/AP65 trajectory-closure diagnostics and the
  FB36 dynamic restart-state payload refresh path.
- **Scope:** profile the exact AP65 combined angular+`pstf_radial` collision
  payload refresh unit used by the CPU-JAX/Rodas5P live-source chain before
  landing a further JAX-native collision optimization.
- **Key files:** `src/rabbit/validation/augmented_dynamic_collision_hotpath_profile.py`,
  `scripts/profile_augmented_dynamic_collision_hotpath.py`,
  `tests/test_augmented_dynamic_collision_hotpath_profile.py`, and
  `docs/audit/fb68_dynamic_collision_hotpath_profile.md`.
- **Physics added/changed:** no new collision formula, weak-rate path, public
  dispatch route, production SMC path, or QKE path.  FB68 isolates
  `_dynamic_collision_source_payload_from_restart_state(...)`, records
  cache-disabled cold, shared-cache cold-miss, and shared-cache warm-hit timing
  rows, validates payload and AP65 closure contracts, preserves effective
  `pstf_radial` source diagnostics, records source-factory/radial-grid cache
  entries, and includes cProfile top rows for the first call in each profile case.
- **Validation:** focused FB68 tests cover successful cold/warm cache profiling,
  fail-closed missing warm cache-hit semantics, first-warm-miss rejection,
  missing effective radial-source diagnostics, requested-cProfile failure, and
  CLI dry run.  A real 4-species CPU-JAX smoke passed with shared-cache
  cold-miss median `1.6804822400445119 s`, warm-hit median
  `0.00948077195789665 s`, speedup factor `177.2516254485815`, one
  source-factory cache entry, first-warm cache hit, and 18 radial-grid cache
  entries.  Timing medians exclude cProfile instrumentation.
- **Scope boundary:** FB68 is diagnostic hot-path evidence only.  It does not
  land a speculative optimization, run collision evaluation inside the JAX RHS,
  register public dispatch, claim production SMC validation, add QKE, or make
  the all-freedom full-BBN path publication-ready.
- **Exit gate:** focused FB68 tests, the real CPU-JAX profile smoke,
  py_compile, registry/WBS sync, internal review, and `git diff --check`.

---

### FB-69-CONTINUOUS-AP65-SOURCE-RHS-PROTOTYPE  Current-state source prototype

- **Status:** stage-recorded (historical; not current capability).
- **Depends on:** FB68 dynamic collision hot-path profile and the FB36 dynamic
  restart-state payload refresh path.
- **Scope:** run a private host-stepped Rodas5P-tableau micro-window prototype
  that recomputes the AP65 combined angular+`pstf_radial` collision source from
  the current RHS/stage state instead of freezing a payload at the window
  boundary.
- **Key files:** `src/rabbit/validation/augmented_continuous_ap65_rhs.py`,
  `scripts/run_augmented_continuous_ap65_source_rhs_prototype.py`, and
  `tests/test_augmented_continuous_ap65_rhs.py`.
- **Physics added/changed:** no new collision formula, weak-rate path, public
  dispatch route, production SMC path, or QKE path.  FB69 reuses the existing
  AP65 combined source evaluator and live-source RHS block, but evaluates the
  source from current state vectors at RHS/stage points, records payload/state
  fingerprints, cache-hit diagnostics, finite-difference Jacobian policy, RHS
  deltas, BBN readouts, adjacent step-cap deltas, and frozen-window dynamic
  reference deltas.
- **Validation:** focused FB69 tests cover current-state source recomputation,
  finite step-cap RHS/BBN deltas, fail-closed nonfinite source payloads with raw
  state traces, and CLI dry run.  A real CPU-JAX smoke over `q=(0.5,1.5,3.0)`
  and `h_max=(5e-11,2.5e-11)` passed with 178 source evaluations, 17
  source-factory cache entries, 162 radial-grid cache entries, step-cap BBN
  delta abs max `1.1127229005893926e-16`, and reference BBN delta abs max
  `3.019806626980426e-14`.
- **Scope boundary:** FB69 is a private continuous-source prototype only.  The
  public jitted CPU-JAX/Rodas5P replay and chain paths remain frozen-payload or
  window-boundary-refresh diagnostics; FB69 does not register public dispatch,
  claim production SMC validation, add QKE, or make the all-freedom full-BBN
  path publication-ready.
- **Exit gate:** focused FB69/replay tests, the real CPU-JAX prototype smoke,
  py_compile, registry/WBS sync, internal review, and `git diff --check`.

---

### FB-70-CONTINUOUS-AP65-FULL-BBN-SPAN-LADDER  Physical span-expansion classifier

- **Status:** stage-recorded (historical; not current capability).
- **Depends on:** FB69 private continuous AP65 source RHS prototype and FB66
  full-BBN endpoint/failure-region conventions.
- **Scope:** run the FB69 current-state AP65 RHS prototype over increasing
  private `N_span_end` rungs, record the terminal MeV temperature per rung,
  classify `full_bbn_completed`, `completed_hot_endpoint`, and failed rows
  against the `0.01 MeV` full-BBN endpoint, preserve active freedoms, and check
  raw `Y_p`/D/H without positivity truncation.
- **Key files:** `src/rabbit/validation/augmented_continuous_ap65_full_bbn_span_ladder.py`,
  `scripts/run_augmented_continuous_ap65_full_bbn_span_ladder.py`, and
  `tests/test_augmented_continuous_ap65_full_bbn_span_ladder.py`.
- **Physics added/changed:** no new collision formula, weak-rate path, public
  dispatch route, production SMC path, or QKE path.  FB70 adds endpoint and
  failure-region accounting around the private continuous AP65 RHS prototype so
  span expansion can be measured without claiming full-BBN readiness early.
- **Validation:** focused FB70 tests cover hot-endpoint classification,
  endpoint-ready classification, fail-closed negative/unphysical BBN
  observables, and CLI dry run.  A real finite-difference CPU-JAX smoke over
  `N_span_end=(5e-11,1e-10)` passed as diagnostic execution with
  `physical_full_bbn_span_ready=false`,
  `T_gamma_final=0.7999999999214282--0.7999999999607141 MeV`, zero
  endpoint-reaching rows, 178 source evaluations, 21 stage-source evaluations,
  and two `completed_hot_endpoint` rows.  A zero-Jacobian probe over the same
  spans failed raw `Y_p`/D/H bounds and remains diagnostic failure evidence
  only.
- **Scope boundary:** FB70 is private span-expansion evidence only.  It does not
  reroute public CPU-JAX/Rodas5P dispatch, claim production SMC validation, add
  QKE, or make the all-freedom full-BBN path publication-ready.
- **Exit gate:** focused FB70/FB69 tests, finite-difference CPU-JAX smoke,
  py_compile, registry/WBS sync, internal review, and `git diff --check`.

---

### FB-71-FULL-BBN-WEAK-RATE-CONVERGENCE  Weak-rate full-BBN pair index

- **Status:** stage-recorded (historical; not current capability).
- **Depends on:** FB52 progressive freedom full-BBN ladder and optional AP80
  profile-level coupled weak-rate convergence evidence.
- **Scope:** index full-BBN weak-rate effects by pairing weak-off and weak-on
  rows in the same active-freedom context, requiring endpoint coverage and raw
  observable bounds before recording `Y_p`/D/H deltas.
- **Key files:** `src/rabbit/validation/augmented_full_bbn_weak_rate_convergence.py`,
  `scripts/build_augmented_full_bbn_weak_rate_convergence.py`, and
  `tests/test_augmented_full_bbn_weak_rate_convergence.py`.
- **Physics added/changed:** no new weak-rate formula, collision formula,
  public dispatch route, production SMC path, or QKE path.  FB71 reuses existing
  full-BBN freedom-ladder rows and optional AP80 weak-rate convergence metadata
  to separate full-BBN endpoint weak-pair evidence from tiny-span AP80 evidence.
- **Validation:** focused FB71 tests cover valid full-BBN weak-pair indexing
  with AP80 linkage, guarded all-three rows that keep the artifact diagnostic
  but not ready, fail-closed unphysical weak rows, and CLI dry run.  A real CPU
  index over `diagnostic_outputs/fb52_progressive_freedom_full_bbn/fb52_progressive_freedom_full_bbn_ladder.json`
  passed four full-BBN weak pairs with `rows_reaching_full_bbn_endpoint=8`,
  `max_abs_weak_delta_Yp=0.006193828014246616`, and
  `max_abs_weak_delta_DH=3.452364587852889e-07`; `ap80_to_full_bbn_bridge_ready=false`
  because no AP80 JSON artifact was supplied.
- **Scope boundary:** FB71 is a private diagnostic index only.  It does not run a
  new solver, register public dispatch, claim production SMC validation, add
  QKE, prove all-freedom publication readiness, or remove promotion-grade
  weak-rate convergence blockers.
- **Exit gate:** focused FB71 tests, real FB52 index build, py_compile,
  registry/WBS sync, internal review, and `git diff --check`.

---

### FB-72-AP80-FB71-FULL-BBN-WEAK-BRIDGE  AP80/full-BBN weak-rate evidence bridge

- **Status:** stage-recorded (historical; not current capability).
- **Depends on:** FB71 full-BBN weak-rate pair index and AP80 profile-level
  coupled weak-rate convergence evidence.
- **Scope:** compose AP80 profile evidence with a nested FB71 index before
  downstream figures consume weak-rate claims, while preserving the distinction
  between AP80 tiny-span q-ladder diagnostics and FB71 full-BBN endpoint
  weak/control pairs.
- **Key files:** `src/rabbit/validation/augmented_full_bbn_weak_rate_bridge.py`,
  `scripts/run_augmented_full_bbn_weak_rate_bridge.py`, and
  `tests/test_augmented_full_bbn_weak_rate_bridge.py`.
- **Physics added/changed:** no new weak-rate formula, collision formula,
  public dispatch route, production SMC path, or QKE path.  FB72 is evidence
  composition: it generates or consumes AP80, builds FB71 with AP80 supplied,
  and requires profile-count/name/q-delta consistency before bridge readiness.
- **Validation:** focused FB72 tests cover valid AP80/FB71 bridge composition,
  stale FB71-without-AP80 rejection, invalid AP80 profile evidence rejection,
  nested artifact writing with a supplied AP80 artifact, and CLI dry run.  A
  real CPU smoke over
  `diagnostic_outputs/fb52_progressive_freedom_full_bbn/fb52_progressive_freedom_full_bbn_ladder.json`
  passed with `ap80_profile_count=1`, `ap80_total_nfev=7596`,
  `ap80_applied_rate_q_relative_delta_abs_max=0.0024445680701901517`,
  `fb71_passed_pair_count=4`, `fb71_rows_reaching_full_bbn_endpoint=8`, and
  `ap80_fb71_bridge_ready=true`.
- **Scope boundary:** FB72 is a private diagnostic bridge only.  AP80 remains
  profile/tiny-span evidence, not a full-BBN weak-rate convergence proof.  FB72
  does not register public dispatch, claim production SMC validation, add QKE,
  prove all-freedom publication readiness, or remove promotion-grade weak-rate
  blockers.
- **Exit gate:** focused FB72/FB71/AP80 tests, real CPU bridge smoke,
  py_compile, registry/WBS sync, internal review, and `git diff --check`.

---

### FB-73-PUBLICATION-FIGURE-RENDERER-V2  Current-artifact figure renderer

- **Status:** stage-recorded (historical; not current capability).
- **Depends on:** FB60 full-BBN diagnostic suite, FB66 freedom-ladder
  full-BBN sweep, FB70 continuous-AP65 full-BBN span ladder, and FB72
  AP80-FB71 full-BBN weak-rate bridge.
- **Scope:** replace the legacy-based plotting path for the current full E2E
  BBN line with a new renderer that consumes current artifacts directly,
  validates claim boundaries, and writes a hashed manifest plus diagnostic
  plot metadata.
- **Key files:** `src/rabbit/validation/augmented_publication_figure_renderer_v2.py`,
  `scripts/render_augmented_publication_figures_v2.py`,
  `tests/test_augmented_publication_figure_renderer_v2.py`, and
  `docs/audit/fb73_publication_figure_renderer_v2.md`.
- **Physics added/changed:** no new collision formula, weak-rate formula,
  solver kernel, public dispatch route, production SMC path, or QKE path.
  FB73 validates existing FB60/FB66/FB70/FB72 artifacts and renders endpoint
  coverage, freedom-ladder terminal yields, weak-rate bridge deltas, and the
  continuous-AP65 span boundary from those artifacts.
- **Validation:** focused FB73 tests cover valid current-artifact rendering,
  exact manifest/source/PNG hash checks, claim-boundary leak rejection,
  incomplete FB66 sweep rejection, inconsistent FB72 bridge rejection,
  nonpositive log-axis quantity rejection, in-memory source hashing, and CLI
  dry run.  A real render over the current ignored diagnostic artifacts passed
  with artifact payload SHA256
  `6a62a214b0d13e38e5d76bac652c8ce02caf4ec93880157b49908a75a567ceae`,
  `plot_count=4`, `full_bbn_T_final_MeV=0.004996944944314105--0.005000010963688484`,
  `freedom_sweep_completed_rows=8`, `weak_rate_bridge_passed_pair_count=4`,
  and `publication_readiness_blocker=continuous_ap65_full_bbn_span_not_ready`.
- **Scope boundary:** FB73 is diagnostic current-artifact figure evidence only.
  It does not reuse legacy plotting code, run a new solver, register public
  dispatch, claim production SMC validation, add QKE, prove all-freedom
  publication readiness, or remove the continuous AP65 full-BBN span blocker.
- **Exit gate:** focused FB73/FB72 tests, real current-artifact render,
  py_compile, registry/WBS sync, internal review, and `git diff --check`.

---

### FB-74-PUBLICATION-FIGURE-BUNDLE-QA  Current-artifact figure QA bundle

- **Status:** stage-recorded (historical; not current capability).
- **Depends on:** FB73 current-artifact publication figure renderer V2.
- **Scope:** validate and package the FB73 diagnostic figure manifest as a
  reproducible QA bundle without rerendering figures or calling legacy plotting
  scripts.
- **Key files:** `src/rabbit/validation/augmented_publication_figure_bundle_qa.py`,
  `scripts/package_augmented_publication_figure_bundle_qa.py`,
  `tests/test_augmented_publication_figure_bundle_qa.py`, and
  `docs/audit/fb74_publication_figure_bundle_qa.md`.
- **Physics added/changed:** no new collision formula, weak-rate formula,
  solver kernel, public dispatch route, production SMC path, or QKE path.
  FB74 recomputes the FB73 payload hash, verifies FB73/source/plot/copy
  hashes, checks diagnostic captions and claim labels, and records a QA
  manifest with copied PNG provenance.
- **Validation:** focused FB74 tests cover accepted FB73 bundle packaging,
  stale FB73 payload-hash rejection, stale source-artifact hash rejection,
  stale plot hash rejection, publication-ready claim rejection, caption
  overclaim rejection, duplicate plot basename rejection, and CLI dry run.  A
  real QA run passed with stable rerun artifact payload SHA256
  `e22a8f6ea68b24e376b1ded12b6bb531199005bead8b4b7ef6d187f76f645e45`,
  manifest file SHA256
  `d609ba75756bbb9be0c7dd1fa256b6ad167eda1974a005527f8a08354c664cd5`,
  source FB73 payload SHA256
  `6a62a214b0d13e38e5d76bac652c8ce02caf4ec93880157b49908a75a567ceae`,
  `plot_count=4`, `copied_plot_count=4`, and `qa_checks=10`.
- **Scope boundary:** FB74 is diagnostic QA evidence only.  It does not render
  new plots, reuse legacy plotting code, run a solver, register public
  dispatch, claim production SMC validation, add QKE, prove all-freedom
  publication readiness, or remove the continuous AP65 full-BBN span blocker.
- **Exit gate:** focused FB74/FB73 tests, real current-artifact QA run,
  py_compile, registry/WBS sync, internal review, and `git diff --check`.

---

### FB-75-GUARDED-SMC-PILOT-GATE  Guarded diagnostic SMC pilot gate

- **Status:** stage-recorded (historical; not current capability).
- **Depends on:** AP72 full-chain physical-smoke validation, FB60 full-BBN
  diagnostic suite, FB66 freedom-ladder sweep, FB70 continuous-AP65 span
  ladder, FB72 AP80-to-FB71 weak-rate bridge, and FB74 figure bundle QA.
- **Scope:** validate that the current full-BBN diagnostic products are usable
  as guarded statistical-pilot inputs while preserving fail-closed public,
  production-SMC, QKE, and dispatch boundaries.
- **Key files:** `src/rabbit/validation/augmented_guarded_smc_pilot_gate.py`,
  `scripts/build_augmented_guarded_smc_pilot_gate.py`,
  `tests/test_augmented_guarded_smc_pilot_gate.py`, and
  `docs/audit/fb75_guarded_smc_pilot_gate.md`.
- **Physics added/changed:** no new solver kernel, sampler run, collision
  formula, weak-rate formula, public dispatch route, production SMC path, or
  QKE path.  FB75 recomputes embedded source payload hashes where available,
  verifies AP72/FB60/FB66/FB70/FB72/FB74 contracts and claim boundaries,
  requires file-backed source hashes, preserves the FB70 continuous-AP65
  full-span blocker, and records an AP69 SMC schema snapshot for a diagnostic
  pilot input handoff.
- **Validation:** focused FB75 tests cover accepted current diagnostic
  products, AP72 physical-smoke missing fail-closed behavior, FB66
  full-BBN-readiness rejection, inconsistent FB70 physical-span claims, stale
  hashed source rejection, hashed writer output, and CLI dry run.  A real
  current-artifact gate run passed with
  `artifact_payload_sha256=18841d947067979eb5cdfddeef1a4c55656fbc62e92257a6e63197820bfea352`,
  manifest file SHA256
  `6087af94215ff25628c18e7a5fa3fd9a22ae2981166ec0ae467d6c036e661922`,
  `guarded_smc_pilot_input_ready=true`,
  `validated_full_bbn_product_inputs_ready=true`,
  `statistical_pilot_input_ready=true`, `source_hashes_checked=true`,
  `runs_new_smc_sampler=false`, `fb66_completed_rows=8`,
  `fb72_rows_reaching_full_bbn_endpoint=8`, and
  `pilot_blockers=[continuous_ap65_full_bbn_span_not_ready]`.
- **Scope boundary:** FB75 does not run SMC, register candidate or public
  dispatch, claim sampler readiness, claim production SMC validation, add QKE,
  prove all-freedom publication readiness, or remove the continuous AP65
  full-BBN span blocker.
- **Exit gate:** focused FB75 tests, real current-artifact gate run,
  py_compile, registry/WBS sync, internal review, and `git diff --check`.

---

### FB-76-INTERNAL-CANDIDATE-DISPATCH-DECISION  Fail-closed internal dispatch decision

- **Status:** stage-recorded (historical; not current capability).
- **Depends on:** FB75 guarded SMC pilot-input gate, AP68 guarded candidate
  forward-model entrypoints, AP69 schema metadata, and the backend capability
  registry.
- **Scope:** decide whether an internal candidate-dispatch surface is warranted
  without registering a backend alias or changing canonical/public dispatch.
- **Key files:** `src/rabbit/validation/augmented_internal_candidate_dispatch_decision.py`,
  `scripts/build_augmented_internal_candidate_dispatch_decision.py`,
  `tests/test_augmented_internal_candidate_dispatch_decision.py`, and
  `docs/audit/fb76_internal_candidate_dispatch_decision.md`.
- **Physics added/changed:** no new solver kernel, sampler run, collision
  formula, weak-rate formula, public dispatch route, production SMC path, or
  QKE path.  FB76 hash-checks the FB75 input and every FB75 nested source file
  SHA, verifies AP68 callable symbols without executing solves, checks that
  the augmented staging capability
  remains absent from `CAPABILITY_BY_BACKEND`, and records the decision plus
  blockers.
- **Validation:** focused FB76 tests cover the current blocker-driven defer
  decision, the blocker-cleared warrant-but-not-register branch, sampler
  readiness overclaim rejection, mapping-only source rejection, unexpected
  backend-dispatch registration rejection, hashed writer output, and CLI dry
  run.  A real current-artifact decision run passed with
  `artifact_payload_sha256=d361d9dab63dde7b54fed1656b6d7f61f5a10ec2ffdb2e6467dbb4d3f3b09518`,
  manifest file SHA256
  `2e871e1ec920371779bb22570e69ec4925369e0ef51d3a4f005cbb466604eac3`,
  `internal_candidate_dispatch_decision=defer`,
  `internal_candidate_dispatch_warranted=false`, `registers_dispatch=false`,
  `canonical_forward_solver_registered=false`, and
  `decision_blockers=[continuous_ap65_full_bbn_span_not_ready]`.
- **Scope boundary:** FB76 does not add a backend alias, alter
  `canonical_forward_solver`, run SMC, register candidate or public dispatch,
  claim sampler readiness, claim production SMC validation, add QKE, prove
  all-freedom publication readiness, or remove the continuous AP65 full-BBN
  span blocker.
- **Exit gate:** focused FB76 tests, real current-artifact decision run,
  py_compile, registry/WBS sync, internal review, and `git diff --check`.

---

### FB-77-CLAIM-READINESS-REVIEW  Diagnostic claim ledger

- **Status:** stage-recorded (historical; not current capability).
- **Depends on:** FB76 internal candidate-dispatch decision and the roadmap
  claim-boundary documents.
- **Scope:** state the strongest defensible current claim and the remaining
  blockers without running a solver, running SMC, registering dispatch, or
  changing public/canonical support.
- **Key files:** `src/rabbit/validation/augmented_claim_readiness_review.py`,
  `scripts/build_augmented_claim_readiness_review.py`,
  `tests/test_augmented_claim_readiness_review.py`, and
  `docs/audit/fb77_claim_readiness_review.md`.
- **Physics added/changed:** no new solver kernel, sampler run, collision
  formula, weak-rate formula, public dispatch route, production SMC path, or
  QKE path.  FB77 hash-checks FB76, requires FB76 nested source-hash
  verification, hashes the current roadmap docs with FB77 self-reference
  artifact-hash lines redacted, and records a bounded diagnostic claim ledger.
- **Validation:** focused FB77 tests cover the accepted current claim-review
  ledger, mapping-only FB76 rejection, stale FB76 payload-hash rejection,
  public-dispatch leak rejection, missing FB76 nested source-hash verification
  rejection, incomplete FB76 nested source-hash ledger rejection, FB76
  nested source expected/actual SHA mismatch rejection, FB77 self-hash-only
  doc redaction, hashed writer output, and CLI dry run.  A real current-artifact
  review passed with
  `artifact_payload_sha256=e38abd1c8de1b7f61755fff396c9465a3679ba0f657676fa2b934b931da06f95`,
  manifest file SHA256
  `13c83abdb5fa0f66f61f2158f5f4e50b9c89d1dbf9a178ac8c41ad2babdddd13`,
  `claim_readiness_level=diagnostic_evidence_chain_ready`,
  `strongest_defensible_claim_key=guarded_internal_diagnostic_evidence_chain`,
  `public_dispatch_ready=false`, `production_smc_validation_ready=false`,
  `publication_ready_all_freedom_full_bbn=false`, `qke_scope=out_of_scope`,
  `registers_dispatch=false`, and
  `recommended_next_physics_pr=extend_continuous_ap65_full_bbn_span_to_0p01_MeV`.
- **Scope boundary:** FB77 does not add a backend alias, alter
  `canonical_forward_solver`, run SMC, run a solver, register candidate or
  public dispatch, claim sampler readiness, claim production SMC validation,
  add QKE, prove all-freedom publication readiness, or remove the continuous
  AP65 full-BBN span blocker.
- **Exit gate:** focused FB77/FB76 tests, real current-artifact review run,
  py_compile, registry/WBS sync, internal review, and `git diff --check`.

---

### FB-78-CONTINUOUS-AP65-CHAINED-SPAN-LADDER  Consecutive-window restart handoff

- **Status:** stage-recorded (historical; not current capability).
- **Depends on:** FB69 private continuous AP65 source RHS prototype and FB70
  continuous-AP65 full-BBN span ladder.
- **Scope:** let the private continuous-AP65 span ladder run as consecutive
  windows by feeding each passing FB69 terminal restart state into the next
  FB69 window.
- **Key files:** `src/rabbit/validation/augmented_continuous_ap65_rhs.py`,
  `src/rabbit/validation/augmented_continuous_ap65_full_bbn_span_ladder.py`,
  `scripts/run_augmented_continuous_ap65_full_bbn_span_ladder.py`,
  `tests/test_augmented_continuous_ap65_rhs.py`,
  `tests/test_augmented_continuous_ap65_full_bbn_span_ladder.py`, and
  `docs/audit/fb78_continuous_ap65_chained_span_ladder.md`.
- **Physics added/changed:** no new collision formula, weak-rate formula,
  solver kernel, public dispatch route, production SMC path, or QKE path.
  FB78 adds state handoff around the existing private FB69 current-state AP65
  RHS prototype.  FB69 records whether initial conditions or supplied restart
  kwargs seeded the row, fingerprints supplied restart inputs, and emits
  terminal restart kwargs from the final state.  FB70 can then use those restart
  kwargs to run each span rung as `(previous_end,current_end)` and refuses to
  propagate restart payloads from failed or unphysical rows.
- **Validation:** focused tests cover supplied FB69 restart kwargs, terminal
  restart emission, FB70 chained consecutive spans, previous-window restart
  propagation, and fail-closed halt when a failed window still carries a restart
  payload.  A real finite-difference CPU-JAX smoke over
  `N_span_end=(5e-11,1e-10,2e-10,5e-10)` passed four chained windows with
  `artifact_payload_sha256=463418cba619ef8199b642debcd3425f54a3fd21f24b62038a85ecba5f1e46b9`,
  manifest file SHA256
  `f3c22071c252c990041aea33471db0b52ecb2da59cddf59872300ba84bdc36fa`,
  `restart_handoff_ready_rows=4`, `source_evaluations_total=588`,
  `step_count_total=10`, and
  `T_gamma_final=0.799999999607141--0.7999999999607141 MeV`.
- **Scope boundary:** FB78 is private span-handoff evidence only.  It remains a
  hot-endpoint diagnostic with `physical_full_bbn_span_ready=false` and
  `rows_reaching_endpoint=0`; it does not reroute public CPU-JAX/Rodas5P
  dispatch, claim production SMC validation, add QKE, or make the all-freedom
  full-BBN path publication-ready.
- **Exit gate:** focused FB78/FB70/FB69 tests, real chained-span CPU-JAX smoke,
  py_compile, registry/WBS sync, internal review, and `git diff --check`.

---

### FB-79-CONTINUOUS-AP65-SPAN-BRACKET  Chained-span stability bracket

- **Status:** stage-recorded (historical; not current capability).
- **Depends on:** FB78 chained AP65 span handoff and FB70 span/failure-region
  classification.
- **Scope:** run multiple private chained FB70 span profiles under the same
  physics and solver controls, then record the last passing profile and first
  observed failing endpoint.
- **Key files:** `src/rabbit/validation/augmented_continuous_ap65_span_bracket.py`,
  `scripts/run_augmented_continuous_ap65_span_bracket.py`,
  `tests/test_augmented_continuous_ap65_span_bracket.py`, and
  `docs/audit/fb79_continuous_ap65_span_bracket.md`.
- **Physics added/changed:** no new collision formula, weak-rate formula,
  solver kernel, public dispatch route, production SMC path, or QKE path.
  FB79 is diagnostic orchestration around FB70: it requires nested FB70
  no-public/no-production/no-QKE boundaries, preserves nested failure regions,
  and reports a stable span bracket for the current continuous-AP65 chained
  surface.
- **Validation:** focused tests cover pass/fail bracket extraction,
  no-passing-profile fail-closed behavior, nested public-dispatch leak
  rejection, and CLI dry run.  A real finite-difference CPU-JAX bracket passed
  with `artifact_payload_sha256=e1c73bdae84d013a3ac0551bff404716f78bf3fdcd37a337326f3f5740e8df35`,
  manifest file SHA256
  `2cb311b2779809db259d0574b26c1def21057ebbe152ddb97711fdf82d260557`,
  `bracket_status=pass_fail_bracketed`, `largest_passing_N_span_end=5e-10`,
  `first_failing_N_span_end=1e-09`,
  `best_passing_T_final_MeV=0.799999999607141`, and
  `first_failing_T_final_MeV=0.7999999992142808`.
- **Scope boundary:** FB79 is private bracket evidence only.  The first failing
  profile remains above the full-BBN endpoint and fails with raw nonpositive
  `Y_p`; FB79 does not reroute public CPU-JAX/Rodas5P dispatch, claim
  production SMC validation, add QKE, or make the all-freedom full-BBN path
  publication-ready.
- **Exit gate:** focused FB79 tests, real chained-span bracket run,
  py_compile, registry/WBS sync, internal review, and `git diff --check`.

---

### FB-80-CONTINUOUS-AP65-HMAX-SENSITIVITY  Step-size recovery diagnostic

- **Status:** stage-recorded (historical; not current capability).
- **Depends on:** FB79 span bracket and FB70 span/failure-region
  classification.
- **Scope:** hold the private continuous-AP65 FB70 target span fixed at
  `N_span_end=1e-9`, sweep a strictly decreasing `h_max` ladder, and classify
  whether the FB79 first failure is recovered by smaller internal steps.
- **Key files:** `src/rabbit/validation/augmented_continuous_ap65_hmax_sensitivity.py`,
  `scripts/run_augmented_continuous_ap65_hmax_sensitivity.py`,
  `tests/test_augmented_continuous_ap65_hmax_sensitivity.py`, and
  `docs/audit/fb80_continuous_ap65_hmax_sensitivity.md`.
- **Physics added/changed:** no new collision formula, weak-rate formula,
  solver kernel, public dispatch route, production SMC path, or QKE path.
  FB80 is diagnostic orchestration around FB70: it keeps the physics target
  fixed, varies only `h_max`, checks nested no-public/no-production/no-QKE
  boundaries, and distinguishes recoverable coarse-step observable failure
  from unrecovered or unexpected nested failures.
- **Validation:** focused tests cover h_max refinement recovery, nested
  claim-boundary leak rejection, unrecovered-failure fail-closed behavior, and
  CLI dry run.  A real finite-difference CPU-JAX diagnostic passed with
  `artifact_payload_sha256=84d6aac41fc673889320ebc5802fa78049977917da193ad9154fc487048558e4`,
  manifest file SHA256
  `93a3dc65aa573f0217063fc947084caa97f6c5409e6d592cf8a0d8005a927912`,
  `classification=h_max_refinement_recovers_observable_failure`,
  `largest_failing_h_max=1e-09`,
  `first_passing_h_max_after_failure=5e-10`,
  `smallest_passing_h_max=2.5e-10`, `rows_failed=1`, and `rows_passed=2`.
- **Scope boundary:** FB80 is private sensitivity evidence only.  The target
  remains far above the `0.01 MeV` endpoint and does not prove full-BBN
  completion, public dispatch readiness, production SMC validation, QKE
  support, or publication-ready all-freedom support.
- **Exit gate:** focused FB80 tests, real h_max sensitivity run, py_compile,
  registry/WBS sync, internal review, and `git diff --check`.

---

### FB-81-CONTINUOUS-AP65-REFINED-SPAN-BRACKET  Refined-step span bracket

- **Status:** stage-recorded (historical; not current capability).
- **Depends on:** FB80 h_max sensitivity and FB70 span/failure-region
  classification.
- **Scope:** run one private chained FB70 span ladder with the refined
  `h_max=2.5e-10` policy and record the largest passing endpoint plus the
  first refined-hmax failing endpoint.
- **Key files:** `src/rabbit/validation/augmented_continuous_ap65_refined_span_bracket.py`,
  `scripts/run_augmented_continuous_ap65_refined_span_bracket.py`,
  `tests/test_augmented_continuous_ap65_refined_span_bracket.py`, and
  `docs/audit/fb81_continuous_ap65_refined_span_bracket.md`.
- **Physics added/changed:** no new collision formula, weak-rate formula,
  solver kernel, public dispatch route, production SMC path, or QKE path.
  FB81 is diagnostic orchestration around FB70: it applies the FB80 refined
  step size to the span ladder, checks nested no-public/no-production/no-QKE
  boundaries, and preserves first-failure evidence.
- **Validation:** focused tests cover refined pass/fail bracket extraction,
  nested claim-boundary leak rejection, no-passing-prefix fail-closed behavior,
  unexpected first-failure fail-closed behavior, and CLI dry run.  A real
  finite-difference CPU-JAX diagnostic passed with
  `artifact_payload_sha256=76bf833035a9a23f7b444786d19924d7d676d23d2f79c086703faeb0ae3f212e`,
  manifest file SHA256
  `243e1170947e2ce33271c0410169be55f73fa5383de5ac305bb9005e051ab2f9`,
  `classification=refined_span_pass_fail_bracketed`,
  `largest_passing_N_span_end=1e-09`,
  `first_failing_N_span_end=1.5e-09`, `rows_passed=2`, and `rows_failed=2`.
- **Scope boundary:** FB81 is private refined-span bracket evidence only.  The
  target remains far above the `0.01 MeV` endpoint and does not prove full-BBN
  completion, public dispatch readiness, production SMC validation, QKE
  support, or publication-ready all-freedom support.
- **Exit gate:** focused FB81 tests, real refined-span bracket run, py_compile,
  registry/WBS sync, internal review, and `git diff --check`.

---

### FB-82-CONTINUOUS-AP65-FIRST-FAILURE-TRIAGE  Strict-Yp failure triage

- **Status:** stage-recorded (historical; not current capability).
- **Depends on:** FB81 refined-span bracket.
- **Scope:** rerun the private FB81 chained span bracket, extract the first
  failing row, and record strict `Y_p` positivity, abundance-bound tolerance,
  BBN observables, restart handoff, and source-evaluation counts as separate
  diagnostic evidence.
- **Key files:** `src/rabbit/validation/augmented_continuous_ap65_failure_triage.py`,
  `scripts/run_augmented_continuous_ap65_failure_triage.py`,
  `tests/test_augmented_continuous_ap65_failure_triage.py`, and
  `docs/audit/fb82_continuous_ap65_failure_triage.md`.
- **Physics added/changed:** no new collision formula, weak-rate formula,
  solver kernel, public dispatch route, production SMC path, or QKE path.
  FB82 is a diagnostic layer around FB81: it makes the strict-sign failure
  explicit and prevents abundance-bound tolerance from being treated as a
  sign repair.
- **Validation:** focused tests cover strict-`Y_p` failure inside abundance
  tolerance, strict-`Y_p` failure outside abundance tolerance, nested FB81
  boundary leak rejection, missing first-failure fail-closed behavior, missing
  first-failure BBN observables, and CLI dry run.  A real finite-difference
  CPU-JAX diagnostic passed with
  `artifact_payload_sha256=c64bf7175a6935b39859ae521a05fadd6548fcfa5e2326d3faff9da1e9f9a783`,
  manifest file SHA256
  `6eed5354131696519b92f3e7ba4c2132cf5f37f9b88e4911ebcabb7012649b0b`,
  `classification=strict_y_p_sign_failure_within_abundance_tolerance`,
  `first_failing_N_span_end=1.5e-09`,
  `Yp=-1.2294890184644955e-30`,
  `abundance_bound_tolerance=1e-18`,
  `abundance_bounds_ok=true`, and
  `bound_tolerance_masks_strict_sign=true`.
- **Scope boundary:** FB82 is private first-failure triage evidence only.  It
  does not relax strict positivity, truncate or repair abundances, prove
  full-BBN completion, open public dispatch, claim production SMC validation,
  add QKE, or make the all-freedom path publication-ready.
- **Exit gate:** focused FB82 tests, real failure-triage run, py_compile,
  registry/WBS sync, internal review, and `git diff --check`.

---

### FB-83-CONTINUOUS-AP65-YP-SOURCE-PROBE  Packed-state Yp source probe

- **Status:** stage-recorded (historical; not current capability).
- **Depends on:** FB82 strict-`Y_p` failure triage.
- **Scope:** compare the first failing terminal BBN `Yp` against `He4` in the
  packed FB69 `last_attempted_state_vector` tail, using the live-source replay
  `X_phase2_shape=(9,)` contract.
- **Key files:** `src/rabbit/validation/augmented_continuous_ap65_y_p_source_probe.py`,
  `scripts/run_augmented_continuous_ap65_y_p_source_probe.py`,
  `tests/test_augmented_continuous_ap65_y_p_source_probe.py`, and
  `docs/audit/fb83_continuous_ap65_y_p_source_probe.md`.
- **Physics added/changed:** no new collision formula, weak-rate formula,
  solver kernel, public dispatch route, production SMC path, or QKE path.
  FB83 is a diagnostic localization layer: it makes the terminal sign crossing
  visible without changing solver states or repairing abundances.
- **Validation:** focused tests cover sub-tolerance terminal sign crossing
  after positive last-stage `He4`, last-stage `He4` already nonpositive,
  nested FB82 boundary leak rejection, missing first-failure rows, missing
  state vectors, and CLI dry run.  A real finite-difference CPU-JAX diagnostic
  passed with
  `artifact_payload_sha256=bf8c39c5947c063cb800c2f2b34f75bb3a70ac311aa3a388d6578a07a6692bb1`,
  manifest file SHA256
  `234d89567922ddad92d0c3750c35522f0c205616bfb3dbb3f47f8885e936d3d5`,
  `classification=terminal_y_p_sign_crossing_below_tolerance_after_positive_last_stage_he4`,
  `first_failing_terminal_Yp=-1.2294890184644955e-30`,
  `first_failing_last_attempted_He4=2.2765668298302704e-32`,
  `terminal_sign_transition=positive_to_nonpositive`,
  `x_phase2_tail_start=41`, and `he4_tail_index=46`.
- **Scope boundary:** FB83 is private source-localization evidence only.  It
  does not relax strict positivity, truncate or repair abundances, prove
  full-BBN completion, open public dispatch, claim production SMC validation,
  add QKE, or make the all-freedom path publication-ready.
- **Exit gate:** focused FB83 tests, real Yp source-probe run, py_compile,
  registry/WBS sync, internal review, and `git diff --check`.

---

### FB-84-CONTINUOUS-AP65-TERMINAL-FINAL-STATE-PROBE  Terminal tail provenance

- **Status:** stage-recorded (historical; not current capability).
- **Depends on:** FB70 span ladder and FB83 packed-state source probe.
- **Scope:** preserve the terminal FB69 `final_state_vector[-9:]` `X_phase2`
  tail in FB70 rows and compare `X_phase2[5]` against terminal BBN `Yp`.
- **Key files:** `src/rabbit/validation/augmented_continuous_ap65_full_bbn_span_ladder.py`,
  `tests/test_augmented_continuous_ap65_full_bbn_span_ladder.py`, and
  `docs/audit/fb84_continuous_ap65_terminal_final_state_probe.md`.
- **Physics added/changed:** no new collision formula, weak-rate formula,
  solver kernel, public dispatch route, production SMC path, or QKE path.
  FB84 is row-level provenance enrichment: it preserves terminal final-state
  tail evidence without changing evolution or positivity policy.
- **Validation:** focused tests cover available terminal final-state tail
  extraction and unavailable-probe behavior when `final_state_vector` is
  missing.  A real finite-difference CPU-JAX refresh through FB82 passed with
  nested
  `artifact_payload_sha256=47efcd214cc16b0810797d19d59baca5ab0a1e965ab169416ac2cdb3fe486609`,
  manifest file SHA256
  `81cfb5fc61419c14306f703326d333135cc34d0a4d172bc545cb27195d065acb`,
  `terminal_final_state_probe.he4_tail_index=46`,
  `terminal_final_state_probe.he4_from_final_state_vector=-1.2294890184644955e-30`,
  `terminal_final_state_probe.terminal_observable_Yp=-1.2294890184644955e-30`,
  and `terminal_final_state_probe.terminal_y_p_minus_final_state_he4=0.0`.
- **Scope boundary:** FB84 is private provenance evidence only.  It does not
  relax strict positivity, truncate or repair abundances, prove full-BBN
  completion, open public dispatch, claim production SMC validation, add QKE,
  or make the all-freedom path publication-ready.
- **Exit gate:** focused FB70 tests, real FB82 refresh run, py_compile,
  registry/WBS sync, internal review, and `git diff --check`.

---

### FB-85-CONTINUOUS-AP65-ADAPTIVE-STEP-ACCEPTANCE  Host step reject/retry

- **Status:** stage-recorded (historical; not current capability).
- **Depends on:** FB69 continuous host RHS prototype and FB70 span ladder.
- **Scope:** make the private host-stepped continuous AP65 Rodas5P prototype
  reject `err_norm > 1` attempts and retry at smaller `h` without advancing
  `N` or `y`; preserve accept/reject telemetry through FB70 rows.
- **Key files:** `src/rabbit/validation/augmented_continuous_ap65_rhs.py`,
  `src/rabbit/validation/augmented_continuous_ap65_full_bbn_span_ladder.py`,
  `tests/test_augmented_continuous_ap65_rhs.py`,
  `tests/test_augmented_continuous_ap65_full_bbn_span_ladder.py`, and
  `docs/audit/fb85_continuous_ap65_adaptive_step_acceptance.md`.
- **Physics added/changed:** no new collision formula, weak-rate formula,
  public dispatch route, production SMC path, QKE path, or abundance repair.
  FB85 is private solver-control hardening for the diagnostic host stepper.
- **Validation:** focused tests cover reject/retry behavior and FB70 telemetry
  propagation.  A real finite-difference CPU-JAX refresh through FB82 passed
  with nested
  `artifact_payload_sha256=9a0e0fe58cf8e318777b6b2a3cadae4cc367dd3424b6df75da178ba4a41b04dd`,
  manifest file SHA256
  `9a5d0cd620a4036ed1dc65c20842f6efc068d6de18fcd4091baffb9fad4ebee5`,
  `first_failure_row.attempt_count=2`,
  `first_failure_row.n_rejected=0`,
  `first_failure_row.error_norm_max=3.021584391530104e-14`,
  `first_failure_row.rejected_error_norm_max=0.0`, and unchanged
  `Yp=-1.2294890184644955e-30`.
- **Scope boundary:** FB85 is private solver-control evidence only.  It does
  not relax strict positivity, truncate or repair abundances, prove full-BBN
  completion, open public dispatch, claim production SMC validation, add QKE,
  or make the all-freedom path publication-ready.
- **Exit gate:** focused FB69/FB70 tests, real FB82 refresh run, py_compile,
  registry/WBS sync, internal review, and `git diff --check`.

---

### FB-86-CONTINUOUS-AP65-HE4-RHS-PROBE  Phase-2 boundary sign source

- **Status:** stage-recorded (historical; not current capability).
- **Depends on:** FB82 first-failure triage, FB84 terminal final-state probe,
  and FB85 adaptive step telemetry.
- **Scope:** evaluate the JAX phase-2 network RHS at raw terminal and
  last-attempted `X_phase2` tails plus diagnostic-only `He4=0`,
  `He4=1e-30`, and nonnegative trace-species counterfactual points.
- **Key files:** `src/rabbit/validation/augmented_continuous_ap65_he4_rhs_probe.py`,
  `scripts/run_augmented_continuous_ap65_he4_rhs_probe.py`,
  `tests/test_augmented_continuous_ap65_he4_rhs_probe.py`, and
  `docs/audit/fb86_continuous_ap65_he4_rhs_probe.md`.
- **Physics added/changed:** no new collision formula, weak-rate formula,
  solver kernel, public dispatch route, production SMC path, QKE path, or
  abundance repair.  FB86 is RHS-localization evidence for the trace-species
  positivity blocker.
- **Validation:** focused tests cover the negative-trace classification,
  fail-closed missing first-failure behavior, terminal `Y_p` versus final-state
  `He4` mismatch, trace-only flooring attribution, and CLI dry-run contract.  A
  real finite-difference CPU-JAX diagnostic passed with
  `artifact_payload_sha256=ebd16b1fa3b6d4b673e33c2cff07855a075d3ca7f288230ccf2b9fe24b275fdf`,
  manifest file SHA256
  `b170d117620b40dcf413e94b865774b3a6f0c4dc12b2e52d8568130cf3baf201`,
  `classification=he4_boundary_negative_due_to_negative_trace_intermediates`,
  `first_failure_negative_trace_indices=[3,4,6,7]`,
  `first_failure_negative_core_non_he4_indices=[]`,
  `first_failure_he4_zero_dHe4_network_rhs=-2.618301171321943e-21`, and
  `first_failure_nonnegative_trace_he4_zero_dHe4_network_rhs=7.273403769914826e-286`.
- **Scope boundary:** FB86 is private RHS-localization evidence only.  It does
  not relax strict positivity, truncate or repair abundances, prove full-BBN
  completion, open public dispatch, claim production SMC validation, add QKE,
  or make the all-freedom path publication-ready.
- **Exit gate:** focused FB86 tests, real FB86 run, py_compile,
  registry/WBS sync, internal review, and `git diff --check`.

---

### FB-87-CONTINUOUS-AP65-TRACE-POSITIVITY-GATE  Private RHS positivity policy

- **Status:** stage-recorded (historical; not current capability).
- **Depends on:** FB86 RHS boundary probe and FB70 span ladder.
- **Scope:** add an opt-in private continuous-AP65
  `abundance_positivity_policy=trace_boundary`, keep the raw network RHS
  available, and compare raw vs trace-boundary ladders over the same smoke spans.
- **Key files:** `src/rabbit/jax/augmented_typeI_replay.py`,
  `src/rabbit/validation/augmented_continuous_ap65_rhs.py`,
  `src/rabbit/validation/augmented_continuous_ap65_full_bbn_span_ladder.py`,
  `src/rabbit/validation/augmented_continuous_ap65_trace_positivity_gate.py`,
  `scripts/run_augmented_continuous_ap65_trace_positivity_gate.py`,
  `tests/test_jax_augmented_typeI_replay.py`,
  `tests/test_augmented_continuous_ap65_rhs.py`,
  `tests/test_augmented_continuous_ap65_trace_positivity_gate.py`, and
  `docs/audit/fb87_continuous_ap65_trace_positivity_gate.md`.
- **Physics added/changed:** a private phase-2 evolution RHS policy constrains
  trace/`He4` activities and active lower-bound derivatives, with raw-vs-policy
  phase-2 mass-fraction sum residuals recorded and gated by the comparison
  artifact.  This is not terminal `Y_p` truncation, public dispatch, QKE, or a
  promoted collision path.
- **Validation:** focused tests cover the direct boundary RHS helper, FB69
  metadata propagation, the FB87 raw-vs-policy artifact contract, fail-closed
  persistence behavior, and CLI dry-run.  A real finite-difference CPU-JAX
  diagnostic passed with
  `artifact_payload_sha256=dcdae7615088893f2bfbbece52620b8d81e60b1e775cb7ca8059c9d65a755276`,
  manifest file SHA256
  `99333c8747f006758fe9da2f0a2c8e633584a3e44a9d111999f8880d149f759d`,
  `classification=trace_boundary_resolves_smoke_y_p_sign_failure_with_conservation_gate`, raw first
  failure `N_span=[0.0,1.5e-09]`, raw `Yp=-1.2294890184644993e-30`,
  raw `Yp` failure rows `2`, trace-boundary failure rows `0`, and
  trace-boundary largest passing endpoint `2e-09`, with raw conservation max
  `6.284872348663924e-18`, trace-boundary conservation max
  `8.110492019931864e-18`, and conservation limit `1e-16`.
- **Scope boundary:** FB87 is private evolution-policy evidence only.  It does
  not prove full-BBN completion, open public dispatch, claim production SMC
  validation, add QKE, repair terminal observables, or make the all-freedom path
  publication-ready.
- **Exit gate:** focused FB87 tests, real FB87 run, py_compile,
  registry/WBS sync, internal review, and `git diff --check`.

---

### FB-88-CONTINUOUS-AP65-TRACE-SPAN-EXTENSION  Trace-boundary ladder extension

- **Status:** stage-recorded (historical; not current capability).
- **Depends on:** FB87 trace-boundary positivity gate and FB70 span ladder.
- **Scope:** extend the private trace-boundary continuous-AP65 span ladder
  beyond the FB87 `N_span_end=2e-9` smoke endpoint while preserving
  conservation, stiffness, and solver-effort telemetry.
- **Key files:** `src/rabbit/validation/augmented_continuous_ap65_trace_span_extension.py`,
  `scripts/run_augmented_continuous_ap65_trace_span_extension.py`,
  `tests/test_augmented_continuous_ap65_trace_span_extension.py`, and
  `docs/audit/fb88_continuous_ap65_trace_span_extension.md`.
- **Physics added/changed:** no new public physics or dispatch route.  FB88
  runs FB70 with `abundance_positivity_policy=trace_boundary` and chained
  restart handoff, then summarizes clean extension versus first-failure bracket
  plus conservation/stiffness/effort metadata.
- **Validation:** focused tests cover all-pass extension, pass/fail bracket
  classification, conservation-residual fail-closed behavior, missing
  row-level conservation/effort/stiffness telemetry, rejection-budget
  exceedance, nested public/QKE/full-BBN-readiness boundary violations, and CLI
  dry-run.
  A real finite-difference CPU-JAX diagnostic passed with
  `artifact_payload_sha256=49b2e9e858ffb87fece72c0ea2a031ed174eae9a0934db2e086c74c9997ba251`,
  manifest file SHA256
  `01be70a682384b58296b85a712ba4f05b9898fa95810804b8ba17ec8ca8507fb`,
  `classification=trace_boundary_extension_all_requested_spans_passed`, largest
  passing endpoint `5e-09`, rows passed/failed `3/0`, best
  `T_final_MeV=0.7999999960714048`, conservation max
  `8.746901892447222e-18`, conservation limit `1e-16`, complete
  conservation/solver/stiffness rows `3/3/3`, step/attempt totals `20/20`,
  rejected steps `0`, `error_norm_max=0.0006360385926131681`, source
  evaluations `1166`, and stage source evaluations `140`.
- **Scope boundary:** FB88 is private hot-endpoint span-extension evidence only.
  It does not prove full-BBN completion below `0.01 MeV`, open public dispatch,
  claim production SMC validation, add QKE, or make the all-freedom path
  publication-ready.
- **Exit gate:** focused FB88 tests, real FB88 run, py_compile,
  registry/WBS sync, internal review, and `git diff --check`.

---

## FB-89 — Continuous AP65 trace-boundary span growth scout

- **Scope:** add a private multiplicative span-growth scout above FB88 without
  opening public dispatch, production SMC validation, QKE, terminal abundance
  repair, or full-BBN readiness claims.
- **Key files:** `src/rabbit/validation/augmented_continuous_ap65_trace_span_growth.py`,
  `scripts/run_augmented_continuous_ap65_trace_span_growth.py`,
  `tests/test_augmented_continuous_ap65_trace_span_growth.py`, and
  `docs/audit/fb89_continuous_ap65_trace_span_growth.md`.
- **Physics added/changed:** no new public physics or dispatch route.  FB89
  generates a geometric trace-boundary ladder from the FB88 baseline, runs the
  FB88 gate, and re-checks nested FB88 claim boundaries plus conservation,
  stiffness, solver-effort, and rejection metadata.
- **Validation:** focused tests cover clean growth, pass/fail bracket
  propagation, nested FB88 ladder/row/telemetry mismatch, nested FB88 gate
  failure, nested public/QKE/full-BBN claim leakage, not-beyond-baseline
  failure, invalid geometric inputs, and CLI dry-run.  A real finite-difference
  CPU-JAX diagnostic passed with
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
  evaluations `2326`, and stage source evaluations `280`.
- **Scope boundary:** FB89 is private hot-endpoint span-growth evidence only.
  It does not prove full-BBN completion below `0.01 MeV`, open public dispatch,
  claim production SMC validation, add QKE, or make the all-freedom path
  publication-ready.
- **Exit gate:** focused FB89 tests, real FB89 run, py_compile,
  registry/WBS sync, internal review, and `git diff --check`.

---

## Appendix — Stable red-test baseline (as of PR-DOCS)

Documented for reference so subsequent PRs can tell whether a failure
is pre-existing:

1. `tests/test_registry_sync.py::test_supported_capabilities_mentions_features`
   — `SUPPORTED_CAPABILITIES.md` is missing the literal string
   "Inference".  **Pure documentation fix**; no roadmap PR owns it
   except implicitly PR-R.
2. `tests/test_production_gates.py::test_classB_typeV_bbn_gold`
   — Class B Type V Y_p fixture drift (rel 9 × 10⁻⁴).  Out of scope
   for Type I roadmap; belongs to the Class A/B effort, which has its
   own tracking.
3. `tests/test_production_gates.py::test_jax_flrw_gold`
   — **Test-side bug**: the test runs with
   `use_live_weak_monopoles=False` (equilibrium FD) but compares
   against the `live_f0_cl0` gold entry (0.26125) rather than the
   `jax_flrw_equilibrium` entry (0.2423504).  Trivial fix scheduled
   for PR-R.
4. `tests/test_production_gates.py::test_anisotropy_signal_parity`
   — Apples-vs-oranges: compares SciPy characteristic (nonperturbative)
   against JAX linearised PSTF (perturbative).  This item is now closed:
   PR-D rewrote the gate to compare SciPy reference against the promoted
   bounded `backend='auto'` characteristic surface.

Historically these four, and only these four, were the baseline.  With
PR-D merged, three legacy reds remain from that list.  Any new failure
introduced by a subsequent PR must still be explained in that PR's
catalogue entry.
