# RABBIT Forward PR Roadmap — SDD / WBS

> **DEPRECATED as a forward plan (PUB-00, 2026-07-12).**  This document is a
> historical SDD/WBS record.  Its dependency graph, publication-grade targets,
> and acceptance instructions MUST NOT drive new work.  Current ordering and
> claim ceilings live only in
> [TYPEI_AUGMENTED_NOQKE_UNIFIED_FUTURE_PLAN.md](TYPEI_AUGMENTED_NOQKE_UNIFIED_FUTURE_PLAN.md).

The entries below preserve the historical design record and are not current
implementation instructions.

Companion documents: [ROADMAP_INDEX.md](ROADMAP_INDEX.md) for
navigation, [ROADMAP_STATE_OF_RECORD.md](ROADMAP_STATE_OF_RECORD.md)
for the baseline this roadmap starts from,
[ROADMAP_SELF_AUDIT.md](ROADMAP_SELF_AUDIT.md) for the audit template
that closes each PR, and [ROADMAP_PR_CATALOG.md](ROADMAP_PR_CATALOG.md)
for the completion log.

The roadmap contains **no calendar time-lines**.  Ordering is by
dependency, not by date.

---

## 0.  Dependency graph

```
                              (baseline: ROADMAP_STATE_OF_RECORD.md)
                                           │
                ┌──────────────────────────┼──────────────────────────┐
                ▼                          ▼                          ▼
          PR-A: Analytic J_j        PR-D: Auto backend           PR-N1: Non-LRS ray
          elimination               promotion to                 grid (S² quadrature)
          (state 37 → 25 DOF)       characteristic               + analytic direction map
                │                          │                          │
                ▼                          ▼                          ▼
          PR-J: Analytic            (roadmap-level                PR-N2: Non-LRS
          Jacobian blocks           configuration                 driver integration
          (remove jacfwd)           only)                         (char + PSTF)
                │                                                     │
                └──────────────────────────┬──────────────────────────┘
                                           ▼
                                      PR-G: GPU vmap batched
                                      solve (event-masked
                                      lax.while_loop)
                                           │
                ┌──────────────────────────┼──────────────────────────┐
                ▼                          ▼                          ▼
          PR-T3A: q-advection        PR-T3B: ν–e elastic +        PR-T3C: diagonal
          kernel + per-ray           pair collision wiring        ν–ν scattering
          f_j(q_k) state                                          operator
                │                          │                          │
                └──────────────────────────┼──────────────────────────┘
                                           ▼
                                      PR-T3D: Tier-3 full-
                                      collision integration +
                                      LASAGNA/FortEPiaNO cross-
                                      code lock
                                           │
                                           ▼
                                      PR-R: Release gate +
                                      roadmap catalogue sync
```

Each node is a PR; arrows are hard dependencies.  A PR may start before
its upstream merges, but cannot pass the audit gate without it.

---

## PR-A — Analytic J_j elimination

Reduce the characteristic-driver state from 37 DOF to 25 DOF by
replacing the numerically-evolved angular Jacobian `J_j` with its
closed-form expression.

### Purpose

`J_j` satisfies `dJ_j/dN = 3Σ(1-3μ²)J_j` with the analytic solution
(paper eq 51):
```
J_j = e^{-6 S} · (1 - μ_{j,0}²)² / (1 - μ_j²)²
```
Carrying `J_j` as a dynamical state variable is therefore unnecessary
— it can be recomputed from `(S, μ_{j,0}, μ_j(S))` at every RHS call
for negligible extra cost.

### Dependencies
None (works on the existing tier-1 or tier-2 characteristic layout).
Recommended to land before PR-J and PR-T3* because those amplify the
benefit.

### Architecture

1. Remove the `J_j` slots from `_char_layout`.  New layout:
   `[Σ_+, Σ_-, I_0..I_{Nμ-1}, S, T_γ, (T_νₑ, T_νₓ), X_i]`.
2. Add a helper
   `_jacobian_from_shear(S, X0, mu, mu0, signs)` that returns the
   per-ray `J_j` as a closed-form function.  Use it inside
   `_rhs_core` wherever `J_vals` was previously read.
3. Update all observable extractors (`extract_stress_jax`,
   `extract_monopole_jax`, `extract_monopole_from_background`) that
   accept `J` as input: they remain unchanged (they still accept `J`),
   but the caller now supplies the closed-form `J` rather than a state
   slice.
4. Update `characteristic_rhs_jax`: drop the `dJ/dN` return, keep
   `(dI, dS)`.
5. Update `_char_active_indices` for block-sparse mode: with `J`
   removed there is no passive block containing state; the I block is
   the only candidate.

### WBS

1. **Analytic-J_j test scaffold** (pre-code).
   Unit-test `J(S, μ_{j,0})` against numerically-integrated `J_j(N)`
   from the current 37-DOF driver at Σ_H = 0.1, 0.3, 0.5, 12 rays.
   Agreement must be ≤ 1 × 10⁻⁹ pointwise after the driver's
   rtol=1e-8.
2. **Layout migration.**  Edit `_char_layout`, `_char_active_indices`,
   initial-condition packer, phase-1→phase-2 handoff, observable
   extractor, metadata.
3. **RHS migration.**  Call `_jacobian_from_shear` inside `_rhs_core`;
   remove `dJ` from `jax.lax.dynamic_update_slice` call.
4. **Driver-entry migration.**  `_run_char_impl` no longer initialises
   `J` slots.
5. **Parity lock.**  Re-run
   `tests/test_jax_typeI_characteristic_parity.py` and
   `tests/test_jax_typeI_characteristic_tier2.py`.  Target:
   unchanged within solver noise (≤ 5 × 10⁻⁸ Y_p).
6. **Documentation.**  Update
   [ROADMAP_STATE_OF_RECORD.md §1.2](ROADMAP_STATE_OF_RECORD.md#12-three-transport-methods-in-the-codebase)
   (state DOF: 37 → 25); append entry to
   [ROADMAP_PR_CATALOG.md](ROADMAP_PR_CATALOG.md).

### Exit criteria

- [ ] Unit test `J(S,μ₀) == J(N)` numerical parity ≤ 1 × 10⁻⁹.
- [ ] Existing tier-1 / tier-2 parity tests all green at baseline tol.
- [ ] Warm single-solve timing ≤ current baseline (no regression).
- [ ] Dense Jacobian column count is 25, not 37, as measured by
      `jax.jacfwd(rhs_p1, argnums=1)(...).shape[1]`.
- [ ] `STATE_OF_RECORD.md` and `PR_CATALOG.md` updated.

### Risk

- Near `μ_j = ±1` the closed form diverges.  Gauss–Legendre nodes
  never place a ray at `|μ| = 1`, so the pole is avoided by
  construction — the same holds in SciPy already, and the unit test
  in WBS #1 will catch any regression.

### Topic guide reference
[IMPLEMENTATION_GUIDE_FULL_COLLISION_NOQKE.md §2](IMPLEMENTATION_GUIDE_FULL_COLLISION_NOQKE.md#2-state-vector-upgrade-paper-appendix-e)
motivates the state-dim reduction; WBS here targets the LRS tier-1/2
driver, but the pattern carries over to tier-3.

---

## PR-J — Analytic Jacobian blocks

Replace `jax.jacfwd(rhs, argnums=1)` with hand-written closed-form
Jacobian blocks so that every accepted Rodas5P step performs O(1) RHS
evaluations instead of `dim(state)` forward-mode tangents.

### Purpose

Rodas5P requires a Jacobian at every accepted step.  At the current
37-DOF state (or 25 after PR-A), `jacfwd` performs 37 (or 25) forward
RHS evaluations per step.  This is the dominant single-solve cost
after the RHS-identity cache fix.  Closed-form Jacobian blocks reduce
this to ~1 effective RHS eval per step.

### Dependencies

PR-A (reduces Jacobian size 37 → 25 → smaller analytic blocks).

### Architecture

Assemble the Jacobian `∂(dy)/∂y` as a sparse block structure.  Each
non-zero block is given by a closed-form expression derived from the
RHS — no autodiff.

Blocks to implement (LRS, tier-1 first; tier-2 adds two thermo rows):

| Block | Expression |
|---|---|
| ∂(dΣ_+)/∂Σ_+ | `−(1 − Σ²) + 2Σ_+ Σ_+` |
| ∂(dΣ_+)/∂S | `f_ν Σ_j w_j J_j · P_2'(μ_j) · (∂μ_j/∂S) e^{-8I_j}` |
| ∂(dΣ_+)/∂I_j | `−8 f_ν w_j J_j P_2(μ_j) e^{-8I_j}` |
| ∂(dI_j)/∂Σ_+ | `P_2(μ_j)` |
| ∂(dI_j)/∂S | `Σ_+ · P_2'(μ_j) · (∂μ_j/∂S)` |
| ∂(dS)/∂Σ_+ | `1` |
| ∂(dT_γ)/∂T_γ | from `tier1_dT_gamma_dN_jax` or `coupled_3T_rhs_jax` autodiff |
| ∂(dX)/∂I_j | weak-rate chain rule through f̃₀(q) |
| ∂(dX)/∂X | from `abundance_rhs_phase{1,2}` (thermo_tier-independent) |

The weak-rate chain rule is the hardest block.  Fallback: keep
`jacfwd` for the `X`-row through weak rates only, since the weak
sector has O(9) rows (= n_species) and the tangent cost is modest.
This gives an **80/20** split: the 25 geometry/transport rows use
analytic blocks, the 9 network rows use targeted autodiff.

### WBS

1. **Symbolic derivation notebook** (not committed) — derive each
   block's closed form from the RHS in `_rhs_core`.
2. **Block implementation.**  One function per block, tested against
   `jax.jacfwd` at 10⁻¹⁰ element-wise tolerance.
3. **Jacobian assembler.**  Replace
   `_cached_jacfwd(rhs_fn)` with a new
   `_analytic_jac_fn(rhs_fn)` that returns a `(state_dim, state_dim)`
   dense matrix assembled from blocks.  Keep the Schur-friendly
   sparsity so that `_rodas5p_step_schur` still works when requested.
4. **Hybrid fallback.**  For the X-row through weak rates, use a
   targeted `jax.vjp` over just those rows.
5. **Benchmark.**  Target: warm single-solve time halved or better at
   25-DOF post-PR-A state.
6. **Parity lock.**  Re-run both tier-1 and tier-2 parity suites;
   target unchanged.
7. **Documentation.**  Update
   [JAX_CHAR_GPU_OPTIMIZATION_PLAN.md §2.4](JAX_CHAR_GPU_OPTIMIZATION_PLAN.md#24-analytic-jacobian-in-place-of-jacfwd)
   to reference the implementation.  Append to catalogue.

### Exit criteria

- [ ] Block-wise unit tests vs `jacfwd` ≤ 10⁻¹⁰ all greens.
- [ ] Single-solve warm time ≤ 65 %% of baseline.
- [ ] All tier-1/tier-2 parity tests unchanged.
- [ ] `_analytic_jac_fn` preserves the block-sparse partitioning so
      that GPU `jacobian_mode="block_sparse"` remains usable.

### Risk

- Symbolic drift.  Any hand-derived block risks typographic error.
  Mitigated by the exhaustive `jacfwd` cross-check at WBS #2.
- Weak-rate block complexity.  The fallback to targeted `vjp` is
  cheap and eliminates this risk.

---

## PR-N1 — Non-LRS ray grid and direction map

Add the S² quadrature machinery and non-LRS direction forward map
required for generic (Σ_- ≠ 0) Bianchi Type I transport.  Stand-alone
PR: no driver wiring yet.

### Purpose

Today's characteristic driver collapses the direction to `μ ∈ [-1, 1]`
with an analytic forward integral that is only valid in LRS.  Non-LRS
Type I has two distinguished axes, so the ray direction lives on the
full sphere S².

This PR adds the mathematical primitives without changing any
driver's default behaviour.

### Dependencies
None.  Can land in parallel with PR-A / PR-J.

### Architecture

New module `src/rabbit/jax/characteristic_rays_nonlrs_jax.py`:

- `setup_ray_grid_S2(N_θ, N_φ)` — tensor-product Gauss–Legendre (θ) ×
  uniform (φ) grid; returns `(mu_grid, phi_grid, w_grid)` flat arrays
  of shape `(N_θ · N_φ,)`.
- Optional Lebedev loader for later optimisation.
- `mu_current_nonlrs(mu_0, phi_0, S_plus, S_minus)` — forward map from
  initial direction + two accumulated shear integrals to current
  `(μ, φ)`.  Derivation: paper §2.1 geodesic equation with two shear
  eigenvalues, integrated via Rodrigues' rotation formula for the
  generator `diag(Σ_+ + √3 Σ_-, Σ_+ − √3 Σ_-, −2 Σ_+)`.
- `characteristic_rhs_nonlrs_jax(Sigma_plus, Sigma_minus, I, J, mu, phi)` —
  returns `(dI, dJ, dS_plus, dS_minus)` where now `dS_plus/dN = Σ_+`,
  `dS_minus/dN = Σ_-`.
- `extract_stress_plus_S2(I, J, mu, phi, w, f_nu)`,
  `extract_stress_minus_S2(I, J, mu, phi, w, f_nu)` — two stress
  components (paper eq 14 generalised to `|m|=2` mode of the PSTF
  quadrupole).
- `extract_monopole_S2(I, J, w, q_nodes)` — solid-angle-averaged
  monopole.

### WBS

1. **S² quadrature builder.**
2. **Non-LRS forward-map derivation + unit test.**  Test against
   numerical integration of `dμ/dN, dφ/dN` for 5 random initial
   directions, Σ_+ = 0.2, Σ_- = 0.1, over 8 e-folds.  Tolerance 10⁻⁹.
3. **Stress / monopole extractors.**  Unit-test against the LRS
   limit: set `Σ_- = 0`, use `N_φ = 1` tensor product with trivial φ
   weight, confirm equality to the LRS `extract_stress_jax` /
   `extract_monopole_jax` to 10⁻¹⁰.
4. **Documentation.**  Cross-reference from
   [IMPLEMENTATION_GUIDE_NON_LRS_TYPEI.md](IMPLEMENTATION_GUIDE_NON_LRS_TYPEI.md).

### Exit criteria

- [ ] All unit tests green.
- [ ] LRS reduction test (N_φ = 1) exact to 10⁻¹⁰.
- [ ] No change to existing driver numerics.

### Risk

- Two-component direction ODE stiffness near `|μ| = 1`.  Guard by
  mirroring the LRS's `1 - μ² > ε` floor.

---

## PR-N2 — Non-LRS driver integration

Wire the non-LRS primitives from PR-N1 into the JAX characteristic
driver behind a new transport mode and backend key.

### Purpose
Deliver a publication-grade non-LRS Bianchi I BBN path.

### Dependencies
PR-N1.  Benefits from PR-A (smaller state) and PR-J (analytic
Jacobian) but does not require them.

### Architecture

- New config switch `transport_mode="characteristic_nonlrs"` in
  `JAXTypeICharConfig`.
- New layout helper `_char_layout_nonlrs(N_θ, N_φ, n_species,
  thermo_tier)` returning the extended index set
  `[Σ_+, Σ_-, I_jk, J_jk, S_+, S_-, T_γ, (T_νₑ, T_νₓ), X_i]`.
- New RHS kernel
  `_rhs_core_nonlrs(...)` that branches on `transport_mode` and uses
  the PR-N1 primitives.
- Dispatch branch in
  `JAXTypeIConfig.__post_init__` that *accepts* `Sigma_H_minus ≠ 0`
  when `transport_mode="characteristic_nonlrs"`, and still raises for
  the LRS characteristic path.
- New backend capability `JAX_TYPEI_CHARACTERISTIC_NONLRS_TIER1` and
  backend key `jax_characteristic_nonlrs` registered in
  [`src/rabbit/config/backend_capabilities.py`](../src/rabbit/config/backend_capabilities.py).
- Dispatch branch in
  [`src/rabbit/inference/forward_likelihood.py`](../src/rabbit/inference/forward_likelihood.py).

### WBS

1. **Layout + initial conditions + handoff.**
2. **RHS kernel with tier-1 / tier-2 branch.**
3. **Stress → geometry feedback on Π_+ and Π_-** (both non-zero in
   generic Type I).
4. **Config validation, backend registration, dispatch.**
5. **Parity tests.**
   - LRS limit: `Σ_- = 0` with `N_φ = 1` matches LRS driver to 10⁻¹⁰.
   - Swap symmetry: `(Σ_+, Σ_-) ↔ (Σ_+, −Σ_-)` (reflection) gives
     identical observables.
   - Small-shear generic: at `Σ_+ = Σ_- = 0.05`, agree with
     linearised-PSTF generic (n_ell=3) to within the known 20–30 %%
     nonlinear gap (direction-sign must agree).
6. **Documentation.**  Update
   [IMPLEMENTATION_GUIDE_NON_LRS_TYPEI.md](IMPLEMENTATION_GUIDE_NON_LRS_TYPEI.md#test-matrix-must-all-pass-before-promotion)
   test-matrix results.

### Exit criteria

- [ ] LRS reduction bitwise parity.
- [ ] Swap-symmetry test at 10⁻¹⁰.
- [ ] 6-point Σ_+, Σ_- grid parity matrix filled.
- [ ] `STATE_OF_RECORD.md §2.3` extended with non-LRS row;
      `§4` table gains non-LRS parity entries.

### Risk

- Rodas5P state grows from 25 DOF (LRS post-PR-A) to `2 + 2·N_θ·N_φ +
  2 + ...` ~ 200 DOF at (N_θ=12, N_φ=16).  `jacfwd` cost matters here.
  Strongly prefer PR-J to land first.

---

## PR-D — Auto-backend promotion to characteristic

Configuration-only PR: make `backend="auto"` resolve to the JAX
characteristic path so the published inference defaults use the
publication-grade transport.

### Purpose

Today `backend="auto"` resolves to `jax_typeI_liveweak_cl3_tier1`
(linearised PSTF).  Downstream tests that compare `auto` against
`scipy` (characteristic) report the pre-existing apples-vs-oranges
physics gap as a test failure (paper §6.8, ~21 %% of the shear-induced
Y_p shift recovered by linearised).

### Dependencies
None.  Pure dispatch configuration.

### Architecture

- `CAPABILITY_BY_BACKEND["auto"]` and
  `resolve_typeI_auto_backend` logic switched from
  `JAX_TYPEI_LIVEWEAK_CL3_TIER1` to `JAX_TYPEI_CHARACTERISTIC_TIER1`
  (tier-1) or, behind an explicit `jax_thermo_tier=2` knob, to
  `JAX_TYPEI_CHARACTERISTIC_TIER2`.
- Corresponding updates to `STATUS.md`, `README.md`,
  `BACKEND_CAPABILITY_MATRIX.md`.

### WBS

1. **Update capability dispatch table and auto-resolver.**
2. **Update documentation strings and doc fixtures to reflect new
   default.**
3. **Test cleanup.** The red test
   `test_production_gates.py::test_anisotropy_signal_parity` now
   passes because both backends use the characteristic path.
4. **Documentation.**  Update
   [ROADMAP_STATE_OF_RECORD.md §2.4](ROADMAP_STATE_OF_RECORD.md#24-other-dispatch-backends)
   (auto row).

### Exit criteria

- [ ] `auto` resolves to `jax_typeI_characteristic_tier1`.
- [ ] `test_anisotropy_signal_parity` now passes.
- [ ] `test_registry_sync.py` doc-sync tests unchanged (the
      `auto → liveweak` claim in `STATUS.md` must be updated to
      `auto → characteristic`).

### Risk

- Downstream consumers that relied on `auto` reproducing linearised
  numbers will see a physics shift.  Mitigation: document the flip in
  `STATE_OF_RECORD.md §3` design-decisions and keep `jax` (explicit)
  pointing to linearised for backward compatibility.

---

## PR-G — GPU vmap batched solve

Add a vmap-compatible Rodas5P runner so that batch inference
(≥ 256 solves) can amortise GPU kernel-launch overhead.  **Rodas5P is
retained** — the vmap is added on top of the existing
`_solve_core`-style pure-JAX solver, not on top of diffrax.

### Purpose

The 37-DOF (or 25-DOF post-PR-A) characteristic state is kernel-launch
bound on GPU.  Single solves are slower on GPU than on CPU.  A vmap
over ≥ 64 simultaneous solves, however, amortises launch cost over
the batch and delivers GPU's real benefit.

### Dependencies

PR-A and PR-J recommended for throughput; not strictly required.
Non-LRS (PR-N2) benefits from this but can also land independently.

### Architecture

- Add `_solve_core_event_masked` to
  [`src/rabbit/jax/solver_jax_rodas5p.py`](../src/rabbit/jax/solver_jax_rodas5p.py)
  — a pure-JAX event-aware solve loop that carries a boolean
  "finished" mask per batch element.  Finished elements freeze their
  state while other elements continue stepping; the `lax.while_loop`
  exits when every element has triggered its event or hit `max_steps`.
- New top-level helper `run_char_batch(sigmas, common_config)` in
  `driver_typeI_char.py`:
  ```python
  rhs_p1, layout_p1 = _get_char_rhs(phase=1, ...)
  rhs_p2, layout_p2 = _get_char_rhs(phase=2, ...)

  def solve_one(sigma_plus):
      y0 = _build_y0(sigma_plus, layout_p1)
      y1, N1, ok1 = _solve_core_event_masked(rhs_p1, y0, event_p1, ...)
      y2_init = _handoff(y1, layout_p2)
      y_final, N_end, ok2 = _solve_core_event_masked(rhs_p2, y2_init, event_p2, ...)
      return _observables(y_final, layout_p2), ok1 & ok2

  observables_batch, success_batch = jax.vmap(solve_one)(sigmas)
  ```
- Env-var documentation: `XLA_PYTHON_CLIENT_PREALLOCATE=false`,
  `XLA_PYTHON_CLIENT_ALLOCATOR=platform` or
  `XLA_PYTHON_CLIENT_MEM_FRACTION=0.10`.

### WBS

1. **Event-masked Rodas5P core.**  New pure-JAX helper; unit test
   against the existing event-detect path for 3 distinct σ values
   solved sequentially vs once in a batch (must agree bitwise).
2. **Batch-dispatcher helper** `run_char_batch`.
3. **Breakeven benchmark.**  Measure GPU vs CPU throughput at
   N ∈ {1, 16, 64, 256, 1024}; record in
   [JAX_CHAR_GPU_OPTIMIZATION_PLAN.md §2.2](JAX_CHAR_GPU_OPTIMIZATION_PLAN.md#22-vmap-batching-on-top-of-rodas5p-the-headline-gpu-win).
4. **Documentation.**  Cross-reference from
   [ROADMAP_STATE_OF_RECORD.md §3.1](ROADMAP_STATE_OF_RECORD.md#31-cpu-preferred-default-for-the-characteristic-driver)
   (CPU-preferred remains default; GPU is opt-in via the new helper).

### Exit criteria

- [ ] Bitwise parity (tight: ≤ 10⁻¹²) between batch and sequential
      solves at N = 16.
- [ ] GPU batch solve of N = 256 completes with peak VRAM ≤ 2 GB when
      `XLA_PYTHON_CLIENT_MEM_FRACTION=0.10` is set.
- [ ] CPU default path unchanged.

### Risk

- Finished-mask semantics.  If the mask is implemented incorrectly,
  finished elements could continue to drift and contaminate shared
  batch state.  Mitigation: bitwise parity test.
- Rodas5P step rejection under vmap.  A rejected step from one
  element forces the whole batch to retake the step; this is
  acceptable (worst-case cost is equal to the slowest element's step
  count, already a documented property of vmap-over-while_loop).

---

## PR-T3A — q-advection kernel + full-phase-space state

Start the tier-3 stack: replace the scalar per-ray state `I_j` with
the full momentum-resolved distribution `f_j(q_k)` and add a stable
q-advection scheme for the gravitational redshift term.

### Purpose
Foundation for full-collision incomplete decoupling.  No collision
wiring yet — this PR verifies that the new state representation
recovers the existing (Paper I) characteristic driver in the
collisionless limit.

### Dependencies
PR-A (state layout flexibility), PR-J (analytic Jacobian becomes
mandatory at ~975 DOF).

### Architecture

- New module
  `src/rabbit/jax/q_advection_jax.py`:
  ```python
  def semi_lagrangian_step(f, q_nodes, dI_step, spline_matrix):
      """Advect f(q) → f(q · exp(-2 dI)) via cubic spline."""
      q_new = q_nodes * jnp.exp(-2 * dI_step)
      return cubic_interp(f, q_nodes, q_new, spline_matrix)
  ```
  with PCHIP fallback if the not-a-knot spline produces negative
  distribution values at sharp spectral features.
- New driver
  `src/rabbit/jax/driver_typeI_full_boltzmann.py`:
  per-species, per-ray, per-momentum state; advection applied at
  every RHS evaluation.
- `run_full_boltzmann_config` switch (not yet the default).

### WBS

1. **q-advection helper + spline matrix builder.**
2. **Driver shell** (no collisions).
3. **Collisionless reduction test.**  Disable all collisions →
   results must match `rabbit.jax.driver_typeI_char` (post-PR-A) to
   ≤ 10⁻⁸.
4. **Documentation.**  Update
   [IMPLEMENTATION_GUIDE_FULL_COLLISION_NOQKE.md §3.1](IMPLEMENTATION_GUIDE_FULL_COLLISION_NOQKE.md#phase-1--q-advection-kernel-400-loc--200-tests)
   with measured diffusion bounds.

### Exit criteria

- [ ] Free-streaming test: `f_j(q) = f_FD(q e^{2 I_j})` recovered to
      ≤ 10⁻¹⁰ after 500 steps.
- [ ] Collisionless Y_p matches characteristic driver to ≤ 10⁻⁸ at
      Σ_H = 0.1, N_μ=12, N_q=20.
- [ ] Positivity: `f_j(q_k) ∈ [0, 1]` throughout the integration.

### Risk

- Spline reconstruction injects negative distribution values at
  sharp features.  Guarded by PCHIP / WENO fallback.

---

## PR-T3B — ν–e elastic + pair collision wiring

Turn on the two existing collision operators (Hannestad–Madsen ν–e and
the pair process operator) inside the full-phase-space driver from
PR-T3A.

### Purpose
Recover N_eff ≈ 3.044 at FLRW from a *physical* evolution of the
neutrino distribution rather than the momentum-averaged tier-2
source.

### Dependencies
PR-T3A.

### Architecture

Reuse:
- [`src/rabbit/collisions/nu_e_scattering.py`](../src/rabbit/collisions/nu_e_scattering.py)
- [`src/rabbit/collisions/pair_processes.py`](../src/rabbit/collisions/pair_processes.py)

Gather–collide–scatter cycle at each timestep (paper Appendix E.3):
```
f̃_0(q), f̃_2(q) ← gather from rays
C_0(q), C_2(q)   ← NuEScatteringOperator(f̃_0, T_γ, T_νₑ) + PairProcessOperator(f̃_0, f̃_0̄)
df_j(q_k)/dN   += [C_0(q_k) + C_2(q_k) P_2(μ_j)] / H    (added to the advection term)
```

Wire the collision **energy transfer** into `coupled_3T_rhs_jax` as
an external `dQ_α/dN` argument (currently hard-coded to zero).  This
requires a new signature
`coupled_3T_rhs_with_sources_jax(T_γ, T_νₑ, T_νₓ, dQ_nue, dQ_nux,
H_MeV=...)`.

### WBS

1. **Add dQ_α/dN pathway in `coupled_3T_rhs_jax`.**  Preserve
   backward compatibility by defaulting external sources to 0.
2. **Wire collision operators into the tier-3 driver RHS.**
3. **FLRW N_eff lock test:** Σ_H = 0, full tier-3 → N_eff = 3.044 ±
   0.005.  Reference: Mangano et al. 2005.
4. **Energy/momentum conservation test:** per-step residual of the
   total stress–energy conservation should be < 10⁻⁸ of the radiation
   density.
5. **Documentation.**  Update
   [IMPLEMENTATION_GUIDE_FULL_COLLISION_NOQKE.md §3.2](IMPLEMENTATION_GUIDE_FULL_COLLISION_NOQKE.md#phase-2--per-ray-distribution-driver-500-loc)
   and
   [ROADMAP_STATE_OF_RECORD.md §4.3](ROADMAP_STATE_OF_RECORD.md#43-flrw-cross-code-target-future-tier-3).

### Exit criteria

- [ ] `|N_eff − 3.044| < 0.01` at FLRW, CL0.
- [ ] Stress–energy conservation ≤ 10⁻⁸ per step.
- [ ] Cross-code lock vs at least one of LASAGNA/FortEPiaNO at
      `|ΔY_p| < 5 × 10⁻⁴`.

### Risk

- Stiffness at `T_γ ~ 3 MeV` where Γ_ν/H ≫ 1.  Rodas5P handles this
  by design (confirmed in paper §16.1 rationale).  Monitor step
  rejection rate.
- Detailed-balance drift.  Check every 100 steps that `C[f_eq] = 0`
  to 10⁻¹² at the current T_γ.

---

## PR-T3C — Diagonal ν–ν scattering operator

Implement the ν–ν scattering operator absent from the current
codebase.  Covers diagonal-flavour terms only (Fierz-diagonal);
off-diagonal terms belong to Tier 4 / QKE and remain out of scope.

### Purpose
Close the remaining ~0.01 gap in `N_eff` after PR-T3B.

### Dependencies
PR-T3B (the integration scaffolding).  PR-A / PR-J strongly
recommended (state is ~975 DOF at Tier-3).

### Architecture

New module `src/rabbit/collisions/nu_nu_scattering.py`:

- Port the Mangano 2005 kernel (paper refs [17, 18]) for the diagonal
  ν–ν process `ν_α + ν_β → ν_α + ν_β`.
- Ultra-relativistic massless limit; angular integration reduces to
  1D in the partner momentum `y_2`.
- Statistical factor uses the gathered monopole `f̃_0(q)` from all
  neutrino species.
- Interface mirrors `NuEScatteringOperator`:
  `evaluate(delta_f, q_nodes, T_nu) → C_array`.

Register in the tier-3 driver's gather–collide–scatter loop.

### WBS

1. **ν–ν kernel derivation + implementation.**
2. **Detailed-balance test:** `C_νν[f_eq] = 0` to 10⁻¹⁴.
3. **Energy conservation test:** ∫ y³ C_νν(y) dy = 0 to 10⁻¹²
   (elastic within the neutrino sector).
4. **Tier-3 integration.**  Wire the operator into the GCS loop;
   verify that N_eff moves by ≲ 0.005 at FLRW relative to PR-T3B.
5. **Documentation.**  Update
   [IMPLEMENTATION_GUIDE_FULL_COLLISION_NOQKE.md §3.3](IMPLEMENTATION_GUIDE_FULL_COLLISION_NOQKE.md#phase-3--νν-scattering-operator-300-loc--200-tests).

### Exit criteria

- [ ] ν–ν detailed balance ≤ 10⁻¹⁴.
- [ ] ν–ν elastic energy conservation ≤ 10⁻¹².
- [ ] FLRW N_eff = 3.044 ± 0.005 (lock).
- [ ] No worsening of the energy-momentum conservation test from
      PR-T3B.

### Risk

- Detailed-balance numerical drift at the Fierz-diagonal truncation.
  Test at baseline, then at `T = 0.1` MeV after 10⁸ s of evolution.

---

## PR-T3D — Tier-3 full-collision integration + cross-code lock

> **Legacy status warning (2026-05-09):** this PR-T3D WBS entry is retained as
> a historical target, not as the current capability claim.  The live state of
> record is the AP-form no-QKE tier-3 candidate/diagnostic surface documented in
> `ROADMAP_STATE_OF_RECORD.md`, `SUPPORTED_CAPABILITIES.md`, and
> `IMPLEMENTATION_GUIDE_FULL_COLLISION_NOQKE.md`.  Do not read the
> "publication-grade full collision" wording below as landed public production
> support.

Promote the Tier-3 full-collision driver to the primary JAX path for
incomplete-decoupling science.

### Purpose
Replace the momentum-averaged tier-2 thermo collision source with the
full kernel; land the publication-grade N_eff = 3.044 baseline.

### Dependencies
PR-T3A, PR-T3B, PR-T3C, PR-J (mandatory for tier-3 throughput).

### Architecture

- New backend key `jax_full_collision_tier3` registered in
  `backend_capabilities.py`.
- New capability `JAX_TYPEI_FULL_BOLTZMANN_TIER3`.
- `canonical_forward_solver(backend="jax_full_collision_tier3", ...)`
  dispatch.
- Metadata keys:
  `transport_mode="full_boltzmann_ray"`,
  `collision_closure_mode="full_nonperturbative_tier3_no_qke"`,
  `production_authority="paper_II_candidate"`.

### WBS

1. **Backend registration + dispatch.**
2. **Cross-code parity suite.**  New tests in
   `tests/test_jax_tier3_cross_code_parity.py`:
   - FLRW Y_p vs LASAGNA / FortEPiaNO / PRIMAT-AC2024 within
     `|ΔY_p| < 5 × 10⁻⁴`.
   - N_eff within 3.044 ± 0.005.
3. **Anisotropic tier-3 stability sweep:** Σ_H ∈ {0, 0.1, 0.3, 0.5},
   confirm N_eff moves < 10⁻³ across the sweep.
4. **Retire momentum-averaged tier-2 as publication path.**  Keep it
   as a fast-approximation mode but document that Tier-3 is the
   publication default.
5. **Documentation.**  Final update to
   [IMPLEMENTATION_GUIDE_FULL_COLLISION_NOQKE.md](IMPLEMENTATION_GUIDE_FULL_COLLISION_NOQKE.md);
   update `STATE_OF_RECORD.md §4` parity table with tier-3 rows;
   catalogue entry.

### Exit criteria

- [ ] All cross-code targets met.
- [ ] No regression in tier-1 / tier-2 parity.
- [ ] Tier-3 documented as the publication-grade path in
      `STATE_OF_RECORD.md §2.3`.

### Risk

- Cross-code agreement tighter than 5 × 10⁻⁴ may reveal subtle
  differences in ν-ν Fierz conventions or QED-EoS conventions.
  Document any residual offset and its origin.

---

## PR-R — Release gate and roadmap catalogue sync

Final ship-readiness PR.

### Purpose

Run the canonical acceptance checklist, freeze the release tag, and
bring all cross-references up to date in the catalogue.

### Dependencies
All of PR-A, PR-J, PR-N1/N2, PR-D, PR-G, PR-T3A/B/C/D.

### Architecture
Pure documentation + CI gating.

### WBS

1. **Canonical acceptance checklist.**  Re-run the
   `ROADMAP_SELF_AUDIT.md` audit template for every backend.
2. **Update `STATE_OF_RECORD.md` to the post-tier-3 state.**
3. **Update `README.md` capability table.**
4. **Update `PROMOTION_GATES.md` with new canonical tier rows.**
5. **Update `CAPABILITY_BY_BACKEND` dispatch comment.**
6. **Cross-code fixture freeze.**
7. **Tag release.**

### Exit criteria

- [ ] Every `STATE_OF_RECORD.md` section reflects post-tier-3 reality.
- [ ] `ROADMAP_PR_CATALOG.md` contains a completion record for every
      PR in this roadmap.
- [ ] Every test passes or has a documented, stable `xfail` with
      rationale.

---

## Appendix A — Work-item sizing

Indicative relative sizing (not timeline):

| PR | Approx. LOC | Test LOC | Relative difficulty |
|---|---|---|---|
| PR-A | 200 | 150 | Low-medium |
| PR-J | 500 | 400 | Medium-high |
| PR-N1 | 300 | 200 | Medium |
| PR-N2 | 400 | 400 | Medium-high |
| PR-D | 30 | 20 | Trivial |
| PR-G | 400 | 300 | Medium |
| PR-T3A | 600 | 300 | High |
| PR-T3B | 200 | 300 | Medium |
| PR-T3C | 400 | 300 | Medium-high |
| PR-T3D | 200 | 500 | Medium (depends on T3A–C) |
| PR-R | 100 (docs only) | — | Trivial |

These sizings are inherited from the implementation guides' sizing
sections and should be treated as order-of-magnitude estimates.

---

## Appendix B — Per-PR audit-and-docs gate

Every PR in this roadmap closes with the steps prescribed in
[ROADMAP_SELF_AUDIT.md](ROADMAP_SELF_AUDIT.md):

1. Run the **audit checklist** for the PR's scope (physics, numerics,
   test coverage, performance).
2. Run the **documentation-update script** (see `ROADMAP_SELF_AUDIT.md`
   §3) that synchronises:
   - `ROADMAP_STATE_OF_RECORD.md` — state, parity numbers, file
     inventory, test count.
   - `ROADMAP_PR_CATALOG.md` — appended completion record.
   - Relevant topic guide(s) — status transitions (candidate →
     production, etc.).
3. Commit the audit report and documentation updates **in the same
   PR** as the code change.
