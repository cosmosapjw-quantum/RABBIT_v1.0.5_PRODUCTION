# PR-T3B JAX Operator Port — Preflight Audit

## Scope

Landed:
- `src/rabbit/jax/collisions_jax.py`
- `tests/test_pr_t3b_jax_operator_parity.py`

This patch is a **bounded preflight** that lands the pure-JAX ports of
the existing SciPy `NuEScatteringOperator` and `PairProcessOperator`.
The ports are JIT-compatible 1D collision kernels intended for the
future full-Boltzmann gather-collide-scatter (GCS) cycle without host
callbacks.  No driver, no public backend, no inference dispatch
changes; this PR only adds a new module and a new test file.

It does **not**:
- wire collisions into `driver_typeI_full_boltzmann.py`,
- extend `coupled_3T_rhs_jax` with external `dQ` sources (already
  delivered via the existing
  `coupled_3T_rhs_from_collision_moments_jax`),
- add detailed-balance or FLRW `N_eff` runtime locks on a real solver
  trajectory.

Those remain follow-up work for the full PR-T3B.

## Algorithmic mirror of the SciPy reference

The two new top-level kernels structurally mirror their SciPy
counterparts:

`nu_e_collision_jax` ↔ `NuEScatteringOperator._evaluate_vectorized`
- y3 (outgoing neutrino) integral on Gauss-Laguerre with `N_int`
  nodes, `N_int = min(N_q, 24)` to mirror the SciPy default.
- y2 (incoming electron) integral on Gauss-Laguerre with
  `N_quad_electron = 32` (default) and thermal FD weight
  `f_eq(y2) = 1/(e^{y2}+1)`.
- Off-node evaluation of the input distribution via the existing
  PCHIP cubic Hermite (`q_advection_jax::_pchip_slopes` /
  `_hermite_interp`).  When the Laguerre grid is shared
  (`q_nodes = laggauss(N_q) = y3_nodes`) the interpolation is identity
  at the input nodes.
- Matrix element, statistical factor, prefactor and `1/y1²` divisor
  bit-for-bit identical to the SciPy reference.

`pair_collision_jax` ↔ `PairProcessOperator.evaluate`
- y2 (antineutrino) integral on Gauss-Laguerre.
- y3 (outgoing positron) integral on Gauss-Legendre on `[0, y_sum]`
  with the SciPy mapping `y3 = 0.5*(y1+y2)*(leg+1)` and weight
  `0.5*(y1+y2)*leg_weights`.
- Off-node evaluation of `f_nubar` at `y2_nodes` via PCHIP; SciPy uses
  `scipy.interpolate.interp1d(kind='cubic')` (natural cubic spline).
  Element-wise parity is therefore identity-bound on the shared
  Laguerre grid (`N_q == N_quad`) and interpolant-bound otherwise.

## Reused surface

- `rabbit.collisions.kernels` for `G_F_MEV`, `G_L_NUE/NUX`,
  `G_R_NUE/NUX`.
- `rabbit.jax.q_advection_jax` for `_pchip_slopes` and `_hermite_interp`.
- `numpy.polynomial.laguerre.laggauss` and
  `numpy.polynomial.legendre.leggauss` for cached host-side
  quadrature builders, surfaced through `laguerre_grid` /
  `legendre_grid` (LRU-cached).

The ports do **not** reach into the existing preflight bridge
(`full_boltzmann_collision_preflight.py`) or the SciPy reference
operators; they live alongside as a JIT-compatible alternative.

## Verification

Smoke baseline (pre-change):
- `tests/test_pr_t3a_collisionless_driver.py` — `14 passed in 135.4 s`
- `tests/test_pr_t3b_collision_preflight.py` — `11 passed in 38.2 s`
- `tests/test_jax_typeI_characteristic_parity.py` — `18 passed in 58.4 s`

Post-change targeted bundle (regression):
- `tests/test_pr_t3a_collisionless_driver.py + test_pr_t3b_collision_preflight.py + test_pr_t3b_jax_operator_parity.py + test_jax_typeI_characteristic_parity.py`
  - `59 passed in 225.7 s` (43 prior + 16 new, no regression)

New parity tests (`tests/test_pr_t3b_jax_operator_parity.py`):
- `test_nu_e_jax_matches_scipy_shared_laguerre[N_q=12,20,24, species=nue,nux]`
  — element-wise `|Δ| < 1e-30` absolute on shared Laguerre grid.
- `test_nu_e_jax_detailed_balance_at_fd[species=nue,nux]` —
  `max|C[f_FD]| < 1e-30`.
- `test_pair_jax_detailed_balance_at_fd[species=nue,nux]` — at the
  matched-grid configuration `N_q == N_quad == 24`,
  `max|C_pair[f_FD, f_FD]| < 1e-30` (algebraic detailed balance is
  preserved when no interpolation is invoked).
- `test_pair_jax_matches_scipy_shared_laguerre[species=nue,nux]` —
  element-wise `|Δ| < 1e-30` absolute (or `rel < 1e-12`) on shared
  Laguerre grid `N_q == N_quad == 24` with a 50% above-FD
  perturbation that lifts the signal above any DB residual.
- `test_pair_jax_off_grid_pchip_close_to_scipy_cubic[species=nue,nux]`
  — at the off-grid configuration `N_q = 20`, `N_quad = 24`, the
  measured PCHIP-vs-scipy-cubic relative gap is `~8.3%`, locked at
  `< 10%` and intended to flag any further degradation.  Tightening
  this bound requires a JAX-native natural cubic spline replacement
  for PCHIP in `pair_collision_jax` — recorded as future work.

## Adversarial self-audit

Phase-prompt items that this preflight closes:
1. **(1) JAX operator ports match SciPy at 1e-12 elementwise** — closed
   for the shared-Laguerre configuration (the configuration the phase
   prompt explicitly references).  Off-grid parity remains at the
   PCHIP-vs-cubic interpolant gap (`~8.3% rel`).
2. **(2) Detailed balance** — closed for matched-grid configurations
   on both NuE and Pair.

Phase-prompt items that remain open (future PR-T3B steps):
3. **FLRW N_eff = 3.044 ± 0.01** — requires GCS wiring + `coupled_3T`
   coupling on a real Rodas5P trajectory; not in scope of this
   preflight.
4. **Sign convention `dQ_α > 0` when plasma hotter** — same.
5. **Rodas5P step rejection near freeze-out** — same.
6. **`coupled_3T_rhs_jax` backward compat** — not exercised here; the
   3T sector is unchanged by this patch.

Adversarial probes:
- **Sign / matrix element**: bit-for-bit copy of the SciPy
  `_matrix_element` and `_matrix_element_ann` formulas, cross-checked
  against the parity tests at shared grid.
- **Statistical factor**: `gain - loss` form mirrors SciPy.
- **Energy conservation cutoff**: `y4 = y1 + y2 - y3 > 0` mirrors
  SciPy's `if y4 <= 0: continue` guard via `jnp.where`.
- **Prefactor / divisor**: `G_F² T⁴ / (4π³)` and `1 / max(y1², 1e-30)`
  mirror SciPy.
- **Skip near-zero y1**: `q_nodes < 1e-15 → 0` mirrors SciPy.
- **Backward compatibility**: the patch only adds a new file
  (`collisions_jax.py`) and a new test file; no existing code paths
  are touched.

## Verdict

Conditional pass.

This closes the pure-JAX bridge between the SciPy reference operators
and the existing tier-3 preflight surface, while remaining a strictly
additive module change.  The shared-Laguerre 1e-12 parity gate is
green; the off-grid parity gap is documented and locked.  No public
backend, no driver, no inference dispatch is altered.

What remains for a real PR-T3B runtime patch:
- choose between (a) keeping the existing host-callback
  `direct_kernel_preflight` path and (b) replacing it with the
  pure-JAX kernels landed here,
- if (b), wire `nu_e_collision_jax` + `pair_collision_jax` into
  `driver_typeI_full_boltzmann.py` GCS,
- close FLRW `N_eff` lock, sign-convention check on `dQ_α`, and
  Rodas5P step rejection diagnostics on a real solver trajectory,
- consider replacing PCHIP with a JAX-native natural cubic spline so
  the off-grid pair-process parity tightens from `~8% rel` to
  reduction-order.
