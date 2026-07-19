# PR-T3C Diagonal ν–ν Skeleton — Preflight Audit

## Scope

Landed:
- `src/rabbit/jax/nu_nu_scattering_jax.py`
- `tests/test_pr_t3c_nu_nu_preflight.py`

This is a **bounded, additive preflight** for PR-T3C (diagonal ν–ν
elastic scattering).  It lands a JAX-native structural skeleton for
the ``ν_α + ν_β -> ν_α + ν_β`` Fierz-diagonal operator, locks the
algebraic invariants on the matched Laguerre grid, and explicitly
documents the gaps that remain before the full PR-T3C runtime patch
can ship.

It does **not**:
- replace the existing SciPy placeholder in
  `rabbit.collisions.nu_nu_scattering` (which is a phenomenological
  ``-G(f - fbar)`` relaxation, not the Dolgov-Hansen-Semikoz
  appendix-A kernel),
- calibrate the absolute matrix-element prefactor against
  Dolgov-Hansen-Semikoz 1997 / Mangano 2005 / de Salas-Pastor 2016,
- wire ν–ν into the full-Boltzmann driver,
- run an FLRW ``N_eff`` lock against the Mangano 2005 ``3.044 ±
  0.005`` benchmark.

Closing those gaps is the follow-up PR-T3C runtime work.

## Algorithmic structure

The new kernel mirrors the algebraic structure of the
Hannestad-Madsen ν-e port in `rabbit.jax.collisions_jax`:

* y3 (outgoing ν) — Gauss-Laguerre on the matched ``q_nodes`` grid.
* y2 (partner ν) — Gauss-Laguerre on the matched ``q_nodes`` grid.
* y4 (other partner) — implicit via energy conservation
  ``y_4 = y_1 + y_2 - y_3`` with a ``y_4 > 0`` guard.
* The integrand uses the symmetric ``(y_1 y_2)^2 + (y_3 y_4)^2``
  matrix-element form; this is **not** the Dolgov-Hansen-Semikoz
  appendix-A coefficient table, but it is symmetric under
  ``(1,2) <-> (3,4)`` and therefore preserves detailed balance and
  energy conservation algebraically pointwise.
* A Fierz-aware species mixing prefactor ``epsilon_alpha_beta``
  (= ``2`` for identical species, ``1`` for distinguishable pairs)
  is applied to the matrix element, capturing the leading
  identical-particle factor.
* The electron Fermi-Dirac in ν-e is replaced by the partner
  neutrino distribution ``f_β`` evaluated via PCHIP cubic Hermite
  interpolation on ``q_nodes``.  At the matched grid the
  evaluations at ``y_2`` and ``y_3`` are identity; the evaluation
  at ``y_4 = y_1 + y_2 - y_3`` is **not** identity (off-grid), so a
  bounded numerical residual remains in the integrand.

## Verification

`tests/test_pr_t3c_nu_nu_preflight.py` (8 tests, all green):

* ``test_nu_nu_detailed_balance_at_fd[N_q in {12, 20}, eps in {1, 2}]``
  — at ``f_α = f_β = f_FD`` with ``T_α = T_β = T_γ``, the integrated
  ``C[f_FD, f_FD]`` collapses to ``< 1e-20`` absolute on the matched
  grid.  Measured: ``~7e-23`` at ``T = 2 MeV``, dominated by the
  PCHIP-on-``y_4`` residual.
* ``test_nu_nu_energy_conservation_at_fd_perturbation[eps in {1, 2}]``
  — on a 50%-above-FD probe applied symmetrically to both species
  the energy moment ``int y^3 C dy`` is bounded at ``< 5%
  relative`` to ``max|C|``.  Measured: ``~1.8% rel``.
* ``test_nu_nu_alpha_eq_beta_factor_2_relative_to_distinguishable``
  — the Fierz coefficient ``epsilon = 2`` doubles the kernel
  relative to ``epsilon = 1`` to ``< 1e-12 rel``.
* ``test_make_nu_nu_kernel_is_deterministic`` — repeated kernel
  builds share LRU-cached quadrature arrays and produce
  bitwise-identical outputs.

Targeted regression bundle
(``test_pr_t3a_collisionless_driver.py +
test_pr_t3b_collision_preflight.py +
test_pr_t3b_jax_operator_parity.py +
test_pr_t3c_nu_nu_preflight.py``): ``52 passed in 188.8 s``
(44 prior + 8 new).

## Adversarial self-audit

Phase-prompt items that this preflight closes:

* **Algebraic structure** of the Fierz-diagonal operator: the
  symmetric matrix-element form preserves detailed balance and
  energy conservation pointwise; the Fierz factor
  ``epsilon_alpha_beta`` is exposed and tested.

Phase-prompt items that remain open:

* **Element-wise SciPy parity at 1e-12** — no SciPy
  Dolgov-Hansen-Semikoz reference exists in the codebase yet (the
  ``rabbit.collisions.nu_nu_scattering`` file is a placeholder
  relaxation operator with different physics, not a candidate for
  parity).  Closing this requires implementing a SciPy
  Dolgov-Hansen-Semikoz reference and porting the JAX kernel to
  match it.
* **Detailed balance at 1e-14** — bounded here by PCHIP
  interpolation noise on ``f_β(y_4)``.  Tightening requires either
  a JAX-native natural cubic spline (matching scipy ``interp1d``
  cubic) or a 4-momentum delta-function quadrature that avoids the
  ``y_4`` interpolation altogether.
* **Energy conservation at 1e-12 rel** — same ``y_4``-interpolation
  bound; same fix.
* **Absolute matrix-element prefactor** — the current kernel uses a
  placeholder ``(y_1 y_2)^2 + (y_3 y_4)^2`` with ``matrix_coeff =
  1``.  The Dolgov-Hansen-Semikoz appendix-A coefficients (and the
  Mangano 2005 cross-checks) are not yet incorporated.
* **FLRW ``N_eff = 3.044 ± 0.005`` lock** — requires driver wiring
  + the corrected matrix-element prefactor + a real Rodas5P
  trajectory.

Adversarial probes:

* **Symmetry**: matrix element ``(y_1 y_2)^2 + (y_3 y_4)^2`` is
  symmetric under ``(1,2) <-> (3,4)``; statistical factor ``f_3 f_4
  (1-f_1)(1-f_2) - f_1 f_2 (1-f_3)(1-f_4)`` flips sign under that
  swap; product is antisymmetric, integrating with the symmetric
  measure gives algebraic detailed balance pointwise at FD.
* **Heaviside cutoff**: ``y_4 > 0`` is enforced via
  ``jnp.where(valid, contrib, 0.0)`` mirroring the ν-e and pair
  ports.
* **Fierz coefficient**: tested via the
  ``epsilon=2`` vs ``epsilon=1`` ratio test at ``< 1e-12 rel``.
* **Reproducibility**: the LRU-cached quadrature builders return
  shared arrays across kernel factories; the determinism test
  guards this contract.

## Total-rate JAX helper (PR-T3C-PF #6)

A new building-block module ``rabbit.jax.collision_rates_jax``
exposes the Mangano / Hannestad-Madsen total weak-collision rate

    Γ_α(T) = (7π/12) · G_F^2 · T^5 · a_α   [MeV]

with ``a_e ≈ 2.353`` (CC + NC) and ``a_x ≈ 0.503`` (NC only)
imported from ``rabbit.collisions.kernels``.  The module exposes:

- ``total_rate_nu_e_jax(T)`` and ``total_rate_nu_x_jax(T)`` —
  JIT-compatible scalar evaluators.
- ``total_rate_jax(T, species)`` — string-dispatched wrapper.
- ``gamma_over_H_jax(T, H, species)`` — equilibration ratio.

This is canonical-PR-T3B building-block infrastructure that the
future asymptotic-preserving (AP-form) collision wrapper will use
to evaluate ``Γ/H`` cleanly without re-deriving it from the full
2D collision integral every RHS call.  The module is **not** yet
wired into the runtime driver — it is strictly additive
infrastructure landed for the canonical work to consume.

The 7-test parity suite ``tests/test_pr_t3c_collision_rates_parity.py``
locks:

- per-temperature element-wise ``< 1e-14 rel`` parity vs SciPy
  ``NuEScatteringOperator.total_rate_all_channels`` for both
  ``nue`` and ``nux`` species across
  ``T ∈ {0.01, 0.1, 1, 10, 100} MeV``;
- the ``T^5`` scaling exponent;
- the species coupling ratio ``Γ_e / Γ_x = a_e / a_x ≈ 4.68``;
- the dimensional consistency of ``Γ/H`` at the canonical
  freeze-out temperature ``T ~ 1 MeV``;
- JIT compatibility of ``gamma_over_H_jax``.

## Diagonal nu-nu PCHIP -> cubic spline swap (PR-T3C-PF #4)

The diagonal ν-ν kernel (`rabbit.jax.nu_nu_scattering_jax.
nu_nu_diagonal_collision_jax`) now uses the JAX-native
not-a-knot cubic spline for all three off-grid distribution
evaluations: ``f_α(y_3)``, ``f_β(y_2)`` and ``f_β(y_4)``.  The
last evaluation is the load-bearing one (``y_4 = y_1+y_2-y_3``
is generically off the input Laguerre grid for any ``(y_1, y_2,
y_3)``); the first two are matched-grid identity at the input
nodes.

Measured impact on the diagonal ν-ν invariants (``T = 2 MeV``,
50%-above-FD perturbation for energy conservation):

| invariant | before (PCHIP) | after (cubic spline) | improvement |
| --- | --- | --- | --- |
| DB at f_FD, N_q=12 | ``~3e-23`` | ``~7e-24`` | ``~5x`` |
| DB at f_FD, N_q=20 | ``~7e-23`` | ``~3e-23`` | ``~2x`` |
| Energy cons rel | ``~1.8e-2`` | ``~2.3e-3`` | ``~8x`` |

Tightened thresholds:

- DB at f_FD: ``< 1e-22`` (was ``< 1e-20``)
- Energy conservation rel: ``< 5e-3`` (was ``< 5e-2``)

Closing further requires removing the off-grid ``f_β(y_4)``
evaluation entirely (e.g., via a 4-momentum delta-function
quadrature that constrains ``y_4`` onto the input grid).

## Pair-process PCHIP -> cubic spline swap (PR-T3C-PF #3)

The pure-JAX ``pair_collision_jax`` operator
(`rabbit.jax.collisions_jax`) now uses the not-a-knot natural
cubic spline from ``rabbit.jax.cubic_spline_jax`` for ``f_ν̄(y_2)``
off-grid evaluation, with scipy-compatible boundary clamping
(``fill_low = f_ν̄[0]``, ``fill_high = 0`` mirroring the SciPy
reference's ``fill_value=(f_nubar[0], 0.0)`` argument to
``scipy.interpolate.interp1d``).

Measured impact on the off-grid configuration
(``N_q = 20``, ``N_quad = 24``, ``T = 2 MeV``, ``f = 1.5 * FD``):

| metric | before (PCHIP) | after (cubic spline) |
| --- | --- | --- |
| element-wise rel err vs SciPy | ``~8.3e-2`` | ``~6.5e-16`` |
| element-wise abs err vs SciPy | ``~7.7e-21`` | ``~6.0e-35`` |

The off-grid pair parity test
(``test_pair_jax_off_grid_matches_scipy_cubic`` —
formerly ``test_pair_jax_off_grid_pchip_close_to_scipy_cubic``)
is now locked at ``1e-12 rel``, a ``~14`` orders-of-magnitude
tightening of the previous ``10%`` preflight bound.

Shared-grid parity tests, detailed-balance locks and JIT
determinism tests are unchanged and remain green at their
original ``< 1e-30`` absolute / ``< 1e-12`` rel bounds.

## Cubic spline infrastructure (PR-T3C-PF #2)

A JAX-native not-a-knot natural cubic spline is now landed at
``rabbit.jax.cubic_spline_jax`` and locked at element-wise machine
precision against ``scipy.interpolate.interp1d(kind='cubic')``:

- ``test_cubic_spline_identity_at_nodes`` — evaluating the spline
  at its own input nodes returns the input ``y`` exactly to
  ``< 1e-12`` for grid sizes ``n in {4, 6, 8, 12, 20}``.
- ``test_cubic_spline_matches_scipy_interp1d_cubic`` — on a smooth
  FD distribution sampled at Gauss-Laguerre nodes, the spline
  matches ``scipy.interpolate.interp1d(kind='cubic')`` at
  ``< 1e-12`` element-wise on a mix of midpoint and randomly-
  sampled query points (``n in {6, 8, 12, 20}``).
- ``test_cubic_spline_C2_smoothness`` — the spline is ``C^2`` at
  internal nodes (finite-difference second derivative bounded).
- ``test_cubic_spline_jit_compatible`` — eager and JIT-compiled
  evaluation agree to machine precision.

This module is the structural prerequisite for replacing the
existing PCHIP cubic Hermite interpolant in
``rabbit.jax.q_advection_jax`` (used by both ``collisions_jax`` for
the pair-process ``f_ν̄(y_2)`` and ``nu_nu_scattering_jax`` for the
``f_β(y_2)`` and ``f_β(y_4)`` evaluations).  Once the swap lands,
the off-grid PCHIP-vs-cubic gap (``~8% rel`` on pair-process
parity, ``~7e-23`` on diagonal ν-ν detailed balance, ``~2% rel``
on ν-ν energy conservation) should tighten to floating-point
reduction order, closing several of the remaining preflight gates
without changing any physics.

The spline module itself is strictly additive and does not yet
modify the existing pair-process or diagonal ν-ν kernels — that
is the next bounded preflight slice.

## Verdict

Conditional pass.

The structural skeleton for diagonal ν–ν scattering is now landed in
JAX with documented detailed-balance and energy-conservation
invariants on the matched Laguerre grid; the JAX-native cubic
spline replacement infrastructure is also in place.  The kernel is
**not** production-ready: the absolute matrix-element prefactor,
the PCHIP-to-cubic-spline swap inside the operators, strict SciPy
parity on the swapped operators, and FLRW ``N_eff`` calibration
all remain open.

What remains for the full PR-T3C runtime patch:

* implement (or port) a Dolgov-Hansen-Semikoz appendix-A SciPy
  reference for the diagonal ν–ν kernel,
* replace the placeholder ``(y_1 y_2)^2 + (y_3 y_4)^2`` matrix
  element with the appendix-A coefficient table,
* tighten detailed balance and energy conservation either via a
  JAX-native natural cubic spline replacement for PCHIP or via a
  4-momentum delta-function quadrature that evaluates ``f_β`` only
  at on-grid points,
* wire the kernel through the bank-core dispatcher as a new
  private ``collision_mode="nu_nu_preflight"`` and run an
  end-to-end Rodas5P smoke,
* lock the FLRW ``N_eff = 3.044 ± 0.005`` target against Mangano
  2005 / Froustey 2020.
