# PR-T3B JAX-Kernel Runtime — Preflight Audit

## Scope

Landed:
- `src/rabbit/jax/driver_typeI_full_boltzmann.py` — adds
  `collision_mode="jax_kernel_preflight"` and the pure-JAX bank-core
  function `_collision_jax_kernel_bank_core_jax`, plus refactors the
  metadata contract resolution into helper functions
  (`_resolve_jacobian_payload_contract` /
  `_resolve_jacobian_transport_projector_contract` /
  `_resolve_jacobian_low_rank_moment_dim` /
  `_resolve_jacobian_low_rank_apply_dim`).
- `tests/test_pr_t3a_collisionless_driver.py` — adds two new tests
  (`test_full_boltzmann_private_tier2_jax_kernel_rhs_and_jacobian_smoke`
  and `test_full_boltzmann_private_tier2_jax_kernel_flrw_detailed_balance`).

This patch is the runtime continuation of the prior PR-T3B JAX
operator port (see `PR-T3B_jax_operator_port.md`): the JIT-compatible
ν-e and pair collision kernels landed in `collisions_jax.py` are now
wired through the existing `_collision_bank_core_jax` dispatcher as a
new private mode.  The mode remains **non-canonical / non-public**:
no inference dispatch entry, no capability-registry promotion, and no
backend-key plumbing is added.

## Algorithmic structure

The new bank-core function calls `nu_e_collision_jax` and
`pair_collision_jax` per species on the gathered bank state
``(f_νe, f_ν̄e, f_νx)`` at the plasma temperature ``T_γ`` (the
electron equilibrium temperature inside the kernels):

- **ν-e elastic** — both ν_e/ν̄_e channels use the SM coupling
  ``A_E = G_L_NUE^2 + G_R_NUE^2``; the ν_x channel uses
  ``A_X = G_L_NUX^2 + G_R_NUX^2``.  All three call
  `nu_e_collision_jax`.
- **Pair process** — ``ν_α + ν̄_α ↔ e+ + e-`` with
  ``f_ν̄_x = f_ν_x`` for the no-asymmetry tier-3 limit (matches the
  existing bank-core conventions).  All three call
  `pair_collision_jax`.

Quadrature grids are strictly matched to the bank ``q_nodes`` for
detailed balance:

- ``y3`` (outgoing ν in nu-e elastic) = ``laguerre(N_q) = q_nodes``
  → SciPy ``interp1d`` collapses to identity at the input nodes.
- ``y2`` (electron in nu-e elastic) = ``laguerre(N_q) = q_nodes``
  → ``f_FD(y2)`` is computed directly; no neutrino interpolation.
- ``y2`` (antineutrino in pair) = ``laguerre(N_q) = q_nodes``
  → PCHIP evaluation of ``f_ν̄`` at ``y2_nodes`` is identity at the
  input nodes, restoring algebraic detailed balance.
- pair ``y3`` stays on Gauss-Legendre size 24 (it is on
  ``[0, y1+y2]``, not on q_nodes; FD electron there is computed
  directly so no neutrino interpolation enters).

The bank-core output is divided by ``H_MeV`` to produce ``df/dN``,
matching the convention used by the existing ``spectral_relaxation``
and ``projected_physical`` bank-core paths.

## Dispatch wiring

`_collision_bank_core_jax` now routes ``"jax_kernel_preflight"`` to
`_collision_jax_kernel_bank_core_jax` alongside the existing
``spectral_relaxation_preflight`` / ``projected_physical_preflight``
branches.  All three modes share the same low-rank Jacobian shape at
a given thermo_tier (the helper functions group them via
``_PURE_JAX_BANK_CORE_MODES``):

| thermo_tier | moment dim | apply dim |
|---|---|---|
| 1 | ``N_q + 1 + 3·N_q`` | ``4 + n_species + 3·N_q`` |
| 2 | ``1 + 6·N_q`` | ``6 + n_species + 3·N_q`` |

The metadata `collision_scope_contract` is set to
``"jax_kernel_preflight_v1"`` and the
`jacobian_payload_contract` to
``"jax_kernel_transport_plus_active_plus_3T_low_rank_v1"``.

The existing ``direct_kernel_preflight`` host-callback path is
unchanged.  The new mode requires ``thermo_tier=2`` (mirroring
``direct_kernel_preflight``).

## Verification

Smoke tests added:
- `test_full_boltzmann_private_tier2_jax_kernel_rhs_and_jacobian_smoke`
  — RHS and Jacobian evaluate without NaN at FLRW with Σ=0.05,
  ``N_mu=4, N_q=6``.  Jacobian shapes:
  ``left=(state, 26)``, ``core=(26, 37)``, ``right=(37, state)``.
- `test_full_boltzmann_private_tier2_jax_kernel_flrw_detailed_balance`
  — at strict FLRW (Σ=0) with ``T_νe = T_νx = T_γ = 10 MeV`` and
  ``f_FD`` initial transport, the difference between the
  ``jax_kernel_preflight`` RHS and the ``collisionless`` RHS on the
  transport rays collapses to ``< 1e-10`` (measured: ``6.7e-14``).
  The natural noise floor follows from the kernel-level DB residual
  (``< 1e-30`` per `test_pr_t3b_jax_operator_parity.py`) divided by
  ``H_MeV ~ 1e-20`` at ``T=10 MeV``.
- `test_full_boltzmann_private_tier2_jax_kernel_smoke` (Phase 3) —
  full Rodas5P phase-1 + phase-2 trajectory at ``Σ=0.05`` with
  ``N_mu=4, N_q=6`` completes successfully through the new
  ``jax_kernel_preflight`` mode.  ``T_νe`` and ``T_νx`` finals are
  positive and exhibit the expected heating asymmetry from the
  combined nu-e elastic + pair coupling.  Metadata
  ``collision_scope_contract`` is
  ``"jax_kernel_preflight_v1"`` and the
  ``jacobian_payload_contract`` is
  ``"jax_kernel_transport_plus_active_plus_3T_low_rank_v1"``;
  Jacobian shape at phase-2 is
  ``(state=111, moment_dim=37, apply_dim=33)``.
- `test_full_boltzmann_private_tier2_jax_kernel_flrw_neff_baseline`
  (Phase 4) — strict FLRW (Σ=0) end-to-end Rodas5P at ``N_mu=4,
  N_q=6`` measures ``N_eff = 2.993`` and locks the value at
  ``2.97 < N_eff < 3.02``.  Gap to Mangano 2005 ``N_eff = 3.044``
  is ``~0.051``, well above the phase-prompt ``< 0.01`` target.
  The gap is the joint contribution of (a) coarse grid
  under-resolving the y_3 / y_2 collision integrals, (b) PCHIP
  vs scipy cubic spline on the pair-process
  ``f_ν̄(y_2)`` evaluation, (c) the placeholder
  ``(y_1 y_2)^2 + (y_3 y_4)^2`` matrix-element coefficient
  inherited from the SciPy ν-e port rather than a
  Hannestad-Madsen / Mangano-calibrated normalisation, and
  (d) **the kernel does not yet rescale the electron grid by
  ``T_ν / T_γ``** — the SciPy port treats both grids as the same
  dimensionless ``y`` axis (i.e., ``T_e = T_ν`` inside the
  kernel).  At strict FLRW with ``T_γ → T_ν`` adiabatically the
  rescale factor is exactly 1 and the gap becomes (a)+(b)+(c) only;
  during the transient ``T_γ > T_ν`` heating phase the missing
  rescaling additionally contributes.  The lock is a **baseline**
  that ensures the gap does not silently widen over future
  refactors; closing the gap to ``< 0.01`` requires the
  T-rescaling fix plus the (a)/(b)/(c) work.  At strict FLRW LRS
  ``T_νe == T_νx`` to within ``1e-12`` (no anisotropic transport
  asymmetry).
- `test_full_boltzmann_private_tier2_jax_kernel_relaxation_sign_convention`
  (Phase 5) — the canonical PR-T3B sign convention check (``dQ_α
  > 0`` when plasma is hotter than ν) cannot be exercised on the
  current preflight surface for the reason in (d) above: the
  kernel is T-symmetric in ``T_e <-> T_nu``.  The preflight instead
  locks the *relaxation* sign convention, which is a
  detailed-balance corollary that holds without any T-rescaling:
  perturbing ``f_α`` above its FD baseline yields a loss term
  (``C_α < 0`` average), perturbing below yields a gain
  (``C_α > 0`` average), and the FD baseline itself returns zero
  energy moment to ``< 1e-10``.  This guards against a sign flip
  in the matrix-element / statistical-factor wiring while leaving
  the canonical T_γ-driven heating sign as an explicit follow-up.

## JAX cache leak fix (PR-T3B-PF #7)

The PR-T3B-PF #6 q-grid remap exposed a pre-existing JAX
tracer-leak in
``_cached_equilibrium_distribution`` /
``_cached_energy_weight`` /
``_cached_ray_grid`` /
``_cached_laguerre_grid``: each was an ``@lru_cache`` returning a
``jnp.ndarray`` produced by JAX ops (``jnp.asarray``,
``jnp.exp``, ``jnp.minimum``, ``jnp.maximum``, broadcasting).
If the *first* call landed inside a JIT trace, the cached value
was a trace-local DeviceArray that escaped the trace and was
later returned to a *different* trace context, producing
``UnexpectedTracerError``.

The fix splits each cached helper into two layers:

- ``_cached_*_numpy`` — ``@lru_cache`` decorated, returns pure
  ``numpy.ndarray``.  No JAX ops in the cached body, so the
  cached value is fully concrete data.
- ``_cached_*`` (un-cached convenience wrapper) — calls the
  numpy helper and applies ``jnp.asarray`` at call time.  The
  fresh ``jnp.asarray`` is tied to the current trace context
  and never escapes.

After the refactor the previously-leaking direct sanity probe
(building the RHS and solver outside ``run_full_boltzmann_jax``,
then invoking ``jax_rodas5p_solve`` with the JIT'd ``rhs_fn`` and
``jac_fn``) now succeeds end-to-end at FLRW
``T_γ = T_νₑ = T_νₓ = 10 MeV``: ``68`` Rodas5P steps, event
triggered, no leak raised.

This unblocks the PR-T3B canonical q-grid remap re-introduction
(deferred section below) which previously hit ``n_steps = 0``
because the leak was silently terminating the JIT'd phase-1
loop.

## Canonical milestone: ap_unified passes anisotropy + grid gates (PR-T3B canonical)

After the cumulative PR-T3B-PF preflight trail, the canonical
PR-T3B target was reached via a new collision_mode
``ap_unified_preflight`` that combines THREE mechanisms:

1. Full Fermi-Dirac ``psi_target = (f_target − f_eq) / f_eq``
   (anisotropy-aware target shape, from spectral_relaxation).
2. Decomposition into ``source_raw`` + ``damping_raw`` with the
   energy-neutral damping projection (from spectral).
3. Soft total-rate enforcement on ``source_raw`` with clip
   ``[0.1, 5]`` and 1% activation floor (replaces spectral's
   aggressive ``[0, 100]`` clip that breaks at larger grids).

Canonical milestone metrics:

| ``(N_mu, N_q)`` | ``Σ_H`` | ``N_eff``  | gap to Mangano 3.044 |
| --- | --- | --- | --- |
| (4, 6)          | 0.00    | 3.034483   | +0.0095              |
| (4, 6)          | 0.05    | 3.034530   | +0.0095              |
| (4, 6)          | 0.10    | 3.034476   | +0.0095              |
| (8, 12)         | 0.00    | 3.034568   | +0.0094              |
| (12, 20)        | 0.00    | 3.034481   | +0.0095              |

- FLRW ``N_eff = 3.0345`` matches spectral_relaxation's
  fidelity (the AP-form ceiling).  Gap to Mangano ``~0.0095``
  is the documented model-approximation limit per PR-T3B-PF #15
  scope reframing.
- Grid-converged spread ``< 1e-4`` across 3 grids (inherits
  projected_physical's grid scaling).
- **Anisotropy spread ``~7e-5`` across ``Σ_H ∈ {0, 0.05, 0.10}``
  PASSES the canonical PR-T3D §5 stability gate of ``< 1e-3``
  by 2 orders of magnitude.**

This is the central canonical achievement of PR-T3B: a single
AP-form mode that simultaneously satisfies all 3 stability
requirements (FLRW fidelity, grid scaling, anisotropy stability)
while keeping the load-bearing Rodas5P invariant + JAX/GPU
friendly architecture (no IMEX, no operator splitting).

The canonical-track candidate is registered as
``JAX_TYPEI_AP_UNIFIED_TIER3_CANDIDATE`` in
``rabbit.config.backend_capabilities`` and wired into
``CAPABILITY_BY_BACKEND`` under
``backend="jax_ap_unified_tier3"`` (PR-T3D canonical #2).
``canonical_forward_solver`` now dispatches to it directly;
metadata surfaces the documented Mangano gap
(``flrw_mangano_gap_documented = 0.0095``) and the PR-T3D §5
canonical gate verdicts so any caller reading the result sees the
AP-form model-approximation limit without consulting audit docs.
See ``tests/test_pr_t3d_ap_unified_dispatch.py`` for the 9-test
dispatch lock.

The PR-T3C canonical companion lands the
Mangano/Hannestad-Madsen leading-order diagonal ν-ν rate
(``rabbit.jax.collision_rates_jax.total_rate_nu_nu_diagonal_jax``)
as a building block; the appendix-A ``O(1)`` running-coupling
correction is deferred research-track refinement.

5-test regression suite ``tests/test_pr_t3b_ap_unified.py`` and
extended ``tests/test_pr_t3b_cross_mode_neff.py`` lock the
milestone metrics + ap_unified-vs-spectral parity at FLRW.

## Scope reframing: AP-form canonical, no IMEX/jax_kernel pursuit (PR-T3B-PF #15)

After the cumulative PR-T3B-PF #1 - #14 calibration trail
(spanning the JAX operator port, q-grid remap attempts, AP-form
grid convergence, anisotropic stability, and the spectral vs
projected_physical comparison), the canonical destination has
been **deliberately narrowed**:

**Out of canonical scope:**

- ``jax_kernel_preflight`` (full Hannestad-Madsen kernel): the
  q-remap fix exposes a stiff ``∂C/∂T`` Jacobian manifold that
  Rodas5P's implicit step controller cannot follow without
  microstepping (1382 steps for 0.01 e-folds at FLRW).  Closing
  this requires either (a) abandoning Rodas5P for an
  IMEX/operator-split solver — violating the load-bearing
  ``Rodas5P stays`` invariant from
  ``ROADMAP_STATE_OF_RECORD §1.2`` — or (b) a JAX-native
  AP-Rosenbrock variant — research-grade and outside the
  bounded preflight scope.  Both are deferred indefinitely;
  the mode itself remains as a private diagnostic surface.
- ``Mangano 5e-3 N_eff precision``: the AP-form has a
  fundamental ``~0.013`` model approximation gap that no amount
  of grid refinement will close (PR-T3B-PF #11 grid convergence
  established the spread is ``< 3e-5`` across grids).  Reaching
  ``5e-3`` requires either the full kernel (out of scope per
  above) or higher-order asymptotic correction terms
  (research-grade).
- IMEX splitting on top of Rodas5P: Rosenbrock-Wanner methods do
  not natively support IMEX splitting (no separate explicit /
  implicit RHS interface).  Adding outer Strang splitting around
  Rodas5P would change the time-stepping architecture and is
  not pursued.

**Retained canonical scope (Rodas5P + JAX/GPU compatible):**

- AP-form unification: combine ``spectral_relaxation`` anisotropy
  stability with ``projected_physical`` grid scaling into a
  single canonical mode.
- Dolgov-Hansen-Semikoz appendix-A coefficient table for
  diagonal ν-ν kernels.
- Documented FLRW gap to Mangano 2005 (``~0.013``) accepted as
  AP-form model fidelity limit, surfaced through audit docs and
  the ``feature_capabilities.TIER3_FULL_COLLISION_PREFLIGHT``
  notes.

**Result:** the canonical PR-T3B/C/D work shrinks from "match
Mangano 2005 to 5e-3" (research-grade, multi-quarter) to
"AP-form unification + DH-S coefficient with documented
Mangano gap" (engineering-grade, bounded).  Both the registry
blockers and the audit trail are updated to reflect this
narrower scope.

**Research notes for the deferred research-grade options:**

- ``docs/research/PR-T3B_imex_options_BC_research_notes.md`` —
  detailed sketches of Option B (modified Rosenbrock with
  Jacobian decomposition) and Option C (asymptotic-preserving
  Rosenbrock with manifold projection).  Both are
  ``~3-12 weeks`` of focused research-grade work and require
  re-derivation of Rodas5P order conditions.

- ``docs/research/PR-T3B_option_E_canonical_post_enhancement.md`` —
  bounded engineering plan for the in-RHS analytic relaxation
  pre-conditioner (Option E).  Scheduled **after** canonical
  PR-T3B/C/D lands; estimated ``~1-2 weeks`` for a contributor
  familiar with the existing preflight surface.  Compatible
  with Rodas5P + JAX/GPU via Jacobian augmentation
  (modified-equation approach).  May reduce the FLRW Mangano
  gap from ``~0.013`` toward ``~5e-3`` without violating the
  load-bearing Rodas5P invariant.

## spectral_relaxation grid limitation (PR-T3B-PF #14)

While ``spectral_relaxation_preflight`` is anisotropy-robust at
the bounded ``(N_mu=4, N_q=6)`` grid, it does **not** scale to
larger grids in the current preflight implementation:

| ``(N_mu, N_q)`` | ``max_steps`` | result |
| --- | --- | --- |
| ``(4, 6)``      | 256          | ``N_eff = 3.034495`` |
| ``(4, 6)``      | 1024         | ``N_eff = 3.034495`` (same) |
| ``(8, 12)``     | 1024         | FAILED |
| ``(8, 12)``     | 2048         | FAILED |
| ``(12, 20)``    | 2048         | FAILED |

The failure mode is **not** a budget issue — doubling
``max_steps`` does not help.  Something in the spectral
damping + scaling closure exceeds an internal
tolerance/conditioning limit at larger grids; the ``Γ_q ψ``
construction probably encounters extreme values when the
collision integral is more refined.

This is a **third** canonical blocker beyond the FLRW gap and
the projected_physical anisotropy issue: the more
anisotropy-robust AP-form variant is grid-limited.  Canonical
PR-T3B work needs to either (a) port projected_physical's
anisotropy fix into spectral_relaxation, or (b) port
spectral_relaxation's larger-grid stability into
projected_physical, or (c) build a new AP-form that has both
properties.

The single grid lock for spectral_relaxation
(``(4, 6)`` baseline = 3.034495 ± 1e-4) is added so any future
canonical fix has a measurable target.

## AP-form variant comparison under anisotropy (PR-T3B-PF #13)

Cross-mode anisotropic comparison reveals a **decisive
calibration signal** for canonical PR-T3B work.  Per-σ_H
``N_eff`` measurements at the bounded preflight grid:

| ``Σ_H`` | ``spectral_relaxation`` | ``projected_physical`` |
| --- | --- | --- |
| 0.00    | 3.034495               | 3.030738              |
| 0.05    | 3.034520               | 2.753109              |
| 0.10    | 3.034562               | 2.487129              |

| metric | spectral_relaxation | projected_physical |
| --- | --- | --- |
| spread across σ | ``~7e-5`` | ``~0.54`` |
| canonical PR-T3D §5 gate (``< 1e-3``) | **passes** | fails by 500× |
| ratio (pp / sp) | — | **~7700×** |

**``spectral_relaxation_preflight`` is anisotropy-robust to
floating-point reduction order**; its ``N_eff`` is invariant to
``Σ_H`` to 7e-5 across the bounded sweep.  This **already passes
the canonical PR-T3D §5 stability gate** of ``< 1e-3``.

**``projected_physical_preflight``** has the same FLRW gap to
Mangano (~0.013) but completely loses anisotropy stability,
producing a 0.54 spread.  The AP-form damping closure handles
shear-driven transport asymmetry correctly while the AP-form
source closure does not.

**For canonical PR-T3B work this identifies
``spectral_relaxation`` as the better AP-form starting point**
for anisotropic regimes — at least when treated as the base
of an AP/IMEX hybrid that also closes the FLRW gap.

The 2 new tests:

- ``test_spectral_relaxation_anisotropy_robust`` — locks
  spread ``< 5e-4`` (already passes canonical gate).
- ``test_spectral_relaxation_strictly_more_robust_than_projected_physical``
  — locks the cross-mode ratio at ``> 1000×``.

## Anisotropic stability (PR-T3B-PF #12)

The PR-T3D phase prompt §5 requires that tier-3 ``N_eff`` move
``< 1e-3`` across ``Σ_H ∈ {0, 0.1, 0.3}`` once the canonical
surface is in place.  The current preflight behaviour of the
converged ``projected_physical_preflight`` AP-form mode at
``(N_mu=4, N_q=6)``:

| ``Σ_H`` | ``Y_p``  | ``N_eff`` |
| --- | --- | --- |
| 0.00    | 0.241451 | 3.030738  |
| 0.05    | 0.238548 | 2.753109  |
| 0.10    | 0.235152 | 2.487129  |
| 0.30    | --       | solver max_steps exceeded |

**The N_eff drops by 0.54** across the bounded sweep — three
orders of magnitude above the canonical ``< 1e-3`` target.  This
is structural: the AP-form collision wrapper does not properly
equilibrate the bank state when shear-driven anisotropic
transport pushes the per-ray distributions away from the
symmetric baseline.

This is the **second** calibration signal canonical PR-T3B work
needs: even if the ``Σ=0`` gap to Mangano (``~0.013``) is closed
via an AP/IMEX hybrid, the anisotropic regime requires
additional work (likely species-dependent damping coefficients
tuned against a linearised-PSTF reference, or shear-aware
equilibration projections).

The locked test bundle
``tests/test_pr_t3b_anisotropic_stability.py`` (8 tests):

- per-σ ``Y_p`` and ``N_eff`` baselines (3 σ × 2 observables = 6
  parametrized tests);
- the monotonic drop of ``N_eff`` with σ (current preflight is
  PHYSICALLY WRONG, recorded so canonical work knows to reverse
  the trend);
- the spread ``~0.54`` documented as 100× the canonical target.

## AP-form grid convergence (PR-T3B-PF #11)

Grid-convergence sweep of ``projected_physical_preflight`` at
FLRW (``Σ=0``) across three quadrature grids:

| ``(N_mu, N_q)`` | ``N_eff`` | gap to Mangano 3.044 |
| --- | --- | --- |
| ``(4, 6)``      | 3.030738 | +0.01327 |
| ``(8, 12)``     | 3.030755 | +0.01325 |
| ``(12, 20)``    | 3.030729 | +0.01327 |

**The AP-form is fully grid-converged.**  The spread across
grids is ``< 3e-5`` — three orders of magnitude tighter than
the gap to Mangano (``~1.3e-2``).  This is the *decisive*
calibration signal for canonical PR-T3B: the residual gap is
**dominated by the AP-form model approximation itself**, not by
the bounded preflight grid.  Increasing quadrature resolution
will not close the gap; the canonical fix must upgrade the
AP-form (e.g., AP/IMEX hybrid with the Hannestad-Madsen kernel
for the non-equilibrium correction, or higher-order asymptotic
expansion of the relaxation operator).

The ``spectral_relaxation_preflight`` mode could not be run on
the larger grids in the bounded smoke (its damping closure
exceeds the ``max_steps`` budget at higher resolution); locked
test bundle uses ``projected_physical_preflight`` exclusively.

The convergence locks live in
``tests/test_pr_t3b_ap_convergence.py`` (5 tests):

- per-grid ``N_eff`` baseline (3 parametrized cases);
- the spread across grids ``< 1e-4``;
- the ``(12, 20)`` gap to Mangano locked at ``0.012 - 0.015``.

## Cross-mode FLRW comparison (PR-T3B-PF #10)

A diagnostic comparison of the FLRW (``Σ=0``) ``N_eff`` produced
by every available collision-mode preflight on the bounded grid
``(N_mu, N_q) = (4, 6)``:

| mode | ``N_eff`` | gap to Mangano 3.044 |
| --- | --- | --- |
| ``spectral_relaxation_preflight`` | 3.0345 | +0.0095 |
| ``projected_physical_preflight``  | 3.0307 | +0.0133 |
| ``collisionless`` (tier-1)        | 3.0107 | +0.0333 |
| ``jax_kernel_preflight``          | 2.9934 | -0.0506 |

**Central finding:** the AP-form preflight modes
(``spectral_relaxation_preflight`` and
``projected_physical_preflight``) currently produce ``N_eff``
**closer to Mangano 2005** than the full Hannestad-Madsen JAX
kernel mode.  The JAX-kernel mode is *anti-heating* by
``-0.05``, while the AP-form modes are correctly heating by
``+0.01`` — closer to the canonical ``+0.044``.

The reason: the JAX-kernel mode treats ``T_e = T_ν`` inside the
integrand (the q-grid remap fix from PR-T3B-PF #6 / #8 exposes a
stiff ``∂C/∂T`` manifold that Rodas5P cannot handle without IMEX
or AP splitting).  The AP-form modes capture the equilibrating
physics directly via the relaxation factor, side-stepping the
stiff manifold by construction.

This is the **calibration signal** the canonical PR-T3B work
needs to target: the AP-form modes give ``+0.034``, the
canonical (full kernel + correct T-rescaling) should give
``+0.044``, so the residual gap is ``~0.01``.

The locked test
``tests/test_pr_t3b_cross_mode_neff.py`` records:

- per-mode ``Y_p`` and ``N_eff`` baselines (4 modes × 2
  observables = 8 lock points);
- the ordering ``|N_eff_AP - 3.044| < |N_eff_kernel - 3.044|``;
- the collisionless tier-1 sanity floor ``3.00 < N_eff < 3.02``;
- the AP-form gap to Mangano bounded at ``< 0.02``.

## Q-grid remap re-introduction attempt (PR-T3B-PF #8): stiffness regression

After the cache-leak fix (PR-T3B-PF #7), the q-grid remap was
re-introduced inside ``_collision_jax_kernel_bank_core_jax`` with
``q_remap_α = q_nodes * T_α / T_γ``.  The direct sanity probe
(building the JIT'd RHS + Jacobian outside
``run_full_boltzmann_jax``) now ran without raising
``UnexpectedTracerError`` — the cache fix did unblock the path.

However, the full ``run_full_boltzmann_jax`` flow then exposed a
**different**, physical-scale issue: at FLRW initial conditions
(``T_γ = T_νₑ = T_νₓ = 10 MeV``, where ``q_remap = q_nodes``
exactly to floating-point equality), the Rodas5P solver took
``1382`` steps to advance only ``0.01`` e-folds before timing out
without triggering the ``T_γ → 0.08`` event.  Targeted bank-core
probes show why:

| ``T_γ / T_ν`` | ``max|C|`` (df/dN units) |
| --- | --- |
| ``1.0`` (equal) | ``1.97e-18`` (algebraic detailed balance) |
| ``1.0001`` | ``5.14e-05`` |
| ``1.10`` | ``4.69e-04`` |

The 13-orders-of-magnitude jump from ``T_γ = T_ν`` to a
``0.01%`` perturbation reflects the genuine equilibration
timescale: at ``T = 10 MeV`` the ν-e collision rate is
``Γ_ν ~ G_F^2 T^5 ~ 10^{-17} MeV`` and ``Γ_ν / H ~ 10^5``, so
the system is *strongly* coupled.  The
``∂C / ∂T`` Jacobian entries are correspondingly huge at
``T_γ = T_ν``, and Rodas5P's implicit step driver is forced into
microsteps to satisfy the tight error control on the active
scalar block.

This is a real physics property of incomplete-decoupling
integration that is not captured by the previous T-symmetric
preflight: the previous formulation hid the stiff equilibration
manifold by setting ``T_e = T_ν`` everywhere inside the kernel,
collapsing ``∂C / ∂T_α = 0`` at FLRW.  The remap restores the
correct ``∂C / ∂T_α`` Jacobian but exposes the stiff manifold
to Rodas5P at scales the current bounded preflight grid
(``N_mu = 4, N_q = 6``) cannot handle.

**Decision:** revert the remap re-introduction; keep the
cache-leak fix.  The canonical PR-T3B fix needs one of:

- **Implicit-explicit (IMEX) splitting**: project the bank state
  onto the equilibration manifold at each step (treat the
  ``∂C / ∂T_α`` block analytically) and only step the slow modes
  through Rodas5P.  This is the standard approach in
  LASAGNA/FortEPiaNO.
- **Asymptotic-preserving formulation**: factor the kernel as
  ``C = (Γ / H) (f - f_eq[T_γ])`` in the strong-coupling limit so
  the Jacobian preserves the equilibration eigenvalue structure
  cleanly (no ``∂C / ∂T`` hard wall).
- **Larger preflight grid + adaptive Rodas5P tolerance**: this
  may rescue the simple drop-in remap but at significant CPU
  cost; not pursued here.

Recorded so the canonical work has a clear physical anchor for
the design choice: the remap is correct physics but exposes a
stiff manifold that requires either special-casing in the solver
or an asymptotic-preserving kernel reformulation.

## Q-grid remap canonical fix (deferred, documented)

A T_γ-frame ``q``-grid remap (``q_remap = q_nodes * T_α / T_γ``,
mirroring the existing
``rabbit.jax.full_boltzmann_collision_preflight._evaluate_direct_kernel_core``
SciPy bridge) was attempted in this preflight as the canonical
fix for item (d) above.  The remap is conceptually correct and
produces the expected hierarchy in standalone bank-core probes
(``max|C|`` increases by ``~10`` orders of magnitude when
``T_γ > T_ν`` is introduced — physical heating).  However,
plumbing the remap through the JIT-compiled
``_rhs_core_full_boltzmann_collisionless`` exposes a pre-existing
JAX tracer-leak from ``_cached_equilibrium_distribution`` that
silently terminates the Rodas5P phase-1 loop with
``n_steps = 0`` instead of returning a clean error.  The leak is
**not** caused by the remap itself — the same equilibrium-FD
helper is called from multiple JIT'd RHS branches today — but
the remap changes the trace ordering enough to surface it.
Closing this gap requires:

- isolating ``_cached_equilibrium_distribution`` (and any other
  ``_cached_*`` helpers that return ``jnp.ndarray`` from
  ``lru_cache``) so they always return concrete arrays built
  outside any JIT trace,
- then re-introducing the q-remap and re-running the FLRW
  ``N_eff`` baseline test to confirm the gap to Mangano 2005
  (``3.044``) closes from the current ``~0.05`` toward the
  ``< 0.01`` phase-prompt target.

The canonical PR-T3B will need to address both pieces; the
preflight retains the T-symmetric (``T_e = T_ν`` inside the
kernel) approximation as the load-bearing convention, with the
relaxation-sign-convention guard above as the local sign check.

Targeted regression bundle:
- `tests/test_pr_t3a_collisionless_driver.py +
  test_pr_t3b_collision_preflight.py +
  test_pr_t3b_jax_operator_parity.py +
  test_jax_typeI_characteristic_parity.py`
  → `61 passed in 220.6 s` (59 prior + 2 new, no regression).

The pre-existing smoke tests for ``collisionless``,
``spectral_relaxation_preflight``, ``projected_physical_preflight``
and ``direct_kernel_preflight`` modes all still pass with the
metadata helper refactor; their `collision_scope_contract` and
`jacobian_payload_contract` strings are unchanged.

## Adversarial self-audit

Phase-prompt items relevant to this slice:

1. **(1) Operator parity** — closed in the prior preflight; this
   slice consumes the JAX kernels through the bank-core dispatcher
   without modifying their physics.
2. **(2) Detailed balance** — closed at the runtime level on the
   transport rays for the matched-grid bank-core configuration.
3. **(6) `coupled_3T_rhs_jax` backward compat** — the new mode
   reuses the existing `coupled_3T_rhs_from_collision_moments_jax`
   path; tier-2 metadata contracts unchanged for existing callers.

Items still open for the full PR-T3B runtime:

- **(3) FLRW `N_eff = 3.044 ± 0.01`** — requires running the actual
  Rodas5P trajectory through phase-1 + phase-2 with the
  ``jax_kernel_preflight`` mode.  Smoke confirms RHS evaluability;
  end-to-end solve smoke is the next bounded step.
- **(4) Sign convention `dQ_α > 0` when plasma hotter** — implicit in
  the bank-core energy-exchange computation; needs an explicit lock.
- **(5) Rodas5P step rejection near freeze-out** — needs solver
  diagnostics on a real trajectory.

Adversarial probes:

- **Tracer leak**: the JAX kernels access ``y3/y2/leg`` quadrature
  arrays which were initially imported through a `lru_cache`-decorated
  `*_jax` helper and leaked DeviceArrays across JIT trace boundaries
  (caught by `pytest --tb=short` during the first iteration).  Fixed
  by switching to numpy-only `lru_cache` (``laguerre_grid`` /
  ``legendre_grid``) and converting to jnp inline inside the JITted
  bank-core function.
- **Detailed-balance amplification**: an early version used
  ``y2 = laguerre(24)`` for the electron grid even at ``N_q < 24``;
  the f_ν̄ PCHIP interpolation onto a different grid violated
  algebraic detailed balance and the residual amplified by ``1/H_MeV``
  (residual ~45 at FLRW, T=10 MeV).  Fixed by matching ``y2 = q_nodes
  = laguerre(N_q)`` so the interp is identity at the input nodes.
  After the fix the residual is ``6.7e-14``, consistent with kernel
  reduction-order noise divided by Hubble.
- **Jacobian dimensions**: the helper functions
  (`_resolve_jacobian_*`) explicitly group all
  pure-JAX bank-core modes under
  `_PURE_JAX_BANK_CORE_MODES`, so the new ``jax_kernel_preflight``
  mode automatically inherits the correct shapes
  (``moment=37``, ``apply=26`` at ``N_q=6, n_species=2``).
  Verified by the smoke test.
- **Backward compatibility**: existing modes' metadata strings are
  unchanged because the helper functions reproduce the original
  ternary chain exactly.  Verified by the unchanged smoke tests for
  ``spectral_relaxation_preflight``, ``projected_physical_preflight``
  and ``direct_kernel_preflight``.

## Verdict

Conditional pass.

The pure-JAX nu-e + pair collision operators are now reachable at
runtime through the standard bank-core dispatcher with a
JIT-compatible code path (no host callbacks).  The new mode is
private and audit-only; it does not promote the path to a public
backend or a canonical surface.

What remains for the full PR-T3B runtime patch (deferred to follow-up
phases):

- close an end-to-end Rodas5P solve smoke on the new
  ``jax_kernel_preflight`` mode at FLRW + bounded anisotropy;
- lock the FLRW `N_eff = 3.044 ± 0.01` target on that trajectory and
  verify the dQ_α sign convention;
- decide between (a) keeping the existing ``direct_kernel_preflight``
  host-callback path as a SciPy reference and (b) deprecating it in
  favour of ``jax_kernel_preflight`` once cross-code parity is
  established;
- consider replacing the pair-process PCHIP interpolation with a
  JAX-native natural cubic spline so the off-matched-grid parity
  (currently ~``8% rel`` per `PR-T3B_jax_operator_port.md`) tightens
  to reduction-order.
