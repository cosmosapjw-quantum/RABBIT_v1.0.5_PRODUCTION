# PR-T3B Option E — Canonical-After Enhancement Plan

> **Status**: bounded engineering enhancement, scheduled
> **AFTER** the canonical PR-T3B/C/D work lands (AP-form
> unification + DH-S coefficient).  Documented here as a
> design note so the canonical author has a clear handoff for
> the partial-IMEX path that stays compatible with Rodas5P +
> JAX/GPU.
>
> **Prerequisites**: canonical PR-T3B (AP-form unification) and
> PR-T3C (DH-S coefficient calibration) must land first.  Option
> E is a follow-up enhancement, not a canonical blocker.

## Goal

Reduce the FLRW ``N_eff`` gap to Mangano 2005 from the AP-form
documented limit (``~0.013``) toward ``~5e-3`` by absorbing the
fast equilibration transient inside the JIT-compiled RHS,
without changing Rodas5P or violating the JAX/GPU friendly
invariant.

## Architecture

A pre-conditioner is added to the existing JAX collision RHS
that performs an **analytic** relaxation of the bank state
toward equilibrium **before** the Rodas5P step controller sees
it.  The standard collision integral remains; the
pre-conditioner adds a small extra term that captures the
``f → f_eq`` exponential decay that the implicit solver would
otherwise resolve via microsteps.

### Mathematical form

For each species ``α`` and momentum bin ``q``:

    f_α(q, N + dN) = f_eq[T_γ](q) + e^{−Γ_α(T_γ) · dN} · (f_α(q, N) − f_eq[T_γ](q))

In the differential form fed to Rodas5P:

    df_α/dN
        = (transport term)
        + (full collision term — Hannestad-Madsen + pair + ν-ν)
        − Γ_α(T_γ) · (f_α − f_eq[T_γ])      ← pre-conditioner
        + Γ_α(T_γ) · (f_α − f_eq[T_γ])      ← compensation (so the net contribution is unchanged)

The ``±Γ·δf`` cancellation is **not** trivial: the
pre-conditioner is added at the **Jacobian** level, not the
RHS value level.  Specifically, the modified RHS function

    rhs_eff(N, y) = rhs_full(N, y)

is **identical** in value, but the **jacobian function** returns

    jac_eff(N, y) = jac_full(N, y) + (− Γ · I_bank ⊕ 0_other)

The diagonal ``+ Γ`` correction on the bank block forces the
implicit solver to "see" a stable manifold structure even when
the underlying collision integral has a near-zero ``∂C/∂T`` at
``T_γ = T_ν``.  Rodas5P's ``(I − γ h J)^{-1}`` operator then
projects efficiently onto the slow subspace.

This is the **modified equation** approach: same RHS, augmented
Jacobian.  Order of accuracy is preserved (the augmentation
vanishes as ``h → 0``); stability is improved because the
augmented Jacobian has bounded eigenvalues.

## JAX/GPU compatibility

- The pre-conditioner is a fixed diagonal addition to the bank
  block of the Jacobian.  Implementation: a single ``jnp.diag``
  with cached ``Γ_α(T_γ) / H_MeV`` values per RHS call.
- ``vmap`` over batch is unchanged.
- ``jit`` traces correctly (no Python branching, no host
  callback).
- Compatible with the existing low-rank Jacobian factorization
  used by ``solver_jax_rodas5p`` (the ``jac_eff`` simply adds a
  rank-``3·N_q`` correction to the existing ``base_jacobian``).

## Required deliverables

1. **Helper module** ``src/rabbit/jax/collision_ap_preconditioner_jax.py``:
   - ``compute_ap_preconditioner_diag(T_gamma, H_MeV, q_nodes,
     N_mu, n_species)``
     returns the diagonal correction array ``Γ/H * I_bank``.
   - ``apply_to_jacobian(base_jacobian, ap_diag, layout)`` adds
     it cleanly to the existing Jacobian, preserving the
     low-rank ``LowRankJacobianFactors`` structure.
2. **Driver wiring**: a new private
   ``collision_mode="ap_preconditioned_canonical"`` (or
   integration into the unified canonical mode from PR-T3B
   canonical) that adds the pre-conditioner to the existing
   Jacobian path.
3. **Unit tests**:
   - The pre-conditioner adds the expected diagonal entries
     (cross-checked against ``collision_rates_jax.gamma_over_H_jax``).
   - The modified Jacobian preserves ``LowRankJacobianFactors``
     shape contracts.
   - JIT determinism + vmap compatibility.
4. **Calibration tests**:
   - Re-run the FLRW ``N_eff`` baseline with the pre-conditioner;
     measure improvement vs ``0.013`` baseline.
   - Re-run the anisotropic stability sweep; ensure spread stays
     ``< 1e-3``.
   - Verify Rodas5P step count drops (pre-conditioner is doing
     its job).
5. **Audit doc** ``docs/audit/PR-T3B_option_E_<date>.md`` with
   measured results.

## Estimated effort

- ~1-2 weeks for a contributor familiar with the existing
  preflight surface.  Bounded because:
  - The math is well-understood (modified-equation argument).
  - The infrastructure (cubic spline, rate helpers, Jacobian
    factorization) is already in place.
  - The validation loop is locked at the existing 122+ test
    bundle.

## Risks

- The pre-conditioner trades a stiff manifold for a Jacobian
  augmentation that makes the linear solve slightly larger.
  Performance impact must be measured (likely modest at
  bounded ``N_q`` grids).
- The exact ``Γ`` to use (Mangano total rate vs Hannestad-Madsen
  channel-resolved rate) affects the calibration.  This is a
  hyperparameter that should be tuned during the calibration
  phase.
- The pre-conditioner does not close the model approximation
  gap entirely; it absorbs the **transient** that Rodas5P would
  microstep through.  The remaining ``~0.005-0.013`` gap is the
  AP-form residual.

## Decision criterion

If after Option E lands the FLRW Mangano gap is still ``> 5e-3``,
the conclusion is that the AP-form residual is the dominant
source and Options B / C (full IMEX or AP-Rosenbrock) become the
only path to canonical Mangano precision.  At that point a
research-grade decision is needed.

If the Mangano gap drops to ``< 5e-3`` after Option E, the
canonical surface can be promoted to "publication-grade
incomplete-decoupling" without invoking the load-bearing Rodas5P
invariant violation.

## Hand-off checklist for the canonical author

- [ ] Confirm canonical AP-form unification (PR-T3B canonical)
      has landed and is regression-locked.
- [ ] Confirm Dolgov-Hansen-Semikoz coefficient (PR-T3C
      canonical) has landed.
- [ ] Confirm the cumulative preflight FLRW Mangano gap is
      ``~0.013`` (measured at the canonical AP-form mode).
- [ ] Read this note + the Options-B/C research note in the
      same directory.
- [ ] Implement the helper module + driver wiring.
- [ ] Run the calibration tests; record results in the new
      audit doc.
- [ ] If gap closes to ``< 5e-3``: open a separate canonical-
      promotion phase (PR-T3B-Mangano).
- [ ] If gap remains ``> 5e-3``: archive Option E as "modest
      improvement, insufficient for Mangano precision" and
      defer Mangano precision to Options B/C indefinitely.
