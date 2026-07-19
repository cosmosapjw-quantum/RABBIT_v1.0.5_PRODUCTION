# PR-T3B IMEX/Stiff-Manifold Hybrid Options B and C — Research Notes

> **Status**: deferred research-grade, **OUT** of bounded canonical
> scope per PR-T3B-PF #15.  Documented here as a research-grade
> reference so a future contributor exploring partial-IMEX
> integration with Rodas5P + JAX/GPU has a clear baseline.
>
> **Companion**: ``docs/audit/PR-T3B_jax_kernel_runtime.md``
> (cumulative preflight calibration + scope reframing rationale).

## Context

The cumulative PR-T3B preflight trail (commits `e5d671b` through
`b0c7d7d`) established that:

1. The full Hannestad-Madsen + pair JAX kernel
   (``jax_kernel_preflight``) is anti-heating at FLRW
   (``N_eff = 2.993`` vs Mangano ``3.044``) because the SciPy
   reference implicitly assumes ``T_e = T_ν``.
2. The q-grid remap fix (``q_remap = q_nodes * T_α / T_γ``)
   physically corrects the sign convention but exposes a stiff
   ``∂C/∂T`` Jacobian manifold at FLRW where ``T_γ ≈ T_ν``.
3. Rodas5P's implicit step controller cannot follow the stiff
   manifold without microstepping (1382 steps for 0.01 e-folds
   measured at FLRW initial conditions).
4. AP-form preflight modes (``spectral_relaxation``,
   ``projected_physical``) sidestep the stiff manifold but cap
   N_eff fidelity at ``~0.013`` from Mangano.

The canonical destination was narrowed to AP-form unification +
Dolgov-Hansen-Semikoz coefficients with the ``0.013`` gap
documented as an AP-form model approximation limit
(PR-T3B-PF #15).

This research note captures **Options B and C** — the two
research-grade paths to closing that gap that were deemed out of
scope but should be revisited if a future contributor decides to
push to ``5e-3`` Mangano precision.

## Option B: Modified Rosenbrock with Jacobian decomposition

### Sketch

A Rosenbrock-Wanner method solves at each stage

    (I − γ h J) k_i = h f(y_n + Σ a_ij k_j) + h J Σ c_ij k_j

The proposal: split the Jacobian as

    J = J_slow + J_fast

where ``J_fast`` captures the stiff equilibration block (``∂C/∂T``
manifold from the q-remap) and ``J_slow`` captures the rest
(transport, weak rates, network).  Form

    (I − γ h J_fast − γ h J_slow) k_i = ...

with ``J_fast`` treated as a structured low-rank correction so
the inverse can be applied via Sherman-Morrison.

### Required infrastructure

- Analytic factorization of ``J_fast`` with a known low-rank
  structure (typically rank ``≤ 6`` from the active scalar block
  ``[T_γ, T_νₑ, T_νₓ, Σ_+, Σ_−, S]``).
- Modified `_rodas5p_step` that accepts a pre-factorized
  ``J_fast`` and applies Sherman-Morrison.
- Updated step controller that accounts for the absorbed
  fast manifold (otherwise it still sees the original error
  estimator and microstpes).

### JAX/GPU compatibility

- Sherman-Morrison update is pure linear algebra; ``jnp`` ops only.
- ``vmap`` over batch dimension still works.
- Step controller modification is the trickiest piece — needs
  to coexist with the existing event-detect ``lax.while_loop``.

### Why this is research-grade

- Re-derivation of the order conditions for Rosenbrock with split
  Jacobian (Steinebach 2023's Rodas5P assumes a single ``J``).
- Loss of the existing 8-stage 5(4) error embedding may force a
  re-tuning.
- Risk of order reduction in the stiff regime (a known issue
  for Rosenbrock-Wanner with structured Jacobians).
- Validation requires reproducing the existing tier-1 / tier-2
  parity envelope (``|ΔY_p| < 5 × 10⁻⁵``) — the modified solver
  must not regress on the canonical surfaces.

### Estimated effort

- ~3-6 weeks of focused research-grade work.
- Multi-iteration validation loop required.

### References

- Steinebach 2023, *BIT Numerical Mathematics* 63:27 (Rodas5P
  derivation; this is the load-bearing reference for the current
  solver).
- Hairer & Wanner, *Solving ODEs II* §IV.7 (Rosenbrock-Wanner
  order conditions).
- Schneider 2003, *J. Comput. Appl. Math.* 154:33 (W-methods with
  approximate Jacobians; closest existing literature).

## Option C: Asymptotic-Preserving Rosenbrock with manifold projection

### Sketch

An AP-Rosenbrock scheme replaces each Rodas5P step with a
manifold-aware projection: at each accepted step,

1. Compute the equilibrium manifold ``f_eq[T_γ]`` for the bank
   state.
2. Decompose the error estimator into "deviation from manifold"
   + "drift along manifold".
3. Allow large step sizes when the trajectory has projected onto
   the manifold (i.e., when ``||f − f_eq|| < tolerance``).
4. Use small step sizes only during the rapid initial transient
   to the manifold.

### Required infrastructure

- AP-aware error estimator: ``||L(f − f_eq)|| / atol`` instead of
  ``||L(y_new − y_old)|| / atol``, where ``L`` is the projection
  onto the slow subspace.
- Asymptotic expansion of the Rosenbrock stages around the
  manifold (``f_n = f_eq + ε δf + ε² δ²f + ...``) with
  step-size control on ``ε`` rather than ``f``.
- Fall-back to the standard Rodas5P controller when the system
  is far from the manifold (e.g., during freeze-out transitions).

### JAX/GPU compatibility

- Manifold projection ``f → f_eq + (f − f_eq)`` is pure jnp.
- The asymptotic expansion uses a fixed power-of-ε truncation;
  no Python branching.
- ``vmap`` over batch is unchanged.

### Why this is research-grade

- AP-Rosenbrock variants exist in the literature (Hu et al. 2018,
  Boscarino et al. 2020) but no published JAX implementation.
- Rigorous error analysis required: the AP property must hold
  uniformly in the stiffness parameter ``ε = H/Γ``, otherwise
  the canonical parity envelope can drift unpredictably.
- Validation harder than Option B because the error estimator
  itself is now implementation-defined.

### Estimated effort

- ~6-12 weeks of focused research-grade work.
- Requires a coupled physics + numerics validation pass.

### References

- Hu, Jiang, Pareschi 2018, *J. Sci. Comput.* 76:1858
  (AP IMEX-Rosenbrock for hyperbolic-relaxation systems).
- Boscarino, Russo, Scandurra 2020, *J. Sci. Comput.* 82:71
  (asymptotic-preserving methods for stiff kinetic equations).
- Filbet & Jin 2010, *J. Comput. Phys.* 229:7625 (foundational
  AP scheme for the Boltzmann equation).

## Observations across both options

- Both options keep the existing ``rabbit.jax.driver_typeI_full_boltzmann``
  RHS structure unchanged; the modification lives in
  ``rabbit.jax.solver_jax_rodas5p`` (or a new
  ``solver_jax_ap_rodas5p`` module).
- Both options preserve the JAX/GPU friendly architecture if
  implemented as ``jnp``-only operations inside the solver core.
- Both options would land as a new ``solver_jax_*`` module
  rather than modifying the existing canonical Rodas5P; the
  existing solver remains the load-bearing default per the
  invariant in ``ROADMAP_STATE_OF_RECORD §1.2``.
- Either option, if landed, would close the FLRW Mangano gap
  from ``0.013`` toward ``5e-3`` while preserving the bounded
  preflight calibration baseline (the AP-form modes remain
  available as fallback / cross-check surfaces).

## Decision

Both options are **deferred indefinitely**.  The cumulative
preflight calibration trail (122+ regression-locked tests) gives
a future contributor a measurable starting point if they decide
to revisit either option.

The canonical PR-T3B/C/D work continues with the narrower scope
(AP-form unification + DH-S coefficient + documented Mangano
gap) per PR-T3B-PF #15.

The post-canonical ``Option E`` (in-RHS analytic relaxation
pre-conditioner; see
``docs/research/PR-T3B_option_E_canonical_post_enhancement.md``)
is the bounded-effort partial-IMEX path that may be revisited
sooner.
