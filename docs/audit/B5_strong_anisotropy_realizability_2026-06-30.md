# B5 — strong-anisotropy realizability: the band-limited logit vs the divergent maxent multiplier

**Date:** 2026-06-30
**Purpose:** The plan's B5 gate was "`0 <= f <= 1` on a Lebedev grid at Σ = 0.3, 0.5". This records why
that gate is **vacuous** and replaces it with the non-trivial strong-anisotropy well-posedness
measurement, including the maxent-divergence contrast that makes B5's central claim testable.

## Why the stated gate is vacuous

The forward solve enforces realizability with the **band-limited logit**: `f = 1/(1+exp(-(q+A)))` is
in `(0,1)` for ANY finite logit coefficient `A` (`augmented_pstf_distribution.fermi_dirac_from_logit`,
overflow-safe). So `0 <= f <= 1` measures the sigmoid's codomain, not the physics — it passes on
arbitrary finite `A`. (Lebedev is also roadmap-only; the real grid is tensor-product
Gauss-Legendre(μ) × uniform-midpoint(φ).)

## The non-vacuous measurement (operator-level, frozen Σ)

Exercising the existing AP62 operator (`augmented_nonlrs_nonlinear_mode_rhs`) from the FLRW logit
(A=0) under frozen Σ at |Σ| ∈ {0.3, 0.5} over 250 explicit-Euler steps (`tests/test_b5_strong_anisotropy_realizability.py`):

| measurement | result at |Σ| ≤ 0.5 |
|-------------|---------------------|
| A_modes finite (RHS hard-raises on NaN/Inf) | finite throughout |
| max\|A\| growth | 0.02 → ≤ 7.6, sub-exponential (growth ratio ~1.9–2.1) |
| FD-boundary margin `min(f.min, 1−f.max)` | strictly > 0 (down to ~2e-17, far from underflow; logit arg ≤ ~38 ≪ 710) |
| FLRW fixed point | source = 0 EXACTLY at Σ=0 for any A (∝ Σ) |
| Σ₋ source | exactly antisymmetric in Σ₋; W₊ unsourced |
| quadrupole π₊ | inside the realizable interval [W₊.min, W₊.max] = [−0.49, 0.98] |

The honest statements: realizability is **unconditional** (the logit never crosses the boundary);
the load-bearing strong-shear quantities are the **A_modes amplitude** (finite, bounded — but the
ceiling is an empirical regression lock, *not* a proven a-priori bound; there is no documented
guardrail on A_modes growth) and the **FD-boundary margin** (approaches but never reaches 0).

## The decisive contrast — maxent diverges where the logit stays finite

New diagnostic `src/rabbit/transport/realizability_diagnostics.py`:
- `maxent_M2_multiplier(π, kernel, weights)` — the Levermore/maxent Lagrange multiplier of the
  entropy closure `g ∝ exp(λ·kernel)` reproducing moment `π`. Verified correct: on a `{−1,+1}` kernel
  it equals `atanh(π)` exactly; round-trip recovery to ~1e-15. It **diverges** as `π → W₊.max` (λ: 0→0,
  0.5→2.3, 0.9→10.4, 0.97→34.5) and **raises `RealizabilityBoundaryError`** for `π` outside
  `(W₊.min, W₊.max)` — the multiplier does not exist beyond the realizability boundary.
- `fd_boundary_margin(f)` — the honest margin metric.

So at the strong-anisotropy boundary the maxent/M_N closure has **no finite multiplier**, while the
band-limited logit reaches the same regime with a finite, modest coefficient and a positive FD margin
— *no moment inversion required*. This is exactly the well-posedness mechanism B5/B.HR attributes to
the logit (`docs/audit/BHR_wellposedness_2026-06-30.md`): the logit replaces the divergent M_N
multipliers at the realizability boundary.

## Scope update (auto-managed)

- **B5 realizability mechanism certified at the operator level.** Gate:
  `tests/test_b5_strong_anisotropy_realizability.py` + the reusable `realizability_diagnostics`.
- **Honest edge:** the A_modes growth ceiling is an empirical regression lock at fixed (dN, n_steps,
  grid), not a theorem. A driver-level integration that co-evolves Σ to 0.5 (with the geometry
  feedback `dΣ = −(1−σ²)Σ + Π`) is a separate, heavier study; the strong-anisotropy *driver* run is
  not yet a unit test. If A_modes ever truly blow up at longer integration, this gate would begin to
  surface it but does not fully close that question.
