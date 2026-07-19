# A2 follow-on — Exact-kernel resolution ladder (Track A, decides T2.2)

**Date:** 2026-06-30
**Purpose:** The A2 measurement gate left one open question: the exact-kernel path
(`jax_kernel_preflight`) gave N_eff=2.9934 (gap 5.06e-2) at N_q=4 — was that a low-resolution
artifact that converges UP toward Mangano 3.044 as the grid refines (→ T2.2 justified), or is it
a converged physics error (→ T2.2 refuted)? This ladder answers it.
**Method:** `run_full_boltzmann_jax`, fiducial (Sigma_H_plus=0, eta=6.104e-10, tau_n=878.4,
correction_level=0, thermo_tier=2), N_q × N_mu = (4,2), (6,4), (8,4), two collision modes.

## Results

| mode                    | N_q | N_mu | N_eff_measured | Mangano gap | n_steps_p1 | wall (s) |
|-------------------------|----:|-----:|---------------:|------------:|-----------:|---------:|
| `jax_kernel_preflight`  |   4 |    2 |       2.993416 |    5.058e-2 |        199 |     12.3 |
| `jax_kernel_preflight`  |   6 |    4 |       2.993422 |    5.058e-2 |        177 |     33.0 |
| `jax_kernel_preflight`  |   8 |    4 |       2.993421 |    5.058e-2 |        171 |     37.1 |
| `ap_unified_preflight`  |   4 |    2 |       3.034482 |    9.518e-3 |        218 |     24.2 |
| `ap_unified_preflight`  |   6 |    4 |       3.034485 |    9.515e-3 |        178 |     25.3 |
| `ap_unified_preflight`  |   8 |    4 |       3.034482 |    9.518e-3 |        171 |     25.4 |

## Finding — both modes are already converged at N_q=4

N_eff is **flat to ~6 significant figures** across N_q = 4 → 6 → 8 for BOTH modes (Δ ~ 6e-6).
The Mangano gap is therefore **not** a resolution artifact. Specifically:

- The exact kernel (`jax_kernel_preflight`) **converges to N_eff = 2.9934** (gap 5.06e-2) and
  stays there. It does NOT approach 3.044 — it sits *below 3.0*, **further from Mangano than the
  AP-form** (3.0345, gap 9.5e-3).
- The AP-form (`ap_unified_preflight`) converges to 3.0345 (gap 9.5e-3, the documented AP-form
  residual).

## Decision — T2.2 (exact-kernel defect correction) is REFUTED

The plan's working hypothesis (eval doc + `test_ap_preconditioned_mangano_gap` skip-reason) was
that "the gap is closure, fixable by routing the collision RHS through a full HM kernel." **The
ladder refutes this for the wired exact kernel:** it is converged at gap 5e-2 — a defect
correction `D = C_exact − C_AP` toward it would move N_eff from 3.0345 to 2.9934, i.e. **ENLARGE
the gap by ~5×.** T2.2 as a gap-closer is dropped.

## What the gap actually is

The Mangano +0.0095 is a **converged physics/model error**, integrator-agnostic (eval doc [L3]),
and — newly — **not closable by the exact kernel** (which is itself further off, at 2.9934). The
exact kernel landing *below 3.0* points to missing/under-counted physics in the exact-kernel
implementation (e.g. under-resolved e± annihilation reheating, a weak-rate normalization, or a
spectral-distortion term), not a solver/integrator/Jacobian issue. **Closing the gap is physics
work, outside the solver-optimization track.**

## Revised Tier-2 / Track-A status

- **T2.2 — DROPPED** (refuted above).
- **T2.1 (exp-Rosenbrock) — still gated on root-causing the N_q=4 `ap_preconditioned_canonical`
  1378-step anomaly** (A2). Note: this ladder shows the OTHER modes' `n_steps_p1` *converge* to
  ~171 at N_q=6/8 (the N_q=4 counts 199/218 are the coarse-grid transient). The 1378 outlier was
  not re-tested at higher N_q; if it too converges to ~171, T2.1 is unjustified. One more run
  (`ap_preconditioned_canonical` at N_q=6/8) would settle it.
- **A1b (analytic-Jacobian wiring) — remains a valid [L1] speedup** (kills a 2×N_state FD loop),
  but its production target is elusive (the `direct_kernel` FD path uses legacy scat/pair
  operators, not the channel-grid contraction the analytic Jacobian targets — see plan A1b notes).
- **Net:** the production solver-optimization track's clean wins are largely **landed** (the
  analytic channel + JVP + LRS-moment Jacobians, PR#1/#2/#4); the measurement gates have
  decisively shown T2.1/T2.2 are not clean wins. The remaining production lever (the Mangano gap)
  is a **physics defect**, not solver work.

## Cross-track opportunity (flagged, not actioned)

The Bianchi Track-B exact-PSTF substrate (isotropic collision operator, exact detailed balance,
F_0 brute-force validated) is an independent exact HM kernel. A high-value diagnostic: compute its
isotropic-limit N_eff and compare to this production exact kernel's 2.9934. If the Track-B kernel
lands at ~3.04+, it would localize the production exact-kernel's missing physics — turning the
"physics defect" into a concrete fix. This bridges the two tracks but is a separate study.

*Caveat: deterministic CPU/JAX (ROCm no-GPU warning is a harmless fallback). N_eff converged in
N_q/N_mu; absolute values are the model's, not a claim of correctness vs Mangano.*
