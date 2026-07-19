# A2 — Free Measurement Gate (Track A, solver-optimization plan)

**Date:** 2026-06-30
**Purpose:** Plan Track A2 — baseline the already-landed AP preconditioner step counts and
the exact-kernel N_eff vs Mangano 3.044, to decide whether Tier-2 work (T2.1 well-balanced
exponential-Rosenbrock; T2.2 exact-kernel defect correction) is justified before building it.
**Method:** `run_full_boltzmann_jax` at the fiducial low-res config (N_q=4, N_mu=2,
Sigma_H_plus=0, eta=6.104e-10, tau_n=878.4, correction_level=0, thermo_tier=2), three
collision modes. CPU/JAX (the ROCm-plugin warning is a harmless no-GPU fallback).
Reference: Mangano et al. 2005, N_eff = 3.044 (`decoupling/moments.py:79`).

## Measurements

| collision_mode                | n_steps_p1 | n_steps_p2 | N_eff_measured | Mangano gap | wall (s) |
|-------------------------------|-----------:|-----------:|---------------:|------------:|---------:|
| `ap_unified_preflight`        |        218 |        313 |       3.034482 |    9.518e-3 |     25.4 |
| `ap_preconditioned_canonical` |       1378 |        313 |       3.034486 |    9.514e-3 |     26.1 |
| `jax_kernel_preflight` (exact)|        199 |        313 |       2.993416 |    5.058e-2 |     32.3 |

## A2(a) — AP preconditioner step counts → T2.1 decision

**Finding: the AP preconditioner does NOT remove microstepping; it makes phase 1 ~6× WORSE**
(218 → 1378 steps) at this config, for an N_eff shift of only ~4e-6. n_steps_p2 is identical
(313) across all modes — the regime change is entirely in phase 1.

Implications:
- The AP preconditioner (`collision_mode="ap_preconditioned_canonical"`) is **not** delivering
  its intended [L2] benefit here — the opposite. Before any T2.1 (well-balanced
  exponential-Rosenbrock), the **6× phase-1 step blowup must be root-caused** (mis-tuned
  preconditioner diagonal? a more conservative controller in the canonical mode? a conditioning
  regression?). A solver mode that sextuples the step count is a regression, not an optimization.
- T2.1 is therefore **conditionally justified** (residual phase-1 stiffness clearly exists — even
  the baseline 218 steps vs 313 in phase 2 shows phase-1 is the stiff regime), but the FIRST
  action is to diagnose the AP-preconditioned step explosion, not to add a new integrator mode.

## A2(b) — exact-kernel N_eff vs Mangano → T2.2 decision

**Finding: the existing exact-kernel path does NOT close the Mangano gap at this resolution —
it OVERSHOOTS the other way.** `jax_kernel_preflight` gives N_eff = 2.9934 (gap 5.06e-2),
*further* from 3.044 than the AP-form's 9.5e-3 (and below 3.0). The AP-form residual (9.5e-3)
is confirmed and matches `test_ap_preconditioned_mangano_gap.py`.

Implications:
- The plan's working assumption (from the eval doc + the mangano-gap test note) that "the gap is
  closure, fixable by routing through a full HM kernel" is **NOT confirmed by this measurement**:
  the wired exact kernel at N_q=4 lands at gap 5e-2, worse than the AP-form. So T2.2
  (exact-kernel defect correction) is **not yet justified as a gap-closer** — it would currently
  *enlarge* the gap.
- The likely cause is **resolution**: N_q=4 / N_mu=2 is far below convergence; the absolute N_eff
  of every mode is resolution-limited (the AP-form's proximity to 3.044 at N_q=4 may itself be
  partly cancellation, not accuracy). **The required next measurement is a resolution-convergence
  ladder** (N_q = 4 → 6 → 8 → …, N_mu likewise) for `jax_kernel_preflight`: does the exact-kernel
  N_eff converge UP toward 3.044, and does the AP-form gap stay pinned at ~9.5e-3? Only a
  converging exact kernel justifies T2.2.

## Track convergence note

The Bianchi-I Track-B exact-PSTF substrate (the isotropic collision operator with exact detailed
balance, F_0 independently brute-force validated) is an exact HM monopole kernel and remains a
candidate to close the production gap. **But this measurement shows the question is subtler than
"use the exact kernel":** the existing exact-kernel path undershoots at low res. Closing the gap
needs (1) a converged exact kernel and (2) confirmation it lands at 3.044 — both pending the
resolution ladder above.

## Decisions

1. **T2.1 (exp-Rosenbrock): GATED on root-causing the AP-preconditioner 6× phase-1 step blowup
   first.** Do not add a new integrator mode until the existing preconditioner regression is
   understood.
2. **T2.2 (exact-kernel defect correction): GATED on a resolution-convergence ladder** showing
   `jax_kernel_preflight` N_eff → 3.044 as the grid refines. Currently it lands at 2.993 (gap
   worse than the AP-form), so a defect correction toward it would *not* close the gap today.
3. **The "free measurement gate" did its job:** both Tier-2 items would have been premature. The
   next Track-A measurement is the N_q/N_mu convergence ladder (slow; one focused study).

*Caveat: N_q=4, N_mu=2 is a smoke resolution; absolute N_eff are not converged. These are
relative/diagnostic findings (step-count regime, gap ordering), not production N_eff claims.*
