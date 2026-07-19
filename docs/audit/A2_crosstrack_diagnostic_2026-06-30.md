# A2 cross-track diagnostic — can the Bianchi νν kernel localize the Mangano gap?

**Date:** 2026-06-30
**Purpose:** Act on the cross-track opportunity flagged (not actioned) in
`A2_resolution_ladder_2026-06-30.md` §"Cross-track opportunity" (lines 66–72): compute the
Track-B (Bianchi) exact νν kernel's isotropic energy-transfer rate and compare it to the
production exact kernel's converged **N_eff = 2.9934** (gap 5.06e-2 below Mangano 3.044), to turn
the "physics defect" into a localized, concrete fix.

**Method:** three independent code mappings (production νν kernel; N_eff=2.993 provenance; Bianchi
operator convention) + a synthesis feasibility judgement, then a self-contained Bianchi-side
measurement (`tests/test_pstf_isotropic_energy_transfer.py`).

---

## Finding 1 — the matched two-kernel comparison is INFEASIBLE as framed

The ladder's hopeful framing ("if the Track-B kernel lands at ~3.04+, it localizes the production
missing physics") **does not hold**, for three independent, code-grounded reasons:

1. **Channel mismatch at the source.** The production N_eff=2.993416 is computed by the
   `jax_kernel_preflight` collision mode, which contains **only νe elastic scattering + e⁺e⁻ pair
   annihilation and explicitly EXCLUDES νν scattering**
   (`jax/driver_typeI_full_boltzmann.py:748`; `N_eff_from_3T_jax` at `:2866`). There is therefore
   **no production N_eff that carries a production νν rate** to compare the Bianchi νν kernel
   against. The Mangano gap *of that run* cannot live in a νν kernel — νν is not in it.

2. **The production νν kernel that exists is uncalibrated.** `nu_nu_diagonal_collision_tensorized_jax`
   (`jax/nu_nu_scattering_jax.py:277`) is **not wired into any driver**, and its matrix-element
   normalization is a placeholder: `matrix_coeff = 1.0` (the νe `G_L²+G_R²` stand-in), with the
   Dolgov–Hansen–Semikoz νν coefficient **deferred to PR-T3C** (`docs/audit/PR-T3C_preflight.md`).
   Any numeric ratio against it is dominated by an unlocked O(1) coupling, not physics.

3. **Different matrix-element reductions.** Production uses the angle-AVERAGED
   `M² = (y₁y₂)² + (y₃y₄)²` over a Laguerre `exp(y)` measure with `G_F²/(4π³)`; Bianchi uses the
   FULL angular `W = 32 G_F² (E₁E₂ − p₁p₂ x)²` reduced by the body-frame `Θ(D)/√D` x-integral with
   `2/(2π)⁴` and explicit `p²/2E` measures. These coincide **only after** the body-frame x-integral
   (an angular-average identity), never pointwise — so even a "correct" comparison needs that
   identity proven first, plus a leg-by-leg measure/prefactor conversion.

**Correction to the ladder report:** the cross-track paragraph (lines 66–72) is too optimistic.
The honest matched comparison is a **multi-PR effort** — wire νν as a third production channel,
land the DHS calibration (PR-T3C), and derive+validate the measure/prefactor conversion — not a
one-PR diagnostic. It is recorded as a scope item, not actioned here.

## Finding 2 — what the Bianchi νν operator CAN be certified to do (self-contained)

`tests/test_pstf_isotropic_energy_transfer.py` certifies the independent exact reference on its own
terms (no production import, no driver wiring):

| property | result | significance |
|----------|--------|--------------|
| equilibrium null | energy & number rate ~1e-38 ≪ 1e-24 at FD, even at the convergence grid | DB-exact: once wired it can only shift N_eff via genuine distortion, adds no spurious heating |
| **resolution** | sign-unstable at N≤8; converges at **N≥12** (N=20 vs 24 agree to ~2e-4) | a **stricter** momentum-grid bar than the production N_eff (flat already at N_q=4) — any matched comparison must run the Bianchi reference at N_q≥12 |
| **nonperturbative** | super-linear in distortion amplitude (ratio ~3.3–3.8 per doubling vs 2.0 linear; effective local exponent ~1.8) | beyond-linear-response / nonlinear Pauli-blocking — NOT a cubic power law (cubic→ratio 8). Distinct from the shear test's Ĉ₁, a parity-forced *exact* odd cubic in the dipole amplitude (`test_pstf_collision_operator_shear.py`) |

## Implication for the Mangano gap

Because the converged production deficit (N_eff=2.993, below 3.0) comes from a **νe + pair only**
run, the gap of *that run* **cannot be νν — νν is not in it** (definitional exclusion, not a proof).
The remaining un-excluded candidates are **νe / e⁺e⁻ / finite-T QED corrections / grid /
normalization**; these are *candidates*, not a localized cause. νν self-scattering thermalizes the
spectrum (relaxational, near-elastic within the sea), so it is not a plausible source of *that*
number. **Honest bound (external audit 2026-06-30):** excluding νν from this exact-kernel-preflight
run does **not** thereby prove the weak-rate / QED / reheating implementation is correct — it only
removes one channel from the search. The +0.0095-class Mangano gap remains a physics/closure
question, narrowed (νν ruled out for this run) but not localized to a specific defect.

## Scope update (auto-managed)

- **NEW item — "νν as a 3rd production channel + measure conversion" (depends on PR-T3C).** The
  only path to an apples-to-apples Bianchi-vs-production νν comparison. Multi-PR; gated on the DHS
  calibration. Logged in the plan; not started.
- The production solver-optimization track (Track A) remains as the ladder left it: clean wins
  landed (analytic Jacobians); T2.1/T2.2 refuted by measurement; the gap is physics, and is now
  further localized **away from νν**.

*Caveat: deterministic CPU/JAX. The Bianchi rate's absolute magnitude is the model's νν
energy-transfer at the test state, not a claim of correctness vs any reference; the certified facts
are the equilibrium null, the N≥12 resolution bar, and the super-linear (nonperturbative) scaling.*
