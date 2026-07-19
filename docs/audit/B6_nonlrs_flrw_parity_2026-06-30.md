# B6 — non-LRS cross-backend FLRW parity (Yp / D-H anchored to the SciPy reference)

**Date:** 2026-06-30
**Purpose:** Close the documented B6 gap. The candidate `jax_characteristic_nonlrs` backend had **no
cross-backend (JAX-vs-SciPy) parity gate on full BBN observables**. Existing coverage either compares
JAX-vs-JAX at the **degenerate N_phi=1** axisymmetric slice (`test_pr_n2_nonlrs_driver.py`) or compares
only an `N_eff_3T` temperature-ratio proxy (the controlled-FLRW ladder, explicitly *not full QKE*) —
neither pins the non-LRS backend's Yp/D-H to the canonical SciPy reference on a genuine S² grid.

**Method:** at Σ_H+ = Σ_H− = 0, on a genuine N_phi=8 S² grid (N_q=12, N_theta=16, n_rx=12), compare
the non-LRS characteristic backend (`run_full_coupled_typeI_char_jax`, `transport_mode='characteristic_nonlrs'`)
against the SciPy characteristic reference (`run_full_coupled_typeI`, `TransportMode.CHARACTERISTIC`).

## Result — anchored at the canonical publication bar

| cl | \|ΔYp\| (gate 5e-7) | margin | ΔD-H / D-H (gate 1e-4) | margin |
|---:|--------------------:|-------:|-----------------------:|-------:|
| 0  | 7.48e-8 | 6.7× | 1.66e-5 | 6× |
| 1  | 2.28e-8 | 22× | 2.17e-5 | 4.6× |
| 2  | 1.18e-9 | 425× | 2.02e-5 | 5× |

- The non-LRS backend reproduces the SciPy FLRW reference (`Yp = 0.2423494736`) at the **same Yp
  tolerance the canonical LRS tier-1 gate uses** (5e-7), with a **tightened** D-H lock (1e-4, vs the
  canonical 1e-3 — the measured residual is ~2e-5).
- The two JAX backends (non-LRS N_phi=8 vs LRS) agree to ~7e-9 at the FLRW limit.
- The inference-level `canonical_forward_solver(backend='jax_characteristic_nonlrs')` dispatches to
  `capability_key='jax_typeI_characteristic_nonlrs_tier1'` and reproduces the SciPy Yp.

## Honest scope (residual gap)

At Σ=0 the non-LRS forward direction map is the **identity** (verified: `I≡0`, `J≡1`, `μ≡μ₀` to
floating-point zero) — the entire shear-stretch arithmetic is dead. So this certifies the **FLRW
limit** of the non-LRS machinery: the correct reduction, the S² grid construction + weights, and the
`extract_monopole_S2` contraction (these stay live — doubling `w_s2` scales the monopole, flowing to
Yp), **not** the anisotropic transport at finite shear. Finite-shear non-LRS BBN has **no external
Bianchi-I code to anchor against** (the standing B8 gap); this is the strongest cross-backend anchor
available. It is a real regression lock: a sign/normalization/grid-weight bug in the non-LRS
intensity/Jacobian/monopole reconstruction would break it. (The N_phi metadata guard prevents a
silent collapse to the N_phi=1 LRS slice — the contract string alone does not depend on N_phi.)

## B6b addendum (PR#19) — FINITE-shear axisymmetric reduction (the live shear-stretch)

The Σ=0 gate above leaves the non-LRS shear-stretch arithmetic dead. PR#19
(`tests/test_nonlrs_finite_shear_lrs_reduction.py`) closes that for the axisymmetric direction: at
Σ_minus=0, Σ_plus∈{0.1, 0.3} on the genuine N_phi=8 grid the non-LRS backend reduces to the **trusted
LRS backend** on Yp/D-H (|ΔYp| 4e-9 / 2.6e-7; gate 1e-6), now exercising the LIVE
`diag(e^{-S+}, e^{-S+}, e^{2S+})` stretch (intensity_shift/jacobian_nonlrs). The shear genuinely moves
Yp (FLRW 0.2424 → 0.2440 at Σ_plus=0.3, monotone), so a sign/normalization bug in the finite-shear
non-LRS reconstruction — invisible at Σ=0 — is caught. Transitive anchor: non-LRS == LRS (here) and
LRS == scipy (canonical gate, Σ up to 0.5) ⟹ non-LRS finite-shear == scipy. Capped at Σ_plus≤0.3:
the LRS(N_mu) vs non-LRS(N_theta×N_phi) angular-quadrature difference grows steeply with shear
(7.4e-5 at Σ_plus=0.5 — a resolution artifact, not a defect). Still axisymmetric-only; Σ_minus≠0
finite-shear remains the B8 gap (anchored at the observable level by PR#16's machine-precision oracle).

## Scope update (auto-managed)

- **B6 non-LRS FLRW parity LANDED.** Gate: `tests/test_jax_typeI_characteristic_nonlrs_flrw_parity.py`
  (marked `slow`). The non-LRS candidate is now FLRW-anchored at zero shear on full BBN observables.
- **Residual (B8, unbounded):** the anisotropic (finite-shear) Bianchi path has no external cross-code
  gold anchor — no comparable external Bianchi-I BBN code exists. Internal anisotropic gold
  (`tests/fixtures/jax_bbn_gold.json`, 51 Σ_H cells) is a self-consistency regression lock only;
  shared-bias errors in anisotropic transport would pass undetected. This is the honest open frontier.
