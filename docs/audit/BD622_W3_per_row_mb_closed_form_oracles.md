# BD622 W3 — Per-row Maxwell–Boltzmann closed-form collision-loss oracles

Status: **DERIVED + internally cross-checked** (rel. err ≤ 4e-14). Prospective T01 falsifier
foundation. Read-only derivation; no repository collision kernel, comparator, GL48/64, trajectory,
or endpoint was executed. The only computation is standalone NumPy quadrature of the analytic
formulas below (zero repo import), reproduced in §6. This checks the MB reduction and coefficient
ratios; it is not an independent absolute phase-space-normalization validation.

## 1. Purpose

BD622 finding **F-5 / H2 residual**: the tagged matrix-element coefficients `{64, 128, 32}` and the
`K_s / K_t` per-row kernel assignments are transcribed from the *same* published sources
(HM 1995 + DHS 1997) on both the Rust production side (`neutrino_self_spectral.rs`) and the D-028
comparator (`_independent_noqke.py`). The only in-repo primary anchor is the row-1 normalization
(`8/π³`, `neutrino_self_spectral.rs:1073-1074`) and the K_t azimuth identity. A shared
transcription error in the **K_t rows {128, 32}** would scale out of every current gate (nulls,
conservation, CP, μτ covariance) and be invisible. This document derives and freezes an
**independent closed-form loss-rate ratio/reduction oracle for every massless neutrino self row**.
The absolute invariant phase-space normalization, a direct row-ii v-spinor trace, and the
finite-electron-mass rows remain assigned to W7.

## 2. Setup

Massless flavour-diagonal neutrino 2↔2 collision, one helicity dof. In the dimensionless comoving
variable `y = q/T_cm`, per-target loss rate for a row with tagged coefficient `c` and kernel
`K ∈ {K_s, K_t}` (matching `tagged_massless_mb_loss`, `neutrino_self_spectral.rs:825-841`):

```
C_loss(y1) = [ c · G_F² · T_cm⁵ / (256 π³ y1) ] · ∫₀^∞ dy2 ∫₋₁¹ dμ12 ∫₋₁¹ dz*  y2 · K · F_loss
```

with, exactly as coded (`:267-297`):

- `s = 2 y1 y2 (1 − μ12)`,  `K_s = ¼ s²`  (independent of `z*`),
- `B² = (y1−y2)² + 2 y1 y2 (1+μ12)`,  `χ = (y1−y2)/B`,  `1−χ² = 2 y1 y2 (1+μ12)/B²`,
- azimuth-averaged `K_t = (s²/16) · [ (1+χ z*)² + ½ (1−χ²)(1 − z*²) ]`.

For the loss oracle the occupations are Maxwell–Boltzmann with no Pauli blocking,
`f_i = e^{−y_i}`, so `F_loss = f1 f2 = e^{−(y1+y2)}` (the gain term equals it at MB equilibrium by
energy conservation `y3+y4 = y1+y2`; the oracle fixes the *magnitude*).

## 3. Azimuth reduction — the K_t/K_s = 1/3 identity (χ-independent)

The bracket integrates over `z*` with the χ-dependence cancelling exactly:

```
∫₋₁¹ (1+χz)² dz            = 2 + (2/3)χ²
∫₋₁¹ ½(1−χ²)(1−z²) dz      = (2/3)(1−χ²)
sum                        = 2 + (2/3)χ² + (2/3) − (2/3)χ² = 8/3       (independent of χ)
```

Hence `∫₋₁¹ K_t dz* = (s²/16)(8/3) = s²/6`, while `∫₋₁¹ K_s dz* = ¼s²·2 = s²/2`. Therefore

```
∫ dz* K_t  =  (1/3) · ∫ dz* K_s      pointwise in (y1, y2, μ12).
```

Because the remaining `y2` and `μ12` integrals and the MB weight are identical for K_s and K_t
rows, **every K_t-row loss rate is exactly ⅓ of the same-coefficient K_s row.** This reproduces the
in-repo `∫K_t/∫K_s = 1/3` test (`:1092-1134`) from first principles and, crucially, shows it is a
*pointwise* reduction, not merely an integrated coincidence.

## 4. The Maxwell–Boltzmann integral

With `K_s = y1²y2²(1−μ12)²` and `∫dz*=2`:

```
∫dμ12 (1−μ12)² = 8/3 ,    ∫₀^∞ dy2  y2·y2²·e^{−y2} = Γ(4) = 6
∫ (K_s loss integrand) = y1² e^{−y1} · 6 · (8/3) · 2 = 32 · y1² e^{−y1}
```

so, in units of `G_F² T⁵ · y e^{−y}`:

```
C_loss^{Ks}(c) = c · 32 / (256 π³) = c / (8 π³)
C_loss^{Kt}(c) = (1/3) · c / (8 π³) = c / (24 π³)
```

The loss rate is exactly `∝ y·e^{−y}` (no residual y-dependence in the coefficient), confirmed to
~1e-14 in §6.

## 5. Frozen per-row oracle table (T01)

Units: `C_loss(y) / [ G_F² T_cm⁵ · y · e^{−y} ]`. Rows i–v are the reversible massless neutrino
self channels (audit §3.2 enumeration; folded partition E={ν_e,ν̄_e}, X={ν_μ,ν̄_μ,ν_τ,ν̄_τ}).

| row | reaction | diagrams | tagged c | kernel | **closed form** | decimal |
|---|---|---|---:|---|---|---|
| i | ν_αν_α→ν_αν_α (& ν̄ν̄) | t+u Fierz, S=½ | 64 | K_s | **8/π³** | 0.25801227546 |
| ii | ν_αν̄_α→ν_αν̄_α | t+s | 128 | K_t | **16/(3π³)** | 0.17200818364 |
| iii | ν_αν_β→ν_αν_β (α≠β, same sign) | t | 32 | K_s | **4/π³** | 0.12900613773 |
| iv | ν_αν̄_β→ν_αν̄_β (α≠β) | t | 32 | K_t | **4/(3π³)** | 0.04300204591 |
| v | ν_αν̄_α→ν_βν̄_β (α≠β) | s | 32 | K_t | **4/(3π³)** | 0.04300204591 |

The Rust folded channels (`ROW1..ROW9`, `neutrino_self_spectral.rs:136-170`) and the comparator's
`independent_pair_row_fingerprint` must reproduce these five magnitudes at the MB state.

## 6. Independent numerical verification (standalone; reproducible)

`venv/bin/python` with NumPy only, no repo import. Gauss–Laguerre(64)×Gauss–Legendre(40)²:

```
row                       c kernel         numeric     closed-form     rel.err
i  nu_a nu_a (self)      64  Ks    0.2580122755    0.2580122755    3.64e-14  (y-spread 6.2e-14)
iii nu_a nu_b same       32  Ks    0.1290061377    0.1290061377    3.64e-14  (y-spread 6.2e-14)
ii  nu_a nubar_a        128  Kt    0.1720081836    0.1720081836    3.99e-14  (y-spread 1.3e-14)
iv  nu_a nubar_b         32  Kt    0.0430020459    0.0430020459    3.99e-14  (y-spread 1.3e-14)
v   nunubar->nunubar     32  Kt    0.0430020459    0.0430020459    3.99e-14  (y-spread 1.3e-14)
Kt/Ks ratio at fixed c=32: 0.333333  (expect 1/3)
azimuth int_-1^1 bracket dz = 2.666666666667 for chi in {-.9,-.3,.3,.9}  (expect 8/3)
```

Script: `scripts/audit/w3_mb_oracle_check.py` (kernels re-transcribed from the .rs formulas by
hand; no import of `rabbit` or the native crate).

## 7. Coefficient provenance — partial independent spin-trace derivation (F-5 coefficient leg)

The MB **reduction machinery** in §3–§4 (kernel → `c/(8π³)` or `c/(24π³)`) is first-principles. The
**ratios of the tagged coefficients `{64, 128, 32}` and the K_s/K_t assignment are independently
checked** from Dirac spin traces of the massless V-A neutral current, with **no code constant
used**. The overall normalization is not fixed by this calculation.

Method: effective contact interaction with vertex `V^μ = γ^μ (1−γ⁵)/2`; massless spin sums
`Σ u ū = Σ v v̄ = p̸`; a single distinguishable t-channel diagram
`[ū₃ V^μ u₁][ū₄ V_μ u₂]` sets the base. Explicit γ-matrices, generic null kinematics, numerical
trace evaluation (`scripts/audit/w3_spin_traces.py`, zero repo import):

| row | topology | trace result | kernel | ratio to base |
|---|---|---|---|---|
| iii ν_αν_β→ν_αν_β | single t | 16·K_s | K_s | **1** |
| iv ν_αν̄_β→ν_αν̄_β | single t (ν̄ leg) | 16·K_t | K_t | **1** |
| v ν_αν̄_α→ν_βν̄_β | single s (annihilation) | 16·K_t | K_t | **1** |
| i ν_αν_α→ν_αν_α | t+u, S=½ | 32·K_s | K_s | **2** |
| ii ν_αν̄_α→ν_αν̄_α | t+s | 4×base·K_t | K_t | **4** (crossing) |

The base (my coupling convention `16`) maps to the code's tagged `32`; the **ratios**
`{i:ii:iii:iv:v} = {2:4:1:1:1} = {64:128:32:32:32}` and the kernels `{K_s,K_t,K_s,K_t,K_t}` match
the production table exactly. Rows iii/iv/v/i are confirmed **numerically** (single-channel
universality + identical-particle t+u with S=½ doubling, correct interference trace); **ii = 4×base
K_t is inferred by crossing** from the numerically-verified `i` amplitude (t+u → t+s continues
the invariant, the identical-particle ½ drops, and the fermion-exchange sign flips → the base
doubles again). A direct v-spinor evaluation of row ii is still required. Consequently F-5 is
only narrowed: ratio or kernel-assignment errors would be caught, while a shared overall
normalization or row-ii crossing/interference error could still survive. W7 owns those gaps.

## 8. T01 oracle specification (frozen)

- **Identity**: at the MB state (`f_i = e^{−y_i}`, no blocking), the per-row massless loss rate
  equals the §5 closed form.
- **Inputs**: MB occupations; frozen grids; no trajectory.
- **Norm / cap**: relative per row; cap **1e-11** (smooth-integrand spectral class, two-grid
  Richardson-derived); expected conditioning benign (no cancellation, no `1/y²`).
- **Pass rule**: every row within cap. **On failure**: reopen the catalogue-coefficient
  uncertainty class (F-5); halt the T-ladder. No cap/order retry.
- **Scope**: massless neutrino self rows i–v. The finite-`m_e` electron rows (vi, vii) with `K_u`
  and interference terms have no pure-massless closed form and are deferred to the W7 MPFR oracle
  (audit T04).
