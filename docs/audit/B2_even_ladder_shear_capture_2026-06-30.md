# B2 — the even-ladder hierarchy captures only ~28% of the exact shear response (W2-PR6 is structural)

**Date:** 2026-06-30
**Purpose:** The plan's **B1-B2** aimed to validate the high-ℓ even-ladder moment hierarchy against a
truncation-free reference at nonzero shear and thereby "kill the W2-PR6 19%" (the linearized ℓ=2
transport capturing only ~20-28% of the full shear response). This records the measurement — which
**refutes the recovery premise** and instead localizes W2-PR6 as a *structural* limitation.

**Method:** matched, collisionless-transport-only comparison of the energy-density anisotropy
`δρ = ρ_ν/ρ_FD − 1` between (a) the even-ladder moment hierarchy
(`integrate_even_ladder_final_state`, ℓ_max ∈ {2,4,6,8}) and (b) the EXACT characteristic-ray
collisionless monopole (`characteristic_rays_jax`: `mu_current`/`intensity_shift`/`jacobian`/
`extract_monopole`). Both start from `F_0 = f_FD`, `F_{ℓ≥2}=0` on the same `MomentumGrid(N_q=20)`;
the even-ladder holds `Σ_H` constant over `N ∈ [0, 0.5]` so the accumulated shear is exactly
`S = Σ_H · 0.5`, at which the characteristic primitives are evaluated. `N_μ = 24` (converged to ~1e-5).

## Result

| Σ_H | S = Σ_H·0.5 | exact δρ (characteristic) | even-ladder δρ (ℓ=2…8) | capture |
|----:|------------:|--------------------------:|-----------------------:|--------:|
| 0.1 | 0.05 | 2.677e-2 | 7.482e-3 (flat in ℓ) | **0.280** |
| 0.3 | 0.15 | 2.296e-1 | 6.568e-2 (flat in ℓ) | **0.286** |

- **ℓ-saturated.** ℓ_max = 2, 4, 6, 8 agree to < 0.2% in δρ (rel. diff 5e-5 at Σ_H=0.1, 4.5e-4 at
  Σ_H=0.3). Adding moments past the quadrupole changes nothing.
- **~28% capture, ~72% shortfall.** The saturated hierarchy reproduces only ~0.28 of the exact
  collisionless δρ, at BOTH shear amplitudes (the fraction is shear-robust to within 0.007).
- **Null at zero shear:** both paths give δρ = 0 (characteristic residual 3e-8), so the shortfall is
  a genuine shear effect, not a baseline/normalization mismatch.

## Root cause — structural, not truncation

ℓ_max cannot fix it because the deficit is not in the ℓ ladder, and it is a **leading-order**
deficit. `δρ` is a *second-order-in-shear* quantity: its O(S) part vanishes by angular parity
(`Σ_j w_j P₂(μ_j) = 0`), so `δρ ~ S²` — verified, `δρ/S²` is ~constant on both paths (char ≈ 10.9·S²,
even ≈ 2.9·S²). The even-ladder is exact only to **linear order in accumulated shear**
(`rhs_typeI.py`): its monopole source is the *linear* `(Σ_H/5) q ∂_q F_2`
(`typeI_even_ladder_hierarchy.py:62`), whereas the exact characteristic monopole is the *nonlinear*
energy shift `f_0(q e^{2 I_j})` with `I_j` accumulated over the full ray history
(`extract_monopole_jax`). So the hierarchy captures only the part of the leading O(S²) anisotropy
supplied by the linear F₂→F₀ feedback (~27%) and **the capture has a finite S→0 limit ≈ 0.27 (≠ 1)**
— measured down to Σ_H = 0.003. A leading-order limit ≠ 1 rules out the "they agree at linear order
and differ only nonlinearly" alternative: the shortfall is structural at the leading nontrivial
order. This **confirms and explains** the W2-PR6 ~20-28% capture rather than recovering it.

*Independently reviewed (cavecrew-reviewer + pr-test-analyzer): both verified the comparison is
apples-to-apples (matched grid, IC, shear-accumulation S=Σ_H·0.5, and δρ observable) and that the
~28% is real physics, not a units/normalization artifact; the result is f32-vs-f64 invariant.*

## Implications / scope update (auto-managed)

- **The "kill W2-PR6 by higher ℓ" premise is REFUTED.** The even-ladder hierarchy is **not** a
  shear-faithful surrogate at the energy-density level; it is ℓ-saturated at ~28% capture. Higher ℓ
  does not rescue `linearized_pstf`.
- **The characteristic-ray path is the shear-faithful reference** for the transport sector. The
  forward shear response (B3-B5, the Yp anisotropy) should be carried by the characteristic driver,
  not the moment hierarchy — consistent with `transport_mode.CHARACTERISTIC` being the exact,
  no-ℓ-truncation mode.
- **Regression gate landed:** `tests/test_even_ladder_vs_char_collisionless_gate.py` pins the
  ℓ-saturation + the 0.20–0.35 capture band + the zero-shear null, so any future change to the
  even-ladder transport that silently alters this structural fraction is caught.
- **Caveat:** this is the transport-sector δρ analogue of W2-PR6's Yp number (which adds Hubble
  damping + thermo + the weak network via the full characteristic BBN solve). The ~28% δρ capture is
  the same structural shortfall, not the identical Yp quantity. Follow-on (optional): a numpy mirror
  of the four characteristic primitives + a jax-parity test so the gate runs without JAX in CI.
