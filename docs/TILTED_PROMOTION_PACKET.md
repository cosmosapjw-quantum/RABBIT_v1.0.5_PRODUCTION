# Tilted BBN Promotion Packet

**Feature**: Principal-axis tilted velocity extension for Bianchi BBN
**Promotion target**: Candidate (principal-axis scalar, Type I baseline recovery verified)
**Date**: 2026-04-05

---

## 1. Capability Spec

### Supported
- Principal-axis scalar tilt (axis 1, 2, or 3 in the diagonal WH frame)
- All Bianchi labels through the reduced tilted JAX runner; Type I remains the strongest baseline slice
- Diagonal non-LRS reduced transport: public tilted runs select n_ell=3 when Σ₋ is nonzero or tilt axis is 1/2
- CL0-CL3 weak rates; CL3 defaults to scalar finite-mass/recoil/weak-magnetism on the live monopole
- Small-tilt public regime: |v₀| ≤ 10⁻³, with v₀ ≲ 10⁻⁷ the conservative BBN-memory regime
- Expansion-history modification H/√(1-v²), with opt-in stress-energy T00 closure
- Opt-in tilted perfect-fluid anisotropic stress feedback
- Opt-in algebraic G0i heat-flux closure
- Opt-in Lorentz-boosted FD weak-rate monopole in the plasma frame
- Opt-in CL3 principal-axis angular weak kernel using boosted FD f0/f1/f2 moments

### NOT supported
- Mixed-axis vector tilt with off-diagonal shear/frame states
- Full m-decomposed curved PSTF transport
- Dynamical heat-flux / dipole Boltzmann transport
- General angle-dependent weak collision solver or dynamically evolved angular weak moments
- Teff + tilt combined

### Physics path
Tilt can enter through four explicitly separated channels:
  v → modified H(T) via legacy Γ or opt-in normal-frame T00 closure → shifted freeze-out → ΔY_p
  v² → tilted perfect-fluid Π₊/Π₋ normalized by the selected Hubble closure → shear RHS
  v → boosted FD monopole in the plasma frame → live weak-rate functional
  CL3 → scalar finite-mass/recoil/weak-magnetism weak budget on that live monopole
  CL3 + opt-in angular kernel → boosted FD f0/f1/f2 → K0/K1/K2 weak-kernel projection
  Σ₋ + n_ell=3 → Ψ₋ quadrupole → Π₋ neutrino stress → Σ₋ shear RHS
The angular-kernel path is a principal-axis finite-mass projection, not a full angular collision solver.

---

## 2. Physics Contract

### Equations modified
- Tilt evolution: dv/dN = v(1−v²)/G × [(2−q)(1−(γ−1)v²)/G + λ₃ − (3γ−4)]
- Hubble correction: legacy H→ΓH, or opt-in H²→H²[1+γΓ²v²] from tilted-fluid T00
- Stress/G0i normalization: opt-in T00 closure uses Ω_fluid=Ω/[1+γΓ²v²]
- Principal-axis stress: π_ab/(3H²)=γΩΓ²(v_a v_b-v²δ_ab/3)
- Diagonal non-LRS transport: dΨ₋/dN = -(8/15)Σ₋ - DΨ₋ in the reduced PSTF closure
- Optional weak boost: f0(q)=1/2∫dμ [exp(Γ q(1+v μ))+1]^-1
- Optional CL3 angular weak boost: <K f>=K0 f0 + K1 f1/3 + K2 f2/5 through l≤2
- CL3 weak budget: Coulomb+Sirlin+scalar finite-mass/recoil/weak-magnetism integrand corrections
- ΔN_eff ∝ v² (tilt-equivalent N_eff shift)
- γ_crit = 14/9 (tilt grows for γ < γ_crit, i.e. radiation era)

### Limit recovery
- v = 0 → tilt_rhs = 0 exactly ✅
- v = 0 → hubble_correction = 1.0 exactly ✅
- v = 0 → ΔN_eff = 0 exactly ✅
- v = 0 → weak rates unchanged ✅

---

## 3. Numerical Contract

- Backend: JAX Rodas5P (via jax_tilted dispatch)
- v₀ ≲ 10⁻⁷ for BBN-validated regime (v₀ > 10⁻⁵ is exploratory)
- Causality bound: v < 1 enforced in TiltState
- State variable: v is positive-definite via construction

---

## 4. Validation Matrix

| Test | Status |
|---|---|
| v=0 geometry recovery | ✅ |
| v=0 Hubble correction = 1 | ✅ |
| v=0 ΔN_eff = 0 | ✅ |
| v=0 weak rates unchanged | ✅ |
| Small-v perturbative | ✅ |
| Smooth near v=0 | ✅ |
| Tilt+shear coupling | ✅ |
| Eigenvalue = +1 (growth) | ✅ |
| Principal-axis stress signs | ✅ |
| n_ell=3 diagonal Π₋ transport feedback | ✅ |
| Algebraic G0i heat-flux closure | ✅ |
| Boosted FD monopole v=0 recovery | ✅ |
| Boosted FD f0/f1/f2 v=0 recovery | ✅ |
| CL3 angular kernel isotropic limit → scalar CL3 | ✅ |

---


## v₀ Observable Envelope (Type I, Σ_H=0.1, CL0, N_q=6)

| v₀ | Yp | ΔYp | ΔYp/Yp | Status |
|---|---|---|---|---|
| 0 | 0.2423892198 | — | — | recovery baseline |
| 1e-9 | 0.2423892198 | +6.3e-12 | +2.6e-11 | solver noise |
| 1e-7 | 0.2423892198 | +2.9e-12 | +1.2e-11 | solver noise |
| 1e-5 | 0.2423892780 | +5.8e-8 | +2.4e-7 | **observable threshold** |
| 1e-3 | 0.2429474662 | +5.6e-4 | +2.3e-3 | **physically significant** |

Observable regime: v₀ ≳ 10⁻⁵ for ΔYp above solver noise.
At v₀ = 10⁻³: tilt shifts Yp by +0.23%, comparable to moderate shear.
BBN requires v₀ ≲ 10⁻⁷ (memory constraint); v₀ = 10⁻³ is exploratory.

## 5. Support Matrix

| Config | Status |
|---|---|
| Type I, principal-axis scalar, v₀≤10⁻⁷, CL0-CL2 | **Candidate strong slice** |
| Principal-axis stress / n_ell=3 Π₋ / weak boost / CL3 angular K_l / algebraic G0i closure | **Opt-in candidate** |
| v₀>10⁻⁵ | **Experimental response regime** |
| Mixed-axis vector tilt | **Disabled** |
| Dynamical heat-flux transport | **Disabled** |

---

## 6–8. API / Reproducibility / Claims

### Entrypoint
```python
canonical_forward_solver(backend='jax_tilted', v0=1e-7, Sigma_H=0.1)
```

### Permitted claims
- ✅ "Principal-axis tilted BBN candidate surface with Type I baseline recovery"
- ✅ "Tilt grows during radiation (eigenvalue +1), suppressed at O(v²)"
- ✅ "Opt-in stress-energy T00 Hubble closure modifies the tilted expansion history"
- ✅ "Diagonal n_ell=3 non-LRS quadrupole feedback sources Π₋ from Σ₋ in the reduced tilted transport path"
- ✅ "CL3 scalar finite-mass weak budget runs on the tilted live-monopole path"
- ✅ "Opt-in boosted-FD weak monopole modifies weak rates in the tilted plasma frame"
- ✅ "Opt-in CL3 principal-axis K_l angular weak-kernel projection couples boosted FD f0/f1/f2 to weak rates"

### Forbidden claims
- ❌ "Full tilted Bianchi BBN"
- ❌ "Full vector-tilt BBN"
- ❌ "Full m-decomposed curved PSTF hierarchy"
- ❌ "Dynamical heat-flux Boltzmann hierarchy"
- ❌ "Full angle-dependent weak collision solver"
- ❌ "Full unqualified tilted Bianchi support for all types"
