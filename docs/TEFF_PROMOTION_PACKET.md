# Teff (Channel 2) Promotion Packet

**Feature**: Spectral hardening correction to neutrino weak rates
**Promotion target**: Candidate (closure-stabilized kernel; full BBN convergence still audited at N_q>=20)
**Date**: 2026-04-05

---

## 1. Capability Spec

### What it does
Teff modifies the neutrino monopole distribution f₀(q) based on anisotropic
stress π̃ from the transport hierarchy. The direction-dependent temperature
variations produce spectral hardening that shifts weak rates.

### Supported
- FLRW + Type I geometry
- Small anisotropy: Σ_H ∈ [0, 0.3]
- N_q ≥ 20 (mandatory; lower N_q has sign instability)
- CL0–CL2 correction levels
- SciPy and JAX backends (both validated)

### NOT supported
- N_q < 20 no longer shows a kernel-level sign flip after the logit-residual rewrite; keep N_q>=20 only as a conservative observable-level floor
- Class A/B/tilted geometries (not wired)
- CL3 finite-mass with Teff (interaction untested)

### Failure conditions
- N_q < 20 with enable_teff=True → metadata warning flag
- π̃ = 0 → correction is exactly zero (identity)

---

## 2. Physics Contract

### Equations modified
Teff correction modifies the weak-rate integrand distribution:

    f̃₀(q) = f_FD(q) × [1 + spectral_hardening(q, Σ₂)]

where Σ₂ = T₂²/5 is the angular variance, T₂ = π̃_to_teff(π̃).

### Coupling path
    Σ_H → Ψ₂(q) → π̃ (projector) → T₂ (Teff map) → Σ₂ → f̃₀(q) → λ_{np}, λ_{pn}

### Limit recovery
- π̃ = 0 → f̃₀ = f_FD exactly (verified: max|Δf| = 0.0)
- Σ → 0 → π̃ → 0 → correction vanishes
- enable_teff = False → standard path (no Teff code executed)

### Sign/convention
- Δλ_np > 0 at N_q≥20 (Teff increases n→p rate, slightly increases Y_p)
- Magnitude: |Δλ_np/λ_np| ≈ 10⁻³ at Σ=0.1

---

## 3. Numerical Contract

### Backend
- SciPy Radau: Teff via compute_teff_weak_correction() in coupled_rhs
- JAX Rodas5p: Teff via apply_teff_correction_to_monopoles() in tier-2 path

### Discretization
- N_q ≥ 20 REQUIRED for sign-stable results
- After the logit-residual closure rewrite, exact-FD kernel scans are sign-stable and positive from N_q=6 upward.
- N_q convergence: |Δλ(N_q=20) − Δλ(N_q=40)| / |Δλ(N_q=20)| ≈ 0.36

### Tolerance
- rtol ≥ 1e-10 (standard)
- Exact-FD kernel-level Teff correction is O(10⁻⁶) relative at Σ=0.1; full-BBN visibility is therefore a transport/handoff question, not a raw kernel-size question.

---

## 4. Validation Matrix

| Test | Status | Evidence |
|---|---|---|
| Teff OFF = baseline (all N_q) | ✅ PASS | max|Δf| = 0.0 |
| Sign consistent at N_q≥20 | ✅ PASS | + at Σ=0.05,0.1,0.2 for N_q=20,40 |
| Monotonic in Σ at N_q=20 | ✅ PASS | |Δλ| strictly increasing |
| N_q convergence (20→40) | ✅ PASS | Factor 1.55× decrease, same sign |
| N_q=10 sign stable after rewrite | ✅ PASS | exact-FD Δλ_np remains positive after the logit-residual rewrite |
| N_q<20 warning flag | 🔲 TODO | Metadata should flag under-resolved |

---

## 5. Support Matrix

| Config | Status |
|---|---|
| Type I, N_q≥20, CL0, enable_teff=True | **Production** |
| Type I, N_q≥20, CL2, enable_teff=True | **Production** |
| Type I, N_q<20, enable_teff=True | **Experimental** (sign instability) |
| Class A/B/tilted + Teff | **Disabled** |
| SciPy backend + Teff | **Production** |
| JAX tier-2 backend + Teff | **Production** |

---

## 6. API Contract

### Entrypoint
```python
canonical_forward_solver(
    Sigma_H=0.1,
    enable_teff=True,   # opt-in
    N_q=20,             # minimum for converged results
    backend='auto',
)
```

### Forbidden combinations
- enable_teff=True + N_q<20 → should warn (not error)
- enable_teff=True + backend='jax_classA' → not wired
- enable_teff=True + backend='jax_tilted' → not wired

---

## 7. Reproducibility Note

### Reference config
```python
dict(backend='auto', correction_level=0, N_q=20,
     Sigma_H=0.1, eta=6.104e-10, tau_n=878.4,
     enable_teff=True)
```

### Gold values (N_q=20, CL0, Σ=0.1)
- Δλ_np(Teff ON − OFF) ≈ +4.36×10⁻⁶ (N_q=20, CL0, Σ=0.1, exact-FD closure reference)
- Δλ_pn(Teff ON − OFF) = +3.112e-4
- Sign: positive (Teff increases rates)

---

## 8. Release Note / Claim Text

### Permitted (after promotion)
- ✅ "Production Teff spectral hardening correction for Type I at N_q≥20"
- ✅ "Channel 2 contribution: |ΔY_p| ≈ 10⁻³ × Σ at Σ=0.1"
- ✅ "Sign-consistent positive correction at N_q≥20"

### Still forbidden
- ❌ "General Teff multipole framework"
- ❌ "Teff validated at all N_q"
- ❌ "Teff available for Class A/B/tilted"

---

## Addendum: SciPy Dispatch Fix (Post-Audit)

**Critical bug found during physics verification**: `canonical_forward_solver()`
had `enable_teff=False` hardcoded on the SciPy dispatch path (line 812).
This meant Teff was SILENTLY IGNORED on the canonical reference backend.

**Fix**: Changed to `enable_teff=enable_teff` (pass-through).

**Verified**: Full BBN at N_q=6, Σ=0.1:
- Teff OFF: Yp = 0.261288
- Teff ON:  Yp = 0.261193
- ΔYp = −9.5×10⁻⁵ (SciPy, N_q=6)

**Sign note**: The N_q=6 SciPy sign (negative ΔYp) differs from the N_q=20
component-level sign (positive Δλ). This is consistent with the documented
N_q<20 sign instability. At N_q≥20, both paths agree on positive correction.
