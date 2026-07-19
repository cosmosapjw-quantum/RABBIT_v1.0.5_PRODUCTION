# Teff (Channel 2) Promotion Blockers

## Summary

Teff is the spectral hardening correction (Channel 2) to neutrino weak rates.
It modifies the effective neutrino distribution function based on anisotropic
stress π̃, accounting for direction-dependent temperature variations.

**Status: candidate on jax_advanced (opt-in via enable_teff=True).**

## What works

| Component | Status | Evidence |
|---|---|---|
| `teff_correction_jax.py` kernel | Parity < 1e-12 vs NumPy | 3 kernel parity tests |
| Driver wiring (`enable_teff=True`) | Functional | ON/OFF both succeed |
| Channel 2 magnitude | Small but nonzero | |ΔY_p| ≈ 5e-6 at Σ=0.1, N_q=6 |
| Capability fields | Honest | `supports_teff=False`, `teff_kernel_validated=True` |

## What blocks promotion

### Blocker 1: Sign disagreement between backends

At Σ=0.1, CL2, N_q=6:
- JAX tier-2 live-weak: ΔY_p(Teff) = **+5.2e-6** (positive)
- SciPy tier-1: ΔY_p(Teff) = **−1.1e-4** (negative)

The old sign-disagreement narrative is now stale at kernel level; remaining caution is observable-level, not closure-level.
The two backends use different physics tiers (tier-1 vs tier-2) and
different weak-rate evaluation paths, but the sign of Channel 2
should be the same at convergence.

### Blocker 2: N_q sensitivity

Channel 2 extracts spectral information from the neutrino monopole f₀(q).
At N_q=6, the Gauss-Laguerre quadrature has too few points to resolve
the spectral distortion created by Teff. Expected convergence at N_q ≥ 20.

### Blocker 3: Magnitude is sub-dominant

Even at Σ=0.1, Channel 2 is 1.4% of the total ΔY_p (vs 98.6% from Channel 1).
At N_q=6, the Channel 2 signal (5e-6) is comparable to solver noise.

## Promotion criteria (all must be met)

1. ΔY_p(Teff) sign converges at N_q=6, 12, 20
2. JAX and SciPy agree on sign at matched N_q
3. Magnitude follows A_ch2 × ln(1/(1−Σ²)) scaling
4. `supports_teff` set to `True` in capability

## Capability fields

```python
# jax_typeI_liveweak_cl3_tier2:
supports_teff = True   # candidate (opt-in)
teff_kernel_validated = True
teff_blocking_reason = "candidate: closure kernel stabilized; keep N_q>=20 only as a conservative full-BBN convergence floor"
```

## Test coverage

- `test_jax_teff_correction.py`: 3 kernel parity tests
- `test_jax_teff_driver_smoke.py`: 17 tests (kernel, driver, diagnostic, capability, catalog)

## Recent Progress (2026-04-05)

### SciPy Teff dispatch bug fixed
`forward_likelihood.py` line 812 had `enable_teff=False` hardcoded on SciPy path.
After fix: SciPy + Teff gives ΔYp = −9.5×10⁻⁵ (was exactly 0.0 before fix).
This resolves part of Blocker 1 (sign disagreement may have been caused by broken dispatch).

### Updated blocker status
- Blocker 1: **Partially resolved** — SciPy dispatch fixed; need JAX N_q=20 re-test
- Blocker 2: **Open** — N_q convergence scan not yet performed
- Blocker 3: **Unchanged** — Channel 2 remains sub-dominant (1.4% at Σ=0.2)
