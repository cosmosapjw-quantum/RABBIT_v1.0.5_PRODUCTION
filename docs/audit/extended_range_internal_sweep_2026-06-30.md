# Extended-range internal sweep (Σ_H → 0.95) — the gate's INTERNAL half, and why it is not validation

**Date:** 2026-06-30
**Purpose:** `GATE_SIGMA_H_TO_0P95` (claim: "Type I anisotropic BBN validated for Σ_H ∈ [0, 0.95]")
named two **internal** test nodes in `tests/test_typeI_extended_range.py` that did not exist. This
records the measurement that wrote them honestly, and why the obvious "constraint residual < 1e-10"
reading would have been **vacuous**.

## The vacuous reading (refuted)

The plausible implementation — assert `friedmann_residual_typeI(Σ₊, Σ₋) < 1e-10` over the sweep — is
a tautology. `rabbit.geometry.constraints.friedmann_residual_typeI` is literally:

```python
Sigma_sq = Sigma_plus**2 + Sigma_minus**2
Omega = max(0.0, 1.0 - Sigma_sq)        # Ω DEFINED as 1 - Σ²
return abs(1.0 - Omega - Sigma_sq)      # ≡ |1 - (1-Σ²) - Σ²| = 0, exactly
```

Its own docstring says "Identically zero by construction." It takes only (Σ₊, Σ₋), not the evolved
solution, so it measures nothing about the integrator at Σ_H = 0.95. `assert 0 < 1e-10` is not a
validation — it is the same self-consistency-as-validation trap the external audit (2026-06-30)
flagged elsewhere (cf. the B2 ~28% and B5 logit-codomain vacuity findings).

## The non-vacuous measurement (CL0, N_q=20, N_μ=12, n_rx=12, scipy characteristic tier-1)

| Σ_H | Yp | D/H | mass-conservation \|ΣXᵢ−1\| | Σ_H>0.75 warning |
|----:|-----:|------:|--------------------------:|:----------------:|
| 0.00 | 0.24234943 | 2.489e-5 | 1.1e-16 | no |
| 0.30 | 0.24401979 | 2.499e-5 | 1.3e-15 | no |
| 0.60 | 0.24840852 | 2.526e-5 | 3.4e-14 | no |
| 0.75 | 0.25565792 | 2.572e-5 | 9.3e-15 | no |
| 0.85 | 0.32449717 | 3.013e-5 | 4.4e-16 | **yes** |
| 0.95 | 0.33128535 | 3.064e-5 | 4.6e-14 | **yes** |

What is genuinely checked (and locked in the two nodes):
- **Solver completes to Σ_H = 0.95** with finite observables (it does not NaN/diverge at the Milne
  edge Ω = 1 − Σ² → 0.0975).
- **Evolved mass-fraction closure** `|Σ Xᵢ − 1|` stays at machine precision (≤ 4.6e-14) across the
  whole range — a real integrator-drift diagnostic (this is `constraint_diag.mass_conservation`, the
  only constraint the driver stores; not the tautological Friedmann residual).
- **The documented Σ_H > 0.75 caution warning** fires above 0.75 and not below — pinning the
  advisory safe boundary (`full_coupled_typeI._SIGMA_SAFE_MAX = 0.75`).
- **Yp is monotone** in shear and **regression-locked** at representative cells.

## Honest scope (what this is NOT)

This is the **internal** half of the gate — self-consistency + regression only. It does **not**
validate Σ_H = 0.95 against nature:
- Above Σ_H = 0.75 the solver itself warns ("Ω → 0 Milne regime ... interpret with caution"), and Yp
  rises steeply there (0.256 → 0.324 across 0.75 → 0.85). Those values are locked against drift but
  are explicitly in the caution regime.
- Mass-conservation closure and a Yp regression lock are self-consistency; a shared-bias error in the
  anisotropic transport would pass both.

So `GATE_SIGMA_H_TO_0P95` stays **RED** even with these two nodes green: its `external_anchor` node
(`tests/test_finite_shear_external_anchor.py`, fail-closed) requires an independent finite-shear
cross-code benchmark that does not exist yet (plan B8). The permitted_text was corrected to name the
**evolved mass-fraction closure** (not the tautological Friedmann residual) as the internal check.

## Scope update (auto-managed)

- **Internal extended-range nodes LANDED** (`tests/test_typeI_extended_range.py`, `slow`): the two
  nodes `GATE_SIGMA_H_TO_0P95` references now exist and pass; the gate's red state is now carried
  SOLELY by the missing external anchor (the honest, intended state).
- **Residual (B8):** unchanged — finite-shear anisotropic BBN has no external cross-code anchor;
  these internal nodes do not change that.
