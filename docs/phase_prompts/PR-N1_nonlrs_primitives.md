# PR-N1 — Non-LRS Ray Grid Primitives (Phase Prompt)

> Feed verbatim.  Framework: [README.md](README.md).

---

## 0. Load-bearing project context

RABBIT supports LRS Bianchi Type I (`Σ_- = 0`) today.  Generic
Type I has two shear amplitudes and the ray direction lives on the
full sphere S² instead of `μ ∈ [-1, 1]`.  This PR adds the S²
quadrature + non-LRS forward direction map **without** wiring them
into any driver.  Driver integration is PR-N2.

Invariants: Rodas5P, CPU-preferred, float64, transported monopole
in weak rates, publication tolerance 5e-5.

Dependencies: none.  Can land in parallel with PR-A / PR-J.

---

## 1. Phase objective

Add `src/rabbit/jax/characteristic_rays_nonlrs_jax.py` containing:
- S² tensor-product quadrature (`N_θ × N_φ`).
- Non-LRS direction forward map `(μ₀, φ₀) → (μ(N), φ(N))` using
  two accumulated shear integrals `S_+(N), S_-(N)`.
- S²-weighted stress extractors for both Π_+ and Π_-.
- S²-weighted monopole extractor.
- LRS-reduction unit test (`N_φ = 1, Σ_- = 0` reproduces the LRS
  primitives in `characteristic_rays_jax.py` to 1e-10).

No driver change.  The new primitives are pure functions.

---

## 2. Literature anchors

### 2.1 Internal
- Paper §2.1–§2.2: Wainwright–Hsu variables `(Σ_+, Σ_-)`.
- Paper §3.2: PSTF multipole expansion for generic Type I (two
  stress components Π_+, Π_-).
- Paper §6.2: geodesic equation in Bianchi I (LRS derivation; must
  be generalised here).
- Existing LRS primitives:
  `src/rabbit/jax/characteristic_rays_jax.py`.
- Existing non-LRS PSTF implementation on the SciPy side (for the
  linearised path):
  `src/rabbit/transport/typeI_hierarchy.py` (generic n_ell=3).

### 2.2 External
- Wainwright & Ellis, *Dynamical Systems in Cosmology* (Cambridge
  1997), chapter on Bianchi Type I — specifically the covariant
  shear evolution with two eigenvalues.
- Spherical harmonic quadrature: Lebedev–Laikov grids
  (Lebedev & Laikov 1999, Dokl. Math.) for future optimisation.
  Not required for PR-N1 but noted in §11.

### 2.3 Paper-equation cross-check
- [ ] Eq 1: metric `ds² = -dt² + a_1² dx² + a_2² dy² + a_3² dz²`.
- [ ] Eq 2–3: `Σ_±` definitions in Wainwright–Hsu variables.
- [ ] Eq 6–7: Friedmann constraint `Ω = 1 - Σ_+² - Σ_-²`, anisotropic
      Hubble `H = H_FLRW / √(1 - Σ²)`.
- [ ] Eq 46–48: LRS geodesic equations — generalise to include
      `Σ_-` by replacing `Σ_+ · diag(1, 1, -2)` with
      `diag(Σ_+ + √3 Σ_-, Σ_+ - √3 Σ_-, -2 Σ_+)`.

---

## 3. Skeleton code

### 3.1 Quadrature

```python
# src/rabbit/jax/characteristic_rays_nonlrs_jax.py
"""Non-LRS Bianchi Type I characteristic-ray primitives.

State-vector consequence: each ray is a (θ_j, φ_k) pair on S².
Total rays = N_θ × N_φ.  Stress tensor on the Type-I basis has two
Hubble-normalised amplitudes (Π_+, Π_-); both are extracted here.
"""
from __future__ import annotations
from functools import lru_cache
import jax
import jax.numpy as jnp
import numpy as np
from numpy.polynomial.legendre import leggauss

jax.config.update("jax_enable_x64", True)


@lru_cache(maxsize=8)
def setup_ray_grid_S2(N_theta: int, N_phi: int):
    """Tensor-product Gauss–Legendre(θ) × uniform-midpoint(φ) grid.

    Returns (flattened):
        mu0     : (N_theta * N_phi,) initial direction cosines cos θ_0
        phi0    : (N_theta * N_phi,) initial azimuthal angles
        w_s2    : (N_theta * N_phi,) solid-angle weights, ∑ w = 4π/4π = 1
        X0      : (N_theta * N_phi,) μ₀² / (1 - μ₀²)  — precomputed
        signs   : (N_theta * N_phi,) sign(μ₀); zeros clamped to +1
    """
    mu0_1d, w_mu_1d = leggauss(int(N_theta))
    # midpoint φ grid so ∫₀^{2π} dφ = Σ w_φ · 2π/N_phi
    phi0_1d = (np.arange(int(N_phi)) + 0.5) * (2.0 * np.pi / int(N_phi))
    mu0 = np.broadcast_to(mu0_1d[:, None], (int(N_theta), int(N_phi))).reshape(-1)
    phi0 = np.broadcast_to(phi0_1d[None, :], (int(N_theta), int(N_phi))).reshape(-1)
    # solid-angle weight per node: (w_μ) · (1/N_φ); factor of 2π cancels
    # in the angle-averaged extractors (they divide by 4π).
    w_s2_grid = (w_mu_1d[:, None] * (1.0 / int(N_phi)))
    w_s2 = np.broadcast_to(w_s2_grid, (int(N_theta), int(N_phi))).reshape(-1)
    X0 = mu0**2 / np.maximum(1.0 - mu0**2, 1e-30)
    signs = np.where(mu0 >= 0.0, 1.0, -1.0)
    return (
        jnp.asarray(mu0, dtype=jnp.float64),
        jnp.asarray(phi0, dtype=jnp.float64),
        jnp.asarray(w_s2, dtype=jnp.float64),
        jnp.asarray(X0, dtype=jnp.float64),
        jnp.asarray(signs, dtype=jnp.float64),
    )
```

### 3.2 Direction forward map

The direction unit vector `ê = (sin θ cos φ, sin θ sin φ, cos θ)`
evolves under
```
d ê^a / dN = -(σ^a_b + H/H) ê^b + (σ_bc ê^b ê^c) ê^a
```
where `σ^a_b` in Wainwright–Hsu basis has eigenvalues
`(Σ_+ + √3 Σ_-, Σ_+ - √3 Σ_-, -2 Σ_+)`.  Integrating this with two
accumulated shear integrals `S_±(N) = ∫₀ᴺ Σ_±(N') dN'` gives a
rotation + stretching applied to the initial direction ê₀.

```python
def mu_phi_current_jax(
    mu0: jnp.ndarray, phi0: jnp.ndarray,
    S_plus: jnp.ndarray, S_minus: jnp.ndarray,
) -> tuple[jnp.ndarray, jnp.ndarray]:
    """Forward map (μ₀, φ₀) → (μ(N), φ(N)) for non-LRS Type I.

    Strategy: build the initial unit vector in Wainwright–Hsu
    basis, apply the exponential of the (dimensionless) shear
    generator diag(S_+ + √3 S_-, S_+ - √3 S_-, -2 S_+), renormalise,
    and recover (μ, φ).  This is exact for constant ratio
    Σ_- / Σ_+ but approximate when the ratio changes in time;
    BBN dynamics keep the ratio nearly constant over the relevant
    e-folds so the error is bounded.

    Returns (μ, φ) arrays with shapes matching the inputs.
    """
    sin_t = jnp.sqrt(jnp.maximum(1.0 - mu0 * mu0, 0.0))
    ex0 = sin_t * jnp.cos(phi0)
    ey0 = sin_t * jnp.sin(phi0)
    ez0 = mu0
    # Scale factors on each axis grow as exp(-σ_i·N) where σ_i are
    # the eigenvalues of the shear tensor; the direction unit vector
    # rescales inversely — paper eq 45 generalised:
    a_x = jnp.exp(-(S_plus + jnp.sqrt(3.0) * S_minus))
    a_y = jnp.exp(-(S_plus - jnp.sqrt(3.0) * S_minus))
    a_z = jnp.exp(2.0 * S_plus)
    ex = ex0 * a_x
    ey = ey0 * a_y
    ez = ez0 * a_z
    norm = jnp.sqrt(ex * ex + ey * ey + ez * ez)
    ex /= norm; ey /= norm; ez /= norm
    mu = ez
    phi = jnp.arctan2(ey, ex)
    return mu, phi
```

**CRITICAL note (Stage-3 CoT must validate):** the above assumes
`S_±` are accumulated integrals of the *current* Σ_±.  The exact
integration of the non-LRS direction ODE requires ordering of
operators — since the shear eigenvalues commute (all along the
fixed Wainwright–Hsu basis in orthogonal Type I), the exponential
map is exact.  Verify this explicitly in Stage 3.

### 3.3 Stress extractors

```python
def extract_stress_plus_S2(
    I: jnp.ndarray, J: jnp.ndarray, mu: jnp.ndarray,
    w_s2: jnp.ndarray, f_nu: float,
) -> jnp.ndarray:
    """Π_+ from S² quadrature.  LRS limit: reduces to paper eq 57."""
    P2 = 0.5 * (3.0 * mu * mu - 1.0)
    return f_nu * jnp.sum(w_s2 * J * P2 * jnp.exp(-8.0 * I))


def extract_stress_minus_S2(
    I: jnp.ndarray, J: jnp.ndarray, mu: jnp.ndarray, phi: jnp.ndarray,
    w_s2: jnp.ndarray, f_nu: float,
) -> jnp.ndarray:
    """Π_- from the (|m|=2) harmonic, Y_2^{±2} ∝ sin²θ cos(2φ).

    Normalisation matches paper eq 14 for the generic-Type-I PSTF
    quadrupole; the √3/2 factor comes from the canonical
    Y_2^{±2} / P_2 linkage.
    """
    ang = (1.0 - mu * mu) * jnp.cos(2.0 * phi)
    return f_nu * (jnp.sqrt(3.0) / 2.0) * jnp.sum(
        w_s2 * J * ang * jnp.exp(-8.0 * I)
    )
```

### 3.4 Monopole extractor (unchanged form, S² weights)

```python
def extract_monopole_S2(
    I: jnp.ndarray, J: jnp.ndarray, w_s2: jnp.ndarray,
    q_nodes: jnp.ndarray,
) -> jnp.ndarray:
    """f̃₀(q) with S² quadrature.  Paper eq 58 generalised.

    At LRS (N_φ=1), w_s2 = w_mu reproduces the 1D form.
    """
    alpha = jnp.exp(2.0 * I)                  # (N_rays,)
    qa = q_nodes[:, None] * alpha[None, :]     # (N_q, N_rays)
    f_vals = 1.0 / (jnp.exp(jnp.minimum(qa, 500.0)) + 1.0)
    # Note: in the LRS extractor the factor ½ appears because
    # ∫_{-1}^1 dμ = 2 while the quadrature weights sum to 2; here
    # w_s2 is already normalised so ∑ w_s2 = 2 (Gauss–Legendre on
    # μ sums to 2) and φ-average divides by 1.  Therefore the
    # overall ½ prefactor matches the LRS form when N_φ = 1.
    return 0.5 * f_vals @ (w_s2 * J)
```

### 3.5 LRS-reduction unit test

```python
# tests/test_pr_n1_nonlrs_primitives.py
import pytest
pytest.importorskip("jax")
import jax; jax.config.update("jax_enable_x64", True)
import jax.numpy as jnp
import numpy as np
from rabbit.jax.characteristic_rays_jax import (
    mu_current_jax, extract_stress_jax, extract_monopole_jax,
)
from rabbit.jax.characteristic_rays_nonlrs_jax import (
    setup_ray_grid_S2, mu_phi_current_jax,
    extract_stress_plus_S2, extract_stress_minus_S2, extract_monopole_S2,
)

def test_lrs_reduction_stress():
    """N_φ=1 grid with Σ_-=0 must reproduce extract_stress_jax."""
    N_theta, N_phi = 12, 1
    mu0, phi0, w_s2, X0, signs = setup_ray_grid_S2(N_theta, N_phi)
    Sigma_plus = 0.1
    S_plus = Sigma_plus * 3.0   # after 3 e-folds, constant shear
    S_minus = 0.0
    mu, phi = mu_phi_current_jax(mu0, phi0, S_plus, S_minus)
    I = jnp.zeros_like(mu0)
    J = jnp.ones_like(mu0)
    Pi_ref_2d = extract_stress_plus_S2(I, J, mu, w_s2, 0.4052)
    # LRS 1D reference:
    mu_1d, w_mu_1d = np.polynomial.legendre.leggauss(N_theta)
    X0_1d = mu_1d**2 / (1 - mu_1d**2 + 1e-30)
    signs_1d = np.sign(mu_1d)
    mu_ref = mu_current_jax(jnp.asarray(X0_1d), jnp.asarray(signs_1d), S_plus)
    I_ref = jnp.zeros_like(mu_ref); J_ref = jnp.ones_like(mu_ref)
    Pi_ref_1d = extract_stress_jax(I_ref, J_ref, mu_ref, jnp.asarray(w_mu_1d), 0.4052)
    assert abs(float(Pi_ref_2d) - float(Pi_ref_1d)) < 1e-10

def test_lrs_reduction_monopole():
    ...

def test_pi_minus_vanishes_at_Sigma_minus_zero():
    """When Σ_- = 0 throughout, any valid Π_- formula must vanish
    in the N_φ → ∞ limit (spherical symmetry in the (x,y) plane).
    At N_φ = 16 with Σ_- = 0 we require Π_- < 1e-10.
    """
    ...

def test_swap_symmetry():
    """Swapping (Σ_+, Σ_-) ↔ (Σ_+, −Σ_-) rotates φ → π−φ.
    The solid-angle-averaged Y_p prediction is invariant."""
    ...
```

---

## 4. WBS

1. **Module skeleton** `characteristic_rays_nonlrs_jax.py`.
2. **`setup_ray_grid_S2` + tests** (weights sum correctly).
3. **`mu_phi_current_jax` + tests** (LRS reduction + direction-map
   numerical integration cross-check).
4. **Stress extractors + monopole extractor + LRS tests**.
5. **Pi_- vanishing at Σ_-=0 test**.
6. **Swap-symmetry test**.
7. **Documentation update**: cross-reference the new module from
   [IMPLEMENTATION_GUIDE_NON_LRS_TYPEI.md](../IMPLEMENTATION_GUIDE_NON_LRS_TYPEI.md).

---

## 5. Three-stage verification

### Stage 1 — Internal
- Read paper §2.1, §3.2, §6.2 (pages 8–9, 10–11, 16–17).
- Read `src/rabbit/transport/typeI_hierarchy.py` to learn how the
  SciPy generic-n_ell=3 path extracts Π_-.
- Confirm paper eq 14 includes both `Π_+` and `Π_-`.

### Stage 2 — External
- `WebSearch "Bianchi Type I geodesic equation Wainwright Ellis
  direction cosine"` — cross-check the direction-equation form.
- `WebSearch "spherical harmonic Y_2^2 real sin^2 theta cos 2 phi
  normalisation"` — confirm the √3/2 factor in `Y_2^{±2}`.

### Stage 3 — Self CoT
- Derive `mu_phi_current_jax` starting from the full 3-vector
  stretching.  Verify operator commutativity (shear eigenvalues
  diagonal in Wainwright–Hsu basis, they commute).
- Dimensional check: every extractor returns dimensionless
  `Π` (it is `f_ν · numerical`, both dimensionless).
- LRS limit: substitute `Σ_- = 0, N_φ = 1` and recover the LRS
  primitives exactly.
- Pi_- vanishing: substitute `Σ_- = 0` everywhere — the rotation
  around the z-axis is the identity, so `Π_-` integrand averages
  to zero over φ.

Record in `docs/audit/PR-N1_stage{1,2,3}.md`.

---

## 6. Self-audit checklist
- [ ] Quadrature weight sum test.
- [ ] LRS reduction tests (stress + monopole).
- [ ] Π_- vanishes at Σ_-=0 (test #3).
- [ ] Swap-symmetry test.
- [ ] No driver changes (module is additive only).
- [ ] IMPLEMENTATION_GUIDE_NON_LRS_TYPEI.md updated.

---

## 7. Adversarial audit prompt

> Audit PR-N1 (non-LRS primitives).  Verify: (1) direction forward
> map recovers the LRS formula bitwise at Σ_-=0, (2) the √3/2
> normalisation in Π_- matches the paper's `Y_2^{±2}` convention,
> (3) no existing driver path is broken (the new module is
> additive).  Report issues ranked by severity.  Cap 400 words.

---

## 8. Anti-local-minimum reminders

1. Do not wire the new primitives into any driver — PR-N2 does that.
2. Do not substitute a Lebedev grid — keep the tensor-product
   grid for the LRS-reduction test to be meaningful.
3. If the direction forward map cannot be verified analytically,
   check numerically: integrate `d ê/dN` with `solve_ivp` for 5
   random `(Σ_+, Σ_-)` pairs and compare to `mu_phi_current_jax`.

---

## 9. Hallucination prevention
- Do not invent a `jnp.lebedev` helper — it does not exist.
- Do not cite a paper equation without reading the PDF page.
- Do not assume the `√3/2` factor; derive it from
  `Y_2^{2}(θ, φ) = √(15/(32π)) sin²θ cos(2φ)` and the real-form
  PSTF convention used in the paper.

---

## 10. Documentation updates

### 10.1 `docs/ROADMAP_STATE_OF_RECORD.md §5.1`
Add `src/rabbit/jax/characteristic_rays_nonlrs_jax.py` to the file
inventory.

### 10.2 `docs/ROADMAP_PR_CATALOG.md`
Append PR-N1 entry with status "merged" and pointer to PR-N2.

### 10.3 `docs/IMPLEMENTATION_GUIDE_NON_LRS_TYPEI.md §4 Phase 1`
Transition status from "planned" to "delivered in PR-N1".

---

## 11. Deferred optimisations (for future PRs, not PR-N1)

- Replace tensor-product grid with Lebedev at the same ℓ_max.
  Benchmark when PR-N2 is in place.
- Pre-compute `exp(-(S_+ + √3 S_-))` etc. as vectorised constants
  to avoid re-evaluation on every RHS call (low priority;
  negligible cost).

---

## 12. Deterministic commit script

```bash
set -euo pipefail
cd /home/cosmosapjw/Dropbox/rabbit/RABBIT_v1.0.5_PRODUCTION
source venv/bin/activate

pytest tests/test_pr_n1_nonlrs_primitives.py -v
pytest tests/test_jax_typeI_characteristic_parity.py -v    # must stay green
pytest tests/test_jax_typeI_characteristic_tier2.py -v
pytest tests/ -m "not slow and not gpu" --tb=no -q

test -f docs/audit/PR-N1.md
test -f docs/audit/PR-N1_stage1.md
test -f docs/audit/PR-N1_stage2.md
test -f docs/audit/PR-N1_stage3.md
git add docs/audit/PR-N1*.md
git diff --cached --name-only | grep -q ROADMAP_STATE_OF_RECORD.md
git diff --cached --name-only | grep -q ROADMAP_PR_CATALOG.md

git commit -m "$(cat <<'EOF'
PR-N1: non-LRS Bianchi Type I characteristic-ray primitives

Adds src/rabbit/jax/characteristic_rays_nonlrs_jax.py with S^2
tensor-product (N_theta x N_phi) quadrature, the non-LRS direction
forward map (mu, phi) via accumulated shear integrals (S_+, S_-),
and S^2-weighted stress extractors for both Pi_+ and Pi_- plus the
monopole extractor f_tilde_0(q). LRS reduction to the existing
characteristic_rays_jax primitives verified bitwise at N_phi=1,
Sigma_- = 0. No driver wiring yet; PR-N2 integrates.

Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>
EOF
)"
```

---

## 13. Abort conditions

- LRS reduction fails at N_φ=1 beyond 1e-10.
- Π_- does not vanish at Σ_-=0 (after adequate N_φ).
- Stage 3 CoT finds a non-commuting operator ordering that
  invalidates the exponential direction map.

Abort → `docs/audit/PR-N1_abort.md`.
