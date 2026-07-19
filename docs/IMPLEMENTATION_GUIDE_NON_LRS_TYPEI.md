# Implementation Guide — Non-LRS Bianchi Type I (generic Σ₋ ≠ 0)

**Scope:** Lift the JAX characteristic-ray Type I BBN driver from the LRS
restriction (Σ_- = 0, axial symmetry) to the **generic orthogonal
Type I** case with two independent Hubble-normalised shear amplitudes
Σ_+, Σ_-.  The SciPy path already tracks Σ_- in its geometry layer but
the characteristic transport is LRS-only; the JAX linearised PSTF path
supports generic Type I at ``n_ell=3``.  The exact characteristic
method is now available in JAX as the explicit candidate backend
``jax_characteristic_nonlrs`` at tier-1/tier-2 thermodynamics scope
with collisionless public non-LRS transport.  A private CPU-JAX/Rodas5P
staging helper now carries the explicit S² per-species residual state needed
for non-LRS anisotropic residual relaxation, but it is not public dispatch and
does not enable collision-coupled production claims.

**Physics foundation:** RABBIT report §2.1–§2.2 (Wainwright–Hsu
variables, Friedmann constraint), §3.2 (PSTF multipole expansion in
generic Type I), §4.1 (coupled eigenvalue system).

---

## 1. Why LRS is not enough

In the LRS limit the neutrino distribution is axisymmetric about the
single distinguished direction.  The characteristic ray state per
direction is a **single** direction cosine `μ = cos θ ∈ [-1, 1]`, and
the whole transport sector collapses to 2 N_μ + 1 scalar ODEs.
Generic Type I has two preferred axes (the two Σ eigenvectors), so a
ray's direction is a full point on S² parameterised by
`(μ, φ) = (cos θ, azimuth)`.

Two consequences that cascade through the driver:

1. **Angular quadrature becomes two-dimensional.**  A Gauss–Legendre
   grid on `μ ∈ [-1, 1]` is insufficient — every ray-space integral
   (stress Π_±, monopole f̃₀, quadrupole f̃₂, weak-rate monopole)
   becomes a spherical integral over S².
2. **The PSTF quadrupole splits.**  In LRS the only nonzero ℓ=2 mode
   is `Ψ_2^0` (axisymmetric).  In generic Type I, `Ψ_2^{±2}` also
   source and feedback, giving two independent stress components
   Π_+, Π_-.  The geometry RHS now reads (paper eq 4, paper eq 147)

   ```
   dΣ_+/dN = -(2-q) Σ_+ + Π_+,      dΣ_-/dN = -(2-q) Σ_- + Π_-,
   q = 1 + Σ_+² + Σ_-²
   ```

   with Π_- *no longer* identically zero.

---

## 2. Target physics (paper references)

### 2.1 Geometry (unchanged)

The Wainwright–Hsu evolution in §2.1 already treats (Σ_+, Σ_-) as
coequal variables; the Friedmann constraint is `Ω = 1 - Σ² =
1 - (Σ_+² + Σ_-²)` (paper eq 6).  No change needed to the geometry
RHS once Π_- is correctly supplied.

### 2.2 Direction evolution on S²

The massless-particle geodesic equation in orthogonal Type I gives
one ODE per angular coordinate.  Writing the direction unit vector
`ê = (sin θ cos φ, sin θ sin φ, cos θ)` in the Wainwright–Hsu basis,
the per-ray geodesic evolution is

```
dμ/dN = 3 Σ_+ μ (1 - μ²) + 3 Σ_- (μ (1-μ²) + 2 (1-μ²) · [something in φ])
dφ/dN = f(Σ_-, μ, φ)
```

A clean covariant form, derivable from paper eq (46) generalised to
two shear eigenvalues, is:

```
d/dN (p_i/E) = -(H δ_{ij} + σ_{ij}) (p_j/E) + [p_k/E · σ_{km} · p_m/E] (p_i/E)
```

The analytic LRS shortcut `X = μ²/(1-μ²) → X_0 e^{6S}` no longer
applies: the full 2-component angular ODE must be integrated
numerically for non-axisymmetric shear.

### 2.3 Stress extraction

The anisotropic stress tensor projections onto the Σ_+ and Σ_-
directions are (paper eq 14 generalised):

```
π̃_+ = (1/f_ν) · ∫ dΩ f(ê) · (3 cos² θ - 1)/2 · [energy weight]
π̃_- = (1/f_ν) · ∫ dΩ f(ê) · sin² θ cos(2φ) · √3/2 · [energy weight]

Π_+ = 6 f_ν π̃_+,  Π_- = 6 f_ν π̃_-
```

The characteristic identity `Π_+ = f_ν Σ_j w_j J_j P_2(μ_j) e^{-8 I_j}`
(paper eq 57) remains correct with two modifications:

- The quadrature `Σ_j w_j (...)` is now a S² quadrature, not
  `∫_{-1}^{1} dμ`.
- The `e^{-8 I_j}` energy-shift factor still comes from the per-ray
  gravitational redshift accumulation, but `I_j` now solves a 2D-path
  version of `dI_j/dN = Σ_ab ê^a_j(N) ê^b_j(N)` with `Σ_ab` containing
  both shear eigenvalues.

### 2.4 Monopole and higher moments

The weak-rate monopole f̃₀(q) still comes from angle-averaging:

```
f̃_0(q) = (1/4π) ∫_{S²} dΩ f(ê, q)
       = Σ_j w_j J_j f_FD(q e^{2 I_j})     (characteristic form)
```

The quadrupole moments f̃_2 (paper eq 186) now carry two m-components
`f̃_2^{m=0}` and `f̃_2^{|m|=2}`, which source the two stress
components respectively.

---

## 3. State-vector design

Two options, corresponding to how the S² quadrature is discretised.

### Option A — tensor-product ray grid (simpler)

Use `N_θ × N_φ` rays:

```
y = [Σ_+, Σ_-,
     I_{θ_j, φ_k},   for j=1..N_θ, k=1..N_φ,
     J_{θ_j, φ_k},
     S_+, S_-,       # accumulated shear integrals (two components)
     T_γ,
     X_i]
```

Total ray DOF: `2 · N_θ · N_φ`.  With (N_θ, N_φ) = (8, 16) this is
256 ray DOF, up from 24 in LRS.  Plus the same 4 + N_species trailing
scalars.

**Quadrature weight:** `w_jk = w^{(θ)}_j · w^{(φ)}_k · sin(θ_j)` with
trapezoidal (or Gauss) weights in φ and Gauss–Legendre in cos θ.

**Pros:** product-factor simplifies the solid-angle integration and
makes the path reduce to LRS exactly when the φ-sum averages out.

**Cons:** 256 DOF is ~10× larger than LRS; Rodas5P Jacobian is now
~260×260 (dense), ~4× larger memory.

### Option B — Lebedev quadrature (fewer points, same accuracy)

Lebedev–Laikov grids on S² deliver exact integration of spherical
harmonics up to degree ℓ_max at far fewer nodes than the product grid:

- ℓ_max = 5 ⇒ 26 nodes (covers up to Ψ_2 exactly)
- ℓ_max = 11 ⇒ 110 nodes (generous margin)

Use `N_S² = 26–50` rays with Lebedev weights `w_j`.  Memory-optimal.

**Pros:** ~5× fewer DOF than tensor product at the same `ℓ_2` fidelity.
**Cons:** custom Lebedev weight table; less intuitive; no analytic
LRS-limit check via direct axial slicing.

### Recommendation

Start with **Option A** at (N_θ, N_φ) = (12, 16) = 192 rays so the
LRS limit is trivially recovered by setting `N_φ = 1` and all
φ-weights to 1.  Add Option B as a performance knob after parity
lock.

---

## 4. Implementation plan

### Phase 0 — audit and scope (1 day)

1. Grep the SciPy generic Type I geometry sector
   ([`rabbit.geometry.typeI`](../src/rabbit/geometry/typeI.py)) — it
   already tracks Σ_-.  Confirm the stress pipeline
   ([`rabbit.transport.projectors.extract_aniso_stress`](../src/rabbit/transport/projectors.py))
   already returns both Π_+ and Π_- for the PSTF path.
2. Confirm the linearised-PSTF generic path is unit-tested
   (`n_ell=3`).  Use those tests as a weak regression for the new
   characteristic path.

### Phase 1 — S² ray grid (delivered in PR-N1)

Delivered as
[`src/rabbit/jax/characteristic_rays_nonlrs_jax.py`](../src/rabbit/jax/characteristic_rays_nonlrs_jax.py).
The implementation keeps the tensor-product quadrature because the
exact LRS-reduction tests are easiest to state on that grid.

```python
@lru_cache(maxsize=8)
def setup_ray_grid_S2(N_theta: int, N_phi: int):
    mu0, w_mu = leggauss(N_theta)              # cos θ nodes
    phi0 = jnp.linspace(0, 2*jnp.pi, N_phi, endpoint=False)
    w_phi = 2*jnp.pi / N_phi
    # Build full S² mesh (N_theta * N_phi rays)
    mu_grid = jnp.broadcast_to(mu0[:, None],   (N_theta, N_phi)).reshape(-1)
    phi_grid = jnp.broadcast_to(phi0[None, :], (N_theta, N_phi)).reshape(-1)
    w_grid  = jnp.broadcast_to(
        (w_mu[:, None] * w_phi / (4*jnp.pi)),   (N_theta, N_phi)
    ).reshape(-1)
    return mu_grid, phi_grid, w_grid
```

### Phase 2 — direction map on S² (delivered in PR-N1)

PR-N1 chose the accumulated-integral route.  In orthogonal Type I the
generator stays diagonal in the fixed Wainwright-Hsu basis, so the
integrated stretch operators commute and the exact map is an axis-wise
rescaling followed by renormalisation.  This keeps PR-N2 free to use
the same low-DOF `S_+, S_-` strategy instead of inflating the Rodas5P
state with explicit `(μ_j, φ_j)` ODE variables.

### Phase 3 — stress / monopole extraction on S² (delivered in PR-N1)

Delivered in the same module with the LRS-compatible weight convention
`sum(w_s2) = 2`, so the existing `0.5 * (...)` monopole normalisation
continues to hold in the `N_phi = 1` reduction.

General form:

```python
def extract_stress_plus_S2(I, J, mu, w, f_nu):
    P2 = 0.5 * (3 * mu**2 - 1)
    return f_nu * jnp.sum(w * J * P2 * jnp.exp(-8 * I))

def extract_stress_minus_S2(I, J, mu, phi, w, f_nu):
    # Y_2^{±2} combination → sin²θ cos(2φ)
    X = (1 - mu**2) * jnp.cos(2 * phi)
    return f_nu * jnp.sum(w * J * X * jnp.exp(-8 * I)) * (jnp.sqrt(3.0) / 2.0)

def extract_monopole_S2(I, J, w, q_nodes):
    # identical to LRS form but w is S² weight
    alpha = jnp.exp(2 * I)
    qa = q_nodes[:, None] * alpha[None, :]
    f_vals = 1.0 / (jnp.exp(jnp.minimum(qa, 500)) + 1)
    return 0.5 * f_vals @ (w * J)
```

### Phase 4 — driver integration (delivered in PR-N2)

[`rabbit.jax.driver_typeI_char`](../src/rabbit/jax/driver_typeI_char.py)
now accepts `transport_mode="characteristic_nonlrs"`:

- `_char_layout_nonlrs(N_theta, N_phi, n_species)` keeps the state
  compact: `[Σ_+, Σ_-, S_+, S_-, T_γ, X_i]` at tier-1.  PR-N2 did
  **not** add an explicit `(N_theta × N_phi)` ray-state block.
- `_rhs_core_nonlrs` reconstructs `(μ, φ, I, J)` analytically from
  `(S_+, S_-)` on every RHS call and reuses the same thermo / weak /
  network sectors as the LRS characteristic path.
- `JAXTypeICharConfig` and `JAXTypeIConfig` both accept the new
  transport mode, and the public inference surface is
  `canonical_forward_solver(..., backend="jax_characteristic_nonlrs")`.
- Scope guardrails landed with the driver:
  - tier-1 by default, with explicit tier-2 3T opt-in,
  - collisionless public dispatch only,
  - `N_phi=1` reserved for the exact LRS reduction slice,
  - generic `Σ_- ≠ 0` requires `N_phi >= 2`.

### Phase 5 — validation (landed subset in PR-N2)

PR-N2 landed the first integration-grade validation bundle:

1. **Analytic Jacobian audit.**  `jacobian_nonlrs_jax(...)` matches
   `jax.jacfwd` of the exact forward map at `< 1e-11`.
2. **LRS reduction.**  With `N_phi = 1` and `Σ_- = 0`, the generic path
   reduces to the LRS characteristic driver to `|ΔY_p| ~ 1e-8`,
   `|ΔD/H| ~ 1e-10` on the locked regression cell.
3. **Generic sign audit.**  At small shear (`Σ_+ = Σ_- = 0.05`) the
   exact non-LRS path agrees in **sign** with the generic linearized
   PSTF reference, but its magnitude is about `4x` larger on the
   current lock cell.  This is recorded as candidate-scope behavior,
   not canonical parity.
4. **Large-shear smoke.**  A representative `(Σ_+, Σ_-) = (0.3, 0.1)`
   run completes with `Ω_final > 0`.

### Phase 6 — dispatch + metadata (landed)

- `canonical_forward_solver(backend="jax_characteristic_nonlrs", ...)`
- New capability entry
  `JAX_TYPEI_CHARACTERISTIC_NONLRS_TIER1` mirroring the LRS version
  but with `transport_scope_contract="characteristic_nonlrs_exact_v1"`.
- `Sigma_H_minus` accepted in the signature of `canonical_forward_solver`
  for the new backend.
- `N_theta` and `N_phi` are public forward-solver controls for the
  tensor-product S² angular grid.
- Tier-2 3T thermodynamics is an explicit candidate opt-in via
  `jax_thermo_tier=2`; collision-coupled non-LRS transport remains out
  of scope.

### Phase 7 — explicit residual-state staging (landed private surface)

The compact public non-LRS driver cannot honestly apply the SciPy
anisotropic residual closure, because `apply_species_residual_relaxation`
generates raywise `delta_I` and `delta_J` corrections that are not
representable by only `(S_+, S_-)`.  The private
`run_nonlrs_tier2_residual_state_jax(...)` helper therefore adds the missing
state variables without changing public dispatch:

```
y = [Σ_+, Σ_-,
     S_+, S_-,
     R_I[ν_e,ν̄_e,ν_x; θ,φ],
     R_J[ν_e,ν̄_e,ν_x; θ,φ],
     T_γ, T_νe, T_νx,
     X_i]
```

The total ray variables are reconstructed as
`I_species = I_geo(S_+,S_-) + R_I_species` and
`J_species = J_geo(S_+,S_-) + R_J_species`.  The residual RHS then applies the
same mean-preserving relaxation contract as the SciPy residual closure on the
S² bundle:

```
dR_I/dN = -κ0 (Γ/H) [I - <I>]
dK/dN   = -κ2 (Γ/H) [K - 1],   K = J exp[-8(I-<I>)]
```

The smoke surface records `transport_species_mode="per_species"`,
`collision_closure_mode="nonlrs_s2_residual_relaxation_v1"`,
`public_dispatch_ready=False`, and `qke_scope="out_of_scope"`.  It does not
make `canonical_forward_solver(..., backend="jax_characteristic_nonlrs",
enable_collisions=True)` valid; that public path still fails closed.

---

## 5. Test matrix (must all pass before promotion)

| Σ_+ | Σ_- | Target |
|---|---|---|
| 0.0 | 0.0 | FLRW limit, \|ΔY_p\| < 1e-10 vs SciPy FLRW |
| 0.1 | 0.0 | LRS limit, \|ΔY_p\| ~ 1e-8 vs current LRS driver |
| 0.0 | 0.1 | Σ_- sector alone, \|ΔY_p\| by 90°-rotation symmetry |
| 0.1 | 0.1 | generic, sign-agreement with linearised PSTF; magnitude remains candidate-only |
| 0.3 | 0.1 | large Σ_+, moderate Σ_- |
| 0.1 | 0.3 | swap test — observables invariant under coordinate relabelling |
| 0.04 | 0.015 | private residual-state Tier-2 smoke: finite Rodas5P solve, per-species residual metadata, mean-preserving `delta_I` |

---

## 6. Risk register

- **S² quadrature convergence at moderate shear.**  When Σ ~ 0.3 the
  angular distribution develops `Ψ_4`-level structure; the product
  grid (N_θ, N_φ) = (12, 16) may leak ~10⁻³ error into Π_±.  Bump to
  Lebedev N=110 if convergence tests fail at Σ = 0.5.
- **Rodas5P Jacobian growth.**  The landed compact driver does not
  carry explicit ray-state ODE blocks; it evolves the shear integrals
  and reconstructs ray observables analytically.  The remaining risk is
  angular-grid cost inside RHS evaluations, especially for tier-2 3T
  runs.
- **Residual collision state is not compact.**  Non-LRS anisotropic
  residual relaxation cannot be represented by the compact public
  `(S_+, S_-)` state alone.  The private residual-state helper now lands
  the required explicit S² per-species `R_I/R_J` variables as a staged
  physical surface while public collision dispatch remains closed.
- **Near-Milne limit.**  Ω → 0 when Σ → 1.  The existing
  `enforce_positive_typeI_omega` guard in the SciPy geometry layer
  must be replicated in JAX with a JAX-safe clip (no Python raise
  inside jit).

---

## 7. LOC budget and timeline

| Phase | Description | LOC | Days |
|---|---|---|---|
| 0 | Scope audit | — | 1 |
| 1 | S² ray grid | 200 | 1 |
| 2 | Direction ODE | 300 | 2 |
| 3 | Stress / monopole extraction | 150 | 1 |
| 4 | Driver integration | 200 | 1 |
| 5 | Validation tests | 300 | 2 |
| 6 | Dispatch + metadata | 80 | 0.5 |
| **Total** | | **≈ 1230** | **≈ 8.5** |

---

## 8. What this does NOT deliver

- Bianchi Types II, VII₀, VIII, IX — those require curved-kernel
  transport (paper §D.1, Class A).  Non-LRS Type I is still the
  simplest Bianchi family; the S²-quadrature machinery introduced here
  is reused by Class A but the direction ODE is different.
- Tilted Type I (non-zero tilt velocity).  Requires the `ȧ_a e^a/H`
  source term in the energy-shift integral (paper §6 generalisation).
- Collision coupling on non-LRS rays.  The supported candidate surface
  is collisionless transport with tier-1 or tier-2 thermodynamics.  See
  the incomplete-decoupling guide for the collision layer.
