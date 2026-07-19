# Tier-3 Full-Collision Preflight (2026-04-21)

## Scope

This note is a bounded preflight for the deferred no-QKE tier-3 path:

- full per-ray, per-momentum transport state
- q-advection inside the existing JAX Rodas5P stack
- analytic Jacobian strategy that remains viable near the planned
  `~975`-DOF surface

Supporting experiment:

- [`tmp/tier3_full_collision_preflight.py`](../../tmp/tier3_full_collision_preflight.py)

Key local references:

- [`docs/IMPLEMENTATION_GUIDE_FULL_COLLISION_NOQKE.md`](../IMPLEMENTATION_GUIDE_FULL_COLLISION_NOQKE.md)
- [`docs/phase_prompts/PR-T3A_q_advection.md`](../phase_prompts/PR-T3A_q_advection.md)
- [`src/rabbit/jax/solver_jax_rodas5p.py`](../../src/rabbit/jax/solver_jax_rodas5p.py)

---

## 1. Divergence

The open design forks are not "which code to write first", but which
numerical contract is compatible with the current solver stack.

Candidate A: operator-split semi-Lagrangian advection

- Pros: best positivity/shape preservation; naturally matches the exact
  characteristic map `q -> q exp(Delta)`.
- Cons: current Rodas5P expects a continuous RHS. A split remap stage
  sits outside the Rosenbrock error controller and event logic.

Candidate B: inline semidiscrete q-advection in the RHS

- Pros: compatible with the current adaptive/evented Rodas5P surface.
- Cons: advection discretization becomes diffusive or oscillatory unless
  the stencil is chosen carefully.

Candidate C: full dense 975x975 analytic Jacobian

- Pros: minimal solver redesign.
- Cons: wastes the structure induced by the gather-collide-scatter (GCS)
  formulation.

Candidate D: factorized GCS Jacobian

- Gather ray data to moment space.
- Differentiate the collision core in moment space.
- Lift the collision Jacobian back to rays.

This is the only candidate that exploits both the paper-level GCS
physics and the planned tier-3 state layout.

---

## 2. Metacognition

The local docs were internally inconsistent. The implementation guide
leaned toward semi-Lagrangian/PCHIP, while the PR-T3A prompt correctly
flagged that embedding a semi-Lagrangian remap *inside* a Rodas5P RHS is
not straightforward.

The current solver matters here. The existing Schur/block-sparse path in
[`solver_jax_rodas5p.py`](../../src/rabbit/jax/solver_jax_rodas5p.py)
assumes `J[passive, passive] = 0`. That was valid for the old
collisionless passive transport block, but it is false for tier-3
full-collision transport. Reusing that assumption would be a category
error.

So the real question is not "semi-Lagrangian vs finite differences" in
the abstract. It is:

- what advection form preserves the existing Rodas5P contract today?
- what Jacobian representation makes solver-side work scale acceptably
  tomorrow?

---

## 3. Verification

### 3.1 Local code experiment

Command:

```bash
venv/bin/python tmp/tier3_full_collision_preflight.py
```

The script compares three bounded q-advection proxies against the exact
collisionless characteristic shift on the Laguerre q-grid.

Representative results at `N_q=20`:

- shift `0.2`
  - centered semidiscrete: `rel_l2 = 1.04e-2`, `min = -3.8e-05`
  - upwind semidiscrete: `rel_l2 = 3.06e-2`, near-positive but more diffusive
  - PCHIP exact remap: `rel_l2 = 2.38e-3`, `min = 0.0`
- shift `0.5`
  - centered semidiscrete: `rel_l2 = 3.16e-2`, `min = -3.05e-3`
  - upwind semidiscrete: `rel_l2 = 7.99e-2`
  - PCHIP exact remap: `rel_l2 = 1.80e-3`, `min = 0.0`

Interpretation:

- PCHIP remap is the best *transport oracle*.
- Centered semidiscretization is sharper than upwind but breaks
  positivity.
- Upwind is the safest continuous-RHS baseline, but it is visibly
  diffusive.

The same script probes the planned GCS Jacobian structure at
`4 species x 12 rays x 20 q = 960` transport DOF:

- transport dimension: `960`
- total state dimension with scalars: `975`
- moment-space dimension for `(f_0, f_2)`: `160`
- materialized transport Jacobian entries: `921600`
- dense moment-core entries: `25600` (`2.78%` of the materialized
  transport matrix)
- factorized storage for `A @ M @ G`: `29440` nonzeros (`3.19%` of the
  materialized transport matrix)
- materialized lifted collision block rank: `160`

This is the key result. The tier-3 collision Jacobian is dense *if
materialized on rays*, but it is naturally low-rank/factorized through a
160-dimensional moment core.

### 3.2 Local architectural readout

The implementation guide already encodes the GCS cycle:

- gather `f_0(q), f_2(q)` from the rays
- collide in moment space
- apply the result back to each ray

That implies

```text
J_collision = A_apply * (dC/dm) * G_gather
```

with `m = (f_0, f_2)`.

So the best Jacobian target is *not* a hand-written 975x975 dense
matrix. It is a moment-core Jacobian `dC/dm` plus fixed gather/apply
maps.

### 3.3 Web CRAG

External sources support the same convergence:

- Staniforth & Cote's review says semi-Lagrangian methods are attractive
  because of their accuracy, stability, and efficiency properties.
  Source: https://www.osti.gov/biblio/6059770
- Fritsch-Carlson gives a monotone piecewise cubic interpolant for
  monotone data, which is exactly why PCHIP is the right oracle for the
  collisionless q-shift.
  Source: https://www.osti.gov/biblio/6893418
- Boscheri et al. show why semi-Lagrangian methods are awkward to mix
  with Runge-Kutta time stepping: SL couples space and time directly,
  so standard RK integration is not a natural fit without a dedicated
  IMEX formulation.
  Source: https://link.springer.com/article/10.1007/s10915-022-01768-0
- Lentine et al. make the conservation caveat explicit: standard
  semi-Lagrangian interpolation is not automatically conservative.
  Source: https://www.sciencedirect.com/science/article/abs/pii/S0021999110007102
- The SciML Rosenbrock docs say Rosenbrock methods are best for small to
  medium stiff systems (`< 1000` ODEs) when Jacobians are available or
  can be computed efficiently.
  Source: https://docs.sciml.ai/OrdinaryDiffEq/v6.102/semiimplicit/Rosenbrock/
- Froustey, Pitrou, and Volpe show the decisive Jacobian lesson for
  neutrino decoupling: finite-difference Jacobians push cost from
  `O(N^3)` collision evaluation to `O(N^4)`, while a direct Jacobian
  construction restores `O(N^3)` and gives an `N/5` speed-up.
  Source: https://www2.iap.fr/users/pitrou/publi/2008.01074.pdf

---

## 4. Convergence

### 4.1 Best q-advection path

For the first landed tier-3 path, use a **continuous semidiscrete RHS**
inside Rodas5P, not an operator-split semi-Lagrangian remap.

Recommended order:

1. Implement a monotone one-sided q-advection stencil in the continuous
   RHS.
2. Use the exact PCHIP remap only as a collisionless regression oracle.
3. Revisit SL/IMEX only after a dedicated split-integrator surface
   exists.

Reason:

- PCHIP remap is numerically best.
- But with the current solver contract it belongs in verification, not
  in the first production tier-3 driver.

### 4.2 Best Jacobian path

Do **not** target a fully materialized dense tier-3 Jacobian as the
primary design.

Instead target:

```text
J_total = J_q-advection(banded)
        + A_apply * J_collision_core(moment-space) * G_gather
        + small scalar couplings
```

where:

- `J_q-advection` is block-banded over `(species, ray, q)`
- `J_collision_core` lives on the `160`-DOF `(f_0, f_2)` moment space
- `A_apply` and `G_gather` are fixed linear maps

This is the numerically honest way to exploit the GCS formulation.

### 4.3 Consequence for the roadmap

The next real dependency is solver-side, not transport-side.

Before a serious tier-3 driver attempt, add one of:

- a low-rank / matrix-free `W` operator path
- a custom linear-solve hook that can consume
  `J_q + A M G` without materializing the full dense transport matrix

Without that, the old passive-zero Schur shortcut cannot be reused, and
the tier-3 driver will land on the wrong Jacobian surface.

---

## 5. Decision

**Decision:** tier-3 preflight converges on

- continuous semidiscrete q-advection for the first production attempt
- PCHIP exact remap as the collisionless oracle, not the in-solver step
- factorized moment-core Jacobian as the primary analytic Jacobian target

In short:

- `PR-T3A` should become "continuous q-advection + collisionless
  reduction + oracle checks"
- the old `PR-J` idea should evolve into a tier-3-specific
  "moment-core Jacobian + solver hook" line, not a dense full-state
  Jacobian line
