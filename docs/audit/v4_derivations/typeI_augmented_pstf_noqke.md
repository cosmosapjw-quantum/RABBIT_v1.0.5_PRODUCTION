# Derivation Note — Type I Augmented PSTF Full Boltzmann, No QKE

**Purpose.**  This note fixes the equation provenance for the
nonperturbative augmented-PSTF Type I programme.  It is not a runtime
implementation and it does not promote any backend capability.

**Scope.**  Orthogonal Bianchi Type I, including LRS and diagonal
non-LRS shear, classical scalar neutrino Boltzmann transport, no QKE.

---

## 1. References And Local Sources

Local report anchors:

- `docs/RABBIT_report/sections/03_neutrino_transport_linearised_pstf_hierarchy.tex`
  for the existing linearised PSTF baseline.
- `docs/RABBIT_report/sections/06_exact_characteristic_ray_transport.tex`
  for exact Type I collisionless characteristic transport.
- `docs/RABBIT_report/sections/07_the_collision_integral_general_structure.tex`
  for the classical 2-to-2 collision integral and statistical factor.
- `docs/RABBIT_report/sections/08_the_hannestad_madsen_collision_kernel.tex`
  for the reduced `nu-e` collision kernel.
- `docs/RABBIT_report/sections/13_weak_n_p_interconversion_rates.tex`
  for live weak-rate functionals of the neutrino monopole.
- `docs/RABBIT_report/sections/A05_full_phase_space_ray_boltzmann_equation.tex`
  for the full per-ray, per-momentum Boltzmann equation.

External formalism anchors:

- Thorne, *Relativistic radiative transfer: moment formalisms*,
  MNRAS 194, 439 (1981), DOI `10.1093/mnras/194.2.439`.
- Ellis, Treciokas, and Matravers, *Anisotropic solutions of the
  Einstein-Boltzmann equations*, Annals of Physics 150, 455 (1983),
  DOI `10.1016/0003-4916(83)90023-4`.
- Challinor and Lasenby, covariant PSTF CMB hierarchy work,
  Annals of Physics 280, 301 (2000), DOI `10.1006/aphy.2000.6033`.

---

## 2. Covariant Distribution Decomposition

Use the fundamental 4-velocity `u^a` of the homogeneous cosmological
frame and the spatial projector

```text
h_ab = g_ab + u_a u_b .
```

For a massless neutrino momentum,

```text
p^a = E (u^a + e^a),     e^a u_a = 0,     e^a e_a = 1 .
```

The scalar distribution is `f_s(q, e^a, N)`, with species label `s`.
The angular dependence is expanded in real PSTF moments:

```text
A_s(q, e, N) = sum_{ell=0}^{ell_max}
               A^{(s)}_{<a1...aell>}(q,N)
               e^{<a1} ... e^{aell>}.
```

The nonperturbative augmented distribution is

```text
f_s(q, e, N) = 1 / (exp(q + A_s(q,e,N)) + 1).
```

This replaces the legacy linearised form

```text
f = f0(q) * (1 + Psi)
```

and therefore does not assume `|Psi| << 1`.  The finite `ell_max`
remains a truncation of the angular representation, so convergence in
`ell_max` is mandatory.

### LRS Reduction

For LRS Type I, the frame has one symmetry axis and only `m=0` real
modes survive:

```text
A_s(q, mu, N) = sum_{ell even <= ell_max} A^{(s)}_ell(q,N) P_ell(mu).
```

### Diagonal non-LRS Type I

For diagonal non-LRS Type I, use the principal-axis frame of the shear.
Reflection-symmetric no-tilt initial data uses even `ell` and even
cosine real modes by default.  Sine partners are diagnostic opt-ins,
not part of the default diagonal model.

---

## 3. Stress-Energy And Einstein Feedback

The collision-coupled Einstein RHS must use the stress tensor of the
actual reconstructed distribution, not a linearised moment surrogate.
For each species,

```text
T_ab[s] = integral p_a p_b f_s dP .
```

For massless particles in the `u^a` frame,

```text
rho_s  proportional to integral dq q^3 integral dOmega f_s(q,e),
pi_ab[s] proportional to integral dq q^3 integral dOmega
                      f_s(q,e) e_<a e_b>.
```

The Type I geometry uses the two diagonal shear projections:

```text
Pi_+ = projection of total pi_ab onto the Sigma_+ basis tensor,
Pi_- = projection of total pi_ab onto the Sigma_- basis tensor.
```

In an `(mu,phi)` angular coordinate convention aligned with the local
principal axes, the corresponding angular weights are the real
quadrupole functions:

```text
W_+(mu,phi) = (3 mu^2 - 1) / 2,
W_-(mu,phi) = sqrt(3) * (1 - mu^2) * cos(2 phi) / 2 .
```

The exact normalization remains the responsibility of the existing
RABBIT weight convention for `Pi_+` and `Pi_-`; this note fixes only
the required dependence on the live distribution and on the two real
quadrupole components.

---

## 4. Boltzmann Equation In Augmented Variables

The classical no-QKE Boltzmann equation is

```text
df_s/dN + streaming[f_s] = C_s[f] / H .
```

For full phase-space Type I transport the local form is the same as in
the local report appendix:

```text
partial_N f_j(q)
  = - (d ln q / dN)_j q partial_q f_j(q) + C[f](q, n_j) / H .
```

In the augmented variable `G_s = q + A_s`,

```text
f_s = sigmoid(-G_s),
partial f_s / partial A_s = - f_s (1 - f_s).
```

Therefore a nodal RHS for the distribution maps back to augmented
coefficients through

```text
dA_s/dN = - (df_s/dN) / max(f_s (1 - f_s), eps_f).
```

The denominator floor `eps_f` is a numerical regularization and must
be metadata-visible once this path enters a solver.  After computing
nodal `dA_s/dN`, the solver projects it back to the active PSTF basis
with the same S_N quadrature used for stress and weak-rate moments.

---

## 5. Weak-Rate Contract

At Born/CL0-CL2 level, the homogeneous electron-positron and baryon
backgrounds are angularly isotropic.  The neutrino contribution to
the weak rates therefore enters through the angular monopoles

```text
f0_nue(q)     = (1 / 4pi) integral dOmega f_nue(q,e),
f0_nuebar(q) = (1 / 4pi) integral dOmega f_nuebar(q,e).
```

This matches the live-rate contract in the local weak-rate report
section.  Pure anisotropic modes that leave the monopole unchanged
must not change Born weak rates.

Finite-mass, recoil, and weak-magnetism CL3 angular corrections are a
separate explicit layer.  If they consume quadrupole or higher angular
moments, that dependency must be exposed in metadata and tested
separately from the Born monopole contract.

---

## 6. Collision Contract

The classical 2-to-2 collision term has the local-report form

```text
C[f_1] = integral dPi_2 dPi_3 dPi_4
         delta^4(p1+p2-p3-p4) |M|^2 S,

S = f_3 f_4 (1-f_1)(1-f_2)
    - f_1 f_2 (1-f_3)(1-f_4).
```

The required no-QKE processes are:

```text
nu_alpha + e^\pm       -> nu_alpha + e^\pm,
nu_alpha + anti_nu_alpha <-> e^+ + e^-,
nu_alpha + nu_beta     -> nu_alpha + nu_beta    (diagonal no-QKE).
```

The deterministic reference implementation evaluates collision
moments from the current `f_s(q,e)` using fixed quadrature and S_N
angular weights.  Detailed balance at common-temperature FD must give
zero collision source within tolerance.  Number and energy residuals
are mandatory diagnostics.

Sampling accelerators are allowed only after this deterministic
reference exists.  They must use fixed samples per solve and converge
to the deterministic reference under sample-count refinement.

---

## 7. Transport-Method Decision

The reference angular method is S_N plus PSTF projection:

```text
f(q,e) -> evaluate on angular nodes -> integrate/project moments.
```

Reasons:

- it directly represents the anisotropic distribution used by weak and
  collision integrals;
- it gives a clean angular-grid convergence knob independent of
  `ell_max`;
- it naturally supports both LRS and non-LRS Type I;
- it avoids committing the production path to a low-moment closure.

M1 is reserved for diagnostic comparisons because it closes at two
moments and cannot replace an `ell_max` convergence ladder.  DSMC is
not used as a live stiff-RHS method because random RHS noise would
break reproducibility and convergence diagnostics.

---

## 8. Promotion Requirements

No implementation under this programme may be labelled production or
canonical until the following are recorded:

- LRS and non-LRS mode reconstruction tests.
- `Sigma_- = 0` non-LRS reduction to LRS.
- `ell_max = 2,4,6,8` convergence, extending to `10` if needed.
- Angular-grid and q-grid convergence separated from `ell_max`.
- FD detailed-balance collision test.
- Weak-rate Born monopole invariance test.
- SciPy stability gate before JAX parity.
- JAX CPU parity before XLA/GPU runtime gates.
