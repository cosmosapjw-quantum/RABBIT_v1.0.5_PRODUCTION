# BD622 W6 — B3-v2 independent comparator and W7 oracle contract

Date: 2026-07-23  
Status: **PROPOSED / DESIGN ONLY / NOT IMPLEMENTATION AUTHORITY**  
D-035 remediation revision: **exact-byte candidate for fresh blind review; not yet reviewed**

This file replaces the rejected species-blind B3 proof pack. It authorizes no W7 numerical
output, B3 collision output, T01--T12 execution, Radau construction, trajectory, endpoint,
RABBIT unblinding, QKE, Type-I, or public runtime. D-027, D-028, and D-029 remain immutable
failure/rejection evidence. A reviewer must be able to reconstruct one design from the bytes
below without reading a RABBIT collision implementation or choosing an omitted convention.
D-034 adjudication `95870216` rejected the prior bytes. This revision changes only the named
normalization, constant, finite-mass, basis/ladder, arithmetic-graph, and tool-containment
defects; its retained algebraic subclaims are not promoted.

## 1. Scope, sources, and non-goals

The target is the smallest structurally independent classical, flavour-diagonal,
zero-lepton-asymmetry, flat-FLRW, no-QKE full-spectral neutrino comparator needed by
`G-F10-INDEPENDENT-FLRW`.

The external microphysics anchors are Hannestad--Madsen 1995, Tables I--II and Eq. (3),
<https://arxiv.org/abs/astro-ph/9506015>, and Dolgov--Hansen--Semikoz 1997,
<https://arxiv.org/abs/hep-ph/9703315>. The former explicitly supplies the finite-electron-mass
matrix elements, phase-space measure, spin convention, and one `1/2!` for every identical
integrated initial/final pair. W3 remains only a derived coefficient-ratio/MB-reduction check.

The comparator must not:

- import, call, or line-translate the Rust, JAX, Fort, or frozen D-028 collision evaluator;
- evolve folded electron/heavy pair shapes internally;
- use a target-dependent support selector, exact-node one-hot exception, or conservation repair;
- reuse D-028's `1/y^2` pointwise inversion or method-specific native cap;
- change a graph, grid, state, norm, cap, event order, or tolerance after candidate output;
- add a public API, capability entry, backend selector, policy knob, or runtime dispatch.

## 2. Frozen conventions and dimensional map

- Natural units `c = hbar = k_B = 1`; energy, momentum, mass, and temperature are MeV.
- `g_phys = diag(-1,+1,+1,+1)`, `p.p = g_phys(p,p)`, and `chi_ij = -p_i.p_j`.
- Future-directed on-shell legs have `p^0 > 0`, `p.p = -m^2`, and `chi_ij >= 0`.
- `K_s = chi_12 chi_34`, `K_t = chi_14 chi_23`, `K_u = chi_13 chi_24`.
- Species order is exactly `[nu_e, antinu_e, nu_mu, antinu_mu, nu_tau, antinu_tau]`.
- Each neutrino/antineutrino species has one occupied helicity state. No neutrino spin average is
  taken. Electron/positron distributions are per spin state; the matrix element sums both initial
  and final electron/positron spins, so no separate bath factor two and no initial spin average is
  allowed.
- `G_F = 1.1663788e-11 MeV^-2`, `m_e = 0.5109989500 MeV`, and
  `sin2_theta_W = 0.23122` are exact decimal input strings for this design.
- `y = p_nu/T_cm`; `x = |p_e|/T_gamma`; `E_e = sqrt((T_gamma x)^2+m_e^2)`.
- An event is oriented `1+2 <-> 3+4`. Define
  `F=f1 f2 (1-f3)(1-f4)`, `R=f3 f4 (1-f1)(1-f2)`, `J=F-R`, and
  `nu_leg=(-1,-1,+1,+1)`.
- `Q_nu > 0` is energy gained by neutrinos; `Q_EM=-Q_nu`; the equilibrium bath contribution to
  total entropy is `Q_EM/T_gamma`.

The invariant global event measure is, with no hidden degeneracy,

```text
dGamma_r = O_r S_r (2*pi)^4 delta^4(p1+p2-p3-p4)
           * product[l=1..4] {d^3 p_l / [(2*pi)^3 2 E_l]}
           * sum_spins |M_r|^2,
S_r = 1 / [product_a n_in(a)! product_b n_out(b)!].
O_r = 1/2 if the unordered initial and final species multisets are equal,
      1 otherwise.
```

`O_r` quotients the initial/final orientation of a physical elastic event. On the full labeled
four-leg domain, initial/final reversal changes both `J` and the signed test-leg action, so their
product is invariant and would otherwise be integrated twice. Conversion and annihilation rows
have different initial/final species multisets, their orientation is fixed by the canonical row,
and `R` alone represents the reverse. `O_r` is independent of `S_r`; neither may be absorbed into
an amplitude coefficient.

For a neutrino test mode `psi_i`, the dimensionless weak moment is

```text
q[s,i] = (2*pi^2/T_cm^3) sum_r integral dGamma_r J_r
         * sum[l=1..4] nu_leg[l] 1(species_l=s) psi_i(y_l).
```

The prefactor follows from
`n_s=T_cm^3/(2*pi^2) integral_0^infinity y^2 f_s(y) dy`. It is not folded into a tagged
matrix-element coefficient. The physical neutrino energy transfer uses the same event once:

```text
Q_nu = sum_r integral dGamma_r J_r
       * sum_neutrino_legs nu_leg[l] E_l;  Q_EM=-Q_nu.
```

Thus `sum|M|^2` is dimensionless (`G_F^2` times a degree-four momentum invariant), `q` has
dimension MeV, and a massless thermal rate scales as `G_F^2 T_cm^5`.

## 3. Exact six-species reaction generator

Let flavours be ordered `e < mu < tau`; let `n_f` and `a_f` denote neutrino and antineutrino.
Every generated record has `nu_leg=(-1,-1,+1,+1)`, rational `orbit_multiplicity=1`, and the
listed orientation, symmetry, and amplitude IDs. Reverse processes are represented by `R`, never
by a second record. The generator loops in the exact order printed below.

| orbit IDs / loop | oriented legs `(1,2;3,4)` | count | `O_r` | `S_r` | amplitude |
|---|---|---:|---:|---:|---|
| `S00..S05`: species `a` in species order | `(a,a;a,a)` | 6 | `1/2` | `1/4` | `N128_KS_IDENTICAL` |
| `S06..S08`: flavour `f` | `(n_f,a_f;n_f,a_f)` | 3 | `1/2` | `1` | `N128_KT` |
| `S09..S14`: sign `nu,anti`, then `f<g` | `(s_f,s_g;s_f,s_g)` | 6 | `1/2` | `1` | `N32_KS` |
| `S15..S20`: ordered `f`, then ordered `g!=f` | `(n_f,a_g;n_f,a_g)` | 6 | `1/2` | `1` | `N32_KT` |
| `S21..S23`: `f<g` | `(n_f,a_f;n_g,a_g)` | 3 | `1` | `1` | `N32_KT` |
| `E00..E11`: species `a`, then charge `e-,e+` | `(a,e^q;a,e^q)` | 12 | `1/2` | `1` | charge/CP rule in §4 |
| `E12..E14`: flavour `f` | `(n_f,a_f;e-,e+)` | 3 | `1` | `1` | `PAIR_f` |

There are exactly 24 self and 15 electron records. For distinct elastic rows the two
reversal-equivalent global leg depositions times `O_r=1/2` give tagged factor one. For
`(a,a;a,a)`, four equivalent global depositions times `O_r S_r=(1/2)(1/4)` give tagged factor
`1/2`; its global coherent coefficient 128 therefore gives the W3 tagged coefficient 64. The
distinct coherent coefficient 32 gives tagged 32, preserving the derived identical/distinct 2:1
ratio. Conversion/annihilation rows have `O_r=1`. This ledger is the only global-to-tagged
mapping; no additional target, flavour-orbit, bath, or CP multiplicity is permitted.

The massless self amplitudes, before `S_r`, are

```text
N128_KS_IDENTICAL = 128 G_F^2 K_s
N128_KT = 128 G_F^2 K_t
N32_KS  =  32 G_F^2 K_s
N32_KT  =  32 G_F^2 K_t.
```

The exact rational species-count stoichiometric matrix has one column per record, with electron
legs omitted from its six rows. Its full-graph left nullspace must equal
`span_Q{(1,-1,0,0,0,0),(0,0,1,-1,0,0),(0,0,0,0,1,-1)}`. Energy conservation is checked from
four-momenta, not inserted as a species-count null vector. A per-record invariant failure cannot
be rescued by an aggregate zero.

## 4. Finite-electron-mass weak-current ledger

For flavour `f`, define exact-decimal couplings

```text
(C_V^e,C_A^e)       = ( 1/2 + 2 sin2_theta_W,  1/2)
(C_V^mu,C_A^mu)     = (-1/2 + 2 sin2_theta_W, -1/2)
(C_V^tau,C_A^tau)   = (C_V^mu,C_A^mu)
g_L^f = C_V^f + C_A^f;  g_R^f = C_V^f - C_A^f.
```

With electron-event ordering fixed as `1=nu/antinu`, `2=e-/e+`, `3=nu/antinu`,
`4=e-/e+`, the spin-summed, unaveraged amplitudes are

```text
M2(nu_f e- -> nu_f e-) = M2(antinu_f e+ -> antinu_f e+)
  = 32 G_F^2 [g_L^2 K_s + g_R^2 K_t - g_L g_R m_e^2 chi_13],

M2(nu_f e+ -> nu_f e+) = M2(antinu_f e- -> antinu_f e-)
  = 32 G_F^2 [g_L^2 K_t + g_R^2 K_s - g_L g_R m_e^2 chi_13],

M2(nu_f antinu_f -> e- e+)
  = 32 G_F^2 [g_L^2 K_t + g_R^2 K_u + g_L g_R m_e^2 chi_12].
```

These formulas include the charged-current shift for electron flavour through its stated
couplings. Electron/positron occupations are exactly
`f_e(E,T_gamma)=1/(exp(E/T_gamma)+1)` with zero chemical potential and no extra spin factor.
The reaction record chooses the first or second scattering line only from `(neutrino CP,
electron charge)`; a consumer may not infer it by swapping invariant labels later.

## 5. Unique finite-`m_e` coordinate map, shell, and Jacobian

No cleared polynomial root is used. The binding source entry is
`two_body_outgoing(p1,p2,process,m_e,z_star,phi_star)`, where `process` is exactly
`elastic`, `pair`, or `massless`; a generic outgoing-mass Källén branch is forbidden. For each
incoming on-shell pair form `P=p1+p2`, `s=-P.P`, `beta_vec=P_vec/P0`, `beta2=beta_vec.beta_vec`,
and `gamma=P0/sqrt(s)` exactly once. Every boost consumes that same stored `(beta,beta2,gamma)`.
Reject unless `P0>0`, `s>0`, and `0<=beta2<1`. If `beta2=0`, both boosts are the identity.
Otherwise the lab-to-COM boost is

```text
E* = gamma (E - beta_vec.p),
p* = p + [((gamma-1)(beta_vec.p)/|beta|^2) - gamma E] beta_vec,
```

and the inverse changes both minus signs in the bracket to plus. The outgoing COM magnitude is

```text
elastic (0,m_e -> 0,m_e): k*=(s-m_e^2)/(2 sqrt(s)),
  E3*=k*, E4*=(s+m_e^2)/(2 sqrt(s)), require s>m_e^2;
pair (0,0 -> m_e,m_e): k*=sqrt(s-4m_e^2)/2,
  E3*=E4*=sqrt(s)/2, require s>4m_e^2.
```

For `elastic`, reject `s<m_e^2`; for `pair`, reject `s<4m_e^2`. Equality returns a typed
support-only record containing the process, exact `s`, threshold, `k*=0`, `dPhi_2=0`, and
`derivative_side=physical-right`; it contains no frame or outgoing momenta and never enters a
collision quadrature. Interior construction therefore uses strict `>` and never divides by zero.
Set `e_z=p1*/|p1*|`. Choose from
Cartesian `x,y,z` the axis minimizing `|axis.e_z|`, ties in `x<y<z` order; Gram-normalize it as
`e_x`, set `e_y=e_z cross e_x`, and for `z*=cos(theta*)` and `phi*` use

```text
p3*_vec = k*[z* e_z + sqrt(1-z*^2)(cos(phi*) e_x + sin(phi*) e_y)],
p4*_vec = -p3*_vec.
```

Require `-1<=z*<=1` before forming the square root; any outside value is a raw failure and
`max(0,1-z*^2)` is forbidden. Boost back and require positive interior energies. With
`epsilon64=2^-52` and
`P_scale=max(1,abs(all input/output components))`, the componentwise conservation residual must
be `<=64 epsilon64 P_scale`. Every shell residual `|p.p+m^2|` must be
`<=128 epsilon64 max(1,E^2,|p_vec|^2,m^2)`, and the Mandelstam residual
`|s+t+u-sum_l m_l^2|` must be `<=256 epsilon64 max(1,|s|,|t|,|u|,sum_l m_l^2)`.
No aggregate pass rescues an individual failure. The two-body phase space after the outgoing
delta functions is exactly

```text
dPhi_2 = [k*/(16*pi^2*sqrt(s))] dz* dphi*.
```

The only algebraic denominators are `P0`, `sqrt(s)`, `beta2` in the nonzero-beta branch, and
the Gram norm. The boost identity branch removes the apparent `|beta|^2` singularity. Threshold
square roots use the principal nonnegative branch. Derivatives at `s=m_e^2` or `4m_e^2` are
one-sided from the physical side; no two-sided derivative is claimed. This coordinate map is the
binding B3/W7 map, so a different invariant-root elimination is not conforming.

An evolved neutrino leg outside `[0,Y_max]` rejects the whole base event; no clipping occurs. Its
signed reference contribution is accumulated separately in the Stage-B lost/tail enclosure.
The same tuple, amplitude, and energy debit is used for `F` and `R`.

## 6. Deterministic entropy-variable basis and solve

For every `(N_q,Y_max)`, obtain Gauss--Legendre roots by the 256-bit Newton recurrence specified
in the unexecuted W7 source: positive roots are solved in descending-root order, mirrored exactly,
and rounded once to binary64; weights are rounded once after `2/[(1-x^2)P'_N(x)^2]`. The binding
binary64 affine map is exactly `mid=0.5*(lower+upper)`, `half=0.5*(upper-lower)`,
`node=mid+half*root`, `weight=half*weight`, in that order, with `lower=0` and
`upper=Y_max`. The A2 directed enclosure brackets each root by opposite directed signs of `P_N`,
requires the directed derivative interval to exclude zero, bisects left-first to width `<=2^-240`,
and applies the same affine operation graph outward. Let

```text
f0_j=1/(exp(y_j)+1); rho_j=omega_j y_j^2 f0_j(1-f0_j),
y_bar=sum(rho_j y_j)/sum(rho_j),
s_y=sqrt(sum(rho_j (y_j-y_bar)^2)/sum(rho_j)).
```

Define `<a,b>_rho=sum_j ((rho_j a_j) b_j)/sum_j rho_j`. All binding binary64 sums use the
balanced pairwise tree over increasing node index, all products are evaluated left-associated
without FMA, and `sqrt` is correctly rounded binary64. Set `z=(y-y_bar)/s_y`. Candidates are
the coefficient vectors of `v0=1`, `v1=z`, and `v_m=z^m`; node powers are formed only by the
left-associated recurrence `power_m=power_(m-1)*z`, never `z**m`.

The binding weighted QR is two-pass modified Gram--Schmidt with no pivoting, columns in increasing
`m`, and the normalized inner product above using the same pairwise tree. Keep `psi_0=1`
literally and `psi_1=z`; for each higher candidate subtract
`<prior,candidate>_rho/<prior,prior>_rho` against `psi_0..psi_(m-1)` in that order twice from both
its node values and its `z`-power coefficient vector. Divide both by the same single computed
`sqrt(<candidate,candidate>_rho)`. Choose its sign so the first increasing-node component with
`abs(value)>64*epsilon64*max_j abs(value)` is positive, and apply that sign to both
representations. Rank failure or absence of such a component rejects the row. No BLAS/LAPACK
routine participates.

Every consumer, including projection nodes, outgoing event momenta, the 4097 reference grid,
mass assembly, and reconstruction, evaluates the stored coefficient vector at arbitrary
`0<=y<=Y_max` by descending-degree Horner `value=value*z+coefficient`. Node-value lookup and a
different interpolant are forbidden. Thus one polynomial represents each `psi_m` on and off the
design grid.

For finite modal coefficients

```text
eta_s(y_j)=sum_m beta[s,m] psi_m(y_j);  f_s(y_j)=logistic(eta_s(y_j)),
M_s[i,m]=sum_j omega_j y_j^2 f_s(j)(1-f_s(j)) psi_i(j) psi_m(j),
M_s beta_dot_s=q_s.
```

The state has `6*M` coefficients. Modal and matrix sums use increasing index plus the canonical
tree. The solve is a serial lower Cholesky (`i`, then `j<=i`, then `k<j` order), followed by
forward/back substitution; no pivot, equilibration, BLAS, or FMA is permitted. Nonpositive pivot
or nonfinite value is a raw failure. The directed MPFR condition certificate solves every
canonical unit column with that same interval Cholesky/substitution graph. Because both `M` and
`M^-1` are symmetric, it uses the rigorous bound
`cond_2(M)<=||M||_infinity ||M^-1||_infinity`, with each absolute row sum rounded upward.
An interval pivot not strictly positive or a nonfinite bound is raw, and the outward
`cond2_upper` must be `<=1e6`. The smooth directional
derivative is exactly `M^-1(Dq-DM M^-1 q)`; thresholds have only the §5 one-sided contract.

## 7. Canonical event and arithmetic ordering

The event key, formed before concrete mu/tau labels are substituted, is

```text
(orbit_id, unordered_initial_species_pair, unordered_final_species_pair,
 radial1_index, radial2_index, mu12_index, mustar_index, phi_index,
 deposited_species, test_mode, leg_index).
```

Keys are lexicographically sorted by their integer/enum fields. Each scalar sum uses one balanced
binary tree: adjacent elements are added left-to-right, an odd last element is promoted unchanged,
and levels repeat until one value remains. Equal mu/tau inputs must give bitwise-equal event
multisets, `M`, `q`, and solves. A separately constructed block permutation must satisfy the
structural norms in §10. Naive and Neumaier sums are retained diagnostics only.

The source `invariant_event_prefactor` is the binding reduction of §2: it multiplies the two
incoming `p^2 dp/[2E(2*pi)^3]` measures, fixed isotropic angles `(4*pi)(2*pi)`, §5
`dPhi_2`, `O_r S_r`, the matrix element, radial/angular node weights, and finally
`2*pi^2/T_cm^3`, strictly in its printed left-associated order. `event_leg_contributions` forms
the same `J` once and deposits every explicit neutrino leg; `reduce_event_contributions` sorts by
the complete key and applies the canonical tree. These three functions are the A0 q graph; no
separate tagged evaluator or hidden multiplicity participates.

## 8. Fully preregistered basis/grid/event ladder

The Cartesian product is

```text
M=[4,8,12,16,24]
(N_q,Y_max)=[(48,24),(64,28),(80,32),(96,40)].
```

Its 20 rows are tried in increasing lexicographic key `(N_q*M,N_q,Y_max,M)`. No other row exists.
For every radial variable, base events use affine Gauss--Legendre order 64 on the declared finite
interval; reference events use order 96. Angular rules are base/reference
`mu12=(12,16)`, `z*=(12,16)`, and midpoint trapezoid `phi*=(16,32)` on `[0,2*pi)`.
Both base and reference collision rules use neutrino radial intervals `[0,Y_max]`; reference
means higher order on the same represented domain, never off-domain basis evaluation. Electron
`x` uses `[0,40]`. A separate analytic-target tail oracle extends neutrino inputs beyond
`Y_max`; adversarial coefficient states are zero-extended by definition. Every tail uses
`z=b+(1+u)/(1-u)`, `u in [-1,1)`, with `b=Y_max` for analytic neutrino representation/input
tails and `b=40` for electron tails. The directed 256-bit integrator
uses interval Gauss--Legendre 32, always bisects the leftmost widest interval first with a
left-child-before-right-child tie, and stops only when the sum of active interval widths is
`<=2^-180*max(total_upper,2^-200)`, with at most `2^18` leaves. On the final interval adjacent to
`u=1`, electron FD uses `f_e(x)<=exp(-x)`. Every named analytic neutrino profile in §9 uses the
single prospective envelope
`f_nu(z)<=exp[-z+0.10(1+z^2)exp(-z/6)]`, obtained from `f<=exp(eta)` and a termwise upper bound
on all listed perturbations. A nonfinite endpoint, leaf cap, or unmet width criterion is a raw
failure.

Stage A, which has no collision evaluation, projects each analytic state in §9 and reconstructs
`f_M(y)=logistic(sum_m beta_m psi_m(y))` using only the §6 Horner evaluator. On affine
Gauss--Legendre 256 nodes on `[0,Y_max]`, with directed target tail
`T=[T_lower,T_upper]=integral_Ymax^infinity y^2 f_target(y)dy`, define

```text
E_L1_upper = [sum_j w_j y_j^2 |f_M(y_j)-f_target(y_j)| + T_upper]
             / [sum_j w_j y_j^2 f_target(y_j) + T_lower].
```

This is the zero-extension error of the finite-support representation; a nonpositive denominator
is raw. On the 4097 points `y_k=k*Y_max/4096`, define
`E_point=max |f_M-f_target|/f_target` only where `f_target>1e-8`; an empty resolved set is raw.
The representation errors are evaluated for every analytic target/species and its mu/tau
permutation partner. The mass-matrix condition certificate is evaluated on all projected analytic
states plus all 100 adversarial coefficient states and the complete `P` closure. A row is
Stage-A eligible only if the worst analytic `E_L1_upper<=2.5e-4`, worst analytic
`E_point<=1e-2`, and worst all-state `cond2_upper<=1e6`. Interval overlap with a cap is a failure,
not an inconclusive pass.

Only after the later W7 execution and its fresh adjudication may Stage B evaluate every Stage-A
eligible row. For each frozen state and temperature, let `q_base` use the base rules,
`q_ref` use the reference rules on that same represented domain, and `q_lost_tail` be the
directed signed enclosure of all reference contributions whose evolved neutrino leg lies outside
the row's `[0,Y_max]`, plus analytic-target neutrino input tails beyond `Y_max` and electron
tails beyond 40. The adversarial input tail is exactly zero under its declared zero extension.
With one component domain and

```text
D_q=max(||q_ref||_infinity,2^-40 G_F^2 T_cm^5),
E_collision=||q_base-q_ref||_infinity/D_q,
E_lost_tail=sup ||q_lost_tail||_infinity/D_q,
```

the maxima over every state/species/mode/temperature must be `<=5e-3` and `<=2.5e-3`.
Stage B must seal results for exactly every Stage-A-eligible row before selection. The selected
row is the first passing row in the frozen 20-row order. Missing/extra rows, a branch/interval
failure, or no passing row rejects B3. No order/domain retry is allowed. This document freezes
the selection algorithm; it does not execute either stage.

## 9. Frozen state and support-vector generator

Analytic profiles are projected by `beta_m=<psi_m,eta_target>_rho` in increasing `m`; no fit or
optimizer is used.

- E0: `eta=-y` for all six species at `T_gamma=T_cm` in `{10,5,3,1,0.1}` MeV.
- S-self: `eta=-y+0.10*y*exp(-y/4)` for all species at 3 MeV.
- S-split at `T_gamma=T_cm=3 MeV`: electron pair
  `-y+0.08*y*exp(-y/3)`; muon pair
  `-y-0.05*y^2/(1+y^2)*exp(-y/5)`; tau pair
  `-y+0.03*sin(y/2)*exp(-y/6)`.
- S-electron: neutrinos `eta=-y` at `T_cm=1 MeV`, `T_gamma=1.2 MeV`.
- CP controls: at 3 MeV add `+0.02*exp(-y/4)` to `nu_e` and the negative to `antinu_e`, then the
  sign-reversed partner; all other species use `-y`.

The 100 adversarial states use SplitMix64 seed `0xBD622B3A7C0FFEE1`. For each candidate, generate
`6*M` coefficients species-major/mode-minor with the standard SplitMix64 transitions
`+0x9E3779B97F4A7C15`, xor-shifts `30,27,31`, and multipliers
`0xBF58476D1CE4E5B9`, `0x94D049BB133111EB`; map the upper 53 bits as
`u=((z>>11)+0.5)*2^-53`, `beta=(2*u-1)/4`. Accept in stream order only if
`max|eta|<=20` at all design nodes and 4097 reference points. No reseeding is allowed.
The evaluation domain is the bitwise-deduplicated union of every analytic/adversarial state and
its exact mu/tau block permutation `P beta`; thus the domain is closed under `P` even when a
random draw has no independently drawn partner. The 100 count names the accepted generator rows,
not the size after this mandatory closure.

For each grid, support vectors contain every GL node and `nextafter(node,-infinity/+infinity)`,
zero and its positive successor, and `Y_max` with both neighbours. The four domain triples in
binary64 hex are

```text
24: (0x1.7ffffffffffffp+4,0x1.8000000000000p+4,0x1.8000000000001p+4)
28: (0x1.bffffffffffffp+4,0x1.c000000000000p+4,0x1.c000000000001p+4)
32: (0x1.fffffffffffffp+4,0x1.0000000000000p+5,0x1.0000000000001p+5)
40: (0x1.3ffffffffffffp+5,0x1.4000000000000p+5,0x1.4000000000001p+5).
```

Threshold controls use COM inputs with `s` equal to, immediately below, and immediately above
`m_e^2` (elastic) or `4m_e^2` (pair), where neighbours are binary64 `nextafter`. Each generated
state/vector bundle is hashed before W7 execution; these algorithms, not a mutable output file,
are its source of truth.

## 10. W7 oracle and D-030 arithmetic/metrology graph

W7 imports no RABBIT production code. Its continuum axis derives the direct row-ii coherent
`t+s` v-spinor trace, every `2*pi`/spin/symmetry/orbit factor, all §4 finite-mass amplitudes,
the §3 graph, entropy affinity, the §5 shell/Jacobian, and directed tails. Its arithmetic axis
replays the exact B3 graph at MPFR 256 bits.

The reviewed, deliberately unexecuted contract sources are
`scripts/audit/w7_b3v2_contract_source.py` at SHA-256
`60fc9668a72bba0ef576138c17b1bfe4f435bc215b8850dde4ac424cdb66dfbe` and
`docs/audit/BD622_W7_exact_test_vectors.json` at SHA-256
`e991d415284da9f6d552e6d011739b0948aef03a3a47051e425602fcd3c6e3cd`.
They are `SPECIFIED_NOT_EXECUTED`, not W7 evidence.

The future isolated execution is pinned to CPython 3.12.3 on x86-64 glibc and
`gmpy2==2.3.1`; the PyPI CPython-3.12 manylinux x86-64 wheel is
`gmpy2-2.3.1-cp312-cp312-manylinux2014_x86_64.manylinux_2_17_x86_64.manylinux_2_28_x86_64.whl`,
SHA-256 `9e9cc3703df3bf740f87744cd9816c3bb65da68a605d31d38173c38414dfb516`.
Set `OMP_NUM_THREADS=OPENBLAS_NUM_THREADS=MKL_NUM_THREADS=NUMEXPR_NUM_THREADS=1`; binding code uses
no BLAS. Record kernel, glibc, CPU, Python, GMP, MPFR, MPC, and gmpy2 versions at execution.

Before any W7 result, the exact binary64 evaluator must emit one dynamic arithmetic trace in
execution order. Every node records contiguous `node_id`, owner, opcode, earlier operand IDs,
the full canonical loop key, and whether it is a declared rounding interface. Owners are exactly
`input`, `event_primitive`, `canonical_reduction`, `gl_root`, `affine_rule`, `weighted_qr`,
`basis_evaluation`, `reconstruction`, `mass_assembly`, `cholesky`, `substitution`, and
`final_reconstruction`. Opcodes are the source's fixed add/subtract/multiply/divide/sqrt/
exp/sin/cos/branch/pairwise-add/binary64-round semantics. The source
`validate_arithmetic_trace` rejects a non-topological trace, a noncanonical loop order, missing
owner, unknown opcode, invalid final output, or undeclared interface. The complete trace, its
operation counts by owner/opcode, and final-output IDs are sealed before observable values are
read. `replay_arithmetic_trace` then executes exactly those nodes under A0--A4, requires every
directed-to-binary edge to terminate at a declared interface, and raw-fails any denominator,
branch, root, or transcendental interval that cannot be resolved under the frozen policy.

For each state and its exact mu/tau-permuted partner define the dimensionless native observable

```text
O[s,j] = f_s(j)(1-f_s(j)) * sum_m psi_m(j) beta_dot[s,m]
         / (G_F^2 T_cm^5).
D_pair = max(||A0(beta)||_inf,||A0(P beta)||_inf,
             sup||A4(beta)||_inf,sup||A4(P beta)||_inf,2^-40).
```

Every stage uses that same `beta/P beta` component domain and `D_pair`; no component, state, or
stage changes denominator. Directed MPFR intervals enclose consecutive same-node stages. The
stage policies are executable data in `STAGE_POLICIES` and are exactly:

```text
A0: replay every trace node as binary64 round-to-nearest-ties-even, without FMA.
A1: promote event_primitive and canonical_reduction to directed MPFR-256; round once only at each
    completed event-contribution and completed pairwise-reduction interface.
A2: additionally promote gl_root, affine_rule, weighted_qr, basis_evaluation, and reconstruction;
    round once only at the completed basis object, projected state, and q-vector interfaces.
A3: promote every non-input owner through final_reconstruction to directed MPFR-256; perform no
    intermediate binary64 rounding and round only the final O interval for reporting.
A4: repeat A3 at MPFR-384, importing every exact binary64 input as an outward singleton; perform
    no intermediate binary64 rounding.
```

At A2 the Legendre root interval follows the §6 sign-bracket/derivative/bisection contract; a
nearest Newton iterate is not an enclosure. At A3/A4 the condition certificate follows the fixed
directed Cholesky/inverse-column/norm graph. Every elementary operation follows §§5--7.
Transcendentals use MPFR
directed rounding; binary64 inputs are imported exactly; interval subtraction uses all four
endpoint combinations. An interval containing zero in a denominator or Cholesky pivot, crossing
a branch/support boundary, losing a root sign/uniqueness certificate, or containing a nonfinite
value is a raw failure, never widened until it passes.

For each pair-domain component, define

```text
B_sum      = max sup|A0-A1|/D_pair,
B_basis    = max sup|A1-A2|/D_pair,
B_solve    = max sup|A2-A3|/D_pair,
B_interval = max [radius(A3)+radius(A4)+|mid(A3)-mid(A4)|]/D_pair.
```

```text
B_fp(n)=128*(ceil(log2(max(n,1)))+8)*epsilon64       # reported cross-check only
B_native=4*(B_sum+B_basis+B_solve+B_interval).
```

The factor four is frozen before output. A4 must additionally prove that, after applying the
recorded block permutation, the `beta` and `P beta` trace IDs/opcodes/loop keys are identical and
their corresponding directed intervals overlap componentwise; otherwise the arithmetic graph
fails. This establishes the high-precision middle equality required by the triangle bound. The
binding native defect is

```text
R_native=||P A0(beta)-A0(P beta)||_infinity/D_pair,
```

and passes only if `R_native<=B_native`. W7 must write the four terms, every `D_pair`, worst
state/component, all A0--A4 endpoints, trace and operation counts, source/vector/environment
hashes, and the A4 permutation certificate before any B3 collision result exists. D-028's cap is
never consulted.

The binding structural covariance tests are separately

```text
R_q=||P q(beta)-q(P beta)||_inf /
    max(||q(beta)||_inf,||q(P beta)||_inf,2^-40 G_F^2 T_cm^5),
R_M=||P[M(beta)beta_dot(beta)]-M(P beta)P beta_dot(beta)||_inf / same_denominator,
```

with cap `1e-10` for both. `P` is the exact mu/tau block swap. The resolved physical/native
metric uses `O` and `D_pair`. All frozen states are closed under `P`. This keeps the structural
identity distinct from conditioning and makes the native triangle proof use one domain and one
denominator.

## 11. Exact CAS vectors and four-axis boundary

The run-local `CAS_CONTRACT.json` must reproduce these exact vectors:

- `V-CONSTANTS-D035`: exact decimal strings `G_F=1.1663788e-11 MeV^-2`,
  `m_e=0.5109989500 MeV`, and `sin2_theta_W=0.23122`.
- `V-ORIENTATION-LEDGER`: distinct elastic `(O,S,global legs,effective tagged)` is
  `(1/2,1,2,1)`; identical elastic is `(1/2,1/4,4,1/2)`, mapping global coefficient 128 to
  tagged 64; conversion/pair has `O=1`.
- `V-ELASTIC-THRESHOLD` and `V-PAIR-THRESHOLD`: exact binary64 neighbours below/equal/above
  `s=m_e^2` and `s=4m_e^2` must respectively raw-fail, return support-only `k*=0`, and return an
  interior `k*>0` result.
- `V-ZSTAR-DOMAIN`: exact endpoints `-1,1` are accepted; their immediate outside binary64
  neighbours are rejected without clipping.
- `V-NULL-COM-345`: massless COM momenta
  `(1,0,0,1),(1,0,0,-1),(1,3/5,0,4/5),(1,-3/5,0,-4/5)`;
  expected `(K_s,K_t,K_u)=(4,81/25,1/25)`.
- `V-ELASTIC-ME-345` with `m_e=1`:
  `(3/4,0,0,3/4),(5/4,0,0,-3/4),(3/4,9/20,0,3/5),
  (5/4,-9/20,0,-3/5)`; expected shells, conservation, `s+t+u=2`.
- `V-PAIR-ME-345` with `m_e=1`:
  `(5/4,0,0,5/4),(5/4,0,0,-5/4),(5/4,9/20,0,3/5),
  (5/4,-9/20,0,-3/5)`; expected shells, conservation, `s+t+u=2`.
- `V-ENTROPY-RATIONAL`: occupations `(1/5,2/5,3/5,4/5)`; expected
  `F=4/625`, `R=144/625`, `J=-28/125`, `A=log(1/36)`, `A*J>0`.
- `V-NULLSPACE-SPECIES`: the three left-null vectors printed in §3.

Blind Wolfram+xAct, Sage+Singular, Lean, and SymPy axes share metric/gamma conventions, domains,
branches, canonical forms, and these vectors, but not sibling work. A missing engine theorem is
`INCONCLUSIVE`, not a contradiction and not a pass. The exact nullspace, three-axis row-ii, and
three-axis entropy results from the failed prior run remain evidence but do not pre-answer this
fresh review.

## 12. Static falsifiers and result schema

| id | binding obligation |
|---|---|
| T01 | exact graph, factors, direct row-ii trace, all-channel absolute normalization |
| T02 | raw common-FD affinity/action at all E0 temperatures; no zero overwrite |
| T03 | self redistribution, record-valid number/energy invariants, entropy |
| T04 | electron elastic/pair exchange, charge/number, and `Q_nu+Q_EM` |
| T05 | exact stoichiometric nullspace and eventwise first law |
| T06 | total neutrino-plus-bath entropy on named and 100 adversarial states |
| T07 | equal-block bitwise mu/tau, arbitrary swap/CP, electron/muon negative control |
| T08 | shell/support/node/tail continuity and physical-side derivatives |
| T09 | canonical binary64 versus MPFR; naive/Neumaier diagnostics retained |
| T10 | weak, mass-weighted, native, and resolved-pointwise metric bundle |
| T11 | `D(M^-1q)` smooth ladder and threshold one-sided derivative |
| T12 | independent modal/grid/event/tail budgets and fixed selection |

Each eventual JSONL record has exactly

```text
test_id,state_id,source_sha256,oracle_sha256,inputs_sha256,value_bits,norm,
denominator_bits,cap,cap_derivation,conditioning,arithmetic_mode,raw_failure,pass,utc
```

Any binding failure preserves raw evidence and stops. No cap, state, norm, grid, reduction,
branch, or tolerance retry is allowed.

## 13. Staged authority and failure classes

1. **Current authority:** amend/review these design bytes and draft unexecuted W7 source/test
   vectors; perform a fresh registered blind design/CAS adjudication only.
2. **Fresh design PASS plus explicit owner decision:** execute W7 microphysics/arithmetic only,
   instantiate the numeric §10 ledger, and perform a new blind adjudication. Still no B3 output.
3. **Executed W7 PASS, fresh adjudication PASS, and a new explicit owner decision:** implement B3
   privately and execute T01--T12 only.
4. **T01--T12 PASS plus fresh owner decision:** one independent 10-to-3 MeV Radau segment.
5. **Segment PASS plus fresh owner decision:** full `T_gamma=0.005 MeV` independent endpoint,
   sealed before RABBIT unblinding.

Failure classes are fixed:

- W7 normalization/trace conflict: `MICROPHYSICS_AUTHORITY_CONFLICT`; no B3 output.
- Rust raw exact-FD failure: `RUST_PHYSICS_SUSPECT`; no overwrite, clip, or anchor update.
- B3 weak invariant/entropy/rank failure: `INDEPENDENT_DISCRETIZATION_SUSPECT`.
- B3 required native diagnostic failure: `INDEPENDENT_METROLOGY_SUSPECT`.
- uncertainty above half a later hard endpoint cap: `INCONCLUSIVE`, never PASS.

PCED is only a fallback design family. A B3 failure does not authorize its implementation.

## 14. Claim and cost boundary

Until T01--T12 pass, B3-v2 remains `PROPOSED`; later source existence alone is `IMPLEMENTED`, not
`VALIDATED`. W7 can be `VALIDATED` only inside its executed and adjudicated contract.
`G-F10-INDEPENDENT-FLRW` and `G-F10-COVARIANCE-METROLOGY` remain `FAIL`. F-11, QKE, and public
production remain `FORBIDDEN`.

Every retained patch reports added/deleted/net lines, commands/raw failures, blocker-movement
ratio, and cost verdict. Exact token use is
`UNAVAILABLE — no reliable stage-scoped counter`. No new readiness/dashboard/manifest gate is
created; existing SSOT/gate entries are amended in place.
