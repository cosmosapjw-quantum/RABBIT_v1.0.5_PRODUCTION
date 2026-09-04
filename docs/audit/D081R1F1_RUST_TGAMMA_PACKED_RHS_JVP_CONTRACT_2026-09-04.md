# D-081R1F1 — Rust `T_gamma` packed-RHS analytic-JVP contract

**Date:** 2026-09-04  
**Repository:** `cosmosapjw-quantum/RABBIT_v1.0.5_PRODUCTION`  
**Branch:** `research/d081r1f1-rust-tgamma-jvp-20260904`  
**Stacked base:** `research/d081r1f0-rust-c-only-jvp-20260904`  
**Base commit:** `c29b6d26599da0bead66482373d8ec2cdc8f06c4`  
**Base tree:** `f467ea43e78611cef47342d1416e2b372e90d6ac`  
**Status:** `CONTRACT_FROZEN_IMPLEMENTATION_ABSENT_RED_REQUIRED`

## 1. Admitted object

For packed state

\[
y=(c_e,c_\mu,c_\tau,T_\gamma,t_{\rm elapsed})
\in \mathbb R^{3n+2},
\]

and independent variable \(N=\ln a\), let \(F(N,y)\) be the already admitted
Rust packed RHS.  This node may implement only the directional derivative

\[
D_{T_\gamma}F(N,y)[\delta T_\gamma]
=\delta T_\gamma\,\frac{\partial F}{\partial T_\gamma}
\]

at fixed \(N\), fixed neutrino cloglog coordinates \(c\), fixed
\(T_{\rm cm}=10\,\mathrm{MeV}\,e^{-N}\), fixed stored elapsed coordinate, and
fixed collision configuration.

The API is a matrix-free scalar-direction JVP.  It is not a dense Jacobian,
solver callback, trajectory operator, endpoint calculation, or performance
claim.

## 2. Frozen authority map

The implementation and every oracle must fail closed unless these identities
match exactly.

```text
R1F0 parent closeout commit:
c29b6d26599da0bead66482373d8ec2cdc8f06c4

R1F0 retained-holdout receipt blob:
01a87b4227fbe11e83412a5899a1eead69fbda3c

R1F0 symbolic fallback receipt blob:
ba8243404c5932e73b9bd8a5ca380f04a62c41d8

frozen private comparator blob:
de44feee0aa484abe26976c7dc34c579643005b5

D-080C mathematical contract blob:
5fe1c43525189694031723715603207e62501090a

D-080C final v2 receipt blob:
1ebb9a854f1db42d76ae36cfbe0926b016b67b7e

D-080A moving-kinematics/EOS source blob:
c585d5865fd68a90a04a76ab540b8437fba8cfce

D-080B full collision-column source blob:
78489c43f3046db09d8ba2d96070124ed7b0aa91

Rust Cargo.lock blob:
a1b5035da5c20712d1a2a4ab077da255ff94a014
```

The D-080C branch closeout commit is
`718f5c839635f322f5f2e900f057709862a03a93`, tree
`6a34431e142cf4f2541b2f96f4c568dd184e43a7`.  The final classification of
that authority is `FULL_STATIC_TGAMMA_RHS_COLUMN`.

## 3. Units and row-wise dimensions

The repository's existing natural-unit MeV convention is retained.  No unit
conversion is introduced in this node.

```text
c_f                         dimensionless
T_gamma, T_cm, H            MeV
rho, p                      MeV^4
chi_gamma = d rho_em/dT     MeV^3
chi_gamma,T                 MeV^2
Q_em                        MeV^5
Q_em,T                      MeV^4
F_c = dc/dN                 dimensionless
partial F_c/partial T       MeV^-1
F_gamma = dT_gamma/dN       MeV
partial F_gamma/partial T   dimensionless
F_t = dt/dN = 1/H           MeV^-1
partial F_t/partial T       MeV^-2
```

A single Euclidean norm over all output rows is forbidden.  Every comparison
must report separately dimensionless spectral, photon-temperature, and elapsed
row residuals; the admission metric is their maximum only after each block has
been scaled by a same-dimension reference.

## 4. Thermodynamic and packed-RHS derivation

At fixed \(c,T_{\rm cm},N\), only the electromagnetic energy density changes
under the input derivative.  Define

\[
\chi_\gamma=\frac{\partial\rho_{\rm em}}{\partial T_\gamma},
\qquad
\chi_{\gamma,T}=\frac{\partial^2\rho_{\rm em}}{\partial T_\gamma^2},
\qquad
p_{{\rm em},T}=\frac{\partial p_{\rm em}}{\partial T_\gamma}.
\]

Since \(H^2\propto\rho_{\rm total}\),

\[
\boxed{
 h_T\equiv\frac{H_T}{H}
 =\frac{\chi_\gamma}{2\rho_{\rm total}}
}
\]

with units MeV\(^{-1}\).

For a spectral row, write the primal RHS as

\[
F_c=\frac{P}{Hq},
\]

where \(q\) is the already admitted cloglog chart Jacobian and is fixed in a
pure \(T_\gamma\) input direction.  Therefore

\[
\boxed{
F_{c,T}=\frac{P_T}{Hq}-F_c h_T
}
\]

and the directional output is \(\delta T_\gamma F_{c,T}\).

For the photon-temperature row define

\[
\mathcal N_\gamma=-3(\rho_{\rm em}+p_{\rm em})+\frac{Q_{\rm em}}{H},
\qquad
F_\gamma=\frac{\mathcal N_\gamma}{\chi_\gamma}.
\]

Then

\[
\mathcal N_{\gamma,T}
=-3(\chi_\gamma+p_{{\rm em},T})
+\frac{Q_{{\rm em},T}}{H}
-\frac{Q_{\rm em}}{H}h_T,
\]

and

\[
\boxed{
F_{\gamma,T}
=\frac{\mathcal N_{\gamma,T}}{\chi_\gamma}
-F_\gamma\frac{\chi_{\gamma,T}}{\chi_\gamma}
}.
\]

For the elapsed-time output,

\[
F_t=\frac1H,
\qquad
\boxed{
F_{t,T}=-\frac1H h_T
}.
\]

The stored elapsed coordinate is not an input to the static RHS.  Changing
only that stored coordinate must leave the entire \(T_\gamma\) JVP bitwise
unchanged.

## 5. Collision derivative boundary

The neutrino self-interaction action has exact zero \(T_\gamma\) tangent at
fixed \(c,T_{\rm cm}\).  The electron/positron action must differentiate the
same admitted discrete operator, including all of the following terms.

### Elastic channels

- the half-line electron nodes and quadrature weights, both proportional to
  \(T_\gamma\);
- incoming electron energy \(E_2=\sqrt{p_2^2+m_e^2}\);
- the center-of-momentum boost and outgoing four-momenta;
- Kallen and phase-space factors;
- every Minkowski dot product used in the weak matrix element;
- any piecewise matrix roundoff correction on the unchanged branch;
- electron and positron Fermi-Dirac logits;
- moving outgoing-neutrino interpolation locations;
- the derivative of the mapped modal projection basis;
- kinematic energy-transfer weights entering \(Q_\nu\) and \(Q_{\rm em}\).

### Pair channels

At fixed neutrino quadrature and fixed masses, the pair-channel kinematic
batch, measure, and matrix element have zero explicit \(T_\gamma\) tangent.
The outgoing electron/positron occupations and the corresponding Pauli factor
must still be differentiated.  Energy-transfer weights remain those of the
primal pair event.

### Discrete branches

Support masks and matrix-correction predicates are not differentiated.  A
centered witness is admissible only when the plus, base, and minus evaluations
have the same support/domain and correction signature.  A branch change is
classified as `NONDIFFERENTIABLE_DISCRETE_EVENT`; it may not be hidden by
increasing a tolerance or deleting the sample.

## 6. Required Rust interface

The intended public-in-crate interface is

```rust
pub fn evaluate_f10_packed_rhs_tgamma_jvp(
    grid: &F10ActionGrid,
    ln_a: f64,
    state: &[f64],
    delta_tgamma_mev: f64,
    config: F10PackedRhsConfig,
) -> Result<F10PackedRhsTgammaJvp, F10PackedRhsTgammaJvpError>;
```

The result must expose at least

```text
base packed RHS
full directional JVP values, length 3n+2
T_gamma column values per MeV
collision-action T_gamma tangent
Q_nu,T and Q_em,T
delta H/H for the supplied direction
chi_gamma, p_em,T, chi_gamma,T
first-law tangent residual
support/correction branch signature
component reconstruction diagnostics
```

`delta_tgamma_mev = 0` must return an exactly zero directional JVP without
changing or re-evaluating the semantic base result.  Nonfinite directions,
invalid state size, nonpositive temperatures, nonfinite intermediates,
materially negative matrix elements, authority mismatches, and branch-boundary
states must fail closed with distinct error variants.

Suggested internal modules are

```text
f10_tgamma_tangent.rs
f10_electron_action/tgamma_jvp.rs
f10_combined_action_tgamma_jvp.rs
f10_packed_rhs_tgamma_jvp.rs
```

Names may change only if the resulting dependency boundary remains equivalent.
No solver-facing callback belongs in this node.

## 7. Pre-registered fixtures

### 7.1 Order-8 thermal split

```text
order:       8
y_max:       8
T_cm:        2.00 MeV
T_gamma:     2.05 MeV
epsilon MeV: [1e-2, 3e-3, 1e-3, 3e-4, 1e-4]
```

The D-080C Python authority obtained best block residual
`1.730599336173931e-7` at the final ladder point, with unchanged branch.

### 7.2 Order-8 manufactured weak tail

```text
order:       8
y_max:       10
T_cm:        0.45 MeV
T_gamma:     0.50 MeV
epsilon MeV: [1e-3, 3e-4, 1e-4, 3e-5, 1e-5]
```

The D-080C Python authority obtained best block residual
`2.454956136786496e-8`.  This remains a controlled static probe and must not be
called retained late-trajectory evidence.

### 7.3 Exact equilibrium

The equilibrium discriminator must reproduce

```text
Q_nu,T:        +8.606919856943374e-20 MeV^4
Q_em,T:        -8.606919856943374e-20 MeV^4
first law:     0
H_T/H:         0.5231017159993603 MeV^-1
elapsed row:  -2.9682069952782854e20 MeV^-2
```

within the blockwise gates below.

### 7.4 Retained calibration state

```text
source commit:
78f5b091bd8de99bd8d0b1eddd8a0b7978f5da2b

path:
.agent-harness/runs/run-20260805-f10-v3-campaign/v3a_r2/domain/state_1200.npz

Git blob:
af686f69b5eec6bc699f60e0e63fe47e958c0802

SHA-256:
c0ad51adbd34d4fb8408f566c7bc573ca3d5bbd8c1d1f5d89f50df2e6b5d3380
```

This is calibration evidence, not an unseen holdout.

### 7.5 Retained post-calibration holdout

The post-calibration holdout is pre-registered now, before any Rust
\(T_\gamma\) implementation or column output is generated:

```text
source commit:
78f5b091bd8de99bd8d0b1eddd8a0b7978f5da2b

path:
.agent-harness/runs/run-20260805-f10-v3-campaign/v3a_r2/domain/state_2000.npz

Git blob:
cfb17344ae166c01c2e5bcb14acae0d968e49477
```

The first execution workflow must compute and publish its SHA-256 in a
byte-identity seal before generating a Python oracle or Rust result.  The
holdout may be opened only after a terminal retained-calibration receipt.
`state_3000.npz` is reserved for the later full-Jacobian node and is outside
R1F1.

## 8. Frozen numerical gates

These gates are fixed before any Rust \(T_\gamma\) output is observed.

```text
exact-zero direction:                    bitwise zero
stored-elapsed invariance:                bitwise identical at +/-1e40
scalar linearity/additivity:              <= 5e-12
base/component reconstruction:            <= 2e-12
EOS and moving-kinematics parity:          <= 1e-7
collision modal/native component parity:  <= 1e-7
Q_nu,T and Q_em,T parity:                  <= 1e-7
H_T/H parity:                              <= 1e-7
packed spectral block parity:              <= 2e-4
packed photon-temperature row parity:      <= 2e-4
packed elapsed row parity:                 <= 2e-4
first-law tangent residual:                <= 2e-9
charge-conjugation diagnostic:             report, not sole gate
thermal FD ladder minimum block residual:  <= 2e-6
weak-tail FD ladder minimum block residual: <= 2e-6
retained centered witness:                 <= 2e-4
```

The two order-8 finite-difference caps are prospective ten-order-of-magnitude
separators from the weakest admitted granular mutation and remain more than an
order of magnitude above the historical correct Python witnesses.  They are
not cross-language parity caps.

Every centered witness must preserve the exact support/correction signature.
The retained holdout must additionally test mu/tau permutation covariance,

\[
\boxed{
K_T(Sy)=S K_T(y),
\qquad
K_T(y)=\frac{\partial F}{\partial T_\gamma}(y)
},
\]

with component-modal covariance cap `1e-7` and packed blockwise covariance cap
`2e-4`.

## 9. Required mutation kills

For the order-8 thermal split, each mutation must produce

\[
r_{\rm mut}\ge\max(10^{-4},100\,r_{\rm correct}),
\]

using the same blockwise metric and unchanged branch.  Required mutations are

```text
omit collision derivative
omit all Hubble feedback
omit spectral-row Hubble feedback
omit photon-temperature-row Hubble feedback
omit elapsed-row Hubble feedback
omit chi_gamma,T heat-capacity derivative
reverse Q_em,T sign
swap photon-temperature and elapsed output rows
freeze electron quadrature nodes
freeze electron quadrature weights
freeze outgoing interpolation locations
omit modal-basis derivative
omit matrix-element tangent
omit kinematic energy-weight tangent
make the elapsed-input column nonzero
```

A mutation that changes support/correction branch is not credited as a
scientific kill for the smooth-column claim; it must be reported separately.

## 10. TDD and execution order

The only admissible order is

```text
R1F1-RED
  temporary absent-API compile probe; preserve exact compiler failure receipt

R1F1-P0
  Rust EOS, moving-rule, kinematic, matrix, and interpolation primitives

R1F1-P1
  electron/positron collision-action T_gamma JVP

R1F1-P2
  packed-RHS T_gamma JVP, exact-zero, linearity, units, reconstruction

R1F1-O8
  thermal split, equilibrium, weak-tail, mutation matrix

R1F1-RCAL
  retained state_1200 calibration and durable receipt

R1F1-RHOLD
  only after RCAL: seal and open state_2000, then run retained holdout and
  mu/tau covariance

R1F1-AUDIT
  independent PHYS-MATH and PHYS-MATH-CODE review
```

The RED probe must be executed before implementation.  It may be generated in
a temporary workflow worktree so that the feature branch itself remains
buildable.  No holdout fixture or result may be generated during RED, P0, P1,
P2, O8, or a failed RCAL.

## 11. Stop conditions

```text
AUTHORITY_MISMATCH
STATE_BYTE_IDENTITY_MISMATCH
NONDIFFERENTIABLE_DISCRETE_EVENT
BASE_RECONSTRUCTION_FAILED
COMPONENT_RECONSTRUCTION_FAILED
EOS_OR_KINEMATICS_PARITY_FAILED
COLLISION_COLUMN_PARITY_FAILED
PACKED_COLUMN_PARITY_FAILED
CONSERVATION_TANGENT_FAILED
MUTATION_NOT_KILLED
RETAINED_CALIBRATION_FAILED
RETAINED_HOLDOUT_FAILED
SYMMETRY_COVARIANCE_FAILED
```

No stop condition authorizes a tolerance increase, support projection,
comparator change, or physics change in the same admission run.  A repair
requires a new bounded node with the failed result preserved.

## 12. Claim ceiling and next node

A successful R1F1 establishes only the analytic Rust \(T_\gamma\) input JVP of
the admitted static packed RHS on fixed support/correction branches.

It does not establish

```text
full 3n+2 square Jacobian
arbitrary combined-direction JVP
sparse Jacobian pattern
transpose action
solver callback
Newton/BDF convergence
trajectory or endpoint
N_eff
performance
release or publication readiness
G-F10-INDEPENDENT-FLRW movement
```

After R1F1 closes, the next node may compose the already admitted spectral
operator and thermal column,

\[
Jv=J_c v_c+v_T K_T,
\]

with an exact zero elapsed-input contribution, and then test matrix/JVP
identity before any solver integration.