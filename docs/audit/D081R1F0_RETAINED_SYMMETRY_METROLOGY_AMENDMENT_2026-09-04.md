# D-081R1F0 retained symmetry-metrology amendment

Date: 2026-09-04  
Repository: `cosmosapjw-quantum/RABBIT_v1.0.5_PRODUCTION`  
Original contract blob: `ac7149fe5d5ec327cdc168d1eba7fe4a68ce3221`  
Original calibration workflow: `33835819255`  
Diagnostic workflow: `33837198642`  
Diagnostic source commit: `29ee91a27536450d0ddce59f4f62b7a8dba0a77f`  
Status: **PROSPECTIVE AMENDMENT BEFORE UNSEEN HOLDOUT**

## 1. Preserved original failure

The original contract required, in the symmetric retained calibration lane,

\[
r_{\mu\tau}\le 2\times 10^{-9},
\]

where

\[
r_{\mu\tau}
=
\frac{\|\bar C_\mu-\bar C_\tau\|_\infty}
{\max(\|\bar C_\mu\|_\infty,\|\bar C_\tau\|_\infty)},
\qquad
\bar C_f=\frac{C_{\nu_f}+C_{\bar\nu_f}}2.
\]

The exact calibration run failed this gate with the Rust value

```text
2.53056882704695140e-9 > 2.00000000000000012e-9.
```

The failure is retained and is not reclassified as a PASS under the original
metric.

The diagnostic then established that the frozen Python authority also fails
the same threshold:

```text
Python stored/recomputed residual: 2.52375045358395133e-9
Python numerator:                  3.21284039573526329e-29
Python scale:                      1.27304202805500941e-20

Rust residual:                     2.53056882704695140e-9
Rust numerator:                    3.22152047171665042e-29
Rust scale:                        1.27304202805501452e-20

Rust/Python mu-pair array residual:  1.33925624268166116e-11
Rust/Python tau-pair array residual: 6.57418896388758385e-12
```

Therefore the original scalar gate is reference-self-inconsistent. Raising
`2e-9` after seeing the output is forbidden. The raw scalar remains recorded
as a diagnostic.

## 2. Conditioned cross-language consistency

Let Rust and Python pair-averaged arrays be
\((\mu_R,\tau_R)\) and \((\mu_P,\tau_P)\). Define

\[
n_X=\|\mu_X-\tau_X\|_\infty,
\qquad
s_X=\max(\|\mu_X\|_\infty,\|\tau_X\|_\infty),
\qquad
r_X=n_X/s_X,
\]

and

\[
\delta_\mu=\|\mu_R-\mu_P\|_\infty,
\qquad
\delta_\tau=\|\tau_R-\tau_P\|_\infty.
\]

The reverse triangle inequality gives

\[
|n_R-n_P|\le\delta_\mu+\delta_\tau,
\]

and

\[
|s_R-s_P|\le\max(\delta_\mu,\delta_\tau).
\]

Hence

\[
\boxed{
|r_R-r_P|
\le
\frac{\delta_\mu+\delta_\tau}{s_R}
+
\frac{n_P\max(\delta_\mu,\delta_\tau)}{s_Rs_P}
+B_{\rm round}
}
\]

with an explicit binary64 evaluation-roundoff allowance

\[
B_{\rm round}=64\epsilon_{64}\max(1,|r_R|,|r_P|).
\]

For the preserved diagnostic the observed scalar difference is

```text
|r_R-r_P| = 6.8183734630000724e-12,
```

while the conservative bound evaluated from the reported array residuals is
approximately

```text
1.9980962279218882e-11.
```

This is a conditioning statement, not a replacement physical threshold.

## 3. Amended calibration gates

The retained calibration may proceed only if all of the following pass:

1. the Python stored residual equals the residual recomputed from the frozen
   Python arrays;
2. the Rust stored residual equals the residual recomputed from the Rust
   arrays;
3. the Rust/Python mu and tau pair-average arrays each pass the already frozen
   retained component parity cap;
4. `|r_R-r_P|` is no larger than the derived propagation bound above;
5. the raw Rust and Python residuals, numerators, scales, array residuals, and
   propagation bound are printed and preserved in the receipt;
6. all original modal, packed-JVP, first-law, self-number, self-energy,
   support/correction, and centered-witness gates remain unchanged.

The original `2e-9` result remains a recorded failed legacy diagnostic and is
not silently deleted.

## 4. Physical symmetry gate moved to a well-conditioned covariance test

The physical statement is equivariance under the mu/tau permutation `S`:

\[
J(Sy)\,Sv=S\,J(y)v.
\]

A tiny difference of two nearly equal outputs in one symmetric direction is a
poor standalone floating-point test of this statement. Before the unseen
holdout output is generated, the holdout contract must therefore add a
non-symmetric preregistered direction and its mu/tau-swapped partner. The
following remain frozen before holdout inspection:

- self, electron, and total modal covariance residuals use the existing
  retained component cap `1e-7`;
- the packed-RHS JVP covariance residual uses the existing retained packed-JVP
  cap `2e-4`;
- state, direction, support, and matrix-correction branches are permuted and
  compared explicitly;
- the raw symmetric-lane `r_mu_tau` diagnostic remains reported.

If covariance fails, this amendment does not authorize another tolerance
change. The outcome is `SYMMETRY_COVARIANCE_FAILED`.

## 5. Non-changes

This amendment changes no:

- collision equation, reaction catalogue, coefficient, sign, or species order;
- state, direction, grid, quadrature, support predicate, or correction policy;
- Rust production JVP implementation;
- finite-difference ladder;
- modal, packed-JVP, first-law, conservation, or centered-witness thresholds;
- solver or trajectory setting.

## 6. Claim ceiling

A successful amended calibration establishes only cross-language consistency
of the fixed-state, fixed-grid, fixed-support spectral-`c` JVP. Physical
mu/tau equivariance is not admitted until the preregistered swapped-direction
covariance holdout passes. No full Jacobian, solver, trajectory, endpoint,
`N_eff`, performance, publication, main-integration, or
`G-F10-INDEPENDENT-FLRW` claim follows.
