# BD623 D-079 — Cloglog Event-JVP Static Physical Result

Date: 2026-09-01  
Base: `plan/ode-r1-r3-remediation-v2-20260824` at merged D-078 commit `4fd19764a0acfc60955b8ed819158e599867321f`  
Validated evidence commit: `a13aee2d2be68bd461d744f0c78ff559bae8f766`  
Evidence tree: `82516f0dc3b132126165a653d75630c7ca2807fb`  
Workflow: `d079-static-physical-jvp`, run `33523417075`, **SUCCESS**  
Verdict: **D-079 VALIDATED; G-F10-INDEPENDENT-FLRW REMAINS FAIL**

## 1. What was completed

D-079 implements an analytic directional derivative of the frozen private
no-QKE collision action with respect to all three native complementary-log-log
spectral blocks. It differentiates every retained self-interaction and
electron/positron event through the actual event quadrature, then pushes that
JVP through the complete static RHS at fixed independent variable `N`, fixed
`T_cm(N)`, and fixed input `T_gamma`.

The comparator source, reaction data, kinematics, matrix elements, support
masks, quadrature, interpolation, modal reconstruction, thermodynamics, Hubble
law, and failure semantics remain unchanged. No ODE integrator was called by
the research probe.

## 2. Exact mathematical result

For

\[
 c=\log[-\log(1-f)],\qquad
 q=\frac{df}{dc}=e^{c-e^c},
\]

Wolfram exact differentiation verified

\[
 \frac{du}{dc}=\frac{e^c}{f},\qquad
 \delta\log q=(1-e^c)\delta c.
\]

For the Pauli event factor `P=G-L`, with reaction affinity
`a=u_3+u_4-u_1-u_2`, the implemented exact tangent is

\[
 \delta P=P\,\delta\log L+G\,\delta a.
\]

At detailed balance, this reduces to `delta P=G delta a`; therefore a
cancellation-small base collision factor is not incorrectly promoted to a zero
tangent.

For the static spectral RHS `g=C/(Hq)`,

\[
 \delta g=\frac{\delta C}{Hq}
 -g\left(\frac{\delta H}{H}+\delta\log q\right),
\]

with the neutrino-energy contribution to `delta H/H` and the induced
photon-temperature and elapsed-time output-row tangents included. Units are
consistent in the comparator's natural-unit MeV convention: `c`, `f`, and `N`
are dimensionless; `H` and collision actions have MeV; `dT_gamma/dN` has MeV;
`dt/dN` has MeV^-1.

## 3. Machine results

The deterministic small-grid physical probe used order 8, `y_max=8`,
`T_cm=2.0 MeV`, and `T_gamma=2.05 MeV`.

| Quantity | Result |
|---|---:|
| best collision-action directional residual | `1.1255297778462666e-10` |
| best complete static-RHS directional residual | `3.073282849486843e-11` |
| differentiated first-law residual | `0.0` |
| self-collision number tangent residual | `1.227202594608874e-16` |
| self-collision energy tangent residual | `5.619621284928538e-16` |
| CP tangent residual | `3.252500847037205e-14` |
| mu/tau tangent residual | `1.5043182973550342e-14` |

The centered-difference ladder has the expected truncation/roundoff U-shape.
Both collision and complete-RHS residuals improve approximately quadratically
from `epsilon=1e-2` to a common optimum near `epsilon=1e-4`; decreasing epsilon
further no longer improves the witness. This is evidence for a resolved local
derivative window, not permission to use an arbitrary forward difference.

## 4. Adversarial mutation results

| Candidate | Residual against best centered witness | Verdict |
|---|---:|---|
| correct analytic JVP | `1.1255297778462666e-10` | survives |
| sign mutation | `1.9999999999016218` | killed |
| 1% scale mutation | `9.900990001606035e-3` | killed |
| flavour-index swap | `1.0309216161977177` | killed |
| omitted electron block | `8.242830692183193e-1` | killed |

The weakest listed mutant is separated from the correct JVP by about eight
orders of magnitude. The discriminator is therefore not merely checking that
an array is finite or roughly scaled.

## 5. Retained stalled-region discriminator

The workflow fetched the exact retained diagnosis branch
`78f5b091bd8de99bd8d0b1eddd8a0b7978f5da2b`, verified the archived
`state_1200.npz` SHA-256
`c0ad51adbd34d4fb8408f566c7bc573ca3d5bbd8c1d1f5d89f50df2e6b5d3380`,
and evaluated the order-60, `y_max=30` state at
`N=0.16286930247517223`.

A normalized local spectral direction with centered step `3e-6` stayed inside
the strict-open occupation domain and passed the original-RHS directional
residual threshold `<2e-4` and the differentiated first-law threshold `<2e-9`.
This removes the specific D-078 blocker “no actual stalled-region physical JVP
has been demonstrated.” It does not prove that BDF with the future full
Jacobian completes the stalled trajectory.

## 6. PHYS-MATH audit

### PASS

- chart definition, sign, and domain are explicit;
- Pauli gain-minus-loss tangent and detailed-balance limit are exact;
- dimensions of collision, Hubble, temperature, and elapsed-time rows agree;
- self-collision number and energy tangent invariants close;
- electron-neutrino differentiated first law closes;
- CP and mu/tau symmetric limits close;
- no floor, clipping, projection, hidden state repair, or changed support mask
  was introduced.

### Remaining mathematical risks

- **P1:** the input `T_gamma` derivative is absent. It differentiates bath
  logits, scaled electron radial nodes and weights, kinematics, support
  boundaries, matrix elements, electromagnetic EOS, and Hubble terms. It is
  not a one-line extension of the spectral JVP.
- **P2:** the numerical Pauli path can underflow in extreme tails. D-079 is the
  derivative of the frozen binary64 path and has not established an
  arbitrary-precision tail theorem.
- **P2:** support-boundary tangents are piecewise-smooth. A `T_gamma` column
  must type or exclude exact support kinks rather than silently choose a side.

## 7. PHYS-MATH-CODE audit

### PASS

- equation-to-code mapping follows chart -> modal logit tangent -> event Pauli
  tangent -> signed four-leg assembly -> modal/native action -> static RHS;
- comparator blob identity is pinned before tests;
- the original comparator file is unchanged;
- small-grid tests and exact retained-state slow test both execute the original
  collision/RHS code as the centered witness;
- generated JSON and plots are deterministic and fail-closed;
- sign, normalization, flavour routing, and omitted-physics mutations are
  discriminated.

### Remaining software risks

- **P1:** no full-state dense Jacobian, `LinearOperator`, sparsity policy, or
  SciPy-BDF `jac=` adapter exists.
- **P1:** only the spectral input block is implemented. The elapsed-time input
  column is structurally zero, but the `T_gamma` input column is load-bearing
  and missing.
- **P1:** the static test matrix is not yet a prospectively sealed multi-regime
  set covering early equilibrium, transition, asymmetric flavour, retained
  stalled region, and late weak-collision states with multiple independent
  directions.
- **P2:** D-079 deliberately mirrors the event traversal while importing the
  frozen low-level helpers. Future comparator edits require blob-pin failure
  and explicit re-adjudication; silent drift is forbidden.
- **P2:** no cost model or same-budget performance measurement exists.

## 8. Plot-driven CRAG audit

### Correctness

The residual ladder displays a common minimum around `epsilon=1e-4`, followed
by a roundoff plateau; this matches a centered second-order witness. Invariant
residuals are at `0` or `1e-14`--`1e-16`. The first-law value is exactly zero in
the receipt; its logarithmic bar is only a plotting floor and must not be read
as a measured nonzero residual.

### Retrieval

The result is consistent with precision neutrino-decoupling literature in
which direct Jacobian computation is used to accelerate a stiff differential
system. That literature supports the method choice only; RABBIT's formulas and
coefficients are established by the frozen comparator and the present exact
and numerical checks.

### Augmentation

The conclusion survives one manufactured physical state and the exact retained
order-60 stalled-region state. It has not yet been augmented across the full
thermal history or the missing thermal input direction.

### Generation

The plots predict that a properly implemented full-state Jacobian should use a
local derivative contract and true residual certification. They do not predict
trajectory completion or wall-time margin.

### Figure honesty

The 6.8-by-4.2-inch audit figures are suitable as internal/double-column
artifacts. Shrinking them to a single 89--90 mm journal column would reduce
default text toward roughly 5 pt and make rotated categorical labels marginal.
They are not promoted as publication figures without a separate typography
pass.

## 9. Completion assessment

Percentages below are engineering estimates, not gate scores.

| Layer | Completion | Status |
|---|---:|---|
| D-077 method authority | 100% | merged |
| D-078 generic transformed-Jacobian research | 100% | merged |
| D-079 spectral `c` event-JVP and static RHS JVP | 100% | validated |
| spectral-block physical validation | 85% | two principal states; broader state/direction matrix still needed |
| full-state static Jacobian/JVP | 70--80% | spectral block done; thermal column dominates remaining work |
| BDF `jac=` instrument and telemetry | 0% | forbidden before full-state static certificate |
| stalled-prefix completion under frozen budget | 0% | untested |
| endpoint/holdout/F10 gate closure | 0% | gate remains FAIL |
| overall D-077 numerical-equivalence lane | about 55% | method and spectral derivative closed; solver and endpoint evidence absent |

A naive coordinate count would say that 180 spectral inputs are covered and
the elapsed-time input is structurally zero, leaving only one nonzero-relevant
state column, `T_gamma`. That would misleadingly imply near-completion. The
thermal column is much harder than one spectral column because it changes
bath distributions, quadrature geometry, support, kinematics, EOS, and Hubble
coupling. The difficulty-weighted estimate above is the honest one.

## 10. Next DAG node: D-080

The next admissible task is **full-state static Jacobian completion**, not a
trajectory.

D-080 must:

1. derive and implement the `T_gamma` input-direction tangent of the electron
   collision sector, electromagnetic EOS, Hubble rate, and photon-temperature
   row;
2. classify exact support-boundary crossings and fail closed there;
3. certify the elapsed-time input column as exactly zero;
4. expose a full-state JVP and, if justified, assemble a dense Jacobian for the
   182-state system;
5. freeze a multi-regime state/direction matrix: early equilibrium, asymmetric
   flavour, transition, retained `creep_1200`, and late weak-collision;
6. repeat conservation, first-law, CP, mu/tau, strict-domain, and mutation
   checks;
7. produce thermal-direction residual ladders and support-boundary plots.

`N`/`T_cm(N)` is the independent variable dependence of the nonautonomous RHS,
not a state column required by SciPy BDF's `jac(t,y)`. It should be tested as an
explicit-time derivative only for diagnostics or an augmented formulation; it
must not be confused with the missing `T_gamma` Jacobian column.

Only after D-080 passes may D-081 create a separately sealed BDF instrument
with the full Jacobian, Newton/order/step/rejection telemetry, true residual
checks, and the unchanged wall budget. F10 can move only after later
stalled-prefix, endpoint, holdout, and mutation evidence passes.

## 11. Final claim boundary

Allowed now:

> The frozen private comparator has an independently coded analytic spectral
> cloglog JVP that differentiates the actual event quadrature, closes static
> conservation and first-law tangents, kills designated mutations, and agrees
> with the unchanged original RHS both on a manufactured physical state and on
> the retained stalled-region state.

Not allowed now:

> The full Jacobian is complete; BDF is fixed; the stalled trajectory
> completes; the wall budget is met; the endpoint is validated; or
> `G-F10-INDEPENDENT-FLRW` passes.
