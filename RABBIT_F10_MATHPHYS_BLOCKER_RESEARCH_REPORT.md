# RABBIT F10 Mathematical-Physics Blocker Resolution Research Report

DATE: 2026-08-06
MODE: research-only, no source-repository edits, no gate movement
HARNESS: PHYS–MATH RESEARCH HARNESS 5.6

## Executive conclusion

The research loop does **not** produce a validated replacement solver. It does produce one serious direction whose mathematical form is eligible for D-071 validation, subject to a mandatory D-029 novelty comparison, and eliminates several attractive but insufficient shortcuts.

The primary survivor is **ECEM-3C**, a seven-state energy-constrained entropic moment formulation with reaction-aware tail control and goal-oriented output certification. Its distinctive feature is not any single known ingredient. It combines: exact total-energy structure, a realizable two-species three-moment manifold, explicit discrete-conservation accounting, a projected entropy-metric defect bound, and a three-channel domain/output certificate. This is materially different from pinning the finite-difference Jacobian or swapping the integrator. It is also operation-level distinct from D-029: D-029 reduced one off-grid collision leg with a target-dependent three-node selector, whereas ECEM-3C proposes a fixed global state manifold and retains the collision moment map as a separate operator. The original D-029 envelopes are nevertheless absent, so this distinction is credible but not yet a complete historical novelty proof.

Research-stage probes support feasibility but not promotion:

- the exact comoving-energy identity is derived and symbolically checked;
- a coordinate change cannot remove stiff eigenvalues, but can improve nonnormal scaling in the favorable heat-capacity regime;
- frozen endpoint spectra are moment-matched by a three-parameter realizable family with sub-`1e-3` energy-weighted L1 error, while 4–6 parameters rapidly worsen inverse-map conditioning;
- the equilibrium `y>30` energy tail is `4.92e-10`, but `k=8–10` tails reach `2.05e-6` to `2.24e-5`;
- an explicit nonnormal counterexample shows why the measured Hurwitz creep-state eigenvalues do not certify contraction;
- exact symmetry/time elimination can provide only about `1.5×` by state/Jacobian dimension arithmetic.

The formal decision is `REOPEN_VALIDATION`, not `PROMOTE`. The next step is a static retained-checkpoint package, followed—only if it passes—by a prospectively sealed discriminator capped at 5,500 full-cost collision calls. At the conservative `4.5 s` call ceiling, a 2.5× safety factor still fits inside the frozen 18 h budget.

---

## 1. Contract and baseline

# Phase 1 — Research Contract

## PRIMARY_RQ

Can the closed `G-F10-INDEPENDENT-FLRW` trajectory lane be reopened by a materially new mathematical-physics formulation which (i) preserves or explicitly bounds the relevant neutrino-decoupling physics, (ii) survives the measured creep regime near `N≈0.163`, and (iii) prospectively predicts completion within the frozen 64,800 s wall budget with margin?

## SUB_RQS

1. Can exact conservation laws eliminate the numerically fragile electromagnetic-temperature exchange equation without silently changing the discrete physics?
2. Is the solution trajectory confined close enough to a low-dimensional realizable entropic manifold to replace 180 spectral degrees of freedom by a few moment variables?
3. Can the omitted `y>30` domain and model-reduction residual be bounded in the actual collision and output channels, not only in equilibrium energy density?
4. Which norm or Lyapunov structure can certify stability despite local nonnormality, given that Hurwitz eigenvalues alone are insufficient?
5. What bounded, prospectively sealed discriminator would distinguish a real reopening route from another fast-looking local repair?

## IN_SCOPE

- Standard no-QKE independent-FLRW comparator and its exact symmetries.
- Mathematical reformulation, asymptotic/moment reduction, analytic domain control, conservation projection, output-oriented certification.
- Short static or toy numerical probes using already retained endpoint data and analytic counterexamples.
- A design for a future prospectively sealed stalled-phase discriminator.

## OUT_OF_SCOPE

- Harness repair; it is a separate lane by user instruction.
- Generic solver tuning, a larger wall budget, faster hardware, or direct Jacobian repair as the reopening argument.
- Full end-to-end production implementation in this cycle.
- QKE, public-production claims, and DSMC as a candidate route.
- Editing the source repository or changing gate status.

## Conventions

- Metric signature `(-,+,+,+)` where spacetime conventions enter.
- E-fold time `N=ln a`; `T_cm=T_start exp(-N)`.
- Natural units follow the frozen source, with energy densities in powers of MeV; dimensions are stated explicitly.
- Neutrinos are effectively massless over the decoupling interval, hence `P_nu=rho_nu/3`.
- `x` denotes the exactly degenerate mu–tau pair when the frozen no-QKE assumptions hold.

## Completion bar

This cycle is complete when it has: (a) an audited evidence map; (b) genuinely distinct candidate families; (c) at least one symbolic identity and several numerical/counterexample probes; (d) a survivor with explicit proof obligations; (e) a prospective discriminator whose pass/kill criteria enforce the 18 h budget with margin; and (f) an honest external-gate disposition. No gate movement is permitted.


---

## 2. Evidence and claim audit

# Phase 2–3 — Evidence Synthesis and Claim–Source Audit

## Baseline facts

1. **SUPPORTED:** The retained order-60, `y_max=30` domain holdout did not reach a scientific predicate. It consumed the 18 h budget and stalled near `N≈0.163`; D-071 therefore closes the instrument, not the physical proposition.
2. **SUPPORTED:** V3 localizes a severe numerical pathology: persistent finite-difference step factors corrupt selected Jacobian columns, Newton failures reset BDF progress, and the solver remains at low order. The true sampled operator is Hurwitz, but the report explicitly keeps the causal chain as inference because the controlled repair is disallowed.
3. **SUPPORTED:** A completing lower-resolution run exhibits the same factor ratchet. Therefore ratchet presence is not a sufficient discriminator; a proposed method must show that the reduced dynamics avoid the corruption-sensitive Newton loop.
4. **SUPPORTED:** Direct Jacobians or solver changes can be powerful in other neutrino-decoupling codes, but D-071 explicitly excludes a faster implementation of the same instrument from reopening.
5. **SUPPORTED:** Low-dimensional spectral and entropic descriptions of relic-neutrino spectra exist in the literature. They establish plausibility, not a project-specific error certificate.

## Claim audit

| Claim | Status | Audit judgment |
|---|---|---|
| C-001 Current failure disproves the underlying physics | UNSUPPORTED / REMOVED | No scientific predicate was evaluated. |
| C-002 Jacobian-factor pathology materially contributes to the stalled instrument | PARTIALLY_SUPPORTED | Strong measured association and exclusion of alternatives; controlled causality remains unavailable. |
| C-003 Pinning/capping the Jacobian alone reopens D-071 | CONTRADICTED | D-071 excludes it, and measured diagnostic variants remain far outside the required operating envelope. |
| C-004 A dynamically adapted low-dimensional spectrum is plausible | SUPPORTED | Birrell et al. and Bond et al.; endpoint probe agrees qualitatively. |
| C-005 Three parameters per species are sufficient for this full trajectory | INFERENCE_ONLY | Literature and one endpoint support a candidate; no creep-state or trajectory certificate exists. |
| C-006 `y>30` is negligible for every relevant quantity | CONTRADICTED AS BROAD CLAIM | Energy tail is tiny, but high-order moments grow to `~2.2e-5` at `k=10`; collision and feedback channels remain. |
| C-007 Hurwitz creep-state Jacobians certify contraction | CONTRADICTED | A simple nonnormal counterexample has the same negative spectral abscissa with large transient growth. |
| C-008 An exact energy coordinate removes physical stiffness | CONTRADICTED AS BROAD CLAIM | Eigenvalues are invariant under an exact coordinate change; it can only improve scaling/cancellation/nonnormality conditionally. |
| C-009 Exact mu–tau symmetry plus postprocessed time closes the runtime gap | CONTRADICTED | State/Jacobian-column arithmetic gives at most about 1.5× dense-FD reduction. |
| C-010 A composite energy-constrained entropic reduction with certified residuals is a legitimate D-071 candidate | HYPOTHESIS | It is structurally different from a solver repair, but no end-to-end certificate exists yet. |

## Conflict matrix

| Tension | Resolution in this loop |
|---|---|
| Three-parameter spectra are accurate vs. moment inversion becomes ill-conditioned | Both can hold. Representation error and inverse-map conditioning are different properties; use a Fisher/preconditioned coordinate and retain a conditioning gate. |
| Energy-coordinate reformulation is exact vs. the discrete collision action has a finite first-law residual | Separate a discrete-equivalent lane with an explicit residual from a prospectively defined method-internal conservative-correction lane. Never silently set the residual to zero, and forbid post-hoc projection. |
| Equilibrium tail is tiny vs. T13 is a domain blocker | T13 is not certified by the energy tail alone. Bound direct output, missing collision source, and propagated interior feedback separately. |
| Stable eigenvalues vs. BDF failure | Stability of eigenvalues does not control nonnormal transient growth or finite-difference Newton quality. Use an entropy/Lyapunov metric after projecting macro modes. |

## Missing evidence

- Entropic-manifold coverage at the three retained creep checkpoints and over the full retained base trajectory.
- Reaction-by-reaction polynomial/exponential envelope for domain-crossing collision terms.
- Entropy-metric logarithmic norm or spectral gap on a tube around the candidate manifold.
- A goal-oriented output bound for every fold-level F10 statistic.
- A prospectively sealed call-count/runtime discriminator on the stalled phase.
- The four hash-locked D-029 result/adjudication envelopes, needed for a final operation-graph novelty audit.
- A real creep-state collision invariance-defect measurement; it was not executed in this loop because a complete source snapshot was unavailable locally and the prospective execution contract had not been sealed.


---

## 3. Hypothesis space

# Phase 4 — Divergent Hypothesis Tree

## Root question

```
Reopen G-F10 without treating a faster copy of the failed BDF instrument as new evidence
├── A. Exact-structure reformulations
│   ├── H-001 Comoving total-energy coordinate and algebraic T_gamma
│   ├── H-002 Exact mu–tau symmetry + postprocessed cosmic time
│   └── H-003 Conservation projection of the discrete collision exchange
├── B. Low-dimensional kinetic representations
│   ├── H-004 Dynamic orthogonal-polynomial / fugacity basis
│   ├── H-005 Three-parameter Fermi entropic manifold per independent species
│   ├── H-006 First invariant-manifold correction L^{-1} Delta
│   └── H-007 Dynamical low rank
├── C. Domain and error certification
│   ├── H-008 Three-channel analytic tail certificate
│   ├── H-009 Entropy-metric defect/tube bound
│   ├── H-010 Goal-oriented adjoint output certificate
│   └── H-011 Certified empirical operator interpolation of collision moments
├── D. Shortcuts and controls
│   ├── H-012 Published momentum-averaged T/mu model as direct replacement
│   ├── H-013 Late-time free-streaming termination
│   ├── H-014 Direct Jacobian/AP preconditioner only
│   └── H-015 Generic hardware/parallel/solver change
└── E. Composite
    └── H-016 ECEM-3C = H-001 + H-003 + H-005 + H-008 + H-009 + H-010,
                       with H-006/H-011 conditional
```

## Candidate records

### H-001 — Exact comoving-energy coordinate
CORE_CLAIM: evolve `W=a^4(rho_nu+rho_EM)` and recover `T_gamma` algebraically.
MINIMAL_ASSUMPTIONS: massless neutrinos; monotone electromagnetic EOS; explicit treatment of discrete first-law residual.
DISTINCTIVE_PREDICTION: the exchange term `Q_EM/H` disappears from the total-energy equation, while the transformed continuous system has identical eigenvalues.
FATAL_VULNERABILITY: no improvement in the entropy/logarithmic norm at creep states, or an ill-conditioned algebraic inversion.
STATUS: active-supporting.

### H-002 — Exact symmetry/time elimination
CORE_CLAIM: merge mu and tau spectra and integrate cosmic time after the thermodynamic solve.
DISTINCTIVE_PREDICTION: order-60 state count falls from 182 to 121.
FATAL_VULNERABILITY: the maximum dimension-only dense-FD speedup is only about 1.5×.
STATUS: supporting only.

### H-003 — Conservation projection
CORE_CLAIM: minimally project the discrete collision exchange onto the exact energy-conserving subspace.
FATAL_VULNERABILITY: correction exceeds the frozen first-law tolerance or changes F10 observables beyond their certified slack.
STATUS: active-supporting.

### H-004 — Dynamic orthogonal basis
CORE_CLAIM: temperature/fugacity-adapted basis moves dominant thermodynamic motion into the lowest modes.
KNOWN_LIMIT: already established in literature and partly prototyped internally; not sufficient novelty by itself.
STATUS: merged into H-005/016.

### H-005 — Three-parameter Fermi entropic manifold
CORE_CLAIM: each independent species is represented by number, energy, and one bounded shape moment on a realizable Fermi manifold.
DISTINCTIVE_PREDICTION: exact matched moments with a low shape error and bounded exponential tail; substantially fewer dynamical variables.
FATAL_VULNERABILITY: trajectory leaves the manifold or the moment map becomes unusably ill-conditioned.
STATUS: active-primary component.

### H-006 — First invariant-manifold correction
CORE_CLAIM: approximate the micro component by solving a projected linear equation `L_theta g_1=-Delta_theta`.
FATAL_VULNERABILITY: no projected spectral gap or correction is as costly as the original trajectory.
STATUS: conditional.

### H-007 — Dynamical low rank
CORE_CLAIM: low-rank collision representation reduces nonlinear work.
FATAL_VULNERABILITY: the present problem has no large physical-space × velocity tensor; the main rank-separation mechanism is absent.
STATUS: rejected as primary.

### H-008 — Three-channel tail certificate
CORE_CLAIM: certify omitted momentum domain through direct output, missing collision source, and propagated feedback channels.
DISTINCTIVE_PREDICTION: a gamma-function envelope closes each reaction channel at `Y=30` or determines a larger analytic cutoff.
FATAL_VULNERABILITY: a reaction kernel or domain-crossing geometry violates the assumed exponential-polynomial envelope.
STATUS: active-primary component.

### H-009 — Entropy-metric defect/tube bound
CORE_CLAIM: after projecting macro modes, the collision dynamics contract in the Hessian metric of Fermi relative entropy, permitting a tube bound for the micro residual.
FATAL_VULNERABILITY: nonnormal growth remains positive in every usable metric or the tube constants are too pessimistic.
STATUS: active-primary certificate.

### H-010 — Goal-oriented adjoint certificate
CORE_CLAIM: bound only the F10 quantities of interest using an adjoint-weighted residual instead of demanding uniform full-state accuracy.
FATAL_VULNERABILITY: adjoint amplification or fold-level nonsmoothness makes the bound vacuous.
STATUS: active-primary certificate.

### H-011 — Certified collision hyper-reduction
CORE_CLAIM: approximate only the moment collision map offline and carry a rigorous online residual bound.
FATAL_VULNERABILITY: training coverage is not prospective or positivity/conservation is lost.
STATUS: conditional acceleration.

### H-012 — Published momentum-averaged model as direct replacement
CORE_CLAIM: use a fast effective-temperature/chemical-potential model directly.
FATAL_VULNERABILITY: reported sub-0.04% integrated-observable agreement does not establish the tighter, fold-specific F10 predicates.
STATUS: rejected shortcut; retain as baseline/control.

### H-013 — Late-time free-streaming termination
CORE_CLAIM: stop when the integrated collision-to-Hubble residual is bounded.
FATAL_VULNERABILITY: the measured stall is at `N≈0.163`, before late free streaming; this cannot solve the primary blocker.
STATUS: rejected as primary; possible downstream optimization.

### H-014 — Direct Jacobian/AP preconditioner only
CORE_CLAIM: cure stiffness by analytic Jacobian or AP splitting.
FATAL_VULNERABILITY: D-071 explicitly excludes a faster implementation of the same instrument, and existing prototypes make this non-novel.
STATUS: rejected as reopening argument; supporting implementation tool only.

### H-015 — Hardware/parallelism/general solver
STATUS: rejected by D-071 and by the step-count-dominated miss.

### H-016 — ECEM-3C composite
CORE_CLAIM: a seven-state energy-constrained entropic moment system, with conservative collision projection and three-channel/goal-oriented certification, can replace the failed 182-state trajectory as a materially new validation instrument.
DECISIVE_TEST: the prospectively sealed stalled-phase discriminator in Phase 7.
STATUS: survivor candidate.


---

## 4. Adversarial review

# Phase 5 — Adversarial Review

This review was performed as a separate skeptical pass after candidate generation. It is independent in role and criteria, not in operator identity; therefore it does not satisfy the Phase-8 external reviewer requirement.

| Hypothesis | Review role | Main objection | Disposition before validation |
|---|---|---|---|
| H-001 energy coordinate | TRADEOFF | Exact coordinate changes cannot delete physical stiff eigenvalues; benefit is scaling-dependent | retain only as part of composite |
| H-002 symmetry/time elimination | INFORMATIVE_FAILURE | Correct but at most ~1.5× from dense-FD dimension arithmetic | support only |
| H-003 conservation projection | NEEDS_MORE_EVIDENCE | Projection may alter the discrete comparator | retain with explicit correction bound |
| H-005 3-parameter entropic manifold | DEFENDED WITH LIMITS | One endpoint is not trajectory coverage; inverse map conditioning rises rapidly | serious candidate |
| H-006 invariant-manifold correction | NEEDS_MORE_EVIDENCE | Requires an invertible projected linearized collision operator | conditional |
| H-007 dynamical low rank | REJECTED_SHORTCUT | Low-rank tensor structure is largely absent | reject primary |
| H-008 tail certificate | DEFENDED WITH LIMITS | Energy tail alone is misleading; collision and feedback must be included | serious candidate |
| H-009 entropy-metric tube | DEFENDED CONCEPTUALLY | Proving a usable gap over a tube may be the hardest task | serious certificate |
| H-010 adjoint output certificate | TRADEOFF | Bound may blow up near nonsmooth fold statistics | serious if fold objectives are differentiable or bounded piecewise |
| H-011 hyper-reduction | TRADEOFF | Can overfit the retained trajectory and violate conservation | only after the analytic structure is fixed |
| H-012 published averaged model | REJECTED_SHORTCUT | External accuracy statement is not the gate-specific certificate | baseline only |
| H-013 late freeze-out stop | REJECTED_SHORTCUT | Does not touch the early creep epoch | reject primary |
| H-014 direct Jacobian/AP only | REJECTED_SHORTCUT | Solver repair; D-071-ineligible | support only |
| H-016 ECEM-3C | DEFENDED, NOT PROMOTED | Only candidate addressing step count, domain, realizability, and output certification together | send to Phase 6–7 |

## Red-team conclusions

1. The elegant part—exact energy conservation—is not the decisive speedup by itself.
2. The low-dimensional part is only credible if the residual orthogonal to the manifold is measured at creep states, not merely at the endpoint.
3. The certification part must control nonnormal amplification; eigenvalue plots are insufficient.
4. The tail proof must be reaction-aware. A scalar Fermi energy-tail fraction cannot license T13.
5. The composite is not allowed to hide its cost in an offline phase trained on post-hoc outputs. The basis, moments, cutoff, metrics and thresholds must be sealed before discriminator output.


---

## 4A. D-029 novelty boundary

The retained historical audit reconstructs D-029 as a **three-node maximum-relative-entropy local deposition**. For a fixed support triple its positive maximum-entropy weights and two moment constraints were sound. The route failed because the executable target-dependent support selector, local prior, exact-node one-hot branch, continuity, Jacobian and permutation-covariance obligations were never unified into one global semidiscrete theorem. D-029 therefore remains closed; a globally smooth entropy-local basis was explicitly left open as a different design class.

ECEM-3C differs at the operation level:

| Axis | D-029 | ECEM-3C |
|---|---|---|
| Reduced object | one off-grid collision leg | the global species distribution along the trajectory |
| Support | target-dependent three-node subset | fixed analytic functions on `y\in[0,\infty)` |
| Branching | selector ties and exact-node one-hot rescue | no support selector; one smooth inverse moment map is required |
| Evolved object | no completed global collision/trajectory system | six moments plus total comoving energy |
| Entropy role | local positive deposition weights | realizability plus a global Hessian metric for residual control |
| Certificate | fixed-triple theorem | manifold defect + entropy tube + three-channel tail + output adjoint |

This distinction is a **research finding**, not a final novelty verdict. The raw D-029 derivation, binary64 audit, design adjudication and reopen audit are not present here. Gate A0 below therefore requires those exact envelopes when available and kills ECEM-3C if it reintroduces an adaptive support/active-set rule equivalent to D-029. The full comparison is retained in `research_artifacts/reports/04A_D029_NOVELTY_BOUNDARY.md`.

---

## 5. Physics and mathematics validation

# Phase 6 — Physics and Mathematics Validation

## 1. Exact comoving-energy identity

Let `N=ln a` and define

\[
W(N)=e^{4N}\left[\rho_\nu(N)+\rho_{\rm EM}(T_\gamma(N))\right].
\]

Total energy conservation in a flat FLRW background gives

\[
\frac{d\rho_{\rm tot}}{dN}=-3(\rho_{\rm tot}+P_{\rm tot}).
\]

With effectively massless neutrinos, `P_nu=rho_nu/3`, hence

\[
\boxed{\frac{dW}{dN}=e^{4N}\left(\rho_{\rm EM}-3P_{\rm EM}\right)}.
\]

The neutrino–electromagnetic exchange cancels exactly. The source is the electromagnetic trace, dominated by the electron mass; the photon trace vanishes.

Recover `T_gamma` from

\[
\rho_{\rm EM}(T_\gamma)=e^{-4N}W-\rho_\nu[f,T_{\rm cm}],
\]

which is locally unique wherever `d rho_EM/dT_gamma>0`. Implicit differentiation gives

\[
\frac{\partial T_\gamma}{\partial W}=
\frac{e^{-4N}}{\rho'_{\rm EM}(T_\gamma)},\qquad
\frac{\partial T_\gamma}{\partial f_j}=
-\frac{\partial\rho_\nu/\partial f_j}{\rho'_{\rm EM}(T_\gamma)}.
\]

Dimensions: `[W]=MeV^4`, `[partial T/partial W]=MeV^{-3}`, and `partial T/partial f_j` has units MeV. The sign is physical: at fixed total comoving energy, increasing neutrino occupancy lowers the plasma temperature.

**Validation status:** PASS for the continuous equations; CONCERN for discrete equivalence. The existing collision discretization carries a measured finite first-law residual. A new instrument must either retain that residual explicitly,

\[
W'=e^{4N}(\rho_{\rm EM}-3P_{\rm EM})+e^{4N}R_E,
\]

or project the collision exchange onto the exact conservation subspace and certify the induced correction. Setting `R_E=0` without that comparison is forbidden.

## 2. Energy coordinate does not erase stiffness

The toy exchange model

\[
x'=-K(x-aT),\qquad T'=\frac K C(x-aT)-\epsilon T,
\qquad W=x+CT
\]

was transformed exactly. The eigenvalues agree to floating-point accuracy for every tested heat capacity. However, the Euclidean logarithmic norm changes strongly with scaling: for `C=1` the energy coordinate worsens it, while for `C=1000` it changes a large positive value to about `-0.997`.

**Conclusion:** the coordinate can remove exchange cancellation and improve nonnormal scaling, but the benefit is conditional and must be measured on the retained creep states. It is not a proof of speedup.

## 3. Realizable entropic manifold

For each independent species `a∈{e,x}`, define the three-moment state

\[
m_{a,j}=\int_0^\infty y^2\psi_j(y)f_a(y)\,dy,
\quad \psi=(1,y,\phi),\quad \phi(y)=\frac{y}{1+y}.
\]

Use the Fermi exponential family in natural parameters

\[
f_a(y;\lambda_a)=\frac{1}{1+\exp[-u_a(y)]},\qquad
u_a(y)=\alpha_a+\lambda_{1,a} y+\eta_a\phi(y),\quad \lambda_{1,a}<0.
\]

Equivalently, `\beta_a=-\lambda_{1,a}>0`. This guarantees `0<f<1` and an exponential tail. In the natural coordinates `\lambda=(\alpha,\lambda_1,\eta)`, the moment Jacobian is the symmetric positive Gram/covariance matrix

\[
\frac{\partial m_i}{\partial\lambda_j}
=\int_0^\infty y^2\psi_i(y)\psi_j(y)f(1-f)\,dy,
\]

provided the sufficient statistics are linearly independent in the weighted space. In the implementation coordinates `(alpha,beta,eta)` the `beta` column has the opposite sign; invertibility is unchanged, but symmetry/positive-definiteness should not be claimed in those coordinates.

A frozen 48-node endpoint probe exactly matched moments `k=2,3,4` with three parameters. The energy-weighted L1 errors were about `7.63e-4` for `nu_e` and `2.53e-4` for `nu_x`; the next `k=5` relative errors were `2.62e-4` and `8.52e-5`. Adding parameters lowered some fit errors but drove the unscaled inverse-moment Jacobian condition from roughly `1.35e3` at 3 parameters to `4.1e4`, `1.2e6`, and `3.3e7` at 4–6 parameters.

**Conclusion:** three parameters form a plausible accuracy/conditioning knee, not a theorem. The condition number is coordinate-dependent, so a Fisher/Cholesky-preconditioned inversion should be tested. No trajectory coverage is claimed.

## 4. Tail bounds

For the equilibrium Fermi tail,

\[
I_k(Y)=\int_Y^\infty\frac{y^k}{e^y+1}\,dy
\le \int_Y^\infty y^k e^{-y}\,dy
=\Gamma(k+1,Y).
\]

The two-term alternating expansion also gives

\[
\Gamma(k+1,Y)-2^{-(k+1)}\Gamma(k+1,2Y)\le I_k(Y).
\]

At `Y=30`, the relative tail is `4.92e-10` for the energy moment `k=3`, but rises to `2.05e-6`, `7.13e-6`, and `2.24e-5` for `k=8,9,10`. Thus high-order collision weights can make the missing tail materially larger than the thermodynamic energy tail.

For any certified envelope `f(y)≤exp(A-b y)` with `b>0`,

\[
\int_Y^\infty y^k f(y)\,dy
\le e^A b^{-(k+1)}\Gamma(k+1,bY).
\]

**Validation status:** PASS for the scalar integral bound; CONCERN for the full T13 claim. Reaction kinematics, domain-crossing events and feedback into the interior require separate constants.

## 5. Hurwitz is not contraction

The family

\[
A_M=\begin{pmatrix}-5.75&M\\0&-5.75\end{pmatrix}
\]

has spectral abscissa `-5.75` for every `M`, yet the measured maximum `||exp(A_M t)||_2` on `0≤t≤3` grows to `6.42` at `M=100` and `19.2` at `M=300`. The Euclidean logarithmic norm is positive. A Lyapunov metric exists, but becomes poorly conditioned as `M` grows.

**Conclusion:** V3's negative eigenvalues exclude a physical linear instability but do not certify a small propagation factor. The candidate requires an entropy-weighted metric after macro tangent modes are projected out.

## 6. Exact symmetry arithmetic

The order-60 state has `3×60+2=182` variables. Exact mu–tau reduction and postprocessing of cosmic time give `2×60+1=121`. The corresponding dense finite-difference column ratio is approximately `122/183`, so dimension alone offers at most about `1.5×`. This is useful but cannot close the multi-thousand-fold evaluation-count miss.

## 7. Composite mathematical structure

Write `f=f_theta+g`, where `f_theta` is the six-parameter two-species entropic manifold and `g` satisfies the three moment orthogonality conditions per species. Let `P_theta` be the tangent/moment projector and

\[
\Delta_\theta=(I-P_\theta)Q[f_\theta]
\]

be the invariance defect. In the Fermi relative-entropy Hessian metric

\[
\|g\|_{H_\theta}^2=\sum_a\int
\frac{y^2|g_a|^2}{f_{\theta,a}(1-f_{\theta,a})}\,dy,
\]

a usable tube theorem would have the schematic form

\[
\frac{d}{dN}\|g\|_H
\le-\gamma(N)\|g\|_H+\|\Delta_\theta\|_H+L\|g\|_H^2.
\]

For a quantity of interest `J`, an adjoint `z` yields

\[
|\delta J|\lesssim |z(N_0)\delta x_0|
+\int\|z\|_*\bigl(\|r_{\rm model}\|+\|r_{\rm tail}\|+\|r_{\rm quad}\|\bigr)dN
+R_{\rm nl}.
\]

These are proof obligations, not results. The composite remains NOT_YET_TESTABLE until the metric, tube, reaction constants and fold-level output functional are instantiated.


---

## 6. Decisive verification design

# Phase 7 — Minimal Decisive Verification Design

## Candidate under test

**ECEM-3C:** Energy-Constrained Entropic-Manifold Reduction with Three-Channel Certification.

The dynamical state is seven-dimensional: three moments for `nu_e`, three for the exactly degenerate `nu_x=(nu_mu,nu_tau)`, and total comoving energy `W`. `T_gamma` and the entropic parameters are algebraic; cosmic time is postprocessed. The collision moment map is evaluated exactly first and may be hyper-reduced only after an error certificate exists.

## Gate A0 — Historical novelty and operation-graph exclusion

1. **Claim:** ECEM-3C is not a continuation or relabelling of the closed D-029 selector-based three-node deposition route.
2. **Inputs:** the retained D-029 audit summary immediately; the four hash-locked D-029 raw envelopes when available; the frozen ECEM-3C operation graph.
3. **Pass:** fixed global basis, no target-dependent support/active-set selector, no exact-node rescue branch, a globally regular inverse moment map, and a distinct state-level reconstruction–collision–moment composition.
4. **Kill:** any adaptive support rule equivalent to D-029, only a fixed-state local theorem, or inability to establish uniform continuity/Jacobian/permutation covariance.
5. **Current status:** partial pass at the summary level; raw-envelope comparison remains missing and blocks a final novelty or external promotion claim.

## Gate A — Static exactness and conditioning

1. **Claim:** the energy-coordinate system is algebraically equivalent to the frozen discrete system when the measured discrete first-law residual is retained.
2. **Inputs:** the three retained V3 creep checkpoints and their pinned-step Jacobians.
3. **Pass:** transformed and original Jacobian eigenvalues agree to relative `1e-10`; algebraic `T_gamma` inversion is unique; the conservation residual is reproduced to the frozen tolerance.
4. **Kill:** loss of uniqueness, a hidden residual term, or a correction larger than the available F10 error budget.
5. **Ambiguity:** coordinate helps only in a weighted norm; proceed to Gate C without claiming speed.

## Gate B — Manifold coverage

1. **Claim:** the two-species three-moment manifold covers the retained base and creep trajectories.
2. **Inputs:** every retained full state from D-069/V3, with endpoint states held out from basis selection.
3. **Measurements:** energy/number exactness, shape-moment residual, energy-weighted L1, collision-moment error, Fisher-metric condition, invariance defect.
4. **Pass:** no post-hoc basis change; every held-out state is realizable; the eventual goal-oriented bound, not an arbitrary state norm, fits inside one quarter of the available fold-level gate slack.
5. **Kill:** nonrealizability, unbounded moment inversion, or any held-out creep state whose certified output contribution exceeds half the gate slack.
6. **Ambiguity:** isolated local failure triggers one predeclared four-parameter comparator, not adaptive basis hunting.

## Gate C — Entropy-metric stability

1. **Claim:** the projected micro dynamics are dissipative on a tube around the manifold.
2. **Inputs:** projected pinned-step Jacobians at the three creep states plus early/late retained checkpoints.
3. **Pass:** a prospectively specified metric has `mu_H≤-gamma_*<0` after macro tangent modes are removed, with a tube/conditioning factor that keeps the integrated defect bound finite and below the Gate-B allowance.
4. **Kill:** only eigenvalue stability is available, the metric condition makes the bound vacuous, or `mu_H` is positive in the stalled window.

## Gate D — Three-channel domain certificate

For each reaction and each F10 quantity, decompose the `y>30` error into:

1. direct omitted output;
2. missing collision source, including domain-crossing incoming/outgoing events;
3. propagated feedback into the retained domain.

**Pass:** the sum of all three certified contributions is below one quarter of the fold-level gate slack. **Kill:** a reaction lacks a valid exponential-polynomial envelope or the feedback bound dominates the slack.

## Gate E — Prospectively sealed stalled-phase runtime discriminator

This is the D-071 reopening discriminator. Its contract must be committed before any candidate output.

- Windows: initialization, `N∈[0.14,0.22]` including the measured creep interval, a collision-decay window, and a late weak-collision window.
- Record: accepted/rejected steps, nonlinear iterations, RHS calls, collision-map wall, algebraic inversion failures, metric/defect indicators.
- Conservative per-call ceiling: `4.5 s`, slightly above the retained T13 post-drop value `4.42062 s/eval`.
- Full-trajectory call-count upper bound: `5,500` calls.
- Budget arithmetic: `5,500×4.5 s=24,750 s=6.875 h`; a `2.5×` safety factor gives `61,875 s=17.19 h`, still below `64,800 s`.

**PASS only if all hold:**

1. the upper, not central, call-count projection is at most 5,500;
2. the 95th-percentile collision-map call is at most 4.5 s or the product bound is tightened correspondingly;
3. no order-1/failure-reset limit cycle appears in `0.14≤N≤0.22`;
4. all Gates A–D remain within their predeclared budgets;
5. the projected total including certification overhead remains below 64,800 s.

**KILL if any hold:** projected calls exceed 5,500; the creep window shows nonprogress; a certificate is vacuous; or thresholds are changed after output.

## Gate F — Fold-level scientific discriminator

Only after A–E pass, run the sealed five-fold scientific contract. Propagate certified per-trajectory error into RMSE/MAE and the `>=20%` coupling improvement predicate. The lower confidence/certification bound—not the point estimate—must clear each gate. Deterministic rerun and no-test-fitting requirements remain unchanged.

## Cheapest next action

No new full trajectory. First compute Gates A–D from retained checkpoints and frozen endpoint/base states. Only a clean static package justifies the bounded Gate-E implementation.


---

## 7. External decision status

# Phase 8 — Provisional External Decision Gate

## Decision

`REOPEN_VALIDATION` for H-016 / ECEM-3C.

This is not `PROMOTE`. The candidate has a coherent mathematical mechanism, literature precedents for its components, and research-stage probes that reject several shortcuts. However, the decisive evidence—creep-state manifold coverage, entropy-metric tube constants, three-channel collision-tail bounds and the prospectively sealed runtime discriminator—has not been produced.

## Dimension-by-dimension review

- Evidence: sufficient to motivate, insufficient to validate.
- Physical validity: exact continuous conservation law passes; discrete conservation treatment remains open.
- Mathematical validity: realizability and scalar tail bounds pass; tube and output theorems remain obligations.
- Novelty: the individual ingredients are not new. The operation graph is distinct from the retained D-029 summary, but the original D-029 envelopes are absent; the project-specific synthesis may qualify only after Gate A0. No broad novelty claim is made.
- Testability: high; Phase 7 provides explicit pass/kill criteria.
- Robustness: unknown outside the frozen endpoint and toy probes.
- Tractability: plausible but unmeasured; the 5,500-call bound is a future gate, not a prediction already earned.
- Assumption burden: moderate-to-high, concentrated in entropy contraction and reaction-tail envelopes.

## Externality limitation

The harness calls for a reviewer who did not generate or validate the candidates. This cycle used a separate skeptical pass and independent symbolic/numerical tools, but not a distinct human or organizational authority. Therefore the formal external gate is unresolved and must be repeated before any D-071 reopen claim.


---

## 8. Survivor formalization

# Phase 9 — Survivor Formalization: ECEM-3C

STATUS: research candidate; `REOPEN_VALIDATION`, not promoted.

## Definition 1 — Macro state

Under exact mu–tau degeneracy, define

\[
X=(m_{e,0},m_{e,1},m_{e,2},m_{x,0},m_{x,1},m_{x,2},W)\in\mathbb R^7.
\]

`m_0` is the number moment, `m_1` the energy moment and `m_2` a bounded shape moment. `W` is total comoving energy.

## Definition 2 — Entropic reconstruction

For `a=e,x`, reconstruct

\[
f_a^\lambda(y)=\sigma\!\left(\alpha_a+\lambda_{1,a} y+\eta_a\frac{y}{1+y}\right),
\quad \sigma(u)=\frac1{1+e^{-u}},\quad \lambda_{1,a}<0,
\]

by matching the three moments. The equivalent tail-slope parameter is `beta_a=-lambda_{1,a}>0`. Use a predeclared Fisher/Cholesky scaling of the inverse moment problem. Four or more parameters are comparators, not adaptive rescue paths.

## Definition 3 — Macro dynamics

\[
\frac{dm_{a,j}}{dN}=\int_0^\infty y^2\psi_j(y)\frac{Q_a[f^\theta,T_\gamma]}{H}\,dy,
\]

and

\[
W'=e^{4N}(\rho_{\rm EM}-3P_{\rm EM})+e^{4N}R_E.
\]

`R_E` is either the explicitly retained discrete residual or zero under a prospectively defined, method-internal conservative correction with its own bound; post-hoc projection is excluded. `T_gamma` is recovered from the EOS constraint.

## Definition 4 — Micro defect and certificate

\[
\Delta_\theta=(I-P_\theta)Q[f^\theta],
\]

with the micro norm induced by the Fermi entropy Hessian. The model is admissible only when a projected dissipativity/tube estimate and a goal-oriented output error bound are available.

## Definition 5 — Three-channel tail accounting

Every omitted-domain bound is the sum of direct, collision-source and propagated-feedback terms. A thermodynamic tail fraction alone is never a domain certificate.

## Conjecture ECEM-3C-1 — Stalled-phase regularization

On the frozen no-QKE Standard-Model trajectory, the seven-state macro system avoids the finite-difference-Jacobian ratchet-sensitive Newton limit cycle and has a prospectively bounded full-trajectory call count below 5,500 at the frozen collision-map cost ceiling.

STATUS: HYPOTHESIS.

## Conjecture ECEM-3C-2 — Output certification

The sum of manifold, tail, quadrature and time-integration error propagated by the fold-level adjoint remains below the available G-F10 gate slack, so the certified lower bound of the coupling improvement can be evaluated.

STATUS: HYPOTHESIS.

## Proposition-level results already established in this cycle

1. Continuous comoving-energy identity and implicit derivatives: DERIVED / symbolically checked.
2. Fermi-tail incomplete-gamma bounds: DERIVED / numerically cross-checked.
3. Exact coordinate transformation preserves eigenvalues: DERIVED / numerically checked.
4. Hurwitz eigenvalues do not imply contraction: explicit counterexample / checked.
5. Exact symmetry reduction alone is insufficient: arithmetic bound / checked.
6. Three-parameter endpoint realizability and moment matching: exploratory numerical result / not trajectory validation.

## Paper/research memo structure

1. Measured failure geometry and D-071 contract.
2. Exact energy-constrained formulation.
3. Entropic moment manifold and realizability.
4. Projected entropy stability and invariant-manifold defect.
5. Reaction-aware tail and output certificates.
6. Prospectively sealed creep discriminator.
7. Five-fold independent scientific validation.
8. Limitations and failure cases.


---

## 9. Negative results and limitations

The exploratory endpoint fit is one frozen state, not a trajectory validation. The toy exchange model demonstrates possibility and failure modes, not the real creep-state conditioning. Scalar incomplete-gamma bounds do not include the full collision geometry. No full candidate trajectory, fold-level CV, or independent external decision was run. A source-level single-state collision invariance-defect probe was considered but not executed: the complete source snapshot was not available in the working container, the network clone failed, and no prospectively sealed collision-execution contract was in force. The raw V3 checkpoints needed for the next static package are retained in the source project's local evidence tree and were not reproduced here.

No novelty claim is made for entropic moments, AP micro–macro methods, reduced bases, or goal-oriented error estimation individually. The candidate novelty question concerns only the project-specific synthesis. It must be audited against the original D-029 envelopes and the later B3-v2/W7 operation graph before any publication or D-071 promotion claim.

## 10. Reproduction

```bash
python3 research_artifacts/experiments/run_mathphysics_probes.py
python3 research_artifacts/experiments/make_figures.py
python3 -m pytest -q research_artifacts/experiments/test_probe_invariants.py
make validate
```

Machine-readable results and source scripts are under `research_artifacts/results/` and `research_artifacts/experiments/`. The evidence and decision state are under `state/`.
