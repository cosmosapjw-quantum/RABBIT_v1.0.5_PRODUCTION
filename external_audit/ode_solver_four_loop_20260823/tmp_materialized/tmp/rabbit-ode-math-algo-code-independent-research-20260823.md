# Independent ODE remedy research record

Date: 2026-08-23 Asia/Seoul
Repository: `/home/cosmosapjw/Dropbox/rabbit/RABBIT_v1.0.5_PRODUCTION`
Source state: `diagnosis_report@78f5b091bd8de99bd8d0b1eddd8a0b7978f5da2b`
Question: Starting only from the original ODE audit register and current source, which mathematical, numerical-algorithmic, and coding-contract remedies can eliminate or materially reduce every solver issue/blocker, and what evidence would falsify each remedy?

## Independence boundary (frozen before evidence collection)

- Allowed inputs: current source and tests; the original audit result `A-ODE-ADJ3.json`; mandatory anti-drift policy; installed dependency manifests; fresh primary literature and official version-matched documentation; results produced by this new run.
- Excluded inputs: every prior physics-specific mitigation result, prior mitigation harness run, prior mitigation scratch note, and sibling results before adjudication.
- The procedure is independently re-run and blind across three axes. The root model has seen earlier conversation, so this is procedural independence, not a claim of impossible cognitive erasure.

## Competing hypotheses and falsifiers (frozen)

H1. A material subset of failures are acceptance-contract failures rather than integrator formula failures; typed terminal outcomes, solver-success admission, expected-event proof, residual/bracket checks, finite/raw-state checks, and invariant postconditions can fail-close them without changing the physical RHS.
Falsifier: a source-bound trace shows all such postconditions already enforced at every consumer, or the proposed contract cannot distinguish an observed false success from a valid endpoint.

H2. The endpoint/runtime gap cannot be closed by tolerance or generic stepper tuning alone; at least one asymptotic reduction in dominant RHS/Jacobian/operator work or state dimension is required.
Falsifier: a same-physics endpoint work model and reproducible current-path benchmark show a version-supported solver-policy change alone meets the hard wall with raw-state and accuracy gates preserved.

H3. Structure-exploiting stiff integration (analytic/block/sparse/matrix-free derivatives, inexact Newton-Krylov, Rosenbrock/exponential/IMEX/multirate methods, or justified switching) can reduce dominant work, but only when its stability domain, order/error estimator, preconditioner cost, and source-call count are derived for the actual operator structure.
Falsifier: the current Jacobian/operator lacks the assumed structure, setup/application cost dominates, nonlinear convergence is not preserved, or the resulting work bound cannot plausibly move the measured endpoint blocker.

H4. Goal-oriented and residual-based error control can allocate numerical work to endpoint observables more effectively than uniform componentwise tolerances while retaining a defensible global error budget.
Falsifier: adjoint/dual weights are unstable or too costly, neglected state errors couple strongly into the target observables, or an independent refinement ladder violates the promised bound.

H5. Dimension/order reduction (adaptive momentum representation, low-rank/separable operator compression, slow-manifold/DAE reduction, conservative remap, or active-set evolution) is potentially the only orders-of-magnitude route, but is admissible only with conservation, positivity, limiting-case, and a-posteriori error certificates against the number-of-record path.
Falsifier: rank/grid requirements grow to the full problem in the activation window, conservation or detailed balance cannot be preserved, remap error dominates, or certificate cost erases the gain.

H6. Some original findings are not repairable by a new algorithm: they require retained fail-closed claim ceilings, deletion/retirement of misleading backends, or new endpoint evidence. These must be classified explicitly rather than presented as solved.
Falsifier: a concrete current-path implementation and executed evidence directly discharges the original finding under its governing gate.

## Methods (frozen)

M1. Mechanically enumerate every finding in the original 34-item audit and reconstruct current source reachability without importing prior remedy classifications.
M2. For each open or inconclusive item, record failure mechanism, mathematical condition, proposed remedy, prerequisites, stability/consistency/complexity argument, coding insertion point, minimal falsification test, and residual claim ceiling.
M3. Run three blind reviews: mathematical analysis; numerical algorithm and work model; code/API/contracts/testing. Each may read only its assignment slice and may not read sibling or excluded prior results.
M4. Verify local package versions first, then use only primary papers and official documentation for technical claims. Record access date and distinguish version-matched facts from general theory.
M5. A fresh adjudicator reads only the three new blind results plus the original audit/current source and returns a non-inconclusive disposition for every original finding, deduplicated by claim and evidence.
M6. No production code, tests, gates, capability registries, or shared scientific documents are modified; this work produces analysis only.

## Acceptance and stopping rules (frozen)

- Every original finding has exactly one current disposition: open defect, retained blocker, inconclusive requiring a named experiment, or closed item; decomposed roots may be added but none may disappear.
- Every non-closed item has at least one concrete remedy or an explicit mathematical/engineering no-go explanation, with a falsifier and minimal evidence plan.
- The final portfolio contains at most three first implementation candidates, ranked by expected hard-blocker movement per line and per wall-time, with no segment-only result described as endpoint progress.
- External claims have primary/official sources and version boundaries; unexecuted proposals remain PROPOSED or SPECIFIED.
- The fresh adjudicator must produce pass/fail, not inconclusive. If evidence is insufficient, the corresponding blocker remains fail-closed.
- Stop after one blind wave, one adjudication/review round, and at most one closeout correction. New examples outside these criteria are deferred.

## Evidence log

This section is append-only after the freeze marker below.

FROZEN: hypotheses, falsifiers, methods, and stopping rules above are immutable for this work unit.

### 2026-08-23 source and environment observations

- The original adjudicated register contains 34 findings, numbered `F-ODE-ADJ3-001` through `F-ODE-ADJ3-034`: 24 `fail`, 7 `inconclusive`, and 3 `pass`. Its SHA-256 is `18195e473776b0fd0bcc23b1be41c45338d529bffaf75262cb2f99e9b6a174c9`.
- The live Python environment is NumPy 2.4.2 and SciPy 1.17.0; JAX and Diffrax are unavailable. The Python project specifies lower bounds (`scipy>=1.10`, `jax>=0.4.20`, `diffrax>=0.6`) rather than an exact runtime lock. The Rust lock pins diffsol 0.16.0. Consequently, no exact-environment JAX/Diffrax or release-endpoint validation can be claimed from this session.
- The D-071 trajectory state has `3*order+2` components; at order 60 this is 182. The last coordinate is passive elapsed time with derivative `1/H`, while the BDF call supplies neither `jac` nor `jac_sparsity` and requests dense output. The physical-prefix finite-difference rule scales its perturbation with the unweighted norm of this augmented state. This makes the recorded approximately 4.3e18 elapsed-time coordinate numerically decisive even though it is block-triangular/passive for the collision dynamics.
- A scale-admissible derivative calculation must use a dimensionless state `x=S^-1 y` and weighted norm, or remove the passive time coordinate from the Krylov state and recover it by quadrature. A direct event-stream directional derivative is preferable to a domain-leaving finite difference. If finite differences remain, the perturbation must be representable, domain-limited, and scale-aware; central differences suggest an `eps^(1/3)` scale and one-sided forward differences an `eps^(1/2)` scale, subject to a step ladder rather than a single magic constant.

### 2026-08-23 work-model bounds

- Retained D-071 evidence gives 32,704,637 projected further RHS evaluations at 4.42062 seconds/evaluation and a frozen 64,800-second wall. Under the deliberately over-generous counterfactual that every block of 182 calls were solely a removable dense finite-difference Jacobian, the residual is 179,695.81 calls and 794,366.88 seconds: 9.19 days, or 12.259 times the wall. This is not a universal solver lower bound; it is a no-go for claiming that dense finite-difference elimination alone, with no further step/model reduction, reopens D-071. Relative to the 5,500-call full-prefix cap, the same counterfactual is still 32.672 times too large.
- The recorded 1,477,632 whole-reaction rejections per state multiplied by the 32,704,637 projected calls is approximately 4.833e13 rejection checks. This multiplication is a workload indicator only; it is not a validated physical defect and must not be used as one.
- At the current N48 production-shaped rule (self angular order 12; electron radial 6, angular 4), source loop bounds imply up to 331,776 topology slots per self-collision channel, 1,327,104 across four such channels, and 663,552 electron-event slots: 1,990,656 event/channel visits per collision action before derivative work. The full analytic Jacobian then nests up to six output deposits by six input responses per retained event. A matrix-free event-stream `Jv` can contract each event with one supplied direction without materializing those output-by-input entries; this is a structurally plausible constant/asymptotic reduction, not yet an executed benchmark.
- The 3.5% Nq=16-to-32 Yp drift against 0.1% is a factor-35 gap. If, and only if, an asymptotic algebraic error law held, a crude extrapolation from N=32 gives N about 189 for order 2, 78 for order 4, or 58 for order 6. Because neither order nor asymptotic regime is established, these are hypothesis-sizing numbers, not grid requirements. A mixed absolute/relative convergence criterion and independent grid-family/domain holdouts are required.

### 2026-08-23 independent candidate architecture before blind results

1. `PROPOSED` — conservative slow-manifold/equilibrium-nullspace reduction: split the stiff collision operator into fast relaxing and slow/residual parts, project exact conserved/null modes, and switch back to the full operator when a residual or spectral-gap condition fails. It is falsified by absence of a usable spectral gap, loss of positivity/conservation/detailed balance, or failure of an independently refined endpoint-QoI error bound.
2. `SPECIFIED`, not implemented — scale-admissible direct `Jv` plus the already named EC-EXPRB-K prefix: event-level directional derivatives, algebraic photon-temperature recovery, and Krylov happy-breakdown handling must meet the existing N>=0.25, <=5,500 full-RHS-equivalent calls, and <=64,800-second caps. A static state, segment timing, or common-method parity does not count.
3. `PROPOSED` — certified adaptive operator/state compression: low-rank or separable collision kernels and adaptive momentum representation, with exact conservative corrections and residual/adjoint error enclosure for Yp/Neff. It is falsified if required rank approaches the full representation in the activation window, invariants/strict occupations fail, or certification cost erases endpoint savings.

All three candidates remain subordinate to fail-closed terminal-result contracts. None repairs false success, event-refinement publication, raw-state clipping, missing counters, or partial-state consumption without explicit code/API changes.

### 2026-08-23 external source register (accessed 2026-08-23)

- SciPy 1.17.0 `solve_ivp` official manual: https://docs.scipy.org/doc/scipy-1.17.0/reference/generated/scipy.integrate.solve_ivp.html . Version-matched to the live SciPy import. It documents event sign-change limitations, terminal/direction attributes, componentwise tolerance scaling, callable/sparse Jacobians and `jac_sparsity`, and the distinct `status`, `message`, `success`, `nfev`, `njev`, and `nlu` outputs.
- Diffrax event documentation: https://docs.kidger.site/diffrax/api/events/ . This is current official documentation but not version-matched because Diffrax is absent locally; it supports the general API comparison only, not current-code validation.
- SUNDIALS CVODE 7.2 official documentation: https://sundials.readthedocs.io/en/v7.2.0/cvode/Usage/ . Used only as a mature counter/cancellation/root-contract design precedent; it is not a dependency recommendation or evidence that RABBIT presently has those semantics.
- Rosenbrock-Krylov order theory: Tranquilli and Sandu, https://arxiv.org/abs/1305.5481 . Exponential-Krylov: Tranquilli and Sandu, https://arxiv.org/abs/1401.2125 . Stiffly accurate EPIRK: Rainwater and Tokman, https://arxiv.org/abs/1604.00583 . These justify research candidates and their order/Krylov obligations; they do not validate EC-EXPRB-K on the current operator.
- Multirate infinitesimal GARK: Roberts et al., https://arxiv.org/abs/1812.00808 . Used to define coupling/order conditions that a multirate proposal would have to satisfy, not to assert a usable RABBIT split.
- Adjoint global error estimation/control: Cao and Petzold, https://escholarship.org/uc/item/3c59x9vs . Used to motivate a goal-oriented error budget, conditional on stable dual weights and independent refinement.
- Adaptive dynamical low-rank nonlinear Boltzmann: Einkemmer et al., https://arxiv.org/abs/2112.02695 . Asymptotic-preserving stiff kinetic schemes: Filbet and Jin, https://arxiv.org/abs/0905.1378 . These are transferable mathematical hypotheses only; neither paper validates RABBIT's cosmological collision operator, invariants, or endpoint observables.

The local diffsol 0.16.0 source/lock, rather than an unmatched web page, is the authority for Rust API details in this analysis.

- Relativistic fermion basis adapted to chemical non-equilibrium: Birrell, Wilkening, and Rafelski, https://arxiv.org/abs/1403.2019 . This is unusually close to the present problem class: a homogeneous/isotropic relativistic Boltzmann equation, time-dependent effective temperature and occupation factor, and a neutrino-freeze-out-motivated model. It supports a moment-constrained adaptive-basis discriminator, but its model result is not endpoint validation for the present collision catalogue.
- High-order asymptotic-preserving exponential schemes for quantum Boltzmann/Fermi gases: Hu, Li, and Pareschi, https://arxiv.org/abs/1310.7658 . It supports an AP research discriminator across stiff kinetic-to-fluid regimes; adapting its equilibrium/penalization split to the current cosmological operator and exact invariants remains a new derivation.

### 2026-08-23 blind-wave closeout

- Mathematical envelope: `A-MAC-MATH.json`, SHA-256 `0e0035f8330e077853be267644337d8953841e49f6dd1fb4a65dca60a2fb32dd`, top-level `fail`; all 34 original IDs occur exactly once.
- Numerical-algorithm envelope: `A-MAC-ALGO.json`, SHA-256 `19823e201ef8fd447123a9ee07023c33278556a73aaf9c108e119da675b74dfb`, top-level `fail`; all 34 original IDs occur exactly once in the envelope.
- Code-contract envelope: `A-MAC-CODE.json`, SHA-256 `c77bc660acfdef7eec919f0df0d2fad461df77902e371860ef9f0f04c580b00d`, top-level `fail`; its `remedy_matrix` contains 34 rows and all 34 original IDs occur exactly once.
- Exact assignment/role/template verification passed for all three. Each was forked with no parent conversation and contractually prohibited from reading siblings or any prior physics-mitigation material. The root did not open any envelope until all three had stopped.
- Blind verdict conflicts are limited to original 018 (mathematics: contract remains inconclusive; algorithm/code: fail/defect repair) and original 031 (mathematics: pass as an extrinsic non-ODE item; algorithm/code: inconclusive as a still-open harness limitation). These must be resolved by contract wording and gate scope rather than a majority vote.
- The algorithm envelope's `n=122` work model is the folded two-bank Rust N60 consumer, while the independent Python trajectory/JVP state is 182 (`3*60+2`). They are different implementations and cannot share one state-dimension claim. Event-stream counts likewise depend on the actual production rule and channel accounting.
- A dedicated `adjudicator` runtime failed before substantive work with HTTP 400 because that runtime/model is unsupported by the current account. This is retained as a harness limitation. One schema-authorized fallback assignment uses runtime type `default` with the exact sealed adjudicator role and result template; its evidence inputs and adjudication task are unchanged.

### 2026-08-23 stronger derivative-only no-go bound

The earlier 182-fold counterfactual treated one state dimension per eliminated RHS group. An even more favorable bound assumes every projected call belongs to a centered dense finite-difference Jacobian with `2n+1=365` RHS calls and that an analytic derivative removes all but one at no setup cost while leaving everything else unchanged. Then 32,704,637/365 = 89,601.75 full-RHS-equivalent calls, requiring 396,095.27 seconds at 4.42062 seconds/call: 4.58 days, 6.113 times the 64,800-second wall and 16.291 times the 5,500-call prefix cap. Therefore derivative-call elimination alone is insufficient even under this deliberately impossible best case. This remains a conditional work-model no-go, not a universal complexity lower bound; a viable method must additionally reduce accepted/rejected steps or the full collision/operator cost by more than those residual factors, with margin.

### 2026-08-23 final adjudication

- Fallback adjudicator envelope: `A-MAC-ADJ2.json`, SHA-256 `1990816777280bbbf4fe3f3f4c59fba250e6a0a4d2c630308dcd0935dc6f5842`, top-level `fail`.
- Mechanical check proves the ordered ID array is exactly `F-ODE-ADJ3-001` through `F-ODE-ADJ3-034`, unique and length 34. Verdicts remain 24 fail, 7 inconclusive, and 3 pass. Every row contains a disposition, remedy/no-go, mathematical prerequisite, algorithm/complexity, code insertion/test, hard falsifier/decisive experiment, and residual claim ceiling.
- The adjudicator resolves 018 as `fail`: the ambiguity is removed by specifying both `last_accepted` and an actual `failure_snapshot`; when the dependency cannot expose the latter, the result must say `snapshot_unavailable` with a reason, not substitute the step start. It resolves 031 as `inconclusive` and extrinsic: no ODE patch is appropriate, while `G-HARNESS-INTEGRITY` remains FAIL.
- The adjudicator rejects a sibling's assumption of a retained N approximately 0.1653 restart checkpoint because the D-071 domain phase retained no state dump/checks/step trace. Static physical-prefix states can only support a static discriminator. A gate-bearing full-prefix attempt must begin from the physical initial state, include N=0.14 and 0.22, reach N>=0.25, and meet both the 5,500-call and 64,800-second caps.
- The adjudicator also rejects a sibling's manifest-derived package versions as executed runtime evidence. Live imports in this session are NumPy 2.4.2 and SciPy 1.17.0; JAX and Diffrax are unavailable. No JAX/Diffrax or exact release-environment execution is claimed.
- Ranked work remains bounded to: (1) raw accepted-state physical admission in `dynamic_collision_driver`; (2) typed NumPy Rodas event/admission repair; (3) a `PROPOSED` rigorous moment-constrained slow-manifold discriminator with full-grid residual and adjoint QoI certificate. Fermi-Dirac AP penalization remains a scientifically plausible second-line hypothesis, not a first implementation candidate.
- No production source, test, gate, registry, shared scientific document, or endpoint trajectory was modified or executed. Gate states remain `G-F10-INDEPENDENT-FLRW=FAIL/CLOSED_ON_CURRENT_MEASUREMENT` and `G-HARNESS-INTEGRITY=FAIL`.

### 2026-08-23 integrity closeout

- Harness validation passed with the independent run active, then `ACTIVE_RUN` and the two generated context timestamps were restored byte-for-byte to the pre-work state and harness validation passed again for `run-20260806-bd623-audit-triage`.
- `git diff --check` passed; the restored context files have no diff; final `git status --short` contains only the pre-existing untracked `RABBIT_diagnosis_report.bundle`.
- The ephemeral bounded-work contract validated with terminal decision `STOP_INVALID`, not `COMPLETE_ACCEPTANCE`: frozen criterion MAC-4 named `A-MAC-ADJ.json`, the dedicated runtime failed before work, and the one allowed repair-closeout produced the substantively valid sealed-role result at `A-MAC-ADJ2.json`, outside that frozen evidence cell. The frozen hash/path was not rewritten and the fallback was not impersonated as the failed runtime.
