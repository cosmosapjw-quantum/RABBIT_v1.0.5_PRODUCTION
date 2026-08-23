# SCIENTIFIC_CONTRACT.md

## Scientific objective

Determine which prior mathematical/algorithmic/coding remedies can make the current rec_bianchi ODE/DAE stack adjudicably correct, stable, fail-closed, and reproducible, and produce a future execution plan without implementing it.

## Governing definitions

- Local residual contract `F(eta,y,y_eta)=0`; square named row/variable partitions; native suboperator clocks and conversion/JVP; scaled `F_ydot` rank, pencil regularity and (where semi-explicit) `g_z`; continuous event functions `g_k(eta,y)`; accepted/provisional history and restart; nonlinear/linear residual certificates, invariants, and stage-specific outcome taxonomy.
- An accepted state is durable only after independent domain, residual, invariant, endpoint, operator/policy, history and event-generation validation.
- Numerical success, physical/scientific admission, replay identity and trajectory validation are distinct claims.

## Conventions

- metric signature: inherited current source `(-,+,+,+)` where applicable; not changed here
- Fourier convention: not applicable to this design audit unless a current kernel explicitly uses it
- index convention: source-declared array/state roles; no inferred reordering
- unit system: current SI/public source units; every tolerance/error weight must be dimensionally typed. Per-second `R_t` enters `eta=ln(a)` as `R_t/H`, with JVP `DR_t/H-R_t DH/H^2`; every suboperator declares its native clock and orientation.
- normalization: nonlinear admission uses a first-order componentwise scaled-defect certificate with an assembled or certified upper-bound absolute-Jacobian scale and remainder scope; exact componentwise backward-error terminology is reserved for justified linear subproblems. Verdicts remain invariant to equivalent unit and null-basis scaling.
- stochastic convention: deterministic probes; no new random or Monte Carlo claims
- coordinate/gauge convention: current independent variables are preserved; no physics/frame/background change is authorized

## Valid regime

- parameter range: only current declared source/test regimes and bounded analytic counterexamples
- resolution range: unit/small-system probes and targeted existing tests; no production grid/trajectory
- asymptotic assumptions: only theorem-declared limits; AP/preconditioner claims remain conditional until a completed operator/reduced model exists
- perturbative order: unchanged current implementation
- excluded singular regimes: unimplemented E1C/full macro, unidentified point support, unsupported background regimes, and any rank/event segment lacking certification

## Required invariants

- dimensions/clocks: acceptance is invariant to equivalent unit and per-second/per-eta representation, including nonzero clock derivative; function/time tolerances are distinct
- conservation laws: independently assembled applicable invariants, not tautological opposite-sign ledgers
- symmetry: nullspace and rank decisions invariant to basis scaling/rotation
- positivity: no hidden clipping; domain failures are typed retry/fatal outcomes
- normalization: certified first-order nonlinear scaled defect, linear componentwise backward error, relative structural thresholds and independently checked denominators
- DAE regularity: no index-one claim from a mask alone; square partitions, scaled rank, pencil/`g_z` checks and current-cell sweeps or typed irregular/higher-index failure
- exact identities: rejected trials and provisional overlays do not mutate accepted state/history; primitive/derived digests stay coherent; restart cannot cross clock/operator/policy/event/code/environment identity
- event semantics: continuous standard guards, deterministic simultaneous-event batch/priority, explicit left/right state, zero-speed events and typed chattering/Zeno stop

## Known limits

| Limit | Expected result | Tolerance | Reference/test |
|---|---|---:|---|
| zero optical depth | finite analytic transfer/JVP series | multiprecision forward-error budget | X4 |
| null-basis scaling/rotation | identical projection/rank/outcome | relative SVD/QR budget | X5 |
| unit/component/clock rescaling | identical residual/JVP/admission, including `DH` term | independent absolute-Jacobian/clock oracle | X5/V01--V04 |
| event root inside `h_min` | positive-control certificate and convergence; otherwise typed unsupported/uncertified | separate time/function tolerances | X2/X3/V06 |
| reject/resume | accepted state/history unchanged; provisional overlay destroyed; split semantics equivalent | exact local identity plus physics tolerance | X2/V05/V11--V12 |

## Reference cases

- analytic toy case: backward Euler and Volterra/delay step doubling, per-second/per-eta clock conversion, polynomial/grazing/simultaneous/Zeno events, removable transfer singularity, regular/irregular DAE and rank/null compatibility fixtures
- trusted numerical reference: high-precision mpmath for local primitives; independent analytic/root-count oracles
- previous implementation: current HEAD only; prior research report is seed, not reference authority
- published benchmark: none admitted for the full current operator

## Numerical requirements

- target precision: componentwise forward budgets, first-order nonlinear scaled-defect budgets and justified linear backward-error budgets derived per test; no global magic tolerance
- convergence order: future method-specific order and event-time convergence; not claimed in this loop
- stability expectations: exhaustive stage-typed outcomes, feasible positive controls must succeed, no NaN/Inf success, no silent least-squares semantic switch, no always-retry/always-unsupported/always-uncertified implementation
- random seed policy: deterministic fixed inputs only
- ensemble size: not applicable
- acceptable runtime/memory: bounded unit probes and targeted tests; history design must avoid quadratic append/hash growth; any scalability claim uses the preregistered V13 grid/stiffness/iteration/memory thresholds

## Failure semantics

다음 상태를 성공처럼 처리하지 않는다: NaN/Inf, non-convergence, empty result, clipped invalid values, silently replaced missing data, fallback approximation outside its regime, irregular/higher-index DAE, unresolved simultaneous/Zeno event, infeasible remap, incomplete problem, budget exhaustion, rejected staging generation.

## Change control

convention, baseline, tolerance, approximation order, output semantics 변경은 승인 필요.
