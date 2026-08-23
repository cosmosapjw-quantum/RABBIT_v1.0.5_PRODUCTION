# Phase 7 - minimal decisive verification design

All tests below are prospective designs. None ran in this research cycle. `PASS` means only that the candidate may advance to its stated escalation; it does not reopen D-071 or establish endpoint/scientific authority. H0 remains the null whenever a result is ambiguous.

## V0 - common null and evidence identity

TESTABLE_CLAIM: A candidate result is attributable to the proposed mechanism rather than stale bytes, a changed physical case, an omitted work bucket, or an unmeasured solver state.

REQUIRED_INPUTS: Exact HEAD/tree/environment identity; independent-Python versus folded-Rust state-layout declaration; frozen physical case; raw accepted/failing states; terminal cause; accepted/rejected/attempt counts; RHS/J/Jv/dfdt/LU/refinement and collision-kernel work; physical invariant residuals.

EXPECTED_IF_CORRECT: Candidate-specific predicate changes while physical case, claim ceiling, and all unrelated invariants remain fixed.

EXPECTED_UNDER_H0: No stable mechanism-specific improvement, or the observed difference is explained by state/config/work-accounting drift.

PASS_CRITERION: Exact identity and complete raw/work evidence are present and the candidate-specific criterion below passes prospectively.

KILL_CRITERION: Identity mismatch, missing raw failure, fabricated/partial counter, post-output tuning, or static/segment evidence offered as trajectory/endpoint evidence.

AMBIGUITY: Evidence is internally consistent but lacks the solver state or independent oracle needed to attribute cause.

CHEAPEST_NEXT_ACTION: Freeze one case manifest inside an existing result surface; do not add a new gate or registry.

ESCALATION_PATH: None until V0 passes. H0 remains controlling on ambiguity.

## V1 - H1 two-layer admission versus software-only comparator

1. TESTABLE_CLAIM: The physical-coordinate/invariant layer rejects physically invalid raw states beyond what a typed software-only terminal certificate rejects, without clipping valid output or conflating last accepted and failing states.

2. REQUIRED_INPUTS: A fixed table containing initial/exact-zero event, wrong direction, nonfinite event, failed refinement, contradictory target/failure, failed partial state, exact `f=0/1`, occupation saturation, nonpositive `T/H`, invalid abundance/normalization, plus matched valid controls; declared physical error scales for `delta_f=f(1-f)delta_u` and `delta_X=X delta_z`.

3. EXPECTED_IF_CORRECT: Both candidates reject software-status mutants; only the H1 physical layer additionally rejects raw domain/balance violations; no invalid value is clipped/floored/projected; valid controls are bitwise or tolerance-equivalent and retain raw states.

4. EXPECTED_UNDER_STRONGEST_COMPETITOR: Software-only typed outcomes kill event/status mutants but add no independent physical-domain discrimination; if it kills the same complete table, H1's coordinate claim adds no value.

5. PASS_CRITERION: 100% of the prospectively named critical mutants are killed; 0 valid controls are rejected; every failure exposes both terminal cause and raw offending state separately from last accepted; H1 kills at least one physical mutant that the software-only comparator admits; no projection is used.

6. KILL_CRITERION: Any critical mutant is admitted, any invalid raw state is hidden, any valid strict-domain control is falsely rejected due to coordinate saturation, or H1 adds zero discrimination over the comparator.

7. AMBIGUITY: An extreme-tail state is mathematically physical but unrepresentable and no higher-precision oracle decides whether the error allocation should admit it.

8. CHEAPEST_NEXT_ACTION: Specify and execute the table against existing outcome/admission functions only; no production refactor first.

9. ESCALATION_PATH: A bounded PR-A that repairs the existing canonical/NumPy outcome path and reuses the same table. This remains correctness-only.

## V2 - H2 nonredundant electromagnetic-energy coordinate and clock

1. TESTABLE_CLAIM: Replacing only `T_gamma` by `rho_em` on the spectral path yields a well-conditioned one-to-one EOS coordinate, and `x=-ln(T_gamma/T_ref)` is admissible on the full proposed flat-FLRW branch.

2. REQUIRED_INPUTS: Frozen EOS and physical-state envelope including electron-annihilation, energy-transfer extrema, completing/stalled configurations and boundaries; independently bounded `rho_em(T)`, `d rho_em/dT`, `Q_e+2Q_x`, `H`; allocated EOS/derivative errors; competitor using `T_gamma` in `N`. `rho_nu` must be derived from `f`, not added as a redundant state.

3. EXPECTED_IF_CORRECT: `d rho_em/dT` stays finite and positive; inversion round-trips within its allocated error; the certified lower bound `underline(alpha)=min(alpha_hat-delta_alpha)` is positive; energy-coordinate derivative/first-law error is smaller and no worse in any state than the analytic-`T` competitor.

4. EXPECTED_UNDER_STRONGEST_COMPETITOR: Analytic thermodynamic derivatives in the existing `T_gamma(N)` coordinate match or outperform the energy coordinate, while a certified event refiner handles the endpoint without dividing by `alpha`.

5. PASS_CRITERION: Across every frozen state, `inf(d rho_em/dT - delta_cv)>0`, EOS round-trip error is at most one half of its preallocated state-error budget, `underline(alpha)>0`, flat-FLRW first-law residual stays within its frozen cap, and the energy coordinate reduces the worst derivative/cancellation error by at least 10x without increasing any allocated physical error.

6. KILL_CRITERION: Any nonunique/ill-conditioned inverse, `underline(alpha)<=0`, incorrect heavy-species multiplicity/sign, first-law failure, redundant energy state, or no conditioning advantage over analytic `T_gamma(N)`.

7. AMBIGUITY: All static states pass but the sealed envelope omits a transfer extremum or the numerical error bound is larger than the alpha margin.

8. CHEAPEST_NEXT_ACTION: A read-only EOS/derivative/alpha calculation on the frozen state envelope; stop before a coordinate implementation if any kill condition occurs.

9. ESCALATION_PATH: One Rust endpoint-consumed coordinate slice, then same-physics activation/cold comparison. A clock does not replace general event semantics and has no independent D-071 authority.

## V3 - H3 one-channel support-safe affinity and scaled Jv

1. TESTABLE_CLAIM: For one exact endpoint-consumed reaction channel, a piecewise cancellation-safe gain/loss form plus exact channel invariants and physically scaled derivatives is correct over equilibrium, nonequilibrium and support edges and can justify one active-path derivative experiment.

2. REQUIRED_INPUTS: Sealed current channel bytes, legs, units, support/threshold, multiplicity, blocking factors, chemical-potential convention, reaction vector and bath coupling; states with `(G,L)=(0,0),(0,+),(+,0),(+,+)`, near equilibrium and far from it; high-precision direct oracle; exact valid left-invariant rows; fixed block/error scales; centered epsilon ladder; full hidden-work accounting.

3. EXPECTED_IF_CORRECT: With `C=G-L`, branches return `0`, `-L`, `G`, and for `G,L>0`, `-G*expm1(-A)` when `A=ln(G/L)>=0` or `L*expm1(A)` when `A<0`; values and directional derivatives match the oracle; only valid invariant rows annihilate action/Jacobian; near-equilibrium cancellation improves.

4. EXPECTED_UNDER_STRONGEST_COMPETITOR: Existing analytic dense blocks plus physically scaled centered differences match correctness; affinity improves cancellation but not work/step behavior, or a hidden dense Jacobian supplies the apparent Jv.

5. PASS_CRITERION: Every support branch is finite and oracle-consistent within frozen mixed absolute/relative tolerances; equilibrium and invariant residuals stay below their predeclared caps; Jv error contracts on at least two consecutive epsilon halvings before roundoff and meets the cap in every declared direction; no passive-time component or hidden dense work enters geometry; the later endpoint-consumed same-case experiment must reduce full-RHS-equivalent calls or wall by at least 10% with raw parity/invariants preserved.

6. KILL_CRITERION: Undefined/overflowing zero-support branch, wrong sign/multiplicity/invariant, nonconvergent Jv ladder, domain escape, hidden dense fallback, raw-parity failure, or less than 10% same-case movement at escalation.

7. AMBIGUITY: The one channel passes but profiling shows it is non-dominant or its invariant space changes when folded into the full bath-coupled operator.

8. CHEAPEST_NEXT_ACTION: Complete the exact algebra and high-precision one-channel oracle first; no solver/backend code.

9. ESCALATION_PATH: One Rust analytic/block derivative vertical slice selected by measured N48 structure. Even a pass is local correctness/performance, not D-071 reopening.

## V4 - H4 tiny actual-kernel tail majorant

1. TESTABLE_CLAIM: For one actual collision channel and prospectively bounded state family, an absolute tail majorant can tightly contain the independently measured extended-domain lost action; population/energy moments alone are not used.

2. REQUIRED_INPUTS: A sealed non-equilibrium state family with proved weighted decay envelope such as `f_s(y)<=A_s exp(-beta_s y)`; exact finite-mass kernel/support/routing/blocking factors; an absolute kernel majorant; cutoff ladder; independent high-precision extended-domain oracle; preallocated local collision-source error `epsilon_C` and weak-rate weights.

3. EXPECTED_IF_CORRECT: The analytic/numerical bound contains absolute lost action for every state and cutoff, contracts as the domain grows, and weak-rate weighting can require a different domain than energy weighting while the joint rule remains feasible.

4. EXPECTED_UNDER_STRONGEST_COMPETITOR: Direct extended-domain computation is cheaper or more accurate; the bound is valid but too loose to decide, or signed cancellation makes moment agreement look good while absolute lost action remains large.

5. PASS_CRITERION: No oracle escape; monotone contraction on the frozen cutoff ladder; `B_tail<=epsilon_C/2` for every state; tightness `B_tail/max(oracle_lost_action,epsilon_floor)<=10`; both energy and weak-rate weighted variants pass. All constants are frozen before oracle output.

6. KILL_CRITERION: No provable decay envelope, nonfinite majorant, any oracle escape, noncontraction, `B_tail>epsilon_C`, or a bound whose cost exceeds the extended-domain comparator.

7. AMBIGUITY: Containment holds but `epsilon_C/2 < B_tail <= epsilon_C`, tightness lies between 10 and 100, or the reference shares the same routing/kernel implementation.

8. CHEAPEST_NEXT_ACTION: One tiny channel/state/cutoff proof-and-oracle calculation; stop before weak/abundance propagation if it is not tight.

9. ESCALATION_PATH: Add weak-rate functionals, then a separately justified sensitivity/adjoint or paired-trajectory propagation to `Yp`, `D/H`, `Neff`. H4 must pass before any H5 reduced-domain claim.

## V5 - H5 exact invariants and non-normal micro-decay discriminator

1. TESTABLE_CLAIM: The exact coupled collision+bath operator admits a unit-consistent macro/micro split whose micro dynamics are uniformly contractive in a prospectively fixed physical/entropy metric across a contiguous relevant regime and multiple grids.

2. REQUIRED_INPUTS: Exact reaction/bath operator and multiplicities; conserved rows `L`, equilibrium map `E(U)`, tangent, projector `P`; completing and stalled state sets across pre-creep/creep/decoupling; at least two grids; fixed nondimensionalization and SPD metric `M`; if state-dependent, the along-path `M'`; generalized eigenvalue/log-norm, resolvent/pseudospectral and transient-growth calculations; approximation threshold `kappa_*` and amplification cap `G_*` derived prospectively from the frozen error/budget allocation.

3. EXPECTED_IF_CORRECT: `L R=0`, `R(E(U))=0` and projector identities hold; on the microspace, `P^T(MJ+J^T M)P/2 <= -lambda P^T M P` with `lambda/H >= kappa_*` throughout the claimed regime, or the state-dependent version including `M'` holds; transient amplification is at most `G_*`; results are grid- and metric-stable; overlap with full dynamics can be bounded.

4. EXPECTED_UNDER_H0/H3: No uniform coercivity margin; apparent eigenvalue gap is scaling/grid-dependent or accompanied by large non-normal transient growth; the stall instead correlates with derivative/cancellation or unrecorded solver behavior.

5. PASS_CRITERION: Exact invariants/equilibrium pass; a predeclared metric gives a strictly positive lower coercivity margin meeting `kappa_*` across every frozen state and grid; transient/resolvent amplification stays below `G_*`; completing/stalled comparison does not destroy the margin; the resulting cost/error model projects the later sealed prefix within 5500 full-RHS-equivalent calls and 64800 s with uncertainty margin before implementation authority is requested.

6. KILL_CRITERION: Missing/incorrect invariant or equilibrium, singular Fermi metric at admitted states without a controlled interior margin, nonpositive/nonuniform coercivity, large pseudospectral/transient amplification, material grid/metric sensitivity, fixed-cut relabeling, or cost model outside the registered prefix budget.

7. AMBIGUITY: Positive decay exists only on isolated states, depends on post-output metric choice, or clears physics but not the prospective cost threshold.

8. CHEAPEST_NEXT_ACTION: Zero/new-production-code derivation of `L,E,P,M` and the symmetric-part/pseudospectral calculation on the existing state set. Stop immediately on a kill condition.

9. ESCALATION_PATH: Only after V4 safety and V5 mechanism both pass, request a separately authorized single Rust AP vertical slice with overlap and the existing N=0.14/0.22 to N>=0.25 discriminator. No AP framework first.

## V6 - H6 independent exact-limit mutant table

1. TESTABLE_CLAIM: Independently derived, correctly scoped physical limits plus semantic counterexamples kill the named common-mode false greens while retaining matched good paths.

2. REQUIRED_INPUTS: Operator/scope table for collisionless homogeneous isotropic `df(q)/dN=0`, common-temperature/compatible-chemical-potential collision null, flat-FLRW first law with `Q_e+2Q_x`, nuclear-suboperator baryon/charge, weak n<->p baryon-only conservation, and event initial-zero/wrong-direction/transverse/grazing/nonfinite cases; independent analytic/high-precision or cross-formulation oracles; one good control per mutant.

3. EXPECTED_IF_CORRECT: Each retained defect mutant violates at least one independent predicate; good controls remain accepted; charge/energy invariants are never asserted outside their suboperator/model.

4. EXPECTED_UNDER_STRONGEST_COMPETITOR: Existing backend parity passes both original and mutant due to shared helpers/tableau; broad happy-path tests show green without killing semantic or physics mutants.

5. PASS_CRITERION: 100% kill rate on the frozen critical-mutant set, zero good-control false positives, an oracle provenance distinct from the mutated path, and exact scope/units/signs recorded for every predicate.

6. KILL_CRITERION: Any critical mutant survives, any oracle reuses the mutated computation, a limit is applied outside its model/channel, or tests reject all cases indiscriminately.

7. AMBIGUITY: A mutant does not perturb the measured predicate enough to decide or independent formulations disagree within unallocated numerical error.

8. CHEAPEST_NEXT_ACTION: A bounded table using existing test surfaces and hand-derived or high-precision oracles; no generic mutation framework.

9. ESCALATION_PATH: Reuse the exact table in H1/H3/H5 implementation reviews. Passing remains one-sided falsification evidence, not trajectory validation.

## Information-priority order

1. V1/V6: immediately retire false-success and common-mode-validation uncertainty at low cost.
2. V3: derive one exact current channel and establish whether structured derivatives are even admissible.
3. V2: kill or retain the thermodynamic coordinate before implementation.
4. V4: test whether a useful tail certificate is mathematically feasible.
5. V5: only after exact operator ingredients exist; it is the highest endpoint leverage and highest assumption burden.

This order is epistemic, not implementation authorization.
