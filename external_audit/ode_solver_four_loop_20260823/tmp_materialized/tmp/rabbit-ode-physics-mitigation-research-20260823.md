# RABBIT ODE physics-specific mitigation research record

## Frozen preregistration

Question: For every open ODE issue/blocker at diagnosis_report HEAD 78f5b091bd8de99bd8d0b1eddd8a0b7978f5da2b, which remedies use the actual BBN, neutrino-transport, cosmological, thermodynamic, and reaction-network structure to remove or materially reduce the failure, and which remedies are merely generic solver plumbing or unsupported speculation?

Hypotheses:

- H1: Reparameterizing evolved states by their physical domains (simplex/log-ratio abundances, open-interval occupation variables, positive temperature/scale variables) plus raw residual reconstruction can prevent invalid accepted states without output clipping. Falsifier: transformed dynamics loses exact conservation/equilibrium limits, becomes singular in the physically visited regime, or does not reduce endpoint failures.
- H2: Exposing collision/network block structure, analytic thermodynamic derivatives, detailed-balance linearization, and conservation nullspaces to an implicit solver can reduce the activation/cold endpoint wall materially. Falsifier: same-physics endpoint-consumed runs show no material RHS/Jacobian/linear-solve or wall reduction, or parity/conservation degrades.
- H3: Physics-defined terminal surfaces and monotone thermodynamic clocks can make event admission fail-closed and backend-independent. Falsifier: the chosen clock/event is not monotone across allowed anisotropic states, or event residuals do not bound observable error.
- H4: Asymptotic/equilibrium partitioning based on reaction rates versus Hubble expansion can replace prohibitively long full-dynamics tails while preserving observables to a prospectively fixed tolerance. Falsifier: freeze-out/tail matching errors exceed the tolerance or omit a slow mode relevant to Y_p, D/H, or N_eff.
- H5: Physics invariants and independent reduced limits can turn current false-green tests into discriminating validation without adding standalone gate plumbing. Falsifier: proposed invariants are not exact/controlled for the implemented model or cannot distinguish known counterexamples.

Methods:

- M1 local-authority map: bind each of the 32 open root issues to current code, equations, gate authority, and active/deprecated runtime status.
- M2 literature review: primary papers and official solver documentation on BBN kinetics, neutrino Boltzmann collision structure, positivity/conservation-preserving kinetics, stiff BDF/Rosenbrock methods, event localization, sparse/block Jacobians, and asymptotic-preserving reduction.
- M3 physics audit: check assumptions, variables, units, signs, equilibrium and FLRW/LRS/no-collision limits, boundary conditions, conservation, positivity, and stiffness scales for each proposed remedy.
- M4 implementation feasibility: identify the smallest Rust-first executable change, expected blocker movement, failure risks, and a falsifying BBN-shaped test; SciPy remains number-of-record and JAX frozen oracle.
- M5 independent adjudication: a fresh reviewer receives the frozen plan, evidence log, and candidate claims and returns confirmed, partially-confirmed, rejected, or inconclusive.

Confirmatory work:

- C1: Map all 32 open roots to at least one physics-specific mitigation or explicitly prove that the remedy is necessarily software-contract-only.
- C2: For every proposed physics remedy, record at least one primary/authoritative source and one current local-code touchpoint.
- C3: Rank at most three executable first-PR candidates using blocker movement per line, endpoint relevance, raw-state preservation, and falsification strength.
- C4: Preserve the current claim ceiling: no QKE, public-production, scientific-promotion, or endpoint-completion claim.

Exploratory work:

- X1: Inspect whether reaction-timescale partitioning, equilibrium-manifold projection, Schur/block preconditioning, or moment-conservative collision projection appears compatible with the current state layout.
- X2: Identify failure modes where a physics transformation worsens conditioning or hides raw failure.

Stopping rules:

- Stop when C1-C4 are complete, the independent verdict is not inconclusive, the bounded agent/review budget is exhausted, or evidence cannot distinguish a candidate from generic tuning.
- No new full trajectory, endpoint, QKE, production promotion, dependency installation, code edit, gate/ledger edit, or external publication action is authorized.

## Evidence log

- E000 | 2026-08-23 | Local pre-read before freeze: exact HEAD, dirty-tree custody, mandatory anti-drift and prior exhaustive audit were inspected to define scope. These observations are context, not confirmatory results.
- E001 | 2026-08-23 | Current audit authority: `A-ODE-ADJ3.json` contains 34 adjudicated records (24 fail, 7 inconclusive, 3 pass/closed). The open engineering register used here is the previously normalized 32-root register; retained gate authority is `G-F10-INDEPENDENT-FLRW=FAIL` and `G-HARNESS-INTEGRITY=FAIL`. The independent order-60/ymax-30 SciPy-BDF holdout evaluated no scientific predicate and projected an approximately 8305x wall miss. Source: `.agent-harness/runs/run-20260806-bd623-audit-triage/results/A-ODE-ADJ3.json`.
- E002 | 2026-08-23 | Current Rust physics structure: natural units and Friedmann sign convention are explicit in `native/rabbit_cpu/src/flrw.rs`; `N=ln(a)` and `q=ap` make the massless collisionless invariant exactly `df(q)/dN=0`; the full isotropic state already uses strict-open logit occupations but also carries a passive elapsed-time coordinate. The thermal network already uses log abundances/log baryon density and an analytic abundance Jacobian; every selected reaction is checked for baryon and charge conservation. These are IMPLEMENTED baselines, not new proposals. Sources: `native/rabbit_cpu/src/{flrw,isotropic_boltzmann,thermal_bbn,minimal_network}.rs`.
- E003 | 2026-08-23 | Physical-variable conditioning is locally measured: the historical 3T temperature equation divides collision-energy error by `c_v proportional to T_nux^3`, while the energy equation `d rho_nux/dN=-4 rho_nux+dQ_nux/dN` produced an approximately 1e12 conditioning improvement in its bounded probe. A Python clean-core formula exists, but its helper still contains floors/projections and is not Rust endpoint authority. Current Rust `ThreeTemperatureFlrwSystem` still evolves temperatures and uses relative-only finite differences. Sources: `docs/audit/BD205_qlaguerre_collision_root_cause_2026-07-01.md`, `src/rabbit/collisions/dynamic_collision_core.py`, `native/rabbit_cpu/src/flrw.rs`.
- E004 | 2026-08-23 | Negative-result constraint: simple momentum-tail algebraic slaving is rejected. Across all cuts the required fast/slow separation never reaches one; only a three-dimensional tail is genuinely gapped; the separation disappears during decoupling; a nonlinear 54-equation closure costs a full collision RHS per residual and yields at most 1.25x with one inner iteration, slower thereafter. Removing all Jacobian-construction calls still misses the wall by 77x-694x, and Rodas5P was historically 2x-3x slower. Source: `docs/audit/BD622_V2_option3_closed_and_protocol_2026-08-04.md` (superseding the earlier V1C motivation).
- E005 | 2026-08-23 | Primary BBN physics: PRIMAT derives plasma entropy/heat-transfer relations `a(T)`, Friedmann evolution, weak freeze-out by competition of weak rates and Hubble expansion, exact baryon normalization, temperature-regime network activation, and stresses that detailed-balance violations artificially shift the equilibrium neutron abundance. Source accessed 2026-08-23: https://arxiv.org/html/1801.08023v3 .
- E006 | 2026-08-23 | Primary neutrino-decoupling numerics: Froustey-Pitrou-Volpe use comoving `x=m_e/T_cm`, `y=p/T_cm`, `z=T_gamma/T_cm` plus the total-energy equation. They show a collision RHS costs O(N^3), a finite-difference Jacobian O(N^4), and a direct differentiated collision Jacobian O(N^3), measured about `N/5` faster than finite differencing. This supports analytic/block JVP work but cannot by itself close an 8305x wall. Source accessed 2026-08-23: https://arxiv.org/html/2008.01074v2 .
- E007 | 2026-08-23 | Primary coupled-physics warning: Grohs et al. solve multi-energy neutrino transport jointly with thermodynamics and the nuclear network and report nonlinear feedback in the scale-factor/temperature phasing, neutron-proton ratio, yields and N_eff. Therefore an endpoint surrogate must bound both total-energy and weak-rate weighted spectral errors, not N_eff alone. Source accessed 2026-08-23: https://arxiv.org/abs/1512.02205 .
- E008 | 2026-08-23 | Primary asymptotic-preserving condition: for `U'=F(U)+R(U)/epsilon`, AP schemes must reduce consistently to the equilibrium system on `R(U)=0`, preserve conserved quantities `Q R=0`, and handle non-well-prepared initial layers; stiff accuracy matters for nonconserved components. This supports a collision-invariant micro-macro hypothesis, not an unverified switch based only on temperature. Source accessed 2026-08-23: https://arxiv.org/html/1009.2757 .
- E009 | 2026-08-23 | Primary structure-preserving precedent: mass-action integration should preserve both nonnegative concentrations and mass conservation; conservative spectral Boltzmann methods enforce collision invariants through a constrained correction. Applicability here is methodological only: the neutrino channels require channel-specific invariants and Pauli/detailed-balance structure. Sources accessed 2026-08-23: https://epubs.siam.org/doi/10.1137/100789592 and https://arxiv.org/abs/1306.4625 .
- E010 | 2026-08-23 | Official solver boundary: SciPy 1.17 documents that events are detected from per-step sign changes and multiple crossings can be missed, and recommends supplied Jacobians/sparsity for BDF/Radau. Therefore a monotone thermodynamic clock can remove a physics endpoint event, but initial-zero, NaN, direction, status, and refinement semantics remain software obligations. Source accessed 2026-08-23: https://docs.scipy.org/doc/scipy-1.17.0/reference/generated/scipy.integrate.solve_ivp.html .
- E011 | 2026-08-23 | Derived candidate: use a three-regime, invariant-aware formulation. At high temperature evolve the equilibrium macro manifold and a prospectively bounded first-order micro correction; in the decoupling window retain full spectral dynamics; at low temperature switch to collisionless comoving transport only after an integrated collision-source bound closes simultaneously for energy density, weak-rate kernels, Yp and D/H. This differs from the rejected fixed momentum-tail slaving because the split is by collision invariants and `Gamma/H`, and it must be falsified by the smallest nonzero collision-mode rate, handoff residual, and endpoint parity.
- E012 | 2026-08-23 | Derived event/JVP candidates: (a) if `dT_gamma/dN` is strictly negative with a certified lower magnitude, use `x=-ln(T_gamma/T_start)` and `dy/dx=F/[-d ln T_gamma/dN]`, eliminating the temperature root; (b) do not use one Euclidean finite-difference epsilon on `(state,N)`. Separate `J_y F v_y` and `partial_N F v_N`, nondimensionalize by physical block/error scales, and exclude passive quadratures such as elapsed time from the perturbation norm. Both require direct raw-state and first-law checks.

## Deviations and pivots

None at freeze time.

## Final verdict

Pending independent adjudication.

## Pre-adjudication synthesis (frozen after independent blind reviews)

### Remedy families

- R1 domain and raw admission: retain the Rust strict-open occupation logit and log-abundance/stoichiometric structures; remove output/RHS clipping, H floors, observable denominator floors and post-solve normalization. Admit raw states only after status, expected terminal cause, residual, finite/domain, balance and normalization checks.
- R2 thermodynamic variables: for a heat source `Q>0` into neutrinos, use `d rho_nu/dN=-4 rho_nu+Q/H` and `d rho_em/dN=-3(rho_em+P_em)-Q/H`; recover `T_gamma` only through a verified monotone EOS. If `alpha=-d ln(T_gamma)/dN` has a sealed positive lower bound, use `x=-ln(T_gamma/T_ref)` so the endpoint is fixed rather than root-found.
- R3 collision structure and derivatives: write each Fermi 2-to-2 gain-loss bracket as `L*expm1(A)` near detailed balance, where `A=logit(f3)+logit(f4)-logit(f1)-logit(f2)` with the reaction's chemical-potential convention. Enforce only channel-valid discrete invariant rows, with electron-bath energy exchange carried in the electromagnetic macro equation. Use analytic/block JVPs and separate `J_y F v_y` from `partial_N F v_N`; exclude passive elapsed time from perturbation/error geometry.
- R4 observable-aware discretization: choose and test the momentum rule jointly against energy transfer, `lambda_np/H`, `lambda_pn/H`, collision lost action and final `Yp`, `D/H`, `N_eff`; an energy-only or equilibrium-population tail test is insufficient.
- R5 conditional AP reduction: if the projected non-null collision spectrum shows a sealed gap, decompose `f=M(U)+g` with `g` orthogonal to the exact discrete invariants. Use a high-temperature equilibrium/micro-correction regime, full N48 dynamics through decoupling, and a low-temperature collisionless regime only after integrated source bounds close for energy, weak-rate kernels and final abundances. This is distinct from rejected fixed momentum-tail slaving but remains PROPOSED.
- R6 physics falsifiers: common-temperature detailed-balance null, collisionless `df(q)/dN=0`, FLRW and generic-to-LRS limits, exact reaction stoichiometry, finite-mass first law, rate-equilibrium neutron fraction, transverse/nontransverse event cases and production-state derivative ladders.

### Mechanical 32-root coverage

Legend: `P` physics can materially attack the root; `A` physics assists but software-contract repair is mandatory; `S` no honest physics-only remedy; `E` missing execution evidence.

| Root | Prior finding/surface | Class | Frozen mitigation and residual blocker |
|---|---|---:|---|
| C1 | F-ODE-ADJ3-001, D-071 endpoint wall | P | R5 is the only identified route plausibly able to change the asymptotic cost; require channel nullspaces, projected spectrum, overlap and the sealed stalled-phase/endpoint discriminator. No reopen now. |
| C2 | F-ODE-ADJ3-002, invalid physical-prefix JVP | P | R3: hold N fixed, remove passive elapsed time, block-scale directions, preserve domains, compare analytic/dense/high-precision Jv. Static states remain non-trajectory evidence. |
| C3 | F-ODE-ADJ3-003, tier-2 Yp grid drift | P | R3+R4: conservative gain/loss, detailed-balance affinity and weak-rate-weighted grid/domain ladder through Yp/DH/Neff. Do not loosen the xfail threshold. |
| C4 | F-ODE-ADJ3-004, classA false success | A | Status/event first, fixed-temperature clock if transverse, class-A Hamiltonian/momentum residual and raw abundance admission; physics cannot replace outcome precedence. |
| C5 | F-ODE-ADJ3-005, tilted false success | A | Use rapidity `v=tanh(u)` only if its near-light-boundary conditioning passes, plus expected event and momentum constraint; still requires software status checks and a real T_end. |
| C6 | F-ODE-ADJ3-006, transport convergence false green | A | Scale near-zero anisotropic-stress differences by `atol*rho_nu+rtol*max(|pi|)` and include induced shear/observables, LRS/free-streaming limits and explicit no-convergence; never default to the first candidate. |
| C7 | F-ODE-ADJ3-007, NumPy Rodas fabricated/broken events | S/A | A monotone T clock can remove the physical cold root, but the general initial-zero, nonfinite, direction, refinement and typed-terminal defects require direct solver repair. |
| H1 | F-ODE-ADJ3-008, JAX Rodas false success | S | Freeze/disable oracle output until invalid inputs, event semantics, nonfinite state and status precedence are repaired; no physics transformation licenses success. |
| H2 | F-ODE-ADJ3-009, clipped occupations and floored H | P/A | R1: strict coordinates and positive physical components; fail on invalid raw rho/H. Rust logit is already implemented; do not port another clamp. |
| H3 | F-ODE-ADJ3-010, wrappers consume failed partial state | S | Check terminal cause before final RHS/observables; preserve both last accepted and raw failing evidence. Physical residuals are downstream only. |
| H4 | F-ODE-ADJ3-011, incomplete full-coupled admission | A | R1+R2: ordered temperatures, certified target, finite raw state, domains, first law, Type-I constraint, abundance normalization and counters before observables. |
| H5 | F-ODE-ADJ3-012, NumPy config/telemetry | S/A | Reject invalid tolerances/state/step bounds; expose real counters/budget. R3 can supply Jacobian/Jv and component scales but cannot repair coercion or fabricated nfev. |
| H6 | F-ODE-ADJ3-013, classifier precedence/counts | S | Failure/status precedes target; report stored points separately from attempts and include endpoint residual/time/counters/raw finiteness. |
| H7 | F-ODE-ADJ3-014, Python Jacobian/resource drift | P/A | Apply R3 only on the active Rust endpoint-consumed path unless a Python oracle test is required; strict wall/attempt caps remain software. No segment-only promotion. |
| H8 | F-ODE-ADJ3-015, Rust BDF taxonomy collapse | S | Typed initialise/nonlinear/error-test/linear/h-floor/event/domain categories; physics residual can enrich, not replace, the reason. |
| H9 | F-ODE-ADJ3-016, BDF max_attempts not strict | S | Enforce or honestly rename one total budget across dependency-internal retries; no physics remedy. |
| H10 | F-ODE-ADJ3-017, work counters undercount | S | Count at collision/network kernel boundaries and actual LU/J/Jv/dfdt/refinement sites; physical full-RHS-equivalent accounting assists cost interpretation only. |
| H11 | F-ODE-ADJ3-018, raw failure is not failing stage | S | Preserve failing t/stage/state/reason separately from last accepted state without projection; evaluate physical residual only if meaningful. |
| H12 | F-ODE-ADJ3-019, N48 derivative evidence absent | P | R3 production N48 analytic/block T, N and collision JVPs; centered scale ladder, detailed-balance/invariant-null and independent high-precision tiny-grid checks. |
| H13 | F-ODE-ADJ3-020, test precedence and coverage gaps | A | Fix boolean precedence and inject every known failure; add R6 metamorphic physics tests. Shared-tableau parity is not independence. |
| H14 | F-ODE-ADJ3-021, no fresh exact N48 endpoint/environment identity | E | Exact-environment rerun and independent endpoint evidence are indispensable; physics reasoning cannot substitute. |
| H15 | gradient_bridge `_raw_solve` drops status | S | Return/require admitted terminal certificate for baseline and every perturbation before any gradient; failed solves have no gradient authority. |
| M1 | F-ODE-ADJ3-022, directionless sign-product underflow | S | Inclusive sign comparisons without multiplication; T clock only bypasses the physical endpoint instance. |
| M2 | F-ODE-ADJ3-023, uncertified Rust refiners | A | Retain two finite evaluated bracket endpoints and require width plus unit-bearing T residual and transversality; exhaustion is failure. |
| M3 | F-ODE-ADJ3-024, BDF hmax/tstop history semantics | S/A | Validate chunked vs unchunked history/order and hmin semantics; rate/H timescales may set a prospective h ceiling but cannot validate solver history. |
| M4 | F-ODE-ADJ3-025, Rodas immediate hard fail | A | Retry only finite, domain-admissible, plausibly step-size-dependent LU/nonlinear failures with fresh Jacobian under strict budget; nonfinite/domain failures remain fatal. |
| M5 | F-ODE-ADJ3-026, subnormal relative FD | P | Replace T differences with analytic derivative or dimensionless `ln(T/MeV)` perturbation/domain scale; energy coordinates are conditional. |
| M6 | F-ODE-ADJ3-027, Python input/failure drift | A | R1 plus rapidity/positive coordinates where validated; uniform finite input and raw-failure contract is still software. |
| M7 | F-ODE-ADJ3-028, Rust reason-branch gaps | S/A | Direct reason assertions plus R6 production-shaped states; absence of coverage is not cured by a conservation check. |
| M8 | JAX adjoint tape/incomplete force-accept | S/A | Record the actual accepted signed tape and terminal cause; reject incomplete/nonfinite/error>1 replay. Event sensitivity requires `dt*/dtheta=-(g_y S+g_theta)/(g_t+g_y f)` with a nonzero denominator. Frozen oracle only. |
| L1 | F-ODE-ADJ3-029, Diffrax API/orphan surface | S | Unify error type or retire unused surfaces; no physics content and no promotion implication. |
| L2 | F-ODE-ADJ3-030, config provenance/stale module physics | S | Correct the stale catalogue text and consolidate config identity into an existing result/receipt; anti-drift forbids a new manifest/gate. |

Affected callers are covered by the same roots: eight diagnostic solve scripts that miss events map to C4/C5/C7/H3; the stage-A/B partial-state consumer maps to H3/H4; retired JAX classA/B/tilted and JAX full-Type-I partial-observable lanes map to H1/H3/H15/M8; JAX batch inactive-lane work/domain handling maps to H1/H5/H10. They do not create distinct scientific roots.

### Candidate priority and negative results

1. Candidate PR-A, correctness-first: repair the existing NumPy/canonical outcome path and consolidate R1 admission; remove clipping/flooring from consumed states, certify the temperature surface or use R2 only after a monotonicity bound. Expected movement: C4-C7, H2-H6, H11, M1-M2/M6; no endpoint-speed or D-071 claim.
2. Candidate PR-B, Rust endpoint-consumed operator slice: R3 scale-admissible analytic/block JVP, analytic T/N derivatives and passive-time separation, followed by a measured Schur/Krylov choice only if the N48 pattern supports it. Expected movement: C2, H7, H10, H12, M5 and possibly the measured wall. Falsify on <10% same-case activation/cold call or wall movement, loss of raw parity/invariants, or hidden dense fallback.
3. Candidate PR-C is conditional, not authorized until a zero/new-production-code discriminator passes: derive channel discrete detailed balance/nullspaces and the projected collision spectrum across the stalled band; if a fast gap exists, implement one Rust R5 high-temperature vertical slice with R4 observable bounds. This is the only candidate aimed at C1's asymptotic wall and also supports C3. Falsify on absent gap, handoff error, non-well-prepared layer, invariant/first-law failure, or sealed prefix/endpoint budget miss.

Rejected as present solutions: fixed momentum-tail algebraic slaving (measured no gap and poor cost); generic solver swap, tolerance loosening, hardware or longer budget; post-step projection/clipping/flooring; energy-only spectral replacement across weak freezeout; assuming neutrino number/energy conservation for electron-pair channels; a full network/Patankar rewrite when Rust log-X/stoichiometry already exists; temperature clock without a transversality bound; logit alone as a far-tail conditioning cure; JAX/Diffrax forward promotion; new wrapper/gate/telemetry surfaces; static states, same-code parity or segment timing as endpoint evidence.

Independent blind-review inputs: `A-PM-TRANSPORT.json` (5 findings, status fail), `A-PM-NUMERIC.json` (7 findings, status fail), `A-PM-PHYS2.json` (10 findings, status fail). The earlier `A-PM-PHYS.json` stopped on an absent sealed path and is treated only as an assignment defect, not scientific evidence.

## Independent adjudication and final verdict

- E013 | 2026-08-23 | The blind transport, numerical and physics reviews all returned `fail`: every remedy remains either already-implemented bounded structure, DERIVED but unimplemented, PROPOSED, or software-only; none supplies measured endpoint movement or a D-071 reopen.
- E014 | 2026-08-23 | Caller correction: `src/rabbit/transport/stageAB_analysis.py` is fail-loud. The partial-state consumer is `scripts/run_stageAB_transport_probe_pr1.py`. The eight failure-only/no-certified-event script surfaces are `probe_tier2_bridge_only.py`, `trace_scipy_phase1_characteristic_general.py`, `trace_scipy_typeI_phase1_external.py`, `trace_species_rays_phase1.py`, `trace_tier2_branch_point.py`, `run_characteristic_ablation.py`, `run_nonlinear_ablation.py`, and `stress_test_characteristic.py`; these map to existing C4/C5/C7/H3 roots rather than new causal roots.
- E015 | 2026-08-23 | Independent adjudicator `A-PM-ADJ` returned overall `fail` with mechanical-root completeness `pass`, physics/math audit `pass` under binding assumptions, dead-end rejection `pass`, and bounded three-candidate ranking `pass`. It confirmed the 32-root normalization and added the existing H13/M8 counterexample that `gradient_check` can replace a nonzero small-gradient disagreement with zero relative error.
- E016 | 2026-08-23 | Acceptance cells PM-A1 through PM-A6 are complete for the research deliverable: 32/32 normalized roots and affected callers mapped; assumptions/units/signs/limits recorded; primary sources and local touchpoints bound; at most three candidates ranked; independent verdict non-inconclusive. This completes the research map, not any implementation, runtime validation, endpoint, gate, or scientific promotion.

Final verdict: `RESEARCH COMPLETE / SCIENTIFIC AND RUNTIME BLOCKERS RETAINED`. Candidate PR-A is a correctness repair only; PR-B is an endpoint-consumed derivative/performance experiment with a >=10% same-case movement falsifier; PR-C is only a zero/new-production-code discriminator until exact channel nullspaces, projected spectrum, reaction-tail bounds and overlap pass. `G-F10-INDEPENDENT-FLRW=FAIL/CLOSED_ON_CURRENT_MEASUREMENT` and `G-HARNESS-INTEGRITY=FAIL`. QKE, public production, publication authority, JAX forward promotion and D-071 reopening remain FORBIDDEN claims.
