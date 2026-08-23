# HYPOTHESIS_GRAPH

Six families exhaust the ten normalized seeds without treating numerical carriers as physical closure. `SERIOUS` means worthy of the next implementation-design gate, not implemented or trajectory-validated.

## H-001 — Covariant constrained phase-space formulation

CORE_CLAIM: For the present spatially homogeneous, scalar, unpolarized scope, a single tagged photon four-momentum, invariant mass-shell measure, complete frequency-plus-angular Liouville operator, and family-specific constrained Bianchi background remove the frame/background formulation blockers without giving the chosen numerical storage tetrad preferred physical status.

MINIMAL_ASSUMPTIONS: Spatial homogeneity; scalar unpolarized distribution `f`; subluminal tetrad boost; declared `(-,+,+,+)` convention and numerical storage tetrad; hydrogen spatial-axis transport and collision-clock conversion declared; metric/connection and matter model known; family-specific algebraic constraint Jacobian regular in the active chart.

MECHANISM: Construct `p^a` once, project it into every observer tetrad, derive frequency/direction characteristics and collision-time conversion from the same covariant equation, and assemble conservative number/four-momentum transport. In BII, integrate an admissible interior chart, reconstruct dense snapshots there, and compare algebraic Omega with an independently evolved matter-conservation shadow. Use positive switch surfaces for boundary charts and distinct VI_h, exceptional VI_-1/9 and IX state spaces/charts.

KNOWN_LIMITS: Pontzen-Challinor is perturbative; current executable support is BII only; the BII interior chart sends the vacuum/type-I invariant boundaries to infinity and needs a separate boundary atlas; the located IX normalization is LRS only; Hubble normalization fails at `H=0`; a conservative continuum identity does not automatically survive quadrature/JVP discretization.

DISTINCTIVE_PREDICTIONS: Two code paths fed the same `p^a` and clock convention produce identical frame observables/actions; finite tilt generates a required angular action; beta->0 and shear/curvature->0 recover FLRW; number and four-momentum balance close independently; BII shadow/algebraic Omega and dense queries agree under refinement without clipping.

COMPETING_HYPOTHESES: H-006 diagnostic-only null; frame-local patching with no shared four-momentum; reuse of the BII chart in all Bianchi families.

DECISIVE_TEST: A finite-tilt manufactured geodesic and distribution with analytic Lorentz moments, nonzero angular drift, plus BII constraint/event refinement. Pass only if both frame paths, conservative moments, JVP, limiting cases, and independent Omega shadow converge. Kill on frame-dependent physical moments, absent angular mode, negative admitted Omega, or nonconvergent event/constraint error. Unsupported families return `REGIME_UNIMPLEMENTED`, not extrapolation.

SUPPORTING_EVIDENCE: E-L01, E-L02, E-L11, E-L12, E-P01--E-P08, E-M03, E-X01, E-X02.

THREATENING_EVIDENCE: Perturbative hierarchy scope; no present angular action; no generic/tilted IX source or provider.

STATUS: SERIOUS, restricted to frame completion plus current BII; other families retained as fail-closed sub-blockers.

## H-002 — Microreversible, nonoverlapping kinetic residual with causal support evolution

CORE_CLAIM: A reaction-family-aware reciprocal Bose graph, native-exterior/COM-interior weak split with one physical interface flux, measure-aware ALE only on identifiable finite cells, and accepted retarded history can define one nonduplicated radiation-matter residual.

MINIMAL_ASSUMPTIONS: Positive mode measures; event-level microreversibility where physically applicable; reaction stoichiometry and material response known; disjoint support registry; identifiable finite-volume cell faces for ALE; causal query delays and accepted endpoint fields known.

MECHANISM: Derive collision left nulls from event incidence, equilibrium nulls from connected components, and free-energy sign from reciprocal conductances. Keep open emission/absorption and material four-force separately owned. Couple native point sources and COM volumes by one weak total physical flux. Apply geometric conservation and convex-feasible remap only to finite cells. Treat accepted history as a functional state with atomic commit/restart.

KNOWN_LIMITS: Current aggregate atom four-force and some interface ledgers are defined as negatives and are not independent checks; executable E1C replacement is absent; mortar theory is from another PDE class; a scalar remap cannot always preserve positivity, number and energy locally; point spikes have no recoverable width; causal history does not turn the global problem into a finite local DAE.

DISTINCTIVE_PREDICTIONS: Closed elastic components preserve their incidence moments and have one activity parameter per connected component; the frozen-bath closed reciprocal collision contribution makes bath-relative free energy nonincreasing; open reactions/evolving bath/ALE/material work appear as separately signed balance terms; the native+COM constant-test interface term cancels exactly; pure grid motion preserves constant `f`; infeasible remaps are detected; rejected trials leave history byte-identical.

COMPETING_HYPOTHESES: Independent native and COM evolution plus ledger reconciliation; fabricated spike cells/global interpolation; generic conservative remap; all-in-one photon conservation claim.

DECISIVE_TEST: A manufactured mixed reaction graph plus a one-interface transport/recoil problem, exercised on fixed and moving grids and through reject/event/restart. Assemble photon and material exchanges independently. Pass only if reaction-specific invariants, equilibrium/nullity, free-energy sign, single ownership, Schur parity, GCL, remap feasibility, and history equivalence all hold under refinement. Kill on any double/unowned edge, tautological conservation, fabricated point mass, false-feasible remap, or rejected-history mutation.

SUPPORTING_EVIDENCE: E-L03--E-L07, E-P09--E-P16, E-M01--E-M02.

THREATENING_EVIDENCE: Current implementation is witness-only at E1C; source geometry and full material equations are incomplete.

STATUS: SERIOUS as a specification; not an implementation or trajectory cure.

## H-003 — Local DAE and continuous-event enabling layer

CORE_CLAIM: Once H-001/H-002 identify the physical residual, a scaled local index-one `F(t,y,ydot)=0` formulation, consistent reinitialization, continuous event roots, and accepted fine step can make integration faithful.

MINIMAL_ASSUMPTIONS: Nonsingular scaled algebraic Jacobian after physical null removal; continuous event functions; defined branch transition; history kept separate; component units and tolerances declared.

MECHANISM: Row/component equilibration, componentwise backward error, consistent initial/post-event algebraic solves, two-half-state acceptance, and state-dependent earliest-root restart.

KNOWN_LIMITS: Cannot identify missing point mass, interface ownership, frame terms, or a physical root; multiple/grazing roots require certification; local error control is not a global trajectory bound.

DISTINCTIVE_PREDICTIONS: Algebraic residual and constraints remain scaled-small; nonautonomous step-doubling converges at the expected order; event time converges under step refinement; split/restart equals uninterrupted accepted states/history.

COMPETING_HYPOTHESES: Generic ODE solve with global residual norm or supplied timestamps.

DECISIVE_TEST: Run analytic index-one DAE/event/history manufactured cases including simultaneous/grazing and nonfinite half-step failures, with independent endpoint/event oracles.

SUPPORTING_EVIDENCE: E-L06--E-L08, E-P14--E-P16.

THREATENING_EVIDENCE: Physical residual is incomplete; current adaptive interfaces fail the necessary semantics.

STATUS: HOLD as an enabling layer downstream of H-001/H-002.

## H-004 — Physics-mode preconditioning, PTC and asymptotic preservation

CORE_CLAIM: Splitting true collision invariants from relaxing modes and using a physics-derived Schur/preconditioner may make the completed residual tractable and AP in a derived stiff limit.

MINIMAL_ASSUMPTIONS: Completed physical operator; correct nullspace; independently derived reduced limit; physical mass matrix; spectral/performance benchmark protocol.

MECHANISM: Micro-macro projection by event-incidence moments, equilibrium-constrained Schur complement, PTC globalization of the same transient, and fixed-step epsilon-to-zero comparison.

KNOWN_LIMITS: Literature examples are different linear/thermal models; PTC can select another root; L-stability and a large-step solution are not AP; performance is hardware/operator specific.

DISTINCTIVE_PREDICTIONS: Iteration counts remain controlled with refinement/stiffness and fixed-step solutions converge to the independently discretized reduced model as epsilon->0.

COMPETING_HYPOTHESES: Generic diagonal preconditioning, solver replacement, or tolerance tightening.

DECISIVE_TEST: On the completed operator, scale collision rates by `1/epsilon`, hold macro step fixed, compare to an invariant-constrained reduced model, then repeat `h,h/2,h/4` and measure work.

SUPPORTING_EVIDENCE: E-L08, E-P17--E-P19.

THREATENING_EVIDENCE: No current full residual/reduced model; all transfer claims are conditional or inference-only.

STATUS: HOLD; premature before correctness closure.

## H-005 — Complete domain-separated identity and claim firewall

CORE_CLAIM: A canonical complete commitment must domain-separate and cover both the physical operator and numerical policy to prevent substitute-problem restart/acceptance. Two exposed digests are recommended for auditability but are not mathematically necessary; one complete domain-separated aggregate can provide the same mutation detection.

MINIMAL_ASSUMPTIONS: Canonical serialization and complete dependency/operator/history policy manifests.

MECHANISM: Bind every accepted endpoint/restart either to two complete domain-separated digests or to one canonical aggregate over explicitly tagged physical and numerical domains; require exact local bytes and tolerance-bounded cross-platform physical equivalence.

KNOWN_LIMITS: Provenance cannot repair physics, convergence, or information loss.

DISTINCTIVE_PREDICTIONS: Any operator/policy/history mutation invalidates exact replay; platform-only formatting differences do not imply physical disagreement.

COMPETING_HYPOTHESES: Complete tagged aggregate commitment (security-equivalent competitor), incomplete aggregate/receipt-only admission, or dual digests with omitted fields.

DECISIVE_TEST: Mutate, truncate, reorder and domain-swap every manifest field independently across complete-dual, complete-tagged-aggregate and incomplete/receipt-only designs. Both complete designs must fail closed; only audit localization/operational clarity may distinguish dual from aggregate. Compare identical physical observables across supported platforms separately.

SUPPORTING_EVIDENCE: E-L09.

THREATENING_EVIDENCE: Current digests are externally supplied and incomplete.

STATUS: RETAINED NARROWED GATE, explicitly not a physics remedy; exact dual-identity necessity REJECTED.

## H-006 — Diagnostic/harness-only null or solver-only cure

CORE_CLAIM: The blockers are report artifacts or can be removed universally by tolerance tightening, clipping, precision, or solver replacement without changing the physical operator.

MINIMAL_ASSUMPTIONS: Current residual already contains all physical states, terms, ownership and information.

MECHANISM: Re-run or tune the existing implementation.

KNOWN_LIMITS: Contradicted by source-level frame reinterpretation, omitted angular action, nonidentified point mass, witness-only E1C and endpoint/event interface gaps.

DISTINCTIVE_PREDICTIONS: Independent manufactured physics oracles agree with the current operator and only numerical residuals change with solver settings.

COMPETING_HYPOTHESES: H-001/H-002.

DECISIVE_TEST: Inventory every physical term/owner and compare against analytic frame, weak-interface and history oracles before tuning.

SUPPORTING_EVIDENCE: None for the strong form; higher precision remains valid only for a separately proven local cancellation defect.

THREATENING_EVIDENCE: E-L01, E-L04, E-L07, E-L10.

STATUS: REJECTED in strong form; retained only as a negative control.
