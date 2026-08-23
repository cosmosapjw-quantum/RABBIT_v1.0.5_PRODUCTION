# EVIDENCE_LEDGER

Phase 2 draft, pending independent Phase 3 claim-source audit. Claim audit status is deliberately not assigned yet.

| Evidence ID | Claim ID | Source | Exact support | Phase-2 role | Reliability | Assumptions / boundary | Conflict or gap |
|---|---|---|---|---|---|---|---|
| E-001 | C-GATE | `docs/audit/BD622_D071_trajectory_closure_2026-08-04.md:96-100` | Retained order-60 run evaluated no predicate; projected 32,704,637 more RHS calls, 144,574,895 s, about 8305x miss | limits | high/local authority | Bound HEAD and existing gate contract | No new trajectory in this cycle |
| E-002 | C-SEEDS | prior `A-PM-ADJ.json` and `/tmp/rabbit-ode-physics-mitigation-research-20260823.md` | Defines R1-R6 and retains the gate ceiling | contextual | high/previous adjudication | Seeds are inputs, not accepted conclusions | Must be re-audited here |
| E-101 | C-R1 | `native/rabbit_cpu/src/isotropic_boltzmann.rs:352-386,1013-1034` | Strict-open occupation/logit conversion and raw error path exist | supports | high/current source | `0<f<1`; no clipping as admission | Coordinate saturation can still be ill-conditioned |
| E-102 | C-R1 | current audit false-success findings | Status/event precedence and suppression of post-failure observables are required | supports | high/local audit | Software outcome contract | No physics paper can supply repository failure taxonomy |
| E-201 | C-R2 | `native/rabbit_cpu/src/flrw.rs:289-329,489-549`; `isotropic_boltzmann.rs:192-209,428-504` | Finite-mass EOS derivatives and energy-transfer temperature equations are implemented locally | supports | high/current source | Flat-FLRW seed conventions | Nonzero-shear continuity not derived here |
| E-202 | C-R2 | Pitrou et al., PRIMAT, [arXiv:1801.08023v3](https://arxiv.org/abs/1801.08023) | FLRW thermodynamics, plasma temperature, entropy/energy evolution, weak freeze-out and baryon normalization | supports mechanism | high/primary | Paper's stated FLRW model | Does not prove current full-path clock monotonicity |
| E-203 | C-R2 | current local evidence | No sealed positive lower bound for `alpha=-d ln(T_gamma)/dN` and no EOS inversion certificate on the relevant Type-I path | missing/limits | high/local absence | Clock requires a monotone transverse branch | General event semantics remain software-only |
| E-301 | C-R3 | Froustey, Pitrou & Volpe, [arXiv:2008.01074v2](https://arxiv.org/abs/2008.01074) | Structured differentiation reduces collision-Jacobian complexity from nominal O(N^4) finite differences to O(N^3) and reports about N/5 comparison | supports mechanism | high/primary | Reuse only classical collision derivative argument; QKE/flavour scope excluded | Does not imply 8305x endpoint movement |
| E-302 | C-R3 | `isotropic_boltzmann.rs:563-658`; `thermal_bbn.rs:247-298,326-371` | Analytic occupation/network blocks and finite-difference thermal blocks exist | supports | high/current source | Exact current configuration | Full scale-admissible Jv and channel nullspaces missing |
| E-303 | C-R3 | D-071 and V2 closure | Removing Jacobian construction still misses the wall by orders of magnitude | contradicts strong endpoint claim | high/local authority | Same retained measurement | Derivative work can be a local experiment only |
| E-401 | C-R4 | Grohs et al., [arXiv:1512.02205](https://arxiv.org/abs/1512.02205) | Multi-energy neutrino transport couples nonlinearly to thermodynamics, n/p, yields and Neff | supports joint-observable requirement | high/primary | Paper's coupled BBN model | Does not prove a rigorous RABBIT reaction-tail enclosure |
| E-402 | C-R4 | current local evidence | Positive quadrature nodes/weights and moment tests exist | contextual | high/current source | Current spectral grid | No weak-rate, reaction-tail or abundance propagation certificate |
| E-403 | C-R4 | current evidence gap | No source supplies a simultaneous enclosure for energy transfer, weak rates, Yp, D/H and Neff for current finite-mass kernels | missing | high | Exact present operator required | Energy-only agreement is insufficient |
| E-501 | C-R5 | Boscarino, Pareschi & Russo, [arXiv:1009.2757](https://arxiv.org/abs/1009.2757) | AP relaxation conditions include exact conserved rows, equilibrium manifold consistency, stiff-accuracy and initial-layer caveats | supports generic method conditions | high/primary | Generic relaxation system, not current Fermi operator | Only analogical until current discrete identities/gap exist |
| E-502 | C-R5 | `docs/audit/BD622_V2_option3_closed_and_protocol_2026-08-04.md` | Fixed momentum-tail slaving has no usable gap, only three gapped dimensions, loses separation during decoupling, and at best gives about 1.25x | contradicts relabeled fixed-tail route | high/local authority | Applies to tested fixed-cut option | Does not logically rule out every invariant-aware AP method |
| E-503 | C-R5 | current evidence gap | Exact current per-channel `Q`, grid-stable projected spectrum, overlap/handoff and response enclosure absent | missing | high | Must precede reduction code | R5 cannot yet claim applicability |
| E-601 | C-R6 | `isotropic_boltzmann.rs:192-209,1013-1034`; `flrw.rs:721-746,892-927` | Collisionless, strict-domain and first-law/temperature limit tests exist locally | supports bounded falsifiers | high/current source | Each invariant only in its exact model/channel | Static tests are not trajectory evidence |
| E-602 | C-R6 | SciPy 1.17.0 [`solve_ivp`](https://docs.scipy.org/doc/scipy-1.17.0/reference/generated/scipy.integrate.solve_ivp.html) | Events are sign-change based and may miss multiple crossings; status/counters are explicit API outputs | limits event claims | high/official docs | SciPy 1.17.0 | Does not certify RABBIT event refinement or physical admission |

## Source lineage

1. `A-ODE-ADJ3` supplies the exhaustive issue/gate authority.
2. The prior physics-mitigation adjudication supplies R1-R6 as seeds and carries the D-071/V2 negative results.
3. `A-PSL-EVIDENCE` re-maps those seeds to current source and primary/official literature without inheriting a promotion verdict.

## Conflict matrix

| Seed | Conflict | Phase-2 resolution |
|---|---|---|
| R1 | Physics coordinates cannot determine solver outcome taxonomy | Keep representational part physical; label precedence/taxonomy software-only |
| R2 | Temperature clock requires a global monotonic/transverse certificate | Conditional only |
| R3 | Derivative speedup is orders short of D-071 | No endpoint inference |
| R4 | Energy-only accuracy misses weak/yield sensitivity | Require joint observable and propagated-error bounds |
| R5 | V2 rejects fixed momentum-tail separation | Exact-invariant/projected-gap discriminator before any reduction |
| R6 | Static limits are not trajectory evidence | Falsifier role only |

## Missing evidence

- Exact per-channel discrete detailed-balance identities and left nullspaces for current bytes.
- Grid-stable projected non-null collision spectrum throughout the stalled band.
- Reaction-tail, weak-rate, and abundance propagated enclosure.
- Sealed `alpha>0` lower bound and EOS inversion certificate on the relevant path.
- Any full-prefix/endpoint evidence or measured active-path blocker movement.

## Phase 3 independent claim-source audit

Result: `.agent-harness/runs/run-20260823-ode-physmath-loop/results/A-PSL-CLAIM-AUDIT.json`, SHA-256 `78a97db76bf816a67da2761a135377ecd85b1477f5198290d7e5743aeae97865`.

Counts: `SUPPORTED 20`, `PARTIALLY_SUPPORTED 8`, `CONTESTED 1`, `UNSUPPORTED 9`, `MISATTRIBUTED 4`, `INFERENCE_ONLY 4` (46 total).

| Audit ID | Seed | Evidence | Status | Phase-4 action | Audited claim |
|---|---|---|---|---|---|
| CA-00-001 | CONTROL | E-001 | SUPPORTED | KEEP_WITH_SCOPE | 32,704,637 further evaluations, 144,574,895 s, about 8305x are a retained-window extrapolation, not a completion date |
| CA-00-002 | CONTROL | E-001 | PARTIALLY_SUPPORTED | REFORMULATE_WITH_CORRECT_CITATION | The run evaluated no registered scientific predicate; distinguish this from every possible diagnostic quantity |
| CA-00-003 | CONTROL | E-002 | SUPPORTED | KEEP_CONTEXT_ONLY | R1-R6 are seeds and the gate ceiling is unchanged |
| CA-R1-001 | R1 | E-101 | SUPPORTED | KEEP_WITH_SCOPE | Rust has stable logit mapping and rejects finite-precision `f=0/1` raw |
| CA-R1-002 | R1 | E-101 | PARTIALLY_SUPPORTED | LOCAL_REPRESENTATION_ONLY | This local coordinate path is not a complete admission remedy for every caller |
| CA-R1-003 | R1 | E-102 | SUPPORTED | KEEP_WITH_EXACT_CITATION | Current audit contains false-success, missing-event and post-failure-consumption defects |
| CA-R1-004 | R1 | E-102 | INFERENCE_ONLY | SOFTWARE_OBLIGATION_ONLY | Outcome precedence does not follow from transport physics |
| CA-R1-005 | R1 | prior PRIMAT citation | MISATTRIBUTED | REMOVE_CITATION | Baryon normalization does not support occupation coordinates or solver admission |
| CA-R2-001 | R2 | E-201 | SUPPORTED | KEEP_WITH_SCOPE | Flat-FLRW finite-mass EOS derivatives and collision energy-transfer temperature equations exist |
| CA-R2-002 | R2 | E-202 | SUPPORTED | KEEP_WITH_SCOPE | PRIMAT supports its stated FLRW thermodynamic and inversion relations |
| CA-R2-003 | R2 | E-202 | PARTIALLY_SUPPORTED | CONDITIONAL_ONLY | PRIMAT does not validate the current full-path clock |
| CA-R2-004 | R2 | E-202 | UNSUPPORTED | REMOVE_PREMISE | No primary-source proof of global `alpha>0`, current EOS inverse, or Type-I shear extension |
| CA-R2-005 | R2 | E-203 | PARTIALLY_SUPPORTED | KEEP_SCOPED_GAP | Relevant transversality/inversion certificate is absent from the sealed evidence |
| CA-R2-006 | R2 | SciPy official docs | SUPPORTED | SOFTWARE_LIMIT_ONLY | Per-step sign-change detection and missed multiple crossings are documented |
| CA-R2-007 | R2 | SciPy official docs | MISATTRIBUTED | REMOVE_PHYSICS_CITATION | SciPy docs do not prove cosmological monotonicity |
| CA-R3-001 | R3 | E-301 | SUPPORTED | KEEP_WITH_SOURCE_SCOPE | Source-specific O(N^4) versus O(N^3), with N/5 in that implementation |
| CA-R3-002 | R3 | E-301 | PARTIALLY_SUPPORTED | MECHANISM_ANALOGY_ONLY | Classical structured differentiation motivates investigation, not transfer of a speed factor |
| CA-R3-003 | R3 | E-301 | UNSUPPORTED | REMOVE_PREMISE | N/5 is not a RABBIT active-path/endpoint prediction |
| CA-R3-004 | R3 | E-302 | SUPPORTED | KEEP_WITH_SCOPE | Current Rust has analytic occupation/network blocks and FD thermal columns |
| CA-R3-005 | R3 | E-302 | UNSUPPORTED | REMOVE_PREMISE | Full scaled Jv and exact channel nullspaces do not already exist |
| CA-R3-006 | R3 | SciPy official docs | SUPPORTED | INTERFACE_CONTEXT_ONLY | Dense/sparse Jacobian interfaces exist |
| CA-R3-007 | R3 | SciPy official docs | MISATTRIBUTED | REMOVE_CITATION | Interface docs are not a matrix-free Jv or nullspace certificate |
| CA-R3-008 | R3 | E-303 | CONTESTED | REMOVE_PREMISE | Structured Jacobian work alone is not admitted as a D-071 endpoint remedy |
| CA-R3-009 | R3 | prior affinity claim | UNSUPPORTED | DERIVE_FIRST | Current detailed-balance affinity/nullspaces are not established |
| CA-R4-001 | R4 | E-401 | SUPPORTED | KEEP_WITH_SCOPE | Multi-energy transport feeds thermodynamics, phasing, n/p, abundances and Neff |
| CA-R4-002 | R4 | PRIMAT | SUPPORTED | KEEP_WITH_SCOPE | Energy density alone does not capture all spectral weak-rate effects |
| CA-R4-003 | R4 | E-401 | INFERENCE_ONLY | LABELED_DESIGN_REQUIREMENT | Joint energy/weak-rate/Yp/DH/Neff control is a design inference, not a theorem |
| CA-R4-004 | R4 | E-402 | SUPPORTED | KEEP_WITH_SCOPE | Positive selected quadrature nodes/weights and FD moment tests exist |
| CA-R4-005 | R4 | E-402 | UNSUPPORTED | REMOVE_PREMISE | These tests do not imply nonlinear reaction-tail/abundance enclosure |
| CA-R4-006 | R4 | E-403 | PARTIALLY_SUPPORTED | KEEP_SCOPED_GAP | No such RABBIT-specific simultaneous enclosure appears in the audited set |
| CA-R4-007 | R4 | E-403 | UNSUPPORTED | REMOVE_PREMISE | A current rigorous enclosure does not exist |
| CA-R5-001 | R5 | E-501 | SUPPORTED | KEEP_WITH_SOURCE_SCOPE | Generic AP source requires exact conserved rows, equilibrium manifold and handles initial-layer/stiff-accuracy caveats |
| CA-R5-002 | R5 | E-501 | PARTIALLY_SUPPORTED | ANALOGICAL_CHECKLIST_ONLY | Conditions are useful to audit, not proof of applicability |
| CA-R5-003 | R5 | E-501 | UNSUPPORTED | REMOVE_PREMISE | Current operator has not met those conditions |
| CA-R5-004 | R5 | E-502 | SUPPORTED | KEEP_NEGATIVE_RESULT | Fixed-cut slaving lacks usable separation and gives at most about 1.25x in its model |
| CA-R5-005 | R5 | E-502 | UNSUPPORTED | REMOVE_PREMISE | V2 does not prove every invariant-aware AP method impossible |
| CA-R5-006 | R5 | E-502 | INFERENCE_ONLY | NO_APPLICABILITY_PREMISE | Logical possibility is not positive evidence |
| CA-R5-007 | R5 | E-503 | PARTIALLY_SUPPORTED | KEEP_SCOPED_GAP | Exact `Q`, projected spectrum, handoff and response enclosure are absent from the audited set |
| CA-R6-001 | R6 | E-601 | SUPPORTED | BOUNDED_INGREDIENTS_ONLY | Local collisionless, raw-domain, first-law and EOS-limit tests exist |
| CA-R6-002 | R6 | E-601 | MISATTRIBUTED | CORRECT_TO_LINES_231_237 | `df(q)/dN=0` was cited to the wrong source range |
| CA-R6-003 | R6 | current network | SUPPORTED | KEEP_WITH_SCOPE | Reaction-wise baryon/charge and closed-RHS conservation checks exist |
| CA-R6-004 | R6 | PRIMAT | SUPPORTED | PHYSICAL_ANCHOR_ONLY | `sum X_i=1` is independently anchored |
| CA-R6-005 | R6 | E-602 | SUPPORTED | API_BOUNDARY_ONLY | Event sign-change limitation is official |
| CA-R6-006 | R6 | E-602 | SUPPORTED | API_BOUNDARY_ONLY | Explicit work/status return fields are official |
| CA-R6-007 | R6 | E-601+E-602 | UNSUPPORTED | REMOVE_PREMISE | Static limits/API semantics do not certify trajectory or endpoint |
| CA-R6-008 | R6 | proposed mutants | INFERENCE_ONLY | NOT_FACTUAL | Unexecuted falsifiers are not proven to catch the retained defects |

Sanitized premises entering Phase 4:

- R1: local strict-domain representation plus a separate software admission problem; remove the PRIMAT citation.
- R2: conditional energy/clock hypothesis only; no `alpha>0`, global inversion, or shear premise.
- R3: structured-derivative mechanism analogy plus current block inventory; no inherited speed factor, affinity/nullspace fact, or D-071 remedy premise.
- R4: coupled-observable sensitivity and a labeled design requirement; no current enclosure premise.
- R5: generic checklist plus fixed-cut negative result; current applicability and universal impossibility are both removed.
- R6: bounded ingredients with corrected `df(q)/dN=0` citation; no trajectory or mutant-efficacy premise.
