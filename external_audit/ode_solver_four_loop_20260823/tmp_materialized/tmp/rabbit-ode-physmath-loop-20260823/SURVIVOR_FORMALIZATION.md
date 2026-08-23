# Phase 9 - survivor-only formalization

Only H0/V0, H1/V1 and H6/V6 are formalized because only they received Phase-8 `PROMOTE`. This document specifies verification objects, not implementation or validated results.

## Common authority and notation

- Bound source: `diagnosis_report@78f5b091bd8de99bd8d0b1eddd8a0b7978f5da2b`.
- Natural units; `N=ln(a)`; `q=ap`; `Q>0` into neutrinos; flat-FLRW identities are not extended to shear.
- `PROMOTE` = admit to the named prospective V-test only.
- Raw failing state `y_fail` and last accepted state `y_acc` are distinct. Neither may be clipped, floored, projected or silently interchanged.

## S0 - V0 evidence identity control

SUPPORTED: D-071's retained instrument did not complete the scientific predicate and did not retain enough solver state to identify the creep mechanism.

INFERENCE: Every mechanism test needs a common identity/work contract to avoid false attribution.

FORMAL OBJECT:

`E = (HEAD, tree, environment, state_layout, physical_case, y_acc, y_fail, terminal_cause, event_residual, work_vector, invariant_vector)`.

V0 is admissible only when every element is prospectively bound and finite/typed where applicable. `work_vector` separates accepted/rejected/attempt counts and actual RHS, collision, J, Jv, dfdt, LU and refinement work. A missing element yields `AMBIGUOUS`, never success.

LIMITATION: V0 can prevent new attribution errors; it cannot reconstruct missing D-071 history or identify a mechanism by itself.

## S1 - V1 two-layer admission discriminator

SUPPORTED: The Rust spectral path has a strict-open logit decode/raw failure, and the audited codebase contains false-success, missed-event and post-failure-consumption defects.

HYPOTHESIS: Physical-domain/invariant admission adds at least one decisive rejection beyond a typed software-only terminal certificate.

FORMAL OBJECT:

Let the software certificate be

`A_sw = success AND expected_terminal_cause AND certified_event_or_end AND complete_failure_precedence`.

Let the physical certificate be

`A_phys = finite_raw_state AND representable_domain AND channel_valid_balance AND normalization AND unit_bearing_residual`.

The admitted outcome is `A = A_sw AND A_phys`. Neither factor may repair the other, and observables are undefined when `A=false`.

For `f=sigmoid(u)` and `X=exp(z)`, error allocation is performed in physical variables through `delta_f ~= f(1-f) delta_u` and `delta_X ~= X delta_z`; a prospectively frozen interior/representability margin is required.

V1 acceptance: the frozen table kills every critical mutant, rejects no good control, exposes `y_fail` separately, performs no projection, and H1 kills at least one physical mutant admitted by `A_sw` alone.

LIMITATION: Passing V1 is correctness evidence only. If `A_sw` kills the same table, H1 is rejected as unnecessary.

## S6 - V6 independent one-sided falsification

SUPPORTED: Bounded collisionless, first-law, EOS, network-stoichiometry and event-API ingredients exist. Their scope is not universal.

HYPOTHESIS: Subject-independent oracles plus matched mutants detect retained common-mode false greens better than shared-backend parity.

FORMAL SCOPE TABLE:

| Predicate | Exact scope | Forbidden extension |
|---|---|---|
| `df(q)/dN=0` | massless homogeneous isotropic collisionless transport at fixed `q=ap` | collision-coupled or generic anisotropic transport |
| collision equilibrium null | common temperature and reaction-compatible chemical potentials for the sealed channel | arbitrary spectra or bath mismatch |
| combined first law | full flat-FLRW neutrino plus EM system with `Q_e+2Q_x` and opposite bath debit | neutrino-only energy conservation for electron channels; nonzero shear |
| baryon and nuclear charge | selected nuclear reaction suboperator | nuclear-plus-weak charge conservation; weak n<->p preserves baryon only |
| event semantics | initial zero, direction, transverse/grazing, nonfinite and refinement cases | physical invariant used as a substitute for software status |

Every mutant has an independent oracle and a matched good-path specificity control. V6 acceptance is 100% critical-mutant kill, zero good-control false positives, and documented oracle provenance distinct from the mutated path.

LIMITATION: Passing a finite mutant table is one-sided falsification evidence, not trajectory or endpoint validation.

## Formalized next-work boundary

- Allowed after separate execution authorization: prospectively bind and run V0, V1 and V6 against existing surfaces.
- Not formalized: H2, H3, H4, H5; any production change; any AP/derivative framework; any trajectory, endpoint, D-071 reopen, science or public claim.
