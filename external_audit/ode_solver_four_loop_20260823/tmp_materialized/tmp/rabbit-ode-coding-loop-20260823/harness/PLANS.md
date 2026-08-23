# PLANS.md

## Current task

TASK: run the attached coding harness as a research/design loop seeded only by the prior independent mathematics/algorithm/coding remedies.

OUTCOME: at most three source-localized candidate specifications, one bounded later-implementation recommendation at most, and a fail-closed final decision.

IN_SCOPE: contract, baseline, exact-head localization, bounded design, execution-plan design, validation design, independent review, decision, integrity closeout.

OUT_OF_SCOPE: production code/tests, worktree implementation, runtime/endpoint execution, new dependencies or gates, JAX forward development, QKE/public/promotion claims, and remedy input from the earlier physics-specific loop.

## Milestones

| ID | Milestone | Files | Validation | Status |
|---|---|---|---|---|
| P-000 | task and scientific contract | `SCIENTIFIC_CONTRACT.md`, research record | frozen-contract check | Done |
| P-001 | reproduction/acceptance baseline | prior adjudication, current gates | exact hashes and mechanical seed inventory | Done |
| P-002 | repository localization | registered result `A-CH-LOC.json` | exact file/symbol/call-path evidence | Done |
| P-003 | compare at most three solutions | this file, research record | scope, risk, falsifier, cost comparison | Done |
| P-004 | selected execution-plan design | this file | exact diff/test/rollback plan | Done |
| P-005 | isolated implementation | production source/tests | separate user authority | Cancelled |
| P-006 | software validation | `VALIDATION_MATRIX.md` | design completeness done; runtime checks NOT_RUN | Done |
| P-007 | scientific validation | `VALIDATION_MATRIX.md`, independent review | contract review found and repaired C1; C2/C3 retained REWORK | Done |
| P-008 | numerical/reproducibility validation | `VALIDATION_MATRIX.md`, independent review | unexecutable items remain NOT_RUN/CONCERN | Done |
| P-009 | independent diff review | registered result `A-CH-ADV.json` | no diff; one independent design review `MAJOR_REVISIONS` | Done |
| P-010 | promote/hold/rework/revert closeout | `RUN_STATE.md`, logs, adjudication | harness and integrity checks | Done |

Status: `Planned / In Progress / Done / Blocked / Cancelled`

## Candidate comparison

Blind design envelope: `A-CH-DESIGN.json`, SHA-256 `c8e8aba42c2acb4e166620fe03b96bdbf3c5e96ee5cd837a4955bb8a0eb80630`. It mechanically preserves all 34 IDs: 7 candidate-linked rows, 16 companion/future tasks, 7 retained blockers/decisive experiments, 3 closed/no-action rows, and 1 extrinsic harness issue.

| Candidate | Exact current surface | Net line estimate | Decision | Immediate claim ceiling |
|---|---|---:|---|---|
| C1 raw accepted-state admission | `dynamic_collision_driver.py`, `test_dynamic_collision_driver.py` | about +33; hard cap +40 | PROMOTE_FOR_LATER_IMPLEMENTATION, next | correctness only |
| C2 typed NumPy Rodas event/admission | Rodas core/adapter/classifier, current consumer tolerance metadata, and focused tests | no admitted cap; prior +120 unsupported | REWORK | reference/opt-in correctness only |
| C3 certified slow-manifold discriminator | no identity-consistent executable surface yet | unknown until 182/122 identity issue closes | REWORK | proposed/static only |

### C1 — selected next code task

Purpose: prevent an invalid raw accepted decoupling state from reaching clipping/flooring, Hubble-derived quantities, `N_eff`, or success-like output.

Exact behavior contract:

1. Preserve solver success plus terminal-event admission and move that check before endpoint postprocessing.
2. Add one local structured `RawStateAdmissionError(RuntimeError)` so existing `RuntimeError` callers remain compatible while tests can inspect a stable reason, accepted-sample index or `N`, offending component/value, and a bounded raw snapshot.
3. The two boundaries are explicit. `_make_rhs` receives accepted, trial, finite-difference, and rejected states; preserve its occupation clips and `_resample_comoving_to_thermal` evaluator clamp. They are solver-probe protection, never admission evidence. Reject only nonfinite or nonpositive H before division; do not infer that an RHS call was accepted.
4. Post-solve admission runs only after solver success and exactly one expected terminal event, and only over the nonempty stored `sol.y` columns. Every stored column must have finite `T_gamma>0` and finite strict-domain occupations. For each stored column, recompute diagnostic temperatures and Hubble from that raw column; H must be finite and positive.
5. Validate every stored column, not merely `state_final`. Only after all columns pass may endpoint derivation start. Delete only the final-state `np.clip`; use the raw admitted final arrays for moments, Hubble-derived values, and `N_eff`.
6. Preserve `max_clip_excursion` as a raw stored-sample diagnostic, but any positive physical-domain violation is failure rather than a pass-band criterion. RHS probe excursions do not enter this metric.
7. The implementation may not introduce a repair tolerance silently. If a current valid stored trajectory touches a strict boundary, stop for scientific adjudication rather than widen the domain after observing the result.
8. Preserve units, comoving frame, energy coupling, event definition, initial/boundary conditions, quadrature, and all valid-path observables.

Files and expected diff:

- `src/rabbit/collisions/dynamic_collision_driver.py`: structured exception, one pure raw-state validator, reordered admission, deletion of final-state clipping/floor-as-validity logic.
- `tests/test_dynamic_collision_driver.py`: deterministic monkeypatched-solver failure cases and valid-path characterization.
- Added <=45, deleted >=5, net <=40; no new module, dependency, flag, schema, gate, manifest, or public capability.

Minimal red/green tests for a future authorized implementation:

- solver success plus terminal event with an invalid nonfinal stored column and a valid final column must raise `RawStateAdmissionError` before endpoint derivation;
- a separate invalid final-column case covers `f<0`, `f>1`, `NaN`, `Inf`, and `T<=0`;
- a fake solve invokes the RHS on an out-of-domain probe and then returns a valid stored trajectory, proving the evaluator clamp is preserved and probes are not admitted samples;
- monkeypatched `hubble_3T` returning nonfinite or nonpositive must fail before division/collision postprocessing;
- spies prove clipping-derived moment/Hubble/observable helpers do not decide invalid-state success;
- a literal valid dummy result keeps every returned field bitwise identical, followed by the existing valid-path characterization;
- existing conservation, prefactor, failed-span, and collisionless convergence tests remain green; slow collision-on endpoint tests are separately reported, not silently skipped.

Future commands (not run in this research loop):

```text
pytest -q tests/test_dynamic_collision_driver.py -k "raw_state or failed_integration or energy_conserving or collisionless"
pytest -q tests/test_dynamic_collision_driver.py
```

Hard falsifier: any invalid raw accepted state reaches an observable or success, any failure is decided from clipped values, any valid-path observable changes, or the patch requires relaxing the physical domain after seeing results.

Rollback: revert exactly the two files; no migration. Even a successful repair does not move D-071, endpoint authority, production support, or either failing gate.

### C2 — rework before implementation

Purpose: replace fabricated/message-derived NumPy Rodas terminal success and dishonest work classification with one typed, fail-closed state machine. The direction is retained, but it is not implementation-ready because the generic event's units/tolerances and the expanded line budget are not currently authoritative.

Exact behavior contract:

1. `rodas5p.py` gains a private mutually exclusive event-refinement outcome containing convergence, bracket, root time/state/value, residual, iterations/RHS work, and failure reason. Exhaustion or a failed substep yields failure and no event; the current endpoint fallback is deleted.
2. Crossing tests use explicit signs, preserve exact initial roots, reject nonfinite values, and interpret any finite float direction by sign. The sequential truth table is fixed below.
3. `rodas5p_adapter.py` supports only `events=None` or exactly one scalar callable with `terminal is True`. Missing/false terminal, a sequence, a nonscalar/complex/nonfinite value, or missing event-scale metadata is `UnsupportedEventContract` before stepping. It never truncates a direction, infers outcome from message text, or ORs an unverified event into success.
4. A later event callable must supply positive finite time-absolute and event-value-absolute tolerances in their own units; optional relative terms require a positive declared scale in the same units. The exact current Type-I values and their scientific source remain unresolved, so no implementation assignment is admitted. Convergence will require both the time-bracket and event-residual predicates.
5. Existing result objects must use the counter schema below. `len(t)` becomes `stored_output_points`, never a step count; unavailable units use JSON/Python null, never fabricated zero.
6. `ivp_outcome.py` becomes a total precedence-ordered classifier: contract/nonfinite/solver failure precedes target success, and contradictory `(success,status,event)` states are contract violations.
7. No JAX source, production-default dispatch, solver promotion, dependency, or public capability changes.

Frozen event-semantics table for any later repair:

| Case | Outcome |
|---|---|
| `events=None` | ordinary interval solve; no event arrays |
| one terminal scalar, finite `g0==0` | terminal at initial state, irrespective of direction |
| direction `0`, `g_prev<0<=g_new` or `g_prev>0>=g_new` | oriented bracket; refine |
| direction `>0`, `g_prev<0<=g_new` | oriented bracket; refine |
| direction `<0`, `g_prev>0>=g_new` | oriented bracket; refine |
| reverse time | apply the same truth table to values in integration order; do not flip direction numerically |
| wrong direction or no sign transition | no event |
| missing/false terminal, sequence, nonscalar/complex/nonfinite value | typed unsupported/invalid failure, no event arrays |
| failed substep, bracket loss, iteration/RHS budget exhaustion, or either tolerance unmet | typed refinement failure, no event arrays |

The supported subset assumes at most one continuous root per accepted step. As SciPy documents, multiple crossings inside one step may be missed; this lane cannot claim otherwise. Any gate-bearing consumer must separately justify monotonicity or a step cap for its event.

Required tolerance metadata before C2 can leave REWORK:

- `event_time_atol>0` in integration-variable units and optional `event_time_rtol>=0`;
- `event_value_atol>0` in event units, optional `event_value_rtol>=0`, and a positive event-value scale when the relative term is used;
- `tau_t = event_time_atol + event_time_rtol*max(abs(t_lo),abs(t_hi))`;
- `tau_g = event_value_atol + event_value_rtol*event_value_scale`;
- convergence requires bracket width `<=tau_t` and `abs(g_event)<=tau_g` with bracket orientation retained.

Required budget and counter schema:

- a prospectively fixed `max_refinement_iterations` and `max_refinement_rhs_calls`; the current value 60 is only an implementation ceiling, not acceptance evidence;
- `stored_output_points=len(t)`;
- `attempted_steps=accepted_steps+rejected_steps`;
- `nfev_total` counts every `fun` call, including startup, finite-difference Jacobian, stage, failed, and refinement calls;
- `njev` counts Jacobian builds;
- do not call linear solves `nlu`; expose exact `n_linear_solves`, and set SciPy-shaped `nlu=None` unless actual factorizations are instrumented;
- `event_refinement_iterations` and `event_refinement_nfev` are explicit subsets of the totals;
- every field has a stable integer unit or `None`; no nontrivial solve reports zero merely because a counter is unavailable.

Prospective coherent surface after tolerance authority closes:

- `src/rabbit/solver/rodas5p.py`
- `src/rabbit/solver/rodas5p_adapter.py`
- `src/rabbit/solver/ivp_outcome.py`
- the current `full_coupled_typeI` event-definition surface that must declare scientifically sourced event tolerances
- `tests/test_rodas5p_upgrades.py`
- `tests/test_rodas5p_adapter.py`
- `tests/test_scipy_ivp_outcome_classifier.py`

The classifier file/test are required to close F-ODE-ADJ3-013. F-ODE-ADJ3-012's supplied/scale-aware Jacobian and full configuration work is returned to the companion queue; C2 may only remove fabricated counter values and define honest units. A four-file event-only patch must not claim the full C2 bundle.

Prospective per-file sketch, not an admitted cap: core `+60/-15`, adapter `+35/-20`, classifier `+25/-10`, three test modules `+110/-15`, and consumer tolerance metadata/tests at least `+15/-0`: approximately `+245/-60`, net `+185`. This is inside the policy's 80–200 acceptable band only barely and therefore must be split or further deleted before authorization. The former +120 cap is withdrawn.

Minimal future validation:

- retained residual-123 all-refinement-failed counterexample: typed failure, no `t_events`;
- initial/exact endpoint zero, NaN/Inf, directions `-1,0,+0.5`, time reversal, wrong direction, unsupported multiple/nonterminal events;
- invalid span/state/tolerance/step/reuse settings fail before RHS;
- outcome Cartesian truth table and stored-output-count != attempted-step case;
- deterministic mutants reintroducing fallback success, direction truncation, Boolean precedence, ignored failure, or fabricated zero work are killed.

No future command is admitted until tolerance metadata, budgets, and a <=200 net split are frozen. The earlier illustrative pytest commands remain NOT_RUN and are not an implementation assignment.

Hard falsifier: any refinement failure emits success/event, an unsupported case is silently accepted, a nontrivial solve reports fabricated zero work, the total classifier selects success over contradictory failure, or a seeded mutant survives.

Rollback concept: one event/refinement/classifier slice only after it is re-bounded. Passing would establish only opt-in/reference solver-contract correctness.

### C3 — rework before any code

The mathematical seed remains interesting, but there is no currently localized executable that simultaneously satisfies its identity and implementation constraints:

- D-071/physical-prefix authority is the independent 182-state Python system;
- the localized Rust AOT operator is a distinct folded 122-state system;
- a static result from the 122-state model cannot certify or replace the 182-state blocker;
- no restartable state exists near `N≈0.1653`;
- no frozen moment basis, rank/grid ladder, entropy metric, uniform coercivity neighborhood, nonlinear remainder, tail enclosure, or endpoint adjoint QoI allocation exists.

Required rework sequence:

1. No-code identity decision: either derive an explicit, testable map showing exactly which invariant/manifold statements transfer between 182 and 122 without substituting their work/trajectory claims, or retain C3 as an 182-state offline mathematical study. Creating a new Python production twin or silently treating folded multiplicity as equivalence is forbidden.
2. Freeze a reduction contract: discrete moments and weights, branch choice, parameter-to-state map, strict positivity mechanism, P/Q projectors, entropy-weighted norm, rank selection independent of results, coercivity/remainder inequalities, full-grid and radial/angular/tail residual, adjoint QoIs, and total error allocation.
3. Only then may one existing private audit surface host a static falsification helper. It must state `physical_prefix_executed=false`, `d071_reopen_earned=false`, and stop on rank growth, nonunique/ill-conditioned moment inversion, invariant/positivity failure, or certificate cost near the full model.
4. A later gate-bearing execution requires separate authority and the full physical-start, checkpoint, call, and wall obligation. Static evidence never advances that gate.

Decision: REWORK. No line estimate or implementation task is admitted until steps 1–2 close. Fermi-Dirac AP penalization remains HOLD behind the same invariant/identity proof.
