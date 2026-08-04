# V0 — pre-output protocol for the V1 bounded solver diagnostic

**Date:** 2026-08-04
**Authority:** owner grant to build and run a temporary diagnostic instrument that records the
solver information the r4 run did not retain.
**Sealed before any output byte.** This document is committed before the diagnostic code exists.
Nothing below may be edited after the first run byte; a wrong prediction is preserved as a wrong
prediction.

**Claim ceiling, stated first:** V1 output is **diagnostic evidence only**. It cannot satisfy any
D-071 reopen conjunct, cannot move a gate, cannot become an evidence package, and does not
constitute a new measurement of the physical proposition. `G-F10-INDEPENDENT-FLRW` stays FAIL and
`G-HARNESS-INTEGRITY` stays FAIL. `implies_status: null`.

---

## 1. Pins (item 1–3 of §5's V0 list)

```text
input_commit                f3b2e5d679894b0cc9b169c888f65dbc1a619c00
frozen_module               src/rabbit/decoupling/_independent_noqke.py
frozen_module_sha256        760a7c044081e507fae9d5695b301bd44f6466d96322c46f53b77161e32b558a
python                      3.12.3
numpy                       2.4.2
scipy                       1.17.0
scipy_bdf_source_sha256     a5b75a2ae8aca2e66cc35ed268af51d8a9685878209d5cf77a44c8ccef3b76e6
blas                        scipy-openblas
host_cores                  24
```

**Correction to the prior design record:** the revised design document states SciPy 1.17.1. The
installed and pinned version is **1.17.0**. The diagnostic pins the installed version and verifies
the digest above at runtime.

Configurations, both taken unchanged from `scripts/audit/d069_independent_trajectory_r4.py`:

```text
domain (primary)   order=60  y_max=30.0  rtol=1e-6  atol=1e-9  t_start=10.0  t_gamma_end=0.005
base   (control)   order=48  y_max=24.0  rtol=1e-6  atol=1e-9  t_start=10.0  t_gamma_end=0.005
incoming_polar_order=4  final_polar_order=4  electron_radial_order=24  (both)
```

## 2. Why a control phase is mandatory

The §11 addendum measured that dense-FD Jacobian dominance and multi-row plateaus occur in the
**completed** base phase as well as the stalled domain phase, and that §5's V1 signature table row 1
therefore fires on a healthy run. A diagnostic that observes only the stalled configuration would
repeat exactly that error. **Every signature below is defined as a comparison against the base
control measured by the same instrument**, never as an absolute property of the stalled run.

This is an amendment to the re-audit's V1 design and is the reason the wall cap in §5 is larger
than the re-audit's two hours.

## 3. What the retained r4 run could not answer, and this instrument records

From reading `scipy/integrate/_ivp/bdf.py` at the pinned digest, one control-flow fact drives the
design: **the Jacobian is refreshed only when Newton fails to converge** (`bdf.py:370-375`); on an
error-test failure the code explicitly does not reset `LU` (`bdf.py:398-399`). Therefore
`FD batches ≈ 1 + Newton convergence failures`, and counting batches alone cannot separate mechanisms.

Recorded per trial (attempted step), which r4 retained none of:

- `t_old`, `h_abs`, `order`, proposed `t_new`, acceptance boolean;
- Newton `converged`, `n_iter`, and which branch terminated the trial
  (`newton_failure` / `error_test_failure` / `accepted` / `too_small_step`);
- `error_norm`, and the componentwise scaled error `error / (atol + rtol*|y_new|)` reduced to
  its max, argmax index, and the top-5 offending component indices;
- Jacobian refresh events and LU factorization events;
- `n_equal_steps`, and `min_step` versus `h_abs`.

Recorded per raw RHS call, which r4 retained only in aggregate:

- call index, `t` argument, wall time;
- `whole_reaction_domain_rejections` and `largest_matrix_roundoff_correction`;
- `matrix_roundoff_corrections`;
- cancellation measures available without touching the frozen module:
  `first_law_residual` from the module's own diagnostics, and the channel-cancellation ratio
  `(|electron| + |self_interaction|) / |total|` reduced to max and median over components.

Recorded once, at the accepted state immediately preceding the first captured reject-refresh cycle:

- the accepted state vector, the dense Jacobian, and its full eigenvalue spectrum;
- `h·λ` for the observed `h` and order, against the BDF stability requirement.

## 4. Predeclared mechanisms and counterfactuals (item 4)

Each is judged **domain versus base control**, both measured by this instrument. `R` denotes
"per accepted step" throughout, because §11 showed per-raw-call framings do not discriminate.

| ID | Mechanism | Positive signature | Counterfactual that rejects it as dominant |
|---|---|---|---|
| M1 | Newton-failure cycling | `newton_failure` terminates the majority of trials, and Newton failures per accepted step in domain exceeds the base control by >3x | Newton converges on most trials, or the domain-to-base ratio is <=3x |
| M2 | Error-test / noise floor | Newton converges but `error_norm > 1` repeatedly; rejected `error_norm` clusters just above 1; and the cancellation ratio or `first_law_residual` implies an RHS noise level at or above `atol + rtol*|y|` for the argmax components | `error_norm` on rejections is far above 1 (genuine accuracy limitation, not a floor), or cancellation is comparable to base |
| M3 | Stability wedge / fast modes | the snapshot spectrum has eigenvalues whose `h*lambda` sits outside the A(alpha) region for the observed order, and the largest stable `h` implied by the spectrum is within 3x of the observed `h` | spectrum permits `h` more than 10x larger than observed; or order stays at 4-5 while `h` collapses |
| M4 | Kinematic discontinuity | `whole_reaction_domain_rejections` changes value at or immediately before the trial points where failures occur, and does so more often per accepted step than in base | the counter is constant or varies smoothly across failures in both runs |
| M5 | Step-size floor | `h_abs` reaches `min_step`, or `TOO_SMALL_STEP` is returned | `h_abs` stays orders of magnitude above `min_step` |
| M0 | No pathology (null) | domain trial statistics per accepted step are within 3x of the base control on every axis above, and the step size is simply smaller because the local error demands it | any axis exceeds 3x |

**Row 1 of §5's signature table is withdrawn and replaced by M1 above**, as §11.3 required, because
its original form fired on the completed base phase.

## 5. Hard caps (item 5)

```text
domain phase      1,200 integration raw RHS calls
base control        600 integration raw RHS calls
post-capture Jacobian   one only, <=366 raw RHS calls (183 columns + FD retry headroom)
total wall          3.0 hours across all phases, enforced by a monotonic deadline
reject-refresh      stop the domain phase after 3 complete cycles past the first accepted-state
                    snapshot, if that occurs before the call cap
```

The caps are enforced in code and breaching one ends the phase. **Extending a cap because the
signature is inconvenient is forbidden.** The retained r4 trace shows the first stall episode inside
raw eval 1,100 of the domain phase, so 1,200 covers it.

The wall cap exceeds the re-audit's two hours solely to fund the mandatory base control of §2.
The two phases may run concurrently under the standing parallel-run grant; if they do, **per-call
wall times are not comparable to the retained serial r4 timings and are excluded from any wall
attribution claim.** All step, error, Newton, and spectrum data are unaffected by concurrency.

## 6. Outputs and failure preservation (item 6)

Instrument code lives in the session scratch directory and is **not tracked**. Output data is
written outside the repository to the same scratch directory:

```text
<scratch>/v1_diag/{domain,base}/trials.jsonl        one row per attempted step
<scratch>/v1_diag/{domain,base}/rhs_calls.jsonl     one row per raw RHS call
<scratch>/v1_diag/{domain,base}/accepted.jsonl      one row per accepted step
<scratch>/v1_diag/domain/snapshot.npz               accepted state, dense Jacobian, eigenvalues
<scratch>/v1_diag/{domain,base}/stdout.log          full console output
<scratch>/v1_diag/run_record.json                   pins, caps, outcome, violated assumptions
```

**Every failure, exception, non-reproduction, and cap breach is retained verbatim.** A run that
errors is reported as an errored run; its partial data is kept and labelled partial. No output is
deleted, re-run over, or selected among. If the instrument itself is defective the result is
`FAIL_PROTOCOL`, not a partially informative measurement.

A distilled record — counters, distributions, and the decision — is committed as documentation.
The raw per-call dumps stay in scratch.

## 7. Claim ceiling (item 7)

**V1 output cannot satisfy any D-071 reopen conjunct.** It is not a materially new method, not a
prospectively sealed scientific contract, and not a bounded discriminator with end-to-end margin.
It selects among candidate families and kills wrong ones. A diagnosed controller or tolerance
pathology **does not** license tuning and re-running the existing instrument: order caps, tolerance
changes, analytic Jacobians, faster hosts, and larger budgets remain disqualified by D-071.

## 8. Tracked-file budget (item 8)

```text
production files changed        0
harness files changed           0  (.agent-harness/, .codex/hooks/ untouched; D-074 respected)
gate/registry files changed     0
frozen module changed           0  (digest verified before and after)
persistent diagnostic framework none - the instrument is scratch-only and is not installed,
                                imported by, or referenced from any tracked file
tracked additions               documentation only: this protocol and the later result record
```

The instrumented BDF is a **subclass in scratch** whose `_step_impl` is a verbatim copy of the
pinned SciPy method with logging lines inserted. The instrument verifies this mechanically at
startup: it strips its inserted lines and requires the remainder to match the pinned SciPy source
exactly. A mismatch is `FAIL_PROTOCOL` before any physics runs.

## 9. Decision rule

- `PASS_DIAGNOSTIC` — exactly one mechanism's positive signature holds and its counterfactual is
  absent, judged against the base control.
- `MULTI_MECHANISM` — more than one positive signature holds; reported as such, not collapsed.
- `INCONCLUSIVE` — mixed or missing signature, non-reproduction, or a cap reached first.
- `FAIL_PROTOCOL` — a required counter is missing, the configuration differs from §1, the source
  self-check fails, or raw failure data is lost.

No outcome is a gate pass. `PASS_DIAGNOSTIC` selects at most one family for later contract design;
it does not authorize that design.

```text
added_lines: this document
runtime_behavior_changed: no
physics_behavior_changed: no
known_blocker_reduced: no -- protocol only, sealed before output
blocker_movement_ratio: 0
cost_effectiveness_verdict: RECORD_ONLY
```
