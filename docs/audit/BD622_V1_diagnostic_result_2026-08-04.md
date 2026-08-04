# V1 diagnostic result — the stalled configuration has ten linearly unstable modes, all beyond y = 24

**Date:** 2026-08-04
**Protocol:** `docs/audit/BD622_V0_diagnostic_protocol_2026-08-04.md`, sealed at `e40ff44` before the
instrument existed. Nothing below was added to the protocol after the data landed.
**Decision under the sealed rule: `PASS_DIAGNOSTIC`. Single positive mechanism: `M3` (stability
wedge / fast modes).**

**Claim ceiling, unchanged.** Diagnostic evidence only. This does not satisfy any D-071 reopen
conjunct, does not move a gate, is not an evidence package, `implies_status: null`. The board stays
**6 PASS / 2 FAIL**; `G-F10-INDEPENDENT-FLRW` stays closed on its preserved measurement and
`G-HARNESS-INTEGRITY` stays FAIL under the D-074 freeze. Every figure in this record is printed
from `ANALYSIS.json` by `render_record.py`; none is hand-transcribed.

Raw data, instrument source and analysis scripts are retained at
`.agent-harness/runs/run-20260804-f10-v1-diagnostic/` (28 files, 1.4 MB; that directory is
gitignored, so this document is the tracked record).

---

## 0. What this settles, in one paragraph

The stalled 60/30 configuration and the completed 48/24 configuration were instrumented with the
same code and compared at the same physical epoch. They differ in one measured respect that no
other axis matches: **the 60/30 Jacobian has ten eigenvalues with positive real part and four
complex-conjugate pairs, and the 48/24 Jacobian has none of either.** Every one of those unstable
modes is localized in the momentum region `y > 24`, which exists only in the extended grid. The
run's rejections are exclusively Newton-convergence failures — it never once failed an accuracy
test. Separately, the "unexplained drop" in the retained r4 log is now mechanically explained: the
value 0.1813 was never an accepted state, it was a trial point that failed Newton.

## 1. The 0.1813 question is answered

D-071 §2.3 recorded an "unexplained DROP of −0.0184" at r4 domain evaluation 951, after a "peak"
of 0.1813 at evaluation 751. The §11 addendum read this as a rejected trial; the critical re-audit
correctly ruled that reading `UNVERIFIABLE` from the retained bytes, because the r4 log prints raw
RHS calls and carries no accepted-step trace. The instrumented rerun reproduces r4 exactly and
does carry that trace:

```text
domain accepted states: 55, maximum accepted t = 0.16282186298687312
accepted states in [0.180, 0.182]: none

trials.jsonl, every non-accepted trial in the run:
  step 54  t=0.16282186298687310  h=0.03701  order 4  newton_failure  raw=524
  step 56  t=0.18132778147001450  h=0.01851  order 4  newton_failure  raw=904
  step 56  t=0.17207482222844380  h=0.00925  order 4  newton_failure  raw=907
```

`0.1813` appears in the record exactly once, as a **failed Newton trial at raw call 904**, and the
solver then halved the step and failed again at `0.1721`. The r4 log's later `0.1629` is the last
accepted state, `0.16282`, printed to four decimals. The retreat is the solver falling back from a
failed trial; it is not a regression of the solution and not an anomaly. **The re-audit's
`UNVERIFIABLE` ruling was correct on the evidence then available, and the question is now closed on
new evidence rather than on argument.**

## 2. The mechanism, and the sub-hypothesis this review refuted

All rejections in the stalled run were Newton failures; **error-test failures were zero in both
runs**. Accuracy was never the binding constraint. The natural first explanation — that the BDF
iteration matrix `I - cJ` goes near-singular, since `c = h/alpha` and a positive eigenvalue near
`1/c` would make it so — **was tested and refuted**:

```text
domain, failing step:  h=1.85059e-02  order 4  c=8.52889e-03  singular at lambda = 1/c = 117.2
                       eigenvalues with |1 - c*lambda| < 0.25 : 0
                       min |1 - c*lambda| over the spectrum   : 1.0503
```

The iteration matrix is not near-singular. Nor is the instability a matter of the diagonal: **every
diagonal entry of both spectral blocks is negative** (0 of 180 and 0 of 144 positive), so the
positive eigenvalues arise from off-diagonal coupling between nodes.

The spectra were recomputed on the spectral block alone, because the elapsed-time variable is
structurally decoupled — its column in the Jacobian is exactly zero, which makes the matrix block
triangular and would otherwise let one artifact dominate every eigenvector. On the spectral block:

```text
domain 60/30, 180x180 : 10 eigenvalues with positive real part, 4 complex pairs
base   48/24, 144x144 :  0 eigenvalues with positive real part, 0 complex pairs

leading unstable modes (eigenvector weights on their top components, not artifacts):
  lam=+6.6136e+05  h*Re=+1.457e+04  (s0,n47,y=26.83,w=0.73) (s0,n56,y=29.72,w=0.27)
  lam=+4.9427e+05  h*Re=+1.089e+04  (s1,n47,y=26.83,w=0.73) (s1,n56,y=29.72,w=0.27)
  lam=+2.9047e+04  h*Re=+6.401e+02  (s0,n45,y=25.81,w=0.78) (s0,n54,y=29.34,w=0.23)
  lam=+2.2047e+04  h*Re=+4.859e+02  (s2,n45,y=25.81,w=0.69) (s1,n45,y=25.81,w=0.39)
  ... 6 more, all of the same character

y of the dominant components: min 24.07, median 28.12, max 29.94
fraction beyond y = 24.0    : 1.00
base grid's last node       : y = 23.985
```

**Every unstable mode lives beyond the outer edge of the grid that completed.**

## 3. The coordinate observation, stated as a hypothesis and not as a measurement

The state is integrated in cloglog coordinates and the right-hand side divides by the cloglog chain
factor, `dc/dN = pair_rate / (H * chain)`. At the snapshot states:

```text
                        domain 60/30      base 48/24
minimum occupation       9.4718e-14       3.8326e-11      (a factor of 404)
chain factor             equals the occupation for small f
components below atol    66 of 180        33 of 144       (atol = 1e-9)
tail |J[ii]|             3.4e7            2.4e5           (a factor of ~140)
```

Extending `y_max` from 24 to 30 adds nodes whose occupation falls to `1e-13` — **four orders of
magnitude below the requested absolute tolerance** — and the right-hand side divides by exactly that
quantity. The unstable modes sit on precisely those nodes.

**This is a coherent account, and it is a hypothesis, not a result.** What is measured is the
localization, the magnitudes and the tolerance comparison. What is *not* established is whether the
positive eigenvalues are a genuine property of the discretized dynamics or an artifact of
finite-difference differencing in a coordinate whose chain factor is underflowing. The base control
is partial evidence against generic finite-difference noise — the same differencing at the same
tolerances on the same code produced exactly zero positive eigenvalues — but the domain tail is the
one place where the differencing is most fragile, so the alternative is not excluded. Testing it
means recomputing the Jacobian with a different differencing step or in occupation coordinates,
which is new compute outside the sealed cap and is **not** performed here.

---

## 4. Provenance and instrument integrity, verified at startup

```text
pinned input commit        f3b2e5d679894b0cc9b169c888f65dbc1a619c00
HEAD at run time           e40ff440c0dd200745a5da45d26742e93594258a
pinned is ancestor of HEAD True
input surfaces moved       none
input surfaces dirty       none
frozen_module_sha256       760a7c044081e507fae9d5695b301bd44f6466d96322c46f53b77161e32b558a
scipy                      1.17.0
numpy                      2.4.2
scipy_bdf_source_sha256    a5b75a2ae8aca2e66cc35ed268af51d8a9685878209d5cf77a44c8ccef3b76e6
```

| Copied surface | Probe lines | Pinned lines | Stripped lines | Verbatim |
|---|---|---|---|---|
| `scipy…bdf.BDF._step_impl` | 11 | 142 | 142 | yes |
| `_trajectory_core.make_rhs` | 1 | 54 | 54 | yes |

## 5. Reproduction of the retained r4 run

| Phase | Points compared | Mismatches | Reproduces r4 |
|---|---|---|---|
| domain | 19 | 0 | yes |
| base | 12 | 0 | yes |

## 6. Measured counters, domain versus base control

| Quantity | domain 60/30 | base 48/24 control | domain / base |
|---|---|---|---|
| state dimension | 182 | 146 |  |
| first grid node y | 0.01185 | 0.01475 |  |
| stop reason | cycle_cap:3 | call_cap:600 |  |
| integration raw RHS calls | 907 | 600 |  |
| accepted steps | 55 | 55 |  |
| trials (attempted steps) | 58 | 55 |  |
| N reached | 0.1628 | 0.1881 |  |
| raw calls per accepted step | 16.49 | 10.91 | 1.512 |
| Newton calls | 61 | 57 |  |
| Newton non-convergences | 6 | 2 |  |
| Newton non-conv. per accepted step | 0.1091 | 0.03636 | 3 |
| hard Newton failures (trial killed) | 3 | 0 |  |
| hard Newton failures per accepted | 0.05455 | 0 | inf |
| Jacobian refreshes (integration) | 3 | 2 |  |
| Jacobian refreshes per accepted | 0.05455 | 0.03636 | 1.5 |
| LU factorizations | 28 | 25 |  |
| error tests | 55 | 55 |  |
| error-test failures | 0 | 0 |  |
| error-test failures per accepted | 0 | 0 | 1 |
| median error norm, accepted | 1.232e-04 | 1.231e-04 |  |
| median error norm, rejected | n/a | n/a |  |
| max error norm, rejected | n/a | n/a |  |
| median cancellation ratio | 77.87 | 26.21 | 2.971 |
| p95 cancellation ratio | 3298 | 431.4 |  |
| max first-law residual | 1.005e-13 | 8.306e-15 |  |
| max roundoff correction | 0 | 0 |  |
| kinematic-rejection distinct values | 1 | 1 |  |
| kinematic-rejection changes per accepted | 0 | 0 | 1 |
| h, first accepted | 5.996e-22 | 5.367e-22 |  |
| h, last accepted | 0.01851 | 0.04077 |  |
| h, min accepted | 5.996e-22 | 5.367e-22 |  |
| min_step seen | 4.941e-323 | 4.941e-323 |  |
| TOO_SMALL_STEP returns | 0 | 0 |  |
| order at last accepted step | 4 | 4 |  |
| N advance per accepted step | 0.00296 | 0.00342 |  |

Trial outcome counts — domain `{"accepted": 55, "newton_failure": 3}`, base `{"accepted": 55}`.
Accepted-step order distribution — domain `{"1": 30, "2": 15, "3": 8, "4": 2}`, base `{"1": 30, "2": 12, "3": 12, "4": 1}`.

## 7. Jacobian spectrum at the snapshot accepted state

| Spectrum quantity | domain 60/30 | base 48/24 control |
|---|---|---|
| `N` | 0.1258 | 0.1881 |
| `h` | 0.02204 | 0.04077 |
| `order` | 3 | 4 |
| `jacobian_raw_calls` | 184 | 148 |
| `n_eig` | 182 | 146 |
| `n_eig_structurally_zero` | 1 | 1 |
| `eig_real_max` | 6.614e+05 | 0 |
| `eig_real_min` | -3.600e+08 | -2.168e+06 |
| `eig_absmax` | 3.600e+08 | 2.168e+06 |
| `eig_absmin_nonzero` | 0.5116 | 0.5089 |
| `stiffness_ratio_nonzero` | 7.037e+08 | 4.259e+06 |
| `n_eig_positive_real` | 10 | 0 |
| `n_complex_pairs` | 4 | 0 |
| `max_imag_over_abs_nonzero` | 0.003757 | 0 |
| `bdf_alpha_deg` | 86.03 | 73.35 |
| `max_angle_from_neg_real_deg` | 180 | 0 |
| `n_eig_in_unstable_cone` | 10 | 0 |
| `h_lambda_absmax` | 7.934e+06 | 8.837e+04 |

## 7b. Where the local error sits

| Error-localisation quantity | domain 60/30 | base 48/24 control |
|---|---|---|
| `n_error_tests` | 55 | 55 |
| `argmax_nonspectral` | 50 | 44 |
| `argmax_y_median` | 29.55 | 22.87 |
| `argmax_y_min` | 0.01185 | 0.01475 |
| `argmax_y_max` | 29.85 | 23.92 |
| `argmax_y_frac_median` | 0.9849 | 0.9529 |
| `first_node_y` | 0.01185 | 0.01475 |
| `last_node_y` | 29.99 | 23.99 |

`domain` argmax components, most frequent first:

```text
  47  {"index": 181, "kind": "elapsed_time"}
   3  {"index": 180, "kind": "T_gamma"}
   1  {"index": 52, "kind": "spectral", "node": 52, "species": 0, "y": 28.80117714266441, "y_frac_of_ymax": 0.9600392380888138}
   1  {"index": 55, "kind": "spectral", "node": 55, "species": 0, "y": 29.54552683147579, "y_frac_of_ymax": 0.9848508943825264}
   1  {"index": 56, "kind": "spectral", "node": 56, "species": 0, "y": 29.71600802628897, "y_frac_of_ymax": 0.990533600876299}
   1  {"index": 57, "kind": "spectral", "node": 57, "species": 0, "y": 29.846818428333325, "y_frac_of_ymax": 0.9948939476111108}
```

`base` argmax components, most frequent first:

```text
  43  {"index": 145, "kind": "elapsed_time"}
   4  {"index": 0, "kind": "spectral", "node": 0, "species": 0, "y": 0.014747912970887178, "y_frac_of_ymax": 0.0006144963737869658}
   2  {"index": 41, "kind": "spectral", "node": 41, "species": 0, "y": 22.870549640586837, "y_frac_of_ymax": 0.9529395683577849}
   2  {"index": 44, "kind": "spectral", "node": 44, "species": 0, "y": 23.64709911055497, "y_frac_of_ymax": 0.9852957962731237}
   1  {"index": 144, "kind": "T_gamma"}
   1  {"index": 137, "kind": "spectral", "node": 41, "species": 2, "y": 22.870549640586837, "y_frac_of_ymax": 0.9529395683577849}
```

## 7c. Section-11 inference, checked against recorded Jacobian events

| Quantity | domain 60/30 | base 48/24 control |
|---|---|---|
| `predicted_fd_batch_calls` | 183 | 147 |
| `measured_jacobian_batch_median` | 184 | 148 |
| `constant_N_plateaus` | 61 | 59 |
| `multi_call_plateaus` | 59 | 57 |
| `median_plateau_width_calls` | 2 | 2 |
| `max_plateau_width_calls` | 219 | 181 |
| `calls_inside_multi_plateaus` | 905 | 598 |

Measured Jacobian batch widths — domain `[189, 184, 184]`, base `[148, 148]`.

## 7d. Adjudication against the six mechanisms sealed in V0

| ID | Positive? | Sealed threshold | Measured |
|---|---|---|---|
| `M1_newton_cycling` | **no** | majority of trials AND >3x base | newton_failures_dominate_trials=no; domain_over_base_ratio=3 |
| `M2_error_noise_floor` | **no** | >3x base error-test failures per accepted step AND rejected norms near 1 | error_test_failures_per_accepted_ratio=1; median_rejected_error_norm=n/a; rejected_norm_clusters_just_above_1=no; cancellation_ratio_vs_base=2.971; first_law_residual_vs_base=1 |
| `M3_stability_wedge` | **yes** | any nonzero eigenvalue outside the A(alpha) cone | n_eig_in_unstable_cone=10; bdf_alpha_deg=86.03; max_angle_from_neg_real_deg=180; h_lambda_absmax=7.934e+06; n_complex_pairs=4; stiffness_ratio_nonzero=7.037e+08 |
| `M4_kinematic_discontinuity` | **no** | >3x base changes per accepted step | dom_rej_changes_per_accepted_ratio=1; domain_distinct_values=1; base_distinct_values=1 |
| `M5_step_floor` | **no** | TOO_SMALL_STEP returned, or h within 100x of min_step | h_min_accepted=5.996e-22; min_step_seen=4.941e-323; too_small_step_returns=0 |
| `M0_no_pathology` | **no** | every per-accepted-step axis within 3x of base | all_axis_ratios={'newton_nonconverged_per_accepted': 3.0, 'hard_newton_failures_per_accepted': inf, 'jacobian_refreshes_per_accepted': 1.5, 'error_test_failures_per_accepted': 1.0, 'dom_rej_changes_per_accepted': 1.0, 'raw_calls_per_accepted_step': 1.511666666666667, 'median_cancel_max': 2.9710067534467868, 'median_first_law_residual': 1.0, 'median_roundoff_max': 1.0} |

**Positive mechanisms:** `['M3_stability_wedge']`

**Decision: `PASS_DIAGNOSTIC`**

---

## 7e. One independent corroboration, and one verdict decided at its boundary

**Corroboration.** The local error norm localizes the same way the eigenvectors do, and this is an
independent signal — it comes from the solver's error estimator, not from the Jacobian. In the
stalled run every spectral argmax sits at `y` between 28.8 and 29.8 (median `y` = 29.55, 98.5% of
`y_max`); in the control the spectral argmaxes are split between the far tail and **node 0 near the
origin**, which appears four times. The stalled run's error never concentrates near the origin.
Two different instruments therefore point at the same momentum band.

**Boundary call, stated because a threshold met exactly is not a threshold passed.** `M1` was
sealed as "newton failures terminate the majority of trials **and** exceed 3x the base rate per
accepted step". The measured ratio is exactly `3.0` and hard failures are 3 of 58 trials, so both
conjuncts fail and `M1` is negative under the sealed rule. The underlying asymmetry is nevertheless
real — 3 hard Newton failures against the control's **0** — and it is what makes `M0`'s
`hard_newton_failures_per_accepted` ratio `inf` and correctly kills the null. Read `M1 = no` as
"not the dominant mechanism by the sealed test", not as "no Newton anomaly".

## 8. What this changes for the candidate families — and what it does not authorize

**It does not authorize anything.** Under D-071 a reopen still needs all three of: a materially new
method, a contract sealed before any output byte, and a reviewed bounded discriminator projecting
end-to-end completion inside the wall budget with margin. None of those exists. Under V0 §7 a
diagnosed coordinate or tolerance pathology explicitly **does not** license tuning and re-running
this instrument — changing `atol`, changing `y_max`, changing coordinates, capping the order or
supplying an analytic Jacobian are all "generic optimization" and remain disqualified.

With that stated, the evidence bears unevenly on the families:

| Family | Effect of this measurement |
|---|---|
| H-T1 diagnosis | Discharged. The instrument ran, reproduced r4, and returned a single mechanism. |
| H-T3 analytic domain reduction | **Materially supported, and its reviewer challenge is answered on the specific point.** The adversarial review found my claim that density-only refinement "does not move the first node toward 0" to be false, and warned the driver might be near-origin spacing, which would invert H-T3's prediction. The measurement puts 100% of the unstable-mode weight at `y > 24` and none near the origin. That is evidence for the domain-extension driver. It is **not** a measurement of a 60/24 configuration, so it does not show what a density-only run would do — it removes one of the two candidate drivers, it does not confirm the other. |
| H-T2 asymptotic reformulation | Neither supported nor killed. The stall is not the cancellation noise floor that plain delta-f would cure (error-test failures were zero), so that sub-route loses its motivation here; a micro-macro/AP closure that removes tail modes from the integrated state is untouched by this result. |
| H-T4 defect + log-norm certificate | **Weakened.** A certificate rests on a usefully negative one-sided Lipschitz constant. Ten eigenvalues with positive real part, the largest at `+6.6e5`, make the logarithmic norm of this operator positive in any norm close to the natural one. The V2A screen should treat that as its leading obstacle rather than discovering it later. |
| H-T5 pathology framing | Sharpened into a specific, falsifiable statement, and the vague version is retired. The 60/30 discretization is not "pathological" in general; it carries unstable modes in a specific momentum band. |

## 9. Limits of this result, stated as fixed before the data where they were

1. **The sealed cycle cap bound first.** The domain phase stopped at raw call 907 on its third hard
   Newton failure, not at the 1200-call cap. It therefore did **not** reach r4's evaluation 951,
   although it captured the failing trials at 904 and 907 that produce that print.
2. **It samples the onset of the creep, not the developed creep** at evaluation 11051. No claim is
   made about the regime there.
3. **One snapshot, one state, one host.** The spectrum is from a single accepted state at
   `N = 0.1258`; no profile along the epoch exists.
4. **Both windows include the startup ramp** — 30 of 55 accepted steps are at order 1 in both runs.
   This is a like-for-like comparison of two early regimes, not of two developed ones.
5. **Concurrency.** Both phases ran at once; per-call wall times carry a contention penalty
   (measured 4.94 s/call against r4's 4.42) and are excluded from wall attribution. Step, Newton,
   error-norm and spectrum data are unaffected.
6. **JSONL sinks were block-buffered rather than line-buffered.** Nothing was lost on the normal and
   capped exits taken here, but a SIGKILL would have lost the last partial block. This is a real
   weakness against the protocol's failure-preservation requirement and should be fixed before any
   future run.
7. **The finite-difference alternative to §3 is not excluded**, as that section states.

## 10. Protocol events worth recording

**The instrument's own self-check fired, correctly, and was not loosened.** The first launch
returned `FAIL_PROTOCOL` because sealing V0 advanced `HEAD` past the pinned input commit. Bare
equality on the commit was replaced by an ancestry-plus-input-surface relation — the pinned commit
must be an ancestor of `HEAD`, and the diff between them must touch nothing under `src/`,
`scripts/`, `.agent-harness/` or `.codex/`, with those paths clean — which proves no physics input
moved instead of assuming it from one hash. All digest and version pins stayed strict equality. The
refusal and its payload are retained.

**One pin in the prior design record was wrong and is corrected here.** That record states SciPy
1.17.1; the installed, pinned, and digest-verified version is **1.17.0**.

## 11. Cost

```text
tracked files changed        1 (this record)
production files changed     0
harness files changed        0   (.agent-harness/scripts, .codex/hooks untouched; D-074 respected)
gate or registry changes     0
frozen module               unchanged, digest verified before and after
instrument                  scratch-only, not installed, not imported by any tracked file
compute                     domain 907 + 184 raw RHS calls; base 600 + 148; ~2.0 h wall, capped
runtime_behavior_changed     no
physics_behavior_changed     no
known_blocker_reduced        no -- one mechanism identified, no gate moved, no reopen authorized
blocker_movement_ratio       0
cost_effectiveness_verdict   ACCEPT -- the information gap that blocked analysis is closed
```

## 12. What is not concluded

Nothing about the physical proposition. This measures an instrument, not a question. The gate stays
FAIL on its preserved measurement, the reopen conditions are untouched, and the next step — if the
owner wants one — is a paper exercise (the three-channel tail bound of V3, or the V2A admissibility
screen), not another run of this instrument.
