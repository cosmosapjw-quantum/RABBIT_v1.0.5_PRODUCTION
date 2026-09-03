# V1C — the V1 diagnostic conclusion is withdrawn, and the plan is rewritten around what survives

**Date:** 2026-08-04
**Supersedes:** `docs/audit/BD622_V1_diagnostic_result_2026-08-04.md` (commit `e5fcae8`).
That record is retained unedited, as this project's rule requires; it is retired by replacement.
**Corrected decision: `INCONCLUSIVE`.** The `PASS_DIAGNOSTIC` letter and the `M3` mechanism are
withdrawn. The board is unchanged at **6 PASS / 2 FAIL** and no gate ever moved on the withdrawn
claim.

---

## 1. What was wrong, in one paragraph

The V1 record concluded that the stalled 60/30 configuration carries ten linearly unstable modes and
the completed 48/24 configuration carries none. **That conclusion is false.** The Jacobian it rested
on was computed by SciPy's `num_jac` with a step that had ratcheted to `sqrt(EPS) * 1e5`, making the
tail entries wrong by up to 59% on the diagonal. Recomputed at a converged step, the same operator at
the same state has **zero** eigenvalues with positive real part and a spectral abscissa of
**−6.43**, statistically indistinguishable from the control's **−6.58**. There is no measured
spectral difference between the stalled and the completed configuration. Separately and
independently, three defects in the adjudication meant the sealed protocol's own decision rule would
have returned `INCONCLUSIVE` regardless of the spectrum.

## 2. The finite-difference defect, verified first-hand

`num_jac` carries a stateful per-component `factor` that ratchets upward by ten whenever a
difference is judged too small (`scipy/integrate/_ivp/common.py`, and the retry loop inside
`_dense_num_jac`). Over a run it can climb far above the `sqrt(EPS)` default. Verified directly, with
five right-hand-side evaluations:

```text
stored J[47,47]                       = -1.725641e+07
recomputed at h = 3.998005e-02        = -1.725641e+07   deviation 0.0000%   <- reproduces the stored
                                                        value bit for bit, so this was the step used
recomputed at h = 4.0e-03             = -1.788851e+07   deviation -3.66%
recomputed at h = 4.0e-05             = -1.795948e+07   deviation -4.07%    <- converged plateau
recomputed at h = 4.0e-07             = -1.796020e+07   deviation -4.08%    <- converged plateau
```

The full Jacobian was then recomputed at the un-ratcheted `sqrt(EPS)` step (183 evaluations, 834 s):

| Spectral block, 180x180 | stored | recomputed at a converged step |
|---|---|---|
| eigenvalues with positive real part | 10 | **0** |
| spectral abscissa | +6.613557e+05 | **-6.4271** |
| complex conjugate pairs | 4 | 0 |
| diagonal relative error versus recomputed | — | max 59.2%, 14 components over 1%, 55 over 0.1% |

**This is the reusable methodological finding of the exercise**, and it is worth more than the
conclusion it destroyed: *in this codebase a SciPy-supplied finite-difference Jacobian may not be
used for any spectral or stability claim without pinning the step and demonstrating a convergence
plateau.* The adaptive step is tuned for Newton convergence, not for spectral accuracy, and the two
requirements differ by orders of magnitude in the momentum tail.

**Why the earlier robustness check missed it.** The V1 analysis tested the spectrum against *random*
perturbations and found the leading eigenvalues stable to five digits. That test was the wrong shape:
the finite-difference error is *systematic* and sign-correlated with the diagonal, so no isotropic
perturbation of any amplitude reproduces it. Random-noise robustness is not evidence of accuracy.

## 3. Three adjudication defects, independent of the spectrum

Each of these is a mismatch between the protocol sealed at `e40ff44` and the code that produced the
letter. Any one of them alone changes the answer.

| # | Sealed text (V0) | What was implemented | Consequence |
|---|---|---|---|
| 1 | M3's positive signature is a **conjunction**: eigenvalues outside the A(alpha) region **and** "the largest stable `h` implied by the spectrum is within 3x of the observed `h`" | `analyse.py:286` implements the first clause only: `positive = (cone > 0)` | the only h-dependent content of M3 was dropped; the surviving predicate is scale-free |
| 2 | `PASS_DIAGNOSTIC` requires "its counterfactual is absent"; M3's sealed counterfactual is "order stays at 4-5 while `h` collapses" | no counterfactual is evaluated anywhere | the counterfactual is **measured present** — trials at order 4 with `h` = 3.70e-2, 1.85e-2, 9.25e-3 — so M3 fails on its own sealed terms |
| 3 | `INCONCLUSIVE` is triggered by "mixed or missing signature, non-reproduction, **or a cap reached first**" | `decision` is a pure function of `positives`; caps and reproduction are never consulted | **both** phases stopped on caps (`cycle_cap:3`, `call_cap:600`), so the literal sealed rule returns `INCONCLUSIVE` |

M2, M4 and M5 lost whole conjuncts the same way, and M4's sealed ratio was changed from `>1x` to
`>3x`. Those three are outcome-neutral here, but they are the same defect.

**The governance lesson, which is the durable one.** The instrument carried a *mechanical* verbatim
check binding the borrowed SciPy method and the borrowed repository right-hand side to their pinned
originals — and it worked, catching nothing because nothing was wrong there. No equivalent check
bound the writer's **own sealed decision criteria** to the code that evaluated them, so the criteria
were re-typed, silently weakened, and then printed under a column header reading "Sealed threshold".
A seal that only a human compares to its implementation is not a seal. This is the same class of
defect the generated-status architecture (D-073) was built to remove for gate status, reappearing in
a place that architecture does not cover.

## 4. Corrected status of every V1 claim

| V1 claim | Corrected status |
|---|---|
| "ten eigenvalues with positive real part ... the control has none" | **REFUTED.** Recomputed: 0 and 0. Both configurations are Hurwitz at their snapshots (−6.43 and −6.58). |
| "every unstable mode localises at y > 24" | **VOID** — there are no unstable modes. The localisation arithmetic on the defective matrix was correct; its subject does not exist. |
| "four complex conjugate pairs" | **VOID and was a non-sequitur anyway** — all four sat in the stable half-plane at 0.22 degrees from the negative real axis, inside the wedge. |
| decision `PASS_DIAGNOSTIC`, mechanism M3 | **WITHDRAWN.** Corrected letter: `INCONCLUSIVE`, on three independent grounds plus the refuted spectrum. |
| "0.1813 was never an accepted state; it was a failed Newton trial" | **VALIDATED.** Maximum accepted `t` is 0.16282186; 0.1813 appears once, as a Newton failure at raw call 904. |
| "the r4 log's later 0.1629 is the last accepted state printed to four decimals" | **REFUTED — arithmetic error.** ln(10/8.49696) = 0.16287664, which is 5.478e-05 **past** the last accepted state 0.16282186 and prints differently. It is a distinct, later state, not a reprint. The "drop" is therefore still not fully accounted for. |
| "error-test failures were zero in both runs" | **VALIDATED, and stronger than stated** — the maximum error norm ever seen was 0.131 (domain) and 0.170 (base) against a threshold of 1. |
| "the stalled run's error never concentrates near the origin" | **REFUTED.** Node 0 is among the top five error components in 21 of 55 domain error tests. The record's own §7b table prints `argmax_y_min` = 0.01185, and the renderer's top-6 truncation hid the contradicting row. |
| "compared at the same physical epoch" | **REFUTED.** The snapshots are at N = 0.1258 versus 0.1881, `h` 0.0220 versus 0.0408, BDF order 3 versus 4, and hence different A(alpha) thresholds. |
| "both runs reproduce the retained r4 log exactly" | **VALIDATED but far weaker than "exactly"** — 4-decimal N and 5-decimal T_cm at every-50th-call indices, 19 of 25 domain points, only three distinct non-trivial states, and the six skipped points are 951-1201, precisely the stall window. The comparison silently skips unreachable indices rather than counting them. |
| "3 hard Newton failures against the control's 0" | **NOT A RATE.** Three is the sealed stopping rule; the domain phase was stopped *by* the third. The control was given `cycle_cap = None`. The windows are asymmetrically censored and the ratio is an artefact of that asymmetry. |
| "every figure is printed from ANALYSIS.json; none is hand-transcribed" | **FALSE** for sections 1, 2 and 3, which the renderer never touched. |
| "the refusal and its payload are retained" (FAIL_PROTOCOL) | **FALSE.** No `FAIL_PROTOCOL.json` exists in the retained set; the relaunch overwrote the output root. |
| section 11's "~4,600x collapse in N advance per solver time point" | **RETIRE.** Its denominator counted values printed every 50 raw calls, not solver time points; V1 measures the conversion factor at 14.5x and 13.8x, and it is not common-mode across the epochs compared. What survives from r4 bytes alone is 7,080x (whole base run versus creep, per raw call) or 3,614x (matched-N base segment versus creep). |

## 5. What survives, and is stronger after the correction

The tail block is **not unstable. It is fast and stable** — which is the structure that asymptotic
and slaving methods exist for. Partitioning the corrected 60/30 spectral block at y = 24:

| Corrected 60/30 operator | dim | spectral abscissa | positive eigenvalues | stiffness ratio | 2-norm |
|---|---|---|---|---|---|
| full spectral block | 180 | −6.4271 | 0 | 5.588e+07 | 1.299e+10 |
| `y <= 24` block `Jkk` | 126 | −6.4945 | 0 | 7.981e+05 | 5.210e+07 |
| `y > 24` block `Jdd` | 54 | **−3.2735e+03** | 0 | 1.081e+05 | 1.198e+09 |
| Schur complement `S = Jkk − Jkd Jdd⁻¹ Jdk` | 126 | **−6.4271** | 0 | **1.365e+03** | **2.983e+05** |
| base 48/24 control, whole block | 144 | −6.5805 | 0 | 3.294e+05 | 5.251e+07 |

Three measured facts follow, and none of them depends on the withdrawn claim:

1. **The 60/30 operator is about 170 times stiffer than the 48/24 control** (5.59e7 versus 3.29e5).
   That difference is real, and it is a stiffness difference, not an instability.
2. **The stiffness is carried by the `y > 24` block**, which is strongly damped
   (abscissa −3.27e3, norm 1.20e9) — fast, stable, and contributing nothing to the slow dynamics.
3. **Algebraically slaving that block reproduces the slow spectral abscissa exactly** (−6.4271
   versus −6.4271) while reducing the stiffness ratio by a factor of **4.1e4** and the operator norm
   by **4.4e4** — landing 240 times *below* the healthy control's stiffness.

That is a quantitative, reproducible motivation for a reformulation in D-071's sanctioned class (a),
and it is the only positive result of this exercise that survived audit.

## 6. What is still not known

**The stall was never observed.** The r4 creep runs from evaluation 951 to 11,051; the V1 domain
phase stopped at 907 on its sealed cycle cap. Everything V1 measured is from *before* the stall. In
the window it did cover, the two configurations differ by 1.51x in raw calls per accepted step and
1.155x in N advance per accepted step — nothing that projects to a 3.3e7-evaluation miss. **The
mechanism of the stall remains unmeasured**, and the sealed cap is why.

## 7. Revised plan

### 7.1 Immediately, at zero compute

- This document, superseding the V1 record. **Done here.**
- Retire section 11 of the brainstorm design document and the 4,600x figure with it.
- Record the finite-difference rule of §2 as a standing constraint on any future spectral claim in
  this codebase.

### 7.2 Trajectory families, revised

| Family | Revised state and why |
|---|---|
| H-T1 diagnosis | **Executed and INCONCLUSIVE.** It answered a different question than the one it was built for, because its cap bound before the stall. Re-running it in the creep window is a *new* owner decision with a real cost (the creep begins near evaluation 5,000; reaching it costs roughly 6-7 hours of wall) and it is **not** authorised here. |
| H-T2 asymptotic reformulation | **Promoted to the leading candidate**, on §5. The specific sub-route with measured support is algebraic slaving of the fast tail block, not the plain delta-f variant. Still requires its own derivation, fidelity ladder, sealed contract, and reviewed discriminator; none exists. |
| H-T3 analytic domain reduction | **Weakened to unsupported.** Its V1 support is void, and the adversarial reviewer's original challenge stands unanswered: the target configuration 60/24 has first node 0.009479, finer than both configurations that were run, so the near-origin axis was never bracketed. Its three-channel bound is also mis-specified — the binding channel is operator feedback, measured at 1034 times the direct tail contribution. |
| H-T4 defect plus log-norm certificate | **Not killed by the abscissa any more** (the corrected abscissa is negative), but the planned V2A screen as written is dead: no *diagonal* weighting gives a negative logarithmic norm even on the healthy control. Only a full Lyapunov metric does, at amplification 8e3 to 2e4, which is far too loose for a 2e-4 band. Treat as effectively closed unless someone wants to pay for the Lyapunov route. |
| H-T5 pathology framing | **Withdrawn.** There is no measured pathology. The corrected reading is that the 60/30 configuration is much stiffer, not sick. |

### 7.3 Harness families — unchanged, and untouched by any of this

Nothing in the retained bytes bears on any of the eight `G-HARNESS-INTEGRITY` conjuncts. H-H1 and
H-H4 stay `FORBIDDEN` under D-074, H-H3 stays a rejected shortcut, H-H2 stays the only route to
judgment independence, H-H5 stays the operational default. **The one harness-relevant output of this
exercise is §3's governance lesson**, and acting on it would itself be new machinery under a freeze,
so it is recorded rather than built.

### 7.4 The next decision, stated as a choice rather than a plan

Three options, and the owner picks at most one:

1. **Stop.** The trajectory gate is closed on a preserved measurement, the diagnostic came back
   inconclusive, and the strongest surviving result (§5) is a motivation, not a method. Nothing
   requires further spend.
2. **Pay for the missing measurement.** Re-run the diagnostic with the cap moved into the creep
   window. Cost roughly 6-8 hours, sealed protocol first, with §3's defects fixed — in particular
   the sealed predicates must be compared to the implementation mechanically, not by reading.
3. **Develop the slaving reformulation on paper.** Zero compute. Derive the algebraic tail closure,
   its fidelity budget against the 2e-4 band, and a bounded discriminator, then seek review before
   any implementation. This is the only path that could satisfy D-071's reopen conditions, and it
   starts with a derivation, not a run.

**Recommendation: option 3, or option 1.** Option 2 buys a diagnosis of an instrument that D-071 has
already closed, and a diagnosed pathology would not license reopening it in any case.

## 8. Cost

```text
tracked files changed        1 (this document)
production/harness/gate      0
frozen module                unchanged, 760a7c04...
audit compute, UNBUDGETED    188 RHS evaluations, about 840 s -- recomputing the Jacobian at a
                             converged step. V0 budgeted no audit compute; this is recorded as an
                             overrun, not hidden. It was spent to settle a question two agents
                             disagreed about, and arbitrating between agents is not evidence.
runtime_behavior_changed     no
physics_behavior_changed     no
known_blocker_reduced        no -- a wrong conclusion was withdrawn and one motivation survives
blocker_movement_ratio       0
cost_effectiveness_verdict   ACCEPT -- the withdrawn claim would have misdirected the next stage
```

## 9. What is not concluded

Nothing about the physical proposition. `G-F10-INDEPENDENT-FLRW` stays FAIL on its preserved
measurement, `G-HARNESS-INTEGRITY` stays FAIL under the freeze, no reopen condition is met, and no
evidence package exists. The corrected result is that the diagnostic did not diagnose the stall.
