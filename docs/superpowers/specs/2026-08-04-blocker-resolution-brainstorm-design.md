# Blocker-resolution plan — critical re-audit and bounded decision design

**Date:** 2026-08-04

**Revision:** critical design re-audit of `b24d98b`

**Review verdict:** `MAJOR_REVISIONS`

**Status:** `SPECIFIED` for owner decision only. This document is not an execution grant, a sealed
contract, an accepted evidence package, or a gate decision.

The board remains **6 PASS / 2 FAIL**:

- `G-F10-INDEPENDENT-FLRW=FAIL`, closed on the current instrument and measurement by D-071;
- `G-HARNESS-INTEGRITY=FAIL`, with harness development frozen fix-critical-only by D-074.

Recommended owner disposition:

1. **Trajectory:** reject V2 as originally costed; if further causal localization is worth at most
   two hours, authorize only the bounded, scratch-only V1 diagnostic defined below. Its output is
   diagnostic evidence, never gate or reopen evidence.
2. **Harness:** keep the gate FAIL and the harness frozen. V4 external-adjudicator recruitment is an
   optional owner action outside the repository; it cannot itself move the gate or authorize H-H1/H-H4.
3. **After V1:** select at most one materially new scientific family for contract design. Do not run
   H-T2, H-T3, or a full V2 screen in parallel merely to keep options open.

---

## 1. Reviewer verdict

### Strengths worth preserving

- The proposal separates trajectory and harness failure classes.
- It treats reviewer multiplicity under one operator as computational diversity, not independent
  governance.
- It includes pass, kill, and ambiguity outcomes instead of defining every result as progress.
- It keeps QKE, public-production claims, frozen-module edits, and gate movement out of scope.
- H-T3 already carries a useful self-destruct clause: without a three-channel bound it is only an
  easier configuration of the same instrument.

### Blocking corrections before any execution

| ID | Problem in `b24d98b` | Why it is load-bearing | Required correction |
|---|---|---|---|
| R-01 | RHS progress points were treated as accepted BDF states. | `make_rhs()` logs every raw RHS call, including finite-difference Jacobian columns and failed Newton/error-test trials. The retained log has no accepted-step trace. | Treat the `0.1813 -> 0.1629` sequence as evidence of trial-point retreat consistent with rejection, not proof of the accepted-state chronology. |
| R-02 | `10100 ~= 55.5 * 182` was promoted to a causal explanation. | The arithmetic is consistent with repeated dense finite-difference batches, but r4 retained no `njev`, `nlu`, accepted-step, Newton-failure, or error-test counters. | Keep FD refresh/rejection cycling as `PROPOSED` until V1 records the solver events directly. |
| R-03 | V2 assumed about 1,000 retained checkpoints and a one-RHS-call Jacobian cost. | The report contains exactly 31 base checkpoints of shape `3 x 48`, plus two scalar states. One dense 146-state finite-difference Jacobian requires at least 147 raw RHS calls; 31 require at least 4,557 calls before retries. | Split V2 into a zero-run mathematical admissibility screen and a three-checkpoint local kill screen. Do not call it cheap or rigorous. |
| R-04 | The plan said V1/V2 required no contract or unfreeze. | D-071 closes the trajectory lane; diagnostic work may be proposed, but it needs a separate owner grant and may not be reused as reopen evidence. | Add an explicit authorization state and a pre-output diagnostic protocol. A future reopen discriminator still needs all D-071 conditions. |
| R-05 | Literature and reviewer claims were described as primary-source verified, but no durable source or reviewer-result path was cited. | Session task outputs and prose summaries are not independently auditable repository evidence. | Make literature contextual only until exact source/page/claim mappings are retained. Treat reviewer verdicts as `SUMMARY_ONLY` unless a stable path and digest are supplied. |
| R-06 | H-H1/H-H4 were scheduled after V4 despite D-074. | V4 success is not one of D-074's exact harness-reopen triggers. New CI/anchoring machinery would restart the frozen architecture by implication. | Mark H-H1/H-H4 `FORBIDDEN` now. They require a D-074 trigger or an explicit superseding owner decision. |

### Scope verdict

The useful minimum claim is not “the blocker is nearly solved.” It is:

> The retained trace is compatible with a reject/finite-difference-refresh pathology, but does not
> identify its cause. One bounded diagnostic can test whether a materially new method deserves a
> prospective contract. The current gate remains closed.

That is scientifically useful and proportionate. Anything stronger is premature.

## 2. Claim and evidence audit

### 2.1 Local retained evidence

| Claim ID | Status | Evidence | Adjudicated wording |
|---|---|---|---|
| F-R1A | `VALIDATED` | `scripts/audit/_trajectory_core.py:181-221` | Progress lines are emitted inside the RHS every 50 raw calls. They are not an accepted-step log. |
| F-R1B | `DERIVED` | r4 lines around evals 751 and 951 plus SciPy BDF control flow | The retreat is consistent with a failed trial followed by a smaller trial. Whether either printed point was accepted is `UNVERIFIABLE` from retained bytes. |
| F-R2A | `DERIVED` | 182-state domain system; SciPy 1.17.1 dense `num_jac`; repeated same-`N` blocks | Dense FD work is a plausible dominant contributor. One refresh invokes at least 183 raw RHS calls before any retry. |
| F-R2B | `PROPOSED` | arithmetic coincidence only | “The wall is almost entirely reject-refresh cycling” is a hypothesis for V1, not a finding. |
| F-R3A | `VALIDATED` | r4 base report | The 48/24 base completed in 3,694 raw RHS calls. |
| F-R3B | `DERIVED` | rounded RHS trial log | The 3.27e7 / 4.58-year estimate is conditional on treating trial-point movement as a progress proxy. It remains an order-of-magnitude impracticality argument, not an accepted-state completion forecast. |
| F-R4 | `VALIDATED` | r4 JSON key/count inspection | `domain_holdout` is null. Only 31 completed-base checkpoints survive; stalled-phase state, accepted-step history, and per-evaluation rejection history do not. `base.rejections` is only an aggregate `{min,max,final,count}`. |
| F-R5 | `SPECIFIED` | local driver and installed SciPy source | A scratch diagnostic can leave `_independent_noqke.py` unchanged, but solver failure counters require a version-pinned diagnostic adapter around SciPy internals; public `solve_ivp` output is insufficient. |

Minimal local checks:

```bash
jq '{domain_holdout,
     checkpoints:(.base.checkpoint_states|length),
     state_shape:[(.base.checkpoint_states[0].cloglog|length),
                  (.base.checkpoint_states[0].cloglog[0]|length)],
     rejections:.base.rejections}' \
  .agent-harness/runs/run-20260729-f10-d069-trajectory-r4/raw_logs/r4_trajectory_report.json

nl -ba scripts/audit/_trajectory_core.py | sed -n '181,221p'
nl -ba venv/lib/python3.12/site-packages/scipy/integrate/_ivp/base.py | sed -n '138,170p'
nl -ba venv/lib/python3.12/site-packages/scipy/integrate/_ivp/bdf.py | sed -n '328,405p'
```

### 2.2 Literature and reviewer evidence boundary

The broad survey remains useful for hypothesis generation, not for a decision-bearing runtime claim.
In particular:

- Bennett et al. support precision and momentum-discretization convergence work, but that alone does
  not establish the proposal's universal desktop-runtime or lifetime-evaluation ranges:
  <https://arxiv.org/abs/2012.02726>.
- CODECHECK defines independent execution of an author-provided workflow and explicitly has the
  codechecker record rather than investigate or fix. It is a plausible execution witness, not by
  itself a gate adjudicator: <https://codecheck.org.uk/project/>.
- The FortEPiaNO, NEVO, nudec_BSM, STALD, AP/BGK, logarithmic-norm, validated-integrator, and CLASS
  claims are `SUMMARY_ONLY` here until the exact primary source, version, page/section, and supported
  sentence are recorded in this document or an already-authoritative repository source.
- The two adversarial reviewer reports are also `SUMMARY_ONLY` because “session task outputs” is not
  a durable evidence locator. Their corrections are retained as design improvements, not counted as
  independent validation.

No canonical claim-ledger status changes follow from this document. All candidate-family claims stay
`PROPOSED` or `SPECIFIED`.

## 3. Controlling authority and non-negotiable boundaries

| Authority | Binding consequence for this plan |
|---|---|
| D-071 | The current 60/30 instrument and measurement remain closed. Reopen requires a materially new method, a prospectively sealed contract before output, and a reviewed bounded discriminator covering the stalled phase with end-to-end margin. |
| D-074 | `.agent-harness/` and `.codex/hooks/` are fix-critical-only. V4 recruitment does not authorize new CI, anchoring, guards, declarations, or checkers. |
| D-075 | Four D-065 obligations are discharged, but they are not the eight-conjunct gate. Operator/decision independence and an accepted evidence package remain absent. |
| Anti-drift guardrail | A tracked diagnostic/telemetry wrapper is forbidden unless it directly moves a measured blocker or deletes/consolidates more obsolete surface than it adds. Rust AOT remains the active repeated-run target; SciPy/BDF is only the temporary number-of-record. |
| Scope | QKE, public dispatch, public-production support, W7/B3, unblinding, and frozen-module edits remain forbidden. Raw failures must be preserved. |

Authorization state:

| Action | State now | What would authorize it |
|---|---|---|
| Read-only evidence correction and this design revision | `SPECIFIED` / allowed | Current owner request |
| V4 recruitment outreach | `PROPOSED`, owner action | Owner chooses qualification criteria and deadline |
| V1 diagnostic execution | `NOT_AUTHORIZED` | Separate explicit owner grant accepting the hard cap and diagnostic-only claim ceiling |
| V2A mathematical screen | `PROPOSED` | Owner authorizes design-only work; no numerical output |
| V2B numerical local screen | `NOT_AUTHORIZED` | V2A passes and owner accepts measured cost cap |
| V3 prefix probe or H-T2 fidelity run | `FORBIDDEN` now | All D-071 reopen prerequisites, including a sealed reviewed contract |
| H-H1 CI or H-H4 anchoring machinery | `FORBIDDEN` now | A D-074 reopen trigger or an explicit decision superseding D-074 |

## 4. Candidate-family disposition

### 4.1 Trajectory lane

| Family | Revised state | Retain / reject rationale |
|---|---|---|
| H-T1 bounded causal diagnostic | `SPECIFIED`, conditional | Retain only as scratch-only diagnostic V1. It can localize a solver mechanism but cannot reopen or move the gate. |
| H-T2 micro-macro/AP reformulation | `PROPOSED` | Retain conditionally. Plain delta-f may reduce cancellation but does not remove fast modes; a materially new micro-macro/AP closure needs its own derivation and fidelity ladder. |
| H-T3 analytic domain reduction | `PROPOSED` | Retain conditionally. The three-channel tail bound must be completed before a run; otherwise self-destruct. |
| H-T4 defect/log-norm certificate | `PROPOSED`, redesign required | Do not call the current V2 cheap or rigorous. First define coordinates, norm, output functional, state coverage, and tube argument; then use local samples only as a kill screen. |
| H-T5 pathology interpretation | `PROPOSED` framing only | Keep as a hypothesis to falsify, not a separate execution family. |

### 4.2 Harness lane

| Family | Revised state | Retain / reject rationale |
|---|---|---|
| H-H1 external CI/attestation | `FORBIDDEN` now | Cannot cover trusted local hook conjuncts and would add machinery during D-074 freeze. |
| H-H2 external human adjudication | `PROPOSED` | The only surviving route to judgment independence, but still insufficient without all eight conjuncts and an accepted evidence package. |
| H-H3 second local UID | `FORBIDDEN` shortcut | It does not create operator or decision independence and changes the frozen threat model. |
| H-H4 transparency anchor | `FORBIDDEN` now | Existence-at-time is not content honesty or claim support; unpaired anchoring is ceremony. |
| H-H5 retain terminal FAIL | current default | This is the controlling operational state, not a theorem that independent review can never occur. |

## 5. Revised trajectory verification design

### V0 — pre-output diagnostic protocol

V0 produces no scientific output. Before V1, record in one owner-approved protocol:

1. exact input commit and frozen-module SHA-256;
2. Python, NumPy, and SciPy versions plus SHA-256 of the copied diagnostic BDF source;
3. exact domain setup (`order=60`, `y_max=30`, `rtol=1e-6`, `atol=1e-9`);
4. mechanism predictions and their counterfactual signatures;
5. raw-RHS, wall, and accepted-step hard caps;
6. output paths and mandatory failure preservation;
7. the statement: **V1 output cannot satisfy any D-071 reopen conjunct**;
8. tracked-file budget: zero production/harness/gate files and no persistent diagnostic framework.

If this protocol is not fixed before the first run byte, V1 must not run.

### V1 — bounded accepted-step diagnostic

**Purpose:** decide whether the retained plateau is dominated by finite-difference refresh/rejection,
error-control noise, Newton failure, a kinematic discontinuity, or a genuinely fast mode.

**Implementation boundary:** run only the 60/30 domain phase in a disposable scratch tree. Use the
frozen comparator's existing public functions. Instrument a copied, version-pinned SciPy 1.17.1 BDF
implementation; do not edit the installed environment, frozen module, repository driver, or harness.

**Mandatory observations:** for every attempted step and accepted step, retain:

- `t_old`, proposed `t_new`, accepted `t`, `h_abs`, order, and acceptance boolean;
- `nfev`, `njev`, `nlu`, Newton iteration count, Jacobian-refresh event, and failure class;
- scaled error norm using `atol + rtol*abs(y)` component by component;
- raw RHS-call count separately from SciPy `nfev` because FD calls are excluded from `nfev`;
- kinematic-domain rejection count and largest roundoff correction per raw RHS call;
- gain, loss, and net collision terms for a predeclared small component set, sufficient to compute a
  cancellation ratio without dumping a new general telemetry surface;
- one accepted-state snapshot immediately before the first reproduced reject-refresh cycle and one
  dense Jacobian/eigenvalue calculation at that accepted state.

**Predeclared signatures:**

| Mechanism | Required positive signature | Counterfactual that rejects it as dominant |
|---|---|---|
| Dense-FD refresh cycle | accepted `t` stalls while `njev` increments and a >=183-call same-trial batch follows | accepted progress continues without refresh-aligned call bursts |
| Newton cycling | repeated `converged=false`, refresh/retry, and reduced `h_abs` before any error test | Newton converges and rejection is error-test-only |
| Error/noise floor | Newton converges, scaled error remains >1, and gain/loss cancellation predicts component noise at or above the tolerance scale | scaled residual is well separated from the tolerance floor |
| Kinematic discontinuity | rejection/refresh events co-locate with a jump in the existing kinematic-domain counter and named components | counter and component support remain smooth through failures |
| Stability/fast mode | converged local Jacobian spectrum quantitatively requires the observed step restriction under the actual BDF order | spectrum is compatible with much larger stable steps and another counter explains the failures |

**Hard cap:** stop at the first of:

- 1,200 integration raw RHS calls, plus at most one post-capture dense Jacobian whose own budget is
  capped at 366 raw calls including finite-difference retry columns;
- two hours of total domain-phase wall time, including that Jacobian;
- two complete reject-refresh cycles after the first accepted-state snapshot.

The retained first episode occurs before raw eval 1,100, so extending beyond the cap because the
signature is inconvenient is forbidden.

**Decision:**

- `PASS_DIAGNOSTIC`: one predeclared mechanism accounts for more than half of raw RHS calls or wall
  inside the captured cycle and its counterfactual is absent;
- `INCONCLUSIVE`: mixed or missing signature, non-reproduction, or cap reached first;
- `FAIL_PROTOCOL`: missing counter, changed configuration, unsealed protocol, or lost raw failure.

No result is a gate pass. `PASS_DIAGNOSTIC` only selects which new-method contract, if any, is worth
drafting. `INCONCLUSIVE` stops this diagnostic family; no 13-hour extension follows automatically.

### V2A — zero-run mathematical admissibility screen

Before any Jacobian evaluation, H-T4 must specify:

1. the exact 146-state base operator and coordinates in which the defect is measured;
2. the output functional mapping state error to `Delta N_eff` error;
3. at most three physics-derived, predeclared weighted norms;
4. how discrete checkpoint values control an inter-checkpoint tube;
5. how the completed 48/24 base trajectory can support a claim about a replacement method without
   being misrepresented as stalled 60/30 state evidence;
6. a sign-independent error-budget allocation fixed before numerical values are seen.

If any item is missing, H-T4 is killed before compute. A local matrix measure is not a rigorous
trajectory certificate.

### V2B — local log-norm kill screen, only if V2A passes

The retained report has 31 base checkpoints, not 1,000. The first screen uses the fixed
earliest/middle/latest rows (`N=0.25`, `4.0`, `7.75`) and no adaptive checkpoint selection.

Cost facts:

```text
base state dimension                    = 3*48 + 2 = 146
minimum raw RHS calls / dense Jacobian  = 147
minimum for three-point screen          = 441
minimum for all 31 retained points      = 4,557
31-point wall at 4.42 s/raw RHS         = about 5.6 h before retries
```

Measure one Jacobian's actual wall before authorizing the other two. Stop if the three-point screen
kills every predeclared norm or if measured full-screen cost exceeds the owner cap. Passing three
points means only `NOT_KILLED_LOCAL`; it does not establish negativity across the epoch, a tube bound,
or a `VALIDATED` certificate. A later 31-point screen and tube proof require a separate decision.

### V3 — analytic-domain route, conditional on V1

V3 remains paper-first:

1. V1 must identify a driver absent from the proposed alternative construction.
2. Before output, derive separate bounds for equilibrium storage, distortion tail, and operator
   feedback, and fix how they combine with the existing numerical uncertainty budget.
3. Failure to bound any one channel kills H-T3. Do not run a density-only easier configuration.
4. Only a prospectively sealed and independently reviewed contract may authorize the prefix probe.
5. The probe must use accepted-step progress and include setup/Jacobian costs; RHS trial movement is
   not a completion-rate estimator.
6. Projected end-to-end wall must be <=50% of the frozen budget. A 50-100% projection is ambiguous
   and does not authorize a full run; >100% kills the route.

### H-T2 — reformulation route, conditional on V1

H-T2 is eligible for contract design only if V1 identifies a mechanism the reformulation removes.
Plain delta-f is not credited with stiffness removal. A micro-macro/AP proposal must state its
equilibrium manifold, conserved moments, fast-mode elimination argument, structural-independence
boundary, and a convergence ladder whose fidelity budget is allocated before output. It competes
with V3; both are not implemented in parallel by default.

## 6. Revised harness decision design

### V4 — optional external-adjudicator recruitment probe

V4 is an owner outreach action, not a repository implementation.

Before outreach, the owner fixes:

- qualification criteria, conflicts of interest, and whether the candidate may have contributed to
  the code or prior decisions;
- a deterministic recruitment order or named candidate, compensation if any, response window, and
  what counts as non-response;
- authority to return PASS, FAIL, or INCONCLUSIVE without the writer editing the verdict;
- the exact commit/evidence bundle and an eight-conjunct coverage matrix classifying each conjunct as
  `REEXECUTED`, `RETAINED_BYTES_ADJUDICATED`, or `NOT_COVERED`;
- publication/confidentiality terms and the rule that the writer may package but not rewrite the
  signed verdict.

The coverage matrix must enumerate the gate text rather than summarize it:

1. validator and v2 fixtures exit zero;
2. current registered assignment, exact five-field header, and pre-spawn non-run hashes;
3. trusted-session Start context and runtime identity injection;
4. runtime/assignment type agreement plus review-role, role-file, and result-template verification;
5. blocked invalid first Stop;
6. automatically accepted corrected assignment-hash-bound result;
7. external-tool confinement outside the repository;
8. post-run hashes proving result-only subagent writes.

Outcomes:

- `RECRUITED_WITH_JUDGMENT_AUTHORITY`: proceed to scope negotiation over existing bytes only;
- `EXECUTION_WITNESS_ONLY`: useful CODECHECK-style reproducibility evidence, but judgment independence
  and uncovered live-hook conjuncts remain open;
- `NO_ACCEPTOR_WITHIN_WINDOW`: evidence that this route is unavailable under current resources, not
  proof that independent judgment is impossible. Retain terminal FAIL and stop.

V4 success does not move the gate. Movement would still require all eight gate conjuncts, independent
judgment over the declared scope, and an accepted D-073 evidence package. V4 success is not a D-074
reopen trigger.

### H-H1 and H-H4 — no automatic follow-on

Do not add greenfield CI, a monotone-run index, Rekor/OSF anchoring, a new declaration file, or a
conjunct checker merely because V4 succeeds or fails. Under D-074 those components remain
`FORBIDDEN` unless an exact reopen trigger is reproduced or the owner explicitly supersedes the
freeze. An external reviewer can receive an immutable Git archive and existing hashes without first
building another repository-local assurance layer.

## 7. Sequencing and candidate selection

Do not describe V1, V2, and V4 collectively as three cheap parallel heads. V1 is bounded compute,
V2 was materially under-costed, and V4 is external coordination.

```text
default
  -> preserve D-071 trajectory closure
  -> preserve D-074 harness freeze and terminal FAIL

optional owner actions
  -> V4 recruitment probe (outside repo; no gate/harness effect)
  -> V0 protocol -> V1 bounded diagnostic (only under separate owner grant)
  -> V2A paper screen (independent of V1, zero numerical output)

after evidence, never by default
  V1 mechanism result -> choose at most one of H-T2 or V3 for a D-071 contract
  V2A pass -> V2B three-point local kill screen under a measured cost cap
  V4 recruited -> external scope negotiation over existing bundle
```

Cost-effective candidate comparison:

| Candidate | Tracked files / net lines | Expected gain | Main risk | Minimal discriminator |
|---|---|---|---|---|
| V1 | 0 / 0; scratch only | causal localization, at most BMR 0.25 | mixed signature; SciPy-internal instrumentation error | <=1,200 raw calls or <=2 h |
| V2A | this design only | kill an unsound certificate before compute | no usable norm/tube map | six-item mathematical admissibility review |
| V4 | 0 / 0 | establish whether judgment authority is practically available | execution witness mistaken for adjudicator | frozen qualification/scope plus response window |

## 8. Evidence contract for any authorized diagnostic

Do not create another generic manifest. The one authorized run record must contain:

- input commit, frozen-module digest, interpreter/library versions, diagnostic-source digest;
- exact command, cwd, environment overrides, start/stop/wall, exit, stdout, and stderr;
- every predeclared counter and raw negative/failure state;
- accepted-state and trial-state data explicitly separated;
- observed result, kill/ambiguity decision, and all violated assumptions;
- tracked-file pre/post status proving zero production/harness/gate edits;
- explicit claim ceiling and `implies_status: null`.

Missing required evidence makes the diagnostic `FAIL_PROTOCOL`, not “partially informative.”

## 9. Stopping and reopening rules

### Trajectory

- V1 ends at its first hard cap. No automatic rerun, tolerance tuning, order cap, analytic Jacobian,
  faster host, or larger budget.
- V2A failure ends H-T4. V2B local survival does not authorize a full screen or certificate.
- V3 ends before runtime if any tail channel lacks a bound.
- H-T2/V3 implementation begins only after one family wins an explicit blocker-movement-per-line
  comparison and all D-071 reopen prerequisites are met.
- Any future benchmark must cover the previously stalled phase and accepted-state progress. A kernel
  or cheap-segment speedup is not endpoint progress.

### Harness

- `G-HARNESS-INTEGRITY` remains FAIL unless an accepted package supports every required conjunct and
  independent decision authority adjudicates the declared scope.
- Failed recruitment preserves the current operational stop; it does not create a new gate or proof
  of impossibility.
- H-H1/H-H4 remain frozen absent a D-074 trigger. No “small preparatory” machinery is exempt.

### General

- Preserve raw failures and do not refit bands, norms, checkpoint choices, or mechanism labels after
  output.
- A proposed action whose strongest claim would remain true if the physics RHS were removed is
  governance work, not trajectory-blocker movement.
- Every future PR must report the cost line required by
  `bbn_codex_anti_drift_cost_effective_policy.md`.

## 10. Claim ceiling and current cost

This revision changes the design record only. It does not execute a diagnostic, validate a
reformulation, recruit an adjudicator, create an evidence package, discharge a new obligation, or
move either gate. Canonical `docs/harness/CLAIM_LEDGER.md` rows therefore remain unchanged.

```text
added_lines: 413
deleted_lines: 250
net_lines: 163
files_touched: 1
token_use_exact: UNAVAILABLE
token_use_basis: no reliable task-scoped token counter is exposed
runtime_behavior_changed: no
physics_behavior_changed: no
known_blocker_reduced: no -- design defects and cost error corrected only
blocker_movement_ratio: 0.00
validation_strengthened: yes -- proposal now has discriminating authority, cost, and stop boundaries
cost_effectiveness_verdict: ACCEPT_WITH_LIMITS
```

The next permissible project action is an owner choice, not an inferred continuation.

---

## 11. Post-review measurement addendum — plateau structure of the retained trace

**Author:** writer, after the re-audit. **Inputs:** retained bytes only
(`r4_trajectory_stdout.log`, sha256 `28c541b3…`, and `scripts/audit/_trajectory_core.py:181-221`).
**No run, no contract, no gate, no authorization change.** V1 stays `NOT_AUTHORIZED`.

This addendum exists because the re-audit demoted F-R2B to `PROPOSED` on the grounds that only
total-count arithmetic supported it (R-02). That demotion was correct, and re-measuring produced two
results the re-audit did not have — one of which **refutes the writer's own `b24d98b` claim**, and one
of which **falsifies a row of §5's V1 signature table**.

### 11.1 Method

The RHS logs its time argument every 50 raw calls (`_trajectory_core.py:216`). A dense finite-
difference Jacobian perturbs `y` at fixed `t`, so every call inside one FD batch prints the *same*
`T_cm`. Grouping consecutive log rows by identical printed `T_cm` therefore separates solver time
points, and the run-length of each group measures raw calls spent at one time point.

### 11.2 Results

| Phase | State dim | FD batch (calls) | Raw calls | Distinct `t`-plateaus | Median plateau width | Rows inside multi-row plateaus |
|---|---|---|---|---|---|---|
| base 48/24 (**completed**) | 146 | 147 | 3,651 | 29 | 150 calls | 66/74 (89%) |
| domain 60/30 (**stalled**) | 182 | 183 | 11,051 | 56 | 200 calls | 216/222 (97%) |

**Result 1 — the FD signature is structural, not arithmetic.** Median plateau width tracks state
dimension independently in both phases (147→150, 183→200 under 50-call logging quantization). This is
a second, structural signature, distinct from the total-count coincidence R-02 rightly rejected.

**Result 2 — this refutes the writer's `b24d98b` framing.** The base phase, which *completed* in
2.7 hours, shows the same FD-dominated structure. **Dense-FD Jacobian dominance is this instrument's
normal cost structure, not the pathology.** The `b24d98b` sentence "the creep's wall is almost
entirely Jacobian refreshes", offered as a pathology finding, is equally true of the healthy run and
therefore discriminates nothing.

**Result 3 — what actually differs is step size, by three to four orders of magnitude.**
`N` advance per distinct solver time point:

```text
base 48/24, completed              2.143e-01  (ΔN=+6.2145 over 29 time points)
domain 60/30, whole phase          2.952e-03  (ΔN=+0.1653 over 56 time points)
domain 60/30, post-drop creep      4.615e-05  (ΔN=+0.0024 over 52 time points)
collapse ratio, base : creep       ~4,600x
```

Per-time-point *cost* rose only ~1.6× (126→197 raw calls), consistent with the dimension increase
alone. What collapsed is the step actually taken.

### 11.3 Consequences

- **F-R2B's causal claim is unchanged and stays `PROPOSED`.** Nothing here says *why* `h` collapsed.
  Accepted-step, Newton, error-test, order, and `njev`/`nlu` records remain unretained.
- **V1 remains necessary, and its target sharpens.** The discriminating question is not "are Jacobian
  refreshes frequent" — they are, in the healthy run too — but "why is `h` ~4.6e3× smaller per solver
  time point". Error norm, Newton outcome, and order are what measure that.
- **§5's V1 signature table, row 1, is falsified as written.** Its positive signature ("a ≥183-call
  same-trial batch follows") and its counterfactual ("accepted progress continues without
  refresh-aligned call bursts") both fire on the *completed* base phase. Before any V1 grant, that row
  must be re-specified as refreshes **per accepted step**, or dropped. The other four rows are
  untouched by this measurement.

### 11.4 Limitations

- 50-call logging quantizes plateau widths; 147 and 183 both fall in the 150/200 bins, so the match is
  *consistent with* FD batches, not a unique fit.
- Consecutive refreshes at the same trial `t` merge into one plateau, so distinct-`t` counts are a
  **lower** bound and per-time-point advance an **upper** bound. This applies to both phases, so the
  ratio is more robust than either absolute figure.
- `T_cm` prints to 5 dp, hiding `ΔN` below ~1.2e-6. The measured creep advance (4.6e-5) sits ~38×
  above that floor and is resolvable; sub-floor structure is not.
- "distinct `t`-plateau ≈ solver time point" is an inference from SciPy BDF control flow, not a
  retained solver counter. A version-pinned reading of `num_jac` would harden it; V1 would measure it
  directly.

```text
added_lines: this section only
deleted_lines: 0
files_touched: 1
runtime_behavior_changed: no
physics_behavior_changed: no
known_blocker_reduced: no -- one writer claim refuted, one V1 signature row falsified
blocker_movement_ratio: 0.00
cost_effectiveness_verdict: RECORD_ONLY
```

§10's cost block covers the re-audit revision and is not restated here.
