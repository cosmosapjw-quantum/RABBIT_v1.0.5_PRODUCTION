# Codex Development Directive: RABBIT ODE Solver After the Four-Loop Audit

Date: 2026-08-24 (Asia/Seoul)  
Status: **ADVERSARIAL DEVELOPMENT DIRECTIVE**  
Evidence branch/head: `external-audit/ode-four-loop-complete-20260823` @ `9c05a65eaa5fbb86bec5d131c0d300689217de16`  
Audited code head: `78f5b091bd8de99bd8d0b1eddd8a0b7978f5da2b`  
Disposition: **ACCEPT AS A NEGATIVE RESEARCH RESULT / REJECT AS A SOLVER REMEDIATION**

This document does not reopen a gate, amend the canonical claim ledger, authorize public production, or validate a new solver. It re-adjudicates retained source and evidence; no fresh numerical execution belongs to it. `complete` in the evidence-branch name means packet closure, not solver or physics completion.

## 0. Mandatory context and base identity

Read, in order:

1. `AGENTS.md`
2. `docs/TYPEI_AUGMENTED_NOQKE_CODEX_ANTI_DRIFT_GUARDRAILS.md`
3. `bbn_codex_anti_drift_cost_effective_policy.md`
4. `.agent-harness/runs/run-20260823-ode-four-loop-final-implementation/FINAL_EXTERNAL_AUDIT_REPORT.md`
5. `.agent-harness/runs/run-20260823-ode-four-loop-final-implementation/results/A-IF-REVIEW2.json`
6. this document

The evidence branch is not the next production base. Do not merge it wholesale into `main`. Start executable work from the current canonical development base and verify these audited blobs:

```text
src/rabbit/collisions/dynamic_collision_driver.py  0164b9262244e07770b74afa7db57b27811b8e5f
src/rabbit/collisions/dynamic_collision_core.py    49126d52cff4a150e017a72d98635afbfbd9b7cd
```

If either differs, stop and localize the semantic delta before reusing this directive.

## 1. Executive verdict

C1 did not alter the collision-on numerical trajectory. It altered admission/postprocessing so the same trajectory was rejected instead of clipped and reported. Exact HEAD and C1 retained identical canonical configuration, solver statistics, and raw occupations. C1 was therefore a useful falsification instrument, not an ODE repair.

The strongest justified statement is:

> For the frozen `n_q=24`, collision-on, direct-occupation Radau trajectory with `rtol=1e-8`, scalar `atol=1e-10`, `max_step=0.5`, and `T_gamma_stop=0.01 MeV`, accepted stored high-q occupations leave the Pauli interval. The first rejection is near the initial time; the largest negative excursion is about `5.49e-15`. Strict pointwise raw-state admission is incompatible with that coordinate/error-control pair.

This does not prove continuum negativity, identify the dominant cause, validate logit/Patankar/entropy methods, bound every weak-rate QoI, or authorize C2/C3. The next action is causal discrimination, not another broad candidate.

## 2. Adversarial findings

### A-01 — C1 repaired admission, not dynamics — critical

The experiment proved that current postprocessing launders a raw-state violation. It moved no trajectory, endpoint, or production blocker. Do not revive it with a fitted epsilon, final-only exception, tolerance widening, clipping, or projection.

### A-02 — Causal identification is incomplete — critical

Four hypotheses remain live:

1. scalar absolute error control permits direct-`f` tail overshoot;
2. the discretized/interpolated collision field is not inward on the Pauli boundary;
3. gain-loss cancellation, finite-difference probes, or accumulation error corrupts the tail sign;
4. pointwise error is small for some moments but uncertified for the actual weak-rate QoIs.

Choosing a positivity representation before separating these hypotheses risks another expensive rejected candidate.

### A-03 — Physical, probe, and reported domains are mixed — critical

The direct-occupation driver clips state occupations before RHS evaluation, clips resampled evaluator input, clips final accepted occupations, and floors `H` with `max(H,1e-100)`. A raw accepted state can therefore leave the physical domain, be mapped back into it for collision work, and still yield an endpoint observable. `max_clip_excursion < 1e-6` is a bulk-collapse guard, not a Pauli proof.

Future code must distinguish existing surfaces for:

- admitted physical accepted state;
- internal solver/Jacobian probe and typed recoverability;
- reported endpoint state with raw state and terminal certificate.

Do not add another wrapper or telemetry framework.

### A-04 — Existing logit code is not a shortcut — critical

`flrw_dynamic_collision_rhs` guarantees mapped `0<f<1`, but divides by `f(1-f)`, freezes the tail through `fnn_floor`/`np.where(...,0)`, retains an H floor, and is not validated as the same comoving endpoint path. Reusing it would replace clipping with tail slaving.

Freeze one sign convention:

\[
 u=\log\frac{f}{1-f},\quad f=(1+e^{-u})^{-1},\quad \frac{du}{dN}=\frac{C}{Hf(1-f)},
\]

or

\[
 \ell=\log\frac{1-f}{f}=-u,\quad f=(1+e^{\ell})^{-1},\quad \frac{d\ell}{dN}=-\frac{C}{Hf(1-f)}.
\]

The current `q+A` variable is the second convention even where prose calls it “logit”.

### A-05 — Several tests overstate authority — high

- “analytic `N_eff=3` exactly” is checked with a broad band while another test pins a discretized value near `2.9934`.
- `max_clip_excursion <1e-6` does not establish pointwise positivity.
- collision-on `3.00<=N_eff<=3.15` is an internal smoke envelope.
- same-`df` plasma/neutrino cancellation is a discrete exchange identity, not an independent collision-physics oracle.

Correct these labels only inside the first accepted executable PR or a net-deflation consolidation; do not open a cosmetic PR.

### A-06 — Solver debts have unequal priority — high

NumPy Rodas false-success defects and Rust failure/counter/certificate gaps are real. They are not equally active. The frozen discriminator uses SciPy Radau, Rust AOT is the retained target, and JAX is frozen. Start C2 only if NumPy Rodas is required by the route selected below; otherwise freeze its unsupported semantics.

### A-07 — Green subsets cannot offset the failed physical discriminator — high

Use this validation order:

1. static/local causal discriminator;
2. focused real collision-on short prefix;
3. complete focused module;
4. paired exact-HEAD comparator;
5. production-not-slow;
6. gold/full only after the changed physical path passes.

Do not spend another hour-scale full-suite run on a candidate already falsified at steps 1–3.

### A-08 — Cost and evidence discipline drifted — medium

C1 was described as within a `<=40` production-net cap but violated the reviewer’s total-net cap (`+185`). Future contracts must count production, tests, docs, and committed artifacts prospectively. The audit branch’s raw logs, materialized trees, archives, and duplicated harnesses are evidence-only and must not become the development pattern. Future loops retain at most one compact result and one adjudication; no temporary worktree or repeated harness copies in the canonical tree.

## 3. Physics and numerical conventions

Internal convention on this surface: `c=hbar=k_B=1`.

```text
N=ln(a), q, Y, f       dimensionless
0<=f<=1                Pauli domain
T_gamma, H, C=df/dt    MeV
df/dN=C/H              dimensionless per e-fold
rho and Q/H             MeV^4
Q>0                     energy into neutrinos
```

First-law signs:

\[
\frac{d\rho_\nu}{dN}=-4\rho_\nu+\frac{Q}{H},\qquad
\frac{d\rho_{em}}{dN}=-3(\rho_{em}+P_{em})-\frac{Q}{H}.
\]

A node-local gain-loss representation

\[
C_i=(1-f_i)G_i-f_iL_i,\qquad G_i,L_i\ge0
\]

implies inward Pauli boundaries:

\[
C_i|_{f_i=0}=G_i\ge0,\qquad C_i|_{f_i=1}=-L_i\le0.
\]

For frozen `G_i,L_i,H`, define `f_*=G_i/(G_i+L_i)` and `lambda=(G_i+L_i)/H`. Then

\[
f_i(N+\Delta N)=f_*+[f_i(N)-f_*]e^{-\lambda\Delta N},
\]

which preserves `[0,1]` when the premises hold. This is a discriminator/algorithmic primitive, not yet a validated solver.

## 4. The only authorized next unit

`P0_COLLISION_TAIL_INWARDNESS_AND_CAUSAL_DISCRIMINATOR`

P0 determines whether the first high-q negative tail originates primarily in collision assembly, cancellation/precision, or direct-`f` integration/error scaling. It changes no production source and implements no logit, Patankar, projection, C2, C3, backend, endpoint, or full-suite run.

### 4.1 Frozen case and artifact

```text
n_q=24
collisions=true
method=SciPy Radau
rtol=1e-8
atol=1e-10 scalar baseline
max_step=0.5
T_gamma_stop=0.01 MeV
first implicated node=f_nue[22]
implicated tails=nue[18..23], nux[19..23]
```

Extract without merging the evidence branch:

```bash
git show external-audit/ode-four-loop-complete-20260823:.agent-harness/runs/run-20260823-ode-four-loop-final-implementation/artifacts/raw_trajectory_exact_head_collision_on.json \
  > /tmp/rabbit_raw_trajectory_exact_head_collision_on.json
sha256sum /tmp/rabbit_raw_trajectory_exact_head_collision_on.json
```

Expected SHA-256:

```text
2da11255f25761e4b7e7ea330eda2c4948013e4d1df00c5e127720a90f9a678e
```

Use the last stored state before the first violation and the first violating state. Do not reconstruct a D-071 restart from logs.

### 4.2 P0-A — boundary inwardness and gain-loss reconstruction

At every implicated node in the last valid physical state:

1. hold all other entries fixed;
2. evaluate the node field at `f_i=0` and `f_i=1`;
3. define `G_i=C_i(0)` and `L_i=-C_i(1)`;
4. sign-certify `G_i>=0` and `L_i>=0`;
5. test `C_i(f_i)=(1-f_i)G_i-f_iL_i` at the retained value and one interior value;
6. repeat the worst node with an independent accumulation/precision axis.

Build the sign interval from ordinary-versus-independent accumulation, not a post-hoc epsilon. If it crosses zero, return `INCONCLUSIVE`. The exact 2-to-2 statistical factor is gain minus loss before quadrature, but the interpolated and bank-folded implementation must be tested rather than assumed.

### 4.3 P0-B — cancellation and local stiffness

For each node report:

```text
q, f_previous, f_first_invalid, G, L, C,
(|G|+|L|)/max(|C|, smallest_positive_scale),
(G+L)/H, Delta_N, lambda_Delta_N
```

A large cancellation ratio argues against naive `C/[f(1-f)]`; it does not select a remedy by itself.

### 4.4 P0-C — frozen-coefficient positivity prediction

Use measured `G,L,H,Delta_N` in the exact local update and compare with the next raw accepted state.

- local update physical, Radau state negative: integrator/scaling or coefficient variation remains live;
- local update nonphysical: gain-loss premise or collision discretization failed;
- precision interval dominates: `STOP_INCONCLUSIVE`.

This is not an endpoint benchmark.

### 4.5 P0-D — raw tail effect on QoIs

Without clipping, report signed invalid-tail contributions to:

- number moment;
- energy moment;
- every available weak/collision rate that consumes the same spectrum;
- raw-signed versus clipped diagnostic `N_eff`.

Small moment impact cannot legalize negative `f`; it decides whether the future contract controls pointwise `f`, weak-rate QoIs, moments, or all three.

### 4.6 P0 cost and output

```text
production source changed: 0 files
committed probe/test files: <=1
committed report files: <=1
net committed lines: <=160 total
full endpoint/default-suite runs: 0
candidate implementations: 0
```

P0 may retain one compact result, one concise adjudication, and one regression in an existing test module if it becomes a durable invariant. No registry, gate, readiness wrapper, telemetry class, figure, manifest framework, archive copy, or harness package.

## 5. Decision table

| P0 result | Next admissible route | Forbidden inference |
|---|---|---|
| boundary or `G/L` sign fails | repair collision discretization/interpolation/statistical assembly; test detailed balance, boundary signs, discrete invariants, first-law exchange | do not blame the ODE solver or hide the defect with a transform |
| inwardness passes; cancellation severe | derive factorized gain-loss/affinity and a positivity-preserving local update | do not divide a cancelled `C` by tiny `f(1-f)` or reuse `fnn_floor` |
| inwardness passes; local update physical; direct-`f` overshoots | test one bounded positivity-preserving or physically scaled short-prefix method; retained target is Rust | do not claim endpoint progress from a local step |
| moment impact small but weak-rate bound absent | derive the weak-rate/tail certificate before selecting the norm | do not equate energy convergence with kinetic accuracy |
| precision/decomposition inconclusive | `STOP_INCONCLUSIVE`; improve only the discriminator | do not start C1, C2, C3, logit, Patankar, or a solver swap |

If a positivity-preserving route is selected, prefer gain-loss exponential, Patankar-type, or entropy-compatible algebra. Logit is admissible only with factorized non-cancelled gain/loss or affinity, no tail floor/slaving, explicit zero-flux branch, separate temperature/Hubble domains, order/consistency evidence, and a raw short-prefix comparator. Accepted retained implementation belongs in Rust; SciPy remains temporary number-of-record and JAX remains frozen.

If error scaling is selected, freeze a vector/physics-weighted norm before execution and tie it to explicit pointwise and QoI goals. Removing the observed negatives after tuning does not prove invariance.

## 6. Explicit deferrals and claim language

Until P0 closes:

```text
C1 reimplementation                  FORBIDDEN
C2 NumPy Rodas repair                DEFERRED unless P0 requires it
C3 slow-manifold/D-071               FORBIDDEN
generic Jacobian/Krylov tuning       FORBIDDEN as blocker claim
JAX/Diffrax forward development      FORBIDDEN
full endpoint/hour-scale sweep       FORBIDDEN
public production/publication/QKE    FORBIDDEN
post-hoc epsilon/clipping/projection FORBIDDEN
```

Use these labels:

```text
four-loop packet                     VALIDATED NEGATIVE RESULT
C1 code                              IMPLEMENTED EXPERIMENT / REJECTED / NOT ADMITTED
current clipped endpoint observable  DIAGNOSTIC
max_clip_excursion <1e-6             BULK-STABILITY GUARD
collision-on N_eff band              INTERNAL SMOKE ENVELOPE
same-df energy cancellation          DISCRETE EXCHANGE IDENTITY
logit/Patankar/entropy route         PROPOSED
endpoint/scientific/public authority NOT EARNED
```

## 7. Required closeout record

```text
base_commit:
relevant_source_blob_identity:
production/test/report_files_changed:
added/deleted/net_lines_total:
token_use_exact/token_use_basis:
frozen_case_identity:
boundary_inwardness_verdict:
gain_loss_reconstruction_verdict:
cancellation_diagnosis:
frozen_update_verdict:
raw_tail_qoi_impact:
selected_route:
blocker_movement_ratio:
claim_ceiling:
stop_reason:
```

No additional gate board.

## 8. Ready-to-paste Codex task

```text
Read AGENTS.md, the two anti-drift policies, the four-loop final report,
A-IF-REVIEW2.json, and
CODEX_ODE_SOLVER_POST_FOUR_LOOP_ADVERSARIAL_DIRECTIVE_2026-08-24.md.

Execute only P0_COLLISION_TAIL_INWARDNESS_AND_CAUSAL_DISCRIMINATOR.
Do not edit production source or implement logit, Patankar, projection, C2, C3,
a solver swap, backend, endpoint, or full-suite run. Extract and hash-verify the
retained exact-head collision-on trajectory. At the last valid and first invalid
states, test Pauli-boundary inwardness, reconstruct gain/loss, quantify
cancellation, compare the exact frozen-coefficient positivity update with the
next raw state, and report raw-versus-clipped number, energy, weak-rate, and
N_eff effects.

Use an independent accumulation/precision axis for the worst node. If a sign
interval crosses zero, return STOP_INCONCLUSIVE. Preserve raw signed values and
freeze all tolerances before execution. Produce at most one compact result, one
concise adjudication, and one regression in an existing test module. Add no
manifest, gate, telemetry framework, figure, archive copy, or harness package.

Select exactly one route from the decision table or STOP_INCONCLUSIVE. Claim no
endpoint, scientific, public-production, QKE, Rust-promotion, or D-071 movement.
```

## 9. Final disposition

Credit the four-loop programme for rejecting a false repair, not for selecting the next solver. The next useful result is a sign-certified causal answer: does the discrete collision field point inward at the Pauli boundary, and is the observed tail failure created by collision assembly, cancellation, or direct-`f` integration?

```text
C1=REJECTED
C2=DEFERRED
C3=FORBIDDEN
POSITIVITY METHOD=UNSELECTED
G-F10-INDEPENDENT-FLRW=FAIL
G-HARNESS-INTEGRITY=FAIL
SCIENTIFIC/PUBLIC/ENDPOINT PROMOTION=NOT EARNED
```
