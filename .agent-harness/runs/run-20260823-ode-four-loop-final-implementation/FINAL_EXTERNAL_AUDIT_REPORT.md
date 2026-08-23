# RABBIT ODE solver four-loop integrated conclusion report

Date: 2026-08-23 (Asia/Seoul)  
Run: `run-20260823-ode-four-loop-final-implementation`  
Repository head: `78f5b091bd8de99bd8d0b1eddd8a0b7978f5da2b`  
Context version: `a819af389d4f97e3b641ccb65cd6c120a0943289f8c333ec67490bc3267ad6cb`  
Frozen acceptance hash: `842c695c6efccb05fd3a19d87876ca6a9faa499364a9cec3772b500ac994ceed`  
Final bounded decision: **STOP_INVALID**  
Code disposition: **C1 candidate REJECTED / NOT ADMITTED**  
Gate board: **6 PASS / 2 FAIL, unchanged**

## 1. Executive conclusion

The four prior research loops were integrated without dropping any of the 34
registered ODE findings. The first implementation candidate, C1 raw accepted-state
admission, was then implemented in an isolated worktree under a recorded red-green
cycle and run against both adversarial fakes and the real collision-on FLRW
trajectory.

The implementation is a useful falsification instrument but is not an admissible
repair. It correctly exposes that the current Cartesian occupation-coordinate
trajectory does not remain inside the strict Pauli box: the `n_q=24` collision-on
run has 2,096 negative raw occupation entries in 231 of 234 stored samples, confined
to high-momentum nodes 18--23. The earliest rejected entry is

```
sample = 3
N = -2.302557321652884
component = f_nue[22]
q = 69.96224003510503
f_initial = 4.128432717702736e-31
f_raw = -2.3008216172341183e-30
```

and the largest negative excursion is `-5.491289338418672e-15`. The solver itself
reports success, one terminal event, 234 stored points, 3,722 RHS evaluations,
136 Jacobian evaluations, and 586 LU decompositions. Exact HEAD and the candidate
produce the same stored solver trajectory; exact HEAD sanitizes it at final
postprocessing and returns, while C1 raises at sample 3.

This is decisive because the scalar absolute tolerance is `1e-10`, roughly
`2.4e20` times the initial occupation at the first rejected tail node. Strict
`0 < f < 1` cannot be promised by this raw Cartesian coordinate plus this error
control. Adding a tolerance exception or clipping after observing the result would
violate the prospectively frozen contract and turn an admission test into another
silent sanitizer. The remedy therefore has to move to a positivity-preserving
state representation or invariant-preserving time discretization, with its own
prospectively frozen physics and numerical contract.

No code was committed, pushed, merged, packaged, or admitted into the original
checkout. The rejected exact patch and postimage are retained as audit artifacts.
No D-071, physical-prefix, endpoint, independent-FLRW, public-production, QKE,
JAX-forward-development, or Rust-promotion authority was opened.

## 2. Authority, custody, and scope

The repository controlling documents were read before implementation:

- `docs/TYPEI_AUGMENTED_NOQKE_CODEX_ANTI_DRIFT_GUARDRAILS.md`
- `bbn_codex_anti_drift_cost_effective_policy.md`
- repository `AGENTS.md`
- the bounded-work rules in `~/.codex/bounded-work-harness/RULES.md`

The original checkout was kept at branch `diagnosis_report`, exact HEAD
`78f5b091...`, with the user's pre-existing untracked
`RABBIT_diagnosis_report.bundle` preserved. The experiment used:

```
worktree: /tmp/rabbit-ode-four-loop-wt.peLD3M/worktree
branch:   codex/ode-c1-raw-state-20260823
base:     78f5b091bd8de99bd8d0b1eddd8a0b7978f5da2b
```

The only candidate implementation surfaces were:

- `src/rabbit/collisions/dynamic_collision_driver.py`
- `tests/test_dynamic_collision_driver.py`

The original checkout's two context-pack files were changed only because the
mandatory harness bootstrap rebuilt them; their original bytes were backed up for
restoration at closeout. No user source change was overwritten.

The bounded unit authorized one C1 attempt, one independent review, one repair or
revert closeout, and no C2/C3 implementation. C2 and C3 remain research items.

## 3. Four governing loops and evidence identity

| Loop | Governing result | SHA-256 | Integrated conclusion |
|---|---|---|---|
| 1. exhaustive ODE audit | `A-ODE-ADJ3.json` | `18195e473776b0fd0bcc23b1be41c45338d529bffaf75262cb2f99e9b6a174c9` | 34 IDs exactly; audit status FAIL; bounded static completeness only |
| 2. physics-specific mitigation | `A-PM-ADJ.json` | `d47308b4bb991a4a0c879b7a14b06a958143c3dbe7b46a1cb7bdd43e9c2e675a` | 32 normalized physics roots; research map only; no implemented blocker movement |
| 3. independent math/algorithm/code | `A-MAC-ADJ2.json` | `1990816777280bbbf4fe3f3f4c59fba250e6a0a4d2c630308dcd0935dc6f5842` | 182/122 state identities reconciled; dense-FD no-go; P1--P3 ranked only |
| 4a. coding design | `A-CH-DESIGN.json` | `c8e8aba42c2acb4e166620fe03b96bdbf3c5e96ee5cd837a4955bb8a0eb80630` | 34-ID design ledger |
| 4b. hostile review | `A-CH-ADV.json` | `1090179e537c0b760f404e2b322517c5cea7703fbf5ec6b35934505e18bfa7c1` | pre-repair contradictions retained |
| 4c. coding decision | `A-CH-DECIDE.json` | `c0bcf5ddee457de8f7c5dcdaa3cb1ea15d6884229458015f8d91bc1f2693ec21` | C1 specified for later implementation; C2/C3 rework; no execution had run |

The machine-readable integration is
`artifacts/FOUR_LOOP_EVIDENCE_MAP.json`, SHA-256
`92f4739c1722f012ae8f57499136ee6cc038dc595943a43d1c235007ddbbb56c`.

The loop-4 C1 promotion was explicitly conditional: if a real current trajectory
touched the strict boundary, the required action was scientific stop, not tolerance
widening. This run triggered that stop condition.

## 4. Complete 34-finding disposition

No ID was removed, merged away, or renumbered. “Future” means no implementation
authority in this run.

| ID | Audit state | Subject / final route after execution |
|---:|---|---|
| 001 | FAIL | D-071 dominant call-count gap; C3 remains REWORK |
| 002 | FAIL | static physical-prefix/JVP insufficiency; C3 remains REWORK |
| 003 | FAIL | strict slow-convergence lock; retained blocker |
| 004 | FAIL | Class-A phase admission; future companion |
| 005 | FAIL | tilted/FLRW terminal admission; future companion |
| 006 | FAIL | convergence consumer contract; future companion |
| 007 | FAIL | fabricated NumPy Rodas event; C2 REWORK |
| 008 | FAIL | retired/frozen JAX lane; future companion, no forward work |
| 009 | FAIL | raw accepted-state admission; C1 was implemented experimentally and is now REJECTED/REWORK |
| 010 | FAIL | post-failure observable consumption; future companion |
| 011 | FAIL | canonical phase matrix; future companion |
| 012 | FAIL | former C2 line cap withdrawn; future companion |
| 013 | FAIL | outcome precedence and counters; C2 REWORK |
| 014 | FAIL | generic optimization no-go; decisive evidence retained |
| 015 | FAIL | Rust typed failure taxonomy; future companion |
| 016 | FAIL | diffsol retry semantics; decisive experiment required |
| 017 | FAIL | Rust work accounting; future companion |
| 018 | FAIL | failure snapshot identity; future companion |
| 019 | INCONCLUSIVE | N48 derivative ladder; decisive experiment required |
| 020 | FAIL | outcome properties and mutants; C2 REWORK |
| 021 | INCONCLUSIVE | exact-head dual endpoint; decisive experiment required |
| 022 | FAIL | Rust event sign semantics; future companion |
| 023 | INCONCLUSIVE | event certificate; future companion |
| 024 | INCONCLUSIVE | `hmax`/`T_stop` semantics; decisive experiment required |
| 025 | INCONCLUSIVE | recoverability taxonomy; decisive experiment required |
| 026 | FAIL | subnormal finite-difference step; future repair |
| 027 | FAIL | inconsistent wrapper validation; future consolidated repair |
| 028 | INCONCLUSIVE | Rust reason-branch coverage; future companion |
| 029 | FAIL | frozen Diffrax API semantics; low-priority future hygiene |
| 030 | FAIL | effective config and stale prose; future companion |
| 031 | INCONCLUSIVE | extrinsic harness/environment limitation; retained |
| 032 | PASS | solver-role/gate policy boundary; preserve |
| 033 | PASS | bounded historical good paths; preserve without extrapolation |
| 034 | PASS | declared mechanical inventory; preserve |

The completeness ceiling remains the original audit ceiling. It is not a proof
about generated code, external consumers, unlisted non-Python extensions, or
runtime-supplied higher-order callables.

## 5. Actual C1 code experiment

### 5.1 Implemented candidate behavior

The rejected candidate did all of the following in the isolated worktree:

1. Added `RawStateAdmissionError(RuntimeError)` with stable reason, sample index,
   `N`, component, value, and a bounded eight-value snapshot.
2. Moved solver-success and terminal-event admission before reading the last
   solution column.
3. Required exactly one terminal event.
4. Checked every stored `sol.y` column for structural consistency, finite `N`,
   finite positive `T_gamma`, and strict finite `0 < f < 1` in both banks.
5. Recomputed an unfloored Hubble rate from each stored column and required it to
   be finite and positive.
6. Removed final-state occupation clipping and used the admitted raw final bank.
7. Retained evaluator-side clamps for out-of-domain Radau/Jacobian probes.
8. Changed RHS Hubble handling from `max(H, 1e-100)` to immediate failure on
   nonfinite/nonpositive H.
9. Added adversarial fake-solver tests for nonfinal/final occupation corruption,
   `T_gamma`, Hubble, duplicate events, RHS probe separation, and valid identity.

### 5.2 Diff size and artifact

| Surface | Added | Deleted | Net |
|---|---:|---:|---:|
| production driver | 82 | 43 | +39 |
| focused tests | 146 | 0 | +146 |
| total | 228 | 43 | +185 |

The production net change satisfies the loop-4 `<=40` cap mechanically. The
candidate nevertheless violates the substantive contract in two ways:

- the real collision-on reference path fails, so the patch is not behaviorally
  compatible;
- it changes RHS probe-side H-floor semantics even though the sealed C1 contract
  said the probe-side H-safe behavior remains unchanged. That change may be a
  worthwhile future fail-loud repair, but it is not authorized as part of F-009.

Requiring exactly one event also touches the C2 event-semantics neighborhood and
has no event residual/bracket certificate. It is not counted as C2 movement.

The exact rejected patch is `artifacts/C1_REJECTED_CANDIDATE.patch`, SHA-256
`a12a2c772c288fecb6a74b27d2a7133f2d839273a110276b7a24f850ef96a849`.
The postimage source hashes are:

- driver: `6b33486fb4fa1fce2872db0edeed14683cd3817927126be1026a371daf39158e`
- test: `67ec0b39fbdd69f39ce840f418da292dd42720fbc097a56e7c0945119a8c2dd8`

`git diff --check` passed. No commit exists.

## 6. TDD chronology and validation runs

All raw logs are under `raw_logs/`; exact commands are in each `script(1)` header,
and full hashes are in `RUN_MANIFEST.json`.

| Log | Exit | Result | Authority |
|---|---:|---|---|
| `00_environment.log` | 0 | hardware, OS, head, tree, system packages captured | environment receipt |
| `01_baseline_target_not_slow.log` | 0 | 10 passed, 3 deselected, 4.19 s | pre-edit baseline |
| `02_tdd_red_invalid_nonfinal.log` | 1 | 1 expected failure: did not raise | discriminating RED |
| `03_tdd_green_invalid_nonfinal.log` | 0 | 1 passed, 0.57 s | focused GREEN |
| `04_tdd_red_hubble.log` | 1 | 5 expected failures, 14 deselected | discriminating RED |
| `05_tdd_green_hubble.log` | 0 | 5 passed, 14 deselected | focused GREEN |
| `06_exact_head_mutation_red_final_and_event.log` | 1 | 10 failed, 1 passed | **inadmissible**: worktree conftest contamination |
| `07_exact_head_red_final_and_event.log` | 1 | corrected `--noconftest`: 11 failed, 20 deselected | exact-HEAD RED |
| `08_c1_adversarial_green.log` | 0 | 18 passed, 13 deselected | fake/adversarial GREEN |
| `09_target_not_slow_green.log` | 0 | 28 passed, 3 deselected, 4.22 s | non-slow focused PASS |
| `10_target_full_green.log` | 1 | 3 failed, 28 passed, 55.07 s | filename is stale; actual status FAIL |
| `11_exact_head_target_full_baseline.log` | 0 | 13 passed, 55.08 s | exact-HEAD comparator PASS |
| `12_production_not_slow_gate.log` | 2 | 30 collection errors | system environment/path failure, no test verdict |
| `13_measured_environment_install.log` | 0 | pinned environment installed, wall 1:43.49 | environment construction |
| `14_measured_environment_verify.log` | 0 | pinned imports and CPU JAX device verified | environment identity |
| `15_production_not_slow_measured_env_retry.log` | 0 | 396 passed, 8 skipped, 2317 deselected | production-not-slow PASS |
| `16_gold_measured_env.log` | 1 | 5 failed, 100 passed, 2 skipped | gold FAIL |
| `17_exact_head_gold_failure_attribution.log` | 1 | same selected 5 failures on exact HEAD | exact baseline attribution |
| `18_full_default_measured_env.log` | 1 | 54 failed, 2587 passed, 80 skipped, 9 warnings | full-default FAIL |
| `19_dynamic_collision_module_measured_env.log` | 1 | 3 failed, 28 passed, 49.91 s | C1-attributable focused FAIL |
| `20_exact_head_dynamic_collision_measured_env.log` | 0 | 13 passed, 49.57 s | exact-HEAD focused PASS |

The one allowed repeated premise was used to correct the production collection
environment: the first attempt lacked JAX and omitted the repository root from
`PYTHONPATH`; the measured-environment retry changed both premises and passed.

### 6.1 Measured environment

```
OS:       Linux 7.0.0-29-generic x86_64
CPU:      AMD Ryzen 9 5900X, 12 cores / 24 threads
RAM:      94 GiB
Python:   3.12.3
NumPy:    2.4.4
SciPy:    1.17.1
pytest:   9.0.3
JAX:      0.10.0
JAXLIB:   0.10.0
Diffrax:  0.7.2
device:   CpuDevice(id=0)
```

`requirements.measured.txt` SHA-256:
`b8996835e148f086bf1e6cc2be6a2eee4a91d525421a1e67c687008dcbe2050b`.
The base NumPy/SciPy hash lock SHA-256 is
`fd84fa522d34b5732b063e5f930601cd11ebe86bcf5d53d1452acf146cb094d2`.
The full measured file pins optional packages but does not carry hashes for every
wheel; this is an explicit residual provenance limit.

### 6.2 Resource measurements

| Run | pytest time | wall time | max RSS |
|---|---:|---:|---:|
| production-not-slow | 159.14 s | 162.55 s | 1,800,700 KiB |
| gold | 1,213.43 s | 1,230.20 s | 13,521,376 KiB |
| exact-HEAD five-gold comparator | 196.99 s | 200.92 s | 3,360,264 KiB |
| full default | 4,614.27 s | 4,662 s | 38,324,268 KiB |
| candidate focused module | 49.91 s | 50.21 s | 108,812 KiB |
| exact-HEAD focused module | 49.57 s | 49.86 s | 107,616 KiB |

The full default suite consumed approximately 1 h 17 min wall and 38.3 GB peak
RSS. Passing production-not-slow is not a substitute for failed full or gold gates.

## 7. Raw trajectory analysis

### 7.1 Collision-on, exact HEAD versus C1

| Quantity | Exact HEAD | C1 candidate |
|---|---:|---:|
| solver success | true | true |
| terminal events | 1 | 1 |
| stored points | 234 | 234 |
| `nfev/njev/nlu` | 3722 / 136 / 586 | 3722 / 136 / 586 |
| raw invalid occupations | 2096 | 2096 |
| affected stored samples | 231 | 231 |
| nonfinite / zero / `>=1` | 0 / 0 / 0 | 0 / 0 / 0 |
| largest negative magnitude | `0x1.8bb068b83059ap-48` | same |
| returned result | yes | no; raises at sample 3 |
| exact-HEAD `N_eff` | `0x1.8201af212b61cp+1` | not emitted |

The raw-trajectory artifacts are not toy output; they instrument the actual
`n_q=24`, `collisions=True`, `rtol=1e-8`, `atol=1e-10`, `max_step=0.5`,
`T_stop=0.01 MeV` focused trajectory. Exact HEAD and C1 trajectory artifacts have
different document hashes because one records a return and the other an exception,
but their solver statistics and raw stored states agree.

The exact probe sources are retained as
`artifacts/raw_trajectory_probe.py` (SHA-256
`756af14f2ad378575e391660a1e54346876fec2982884db0eae3aea5f1c204c8`) and
`artifacts/valid_dummy_probe.py` (SHA-256
`5272d20bd38145c39eedc3f2a57f24d13e1a76efe36c67130e89cb2e3b68c6df`).
Reproduction consists of running the same probe with `PYTHONPATH` bound first to
the exact-HEAD checkout and then to the candidate worktree; use `--collisions` for
the collision-on case and omit it for the collisionless control. The scripts emit
canonical sorted JSON to stdout, including every invalid raw entry in IEEE
hexadecimal form.

The negative entries are restricted to:

| Tail component | Negative samples |
|---|---:|
| `f_nue[22]` | 231 |
| `f_nue[20]` | 230 |
| `f_nue[21]` | 228 |
| `f_nux[20]` | 228 |
| `f_nue[19]` | 227 |
| `f_nux[22]` | 227 |
| `f_nux[19]` | 223 |
| `f_nux[21]` | 222 |
| `f_nux[23]` | 213 |
| `f_nue[23]` | 52 |
| `f_nue[18]` | 15 |

This pattern is a tail-coordinate numerical problem, not evidence that the
continuum neutrino distribution is physically negative and not a bulk collapse.
It is also not permission to ignore it: once raw Cartesian values leave the Pauli
domain, a clipped output cannot serve as raw-state admission evidence.

### 7.2 Collisionless control

The same C1 machinery on `collisions=False` completed with one event, 146 stored
points, 1,038 RHS evaluations, three Jacobian evaluations, 26 LU decompositions,
zero invalid raw occupations, minimum occupation
`0x1.572efef6863bap-118`, and `N_eff=0x1.7f28410f9c22ap+1`.

This localizes the boundary problem to the collision-on tail dynamics/error control,
not to stored-sample enumeration itself.

### 7.3 Fake valid-path identity

The fake valid-path artifacts for exact HEAD and C1 are byte-identical:

```
9f25dd6251d3d577675b57e7cf51350397e0252cc39bd4d51cdcc8b0d4fef0b0
```

All scalar values were compared in IEEE hexadecimal form and final arrays by dtype,
shape, and byte hash. This validates local compatibility for the fake admissible
path only; it cannot override the real-trajectory failure.

## 8. Full-suite failure taxonomy

The full default suite's 54 failures were grouped mechanically by test module and
then classified by the observed traceback. Only the three dynamic-collision
failures are directly attributable to the C1 diff.

| Failure group | Count | Root / attribution |
|---|---:|---|
| C1 raw-state admission | 3 | direct C1 regression; exact-HEAD module passes 13/13 |
| inference prediction validation | 25 | missing `canonical_batch_forward_solver` / `_3d` symbols; outside C1 surface |
| native Tier-1 vertical slice | 10 | nine `_rabbit_cpu` unavailable paths and one absent Git revision `0496e5e`; outside C1 |
| JAX kernel-remap envelopes | 3 | numerical envelope drift; two reproduced exactly on HEAD, third outside C1 |
| Type III/IV/IX gold values | 3 | exact values reproduced on HEAD |
| registry/canonical-doc contracts | 3 | stale expected text/registry contracts; outside C1 |
| gradient bridge contracts | 3 | negative tau-gradient sign plus two `rhs_factory` calling-convention errors; outside C1 |
| factorized Jacobian shapes | 2 | observed `(24,31)` vs `(24,25)` and `(23,21)` vs `(23,17)`; outside C1 |
| live AlterBBN anchor | 1 | `Y_p` gap `1.0513455e-3 > 5e-4`; outside C1 |
| strict convergence lock | 1 | `XPASS(strict)`; outside C1 |

The five selected gold failures were rerun on exact HEAD and reproduced with the
same numerical values. The remaining 46 non-C1 failures were not all rerun on exact
HEAD; their attribution is based on changed-surface isolation and explicit
tracebacks. This distinction is retained as an audit ceiling, not silently promoted
to a full baseline replay claim.

## 9. Why C1 is rejected

The frozen acceptance cells evaluate as follows:

| Criterion | Result |
|---|---|
| AC-01 four-loop 34-ID evidence map | PASS |
| AC-02 recorded focused RED | PASS |
| AC-03 two candidate files, production net `<=40` | mechanical PASS, but RHS-H scope violation remains |
| AC-04 adversarial focused tests green | PASS |
| AC-05 complete focused module | **FAIL: 3 real collision-on tests** |
| AC-06 production-not-slow and gold executed | PASS for execution; production PASS, gold baseline FAIL |
| AC-07 full default and no attributable regression | **FAIL** |
| AC-08 independent review | see section 13 |
| AC-09 external audit packet | completed by this report/manifest |
| AC-10 claim and validation ledgers | completed run-locally; canonical ledgers intentionally unchanged |

Because AC-05 and AC-07 fail after the one authorized implementation attempt, the
bounded decision is `STOP_INVALID`. The candidate is not merely “needs another
tolerance”: the present state coordinate and strict admission contract are
mathematically incompatible at the measured tail scale.

The code remains useful as `IMPLEMENTED` experimental evidence in the isolated
worktree. It is not `VALIDATED` as a remedy, is not admitted, and moves zero
production blockers.

## 10. Integrated physics-specific resolution research

### 10.1 Positivity/Pauli-domain state representation

The first physics-correct route is to evolve an unconstrained variable that maps
exactly into the Pauli box, for example

\[
u_i = \log\!\frac{f_i}{1-f_i},\qquad
f_i = \frac{1}{1+e^{-u_i}},\qquad
\frac{du_i}{dN}=\frac{C_i(f,T)}{H f_i(1-f_i)}.
\]

This is `DERIVED` under `df/dN=C/H`, but logit alone is not a complete remedy.
The `1/[f(1-f)]` factor is badly conditioned in the tail. A valid implementation
must factor the collision gain/loss terms using the actual reaction affinity and a
separate zero-flux branch, so equilibrium cancellation is not evaluated as a
subtraction of huge nearly equal quantities. For `1+2<->3+4`, the compatible
affinity is `A=u3+u4-u1-u2` only after the actual leg, sign, and chemical-potential
conventions are fixed.

Required gates for a future positivity-coordinate slice:

1. bitwise or high-precision detailed-balance null at equilibrium;
2. exact discrete collision-invariant nullspaces for the channel catalogue;
3. electron-pair channels checked with the correct total first-law exchange, not a
   fabricated neutrino-number or neutrino-only-energy invariant;
4. `0<f<1` by construction for every stored and evaluated physical state;
5. finite positive temperature, preferably with `theta=log(T_gamma/MeV)`;
6. finite positive H without an artificial accepted-state floor;
7. `n_q` and tail convergence using both energy and weak-rate QoIs;
8. same-case observable comparison against an independently tightened reference;
9. measured endpoint/cold-wall cost, not a segment-only label.

### 10.2 Entropy/Patankar or invariant projection as alternatives

A positivity-preserving exponential, Patankar-type, or entropy-stable collision
step is another `PROPOSED` route. A constrained projection is admissible only if it
is part of a prospectively specified numerical method with proven order, Pauli box,
and exact discrete invariants. Post-hoc `np.clip`, an epsilon acceptance band chosen
after this run, or a returned-value floor remains `FORBIDDEN` as validation.

### 10.3 Error control must match the physics coordinate

The measured tail demonstrates that one scalar `atol=1e-10` does not resolve
occupations near `1e-31`. Per-component scaling or a physically weighted norm may
control moment/QoI error, but it cannot alone prove pointwise positivity. A future
contract must state whether the target is pointwise distribution accuracy, an
energy moment, weak rates, or an adjoint-weighted observable and must carry a
rigorous tail remainder. Energy-only convergence is insufficient through weak
freezeout because spectra with equal energy can give unequal weak rates.

### 10.4 FLRW first-law and temperature clock

With `Q>0` into neutrinos, the consistent equations are

\[
\frac{d\rho_\nu}{dN}=-4\rho_\nu+Q/H,
\qquad
\frac{d\rho_{em}}{dN}=-3(\rho_{em}+P_{em})-Q/H.
\]

Their sum is the FLRW first law. These signs and MeV-four dimensions must be tested
on every coupled implementation. A temperature clock
`x=-log(T_gamma/T_ref)` gives `dN/dx=1/alpha` and `dt/dx=1/(H alpha)` only on a
monotone transverse branch with `alpha=-d log(T_gamma)/dN>0`; it changes the clock,
not the collision cost, and is not an endpoint-speed claim.

### 10.5 Hubble and terminal-event boundaries

Accepted physical states should never use an H floor to turn a singular/nonfinite
state into finite collision work. The future repair must nevertheless separate
physical transformed states from internal nonlinear/Jacobian probes. The rejected
C1 patch changed probe behavior without separate authority.

Event correctness requires a typed terminal cause, direction, finite bracket,
refined event time/value, unit-bearing residual tolerance, and precedence of solver
failure over any event-like payload. That remains C2 REWORK; “one nonempty event
array” and “exactly one fake event” are not a full event certificate.

## 11. Independent mathematics/algorithm/coding conclusions

### 11.1 State identity is not interchangeable

The structurally independent Python system at order 60 has
`3*60 + T_gamma + elapsed_time = 182` states. The folded Rust system has
`2*60 + T_gamma + elapsed_time = 122` states. The latter cannot be substituted into
the D-071 work estimate or physical-prefix claim without an explicit identity and
transfer proof.

### 11.2 Dense finite-difference removal cannot close D-071

Even the deliberately over-generous 365-fold counterfactual leaves
89,601.745 RHS-equivalents, 22.753 times the frozen 3,938 projection, and
396,095.603 seconds, 6.113 times the 64,800-second cap. Analytic/sparse JVP work may
improve local correctness/performance on the active Rust path, but cannot be
reported as D-071 closure.

### 11.3 Ranked remaining work

1. **Reformulated C1 discriminator, not the rejected patch.** Prospectively specify
   a positivity/entropy-preserving representation and run the exact collision-on
   falsifier. No tolerance fitting.
2. **C2 typed NumPy Rodas result/event state machine.** First freeze units, event
   residuals, refinement/attempt budgets, counters, and consumer assumptions. It
   remains reference/opt-in correctness work, not backend promotion.
3. **C3 moment-constrained slow-manifold discriminator.** No production code until
   a static full-grid discriminator establishes branch uniqueness, rank/coercivity,
   exact nullspaces, positivity/conservation, reaction-tail bounds, overlap error,
   adjoint QoI error, and certificate cost. The logged `N≈0.1653` point is not a
   restart state.

Generic Jacobian tuning, sparse/Krylov labels, solver swaps, hardware, more cores,
more budget, tolerance loosening, fixed momentum-tail slaving, JAX/Diffrax forward
promotion, and new readiness/telemetry wrappers remain rejected as blocker remedies.

## 12. Cost-effectiveness and anti-drift verdict

```
production: +82 / -43 / net +39
tests:      +146 / -0 / net +146
total:      +228 / -43 / net +185
accepted blockers closed: 0
accepted gate movement:    0
blocker movement ratio:    0.0
token_use_exact:           UNAVAILABLE
reason:                    the harness exposes no exact token counter
```

The experimental diff is cost-effective as a falsification probe because it found a
previously unmodeled coordinate/tolerance incompatibility. It is not cost-effective
as retained production surface because it moves no accepted blocker and regresses
the focused real trajectory. The anti-drift verdict is therefore **REJECT / retain
evidence only**.

## 13. Independent review and harness boundary

The first dedicated `adjudicator` runtime failed before substantive work with host
HTTP 400: the forced `gpt-5.6` model is unavailable for the current ChatGPT account.
No scientific result is attributed to that attempt. A replacement assignment uses
the supported `default` runtime with the exact sealed logical adjudicator role and
result template, consistent with the repository's documented fallback pattern.

Final independent adjudication: **REJECT / STOP_INVALID**. The sealed replacement
review (`A-IF-REVIEW2`, SHA-256
`c84de9341601ccb91dfd257f7311ac427275b767f033467f0043601406a40932`)
confirmed all of the following:

- the exact-HEAD and C1 canonical `{configuration, solver, raw_occupations}`
  payloads are identical (SHA-256
  `dbb889a696287da2bc02a60a6a42156cece3256f59d618710412d6e5b2c85dc6`);
- the real collision-on solve has 2,096 negative stored occupations over every
  sample 3--233, so the candidate discovers a pre-existing direct-`f`
  positivity-preservation blocker but does not repair it;
- exact HEAD passes the complete focused module (13/13), whereas C1 fails the
  three real collision-on tests (3 failed, 28 passed);
- the candidate also changes the frozen probe-side Hubble floor and strengthens
  nonempty-event admission to exactly one event, both outside C1, and omits an
  explicit finite-positive `z_sample` check;
- the two-file diff is +228/-43/net +185, exceeding the governing <=40 total-net
  boundary despite the production-only subtotal being net +39; and
- the five selected gold failures reproduce on exact HEAD, while exact
  attribution of the other 46 non-C1 full-suite failures remains inconclusive
  without a paired exact-HEAD full run.

The reviewer therefore found zero accepted blocker movement and zero gate
movement. The patch, trajectories, and logs are retained only as reproducible
negative evidence in the isolated worktree packet.

After closeout, the original checkout's generated context and active-run pointer
were restored byte-for-byte to their pre-run SHA-256 values. Its final Git status
contains only the pre-existing untracked `RABBIT_diagnosis_report.bundle`; no
production file, commit, push, or merge was added there. The rejected two-file
candidate remains isolated on `codex/ode-c1-raw-state-20260823`.

The failed dedicated-runtime receipt remains an explicit host-boundary artifact; it
is not presented as consumed or successful. `G-HARNESS-INTEGRITY` remains FAIL
regardless of the replacement review.

Closeout harness validation itself exited 0 with `ok: true`, zero pending results,
SSOT consistency `ok`, and canary freshness `ok`. It also enumerated the failed
dedicated `A-IF-REVIEW` receipt as open, exactly as preserved above. The replacement
`A-IF-REVIEW2` assignment, sealed adjudicator role, and result template each passed
hash verification, and its admission receipt was consumed with the result hash.

## 14. Claim ledger

| Claim | Status | Verdict / ceiling |
|---|---|---|
| 34-ID mechanical evidence map | VALIDATED | all 001--034 present once; bounded scope only |
| original C1 contract | SPECIFIED | internally coherent before real trajectory execution |
| isolated C1 candidate code | IMPLEMENTED | exists and executed; REJECTED/NOT ADMITTED |
| adversarial fake-state rejection | VALIDATED | 18 focused cases pass |
| valid fake-path identity | VALIDATED | byte-identical artifact |
| real collision-on strict admission | VALIDATED | candidate rejects at sample 3; this falsifies compatibility |
| independent C1 adjudication | VALIDATED | REJECT/STOP_INVALID; result SHA-256 `c84de934...a40932` |
| C1 blocker movement | PROPOSED | FAIL; no accepted movement |
| positivity/logit/entropy route | PROPOSED | equations derived, implementation/certificates absent |
| C2 typed Rodas repair | PROPOSED | REWORK; tolerances/budgets/consumer contract absent |
| C3 slow-manifold route | PROPOSED | REWORK; static discriminator only |
| production-not-slow gate on candidate | VALIDATED | 396 passed, 8 skipped |
| gold gate | VALIDATED as executed | FAIL; five selected failures reproduced on HEAD |
| full default suite | VALIDATED as executed | FAIL; 54 failures, three C1-attributable |
| G-F10-INDEPENDENT-FLRW | retained FAIL | no reopen |
| G-HARNESS-INTEGRITY | retained FAIL | no movement |
| endpoint/scientific/public-production authority | FORBIDDEN | not earned |
| QKE and JAX forward-development claims | FORBIDDEN | out of scope |

## 15. Residual risks and completeness ceiling

1. The raw tail data establish a numerical-domain problem but do not yet distinguish
   the relative contributions of scalar error weighting, finite-difference Jacobian
   probes, interpolation, and collision cancellation.
2. No high-precision independent collision-on trajectory was executed in this run.
3. No logit, entropy-stable, Patankar, or invariant-projection implementation was
   attempted.
4. The measured optional environment is version-pinned but not fully hash-locked.
5. Only five non-C1 failures were replayed on exact HEAD. The other 46 non-C1
   failures have traceback/changed-surface attribution, not full exact-HEAD replay.
6. The native extension was not built, and the isolated worktree lacks historical
   Git revision `0496e5e`; native/full-history failures therefore remain explicit.
7. Static audit completeness excludes generated, external, dynamic, and unlisted
   non-Python call surfaces.
8. Passing production-not-slow cannot promote a candidate that fails its focused
   physical trajectory and full suite.

## 16. Final disposition

**Do not merge or reuse the C1 patch as a production fix.** Preserve it as a
reproducible rejected candidate and as evidence that the current raw-state contract
is incompatible with the current collision-tail coordinate/error control.

The next admissible action is a separately frozen, small, physics-first positivity
coordinate or invariant-preserving discriminator. It must use the exact recorded
collision-on case as a mandatory falsifier, retain raw output, compare against a
tight independent reference, and stop on any Pauli, first-law, detailed-balance,
tail-QoI, or cost failure. Only a pass may request implementation authority.

Until then:

```
C1 = REJECTED / REWORK
C2 = REWORK
C3 = REWORK
G-F10-INDEPENDENT-FLRW = FAIL
G-HARNESS-INTEGRITY = FAIL
gate board = 6 PASS / 2 FAIL
scientific/public/endpoint promotion = NOT EARNED
```
