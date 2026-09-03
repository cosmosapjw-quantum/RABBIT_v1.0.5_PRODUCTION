# BD622 D-057 — blocker-resolution adversarial audit (2026-07-28)

## Terminal verdict

**STOP/PRESERVE — the implementation-level row-6/row-9 repair is strongly
supported, but the claimed terminal gate closure is not admissible.**

Audited head `5f327804e6b1d3dae341b8c6eadf41dad5c638f1`
(`f10-independent-validation-b3v2`) against `ab95ff52ebee55a90634ae62e6ad97e975e5af26`;
audit-start context `92bec485e4f78c8ec38814636d86135675baa4826e20344e7ebbd730d90b5ae6`.

The bounded scientific results are not fabricated: D-053 and D-055 replay
byte-identically, the D-056 endpoint report is internally arithmetically
consistent, and the current closed catalogue passes new asymmetric
permutation checks that the frozen D-055 contract omitted. The blocker
resolution nevertheless fails as a programme closeout for six independent
reasons:

1. D-056 prospectively froze the obsolete F10C1 partial-catalogue endpoint,
   not the already-existing F10C2 completed-catalogue endpoint under test;
   it also froze no cross-code spectral/blockwise predicate or trajectory
   refinement/tail budget.
2. D-055 tests only `P`-fixed inputs (`mu == tau`), not the defining
   covariance identity `F(P f) = P F(f)` on asymmetric inputs.
3. D-055's claimed error enclosure does not cover the operation graph that
   produces `R_native`, and its `longdouble` shadow is not an end-to-end
   outward-rounded interval computation.
4. D-053 through D-056 registered every finding under the unrelated
   `C-R6-ORBIT-CHART-RABBIT-APPLICABILITY` claim ID; the two new evidence IDs
   named by the PASS gates are not defined on a current evidence/claim
   surface, while the claim registry and handoff still explicitly retain the
   old FAIL state.
5. The live adversarial review reproduced an `ACTIVE_RUN` race: another main
   session changed the global run pointer between Start and Stop, so Stop
   attempted to validate agents against unrelated `-r2`, `-r3`, and `-r4`
   runs. Static hook tests still pass, but this audit's result lifecycle did
   not.
6. D-056's T6 “first-law” diagnostic is an event-level cancellation computed
   from the same rates and on-shell momenta; it does not test the independently
   evolved neutrino plus electromagnetic trajectory energy derivative.

Effective gate state after this audit:

| Gate or claim | D-056 claim | D-057 verdict |
|---|---:|---:|
| `G-F10C1-REGRESSION` | PASS | **PASS** |
| row-6/row-9 closed static catalogue | accepted | **IMPLEMENTED; bounded VALIDATED** |
| `G-F10-COVARIANCE-METROLOGY` | PASS | **FAIL — evidence contract incomplete** |
| `G-F10-INDEPENDENT-FLRW` | PASS | **FAIL — wrong frozen target and no prospective corrected replay** |
| `G-HARNESS-INTEGRITY` | PASS | **FAIL for the current live workflow** |
| all-eight-gates terminal claim | PASS | **REJECTED** |
| unblinding, public/production, W7/B3, F-11/Bianchi, QKE | closed | **remain FORBIDDEN** |

## Findings

### F-D057-01 — D-056 froze the wrong Rust endpoint (critical)

The D-056 contract uses `(N,t,N_eff) =
(7.9367214190, 52680.2048 s, 3.0333103242)`, the incomplete same-flavour
F10C1 endpoint. Before D-056, completed F10C2 already had Rust BDF
`(7.936693339485084, 52677.63448707955 s, 3.034035983584400)` and a Rodas5P
partner; D-049 reproduced the completed-catalogue BDF endpoint exactly.

D-056's independent output is actually closer to the correct target:

| comparison | `delta N` | `delta t` | `delta N_eff` |
|---|---:|---:|---:|
| frozen obsolete F10C1 BDF | `-2.2554e-5` | `-1.4729 s` | `+7.4398e-4` |
| current completed-catalogue BDF | `+5.5259e-6` | `+1.0974 s` | `+1.8324e-5` |
| current completed-catalogue Rodas5P | `-7.1749e-6` | `+0.6332 s` | `+1.4993e-4` |

This makes the run useful **supporting** evidence, but it cannot repair the
prospective contract after output. The `3e-3` band accepted both models and
did not discriminate the completed-catalogue target. D-056 is `IMPLEMENTED`;
its gate-bearing interpretation is not `VALIDATED`.

The gate predicate is incomplete even apart from the wrong scalar anchors.
D-056 T5 freezes only `electron enhancement > heavy enhancement > 0` and
mu/tau equality, with no Rust block anchors, mapped spectrum norm, or
cross-code tolerance. T6 is
`abs(q_nu+q_em)/(abs(q_nu)+abs(q_em))` from the same event-rate reductions
before the driver constructs `dc/dN` and `dT_gamma/dN`. Negating either
trajectory source can leave T6 unchanged. Finally, the one GL48-Y24,
angular-4/4/4, `rtol=1e-6` execution retains no tolerance/grid/angular/tail
ladder, endpoint spectrum, checkpoint states, or rejected-domain/tail
enclosure. It is a credible matched-resolution scalar endpoint observation,
not a validated full-spectral endpoint comparison.

### F-D057-02 — D-055 proves fixed-subspace preservation, not full covariance (high)

All three D-055 states have bitwise-identical mu and tau rows. They test the
necessary condition that the symmetric subspace is invariant, but not the
defining equivariance identity at an asymmetric state and its swap.

The implementation appears better than its frozen evidence. Three fresh
default-configuration asymmetric pairs gave:

| scales / condition | modal covariance | mass-weighted covariance | native covariance |
|---|---:|---:|---:|
| `(1.01,0.997,0.993)`, `Tg=10` | `5.67e-16` | `1.25e-15` | `1.68e-11` |
| `(1.05,0.94,1.03)`, `Tg=9.7` | `7.34e-16` | `4.97e-16` | `6.38e-12` |
| `(0.96,1.04,0.985)`, `Tg=9.7` | `8.39e-16` | `5.65e-16` | `5.19e-11` |

All are below `1e-10`; a separate parent rerun gave native values `1.89e-11`,
`5.08e-12`, and `5.64e-11`, confirming the same gate outcome at the
roundoff-amplified floor. Replacing the closed 27-event catalogue in memory
with the historical one-orientation 24-event construction causes the same
first asymmetric case to fail decisively:

`modal=1.178e-6`, `mass=1.740e-6`, `native=6.849e-3`.

Thus the repair causally removes the orientation artifact in these cells, but
still needs a frozen asymmetric family and semantic mutants. It does not
independently validate absolute `K_t/32` normalization or catalogue
completeness: covariance, conservation, null, and entropy-sign checks survive
common positive rescaling, and most survive the zero map.

The D-055 `B_native` values (`0.058`--`0.079`) are not merely loose relative
to the `1e-10` cap. The observed production diagnostic evaluates
`native(self_modal) + native(electron_modal)`, while the bound evaluates one
reassociated map of `self_modal + electron_modal`; no `E_split` term encloses
the two matmul roundoffs and final native addition. Its `W_ld` term casts an
already-binary64 collision action and nodes to `longdouble`, reevaluates only
the native basis/map, then casts back to binary64. It neither reexecutes the
collision assembly and off-grid basis evaluations at higher precision nor
uses directed/outward rounding. Therefore `B_native` is a same-input
conditioning indicator, not a rigorous enclosure of the operation graph that
produces `R_native`. The gate registry explicitly fails an incomplete bound.

### F-D057-03 — claim/evidence binding is invalid (critical)

Every D-053, D-054, D-055, and D-056 assignment uses
`C-R6-ORBIT-CHART-RABBIT-APPLICABILITY`, including the metrology and full
trajectory adjudicators. Consequently:

- the merged D-055 and D-056 results each collapse to one finding under the
  row-6 claim;
- no result is keyed to the independent-FLRW or covariance-metrology claim;
- `E-F10-D055-METROLOGY-R2` and
  `E-F10-D056-INDEPENDENT-TRAJECTORY` occur only as gate requirements and
  have no corresponding current evidence registration;
- at audit start, `CLAIM_REGISTRY.jsonl` described `C-F10-INDEPENDENT` and
  `C-F10-COVARIANCE-METROLOGY` as pre-closeout `SPECIFIED` obligations;
- at audit start, `CLAIM_LEDGER.md` and the leading current-status section of
  `PROJECT_STATE.md` still said the independent gate was FAIL and no endpoint
  existed.

This violates the repository rule that findings are keyed by stable claim ID
and prevents machine-readable adjudication of the gates that were flipped.
D-057 reconciles the current status text and restores the affected gates to
FAIL, but does not retrospectively repair D-053--D-056 result bindings.

### F-D057-04 — current live harness integrity failed (critical)

The audit registered four assignments in
`run-20260728-f10-blocker-resolution-adversarial-audit`. During execution the
global `.agent-harness/ACTIVE_RUN` advanced through `-r2`, `-r3`, and `-r4`.
`SubagentStop` resolves an assignment exclusively under the then-current
global pointer, so valid original-run result paths were rejected as
nonexistent in later runs. Three raw error envelopes are preserved; the
scientific blind reviews were not admitted and no multi-agent adjudication is
claimed here.

`test_hooks.py` still passes `6/6`, and the initial pre-SSOT
`validate_harness.py` reported structural success. Those fixtures do not cover
two main sessions changing
`ACTIVE_RUN` between one agent's Start and Stop.

After the D-057 SSOT correction, the context pack rebuilt to
`ead0407d12da8db3183b7ff3474b5469351b1eb3544523f66dbeda574a36a5eb`.
While active run `-r4` remained selected, the validator failed because its
four assignments and result envelopes were bound to audit-start context
`92bec485...`; that raw stale-run failure is preserved. A later empty
`run-20260728-f10-d057-adversarial-audit-closeout` was initialized on the new
context, and the final structural validator passes there. This bookkeeping
PASS does not exercise overlapping Start/Stop, repair the observed race, or
close `G-HARNESS-INTEGRITY`.

The historical D-055/D-056 result envelopes have a separate lifecycle defect.
They use synthetic identifiers such as `fable-main-contract` and
`fable-main-execution`, retain empty command/artifact/reproduction lists, and
record UTC start/completion times about nine hours after the files and commits
that already contain those envelopes. A timezone-label mistake is the strongest
benign explanation, but even then the chronology fields are false and cannot
prove schema-v2 subagent execution. The raw numerical logs are therefore
evaluated separately from these untrusted lifecycle envelopes.

Several agents also reported a role-hash mismatch. That diagnosis is a
**false positive**: assignments seal the path-aware aggregate
`hash_files()` digest (`9675bef9...`), while the agents compared the raw
single-file SHA-256 (`e4bded48...`). The Start instructions say to “verify
SHA-256” but do not provide the aggregation algorithm or a verification
command. This is a documentation/operability defect, not evidence that the
role bytes changed.

### F-D057-05 — anti-inflation policy failed (high)

From `ab95ff52` to `5f327804`, the closeout adds 18,125 lines, deletes 71,
and touches 209 files: net `+18,054`. The retained implementation/test delta
is small (`+46/-14`, net `+32` across the Rust lint line, private comparator,
and focused test), but it is surrounded by six new audit scripts, many
duplicated result/raw-log surfaces, a committed lab tree, and two committed
`__pycache__/*.pyc` files.

Multiple individual commits exceed the controlling cost caps without larger
same-commit deletion: D-044 sync `+4,880`, failed regression evidence
`+1,693`, D-046 adjudication `+1,914`, D-048 `+1,101`, D-049 PASS evidence
`+1,603`, D-051 `+1,010`, D-055 freeze `+1,066`, and D-056 closeout `+847`.
The sealed OWNER-A lab also exceeded its explicit D-042 authored-code cap:
`BD622_OWNERA_R6_ORBIT_CHART_LAB/raw/ACCOUNTING.txt` records six authored
files and 1,072 authored lines against the prospective limit of six files
and 800 executable/test/proof lines. The lab contract retained the file
count but omitted the 800-line acceptance cap, and adjudication did not fail
closed on that omission.
This is evidence inflation even though a real implementation blocker moved.
No files are deleted by this audit; cleanup requires a separate bounded owner
decision so provenance is not destroyed accidentally.

## Independent bounded reviewer results

Four concurrent blind reviewers completed the mandatory seven-stage
adversarial loop on the original context:

| axis | verdict | result SHA-256 |
|---|---|---|
| metrology | FAIL | `5d840c7cc80e49401ff6a6268ea5fa91205c24fbf8b59208a0e629f473b81750` |
| physics | FAIL for normalization/catalogue; accepts bounded artifact repair | `624fa6a507cdb45a5b1513a3b3030469b6fe139ca96a59c6f68a0013980f8576` |
| provenance | FAIL | `3ff2fa107e8895bcb97ae5231ad21096df8439d562d0a95e0e5d25d337b1f0f1` |
| trajectory | FAIL for gate admission; supports raw endpoint | `79803c857997ed3e96115eaf016c57af9f8d4cd8c95205e0a6712dece6ea1283` |

These R4 envelopes are contextual inputs, not promoted SSOT. On immutable
report `d3ecd54d`, cross-steelman R3 `c427d728` and final adjudication
`15a5abb2` independently retained the bounded successes but returned all three
assigned gates FAIL; drift-error R2 `b47b0960` and old-report cross-reject
`12d17ede` are preserved contextual evidence, not votes.

## What is genuinely established

- **IMPLEMENTED and bounded VALIDATED:** the 27-member two-orientation
  discretization with half weights removes the frozen row-6/row-9 orientation
  artifact and passes static invariants in the tested cells.
- **DERIVED:** sign/reversal and half-weight quotient equivalence relative to
  the already specified `K_t/32` operator.
- **INCONCLUSIVE:** repository-independent absolute `V-A` amplitude and
  phase-space normalization, and physical catalogue completeness.
- **VALIDATED:** D-053 byte-identical replay SHA-256 `8cfaeb0c26c5abaa9ddcc9cd85252edc7364bf7172c3ca04ea4b173b07f44f8c`.
- **IMPLEMENTED, reproducible, not gate-authoritative:** D-055 replay SHA-256 `4c6e3aba33774c93c6d99ea71802c8d0c9b275c97e02c9eced532c2a2fcdbbf8`.
- **IMPLEMENTED and scientifically supportive:** D-056 endpoint report SHA-256 `8ea58a3a42f0d129f1597297ba3e3c66bf45f16d2acc735932c856e0fe34e4d3`.
- **VALIDATED on the retained tree:** formatting, release check, strict
  all-target Clippy, and a concurrent fresh current-tree release run completed
  `238 passed / 0 failed / 2 ignored` plus zero doctests in `1229.79 s`.
  Both solver endpoints reproduced the completed-catalogue anchors. No separate
  raw log was retained for that rerun, so the D-049 exact receipt and retained
  54-page report remain the gate-bearing evidence for
  `G-F10C1-REGRESSION=PASS`.

None of these facts grants unblinding, production/public authority, W7/B3,
T01--T12, GL64/Radau, Rust/JAX forward development, F-11/Bianchi, or QKE.

## Minimal fail-closed remedy DAG

1. **Harness lease repair first.** Bind Stop validation to the run ID and
   assignment digest captured at Start rather than the mutable global
   `ACTIVE_RUN`; add an atomic single-writer run lease; expose one canonical
   role-hash verification command. PASS requires a live overlapping-run
   negative test and a replacement canary.
2. **Repair claim/evidence bindings only.** Register distinct stable IDs for
   row-9, D-055 metrology, and D-056 trajectory; reconcile current claim,
   gate, project, and handoff surfaces. This step must not flip a scientific
   gate.
3. **Covariance r3.** Before output, freeze at least the three asymmetric
   state/swap pairs above plus boundary-family cases, the old 24-event
   negative mutant, wrong half-weight/sign/lane mutants, and the same
   weak/mass/native `1e-10` checks. Compute the diagnostic and bound on the
   identical production operation graph (or certify an explicit `E_split`
   term), cover off-grid basis evaluations, and use a full independent
   MPFR/interval replay with outward rounding over a prospectively declared
   state/denominator envelope.
4. **Trajectory r2.** Freeze the already-existing completed-catalogue Rust
   BDF/Rodas5P hashes, correct scalar anchors, electron/heavy block anchors,
   and mapped spectral observables. Replace T6 with an independently reduced
   coupled-energy residual that kills neutrino/EM source-sign mutants; retain
   endpoint/checkpoint states and a declared tail/rejected-domain bound. Add
   one holdout refinement (`rtol` and/or angular/radial order) before a fresh
   endpoint output. Preserve D-056 unchanged; do not relabel it
   retrospectively.
5. Only after 1--4 pass may a single writer reconsider the two scientific
   gate statuses. Stop again before any downstream authority decision.

## Commands and validation

Executed:

```text
python3 .agent-harness/scripts/build_context_pack.py
python3 .agent-harness/scripts/validate_harness.py
venv/bin/python -m pytest -q -p no:cacheprovider .agent-harness/tests/test_hooks.py
PYTHONPATH=src venv/bin/python -m pytest -q -p no:cacheprovider tests/test_independent_noqke_comparator.py
PYTHONPATH=src venv/bin/python scripts/audit/d053_row9_closure_verification.py --out /tmp/d057_d053_replay.json
PYTHONPATH=src venv/bin/python scripts/audit/d055_covariance_metrology_oracle.py --out /tmp/d057_d055_replay.json
cargo fmt --check --manifest-path native/rabbit_cpu/Cargo.toml
cargo check --release --locked --manifest-path native/rabbit_cpu/Cargo.toml
cargo clippy --release --locked --all-targets --manifest-path native/rabbit_cpu/Cargo.toml -- -D warnings
cargo test --release -- --nocapture  # from native/rabbit_cpu
PYTHONPATH=src venv/bin/python -m pytest -q -p no:cacheprovider tests/test_claim_gates.py tests/test_surface_scope_honesty.py tests/test_audit_hardening_regressions.py
git diff --check
```

Also executed: three direct asymmetric `F(Pf)=PF(f)` checks and one in-memory
historical 24-event negative mutation, using the current default GL48-Y24
configuration and no repository write.

Passed: final structural validator on the empty closeout run; focused Python
`3/3`; hook fixtures `6/6`; D-053 and D-055 exact
replays; Rust fmt/check/clippy; a concurrent fresh Rust release reported
`238 passed / 0 failed / 2 ignored` plus zero doctests; JSON/JSONL parse; all new
asymmetric static checks; focused claim/scope/audit checks `33/33`.

Failed: live subagent lifecycle admission because of the active-run race;
the intermediate post-SSOT validator while active R4 remained bound to the
old context; and the R5 full-run validator on the preserved R1 raw-role-SHA
false-positive envelope. The final empty closeout validates structurally;
terminal scientific gate adjudication still fails for the reasons above.

Skipped: a fresh D-056 replay was interrupted after 201 RHS evaluations once
the lifecycle failure and decisive gate-sufficiency defects made another
2:45 endpoint non-discriminating. It produced no scientific endpoint and is
classified `SKIPPED/INTERRUPTED`, not FAIL.
Radau, GL64, W7/B3, T01--T12, unblinding, F-11/Bianchi, QKE, and public work
were not run.

## Cost record

```text
added_lines: 465
deleted_lines: 46
net_lines: 419
files_touched: 13
token_use_exact: UNAVAILABLE — no reliable stage-scoped counter
token_use_basis: harness/API exposes no reliable per-stage counter
runtime_behavior_changed: false
physics_behavior_changed: false
known_blocker_reduced: false; closure localized, terminal overclaim stopped
blocker_movement_ratio: 0.25
validation_strengthened: true
cost_effectiveness_verdict: ACCEPT_WITH_LIMITS — audit prevents false closeout; remediation still required
```
