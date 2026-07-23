# BD622 D-038 Historical Adversarial Root-Cause Audit

Date: 2026-07-23

Run: `run-20260723-f10-d038-historical-adversarial-audit`

Audit context: `2321fda01cdd5ef03c84b059ef9ddd368e95edc52b99c6f9c79bedf894cff001`

Protocol SHA-256: `8c40864753ed8f3813c47972221fea03abce624d9e56f216c18573678381cdef`

Final adjudication SHA-256: `c145f298ac65dbcf76676542ff4ebcbb85f14a157dd956a6e0b2f71d2c0a3e6f`

## Verdict

The historical audit is complete; the current independent-validation route is
**FAIL**. Exact-byte remediation readiness, W7 execution, B3 implementation,
and independent-FLRW closure are each **FAIL**. Technical repairability is only
`PROPOSED`; D-038 grants no remediation or execution authority.

The strongest supported causal statement is narrower than “one bug repeated.”
The current D-035/W6 candidate has a specification-to-executable-contract root
defect. Across Fort, D-027, D-028, and D-029, architecture-before-closed-
invariants is a high-confidence recurrence amplifier, not a universal root
cause. Those candidates retain distinct proximate failures and rational bounded
discovery value.

## Scope and evidence order

This was a read-only current/legacy Git and artifact audit. No W7 output, B3
collision output, T01--T12, Rust collision execution, Radau, trajectory,
endpoint, unblinding, F-11, QKE, or public-runtime work ran.

Evidence was ordered as current executable source and exact machine results,
historical blobs/diffs/raw runs, current SSOT, historical prose, then commit
narrative. The sources included the 12-commit curated Git, preserved blocker
and solver-harness branches, the verified 37-ref legacy bundle/full bare Git,
quarantined historical harness runs, and the exact D-037 envelopes.

Every substantive auditor performed the required internal loop:
`self-discover -> step-back -> metacognitive self-ask -> CoVe -> adversarial
self-ask -> CCoT -> PDR`. Results expose only checkable evidence, falsifiers,
contrast tables, and PDRs, not private chain-of-thought.

## Registered evidence

| Assignment/result | Status | SHA-256 | Adjudicated use |
|---|---:|---|---|
| `A-HISTORY-MAPPER` | error | `7beb87a448dc42b7d9c72eeea5415696ef522a179e7a80a804692c7c80c3904b` | raw-vs-aggregate role-hash false positive; process only |
| `A-HISTORY-MAPPER-R2` | error | `f7c47490c409afb7c10f40aea6cb17357d055fac893ba075a306fa260fa75087` | nonexistent manually named role input; process only |
| `A-HISTORY-MAPPER-R3` | fail | `7e0dc3a7aeb067178ac9a17d9db86f81da4aa65b134237e5f1217ac2afb14e02` | substantive, qualified by disclosed idempotent write attempt |
| `A-PHYSICS-DESIGN-AUDITOR` | fail | `b49b7bf0fe244313e16899af92d08fb26a956573a9c4c57125a38c2d55f94302` | blind physics/design axis |
| `A-SYSTEMS-PROCESS-AUDITOR` | fail | `92104e01d649e8439191ddbaf60f9d2e8e35cd08bc60665019b3ff93aadfdca0` | blind numerical/process/provenance axis |
| `A-STEELMAN-DESIGN` | fail | `af052d01138a332f2b249dab7abb99bd6c3641ca0c551d1e04e35f67b5832d74` | design-evolution steelman/debate |
| `A-STEELMAN-AUDIT` | fail | `0c1153c7c5859877991baebdf236df980d112dc6b47901aca8ccb0d13bbefa30` | systemic-audit steelman/debate |
| `A-ADJUDICATOR` | fail | `c145f298ac65dbcf76676542ff4ebcbb85f14a157dd956a6e0b2f71d2c0a3e6f` | final adjudication |

Exact deduplication by `(claim_id, evidence_fingerprint, verdict)` produced
`32 raw / 32 unique / 0 duplicate` findings. Three findings in the first two
mapper envelopes are process-only. The remaining overlap is semantic, so the
adjudicator mapped every exact finding once into seven causal families.

## Historical reconstruction

| Interval | Observation | Classification |
|---|---|---|
| 2026-07-17 Fort design | Constants, layout, physical mapping, event, and norm definitions were not executable before output. | executable-contract antecedent |
| Fort r2--r6 | Cache/authority defects were separated from the real strict-lower-node interpolation defect; the later common-FD statistic was direction-dependent `0/0`; no endpoint ran. | process defects plus one scientific proximate cause and one oracle defect |
| D-027 | Pointwise target-leg deposition passed a null case but failed resolved non-null number/energy/exchange gates. | non-adjoint deposition proximate cause |
| D-028 | Conservative Galerkin-Petrov passed all recorded gates except native mu-tau covariance `4.666064e-10 > 1e-10`. | rational empirical discovery; native small-`y` conditioning/covariance proximate cause |
| D-029 | Fixed-triple theorem did not cover the changing selector, support switch, prior, exact-node branch, and Jacobian. | successful preimplementation rejection |
| 2026-07-19 reroot | Reachable history, full legacy Git, bundle, dirty patches, and quarantine were preserved. | auditability amplifier; science non-cause |
| D-033--D-035 | Successive packets corrected genuine normalization, `G_F`, orientation, finite-mass, basis, ladder, and metrology defects. | real partial progress, incomplete whole-contract closure |
| D-036--D-037 | Schema-v2 admission passed; exact current B3/W7 bytes retained four bounded passes and seven binding failures. | decisive current-source evidence |

## Deduplicated causal findings

| Family | Classification | Adjudicated finding |
|---|---|---|
| Candidate-specific mechanisms | proximate causes | Fort interpolation, D-027 deposition, D-028 native conditioning, and D-029 theorem-to-selector mismatch are distinct; none is a harness or reroot artifact. |
| Current D-035/W6 | root cause | Per-leg shells, one coefficient-Horner basis, directed signed-tail enclosure, common outward denominator/A4 overlap, and canonical trace completeness are binding but bypassable, duplicated, absent, or caller-defined. |
| Early oracles | detection gap | Null/local identities, nominal vectors, counts, and object equality preceded the cheapest non-null/native/branch/all-consumer/tail/trace negative controls. |
| Programme sequencing | recurrence amplifier | Last-defect packetization and additive surface preceded a finite whole-obligation graph; “architecture-first” is sustained here, not as a universal historical root. |
| Harness and isolation | process root; science non-cause | D-031--D-036 exposed real admission/write-attribution defects, but D-037 reproduced source failures after repair. |
| Reroot and layered SSOT | auditability/detection amplifier; science non-cause | Exact commit dating is weakened, while preserved bundles/hashes and current counterexamples retain the verdict. |
| Fail-closed stops | safety benefit; non-cause | Stops prevented uninterpretable W7/B3/endpoint output and preserved genuine partial findings. |

## Steelman and debate disposition

Two agents completed six rounds: agreed facts, design steelman, audit steelman,
cross-examination, cost/sequence, and a joint decision tree. Each accepted the
opposing case and made at least three material concessions.

The design case prevailed on two limits: D-028 was mostly rational empirical
discovery, D-029 was a successful early stop, and the history does not prove B3
impossible. The audit case prevailed on the current candidate: all seven D-037
failure classes were already binding at D-035 sealing, no listed closure class
is optional for a rigorous W7 bound, and a larger implementation push has
negative evidence value while static contradictions remain.

The only unresolved issue is retrospective causal weight: how many later tests
were cheaply knowable at each earlier choice point. There is no operational
disagreement and the current hard stop is unaffected.

## Readiness and authority

| Route | Verdict | Reopen boundary |
|---|---:|---|
| Exact-byte remediation | **FAIL** / `PROPOSED` option | New explicit owner decision plus a finite closure matrix and one-cycle stop rule. |
| W7 execution | **FAIL** / `FORBIDDEN` | Fresh unanimous whole-design PASS, then a separate owner W7 decision. |
| B3 implementation | **FAIL** / `FORBIDDEN` | Executed W7 evidence, fresh adjudication PASS, then a separate owner B3 decision. |
| Independent-FLRW closure | **FAIL** / `SPECIFIED` | Static, trajectory/endpoint, covariance-metrology, and current-tree regression gates must pass; authority cannot substitute for evidence. |

`G-HARNESS-INTEGRITY` remains PASS only inside the D-036 schema-v2
parent/Start/Stop boundary. Mapper R3 ran `build_context_pack.py` despite its
result-only assignment. The tracked diff remained exactly
`59e739df67dcc9443b24e26512363f3333c4438729b0b707cf08978da1b2d755`,
so no tracked content mutation is established; the idempotent write attempt is
nonetheless retained as a process deviation.

## Constructive next decision

Default: **STOP/PRESERVE**.

If the owner explicitly chooses `ONE_FINAL_DESIGN_ONLY_REMEDIATION`:

1. Freeze one finite clause-to-symbol-to-operation-to-evidence-to-hostile-
   mutation matrix before editing.
2. Touch only the existing W6/source/vector surfaces; consolidate duplicate
   basis, tail, trace, and validator paths; add no gate, wrapper, telemetry,
   policy, or numerical-output surface.
3. Require two blind readers to reconstruct the same executable graph and to
   reject unequal-scale shell rescue, cached-node consumers, scalar/missing
   tail remainder, and incomplete/noncanonical A0--A4/A4 traces.
4. Run one fresh exact-byte whole-design review with a first-non-PASS stop.
   All seven current classes must close, all four passes must remain, and no new
   binding class may appear.
5. Any non-PASS, ambiguity, or new class terminates and archives the route. A
   unanimous PASS changes design readiness only and still requires a separate
   owner decision before W7.

## Cost and validation accounting

- Added lines: `218`.
- Deleted lines: `22`.
- Net lines: `+196`.
- Files touched: `8` (one audit report, five existing SSOT/ledger surfaces,
  and two rebuilt context surfaces).
- Runtime behavior changed: no.
- Physics behavior changed: no.
- Scientific gate changed: no.
- Token use exact: `UNAVAILABLE — no reliable stage-scoped counter`.
- Adjudicator-only blocker movement ratio: `0.00` because it changes no gate.
- Workflow-level blocker movement ratio: `0.25` because root/proximate causes,
  non-causes, stop-loss, and the one-cycle falsifier are materially localized.
- Cost-effectiveness verdict: `ACCEPT_WITH_LIMITS` for the bounded audit packet;
  `STOP_DEFAULT` for the route. A larger implementation/test push is not
  cost-effective and remains forbidden.

No production, test, gate, spec, or scientific source file is changed by this
audit.
