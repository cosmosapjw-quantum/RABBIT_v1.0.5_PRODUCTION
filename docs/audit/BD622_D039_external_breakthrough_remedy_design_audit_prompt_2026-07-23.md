# BD622 D-039 External Breakthrough/Remedy Design Audit Request

Copy this document verbatim to the external auditor. It is a read-only design
request, not permission to remediate, implement, execute, or reopen F-10.

## 1. Binding owner decision

The owner has selected **STOP/PRESERVE** after D-038. Preserve the current
branch, dirty tree, failed exact bytes, raw failures, unfinished axes, legacy
Git, and user-owned evidence. This request does not activate D-038's optional
one-final-design-cycle path.

The current route verdicts are binding:

- exact-byte remediation readiness: **FAIL**;
- W7 execution/static implementation: **FAIL / FORBIDDEN**;
- B3 implementation or collision output: **FAIL / FORBIDDEN**;
- `G-F10-INDEPENDENT-FLRW`, `G-F10-COVARIANCE-METROLOGY`, and
  `G-F10C1-REGRESSION`: **FAIL**;
- `G-HARNESS-INTEGRITY`: **PASS only inside the validated schema-v2
  parent/SubagentStart/SubagentStop process boundary**, with no scientific
  implication.

Your work may produce only a `PROPOSED` breakthrough/remedy design, a
`SPECULATIVE` lead, or a reasoned no-go/stop conclusion. This report can at
most inform a later owner choice between `DO_NOT_REOPEN` and
`AUTHORIZE_ONE_FRESH_DESIGN_ONLY_REVIEW`. It cannot support repository
remediation, W7, B3, numerical, implementation, trajectory, or endpoint
authority.

The frozen downstream ladder is not compressible: a fresh whole-design blind
review must be unanimously PASS before a separate later owner decision could
authorize W7 execution; executed W7 evidence and a fresh adjudication PASS are
then required before another separate owner decision could authorize B3.

## 2. Audit objective

Determine whether a materially new, scientifically defensible route could
break the recurrent independent-validation failure pattern without becoming
another last-defect patch, wrapper, gate, or expensive output-driven search.
You may return at most three ranked candidates. Returning no candidate is a
valid and preferred result when the evidence does not support one.

A qualifying candidate must change the mathematical or executable-contract
mechanism, not merely increase precision, grid order, tolerance effort, test
count, reviewers, or documentation. It must address the complete static
obligation graph before numerical output and explain why it is not Fort,
D-027, D-028, D-029, or the sealed B3-v2/W7 route in disguise.

## 3. Repository and immutable evidence

- Repository:
  `/home/cosmosapjw/Dropbox/rabbit/RABBIT_v1.0.5_PRODUCTION`
- Branch: `f10-independent-validation-b3v2`
- Preserved base:
  `main@ca3a0138d496732edfc13559fd8f7ceec7ef4d6e`
- D-038 report:
  `docs/audit/BD622_D038_historical_adversarial_root_cause_audit_2026-07-23.md`
  (`860f9121bcd555fad95233959fc7b9e003caedd70fbad303ebb85f20564cec65`)
- D-038 adjudication:
  `.agent-harness/runs/run-20260723-f10-d038-historical-adversarial-audit/results/A-ADJUDICATOR.json`
  (`c145f298ac65dbcf76676542ff4ebcbb85f14a157dd956a6e0b2f71d2c0a3e6f`)
- Sealed W6 design:
  `docs/audit/BD622_W6_comparator_design_proof_pack.md`
  (`9caaa445f1606d893c5a9a7f226358b17f8a3cedd26664228cd4db0668e0ea94`)
- Unexecuted W7 contract source:
  `scripts/audit/w7_b3v2_contract_source.py`
  (`60fc9668a72bba0ef576138c17b1bfe4f435bc215b8850dde4ac424cdb66dfbe`)
- Exact vectors:
  `docs/audit/BD622_W7_exact_test_vectors.json`
  (`e991d415284da9f6d552e6d011739b0948aef03a3a47051e425602fcd3c6e3cd`)
- D-037 B3/W7 results:
  `A-B3-DESIGN.json` (`bead9788997a73e7d3a64b1527f11b3450be2c8c3a553bd757a20a73085854f4`)
  and `A-W7-METROLOGY.json`
  (`f785c2d21641a9ce40454986a514d0b7e8f90ab848d300650874ab8a70fafb9a`)
  under
  `.agent-harness/runs/run-20260723-f10-b3v2-d037-blind-review/results/`.

Also preserve, without editing, adding, or promoting:

- `RABBIT_f10_independent_validation_branch_review.md`
  (`a1742101b68649cb42f48ec98a069ea720721dea1a47affbcd1bf1b08c420575`);
- `.singularhistory`
  (`5abbc84202f8e5354776242626083a79d84e668f9cab6b87030765e6c5f94ae1`);
- `src/rabbit/decoupling/_independent_noqke.py`
  (`535370c1f7e2b10c2cd1c812a0bd83e2519606f1e526a146735348d41d723ac3`);
- every D-027/D-028/D-029 source, result, and adjudication.

Recompute every full hash you rely on. Treat a mismatch as provenance drift and
stop before drawing a design conclusion.

### Exact historical novelty manifest

The ignored `.publish-local` paths below are owner-machine evidence, not public
tree authority. If absent, report the evidence gap; do not rediscover or replace
them.

- Fort/history locator: `.agent-harness/runs/run-20260723-f10-d038-historical-adversarial-audit/results/A-HISTORY-MAPPER-R3.json` — `7e0dc3a7aeb067178ac9a17d9db86f81da4aa65b134237e5f1217ac2afb14e02`.
- Preserved legacy refs: `.publish-local/20260719T052443Z/history/rabbit-legacy-all-refs.bundle` — `9f22f318f50a994a0a17c11ec831c5beaeeb42b67fbcae8111f9fee41af21920`.
- D-027 result/adjudication: `.publish-local/20260719T052443Z/quarantine/.agent-harness/runs/run-20260718-f10-minimal-independent-noqke-r2-implementation/artifacts/M1_POINTWISE_STATIC_RESULT.json` — `cab73790f74d0ebe36ceb956adb26076949e896d496122ddc04d521c3ebe91c4`; sibling `ADJUDICATION.md` — `7e9b16d10436bb0d348524563c8acd475cfe8df951b15a0a51c1586d3a8adc7c`.
- D-028 result/adjudication: `.publish-local/20260719T052443Z/quarantine/.agent-harness/runs/run-20260718T095658Z/artifacts/GL48_STATIC_R1.json` — `4726c40e403aad42d15503b0ed11760558f98de06a8e6f54259ab219887c73db`; sibling `ADJUDICATION.md` — `e4fd24e22ca619e3af8052b35562cf33ed56e0a651101276f6fe4323cfe08048`.
- D-028 method/covariance adjudications: `.publish-local/20260719T052443Z/quarantine/.agent-harness/runs/run-20260718T095658Z/results/A-CONSERVATIVE-METHOD-ADJUDICATION.json` — `ab8b2bc73d005ced403a686b2cc0f53b05785dee968c58e1af6b8af495017b81`; sibling `A-GL48-COVARIANCE-ADJUDICATION.json` — `3002ad870c60cefcf0bdd3c20f990f025a2aa7fee64f7ae979ca95678b37acbd`.
- D-029 derivation/binary64: `.publish-local/20260719T052443Z/quarantine/.agent-harness/runs/run-20260718-f10-static-fail-closeout/results/A-THREE-NODE-MAXENT-DERIVATION.json` — `c336130ba4ba1e34bbc9b7b46e7c91f0b4a3817bd0a0d2cad43d0d9dbf61fd6b`; sibling `A-THREE-NODE-BINARY64-AUDIT.json` — `6f7744569e8b7c9347ae7c2f21e490c5748373c3ea4e99b87ef1f82cb4899f7d`.
- D-029 adjudication/reopen audit: `.publish-local/20260719T052443Z/quarantine/.agent-harness/runs/run-20260718-f10-static-fail-closeout/results/A-THREE-NODE-DESIGN-ADJUDICATION.json` — `614999cdf68a1f3492eb543a1a297454fa6c5f529463156ce7fd23e4ff94c159`; sibling `A-F10-METHOD-REOPEN-AUDIT.json` — `bf8ea677f31d3f8673bab46e348b77e942c7cbcb38a1b47cc150292c2310fba4`.

### Current control-surface manifest

These are prompt-generation hashes, not permission to prefer stale SSOT. Any
live mismatch is a drift finding and requires reconciliation before analysis.

- `.agent-harness/context/FROZEN_DECISIONS.md` — `6723d1093c9679ffce058e044a3288b80507afbe64fdec6decad11909498b0ee`.
- `.agent-harness/context/GATE_REGISTRY.json` — `ed10fd9cb00efb4b48a709ef7a9f3e0188e2a397b65b3c717a7791c1c3eda3b1`.
- `.agent-harness/context/CLAIM_REGISTRY.jsonl` — `53b445c6bc7eae8ee5dea3f3037c4dbd4b96d2243329bf1c15ed79d6e9ea49ee`.
- `docs/harness/PROJECT_STATE.md` — `e63e5a998875873182439b0e7586eb7cf1fba4040ee84c3122898950fedcff21`.
- `docs/harness/CLAIM_LEDGER.md` — `9658f5297a64411365bbf0c624a78a0c6e9ac819f2dbd552557053a67a2cffce`.
- `docs/harness/VALIDATION_LEDGER.md` — `4c9a325a7fa8280c6164238a3636ee2075f6479bfaa82cdf6ff1bdef69615850`.
- `docs/harness/DECISION_LOG.md` — `290e16d1ab65b9756df32fe37c7d5242c428a0bd00ba01706acd254b69aeaedc`.
- `docs/harness/NEXT_SESSION_PROMPT.md` — `57bd562afad21f82e493386e9041c5f6dfaf7765c424bc20bca102bb205c9da8`.

## 4. Required reads and evidence order

Read completely, in this order:

1. `AGENTS.md`,
   `docs/TYPEI_AUGMENTED_NOQKE_CODEX_ANTI_DRIFT_GUARDRAILS.md`, and
   `bbn_codex_anti_drift_cost_effective_policy.md`;
2. `.agent-harness/generated/CONTEXT_PACK.md`,
   `.agent-harness/context/{FROZEN_DECISIONS.md,GATE_REGISTRY.json,CLAIM_REGISTRY.jsonl}`,
   and `docs/harness/{PROJECT_STATE,CLAIM_LEDGER,VALIDATION_LEDGER,DECISION_LOG,NEXT_SESSION_PROMPT}.md`;
3. the D-038 report and adjudication, then the exact D-037 B3/W7 envelopes;
4. `docs/audit/BD622_W3_per_row_mb_closed_form_oracles.md`
   (`a8b76b17540011fca0d3a487132b8809c57c927a7ffdb0b5409b11d246c87b3a`),
   W5 (`83a37543ecbd2dc837f9089d5531d406d836c2b364b46d904cdcfb935b1a0eb0`),
   W6, the unexecuted W7 source, and exact vectors;
5. only the exact historical novelty manifest above.

D-037 stopped on its first admitted non-PASS. It did not complete four CAS
axes, merge results, or produce an adjudicator verdict. Preserve its four
bounded passes and seven binding failures without calling it a completed
four-axis adjudication.

Use this authority order: executable source and machine result; preserved
historical bytes/raw runs; current SSOT; historical prose; commit narrative.
Do not treat the supplied diagnosis as true merely because it is documented.

Start with only these safe commands:

```bash
git status --short --branch
git rev-parse HEAD main
python3 .agent-harness/scripts/validate_harness.py
sha256sum <each-exact-path-listed-above>
```

## 5. Hard prohibitions

Work read-only. Do not edit or create a repository file. Do not run a collision
kernel, W7 numerical oracle, T01--T12, MPFR output campaign, Rust collision
code, GL48/64 comparator, Jacobian, Radau, step, trajectory, endpoint, or
RABBIT unblinding. Do not start F-11, Bianchi/Type-I, QKE, public runtime, or
production work.

Do not propose:

- post-hoc projection, output symmetrization, cap/tolerance/grid tuning, or
  state-dependent rescue after observing results;
- more telemetry, manifests, wrappers, policy knobs, or standalone gates;
- wholesale FortEPiaNO revival or reuse of Rust/JAX/Fort/D-028 collision code;
- “use higher precision,” “add more tests,” or “use AI” without a new closed
  mathematical mechanism and an executable operation graph;
- any claim that owner approval can substitute for scientific evidence.

Primary-source literature review is allowed. Cite precise equations, algorithms,
and limitations, and separate literature support from repository evidence.

## 6. Questions the audit must answer

1. Does D-038 correctly distinguish the current executable-contract root from
   the distinct Fort/D-027/D-028/D-029 proximate causes? State the strongest
   falsifier of that conclusion.
2. Is there a no-go criterion showing that this validation target cannot be
   met cost-effectively under the frozen independence and binary64 constraints?
3. Can conservation, detailed balance, entropy, covariance, positivity,
   finite-`m_e` support, tail control, and native conditioning be built into one
   semidiscrete representation rather than checked as downstream repairs?
4. Can one authoritative clause-to-symbol-to-operation compiler or proof
   object eliminate the present prose/source/vector divergence without adding
   another wrapper or duplicated truth surface?
5. Is a genuinely different external comparator possible while retaining the
   exact physics, six-species catalogue, normalization, structural
   independence, and eventual FLRW relevance?
6. Which cheapest pre-output negative control would kill each proposed route
   before implementation or numerical output?

## 7. Candidate contract

For each of no more than three candidates, provide:

1. name, status (`PROPOSED` or `SPECULATIVE`), and one-sentence hypothesis;
2. exact D-038 causal family and D-037 finding(s) addressed;
3. continuum assumptions, metric/gamma conventions, units, species ordering,
   normalization, spin/identical-particle factors, and branch choices;
4. the semidiscrete state, invariant measure, event orientation, reaction
   action, reconstruction/deposition duality, support/shell handling, tail
   enclosure, native map, conditioning model, and Jacobian existence;
5. a finite clause-to-symbol-to-operation-to-evidence-to-falsifier matrix;
6. a comparison against Fort, D-027, D-028, D-029, and B3-v2/W7 proving
   structural novelty rather than renaming;
7. limiting cases, conservation/nullspace, detailed-balance, entropy,
   covariance, positivity, dimensional, and permutation checks;
8. the first three hostile mutations or counterexamples and their expected
   rejection mechanism;
9. explicit falsifiers, no-go conditions, and a terminal stop-loss;
10. the smallest design-only artifact needed for blind review, with no
    executable scientific output;
11. estimated touched existing files, added/deleted/net lines, specialist and
    compute cost, evidence gain, and expected blocker-movement ratio;
12. the staged authority ladder, making clear that this audit grants none.

Prefer consolidation or deletion. A candidate that needs a new source of truth
must show which older surface it removes and why its net evidence value is
positive.

The matrix universe is closed: all four D-037 bounded passes, all seven D-037
binding failures, every current gate above, every declared consumer, and every
candidate operation/evidence/falsifier must have stable IDs and reported row
counts. Require bidirectional coverage and zero unmapped binding items. Any
omission, duplicate authority path, or unproved consumer makes the candidate
`INCONCLUSIVE` or no-go.

Each novelty claim must name an operation-level mechanism and one pre-output
discriminator that differs from Fort, D-027, D-028, D-029, and B3-v2/W7.
Renaming, higher precision alone, or a governance-only wrapper fails novelty.

## 8. Required adversarial method

Conduct four explicit review angles: early-universe kinetic theory,
discrete/numerical methods, software/reproducibility, and skeptical
cost/novelty editing. Internally perform:

`self-discover -> step-back -> metacognitive self-ask -> CoVe ->
adversarial self-ask -> CCoT -> PDR`.

Expose only checkable evidence, derivations, falsifiers, a concise contrast
table, and the post-decision review; do not expose private chain-of-thought.
Steelman both:

- the owner case that STOP/PRESERVE is already the highest-value decision; and
- the strongest materially new remedy case.

Then state which steelman survives cross-examination and what evidence would
reverse it.

## 9. Required output

Return one self-contained report with:

1. snapshot, hashes, tools/versions, and read-only commands;
2. route verdict:
   `STOP_CONFIRMED`, `MATERIALLY_NEW_CANDIDATE_PROPOSED`, or `INCONCLUSIVE`;
3. a claim ledger using only repository status vocabulary;
4. D-038 root-cause challenge and strongest counterevidence;
5. deduplicated failure-family and candidate comparison tables;
6. full candidate contracts for at most three ranked candidates, or a no-go
   analysis;
7. steelman/cross-examination disposition and PDR;
8. report-contract/editorial-quality verdict only—never candidate readiness or
   a design-gate verdict: `PASS`, `PASS WITH MINOR REVISION`,
   `PASS WITH MAJOR REVISION`, `REJECT`, or `INTERNAL ONLY`;
9. minimum revision set, claims to weaken, and claims that could be
   strengthened only by named future evidence;
10. exactly one proposed next owner choice: `DO_NOT_REOPEN` or
    `AUTHORIZE_ONE_FRESH_DESIGN_ONLY_REVIEW`;
11. added/deleted/net lines (`0/0/0` for a read-only audit), exact token use or
    `UNAVAILABLE — no reliable stage-scoped counter`, achieved scientific
    blocker-movement ratio fixed at `0.00`, and cost-effectiveness verdict.
    Any future candidate movement must be a separately labelled estimate with
    assumptions, never achieved evidence.

Do not end with a generic recommendation to “continue research.” End with one
bounded owner choice and its falsifiable stop condition. Until the owner makes
a later explicit choice, **STOP/PRESERVE remains binding**.
