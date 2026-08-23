# VALIDATION_MATRIX.md

| Requirement | Test/check | Level | Expected | Status | Evidence |
|---|---|---|---|---|---|
| import 가능 | smoke import | software | pass | PASS | X7 imported all five focal test modules under current runtime. |
| 핵심 기능 | targeted test | software | pass | PASS | X7: 40 passed; characterization only. |
| 기존 동작 | regression suite | software | unchanged | NOT_RUN | |
| 단위 일관성 | dimension check | scientific | exact | NOT_RUN | |
| 알려진 극한 | analytic-limit test | scientific | within tolerance | NOT_RUN | |
| 보존법칙 | invariant test | scientific | within tolerance | NOT_RUN | |
| 수렴성 | resolution sweep | numerical | expected order | NOT_RUN | |
| 안정성 | timestep/solver sweep | numerical | stable range | NOT_RUN | |
| 재현성 | fixed-seed rerun | operational | reproducible | NOT_RUN | |
| 성능 | reference benchmark | operational | within budget | NOT_RUN | |
| X1 exact identity/environment | git/version/hash custody | reproducibility | exact current receipt | PASS | HEAD/tree/dirty custody, Python/NumPy/SciPy/mpmath, harness/seed/focal hashes recorded in E-C01. |
| X2 adaptive transaction | bounded counterexamples | software/numerical | fine state, typed shrink, event-safe, endpoint-bound history | FAIL | E-C03: coarse state accepted; failure did not shrink; h_min overran event; rejected diagnostics contaminated; arbitrary history committed. |
| X3 event completeness | analytic roots + installed solver/helper | numerical | discover or return uncertified; no false roots | FAIL | E-C04: two simple and grazing roots missed with success; project helper produced false/missed roots. |
| X4 stable primitive | multiprecision local oracle | numerical | stable forward/JVP error across zero | FAIL | E-C05: JVP relative error reached 6.5e19; zero-distance path bypassed validation. |
| X5 scale/null/outcome | metamorphic and fabrication probes | software/numerical | scale invariant and fail closed | FAIL | E-C06: unit/basis dependence, mutable identity, fabricated commit, nonfinite restart, invalid policies and structural false passes. |
| X6 history complexity | source complexity + boundary probe | operational | future query rejects; nonquadratic design required | FAIL | E-C07: future one-slice query became thermal zero; >=140,231,525,000 B append-copy lower bound; nonhomogeneous JVP branch guard. |
| X7 targeted baseline | existing focused tests | software | characterize current behavior | PASS | 40 passed in 4.58s; not an admission gate. |
| seed inventory completeness | mechanical N/R ID enumeration | research/reproducibility | N1--N6 and consecutive R01--R56, no gaps/duplicates | PASS | Seed SHA and inventory script recorded in E-C01; 6 hypotheses and 56 remedy rows preserved. |
| current-source adjudication | source/caller/test/probe matrix | research | every R01--R56 has current disposition and kill gate | PASS | 45 CONFIRMED, 11 CORRECTED, 0 CLOSED; corrected rows retain conditional/inconclusive subclaims explicitly. |
| bounded design selection | maximum-three comparison | design | one candidate with residual risks and no sufficiency overclaim | CONCERN | Program C remains the selected candidate; the sole reviewer returned REWORK and the one reconciliation has not been independently re-reviewed. |
| future gate coverage | V01--V16 atomic mapping and discriminators | design | every row mapped; positive and negative controls; no circular oracle | CONCERN | Mapping is complete. Reconciliation added independent clock/owner/dependency/denominator oracles, positive controls, crash tests and thresholds, but post-review adequacy is not independently approved. |
| implementation | source/test diff | software | user authorization required | NOT_APPLICABLE | Research-only loop; Phase 5 excluded. |
| independent gate | fresh design review | review | non-inconclusive tiered verdict preserved | CONCERN | Sole verdict REWORK. Mechanical coverage passed; required design/gate/DAG corrections were addressed once. No second review and no promotion claim. |
| reconciliation budget | required-review finding matrix | review | at most one repair round | PASS | One round updated clock/DAE/certificate/outcome/overlay/event/identity/dependency/crash/DAG contracts; no repository changes. |
| harness/contract closeout | harness validator + bounded contract check + custody | reproducibility | PASS and unchanged checkout | PASS | Reconciled harness validation PASS; bounded acceptance/hash/budget check PASS; exact HEAD/tree and sole pre-existing dirty file unchanged. |

Status: `PASS / CONCERN / FAIL / NOT_RUN / NOT_APPLICABLE`
