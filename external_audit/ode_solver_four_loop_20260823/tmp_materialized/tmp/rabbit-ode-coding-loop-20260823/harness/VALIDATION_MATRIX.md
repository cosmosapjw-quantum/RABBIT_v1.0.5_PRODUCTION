# VALIDATION_MATRIX.md

| Requirement | Test/check | Level | Expected | Status | Evidence |
|---|---|---|---|---|---|
| seed identity | SHA-256 and ordered candidate extraction | provenance | exact | PASS | A-MAC-ADJ2 hash and ranked list recorded |
| source localization | registered mapper with exact refs | software design | every seed mapped | PASS | A-CH-LOC, SHA `8c74fb96...2625` |
| 34-ID coverage | ordered unique ledger | design completeness | 001..034 exactly once | PASS | A-CH-DESIGN, SHA `c8e8aba4...0630` |
| candidate bound | count materially distinct designs | design completeness | <=3 | PASS | exactly 3 |
| executable identity | 182-state authority vs 122-state Rust | scientific design | no substitution | CONCERN | C3 REWORK; no matching surface |
| import/syntax | smoke import | software runtime | pass | NOT_APPLICABLE | no code change |
| focused repair tests | adversarial property/mutation tests | software runtime | pass | NOT_RUN | future implementation only |
| regression suite | affected current tests | software runtime | unchanged | NOT_RUN | future implementation only |
| raw-state admission | invalid raw state cannot produce observable | scientific design | fail-closed | NOT_RUN | C1 future tests specified |
| event semantics | exhausted refinement cannot fabricate event | scientific design | fail-closed | NOT_RUN | C2 future tests specified |
| dimensions/units/signs | independent review of planned predicates and residuals | scientific design | unchanged/exact or blocked | CONCERN | C2 event units unresolved; C3 certificate absent |
| conservation/positivity | invariant projection plus raw/full-grid checks | scientific design | predeclared tolerance | NOT_RUN | candidate 3 future discriminator |
| equilibrium/null limit | analytic and discrete limiting cases | numerical design | predeclared tolerance | NOT_RUN | candidate 3 future discriminator |
| resolution/rank convergence | grid-family and rank holdout | numerical runtime | stable certified envelope | NOT_RUN | future discriminator only |
| QoI error | primal residual plus adjoint enclosure | numerical runtime | bounds independent reference delta | NOT_RUN | future discriminator only |
| full-prefix performance | physical start; N=0.14,0.22; N>=0.25 | endpoint runtime | <=5500 calls and <=64800 s | NOT_RUN | separately authorized governing run only |
| reproducibility | exact source/config/dependency identity | operational | immutable and rerunnable | NOT_RUN | future implementation only |
| independent review | adversarial design review | review | blocking findings explicit | FAIL | A-CH-ADV `MAJOR_REVISIONS`, SHA `1090179e...a7c1` |
| C1 review repair | accepted/probe boundary and discriminator tests | design repair | implementation-ready specification | PASS | root single repair-closeout; final adjudication accepted |
| C2 review repair | event units/tolerances/budgets/counters/cost | design repair | no unsafe implementation promotion | CONCERN | downgraded to REWORK; numeric authority unresolved |
| C3 review | executable identity and certificate | design review | no substitution | PASS | REWORK/no-code defended |
| final adjudication | registered adjudicator over design, review, and repair | review | fail-closed terminal disposition | PASS | A-CH-DECIDE, SHA `c0bcf5dd...ec21`; research loop complete |
| governing gate state | exact current gate summary | project authority | unchanged from baseline | PASS | `6 PASS / 2 FAIL`; both pre-existing FAIL gates retained |
| production/JAX/QKE/public | promotion or forward work | scope | forbidden | NOT_APPLICABLE | explicit out of scope |

Status: `PASS / CONCERN / FAIL / NOT_RUN / NOT_APPLICABLE`
