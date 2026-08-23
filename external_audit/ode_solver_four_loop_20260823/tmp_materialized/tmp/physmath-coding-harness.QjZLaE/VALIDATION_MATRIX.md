# VALIDATION_MATRIX.md

| Requirement | Test/check | Level | Expected | Status | Evidence |
|---|---|---|---|---|---|
| import 가능 | smoke import | software | pass | NOT_RUN | |
| 핵심 기능 | targeted test | software | pass | NOT_RUN | |
| 기존 동작 | regression suite | software | unchanged | NOT_RUN | |
| 단위 일관성 | dimension check | scientific | exact | NOT_RUN | |
| 알려진 극한 | analytic-limit test | scientific | within tolerance | NOT_RUN | |
| 보존법칙 | invariant test | scientific | within tolerance | NOT_RUN | |
| 수렴성 | resolution sweep | numerical | expected order | NOT_RUN | |
| 안정성 | timestep/solver sweep | numerical | stable range | NOT_RUN | |
| 재현성 | fixed-seed rerun | operational | reproducible | NOT_RUN | |
| 성능 | reference benchmark | operational | within budget | NOT_RUN | |

Status: `PASS / CONCERN / FAIL / NOT_RUN / NOT_APPLICABLE`
