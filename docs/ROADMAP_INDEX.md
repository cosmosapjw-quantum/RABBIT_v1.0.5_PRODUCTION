# RABBIT Development Roadmap — Master Index

> **Historical navigation index (PUB-00, 2026-07-12).**  This file indexes
> provenance and landed-work documents; it no longer selects future PRs.  The
> sole normative publication-code order is
> [TYPEI_AUGMENTED_NOQKE_UNIFIED_FUTURE_PLAN.md](TYPEI_AUGMENTED_NOQKE_UNIFIED_FUTURE_PLAN.md).

**Purpose.** Historical navigation for RABBIT design, landed-work, and audit
records.  Current publication-code work is intentionally excluded from the old
multi-document roadmap authority and is ordered only by the unified plan.

This index points to historical subordinate roadmap documents and the one
current publication-code specification:

| # | Document | Topic |
|---|---|---|
| 1 | [ROADMAP_STATE_OF_RECORD.md](ROADMAP_STATE_OF_RECORD.md) | What has been built, what decisions were made, current parity numbers, file inventory. |
| 2 | [ROADMAP_PR_WBS.md](ROADMAP_PR_WBS.md) | Historical pre-PUB-00 PR list; provenance only. |
| 3 | [ROADMAP_SELF_AUDIT.md](ROADMAP_SELF_AUDIT.md) | Per-PR self-audit checklist template + documentation-update automation. |
| 4 | [ROADMAP_PR_CATALOG.md](ROADMAP_PR_CATALOG.md) | Completed and planned PR catalogue with provenance, parity deltas, and staged plan entries. |
| 5 | [phase_prompts/README.md](phase_prompts/README.md) | Historical phase prompts; do not execute as current instructions. |
| 6 | [IMPLEMENTATION_GUIDE_TYPEI_AUGMENTED_PSTF_NOQKE_WBS.md](IMPLEMENTATION_GUIDE_TYPEI_AUGMENTED_PSTF_NOQKE_WBS.md) | Historical SDD/WBS ledger for the Type I nonperturbative augmented-PSTF full-Boltzmann no-QKE programme; keep for provenance after BD186. |
| 7 | [TYPEI_AUGMENTED_NOQKE_CODEX_ANTI_DRIFT_GUARDRAILS.md](TYPEI_AUGMENTED_NOQKE_CODEX_ANTI_DRIFT_GUARDRAILS.md) | Mandatory pre-read for further augmented Type-I no-QKE work; blocks gate-only PR inflation and defines the breakthrough WBS/DAG. |
| 8 | [TYPEI_AUGMENTED_NOQKE_UNIFIED_FUTURE_PLAN.md](TYPEI_AUGMENTED_NOQKE_UNIFIED_FUTURE_PLAN.md) | Sole active publication-code specification and PUB/M/N/B dependency order. |

The roadmap set cross-references the existing topic guides:

- [JAX_CHAR_GPU_OPTIMIZATION_PLAN.md](JAX_CHAR_GPU_OPTIMIZATION_PLAN.md)
  — GPU optimization strategy for the characteristic-ray driver.
- [IMPLEMENTATION_GUIDE_NON_LRS_TYPEI.md](IMPLEMENTATION_GUIDE_NON_LRS_TYPEI.md)
  — non-LRS (generic Σ_-) extension plan.
- [IMPLEMENTATION_GUIDE_3T_THERMO.md](IMPLEMENTATION_GUIDE_3T_THERMO.md)
  — three-temperature thermodynamics (now implemented; see `ROADMAP_PR_CATALOG.md` PR-T3T).
- [IMPLEMENTATION_GUIDE_FULL_COLLISION_NOQKE.md](IMPLEMENTATION_GUIDE_FULL_COLLISION_NOQKE.md)
  — fully nonperturbative incomplete decoupling (no QKE).
- [IMPLEMENTATION_GUIDE_TYPEI_AUGMENTED_PSTF_NOQKE_WBS.md](IMPLEMENTATION_GUIDE_TYPEI_AUGMENTED_PSTF_NOQKE_WBS.md)
  — historical SDD/WBS ledger for the augmented distribution, PSTF/S_N
  decomposition, ell_max convergence, and implementation provenance.
- [TYPEI_AUGMENTED_NOQKE_UNIFIED_FUTURE_PLAN.md](TYPEI_AUGMENTED_NOQKE_UNIFIED_FUTURE_PLAN.md)
  — sole active publication-code plan for the PUB/M/N/B dependency sequence.

Historical report snapshot: `docs/RABBIT_report/RABBIT_report.pdf`
with build companion `docs/RABBIT_report/main.pdf`. These PDFs are provenance,
not current publication evidence or a canonical physics authority; the unified
future plan and ledgers control present claims.

## Navigation

- Starting fresh on publication-code work → read the anti-drift guardrails,
  then `TYPEI_AUGMENTED_NOQKE_UNIFIED_FUTURE_PLAN.md`.
- Continuing augmented Type-I no-QKE work → read
  `TYPEI_AUGMENTED_NOQKE_CODEX_ANTI_DRIFT_GUARDRAILS.md` and then
  `TYPEI_AUGMENTED_NOQKE_UNIFIED_FUTURE_PLAN.md` before touching code or
  adding another AP/FB artifact.
- Picking up a current PR → use its frozen spec in the unified plan; old WBS
  and phase prompts are read-only provenance.
- Closing a current PR → update the existing claim and validation ledgers and
  record evidence in the PR body.  Do not add another per-PR audit surface.

No document in this set contains calendar timelines (weeks, dates).
Each PR is sequenced by dependencies, not schedule.
