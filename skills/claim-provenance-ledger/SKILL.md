---
name: claim-provenance-ledger
description: Use when auditing, writing, or updating research claims; separating implemented, validated, derived, specified, proposed, speculative, deprecated, and forbidden claims.
---

# Claim Provenance Ledger Skill

## Purpose

Prevent claim drift.

This skill keeps research documents honest by requiring each important claim to carry a status and evidence type.

## Claim labels

Use exactly these unless the project defines a stricter vocabulary:

- IMPLEMENTED: code exists and is integrated.
- VALIDATED: tested, benchmarked, reproduced, or independently checked.
- DERIVED: follows from a stated derivation/proof.
- SPECIFIED: design/spec exists, implementation not complete.
- PROPOSED: research direction or planned extension.
- SPECULATIVE: interesting but currently unsupported.
- DEPRECATED: no longer controlling.
- FORBIDDEN: explicitly disallowed.

## Required workflow

1. Extract strong claims from the changed document/code comments.
2. Assign each claim a status.
3. Identify evidence:
   - equation/proof,
   - test,
   - benchmark,
   - figure,
   - literature citation,
   - design document,
   - none.
4. Downgrade claims whose evidence is missing.
5. Update `docs/harness/CLAIM_LEDGER.md` if present.
6. Flag contradictions between docs and code.

## Required output

```markdown
## Claim audit

| Claim | Status | Evidence | Risk | Required fix |
|---|---|---|---|---|

## Upgraded claims

## Downgraded claims

## Contradictions

## Forbidden drift detected
```

## Hard prohibitions

- Do not allow SPECIFIED or PROPOSED claims to appear as IMPLEMENTED.
- Do not allow smoke tests to justify publication-grade validation.
- Do not allow hidden calibration/fudge factors to be described as physical derivation.
