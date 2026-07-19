---
name: ssot-handoff-maintainer
description: Use when updating project state, handoff packets, next-session prompts, archive bundles, decision logs, status documents, or source-of-truth documentation after research/code changes.
---

# SSOT / Handoff Maintainer Skill

## Purpose

Prevent context loss across long research sessions.

Use this skill when:
- a project state changed;
- a design decision was made;
- files/artifacts were created;
- validation was run or skipped;
- a next-session prompt is needed;
- a handoff packet or archive bundle is requested.

## Required workflow

1. Locate controlling source-of-truth documents.
2. Update project state.
3. Separate:
   - facts,
   - derived conclusions,
   - conditional conclusions,
   - open risks,
   - deprecated ideas.
4. Update validation status if commands were run.
5. Update next actions.
6. Generate or update a next-session prompt.
7. Preserve file provenance.

## Required files

Prefer this structure:

```text
docs/harness/
  PROJECT_STATE.md
  CLAIM_LEDGER.md
  VALIDATION_LEDGER.md
  DECISION_LOG.md
  DEPRECATED_IDEAS.md
  NEXT_SESSION_PROMPT.md
```

## Output format

```markdown
## Updated handoff files

## Current status

## Decisions recorded

## Validation recorded

## Open risks

## Next session prompt
```

## Hard prohibitions

- Do not rewrite history to make failed paths look successful.
- Do not delete deprecated ideas without recording why they were abandoned.
- Do not merge speculative ideas into accepted design without explicit status change.
