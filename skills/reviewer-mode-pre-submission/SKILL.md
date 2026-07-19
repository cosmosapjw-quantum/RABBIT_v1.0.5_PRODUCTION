---
name: reviewer-mode-pre-submission
description: Use when performing adversarial peer-review style critique of a manuscript, report, proposal, design document, or pre-submission research package.
---

# Reviewer Mode / Pre-submission Audit Skill

## Purpose

Simulate a severe but fair reviewer before submission or release.

Use this skill when:
- reviewing a manuscript;
- assessing novelty/scope;
- checking if a design is implementation-ready;
- preparing response-to-reviewer;
- deciding whether a package is submission-track.

## Required reviewer roles

If the work is technical, evaluate from these angles:

1. Domain expert:
   - physics/math validity,
   - novelty,
   - scope.
2. Numerical methods reviewer:
   - stability,
   - convergence,
   - reproducibility.
3. Software/reproducibility reviewer:
   - code availability,
   - artifact provenance,
   - tests.
4. Skeptical editor:
   - why should the journal/committee care?
   - what is the minimal publishable claim?

## Required output

```markdown
## Verdict

Choose one:
- PASS
- PASS WITH MINOR REVISIONS
- MAJOR REVISIONS
- REJECT / NOT READY
- INTERNAL ONLY

## Summary

## Strengths

## Fatal blockers

## Major concerns

## Minor concerns

## Novelty and scope

## Required validation

## Minimum revision plan

## Suggested downclaims

## Suggested stronger claims
```

## Hard prohibitions

- Do not flatter.
- Do not rescue a weak paper by vague rhetoric.
- Do not conflate ambition with demonstrated contribution.
- Do not ignore missing validation just because the idea is interesting.
