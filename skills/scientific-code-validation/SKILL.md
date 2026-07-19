---
name: scientific-code-validation
description: Use when editing, reviewing, or validating scientific code, numerical simulations, tests, scripts, JAX/NumPy/SciPy/Rust/Fortran code, reproducibility harnesses, or generated numerical artifacts.
---

# Scientific Code Validation Skill

## Purpose

Prevent fake validation, numerical drift, hidden calibration, and untested scientific claims.

Use this skill when:
- code is edited;
- tests are added or modified;
- numerical results, plots, tables, or benchmark claims are produced;
- a solver, integrator, likelihood, hierarchy, kernel, projector, or data pipeline changes.

## Required workflow

1. Identify changed files.
2. Classify the change:
   - pure docs,
   - tests only,
   - numerical method,
   - physics model,
   - plotting/reporting,
   - infrastructure.
3. Identify the smallest relevant validation command.
4. Run validation if possible.
5. Capture exact commands and outputs.
6. Check for mock/demo/calibration leakage.
7. Check reproducibility:
   - random seed,
   - dependency versions,
   - input data path,
   - tolerances,
   - hardware assumptions.
8. Summarize physical/numerical impact.
9. Update `docs/harness/VALIDATION_LEDGER.md` if present.

## Required output

```markdown
## Validation summary

## Changed files

## Commands run

| Command | Result | Notes |
|---|---:|---|

## Numerical impact

## Reproducibility notes

## Failures / skipped checks

## Remaining risks

## Next validation step
```

## Hard prohibitions

- Never claim a test passed without running it.
- Never use a single multiplicative fudge/calibration factor to hide a structural residual.
- Never treat toy/smoke plots as publication-grade evidence.
- Never silently relax tolerances.
- Never overwrite reference data without explaining why.
