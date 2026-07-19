---
name: research-plot-summary
description: Use when code changes produce or should produce plots, diagnostics, smoke figures, convergence plots, phase plots, residual plots, or physical interpretation figures.
---

# Research Plot Summary Skill

## Purpose

Ensure numerical/physics changes are accompanied by interpretable visual diagnostics when useful.

Use this skill when:
- a solver, model, likelihood, hierarchy, kernel, or parameter scan changes;
- a user asks for physical interpretation of code changes;
- a validation plot, smoke plot, phase plot, residual plot, or comparison plot is needed.

## Required workflow

1. Identify the changed physical/numerical quantity.
2. Choose the minimum useful diagnostic plot:
   - residual vs parameter,
   - convergence vs resolution/cutoff,
   - phase portrait,
   - benchmark comparison,
   - error ratio,
   - before/after delta.
3. Generate the plot from a script, not by hand.
4. Save:
   - plot image,
   - script,
   - input data or seed,
   - short interpretation.
5. Label plot as:
   - SMOKE,
   - DIAGNOSTIC,
   - VALIDATION,
   - PUBLICATION-CANDIDATE.
6. Never overclaim the plot category.

## Required output

```markdown
## Plot summary

## Quantity plotted

## Script / data provenance

## Interpretation

## What this plot does not show

## Next plot needed
```

## Hard prohibitions

- Do not call a smoke plot validation.
- Do not hide axis units.
- Do not use unexplained arbitrary normalization.
- Do not overwrite previous plots without preserving provenance.
