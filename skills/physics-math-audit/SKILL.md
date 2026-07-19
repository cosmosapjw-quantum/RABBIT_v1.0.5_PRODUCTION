---
name: physics-math-audit
description: Use when reviewing or editing physics/math derivations, equations, assumptions, conventions, dimensional analysis, signs, limits, tensors, GR/cosmology/statistical mechanics arguments, or theoretical claims.
---

# Physics / Math Audit Skill

## Purpose

Audit theoretical physics and mathematical content before it is treated as reliable research material.

Use this skill when the task includes:
- equations, tensor/index notation, variational principles, GR/cosmology, statistical mechanics, kinetic theory, QFT-inspired analogies;
- derivations or theorem/proposition style claims;
- sign, unit, convention, or limiting-case risks;
- reviewer-style critique of a theoretical argument.

## Required workflow

1. Identify the controlling assumptions.
2. Identify conventions:
   - metric signature,
   - units,
   - index placement,
   - Fourier/spherical harmonic normalization,
   - frame/gauge/observer choices,
   - sign conventions.
3. Check dimensions/units.
4. Check signs and normalization.
5. Check limiting cases.
6. Check boundary/initial conditions.
7. Check whether the stated conclusion follows from the stated assumptions.
8. Separate:
   - proved/derived claims,
   - plausible but unproven claims,
   - speculative interpretations,
   - implementation-dependent claims.
9. Produce an audit ledger.

## Default conventions

Unless the project says otherwise:
- metric signature: `(-,+,+,+)`;
- keep `c`, `hbar`, `k_B`, and `G` explicit unless natural units are explicitly declared;
- never silently switch units or conventions.

## Output format

```markdown
## Short verdict

## Assumptions and conventions

## Equation-by-equation audit

| Item | Status | Issue | Fix |
|---|---:|---|---|

## Dimensional / sign / limit checks

## Fatal blockers

## High-priority fixes

## Safe claims

## Claims requiring downscoping

## Minimal revision plan
```

## Hard prohibitions

- Do not call a result “proved” unless the proof is present or reconstructible from stated assumptions.
- Do not hide a convention change.
- Do not infer physical validity from formal analogy alone.
- Do not treat numerical evidence as derivation.
