# BD622 D-049/D-051 — Lint fix, regression gate PASS, and OWNER-C closure acceptance (2026-07-27)

## D-049 — regression gate closed

One `#[allow(clippy::too_many_arguments)]` attribute on the `cfg(test)`
Gauss–Kronrod helper `gk_recurse` (commit `1697e76`, zero codegen change)
fixed the sole D-045 bundle failure. Full bundle on the fixed tree (run
`run-20260727-f10-d049-regression-fix`): fmt/check/clippy clean, release
tests 238 passed / 0 failed / 2 ignored (880.45 s), 54-page report with
clean scan and page 24/25/32/33 inspection. **`G-F10C1-REGRESSION`
flipped fail → pass** (commit `c054080`).

## D-050/D-051 — OWNER-C row-6 closure accepted

Frozen contract:
[BD622_D050_ownerc_row6_closure_contract_2026-07-27.md](BD622_D050_ownerc_row6_closure_contract_2026-07-27.md)
(freeze commit `d22d001`, before any validation output; blind diff review
fixed one deterministic harness blocker and one unmeetable agreement band
pre-freeze). Edit: the mu/tau pair conversion in
`independent_self_events()` now carries **both ordered orientation members
at `K_t`/`16.0`** (derived Reynolds one-half quotient on the frozen
`32.0`); catalogue 24 → 25; fingerprint `(2,1,6,2,2,2,4,4,2)`; row 9
unchanged; evaluator and authority ceiling untouched. Module sha256
`29a652fd` (was `535370c1`).

Acceptance (run `run-20260727-f10-d050-ownerc-closure`, adjudication
`3aa9b7d2`, merged `98340dcc`):

| Check | Result |
|---|---|
| C1 catalogue structure | exact (25 events, 2 members, aggregate reconstruction) |
| C2 focused tests (venv) | 3/3 incl. the frozen D-028 executing-gate assertion |
| C3 GL48 D-028 cell | `mu_tau_residual = 2.6019033208963758e-11` ∈ `[1e-12, 1e-10]`; agreement `0.330 ≤ 5e-1` |
| C4 diagnostics | number `1.97e-16`, energy `4.76e-17`, first law `0.0`, CP `4.587059566179814e-11`, entropy `≥ 0` |
| C5 equilibrium null | `6.49e-12` / `1.54e-11` (recorded S0 class) |

```text
VERDICT: PASS — BYTES ACCEPTED
TERMINAL: STATIC_GATES_PASS_ON_CLOSED_SOURCE
```

## Unchanged and open

The historical D-028 FAIL for bytes `535370c1` stands as a true record.
`G-F10-COVARIANCE-METROLOGY` remains FAIL — its weak/mass-weighted
identity and prospectively frozen `B_native` bound program are unbuilt
(do not reuse the D-028 `1/y^2` model or `~1.2e-8` cap).
`G-F10-INDEPENDENT-FLRW` remains FAIL. Three owner decision points are
open: the metrology program, trajectory/endpoint eligibility on the
accepted static bytes (newly live), and row-9 orientation-closure
validation. Limitations of record: same-model reviews, single execution
per check, CP margin halved on this draw, same-noise-class (not
digit-level) prediction agreement, W5/D-047 scripts non-rerunnable
against the new bytes.
