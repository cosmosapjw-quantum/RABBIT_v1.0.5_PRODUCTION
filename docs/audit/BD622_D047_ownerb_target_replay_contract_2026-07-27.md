# BD622 D-047 — OWNER-B target replay contract (2026-07-27)

Prospectively frozen before any replay output. Owner grant: OWNER-B
(2026-07-27) — one read-only target replay of the D-046-validated
`C-R6-ORBIT-CHART` closure construction against the live
`_independent_noqke.py` row-6 evaluator. `_independent_noqke.py` is imported
unmodified; OWNER-C remains ungranted; no gate moves regardless of outcome.

## Relation to the frozen D-028 decision

D-028 forbids symmetrizing outputs or averaging event orientations **as a
rescue of the failed static M1 verdict for those bytes**. This replay does
not reopen D-028: its recorded FAIL and exact bytes stand. The owner's
explicit OWNER-B grant authorizes a new, separately frozen experiment that
tests whether the D-046-validated two-ordered-member closure with the
*derived* (not fitted) Reynolds one-half quotient explains the D-028
residual mechanism. The D-044 objection ("a two-chart Reynolds average is
covariant for any wrong or zero base map") is answered structurally: the
quotient is derived (D-046, orbit-stabilizer), the identity
`C(f; M-) = P.C(Pf; M+)` is tested bitwise on the live evaluator on two
states, and localization plus a negative control keep the covariance PASS
from being constructional-only.

## Frozen cell

- Grid `build_independent_grid(48, 24.0)` (affine GL48-Y24);
  `IndependentCollisionConfig()` defaults (12/12/4/48);
  `temperature_cm_mev = temperature_gamma_mev = 10.0`.
- D-028 state: `pair_logits_to_cloglog(stack(-nodes/s for s in (1.01, 0.995, 0.995)))`
  — mu and tau logits bitwise identical (exact-arithmetic covariance target 0).
- Asymmetric identity state (informative cell, gated identity only): W6
  S-split logit profiles `e: -y + 0.08 y exp(-y/3)`,
  `mu: -y - 0.05 y^2/(1+y^2) exp(-y/5)`, `tau: -y + 0.03 sin(y/2) exp(-y/6)`
  evaluated on this GL48-Y24 cell at 10 MeV (recorded adaptation of the W6
  3 MeV definition).
- Environment: python 3.12.3, numpy 2.4.2, scipy 1.17.0,
  `OPENBLAS_NUM_THREADS=OMP_NUM_THREADS=MKL_NUM_THREADS=1`. The recorded
  GL48_STATIC_R1 value `4.666064056497196e-10` was produced under
  numpy 2.4.4/scipy 1.17.1; the W5 same-mechanism replay under drifted
  libraries gave `4.784280e-10`. T-A therefore bounds a range, not a digit
  match.

## Construction

External mirror of the `_assemble_self` event loop (module helpers
`_SpectralLogits`, `_two_body_kinematics`, `_event_measure`, `_self_matrix`,
`_stable_pauli_gain_minus_loss`, `_modal_product`, `_native_action`) applied
to exactly one caller-built `IndependentSelfEvent` with `kernel="K_t"`,
`coefficient=32.0`:

- `M+ = (nu_mu, antinu_mu -> nu_tau, antinu_tau)` (the live row-6 member);
- `M- = (nu_tau, antinu_tau -> nu_mu, antinu_mu)` (the absent reverse member).

`_self_matrix` reads only `kernel`/`coefficient`, so the matrix element is
identical by construction. Closure: `modal_closed = (modal_P + modal_M)/2`
(derived quotient 1/2). Closed total is formed modally
(`modal_total - modal_P + modal_closed`) and mapped once through the linear
`_native_action`. The mu-tau residual metric is the module's own
`_relative_max_difference` on the half-sum charge-pair native actions —
identical to `diagnostics["mu_tau_residual"]`.

## Frozen checks (all gated unless marked informative)

| ID | Statement | Threshold |
|---|---|---|
| T-A | Full live evaluation reproduces the D-028-class FAIL: `mu_tau_residual` in `(1e-10, 1e-8)` | range |
| T-B | External M+ native row-6 action matches `action.self_rows[6]` | rel-Linf <= 1e-13 (bitwise recorded) |
| T-C | Live-code lab identity on the D-028 state: `modal_M- == P.modal_M+` | rel-Linf <= 1e-13 (bitwise recorded) |
| T-C2 | Identity on the asymmetric state: `modal_M-(f) == P.modal_M+(Pf)` | rel-Linf <= 1e-13 (bitwise recorded) |
| T-D | Closed row-6 pair antisymmetry vanishes on the D-028 state | rel-Linf <= 1e-13 |
| T-E | **Headline**: closed-total native mu-tau residual meets the D-028 cap | <= 1e-10 |
| T-F | Internal localization: total minus row-6 alone meets the cap | <= 1e-10 |
| T-G | Closed self-sector conservation: signed/absolute number and energy ratios | <= 1e-12 each |
| T-H | Negative control: single-member row-6 native pair residual demonstrates the artifact | > 1e-10 |
| I | Entropy-production sums per member (both states), asymmetric closed number/energy ratios, CP residual, rejection count | informative |

## Decision rule (frozen)

- All of T-A..T-H pass → `D046-r6-orbit-chart-rabbit-applicability` upgrades
  `PROPOSED -> VALIDATED`, bounded to: the D-028 native mu-tau covariance
  residual at the frozen state is the row-6 single-ordered-member
  orientation artifact and is removed by the derived-quotient closure. No
  gate movement; no production, trajectory, endpoint, or normalization
  claim; `_independent_noqke.py` unmodified (OWNER-C separate).
- Any gated check fails → claim remains `PROPOSED` or is set `FAILED` per
  adjudication; record exact first failure; no retry with altered
  thresholds.
- Mechanical errors (exceptions) are fixable and rerunnable; every rerun is
  recorded. Thresholds and formulas above may not change after first output.

## Terminal

On PASS the replay may only request OWNER-C consideration; it grants
nothing. On FAIL the construction is not transferred and STOP/PRESERVE
controls.
