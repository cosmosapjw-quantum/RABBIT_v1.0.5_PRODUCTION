# BD622 D-053 — Row-9 orientation-closure contract (2026-07-28)

Prospectively frozen before any validation output, under the D-052 omnibus
grant, stage 1. Edit: close the remaining single-ordered-member pair
conversions — `(nu_e, antinu_e -> nu_mu, antinu_mu)` and
`(nu_e, antinu_e -> nu_tau, antinu_tau)` — with the same two-ordered-member
construction validated at D-046/D-048 and implemented for row 6 at
D-050/D-051: both ordered members at `K_t`/`16.0` (derived Reynolds
one-half quotient on the frozen `32.0`). All three pair conversions then
carry uniform closure; the special-case branch is removed. Catalogue
25 -> 27; fingerprint `(2,1,6,2,2,2,4,4,2)` -> `(2,1,6,2,2,2,4,4,4)`;
per-(category, species) aggregates unchanged (each flavour pair still
contributes `32.0` per species). Evaluator loop, kinematics, kernels,
electron sector, and the static M0/M1 authority ceiling untouched. Test
file updated to the uniform-closure assertions.

Justification for extending beyond row 6 without a per-row replay: the
closure mechanism was validated generically on manufactured objects
(D-046) and on the live evaluator for row 6 (D-048); rows 6 and 9 are
structurally identical single-ordered pair-conversion events differing
only in flavour labels, and the self-sector is flavour-symmetric by
construction (identical kernels and coefficients). The discriminating
check E3 below directly measures the row-9(e,mu) artifact and its removal
on the live evaluator, playing the same role T-D/T-H played for row 6.

## Frozen acceptance checks

| ID | Statement | Threshold |
|---|---|---|
| E1 | Catalogue: 27 events; fingerprint `(2,1,6,2,2,2,4,4,4)`; exactly six `pair_conversion` members forming three mutually-reverse ordered pairs at `K_t`/`16.0`; two-sided per-(category, species) aggregate reconstruction equals the unchanged 48-row catalogue | exact |
| E2 | Focused test file passes under the venv environment | all pass |
| E3 | e/mu discriminator cell (GL48-Y24, T=10 MeV, scales `(0.995, 0.995, 1.01)` so the e and mu logits are bitwise identical): self-sector e/mu residual `rel_Linf(pair_self[e], pair_self[mu]) <= 1e-10`; negative control: the external single-member row-9(e,mu) evaluation has native pair residual `> 1e-10` (order-one artifact expected) | both |
| E4 | D-028 cell regression: `1e-12 <= mu_tau_residual <= 1e-10` (fresh rounding realization on the new bytes; exact value recorded) | range |
| E5 | D-028 cell invariants: self number/energy ratios `<= 1e-12`; `first_law_residual <= 1e-8`; `charge_conjugation_residual <= 1e-10`; `entropy_production >= -1e-24`; equilibrium H-normalized number/energy `<= 1e-10` | listed |

Decision rule (frozen): all of E1..E5 pass -> the bytes are accepted as the
final static catalogue for the D-052 stages 2-3. Any failure: record, no
threshold change, revert-or-stop per adjudication. Mechanical errors
rerunnable and recorded. No gate moves at this stage.

## Verification harness

`scripts/audit/d053_row9_closure_verification.py` (frozen with this
contract; BLAS pins self-set): E1, E3, E4, E5; E2 via
`venv/bin/python -m pytest tests/test_independent_noqke_comparator.py`.
