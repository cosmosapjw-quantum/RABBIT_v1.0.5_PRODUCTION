# BD622 D-050 — OWNER-C row-6 closure implementation contract (2026-07-27)

Prospectively frozen before any validation output. Owner grant: OWNER-C
(2026-07-27) — modify `src/rabbit/decoupling/_independent_noqke.py` (and its
focused test file) to carry the D-046/D-048-validated two-ordered-member
orientation closure for the mu/tau pair-conversion event (row 6).

## Planned edit (frozen)

In `independent_self_events()`, the single mu/tau `pair_conversion` event
`(nu_mu, antinu_mu -> nu_tau, antinu_tau)` with `K_t`, `32.0` is replaced by
**both ordered orientation members at coefficient `16.0` each** (the derived
Reynolds one-half quotient applied to the frozen absolute factor `32.0`):

- `(nu_mu, antinu_mu, nu_tau, antinu_tau)`, `"pair_conversion"`, `"K_t"`, `16.0`
- `(nu_tau, antinu_tau, nu_mu, antinu_mu)`, `"pair_conversion"`, `"K_t"`, `16.0`

Catalogue count 24 -> 25; `independent_pair_row_fingerprint()`
`(2,1,6,2,2,1,4,4,2)` -> `(2,1,6,2,2,2,4,4,2)`; module count assertion and
docstrings updated with closure provenance. The 48-row target-directed
reaction catalogue is unchanged (aggregate coefficient x leg-count per
species remains `32.0`). The e-mu and e-tau pair-conversion events (row 9)
are **not** closed — the D-048 evidence covers row 6 only; row-9 orientation
closure is recorded as an open question requiring its own validation. The
static M0/M1 authority ceiling in the module docstring is unchanged. The
evaluator loop, kinematics, kernels, and electron sector are untouched.
Test file: event count 25, fingerprint, per-(category, species) aggregate
reconstruction, and explicit two-member closure assertions.

## Frozen acceptance checks

| ID | Statement | Threshold |
|---|---|---|
| C1 | Catalogue structure: 25 events; fingerprint `(2,1,6,2,2,2,4,4,2)`; exactly two `pair_conversion` members on the mu/tau pair, `K_t`/`16.0`, mutually reverse-ordered; aggregate reconstruction equals the unchanged 48-row catalogue | exact |
| C2 | Focused test file passes under the venv environment (python 3.12.3, numpy 2.4.4, scipy 1.17.1, pytest 9.0.3) | all pass |
| C3 | GL48-Y24 D-028 cell (system python 3.12.3 / numpy 2.4.2 / scipy 1.17.0, single-thread BLAS pinned by the script): `1e-12 <= mu_tau_residual <= 1e-10` AND relative agreement with the D-048 external-closure prediction `3.8807636157303184e-11` within `5e-1` (same-noise-class band: the in-catalogue accumulation is a fresh rounding realization — the recorded T-B parity is `1.01e-14` bitwise-false and a single external reassociation shifted the residual 7.7 percent, so a digit-level band would gate on float reassociation, not mechanism; the exact value and agreement are recorded) | all |
| C4 | Same run, gated: self signed/absolute number and energy ratios `<= 1e-12`; `first_law_residual <= 1e-8`; `charge_conjugation_residual <= 1e-10`; `entropy_production >= -1e-24`; exact values recorded (CP expectation `~3.49e-11` class) | listed |
| C5 | Common-FD equilibrium state at GL48 (split=False): H-normalized absolute number and energy `<= 1e-10` (mirror of the recorded S0 anchors `6.59e-12` / `1.56e-11` class) | both |

Decision rule (frozen): all of C1..C5 pass -> the module modification is
accepted; the frozen static physics gates of the comparator (the focused
test file) then pass on the new bytes. This does NOT flip
`G-F10-COVARIANCE-METROLOGY` (its weak/mass-weighted identity and
prospectively frozen `B_native` bound program remain unbuilt) and grants no
trajectory, endpoint, GL64, Radau, W7, B3, or unblinding authority — per
the module ceiling, trajectory/endpoint eligibility after passing static
gates is a separate future owner decision. Any check failure: record, no
threshold change, revert-or-stop per adjudication. Mechanical errors
rerunnable and recorded.

## Verification harness

`scripts/audit/d050_ownerc_closure_verification.py` (frozen with this
contract): runs C1, C3, C4, C5 and emits a JSON report; C2 via
`venv/bin/python -m pytest tests/test_independent_noqke_comparator.py`.
