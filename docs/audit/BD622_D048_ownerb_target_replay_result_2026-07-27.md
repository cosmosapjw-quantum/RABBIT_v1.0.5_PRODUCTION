# BD622 D-048 — OWNER-B target replay result (2026-07-27)

Frozen contract:
[BD622_D047_ownerb_target_replay_contract_2026-07-27.md](BD622_D047_ownerb_target_replay_contract_2026-07-27.md)
(freeze commit `9c99ecf`; contract `8f29a2b0`; script `c90a7e4d`; live module
`535370c1`, byte-untouched throughout). Run
`run-20260727-f10-d047-ownerb-target-replay`; report `89938c3c`;
adjudication `b2912459`; merged `110a00e4`. Single frozen execution,
`PYTHONPATH=src`, single-thread BLAS pins, wall 18.88 s.

## Gated results

| Check | Threshold | Value | Result |
|---|---|---|---|
| T-A reproduction | in (1e-10, 1e-8) | `4.666064056497196e-10` (digit-exact vs the recorded GL48_STATIC_R1 value) | PASS |
| T-B external-vs-live row-6 parity | <= 1e-13 | `1.0122040631628722e-14` (bitwise false: BLAS shape dispatch) | PASS |
| T-C identity, D-028 state | <= 1e-13 | `0.0` bitwise | PASS |
| T-C2 identity, asymmetric state | <= 1e-13 | `0.0` bitwise | PASS |
| T-D closed row-6 antisymmetry | <= 1e-13 | `0.0` | PASS |
| T-E closed-total residual (headline) | <= 1e-10 | `3.8807636157303184e-11` | PASS |
| T-F no-row6 localization | <= 1e-10 | `4.1812242095923674e-11` | PASS |
| T-G closed self-conservation | <= 1e-12 | number `1.97e-16`, energy `2.86e-16` | PASS |
| T-H negative control | > 1e-10 | `1.0000171560718123` | PASS |

## Established (internal, two frozen states, GL48-Y24/10 MeV)

The D-028 native mu-tau covariance residual is the **row-6
single-ordered-member orientation artifact amplified by the `1/y^2` native
map**: the single member carries an order-one native pair antisymmetry
(T-H); deleting row 6 alone brings the total under the cap (T-F); the
D-046-derived Reynolds one-half closure of the two ordered members removes
the artifact exactly at row level (T-D) and brings the closed total to
`3.88e-11` (T-E); the closure's live-code anchor
`modal_M-(f) == P.modal_M+(Pf)` holds bitwise on both states (T-C/T-C2);
conservation is preserved at machine epsilon (T-G).

## Not established

No `B_native` metrology bound; no general-state proof beyond the two frozen
states; no trajectory, endpoint, `N_eff`, or production claim. The D-028
recorded FAIL for its exact bytes stands unreopened.
`G-F10-COVARIANCE-METROLOGY` remains FAIL: its weak/mass-weighted identity
and prospectively frozen `B_native` bound are unbuilt.

## Limitations of record

Same-model blind review/verification (procedural independence only); single
deterministic execution (no fresh-process replay pair); T-B bitwise-false;
the sub-cap `4.18e-11` floor undecomposed beyond row 6; the digit-exact T-A
reproduction under numpy 2.4.2/scipy 1.17.0 attributes the W5 drift
(`4.78428e-10`) to the W5 run environment, which remains uncharacterized;
the asymmetric identity state is a 10 MeV adaptation of the 3 MeV W6
S-split profile, gating only the identity check.

## Terminal

```text
VERDICT: PASS
D046-r6-orbit-chart-rabbit-applicability = VALIDATED (bounded)
GATES: UNCHANGED
TERMINAL: REQUEST_OWNER_C_CONSIDERATION
```

`OWNER-C=REQUIRED_NOT_GRANTED`: only an explicit owner decision may
authorize modifying `_independent_noqke.py` (or an equivalent
implementation route) to carry the closure, followed by the metrology work
the gate requires.
