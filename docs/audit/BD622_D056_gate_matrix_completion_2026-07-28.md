# BD622 D-052–D-056 — F-10 gate-matrix completion (2026-07-28)

Under the D-052 omnibus grant, three staged programs closed the last two
FAIL gates. All work on comparator bytes `760a7c04` (D-053 final static
catalogue). All contracts prospectively frozen before output; all
adjudications recomputed the numbers.

## Stage 1 — Row-9 closure (D-053, PASS)

All pair conversions now carry both ordered orientation members at
`K_t`/`16.0` (catalogue 27, fingerprint `(2,1,6,2,2,2,4,4,4)`). The e/mu
self-sector artifact closed to `3.53e-11`; mu/tau regression `2.45e-11`;
CP improved to `3.14e-11`. Freeze `94d0f00`; run
`run-20260728-f10-d053-row9-closure`.

## Stage 2 — Covariance metrology (D-054 FAIL preserved; D-055 r2 PASS)

First contract (`46ef477`) FAILED on two adjudicated design defects: a
bitwise-zero electron lemma (realized `~1e-14` relative roundoff) and a
degenerate equilibrium state (relative metrics = noise/noise). FAIL
preserved; the r2 reissue (`904acb2`) changed exactly those two items per
the adjudicated supersession scope and PASSED on all three P-closed
states: weak identity max `1.144e-15`, mass-weighted max `2.965e-15`,
native max `2.448e-11` within both the fresh seven-term `B_native`
enclosure (measured summation via a Higham envelope over an
absolute-value shadow assembly, basis and interval via 80-bit longdouble
mirrors, transform, conditioning `A_cond = 2.994e3`, denominator, family
envelope) and the `1e-10` cap. **`G-F10-COVARIANCE-METROLOGY`: fail →
pass.** Runs `run-20260728-f10-d054-covariance-metrology` (report
`f456f3dd`) and `run-20260728-f10-d055-metrology-r2` (report `4c6e3aba`).

## Stage 3 — Independent trajectory (D-056, PASS)

Structurally independent stack (affine GL48-Y24 vs the Rust exponential
N48 rule; cloglog vs logit state; SciPy BDF vs Rust BDF/Rodas5P;
NumPy/SciPy vs Rust; independent adaptive-quad EOS and constants)
integrated the collision-coupled FLRW system from 10 MeV equilibrium to
the `T_gamma = 0.005 MeV` cold endpoint: 2:45:28 wall, 3694 evaluations.

| Observable | Independent | Rust N48 BDF anchor | Delta | Band |
|---|---|---|---|---|
| `N_eff` | 3.034054308 | 3.0333103242 | `+7.44e-4` | `3e-3` |
| `N_end` | 7.936698865 | 7.9367214190 | `-2.26e-5` | `3e-3` |
| `t_end` | 52678.732 s | 52680.2048 s | `-1.47 s` | `15 s` |

Heating ratio `1.3990542` (below instantaneous-decoupling `1.40102`, as
physics requires); blockwise enhancements e `9.37e-3` > mu/tau
`3.85e-3` > 0 with mu–tau agreement `4.69e-10`; first law `<= 8.31e-15`
at every evaluation. The endpoint also sits within band of the Rodas5P
partner anchor. Pre-freeze quadrature discriminator bounded the reduced
4/4/24 trajectory config at `3.94e-5` net-transfer agreement vs the
`3e-3` band. **`G-F10-INDEPENDENT-FLRW`: fail → pass.** Freeze
`5d9f521`; run `run-20260728-f10-d056-independent-trajectory` (report
`8ea58a3a`).

## Terminal state

```text
G-F10C1-RADIAL            PASS
G-F10C1-REGRESSION        PASS
G-F10-PERFORMANCE         PASS
G-F10-CATALOGUE           PASS
G-F10-INDEPENDENT-FLRW    PASS   (D-056)
G-F10-COVARIANCE-METROLOGY PASS  (D-055)
G-F10-SCOPE               PASS
G-HARNESS-INTEGRITY       PASS
```

Limitations of record: matched-resolution N48-class agreement on the
frozen 10 MeV → 5 keV cell; no continuum claim; single execution/platform
for the Python stack; `rtol 1e-6`; same-model reviews throughout.
Unblinding, public/production claims, W7/B3, T01–T12, GL64/Radau,
Rust/JAX forward work, F-11/Bianchi, and QKE remain closed — separate
owner decisions under `G-F10-SCOPE`.
