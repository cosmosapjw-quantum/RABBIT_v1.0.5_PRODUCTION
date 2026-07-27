# BD622 D-054 — Covariance-metrology contract (2026-07-28)

Prospectively frozen before any oracle output, under the D-052 omnibus
grant, stage 2. Object: the accepted comparator `_independent_noqke.py`
sha256 `760a7c04...` (D-053 final static catalogue). Decides
`G-F10-COVARIANCE-METROLOGY`. The D-028 `1/y^2` model and the historical
`~1.2e-8` cap are not consulted; every bound term below is derived fresh
and measured by the frozen oracle.

## Frozen cell and state family (P-closed envelope)

GL48-Y24, default config, `T_cm = T_gamma = 10 MeV`. The mu/tau block swap
`P` requires P-closed states; the frozen family is three states with
bitwise-identical mu/tau logit rows:

- S-A (D-028 cell): scales `(1.01, 0.995, 0.995)`
- S-B (equilibrium): scales `(1.0, 1.0, 1.0)`
- S-C (envelope point): scales `(1.02, 0.99, 0.99)`

## Frozen metrics

With `pair(X)` the half-sum charge-pair stack and
`D_* = max(||X_mu||_inf, ||X_tau||_inf, tiny)` the module's own
denominator convention:

- `R_weak` — weak (Galerkin modal) identity: rel-Linf of the mu vs tau
  pair rows of `modal_total`.
- `R_mass` — mass-weighted identity: rel-Linf of `m * A_mu` vs `m * A_tau`
  with `m = weights * nodes^2` on the native total (the `1/y^2` cancels).
- `R_native` — the module's native `mu_tau_residual`.

## Frozen bound (all terms measured; frozen factor 4 per W6 precedent)

`B_native = 4 * [ A_cond * (N_weak + B_sum + B_basis) + B_map + W_ld * D_native ] / D_native`

| Term | Gate requirement | Definition (measured by the oracle) |
|---|---|---|
| `N_weak` | summation (realized) | `|| modal_mu_pair - modal_tau_pair ||_inf` of `modal_total` |
| `B_sum` | summation (envelope) | Higham `gamma_n * S_abs`: `S_abs` = inf-norm of the absolute-value shadow assembly of the self sector over the mu/tau pair rows (abs of every rate and basis factor inside every reduction); `n` = measured total addend count (valid samples + nodes + events), `gamma_n = n*eps/(1-n*eps)` |
| `B_basis` | basis | `delta_basis * S_signed`: `delta_basis` = max relative float64-vs-longdouble discrepancy of the modal basis matrix on nodes; `S_signed` = max inf-norm of the signed modal pair rows |
| `B_map` | mass-solve/transform | `gamma_48 * max_i (|modal| @ |B_i|) / (prefactor * y_i^2)` — rounding envelope of the modal-to-native transform via absolute-value matmul |
| `A_cond` | conditioning | `max_i ||B(y_i,:)||_1 / (prefactor * y_i^2)` — the native map's modal-to-native amplification |
| `W_ld` | interval | `|| native64 - native_longdouble ||_inf / D_native` — end-to-end longdouble shadow enclosure width of the native pair rows |
| `D_native` | denominator | `max(||A_mu||_inf, ||A_tau||_inf, float_tiny)` |
| family max | state-envelope | every metric and term computed per state; PASS requires per-state satisfaction and the recorded family maxima |

## Frozen acceptance checks

| ID | Statement | Threshold |
|---|---|---|
| M1 | `R_weak <= 1e-10` for every family state | per state |
| M2 | `R_mass <= 1e-10` for every family state | per state |
| M3 | Structural lemma: the electron-sector modal mu/tau pair difference is bitwise `0.0` on every family state (flavour-symmetric code on bitwise-equal inputs); hence the native difference is carried by the self sector alone | exact |
| M4 | `R_native <= B_native` for every family state, with all seven terms recorded per state | per state |
| M5 | `R_native <= 1e-10` for every family state (the executing-gate native cap, for the record) | per state |

Decision rule (frozen): all of M1..M5 pass -> `G-F10-COVARIANCE-METROLOGY`
flips fail -> pass by single-writer update citing this run; the bound and
all terms are recorded in the ledger. Any failure: gate stays FAIL,
record, no term or threshold refitting. Mechanical errors rerunnable and
recorded. The bound is a conditioning enclosure, not an accuracy claim; no
trajectory/endpoint or production authority follows.

## Verification harness

`scripts/audit/d054_covariance_metrology_oracle.py` (frozen with this
contract; BLAS pins self-set; longdouble = x86 80-bit extended,
recorded).
