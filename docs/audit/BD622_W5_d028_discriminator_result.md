# BD622 W5 — D-028 μτ-covariance discriminator result (E4 + E5)

Status: **VALIDATED localization evidence** (owner-authorized static run on the FROZEN comparator
bytes; no gate changed, no frozen decision altered, comparator source unmodified). Harness:
`scripts/audit/w5_d028_discriminator.py`.
Reproduces `evaluate_independent_collision_action` at the recorded GL48-Y24 split-scale S1 state
(scales 1.01/0.995/0.995), default full-degree config, T_cm=T_γ=10 MeV.

## Result

```
grid GL48-Y24  y_min=1.474791e-02  1/y_min^2=4.598e+03
[D-028 observable] native mu_tau relative-Linf = 4.784280e-10   (cap 1e-10)  -> FAIL (reproduced)
[E5 dual-norm] modal-coeff residual            = 5.260348e-13   -> PASS
[E5 dual-norm] measure-weighted (m*A) residual = 5.894571e-13   -> PASS
[E4] argmax node j=0  y[j]=1.474791e-02  (the smallest node)
[E4] subgrid relative-Linf:  y>=0: 4.78e-10 | y>=0.078: 2.98e-11 | y>=0.19: 6.49e-12
                             y>=0.5: 1.52e-12 | y>=1: 1.12e-12
VERDICT: METROLOGY LOCALIZATION / 1-over-y^2 AMPLIFICATION
```

## Interpretation

- **The D-028 FAIL is reproduced** (native 4.78e-10, same magnitude class as the recorded
  4.666e-10; minor difference = config/BLAS/library drift from the quarantined `GL48_STATIC_R1`,
  immaterial to the mechanism).
- **The identical data passes at ~5e-13 in the well-conditioned weak metric** (modal coefficients
  and measure-weighted `m_j A_j`, where the `1/y²` cancels), matching the audit's predicted
  ~1e-13 floor. This localizes the recorded failure to the native reconstruction path, but does
  not by itself prove that the underlying weak residual is pure roundoff.
- **The residual is localized at the smallest node** (`y_min=0.0147`, `1/y²=4598`) and **collapses
  ~300× when `y<0.5` is excluded** — the fingerprint of small-node `1/y²` amplification.
- At the S1 state the μ,τ logits are bitwise-equal, so exact-arithmetic covariance should be
  **0** for a canonical flavour-covariant event graph. The observed native residual is consistent
  with amplification of a much smaller weak residual. Multi-precision replay and arbitrary
  flavour-role swaps remain required before calling the weak residual exclusively roundoff.

## Consequences for the solution (both settled by this run)

1. **F-1 / F-2 localized with limits** — the condition-blind native `1/y²` map amplifies the
   recorded residual and is unsuitable for a universal `1e-10` pointwise cap. This does not
   exclude an event-order or species-bookkeeping defect beneath the amplification.
2. **No reversal or cap transfer** — the weak-content value near `5e-13` is useful localization
   evidence, while the historical `~1.2e-8` native proposal was not a prospectively frozen B3
   cap. B3-v2 must derive its own `B_native` from its basis, mass solve, reduction tree, interval
   enclosure, denominator, and state envelope before output. D-028's recorded FAIL is unchanged.

## Scope / hardening note

E4 + E5 establish localization (the weak metric passes; the native fails; the maximum is at the
smallest node). They do not establish root-cause exclusivity. E1 multi-precision replay and E3
arbitrary role swaps are mandatory inputs to the successor method's metrology contract.
