# BD622 W5 — D-028 μτ-covariance discriminator result (E4 + E5)

Status: **executed** (owner-authorized static run on the FROZEN comparator bytes; no gate changed,
no frozen decision altered, comparator source unmodified). Harness: `scripts/audit/w5_d028_discriminator.py`.
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
VERDICT: METROLOGY / 1-over-y^2 CONDITIONING (F-1/F-2 confirmed)
```

## Interpretation

- **The D-028 FAIL is reproduced** (native 4.78e-10, same magnitude class as the recorded
  4.666e-10; minor difference = config/BLAS/library drift from the quarantined `GL48_STATIC_R1`,
  immaterial to the mechanism).
- **The identical data passes at ~5e-13 in the well-conditioned weak metric** (modal coefficients
  and measure-weighted `m_j A_j`, where the `1/y²` cancels), matching the audit's predicted
  ~1e-13 floor. A genuine semidiscrete covariance defect would appear in the weak metric too; it
  does not.
- **The residual is localized at the smallest node** (`y_min=0.0147`, `1/y²=4598`) and **collapses
  ~300× when `y<0.5` is excluded** — the fingerprint of small-node `1/y²` amplification.
- At the S1 state the μ,τ logits are bitwise-equal, so the exact-arithmetic covariance is **0**;
  `4.78e-10` is the binary64 floor amplified by the native inversion. The `1e-10` cap lies inside
  that amplified floor band, so a physically-correct discretization fails it at roundoff.

## Consequences for the solution (both settled by this run)

1. **F-1 / F-2 empirically confirmed** — the recurrence engine is the condition-blind native
   `1/y²` gate, not a physics defect. This is now an executed result, not just source analysis.
2. **No reversal** — this is not an E1 precision-independent plateau; the metrology-first framing
   holds and the program proceeds. **W1's dual co-gate is justified**: G2a on the weak content
   reads ~5e-13 (PASS); a G2b native diagnostic with the conditioning-derived cap ~1.2e-8 reads
   4.78e-10 (PASS). D-028's recorded FAIL is **unchanged** (this run adds a diagnostic; it does not
   reclassify D-028).

## Scope / hardening note

E4 + E5 jointly discriminate roundoff-amplification from a real covariance defect (the weak metric
passes; the native fails; localized at the small node). The optional E1 multi-precision ladder
(binary64 / double-double / ≥50-digit) would additionally show the native residual scaling with
arithmetic ε — belt-and-suspenders, not required for the verdict. E3 (role-swap `FLAVOURS=(e,τ,μ)`)
needs a runtime rebuild of the reaction tables and is deferred as further hardening.
