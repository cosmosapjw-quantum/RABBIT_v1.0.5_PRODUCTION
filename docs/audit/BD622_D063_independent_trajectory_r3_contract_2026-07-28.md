# BD622 D-063 — independent trajectory r3 contract (two-change reissue)

Status: prospectively frozen BEFORE any output. Claim `C-F10-TRAJECTORY-R2`;
prospective evidence `E-F10-D061-TRAJECTORY-R2`; gate decided:
`G-F10-INDEPENDENT-FLRW` (flip deferred to the D-057 remedy step 5
single-writer reconsideration after steps 1--4 all pass). Driver:
`scripts/audit/d063_independent_trajectory_r3.py` (frozen in the same
commit). D-056 and its artifacts are preserved unchanged as supportive
evidence (`E-F10-D056-INDEPENDENT-TRAJECTORY`) and are never relabelled.

This is the corrective successor to the preserved D-056 run, repairing the
four D-057 defects: obsolete F10C1 anchors; no cross-code block/spectral
predicate; T6 as event-level bookkeeping; no refinement/tail budget.

## r3 supersession delta (adjudicated, D-062)

Exactly-two-change derivative of the preserved D-062 r2 FAIL, permitted by
the registered D-062 adjudication under the D-054/D-055 and D-060/D-061
precedent. The r2 execution failed T8 and T10 on frozen-parameter design
defects while every physics observable passed (endpoint bitwise-replaying
D-056; scalar/block/split/coupled-energy checks in band):

1. **T8 support gating.** The r2 sup (`2.0801e-3` vs the `2e-3` band, a 4%
   overage) occurred at the single Rust node `y = 0.0018` — below the first
   affine GL48 collocation node (`y ~= 0.01475`), where the degree-47
   interpolant is an extrapolation outside its support and the `y^2` measure
   suppresses any physical weight (all integrated observables agree at the
   `1e-5` class). r3 gates the sup on Rust nodes with `y >= GRID.nodes[0]`
   (46 of 48 nodes); the two sub-support deviations are recorded non-gating
   (`spectrum_sup_rel_full` retained). Within support the realized r2 sup was
   `1.59e-3`, in band. The band itself is unchanged.
2. **T10 mutant-window relocation.** The r2 frozen restart `N = 4.00` sits
   where the measured base heating activity (`3.8e-4` per e-fold) is below
   the `5e-4` activity floor, so both sign mutants terminated
   inconclusive-SURVIVED without the flipped physics ever being probed —
   structurally unreachable kills. r3 restarts at the measured activity peak
   `N = 2.00` (`2.4e-3`, 4.8x above the floor): span `[2.00, 2.20]`, FD
   samples `{2.05, 2.10, 2.15}`. The activity floor, kill level `1.0`, and
   residual formulas are unchanged; the structural mutant value `~2.0` and
   its margin analysis carry over to the new window.

No band, anchor, threshold, checkpoint set, or formula changes beyond these
two items (mechanical diff). The reissue re-executes the full frozen
sequence (base -> mutants -> holdout) from scratch; the r2 base's bitwise
replay of D-056 (3694 evaluations, identical endpoint digits) evidences the
determinism of the re-execution.

## Environment guards (exit 30)

Full module sha256
`760a7c044081e507fae9d5695b301bd44f6466d96322c46f53b77161e32b558a`;
embedded 48-tuple Rust spectrum constants self-hash; both anchor logs
present with frozen sha256 (`15d30ef3...` D-049, `0c3ea522...` D-045).
BLAS pins at import. `PYTHONPATH=src venv/bin/python`, `--out` only.

## Frozen anchors (completed-catalogue F10C2, current tree)

Rust BDF: `N = 7.93669333948508360`, `t = 5.26776344870795510e4 s`,
`N_eff = 3.03403598358439952` (source constants
`native/rabbit_cpu/src/isotropic_boltzmann.rs:1232-1233`; `t` from the
retained log readout; steps 1459/1). Rust Rodas5P partner:
`N = 7.93670604025685922`, `t = 5.26780987530529019e4 s`,
`N_eff = 3.03390437892078513` (steps 253/7). Both bitwise identical in the
two retained D-045/D-049 logs.

**Historical divergence recorded:** the 2026-07-16 rows of
`docs/harness/VALIDATION_LEDGER.md` and `docs/harness/CLAIM_LEDGER.md`
(F10C2 endpoint entries) carry the pre-optimization Rodas5P endpoint
`(7.936706017467941, 52678.09666722039, 3.033904967773792)` (steps 271/29)
and cite `run-20260716-f10c2-perf`, which no longer exists on disk. r2
freezes the current-tree retained-log values above; the BDF endpoint is
bitwise unchanged since 2026-07-16.

Block anchors (`F10C2_BDF_BLOCKS`): `rho_nu = 2.83361184184362380e-10`,
`n_e = 8.38689035133557397e-9`, `rho_e = 9.47981915653400480e-11`,
`n_x = 8.35845135010454098e-9`, `rho_x = 9.42814963095111658e-11`; the
flavour split ratio `rho_e/rho_x - 1 = 5.480346e-3` is the T-drift-free
catalogue-structure discriminator. `Qe_*` values are recorded non-gating
(no cross-code evidence base for a band). Endpoint spectrum: all 48
`F10C2_BDF_SPECTRUM` tuples `(y, w, f_e, f_x)` embedded with a self-hash.

## Frozen run configuration

Identical to the D-056 base: GL48-Y24, config `(4, 4, 4, 24)`, BDF
`rtol 1e-6`, `atol 1e-9`, start 10 MeV equilibrium, terminal
`T_gamma = 0.005 MeV`. New: `dense_output=True`; checkpoints
`N_k = 0.25k, k = 1..31`; FD half-step `H = 0.02`; transfer window
`N in {3.00, 3.25, ..., 5.50}`; mutant restart `N0 = 2.00`, span
`[2.00, 2.20]`, FD samples `{2.05, 2.10, 2.15}` (r3 change 2); holdout `rtol 3e-7`
(axis chosen over angular 6/6: 2.25x per-eval cost and an operator
confound of the recorded `~4.45e-4`-class quadrature discrepancy).
Frozen wall budget 10 h (breach = exit 20). Report flushed after every
phase with verdict `IN_PROGRESS` so a crash preserves prior evidence.

## Coupled-energy residual (T9, replaces the vacuous T6)

At dense-output states only (no event bookkeeping): comoving energies
`W_nu = e^{4N} rho_nu`, `W_tot = e^{4N}(rho_nu + rho_EM)` with `rho_nu`
spectrally reduced by `independent_thermodynamics` and `rho_EM/P_EM` from
the adaptive EOS. Exact physics: `dW_tot/dN = e^{4N}(rho_EM - 3 P_EM)`
(the +-Q/H exchange cancels between sectors; only the electron-mass trace
survives). Central differences with `H = 0.02`:

```text
R_total(N_k)    = |D[W_tot] - e^{4N}(rho_EM - 3P_EM)| / (e^{4N}(rho_tot + P_tot))
R_transfer(N_k) = same numerator / |D[W_nu]|,  active iff |D[W_nu]| >= 5e-4 W_nu
```

A neutrino or EM source-sign flip makes the two sources add instead of
cancel: the numerator becomes `~2|D[W_nu]|`, so `R_transfer ~= 2`
structurally, independent of `|Q|` — the O(1) kill the D-057 row demands.

## Checks (frozen; any scientific miss = FAIL preserved, no refit)

| ID | Predicate | Band / margin basis |
|---|---|---|
| T1 | terminal event reached, occupations strict-open, no fail-closed error | — |
| T2 | `abs(N_eff - 3.03403598358439952) <= 3e-4` | D-056 measured `+1.83e-5` vs this anchor (16x); REJECTS the obsolete-anchor class `7.44e-4` |
| T3 | `abs(N - 7.93669333948508360) <= 1e-4` | measured `+5.5e-6` (18x); non-discriminating vs the obsolete class (honestly recorded) — N_eff carries scalar discrimination |
| T4 | `abs(t - 52677.6344870795510) <= 5 s` | measured `+1.10 s` (4.6x) |
| T5 | Rodas5P partner: `abs(dNeff) <= 3.5e-4`, `abs(dN) <= 1e-4`, `abs(dt) <= 5 s` | measured `1.50e-4` (2.3x, thinnest scalar band — consistency check, documented) |
| T6 | `e_enh > mu_enh > 0` and heavy relative gap `<= 1e-6` | D-056: `4.7e-10` |
| T7 | block densities `rho_nu, n_e, rho_e, n_x, rho_x` within rel `1e-3` of `F10C2_BDF_BLOCKS` (independent side: pair densities `2 T^{3,4}/(2 pi^2) sum w y^{2,3} f`); split ratio `abs(delta) <= 2e-4` | T-drift imprint `~2.2e-5` + physics `~3e-5` (~30x); split measured delta `1.80e-5` (11x), T-drift cancels |
| T8 | mapped spectrum `f_ind = FD(GRID.interpolate(logit, y_rust))` vs `f_e/f_x`: `sup abs(delta_f)/max(f_rust, 1e-8) <= 2e-3` over the 46 in-support Rust nodes (`y >= GRID.nodes[0]`) x 3 flavours; the two sub-support nodes recorded non-gating (r3 change 1) | expected `~1e-4` class (20x); the 24-event-class imprint `~1.8e-3` is borderline at T8 alone — layered rejection is carried by T2 (anchor class), T7-split (flavour structure), T10 (signs); stated honestly |
| T9 | `R_total <= 5e-4` at all in-range checkpoints AND `>= 4` CLEAN window checkpoints, where clean = active (`abs(D[W_nu]) >= 5e-4 W_nu`) AND `R_transfer <= 0.3`. Count-based by design: a marginally-active checkpoint (activity in `[5e-4, ~1.1e-3]`) can carry floor-level FD noise up to `R_transfer ~0.7` on a correct trajectory, so the max over active rows is recorded but not gated; the mutant kill level 1.0 (T10) is unaffected since mutant residuals are structural `~2.0` at strongly-active samples | FD noise floor `~1.1e-4` on `R_total` (4.5x); `~7-8` of 11 window checkpoints expected active, most well above the marginal band |
| T10 | mutants M1 (`s_pair = -1`) and M2 (`s_qem = -1`) restarted from the measured-activity-peak `N = 2.00` checkpoint (r3 change 2): KILL = fail-closed error OR solver failure OR `max R_transfer >= 1.0` over the mutant's active FD samples; no active sample = SURVIVED (fail-closed, recorded as activity-floor inconclusive) | structural mutant value exactly `~2.0` for both signs (numerator `2 q_nu/H`-class vs denominator `q_nu e^{4N}/H`), 2x over kill |
| T11 | last-NODE quadrature-weight energy fraction `<= 5e-7` (the gated proxy for tail growth; its equilibrium value is `~1.4e-8` in the 3-flavour-numerator form, so the bound fires only on `~35x` tail amplification; the beyond-domain truncation itself is separately computed as `1.04e-7` of the equilibrium energy and is declared context, not gated); per-eval `whole_reaction_domain_rejections in [8.5e5, 1.05e6]` (probe envelope 920088--980472, ~7% headroom; counts include Jacobian/rejected-trial rhs calls); `largest_matrix_roundoff_correction <= 1e-10` (observed 0.0) | analytic + probe-backed |
| T12 | holdout `rtol 3e-7` full run reaches the event; drift `abs(dNeff) <= 2e-4`, `abs(dN) <= 5e-5`, `abs(dt) <= 3 s` vs base | expected `2-5e-5` in `N_eff` (integrator self-refinement, well under the cross-method `1.32e-4` spread) |

Decision rule: PASS requires T1--T12. Retained evidence: endpoint spectra
(GL nodes + mapped-at-Rust-nodes), full 31-row checkpoint/residual tables
for base and holdout, mutant residuals and modes, tail/rejection extrema
and counts, wall time. Exit codes 0/10/20/30; single execution of the
frozen sequence (base -> mutants -> holdout).

## Runtime plan

Base `~2:45-3:00 h` (D-056 measured 3694 evals x `~2.7 s`); mutants
`~10-20 min` each (BDF Jacobian-dominated on the 146-dim system); holdout
`~3.5-5 h` (`rtol` 1e-6 -> 3e-7: step count x `~1.3-1.5`). Total
`~6.5-8.5 h` within the frozen 10 h envelope; progress prints every 50
evaluations.

## Limitations declared pre-output

Matched-resolution N48-class comparison on the frozen 10 MeV -> 5 keV cell
only; single platform; same-model authorship/reviews; T3/T4 scalar bands do
not discriminate the obsolete-anchor class on their own (T2 does); T8 alone
does not certify rejection of the 24-event class (layered with T2/T7/T10);
no continuum, production, unblinding, F-11, or QKE claim.
