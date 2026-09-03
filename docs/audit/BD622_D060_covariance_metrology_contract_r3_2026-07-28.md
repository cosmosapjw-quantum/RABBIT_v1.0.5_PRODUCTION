# BD622 D-060 — covariance-metrology contract, third issue (asymmetric r3)

Status: prospectively frozen BEFORE any family output. Claim
`C-F10-METROLOGY-R3`; prospective evidence `E-F10-D060-METROLOGY-R3`;
gate decided: `G-F10-COVARIANCE-METROLOGY` (flip deferred to the D-057
remedy step 5 single-writer reconsideration after steps 1--4 all pass).
Oracle: `scripts/audit/d060_covariance_metrology_r3_oracle.py` (frozen in
the same commit as this contract). Object: the protected comparator
`src/rabbit/decoupling/_independent_noqke.py`, full sha256
`760a7c044081e507fae9d5695b301bd44f6466d96322c46f53b77161e32b558a`
(guarded; never edited; mutants are in-memory monkeypatches only).

This is the corrective successor to D-054 (FAIL preserved) and D-055 r2
(reproducible, not gate-authoritative per D-057). It repairs exactly the two
D-057 evidence defects: (1) only P-fixed states were frozen; (2) the error
bound covered a reassociated one-map graph, not the production
`native(self_modal) + native(electron_modal)` two-map-plus-add diagnostic,
with no off-grid basis coverage and no rigorous interval.

## Environment guards (exit 30)

Module sha prefix `760a7c04`; `numpy == 2.4.4`; `mpmath == 1.3.0`
(owner-approved venv install 2026-07-28, pure Python); x86 80-bit
`longdouble` present; BLAS pins `OPENBLAS/OMP/MKL_NUM_THREADS=1` at import.
Execution: `PYTHONPATH=src venv/bin/python` with `--out`; the frozen family
run passes NO other flag.

## Frozen family

Grid `build_independent_grid(48, 24.0)`; config
`IndependentCollisionConfig()` (module defaults 12/12/4, electron radial
48) — byte-identical to D-055 for comparability. `T_cm = 10` MeV. State
generator identical to D-055:
`pair_logits_to_cloglog(stack([-nodes/s for s in scales]))`. `P` acts as
state rows `[0, 2, 1]`; species permutation `PI = (0,1,4,5,2,3)`; pair-row
permutation `sigma = (0,2,1)`.

| ID | scales (e, mu, tau) | T_gamma | block |
|---|---|---|---|
| SA-1 | (1.01, 0.997, 0.993) | 10.0 | audit pair (D-057 probe 1) |
| SA-2 | (1.05, 0.94, 1.03) | 9.7 | audit pair (probe 2) |
| SA-3 | (0.96, 1.04, 0.985) | 9.7 | audit pair (probe 3, worst native) |
| SB-1 | (0.90, 1.10, 0.95) | 10.0 | boundary: widest flavour asymmetry |
| SB-2 | (1.03, 0.94, 1.06) | 9.0 | boundary: largest gamma/nu split |
| SS-A / SS-D / SS-C | (1.01,.995,.995) / (1.0,.98,.98) / (1.02,.99,.99) | 10.0 | D-055 continuity (P-fixed) |

SA/SB members are evaluated at `f` AND `Pf` (10 production evaluations);
SS members once (3). Development hardening used ONLY the disclosed
non-family state `(1.005, 0.998, 0.996) @ T_gamma 9.9` via `--dev-state`;
no family member was executed before this freeze.

**M-ENV prospective envelope** (violation = exit 20, run void, no refit):
per evaluation `D_cov in [1e-23, 1e-17]` and `max|modal_total| in
[1e-20, 1e-15]`. Denominator floors: `F_weak = F_mass = 1e-21`,
`F_native = 1e-24`. `D_cov = max(sup|pt(f)|, sup|pt(Pf)|, F_native)` where
`pt` is the production `pair_total` recomputed exactly as module L1285.

## Checks (all frozen; any failure = FAIL preserved, no refit)

| ID | Predicate | Threshold |
|---|---|---|
| N1 weak | `sup‖modal_total(Pf)[PI] − modal_total(f)‖ / max-norm-floor` (all 6 species rows) | `<= 1e-10` |
| N2 mass | same form on `(w·y²)·total(g)` native rows | `<= 1e-10` |
| N3 native | `sup‖pt(Pf)[sigma] − pt(f)‖ / D_cov` | `<= 1e-10` (SA/SS); `<= 2.5e-10` (SB, owner-granted 2026-07-28: the native diagnostic's measured amplified-roundoff floor is ~5e-11 at the audit probes and the boundary states are unprobed; boundary equivariance stress is carried by N1/N2 at 1e-10) |
| N4 electron lemma | `sup‖modal_electron(Pf)[PI] − modal_electron(f)‖` relative over mu/tau rows (absolute recorded) | `<= 1e-12` |
| N5 entropy | `entropy_production > 0` at every evaluation | strict |
| N6 Higham | per sector s in {self, electron}: `G_s = sup‖modal_s(Pf)[PI] − modal_s(f)‖ <= 2·gamma(n_s)·S_abs_s` with `n_s`, `S_abs_s` from the paired absolute-value shadow assemblies (max over both runs) | must hold |
| N7 bitwise | inside the paired shadows: per-event rate arrays satisfy `rates_Pf[permP(e)] == rates_f[e]` bitwise at every node for the 23 leg-order-closed self events and all 15 electron events (`permP` asserted a bijection; the four `{mu,tau}` distinct-elastic events form the exchange class — the same-sign members partner with themselves, the opposite-sign members with each other — and are excluded here, carried by N7X), and the off-grid interpolated legs agree bitwise per flavour (max abs diff recorded — this is the off-grid basis coverage) | exact 0 |
| N7X exchange | the four `{mu,tau}` distinct-elastic self events are not leg-order P-closed; their equivariance runs through the exact tensor-quadrature exchange bijection `rate_Pf[e](alpha, beta, i12, i*, iphi) = rate_f[permP(e)](beta, alpha, i12, flip(i*), iphi)` — the same-sign members partner with themselves and the opposite-sign members with each other; the frame change is an about-y rotation composed with an x-reflection so the azimuth index maps to itself (rates are even in phi). Real-arithmetic identity from CM back-to-back symmetry and the symmetric leggauss rule; float64 agreement is fp-roundoff class (dev-state measured `~1.5e-12` sup). Checked on the frozen node subset `(4, 14, 24, 34, 43, 47)` over all 36 ordered node pairs; boundary support flips excluded and counted | `sup abs(delta) / max abs(rate) <= 1e-11`; mask mismatches `<= 16` |
| N8 anchors | at SS states, in the D-055 convention (`D_native = max(sup‖pt_mu‖, sup‖pt_tau‖, tiny)` over the mu/tau pair rows only): `D_anchor` and the module `mu_tau_residual` equal the D-055 recorded values EXACTLY (SS-A `6.261815481020115e-21` / `2.447560236182917e-11`; SS-D `1.0279153398476508e-20` / `1.5745492982469787e-11`; SS-C `1.261753327326722e-20` / `5.935964835407989e-12`) and `mu_tau_residual == numerator/D_anchor` with numerator `sup abs(pt_mu − pt_tau)` (bitwise identity for P-fixed states); plus signed-shadow ≡ production `modal_self_interaction`. Bitwise reproducibility provenance: the D-057 audit replayed the D-055 oracle byte-identically UNDER THIS VENV (`venv/bin/python`, numpy 2.4.4), so exact equality is evidenced, not assumed. | exact / `<= 1e-13` |
| N9 split-graph bound | `N3 <= B_cov = 4·[A_cond·(2γ(n_self)S_abs_self + 2γ(n_elec)S_abs_elec) + E_split]/D_cov` with `E_split = 2(B_map_self + B_map_elec) + 2E_add + 2E_pair`; `B_map_s = γ(50)·sup(|modal_s|·|B|ᵀ/scale)` per run (max), `E_add = eps·sup(|native_self|+|native_elec|)`, `E_pair = eps·sup 0.5(|total_2i|+|total_2i+1|)`, `A_cond = max_j ‖B_j‖₁/scale_j`. The bound covers the identical production two-map-plus-add graph; the shared float64 basis cancels between the two runs, so no separate basis-truth term enters the covariance bound. Frozen headroom factor 4. | must hold |
| M-INT-1 | mpmath.iv 192-bit outward replay of the production map stage (two matmul/divide maps + add + pair rows) from the realized float64 modal arrays at ALL evaluations: per-sector map-stage error `W_int <= 1e-8`; certified residual interval `[lo, hi]` of the N3 numerator satisfies `hi/D_cov <=` the block's N3 cap; float64 N3 lies in `[lo/D_cov − slack, hi/D_cov + slack]`, `slack = 4·ΣW_int + 16·eps` | as stated |
| M-INT-2 | mpmath.iv 128-bit full per-node replay of the self assembly at frozen node 24 for the SA-1 pair: independent interval kinematics/measure/kernels/off-grid Legendre recurrence/interpolants/Pauli; (a) containment of every captured float64 invariant, measure, kernel, and per-event rate within intervals padded by the frozen error model — relative term `2e-11` plus cancellation-aware absolute terms: dots `+512 eps * (leg-energy x total-energy)` (float64's own rounding is O(1) RELATIVE at cancelled dots and near-zero Pauli affinities while staying tiny on the physical scale), kernels propagate the dot pads, rates add `4 exp(log_loss) * [2 gamma(50)(sum abs(b_m c_m) over legs 3+4) + 64 eps (abs(u1)+abs(u2))]` for the float64 interpolant/Pauli error; (b) interval identity of the f and Pf rate replays under `permP` for the leg-order-closed events (exchange class carried by N7X); (c) reduced-mode contraction containment on frozen modes `(0,9,19,29,39,47)`, legs 3/4, with a Higham pad from the absolute float64 term sum. Float64 branch decisions are handled as: domain mask and roundoff clipping adopted from the captured production data; the Pauli branch recomputed from interval midpoints (both branch forms are the same real function, so a branch flip cannot break containment). | all three |

**Feasibility scope (declared pre-output):** a full-assembly interval replay
at production orders is `>= 1e8`--`1e9` interval operations per state —
infeasible in pure-Python mpmath. The frozen three-tier scope (M-INT-1 map
stage everywhere + M-INT-2 full deep-node replay at one frozen node of one
pair + N6/N7 discharging the remaining nodes via the Higham envelope over
bitwise-equivariant rates) is the contract's rigorous instantiation of the
D-057 interval requirement; N7's bitwise identity extends the deep-node
certification to every node because identical input bits through the
identical graph yield identical rates.

## Mutants (in-memory, preregistered kills, restoration canary)

| ID | Construction | Kill predicate |
|---|---|---|
| MUT-24 | historical 24-event one-orientation catalogue: drop the `(b,a)` orientation members, coefficient 32.0 | N1 `>= 1e-8` AND N3 `>= 1e-8` at SA-1 (audit measured 1.178e-6 / 6.849e-3) |
| MUT-HALF | 27 events, every pair-conversion coefficient 16.0 -> 32.0 (covariance-symmetric double count) | `pair_total(f; SA-1)` rel deviation from the base run `>= 1e-6` |
| MUT-SIGN | negate the self kernel matrix post-guard (wrapper on `_self_matrix`) | `pair_total(f; SA-1)` rel deviation `>= 1e-3` (entropy sign recorded) |
| MUT-LANE | replace each `(b,a)` orientation member by a lane-swap duplicate of the `(a,b)` member (label-only closure) | N1 `>= 1e-8` AND N3 `>= 1e-8` at SA-1 |
| MUT-GML | negate `_stable_pauli_gain_minus_loss` (global gain/loss flip) | `entropy_production <= 0` at SA-1(f) |

Any mutant SURVIVED = FAIL. Canary: after restoration, SA-1(f)
`pair_total` must reproduce the base run bitwise (mismatch = exit 20).

## Verdict and budget

PASS requires every check for every family member plus all mutant kills.
Exit 0 PASS / 10 FAIL (preserved; supersession only as a new owner
decision) / 20 mechanical (rerunnable; ERROR report retained) / 30
environment. Frozen wall budget 8 h (breach = exit 20). Single execution;
report JSON retained under the registered run with per-state terms,
interval results, mutant table, and wall time.

## Limitations declared pre-output

Single host/platform; same-model authorship and blind reviews; the interval
tiers certify the production graph on the frozen family, not a general-state
proof; absolute physical normalization and catalogue completeness remain
outside scope (unchanged from D-059's bounded claim registration).
