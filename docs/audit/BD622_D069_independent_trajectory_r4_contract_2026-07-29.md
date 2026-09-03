# BD622 D-069 — independent trajectory r4 contract (frozen before output)

Date: 2026-07-29
Claim: `C-F10-INDEPENDENT` · prospective evidence key: `E-F10-D069-TRAJECTORY-R4`
Driver: `scripts/audit/d069_independent_trajectory_r4.py`
Shared machinery: `scripts/audit/_trajectory_core.py` (D-068)
Anchors: `scripts/audit/_f10c2_anchors.py`
Frozen module under test: `src/rabbit/decoupling/_independent_noqke.py`,
SHA-256 `760a7c044081e507fae9d5695b301bd44f6466d96322c46f53b77161e32b558a`
(never edited; `order` and `y_max` are public parameters of
`build_independent_grid`, and the r4 companion run uses only those)

**This contract and driver are committed before the driver produces its first
output byte.** Any reissue requires a registered adjudication authored strictly
after a complete report exists, and may change only the items that adjudication
names. No band, node mask, or tail proxy may be fitted after output.

## 1. What this must discharge

D-065 finding F-D065-04 restored `G-F10-INDEPENDENT-FLRW` to FAIL. It did not
dispute the D-063 payload — an independent audit reproduced all 48 embedded
spectrum tuples exactly, reconstructed the residuals to 2.71e-20, and confirmed
all 12 coded predicates. It disputed the *evidence*, on five counts:

1. the frozen T8 norm was narrowed after output (full-domain 2.0800641747316935e-3
   against a 2e-3 band; masked 1.5881626368027338e-3);
2. checkpoints retained nine reduced scalars, not the state needed to recompute;
3. `tail_last_node_fraction` is an in-domain last-GL-node contribution, not a
   `y > 24` enclosure;
4. the only holdout axis was `rtol`;
5. the adjudication chronology was impossible.

Each is addressed below with a prospective predicate.

## 2. Why the full-domain pointwise sup is not the right gate

The two disputed Rust nodes sit at y = 0.0018440557621953763 and
0.009720472536706333, below the independent grid's first affine GL48 collocation
node 0.014747912970887178. There the degree-47 interpolant is an
*extrapolation*: it is inside the declared domain `[0, 24]` but outside the data.
Bringing y = 0.00184 inside the collocation support requires order ≈ 136, since
the first Gauss–Legendre node scales as `y_max / order²`. That is far outside
this cost class, and the density-matched companion run at order 60 / y_max 30
still has its first node at 0.0118482.

So the choice is not "mask or don't mask" — it is "gate a quantity that is
measurable, or gate one that no feasible N48-class grid can measure." r4 gates
measurable quantities and records the unmeasurable one.

The measured weight of those two nodes settles what is at stake. Two numbers,
because they are weighted by different spectra and both appear in the evidence:

- **8.894740214084245e-10** of the Rust energy measure on the disclosed
  *equilibrium* spectrum — the value derived in §4 and quoted for the bands;
- **8.812e-10** on the *anchor* spectrum `f_e`, which is what the driver records
  per node as `energy_weight_share`.

Either way, a 100 % pointwise error at both nodes moves the energy-weighted norm
by under 1e-9, and they carry 2.9e-7 of the number measure.

## 3. The Rust anchor rule

Established before freezing, from the embedded table alone: the anchor node set
is **not** Gauss–Legendre on `[0, 24]`. It is a 48-node half-line rule with
maximum node 22.18412259016373 and weight sum 26.752783050383314, and it
reproduces the exact Fermi–Dirac moments over `[0, ∞)`:

| moment | rule value vs exact | interpretation |
|---|---:|---|
| `∫ y² f_eq` | −5.401621150369351e-09 | rule integrates the whole half-line |
| `∫ y³ f_eq` | −5.180810813687486e-08 | its own error dominates any comparison |

Consequently a cross-code moment comparison is limited by the *Rust* rule's
quadrature error, not by the independent grid.

## 4. Band derivation (pre-freeze, disclosed non-anchor state)

Computed on the disclosed equilibrium state `f = 1/(1+e^y)` — no trajectory, no
anchor spectrum, no endpoint. Regenerate with
`scripts/audit/d069_band_derivation.py`; the committed output is
`scripts/audit/d069_band_derivation.json`:

| k | cross-rule floor after tail correction, order 48 / y_max 24 | order 60 / y_max 30 |
|---|---:|---:|
| 2 | +5.402e-09 | +5.402e-09 |
| 3 | +5.181e-08 | +5.181e-08 |
| 4 | +3.188e-07 | +3.188e-07 |
| 5 | +1.375e-06 | +1.375e-06 |

The two domains agree to seven significant figures after the analytic tail
correction (5.40163e-09 at both, and likewise for k = 3, 4, 5). That is an
independent pre-freeze validation of the correction procedure used by T8a and T13: the correction absorbs the domain difference
exactly, so what remains is the Rust rule's own error.

Bands are therefore **inherited, not fitted**:

- T8a uses `1e-3` for **every** k, the tolerance already frozen at T7 for
  cross-code densities — the moments *are* densities. No looser tier for the
  tail-weighted k = 4, 5.
- T8a split ratio uses `2e-4`, the value already frozen at T7.
- T8b uses `1e-3`, the same cross-code density tolerance, on a strictly harder
  quantity (pointwise deviations cannot cancel).
- T8c keeps `2e-3`, the r3 predicate, unchanged.
- T13 uses `2e-4` for the measured domain shift (the holdout band class already
  frozen at T12) and requires the total enclosure to stay under `3e-4`, the T2
  band — a domain uncertainty larger than the discriminating band would make the
  cross-code comparison meaningless.

**Margin against the deviation actually expected, not against the floor.**
Quoting headroom over the quadrature floor would overstate the case by orders of
magnitude, so the operative margins are reconstructed here from the *preserved*
d063 endpoint spectrum — already-published evidence, not new output:

| predicate | expected value | band | margin |
|---|---:|---:|---:|
| T8a max abs rel over k, flavours | 6.1e-5 | 1e-3 | ~16x |
| T8a split ratio | 1.8e-5 | 2e-4 | ~11x |
| T8b weighted, all 48 nodes | 2.1e-4 | 1e-3 | ~4.8x |
| T8c in-support sup | 1.588e-3 | 2e-3 | ~1.26x |

T8c is the thin one, inherited unchanged from r3 and labelled a floor. No band
here is the minimal passing choice for its predicate.

## 5. Frozen anchors

Completed-catalogue current-tree endpoints, both retained release logs bitwise
(`15d30ef3…`, `0c3ea522…`):

| quantity | BDF | Rodas5P |
|---|---:|---:|
| `N_end` | 7.93669333948508360 | 7.93670604025685922 |
| `t_end` (s) | 5.26776344870795510e4 | 5.26780987530529019e4 |
| `N_eff` | 3.03403598358439952 | 3.03390437892078513 |

Block anchors `rho_nu`, `n_e`, `rho_e`, `n_x`, `rho_x` and the flavour split
ratio 5.480346367569888e-3 are unchanged. The 48-tuple spectrum table is
single-sourced in `_f10c2_anchors.py` and self-hashed to
`3df5a9907f8a7e7e5168a4659504387f16a0c795878a833df97f7a7e2613fa58`, verified
equal to the frozen d063 constant.

## 6. Checks

`T1`–`T7`, `T9`, `T10`, `T12` carry the r3 semantics unchanged. `T11` keeps the
r3 predicate **in full** — `tail_last_node_fraction <= 5e-7` together with the
per-evaluation rejection band and the roundoff cap — and is renamed
`T11_tail_and_rejections` only to say what it gates. r4 adds T13 rather than
replacing that gate: dropping a predicate while claiming nothing changed would
be the very failure D-065 punished. New or replaced:

**T8a — moment norm (gating).** `M_k = Σ w y^k f` for k = 2, 3, 4, 5 per flavour,
computed on each code's own nodes and weights, no interpolation in either
direction. The independent side is divided by `(1 − tail_k)` where `tail_k` is
the exact analytic equilibrium fraction beyond its own `y_max`. Also gates the
flavour split ratio built from the k = 3 moments against the Rust rule's own.

**T8b — true-weight pointwise norm (gating).**
`Σ_i w_i y_i³ |f_ind(y_i) − f_rust(y_i)| / Σ_i w_i y_i³ f_rust(y_i)` over **all
48** Rust nodes, the two sub-support nodes included at their real weight.
Nothing is excluded.

**T8c — in-support sup (gating).** The r3 predicate over the 46 in-support
nodes, retained as a floor and labelled as such.

**Recorded, non-gating:** the full-domain sup, the complete 48 × 3 per-node
deviation array, the mapped values at all 48 nodes, and an explicit list of the
sub-support nodes with y, w, and energy-weight share.

**Pre-committed here, before output:** that recorded full-domain sup is expected
to land at approximately **2.080e-3**, above the 2e-3 band D-062 gated it
against. It is stated now, in the frozen contract, precisely so that it cannot
later look like an excuse invented to explain an inconvenient number. It is not
gated because it is dominated by degree-47 extrapolation at y = 0.00184, a
quantity carrying 8.9e-10 of the energy measure that no feasible N48-class grid
can measure — see §2. If it instead lands far from 2.080e-3, that is itself a
finding and must be reported as one.

Honest statement of what T8b does and does not achieve: it is full-domain in
*form*, and the two disputed nodes move it by roughly 4e-6 relative. T8b is not
a strong test of those nodes. Nothing gates them, and the contract does not
pretend otherwise; what changed versus r3 is that they are no longer silently
dropped from a norm after the number was seen.

**T13 — evolved beyond-domain enclosure and domain holdout (gating).** A
density-matched companion integration at `build_independent_grid(60, 30.0)`
(node density 2.0 per unit y, identical to 48/24), same collision config, same
terminal condition. Gates: the companion reaches the terminal condition;
`|ΔN_eff| ≤ 2e-4`; the analytic equilibrium energy fraction beyond y = 30
is `≤ 1e-6`; and `|ΔN_eff| + N_eff × residual ≤ 3e-4`. Both runs' whole-reaction
domain-rejection counters are recorded. Three honest caveats:

- order 60 / y_max 30 also refines the collocation spacing (first node
  0.0118482 vs 0.0147479), so the measured shift is an **upper** bound on domain
  truncation alone;
- the analytic clause is a computed constant (4.921721142238624e-10 against a
  1e-6 band) and the total clause is implied by the measured one, so T13's
  operative content is `domain run reaches terminal` **and**
  `|ΔN_eff| ≤ 2e-4`. The constant is retained because it bounds what the
  measured shift cannot see — content beyond y = 30 — not because it discriminates;
- whether the N48 representation is converged tightly enough for the order-60
  companion to land inside 2e-4 is genuinely unknown before the run. A miss is
  real information about the representation and stands as a preserved FAIL.

**T10 caveat.** A mutant counts as killed on any solver failure or exception, not
only on the coupled-energy residual; r3's M1 died with "Required step size is
less than spacing between numbers." The kill *mode* and `max_R_transfer` are
surfaced in the check output so an adjudicator can see which kind of kill it was
without opening the raw report.

**T14 — recomputable checkpoints (gating).** Every checkpoint retains the full
integrator state: all `3 × order` cloglog components, `T_gamma`, `T_cm`, and
cosmic time. At N = 1.00, 4.00, 7.00 the driver re-derives `W_nu` and `W_tot`
from the stored state alone and requires agreement to `1e-12` relative against
the streamed values. This is the predicate that makes independent recomputation
from the report a fact rather than a claim.

## 7. Execution and chronology

Phase order — base, T14 re-derivation, mutants, **domain holdout**, rtol holdout
— front-loads the new obligation, so a wall-budget breach still leaves the
D-065-critical evidence flushed. The report is written once per phase to the
`--out` path and never duplicated into stdout; stdout carries progress lines
only.

Projected wall time from measured per-evaluation cost on this host (2.807 s at
48/24, 4.754 s at 60/30) and from the *measured* r3 mutant-battery duration in
the retained d063 log (1.58 h, not the 0.4 h a naive per-evaluation estimate
gives): base ≈ 2.9 h, mutants ≈ 1.6 h, domain ≈ 5.2 h, rtol ≈ 3.4 h,
total ≈ 13.1 h. Frozen budget **18 h**; breach is a mechanical exit 20, never a
scientific verdict. Note that a breach yields no `checks` block at all — phase
front-loading preserves the evidence, not the verdict.

BLAS threading is pinned to one thread in the driver *before* `import numpy`,
because OpenBLAS reads those variables at library initialisation; pinning after
the import is a no-op and would silently produce a multi-threaded, non-bitwise
run. `--out` is required, so a mis-invocation cannot discard a 13-hour report
while still exiting 0.

`started_at`, `completed_at`, and `wall_seconds` are machine UTC stamps from the
report writer. The adjudication envelope must carry timestamps captured with
`date -u` at real moments and must satisfy
`freeze commit < report.started_at < report.completed_at < adjudication.started_at
< adjudication.completed_at < containing commit`.

Exit codes: 0 PASS, 10 FAIL (preserved, no refit), 20 mechanical, 30 environment.

## 8. What a PASS would and would not establish

It would establish that a structurally independent full-spectral FLRW solve
reproduces the completed-catalogue endpoint, blocks, and spectrum on the frozen
10 MeV → 5 keV cell at N48 resolution class, with the beyond-domain contribution
enclosed, the representation axis held out, and every reported number
recomputable from the retained state.

It would **not** establish a continuum result, absolute normalization,
cross-platform reproducibility, agreement outside the frozen cell, or anything
about QKE, Bianchi, or production use. `G-F10-SCOPE` prohibitions stand.
