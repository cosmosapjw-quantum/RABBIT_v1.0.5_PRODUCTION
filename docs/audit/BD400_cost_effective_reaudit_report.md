# BD400 Cost-Effective Follow-Up Re-Audit — Report

**Target:** RABBIT augmented Type-I PSTF no-QKE BBN branch, after BD398→BD401.
**Commits under review:** `f4c8183` (BD399: staged prototype + bridge dQ harness), BD400 (standard phase-2 adapter + cost policy), BD401 (long-run payload-reuse A/B — evidence, not code).
**Date:** 2026-06-07
**Scope:** Hostile review of whether recent PRs moved real physics/numerical blockers or only added code surface. Not a public-production, publication-ready, endpoint-parity, or QKE claim. QKE out of scope.

---

## How this re-audit was conducted

- Read all required pre-reads (README, cost ledger, missing-info remediation, auditor requests, BD398 report, the cost policy, `augmented_staged_bbn.py`, the bridge harness, `nudec_tables.py`, the staged runner, the two new tests, the two key artifacts, the BD401 long-run results).
- Rebuilt a runnable tree by overlaying the 12 BD400 snapshot source files + PRIMAT JSON data onto the full BD397 source.
- Independently ran the staged test suite, ran the `standard_phase2` CLI directly, inspected the bridge q4/q8/q16 artifact, and traced the harness's "native" path into the code.
- Verified the adapter delegates bitwise to the in-tree PRIMAT RHS, and traced the bridge harness mismatch to its root cause.

Delivered artifact (alongside this report): `bridge_native_moment_harness_fix.py` — a corrected harness that extracts the native dQ from the in-solver `coupled_3T_rhs_from_collision_moments` path instead of an unnormalized standalone radial moment.

---

## 1. Executive verdict

**`ACCEPT_WITH_LIMITS`.**

BD399–BD401 are genuine, disciplined, honestly-labeled progress on the staging *infrastructure* and on the *missing-evidence* backlog — they implemented the staged prototype, wired a real (not toy) PRIMAT phase-2 RHS, added a lean cost policy, and ran the payload-reuse A/B I had flagged as missing, all without false-green and with raw failing states preserved (exit code 1 kept). But no physics blocker moved: the fused AP65 endpoint is untouched, the activation window still does not complete, and the one artifact meant to resolve the bridge calibration mechanism (the q4/q8/q16 bridge harness) turns out to measure the wrong quantity. This matches the project's own self-assessment (`ACCEPT_WITH_LIMITS`, blocker-movement 0.25), which is itself a good sign — no overclaiming.

---

## 2. Claim ledger

Labels: IMPLEMENTED / VALIDATED / PARTIAL / UNTESTED / FORBIDDEN.

| Claim | Label | Evidence (verified this audit) |
|---|---|---|
| Staged prototype executes a real in-tree 9-species PRIMAT phase-2 RHS | **VALIDATED** | CLI run: `n_species=9, n_reactions=12, network_success=True`, honest scope label |
| Adapter delegates to `abundance_rhs_phase2`, no duplicated physics | **VALIDATED** | `test_standard_phase2_network_adapter_matches_in_tree_rhs`: `assert_allclose(got, expected, rtol=0, atol=0)` — bitwise equal |
| Log-abundance network keeps abundances positive | **VALIDATED** | `np.all(result.Y > 0.0)` test passes; positivity structural in log coords |
| Parity comparator preserves misses (no clipping) | **VALIDATED** | flags `missing_staged` / `fail` / `nonfinite`; 5/7 staged tests pass on merged tree (2 "fails" are subprocess-cwd artifacts; script runs clean directly) |
| Cost policy deflated to a lean rule set | **VALIDATED** | 130 lines (from 500), enforceable PR cost-line template + blocker-movement metric |
| `standard_3t_plasma` closure matches the table exactly | **VALIDATED** | artifact `max_closed_effective_abs_excess_fraction = 0.0`; closed native excess ~1e-11 (re-confirms bridge:2251 uses `total_energy_transfer`) |
| Payload reuse cuts q4 wall and memory substantially | **VALIDATED** | BD401: wall 532→318 s (−40%), payload wall 285→72 s (−75%), payload builds 4940→1189, RSS 3.53→1.58 GB (−55%) |
| With reuse on, phase-2 is the dominant activation-window cost | **VALIDATED** | BD401: phase-2 wall 192.9 vs 192.5 s (unchanged) ⇒ 61% of the 318 s reuse-on wall |
| q4/q8/q16 artifact falsifies the q-grid excess hypothesis | **UNTESTED** (claim rejected) | harness "native" arm is an *unnormalized* radial moment: −1085% (q4), sign-flips to +1.05M% (q8), +1.25M% (q16) — not the in-solver dQ; wrong instrument |
| The +0.665% trajectory N_eff excess is explained | **UNTESTED** | neither confirmed nor falsified; requires the corrected in-solver-moment harness |
| Payload reuse is physics-neutral (safe to default-on) | **PARTIAL** | terminal Yp differs by +4e-6 (~3e-5 rel), D/H −6e-8, N_eff +1e-4; below observational precision but **not** bitwise → needs a documented tolerance, and the two runs stop at slightly different T_γ |
| Staged path is AP65 default behavior | **FORBIDDEN** | correctly not claimed; prototype-only |
| Fused AP65 endpoint moved | **FORBIDDEN** | correctly not claimed; both A/B runs exit 1 above endpoint |

---

## 3. Code-cost critique

BD399 cost 1034 net lines for harness/prototype; BD400 cost 293 net lines for the adapter + policy. The BD400 spend is efficient and well-targeted: the adapter is ~40 lines of thin delegation, the contract test proves it bitwise-equal to the in-tree RHS, and the comparator is a clean no-clip parity reporter. The policy at 130 lines is acceptable (down from a 500-line draft). The one cost concern is cumulative: BD399's 1034 lines bought a prototype that, by the project's own Q3/Q5, may be deleted after extracting one contract test if it does not lead to endpoint movement. That is the correct disposition — keep the prototype on a short leash.

Code-review answers: the adapter preserves `abundances_standard` physics exactly (bitwise test); weak rates `lambda_np`/`lambda_pn` and `eta` are threaded through `_evaluate_scalar_rate` with explicit positivity validation; `network_nfev=5365` over a 1e-9 e-fold span is a log-coordinate startup/conditioning artifact with tiny seeded species, **not** evidence of physical BBN-range stiffness — but it is an early warning that SciPy-Radau-in-log-space is unproven at activation→endpoint scale and should be benchmarked (not fixed) before the staged path is trusted; SciPy Radau is acceptable for this temporary harness but is not the in-tree Rodas5P host and must not become a default.

---

## 4. The bridge q4/q8/q16 artifact — harness mismatch (the key new finding)

The artifact reports raw "native" bridge excess of **−1085% (q4)**, **+1,046,575% (q8)**, **+1,249,447% (q16)**, with a **sign flip** between q4 (−4.52e-9) and q8 (+4.79e-6) while the table reference stays +4.58e-10. A convergent quadrature error of a physical dQ is sub-percent and never sign-flips by four orders of magnitude. Tracing the harness, "native" is wired to `build_augmented_pstf_radial_moment_thermo_source_from_geometry(..., energy_normalization="raw")` — an **unnormalized standalone radial moment**, not the physical net energy transfer the solver integrates. The `standard_3t_plasma` arm matches the table to ~1e-11, which only re-confirms (from `augmented_collision_bridge.py:2251`) that the closure renormalizes to `total_energy_transfer`; it says nothing about quadrature convergence of the physical dQ.

**Decision (Q6/Q7):** this is a **harness mismatch**, not a falsification of the q-grid hypothesis and not (yet) evidence of a convention bug. The BD398 q-grid hypothesis returns to **UNTESTED** — the instrument built to test it measured the wrong quantity. The real object to explain is the ~0.665% offset on T_ν/T_γ from BD397/BD398, which is a different quantity entirely. The corrected test (delivered) extracts the native dQ from `coupled_3T_rhs_from_collision_moments` fed by the bridge moments the solver actually consumes, at matched (T_γ, T_νe, T_νx), and compares the resulting dT_ν/dN against the table-driven RHS across q4/q8/q16. Its falsifier: if the in-solver bridge dT_ν/dN matches the table to <0.1% at all q, the excess is not a per-step dQ error and PR-2 must be re-scoped to accumulated-state or QED/EOS-convention causes.

---

## 5. BD401 payload-reuse A/B — interpretation and ranking

This is fresh long-run evidence (q4 bounded activation replay, terminal T_γ≈0.070 MeV ≈ N~2.75), not smoke. It cleanly isolates payload reuse: source evaluations are identical (4940 vs 4940), phase-2 wall is unchanged (192.9 vs 192.5 s), and the entire 214 s wall reduction is payload (285→72 s), with RSS halved (3.53→1.58 GB).

Two conclusions follow. First, payload reuse is a **large, cheap, one-time win** and should become default-on — *after* a documented physics-parity tolerance, because terminal observables are **not** bitwise identical (Yp +4e-6 ≈ 3e-5 relative, below the ~1e-3 observational precision on Yp but nonzero, and the two runs stop at slightly different T_γ). Second, and decisively for ranking: once reuse is on, **phase-2 becomes 61% of the activation-window wall**. This is exactly the BD397/BD398 thesis — payload reuse is worth banking, but it does not touch the activation/endpoint blocker, which is phase-2-dominated and needs the staged network. So the ablation strengthens the case for *both* the cheap payload default-on *and* the phase-2 staging work, while confirming neither moves the endpoint by itself.

---

## 6. Missing dynamic evidence

The deliberately-missing items are acceptable for a follow-up *audit* packet but are not optional before specific decisions:

1. **Fresh cold endpoint with full component telemetry** — blocks any endpoint-parity or default-on optimization decision; still the stale BD346 zero-attribution artifact.
2. **A completed activation-window run** — still none; the activation window remains the unproven-to-complete region.
3. **High-q full-solver N_eff convergence via the *corrected* in-solver-moment path** — blocks the bridge-mechanism conclusion (the included q4/q8/q16 artifact does not serve this purpose).

The BD401 same-HEAD A/B is sufficient to rank payload reuse against phase-2 *for the q4 activation replay* (it does), but endpoint/cold long runs remain mandatory before any default-on decision or endpoint claim.

---

## 7. Required next commands

```bash
# (a) Corrected bridge native-moment test (resolves the q-grid question properly)
PYTHONPATH=src JAX_PLATFORMS=cpu python scripts/bridge_native_moment_harness_fix.py
#     Falsifier: in-solver bridge dT_nu/dN within 0.1% of table at all q ⇒ excess is not per-step dQ.

# (b) Staged-vs-fused parity on a real cold endpoint case (the endpoint stepping stone)
PYTHONPATH=src JAX_PLATFORMS=cpu python -m pytest -q tests/test_staged_vs_fused_parity.py
#     Falsifier: any of Yp / D-over-H / Sigma_H / asymptotic N_eff diverges beyond documented tol.

# (c) Payload-reuse parity sign-off before default-on
PYTHONPATH=src JAX_PLATFORMS=cpu python scripts/run_augmented_continuous_ap65_full_bbn_span_ladder.py ... \
  --stage-collision-payload-policy thermo_state_tolerance_reuse   # vs current_state, SAME stop point
#     Falsifier: terminal Yp relative delta > 1e-4 at a matched terminal T_gamma.
```

---

## 8. Top three PR recommendations

| PR | Scope | ≤5-PR or breakthrough | Files | Success metric | Falsifier |
|---|---|---|---|---|---|
| **PR-A** AP65↔staged handoff / endpoint initialization threading | Feed a real AP65 background segment into the staged network and run to a deeper endpoint than the activation cutoff; this is the first step that can actually move the endpoint blocker | **Breakthrough** (prototype step ≤1 PR) | `augmented_staged_bbn.py`, `augmented_continuous_ap65_full_bbn_span_ladder.py` (read), new handoff module | Staged network completes a span the fused path cannot within 420 s, with observables within tol of the fused segment at the handoff point | Staged handoff diverges from fused at the handoff T_γ beyond tol, or fails to complete |
| **PR-B** Corrected in-solver bridge native-moment harness + high-q re-run | Replace the unnormalized-radial-moment "native" arm with the `coupled_3T_rhs_from_collision_moments` path; re-run q4/q8/q16 | **≤5 PR** | `scripts/bridge_native_moment_harness_fix.py` (delivered), wire to replay moment path | Native dT_ν/dN vs table converges (or is shown q-independent) at <0.1% | Excess stays large/sign-flipping ⇒ genuine convention bug, escalate |
| **PR-C** Payload-reuse parity sign-off → default-on | Document a parity tolerance, prove reuse within it at a matched stop point, flip default | **≤5 PR** | span ladder policy default, a parity test | Reuse within documented tol at matched terminal T_γ; default flipped; banked −40% wall / −55% RSS | Reuse exceeds tol at matched stop ⇒ keep opt-in |

Log-abundance coordinates already live inside the staged network (good). Network solver ownership: keep SciPy Radau only as the prototype harness; the `network_nfev=5365` signal means a Rodas5P/diffrax comparison is warranted *inside PR-A*, not as separate policy work.

---

## 9. Overengineering / deflation actions

Unchanged structural debt from BD397/BD398, still untouched (correctly out of scope this round): freeze/move the ~12k-LOC `augmented_publication_*`/`*readiness*` suite out of the active path; delete count-lock tests (`plot_count==5`, etc.); split the 25.8k-line `augmented_continuous_ap65_rhs.py` into `background/`/`network/`/`phase2/`/`telemetry/` — with the staged rewrite serving as the vehicle for the background/network split. New this round: per the project's own Q3/Q5, if PR-A does not produce endpoint handoff movement within ~3 PRs, **delete the staged prototype after extracting the one contract test and the parity comparator** — do not let it become a parallel local minimum. The cost policy's one enforceable metric to retain is: every PR must declare a measured blocker it moved and a falsification command, or it is drift.

---

## 10. Red-team objections to my own recommendation

- **"You accept the round but admit nothing moved the endpoint — that rewards motion without progress."** The acceptance is explicitly *with limits* and scores the same low blocker-movement as the project does. What is being rewarded is the honesty and the clearing of the evidence backlog (payload A/B, PRIMAT adapter, near-FLRW regression), which are prerequisites to endpoint work — not the (absent) endpoint progress itself.
- **"Calling the bridge artifact a harness mismatch lets the team off the hook for a possible real bug."** It does the opposite: it says the question is still open (UNTESTED) and hands them the corrected instrument plus a falsifier that *will* expose a convention bug if one exists. The unnormalized-radial-moment evidence (sign flip, 4-orders-of-magnitude swing) is conclusive that the *current* artifact cannot answer the question.
- **"I previously pushed the q-grid hypothesis; am I now rationalizing to protect it?"** No — I am demoting my own prior claim from DERIVED to UNTESTED because the test came back uninterpretable through no fault of the hypothesis. The hypothesis gains no support here; it simply remains unfalsified by a wrong instrument.
- **"Payload reuse halves wall and memory — surely that's the win to chase, not phase-2."** It is a win to *bank* (PR-C), but the same A/B shows phase-2 is 61% of the activation wall once reuse is on, and reuse leaves the endpoint untouched (exit 1). Chasing further payload work after PR-C would be optimizing a now-minority bucket.
- **"You verified on a merged tree and 2 CLI tests failed."** The 12 modified files are byte-identical to the snapshot; the 2 failures are subprocess-cwd artifacts (the CLI ran clean when invoked directly, reproducing `n_species=9, n_reactions=12, network_nfev=5365`). The substantive contract test (bitwise adapter equality) passed.

---

## References (external comparison anchors)

- LINX — arXiv:2408.14538 (staged background→network architecture; the structure PR-A moves toward).
- PRIMAT — arXiv:1909.12046 (the nuclear-network/reaction-rate backbone the adapter delegates to; `primat_ac2024_12rxn.json`).
- PRyMordial — PMC11266446; AlterBBN v2 — arXiv:1806.11095; PArthENoPE — arXiv:0705.0290.
- NUDEC_BSM — arXiv:2001.04466 (momentum-averaged neutrino-decoupling; `nudec_tables` calibration source).
- Precision N_eff: arXiv:2012.02726 (N_eff^SM = 3.0440 ± 0.0002); review arXiv:2301.12299 (no-oscillation Boltzmann transport → ΔN_eff ≈ 0.034, the calibrated no-QKE target).
