# D-071 — Seal the trajectory lane as FAIL on current measurement

**Date:** 2026-08-04
**Decision:** `CLOSE_AS_FAIL_ON_CURRENT_MEASUREMENT` for `G-F10-INDEPENDENT-FLRW`.
**Gate movement:** none. The gate was FAIL and stays FAIL. This changes its *disposition* — from
"remediation in progress" to "closed on the current instrument and current measurement" — which is a
record change, not a grade.

This decision also **withdraws the D-071 robustness envelope** that earlier surfaces forward-reference.

---

## 1. What this decision withdraws, and why it must be D-071 rather than a new number

`check_ssot_consistency.py` reports `forward_references: ["D-071"]`: the record already promises a
D-071, and that promise is the nine-axis robustness envelope (R1–R9: angular and radial holdouts, a
density ladder at orders 60/72/96, a second-venv reproducibility axis, and a bitwise replication axis).

That plan rested on one assumption — that an order-60 integration is tractable within the frozen wall
budget. **The D-069 r4 run disproved it.** R5 was order 60; R6 and R7 were 72 and 96. The measurement
below shows order 60 alone did not complete in 18 hours, and would not complete in years.

D-071 is therefore *this* decision: the plan the forward reference anticipated is retired inside the
decision it pointed at. Renumbering, or writing the closure as a fresh D-075, would leave a dangling
forward reference to a plan nothing withdrew — the record-drift class this chain has repeatedly caught
in its own surfaces.

---

## 2. Re-derived measurement — not quoted

Every number below was recomputed from the retained bytes for this decision. The retained files and
their digests:

| File | SHA-256 |
|---|---|
| `.agent-harness/runs/run-20260729-f10-d069-trajectory-r4/raw_logs/r4_trajectory_report.json` | `fbadb3e3dbf44509694d8b482915a98d78857d1090420181191ee191350044f2` |
| `…/raw_logs/r4_trajectory_stdout.log` | `28c541b35f68469cdda7f981c9949e8d4631fbc692b003a3c4bdce0d3462d93c` |

### 2.1 Mechanical outcome

`verdict: ERROR`; `error: "wall budget: frozen wall budget exceeded"`; `wall_seconds: 64801.9`
against a frozen budget of 64,800 s; started `2026-07-29T03:49:26Z`, completed `2026-07-29T21:49:28Z`.
**There is no top-level `checks` block** — confirmed by direct key inspection, not by reading prose.

### 2.2 Phase structure — and the error that window choice invites

The stdout log holds four phases. The evaluation counter **restarts in each**, which is the trap:

| Phase | Wall start | Evaluations | N |
|---|---|---|---|
| base, BDF order 48, `y_max` 24 | 0.0 s | 1 … 3651 | 0 → **6.2145**, completed `status=1` |
| mutant `M1_pair_sign` | 9,818.4 s | — | killed (solver failure) |
| mutant `M2_qem_sign` | 14,851.3 s | — | killed (coupled-energy residual) |
| **domain-holdout (T13), BDF order 60, `y_max` 30** | 15,860.7 s | 1 … 11051 | 0 → 0.1653 |

The base phase **completed**. Only the domain holdout stalled. A first pass at this re-derivation
mixed the two phases — because both log `eval 301` — and produced a nonsense estimate. It is recorded
here so the next reader does not repeat it.

### 2.3 The domain-holdout trace

```
eval     1   t=15865.2s   N=0.0000
eval   751   t=19173.7s   N=0.1813   <- peak, and the last new maximum
eval   951   t=~20000s    N=0.1629   <- unexplained DROP of -0.0184
eval  1501   t=22542.2s   N=0.1629
…
eval 11051   t=64718.9s   N=0.1653   <- final, wall budget exhausted
```

Two facts the retained trace establishes directly:

- **No new maximum was set in the final 10,300 evaluations**, consuming 45,545 s — **70 % of the
  entire wall budget**.
- Between the peak and termination the net change in `N` is **negative** (−0.0160).

### 2.4 Extrapolation, and its extreme window sensitivity

The estimate depends entirely on which window is used, and this must be stated because three different
windows give three different answers:

| Window | ΔN | Δevals | Projection |
|---|---|---|---|
| whole domain phase, from eval 1 | +0.1653 | 11,050 | 0.07 years |
| from eval 351 (straddles the drop) | +0.0025 | 10,700 | 4.66 years |
| **post-drop creep, eval 951 → 11051** | **+0.0024** | **10,100** | **4.58 years** |
| from the peak (spans the drop) | −0.0160 | 10,300 | no finite estimate |

**The post-drop creep is the defensible window**: it is the regime the integrator was actually in when
the budget expired. From it:

```
rate                 = 2.376238e-07 N/eval
seconds per eval      = 4.42062
N remaining to target = 7.771399   (target N = 7.936698865363719)
further evaluations   = 32,704,637          (3.27e7)
further wall          = 144,574,895 s       = 4.58 years
miss factor           = 8,305x the frozen 3,938-evaluation projection
```

Log rounding alone (`N` is printed to four decimals) spans **4.40 – 4.78 years**. This independently
reproduces the prior adjudication and the third-party auditor (~4.66 y, ~8,446×) to within that
envelope.

### 2.5 Uncertainty stated honestly

- The rounding envelope above is **not** a total uncertainty.
- **Model uncertainty is unquantified.** No step-size trace, rejection counter, error norm, or domain
  state dump survives.
- **The −0.0184 drop at eval 951 is unexplained and is not modelled by the extrapolation.** Whatever
  caused it could recur; if it does, the projection is optimistic.
- The defensible claim is **order-of-magnitude impracticality**, not a completion date.

---

## 3. What is NOT concluded

- **The underlying physical proposition is untouched.** r4 evaluated **no scientific predicate at
  all**: `evaluate()` sits after the wall-budget error handler and is unreachable on the retained
  path, and there is no `checks` block to inspect. This is a statement about an *instrument*, not
  about the physics.
- **T13 produced no result of any kind** — no `domain_holdout` output, no terminal state, no enclosure.
- Byte agreement across r4/r3/r2/D-056 on shared payload fields is **deterministic same-host
  reproducibility**, not independent corroboration: it is one computation re-run on one machine.
- Nothing here says a structurally independent validation is impossible. It says *this* instrument, at
  this order, on this host, does not complete.

---

## 4. Disposition and reopen conditions

`G-F10-INDEPENDENT-FLRW` is recorded **FAIL, closed on current measurement**. The reopen conditions
are written into the gate registry's `fail_disposition` field rather than into prose, so they are
carried with the gate.

**Reopening requires all of:**

1. a **materially new method** — an asymptotic reformulation, a rigorous surrogate bound, or an
   analytically constrained domain reduction — not a faster implementation of this one;
2. a **prospectively sealed contract**, frozen before any output byte, as D-069 required;
3. a **bounded discriminator** covering the phase that stalled here, projecting end-to-end completion
   inside the existing wall budget **with margin**, reviewed before implementation.

**Explicitly NOT reopening conditions:** generic optimization, faster hardware, more cores, or a
larger wall budget. A 10×–100× hardware gain does not close an ~8,300× evaluation-count miss, and the
retained failure is step-count dominated rather than per-evaluation-cost dominated.

---

## 5. Evidence preserved

All r4, r3, r2 and D-056 artifacts are retained unchanged. Nothing in this decision edits, deletes, or
rewrites a retained run. The D-069 contract and driver remain frozen and committed.

---

## 6. Cost

```
added_lines: this document + registry disposition field + three record rows
deleted_lines: 0
runtime_behavior_changed: no
physics_behavior_changed: no
known_blocker_reduced: no — this closes a lane, it does not clear a gate
blocker_movement_ratio: 0
cost_effectiveness_verdict: RECORD_ONLY
```

The board remains **6 PASS / 2 FAIL**.
