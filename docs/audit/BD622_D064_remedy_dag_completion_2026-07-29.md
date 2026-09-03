# BD622 D-058–D-064 — D-057 remedy DAG completion (2026-07-29)

Under the owner's full-DAG grant (2026-07-28), the five-step fail-closed
remedy prescribed by the D-057 adversarial audit executed to completion.
Every intermediate FAIL is preserved; every reissue was scoped by a
registered, lease-validated adjudication; all contracts froze before
output.

## Step 1 — run-identity lease (D-058, PASS)

`SubagentStart` seals an atomic per-agent lease (Start-time run id + raw
sha256 of every registered assignment); `SubagentStop` resolves the run
via the lease with legacy `ACTIVE_RUN` fallback, blocks unparseable
leases and post-Start tampering, consumes the lease only on acceptance.
`verify_assignment.py` is the canonical sealed-hash verifier (ends the
F-D057-04 role-hash false-positive trap). Live canary under two
overlapping runs: race accept under a stolen pointer, post-Start tamper
block + restored-retry accept, replacement canary clean; 12/12 hook
fixtures. `Q-HOOK-01` RESOLVED; `G-HARNESS-INTEGRITY` fail → pass.
Commit `b28ea0b`; evidence `E-HARNESS-D058-LEASE-CANARY`.

## Step 2 — claim/evidence rebinding (D-059, PASS, no gate moved)

Distinct stable claims registered (`C-F10-ROW9-CLOSURE` VALIDATED
bounded; `C-F10-METROLOGY-R3` / `C-F10-TRAJECTORY-R2` SPECIFIED); the
two dangling gate evidence IDs defined as preserved supportive;
prospective D-060/D-061 evidence keys added to the FAIL gates. Bounded
`.pyc` cleanup (`3f7a10a`). Commit `92ca30d`.

## Step 3 — asymmetric covariance (D-060 FAIL preserved → D-061 r4 PASS)

The r3 contract (freeze `17fedc6`) FAILED only on the degenerate
P-fixed N7X check — elastic energy conservation zeroes the
exchange-event Pauli affinity when mu/tau share one logit slope, leaving
`~1e-34` cancellation noise (D-054 S-B class contract-design defect;
FAIL preserved at `a754a4b`, report `e40d80a7`). The adjudicated
one-change r4 reissue (freeze `8492382`, report `56529e65`) PASSED all
11 checks on all 8 family members with physics bitwise-identical to r3:
equivariance `F(Pf)=PF(f)` at N1/N2 `<=1.2e-15` and native N3 max
`5.092e-11` under the `1e-10` cap on the asymmetric family; bitwise/N7X
rate equivariance covering off-grid legs; the identical production
two-map-plus-add graph bound with explicit `E_split`; outward mpmath.iv
tiers — 192-bit map replay everywhere, 128-bit full deep-node assembly
replay with `1522656` containment checks and zero failures; five mutant
kills with a bitwise restoration canary; D-055 anchors bitwise.
`C-F10-METROLOGY-R3` VALIDATED (`55b66e1`).

## Step 4 — completed-catalogue trajectory (D-062 FAIL preserved → D-063 r3 PASS)

The r2 contract (freeze `3849193`) FAILED on exactly two
frozen-parameter defects — the T8 sup at a Rust node below the first
GL48 collocation node, and the T10 mutant window below the measured
activity floor — while all ten physics checks passed (FAIL preserved at
`f938a12`, report `fc6f98bd`). The adjudicated two-change r3 reissue
(freeze `db9d0e3`, report `daf4fc06`) PASSED all 12 checks in a full
fresh 7:35 execution:

| Observable | Independent | Rust F10C2 BDF anchor | Delta | Band |
|---|---|---|---|---|
| `N_eff` | 3.034054308076679 | 3.03403598358439952 | `+1.83e-5` | `3e-4` (rejects the `7.44e-4` obsolete-anchor class) |
| `N_end` | 7.936698865363719 | 7.93669333948508360 | `+5.53e-6` | `1e-4` |
| `t_end` | 52678.732 s | 52677.634 s | `+1.10 s` | `5 s` |

Cross-code block anchors within `1e-3`; the T-drift-free flavour split
ratio delta `1.80e-5` vs `2e-4`; in-support mapped-spectrum sup
`1.588e-3` vs `2e-3` (sub-support extrapolation recorded non-gating);
coupled-energy residual `R_total <= 6.97e-5` with both source-sign
mutants KILLED at the measured activity peak (M2 `max_R_transfer 2.011`
vs kill `1.0`, matching the structural `~2.0` prediction; M1 by solver
collapse); tail/rejections in envelope; `rtol 3e-7` holdout drift
`-1.2e-5` vs `2e-4`. Base and holdout are bitwise-reproduced across
three executions (D-056 / r2 / r3). `C-F10-TRAJECTORY-R2` VALIDATED
(`8d92f24`).

## Step 5 — single-writer reconsideration (D-064)

With steps 1–4 complete: `G-F10-COVARIANCE-METROLOGY` and
`G-F10-INDEPENDENT-FLRW` fail → pass; `C-F10-COVARIANCE-METROLOGY` and
`C-F10-INDEPENDENT` → VALIDATED at matched-resolution scope.

## Terminal state

```text
G-F10C1-RADIAL             PASS
G-F10C1-REGRESSION         PASS
G-F10-PERFORMANCE          PASS
G-F10-CATALOGUE            PASS
G-F10-INDEPENDENT-FLRW     PASS   (D-064; evidence D-062+D-063)
G-F10-COVARIANCE-METROLOGY PASS   (D-064; evidence D-060+D-061)
G-F10-SCOPE                PASS
G-HARNESS-INTEGRITY        PASS   (D-058)
```

Limitations of record: matched-resolution N48-class agreement on the
frozen 10 MeV → 5 keV cell, module bytes `760a7c04` only; single
platform for the Python stack; same-model authorship and reviews;
degenerate-observable exclusions (P-fixed N7X, sub-support spectrum
nodes) rest on recorded structural rationales; no continuum,
absolute-normalization, or production claim. **The flips authorize
nothing further**: unblinding/handoff, public/production claims, W7/B3,
T01–T12, GL64/Radau, Rust/JAX forward work, F-11/Bianchi, and QKE remain
closed owner decisions under `G-F10-SCOPE`.
