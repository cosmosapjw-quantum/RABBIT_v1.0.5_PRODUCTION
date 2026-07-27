# BD622 D-045/D-046 — Owner grants and OWNER-A C-R6 orbit-chart closeout (2026-07-27)

## D-045 owner grants

The owner granted, in the 2026-07-27 blocker-resolution session: (1) handoff
durability for D-044; (2) one validation-only re-run of the full
`G-F10C1-REGRESSION` command bundle; (3) **OWNER-A** for exactly one
prospectively sealed, repository-free, nonphysical `C-R6-ORBIT-CHART`
discriminator with local sealed execution. OWNER-B (target replay against
the live `_independent_noqke.py` row-6 evaluator) and OWNER-C
(`_independent_noqke.py` modification) were defined and remain ungranted.
Durability landed as commits `5ae1a03` (D-044 sync + evidence) and
`b2542fc` (D-045 record, context `815eb4b9` → `400b0dc9`).

## Regression bundle result

Full bundle on commit `b2542fc` (run
`run-20260727-f10-d045-regression-rerun`, commit `717f193`):

| Leg | Result |
|---|---|
| `cargo fmt --all -- --check` | PASS |
| `cargo check --release --locked` | PASS |
| `cargo clippy --release --all-targets -- -D warnings` | **FAIL** — `clippy::too-many-arguments`, `gk_recurse` 10/7 args, `src/isotropic_boltzmann.rs:2513` (lib-test target) |
| `cargo test --release` | PASS — 238 passed / 0 failed / 2 ignored, 862.23 s |
| LaTeX report | PASS — 54 pages, zero undefined refs/citations, pages 24/25/32/33 visually clean |

The sole failure is a successor-branch lint regression absent from
historical closeouts; the historical `221/221`/`230/230` evidence is
confirmed stale for this HEAD. Validation-only stage: no source edited,
`G-F10C1-REGRESSION` stays FAIL under its own fail condition.

## D-046 OWNER-A discriminator — adjudicated PASS

Sealed lab `BD622_OWNERA_R6_ORBIT_CHART_LAB/` (seal commit `4c05ce2`,
contract `41bfb8ba`, run `run-20260727-f10-ownera-r6-orbit-chart-lab`).
Finalist: two explicitly paired ordered orientation members
`M+ = (νμ, ν̄μ → ντ, ν̄τ)`, `M− = (ντ, ν̄τ → νμ, ν̄μ)`, each with
native-incoming legs 1,2 and interpolated-outgoing legs 3,4; frozen
`(+,+,−,−) × gain-minus-loss` product; derived Reynolds quotient `1/2` by
orbit–stabilizer for the free Z2 action `σ = P∘R`.

Sealed execution (single `./run_lab.sh`, `unshare -rn`, negative network
probes, python 3.12.3 / numpy 2.4.2 / scipy 1.17.0 runtime-asserted,
wall 353 s): acceptance battery ANCHOR/A1–A7/A9/A10 all PASS, including:

- exact slot-identity witness `D(f;z)=1/12`, `D(Pf;z)=5/144=−D(f;Rz)`;
- single-member equivariance fails exactly (max residual
  `−3254999227/17179869184`) while the closed two-member object is
  equivariant exactly (rational) and bitwise (both float engines);
- Reynolds idempotence holds for `q=1/2` and kills `q∈{1, 1/4}`;
- antisymmetric response retained: `S_anti = 2194213851/268435456`
  (rational) / `7.264887185935607` (GL8), exactly zero on the symmetric
  fixture;
- exact number/energy nulls and strict per-event entropy certificates over
  84 traced events; dual-number JVP carries both member contributions;
- 4096-case dyadic sweep bitwise-equivariant with exact float number nulls;
- 10/10 semantic mutants KILLED for their preregistered reasons;
- primary + two fresh-process replays byte-identical on
  `{RESULTS.json, MUTANTS.json}`; complete 29-file `MANIFEST.sha256`.

Evidence hashes: RESULTS `b2ac53f7`; MUTANTS `9da373d1`; manifest
`504095f2`; chronology `69fec2dd`; adjudication `569f7d6e`; merged
`f7a74b26` (six envelopes, 9/9 exact-unique findings). Context
`400b0dc9` → `e38004aa`; closeout run
`run-20260727-f10-d046-ownera-closeout`.

## Terminal statuses

```text
VERDICT: PASS
C-R6-ORBIT-CHART-MICROCASE = VALIDATED   (manufactured nonphysical microcase only)
lab code = IMPLEMENTED
C-R6-ORBIT-CHART-RABBIT-APPLICABILITY = PROPOSED
GATES: UNCHANGED
TERMINAL: REQUEST_OWNER_B_AUTHORIZATION_FOR_TARGET_REPLAY_REVIEW
```

Recorded limitations: same-model blind reviews (intra-session, blind to
results but not organizationally external); pre-seal static review-fix loop
(three would-be-fatal defects removed before any output existed);
manufactured-microcase-only scope (no collision physics, normalization,
finite-`m_e`, or F-10 gate validation); probabilistic MUT-5 kill;
wrapper console-line gap; version-string pins in place of a digest-pinned
image; single-host determinism.

## Boundary

The terminal line is a request only. `OWNER-B=REQUIRED_NOT_GRANTED`; a
structurally passing replay would still require OWNER-C for any
`_independent_noqke.py` modification. Every `G-F10-*` gate,
`G-EXT-FEASIBILITY`, W7, B3, T01–T12, Rust/JAX forward work, Radau,
trajectory, endpoint, unblinding, F-11/Bianchi, QKE, and public work remain
unchanged and closed.
