# Design Spec — Dual-Number AD in the Production BBN Collision Kernel

**Date:** 2026-07-21
**Status:** Design approved; implementation deferred (this document + adversarial audit only)
**Scope owner:** RABBIT solver / `native/rabbit_cpu`
**Related memory:** `project-rodas5p-ad-experiment`, `project-rodas5p-external-lab-analysis`, `project-rodas5p-bdf-cost-structure`

---

## 1. Goal

Port forward-mode dual-number automatic differentiation into the production RABBIT BBN
collision kernel so that Rodas5P can build an **exact** Jacobian and time-derivative (`dfdt`),
then run a **fair three-way comparison**: existing FD-Rodas5P vs BDF vs AD-Rodas5P.

This is **preliminary infrastructure for the Bianchi expansion** (state dimension grows
FLRW `n≈98` → Bianchi `n≈975`), where a dual-capable RHS is the prerequisite for
matrix-free AD-JVPs that avoid forming the dense `n²` Jacobian. It is **not** expected to
speed up FLRW — see §11.

## 2. Locked decisions (with rationale)

| Decision | Choice | Rationale |
|---|---|---|
| **Depth** | Full genericization of the collision RHS over a `Scalar` trait | Any operator-overloading AD needs a dual-capable RHS; every derivative routes through the same physics. No cheap partial exists. |
| **Engine** | Hand-rolled forward-mode dual numbers on stable Rust 1.94.1 | `std::autodiff`/Enzyme is **hard-blocked here**: the installed nightly lacks `libEnzyme` (`autodiff backend not found in the sysroot`), it is not yet rustup-distributed, and it rides nightly — breaking the pinned stable-1.94.1 + `cargo --locked` reproducibility contract on production physics code. Dual numbers are stable, deterministic, byte-reproducible, and lab-validated exact (relerr 0–3e-16). |
| **Consumer** | AD-Rodas5P builds the **full dense Jacobian** by `n` forward seeds + AD `dfdt` | User choice. Doubles as a bitwise-independent cross-check of the hand-derived analytic Jacobian. Expected slow at FLRW (see §11) — that is an honest, informative data point. |
| **Frozen paths** | BDF and existing FD-Rodas5P code paths untouched; f64 numerics byte-frozen | Preserves the frozen endpoint anchors (the same fixtures the B2 Illinois experiment broke). |
| **Deliverable** | Fair 3-way comparison as the capstone/acceptance test | Proves the AD path yields correct endpoints and measures its true cost. |

### Enzyme note (deferred, not rejected)
Enzyme differentiates the existing f64 code at LLVM level — it would **not** need the
`Scalar` rewrite (physics untouched), which is its real appeal. It is deferred purely on
availability: revisit once rustup distributes `libEnzyme`. The generic dual RHS and a future
Enzyme path are not mutually exclusive — Enzyme can later cross-validate the dual Jacobian.

## 3. Non-goals

- **No FLRW speedup claim.** The lab already established AD gives no integration gain at
  rtol=2e-7 (FD noise sits below the binding tolerance). This task delivers infrastructure +
  validation, not performance.
- **No matrix-free / Krylov linear solve for FLRW.** Memory: Krylov produces NaN/wrong
  answers on stiff BBN chemistry; dense LU at `n=98` is trivial. Matrix-free AD-JVP is a
  **Bianchi-only** architecture, unit-tested here but not wired to the FLRW hot stepper.
- **No genericization of the analytic-Jacobian machinery** (`jacobian_mev`,
  `jacobian_logit_mev`, `write_jacobian`). It stays f64, used only by FD-Rodas5P.
- **No change to default `OdeConfig`, `h_max`, or method selection** for BDF / FD-Rodas5P.
- **No hyper-dual (second-order) numbers** unless the T_γ-column consistency spike (§9, §10)
  proves first-order duals insufficient.

## 4. Architecture — single generic source

Rewrite each physics function `f(x: f64) -> f64` → `f<S: Scalar>(x: S) -> S` **in place**.
The f64 path becomes the `f::<f64>` instantiation — **one source of truth, no duplicate to
drift.**

### `Scalar` trait
Abstracts every operation the physics uses, each carrying its derivative rule:
`+ - * /` (both operand orders with f64), `exp`, `ln`, `sqrt`, `powi`, `powf`, `recip`,
`abs`, `max`/`min` (branch on `.re`, propagate the corresponding `.du`), ordering
comparisons (on `.re` only — matches f64 branching), `from_f64`, `is_finite` (on `.re`).
Implemented for `f64` (identity: every op maps 1:1 to the native IEEE op, same order) and for
`Dual`.

### `Dual` type
`Dual { re: f64, du: f64 }` — value plus a single directional derivative. **Single-direction**
by design:
- It is exactly the Bianchi matrix-free JVP primitive (one direction `v`).
- It is dimension-agnostic — no const-generic `N`, works unchanged FLRW→Bianchi.
- Full Jacobian = `n` seeded passes (seed `state[j]` with `du=1`, rest `du=0` → column `j`);
  `dfdt` = one pass seeding `ln_a` with `du=1`.

Vector-forward (K-channel `du`) to amortize the `n`-pass cost is an **optional optimization**
(§10 item R7), not baseline — baseline favors correctness and dimension-agnosticism.

### Rejected alternatives
- **Parallel dual copy** (leave f64 frozen, write a second dual RHS): ~3,700 lines
  duplicated → guaranteed drift as Bianchi physics grows. Rejected.
- **Hybrid** (genericize leaf math, keep f64+dual orchestration wrappers): less churn but two
  orchestration paths and boundary complexity. Rejected.

## 5. Scope refinements (discovered during design exploration)

1. **Values-path only for the spectral modules.** Both `electron_spectral` and
   `neutrino_self_spectral` already factor through `_impl(input, include_jacobian: bool)` with
   `_values = _impl(…, false)`. Only the **values arithmetic** is genericized; the
   `include_jacobian=true` branches (analytic Jacobian) stay f64. This roughly halves the
   genericization surface in the largest file (`neutrino_self_spectral.rs`, 2186 lines).
2. **`qed_eos` genericizes cleanly, derivative machinery included.** The `OnceLock` caches
   Gauss-Legendre **rule nodes/weights** (f64 constants, T_γ-independent), read as f64 and
   lifted into `Scalar`. The rule's Newton construction is not in the differentiated path.
   **However**, the RHS divides by `drho_dt` (an EoS *first* derivative used as a value), so
   the AD T_γ column needs ∂(drho_dt)/∂T_γ = **second-order**. `qed_eos` already computes the
   underlying integrals' "first two derivatives" in closed form; genericizing that machinery
   and seeding T_γ once produces the needed term. This makes the `qed_eos` derivative code
   part of the genericization surface (not just its value integrals).
3. **Serial RHS.** No rayon/threads in the differentiated path → bitwise determinism holds;
   no reduction-order hazard.

### Genericization surface (approximate)
| Module | Lines | Genericized portion |
|---|---|---|
| `qed_eos.rs` | 653 | value + first/second-derivative integral machinery (not the rule builder) |
| `quadrature.rs` / moments | 103 + `pair_moments` | integrand accumulation |
| `electron_spectral.rs` | 779 | values path (`include_jacobian=false`) |
| `neutrino_self_spectral.rs` | 2186 | values path only |
| `isotropic_boltzmann.rs` | RHS orchestration | `physical_state_impl` values path, `write_rhs`, `occupation_values`, `logits`, `collision_energy_moment` |

## 6. Components

1. **`autodiff.rs`** (new): `Dual{re,du}`, `Scalar` trait, f64 + Dual impls. Ported from the
   validated lab module, production-hardened, dimension-agnostic, standalone (zero physics
   risk). Unit-tested for exactness against analytic derivatives on toy functions.
2. **Generic physics** (the surface in §5): each function genericized over `Scalar`; f64
   instantiation byte-frozen (§7).
3. **AD Jacobian / dfdt** in `IsotropicBoltzmannFlrwSystem`: `n` forward seeds → full dense
   Jacobian; one `ln_a` seed → `dfdt`. Reuses the generic values RHS.
4. **Solver wiring — `AdSystem` wrapper** (decided; supersedes an earlier "new `SolverKind` or
   `OdeConfig` flag" sketch). AD lives in the *system*, not the solver: an
   `AdSystem<'a>(&'a IsotropicBoltzmannFlrwSystem)` newtype implements `OdeSystem`, forwarding
   `rhs` to the base f64 path and overriding `jacobian`/`dfdt` to the AD (full-J / seeded-`ln_a`)
   path built on the generic `Scalar` RHS. The comparison then runs the **same**
   `solve(SolverKind::Rodas5p, &ad_system, …)` — `SolverKind` and the BDF/solver enums are
   untouched. This decouples AD from solver selection and is the Bianchi-aligned home for the
   AD-JVP primitive. The AD Jacobian satisfies the existing `jacobian` contract (bit-stable per
   finite `(t,y)` within a solve — pure deterministic arithmetic, no seed-loop state).
5. **3-way comparison harness**: one `#[ignore]`d bench test, **identical `OdeConfig`** for
   all three solvers, reporting wall / n_rhs / n_dfdt / n_jac / n_ad_primal / endpoints
   (N, t, N_eff to 17 sig digits) side by side. AD-primal passes counted separately for
   transparency; wall time is the primary metric.

## 7. Bitwise-frozen f64 contract (the safety net)

**Claim:** `f::<f64>` reproduces the original literal-f64 function bit-for-bit. Rust performs
no fast-math contraction (no implicit fma; `a*b+c` stays two ops) and the `Scalar` f64 impl
maps each operation 1:1 in the same order, so the claim is expected to hold — **but it is a
hypothesis validated per function, not a language guarantee.**

- **Per-function guard test**: for every genericized function, a `to_bits()` equality test of
  `f::<f64>(x)` vs a retained reference literal-f64 computation, on a perturbed
  non-equilibrium input. (Pattern: the A2 `fused_rhs_and_jacobian_matches_separate_calls_bitwise`
  test.)
- **Fallback if a guard fails** (§10 R1): (a) adjust the generic expression to restore
  bit-identity (usually an operand-order or literal-lift fix); (b) if irreducible, keep that
  function f64-only and interpose a thin f64→Dual re-seed at its boundary (accepting AD
  inexactness localized there, documented); (c) worst case, exclude that term from the AD
  variant and record it. The frozen endpoint anchors for BDF / FD-Rodas5P are the ultimate
  gate — they must still pass bitwise.

### Validation cross-checks
- **AD-J vs analytic-J**, column-dependent tolerance:
  - occupation block (analytic, exact): agree to ~1e-11 relative (round-off of two different
    but exact arithmetic paths).
  - **T_γ column**: the analytic-J uses **FD** here, so AD-J vs analytic-J agree only to the
    FD noise floor (~1e-6…1e-10). Compare AD-T_γ-column against the FD column at FD tolerance,
    **not** machine precision. (A naive "match to machine eps" cross-check would fail — see
    §10 R4.)
- **AD-dfdt vs FD-dfdt**: agree to the FD floor; AD is the exact reference.

## 8. Fairness rules

- Identical `OdeConfig` for all three solvers in every recorded run.
- BDF and FD-Rodas5P numerics unchanged; only additive counters/instrumentation touch them.
- AD-Rodas5P is a **new variant** with its **own** endpoint row (enveloped, not held to the
  frozen bitwise anchors — its Jacobian differs from FD-Rodas5P's by construction).
- Report wall / n_rhs / n_dfdt / n_jac / n_ad_primal / endpoint side by side; same host,
  release build, `--locked`.
- AD-primal cost is reported honestly; no metric cherry-picking.

## 9. Staging (each stage committable + tested; comparison is the capstone)

| Stage | Content | Guard |
|---|---|---|
| **S0** | `autodiff.rs` module + unit tests (exactness vs analytic on toy fns) | module tests |
| **S1 (spike — architecture GO/NO-GO)** | Genericize `qed_eos` **including derivative machinery**; verify (a) `qed_eos::<f64>` reproduces the original bit-for-bit **and the frozen BDF/FD anchors still pass**, (b) AD T_γ EoS second-order term matches FD at FD tolerance | bitwise f64 guard + **frozen anchors** + T_γ 2nd-order cross-check |
| **S2** | Genericize `quadrature` / `pair_moments` | bitwise f64 guard |
| **S3** | Genericize `electron_spectral` values path | bitwise f64 guard |
| **S4 (spike)** | Genericize `neutrino_self_spectral` values path (largest, highest-risk) | bitwise f64 guard; feasibility spike first |
| **S5** | Genericize `isotropic_boltzmann` RHS orchestration; AD Jacobian + AD dfdt; AD-J-vs-analytic-J cross-check | bitwise f64 guard + column-dependent cross-check |
| **S6** | `AdSystem` wrapper (§6.4) + 3-way comparison harness; run; record ledger rows | frozen anchors (BDF/FD) + envelope (AD) |

**S1 is the architecture gate.** The whole single-source in-place strategy (§4) *bets* that
`f::<f64>` reproduces the original bit-for-bit. That bet is unproven until tried. `qed_eos` is
the smallest self-contained module and the first to touch BDF's f64 path, so S1 is the
GO/NO-GO: if `qed_eos::<f64>` cannot be made bit-identical (frozen anchors break and no
operand-order/literal-lift fix restores them), **stop and reconsider the architecture before
genericizing the other ~3,000 lines** — the §7 fallback ladder locally degrades toward the
very "parallel dual copy" §4 rejected, so widespread fallback means the chosen architecture is
wrong for this codebase, not merely inconvenient.

**De-risking option (§10 R5):** an intermediate "partial-AD" checkpoint after S3 — AD where
genericized, FD elsewhere — yields an early honest data point before committing to the 2186-line
S4. Decide at S3 whether to take it.

### Gates per stage
`cargo fmt --all -- --check` · `cargo check --release --locked` ·
`cargo clippy --release --all-targets -- -D warnings` · relevant module tests ·
the cheap frozen-anchor endpoint pair.

## 10. Risk register (feeds the adversarial audit — §12)

| ID | Risk | Severity | Mitigation |
|---|---|---|---|
| R1 | Bitwise-frozen f64 is a hypothesis, not guaranteed; a genericized fn may not reproduce bits | High | Per-fn `to_bits` guard; fallback ladder (§7). Anchors are the final gate. |
| R2 | T_γ-column needs 2nd-order EoS term; first-order duals correct **only if** `qed_eos` derivatives are closed-form-consistent (no hidden internal FD/table in the derivative path) | High | S1 spike; cross-check vs FD column. **Concrete cheap fallback**: keep the T_γ column via the existing FD in the AD variant (the AD variant becomes "AD everywhere except the T_γ column"), still a legitimate variant. Hyper-dual only if that fallback is also rejected. |
| R3 | Full-AD-J cost is `n` primal passes × ~300 Jacobian points, **and each dual primal pass is itself ~2–4× an f64 RHS** (double arithmetic, weaker autovectorization) → AD-Rodas may be tens× slower; a full-grid comparison could take many minutes | Medium (expected, not a defect) | Report honestly; wall is primary but note `n_ad_primal` cost ≠ `n_rhs` cost. If full-grid AD-Rodas is impractical to run repeatably, run the comparison on a **reduced grid** clearly labelled as such. Optional vector-forward (R7). Reinforces "don't use full-AD-J for FLRW". |
| R4 | AD-J-vs-analytic-J cross-check tolerance is column-dependent (analytic block tight, T_γ column FD-loose) | Medium | Column-split tolerances (§7). |
| R5 | All-or-nothing staging: comparison only lands at S6; a blocker at S4 wastes prior effort | Medium | Partial-AD checkpoint after S3 (§9). |
| R6 | `neutrino_self_spectral` (2186 lines) may contain constructs hard to genericize (f64-keyed lookups, special functions, match-on-f64) | Medium | S4 feasibility spike before committing. |
| R7 | Single-direction dual is simple but slow for full-J | Low | Vector-forward K-channel `du` as a later optimization; not needed for correctness. |
| R8 | edition-2024 trait-method vs inherent-method resolution (`x.exp()`) ambiguity in generic context | Low | Genericized code routes all calls through `Scalar`; resolve shadowing warnings explicitly. |
| R9 | AD `dfdt` may capture ln_a dependence more completely than FD, shifting AD-Rodas trajectory vs FD-Rodas | Low (expected) | Different variant, enveloped endpoint; documented. |
| R10 | Cost accounting: AD-primal passes are not trait-level `rhs` calls; counter semantics must be defined | Low | Dedicated `n_ad_primal` counter; wall time primary. |

## 11. Expected outcome

Per the lab verdict (`project-rodas5p-ad-experiment`): **AD-Rodas5P ≥ FD-Rodas5P wall, likely
notably slower** (full-AD-J is `n` primal passes vs analytic O(1)), both **~2× BDF**. FD noise
sits below rtol=2e-7, so exactness buys no integration gain at FLRW scale.

**Be honest about which result matters.** The wall-time ranking is close to a foregone
conclusion (AD-full-J is expensive; nobody expected otherwise). The genuinely *new* scientific
output of this task is the **AD-J-vs-analytic-J cross-check** — an independent, mechanically
different confirmation that the hand-derived analytic Jacobian is correct (occupation block to
~round-off; T_γ column to the FD floor). Plus the reusable **Bianchi-ready generic AD kernel**.
The three-way wall comparison is the acceptance test that the AD path integrates the real
problem to correct endpoints; it is not where the value lives.

## 12. Bianchi-forward notes

- The single-direction `Dual` is the exact matrix-free JVP primitive Bianchi needs (one
  direction `v`, no dense `n²` Jacobian).
- The generic `Scalar` RHS is the reusable foundation: Bianchi's expanded neutrino Boltzmann /
  collision terms, once written generic, get AD for free.
- When Bianchi's q-advection adds banded structure, right-preconditioned matrix-free GMRES
  (banded preconditioner) + AD-JVP + inexact-solve control in the scaled wrms norm is the
  route (validated by the external lab). Do **not** build that for FLRW.

---

*Implementation is deferred. This document is the design artifact; the adversarial self-audit
that accompanies it is recorded in §10 (risk register) and the audit summary delivered
alongside.*
