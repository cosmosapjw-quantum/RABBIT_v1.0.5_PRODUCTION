# BD611 Clean-Core Development Direction Meta-Plan

Date: 2026-07-02

Scope: future planning after the clean dynamic-collision FLRW driver and AP65
deflation track. This is not an implementation PR and not a validation ledger.
It is a decision document for what to measure next, when to preserve the current
Python/JAX/Rodas5P path, and when to switch platform before development speed
collapses under slow numerical loops.

Controlling constraints:

- QKE remains out of scope.
- Public production and publication-ready support must not be claimed.
- Do not add another gate, manifest, figure, hash, or readiness surface unless
  it deletes or consolidates stale surface and moves a runtime physics, solver,
  endpoint, or performance blocker.
- Preserve raw failed numerical states. Do not hide negative, clipped,
  nonfinite, or non-endpoint evidence.
- Internal consistency is progress, but not physical validation.

## 1. Current Implementation Level, Without Erasing Progress

The correct tone is not "nothing was achieved." The correct tone is:

| Surface | Claim status | What it gives us | What it does not give us |
|---|---|---|---|
| Energy-variable clean 3T core | IMPLEMENTED | Removes the old `1/T_nu^3` temperature-amplifier collapse mode and gives a bounded bank-energy formulation. | Does not prove the collision rate normalization is physically correct. |
| Comoving FLRW decoupling driver | IMPLEMENTED | Executes a fixed-comoving-momentum decoupling solve to the endpoint, with fail-closed endpoint guards and no heavy-bank collapse in focused tests. | Does not externally validate `N_eff` or spectral distortions. |
| By-construction energy conservation | IMPLEMENTED | Ensures the plasma loses exactly the energy gained by the evolved neutrino state in the implemented discretization. | Does not prove that the gained energy equals the continuum collision transfer. |
| Collisionless entropy limit | IMPLEMENTED / PARTIAL | Reproduces a stable finite-mu/QED EOS collisionless endpoint near `N_eff = 2.9934`, independent of `n_q`. | Should not be described as exact ideal `N_eff = 3` unless the ideal EOS path is separately run. |
| `deterministic_reference` collision kernel | IMPLEMENTED / BLOCKED | Gives detailed-balance zero and correct heating sign on focused FD offsets. | Rate dimension and normalization remain a load-bearing physics blocker. |
| AP65 deflation | IMPLEMENTED / PARTIAL | Deletes a large stale audit-only surface and preserves a smaller B5 operator substrate. | Leaves stale scripts, docs, and capability strings that must not be treated as current capability. |
| B5 non-LRS island | IMPLEMENTED AS SUBSTRATE | Operator-level realizability and S2 radial-grid support remain useful. | Not a driver-integrated non-LRS/full-BBN collision capability. |

Safe current summary:

> RABBIT now has an executable clean FLRW decoupling core that fixes the
> q-Laguerre collapse failure mode as an engineering blocker. Its physical
> collision normalization, exact energy-transfer accuracy, and external FLRW
> anchor remain unresolved. The post-deflation codebase is smaller, but its
> stale AP65 claim and script surfaces still need cleanup.

Forbidden summary:

> The clean core has validated Standard Model neutrino decoupling, Bianchi-I
> collision physics, non-LRS full-BBN, or publication-ready `N_eff`.

## 2. Diverge Stage: Directly Executable Numerical Experiment Candidates

The next work should branch into experiments that can be executed in hours, not
weeks. Each candidate must either promote a specific claim boundary or kill a
bad direction cheaply.

### E1. Exact-Transfer Plasma Coupling Probe

Question: does by-construction conservation hide a biased energy transfer?

Chain-of-code:

1. Build the existing `integrate_flrw_decoupling` trajectory.
2. At accepted trajectory samples, recompute:
   - current driver gain: `G_df = integral Y^3 df(Y) dY`;
   - exact thermal transfer: `G_th = z^4 * integral q^3 C(q) dq / H`.
3. Run two endpoint variants:
   - current df-derived plasma coupling;
   - exact thermal-transfer plasma coupling.
4. Record `rel(G_df/G_th - 1)`, `N_eff`, `T_nu_e/T_gamma`,
   `T_nu_x/T_gamma`, accepted clip excursion, and wall time.

Minimal command shape:

```bash
env -u RABBIT_NUDEC_BSM_PATH PYTHONPATH=src JAX_PLATFORMS=cpu \
  venv/bin/python scripts/probe_clean_core_transfer_accuracy.py \
  --n-q 16 24 32 40 --rtol 1e-8 --out audit_outputs/bd611_transfer_probe.json
```

Acceptance:

- Internal: endpoint reached, no accepted Pauli/positivity repair above `1e-6`.
- Accuracy: active-region median transfer mismatch below 1 percent, or the
  mismatch is explicitly treated as a blocker.
- Claim boundary: `N_eff` cannot be promoted while current-vs-exact transfer
  endpoint delta remains larger than the target tolerance.

### E2. Collision Normalization and Gamma/H Scaling Probe

Question: is `deterministic_reference` a rate in MeV with the correct weak
2-to-2 scaling?

Chain-of-code:

1. Construct a small FD perturbation, e.g. `f = f_FD(q / 0.99)`.
2. Evaluate the deterministic collision field for `T = 10, 5, 3, 2, 1.5, 1`
   MeV.
3. Convert the energy relaxation to an effective `Gamma/H`.
4. Compare the log-slope against the canonical rate helper:
   `rabbit.thermo.rate_prefactors.total_rate_nu_nu_diagonal` and
   `rabbit.jax.collision_rates_jax`, both using `Gamma ~ G_F^2 T^5`.
5. Repeat with a controlled `T` prefactor multiplier only as a diagnostic, not
   as a tuned fix.

Minimal command shape:

```bash
env -u RABBIT_NUDEC_BSM_PATH PYTHONPATH=src JAX_PLATFORMS=cpu \
  venv/bin/python scripts/probe_clean_core_collision_rate_scaling.py \
  --n-q 24 --temperatures 10 5 3 2 1.5 1 \
  --out audit_outputs/bd611_gamma_over_h_scaling.json
```

Acceptance:

- `Gamma/H` follows the expected weak-rate scaling after division by the same
  Hubble model.
- The inferred decoupling window is plausible without tuning `N_eff`.
- If `T^4` vs `T^5` remains ambiguous, the clean-core `N_eff` stays
  IMPLEMENTED/BLOCKED, not VALIDATED.

### E3. Collisionless EOS Split Probe

Question: is the `N_eff = 2.9934` undershoot a physical EOS convention effect
or a frame/moment bug?

Chain-of-code:

1. Run current finite-mu/QED plasma EOS collisionless driver.
2. Run a deliberately idealized entropy/EOS variant where the analytic
   collisionless endpoint should return ideal `N_eff = 3`.
3. Compare `z_final`, `I_pair/I_FD`, `T_nu/T_gamma`, and `N_eff`.

Minimal command shape:

```bash
env -u RABBIT_NUDEC_BSM_PATH PYTHONPATH=src JAX_PLATFORMS=cpu \
  venv/bin/python scripts/probe_clean_core_collisionless_eos_split.py \
  --n-q 16 24 32 48 --out audit_outputs/bd611_collisionless_eos_split.json
```

Acceptance:

- finite-mu/QED path reproduces its own entropy endpoint.
- ideal path, if implemented, returns ideal `N_eff = 3` within tight tolerance.
- Documentation says which EOS convention each number belongs to.

### E4. N_q, Tolerance, and Integrator Convergence Ladder

Question: is the collision-on endpoint converging to a continuum value or to a
self-consistent discretization value?

Chain-of-code:

1. Run `n_q = 16, 24, 32, 40, 48` for both current and exact-transfer plasma
   variants.
2. Run `rtol = 1e-6, 1e-8, 1e-9` at the selected `n_q`.
3. Compare Radau against one alternate stiff method only as a diagnostic.
4. Record endpoint, step count, RHS calls, accepted clip excursion, and wall.

Minimal command shape:

```bash
env -u RABBIT_NUDEC_BSM_PATH PYTHONPATH=src JAX_PLATFORMS=cpu \
  venv/bin/python scripts/probe_clean_core_convergence_ladder.py \
  --n-q 16 24 32 40 48 --rtol 1e-6 1e-8 1e-9 \
  --out audit_outputs/bd611_convergence_ladder.json
```

Acceptance:

- Endpoint differences shrink monotonically or the nonconvergence mechanism is
  localized.
- Segment-only convergence is labeled segment-only.
- No `N_eff` number is promoted solely because it is close to 3.044.

### E5. Backend Bake-Off Microbenchmark

Question: is development speed now limited enough that the runtime substrate
must change?

Chain-of-code:

1. Time fixed-size kernels, not whole vague workflows:
   - one deterministic collision field evaluation;
   - one RHS call inside `_make_rhs`;
   - one full endpoint solve at `n_q = 16` and `n_q = 24`;
   - optional `n_q = 40` if the first two are fast enough.
2. Compare current Python/NumPy/SciPy against existing JAX collision kernels
   where parity is already available.
3. Prototype Numba/Pythran/Cython only for the collision quadrature loop if the
   profile proves Python loop overhead dominates.
4. Record compile time separately from steady-state runtime.

Minimal command shape:

```bash
env -u RABBIT_NUDEC_BSM_PATH PYTHONPATH=src JAX_PLATFORMS=cpu \
  venv/bin/python scripts/probe_clean_core_backend_bakeoff.py \
  --n-q 16 24 --repeat 5 --out audit_outputs/bd611_backend_bakeoff.json
```

Acceptance:

- A backend migration is justified only if it moves measured wall time on the
  same physics case by at least 3x or removes an interactive development
  bottleneck without weakening invariants.
- Compile time and first-run latency must be reported separately.
- Backend parity must include detailed balance, heating sign, degeneracy, and
  transfer-moment checks before endpoint claims.

### E6. Post-Deflation Runtime Hygiene Sweep

Question: is the smaller repository actually easier to run, or did deletion
leave runtime traps?

Chain-of-code:

1. AST-scan active scripts for imports of deleted `rabbit.validation`,
   `rabbit.jax`, `rabbit.weak`, and `rabbit.transport` modules.
2. Run representative `--help` or dry-run commands for active scripts only.
3. Run focused clean-core tests and a non-slow test slice.

Minimal command shape:

```bash
env -u RABBIT_NUDEC_BSM_PATH PYTHONPATH=src JAX_PLATFORMS=cpu \
  venv/bin/python scripts/probe_post_deflation_runtime_hygiene.py \
  --scripts scripts --out audit_outputs/bd611_deflation_hygiene.json
```

Acceptance:

- No active script imports deleted AP65 modules.
- Historical scripts are either deleted, moved to provenance, or fail with an
  explicit deprecated-surface message.
- `promotion_check.py --status` remains fail-closed rather than falsely green.

## 3. Metacognitive Filter Before Selecting Work

Ask these before each PR:

1. Am I merely lowering the tone of claims, or am I preserving a real executable
   path and giving it the next decisive experiment?
2. Am I treating `N_eff` closeness, self-consistency, or conservation as
   physical validation?
3. Am I optimizing a cheap segment while the measured wall bottleneck or
   physics-normalization blocker remains?
4. Am I proposing a platform rewrite because the code is slow, or because a
   bake-off proves the current substrate blocks the next necessary experiment?
5. Will the proposed patch delete or consolidate stale surface, or does it add
   another planning/gate layer?

Failure answer: if the answer to 2 or 3 is "yes", stop. If the answer to 4 is
"because it feels slow", run E5 before migrating.

## 4. CoVe: Verification Questions for the Roadmap Itself

| Claim to verify | Independent verification question | Decision if answer is bad |
|---|---|---|
| Clean core is useful progress | Can it execute endpoint FLRW decoupling, fail closed, and avoid heavy-bank collapse under focused tests? | Preserve it as engineering substrate, but do not promote physics. |
| Transfer coupling is trustworthy | Does exact thermal transfer agree with df-derived transfer along the trajectory? | If no, fix transfer before external anchor. |
| Collision normalization is trustworthy | Does deterministic `Gamma/H` share canonical weak-rate scaling and plausible decoupling window? | If no, freeze `N_eff` claims and fix prefactor/dimension. |
| Current platform is viable | Does backend bake-off show acceptable wall time for the next 20-50 experiment runs? | If no, migrate the hot kernel, not the whole repo. |
| Deflation improved maintainability | Are active scripts/tests import-clean after AP65 deletion? | If no, cleanup is a blocker, not optional polish. |

## 5. CCoT: Correct Path vs Common Wrong Path

| Topic | Correct path | Common wrong path to reject |
|---|---|---|
| Conservation | Plasma loss equals evolved neutrino gain, then separately test gain accuracy. | Declare conservation as collision-rate validation. |
| N_eff | Treat endpoint number as an observable to anchor after rate/transfer checks. | Tune prefactors until `N_eff` is near 3.044. |
| Platform | Profile fixed kernels and migrate only hot, parity-locked kernels. | Rewrite the whole project into a new language before physics contracts settle. |
| JAX | Use for static-shape batched/repeated kernels with compile amortized. | Put dynamic solver control and Python-heavy callbacks behind JIT and call it done. |
| Numba/Cython/Pythran | Use for deterministic CPU quadrature loops if Python overhead dominates. | Add a second implementation without parity gates. |
| Rust/C++/Fortran | Consider for stable kernels after contracts freeze. | Port moving physics code and multiply validation burden. |
| Julia/SciML | Use as an external experimental oracle or prototype if it gives solver insight. | Make it the production rewrite without migration criteria. |

## 6. Converged Experiment Order

### P0. Collision Normalization First

Run E2 before changing backend or expanding physics. If the collision field is
dimensionally wrong or under-normalized/over-normalized, every later performance
result optimizes the wrong equation.

Output required:

- `Gamma/H` ladder.
- log-slope estimate.
- comparison to `rabbit.thermo.rate_prefactors` and
  `rabbit.jax.collision_rates_jax`.
- decision: `T^4 accepted`, `T^5 repair required`, or `normalization unresolved`.

### P0/P1. Exact Transfer Second

Run E1 after or alongside E2. This isolates the conservation-vs-accuracy
problem without touching non-LRS/full-BBN scope.

Output required:

- current vs exact-transfer `N_eff` delta.
- transfer mismatch statistics over accepted samples.
- endpoint diagnostics and wall time.

### P1. Backend Bake-Off Third

Run E5 once the two physics probes define the correct RHS shape. Do not wait
until the whole project is too slow to move, but do not migrate before the hot
equation is known.

Output required:

- Python/NumPy/SciPy baseline timing.
- existing JAX-kernel parity/timing where available.
- compile vs steady-state separation.
- recommendation: keep, JAX-port, Numba/Pythran/Cython, Rust/C++, or external
  prototype only.

### P1. Deflation Hygiene in Parallel

Run E6 in parallel with the physics probes if write sets stay separate. This is
not a validation upgrade; it prevents false green and restores developer speed.

Output required:

- missing import list.
- active vs historical script classification.
- focused test result.

### P2. EOS and Convergence Ladders

Run E3 and E4 after the normalization/transfer decision. They refine endpoint
confidence; they should not block the first two decisive probes.

## 7. Platform and Language Decision Matrix

Current code is too slow for unconstrained exploratory development. That does
not automatically imply a whole-platform rewrite. It implies a measured
migration gate.

| Option | Use when | Do not use when | Near-term verdict |
|---|---|---|---|
| Python + NumPy + SciPy | Reference correctness, readable probes, one-off audit experiments. | Repeated sweeps require many endpoint solves or Python callback overhead dominates. | Keep as reference lane. |
| JAX/XLA CPU/GPU | Static-shape kernels, batched sweeps, repeated inference, existing parity surfaces. | Dynamic solver control and shape-changing quadrature dominate; compile time is not amortized. | Candidate for collision/RHS kernels after E2/E1. |
| Diffrax/JAX solver | Need fully compiled differentiable solve with stable shapes. | Stiff event-driven control and fail-closed endpoint handling become harder than SciPy. | Prototype only after RHS parity. |
| Numba / Pythran / Cython | Deterministic CPU quadrature loop is the bottleneck and contracts are stable. | Need GPU/autodiff or many variant kernels. | Strong near-term candidate for `deterministic_reference` hot loops. |
| Rust or C++ extension | Stable kernels need long-term speed and memory control. | Physics formulas, frame contracts, and normalization are still moving. | Defer until E1/E2 settle; then consider one kernel, not repo rewrite. |
| Fortran/SUNDIALS/CVODE style core | Stiff solver linear algebra dominates and Python callbacks are the bottleneck. | Collision kernel is still the dominant cost or state layout is unstable. | Candidate only after bake-off proves solver-core bottleneck. |
| Julia/SciML | Need an external solver oracle or rapid stiff-method prototype. | Want production continuity inside current Python package. | Use as external comparison, not migration target. |

Migration trigger:

```text
backend_change_allowed = (
  physics_contract_frozen_for_target_kernel
  and same_case_speedup >= 3x
  and parity_tests_cover_detailed_balance_sign_degeneracy_transfer
  and compile_or_bridge_cost_reported
)
```

If this condition is false, platform work is likely drift. If it is true,
delaying migration is also drift because it slows every subsequent scientific
experiment.

## 8. Minimal PR Sequence

### PR-1: Gamma/H Scaling Probe and Decision

- Type: tests/script + short audit note.
- Expected cost: 1-2 files, net <= 200 LOC.
- Blocker movement: decides whether the clean-core collision field is a
  physical rate or requires prefactor repair.
- Validation: E2.
- Claim status after pass: collision normalization can move from BLOCKED to
  DERIVED/PARTIAL, not VALIDATED unless externally anchored.

### PR-2: Exact-Transfer Endpoint Variant

- Type: small driver option or script-only comparative probe.
- Expected cost: 1-3 files, net <= 250 LOC.
- Blocker movement: separates conservation from energy-transfer accuracy.
- Validation: E1 plus focused driver tests.
- Claim status after pass: endpoint `N_eff` can be reported as internally
  transfer-consistent; still not external validation.

### PR-3: Backend Bake-Off and Hot-Kernel Migration Decision

- Type: benchmark script and decision note; optional one-kernel prototype only
  if E5 shows a clear target.
- Expected cost: 1-3 files for bake-off; migration split into a later PR.
- Blocker movement: reduces iteration wall time or decides not to migrate.
- Validation: E5 plus parity checks.
- Claim status after pass: performance substrate decision SPECIFIED.

### PR-4: Post-Deflation Hygiene Cleanup

- Type: deletion/consolidation, not new gates.
- Expected cost: net negative LOC preferred.
- Blocker movement: removes stale AP65 runtime traps and false-green risk.
- Validation: E6, focused import tests, `promotion_check.py --status`.
- Claim status after pass: current capability surface becomes honest.

## 9. Claim Ledger for Future Work

| Future claim | Allowed status before experiments | Promotion condition |
|---|---|---|
| Clean FLRW driver reaches endpoint | IMPLEMENTED | Focused endpoint tests stay green after E1/E2 changes. |
| Collision-on `N_eff` is physically meaningful | PROPOSED / BLOCKED | E2 passes, E1 mismatch controlled, FLRW external anchor passes. |
| Clean core is faster enough for development | SPECIFIED | E5 proves wall-time budget is acceptable. |
| Backend migration is necessary | PROPOSED | E5 proves current substrate blocks planned experiment throughput. |
| B5 supports future non-LRS extension | IMPLEMENTED AS SUBSTRATE | Driver-level integration and non-LRS tests are added later. |
| Full-BBN coupling is ready | PROPOSED | FLRW decoupling is normalized/anchored and a network interface is specified. |

## 10. What Not To Do Next

- Do not add another broad roadmap, readiness gate, or claim wrapper.
- Do not port the whole repo to Rust, C++, Julia, or JAX before E1/E2 define
  the correct equation to port.
- Do not tune collision constants to recover 3.044.
- Do not start non-LRS/full-BBN expansion until FLRW collision normalization and
  transfer accuracy are settled.
- Do not delete B5 simply because it is not driver-integrated; keep it as a
  named substrate unless E6 shows it is unreferenced dead code.

## 11. Next-Session Prompt

```text
Read AGENTS.md, docs/TYPEI_AUGMENTED_NOQKE_CODEX_ANTI_DRIFT_GUARDRAILS.md,
bbn_codex_anti_drift_cost_effective_policy.md, and
docs/audit/BD611_clean_core_development_direction_meta_plan_2026-07-02.md.

Goal: execute the next Rabbit clean-core no-QKE PR only if it runs one of the
BD611 decisive probes and moves a named blocker. Start with E2 Gamma/H scaling
or E1 exact-transfer coupling. Do not claim physical validation from
conservation, self-consistency, or N_eff proximity. Do not do a platform rewrite
unless E5 proves a same-case speedup target with parity gates. Prefer net
deletion or small scripts over new planning surfaces. Report added/deleted/net
lines, exact commands, validation status, and remaining blocker.
```
