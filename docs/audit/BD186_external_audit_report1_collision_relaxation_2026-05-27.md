# External Audit Report — BD186 Augmented Type-I No-QKE Collision Blocker

Reviewer scope: independent read of the attached source, docs, and the fresh BD186
JSON artifacts. Everything below was checked against the code and the telemetry, not
inferred from the self-overview. QKE treated as out of scope; no public-production
claim is made or assumed; CPU-JAX/Rodas5P kept as the backend target.

---

## Executive verdict

The self-diagnosis ("the blocker is the dynamic collision source/Jacobian coupling,
not weak rates or the phase-2 corrector") is **directionally correct but imprecise in a
way that matters for the next PR.** The precise failure, read off the artifacts, is:

> The `q14_collision_N4p8` row **does not run out of budget or steps** — it terminates
> on `continuous_ap65_final_state_nonfinite` + `continuous_ap65_source_payload_exception`
> at N≈3.9 of 4.8, having used 131.6 s of a 300 s budget. It is a **numerical blow-up**,
> not a slowness/OOM limit.

The blow-up channel is concrete and code-verifiable, and it is the *interaction* of three
things, not any one of them:

1. The collision source is treated **explicitly** in the implicit solve (the
   `frozen_source_jax_full_jvp` Jacobian holds the payload constant, so
   ∂(collision)/∂(A, Σ, T) ≡ 0 in the operator).
2. The neutrino distribution is reconstructed and fed into **energy moments whose
   quadrature weights span ~9 orders of magnitude** (w_E ≈ 2.5e-4 at q=0.1 vs ≈ 9.8e5
   at q=44.4 for the q14 grid). Those moments drive the anisotropic stress Π, which drives
   the shear Σ.
3. The q-transport operator is a **non-conservative pointwise polynomial derivative**
   (`f' ≈ D@f`) with **no occupation bound** on the live path — unlike the prototype
   `evolve_nonlinear`, which clips `f∈[0,1]` every step.

So a small high-q tail error grows (explicit collision + non-conservative q-transport),
gets amplified ~1e6 in the energy/stress moment, blows up Σ, and the next collision-source
build hits `_finite_array(dA_modes)` (replay.py:1250), which raises. The exception is a
**symptom of an upstream state that the solver was allowed to reach**, not the root cause.

This refinement changes the recommended PR (see Q7): a structured collision Jacobian alone
is necessary but probably **not sufficient** unless it is *sign-complete* (the current
optional diagonal captures only damping) and paired with a bound that prevents the source
from ever being evaluated on an unphysical distribution.

---

## Q1 — Is the blocker diagnosis correct? (collision source/Jacobian vs weak rates vs phase-2)

**Largely yes for the attribution; the mechanism is sharper than "stiffness."**

Evidence supporting the collision attribution (independently confirmed):
- Non-LRS-only and non-LRS+weak rows reach the endpoint; the collision-on row is the first
  to fail. The weak-rate path completes with similar host counters. Weak rates are correctly
  demoted.
- The phase-2 corrector completes on the failing row (341 corrector steps, mass-conservation
  delta max ≈ 9.7e-13, raw-negative candidate count 0). It is not the thing that dies.

Where the self-diagnosis is *imprecise*: it frames the residual as "numerical
stiffness/source-coupling plus expensive collision contractions." The artifact shows a
**hard nonfinite termination with budget unexhausted** (`failed_unknown_temperature`,
violations include `continuous_ap65_final_state_nonfinite` and
`continuous_ap65_source_payload_exception`; 131.6 s / 300 s; reached N≈3.9/4.8). That is a
stability/representation failure, not a cost ceiling. Calling it "stiffness" invites a
step-control or cost fix; the actual lever is the explicit-collision treatment plus the
unbounded high-q amplification path.

Net: the blocker *is* in the dynamic collision path, but it is a **collision-driven
instability amplified through the energy-moment → shear feedback**, surfacing as a source-build
exception. Correct family, wrong adjective.

---

## Q2 — Is the collision source algebra dimensionally and sign-consistent?

**Yes, at every level I could verify numerically and by reading. No sign or normalization
defect was found.** The fragility is robustness, not correctness.

Checked and confirmed:
- **Raw vs energy weight separation** (the BD-series fix): numerically exact. For the q14
  grid, Σ w_raw = 1.000000 (= ∫e^−q dq); w_E = w_raw·e^q·q³ to machine precision. The
  Gauss-Laguerre rule reproduces the FD energy moment ∫q³/(e^q+1) = 7π⁴/120 to **6 significant
  figures** (5.682198 vs 5.682197) and the number moment 3/2·ζ(3) to 6 figures. The contract
  is right and accurate for FD-like integrands.
- **PSTF radial moment powers**: number ∝ E², energy ∝ E³ (process_catalog.py:920–921) —
  dimensionally correct phase-space weights.
- **Moment-neutral projection**: built on the FD linear-response basis f_eq(1−f_eq) and
  E·f_eq(1−f_eq), inverted with a regularized pinv (rcond=1e-14). Reasonable and stable.
- **p4 interpolation / six-monomial contraction**: structurally consistent with the
  gain–loss bookkeeping; no inverted sign in the dQ/dA assembly that I could find.

The real Q2 issue is not the algebra but **conditioning**: the energy weights span ~1e6, so
the moments (and therefore Π, Σ, and the stress feedback in the Jacobian) are only trustworthy
while the reconstructed distribution keeps FD-like exponential decay. The code's pervasive
`_finite_array(...)` guards convert any loss of that property into a hard `ValueError` rather
than a graceful degradation — which is exactly the `source_payload_exception` seen. So the
source is *correct but brittle*: correct on physical states, exception-raising on the drifted
states the solver is currently allowed to reach.

---

## Q3 — Is the host Jacobian missing a structurally important derivative?

**Yes, and the specific missing piece is identifiable.**

The blocker row ran `jacobian_policy = frozen_source_jax_full_jvp`. In that policy the collision
payload (`dA_modes`, `dQ_*`) is **frozen** during the JVP, so the implicit operator contains
**zero** contribution from ∂(collision.dA_modes)/∂A, ∂(collision)/∂(Σ,T), or
∂(dQ)/∂(T,A). The stress-feedback path A→ρ,Π→Σ *is* in the Jacobian (those moments are
recomputed inside the JVP from `reconstruct(A)`), but the collision's own response is not.

The optional repair (`frozen_source_jax_full_jvp_collision_diag`) is **inadequate by
construction and was OFF in the blocker run anyway:**

```python
# rhs.py:11453
damping = finite & (np.abs(A) > floor) & ((A * dA) < 0.0)   # only A·dA < 0
diag[damping] = dA[damping] / A[damping]
diag = np.clip(diag, -cap, 0.0)                             # clamp to ≤ 0
```

This (a) is a **secant ratio dA/A, not a derivative**; (b) activates **only in the damping
direction** (`A·dA<0`) and zeroes the **growth direction** (`A·dA>0`) — the code's own
metadata says so: `collision_source_growth_derivatives_not_matching_damping_sign`; and (c)
is **diagonal-only**: no cross-q coupling (the source is built from q-moments, so
∂dA_q/∂A_q' is genuinely q-dense), no cross-mode, no collision→temperature→Hubble coupling.

For an implicit solver the **growth direction is precisely the destabilizing one**. A Jacobian
that captures damping and discards growth lets the controller take confident over-large steps
into the amplifying mode — which is the observed h-collapse-then-blow-up. So the answer is not
just "add a block Jacobian"; it is "add the **sign-complete** local relaxation, including the
q-band coupling, and treat it implicitly." A full finite-difference source-response Jacobian is
**not affordable** (≈182 columns × ~0.1 s/source-build × 342 builds → hours), which is why the
fix must be a cheap structured/directional approximation, not a dense FD Jacobian.

---

## Q4 — Is the fixed three-mode diagonal non-LRS representation adequate for staging?

**Defensible as a staging endpoint, but only with two conditions, and I agree it should not
be replaced by a generic ell/m ladder yet.**

The live full-BBN path uses fixed `A_modes` of shape (species, 3, N_q) — three diagonal
S2-projected modes — while generic `AngularDecompositionSpec`/ell utilities live elsewhere and
the LRS collisionless ell ladder converges at ell_max=4 on a short span. The non-collision
S2-grid pair already shows `Sigma_H` is **not angular-grid converged** (Δ≈0.0122 between
(3,5) and (5,7)) even though abundances barely move. That is an existing anisotropic-transport
gap independent of collisions.

Conditions for the three-mode path to be a legitimate intermediate target:
1. **Relabel it.** "Full non-LRS ell convergence" must not be inferred from this path. Name it
   what it is: a *fixed diagonal three-mode S2-projection* endpoint. This is a one-line honesty
   fix and the overview's Finding 4 already concedes it.
2. **Add a cheap closure-residual diagnostic**, not a new ladder: project the nodal collision
   source onto modes {0,1,2} and report the discarded angular residual norm. If dynamic
   collision pumps significant power into ℓ>2 / off-diagonal structure, the three-mode
   projection is silently dropping it and the collision-on result would be uninterpretable even
   if it converged numerically. This tells you whether the representation is self-consistent
   under collision *before* you invest in a real ell/m hierarchy.

Building the generic dynamic ell/m ladder now would be premature — it adds large surface area
before the fixed-mode path even reaches the endpoint. Defer it (agreeing with the overview),
but gate the deferral on the residual diagnostic above so the deferral is evidence-based rather
than hopeful.

---

## Q5 — Is this a Python-native performance limit?

**No.** The cost decomposition from the failing row's own telemetry:

| Bucket | Measured | Share of 131.6 s |
|---|---|---|
| Dynamic collision payload build | 33.81 s (344 builds) | ~26% |
| Cache prewarm | 1.74 s | ~1% |
| Full-JVP dense Jacobian (342 builds × ~182 RHS-evals ≈ 62k RHS evals) | remainder, dominant | ~60–70% (inferred) |
| Host LU factorizations (546) + solves (4368) on a ~182×182 matrix | sub-second flops | small |
| Phase-2 corrector (13,605 Newton iters on 9×9 systems) | small flops; Python-loop overhead | small |

Two firm conclusions:
- **The payload build is not the bottleneck** (26%), and it is *not* growing pathologically —
  `stage_collision_payload_reuse_total = 3822` shows the cache works (payload reused across
  stages). The BD185 cache bounds did their job.
- **The dominant cost is the Jacobian *policy*, not the language.** `full_jvp` builds a dense
  ~182-column Jacobian roughly once per step. That is O(N) RHS-evaluations per step *and* it is
  the policy that omits the collision response. The same change that fixes the physics (Q3) also
  removes most of the cost: a structured/sparse Jacobian exploiting that transport is banded in q
  (the D_q stencil) and local in mode would cut the per-step Jacobian work by an order of magnitude.

So: **do not rewrite in another language now.** No compiled kernel is justified yet, because the
expensive thing (full dense JVP) is about to be replaced by a structured Jacobian, and the
collision contractions are 26% and already cached. Re-profile *after* the Jacobian/source-treatment
change; only then, if the radial contraction or a specific NumPy reduction is shown to dominate a
*stable* run, move that single kernel (Numba/JAX-static), not the orchestration.

---

## Q6 — What is overengineered?

The self-critique (Findings 1–4: two giant modules, artifact schema leaking into runtime, too
many policy axes, overloaded "ell" language) is **correct and confirmed by independent evidence.**
The git log is the clearest exhibit. Of the ~40 most recent commits (BD145→BD185):

- Substantive solver/physics: ≈5 (BD146 split A-mode JVP block, BD159 high-q derivative repair,
  BD160/BD167 collision-diag pilot — itself the incomplete damping-only one, BD183 shear boundary).
- Everything else is **telemetry / attribution / caching / flushing / metadata / probe-classification**:
  BD150–158, BD161–166, BD168–169, BD172–176, BD182, BD184–185.

That is ~6:1 evidence-plumbing to physics. The phase-2 corrector is the most striking instance:
both the *successful* isotropic run and the *failing* collision run perform **110k–124k AB2-Newton
initial-guess attempts** with ~7,300 displacement-guard rejections each — nearly identical counts.
A predictor/guard subsystem that fires ~230 times per accepted step, on both success and failure
paths, is doing far more bookkeeping than the 9×9 network needs.

Concrete recommendations (ordered):
1. **Freeze all non-essential policy axes** for the next blocker PR. Vary exactly one of:
   {collision Jacobian structure, collision source scale}. Every other knob (h ladders, retry
   factors, metadata modes, background policies, q-grid policies) is frozen until a collision-on
   row reaches the endpoint.
2. **Extract the dynamic collision source/Jacobian into its own module** with a small array-only
   API (in: A, Σ, T, grid; out: dA_modes, dQ, and now a relaxation block). `rhs.py` at 16.8k lines
   and the ladder at 9.4k lines should lose the collision payload construction and the artifact
   summarization, in that order.
3. **Separate runtime payloads from audit payloads.** Runtime carries arrays + compact counters;
   the artifact writer expands summaries on request. The ~140 `selected_*` fields per row do not
   belong on hot objects.
4. **Collapse the phase-2 AB2 initial-guess guard machinery** to one predictor + one guard, or
   profile and justify it. Default to a single compact metadata mode for long rows.
5. **Relabel "VALIDATED" / "physical_full_bbn_span_ready"** — see the standalone flag below; this
   is the highest-leverage honesty fix and costs almost nothing.

What **not** to simplify (agreeing with the packet): keep the raw negative-candidate diagnostics,
keep the split phase-2 corrector out of the Rodas stages, keep the freedom-attribution matrix until
collision rows complete.

---

## Standalone physics flag (not in the question list, but material)

The "strict isotropic full-BBN" run that the overview marks **VALIDATED** reports
**`Yp = 0.16413`**. The canonical BBN helium mass fraction is ≈ 0.247; this is ~33% low. (D/H =
2.12e-5 is, by contrast, near the observed ≈ 2.5e-5.) A low Yp with a near-physical D/H points to
n/p being too low at nucleosynthesis (Yp = 2x/(1+x) ⇒ x = n/p ≈ 0.089 here, vs the canonical
≈ 0.14).

This is most likely a **known consequence of the diagnostic starting at T_γ0 = 0.8 MeV** (weak
freeze-out is right at the start, so n/p is essentially imposed by the initial state rather than
evolved from T≫1 MeV), not necessarily a network bug. But two things follow:
- The word **"VALIDATED" is overclaiming.** This run validates that the *solver completes the
  integration span to T<0.01 MeV*. It does **not** validate the *physics* — the primary observable
  is 33% off canonical. Relabel as a solver-span milestone (e.g., `span_endpoint_reached`), reserving
  validation vocabulary for runs whose abundances are checked against a reference BBN code.
- **Confirm** that 0.164 is the expected output of the 0.8 MeV truncated start (run one isotropic
  case from T≳3 MeV, or against PRIMAT/standard initial n/p, and check Yp→~0.247). If it is not,
  there is a network/initial-condition issue that is independent of, and more fundamental than, the
  collision blocker.

This is exactly the "overinterpret readiness language" risk the packet already worries about; here
it is realized.

---

## Q7 — The exact next PR

**Title: Semi-implicit, sign-complete collision relaxation in the host operator (+ occupation
bound on the moment-feeding distribution).**

This is one focused change that moves the runtime blocker, tests Hypothesis A directly, and is
falsifiable. It does *not* add a readiness/manifest/hash/figure gate, does not touch QKE, keeps
Rodas5P/CPU-JAX, and does not truncate abundances.

**What it does**
1. Replace the damping-only diagonal (`_frozen_source_collision_dA_diagonal_jacobian`) with a
   **sign-complete local relaxation block** `L` in (species, mode, q):
   - Obtain `L` from **one directional difference of the collision source** per Jacobian build:
     `L·v ≈ [S(A + ε v) − S(A_frozen)] / ε` along the current Newton/stage direction (one extra
     payload build per step — affordable, since payload build is 26% and one extra build ≈ +0.1 s/step),
     OR analytically from the local q-band of the radial source if cheaper.
   - **Do not restrict to `A·dA < 0`.** Keep both signs. Stabilize only by capping the *positive*
     (growth) eigenvalue contribution at the controller level, not by zeroing it.
   - Add it **implicitly**: assemble `W = I/h − (J_transport+stress + L)` so the relaxation is in
     the linear solve (IMEX/Rosenbrock-with-source treatment), instead of the source being purely
     explicit.
2. Add an **occupation bound on the distribution that feeds the energy moments and the source
   build** — `f_recon` clamped to [0,1] (this is the fermion occupation number, *not* an abundance
   or Y_p, so it does not violate the no-truncation constraint). The prototype `evolve_nonlinear`
   already does this; the live RHS does not. Apply it only to the reconstructed `f` used for
   ρ/Π/source, and record the clamp-event count as raw evidence (clamps firing = the instability
   the relaxation block must absorb).

**Files touched**
- `src/rabbit/validation/augmented_continuous_ap65_rhs.py` — generalize the diagonal Jacobian to
  the sign-complete relaxation block; new policy `frozen_source_jax_full_jvp_collision_relax`; wire
  `L` into `W`.
- `src/rabbit/jax/augmented_typeI_replay.py` — expose a cheap single-direction collision source
  response (`_dynamic_collision_source_directional_response`) reusing the existing payload path; add
  the [0,1] clamp on `f_recon` in `_live_source_rhs_vector` (lines ~1815, ~1869–1886) with a counter.
- (No new module is required for the PR, but this is the natural seam to begin the collision-source
  extraction recommended in Q6.)

**Tests / artifacts that would falsify it**
- Rerun `q14_collision_N4p8` with `collision_relax` under a **collision_source_scale ladder
  {0.25, 0.5, 1.0}**, all other axes frozen. Falsification criteria:
  - *Hypothesis A confirmed* if reachable N (or T_final) **increases monotonically** with the
    relaxation block on, and the row reaches a lower T than N≈3.9 / dies later.
  - *Hypothesis A falsified* if the row still dies at N≈3.9 with `final_state_nonfinite` and the
    `f_recon` clamp counter is ~0 — then the problem is the q-representation (non-conservative
    transport), and the next PR is a conservative/SBP q-operator, not the Jacobian.
  - The clamp counter discriminates the two: heavy clamping + later death ⇒ amplification path is
    real and the relaxation block is helping; no clamping + same death ⇒ the source itself is
    producing nonfinite output on a still-physical state (a normalization bug after all, contra Q2).
- Add a unit test on `L`: for a manufactured FD-perturbed `A`, check `L·v` matches a brute-force
  finite-difference of `S` to a few digits and has the correct sign on both a damping and a growth
  perturbation (the current code would fail the growth case).

**Why this and not the alternatives**
- A dense FD source-response Jacobian is correct but unaffordable (Q3/Q5).
- A pure cost fix (compiled kernel) is premature — the dominant cost is the soon-to-be-replaced
  dense JVP, and the failure is a blow-up, not a timeout (Q5).
- Another component-attribution probe is the path of least resistance the project has already taken
  ~30 times (Q6); the directional-`L` + clamp change *is* the experiment, and it leaves a working
  solver behind rather than another evidence wrapper.

---

## One-line summary

The collision row blows up (not times out) because the implicit solver treats a stiff collision
source explicitly while a non-conservative, 1e6-amplifying high-q path is left unbounded; the source
algebra itself is correct, the dominant cost is the dense frozen-JVP Jacobian (not Python, not the
payload), and the decisive next step is a **sign-complete semi-implicit collision relaxation block
plus a fermion-occupation bound on the moment-feeding distribution** — with a collision-scale ladder
and a clamp counter that will cleanly confirm or falsify the Jacobian hypothesis.