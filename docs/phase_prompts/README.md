# Phase-Prompts Index and Prompt-Engineering Framework

> **DEPRECATED execution surface (PUB-00, 2026-07-12).**  These prompts are
> historical and contain stale solver, capability, dependency, and
> publication-grade assertions.  Do not feed them to an implementation agent.
> Current work starts from
> `docs/TYPEI_AUGMENTED_NOQKE_UNIFIED_FUTURE_PLAN.md`.

This directory preserves one self-contained prompt document per PR in the old
roadmap ([docs/ROADMAP_PR_WBS.md](../ROADMAP_PR_WBS.md)).  The prompts document
the earlier execution framework and are retained for provenance only.

Historically this README was the shared preamble for every phase prompt.  The
phase documents preserve that context for audit purposes, but it is not current
implementation guidance.

## Phase-prompt inventory

| Phase | Document | Depends on |
|---|---|---|
| PR-A | [PR-A_analytic_J.md](PR-A_analytic_J.md) | — |
| PR-J | [PR-J_analytic_jacobian.md](PR-J_analytic_jacobian.md) | PR-A |
| PR-N1 | [PR-N1_nonlrs_primitives.md](PR-N1_nonlrs_primitives.md) | — |
| PR-N2 | [PR-N2_nonlrs_driver.md](PR-N2_nonlrs_driver.md) | PR-N1 |
| PR-D | [PR-D_auto_backend_flip.md](PR-D_auto_backend_flip.md) | — |
| PR-G | [PR-G_gpu_vmap_batch.md](PR-G_gpu_vmap_batch.md) | PR-A, PR-J recommended |
| PR-T3A | [PR-T3A_q_advection.md](PR-T3A_q_advection.md) | PR-A, PR-J |
| PR-T3B | [PR-T3B_nu_e_pair.md](PR-T3B_nu_e_pair.md) | PR-T3A |
| PR-T3C | [PR-T3C_nu_nu.md](PR-T3C_nu_nu.md) | PR-T3B |
| PR-T3D | [PR-T3D_tier3_integration.md](PR-T3D_tier3_integration.md) | PR-T3A/B/C |
| PR-R | [PR-R_release_gate.md](PR-R_release_gate.md) | all of the above |

The dependency graph is reproduced in
[../ROADMAP_PR_WBS.md §0](../ROADMAP_PR_WBS.md#0-dependency-graph).

---

## 1.  Load-bearing project context (copy into every phase prompt)

The block below is reproduced verbatim at the top of every phase
document so that a fresh session has a consistent anchor.

### 1.1 What RABBIT is

RABBIT (Rosenbrock Adaptive BBN with Bianchi-type Integration Toolkit)
computes Big Bang Nucleosynthesis observables on an **anisotropic
Bianchi Type I background**.  The physics reference is the repository
file `RABBIT_report_H1_revision_typo_pass2.pdf`.  Primary observable:
⁴He mass fraction `Y_p`; secondary: `D/H`, `N_eff`, `⁷Li`, `⁶Li`.
Anisotropy is parametrised by Hubble-normalised shear
`(Σ_+, Σ_-)`; LRS means `Σ_- = 0`.

### 1.2 Solver choice is load-bearing

The production integrator is **Rodas5P** (Steinebach 2023, 8-stage
order-5(4) Rosenbrock–Wanner, paper eq 129).  Chosen because stiffness
handling at weak-freeze-out (`Γ_ν/H ~ 1` transition) is materially
better than `diffrax`'s explicit / ESDIRK stiff tableaux.
**No PR may propose replacing Rodas5P with diffrax for stiffness or
AD reasons.**  AD is provided at the outer layer via
`jax.custom_vjp` in `src/rabbit/jax/gradient_bridge.py` (finite-
difference fallback when event-triggered `while_loop` AD is
unavailable).

### 1.3 Three transport methods

| Method | State DOF | Fidelity |
|---|---|---|
| Linearised PSTF (ℓ=2) | 6·n_ell·N_q ≈ 240 | Reference model.  Misses 27–36 %% nonlinear characteristic correction (paper §6.8). |
| Characteristic-ray (Paper I) | 2·N_μ + 1 = 25 (post-PR-A) / 37 (pre-PR-A) | **Publication-grade** exact collisionless solution. |
| Full phase-space ray (Paper II) | N_μ·N_q ≈ 240 | Paper-II scope; tier-3 with full collisions. **Not yet implemented.** |

### 1.4 Tier hierarchy for incomplete decoupling

| Tier | Thermo | Collisions |
|---|---|---|
| 1 | Single helper `T_ν` from plasma entropy conservation | None |
| 2 | 3T coupled `(T_γ, T_νₑ, T_νₓ)` | Mangano momentum-averaged ν–e energy transfer |
| 3 | 3T + per-momentum collisions | Full Boltzmann: ν–e elastic + pair + diagonal ν–ν |
| 4 | Tier-3 + flavour oscillations (QKE) | Out of scope for this roadmap |

### 1.5 Baseline parity (matched-physics SciPy ↔ JAX characteristic)

- Tier-1: |ΔY_p| ≤ 4 × 10⁻⁸ across Σ_H ∈ [0, 0.5], CL0–CL2
- Tier-2: |ΔY_p| ≤ 7 × 10⁻⁸ across the same grid
- Publication tolerance: 5 × 10⁻⁵ (3–4 orders of headroom)

### 1.6 Key invariants — do not violate

- **Rodas5P stays.**  Adaptive stiff solver with 8-stage error
  control, Gustafsson controller, `lax.while_loop` event detection.
- **float64 state vectors.**  `dY_p/dη ~ 10⁸`; float32 noise floor
  of 10⁻⁵ would exceed publication tolerance.  Weak-rate tables may
  stay float32 only with an explicit audit trail.
- **CPU-preferred default.**  37-DOF state is kernel-launch-bound.
  GPU is opt-in via `runtime_device_policy="gpu_then_cpu_retry"`
  plus `XLA_PYTHON_CLIENT_MEM_FRACTION=0.10`.
- **Stable-identity RHS cache.**  Rodas5P's solver-runner cache keys
  on `id(rhs_fn)`; rebuild the closure per call and you force a
  full re-trace.  The fix is a host-side cache keyed by
  `(phase, CL, tier, shape, physical-params)`.
- **Transported monopole in weak rates.**  CL0–CL3 live weak rates
  read `f̃₀(q) = ½ Σ_j w_j J_j f_FD(q e^{2 I_j})` (paper eq 58),
  *not* the equilibrium Fermi–Dirac.

### 1.7 Key files (read-only anchors during every PR)

- `src/rabbit/jax/driver_typeI_char.py` — JAX char driver (tier-1 + tier-2)
- `src/rabbit/jax/driver_typeI.py` — dispatch hub + linearised-PSTF driver
- `src/rabbit/jax/solver_jax_rodas5p.py` — Rodas5P + block-sparse Schur
- `src/rabbit/jax/characteristic_rays_jax.py` — ray map, stress, monopole
- `src/rabbit/jax/nudec_coupled_jax.py` — `hubble_3T_jax`, `coupled_3T_rhs_jax`, `N_eff_from_3T_jax`
- `src/rabbit/jax/thermo_provider_jax.py` — tier-1 thermo
- `src/rabbit/jax/weak_live_jax.py` — CL0–CL3 live weak kernels
- `src/rabbit/jax/network_jax.py` — PRIMAT AC2024 network
- `src/rabbit/drivers/full_coupled_typeI.py` — SciPy reference (tier-2 fall-through fixed)
- `src/rabbit/config/backend_capabilities.py` — capability registry
- `src/rabbit/inference/forward_likelihood.py` — `canonical_forward_solver` dispatch

---

## 2.  Three-stage verification protocol (run at every phase)

Every phase prompt instructs the agent to perform **three separate
verification passes** before committing.  The three passes are
independent: a failure in any one blocks the commit.

### Stage 1 — Internal repo literature verification

- Grep the repo for every claimed function/file reference.
- Read the paper PDF for every cited equation number.
- Read the prior ROADMAP docs for every claimed decision.
- Produce a verdict line per claim:
  ```
  claim: f_nu = 0.40520      → VERIFIED at src/rabbit/jax/driver_typeI.py:94
  claim: paper eq 58 is f̃₀  → VERIFIED (PDF pages 17-18)
  claim: PR-T3T used Mangano → VERIFIED at nudec_coupled_jax.py:69-88
  ```

### Stage 2 — External literature / web verification

- Look up external references cited (PRIMAT, LASAGNA, Mangano 2005,
  Froustey 2020, Escudero 2019, Steinebach 2023).
- Cross-check numerical values (weak rates, `N_eff`, decay constants).
- Search for known implementation pitfalls in the specific kernel
  being coded (e.g. "Hannestad-Madsen angular integration limits").
- Record URLs + excerpted key lines.

### Stage 3 — Self chain-of-thought verification

- Derive the result from first principles, independent of the
  skeleton code.
- Perform dimensional analysis: every term in the RHS has units of
  `[state] / efold`, equivalent to `[state] · Hubble / H`.
- Check physical limits:
  - `Σ → 0` reduces to the FLRW / tier-1 baseline.
  - `T_γ → ∞` all rates are in thermal equilibrium.
  - `T_γ → 0` network freezes, no more reactions.
- Adversarially probe for:
  - Off-by-one in state indices
  - NaN paths (1 - Σ² → 0, T_γ → 0, division by H)
  - Sign errors (easiest to catch: compare sign of one term to the
    SciPy reference for the same quantity)
  - Unit mismatches (MeV vs 1/s, comoving vs physical q)
  - Species / flavour mix-ups

---

## 3.  Core principles applied to every phase

### 3.1 No hallucinated physics
Every formula must be traceable to a paper equation number or to an
already-verified internal source file.  If a formula is needed that
is *not* in the paper, the prompt must either (a) forbid making it
up and require the agent to refuse / defer, or (b) explicitly point
to an external reference to be cited.

### 3.2 No unlabelled mocks
If an approximation is introduced for pragmatic reasons, it must be
labelled in code (`_teff_bridge_candidate`,
`production_authority="raw_..."`, etc.) **and** in
`ROADMAP_STATE_OF_RECORD.md §3`.

### 3.3 Rodas5P stays
Repeated because it is the single most common local minimum that
tempts a speed-optimiser: "just switch to diffrax for AD / GPU."
Any prompt that slides in that direction must be rejected.

### 3.4 Publication-grade parity is the acceptance gate
`|ΔY_p| < 5 × 10⁻⁵` vs the SciPy reference driver at matched
physics, measured on at least `Σ_H ∈ {0, 0.1, 0.3, 0.5} × CL ∈ {0, 1, 2}`.

### 3.5 Document the decision, not just the outcome
`STATE_OF_RECORD.md §3` records decisions *and their rationale*
inline; any new decision must be added with the rationale, not
merely the action.

---

## 4.  Prompt-engineering patterns used across phase prompts

### 4.1 Anti-local-minimum anchoring

Every phase prompt concludes with an "anti-local-minimum" reminder:

> Before committing:
> 1. Re-read §1.6 of this prompt (the invariants).  If any invariant
>    is threatened, stop and ask the user.
> 2. Compare your solution against the **alternative approach** listed
>    in this phase's §[phase-specific number].  Did you actually pick
>    the better one, or did you default to the first approach that
>    worked?
> 3. Quote one specific line of the paper or the SciPy source that
>    your implementation **directly** mirrors.  If you cannot, the
>    implementation lacks provenance and must be revisited.

### 4.2 Adversarial auditor handoff

After Stage-3 verification, every phase prompt invokes an
adversarial auditor via `Agent(subagent_type="general-purpose")`
whose role is to audit the diff for hallucinations / mocks / off-by-
one.  The auditor's verdict is committed as
`docs/audit/PR-<ID>.md`.

### 4.3 Checkpoint-driven parity

Parity tests run **after each WBS step**, not only at the end.
Intermediate green/red signals prevent accumulating errors into
late-stage commits.

### 4.4 Explicit failure-mode enumeration

Before starting the implementation, the prompt has the agent
enumerate "what can go wrong" in Stage-3 CoT.  This surfaces NaN
paths, boundary conditions, and sign errors *before* they manifest
in test failures.

### 4.5 Doc-update-before-code pattern

The prompt directs the agent to update `STATE_OF_RECORD.md` /
`PR_CATALOG.md` *in the same PR* (not afterward).  This prevents
silent documentation drift.

### 4.6 Deterministic commit script

Each phase prompt ends with a verbatim shell block that:
1. Runs the parity suite
2. Runs the full test sweep
3. Updates the documentation per `ROADMAP_SELF_AUDIT.md §3.2-§3.4`
4. Creates the git commit with a standardised message

The script is designed to be run by the agent without further
prompting.

---

## 5.  Hallucination-prevention protocol

Concrete "what not to do" list reproduced in every phase:

- **Do not invent** a function name without first `grep`-ing the
  repo.
- **Do not cite** a paper equation number without reading the
  surrounding page from the PDF (via `Read` with `pages` argument).
- **Do not mock** a collision operator, monopole extractor, or
  solver kernel "temporarily to unblock progress".  If the real
  implementation is missing, pause and re-scope.
- **Do not claim parity** without running the actual test.  Parity
  numbers in the PR catalogue are **measured**, not estimated.
- **Do not modify** `src/rabbit/drivers/full_coupled_typeI.py` or
  other SciPy reference code unless the phase is explicitly
  SciPy-targeted (PR-S* in the catalogue convention).  The JAX
  side is where all physics extensions belong.

---

## 6.  How the three verification stages produce a phase log

At the end of each phase, the concatenation of Stage-1, Stage-2, and
Stage-3 outputs (plus the adversarial auditor verdict and the parity
numbers) becomes the PR's verification log, stored as
`docs/audit/PR-<ID>.md`.  Combined with the `PR_CATALOG.md` entry,
this forms the **permanent record** of why the PR is believed
correct.

A reviewer who opens `PR_CATALOG.md`, clicks through to
`audit/PR-<ID>.md`, and reads the three stages should be able to
reconstruct the physics provenance of every formula, every numerical
value, and every design decision in the PR.

---

## 7.  Local-minimum checklist

Common traps that each phase prompt is engineered to avoid:

1. **"First working approach" trap.**  Fix: the prompt always asks
   the agent to *enumerate* alternatives before coding.
2. **"Parity at the end" trap.**  Fix: the WBS interleaves parity
   tests into every step.
3. **"Diffrax would be easier" trap.**  Fix: §1.2 + §3.3 of every
   prompt explicitly forbid this.
4. **"GPU now" trap.**  Fix: CPU-preferred is the invariant until
   batch size ≥ 64 (PR-G).
5. **"Skip the adversarial review" trap.**  Fix: the commit script
   refuses to commit unless the auditor verdict file exists.
6. **"Doc update next PR" trap.**  Fix: the commit script requires
   `STATE_OF_RECORD.md` and `PR_CATALOG.md` to be staged.

Each phase prompt carries these reminders inline so that a fresh
agent without context is still protected.

---

## 8.  Feedback-loop into the roadmap

If a phase reveals that the roadmap needs adjustment (e.g. a
dependency was missed, a decision was wrong), the phase prompt has
one final step that writes a **roadmap-amendment note** to
`docs/audit/PR-<ID>.md#roadmap-amendments`.  Subsequent phases are
then expected to read that note before starting.

---

## 9.  Starting a phase

To execute any phase:
1. Open the phase prompt, e.g. `PR-A_analytic_J.md`.
2. Feed it verbatim to a fresh Claude session (or use it as the body
   of a slash-command invocation).
3. Let the agent execute the three verification stages, the WBS,
   the adversarial handoff, and the commit script.
4. Review the commit + `PR_CATALOG.md` append + `audit/PR-<ID>.md`
   together.

If a verification stage fails, the phase prompt directs the agent to
stop and report — it does **not** fall back on plausible guesses.
