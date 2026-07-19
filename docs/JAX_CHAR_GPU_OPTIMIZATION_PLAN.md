# JAX Characteristic-Ray Driver: GPU Optimization Plan

**Scope:** `rabbit.jax.driver_typeI_char` — JAX-native characteristic-ray
Bianchi I BBN driver, LRS, tier-1, CL0–CL3.
**Solver constraint:** Rodas5P (Rosenbrock-Wanner) is kept as the
integrator because its stiffness handling at weak-freeze-out is
materially better than diffrax's current stiff tableaux. The optimization
plan therefore must work **within** the existing
`rabbit.jax.solver_jax_rodas5p` architecture — no diffrax swap.

**Status today (2026-04-21):**

- 13-DOF tier-1 state / 15-DOF tier-2 state after compact analytic
  characteristic transport
- `runtime_device_policy="cpu_preferred"` default → 0 GB VRAM on the
  canonical scalar path
- PR-G delivered `run_char_batch_tier1(...)` and
  `run_char_batch_tier2(...)` on top of an event-masked pure-JAX
  Rodas5P core
- Publication parity remains locked by the scalar driver:
  \|ΔY_p\| ≤ 4 × 10⁻⁸ at tier-1 and \|ΔY_p\| ≤ 7 × 10⁻⁸ at tier-2
- Local warm tier-1 batch throughput on RX 6950 XT:
  - CPU: `48.23 ms/solve` at `N=1`, `10.48 ms/solve` at `N=64`,
    `11.83 ms/solve` at `N=256`
  - GPU: `1156 ms/solve` at `N=1`, `19.12 ms/solve` at `N=64`,
    `9.91 ms/solve` at `N=128`, `7.44 ms/solve` at `N=256`

For single solves and ≤ 100-point scans, the CPU path is strictly
faster than any GPU path. The question answered here is: **what would
need to change to make GPU execution actually worth it, without giving
up Rodas5P?**

---

## 1. Why the GPU is a net loss today

1. **State is too small.** 13 DOF per solve at tier-1 (15 at tier-2).
   Each XLA kernel launch on
   GPU costs 10–50 μs; the actual compute per RHS is a few μs. With
   ~500 Rodas5P steps × 8 stages + 1 Jacobian = ~4 500 kernel launches
   per solve, launch overhead dominates.
2. **Rodas5P is sequential.** Each step depends on the previous
   accepted state, so step-level parallelism is impossible. The only
   parallelism available is across independent solves.
3. **JAX's default GPU memory preallocation** (`XLA_PYTHON_CLIENT_MEM_FRACTION=0.75`)
   reserves multi-GB VRAM at import, regardless of need. The actual
   working set for one compact characteristic solve is far smaller.
4. **Dense 13 × 13 / 15 × 15 LU per step** fits easily in L1/L2 on CPU; on GPU it
   launches a cuBLAS/rocBLAS kernel per factorisation, which is
   overhead-bound at that size.

Consequence: **the only regime where GPU beats CPU is large-batch
inference.**

---

## 2. Optimizations that are Rosenbrock-compatible

### 2.1 VRAM preallocation (zero-risk, zero physics impact)

Set **before** `import jax`:

```bash
export XLA_PYTHON_CLIENT_PREALLOCATE=false
export XLA_PYTHON_CLIENT_ALLOCATOR=platform
# or, for a fractional cap
export XLA_PYTHON_CLIENT_MEM_FRACTION=0.10
```

This alone turns "12 GB permanently reserved" into "≤ 500 MB on demand"
for the 25-DOF characteristic driver. No code change, no numerical
change, no interaction with Rodas5P.

### 2.2 vmap batching on top of Rodas5P (delivered in PR-G)

PR-G added:
- `_solve_core_event_masked(...)` in `solver_jax_rodas5p.py`
- `run_char_batch_tier1(...)` and `run_char_batch_tier2(...)` in
  `driver_typeI_char.py`

The batched solve keeps Rodas5P and uses a single `lax.while_loop`
with per-element finish masks.  Finished lanes freeze their state while
the rest of the batch continues until every active lane has triggered
its event or exhausted `max_steps`.

**Observed tradeoff:**
- The "slowest-element tax" is real: the batch runs until the slowest
  lane finishes.
- Even so, launch amortisation eventually beats CPU throughput.
- On the local RX 6950 XT audit rig, GPU does **not** break even at
  `N=64`; the crossover appears around `N≈128`.

**Measured warm tier-1 throughput (lower is better):**

| Batch size N | CPU ms/solve | GPU ms/solve | Winner |
|---|---|---|---|
| 1 | 48.23 | 1156.18 | CPU |
| 8 | 21.46 | 141.22 | CPU |
| 32 | 11.89 | 37.44 | CPU |
| 64 | 10.48 | 19.12 | CPU |
| 128 | 12.91 | 9.91 | GPU |
| 256 | 11.83 | 7.44 | GPU |

**Operational guidance:**
- Keep CPU as the default for scalar and small-batch solves.
- Use GPU only through the explicit batch helpers, with the VRAM env
  vars from §2.1 set before `import jax`.
- Treat `N≈128` as the local breakeven estimate for this compact state
  and this GPU class; do not hard-code the old `N=64` expectation into
  tests or docs.

### 2.3 Analytic elimination of J_j (delivered in PR-A)

PR-A already removed `J_j` from the JAX characteristic state. One
subtle point mattered: the OCR text around paper eq (51) reads like the
inverse Jacobian `dμ_{j,0}/dμ_j`, but the production transport weight
carried by the code is the **forward** Jacobian `J_j = dμ_j/dμ_{j,0}`,
as required by the ODE actually used in the solver,
`dJ_j/dN = 3Σ₊(1-3μ_j²)J_j` (paper eq 55). Differentiating the analytic
forward map therefore gives

$$
J_j(N) = \frac{d\mu_j}{d\mu_{j,0}}
       = \frac{\mu_j(N)\,[1-\mu_j(N)^2]}
              {\mu_{j,0}\,[1-\mu_{j,0}^2]},
$$

with absolute values used in code so the pushed-forward quadrature
weights stay positive on the sign-preserving `μ<0` and `μ>0` branches.

**Effect at N_μ = 12, phase 2:**
- State DOF: 37 → 25 (**32 % reduction**) at tier-1
- Jacobian columns (via `jacfwd`): 37 → 25
- Dense LU cost per step: 37³ → 25³ ≈ **3.3 × cheaper**
- Per-step memory bandwidth similarly reduced

**Cost:** `J_j` is recomputed at every RHS evaluation from
`(X0, S, μ_j(S))`. This is negligible relative to the weak-rate
Laguerre integrals.

**Risk:** near `μ_j = ±1`, `1-μ_j² → 0`. The Gauss–Legendre grid never
places a node at the pole, and the implementation additionally floors
the denominator at `1e-30`.

**Status vs SciPy parity:** SciPy still evolves `J_j` as an ODE state.
The two drivers therefore diverge at the state-layout level but remain
observationally identical to roadmap tolerance because the JAX
reconstruction is the exact closed-form solution of the carried ODE.

### 2.4 Analytic Jacobian in place of jacfwd

Rodas5P **requires a Jacobian at every accepted step** — that is
roughly `N_steps` Jacobian evaluations per solve. Currently we use
`jax.jacfwd(rhs)`, which costs `dim(state)` forward RHS evaluations
per Jacobian. At 25 DOF that is 25 extra RHS evaluations per step.

The characteristic RHS has enough analytic structure that the
Jacobian can be assembled in closed form, block by block:

| Block | Derivative | Complexity |
|---|---|---|
| `∂(dΣ₊)/∂Σ₊` | `2 Σ₊ · Σ₊ − (1 − Σ²)` from `q = 1+Σ²` | trivial |
| `∂(dΣ₊)/∂I_j` | `−8 f_ν w_j J_j P₂(μ_j) e^{-8 I_j}` | closed form |
| `∂(dΣ₊)/∂S` | `f_ν Σ_j w_j (∂J/∂S · P₂ + J · P₂' · ∂μ/∂S) e^{-8I}` | chain rule |
| `∂(dI_j)/∂Σ₊` | `P₂(μ_j)` | trivial |
| `∂(dI_j)/∂S` | `Σ₊ · P₂'(μ_j) · ∂μ_j/∂S` | chain rule, `∂μ/∂S = 3 μ (1−μ²)` |
| `∂(dX)/∂I_j` | weak-rate chain rule through `f̃₀(q)` | moderate |

**Benefit:** per step, we drop from `~37 RHS evals` (jacfwd) to
`~1 RHS eval + closed-form algebra`. This is a **~25–30 × Jacobian
cost reduction**, and it compresses the XLA graph (fewer fused
operations per step), which is especially valuable on GPU where
launch overhead is the bottleneck.

**Cost:** ~400–600 LOC of careful derivative work, plus cross-checks
against `jacfwd` to lock the analytic formula.

**Risk:** any bug in the analytic derivative silently degrades
convergence, not correctness (since Rodas5P recovers from Jacobian
staleness via error control). Mitigation: comprehensive unit tests
that compare analytic vs `jacfwd` at 1e-10 elementwise tolerance.

### 2.5 Block-sparse Jacobian within Rodas5P (already implemented, currently off by default)

The solver core exposes `active_indices` → `_rodas5p_step_schur`.
That code assumes `J[passive, passive] = 0` strictly. In the
characteristic driver:

- `I_j` variables: `J[I_j, anything] = 0` except for
  `∂(dI_j)/∂Σ₊, ∂(dI_j)/∂S`. Strictly zero in the `(I, I)` block ✓
PR-A removed `J_j`, so the strict-passive block is now just `I_j` and
the tier-1 Schur partition is `13 active + 12 passive`.  The accepted
performance datum is still the pre-PR-A measurement: at 37 DOF, Schur
overhead (vmap + vjp passes, `W_ap @ W_pa` matrix products) **exceeded**
the savings from cutting `jacfwd` from 37 to 25 columns, giving
0.71 × speedup vs dense.

**On GPU the calculus may flip.** The extra matrix products are
cheap on GPU (one big GEMM), and the 13 × 13 LU now launches
materially fewer kernels than the dense 25 × 25 path. Benchmark before
deciding. Combined with §2.4 (analytic Jacobian), this remains the
likely endgame.

### 2.6 Loosen the weak-rate quadrature (precision-for-throughput knob)

Live-rate integrand uses 32-node Laguerre × 32-node Legendre per
channel × 6 channels = 6144 function evaluations per RHS call.
The 32/32 choice comes from the SciPy convention
(`weak.quadrature.DEFAULT_WEAK_QUADRATURE`) and carries several
orders of headroom versus the target `|ΔY_p| < 5 × 10⁻⁵` precision.

**Diagnostic:** paper §D.4 reports convergence of `Y_p` to `<1e-4`
at N_Laguerre = 16. Dropping from 32 → 24 preserves publication
precision with ~25 % fewer function evaluations inside the RHS.

**Status:** not recommended as a default — the CPU cost is already
acceptable. Use only for GPU batch scans where the **per-RHS** cost
is bandwidth-bound.

### 2.7 Reject mixed precision for BBN

`dY_p/dη ~ 10⁸`, so the baryon-to-photon ratio is accurate to about
`1e-8 × precision_level`. float32 roundoff (~ 1e-7 per op) would
inject ~ 1e-5 noise into `Y_p`, *above* the 5 × 10⁻⁵ publication
tolerance. Keep the full state at `float64`.

The ancillary weak-rate tables can in principle run float32 since
the chain-rule amplification is weaker there, but the combined
speedup is small and the parity cost is large.

---

## 3. Recommended order of implementation (if GPU path is prioritised)

1. **Phase 0 — zero-cost** (5 min, 0 LOC)
   Document and set the `XLA_*` preallocation environment variables.
   Effect: VRAM from 12 GB → ≤ 500 MB per process.

2. **Phase 1 — analytic J_j elimination** (**delivered in PR-A**)
   `J_j` has been removed from the state vector and reconstructed
   analytically inside the RHS / observable path. The remaining open
   work starts at batch execution and analytic Jacobians.

3. **Phase 2 — vmap batched solve** (**delivered in PR-G**)
   `_solve_core_event_masked(...)` now provides the event-aware batch
   core, and `run_char_batch_tier1(...)` / `run_char_batch_tier2(...)`
   expose the public LRS batch surface.  The local breakeven table is
   now measured rather than estimated: GPU is still slower at `N=64`
   but faster by `N=128`, with clear payoff at `N=256`.

4. **Phase 3 — analytic Jacobian** (est. 500 LOC + tests)
   Replace `jacfwd` with closed-form blocks. Keep `jacfwd` as a
   regression reference. Unit-test elementwise agreement to 1e-10.
   Effect: ~25 × Jacobian cost, biggest single-solve speedup.

5. **Phase 4 — re-benchmark block-sparse** on GPU with analytic
   Jacobian in place. If it wins, flip default `jacobian_mode`
   per-device.

Phases 1, 3, 4 also improve CPU performance. Phase 2 is the only
GPU-specific piece.

---

## 4. What **not** to do

- **Do not replace Rodas5P with a diffrax solver.** diffrax's
  `Kvaerno5`/`Tsit5` have weaker stiffness handling at the
  weak-freeze-out transition (PRIMAT AC2024 network mixed with the
  weak-rate timescale). Rodas5P with our stability bound is the
  correct tool. AD support is handled at the outer layer via
  `jax.custom_vjp` in `gradient_bridge.py`; we do not need AD
  through the inner solver for publication results.
- **Do not migrate the state to float32.** See §2.7.
- **Do not drop below N_Laguerre = 24** for any published result.
- **Do not invoke GPU for batch sizes < 128** on the compact
  characteristic surface unless you have hardware-specific evidence to
  the contrary.  CPU wins in the measured small-batch regime.
  sub-batch regime.

---

## 5. Quick-reference decision matrix

| Workload | Action |
|---|---|
| Single BBN solve | CPU (default). GPU is always worse. |
| 10–100 point scan | CPU with warm JIT cache (current code already does this). |
| 128–1024 point grid (inference) | GPU + §2.2 vmap + §2.1 VRAM env vars. |
| > 10⁴ solves (SMC / nested sampling) | GPU + §2.1 + §2.2 + §2.4. |
| Fine-tuning CPU performance | §2.3 + §2.4. No GPU touched. |

---

## 6. Provenance

- Rodas5P coefficients: Steinebach (2023), BIT 63:27.
- Characteristic ray formulas: RABBIT report `RABBIT_report_H1_revision_typo_pass2.pdf`
  §§2, 6, eq (41)–(58); appendix D.3.
- JAX VRAM defaults:
  <https://docs.jax.dev/en/latest/gpu_memory_allocation.html>
- J_j analytic form: production uses the forward measure
  `J_j = dμ_j/dμ_{j,0} = μ_j(1-μ_j²)/[μ_{j,0}(1-μ_{j,0}²)]`; the
  paper OCR around eq (51) is ambiguous and was resolved in the PR-A
  audit by matching the carried ODE (eq 55) and direct numerical
  integration.
- Weak rate quadrature convergence: paper §D.4.

**File under review:** `src/rabbit/jax/driver_typeI_char.py` (current
revision, CPU-preferred, stable RHS identity cache, dense Jacobian
default, block-sparse mode available).
