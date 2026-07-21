# BD622 W6 — Independent comparator design proof-pack (B3) for blind review

Status: **DESIGN ONLY**. No implementation, no collision/comparator/GL/Radau/trajectory/endpoint
execution. This is the prospectively-frozen contract a blind reviewer must accept (or reject)
*before* any implementation, satisfying the D-029 reopen condition and both BD622 reports' P3/Patch-3.

Primary method: **B3 — fermionic entropy-variable (logit-renormalized) Galerkin** (Report 2's
primary; Report 1's fallback family agrees). Alternative for adjudication at review: **PCED**
(Report 1's primary, positive conservative event/deposition). This pack argues B3; §9 states the
one PCED caveat the reviewer must weigh.

## 1. Purpose and boundary

Provide a **structurally independent** static comparator for the classical, flavour-diagonal,
zero-lepton, flat-FLRW, no-QKE full-spectral neutrino collision operator, whose every gated
identity (conservation, entropy, Pauli bound, μτ covariance, Jacobian regularity) is a
**pre-output theorem** rather than a monitored residual. It must (a) not reintroduce the D-027
non-adjoint defect, (b) not route any observable through the D-028 native `1/y²` inversion (which
W5 proved is the recurrence engine), and (c) not use the D-029 target-dependent selector.

## 2. State and ansatz

For each folded neutrino pair-flavour `a ∈ {e, μ, τ}` (ν+ν̄ at zero lepton asymmetry) evolve the
**entropy-variable modal coefficients** `β^a ∈ R^M`:

```
η_a(y,t) = Σ_{m<M} β_m^a(t) ψ_m(y),     span{ψ_0,…,ψ_{M-1}} ⊇ {1, y}
f_a(y,t) = σ(η_a) = 1 / (1 + e^{−η_a})            (occupation = logistic of the entropy variable)
```

Basis `{ψ_m}`: polynomials on the momentum grid orthonormalized under the fixed FD-adapted inner
product `⟨u,v⟩_w = Σ_q w_q ŵ(y_q) u v`, `ŵ ≈ y² f_FD(1−f_FD)` at a reference epoch, with **`{1, y}`
in the span** (load-bearing for conservation). The electron/positron legs are a fixed FD bath at
`T_γ`, entropy variable `η_{e±} = −E/T_γ`, `E = √(y² + (m_e/T)²)`.

## 3. Semidiscrete method (CSWF Galerkin)

**Conservative symmetrized weak form (CSWF).** Every sampled reaction tuple
`(y₁, y₂, y₃)` with `y₄ = y₁+y₂−y₃` (energy conservation; e± legs analytic FD) contributes to a
test moment `φ`, from **one shared tuple evaluation**:

```
q[φ]  +=  W · Λ · [ φ(y₁) + φ(y₂) − φ(y₃) − φ(y₄) ]
```

- `W = (quadrature weight) × (tagged kernel)` with the tagged coefficient + `K_s/K_t/K_u`
  assignment taken from the **W3-verified catalogue** (`docs/audit/BD622_W3_per_row_mb_closed_form_oracles.md`;
  {64,128,32}×{K_s,K_t} re-derived from spin traces + the MB closed forms, severing F-5).
- `Λ` = Uehling–Uhlenbeck bracket `f₃f₄(1−f₁)(1−f₂) − f₁f₂(1−f₃)(1−f₄)`, evaluated in
  cancellation-stable logit form (`expm1`-based, sign-exact at zero affinity).
- The FLRW plasma (`T_γ`) / `z`-equation receives the **same tuples with opposite sign** (the
  electromagnetic energy debit).

**Galerkin projection in the FD-entropy pairing:**

```
M_a(β) β̇^a = q^a ,
M_{a,im}(β) = Σ_q w_q y_q² f_a(1−f_a) ψ_i(y_q) ψ_m(y_q)      (SPD; σ' = f(1−f) ∈ (0, ¼])
q_i^a       = q[ψ_i]   for flavour a  (CSWF above)
```

Radau consumes the mass-matrix form `M(β) β̇ = q` natively.

## 4. Proof obligations discharged as theorems

**T-1 (Conservation — structural, exact arithmetic).** With `{1, y} ⊆ span{ψ}`:
- `φ = 1`: each tuple contributes `W·Λ·(1+1−1−1) = 0` → number conserved **per tuple**.
- `φ = y`: `W·Λ·(y₁+y₂−y₃−y₄) = 0` since `y₃+y₄ = y₁+y₂` by construction → energy conserved
  **per tuple**. The number/energy moments are the `q`-components along the `{1,y}` directions of
  the test space and vanish identically per reaction class. Binary64: number telescopes exactly;
  energy is exact up to the `y₄ = total−y₃` shell rounding (report field, not a gate). ∎

**T-2 (Discrete H-theorem — entropy sign, structural).** With `S_h = −Σ_q w_q y_q²[f ln f +
(1−f)ln(1−f)]`, `dS_h/df = −η`. Writing `Λ = P·(e^{η₃+η₄} − e^{η₁+η₂})` with `P > 0` a
Pauli/measure factor, the per-tuple production is

```
Ṡ|_tuple = W · P · (η₃+η₄ − η₁−η₂)(e^{η₃+η₄} − e^{η₁+η₂}) ≥ 0
```

by the elementary identity `(x−y)(eˣ−eʸ) ≥ 0`. Sign-definite **per quadrature tuple**, including
the e± legs at `η_{e±} = −E/T_γ` (monotone total ν+plasma entropy). FD equilibria are affine
`η ∈ span{ψ}` ⇒ exact discrete fixed points. ∎

**T-3 (Pauli bound — unconditional).** `f_a = σ(η_a) ∈ (0,1)` for every finite `β^a`. The feasible
set is all of `R^{2M}`: no realizability boundary, no support selector, no one-hot branch. This
**retires the D-029 failure mode by construction**. ∎

**T-4 (μτ covariance — bitwise involution).** The tagged kernels/catalogue are **flavour-blind**
(coefficients depend on the reaction *class*, not the flavour label). Swapping μ↔τ is exactly the
exchange of the `β^μ`, `β^τ` blocks — a bitwise involution of the assembly. Hence if `β^μ = β^τ`
then `q^μ = q^τ` and `M^μ = M^τ` **bitwise**, and the μτ covariance residual is **exactly 0**, not
a `1/y²`-amplified roundoff. Every physical observable (`N_eff`, distortions, `z`) is a direct
quadrature functional of `σ(η)` — **no native mass inversion anywhere** ⇒ the D-028 recurrence
engine (W5-confirmed) is structurally absent. ∎

**T-5 (Jacobian regularity).** `M(β)` is SPD (weight `w y² f(1−f) ∈ (0, ¼]`), so `M β̇ = q` is a
well-posed index-1 system. `σ ∈ C^∞`, `Λ ∈ C^∞` in the logits ⇒ `q(β)`, `M(β) ∈ C^∞` and the
Jacobian `∂(M⁻¹q)/∂β` exists and is smooth everywhere — no branches or selectors to straddle. ∎

**T-6 (No non-adjoint defect).** Deposition and evaluation are the same CSWF tuple sum against the
`{ψ_i}` test space; the interpolation/deposition pair is the exact `⟨·,·⟩_w` adjoint (the property
D-027 lacked). Constants and the energy coordinate lie in the test space, so the conserved moments
are exact by construction, not quadrature-limited. ∎

## 5. Structural-independence statement

| axis | Rust production | B3 comparator | class |
|---|---|---|---|
| derivation/geometry | folded event stream, bracket deposition | CSWF weak form, entropy-variable Galerkin | **independent** |
| unknowns | nodal logits (2n) | entropy-variable modal coefficients β (3·M) | **independent** |
| catalogue coefficients | frozen {64,128,32} | same values, but **independently re-derived** (W3 spin traces + MB oracles) | **severed** (F-5) |
| output metrology | frozen caps | weak/measure-weighted (G2a), native only as diagnostic (G2b) | **independent of the 1/y² trap** |
| integrator | diffsol BDF / Rodas5P | SciPy Radau on `M(β)β̇=q` | **independent** |

No translation path from the Rust algorithm to B3 exists (different unknowns, geometry, and
conservation/positivity/entropy mechanisms). Shared, and declared: universal constants, the
published matrix-element *values* (now severed by W3's independent derivation), the event
definition, and the checkpoint set.

## 6. Falsifier battery (frozen caps; static-pass = all green)

- **T01** explicit-species reaction/multiplicity oracle — the **W3 MB closed forms** (rows i–v)
  and spin-trace ratios; cap 1e-11 (smooth-integrand). Failure ⇒ reopen coefficient class.
- **T02** common-FD nulls at `T∈{10,5,3,1,0.1}` with the Rust electron branch disabled (W4);
  H-normalized ≤1e-10, floor ~1e-13.
- **T03/T04** resolved self-scattering / electron-exchange non-eq states (incl. the non-FD
  equal-μτ shape that activates any latent orientation asymmetry).
- **T05** exact weak moments + FLRW first law; qν, qEM accumulated independently; first-law ≤1e-8.
- **T06** entropy sign — per-tuple `Ṡ ≥ −64ε·scale` (T-2 is structural, so any failure is a real
  bug).
- **T07** CP/family/μτ under the **W1 dual co-gate `G-F10-COVARIANCE-METROLOGY`**: G2a weak/modal
  ≤1e-10 (binding, floor ~1e-13); G2b native diagnostic ≤ C·ε·G_alg (~1.2e-8 at GL48) with
  argmax/subgrid/BLAS-provenance reporting. (T-4 makes G2a structurally exact when `β^μ=β^τ`.)
- **T08** support-edge / exact-node / tie / tail; declared support convention (F-9).
- **T09** arithmetic ladder (naive/pairwise/compensated/high-precision) — cross-checks against the
  W7 MPFR oracle.
- **T10** native/weak/physically-weighted norm triple, denominators frozen by a blinded agent.
- **T11** directional Jacobian across the whole domain (T-5: no branches, so this is a smoothness
  confirmation, cap `10·(ε·κ_row)^{2/3}`).
- **T12** base/reference quadrature + tail convergence, enclosed by the W7 oracle.

## 7. Abandonment criteria (pre-frozen)

Halt B3 and fall back (to PCED, or to the W7 oracle as sole arbiter) if any of:
- `κ(M) > 1e6` along a representative trajectory (the FD-entropy weight `f(1−f)` degenerates in the
  deep tail where `f→0`);
- modal resolution `M > 24` needed to meet the T03/T05 caps (intractable);
- the T-2 per-tuple entropy sign is compromised for the finite-`m_e` electron legs (the one place
  the clean `(x−y)(eˣ−eʸ)` argument must be re-checked with `η_{e±}=−E/T_γ`).

## 8. What the blind reviewer must check (contract)

1. `{1,y} ⊆ span{ψ}` is enforced by construction (T-1 hinges on it).
2. `Λ = P·(e^{η₃+η₄}−e^{η₁+η₂})`, `P>0`, holds for **every** reaction class incl. the finite-`m_e`
   electron legs (T-2). This is the load-bearing algebra; reject if it fails for any leg.
3. The catalogue `W` matches the W3 oracle values *and* ratios (no silent re-transcription).
4. No observable in T01–T12 routes through `A = modal/(prefactor·y²)` (the D-028 trap).
5. The `M(β)` conditioning stays below the §7 bound on the declared frozen state family.
6. The independence table (§5) has no hidden shared reduction/selector.

## 9. PCED alternative — the one caveat to adjudicate

Report 1's PCED (nodal logits + all-leg local hat deposition + local positive mass) also avoids
the `1/y²` inversion and gets the Pauli bound from the logistic. Report 2's objection: the e± legs
have `E=√(y²+(m_e/T)²)`, not grid-closed, so a discrete-velocity/event method must hybridize with
deposition for the electron sector anyway — reintroducing the adjointness burden B3 avoids by being
weak-form from the start. **Reviewer decision:** if PCED's electron-leg hat deposition is shown to
preserve the T-1/T-2/T-6 theorems as cleanly as B3, PCED is admissible (smaller surface); otherwise
B3 is primary. Either way the **W7 MPFR/interval oracle is mandatory** as the arithmetic arbiter.

## 10. Deliverable sequence (post-approval)

On blind-review **pass**: implement B3 static M0/M1 (new private module + one test module,
≈700–1100 lines), keep `_independent_noqke.py` byte-frozen as D-028 evidence, and delete the dead
Fort exporter (`isotropic_boltzmann.rs:1531-3922`, −2,392 lines; also clears the pre-existing
`gk_recurse` clippy failure noted in W4) → net-negative. Then W7 (MPFR oracle), then the T01–T12
battery to the static-pass gate. Segment/endpoint remain separately owner-gated.
