# BD205 — q-Laguerre dynamic-collision blocker: root cause + fix + clean-core contract

**Date:** 2026-07-01
**Continues:** BD199–BD204 (PROJECT_STATE.md). Those runs narrowed the blocker to the
dynamic q-Laguerre collision/3T energy path (q4/q5 FLRW collision-on fails before the
BBN endpoint: Rodas5P h-min at N=3→4, `T_νx/T_γ~0.0012`, `N_eff_3T~1.15`, spurious shear;
`dQ_nux_bank<0` wrong-signed/too-large; dQ_only✓ / dA_only✓ / dQ+dA coupled ✗).
**Status:** root cause pinned with executable evidence; fix direction validated; clean-core
acceptance contract defined (`tests/test_qlaguerre_collision_conditioning.py`).

---

## 1. Root cause — an ill-conditioned VARIABLE, not a physics bug

The heavy-bank temperature source is
```
dT_νx/dN = −T_νx + dQ_nux_bank_N / d2 ,   d2 = 2·(dρ_ν_pair/dT) ∝ T_νx³
```
(`thermo/nudec_coupled.py:223-224`; JAX twin `jax/nudec_coupled_jax.py:224-227`).

As the heavy bank nears decoupling, `T_νx → 0`, so `d2 → 0` and the source becomes
`O(dQ / T_νx³)`. **Any** error in `dQ_nux_bank_N` — from any origin — is amplified by
`~1/T_νx³` into a catastrophic temperature kick. Measured (fixed `dQ = −1e-6`):

| T_νx [MeV] | dT_νx/dN |
|-----------|----------|
| 1.0       | −1.0     |
| 1e-2      | −0.23    |
| 3.76e-5 (BD203) | **−4.1e+06** |

The `d2 > 1e-50` guard is a **no-op** here — it floors the denominator far below where the
blow-up already occurs. This is ill-conditioning of the *variable choice*, not a floor bug
and not a first-principles `N_eff` limitation.

## 2. The dQ moment is accurate — so amplification, not quadrature, is the fault

The q-Laguerre energy moment uses plain weights `w_i·exp(q_i)`
(`transport/augmented_collision_bridge.py:4388-4400`) (file deleted in PR-D1..D3; citation is historical). Measured: for smooth sources (FD, and a
slower-decaying `exp(−q/2)` distortion) the plain-weight moment matches the analytic integral to
**~machine precision** on the N_q=80 grid. Gauss-Laguerre resolves smooth distributions fine.

So `dQ_nux_bank` noise from a smooth A-distortion is small; it is the **1/T_νx³ amplifier** that
makes it fatal. (This refutes the "slower-decaying tail breaks the exp(q) cancellation" hypothesis
for smooth distortions — that specific mechanism does not fire. Residual dQ error from a
truncated/oscillatory A-mode reconstruction may still exist, but it is *amplified* noise, not the
root cause.) BD204's dQ_only✓/dA_only✓/coupled✗ is consistent: only the coupled path drives `T_νx`
small enough for the amplifier to bite.

## 3. The fix — evolve the heavy-bank ENERGY, not the temperature

Evolve `ρ_νx` with
```
dρ_νx/dN = −4·ρ_νx + dQ_nux_bank_N
```
and recover `T_νx ∝ ρ_νx^{1/4}` diagnostically. There is no division by `c_v ∝ T_νx³`, so a small
`dQ` error produces a **bounded** `dρ/dN` regardless of how small `T_νx` is. Measured (same
`dQ = −1e-6`): `dρ_νx/dN = −1e-6` at `T_νx = 3.76e-5` (vs `dT_νx/dN = −4.1e+06` for the temperature
form) — a **~1e12 conditioning improvement**. Do NOT "fix" this by only flooring `1/T³` — that
hides the source error and keeps the variable ill-conditioned.

## 4. Clean-core acceptance contract (Phase 2)

`src/rabbit/collisions/dynamic_collision_core.py` must satisfy, and is gated by
`tests/test_qlaguerre_collision_conditioning.py` (anchored on surviving modules so it outlives the
AP65 deletion) plus its own tests:

1. **Well-conditioned heavy bank:** evolve `ρ_νx` (or `ρ^{1/4}`/log), never `T_νx` divided by
   `c_v ∝ T³`. Locked by `test_energy_variable_evolution_is_well_conditioned`.
2. **Accurate + positivity-guarded dQ moment** (quadrature-consistency check retained as secondary
   hardening).
3. **FLRW isotropy invariant:** an exactly-FLRW q-Laguerre payload gives `Π₊=Π₋=0` within quadrature
   tolerance (delivered with the clean core, where the angular payload is controlled).
4. **Acceptance:** the q4/q5 FLRW collision-on case that failed at N=3→4 reaches the BBN endpoint
   with finite Yp/DH, physical `N_eff_3T`, no heavy-bank collapse.

---

*Anti-drift footer: measured driver/function outputs; no external anchor exists for Bianchi-I BBN
with a neutrino collision term (no such paper), so this is an internal conditioning/consistency
contract, not external validation.*
