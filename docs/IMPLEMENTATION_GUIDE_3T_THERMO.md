# Implementation Guide — Tier-2 Three-Temperature Thermodynamics

**Scope:** Lift the JAX characteristic-ray Type I BBN driver from
tier-1 entropy-conservation thermodynamics (single helper T_ν) to the
tier-2 three-temperature system that independently evolves
(T_γ, T_νₑ, T_νₓ).  The SciPy characteristic driver already supports
this; the JAX characteristic driver does not.

**Physics foundation:** RABBIT report §11.3 (entropy conservation with
collision source term), §11.6 (N_eff in the tiered hierarchy),
§14 (QED equation of state, Bennett 𝒪(e³)).  Existing JAX primitives
in [`rabbit.jax.thermo_provider_jax`](../src/rabbit/jax/thermo_provider_jax.py)
already expose `Tier2ThermoProvider`, `hubble_3T_jax`,
`coupled_3T_rhs_jax`, `N_eff_from_3T_jax` — these kernels just need to
be wired into the characteristic driver.

---

## 1. Why tier-2 matters

Tier-1 (paper §11.2, Table 2) uses the entropy-conservation cooling
law `dT_γ/dN = -T_γ / (1 + T_γ S'(T_γ)/(3S(T_γ)))` and a helper
neutrino temperature `T_ν = T_γ (g_s(T_γ)/g_s(T_dec))^{1/3}` below
`T_dec ≈ 2 MeV`.  This reproduces the instantaneous-decoupling
baseline `N_eff = 3.000` exactly but cannot capture the two real
effects of incomplete decoupling (paper §11.1):

1. **Reheating** — `e⁺e⁻` annihilation dumps ~0.3 % of the plasma
   entropy into neutrinos *after* their nominal decoupling,
   shifting `N_eff` to the Standard Model benchmark 3.044.
2. **Flavour-dependent heating** — because `ν_e` remains coupled ~5×
   longer than `ν_μ, ν_τ` (the coupling ratio
   `a_e / a_x ≈ 4.68`, paper §8.2), the two populations decouple at
   different effective temperatures.

Tier-2 tracks T_γ, T_νₑ, and T_νₓ ≡ T_{ν_μ} = T_{ν_τ} as three
coupled ODEs with energy exchange terms supplied by the collision
sector.  Without collisions wired in, tier-2 still provides the
correct *architectural* layer — the temperatures evolve freely
(T_νₑ = T_νₓ = T_γ until decoupling, then free-stream) — but the
ν–e heating source is zero, so it reduces to an over-parameterised
tier-1.

The production value of tier-2 therefore requires **two** pieces:
the temperature ODEs *and* the collision source term.  The collision
side has its own implementation guide; this document focuses on
the thermodynamic layer and how to plumb a collision source term
once available.

---

## 2. Target physics (paper references)

### 2.1 Three-temperature evolution

Paper eq (161) gives the plasma side:

```
dT_γ/dN = -T_γ / (1 + T_γ S'/(3S)) + dQ_{ν→plasma}/dN / (∂ρ_plasma/∂T_γ)
```

where `S(T_γ)` is the photon–electron entropy density including QED
corrections (paper §14, Bennett 2020) and
`dQ_{ν→plasma}/dN > 0` when the plasma heats from neutrinos (rare;
usually neutrinos receive heat from the plasma during annihilation).

Paper eq (162) gives the matching neutrino equations:

```
dT_{νₑ}/dN = -T_{νₑ} - (dQ_{νₑ}/dN) / (∂ρ_{νₑ}/∂T_{νₑ})
dT_{νₓ}/dN = -T_{νₓ} - (dQ_{νₓ}/dN) / (∂ρ_{νₓ}/∂T_{νₓ})
```

with the convention that `dQ_{ν_α}/dN > 0` is heat received by the
α-neutrino sector.  Globally the collision integral conserves total
stress–energy (paper §8.4), so
`dQ_{γ→plasma} + Σ_α dQ_{ν_α→plasma} = 0` up to Hubble redshift.

### 2.2 Hubble rate

Paper eq (7) in the anisotropic form:

```
H² = 8πG_N (ρ_plasma(T_γ) + ρ_νₑ(T_νₑ) + 2 ρ_νₓ(T_νₓ)) / (3 (1 - Σ²))
ρ_νₐ(T) = (T⁴ / 2π²) · ∫₀^∞ y³ f_νₐ(y) dy
```

In tier-2, `ρ_νₐ` is computed from the **evolved** non-equilibrium
spectrum when the collision side delivers it (paper eq 90).  At
tier-2 *without* collisions, the spectrum is equilibrium Fermi–Dirac
at T_νₐ so `ρ_νₐ = (7/8) (π²/30) · 2 · T_νₐ⁴` (2 is the 2-helicity
factor per species).

### 2.3 N_eff from the three-temperature state

Paper eq (163) defines:

```
N_eff^{(3T)} = (T_νₑ / T_ν^{std})⁴ + 2 (T_νₓ / T_ν^{std})⁴
T_ν^{std}    = T_γ · (4/11)^{1/3}
```

In the fully-decoupled limit `(T_νₑ, T_νₓ) → (T_ν^{std}, T_ν^{std})`
and `N_eff → 3.000`.  Collision heating raises `T_νₑ` and `T_νₓ`
above that limit to give `N_eff ≈ 3.044`.

---

## 3. What already exists in the repo

The hard work is already done — it just hasn't been connected to
the characteristic driver:

| Primitive | Location | Role |
|---|---|---|
| `tier1_T_nu_from_T_gamma_jax` | `rabbit/jax/thermo_provider_jax.py` | Currently used helper |
| `tier1_hubble_aniso_jax` | " | Tier-1 Hubble |
| `tier1_dT_gamma_dN_jax` | " | Tier-1 photon cooling |
| `hubble_3T_jax` | `rabbit/jax/nudec_coupled_jax.py` | **Tier-2 Hubble** |
| `coupled_3T_rhs_jax` | " | **Tier-2 coupled dT/dN** |
| `N_eff_from_3T_jax` | " | **Tier-2 diagnostic** |
| `Tier2ThermoProvider` | `rabbit/jax/thermo_provider_jax.py` | Dataclass wrapper |

The linearised-PSTF JAX driver ([`driver_typeI.py`](../src/rabbit/jax/driver_typeI.py),
kernels `_coupled_rhs_phase{1,2}_{reduced,live}_tier2`) already uses
`hubble_3T_jax` and `coupled_3T_rhs_jax`.  Wire the same calls into
the characteristic driver.

---

## 4. Implementation plan

### Phase 1 — state-vector extension (~80 LOC)

Add two new slots to the characteristic layout:

```python
# rabbit/jax/driver_typeI_char.py
def _char_layout(N_mu: int, n_species: int, thermo_tier: int = 1) -> dict:
    i_I = _IDX_I_START
    i_J = i_I + N_mu
    i_S = i_J + N_mu
    i_tg = i_S + 1
    if thermo_tier >= 2:
        i_tne = i_tg + 1
        i_tnx = i_tg + 2
        i_net = i_tg + 3
    else:
        i_tne = -1
        i_tnx = -1
        i_net = i_tg + 1
    n_total = i_net + n_species
    return {..., "i_tne": i_tne, "i_tnx": i_tnx, ...}
```

State at N_μ=12, phase 2, tier 2, n_species=9:
`2 + 2·12 + 1 + 3 + 9 = 39 DOF` (up from 37 tier-1).

### Phase 2 — RHS rewrite (~120 LOC)

Replace the thermo block in `_rhs_core`:

```python
if thermo_tier == 1:
    T_nu = tier1_T_nu_from_T_gamma_jax(T_gamma)
    H_MeV = tier1_hubble_aniso_jax(T_gamma, T_nu, N_eff_param, Sigma_sq)
    dT_gamma = tier1_dT_gamma_dN_jax(T_gamma)
    dT_nu_e  = jnp.zeros(())   # not an independent variable
    dT_nu_x  = jnp.zeros(())
else:                           # thermo_tier == 2
    T_nu_e = y[layout["i_tne"]]
    T_nu_x = y[layout["i_tnx"]]
    H_MeV = hubble_3T_jax(T_gamma, T_nu_e, T_nu_x, Sigma_sq)
    dT_gamma, dT_nu_e, dT_nu_x = coupled_3T_rhs_jax(
        T_gamma, T_nu_e, T_nu_x, H_MeV=H_MeV
    )
    T_nu = T_nu_e   # what feeds the weak rates
```

The key physical subtlety: **the temperature that feeds the weak
rate monopole reconstruction is T_νₑ, not T_γ · (4/11)^{1/3}**.
Paper §13.3 uses `T_ν_for_rates = T_νₑ` when running tier 2 — this
is what makes the CL0/CL1/CL2 live rates sensitive to the heating
asymmetry.

### Phase 3 — initial conditions (~30 LOC)

At `T_start = 10 MeV` both neutrino populations are still strongly
coupled to the plasma, so
`T_νₑ(0) = T_νₓ(0) = T_γ(0) = T_start`.  Initial spectra are
Fermi–Dirac at `T_νₑ(0)` (reconstructed via `f̃₀ = f_eq`, since
`I_j = 0`).

### Phase 4 — collision source term (wiring only; collision kernel
covered in the separate guide)

The collision sector will eventually supply

```python
dQ_nue_dN, dQ_nux_dN = compute_collision_energy_exchange(
    T_gamma, T_nu_e, T_nu_x, f_mono_nue, f_mono_nux, q_nodes, q_weights
)
```

Feed these into `coupled_3T_rhs_jax` via an overloaded variant that
accepts external `dQ_α/dN` arguments (currently the function
hard-codes `dQ = 0` → that's why tier-2 collapses to tier-1 when
collisions are off).  Match the SciPy contract at
[`rabbit/drivers/full_coupled_typeI.py`](../src/rabbit/drivers/full_coupled_typeI.py)
lines 785–812.

### Phase 5 — Hubble enhancement and Σ² term

The anisotropic Hubble in `hubble_3T_jax` needs the same
`(1 - Σ²)^{-1/2}` factor used by `tier1_hubble_aniso_jax`.  Read the
current signature:

```python
# rabbit/jax/nudec_coupled_jax.py
def hubble_3T_jax(T_gamma, T_nu_e, T_nu_x, Sigma_sq=0.0):
    rho_pl = rho_plasma(T_gamma)
    rho_nu = rho_photon(T_nu_e) * (7.0/8.0) + 2 * rho_photon(T_nu_x) * (7.0/8.0)
    rho_total = rho_pl + rho_nu
    Omega = 1.0 - Sigma_sq
    H_sq = 8.0 * PI / 3.0 * G_N * rho_total / Omega
    return jnp.where((Omega > 0.0) & (H_sq > 0.0), jnp.sqrt(H_sq), jnp.nan)
```

This is already present.  **No change needed** beyond threading
`Sigma_sq = Σ_+² + Σ_-²` through the call site.

### Phase 6 — config + dispatch (~60 LOC)

- Add `thermo_tier: int = 1` to `JAXTypeICharConfig` (default
  preserves behaviour).
- Remove the `thermo_tier == 1 only` NotImplementedError in the
  `transport_mode="characteristic"` dispatch branch of
  `driver_typeI.py`.
- Metadata: extend `result.metadata["thermo_tier"]` to the
  reported value, mirroring the linearised-PSTF tier-2 keys
  (`thermo_provider_mode`, `temperatures_evolved`,
  `N_eff_method="3T_ratio"`).

### Phase 7 — validation (~200 LOC tests)

Four gates:

1. **Tier-2 reduces to tier-1 without collisions.**  With all
   `dQ_α = 0`, tier-2 evolves `T_νₑ = T_νₓ = T_γ (4/11)^{1/3}` after
   decoupling, identical to the tier-1 helper within 0.05 % (loose
   because the tier-1 step function for g_*s has kinks the smooth 3T
   system does not).
2. **SciPy parity with characteristic + tier-2 + collisions off.**
   `|ΔY_p| < 5e-7` vs the SciPy characteristic driver at
   `tier=2, enable_collisions=False`.
3. **N_eff at 3.000 without collisions.**  `|N_eff - 3.000| < 0.01`
   for the collision-free tier-2 solve.
4. **Monotonicity.**  Confirm that turning on a synthetic
   `dQ_α/dN > 0` source raises `N_eff` monotonically and
   `Y_p` decreases monotonically (positive collision heating
   populates high-q neutrinos, raising λ_{np}, lowering the n/p
   freeze-out ratio).

---

## 5. Test matrix

| Σ_+ | tier | collisions | Target |
|---|---|---|---|
| 0.0 | 1 | off | regression-lock (unchanged) |
| 0.0 | 2 | off | tier-2 vs tier-1 at FLRW, \|ΔY_p\| < 1e-6 |
| 0.1 | 2 | off | tier-2 anisotropic parity vs SciPy char tier 2 |
| 0.3 | 2 | off | large-shear tier-2 stress test |
| 0.0 | 2 | on | reheating sector (depends on collision guide) |

---

## 6. Risk register

- **Jacobian stiffness near decoupling.**  Around `T_γ ≈ 2 MeV` the
  coupling between T_γ and T_νₑ is strongest; Rodas5P's Gustafsson
  controller may want smaller steps.  Monitor rejection count; if
  elevated, tighten `atol` for the thermo slots in the weighted
  norm or adopt the SciPy driver's step schedule.
- **NaN propagation.**  `hubble_3T_jax` returns NaN if `Ω ≤ 0`.
  With two shear variables the Ω floor is easier to reach.  The JAX
  solver will terminate with `status=-1`; this must be detected and
  surfaced as a driver failure, not a silent NaN observable.
- **Thermo-weak tight coupling.**  The live weak monopole uses
  T_νₑ directly; a small T_νₑ error leaks quadratically into λ_np
  via the `ε_ν = q T_νₑ / m_e` transformation (paper eq 164).
  Regression-lock λ_np at handoff to the SciPy characteristic
  tier-2 reference.

---

## 7. LOC budget and timeline

| Phase | Description | LOC | Days |
|---|---|---|---|
| 1 | State-vector extension | 80 | 0.5 |
| 2 | Tier-2 RHS wiring | 120 | 0.5 |
| 3 | Initial conditions | 30 | 0.25 |
| 4 | Collision hooks (wiring only) | 60 | 0.25 |
| 5 | Hubble/Σ² threading | — | 0 (already present) |
| 6 | Config + dispatch | 60 | 0.5 |
| 7 | Validation tests | 200 | 1.5 |
| **Total** | | **≈ 550** | **≈ 3.5** |

The tier-2 upgrade is the smallest of the three extension guides —
most of the physics is already implemented, only the
characteristic-driver integration is missing.

---

## 8. What this does NOT deliver

- **Actual collision heating.**  Without the collision kernel wired
  in (see the dedicated guide), tier-2 tracks the three
  temperatures but they remain on the collisionless trajectory.
  `N_eff` lands at ~3.000, not 3.044.
- **Anisotropic neutrino temperature.**  The 3T system still treats
  the neutrino sector as three species each with *one* temperature.
  Anisotropic energy redistribution (captured by per-ray I_j) lives
  in the transport sector, not the thermo sector.
- **Tier-3 diagonal ν–ν scattering.**  Requires the full collision
  kernel including the ν–ν channel — see the dedicated guide.
- **QED EoS upgrade.**  Tier-2 inherits the Bennett 𝒪(e³) plasma
  from `rho_plasma` / `entropy_density_plasma`.  No change here.
