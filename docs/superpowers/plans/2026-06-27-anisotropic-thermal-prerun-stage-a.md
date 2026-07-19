# Anisotropic Thermal Prerun — Stage A Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Create the anisotropic thermal prerun module and its public function, with the isotropic limit (σ=0, A=0) reproducing the existing FLRW prerun neutrino temperatures exactly, deferring shear/PSTF integration to Stage B.

**Architecture:** A new focused module `src/rabbit/validation/anisotropic_thermal_prerun.py` exposes `anisotropic_thermal_prerun_to_T0(...)`. In the isotropic limit it computes neutrino temperatures with the same engine the FLRW prerun uses (`thermo.nudec_coupled.asymptotic_N_eff_3T_payload`), guaranteeing bit-for-bit FLRW parity, and returns an extended payload (decoupled T_ν, zero effective shear, zero A modes) shaped for later restart-state consumption. Nonzero shear or A-offset raises `NotImplementedError` (lands in Stage B/C).

**Tech Stack:** Python 3.12, NumPy, the in-tree `rabbit.thermo.nudec_coupled` 3T closure. CPU.

## Global Constraints

- QKE out of scope; no public-production / publication / SMC claim.
- The existing FLRW prerun (`_phase1_thermo_prerun_to_T0`) and all current defaults are unchanged by this stage.
- Isotropic-limit (σ=0, A=0) neutrino temperatures from the new function must equal the FLRW prerun's to emitted precision (same engine, same call).
- Preserve raw decoupling evidence in the returned payload; no clipping.
- Reuse the existing 3T engine; do not duplicate decoupling physics.
- Small reviewable commit; report the cost line.
- Run tests with `PYTHONPATH=src JAX_PLATFORMS=cpu venv/bin/python -m pytest`.

---

### Task 1: Anisotropic prerun module — isotropic-limit foundation

**Files:**
- Create: `src/rabbit/validation/anisotropic_thermal_prerun.py`
- Test: `tests/test_anisotropic_thermal_prerun.py`

**Interfaces:**
- Consumes: `rabbit.thermo.nudec_coupled.asymptotic_N_eff_3T_payload(T_gamma, T_nu_e, T_nu_x, *, T_stop_MeV)` (returns a dict with `available`, `tail_reached_stop`, `T_gamma_asymptotic_MeV`, `T_nu_e_asymptotic_MeV`, `T_nu_x_asymptotic_MeV`); `rabbit.thermo.nudec_coupled.N_eff_from_3T(T_gamma, T_nu_e, T_nu_x)`.
- Produces: `anisotropic_thermal_prerun_to_T0(*, T_start_MeV, T_end_MeV, n_species, q_count, sigma_plus0=0.0, sigma_minus0=0.0, initial_A_monopole_offset=0.0) -> dict`. Payload keys: `available` (bool), `policy` (`"phase1_thermo_prerun_anisotropic"`), `policy_scope` (`"anisotropic_isotropic_limit"` in this stage), `T_phase1_start_MeV`, `T_phase1_end_MeV`, `T_gamma0_effective_MeV`, `T_nu_e0_effective_MeV`, `T_nu_x0_effective_MeV`, `Sigma_plus0_effective` (float), `Sigma_minus0_effective` (float), `initial_A_modes` (nested list, shape `(n_species, 3, q_count)`), `N_eff_3T_at_T0` (float), `thermo_tail_payload` (dict). Later stages (C) consume `T_nu_e0_effective_MeV`, `T_nu_x0_effective_MeV`, `Sigma_plus0_effective`, `Sigma_minus0_effective`, `initial_A_modes` for the phase-2 restart state.

- [ ] **Step 1: Write the failing tests**

```python
# tests/test_anisotropic_thermal_prerun.py
import numpy as np
import pytest

from rabbit.validation.anisotropic_thermal_prerun import (
    anisotropic_thermal_prerun_to_T0,
)
from rabbit.validation.augmented_continuous_ap65_rhs import (
    _phase1_thermo_prerun_to_T0,
)


def test_isotropic_limit_matches_flrw_prerun_neutrino_temperatures():
    flrw = _phase1_thermo_prerun_to_T0(T_start_MeV=3.0, T_end_MeV=1.0)
    aniso = anisotropic_thermal_prerun_to_T0(
        T_start_MeV=3.0, T_end_MeV=1.0, n_species=9, q_count=4
    )
    assert aniso["available"] is True
    assert aniso["policy"] == "phase1_thermo_prerun_anisotropic"
    assert aniso["T_nu_e0_effective_MeV"] == flrw["T_nu_e0_effective_MeV"]
    assert aniso["T_nu_x0_effective_MeV"] == flrw["T_nu_x0_effective_MeV"]
    assert aniso["T_gamma0_effective_MeV"] == flrw["T_gamma0_effective_MeV"]


def test_isotropic_limit_returns_zero_shear_and_zero_A_modes():
    aniso = anisotropic_thermal_prerun_to_T0(
        T_start_MeV=3.0, T_end_MeV=1.0, n_species=9, q_count=4
    )
    assert aniso["Sigma_plus0_effective"] == 0.0
    assert aniso["Sigma_minus0_effective"] == 0.0
    modes = np.asarray(aniso["initial_A_modes"], dtype=float)
    assert modes.shape == (9, 3, 4)
    assert np.all(modes == 0.0)


def test_nonzero_shear_raises_not_implemented_until_stage_b():
    with pytest.raises(NotImplementedError):
        anisotropic_thermal_prerun_to_T0(
            T_start_MeV=3.0,
            T_end_MeV=1.0,
            n_species=9,
            q_count=4,
            sigma_plus0=1.0e-3,
        )


def test_nonzero_A_offset_raises_not_implemented_until_stage_b():
    with pytest.raises(NotImplementedError):
        anisotropic_thermal_prerun_to_T0(
            T_start_MeV=3.0,
            T_end_MeV=1.0,
            n_species=9,
            q_count=4,
            initial_A_monopole_offset=1.0e-5,
        )


def test_invalid_temperature_ordering_raises():
    with pytest.raises(ValueError):
        anisotropic_thermal_prerun_to_T0(
            T_start_MeV=1.0, T_end_MeV=3.0, n_species=9, q_count=4
        )
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `PYTHONPATH=src JAX_PLATFORMS=cpu venv/bin/python -m pytest tests/test_anisotropic_thermal_prerun.py -q`
Expected: FAIL — `ModuleNotFoundError: No module named 'rabbit.validation.anisotropic_thermal_prerun'`.

- [ ] **Step 3: Write the module**

```python
# src/rabbit/validation/anisotropic_thermal_prerun.py
"""Anisotropic neutrino-decoupling thermal prerun.

Stage A: isotropic-limit foundation. For zero shear and zero A-monopole
offset this reproduces the FLRW 3T-closure prerun neutrino temperatures
exactly (same engine, same call) and returns an extended payload shaped for
the phase-2 restart state. Nonzero shear or A-offset is deferred to Stage B
(full quadrupole-coupled PSTF integration) and raises NotImplementedError.
"""
from __future__ import annotations

from typing import Any

import numpy as np

from rabbit.thermo.nudec_coupled import N_eff_from_3T, asymptotic_N_eff_3T_payload

_SHEAR_TOL = 1.0e-15


def _finite_float(value: Any, *, name: str) -> float:
    out = float(value)
    if not np.isfinite(out):
        raise ValueError(f"{name} must be finite; got {value!r}.")
    return out


def anisotropic_thermal_prerun_to_T0(
    *,
    T_start_MeV: float,
    T_end_MeV: float,
    n_species: int,
    q_count: int,
    sigma_plus0: float = 0.0,
    sigma_minus0: float = 0.0,
    initial_A_monopole_offset: float = 0.0,
) -> dict[str, Any]:
    """Integrate the neutrino-decoupling thermal prerun to the AP65 start.

    Stage A handles only the isotropic limit; nonzero shear or A-offset is
    deferred to Stage B.
    """
    T_start = _finite_float(T_start_MeV, name="T_phase1_start_MeV")
    T_end = _finite_float(T_end_MeV, name="T_phase1_end_MeV")
    if T_start <= 0.0 or T_end <= 0.0:
        raise ValueError("prerun temperatures must be positive.")
    if T_start < T_end:
        raise ValueError("T_phase1_start_MeV must be >= T_phase1_end_MeV.")
    n_species_int = int(n_species)
    q_count_int = int(q_count)
    if n_species_int <= 0 or q_count_int <= 0:
        raise ValueError("n_species and q_count must be positive.")

    sigma_plus = _finite_float(sigma_plus0, name="sigma_plus0")
    sigma_minus = _finite_float(sigma_minus0, name="sigma_minus0")
    a_offset = _finite_float(
        initial_A_monopole_offset, name="initial_A_monopole_offset"
    )
    isotropic = (
        abs(sigma_plus) <= _SHEAR_TOL
        and abs(sigma_minus) <= _SHEAR_TOL
        and abs(a_offset) <= _SHEAR_TOL
    )
    if not isotropic:
        raise NotImplementedError(
            "anisotropic (nonzero shear or A-monopole offset) thermal prerun "
            "integration is implemented in Stage B; Stage A covers the "
            "isotropic limit only."
        )

    prerun = asymptotic_N_eff_3T_payload(
        T_start, T_start, T_start, T_stop_MeV=T_end
    )
    if not bool(prerun.get("available")) or not bool(
        prerun.get("tail_reached_stop")
    ):
        raise RuntimeError(
            "anisotropic thermo prerun did not reach T_gamma0: "
            + str(prerun.get("unavailable_reason") or prerun.get("solver_message"))
        )
    Tg = _finite_float(
        prerun["T_gamma_asymptotic_MeV"], name="prerun.T_gamma_at_T0"
    )
    Te = _finite_float(
        prerun["T_nu_e_asymptotic_MeV"], name="prerun.T_nu_e_at_T0"
    )
    Tx = _finite_float(
        prerun["T_nu_x_asymptotic_MeV"], name="prerun.T_nu_x_at_T0"
    )
    A0 = np.zeros((n_species_int, 3, q_count_int), dtype=float)
    return {
        "available": True,
        "policy": "phase1_thermo_prerun_anisotropic",
        "policy_scope": "anisotropic_isotropic_limit",
        "T_phase1_start_MeV": float(T_start),
        "T_phase1_end_MeV": float(T_end),
        "T_gamma0_effective_MeV": float(Tg),
        "T_nu_e0_effective_MeV": float(Te),
        "T_nu_x0_effective_MeV": float(Tx),
        "T_nu_e0_over_T_gamma0": float(Te / Tg),
        "T_nu_x0_over_T_gamma0": float(Tx / Tg),
        "Sigma_plus0_effective": 0.0,
        "Sigma_minus0_effective": 0.0,
        "initial_A_modes": A0.tolist(),
        "N_eff_3T_at_T0": float(N_eff_from_3T(Tg, Te, Tx)),
        "thermo_tail_payload": dict(prerun),
    }
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `PYTHONPATH=src JAX_PLATFORMS=cpu venv/bin/python -m pytest tests/test_anisotropic_thermal_prerun.py -q`
Expected: PASS (5 tests).

- [ ] **Step 5: Confirm the FLRW prerun is untouched**

Run: `git diff --stat src/rabbit/validation/augmented_continuous_ap65_rhs.py`
Expected: no output (the FLRW prerun file is unmodified; this stage adds a new module only).

- [ ] **Step 6: Commit**

```bash
git add src/rabbit/validation/anisotropic_thermal_prerun.py tests/test_anisotropic_thermal_prerun.py
git commit -m "anisotropic prerun Stage A: isotropic-limit foundation + FLRW parity"
```

---

## Self-Review

- **Spec coverage:** Stage A of the spec (prerun driver skeleton + FLRW-limit parity) is covered by Task 1. Stages B (shear+PSTF), C (wiring/guard removal), D (endpoint validation) are explicitly deferred to subsequent plans and are not in scope here.
- **Placeholder scan:** none — Task 1 contains full module and test code, exact commands, and expected output.
- **Type consistency:** the function signature and payload keys in the Interfaces block match the module code and the tests (`T_nu_e0_effective_MeV`, `Sigma_plus0_effective`, `initial_A_modes` shape `(n_species, 3, q_count)`).

## Next stages (separate plans, after Stage A validates)

- **Stage B:** map the phase-2 transport API (`augmented_typeI_replay`, `augmented_collision_bridge`, `augmented_nonlrs_transport`, `solver_jax_rodas5p`) and integrate the collision-coupled neutrino PSTF hierarchy + shear from 3 MeV to T_γ0; replace the isotropic-limit body. Gate: σ→0 continuity to Stage A; shear-generated quadrupole; collision closure.
- **Stage C:** register the `phase1_thermo_prerun_anisotropic` policy, dispatch from `_default_restart_kwargs`, thread the prerun output into the restart state, remove the FLRW shear guard. Gate: σ=0 endpoint bit-identical to BD591.
- **Stage D:** redo BD596 live-shear via the new prerun; compare.
