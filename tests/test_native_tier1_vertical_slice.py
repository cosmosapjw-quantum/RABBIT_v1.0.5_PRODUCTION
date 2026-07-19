"""R-01 falsifiers for the private Rust Tier-1 LRS vertical slice.

These tests deliberately do not use ``importorskip``.  Native mode is an
explicit request: an unavailable extension, invalid ABI buffer, or fallback to
the Python characteristic core must fail loudly.
"""

from __future__ import annotations

import importlib
import json
import os
from pathlib import Path
import statistics
import subprocess
import sys
from time import perf_counter

import numpy as np
import pytest
from scipy.special import roots_laguerre

from rabbit.drivers import full_coupled_typeI as driver
from rabbit.transport.characteristic_rays import setup_ray_grid


def _workspace(*, n_mu: int = 12, n_q: int = 20):
    bridge = importlib.import_module("rabbit.native_tier1")
    q_nodes, q_weights = roots_laguerre(n_q)
    return bridge.NativeTier1Workspace(
        ray_grid=setup_ray_grid(n_mu),
        momentum_grid=(q_nodes, q_weights),
        N_eff=3.044,
        f_nu=0.40520,
        ablate_hubble_anisotropy=False,
    )


def _buffers(*, n_mu: int = 12, n_q: int = 20):
    return (
        np.full(4, np.nan, dtype=np.float64),
        np.full(n_q, np.nan, dtype=np.float64),
        np.full(9, np.nan, dtype=np.float64),
        np.full(5 * n_mu, np.nan, dtype=np.float64),
    )


def test_rust_extension_and_bridge_import_without_skip(monkeypatch):
    extension = importlib.import_module("_rabbit_cpu")
    bridge = importlib.import_module("rabbit.native_tier1")

    assert extension is not None
    assert callable(bridge.NativeTier1Workspace)

    def missing_extension():
        raise ModuleNotFoundError("forced missing native extension")

    monkeypatch.setattr(bridge, "_load_extension", missing_extension)
    with pytest.raises(
        bridge.NativeTier1UnavailableError,
        match="native|extension|unavailable|required",
    ):
        _workspace(n_mu=4, n_q=4)


def test_eos_and_compact_characteristic_interface_exists():
    workspace = _workspace()
    derivative, monopole, aux, ray_work = _buffers()

    workspace.evaluate_into(
        np.array([0.1, 0.0, 0.0, 0.5], dtype=np.float64),
        derivative,
        monopole,
        aux,
        ray_work,
    )

    assert np.all(np.isfinite(derivative))
    assert np.all(np.isfinite(monopole))
    assert np.all(np.isfinite(aux))
    assert np.all(np.isfinite(ray_work))
    assert derivative[1] == 0.0
    assert derivative[2] == pytest.approx(0.1, abs=0.0)
    assert derivative[3] < 0.0
    assert aux[0] > 0.0 and aux[1] > 0.0
    assert aux[2] > 0.0
    assert np.all((monopole >= 0.0) & (monopole <= 1.0))


def test_native_boundary_rejects_nonfinite_omega_and_shape_without_clamp():
    workspace = _workspace()

    for state in (
        np.array([np.nan, 0.0, 0.0, 0.5], dtype=np.float64),
        np.array([1.0, 0.0, 0.0, 0.5], dtype=np.float64),
        np.array([0.1, 1.0e-6, 0.0, 0.5], dtype=np.float64),
    ):
        derivative, monopole, aux, ray_work = _buffers()
        with pytest.raises((ValueError, RuntimeError)):
            workspace.evaluate_into(state, derivative, monopole, aux, ray_work)

    derivative, monopole, aux, ray_work = _buffers()
    with pytest.raises((ValueError, RuntimeError)):
        workspace.evaluate_into(
            np.array([0.1, 0.0, 0.0, 0.5], dtype=np.float64),
            derivative[:-1],
            monopole,
            aux,
            ray_work,
        )

    shared = np.zeros(13, dtype=np.float64)
    shared[:4] = (0.1, 0.0, 0.0, 0.5)
    with pytest.raises(ValueError, match="alias"):
        workspace.evaluate_into(shared[:4], shared[:4], monopole, aux, ray_work)
    with pytest.raises(ValueError, match="alias"):
        workspace.evaluate_into(
            np.array([0.1, 0.0, 0.0, 0.5]),
            shared[:4],
            monopole,
            shared[2:11],
            ray_work,
        )


def test_full_coupled_config_has_explicit_rust_native_mode():
    assert driver.FullCoupledConfig().tier1_lrs_implementation == "python_reference"
    config = driver.FullCoupledConfig(
        tier1_lrs_implementation="rust_native",
        N_q=20,
        N_mu=12,
        tier=1,
        enable_collisions=False,
        enable_teff=False,
    )
    assert config.tier1_lrs_implementation == "rust_native"

    with pytest.raises((ValueError, NotImplementedError)):
        driver.FullCoupledConfig(
            tier1_lrs_implementation="rust_native",
            Sigma_H_minus=1.0e-6,
        )
    with pytest.raises((ValueError, NotImplementedError)):
        driver.FullCoupledConfig(
            tier1_lrs_implementation="rust_native",
            ablate_weak_transport_monopole=True,
        )

    rejected = (
        {"tier1_lrs_implementation": "unknown"},
        {"tier1_lrs_implementation": "rust_native", "tier": 2},
        {"tier1_lrs_implementation": "rust_native", "enable_collisions": True},
        {
            "tier1_lrs_implementation": "rust_native",
            "transport_mode": driver.TransportMode.LINEARIZED_PSTF,
        },
        {"tier1_lrs_implementation": "rust_native", "enable_teff": True},
    )
    for overrides in rejected:
        with pytest.raises((ValueError, NotImplementedError)):
            driver.FullCoupledConfig(**overrides)


def test_native_bdf_jacobian_respects_frozen_sigma_minus_row(monkeypatch):
    observed_rows = []
    original_solve_ivp = driver.solve_ivp

    def inspect_jacobian(*args, **kwargs):
        jacobian = kwargs.get("jac")
        if callable(jacobian):
            matrix = np.asarray(jacobian(kwargs["t_span"][0], kwargs["y0"]))
            observed_rows.append(matrix[1].copy())
        return original_solve_ivp(*args, **kwargs)

    monkeypatch.setattr(driver, "solve_ivp", inspect_jacobian)
    result = driver.run_full_coupled_typeI(
        driver.FullCoupledConfig(
            tier1_lrs_implementation="rust_native",
            Sigma_H_plus=0.1,
            N_q=8,
            N_mu=8,
        )
    )
    assert result.metadata["n_dof"] == 13
    assert len(observed_rows) == 2
    assert all(np.array_equal(row, np.zeros(13)) for row in observed_rows)


def test_real_endpoint_consumes_native_pack_in_both_phases(monkeypatch):
    config = driver.FullCoupledConfig(
        tier1_lrs_implementation="rust_native",
        Sigma_H_plus=0.1,
        Sigma_H_minus=0.0,
        N_q=8,
        N_mu=8,
        n_reactions=12,
        tier=1,
        enable_collisions=False,
        enable_teff=False,
    )
    bridge = importlib.import_module("rabbit.native_tier1")
    temperatures: list[float] = []
    original_evaluate = bridge.NativeTier1Workspace.evaluate_into

    def record_evaluate(self, state, derivative, monopole, aux, ray_work):
        temperatures.append(float(state[3]))
        return original_evaluate(self, state, derivative, monopole, aux, ray_work)

    def forbidden_python_core(*_args, **_kwargs):
        raise AssertionError("rust_native fell back to the Python Tier-1 core")

    for name in (
        "compute_typeI_geometry_rhs",
        "mu_current",
        "characteristic_transport_rhs",
        "char_extract_stress",
        "char_extract_monopole",
        "dT_gamma_dN_tier1",
        "T_nu_from_T_gamma_tier1",
        "_hubble_invsec",
    ):
        monkeypatch.setattr(driver, name, forbidden_python_core)

    with monkeypatch.context() as patch:
        patch.setattr(bridge.NativeTier1Workspace, "evaluate_into", record_evaluate)
        result = driver.run_full_coupled_typeI(config)

    assert temperatures
    assert any(value > 0.08 for value in temperatures)
    assert any(value <= 0.08 for value in temperatures)
    assert result.metadata["tier1_lrs_implementation"] == "rust_native"
    assert result.metadata["n_dof"] == 13
    assert abs(float(result.trajectory["T_gamma"][-1]) - 0.005) <= 1.0e-10

    def poison_derivative(_derivative, _monopole, _aux):
        _derivative[3] = np.nan

    def poison_hubble(_derivative, _monopole, _aux):
        _aux[1] = 0.0

    def poison_monopole(_derivative, _monopole, _aux):
        _monopole[0] = -1.0e-6

    for poison in (poison_derivative, poison_hubble, poison_monopole):
        calls = []

        def poison_evaluate(self, state, derivative, monopole, aux, ray_work):
            calls.append(float(state[3]))
            original_evaluate(self, state, derivative, monopole, aux, ray_work)
            poison(derivative, monopole, aux)

        with monkeypatch.context() as patch:
            patch.setattr(
                bridge.NativeTier1Workspace,
                "evaluate_into",
                poison_evaluate,
            )
            with pytest.raises(
                (ValueError, RuntimeError),
                match="native|non-finite|finite|Hubble|monopole",
            ):
                driver.run_full_coupled_typeI(config)
        assert len(calls) == 1


def _relative_error(actual: float, expected: float) -> float:
    return abs(float(actual) - float(expected)) / max(abs(float(expected)), 1.0e-300)


def _discrete_eos_complex(temperature):
    """Independent complex-step form of the frozen GL64 EOS functional."""
    from rabbit.thermo import eos_photon_electron as eos

    t = temperature
    x = eos._M_E / t
    rho_gamma = eos._RHO_GAMMA_PREFACTOR * t**4
    if float(np.real(x)) > 50.0:
        return rho_gamma, rho_gamma / 3.0, (4.0 / 3.0) * rho_gamma / t
    u = eos._GL_NODES.astype(np.complex128)
    weights = eos._GL_WEIGHTS.astype(np.complex128)
    energy = u + x
    momentum = np.sqrt(u * u + 2.0 * u * x)
    distribution = 1.0 / (np.exp(x) + np.exp(-u))
    i21 = np.sum(weights * energy**2 * momentum * distribution)
    i03 = np.sum(weights * momentum**3 * distribution)
    rho_e = 2.0 * t**4 * i21 / np.pi**2
    pressure_e = 2.0 * t**4 * i03 / (3.0 * np.pi**2)
    qed_scale = (5.0 * eos._ALPHA / (4.0 * np.pi)) * (
        1.0 - 0.20 * np.sqrt(eos._ALPHA / np.pi)
    )
    rho = rho_gamma + (1.0 + qed_scale) * rho_e
    pressure = rho_gamma / 3.0 + pressure_e + qed_scale * rho_e / 3.0
    return rho, pressure, (rho + pressure) / t


def test_native_eos_derivatives_temperature_and_hubble_parity():
    from rabbit.drivers.full_coupled_typeI import _MEV_TO_S, _hubble_invsec
    from rabbit.thermo.eos_photon_electron import (
        entropy_density_plasma,
        pressure_plasma,
        rho_plasma,
    )
    from rabbit.thermo.incomplete_decoupling import (
        T_nu_from_T_gamma_tier1,
        dT_gamma_dN_tier1,
    )

    workspace = _workspace()
    temperatures = (0.005, 0.01, 0.1, 0.5, 1.0, 2.0, 10.0)
    basic_errors = []
    derivative_errors = []
    for temperature in temperatures:
        derivative, monopole, aux, ray_work = _buffers()
        workspace.evaluate_into(
            np.array([0.1, 0.0, 0.0, temperature]),
            derivative,
            monopole,
            aux,
            ray_work,
        )
        t_nu = T_nu_from_T_gamma_tier1(temperature)
        h_inv_s = _hubble_invsec(temperature, t_nu, 3.044, 0.1**2)
        references = (
            (aux[4], rho_plasma(temperature)),
            (aux[5], pressure_plasma(temperature)),
            (aux[6], entropy_density_plasma(temperature)),
            (aux[2], t_nu),
            (aux[0], h_inv_s / _MEV_TO_S),
            (aux[1], h_inv_s),
        )
        basic_error = max(_relative_error(actual, expected) for actual, expected in references)
        basic_errors.append(basic_error)

        complex_step = 1.0e-20
        rho_cs, _pressure_cs, entropy_cs = _discrete_eos_complex(
            np.complex128(temperature + 1j * complex_step)
        )
        drho_dt = float(np.imag(rho_cs) / complex_step)
        ds_dt = float(np.imag(entropy_cs) / complex_step)
        dt_d_n = -3.0 * float(np.real(_discrete_eos_complex(temperature)[2])) / ds_dt
        derivatives = (
            (aux[8], drho_dt),
            (aux[7], ds_dt),
            (derivative[3], dt_d_n),
            (derivative[3], dT_gamma_dN_tier1(temperature)),
        )
        derivative_error = max(
            _relative_error(actual, expected) for actual, expected in derivatives
        )
        derivative_errors.append(derivative_error)
    print("R01_THERMO_PARITY " + json.dumps({
        "temperatures_MeV": temperatures,
        "relative_basic_errors": basic_errors,
        "relative_derivative_errors": derivative_errors,
        "max_relative_basic_error": max(basic_errors),
        "max_relative_derivative_error": max(derivative_errors),
    }, sort_keys=True))
    assert max(basic_errors) <= 1.0e-12
    assert max(derivative_errors) <= 2.0e-9


def test_native_characteristic_direct_ode_stress_and_monopole_parity():
    from scipy.integrate import solve_ivp

    from rabbit.transport.characteristic_rays import extract_monopole, extract_stress

    n_mu = 12
    n_q = 20
    mu0, w0, _x0, _signs = setup_ray_grid(n_mu)
    q_nodes, _q_weights = roots_laguerre(n_q)
    workspace = _workspace(n_mu=n_mu, n_q=n_q)

    def direct_rhs(_s, state):
        mu = state[:n_mu]
        jacobian = state[2 * n_mu:]
        p2 = 0.5 * (3.0 * mu * mu - 1.0)
        return np.concatenate((
            3.0 * mu * (1.0 - mu * mu),
            p2,
            3.0 * (1.0 - 3.0 * mu * mu) * jacobian,
        ))

    initial = np.concatenate((mu0, np.zeros(n_mu), np.ones(n_mu)))
    records = []
    for accumulated_shear in (-0.3, -0.1, 0.0, 0.1, 0.3):
        derivative, monopole, aux, ray_work = _buffers(n_mu=n_mu, n_q=n_q)
        workspace.evaluate_into(
            np.array([0.1, 0.0, accumulated_shear, 0.5]),
            derivative,
            monopole,
            aux,
            ray_work,
        )
        native_rays = ray_work.reshape(5, n_mu)
        if accumulated_shear == 0.0:
            direct = initial
        else:
            solution = solve_ivp(
                direct_rhs,
                (0.0, accumulated_shear),
                initial,
                method="DOP853",
                rtol=1.0e-13,
                atol=1.0e-15,
            )
            assert solution.success
            direct = solution.y[:, -1]
        intensity_error = float(np.max(np.abs(native_rays[1] - direct[n_mu:2 * n_mu])))
        jacobian_error = float(np.max(np.abs(native_rays[2] - direct[2 * n_mu:])))

        stress = extract_stress(
            native_rays[1], native_rays[2], native_rays[0], w0, 0.40520
        )
        monopole_reference = extract_monopole(
            native_rays[1], native_rays[2], w0, q_nodes
        )
        stress_error = float(abs(aux[3] - stress))
        monopole_error = float(np.max(np.abs(monopole - monopole_reference)))
        records.append({
            "S": accumulated_shear,
            "intensity_abs_error": intensity_error,
            "jacobian_abs_error": jacobian_error,
            "stress_abs_error": stress_error,
            "monopole_abs_error": monopole_error,
            "stress": float(aux[3]),
        })
    print("R01_CHARACTERISTIC_PARITY " + json.dumps(records, sort_keys=True))
    for record in records:
        assert record["intensity_abs_error"] <= 1.0e-9
        assert record["jacobian_abs_error"] <= 1.0e-9
        assert record["stress_abs_error"] <= 1.0e-12 * max(1.0, abs(record["stress"]))
        assert record["monopole_abs_error"] <= 1.0e-12
        if record["S"] == 0.0:
            assert abs(record["stress"]) <= 1.0e-14


def _endpoint_config(mode: str, *, sigma: float = 0.1, correction_level: int = 0):
    return driver.FullCoupledConfig(
        tier1_lrs_implementation=mode,
        Sigma_H_plus=sigma,
        Sigma_H_minus=0.0,
        N_q=20,
        N_mu=12,
        n_reactions=12,
        correction_level=correction_level,
        tier=1,
        enable_collisions=False,
        enable_teff=False,
    )


@pytest.mark.slow
def test_native_six_cell_endpoint_matrix_matches_python_reference(monkeypatch):
    terminal_abundances = []
    original_mass_residual = driver.mass_conservation_residual

    def capture_terminal_abundances(values):
        terminal_abundances.append(np.asarray(values, dtype=np.float64).copy())
        return original_mass_residual(values)

    monkeypatch.setattr(
        driver,
        "mass_conservation_residual",
        capture_terminal_abundances,
    )
    records = []
    for sigma in (0.0, 0.1, 0.3):
        for correction_level in (0, 2):
            reference = driver.run_full_coupled_typeI(
                _endpoint_config("python_reference", sigma=sigma, correction_level=correction_level)
            )
            reference_abundances = terminal_abundances[-1]
            native = driver.run_full_coupled_typeI(
                _endpoint_config("rust_native", sigma=sigma, correction_level=correction_level)
            )
            native_abundances = terminal_abundances[-1]
            records.append({
                "Sigma_plus": sigma,
                "correction_level": correction_level,
                "reference_Yp": reference.observables.Yp,
                "native_Yp": native.observables.Yp,
                "reference_DH": reference.observables.DH,
                "native_DH": native.observables.DH,
                "reference_Neff": reference.observables.N_eff,
                "native_Neff": native.observables.N_eff,
                "reference_Li7H": reference.observables.Li7H,
                "native_Li7H": native.observables.Li7H,
                "reference_Li6H": reference.observables.Li6H,
                "native_Li6H": native.observables.Li6H,
                "reference_terminal_X": reference_abundances.tolist(),
                "native_terminal_X": native_abundances.tolist(),
                "reference_terminal_X_min": float(np.min(reference_abundances)),
                "native_terminal_X_min": float(np.min(native_abundances)),
                "abs_delta_Yp": abs(native.observables.Yp - reference.observables.Yp),
                "relative_delta_DH": _relative_error(
                    native.observables.DH, reference.observables.DH
                ),
                "abs_delta_Neff": abs(
                    native.observables.N_eff - reference.observables.N_eff
                ),
                "mass_residual": abs(native.metadata["mass_conservation"]),
                "native_wall_s": native.wall_time_s,
                "reference_wall_s": reference.wall_time_s,
                "native_n_dof": native.metadata["n_dof"],
                "native_final_T_gamma": float(native.trajectory["T_gamma"][-1]),
            })
    print("R01_ENDPOINT_PARITY " + json.dumps(records, sort_keys=True))
    for record in records:
        assert record["abs_delta_Yp"] <= 5.0e-7
        assert record["relative_delta_DH"] <= 1.0e-3
        assert record["abs_delta_Neff"] <= 1.0e-10
        assert record["mass_residual"] <= 1.0e-10
        assert record["native_n_dof"] == 13
        assert abs(record["native_final_T_gamma"] - 0.005) <= 1.0e-10
        for name in ("Yp", "DH", "Neff", "Li7H", "Li6H"):
            assert np.isfinite(record[f"reference_{name}"])
            assert np.isfinite(record[f"native_{name}"])
            assert record[f"reference_{name}"] >= 0.0
            assert record[f"native_{name}"] >= 0.0
        for mode in ("reference", "native"):
            terminal = np.asarray(record[f"{mode}_terminal_X"])
            assert terminal.shape == (9,)
            assert np.all(np.isfinite(terminal))
            assert np.all(terminal >= 0.0)


def _fresh_process_rss(mode: str) -> int:
    source = """
import json, resource, sys
from rabbit.drivers.full_coupled_typeI import FullCoupledConfig, run_full_coupled_typeI
mode = sys.argv[1]
result = run_full_coupled_typeI(FullCoupledConfig(
    tier1_lrs_implementation=mode, Sigma_H_plus=0.1, Sigma_H_minus=0.0,
    N_q=20, N_mu=12, n_reactions=12, correction_level=0, tier=1,
    enable_collisions=False, enable_teff=False))
print(json.dumps({"rss": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,
                  "Yp": result.observables.Yp}))
"""
    environment = os.environ.copy()
    environment.update({
        "PYTHONPATH": "src",
        "RAYON_NUM_THREADS": "1",
        "OMP_NUM_THREADS": "1",
        "OPENBLAS_NUM_THREADS": "1",
        "MKL_NUM_THREADS": "1",
    })
    output = subprocess.check_output(
        [sys.executable, "-c", source, mode],
        cwd=Path(__file__).resolve().parents[1],
        env=environment,
        text=True,
    )
    return int(json.loads(output.strip().splitlines()[-1])["rss"])


@pytest.mark.slow
def test_native_full_endpoint_time_and_rss_gate():
    modes = ("python_reference", "rust_native")
    for mode in modes:
        driver.run_full_coupled_typeI(_endpoint_config(mode))

    timings = {mode: [] for mode in modes}
    orders = (
        modes,
        tuple(reversed(modes)),
        modes,
        tuple(reversed(modes)),
        modes,
    )
    for order in orders:
        for mode in order:
            start = perf_counter()
            driver.run_full_coupled_typeI(_endpoint_config(mode))
            timings[mode].append(perf_counter() - start)

    reference_median = statistics.median(timings["python_reference"])
    native_median = statistics.median(timings["rust_native"])

    reference_rss = _fresh_process_rss("python_reference")
    native_rss = _fresh_process_rss("rust_native")
    print("R01_ENDPOINT_PERFORMANCE " + json.dumps({
        "timings_s": timings,
        "reference_median_s": reference_median,
        "native_median_s": native_median,
        "native_to_reference_ratio": native_median / reference_median,
        "reference_peak_rss": reference_rss,
        "native_peak_rss": native_rss,
        "native_to_reference_rss_ratio": native_rss / reference_rss,
    }, sort_keys=True))
    assert native_median <= 0.75 * reference_median, timings
    assert native_rss <= 1.2 * reference_rss, {
        "python_reference": reference_rss,
        "rust_native": native_rss,
    }


def test_r01_loc_budget():
    root = Path(__file__).resolve().parents[1]
    rust_lines = 0
    for relative in ("native/rabbit_cpu/src/lib.rs", "native/rabbit_cpu/src/tier1_lrs.rs"):
        production = (root / relative).read_text().split("#[cfg(test)]", 1)[0]
        rust_lines += len(production.splitlines())

    wrapper_lines = len((root / "src/rabbit/native_tier1.py").read_text().splitlines())
    diff = subprocess.check_output(
        [
            "git", "diff", "--unified=0", "0496e5e", "--",
            "src/rabbit/drivers/full_coupled_typeI.py",
        ],
        cwd=root,
        text=True,
    )
    driver_added = sum(
        line.startswith("+") and not line.startswith("+++")
        for line in diff.splitlines()
    )
    print("R01_LOC_BUDGET " + json.dumps({
        "native_production_lines": rust_lines,
        "python_wrapper_lines": wrapper_lines,
        "driver_added_lines": driver_added,
        "python_integration_total": wrapper_lines + driver_added,
    }, sort_keys=True))
    assert rust_lines <= 600
    assert wrapper_lines + driver_added <= 200, {
        "wrapper_lines": wrapper_lines,
        "driver_added": driver_added,
    }
