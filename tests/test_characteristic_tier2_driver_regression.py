from __future__ import annotations

from types import SimpleNamespace

import numpy as np
import pytest
from scipy.sparse import issparse

from rabbit.config.solver_config import SolverConfig, SolverMethod
from rabbit.drivers import full_coupled_typeI as mod
from rabbit.drivers.full_coupled_typeI import FullCoupledConfig, run_full_coupled_typeI


def test_driver_tier2_per_species_regression():
    r = run_full_coupled_typeI(FullCoupledConfig(
        Sigma_H_plus=0.02,
        tier=2,
        enable_collisions=True,
        correction_level=3,
        enable_teff=False,
        N_q=12,
        N_mu=12,
    ))
    assert r.metadata["transport_species_mode"] == "per_species"
    assert r.metadata["species_identical_approx"] is False
    assert r.metadata["production_authority"] == "characteristic_decoupling_backbone_residual_relaxation"
    assert r.metadata["weak_background_mode"] == "isotropic_decoupling_backbone_v1"
    assert r.metadata["decoupling_backbone_mode"] == "isotropic_momentum_grid_v1"
    assert r.metadata["residual_rate_calibration_mode"] == "spectrum+blocking+mismatch+distortion_v1"
    assert r.observables.Yp > 0.0
    assert r.observables.DH > 0.0


def test_characteristic_helper_cache_and_event_contract(monkeypatch):
    grid_a = mod.setup_ray_grid(12)
    grid_b = mod.setup_ray_grid(12)
    grid_c = mod.setup_ray_grid(11)

    assert grid_a[0] is grid_b[0]
    assert grid_a[1] is grid_b[1]
    assert grid_a[2] is grid_b[2]
    assert grid_a[3] is grid_b[3]
    assert grid_a[0] is not grid_c[0]

    cfg = FullCoupledConfig(
        Sigma_H_plus=0.0,
        N_q=4,
        N_mu=12,
        tier=1,
        enable_teff=False,
        solver=SolverConfig(
            method=SolverMethod.BDF,
            rtol=1e-6,
            atol=1e-8,
            max_step=0.5,
        ),
    )
    _, _, i_tg, _, _, i_net, _, _, _, _ = mod._layout_characteristic(cfg.N_mu, tier=cfg.tier)
    captured_calls: list[dict[str, object]] = []

    def fake_solve_ivp(fun, t_span, y0, events=None, **kwargs):
        y_start = np.array(y0, dtype=np.float64, copy=True)
        y_stop = y_start.copy()
        if len(captured_calls) == 0:
            y_stop[i_tg] = cfg.T_handoff
        else:
            y_stop[i_tg] = cfg.T_end
            y_stop[i_net + 1] = 0.75
            y_stop[i_net + 2] = 5.0e-5
            y_stop[i_net + 5] = 0.24
            y_stop[i_net + 6] = 1.0e-12
            y_stop[i_net + 7] = 1.0e-12
            y_stop[i_net + 8] = 1.0e-15
        captured_calls.append(
            {
                "events": events,
                "kwargs": kwargs,
                "y0": y_start,
            }
        )
        return SimpleNamespace(
            t=np.array([t_span[0], t_span[1]], dtype=np.float64),
            y=np.column_stack([y_start, y_stop]),
            t_events=[np.array([t_span[1]], dtype=np.float64)],
            status=1,
            success=True,
            message="event reached",
        )

    monkeypatch.setattr(mod, "solve_ivp", fake_solve_ivp)

    result = run_full_coupled_typeI(cfg)

    assert len(captured_calls) == 2
    event_p1 = captured_calls[0]["events"]
    event_p2 = captured_calls[1]["events"]
    assert callable(event_p1)
    assert callable(event_p2)
    assert event_p1 is not event_p2
    assert event_p1.terminal is True
    assert event_p2.terminal is True
    assert event_p1.direction == -1
    assert event_p2.direction == -1
    assert event_p1(0.0, captured_calls[0]["y0"]) == pytest.approx(
        captured_calls[0]["y0"][i_tg] - cfg.T_handoff
    )
    assert event_p2(0.0, captured_calls[1]["y0"]) == pytest.approx(
        captured_calls[1]["y0"][i_tg] - cfg.T_end
    )
    assert issparse(captured_calls[0]["kwargs"]["jac_sparsity"])
    assert result.observables.Yp == pytest.approx(0.24)


def test_characteristic_odd_n_mu_keeps_exact_central_ray():
    sigma_plus = 0.125
    mu0, w0, X0, signs = mod.setup_ray_grid(11)
    center = len(mu0) // 2

    assert w0.shape == (11,)
    assert mu0[center] == pytest.approx(0.0, abs=1e-15)
    assert signs[center] == pytest.approx(0.0, abs=0.0)

    for shear_integral in (-1.25, 0.0, 1.25):
        mu = mod.mu_current(X0, signs, shear_integral)
        assert mu[center] == pytest.approx(0.0, abs=1e-15)

    I = np.zeros_like(mu0)
    J = np.ones_like(mu0)
    dI, dJ, dS = mod.characteristic_transport_rhs(
        sigma_plus,
        I,
        J,
        mod.mu_current(X0, signs, 0.7),
    )

    assert dI[center] == pytest.approx(-0.5 * sigma_plus)
    assert dJ[center] == pytest.approx(3.0 * sigma_plus)
    assert dS == pytest.approx(sigma_plus)


def test_characteristic_sparse_jacobian_helper_matches_driver_wiring(monkeypatch):
    cfg = FullCoupledConfig(
        Sigma_H_plus=0.01,
        N_q=4,
        N_mu=8,
        tier=1,
        enable_teff=False,
        solver=SolverConfig(
            method=SolverMethod.BDF,
            rtol=1e-6,
            atol=1e-8,
            max_step=0.5,
        ),
    )
    expected = mod._characteristic_jac_sparsity(cfg.N_mu, tier=cfg.tier, per_species=False)
    captured = []

    _, _, i_tg, _, _, i_net, _, _, _, _ = mod._layout_characteristic(cfg.N_mu, tier=cfg.tier)

    def fake_solve_ivp(fun, t_span, y0, events=None, **kwargs):
        y_start = np.array(y0, dtype=np.float64, copy=True)
        y_stop = y_start.copy()
        y_stop[i_tg] = cfg.T_handoff if len(captured) == 0 else cfg.T_end
        if len(captured) > 0:
            y_stop[i_net + 1] = 0.75
            y_stop[i_net + 2] = 5.0e-5
            y_stop[i_net + 5] = 0.24
        captured.append(kwargs["jac_sparsity"])
        return SimpleNamespace(
            t=np.array([t_span[0], t_span[1]], dtype=np.float64),
            y=np.column_stack([y_start, y_stop]),
            t_events=[np.array([t_span[1]], dtype=np.float64)],
            status=1,
            success=True,
            message="event reached",
        )

    monkeypatch.setattr(mod, "solve_ivp", fake_solve_ivp)

    run_full_coupled_typeI(cfg)

    assert len(captured) == 2
    for actual in captured:
        assert issparse(actual)
        assert actual.shape == expected.shape
        assert actual.nnz == expected.nnz
        diff = (actual != expected).nnz
        assert diff == 0


@pytest.mark.slow
def test_characteristic_bdf_sparse_smoke_matches_radau():
    base = dict(
        Sigma_H_plus=0.03,
        N_q=6,
        N_mu=8,
        tier=1,
        n_reactions=12,
        correction_level=0,
        enable_teff=False,
    )
    radau = run_full_coupled_typeI(
        FullCoupledConfig(
            solver=SolverConfig(
                method=SolverMethod.RADAU,
                rtol=1e-6,
                atol=1e-8,
                max_step=0.5,
            ),
            **base,
        )
    )
    bdf = run_full_coupled_typeI(
        FullCoupledConfig(
            solver=SolverConfig(
                method=SolverMethod.BDF,
                rtol=1e-6,
                atol=1e-8,
                max_step=0.5,
            ),
            **base,
        )
    )

    assert radau.metadata["transport_mode"] == "characteristic"
    assert bdf.metadata["transport_mode"] == "characteristic"
    assert abs(radau.observables.Yp - bdf.observables.Yp) < 1.0e-4
    rel_dh = abs(radau.observables.DH - bdf.observables.DH) / max(radau.observables.DH, 1.0e-30)
    assert rel_dh < 2.0e-3
