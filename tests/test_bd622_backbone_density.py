"""BD622-R1 (audit F-025): the tier-2 decoupling backbone must keep its momentum-node
density n_y / y_max at or above the target as N_q grows.

Pre-BD622, n_y = max(81, 2*N_q+1) stayed pinned at 81 while y_max grew with the largest
Gauss-Laguerre root, so refining N_q collapsed the backbone density (N_q=20 -> 0.377,
N_q=32 -> 0.224) and the BD621 promotion ladder walked the under-resolved regime.
"""

import numpy as np
import pytest

import rabbit.drivers.full_coupled_typeI as fct
from rabbit.config.grids import MomentumGrid


class _CaptureSentinel(Exception):
    pass


def _captured_backbone_grid(monkeypatch, n_q: int):
    """Return the (n_y, y_max) that _build_decoupling_backbone selects for N_q=n_q."""
    captured = {}

    def _capture(config):
        captured["n_y"] = config.n_y
        captured["y_max"] = config.y_max
        raise _CaptureSentinel

    monkeypatch.setattr(fct, "solve_isotropic_decoupling", _capture)
    fct._DECOUPLING_BACKBONE_CACHE.clear()

    grid = MomentumGrid(N_q=n_q)
    config = fct.FullCoupledConfig(N_q=n_q, tier=2, enable_collisions=True)
    with pytest.raises(_CaptureSentinel):
        fct._build_decoupling_backbone(config, grid.nodes, grid.weights)
    fct._DECOUPLING_BACKBONE_CACHE.clear()
    return captured["n_y"], captured["y_max"]


@pytest.mark.parametrize("n_q", [12, 20, 32])
def test_backbone_density_meets_target(monkeypatch, n_q):
    n_y, y_max = _captured_backbone_grid(monkeypatch, n_q)
    density = n_y / y_max
    assert density >= fct._BACKBONE_NODE_DENSITY_TARGET, (
        f"N_q={n_q}: backbone density {density:.3f} (n_y={n_y}, y_max={y_max:.1f}) "
        f"below target {fct._BACKBONE_NODE_DENSITY_TARGET}"
    )


def test_backbone_density_does_not_collapse_with_n_q(monkeypatch):
    densities = []
    for n_q in (12, 20, 32):
        n_y, y_max = _captured_backbone_grid(monkeypatch, n_q)
        densities.append(n_y / y_max)
    # The pre-fix failure mode: density monotonically COLLAPSING as N_q grows
    # (0.485 -> 0.377 -> 0.224). Post-fix every rung must hold the floor.
    assert min(densities) >= fct._BACKBONE_NODE_DENSITY_TARGET


def test_backbone_n_y_is_odd(monkeypatch):
    for n_q in (12, 20, 32):
        n_y, _ = _captured_backbone_grid(monkeypatch, n_q)
        assert n_y % 2 == 1
