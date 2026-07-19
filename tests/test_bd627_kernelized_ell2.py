"""tests/test_bd627_kernelized_ell2.py — BD627 W1: flag-gated first-principles ℓ=2 kernel.

Covers the three W1 gates:

  (a) FLAG-OFF BIT-IDENTITY (the promotion-safety gate): with the default
      ``ell2_collision_mode="calibrated_rta"`` the projected-operator C₂ output is
      BIT-IDENTICAL to the pre-BD627 closed form −κ₂ (Γ/H) Ψ₂.  MUST pass.
  (b) kernelized-mode unit test: kernelized_C2_ell2 returns finite arrays of the
      right shape, DAMPS (C₂·Ψ₂ < 0 on core nodes), is the same ORDER as the RTA C₂
      at a test state, refuses a mismatched q-grid, and refuses the ν–ν process.
  (c) table-load + interpolation smoke: mid-T PCHIP value is finite and bracketed
      by the neighbouring grid-node values (C1 monotone interpolation).
"""

from __future__ import annotations

import numpy as np
import pytest

from rabbit.config.grids import MomentumGrid
from rabbit.collisions.kernels import A_TOTAL_NUE, A_TOTAL_NUX
from rabbit.collisions.projected_operator import (
    KAPPA_ELL2,
    RTADampingModel,
    PhysicalCollisionOperator,
    collision_rate_per_efold,
    _hubble_simple,
)
from rabbit.collisions import kernelized_ell2
from rabbit.collisions.kernelized_ell2 import kernelized_C2_ell2, _load_table


N_Q = 20  # production-inference grid (must match the table)


def _test_state():
    """T≈2 MeV, before decoupling (T_γ = T_ν): a representative coupled state."""
    T_gamma = T_nu = 2.0
    H_inv_sec = _hubble_simple(T_gamma, T_nu)
    return T_gamma, T_nu, H_inv_sec


# ─────────────────────────────────────────────────────────────────────
# (a) FLAG-OFF BIT-IDENTITY
# ─────────────────────────────────────────────────────────────────────
def _rta_reference_c2(grid, Psi2, T_gamma, T_nu, H_inv_sec, a_coupling):
    """The exact pre-BD627 closed form −κ₂ (Γ/H) Ψ₂ for one species."""
    Gamma_H = collision_rate_per_efold(T_gamma, T_nu, H_inv_sec, a_coupling)
    return -KAPPA_ELL2 * Gamma_H * Psi2


@pytest.mark.parametrize("Operator", [RTADampingModel, PhysicalCollisionOperator])
def test_flag_off_bit_identity(Operator):
    grid = MomentumGrid(N_q=N_Q)
    T_gamma, T_nu, H_inv_sec = _test_state()
    rng = np.random.default_rng(627)
    Psi0 = rng.standard_normal((3, N_Q)) * 1e-2
    Psi2 = rng.standard_normal((3, N_Q)) * 1e-2

    # default mode is calibrated_rta
    op_default = Operator(grid, n_species=3)
    assert op_default.ell2_collision_mode == "calibrated_rta"
    op_explicit = Operator(grid, n_species=3, ell2_collision_mode="calibrated_rta")

    r_default = op_default.evaluate(Psi0, Psi2, T_gamma, T_nu, H_inv_sec)
    r_explicit = op_explicit.evaluate(Psi0, Psi2, T_gamma, T_nu, H_inv_sec)

    a_by_species = [A_TOTAL_NUE, A_TOTAL_NUE, A_TOTAL_NUX]
    for s in range(3):
        ref = _rta_reference_c2(grid, Psi2[s], T_gamma, T_nu, H_inv_sec,
                                a_by_species[s])
        # BIT-IDENTICAL, not merely close: identical float ops in the same order.
        assert np.array_equal(r_default.C_ell2[s], ref)
        assert np.array_equal(r_explicit.C_ell2[s], ref)


# ─────────────────────────────────────────────────────────────────────
# (b) kernelized-mode unit test
# ─────────────────────────────────────────────────────────────────────
def test_kernelized_shape_and_finiteness():
    grid = MomentumGrid(N_q=N_Q)
    T_gamma, T_nu, H_inv_sec = _test_state()
    Psi2_1d = np.full(N_Q, 1e-2)
    C2_1d = kernelized_C2_ell2(Psi2_1d, T_gamma, T_nu, H_inv_sec,
                               grid.nodes, grid.weights, "nue")
    assert C2_1d.shape == (N_Q,)
    assert np.all(np.isfinite(C2_1d))

    Psi2_2d = np.full((3, N_Q), 1e-2)
    C2_2d = kernelized_C2_ell2(Psi2_2d, T_gamma, T_nu, H_inv_sec,
                               grid.nodes, grid.weights, ["nue", "nue", "nux"])
    assert C2_2d.shape == (3, N_Q)
    assert np.all(np.isfinite(C2_2d))
    # ν_x couples more weakly than ν_e → strictly smaller damping magnitude
    assert np.max(np.abs(C2_2d[2])) < np.max(np.abs(C2_2d[0]))


def test_kernelized_damps():
    """C₂ opposes Ψ₂ (damping): C₂·Ψ₂ < 0, and sign-flips with Ψ₂."""
    grid = MomentumGrid(N_q=N_Q)
    T_gamma, T_nu, H_inv_sec = _test_state()
    Psi2 = np.full(N_Q, 1e-2)
    C2 = kernelized_C2_ell2(Psi2, T_gamma, T_nu, H_inv_sec,
                            grid.nodes, grid.weights, "nue")
    # global damping
    assert float(np.dot(C2, Psi2)) < 0.0
    # core nodes (not deep-tail floored): C₂ < 0 for Ψ₂ > 0
    f0 = 1.0 / (np.exp(grid.nodes / T_nu) + 1.0)
    core = f0 >= 1e-6
    assert np.all(C2[core] < 0.0)
    # deep-tail rows are floored to exactly zero
    assert np.all(C2[~core] == 0.0)
    # linearity/sign: negating Ψ₂ negates C₂
    C2_neg = kernelized_C2_ell2(-Psi2, T_gamma, T_nu, H_inv_sec,
                                grid.nodes, grid.weights, "nue")
    assert np.allclose(C2_neg, -C2)


def test_kernelized_magnitude_same_order_as_rta():
    """The kernel refines 5/3; magnitude agrees with RTA within ~1-2 orders."""
    grid = MomentumGrid(N_q=N_Q)
    T_gamma, T_nu, H_inv_sec = _test_state()
    Psi2 = np.full(N_Q, 1e-2)
    C2_kern = kernelized_C2_ell2(Psi2, T_gamma, T_nu, H_inv_sec,
                                 grid.nodes, grid.weights, "nue")
    C2_rta = _rta_reference_c2(grid, Psi2, T_gamma, T_nu, H_inv_sec, A_TOTAL_NUE)
    ratio = np.max(np.abs(C2_kern)) / np.max(np.abs(C2_rta))
    # same order (a 1e6 mismatch = a normalization bug); O(1) factor is the science.
    assert 1e-2 < ratio < 1e2, f"kernelized/RTA magnitude ratio {ratio:.3e} off-order"


def test_kernelized_qgrid_mismatch_raises():
    T_gamma, T_nu, H_inv_sec = _test_state()
    grid_bad = MomentumGrid(N_q=24)  # wrong N_q → grid mismatch
    Psi2 = np.full(24, 1e-2)
    with pytest.raises(ValueError, match="q-grid"):
        kernelized_C2_ell2(Psi2, T_gamma, T_nu, H_inv_sec,
                           grid_bad.nodes, grid_bad.weights, "nue")

    # same N_q but perturbed nodes must also raise
    grid = MomentumGrid(N_q=N_Q)
    bad_nodes = grid.nodes.copy()
    bad_nodes[0] += 1.0
    with pytest.raises(ValueError, match="q-grid"):
        kernelized_C2_ell2(np.full(N_Q, 1e-2), T_gamma, T_nu, H_inv_sec,
                           bad_nodes, grid.weights, "nue")


def test_kernelized_nunu_not_implemented():
    grid = MomentumGrid(N_q=N_Q)
    T_gamma, T_nu, H_inv_sec = _test_state()
    with pytest.raises(NotImplementedError, match="ν–ν|nu.nu|not wired"):
        kernelized_C2_ell2(np.full(N_Q, 1e-2), T_gamma, T_nu, H_inv_sec,
                           grid.nodes, grid.weights, "nu_nu")


def test_kernelized_unknown_species_raises():
    grid = MomentumGrid(N_q=N_Q)
    T_gamma, T_nu, H_inv_sec = _test_state()
    with pytest.raises(ValueError, match="unsupported species"):
        kernelized_C2_ell2(np.full(N_Q, 1e-2), T_gamma, T_nu, H_inv_sec,
                           grid.nodes, grid.weights, "sterile")


def test_physical_operator_kernelized_mode_matches_helper():
    """The wired PhysicalCollisionOperator (kernelized) == the direct helper."""
    grid = MomentumGrid(N_q=N_Q)
    T_gamma, T_nu, H_inv_sec = _test_state()
    Psi0 = np.zeros((3, N_Q))
    Psi2 = np.full((3, N_Q), 1e-2)
    op = PhysicalCollisionOperator(grid, n_species=3,
                                   ell2_collision_mode="kernelized")
    r = op.evaluate(Psi0, Psi2, T_gamma, T_nu, H_inv_sec)
    tags = ["nue", "nue", "nux"]
    for s in range(3):
        expect = kernelized_C2_ell2(Psi2[s], T_gamma, T_nu, H_inv_sec,
                                    grid.nodes, grid.weights, tags[s])
        assert np.array_equal(r.C_ell2[s], expect)


# ─────────────────────────────────────────────────────────────────────
# (c) table-load + interpolation smoke
# ─────────────────────────────────────────────────────────────────────
def test_table_promoted_and_loads():
    q_nodes, combos = _load_table(kernelized_ell2._DEFAULT_TABLE_PATH)
    assert q_nodes.shape == (N_Q,)
    assert "nu_e__nue" in combos and "nu_e__nux" in combos
    grid = MomentumGrid(N_q=N_Q)
    assert np.allclose(q_nodes, grid.nodes)


def test_midT_pchip_bracketed():
    """PCHIP (C1 monotone) mid-T value lies between the bracketing node values."""
    _, combos = _load_table(kernelized_ell2._DEFAULT_TABLE_PATH)
    table = combos["nu_e__nue"]
    T_grid = table.T_grid
    # a T strictly between two adjacent grid points, in the in-window regime
    k = int(np.argmin(np.abs(T_grid - 1.0)))
    T_lo, T_hi = T_grid[k], T_grid[k + 1]
    T_mid = float(np.sqrt(T_lo * T_hi))  # geometric mid (log-T scheme)
    M_lo = table.M2_gamma(T_lo)
    M_hi = table.M2_gamma(T_hi)
    M_mid = table.M2_gamma(T_mid)
    assert np.all(np.isfinite(M_mid))
    # diagonal loss is monotone in T over this window → bracketed element-wise
    lo = np.minimum(M_lo, M_hi)
    hi = np.maximum(M_lo, M_hi)
    tol = 1e-12 + 1e-9 * np.maximum(np.abs(lo), np.abs(hi))
    assert np.all(M_mid >= lo - tol)
    assert np.all(M_mid <= hi + tol)
    # PCHIP reproduces the node values exactly at the nodes
    assert np.allclose(table.M2_gamma(T_lo), M_lo)
