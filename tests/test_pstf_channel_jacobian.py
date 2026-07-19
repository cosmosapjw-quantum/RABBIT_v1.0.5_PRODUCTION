"""tests/test_pstf_channel_jacobian.py — analytic product-rule Jacobian of the trilinear tensor.

The precomputed channel collision moment C (collisions/pstf_contractions.py, the six-monomial
einsum "34"/"12"/"123"/"124"/"134"/"234") is a cubic polynomial in the leg mode coefficients
F1,F2,F3,F4, each F appearing at most ONCE per monomial. Therefore dC/dF_k is the SAME einsum
with leg k dropped (its mode index freed) -- machine-exact, no AD. F4 enters only through the
fixed on-shell interpolation gather F4_interp = (1-w) F4[left] + w F4[left+1], so dF4_interp/dF4
is a constant interpolation matrix (chain rule).

This pins the analytic Jacobian against central finite differences of the tested contraction
(Track A1 of the solver-optimization plan; the production gate is the jacfwd-parity test).
"""

from __future__ import annotations

import numpy as np
import pytest

from rabbit.collisions.pstf_contractions import (
    _contract_pstf_channel_radial_grid_precomputed,
    build_pstf_channel_radial_grid_jacobian,
    pstf_channel_jacobian_jvp,
)


def _random_inputs(seed=0, ni=3, nm=2):
    rng = np.random.default_rng(seed)
    nj = nk = ni
    nn = ni + 2  # F4 first-axis size (must exceed max(left)+1)

    def Kt(*extra):
        return rng.standard_normal((ni, nj, nk, nm, *extra))

    K = {
        "34": Kt(nm, nm),      # ijk a d e
        "12": Kt(nm, nm),      # ijk a b c
        "123": Kt(nm, nm, nm),  # ijk a b c d
        "124": Kt(nm, nm, nm),  # ijk a b c e
        "134": Kt(nm, nm, nm),  # ijk a b d e
        "234": Kt(nm, nm, nm),  # ijk a c d e
    }
    F1 = rng.standard_normal((ni, nm))
    F2 = rng.standard_normal((nj, nm))
    F3 = rng.standard_normal((nk, nm))
    F4 = rng.standard_normal((nn, nm))
    q2_weights = rng.uniform(0.5, 1.5, nj)
    q3_weights = rng.uniform(0.5, 1.5, nk)
    p4_left = rng.integers(0, nn - 1, size=(ni, nj, nk))
    p4_right_weight = rng.uniform(0.0, 1.0, size=(ni, nj, nk))
    valid = rng.uniform(size=(ni, nj, nk)) > 0.2  # some masked-out configs
    return dict(F1=F1, F2=F2, F3=F3, F4=F4, q2_weights=q2_weights, q3_weights=q3_weights,
                K=K, p4_energies=np.zeros((ni, nj, nk)), p4_left=p4_left,
                p4_right_weight=p4_right_weight, valid=valid)


def _C(inp):
    return _contract_pstf_channel_radial_grid_precomputed(
        inp["F1"], inp["F2"], inp["F3"], inp["F4"],
        q2_weights=inp["q2_weights"], q3_weights=inp["q3_weights"],
        K=inp["K"], p4_energies=inp["p4_energies"], p4_left=inp["p4_left"],
        p4_right_weight=inp["p4_right_weight"], valid=inp["valid"]).C_modes


def _fd_block(inp, leg, eps=1e-6):
    """Central-difference dC/dF_leg -> array [i, a, n_leg, mode]."""
    F = inp[leg]
    nrad, nm = F.shape
    ni, na = _C(inp).shape
    out = np.zeros((ni, na, nrad, nm))
    for n in range(nrad):
        for m in range(nm):
            up = {**inp, leg: F.copy()}; up[leg][n, m] += eps
            dn = {**inp, leg: F.copy()}; dn[leg][n, m] -= eps
            out[:, :, n, m] = (_C(up) - _C(dn)) / (2 * eps)
    return out


@pytest.mark.parametrize("leg", ["F1", "F2", "F3", "F4"])
def test_analytic_jacobian_matches_finite_difference(leg):
    inp = _random_inputs(seed=3)
    jac = build_pstf_channel_radial_grid_jacobian(
        inp["F1"], inp["F2"], inp["F3"], inp["F4"],
        q2_weights=inp["q2_weights"], q3_weights=inp["q3_weights"],
        K=inp["K"], p4_energies=inp["p4_energies"], p4_left=inp["p4_left"],
        p4_right_weight=inp["p4_right_weight"], valid=inp["valid"])
    analytic = {"F1": jac.J1_full, "F2": jac.J2, "F3": jac.J3, "F4": jac.J4}[leg]
    fd = _fd_block(inp, leg)
    assert np.max(np.abs(analytic - fd)) < 1e-6, f"{leg}: max diff {np.max(np.abs(analytic-fd)):.2e}"


def _jac_kwargs(inp):
    return dict(q2_weights=inp["q2_weights"], q3_weights=inp["q3_weights"], K=inp["K"],
                p4_energies=inp["p4_energies"], p4_left=inp["p4_left"],
                p4_right_weight=inp["p4_right_weight"], valid=inp["valid"])


def test_jvp_matches_finite_difference_and_dense_blocks():
    """Matrix-free JVP J@dF == central FD of the contraction AND == dense-block contraction."""
    inp = _random_inputs(seed=7)
    rng = np.random.default_rng(11)
    dF = {f: rng.standard_normal(inp[f].shape) for f in ("F1", "F2", "F3", "F4")}

    dC = pstf_channel_jacobian_jvp(
        inp["F1"], inp["F2"], inp["F3"], inp["F4"],
        dF["F1"], dF["F2"], dF["F3"], dF["F4"], **_jac_kwargs(inp))

    # central finite difference of the real contraction along dF
    eps = 1e-6
    up = {k: (inp[k] + eps * dF[k] if k in dF else inp[k]) for k in inp}
    dn = {k: (inp[k] - eps * dF[k] if k in dF else inp[k]) for k in inp}
    fd = (_C(up) - _C(dn)) / (2 * eps)
    assert np.max(np.abs(dC - fd)) < 1e-6

    # internal consistency: JVP == sum of dense Jacobian blocks contracted with dF (machine)
    jac = build_pstf_channel_radial_grid_jacobian(
        inp["F1"], inp["F2"], inp["F3"], inp["F4"], **_jac_kwargs(inp))
    dC_blocks = (np.einsum("iab,ib->ia", jac.J1_diag, dF["F1"])
                 + np.einsum("iajc,jc->ia", jac.J2, dF["F2"])
                 + np.einsum("iakd,kd->ia", jac.J3, dF["F3"])
                 + np.einsum("iane,ne->ia", jac.J4, dF["F4"]))
    assert np.max(np.abs(dC - dC_blocks)) < 1e-12


@pytest.mark.parametrize("ni,nm", [(2, 1), (4, 3), (5, 2)])
def test_jvp_matches_fd_across_shapes(ni, nm):
    """Mode/radial-index robustness: vary n_modes (1,3,2) and grid size so a square (3,2)
    fixture cannot mask an index-broadcast bug."""
    inp = _random_inputs(seed=ni * 10 + nm, ni=ni, nm=nm)
    rng = np.random.default_rng(ni + nm)
    dF = {f: rng.standard_normal(inp[f].shape) for f in ("F1", "F2", "F3", "F4")}
    dC = pstf_channel_jacobian_jvp(
        inp["F1"], inp["F2"], inp["F3"], inp["F4"],
        dF["F1"], dF["F2"], dF["F3"], dF["F4"], **_jac_kwargs(inp))
    eps = 1e-6
    up = {k: (inp[k] + eps * dF[k] if k in dF else inp[k]) for k in inp}
    dn = {k: (inp[k] - eps * dF[k] if k in dF else inp[k]) for k in inp}
    fd = (_C(up) - _C(dn)) / (2 * eps)
    assert np.max(np.abs(dC - fd)) < 1e-6


def test_jvp_is_linear_in_perturbation():
    inp = _random_inputs(seed=9)
    rng = np.random.default_rng(13)
    dF = {f: rng.standard_normal(inp[f].shape) for f in ("F1", "F2", "F3", "F4")}
    args = lambda scale: (inp["F1"], inp["F2"], inp["F3"], inp["F4"],
                          scale * dF["F1"], scale * dF["F2"], scale * dF["F3"], scale * dF["F4"])
    d1 = pstf_channel_jacobian_jvp(*args(1.0), **_jac_kwargs(inp))
    d3 = pstf_channel_jacobian_jvp(*args(3.0), **_jac_kwargs(inp))
    assert np.allclose(d3, 3.0 * d1, atol=1e-12)


def test_J1_is_diagonal_in_leg1_radial_index():
    """C[i,a] depends only on F1[i,:] -> dC/dF1 nonzero only at n==i."""
    inp = _random_inputs(seed=5)
    jac = build_pstf_channel_radial_grid_jacobian(
        inp["F1"], inp["F2"], inp["F3"], inp["F4"],
        q2_weights=inp["q2_weights"], q3_weights=inp["q3_weights"],
        K=inp["K"], p4_energies=inp["p4_energies"], p4_left=inp["p4_left"],
        p4_right_weight=inp["p4_right_weight"], valid=inp["valid"])
    ni = inp["F1"].shape[0]
    for i in range(ni):
        for n in range(ni):
            if n != i:
                assert np.max(np.abs(jac.J1_full[i, :, n, :])) == 0.0
