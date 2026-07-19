"""tests/test_pstf_lrs_moment_jacobian.py — analytic Jacobian of the LRS collision moment.

The axisymmetric LRS collision moment C_L (pstf_lrs_moment_kernel.axisymmetric_collision_moment)
is CUBIC in the amplitudes Gamma^(i)_l: a linear term sum_i eps_i F^(1;i)_L Gamma^(i)_L plus a
trilinear term sum F^(3) <..><..> Gamma Gamma Gamma. So dC_L/dGamma is the SAME product rule as
the production tensor Jacobian (Track A1): the linear term contributes eps_i F^(1;i)_L at rank L,
and each trilinear monomial contributes its value with ONE Gamma-leg dropped. This is the analytic
moment-Jacobian the transport solve uses (plan B.J) -- no AD of the kernel.

Pinned against central finite differences of the moment itself.
"""

from __future__ import annotations

import numpy as np
import pytest

from rabbit.collisions.pstf_lrs_moment_kernel import (
    axisymmetric_collision_moment,
    axisymmetric_collision_moment_jacobian,
)
from rabbit.collisions.pstf_collision_operator import _trilinear_rank_combos


def _synthetic(L, ell_max=2, seed=0):
    rng = np.random.default_rng(seed)
    gammas = {i: rng.standard_normal(ell_max + 1) for i in range(4)}
    reduced_linear = {i: {L: float(rng.standard_normal())} for i in range(4)}
    triples = ((0, 1, 2), (0, 1, 3), (0, 2, 3), (1, 2, 3))
    combos = _trilinear_rank_combos(L, ell_max)
    reduced_trilinear = {
        trip: {(L, li, lj, lk, J): float(rng.standard_normal()) for (li, lj, lk, J) in combos}
        for trip in triples
    }
    return reduced_linear, reduced_trilinear, gammas


def _moment(L, rl, rt, gammas):
    return axisymmetric_collision_moment(L, rl, rt, gammas)


@pytest.mark.parametrize("L", [0, 1, 2])
def test_moment_jacobian_matches_finite_difference(L):
    rl, rt, gammas = _synthetic(L, ell_max=2, seed=L + 1)
    jac = axisymmetric_collision_moment_jacobian(L, rl, rt, gammas)

    eps = 1e-6
    for m in range(4):
        n_ell = len(gammas[m])
        assert jac[m].shape == (n_ell,)
        for ell in range(n_ell):
            up = {i: gammas[i].copy() for i in range(4)}
            dn = {i: gammas[i].copy() for i in range(4)}
            up[m][ell] += eps
            dn[m][ell] -= eps
            fd = (_moment(L, rl, rt, up) - _moment(L, rl, rt, dn)) / (2 * eps)
            assert jac[m][ell] == pytest.approx(fd, abs=1e-6), f"L={L} leg={m} ell={ell}"


def test_moment_jacobian_directional_consistency():
    """sum_{m,ell} J[m][ell] dGamma[m][ell] == directional derivative (FD) along dGamma."""
    L = 1
    rl, rt, gammas = _synthetic(L, ell_max=2, seed=42)
    jac = axisymmetric_collision_moment_jacobian(L, rl, rt, gammas)
    rng = np.random.default_rng(7)
    dG = {m: rng.standard_normal(len(gammas[m])) for m in range(4)}
    analytic = sum(float(np.dot(jac[m], dG[m])) for m in range(4))
    eps = 1e-6
    up = {m: gammas[m] + eps * dG[m] for m in range(4)}
    dn = {m: gammas[m] - eps * dG[m] for m in range(4)}
    fd = (_moment(L, rl, rt, up) - _moment(L, rl, rt, dn)) / (2 * eps)
    assert analytic == pytest.approx(fd, abs=1e-6)


def test_jacobian_tolerates_missing_leg():
    """Sparse process input: a leg absent from `gammas` contributes nothing; jac leg is empty."""
    L = 1
    gammas = {0: np.array([0.3, 0.1]), 1: np.array([0.2, -0.4]), 3: np.array([0.5, -0.1])}  # no leg 2
    rl = {0: {L: 0.5}, 1: {L: -0.3}}
    rt = {(0, 1, 3): {(L, 1, 1, 1, 1): 0.7}}  # triple without the missing leg 2
    jac = axisymmetric_collision_moment_jacobian(L, rl, rt, gammas)
    assert jac[2].shape == (0,)  # missing leg -> empty Jacobian row, no KeyError
    eps = 1e-6
    for m in (0, 1, 3):
        for ell in range(len(gammas[m])):
            up = {i: gammas[i].copy() for i in gammas}
            dn = {i: gammas[i].copy() for i in gammas}
            up[m][ell] += eps
            dn[m][ell] -= eps
            fd = (axisymmetric_collision_moment(L, rl, rt, up)
                  - axisymmetric_collision_moment(L, rl, rt, dn)) / (2 * eps)
            assert jac[m][ell] == pytest.approx(fd, abs=1e-6)


def test_jacobian_skips_ranks_beyond_spectrum():
    """ell_max truncation: a trilinear entry whose rank exceeds a leg's spectrum is skipped."""
    L = 1
    gammas = {i: np.array([0.3, -0.2]) for i in range(4)}  # length 2 -> ranks 0,1 only
    base = {(0, 1, 3): {(L, 1, 1, 1, 1): 0.7}}
    # same plus an entry referencing rank 2 on leg k=3 (>= len 2) -> must be silently skipped
    trunc = {(0, 1, 3): {(L, 1, 1, 1, 1): 0.7, (L, 1, 1, 2, 2): 0.9}}
    jac_base = axisymmetric_collision_moment_jacobian(L, {}, base, gammas)
    jac_trunc = axisymmetric_collision_moment_jacobian(L, {}, trunc, gammas)
    for m in range(4):
        assert np.allclose(jac_base[m], jac_trunc[m], atol=1e-15)  # truncated rank -> no effect


def test_linear_term_jacobian_is_eps_F1_at_rank_L():
    """With no trilinear kernels, dC_L/dGamma^(i)_L = eps_i F^(1;i)_L exactly."""
    from rabbit.collisions.pstf_lrs_moment_kernel import LEG_SIGNS
    L = 2
    gammas = {i: np.zeros(3) for i in range(4)}
    reduced_linear = {i: {L: 0.5 + i} for i in range(4)}
    jac = axisymmetric_collision_moment_jacobian(L, reduced_linear, {}, gammas)
    for i in range(4):
        assert jac[i][L] == pytest.approx(LEG_SIGNS[i] * (0.5 + i), abs=1e-13)
        # ranks != L get no linear contribution
        for ell in (0, 1):
            assert jac[i][ell] == pytest.approx(0.0, abs=1e-13)
