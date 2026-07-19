"""tests/test_even_ladder_wellposedness.py — PR#14: well-posedness of the even-ladder transport.

The Type-I even-ell shear-advection RHS (compute_hierarchy_rhs_typeI_even_ladder) is

    RHS = Sigma_H [ M_stream @ (q d/dq) F + 1.5 M_angle @ F ],

with M_stream / M_angle constant ell-coupling matrices. Because Bianchi I is spatially homogeneous
(D_a = 0), there is NO spatial flux Jacobian and therefore NO Grad-type transport-hyperbolicity loss
for a Cai-Fan-Li operator-projection to repair (see docs/audit/BHR_wellposedness_2026-06-30.md). The
ONLY well-posedness question is whether M_stream -- the matrix multiplying the derivative q d/dq --
has a real, diagonalizable spectrum (the characteristic q-advection speeds). These tests prove it
does, for ell_max up to 8, so the operator is ALREADY hyperbolic in q and B.HR / CFL is unnecessary.

The M_angle SOURCE matrix (zeroth order, multiplies no derivative) is NOT a hyperbolicity object --
the gate keys only on M_stream. But (external audit 2026-06-30) it is also NOT a "harmless bounded
oscillation": its eigenvalues carry a measured POSITIVE real-part growth spectrum (max Re ~ +0.27 --
an F_ell mode then grows at rate 1.5*Sigma_H*Re; pure-real growth at ell_max=2, oscillatory imaginary
parts only at ell_max>=4).
It is a genuine shear-driven growth source; the disambiguation pinned below records BOTH the growth
(real part) and the oscillation (imag part) so the source is never mistaken for a transport
ill-posedness AND never mislabelled as non-growing.
"""

from __future__ import annotations

import numpy as np
import pytest

from rabbit.config.grids import MomentumGrid
from rabbit.transport.stageAB_state import AxisymmetricHierarchyState
from rabbit.transport.typeI_even_ladder_hierarchy import (
    active_even_ells,
    compute_hierarchy_rhs_typeI_even_ladder,
)
from rabbit.transport.typeI_stageA_hierarchy import q_d_dq
from rabbit.transport.even_ladder_wellposedness import (
    advection_eigenvalues,
    angle_source_eigenvalues,
    build_angle_matrix,
    build_streaming_matrix,
    q_advection_is_hyperbolic,
    spectrum_is_real_and_diagonalizable,
)


@pytest.mark.parametrize("lmax", [2, 4, 6, 8])
def test_streaming_matrix_real_spectrum_and_diagonalizable(lmax):
    """M_stream (the q d/dq coupling) has a REAL spectrum + non-defective eigenbasis -> the
    even-ladder q-advection is hyperbolic. This is the proof that B.HR / CFL is unnecessary."""
    active = active_even_ells(lmax)
    M = build_streaming_matrix(active)
    w, V = np.linalg.eig(M)
    assert np.max(np.abs(w.imag)) < 1e-12, f"complex characteristic speeds at lmax={lmax}: {w}"
    assert np.linalg.cond(V) < 50.0, f"defective/ill-conditioned eigenbasis at lmax={lmax}"
    assert q_advection_is_hyperbolic(active)
    # genuine, distinct propagation speeds (not a degenerate near-zero operator)
    assert np.ptp(w.real) > 0.5
    # advection_eigenvalues is the public accessor and agrees with the matrix spectrum
    np.testing.assert_allclose(np.sort(advection_eigenvalues(active).real), np.sort(w.real),
                               rtol=1e-12, atol=1e-14)


@pytest.mark.parametrize("lmax", [0, 2, 4, 6, 8])
def test_streaming_and_angle_matrices_reproduce_the_live_operator(lmax):
    """ANTI-DRIFT: the assembled M_stream / M_angle EXACTLY reproduce what the live RHS applies,
    so the eigenvalue verdict is about the real operator, not a paraphrase. With a separable
    F_ell(q) = c_ell g(q): RHS_ell = Sigma_H (M_stream @ c)_ell q_d_dq(g) + 1.5 Sigma_H (M_angle @ c)_ell g.

    lmax=0 exercises the degenerate single-multipole (no F_2, has_f2=False) branch where both
    matrices collapse to [[0]] and the RHS is identically zero."""
    active = active_even_ells(lmax)
    n_ell = len(active)
    grid = MomentumGrid(N_q=16)
    g = np.exp(-0.3 * grid.nodes) * (1.0 + grid.nodes ** 2)  # arbitrary non-trivial profile
    qddq_g = q_d_dq(g, grid.nodes)
    base_c = np.array([0.7, -0.4, 0.9, 1.3, -1.1])[:n_ell]
    n_species = 2
    data = np.empty((n_species, n_ell, grid.N_q))
    for s in range(n_species):
        for i in range(n_ell):
            data[s, i, :] = (base_c[i] + 0.1 * s) * g
    state = AxisymmetricHierarchyState(data=data, grid=grid, active_ells=active)

    Sigma_H = 0.137
    rhs = compute_hierarchy_rhs_typeI_even_ladder(state, Sigma_H).reshape(n_species, n_ell, grid.N_q)

    Ms = build_streaming_matrix(active)
    Ma = build_angle_matrix(active)
    for s in range(n_species):
        cs = base_c + 0.1 * s
        stream_coeff = Ms @ cs
        angle_coeff = Ma @ cs
        for i in range(n_ell):
            expected = Sigma_H * stream_coeff[i] * qddq_g + 1.5 * Sigma_H * angle_coeff[i] * g
            np.testing.assert_allclose(
                rhs[s, i, :], expected, rtol=1e-12, atol=1e-14,
                err_msg=f"matrix assembly != live RHS at lmax={lmax} s={s} ell={active[i]}")
    if lmax == 0:
        # degenerate branch: collapses to a single zero row, RHS must be identically zero
        assert Ms.shape == (1, 1) and Ms[0, 0] == 0.0
        np.testing.assert_array_equal(rhs, 0.0)


def test_gate_detects_non_hyperbolic_matrices():
    """NEGATIVE test for the gate's detection logic: the matrix-level predicate must return False
    for a complex-spectrum matrix AND for a defective (non-diagonalizable) one, and True for a real
    diagonalizable matrix. Without this, a regression that hard-wires the gate to True (or inverts a
    tolerance / swaps `and`) would pass every positive-input test silently."""
    rotation = np.array([[0.0, -1.0], [1.0, 0.0]])     # eigenvalues +-i -> complex spectrum
    assert not spectrum_is_real_and_diagonalizable(rotation)
    defective = np.array([[1.0, 1.0], [0.0, 1.0]])     # real spectrum {1,1} but Jordan-defective
    assert not spectrum_is_real_and_diagonalizable(defective)
    diagonalizable = np.array([[2.0, 1.0], [0.0, 3.0]])  # real, distinct eigenvalues -> hyperbolic
    assert spectrum_is_real_and_diagonalizable(diagonalizable)
    # the active_ells gate delegates to the same predicate and stays True for the real operator
    assert q_advection_is_hyperbolic(active_even_ells(8))


@pytest.mark.parametrize("lmax", [2, 4, 6, 8, 10, 12])
def test_angle_source_is_a_growth_source_not_harmless_oscillation(lmax):
    """The zeroth-order algebraic source M_angle multiplies NO derivative, so it is NOT a
    hyperbolicity object -- the gate keys ONLY on M_stream and transport stays well-posed. But
    (external audit 2026-06-30) M_angle is ALSO not a "harmless bounded oscillation": its spectrum
    carries a MEASURED POSITIVE real-part growth rate. This pins BOTH facts so the source is never
    mistaken for a transport ill-posedness AND never mislabelled as non-growing.

    Non-vacuity of the real-part assertion: a purely oscillatory source has Re(eig)=0 (cf. the
    rotation matrix [[0,-1],[1,0]] in test_gate_detects_non_hyperbolic_matrices). M_angle's strictly
    positive max Re is therefore a real discriminator, not a tautology."""
    active = active_even_ells(lmax)
    wa = angle_source_eigenvalues(active)

    # HEADLINE (the audit fix): M_angle is a GROWTH source -- max real part is strictly positive,
    # ~+0.27 in units of 1.5 Sigma_H across all ell_max (NOT zero, as a pure oscillation would be).
    max_re = float(wa.real.max())
    assert 0.2 < max_re < 0.35, (
        f"M_angle growth rate (max Re eig) out of measured band at lmax={lmax}: {max_re:+.4f}")
    # the single source-free F_0 row contributes an exact zero mode; no NEGATIVE (decaying) mode
    assert float(wa.real.min()) > -1e-12, f"unexpected decaying mode at lmax={lmax}: {wa.real.min()}"

    # SECONDARY feature: the oscillatory imaginary part is lmax-dependent -- absent at lmax=2
    # (pure real growth), present and growing at lmax>=4. So "oscillation" is not even the whole
    # story at low ell_max; the growth is.
    max_im = float(np.abs(wa.imag).max())
    if lmax == 2:
        assert max_im < 1e-12, f"unexpected oscillation at lmax=2 (should be pure real): {max_im}"
    else:
        assert max_im > 0.1, f"expected oscillatory source eigenvalues at lmax={lmax}"

    assert q_advection_is_hyperbolic(active)  # ... yet transport (principal symbol) is still hyperbolic


def test_build_matrices_reject_invalid_ladders():
    """The ladder must be the contiguous even sequence 0,2,...,lmax (matches the live operator's
    active_ells contract) -- silent acceptance of a malformed ladder would fabricate a wrong M."""
    for bad in [(0, 1, 2), (2, 4), (0, 4), (0, 2, 6), (0, 2, 4, 8), (), (2,)]:
        with pytest.raises(ValueError):
            build_streaming_matrix(bad)
        with pytest.raises(ValueError):
            build_angle_matrix(bad)
