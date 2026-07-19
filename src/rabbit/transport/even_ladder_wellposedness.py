"""rabbit.transport.even_ladder_wellposedness — hyperbolicity gate for the even-ladder transport.

The collisionless Type-I even-ell shear-advection RHS
(:func:`rabbit.transport.typeI_even_ladder_hierarchy.compute_hierarchy_rhs_typeI_even_ladder`) is

    RHS = Sigma_H [ M_stream @ (q d/dq) F + 1.5 M_angle @ F ],

where ``F`` is the stacked multipole vector ``(F_0, F_2, ..., F_lmax)`` and ``M_stream`` /
``M_angle`` are CONSTANT (state- and q-independent) pentadiagonal-in-ell matrices assembled from the
A/B/C (and tilde) Legendre-recurrence coefficients already in ``typeI_even_ladder_hierarchy``.

Why there is no Cai-Fan-Li regularization to apply here
-------------------------------------------------------
A Cai-Fan-Li operator-projection repairs a Grad moment system whose SPATIAL flux Jacobian dF/dU
loses real eigenvalues near equilibrium. Bianchi I is spatially homogeneous (D_a = omega_ab = 0):
there is no spatial transport derivative and hence no spatial flux Jacobian to lose hyperbolicity.
The only derivative in the operator is the shared momentum-magnitude advection ``q d/dq``, and the
``M_stream`` matrix that multiplies it is constant. So well-posedness is the STATIC question "does
M_stream have a real, diagonalizable spectrum" (real characteristic q-advection speeds), answerable
once per ``ell_max`` -- not a runtime projection. Measured: M_stream is real-spectrum and
well-conditioned for ell_max up to 12, so the operator is already hyperbolic in q and no
regularization is needed (see ``docs/audit/BHR_wellposedness_2026-06-30.md``).

The ``M_angle`` matrix is the ZEROTH-order algebraic source (it multiplies no derivative). It is
NOT a hyperbolicity object -- the principal symbol is ``M_stream @ (q d/dq)`` only -- so the gate
below keys ONLY on ``M_stream`` and callers must not "regularize" the ``M_angle`` spectrum. But it
is also NOT a "harmless bounded oscillation": its eigenvalues carry a measured POSITIVE real-part
growth spectrum (max Re ~ +0.27 across ell_max 2..12 -- an F_ell mode then grows at rate
1.5*Sigma_H*Re; pure-real growth at ell_max 2, with oscillatory imaginary parts appearing only at
ell_max >= 4). ``M_angle`` is a
genuine shear-driven growth source; its amplitude stays bounded in practice because the logit
realizability lock caps the A_modes (B5/AP62), not because the source is non-growing. See
:func:`angle_source_eigenvalues` and ``docs/audit/BHR_wellposedness_2026-06-30.md``.

This module is the reusable well-posedness gate for the transport-wired stages (B2/B4/B5): assert
:func:`q_advection_is_hyperbolic` before trusting a new/extended even-ladder operator.
"""

from __future__ import annotations

from typing import Sequence, Tuple

import numpy as np

from rabbit.transport.typeI_even_ladder_hierarchy import (
    A_coeff,
    Atilde_coeff,
    B_coeff,
    Btilde_coeff,
    C_coeff,
    Ctilde_coeff,
)

#: F_0 couples only to F_2 with this factor (compute_hierarchy_rhs_typeI_even_ladder line 62:
#: ``rhs[..ell 0..] = (Sigma_H / 5) q_d_dq(F_2)``).
_F0_TO_F2_STREAM: float = 1.0 / 5.0


def _validate_active(active_ells: Sequence[int]) -> Tuple[int, ...]:
    """The ladder must be the contiguous even sequence 0,2,...,lmax the live operator requires."""
    active = tuple(int(e) for e in active_ells)
    if not active or active[0] != 0 or any(e % 2 != 0 for e in active):
        raise ValueError(f"active_ells must be even and start at 0, got {active}")
    if list(active) != list(range(0, active[-1] + 2, 2)):
        raise ValueError(f"active_ells must be contiguous even 0,2,...,lmax, got {active}")
    return active


def build_streaming_matrix(active_ells: Sequence[int]) -> np.ndarray:
    """Constant ell-coupling matrix ``M_stream`` that multiplies ``q d/dq`` in the even-ladder RHS.

    Mirrors :func:`compute_hierarchy_rhs_typeI_even_ladder` EXACTLY: the F_0 row couples to F_2 with
    1/5; each F_ell (ell >= 2) row couples ``A_coeff(ell)`` to F_{ell-2}, ``B_coeff(ell)`` to F_ell,
    ``C_coeff(ell)`` to F_{ell+2}, with the F_{lmax+2}=0 closure dropping the top ell+2 term.
    """
    active = _validate_active(active_ells)
    idx = {e: i for i, e in enumerate(active)}
    n = len(active)
    M = np.zeros((n, n))
    if 2 in idx:
        M[idx[0], idx[2]] = _F0_TO_F2_STREAM
    for e in active[1:]:
        if e - 2 in idx:
            M[idx[e], idx[e - 2]] = A_coeff(e)
        M[idx[e], idx[e]] = B_coeff(e)
        if e + 2 in idx:
            M[idx[e], idx[e + 2]] = C_coeff(e)
    return M


def build_angle_matrix(active_ells: Sequence[int]) -> np.ndarray:
    """Constant ell-coupling matrix ``M_angle`` of the zeroth-order ``1.5 Sigma_H`` algebraic source.

    Mirrors the ``angle`` accumulation in :func:`compute_hierarchy_rhs_typeI_even_ladder`: each
    F_ell (ell >= 2) row couples ``-Atilde_coeff(ell)`` to F_{ell-2}, ``Btilde_coeff(ell)`` to F_ell,
    ``Ctilde_coeff(ell)`` to F_{ell+2}; the F_0 row has no source term.
    """
    active = _validate_active(active_ells)
    idx = {e: i for i, e in enumerate(active)}
    n = len(active)
    M = np.zeros((n, n))
    for e in active[1:]:
        if e - 2 in idx:
            M[idx[e], idx[e - 2]] = -Atilde_coeff(e)
        M[idx[e], idx[e]] = Btilde_coeff(e)
        if e + 2 in idx:
            M[idx[e], idx[e + 2]] = Ctilde_coeff(e)
    return M


def advection_eigenvalues(active_ells: Sequence[int]) -> np.ndarray:
    """Characteristic q-advection speeds (in units of Sigma_H): the eigenvalues of ``M_stream``.

    Returns the raw (possibly complex-typed) eigenvalue array; for a hyperbolic operator the
    imaginary parts are ~0 and ``.real`` are the propagation speeds.
    """
    return np.linalg.eigvals(build_streaming_matrix(active_ells))


def angle_source_eigenvalues(active_ells: Sequence[int]) -> np.ndarray:
    """Eigenvalues of the ZEROTH-order ``M_angle`` algebraic source.

    These are dimensionless matrix eigenvalues. An F_ell eigenmode with eigenvalue ``lam`` evolves
    in the RHS as ``dF/dt ~ 1.5 * Sigma_H * lam * F`` -- i.e. its growth/oscillation rate is
    ``1.5 * Sigma_H * lam`` (compare :func:`advection_eigenvalues`, whose M_stream eigenvalues give
    advection speeds ``Sigma_H * lam``). Unlike advection speeds these are NOT propagation speeds --
    ``M_angle`` multiplies
    no derivative, so its spectrum governs the *growth/rotation of the source itself*. The honest
    diagnostic the external audit (2026-06-30) asked for: the **real parts** are a measured growth
    rate (max Re ~ +0.27 across ell_max 2..12, with a single zero mode from the source-free F_0 row),
    NOT zero -- so ``M_angle`` is a genuine growth source, not a purely oscillatory one. The imaginary
    parts (which appear only at ell_max >= 4 and grow with ell_max) are a secondary oscillatory
    feature layered on that growth. The source amplitude stays bounded in practice because the logit
    realizability lock caps the A_modes (B5/AP62), not because ``M_angle`` is non-growing.
    """
    return np.linalg.eigvals(build_angle_matrix(active_ells))


def spectrum_is_real_and_diagonalizable(M: np.ndarray, *, imag_tol: float = 1e-9,
                                        cond_tol: float = 1e8) -> bool:
    """True iff ``M`` has a real spectrum AND a non-defective eigenbasis (the hyperbolicity test).

    Matrix-level predicate so the gate logic is testable against deliberately non-hyperbolic inputs
    (complex-spectrum and defective-Jordan matrices), independent of which ``active_ells`` happen to
    produce. ``cond_tol`` is a generous *defectiveness* ceiling: a non-diagonalizable (defective)
    matrix sends ``cond(eigenvectors) -> inf``, so 1e8 cleanly separates "diagonalizable" from
    "defective" at double precision. It is NOT a per-operator conditioning-quality bar -- a caller
    wanting a tighter well-conditioning guarantee for a specific operator should assert ``cond`` of
    :func:`build_streaming_matrix`'s eigenbasis directly (the test pins it at < 50 for this operator).
    """
    w, V = np.linalg.eig(M)
    scale = max(1.0, float(np.max(np.abs(w.real))))
    real_spectrum = bool(np.max(np.abs(w.imag)) <= imag_tol * scale)
    well_conditioned = bool(np.linalg.cond(V) < cond_tol)
    return real_spectrum and well_conditioned


def q_advection_is_hyperbolic(active_ells: Sequence[int], *, imag_tol: float = 1e-9,
                              cond_tol: float = 1e8) -> bool:
    """True iff ``M_stream`` has a real spectrum AND a non-defective (well-conditioned) eigenbasis.

    This is the precise well-posedness condition for the even-ladder q-advection: real characteristic
    speeds + a diagonalizable eigenbasis => the method-of-characteristics upwind is well-defined and
    the moment system is hyperbolic in q. A ``False`` result is the ONLY thing that could motivate a
    symmetrization of M_stream (a static similarity transform) -- NOT a Cai-Fan-Li spatial-flux
    projection, which has no target in a spatially-homogeneous Bianchi I hierarchy.
    """
    return spectrum_is_real_and_diagonalizable(
        build_streaming_matrix(active_ells), imag_tol=imag_tol, cond_tol=cond_tol)
