"""rabbit.collisions.pstf_pi3_analytic — exact analytic Pi^(3) via the CAS artifact (plan B.F3).

Loads the CAS-generated polynomial functions (``_pstf_pi3_generated.py``, emitted by
``scripts/cas/gen_pstf_pi3.py --emit-module``) and exposes the generalized HM angular
polynomial Pi^(3)(x,y) as an EXACT closed form -- replacing the numerical theta-quadrature
evaluator (:func:`pstf_body_frame.angular_polynomial_n3`) where a generated case exists. The
two are cross-checked to machine precision by tests/test_pstf_pi3_cas.py.
"""

from __future__ import annotations

import math

from rabbit.collisions._pstf_pi3_generated import PI3_FUNCS, PI3_XCOEFFS


def has_analytic_pi3(L, J, ranks, legs) -> bool:
    """Whether a CAS-generated Pi^(3) exists for this (L, J, ranks, legs)."""
    return (L, J, tuple(ranks), tuple(legs)) in PI3_FUNCS


def analytic_pi3_xcoeffs(L, J, ranks, legs):
    """The Pi^(3) x-polynomial coefficient function: ``fn(y, p1, p2, p3, p4) -> (A_0, A_1, ...)``
    ascending (A[n] = coeff of x^n), for the closed-form x-integral in reduced_kernel_F3.

    Returns None if the case was not generated.
    """
    return PI3_XCOEFFS.get((L, J, tuple(ranks), tuple(legs)))


def analytic_pi3(L, J, ranks, legs, x, y, p1, p2, p3, m4: float = 0.0, *, p4=None) -> float:
    """Exact Pi^(3)(x,y) from the CAS-generated polynomial (SCALAR x, y).

    Leg-4 on-shell closure: by default MASSLESS legs 1-3 (E_i = p_i), so E4 = p1+p2-p3 and
    p4 = sqrt(E4^2 - m4^2). For MASSIVE legs 1-3 (E_i != p_i) that default is wrong -- pass the
    correct ``p4`` explicitly (computed by the caller from E4 = E1+E2-E3 of the actual energies);
    `p1,p2,p3` are always the momenta the body frame uses. Raises KeyError if the case was not
    generated.

    Scalar-only: the generated polynomials include constant cases (e.g. all-monopole -> 1) that
    return a scalar regardless of x, so this is NOT array-vectorized -- the downstream kernel
    wiring (reduced_kernel_F3) evaluates per quadrature point.
    """
    key = (L, J, tuple(ranks), tuple(legs))
    func = PI3_FUNCS.get(key)
    if func is None:
        raise KeyError(
            f"no CAS-generated Pi^(3) for {key}; add it to gen_pstf_pi3._GATE_CASES and "
            f"regenerate (python3 scripts/cas/gen_pstf_pi3.py --emit-module)")
    if p4 is None:
        e4 = p1 + p2 - p3                       # massless legs 1-3: E4 = p1+p2-p3
        p4 = math.sqrt(e4 * e4 - m4 * m4) if m4 else e4
    return float(func(x, y, p1, p2, p3, float(p4)))
