"""rabbit.geometry.constraints — Hamiltonian + momentum constraint monitors.

The Friedmann constraint (Hamiltonian, G^0_0) is algebraic.  The momentum
constraint (G^0_i) is similarly algebraic in Hubble-normalized variables
and serves as the primary diagnostic for the consistency of any tilted
Bianchi model: if G^0_i + (rho+p) gamma^2 v_i = 0 fails, the tilt is not
in equilibrium with the geometry/heat-flux source.

Foundation block F0 of the v1.0.5 -> v2.0.0 upgrade plan.  Required by
every tilted non-Type-I cell PR (waves 1-7).

Type I:     |1 - Omega - Sigma^2| ~ 0 by construction.
Class A:    |1 - Omega - Sigma^2 - K|.
Class B:    |1 - Omega - Sigma^2 - K - c A^2|, c from classB_c_factor.

Momentum:   |G^0_i_geometry + (rho+p) gamma^2 v_i + psi_i_dipole| < tol.
            For orthogonal models (v == 0, no dipole) this is trivially 0.
"""

from __future__ import annotations

from typing import Sequence, Tuple

import numpy as np


# ═══════════════════════════════════════════════════════════════════════
# §1. Hamiltonian (Friedmann) residuals
# ═══════════════════════════════════════════════════════════════════════

def friedmann_residual_typeI(
    Sigma_plus: float,
    Sigma_minus: float,
) -> float:
    """Friedmann constraint residual for Type I.

    Residual = |1 - Omega - Sigma^2| where Omega = 1 - Sigma^2.
    Identically zero by construction; serves as a roundtrip consistency check.
    """
    Sigma_sq = Sigma_plus**2 + Sigma_minus**2
    Omega = max(0.0, 1.0 - Sigma_sq)
    return abs(1.0 - Omega - Sigma_sq)


def curvature_K_classA(N1: float, N2: float, N3: float) -> float:
    """Class A normalized curvature

    K = (1/12)(N1^2 + N2^2 + N3^2 - 2 N1 N2 - 2 N2 N3 - 2 N3 N1).

    Type I: N1 = N2 = N3 = 0 -> K = 0.
    """
    return (1.0 / 12.0) * (
        N1**2 + N2**2 + N3**2
        - 2.0 * N1 * N2 - 2.0 * N2 * N3 - 2.0 * N3 * N1
    )


def friedmann_residual_classA(
    Sigma_plus: float,
    Sigma_minus: float,
    N1: float,
    N2: float,
    N3: float,
    Omega: float,
) -> float:
    """Friedmann constraint residual for Class A: |1 - Omega - Sigma^2 - K|."""
    Sigma_sq = Sigma_plus**2 + Sigma_minus**2
    K = curvature_K_classA(N1, N2, N3)
    return abs(1.0 - Omega - Sigma_sq - K)


def friedmann_residual_classB(
    Sigma_plus: float,
    Sigma_minus: float,
    N1: float,
    N2: float,
    N3: float,
    A: float,
    Omega: float,
    c_factor: float = 3.0,
) -> float:
    """Friedmann constraint residual for Class B.

    Residual = |1 - Omega - Sigma^2 - K - c A^2|

    where c = 3 for V, IV, III, VI_h, VII_h and c = 15/4 for VI_{-1/9}
    (use ``rabbit.jax.geometry_bianchi_base.classB_c_factor`` to fetch
    the right coefficient for a given type).

    F0 contract: every Class B driver computes this residual at every
    accepted step and stores ``max`` into
    ``BBNPrediction.metadata["constraint_residuals"]["friedmann"]``.
    """
    Sigma_sq = Sigma_plus**2 + Sigma_minus**2
    K = curvature_K_classA(N1, N2, N3)  # The N_i-derived part is shared.
    return abs(1.0 - Omega - Sigma_sq - K - c_factor * A * A)


def friedmann_residual_general(
    Sigma_plus: float,
    Sigma_minus: float,
    N1: float,
    N2: float,
    N3: float,
    A: float,
    Omega: float,
    bianchi_type_str: str,
) -> float:
    """Dispatching wrapper that picks the right residual for any Bianchi type.

    Cell drivers can call this without knowing whether their type is Class
    A or Class B, eliminating per-driver duplication.
    """
    # Local import to avoid circular dependency at module load (geometry
    # consumers may import constraints before geometry_bianchi_base is ready).
    from rabbit.jax.geometry_bianchi_base import (
        classB_c_factor,
        is_class_A,
        is_class_B,
    )

    if is_class_A(bianchi_type_str):
        return friedmann_residual_classA(
            Sigma_plus, Sigma_minus, N1, N2, N3, Omega
        )
    if is_class_B(bianchi_type_str):
        c = classB_c_factor(bianchi_type_str)
        return friedmann_residual_classB(
            Sigma_plus, Sigma_minus, N1, N2, N3, A, Omega, c_factor=c
        )
    raise ValueError(f"Unknown Bianchi type for Friedmann residual: {bianchi_type_str!r}")


# ═══════════════════════════════════════════════════════════════════════
# §2. Momentum constraint G^0_i monitor (foundation F0)
# ═══════════════════════════════════════════════════════════════════════

GAMMA_RAD_DEFAULT = 4.0 / 3.0


def _gamma_lorentz_sq(v_vec: Sequence[float]) -> float:
    v_sq = float(v_vec[0]) ** 2 + float(v_vec[1]) ** 2 + float(v_vec[2]) ** 2
    return 1.0 / max(1.0 - v_sq, 1e-30)


def momentum_matter_source(
    v_vec: Sequence[float],
    rho_plus_p_over_3H2: float,
    psi_dipole: Sequence[float] = (0.0, 0.0, 0.0),
) -> Tuple[float, float, float]:
    """Hubble-normalized matter-side contribution to G^0_i.

    Returns
    -------
    (m1, m2, m3) such that

        m_i = (rho + p) * gamma^2 * v_i / (3 H^2) + psi_dipole_i

    where ``rho_plus_p_over_3H2`` is the Hubble-normalized enthalpy.

    For radiation: rho + p = (4/3) rho.  For pressureless matter: rho.
    psi_dipole is the radiation-distribution dipole moment Psi_(1)_i (the
    PSTF ell=1 mode, when an anisotropic transport hierarchy is solved);
    pass zeros when no dipole is being tracked.
    """
    g2 = _gamma_lorentz_sq(v_vec)
    coupling = rho_plus_p_over_3H2 * g2
    m = (
        coupling * float(v_vec[0]) + float(psi_dipole[0]),
        coupling * float(v_vec[1]) + float(psi_dipole[1]),
        coupling * float(v_vec[2]) + float(psi_dipole[2]),
    )
    return m


def momentum_geometry_source_classA(
    Sigma_plus: float,
    Sigma_minus: float,
    N1: float,
    N2: float,
    N3: float,
) -> Tuple[float, float, float]:
    """Hubble-normalized geometry contribution to G^0_i for Class A.

    For an orthogonal Class A model with hypersurface-orthogonal initial
    data, G^0_i = 0 identically; numerical drift then shows up as a
    nonzero residual.  This routine returns the *expected* zero plus the
    structure-constant-driven kinematic source that can lift it away
    from zero in tilted/anisotropic configurations:

        s_i = sum_jk eps_ijk N^jl Sigma_lk * (small)

    For F0, the orthogonal contribution is taken to vanish (returns
    zeros) and the function is supplied as a hook that future per-cell
    PRs can extend with explicit Class-specific algebra.
    """
    # Orthogonal Class A contribution is identically zero in canonical
    # Wainwright-Hsu variables; the per-cell PRs that introduce
    # non-trivial source terms (tilted VIII, IX, etc.) override this.
    del Sigma_plus, Sigma_minus, N1, N2, N3
    return (0.0, 0.0, 0.0)


def momentum_geometry_source_classB(
    Sigma_plus: float,
    Sigma_minus: float,
    N1: float,
    N2: float,
    N3: float,
    A: float,
) -> Tuple[float, float, float]:
    """Hubble-normalized geometry contribution to G^0_i for Class B.

    Class B carries the frame vector a_a = (A, 0, 0) (in canonical
    Wainwright-Hsu LRS frames); the geometry-side momentum constraint
    contribution starts as

        s_1 = -3 A * Sigma_plus + (frame-vector cross-coupling),
        s_2 = +sqrt(3) A * Sigma_minus,
        s_3 = +sqrt(3) A * Sigma_minus

    For pure orthogonal types V, IV (LRS) the residual is identically
    zero; for III, VI_h, VII_h, VI_{-1/9} the additional
    N_i-cross-coupling terms enter and per-cell PRs override the trivial
    return below.
    """
    # Minimal LRS Class B (V, IV) contribution.
    s1 = -3.0 * A * Sigma_plus
    s2 = float(np.sqrt(3.0)) * A * Sigma_minus
    s3 = -float(np.sqrt(3.0)) * A * Sigma_minus
    # The N_i cross-couplings for III, VI_h, VII_h, VI_{-1/9} are added in
    # per-cell PRs; consume them silently here.
    del N1, N2, N3
    return (s1, s2, s3)


def momentum_geometry_source_general(
    Sigma_plus: float,
    Sigma_minus: float,
    N1: float,
    N2: float,
    N3: float,
    A: float,
    bianchi_type_str: str,
) -> Tuple[float, float, float]:
    """Dispatching geometry-side contribution to the momentum constraint."""
    from rabbit.jax.geometry_bianchi_base import is_class_A, is_class_B

    if is_class_A(bianchi_type_str):
        return momentum_geometry_source_classA(Sigma_plus, Sigma_minus, N1, N2, N3)
    if is_class_B(bianchi_type_str):
        return momentum_geometry_source_classB(Sigma_plus, Sigma_minus, N1, N2, N3, A)
    raise ValueError(f"Unknown Bianchi type for momentum source: {bianchi_type_str!r}")


def required_momentum_dipole(
    Sigma_plus: float,
    Sigma_minus: float,
    N1: float,
    N2: float,
    N3: float,
    A: float,
    bianchi_type_str: str,
    v_vec: Sequence[float] = (0.0, 0.0, 0.0),
    rho_plus_p_over_3H2: float = GAMMA_RAD_DEFAULT * (1.0 / 3.0),
) -> Tuple[float, float, float]:
    """Return the heat-flux/dipole vector that closes ``G^0_i``.

    The momentum constraint has the form

        s_i(geometry) + (rho+p) Gamma^2 v_i / (3H^2) + psi_i = 0.

    This helper returns the algebraic ``psi_i`` required by that equation.
    It is not an evolution law; drivers that use it should label the closure
    as algebraic unless they also evolve a heat-flux transport state.
    """
    s = momentum_geometry_source_general(
        Sigma_plus, Sigma_minus, N1, N2, N3, A, bianchi_type_str
    )
    m = momentum_matter_source(v_vec, rho_plus_p_over_3H2, psi_dipole=(0.0, 0.0, 0.0))
    return (-(s[0] + m[0]), -(s[1] + m[1]), -(s[2] + m[2]))


def momentum_residual(
    Sigma_plus: float,
    Sigma_minus: float,
    N1: float,
    N2: float,
    N3: float,
    A: float,
    bianchi_type_str: str,
    v_vec: Sequence[float] = (0.0, 0.0, 0.0),
    rho_plus_p_over_3H2: float = GAMMA_RAD_DEFAULT * (1.0 / 3.0),
    psi_dipole: Sequence[float] = (0.0, 0.0, 0.0),
) -> float:
    """Hubble-normalized momentum constraint magnitude.

    Returns ``max_i |s_i_geometry + (rho+p) gamma^2 v_i + psi_dipole_i|``.

    Orthogonal models (v == 0, no dipole, hypersurface-orthogonal IC) yield
    zero for Class A I, II, VI0, VII0, VIII, IX and Class B V, IV.

    Tilted models must satisfy this residual at every step; drivers gate
    on ``residual < tol_M`` (default 1e-10 over the trajectory) before
    flipping a cell's tier from candidate to canonical.

    The function is plain-Python / numpy and is jit-compatible only when
    its inputs are jnp scalars; jax-traced drivers should re-implement
    the body using jnp primitives, but this is the reference oracle and
    the test contract.
    """
    s = momentum_geometry_source_general(
        Sigma_plus, Sigma_minus, N1, N2, N3, A, bianchi_type_str
    )
    m = momentum_matter_source(v_vec, rho_plus_p_over_3H2, psi_dipole)
    diffs = (s[0] + m[0], s[1] + m[1], s[2] + m[2])
    return float(max(abs(diffs[0]), abs(diffs[1]), abs(diffs[2])))


__all__ = [
    "curvature_K_classA",
    "friedmann_residual_typeI",
    "friedmann_residual_classA",
    "friedmann_residual_classB",
    "friedmann_residual_general",
    "momentum_matter_source",
    "momentum_geometry_source_classA",
    "momentum_geometry_source_classB",
    "momentum_geometry_source_general",
    "required_momentum_dipole",
    "momentum_residual",
    "GAMMA_RAD_DEFAULT",
]
