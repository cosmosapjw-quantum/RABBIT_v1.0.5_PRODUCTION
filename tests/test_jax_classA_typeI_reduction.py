"""Type I reduction — covered by Gate 1 in test_jax_classA_geometry.py.

This file retained for backward compatibility. The canonical tests
are in TestGate1TypeIReduction.
"""
from rabbit.config.conventions import BianchiType
from rabbit.geometry.typeI import compute_typeI_geometry_rhs
from rabbit.jax.geometry_classA_jax import TYPE_MASKS, compute_classA_geometry_rhs_jax


def test_classA_typeI_reduction_matches_typeI_geometry_rhs():
    Sp, Sm, pip, pim = 0.09, -0.02, 1.5e-3, -7.0e-4
    Omega = 1.0 - Sp**2 - Sm**2
    dSp_ref, dSm_ref = compute_typeI_geometry_rhs(Sp, Sm, pip, pim, Omega)
    out = compute_classA_geometry_rhs_jax(
        Sp, Sm, 0.0, 0.0, 0.0,
        pi_shear_plus=pip, pi_shear_minus=pim,
        type_mask=TYPE_MASKS[BianchiType.TYPE_I])
    assert abs(out.dSigma_plus_dN - dSp_ref) < 1e-12
    assert abs(out.dSigma_minus_dN - dSm_ref) < 1e-12
