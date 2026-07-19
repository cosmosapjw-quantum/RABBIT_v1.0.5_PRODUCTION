import numpy as np
from scipy.special import roots_laguerre

from rabbit.collisions.nu_nu_scattering import (
    nunu_diagonal_monopole_collision,
    energy_conservation_residual,
)


def _fd(q, T=1.0):
    x = q / T
    return 1.0 / (np.exp(np.minimum(x, 500.0)) + 1.0)


def test_nunu_diagonal_zero_for_identical_species():
    q_gl_np, q_wgl_np = roots_laguerre(20)
    q = q_gl_np.astype(np.float64)

    f = _fd(q, 1.0)
    C = nunu_diagonal_monopole_collision(
        {"nue": f, "nuebar": f, "nux": f},
        T_nu_e=0.0591,
        T_nu_x=0.0591,
        gamma=0.1,
    )

    for arr in C.values():
        assert np.max(np.abs(arr)) < 1e-15


def test_nunu_diagonal_energy_conserving():
    q_gl_np, q_wgl_np = roots_laguerre(20)
    q = q_gl_np.astype(np.float64)
    w = q_wgl_np.astype(np.float64)

    f0 = _fd(q, 1.0)
    f1 = f0 * (1.0 - 0.01 * np.exp(-q / 3.0))
    f2 = f0 * (1.0 + 0.02 * np.exp(-q / 4.0))
    f3 = f0 * (1.0 - 0.005 * np.exp(-q / 2.0))

    C = nunu_diagonal_monopole_collision(
        {"nue": f1, "nuebar": f2, "nux": f3},
        T_nu_e=0.0591,
        T_nu_x=0.0589,
        gamma=0.1,
    )

    resid = energy_conservation_residual(C, q, w)
    assert abs(resid) < 1e-12
