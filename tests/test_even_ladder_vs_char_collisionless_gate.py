"""tests/test_even_ladder_vs_char_collisionless_gate.py — PR#15 (B1-B2): the moment hierarchy vs
the characteristic-ray EXACT collisionless reference at nonzero shear.

DECISIVE NEGATIVE RESULT (the W2-PR6 root cause). The even-ladder moment hierarchy is ell-SATURATED
at lmax=2: lmax = 2,4,6,8 agree to < 0.2% in the energy-density anisotropy delta-rho = rho_nu/rho_FD
- 1. Yet the saturated value captures only ~28% of the EXACT characteristic-ray collisionless
delta-rho at Sigma_H in {0.1, 0.3}. **Adding ell moments does NOT close the gap** -- the ~70%
shortfall is STRUCTURAL, not a truncation artifact: the even-ladder monopole source is the LINEAR
term (Sigma_H/5) q d/dq F_2, whereas the exact characteristic monopole is the NONLINEAR energy shift
f0(q e^{2 I}) accumulated over the full ray history (characteristic_rays_jax.extract_monopole_jax).
delta-rho is a SECOND-order-in-shear quantity (its O(S) part vanishes by angular parity,
sum_j w_j P2(mu_j) = 0, so delta-rho ~ S^2 -- verified: delta-rho/S^2 is ~constant), while the
even-ladder is exact only to LINEAR order in accumulated shear (rhs_typeI.py). Hence the capture has
a FINITE S->0 limit ~0.27 (!= 1): the linearized hierarchy gets only ~27% of the leading O(S^2)
energy anisotropy, and no ell moment supplies the rest. (f32-vs-f64 invariant; both reviewers
verified the setup is apples-to-apples and the result is real physics, not a normalization artifact.)

This file therefore does NOT assert "higher ell recovers the shear response" (that is FALSE). It
PINS the structural shortfall as a regression gate -- confirming, not fixing, W2-PR6 -- and certifies
the characteristic-ray path as the shear-faithful reference. See
docs/audit/B2_even_ladder_shear_capture_2026-06-30.md.

Matched setup (apples-to-apples, collisionless-transport-only): both paths start from F_0 = f_FD,
F_{ell>=2} = 0 on the same Gauss-Laguerre MomentumGrid(N_q=20); the even-ladder holds Sigma_H
constant over N in [0, 0.5] so the accumulated shear is exactly S = Sigma_H * 0.5, at which the
characteristic primitives are evaluated. No Hubble damping, thermo, or weak network on either side.
"""

from __future__ import annotations

import numpy as np
import pytest

pytest.importorskip("jax")  # the EXACT reference uses the jax characteristic-ray primitives
import jax  # noqa: E402
jax.config.update("jax_enable_x64", True)  # f64 reference, self-contained (matches sibling tests)
import jax.numpy as jnp  # noqa: E402

from rabbit.config.grids import MomentumGrid  # noqa: E402
from rabbit.jax.characteristic_rays_jax import (  # noqa: E402
    extract_monopole_jax,
    intensity_shift_jax,
    jacobian_jax,
    mu_current_jax,
)
from rabbit.transport.even_ladder_analysis import integrate_even_ladder_final_state  # noqa: E402
from rabbit.transport.even_ladder_observable_bridge import (  # noqa: E402
    energy_density_ratio_from_state,
)
from rabbit.transport.stageAB_state import fermi_dirac  # noqa: E402

_N_Q = 20
_N_MU = 24
_N_END = 0.5


def _char_delta_rho(S: float, grid: MomentumGrid, n_mu: int = _N_MU) -> float:
    """EXACT collisionless energy-density anisotropy at accumulated shear S, from the
    characteristic-ray monopole f_mono(q) = (1/2) sum_j w_j J_j f0(q e^{2 I_j})."""
    q = grid.nodes
    mu0, w0 = np.polynomial.legendre.leggauss(n_mu)
    X0 = mu0 ** 2 / (1.0 - mu0 ** 2)
    signs = np.sign(mu0)
    Xj, Sj, sj, wj, qj = (jnp.asarray(X0), jnp.asarray(float(S)), jnp.asarray(signs),
                          jnp.asarray(w0), jnp.asarray(q))
    mu = mu_current_jax(Xj, sj, Sj)
    I = intensity_shift_jax(Xj, Sj)
    J = jacobian_jax(Xj, Sj, mu)
    f_mono = np.asarray(extract_monopole_jax(I, J, wj, qj))
    fFD = fermi_dirac(q)
    return float(np.trapezoid(q ** 3 * f_mono, q) / np.trapezoid(q ** 3 * fFD, q) - 1.0)


def _even_delta_rho(lmax: int, sigma_h: float, grid: MomentumGrid) -> float:
    state = integrate_even_ladder_final_state(lmax=lmax, sigma_h=sigma_h, n_q=_N_Q, n_end=_N_END)
    return energy_density_ratio_from_state(state, species_idx=0) - 1.0


@pytest.mark.parametrize("sigma_h", [0.1, 0.3])
def test_even_ladder_is_ell_saturated(sigma_h):
    """Adding ell moments past lmax=2 does ~nothing: lmax 2,4,6,8 agree to < 0.2% in delta-rho.
    The hierarchy converges in ell -- to the WRONG limit (see the capture test)."""
    grid = MomentumGrid(N_q=_N_Q)
    d = {lmax: _even_delta_rho(lmax, sigma_h, grid) for lmax in (2, 4, 6, 8)}
    assert abs(d[8] - d[2]) / abs(d[8]) < 2e-3, f"not ell-saturated lmax2->8: {d}"
    assert abs(d[8] - d[6]) / abs(d[8]) < 1e-6, f"adjacent high-ell not identical: {d}"  # ~3.7e-9


@pytest.mark.parametrize("sigma_h", [0.1, 0.3])
def test_even_ladder_captures_only_a_structural_fraction(sigma_h):
    """The ell-saturated even-ladder captures only ~0.28 of the EXACT characteristic delta-rho --
    a ~70% shortfall that no number of ell moments closes (confirms W2-PR6's 20-28% capture)."""
    grid = MomentumGrid(N_q=_N_Q)
    d_char = _char_delta_rho(sigma_h * _N_END, grid)
    d_even = _even_delta_rho(8, sigma_h, grid)  # ell-saturated value
    assert d_char > 1e-3 and d_even > 1e-4, f"a path is ~0: char={d_char} even={d_even}"
    capture = d_even / d_char
    assert 0.20 < capture < 0.35, f"capture {capture} outside structural band at Sigma_H={sigma_h}"


def test_capture_fraction_is_shear_robust():
    """The ~28% shortfall is a stable STRUCTURAL fraction, not a Sigma-dependent artifact:
    the capture at Sigma_H = 0.1 and 0.3 agree to within 0.05."""
    grid = MomentumGrid(N_q=_N_Q)
    cap = {s: _even_delta_rho(8, s, grid) / _char_delta_rho(s * _N_END, grid) for s in (0.1, 0.3)}
    assert abs(cap[0.1] - cap[0.3]) < 0.05, f"capture not shear-robust: {cap}"


def test_capture_has_finite_small_shear_limit_not_one():
    """DECISIVE structural evidence (the mechanism, not just the symptom): as shear -> 0 the capture
    does NOT approach 1.0 -- it plateaus at ~0.27. delta-rho is a SECOND-order-in-shear quantity (the
    O(S) part vanishes by angular parity, sum_j w_j P2(mu_j) = 0, so delta-rho ~ S^2), and the
    even-ladder is exact only to LINEAR order in accumulated shear (rhs_typeI.py). It therefore
    captures only the part of the leading O(S^2) energy anisotropy supplied by the linear F_2->F_0
    feedback. A finite S->0 capture limit != 1 proves the deficiency is leading-order / structural,
    not a finite-shear nonlinear effect -- ruling out the "agree at linear order, differ only
    nonlinearly" alternative."""
    grid = MomentumGrid(N_q=_N_Q)
    cap_small = _even_delta_rho(8, 0.003, grid) / _char_delta_rho(0.003 * _N_END, grid)
    cap_ref = _even_delta_rho(8, 0.1, grid) / _char_delta_rho(0.1 * _N_END, grid)
    # still ~0.27 at near-zero shear: a finite limit well below 1.0 (NOT leading-order agreement)
    assert 0.20 < cap_small < 0.35, f"small-shear capture {cap_small} left the structural band"
    # plateaued: the S->0 limit and the Sigma_H=0.1 value agree -> it is the leading-order ratio
    assert abs(cap_small - cap_ref) < 0.02, f"capture not plateaued: small={cap_small} ref={cap_ref}"


def test_both_paths_agree_at_zero_shear():
    """At zero shear both paths reduce to FLRW (delta-rho = 0): the ~70% shortfall is a genuine
    shear effect, not a baseline/normalization mismatch between the two observables."""
    grid = MomentumGrid(N_q=_N_Q)
    assert abs(_char_delta_rho(0.0, grid)) < 1e-6
    assert abs(_even_delta_rho(8, 0.0, grid)) < 1e-6
