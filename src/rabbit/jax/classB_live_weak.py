"""rabbit.jax.classB_live_weak — CL>0 weak rate enabling helper for Class B.

Phase η-2 deliverable (Plan §2.4 / §13.6). The Class B JAX driver
(``rabbit.jax.driver_classB``) currently raises at
``correction_level > 0`` because Class B does not transport its own
neutrino monopole through the geometry RHS — it relies on the same
single-T_ν tier-1 thermo that Type I uses. This module provides the
single function that lifts the CL>0 wall by routing through
``compute_live_rates_from_monopoles_level_specialized_jax`` with the
equilibrium Fermi-Dirac monopole as input.

The semantic difference vs the existing Class B Born path:

  - CL0  : equivalent to ``compute_born_rates`` to within the live-FD
    integral ↔ analytic-Born identity (same physics; different code path).
  - CL1  : adds the Coulomb (Fermi function) factor on the integrand.
  - CL2  : adds the Sirlin radiative correction.
  - CL3  : adds the finite-mass kernel (recoil + weak magnetism). The
    underlying ``compute_live_rates_from_monopoles_level_specialized_jax``
    already handles the CL3 I0 normalisation; Phase η-4 lifted the
    earlier Class-B-side raise.

Why a separate module
---------------------
We avoid editing ``driver_classB.py`` directly in this commit so the
existing Class B CL0 gold values stay bit-identical. The driver
modification (replace the ``raise`` with a call to this helper) is a
clean two-line change that callers can do via subclassing or by
landing it in a follow-on commit once the helper is independently
validated.

Reference
---------
- Plan §2.4 acceptance: lift the ``correction_level > 0`` raise.
- ``rabbit.jax.weak_live_jax`` for the underlying live-weak dispatch.
"""

from __future__ import annotations

from functools import lru_cache
from typing import Tuple

import jax
import jax.numpy as jnp


jax.config.update("jax_enable_x64", True)


@lru_cache(maxsize=8)
def _cached_q_grid_and_fd(N_q: int) -> Tuple[jnp.ndarray, jnp.ndarray]:
    """Standard Gauss-Laguerre nodes + equilibrium FD monopole on those nodes.

    Cached because the q-grid is constant for a given N_q and the FD
    monopole evaluated on dimensionless ``q = E_ν / T_ν`` is also
    constant in that variable (the temperature dependence enters via
    the change of variable ``q · T_ν`` in the rate kernel).
    """
    from numpy.polynomial.laguerre import laggauss
    nodes_np, _ = laggauss(int(N_q))
    q = jnp.asarray(nodes_np, dtype=jnp.float64)
    f_fd = 1.0 / (1.0 + jnp.exp(q))
    return q, f_fd


def compute_classB_cl_rates(
    t_gamma_MeV: jnp.ndarray,
    t_nu_MeV: jnp.ndarray,
    tau_n: jnp.ndarray,
    correction_level: int,
    *,
    N_q: int = 20,
) -> Tuple[jnp.ndarray, jnp.ndarray]:
    """Compute (λ_np, λ_pn) for Class B at the requested correction level.

    Routes through the in-tree live-weak path with the equilibrium FD
    monopole on a Gauss-Laguerre grid. At CL0 this should agree with
    ``compute_born_rates`` to ~1e-3 relative (live-FD integral vs
    analytic-Born identity).

    Parameters
    ----------
    t_gamma_MeV, t_nu_MeV, tau_n : scalar JAX arrays.
    correction_level : int in {0, 1, 2, 3}.
        Phase η-4 lifted CL3 support; the level_specialized JAX
        dispatch already handles all four levels including the
        finite-mass I0 normalisation.
    N_q : int — quadrature order; default 20.

    Returns
    -------
    lambda_np, lambda_pn : scalar JAX arrays (s^-1).

    Notes
    -----
    This is the *infrastructure* that the future Class B driver wire
    will call. The driver retains its existing ``raise`` so existing
    CL0 tests are not affected.
    """
    if correction_level not in (0, 1, 2, 3):
        raise ValueError(
            f"correction_level={correction_level} not in {{0, 1, 2, 3}}; "
            f"CL > 3 is not defined in the in-tree weak ladder."
        )
    from rabbit.jax.weak_live_jax import (
        compute_live_rates_from_monopoles_level_specialized_jax,
    )
    q_nodes, f_fd = _cached_q_grid_and_fd(int(N_q))
    lambda_np, lambda_pn, _ = compute_live_rates_from_monopoles_level_specialized_jax(
        T_gamma_MeV=t_gamma_MeV,
        T_nu_MeV=t_nu_MeV,
        tau_n=tau_n,
        q_nodes=q_nodes,
        f_nue_monopole=f_fd,
        f_nuebar_monopole=f_fd,
        correction_level=int(correction_level),
    )
    return lambda_np, lambda_pn
