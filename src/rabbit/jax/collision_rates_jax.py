"""Total weak-collision rates for the canonical PR-T3B preflight.

Provides JIT-compatible JAX evaluators for the Mangano /
Hannestad-Madsen total ν-e + pair-process scattering rate, in the
ultra-relativistic massless-electron limit:

    Γ_α(T) = (7π/12) · G_F^2 · T^5 · a_α

where the species coefficient ``a_α`` absorbs the CC + NC mixing:

    a_e = 1 + 4 sin²θ_W + 8 sin⁴θ_W ≈ 2.353  (ν_e channel)
    a_x = 1 - 4 sin²θ_W + 8 sin⁴θ_W ≈ 0.503  (ν_μ, ν_τ channel)

The constants are imported from ``rabbit.collisions.kernels``
(``A_TOTAL_NUE``, ``A_TOTAL_NUX``).  The returned rate is in
natural-unit ``[MeV]`` (= ``[s^-1]`` in natural units); divide by
``H_MeV`` to convert to ``d/dN``.

This module is canonical-PR-T3B building-block infrastructure
that the future asymptotic-preserving (AP-form) collision wrapper
will need to evaluate the equilibration rate ``Γ/H`` cleanly,
without re-deriving it from the full 2D collision integral every
RHS call.  It is not yet wired into the runtime driver.

References:

    Hannestad & Madsen, Phys. Rev. D 52, 1764 (1995)
    Notzold & Raffelt, Nucl. Phys. B 307, 924 (1988)
    Mangano et al., Nucl. Phys. B 729, 221 (2005)
"""
from __future__ import annotations

import jax
import jax.numpy as jnp

from rabbit.collisions.kernels import (
    A_TOTAL_NUE,
    A_TOTAL_NUX,
)
from rabbit.jax.kernel_backend import inspect_kernel_backend, resolve_kernel_backend
from rabbit.thermo.rate_prefactors import WEAK_2TO2_RATE_PREFACTOR_MEV

jax.config.update("jax_enable_x64", True)


_PREFACTOR_MEV = jnp.asarray(WEAK_2TO2_RATE_PREFACTOR_MEV, dtype=jnp.float64)


def _resolve_kernel_backend(backend: str, kernel_backend: str | None) -> str:
    return resolve_kernel_backend(backend=backend, kernel_backend=kernel_backend)


def collision_rate_kernel_backend_report(
    kernel_backend: str | None = "jax",
) -> dict[str, object]:
    """Return staged-forwarding metadata for collision-rate helpers."""

    meta = inspect_kernel_backend(
        kernel_backend, scope="collision_rate"
    ).as_dict()
    return meta


def total_rate_nu_e_jax(
    T_MeV: jnp.ndarray, *, backend: str = "jax", kernel_backend: str | None = None
) -> jnp.ndarray:
    """``Γ_e(T) = (7π/12) G_F^2 T^5 a_e``  [MeV] for ν_e channel.

    Parameters
    ----------
    T_MeV : array-like
        Photon temperature.
    backend, kernel_backend : {'jax', 'auto'}
        JAX/XLA-only backend dispatcher. ``'auto'`` resolves to the in-tree
        closed-form JAX expression. ``kernel_backend`` is the preferred staged
        name; ``backend`` is retained for compatibility.
    """
    backend = _resolve_kernel_backend(backend, kernel_backend)
    if backend != "jax":
        raise ValueError(f"unknown backend={backend!r}; choose 'jax' or 'auto'")
    T = jnp.asarray(T_MeV, dtype=jnp.float64)
    return _PREFACTOR_MEV * T**5 * jnp.asarray(A_TOTAL_NUE, dtype=jnp.float64)


def total_rate_nu_x_jax(
    T_MeV: jnp.ndarray, *, backend: str = "jax", kernel_backend: str | None = None
) -> jnp.ndarray:
    """``Γ_x(T) = (7π/12) G_F^2 T^5 a_x``  [MeV] for ν_x (μ/τ) channel.

    See :func:`total_rate_nu_e_jax` for backend dispatch semantics.
    """
    backend = _resolve_kernel_backend(backend, kernel_backend)
    if backend != "jax":
        raise ValueError(f"unknown backend={backend!r}; choose 'jax' or 'auto'")
    T = jnp.asarray(T_MeV, dtype=jnp.float64)
    return _PREFACTOR_MEV * T**5 * jnp.asarray(A_TOTAL_NUX, dtype=jnp.float64)


def total_rate_jax(
    T_MeV: jnp.ndarray,
    species: str = "nue",
    *,
    backend: str = "jax",
    kernel_backend: str | None = None,
) -> jnp.ndarray:
    """Dispatch helper: ``Γ_α(T)`` for ``α in {nue, nux}``."""
    backend = _resolve_kernel_backend(backend, kernel_backend)
    sp = species.strip().lower()
    if sp == "nue":
        return total_rate_nu_e_jax(T_MeV, backend=backend)
    if sp == "nux":
        return total_rate_nu_x_jax(T_MeV, backend=backend)
    raise ValueError(f"unknown species: {species!r}")


def gamma_over_H_jax(
    T_MeV: jnp.ndarray,
    H_MeV: jnp.ndarray,
    species: str = "nue",
    *,
    kernel_backend: str | None = None,
) -> jnp.ndarray:
    """Equilibration ratio ``Γ_α(T) / H``  [dimensionless].

    This is the scalar that drives the AP-form collision relaxation
    timescale: ``C_AP[f] = -(Γ/H) (f - f_eq)`` evolves ``f`` toward
    equilibrium with characteristic ``e``-fold count ``H/Γ``.  When
    ``Γ/H >> 1`` the system is strongly coupled (early universe);
    when ``Γ/H << 1`` it has decoupled (late universe).
    """
    rate = total_rate_jax(T_MeV, species, kernel_backend=kernel_backend)
    return rate / jnp.maximum(jnp.asarray(H_MeV, dtype=jnp.float64), 1.0e-100)


def total_rate_nu_nu_diagonal_jax(
    T_MeV: jnp.ndarray, alpha_eq_beta: bool = False
) -> jnp.ndarray:
    """Mangano 2005 total ν-ν elastic scattering rate (diagonal,
    Fierz-allowed channels only).

    For ``ν_α + ν_β → ν_α + ν_β`` in the Standard Model with
    Fierz-diagonal selection (no flavour mixing):

        Γ_νν,diag(T) ≈ (7π/12) · G_F² · T⁵ · a_αβ

    where ``a_αβ`` is the Fierz / coupling coefficient
    (``a_αβ = 2`` for identical species, ``a_αβ = 1`` for
    distinguishable).  This mirrors the Hannestad-Madsen ν-e
    convention by absorbing the Fierz factor into a single
    species coefficient; the per-channel matrix element is
    structurally identical to ν-e (both are 2→2 weak processes
    in the UR massless limit, with the angular-averaged
    Mandelstam dependence ``(s² + u²) / s²``).

    The full Dolgov-Hansen-Semikoz appendix-A coefficient table
    refines the ``a_αβ`` numerical value with a small
    ``O(1)`` correction for the running of the weak couplings;
    the simplified form here is the ``leading-order Mangano``
    convention used in the AP-form preflight surface.
    """
    T = jnp.asarray(T_MeV, dtype=jnp.float64)
    a_alpha_beta = 2.0 if bool(alpha_eq_beta) else 1.0
    return _PREFACTOR_MEV * T**5 * jnp.asarray(a_alpha_beta, dtype=jnp.float64)


def gamma_nu_nu_over_H_jax(
    T_MeV: jnp.ndarray, H_MeV: jnp.ndarray, alpha_eq_beta: bool = False
) -> jnp.ndarray:
    """``Γ_νν / H`` equilibration ratio for diagonal ν-ν."""
    rate = total_rate_nu_nu_diagonal_jax(T_MeV, alpha_eq_beta=alpha_eq_beta)
    return rate / jnp.maximum(jnp.asarray(H_MeV, dtype=jnp.float64), 1.0e-100)


__all__ = [
    "total_rate_nu_e_jax",
    "total_rate_nu_x_jax",
    "total_rate_jax",
    "collision_rate_kernel_backend_report",
    "gamma_over_H_jax",
    "total_rate_nu_nu_diagonal_jax",
    "gamma_nu_nu_over_H_jax",
]
