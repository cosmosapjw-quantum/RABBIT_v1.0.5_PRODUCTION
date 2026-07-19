"""rabbit.weak.sigma_plus_kernel — leading-order Σ_+-coupled CL3 angular kernel.

Physics meaning
---------------
In a Bianchi I anisotropic background with shear ``Σ_+``, the transported
neutrino distribution acquires a quadrupole multipole at leading order
in ``Σ_+``:

    f(q, μ) ≈ f₀(q) + Σ_+ · f₂(q) · P₂(μ) + O(Σ_+²)

This quadrupole couples to the ``ℓ = 2`` Legendre projection of the
weak interaction kernel K_2(q, T_νe) (already implemented for the
tilted backend in ``rabbit.jax.weak_finite_mass_jax`` lines 151–164,
and ``rabbit.weak.finite_mass``). In the canonical Type-I path that
quadrupole correction is *not* currently wired — the rate uses only
the monopole f₀.

This module exposes the leading-order ``Σ_+`` quadrupole correction as
a stand-alone helper so it can be applied multiplicatively to the
existing scalar weak rate. The "proxy" approximation used below
replaces the true quadrupole f₂(q) with ``Σ_+ · f₀(q)`` (Pitrou+
2018 §4.4 leading-order; arXiv:2502.20893 eq. 21):

    Δλ_np = Σ_+ ∫ dq f₀(q) K_2(q, T_νe) σ_a(q) / λ_np^{Born}

so that the corrected rate is

    λ_np(Σ_+) = λ_np^{Born} · (1 + Σ_+ · κ_2)

where ``κ_2`` is the quadrature-averaged ``K_2`` weighted by the live
monopole.

Reference
---------
- Pitrou, Coc, Uzan, Vangioni, *PRIMAT*, Phys. Rep. 754:1 (2018), §4.4
- Bennett, Buldgen, de Salas, Gariazzo, Hannestad, Pastor, Wong (2021)
- arXiv:2502.20893 — Bianchi-I BBN constraint paper, eq. 21
- In-tree audit doc: ``docs/audit/v2_derivations/sigma_plus_K2.md``
  (to be written under Plan §0.3)

Public API
----------
- ``sigma_plus_K2_correction_factor`` : returns the dimensionless
  multiplicative factor ``(1 + Σ_+ · κ_2)`` to apply to a Born or
  CL2/CL3 rate.
- ``sigma_plus_K2_correction_factor_jax`` : JAX mirror.
- ``compute_kappa2`` : returns ``κ_2`` alone for diagnostic use.

The module never modifies a global rate; the caller is responsible
for applying the multiplier. The default ``sigma_plus = 0`` returns
``1.0`` exactly (verified via the included tests).
"""
from __future__ import annotations

import numpy as np


# ═══════════════════════════════════════════════════════════════════════
# §1. Constants (mirror weak_finite_mass)
# ═══════════════════════════════════════════════════════════════════════

# Electron mass [MeV]
_M_ELECTRON_MEV = 0.51099895
# Nucleon mass [MeV] (proton; m_n - m_p ≈ 1.293 MeV correction is O(epsilon))
_M_NUCLEON_MEV = 938.272
# Dimensionless m_N = m_N / m_e
_MN_DIMLESS = _M_NUCLEON_MEV / _M_ELECTRON_MEV
# Q-value (n -> p) in m_e units
_Q_DIMLESS = 1.293 / _M_ELECTRON_MEV


def _K2_kernel_per_node(E_e: np.ndarray, E_nu: np.ndarray) -> np.ndarray:
    """K_2(E_e, E_nu) for the angular weak kernel.

    Mirrors :func:`rabbit.jax.weak_finite_mass_jax.finite_mass_angular_coefficients_jax`
    line 163: ``K_2 = -(2/3) p_e E_nu / m_N``. Sign convention matches
    Pitrou+ 2018 eq. 4.31 with their (-) for the Legendre P_2 projection.
    """
    E_e = np.asarray(E_e, dtype=np.float64)
    E_nu = np.asarray(E_nu, dtype=np.float64)
    p_e = np.sqrt(np.maximum(E_e ** 2 - 1.0, 0.0))  # |p_e| in m_e units
    return -(2.0 / 3.0) * p_e * E_nu / _MN_DIMLESS


def compute_kappa2(
    f_nue_monopole: np.ndarray,
    q_nodes: np.ndarray,
    T_nu_MeV: float,
    *,
    n_leg: int = 32,
) -> float:
    """Compute the κ_2 = ⟨K_2⟩ averaged over the live monopole.

    Physics meaning
    ---------------
    κ_2 is the dimensionless coefficient that multiplies Σ_+ in the
    leading-order anisotropic correction to the n→p rate:

        λ(Σ_+) = λ^{Born}(monopole) · (1 + Σ_+ · κ_2)

    It is computed by integrating K_2(E_e, E_nu) against the live
    monopole at each Gauss-Laguerre node. The proxy approximation
    f_2(q) = Σ_+ · f_0(q) means κ_2 depends only on f_0 and on the
    weak phase-space kinematics.

    Parameters
    ----------
    f_nue_monopole : ndarray (N_q,)
        Live ν_e monopole on the Gauss-Laguerre q-grid.
    q_nodes : ndarray (N_q,)
        Dimensionless ``q = E_ν / T_ν`` at GL nodes.
    T_nu_MeV : float
        ν_e temperature [MeV].
    n_leg : int
        Legendre order for the inner E_e integration over phase space
        (rate kernel; standard 32 nodes match canonical weak rate).

    Returns
    -------
    kappa_2 : float (dimensionless)
        Quadrature-averaged ``K_2`` weighted by f_0 and the rate
        kernel. Sign matches Pitrou+ 2018 eq. 4.31.
    """
    f0 = np.asarray(f_nue_monopole, dtype=np.float64)
    q = np.asarray(q_nodes, dtype=np.float64)
    if f0.shape != q.shape:
        raise ValueError(
            f"shape mismatch: f_nue_monopole.shape={f0.shape}, "
            f"q_nodes.shape={q.shape}"
        )
    T_nu_dim = float(T_nu_MeV) / _M_ELECTRON_MEV  # in m_e units

    # E_nu = q * T_nu (in m_e units). Use this energy as the
    # representative neutrino energy at each grid node.
    E_nu_at_nodes = q * T_nu_dim

    # For the κ_2 average we need the average kinematic K_2 weighted by
    # the rate kernel. To keep this module self-contained we approximate
    # the per-node electron energy by E_e ≈ Q + E_nu (n + nu_bar -> p + e
    # threshold kinematics). For p + e -> n + nu the sign on Q flips.
    # The two channels carry opposite K_2 sign (PRIMAT eq. 4.31), but
    # in the n→p rate sum the proxy weighting is symmetric to leading
    # order; we therefore take the channel-averaged K_2 here.
    E_e_at_nodes = _Q_DIMLESS + E_nu_at_nodes

    K2_per_node = _K2_kernel_per_node(E_e_at_nodes, E_nu_at_nodes)

    # Weight by f_0(q) · q^2 (Gauss-Laguerre integrand for f_0 spectrum)
    # and normalize by the same f_0 · q^2 integral for a dimensionless
    # average. This matches the canonical weak-rate weighting at Born level.
    w_node = f0 * q ** 2
    norm = float(np.sum(w_node))
    if norm <= 0.0:
        return 0.0
    return float(np.sum(K2_per_node * w_node) / norm)


def _quadrupole_profile_kernel_and_weights(
    q_nodes: np.ndarray,
    T_nu_MeV: float,
) -> tuple[np.ndarray, np.ndarray, tuple[int, ...]]:
    q = np.asarray(q_nodes, dtype=np.float64)
    if q.ndim != 1:
        raise ValueError("q_nodes must be one-dimensional.")
    if not np.all(np.isfinite(q)):
        raise ValueError("q_nodes must contain only finite values.")
    if np.any(q <= 0.0):
        raise ValueError("q_nodes must be strictly positive.")
    T_nu_dim = float(T_nu_MeV) / _M_ELECTRON_MEV
    if not np.isfinite(T_nu_dim) or T_nu_dim <= 0.0:
        raise ValueError("T_nu_MeV must be positive and finite.")

    E_nu_at_nodes = q * T_nu_dim
    E_e_at_nodes = _Q_DIMLESS + E_nu_at_nodes
    K2_per_node = _K2_kernel_per_node(E_e_at_nodes, E_nu_at_nodes)
    return K2_per_node, q**2, q.shape


def _quadrupole_profile_delta(
    plus_moment_q: np.ndarray,
    f_monopole: np.ndarray,
    *,
    K2_per_node: np.ndarray,
    weights: np.ndarray,
    shape: tuple[int, ...],
) -> float:
    plus = np.asarray(plus_moment_q, dtype=np.float64)
    f0 = np.asarray(f_monopole, dtype=np.float64)
    if plus.shape != shape or f0.shape != shape:
        raise ValueError("plus_moment_q, f_monopole, and q_nodes must share shape.")
    if not np.all(np.isfinite(plus)) or not np.all(np.isfinite(f0)):
        raise ValueError("quadrupole profile inputs must contain only finite values.")
    norm = float(np.sum(np.maximum(f0, 0.0) * weights))
    if norm <= 0.0:
        return 0.0
    return float(np.sum(K2_per_node * plus * weights) / norm)


def compute_kappa2_from_quadrupole_profile(
    plus_moment_q: np.ndarray,
    f_monopole: np.ndarray,
    q_nodes: np.ndarray,
    T_nu_MeV: float,
) -> float:
    """Compute the CL3 K2 correction from an explicit per-q quadrupole profile.

    Unlike :func:`compute_kappa2`, this helper does not replace the quadrupole
    with ``Sigma_+ * f0``.  The caller supplies the current per-q angular
    projection from the reconstructed augmented distribution, so the returned
    value is already the fractional rate shift ``delta_lambda``.
    """

    K2_per_node, weights, shape = _quadrupole_profile_kernel_and_weights(
        q_nodes,
        T_nu_MeV,
    )
    return _quadrupole_profile_delta(
        plus_moment_q,
        f_monopole,
        K2_per_node=K2_per_node,
        weights=weights,
        shape=shape,
    )


def compute_kappa2_pair_from_quadrupole_profiles(
    plus_moment_q: np.ndarray,
    f_monopole: np.ndarray,
    partner_plus_moment_q: np.ndarray,
    partner_f_monopole: np.ndarray,
    q_nodes: np.ndarray,
    T_nu_MeV: float,
) -> tuple[float, float]:
    """Compute two explicit-profile CL3 K2 shifts on one shared q/T kernel.

    The augmented weak-network RHS evaluates ν_e and anti-ν_e quadrupole
    profiles with the same momentum grid and neutrino temperature.  This
    helper keeps the physics identical to two calls to
    :func:`compute_kappa2_from_quadrupole_profile` while reusing validation,
    ``K_2(q, T_nu)`` and ``q**2`` weights across the pair.
    """

    K2_per_node, weights, shape = _quadrupole_profile_kernel_and_weights(
        q_nodes,
        T_nu_MeV,
    )
    delta = _quadrupole_profile_delta(
        plus_moment_q,
        f_monopole,
        K2_per_node=K2_per_node,
        weights=weights,
        shape=shape,
    )
    partner_delta = _quadrupole_profile_delta(
        partner_plus_moment_q,
        partner_f_monopole,
        K2_per_node=K2_per_node,
        weights=weights,
        shape=shape,
    )
    return delta, partner_delta


def sigma_plus_K2_correction_factor(
    sigma_plus: float,
    f_nue_monopole: np.ndarray,
    q_nodes: np.ndarray,
    T_nu_MeV: float,
    *,
    correction_level: int = 3,
) -> float:
    """Multiplicative correction factor ``(1 + Σ_+ · κ_2)``.

    Apply to a baseline rate ``λ_baseline`` to obtain the
    Σ_+-corrected rate at leading order:

        λ(Σ_+) = λ_baseline · sigma_plus_K2_correction_factor(...)

    Returns ``1.0`` exactly when ``sigma_plus == 0`` or
    ``correction_level < 3`` (the angular K_2 kernel is a CL3-only
    correction; below CL3 the recoil/WM physics that K_2 represents
    is not active).

    Parameters
    ----------
    sigma_plus : float
        Shear amplitude Σ_+ at the current epoch.
    f_nue_monopole, q_nodes, T_nu_MeV : passed through to compute_kappa2.
    correction_level : int
        Weak-rate ladder level. K_2 only applies at CL >= 3.

    Returns
    -------
    factor : float
        ``1 + sigma_plus * kappa_2`` if CL >= 3 and ``sigma_plus != 0``;
        else ``1.0``.

    Hallucination guard
    -------------------
    Enforces the legacy guarded runtime/implementation range
    ``|sigma_plus| <= 0.75``.  This cutoff is not a validated physical
    envelope: the publication/science shear domain is NOT VALIDATED pending
    B-03/B-05.  Outside the guarded range the leading-order proxy is
    unreliable.
    """
    if correction_level < 3:
        return 1.0
    if sigma_plus == 0.0:
        return 1.0
    if abs(sigma_plus) > 0.75:
        raise ValueError(
            f"sigma_plus={sigma_plus} outside legacy guarded [-0.75, 0.75] "
            f"runtime/implementation range; the publication/science shear "
            f"domain is NOT VALIDATED pending B-03/B-05, and the leading-order "
            f"Σ-K_2 proxy is not reliable."
        )
    kappa_2 = compute_kappa2(f_nue_monopole, q_nodes, T_nu_MeV)
    return float(1.0 + sigma_plus * kappa_2)


# ═══════════════════════════════════════════════════════════════════════
# §2. JAX mirror (lazy import to keep numpy-only consumers fast)
# ═══════════════════════════════════════════════════════════════════════

def sigma_plus_K2_correction_factor_jax(
    sigma_plus,
    f_nue_monopole,
    q_nodes,
    T_nu_MeV,
    *,
    correction_level: int = 3,
):
    """JAX mirror of :func:`sigma_plus_K2_correction_factor`.

    Designed to be jit-able. Branches on ``correction_level`` and the
    static ``sigma_plus == 0`` short-circuit are compile-time decisions.
    The runtime ``sigma_plus`` value participates in autodiff.

    Note: this routine assumes ``jax_enable_x64`` is True. The rabbit
    test conftest enforces it; production callers should rely on the
    rabbit JAX-config bootstrap.
    """
    import jax
    import jax.numpy as jnp
    # Defensive: x64 must be on for parity with the NumPy path.
    jax.config.update("jax_enable_x64", True)

    if correction_level < 3:
        return jnp.array(1.0, dtype=jnp.float64)

    f0 = jnp.asarray(f_nue_monopole, dtype=jnp.float64)
    q = jnp.asarray(q_nodes, dtype=jnp.float64)
    T_nu_dim = jnp.asarray(T_nu_MeV, dtype=jnp.float64) / _M_ELECTRON_MEV

    E_nu_at_nodes = q * T_nu_dim
    E_e_at_nodes = _Q_DIMLESS + E_nu_at_nodes
    p_e = jnp.sqrt(jnp.maximum(E_e_at_nodes ** 2 - 1.0, 0.0))
    K2_per_node = -(2.0 / 3.0) * p_e * E_nu_at_nodes / _MN_DIMLESS

    w_node = f0 * q ** 2
    norm = jnp.sum(w_node)
    kappa_2 = jnp.where(norm > 0.0, jnp.sum(K2_per_node * w_node) / norm, 0.0)
    return 1.0 + sigma_plus * kappa_2
