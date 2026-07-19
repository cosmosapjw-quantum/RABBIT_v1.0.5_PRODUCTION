"""
rabbit.transport.teff_collision_bridge — Teff gather-scatter for characteristic + collision.

This module bridges exact characteristic ray transport with physical collision
operators using the Teff representation as a compression/decompression layer.

Algorithm:
    1. GATHER:  ray data (I_j, J_j) → per-ray Θ_j → angle-averaged f̃₀(q)
    2. COLLIDE: C₀[f̃₀](q) via NuEScatteringOperator + PairProcessOperator
    3. SCATTER: energy conservation → δI_j per ray

HONESTY NOTE (external audit BD601, 2026-06-30): the implemented ``tangency_diagnostic`` below is a
DEGENERATE proxy, NOT a validated Teff-manifold guard. Its P2 term collapses to the input-independent
constant -0.75, so it returns ~0 for every realistic state and never flags spectral distortion. It
does NOT compute the §5.4 entropy-projection inequality and gives NO entropy-cost guarantee. It is
reported only (no code gates on it). Do not read D2~0 as "Teff-valid" — read it as "uninformative".
See ``tests/test_teff_bridge_tangency_degenerate.py``.

References (the INTENT; not what the degenerate proxy computes):
    §5 of RABBIT report: Teff multipole framework
    §5.3: Tangency diagnostic
    §5.4: Entropy-projection inequality
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import os
import numpy as np

from rabbit.collisions.projected_operator import PhysicalCollisionOperator
from rabbit.config.grids import MomentumGrid

_COLLISION_OPERATOR_CACHE: dict[int, PhysicalCollisionOperator] = {}
_BRIDGE_QUAD_CACHE: dict[
    tuple[bytes, bytes],
    tuple[np.ndarray, np.ndarray, np.ndarray, float],
] = {}



def _bridge_env_float(name: str, default: float) -> float:
    raw = os.environ.get(name, "")
    if raw == "":
        return float(default)
    try:
        return float(raw)
    except Exception:
        return float(default)

def _safe_bridge_energy_weight(q_nodes, q_weights):
    q = np.asarray(q_nodes, dtype=np.float64)
    w = np.asarray(q_weights, dtype=np.float64)
    q_cap = _bridge_env_float("RABBIT_COLLISION_QMAX", 80.0)
    q_eff = np.minimum(q, q_cap)
    with np.errstate(over="ignore", invalid="ignore", under="ignore"):
        base = w * np.exp(q_eff) * q**3
    return np.nan_to_num(base, nan=0.0, posinf=0.0, neginf=0.0)


def _bridge_quadrature_data(q_nodes, q_weights):
    key = (
        np.asarray(q_nodes, dtype=np.float64).tobytes(),
        np.asarray(q_weights, dtype=np.float64).tobytes(),
    )
    cached = _BRIDGE_QUAD_CACHE.get(key)
    if cached is not None:
        return cached

    q_nodes_arr = np.asarray(q_nodes, dtype=np.float64)
    q_weights_arr = np.asarray(q_weights, dtype=np.float64)
    f0 = 1.0 / (np.exp(np.minimum(q_nodes_arr, 500.0)) + 1.0)
    bridge_weight = _safe_bridge_energy_weight(q_nodes_arr, q_weights_arr)
    delta_rho_weight = q_weights_arr * np.exp(q_nodes_arr) * q_nodes_arr**3 * f0
    rho_ref = float(np.sum(bridge_weight * f0, dtype=np.float64))
    cached = (f0, delta_rho_weight, bridge_weight, rho_ref)
    _BRIDGE_QUAD_CACHE[key] = cached
    return cached

@dataclass
class GatherScatterResult:
    """Result of a Teff gather-scatter collision step."""
    delta_I: np.ndarray        # (N_mu,) correction to energy shift integrals
    delta_rho_nu: float        # net energy transfer to ν sector [MeV⁴ units]
    C_monopole: np.ndarray     # (N_q,) monopole collision integral
    theta_per_ray: np.ndarray  # (N_mu,) per-ray effective temperatures
    f_monopole: np.ndarray     # (N_q,) angle-averaged monopole
    tangency_D2: float         # tangency diagnostic


def gather_ray_temperatures(
    I: np.ndarray,
    J: np.ndarray,
    w0: np.ndarray,
) -> np.ndarray:
    """Extract per-ray effective temperature from characteristic data.

    Each ray has an energy shift exp(2I_j), so the effective temperature
    measured by a comoving observer along ray j is:

        Θ_j = exp(-2 I_j)

    A neutrino with comoving momentum q along ray j has shifted momentum
    q_eff = q × exp(2I_j) = q / Θ_j, hence f = f_FD(q/Θ_j).

    Parameters
    ----------
    I : (N_mu,) energy shift integrals
    J : (N_mu,) angular Jacobians (unused here but kept for API symmetry)
    w0 : (N_mu,) quadrature weights (unused here)

    Returns
    -------
    theta : (N_mu,) per-ray effective temperatures
    """
    return np.exp(-2.0 * I)


def gather_monopole(
    I: np.ndarray,
    J: np.ndarray,
    w0: np.ndarray,
    q_nodes: np.ndarray,
    *,
    theta: np.ndarray | None = None,
) -> np.ndarray:
    """Angle-average the ray distributions to get the monopole.

    f̃₀(q) = (1/2) Σ_j w_j J_j f_FD(q / Θ_j)

    This is identical to characteristic_rays.extract_monopole but
    expressed in the Teff language for clarity.
    """
    if theta is None:
        theta = gather_ray_temperatures(I, J, w0)
    qa = q_nodes[:, None] / theta[None, :]  # (N_q, N_mu)
    f_vals = 1.0 / (np.exp(np.minimum(qa, 500)) + 1)
    return 0.5 * f_vals @ (w0 * J)


def tangency_diagnostic(
    I: np.ndarray,
    J: np.ndarray,
    w0: np.ndarray,
    q_nodes: np.ndarray,
    q_weights: np.ndarray,
) -> float:
    """DEGENERATE tangency proxy — NOT a Teff-manifold validity guard.

    INTENT (RABBIT report §5.3) was to measure deviation from the Teff manifold
    f(q,μ)=f_FD(q/Θ(μ)) (a single temperature per direction). As IMPLEMENTED it does NOT:
      * it sees only the per-ray Θ_j = exp(-2 I_j) — its inputs are (I, J, w0, q_nodes), never the
        spectral shape ALONG a ray — so a genuine non-FD distortion is invisible to it by construction;
      * its ``P2`` term collapses to the input-independent constant -0.75 (because
        Σ w0 J θ = 2 Θ_mean by the definition of Θ_mean), so
        ``D2 = sqrt(max(T2 - 0.5625, 0))`` with ``T2 = Θ_var/Θ_mean²``. This is ~0 for every realistic
        state and only becomes nonzero when the cross-ray Θ spread is absurd (Θ_std/Θ_mean > 0.75,
        far beyond any physical shear). It returns ~0 even for a strongly-collided state.
    So it does NOT detect the spectral distortion it was meant to, does NOT compute the §5.4
    entropy-projection inequality, and is NOT a validity guarantee. Reported only; nothing gates on
    it. Treat D2~0 as "uninformative", not "Teff-valid". Pinned by
    ``tests/test_teff_bridge_tangency_degenerate.py``.

    Returns
    -------
    D2 : float >= 0.  DEGENERATE: ~0 for realistic states regardless of actual distortion.
    """
    # DEGENERATE proxy (see docstring): this only forms the cross-ray Θ variance; it never inspects
    # the spectral shape along a ray, and the P2 term below is the input-independent constant -0.75,
    # so D2 = sqrt(max(T2 - 0.5625, 0)). The (q_nodes, q_weights) args are unused here -- kept for API
    # symmetry with the other gather/scatter helpers.
    theta = gather_ray_temperatures(I, J, w0)
    theta_mean = 0.5 * np.sum(w0 * J * theta)
    theta_var = 0.5 * np.sum(w0 * J * (theta - theta_mean)**2)
    T2 = theta_var / max(theta_mean**2, 1e-30)

    # P2 collapses to the constant -0.75: since Σ w0 J θ = 2 θ_mean (definition of θ_mean above),
    # the m=0 sum reduces to -0.5, and P2 = 0.5 * 3 * (-0.5) = -0.75. Preserved verbatim (not
    # behaviour-changed) so the value is exactly what callers/parity tests expect.
    P2 = 0.5 * (3 * np.array([0.5 * np.sum(w0 * J * (theta/theta_mean) *
                  (0.5*(3*m**2-1))) for m in [0]])[0])  # input-independent constant -0.75
    D2 = max(T2 - P2**2, 0.0)

    return float(np.sqrt(max(D2, 0.0)))


def scatter_collision_to_rays(
    C_monopole: np.ndarray,
    I: np.ndarray,
    J: np.ndarray,
    w0: np.ndarray,
    q_nodes: np.ndarray,
    q_weights: np.ndarray,
    H_inv_sec: float,
) -> np.ndarray:
    """Scatter collision integral back to ray energy shifts.

    The isotropic part of the collision integral C₀(q) changes the
    total neutrino energy density by:

        δρ_ν = ∫ q³ f₀(q) C₀(q) dq / H

    For isotropic collisions (Born-level V-A), all rays receive the
    same fractional energy change:

        δI_j = -(δρ_ν) / (8 ρ_ν)

    This preserves the FD spectral shape along each ray (the Teff
    manifold), with the collision effect encoded as a temperature shift.

    Parameters
    ----------
    C_monopole : (N_q,) collision integral on the monopole
    I : (N_mu,) current ray energy shift integrals
    J : (N_mu,) angular Jacobians
    w0 : (N_mu,) quadrature weights
    q_nodes : (N_q,)
    q_weights : (N_q,) Gauss-Laguerre weights
    H_inv_sec : Hubble rate [s⁻¹]

    Returns
    -------
    delta_I : (N_mu,) correction to energy shift integrals
    """
    _, delta_rho_weight, _, rho_ref = _bridge_quadrature_data(q_nodes, q_weights)
    C_monopole = np.asarray(C_monopole, dtype=np.float64)
    # BD612-F6: do NOT sanitize a non-finite runtime collision field. Propagating a
    # NaN/Inf makes a bad collision state fail loud downstream (the driver surfaces
    # collision_accumulator_finite) instead of silently zeroing it. rho_ref is a
    # deterministic quadrature constant, so the guard below is a CONSTRUCTION guard
    # (a pathological q-grid), not runtime-state hiding — kept.
    if (not np.isfinite(rho_ref)) or rho_ref <= 1.0e-300:
        return np.zeros_like(I, dtype=np.float64)

    # Energy transfer from collision
    delta_rho = np.sum(delta_rho_weight * C_monopole, dtype=np.float64)

    # Isotropic scatter: all rays get the same δI
    # Since Θ_j = exp(-2I_j), and δΘ_j/Θ_j = δρ/(4ρ),
    # we have δI_j = -δΘ_j/(2Θ_j) = -δρ/(8ρ)
    frac = delta_rho / max(abs(rho_ref), 1e-30)
    delta_I = np.full_like(I, -frac / 8.0)

    relax = _bridge_env_float("RABBIT_COLLISION_BRIDGE_RELAX", 1.0)
    # BD612-F6: propagate a non-finite delta_I (fail loud) rather than zeroing it.
    return relax * delta_I


def apply_gather_scatter_collision(
    I: np.ndarray,
    J: np.ndarray,
    w0: np.ndarray,
    q_nodes: np.ndarray,
    q_weights: np.ndarray,
    T_gamma: float,
    T_nu: float,
    H_inv_sec: float,
    collision_operator=None,
    compute_tangency: bool = True,
) -> GatherScatterResult:
    """Full gather-scatter collision step.

    This is the top-level function called by the driver to apply
    physical collision effects within the characteristic transport
    framework.

    Steps:
        1. Gather: I_j → Θ_j → f̃₀(q)
        2. Tangency check: D≥₂ < tolerance
        3. Collision: C₀[f̃₀](q) from operator
        4. Scatter: energy conservation → δI_j

    Parameters
    ----------
    I, J, w0 : characteristic ray state
    q_nodes, q_weights : momentum grid
    T_gamma, T_nu : temperatures [MeV]
    H_inv_sec : Hubble rate [s⁻¹]
    collision_operator : object with evaluate(f_mono, f_quad, T_gamma, T_nu, H)
        If None, uses PhysicalCollisionOperator from projected_operator.

    Returns
    -------
    GatherScatterResult

    Candidate-path caveats (BD612-F6; this is a CALIBRATED-RTA candidate surface,
    default-OFF and never on public inference dispatch):
      * The scatter normalizes δI by a FIXED equilibrium reference density
        ``rho_ref`` (``_bridge_quadrature_data``), not the actual per-bundle ρ_ν the
        δI = −δρ_ν/(8 ρ_ν) formula names — an equilibrium-reference approximation.
      * ``RABBIT_COLLISION_QMAX`` / ``RABBIT_COLLISION_BRIDGE_RELAX`` are read from
        the environment per call here, but the JAX twin takes them as static args;
        a non-default env therefore makes this numpy path diverge from the
        "parity-locked" twin.
      * ``delta_rho_nu`` is reported but NOT consumed by the tier-2 3T plasma
        channel (which runs its own momentum-averaged source), so the ν-sector
        energy is representable twice with no RHS-level budget reconciliation.
    """
    # ── 1. Gather ──
    theta = gather_ray_temperatures(I, J, w0)
    f_mono = gather_monopole(I, J, w0, q_nodes, theta=theta)

    # ── 2. Tangency diagnostic ──
    D2 = tangency_diagnostic(I, J, w0, q_nodes, q_weights) if compute_tangency else 0.0

    # ── 3. Collision ──
    if collision_operator is None:
        N_q = int(len(q_nodes))
        collision_operator = _COLLISION_OPERATOR_CACHE.get(N_q)
        if collision_operator is None:
            collision_operator = PhysicalCollisionOperator(MomentumGrid(N_q=N_q), n_species=3)
            _COLLISION_OPERATOR_CACHE[N_q] = collision_operator

    # The collision operator expects Ψ₀ (perturbation from equilibrium)
    # Compute Ψ₀ = (f_mono - f_eq) / f_eq
    f_eq, delta_rho_weight, _, _ = _bridge_quadrature_data(q_nodes, q_weights)
    Psi0 = (f_mono - f_eq) / np.maximum(f_eq, 1e-30)

    # All species get the same monopole perturbation (isotropic approx)
    Psi0_3sp = np.tile(Psi0, (3, 1))  # (3, N_q)
    Psi2_3sp = np.zeros((3, len(q_nodes)))  # quadrupole = 0 for monopole collision

    coll_result = collision_operator.evaluate(
        Psi0_3sp, Psi2_3sp, T_gamma, T_nu, H_inv_sec)

    # Sum over species for total monopole collision integral
    C_mono = np.sum(coll_result.C_ell0, axis=0)  # (N_q,)

    # ── 4. Scatter ──
    delta_I = scatter_collision_to_rays(
        C_mono, I, J, w0, q_nodes, q_weights, H_inv_sec)

    # Net energy transfer
    delta_rho = np.sum(delta_rho_weight * C_mono, dtype=np.float64)

    return GatherScatterResult(
        delta_I=delta_I,
        delta_rho_nu=float(delta_rho),
        C_monopole=C_mono,
        theta_per_ray=theta,
        f_monopole=f_mono,
        tangency_D2=float(D2),
    )
