"""
rabbit.validation.transport_convergence — ℓ_max convergence for curved types.

Tests the transport hierarchy's ℓ-truncation convergence by integrating
the coupled {Σ, Ψ_ℓ} system at various ℓ_max values and comparing
the resulting π̃ and ΔY_p.

For Type I (κ=0): ℓ_max=2 is EXACT (no free-streaming cascade).
For Type II (κ≠0): ℓ_max=2 is APPROXIMATE; convergence requires ℓ_max=4-8.

The convergence criterion: |π̃(ℓ_max) − π̃(ℓ_max−2)| / |π̃(ℓ_max)| < ε.
"""
from __future__ import annotations
from dataclasses import dataclass
from typing import Dict, List, Tuple
import numpy as np
from scipy.integrate import solve_ivp


# ═══════════════════════════════════════════════════════════════
# §1. Minimal curved hierarchy integrator
# ═══════════════════════════════════════════════════════════════

_F_NU = 0.40520
_A_FB = 6.0
_B_SRC = 8.0 / 15.0


def _curved_hierarchy_rhs(N, y, kappa, ell_max, f_nu=_F_NU):
    """RHS for the coupled {Σ₊, Ψ₀, Ψ₂, Ψ₄, ...} system.

    State: y = [Σ₊, Ψ₀, Ψ₂, Ψ₄, ..., Ψ_{ℓ_max}]
    with even ℓ only (orthogonal models).

    IMPORTANT: The Hubble-normalized damping D_ℓ is REQUIRED for the
    standalone reduced system. In the full 960-DOF production code,
    this damping emerges automatically from the energy-weighted
    projection integrals. Here it must be explicit.

    From Paper I eigenvalue analysis: trace(M) = -5 requires D₂ = 4.
    For higher ℓ: D_ℓ = 4 + (ℓ−2)/2 (increasing free-streaming leak).
    """
    Sp = y[0]
    n_ell = (ell_max // 2) + 1
    psi = y[1:1 + n_ell]

    Sigma_sq = Sp**2
    Omega = max(1 - Sigma_sq, 0.01)
    q_dec = 1 + Sigma_sq

    # Shear
    Pi = -_A_FB * f_nu * psi[1] * Omega  # psi[1] = Ψ₂
    dSp = -(2 - q_dec) * Sp + Pi

    # Transport hierarchy with Hubble-normalized damping
    dpsi = np.zeros(n_ell)

    for li in range(n_ell):
        ell = 2 * li

        if ell == 0:
            # Monopole: κ-coupling to quadrupole only
            if n_ell > 1 and abs(kappa) > 1e-30:
                dpsi[0] = -(2.0 / 3.0) * kappa * psi[1]

        elif ell == 2:
            # Quadrupole: shear source + κ-coupling + Hubble damping
            dpsi[li] = -_B_SRC * Sp - 4.0 * psi[li]  # D₂ = 4
            if abs(kappa) > 1e-30:
                dpsi[li] += (2.0 / 5.0) * kappa * psi[0]
                if li + 1 < n_ell:
                    dpsi[li] -= (3.0 / 5.0) * kappa * psi[li + 1]

        else:
            # Higher ℓ: streaming coupling + Hubble damping
            D_ell = 4.0 + (ell - 2) / 2.0  # D₄=5, D₆=6, D₈=7, ...
            coeff_lo = ell / (2.0 * ell + 1.0)
            coeff_hi = (ell + 1.0) / (2.0 * ell + 1.0)

            dpsi[li] = -D_ell * psi[li]  # Hubble damping

            if li > 0:
                dpsi[li] += coeff_lo * kappa * psi[li - 1]
            if li + 1 < n_ell:
                dpsi[li] -= coeff_hi * kappa * psi[li + 1]
            else:
                # Closure: additional absorptive damping at boundary
                D_closure = coeff_hi * max(abs(kappa), 0.1)
                dpsi[li] -= D_closure * psi[li]

    dy = np.zeros(1 + n_ell)
    dy[0] = dSp
    dy[1:1 + n_ell] = dpsi
    return dy


# ═══════════════════════════════════════════════════════════════
# §2. Convergence scan
# ═══════════════════════════════════════════════════════════════

@dataclass
class ConvergenceResult:
    """Result of ℓ_max convergence test."""
    ell_max_values: List[int]
    pi_tilde_final: List[float]
    Sigma_final: List[float]
    relative_change: List[float]  # |π̃(ℓ) − π̃(ℓ−2)| / |π̃(ℓ)|
    converged_ell_max: int
    convergence_threshold: float


def convergence_scan(
    Sigma_H_0: float = 0.1,
    kappa: float = 0.1,
    ell_max_values: List[int] = None,
    N_end: float = 15.0,
    threshold: float = 0.01,
) -> ConvergenceResult:
    """Scan ℓ_max convergence for given κ.

    Parameters
    ----------
    Sigma_H_0 : float
        Initial shear.
    kappa : float
        Effective curvature wavenumber.
    ell_max_values : list of int
        ℓ_max values to test (must be even).
    N_end : float
        Integration endpoint.
    threshold : float
        Convergence criterion for |Δπ̃/π̃|.

    Returns
    -------
    ConvergenceResult
    """
    if ell_max_values is None:
        ell_max_values = [2, 4, 6, 8, 10]

    pi_finals = []
    sigma_finals = []

    for lmax in ell_max_values:
        n_ell = (lmax // 2) + 1
        y0 = np.zeros(1 + n_ell)
        y0[0] = Sigma_H_0  # Σ₊(0)
        # All Ψ_ℓ(0) = 0

        sol = solve_ivp(
            lambda N, y: _curved_hierarchy_rhs(N, y, kappa, lmax),
            [0, N_end], y0, method='Radau',
            rtol=1e-10, atol=1e-12,
        )

        Sp_f = float(sol.y[0, -1])
        Psi2_f = float(sol.y[2, -1])  # index 2 = Ψ₂ (index 0=Σ, 1=Ψ₀, 2=Ψ₂)

        pi_finals.append(Psi2_f)
        sigma_finals.append(Sp_f)

    # Relative changes
    rel_changes = [0.0]
    for i in range(1, len(pi_finals)):
        if abs(pi_finals[i]) > 1e-30:
            rel_changes.append(abs(pi_finals[i] - pi_finals[i - 1]) / abs(pi_finals[i]))
        else:
            rel_changes.append(0.0)

    # Find convergence
    converged = ell_max_values[0]
    for i, rc in enumerate(rel_changes):
        if i > 0 and rc < threshold:
            converged = ell_max_values[i]
            break

    return ConvergenceResult(
        ell_max_values=ell_max_values,
        pi_tilde_final=pi_finals,
        Sigma_final=sigma_finals,
        relative_change=rel_changes,
        converged_ell_max=converged,
        convergence_threshold=threshold,
    )


# ═══════════════════════════════════════════════════════════════
# §3. Type I exactness test
# ═══════════════════════════════════════════════════════════════

def type_I_exactness_test(Sigma_H_0: float = 0.1, N_end: float = 10.0) -> Dict:
    """Verify that κ=0 gives identical results for all ℓ_max.

    For Type I (κ=0), the hierarchy truncates exactly at ℓ=2:
    higher ℓ modes are completely decoupled and stay at 0.
    So ℓ_max=2 and ℓ_max=10 must give identical {Σ, Ψ₂}.
    """
    results = {}
    for lmax in [2, 4, 6, 10]:
        n_ell = (lmax // 2) + 1
        y0 = np.zeros(1 + n_ell)
        y0[0] = Sigma_H_0

        sol = solve_ivp(
            lambda N, y: _curved_hierarchy_rhs(N, y, 0.0, lmax),
            [0, N_end], y0, method='Radau',
            rtol=1e-12, atol=1e-14,
        )

        results[lmax] = {
            'Sigma_final': float(sol.y[0, -1]),
            'Psi2_final': float(sol.y[2, -1]),
        }

    # All should be identical
    ref = results[2]
    max_err = 0.0
    for lmax, r in results.items():
        err_S = abs(r['Sigma_final'] - ref['Sigma_final'])
        err_P = abs(r['Psi2_final'] - ref['Psi2_final'])
        max_err = max(max_err, err_S, err_P)

    return {
        'results': results,
        'max_discrepancy': max_err,
        'exact': max_err < 1e-10,
    }
