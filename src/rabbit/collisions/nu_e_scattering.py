"""
rabbit.collisions.nu_e_scattering — Neutrino–electron scattering kernel.

Implements the 2→2 collision integral for ν(y₁) + e±(y₂) → ν(y₃) + e±(y₄)
in the ultra-relativistic massless limit (appropriate for T ≳ m_e during
neutrino decoupling).

After Hannestad–Madsen angular integration, the collision integral reduces
to a 1D integral over the outgoing neutrino momentum y₃, with the
electron thermal average built into the kernel.

Coupling constants:
    ν_e + e: CC + NC → G_L = 1/2 + sin²θ_W,  G_R = sin²θ_W
    ν_x + e: NC only → G_L = -1/2 + sin²θ_W,  G_R = sin²θ_W

The collision integral satisfies:
    1. Detailed balance: C[f_eq] = 0 (statistical factor vanishes identically)
    2. Energy conservation: ∫ y³ C[f] dy = 0 (from 4-momentum conservation)

References:
    Hannestad & Madsen, Phys. Rev. D 52, 1764 (1995)
    Dolgov et al., Nucl. Phys. B 503, 426 (1997)
    Bennett et al., JCAP 04, 073 (2020)
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import numpy as np
from numpy.polynomial.laguerre import laggauss

from rabbit.collisions.deterministic_reference import collision_statistical_factor
from rabbit.collisions.kernels import (
    G_F_MEV,
    G_L_NUE,
    G_R_NUE,
    G_L_NUX,
    G_R_NUX,
    A_TOTAL_NUE, A_TOTAL_NUX,
    hm_reduced_collision_prefactor,
)


# ═══════════════════════════════════════════════════════════════════════
# §1. Equilibrium distributions
# ═══════════════════════════════════════════════════════════════════════

def _fermi_dirac(y: np.ndarray) -> np.ndarray:
    """Fermi–Dirac distribution f₀(y) = 1/(e^y + 1)."""
    return 1.0 / (np.exp(np.minimum(y, 500.0)) + 1.0)


# ═══════════════════════════════════════════════════════════════════════
# §2. Scattering collision operator
# ═══════════════════════════════════════════════════════════════════════

@dataclass
class NuEScatteringOperator:
    """Neutrino–electron scattering collision operator.

    Computes C_scat[f](y_i) for each momentum node using a 2D
    quadrature over the outgoing neutrino momentum y₃ and the
    incoming electron momentum y₂.

    In the UR massless limit, the angular-integrated matrix element
    for ν(y₁) + e(y₂) → ν(y₃) + e(y₄) with y₄ = y₁ + y₂ − y₃ is:

        |M|² ∝ (G_L² + G_R²) × [(y₁y₂)² + (y₃y₄)²]
                + (G_L² - G_R²) × [(y₁y₄)² + (y₂y₃)²]

    This is the Mandelstam-variable result after angular averaging.

    Parameters
    ----------
    G_L : float
        Left-handed coupling.
    G_R : float
        Right-handed coupling.
    N_quad_electron : int
        Gauss–Laguerre nodes for the electron momentum integral.
    """
    G_L: float = G_L_NUE
    G_R: float = G_R_NUE
    N_quad_electron: int = 32

    def __post_init__(self):
        self._e_nodes, self._e_weights = laggauss(self.N_quad_electron)
        self._GL2_GR2 = self.G_L**2 + self.G_R**2
        self._GL2_mGR2 = self.G_L**2 - self.G_R**2

    @property
    def coupling_strength(self) -> float:
        """Total coupling: G_L² + G_R²."""
        return self._GL2_GR2

    def _matrix_element(self, y1, y2, y3, y4):
        """Angular-averaged |M|² for ν-e scattering (UR massless).

        Symmetric form: includes both s- and t-channel contributions.
        """
        # s-channel: (y₁y₂)² + (y₃y₄)²
        # t-channel: (y₁y₄)² + (y₂y₃)²
        M2 = (self._GL2_GR2 * ((y1 * y2)**2 + (y3 * y4)**2)
              + self._GL2_mGR2 * ((y1 * y4)**2 + (y2 * y3)**2))
        return M2

    def _statistical_factor(self, f1, f2, f3, f4):
        """Gain − loss for 2→2 process.

        The shared deterministic helper evaluates the quartic-cancelled
        six-monomial Pauli polynomial.  That keeps this legacy SciPy operator
        aligned with the augmented PSTF collision reference path.
        """
        return float(collision_statistical_factor(f1, f2, f3, f4))

    def evaluate(
        self,
        f_nu: np.ndarray,
        q_nodes: np.ndarray,
        T_MeV: float,
    ) -> np.ndarray:
        """Evaluate the scattering collision integral at each momentum node.

        Parameters
        ----------
        f_nu : ndarray, shape (N_q,)
            Neutrino distribution f(y_i) at each comoving momentum node.
        q_nodes : ndarray, shape (N_q,)
            Gauss–Laguerre momentum nodes (y = q = p/T).
        T_MeV : float
            Plasma temperature [MeV].

        Returns
        -------
        ndarray, shape (N_q,)
            C[f](y_i) in units of [MeV] (natural units: energy = rate).
            Divide by H to convert to d/dN.
        """
        N_q = len(q_nodes)
        C = np.zeros(N_q)

        prefactor = hm_reduced_collision_prefactor(T_MeV)

        # Electron momentum quadrature: y₂ = Laguerre node × 1 (since T_e = T_ν)
        y2_nodes = self._e_nodes
        y2_weights = self._e_weights

        for i in range(N_q):
            y1 = q_nodes[i]
            if y1 < 1e-15:
                continue

            f1 = f_nu[i]
            total = 0.0

            # Integrate over outgoing ν momentum y₃
            for j in range(N_q):
                y3 = q_nodes[j]
                f3 = f_nu[j]

                # Integrate over electron momentum y₂
                for k in range(len(y2_nodes)):
                    y2 = y2_nodes[k]
                    y4 = y1 + y2 - y3

                    if y4 <= 0.0:
                        continue

                    # Electron distributions (thermal at same T)
                    f2 = _fermi_dirac(np.array([y2]))[0]
                    f4 = _fermi_dirac(np.array([y4]))[0]

                    M2 = self._matrix_element(y1, y2, y3, y4)
                    S = self._statistical_factor(f1, f2, f3, f4)

                    # Laguerre weight includes e^{y₂} factor
                    total += (q_nodes[j] * q_nodes[j]  # Jacobian for y₃ integral
                              * y2_weights[k] * np.exp(y2)  # undo Laguerre weight
                              * M2 * S)

            # Weight for y₃ integration (using same Gauss-Laguerre grid)
            # We approximate ∫dy₃ ≈ Σ_j w_j e^{y₃_j} × integrand
            # But we already included q_nodes[j]² above, so we need
            # the proper measure: ∫₀^∞ dy₃ ≈ Σ_j w_j e^{y₃_j} for Laguerre

            # Actually, let me restructure this properly.
            # ∫₀^∞ g(y₃) dy₃ = ∫₀^∞ [g(y₃) e^{y₃}] e^{-y₃} dy₃ ≈ Σ_j w_j [g(y₃_j) e^{y₃_j}]
            C[i] = 0.0  # reset

        # ── Restructured clean implementation ──
        # Use the same Laguerre grid for both y₃ and y₂ integrals
        for i in range(N_q):
            y1 = q_nodes[i]
            if y1 < 1e-15:
                continue
            f1 = f_nu[i]

            integral = 0.0
            for j in range(N_q):  # y₃ integral
                y3 = q_nodes[j]
                f3 = f_nu[j]
                w3 = q_nodes[j]  # not used, see below

                inner = 0.0
                for k in range(len(y2_nodes)):  # y₂ integral
                    y2 = y2_nodes[k]
                    y4 = y1 + y2 - y3
                    if y4 <= 0:
                        continue

                    f2 = _fermi_dirac(np.array([y2]))[0]
                    f4 = _fermi_dirac(np.array([y4]))[0]

                    M2 = self._matrix_element(y1, y2, y3, y4)
                    S = self._statistical_factor(f1, f2, f3, f4)

                    # Laguerre weight for y₂: w_k × e^{y₂_k} to undo e^{-y₂}
                    inner += y2_weights[k] * np.exp(y2) * M2 * S

                # Laguerre weight for y₃: w_j × e^{y₃_j}
                # Using the momentum grid weights for y₃
                # Since q_nodes are Laguerre nodes: ∫ g(y) e^{-y} dy ≈ Σ w_j g(y_j)
                # So ∫ g(y) dy ≈ Σ w_j e^{y_j} g(y_j)
                # But we don't have weights for q_nodes here (they're transport grid)
                # Use a simple approach: treat q_nodes as a Gauss-Laguerre grid
                # and use the associated weights
                pass

            C[i] = prefactor * integral / max(y1**2, 1e-30)

        # ── Vectorized clean implementation ──
        return self._evaluate_vectorized(f_nu, q_nodes, T_MeV)

    def _evaluate_vectorized(
        self,
        f_nu: np.ndarray,
        q_nodes: np.ndarray,
        T_MeV: float,
    ) -> np.ndarray:
        """Vectorized collision integral computation.

        Uses double Gauss–Laguerre quadrature: one for the outgoing
        neutrino y₃ and one for the incoming electron y₂.
        """
        from numpy.polynomial.laguerre import laggauss

        N_q = len(q_nodes)
        C = np.zeros(N_q)

        prefactor = hm_reduced_collision_prefactor(T_MeV)

        # Use dedicated quadrature grids for the integration
        # (separate from the momentum grid where f is defined)
        N_int = min(N_q, 24)  # integration nodes
        y3_nodes, y3_weights = laggauss(N_int)
        y2_nodes, y2_weights = laggauss(self.N_quad_electron)

        # Interpolate f_nu onto the integration grid
        from scipy.interpolate import interp1d
        f_interp = interp1d(q_nodes, f_nu, kind='cubic',
                            bounds_error=False,
                            fill_value=(f_nu[0], 0.0))

        for i in range(N_q):
            y1 = q_nodes[i]
            if y1 < 1e-15:
                continue
            f1 = f_nu[i]

            total = 0.0
            for j in range(N_int):  # y₃
                y3 = y3_nodes[j]
                f3 = float(np.clip(f_interp(y3), 0, 1))

                for k in range(len(y2_nodes)):  # y₂
                    y2 = y2_nodes[k]
                    y4 = y1 + y2 - y3
                    if y4 <= 0:
                        continue

                    f2 = float(_fermi_dirac(np.array([y2]))[0])
                    f4 = float(_fermi_dirac(np.array([y4]))[0])

                    M2 = self._matrix_element(y1, y2, y3, y4)
                    S = self._statistical_factor(f1, f2, f3, f4)

                    # Both integrals use Laguerre: ∫ g(y) dy = Σ w e^y g(y)
                    total += (y3_weights[j] * np.exp(y3)
                              * y2_weights[k] * np.exp(y2)
                              * M2 * S)

            C[i] = prefactor * total / max(y1**2, 1e-30)

        return C

    def total_rate(self, T_MeV: float) -> float:
        """Total ν-e scattering rate Γ at temperature T.

        Γ = (7π/12) × G_F² × T⁵ × a_α

        where a_α = G_L² + G_R² (scattering only, single channel).

        For the complete rate (all channels including annihilation):
        Use ``total_rate_all_channels()``.

        Returns
        -------
        float
            Rate [MeV] (= [s⁻¹] in natural units).
        """
        return (7.0 * np.pi / 12.0) * G_F_MEV**2 * T_MeV**5 * self._GL2_GR2

    def total_rate_all_channels(self, T_MeV: float) -> float:
        """Total rate including scattering + annihilation.

        Γ_total = (7π/12) × G_F² × T⁵ × a_total

        where a_total = 1 ± 4sin²θ_W + 8sin⁴θ_W (Notzold & Raffelt).
        """
        if abs(self.G_L - G_L_NUE) < 1e-10:
            a = A_TOTAL_NUE
        else:
            a = A_TOTAL_NUX
        return (7.0 * np.pi / 12.0) * G_F_MEV**2 * T_MeV**5 * a

    def energy_transfer_rate(
        self,
        f_nu: np.ndarray,
        q_nodes: np.ndarray,
        q_weights: np.ndarray,
        T_MeV: float,
    ) -> float:
        """Energy transfer rate dQ/dt = ∫ y³ C[f](y) w(y) dy.

        Should vanish for energy-conserving collisions.
        """
        C = self._evaluate_vectorized(f_nu, q_nodes, T_MeV)
        # Energy density integral: ∫ y³ C(y) dy ≈ Σ w_i e^{y_i} y_i³ C_i
        return float(np.sum(q_weights * np.exp(q_nodes) * q_nodes**3 * C))


# ═══════════════════════════════════════════════════════════════════════
# §3. Factory functions
# ═══════════════════════════════════════════════════════════════════════

def nu_e_scattering_operator() -> NuEScatteringOperator:
    """Create ν_e scattering operator (CC + NC)."""
    return NuEScatteringOperator(G_L=G_L_NUE, G_R=G_R_NUE)


def nu_x_scattering_operator() -> NuEScatteringOperator:
    """Create ν_x scattering operator (NC only)."""
    return NuEScatteringOperator(G_L=G_L_NUX, G_R=G_R_NUX)
