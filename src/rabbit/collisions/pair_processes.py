"""
rabbit.collisions.pair_processes — Pair annihilation/creation kernel.

Implements the collision integral for ν_α(y₁) + ν̄_α(y₂) ↔ e⁺(y₃) + e⁻(y₄)
in the ultra-relativistic massless limit.

For ν_e: both CC and NC channels contribute.
For ν_x: NC only.

The coupling constants are the same as for scattering (G_L, G_R)
but the Mandelstam variable assignments differ (s-channel for
annihilation vs t-channel for scattering).

Like scattering, the collision integral satisfies:
    1. Detailed balance: C[f_eq] = 0
    2. Energy conservation: ∫ y³ C[f] dy = 0

but does NOT conserve neutrino number (∫ y² C dy ≠ 0 in general).
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from numpy.polynomial.laguerre import laggauss
from numpy.polynomial.legendre import leggauss

from rabbit.collisions.deterministic_reference import collision_statistical_factor
from rabbit.collisions.kernels import (
    G_F_MEV,
    G_L_NUE,
    G_R_NUE,
    G_L_NUX,
    G_R_NUX,
    hm_reduced_collision_prefactor,
)


def _fermi_dirac(y):
    """Fermi–Dirac: f(y) = 1/(e^y + 1)."""
    return 1.0 / (np.exp(np.minimum(np.asarray(y, dtype=float), 500.0)) + 1.0)


@dataclass
class PairProcessOperator:
    """Pair annihilation/creation collision operator.

    ν(y₁) + ν̄(y₂) ↔ e⁺(y₃) + e⁻(y₄)

    Parameters
    ----------
    G_L, G_R : float
        Weak coupling constants.
    N_quad : int
        Quadrature nodes for integrals.
    """
    G_L: float = G_L_NUE
    G_R: float = G_R_NUE
    N_quad: int = 24

    def __post_init__(self):
        self._nodes, self._weights = laggauss(self.N_quad)
        self._leg_nodes, self._leg_weights = leggauss(self.N_quad)
        self._GL2_GR2 = self.G_L**2 + self.G_R**2

    @property
    def coupling_strength(self) -> float:
        return self._GL2_GR2

    def _statistical_factor(self, f1, f2, f3, f4) -> float:
        """Return the shared quartic-cancelled six-monomial Pauli factor."""

        return float(collision_statistical_factor(f1, f2, f3, f4))

    def _matrix_element_ann(self, y1, y2, y3, y4):
        """Angular-averaged |M|² for pair annihilation (UR massless).

        For annihilation ν(1)+ν̄(2)→e⁺(3)+e⁻(4):
        The s-channel dominates; matrix element ∝ (G_L²+G_R²)(t²+u²).
        """
        GL2_GR2 = self._GL2_GR2
        GL2_mGR2 = self.G_L**2 - self.G_R**2
        M2 = (GL2_GR2 * ((y1 * y3)**2 + (y2 * y4)**2)
              + GL2_mGR2 * ((y1 * y4)**2 + (y2 * y3)**2))
        return M2

    def evaluate(
        self,
        f_nu: np.ndarray,
        f_nubar: np.ndarray,
        q_nodes: np.ndarray,
        T_MeV: float,
    ) -> np.ndarray:
        """Evaluate pair process collision integral for ν.

        Parameters
        ----------
        f_nu : ndarray
            Neutrino distribution at momentum nodes.
        f_nubar : ndarray
            Antineutrino distribution at momentum nodes.
        q_nodes : ndarray
            Momentum grid nodes.
        T_MeV : float
            Temperature [MeV].

        Returns
        -------
        ndarray
            C_pair[f](y_i) at each node [MeV].
        """
        from scipy.interpolate import interp1d

        N_q = len(q_nodes)
        C = np.zeros(N_q)
        prefactor = hm_reduced_collision_prefactor(T_MeV)

        y2_nodes, y2_weights = self._nodes, self._weights

        f_nu_interp = interp1d(q_nodes, f_nu, kind='cubic',
                               bounds_error=False, fill_value=(f_nu[0], 0.0))
        f_nubar_interp = interp1d(q_nodes, f_nubar, kind='cubic',
                                  bounds_error=False, fill_value=(f_nubar[0], 0.0))

        for i in range(N_q):
            y1 = q_nodes[i]
            if y1 < 1e-15:
                continue
            f1 = f_nu[i]

            total = 0.0
            for k in range(len(y2_nodes)):  # ν̄ momentum y₂
                y2 = y2_nodes[k]
                f2 = float(np.clip(f_nubar_interp(y2), 0, 1))

                # e⁺e⁻ momenta: integrate over y₃ on the finite interval
                # [0, y₁ + y₂] using Gauss-Legendre, with y₄ = y₁ + y₂ − y₃.
                y_sum = y1 + y2
                if y_sum < 1e-15:
                    continue

                y3_nodes = 0.5 * y_sum * (self._leg_nodes + 1.0)
                y3_weights = 0.5 * y_sum * self._leg_weights

                for m in range(len(y3_nodes)):
                    y3 = y3_nodes[m]
                    y4 = y_sum - y3

                    # Electron distributions (thermal)
                    f3 = float(_fermi_dirac(y3))
                    f4 = float(_fermi_dirac(y4))

                    M2 = self._matrix_element_ann(y1, y2, y3, y4)

                    # Gain: e⁺e⁻ → ν ν̄ (creation)
                    # Loss: ν ν̄ → e⁺e⁻ (annihilation)
                    S = self._statistical_factor(f1, f2, f3, f4)

                    total += (y2_weights[k] * np.exp(y2)
                              * y3_weights[m]
                              * M2 * S)

            C[i] = prefactor * total / max(y1**2, 1e-30)

        return C

    def total_rate(self, T_MeV: float) -> float:
        """Approximate pair annihilation rate."""
        return (7.0 * np.pi / 12.0) * G_F_MEV**2 * T_MeV**5 * self._GL2_GR2


def nu_e_pair_operator() -> PairProcessOperator:
    """Pair process operator for ν_e (CC + NC)."""
    return PairProcessOperator(G_L=G_L_NUE, G_R=G_R_NUE)


def nu_x_pair_operator() -> PairProcessOperator:
    """Pair process operator for ν_x (NC only)."""
    return PairProcessOperator(G_L=G_L_NUX, G_R=G_R_NUX)
