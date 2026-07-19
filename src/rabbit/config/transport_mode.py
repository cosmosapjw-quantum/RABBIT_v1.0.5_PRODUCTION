"""
rabbit.config.transport_mode — Transport evolution strategy selection.

Two modes:
    LINEARIZED_PSTF:  Current code. dΨ₂/dN = -(8/15)Σ, q-independent.
                       960 DOF (6 species × 2 ℓ × 80 q-bins).
                       Teff closure optionally reconstructs q-dependent monopole.
    CHARACTERISTIC:    Exact collisionless Boltzmann via characteristic curves.
                       f(q,μ,N) = f₀(q exp(2I(N,μ₀))). No ℓ-truncation,
                       no Ψ-linearization. 2N_μ+1 DOF (25 at N_μ=12).
                       Stress and monopole computed exactly.
"""
from __future__ import annotations
from enum import Enum


class TransportMode(Enum):
    LINEARIZED_PSTF = "linearized"
    CHARACTERISTIC = "characteristic"
