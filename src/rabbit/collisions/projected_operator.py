"""
rabbit.collisions.projected_operator — RTA damping model for PSTF hierarchy.

Implements a RELAXATION-TIME APPROXIMATION (RTA) for the ℓ=0,2 PSTF
multipole perturbations.  This is a calibrated damping model, NOT a
physical collision integral.  In particular:

  - C[Ψ] = 0 whenever Ψ = 0, regardless of T_γ vs T_ν.
  - It cannot generate incomplete-decoupling source terms.
  - It damps existing perturbations towards equilibrium.

For a physical collision operator that generates source terms when
T_γ ≠ T_ν, see the planned q-mixing upgrade in Phase 2 (P2-A) using
the scattering kernels from nu_e_scattering.py and pair_processes.py.

Model:
    C_ℓ[Ψ_ℓ](q) = -κ_ℓ × (Γ_s/H) × Ψ_ℓ(q)

where:
    Γ_s = (C_rate/π⁵) × G_F² × a_s × F(m_e/T_γ) × T_γ⁴ × T_ν
    κ₀ = 1.0     monopole damping
    κ₂ = 5/3     quadrupole damping (higher ℓ → faster isotropization)
    C_rate = 210  calibrated constant (shared with nudec_tables.py)

Species coupling (Notzold & Raffelt 1988):
    a_e = 1 + 4sin²θ_W + 8sin⁴θ_W ≈ 2.353  (CC + NC)
    a_x = 1 − 4sin²θ_W + 8sin⁴θ_W ≈ 0.503  (NC only)
    ratio a_e/a_x ≈ 4.68

Limitations:
    - No source term: C=0 at Ψ=0 even when T_γ ≠ T_ν
    - q-independent damping rate (no momentum-dependent kernels)
    - Calibrated to reproduce N_eff ≈ 3.044, not derived from first principles

References:
    Hannestad & Madsen (1995) PRD 52, 1764
    Escudero (2019) JCAP 02:007
"""
from __future__ import annotations
from dataclasses import dataclass
import numpy as np

from rabbit.collisions.kernels import (
    G_F_MEV, A_TOTAL_NUE, A_TOTAL_NUX, COUPLING_RATIO_NUE_NUX,
)
from rabbit.thermo.nudec_tables import _C_RATE

_PI = np.pi
_M_E = 0.5109989
_MEV2S = 1.519267447e21
_G_N = 6.70883e-45

KAPPA_ELL0 = 1.0
KAPPA_ELL2 = 5.0 / 3.0

# ── ℓ=2 collision mode (BD627 W1 / BD628 W-complete) ──────────────────
# "calibrated_rta" (DEFAULT): the calibrated ansatz −κ₂ (Γ/H) Ψ₂.
# "kernelized": the candidate v1 electron-minus-only ℓ=2 kernel
#   (collisions.kernelized_ell2).  Its complete electron-sector, detailed-
#   balance, convergence, and between-node parity claims are NOT VALIDATED.
#   Default OFF: bit-identical to the calibrated path unless explicitly selected.
# "kernelized_scalar" (BD628, F-033): the ray-space CHARACTERISTIC lane's scalar
#   κ₂ → κ₂_eff(T) replacement (see transport.characteristic_residual).  It has no
#   distinct meaning in this Ψ-space projected operator — the full "kernelized" mode
#   already supersedes it here — so this operator treats it as the calibrated ansatz.
ELL2_COLLISION_MODES = ("calibrated_rta", "kernelized", "kernelized_scalar")


def _ell2_collision_source(mode, f_quadrupole_s, Gamma_H_s, *, species_tag,
                           T_gamma, T_nu, H_inv_sec, q_nodes, q_weights):
    """Shared ℓ=2 collision source for the RTA and physical operators (BD627 W1).

    Both projected-operator C₂ sites route through this one helper so the
    calibrated↔kernelized switch is applied consistently.

    mode == "calibrated_rta" (DEFAULT): returns −κ₂ (Γ/H) Ψ₂, BIT-IDENTICAL to
        the pre-BD627 code path.
    mode == "kernelized": returns the candidate v1 electron-minus-only ℓ=2 source
        (see rabbit.collisions.kernelized_ell2), in the SAME per-e-fold Ψ₂
        convention as the calibrated ansatz it replaces.
    """
    if mode == "calibrated_rta" or mode == "kernelized_scalar":
        # "kernelized_scalar" is a ray-space (characteristic_residual) concept only;
        # on this Ψ-space projected lane it collapses to the calibrated ansatz.
        return -KAPPA_ELL2 * Gamma_H_s * f_quadrupole_s
    if mode == "kernelized":
        from rabbit.collisions.kernelized_ell2 import kernelized_C2_ell2
        return kernelized_C2_ell2(
            f_quadrupole_s, T_gamma, T_nu, H_inv_sec, q_nodes, q_weights,
            species_tag)
    raise ValueError(
        f"unknown ell2_collision_mode {mode!r}; expected one of "
        f"{ELL2_COLLISION_MODES}")


def _ell2_species_tags(n_species):
    """Table combo tags per species: (ν_e, ν̄_e) → 'nue', ν_x → 'nux'."""
    return tuple("nue" if s < 2 else "nux" for s in range(n_species))


def _electron_mass_suppression(T_MeV):
    """Suppression of ν-e rates when e± are Boltzmann-suppressed."""
    if T_MeV > 50.0:
        return 1.0
    if T_MeV < 0.01:
        return 0.0
    x = _M_E / T_MeV
    return 1.0 / (1.0 + np.exp(2.5 * (x - 2.3)))


def collision_rate_per_efold(T_gamma, T_nu, H_inv_sec, a_coupling):
    """Dimensionless Γ/H for one species.  Large → coupled, small → free."""
    F = _electron_mass_suppression(T_gamma)
    Gamma_MeV = (_C_RATE / _PI**5) * G_F_MEV**2 * a_coupling * F * T_gamma**4 * T_nu
    H_MeV = H_inv_sec / _MEV2S
    if H_MeV < 1e-50:
        return 0.0
    return Gamma_MeV / H_MeV


def _hubble_simple(T_gamma, T_nu):
    """Quick Hubble rate [s⁻¹] for standalone collision tests."""
    P = _PI**2 / 15.0
    rho_g = P * T_gamma**4
    rho_e = 0.0
    if T_gamma > 0.1:
        x = _M_E / T_gamma
        if x < 20:
            rho_e = (7.0 / 8.0) * P * T_gamma**4 / (1.0 + np.exp(2.5 * (x - 2.3)))
    rho_nu = 3.044 * (7.0 / 8.0) * P * T_nu**4
    H_MeV = np.sqrt(max(8.0 * _PI * _G_N / 3.0 * (rho_g + rho_e + rho_nu), 0.0))
    return H_MeV * _MEV2S


# ─── Result ───────────────────────────────────────────────────

@dataclass
class CollisionResult:
    """Result of RTA damping evaluation.

    Note: energy_residual is NOT a conservation diagnostic in this model.
    The RTA damping transfers energy to/from the plasma without tracking
    where it goes.  For true energy conservation tests, use the planned
    physical collision operator (P2-A).
    """
    C_ell0: np.ndarray           # (n_species, N_q)
    C_ell2: np.ndarray           # (n_species, N_q)
    Gamma_over_H: np.ndarray     # (n_species,)
    energy_residual: float = 0.0
    detailed_balance_residual: float = 0.0


# ─── Operator ─────────────────────────────────────────────────

class RTADampingModel:
    """Relaxation-time approximation damping model for ℓ=0,2.

    This is a DAMPING model: it drives perturbations Ψ towards zero
    at a rate proportional to Γ/H.  It does NOT generate source terms.
    For T_γ > T_ν with Ψ = 0, it returns C = 0 (no heating).

    For a physical collision operator that generates incomplete-decoupling
    source terms, see the planned P2-A upgrade.

    Parameters
    ----------
    grid : MomentumGrid
    n_species : int
        3 = (ν_e, ν̄_e, ν_x).
    """

    def __init__(self, grid, n_species=3, ell2_collision_mode="calibrated_rta"):
        self.grid = grid
        self.n_species = n_species
        self.a_coupling = np.array([A_TOTAL_NUE, A_TOTAL_NUE, A_TOTAL_NUX])
        self.ell2_collision_mode = ell2_collision_mode
        self._ell2_species_tags = _ell2_species_tags(n_species)

    def evaluate(self, f_monopole, f_quadrupole, T_gamma, T_nu=None, H_inv_sec=None):
        """Evaluate projected collision integrals.

        Parameters
        ----------
        f_monopole : ndarray, shape (n_species, N_q) or (N_q,)
            Monopole perturbation Ψ₀ (NOT full distribution).
        f_quadrupole : ndarray, shape (n_species, N_q) or (N_q,)
            Quadrupole perturbation Ψ₂.
        T_gamma : float
            Photon temperature [MeV].
        T_nu : float or None
            Neutrino temperature [MeV].
        H_inv_sec : float or None
            Hubble rate [s⁻¹].

        Returns
        -------
        CollisionResult
        """
        if T_nu is None:
            T_nu = T_gamma * (4.0 / 11.0) ** (1.0 / 3.0) if T_gamma < 1.5 else T_gamma
        if H_inv_sec is None:
            H_inv_sec = _hubble_simple(T_gamma, T_nu)

        N_q = self.grid.N_q
        ns = self.n_species

        if f_monopole.ndim == 1:
            f_monopole = np.tile(f_monopole, (ns, 1))
        if f_quadrupole.ndim == 1:
            f_quadrupole = np.tile(f_quadrupole, (ns, 1))

        Gamma_H = np.array([
            collision_rate_per_efold(T_gamma, T_nu, H_inv_sec, self.a_coupling[s])
            for s in range(ns)
        ])

        C0 = np.zeros((ns, N_q))
        C2 = np.zeros((ns, N_q))
        for s in range(ns):
            C0[s] = -KAPPA_ELL0 * Gamma_H[s] * f_monopole[s]
            C2[s] = _ell2_collision_source(
                self.ell2_collision_mode, f_quadrupole[s], Gamma_H[s],
                species_tag=self._ell2_species_tags[s],
                T_gamma=T_gamma, T_nu=T_nu, H_inv_sec=H_inv_sec,
                q_nodes=self.grid.nodes, q_weights=self.grid.weights)

        # Energy residual: total across species (should be small, not exactly 0
        # because energy goes to/from plasma, not tracked here)
        q = self.grid.nodes
        w = self.grid.weights
        f0 = 1.0 / (np.exp(np.clip(q, -500, 500)) + 1.0)
        ew = w * np.exp(q) * q**4 * f0
        e_res = abs(sum(np.sum(ew * C0[s]) for s in range(ns)))

        return CollisionResult(
            C_ell0=C0, C_ell2=C2, Gamma_over_H=Gamma_H,
            energy_residual=e_res, detailed_balance_residual=0.0,
        )

    def detailed_balance_test(self, T_gamma):
        """C[f_eq] = 0.  Trivial in relaxation approximation (Ψ=0 → C=0)."""
        z = np.zeros((self.n_species, self.grid.N_q))
        r = self.evaluate(z, z, T_gamma)
        return max(np.max(np.abs(r.C_ell0)), np.max(np.abs(r.C_ell2)))

    def energy_conservation_test(self, f_monopole, T_gamma):
        """Total energy transfer magnitude (not strictly zero; see docstring)."""
        z = np.zeros_like(f_monopole)
        r = self.evaluate(f_monopole, z, T_gamma)
        return r.energy_residual


# ═══════════════════════════════════════════════════════════════════════
# §4. Physical collision operator with source term
# ═══════════════════════════════════════════════════════════════════════

def _fermi_dirac(q):
    """Fermi-Dirac: f₀(q) = 1/(e^q + 1)."""
    return 1.0 / (np.exp(np.minimum(np.asarray(q, dtype=float), 500.0)) + 1.0)


@dataclass
class PhysicalCollisionResult:
    """Result of physical collision operator evaluation.

    Unlike CollisionResult, tracks energy transfer to plasma
    and the source term magnitude separately from damping.
    """
    C_ell0: np.ndarray           # (n_species, N_q) — total monopole collision
    C_ell2: np.ndarray           # (n_species, N_q) — quadrupole damping
    source_ell0: np.ndarray      # (n_species, N_q) — source part only
    damping_ell0: np.ndarray     # (n_species, N_q) — damping part only
    Gamma_over_H: np.ndarray     # (n_species,) — collision rate
    energy_transfer: np.ndarray  # (n_species,) — dQ/dN per species [MeV⁵ units]
    total_energy_transfer: float # sum over species — must equal plasma exchange


class PhysicalCollisionOperator:
    """Collision operator with source term for incomplete decoupling.

    Unlike RTADampingModel, this generates NONZERO collision integrals
    when Ψ₀ = 0 but T_γ ≠ T_ν.  The source drives the neutrino
    distribution towards the electron bath temperature.

    Physics model (linearized Boltzmann in relaxation-time approximation):

        C₀(q) = (Γ_s/H) × [Ψ₀^target(q) − Ψ₀(q)]
        C₂(q) = −κ₂ × (Γ_s/H) × Ψ₂(q)

    where the target perturbation (neutrinos at T_γ instead of T_ν) is:

        Ψ₀^target(q) = (T_γ − T_ν)/T_γ × q × [1 − f₀(q)]

    This is the linearized result from:
        f_FD(q × T_ν/T_γ) − f_FD(q) = (ΔT/T_γ) × q × f₀(1−f₀)
        ⟹ Ψ₀_target = δf/f₀ = (ΔT/T_γ) × q × (1−f₀)

    Key properties:
        1. T_γ = T_ν, Ψ₀=0: C₀ = 0  (thermal equilibrium)
        2. T_γ > T_ν, Ψ₀=0: C₀ > 0  (neutrino heating — GENERATES source)
        3. Ψ₀ = Ψ₀_target: C₀ = 0   (new equilibrium at T_γ)
        4. Energy conservation: ∫ q³ f₀ C₀ dq gives net ν heating rate
           which must equal the plasma cooling rate (tracked separately)

    Limitations:
        - Linearized in ΔT/T; good to ~7% at ΔT/T = 1%
        - q-independent Γ (like RTA; momentum-dependent rate → future P2-B+)
        - No ν-ν scattering (Tier 3+)
        - No flavor oscillations (Tier 4)

    Parameters
    ----------
    grid : MomentumGrid
    n_species : int
        3 = (ν_e, ν̄_e, ν_x) where ν_x has degeneracy 4.
    """

    def __init__(self, grid, n_species=3, ell2_collision_mode="calibrated_rta"):
        self.grid = grid
        self.n_species = n_species
        self.ell2_collision_mode = ell2_collision_mode
        self._ell2_species_tags = _ell2_species_tags(n_species)
        # Coupling constants: [ν_e, ν̄_e, ν_x]
        self.a_coupling = np.array([A_TOTAL_NUE, A_TOTAL_NUE, A_TOTAL_NUX])
        # Precompute equilibrium distribution at grid nodes
        self._f0 = _fermi_dirac(grid.nodes)
        self._q = grid.nodes
        # Energy weight for quadrature: w_i × e^{q_i} × q_i³ × f₀(q_i)
        self._energy_weight = (grid.weights * np.exp(grid.nodes)
                               * grid.nodes**3 * self._f0)

    def source_term(self, T_gamma, T_nu):
        """Compute Ψ₀^target(q) for each momentum node.

        Returns
        -------
        ndarray, shape (N_q,)
            Target monopole perturbation.  Positive when T_γ > T_ν
            (heating), zero when T_γ = T_ν.
        """
        if abs(T_gamma - T_nu) < 1e-15 * max(abs(T_gamma), abs(T_nu), 1e-30):
            return np.zeros(self.grid.N_q)
        DT_over_Tg = (T_gamma - T_nu) / max(T_gamma, 1e-30)
        return DT_over_Tg * self._q * (1.0 - self._f0)

    def evaluate(self, f_monopole, f_quadrupole, T_gamma, T_nu=None,
                 H_inv_sec=None):
        """Evaluate physical collision integrals with source term.

        Parameters
        ----------
        f_monopole : ndarray, shape (n_species, N_q) or (N_q,)
            Monopole perturbation Ψ₀.
        f_quadrupole : ndarray, shape (n_species, N_q) or (N_q,)
            Quadrupole perturbation Ψ₂.
        T_gamma : float
            Photon/electron temperature [MeV].
        T_nu : float or None
            Neutrino temperature [MeV].  If None, estimated from T_gamma.
        H_inv_sec : float or None
            Hubble rate [s⁻¹].  If None, computed internally.

        Returns
        -------
        PhysicalCollisionResult
        """
        if T_nu is None:
            T_nu = T_gamma * (4.0/11.0)**(1.0/3.0) if T_gamma < 1.5 else T_gamma
        if H_inv_sec is None:
            H_inv_sec = _hubble_simple(T_gamma, T_nu)

        N_q = self.grid.N_q
        ns = self.n_species

        if f_monopole.ndim == 1:
            f_monopole = np.tile(f_monopole, (ns, 1))
        if f_quadrupole.ndim == 1:
            f_quadrupole = np.tile(f_quadrupole, (ns, 1))

        # Collision rate Γ/H per species
        Gamma_H = np.array([
            collision_rate_per_efold(T_gamma, T_nu, H_inv_sec, self.a_coupling[s])
            for s in range(ns)
        ])

        # Source term (same shape for all species, but scaled by Γ/H)
        psi0_target = self.source_term(T_gamma, T_nu)

        # Monopole: C₀ = (Γ/H) × [Ψ₀_target − Ψ₀]
        source = np.zeros((ns, N_q))
        damping = np.zeros((ns, N_q))
        C0 = np.zeros((ns, N_q))
        for s in range(ns):
            source[s] = KAPPA_ELL0 * Gamma_H[s] * psi0_target
            damping[s] = -KAPPA_ELL0 * Gamma_H[s] * f_monopole[s]
            C0[s] = source[s] + damping[s]

        # Quadrupole: C₂ = −κ₂ × (Γ/H) × Ψ₂ (calibrated) or the kernelized ν_e
        # ℓ=2 source (BD627 W1) — shared helper, default calibrated (bit-identical).
        C2 = np.zeros((ns, N_q))
        for s in range(ns):
            C2[s] = _ell2_collision_source(
                self.ell2_collision_mode, f_quadrupole[s], Gamma_H[s],
                species_tag=self._ell2_species_tags[s],
                T_gamma=T_gamma, T_nu=T_nu, H_inv_sec=H_inv_sec,
                q_nodes=self.grid.nodes, q_weights=self.grid.weights)

        # Energy transfer per species: dE_s/dN = ∫ q³ f₀ × C₀_s dq
        energy_xfer = np.array([
            np.sum(self._energy_weight * C0[s])
            for s in range(ns)
        ])

        return PhysicalCollisionResult(
            C_ell0=C0, C_ell2=C2,
            source_ell0=source, damping_ell0=damping,
            Gamma_over_H=Gamma_H,
            energy_transfer=energy_xfer,
            total_energy_transfer=float(np.sum(energy_xfer)),
        )

    def detailed_balance_test(self, T_gamma, T_nu=None):
        """C[f_eq] = 0 when T_γ = T_ν and Ψ = 0.

        For the physical operator, detailed balance requires BOTH
        T_γ = T_ν AND Ψ₀ = 0.  When T_γ ≠ T_ν, C[0] ≠ 0 is CORRECT.
        """
        if T_nu is None:
            T_nu = T_gamma  # same temperature = equilibrium
        z = np.zeros((self.n_species, self.grid.N_q))
        r = self.evaluate(z, z, T_gamma, T_nu)
        return max(np.max(np.abs(r.C_ell0)), np.max(np.abs(r.C_ell2)))

    def heating_test(self, T_gamma, T_nu):
        """Verify nonzero heating when T_γ > T_ν with Ψ = 0.

        Returns (C0_max, energy_transfer_total).
        C0_max should be > 0 when T_γ > T_ν.
        """
        z = np.zeros((self.n_species, self.grid.N_q))
        r = self.evaluate(z, z, T_gamma, T_nu)
        return float(np.max(np.abs(r.C_ell0))), r.total_energy_transfer

    def energy_conservation_test(self, T_gamma, T_nu):
        """Total ν + plasma energy conservation.

        The ν energy gain from collisions must equal the plasma energy loss.
        In our model, we only track the ν side; the plasma side is:
            dQ_plasma/dN = −Σ_s ∫ q³ f₀ C₀_s dq

        This test verifies that the energy transfer has the correct sign
        and is consistent between source and damping components.
        """
        z = np.zeros((self.n_species, self.grid.N_q))
        r = self.evaluate(z, z, T_gamma, T_nu)
        # When T_gamma > T_nu: energy flows ν ← plasma, so
        # ν energy transfer > 0 and plasma energy transfer < 0
        return r.total_energy_transfer  # positive = ν heating


# ─── Convenience for hierarchy RHS ────────────────────────────

def collision_rhs_ell0(Psi0, T_gamma, T_nu, H_inv_sec, grid, n_species=3):
    """dΨ₀/dN RTA damping source.  Shape: (n_species, N_q)."""
    op = RTADampingModel(grid, n_species)
    return op.evaluate(Psi0, np.zeros_like(Psi0), T_gamma, T_nu, H_inv_sec).C_ell0


def collision_rhs_ell2(Psi2, T_gamma, T_nu, H_inv_sec, grid, n_species=3):
    """dΨ₂/dN RTA damping sink.  Shape: (n_species, N_q)."""
    op = RTADampingModel(grid, n_species)
    return op.evaluate(np.zeros_like(Psi2), Psi2, T_gamma, T_nu, H_inv_sec).C_ell2


def physical_collision_rhs(Psi0, Psi2, T_gamma, T_nu, H_inv_sec, grid, n_species=3):
    """Full physical collision RHS with source term.

    Returns (C_ell0, C_ell2), each shape (n_species, N_q).
    """
    op = PhysicalCollisionOperator(grid, n_species)
    r = op.evaluate(Psi0, Psi2, T_gamma, T_nu, H_inv_sec)
    return r.C_ell0, r.C_ell2


# ─── Backward compatibility alias ────────────────────────────
# Previous name was ProjectedCollisionOperator.  Alias ensures
# existing imports (e.g., test_collision_operator.py) continue to work.
ProjectedCollisionOperator = RTADampingModel
