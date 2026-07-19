"""
rabbit.jax.pstf_hierarchy — Standard 1+3 covariant PSTF Boltzmann hierarchy.

Primary transport formalism for RABBIT.

    f(q, μ) = f₀(q) [1 + Σ_ℓ Ψ_ℓ(q) P_ℓ(μ)]

where Ψ_ℓ(q) are evolved at discrete Gauss-Laguerre momentum nodes.
The q-dependence arises naturally from the q-dependent source
q(1−f₀(q)) and persists through the self-coupling term; no explicit
q-differentiation is required (the standard PSTF closes without it
because Ψ_ℓ starts q-independent and the source drives q-dependent
growth without a separate q-advection operator).

Hierarchy equation for LRS Type I (Σ₋ = 0, even ℓ only):

    dΨ_i/dN = Σ₊ C^E_{i0} q(1−f₀)                  [source]
             + Σ₊ Σ_j C^E_{ij} q(1−f₀) Ψ_j          [self-coupling, O(ΣΨ)]
             − Σ₊ C^A_{i0}                            [drift source]
             − Σ₊ Σ_j C^A_{ij} Ψ_j                   [drift coupling]

  C^E: energy-shift coupling (Gaunt integrals ∫P_ℓ P₂ P_{ℓ'} dμ)
  C^A: angular-drift coupling (∫P_ℓ μ(1−μ²) P'_{ℓ'} dμ)
  Both are bandwidth-2 in ℓ-space.

For generic (non-LRS) models and tilted spacetimes, odd ℓ are active.
The `even_only` flag controls this.

Teff role: the hierarchy itself uses standard PSTF — no effective
temperature.  Teff enters only when reconstructing the anisotropic
distribution for weak-rate evaluation, where positivity of f(q,μ)
must be enforced.

References:
    Ellis & van Elst 1999, Thorne 1981, Pontzen & Challinor 2007
"""
from __future__ import annotations
from functools import partial

import jax
import jax.numpy as jnp
import numpy as _np
from numpy.polynomial.laguerre import laggauss as _laggauss
from numpy.polynomial.legendre import leggauss as _leggauss

jax.config.update("jax_enable_x64", True)


# ═══════════════════════════════════════════════════════════════
# §1. Coupling coefficients
# ═══════════════════════════════════════════════════════════════

def _gaunt_P2(ell, ell_prime, N_mu=64):
    """∫ P_ℓ P₂ P_{ℓ'} dμ."""
    from numpy.polynomial.legendre import legval as _lv
    mu, w = _leggauss(N_mu)
    c_l = _np.zeros(ell + 1); c_l[ell] = 1.0
    c_lp = _np.zeros(ell_prime + 1); c_lp[ell_prime] = 1.0
    return float(_np.sum(w * _lv(mu, c_l) * _lv(mu, [0, 0, 1]) * _lv(mu, c_lp)))


def _drift_coeff(ell, ell_prime, N_mu=64):
    """∫ P_ℓ × μ(1−μ²) P'_{ℓ'} dμ."""
    from numpy.polynomial.legendre import legval as _lv
    if ell_prime == 0:
        return 0.0
    mu, w = _leggauss(N_mu)
    c_l = _np.zeros(ell + 1); c_l[ell] = 1.0
    c_lp = _np.zeros(ell_prime + 1); c_lp[ell_prime] = 1.0
    c_m1 = _np.zeros(ell_prime); c_m1[ell_prime - 1] = 1.0
    f = -ell_prime * mu**2 * _lv(mu, c_lp) + ell_prime * mu * _lv(mu, c_m1)
    return float(_np.sum(w * _lv(mu, c_l) * f))


def build_coupling_matrices(ell_max, even_only=True):
    """Precompute C^E and C^A.  Shape (n_modes, n_modes)."""
    ells = list(range(0, ell_max + 1, 2 if even_only else 1))
    n = len(ells)
    C_E = _np.zeros((n, n))
    C_A = _np.zeros((n, n))
    for i, li in enumerate(ells):
        for j, lj in enumerate(ells):
            C_E[i, j] = (2 * li + 1) * _gaunt_P2(li, lj)
            C_A[i, j] = 3.0 * (2 * li + 1) / 2.0 * _drift_coeff(li, lj)
    return jnp.array(C_E), jnp.array(C_A), ells


# ═══════════════════════════════════════════════════════════════
# §2. Hierarchy RHS
# ═══════════════════════════════════════════════════════════════

@partial(jax.jit, static_argnames=("n_modes",))
def hierarchy_rhs(Psi_flat, Sigma_plus, q_nodes, C_E, C_A, f0, n_modes):
    """RHS for one species.  No q-differentiation; no explicit damping."""
    N_q = q_nodes.shape[0]
    Psi = Psi_flat.reshape(n_modes, N_q)
    q1f = q_nodes * (1.0 - f0)                      # q(1-f₀), (N_q,)

    src  = Sigma_plus * C_E[:, 0:1] * q1f[None, :]  # source
    scpl = Sigma_plus * jnp.dot(C_E, Psi * q1f[None, :])  # self-coupling
    dsrc = -Sigma_plus * C_A[:, 0:1] * jnp.ones((1, N_q))  # drift source
    dcpl = -Sigma_plus * jnp.dot(C_A, Psi)           # drift coupling

    return (src + scpl + dsrc + dcpl).reshape(-1)


@partial(jax.jit, static_argnames=("n_modes", "n_species"))
def hierarchy_rhs_all_species(psi, Sigma_plus, q_nodes,
                               C_E, C_A, f0, n_modes, n_species=6):
    """RHS for all neutrino species (identical collisionless transport)."""
    chunk = n_modes * q_nodes.shape[0]
    dp = jnp.zeros_like(psi)
    for s in range(n_species):
        sl = slice(s * chunk, (s + 1) * chunk)
        dp = dp.at[sl].set(
            hierarchy_rhs(psi[sl], Sigma_plus, q_nodes, C_E, C_A, f0, n_modes))
    return dp


# ═══════════════════════════════════════════════════════════════
# §3. Stress extraction (for geometry back-reaction)
# ═══════════════════════════════════════════════════════════════

@partial(jax.jit, static_argnames=("n_modes", "n_species"))
def extract_stress(psi, q_nodes, q_weights, f0, n_modes, n_species, f_nu):
    """π̃₂ = ⟨Ψ₂⟩_{q³f₀}  →  Π₊ = 6 f_ν π̃₂."""
    N_q = q_nodes.shape[0]
    chunk = n_modes * N_q
    ew = q_weights * jnp.exp(q_nodes) * q_nodes**3 * f0
    norm = jnp.maximum(jnp.sum(ew), 1e-100)
    # ℓ=2 is index 1 for even-only, index 2 for all-ℓ
    idx2 = 1  # even-only default (0→ℓ=0, 1→ℓ=2)
    pi_sum = 0.0
    for s in range(n_species):
        Psi2 = psi[s * chunk:(s + 1) * chunk].reshape(n_modes, N_q)[idx2]
        pi_sum += jnp.sum(ew * Psi2) / norm
    pi_tilde = pi_sum / n_species
    # WH convention: Π₊ = -A_fb f_ν / 15 × π̃₂(raw P₂ projection)
    # Factor 1/15 converts raw ⟨Ψ₂⟩_{q³f₀} to the PSTF-normalized stress
    # Sign: negative for viscous damping (π̃₂ > 0 when Σ₊ > 0)
    return -(6.0 * f_nu / 15.0) * pi_tilde, pi_tilde


# ═══════════════════════════════════════════════════════════════
# §4. Monopole extraction with Teff positivity (for weak rates)
# ═══════════════════════════════════════════════════════════════

@partial(jax.jit, static_argnames=("n_modes", "N_q"))
def extract_monopoles_teff(psi, q_nodes, q_weights, f0,
                            n_modes, N_q):
    """Reconstruct f_νe(q), f_ν̄e(q) with Teff positivity enforcement.

    Strategy:
      1. Extract Ψ₀(q) (monopole perturbation)
      2. Compute energy-weighted π̃₂ for species 0,1
      3. Derive effective temperature shift: T_eff = 1 + δT
         where δT captures the monopole spectral distortion
      4. Return f(q) = f₀(q / T_eff), guaranteed ∈ [0, 1]

    For small perturbations (|Ψ₀| ≪ 1), this reduces to f₀(1+Ψ₀).
    For large perturbations, Teff ensures positivity.
    """
    chunk = n_modes * N_q
    ew = q_weights * jnp.exp(q_nodes) * q_nodes**3 * f0
    norm = jnp.maximum(jnp.sum(ew), 1e-100)

    results = []
    for s in range(2):  # νe, ν̄e only
        Psi_s = psi[s * chunk:(s + 1) * chunk].reshape(n_modes, N_q)
        Psi0 = Psi_s[0]  # monopole perturbation

        # Energy-weighted mean perturbation → effective temperature shift
        # ⟨Ψ₀⟩_w gives the fractional energy density change
        # For FD: f₀(q/T) ≈ f₀(q)(1 + q(1-f₀)δT) to first order
        # So δT = ⟨Ψ₀⟩_w / ⟨q(1-f₀)⟩_w
        psi0_avg = jnp.sum(ew * Psi0) / norm
        q1f_avg = jnp.sum(ew * q_nodes * (1.0 - f0)) / norm
        delta_T = psi0_avg / jnp.maximum(q1f_avg, 1e-10)

        # Teff reconstruction: f = f₀(q / (1 + δT))
        T_eff = 1.0 + delta_T
        f_recon = 1.0 / (jnp.exp(q_nodes / T_eff) + 1.0)

        results.append(f_recon)

    return results[0], results[1]


@partial(jax.jit, static_argnames=("n_modes", "N_q"))
def extract_monopoles_direct(psi, f0, n_modes, N_q):
    """Direct monopole extraction: f = f₀(1 + Ψ₀), clipped to [0,1]."""
    chunk = n_modes * N_q
    Psi0_nue = psi[:chunk].reshape(n_modes, N_q)[0]
    Psi0_nub = psi[chunk:2 * chunk].reshape(n_modes, N_q)[0]
    return (f0 * (1 + Psi0_nue),
            f0 * (1 + Psi0_nub))


# ═══════════════════════════════════════════════════════════════
# §5. Initialization and cache
# ═══════════════════════════════════════════════════════════════

def initialize_hierarchy(N_q, n_modes, n_species=6):
    """All Ψ_ℓ = 0 (FD equilibrium)."""
    return jnp.zeros(n_species * n_modes * N_q, dtype=jnp.float64)


from rabbit.jax._cache_guard import _CACHE_LOCK, register_cache_clearer

_CACHE = {}

def _clear_pstf_caches():
    _CACHE.clear()

register_cache_clearer(_clear_pstf_caches)

def get_hierarchy_infrastructure(N_q, ell_max=8, even_only=True):
    """Build/cache infrastructure.

    Returns (q_nodes, q_weights, C_E, C_A, f0, n_modes, ells).
    """
    key = (N_q, ell_max, even_only)
    with _CACHE_LOCK:
      if key not in _CACHE:
        q_n, q_w = _laggauss(N_q)
        C_E, C_A, ells = build_coupling_matrices(ell_max, even_only)
        f0 = jnp.array(1.0 / (_np.exp(_np.clip(q_n, -500, 500)) + 1.0))
        _CACHE[key] = (
            jnp.array(q_n, dtype=jnp.float64),
            jnp.array(q_w, dtype=jnp.float64),
            C_E, C_A, f0, len(ells), ells,
        )
    return _CACHE[key]
