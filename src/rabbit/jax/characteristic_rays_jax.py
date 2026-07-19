"""
rabbit.jax.characteristic_rays_jax — JAX-compatible characteristic ray transport.

Pure JAX functions for JIT compilation and automatic differentiation.
See rabbit.transport.characteristic_rays for derivation and documentation.
"""
from __future__ import annotations

import jax
import jax.numpy as jnp


# ═══════════════════════════════════════════════════════════════════════
# §1. LRS characteristic map
# ═══════════════════════════════════════════════════════════════════════

def mu_current_jax(X0: jnp.ndarray, signs: jnp.ndarray, S: jnp.ndarray) -> jnp.ndarray:
    """Forward map: μ_j(S) from precomputed X0_j = μ₀²/(1-μ₀²).

    The logistic form avoids overflow when ``6 S`` is large and preserves the
    exact central-ray limit ``μ(S) = 0`` for ``X0 = 0``.
    """
    log_x0 = jnp.where(X0 > 0.0, jnp.log(X0), -jnp.inf)
    mu_sq = jax.nn.sigmoid(log_x0 + 6.0 * S)
    return signs * jnp.sqrt(jnp.minimum(mu_sq, 1.0 - 1e-15))


@jax.jit
def jacobian_jax(X0: jnp.ndarray, S: jnp.ndarray, mu: jnp.ndarray) -> jnp.ndarray:
    """Closed-form angular Jacobian in the production-code convention.

    The transport code carries ``J_j = dμ_j / dμ_{j,0}``, which is the
    quadrature-weight factor needed to push the initial Gauss-Legendre
    measure forward to the current ray directions.

    With ``X0 = μ_{j,0}² / (1 - μ_{j,0}²)``, we reconstruct
    ``|μ_{j,0}| = sqrt(X0 / (1 + X0))`` and use the exact closed form

        dμ_j / dμ_{j,0} = μ_j (1 - μ_j²) / [μ_{j,0} (1 - μ_{j,0}²)].

    This matches the existing ODE sign convention
    ``dJ_j/dN = 3 Σ₊ (1 - 3 μ_j²) J_j`` and is algebraically equivalent
    to differentiating the analytic forward map ``μ_j(S)``.

    For the central Gauss-Legendre ray at ``μ_{j,0} = 0`` (present when
    ``N_mu`` is odd), the removable singularity is resolved by the exact
    limit ``dμ_j/dμ_{j,0} -> exp(3S)``.
    """
    one_minus_mu0_sq = 1.0 / (1.0 + X0)
    mu0_abs = jnp.sqrt(jnp.maximum(X0 * one_minus_mu0_sq, 1e-30))
    one_minus_mu_sq = jnp.maximum(1.0 - mu * mu, 1e-30)
    generic = jnp.abs(mu) * one_minus_mu_sq / jnp.maximum(mu0_abs * one_minus_mu0_sq, 1e-30)
    central_limit = jnp.exp(3.0 * S)
    return jnp.where(X0 <= 1e-30, central_limit, generic)


def P2_jax(mu: jnp.ndarray) -> jnp.ndarray:
    """P₂(μ) = (3μ²-1)/2."""
    return 0.5 * (3 * mu**2 - 1)


# ═══════════════════════════════════════════════════════════════════════
# §2. Analytic characteristic reconstruction
# ═══════════════════════════════════════════════════════════════════════

@jax.jit
def intensity_shift_jax(X0: jnp.ndarray, S: jnp.ndarray) -> jnp.ndarray:
    """Closed-form characteristic energy shift ``I_j(S)``.

    Solves ``dI/dS = P₂(μ(S))`` with ``I(0) = 0`` in the LRS characteristic
    map. The log-domain form avoids overflow at large positive ``S``.
    """
    log_x0 = jnp.where(X0 > 0.0, jnp.log(X0), -jnp.inf)
    log_num = jnp.logaddexp(jnp.zeros_like(X0), log_x0 + 6.0 * S)
    return 0.25 * (log_num - jnp.log1p(X0)) - 0.5 * S


def characteristic_rhs_jax(Sigma_plus: jnp.ndarray) -> jnp.ndarray:
    """Accumulated-shear transport RHS ``dS/dN`` for JAX."""
    return jnp.asarray(Sigma_plus)


# ═══════════════════════════════════════════════════════════════════════
# §3. Observable extraction
# ═══════════════════════════════════════════════════════════════════════

def extract_stress_jax(
    I: jnp.ndarray,
    J: jnp.ndarray,
    mu: jnp.ndarray,
    w0: jnp.ndarray,
    f_nu: float,
) -> jnp.ndarray:
    """Π₊ = f_ν Σ_j w_j J_j P₂(μ_j) exp(-8I_j)."""
    P2 = P2_jax(mu)
    return f_nu * jnp.sum(w0 * J * P2 * jnp.exp(-8 * I))


def extract_monopole_jax(
    I: jnp.ndarray,
    J: jnp.ndarray,
    w0: jnp.ndarray,
    q_nodes: jnp.ndarray,
) -> jnp.ndarray:
    """f_mono(q) = (1/2) Σ_j w_j J_j f₀(q exp(2I_j)). Vectorized."""
    alpha = jnp.exp(2 * I)                              # (N_mu,)
    qa = q_nodes[:, None] * alpha[None, :]               # (N_q, N_mu)
    f_vals = 1.0 / (jnp.exp(jnp.minimum(qa, 500)) + 1)  # (N_q, N_mu)
    return 0.5 * f_vals @ (w0 * J)                       # (N_q,)
