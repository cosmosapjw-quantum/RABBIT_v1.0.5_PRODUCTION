"""Host SciPy BBN likelihoods and unavailable JAX sampler wrappers.

The scalar likelihood factories remain available for host-side diagnostic
inference.  The BBN-specific NUTS/NSS wrappers fail closed until B-05 because
the current forward solver is not traceable through JAX sampler control flow.
Generic sampler engines live in :mod:`rabbit.inference.jax_nuts` and
:mod:`rabbit.inference.jax_nested` and are unaffected by this boundary.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import jax.numpy as jnp
import numpy as np

from rabbit.inference.observables import (
    BBN_JAX_SAMPLER_UNAVAILABLE,
    validate_prediction_for_inference,
)


# ═══════════════════════════════════════════════════════════════
# §1. Observational constraints
# ═══════════════════════════════════════════════════════════════
#
# Loaded from the versioned registry (rabbit.data, foundation F12).
# eta is exposed as eta_10 = eta * 1e10 for back-compat with the
# inline tuples this module previously held.

from rabbit.data import load_observations as _load_observations

_OBS = _load_observations(version="1.0")
YP_OBS, YP_ERR = _OBS["Yp"].value, _OBS["Yp"].sigma
DH_OBS, DH_ERR = _OBS["DH"].value, _OBS["DH"].sigma
ETA_OBS, ETA_ERR = _OBS["eta"].value * 1e10, _OBS["eta"].sigma * 1e10  # ×10^10
TAUN_OBS, TAUN_ERR = _OBS["tau_n"].value, _OBS["tau_n"].sigma


# ═══════════════════════════════════════════════════════════════
# §2. Removed high-level JAX forward compatibility boundary
# ═══════════════════════════════════════════════════════════════

def _jax_forward_solve(
    Sigma_H: float,
    eta: float,
    tau_n: float,
    N_q: int = 20,
    correction_level: int = 0,
    use_live_weak: bool = True,
) -> dict:
    """Fail closed for callers of the removed high-level Type-I JAX solve."""
    raise RuntimeError(BBN_JAX_SAMPLER_UNAVAILABLE)


def _scipy_forward_solve(
    Sigma_H: float,
    eta: float,
    tau_n: float,
    N_q: int = 20,
    correction_level: int = 0,
) -> dict:
    """Run SciPy characteristic Type I BBN solver and return observables."""
    from rabbit.drivers.full_coupled_typeI import run_full_coupled_typeI, FullCoupledConfig
    from rabbit.config.transport_mode import TransportMode

    cfg = FullCoupledConfig(
        Sigma_H_plus=Sigma_H,
        eta=eta,
        tau_n=tau_n,
        N_q=N_q,
        correction_level=correction_level,
        transport_mode=TransportMode.CHARACTERISTIC,
    )
    try:
        result = run_full_coupled_typeI(cfg)
        yp = float(result.observables.Yp)
        dh = float(result.observables.DH)
        return {'Yp': yp, 'DH': dh, 'success': bool(np.isfinite(yp) and np.isfinite(dh))}
    except Exception as e:
        return {'Yp': float('nan'), 'DH': float('nan'), 'success': False, 'error': str(e)}


# ═══════════════════════════════════════════════════════════════
# §3. Log-likelihood / log-posterior (Python-side, for NSS)
# ═══════════════════════════════════════════════════════════════

def make_log_likelihood(
    Sigma_H: float = 0.0,
    N_q: int = 20,
    correction_level: int = 0,
    backend: str = "scipy",
    obs_Yp: tuple = (YP_OBS, YP_ERR),
    obs_DH: tuple = (DH_OBS, DH_ERR),
):
    """Create a log-likelihood function for (η₁₀, τ_n) at fixed Σ_H.

    Parameters
    ----------
    Sigma_H : fixed shear value
    backend : "scipy" (temporary reference)

    Returns
    -------
    loglike(params) where params = [η₁₀, τ_n]
    """
    if backend != "scipy":
        raise ValueError(
            "Only backend='scipy' remains available for host BBN likelihoods; "
            "the high-level Type-I JAX runtime was removed."
        )
    solve_fn = _scipy_forward_solve

    def loglike(params):
        eta_10 = float(params[0])
        tau_n = float(params[1])
        r = solve_fn(Sigma_H, eta_10 * 1e-10, tau_n, N_q, correction_level)
        if not validate_prediction_for_inference(r):
            return -np.inf
        ll = (-0.5 * ((r['Yp'] - obs_Yp[0]) / obs_Yp[1]) ** 2
              - 0.5 * ((r['DH'] - obs_DH[0]) / obs_DH[1]) ** 2)
        return float(ll)

    return loglike


def make_log_likelihood_sigma(
    N_q: int = 20,
    correction_level: int = 0,
    backend: str = "scipy",
    obs_Yp: tuple = (YP_OBS, YP_ERR),
    obs_DH: tuple = (DH_OBS, DH_ERR),
):
    """Create a log-likelihood for Σ_H (1D scan/NSS).

    Returns loglike(params) where params = [Σ_H].
    Uses fixed standard (η, τ_n).
    """
    if backend != "scipy":
        raise ValueError(
            "Only backend='scipy' remains available for host BBN likelihoods; "
            "the high-level Type-I JAX runtime was removed."
        )
    solve_fn = _scipy_forward_solve

    def loglike(params):
        sigma_h = float(jnp.clip(params[0], 0.0, 0.999))
        r = solve_fn(sigma_h, ETA_OBS * 1e-10, TAUN_OBS, N_q, correction_level)
        if not validate_prediction_for_inference(r):
            return -np.inf
        ll = (-0.5 * ((r['Yp'] - obs_Yp[0]) / obs_Yp[1]) ** 2
              - 0.5 * ((r['DH'] - obs_DH[0]) / obs_DH[1]) ** 2)
        return float(ll)

    return loglike


# ═══════════════════════════════════════════════════════════════
# §4. BBN NUTS convenience wrapper (unavailable until B-05)
# ═══════════════════════════════════════════════════════════════

@dataclass
class BBNNUTSConfig:
    """Configuration for BBN NUTS inference."""
    Sigma_H: float = 0.0
    correction_level: int = 0
    N_q: int = 20
    num_warmup: int = 200
    num_samples: int = 500
    num_chains: int = 1
    target_acceptance_rate: float = 0.80
    max_tree_depth: int = 8


def run_bbn_nuts_production(
    config: BBNNUTSConfig = None,
    rng_key: Optional[jnp.ndarray] = None,
) -> dict:
    """Fail closed until the host BBN forward is JAX-traceable in B-05."""
    raise RuntimeError(BBN_JAX_SAMPLER_UNAVAILABLE)


# ═══════════════════════════════════════════════════════════════
# §5. BBN NSS convenience wrapper (unavailable until B-05)
# ═══════════════════════════════════════════════════════════════

@dataclass
class BBNNSSConfig:
    """Configuration for BBN nested sampling."""
    correction_level: int = 0
    N_q: int = 20
    backend: str = "scipy"  # characteristic host reference
    n_live: int = 100
    max_iterations: int = 2000
    dlogz_threshold: float = 0.01


def run_bbn_nss_production(
    config: BBNNSSConfig = None,
    rng_key: Optional[jnp.ndarray] = None,
) -> dict:
    """Fail closed until the host BBN forward is JAX-traceable in B-05."""
    raise RuntimeError(BBN_JAX_SAMPLER_UNAVAILABLE)
