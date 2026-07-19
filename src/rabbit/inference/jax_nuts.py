"""
rabbit.inference.jax_nuts — Experimental BlackJAX NUTS wrapper.

This module is exploratory only. It wraps BlackJAX's NUTS for package-internal
experiments, but native-gradient publication inference is NOT validated in the
current freeze posture.  The generic engine accepts a caller-supplied traceable
log density; BBN-specific host-solver wrappers are fail-closed until B-05.

Usage:
    from rabbit.inference.jax_nuts import run_nuts, NUTSConfig

    result = run_nuts(
        log_posterior_fn=log_post,   # from gradient_bridge.make_log_posterior
        initial_position=jnp.array([6.1e-10, 878.4]),
        config=NUTSConfig(num_warmup=500, num_samples=2000),
        rng_key=jax.random.PRNGKey(0),
    )
    # result.samples: (num_samples, n_dim)
    # result.log_posterior: (num_samples,)
    # result.diagnostics: acceptance, divergences, step_size, etc.

Architecture:
    1. Window adaptation (warmup): tune step_size + mass_matrix
    2. NUTS sampling with tuned parameters
    3. Multi-chain support via jax.vmap over PRNGKey
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable, Optional
import time

import jax
import jax.numpy as jnp
import numpy as np

from rabbit.inference.observables import BBN_JAX_SAMPLER_UNAVAILABLE

jax.config.update("jax_enable_x64", True)


@dataclass
class NUTSConfig:
    """Configuration for NUTS sampling."""
    num_warmup: int = 500
    num_samples: int = 2000
    num_chains: int = 1
    target_acceptance_rate: float = 0.80
    max_tree_depth: int = 10
    initial_step_size: float = 0.1
    is_mass_matrix_diagonal: bool = True
    progress_bar: bool = False


@dataclass
class NUTSResult:
    """NUTS sampling result."""
    samples: np.ndarray            # (num_samples, n_dim) or (num_chains, num_samples, n_dim)
    log_posterior: np.ndarray      # (num_samples,) or (num_chains, num_samples)
    acceptance_rate: float
    num_divergent: int
    step_size: float
    wall_time_s: float
    num_warmup: int
    num_samples: int
    num_chains: int
    metadata: dict = field(default_factory=dict)


def run_nuts(
    log_posterior_fn: Callable,
    initial_position: jnp.ndarray,
    config: NUTSConfig = None,
    rng_key: Optional[jnp.ndarray] = None,
) -> NUTSResult:
    """Run NUTS with BlackJAX window adaptation.

    Parameters
    ----------
    log_posterior_fn : callable(params) → float
        Must be JAX-differentiable (via gradient_bridge or native).
    initial_position : jnp.ndarray, shape (n_dim,)
        Starting point in parameter space.
    config : NUTSConfig
    rng_key : JAX PRNGKey (default: PRNGKey(42))
    """
    import blackjax

    if config is None:
        config = NUTSConfig()
    if rng_key is None:
        rng_key = jax.random.PRNGKey(42)

    initial_position = jnp.asarray(initial_position, dtype=jnp.float64)
    n_dim = initial_position.shape[0]

    t0 = time.perf_counter()

    # ── Single-chain NUTS ──
    def _run_single_chain(key, init_pos):
        warmup_key, sample_key = jax.random.split(key)

        # Window adaptation (warmup)
        warmup = blackjax.window_adaptation(
            blackjax.nuts,
            log_posterior_fn,
            is_mass_matrix_diagonal=config.is_mass_matrix_diagonal,
            initial_step_size=config.initial_step_size,
            target_acceptance_rate=config.target_acceptance_rate,
            progress_bar=config.progress_bar,
        )

        (state, parameters), adapt_info = warmup.run(
            warmup_key, init_pos, num_steps=config.num_warmup)

        # Build tuned kernel
        kernel = blackjax.nuts(
            log_posterior_fn,
            step_size=parameters["step_size"],
            inverse_mass_matrix=parameters["inverse_mass_matrix"],
            max_num_doublings=config.max_tree_depth,
        ).step

        # Sampling loop
        def step_fn(carry, key):
            state = carry
            state, info = kernel(key, state)
            return state, (state, info)

        sample_keys = jax.random.split(sample_key, config.num_samples)
        _, (states, infos) = jax.lax.scan(step_fn, state, sample_keys)

        return states, infos, parameters

    if config.num_chains == 1:
        states, infos, parameters = _run_single_chain(rng_key, initial_position)
        samples = np.asarray(states.position)
        log_post = np.asarray(states.logdensity)

        # Diagnostics
        is_divergent = np.asarray(infos.is_divergent)
        num_divergent = int(np.sum(is_divergent))
        acceptance = float(np.mean(np.asarray(infos.acceptance_rate)))
        step_size = float(parameters["step_size"])

    else:
        # Multi-chain: vmap over keys
        chain_keys = jax.random.split(rng_key, config.num_chains)
        init_positions = jnp.broadcast_to(
            initial_position, (config.num_chains, n_dim))

        # Add small perturbation to each chain
        perturb_keys = jax.random.split(chain_keys[0], config.num_chains)
        perturbations = jax.vmap(
            lambda k: jax.random.normal(k, shape=(n_dim,)) * 0.01
        )(perturb_keys)
        init_positions = init_positions + perturbations

        all_states, all_infos, all_params = jax.vmap(
            _run_single_chain)(chain_keys, init_positions)

        samples = np.asarray(all_states.position)
        log_post = np.asarray(all_states.logdensity)

        is_divergent = np.asarray(all_infos.is_divergent)
        num_divergent = int(np.sum(is_divergent))
        acceptance = float(np.mean(np.asarray(all_infos.acceptance_rate)))
        step_size = float(np.mean(np.asarray(
            jax.tree.leaves(all_params)[0])))

    wall_time = time.perf_counter() - t0

    return NUTSResult(
        samples=samples,
        log_posterior=log_post,
        acceptance_rate=acceptance,
        num_divergent=num_divergent,
        step_size=step_size,
        wall_time_s=wall_time,
        num_warmup=config.num_warmup,
        num_samples=config.num_samples,
        num_chains=config.num_chains,
        metadata={
            'sampler': 'blackjax_nuts',
            'n_dim': n_dim,
            'target_acceptance': config.target_acceptance_rate,
            'max_tree_depth': config.max_tree_depth,
            'divergent_fraction': num_divergent / max(config.num_samples * config.num_chains, 1),
        },
    )


# ═══════════════════════════════════════════════════════════════
# §2. BBN-specific NUTS wrapper
# ═══════════════════════════════════════════════════════════════

def run_bbn_nuts(
    Sigma_H: float = 0.1,
    backend: str = "jax",
    N_q: int = 6,
    correction_level: int = 0,
    num_warmup: int = 200,
    num_samples: int = 500,
    rng_key: Optional[jnp.ndarray] = None,
) -> NUTSResult:
    """Fail closed until the host BBN forward is JAX-traceable in B-05."""
    raise RuntimeError(BBN_JAX_SAMPLER_UNAVAILABLE)
