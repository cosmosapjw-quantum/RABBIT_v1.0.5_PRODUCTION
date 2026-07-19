"""
rabbit.inference.jax_nested — Experimental nested-sampling prototype in JAX.

GPU-friendly nested sampling with:
  - Clustered live points (K-means in whitened space)
  - Whitened slice sampling within clusters
  - Batch dead-point replacement
  - Log-evidence accumulation (Skilling quadrature)

Algorithm (per outer iteration):
  1. Select k lowest-likelihood live points → dead
  2. Refresh clusters (periodically)
  3. For each dead slot (parallel):
     a. Sample cluster proportional to volume
     b. Pick seed from cluster
     c. Whitened slice chain (n_repeats = α×d) above L*
     d. Replace dead point
  4. Update logX, logZ, record dead points

References:
  Handley, Hobson, Lasenby (2015) — PolyChord
  Skilling (2006) — Nested sampling for Bayesian computation
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable, Optional, Tuple
from functools import partial
import time

import jax
import jax.numpy as jnp
import jax.random as jr
import numpy as np

from rabbit.inference.observables import BBN_JAX_SAMPLER_UNAVAILABLE

jax.config.update("jax_enable_x64", True)


# ═══════════════════════════════════════════════════════════════
# §1. Configuration
# ═══════════════════════════════════════════════════════════════

@dataclass
class NSSConfig:
    """Nested Slice Sampling configuration."""
    n_live: int = 250              # number of live points
    n_delete: int = 1              # dead points per iteration (batch size)
    n_repeats_factor: float = 5.0  # n_repeats = factor × n_dim
    max_iterations: int = 5000     # outer loop limit
    dlogz_threshold: float = 0.01  # convergence criterion
    n_clusters_max: int = 8        # max clusters for K-means
    cluster_refresh_interval: int = 50  # re-cluster every N iterations
    slice_width: float = 1.0       # initial slice width in whitened space
    slice_max_steps: int = 100     # max slice expansions


@dataclass
class NSSResult:
    """Nested sampling result."""
    log_evidence: float            # ln(Z)
    log_evidence_error: float      # σ(ln Z)
    samples: np.ndarray            # weighted posterior samples (n_dead, n_dim)
    log_weights: np.ndarray        # log-importance weights
    log_likelihoods: np.ndarray    # log L for each dead point
    n_dead: int                    # number of dead points
    n_like_calls: int              # total likelihood evaluations
    n_iterations: int              # outer loop iterations
    wall_time_s: float
    H: float                       # information (nats)
    metadata: dict = field(default_factory=dict)


# ═══════════════════════════════════════════════════════════════
# §2. Clustering
# ═══════════════════════════════════════════════════════════════

def _kmeans_jax(points: jnp.ndarray, k: int, key: jnp.ndarray,
                max_iter: int = 30) -> Tuple[jnp.ndarray, jnp.ndarray]:
    """Simple K-means in JAX (GPU-friendly).

    Returns (labels, centroids).
    """
    n, d = points.shape
    # Initialize centroids by random selection
    idx = jr.choice(key, n, shape=(k,), replace=False)
    centroids = points[idx]

    def step(carry, _):
        centroids = carry
        # Assign
        dists = jnp.sum((points[:, None, :] - centroids[None, :, :]) ** 2, axis=-1)
        labels = jnp.argmin(dists, axis=1)
        # Update centroids
        new_centroids = jnp.zeros_like(centroids)
        counts = jnp.zeros(k)
        for c in range(k):
            mask = (labels == c).astype(jnp.float64)
            count = jnp.sum(mask)
            counts = counts.at[c].set(count)
            centroid = jnp.where(count > 0,
                                 jnp.sum(points * mask[:, None], axis=0) / jnp.maximum(count, 1),
                                 centroids[c])
            new_centroids = new_centroids.at[c].set(centroid)
        return new_centroids, labels

    centroids, labels = jax.lax.scan(step, centroids, None, length=max_iter)
    labels = labels[-1]  # last iteration's labels
    return labels, centroids


def compute_cluster_stats(points: jnp.ndarray, labels: jnp.ndarray,
                          k: int) -> dict:
    """Compute per-cluster mean, Cholesky, and volume.

    Returns dict with keys: mu, chol, log_volume, counts
    Each is (k, ...) shaped.
    """
    n, d = points.shape
    mus = jnp.zeros((k, d))
    chols = jnp.zeros((k, d, d))
    log_volumes = jnp.zeros(k)
    counts = jnp.zeros(k)

    for c in range(k):
        mask = (labels == c).astype(jnp.float64)
        count = jnp.sum(mask)
        counts = counts.at[c].set(count)

        if_enough = count > d + 1
        mu = jnp.where(if_enough,
                        jnp.sum(points * mask[:, None], axis=0) / jnp.maximum(count, 1),
                        jnp.zeros(d))
        mus = mus.at[c].set(mu)

        # Covariance (regularized)
        diff = (points - mu) * mask[:, None]
        cov = jnp.dot(diff.T, diff) / jnp.maximum(count - 1, 1)
        cov = cov + 1e-8 * jnp.eye(d)  # regularization

        # Cholesky (for whitening)
        chol = jnp.where(if_enough, jnp.linalg.cholesky(cov), jnp.eye(d))
        chols = chols.at[c].set(chol)

        # Log-volume ∝ log det(L) + d/2 log(2π) + expansion factor
        log_det = jnp.where(if_enough, jnp.sum(jnp.log(jnp.diag(chol))), 0.0)
        # Enlarge by factor (1 + 1/count) for exploration margin
        enlargement = 0.5 * d * jnp.log(1.0 + 1.0 / jnp.maximum(count, 1))
        log_volumes = log_volumes.at[c].set(log_det + enlargement)

    return {
        'mu': mus,           # (k, d)
        'chol': chols,       # (k, d, d)
        'log_volume': log_volumes,  # (k,)
        'counts': counts,    # (k,)
    }


# ═══════════════════════════════════════════════════════════════
# §3. Whitened slice sampling
# ═══════════════════════════════════════════════════════════════

def whitened_slice_chain(
    key: jnp.ndarray,
    x0: jnp.ndarray,
    mu: jnp.ndarray,
    chol: jnp.ndarray,
    logprior_fn: Callable,
    loglike_fn: Callable,
    L_star: float,
    n_repeats: int,
    width: float = 1.0,
    max_steps: int = 100,
) -> Tuple[jnp.ndarray, int]:
    """Whitened slice sampling chain (PolyChord-style).

    Samples a new point above L_star via repeated axis-aligned
    slices in the whitened coordinate system defined by (mu, chol).

    Returns (x_new, n_like_calls).
    """
    d = x0.shape[0]
    chol_inv = jnp.linalg.inv(chol)

    # Whiten x0
    w0 = jnp.dot(chol_inv, x0 - mu)

    def single_slice(carry, key_i):
        w, n_calls = carry
        key_dir, key_pos, key_shrink = jr.split(key_i, 3)

        # Random axis in whitened space
        axis_idx = jr.randint(key_dir, (), 0, d)
        direction = jnp.zeros(d).at[axis_idx].set(1.0)

        # Slice bounds
        u = jr.uniform(key_pos)
        lo = -width * u
        hi = lo + width

        # Shrink until we find a valid point
        def shrink_step(carry, _):
            lo, hi, w, n_calls, found = carry
            key_s = jr.fold_in(key_shrink, n_calls)
            t = lo + jr.uniform(key_s) * (hi - lo)
            w_prop = w + t * direction - w.at[axis_idx].get() * direction
            w_prop = w_prop.at[axis_idx].set(w[axis_idx] + t)

            # Unwhiten
            x_prop = jnp.dot(chol, w_prop) + mu

            logL_prop = loglike_fn(x_prop)
            logpi_prop = logprior_fn(x_prop)
            n_calls = n_calls + 1

            valid = (logL_prop >= L_star) & jnp.isfinite(logpi_prop) & (~found)

            # Accept or shrink
            w_new = jnp.where(valid, w_prop, w)
            lo_new = jnp.where(valid | (t >= 0), lo, t)
            hi_new = jnp.where(valid | (t < 0), hi, t)
            found_new = found | valid

            return (lo_new, hi_new, w_new, n_calls, found_new), None

        init = (lo, hi, w, n_calls, False)
        (_, _, w_new, n_calls_new, found), _ = jax.lax.scan(
            shrink_step, init, None, length=max_steps)

        # If not found, keep current
        w_final = jnp.where(found, w_new, w)
        return (w_final, n_calls_new), None

    keys = jr.split(key, n_repeats)
    (w_final, total_calls), _ = jax.lax.scan(single_slice, (w0, 0), keys)

    # Unwhiten
    x_new = jnp.dot(chol, w_final) + mu
    return x_new, total_calls


# ═══════════════════════════════════════════════════════════════
# §4. Log-evidence accumulation
# ═══════════════════════════════════════════════════════════════

def _logaddexp(a, b):
    return jnp.logaddexp(a, b)


def _log_evidence_step(logX_prev, logL_dead, n_live):
    """Single dead-point evidence contribution (Skilling)."""
    # X shrinks by factor n_live/(n_live+1) per dead point
    log_shrink = jnp.log(n_live) - jnp.log(n_live + 1.0)
    logX_new = logX_prev + log_shrink
    # Weight = X_prev - X_new ≈ X_prev / (n_live+1)
    log_weight = logX_prev - jnp.log(n_live + 1.0)
    # Contribution to evidence
    log_dZ = log_weight + logL_dead
    return logX_new, log_weight, log_dZ


# ═══════════════════════════════════════════════════════════════
# §5. Main nested sampling loop
# ═══════════════════════════════════════════════════════════════

def run_nss(
    loglike_fn: Callable,
    logprior_fn: Callable,
    prior_sample_fn: Callable,
    n_dim: int,
    config: NSSConfig = None,
    rng_key: Optional[jnp.ndarray] = None,
) -> NSSResult:
    """Run the experimental nested-sampling prototype.

    Parameters
    ----------
    loglike_fn : callable(x) → float
        Log-likelihood function.
    logprior_fn : callable(x) → float
        Log-prior function (for boundary enforcement).
    prior_sample_fn : callable(key) → x
        Draw a sample from the prior (for initialization).
    n_dim : int
        Number of dimensions.
    config : NSSConfig
    rng_key : JAX PRNGKey
    """
    if config is None:
        config = NSSConfig()
    if rng_key is None:
        rng_key = jax.random.PRNGKey(42)

    n_live = config.n_live
    n_delete = config.n_delete
    n_repeats = int(config.n_repeats_factor * n_dim)
    max_iter = config.max_iterations

    t0 = time.perf_counter()
    n_like_calls = 0

    # ── Initialize live points from prior ──
    init_keys = jr.split(rng_key, n_live + 1)
    rng_key = init_keys[0]

    live_x = np.zeros((n_live, n_dim))
    live_logL = np.full(n_live, -np.inf)
    live_logpi = np.zeros(n_live)

    for i in range(n_live):
        x = np.asarray(prior_sample_fn(init_keys[i + 1]))
        live_x[i] = x
        live_logL[i] = float(loglike_fn(jnp.asarray(x)))
        live_logpi[i] = float(logprior_fn(jnp.asarray(x)))
        n_like_calls += 1

    # ── Storage for dead points ──
    max_dead = max_iter * n_delete + n_live
    dead_x = np.zeros((max_dead, n_dim))
    dead_logL = np.zeros(max_dead)
    dead_log_weights = np.full(max_dead, -np.inf)
    n_dead = 0

    # ── Evidence accumulators ──
    logX = 0.0          # ln(X), starts at ln(1) = 0
    logZ = -1e30        # ln(Z), starts at -inf
    H = 0.0             # information

    # ── Cluster state ──
    k_clusters = min(config.n_clusters_max, max(1, n_live // 20))
    cluster_labels = np.zeros(n_live, dtype=int)
    cluster_stats = None
    last_cluster_refresh = -config.cluster_refresh_interval  # force initial clustering

    # ── Main loop ──
    for iteration in range(max_iter):

        # 1. Select k lowest-likelihood live points → dead
        dead_idx = np.argpartition(live_logL, n_delete)[:n_delete]
        L_star = float(np.max(live_logL[dead_idx]))

        # Record dead points
        for j in dead_idx:
            # Evidence contribution
            logX_new, log_weight, log_dZ = _log_evidence_step(
                jnp.array(logX), jnp.array(live_logL[j]), jnp.array(float(n_live)))
            logX = float(logX_new)
            log_weight = float(log_weight)

            dead_x[n_dead] = live_x[j]
            dead_logL[n_dead] = live_logL[j]
            dead_log_weights[n_dead] = log_weight
            logZ = float(_logaddexp(jnp.array(logZ), jnp.array(float(log_dZ))))
            n_dead += 1

        # 2. Refresh clusters periodically
        if iteration - last_cluster_refresh >= config.cluster_refresh_interval:
            try:
                key_cluster, rng_key = jr.split(rng_key)
                jax_live = jnp.asarray(live_x)
                labels, centroids = _kmeans_jax(jax_live, k_clusters, key_cluster)
                cluster_labels = np.asarray(labels)
                cluster_stats = compute_cluster_stats(jax_live, labels, k_clusters)
                last_cluster_refresh = iteration
            except Exception:
                # Fallback: single cluster
                k_clusters = 1
                cluster_labels[:] = 0
                mu = jnp.mean(jnp.asarray(live_x), axis=0)
                cov = jnp.cov(jnp.asarray(live_x).T) + 1e-8 * jnp.eye(n_dim)
                chol = jnp.linalg.cholesky(cov)
                cluster_stats = {
                    'mu': mu[None, :],
                    'chol': chol[None, :, :],
                    'log_volume': jnp.array([jnp.sum(jnp.log(jnp.diag(chol)))]),
                    'counts': jnp.array([float(n_live)]),
                }

        # 3. Replace dead points
        for j_idx, j in enumerate(dead_idx):
            key_sample, key_chain, rng_key = jr.split(rng_key, 3)

            # Sample cluster proportional to volume
            log_vols = np.asarray(cluster_stats['log_volume'])
            log_vols_shifted = log_vols - np.max(log_vols)
            probs = np.exp(log_vols_shifted)
            probs = probs / np.sum(probs)
            c = int(jr.choice(key_sample, k_clusters, p=jnp.asarray(probs)))

            # Pick seed from cluster
            in_cluster = np.where(cluster_labels == c)[0]
            if len(in_cluster) == 0:
                in_cluster = np.arange(n_live)
            seed_idx = int(jr.choice(key_sample, len(in_cluster)))
            x0 = jnp.asarray(live_x[in_cluster[seed_idx]])

            # Whitened slice chain
            mu_c = cluster_stats['mu'][c]
            chol_c = cluster_stats['chol'][c]

            x_new, calls = whitened_slice_chain(
                key_chain, x0, mu_c, chol_c,
                logprior_fn, loglike_fn,
                L_star=L_star,
                n_repeats=n_repeats,
                width=config.slice_width,
                max_steps=config.slice_max_steps,
            )
            n_like_calls += int(calls)

            # Replace
            live_x[j] = np.asarray(x_new)
            live_logL[j] = float(loglike_fn(x_new))
            live_logpi[j] = float(logprior_fn(x_new))
            n_like_calls += 1

        # 4. Convergence check
        # Remaining evidence in live points
        logL_max = float(np.max(live_logL))
        logZ_remaining = logX + logL_max
        # Stop if remaining evidence is small relative to accumulated
        if logZ_remaining < logZ + np.log(config.dlogz_threshold):
            break

    # ── Add remaining live points to evidence ──
    sorted_live = np.argsort(live_logL)
    for i, idx in enumerate(sorted_live):
        logX_new, log_weight, log_dZ = _log_evidence_step(
            jnp.array(logX), jnp.array(live_logL[idx]), jnp.array(float(n_live - i)))
        logX = float(logX_new)
        dead_x[n_dead] = live_x[idx]
        dead_logL[n_dead] = live_logL[idx]
        dead_log_weights[n_dead] = float(log_weight)
        logZ = float(_logaddexp(jnp.array(logZ), jnp.array(float(log_dZ))))
        n_dead += 1

    # ── Compute posterior samples (importance-weighted) ──
    dead_x = dead_x[:n_dead]
    dead_logL = dead_logL[:n_dead]
    dead_log_weights = dead_log_weights[:n_dead]

    # Normalize log-weights
    log_wt = dead_log_weights + dead_logL - logZ
    log_wt -= np.max(log_wt)  # numerical stability

    # Information
    wt = np.exp(log_wt)
    wt /= np.sum(wt)
    H = float(np.sum(wt * (dead_logL - logZ)))

    # Evidence error (Skilling approximation)
    logZ_error = np.sqrt(abs(H) / n_live) if n_live > 0 else np.inf

    wall_time = time.perf_counter() - t0

    return NSSResult(
        log_evidence=logZ,
        log_evidence_error=logZ_error,
        samples=dead_x,
        log_weights=dead_log_weights + dead_logL,
        log_likelihoods=dead_logL,
        n_dead=n_dead,
        n_like_calls=n_like_calls,
        n_iterations=iteration + 1,
        wall_time_s=wall_time,
        H=H,
        metadata={
            'sampler': 'jax_nss_polychord_like',
            'n_live': n_live,
            'n_delete': n_delete,
            'n_dim': n_dim,
            'n_clusters': k_clusters,
            'n_repeats': n_repeats,
            'dlogz_threshold': config.dlogz_threshold,
            'convergence_iteration': iteration + 1,
        },
    )


# ═══════════════════════════════════════════════════════════════
# §6. BBN-specific NSS wrapper
# ═══════════════════════════════════════════════════════════════

def run_bbn_nss(
    backend: str = "jax",
    N_q: int = 6,
    correction_level: int = 0,
    n_live: int = 100,
    max_iterations: int = 2000,
    rng_key: Optional[jnp.ndarray] = None,
) -> NSSResult:
    """Fail closed until the host BBN forward is JAX-traceable in B-05."""
    raise RuntimeError(BBN_JAX_SAMPLER_UNAVAILABLE)
