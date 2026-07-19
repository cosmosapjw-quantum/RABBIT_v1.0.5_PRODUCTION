"""rabbit.inference.joint_3d_inference — joint 3D Bayesian inference over (Σ_H, η, τ_n).

Provides a host-only diagnostic log-posterior over ``(Σ_H, η, τ_n)``.
The BBN NUTS convenience runner is fail-closed because the host forward solver
is not JAX-traceable.  No gradient-based publication inference is available.

``ad_backend='rosenbrock_native'`` is retained as an explicit research
entry point that raises a documented ``NotImplementedError``.  The older
``diffrax_native`` joint-3D option is deliberately not accepted here because
no joint-3D dispatch branch exists; Diffrax remains only a gradient-bridge
cross-validation reference.

Historical diagnostic priors (not a validated shear domain)
------------------------------------------------
  Σ_H  ~ Half-Normal(0.3)         (legacy diagnostic scale)
  η_10 ~ Gaussian(6.104, 0.058)   (Planck 2018 + PDG)
  τ_n  ~ Gaussian(878.4, 0.5)     (PDG 2024)

Half-Normal is implemented as a Gaussian centered at 0 with σ=0.3 plus a hard
``-inf`` host-log-density boundary for Σ_H < 0.  The BBN NUTS wrapper is not
available, so this prior does not establish a sampled constraint.

Reference
---------
- BlackJAX NUTS implementation
- Plan §4.3 priors and gates
- Audit P1 gap (gradient_based_inference_ready)
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Optional, Sequence

import jax
import jax.numpy as jnp
import numpy as np

from rabbit.inference.observables import (
    BBN_JAX_SAMPLER_UNAVAILABLE,
    validate_prediction_for_inference,
)


jax.config.update("jax_enable_x64", True)


# ═══════════════════════════════════════════════════════════════════════
# §1. Default observations and priors
# ═══════════════════════════════════════════════════════════════════════

# Observational constraints (Aver+ 2021 + Cooke+ 2018 + Planck 2018 + PDG 2024).
DEFAULT_YP_OBS = 0.2449
DEFAULT_YP_ERR = 0.0040
DEFAULT_DH_OBS = 2.547e-5
DEFAULT_DH_ERR = 0.029e-5
DEFAULT_ETA10_OBS = 6.104
DEFAULT_ETA10_ERR = 0.058
DEFAULT_TAUN_OBS = 878.4
DEFAULT_TAUN_ERR = 0.5

# Half-Normal prior width on Σ_H (arXiv:2502.20893 motivates σ/H_BBN < 0.3)
DEFAULT_SIGMA_H_PRIOR = 0.3


# ═══════════════════════════════════════════════════════════════════════
# §2. log-posterior factory
# ═══════════════════════════════════════════════════════════════════════

@dataclass
class JointInference3DConfig:
    """Configuration for the 3D BBN log-posterior."""

    # Observations
    yp_obs: float = DEFAULT_YP_OBS
    yp_err: float = DEFAULT_YP_ERR
    dh_obs: float = DEFAULT_DH_OBS
    dh_err: float = DEFAULT_DH_ERR

    # Priors (Gaussian centered)
    eta10_obs: float = DEFAULT_ETA10_OBS
    eta10_err: float = DEFAULT_ETA10_ERR
    taun_obs: float = DEFAULT_TAUN_OBS
    taun_err: float = DEFAULT_TAUN_ERR
    sigma_h_prior_width: float = DEFAULT_SIGMA_H_PRIOR

    # Forward solver
    backend: str = "scipy"
    correction_level: int = 2
    N_q: int = 20

    # AD backend
    ad_backend: str = "fd_legacy"  # 'fd_legacy' | 'rosenbrock_native' (research entry)
    fd_eps: float = 1.0e-5


# ═══════════════════════════════════════════════════════════════════════
# §2b. Σ_H identifiability / shrinkage diagnostic (BD598 PR-C, D-R1/D-R2)
# ═══════════════════════════════════════════════════════════════════════

#: A Σ_H constraint counts as data-driven only if the marginal posterior is
#: meaningfully tighter than the prior (posterior_sigma < this × prior_sigma).
SHRINKAGE_INFORMATIVE = 0.9


def _sigma_h_informative(rank_likelihood: int, shrinkage: float) -> bool:
    """Σ_H is data-informative only if the likelihood identifies all three
    directions (rank 3) AND the marginal shrinks below the threshold.

    Rank is the robust gate: at/near the FLRW null the likelihood is rank-2
    (Σ_H enters as Σ_H²) and the FD marginal is ε-noise (D-R5), so the shrinkage
    alone must never promote a degenerate fit to "informative".

    STRUCTURAL CEILING (BD599 INF-1): with the DEFAULT 2-observable likelihood
    (Yp, D/H) for 3 parameters the Fisher rank is <= 2 at EVERY fiducial — not
    only the null — so this returns False by default. The positive (rank>=3)
    branch becomes reachable off-null with the OPT-IN Li7/H 3rd observable
    (sigma_h_identifiability(..., include_li7h=True), S1), but that path is
    EXPLORATORY (the lithium problem forbids a face-value Li7 constraint). With
    the default 2-observable likelihood the honest reportable verdict remains
    rank/degeneracy, NOT an informative Σ_H constraint.
    """
    return bool(rank_likelihood >= 3 and shrinkage < SHRINKAGE_INFORMATIVE)


@dataclass(frozen=True)
class SigmaHIdentifiability:
    """Honest read-out of whether the BBN data constrains Σ_H at a fiducial.

    Because Y_p enters the Friedmann equation as Σ_H², the likelihood-only
    Fisher is rank-deficient at the FLRW null (Σ_H direction degenerate): any
    Σ_H "constraint" there is the prior, not data. ``informative`` is the
    trustworthy flag — it is True only when the data identifies all three
    directions (``degenerate`` False) AND shrinks the Σ_H marginal below
    ``SHRINKAGE_INFORMATIVE`` × prior.

    ``posterior_sigma`` / ``shrinkage`` are reported for transparency but are
    NOT trustworthy when ``degenerate`` is True: central FD of the even Σ_H²
    response at the null is ε-noise (BD598 D-R5), so the marginal there is a
    numerical artifact, not a physical error. Use ``informative`` / rank, not
    the raw shrinkage, for any claim at or near the null.
    """
    fiducial_sigma_H: float
    rank_likelihood: int
    degenerate: bool
    cond_likelihood: float
    cond_posterior: float
    prior_sigma: float
    posterior_sigma: float
    shrinkage: float
    informative: bool


def sigma_h_identifiability(
    config: Optional[JointInference3DConfig] = None,
    *,
    sigma_H_fid: float = 0.0,
    include_li7h: bool = False,
) -> SigmaHIdentifiability:
    """Likelihood-rank + Σ_H prior→posterior shrinkage at a fiducial point.

    Reuses ``rabbit.inference.fisher`` (no duplicate Fisher code). The prior
    Fisher is built in the (Σ_H, log η, log τ_n) parametrisation used by
    ``fisher.bbn_fisher``; the η/τ_n Gaussian widths are mapped to log-space via
    σ_log ≈ σ/μ. The Σ_H half-normal scale is compared directly to the posterior
    marginal (matching the recovery-test convention of σ_marg vs prior 0.30).
    """
    from rabbit.inference import fisher

    if config is None:
        config = JointInference3DConfig()

    F_lik, diag_lik = fisher.bbn_fisher(
        sigma_H_fid=float(sigma_H_fid),
        eta_fid=config.eta10_obs * 1e-10,
        tau_n_fid=config.taun_obs,
        yp_sigma=config.yp_err,
        dh_sigma=config.dh_err,
        backend=config.backend,
        N_q=config.N_q,
        correction_level=config.correction_level,
        include_li7h=include_li7h,  # S1: opt-in 3rd observable breaks the rank ceiling off-null
    )

    # Prior Fisher in the SAME basis as bbn_fisher: theta = [Σ_H, log η, log τ_n].
    # Σ_H is linear (half-normal scale used directly); the η/τ_n Gaussian priors
    # are mapped to log-space first order (σ_log ≈ σ/μ). The Σ_H identifiability
    # verdict is rank-based (likelihood-only, prior-independent), so this mapping
    # affects only the η/τ_n posterior marginals, not the Σ_H conclusion.
    prior_sigma = float(config.sigma_h_prior_width)
    s_log_eta = config.eta10_err / config.eta10_obs
    s_log_tau = config.taun_err / config.taun_obs
    prior_F = fisher.gaussian_prior_fisher([prior_sigma, s_log_eta, s_log_tau])
    diag_post = fisher.fisher_diagnostics(F_lik, prior_F=prior_F)

    i_sh = 0  # Σ_H is the first parameter (fisher.PARAM_NAMES)
    posterior_sigma = float(diag_post.sigma_marginalized[i_sh])
    shrinkage = posterior_sigma / prior_sigma if prior_sigma > 0 else float("inf")
    # Rank is the robust identifiability signal; the FD shrinkage is unreliable
    # when degenerate (even-response ε-noise at the null, D-R5).
    degenerate = bool(diag_lik.rank < 3)
    informative = _sigma_h_informative(int(diag_lik.rank), shrinkage)

    return SigmaHIdentifiability(
        fiducial_sigma_H=float(sigma_H_fid),
        rank_likelihood=int(diag_lik.rank),
        degenerate=degenerate,
        cond_likelihood=float(diag_lik.condition_number),
        cond_posterior=float(diag_post.condition_number),
        prior_sigma=prior_sigma,
        posterior_sigma=posterior_sigma,
        shrinkage=float(shrinkage),
        informative=informative,
    )


def make_log_posterior_3d(config: Optional[JointInference3DConfig] = None) -> Callable:
    """Build the 3D log-posterior closure ``log_post(params) -> scalar``.

    Parameters
    ----------
    config : JointInference3DConfig
        Observations, priors, forward-solver settings, AD backend.

    Returns
    -------
    log_post_fn : callable(jnp.ndarray of shape (3,)) -> jnp.ndarray scalar
        ``params = [Sigma_H, eta_10, tau_n]``. Host evaluation is diagnostic;
        gradient tracing fails closed until B-05.
    """
    if config is None:
        config = JointInference3DConfig()

    if config.ad_backend == "diffrax_native":
        raise ValueError(
            "ad_backend='diffrax_native' is not wired for joint_3d_inference; "
            "Diffrax remains a gradient_bridge cross-validation reference. "
            "Use ad_backend='fd_legacy' or the documented "
            "ad_backend='rosenbrock_native' research entry point."
        )
    if config.ad_backend not in ("fd_legacy", "rosenbrock_native"):
        raise ValueError(
            f"ad_backend={config.ad_backend!r} not in "
            "{'fd_legacy', 'rosenbrock_native'}"
        )

    yp_obs, yp_err = config.yp_obs, config.yp_err
    dh_obs, dh_err = config.dh_obs, config.dh_err
    eta10_obs, eta10_err = config.eta10_obs, config.eta10_err
    taun_obs, taun_err = config.taun_obs, config.taun_err
    sigma_h_w = config.sigma_h_prior_width

    def _raw_forward(params: np.ndarray) -> np.ndarray:
        """Return [Y_p, D/H] at (Sigma_H, eta_10, tau_n)."""
        sigma_H = float(np.maximum(params[0], 0.0))   # half-line constraint
        eta_10 = float(params[1])
        tau_n = float(params[2])
        if config.backend == "scipy":
            from rabbit.inference.bbn_inference import _scipy_forward_solve
            r = _scipy_forward_solve(
                Sigma_H=sigma_H, eta=eta_10 * 1e-10, tau_n=tau_n,
                N_q=config.N_q, correction_level=config.correction_level,
            )
        else:
            from rabbit.inference.bbn_inference import _jax_forward_solve
            r = _jax_forward_solve(
                Sigma_H=sigma_H, eta=eta_10 * 1e-10, tau_n=tau_n,
                N_q=config.N_q, correction_level=config.correction_level,
            )
        if not validate_prediction_for_inference(r):
            return np.array([yp_obs, dh_obs, 0.0], dtype=np.float64)
        return np.array([r["Yp"], r["DH"], 1.0], dtype=np.float64)

    if config.ad_backend == "fd_legacy":
        @jax.custom_vjp
        def forward(params: jnp.ndarray) -> jnp.ndarray:
            return jnp.asarray(_raw_forward(np.asarray(params, dtype=np.float64)))

        def forward_fwd(params):
            y = jnp.asarray(_raw_forward(np.asarray(params, dtype=np.float64)))
            return y, params

        def forward_bwd(params, g):
            """Reject the unavailable host-solver gradient bridge."""
            raise RuntimeError(BBN_JAX_SAMPLER_UNAVAILABLE)

        forward.defvjp(forward_fwd, forward_bwd)
    elif config.ad_backend == "rosenbrock_native":
        # v3.1 Phase δ: API entry point for native rosenbrock AD via
        # the gradient_bridge.make_log_posterior(bridge_mode=
        # 'rosenbrock_native') factory. Activating this path requires
        # _jax_forward_solve to be JAX-traceable end-to-end (no
        # float() extraction). The current implementation extracts
        # Python floats at the boundary, which blocks AD tracing.
        #
        # Honest scope (Plan §δ acceptance):
        #   The refactor of _jax_forward_solve to return jnp.ndarray
        #   end-to-end is research-grade infrastructure work because
        #   run_full_coupled_typeI_jax internally uses lax.while_loop
        #   with dynamic stop conditions that block reverse-mode AD.
        #   Until that refactor lands, this path raises with the
        #   documented path forward.
        #
        # Path forward (v3.1+ research-grade follow-on):
        #   1. Wrap the canonical Type-I forward via
        #      rabbit.jax.solver_jax_rodas5p_adjoint.replay_solve_with_h_tape
        #      so it accepts a flat parameter vector and runs in a
        #      lax.scan (AD-friendly).
        #   2. Make _jax_forward_solve_arrays return a jnp.ndarray
        #      [Yp, DH] without float() extraction.
        #   3. Wire make_log_posterior(bridge_mode='rosenbrock_native')
        #      over that path; flip _raw_forward dispatch to use it
        #      when ad_backend='rosenbrock_native'.
        raise NotImplementedError(
            "ad_backend='rosenbrock_native' API entry point exists but "
            "the underlying _jax_forward_solve refactor for end-to-end "
            "JAX-traceable forward is research-grade follow-on per "
            "B-05. Host scalar evaluation remains diagnostic; no gradient "
            "backend is currently available."
        )
    else:
        raise AssertionError(
            f"unreachable: ad_backend validation passed but no dispatch "
            f"branch for {config.ad_backend!r}"
        )

    def log_posterior(params: jnp.ndarray) -> jnp.ndarray:
        sigma_H = params[0]
        eta_10 = params[1]
        tau_n = params[2]

        # Hard half-line constraint on Σ_H (Half-Normal prior).
        boundary_penalty = jnp.where(sigma_H >= 0.0, 0.0, -jnp.inf)

        obs = forward(params)
        valid_obs = obs[2] > 0.5
        yp, dh = obs[0], obs[1]

        log_lik = (
            -0.5 * ((yp - yp_obs) / yp_err) ** 2
            - 0.5 * ((dh - dh_obs) / dh_err) ** 2
        )
        log_prior_sigma = -0.5 * (sigma_H / sigma_h_w) ** 2
        log_prior_eta = -0.5 * ((eta_10 - eta10_obs) / eta10_err) ** 2
        log_prior_tau = -0.5 * ((tau_n - taun_obs) / taun_err) ** 2

        log_post = (
            log_lik
            + log_prior_sigma
            + log_prior_eta
            + log_prior_tau
            + boundary_penalty
        )
        return jnp.where(valid_obs, log_post, -jnp.inf)

    return log_posterior


# ═══════════════════════════════════════════════════════════════════════
# §3. NUTS-3D entry point
# ═══════════════════════════════════════════════════════════════════════

@dataclass
class BBNNuts3DConfig:
    """Configuration for the 3D NUTS sampler."""

    inference: JointInference3DConfig = None
    num_warmup: int = 100
    num_samples: int = 200
    max_tree_depth: int = 8
    target_acceptance_rate: float = 0.80

    def __post_init__(self):
        if self.inference is None:
            self.inference = JointInference3DConfig()


def run_bbn_nuts_3d(
    config: Optional[BBNNuts3DConfig] = None,
    rng_key: Optional[jnp.ndarray] = None,
) -> dict:
    """Fail closed until the host BBN forward is JAX-traceable in B-05."""
    raise RuntimeError(BBN_JAX_SAMPLER_UNAVAILABLE)
