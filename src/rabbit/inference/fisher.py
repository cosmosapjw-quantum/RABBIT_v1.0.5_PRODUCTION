"""rabbit.inference.fisher — Fisher information matrix for BBN inference.

Physics meaning
---------------
The Fisher information matrix at a fiducial parameter point ``θ₀`` is

    F_ij(θ₀) = Σ_k (1/σ_k²) (∂μ_k/∂θ_i)(∂μ_k/∂θ_j)

where ``μ_k`` is the predicted observable (here ``Y_p``, ``D/H``) and
``σ_k`` is its observational uncertainty. ``F`` is positive semi-definite;
its eigenvalues give the inverse of squared marginalized errors along
principal axes; its condition number measures parameter degeneracy.

This module is the audit-closure response to "no live identifiability
infrastructure" (Plan §4.1, P0 gap). It exposes:

- ``fisher_info`` : generic backend; takes any predict-callable.
- ``fisher_diagnostics`` : eigenvalues, condition number, marginalized σ.
- ``bbn_fisher`` : convenience wrapper around the canonical BBN forward.

Two AD backends
---------------
``backend='fd'`` (default; works today):
    Computes ``∂μ/∂θ`` via central finite differences on the canonical
    solver. Cost ``2n`` forward solves per row of the Jacobian.

``backend='diffrax'``:
    Uses native reverse-mode AD via ``rabbit.jax.solver_diffrax_canonical``.
    Cost ``≈ 2 forward solves`` regardless of ``n``. Activates once a
    Diffrax-wrapped canonical forward is registered with this module
    (Phase γ work).

Acceptance gates
----------------
- ``cond(F_lik)`` < 1e8 at fiducial (Plan §4.1)
- ``cond(F_post)`` < 1e6 (with prior added)

Reference
---------
Standard Fisher analysis pattern from JAX-COSMO (arXiv:2302.05163),
``smsharma/clax``. ``F`` and its diagnostics also feed the
``test_fisher.py::TestFIM::test_bbn_cond_number_below_budget`` gate
referenced in ``rabbit.config.claim_gates``.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Dict, Optional, Sequence, Tuple

import numpy as np
import jax
import jax.numpy as jnp


jax.config.update("jax_enable_x64", True)


# ═══════════════════════════════════════════════════════════════════════
# §1. Result containers
# ═══════════════════════════════════════════════════════════════════════

@dataclass(frozen=True)
class FisherDiagnostics:
    """Decomposed Fisher matrix diagnostics.

    Attributes
    ----------
    F : (P, P) Fisher matrix (likelihood-only or with prior added).
    eigenvalues : (P,) sorted ascending. Tiny eigenvalues indicate
        directions in parameter space that the data does not constrain.
    eigenvectors : (P, P) columns are eigenvectors corresponding to
        ``eigenvalues``.
    condition_number : ``λ_max / λ_min``. > 1e8 means the data does not
        identify all parameters jointly.
    sigma_marginalized : (P,) ``√(F⁻¹)_ii`` — the 1-σ marginalized error
        on each parameter.
    rank : numerical rank of F (eigenvalues > rtol * λ_max).
    informative_directions : list of (eigenvalue, eigenvector) pairs in
        descending order of constraint strength, useful for reporting.
    """
    F: np.ndarray
    eigenvalues: np.ndarray
    eigenvectors: np.ndarray
    condition_number: float
    sigma_marginalized: np.ndarray
    rank: int
    informative_directions: list


# ═══════════════════════════════════════════════════════════════════════
# §2. Generic Fisher computation
# ═══════════════════════════════════════════════════════════════════════

def _central_fd_jacobian(
    predict_fn: Callable[[np.ndarray], np.ndarray],
    theta: np.ndarray,
    eps_rel: float = 1.0e-4,
    eps_abs_floor: float = 1.0e-5,
) -> np.ndarray:
    """Central finite-difference Jacobian.

    Returns ``J[k, i] = ∂μ_k/∂θ_i`` shape ``(K, P)``.

    Hallucination guard: if ``predict_fn`` returns NaN at any perturbation,
    the corresponding column is recomputed at half ``eps``; if still NaN,
    we set it to zero and emit a warning rather than producing a Fisher
    contaminated by NaN.
    """
    theta = np.asarray(theta, dtype=np.float64)
    P = theta.shape[0]
    mu0 = np.asarray(predict_fn(theta), dtype=np.float64)
    K = mu0.shape[0]

    J = np.zeros((K, P), dtype=np.float64)
    for i in range(P):
        eps = max(abs(float(theta[i])) * eps_rel, eps_abs_floor)
        plus = theta.copy()
        plus[i] = theta[i] + eps
        minus = theta.copy()
        minus[i] = theta[i] - eps
        mp = np.asarray(predict_fn(plus), dtype=np.float64)
        mm = np.asarray(predict_fn(minus), dtype=np.float64)
        if np.any(~np.isfinite(mp)) or np.any(~np.isfinite(mm)):
            # Retry at half epsilon
            eps2 = eps * 0.5
            plus2 = theta.copy(); plus2[i] = theta[i] + eps2
            minus2 = theta.copy(); minus2[i] = theta[i] - eps2
            mp = np.asarray(predict_fn(plus2), dtype=np.float64)
            mm = np.asarray(predict_fn(minus2), dtype=np.float64)
            eps = eps2
            if np.any(~np.isfinite(mp)) or np.any(~np.isfinite(mm)):
                import warnings
                warnings.warn(
                    f"FD Jacobian column {i} is NaN at theta[{i}]={theta[i]}; "
                    f"setting to zero. Fisher will be biased."
                )
                continue
        J[:, i] = (mp - mm) / (2.0 * eps)
    return J


def _native_ad_jacobian(
    predict_fn_jax: Callable[[jnp.ndarray], jnp.ndarray],
    theta: jnp.ndarray,
) -> np.ndarray:
    """Reverse-mode AD Jacobian via ``jax.jacrev``.

    Requires ``predict_fn_jax`` to be JAX-traceable end-to-end (no host
    callbacks). Returns numpy array ``(K, P)``.
    """
    theta_jax = jnp.asarray(theta, dtype=jnp.float64)
    J = jax.jacrev(predict_fn_jax)(theta_jax)
    return np.asarray(J, dtype=np.float64)


def fisher_info(
    predict_fn: Callable,
    theta_fid: np.ndarray,
    sigma_obs: np.ndarray,
    *,
    backend: str = "fd",
    fd_eps_rel: float = 1.0e-4,
    fd_eps_abs_floor: float = 1.0e-10,
) -> np.ndarray:
    """Compute the Fisher information matrix.

    Parameters
    ----------
    predict_fn : callable(theta) -> mu
        Maps parameter vector ``θ`` (shape ``(P,)``) to predicted observables
        ``μ`` (shape ``(K,)``). For ``backend='fd'`` may return numpy or JAX.
        For ``backend='diffrax'`` must be JAX-traceable end-to-end.
    theta_fid : array (P,)
        Fiducial parameter point.
    sigma_obs : array (K,)
        Observational uncertainties (one per observable).
    backend : {'fd', 'diffrax'}
        Jacobian computation method.
    fd_eps_rel, fd_eps_abs_floor : FD step controls.

    Returns
    -------
    F : (P, P) Fisher matrix.
    """
    theta = np.asarray(theta_fid, dtype=np.float64)
    sigma = np.asarray(sigma_obs, dtype=np.float64)

    if backend == "fd":
        J = _central_fd_jacobian(predict_fn, theta, fd_eps_rel, fd_eps_abs_floor)
    elif backend == "diffrax":
        J = _native_ad_jacobian(predict_fn, jnp.asarray(theta))
    else:
        raise ValueError(f"unknown backend={backend!r}; choose 'fd' or 'diffrax'")

    Sigma_inv = np.diag(1.0 / sigma**2)
    F = J.T @ Sigma_inv @ J
    return F


# ═══════════════════════════════════════════════════════════════════════
# §3. Diagnostics
# ═══════════════════════════════════════════════════════════════════════

def fisher_diagnostics(
    F: np.ndarray,
    prior_F: Optional[np.ndarray] = None,
    *,
    rank_rtol: float = 1.0e-12,
) -> FisherDiagnostics:
    """Eigendecompose F (optionally with prior added) and report diagnostics.

    Parameters
    ----------
    F : (P, P) Fisher matrix (likelihood-only).
    prior_F : optional (P, P) Fisher contribution from the prior.
    rank_rtol : tolerance for numerical rank.

    Notes
    -----
    The condition number is reported on the *combined* matrix when a prior
    is supplied (this is what the sampler sees), but the ``F`` field stores
    whichever input was used. To get the likelihood-only condition number,
    pass ``prior_F=None``.
    """
    F = np.asarray(F, dtype=np.float64)
    if prior_F is not None:
        F_eff = F + np.asarray(prior_F, dtype=np.float64)
    else:
        F_eff = F.copy()
    # Symmetrize against numerical drift
    F_eff = 0.5 * (F_eff + F_eff.T)
    # Eigh (symmetric), ascending
    eigvals, eigvecs = np.linalg.eigh(F_eff)
    lam_max = float(eigvals[-1])
    lam_min = float(eigvals[0])
    cond = lam_max / max(abs(lam_min), 1.0e-300)

    # Marginalized 1-σ (diagonal of inverse Fisher)
    try:
        F_inv = np.linalg.inv(F_eff)
        sigma_marg = np.sqrt(np.maximum(np.diag(F_inv), 0.0))
    except np.linalg.LinAlgError:
        sigma_marg = np.full_like(eigvals, np.inf)

    # Numerical rank
    rank = int(np.sum(eigvals > rank_rtol * lam_max))

    # Informative directions (descending eigenvalue order)
    order = np.argsort(eigvals)[::-1]
    informative = [(float(eigvals[k]), eigvecs[:, k].tolist()) for k in order]

    return FisherDiagnostics(
        F=F_eff,
        eigenvalues=eigvals,
        eigenvectors=eigvecs,
        condition_number=cond,
        sigma_marginalized=sigma_marg,
        rank=rank,
        informative_directions=informative,
    )


# ═══════════════════════════════════════════════════════════════════════
# §4. BBN-specific convenience: 3-parameter (Σ_H, log η, log τ_n)
# ═══════════════════════════════════════════════════════════════════════

# Parameter conventions for the canonical BBN Fisher: log-transform η and
# τ_n so that priors and Fisher diagonals are dimensionless and comparable.
PARAM_NAMES = ("Sigma_H", "log_eta", "log_tau_n")


# Representative Li7/H scale used as the 3rd-observable identifiability error
# (S1). NOT a precision observational error: the Spite-plateau scatter is ~0.04
# dex but BBN over-predicts Li7/H by a factor ~3 (the unresolved LITHIUM PROBLEM),
# so Li7/H must NOT be wired into the default likelihood at face value — it is an
# OPT-IN identifiability observable only, and any constraint using it is
# EXPLORATORY and carries the lithium-problem caveat.
LI7H_SIGMA_DEFAULT = 1.0e-10


def make_bbn_predict_callable(
    *,
    backend: str = "scipy",
    N_q: int = 20,
    correction_level: int = 3,
    include_li7h: bool = False,
) -> Callable[[np.ndarray], np.ndarray]:
    """Return a numpy-callable ``predict_fn(theta) -> [Y_p, D/H]`` (or +Li7/H).

    Parameters
    ----------
    theta : array (3,) — ``[Sigma_H, log_eta, log_tau_n]``.
    backend : passed to ``canonical_forward_solver``. For Fisher work the
        canonical SciPy reference is recommended; once Diffrax is wired
        end-to-end (Phase γ) this should switch to the JAX adjoint path.
    include_li7h : if True, append Li7/H as a 3rd observable (S1) so the Fisher
        can reach rank 3 off-null (breaking the 2-observable INF-1 ceiling).
        OPT-IN only — see ``LI7H_SIGMA_DEFAULT`` for the lithium-problem caveat.
    """
    from rabbit.inference.forward_likelihood import canonical_forward_solver

    def predict(theta: np.ndarray) -> np.ndarray:
        sigma_H = float(theta[0])
        eta = float(np.exp(theta[1]))
        tau_n = float(np.exp(theta[2]))
        pred = canonical_forward_solver(
            Sigma_H=sigma_H,
            eta=eta,
            tau_n=tau_n,
            N_q=N_q,
            correction_level=correction_level,
            backend=backend,
        )
        if include_li7h:
            li7h = pred.metadata.get("Li7H")
            return np.array([pred.Yp, pred.DH, float(li7h)], dtype=np.float64)
        return np.array([pred.Yp, pred.DH], dtype=np.float64)

    return predict


def bbn_fisher(
    *,
    sigma_H_fid: float = 0.0,
    eta_fid: float = 6.104e-10,
    tau_n_fid: float = 878.4,
    yp_sigma: float = 4.0e-3,
    dh_sigma: float = 3.0e-7,
    backend: str = "scipy",
    N_q: int = 20,
    correction_level: int = 3,
    fd_eps_rel: float = 1.0e-4,
    include_li7h: bool = False,
    li7h_sigma: float = LI7H_SIGMA_DEFAULT,
) -> Tuple[np.ndarray, FisherDiagnostics]:
    """Compute the BBN Fisher matrix and diagnostics at a fiducial point.

    The default fiducial uses Aver+ 2021 σ(Y_p) and Cooke+ 2018 σ(D/H). With
    ``include_li7h=True`` an opt-in Li7/H 3rd observable is added (S1), which
    lifts the off-null Fisher to rank 3 — see ``LI7H_SIGMA_DEFAULT`` for the
    lithium-problem caveat (opt-in identifiability only, not a default constraint).
    """
    theta_fid = np.array([sigma_H_fid, np.log(eta_fid), np.log(tau_n_fid)])
    sigma_obs = (np.array([yp_sigma, dh_sigma, li7h_sigma]) if include_li7h
                 else np.array([yp_sigma, dh_sigma]))
    predict = make_bbn_predict_callable(
        backend=backend, N_q=N_q, correction_level=correction_level,
        include_li7h=include_li7h,
    )
    F = fisher_info(predict, theta_fid, sigma_obs, backend="fd", fd_eps_rel=fd_eps_rel)
    diag = fisher_diagnostics(F, prior_F=None)
    return F, diag


# ═══════════════════════════════════════════════════════════════════════
# §5. Prior Fisher constructors (for posterior conditioning)
# ═══════════════════════════════════════════════════════════════════════

def gaussian_prior_fisher(prior_sigmas: Sequence[float]) -> np.ndarray:
    """Diagonal Fisher contribution from independent Gaussian priors.

    For prior ``θ_i ~ Normal(μ_i, σ_i)``, the Fisher contribution is
    ``F^{prior}_{ii} = 1 / σ_i²``. Off-diagonals are zero.
    """
    s = np.asarray(prior_sigmas, dtype=np.float64)
    return np.diag(1.0 / s**2)


# ═══════════════════════════════════════════════════════════════════════
# §6. Gate budgets (mirrors plan §0 / §4.1)
# ═══════════════════════════════════════════════════════════════════════

# Likelihood-only condition number budget. Above this, the data does not
# identify all parameters jointly even with reasonable priors.
#
# NOTE on the FLRW boundary (Plan §0.4 physical-meaning rule). At Sigma_H = 0
# the linear sensitivity dY/dSigma_H vanishes because the shear contribution
# enters the Friedmann equation as Sigma_H^2 (Pitrou 2018 §3; arXiv:2502.20893
# eq. 3). The likelihood-only Fisher is therefore RANK-DEFICIENT at the
# FLRW fiducial: rank = 2, not 3. The cond(F_lik) gate is consequently
# applied OFF the FLRW boundary; at Sigma_H = 0 the rank-2 condition is the
# physically meaningful test, and cond(F_post) (with priors) is the
# posterior identifiability gate.
COND_LIKELIHOOD_BUDGET = 1.0e8

# Posterior condition number budget (with default priors added).
COND_POSTERIOR_BUDGET = 1.0e6

# Minimum likelihood-Fisher rank at the FLRW boundary. The Sigma_H direction
# is degenerate by physics (Sigma_H^2 sensitivity) AT the null. Note a STRUCTURAL
# ceiling (BD599 INF-1): make_bbn_predict_callable returns only 2 observables
# (Yp, D/H) for 3 parameters, so the likelihood Fisher has rank <= 2 at EVERY
# fiducial, not just the null, with the DEFAULT 2-observable likelihood. Reaching
# rank 3 (an actually data-identified Sigma_H) requires a 3rd independent
# observable, now available OPT-IN as Li7/H via include_li7h=True (S1): with it the
# off-null Fisher reaches rank 3 (cond ~1e5). It stays OFF by default because a
# face-value Li7 likelihood is confounded by the unresolved LITHIUM PROBLEM (see
# LI7H_SIGMA_DEFAULT), so any Li7-enabled Sigma_H constraint is EXPLORATORY.
MIN_RANK_AT_FLRW = 2
