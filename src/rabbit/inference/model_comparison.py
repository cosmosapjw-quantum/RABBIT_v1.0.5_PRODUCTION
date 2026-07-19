"""
rabbit.inference.model_comparison — Bayesian model comparison for BBN.

MATURITY: EXPERIMENTAL
  Bayes factors and evidence values from this module require:
  (1) A validated forward model (canonical_forward_solver, not surrogate)
  (2) Synthetic null/recovery calibration (test_inference_validation.py)
  Null/recovery calibration tests pass (test_inference_null_recovery.py, 4/4).  Evidence values remain exploratory pending full SMC posterior validation.

Implements:
  1. Savage-Dickey density ratio for nested model comparison
  2. Profile likelihood ratio test
  3. Information criteria (AIC, BIC)
  (Posterior predictive checks against real Aver/Cooke data are NOT implemented —
   PROPOSED only; do not cite a PPC capability. BD599 INF-3.)

The key question: does the data prefer Bianchi (Σ_H free) over FLRW (Σ_H=0)?

Method: Savage-Dickey density ratio (Dickey 1971, Verdinelli & Wasserman 1995)
  For nested models M₀ ⊂ M₁ where M₀ fixes Σ_H = 0:

      B₀₁ = P(data|M₀) / P(data|M₁) = p(Σ_H=0 | data, M₁) / p(Σ_H=0 | M₁)

  where:
    - p(Σ_H=0 | data, M₁) is the posterior density at Σ_H=0 under M₁
    - p(Σ_H=0 | M₁) is the prior density at Σ_H=0 under M₁

  B₀₁ > 1 → data prefers FLRW
  B₀₁ < 1 → data prefers Bianchi

  Jeffreys scale:
    |ln B₀₁| < 1:     inconclusive
    1 < |ln B₀₁| < 2.5: moderate
    2.5 < |ln B₀₁| < 5: strong
    |ln B₀₁| > 5:       decisive

This module provides host-side grid/profile diagnostics.  No BBN-specific
BlackJAX/NSS wrapper is currently available: those entry points fail closed
until B-05, and no output from this module is a production constraint.

References:
    Dickey 1971, Ann. Math. Stat. 42, 204
    Trotta 2008, Contemp. Phys. 49, 71 (Bayesian cosmology review)
    Historical RABBIT report §VII (diagnostic Σ_H profile; not validated)
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple
import numpy as np

from rabbit.inference.forward_likelihood import (
    BBNLikelihood, ForwardModel, GridResult,
    grid_scan, confidence_interval_1d,
    DELTA_CHI2_1D, DELTA_CHI2_2D,
)


# ═══════════════════════════════════════════════════════════════
# §1. Prior specifications
# ═══════════════════════════════════════════════════════════════

@dataclass
class Prior:
    """1D prior distribution."""
    name: str
    dist_type: str  # 'uniform', 'gaussian', 'half_normal'
    params: Dict[str, float]

    def log_pdf(self, x: float) -> float:
        if self.dist_type == 'uniform':
            lo, hi = self.params['low'], self.params['high']
            if lo <= x <= hi:
                return -np.log(hi - lo)
            return -np.inf
        elif self.dist_type == 'gaussian':
            mu, sigma = self.params['mean'], self.params['std']
            return -0.5 * ((x - mu) / sigma) ** 2 - np.log(sigma * np.sqrt(2 * np.pi))
        elif self.dist_type == 'half_normal':
            sigma = self.params['std']
            if x < 0:
                return -np.inf
            return -0.5 * (x / sigma) ** 2 + np.log(2) - np.log(sigma * np.sqrt(2 * np.pi))
        raise ValueError(f"Unknown dist: {self.dist_type}")

    def pdf(self, x: float) -> float:
        return np.exp(self.log_pdf(x))

    def pdf_at_zero(self) -> float:
        """Prior density at the nested-model point (Σ_H = 0)."""
        return self.pdf(0.0)


# Standard priors for BBN parameters
PRIOR_SIGMA_H = Prior("Sigma_H", "uniform", {"low": 0.0, "high": 0.5})
PRIOR_SIGMA_H_WIDE = Prior("Sigma_H", "uniform", {"low": 0.0, "high": 1.0})
PRIOR_ETA = Prior("eta", "gaussian", {"mean": 6.104e-10, "std": 0.058e-10})
PRIOR_TAU_N = Prior("tau_n", "gaussian", {"mean": 878.4, "std": 0.5})


# ═══════════════════════════════════════════════════════════════
# §2. Savage-Dickey density ratio
# ═══════════════════════════════════════════════════════════════

@dataclass
class SavageDickeyResult:
    """Result of Savage-Dickey Bayes factor computation."""
    B01: float               # Bayes factor P(data|FLRW)/P(data|Bianchi)
    ln_B01: float            # log Bayes factor
    posterior_at_zero: float  # p(Σ=0|data, M₁)
    prior_at_zero: float     # p(Σ=0|M₁)
    interpretation: str      # Jeffreys scale interpretation
    posterior_grid: Tuple[np.ndarray, np.ndarray]  # (Σ, p(Σ|data))


def _warn_exploratory(fn_name: str, _acknowledge: bool) -> None:
    """Emit the EXPLORATORY warning unless the caller acknowledges (BD599 INF-4).

    These primitives are independently importable and emit Bayes-factor / evidence
    / p-value / sigma strings; without this guard a single direct import is one
    step from a headline. The guard previously lived only in run_full_analysis.
    """
    if not _acknowledge:
        import warnings
        warnings.warn(
            f"model_comparison.{fn_name} is EXPLORATORY: Bayes factors / evidence "
            "/ p-values require synthetic null/recovery calibration before citation. "
            "Pass _acknowledge_exploratory=True to suppress.",
            RuntimeWarning, stacklevel=3,
        )


def savage_dickey_bf(
    grid_result: GridResult,
    param_name: str = 'Sigma_H',
    prior: Prior = None,
    _acknowledge_exploratory: bool = False,
) -> SavageDickeyResult:
    """Compute Savage-Dickey Bayes factor from grid posterior.

    B₀₁ = p(Σ=0 | data, M₁) / p(Σ=0 | M₁)

    Parameters
    ----------
    grid_result : GridResult
        From grid_scan with param_name as one axis.
    param_name : str
        The parameter fixed in the null model (default: Σ_H).
    prior : Prior
        Prior on param_name under M₁.

    Returns
    -------
    SavageDickeyResult
    """
    _warn_exploratory("savage_dickey_bf", _acknowledge_exploratory)
    if prior is None:
        prior = PRIOR_SIGMA_H

    if param_name not in grid_result.marginalized_1d:
        raise ValueError(f"No marginalized 1D for {param_name}")

    grid, p1d = grid_result.marginalized_1d[param_name]

    # Posterior density at Σ=0
    # Interpolate to get exact value at 0
    if grid[0] == 0.0:
        post_at_zero = float(p1d[0])
    else:
        from scipy.interpolate import interp1d
        interp = interp1d(grid, p1d, kind='linear', fill_value=0,
                          bounds_error=False)
        post_at_zero = float(interp(0.0))

    # Prior density at Σ=0
    prior_at_zero = prior.pdf_at_zero()

    # Bayes factor
    if prior_at_zero > 0 and post_at_zero > 0:
        B01 = post_at_zero / prior_at_zero
    elif post_at_zero == 0:
        B01 = 0.0  # data strongly disfavors FLRW
    else:
        B01 = np.inf

    ln_B01 = np.log(B01) if B01 > 0 else -np.inf

    # Jeffreys scale interpretation
    abs_ln = abs(ln_B01)
    if abs_ln < 1:
        interp_str = "inconclusive"
    elif abs_ln < 2.5:
        interp_str = "moderate"
    elif abs_ln < 5:
        interp_str = "strong"
    else:
        interp_str = "decisive"

    if ln_B01 > 0:
        interp_str += " evidence for FLRW"
    elif ln_B01 < 0:
        interp_str += " evidence for Bianchi"

    return SavageDickeyResult(
        B01=B01,
        ln_B01=ln_B01,
        posterior_at_zero=post_at_zero,
        prior_at_zero=prior_at_zero,
        interpretation=interp_str,
        posterior_grid=(grid, p1d),
    )


# ═══════════════════════════════════════════════════════════════
# §3. Profile likelihood ratio
# ═══════════════════════════════════════════════════════════════

@dataclass
class ProfileLikelihoodResult:
    """Result of profile likelihood analysis."""
    chi2_min: float          # global minimum χ²
    chi2_at_zero: float      # χ² at Σ_H = 0
    delta_chi2: float        # χ²(Σ=0) − χ²_min
    p_value: float           # p-value from Δχ² with 1 dof
    sigma_equivalent: float  # Gaussian σ equivalent
    best_fit_Sigma: float
    upper_limits: Dict[float, float]  # {CL: upper_limit}; inf when unconstrained
    # {CL: True} when the allowed region reaches the last finite grid point, i.e.
    # Σ_H is unconstrained and the "limit" would merely track grid extent (INF-2).
    upper_limit_unconstrained: Dict[float, bool] = field(default_factory=dict)


def profile_likelihood(
    grid_result: GridResult,
    param_name: str = 'Sigma_H',
    _acknowledge_exploratory: bool = False,
) -> ProfileLikelihoodResult:
    """Profile likelihood analysis for Σ_H.

    Tests H₀: Σ_H = 0 vs H₁: Σ_H free.
    """
    _warn_exploratory("profile_likelihood", _acknowledge_exploratory)
    from scipy.stats import chi2 as chi2_dist

    idx = grid_result.param_names.index(param_name)
    grid = grid_result.param_grids[param_name]
    chi2_all = grid_result.chi2

    # Profile over other parameters
    axes = tuple(j for j in range(len(grid_result.param_names)) if j != idx)
    if axes:
        chi2_prof = np.min(chi2_all, axis=axes)
    else:
        chi2_prof = chi2_all.copy()

    # Mask solver-failure (non-finite χ²) points so they cannot fabricate an
    # exclusion or corrupt the minimum (INF-2).
    finite = np.isfinite(chi2_prof)
    if not np.any(finite):
        raise ValueError("profile_likelihood: all χ² are non-finite (solver failure)")
    chi2_min = float(np.min(chi2_prof[finite]))
    best_idx = int(np.where(finite)[0][np.argmin(chi2_prof[finite])])
    best_S = float(grid[best_idx])

    # χ² at Σ=0
    if grid[0] == 0.0:
        chi2_zero = float(chi2_prof[0])
    else:
        from scipy.interpolate import interp1d
        interp = interp1d(grid, chi2_prof, kind='linear')
        chi2_zero = float(interp(0.0))

    delta = chi2_zero - chi2_min

    # p-value (Wilks' theorem, 1 dof — but Σ≥0 is boundary)
    # For boundary: p-value = 0.5 × P(χ²_1 > Δχ²)
    p_value = 0.5 * chi2_dist.sf(max(delta, 0), df=1)

    # Gaussian equivalent
    from scipy.stats import norm
    sigma_eq = norm.isf(p_value) if p_value > 0 else np.inf

    # Upper limits at various CL. If the allowed region reaches the last finite
    # grid point, Σ_H is unconstrained — the grid edge is NOT a data limit, so we
    # report inf and flag it rather than fabricating a constraint (INF-2).
    last_finite = int(np.where(finite)[0][-1])
    upper_limits = {}
    upper_limit_unconstrained = {}
    for cl, thresh in DELTA_CHI2_1D.items():
        allowed = finite & (chi2_prof - chi2_min <= thresh)
        if np.any(allowed):
            last = int(np.where(allowed)[0][-1])
            unconstrained = last >= last_finite
            upper_limits[cl] = float("inf") if unconstrained else float(grid[last])
            upper_limit_unconstrained[cl] = bool(unconstrained)
        else:
            upper_limits[cl] = float(grid[0])
            upper_limit_unconstrained[cl] = False

    return ProfileLikelihoodResult(
        chi2_min=chi2_min,
        chi2_at_zero=chi2_zero,
        delta_chi2=delta,
        p_value=p_value,
        sigma_equivalent=sigma_eq,
        best_fit_Sigma=best_S,
        upper_limits=upper_limits,
        upper_limit_unconstrained=upper_limit_unconstrained,
    )


# ═══════════════════════════════════════════════════════════════
# §4. Information criteria
# ═══════════════════════════════════════════════════════════════

def information_criteria(
    chi2_flrw: float,
    chi2_bianchi: float,
    n_data: int = 2,  # Y_p + D/H
    k_flrw: int = 2,  # η, τ_n
    k_bianchi: int = 3,  # Σ_H, η, τ_n
    _acknowledge_exploratory: bool = False,
) -> Dict[str, float]:
    """Compute AIC and BIC for model comparison.

    AIC = χ² + 2k
    BIC = χ² + k ln(n)

    ΔAIC > 0 → Bianchi preferred
    ΔBIC > 0 → Bianchi preferred
    """
    _warn_exploratory("information_criteria", _acknowledge_exploratory)
    aic_flrw = chi2_flrw + 2 * k_flrw
    aic_bianchi = chi2_bianchi + 2 * k_bianchi
    bic_flrw = chi2_flrw + k_flrw * np.log(n_data)
    bic_bianchi = chi2_bianchi + k_bianchi * np.log(n_data)

    return {
        'AIC_FLRW': aic_flrw,
        'AIC_Bianchi': aic_bianchi,
        'delta_AIC': aic_flrw - aic_bianchi,
        'BIC_FLRW': bic_flrw,
        'BIC_Bianchi': bic_bianchi,
        'delta_BIC': bic_flrw - bic_bianchi,
    }


# ═══════════════════════════════════════════════════════════════
# §5. Full analysis pipeline
# ═══════════════════════════════════════════════════════════════

@dataclass
class FullAnalysisResult:
    """Complete model comparison result."""
    # Grid scan
    grid_result: GridResult
    # Profile likelihood
    profile: ProfileLikelihoodResult
    # Savage-Dickey
    savage_dickey: SavageDickeyResult
    # Information criteria
    ic: Dict[str, float]
    # Summary
    sigma_upper_95: float
    bayes_factor: float
    interpretation: str


def run_full_analysis(
    likelihood: BBNLikelihood,
    Sigma_grid: np.ndarray = None,
    prior: Prior = None,
    verbose: bool = True,
    prewarm_likelihood: bool = False,
    _acknowledge_exploratory: bool = False,
) -> FullAnalysisResult:
    """Run complete Σ_H constraint + model comparison analysis.

    MATURITY: EXPLORATORY.  Results should not be cited as
    science constraints without B-05 canonical inference and synthetic
    null/recovery calibration. Set _acknowledge_exploratory=True to suppress
    this warning.

    Parameters
    ----------
    likelihood : BBNLikelihood
    Sigma_grid : 1D array
    prior : Prior on Σ_H
    verbose : bool
    prewarm_likelihood : bool
        If True, stage any available likelihood/forward-model prewarm contract
        before the grid-scan body.

    Returns
    -------
    FullAnalysisResult
    """
    if not _acknowledge_exploratory:
        import warnings
        warnings.warn(
            "model_comparison.run_full_analysis is EXPLORATORY. "
            "Bayes factors and evidence values require synthetic "
            "null/recovery calibration before citation. "
            "Pass _acknowledge_exploratory=True to suppress.",
            RuntimeWarning, stacklevel=2,
        )

    if Sigma_grid is None:
        Sigma_grid = np.linspace(0, 0.5, 51)
    if prior is None:
        prior = PRIOR_SIGMA_H

    # 1. Grid scan
    if verbose:
        print("  Step 1: Grid scan...")
    gr = grid_scan(likelihood,
                    param_grids={'Sigma_H': Sigma_grid},
                    verbose=False,
                    prewarm_likelihood=prewarm_likelihood)

    # 2. Profile likelihood
    if verbose:
        print("  Step 2: Profile likelihood...")
    prof = profile_likelihood(gr, 'Sigma_H', _acknowledge_exploratory=True)

    # 3. Savage-Dickey
    if verbose:
        print("  Step 3: Savage-Dickey Bayes factor...")
    sd = savage_dickey_bf(gr, 'Sigma_H', prior, _acknowledge_exploratory=True)

    # 4. Information criteria
    if verbose:
        print("  Step 4: Information criteria...")
    ic = information_criteria(
        chi2_flrw=prof.chi2_at_zero,
        chi2_bianchi=prof.chi2_min,
        _acknowledge_exploratory=True,
    )

    sigma_95 = prof.upper_limits.get(0.95, Sigma_grid[-1])

    if verbose:
        print(f"\n  ═══ RESULTS ═══")
        print(f"  Σ_H best fit:  {prof.best_fit_Sigma:.4f}")
        print(f"  Σ_H < {sigma_95:.3f} (95% CL)")
        print(f"  Δχ²(Σ=0):     {prof.delta_chi2:.3f}")
        print(f"  p-value:       {prof.p_value:.4f}")
        print(f"  B₀₁ (FLRW/Bianchi): {sd.B01:.3f}")
        print(f"  ln B₀₁:       {sd.ln_B01:.3f}")
        print(f"  Interpretation: {sd.interpretation}")
        print(f"  ΔAIC:          {ic['delta_AIC']:+.2f}")
        print(f"  ΔBIC:          {ic['delta_BIC']:+.2f}")

    return FullAnalysisResult(
        grid_result=gr,
        profile=prof,
        savage_dickey=sd,
        ic=ic,
        sigma_upper_95=sigma_95,
        bayes_factor=sd.B01,
        interpretation=sd.interpretation,
    )



def run_canonical_full_analysis(
    Sigma_grid: np.ndarray = None,
    prior: Prior = None,
    verbose: bool = True,
    prewarm_likelihood: Optional[bool] = None,
    observations: Optional[List] = None,
    parameter_priors: Optional[Dict[str, object]] = None,
    **solver_kwargs,
) -> FullAnalysisResult:
    """Build the canonical likelihood and run the full analysis pipeline.

    This wrapper keeps top-level inference entrypoints aligned with the
    validated canonical forward model and carries the JAX prewarm lifecycle up
    to the production-style analysis API.
    """
    from rabbit.inference.forward_likelihood import make_canonical_likelihood

    if prewarm_likelihood is None:
        prewarm_likelihood = bool(
            solver_kwargs.get('backend', 'auto') == 'jax'
            and solver_kwargs.get('prewarm_jax', False)
        )

    likelihood = make_canonical_likelihood(
        observations=observations,
        parameter_priors=parameter_priors,
        auto_prewarm_on_first_loglike=False,
        **solver_kwargs,
    )
    return run_full_analysis(
        likelihood,
        Sigma_grid=Sigma_grid,
        prior=prior,
        verbose=verbose,
        prewarm_likelihood=bool(prewarm_likelihood),
    )
