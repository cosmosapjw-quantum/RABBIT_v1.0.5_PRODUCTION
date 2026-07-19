"""
rabbit.inference.forward_likelihood — Forward likelihood for BBN inference.

Maps physical parameters θ = {Σ_H, η, τ_n} to observables {Y_p, D/H}
via the registry-dispatched BBN forward model, and computes the log-likelihood
against observational data.

This module provides:
  1. ForwardModel: wrapper around any BBN driver
  2. BBNLikelihood: Gaussian likelihood from (Y_p, D/H) observations
  3. canonical_forward_solver: registry-canonical runtime dispatch
  4. _simplified_bbn_solver: SURROGATE for pipeline testing (NOT for publication)

Dispatch tiers (see backend_capabilities.py):
  - canonical default (auto): SciPy/BDF reference path
  - canonical reference (scipy): SciPy full_coupled_typeI, regression-locked

Current policy:
  - `backend="auto"` resolves statically to the SciPy Type-I reference.
  - Retired JAX endpoint and alternate-geometry names fail closed.
  - Full-Boltzmann, alternate-geometry, and low-level component source is
    preserved outside this public forward dispatcher.

PORT-00 compatibility notice: legacy functions, metadata keys, contracts, and
error payloads containing ``canonical``, ``production``, ``promoted``, or
``readiness`` are frozen historical wire vocabulary. They grant no current or
future authority and are removed or renamed at R-06; no new JAX work may be
selected from them.

The _simplified_bbn_solver is a calibrated polynomial surrogate.
It is NOT connected to the ODE solver and must NOT be used for paper claims.

Observational constraints:
    Y_p = 0.2449 ± 0.0040  (Aver et al. 2021)
    D/H = (2.547 ± 0.029) × 10⁻⁵  (Cooke, Pettini & Steidel 2018)
    τ_n = 878.4 ± 0.5 s  (PDG 2024)
    η = (6.104 ± 0.058) × 10⁻¹⁰  (Planck 2018 + PDG)

References:
    PRIMAT: Pitrou et al. 2018, §V (constraints)
    Historical RABBIT report: §IV.E (diagnostic Σ_H profile; not validated)
"""
from __future__ import annotations

from dataclasses import dataclass, field, replace
from typing import Callable, Dict, List, Optional, Tuple
import numpy as np
from rabbit.config.backend_capabilities import CAPABILITY_BY_BACKEND
from rabbit.inference.observables import (
    prediction_valid_mask_for_inference,
    prediction_value_for_observation,
    resolve_observable_key,
    validate_prediction_for_inference,
)


# ═══════════════════════════════════════════════════════════════
# §1. Observational data
# ═══════════════════════════════════════════════════════════════
#
# Values are loaded from the versioned registry rabbit.data (single source
# of truth introduced in v2 foundation F12).  The local Observation class
# is retained as a back-compat alias of rabbit.data.Observation so existing
# call sites that import Observation from this module keep working.

from rabbit.data import Observation, load_observations

_OBS = load_observations(version="1.0")

# Standard BBN observations (back-compat names; sourced from registry)
OBS_YP = _OBS["Yp"]
OBS_DH = _OBS["DH"]
OBS_TAU_N = _OBS["tau_n"]
OBS_ETA = _OBS["eta"]

_REMOVED_TYPEI_JAX_BACKENDS = frozenset({
    "jax",
    "jax_advanced",
    "jax_characteristic",
    "jax_characteristic_tier2",
    "jax_characteristic_nonlrs",
    "jax_ap_unified_tier3",
    "jax_classA",
    "jax_classB",
    "jax_tilted",
    "jax_tilted_full_coupled",
})


# ═══════════════════════════════════════════════════════════════
# §2. Forward model result
# ═══════════════════════════════════════════════════════════════

@dataclass
class BBNPrediction:
    """Output of the forward BBN model."""
    Yp: float
    DH: float
    params: Dict[str, float] = field(default_factory=dict)
    metadata: Dict = field(default_factory=dict)
    success: bool = True


def get_reported_N_eff(prediction: "BBNPrediction") -> float:
    """Return the prediction N_eff, warning if it is not a DERIVED value.

    Enforces the PR-B / BD599 W2-R3 provenance contract: at Tier 1 the reported
    N_eff is the parametric entropy-ratio diagnostic, not a measurement. A derived
    N_eff requires tier>=2 / a decoupling backbone
    (metadata['N_eff_is_derived']=True). Cite this helper rather than reading
    metadata['N_eff'] directly so a parametric value cannot silently read as data.
    """
    if not prediction.metadata.get('N_eff_is_derived', False):
        import warnings
        warnings.warn(
            "Reported N_eff is the parametric Tier-1 diagnostic (not derived from "
            "an evolved neutrino sector); do not cite it as a measured N_eff — use a "
            "tier>=2 forward. (BD599 W2-R3 / BD598 A-R1.)",
            RuntimeWarning, stacklevel=2,
        )
    return prediction.metadata.get('N_eff')


class ForwardModel:
    """Wrapper around a BBN driver for inference.

    The forward model maps θ = {Σ_H, η, τ_n, ...} to {Y_p, D/H}.

    Parameters
    ----------
    solver_fn : callable
        Function (Sigma_H, eta, tau_n, **kwargs) → BBNPrediction
    default_params : dict
        Default parameter values for parameters not being varied.
    prewarm_fn : callable, optional
        Zero-side-effect runtime warmup hook. When present, it is executed once
        before the first prediction if ``auto_prewarm_on_first_predict`` is True.
    auto_prewarm_on_first_predict : bool
        Enable one-shot prewarm before the first solver call. This is useful for
        JAX backends where compile/setup cost should be staged explicitly.
    """

    def __init__(
        self,
        solver_fn: Callable,
        default_params: Dict[str, float] = None,
        prewarm_fn: Optional[Callable] = None,
        auto_prewarm_on_first_predict: bool = False,
    ):
        self.solver_fn = solver_fn
        self.default_params = default_params or {
            'Sigma_H': 0.0,
            'eta': 6.104e-10,
            'tau_n': 878.4,
        }
        self.prewarm_fn = prewarm_fn
        self.auto_prewarm_on_first_predict = bool(auto_prewarm_on_first_predict)
        self._prewarmed = False
        self._last_prewarm_summary = None

    def prewarm(self):
        """Execute the optional one-shot runtime warmup hook."""
        if self._prewarmed:
            return self._last_prewarm_summary
        if self.prewarm_fn is None:
            self._prewarmed = True
            self._last_prewarm_summary = None
            return None
        self._last_prewarm_summary = self.prewarm_fn()
        self._prewarmed = True
        return self._last_prewarm_summary

    def predict(self, **params) -> BBNPrediction:
        """Run the forward model with given parameters."""
        full_params = {**self.default_params, **params}
        try:
            if self.auto_prewarm_on_first_predict and not self._prewarmed:
                self.prewarm()
            pred = self.solver_fn(**full_params)
            if self._last_prewarm_summary is not None:
                pred.metadata = dict(pred.metadata)
                pred.metadata.setdefault('forward_model_prewarm_summary', self._last_prewarm_summary)
                pred.metadata.setdefault('forward_model_auto_prewarm', bool(self.auto_prewarm_on_first_predict))
            return pred
        except Exception as e:
            return BBNPrediction(
                Yp=np.nan, DH=np.nan,
                params=full_params,
                metadata={'error': str(e)},
                success=False,
            )

    def log_likelihood(self, observation, **params) -> float:
        """Convenience: compute Gaussian log-likelihood for a single observation.

        Parameters
        ----------
        observation : Observation (or list of Observation)
            Observable(s) to compare against.
        **params : forwarded to predict().

        Returns
        -------
        float : log L = -0.5 * sum_i ((pred_i - obs_i) / sigma_i)^2
        """
        pred = self.predict(**params)
        obs_list = observation if isinstance(observation, (list, tuple)) else [observation]
        if not validate_prediction_for_inference(
            pred, [resolve_observable_key(obs.name) for obs in obs_list]
        ):
            return -np.inf
        ll = 0.0
        for obs in obs_list:
            pred_val = prediction_value_for_observation(pred, obs)
            ll -= 0.5 * ((pred_val - obs.value) / obs.sigma) ** 2
        return float(ll)


# ═══════════════════════════════════════════════════════════════
# §4. Likelihood
# ═══════════════════════════════════════════════════════════════

class BBNLikelihood:
    """Gaussian likelihood for BBN observables.

    log L(θ) = -½ Σ_i [(O_i^pred(θ) - O_i^obs) / σ_i]²

    Parameters
    ----------
    forward_model : ForwardModel
    observations : list of Observation
        Which observables to include in the likelihood.
    parameter_priors : dict of Observation, optional
        Gaussian priors on parameters (e.g., τ_n, η).
    auto_prewarm_on_first_loglike : bool
        When True, stage any available forward-model prewarm contract before the
        first likelihood evaluation. This keeps compile/setup tax outside the
        repeated body of grid scans or direct-sampling loops.
    """

    def __init__(self,
                 forward_model: ForwardModel,
                 observations: List[Observation] = None,
                 parameter_priors: Dict[str, Observation] = None,
                 auto_prewarm_on_first_loglike: bool = False):
        self.forward_model = forward_model
        self.observations = observations or [OBS_YP, OBS_DH]
        self.parameter_priors = parameter_priors or {}
        self.auto_prewarm_on_first_loglike = bool(auto_prewarm_on_first_loglike)
        self._prewarmed = False
        self._last_prewarm_summary = None

    def prewarm(self):
        """Forward any optional one-shot prewarm contract to the model."""
        if self._prewarmed:
            return self._last_prewarm_summary
        prewarm_fn = getattr(self.forward_model, 'prewarm', None)
        if callable(prewarm_fn):
            self._last_prewarm_summary = prewarm_fn()
        else:
            self._last_prewarm_summary = None
        self._prewarmed = True
        return self._last_prewarm_summary

    def log_likelihood(self, **params) -> float:
        """Compute log-likelihood at given parameters."""
        if self.auto_prewarm_on_first_loglike and not self._prewarmed:
            self.prewarm()
        pred = self.forward_model.predict(**params)
        if not validate_prediction_for_inference(
            pred, [resolve_observable_key(obs.name) for obs in self.observations]
        ):
            return -np.inf

        ll = 0.0

        # Observable likelihood
        for obs in self.observations:
            ll += obs.log_likelihood(prediction_value_for_observation(pred, obs))

        # Parameter priors
        for pname, prior in self.parameter_priors.items():
            if pname in params:
                ll += prior.log_likelihood(params[pname])

        return ll

    def chi2(self, **params) -> float:
        """Compute total chi² at given parameters."""
        return -2.0 * self.log_likelihood(**params)

    def delta_chi2(self, params_test: dict, params_ref: dict = None) -> float:
        """Δχ² relative to reference (FLRW by default)."""
        if params_ref is None:
            params_ref = self.forward_model.default_params
        return self.chi2(**params_test) - self.chi2(**params_ref)


# ═══════════════════════════════════════════════════════════════
# §5. Grid posterior
# ═══════════════════════════════════════════════════════════════

@dataclass
class GridResult:
    """Result of grid-based posterior evaluation."""
    param_names: List[str]
    param_grids: Dict[str, np.ndarray]
    log_likelihood: np.ndarray  # shape matches meshgrid
    chi2: np.ndarray
    best_fit: Dict[str, float]
    best_chi2: float
    # Marginalized 1D posteriors
    marginalized_1d: Dict[str, Tuple[np.ndarray, np.ndarray]] = field(default_factory=dict)


def grid_scan(
    likelihood: BBNLikelihood,
    param_grids: Dict[str, np.ndarray],
    fixed_params: Dict[str, float] = None,
    verbose: bool = True,
    prewarm_likelihood: bool = False,
) -> GridResult:
    """Brute-force grid evaluation of the likelihood.

    Parameters
    ----------
    likelihood : BBNLikelihood
    param_grids : dict
        {param_name: 1D array of values} for each parameter to scan.
    fixed_params : dict
        Fixed parameter values (not scanned).
    verbose : bool
        Print progress.
    prewarm_likelihood : bool
        If True, run any available one-shot prewarm contract before entering the
        scan loop. This is mainly useful for JAX direct-solver likelihoods where
        compile/setup cost should be staged outside the timed grid body.

    Returns
    -------
    GridResult
    """
    if prewarm_likelihood and callable(getattr(likelihood, 'prewarm', None)):
        likelihood.prewarm()

    fixed = fixed_params or {}
    names = list(param_grids.keys())
    grids = [param_grids[n] for n in names]

    # Create meshgrid
    if len(names) == 1:
        mesh = [grids[0]]
        shape = (len(grids[0]),)
    elif len(names) == 2:
        mesh = np.meshgrid(grids[0], grids[1], indexing='ij')
        shape = mesh[0].shape
    elif len(names) == 3:
        mesh = np.meshgrid(grids[0], grids[1], grids[2], indexing='ij')
        shape = mesh[0].shape
    else:
        raise ValueError("Grid scan supports 1-3 parameters")

    ll_grid = np.full(shape, -np.inf)
    total = np.prod(shape)

    for idx in np.ndindex(shape):
        params = dict(fixed)
        for i, name in enumerate(names):
            params[name] = float(mesh[i][idx])

        ll = likelihood.log_likelihood(**params)
        ll_grid[idx] = ll

        if verbose and (np.ravel_multi_index(idx, shape) + 1) % max(total // 10, 1) == 0:
            pct = 100 * (np.ravel_multi_index(idx, shape) + 1) / total
            print(f"  Grid scan: {pct:.0f}% ({np.ravel_multi_index(idx, shape)+1}/{total})")

    chi2_grid = -2.0 * ll_grid
    best_idx = np.unravel_index(np.argmin(chi2_grid), shape)
    best_fit = {n: float(mesh[i][best_idx]) for i, n in enumerate(names)}
    best_chi2 = float(chi2_grid[best_idx])

    # Marginalized 1D posteriors
    marg_1d = {}
    ll_shifted = ll_grid - np.nanmax(ll_grid)  # shift for numerical stability
    posterior = np.exp(ll_shifted)

    for i, name in enumerate(names):
        axes = tuple(j for j in range(len(names)) if j != i)
        if axes:
            p_marg = np.sum(posterior, axis=axes)
        else:
            p_marg = posterior
        p_marg /= np.sum(p_marg) * (grids[i][1] - grids[i][0]) if len(grids[i]) > 1 else 1
        marg_1d[name] = (grids[i], p_marg)

    result = GridResult(
        param_names=names,
        param_grids=param_grids,
        log_likelihood=ll_grid,
        chi2=chi2_grid,
        best_fit=best_fit,
        best_chi2=best_chi2,
        marginalized_1d=marg_1d,
    )
    return result


# ═══════════════════════════════════════════════════════════════
# §6. Confidence intervals from Δχ²
# ═══════════════════════════════════════════════════════════════

# Δχ² thresholds for 1 parameter
DELTA_CHI2_1D = {
    0.68: 1.00,   # 1σ
    0.90: 2.71,
    0.95: 3.84,   # 95% CL
    0.99: 6.63,
}

# Δχ² thresholds for 2 parameters
DELTA_CHI2_2D = {
    0.68: 2.30,
    0.90: 4.61,
    0.95: 5.99,
    0.99: 9.21,
}


def confidence_interval_1d(
    grid_result: GridResult,
    param_name: str,
    cl: float = 0.95,
) -> Tuple[float, float]:
    """Extract 1D confidence interval from grid scan via Δχ².

    Parameters
    ----------
    grid_result : GridResult
    param_name : str
    cl : float
        Confidence level (0.68, 0.90, 0.95, 0.99).

    Returns
    -------
    (lower, upper) bounds.
    """
    if param_name not in grid_result.param_grids:
        raise ValueError(f"Parameter {param_name} not in grid")

    idx = grid_result.param_names.index(param_name)
    grid = grid_result.param_grids[param_name]

    # Profile likelihood: minimize over other parameters
    chi2 = grid_result.chi2
    axes = tuple(j for j in range(len(grid_result.param_names)) if j != idx)
    if axes:
        chi2_profile = np.min(chi2, axis=axes)
    else:
        chi2_profile = chi2

    delta = chi2_profile - grid_result.best_chi2
    threshold = DELTA_CHI2_1D.get(cl, 3.84)

    # Find where Δχ² crosses threshold
    below = delta <= threshold
    if not np.any(below):
        return (grid[0], grid[-1])  # all excluded

    indices = np.where(below)[0]
    lower = grid[indices[0]]
    upper = grid[indices[-1]]

    return (float(lower), float(upper))


# ═══════════════════════════════════════════════════════════════
# §7. Simplified BBN solver for standalone testing
# ═══════════════════════════════════════════════════════════════

def _simplified_bbn_solver(
    Sigma_H: float = 0.0,
    eta: float = 6.104e-10,
    tau_n: float = 878.4,
) -> BBNPrediction:
    """SURROGATE — Calibrated analytical scaling, NOT the full ODE solver.

    DO NOT use for publication results. No forward path has publication
    authority until its named Rust-first gates pass.

    Uses analytical scaling relations:
        Y_p ≈ Y_p(FLRW) + A × Σ² + B × Δτ_n + C × Δη
        D/H ≈ D/H(FLRW) × (η₀/η)^1.6

    These are calibrated to match the full solver at the 10% level
    for |Σ_H| < 0.5, |Δτ_n| < 2 s, |Δη/η| < 0.1.
    """
    import warnings
    warnings.warn(
        "_simplified_bbn_solver is a calibrated surrogate, NOT the full ODE "
        "solver. No current forward path has publication authority.",
        DeprecationWarning, stacklevel=2)
    # FLRW baseline
    Yp_flrw = 0.2422  # Born, Tier 1
    DH_flrw = 2.488e-5

    # Shear effect: ΔY_p = 0.0042 × ln(1/(1−Σ²)) (ODE-calibrated, canonical v8)
    delta_Yp_shear = 0.0042 * np.log(1.0 / (1.0 - Sigma_H**2))

    # τ_n sensitivity: dY_p/dτ_n ≈ 5×10⁻⁴ per s
    delta_Yp_tau = 5e-4 * (tau_n - 878.4)

    # η sensitivity: dY_p/dη ≈ -0.015 per 10⁻¹⁰
    delta_Yp_eta = -0.015 * (eta - 6.104e-10) / 1e-10

    Yp = Yp_flrw + delta_Yp_shear + delta_Yp_tau + delta_Yp_eta

    # D/H: strong η dependence (power law)
    DH = DH_flrw * (6.104e-10 / max(eta, 1e-12)) ** 1.6

    return BBNPrediction(
        Yp=Yp, DH=DH,
        params={'Sigma_H': Sigma_H, 'eta': eta, 'tau_n': tau_n},
        # BD599 BIA-3: mark surrogate output so it cannot be silently fed into a
        # likelihood/posterior as if it were the full-solver yield.
        metadata={'surrogate': True},
    )


def _surface_scope_metadata(
    capability,
    *,
    transport_mode: str | None = None,
    transport_species_mode: str | None = None,
    production_authority: str | None = None,
    decoupling_backbone_mode: str | None = None,
):
    transport_scope_contract = (
        capability.transport_scope_contract or "scope_unspecified_v1"
    )
    thermo_scope_contract = (
        capability.thermo_scope_contract or "scope_unspecified_v1"
    )
    collision_scope_contract = (
        capability.collision_scope_contract or "scope_unspecified_v1"
    )
    runtime_surface_contract = (
        capability.readiness_scope_contract
        or f"{capability.effective_surface_class}_scope_unspecified_v1"
    )

    if production_authority == "characteristic_decoupling_backbone_residual_relaxation":
        runtime_surface_contract = "scipy_characteristic_decoupling_backbone_runtime_v1"
        transport_scope_contract = "characteristic_per_species_transport_v1"
        thermo_scope_contract = "tier2_isotropic_decoupling_backbone_v1"
        collision_scope_contract = "anisotropic_residual_relaxation_v1"
    elif production_authority == "legacy_characteristic_species_bridge":
        if decoupling_backbone_mode == "isotropic_momentum_grid_v1":
            runtime_surface_contract = "scipy_characteristic_decoupling_backbone_runtime_v1"
            transport_scope_contract = "characteristic_per_species_transport_v1"
            thermo_scope_contract = "tier2_isotropic_decoupling_backbone_v1"
            collision_scope_contract = "legacy_species_residual_bridge_v1"
        else:
            runtime_surface_contract = "scipy_characteristic_legacy_collision_runtime_v1"
            transport_scope_contract = "characteristic_per_species_transport_v1"
            thermo_scope_contract = "tier2_table_3T_transitional_v1"
            collision_scope_contract = "legacy_species_bridge_collision_v1"
    elif production_authority == "raw_characteristic_per_species":
        runtime_surface_contract = "scipy_characteristic_per_species_runtime_v1"
        transport_scope_contract = "characteristic_per_species_transport_v1"
        thermo_scope_contract = "tier2_3T_characteristic_bounded_v1"
        collision_scope_contract = "production_per_species_characteristic_closure_v1"
    elif capability.key == "scipy_typeI_reference":
        runtime_surface_contract = "scipy_typeI_core_reference_runtime_v1"

    canonical_surface = "none"
    if capability.key == "scipy_typeI_reference":
        canonical_surface = "typeI_characteristic"

    return {
        "surface_class": capability.effective_surface_class,
        "canonical_surface": canonical_surface,
        "canonical_claim_scope": (
            "canonical Type I characteristic BBN core"
            if canonical_surface == "typeI_characteristic"
            else "not a canonical Type I characteristic claim surface"
        ),
        "validation_mode": capability.validation_mode,
        "readiness_scope_contract": (
            capability.readiness_scope_contract or "scope_unspecified_v1"
        ),
        "transport_scope_contract": transport_scope_contract,
        "thermo_scope_contract": thermo_scope_contract,
        "collision_scope_contract": collision_scope_contract,
        "runtime_surface_contract": runtime_surface_contract,
        "runtime_transport_mode": transport_mode,
        "runtime_transport_species_mode": transport_species_mode,
        "runtime_production_authority": production_authority,
    }


def canonical_forward_solver(
    Sigma_H: float = 0.0,
    eta: float = 6.104e-10,
    tau_n: float = 878.4,
    correction_level: int = 0,
    n_reactions: int = 12,
    N_q: int = 20,
    backend: str = "auto",
    enable_teff: bool = False,
    enable_collisions: bool = False,
    tier: int = 1,
    rtol: float = 1e-8,
    atol: float = 1e-10,
) -> BBNPrediction:
    """Run the public SciPy Type-I BBN forward surface.

    ``backend='auto'`` resolves statically to ``'scipy'``. Retired JAX
    endpoint and alternate-geometry backend names fail explicitly; their
    underlying research/component modules are not public inference backends.
    """
    if backend in _REMOVED_TYPEI_JAX_BACKENDS:
        raise ValueError(
            f"backend={backend!r} is retired from the public forward surface. "
            "Use backend='scipy' or backend='auto'."
        )
    if backend not in CAPABILITY_BY_BACKEND:
        allowed = "', '".join(CAPABILITY_BY_BACKEND.keys())
        raise ValueError(
            f"Unknown backend: {backend!r}. Use '{allowed}'."
        )
    if backend == "auto":
        backend = "scipy"

    capability = CAPABILITY_BY_BACKEND[backend]
    if correction_level > capability.max_correction_level:
        raise ValueError(
            f"Backend {backend!r} supports weak corrections only up to "
            f"CL{capability.max_correction_level}; received CL{correction_level}."
        )
    # SciPy interim reference path
    from rabbit.drivers.full_coupled_typeI import run_full_coupled_typeI, FullCoupledConfig
    from rabbit.config.transport_mode import TransportMode
    from rabbit.config.solver_config import PRODUCTION_CONFIG

    # ── P0-3: Teff is superseded by characteristic transport on SciPy path ──
    if enable_teff:
        raise ValueError(
            f"enable_teff=True is deprecated legacy and is not supported on backend='{backend}' "
            f"(canonical characteristic transport supersedes Teff closure). "
            f"Characteristic transport captures the full q-dependent monopole distortion "
            f"exactly, making the Teff approximation unnecessary. "
            f"Teff is deprecated legacy and has no public runtime path."
        )

    # Transport mode: characteristic is default; collisions now supported
    # via Teff gather-scatter bridge in both modes.
    if enable_collisions:
        transport = TransportMode.CHARACTERISTIC
        effective_tier = max(tier, 2)  # collisions need tier >= 2
    else:
        transport = TransportMode.CHARACTERISTIC
        effective_tier = tier

    solver = PRODUCTION_CONFIG
    if not (
        float(rtol) == float(PRODUCTION_CONFIG.rtol)
        and float(atol) == float(PRODUCTION_CONFIG.atol)
    ):
        # BD613: inherit the production method (no hard-coded BDF) so an
        # rtol/atol override cannot silently change the solver identity.
        solver = replace(PRODUCTION_CONFIG, rtol=float(rtol), atol=float(atol))

    cfg = FullCoupledConfig(
        Sigma_H_plus=Sigma_H,
        eta=eta,
        tau_n=tau_n,
        correction_level=correction_level,
        n_reactions=n_reactions,
        N_q=N_q,
        transport_mode=transport,
        enable_collisions=enable_collisions,
        tier=effective_tier,
        characteristic_species_mode="auto",
        enable_teff=False,
        allow_species_identical_research=False,
        solver=solver,
    )
    try:
        result = run_full_coupled_typeI(cfg)
        return BBNPrediction(
            Yp=result.observables.Yp,
            DH=result.observables.DH,
            params={'Sigma_H': Sigma_H, 'eta': eta, 'tau_n': tau_n},
            metadata={'N_eff': result.observables.N_eff,
                      'N_eff_is_derived': bool(result.observables.N_eff_is_derived),
                      'Li7H': result.observables.Li7H,
                      'Li6H': result.observables.Li6H,
                      'T_final': (
                          float(result.trajectory['T_gamma'][-1])
                          if result.trajectory.get('T_gamma') is not None
                          and len(result.trajectory['T_gamma']) > 0
                          else None
                      ),
                      'phase1_solver_diagnostics': result.metadata.get('phase1_solver_diagnostics'),
                      'phase2_solver_diagnostics': result.metadata.get('phase2_solver_diagnostics'),
                      'weak_budget_mode': result.metadata.get('weak_budget_mode'),
                      'correction_budget_channels': result.metadata.get('correction_budget_channels'),
                      'collision_closure_mode': result.metadata.get('collision_closure_mode'),
                      # BD612-F7: mark the public SciPy opt-in collisional output as an
                      # unvalidated calibrated-RTA candidate (its collisional characteristic
                      # reference is a documented anomaly; see B4-PR4 / the unsound test).
                      'collision_model': (
                          'calibrated_rta_candidate' if bool(enable_collisions) else None
                      ),
                      'residual_rate_calibration_mode': result.metadata.get('residual_rate_calibration_mode'),
                      'thermo_exchange_mode': result.metadata.get('thermo_exchange_mode'),
                      'weak_background_mode': result.metadata.get('weak_background_mode'),
                      'decoupling_backbone_mode': result.metadata.get('decoupling_backbone_mode'),
                      'decoupling_backbone_solver_diagnostics': result.metadata.get('decoupling_backbone_solver_diagnostics'),
                      'correction_level': correction_level,
                      'dispatch_backend': backend,
                      'capability_key': capability.key,
                      'backend': capability.key, 'enable_teff': bool(enable_teff),
                      'physics_scope': capability.physics_scope,
                      'weak_mode': capability.weak_mode,
                      'maturity': capability.maturity,
                      'thermo_tier': int(effective_tier),
                      'rtol': float(result.solver.rtol),
                      'atol': float(result.solver.atol),
                      'solver_method_requested': result.metadata.get('solver_method_requested'),
                      'solver_method_effective': result.metadata.get('solver_method_effective'),
                      'transport_mode': 'characteristic',
                      'characteristic_species_mode_requested': result.metadata.get(
                          'characteristic_species_mode_requested',
                          'auto' if bool(enable_collisions) and int(effective_tier) >= 2 else 'shared',
                      ),
                      'characteristic_species_mode_resolved': result.metadata.get(
                          'characteristic_species_mode_resolved',
                          result.metadata.get('transport_species_mode', 'shared'),
                      ),
                      'transport_species_mode': result.metadata.get('transport_species_mode', 'shared'),
                      'species_identical_approx': result.metadata.get('species_identical_approx', True),
                      # Legacy compatibility metadata; PORT-00 grants it no authority.
                      'production_authority': result.metadata.get('production_authority', 'raw_characteristic'),
                      'teff_mode': 'superseded_by_characteristic',
                      'teff_nq_warning': False,
                      **_surface_scope_metadata(
                          capability,
                          transport_mode='characteristic',
                          transport_species_mode=result.metadata.get('transport_species_mode', 'shared'),
                          production_authority=result.metadata.get('production_authority', 'raw_characteristic'),
                          decoupling_backbone_mode=result.metadata.get('decoupling_backbone_mode'),
                      )},
        )
    except Exception as e:
        return BBNPrediction(
            Yp=np.nan, DH=np.nan,
            params={'Sigma_H': Sigma_H, 'eta': eta, 'tau_n': tau_n},
            metadata={'error': str(e)},
            success=False,
        )


def make_canonical_forward_model(**kwargs) -> ForwardModel:
    """Create a ForwardModel using the canonical full ODE solver.

    Any keyword arguments are passed as fixed defaults to
    canonical_forward_solver (e.g., correction_level=2, N_q=20).

    The superseded high-level Type-I JAX prewarm surface has been removed.
    """
    from functools import partial

    fixed = dict(kwargs)
    if bool(fixed.pop('prewarm_jax', False)):
        raise ValueError(
            "prewarm_jax=True is unavailable because the high-level Type-I "
            "JAX runtime was removed."
        )
    fixed.pop('jax_prewarm_scope', None)
    return ForwardModel(solver_fn=partial(canonical_forward_solver, **fixed))


def make_canonical_likelihood(
    observations: Optional[List[Observation]] = None,
    parameter_priors: Optional[Dict[str, Observation]] = None,
    auto_prewarm_on_first_loglike: Optional[bool] = None,
    **solver_kwargs,
) -> BBNLikelihood:
    """Construct a canonical observational likelihood in one step.

    This is the registry-canonical runtime convenience wrapper for host-side
    diagnostic inference.  It does not establish a validated finite-shear
    publication likelihood; that claim remains blocked until B-05.
    """
    forward_model = make_canonical_forward_model(**solver_kwargs)

    if auto_prewarm_on_first_loglike is None:
        auto_prewarm_on_first_loglike = False

    return BBNLikelihood(
        forward_model=forward_model,
        observations=observations,
        parameter_priors=parameter_priors,
        auto_prewarm_on_first_loglike=bool(auto_prewarm_on_first_loglike),
    )


# ═══════════════════════════════════════════════════════════════
# §8. Quick constraint derivation
# ═══════════════════════════════════════════════════════════════

def derive_sigma_constraint(
    likelihood: BBNLikelihood,
    Sigma_grid: np.ndarray = None,
    fixed_params: Dict[str, float] = None,
    cl: float = 0.95,
    prewarm_likelihood: bool = False,
) -> Dict:
    """Derive upper limit on Σ_H from Y_p + D/H observations.

    Parameters
    ----------
    likelihood : BBNLikelihood
    Sigma_grid : 1D array of Σ_H values to scan
    fixed_params : dict of other parameters to fix
    cl : float
        Confidence level for upper limit

    Returns
    -------
    dict with 'upper_limit', 'best_fit', 'chi2_profile', 'grid'
    """
    if Sigma_grid is None:
        Sigma_grid = np.linspace(0, 0.5, 51)

    if prewarm_likelihood and callable(getattr(likelihood, 'prewarm', None)):
        likelihood.prewarm()

    fixed = fixed_params or {}

    chi2_vals = []
    for S in Sigma_grid:
        params = {**fixed, 'Sigma_H': S}
        chi2_vals.append(likelihood.chi2(**params))

    chi2_vals = np.array(chi2_vals)
    best_idx = np.argmin(chi2_vals)
    best_S = Sigma_grid[best_idx]
    delta_chi2 = chi2_vals - chi2_vals[best_idx]

    threshold = DELTA_CHI2_1D.get(cl, 3.84)
    allowed = delta_chi2 <= threshold
    if np.any(allowed):
        upper = Sigma_grid[np.where(allowed)[0][-1]]
    else:
        upper = Sigma_grid[0]

    return {
        'upper_limit': float(upper),
        'best_fit': float(best_S),
        'best_chi2': float(chi2_vals[best_idx]),
        'chi2_profile': chi2_vals,
        'delta_chi2': delta_chi2,
        'grid': Sigma_grid,
        'cl': cl,
    }
