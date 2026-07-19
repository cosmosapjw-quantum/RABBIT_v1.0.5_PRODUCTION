"""
rabbit.inference.sampler — Nested sampling (dynesty) & grid interpolation.

Two solver-support strategies are implemented here:

Strategy 1: GRID INTERPOLATION (diagnostic pre-computed emulator)
    Pre-compute BBN observables on a D-dimensional grid, then interpolate.
    This is diagnostic/build/validation infrastructure only.  Surrogate
    predictions are forbidden at likelihood and posterior boundaries.

Strategy 2: DIRECT SAMPLING (dynesty or BlackJAX)
    Run the full ODE solver at each likelihood evaluation.
    Necessary for validation and when the grid doesn't cover the posterior.

Dynesty configuration (PolyChord-equivalent tuning):
    nlive = max(100*D, 500)         — production
    nlive = max(50*D, 250)          — PolyChord-equiv (faster)
    slices = max(5*D, D*(D+3))      — slice sampling efficiency
    dlogz = 0.01–0.05               — evidence tolerance
    pfrac = 0.8                     — dynamic: 80% posterior, 20% evidence
"""
from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, Optional, Tuple, Sequence
import inspect
import numpy as np
import time

from rabbit.data import load_observations as _load_observations
from rabbit.inference.observables import (
    mapping_value_for_observation,
    validate_prediction_for_inference,
)

_OBS_REGISTRY = _load_observations(version="1.0")


# ═══════════════════════════════════════════════════════════════
# §1. Sampler configuration
# ═══════════════════════════════════════════════════════════════

@dataclass
class DynestyConfig:
    """Dynesty nested sampling configuration.

    Two presets match PolyChord-equivalent settings:
        DynestyConfig.production(n_dim)  — high-fidelity
        DynestyConfig.polychord(n_dim)   — PolyChord-equivalent
    """
    n_dim: int = 2
    nlive: int = 500
    sample: str = 'rslice'         # rslice for D>5, rwalk for D≤5
    slices: int = 10
    walks: int = 25
    dlogz: float = 0.01
    dynamic: bool = True
    pfrac: float = 0.8             # posterior fraction (dynamic mode)
    maxcall: int = 1_000_000
    seed: int = 42

    @classmethod
    def production(cls, n_dim: int) -> 'DynestyConfig':
        """High-fidelity settings: nlive=max(100D, 500)."""
        return cls(
            n_dim=n_dim,
            nlive=max(100 * n_dim, 500),
            sample='rslice' if n_dim > 5 else 'rwalk',
            slices=max(5 * n_dim, n_dim * (n_dim + 3)),
            walks=max(25, 5 * n_dim),
            dlogz=0.01,
            dynamic=True,
            pfrac=0.8,
        )

    @classmethod
    def polychord(cls, n_dim: int) -> 'DynestyConfig':
        """PolyChord-equivalent: nlive=max(50D, 250)."""
        return cls(
            n_dim=n_dim,
            nlive=max(50 * n_dim, 250),
            sample='rslice' if n_dim > 5 else 'rwalk',
            slices=max(5 * n_dim, n_dim * (n_dim + 3)),
            walks=max(25, 5 * n_dim),
            dlogz=0.05,
            dynamic=True,
            pfrac=0.8,
        )


@dataclass
class SMCConfig:
    """BlackJAX tempered SMC configuration (JAX backend)."""
    n_particles: int = 1000
    n_mcmc_steps: int = 10
    hmc_step_size: float = 0.02
    hmc_n_leapfrog: int = 5
    temperature_schedule: Optional[Sequence[float]] = None
    seed: int = 42

    def __post_init__(self):
        if self.temperature_schedule is None:
            self.temperature_schedule = [
                0.001, 0.005, 0.01, 0.02, 0.05, 0.1, 0.15, 0.2,
                0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 0.95, 1.0,
            ]


def _prediction_to_observable_dict(result) -> Dict[str, Any]:
    """Normalize solver outputs for grid/emulator construction.

    Accepts either a plain observable dict or a ``BBNPrediction``-like object.
    """
    # Consult the shared authority with an empty observable set before any
    # normalization reads Yp/DH.  The return value is intentionally ignored:
    # legacy mappings may establish missing ``success`` below, but a known
    # surrogate or malformed metadata contract must fail first.
    validate_prediction_for_inference(result, observation_names=())
    if isinstance(result, Mapping):
        out = dict(result)
        if "success" not in out:
            required = (out.get("Yp"), out.get("DH"))
            required_valid = []
            for value in required:
                try:
                    value_arr = np.ma.asarray(value)
                    valid = (
                        value_arr.ndim == 0
                        and not bool(np.ma.getmaskarray(value_arr))
                        and bool(np.isfinite(value_arr))
                    )
                except (TypeError, ValueError):
                    valid = False
                required_valid.append(valid)
            out["success"] = all(required_valid)
        out.setdefault("metadata", {})
        return out

    out = {}
    if hasattr(result, 'Yp'):
        out['Yp'] = result.Yp
    if hasattr(result, 'DH'):
        out['DH'] = result.DH
    metadata = getattr(result, 'metadata', {})
    if hasattr(result, 'success'):
        out['success'] = getattr(result, 'success')
    out['metadata'] = metadata
    for key in ('Li7H', 'Li6H', 'N_eff'):
        if isinstance(metadata, Mapping) and key in metadata:
            out[key] = metadata[key]
    return out


def _resolve_optional_prewarm(target):
    """Return an optional one-shot prewarm hook from a callable/object.

    Resolution order:
      1. ``target.prewarm`` when ``target`` is model-like
      2. ``target.__self__.prewarm`` for bound methods
      3. ``None`` if no hook is available
    """
    if target is None:
        return None
    direct = getattr(target, 'prewarm', None)
    if callable(direct):
        return direct
    owner = getattr(target, '__self__', None)
    if owner is not None:
        owner_prewarm = getattr(owner, 'prewarm', None)
        if callable(owner_prewarm):
            return owner_prewarm
    return None


def _resolve_solver_adapter(solver_fn):
    """Return ``(call_fn, prewarm_fn)`` for grid/emulator construction.

    Supported inputs:
      - callable(params_dict) -> dict
      - callable(**params) -> BBNPrediction|dict
      - ForwardModel-like object with ``predict(**params)`` and optional ``prewarm()``
      - bound ``predict`` method from a ForwardModel-like object
    """
    prewarm_fn = None

    if hasattr(solver_fn, 'predict') and callable(getattr(solver_fn, 'predict')):
        model = solver_fn
        prewarm_fn = _resolve_optional_prewarm(model)
        def call_fn(params):
            return _prediction_to_observable_dict(model.predict(**params))
        return call_fn, prewarm_fn

    prewarm_fn = _resolve_optional_prewarm(solver_fn)

    sig = inspect.signature(solver_fn)
    params = list(sig.parameters.values())
    if len(params) == 1 and params[0].kind in (inspect.Parameter.POSITIONAL_ONLY, inspect.Parameter.POSITIONAL_OR_KEYWORD):
        def call_fn(arg_dict):
            return _prediction_to_observable_dict(solver_fn(arg_dict))
    else:
        def call_fn(arg_dict):
            return _prediction_to_observable_dict(solver_fn(**arg_dict))
    return call_fn, prewarm_fn


def make_vector_loglike_adapter(likelihood, param_names: Sequence[str]) -> Callable:
    """Adapt a dict-style likelihood to a sampler-friendly vector signature.

    The returned callable preserves any available one-shot ``prewarm()`` hook by
    attaching it as an attribute on the adapter itself. This lets entrypoints
    such as ``run_dynesty`` or ``run_smc`` discover lifecycle staging even when
    the user would otherwise hide the likelihood behind an anonymous closure.
    """
    param_names = tuple(param_names)
    prewarm_fn = _resolve_optional_prewarm(likelihood)

    def loglike(theta):
        theta_arr = np.asarray(theta, dtype=float)
        params = {name: float(theta_arr[i]) for i, name in enumerate(param_names)}
        if hasattr(likelihood, 'log_likelihood') and callable(getattr(likelihood, 'log_likelihood')):
            return float(likelihood.log_likelihood(params))
        return float(likelihood(params))

    loglike.param_names = param_names
    if callable(prewarm_fn):
        loglike.prewarm = prewarm_fn
    return loglike


def make_canonical_direct_likelihood(
    auto_prewarm_on_first_loglike: Optional[bool] = None,
    **solver_kwargs,
) -> 'BBNLikelihood':
    """Build the direct-solver sampler likelihood from the canonical model.

    This mirrors :func:`rabbit.inference.forward_likelihood.make_canonical_likelihood`
    for the direct-sampling API used by dynesty and emulator validation.
    """
    from rabbit.inference.forward_likelihood import make_canonical_forward_model

    model = make_canonical_forward_model(**solver_kwargs)
    if auto_prewarm_on_first_loglike is None:
        auto_prewarm_on_first_loglike = bool(
            solver_kwargs.get('backend', 'auto') == 'jax'
            and solver_kwargs.get('prewarm_jax', False)
        )

    return BBNLikelihood(
        solver_fn=model,
        auto_prewarm_on_first_loglike=bool(auto_prewarm_on_first_loglike),
    )


# ═══════════════════════════════════════════════════════════════
# §2. Grid interpolation emulator
# ═══════════════════════════════════════════════════════════════

@dataclass
class GridEmulatorConfig:
    """Configuration for pre-computed BBN grid.

    Parameters
    ----------
    param_ranges : dict
        {name: (lo, hi, n_points)} for each parameter.
        Example: {'eta': (5.5e-10, 6.7e-10, 30),
                  'tau_n': (870, 886, 20),
                  'Sigma_H': (0, 0.5, 25)}
    observables : tuple of str
        Which BBN outputs to store. Default: ('Yp', 'DH', 'Li7H', 'Li6H').
    """
    param_ranges: Dict[str, Tuple[float, float, int]] = field(default_factory=dict)
    observables: Tuple[str, ...] = ('Yp', 'DH', 'Li7H', 'Li6H')

    @property
    def n_dim(self) -> int:
        return len(self.param_ranges)

    @property
    def grid_shape(self) -> Tuple[int, ...]:
        return tuple(v[2] for v in self.param_ranges.values())

    @property
    def total_points(self) -> int:
        n = 1
        for _, _, npts in self.param_ranges.values():
            n *= npts
        return n

    @classmethod
    def standard_2d(cls) -> 'GridEmulatorConfig':
        """Standard 2D grid: η × τ_n."""
        return cls(param_ranges={
            'eta': (5.5e-10, 6.7e-10, 40),
            'tau_n': (870.0, 886.0, 30),
        })

    @classmethod
    def anisotropic_3d(cls) -> 'GridEmulatorConfig':
        """3D grid: η × τ_n × Σ_H."""
        return cls(param_ranges={
            'eta': (5.5e-10, 6.7e-10, 30),
            'tau_n': (870.0, 886.0, 20),
            'Sigma_H': (0.0, 0.5, 25),
        })


class GridEmulator:
    """Pre-computed BBN grid with N-D interpolation.

    Usage:
        emulator = GridEmulator.build(config, bbn_solver_fn)
        Yp = emulator.predict({'eta': 6.1e-10, 'tau_n': 878.4}, 'Yp')

    The build() step runs the full BBN solver at each grid point (slow),
    but predict() is O(1) via RegularGridInterpolator (~10⁴× faster).

    Persistence: save/load via .to_npz() / .from_npz().
    """

    def __init__(self, config: GridEmulatorConfig, grids: Dict[str, np.ndarray],
                 data: Dict[str, np.ndarray]):
        self.config = config
        self.grids = grids      # {name: 1d array}
        self.data = data        # {obs_name: nd array}
        self._interpolators = {}

    def _get_interp(self, obs_name: str):
        if obs_name not in self._interpolators:
            from scipy.interpolate import RegularGridInterpolator
            axes = [self.grids[name] for name in self.config.param_ranges]
            self._interpolators[obs_name] = RegularGridInterpolator(
                axes, self.data[obs_name], method='linear',
                bounds_error=False, fill_value=None)
        return self._interpolators[obs_name]

    def predict(self, params: Dict[str, float], obs_name: str) -> float:
        """Interpolate BBN observable at given parameters."""
        point = np.array([params[name] for name in self.config.param_ranges])
        return float(self._get_interp(obs_name)(point.reshape(1, -1))[0])

    def predict_all(self, params: Dict[str, float]) -> Dict[str, object]:
        """Interpolate all observables and retain surrogate provenance."""
        prediction = {
            obs: self.predict(params, obs) for obs in self.config.observables
        }
        return {
            **prediction,
            "success": all(np.isfinite(value) for value in prediction.values()),
            "metadata": {
                "surrogate": True,
                "surrogate_kind": "grid_emulator",
            },
        }

    def to_npz(self, path: str):
        """Save grid to compressed npz file."""
        save_dict = {}
        for name, arr in self.grids.items():
            save_dict[f'grid_{name}'] = arr
        for name, arr in self.data.items():
            save_dict[f'data_{name}'] = arr
        # Save param names and ranges as metadata
        save_dict['param_names'] = np.array(list(self.config.param_ranges.keys()))
        save_dict['obs_names'] = np.array(list(self.config.observables))
        np.savez_compressed(path, **save_dict)

    @classmethod
    def from_npz(cls, path: str) -> 'GridEmulator':
        """Load pre-computed grid from npz file."""
        d = np.load(path, allow_pickle=True)
        param_names = list(d['param_names'])
        obs_names = list(d['obs_names'])
        grids = {name: d[f'grid_{name}'] for name in param_names}
        data = {name: d[f'data_{name}'] for name in obs_names}
        # Reconstruct config
        ranges = {}
        for name in param_names:
            arr = grids[name]
            ranges[name] = (float(arr[0]), float(arr[-1]), len(arr))
        config = GridEmulatorConfig(param_ranges=ranges, observables=tuple(obs_names))
        return cls(config, grids, data)

    @classmethod
    def build(cls, config: GridEmulatorConfig,
              solver_fn: Callable,
              verbose: bool = True) -> 'GridEmulator':
        """Build grid by running BBN solver at each point.

        Parameters
        ----------
        solver_fn : callable or ForwardModel-like object
            Supported forms:
              - solver_fn(params_dict) -> dict
              - solver_fn(**params) -> BBNPrediction|dict
              - ForwardModel-like object with ``predict(**params)`` and optional
                ``prewarm()``.
        """
        call_solver, prewarm_fn = _resolve_solver_adapter(solver_fn)
        if prewarm_fn is not None:
            prewarm_fn()

        param_names = list(config.param_ranges.keys())
        grids_1d = {}
        for name, (lo, hi, n) in config.param_ranges.items():
            grids_1d[name] = np.linspace(lo, hi, n)

        # Full grid via meshgrid
        mesh = np.meshgrid(*[grids_1d[n] for n in param_names], indexing='ij')
        flat_points = np.column_stack([m.ravel() for m in mesh])
        n_total = flat_points.shape[0]

        # Initialize output arrays
        data = {obs: np.zeros(config.grid_shape) for obs in config.observables}

        t0 = time.perf_counter()
        for idx in range(n_total):
            params = {name: flat_points[idx, j] for j, name in enumerate(param_names)}
            try:
                result = call_solver(params)
                multi_idx = np.unravel_index(idx, config.grid_shape)
                for obs in config.observables:
                    if obs in result:
                        data[obs][multi_idx] = result[obs]
            except Exception as e:
                if verbose:
                    print(f"  Warning: solver failed at {params}: {e}")

            if verbose and (idx + 1) % max(n_total // 10, 1) == 0:
                elapsed = time.perf_counter() - t0
                eta = elapsed / (idx + 1) * (n_total - idx - 1)
                print(f"  [{idx+1}/{n_total}] {elapsed:.0f}s elapsed, "
                      f"ETA {eta:.0f}s")

        wall_time = time.perf_counter() - t0
        if verbose:
            print(f"  Grid complete: {n_total} points in {wall_time:.1f}s "
                  f"({wall_time/n_total:.3f}s/point)")

        return cls(config, grids_1d, data)


# ═══════════════════════════════════════════════════════════════
# §3. Likelihood wrapper
# ═══════════════════════════════════════════════════════════════

@dataclass
class BBNLikelihood:
    """BBN log-likelihood for parameter estimation.

    Scores canonical direct-solver predictions only.  ``GridEmulator`` remains
    accepted as an explicit argument solely to fail closed before interpolation.

    Parameters
    ----------
    emulator : GridEmulator, optional
        Diagnostic surrogate backend.  Passing it to ``log_likelihood`` raises
        ``ValueError`` before any observable is interpolated or scored.
    solver_fn : callable or ForwardModel-like object, optional
        Direct prediction backend. Supported forms match ``_resolve_solver_adapter``:
          - solver_fn(params_dict) -> dict
          - solver_fn(**params) -> BBNPrediction|dict
          - ForwardModel-like object with ``predict(**params)`` and optional
            ``prewarm()``
          - bound ``predict`` method from a ForwardModel-like object
    auto_prewarm_on_first_loglike : bool
        If True, execute any available one-shot prewarm contract before the first
        direct-solver likelihood evaluation.
    """
    # Observations (loaded from rabbit.data registry; foundation F12).
    # An older inline DH sigma in this file was off by 4e-7 relative to
    # Cooke+ 2018; sourcing from observations.json eliminates the drift.
    Yp_obs: float = _OBS_REGISTRY["Yp"].value
    Yp_err: float = _OBS_REGISTRY["Yp"].sigma
    DH_obs: float = _OBS_REGISTRY["DH"].value
    DH_err: float = _OBS_REGISTRY["DH"].sigma
    use_Li7: bool = False         # Li7 has cosmological lithium problem
    Li7_obs: float = _OBS_REGISTRY["Li7H"].value
    Li7_err: float = _OBS_REGISTRY["Li7H"].sigma
    observations: Optional[Sequence] = None

    # Solver backend
    emulator: Optional[GridEmulator] = None
    solver_fn: Optional[Callable] = None
    auto_prewarm_on_first_loglike: bool = False
    _solver_call: Optional[Callable] = field(default=None, init=False, repr=False)
    _solver_prewarm: Optional[Callable] = field(default=None, init=False, repr=False)
    _prewarmed: bool = field(default=False, init=False, repr=False)
    _last_prewarm_summary: Optional[dict] = field(default=None, init=False, repr=False)

    def __post_init__(self):
        if self.solver_fn is not None:
            self._solver_call, self._solver_prewarm = _resolve_solver_adapter(self.solver_fn)
        else:
            self._solver_call, self._solver_prewarm = None, None

    def prewarm(self):
        """Execute any optional one-shot direct-solver warmup hook."""
        if self._prewarmed:
            return self._last_prewarm_summary
        if callable(self._solver_prewarm):
            self._last_prewarm_summary = self._solver_prewarm()
        else:
            self._last_prewarm_summary = None
        self._prewarmed = True
        return self._last_prewarm_summary

    def log_likelihood(self, params: Dict[str, float]) -> float:
        """Evaluate BBN log-likelihood at given parameters."""
        if self.auto_prewarm_on_first_loglike and not self._prewarmed:
            self.prewarm()

        if self.emulator is not None:
            raise ValueError(
                "Inference received a SURROGATE GridEmulator; use a canonical "
                "direct solver for likelihood or posterior evaluation."
            )
        elif self._solver_call is not None:
            pred = self._solver_call(params)
        elif self.solver_fn is not None:
            pred = _prediction_to_observable_dict(self.solver_fn(params))
        else:
            raise ValueError("No emulator or solver_fn provided")

        required_observables = (
            [obs.name for obs in self.observations]
            if self.observations is not None
            else ["Yp", "DH"]
        )
        extra_prediction_fields = (
            ("Li7H",)
            if self.observations is None and self.use_Li7
            else ()
        )
        if not validate_prediction_for_inference(
            pred,
            required_observables,
            extra_prediction_fields=extra_prediction_fields,
        ):
            return -np.inf

        if self.observations is not None:
            ll = 0.0
            for obs in self.observations:
                ll += obs.log_likelihood(mapping_value_for_observation(pred, obs))
            return float(ll)

        ll = 0.0
        ll -= 0.5 * ((pred['Yp'] - self.Yp_obs) / self.Yp_err) ** 2
        ll -= 0.5 * ((pred['DH'] - self.DH_obs) / self.DH_err) ** 2
        if self.use_Li7:
            ll -= 0.5 * ((pred['Li7H'] - self.Li7_obs) / self.Li7_err) ** 2
        return float(ll)


# ═══════════════════════════════════════════════════════════════
# §4. Dynesty runner
# ═══════════════════════════════════════════════════════════════

def run_dynesty(
    log_likelihood_fn: Callable,
    prior_transform_fn: Callable,
    config: DynestyConfig,
    verbose: bool = True,
    prewarm_likelihood: bool = False,
    prewarm_fn: Optional[Callable] = None,
) -> dict:
    """Run dynesty nested sampling with PolyChord-equivalent tuning.

    Parameters
    ----------
    log_likelihood_fn : callable(theta) → float
        Log-likelihood function.
    prior_transform_fn : callable(u) → theta
        Transform from [0,1]^D to physical parameters.
    config : DynestyConfig
        Sampling configuration.

    Returns
    -------
    dict with keys: 'samples', 'weights', 'logz', 'logzerr', 'ncall', 'wall_time'
    """
    try:
        import dynesty
    except ImportError:
        raise ImportError(
            "dynesty required. Install: pip install dynesty\n"
            "Or: pip install rabbit-bbn[inference]"
        )

    resolved_prewarm = prewarm_fn
    if resolved_prewarm is None and prewarm_likelihood:
        resolved_prewarm = _resolve_optional_prewarm(log_likelihood_fn)

    prewarm_summary = None
    prewarm_wall_time = 0.0
    if callable(resolved_prewarm):
        t_prewarm = time.perf_counter()
        prewarm_summary = resolved_prewarm()
        prewarm_wall_time = time.perf_counter() - t_prewarm

    rng = np.random.default_rng(config.seed)

    t0 = time.perf_counter()

    if config.dynamic:
        sampler = dynesty.DynamicNestedSampler(
            log_likelihood_fn, prior_transform_fn,
            ndim=config.n_dim,
            nlive=config.nlive,
            sample=config.sample,
            slices=config.slices,
            walks=config.walks,
            rstate=rng,
        )
        sampler.run_nested(
            dlogz_init=config.dlogz,
            maxcall=config.maxcall,
            wt_kwargs={'pfrac': config.pfrac},
            print_progress=verbose,
        )
    else:
        sampler = dynesty.NestedSampler(
            log_likelihood_fn, prior_transform_fn,
            ndim=config.n_dim,
            nlive=config.nlive,
            sample=config.sample,
            slices=config.slices,
            walks=config.walks,
            rstate=rng,
        )
        sampler.run_nested(
            dlogz=config.dlogz,
            maxcall=config.maxcall,
            print_progress=verbose,
        )

    wall_time = time.perf_counter() - t0
    results = sampler.results

    # Extract weighted posterior samples
    from dynesty.utils import resample_equal
    samples = resample_equal(results.samples, np.exp(results.logwt - results.logz[-1]))

    return {
        'samples': samples,
        'weights': np.exp(results.logwt - results.logz[-1]),
        'logz': results.logz[-1],
        'logzerr': results.logzerr[-1],
        'ncall': results.ncall,
        'niter': results.niter,
        'wall_time': wall_time,
        'prewarm_applied': callable(resolved_prewarm),
        'prewarm_wall_time': prewarm_wall_time,
        'prewarm_summary': prewarm_summary,
        'raw_results': results,
    }



def run_canonical_dynesty(
    prior_transform_fn: Callable,
    config: DynestyConfig,
    param_names: Sequence[str],
    verbose: bool = True,
    prewarm_likelihood: Optional[bool] = None,
    **solver_kwargs,
) -> dict:
    """Run dynesty on the canonical direct-solver likelihood.

    This wrapper builds the canonical direct likelihood, adapts it to the
    vector signature expected by dynesty, and preserves any available prewarm
    lifecycle through the adapter.
    """
    like = make_canonical_direct_likelihood(
        auto_prewarm_on_first_loglike=False,
        **solver_kwargs,
    )
    loglike = make_vector_loglike_adapter(like, param_names)
    if prewarm_likelihood is None:
        prewarm_likelihood = bool(
            solver_kwargs.get('backend', 'auto') == 'jax'
            and solver_kwargs.get('prewarm_jax', False)
        )
    return run_dynesty(
        loglike,
        prior_transform_fn,
        config,
        verbose=verbose,
        prewarm_likelihood=bool(prewarm_likelihood),
    )
