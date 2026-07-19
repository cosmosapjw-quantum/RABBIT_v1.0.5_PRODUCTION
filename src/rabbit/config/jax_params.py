"""
rabbit.config.jax_params — JAX-compatible parameter pytrees.

All physical parameters that may be differentiated flow through function
arguments as frozen pytree containers — never through module-level globals.
This is the fundamental JAX contract: JIT traces through function args.

Two parameter pytrees:

    BBNParams     Physical parameters (η, τ_n, Σ_H, ...) with a clean split
                  between differentiable leaves (float fields traced by AD)
                  and static aux_data (int fields that trigger recompilation).

    SolverParams  Solver tolerances and step limits. Always static (no AD
                  through solver control parameters).

Registration with jax.tree_util ensures these work with jit, grad, vmap,
and all other JAX transformations.

Usage
-----
    from rabbit.config.jax_params import BBNParams, SolverParams

    params = BBNParams(eta=6.104e-10, tau_n=878.4, Sigma_H=0.1)
    solver = SolverParams(rtol=1e-8)

    # JIT-safe:
    @jax.jit
    def forward(p):
        return p.eta * 2

    # AD-safe:
    jax.grad(lambda p: p.eta)(params)  # works

    # vmap-safe:
    batch = BBNParams(eta=jnp.array([6.0e-10, 6.2e-10]), ...)
    jax.vmap(forward)(batch)

References
----------
    JAX pytree docs: https://jax.readthedocs.io/en/latest/pytrees.html
    RABBIT Paper I §VII: Born-level Σ_H constraint
"""
from __future__ import annotations

from typing import Tuple, Any

# Lazy JAX import: this module can be imported even without JAX installed.
# The actual registration happens only when JAX is available.
_JAX_AVAILABLE = False
try:
    import jax
    import jax.numpy as jnp
    _JAX_AVAILABLE = True
except ImportError:
    pass


# ═══════════════════════════════════════════════════════════════════════
# §1. BBNParams — physical parameter pytree
# ═══════════════════════════════════════════════════════════════════════

# Fields that AD can differentiate through (float leaves of the pytree).
_DIFFERENTIABLE_FIELDS: Tuple[str, ...] = (
    "eta",          # baryon-to-photon ratio
    "tau_n",        # neutron lifetime [s]
    "Sigma_H",      # initial Hubble-normalized shear (|Σ| at N_init)
    "N_eff",        # effective neutrino number
)

# Fields that are static (trigger JIT recompilation on change, not traced).
_STATIC_FIELDS: Tuple[str, ...] = (
    "correction_level",  # 0=Born, 1=+Coulomb, 2=+Sirlin, 3=+FM/WM
    "N_q",               # neutrino momentum grid points
    "ell_max",           # transport truncation order
    "tier",              # thermo tier (1=entropy, 2=3-temperature)
    "enable_teff",       # deprecated legacy switch; kept for config compatibility
)


class BBNParams:
    """Frozen physical parameter container registered as a JAX pytree.

    Differentiable fields (eta, tau_n, Sigma_H, N_eff) are pytree leaves
    that JAX traces through. Static fields (correction_level, N_q, ...)
    are carried as aux_data and trigger recompilation when changed.

    Parameters
    ----------
    eta : float
        Baryon-to-photon ratio. Default: 6.104e-10 (Planck 2018 + PDG).
    tau_n : float
        Neutron lifetime [s]. Default: 878.4 (PDG 2024).
    Sigma_H : float
        Initial Hubble-normalized shear |Σ| at high-T initialization.
        Default: 0.0 (FLRW).
    N_eff : float
        Effective neutrino number. Default: 3.044.
    correction_level : int
        Weak rate correction level. 0=Born, 1=+Coulomb, 2=+Sirlin,
        3=+FM/WM. Default: 0.
    N_q : int
        Number of Gauss-Laguerre momentum grid points. Default: 6
        (fast; production uses 80).
    ell_max : int
        Transport hierarchy truncation. Default: 2 (exact for Type I).
    tier : int
        Thermodynamic tier. 1=entropy conservation, 2=3-temperature ODE.
        Default: 1.
    enable_teff : bool
        Deprecated legacy switch retained for config compatibility. Default: False.
    """
    __slots__ = tuple(_DIFFERENTIABLE_FIELDS) + tuple(_STATIC_FIELDS)

    def __init__(
        self,
        eta: float = 6.104e-10,
        tau_n: float = 878.4,
        Sigma_H: float = 0.0,
        N_eff: float = 3.044,
        correction_level: int = 0,
        N_q: int = 6,
        ell_max: int = 2,
        tier: int = 1,
        enable_teff: bool = False,
    ):
        object.__setattr__(self, "eta", eta)
        object.__setattr__(self, "tau_n", tau_n)
        object.__setattr__(self, "Sigma_H", Sigma_H)
        object.__setattr__(self, "N_eff", N_eff)
        object.__setattr__(self, "correction_level", int(correction_level))
        object.__setattr__(self, "N_q", int(N_q))
        object.__setattr__(self, "ell_max", int(ell_max))
        object.__setattr__(self, "tier", int(tier))
        object.__setattr__(self, "enable_teff", bool(enable_teff))

    def __setattr__(self, name, value):
        raise AttributeError(
            f"BBNParams is frozen. Cannot set '{name}'. "
            "Use replace() to create a modified copy."
        )

    def __delattr__(self, name):
        raise AttributeError("BBNParams is frozen.")

    def replace(self, **kwargs) -> "BBNParams":
        """Return a new BBNParams with specified fields replaced."""
        current = {f: getattr(self, f) for f in self.__slots__}
        current.update(kwargs)
        return BBNParams(**current)

    def __repr__(self) -> str:
        parts = [f"{f}={getattr(self, f)!r}" for f in self.__slots__]
        return f"BBNParams({', '.join(parts)})"

    def __eq__(self, other):
        if not isinstance(other, BBNParams):
            return NotImplemented
        return all(
            getattr(self, f) == getattr(other, f) for f in self.__slots__
        )

    def __hash__(self):
        return hash(tuple(getattr(self, f) for f in self.__slots__))

    # ── JAX pytree protocol ──────────────────────────────────────────

    def tree_flatten(self):
        """Split into (differentiable_leaves, static_aux_data).

        Differentiable leaves: (eta, tau_n, Sigma_H, N_eff) — traced by AD.
        Static aux_data: (correction_level, N_q, ...) — trigger recompilation.
        """
        children = tuple(getattr(self, f) for f in _DIFFERENTIABLE_FIELDS)
        aux_data = tuple(getattr(self, f) for f in _STATIC_FIELDS)
        return children, aux_data

    @classmethod
    def tree_unflatten(cls, aux_data: tuple, children: tuple) -> "BBNParams":
        """Reconstruct from (differentiable_leaves, static_aux_data)."""
        kw = {}
        for f, v in zip(_DIFFERENTIABLE_FIELDS, children):
            kw[f] = v
        for f, v in zip(_STATIC_FIELDS, aux_data):
            kw[f] = v
        return cls(**kw)

    # ── Convenience ──────────────────────────────────────────────────

    @property
    def f_nu(self) -> float:
        """Neutrino energy fraction f_ν = ρ_ν/(ρ_γ + ρ_ν).

        Computed from N_eff: f_ν = (7/8)(4/11)^{4/3} × N_eff /
                                    (1 + (7/8)(4/11)^{4/3} × N_eff)

        Note: this gives f_ν ≈ 0.4087 for N_eff = 3.044, which differs
        from the driver constant _F_NU = 0.40520 (bare 3-species value
        without the 3.044/3 correction). The JAX RHS (J02) must use
        this property consistently rather than the bare constant.
        """
        x = (7.0 / 8.0) * (4.0 / 11.0) ** (4.0 / 3.0) * self.N_eff
        return x / (1.0 + x)

    @property
    def is_flrw(self) -> bool:
        """True if this is an isotropic (FLRW) parameter set."""
        s = self.Sigma_H
        # Handle both float and JAX array
        if _JAX_AVAILABLE and hasattr(s, 'item'):
            s = float(s)
        return abs(s) < 1e-15

    @property
    def differentiable_field_names(self) -> Tuple[str, ...]:
        return _DIFFERENTIABLE_FIELDS

    @property
    def static_field_names(self) -> Tuple[str, ...]:
        return _STATIC_FIELDS


# ═══════════════════════════════════════════════════════════════════════
# §2. SolverParams — solver control pytree (fully static)
# ═══════════════════════════════════════════════════════════════════════

_SOLVER_FIELDS: Tuple[str, ...] = (
    "rtol", "atol", "max_steps", "max_step_size", "min_step_size",
)


class SolverParams:
    """Solver configuration for JAX-native integrators.

    All fields are static (not differentiated). Changes trigger
    JIT recompilation, which is the correct behavior — you don't
    differentiate through solver tolerances.

    Parameters
    ----------
    rtol : float
        Relative tolerance. Default: 1e-8.
    atol : float
        Absolute tolerance. Default: 1e-10.
    max_steps : int
        Maximum number of adaptive steps. Default: 2000.
    max_step_size : float
        Maximum step size in N = ln a. Default: 0.1.
    min_step_size : float
        Minimum step size (solver failure below this). Default: 1e-14.
    """
    __slots__ = _SOLVER_FIELDS

    def __init__(
        self,
        rtol: float = 1e-8,
        atol: float = 1e-10,
        max_steps: int = 2000,
        max_step_size: float = 0.1,
        min_step_size: float = 1e-14,
    ):
        object.__setattr__(self, "rtol", float(rtol))
        object.__setattr__(self, "atol", float(atol))
        object.__setattr__(self, "max_steps", int(max_steps))
        object.__setattr__(self, "max_step_size", float(max_step_size))
        object.__setattr__(self, "min_step_size", float(min_step_size))

    def __setattr__(self, name, value):
        raise AttributeError("SolverParams is frozen.")

    def replace(self, **kwargs) -> "SolverParams":
        current = {f: getattr(self, f) for f in self.__slots__}
        current.update(kwargs)
        return SolverParams(**current)

    def __repr__(self) -> str:
        parts = [f"{f}={getattr(self, f)!r}" for f in self.__slots__]
        return f"SolverParams({', '.join(parts)})"

    def __eq__(self, other):
        if not isinstance(other, SolverParams):
            return NotImplemented
        return all(
            getattr(self, f) == getattr(other, f) for f in self.__slots__
        )

    def __hash__(self):
        return hash(tuple(getattr(self, f) for f in self.__slots__))

    # ── JAX pytree protocol (all static — no differentiable leaves) ──

    def tree_flatten(self):
        """All fields are static aux_data (no differentiable leaves)."""
        children = ()  # no leaves
        aux_data = tuple(getattr(self, f) for f in _SOLVER_FIELDS)
        return children, aux_data

    @classmethod
    def tree_unflatten(cls, aux_data: tuple, children: tuple) -> "SolverParams":
        kw = dict(zip(_SOLVER_FIELDS, aux_data))
        return cls(**kw)


# ═══════════════════════════════════════════════════════════════════════
# §3. JAX pytree registration (conditional on JAX availability)
# ═══════════════════════════════════════════════════════════════════════

def _register_pytrees():
    """Register BBNParams and SolverParams as JAX pytrees.

    Called once at import time if JAX is available. Safe to call
    multiple times (idempotent via try/except).
    """
    if not _JAX_AVAILABLE:
        return

    try:
        jax.tree_util.register_pytree_node(
            BBNParams,
            lambda p: p.tree_flatten(),
            lambda aux, children: BBNParams.tree_unflatten(aux, children),
        )
        jax.tree_util.register_pytree_node(
            SolverParams,
            lambda p: p.tree_flatten(),
            lambda aux, children: SolverParams.tree_unflatten(aux, children),
        )
    except ValueError:
        # Already registered (idempotent)
        pass


# Perform registration at import time
_register_pytrees()


# ═══════════════════════════════════════════════════════════════════════
# §4. Defaults
# ═══════════════════════════════════════════════════════════════════════

#: FLRW baseline at Born level, N_q=6 (fast).
DEFAULT_BBN_PARAMS = BBNParams()

#: Canonical production tolerances for BBN integration.
DEFAULT_SOLVER_PARAMS = SolverParams()

#: Fast settings for smoke tests.
FAST_SOLVER_PARAMS = SolverParams(rtol=1e-6, atol=1e-8, max_steps=1000)

#: Tight settings for reference validation.
REFERENCE_SOLVER_PARAMS = SolverParams(rtol=1e-10, atol=1e-12, max_steps=5000)
