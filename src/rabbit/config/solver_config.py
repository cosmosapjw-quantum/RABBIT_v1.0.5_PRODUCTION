"""
rabbit.config.solver_config — ODE solver configuration (SciPy backends).

Rust AOT is the active implementation and repeated-run design target, while
SciPy/BDF remains the temporary number-of-record during migration.  The
JAX/NumPy Rodas5P twins are frozen numerical-method references only; they do
not select future runtime development.  This module provides the SciPy backend
configurations used for regression-locked reference solves.

BDF has been the *effective* production method on the ``full_coupled_typeI``
SciPy lane since the initial commit: a hidden identity-check remap silently
substituted BDF for the declared Radau method whenever the default config was
used, with zero recorded rationale (BD613). That remap has been removed;
``PRODUCTION_CONFIG`` below now declares BDF directly so the requested method
matches what actually runs (option a: honest declaration, zero runtime
change). Radau↔BDF endpoint parity was measured at |ΔY_p|=8.25e-8, D/H rel
9.87e-6 (BD598 audit), which is why this was a safe declaration change rather
than a physics change.

Radau IIA remains the validation oracle: ``REFERENCE_CONFIG`` (tight
tolerances) and ``PRODUCTION_RADAU_CONFIG`` (production tolerances, Radau
method) are both available for cross-checking and are what ``classA_driver``
uses by default, preserving its historical Radau-at-production-tolerances
behavior verbatim. Any future change to the production *method* must go
through the drift guard in ``full_coupled_typeI._resolve_effective_solver``
plus an explicit promotion gate — it must never again happen implicitly.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum


class SolverMethod(Enum):
    """Available ODE solver methods."""
    RADAU = "Radau"         # Radau IIA (implicit RK, order 5)      — SciPy
    BDF = "BDF"             # Backward differentiation formula       — SciPy
    LSODA = "LSODA"         # Automatic stiff/non-stiff switching    — SciPy
    RODAS5P = "RODAS5P"     # In-tree Rosenbrock-W (rabbit.solver.rodas5p),
                            # NOT a scipy.solve_ivp method — routed through
                            # rabbit.solver.rodas5p_adapter (BD615).


#: Methods that map directly onto ``scipy.integrate.solve_ivp``.
_SCIPY_METHODS = frozenset({SolverMethod.RADAU, SolverMethod.BDF, SolverMethod.LSODA})


@dataclass(frozen=True)
class SolverConfig:
    """Configuration for the ODE integrator.

    Parameters
    ----------
    method : SolverMethod
        Integration method.  Radau is the default and the most robust
        for the stiff BBN system.
    rtol : float
        Relative tolerance.  1e-8 is the reference standard.
    atol : float
        Absolute tolerance.  1e-10 ensures that near-zero abundances
        do not stall the integrator.
    max_step : float
        Maximum step size in the independent variable N = ln a.
        Default 0.1 prevents the solver from skipping important
        physics during the weak freeze-out epoch.
    first_step : float or None
        Initial step size hint.  None lets the solver auto-detect.
    dense_output : bool
        Whether to request dense (interpolated) output from the solver.
    """
    method: SolverMethod = SolverMethod.RADAU
    rtol: float = 1e-8
    atol: float = 1e-10
    max_step: float = 0.1
    first_step: float | None = None
    dense_output: bool = False
    #: RODAS5P lane only (BD617): how many consecutive accepted steps may reuse a
    #: single Jacobian before a forced recompute. 1 = exact Rosenbrock (recompute
    #: every step). >1 amortizes the dense FD Jacobian (N+1 RHS evals/step, the
    #: dominant RODAS5P cost) but is an approximation — must clear the endpoint
    #: parity + wall gate before a preset enables it. Ignored by scipy methods.
    jac_reuse_max_steps: int = 1

    @property
    def is_scipy(self) -> bool:
        """True iff ``method`` maps onto a ``scipy.integrate.solve_ivp`` method."""
        return self.method in _SCIPY_METHODS

    @property
    def scipy_method_str(self) -> str:
        """Method name string for ``scipy.integrate.solve_ivp``."""
        return self.method.value

    def to_scipy_kwargs(self) -> dict:
        """Keyword arguments for ``scipy.integrate.solve_ivp``.

        Raises
        ------
        ValueError
            If ``method`` is not a SciPy method (e.g. ``RODAS5P``). This is a
            fail-loud guard (BD615): a non-scipy method must never leak into a
            ``solve_ivp`` call — the driver routes it through the Rodas5P
            adapter instead.
        """
        if not self.is_scipy:
            raise ValueError(
                f"to_scipy_kwargs() called on non-scipy method {self.method.value!r}; "
                "route this config through rabbit.solver.rodas5p_adapter, not solve_ivp"
            )
        kw = {
            'method': self.scipy_method_str,
            'rtol': self.rtol,
            'atol': self.atol,
            'max_step': self.max_step,
            'dense_output': self.dense_output,
        }
        if self.first_step is not None:
            kw['first_step'] = self.first_step
        return kw


# ═══════════════════════════════════════════════════════════════════════
# Default configurations
# ═══════════════════════════════════════════════════════════════════════

#: SciPy validation oracle: tight tolerances for cross-checking JAX results.
REFERENCE_CONFIG = SolverConfig(
    method=SolverMethod.RADAU,
    rtol=1e-10,
    atol=1e-12,
    max_step=0.05,
)

#: SciPy interim driver config: balanced speed vs accuracy.
#: Used by full_coupled_typeI (SciPy path) until JAX driver is end-to-end.
#: Method is BDF (BD613): this has been the effective production method since
#: the initial commit (previously applied via a hidden identity remap in
#: full_coupled_typeI, now removed); see module docstring for the parity
#: citation and drift-guard design. Radau at the same tolerances is available
#: as ``PRODUCTION_RADAU_CONFIG``.
PRODUCTION_CONFIG = SolverConfig(
    method=SolverMethod.BDF,
    rtol=1e-8,
    atol=1e-10,
    max_step=0.1,
)

#: SciPy production tolerances with the Radau IIA method (BD613). This is the
#: validation-oracle counterpart to ``PRODUCTION_CONFIG`` and is what
#: ``classA_driver`` defaults to, preserving its historical Radau behavior.
PRODUCTION_RADAU_CONFIG = SolverConfig(
    method=SolverMethod.RADAU,
    rtol=1e-8,
    atol=1e-10,
    max_step=0.1,
)

#: SciPy fast config for smoke tests and parameter sweeps.
FAST_CONFIG = SolverConfig(
    method=SolverMethod.RADAU,
    rtol=1e-6,
    atol=1e-8,
    max_step=0.5,
)
