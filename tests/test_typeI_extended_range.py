"""tests/test_typeI_extended_range.py — INTERNAL extended-range sweep nodes for GATE_SIGMA_H_TO_0P95.

HONEST SCOPE (external audit 2026-06-30). These are the INTERNAL half of the claim gate
``GATE_SIGMA_H_TO_0P95`` ("Type I anisotropic BBN validated for Sigma_H in [0, 0.95]"). They are
SELF-CONSISTENCY + regression checks, NOT external validation:

  * The Friedmann residual ``|1 - Omega - Sigma^2|`` is **identically zero by construction** -- Omega
    is DEFINED as ``1 - Sigma^2`` (``rabbit.geometry.constraints.friedmann_residual_typeI``), so a
    "constraint residual < 1e-10" assertion on it would be vacuous (``assert 0 < 1e-10``). The genuine
    internal numerical check is the EVOLVED mass-fraction closure ``|sum X_i - 1|``
    (``constraint_diag.mass_conservation``), which a real integrator drift WOULD break.
  * Above Sigma_H = 0.75 the solver itself warns ("exceeds safe maximum ... Omega -> 0 Milne regime
    ... interpret with caution"); Yp rises steeply there (~0.256 at 0.75 -> ~0.331 at 0.95). Those
    high-shear values are regression-locked but are explicitly in the CAUTION regime, NOT validated.

None of these VALIDATE Sigma_H = 0.95 against nature. The gate's external_anchor node
(``tests/test_finite_shear_external_anchor.py``, fail-closed) carries external validation, so
``GATE_SIGMA_H_TO_0P95`` stays RED until an independent finite-shear cross-code benchmark exists --
even with these two internal nodes green. Writing them only makes the internal evidence real
(necessary, not sufficient).

Marked ``slow`` -- the sweep is six full-BBN coupled solves.
"""

from __future__ import annotations

import warnings
from functools import lru_cache

import numpy as np
import pytest

pytestmark = pytest.mark.slow

_N_Q, _N_MU, _N_RX, _CL = 20, 12, 12, 0
_SIGMAS = (0.0, 0.3, 0.6, 0.75, 0.85, 0.95)
_SAFE_MAX = 0.75  # mirrors full_coupled_typeI._SIGMA_SAFE_MAX (advisory warning boundary)

# Measured CL0 Yp at (N_q=20, N_mu=12, n_reactions=12). A SELF-CONSISTENCY regression lock (catches
# solver/toolchain drift), NOT an external-validation reference. Tolerance ~1e-5: the shear moves Yp
# by ~0.09 over the sweep, so this is a ~9000x signal/tolerance lock while tolerating BLAS drift.
# Restricted to the RELIABLE regime (Sigma_H <= 0.75, where the solver does not warn): the >0.75
# caution cells (Omega -> 0) are deliberately NOT hard-locked here -- the solver itself flags them
# as possibly unreliable, so locking a value there risks flaky cross-platform failures that would
# mask, not catch, drift. Those cells stay covered by monotonicity + finiteness + mass closure.
_YP_GOLD = {0.0: 0.24234943, 0.3: 0.24401979, 0.6: 0.24840852, 0.75: 0.25565792}
_YP_TOL = 1.0e-5


@lru_cache(maxsize=16)
def _run(sigma: float):
    """Cached coupled solve; returns (CanonicalResult, fired_safe_max_warning)."""
    from rabbit.drivers.full_coupled_typeI import FullCoupledConfig, run_full_coupled_typeI
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        res = run_full_coupled_typeI(FullCoupledConfig(
            Sigma_H_plus=float(sigma), correction_level=_CL, N_q=_N_Q, N_mu=_N_MU, n_reactions=_N_RX))
    warned = any("exceeds safe maximum" in str(w.message) for w in caught)
    return res, warned


class TestExtendedRangeSweep:
    def test_sigma_h_up_to_0p95_constraint_residual(self):
        """INTERNAL self-consistency (NOT validation): the solver completes to Sigma_H=0.95 with
        finite observables, the EVOLVED mass-fraction closure stays at machine precision, and the
        documented Sigma_H>0.75 caution warning fires above 0.75 and not below.

        The Friedmann residual is asserted to be the by-construction tautology it is (~0), explicitly
        to DOCUMENT that it is not what carries the constraint check -- the mass_conservation drift is.
        """
        from rabbit.geometry.constraints import friedmann_residual_typeI

        for s in _SIGMAS:
            res, warned = _run(s)
            assert np.isfinite(res.observables.Yp), f"non-finite Yp at Sigma_H={s}"
            assert np.isfinite(res.observables.DH), f"non-finite D/H at Sigma_H={s}"

            mc = res.constraint_diag.mass_conservation
            assert len(mc) > 0, f"no mass_conservation diagnostic recorded at Sigma_H={s}"
            mc_max = float(np.max(np.abs(mc)))
            # measured peak across the sweep is ~4.6e-14; 1e-12 is a ~20x-margin lock that still
            # catches incremental integrator degradation, not just catastrophic failure.
            assert mc_max < 1e-12, (
                f"evolved mass-fraction closure |sum X_i - 1| drifted at Sigma_H={s}: {mc_max:.2e}")

            # The advisory safe-boundary warning fires strictly ABOVE 0.75 (Omega = 1 - Sigma^2 < 0.44).
            assert warned == (s > _SAFE_MAX), (
                f"Sigma_H>{_SAFE_MAX} caution warning mismatch at Sigma_H={s}: warned={warned}")

            # The Friedmann residual is identically 0 by construction (Omega := 1 - Sigma^2); pinning
            # it documents the tautology so this node is never misread as carrying the validation.
            assert friedmann_residual_typeI(s, 0.0) < 1e-12

    def test_sigma_h_up_to_0p95_yp_regression(self):
        """INTERNAL regression lock (NOT external validation): Yp is monotone increasing in Sigma_H
        (more shear -> faster expansion -> more He-4) and matches the measured CL0 values. Above
        Sigma_H=0.75 (the Omega->0 caution regime) Yp rises steeply; those values are regression-
        locked but NOT validated -- the gate's external_anchor node carries that."""
        yps = [_run(s)[0].observables.Yp for s in _SIGMAS]

        for a, b, sa, sb in zip(yps, yps[1:], _SIGMAS, _SIGMAS[1:]):
            assert a < b, f"Yp not monotone in shear: Yp({sa})={a:.8f} >= Yp({sb})={b:.8f}"

        # Non-vacuity: the shear genuinely moves Yp by a large amount over the sweep.
        assert (yps[-1] - yps[0]) > 0.05, f"shear did not move Yp enough: {yps[-1] - yps[0]:.3e}"

        # Regression lock vs the measured CL0 values (self-consistency, catches drift).
        for s, gold in _YP_GOLD.items():
            yp = _run(s)[0].observables.Yp
            assert abs(yp - gold) < _YP_TOL, (
                f"Yp regression drift at Sigma_H={s}: {yp:.8f} vs locked {gold:.8f} "
                f"(|d|={abs(yp - gold):.2e} > {_YP_TOL:.0e})")
