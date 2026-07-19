"""rabbit.external.nudec_bsm — NUDEC_BSM live wrapper (Phase ζ / ζ-2).

Physics meaning
---------------
NUDEC_BSM is a Mathematica + Python package for precision neutrino
decoupling in the SM and beyond (Escudero+ 2019, arXiv:1812.05605).
Returns N_eff ≈ 3.0446 at the SM benchmark — within 6×10⁻⁴ of the
Mangano 2005 SM value of 3.044, well within the published 5×10⁻⁴
numerical error budget.

Availability
------------
NUDEC_BSM is not on PyPI. The Python entry point ships as a git
clone of https://github.com/MiguelEA/nuDec_BSM. To activate the live
test, clone that repo and set the env var
``RABBIT_NUDEC_BSM_PATH=/path/to/nuDec_BSM`` so this wrapper can
prepend it to sys.path before importing.

Phase ζ-2 (this revision) replaces the earlier
``solve_decoupling``-based stub with the actual ``nudec_v2.NuDec``
API: instantiate, call ``evolve()``, then ``Neff(...)`` with the
final-time temperatures.

Static-fixture fallback
-----------------------
The fixture ``tests/fixtures/tier3_cross_code.json`` carries the
reference values used by the always-on test. Live runs additionally
verify drift via ``tests/test_cross_code_drift.py``.
"""

from __future__ import annotations

import importlib.util
import os
import sys
from dataclasses import dataclass
from typing import Optional

from rabbit.external import ExternalCodeUnavailable


@dataclass(frozen=True)
class NudecBsmResult:
    """Subset of NUDEC_BSM output that we cross-check."""
    N_eff: float
    T_gamma_final_MeV: Optional[float] = None
    T_nu_e_final_MeV: Optional[float] = None
    T_nu_mu_final_MeV: Optional[float] = None
    raw_metadata: Optional[dict] = None


def _ensure_nudec_path() -> Optional[str]:
    """Add the configured NUDEC_BSM source directory to sys.path.

    Returns the resolved path on success, or None if no installation
    is reachable. The probe order is:

      1. ``RABBIT_NUDEC_BSM_PATH`` env var (preferred).
      2. ``BasicModules_source.nudec_v2`` already importable
         (e.g. via PYTHONPATH or pip install -e).
    """
    candidate = os.environ.get("RABBIT_NUDEC_BSM_PATH")
    if candidate:
        candidate = os.path.expanduser(candidate)
        if os.path.isdir(candidate):
            if candidate not in sys.path:
                sys.path.insert(0, candidate)
            return candidate
    # No env var; check whether the module is already importable.
    # find_spec raises ModuleNotFoundError (not returns None) when the
    # *parent* package is absent, so treat any import probe error as
    # "backend unreachable" to keep this a pure availability check.
    try:
        if importlib.util.find_spec("BasicModules_source.nudec_v2") is not None:
            return ""
    except (ImportError, ValueError):
        return None
    return None


def has_nudec_bsm() -> bool:
    """Return True iff the NUDEC_BSM Python entry-point is reachable."""
    return _ensure_nudec_path() is not None


def run_nudec_bsm(
    eta: float = 6.104e-10,
    tau_n: float = 878.4,
    n_eff_target: float = 3.044,
    n_extra_neutrinos: float = 0.0,
) -> NudecBsmResult:
    """Run NUDEC_BSM at the requested parameters and return the result.

    Parameters
    ----------
    eta : float
        Absolute baryon-to-photon ratio. **Currently ignored** by
        NUDEC_BSM v2 — that code computes N_eff at the SM benchmark
        independently of η. The argument is retained for API
        compatibility with future BSM extensions.
    tau_n : float
        Neutron lifetime (s). **Currently ignored** — NUDEC_BSM
        focuses on N_eff via T_γ / T_ν decoupling, not weak rates.
    n_eff_target, n_extra_neutrinos : reserved for future BSM use.

    Returns
    -------
    NudecBsmResult with N_eff and the final-time temperatures.

    Raises
    ------
    ExternalCodeUnavailable
        If NUDEC_BSM is not reachable. Set
        ``RABBIT_NUDEC_BSM_PATH=/path/to/nuDec_BSM`` and try again.
    """
    if _ensure_nudec_path() is None:
        raise ExternalCodeUnavailable(
            "nudec_bsm",
            reason=(
                "NUDEC_BSM Python module not reachable. Clone "
                "https://github.com/MiguelEA/nuDec_BSM and set "
                "RABBIT_NUDEC_BSM_PATH=/path/to/nuDec_BSM, OR ensure "
                "BasicModules_source.nudec_v2 is on PYTHONPATH."
            ),
        )

    import BasicModules_source.nudec_v2 as nudec_v2  # type: ignore[import-not-found]

    nudec = nudec_v2.NuDec()
    t, y = nudec.evolve()
    # y rows: [T_gam, T_nue, T_numu, mu_nue, mu_numu, z]
    T_gam = float(y[0][-1])
    T_nue = float(y[1][-1])
    T_numu = float(y[2][-1])
    mu_e = float(y[3][-1])
    mu_m = float(y[4][-1])
    n_eff = float(nudec.Neff(T_gam, T_nue, T_numu, mu_e, mu_m))

    return NudecBsmResult(
        N_eff=n_eff,
        T_gamma_final_MeV=T_gam,
        T_nu_e_final_MeV=T_nue,
        T_nu_mu_final_MeV=T_numu,
        raw_metadata={
            "t_final_s": float(t[-1]),
            "T_gamma_MeV": T_gam,
            "T_nu_e_MeV": T_nue,
            "T_nu_mu_MeV": T_numu,
            "mu_nu_e_MeV": mu_e,
            "mu_nu_mu_MeV": mu_m,
            "eta_argument_ignored": float(eta),
            "tau_n_argument_ignored": float(tau_n),
        },
    )
