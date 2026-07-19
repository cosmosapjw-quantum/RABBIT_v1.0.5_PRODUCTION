"""rabbit.transport.realizability_diagnostics — realizability diagnostics for the augmented PSTF.

The forward solve enforces neutrino-distribution realizability through the **band-limited logit**:
``f = 1 / (1 + exp(-(q + A)))`` is in ``(0, 1)`` for ANY finite logit coefficient ``A``, so
realizability is *unconditional* (`augmented_pstf_distribution.fermi_dirac_from_logit`). This module
provides two diagnostics that make that guarantee measurable and contrast it against the entropy
(maxent / Levermore M_N) closure it replaces:

- :func:`fd_boundary_margin` — the honest realizability metric ``min(f.min(), 1 - f.max())``. Unlike
  the (vacuously-true) ``0 <= f <= 1`` check, this records how *close* the reconstructed distribution
  comes to the Fermi-Dirac boundary at strong shear without ever crossing it.
- :func:`maxent_M2_multiplier` — the Lagrange multiplier of the maximum-entropy angular closure that
  the band-limited logit replaces. It **diverges** as the requested moment approaches the
  realizability boundary (and does not exist beyond it), which is exactly the strong-shear
  ill-posedness the logit avoids (`docs/audit/BHR_wellposedness_2026-06-30.md`). This callable makes
  the "logit replaces divergent M_N multipliers" claim a testable assertion rather than a comment.

The maxent closure here is NOT used in the forward solve; it is the diagnostic counterpart against
which the logit's unconditional realizability is demonstrated.
"""

from __future__ import annotations

import numpy as np
from scipy.optimize import brentq


class RealizabilityBoundaryError(ValueError):
    """Raised when a maxent multiplier does not exist / diverges at the realizability boundary."""


def fd_boundary_margin(f: np.ndarray) -> float:
    """Return ``min(f.min(), 1 - f.max())`` — the distance of ``f`` from the Fermi-Dirac boundary.

    A strictly positive margin means the distribution is realizable with room to spare; a margin
    shrinking toward 0 (but staying > 0) means ``f`` is approaching, but has not crossed, the
    ``[0, 1]`` boundary. This is the non-vacuous realizability metric for the band-limited logit
    (whose codomain is trivially ``(0, 1)``).
    """
    f_arr = np.asarray(f, dtype=float)
    if not np.all(np.isfinite(f_arr)):
        raise ValueError("f must be finite.")
    return float(min(float(f_arr.min()), 1.0 - float(f_arr.max())))


def maxent_M2_multiplier(pi_target: float, kernel: np.ndarray, angular_weights: np.ndarray,
                         *, lam_bracket: float = 600.0) -> float:
    """Maxent (Levermore M_N) Lagrange multiplier ``lambda`` for a single angular kernel mode.

    Solves, on the angular quadrature ``(kernel, angular_weights)``,

        <kernel * exp(lambda * kernel)> / <exp(lambda * kernel)> = pi_target,

    i.e. the multiplier of the maximum-entropy angular distribution ``g ~ exp(lambda * kernel)`` that
    reproduces the requested normalized moment ``pi_target``. The realizable interval is the open
    ``(kernel.min(), kernel.max())``: as ``pi_target`` approaches a boundary, ``|lambda| -> inf``.

    Raises :class:`RealizabilityBoundaryError` when ``pi_target`` is outside the realizable interval
    (no multiplier exists) or when the required ``|lambda|`` exceeds ``lam_bracket`` (the multiplier
    has diverged beyond the representable range). This divergence is the strong-shear ill-posedness
    that the band-limited logit -- which needs no such inversion -- avoids unconditionally.
    """
    k = np.asarray(kernel, dtype=float)
    w = np.asarray(angular_weights, dtype=float)
    if k.shape != w.shape:
        raise ValueError("kernel and angular_weights must have the same shape.")
    if not np.all(np.isfinite(k)) or not np.all(np.isfinite(w)) or np.any(w <= 0.0):
        raise ValueError("kernel finite and angular_weights finite + positive required.")
    k_min, k_max = float(k.min()), float(k.max())
    if not (k_min < float(pi_target) < k_max):
        raise RealizabilityBoundaryError(
            f"pi_target={pi_target} is outside the realizable interval ({k_min}, {k_max}); "
            "the maxent multiplier does not exist (the band-limited logit needs no inversion).")

    def _moment(lam: float) -> float:
        z = lam * k
        z = z - z.max()  # overflow-safe shift (cancels in the ratio)
        e = np.exp(z)
        return float(np.sum(w * k * e) / np.sum(w * e))

    g_lo = _moment(-lam_bracket) - float(pi_target)
    g_hi = _moment(lam_bracket) - float(pi_target)
    if g_lo > 0.0 or g_hi < 0.0:
        raise RealizabilityBoundaryError(
            f"|lambda| exceeds {lam_bracket} for pi_target={pi_target}: the maxent multiplier has "
            "diverged near the realizability boundary (the band-limited logit stays finite).")
    return float(brentq(lambda lam: _moment(lam) - float(pi_target), -lam_bracket, lam_bracket,
                        xtol=1e-12, maxiter=200))
