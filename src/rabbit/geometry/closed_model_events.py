"""rabbit.geometry.closed_model_events — Type IX recollapse + Mixmaster guards.

Foundation block F5 of the v1.0.5 → v2.0.0 upgrade plan.  Required only
by Wave-7 cells (Type IX orthogonal/tilted) but lives in the geometry
layer so non-Wave-7 cells can opt in if they ever explore K>0 regimes.

Type IX is the only Bianchi geometry that recollapses (K_FLRW > 0
throughout); without explicit termination handling, the integrator
runs into ``H → 0`` then ``H < 0`` and produces NaN abundances.  The BKL
Mixmaster regime additionally manifests as rapid sign flips in the
(N₁, N₂, N₃) eigenvalues, which can stall an adaptive solver.

This module provides:

  * ``TerminalReason``: enum classifying how a Type IX trajectory ended.
  * ``classify_typeIX_state``: post-step classifier the driver calls.
  * ``hubble_zero_event_factory``: Diffrax-compatible event_fn that
    fires at H = 0 (the recollapse turning point).
  * ``MixmasterTracker``: stateful tracker counting (N_i) sign flips
    inside a sliding e-fold window; fires at >10 flips inside 0.5 e-folds.
  * ``IXEventConfig``: per-driver knob bundle.

The classifier is jit-incompatible (it returns Python objects); the
Diffrax event_fn returned by the factory is jit-compatible (returns a
single float).  Per-cell PRs that need full jit-traced state machine
should flatten the classifier into integer codes, but F5 keeps the
public API human-readable.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Callable, Optional, Sequence, Tuple


class TerminalReason(Enum):
    """How a Type IX trajectory finished."""
    RUNNING = "running"             # not yet terminated
    BBN_COMPLETE = "bbn_complete"   # T_γ < T_BBN_END, normal exit
    RECOLLAPSED = "recollapsed"     # H ≤ 0; geometry began contracting
    MIXMASTER_DIVERGENT = "mixmaster_divergent"
    OMEGA_OUT_OF_RANGE = "omega_out_of_range"  # Ω budget exceeded


@dataclass(frozen=True)
class IXEventConfig:
    """Tunable thresholds for the Type IX state machine.

    Defaults are conservative: a real Type IX BBN run (Σ_H≲0.5, IC near
    the standard Friedmann-Lemaitre seed) finishes BBN well before the
    recollapse turning point and never enters Mixmaster, so all guards
    stay armed but inactive.
    """
    T_BBN_END_MeV: float = 0.01           # 10 keV; Yp/D/H frozen well above this
    H_recollapse_eps: float = 1.0e-3      # H/H0 fraction below which we declare recollapse
    omega_budget_max: float = 1.5         # Ω + Σ² + K + cA² hard cap
    mixmaster_window_efolds: float = 0.5  # sliding window for sign-flip counter
    mixmaster_max_flips: int = 10         # > this => MIXMASTER_DIVERGENT
    continue_through_bounce: bool = False  # opt-in: flip dN sign at H=0 (out of canonical scope)


def classify_typeIX_state(
    *,
    H_over_H0: float,
    T_gamma_MeV: float,
    omega_budget: float,
    mixmaster_flip_count: int = 0,
    config: Optional[IXEventConfig] = None,
) -> TerminalReason:
    """Classify the current step of a Type IX integration.

    Parameters
    ----------
    H_over_H0
        Hubble rate normalised to its initial value.  Drops toward zero
        as the closed model approaches recollapse.
    T_gamma_MeV
        Photon temperature in MeV.
    omega_budget
        Ω + Σ² + K + c A² evaluated at the current step.  Should remain
        ≈ 1 modulo numerical drift; large excursion ⇒ solver pathology.
    mixmaster_flip_count
        Number of (N₁, N₂, N₃) sign flips inside the current sliding
        window (set by ``MixmasterTracker``).  Default 0 ⇒ no flip
        history available.
    config
        Threshold overrides; defaults to ``IXEventConfig()``.
    """
    cfg = config or IXEventConfig()

    if T_gamma_MeV < cfg.T_BBN_END_MeV:
        return TerminalReason.BBN_COMPLETE
    if H_over_H0 <= cfg.H_recollapse_eps:
        return TerminalReason.RECOLLAPSED
    if abs(omega_budget) > cfg.omega_budget_max:
        return TerminalReason.OMEGA_OUT_OF_RANGE
    if mixmaster_flip_count > cfg.mixmaster_max_flips:
        return TerminalReason.MIXMASTER_DIVERGENT
    return TerminalReason.RUNNING


# ─────────────────────────────────────────────────────────────────────
# Diffrax event factory: H = 0 turning point.
# ─────────────────────────────────────────────────────────────────────

def hubble_zero_event_factory(
    extract_H_from_state: Callable[[float, Sequence[float]], float],
    *,
    H_threshold_fraction: float = 1.0e-3,
    H_initial: float = 1.0,
) -> Callable[[float, Sequence[float]], float]:
    """Return a Diffrax-compatible event function firing at H ≤ ε·H_initial.

    Diffrax events trigger when the returned float crosses zero with the
    correct sign.  We build ``cond(t, y) = H(y) - ε·H_initial`` so that
    the event fires when H drops below ε.

    The driver supplies ``extract_H_from_state(t, y) → H_value`` because
    the position of H in the state vector is per-cell.  ``H_initial`` is
    passed at factory time (kept positional-friendly for jit).
    """
    eps = float(H_threshold_fraction) * float(H_initial)

    def cond_fn(t: float, y: Sequence[float]) -> float:
        return float(extract_H_from_state(t, y)) - eps

    return cond_fn


# ─────────────────────────────────────────────────────────────────────
# MixmasterTracker: BKL oscillation onset detector.
# ─────────────────────────────────────────────────────────────────────

@dataclass
class MixmasterTracker:
    """Sliding-window counter of (N₁, N₂, N₃) sign flips.

    BKL Mixmaster oscillations near the singularity manifest as rapid
    sign flips in the structure-constant eigenvalues.  Adaptive solvers
    typically stall (step rejection chain) when the flip rate exceeds a
    few per e-fold.  This tracker fires ``MIXMASTER_DIVERGENT`` once the
    flip count inside the configured window crosses the threshold.

    The tracker is plain-Python (not jit-traceable) since it carries
    list state.  Drivers call ``observe`` after each accepted step and
    read ``flip_count_in_window`` for the classifier.
    """
    config: IXEventConfig = field(default_factory=IXEventConfig)
    history: list[tuple[float, Tuple[int, int, int]]] = field(default_factory=list)

    def observe(self, N_efold: float, n_signs: Tuple[int, int, int]) -> None:
        """Record an accepted step's (N, sign(N₁), sign(N₂), sign(N₃))."""
        self.history.append((float(N_efold), tuple(int(s) for s in n_signs)))

    @property
    def flip_count_in_window(self) -> int:
        """Number of axis-by-axis sign flips inside the trailing window."""
        if len(self.history) < 2:
            return 0
        cutoff = self.history[-1][0] - self.config.mixmaster_window_efolds
        flips = 0
        prev_signs: Optional[Tuple[int, int, int]] = None
        for N, signs in self.history:
            if N < cutoff:
                continue
            if prev_signs is not None:
                for cur, prev in zip(signs, prev_signs):
                    # sign 0 (zero eigenvalue) is treated as no-flip with itself.
                    if cur != prev and cur != 0 and prev != 0:
                        flips += 1
            prev_signs = signs
        return flips

    def trim(self) -> None:
        """Drop history older than the sliding window (memory hygiene)."""
        if not self.history:
            return
        cutoff = self.history[-1][0] - self.config.mixmaster_window_efolds
        self.history = [(N, s) for (N, s) in self.history if N >= cutoff]


# ─────────────────────────────────────────────────────────────────────
# Public exports.
# ─────────────────────────────────────────────────────────────────────

__all__ = [
    "TerminalReason",
    "IXEventConfig",
    "classify_typeIX_state",
    "hubble_zero_event_factory",
    "MixmasterTracker",
]
