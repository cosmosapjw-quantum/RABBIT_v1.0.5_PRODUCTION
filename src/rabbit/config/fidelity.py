"""
rabbit.config.fidelity — Fidelity level classification.

Every result produced by rabbit carries a FidelityLevel label so that
surrogate and exact computation paths are never silently mixed.  The level
is determined by the least-faithful component in the pipeline: if any
single module operates at LEGACY_SURROGATE level, the entire result is
tagged LEGACY_SURROGATE regardless of other modules' levels.

The hierarchy from most to least faithful:

    REFERENCE_EXACT      Full momentum-resolved Boltzmann + live weak
                         functional + live collisions.  No approximations
                         beyond numerical quadrature.

    PRODUCTION_HIERARCHY Momentum-resolved hierarchy with validated
                         truncation (e.g. ℓ_max = 2 for Type I, which
                         is exact, or ℓ_max = 6 for curved types, which
                         is a validated truncation).

    IMPROVED_APPROX      Intermediate: e.g. parametric N_eff with live
                         weak rates, or moment-based transport without
                         full momentum resolution.

    LEGACY_SURROGATE     v3.1.0-equivalent: tabulated weak rates,
                         perturbative anisotropy corrections, Teff
                         framework.  Retained only for regression.

    STUB                 Placeholder returning hardcoded values.
                         Used during incremental development before
                         a module is implemented.
"""

from __future__ import annotations

from enum import IntEnum


class FidelityLevel(IntEnum):
    """Computation fidelity level, ordered from most to least faithful.

    Comparison operators are meaningful: REFERENCE_EXACT > STUB.
    The ``min()`` of component levels gives the overall result fidelity.
    """
    REFERENCE_EXACT = 5
    PRODUCTION_HIERARCHY = 4
    IMPROVED_APPROX = 3
    LEGACY_SURROGATE = 2
    STUB = 1

    @property
    def is_publishable(self) -> bool:
        """Whether results at this level can appear in a paper."""
        return self >= FidelityLevel.PRODUCTION_HIERARCHY

    @property
    def short_label(self) -> str:
        """Compact label for result tables and filenames."""
        return {
            FidelityLevel.REFERENCE_EXACT: "ref_exact",
            FidelityLevel.PRODUCTION_HIERARCHY: "prod_hier",
            FidelityLevel.IMPROVED_APPROX: "improved",
            FidelityLevel.LEGACY_SURROGATE: "legacy",
            FidelityLevel.STUB: "stub",
        }[self]


def combined_fidelity(*levels: FidelityLevel) -> FidelityLevel:
    """Return the overall fidelity for a pipeline with multiple components.

    The result is the minimum (least faithful) of all component levels.
    """
    if not levels:
        raise ValueError("At least one FidelityLevel required.")
    return FidelityLevel(min(levels))
