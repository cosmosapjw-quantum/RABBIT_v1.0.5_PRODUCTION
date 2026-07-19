"""
rabbit.config.conventions — Fundamental conventions and enumerations.

Defines the physical conventions used throughout the rabbit pipeline:
metric signature, independent variable, species labels, and the ℓ_max = 2
exactness theorem for Type I.

All modules in rabbit import their convention choices from here to
ensure global consistency.
"""

from __future__ import annotations

import math
import warnings
from dataclasses import dataclass, field
from enum import Enum, auto
from typing import Optional, Tuple


# ═══════════════════════════════════════════════════════════════════════
# §1. Spacetime conventions
# ═══════════════════════════════════════════════════════════════════════

#: Metric signature.  (−, +, +, +) throughout.
METRIC_SIGNATURE: Tuple[int, ...] = (-1, +1, +1, +1)

#: Independent variable for all ODEs: N = ln a (number of e-folds).
#: Hubble-normalized quantities are dimensionless in this variable.
TIME_VARIABLE: str = "N"


# ═══════════════════════════════════════════════════════════════════════
# §2. Neutrino species
# ═══════════════════════════════════════════════════════════════════════

class NeutrinoSpecies(Enum):
    """Neutrino species tracked in the transport hierarchy.

    All six (anti)neutrino states are evolved independently in the
    momentum-resolved hierarchy.  Although ν_μ = ν̄_μ = ν_τ = ν̄_τ at
    early times (CPT + flavor universality), separate evolution is
    needed for: (a) the 972-DOF phase-2 canonical state vector, (b) forward
    compatibility with Tier 3+ (diagonal ν–ν scattering breaks
    flavor universality), (c) proper collision kernel bookkeeping.

    For practical purposes at Tier ≤ 2, ν_μ, ν̄_μ, ν_τ, ν̄_τ share
    identical initial conditions and evolution equations, so they can
    be initialized from a single template and verified to remain
    identical as a consistency check.
    """
    NU_E = auto()         # electron neutrino
    NU_E_BAR = auto()     # electron antineutrino
    NU_MU = auto()        # muon neutrino
    NU_MU_BAR = auto()    # muon antineutrino
    NU_TAU = auto()       # tau neutrino
    NU_TAU_BAR = auto()   # tau antineutrino


#: Default species list for transport hierarchy (all 6).
DEFAULT_SPECIES: Tuple[NeutrinoSpecies, ...] = (
    NeutrinoSpecies.NU_E,
    NeutrinoSpecies.NU_E_BAR,
    NeutrinoSpecies.NU_MU,
    NeutrinoSpecies.NU_MU_BAR,
    NeutrinoSpecies.NU_TAU,
    NeutrinoSpecies.NU_TAU_BAR,
)

#: Number of independently evolved species in the hierarchy.
#: 6 species × 2 multipoles × N_q bins = transport DOF.
N_SPECIES_TRANSPORT: int = len(DEFAULT_SPECIES)

#: Species that are "ν_x"-like (identical physics at Tier ≤ 2).
NU_X_SPECIES: Tuple[NeutrinoSpecies, ...] = (
    NeutrinoSpecies.NU_MU,
    NeutrinoSpecies.NU_MU_BAR,
    NeutrinoSpecies.NU_TAU,
    NeutrinoSpecies.NU_TAU_BAR,
)

#: Species that couple to weak n↔p rates (Born level).
WEAK_COUPLED_SPECIES: Tuple[NeutrinoSpecies, ...] = (
    NeutrinoSpecies.NU_E,
    NeutrinoSpecies.NU_E_BAR,
)


# ═══════════════════════════════════════════════════════════════════════
# §3. Multipole conventions
# ═══════════════════════════════════════════════════════════════════════

#: For Type I (K = 0, ν_m = 0): ℓ_max = 2 is EXACT, not a truncation.
#: Vicente–Pereira–Pitrou (2025) proved that the free-streaming coupling
#: coefficients κ_ℓ^m vanish identically when spatial curvature K = 0
#: and eigenvalue ν_m = 0.  The PSTF hierarchy truncates analytically.
#: This constant is the physics truth for Type I, not a tuning parameter.
ELLMAX_TYPE_I: int = 2

#: Active ℓ values for Type I orthogonal: monopole (0) + quadrupole (2).
#: ℓ = 1 vanishes by symmetry in orthogonal models.
ACTIVE_ELL_TYPE_I: Tuple[int, ...] = (0, 2)

#: Active ℓ values for GENERIC Type I (non-LRS): monopole + two quadrupole
#: polarizations Ψ₂⁺ (couples to Σ₊) and Ψ₂⁻ (couples to Σ₋).
#: The −2 label is a convention for the second quadrupole polarization.
ACTIVE_ELL_TYPE_I_GENERIC: Tuple[int, ...] = (0, 2, -2)

#: Number of multipole components per species for Type I.
N_MULTIPOLES_TYPE_I: int = len(ACTIVE_ELL_TYPE_I)
N_MULTIPOLES_TYPE_I_GENERIC: int = len(ACTIVE_ELL_TYPE_I_GENERIC)


# ═══════════════════════════════════════════════════════════════════════
# §4. Nuclear species
# ═══════════════════════════════════════════════════════════════════════

class NuclearSpecies(Enum):
    """Nuclear species in the standard 9-species PRIMAT-style network."""
    NEUTRON = 0
    PROTON = 1
    DEUTERIUM = 2
    TRITIUM = 3
    HELIUM3 = 4
    HELIUM4 = 5
    LITHIUM7 = 6
    BERYLLIUM7 = 7
    LITHIUM6 = 8


#: Species names matching the v3.1.0 convention.
NUCLEAR_SPECIES_NAMES: Tuple[str, ...] = (
    'n', 'p', 'D', 'T', '3He', '4He', '7Li', '7Be', '6Li',
)

#: Mass numbers A_i for each species.
NUCLEAR_MASS_NUMBERS: Tuple[int, ...] = (1, 1, 2, 3, 3, 4, 7, 7, 6)

#: Number of nuclear species.
N_NUCLEAR: int = 9


# ═══════════════════════════════════════════════════════════════════════
# §5. Bianchi type classification
# ═══════════════════════════════════════════════════════════════════════

class BianchiType(Enum):
    """Bianchi type classification (Ellis–MacCallum scheme).

    Type I is the primary target for Paper I.  Class A types (I, II,
    VI₀, VII₀, VIII, IX) share the unified Wainwright–Hsu framework.
    """
    TYPE_I = "I"
    TYPE_II = "II"
    TYPE_VI0 = "VI0"
    TYPE_VII0 = "VII0"
    TYPE_VIII = "VIII"
    TYPE_IX = "IX"
    # Class B:
    TYPE_V = "V"
    TYPE_IV = "IV"
    TYPE_III = "III"
    TYPE_VIH = "VIh"
    TYPE_VIIH = "VIIh"
    TYPE_VI_M19 = "VI_m19"


# ═══════════════════════════════════════════════════════════════════════
# §5b. h-parameter family (Foundation F1)
# ═══════════════════════════════════════════════════════════════════════

#: Types that carry a Wainwright–Hsu h-parameter family (Class B with two
#: nonzero N_i eigenvalues whose ratio is parametrized by h).
H_FAMILY_TYPES: Tuple[BianchiType, ...] = (
    BianchiType.TYPE_VIH,    # h < 0, h ≠ -1
    BianchiType.TYPE_VIIH,   # h > 0
)

#: Default h values for types that have a fixed canonical h.
#: III is the h = -1 limit of VI_h (the special isotropy direction); the
#: VI_{-1/9} exception sits at h = -1/9 (the c = 15/4 Friedmann point).
_DEFAULT_H: dict[BianchiType, Optional[float]] = {
    BianchiType.TYPE_III:    -1.0,
    BianchiType.TYPE_VI_M19: -1.0 / 9.0,
}

#: Hard rejection thresholds.  Beyond these the canonical Wainwright–Hsu
#: chart is singular and the structure constants degenerate.
_H_HARD_LIMIT: float = 100.0
#: Soft warning threshold above which |h| produces extreme-anisotropy
#: configurations the candidate-tier per-cell PRs should flag.
_H_SOFT_LIMIT: float = 10.0
#: Tolerance for the III-degeneracy guard inside VI_h.
_H_TYPE_III_TOL: float = 1.0e-3
#: Tolerance for the VI_{-1/9} exceptional guard inside VI_h.
_H_VI_M19_TOL: float = 1.0e-12


@dataclass(frozen=True)
class BianchiSpec:
    """Dispatch primitive for the 22-cell Bianchi rollout.

    Wraps a ``BianchiType`` plus the Wainwright–Hsu h-parameter and a
    sign convention for the (N₁, N₂, N₃) structure-constant eigenvalues.
    Foundation F1 of v1.0.5 → v2.0.0; consumed by every per-cell PR.

    Construction is the validation step: an invalid combination
    (h = 0 for VI_h, |h| > _H_HARD_LIMIT, h on the wrong sign for the
    type, …) raises at config-build time, never silently at integration
    time.

    Examples
    --------
    >>> BianchiSpec.from_type(BianchiType.TYPE_I)
    >>> BianchiSpec.from_type(BianchiType.TYPE_VIH, h=-3.0)
    >>> BianchiSpec.from_type(BianchiType.TYPE_VIIH, h=2.0)

    The ``Enum`` itself is unchanged — string-keyed dispatch in
    ``backend_capabilities``, ``feature_capabilities`` etc. continues to
    work because callers can recover the enum via ``spec.type``.
    """
    type: BianchiType
    h: Optional[float] = None
    sign_n: Tuple[int, int, int] = (+1, +1, +1)

    # ── construction / validation ─────────────────────────────────────

    @classmethod
    def from_type(
        cls,
        type: BianchiType,
        h: Optional[float] = None,
        sign_n: Tuple[int, int, int] = (+1, +1, +1),
    ) -> "BianchiSpec":
        """Construct a spec, defaulting and validating ``h`` per type.

        - Types in ``H_FAMILY_TYPES`` (VI_h, VII_h) require an explicit
          ``h`` and reject ``h=0``, |h| above the hard limit, and the
          near-III degeneracy for VI_h.
        - Types in ``_DEFAULT_H`` (III, VI_{-1/9}) take their canonical
          h automatically; explicitly passing ``h`` for them must match.
        - All other types reject ``h`` (must be ``None``).
        """
        if type in H_FAMILY_TYPES:
            if h is None:
                raise ValueError(
                    f"BianchiSpec.from_type: {type.name} is an h-family "
                    f"type and requires an explicit h argument."
                )
            cls._validate_h_value(type, float(h))
            return cls(type=type, h=float(h), sign_n=tuple(sign_n))

        if type in _DEFAULT_H:
            canonical = _DEFAULT_H[type]
            if h is not None and not math.isclose(float(h), canonical, abs_tol=1e-12):
                raise ValueError(
                    f"BianchiSpec.from_type: {type.name} has canonical h={canonical!r}; "
                    f"got h={h!r}"
                )
            return cls(type=type, h=canonical, sign_n=tuple(sign_n))

        # Types without an h-parameter.
        if h is not None:
            raise ValueError(
                f"BianchiSpec.from_type: {type.name} does not take an h-parameter; "
                f"got h={h!r}."
            )
        return cls(type=type, h=None, sign_n=tuple(sign_n))

    @staticmethod
    def _validate_h_value(type: BianchiType, h_val: float) -> None:
        """Validate an explicit h for h-family types VI_h / VII_h."""
        if not math.isfinite(h_val):
            raise ValueError(
                f"BianchiSpec: h must be finite; got {h_val!r}."
            )
        if h_val == 0.0:
            raise ValueError(
                f"BianchiSpec: h=0 is degenerate for {type.name}.  Use "
                f"BianchiType.TYPE_VII0 (the h→0 limit of VI_h/VII_h) "
                f"or BianchiType.TYPE_II/V depending on your intent."
            )
        if abs(h_val) >= _H_HARD_LIMIT:
            raise ValueError(
                f"BianchiSpec: |h| >= {_H_HARD_LIMIT} for {type.name} is "
                f"singular in the canonical Wainwright–Hsu chart; got "
                f"h={h_val!r}."
            )

        if type is BianchiType.TYPE_VIH:
            if h_val >= 0.0:
                raise ValueError(
                    f"BianchiSpec: TYPE_VIH requires h < 0; got h={h_val!r}."
                )
            if abs(h_val + 1.0) < _H_TYPE_III_TOL:
                raise ValueError(
                    f"BianchiSpec: TYPE_VIH with h≈-1 is the canonical "
                    f"limit Type III; use BianchiType.TYPE_III explicitly. "
                    f"got h={h_val!r}."
                )
            if abs(h_val + 1.0 / 9.0) < _H_VI_M19_TOL:
                raise ValueError(
                    f"BianchiSpec: TYPE_VIH with h≈-1/9 is the exceptional "
                    f"VI_{{-1/9}} point with c=15/4; use "
                    f"BianchiType.TYPE_VI_M19 explicitly. got h={h_val!r}."
                )
        elif type is BianchiType.TYPE_VIIH:
            if h_val <= 0.0:
                raise ValueError(
                    f"BianchiSpec: TYPE_VIIH requires h > 0; got h={h_val!r}."
                )

        if abs(h_val) > _H_SOFT_LIMIT:
            warnings.warn(
                f"BianchiSpec: |h|={abs(h_val):.2f} for {type.name} is "
                f"in the extreme-anisotropy regime; numerical convergence "
                f"may degrade.",
                RuntimeWarning,
                stacklevel=3,
            )

    def __post_init__(self) -> None:
        # When a spec is constructed directly (not via from_type) we
        # still apply structural validation; the only thing relaxed is
        # that __post_init__ does not auto-fill defaults for III /
        # VI_M19, since the user has chosen to pass h explicitly.
        for s in self.sign_n:
            if s not in (-1, +1):
                raise ValueError(
                    f"BianchiSpec.sign_n entries must be ±1; got {self.sign_n!r}."
                )
        if self.type in H_FAMILY_TYPES:
            if self.h is None:
                raise ValueError(
                    f"BianchiSpec: {self.type.name} requires h.  "
                    f"Use BianchiSpec.from_type(...) to fail loudly."
                )
            self._validate_h_value(self.type, float(self.h))
        elif self.type in _DEFAULT_H:
            canonical = _DEFAULT_H[self.type]
            if self.h is not None and not math.isclose(
                float(self.h), float(canonical), abs_tol=1e-12
            ):
                raise ValueError(
                    f"BianchiSpec: {self.type.name} canonical h={canonical}, "
                    f"got h={self.h!r}."
                )
        else:
            if self.h is not None:
                raise ValueError(
                    f"BianchiSpec: {self.type.name} cannot have h-parameter; "
                    f"got h={self.h!r}."
                )

    # ── classification helpers ────────────────────────────────────────

    @property
    def is_class_A(self) -> bool:
        return self.type in {
            BianchiType.TYPE_I,
            BianchiType.TYPE_II,
            BianchiType.TYPE_VI0,
            BianchiType.TYPE_VII0,
            BianchiType.TYPE_VIII,
            BianchiType.TYPE_IX,
        }

    @property
    def is_class_B(self) -> bool:
        return not self.is_class_A

    @property
    def is_lrs(self) -> bool:
        """LRS = locally-rotationally-symmetric (Σ₋ ≡ 0 by construction)."""
        return self.type in {
            BianchiType.TYPE_I,
            BianchiType.TYPE_II,
            BianchiType.TYPE_V,
            BianchiType.TYPE_IV,
        }

    @property
    def has_h_parameter(self) -> bool:
        return self.h is not None

    # ── structure-constant eigenvalues ───────────────────────────────

    def canonical_n_eigenvalues(self, scale: float = 1.0) -> Tuple[float, float, float]:
        """Return (n₁, n₂, n₃) eigenvalues of the structure-constant tensor.

        Adopts the Wainwright–Hsu canonical chart.  The returned triple
        is multiplied by ``scale`` so callers can drive a small-anisotropy
        limit by scaling all three components together.

        Sign conventions:

        - Class A (Wainwright & Ellis 1997 Tab. 6.2):
            I    → (0, 0, 0)
            II   → (+1, 0, 0)
            VI_0 → (+1, -1, 0)
            VII_0→ (+1, +1, 0)
            VIII → (+1, +1, -1)
            IX   → (+1, +1, +1)
        - Class B without h:
            V    → (0, 0, 0)
            IV   → (+1, 0, 0)
        - Class B with h (Ellis–MacCallum / Wainwright–Hsu):
            III  → (0, +1, -1)              (canonical h = -1 limit)
            VI_h → (0, +1, +h)   with h<0   (h=-1 ↔ III)
            VII_h→ (0, +1, +h)   with h>0   (h→0⁺ ↔ VII_0 family)
            VI_{-1/9} → (0, +1, -1/9)       (the c=15/4 exception)

        The optional ``self.sign_n`` entries flip the corresponding
        eigenvalues; callers pin reflection-symmetry.
        """
        s1, s2, s3 = self.sign_n
        if self.type is BianchiType.TYPE_I:
            n = (0.0, 0.0, 0.0)
        elif self.type is BianchiType.TYPE_II:
            n = (s1 * 1.0, 0.0, 0.0)
        elif self.type is BianchiType.TYPE_VI0:
            n = (s1 * 1.0, s2 * -1.0, 0.0)
        elif self.type is BianchiType.TYPE_VII0:
            n = (s1 * 1.0, s2 * 1.0, 0.0)
        elif self.type is BianchiType.TYPE_VIII:
            n = (s1 * 1.0, s2 * 1.0, s3 * -1.0)
        elif self.type is BianchiType.TYPE_IX:
            n = (s1 * 1.0, s2 * 1.0, s3 * 1.0)
        elif self.type is BianchiType.TYPE_V:
            n = (0.0, 0.0, 0.0)
        elif self.type is BianchiType.TYPE_IV:
            n = (s1 * 1.0, 0.0, 0.0)
        elif self.type is BianchiType.TYPE_III:
            n = (0.0, s2 * 1.0, s3 * -1.0)
        elif self.type is BianchiType.TYPE_VIH:
            n = (0.0, s2 * 1.0, s3 * float(self.h))   # h < 0
        elif self.type is BianchiType.TYPE_VIIH:
            n = (0.0, s2 * 1.0, s3 * float(self.h))   # h > 0
        elif self.type is BianchiType.TYPE_VI_M19:
            n = (0.0, s2 * 1.0, s3 * (-1.0 / 9.0))
        else:
            raise ValueError(f"Unknown Bianchi type: {self.type!r}")
        return (scale * n[0], scale * n[1], scale * n[2])

    # ── representation ────────────────────────────────────────────────

    def __repr__(self) -> str:
        h_part = f", h={self.h}" if self.h is not None else ""
        sn_part = f", sign_n={self.sign_n}" if self.sign_n != (1, 1, 1) else ""
        return f"BianchiSpec({self.type.name}{h_part}{sn_part})"
