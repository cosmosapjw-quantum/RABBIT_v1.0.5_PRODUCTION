"""Unexecuted W7/B3-v2 contract source for BD622 D-035 remediation.

Status: SPECIFIED.  Importing or compiling this file is not W7 execution, B3
implementation authority, or validation.  The module contains only the exact
reaction, kinematic, deterministic-arithmetic, and test-vector source reviewed
before a later owner decision.  It deliberately has no CLI and emits no result.
"""

from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Callable, Iterable, Literal, Mapping, Sequence, TypeVar


GF_DECIMAL = "1.1663788e-11"
ME_DECIMAL = "0.5109989500"
SIN2_THETA_W_DECIMAL = "0.23122"
SPECIES = ("nu_e", "antinu_e", "nu_mu", "antinu_mu", "nu_tau", "antinu_tau")
FLAVOURS = ("e", "mu", "tau")
NU = {"e": "nu_e", "mu": "nu_mu", "tau": "nu_tau"}
ANTINU = {"e": "antinu_e", "mu": "antinu_mu", "tau": "antinu_tau"}
EPSILON64 = 2.0**-52
ENTROPY_SEED = 0xBD622B3A7C0FFEE1
MASK64 = (1 << 64) - 1


@dataclass(frozen=True)
class Reaction:
    orbit_id: str
    legs: tuple[str, str, str, str]
    nu_leg: tuple[int, int, int, int]
    symmetry_num: int
    symmetry_den: int
    amplitude_id: str
    orbit_multiplicity_num: int = 1
    orbit_multiplicity_den: int = 1
    orientation_num: int = 1
    orientation_den: int = 1


def reaction_weight(reaction: Reaction) -> float:
    """Return O_r S_r m_r in the binding left-associated operation order."""
    symmetry = reaction.symmetry_num / reaction.symmetry_den
    orientation = reaction.orientation_num / reaction.orientation_den
    multiplicity = reaction.orbit_multiplicity_num / reaction.orbit_multiplicity_den
    return (orientation * symmetry) * multiplicity


def reaction_graph() -> tuple[Reaction, ...]:
    """Return the unique 24-self/15-electron graph in binding order."""
    rows: list[Reaction] = []
    nu_leg = (-1, -1, 1, 1)

    for a in SPECIES:
        rows.append(
            Reaction(
                f"S{len(rows):02d}",
                (a, a, a, a),
                nu_leg,
                1,
                4,
                "N128_KS_IDENTICAL",
                orientation_den=2,
            )
        )

    for flavour in FLAVOURS:
        rows.append(
            Reaction(
                f"S{len(rows):02d}",
                (NU[flavour], ANTINU[flavour], NU[flavour], ANTINU[flavour]),
                nu_leg,
                1,
                1,
                "N128_KT",
                orientation_den=2,
            )
        )

    for cp_map in (NU, ANTINU):
        for i, left in enumerate(FLAVOURS):
            for right in FLAVOURS[i + 1 :]:
                a, b = cp_map[left], cp_map[right]
                rows.append(
                    Reaction(
                        f"S{len(rows):02d}",
                        (a, b, a, b),
                        nu_leg,
                        1,
                        1,
                        "N32_KS",
                        orientation_den=2,
                    )
                )

    for left in FLAVOURS:
        for right in FLAVOURS:
            if left == right:
                continue
            a, b = NU[left], ANTINU[right]
            rows.append(
                Reaction(
                    f"S{len(rows):02d}",
                    (a, b, a, b),
                    nu_leg,
                    1,
                    1,
                    "N32_KT",
                    orientation_den=2,
                )
            )

    for i, left in enumerate(FLAVOURS):
        for right in FLAVOURS[i + 1 :]:
            rows.append(
                Reaction(
                    f"S{len(rows):02d}",
                    (NU[left], ANTINU[left], NU[right], ANTINU[right]),
                    nu_leg,
                    1,
                    1,
                    "N32_KT",
                )
            )

    if len(rows) != 24:
        raise AssertionError(f"self graph has {len(rows)} rows, expected 24")

    electron_index = 0
    for a in SPECIES:
        is_antinu = a.startswith("antinu")
        for charge in ("e-", "e+"):
            same_cp = (not is_antinu and charge == "e-") or (is_antinu and charge == "e+")
            amplitude = "ELASTIC_S" if same_cp else "ELASTIC_T"
            rows.append(
                Reaction(
                    f"E{electron_index:02d}",
                    (a, charge, a, charge),
                    nu_leg,
                    1,
                    1,
                    amplitude,
                    orientation_den=2,
                )
            )
            electron_index += 1

    for flavour in FLAVOURS:
        rows.append(
            Reaction(
                f"E{electron_index:02d}",
                (NU[flavour], ANTINU[flavour], "e-", "e+"),
                nu_leg,
                1,
                1,
                f"PAIR_{flavour}",
            )
        )
        electron_index += 1

    if len(rows) != 39 or electron_index != 15:
        raise AssertionError("full graph must contain 24 self plus 15 electron rows")
    for reaction in rows:
        elastic = tuple(sorted(reaction.legs[:2])) == tuple(sorted(reaction.legs[2:]))
        expected_orientation = (1, 2) if elastic else (1, 1)
        if (reaction.orientation_num, reaction.orientation_den) != expected_orientation:
            raise AssertionError(f"orientation quotient mismatch for {reaction.orbit_id}")
    return tuple(rows)


def weak_couplings(flavour: str, sin2_theta_w):
    one = 1
    half = one / 2
    if flavour == "e":
        cv, ca = half + 2 * sin2_theta_w, half
    elif flavour in ("mu", "tau"):
        cv, ca = -half + 2 * sin2_theta_w, -half
    else:
        raise ValueError(f"unknown flavour {flavour!r}")
    return cv + ca, cv - ca


def self_matrix_element(amplitude_id: str, gf, k_s, k_t):
    table = {
        "N128_KS_IDENTICAL": 128 * gf * gf * k_s,
        "N128_KT": 128 * gf * gf * k_t,
        "N32_KS": 32 * gf * gf * k_s,
        "N32_KT": 32 * gf * gf * k_t,
    }
    return table[amplitude_id]


def electron_matrix_element(
    amplitude_id: str,
    *,
    gf,
    g_l,
    g_r,
    m_e,
    chi12,
    chi13,
    chi14,
    chi23,
    chi24,
    chi34,
):
    k_s = chi12 * chi34
    k_t = chi14 * chi23
    k_u = chi13 * chi24
    if amplitude_id == "ELASTIC_S":
        body = g_l * g_l * k_s + g_r * g_r * k_t - g_l * g_r * m_e * m_e * chi13
    elif amplitude_id == "ELASTIC_T":
        body = g_l * g_l * k_t + g_r * g_r * k_s - g_l * g_r * m_e * m_e * chi13
    elif amplitude_id.startswith("PAIR_"):
        body = g_l * g_l * k_t + g_r * g_r * k_u + g_l * g_r * m_e * m_e * chi12
    else:
        raise ValueError(f"unknown electron amplitude {amplitude_id!r}")
    return 32 * gf * gf * body


def collision_drive(occupations: Sequence[float]) -> float:
    """Return J=F-R in the binding four-leg operation order."""
    if len(occupations) != 4 or any(
        not math.isfinite(value) or not 0.0 < value < 1.0 for value in occupations
    ):
        raise ValueError("collision occupations must be finite and strict-open")
    f1, f2, f3, f4 = occupations
    forward = ((f1 * f2) * (1.0 - f3)) * (1.0 - f4)
    reverse = ((f3 * f4) * (1.0 - f1)) * (1.0 - f2)
    return forward - reverse


def invariant_event_prefactor(
    reaction: Reaction,
    p1: Sequence[float],
    p2: Sequence[float],
    *,
    dp1_dcoordinate: float,
    dp2_dcoordinate: float,
    phase_space_density: float,
    matrix_element: float,
    quadrature_weights: tuple[float, float, float, float, float],
    t_cm: float,
) -> float:
    """Complete isotropic dGamma and 2*pi^2/T_cm^3 prefactor for one node tuple."""
    if (
        len(p1) != 4
        or len(p2) != 4
        or p1[0] <= 0.0
        or p2[0] <= 0.0
        or dp1_dcoordinate <= 0.0
        or dp2_dcoordinate <= 0.0
        or phase_space_density <= 0.0
        or matrix_element < 0.0
        or t_cm <= 0.0
        or any(weight <= 0.0 for weight in quadrature_weights)
    ):
        raise ValueError("invalid event prefactor input")
    two_pi = 2.0 * math.pi
    phase_denominator = (two_pi * two_pi) * two_pi
    p1_magnitude = norm3(p1[1:])
    p2_magnitude = norm3(p2[1:])
    incoming1 = ((p1_magnitude * p1_magnitude) * dp1_dcoordinate) / (
        (2.0 * p1[0]) * phase_denominator
    )
    incoming2 = ((p2_magnitude * p2_magnitude) * dp2_dcoordinate) / (
        (2.0 * p2[0]) * phase_denominator
    )
    isotropic_angles = (4.0 * math.pi) * (2.0 * math.pi)
    event = reaction_weight(reaction) * isotropic_angles
    event = event * incoming1
    event = event * incoming2
    event = event * phase_space_density
    event = event * matrix_element
    for weight in quadrature_weights:
        event = event * weight
    q_prefactor = (2.0 * math.pi * math.pi) / ((t_cm * t_cm) * t_cm)
    return q_prefactor * event


def event_leg_contributions(
    reaction: Reaction,
    occupations: Sequence[float],
    basis_by_leg: Sequence[Sequence[float] | None],
    *,
    event_key_prefix: tuple[int | str, ...],
    prefactor: float,
) -> tuple[EventContribution, ...]:
    """Deposit one oriented event to every explicit neutrino leg and test mode."""
    if len(basis_by_leg) != 4 or not math.isfinite(prefactor):
        raise ValueError("invalid event deposition input")
    drive = collision_drive(occupations)
    contributions = []
    for leg_index, (species, sign, basis_values) in enumerate(
        zip(reaction.legs, reaction.nu_leg, basis_by_leg)
    ):
        if species not in SPECIES:
            if basis_values is not None:
                raise ValueError("electron leg cannot carry a neutrino basis")
            continue
        if basis_values is None:
            raise ValueError("neutrino leg is missing basis values")
        species_index = SPECIES.index(species)
        for mode_index, basis_value in enumerate(basis_values):
            value = ((prefactor * drive) * sign) * basis_value
            contributions.append(
                EventContribution(
                    key=event_key_prefix
                    + (species_index, mode_index, leg_index),
                    species_index=species_index,
                    mode_index=mode_index,
                    value=value,
                )
            )
    return tuple(contributions)


def dot3(a: Sequence[float], b: Sequence[float]) -> float:
    return a[0] * b[0] + a[1] * b[1] + a[2] * b[2]


def add3(a: Sequence[float], b: Sequence[float]) -> tuple[float, float, float]:
    return a[0] + b[0], a[1] + b[1], a[2] + b[2]


def scale3(c: float, a: Sequence[float]) -> tuple[float, float, float]:
    return c * a[0], c * a[1], c * a[2]


def cross3(a: Sequence[float], b: Sequence[float]) -> tuple[float, float, float]:
    return (
        a[1] * b[2] - a[2] * b[1],
        a[2] * b[0] - a[0] * b[2],
        a[0] * b[1] - a[1] * b[0],
    )


def norm3(a: Sequence[float]) -> float:
    return math.sqrt(dot3(a, a))


@dataclass(frozen=True)
class Boost:
    beta: tuple[float, float, float]
    beta2: float
    gamma: float


@dataclass(frozen=True)
class ThresholdSupport:
    process: Literal["elastic", "pair", "massless"]
    s: float
    threshold: float
    k_star: float
    phase_space_density: float
    derivative_side: Literal["physical-right"]


@dataclass(frozen=True)
class TwoBodyInterior:
    process: Literal["elastic", "pair", "massless"]
    s: float
    k_star: float
    p3: tuple[float, float, float, float]
    p4: tuple[float, float, float, float]
    phase_space_density: float
    boost: Boost
    max_conservation_residual: float
    max_shell_residual: float
    mandelstam_residual: float


@dataclass(frozen=True)
class COMShell:
    process: Literal["elastic", "pair", "massless"]
    threshold: float
    incoming_masses: tuple[float, float]
    outgoing_masses: tuple[float, float]
    k_star: float
    e3_star: float
    e4_star: float
    support_only: bool


def minkowski_dot(p: Sequence[float], q: Sequence[float]) -> float:
    """Physical metric diag(-1,+1,+1,+1), evaluated left to right."""
    return ((-p[0] * q[0] + p[1] * q[1]) + p[2] * q[2]) + p[3] * q[3]


def _make_boost(p0: float, pvec: Sequence[float], root_s: float) -> Boost:
    beta = scale3(1.0 / p0, pvec)
    beta2 = dot3(beta, beta)
    if not 0.0 <= beta2 < 1.0:
        raise ValueError("nonphysical boost")
    gamma = p0 / root_s
    if not math.isfinite(gamma) or gamma < 1.0:
        raise ValueError("nonphysical gamma")
    return Boost(beta=beta, beta2=beta2, gamma=gamma)


def two_body_com_shell(
    process: Literal["elastic", "pair", "massless"], s: float, m_e: float
) -> COMShell:
    """Total process-specific support/root function used by source and threshold vectors."""
    if not math.isfinite(s) or s <= 0.0 or not math.isfinite(m_e) or m_e <= 0.0:
        raise ValueError("invalid COM shell inputs")
    m_e2 = m_e * m_e
    if process == "elastic":
        incoming_masses = (0.0, m_e)
        outgoing_masses = (0.0, m_e)
        threshold = m_e2
    elif process == "pair":
        incoming_masses = (0.0, 0.0)
        outgoing_masses = (m_e, m_e)
        threshold = 4.0 * m_e2
    elif process == "massless":
        incoming_masses = (0.0, 0.0)
        outgoing_masses = (0.0, 0.0)
        threshold = 0.0
    else:
        raise ValueError(f"unknown two-body process {process!r}")
    if s < threshold:
        raise ValueError("subthreshold event")
    if s == threshold:
        return COMShell(
            process=process,
            threshold=threshold,
            incoming_masses=incoming_masses,
            outgoing_masses=outgoing_masses,
            k_star=0.0,
            e3_star=0.0 if process == "elastic" else math.sqrt(s) / 2.0,
            e4_star=math.sqrt(s) if process == "elastic" else math.sqrt(s) / 2.0,
            support_only=True,
        )
    root_s = math.sqrt(s)
    if process == "elastic":
        k_star = (s - m_e2) / (2.0 * root_s)
        e3 = k_star
        e4 = (s + m_e2) / (2.0 * root_s)
    elif process == "pair":
        k_star = math.sqrt(s - 4.0 * m_e2) / 2.0
        e3 = root_s / 2.0
        e4 = root_s / 2.0
    else:
        k_star = root_s / 2.0
        e3 = k_star
        e4 = k_star
    return COMShell(
        process=process,
        threshold=threshold,
        incoming_masses=incoming_masses,
        outgoing_masses=outgoing_masses,
        k_star=k_star,
        e3_star=e3,
        e4_star=e4,
        support_only=False,
    )


def boost_to_com(p: Sequence[float], boost: Boost) -> tuple[float, float, float, float]:
    if boost.beta2 == 0.0:
        return tuple(p)  # type: ignore[return-value]
    spatial = p[1:]
    bp = dot3(boost.beta, spatial)
    energy = boost.gamma * (p[0] - bp)
    factor = ((boost.gamma - 1.0) * bp) / boost.beta2 - boost.gamma * p[0]
    out = add3(spatial, scale3(factor, boost.beta))
    return energy, out[0], out[1], out[2]


def boost_from_com(p: Sequence[float], boost: Boost) -> tuple[float, float, float, float]:
    if boost.beta2 == 0.0:
        return tuple(p)  # type: ignore[return-value]
    spatial = p[1:]
    bp = dot3(boost.beta, spatial)
    energy = boost.gamma * (p[0] + bp)
    factor = ((boost.gamma - 1.0) * bp) / boost.beta2 + boost.gamma * p[0]
    out = add3(spatial, scale3(factor, boost.beta))
    return energy, out[0], out[1], out[2]


def _frame_from_incoming(p1_star: Sequence[float]):
    magnitude = norm3(p1_star[1:])
    if magnitude <= 0.0:
        raise ValueError("incoming COM neutrino has zero momentum")
    e_z = scale3(1.0 / magnitude, p1_star[1:])
    axes = ((1.0, 0.0, 0.0), (0.0, 1.0, 0.0), (0.0, 0.0, 1.0))
    ref = min(enumerate(axes), key=lambda item: (abs(dot3(item[1], e_z)), item[0]))[1]
    trial = add3(ref, scale3(-dot3(ref, e_z), e_z))
    e_x = scale3(1.0 / norm3(trial), trial)
    e_y = cross3(e_z, e_x)
    return e_x, e_y, e_z


def two_body_outgoing(
    p1: Sequence[float],
    p2: Sequence[float],
    *,
    process: Literal["elastic", "pair", "massless"],
    m_e: float,
    z_star: float,
    phi_star: float,
) -> ThresholdSupport | TwoBodyInterior:
    """Construct one process-specific principal-branch COM event.

    Exact threshold equality returns support metadata only.  It never constructs
    a frame, divides by k_star, or enters a collision quadrature.
    """
    if len(p1) != 4 or len(p2) != 4 or not all(math.isfinite(v) for v in (*p1, *p2)):
        raise ValueError("incoming four-vectors must be finite length-four values")
    if not math.isfinite(m_e) or m_e <= 0.0:
        raise ValueError("m_e must be positive and finite")
    if not -1.0 <= z_star <= 1.0:
        raise ValueError("z_star lies outside [-1,1]")
    if not math.isfinite(phi_star):
        raise ValueError("phi_star must be finite")
    p0 = p1[0] + p2[0]
    pvec = add3(p1[1:], p2[1:])
    if p0 <= 0.0:
        raise ValueError("nonpositive total energy")
    s = p0 * p0 - dot3(pvec, pvec)
    if s <= 0.0:
        raise ValueError("nonpositive s")
    shell = two_body_com_shell(process, s, m_e)
    if shell.support_only:
        if p1[0] < 0.0 or p2[0] <= 0.0:
            raise ValueError("nonfuture support-boundary input")
        return ThresholdSupport(
            process=process,
            s=s,
            threshold=shell.threshold,
            k_star=0.0,
            phase_space_density=0.0,
            derivative_side="physical-right",
        )
    if p1[0] <= 0.0 or p2[0] <= 0.0:
        raise ValueError("nonfuture interior input")

    root_s = math.sqrt(s)
    incoming_masses = shell.incoming_masses
    outgoing_masses = shell.outgoing_masses
    k_star = shell.k_star
    e3 = shell.e3_star
    e4 = shell.e4_star

    boost = _make_boost(p0, pvec, root_s)
    p1_star = boost_to_com(p1, boost)
    e_x, e_y, e_z = _frame_from_incoming(p1_star)
    transverse_squared = 1.0 - z_star * z_star
    if transverse_squared < 0.0:
        raise ValueError("negative angular radicand")
    transverse = math.sqrt(transverse_squared)
    direction = add3(
        scale3(z_star, e_z),
        add3(
            scale3(transverse * math.cos(phi_star), e_x),
            scale3(transverse * math.sin(phi_star), e_y),
        ),
    )
    p3_vec = scale3(k_star, direction)
    p4_vec = scale3(-1.0, p3_vec)
    p3 = boost_from_com((e3, *p3_vec), boost)
    p4 = boost_from_com((e4, *p4_vec), boost)
    dphi2_dz_dphi = k_star / (16.0 * math.pi * math.pi * root_s)

    if p3[0] <= 0.0 or p4[0] <= 0.0:
        raise ArithmeticError("nonpositive outgoing interior energy")
    conservation = tuple((p1[i] + p2[i]) - (p3[i] + p4[i]) for i in range(4))
    conservation_residual = max(abs(value) for value in conservation)
    conservation_scale = max(1.0, *(abs(value) for value in (*p1, *p2, *p3, *p4)))
    if conservation_residual > 64.0 * EPSILON64 * conservation_scale:
        raise ArithmeticError("four-momentum postcondition failed")

    momenta = (p1, p2, p3, p4)
    masses = (*incoming_masses, *outgoing_masses)
    shell_residuals = [abs(minkowski_dot(p, p) + mass * mass) for p, mass in zip(momenta, masses)]
    shell_scale = max(
        1.0,
        *(max(p[0] * p[0], dot3(p[1:], p[1:]), mass * mass) for p, mass in zip(momenta, masses)),
    )
    shell_residual = max(shell_residuals)
    if shell_residual > 128.0 * EPSILON64 * shell_scale:
        raise ArithmeticError("mass-shell postcondition failed")

    p13 = tuple(p1[i] - p3[i] for i in range(4))
    p14 = tuple(p1[i] - p4[i] for i in range(4))
    t = -minkowski_dot(p13, p13)
    u = -minkowski_dot(p14, p14)
    mass_sum = pairwise_sum(mass * mass for mass in masses)
    mandelstam_residual = abs(((s + t) + u) - mass_sum)
    mandelstam_scale = max(1.0, abs(s), abs(t), abs(u), mass_sum)
    if mandelstam_residual > 256.0 * EPSILON64 * mandelstam_scale:
        raise ArithmeticError("Mandelstam postcondition failed")

    return TwoBodyInterior(
        process=process,
        s=s,
        k_star=k_star,
        p3=p3,
        p4=p4,
        phase_space_density=dphi2_dz_dphi,
        boost=boost,
        max_conservation_residual=conservation_residual,
        max_shell_residual=shell_residual,
        mandelstam_residual=mandelstam_residual,
    )


T = TypeVar("T")


def pairwise_sum(values: Iterable[T], add: Callable[[T, T], T] | None = None) -> T:
    """Binding balanced tree: adjacent pairs; promote an odd final item."""
    level = list(values)
    if not level:
        raise ValueError("pairwise_sum requires at least one value")
    op = add or (lambda a, b: a + b)  # type: ignore[operator,return-value]
    while len(level) > 1:
        level = [op(level[i], level[i + 1]) for i in range(0, len(level) - 1, 2)] + (
            [level[-1]] if len(level) % 2 else []
        )
    return level[0]


@dataclass(frozen=True)
class MPInterval:
    lower: object
    upper: object


class DirectedMPFR:
    """Minimal outward MPFR arithmetic used by every A1--A4 trace owner."""

    def __init__(self, precision: int):
        if precision not in (256, 384):
            raise ValueError("metrology precision must be exactly 256 or 384 bits")
        import gmpy2

        self.gmpy2 = gmpy2
        self.precision = precision

    def _evaluate(self, rounding, operation):
        context = self.gmpy2.context()
        context.precision = self.precision
        context.round = rounding
        with self.gmpy2.local_context(context):
            return +operation()

    def _down(self, operation):
        return self._evaluate(self.gmpy2.RoundDown, operation)

    def _up(self, operation):
        return self._evaluate(self.gmpy2.RoundUp, operation)

    def singleton_float(self, value: float) -> MPInterval:
        if not math.isfinite(value):
            raise ValueError("nonfinite binary64 interval input")
        exact = self.gmpy2.mpfr(value)
        return MPInterval(exact, exact)

    def singleton_int(self, value: int) -> MPInterval:
        exact = self.gmpy2.mpfr(value)
        return MPInterval(exact, exact)

    def _checked(self, interval: MPInterval) -> MPInterval:
        if interval.lower > interval.upper:
            raise ArithmeticError("reversed directed interval")
        if not self.gmpy2.is_finite(interval.lower) or not self.gmpy2.is_finite(interval.upper):
            raise ArithmeticError("nonfinite directed interval")
        return interval

    def add(self, left: MPInterval, right: MPInterval) -> MPInterval:
        return self._checked(
            MPInterval(
                self._down(lambda: left.lower + right.lower),
                self._up(lambda: left.upper + right.upper),
            )
        )

    def sub(self, left: MPInterval, right: MPInterval) -> MPInterval:
        return self._checked(
            MPInterval(
                self._down(lambda: left.lower - right.upper),
                self._up(lambda: left.upper - right.lower),
            )
        )

    def mul(self, left: MPInterval, right: MPInterval) -> MPInterval:
        lower_products = [
            self._down(lambda a=a, b=b: a * b)
            for a in (left.lower, left.upper)
            for b in (right.lower, right.upper)
        ]
        upper_products = [
            self._up(lambda a=a, b=b: a * b)
            for a in (left.lower, left.upper)
            for b in (right.lower, right.upper)
        ]
        return self._checked(MPInterval(min(lower_products), max(upper_products)))

    def div(self, numerator: MPInterval, denominator: MPInterval) -> MPInterval:
        if denominator.lower <= 0 <= denominator.upper:
            raise ArithmeticError("directed denominator contains zero")
        reciprocal = MPInterval(
            self._down(lambda: 1 / denominator.upper),
            self._up(lambda: 1 / denominator.lower),
        )
        if reciprocal.lower > reciprocal.upper:
            reciprocal = MPInterval(reciprocal.upper, reciprocal.lower)
        return self.mul(numerator, reciprocal)

    def sqrt(self, value: MPInterval) -> MPInterval:
        if value.lower < 0:
            raise ArithmeticError("directed square root crosses its branch")
        return self._checked(
            MPInterval(
                self._down(lambda: self.gmpy2.sqrt(value.lower)),
                self._up(lambda: self.gmpy2.sqrt(value.upper)),
            )
        )

    def exp(self, value: MPInterval) -> MPInterval:
        return self._checked(
            MPInterval(
                self._down(lambda: self.gmpy2.exp(value.lower)),
                self._up(lambda: self.gmpy2.exp(value.upper)),
            )
        )

    def sin(self, value: MPInterval) -> MPInterval:
        if value.lower != value.upper:
            raise ArithmeticError("sin interval requires a recorded monotone branch")
        return self._checked(
            MPInterval(
                self._down(lambda: self.gmpy2.sin(value.lower)),
                self._up(lambda: self.gmpy2.sin(value.upper)),
            )
        )

    def cos(self, value: MPInterval) -> MPInterval:
        if value.lower != value.upper:
            raise ArithmeticError("cos interval requires a recorded monotone branch")
        return self._checked(
            MPInterval(
                self._down(lambda: self.gmpy2.cos(value.lower)),
                self._up(lambda: self.gmpy2.cos(value.upper)),
            )
        )

    def binary64_interface(self, value: MPInterval) -> MPInterval:
        """Round an interval outward once at a declared binary64 interface."""
        lower64 = float(value.lower)
        upper64 = float(value.upper)
        if self.gmpy2.mpfr(lower64) > value.lower:
            lower64 = math.nextafter(lower64, -math.inf)
        if self.gmpy2.mpfr(upper64) < value.upper:
            upper64 = math.nextafter(upper64, math.inf)
        return self._checked(
            MPInterval(self.gmpy2.mpfr(lower64), self.gmpy2.mpfr(upper64))
        )


def interval_pairwise_sum(values: Iterable[MPInterval], arithmetic: DirectedMPFR) -> MPInterval:
    return pairwise_sum(values, add=arithmetic.add)


def legendre_polynomial_interval(
    order: int, x: MPInterval, arithmetic: DirectedMPFR
) -> tuple[MPInterval, MPInterval]:
    """Return directed P_N and P_(N-1) in the binding recurrence order."""
    if order < 1:
        raise ValueError("Legendre order must be positive")
    one = arithmetic.singleton_int(1)
    if order == 1:
        return x, one
    previous_previous, previous = one, x
    for degree in range(2, order + 1):
        leading = arithmetic.mul(arithmetic.singleton_int(2 * degree - 1), x)
        leading = arithmetic.mul(leading, previous)
        lagging = arithmetic.mul(arithmetic.singleton_int(degree - 1), previous_previous)
        current = arithmetic.div(
            arithmetic.sub(leading, lagging), arithmetic.singleton_int(degree)
        )
        previous_previous, previous = previous, current
    return previous, previous_previous


def enclose_legendre_root(
    order: int, approximate_root: object, precision: int = 256
) -> MPInterval:
    """Bracket one Newton root by directed signs and certify local uniqueness."""
    arithmetic = DirectedMPFR(precision)
    gmpy2 = arithmetic.gmpy2
    centre = arithmetic._evaluate(
        gmpy2.RoundToNearest, lambda: gmpy2.mpfr(approximate_root)
    )
    lower = gmpy2.next_below(centre)
    upper = gmpy2.next_above(centre)
    for _ in range(4096):
        p_lower, _ = legendre_polynomial_interval(
            order, MPInterval(lower, lower), arithmetic
        )
        p_upper, _ = legendre_polynomial_interval(
            order, MPInterval(upper, upper), arithmetic
        )
        opposite = p_lower.upper < 0 < p_upper.lower or p_upper.upper < 0 < p_lower.lower
        if opposite:
            break
        lower = gmpy2.next_below(lower)
        upper = gmpy2.next_above(upper)
    else:
        raise ArithmeticError("failed to sign-bracket Legendre root")

    target_width = arithmetic._evaluate(gmpy2.RoundUp, lambda: gmpy2.exp2(-240))
    while arithmetic._up(lambda: upper - lower) > target_width:
        midpoint = arithmetic._evaluate(
            gmpy2.RoundToNearest, lambda: (lower + upper) / 2
        )
        p_midpoint, _ = legendre_polynomial_interval(
            order, MPInterval(midpoint, midpoint), arithmetic
        )
        p_lower, _ = legendre_polynomial_interval(
            order, MPInterval(lower, lower), arithmetic
        )
        if p_midpoint.lower > 0 and p_lower.lower > 0:
            lower = midpoint
        elif p_midpoint.upper < 0 and p_lower.upper < 0:
            lower = midpoint
        elif p_midpoint.lower > 0 or p_midpoint.upper < 0:
            upper = midpoint
        else:
            raise ArithmeticError("Legendre bisection sign is indeterminate")

    enclosure = MPInterval(lower, upper)
    p_n, p_nm1 = legendre_polynomial_interval(order, enclosure, arithmetic)
    numerator = arithmetic.mul(
        arithmetic.singleton_int(order),
        arithmetic.sub(arithmetic.mul(enclosure, p_n), p_nm1),
    )
    denominator = arithmetic.sub(
        arithmetic.mul(enclosure, enclosure), arithmetic.singleton_int(1)
    )
    derivative = arithmetic.div(numerator, denominator)
    if derivative.lower <= 0 <= derivative.upper:
        raise ArithmeticError("Legendre root uniqueness certificate failed")
    return enclosure


def gauss_legendre_mpfr(
    order: int, lower: float, upper: float, precision: int = 256
) -> tuple[tuple[float, ...], tuple[float, ...]]:
    """Generate and affinely map the binding MPFR GL rule.

    Positive roots are solved in descending-root order from the standard cosine
    seeds.  Newton stops at 2^-220 and the rule is mirrored before a single
    nearest-binary64 conversion.  The affine map is then the exact binary64
    graph ``mid=0.5*(lower+upper)``, ``half=0.5*(upper-lower)``,
    ``node=mid+half*x``, and ``weight=half*w``.
    """
    if (
        order < 1
        or precision < 224
        or not math.isfinite(lower)
        or not math.isfinite(upper)
        or not lower < upper
    ):
        raise ValueError("invalid Gauss-Legendre contract")
    import gmpy2

    context = gmpy2.context()
    context.precision = precision
    context.round = gmpy2.RoundToNearest
    with gmpy2.local_context(context):
        mpfr = gmpy2.mpfr
        one = mpfr(1)
        tolerance = gmpy2.exp2(-220)

        def polynomial_and_previous(x):
            p_nm2, p_nm1 = one, x
            if order == 1:
                return x, one
            for degree in range(2, order + 1):
                p_n = ((2 * degree - 1) * x * p_nm1 - (degree - 1) * p_nm2) / degree
                p_nm2, p_nm1 = p_nm1, p_n
            return p_nm1, p_nm2

        positive: list[tuple[object, object]] = []
        for root_index in range(1, order // 2 + 1):
            x = gmpy2.cos(gmpy2.const_pi() * (4 * root_index - 1) / (4 * order + 2))
            for _ in range(128):
                p_n, p_nm1 = polynomial_and_previous(x)
                derivative = order * (x * p_n - p_nm1) / (x * x - one)
                delta = p_n / derivative
                x -= delta
                if abs(delta) < tolerance:
                    break
            else:
                raise ArithmeticError("MPFR Legendre Newton did not converge")
            p_n, p_nm1 = polynomial_and_previous(x)
            derivative = order * (x * p_n - p_nm1) / (x * x - one)
            weight = 2 / ((one - x * x) * derivative * derivative)
            positive.append((x, weight))

        negative = [(-x, weight) for x, weight in positive]
        entries = negative
        if order % 2:
            x = mpfr(0)
            p_n, p_nm1 = polynomial_and_previous(x)
            derivative = order * (x * p_n - p_nm1) / (x * x - one)
            entries.append((x, 2 / (derivative * derivative)))
        entries.extend(reversed(positive))
        roots64 = tuple(float(x) for x, _ in entries)
        weights64 = tuple(float(weight) for _, weight in entries)

    mid = 0.5 * (lower + upper)
    half = 0.5 * (upper - lower)
    nodes = tuple(mid + half * root for root in roots64)
    weights = tuple(half * weight for weight in weights64)
    return nodes, weights


def _inner(rho: Sequence[float], left: Sequence[float], right: Sequence[float]) -> float:
    return pairwise_sum(((rho[j] * left[j]) * right[j] for j in range(len(rho))))


def _normalized_inner(
    rho: Sequence[float], rho_sum: float, left: Sequence[float], right: Sequence[float]
) -> float:
    return _inner(rho, left, right) / rho_sum


def _polynomial_update(
    candidate: Sequence[float], prior: Sequence[float], coefficient: float
) -> list[float]:
    size = max(len(candidate), len(prior))
    out = [0.0] * size
    for degree in range(size):
        left = candidate[degree] if degree < len(candidate) else 0.0
        right = prior[degree] if degree < len(prior) else 0.0
        out[degree] = left - coefficient * right
    return out


def _evaluate_polynomial(coefficients: Sequence[float], z: float) -> float:
    if not coefficients:
        raise ValueError("empty basis polynomial")
    value = coefficients[-1]
    for coefficient in reversed(coefficients[:-1]):
        value = value * z + coefficient
    return value


@dataclass(frozen=True)
class PolynomialBasis:
    domain_lower: float
    domain_upper: float
    nodes: tuple[float, ...]
    weights: tuple[float, ...]
    rho: tuple[float, ...]
    rho_sum: float
    y_bar: float
    s_y: float
    coefficients: tuple[tuple[float, ...], ...]
    node_values: tuple[tuple[float, ...], ...]

    def evaluate(self, mode: int, y: float) -> float:
        """Evaluate psi_mode at an arbitrary in-domain y by fixed-order Horner."""
        if (
            not 0 <= mode < len(self.coefficients)
            or not math.isfinite(y)
            or not self.domain_lower <= y <= self.domain_upper
        ):
            raise ValueError("invalid basis evaluation")
        z = (y - self.y_bar) / self.s_y
        return _evaluate_polynomial(self.coefficients[mode], z)

    def evaluate_all(self, y: float) -> tuple[float, ...]:
        return tuple(self.evaluate(mode, y) for mode in range(len(self.coefficients)))


def deterministic_weighted_basis(
    nodes: Sequence[float],
    weights: Sequence[float],
    modes: int,
    *,
    lower: float,
    upper: float,
) -> PolynomialBasis:
    """Binding two-pass weighted MGS with fixed signs and no BLAS."""
    if (
        len(nodes) != len(weights)
        or not nodes
        or modes < 2
        or not math.isfinite(lower)
        or not math.isfinite(upper)
        or not lower < upper
        or nodes[0] <= lower
        or nodes[-1] >= upper
    ):
        raise ValueError("invalid basis inputs")
    f0 = [1.0 / (math.exp(y) + 1.0) for y in nodes]
    rho = [
        (((weights[j] * nodes[j]) * nodes[j]) * f0[j]) * (1.0 - f0[j])
        for j in range(len(nodes))
    ]
    rho_sum = pairwise_sum(rho)
    y_bar = pairwise_sum((rho[j] * nodes[j] for j in range(len(nodes)))) / rho_sum
    variance = pairwise_sum(
        ((rho[j] * (nodes[j] - y_bar)) * (nodes[j] - y_bar) for j in range(len(nodes)))
    ) / rho_sum
    s_y = math.sqrt(variance)
    if not s_y > 0.0 or modes > len(nodes):
        raise ValueError("rank-deficient basis request")
    z = [(y - y_bar) / s_y for y in nodes]
    basis_values: list[list[float]] = [[1.0] * len(nodes), list(z)]
    basis_coefficients: list[list[float]] = [[1.0], [0.0, 1.0]]
    powers = list(z)
    for degree in range(2, modes):
        powers = [powers[j] * z[j] for j in range(len(z))]
        candidate_values = list(powers)
        candidate_coefficients = [0.0] * degree + [1.0]
        for _ in range(2):
            for prior_values, prior_coefficients in zip(basis_values, basis_coefficients):
                coefficient = _normalized_inner(
                    rho, rho_sum, prior_values, candidate_values
                ) / _normalized_inner(rho, rho_sum, prior_values, prior_values)
                candidate_values = [
                    candidate_values[j] - coefficient * prior_values[j]
                    for j in range(len(nodes))
                ]
                candidate_coefficients = _polynomial_update(
                    candidate_coefficients, prior_coefficients, coefficient
                )
        norm = math.sqrt(_normalized_inner(rho, rho_sum, candidate_values, candidate_values))
        if not norm > 0.0 or not math.isfinite(norm):
            raise ValueError("weighted QR rank failure")
        candidate_values = [value / norm for value in candidate_values]
        candidate_coefficients = [value / norm for value in candidate_coefficients]
        scale = max(abs(value) for value in candidate_values)
        resolved = next(
            (
                value
                for value in candidate_values
                if abs(value) > 64.0 * EPSILON64 * scale
            ),
            None,
        )
        if resolved is None:
            raise ValueError("basis sign is unresolved")
        if resolved < 0.0:
            candidate_values = [-value for value in candidate_values]
            candidate_coefficients = [-value for value in candidate_coefficients]
        basis_values.append(candidate_values)
        basis_coefficients.append(candidate_coefficients)
    return PolynomialBasis(
        domain_lower=lower,
        domain_upper=upper,
        nodes=tuple(nodes),
        weights=tuple(weights),
        rho=tuple(rho),
        rho_sum=rho_sum,
        y_bar=y_bar,
        s_y=s_y,
        coefficients=tuple(tuple(column) for column in basis_coefficients[:modes]),
        node_values=tuple(tuple(column) for column in basis_values[:modes]),
    )


def project_entropy_profile(
    basis: PolynomialBasis, target_eta: Callable[[float], float]
) -> tuple[float, ...]:
    """Project eta by the binding normalized inner product, without fitting."""
    eta_values = tuple(target_eta(y) for y in basis.nodes)
    if not all(math.isfinite(value) for value in eta_values):
        raise ValueError("nonfinite target entropy profile")
    return tuple(
        _normalized_inner(basis.rho, basis.rho_sum, values, eta_values)
        for values in basis.node_values
    )


def reconstruct_entropy(
    basis: PolynomialBasis, coefficients: Sequence[float], y: float
) -> float:
    if len(coefficients) != len(basis.coefficients):
        raise ValueError("coefficient/basis size mismatch")
    return pairwise_sum(
        coefficient * basis.evaluate(mode, y)
        for mode, coefficient in enumerate(coefficients)
    )


def logistic(eta: float) -> float:
    if not math.isfinite(eta):
        raise ValueError("nonfinite entropy variable")
    if eta >= 0.0:
        return 1.0 / (1.0 + math.exp(-eta))
    exponential = math.exp(eta)
    return exponential / (1.0 + exponential)


@dataclass(frozen=True)
class EventContribution:
    key: tuple[int | str, ...]
    species_index: int
    mode_index: int
    value: float


def reduce_event_contributions(
    contributions: Sequence[EventContribution], modes: int
) -> tuple[tuple[float, ...], ...]:
    """Canonical A0 q reduction; each value already includes its complete event weight."""
    buckets: list[list[list[EventContribution]]] = [
        [[] for _mode in range(modes)] for _species in SPECIES
    ]
    for contribution in contributions:
        if not 0 <= contribution.species_index < len(SPECIES):
            raise ValueError("event contribution has invalid species")
        if not 0 <= contribution.mode_index < modes or not math.isfinite(contribution.value):
            raise ValueError("event contribution has invalid mode/value")
        buckets[contribution.species_index][contribution.mode_index].append(contribution)
    reduced: list[tuple[float, ...]] = []
    for species_index in range(len(SPECIES)):
        row = []
        for mode_index in range(modes):
            ordered = sorted(buckets[species_index][mode_index], key=lambda item: item.key)
            if not ordered:
                raise ValueError("q reduction has an empty species/mode bucket")
            row.append(pairwise_sum(item.value for item in ordered))
        reduced.append(tuple(row))
    return tuple(reduced)


def assemble_mass_matrix(
    basis: PolynomialBasis, coefficients: Sequence[float]
) -> tuple[tuple[float, ...], ...]:
    modes = len(basis.coefficients)
    if len(coefficients) != modes:
        raise ValueError("mass matrix coefficient/basis mismatch")
    occupations = []
    for node_index in range(len(basis.nodes)):
        eta = pairwise_sum(
            coefficients[mode] * basis.node_values[mode][node_index]
            for mode in range(modes)
        )
        occupations.append(logistic(eta))
    matrix = [[0.0] * modes for _ in range(modes)]
    for left in range(modes):
        for right in range(left + 1):
            terms = []
            for node_index, y in enumerate(basis.nodes):
                density = occupations[node_index] * (1.0 - occupations[node_index])
                term = (basis.weights[node_index] * y) * y
                term = term * density
                term = term * basis.node_values[left][node_index]
                term = term * basis.node_values[right][node_index]
                terms.append(term)
            matrix[left][right] = pairwise_sum(terms)
            matrix[right][left] = matrix[left][right]
    return tuple(tuple(row) for row in matrix)


def species_native_observable(
    basis: PolynomialBasis,
    coefficients: Sequence[float],
    q: Sequence[float],
    *,
    gf: float,
    t_cm: float,
) -> tuple[float, ...]:
    """Complete A0 mass/solve/final-reconstruction graph for one species."""
    matrix = assemble_mass_matrix(basis, coefficients)
    beta_dot = serial_cholesky_solve(matrix, q)
    scale = (((gf * gf) * t_cm) * t_cm) * t_cm
    scale = (scale * t_cm) * t_cm
    observable = []
    for node_index in range(len(basis.nodes)):
        eta = pairwise_sum(
            coefficients[mode] * basis.node_values[mode][node_index]
            for mode in range(len(coefficients))
        )
        occupation = logistic(eta)
        reconstruction = pairwise_sum(
            basis.node_values[mode][node_index] * beta_dot[mode]
            for mode in range(len(beta_dot))
        )
        observable.append((occupation * (1.0 - occupation)) * reconstruction / scale)
    return tuple(observable)


def directed_logistic(value: MPInterval, arithmetic: DirectedMPFR) -> MPInterval:
    one = arithmetic.singleton_int(1)
    minus_value = arithmetic.sub(arithmetic.singleton_int(0), value)
    return arithmetic.div(one, arithmetic.add(one, arithmetic.exp(minus_value)))


def directed_q_reduction(
    ordered_contributions: Sequence[Sequence[MPInterval]], arithmetic: DirectedMPFR
) -> tuple[MPInterval, ...]:
    if not ordered_contributions or any(not bucket for bucket in ordered_contributions):
        raise ValueError("directed q reduction has an empty bucket")
    return tuple(interval_pairwise_sum(bucket, arithmetic) for bucket in ordered_contributions)


def _directed_inner(
    rho: Sequence[MPInterval],
    left: Sequence[MPInterval],
    right: Sequence[MPInterval],
    arithmetic: DirectedMPFR,
) -> MPInterval:
    return interval_pairwise_sum(
        (
            arithmetic.mul(arithmetic.mul(rho[index], left[index]), right[index])
            for index in range(len(rho))
        ),
        arithmetic,
    )


def _directed_polynomial_update(
    candidate: Sequence[MPInterval],
    prior: Sequence[MPInterval],
    coefficient: MPInterval,
    arithmetic: DirectedMPFR,
) -> list[MPInterval]:
    zero = arithmetic.singleton_int(0)
    size = max(len(candidate), len(prior))
    out = []
    for degree in range(size):
        left = candidate[degree] if degree < len(candidate) else zero
        right = prior[degree] if degree < len(prior) else zero
        out.append(arithmetic.sub(left, arithmetic.mul(coefficient, right)))
    return out


def _directed_polynomial_evaluate(
    coefficients: Sequence[MPInterval], z: MPInterval, arithmetic: DirectedMPFR
) -> MPInterval:
    if not coefficients:
        raise ValueError("empty directed basis polynomial")
    value = coefficients[-1]
    for coefficient in reversed(coefficients[:-1]):
        value = arithmetic.add(arithmetic.mul(value, z), coefficient)
    return value


@dataclass(frozen=True)
class DirectedBasis:
    y_bar: MPInterval
    s_y: MPInterval
    rho: tuple[MPInterval, ...]
    rho_sum: MPInterval
    coefficients: tuple[tuple[MPInterval, ...], ...]
    node_values: tuple[tuple[MPInterval, ...], ...]

    def evaluate(
        self, mode: int, y: MPInterval, arithmetic: DirectedMPFR
    ) -> MPInterval:
        if not 0 <= mode < len(self.coefficients):
            raise ValueError("invalid directed basis mode")
        z = arithmetic.div(arithmetic.sub(y, self.y_bar), self.s_y)
        return _directed_polynomial_evaluate(self.coefficients[mode], z, arithmetic)


def directed_weighted_basis(
    nodes: Sequence[MPInterval],
    weights: Sequence[MPInterval],
    modes: int,
    arithmetic: DirectedMPFR,
) -> DirectedBasis:
    """A2 outward two-pass MGS, including off-grid polynomial coefficients."""
    if len(nodes) != len(weights) or not nodes or modes < 2 or modes > len(nodes):
        raise ValueError("invalid directed basis inputs")
    one = arithmetic.singleton_int(1)
    zero = arithmetic.singleton_int(0)
    f0 = [
        arithmetic.div(one, arithmetic.add(arithmetic.exp(node), one)) for node in nodes
    ]
    rho = []
    for node, weight, occupation in zip(nodes, weights, f0):
        value = arithmetic.mul(weight, node)
        value = arithmetic.mul(value, node)
        value = arithmetic.mul(value, occupation)
        value = arithmetic.mul(value, arithmetic.sub(one, occupation))
        rho.append(value)
    rho_sum = interval_pairwise_sum(rho, arithmetic)
    y_bar = arithmetic.div(
        interval_pairwise_sum(
            (arithmetic.mul(rho[index], nodes[index]) for index in range(len(nodes))),
            arithmetic,
        ),
        rho_sum,
    )
    differences = [arithmetic.sub(node, y_bar) for node in nodes]
    variance = arithmetic.div(
        interval_pairwise_sum(
            (
                arithmetic.mul(arithmetic.mul(rho[index], differences[index]), differences[index])
                for index in range(len(nodes))
            ),
            arithmetic,
        ),
        rho_sum,
    )
    s_y = arithmetic.sqrt(variance)
    z = [arithmetic.div(difference, s_y) for difference in differences]
    basis_values: list[list[MPInterval]] = [
        [one for _node in nodes],
        list(z),
    ]
    basis_coefficients: list[list[MPInterval]] = [[one], [zero, one]]
    powers = list(z)
    for degree in range(2, modes):
        powers = [arithmetic.mul(powers[index], z[index]) for index in range(len(z))]
        candidate_values = list(powers)
        candidate_coefficients = [zero for _lower in range(degree)] + [one]
        for _pass in range(2):
            for prior_values, prior_coefficients in zip(
                basis_values, basis_coefficients
            ):
                numerator = arithmetic.div(
                    _directed_inner(rho, prior_values, candidate_values, arithmetic),
                    rho_sum,
                )
                denominator = arithmetic.div(
                    _directed_inner(rho, prior_values, prior_values, arithmetic),
                    rho_sum,
                )
                coefficient = arithmetic.div(numerator, denominator)
                candidate_values = [
                    arithmetic.sub(
                        candidate_values[index],
                        arithmetic.mul(coefficient, prior_values[index]),
                    )
                    for index in range(len(nodes))
                ]
                candidate_coefficients = _directed_polynomial_update(
                    candidate_coefficients,
                    prior_coefficients,
                    coefficient,
                    arithmetic,
                )
        norm = arithmetic.sqrt(
            arithmetic.div(
                _directed_inner(rho, candidate_values, candidate_values, arithmetic),
                rho_sum,
            )
        )
        candidate_values = [arithmetic.div(value, norm) for value in candidate_values]
        candidate_coefficients = [
            arithmetic.div(value, norm) for value in candidate_coefficients
        ]
        scale = max(
            max(abs(value.lower), abs(value.upper)) for value in candidate_values
        )
        threshold = arithmetic.gmpy2.mpfr(64.0 * EPSILON64) * scale
        sign = None
        for value in candidate_values:
            if value.lower > threshold:
                sign = 1
                break
            if value.upper < -threshold:
                sign = -1
                break
        if sign is None:
            raise ArithmeticError("directed basis sign is unresolved")
        if sign < 0:
            candidate_values = [arithmetic.sub(zero, value) for value in candidate_values]
            candidate_coefficients = [
                arithmetic.sub(zero, value) for value in candidate_coefficients
            ]
        basis_values.append(candidate_values)
        basis_coefficients.append(candidate_coefficients)
    return DirectedBasis(
        y_bar=y_bar,
        s_y=s_y,
        rho=tuple(rho),
        rho_sum=rho_sum,
        coefficients=tuple(tuple(column) for column in basis_coefficients[:modes]),
        node_values=tuple(tuple(column) for column in basis_values[:modes]),
    )


def directed_mass_matrix(
    nodes: Sequence[MPInterval],
    weights: Sequence[MPInterval],
    basis_values: Sequence[Sequence[MPInterval]],
    coefficients: Sequence[MPInterval],
    arithmetic: DirectedMPFR,
) -> tuple[tuple[MPInterval, ...], ...]:
    modes = len(coefficients)
    if (
        len(nodes) != len(weights)
        or not nodes
        or len(basis_values) != modes
        or any(len(column) != len(nodes) for column in basis_values)
    ):
        raise ValueError("directed mass matrix shape mismatch")
    occupations = []
    for node_index in range(len(nodes)):
        eta = interval_pairwise_sum(
            (
                arithmetic.mul(coefficients[mode], basis_values[mode][node_index])
                for mode in range(modes)
            ),
            arithmetic,
        )
        occupations.append(directed_logistic(eta, arithmetic))
    zero = arithmetic.singleton_int(0)
    matrix = [[zero for _right in range(modes)] for _left in range(modes)]
    one = arithmetic.singleton_int(1)
    for left in range(modes):
        for right in range(left + 1):
            terms = []
            for node_index in range(len(nodes)):
                density = arithmetic.mul(
                    occupations[node_index], arithmetic.sub(one, occupations[node_index])
                )
                term = arithmetic.mul(weights[node_index], nodes[node_index])
                term = arithmetic.mul(term, nodes[node_index])
                term = arithmetic.mul(term, density)
                term = arithmetic.mul(term, basis_values[left][node_index])
                term = arithmetic.mul(term, basis_values[right][node_index])
                terms.append(term)
            matrix[left][right] = interval_pairwise_sum(terms, arithmetic)
            matrix[right][left] = matrix[left][right]
    return tuple(tuple(row) for row in matrix)


def directed_cholesky_solve(
    matrix: Sequence[Sequence[MPInterval]],
    rhs: Sequence[MPInterval],
    arithmetic: DirectedMPFR,
) -> tuple[MPInterval, ...]:
    n = len(rhs)
    if len(matrix) != n or any(len(row) != n for row in matrix):
        raise ValueError("directed Cholesky shape mismatch")
    zero = arithmetic.singleton_int(0)
    lower = [[zero for _column in range(n)] for _row in range(n)]
    for i in range(n):
        for j in range(i + 1):
            products = [arithmetic.mul(lower[i][k], lower[j][k]) for k in range(j)]
            subtotal = interval_pairwise_sum(products, arithmetic) if products else zero
            residual = arithmetic.sub(matrix[i][j], subtotal)
            if i == j:
                if residual.lower <= 0:
                    raise ArithmeticError("directed Cholesky pivot is not strictly positive")
                lower[i][j] = arithmetic.sqrt(residual)
            else:
                lower[i][j] = arithmetic.div(residual, lower[j][j])
    forward = [zero for _index in range(n)]
    for i in range(n):
        products = [arithmetic.mul(lower[i][k], forward[k]) for k in range(i)]
        subtotal = interval_pairwise_sum(products, arithmetic) if products else zero
        forward[i] = arithmetic.div(arithmetic.sub(rhs[i], subtotal), lower[i][i])
    solution = [zero for _index in range(n)]
    for i in range(n - 1, -1, -1):
        products = [
            arithmetic.mul(lower[k][i], solution[k]) for k in range(i + 1, n)
        ]
        subtotal = interval_pairwise_sum(products, arithmetic) if products else zero
        solution[i] = arithmetic.div(arithmetic.sub(forward[i], subtotal), lower[i][i])
    return tuple(solution)


def directed_cond2_upper(
    matrix: Sequence[Sequence[MPInterval]], arithmetic: DirectedMPFR
) -> MPInterval:
    """Rigorous symmetric norm bound kappa_2 <= ||M||_inf ||M^-1||_inf."""
    n = len(matrix)
    if n == 0 or any(len(row) != n for row in matrix):
        raise ValueError("condition certificate matrix shape mismatch")

    def absolute_upper(value: MPInterval):
        return max(abs(value.lower), abs(value.upper))

    matrix_row_sums = [
        arithmetic._up(
            lambda row=row: sum(
                (absolute_upper(value) for value in row),
                arithmetic.gmpy2.mpfr(0),
            )
        )
        for row in matrix
    ]
    matrix_norm_upper = max(matrix_row_sums)

    zero = arithmetic.singleton_int(0)
    one = arithmetic.singleton_int(1)
    inverse_columns = []
    for column in range(n):
        rhs = [zero for _row in range(n)]
        rhs[column] = one
        inverse_columns.append(directed_cholesky_solve(matrix, rhs, arithmetic))
    inverse_row_sums = []
    for row in range(n):
        inverse_row_sums.append(
            arithmetic._up(
                lambda row=row: sum(
                    (
                        absolute_upper(inverse_columns[column][row])
                        for column in range(n)
                    ),
                    arithmetic.gmpy2.mpfr(0),
                )
            )
        )
    inverse_norm_upper = max(inverse_row_sums)
    upper = arithmetic._up(lambda: matrix_norm_upper * inverse_norm_upper)
    if not arithmetic.gmpy2.is_finite(upper):
        raise ArithmeticError("nonfinite condition-number certificate")
    return MPInterval(arithmetic.gmpy2.mpfr(0), upper)


def directed_species_observable(
    nodes: Sequence[MPInterval],
    weights: Sequence[MPInterval],
    basis_values: Sequence[Sequence[MPInterval]],
    coefficients: Sequence[MPInterval],
    q: Sequence[MPInterval],
    *,
    gf: MPInterval,
    t_cm: MPInterval,
    arithmetic: DirectedMPFR,
) -> tuple[MPInterval, ...]:
    """Complete A3/A4 mass/q/solve/O interval graph for one species."""
    matrix = directed_mass_matrix(
        nodes, weights, basis_values, coefficients, arithmetic
    )
    beta_dot = directed_cholesky_solve(matrix, q, arithmetic)
    scale = arithmetic.mul(gf, gf)
    for _power in range(5):
        scale = arithmetic.mul(scale, t_cm)
    one = arithmetic.singleton_int(1)
    observable = []
    for node_index in range(len(nodes)):
        eta = interval_pairwise_sum(
            (
                arithmetic.mul(coefficients[mode], basis_values[mode][node_index])
                for mode in range(len(coefficients))
            ),
            arithmetic,
        )
        occupation = directed_logistic(eta, arithmetic)
        reconstruction = interval_pairwise_sum(
            (
                arithmetic.mul(basis_values[mode][node_index], beta_dot[mode])
                for mode in range(len(beta_dot))
            ),
            arithmetic,
        )
        numerator = arithmetic.mul(
            arithmetic.mul(occupation, arithmetic.sub(one, occupation)), reconstruction
        )
        observable.append(arithmetic.div(numerator, scale))
    return tuple(observable)


def serial_cholesky_solve(matrix: Sequence[Sequence[float]], rhs: Sequence[float]) -> tuple[float, ...]:
    """Binding lower-Cholesky/forward/back solve with explicit loop order."""
    n = len(rhs)
    lower = [[0.0] * n for _ in range(n)]
    for i in range(n):
        for j in range(i + 1):
            subtotal = pairwise_sum((lower[i][k] * lower[j][k] for k in range(j))) if j else 0.0
            residual = matrix[i][j] - subtotal
            if i == j:
                if not residual > 0.0:
                    raise ValueError("nonpositive Cholesky pivot")
                lower[i][j] = math.sqrt(residual)
            else:
                lower[i][j] = residual / lower[j][j]
    forward = [0.0] * n
    for i in range(n):
        subtotal = pairwise_sum((lower[i][k] * forward[k] for k in range(i))) if i else 0.0
        forward[i] = (rhs[i] - subtotal) / lower[i][i]
    solution = [0.0] * n
    for i in range(n - 1, -1, -1):
        subtotal = pairwise_sum((lower[k][i] * solution[k] for k in range(i + 1, n))) if i + 1 < n else 0.0
        solution[i] = (forward[i] - subtotal) / lower[i][i]
    return tuple(solution)


def splitmix64_step(state: int) -> tuple[int, int]:
    state = (state + 0x9E3779B97F4A7C15) & MASK64
    z = state
    z = ((z ^ (z >> 30)) * 0xBF58476D1CE4E5B9) & MASK64
    z = ((z ^ (z >> 27)) * 0x94D049BB133111EB) & MASK64
    z ^= z >> 31
    return state, z & MASK64


def adversarial_coefficients(modes: int):
    """Yield the infinite deterministic stream; §9 accepts its first 100 valid rows."""
    state = ENTROPY_SEED
    while True:
        row: list[float] = []
        for _species in SPECIES:
            for _mode in range(modes):
                state, z = splitmix64_step(state)
                u = ((z >> 11) + 0.5) * 2.0**-53
                row.append((2.0 * u - 1.0) / 4.0)
        yield tuple(row)


@dataclass(frozen=True, order=True)
class LadderRow:
    cost: int
    n_q: int
    y_max: int
    modes: int


LADDER_ROWS = tuple(
    sorted(
        (
            LadderRow(n_q * modes, n_q, y_max, modes)
            for modes in (4, 8, 12, 16, 24)
            for n_q, y_max in ((48, 24), (64, 28), (80, 32), (96, 40))
        )
    )
)

SPECTRAL_L1_CAP = 2.5e-4
POINTWISE_CAP = 1.0e-2
POINTWISE_FLOOR = 1.0e-8
COND2_CAP = 1.0e6
COLLISION_REFINEMENT_CAP = 5.0e-3
LOST_TAIL_CAP = 2.5e-3
STRUCTURAL_COVARIANCE_CAP = 1.0e-10
NATIVE_FLOOR = 2.0**-40


@dataclass(frozen=True)
class StageAMetrics:
    cond2_upper: float
    spectral_l1_upper: float
    pointwise_relative_max: float

    @property
    def passed(self) -> bool:
        return (
            math.isfinite(self.cond2_upper)
            and math.isfinite(self.spectral_l1_upper)
            and math.isfinite(self.pointwise_relative_max)
            and self.cond2_upper <= COND2_CAP
            and self.spectral_l1_upper <= SPECTRAL_L1_CAP
            and self.pointwise_relative_max <= POINTWISE_CAP
        )


@dataclass(frozen=True)
class StageBMetrics:
    collision_refinement: float
    lost_tail: float

    @property
    def passed(self) -> bool:
        return (
            math.isfinite(self.collision_refinement)
            and math.isfinite(self.lost_tail)
            and self.collision_refinement <= COLLISION_REFINEMENT_CAP
            and self.lost_tail <= LOST_TAIL_CAP
        )


def spectral_l1_metric(
    nodes: Sequence[float],
    weights: Sequence[float],
    reconstructed: Sequence[float],
    target: Sequence[float],
    *,
    target_tail_lower: float,
    target_tail_upper: float,
) -> float:
    """Upper endpoint of the frozen zero-extension weighted spectral L1 metric."""
    if not (len(nodes) == len(weights) == len(reconstructed) == len(target)):
        raise ValueError("spectral metric shape mismatch")
    finite_numerator = pairwise_sum(
        (((weights[j] * nodes[j]) * nodes[j]) * abs(reconstructed[j] - target[j]))
        for j in range(len(nodes))
    )
    finite_denominator = pairwise_sum(
        (((weights[j] * nodes[j]) * nodes[j]) * target[j]) for j in range(len(nodes))
    )
    if target_tail_lower < 0.0 or target_tail_upper < target_tail_lower:
        raise ValueError("invalid directed tail enclosure")
    denominator_lower = finite_denominator + target_tail_lower
    if not denominator_lower > 0.0:
        raise ValueError("nonpositive spectral L1 denominator")
    return (finite_numerator + target_tail_upper) / denominator_lower


def pointwise_relative_metric(
    reconstructed: Sequence[float], target: Sequence[float]
) -> float:
    if len(reconstructed) != len(target):
        raise ValueError("pointwise metric shape mismatch")
    resolved = [
        abs(reconstructed[j] - target[j]) / target[j]
        for j in range(len(target))
        if target[j] > POINTWISE_FLOOR
    ]
    if not resolved:
        raise ValueError("pointwise metric has no resolved target")
    return max(resolved)


def collision_refinement_metrics(
    q_base: Sequence[float],
    q_reference: Sequence[float],
    q_lost_tail_upper: Sequence[float],
    *,
    gf: float,
    t_cm: float,
) -> StageBMetrics:
    if not (len(q_base) == len(q_reference) == len(q_lost_tail_upper)):
        raise ValueError("collision metric shape mismatch")
    scale = (((gf * gf) * t_cm) * t_cm) * t_cm
    scale = (scale * t_cm) * t_cm
    denominator = max(max(abs(value) for value in q_reference), NATIVE_FLOOR * scale)
    collision = max(abs(q_base[j] - q_reference[j]) for j in range(len(q_base))) / denominator
    lost_tail = max(abs(value) for value in q_lost_tail_upper) / denominator
    return StageBMetrics(collision_refinement=collision, lost_tail=lost_tail)


def select_ladder_row(
    stage_a: Mapping[LadderRow, StageAMetrics],
    stage_b: Mapping[LadderRow, StageBMetrics],
) -> LadderRow | None:
    """Evaluate the sealed 20-row ledger and select the first eligible passing row."""
    if set(stage_a) != set(LADDER_ROWS):
        raise ValueError("Stage A must seal exactly the full 20-row ladder")
    eligible = tuple(row for row in LADDER_ROWS if stage_a[row].passed)
    if set(stage_b) != set(eligible):
        raise ValueError("Stage B must seal exactly every Stage-A-eligible row")
    return next((row for row in eligible if stage_b[row].passed), None)


TRACE_OWNERS = (
    "input",
    "event_primitive",
    "canonical_reduction",
    "gl_root",
    "affine_rule",
    "weighted_qr",
    "basis_evaluation",
    "reconstruction",
    "mass_assembly",
    "cholesky",
    "substitution",
    "final_reconstruction",
)

STAGE_POLICIES = {
    "A0": {
        "precision": 53,
        "directed_owners": (),
        "interface": "every operation is the binding binary64 operation; no FMA",
    },
    "A1": {
        "precision": 256,
        "directed_owners": ("event_primitive", "canonical_reduction"),
        "interface": "round once to binary64 at each complete event contribution and pairwise-reduction output",
    },
    "A2": {
        "precision": 256,
        "directed_owners": (
            "event_primitive",
            "canonical_reduction",
            "gl_root",
            "affine_rule",
            "weighted_qr",
            "basis_evaluation",
            "reconstruction",
        ),
        "interface": "round once at the completed basis object, projected state, and q-vector interfaces",
    },
    "A3": {
        "precision": 256,
        "directed_owners": TRACE_OWNERS[1:],
        "interface": "no intermediate binary64 rounding; round only the final O interval for reporting",
    },
    "A4": {
        "precision": 384,
        "directed_owners": TRACE_OWNERS[1:],
        "interface": "import every A3 binary64 input as an outward 384-bit singleton; no intermediate binary64 rounding",
    },
}

OPCODE_SEMANTICS = {
    "input": "import one recorded binary64 input exactly; no operands",
    "add": "left+right",
    "sub": "left-right",
    "mul": "left*right",
    "div": "left/right; denominator interval containing zero is raw failure",
    "sqrt": "principal nonnegative root; negative or boundary-crossing input is raw failure",
    "exp": "MPFR exp with directed lower/upper rounding",
    "sin": "MPFR sin on the recorded branch with directed lower/upper rounding",
    "cos": "MPFR cos on the recorded branch with directed lower/upper rounding",
    "pairwise_add": "adjacent left-to-right pairs; promote odd final item unchanged",
    "round_binary64": "nearest, ties-to-even, one declared interface only",
    "branch": "record predicate and selected arm; an interval crossing the predicate is raw failure",
}


@dataclass(frozen=True)
class ArithmeticNode:
    node_id: int
    owner: str
    opcode: str
    inputs: tuple[int, ...]
    loop_key: tuple[int | str, ...]
    declared_interface: bool = False


def validate_arithmetic_trace(
    nodes: Sequence[ArithmeticNode], final_outputs: Sequence[int]
) -> Mapping[str, int]:
    """Fail closed unless a dynamic A0 trace is complete and topologically canonical."""
    if not nodes or tuple(node.node_id for node in nodes) != tuple(range(len(nodes))):
        raise ValueError("trace node IDs must be contiguous in execution order")
    counts = {owner: 0 for owner in TRACE_OWNERS}
    previous_key: tuple[int | str, ...] | None = None
    for node in nodes:
        if node.owner not in counts or node.opcode not in OPCODE_SEMANTICS:
            raise ValueError("unknown trace owner or opcode")
        if any(source < 0 or source >= node.node_id for source in node.inputs):
            raise ValueError("trace is not topological")
        if (node.owner == "input") != (node.opcode == "input") or (
            node.owner == "input" and node.inputs
        ):
            raise ValueError("trace input node contract is invalid")
        if previous_key is not None and node.loop_key < previous_key:
            raise ValueError("trace loop keys are not canonical")
        previous_key = node.loop_key
        counts[node.owner] += 1
    if any(output < 0 or output >= len(nodes) for output in final_outputs):
        raise ValueError("trace references an invalid final output")
    required = set(TRACE_OWNERS) - {"input"}
    if any(counts[owner] == 0 for owner in required):
        raise ValueError("trace omits a binding arithmetic owner")
    return counts


def _apply_directed_opcode(
    opcode: str, operands: Sequence[MPInterval], arithmetic: DirectedMPFR
) -> MPInterval:
    if opcode == "add":
        return arithmetic.add(operands[0], operands[1])
    if opcode == "sub":
        return arithmetic.sub(operands[0], operands[1])
    if opcode == "mul":
        return arithmetic.mul(operands[0], operands[1])
    if opcode == "div":
        return arithmetic.div(operands[0], operands[1])
    if opcode == "sqrt":
        return arithmetic.sqrt(operands[0])
    if opcode == "exp":
        return arithmetic.exp(operands[0])
    if opcode == "sin":
        return arithmetic.sin(operands[0])
    if opcode == "cos":
        return arithmetic.cos(operands[0])
    if opcode == "pairwise_add":
        return interval_pairwise_sum(operands, arithmetic)
    if opcode == "round_binary64":
        return arithmetic.binary64_interface(operands[0])
    if opcode == "branch":
        if len(operands) != 3:
            raise ValueError("branch requires predicate, true value, and false value")
        if operands[0].lower > 0:
            return operands[1]
        if operands[0].upper < 0:
            return operands[2]
        raise ArithmeticError("directed branch predicate is indeterminate")
    raise ValueError(f"unsupported directed opcode {opcode!r}")


def _apply_binary64_opcode(
    opcode: str, operands: Sequence[MPInterval], arithmetic: DirectedMPFR
) -> MPInterval:
    rounded = [arithmetic.binary64_interface(value) for value in operands]
    endpoints = [
        (float(value.lower), float(value.upper)) for value in rounded
    ]
    if opcode in ("add", "sub", "mul", "div"):
        if len(endpoints) != 2:
            raise ValueError("binary opcode requires two operands")
        if opcode == "div" and endpoints[1][0] <= 0.0 <= endpoints[1][1]:
            raise ArithmeticError("binary64 denominator interval contains zero")
        operation = {
            "add": lambda left, right: left + right,
            "sub": lambda left, right: left - right,
            "mul": lambda left, right: left * right,
            "div": lambda left, right: left / right,
        }[opcode]
        candidates = [
            operation(left, right)
            for left in endpoints[0]
            for right in endpoints[1]
        ]
    elif opcode in ("sqrt", "exp", "sin", "cos"):
        if len(endpoints) != 1:
            raise ValueError("unary opcode requires one operand")
        lower, upper = endpoints[0]
        if opcode == "sqrt":
            if lower < 0.0:
                raise ArithmeticError("binary64 square-root interval crosses branch")
            candidates = [math.sqrt(lower), math.sqrt(upper)]
        elif opcode == "exp":
            candidates = [math.exp(lower), math.exp(upper)]
        else:
            if lower != upper:
                raise ArithmeticError("binary64 trig interval requires recorded branch split")
            function = math.sin if opcode == "sin" else math.cos
            candidates = [function(lower)]
    elif opcode == "pairwise_add":
        return pairwise_sum(
            rounded, add=lambda left, right: _apply_binary64_opcode(
                "add", (left, right), arithmetic
            )
        )
    elif opcode == "round_binary64":
        if len(rounded) != 1:
            raise ValueError("round interface requires one operand")
        return rounded[0]
    elif opcode == "branch":
        if len(rounded) != 3:
            raise ValueError("branch requires predicate, true value, and false value")
        if rounded[0].lower > 0:
            return rounded[1]
        if rounded[0].upper < 0:
            return rounded[2]
        raise ArithmeticError("binary64 branch predicate is indeterminate")
    else:
        raise ValueError(f"unsupported binary64 opcode {opcode!r}")
    if not candidates or not all(math.isfinite(value) for value in candidates):
        raise ArithmeticError("nonfinite binary64 trace result")
    return MPInterval(
        arithmetic.gmpy2.mpfr(min(candidates)),
        arithmetic.gmpy2.mpfr(max(candidates)),
    )


def replay_arithmetic_trace(
    nodes: Sequence[ArithmeticNode],
    final_outputs: Sequence[int],
    input_values: Mapping[int, float],
    stage: Literal["A0", "A1", "A2", "A3", "A4"],
) -> tuple[tuple[MPInterval, ...], Mapping[str, int]]:
    """Execute the same dynamic trace under one frozen stage policy."""
    counts = validate_arithmetic_trace(nodes, final_outputs)
    policy = STAGE_POLICIES[stage]
    precision = 384 if stage == "A4" else 256
    arithmetic = DirectedMPFR(precision)
    directed_owners = set(policy["directed_owners"])
    if set(input_values) != {node.node_id for node in nodes if node.owner == "input"}:
        raise ValueError("trace inputs do not exactly match input nodes")

    consumers: dict[int, list[ArithmeticNode]] = {node.node_id: [] for node in nodes}
    for node in nodes:
        for source in node.inputs:
            consumers[source].append(node)
    for node in nodes:
        if node.owner in directed_owners and any(
            consumer.owner not in directed_owners for consumer in consumers[node.node_id]
        ) and not node.declared_interface:
            raise ValueError("directed-to-binary stage boundary lacks a declared interface")

    values: list[MPInterval] = []
    for node in nodes:
        if node.owner == "input":
            value = arithmetic.singleton_float(input_values[node.node_id])
        else:
            operands = tuple(values[source] for source in node.inputs)
            if node.owner in directed_owners:
                value = _apply_directed_opcode(node.opcode, operands, arithmetic)
            else:
                value = _apply_binary64_opcode(node.opcode, operands, arithmetic)
            if node.declared_interface and stage in ("A1", "A2"):
                value = arithmetic.binary64_interface(value)
        values.append(value)
    return tuple(values[index] for index in final_outputs), counts


DIRECTED_ROOT_CONTRACT = {
    "seed": "cos(pi*(4*i-1)/(4*N+2)), i increasing over positive roots",
    "newton": "256-bit nearest iterations until abs(delta)<2^-220, maximum 128",
    "enclosure": "expand adjacent 256-bit MPFR numbers around the Newton iterate until directed P_N endpoints have opposite signs; bisect left-to-right until width<=2^-240",
    "uniqueness": "directed P_N derivative interval must exclude zero on the enclosure",
    "mapping": "directed mid=0.5*(lo+hi), half=0.5*(hi-lo), node=mid+half*x, weight=half*w",
    "failure": "no sign bracket, derivative interval containing zero, or branch ambiguity is raw failure",
}

COND2_CERTIFICATE = {
    "matrix": "directed symmetric mass interval after canonical pairwise assembly",
    "inverse": "solve all canonical unit columns by the same directed Cholesky and substitutions",
    "bound": "for symmetric M and inverse(M), kappa_2 <= ||M||_infinity ||inverse(M)||_infinity; every row sum uses directed upper endpoints",
    "decision": "any nonpositive directed Cholesky pivot is raw failure; pass only if the outward cond2_upper<=1e6",
}

TAIL_CONTRACT = {
    "map": "z=b+(1+u)/(1-u), u in [-1,1); b=Y_max for analytic neutrino tails and b=40 for electron tails; adversarial states are zero-extended",
    "integrator": "directed 256-bit interval Gauss-Legendre-32 on deterministic left-first bisection",
    "stop": "sum of active interval widths <= 2^-180 times max(total upper,2^-200), with at most 2^18 leaves",
    "endpoint": "electron f<=exp(-z); named neutrino f<=exp(-z+0.10*(1+z^2)*exp(-z/6)) on the final u interval adjacent to 1",
    "failure": "nonfinite endpoint, leaf cap, or unmet width criterion is raw failure",
}


IntervalEndpoint = tuple[float, float]


def _interval_sup_abs(interval: IntervalEndpoint) -> float:
    lower, upper = interval
    if not (math.isfinite(lower) and math.isfinite(upper) and lower <= upper):
        raise ValueError("invalid interval endpoint")
    return max(abs(lower), abs(upper))


def _interval_difference_sup(left: IntervalEndpoint, right: IntervalEndpoint) -> float:
    return max(abs(left[0] - right[1]), abs(left[1] - right[0]))


@dataclass(frozen=True)
class MetrologyBounds:
    denominator: float
    b_sum: float
    b_basis: float
    b_solve: float
    b_interval: float
    b_native: float


def metrology_bounds(
    stages: Sequence[Sequence[IntervalEndpoint]],
) -> MetrologyBounds:
    """Compute all four terms on one beta/P-beta component domain and denominator."""
    if len(stages) != 5 or not stages[0] or any(len(stage) != len(stages[0]) for stage in stages):
        raise ValueError("A0--A4 stage shape mismatch")
    denominator = max(
        NATIVE_FLOOR,
        *(_interval_sup_abs(value) for value in stages[0]),
        *(_interval_sup_abs(value) for value in stages[4]),
    )
    b_sum = max(
        _interval_difference_sup(left, right) for left, right in zip(stages[0], stages[1])
    ) / denominator
    b_basis = max(
        _interval_difference_sup(left, right) for left, right in zip(stages[1], stages[2])
    ) / denominator
    b_solve = max(
        _interval_difference_sup(left, right) for left, right in zip(stages[2], stages[3])
    ) / denominator

    interval_terms = []
    for a3, a4 in zip(stages[3], stages[4]):
        mid3 = 0.5 * (a3[0] + a3[1])
        mid4 = 0.5 * (a4[0] + a4[1])
        radius3 = 0.5 * (a3[1] - a3[0])
        radius4 = 0.5 * (a4[1] - a4[0])
        interval_terms.append((radius3 + radius4) + abs(mid3 - mid4))
    b_interval = max(interval_terms) / denominator
    b_native = 4.0 * (((b_sum + b_basis) + b_solve) + b_interval)
    return MetrologyBounds(
        denominator=denominator,
        b_sum=b_sum,
        b_basis=b_basis,
        b_solve=b_solve,
        b_interval=b_interval,
        b_native=b_native,
    )


def apply_permutation(values: Sequence[float], permutation: Sequence[int]) -> tuple[float, ...]:
    if sorted(permutation) != list(range(len(values))):
        raise ValueError("permutation is not bijective")
    return tuple(values[permutation[index]] for index in range(len(values)))


def native_permutation_defect(
    observable_beta: Sequence[float],
    observable_p_beta: Sequence[float],
    permutation: Sequence[int],
    *,
    denominator: float,
) -> float:
    if len(observable_beta) != len(observable_p_beta) or denominator < NATIVE_FLOOR:
        raise ValueError("invalid native-defect inputs")
    permuted = apply_permutation(observable_beta, permutation)
    return max(
        abs(permuted[index] - observable_p_beta[index]) for index in range(len(permuted))
    ) / denominator


def structural_covariance_residual(
    value_beta: Sequence[float],
    value_p_beta: Sequence[float],
    permutation: Sequence[int],
    *,
    dimensional_floor: float,
) -> float:
    if len(value_beta) != len(value_p_beta) or dimensional_floor <= 0.0:
        raise ValueError("invalid structural-covariance inputs")
    denominator = max(
        max(abs(value) for value in value_beta),
        max(abs(value) for value in value_p_beta),
        dimensional_floor,
    )
    permuted = apply_permutation(value_beta, permutation)
    return max(
        abs(permuted[index] - value_p_beta[index]) for index in range(len(permuted))
    ) / denominator


EXACT_TEST_VECTORS = {
    "V-CONSTANTS-D035": {
        "G_F_MeV^-2": "1.1663788e-11",
        "m_e_MeV": "0.5109989500",
        "sin2_theta_W": "0.23122",
        "expected": "exact decimal strings match Tier-0; no fitted or alternate literal",
    },
    "V-ORIENTATION-LEDGER": {
        "elastic_distinct": {
            "orientation": "1/2",
            "symmetry": "1",
            "global_leg_factor": "2",
            "effective_tagged_factor": "1",
            "global_amplitude_coefficient": "32",
            "effective_tagged_coefficient": "32",
        },
        "elastic_identical": {
            "orientation": "1/2",
            "symmetry": "1/4",
            "global_leg_factor": "4",
            "effective_tagged_factor": "1/2",
            "global_amplitude_coefficient": "128",
            "effective_tagged_coefficient": "64",
        },
        "conversion_or_pair": {
            "orientation": "1",
            "reason": "initial and final unordered species multisets differ",
        },
    },
    "V-ELASTIC-THRESHOLD": {
        "process": "elastic",
        "m_e": "1",
        "s_binary64_hex": [
            "0x1.fffffffffffffp-1",
            "0x1.0000000000000p+0",
            "0x1.0000000000001p+0",
        ],
        "expected": ["raw subthreshold failure", "support-only k*=0", "interior k*>0"],
    },
    "V-PAIR-THRESHOLD": {
        "process": "pair",
        "m_e": "1",
        "s_binary64_hex": [
            "0x1.fffffffffffffp+1",
            "0x1.0000000000000p+2",
            "0x1.0000000000001p+2",
        ],
        "expected": ["raw subthreshold failure", "support-only k*=0", "interior k*>0"],
    },
    "V-ZSTAR-DOMAIN": {
        "accepted": ["-1", "1"],
        "rejected_binary64_hex": ["-0x1.0000000000001p+0", "0x1.0000000000001p+0"],
        "expected": "no clipping and no max(0,1-z*^2)",
    },
    "V-NULL-COM-345": {
        "masses": ["0", "0", "0", "0"],
        "momenta": [["1", "0", "0", "1"], ["1", "0", "0", "-1"], ["1", "3/5", "0", "4/5"], ["1", "-3/5", "0", "-4/5"]],
        "expected": {"K_s": "4", "K_t": "81/25", "K_u": "1/25"},
    },
    "V-ELASTIC-ME-345": {
        "m_e": "1",
        "masses": ["0", "1", "0", "1"],
        "momenta": [["3/4", "0", "0", "3/4"], ["5/4", "0", "0", "-3/4"], ["3/4", "9/20", "0", "3/5"], ["5/4", "-9/20", "0", "-3/5"]],
        "expected": "positive shells; conserved; s+t+u=2",
    },
    "V-PAIR-ME-345": {
        "m_e": "1",
        "masses": ["0", "0", "1", "1"],
        "momenta": [["5/4", "0", "0", "5/4"], ["5/4", "0", "0", "-5/4"], ["5/4", "9/20", "0", "3/5"], ["5/4", "-9/20", "0", "-3/5"]],
        "expected": "positive shells; conserved; s+t+u=2",
    },
    "V-ENTROPY-RATIONAL": {
        "occupations": ["1/5", "2/5", "3/5", "4/5"],
        "expected": {"F": "4/625", "R": "144/625", "J": "-28/125", "A": "log(1/36)", "A_times_J": ">0"},
    },
    "V-NULLSPACE-SPECIES": {
        "basis": [["1", "-1", "0", "0", "0", "0"], ["0", "0", "1", "-1", "0", "0"], ["0", "0", "0", "0", "1", "-1"]],
        "expected": "full-graph left nullspace over Q equals the displayed span",
    },
}
