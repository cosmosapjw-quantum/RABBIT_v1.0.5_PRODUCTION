"""Shared machinery for independent-trajectory audit drivers (D-068).

Why this module exists: D-065 finding F-D065-SW-02 measured the D-060/D-061 and
D-062/D-063 driver pairs as ~99.5% and ~98.5% identical, and recorded the same
report JSON twice (once as the artifact, once inside the captured stdout log).
Both are pure duplication with no evidentiary value. This module single-sources
the machinery so a new driver states only what it actually changes, and
``write_report`` writes the report exactly once.

Nothing here is retroactive: the frozen d060--d063 drivers keep their own bytes
and their own provenance. This module is authored fresh for D-069 onward.

The setup is parameterised on ``order`` and ``y_max`` because
``build_independent_grid(order, y_max)`` is a public knob of the frozen module
``src/rabbit/decoupling/_independent_noqke.py``. That makes a density-matched
companion run -- the measured beyond-domain enclosure and the spatial/domain
holdout D-065 requires -- reachable without editing the frozen module.
"""

from __future__ import annotations

import hashlib
import json
import os
import time
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any, Callable, Sequence

# Hard assignment, not setdefault: a driver that forgot to pin, or an inherited
# environment that set these to something else, must not silently produce a
# multi-threaded and therefore non-reproducible run. Drivers still pin these
# before their own `import numpy`, because OpenBLAS reads them at init.
for _pin in ("OPENBLAS_NUM_THREADS", "OMP_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ[_pin] = "1"

import numpy as np
from scipy.integrate import solve_ivp

from rabbit.decoupling import _independent_noqke as ind

TINY = float(np.finfo(float).tiny)


# --- environment guards ------------------------------------------------------


def module_sha256() -> str:
    with open(ind.__file__, "rb") as handle:
        return hashlib.sha256(handle.read()).hexdigest()


def constants_sha256(rows: Sequence[Sequence[float]]) -> str:
    """Self-hash for embedded cross-code anchor tables."""

    return hashlib.sha256(
        repr([tuple(float(v) for v in row) for row in rows]).encode()
    ).hexdigest()


class Deadline:
    """Frozen wall budget. Breach is mechanical, never a scientific verdict."""

    def __init__(self, budget_seconds: float) -> None:
        self.budget_seconds = float(budget_seconds)
        self.started = time.monotonic()

    def elapsed(self) -> float:
        return time.monotonic() - self.started

    def check(self) -> None:
        if self.elapsed() > self.budget_seconds:
            raise TimeoutError("frozen wall budget exceeded")


# --- setup -------------------------------------------------------------------


@dataclass(frozen=True)
class Setup:
    """One integration configuration: grid, collision config, and constants."""

    grid: Any
    config: Any
    order: int
    y_max: float
    t_start: float = 10.0
    t_gamma_end: float = 0.005
    rtol: float = 1e-6
    atol: float = 1e-9
    h_fd: float = 0.02
    transfer_floor: float = 5e-4
    label: str = "base"

    @property
    def spectral_size(self) -> int:
        return 3 * int(self.order)

    @property
    def state_size(self) -> int:
        return self.spectral_size + 2

    def describe(self) -> dict[str, Any]:
        return {
            "label": self.label,
            "order": int(self.order),
            "y_max": float(self.y_max),
            "node_density_per_unit_y": float(self.order) / float(self.y_max),
            "first_node": float(self.grid.nodes[0]),
            "last_node": float(self.grid.nodes[-1]),
            "incoming_polar_order": int(self.config.incoming_polar_order),
            "final_polar_order": int(self.config.final_polar_order),
            "final_azimuth_order": int(self.config.final_azimuth_order),
            "electron_radial_order": int(self.config.electron_radial_order),
            "rtol": float(self.rtol),
            "atol": float(self.atol),
            "h_fd": float(self.h_fd),
            "t_start": float(self.t_start),
            "t_gamma_end": float(self.t_gamma_end),
        }


def build_setup(
    order: int = 48,
    y_max: float = 24.0,
    *,
    incoming_polar_order: int = 4,
    final_polar_order: int = 4,
    electron_radial_order: int = 24,
    label: str = "base",
    **constants: float,
) -> Setup:
    grid = ind.build_independent_grid(int(order), float(y_max))
    config = ind.IndependentCollisionConfig(
        incoming_polar_order=incoming_polar_order,
        final_polar_order=final_polar_order,
        electron_radial_order=electron_radial_order,
    )
    return Setup(
        grid=grid, config=config, order=int(order), y_max=float(y_max),
        label=label, **constants,
    )


class Stats:
    def __init__(self) -> None:
        self.evals = 0
        self.occ_ok = True
        self.rejections: list[int] = []
        self.max_roundoff = 0.0


# --- state / right-hand side -------------------------------------------------


def unpack(setup: Setup, y: np.ndarray) -> tuple[np.ndarray, float, float]:
    """(cloglog[3, order], T_gamma [MeV], t [MeV^-1]) from the packed state."""

    spectral = setup.spectral_size
    c = y[:spectral].reshape(3, setup.order)
    return c, float(y[spectral]), float(y[spectral + 1])


def initial_state(setup: Setup) -> tuple[np.ndarray, np.ndarray]:
    c0 = ind.pair_logits_to_cloglog(np.stack([-setup.grid.nodes for _ in range(3)]))
    y0 = np.concatenate((c0.ravel(), [setup.t_start, 0.0]))
    return c0, y0


def make_rhs(
    setup: Setup,
    stats: Stats,
    deadline: Deadline,
    *,
    s_pair: float = 1.0,
    s_qem: float = 1.0,
    log: Callable[[str], None] | None = None,
) -> Callable[[float, np.ndarray], np.ndarray]:
    """dY/dN. ``s_pair``/``s_qem`` are +1 in production; -1 builds a sign mutant."""

    def rhs(n: float, y: np.ndarray) -> np.ndarray:
        deadline.check()
        c, t_gamma, _t = unpack(setup, y)
        t_cm = setup.t_start * float(np.exp(-n))
        occupations = ind.cloglog_to_occupation(c)
        if not (np.all(occupations > 0.0) and np.all(occupations < 1.0)):
            stats.occ_ok = False
            raise ind.IndependentNoQkeError("occupation left (0, 1)")
        action = ind.evaluate_independent_collision_action(
            grid=setup.grid, pair_cloglog=c,
            temperature_cm_mev=t_cm, temperature_gamma_mev=t_gamma,
            config=setup.config,
        )
        thermo = ind.independent_thermodynamics(
            grid=setup.grid, pair_cloglog=c,
            temperature_cm_mev=t_cm, temperature_gamma_mev=t_gamma,
        )
        stats.evals += 1
        stats.rejections.append(int(action.whole_reaction_domain_rejections))
        stats.max_roundoff = max(
            stats.max_roundoff, float(action.largest_matrix_roundoff_correction)
        )
        hubble = thermo.hubble_mev
        total = np.asarray(action.total)
        pair_rate = 0.5 * np.stack(
            (total[0] + total[1], total[2] + total[3], total[4] + total[5])
        )
        chain = ind.cloglog_chain_factor(c)
        dc_dn = s_pair * pair_rate / (hubble * chain)
        eos = ind.electromagnetic_eos_adaptive(t_gamma)
        q_em = action.electron_bath_energy_transfer
        dtg_dn = (
            -3.0 * (eos.rho + eos.pressure) + s_qem * q_em / hubble
        ) / eos.drho_dtemperature
        dt_dn = 1.0 / hubble
        if log is not None and stats.evals % 50 == 1:
            log(
                f"eval {stats.evals}: N={n:.4f} T_cm={t_cm:.5f} "
                f"T_gamma={t_gamma:.5f} ratio={t_gamma / t_cm:.6f}"
            )
        return np.concatenate((dc_dn.ravel(), [dtg_dn, dt_dn]))

    return rhs


def make_terminal_event(setup: Setup) -> Callable[[float, np.ndarray], float]:
    spectral = setup.spectral_size

    def terminal(n: float, y: np.ndarray) -> float:
        return float(y[spectral]) - setup.t_gamma_end

    terminal.terminal = True  # type: ignore[attr-defined]
    terminal.direction = -1  # type: ignore[attr-defined]
    return terminal


def run_integration(
    setup: Setup,
    rtol: float,
    stats: Stats,
    deadline: Deadline,
    log: Callable[[str], None],
    *,
    label: str | None = None,
) -> tuple[Any, bool, np.ndarray]:
    name = label or setup.label
    log(f"{name}: BDF integration (order={setup.order} y_max={setup.y_max} rtol={rtol})")
    c0, y0 = initial_state(setup)
    n_max = float(np.log(setup.t_start / setup.t_gamma_end) + 2.0)
    solution = solve_ivp(
        make_rhs(setup, stats, deadline, log=log),
        (0.0, n_max), y0, method="BDF", rtol=rtol, atol=setup.atol,
        events=make_terminal_event(setup), dense_output=True,
    )
    log(f"{name}: status={solution.status} evals={stats.evals}")
    reached = solution.status == 1 and len(solution.t_events[0]) == 1
    return solution, reached, c0


# --- diagnostics -------------------------------------------------------------


def comoving_energies(
    setup: Setup, n: float, y: np.ndarray
) -> tuple[float, float, float, float, float]:
    """(W_nu, W_tot, rho_EM, P_EM, rho_nu) at the dense-output state y(n)."""

    c, t_gamma, _t = unpack(setup, y)
    t_cm = setup.t_start * float(np.exp(-n))
    thermo = ind.independent_thermodynamics(
        grid=setup.grid, pair_cloglog=c,
        temperature_cm_mev=t_cm, temperature_gamma_mev=t_gamma,
    )
    eos = ind.electromagnetic_eos_adaptive(t_gamma)
    scale = float(np.exp(4.0 * n))
    rho_nu = thermo.energy_density_neutrino
    return scale * rho_nu, scale * (rho_nu + eos.rho), eos.rho, eos.pressure, rho_nu


def coupled_residuals(
    setup: Setup, sol: Callable[[float], np.ndarray], points: Sequence[float]
) -> list[dict[str, Any]]:
    """R_total / R_transfer from central differences of the dense output.

    Exact physics: dW_tot/dN = exp(4N)(rho_EM - 3 P_EM); the +-Q/H exchange
    cancels between the sectors, leaving only the electron-mass trace.
    """

    h = setup.h_fd
    rows: list[dict[str, Any]] = []
    for n_k in points:
        w_nu_m, w_tot_m, *_ = comoving_energies(setup, n_k - h, sol(n_k - h))
        w_nu_p, w_tot_p, *_ = comoving_energies(setup, n_k + h, sol(n_k + h))
        w_nu_0, w_tot_0, rho_em, p_em, rho_nu = comoving_energies(setup, n_k, sol(n_k))
        d_w_nu = (w_nu_p - w_nu_m) / (2.0 * h)
        d_w_tot = (w_tot_p - w_tot_m) / (2.0 * h)
        scale = float(np.exp(4.0 * n_k))
        trace_source = scale * (rho_em - 3.0 * p_em)
        numerator = abs(d_w_tot - trace_source)
        enthalpy = scale * (rho_nu + rho_em + p_em + rho_nu / 3.0)
        r_total = numerator / max(enthalpy, TINY)
        active = abs(d_w_nu) >= setup.transfer_floor * w_nu_0
        r_transfer = numerator / max(abs(d_w_nu), TINY) if active else None
        rows.append({
            "N": float(n_k), "W_nu": w_nu_0, "W_tot": w_tot_0,
            "dW_nu_dN": d_w_nu, "dW_tot_dN": d_w_tot,
            "trace_source": trace_source, "R_total": float(r_total),
            "active": bool(active),
            "R_transfer": (float(r_transfer) if active else None),
        })
    return rows


def checkpoint_states(
    setup: Setup, sol: Callable[[float], np.ndarray], points: Sequence[float]
) -> list[dict[str, Any]]:
    """Full retained state per checkpoint (D-065 finding F-D065-TRJ-02).

    D-063 kept nine reduced scalars, which is not enough for a third party to
    recompute anything. These rows carry the entire integrator state -- every
    cloglog component, T_gamma, and cosmic time -- so every derived quantity in
    the report is independently recomputable from the report alone.
    """

    rows: list[dict[str, Any]] = []
    for n_k in points:
        y = sol(n_k)
        c, t_gamma, t_mev = unpack(setup, y)
        occupations = ind.cloglog_to_occupation(c)
        rows.append({
            "N": float(n_k),
            "T_gamma_mev": float(t_gamma),
            "T_cm_mev": float(setup.t_start * np.exp(-n_k)),
            "t_seconds": float(t_mev / ind.MEV_TO_INVERSE_SECONDS),
            "cloglog": [[float(v) for v in row] for row in c],
            "occupation": [[float(v) for v in row] for row in occupations],
        })
    return rows


def endpoint_summary(
    setup: Setup, n_end: float, y_end: np.ndarray, stats: Stats
) -> dict[str, Any]:
    c_end, t_gamma_end, t_end_mev = unpack(setup, y_end)
    n_end = float(n_end)
    t_cm_end = setup.t_start * float(np.exp(-n_end))
    thermo_end = ind.independent_thermodynamics(
        grid=setup.grid, pair_cloglog=c_end,
        temperature_cm_mev=t_cm_end, temperature_gamma_mev=t_gamma_end,
    )
    rho_gamma = np.pi**2 * t_gamma_end**4 / 15.0
    n_eff = (8.0 / 7.0) * (11.0 / 4.0) ** (4.0 / 3.0) * (
        thermo_end.energy_density_neutrino / rho_gamma
    )
    return {
        "N_end": n_end,
        "t_end_s": float(t_end_mev / ind.MEV_TO_INVERSE_SECONDS),
        "N_eff": float(n_eff),
        "T_gamma_end": t_gamma_end,
        "T_cm_end": t_cm_end,
        "gamma_heating_ratio": float(t_gamma_end / t_cm_end),
        "evals": stats.evals,
        "c_end": c_end,
        "occ_end": ind.cloglog_to_occupation(c_end),
        "thermo": thermo_end,
    }


def pair_densities(
    setup: Setup, occ: np.ndarray, t_cm: float
) -> tuple[np.ndarray, np.ndarray]:
    """Independent pair number/energy densities per flavour (2 helicity states)."""

    w = setup.grid.weights
    y = setup.grid.nodes
    number = 2.0 * t_cm**3 / ind.TWO_PI_SQUARED * np.sum(w * y**2 * occ, axis=1)
    energy = 2.0 * t_cm**4 / ind.TWO_PI_SQUARED * np.sum(w * y**3 * occ, axis=1)
    return number, energy


def spectral_moments(
    setup: Setup, occ: np.ndarray, powers: Sequence[int]
) -> dict[str, dict[str, float]]:
    """Dimensionless moments ``M_k = sum_i w_i y_i^k f_i`` on the setup's own grid.

    Computed on each code's own nodes and weights, so a cross-code comparison of
    these numbers involves no interpolation in either direction and no masking:
    every node contributes at its true quadrature weight.
    """

    w = setup.grid.weights
    y = setup.grid.nodes
    out: dict[str, dict[str, float]] = {}
    for row, name in ((0, "e"), (1, "mu"), (2, "tau")):
        out[name] = {
            f"M{k}": float(np.sum(w * np.power(y, k) * occ[row])) for k in powers
        }
    return out


def anchor_moments(
    nodes: Sequence[float],
    weights: Sequence[float],
    occupations: Sequence[float],
    powers: Sequence[int],
) -> dict[str, float]:
    """Same moments for an external code, on that code's own rule."""

    y = np.asarray(nodes, dtype=np.float64)
    w = np.asarray(weights, dtype=np.float64)
    f = np.asarray(occupations, dtype=np.float64)
    return {f"M{k}": float(np.sum(w * np.power(y, k) * f)) for k in powers}


def equilibrium_tail_fraction(y_min: float, power: int = 3) -> float:
    """Exact ``int_{y_min}^inf y^p/(1+e^y) / int_0^inf y^p/(1+e^y)``.

    The analytic half of the beyond-domain enclosure: whatever the truncated
    domain omits is bounded by the equilibrium content beyond its edge, up to
    the measured distortion, which the companion run supplies.
    """

    from scipy.integrate import quad

    def integrand(y: float) -> float:
        # exp(-y)/(1+exp(-y)) rather than 1/(1+exp(y)): same value, no overflow
        # in the far tail where quad probes large y.
        decay = np.exp(-y)
        return y**power * decay / (1.0 + decay)

    tail, _ = quad(integrand, y_min, np.inf, limit=200)
    total, _ = quad(integrand, 0.0, np.inf, limit=200)
    return float(tail / total)


# --- reporting ---------------------------------------------------------------


@dataclass
class Reporter:
    """Canonical single-destination report writer (D-068).

    The report body is written to ``path`` and nowhere else. Progress goes to
    stdout as short lines, so the captured log is a timeline rather than a
    second copy of the artifact.
    """

    path: str | None
    started_monotonic: float = field(default_factory=time.monotonic)
    started_at: str = field(
        default_factory=lambda: datetime.now(timezone.utc).isoformat(timespec="seconds")
    )

    def log(self, message: str) -> None:
        print(f"[{time.monotonic() - self.started_monotonic:9.1f}s] {message}", flush=True)

    def write(self, report: dict[str, Any]) -> None:
        if self.path is None:
            return
        text = json.dumps(report, sort_keys=True, indent=1) + "\n"
        tmp = f"{self.path}.tmp.{os.getpid()}"
        with open(tmp, "w") as handle:
            handle.write(text)
        os.replace(tmp, self.path)

    def stamp(self, report: dict[str, Any]) -> dict[str, Any]:
        """Machine timestamps only -- never asserted round-minute values."""

        report["started_at"] = self.started_at
        report["completed_at"] = datetime.now(timezone.utc).isoformat(timespec="seconds")
        report["wall_seconds"] = round(time.monotonic() - self.started_monotonic, 1)
        return report
