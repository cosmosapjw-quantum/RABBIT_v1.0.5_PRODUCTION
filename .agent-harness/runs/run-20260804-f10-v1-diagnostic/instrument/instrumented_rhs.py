"""Instrumented copy of ``_trajectory_core.make_rhs``. SCRATCH ONLY.

Same discipline as instrumented_bdf: the body below is a VERBATIM copy of the pinned
repository function with ``#PROBE`` lines inserted. ``verify_verbatim`` strips them and
requires equality, so the physics path cannot silently differ from the audited one.

The sink is a module global rather than a parameter so the ``def`` line stays identical
to the pinned source and the comparison is exact.
"""

from __future__ import annotations

import inspect
import textwrap
from typing import Any, Callable, Sequence

import numpy as np

import _trajectory_core as core
from _trajectory_core import Deadline, Setup, Stats, unpack
from rabbit.decoupling import _independent_noqke as ind

SINK: Any = None


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
        SINK.raw_call(n, t_cm, t_gamma, action, thermo, stats)  #PROBE
        if log is not None and stats.evals % 50 == 1:
            log(
                f"eval {stats.evals}: N={n:.4f} T_cm={t_cm:.5f} "
                f"T_gamma={t_gamma:.5f} ratio={t_gamma / t_cm:.6f}"
            )
        return np.concatenate((dc_dn.ravel(), [dtg_dn, dt_dn]))

    return rhs


def verify_verbatim() -> dict:
    ours = textwrap.dedent(inspect.getsource(make_rhs))
    theirs = textwrap.dedent(inspect.getsource(core.make_rhs))
    stripped = "\n".join(line for line in ours.split("\n") if "#PROBE" not in line)
    return {
        "verbatim_ok": stripped.strip() == theirs.strip(),
        "probe_lines_removed": ours.count("#PROBE"),
        "pinned_lines": len(theirs.strip().split("\n")),
        "stripped_lines": len(stripped.strip().split("\n")),
    }
