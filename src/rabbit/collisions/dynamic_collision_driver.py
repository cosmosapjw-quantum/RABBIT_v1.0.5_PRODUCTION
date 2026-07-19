"""rabbit.collisions.dynamic_collision_driver — FLRW-endpoint decoupling driver (PR-C7).

Wraps the clean dynamic-collision core (``dynamic_collision_core``) in a stiff ODE
integration that reaches the BBN neutrino-decoupling endpoint and reads off a physical
N_eff. This is the acceptance case for the q-Laguerre-collapse fix (the run that died at
N=3→4): the well-conditioned energy-variable core (PR-C2..C6) must now decouple cleanly
across the e± annihilation without the heavy bank collapsing.

The comoving frame (THE crux)
-----------------------------
Integration proceeds in the **comoving-momentum** frame y = p·a, with independent variable
N = ln a. The neutrino state is the comoving distribution f_ν(Y) on the FIXED comoving grid
Y_i = q_nodes (the collision Laguerre nodes reinterpreted as comoving momenta). For massless
ν, y = p·a is conserved under free-streaming, so

    df/dN|_y = C[f]/H         (NO redshift-advection term)

and after decoupling C→0 so f simply freezes in y — exactly what makes T_ν/T_γ → (4/11)^{1/3}
emerge from the e± entropy boost to T_γ. Evolving instead in thermal q = p/T_γ (as the raw
``flrw_dynamic_collision_rhs`` does) would freeze f(p/T_γ) and give the wrong N_eff (the 4.17
anomaly). Normalization: a_init is chosen so z_init = a_init·T_γ,init = 1, hence at N_init the
thermal momentum q = y and the initial comoving distribution is the Fermi–Dirac spectrum in y.

The plasma coupling (energy conservation — BY CONSTRUCTION)
----------------------------------------------------------
The driver evolves the per-node comoving distribution f_ν(Y) (the project goal: dynamic
spectral distortion from a first-principles collision term), so the ONLY honest energy the
plasma may lose per e-fold is exactly what the neutrinos actually gain — the moment of the
SAME comoving rate df/dN that the state integrates. We therefore compute the physical ν
energy-gain directly from that df:

    G_bank = ∫₀^∞ Y³ (df_bank/dN) dY          (per single Weyl species, dimensionless)
    frame  = (T_γ⁴ / z⁴) / (2π²) = a⁻⁴/(2π²)  (ρ_pair = (1/2π²)∫p³(f+f̄)dp, a⁻⁴·z⁴ = T_γ⁴)
    dQ_nue_pair_N = frame · 2 · G_nue         (ν_e + ν̄_e = 2 species)
    dQ_nux_bank_N = frame · 4 · G_nux         (ν_x bank = 4 Weyl = BANK_DEGENERACY[NUX])

These are fed to ``coupled_3T_rhs_from_collision_moments`` as the plasma-loss sources, so
plasma-loss ≡ −(ν-gain) EXACTLY, and TOTAL energy is conserved regardless of interpolation
quality. In the continuum this is IDENTICAL to the old PHYS·(module dQ) coupling
(∫Y³ C_at_Y dY = z⁴∫q³ C dq exactly), but it uses the driver's ACTUAL df instead of a second,
independently-quadratured bank moment. The two quadratures disagree ~8–12% on the coarse
q-Laguerre grid at in-run z (see ``test_interp_roundtrip_bounds_error``); that discretization
error now affects only the collision RATE (which converges with n_q), never conservation. The
prefactor 1/(2π²) (NOT 1/π², a 2× g double-count) is pinned by
``test_energy_conserving_plasma_coupling``, which now exercises the REAL rhs path.

Anti-drift note
---------------
No external anchor exists for Bianchi-I BBN with a neutrino collision term (no such paper).
The gates here are INTERNAL consistency/conditioning contracts — the collisionless limit
recovers the analytic N_eff = 3 and the collision-on limit stays bounded and physical — never
external validation. Do not claim external validation, and do not tune the plasma prefactor to
hit a target N_eff (the conservation test, not a target number, fixes PHYS).
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy.integrate import solve_ivp

from rabbit.collisions.deterministic_reference import build_fixed_collision_quadrature
from rabbit.collisions.dynamic_collision_core import (
    _collision_field,
    spectral_energy_moment,
)
from rabbit.thermo.nudec_coupled import (
    coupled_3T_rhs_from_collision_moments,
    hubble_3T,
)

#: Equilibrium spectral energy moment ∫q³/(e^q+1)dq = 7π⁴/120 (T_ν diagnostic reference).
_I_FD = 7.0 * np.pi**4 / 120.0

#: Standard T_ν/T_γ = (4/11)^{1/3} and z = a·T_γ = (11/4)^{1/3} at full decoupling.
_TNU_TGAMMA_STD = (4.0 / 11.0) ** (1.0 / 3.0)

#: N_eff = (11/4)^{4/3}·z⁻⁴·Σ(I_pair/I_FD) prefactor.
_NEFF_PREFAC = (11.0 / 4.0) ** (4.0 / 3.0)


def plasma_prefactor(T_gamma_MeV: float) -> float:
    """Physical energy-density prefactor PHYS = T_γ⁴/(2π²).

    ρ_pair = (1/2π²)∫p³(f+f̄)dp, so a dimensionless spectral energy moment (in units of T_γ)
    maps to a physical MeV⁴ energy density by T_γ⁴/(2π²). The driver's conserving plasma
    coupling uses the frame form (T_γ⁴/z⁴)/(2π²) = a⁻⁴/(2π²) on the comoving df-moment G (see
    the module docstring); this scalar prefactor is the z=1 case pinned by
    ``test_plasma_prefactor_is_half_over_pi_squared``. The 1/(2π²) — NOT 1/π² (a 2× g
    double-count) — is what ``test_energy_conserving_plasma_coupling`` enforces on the real
    rhs path."""
    T = float(T_gamma_MeV)
    return T**4 / (2.0 * np.pi**2)


@dataclass(frozen=True)
class DecouplingResult:
    """Frozen result of an FLRW decoupling integration.

    N_eff — effective number of neutrino species at the endpoint.
    T_nu_e_over_gamma / T_nu_x_over_gamma — per-bank T_ν/T_γ at the endpoint.
    z_final — comoving photon temperature a·T_γ at the endpoint ((11/4)^{1/3} if decoupled).
    T_gamma_final — plasma temperature [MeV] at the endpoint.
    N_final — ln a at the endpoint.
    reached_endpoint — whether the terminal T_γ event fired AND the solver succeeded.
    solver_success — raw ``solve_ivp`` success flag (fail-closed guard; see FINDING 2).
    solver_message — raw ``solve_ivp`` status message.
    max_clip_excursion — largest raw Pauli/positivity excursion max(max(f−1,0), max(−f,0))
        clamped away over the whole integration (FINDING 3: must stay tiny; a large value
        means the clip silently repaired a collapsing/Pauli-violating state).
    min_hubble_MeV — smallest raw Hubble rate seen before the 1e-100 floor (FINDING 3: must
        stay > 0; the floor masking a true H→0 would be an unphysical singularity).
    f_nue_final / f_nux_final — final comoving distributions on the fixed Y grid (diagnostics).
    """

    N_eff: float
    T_nu_e_over_gamma: float
    T_nu_x_over_gamma: float
    z_final: float
    T_gamma_final: float
    N_final: float
    reached_endpoint: bool
    solver_success: bool
    solver_message: str
    max_clip_excursion: float
    min_hubble_MeV: float
    f_nue_final: np.ndarray
    f_nux_final: np.ndarray


def _resample_comoving_to_thermal(Y, f_comoving, q_nodes, z):
    """Sample the comoving distribution f(Y) at the thermal q_nodes (comoving y = q_nodes·z).

    Tail → 0, left held at f[0], clipped to [0,1] before any collision evaluator sees it."""
    f_th = np.interp(q_nodes * z, Y, f_comoving, left=f_comoving[0], right=0.0)
    return np.clip(f_th, 0.0, 1.0)


def _map_thermal_field_to_comoving(q_nodes, C_th, Y, z):
    """Map a field on the thermal q_nodes back to the comoving grid Y (value at Y_i = C at q=Y_i/z)."""
    return np.interp(Y / z, q_nodes, C_th, left=C_th[0], right=0.0)


def _diagnostic_T_nu(f_th, q_nodes, q_weights, T_gamma):
    """Diagnostic ν temperature T_ν = T_γ·(∫q³ f_th / I_FD)^{1/4} (well-conditioned ρ^{1/4})."""
    ratio = max(spectral_energy_moment(f_th, q_nodes, q_weights) / _I_FD, 0.0)
    return T_gamma * ratio**0.25


def _make_rhs(quad, collisions):
    """Build the comoving-frame RHS d/dN of the flattened state [f_nue, f_nux, T_gamma].

    Returns ``(rhs, Y, q_nodes, q_weights, monitor)``. ``monitor`` is a mutable dict the rhs
    accumulates the smallest raw Hubble rate seen before the 1e-100 floor into (FINDING 3):
    read it after ``solve_ivp`` returns. Clip-excursion tracking is done on the ACCEPTED
    solution samples (``sol.y``) after the solve, NOT here — Radau probes wildly out-of-range
    trial states during Jacobian finite-differencing / rejected steps, and those probes never
    enter the accepted trajectory, so tracking them in the rhs conflates solver noise with a
    genuine state corruption."""
    q_nodes = np.asarray(quad.q_nodes, dtype=float)
    q_weights = np.asarray(quad.q_weights, dtype=float)
    Y = q_nodes.copy()  # comoving grid == numeric Laguerre nodes
    n = Y.size

    # Mutable closure cell for FINDING 3 Hubble tracking (min over accepted + probe evals; a
    # true H→0 would show up in any eval near the singularity, so probe evals only make this
    # more conservative). The frozen dataclass gets the value at construction.
    monitor = {"min_hubble_MeV": np.inf}

    def rhs(N, state):
        # Clip f∈[0,1] before every evaluator.
        f_nue = np.clip(state[:n], 0.0, 1.0)
        f_nux = np.clip(state[n : 2 * n], 0.0, 1.0)
        T_gamma = float(state[2 * n])
        a = np.exp(N)
        z = a * T_gamma

        # 1. comoving → thermal nodes for the evaluators.
        f_th_nue = _resample_comoving_to_thermal(Y, f_nue, q_nodes, z)
        f_th_nux = _resample_comoving_to_thermal(Y, f_nux, q_nodes, z)

        # 2. diagnostic ν temperatures (hubble + plasma only).
        T_nue = _diagnostic_T_nu(f_th_nue, q_nodes, q_weights, T_gamma)
        T_nux = _diagnostic_T_nu(f_th_nux, q_nodes, q_weights, T_gamma)

        # 3. Hubble; record the raw minimum before flooring so a true H→0 is observable.
        H = float(hubble_3T(T_gamma, T_nue, T_nux))
        if np.isfinite(H) and H < monitor["min_hubble_MeV"]:
            monitor["min_hubble_MeV"] = H
        H_safe = max(H, 1.0e-100)

        if collisions:
            # 4. collision fields on the thermal nodes (rate df/dt, MeV).
            C_th_nue = _collision_field(f_th_nue, f_th_nue, quad, T_gamma, "nue", 0.0)
            C_th_nux = _collision_field(f_th_nux, f_th_nux, quad, T_gamma, "nux", 0.0)

            # 5. map collision back to the comoving state grid; df/dN = C_at_Y / H.
            dC_nue_Y = _map_thermal_field_to_comoving(q_nodes, C_th_nue, Y, z)
            dC_nux_Y = _map_thermal_field_to_comoving(q_nodes, C_th_nux, Y, z)
            df_nue = dC_nue_Y / H_safe
            df_nux = dC_nux_Y / H_safe

            # 6. plasma channel: feed it the ν energy the state ACTUALLY gains — the moment of
            #    the same comoving df/dN, so plasma-loss ≡ −(ν-gain) EXACTLY (energy conserved
            #    BY CONSTRUCTION regardless of interp quality; see module docstring).
            G_nue = spectral_energy_moment(df_nue, Y, q_weights)  # ∫Y³ df_nue dY, 1 ν_e species
            G_nux = spectral_energy_moment(df_nux, Y, q_weights)  # ∫Y³ df_nux dY, 1 ν_x species
            frame = (T_gamma**4 / z**4) / (2.0 * np.pi**2)  # a⁻⁴/(2π²), a⁻⁴ = T_γ⁴/z⁴
            dQ_nue_pair_N = frame * 2.0 * G_nue  # ν_e + ν̄_e = 2 species
            dQ_nux_bank_N = frame * 4.0 * G_nux  # ν_x bank = 4 Weyl (BANK_DEGENERACY[NUX])
        else:
            # collisionless: f frozen in comoving y; pure plasma entropy evolution.
            df_nue = np.zeros_like(Y)
            df_nux = np.zeros_like(Y)
            dQ_nue_pair_N = 0.0
            dQ_nux_bank_N = 0.0

        dT_gamma = coupled_3T_rhs_from_collision_moments(
            T_gamma,
            T_nue,
            T_nux,
            dQ_nue_pair_N=dQ_nue_pair_N,
            dQ_nux_bank_N=dQ_nux_bank_N,
        )[0]

        out = np.empty_like(state)
        out[:n] = df_nue
        out[n : 2 * n] = df_nux
        out[2 * n] = dT_gamma
        return out

    return rhs, Y, q_nodes, q_weights, monitor


def integrate_flrw_decoupling(
    *,
    T_gamma_init_MeV: float = 10.0,
    n_q: int = 32,
    collisions: bool = True,
    T_gamma_stop_MeV: float = 1.0e-2,
    rtol: float = 1.0e-8,
    atol: float = 1.0e-10,
    N_span_max: float = 12.0,
    max_step: float = 0.5,
) -> DecouplingResult:
    """Integrate FLRW neutrino decoupling in the comoving frame to the endpoint.

    Starts deep in the coupled regime (T_γ,init = 10 MeV, f = Fermi–Dirac in comoving y) and
    integrates N = ln a with a stiff Radau solver until the plasma cools below
    ``T_gamma_stop_MeV`` (terminal event). Returns a frozen ``DecouplingResult`` with N_eff and
    the per-bank T_ν/T_γ. ``collisions=False`` gives the pure-entropy frame check (Gate A);
    ``collisions=True`` gives the blocker-fixed acceptance (Gate B).

    Fails closed (``RuntimeError``) if the terminal event does not fire or the solver reports
    failure (a finite N_eff would be unphysical), or if an endpoint bank spectral energy moment
    is negative (the heavy-bank-collapse symptom) — it never silently returns garbage N_eff."""
    quad = build_fixed_collision_quadrature(
        N_q=n_q, N_nue_y2=n_q, N_nue_y3=n_q, N_pair_y2=n_q, N_pair_leg=16
    )
    rhs, Y, q_nodes, q_weights, monitor = _make_rhs(quad, collisions)
    n = Y.size

    # a_init so z_init = a_init·T_γ,init = 1 ⟹ N_init = ln(1/T_γ,init); then q = y at N_init.
    a_init = 1.0 / float(T_gamma_init_MeV)
    N_init = np.log(a_init)

    # Initial comoving distribution: FD in y (z_init = 1 ⟹ q = y ⟹ f = 1/(exp(Y)+1)).
    f0 = 1.0 / (np.exp(Y) + 1.0)
    state0 = np.concatenate([f0.copy(), f0.copy(), [float(T_gamma_init_MeV)]])

    def event_Tstop(N, state):
        return state[2 * n] - float(T_gamma_stop_MeV)

    event_Tstop.terminal = True
    event_Tstop.direction = -1

    sol = solve_ivp(
        rhs,
        (N_init, N_init + float(N_span_max)),
        state0,
        method="Radau",
        rtol=rtol,
        atol=atol,
        events=event_Tstop,
        dense_output=False,
        max_step=max_step,
    )

    N_final = float(sol.t[-1])
    state_final = sol.y[:, -1]
    f_nue_final = np.clip(state_final[:n], 0.0, 1.0)
    f_nux_final = np.clip(state_final[n : 2 * n], 0.0, 1.0)
    T_gamma_final = float(state_final[2 * n])
    z_final = np.exp(N_final) * T_gamma_final

    solver_success = bool(sol.success)
    solver_message = str(sol.message)
    # FINDING 3 — largest raw Pauli/positivity excursion max(max(f−1,0), max(−f,0)) over the
    # ACCEPTED solution samples (the trajectory that actually flows through the integration),
    # not the solver's internal probe/rejected-step evals. A tiny value confirms the [0,1]
    # clamp never had to silently repair a collapsing or Pauli-violating accepted state.
    f_samples = sol.y[: 2 * n, :]
    max_clip_excursion = float(
        max(np.max(f_samples - 1.0), np.max(-f_samples), 0.0)
    )
    min_hubble_MeV = (
        float(monitor["min_hubble_MeV"]) if np.isfinite(monitor["min_hubble_MeV"]) else 0.0
    )

    # FINDING 2 — fail closed. A finite N_eff is only physical if the terminal T_γ event fired
    # AND the solver reported success; otherwise the run stalled/failed and any N_eff is garbage.
    reached_endpoint = solver_success and bool(sol.t_events[0].size > 0)
    if not reached_endpoint:
        raise RuntimeError(
            "FLRW decoupling integration did not reach the endpoint (N_eff would be "
            f"unphysical): solver_success={solver_success}, message={solver_message!r}, "
            f"terminal_event_fired={bool(sol.t_events[0].size > 0)}, N_reached={N_final:.4f}, "
            f"T_gamma_final={T_gamma_final:.4e} MeV (T_stop={float(T_gamma_stop_MeV):.4e}). "
            "Increase N_span_max or check the solver tolerances."
        )

    # FINDING 6 — the endpoint bank spectral energy moments must be non-negative; a negative
    # spectral energy is unphysical. Raise (fail loud) rather than silently flooring, and use
    # the SAME (unclamped) moments consistently for N_eff and both T_ν readouts.
    I_nue = spectral_energy_moment(f_nue_final, Y, q_weights)
    I_nux = spectral_energy_moment(f_nux_final, Y, q_weights)
    _neg_tol = 1.0e-12 * _I_FD
    if I_nue < -_neg_tol or I_nux < -_neg_tol:
        raise RuntimeError(
            "Endpoint bank spectral energy moment is negative (unphysical): "
            f"I_nue={I_nue:.4e}, I_nux={I_nux:.4e} (I_FD={_I_FD:.4e}). This is the "
            "heavy-bank-collapse symptom; refusing to report a floored N_eff."
        )

    # N_eff and both T_ν readouts use the SAME moments. Within [-_neg_tol, 0) is accepted
    # above as effectively zero; clamp that residue to 0 for the T_ν quartic root (base must
    # be ≥0) so N_eff and T_ν stay consistent (no T_ν=0-looks-like-collapse while N_eff≠0).
    ratio_nue = max(I_nue / _I_FD, 0.0)
    ratio_nux = max(I_nux / _I_FD, 0.0)
    # N_eff readout on the comoving grid Y (I_pair frozen at I_FD if collisionless).
    N_eff = _NEFF_PREFAC / z_final**4 * (ratio_nue + 2.0 * ratio_nux)
    T_nu_e_over_gamma = ratio_nue**0.25 / z_final
    T_nu_x_over_gamma = ratio_nux**0.25 / z_final

    return DecouplingResult(
        N_eff=float(N_eff),
        T_nu_e_over_gamma=float(T_nu_e_over_gamma),
        T_nu_x_over_gamma=float(T_nu_x_over_gamma),
        z_final=float(z_final),
        T_gamma_final=T_gamma_final,
        N_final=N_final,
        reached_endpoint=reached_endpoint,
        solver_success=solver_success,
        solver_message=solver_message,
        max_clip_excursion=max_clip_excursion,
        min_hubble_MeV=min_hubble_MeV,
        f_nue_final=f_nue_final,
        f_nux_final=f_nux_final,
    )
