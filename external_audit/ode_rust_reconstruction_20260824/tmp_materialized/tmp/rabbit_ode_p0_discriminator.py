from __future__ import annotations

import hashlib
import json
import math
import platform
import time
from pathlib import Path

import numpy as np
import scipy
from scipy.integrate import solve_ivp

import rabbit.collisions.deterministic_reference as det
from rabbit.collisions.dynamic_collision_core import _collision_field
from rabbit.collisions.dynamic_collision_driver import (
    _I_FD,
    _NEFF_PREFAC,
    _diagnostic_T_nu,
    _make_rhs,
    _map_thermal_field_to_comoving,
    _resample_comoving_to_thermal,
)
from rabbit.collisions.deterministic_reference import build_fixed_collision_quadrature
from rabbit.thermo.nudec_coupled import hubble_3T
import rabbit.weak.live_rates as weak


N_Q = 24
RTOL = 1.0e-8
ATOL = 1.0e-10
MAX_STEP = 0.5
PREFIX_DN = 1.0e-2
RECON_REL_TOL = 1.0e-10
CANCELLATION_SEVERE = 1.0 / math.sqrt(np.finfo(float).eps)
IMPLICATED = {
    "nue": tuple(range(18, 24)),
    "nux": tuple(range(19, 24)),
}
RETAINED_SHA256 = "2da11255f25761e4b7e7ea330eda2c4948013e4d1df00c5e127720a90f9a678e"
RETAINED_FIRST_N_HEX = "-0x1.26ba1e8d370afp+1"
RETAINED_FIRST_VALUE_HEX = "-0x1.6f1a30867d34fp-99"


def f64(value: float | np.floating) -> float:
    return float(value)


def float_record(value: float | np.floating) -> dict[str, float | str]:
    x = float(value)
    return {"decimal": x, "hex": x.hex()}


def json_safe(value):
    if isinstance(value, dict):
        return {str(key): json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [json_safe(item) for item in value]
    if isinstance(value, (float, np.floating)):
        number = float(value)
        if math.isnan(number):
            return "NaN"
        if math.isinf(number):
            return "+Infinity" if number > 0.0 else "-Infinity"
        return number
    return value


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def raw_resample(Y: np.ndarray, values: np.ndarray, q: np.ndarray, z: float) -> np.ndarray:
    return np.interp(q * z, Y, values, left=values[0], right=0.0)


def independent_collision_field(
    f_target: np.ndarray,
    f_partner: np.ndarray,
    quad,
    T_MeV: float,
    species: str,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Factorized gain-minus-loss with long-double and math.fsum accumulation.

    Inputs and quadrature nodes originate as binary64.  The independent axis changes
    the Pauli algebra and accumulation, not the physical quadrature contract.
    """

    arrays = quad.arrays()
    q64 = np.asarray(arrays["q_nodes"], dtype=float)
    f64_target = np.asarray(f_target, dtype=float)
    f64_partner = np.asarray(f_partner, dtype=float)
    q = q64.astype(np.longdouble)
    f = f64_target.astype(np.longdouble)
    fbar = f64_partner.astype(np.longdouble)
    _, gl64, gr64 = det._couplings_for_species(species)
    gl = np.longdouble(gl64)
    gr = np.longdouble(gr64)
    gl2_gr2 = gl * gl + gr * gr
    gl2_minus_gr2 = gl * gl - gr * gr
    prefactor = np.longdouble(det.hm_reduced_collision_prefactor(float(T_MeV)))

    y2 = np.asarray(arrays["nue_y2_nodes"], dtype=np.longdouble)
    y2w = np.asarray(det._laguerre_plain_weights(
        arrays["nue_y2_nodes"], arrays["nue_y2_weights"]
    ), dtype=np.longdouble)
    y3 = np.asarray(arrays["nue_y3_nodes"], dtype=np.longdouble)
    y3w = np.asarray(det._laguerre_plain_weights(
        arrays["nue_y3_nodes"], arrays["nue_y3_weights"]
    ), dtype=np.longdouble)
    y1g = q[:, None, None]
    y3g = y3[None, :, None]
    y2g = y2[None, None, :]
    y4g = y1g + y2g - y3g
    valid = y4g > 0
    f1 = f[:, None, None]
    f2 = 1 / (np.exp(y2g) + 1)
    f3 = f[None, :, None]
    f4 = 1 / (np.exp(y4g) + 1)
    gain = f3 * f4 * (1 - f1) * (1 - f2)
    loss = f1 * f2 * (1 - f3) * (1 - f4)
    stat = np.where(valid, gain - loss, 0)
    matrix = (
        gl2_gr2 * ((y1g * y2g) ** 2 + (y3g * y4g) ** 2)
        + gl2_minus_gr2 * ((y1g * y4g) ** 2 + (y2g * y3g) ** 2)
    )
    terms_scatt = y3w[None, :, None] * y2w[None, None, :] * matrix * stat

    pair_y2 = np.asarray(arrays["pair_y2_nodes"], dtype=np.longdouble)
    pair_y2w = np.asarray(det._laguerre_plain_weights(
        arrays["pair_y2_nodes"], arrays["pair_y2_weights"]
    ), dtype=np.longdouble)
    leg = np.asarray(arrays["pair_leg_nodes"], dtype=np.longdouble)
    legw = np.asarray(arrays["pair_leg_weights"], dtype=np.longdouble)
    y1p = q[:, None, None]
    y2p = pair_y2[None, :, None]
    ysum = y1p + y2p
    y3p = np.longdouble(0.5) * ysum * (leg[None, None, :] + 1)
    y4p = ysum - y3p
    validp = (ysum > 0) & (y4p > 0)
    f1p = f[:, None, None]
    f2p = fbar[None, :, None]
    f3p = 1 / (np.exp(y3p) + 1)
    f4p = 1 / (np.exp(y4p) + 1)
    gainp = f3p * f4p * (1 - f1p) * (1 - f2p)
    lossp = f1p * f2p * (1 - f3p) * (1 - f4p)
    statp = np.where(validp, gainp - lossp, 0)
    matrixp = (
        gl2_gr2 * ((y1p * y3p) ** 2 + (y2p * y4p) ** 2)
        + gl2_minus_gr2 * ((y1p * y4p) ** 2 + (y2p * y3p) ** 2)
    )
    y3pw = np.longdouble(0.5) * ysum * legw[None, None, :]
    terms_pair = pair_y2w[None, :, None] * y3pw * matrixp * statp

    total_ld = np.sum(terms_scatt, axis=(1, 2), dtype=np.longdouble)
    total_ld += np.sum(terms_pair, axis=(1, 2), dtype=np.longdouble)
    total_fsum = np.empty(q.size, dtype=float)
    absolute_sum = np.empty(q.size, dtype=np.longdouble)
    for index in range(q.size):
        scatt_values = np.asarray(terms_scatt[index], dtype=float).ravel()
        pair_values = np.asarray(terms_pair[index], dtype=float).ravel()
        total_fsum[index] = math.fsum(scatt_values.tolist() + pair_values.tolist())
        absolute_sum[index] = (
            np.sum(np.abs(terms_scatt[index]), dtype=np.longdouble)
            + np.sum(np.abs(terms_pair[index]), dtype=np.longdouble)
        )
    denominator = np.maximum(q * q, np.longdouble(1.0e-30))
    c_ld = prefactor * total_ld / denominator
    c_fsum = float(prefactor) * total_fsum / np.asarray(denominator, dtype=float)
    c_abs = prefactor * absolute_sum / denominator
    return np.asarray(c_ld, dtype=np.longdouble), c_fsum, np.asarray(c_abs, dtype=np.longdouble)


def physical_context(state: np.ndarray, N: float, Y: np.ndarray, q: np.ndarray, qw: np.ndarray):
    n = Y.size
    T = float(state[2 * n])
    z = math.exp(float(N)) * T
    nue = np.asarray(state[:n], dtype=float)
    nux = np.asarray(state[n:2 * n], dtype=float)
    fth_nue = _resample_comoving_to_thermal(Y, nue, q, z)
    fth_nux = _resample_comoving_to_thermal(Y, nux, q, z)
    T_nue = _diagnostic_T_nu(fth_nue, q, qw, T)
    T_nux = _diagnostic_T_nu(fth_nux, q, qw, T)
    H = float(hubble_3T(T, T_nue, T_nux))
    return T, z, H, fth_nue, fth_nux, T_nue, T_nux


def state_collision_value(
    state: np.ndarray,
    N: float,
    bank: str,
    node: int,
    replacement: float,
    Y: np.ndarray,
    q: np.ndarray,
    qw: np.ndarray,
    quad,
) -> dict[str, float]:
    n = Y.size
    trial = np.array(state, dtype=float, copy=True)
    offset = 0 if bank == "nue" else n
    trial[offset + node] = float(replacement)
    T, z, H, fth_nue, fth_nux, _, _ = physical_context(trial, N, Y, q, qw)
    fth = fth_nue if bank == "nue" else fth_nux
    ordinary_th = _collision_field(fth, fth, quad, T, bank, 0.0)
    independent_ld, independent_fsum, absolute_th = independent_collision_field(
        fth, fth, quad, T, bank
    )
    ordinary = _map_thermal_field_to_comoving(q, ordinary_th, Y, z)[node]
    independent = _map_thermal_field_to_comoving(
        q, np.asarray(independent_ld, dtype=float), Y, z
    )[node]
    fsum = _map_thermal_field_to_comoving(q, independent_fsum, Y, z)[node]
    absolute = _map_thermal_field_to_comoving(
        q, np.asarray(absolute_th, dtype=float), Y, z
    )[node]
    return {
        "ordinary": float(ordinary),
        "factorized_longdouble": float(independent),
        "factorized_fsum": float(fsum),
        "absolute_term_sum": float(absolute),
        "H": H,
        "z": z,
    }


def envelope(values: dict[str, float]) -> tuple[float, float]:
    candidates = [values["ordinary"], values["factorized_longdouble"], values["factorized_fsum"]]
    return min(candidates), max(candidates)


def moment(values: np.ndarray, q: np.ndarray, qw: np.ndarray, power: int) -> np.longdouble:
    plain = np.asarray(qw * np.exp(np.minimum(q, 500.0)), dtype=np.longdouble)
    return np.sum(
        plain * np.asarray(q, dtype=np.longdouble) ** power * np.asarray(values, dtype=np.longdouble),
        dtype=np.longdouble,
    )


def weak_rates_raw_linear(
    f_nue: np.ndarray,
    f_nuebar: np.ndarray,
    q_nodes: np.ndarray,
    T_gamma: float,
    T_nu: float,
) -> dict[str, object]:
    T_e = float(T_gamma) / weak._M_ELECTRON_MEV
    T_nu_dimless = float(T_nu) / weak._M_ELECTRON_MEV
    q_mass = weak._Q_DIMLESS
    kernel = weak._build_channel_kernel_cache(
        T_e, T_nu_dimless, q_mass, 0, weak.DEFAULT_WEAK_QUADRATURE
    )
    I0 = weak.get_I0_born()

    def distribution(values):
        values_ld = np.asarray(values, dtype=np.longdouble)
        nodes_ld = np.asarray(q_nodes, dtype=np.longdouble)

        def evaluate(energy):
            q_eval = np.asarray(energy, dtype=np.longdouble) / np.longdouble(T_nu_dimless)
            return np.interp(
                np.asarray(q_eval, dtype=float),
                np.asarray(nodes_ld, dtype=float),
                np.asarray(values_ld, dtype=float),
                left=float(values_ld[0]),
                right=0.0,
            ).astype(np.longdouble)

        return evaluate

    fn = distribution(f_nue)
    fnbar = distribution(f_nuebar)

    def sum_ld(values) -> np.longdouble:
        return np.sum(np.asarray(values, dtype=np.longdouble), dtype=np.longdouble)

    e, pref, fd = kernel["a"]
    Ia = sum_ld(np.asarray(pref, dtype=np.longdouble) * fn(e) * (1 - np.asarray(fd, dtype=np.longdouble))) if pref.size else np.longdouble(0)
    e, pref, fd = kernel["b"]
    Ib = sum_ld(np.asarray(pref, dtype=np.longdouble) * np.asarray(fd, dtype=np.longdouble) * (1 - fnbar(e)))
    ecf, prefc, fdcf = kernel["c"]
    if prefc.size:
        fbarcf = fnbar(ecf)
        Ic = sum_ld(np.asarray(prefc, dtype=np.longdouble) * (1 - np.asarray(fdcf, dtype=np.longdouble)) * (1 - fbarcf))
        If = sum_ld(np.asarray(kernel["f"][1], dtype=np.longdouble) * np.asarray(fdcf, dtype=np.longdouble) * fbarcf)
    else:
        Ic = If = np.longdouble(0)
    e, pref, fd = kernel["d"]
    Id = sum_ld(np.asarray(pref, dtype=np.longdouble) * np.asarray(fd, dtype=np.longdouble) * (1 - fn(e))) if pref.size else np.longdouble(0)
    e, pref, fd = kernel["e"]
    Ie = sum_ld(np.asarray(pref, dtype=np.longdouble) * fnbar(e) * (1 - np.asarray(fd, dtype=np.longdouble))) if pref.size else np.longdouble(0)
    tau = np.longdouble(878.4)
    lambda_np = (Ia + Ib + Ic) / (np.longdouble(I0) * tau)
    lambda_pn = (Id + Ie + If) / (np.longdouble(I0) * tau)
    return {
        "lambda_np": float(lambda_np),
        "lambda_pn": float(lambda_pn),
        "channels_np": [float(Ia), float(Ib), float(Ic)],
        "channels_pn": [float(Id), float(Ie), float(If)],
    }


def subtract_records(raw: dict[str, object], clipped: dict[str, object]) -> dict[str, object]:
    return {
        "lambda_np": float(raw["lambda_np"]) - float(clipped["lambda_np"]),
        "lambda_pn": float(raw["lambda_pn"]) - float(clipped["lambda_pn"]),
        "channels_np": [a - b for a, b in zip(raw["channels_np"], clipped["channels_np"], strict=True)],
        "channels_pn": [a - b for a, b in zip(raw["channels_pn"], clipped["channels_pn"], strict=True)],
    }


def weak_rates_direct_signed_delta(
    delta_f_nue: np.ndarray,
    delta_f_nuebar: np.ndarray,
    q_nodes: np.ndarray,
    T_gamma: float,
    T_nu: float,
) -> dict[str, object]:
    """Evaluate the signed weak-rate response without subtracting O(1) totals.

    The Born channel cache is affine in each neutrino distribution.  Interpolate
    the signed occupation difference itself so a 1e-30 tail is not rounded away
    when it is first added to an O(1) Fermi--Dirac value.
    """
    T_e = float(T_gamma) / weak._M_ELECTRON_MEV
    T_nu_dimless = float(T_nu) / weak._M_ELECTRON_MEV
    kernel = weak._build_channel_kernel_cache(
        T_e,
        T_nu_dimless,
        weak._Q_DIMLESS,
        0,
        weak.DEFAULT_WEAK_QUADRATURE,
    )

    def signed_distribution(values):
        values_ld = np.asarray(values, dtype=np.longdouble)

        def evaluate(energy):
            q_eval = np.asarray(energy, dtype=np.longdouble) / np.longdouble(T_nu_dimless)
            return np.interp(
                np.asarray(q_eval, dtype=float),
                np.asarray(q_nodes, dtype=float),
                np.asarray(values_ld, dtype=float),
                left=float(values_ld[0]),
                right=0.0,
            ).astype(np.longdouble)

        return evaluate

    delta_fn = signed_distribution(delta_f_nue)
    delta_fnbar = signed_distribution(delta_f_nuebar)

    def sum_ld(values) -> np.longdouble:
        return np.sum(np.asarray(values, dtype=np.longdouble), dtype=np.longdouble)

    e, pref, fd = kernel["a"]
    delta_Ia = (
        sum_ld(
            np.asarray(pref, dtype=np.longdouble)
            * delta_fn(e)
            * (1 - np.asarray(fd, dtype=np.longdouble))
        )
        if pref.size
        else np.longdouble(0)
    )
    e, pref, fd = kernel["b"]
    delta_Ib = -sum_ld(
        np.asarray(pref, dtype=np.longdouble)
        * np.asarray(fd, dtype=np.longdouble)
        * delta_fnbar(e)
    )
    ecf, prefc, fdcf = kernel["c"]
    if prefc.size:
        delta_Ic = -sum_ld(
            np.asarray(prefc, dtype=np.longdouble)
            * (1 - np.asarray(fdcf, dtype=np.longdouble))
            * delta_fnbar(ecf)
        )
        delta_If = sum_ld(
            np.asarray(kernel["f"][1], dtype=np.longdouble)
            * np.asarray(fdcf, dtype=np.longdouble)
            * delta_fnbar(ecf)
        )
    else:
        delta_Ic = delta_If = np.longdouble(0)
    e, pref, fd = kernel["d"]
    delta_Id = (
        -sum_ld(
            np.asarray(pref, dtype=np.longdouble)
            * np.asarray(fd, dtype=np.longdouble)
            * delta_fn(e)
        )
        if pref.size
        else np.longdouble(0)
    )
    e, pref, fd = kernel["e"]
    delta_Ie = (
        sum_ld(
            np.asarray(pref, dtype=np.longdouble)
            * delta_fnbar(e)
            * (1 - np.asarray(fd, dtype=np.longdouble))
        )
        if pref.size
        else np.longdouble(0)
    )
    normalization = np.longdouble(weak.get_I0_born()) * np.longdouble(878.4)
    return {
        "lambda_np": float((delta_Ia + delta_Ib + delta_Ic) / normalization),
        "lambda_pn": float((delta_Id + delta_Ie + delta_If) / normalization),
        "channels_np": [float(delta_Ia), float(delta_Ib), float(delta_Ic)],
        "channels_pn": [float(delta_Id), float(delta_Ie), float(delta_If)],
    }


def main() -> None:
    retained_path = Path("/tmp/rabbit_raw_trajectory_exact_head_collision_on.json")
    retained_digest = sha256(retained_path)
    if retained_digest != RETAINED_SHA256:
        raise RuntimeError(f"retained trajectory SHA mismatch: {retained_digest}")

    quad = build_fixed_collision_quadrature(
        N_q=N_Q,
        N_nue_y2=N_Q,
        N_nue_y3=N_Q,
        N_pair_y2=N_Q,
        N_pair_leg=16,
    )
    rhs, Y, q, qw, _ = _make_rhs(quad, collisions=True)
    n = Y.size
    N0 = math.log(1.0 / 10.0)
    initial = 1.0 / (np.exp(Y) + 1.0)
    y0 = np.concatenate([initial, initial, [10.0]])

    started = time.perf_counter()

    def prefix_event(N, _state):
        return float(N) - (N0 + PREFIX_DN)

    prefix_event.terminal = True
    prefix_event.direction = 1
    solution = solve_ivp(
        rhs,
        (N0, N0 + 12.0),
        y0,
        method="Radau",
        rtol=RTOL,
        atol=ATOL,
        max_step=MAX_STEP,
        dense_output=False,
        events=prefix_event,
    )
    prefix_wall = time.perf_counter() - started
    occupations = np.asarray(solution.y[: 2 * n], dtype=float)
    invalid = (~np.isfinite(occupations)) | (occupations <= 0.0) | (occupations >= 1.0)
    bad_columns = np.flatnonzero(np.any(invalid, axis=0))
    if bad_columns.size == 0 or bad_columns[0] == 0:
        raise RuntimeError("prefix did not contain a valid-to-invalid transition")
    first_index = int(bad_columns[0])
    previous_index = first_index - 1
    previous = np.asarray(solution.y[:, previous_index], dtype=float)
    first = np.asarray(solution.y[:, first_index], dtype=float)
    previous_N = float(solution.t[previous_index])
    first_N = float(solution.t[first_index])
    delta_N = first_N - previous_N
    first_bad_flat = np.flatnonzero(invalid[:, first_index])

    retained = json.loads(retained_path.read_text(encoding="utf-8"))
    retained_first = min(
        retained["raw_occupations"]["entries"], key=lambda entry: entry["sample_index"]
    )
    retained_match = {
        "sample_index": retained_first["sample_index"] == first_index,
        "N_hex": retained_first["N_hex"] == first_N.hex(),
        "bank": retained_first["bank"] == ("nue" if first_bad_flat[0] < n else "nux"),
        "node": retained_first["node"] == int(first_bad_flat[0] % n),
        "value_hex": retained_first["value_hex"] == float(occupations[first_bad_flat[0], first_index]).hex(),
    }

    node_rows = []
    boundary_failure = False
    boundary_inconclusive = False
    reconstruction_failure = False
    any_severe_cancellation = False
    all_local_physical = True
    actual_overshoot = False
    for bank, nodes in IMPLICATED.items():
        offset = 0 if bank == "nue" else n
        for node in nodes:
            retained_f = float(previous[offset + node])
            at_zero = state_collision_value(previous, previous_N, bank, node, 0.0, Y, q, qw, quad)
            at_one = state_collision_value(previous, previous_N, bank, node, 1.0, Y, q, qw, quad)
            at_retained = state_collision_value(previous, previous_N, bank, node, retained_f, Y, q, qw, quad)
            at_mid = state_collision_value(previous, previous_N, bank, node, 0.5, Y, q, qw, quad)
            g_values = {key: at_zero[key] for key in ("ordinary", "factorized_longdouble", "factorized_fsum")}
            l_values = {key: -at_one[key] for key in ("ordinary", "factorized_longdouble", "factorized_fsum")}
            g_low, g_high = min(g_values.values()), max(g_values.values())
            l_low, l_high = min(l_values.values()), max(l_values.values())
            if g_high < 0.0 or l_high < 0.0:
                boundary_failure = True
            if (g_low < 0.0 < g_high) or (l_low < 0.0 < l_high):
                boundary_inconclusive = True

            G = at_zero["ordinary"]
            L = -at_one["ordinary"]
            predicted_retained = (1.0 - retained_f) * G - retained_f * L
            predicted_mid = 0.5 * (G - L)
            scale_retained = max(abs(G) + abs(L) * retained_f, abs(at_retained["ordinary"]), np.finfo(float).tiny)
            scale_mid = max(0.5 * (abs(G) + abs(L)), abs(at_mid["ordinary"]), np.finfo(float).tiny)
            residual_retained = abs(at_retained["ordinary"] - predicted_retained) / scale_retained
            residual_mid = abs(at_mid["ordinary"] - predicted_mid) / scale_mid
            reconstruction_pass = max(residual_retained, residual_mid) <= RECON_REL_TOL
            reconstruction_failure |= not reconstruction_pass

            C = at_retained["ordinary"]
            cancellation = (abs(G) + abs(L)) / max(abs(C), np.finfo(float).tiny)
            severe = cancellation >= CANCELLATION_SEVERE
            any_severe_cancellation |= severe
            H = at_retained["H"]
            if G >= 0.0 and L >= 0.0 and math.isfinite(H) and H > 0.0:
                rate_sum = G + L
                if rate_sum == 0.0:
                    frozen = retained_f
                    lambda_delta = 0.0
                else:
                    f_star = G / rate_sum
                    lambda_delta = rate_sum / H * delta_N
                    frozen = retained_f + (f_star - retained_f) * (-math.expm1(-lambda_delta))
            else:
                frozen = math.nan
                lambda_delta = math.nan
            local_physical = math.isfinite(frozen) and 0.0 <= frozen <= 1.0
            all_local_physical &= local_physical
            next_raw = float(first[offset + node])
            if local_physical and next_raw < 0.0:
                actual_overshoot = True
            node_rows.append({
                "bank": bank,
                "node": node,
                "q": float_record(Y[node]),
                "f_previous": float_record(retained_f),
                "f_first": float_record(next_raw),
                "G": {**float_record(G), "envelope": [g_low, g_high]},
                "L": {**float_record(L), "envelope": [l_low, l_high]},
                "C": float_record(C),
                "cancellation_ratio": cancellation,
                "severe_cancellation": severe,
                "gain_loss_reconstruction": {
                    "retained_relative_residual": residual_retained,
                    "midpoint_relative_residual": residual_mid,
                    "pass": reconstruction_pass,
                },
                "H_MeV": H,
                "delta_N": delta_N,
                "lambda_delta_N": lambda_delta,
                "frozen_update": float_record(frozen),
                "frozen_update_physical": local_physical,
                "ordinary_vs_independent": {
                    "at_zero": at_zero,
                    "at_one": at_one,
                    "at_retained": at_retained,
                },
            })

    raw_nue = first[:n]
    raw_nux = first[n:2 * n]
    clipped_nue = np.clip(raw_nue, 0.0, 1.0)
    clipped_nux = np.clip(raw_nux, 0.0, 1.0)
    T, z, H, _, _, T_nue, _ = physical_context(first, first_N, Y, q, qw)
    qoi = {}
    for name, raw_values, clipped_values in (
        ("nue", raw_nue, clipped_nue),
        ("nux", raw_nux, clipped_nux),
    ):
        qoi[name] = {}
        for power, label in ((2, "number"), (3, "energy")):
            raw_m = moment(raw_values, Y, qw, power)
            clipped_m = moment(clipped_values, Y, qw, power)
            direct_delta_m = moment(raw_values - clipped_values, Y, qw, power)
            qoi[name][label] = {
                "raw": float(raw_m),
                "clipped": float(clipped_m),
                "signed_raw_minus_clipped_total_subtraction": float(raw_m - clipped_m),
                "signed_raw_minus_clipped_direct": float(direct_delta_m),
            }
    raw_neff = np.longdouble(_NEFF_PREFAC) / np.longdouble(z) ** 4 * (
        moment(raw_nue, Y, qw, 3) / np.longdouble(_I_FD)
        + 2 * moment(raw_nux, Y, qw, 3) / np.longdouble(_I_FD)
    )
    clipped_neff = np.longdouble(_NEFF_PREFAC) / np.longdouble(z) ** 4 * (
        moment(clipped_nue, Y, qw, 3) / np.longdouble(_I_FD)
        + 2 * moment(clipped_nux, Y, qw, 3) / np.longdouble(_I_FD)
    )
    direct_neff_delta = np.longdouble(_NEFF_PREFAC) / np.longdouble(z) ** 4 * (
        moment(raw_nue - clipped_nue, Y, qw, 3) / np.longdouble(_I_FD)
        + 2 * moment(raw_nux - clipped_nux, Y, qw, 3) / np.longdouble(_I_FD)
    )
    qoi["N_eff"] = {
        "raw": float(raw_neff),
        "clipped": float(clipped_neff),
        "signed_raw_minus_clipped_total_subtraction": float(raw_neff - clipped_neff),
        "signed_raw_minus_clipped_direct": float(direct_neff_delta),
    }

    raw_th_nue = raw_resample(Y, raw_nue, q, z)
    clipped_th_nue = np.clip(raw_th_nue, 0.0, 1.0)
    weak_raw = weak_rates_raw_linear(raw_th_nue, raw_th_nue, q, T, T_nue)
    weak_clipped = weak_rates_raw_linear(clipped_th_nue, clipped_th_nue, q, T, T_nue)
    qoi["weak_born_raw_linear"] = {
        "raw": weak_raw,
        "clipped": weak_clipped,
        "signed_raw_minus_clipped_total_subtraction": subtract_records(weak_raw, weak_clipped),
        "signed_raw_minus_clipped_direct": weak_rates_direct_signed_delta(
            raw_th_nue - clipped_th_nue,
            raw_th_nue - clipped_th_nue,
            q,
            T,
            T_nue,
        ),
        "note": "diagnostic linear interpolation without the production logit floor; direct delta is the authoritative signed-impact readout",
    }

    collision_qoi = {}
    for bank, raw_values in (("nue", raw_nue), ("nux", raw_nux)):
        raw_th = raw_resample(Y, raw_values, q, z)
        clipped_th = np.clip(raw_th, 0.0, 1.0)
        raw_c, _, _ = independent_collision_field(raw_th, raw_th, quad, T, bank)
        clipped_c, _, _ = independent_collision_field(clipped_th, clipped_th, quad, T, bank)
        delta_c = np.asarray(raw_c - clipped_c, dtype=np.longdouble)
        collision_qoi[bank] = {
            "max_abs_field_delta_MeV": float(np.max(np.abs(delta_c))),
            "number_moment_delta_MeV": float(moment(delta_c, q, qw, 2)),
            "energy_moment_delta_MeV": float(moment(delta_c, q, qw, 3)),
        }
    qoi["collision_field_raw_factorized"] = collision_qoi

    if boundary_inconclusive:
        selected_route = "STOP_INCONCLUSIVE"
        stop_reason = "ordinary/factorized accumulation envelope crosses zero"
    elif boundary_failure or reconstruction_failure:
        selected_route = "COLLISION_DISCRETIZATION_OR_ASSEMBLY_REPAIR"
        stop_reason = "boundary sign or gain-loss reconstruction failed"
    elif any_severe_cancellation:
        selected_route = "FACTORIZED_GAIN_LOSS_AFFINITY_RECONSTRUCTION"
        stop_reason = "inward field with severe gain-loss cancellation"
    elif all_local_physical and actual_overshoot:
        selected_route = "POSITIVITY_PRESERVING_SHORT_PREFIX_INTEGRATOR"
        stop_reason = "local exact update physical while direct-f Radau overshoots"
    else:
        selected_route = "STOP_INCONCLUSIVE"
        stop_reason = "P0 predicates do not isolate a single route"

    payload = {
        "schema": "rabbit_ode_p0_v1",
        "frozen_tolerances": {
            "rtol": RTOL,
            "atol_scalar": ATOL,
            "max_step": MAX_STEP,
            "prefix_delta_N": PREFIX_DN,
            "gain_loss_reconstruction_relative": RECON_REL_TOL,
            "severe_cancellation_threshold": CANCELLATION_SEVERE,
        },
        "environment": {
            "python": platform.python_version(),
            "numpy": np.__version__,
            "scipy": scipy.__version__,
            "platform": platform.platform(),
        },
        "provenance": {
            "retained_path": str(retained_path),
            "retained_sha256": retained_digest,
        },
        "prefix": {
            "success": bool(solution.success),
            "message": str(solution.message),
            "wall_seconds": prefix_wall,
            "nfev": int(solution.nfev),
            "njev": int(solution.njev),
            "nlu": int(solution.nlu),
            "stored_points": int(solution.t.size),
            "previous_index": previous_index,
            "first_invalid_index": first_index,
            "previous_N": float_record(previous_N),
            "first_invalid_N": float_record(first_N),
            "delta_N": float_record(delta_N),
            "first_invalid_entries": [
                {
                    "bank": "nue" if index < n else "nux",
                    "node": int(index % n),
                    "value": float_record(occupations[index, first_index]),
                }
                for index in first_bad_flat
            ],
            "retained_first_entry_match": retained_match,
        },
        "boundary_inwardness_verdict": (
            "INCONCLUSIVE" if boundary_inconclusive else "FAIL" if boundary_failure else "PASS"
        ),
        "gain_loss_reconstruction_verdict": "FAIL" if reconstruction_failure else "PASS",
        "cancellation_diagnosis": "SEVERE" if any_severe_cancellation else "NOT_SEVERE",
        "frozen_update_verdict": "PHYSICAL" if all_local_physical else "NONPHYSICAL",
        "nodes": node_rows,
        "raw_tail_qoi_impact": qoi,
        "selected_route": selected_route,
        "stop_reason": stop_reason,
        "claim_ceiling": "short-prefix causal discriminator only; no endpoint or solver validation",
    }
    print(json.dumps(json_safe(payload), indent=2, sort_keys=True, allow_nan=False))


if __name__ == "__main__":
    main()
