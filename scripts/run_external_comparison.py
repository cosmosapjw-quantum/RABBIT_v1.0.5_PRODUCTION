#!/usr/bin/env python3
"""Validate and report the serialized F-06 through F-08N anchors.

This command does not execute PRIMAT, LINX, or Rust.  The fixture records live
runs captured on 2026-07-15; reading it is a stored-anchor replay.  Live mode
fails closed because no exact external replay adapter exists here.  In
particular, the PRIMAT custom-table result is not stock PRIMAT, LINX is
independent only for its RHS/integrator because its scientific inputs are
shared, and the F-07 PRIMAT endpoint validates only ``dPa+dPe3`` because its
standard thermodynamic tables do not consume ``dPb``.  F-08A validates only
the zero-temperature Coulomb plus resummed-radiative weak layer; later weak
and neutrino corrections remain absent.

F-08B adds only the first-order finite-nucleon-mass weak slice with weak
magnetism forced off.  Its PRIMAT endpoint evidence is Python-only because the
C backend hardcodes physical weak magnetism; the stored replay therefore must
retain an explicit ``SKIPPED`` C record and cannot be reported as stock PRIMAT
or as a live external run.

F-08C activates PRIMAT's physical anomalous weak-magnetism coefficient inside
that same first-order finite-mass algebra.  Its stored evidence includes the
physical Python and C endpoints, a GL320 standalone reference, and Rust
dual-solver records.  Finite-temperature balance-restoring weak corrections
remain outside this slice.

F-08D adds the complete four-subterm finite-temperature radiative correction
over F-08C.  Its component and conditional 12-reaction endpoint evidence is
stored and validated with limits, while strict standalone parity and precision
promotion remain explicitly blocked by the unresolved table/interpolation
authority difference.

F-08N nests Rabbit's selected 31-reaction AC2024 table over the unchanged
F-08D 12-reaction backbone.  The stored replay locks the pre-Rust external
budgets, exact reaction identity, raw failed conservation attempts, and the
conditional dual-solver endpoint without promoting the inherited F-08D block.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import re
import sys
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_FIXTURE = ROOT / "tests" / "fixtures" / "flrw_gold_v861.json"
F08N_NETWORK_TABLE = ROOT / "src" / "rabbit" / "network" / "data" / "primat_ac2024_31rxn.json"
F06_KEY = "f06_matched_standard_anchors"
F07_KEY = "f07_finite_temperature_qed"
F08A_KEY = "f08a_zero_temperature_ccr"
F08B_KEY = "f08b_finite_nucleon_mass_no_weak_magnetism"
F08C_KEY = "f08c_physical_weak_magnetism"
F08D_KEY = "f08d_complete_thermal_radiative"
F08N_KEY = "f08n_selected_31_network"
OBSERVABLES = ("Yp", "DH", "He3H", "Li7H")
ENDPOINT_OBSERVABLES = (*OBSERVABLES, "Neff")
F08N_OBSERVABLES = (*OBSERVABLES, "Li6H")
F08N_PRIMAT_REACTION_ORDER = (
    "n_p__d_g",
    "d_p__He3_g",
    "d_d__He3_n",
    "d_d__t_p",
    "t_p__a_g",
    "t_d__a_n",
    "t_a__Li7_g",
    "He3_n__t_p",
    "He3_d__a_p",
    "He3_a__Be7_g",
    "Be7_n__Li7_p",
    "Li7_p__a_a",
    "Li7_p__a_a_g",
    "Be7_n__a_a",
    "Be7_d__a_a_p",
    "d_a__Li6_g",
    "Li6_p__Be7_g",
    "Li6_p__He3_a",
    "Li6_He3__a_a_p",
    "Li6_t__a_a_n",
    "Li7_He3__Li6_a",
    "Be7_t__Li6_a",
    "Li6_t__Li7_d",
    "Li6_He3__Be7_d",
    "Li7_He3__a_a_d",
    "Be7_t__a_a_d",
    "Be7_t__Li7_He3",
    "Be7_He3__p_p_a_a",
    "d_d__a_g",
    "He3_He3__a_p_p",
    "Li7_d__a_a_n",
)
F08N_REACTION_ORDER = (
    "n + p > d + g",
    "d + p > He3 + g",
    "d + d > He3 + n",
    "d + d > t + p",
    "t + p > a + g",
    "t + d > a + n",
    "t + a > Li7 + g",
    "He3 + n > t + p",
    "He3 + d > a + p",
    "He3 + a > Be7 + g",
    "Be7 + n > Li7 + p",
    "Li7 + p > a + a",
    "Li7 + p > a + a + g",
    "Be7 + n > a + a",
    "Be7 + d > 2a + p",
    "d + a > Li6 + g",
    "Li6 + p > Be7 + g",
    "Li6 + p > He3 + a",
    "Li6 + He3 > a + a  + p",
    "Li6 + t > a + a  + n",
    "Li7 + He3 > Li6 + a",
    "Be7 + t > Li6 + a",
    "Li6 + t > Li7 + d",
    "Li6 + He3 > Be7 + d",
    "Li7 + He3 > a + a + d",
    "Be7 + t > a + a + d",
    "Be7 + t > Li7 + He3",
    "Be7 + He3 > p + p + 2a",
    "d + d > a + g",
    "He3 + He3 > a + p + p",
    "Li7 + d > a + a + n",
)
F08B_CHANNELS = (
    "nu_e_n_to_p_electron",
    "electron_p_to_n_nu_e",
    "positron_n_to_p_anti_nu_e",
    "anti_nu_e_p_to_n_positron",
    "free_neutron_decay",
    "inverse_neutron_decay",
)
COMMIT_RE = re.compile(r"[0-9a-f]{40}")
SHA256_RE = re.compile(r"[0-9a-f]{64}")


class EvidenceError(RuntimeError):
    """Serialized evidence is absent, malformed, or claim-unsafe."""


def _obj(parent: dict[str, Any], key: str, path: str) -> dict[str, Any]:
    value = parent.get(key)
    if not isinstance(value, dict):
        raise EvidenceError(f"{path}.{key} must be an object")
    return value


def _exact(value: dict[str, Any], expected: dict[str, Any], path: str) -> None:
    for key, wanted in expected.items():
        if value.get(key) != wanted:
            raise EvidenceError(
                f"{path}.{key} must be {wanted!r}; found {value.get(key)!r}"
            )


def _closed_exact(
    value: dict[str, Any],
    expected: dict[str, Any],
    path: str,
    *,
    allowed_extra: tuple[str, ...] = (),
) -> None:
    """Require both exact values and a closed object schema.

    F-08N evidence is echoed into the replay report, so accepting undeclared
    fields there would permit an unvalidated claim to ride alongside validated
    evidence.  Earlier fixture schemas remain intentionally unchanged.
    """

    wanted_keys = set(expected) | set(allowed_extra)
    actual_keys = set(value)
    if actual_keys != wanted_keys:
        missing = sorted(wanted_keys - actual_keys)
        unexpected = sorted(actual_keys - wanted_keys)
        raise EvidenceError(
            f"{path} keys must be closed; missing={missing!r}, "
            f"unexpected={unexpected!r}"
        )
    _exact(value, expected, path)


def _number(value: Any, path: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise EvidenceError(f"{path} must be numeric")
    result = float(value)
    if not math.isfinite(result) or result <= 0.0:
        raise EvidenceError(f"{path} must be finite and positive")
    return result


def _finite_number(value: Any, path: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise EvidenceError(f"{path} must be numeric")
    result = float(value)
    if not math.isfinite(result):
        raise EvidenceError(f"{path} must be finite")
    return result


def _nonnegative_number(value: Any, path: str) -> float:
    result = _finite_number(value, path)
    if result < 0.0:
        raise EvidenceError(f"{path} must be non-negative")
    return result


def _nonempty_string(value: Any, path: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise EvidenceError(f"{path} must be a non-empty string")
    return value


def _outputs(record: dict[str, Any], path: str) -> dict[str, float]:
    return {key: _number(record.get(key), f"{path}.{key}") for key in OBSERVABLES}


def _endpoint_outputs(record: dict[str, Any], path: str) -> dict[str, float]:
    return {
        key: _number(record.get(key), f"{path}.{key}")
        for key in ENDPOINT_OBSERVABLES
    }


def _f08n_outputs(record: dict[str, Any], path: str) -> dict[str, float]:
    outputs = _outputs(record, path)
    outputs["Li6H"] = _nonnegative_number(record.get("Li6H"), f"{path}.Li6H")
    return outputs


def _state(record: dict[str, Any], path: str) -> dict[str, float]:
    return {key: _number(record.get(key), f"{path}.{key}") for key in ("N", "Xn")}


def _serialized_endpoint_delta(
    record: dict[str, Any],
    reference: dict[str, float],
    candidate: dict[str, float],
    path: str,
) -> None:
    for key in ENDPOINT_OBSERVABLES:
        actual = _finite_number(record.get(key), f"{path}.{key}")
        expected = candidate[key] - reference[key]
        if actual != expected:
            raise EvidenceError(f"{path}.{key} is inconsistent with serialized runs")


def _commit(source: dict[str, Any], path: str) -> None:
    value = source.get("commit")
    if not isinstance(value, str) or COMMIT_RE.fullmatch(value) is None:
        raise EvidenceError(f"{path}.commit must be a full lowercase Git SHA")


def _sha256(value: Any, path: str) -> None:
    if not isinstance(value, str) or SHA256_RE.fullmatch(value) is None:
        raise EvidenceError(f"{path} must be a full lowercase SHA-256")


def _validate_f08c(
    fixture: dict[str, Any],
    f08b_primat_outputs: dict[str, dict[str, float]],
) -> dict[str, Any]:
    f08c = _obj(fixture, F08C_KEY, "fixture")
    _exact(
        f08c,
        {
            "schema_version": "f08c_physical_weak_magnetism_v1",
            "implementation_status": "IMPLEMENTED",
            "claim_status": "VALIDATED",
            "captured_date": "2026-07-15",
            "evidence_mode": (
                "stored_results_from_live_python_c_runs_and_independent_"
                "component_integrals"
            ),
        },
        F08C_KEY,
    )
    _nonempty_string(f08c.get("scope"), f"{F08C_KEY}.scope")
    replay = _obj(f08c, "stored_replay_contract", F08C_KEY)
    _exact(
        replay,
        {"executes_external_code": False, "may_be_reported_as_a_live_run": False},
        f"{F08C_KEY}.stored_replay_contract",
    )

    convention = _obj(f08c, "convention", F08C_KEY)
    _exact(
        convention,
        {
            "finite_mass_order": "first order in inverse nucleon mass",
            "weak_magnetism_delta_kappa": 3.70589007463,
            "weak_magnetism_status": "PHYSICAL_PRIMAT_V0_3_2",
            "external_endpoint_backend": "PRIMAT v0.3.2 Python and C",
        },
        f"{F08C_KEY}.convention",
    )

    components = _obj(f08c, "component_anchors", F08C_KEY)
    _exact(components, {"execution_status": "VALIDATED"}, f"{F08C_KEY}.component_anchors")
    coefficients = _obj(components, "coefficients", f"{F08C_KEY}.component_anchors")
    _exact(coefficients, {"gA": 1.2756}, f"{F08C_KEY}.component_anchors.coefficients")
    s_plus = _obj(coefficients, "s_plus", f"{F08C_KEY}.component_anchors.coefficients")
    s_minus = _obj(coefficients, "s_minus", f"{F08C_KEY}.component_anchors.coefficients")
    _exact(
        s_plus,
        {
            "f1": 2.487954860124953,
            "f2": -1.5945873479212611,
            "f3": 0.1066324877963081,
            "mass_over_me": 1836.1526734252586,
            "sum": 0.9999999999999998,
        },
        f"{F08C_KEY}.component_anchors.coefficients.s_plus",
    )
    _exact(
        s_minus,
        {
            "f1": -1.5945873479212611,
            "f2": 2.487954860124953,
            "f3": 0.1066324877963081,
            "mass_over_me": 1838.683661717896,
            "sum": 0.9999999999999998,
        },
        f"{F08C_KEY}.component_anchors.coefficients.s_minus",
    )

    fn = _obj(components, "fn", f"{F08C_KEY}.component_anchors")
    fn_ccr = _number(fn.get("ccr_fn"), f"{F08C_KEY}.component_anchors.fn.ccr_fn")
    fn_f08b = _number(
        fn.get("f08b_no_weak_magnetism_primat_total_fn"),
        f"{F08C_KEY}.component_anchors.fn.f08b_no_weak_magnetism_primat_total_fn",
    )
    fn_primat_delta = _finite_number(
        fn.get("primat_physical_finite_mass_delta_fn"),
        f"{F08C_KEY}.component_anchors.fn.primat_physical_finite_mass_delta_fn",
    )
    fn_primat_total = _number(
        fn.get("f08c_physical_primat_total_fn"),
        f"{F08C_KEY}.component_anchors.fn.f08c_physical_primat_total_fn",
    )
    fn_compact_delta = _finite_number(
        fn.get("independent_compact_physical_finite_mass_delta_fn"),
        f"{F08C_KEY}.component_anchors.fn.independent_compact_physical_finite_mass_delta_fn",
    )
    fn_compact_total = _number(
        fn.get("independent_compact_physical_total_fn"),
        f"{F08C_KEY}.component_anchors.fn.independent_compact_physical_total_fn",
    )
    compact_minus_primat = _finite_number(
        fn.get("compact_minus_primat_physical_delta_fn"),
        f"{F08C_KEY}.component_anchors.fn.compact_minus_primat_physical_delta_fn",
    )
    isolated_wm = _finite_number(
        fn.get("isolated_weak_magnetism_delta_fn"),
        f"{F08C_KEY}.component_anchors.fn.isolated_weak_magnetism_delta_fn",
    )
    _exact(
        fn,
        {
            "ccr_fn": 1.7583843867571942,
            "f08b_no_weak_magnetism_primat_total_fn": 1.754764223193087,
            "primat_physical_finite_mass_delta_fn": -0.0036333405331918645,
            "f08c_physical_primat_total_fn": 1.7547510462240024,
            "independent_compact_physical_finite_mass_delta_fn": (
                -0.003633340905019523
            ),
            "independent_compact_physical_total_fn": 1.7547510458521747,
            "compact_minus_primat_physical_delta_fn": -3.718276585067126e-10,
            "isolated_weak_magnetism_delta_fn": -1.3177089418761959e-5,
        },
        f"{F08C_KEY}.component_anchors.fn",
    )
    compact_f08b_delta = -0.003620163815600761
    if fn_primat_total != fn_ccr + fn_primat_delta:
        raise EvidenceError(f"{F08C_KEY}.component_anchors.fn PRIMAT identity is inconsistent")
    if fn_compact_total != fn_ccr + fn_compact_delta:
        raise EvidenceError(f"{F08C_KEY}.component_anchors.fn compact identity is inconsistent")
    if compact_minus_primat != fn_compact_delta - fn_primat_delta:
        raise EvidenceError(f"{F08C_KEY}.component_anchors.fn compact/PRIMAT delta is inconsistent")
    if isolated_wm != fn_compact_delta - compact_f08b_delta:
        raise EvidenceError(f"{F08C_KEY}.component_anchors.fn isolated weak-magnetism delta is inconsistent")
    if fn_f08b != 1.754764223193087 or fn_primat_total != 1.7547510462240024:
        raise EvidenceError(f"{F08C_KEY}.component_anchors.fn physical/no-WM anchors drifted")
    no_ccr = _obj(fn, "no_ccr_negative_control", f"{F08C_KEY}.component_anchors.fn")
    _exact(
        no_ccr,
        {
            "no_weak_magnetism_finite_mass_delta_fn": -0.0033828435120336585,
            "physical_weak_magnetism_finite_mass_delta_fn": -0.0033828435120336585,
            "isolated_weak_magnetism_delta_fn": 0.0,
            "required_max_absolute_shift": 2e-15,
        },
        f"{F08C_KEY}.component_anchors.fn.no_ccr_negative_control",
    )

    compact_grid = _obj(
        components, "compact_vs_expanded_point_grid", f"{F08C_KEY}.component_anchors"
    )
    _exact(
        compact_grid,
        {
            "row_count": 252,
            "max_absolute_difference": 1.0769163338864018e-14,
            "max_scale_relative_difference": 1.7001870954891971e-12,
        },
        f"{F08C_KEY}.component_anchors.compact_vs_expanded_point_grid",
    )
    compact_abs = _nonnegative_number(
        compact_grid.get("max_absolute_difference"),
        f"{F08C_KEY}.component_anchors.compact_vs_expanded_point_grid.max_absolute_difference",
    )
    compact_rel = _nonnegative_number(
        compact_grid.get("max_scale_relative_difference"),
        f"{F08C_KEY}.component_anchors.compact_vs_expanded_point_grid.max_scale_relative_difference",
    )
    if compact_abs >= 1e-12 or compact_rel >= 1e-9:
        raise EvidenceError(f"{F08C_KEY}.component_anchors compact/expanded ceiling exceeded")

    equal = components.get("equal_temperature_rates")
    if not isinstance(equal, list) or len(equal) != 3:
        raise EvidenceError(f"{F08C_KEY}.component_anchors.equal_temperature_rates must have three rows")
    expected_equal_rates = (
        (0.3, 0.009519719396921386, 0.0001280075006855248),
        (1.0, 1.5419344851097214, 0.42392801552626175),
        (3.0, 253.95798518629175, 165.3782209778814),
    )
    for index, (temperature, neutron_to_proton, proton_to_neutron) in enumerate(
        expected_equal_rates
    ):
        row = equal[index]
        path = f"{F08C_KEY}.component_anchors.equal_temperature_rates[{index}]"
        if not isinstance(row, dict):
            raise EvidenceError(f"{path} must be an object")
        _exact(
            row,
            {
                "photon_temperature_mev": temperature,
                "neutrino_temperature_mev": temperature,
                "neutron_to_proton_per_second": neutron_to_proton,
                "proton_to_neutron_per_second": proton_to_neutron,
            },
            path,
        )
        _number(row.get("neutron_to_proton_per_second"), f"{path}.neutron_to_proton_per_second")
        _number(row.get("proton_to_neutron_per_second"), f"{path}.proton_to_neutron_per_second")

    unequal = _obj(
        components, "unequal_temperature_six_channel_anchor", f"{F08C_KEY}.component_anchors"
    )
    _exact(
        unequal,
        {"photon_temperature_mev": 1.0, "neutrino_temperature_mev": 0.8},
        f"{F08C_KEY}.component_anchors.unequal_temperature_six_channel_anchor",
    )
    channels = _obj(
        unequal,
        "channels_per_second",
        f"{F08C_KEY}.component_anchors.unequal_temperature_six_channel_anchor",
    )
    _exact(
        channels,
        {
            "nu_e_n_to_p_electron": 0.29248953906930353,
            "electron_p_to_n_nu_e": 0.22293503132863313,
            "positron_n_to_p_anti_nu_e": 0.748250761627631,
            "anti_nu_e_p_to_n_positron": 0.05361888257945413,
            "free_neutron_decay": 0.0005050673142472589,
            "inverse_neutron_decay": 0.00012307519823550762,
        },
        (
            f"{F08C_KEY}.component_anchors."
            "unequal_temperature_six_channel_anchor.channels_per_second"
        ),
    )
    if set(channels) != set(F08B_CHANNELS):
        raise EvidenceError(f"{F08C_KEY}.component_anchors unequal anchor must contain six named channels")
    channel_values = {
        key: _number(
            channels.get(key),
            f"{F08C_KEY}.component_anchors.unequal_temperature_six_channel_anchor.channels_per_second.{key}",
        )
        for key in F08B_CHANNELS
    }
    totals = _obj(
        unequal,
        "totals_per_second",
        f"{F08C_KEY}.component_anchors.unequal_temperature_six_channel_anchor",
    )
    _exact(
        totals,
        {
            "neutron_to_proton": 1.0412453680111817,
            "proton_to_neutron": 0.27667698910632277,
        },
        (
            f"{F08C_KEY}.component_anchors."
            "unequal_temperature_six_channel_anchor.totals_per_second"
        ),
    )
    n_to_p = _number(totals.get("neutron_to_proton"), f"{F08C_KEY}.component_anchors.unequal_temperature_six_channel_anchor.totals_per_second.neutron_to_proton")
    p_to_n = _number(totals.get("proton_to_neutron"), f"{F08C_KEY}.component_anchors.unequal_temperature_six_channel_anchor.totals_per_second.proton_to_neutron")
    expected_n_to_p = channel_values["nu_e_n_to_p_electron"] + channel_values["positron_n_to_p_anti_nu_e"] + channel_values["free_neutron_decay"]
    expected_p_to_n = channel_values["electron_p_to_n_nu_e"] + channel_values["anti_nu_e_p_to_n_positron"] + channel_values["inverse_neutron_decay"]
    if not math.isclose(n_to_p, expected_n_to_p, rel_tol=0.0, abs_tol=max(math.ulp(n_to_p), math.ulp(expected_n_to_p))) or not math.isclose(p_to_n, expected_p_to_n, rel_tol=0.0, abs_tol=max(math.ulp(p_to_n), math.ulp(expected_p_to_n))):
        raise EvidenceError(f"{F08C_KEY}.component_anchors unequal channel totals are inconsistent")

    balance = _obj(components, "modified_detailed_balance", f"{F08C_KEY}.component_anchors")
    _exact(
        balance,
        {
            "temperature_mev": 1.0,
            "target_name": "(mn/mp)^(3/2)*exp(-Q/T)",
            "target_ratio": 0.27492246823324173,
            "observed_total_ratio": 0.27493257308924884,
            "relative_residual": 3.6755293490786656e-5,
            "recorded_grid_max_absolute_total_relative_residual": (
                0.00011084594732979625
            ),
            "recorded_grid_max_absolute_pair_relative_residual": (
                0.00037789555986433854
            ),
        },
        f"{F08C_KEY}.component_anchors.modified_detailed_balance",
    )
    target = _number(balance.get("target_ratio"), f"{F08C_KEY}.component_anchors.modified_detailed_balance.target_ratio")
    observed = _number(balance.get("observed_total_ratio"), f"{F08C_KEY}.component_anchors.modified_detailed_balance.observed_total_ratio")
    residual = _finite_number(balance.get("relative_residual"), f"{F08C_KEY}.component_anchors.modified_detailed_balance.relative_residual")
    total_ceiling = _number(balance.get("recorded_grid_max_absolute_total_relative_residual"), f"{F08C_KEY}.component_anchors.modified_detailed_balance.recorded_grid_max_absolute_total_relative_residual")
    pair_ceiling = _number(balance.get("recorded_grid_max_absolute_pair_relative_residual"), f"{F08C_KEY}.component_anchors.modified_detailed_balance.recorded_grid_max_absolute_pair_relative_residual")
    serialized_residual = (observed - target) / target
    if residual == 0.0 or not math.isclose(
        residual, serialized_residual, rel_tol=0.0, abs_tol=1e-15
    ):
        raise EvidenceError(f"{F08C_KEY}.component_anchors modified detailed-balance residual is zero or inconsistent")
    if abs(residual) > total_ceiling or total_ceiling >= 2e-4 or pair_ceiling >= 5e-4:
        raise EvidenceError(f"{F08C_KEY}.component_anchors modified detailed-balance ceiling exceeded")

    controls = _obj(components, "numerical_controls", f"{F08C_KEY}.component_anchors")
    _exact(
        controls,
        {
            "max_gl160_minus_gl320_relative": 5.276245651549528e-8,
            "required_max_gl160_minus_gl320_relative": 1e-6,
            "max_adaptive_2e10_minus_2e12_relative": 2.0851767116981855e-13,
            "required_max_adaptive_2e10_minus_2e12_relative": 1e-9,
        },
        f"{F08C_KEY}.component_anchors.numerical_controls",
    )
    if _nonnegative_number(controls.get("max_gl160_minus_gl320_relative"), f"{F08C_KEY}.component_anchors.numerical_controls.max_gl160_minus_gl320_relative") >= 1e-6:
        raise EvidenceError(f"{F08C_KEY}.component_anchors GL160/320 convergence failed")
    if _nonnegative_number(controls.get("max_adaptive_2e10_minus_2e12_relative"), f"{F08C_KEY}.component_anchors.numerical_controls.max_adaptive_2e10_minus_2e12_relative") >= 1e-9:
        raise EvidenceError(f"{F08C_KEY}.component_anchors adaptive convergence failed")

    freezeout = _obj(f08c, "standalone_temperature_variable_freezeout", F08C_KEY)
    _exact(
        freezeout,
        {"schema_version": "f08c_temperature_variable_freezeout_v1", "execution_status": "VALIDATED"},
        f"{F08C_KEY}.standalone_temperature_variable_freezeout",
    )
    _nonempty_string(freezeout.get("claim_scope"), f"{F08C_KEY}.standalone_temperature_variable_freezeout.claim_scope")
    freezeout_config = _obj(freezeout, "configuration", f"{F08C_KEY}.standalone_temperature_variable_freezeout")
    _exact(
        freezeout_config,
        {
            "temperature_start_mev": 10.0,
            "activation_temperature_mev": 0.86164715286738,
            "temperature_end_mev": 0.05,
            "neutron_lifetime_seconds": 878.4,
            "rate_grid_points": 321,
            "gauss_legendre_nodes_per_panel": [80, 160, 320],
            "reference_order": 320,
            "qed_model": "PrimatLeadingE2E3",
            "weak_magnetism_delta_kappa": 3.70589007463,
        },
        f"{F08C_KEY}.standalone_temperature_variable_freezeout.configuration",
    )
    freezeout_states: dict[str, dict[str, dict[str, float]]] = {}
    expected_freezeout_states = {
        "f08b_no_weak_magnetism_gl320": {
            "activation": {"N": 2.4593217700143666, "Xn": 0.24009429546635033},
            "final": {"N": 5.63289932832902, "Xn": 0.0892259942459267},
        },
        "f08c_physical_weak_magnetism_gl320": {
            "activation": {"N": 2.4593217700143666, "Xn": 0.24019626189219623},
            "final": {"N": 5.63289932832902, "Xn": 0.08928032696742821},
        },
    }
    for name in (
        "f08b_no_weak_magnetism_gl320",
        "f08c_physical_weak_magnetism_gl320",
    ):
        record = _obj(freezeout, name, f"{F08C_KEY}.standalone_temperature_variable_freezeout")
        _exact(
            record,
            {"success": True, **expected_freezeout_states[name]},
            f"{F08C_KEY}.standalone_temperature_variable_freezeout.{name}",
        )
        freezeout_states[name] = {
            endpoint: _state(
                _obj(record, endpoint, f"{F08C_KEY}.standalone_temperature_variable_freezeout.{name}"),
                f"{F08C_KEY}.standalone_temperature_variable_freezeout.{name}.{endpoint}",
            )
            for endpoint in ("activation", "final")
        }
    for endpoint in ("activation", "final"):
        if freezeout_states["f08b_no_weak_magnetism_gl320"][endpoint]["N"] != freezeout_states["f08c_physical_weak_magnetism_gl320"][endpoint]["N"]:
            raise EvidenceError(f"{F08C_KEY}.standalone_temperature_variable_freezeout entropy identity drifted")
    freezeout_delta = _obj(freezeout, "f08c_minus_f08b_gl320", f"{F08C_KEY}.standalone_temperature_variable_freezeout")
    for endpoint, key in (("activation", "activation_Xn"), ("final", "final_Xn")):
        actual = _finite_number(freezeout_delta.get(key), f"{F08C_KEY}.standalone_temperature_variable_freezeout.f08c_minus_f08b_gl320.{key}")
        expected = freezeout_states["f08c_physical_weak_magnetism_gl320"][endpoint]["Xn"] - freezeout_states["f08b_no_weak_magnetism_gl320"][endpoint]["Xn"]
        if actual != expected or actual <= 0.0:
            raise EvidenceError(f"{F08C_KEY}.standalone_temperature_variable_freezeout {key} is inconsistent")
    repeat = _obj(freezeout, "repeat_determinism", f"{F08C_KEY}.standalone_temperature_variable_freezeout")
    _exact(repeat, {"f08b_gl320_exact_float_equality": True, "f08c_gl320_exact_float_equality": True}, f"{F08C_KEY}.standalone_temperature_variable_freezeout.repeat_determinism")
    stability = _obj(freezeout, "numerical_stability", f"{F08C_KEY}.standalone_temperature_variable_freezeout")
    _exact(
        stability,
        {
            "gauss_legendre_nodes_per_panel": [80, 160, 320],
            "f08b_max_absolute_Xn_gl320_minus_gl160": 7.920615829881683e-9,
            "f08c_max_absolute_Xn_gl320_minus_gl160": 9.259088912250135e-9,
            "required_max_absolute_Xn_gl320_minus_gl160": 2e-8,
        },
        f"{F08C_KEY}.standalone_temperature_variable_freezeout.numerical_stability",
    )
    if _nonnegative_number(stability.get("f08b_max_absolute_Xn_gl320_minus_gl160"), f"{F08C_KEY}.standalone_temperature_variable_freezeout.numerical_stability.f08b_max_absolute_Xn_gl320_minus_gl160") >= 2e-8 or _nonnegative_number(stability.get("f08c_max_absolute_Xn_gl320_minus_gl160"), f"{F08C_KEY}.standalone_temperature_variable_freezeout.numerical_stability.f08c_max_absolute_Xn_gl320_minus_gl160") >= 2e-8:
        raise EvidenceError(f"{F08C_KEY}.standalone_temperature_variable_freezeout GL320 convergence failed")
    freezeout_acceptance = _obj(freezeout, "rust_acceptance", f"{F08C_KEY}.standalone_temperature_variable_freezeout")
    _exact(
        freezeout_acceptance,
        {
            "reference": "f08c_physical_weak_magnetism_gl320",
            "max_absolute_N_difference": 3e-7,
            "max_absolute_Xn_difference": 6e-8,
            "frozen_before_any_rust_value_was_read": True,
        },
        f"{F08C_KEY}.standalone_temperature_variable_freezeout.rust_acceptance",
    )
    _nonempty_string(freezeout_acceptance.get("basis"), f"{F08C_KEY}.standalone_temperature_variable_freezeout.rust_acceptance.basis")

    endpoint = _obj(f08c, "primat_matched_endpoint", F08C_KEY)
    _exact(
        endpoint,
        {"execution_status": "VALIDATED", "version": "0.3.2", "backend": "python_and_c"},
        f"{F08C_KEY}.primat_matched_endpoint",
    )
    _commit(endpoint, f"{F08C_KEY}.primat_matched_endpoint")
    endpoint_outputs = {
        name: _endpoint_outputs(_obj(endpoint, name, f"{F08C_KEY}.primat_matched_endpoint"), f"{F08C_KEY}.primat_matched_endpoint.{name}")
        for name in ("f08b_python_reference", "f08c_python", "f08c_c")
    }
    expected_endpoint_outputs = {
        "f08b_python_reference": {
            "Yp": 0.2466842688457449,
            "DH": 2.455665403302237e-5,
            "He3H": 1.0433375736211641e-5,
            "Li7H": 5.486433620053931e-10,
            "Neff": 3.009645190024071,
        },
        "f08c_python": {
            "Yp": 0.24683601806557295,
            "DH": 2.4565015281772263e-5,
            "He3H": 1.0434637936584758e-5,
            "Li7H": 5.488592195819055e-10,
            "Neff": 3.009645190024071,
        },
        "f08c_c": {
            "Yp": 0.24683586374148064,
            "DH": 2.4565071974942807e-5,
            "He3H": 1.0434644927261921e-5,
            "Li7H": 5.488576741277125e-10,
            "Neff": 3.009645190024071,
        },
    }
    if endpoint_outputs != expected_endpoint_outputs:
        raise EvidenceError(f"{F08C_KEY}.primat_matched_endpoint outputs drifted")
    if endpoint_outputs["f08b_python_reference"] != f08b_primat_outputs["f08b_python"]:
        raise EvidenceError(f"{F08C_KEY}.primat_matched_endpoint immutable F08B comparator drifted")
    _serialized_endpoint_delta(
        _obj(endpoint, "f08c_python_minus_f08b_python", f"{F08C_KEY}.primat_matched_endpoint"),
        endpoint_outputs["f08b_python_reference"],
        endpoint_outputs["f08c_python"],
        f"{F08C_KEY}.primat_matched_endpoint.f08c_python_minus_f08b_python",
    )
    _serialized_endpoint_delta(
        _obj(endpoint, "f08c_c_minus_python", f"{F08C_KEY}.primat_matched_endpoint"),
        endpoint_outputs["f08c_python"],
        endpoint_outputs["f08c_c"],
        f"{F08C_KEY}.primat_matched_endpoint.f08c_c_minus_python",
    )
    repeat = _obj(endpoint, "repeat_determinism", f"{F08C_KEY}.primat_matched_endpoint")
    for backend in ("python", "c"):
        _exact(
            _obj(repeat, backend, f"{F08C_KEY}.primat_matched_endpoint.repeat_determinism"),
            {"repeat_count": 2, "exact_float_equality": True},
            f"{F08C_KEY}.primat_matched_endpoint.repeat_determinism.{backend}",
        )
    endpoint_acceptance = _obj(endpoint, "rust_acceptance", f"{F08C_KEY}.primat_matched_endpoint")
    _exact(endpoint_acceptance, {"frozen_before_any_rust_value_was_read": True}, f"{F08C_KEY}.primat_matched_endpoint.rust_acceptance")
    _exact(
        _obj(endpoint_acceptance, "direct_endpoint_budgets", f"{F08C_KEY}.primat_matched_endpoint.rust_acceptance"),
        {"Yp_absolute": 2e-5, "DH_relative": 1e-3, "He3H_relative": 2e-3, "Li7H_relative": 4e-3},
        f"{F08C_KEY}.primat_matched_endpoint.rust_acceptance.direct_endpoint_budgets",
    )
    _exact(
        _obj(endpoint_acceptance, "isolated_f08c_minus_f08b_relative_ceilings", f"{F08C_KEY}.primat_matched_endpoint.rust_acceptance"),
        {"Yp": 0.0025, "DH": 0.015, "He3H": 0.015, "Li7H": 0.015},
        f"{F08C_KEY}.primat_matched_endpoint.rust_acceptance.isolated_f08c_minus_f08b_relative_ceilings",
    )

    rust = _obj(f08c, "rust", F08C_KEY)
    _exact(rust, {"execution_status": "VALIDATED"}, f"{F08C_KEY}.rust")
    _exact(
        _obj(rust, "profile", f"{F08C_KEY}.rust"),
        {
            "qed_model": "PrimatLeadingE2E3",
            "weak_model": "PrimatZeroTemperatureCcrFiniteMassPhysicalWeakMagnetism",
            "finite_mass_order": "first order in inverse nucleon mass",
            "weak_magnetism_delta_kappa": 3.70589007463,
            "weak_quadrature_order": 64,
            "weak_leg_relative_tolerance": 2e-8,
            "coupled_relative_tolerance": 1e-9,
            "reaction_count": 12,
            "neutrino_treatment": "ideal high-temperature instantaneous decoupling",
        },
        f"{F08C_KEY}.rust.profile",
    )
    expected_rust = {
        "bdf": {
            "solver": "BDF", "Xn_handoff": 0.24019629115929314,
            "Yp": 0.2468369814321261, "DH": 2.4564568596661556e-5,
            "He3H": 1.0434585907028622e-5, "Li7H": 5.488720109289649e-10,
        },
        "rodas5p": {
            "solver": "Rodas5P", "Xn_handoff": 0.24019629154783906,
            "Yp": 0.24683701452683648, "DH": 2.45645716039475e-5,
            "He3H": 1.0434586301065364e-5, "Li7H": 5.488719893518859e-10,
        },
    }
    rust_outputs: dict[str, dict[str, float]] = {}
    for name, expected in expected_rust.items():
        record = _obj(rust, name, f"{F08C_KEY}.rust")
        _exact(record, expected, f"{F08C_KEY}.rust.{name}")
        rust_outputs[name] = _outputs(record, f"{F08C_KEY}.rust.{name}")

    provenance = _obj(f08c, "provenance", F08C_KEY)
    _exact(
        provenance,
        {
            "primat_version": "0.3.2", "primat_tag": "v0.3.2",
            "python": "3.12.3", "numpy": "2.5.1", "scipy": "1.18.0",
            "physical_delta_kappa": 3.70589007463,
            "validation_checks_passed": 23, "validation_checks_total": 23,
            "mutation_probes_passed": 5, "mutation_probes_total": 5,
        },
        f"{F08C_KEY}.provenance",
    )
    _commit({"commit": provenance.get("primat_commit")}, f"{F08C_KEY}.provenance")
    for key in (
        "result_sha256", "raw_sha256", "validation_sha256", "runner_sha256",
        "validator_sha256", "artifact_manifest_sha256", "primat_c_backend_sha256", "corrections_source_sha256",
        "integrands_source_sha256", "primat_module_sha256", "rabbit_rate_source_sha256",
        "dense_custom_network_sha256", "f07_electron_thermo_cache_sha256",
        "f08b_immutable_result_sha256", "mutation_probe_result_sha256",
        "mutation_probe_script_sha256",
    ):
        _sha256(provenance.get(key), f"{F08C_KEY}.provenance.{key}")
    _exact(
        provenance,
        {
            "result_sha256": "9f876046e8d882be8a9fe66d39152e41f0d55723974f0af06559b618b25f9785",
            "raw_sha256": "5a0eeb0abb0b7fd986171265ef59dc1b85beb5bd918de2f65f2db72dd85eff5d",
            "runner_sha256": "0fa77514cdd9fa64b94f7d80fa5cfee9f2d187c8b3363f4e750968fc2a557b0a",
            "validator_sha256": "92fd6f77b58668699e49da888ecb56b1bbc27f9132621f9a08b586c020e2bbc2",
            "validation_sha256": "e6df83142c412f2f56892f200af6cd74a9692ccce06d55a1c9c115ec13a3f22b",
            "artifact_manifest_path": "/tmp/rabbit_f08c_external_20260715/f08c_final_artifact_hashes.json",
            "artifact_manifest_sha256": "4ec5b95c4ef55024ffd2f009c3d889e509e1ee69e3016b2b9b7c19c1da949920",
            "f08b_immutable_result_sha256": "c3e4536c3051e0cf416ec1942276b4f1873ca60154302bf2bb2d50e9a5705bd2",
            "primat_c_backend_sha256": "5151345bb815936c56eb63afdc1660c16d750098b11fa7ca4e149893f2d6238e",
            "corrections_source_sha256": "191e95ae91b903c26e63d9b476f8b5b684b7dcd9c88348d66ae1d16f01a265f6",
            "integrands_source_sha256": "17f033778065868b3c7461628d0c96b7dd69f2dd3a3c0bd09577e1db08702685",
            "primat_module_sha256": "b1d015b32736cb78b2eebd33b08244b15e5bf6ac69028462a7dd72335881493d",
            "rabbit_rate_source_sha256": "8dacb12bece202a798bae67f2ca89d1fad62f9f02bad38da0f59c19147960b85",
            "dense_custom_network_sha256": "fb30603a1e564fe5cbc22d7610a0ce4f65c3c3faa9653c8b43f2ddb491c4b4e2",
            "f07_electron_thermo_cache_sha256": "235459a6c1259530a7ce2d3a349db8e48373879b975172665061b9e35a7138cb",
            "mutation_probe_result_sha256": "1490804a6b8cd172a5dd68877f9a34a2092c08e5bc1e5fb9d5550c62eeb4f134",
            "mutation_probe_script_sha256": "80fcf0f201c3df872f44056f6b399d348658aa01a4852ff26d5189719d2a8fb9",
        },
        f"{F08C_KEY}.provenance",
    )
    limitations = f08c.get("claim_limitations")
    if not isinstance(limitations, list) or len(limitations) < 6 or not all(isinstance(item, str) and item for item in limitations):
        raise EvidenceError(f"{F08C_KEY}.claim_limitations must contain at least six non-empty strings")
    limitation_text = " ".join(limitations).lower()
    for required in ("not stock", "first order", "weak-only", "qke", "public production"):
        if required not in limitation_text:
            raise EvidenceError(f"{F08C_KEY}.claim_limitations must retain {required!r}")

    return {
        "f08c": f08c,
        "f08c_primat": endpoint_outputs,
        "f08c_rust": rust_outputs,
    }


def _validate_f08d(
    fixture: dict[str, Any],
    f08c_rust_outputs: dict[str, dict[str, float]],
) -> dict[str, Any]:
    f08d = _obj(fixture, F08D_KEY, "fixture")
    _exact(
        f08d,
        {
            "schema_version": "f08d_complete_thermal_radiative_v1",
            "implementation_status": "IMPLEMENTED",
            "claim_status": "VALIDATED",
            "promotion_status": "BLOCKED",
            "captured_date": "2026-07-15",
        },
        F08D_KEY,
    )
    validation_scope = _nonempty_string(
        f08d.get("validation_scope"), f"{F08D_KEY}.validation_scope"
    ).lower()
    if "conditional" not in validation_scope or "blocked" not in validation_scope:
        raise EvidenceError(
            f"{F08D_KEY}.validation_scope must retain conditional and blocked scope"
        )
    _nonempty_string(f08d.get("scope"), f"{F08D_KEY}.scope")

    convention = _obj(f08d, "convention", F08D_KEY)
    _exact(
        convention,
        {
            "radiative_order": (
                "O(alpha) at infinite nucleon mass, additive to the first-order "
                "inverse-nucleon-mass F08C model; mixed alpha*T/M terms absent"
            ),
            "normalization": "unchanged F08C physical-WM Fn*tau_n",
            "fn_with_thermal_off": 1.7547510462240024,
            "fn_with_thermal_on": 1.7547510462240024,
            "directional_storage": (
                "separate complete n_to_p and p_to_n corrections; not assigned "
                "to any of the six Born channels"
            ),
            "chitilde_negative_tail": (
                "physical FD saturation to one; PRIMAT's symmetric zeroing beyond "
                "|argument|=300 is retained only as an external numerical comparator"
            ),
            "second_soft_subtraction": (
                "F_plus preserved as authorized by Phys. Rep. B48-B49"
            ),
        },
        f"{F08D_KEY}.convention",
    )
    if convention["fn_with_thermal_on"] != convention["fn_with_thermal_off"]:
        raise EvidenceError(f"{F08D_KEY}.convention thermal correction changed Fn")

    direct = _obj(f08d, "direct_component_validation", F08D_KEY)
    _exact(
        direct,
        {
            "execution_status": "VALIDATED",
            "external_evidence_claim_status": "VALIDATED_WITH_LIMITS",
            "frozen_before_any_rust_f08d_value_was_read": True,
            "profile": "flat Tnu/Tgamma=0.7138 diagnostic only",
        },
        f"{F08D_KEY}.direct_component_validation",
    )
    quadrature = _obj(
        direct, "rust_quadrature", f"{F08D_KEY}.direct_component_validation"
    )
    _exact(
        quadrature,
        {
            "electron_log_panels": 8,
            "electron_q_boundary": True,
            "photon_boundaries_per_electron": ["k=|E-s*q|", "k=|E+s*q|"],
            "orders_compared": [64, 128],
        },
        f"{F08D_KEY}.direct_component_validation.rust_quadrature",
    )

    anchors = direct.get("complete_sum_anchors")
    expected_anchors = (
        {
            "temperature_K": 200000000.0,
            "direction": "n_to_p",
            "external_reference": 5.993377680674609e-07,
            "criterion": "relative <= 0.15",
            "rust_order_64": 6.00384962069158e-07,
            "rust_order_128": 5.989797076187989e-07,
        },
        {
            "temperature_K": 200000000.0,
            "direction": "p_to_n",
            "external_reference": 1.1714204083071363e-37,
            "criterion": "absolute < 1e-35",
            "rust_order_64": 1.0968755339590862e-37,
            "rust_order_128": 1.096809855547756e-37,
        },
        {
            "temperature_K": 1000000000.0,
            "direction": "n_to_p",
            "external_reference": 0.0001854915,
            "criterion": "relative <= 0.03",
            "rust_order_64": 0.00018581569907234868,
            "rust_order_128": 0.00018591814454086805,
        },
        {
            "temperature_K": 1000000000.0,
            "direction": "p_to_n",
            "external_reference": 2.822955660681275e-10,
            "criterion": "factor in [0.2,5]",
            "rust_order_64": 2.809324270680531e-10,
            "rust_order_128": 2.8093256779332847e-10,
        },
        {
            "temperature_K": 10000000000.0,
            "direction": "n_to_p",
            "external_reference": -1.6784941,
            "criterion": "relative to signed reference <= 0.01",
            "rust_order_64": -1.6772381250859494,
            "rust_order_128": -1.677764350577508,
        },
        {
            "temperature_K": 10000000000.0,
            "direction": "p_to_n",
            "external_reference": 0.48664922,
            "criterion": "relative <= 0.01",
            "rust_order_64": 0.4867800349552119,
            "rust_order_128": 0.48672261467967265,
        },
    )
    if not isinstance(anchors, list) or len(anchors) != len(expected_anchors):
        raise EvidenceError(
            f"{F08D_KEY}.direct_component_validation.complete_sum_anchors "
            "must contain exactly six records"
        )
    for index, (record, expected) in enumerate(zip(anchors, expected_anchors)):
        path = f"{F08D_KEY}.direct_component_validation.complete_sum_anchors[{index}]"
        if not isinstance(record, dict):
            raise EvidenceError(f"{path} must be an object")
        _exact(record, expected, path)
        reference = _finite_number(record.get("external_reference"), f"{path}.external_reference")
        values = (
            _finite_number(record.get("rust_order_64"), f"{path}.rust_order_64"),
            _finite_number(record.get("rust_order_128"), f"{path}.rust_order_128"),
        )
        criterion = record["criterion"]
        if criterion == "absolute < 1e-35":
            accepted = all(abs(value) < 1e-35 for value in values)
        elif criterion == "factor in [0.2,5]":
            accepted = all(0.2 <= value / reference <= 5.0 for value in values)
        else:
            ceiling = 0.15 if criterion == "relative <= 0.15" else (
                0.03 if criterion == "relative <= 0.03" else 0.01
            )
            accepted = all(abs((value - reference) / reference) <= ceiling for value in values)
        if not accepted:
            raise EvidenceError(f"{path} violates its serialized complete-sum contract")

    subterm = _obj(
        direct,
        "resolved_independent_subterm_anchor",
        f"{F08D_KEY}.direct_component_validation",
    )
    _exact(
        subterm,
        {
            "temperature_K": 200000000.0,
            "direction": "n_to_p",
            "acceptance": (
                "true/L1 relative 1e-3; differential-bremsstrahlung relative "
                "0.15; L2+3 absolute max(1e-12,0.2*|reference|)"
            ),
        },
        f"{F08D_KEY}.direct_component_validation.resolved_independent_subterm_anchor",
    )
    external_subterms = _obj(
        subterm,
        "external_dblquad",
        f"{F08D_KEY}.direct_component_validation.resolved_independent_subterm_anchor",
    )
    rust_subterms = _obj(
        subterm,
        "rust_order_128",
        f"{F08D_KEY}.direct_component_validation.resolved_independent_subterm_anchor",
    )
    _exact(
        external_subterms,
        {
            "true_photon": 2.2079759941277295e-05,
            "differential_bremsstrahlung": 3.8311930715818396e-07,
            "L1": -2.1863541480364965e-05,
            "L2_plus_3": -3.0531270769295694e-18,
        },
        f"{F08D_KEY}.direct_component_validation.resolved_independent_subterm_anchor.external_dblquad",
    )
    _exact(
        rust_subterms,
        {
            "true_photon": 2.208027568768772e-05,
            "differential_bremsstrahlung": 3.8315126077354416e-07,
            "L1": -2.1864447241008618e-05,
            "L2_plus_3": 1.6615312272499778e-16,
        },
        f"{F08D_KEY}.direct_component_validation.resolved_independent_subterm_anchor.rust_order_128",
    )
    for key, ceiling in (("true_photon", 1e-3), ("L1", 1e-3), ("differential_bremsstrahlung", 0.15)):
        reference = _finite_number(external_subterms.get(key), f"external_dblquad.{key}")
        candidate = _finite_number(rust_subterms.get(key), f"rust_order_128.{key}")
        if abs((candidate - reference) / reference) > ceiling:
            raise EvidenceError(f"{F08D_KEY} resolved {key} subterm exceeds its ceiling")
    l23_reference = _finite_number(external_subterms.get("L2_plus_3"), "external_dblquad.L2_plus_3")
    l23_candidate = _finite_number(rust_subterms.get("L2_plus_3"), "rust_order_128.L2_plus_3")
    if abs(l23_candidate - l23_reference) > max(1e-12, 0.2 * abs(l23_reference)):
        raise EvidenceError(f"{F08D_KEY} resolved L2+3 subterm exceeds its ceiling")
    _nonempty_string(
        direct.get("tiny_p_to_n_subterm_limit"),
        f"{F08D_KEY}.direct_component_validation.tiny_p_to_n_subterm_limit",
    )
    saturation_delta = _nonnegative_number(
        direct.get("chitilde_saturation_max_complete_raw_delta"),
        f"{F08D_KEY}.direct_component_validation.chitilde_saturation_max_complete_raw_delta",
    )
    saturation_ceiling = _number(
        direct.get("chitilde_saturation_frozen_ceiling"),
        f"{F08D_KEY}.direct_component_validation.chitilde_saturation_frozen_ceiling",
    )
    if saturation_delta != 1.326535849585929e-08 or saturation_ceiling != 1.5e-08:
        raise EvidenceError(f"{F08D_KEY} chitilde saturation anchors drifted")
    if saturation_delta > saturation_ceiling:
        raise EvidenceError(f"{F08D_KEY} chitilde saturation ceiling was exceeded")

    table = _obj(f08d, "private_table", F08D_KEY)
    _exact(
        table,
        {
            "execution_status": "VALIDATED",
            "authority": (
                "Rust direct four-subterm evaluator; no on-disk cache and no "
                "deterministic C-knot consumption"
            ),
            "point_count": 57,
            "temperature_spacing": "logarithmic",
            "floor_K": 158489319.2461111,
            "maximum_MeV": 10.0,
            "direct_order": 64,
            "n_to_p_interpolation": (
                "four-point local cubic in signed value versus linear T"
            ),
            "p_to_n_interpolation": (
                "four-point local cubic in log(value) versus log(T), after "
                "fail-closed positivity verification of every profile knot"
            ),
            "below_floor_required_exact_zero": True,
            "profile_mismatch_required_failure": True,
            "above_10_MeV_required_failure": True,
        },
        f"{F08D_KEY}.private_table",
    )
    midpoint = _obj(table, "all_interval_midpoint_check", f"{F08D_KEY}.private_table")
    _exact(
        midpoint,
        {
            "interval_count": 56,
            "directions_per_interval": 2,
            "reference_order": 128,
            "ceiling": "max(1e-9,0.01*|direct|) raw phase-space units",
            "execution_status": "VALIDATED",
        },
        f"{F08D_KEY}.private_table.all_interval_midpoint_check",
    )
    midpoint_ratio = _nonnegative_number(
        midpoint.get("maximum_difference_over_ceiling"),
        f"{F08D_KEY}.private_table.all_interval_midpoint_check.maximum_difference_over_ceiling",
    )
    if midpoint_ratio != 0.8787904334047045 or midpoint_ratio > 1.0:
        raise EvidenceError(f"{F08D_KEY} table midpoint contract exceeds one ceiling")
    if midpoint["interval_count"] != table["point_count"] - 1:
        raise EvidenceError(f"{F08D_KEY} table interval count is inconsistent")

    standalone = _obj(f08d, "standalone_temperature_variable_freezeout", F08D_KEY)
    _exact(
        standalone,
        {
            "execution_status": "VALIDATED",
            "strict_external_parity_status": "BLOCKED",
            "external_comparator": (
                "Python global-quadratic interpolation over deterministic C VEGAS knots"
            ),
        },
        f"{F08D_KEY}.standalone_temperature_variable_freezeout",
    )
    external_freezeout = _obj(
        standalone, "external", f"{F08D_KEY}.standalone_temperature_variable_freezeout"
    )
    external_states = {
        endpoint: _state(
            _obj(external_freezeout, endpoint, f"{F08D_KEY}.standalone.external"),
            f"{F08D_KEY}.standalone.external.{endpoint}",
        )
        for endpoint in ("activation", "final")
    }
    _exact(
        external_states["activation"],
        {"N": 2.4593217700143666, "Xn": 0.24020400609043143},
        f"{F08D_KEY}.standalone.external.activation",
    )
    _exact(
        external_states["final"],
        {"N": 5.63289932832902, "Xn": 0.08927883084016204},
        f"{F08D_KEY}.standalone.external.final",
    )
    strict_acceptance = _obj(
        standalone,
        "frozen_strict_acceptance",
        f"{F08D_KEY}.standalone_temperature_variable_freezeout",
    )
    _exact(
        strict_acceptance,
        {"max_absolute_N_difference": 3e-07, "max_absolute_Xn_difference": 8e-08},
        f"{F08D_KEY}.standalone.frozen_strict_acceptance",
    )
    rust_freezeout = _obj(
        standalone, "rust", f"{F08D_KEY}.standalone_temperature_variable_freezeout"
    )
    expected_freezeout = {
        "bdf": {
            "activation_N": 2.459321829553052,
            "activation_Xn": 0.24020238686934622,
            "final_N": 5.632899444145241,
            "final_Xn": 0.0892782265389786,
        },
        "rodas5p": {
            "activation_N": 2.459321912340535,
            "activation_Xn": 0.24020238770812863,
            "final_N": 5.632899526728057,
            "final_Xn": 0.08927822674484534,
        },
    }
    strict_differences = _obj(
        standalone,
        "strict_Xn_differences",
        f"{F08D_KEY}.standalone_temperature_variable_freezeout",
    )
    expected_difference_keys = {
        "bdf_activation",
        "bdf_final",
        "rodas5p_activation",
        "rodas5p_final",
    }
    if set(strict_differences) != expected_difference_keys:
        raise EvidenceError(f"{F08D_KEY} standalone Xn difference keys drifted")
    xn_budget_violations = 0
    for solver, expected in expected_freezeout.items():
        record = _obj(rust_freezeout, solver, f"{F08D_KEY}.standalone.rust")
        _exact(record, expected, f"{F08D_KEY}.standalone.rust.{solver}")
        for endpoint in ("activation", "final"):
            n_difference = record[f"{endpoint}_N"] - external_states[endpoint]["N"]
            if abs(n_difference) > strict_acceptance["max_absolute_N_difference"]:
                raise EvidenceError(f"{F08D_KEY} standalone {solver} {endpoint} N exceeds budget")
            xn_difference = record[f"{endpoint}_Xn"] - external_states[endpoint]["Xn"]
            key = f"{solver}_{endpoint}"
            serialized = _finite_number(
                strict_differences.get(key),
                f"{F08D_KEY}.standalone.strict_Xn_differences.{key}",
            )
            if serialized != xn_difference:
                raise EvidenceError(f"{F08D_KEY} standalone {key} Xn delta is inconsistent")
            if abs(xn_difference) >= strict_acceptance["max_absolute_Xn_difference"]:
                xn_budget_violations += 1
    if xn_budget_violations == 0:
        raise EvidenceError(
            f"{F08D_KEY} standalone must retain at least one explicit Xn budget violation"
        )
    blocker = _nonempty_string(
        standalone.get("blocker"), f"{F08D_KEY}.standalone_temperature_variable_freezeout.blocker"
    ).lower()
    for required in ("do not meet", "8e-8", "no precision promotion"):
        if required not in blocker:
            raise EvidenceError(f"{F08D_KEY} standalone blocker must retain {required!r}")

    endpoint = _obj(f08d, "matched_endpoint", F08D_KEY)
    _exact(
        endpoint,
        {
            "execution_status": "VALIDATED",
            "claim_scope": (
                "Conditional 12-reaction endpoint consistency and broad external "
                "envelope only"
            ),
        },
        f"{F08D_KEY}.matched_endpoint",
    )
    external_endpoint = _obj(
        endpoint, "external_deterministic_c_table", f"{F08D_KEY}.matched_endpoint"
    )
    external_outputs = {
        implementation: _outputs(
            _obj(external_endpoint, implementation, f"{F08D_KEY}.matched_endpoint.external"),
            f"{F08D_KEY}.matched_endpoint.external.{implementation}",
        )
        for implementation in ("python", "c")
    }
    expected_external = {
        "python": {
            "Yp": 0.24683372920057065,
            "DH": 2.4564842279090492e-05,
            "He3H": 1.043460922609712e-05,
            "Li7H": 5.48857613026354e-10,
        },
        "c": {
            "Yp": 0.24683473135514053,
            "DH": 2.456498904962319e-05,
            "He3H": 1.0434625862058256e-05,
            "Li7H": 5.488570894042898e-10,
        },
    }
    for implementation in ("python", "c"):
        _exact(
            external_outputs[implementation],
            expected_external[implementation],
            f"{F08D_KEY}.matched_endpoint.external.{implementation}",
        )
    python_minus_c_yp = _finite_number(
        external_endpoint.get("python_minus_c_Yp"),
        f"{F08D_KEY}.matched_endpoint.external.python_minus_c_Yp",
    )
    if python_minus_c_yp != external_outputs["python"]["Yp"] - external_outputs["c"]["Yp"]:
        raise EvidenceError(f"{F08D_KEY} external Python/C Yp delta is inconsistent")

    endpoint_acceptance = _obj(
        endpoint, "frozen_acceptance", f"{F08D_KEY}.matched_endpoint"
    )
    budgets = _obj(
        endpoint_acceptance,
        "direct_endpoint_budgets",
        f"{F08D_KEY}.matched_endpoint.frozen_acceptance",
    )
    _exact(
        budgets,
        {"Yp_absolute": 2e-05, "DH_relative": 0.001, "He3H_relative": 0.002, "Li7H_relative": 0.004},
        f"{F08D_KEY}.matched_endpoint.frozen_acceptance.direct_endpoint_budgets",
    )
    _exact(
        endpoint_acceptance,
        {"isolated_f08d_minus_f08c_sign": "negative for Yp,DH,He3H,Li7H"},
        f"{F08D_KEY}.matched_endpoint.frozen_acceptance",
    )
    envelopes = _obj(
        endpoint_acceptance,
        "isolated_absolute_magnitude_envelopes",
        f"{F08D_KEY}.matched_endpoint.frozen_acceptance",
    )
    expected_envelopes = {
        "Yp": (8e-07, 4.5e-06),
        "DH": (5e-11, 2.8e-10),
        "He3H": (1e-11, 4.5e-11),
        "Li7H": (3e-16, 6e-15),
    }
    parsed_envelopes: dict[str, tuple[float, float]] = {}
    if set(envelopes) != set(expected_envelopes):
        raise EvidenceError(f"{F08D_KEY} isolated endpoint envelope keys drifted")
    for observable, expected in expected_envelopes.items():
        value = envelopes.get(observable)
        if not isinstance(value, list) or len(value) != 2:
            raise EvidenceError(f"{F08D_KEY} {observable} envelope must have two bounds")
        bounds = (
            _number(value[0], f"{F08D_KEY}.{observable}.envelope[0]"),
            _number(value[1], f"{F08D_KEY}.{observable}.envelope[1]"),
        )
        if bounds != expected or bounds[0] >= bounds[1]:
            raise EvidenceError(f"{F08D_KEY} {observable} envelope drifted")
        parsed_envelopes[observable] = bounds

    rust_endpoint = _obj(endpoint, "rust", f"{F08D_KEY}.matched_endpoint")
    expected_rust_endpoints = {
        "bdf": {
            "Xn_handoff": 0.24020238686934622,
            "Yp": 0.24683356615720803,
            "DH": 2.4564374518213406e-05,
            "He3H": 1.0434556632413961e-05,
            "Li7H": 5.488674122424027e-10,
        },
        "rodas5p": {
            "Xn_handoff": 0.24020238770812863,
            "Yp": 0.24683358495050578,
            "DH": 2.4564376830218522e-05,
            "He3H": 1.043455690926778e-05,
            "Li7H": 5.488673896701071e-10,
        },
    }
    rust_outputs: dict[str, dict[str, float]] = {}
    for solver, expected in expected_rust_endpoints.items():
        record = _obj(rust_endpoint, solver, f"{F08D_KEY}.matched_endpoint.rust")
        _exact(record, expected, f"{F08D_KEY}.matched_endpoint.rust.{solver}")
        rust_outputs[solver] = _outputs(record, f"{F08D_KEY}.matched_endpoint.rust.{solver}")
        isolated = _obj(record, "f08d_minus_f08c", f"{F08D_KEY}.matched_endpoint.rust.{solver}")
        for observable in OBSERVABLES:
            actual = _finite_number(
                isolated.get(observable),
                f"{F08D_KEY}.matched_endpoint.rust.{solver}.f08d_minus_f08c.{observable}",
            )
            expected_delta = rust_outputs[solver][observable] - f08c_rust_outputs[solver][observable]
            if actual != expected_delta:
                raise EvidenceError(f"{F08D_KEY} {solver} {observable} isolated delta is inconsistent")
            lower, upper = parsed_envelopes[observable]
            if actual >= 0.0 or not lower <= abs(actual) <= upper:
                raise EvidenceError(f"{F08D_KEY} {solver} {observable} isolated sign/envelope failed")
        for implementation, reference in external_outputs.items():
            if abs(rust_outputs[solver]["Yp"] - reference["Yp"]) > budgets["Yp_absolute"]:
                raise EvidenceError(f"{F08D_KEY} {solver}/{implementation} Yp budget failed")
            for observable in ("DH", "He3H", "Li7H"):
                relative = abs((rust_outputs[solver][observable] - reference[observable]) / reference[observable])
                if relative > budgets[f"{observable}_relative"]:
                    raise EvidenceError(f"{F08D_KEY} {solver}/{implementation} {observable} budget failed")

    repeat = _obj(rust_endpoint, "repeat_determinism", f"{F08D_KEY}.matched_endpoint.rust")
    _exact(repeat, {"bdf_exact_float_equality": True}, f"{F08D_KEY}.matched_endpoint.rust.repeat_determinism")
    wall = _obj(rust_endpoint, "measured_wall_seconds", f"{F08D_KEY}.matched_endpoint.rust")
    _exact(
        wall,
        {"cold_bdf": 11.420426, "warm_repeat_bdf": 3.637269, "rodas5p_after_table_build": 10.33968},
        f"{F08D_KEY}.matched_endpoint.rust.measured_wall_seconds",
    )
    for key, value in wall.items():
        _number(value, f"{F08D_KEY}.matched_endpoint.rust.measured_wall_seconds.{key}")
    if wall["warm_repeat_bdf"] >= wall["cold_bdf"]:
        raise EvidenceError(f"{F08D_KEY} warm repeat must remain below cold-table wall")

    provenance = _obj(f08d, "provenance", F08D_KEY)
    _exact(
        provenance,
        {
            "baseline_head": "0b2339c",
            "branch": "bd612-remediation",
            "source_state": "uncommitted working tree",
            "rust_source_paths": [
                "native/rabbit_cpu/src/thermal_weak.rs",
                "native/rabbit_cpu/src/born_weak.rs",
                "native/rabbit_cpu/src/born_freezeout.rs",
                "native/rabbit_cpu/src/minimal_bbn.rs",
            ],
            "external_result_path": (
                "/tmp/rabbit_f08d_external_20260715/f08d_complete_ccrth_result.json"
            ),
            "external_validation_checks": "25/25 PASSED_WITH_LIMITS",
            "external_mutation_probes": "5/5 rejected",
            "exact_token_counter": (
                "UNAVAILABLE: harness does not expose an exact per-task counter"
            ),
        },
        f"{F08D_KEY}.provenance",
    )
    for key in (
        "external_result_sha256",
        "external_raw_sha256",
        "external_validation_sha256",
        "external_manifest_sha256",
        "external_validator_sha256",
        "external_c_table_sha256",
    ):
        _sha256(provenance.get(key), f"{F08D_KEY}.provenance.{key}")
    checks_match = re.fullmatch(
        r"([1-9][0-9]*)/([1-9][0-9]*) PASSED_WITH_LIMITS",
        provenance["external_validation_checks"],
    )
    mutations_match = re.fullmatch(
        r"([1-9][0-9]*)/([1-9][0-9]*) rejected",
        provenance["external_mutation_probes"],
    )
    if checks_match is None or checks_match.group(1) != checks_match.group(2):
        raise EvidenceError(f"{F08D_KEY} external validation checks must all pass")
    if mutations_match is None or mutations_match.group(1) != mutations_match.group(2):
        raise EvidenceError(f"{F08D_KEY} external mutation probes must all reject")

    limitations = f08d.get("claim_limitations")
    if not isinstance(limitations, list) or len(limitations) < 6 or not all(
        isinstance(item, str) and item.strip() for item in limitations
    ):
        raise EvidenceError(f"{F08D_KEY}.claim_limitations must contain at least six non-empty strings")
    limitation_text = " ".join(limitations).lower()
    for required in (
        "strict standalone",
        "table/interpolation authority remains unresolved",
        "diagnostic only",
        "tiny p_to_n",
        "low-temperature clamp",
        "incomplete neutrino decoupling",
        "collision transport",
        "anisotropy",
        "qke",
        "precision-standard",
        "public-production",
    ):
        if required not in limitation_text:
            raise EvidenceError(f"{F08D_KEY}.claim_limitations must retain {required!r}")

    return {
        "f08d": f08d,
        "f08d_external": external_outputs,
        "f08d_rust": rust_outputs,
    }


def _validate_f08n(
    fixture: dict[str, Any],
    f08d_rust_outputs: dict[str, dict[str, float]],
) -> dict[str, Any]:
    f08n = _obj(fixture, F08N_KEY, "fixture")
    _closed_exact(
        f08n,
        {
            "schema_version": "f08n_selected_ac2024_31_rust_endpoint_v1",
            "implementation_status": "IMPLEMENTED",
            "claim_status": "VALIDATED",
            "promotion_status": "BLOCKED",
            "authority_state": "CONDITIONAL",
            "captured_date": "2026-07-15",
            "scope": (
                "Rust AOT selected 31-reaction AC2024 network over the F08D "
                "complete-thermal, leading-QED, physical finite-mass/weak-magnetism, "
                "instantaneous-decoupling FLRW endpoint, nested against the unchanged "
                "first 12 reactions."
            ),
        },
        F08N_KEY,
        allowed_extra=(
            "network_contract",
            "external",
            "rust",
            "provenance",
            "claim_limitations",
        ),
    )

    network = _obj(f08n, "network_contract", F08N_KEY)
    _closed_exact(
        network,
        {
            "selected_table_sha256": (
                "8dacb12bece202a798bae67f2ca89d1fad62f9f02bad38da0f59c19147960b85"
            ),
            "reaction_count": 31,
            "first12_prefix_preserved": True,
            "selected_is_named_stock_primat_topology": False,
            "reverse_coefficients_consumed_verbatim": True,
            "raw_q_is_not_reverse_rate_authority": True,
            "source_species_order": [
                "n", "p", "d", "t", "He3", "a", "Li7", "Be7", "Li6"
            ],
            "rust_species_order": [
                "n", "p", "D", "T", "He3", "He4", "Li6", "Li7", "Be7"
            ],
            "primat_reaction_order": list(F08N_PRIMAT_REACTION_ORDER),
            "reaction_order": list(F08N_REACTION_ORDER),
        },
        f"{F08N_KEY}.network_contract",
    )
    _sha256(
        network.get("selected_table_sha256"),
        f"{F08N_KEY}.network_contract.selected_table_sha256",
    )
    try:
        selected_table_sha256 = hashlib.sha256(F08N_NETWORK_TABLE.read_bytes()).hexdigest()
    except OSError as exc:
        raise EvidenceError(
            f"cannot hash F-08N selected reaction table {F08N_NETWORK_TABLE}: {exc}"
        ) from exc
    if selected_table_sha256 != network["selected_table_sha256"]:
        raise EvidenceError(
            f"{F08N_KEY}.network_contract.selected_table_sha256 does not match "
            "the repo-local selected reaction table"
        )
    if network["reaction_count"] != len(network["reaction_order"]):
        raise EvidenceError(f"{F08N_KEY} reaction count/order length mismatch")
    if network["reaction_count"] != len(network["primat_reaction_order"]):
        raise EvidenceError(f"{F08N_KEY} PRIMAT reaction count/order length mismatch")

    external = _obj(f08n, "external", F08N_KEY)
    _closed_exact(
        external,
        {
            "execution_status": "VALIDATED",
            "primat_version": "0.3.2",
            "primat_commit": "21ff8f39fa18e3937e9fdf386cfa982361bfdfce",
            "exact_reverse_authority": (
                "PRIMAT Python with explicit private injection of the selected JSON "
                "reverse coefficients"
            ),
            "exact_c_counterpart_available": False,
            "baryon_sum_abs_residual": 1.6426859872353816e-12,
        },
        f"{F08N_KEY}.external",
        allowed_extra=(
            "exact_12",
            "exact_selected_31",
            "exact_selected_31_minus_12",
            "all31_from_tweak_diagnostic",
            "repeat_determinism",
            "frozen_acceptance",
        ),
    )
    _commit(
        {"commit": external.get("primat_commit")},
        f"{F08N_KEY}.external",
    )

    expected_external = {
        "exact_12": {
            "Yp": 0.2468339757668937,
            "DH": 2.4564951285432863e-05,
            "He3H": 1.0434628229940604e-05,
            "Li7H": 5.488544912800621e-10,
            "Li6H": 0.0,
        },
        "exact_selected_31": {
            "Yp": 0.24683389951424803,
            "DH": 2.456571050928129e-05,
            "He3H": 1.0433783687263961e-05,
            "Li7H": 5.437134223492424e-10,
            "Li6H": 7.819032381929294e-15,
        },
        "all31_from_tweak_diagnostic": {
            "Yp": 0.24683390421324497,
            "DH": 2.4565583781652233e-05,
            "He3H": 1.0433764588510948e-05,
            "Li7H": 5.43716567362588e-10,
            "Li6H": 7.818990090809015e-15,
        },
    }
    external_outputs: dict[str, dict[str, float]] = {}
    for name, expected in expected_external.items():
        record = _obj(external, name, f"{F08N_KEY}.external")
        _closed_exact(record, expected, f"{F08N_KEY}.external.{name}")
        external_outputs[name] = _f08n_outputs(
            record, f"{F08N_KEY}.external.{name}"
        )

    nested = _obj(external, "exact_selected_31_minus_12", f"{F08N_KEY}.external")
    expected_nested = {
        "Yp": -7.62526456699053e-08,
        "DH": 7.592238484255243e-10,
        "He3H": -8.445426766421699e-10,
        "Li7H": -5.141068930819691e-12,
        "Li6H": 7.819032381929294e-15,
    }
    _closed_exact(
        nested,
        expected_nested,
        f"{F08N_KEY}.external.exact_selected_31_minus_12",
    )
    for observable in F08N_OBSERVABLES:
        actual = _finite_number(
            nested.get(observable),
            f"{F08N_KEY}.external.exact_selected_31_minus_12.{observable}",
        )
        expected = (
            external_outputs["exact_selected_31"][observable]
            - external_outputs["exact_12"][observable]
        )
        if actual != expected:
            raise EvidenceError(
                f"{F08N_KEY} external selected31-minus-12 {observable} is inconsistent"
            )

    repeat = _obj(external, "repeat_determinism", f"{F08N_KEY}.external")
    _closed_exact(
        repeat,
        {"exact_selected_31_float_equality": True},
        f"{F08N_KEY}.external.repeat_determinism",
    )

    acceptance = _obj(external, "frozen_acceptance", f"{F08N_KEY}.external")
    _closed_exact(
        acceptance,
        {
            "frozen_before_any_rust_f08n_value_was_read": True,
            "primat_default_staging": {"MT_reactions": 17, "LT_reactions": 31},
            "rust_all31_from_existing_f06_handoff_is_diagnostic": True,
        },
        f"{F08N_KEY}.external.frozen_acceptance",
        allowed_extra=(
            "direct_selected31_endpoint_budgets",
            "nested_31_minus_12_contract",
        ),
    )
    direct_budgets = _obj(
        acceptance,
        "direct_selected31_endpoint_budgets",
        f"{F08N_KEY}.external.frozen_acceptance",
    )
    _closed_exact(
        direct_budgets,
        {
            "Yp_absolute": 2e-05,
            "DH_relative": 0.001,
            "He3H_relative": 0.002,
            "Li7H_relative": 0.004,
            "Li6H_absolute_range": [2e-15, 2e-14],
            "baryon_sum_abs_residual": 5e-09,
        },
        f"{F08N_KEY}.external.frozen_acceptance.direct_selected31_endpoint_budgets",
    )
    nested_contract = _obj(
        acceptance,
        "nested_31_minus_12_contract",
        f"{F08N_KEY}.external.frozen_acceptance",
    )
    _closed_exact(
        nested_contract,
        {
            "Yp_absolute_effect_max": 2e-06,
            "Yp_sign_required": None,
            "DH_absolute_effect_range": [2e-10, 2e-09],
            "He3H_absolute_effect_range": [-1.6e-09, -3e-10],
            "Li7H_relative_effect_range": [-0.02, -0.004],
            "Li6H_absolute_range": [2e-15, 2e-14],
        },
        f"{F08N_KEY}.external.frozen_acceptance.nested_31_minus_12_contract",
    )
    if external["baryon_sum_abs_residual"] > direct_budgets["baryon_sum_abs_residual"]:
        raise EvidenceError(f"{F08N_KEY} external baryon residual exceeds frozen budget")
    if abs(nested["Yp"]) > nested_contract["Yp_absolute_effect_max"]:
        raise EvidenceError(f"{F08N_KEY} external nested Yp exceeds frozen effect budget")
    if not nested_contract["DH_absolute_effect_range"][0] <= nested["DH"] <= nested_contract["DH_absolute_effect_range"][1]:
        raise EvidenceError(f"{F08N_KEY} external nested D/H violates its frozen range")
    if not nested_contract["He3H_absolute_effect_range"][0] <= nested["He3H"] <= nested_contract["He3H_absolute_effect_range"][1]:
        raise EvidenceError(f"{F08N_KEY} external nested He3/H violates its frozen range")
    external_li7_relative = nested["Li7H"] / external_outputs["exact_12"]["Li7H"]
    if not nested_contract["Li7H_relative_effect_range"][0] <= external_li7_relative <= nested_contract["Li7H_relative_effect_range"][1]:
        raise EvidenceError(f"{F08N_KEY} external nested Li7/H violates its frozen range")
    if not nested_contract["Li6H_absolute_range"][0] <= nested["Li6H"] <= nested_contract["Li6H_absolute_range"][1]:
        raise EvidenceError(f"{F08N_KEY} external nested Li6/H violates its frozen range")

    diagnostic_delta = {
        observable: (
            external_outputs["all31_from_tweak_diagnostic"][observable]
            - external_outputs["exact_12"][observable]
        )
        for observable in F08N_OBSERVABLES
    }
    if abs(diagnostic_delta["Yp"]) > nested_contract["Yp_absolute_effect_max"]:
        raise EvidenceError(f"{F08N_KEY} all31-from-Tweak Yp budget failed")
    if not nested_contract["DH_absolute_effect_range"][0] <= diagnostic_delta["DH"] <= nested_contract["DH_absolute_effect_range"][1]:
        raise EvidenceError(f"{F08N_KEY} all31-from-Tweak D/H budget failed")
    if not nested_contract["He3H_absolute_effect_range"][0] <= diagnostic_delta["He3H"] <= nested_contract["He3H_absolute_effect_range"][1]:
        raise EvidenceError(f"{F08N_KEY} all31-from-Tweak He3/H budget failed")
    diagnostic_li7_relative = diagnostic_delta["Li7H"] / external_outputs["exact_12"]["Li7H"]
    if not nested_contract["Li7H_relative_effect_range"][0] <= diagnostic_li7_relative <= nested_contract["Li7H_relative_effect_range"][1]:
        raise EvidenceError(f"{F08N_KEY} all31-from-Tweak Li7/H budget failed")
    if not nested_contract["Li6H_absolute_range"][0] <= diagnostic_delta["Li6H"] <= nested_contract["Li6H_absolute_range"][1]:
        raise EvidenceError(f"{F08N_KEY} all31-from-Tweak Li6/H budget failed")

    rust = _obj(f08n, "rust", F08N_KEY)
    _closed_exact(
        rust,
        {
            "execution_status": "VALIDATED",
            "activation_temperature_mev": 0.86164715286738,
            "activation_contract": (
                "All 31 selected reactions are active from the existing F06 "
                "0.86164715286738 MeV Saha handoff; this is a staging diagnostic, "
                "not stock PRIMAT staging."
            ),
        },
        f"{F08N_KEY}.rust",
        allowed_extra=(
            "bdf",
            "rodas5p",
            "refined_bdf_rtol_3e_11",
            "repeat_determinism",
            "raw_red_attempts",
            "measured_wall_seconds",
        ),
    )
    expected_rust = {
        "bdf": {
            "rtol": 1e-10,
            "Xn_handoff": 0.24020238686934622,
            "Yp": 0.24683358124527852,
            "DH": 2.456516625200859e-05,
            "He3H": 1.0433716330832902e-05,
            "Li7H": 5.437257176155468e-10,
            "Li6H": 7.818856581771896e-15,
            "selected31_minus_backbone12": {
                "Yp": 1.508807048744565e-08,
                "DH": 7.917337951831313e-10,
                "He3H": -8.403015810591288e-10,
                "Li7H_relative": -0.009367826386065481,
            },
        },
        "rodas5p": {
            "rtol": 1e-10,
            "Xn_handoff": 0.24020238770812863,
            "Yp": 0.24683358304487701,
            "DH": 2.4565166617141467e-05,
            "He3H": 1.0433716373987523e-05,
            "Li7H": 5.437257132026343e-10,
            "Li6H": 7.818856790219118e-15,
            "selected31_minus_backbone12": {
                "Yp": -1.9056287658969495e-09,
                "DH": 7.897869229441066e-10,
                "He3H": -8.405352802563434e-10,
                "Li7H_relative": -0.009367793686127168,
            },
        },
    }
    rust_outputs: dict[str, dict[str, float]] = {}
    for solver, expected in expected_rust.items():
        record = _obj(rust, solver, f"{F08N_KEY}.rust")
        _closed_exact(record, expected, f"{F08N_KEY}.rust.{solver}")
        rust_outputs[solver] = _f08n_outputs(record, f"{F08N_KEY}.rust.{solver}")
        serialized_delta = _obj(
            record, "selected31_minus_backbone12", f"{F08N_KEY}.rust.{solver}"
        )
        for observable in ("Yp", "DH", "He3H"):
            expected_delta = (
                rust_outputs[solver][observable]
                - f08d_rust_outputs[solver][observable]
            )
            if serialized_delta[observable] != expected_delta:
                raise EvidenceError(
                    f"{F08N_KEY} {solver} nested {observable} is inconsistent"
                )
        li7_relative = (
            rust_outputs[solver]["Li7H"] - f08d_rust_outputs[solver]["Li7H"]
        ) / f08d_rust_outputs[solver]["Li7H"]
        # The serialized relative effect was formed from the unrounded internal
        # endpoints; the endpoint fields above retain their printed f64 values.
        if not math.isclose(
            serialized_delta["Li7H_relative"],
            li7_relative,
            rel_tol=0.0,
            abs_tol=5e-17,
        ):
            raise EvidenceError(f"{F08N_KEY} {solver} nested Li7/H is inconsistent")

        if abs(serialized_delta["Yp"]) > nested_contract["Yp_absolute_effect_max"]:
            raise EvidenceError(f"{F08N_KEY} {solver} nested Yp budget failed")
        if not nested_contract["DH_absolute_effect_range"][0] <= serialized_delta["DH"] <= nested_contract["DH_absolute_effect_range"][1]:
            raise EvidenceError(f"{F08N_KEY} {solver} nested D/H budget failed")
        if not nested_contract["He3H_absolute_effect_range"][0] <= serialized_delta["He3H"] <= nested_contract["He3H_absolute_effect_range"][1]:
            raise EvidenceError(f"{F08N_KEY} {solver} nested He3/H budget failed")
        if not nested_contract["Li7H_relative_effect_range"][0] <= serialized_delta["Li7H_relative"] <= nested_contract["Li7H_relative_effect_range"][1]:
            raise EvidenceError(f"{F08N_KEY} {solver} nested Li7/H budget failed")
        if not nested_contract["Li6H_absolute_range"][0] <= rust_outputs[solver]["Li6H"] <= nested_contract["Li6H_absolute_range"][1]:
            raise EvidenceError(f"{F08N_KEY} {solver} nested Li6/H budget failed")

    refined = _obj(rust, "refined_bdf_rtol_3e_11", f"{F08N_KEY}.rust")
    _closed_exact(
        refined,
        {
            "rtol": 3e-11,
            "Yp": 0.24683358128054264,
            "DH": 2.4565166601201154e-05,
            "He3H": 1.0433716372463938e-05,
            "Li7H": 5.437257096810452e-10,
            "Li6H": 7.818856690501073e-15,
        },
        f"{F08N_KEY}.rust.refined_bdf_rtol_3e_11",
    )
    refined_outputs = _f08n_outputs(
        refined, f"{F08N_KEY}.rust.refined_bdf_rtol_3e_11"
    )
    if refined["rtol"] >= expected_rust["bdf"]["rtol"]:
        raise EvidenceError(f"{F08N_KEY} refined BDF tolerance must be tighter")

    for solver, outputs in (*rust_outputs.items(), ("refined_bdf_rtol_3e_11", refined_outputs)):
        reference = external_outputs["exact_selected_31"]
        if abs(outputs["Yp"] - reference["Yp"]) > direct_budgets["Yp_absolute"]:
            raise EvidenceError(f"{F08N_KEY} {solver} direct Yp budget failed")
        for observable in ("DH", "He3H", "Li7H"):
            relative = abs((outputs[observable] - reference[observable]) / reference[observable])
            if relative > direct_budgets[f"{observable}_relative"]:
                raise EvidenceError(f"{F08N_KEY} {solver} direct {observable} budget failed")
        if not direct_budgets["Li6H_absolute_range"][0] <= outputs["Li6H"] <= direct_budgets["Li6H_absolute_range"][1]:
            raise EvidenceError(f"{F08N_KEY} {solver} direct Li6/H budget failed")

    rust_repeat = _obj(rust, "repeat_determinism", f"{F08N_KEY}.rust")
    _closed_exact(
        rust_repeat,
        {"bdf_exact_float_equality": True},
        f"{F08N_KEY}.rust.repeat_determinism",
    )
    raw_red = rust.get("raw_red_attempts")
    expected_raw_red = [
        {
            "rtol": 1e-09,
            "solver": "bdf",
            "baryon_sum_abs_residual": 2.3085050204763036e-08,
            "frozen_ceiling": 5e-09,
            "status": "RED",
        },
        {
            "rtol": 3e-10,
            "solver": "bdf tolerance-ladder candidate",
            "baryon_sum_abs_residual": 7.518909717063593e-09,
            "frozen_ceiling": 5e-09,
            "status": "RED",
        },
    ]
    if raw_red != expected_raw_red:
        raise EvidenceError(f"{F08N_KEY}.rust.raw_red_attempts drifted")
    for index, record in enumerate(raw_red):
        if record["baryon_sum_abs_residual"] <= record["frozen_ceiling"]:
            raise EvidenceError(
                f"{F08N_KEY}.rust.raw_red_attempts[{index}] no longer records RED"
            )

    wall = _obj(rust, "measured_wall_seconds", f"{F08N_KEY}.rust")
    _closed_exact(
        wall,
        {
            "cold_bdf": 8.779494,
            "warm_repeat_bdf": 4.617595,
            "rodas5p": 18.080271,
            "refined_bdf_rtol_3e_11": 4.973685,
        },
        f"{F08N_KEY}.rust.measured_wall_seconds",
    )
    for key, value in wall.items():
        _number(value, f"{F08N_KEY}.rust.measured_wall_seconds.{key}")
    if wall["warm_repeat_bdf"] >= wall["cold_bdf"]:
        raise EvidenceError(f"{F08N_KEY} warm BDF repeat must remain below cold wall")

    provenance = _obj(f08n, "provenance", F08N_KEY)
    _closed_exact(
        provenance,
        {
            "baseline_head": "0b2339c",
            "branch": "bd612-remediation",
            "source_state": "uncommitted working tree",
            "external_result_path": (
                "/tmp/rabbit_f08n_external_20260715/f08n_selected31_result.json"
            ),
            "external_result_sha256": (
                "c8b54a40fd6c7d4bc15d5bce9a5c0324d96ec0cc3a6e06e62e62dc668594d56a"
            ),
            "external_raw_sha256": (
                "bc51a3903a449a6ec20db42fa5924d380f3a643f8cba4fa9be8eeaa9735a2d22"
            ),
            "external_validation_sha256": (
                "205a399470b76f66f3dcf5cc052cd1f1dd23dfe4fa60cf7e910dadd556faeb45"
            ),
            "external_manifest_sha256": (
                "3ed8754c507d486336a948cc388277d44ecc7cb0eeb59b34e28136ffcc6d74f6"
            ),
            "external_validator_sha256": (
                "a72cf8da629fe8119cc3a828ff9e4b1a4a2d4ce8eb847630c281f6a957c51ba7"
            ),
            "external_mutation_probes_sha256": (
                "e66ac00e00cac2ca6d8c1dc37e3d944b256d60bbf07a6b3aae265e59c790e15d"
            ),
            "external_validation_checks": "46/46 passed",
            "external_mutation_probes": "5/5 rejected",
            "exact_token_counter": (
                "UNAVAILABLE: harness does not expose an exact per-task counter"
            ),
        },
        f"{F08N_KEY}.provenance",
    )
    for key in (
        "external_result_sha256",
        "external_raw_sha256",
        "external_validation_sha256",
        "external_manifest_sha256",
        "external_validator_sha256",
        "external_mutation_probes_sha256",
    ):
        _sha256(provenance.get(key), f"{F08N_KEY}.provenance.{key}")
    checks_match = re.fullmatch(
        r"([1-9][0-9]*)/([1-9][0-9]*) passed",
        provenance["external_validation_checks"],
    )
    mutations_match = re.fullmatch(
        r"([1-9][0-9]*)/([1-9][0-9]*) rejected",
        provenance["external_mutation_probes"],
    )
    if checks_match is None or checks_match.group(1) != checks_match.group(2):
        raise EvidenceError(f"{F08N_KEY} external checks must all pass")
    if mutations_match is None or mutations_match.group(1) != mutations_match.group(2):
        raise EvidenceError(f"{F08N_KEY} external mutations must all reject")

    expected_limitations = [
        (
            "The selected 31 reactions are Rabbit's accepted AC2024 subset, not a "
            "named stock PRIMAT network and not the broader PRIMAT large network."
        ),
        (
            "The external exact-reverse authority uses a private PRIMAT Python injection "
            "because the public custom API recomputes reverse coefficients; there is no "
            "exact C counterpart."
        ),
        (
            "Nuclear rate data are shared with Rust, so this validates solver, topology, "
            "staging consumption, and invariants rather than the underlying measurements."
        ),
        (
            "Rust activates all 31 reactions at the existing F06 handoff while PRIMAT "
            "stages 17 then 31; agreement is conditional on the frozen nested budgets."
        ),
        (
            "The inherited F08D strict standalone Xn authority remains BLOCKED at an "
            "8e-8 external ceiling versus an O(1e-6) discrepancy."
        ),
        (
            "No incomplete decoupling, neutrino transport/collisions, anisotropy, QKE, "
            "precision-standard abundance, or public-production claim follows."
        ),
    ]
    if f08n.get("claim_limitations") != expected_limitations:
        raise EvidenceError(f"{F08N_KEY}.claim_limitations drifted")
    limitations_text = " ".join(expected_limitations).lower()
    for required in (
        "not a named stock primat network",
        "private primat python injection",
        "shared with rust",
        "stages 17 then 31",
        "remains blocked",
        "8e-8",
        "incomplete decoupling",
        "neutrino transport/collisions",
        "anisotropy",
        "qke",
        "precision-standard",
        "public-production",
    ):
        if required not in limitations_text:
            raise EvidenceError(f"{F08N_KEY}.claim_limitations must retain {required!r}")

    return {
        "f08n": f08n,
        "f08n_external": external_outputs,
        "f08n_rust": {**rust_outputs, "refined_bdf_rtol_3e_11": refined_outputs},
    }


def _validate(fixture: dict[str, Any]) -> dict[str, Any]:
    entries = fixture.get("entries")
    if not isinstance(entries, list) or not entries:
        raise EvidenceError("fixture.entries must retain the FLRW gold entries")

    f06 = _obj(fixture, F06_KEY, "fixture")
    _exact(
        f06,
        {
            "schema_version": "f06_matched_standard_anchors_v1",
            "implementation_status": "IMPLEMENTED",
            "claim_status": "VALIDATED",
            "captured_date": "2026-07-15",
            "evidence_mode": "stored_results_from_live_runs",
        },
        F06_KEY,
    )
    replay = _obj(f06, "stored_replay_contract", F06_KEY)
    _exact(
        replay,
        {"executes_external_code": False, "may_be_reported_as_a_live_run": False},
        f"{F06_KEY}.stored_replay_contract",
    )
    matched = _obj(f06, "matched_configuration", F06_KEY)
    _exact(
        matched,
        {"geometry": "FLRW", "reaction_count": 12},
        f"{F06_KEY}.matched_configuration",
    )
    switches = _obj(matched, "physics_switches", f"{F06_KEY}.matched_configuration")
    _exact(
        switches,
        {
            "incomplete_decoupling": False,
            "qed": False,
            "spectral": False,
            "weak_corrections": False,
            "nuclear_qed": False,
        },
        f"{F06_KEY}.matched_configuration.physics_switches",
    )

    primat = _obj(f06, "primat", F06_KEY)
    _exact(
        primat,
        {"version": "0.3.2", "execution_status": "VALIDATED"},
        f"{F06_KEY}.primat",
    )
    _commit(primat, f"{F06_KEY}.primat")
    primat_config = _obj(primat, "configuration", f"{F06_KEY}.primat")
    _exact(
        primat_config,
        {
            "eta": matched.get("eta_late"),
            "omega_b_h2": matched.get("omega_b_h2"),
            "neutron_lifetime_seconds": matched.get("neutron_lifetime_seconds"),
            "temperature_start_mev": matched.get("temperature_start_mev"),
            "temperature_end_mev": matched.get("temperature_end_mev"),
            "incomplete_decoupling": False,
            "qed": False,
            "spectral": False,
            "weak_corrections": False,
            "nuclear_qed": False,
            "caches": False,
        },
        f"{F06_KEY}.primat.configuration",
    )
    stock = _obj(primat, "stock_official_small_12", f"{F06_KEY}.primat")
    custom = _obj(
        primat,
        "matched_rabbit_60_node_piecewise_loglinear_custom_table",
        f"{F06_KEY}.primat",
    )
    _exact(
        custom,
        {"is_stock_primat": False},
        f"{F06_KEY}.primat.matched_rabbit_60_node_piecewise_loglinear_custom_table",
    )

    rust = _obj(f06, "rust", F06_KEY)
    provenance = _obj(rust, "provenance", f"{F06_KEY}.rust")
    _exact(
        provenance,
        {
            "baseline_head": "0b2339c",
            "branch": "bd612-remediation",
            "source_path": "native/rabbit_cpu/src/minimal_bbn.rs",
            "source_state": "uncommitted working tree",
            "committed_rust_blob": False,
        },
        f"{F06_KEY}.rust.provenance",
    )
    profile = _obj(rust, "profile", f"{F06_KEY}.rust")
    _exact(
        profile,
        {"relative_tolerance": 1e-9, "weak_corrections": False, "qed": False},
        f"{F06_KEY}.rust.profile",
    )
    bdf = _obj(rust, "bdf_rtol_1e_9", f"{F06_KEY}.rust")
    rodas = _obj(rust, "rodas5p_rtol_1e_9", f"{F06_KEY}.rust")
    _exact(bdf, {"solver": "BDF", "rtol": 1e-9}, f"{F06_KEY}.rust.bdf_rtol_1e_9")
    _exact(
        rodas,
        {"solver": "Rodas5P", "rtol": 1e-9},
        f"{F06_KEY}.rust.rodas5p_rtol_1e_9",
    )
    _number(bdf.get("Xn_handoff"), f"{F06_KEY}.rust.bdf_rtol_1e_9.Xn_handoff")
    _number(rodas.get("Xn_handoff"), f"{F06_KEY}.rust.rodas5p_rtol_1e_9.Xn_handoff")

    linx = _obj(f06, "linx", F06_KEY)
    _exact(
        linx,
        {"version": "0.1.2", "execution_status": "VALIDATED"},
        f"{F06_KEY}.linx",
    )
    _commit(linx, f"{F06_KEY}.linx")
    independence = _obj(linx, "independence_scope", f"{F06_KEY}.linx")
    _exact(
        independence,
        {
            "rhs_and_integrator_independent": True,
            "background_independent": False,
            "nuclear_input_independent": False,
        },
        f"{F06_KEY}.linx.independence_scope",
    )
    ladder = _obj(linx, "tolerance_ladder", f"{F06_KEY}.linx")
    linx_runs: dict[str, dict[str, float]] = {}
    for name, rtol in (("rtol_3e_7", 3e-7), ("rtol_1e_7", 1e-7), ("rtol_3e_8", 3e-8)):
        run = _obj(ladder, name, f"{F06_KEY}.linx.tolerance_ladder")
        _exact(run, {"rtol": rtol}, f"{F06_KEY}.linx.tolerance_ladder.{name}")
        linx_runs[name] = _outputs(run, f"{F06_KEY}.linx.tolerance_ladder.{name}")

    limitations = f06.get("claim_limitations")
    if not isinstance(limitations, list) or not limitations or not all(
        isinstance(item, str) and item for item in limitations
    ):
        raise EvidenceError(f"{F06_KEY}.claim_limitations must be non-empty strings")

    f07 = _obj(fixture, F07_KEY, "fixture")
    _exact(
        f07,
        {
            "schema_version": "f07_finite_temperature_qed_v1",
            "implementation_status": "IMPLEMENTED",
            "claim_status": "VALIDATED",
            "captured_date": "2026-07-15",
            "evidence_mode": "stored_results_from_live_runs_and_independent_component_integrals",
        },
        F07_KEY,
    )
    f07_replay = _obj(f07, "stored_replay_contract", F07_KEY)
    _exact(
        f07_replay,
        {"executes_external_code": False, "may_be_reported_as_a_live_run": False},
        f"{F07_KEY}.stored_replay_contract",
    )
    convention = _obj(f07, "convention", F07_KEY)
    _number(
        convention.get("fine_structure_alpha"),
        f"{F07_KEY}.convention.fine_structure_alpha",
    )
    _number(
        convention.get("electron_mass_mev"),
        f"{F07_KEY}.convention.electron_mass_mev",
    )
    convention_sources = _obj(convention, "sources", f"{F07_KEY}.convention")
    _commit(
        {"commit": convention_sources.get("primat_commit")},
        f"{F07_KEY}.convention.sources",
    )

    dpb = _obj(f07, "independent_scipy_dpb", F07_KEY)
    _exact(
        dpb,
        {"execution_status": "VALIDATED"},
        f"{F07_KEY}.independent_scipy_dpb",
    )
    dpb_anchors = dpb.get("anchors")
    if not isinstance(dpb_anchors, list) or len(dpb_anchors) != 3:
        raise EvidenceError(
            f"{F07_KEY}.independent_scipy_dpb.anchors must contain three points"
        )
    for index, anchor in enumerate(dpb_anchors):
        if not isinstance(anchor, dict):
            raise EvidenceError(
                f"{F07_KEY}.independent_scipy_dpb.anchors[{index}] must be an object"
            )
        for key in (
            "temperature_mev",
            "pressure_mev4",
            "dpressure_dt_mev3",
            "d2pressure_dt2_mev2",
        ):
            _number(
                anchor.get(key),
                f"{F07_KEY}.independent_scipy_dpb.anchors[{index}].{key}",
            )

    primat_f07 = _obj(f07, "primat_leading_endpoint", F07_KEY)
    _exact(
        primat_f07,
        {"execution_status": "VALIDATED", "version": "0.3.2"},
        f"{F07_KEY}.primat_leading_endpoint",
    )
    _commit(primat_f07, f"{F07_KEY}.primat_leading_endpoint")
    primat_off_f07 = _outputs(
        _obj(
            primat_f07,
            "qed_off_python",
            f"{F07_KEY}.primat_leading_endpoint",
        ),
        f"{F07_KEY}.primat_leading_endpoint.qed_off_python",
    )
    primat_leading_f07 = _outputs(
        _obj(
            primat_f07,
            "qed_leading_python",
            f"{F07_KEY}.primat_leading_endpoint",
        ),
        f"{F07_KEY}.primat_leading_endpoint.qed_leading_python",
    )
    serialized_primat_delta = _obj(
        primat_f07,
        "qed_leading_minus_off_python",
        f"{F07_KEY}.primat_leading_endpoint",
    )
    for key in (*OBSERVABLES, "Neff"):
        actual = _finite_number(
            serialized_primat_delta.get(key),
            f"{F07_KEY}.primat_leading_endpoint.qed_leading_minus_off_python.{key}",
        )
        if key in OBSERVABLES:
            expected = primat_leading_f07[key] - primat_off_f07[key]
            if actual != expected:
                raise EvidenceError(
                    f"{F07_KEY}.primat_leading_endpoint serialized {key} delta is inconsistent"
                )

    rust_f07 = _obj(f07, "rust", F07_KEY)
    _exact(rust_f07, {"execution_status": "VALIDATED"}, f"{F07_KEY}.rust")
    rust_profile = _obj(rust_f07, "profile", f"{F07_KEY}.rust")
    _exact(
        rust_profile,
        {
            "qed_single_integral_panels": 256,
            "qed_exchange_gauss_legendre_order": 48,
            "qed_tail_e_folds": 48.0,
            "relative_tolerance": 1e-9,
            "weak_mode": "Born",
            "reaction_count": 12,
        },
        f"{F07_KEY}.rust.profile",
    )
    rust_f07_outputs: dict[str, dict[str, float]] = {}
    for name, solver, model in (
        ("off_bdf", "BDF", "Off"),
        ("leading_bdf", "BDF", "PrimatLeadingE2E3"),
        ("complete_bdf", "BDF", "PrimatCompleteE2E3"),
        ("complete_rodas5p", "Rodas5P", "PrimatCompleteE2E3"),
    ):
        record = _obj(rust_f07, name, f"{F07_KEY}.rust")
        _exact(
            record,
            {"solver": solver, "qed_model": model},
            f"{F07_KEY}.rust.{name}",
        )
        if name.startswith("complete_"):
            _exact(
                record,
                {"endpoint_claim_status": "IMPLEMENTED"},
                f"{F07_KEY}.rust.{name}",
            )
        _number(record.get("Xn_handoff"), f"{F07_KEY}.rust.{name}.Xn_handoff")
        rust_f07_outputs[name] = _outputs(record, f"{F07_KEY}.rust.{name}")

    f07_limitations = f07.get("claim_limitations")
    if not isinstance(f07_limitations, list) or not f07_limitations or not all(
        isinstance(item, str) and item for item in f07_limitations
    ):
        raise EvidenceError(f"{F07_KEY}.claim_limitations must be non-empty strings")

    f08a = _obj(fixture, F08A_KEY, "fixture")
    _exact(
        f08a,
        {
            "schema_version": "f08a_zero_temperature_ccr_v1",
            "implementation_status": "IMPLEMENTED",
            "claim_status": "VALIDATED",
            "captured_date": "2026-07-15",
        },
        F08A_KEY,
    )
    f08a_convention = _obj(f08a, "convention", F08A_KEY)
    _number(
        f08a_convention.get("threshold_beta_times_fermi"),
        f"{F08A_KEY}.convention.threshold_beta_times_fermi",
    )
    f08a_sources = _obj(f08a_convention, "sources", f"{F08A_KEY}.convention")
    _exact(
        f08a_sources,
        {"primat_version": "0.3.2"},
        f"{F08A_KEY}.convention.sources",
    )
    _commit(
        {"commit": f08a_sources.get("primat_commit")},
        f"{F08A_KEY}.convention.sources",
    )

    f08a_components = _obj(f08a, "component_anchors", F08A_KEY)
    _exact(
        f08a_components,
        {"execution_status": "VALIDATED"},
        f"{F08A_KEY}.component_anchors",
    )
    _number(f08a_components.get("born_fn"), f"{F08A_KEY}.component_anchors.born_fn")
    _number(f08a_components.get("ccr_fn"), f"{F08A_KEY}.component_anchors.ccr_fn")
    equal_temperature = f08a_components.get("equal_temperature_rates")
    if not isinstance(equal_temperature, list) or len(equal_temperature) != 2:
        raise EvidenceError(
            f"{F08A_KEY}.component_anchors.equal_temperature_rates must contain two points"
        )
    for index, anchor in enumerate(equal_temperature):
        if not isinstance(anchor, dict):
            raise EvidenceError(
                f"{F08A_KEY}.component_anchors.equal_temperature_rates[{index}] must be an object"
            )
        for key in (
            "temperature_mev",
            "neutron_to_proton_per_second",
            "proton_to_neutron_per_second",
        ):
            _number(
                anchor.get(key),
                f"{F08A_KEY}.component_anchors.equal_temperature_rates[{index}].{key}",
            )
    _number(
        f08a_components.get("equal_temperature_detailed_balance_max_absolute_residual"),
        f"{F08A_KEY}.component_anchors.equal_temperature_detailed_balance_max_absolute_residual",
    )

    f08a_primat = _obj(f08a, "primat_matched_endpoint", F08A_KEY)
    _exact(
        f08a_primat,
        {"execution_status": "VALIDATED"},
        f"{F08A_KEY}.primat_matched_endpoint",
    )
    f08a_primat_outputs: dict[str, dict[str, float]] = {}
    for name in ("born_python", "ccr_python", "born_c", "ccr_c"):
        f08a_primat_outputs[name] = _outputs(
            _obj(f08a_primat, name, f"{F08A_KEY}.primat_matched_endpoint"),
            f"{F08A_KEY}.primat_matched_endpoint.{name}",
        )
    for backend in ("python", "c"):
        serialized = _obj(
            f08a_primat,
            f"ccr_minus_born_{backend}",
            f"{F08A_KEY}.primat_matched_endpoint",
        )
        for key in OBSERVABLES:
            actual = _finite_number(
                serialized.get(key),
                f"{F08A_KEY}.primat_matched_endpoint.ccr_minus_born_{backend}.{key}",
            )
            expected = (
                f08a_primat_outputs[f"ccr_{backend}"][key]
                - f08a_primat_outputs[f"born_{backend}"][key]
            )
            if actual != expected:
                raise EvidenceError(
                    f"{F08A_KEY}.primat_matched_endpoint serialized {backend} {key} delta is inconsistent"
                )

    f08a_freezeout = _obj(
        f08a, "independent_temperature_variable_freezeout", F08A_KEY
    )
    _exact(
        f08a_freezeout,
        {
            "schema_version": "f08a_temperature_variable_freezeout_oracle_v1",
            "execution_status": "VALIDATED",
        },
        f"{F08A_KEY}.independent_temperature_variable_freezeout",
    )
    f08a_freezeout_method = _obj(
        f08a_freezeout,
        "method",
        f"{F08A_KEY}.independent_temperature_variable_freezeout",
    )
    _exact(
        f08a_freezeout_method,
        {
            "independent_variable": "Tgamma [MeV], decreasing 10 -> 0.05",
            "integrator": "SciPy DOP853",
            "electron_eos": (
                "adaptive quad_vec in y=p/T, not Rust theta/Simpson quadrature"
            ),
            "weak_rates": (
                "adaptive infinite/bounded physical-channel quadrature, not Rust "
                "fixed Gauss-Legendre nodes"
            ),
            "qed_background": (
                "public PRIMAT dPa+dPe3 formulas on nested log-T splines"
            ),
            "initial_neutron_fraction": (
                "model-specific lambda_pn/(lambda_np+lambda_pn) at 10 MeV"
            ),
        },
        f"{F08A_KEY}.independent_temperature_variable_freezeout.method",
    )
    f08a_freezeout_config = _obj(
        f08a_freezeout,
        "configuration",
        f"{F08A_KEY}.independent_temperature_variable_freezeout",
    )
    _exact(
        f08a_freezeout_config,
        {
            "temperature_start_mev": 10.0,
            "activation_temperature_mev": 0.86164715286738,
            "temperature_end_mev": 0.05,
            "neutron_lifetime_seconds": 878.4,
            "baseline_rtol": 1e-9,
            "baseline_atol": 1e-12,
            "rust_standalone_rtol": 2e-8,
            "rust_standalone_atol": [1e-10, 1e-11],
            "qed_grid_points": 2401,
        },
        f"{F08A_KEY}.independent_temperature_variable_freezeout.configuration",
    )
    f08a_freezeout_states: dict[str, dict[str, dict[str, float]]] = {}
    for model in ("born_baseline", "ccr_baseline"):
        record = _obj(
            f08a_freezeout,
            model,
            f"{F08A_KEY}.independent_temperature_variable_freezeout",
        )
        _number(
            record.get("Fn"),
            f"{F08A_KEY}.independent_temperature_variable_freezeout.{model}.Fn",
        )
        f08a_freezeout_states[model] = {}
        for endpoint in ("activation", "final"):
            values = _obj(
                record,
                endpoint,
                f"{F08A_KEY}.independent_temperature_variable_freezeout.{model}",
            )
            f08a_freezeout_states[model][endpoint] = _state(
                values,
                f"{F08A_KEY}.independent_temperature_variable_freezeout.{model}.{endpoint}",
            )
    stability = _obj(
        f08a_freezeout,
        "numerical_stability",
        f"{F08A_KEY}.independent_temperature_variable_freezeout",
    )
    _exact(
        stability,
        {
            "same_process_repeat_bitwise_equal": True,
            "fresh_process_repeat_bitwise_equal": True,
            "nested_qed_grid_points": [601, 1201, 2401],
        },
        f"{F08A_KEY}.independent_temperature_variable_freezeout.numerical_stability",
    )
    for key in (
        "tolerance_ladder_max_absolute_N_delta",
        "tolerance_ladder_max_absolute_Xn_delta",
        "nested_qed_grid_max_N_spread",
        "nested_qed_grid_max_Xn_spread",
        "max_absolute_N_minus_entropy_identity",
    ):
        _number(
            stability.get(key),
            f"{F08A_KEY}.independent_temperature_variable_freezeout.numerical_stability.{key}",
        )
    f08a_freezeout_acceptance = _obj(
        f08a_freezeout,
        "rust_acceptance",
        f"{F08A_KEY}.independent_temperature_variable_freezeout",
    )
    _exact(
        f08a_freezeout_acceptance,
        {
            "max_absolute_N_difference": 3e-7,
            "max_absolute_Xn_difference": 2e-8,
        },
        f"{F08A_KEY}.independent_temperature_variable_freezeout.rust_acceptance",
    )
    f08a_freezeout_provenance = _obj(
        f08a_freezeout,
        "provenance",
        f"{F08A_KEY}.independent_temperature_variable_freezeout",
    )
    _exact(
        f08a_freezeout_provenance,
        {"validation_checks_passed": 11, "validation_checks_total": 11},
        f"{F08A_KEY}.independent_temperature_variable_freezeout.provenance",
    )
    for key in (
        "result_sha256",
        "raw_sha256",
        "validation_sha256",
        "runner_sha256",
        "qed_grid_augmenter_sha256",
        "cold_repeat_augmenter_sha256",
        "validator_sha256",
    ):
        _sha256(
            f08a_freezeout_provenance.get(key),
            f"{F08A_KEY}.independent_temperature_variable_freezeout.provenance.{key}",
        )

    f08a_rust = _obj(f08a, "rust", F08A_KEY)
    _exact(f08a_rust, {"execution_status": "VALIDATED"}, f"{F08A_KEY}.rust")
    f08a_rust_profile = _obj(f08a_rust, "profile", f"{F08A_KEY}.rust")
    _exact(
        f08a_rust_profile,
        {
            "qed_model": "PrimatLeadingE2E3",
            "weak_model": "PrimatZeroTemperatureCcr",
            "weak_quadrature_order": 64,
            "weak_leg_relative_tolerance": 2e-8,
            "coupled_relative_tolerance": 1e-9,
            "reaction_count": 12,
        },
        f"{F08A_KEY}.rust.profile",
    )
    f08a_rust_outputs: dict[str, dict[str, float]] = {}
    for name, solver in (("bdf", "BDF"), ("rodas5p", "Rodas5P")):
        record = _obj(f08a_rust, name, f"{F08A_KEY}.rust")
        _exact(record, {"solver": solver}, f"{F08A_KEY}.rust.{name}")
        _number(record.get("Xn_handoff"), f"{F08A_KEY}.rust.{name}.Xn_handoff")
        f08a_rust_outputs[name] = _outputs(record, f"{F08A_KEY}.rust.{name}")

    f08a_provenance = _obj(f08a, "provenance", F08A_KEY)
    for key in (
        "result_sha256",
        "raw_sha256",
        "validation_sha256",
        "runner_sha256",
        "validator_sha256",
        "primat_c_backend_sha256",
        "rabbit_rate_source_sha256",
        "dense_custom_network_sha256",
        "f07_electron_thermo_cache_sha256",
    ):
        _sha256(f08a_provenance.get(key), f"{F08A_KEY}.provenance.{key}")
    f08a_limitations = f08a.get("claim_limitations")
    if not isinstance(f08a_limitations, list) or not f08a_limitations or not all(
        isinstance(item, str) and item for item in f08a_limitations
    ):
        raise EvidenceError(f"{F08A_KEY}.claim_limitations must be non-empty strings")

    f08b = _obj(fixture, F08B_KEY, "fixture")
    _exact(
        f08b,
        {
            "schema_version": (
                "f08b_finite_nucleon_mass_no_weak_magnetism_v1"
            ),
            "implementation_status": "IMPLEMENTED",
            "claim_status": "VALIDATED",
            "captured_date": "2026-07-15",
            "evidence_mode": (
                "stored_results_from_live_runs_and_independent_component_integrals"
            ),
        },
        F08B_KEY,
    )
    _nonempty_string(f08b.get("scope"), f"{F08B_KEY}.scope")
    f08b_replay = _obj(f08b, "stored_replay_contract", F08B_KEY)
    _exact(
        f08b_replay,
        {"executes_external_code": False, "may_be_reported_as_a_live_run": False},
        f"{F08B_KEY}.stored_replay_contract",
    )

    f08b_convention = _obj(f08b, "convention", F08B_KEY)
    _exact(
        f08b_convention,
        {
            "finite_mass_order": "first order in inverse nucleon mass",
            "weak_magnetism_delta_kappa": 0.0,
            "weak_magnetism_status": "FORCED_OFF",
            "external_endpoint_backend": "PRIMAT v0.3.2 Python",
        },
        f"{F08B_KEY}.convention",
    )

    f08b_components = _obj(f08b, "component_anchors", F08B_KEY)
    _exact(
        f08b_components,
        {"execution_status": "VALIDATED"},
        f"{F08B_KEY}.component_anchors",
    )
    f08b_fn = _obj(f08b_components, "fn", f"{F08B_KEY}.component_anchors")
    fn_ccr = _number(
        f08b_fn.get("ccr_fn"), f"{F08B_KEY}.component_anchors.fn.ccr_fn"
    )
    fn_primat_delta = _finite_number(
        f08b_fn.get("primat_finite_mass_delta_fn"),
        f"{F08B_KEY}.component_anchors.fn.primat_finite_mass_delta_fn",
    )
    fn_total = _number(
        f08b_fn.get("f08b_total_fn"),
        f"{F08B_KEY}.component_anchors.fn.f08b_total_fn",
    )
    fn_compact_delta = _finite_number(
        f08b_fn.get("independent_compact_finite_mass_delta_fn"),
        (
            f"{F08B_KEY}.component_anchors.fn."
            "independent_compact_finite_mass_delta_fn"
        ),
    )
    compact_minus_primat = _finite_number(
        f08b_fn.get("compact_minus_primat_delta_fn"),
        f"{F08B_KEY}.component_anchors.fn.compact_minus_primat_delta_fn",
    )
    if fn_total != fn_ccr + fn_primat_delta:
        raise EvidenceError(
            f"{F08B_KEY}.component_anchors.fn total must equal CCR plus PRIMAT delta"
        )
    if compact_minus_primat != fn_compact_delta - fn_primat_delta:
        raise EvidenceError(
            f"{F08B_KEY}.component_anchors.fn compact delta difference is inconsistent"
        )

    compact_grid = _obj(
        f08b_components,
        "compact_vs_expanded_point_grid",
        f"{F08B_KEY}.component_anchors",
    )
    _exact(
        compact_grid,
        {"row_count": 252},
        f"{F08B_KEY}.component_anchors.compact_vs_expanded_point_grid",
    )
    for key in ("max_absolute_difference", "max_scale_relative_difference"):
        _nonnegative_number(
            compact_grid.get(key),
            f"{F08B_KEY}.component_anchors.compact_vs_expanded_point_grid.{key}",
        )

    equal_temperature = f08b_components.get("equal_temperature_rates")
    if not isinstance(equal_temperature, list) or len(equal_temperature) != 3:
        raise EvidenceError(
            f"{F08B_KEY}.component_anchors.equal_temperature_rates must contain three points"
        )
    for index, expected_temperature in enumerate((0.3, 1.0, 3.0)):
        anchor = equal_temperature[index]
        path = f"{F08B_KEY}.component_anchors.equal_temperature_rates[{index}]"
        if not isinstance(anchor, dict):
            raise EvidenceError(f"{path} must be an object")
        _exact(
            anchor,
            {
                "photon_temperature_mev": expected_temperature,
                "neutrino_temperature_mev": expected_temperature,
            },
            path,
        )
        for key in (
            "neutron_to_proton_per_second",
            "proton_to_neutron_per_second",
        ):
            _number(anchor.get(key), f"{path}.{key}")

    unequal = _obj(
        f08b_components,
        "unequal_temperature_six_channel_anchor",
        f"{F08B_KEY}.component_anchors",
    )
    photon_temperature = _number(
        unequal.get("photon_temperature_mev"),
        (
            f"{F08B_KEY}.component_anchors."
            "unequal_temperature_six_channel_anchor.photon_temperature_mev"
        ),
    )
    neutrino_temperature = _number(
        unequal.get("neutrino_temperature_mev"),
        (
            f"{F08B_KEY}.component_anchors."
            "unequal_temperature_six_channel_anchor.neutrino_temperature_mev"
        ),
    )
    if photon_temperature == neutrino_temperature:
        raise EvidenceError(
            f"{F08B_KEY}.component_anchors unequal-temperature anchor must have Tgamma != Tnu"
        )
    channels = _obj(
        unequal,
        "channels_per_second",
        f"{F08B_KEY}.component_anchors.unequal_temperature_six_channel_anchor",
    )
    if set(channels) != set(F08B_CHANNELS):
        raise EvidenceError(
            f"{F08B_KEY}.component_anchors unequal-temperature anchor must contain exactly six named channels"
        )
    channel_values = {
        key: _number(
            channels.get(key),
            (
                f"{F08B_KEY}.component_anchors."
                f"unequal_temperature_six_channel_anchor.channels_per_second.{key}"
            ),
        )
        for key in F08B_CHANNELS
    }
    unequal_totals = _obj(
        unequal,
        "totals_per_second",
        f"{F08B_KEY}.component_anchors.unequal_temperature_six_channel_anchor",
    )
    unequal_np = _number(
        unequal_totals.get("neutron_to_proton"),
        (
            f"{F08B_KEY}.component_anchors."
            "unequal_temperature_six_channel_anchor.totals_per_second.neutron_to_proton"
        ),
    )
    unequal_pn = _number(
        unequal_totals.get("proton_to_neutron"),
        (
            f"{F08B_KEY}.component_anchors."
            "unequal_temperature_six_channel_anchor.totals_per_second.proton_to_neutron"
        ),
    )
    expected_np = (
        channel_values["nu_e_n_to_p_electron"]
        + channel_values["positron_n_to_p_anti_nu_e"]
        + channel_values["free_neutron_decay"]
    )
    expected_pn = (
        channel_values["electron_p_to_n_nu_e"]
        + channel_values["anti_nu_e_p_to_n_positron"]
        + channel_values["inverse_neutron_decay"]
    )
    np_consistent = math.isclose(
        unequal_np,
        expected_np,
        rel_tol=0.0,
        abs_tol=max(math.ulp(unequal_np), math.ulp(expected_np)),
    )
    pn_consistent = math.isclose(
        unequal_pn,
        expected_pn,
        rel_tol=0.0,
        abs_tol=max(math.ulp(unequal_pn), math.ulp(expected_pn)),
    )
    if not np_consistent or not pn_consistent:
        raise EvidenceError(
            f"{F08B_KEY}.component_anchors unequal-temperature totals are inconsistent beyond one serialized float ULP"
        )

    detailed_balance = _obj(
        f08b_components,
        "modified_detailed_balance",
        f"{F08B_KEY}.component_anchors",
    )
    _exact(
        detailed_balance,
        {
            "temperature_mev": 1.0,
            "target_name": "(mn/mp)^(3/2)*exp(-Q/T)",
        },
        f"{F08B_KEY}.component_anchors.modified_detailed_balance",
    )
    target_ratio = _number(
        detailed_balance.get("target_ratio"),
        f"{F08B_KEY}.component_anchors.modified_detailed_balance.target_ratio",
    )
    observed_ratio = _number(
        detailed_balance.get("observed_total_ratio"),
        (
            f"{F08B_KEY}.component_anchors.modified_detailed_balance."
            "observed_total_ratio"
        ),
    )
    relative_residual = _finite_number(
        detailed_balance.get("relative_residual"),
        f"{F08B_KEY}.component_anchors.modified_detailed_balance.relative_residual",
    )
    recorded_ceiling = _number(
        detailed_balance.get("recorded_grid_max_absolute_relative_residual"),
        (
            f"{F08B_KEY}.component_anchors.modified_detailed_balance."
            "recorded_grid_max_absolute_relative_residual"
        ),
    )
    if relative_residual == 0.0:
        raise EvidenceError(
            f"{F08B_KEY}.component_anchors modified detailed-balance residual must remain nonzero"
        )
    if relative_residual != (observed_ratio - target_ratio) / target_ratio:
        raise EvidenceError(
            f"{F08B_KEY}.component_anchors modified detailed-balance residual is inconsistent"
        )
    if abs(relative_residual) > recorded_ceiling:
        raise EvidenceError(
            f"{F08B_KEY}.component_anchors modified detailed-balance residual exceeds its recorded grid maximum"
        )

    f08b_freezeout = _obj(
        f08b, "standalone_temperature_variable_freezeout", F08B_KEY
    )
    _exact(
        f08b_freezeout,
        {
            "schema_version": "f08b_temperature_variable_freezeout_v1",
            "execution_status": "VALIDATED",
        },
        f"{F08B_KEY}.standalone_temperature_variable_freezeout",
    )
    freezeout_method = _obj(
        f08b_freezeout,
        "method",
        f"{F08B_KEY}.standalone_temperature_variable_freezeout",
    )
    _exact(
        freezeout_method,
        {
            "independent_variable": (
                "T_gamma [MeV], integrated from 10 to 0.05; cosmic time is a "
                "co-evolved dependent state via dt/dT_gamma, not the ODE coordinate."
            ),
            "solver": "scipy.solve_ivp DOP853, rtol=2e-10",
        },
        f"{F08B_KEY}.standalone_temperature_variable_freezeout.method",
    )
    freezeout_config = _obj(
        f08b_freezeout,
        "configuration",
        f"{F08B_KEY}.standalone_temperature_variable_freezeout",
    )
    _exact(
        freezeout_config,
        {
            "temperature_start_mev": 10.0,
            "activation_temperature_mev": 0.86164715286738,
            "temperature_end_mev": 0.05,
            "neutron_lifetime_seconds": 878.4,
            "rate_grid_points": 321,
            "gauss_legendre_nodes_per_panel": [80, 160],
            "qed_model": "PrimatLeadingE2E3",
            "weak_magnetism_delta_kappa": 0.0,
        },
        f"{F08B_KEY}.standalone_temperature_variable_freezeout.configuration",
    )
    f08b_freezeout_states: dict[str, dict[str, dict[str, float]]] = {}
    for model in ("ccr_gl160", "f08b_gl160"):
        record = _obj(
            f08b_freezeout,
            model,
            f"{F08B_KEY}.standalone_temperature_variable_freezeout",
        )
        _exact(
            record,
            {"success": True},
            f"{F08B_KEY}.standalone_temperature_variable_freezeout.{model}",
        )
        f08b_freezeout_states[model] = {}
        for endpoint in ("activation", "final"):
            state = _obj(
                record,
                endpoint,
                f"{F08B_KEY}.standalone_temperature_variable_freezeout.{model}",
            )
            f08b_freezeout_states[model][endpoint] = _state(
                state,
                (
                    f"{F08B_KEY}.standalone_temperature_variable_freezeout."
                    f"{model}.{endpoint}"
                ),
            )
    for endpoint in ("activation", "final"):
        if (
            f08b_freezeout_states["ccr_gl160"][endpoint]["N"]
            != f08b_freezeout_states["f08b_gl160"][endpoint]["N"]
        ):
            raise EvidenceError(
                f"{F08B_KEY}.standalone_temperature_variable_freezeout.{endpoint} N must share the entropy identity"
            )

    freezeout_stability = _obj(
        f08b_freezeout,
        "numerical_stability",
        f"{F08B_KEY}.standalone_temperature_variable_freezeout",
    )
    _exact(
        freezeout_stability,
        {"gauss_legendre_nodes_per_panel": [80, 160]},
        f"{F08B_KEY}.standalone_temperature_variable_freezeout.numerical_stability",
    )
    for key in (
        "ccr_max_absolute_Xn_gl160_minus_gl80",
        "f08b_max_absolute_Xn_gl160_minus_gl80",
    ):
        _nonnegative_number(
            freezeout_stability.get(key),
            (
                f"{F08B_KEY}.standalone_temperature_variable_freezeout."
                f"numerical_stability.{key}"
            ),
        )

    f08b_freezeout_acceptance = _obj(
        f08b_freezeout,
        "rust_acceptance",
        f"{F08B_KEY}.standalone_temperature_variable_freezeout",
    )
    _exact(
        f08b_freezeout_acceptance,
        {
            "max_absolute_N_difference": 3e-7,
            "max_absolute_Xn_difference": 6e-8,
        },
        f"{F08B_KEY}.standalone_temperature_variable_freezeout.rust_acceptance",
    )
    _nonempty_string(
        f08b_freezeout_acceptance.get("basis"),
        (
            f"{F08B_KEY}.standalone_temperature_variable_freezeout."
            "rust_acceptance.basis"
        ),
    )

    f08a_crosscheck = _obj(
        f08b_freezeout,
        "f08a_ccr_crosscheck",
        f"{F08B_KEY}.standalone_temperature_variable_freezeout",
    )
    for endpoint in ("activation", "final"):
        serialized_delta = _obj(
            f08a_crosscheck,
            f"ccr_gl160_minus_f08a_{endpoint}",
            (
                f"{F08B_KEY}.standalone_temperature_variable_freezeout."
                "f08a_ccr_crosscheck"
            ),
        )
        for key in ("N", "Xn"):
            actual = _finite_number(
                serialized_delta.get(key),
                (
                    f"{F08B_KEY}.standalone_temperature_variable_freezeout."
                    f"f08a_ccr_crosscheck.ccr_gl160_minus_f08a_{endpoint}.{key}"
                ),
            )
            expected = (
                f08b_freezeout_states["ccr_gl160"][endpoint][key]
                - f08a_freezeout_states["ccr_baseline"][endpoint][key]
            )
            if actual != expected:
                raise EvidenceError(
                    f"{F08B_KEY}.standalone_temperature_variable_freezeout F08A {endpoint} {key} crosscheck is inconsistent"
                )

    f08b_primat = _obj(f08b, "primat_matched_endpoint", F08B_KEY)
    _exact(
        f08b_primat,
        {
            "execution_status": "VALIDATED",
            "version": "0.3.2",
            "backend": "python",
        },
        f"{F08B_KEY}.primat_matched_endpoint",
    )
    _commit(f08b_primat, f"{F08B_KEY}.primat_matched_endpoint")
    f08b_c_backend = _obj(
        f08b_primat, "c_backend", f"{F08B_KEY}.primat_matched_endpoint"
    )
    _exact(
        f08b_c_backend,
        {"status": "SKIPPED"},
        f"{F08B_KEY}.primat_matched_endpoint.c_backend",
    )
    _nonempty_string(
        f08b_c_backend.get("reason"),
        f"{F08B_KEY}.primat_matched_endpoint.c_backend.reason",
    )
    f08b_primat_outputs = {
        name: _endpoint_outputs(
            _obj(f08b_primat, name, f"{F08B_KEY}.primat_matched_endpoint"),
            f"{F08B_KEY}.primat_matched_endpoint.{name}",
        )
        for name in ("born_python", "ccr_python", "f08b_python")
    }
    for name, reference, candidate in (
        ("ccr_minus_born_python", "born_python", "ccr_python"),
        ("f08b_minus_born_python", "born_python", "f08b_python"),
        ("f08b_minus_ccr_python", "ccr_python", "f08b_python"),
    ):
        _serialized_endpoint_delta(
            _obj(f08b_primat, name, f"{F08B_KEY}.primat_matched_endpoint"),
            f08b_primat_outputs[reference],
            f08b_primat_outputs[candidate],
            f"{F08B_KEY}.primat_matched_endpoint.{name}",
        )

    f08b_rust = _obj(f08b, "rust", F08B_KEY)
    _exact(f08b_rust, {"execution_status": "VALIDATED"}, f"{F08B_KEY}.rust")
    f08b_rust_profile = _obj(f08b_rust, "profile", f"{F08B_KEY}.rust")
    _exact(
        f08b_rust_profile,
        {
            "qed_model": "PrimatLeadingE2E3",
            "weak_model": (
                "PrimatZeroTemperatureCcrFiniteMassNoWeakMagnetism"
            ),
            "finite_mass_order": "first order in inverse nucleon mass",
            "weak_magnetism_delta_kappa": 0.0,
            "weak_quadrature_order": 64,
            "weak_leg_relative_tolerance": 2e-8,
            "coupled_relative_tolerance": 1e-9,
            "reaction_count": 12,
        },
        f"{F08B_KEY}.rust.profile",
    )
    f08b_rust_outputs: dict[str, dict[str, float]] = {}
    for name, solver in (("bdf", "BDF"), ("rodas5p", "Rodas5P")):
        record = _obj(f08b_rust, name, f"{F08B_KEY}.rust")
        _exact(record, {"solver": solver}, f"{F08B_KEY}.rust.{name}")
        _number(record.get("Xn_handoff"), f"{F08B_KEY}.rust.{name}.Xn_handoff")
        f08b_rust_outputs[name] = _outputs(record, f"{F08B_KEY}.rust.{name}")

    f08b_provenance = _obj(f08b, "provenance", F08B_KEY)
    _exact(
        f08b_provenance,
        {
            "primat_version": "0.3.2",
            "primat_tag": "v0.3.2",
            "forced_delta_kappa": 0.0,
        },
        f"{F08B_KEY}.provenance",
    )
    _commit(
        {"commit": f08b_provenance.get("primat_commit")},
        f"{F08B_KEY}.provenance",
    )
    for key in (
        "result_sha256",
        "raw_sha256",
        "validation_sha256",
        "runner_sha256",
        "validator_sha256",
        "corrections_source_sha256",
        "integrands_source_sha256",
        "primat_module_sha256",
        "rabbit_rate_source_sha256",
        "dense_custom_network_sha256",
        "f07_electron_thermo_cache_sha256",
    ):
        _sha256(f08b_provenance.get(key), f"{F08B_KEY}.provenance.{key}")
    checks_passed = f08b_provenance.get("validation_checks_passed")
    checks_total = f08b_provenance.get("validation_checks_total")
    if (
        isinstance(checks_passed, bool)
        or not isinstance(checks_passed, int)
        or isinstance(checks_total, bool)
        or not isinstance(checks_total, int)
        or checks_total <= 0
        or checks_passed != checks_total
    ):
        raise EvidenceError(
            f"{F08B_KEY}.provenance validation checks must be positive and all pass"
        )
    f08b_limitations = f08b.get("claim_limitations")
    if not isinstance(f08b_limitations, list) or not f08b_limitations or not all(
        isinstance(item, str) and item for item in f08b_limitations
    ):
        raise EvidenceError(f"{F08B_KEY}.claim_limitations must be non-empty strings")

    f08c_values = _validate_f08c(fixture, f08b_primat_outputs)
    f08d_values = _validate_f08d(fixture, f08c_values["f08c_rust"])
    f08n_values = _validate_f08n(fixture, f08d_values["f08d_rust"])

    return {
        "f06": f06,
        "stock_c": _outputs(_obj(stock, "c", "primat.stock"), "primat.stock.c"),
        "stock_python": _outputs(
            _obj(stock, "python", "primat.stock"), "primat.stock.python"
        ),
        "custom_c": _outputs(_obj(custom, "c", "primat.custom"), "primat.custom.c"),
        "custom_python": _outputs(
            _obj(custom, "python", "primat.custom"), "primat.custom.python"
        ),
        "rust_bdf": _outputs(bdf, "rust.bdf"),
        "rust_rodas": _outputs(rodas, "rust.rodas5p"),
        "linx": linx_runs,
        "f07": f07,
        "f07_primat_off": primat_off_f07,
        "f07_primat_leading": primat_leading_f07,
        "f07_rust": rust_f07_outputs,
        "f08a": f08a,
        "f08a_primat": f08a_primat_outputs,
        "f08a_rust": f08a_rust_outputs,
        "f08b": f08b,
        "f08b_primat": f08b_primat_outputs,
        "f08b_rust": f08b_rust_outputs,
        **f08c_values,
        **f08d_values,
        **f08n_values,
    }


def _delta(reference: dict[str, float], candidate: dict[str, float]) -> dict[str, Any]:
    return {
        key: {
            "reference": reference[key],
            "candidate": candidate[key],
            "signed_delta": candidate[key] - reference[key],
            "relative_delta": (candidate[key] - reference[key]) / reference[key],
        }
        for key in OBSERVABLES
    }


def _f08n_delta(
    reference: dict[str, float], candidate: dict[str, float]
) -> dict[str, Any]:
    return {
        key: {
            "reference": reference[key],
            "candidate": candidate[key],
            "signed_delta": candidate[key] - reference[key],
            "relative_delta": (candidate[key] - reference[key]) / reference[key]
            if reference[key] != 0.0
            else None,
        }
        for key in F08N_OBSERVABLES
    }


def _load(path: Path) -> dict[str, Any]:
    try:
        with path.open(encoding="utf-8") as stream:
            value = json.load(stream)
    except (OSError, json.JSONDecodeError) as exc:
        raise EvidenceError(f"cannot read valid JSON fixture {path}: {exc}") from exc
    if not isinstance(value, dict):
        raise EvidenceError("fixture root must be an object")
    return value


def _report(path: Path, values: dict[str, Any]) -> dict[str, Any]:
    f06 = values["f06"]
    custom = values["custom_c"]
    return {
        "report_mode": "stored_anchor_replay",
        "live_external_code_executed": False,
        "fixture": str(path),
        "captured_date": f06["captured_date"],
        "claim_status": f06["claim_status"],
        "scope": f06["scope"],
        "serialized_evidence": f06,
        "f07_serialized_evidence": values["f07"],
        "f08a_serialized_evidence": values["f08a"],
        "f08b_serialized_evidence": values["f08b"],
        "f08c_serialized_evidence": values["f08c"],
        "f08d_serialized_evidence": values["f08d"],
        "f08d_promotion_status": values["f08d"]["promotion_status"],
        "f08n_serialized_evidence": values["f08n"],
        "f08n_promotion_status": values["f08n"]["promotion_status"],
        "diagnostic_deltas_no_acceptance_claim": {
            "primat_stock_python_minus_stock_c": _delta(
                values["stock_c"], values["stock_python"]
            ),
            "primat_custom_python_minus_custom_c": _delta(
                custom, values["custom_python"]
            ),
            "rust_bdf_minus_primat_custom_c": _delta(custom, values["rust_bdf"]),
            "rust_rodas5p_minus_primat_custom_c": _delta(custom, values["rust_rodas"]),
            "linx_rtol_1e_7_minus_primat_custom_c": _delta(
                custom, values["linx"]["rtol_1e_7"]
            ),
            "f07_primat_leading_minus_off": _delta(
                values["f07_primat_off"], values["f07_primat_leading"]
            ),
            "f07_rust_leading_minus_off": _delta(
                values["f07_rust"]["off_bdf"],
                values["f07_rust"]["leading_bdf"],
            ),
            "f08a_primat_ccr_minus_born": _delta(
                values["f08a_primat"]["born_python"],
                values["f08a_primat"]["ccr_python"],
            ),
            "f08a_rust_ccr_minus_born": _delta(
                values["f07_rust"]["leading_bdf"],
                values["f08a_rust"]["bdf"],
            ),
            "f08a_rust_bdf_minus_primat_python": _delta(
                values["f08a_primat"]["ccr_python"],
                values["f08a_rust"]["bdf"],
            ),
            "f08b_primat_ccr_minus_born": _delta(
                values["f08b_primat"]["born_python"],
                values["f08b_primat"]["ccr_python"],
            ),
            "f08b_primat_finite_mass_minus_born": _delta(
                values["f08b_primat"]["born_python"],
                values["f08b_primat"]["f08b_python"],
            ),
            "f08b_primat_finite_mass_minus_ccr": _delta(
                values["f08b_primat"]["ccr_python"],
                values["f08b_primat"]["f08b_python"],
            ),
            "f08b_rust_finite_mass_minus_f08a_ccr": _delta(
                values["f08a_rust"]["bdf"],
                values["f08b_rust"]["bdf"],
            ),
            "f08b_rust_bdf_minus_primat_python": _delta(
                values["f08b_primat"]["f08b_python"],
                values["f08b_rust"]["bdf"],
            ),
            "f08b_rust_rodas5p_minus_primat_python": _delta(
                values["f08b_primat"]["f08b_python"],
                values["f08b_rust"]["rodas5p"],
            ),
            "f08c_primat_python_minus_f08b_python": _delta(
                values["f08c_primat"]["f08b_python_reference"],
                values["f08c_primat"]["f08c_python"],
            ),
            "f08c_primat_c_minus_python": _delta(
                values["f08c_primat"]["f08c_python"],
                values["f08c_primat"]["f08c_c"],
            ),
            "f08c_rust_bdf_minus_f08b_rust_bdf": _delta(
                values["f08b_rust"]["bdf"],
                values["f08c_rust"]["bdf"],
            ),
            "f08c_rust_bdf_minus_primat_python": _delta(
                values["f08c_primat"]["f08c_python"],
                values["f08c_rust"]["bdf"],
            ),
            "f08c_rust_rodas5p_minus_primat_python": _delta(
                values["f08c_primat"]["f08c_python"],
                values["f08c_rust"]["rodas5p"],
            ),
            "f08d_external_python_minus_c": _delta(
                values["f08d_external"]["c"],
                values["f08d_external"]["python"],
            ),
            "f08d_rust_bdf_minus_f08c_rust_bdf": _delta(
                values["f08c_rust"]["bdf"],
                values["f08d_rust"]["bdf"],
            ),
            "f08d_rust_rodas5p_minus_f08c_rust_rodas5p": _delta(
                values["f08c_rust"]["rodas5p"],
                values["f08d_rust"]["rodas5p"],
            ),
            "f08d_rust_bdf_minus_external_python": _delta(
                values["f08d_external"]["python"],
                values["f08d_rust"]["bdf"],
            ),
            "f08d_rust_rodas5p_minus_external_c": _delta(
                values["f08d_external"]["c"],
                values["f08d_rust"]["rodas5p"],
            ),
            "f08n_external_exact_selected31_minus_exact12": _f08n_delta(
                values["f08n_external"]["exact_12"],
                values["f08n_external"]["exact_selected_31"],
            ),
            "f08n_rust_bdf_minus_external_exact_selected31": _f08n_delta(
                values["f08n_external"]["exact_selected_31"],
                values["f08n_rust"]["bdf"],
            ),
            "f08n_rust_rodas5p_minus_external_exact_selected31": _f08n_delta(
                values["f08n_external"]["exact_selected_31"],
                values["f08n_rust"]["rodas5p"],
            ),
        },
        "claim_limitations": (
            f06["claim_limitations"]
            + values["f07"]["claim_limitations"]
            + values["f08a"]["claim_limitations"]
            + values["f08b"]["claim_limitations"]
            + values["f08c"]["claim_limitations"]
            + values["f08d"]["claim_limitations"]
            + values["f08n"]["claim_limitations"]
        ),
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--fixture", type=Path, default=DEFAULT_FIXTURE)
    parser.add_argument("--mode", choices=("stored", "live"), default="stored")
    parser.add_argument("--compact", action="store_true")
    args = parser.parse_args(argv)
    if args.mode == "live":
        print(
            "ERROR: exact live PRIMAT/LINX/Rust replay is unsupported; "
            "no stored anchor was relabeled and no result was emitted.",
            file=sys.stderr,
        )
        return 2
    path = args.fixture.resolve()
    try:
        report = _report(path, _validate(_load(path)))
    except EvidenceError as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 2
    json.dump(report, sys.stdout, indent=None if args.compact else 2, sort_keys=True)
    sys.stdout.write("\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
