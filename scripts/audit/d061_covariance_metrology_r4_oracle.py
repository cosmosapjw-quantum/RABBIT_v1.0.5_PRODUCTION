"""BD622 D-061 — covariance-metrology oracle, fourth issue (r4 reissue).

Frozen contract: ``docs/audit/BD622_D061_covariance_metrology_contract_r4_2026-07-28.md``
(claim ``C-F10-METROLOGY-R3``, prospective evidence ``E-F10-D060-METROLOGY-R3``).

One-change derivative of the preserved D-060 r3 FAIL under the adjudicated
supersession scope: N7X gates only on family members with distinct mu/tau
state rows (the SA/SB blocks); on the P-fixed SS continuity states the
exchange-event observable is degenerate (elastic energy conservation zeroes
the Pauli affinity when mu and tau share one logit slope, leaving ~1e-34
cancellation noise), so the N7X value is computed and recorded non-gating
there. Every other member, threshold, formula, mutant, and interval tier is
behaviourally identical to the frozen r3 oracle (mechanical diff).

Repairs the two D-057 evidence defects of the D-055 r2 contract:
  * the defining equivariance identity F(Pf) = P F(f) is tested on frozen
    ASYMMETRIC state/swap pairs (plus boundary members and the P-fixed
    continuity states), with semantic mutants;
  * the native error analysis binds the IDENTICAL production operation graph
    native(self_modal) + native(electron_modal) (two maps + one float64 add)
    with an explicit E_split term, covers off-grid basis evaluations through a
    bitwise rate-equivariance check inside paired shadow assemblies, and adds
    an outward-rounded mpmath.iv interval replay (map stage for every state,
    full per-node assembly replay at one frozen node of the first pair).

Usage: PYTHONPATH=src venv/bin/python scripts/audit/d061_covariance_metrology_r4_oracle.py
           [--out PATH] [--dev-state] [--int2-cap N]

``--dev-state`` runs the disclosed non-family development state only (pre-freeze
hardening); ``--int2-cap`` limits M-INT-2 to the first N valid samples (dev use
only — the frozen family run must not pass either flag).

Exit codes: 0 PASS, 10 FAIL (preserved, no refit), 20 mechanical, 30 environment.
"""

from __future__ import annotations

import json
import os
import sys
import time

for _pin in ("OPENBLAS_NUM_THREADS", "OMP_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ[_pin] = "1"

import hashlib

import mpmath
import numpy as np

from rabbit.decoupling import _independent_noqke as ind

MODULE_SHA_PREFIX = "760a7c04"
NUMPY_VERSION = "2.4.4"
MPMATH_VERSION = "1.3.0"
TINY = float(np.finfo(float).tiny)
EPS = float(np.finfo(float).eps)

# --- frozen family -----------------------------------------------------------
# Asymmetric state/swap pairs (evaluated at f AND Pf), all at T_cm = 10 MeV.
FAMILY_PAIRS = {
    "SA-1": ((1.01, 0.997, 0.993), 10.0),   # D-057 audit probe 1
    "SA-2": ((1.05, 0.94, 1.03), 9.7),      # D-057 audit probe 2
    "SA-3": ((0.96, 1.04, 0.985), 9.7),     # D-057 audit probe 3 (worst native)
    "SB-1": ((0.90, 1.10, 0.95), 10.0),     # boundary: widest flavour asymmetry
    "SB-2": ((1.03, 0.94, 1.06), 9.0),      # boundary: largest gamma/nu split
}
# P-fixed continuity states (mu == tau bitwise; carry the D-055 anchors).
FAMILY_FIXED = {
    "SS-A": ((1.01, 0.995, 0.995), 10.0),
    "SS-D": ((1.0, 0.98, 0.98), 10.0),
    "SS-C": ((1.02, 0.99, 0.99), 10.0),
}
DEV_STATE = ((1.005, 0.998, 0.996), 9.9)    # disclosed non-family dev state
T_CM = 10.0

# --- frozen thresholds -------------------------------------------------------
CAP_DEFAULT = 1e-10                  # N1/N2 everywhere; N3 for SA/SS blocks
CAP_NATIVE_SB = 2.5e-10              # N3 for the SB boundary block (owner-granted)
CAP_ELECTRON = 1e-12                 # N4
CAP_SHADOW_MATCH = 1e-13             # N8 signed-shadow vs production self_modal
F_WEAK = 1e-21                       # denominator floors
F_NATIVE = 1e-24
ENV_D_COV = (1e-23, 1e-17)           # M-ENV prospective envelope
ENV_MODAL = (1e-20, 1e-15)
CAP_W_INT = 1e-8                     # M-INT-1 map-stage error cap (per sector)
INT1_PREC_BITS = 192
INT2_PREC_BITS = 128
INT2_NODE = 24                       # frozen mid-grid node (0-based)
INT2_PAIR = "SA-1"
INT2_WIDEN_REL = 2e-11               # relative containment term; each quantity
                                     # additionally carries a cancellation-aware
                                     # absolute pad from the frozen error model
INT2_K_GEOM = 512                    # geometry/dot absolute-pad multiplier
INT2_K_U = 64                        # native-logit u-error multiplier
INT2_MODES = (0, 9, 19, 29, 39, 47)  # frozen mode subset for contraction containment
N7X_NODES = (4, 14, 24, 34, 43, 47)  # frozen node subset for the exchange check
N7X_TOL = 1e-11                      # sup |rate_Pf - mapped rate_f| / max|rate|
                                     # (dev-state measured ~1.5e-12; fp-roundoff
                                     # class through the exchanged s/kinematics
                                     # recomputation)
N7X_MAX_MASK_MISMATCH = 16           # boundary support flips excluded (counted)
WALL_BUDGET_SECONDS = 8 * 3600.0

# --- D-055 bitwise anchors (N8) ---------------------------------------------
ANCHOR_D_NATIVE = {
    "SS-A": 6.261815481020115e-21,
    "SS-D": 1.0279153398476508e-20,
    "SS-C": 1.261753327326722e-20,
}
ANCHOR_R_NATIVE = {
    "SS-A": 2.447560236182917e-11,
    "SS-D": 1.5745492982469787e-11,
    "SS-C": 5.935964835407989e-12,
}

# --- mutant kill thresholds --------------------------------------------------
KILL_EQUIVARIANCE = 1e-8             # MUT-24 / MUT-LANE on N1 and N3 at SA-1
KILL_HALF_DEVIATION = 1e-6           # MUT-HALF pair_total deviation at SA-1(f)
KILL_SIGN_DEVIATION = 1e-3           # MUT-SIGN pair_total deviation at SA-1(f)

PI_SPECIES = (0, 1, 4, 5, 2, 3)      # X(Pf)[i] == X(f)[PI_SPECIES[i]]
PAIR_PERM = (0, 2, 1)                # pair rows (e, mu, tau) under P
_FLAVOUR_SWAP = {"e": "e", "mu": "tau", "tau": "mu"}

START_TIME = time.monotonic()


def log(message: str) -> None:
    elapsed = time.monotonic() - START_TIME
    print(f"[{elapsed:9.1f}s] {message}", flush=True)


def check_budget() -> None:
    if time.monotonic() - START_TIME > WALL_BUDGET_SECONDS:
        raise TimeoutError("frozen wall budget exceeded")


def swap_species(name: str) -> str:
    flavour = ind._species_flavour(name)
    swapped = _FLAVOUR_SWAP[flavour]
    if swapped == flavour:
        return name
    return ind._species(swapped, ind._is_antineutrino(name))


def rel_gap(left: np.ndarray, right: np.ndarray, floor: float) -> float:
    left = np.asarray(left, dtype=np.float64)
    right = np.asarray(right, dtype=np.float64)
    scale = max(float(np.max(np.abs(left))), float(np.max(np.abs(right))), floor)
    return float(np.max(np.abs(left - right)) / scale)


def pair_total_rows(total: np.ndarray) -> np.ndarray:
    total = np.asarray(total, dtype=np.float64)
    return 0.5 * np.stack(
        (total[0] + total[1], total[2] + total[3], total[4] + total[5])
    )


def state_from_scales(grid, scales) -> np.ndarray:
    return ind.pair_logits_to_cloglog(
        np.stack([-grid.nodes / s for s in scales])
    )


def p_state(state: np.ndarray) -> np.ndarray:
    return np.asarray(state)[[0, 2, 1]]


def self_event_permutation(events):
    """Classify the catalogue under the mu/tau flavour swap.

    Returns ``(perm, exchange_indices)``. For leg-order-closed events, ``perm``
    maps event e to the catalogue index carrying the swapped legs (bitwise
    equivariance class). The four ``{mu, tau}`` distinct-elastic events swap to
    a leg order that is NOT in the catalogue: their equivariance runs through
    the exact (1<->2, 3<->4) exchange bijection of the tensor quadrature
    ((alpha,beta) node exchange, final-polar flip, azimuth half-turn), which
    agrees at real-arithmetic level but only to fp roundoff in float64 — they
    are certified separately (N7X) and mapped to themselves here.
    """

    lookup = {
        (event.legs, event.category, event.kernel, event.coefficient): index
        for index, event in enumerate(events)
    }
    perm = []
    exchange = []
    for index, event in enumerate(events):
        swapped = tuple(swap_species(leg) for leg in event.legs)
        key = (swapped, event.category, event.kernel, event.coefficient)
        if key in lookup:
            perm.append(lookup[key])
            continue
        exchanged = (swapped[1], swapped[0], swapped[3], swapped[2])
        key = (exchanged, event.category, event.kernel, event.coefficient)
        if key not in lookup:
            raise RuntimeError(f"self catalogue is not P-closed at {event}")
        # The swap lands on the (1<->2, 3<->4)-exchanged representative: the
        # same-sign {mu,tau} events partner with themselves, the opposite-sign
        # ones with each other.
        perm.append(lookup[key])
        exchange.append(index)
    if sorted(perm) != list(range(len(events))):
        raise RuntimeError("self event P-map is not a bijection")
    flavours_seen = {
        frozenset(ind._species_flavour(leg) for leg in events[i].legs)
        for i in exchange
    }
    if exchange and flavours_seen != {frozenset({"mu", "tau"})}:
        raise RuntimeError("unexpected exchange-class events")
    for index in exchange:
        if perm[index] not in exchange:
            raise RuntimeError("exchange partner must itself be exchange-class")
    return tuple(perm), tuple(exchange)


def electron_event_permutation(events) -> tuple[int, ...]:
    lookup = {
        (event.target, event.category): index for index, event in enumerate(events)
    }
    perm = []
    for event in events:
        key = (swap_species(event.target), event.category)
        if key not in lookup:
            raise RuntimeError(f"electron catalogue is not P-closed at {event}")
        perm.append(lookup[key])
    if sorted(perm) != list(range(len(events))):
        raise RuntimeError("electron event P-map is not a bijection")
    return tuple(perm)


def abs_modal_product(rates_abs, y, grid):
    basis_abs = np.abs(grid.modal_basis(y))
    out = np.zeros((rates_abs.shape[0], grid.order), dtype=np.float64)
    for start in range(0, rates_abs.shape[1], 4096):
        stop = start + 4096
        out += rates_abs[:, start:stop] @ basis_abs[start:stop]
    return out


def paired_self_shadow(grid, spectra_f, spectra_p, temperature, config,
                       capture_node=None):
    """One node walk computing f and Pf self assemblies together.

    Returns signed/absolute shadows and addend counts for BOTH runs, the
    bitwise rate-equivariance verdict (N7), and — when ``capture_node`` is
    set — the frozen-node raw data needed by the M-INT-2 interval replay.
    """

    events = ind.independent_self_events()
    perm, exchange = self_event_permutation(events)
    exchange_set = set(exchange)
    n_events = len(events)
    signed = {"f": np.zeros((6, grid.order)), "p": np.zeros((6, grid.order))}
    shadow = {"f": np.zeros((6, grid.order)), "p": np.zeros((6, grid.order))}
    addends = 0
    bitwise_ok = True
    max_leg_diff = 0.0
    captured = None
    n7x_rates = {}
    n7x_masks = {}
    subset = list(N7X_NODES)
    p2_nodes, p2_weights = temperature * grid.nodes, temperature * grid.weights
    angular_size = config.incoming_polar_order * config.final_polar_order * 4
    spectra = {"f": spectra_f, "p": spectra_p}
    basis = spectra_f.native_basis
    basis_abs = np.abs(basis)

    for node_index, y1 in enumerate(grid.nodes):
        check_budget()
        p1 = temperature * float(y1)
        outer = temperature**3 * grid.weights[node_index] * y1**2 / ind.TWO_PI_SQUARED
        batch = ind._two_body_kinematics(
            p1=p1, p2_nodes=p2_nodes, p2_weights=p2_weights,
            mass2=0.0, mass3=0.0, mass4=0.0, config=config,
        )
        y3 = batch.p3_magnitude / temperature
        y4 = batch.p4_magnitude / temperature
        domain = (
            batch.support
            & (y3 > 0.0) & (y3 < grid.y_max)
            & (y4 > 0.0) & (y4 < grid.y_max)
        )
        mask, base = domain.ravel(), ind._event_measure(batch, p1, outer, domain)
        y3_valid, y4_valid = y3[domain], y4[domain]
        addends += int(np.count_nonzero(mask)) * n_events
        matrix_cache = {}
        # Off-grid interpolants per flavour/leg (N7 leg comparison + reuse).
        legs3 = {}
        legs4 = {}
        for run in ("f", "p"):
            for flavour in ind.FLAVOURS:
                species = ind._species(flavour, False)
                legs3[(run, flavour)] = ind._strict_interpolated_logit(
                    spectra[run].at(species, y3_valid)
                )
                legs4[(run, flavour)] = ind._strict_interpolated_logit(
                    spectra[run].at(species, y4_valid)
                )
        for flavour in ind.FLAVOURS:
            partner = _FLAVOUR_SWAP[flavour]
            for legs in (legs3, legs4):
                diff = float(np.max(np.abs(
                    legs[("p", flavour)] - legs[("f", partner)]
                ), initial=0.0))
                max_leg_diff = max(max_leg_diff, diff)
                if diff != 0.0:
                    bitwise_ok = False
        rates = {run: np.zeros((n_events, batch.support.size)) for run in ("f", "p")}
        for event_index, event in enumerate(events):
            s1, s2, s3, s4 = event.legs
            key = (event.kernel, event.coefficient)
            matrix_cache.setdefault(key, ind._self_matrix(event, batch, config))
            matrix, _count, _corr = matrix_cache[key]
            for run in ("f", "p"):
                u1 = float(spectra[run].native(s1)[node_index])
                u2 = np.repeat(spectra[run].native(s2), angular_size)[mask]
                u3 = legs3[(run, ind._species_flavour(s3))]
                u4 = legs4[(run, ind._species_flavour(s4))]
                rate = base * matrix[domain] * ind._stable_pauli_gain_minus_loss(
                    u1, u2, u3, u4
                )
                rates[run][event_index, mask] = rate
        # N7: bitwise per-event rate equivariance (leg-order-closed events).
        for event_index in range(n_events):
            if event_index in exchange_set:
                continue
            if not np.array_equal(rates["p"][perm[event_index]], rates["f"][event_index]):
                bitwise_ok = False
        # N7X data: exchange-class rate slices on the frozen node subset.
        if node_index in N7X_NODES:
            shape = batch.support.shape
            n7x_masks[node_index] = domain[subset].copy()
            for event_index in exchange:
                for run in ("f", "p"):
                    n7x_rates[(run, event_index, node_index)] = (
                        rates[run][event_index].reshape(shape)[subset].copy()
                    )
        for run in ("f", "p"):
            run_rates = rates[run]
            rates_abs = np.abs(run_rates)
            sums = np.sum(run_rates, axis=1, dtype=np.float64)
            sums_abs = np.sum(rates_abs, axis=1, dtype=np.float64)
            leg_modes = (
                sums[:, None] * basis[node_index],
                run_rates.reshape(n_events, grid.order, angular_size).sum(axis=2) @ basis,
                ind._modal_product(run_rates[:, mask], y3_valid, grid),
                ind._modal_product(run_rates[:, mask], y4_valid, grid),
            )
            leg_modes_abs = (
                sums_abs[:, None] * basis_abs[node_index],
                rates_abs.reshape(n_events, grid.order, angular_size).sum(axis=2) @ basis_abs,
                abs_modal_product(rates_abs[:, mask], y3_valid, grid),
                abs_modal_product(rates_abs[:, mask], y4_valid, grid),
            )
            for event_index, event in enumerate(events):
                for sign, species, values, values_abs in zip(
                    (1.0, 1.0, -1.0, -1.0), event.legs, leg_modes, leg_modes_abs
                ):
                    index = ind.SPECIES_INDEX[species]
                    signed[run][index] += sign * values[event_index]
                    shadow[run][index] += values_abs[event_index]
        if capture_node is not None and node_index == capture_node:
            captured = {
                "node_index": node_index,
                "p1": p1,
                "outer": outer,
                "domain": domain.copy(),
                "mask": mask.copy(),
                "base": base.copy(),
                "y3_valid": y3_valid.copy(),
                "y4_valid": y4_valid.copy(),
                "rates_f": rates["f"].copy(),
                "rates_p": rates["p"].copy(),
                "matrix_cache": {
                    key: value[0][domain].copy() for key, value in matrix_cache.items()
                },
                "batch_domain": {
                    "p2": batch.p2[domain].copy(),
                    "e2": batch.e2[domain].copy(),
                    "e3": batch.e3[domain].copy(),
                    "e4": batch.e4[domain].copy(),
                    "p3": batch.p3_magnitude[domain].copy(),
                    "p4": batch.p4_magnitude[domain].copy(),
                    "d12": batch.d12[domain].copy(),
                    "d13": batch.d13[domain].copy(),
                    "d14": batch.d14[domain].copy(),
                    "d23": batch.d23[domain].copy(),
                    "d24": batch.d24[domain].copy(),
                    "d34": batch.d34[domain].copy(),
                    "quadrature_weight": batch.quadrature_weight[domain].copy(),
                },
            }
    addends += grid.order + n_events
    return {
        "signed": signed, "shadow": shadow, "addends": addends,
        "bitwise_ok": bitwise_ok, "max_leg_diff": max_leg_diff,
        "captured": captured,
        "n7x_rates": n7x_rates, "n7x_masks": n7x_masks, "exchange": exchange,
        "perm": perm,
    }


def exchange_class_check(shadow_self):
    """N7X: exchange-bijection agreement for the {mu,tau} elastic events.

    rate_Pf[e](alpha_node, beta_p2, i12, i*, iphi) must agree with
    rate_f[e](beta_node, alpha_p2, i12, flip(i*), iphi+2 mod 4) to fp
    roundoff on the frozen node-pair subset; boundary support flips are
    excluded and counted.
    """

    rates = shadow_self["n7x_rates"]
    masks = shadow_self["n7x_masks"]
    exchange = shadow_self["exchange"]
    perm = shadow_self["perm"]
    subset = list(N7X_NODES)
    sup_rel = 0.0
    mask_mismatches = 0
    compared = 0
    for event_index in exchange:
        partner = perm[event_index]
        denom = max(
            max(float(np.max(np.abs(rates[("f", partner, node)])))
                for node in subset),
            TINY,
        )
        for a_pos, alpha in enumerate(subset):
            for b_pos, beta in enumerate(subset):
                lhs = rates[("p", event_index, alpha)][b_pos]
                # Exchange bijection: final-polar flip only. The frame change is
                # an about-y rotation composed with an x-reflection; the rates
                # are even in the azimuth (sin phi enters squared), so phi maps
                # to itself index-wise (phi -> phi + pi would flip cos phi and
                # is wrong for the K_t cross-partner comparison).
                rhs = rates[("f", partner, beta)][a_pos][:, ::-1, :]
                lhs_mask = masks[alpha][b_pos]
                rhs_mask = masks[beta][a_pos][:, ::-1, :]
                both = lhs_mask & rhs_mask
                mask_mismatches += int(np.count_nonzero(lhs_mask ^ rhs_mask))
                compared += int(np.count_nonzero(both))
                if np.any(both):
                    sup_rel = max(
                        sup_rel,
                        float(np.max(np.abs(lhs[both] - rhs[both])) / denom),
                    )
    return {
        "sup_rel": sup_rel,
        "mask_mismatches": mask_mismatches,
        "compared": compared,
        "ok": bool(sup_rel <= N7X_TOL
                   and mask_mismatches <= N7X_MAX_MASK_MISMATCH),
    }


def paired_electron_shadow(grid, spectra_f, spectra_p, temperature_cm,
                           temperature_gamma, electron_mass, config):
    """One node walk computing f and Pf electron assemblies together."""

    events = ind.independent_electron_events()
    perm = electron_event_permutation(events)
    elastic, pairs = events[:12], events[12:]
    signed = {"f": np.zeros((6, grid.order)), "p": np.zeros((6, grid.order))}
    shadow = {"f": np.zeros((6, grid.order)), "p": np.zeros((6, grid.order))}
    addends = 0
    bitwise_ok = True
    electron_p2, electron_weights = ind._electron_half_line_rule(
        config.electron_radial_order, temperature_gamma
    )
    neutrino_p2 = temperature_cm * grid.nodes
    neutrino_weights = temperature_cm * grid.weights
    angular_size = config.incoming_polar_order * config.final_polar_order * 4
    spectra = {"f": spectra_f, "p": spectra_p}
    basis = spectra_f.native_basis
    basis_abs = np.abs(basis)

    for node_index, y1 in enumerate(grid.nodes):
        check_budget()
        p1 = temperature_cm * float(y1)
        outer = temperature_cm**3 * grid.weights[node_index] * y1**2 / ind.TWO_PI_SQUARED
        # -- elastic block --
        batch = ind._two_body_kinematics(
            p1=p1, p2_nodes=electron_p2, p2_weights=electron_weights,
            mass2=electron_mass, mass3=0.0, mass4=electron_mass, config=config,
        )
        y3 = batch.p3_magnitude / temperature_cm
        domain = batch.support & (y3 > 0.0) & (y3 < grid.y_max)
        mask = domain.ravel()
        base = ind._event_measure(batch, p1, outer, domain)
        y3_valid = y3[domain]
        addends += int(np.count_nonzero(mask)) * len(elastic)
        u2 = -batch.e2[domain] / temperature_gamma
        u4 = -batch.e4[domain] / temperature_gamma
        rates = {run: np.zeros((len(elastic), batch.support.size)) for run in ("f", "p")}
        for event_index, event in enumerate(elastic):
            matrix, _count, _corr = ind._electron_matrix(
                event.target, event.category, batch, electron_mass, config
            )
            for run in ("f", "p"):
                u1 = float(spectra[run].native(event.target)[node_index])
                u3 = spectra[run].at(event.target, y3_valid)
                ind._strict_interpolated_logit(u3)
                rate = base * matrix[domain] * ind._stable_pauli_gain_minus_loss(
                    u1, u2, u3, u4
                )
                rates[run][event_index, mask] = rate
        for event_index in range(len(elastic)):
            if not np.array_equal(rates["p"][perm[event_index]], rates["f"][event_index]):
                bitwise_ok = False
        for run in ("f", "p"):
            run_rates = rates[run]
            rates_abs = np.abs(run_rates)
            incoming = np.sum(run_rates, axis=1, dtype=np.float64)[:, None] * basis[node_index]
            outgoing = ind._modal_product(run_rates[:, mask], y3_valid, grid)
            incoming_abs = np.sum(rates_abs, axis=1, dtype=np.float64)[:, None] * basis_abs[node_index]
            outgoing_abs = abs_modal_product(rates_abs[:, mask], y3_valid, grid)
            for event_index, event in enumerate(elastic):
                index = ind.SPECIES_INDEX[event.target]
                signed[run][index] += incoming[event_index] - outgoing[event_index]
                shadow[run][index] += incoming_abs[event_index] + outgoing_abs[event_index]
        # -- pair block --
        batch = ind._two_body_kinematics(
            p1=p1, p2_nodes=neutrino_p2, p2_weights=neutrino_weights,
            mass2=0.0, mass3=electron_mass, mass4=electron_mass, config=config,
        )
        domain, mask = batch.support, batch.support.ravel()
        base = ind._event_measure(batch, p1, outer, domain)
        addends += int(np.count_nonzero(mask)) * len(pairs)
        u3 = -batch.e3[domain] / temperature_gamma
        u4 = -batch.e4[domain] / temperature_gamma
        rates = {run: np.zeros((len(pairs), batch.support.size)) for run in ("f", "p")}
        pair_perm = [perm[12 + i] - 12 for i in range(len(pairs))]
        for event_index, event in enumerate(pairs):
            partner = ind._cp_partner(event.target)
            matrix, _count, _corr = ind._electron_matrix(
                event.target, "pair", batch, electron_mass, config
            )
            for run in ("f", "p"):
                u1 = float(spectra[run].native(event.target)[node_index])
                u2 = np.repeat(spectra[run].native(partner), angular_size)[mask]
                rate = base * matrix[domain] * ind._stable_pauli_gain_minus_loss(
                    u1, u2, u3, u4
                )
                rates[run][event_index, mask] = rate
        for event_index in range(len(pairs)):
            if not np.array_equal(rates["p"][pair_perm[event_index]], rates["f"][event_index]):
                bitwise_ok = False
        for run in ("f", "p"):
            run_rates = rates[run]
            rates_abs = np.abs(run_rates)
            incoming_1 = np.sum(run_rates, axis=1, dtype=np.float64)[:, None] * basis[node_index]
            incoming_2 = run_rates.reshape(len(pairs), grid.order, angular_size).sum(axis=2) @ basis
            incoming_1_abs = np.sum(rates_abs, axis=1, dtype=np.float64)[:, None] * basis_abs[node_index]
            incoming_2_abs = rates_abs.reshape(len(pairs), grid.order, angular_size).sum(axis=2) @ basis_abs
            for event_index, event in enumerate(pairs):
                for species, contribution, contribution_abs in (
                    (event.target, incoming_1[event_index], incoming_1_abs[event_index]),
                    (ind._cp_partner(event.target), incoming_2[event_index], incoming_2_abs[event_index]),
                ):
                    index = ind.SPECIES_INDEX[species]
                    signed[run][index] += contribution
                    shadow[run][index] += contribution_abs
    return {
        "signed": signed, "shadow": shadow, "addends": addends,
        "bitwise_ok": bitwise_ok,
    }


def gamma_factor(count: int) -> float:
    x = count * EPS
    if x >= 1.0:
        raise RuntimeError("Higham gamma factor is undefined for this count")
    return x / (1.0 - x)


def native_map_terms_r3(grid, modal, temperature):
    """A_cond and the gamma-50 map envelope for ONE sector map (production graph)."""

    basis = grid.modal_basis(grid.nodes)
    prefactor = temperature**3 / ind.TWO_PI_SQUARED
    scale = prefactor * np.square(grid.nodes)
    a_cond = float(np.max(np.sum(np.abs(basis), axis=1) / scale))
    abs_map = (np.abs(np.asarray(modal)) @ np.abs(basis).T) / scale[None, :]
    b_map = float(gamma_factor(50) * np.max(abs_map))
    native64 = (np.asarray(modal) @ basis.T) / (prefactor * np.square(grid.nodes)[None, :])
    return a_cond, b_map, np.asarray(native64, dtype=np.float64)


# --- interval machinery ------------------------------------------------------


def iv_context(prec_bits: int):
    ctx = mpmath.iv
    ctx.prec = prec_bits
    return ctx


def iv_matrix_map(ctx, modal, basis, scale_nodes):
    """Interval replay of one native map: (modal @ basis.T) / scale."""

    rows = len(modal)
    order = len(basis)
    out = [[None] * order for _ in range(rows)]
    for j in range(rows):
        row = [ctx.mpf(float(v)) for v in modal[j]]
        for k in range(order):
            acc = ctx.mpf(0)
            bk = basis[k]
            for m in range(order):
                acc += row[m] * bk[m]
            out[j][k] = acc / scale_nodes[k]
    return out


def iv_sup_abs_diff(matrix_iv, matrix64):
    """sup over elements of |float64 - interval| upper bound."""

    sup = mpmath.mpf(0)
    for j, row in enumerate(matrix_iv):
        for k, cell in enumerate(row):
            diff = abs(mpmath.iv.mpf(float(matrix64[j][k])) - cell)
            sup = max(sup, mpmath.mpf(diff.b))
    return float(sup)


def iv_pair_rows(native_iv, ctx):
    half = ctx.mpf(0.5)
    return [
        [half * (native_iv[2 * i][k] + native_iv[2 * i + 1][k])
         for k in range(len(native_iv[0]))]
        for i in range(3)
    ]


def iv_map_replay(grid, temperature, modal_self, modal_elec, native64_sum,
                  d_cov, prec_bits):
    """M-INT-1: interval replay of the production two-map + add + pair stage."""

    ctx = iv_context(prec_bits)
    basis64 = grid.modal_basis(grid.nodes)
    # basis[k][m] = value of mode m at node k (exact float64 constants).
    basis = [[ctx.mpf(float(basis64[k, m])) for m in range(grid.order)]
             for k in range(grid.order)]
    t_iv = ctx.mpf(float(temperature))
    two_pi_sq = ctx.mpf(float(ind.TWO_PI_SQUARED))
    prefactor = (t_iv ** 3) / two_pi_sq
    scale_nodes = [prefactor * ctx.mpf(float(y)) ** 2 for y in grid.nodes]

    native_self_iv = iv_matrix_map(ctx, modal_self, basis, scale_nodes)
    native_elec_iv = iv_matrix_map(ctx, modal_elec, basis, scale_nodes)
    total_iv = [
        [native_self_iv[j][k] + native_elec_iv[j][k] for k in range(grid.order)]
        for j in range(6)
    ]
    prefactor64 = temperature**3 / ind.TWO_PI_SQUARED
    scale64 = prefactor64 * np.square(grid.nodes)
    native_self_64 = (np.asarray(modal_self) @ basis64.T) / scale64[None, :]
    native_elec_64 = (np.asarray(modal_elec) @ basis64.T) / scale64[None, :]
    w_int_self = iv_sup_abs_diff(native_self_iv, native_self_64) / d_cov
    w_int_elec = iv_sup_abs_diff(native_elec_iv, native_elec_64) / d_cov
    w_int_total = iv_sup_abs_diff(total_iv, np.asarray(native64_sum)) / d_cov
    pt_iv = iv_pair_rows(total_iv, ctx)
    return {
        "pt_iv": pt_iv,
        "W_int_self": float(w_int_self),
        "W_int_elec": float(w_int_elec),
        "W_int_total": float(w_int_total),
    }


def iv_certified_residual(pt_iv_f, pt_iv_p):
    """Interval of max-row sup |pt(Pf)[sigma] - pt(f)| (numerator of R3)."""

    lo = mpmath.mpf(0)
    hi = mpmath.mpf(0)
    for row_f, row_p in zip(range(3), PAIR_PERM):
        for k in range(len(pt_iv_f[0])):
            diff = abs(pt_iv_p[row_p][k] - pt_iv_f[row_f][k])
            hi = max(hi, mpmath.mpf(diff.b))
            lo = max(lo, mpmath.mpf(diff.a))
    return float(lo), float(hi)


def iv_log_expit(ctx, x):
    return -ctx.log(1 + ctx.exp(-x))


def iv_pauli(ctx, u1, u2, u3, u4, positive_branch):
    """Interval Pauli gain-minus-loss; also returns exp(log_loss) for pads."""

    affinity = u3 + u4 - u1 - u2
    log_loss = (
        iv_log_expit(ctx, u1) + iv_log_expit(ctx, u2)
        + iv_log_expit(ctx, -u3) + iv_log_expit(ctx, -u4)
    )
    loss = ctx.exp(log_loss)
    if positive_branch:
        return loss * ctx.exp(affinity) * -(ctx.exp(-affinity) - 1), loss
    return loss * (ctx.exp(affinity) - 1), loss


def within_padded(value: float, interval, pad_abs: float) -> bool:
    lo = mpmath.mpf(interval.a)
    hi = mpmath.mpf(interval.b)
    pad = mpmath.mpf(pad_abs)
    return (lo - pad) <= mpmath.mpf(value) <= (hi + pad)


def iv_mid(interval) -> float:
    return float(mpmath.mpf(interval.mid))


def iv_deep_node_replay(grid, config, temperature, states, captured, prec_bits,
                        widen_rel, sample_cap=None):
    """M-INT-2: full interval replay of the self assembly at one frozen node.

    Certifies (a) containment of the float64 kinematic invariants, measure,
    off-grid interpolants, and per-event rates within outward intervals, and
    (b) interval identity of the f and Pf rate replays under the catalogue
    P-bijection. The float64 branch decisions (domain mask, Pauli branch,
    roundoff clipping) are adopted from the captured production data.
    """

    ctx = iv_context(prec_bits)
    events = ind.independent_self_events()
    perm, exchange = self_event_permutation(events)
    exchange_set = set(exchange)
    dom = captured["batch_domain"]
    n_valid = len(captured["base"])
    take = n_valid if sample_cap is None else min(sample_cap, n_valid)
    log(f"M-INT-2: replaying {take}/{n_valid} valid samples at node {captured['node_index']}")

    # Recover per-sample (p2, mu12, mu_star, phi) and the p2 node index from
    # the domain mask layout; verify against the captured batch bitwise.
    incoming_mu, incoming_w, final_mu, final_w, azimuth, azimuth_w = ind._angular_rule(
        config.incoming_polar_order, config.final_polar_order,
        config.final_azimuth_order,
    )
    p2_grid, mu12_grid, mustar_grid, phi_grid = np.meshgrid(
        temperature * grid.nodes, incoming_mu, final_mu, azimuth, indexing="ij"
    )
    w2_grid, w12_grid, wstar_grid, wphi_grid = np.meshgrid(
        temperature * grid.weights, incoming_w, final_w, azimuth_w, indexing="ij"
    )
    p2_index_grid = np.meshgrid(
        np.arange(grid.order), np.arange(len(incoming_mu)),
        np.arange(len(final_mu)), np.arange(len(azimuth)), indexing="ij"
    )[0]
    domain = captured["domain"]
    samples = {
        "p2": p2_grid[domain][:take], "mu12": mu12_grid[domain][:take],
        "mustar": mustar_grid[domain][:take], "phi": phi_grid[domain][:take],
        "w": (w2_grid * w12_grid * wstar_grid * wphi_grid)[domain][:take],
        "p2_index": p2_index_grid[domain][:take],
    }
    if not np.array_equal(samples["p2"], dom["p2"][:take]):
        raise RuntimeError("M-INT-2 sample layout does not match the captured batch")
    if not np.array_equal(samples["w"], dom["quadrature_weight"][:take]):
        raise RuntimeError("M-INT-2 weight layout does not match the captured batch")
    p1 = ctx.mpf(float(captured["p1"]))
    outer = ctx.mpf(float(captured["outer"]))
    pi_iv = ctx.mpf(float(ind.PI))
    y_max = ctx.mpf(float(grid.y_max))
    t_iv = ctx.mpf(float(temperature))
    gf2 = ctx.mpf(float(ind.G_F_MEV_MINUS_2)) ** 2
    scales_iv = [ctx.sqrt((2 * ctx.mpf(m) + 1) / y_max) for m in range(grid.order)]
    coeff_iv = {}
    coeff64_abs = {}
    natives = {}
    for run, state in states.items():
        spectra = ind._SpectralLogits(grid, ind._native_pair_logits(state))
        coeff_iv[run] = {
            flavour: [ctx.mpf(float(v)) for v in spectra.coefficients[ind.PAIR_INDEX[flavour]]]
            for flavour in ind.FLAVOURS
        }
        coeff64_abs[run] = {
            flavour: np.abs(np.asarray(
                spectra.coefficients[ind.PAIR_INDEX[flavour]], dtype=np.float64
            ))
            for flavour in ind.FLAVOURS
        }
        natives[run] = {
            flavour: [float(v) for v in spectra.values[ind.PAIR_INDEX[flavour]]]
            for flavour in ind.FLAVOURS
        }

    containment_failures = 0
    identity_failures = 0
    checked = 0
    max_dev_rel = 0.0

    def record_dev(value: float, interval) -> None:
        nonlocal max_dev_rel
        mid = float(mpmath.mpf(interval.mid))
        scale = max(abs(mid), 1e-300)
        max_dev_rel = max(max_dev_rel, abs(value - mid) / scale)

    def basis_row_iv(y_iv):
        x = 2 * y_iv / y_max - 1
        values = [ctx.mpf(1), x]
        for degree in range(2, grid.order):
            values.append(
                ((2 * degree - 1) * x * values[degree - 1]
                 - (degree - 1) * values[degree - 2]) / degree
            )
        return [values[m] * scales_iv[m] for m in range(grid.order)]

    contraction_iv = {
        run: {leg: {mode: ctx.mpf(0) for mode in INT2_MODES} for leg in (3, 4)}
        for run in ("f", "p")
    }
    contraction_64 = {
        run: {leg: {mode: 0.0 for mode in INT2_MODES} for leg in (3, 4)}
        for run in ("f", "p")
    }
    matrix_keys = {
        (event.kernel, event.coefficient): None for event in events
    }

    for sample in range(take):
        check_budget()
        if sample and sample % 2000 == 0:
            log(f"M-INT-2 sample {sample}/{take}")
        p2 = ctx.mpf(float(samples["p2"][sample]))
        mu12 = ctx.mpf(float(samples["mu12"][sample]))
        mustar = ctx.mpf(float(samples["mustar"][sample]))
        phi = ctx.mpf(float(samples["phi"][sample]))
        w_quad = ctx.mpf(float(samples["w"][sample]))
        e2 = p2
        sin12 = ctx.sqrt(max(ctx.mpf(0), 1 - mu12 ** 2))
        tv_x = p2 * sin12
        tv_z = p1 + p2 * mu12
        total_energy = p1 + e2
        total_mag = ctx.sqrt(tv_x ** 2 + tv_z ** 2)
        s = total_energy ** 2 - total_mag ** 2
        sqrt_s = ctx.sqrt(s)
        k_star = ctx.sqrt(s ** 2) / (2 * sqrt_s)
        e3_star = s / (2 * sqrt_s)
        beta = total_mag / total_energy
        gamma = total_energy / sqrt_s
        par_x, par_z = tv_x / total_mag, tv_z / total_mag
        trans_x_x, trans_x_z = par_z, -par_x
        sin_star = ctx.sqrt(max(ctx.mpf(0), 1 - mustar ** 2))
        t_coeff_x = k_star * sin_star * ctx.cos(phi)
        # transverse_y contributes only to the y-component; |p| picks it up below
        t_coeff_y = k_star * sin_star * ctx.sin(phi)
        p3_par = gamma * (k_star * mustar + beta * e3_star)
        p3_x = t_coeff_x * trans_x_x + p3_par * par_x
        p3_y = t_coeff_y
        p3_z = t_coeff_x * trans_x_z + p3_par * par_z
        e3 = gamma * (e3_star + beta * k_star * mustar)
        e4 = total_energy - e3
        p4_x, p4_y, p4_z = tv_x - p3_x, -p3_y, tv_z - p3_z
        p3_mag = ctx.sqrt(p3_x ** 2 + p3_y ** 2 + p3_z ** 2)
        p4_mag = ctx.sqrt(p4_x ** 2 + p4_y ** 2 + p4_z ** 2)
        d12 = p1 * e2 - p1 * (p2 * mu12)
        d13 = p1 * e3 - p1 * p3_z
        d14 = p1 * e4 - p1 * p4_z
        d23 = e2 * e3 - (tv_x * p3_x + tv_z * p3_z - p1 * p3_z)
        d24 = e2 * e4 - (tv_x * p4_x + tv_z * p4_z - p1 * p4_z)
        d34 = e3 * e4 - (p3_x * p4_x + p3_y * p4_y + p3_z * p4_z)
        phase = k_star / sqrt_s
        measure = outer * w_quad * p2 ** 2 * phase / (e2 * 256 * pi_iv ** 4 * p1)

        # Error-model containment pads (frozen): relative term + a
        # cancellation-aware absolute term scaled to each quantity's
        # pre-cancellation magnitude (float64's own rounding error can be
        # O(1) RELATIVE at cancelled dots / near-zero Pauli affinities while
        # staying tiny on the physical scale).
        te_mid = iv_mid(total_energy)
        e2_mid = iv_mid(e2)
        p1_mid = iv_mid(p1)
        dot_scale = {
            "d12": p1_mid * te_mid, "d13": p1_mid * te_mid,
            "d14": p1_mid * te_mid, "d23": e2_mid * te_mid,
            "d24": e2_mid * te_mid, "d34": te_mid * te_mid,
        }
        for name, value in (
            ("e3", dom["e3"][sample]), ("e4", dom["e4"][sample]),
            ("p3", dom["p3"][sample]), ("p4", dom["p4"][sample]),
            ("d12", dom["d12"][sample]), ("d13", dom["d13"][sample]),
            ("d14", dom["d14"][sample]), ("d23", dom["d23"][sample]),
            ("d24", dom["d24"][sample]), ("d34", dom["d34"][sample]),
        ):
            target = {"e3": e3, "e4": e4, "p3": p3_mag, "p4": p4_mag,
                      "d12": d12, "d13": d13, "d14": d14, "d23": d23,
                      "d24": d24, "d34": d34}[name]
            pad = widen_rel * abs(iv_mid(target))
            if name in dot_scale:
                pad += INT2_K_GEOM * EPS * dot_scale[name]
            checked += 1
            record_dev(float(value), target)
            if not within_padded(float(value), target, pad):
                containment_failures += 1
        checked += 1
        record_dev(float(captured["base"][sample]), measure)
        measure_mid = iv_mid(measure)
        if not within_padded(float(captured["base"][sample]), measure,
                             widen_rel * abs(measure_mid)):
            containment_failures += 1

        y3_iv = p3_mag / t_iv
        y4_iv = p4_mag / t_iv
        basis3 = basis_row_iv(y3_iv)
        basis4 = basis_row_iv(y4_iv)
        basis3_abs = np.asarray([abs(iv_mid(b)) for b in basis3])
        basis4_abs = np.asarray([abs(iv_mid(b)) for b in basis4])
        interp = {}
        absdot = {}
        for run in ("f", "p"):
            for flavour in ind.FLAVOURS:
                coeff = coeff_iv[run][flavour]
                u3 = ctx.mpf(0)
                u4 = ctx.mpf(0)
                for m in range(grid.order):
                    u3 += basis3[m] * coeff[m]
                    u4 += basis4[m] * coeff[m]
                interp[(run, flavour, 3)] = u3
                interp[(run, flavour, 4)] = u4
                coeff_abs = coeff64_abs[run][flavour]
                absdot[(run, flavour, 3)] = float(basis3_abs @ coeff_abs)
                absdot[(run, flavour, 4)] = float(basis4_abs @ coeff_abs)
        gf2_mid = float(ind.G_F_MEV_MINUS_2) ** 2
        kernel_iv = {
            ("K_s", 16.0): 16 * gf2 * (d12 * d34),
            ("K_t", 64.0): 64 * gf2 * (d14 * d23),
            ("K_t", 16.0): 16 * gf2 * (d14 * d23),
        }
        kernel_pad_abs = {}
        for key in matrix_keys:
            coeff_value = key[1]
            if key[0] == "K_s":
                a_mid, b_mid = iv_mid(d12), iv_mid(d34)
                s_a, s_b = dot_scale["d12"], dot_scale["d34"]
            else:
                a_mid, b_mid = iv_mid(d14), iv_mid(d23)
                s_a, s_b = dot_scale["d14"], dot_scale["d23"]
            kernel_pad_abs[key] = (
                coeff_value * gf2_mid * INT2_K_GEOM * EPS
                * (s_a * abs(b_mid) + s_b * abs(a_mid))
            )
            checked += 1
            pad = widen_rel * abs(iv_mid(kernel_iv[key])) + kernel_pad_abs[key]
            if not within_padded(
                float(captured["matrix_cache"][key][sample]), kernel_iv[key], pad
            ):
                containment_failures += 1
        rate_iv_cache = {}
        p2_index = int(samples["p2_index"][sample])
        for event_index, event in enumerate(events):
            s1, s2, s3, s4 = event.legs
            key = (event.kernel, event.coefficient)
            kernel_mid = iv_mid(kernel_iv[key])
            for run in ("f", "p"):
                fl1 = ind._species_flavour(s1)
                fl2 = ind._species_flavour(s2)
                fl3 = ind._species_flavour(s3)
                fl4 = ind._species_flavour(s4)
                u1 = ctx.mpf(natives[run][fl1][captured["node_index"]])
                u2 = ctx.mpf(natives[run][fl2][p2_index])
                u3 = interp[(run, fl3, 3)]
                u4 = interp[(run, fl4, 4)]
                aff64 = (iv_mid(u3) + iv_mid(u4) - iv_mid(u1) - iv_mid(u2))
                pauli, loss = iv_pauli(ctx, u1, u2, u3, u4, aff64 >= 0.0)
                rate = measure * kernel_iv[key] * pauli
                rate_iv_cache[(run, event_index)] = rate
                float_rate = captured[f"rates_{run}"][event_index][captured["mask"]][sample]
                # Pauli pad: float64 u3/u4 dot error (Higham over the
                # absolute basis-coefficient sums) plus native-logit terms,
                # through the O(exp(log_loss)) affinity sensitivity.
                aff_err = (
                    2.0 * gamma_factor(50)
                    * (absdot[(run, fl3, 3)] + absdot[(run, fl4, 4)])
                    + INT2_K_U * EPS * (abs(iv_mid(u1)) + abs(iv_mid(u2)))
                )
                pauli_pad = 4.0 * abs(iv_mid(loss)) * aff_err
                pad = (
                    widen_rel * abs(iv_mid(rate))
                    + abs(measure_mid) * (
                        abs(iv_mid(pauli)) * kernel_pad_abs[key]
                        + abs(kernel_mid) * pauli_pad
                    )
                )
                checked += 1
                record_dev(float(float_rate), rate)
                if not within_padded(float(float_rate), rate, pad):
                    containment_failures += 1
        for event_index in range(len(events)):
            if event_index in exchange_set:
                continue  # exchange-class certified by N7X, not interval identity
            rf = rate_iv_cache[("f", event_index)]
            rp = rate_iv_cache[("p", perm[event_index])]
            if not (mpmath.mpf(rf.a) == mpmath.mpf(rp.a)
                    and mpmath.mpf(rf.b) == mpmath.mpf(rp.b)):
                identity_failures += 1
        for run in ("f", "p"):
            for event_index in range(len(events)):
                rate = rate_iv_cache[(run, event_index)]
                for mode in INT2_MODES:
                    contraction_iv[run][3][mode] += rate * basis3[mode]
                    contraction_iv[run][4][mode] += rate * basis4[mode]

    # float64 reduced contraction over the same samples for containment.
    for run in ("f", "p"):
        rates64 = captured[f"rates_{run}"][:, captured["mask"]][:, :take]
        for leg, y_valid in ((3, captured["y3_valid"][:take]),
                             (4, captured["y4_valid"][:take])):
            basis64 = grid.modal_basis(y_valid)
            summed = rates64.sum(axis=0)
            for mode in INT2_MODES:
                contraction_64[run][leg][mode] = float(
                    np.sum(summed * basis64[:, mode], dtype=np.float64)
                )
    contraction_ok = True
    n_samples_gamma = gamma_factor(max(take * len(events), 2))
    for run in ("f", "p"):
        rates64 = captured[f"rates_{run}"][:, captured["mask"]][:, :take]
        summed64 = rates64.sum(axis=0)
        for leg, y_valid in ((3, captured["y3_valid"][:take]),
                             (4, captured["y4_valid"][:take])):
            basis64 = grid.modal_basis(y_valid)
            for mode in INT2_MODES:
                total_iv = contraction_iv[run][leg][mode]
                value64 = contraction_64[run][leg][mode]
                # Higham pad from the absolute float64 term sum (covers the
                # extra interpolant/Pauli float64 error the point sums carry).
                abs_sum = float(np.sum(np.abs(rates64) * np.abs(basis64[:, mode])[None, :]))
                lo = mpmath.mpf(total_iv.a)
                hi = mpmath.mpf(total_iv.b)
                pad = mpmath.mpf(
                    n_samples_gamma * abs_sum + widen_rel * abs_sum + 1e-280
                )
                if not (lo - pad <= mpmath.mpf(value64) <= hi + pad):
                    contraction_ok = False
    return {
        "samples_checked": int(take),
        "containment_checks": int(checked),
        "containment_failures": int(containment_failures),
        "identity_failures": int(identity_failures),
        "max_dev_rel": float(max_dev_rel),
        "contraction_ok": bool(contraction_ok),
        "ok": bool(containment_failures == 0 and identity_failures == 0
                   and contraction_ok),
    }


# --- mutants -----------------------------------------------------------------


def historical_24_events():
    events = [e for e in ind.independent_self_events()
              if e.category != "pair_conversion"]
    for index, a in enumerate(ind.FLAVOURS):
        for b in ind.FLAVOURS[index + 1:]:
            events.append(ind.IndependentSelfEvent(
                (ind._species(a, False), ind._species(a, True),
                 ind._species(b, False), ind._species(b, True)),
                "pair_conversion", "K_t", 32.0,
            ))
    if len(events) != 24:
        raise RuntimeError("historical catalogue must contain 24 events")
    return tuple(events)


def half_weight_events():
    events = []
    for event in ind.independent_self_events():
        if event.category == "pair_conversion":
            events.append(ind.IndependentSelfEvent(
                event.legs, event.category, event.kernel, 32.0))
        else:
            events.append(event)
    return tuple(events)


def lane_swap_events():
    events = []
    seen_pairs = set()
    for event in ind.independent_self_events():
        if event.category != "pair_conversion":
            events.append(event)
            continue
        pair = frozenset(ind._species_flavour(leg) for leg in event.legs)
        if pair not in seen_pairs:
            seen_pairs.add(pair)
            events.append(event)
        else:
            # Replace the (b,a) orientation member with a lane-swap duplicate of
            # the (a,b) member: incoming pair a, outgoing pair b again.
            a = ind._species_flavour(event.legs[2])
            b = ind._species_flavour(event.legs[0])
            events.append(ind.IndependentSelfEvent(
                (ind._species(a, False), ind._species(a, True),
                 ind._species(b, False), ind._species(b, True)),
                "pair_conversion", event.kernel, event.coefficient,
            ))
    if len(events) != 27:
        raise RuntimeError("lane-swap catalogue must contain 27 events")
    return tuple(events)


def evaluate_state(grid, config, state, temperature_gamma):
    return ind.evaluate_independent_collision_action(
        grid=grid, pair_cloglog=state, temperature_cm_mev=T_CM,
        temperature_gamma_mev=temperature_gamma, config=config,
    )


def run_mutants(grid, config, base_pt_sa1):
    """In-memory mutants with preregistered kills, then a bitwise canary."""

    scales, t_gamma = FAMILY_PAIRS["SA-1"]
    state_f = state_from_scales(grid, scales)
    state_p = p_state(state_f)
    results = {}
    originals = {
        "independent_self_events": ind.independent_self_events,
        "_self_matrix": ind._self_matrix,
        "_stable_pauli_gain_minus_loss": ind._stable_pauli_gain_minus_loss,
    }

    def eval_pair_metrics():
        action_f = evaluate_state(grid, config, state_f, t_gamma)
        action_p = evaluate_state(grid, config, state_p, t_gamma)
        pt_f = pair_total_rows(action_f.total)
        pt_p = pair_total_rows(action_p.total)
        n1 = rel_gap(
            np.asarray(action_p.modal_total)[list(PI_SPECIES)],
            np.asarray(action_f.modal_total), F_WEAK,
        )
        n3 = rel_gap(pt_p[list(PAIR_PERM)], pt_f, F_NATIVE)
        return action_f, n1, n3, pt_f

    try:
        # MUT-24: historical one-orientation catalogue at coefficient 32.
        mutant_events = historical_24_events()
        ind.independent_self_events = lambda: mutant_events
        _a, n1, n3, _pt = eval_pair_metrics()
        results["MUT-24"] = {
            "N1": n1, "N3": n3,
            "killed": bool(n1 >= KILL_EQUIVARIANCE and n3 >= KILL_EQUIVARIANCE),
            "reason": "one-orientation catalogue breaks N1/N3 equivariance",
        }
        ind.independent_self_events = originals["independent_self_events"]
        log(f"MUT-24 N1={n1:.3e} N3={n3:.3e} killed={results['MUT-24']['killed']}")

        # MUT-HALF: both orientations kept at doubled coefficient 32.
        mutant_events = half_weight_events()
        ind.independent_self_events = lambda: mutant_events
        action_f = evaluate_state(grid, config, state_f, t_gamma)
        deviation = rel_gap(pair_total_rows(action_f.total), base_pt_sa1, F_NATIVE)
        results["MUT-HALF"] = {
            "pair_total_deviation": deviation,
            "killed": bool(deviation >= KILL_HALF_DEVIATION),
            "reason": "doubled pair-conversion weight shifts the SA-1 action anchor",
        }
        ind.independent_self_events = originals["independent_self_events"]
        log(f"MUT-HALF dev={deviation:.3e} killed={results['MUT-HALF']['killed']}")

        # MUT-SIGN: negate the self kernel matrix post-guard.
        def negated_self_matrix(event, batch, config_):
            matrix, count, corr = originals["_self_matrix"](event, batch, config_)
            return -matrix, count, corr

        ind._self_matrix = negated_self_matrix
        action_f = evaluate_state(grid, config, state_f, t_gamma)
        deviation = rel_gap(pair_total_rows(action_f.total), base_pt_sa1, F_NATIVE)
        entropy = float(action_f.diagnostics["entropy_production"])
        results["MUT-SIGN"] = {
            "pair_total_deviation": deviation, "entropy_production": entropy,
            "killed": bool(deviation >= KILL_SIGN_DEVIATION),
            "reason": "negated self kernel shifts the SA-1 action anchor",
        }
        ind._self_matrix = originals["_self_matrix"]
        log(f"MUT-SIGN dev={deviation:.3e} killed={results['MUT-SIGN']['killed']}")

        # MUT-LANE: label-only second orientation (duplicate first member).
        mutant_events = lane_swap_events()
        ind.independent_self_events = lambda: mutant_events
        _a, n1, n3, _pt = eval_pair_metrics()
        results["MUT-LANE"] = {
            "N1": n1, "N3": n3,
            "killed": bool(n1 >= KILL_EQUIVARIANCE and n3 >= KILL_EQUIVARIANCE),
            "reason": "label-only lane swap reproduces the one-orientation defect",
        }
        ind.independent_self_events = originals["independent_self_events"]
        log(f"MUT-LANE N1={n1:.3e} N3={n3:.3e} killed={results['MUT-LANE']['killed']}")

        # MUT-GML: global gain/loss flip.
        def negated_pauli(u1, u2, u3, u4):
            return -originals["_stable_pauli_gain_minus_loss"](u1, u2, u3, u4)

        ind._stable_pauli_gain_minus_loss = negated_pauli
        action_f = evaluate_state(grid, config, state_f, t_gamma)
        entropy = float(action_f.diagnostics["entropy_production"])
        results["MUT-GML"] = {
            "entropy_production": entropy,
            "killed": bool(entropy <= 0.0),
            "reason": "gain/loss flip makes entropy production nonpositive",
        }
        ind._stable_pauli_gain_minus_loss = originals["_stable_pauli_gain_minus_loss"]
        log(f"MUT-GML entropy={entropy:.3e} killed={results['MUT-GML']['killed']}")
    finally:
        ind.independent_self_events = originals["independent_self_events"]
        ind._self_matrix = originals["_self_matrix"]
        ind._stable_pauli_gain_minus_loss = originals["_stable_pauli_gain_minus_loss"]

    # Restoration canary: SA-1(f) pair_total must reproduce the base run bitwise.
    action_f = evaluate_state(grid, config, state_f, t_gamma)
    canary_ok = bool(np.array_equal(pair_total_rows(action_f.total), base_pt_sa1))
    results["canary"] = {"bitwise_equal": canary_ok}
    log(f"mutant restoration canary bitwise_equal={canary_ok}")
    if not canary_ok:
        raise RuntimeError("mutant restoration canary failed")
    return results


# --- member evaluation -------------------------------------------------------


def evaluate_member(name, grid, config, scales, t_gamma, p_fixed, native_cap,
                    int1_prec):
    state_f = state_from_scales(grid, scales)
    state_p = state_f if p_fixed else p_state(state_f)
    if p_fixed and not np.array_equal(state_f[1], state_f[2]):
        raise RuntimeError(f"{name}: P-fixed state has distinct mu/tau rows")
    log(f"{name}: evaluating production action (f)")
    action_f = evaluate_state(grid, config, state_f, t_gamma)
    if p_fixed:
        action_p = action_f
    else:
        log(f"{name}: evaluating production action (Pf)")
        action_p = evaluate_state(grid, config, state_p, t_gamma)

    modal_f = np.asarray(action_f.modal_total)
    modal_p = np.asarray(action_p.modal_total)
    total_f = np.asarray(action_f.total)
    total_p = np.asarray(action_p.total)
    perm = list(PI_SPECIES)
    mass_weight = grid.weights * np.square(grid.nodes)

    n1 = rel_gap(modal_p[perm], modal_f, F_WEAK)
    n2 = rel_gap(mass_weight[None, :] * total_p[perm],
                 mass_weight[None, :] * total_f, F_WEAK)
    pt_f = pair_total_rows(total_f)
    pt_p = pair_total_rows(total_p)
    d_cov = max(float(np.max(np.abs(pt_f))), float(np.max(np.abs(pt_p))), F_NATIVE)
    n3 = float(np.max(np.abs(pt_p[list(PAIR_PERM)] - pt_f)) / d_cov)
    elec_f = np.asarray(action_f.modal_electron)
    elec_p = np.asarray(action_p.modal_electron)
    n4_abs = float(np.max(np.abs(elec_p[perm] - elec_f)))
    n4 = n4_abs / max(float(np.max(np.abs(elec_f[2:6]))), TINY)
    entropy = (float(action_f.diagnostics["entropy_production"]),
               float(action_p.diagnostics["entropy_production"]))
    modal_norm = max(float(np.max(np.abs(modal_f))), float(np.max(np.abs(modal_p))))
    if not (ENV_D_COV[0] <= d_cov <= ENV_D_COV[1]
            and ENV_MODAL[0] <= modal_norm <= ENV_MODAL[1]):
        raise RuntimeError(
            f"{name}: M-ENV violated (D_cov={d_cov:.3e}, modal={modal_norm:.3e})"
        )

    # Paired shadows (N6/N7) and the split-graph bound terms (N9).
    spectra_f = ind._SpectralLogits(grid, ind._native_pair_logits(state_f))
    spectra_p = ind._SpectralLogits(grid, ind._native_pair_logits(state_p))
    capture = INT2_NODE if name in (INT2_PAIR, "DEV") else None
    log(f"{name}: paired self shadow")
    shadow_self = paired_self_shadow(grid, spectra_f, spectra_p, T_CM, config,
                                     capture_node=capture)
    log(f"{name}: paired electron shadow")
    shadow_elec = paired_electron_shadow(
        grid, spectra_f, spectra_p, T_CM, t_gamma, ind.M_ELECTRON_MEV, config
    )
    shadow_match = rel_gap(
        shadow_self["signed"]["f"], np.asarray(action_f.modal_self_interaction),
        F_WEAK,
    )
    g_self = float(np.max(np.abs(
        shadow_self["signed"]["p"][perm] - shadow_self["signed"]["f"]
    )))
    g_elec = float(np.max(np.abs(
        shadow_elec["signed"]["p"][perm] - shadow_elec["signed"]["f"]
    )))
    s_abs_self = max(float(np.max(shadow_self["shadow"]["f"])),
                     float(np.max(shadow_self["shadow"]["p"])))
    s_abs_elec = max(float(np.max(shadow_elec["shadow"]["f"])),
                     float(np.max(shadow_elec["shadow"]["p"])))
    envelope_self = 2.0 * gamma_factor(shadow_self["addends"]) * s_abs_self
    envelope_elec = 2.0 * gamma_factor(shadow_elec["addends"]) * s_abs_elec
    n6_ok = bool(g_self <= envelope_self and g_elec <= envelope_elec)
    n7_ok = bool(shadow_self["bitwise_ok"] and shadow_elec["bitwise_ok"])
    n7x = exchange_class_check(shadow_self)

    modal_self_f = np.asarray(action_f.modal_self_interaction)
    modal_elec_f = np.asarray(action_f.modal_electron)
    modal_self_p = np.asarray(action_p.modal_self_interaction)
    modal_elec_p = np.asarray(action_p.modal_electron)
    a_cond, b_map_self_f, native_self_f = native_map_terms_r3(grid, modal_self_f, T_CM)
    _a2, b_map_elec_f, native_elec_f = native_map_terms_r3(grid, modal_elec_f, T_CM)
    _a3, b_map_self_p, _n1 = native_map_terms_r3(grid, modal_self_p, T_CM)
    _a4, b_map_elec_p, _n2 = native_map_terms_r3(grid, modal_elec_p, T_CM)
    e_add = EPS * float(np.max(np.abs(native_self_f) + np.abs(native_elec_f)))
    e_pair = EPS * float(np.max(0.5 * (np.abs(total_f[::2]) + np.abs(total_f[1::2]))))
    e_split = (2.0 * (max(b_map_self_f, b_map_self_p) + max(b_map_elec_f, b_map_elec_p))
               + 2.0 * e_add + 2.0 * e_pair)
    b_cov = 4.0 * (a_cond * (envelope_self + envelope_elec) + e_split) / d_cov
    n9_ok = bool(n3 <= b_cov)

    # M-INT-1 interval map replay for both runs.
    log(f"{name}: M-INT-1 interval map replay")
    int1_f = iv_map_replay(grid, T_CM, modal_self_f, modal_elec_f, total_f,
                           d_cov, int1_prec)
    int1_p = (int1_f if p_fixed else
              iv_map_replay(grid, T_CM, modal_self_p, modal_elec_p, total_p,
                            d_cov, int1_prec))
    lo, hi = iv_certified_residual(int1_f["pt_iv"], int1_p["pt_iv"])
    slack = (4.0 * (int1_f["W_int_self"] + int1_f["W_int_elec"]
                    + int1_p["W_int_self"] + int1_p["W_int_elec"]) + 16.0 * EPS)
    certified = hi / d_cov
    int1_ok = bool(
        int1_f["W_int_self"] <= CAP_W_INT and int1_f["W_int_elec"] <= CAP_W_INT
        and int1_p["W_int_self"] <= CAP_W_INT and int1_p["W_int_elec"] <= CAP_W_INT
        and certified <= native_cap
        and (lo / d_cov - slack) <= n3 <= (hi / d_cov + slack)
    )

    checks = {
        "N1_weak": bool(n1 <= CAP_DEFAULT),
        "N2_mass": bool(n2 <= CAP_DEFAULT),
        "N3_native": bool(n3 <= native_cap),
        "N4_electron": bool(n4 <= CAP_ELECTRON),
        "N5_entropy": bool(entropy[0] > 0.0 and entropy[1] > 0.0),
        "N6_higham": n6_ok,
        "N7_bitwise": n7_ok,
        "N7X_exchange": bool(n7x["ok"] or p_fixed),
        "N8_shadow_match": bool(shadow_match <= CAP_SHADOW_MATCH),
        "N9_bound": n9_ok,
        "M_INT1": int1_ok,
    }
    record = {
        "scales": list(scales), "temperature_gamma": t_gamma,
        "p_fixed": bool(p_fixed), "native_cap": native_cap,
        "N1": n1, "N2": n2, "N3": n3, "N4": n4, "N4_abs": n4_abs,
        "entropy_production": entropy,
        "D_cov": d_cov, "modal_norm": modal_norm,
        "R_native_module": float(action_f.diagnostics["mu_tau_residual"]),
        "shadow_match": shadow_match,
        "G_self": g_self, "G_elec": g_elec,
        "envelope_self": envelope_self, "envelope_elec": envelope_elec,
        "addends_self": shadow_self["addends"], "addends_elec": shadow_elec["addends"],
        "max_leg_diff": shadow_self["max_leg_diff"],
        "N7X": {"sup_rel": n7x["sup_rel"],
                "mask_mismatches": n7x["mask_mismatches"],
                "compared": n7x["compared"],
                "gated": bool(not p_fixed)},
        "A_cond": a_cond, "E_split": e_split, "B_cov": b_cov,
        "W_int": {
            "self_f": int1_f["W_int_self"], "elec_f": int1_f["W_int_elec"],
            "self_p": int1_p["W_int_self"], "elec_p": int1_p["W_int_elec"],
        },
        "certified_residual": {"lo": lo / d_cov, "hi": certified, "slack": slack},
        "checks": checks,
        "ok": bool(all(checks.values())),
    }
    return record, action_f, pt_f, shadow_self


def main(argv):
    out_path = None
    dev_state = False
    int2_cap = None
    args = list(argv[1:])
    while args:
        arg = args.pop(0)
        if arg == "--out":
            if not args:
                print("ERROR --out requires a path")
                return 20
            out_path = args.pop(0)
        elif arg == "--dev-state":
            dev_state = True
        elif arg == "--int2-cap":
            if not args:
                print("ERROR --int2-cap requires an integer")
                return 20
            try:
                int2_cap = int(args.pop(0))
            except ValueError:
                print("ERROR --int2-cap requires an integer")
                return 20
        else:
            print(f"ERROR unknown argument {arg!r}")
            return 20

    with open(ind.__file__, "rb") as fh:
        module_sha = hashlib.sha256(fh.read()).hexdigest()
    if not module_sha.startswith(MODULE_SHA_PREFIX):
        print(f"MODULE_SHA_MISMATCH {module_sha}")
        return 30
    if not float(np.finfo(np.longdouble).eps) < 1e-18:
        print("LONGDOUBLE_UNAVAILABLE")
        return 30
    if np.__version__ != NUMPY_VERSION:
        print(f"NUMPY_VERSION_MISMATCH {np.__version__}")
        return 30
    if mpmath.__version__ != MPMATH_VERSION:
        print(f"MPMATH_VERSION_MISMATCH {mpmath.__version__}")
        return 30

    grid = ind.build_independent_grid(48, 24.0)
    config = ind.IndependentCollisionConfig()

    members = {}
    if dev_state:
        members["DEV"] = (DEV_STATE[0], DEV_STATE[1], False, CAP_DEFAULT)
    else:
        for name, (scales, tg) in FAMILY_PAIRS.items():
            cap = CAP_NATIVE_SB if name.startswith("SB") else CAP_DEFAULT
            members[name] = (scales, tg, False, cap)
        for name, (scales, tg) in FAMILY_FIXED.items():
            members[name] = (scales, tg, True, CAP_DEFAULT)

    try:
        states_report = {}
        all_ok = True
        sa1_pt = None
        sa1_captured = None
        for name, (scales, tg, p_fixed, cap) in members.items():
            record, action_f, pt_f, shadow_self = evaluate_member(
                name, grid, config, scales, tg, p_fixed, cap, INT1_PREC_BITS
            )
            # N8 anchors on the continuity block, in the D-055 convention:
            # D_native = max(sup|pt_mu|, sup|pt_tau|, tiny) over mu/tau rows
            # only, and the residual numerator equals the module's mu/tau
            # numerator bitwise for P-fixed states.
            if name in ANCHOR_D_NATIVE:
                numerator = float(np.max(np.abs(pt_f[1] - pt_f[2])))
                d_anchor = max(float(np.max(np.abs(pt_f[1]))),
                               float(np.max(np.abs(pt_f[2]))), TINY)
                record["D_anchor"] = d_anchor
                record["anchor_numerator"] = numerator
                anchor_ok = (
                    d_anchor == ANCHOR_D_NATIVE[name]
                    and record["R_native_module"] == ANCHOR_R_NATIVE[name]
                    and record["R_native_module"] == numerator / d_anchor
                )
                record["checks"]["N8_anchor"] = bool(anchor_ok)
                record["ok"] = bool(record["ok"] and anchor_ok)
            states_report[name] = record
            all_ok = all_ok and record["ok"]
            log(f"{name}: ok={record['ok']} N1={record['N1']:.3e} "
                f"N3={record['N3']:.3e} certified_hi={record['certified_residual']['hi']:.3e}")
            if name == INT2_PAIR or (dev_state and name == "DEV"):
                sa1_pt = pt_f
                sa1_captured = shadow_self["captured"]
                sa1_states = {
                    "f": state_from_scales(grid, scales),
                    "p": p_state(state_from_scales(grid, scales)),
                }
                sa1_tg = tg

        # M-INT-2 deep-node replay.
        int2 = None
        if sa1_captured is not None:
            int2 = iv_deep_node_replay(
                grid, config, T_CM, sa1_states, sa1_captured, INT2_PREC_BITS,
                INT2_WIDEN_REL, sample_cap=int2_cap,
            )
            all_ok = all_ok and int2["ok"]
            log(f"M-INT-2 ok={int2['ok']} failures="
                f"{int2['containment_failures']}/{int2['identity_failures']}")

        # Mutants (frozen family run only).
        mutants = None
        if not dev_state:
            log("running mutant battery")
            mutants = run_mutants(grid, config, sa1_pt)
            for key, entry in mutants.items():
                if key != "canary" and not entry["killed"]:
                    all_ok = False
    except TimeoutError as error:
        report = {
            "contract": "BD622_D061_covariance_metrology_contract_r4_2026-07-28",
            "module_sha256": module_sha, "verdict": "ERROR",
            "error": f"wall budget exceeded: {error}",
        }
        text = json.dumps(report, sort_keys=True, indent=1)
        print(text)
        if out_path is not None:
            with open(out_path, "w") as fh:
                fh.write(text + "\n")
        return 20
    except Exception as error:  # mechanical failure, preserved
        report = {
            "contract": "BD622_D061_covariance_metrology_contract_r4_2026-07-28",
            "module_sha256": module_sha, "verdict": "ERROR",
            "error": f"{type(error).__name__}: {error}",
        }
        text = json.dumps(report, sort_keys=True, indent=1)
        print(text)
        if out_path is not None:
            with open(out_path, "w") as fh:
                fh.write(text + "\n")
        return 20

    verdict = "PASS" if all_ok else "FAIL"
    report = {
        "contract": "BD622_D061_covariance_metrology_contract_r4_2026-07-28",
        "claim_id": "C-F10-METROLOGY-R3",
        "module_sha256": module_sha,
        "numpy": np.__version__, "mpmath": mpmath.__version__,
        "longdouble_eps": str(np.finfo(np.longdouble).eps),
        "dev_state": bool(dev_state), "int2_cap": int2_cap,
        "wall_seconds": round(time.monotonic() - START_TIME, 1),
        "states": states_report,
        "m_int2": int2,
        "mutants": mutants,
        "verdict": verdict,
    }
    text = json.dumps(report, sort_keys=True, indent=1)
    print(text)
    if out_path is not None:
        with open(out_path, "w") as fh:
            fh.write(text + "\n")
    return 0 if verdict == "PASS" else 10


if __name__ == "__main__":
    sys.exit(main(sys.argv))
