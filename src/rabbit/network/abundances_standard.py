"""
rabbit.network.abundances_standard — Standard BBN network (PRIMAT AC2024).

9 species, 31 reactions.  ALL rates from PRIMAT tabulated data.
No CF88 parametric fits.  No placeholders.

Species (9):
    0: n,  1: p,  2: D,  3: T,  4: ³He,  5: ⁴He,  6: ⁷Li,  7: ⁷Be,  8: ⁶Li

Reactions (31):
    R0–R11:  Backbone (identical to abundances_v2.py / small_network)
    R12–R30: Extension — all PRIMAT-tabulated (AC 2024-10-03)

Backward compatibility:
    n_reactions=12 → exact backbone (abundances_v2 parity)
    n_reactions=31 → full standard network

Rate data source: data/primat_ac2024_31rxn.json
    Parsed from BBNRatesAC2024.dat (Coc & Pitrou, PRIMAT collaboration)
    References per reaction stored in JSON.

Supersedes:
    - network_abundances_ext26.py  (CF88 rates → PRIMAT tabulated)
    - network_extended_network.py  (deprecated legacy)
    - network_extended_rates.py    (deprecated legacy)

References:
    Pitrou et al. 2018, Phys. Rep. 754, 1 — PRIMAT code
    Coc & Pitrou 2024 — AC2024 rate compilation
    Moscoso et al. 2021 — d(p,γ)³He (LUNA successor)
    de Souza et al. 2019, 2020 — T+D, ³He+D, ⁷Be+n updates
"""
import json
import pathlib
import warnings
import numpy as np

from rabbit.validation.truncation_guards import (
    RATE_TABLE_T_MEV_MIN, RATE_TABLE_T_MEV_MAX, validate_manual_network_truncation,
    validate_rate_table_temperature,
)


# ═══════════════════════════════════════════════════════════════
# §1. Species and constants
# ═══════════════════════════════════════════════════════════════

N_SPECIES = 9
N_REACTIONS_FULL = 31
N_REACTIONS_BACKBONE = 12

SPECIES_NAMES = ('n', 'p', 'D', 'T', '3He', '4He', '7Li', '7Be', '6Li')
ATOMIC_MASSES = np.array([1, 1, 2, 3, 3, 4, 7, 7, 6], dtype=float)
CHARGE_NUMBERS = np.array([0, 1, 1, 1, 2, 2, 3, 4, 3], dtype=float)

REACTION_NAMES = (
    'n+p→D+γ',          # R0
    'D+p→³He+γ',        # R1
    'D+D→³He+n',        # R2
    'D+D→T+p',          # R3
    'T+p→⁴He+γ',        # R4
    'T+D→⁴He+n',        # R5
    'T+⁴He→⁷Li+γ',      # R6
    '³He+n→T+p',        # R7
    '³He+D→⁴He+p',      # R8
    '³He+⁴He→⁷Be+γ',    # R9
    '⁷Be+n→⁷Li+p',      # R10
    '⁷Li+p→⁴He+⁴He',    # R11
    '⁷Li+p→⁴He+⁴He+γ',  # R12  radiative variant
    '⁷Be+n→⁴He+⁴He',    # R13  excited state
    '⁷Be+D→2⁴He+p',     # R14
    'D+⁴He→⁶Li+γ',      # R15  6Li production
    '⁶Li+p→⁷Be+γ',      # R16
    '⁶Li+p→³He+⁴He',    # R17  dominant 6Li destruction
    '⁶Li+³He→⁴He+⁴He+p', # R18
    '⁶Li+T→⁴He+⁴He+n',  # R19
    '⁷Li+³He→⁶Li+⁴He',  # R20
    '⁷Be+T→⁶Li+⁴He',    # R21
    '⁶Li+T→⁷Li+D',      # R22
    '⁶Li+³He→⁷Be+D',    # R23
    '⁷Li+³He→⁴He+⁴He+D', # R24
    '⁷Be+T→⁴He+⁴He+D',  # R25
    '⁷Be+T→⁷Li+³He',    # R26
    '⁷Be+³He→pp+2⁴He',  # R27
    'D+D→⁴He+γ',        # R28
    '³He+³He→⁴He+pp',   # R29
    '⁷Li+D→⁴He+⁴He+n',  # R30
)

_MEV_TO_T9 = 11.6045


# ═══════════════════════════════════════════════════════════════
# §2. Stoichiometry matrix (9 × 31)
# ═══════════════════════════════════════════════════════════════

#: S[species, reaction] = net change.
#: Baryon conservation: A^T @ S[:,j] = 0  ∀j  (verified at import).
#: Charge conservation: Z^T @ S[:,j] = 0  ∀j  (verified at import).
STOICHIOMETRY = np.array([
    #  R0  R1  R2  R3  R4  R5  R6  R7  R8  R9 R10 R11 R12 R13 R14 R15 R16 R17 R18 R19 R20 R21 R22 R23 R24 R25 R26 R27 R28 R29 R30
    [ -1,  0, +1,  0,  0, +1,  0, -1,  0,  0, -1,  0,  0, -1,  0,  0,  0,  0,  0, +1,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0, +1],  # n
    [ -1, -1,  0, +1, -1,  0,  0, +1, +1,  0, +1, -1, -1,  0, +1,  0, -1, -1, +1,  0,  0,  0,  0,  0,  0,  0,  0, +2,  0, +2,  0],  # p
    [ +1, -1, -2, -2,  0, -1,  0,  0, -1,  0,  0,  0,  0,  0, -1, -1,  0,  0,  0,  0,  0,  0, +1, +1, +1, +1,  0,  0, -2,  0, -1],  # D
    [  0,  0,  0, +1, -1, -1, -1, +1,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0, -1,  0, -1, -1,  0,  0, -1, -1,  0,  0,  0,  0],  # T
    [  0, +1, +1,  0,  0,  0,  0, -1, -1, -1,  0,  0,  0,  0,  0,  0,  0, +1, -1,  0, -1,  0,  0, -1, -1,  0, +1, -1,  0, -2,  0],  # ³He
    [  0,  0,  0,  0, +1, +1, -1,  0, +1, -1,  0, +2, +2, +2, +2, -1,  0, +1, +2, +2, +1, +1,  0,  0, +2, +2,  0, +2, +1, +1, +2],  # ⁴He
    [  0,  0,  0,  0,  0,  0, +1,  0,  0,  0, +1, -1, -1,  0,  0,  0,  0,  0,  0,  0, -1,  0, +1,  0, -1,  0, +1,  0,  0,  0, -1],  # ⁷Li
    [  0,  0,  0,  0,  0,  0,  0,  0,  0, +1, -1,  0,  0, -1, -1,  0, +1,  0,  0,  0,  0, -1,  0, +1,  0, -1, -1, -1,  0,  0,  0],  # ⁷Be
    [  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0, +1, -1, -1, -1, -1, +1, +1, -1, -1,  0,  0,  0,  0,  0,  0,  0],  # ⁶Li
], dtype=float)

# Verify conservation laws at import time
_baryon_check = ATOMIC_MASSES @ STOICHIOMETRY
assert np.allclose(_baryon_check, 0, atol=1e-10), \
    f"Baryon conservation violated: {np.where(np.abs(_baryon_check)>1e-10)[0]}"
_charge_check = CHARGE_NUMBERS @ STOICHIOMETRY
assert np.allclose(_charge_check, 0, atol=1e-10), \
    f"Charge conservation violated: {np.where(np.abs(_charge_check)>1e-10)[0]}"


# ═══════════════════════════════════════════════════════════════
# §3. Identical particle factors
# ═══════════════════════════════════════════════════════════════

#: Forward rate symmetry factors (1/2 for identical reactant pairs).
#: These multiply the Y_i × Y_j product in the flux expression.
IDENTICAL_FACTOR = np.ones(N_REACTIONS_FULL)
IDENTICAL_FACTOR[2]  = 0.5   # D+D→³He+n
IDENTICAL_FACTOR[3]  = 0.5   # D+D→T+p
IDENTICAL_FACTOR[28] = 0.5   # D+D→⁴He+γ
IDENTICAL_FACTOR[29] = 0.5   # ³He+³He→⁴He+pp

#: Reverse rate symmetry factors (1/2 for identical product pairs).
IDENTICAL_FACTOR_REV = np.ones(N_REACTIONS_FULL)
IDENTICAL_FACTOR_REV[11] = 0.5  # ⁷Li+p ← ⁴He+⁴He
IDENTICAL_FACTOR_REV[12] = 0.5  # ⁷Li+p ← ⁴He+⁴He+γ


# ═══════════════════════════════════════════════════════════════
# §4. Rate table loading (PRIMAT AC2024)
# ═══════════════════════════════════════════════════════════════

_TABLE = None


def _load_table():
    """Load PRIMAT AC2024 31-reaction rate table."""
    global _TABLE
    if _TABLE is not None:
        return _TABLE

    candidates = [
        pathlib.Path(__file__).parent / 'data/primat_ac2024_31rxn.json',
        pathlib.Path(__file__).resolve().parent.parent / 'data/primat_ac2024_31rxn.json',
    ]
    path = None
    for p in candidates:
        if p.exists():
            path = p
            break
    if path is None:
        raise FileNotFoundError("data/primat_ac2024_31rxn.json not found")

    with open(path) as f:
        data = json.load(f)

    T9 = np.array(data['T9'])
    log_T9 = np.log(T9)

    log_rates = np.zeros((N_REACTIONS_FULL, len(log_T9)))
    Q_vals = np.zeros(N_REACTIONS_FULL)
    rev_factors = np.zeros(N_REACTIONS_FULL)
    T9_powers = np.zeros(N_REACTIONS_FULL)
    gammas = np.zeros(N_REACTIONS_FULL)

    for rxn in data['reactions']:
        i = rxn['index']
        lr = np.array(rxn['log_rates'], dtype=float)
        log_rates[i, :] = lr
        Q_vals[i] = rxn['Q_MeV']
        rev_factors[i] = rxn['rev_factor']
        T9_powers[i] = rxn['T9_power']
        gammas[i] = rxn['gamma']

    _TABLE = {
        'T9': T9, 'log_T9': log_T9, 'log_rates': log_rates,
        'Q': Q_vals, 'rev': rev_factors, 'T9p': T9_powers, 'gamma': gammas,
    }
    return _TABLE


# ═══════════════════════════════════════════════════════════════
# §5. Rate evaluation
# ═══════════════════════════════════════════════════════════════

def evaluate_nuclear_rates(T_gamma, n_reactions=N_REACTIONS_FULL, *, strict_bounds: bool = False):
    """Evaluate forward and reverse rates at T_γ [MeV].

    Parameters
    ----------
    T_gamma : float
        Photon temperature [MeV].
    n_reactions : int
        12 = backbone only, 31 = full standard.

    Returns
    -------
    fwd, rev : ndarray, shape (n_reactions,)
        Forward and reverse rates [cm³/mol/s].
    """
    validate_manual_network_truncation(n_reactions, context="SciPy nuclear-rate evaluation")
    validate_rate_table_temperature(T_gamma, context="SciPy nuclear-rate evaluation", strict=strict_bounds)
    tab = _load_table()
    T9 = max(T_gamma * _MEV_TO_T9, 1e-30)
    lT9 = np.log(T9)

    log_T9 = tab['log_T9']
    rates = tab['log_rates'][:n_reactions]

    idx = np.searchsorted(log_T9, lT9, side='right') - 1
    idx = int(np.clip(idx, 0, len(log_T9) - 2))
    x0 = log_T9[idx]
    x1 = log_T9[idx + 1]
    t = float((lT9 - x0) / max(x1 - x0, 1e-30))

    log_fwd = (1.0 - t) * rates[:, idx] + t * rates[:, idx + 1]
    if lT9 <= log_T9[0]:
        log_fwd = rates[:, 0]
    elif lT9 >= log_T9[-1]:
        log_fwd = rates[:, -1]

    fwd = np.exp(log_fwd)

    rf = tab['rev'][:n_reactions]
    tp = tab['T9p'][:n_reactions]
    gm = tab['gamma'][:n_reactions]
    mask = (rf > 0) & (np.abs(tab['Q'][:n_reactions]) > 0)
    log_Keq = np.log(np.where(mask, rf, 1.0)) + tp * np.log(T9) + gm / T9
    rev = np.where(mask, fwd * np.exp(np.clip(log_Keq, -500.0, 500.0)), 0.0)

    return fwd, rev


# ═══════════════════════════════════════════════════════════════
# §6. Reactant index pairs (for flux computation)
# ═══════════════════════════════════════════════════════════════

#: (i, j) = species indices of the two reactants for each reaction.
#: Flux = fwd × Y[i] × Y[j] × rho_fac × identical_factor
REACTANT_PAIRS = [
    (0, 1),  # R0:  n + p
    (2, 1),  # R1:  D + p
    (2, 2),  # R2:  D + D
    (2, 2),  # R3:  D + D
    (3, 1),  # R4:  T + p
    (3, 2),  # R5:  T + D
    (3, 5),  # R6:  T + ⁴He
    (4, 0),  # R7:  ³He + n
    (4, 2),  # R8:  ³He + D
    (4, 5),  # R9:  ³He + ⁴He
    (7, 0),  # R10: ⁷Be + n
    (6, 1),  # R11: ⁷Li + p
    (6, 1),  # R12: ⁷Li + p
    (7, 0),  # R13: ⁷Be + n
    (7, 2),  # R14: ⁷Be + D
    (2, 5),  # R15: D + ⁴He
    (8, 1),  # R16: ⁶Li + p
    (8, 1),  # R17: ⁶Li + p
    (8, 4),  # R18: ⁶Li + ³He
    (8, 3),  # R19: ⁶Li + T
    (6, 4),  # R20: ⁷Li + ³He
    (7, 3),  # R21: ⁷Be + T
    (8, 3),  # R22: ⁶Li + T
    (8, 4),  # R23: ⁶Li + ³He
    (6, 4),  # R24: ⁷Li + ³He
    (7, 3),  # R25: ⁷Be + T
    (7, 3),  # R26: ⁷Be + T
    (7, 4),  # R27: ⁷Be + ³He
    (2, 2),  # R28: D + D
    (4, 4),  # R29: ³He + ³He
    (6, 2),  # R30: ⁷Li + D
]

_REACTANT_I = np.array([ij[0] for ij in REACTANT_PAIRS], dtype=int)
_REACTANT_J = np.array([ij[1] for ij in REACTANT_PAIRS], dtype=int)
_STOICH_POS = np.maximum(STOICHIOMETRY, 0).astype(int)
_PRODUCT_COUNTS = np.sum(_STOICH_POS, axis=0).astype(int)
_REVERSE_SYM = np.ones(N_REACTIONS_FULL)
for _r in range(N_REACTIONS_FULL):
    fac = 1.0
    for _sk in _STOICH_POS[:, _r]:
        if _sk > 1:
            from math import factorial
            fac /= factorial(int(_sk))
    _REVERSE_SYM[_r] = fac


# ═══════════════════════════════════════════════════════════════
# §7. Flux computation and abundance RHS
# ═══════════════════════════════════════════════════════════════

def compute_flux_components(X, T_gamma, eta, n_reactions=N_REACTIONS_FULL):
    """Directional forward and reverse reaction fluxes.

    Reverse-flux formula (general):
        flux_rev = rho_fac^{n_prod−1} × sym_fac × rev × Π Y_k^{S_k}
        n_prod = total product particles (sum of positive S entries).
        sym_fac = 1 / Π_{S_k>0} S_k!  (identical-product symmetry).

    Returns
    -------
    flux_fwd, flux_rev : ndarray, shape (n_reactions,)
    """
    fwd, rev = evaluate_nuclear_rates(T_gamma, n_reactions)

    zeta3 = 1.202056903
    n_gamma = 2.0 * zeta3 / np.pi**2 * T_gamma**3
    n_b = eta * n_gamma
    hbar_c_cm = 197.3269804e-13
    n_b_cgs = n_b / hbar_c_cm**3
    N_A = 6.02214076e23
    rho_fac = n_b_cgs / N_A

    Y = X / ATOMIC_MASSES

    # Forward fluxes: rho_fac × sym × rate × Y_i × Y_j
    flux_fwd = (
        rho_fac
        * IDENTICAL_FACTOR[:n_reactions]
        * fwd
        * Y[_REACTANT_I[:n_reactions]]
        * Y[_REACTANT_J[:n_reactions]]
    )

    # Reverse fluxes: rho_fac^(n_prod-1) × sym × rev × Π Y_k^{S_k}
    logY = np.log(np.clip(Y, 1e-300, None))
    log_Y_prod = _STOICH_POS[:, :n_reactions].T @ logY
    Y_prod = np.exp(np.clip(log_Y_prod, -700.0, 700.0))
    rho_pow = np.maximum(_PRODUCT_COUNTS[:n_reactions] - 1, 0)
    flux_rev = (rho_fac ** rho_pow) * _REVERSE_SYM[:n_reactions] * rev * Y_prod
    flux_rev = np.where(rev > 0.0, flux_rev, 0.0)

    return flux_fwd, flux_rev


def compute_fluxes(X, T_gamma, eta, n_reactions=N_REACTIONS_FULL):
    """Net reaction fluxes for all reactions.

    Returns
    -------
    flux : ndarray, shape (n_reactions,)
    """
    flux_fwd, flux_rev = compute_flux_components(X, T_gamma, eta, n_reactions)
    return flux_fwd - flux_rev


def trace_species_production_destruction_split(
    X,
    T_gamma,
    eta,
    *,
    species_indices=(6, 7, 8),
    n_reactions=N_REACTIONS_FULL,
    abundance_floor=1.0e-300,
    top_k=6,
    clip_negative=False,
):
    """Production/destruction split for selected trace species.

    The split is directional: each forward reaction contributes with
    ``STOICHIOMETRY[:, r]`` and each reverse reaction contributes with the
    opposite stoichiometry.  This keeps production and destruction nonnegative
    for nonnegative physical abundances and exactly reconstructs the nuclear
    part of ``abundance_rhs_phase2``.
    """
    validate_manual_network_truncation(
        n_reactions,
        context="trace-species production/destruction split",
    )
    X_arr = np.asarray(X, dtype=float).reshape(-1)
    if X_arr.size != N_SPECIES:
        raise ValueError(f"X must contain {N_SPECIES} phase-2 species.")
    species = tuple(int(index) for index in species_indices)
    if any(index < 0 or index >= N_SPECIES for index in species):
        raise ValueError("species_indices contains an invalid phase-2 index.")

    raw_negative = [
        int(index)
        for index, value in enumerate(X_arr.tolist())
        if np.isfinite(value) and float(value) < 0.0
    ]
    X_flux = np.maximum(X_arr, 0.0) if bool(clip_negative) else X_arr
    flux_fwd, flux_rev = compute_flux_components(
        X_flux,
        T_gamma,
        eta,
        n_reactions=n_reactions,
    )
    S = STOICHIOMETRY[:, :n_reactions]
    directional_flux = np.concatenate((flux_fwd, flux_rev))
    directional_stoich = np.concatenate((S, -S), axis=1)
    directional_names = [
        f"forward:R{index}:{REACTION_NAMES[index]}"
        for index in range(n_reactions)
    ] + [
        f"reverse:R{index}:{REACTION_NAMES[index]}"
        for index in range(n_reactions)
    ]
    species_rows = []
    top_count = max(0, int(top_k))
    floor = max(float(abundance_floor), 1.0e-300)
    for index in species:
        contributions = (
            float(ATOMIC_MASSES[index])
            * directional_stoich[index, :]
            * directional_flux
        )
        production_terms = np.maximum(contributions, 0.0)
        destruction_terms = np.maximum(-contributions, 0.0)
        production = float(np.sum(production_terms))
        destruction = float(np.sum(destruction_terms))
        net = float(np.sum(contributions))
        x_scale = max(abs(float(X_flux[index])), floor)
        order = np.argsort(-np.abs(contributions))
        dominant = []
        for position in order[:top_count]:
            contribution = float(contributions[int(position)])
            if contribution == 0.0:
                continue
            dominant.append(
                {
                    "label": directional_names[int(position)],
                    "contribution": contribution,
                    "abs_contribution": abs(contribution),
                    "flux": float(directional_flux[int(position)]),
                }
            )
        species_rows.append(
            {
                "species_index": int(index),
                "species_name": str(SPECIES_NAMES[index]),
                "X": float(X_arr[index]),
                "X_flux": float(X_flux[index]),
                "production": production,
                "destruction": destruction,
                "net_dX": net,
                "production_to_X_rate": float(production / x_scale),
                "destruction_rate": float(destruction / x_scale),
                "dominant_abs_reactions": dominant,
            }
        )

    net_values = [float(item["net_dX"]) for item in species_rows]
    destruction_rates = [float(item["destruction_rate"]) for item in species_rows]
    production_rates = [
        float(item["production_to_X_rate"])
        for item in species_rows
    ]
    return {
        "available": True,
        "n_reactions": int(n_reactions),
        "T_gamma_MeV": float(T_gamma),
        "eta": float(eta),
        "species_indices": [int(index) for index in species],
        "species_names": [str(SPECIES_NAMES[index]) for index in species],
        "clip_negative": bool(clip_negative),
        "input_clip_policy": (
            "diagnostic_nonnegative_counterfactual"
            if bool(clip_negative)
            else "raw_abundances"
        ),
        "raw_negative_species_indices": raw_negative,
        "forward_flux_min": float(np.min(flux_fwd)) if flux_fwd.size else 0.0,
        "forward_flux_max": float(np.max(flux_fwd)) if flux_fwd.size else 0.0,
        "reverse_flux_min": float(np.min(flux_rev)) if flux_rev.size else 0.0,
        "reverse_flux_max": float(np.max(flux_rev)) if flux_rev.size else 0.0,
        "directional_flux_nonnegative": bool(
            (not raw_negative or bool(clip_negative))
            and (not flux_fwd.size or np.min(flux_fwd) >= 0.0)
            and (not flux_rev.size or np.min(flux_rev) >= 0.0)
        ),
        "directional_flux_nonnegative_reason": (
            "raw_input_outside_physical_domain"
            if raw_negative and not bool(clip_negative)
            else (
                "diagnostic_counterfactual_clipped_input"
                if raw_negative and bool(clip_negative)
                else "directional_forward_reverse_fluxes_nonnegative"
            )
        ),
        "net_dX_abs_max": float(np.max(np.abs(net_values))) if net_values else 0.0,
        "production_to_X_rate_abs_max": (
            float(np.max(np.abs(production_rates))) if production_rates else 0.0
        ),
        "destruction_rate_abs_max": (
            float(np.max(np.abs(destruction_rates))) if destruction_rates else 0.0
        ),
        "species": species_rows,
    }


def abundance_rhs_phase1(Xn, lambda_np, lambda_pn):
    """Phase 1: only n↔p interconversion."""
    return -lambda_np * Xn + lambda_pn * (1.0 - Xn)


def phase1_to_phase2(Xn):
    """Phase 2 IC from neutron fraction."""
    X = np.full(N_SPECIES, 1e-25)
    X[0] = Xn
    X[1] = 1.0 - Xn
    return X


def abundance_rhs_phase2(X, T_gamma, eta, lambda_np, lambda_pn,
                          n_reactions=N_REACTIONS_FULL):
    """Phase 2 RHS: nuclear network + weak rates.

    Parameters
    ----------
    n_reactions : int
        12 = backbone, 31 = full standard.
    """
    flux = compute_fluxes(X, T_gamma, eta, n_reactions)
    S = STOICHIOMETRY[:, :n_reactions]
    dX = (S @ flux) * ATOMIC_MASSES

    # Weak rates
    dXn_weak = -lambda_np * X[0] + lambda_pn * X[1]
    dX[0] += dXn_weak
    dX[1] -= dXn_weak
    return dX


def mass_conservation_residual(X):
    """Mass-fraction conservation: |Σ X_i − 1|.

    The evolved network variables X_i are mass fractions, so baryon-number
    conservation is equivalent to Σ_i X_i = 1.
    """
    return abs(float(np.sum(X) - 1.0))
