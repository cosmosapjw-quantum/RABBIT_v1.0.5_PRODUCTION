from __future__ import annotations

from enum import Enum


class Species(str, Enum):
    NUE = "nue"
    NUEBAR = "nuebar"
    NUX = "nux"


BANK_DEGENERACY = {
    Species.NUE: 1.0,
    Species.NUEBAR: 1.0,
    Species.NUX: 4.0,   # one effective bank representing 4 ν_x states
}


def as_species(x) -> Species:
    if isinstance(x, Species):
        return x
    return Species(str(x))


def bank_temperature(species, T_nu_e: float, T_nu_x: float) -> float:
    sp = as_species(species)
    if sp in (Species.NUE, Species.NUEBAR):
        return float(T_nu_e)
    return float(T_nu_x)


def bank_energy_transfer_rate(species, T_gamma: float, T_nu_e: float, T_nu_x: float) -> float:
    """
    Species-tagged target energy-transfer rate for the *bank*.

    MVP policy:
    - NUE and NUEBAR each get half of the ν_e-pair transfer
    - NUX bank represents all 4 ν_x states and gets the full 2-pair ν_x transfer

    We avoid top-level imports here to prevent import-cycle / early-import failures.
    """
    sp = as_species(species)

    from rabbit.thermo import nudec_coupled as nd

    # Preferred path: use direct per-species rate functions if available
    fn_nue = getattr(nd, "energy_transfer_rate_nue", None)
    fn_nux = getattr(nd, "energy_transfer_rate_nux", None)

    if callable(fn_nue) and callable(fn_nux):
        if sp is Species.NUE:
            return float(fn_nue(T_gamma, T_nu_e))
        if sp is Species.NUEBAR:
            return float(fn_nue(T_gamma, T_nu_e))
        if sp is Species.NUX:
            return float(4.0 * fn_nux(T_gamma, T_nu_x))
        raise ValueError(f"Unknown species: {species}")

    # Robust fallback: derive bank rates from total_energy_transfer
    # total_energy_transfer returns:
    #   (dQ_nue_pair, dQ_nux_2pairs)
    # where dQ_nue_pair includes ν_e + \barν_e pair,
    # and dQ_nux_2pairs includes all 4 ν_x states.
    dQ_nue_pair, dQ_nux_2pairs = nd.total_energy_transfer(T_gamma, T_nu_e, T_nu_x)

    if sp is Species.NUE:
        return 0.5 * float(dQ_nue_pair)
    if sp is Species.NUEBAR:
        return 0.5 * float(dQ_nue_pair)
    if sp is Species.NUX:
        return float(dQ_nux_2pairs)

    raise ValueError(f"Unknown species: {species}")
