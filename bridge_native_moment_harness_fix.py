"""
BD400 re-audit deliverable — FIX for the bridge-vs-table dQ harness mismatch.

WHY: artifacts/bridge_vs_table_dQ_q4_q8_q16.json reports raw "native" excess of
-1085% (q4), +1,046,575% (q8), +1,249,447% (q16) with a SIGN FLIP between q4 and
q8. A convergent quadrature error of a physical dQ is sub-percent and never
sign-flips by 4 orders of magnitude. The cause: the harness wires "native" to
build_augmented_pstf_radial_moment_thermo_source_from_geometry(..., energy_
normalization="raw"), which is an UNNORMALIZED standalone radial moment, not the
physical net energy transfer the solver actually uses. So the harness does NOT
test the q-grid hypothesis; it measures the wrong quantity.

The standard_3t_plasma closed arm DOES match the table (excess ~1e-11), which
only re-confirms that the closure renormalizes to total_energy_transfer (already
known from augmented_collision_bridge.py:2251). It says nothing about the real
+0.665% trajectory excess seen in BD397/BD398.

CORRECT TEST: extract the native dQ from the SAME path the solver integrates,
coupled_3T_rhs_from_collision_moments, fed by the bridge dQ moments produced
during a real evaluation at matched (T_gamma, T_nu_e, T_nu_x). Compare the
resulting dT_nu/dN (or equivalently the implied dQ) against the table-driven
coupled_3T_rhs. The excess to explain is the ~0.665% on T_nu/T_gamma, not an
unnormalized radial moment.

This file is a concrete shape; wire the marked call to the real bridge moment
producer used inside augmented_typeI_replay.
"""
from __future__ import annotations
import numpy as np
from rabbit.thermo.nudec_coupled import (
    coupled_3T_rhs,
    coupled_3T_rhs_from_collision_moments,
    hubble_3T,
    N_eff_from_3T,
)

# BD397 artifact cold-row state (N=2.5).
STATE = dict(T_gamma=0.08735709565954111,
             T_nu_e=0.06572257387709174,
             T_nu_x=0.06567977286950767)


def table_driven_rhs(Tg, Te, Tx):
    """Reference: thermo RHS driven by the calibrated total_energy_transfer table."""
    H = hubble_3T(Tg, Te, Tx)
    return np.asarray(coupled_3T_rhs(Tg, Te, Tx, H_MeV=H), dtype=float)


def bridge_moment_dQ(Tg, Te, Tx, *, n_q, n_mu):
    """STUB — must return the SAME dQ moments the solver consumes.

    Wire to the in-replay bridge call that produces dQ_nue_pair_N / dQ_nux_bank_N
    (the values fed to coupled_3T_rhs_from_collision_moments), NOT to the raw
    radial-moment source. i.e. reproduce augmented_typeI_replay's collision-moment
    construction at this fixed (Tg,Te,Tx) and read the dQ it hands to the closure.
    """
    raise NotImplementedError(
        "Return (dQ_nue_pair_N, dQ_nux_bank_N) exactly as augmented_typeI_replay "
        "passes them into coupled_3T_rhs_from_collision_moments at this state."
    )


def bridge_driven_rhs(Tg, Te, Tx, *, n_q, n_mu):
    H = hubble_3T(Tg, Te, Tx)
    dQ_nue_pair_N, dQ_nux_bank_N = bridge_moment_dQ(Tg, Te, Tx, n_q=n_q, n_mu=n_mu)
    return np.asarray(
        coupled_3T_rhs_from_collision_moments(
            Tg, Te, Tx, H_MeV=H,
            dQ_nue_pair_N=dQ_nue_pair_N, dQ_nux_bank_N=dQ_nux_bank_N),
        dtype=float,
    )


if __name__ == "__main__":
    ref = table_driven_rhs(**STATE)
    print(f"table-driven d(T_gamma,T_nue,T_nux)/dN = {ref}")
    print("Compare bridge-driven RHS (once wired) at q4/q8/q16:")
    print(f"{'n_q':>4} {'rel dT_nue diff':>16} {'rel dT_nux diff':>16}")
    for n_q in (4, 8, 16):
        try:
            got = bridge_driven_rhs(**STATE, n_q=n_q, n_mu=16)
            print(f"{n_q:>4} {(got[1]/ref[1]-1):>16.6f} {(got[2]/ref[2]-1):>16.6f}")
        except NotImplementedError:
            print(f"{n_q:>4} {'(wire bridge moment)':>16} {'(wire bridge moment)':>16}")
    print("\nFALSIFIER: if the in-solver bridge dT_nu/dN matches the table to <0.1% at")
    print("all q, the +0.665% trajectory excess is NOT a per-step dQ error and must")
    print("come from accumulated state evolution or QED/EOS convention -> re-scope PR-2.")
