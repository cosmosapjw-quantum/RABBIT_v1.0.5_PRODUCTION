"""
RABBIT Standard Network Migration Guide
========================================

Upgrades the nuclear network from 8-species/12-reaction (abundances_v2)
to 9-species/31-reaction (abundances_standard) using PRIMAT AC2024 data.

Backward-compatible: n_reactions=12 (default) gives identical physics.
The 9th species (⁶Li, index 8) is inert when using 12 reactions.


§1. FILES TO DELETE (deprecated)
================================

  network_extended_network.py   — Legacy 12-rxn extended (CF88 R13 only)
  network_extended_rates.py     — CF88 parametric rate functions
  network_abundances_ext26.py   — Buggy 26-rxn with CF88+placeholders


§2. FILES TO ADD
================

  network_abundances_standard.py  — New 9-species/31-reaction module
  primat_ac2024_31rxn.json        — PRIMAT rate table (31 rxn × 60 T9)


§3. DRIVER PATCHES
==================

Both drivers need exactly 3 changes:
  (a) Import path: abundances_v2 → abundances_standard
  (b) Config: add n_reactions parameter
  (c) RHS call: pass n_reactions to abundance_rhs_phase2

Impact:
  - State vector size: +1 (⁶Li slot at end of nuclear block)
  - Species indices 0-7: UNCHANGED
  - Y_p, D/H, ⁷Li/H extraction: UNCHANGED (same indices)
  - New observable: ⁶Li/H = X_final[8] / X_final[1]
"""

# ══════════════════════════════════════════════════════════════════
# §3a. PATCH for drivers_full_coupled_typeI.py
# ══════════════════════════════════════════════════════════════════
#
# --- CHANGE 1: Import (line 52-55) ---
#
# BEFORE:
#   from physics_v2.network.abundances_v2 import (
#       abundance_rhs_phase1, abundance_rhs_phase2,
#       phase1_to_phase2, N_SPECIES, mass_conservation_residual,
#   )
#
# AFTER:
#   from physics_v2.network.abundances_standard import (
#       abundance_rhs_phase1, abundance_rhs_phase2,
#       phase1_to_phase2, N_SPECIES, mass_conservation_residual,
#       SPECIES_NAMES, N_REACTIONS_FULL, N_REACTIONS_BACKBONE,
#   )
#
# --- CHANGE 2: Config dataclass (after existing fields) ---
#
# ADD field:
#   n_reactions: int = 12    # 12=backbone (Paper I), 31=standard
#
# --- CHANGE 3: RHS Phase 2 call (line ~228) ---
#
# BEFORE:
#   dX = abundance_rhs_phase2(X, T_gamma, eta, lnp, lpn) / max(H, 1e-100)
#
# AFTER:
#   dX = abundance_rhs_phase2(X, T_gamma, eta, lnp, lpn,
#                              n_reactions=n_reactions) / max(H, 1e-100)
#
# (n_reactions passed from config through the closure)
#
# --- CHANGE 4: Results extraction (line ~362) ---
#
# ADD after Li7_H extraction:
#   Li6_H = float(X_final[8] / X_final[1]) if X_final[1] > 0 else 0.0


# ══════════════════════════════════════════════════════════════════
# §3b. PATCH for drivers_classA_driver.py
# ══════════════════════════════════════════════════════════════════
#
# --- CHANGE 1: Import (line 61-65) ---
#
# BEFORE:
#   from physics_v2.network.abundances import (
#       N_SPECIES, abundance_rhs_phase1, abundance_rhs_phase2,
#       phase1_to_phase2, mass_conservation_residual,
#       ATOMIC_MASSES,
#   )
#
# AFTER:
#   from physics_v2.network.abundances_standard import (
#       N_SPECIES, abundance_rhs_phase1, abundance_rhs_phase2,
#       phase1_to_phase2, mass_conservation_residual,
#       ATOMIC_MASSES, SPECIES_NAMES, N_REACTIONS_FULL,
#   )
#
# --- CHANGE 2: ClassAConfig (line ~266, add field) ---
#
#   n_reactions: int = 12    # 12=backbone (Paper I), 31=standard
#
# --- CHANGE 3: RHS call (line 219) ---
#
# BEFORE:
#   dX = abundance_rhs_phase2(X, T_gamma, eta, lnp, lpn) / max(H, 1e-100)
#
# AFTER:
#   dX = abundance_rhs_phase2(X, T_gamma, eta, lnp, lpn,
#                              n_reactions=n_reactions) / max(H, 1e-100)
#
# --- CHANGE 4: Results extraction (line ~434) ---
#
# ADD:
#   Li6_H = float(X_final[8] / X_final[1]) if X_final[1] > 0 else 0.0


# ══════════════════════════════════════════════════════════════════
# §4. VERIFICATION CHECKLIST
# ══════════════════════════════════════════════════════════════════
#
# After applying patches, run:
#
# 1. python -c "from physics_v2.network.abundances_standard import *; print(N_SPECIES, N_REACTIONS_FULL)"
#    → Expected: 9 31
#
# 2. python -m pytest tests/test_standard_network.py -v
#    → Expected: 9/9 PASS
#
# 3. Full BBN run with n_reactions=12:
#    → Y_p, D/H, ⁷Li/H identical to previous v2 results (< 10⁻⁵ relative)
#
# 4. Full BBN run with n_reactions=31:
#    → ΔY_p < 10⁻⁴, Δ(D/H) < 0.001%, Δ(⁷Li) ≈ -2%, ⁶Li/H ≈ 10⁻¹⁴
