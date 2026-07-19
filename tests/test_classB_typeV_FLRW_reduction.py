"""Class-B component guard plus F06 public-endpoint retirement lock."""
from __future__ import annotations

import pytest


def test_classb_typev_public_flrw_reduction_endpoint_is_retired():
    from rabbit.inference.forward_likelihood import canonical_forward_solver

    with pytest.raises(ValueError, match="retired from the public forward surface"):
        canonical_forward_solver(Sigma_H=0.0, backend="jax_classB", N_q=12)


def test_classB_cl4_raises_documenting_ladder_envelope():
    """The retained component driver defines only CL0-CL3."""
    from rabbit.config.conventions import BianchiType
    from rabbit.jax.driver_classB import JAXClassBConfig, run_classB_phase1

    cfg = JAXClassBConfig(
        bianchi_type=BianchiType.TYPE_V,
        correction_level=4,
        Sigma_H_plus=0.0,
        eta=6.104e-10,
        tau_n=878.4,
        N_q=12,
        A_init=0.0,
    )
    with pytest.raises(ValueError, match=r"not in \{0, 1, 2, 3\}"):
        run_classB_phase1(cfg)
