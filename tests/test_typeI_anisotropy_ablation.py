from __future__ import annotations

import pytest

from rabbit.drivers.full_coupled_typeI import FullCoupledConfig, run_full_coupled_typeI


def _run(**overrides):
    params = dict(
        Sigma_H_plus=0.3,
        N_q=8,
        N_mu=8,
        n_reactions=12,
        correction_level=0,
        tier=1,
        enable_teff=False,
    )
    params.update(overrides)
    cfg = FullCoupledConfig(**params)
    return run_full_coupled_typeI(cfg)


@pytest.mark.production
@pytest.mark.gold
def test_typeI_shear_signal_requires_hubble_and_transport_weak_channels():
    """End-to-end ablation: Type I shear affects yields through two live channels.

    The control with both channels ablated must return the FLRW yield, while
    removing either the anisotropic Hubble factor or the transported weak-rate
    monopole removes a finite part of the sheared Y_p shift.
    """
    flrw = _run(Sigma_H_plus=0.0)
    full = _run()
    no_hubble = _run(ablate_hubble_anisotropy=True)
    no_weak_monopole = _run(ablate_weak_transport_monopole=True)
    both_ablated = _run(
        ablate_hubble_anisotropy=True,
        ablate_weak_transport_monopole=True,
    )

    assert full.observables.Yp > flrw.observables.Yp
    assert no_hubble.observables.Yp > flrw.observables.Yp
    assert no_weak_monopole.observables.Yp > flrw.observables.Yp

    assert full.observables.Yp - no_hubble.observables.Yp > 2.0e-4
    assert full.observables.Yp - no_weak_monopole.observables.Yp > 1.0e-3
    assert abs(both_ablated.observables.Yp - flrw.observables.Yp) < 5.0e-7

    assert full.metadata["ablation_flags"] == {
        "hubble_anisotropy": False,
        "weak_transport_monopole": False,
    }
    assert no_hubble.metadata["ablation_flags"]["hubble_anisotropy"] is True
    assert no_weak_monopole.metadata["ablation_flags"]["weak_transport_monopole"] is True

    full_m1 = full.metadata["phase1_handoff_monopole_probe"]["m1"]
    ablated_m1 = no_weak_monopole.metadata["phase1_handoff_monopole_probe"]["m1"]
    assert abs(full_m1 - ablated_m1) > 5.0e-5
