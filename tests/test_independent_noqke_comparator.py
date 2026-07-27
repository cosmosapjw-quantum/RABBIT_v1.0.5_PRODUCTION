"""Focused private gates for the independent full-spectral no-QKE comparator."""

import numpy as np

from rabbit.decoupling import _independent_noqke as ind


def _state(grid: ind.IndependentNoQkeGrid, *, split: bool) -> np.ndarray:
    scales = (1.01, 0.995, 0.995) if split else (1.0, 1.0, 1.0)
    logits = np.stack([-grid.nodes / scale for scale in scales])
    return ind.pair_logits_to_cloglog(logits)


def test_modal_basis_and_unique_catalogues_reconstruct_target_views():
    grid = ind.build_independent_grid(8, 8.0)
    basis = grid.modal_basis(grid.nodes)
    np.testing.assert_allclose(basis.T @ (grid.weights[:, None] * basis), np.eye(8), atol=2e-13)
    linear = -0.3 - 0.7 * grid.nodes
    np.testing.assert_allclose(basis @ grid.modal_coefficients(linear), linear, atol=2e-13)

    events = ind.independent_self_events()
    reconstructed: dict[tuple[str, str], float] = {}
    for event in events:
        for species in set(event.legs):
            key = (event.category, species)
            reconstructed[key] = reconstructed.get(key, 0.0) + event.coefficient * event.legs.count(species)
    target: dict[tuple[str, str], float] = {}
    for row in ind.independent_self_reactions():
        key = (row.category, row.target)
        target[key] = target.get(key, 0.0) + row.coefficient
    assert len(events) == 25
    assert reconstructed == target
    assert ind.independent_pair_row_fingerprint() == (2, 1, 6, 2, 2, 2, 4, 4, 2)
    closure = [event for event in events if event.category == "pair_conversion"
               and "nu_mu" in event.legs and "nu_tau" in event.legs]
    assert [event.legs for event in closure] == [
        ("nu_mu", "antinu_mu", "nu_tau", "antinu_tau"),
        ("nu_tau", "antinu_tau", "nu_mu", "antinu_mu"),
    ]
    assert all(event.kernel == "K_t" and event.coefficient == 16.0 for event in closure)

    electron = []
    for event in ind.independent_electron_events():
        targets = (event.target, ind._cp_partner(event.target)) if event.category == "pair" else (event.target,)
        electron += [(target, event.category, 128.0) for target in targets]
    expected = [(row.target, row.category, 128.0) for row in ind.independent_electron_reactions()]
    assert len(ind.independent_electron_events()) == 15
    assert sorted(electron) == sorted(expected)


def test_synthetic_weak_event_embeds_number_and_energy_before_reduction():
    grid = ind.build_independent_grid(8, 8.0)
    y = np.array([1.0, 2.0, 1.4, 1.6])
    jump = grid.modal_basis(y[0]) + grid.modal_basis(y[1])
    jump -= grid.modal_basis(y[2]) + grid.modal_basis(y[3])
    modal = np.zeros((6, grid.order))
    modal[0] = 2.5 * jump
    action = ind._native_action(grid, modal, 1.0)
    moments = ind.independent_action_moments(grid=grid, action=action, temperature_cm_mev=1.0)
    assert abs(moments.signed_number_rate) <= 2e-13
    assert abs(moments.signed_energy_rate) <= 2e-13


def test_low_order_real_operator_static_invariants_and_entropy():
    grid = ind.build_independent_grid(8, 8.0)
    config = ind.IndependentCollisionConfig(
        incoming_polar_order=2, final_polar_order=2, electron_radial_order=8
    )
    common = ind.evaluate_independent_collision_action(
        grid=grid, pair_cloglog=_state(grid, split=False),
        temperature_cm_mev=10.0, temperature_gamma_mev=10.0, config=config,
    )
    thermo = ind.independent_thermodynamics(
        grid=grid, pair_cloglog=_state(grid, split=False),
        temperature_cm_mev=10.0, temperature_gamma_mev=10.0,
    )
    null = ind.independent_action_moments(grid=grid, action=common.total, temperature_cm_mev=10.0)
    assert null.absolute_number_rate / (thermo.hubble_mev * thermo.number_density_neutrino) <= 1e-10
    assert null.absolute_energy_rate / (thermo.hubble_mev * thermo.energy_density_neutrino) <= 1e-10

    result = ind.evaluate_independent_collision_action(
        grid=grid, pair_cloglog=_state(grid, split=True),
        temperature_cm_mev=10.0, temperature_gamma_mev=10.0, config=config,
    )
    self_moments = ind.independent_action_moments(
        grid=grid, action=result.self_interaction, temperature_cm_mev=10.0
    )
    assert abs(self_moments.signed_number_rate) / self_moments.absolute_number_rate <= 1e-10
    assert abs(self_moments.signed_energy_rate) / self_moments.absolute_energy_rate <= 1e-10
    assert result.diagnostics["first_law_residual"] <= 1e-8
    assert result.diagnostics["charge_conjugation_residual"] <= 1e-10
    assert result.diagnostics["mu_tau_residual"] <= 1e-10
    assert result.diagnostics["entropy_production"] >= -1e-24
