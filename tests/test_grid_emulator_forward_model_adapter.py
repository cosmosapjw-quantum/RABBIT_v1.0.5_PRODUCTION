import numpy as np


def test_grid_emulator_build_accepts_forward_model_like_and_prewarms_once():
    from rabbit.inference.forward_likelihood import ForwardModel, BBNPrediction
    from rabbit.inference.sampler import GridEmulator, GridEmulatorConfig

    calls = []

    def fake_prewarm():
        calls.append('prewarm')
        return {'cache_hit': False}

    def fake_solver(Sigma_H=0.0, eta=6.104e-10, tau_n=878.4):
        calls.append(('solve', float(Sigma_H), float(eta), float(tau_n)))
        return BBNPrediction(
            Yp=0.24 + 0.01 * float(Sigma_H),
            DH=2.5e-5 * (6.104e-10 / float(eta)),
            metadata={'Li7H': 5e-10},
        )

    model = ForwardModel(
        solver_fn=fake_solver,
        prewarm_fn=fake_prewarm,
        auto_prewarm_on_first_predict=False,
    )
    cfg = GridEmulatorConfig(param_ranges={
        'Sigma_H': (0.0, 1.0e-3, 2),
        'eta': (6.0e-10, 6.1e-10, 2),
    }, observables=('Yp', 'DH', 'Li7H'))

    emulator = GridEmulator.build(cfg, model, verbose=False)

    assert calls[0] == 'prewarm'
    assert len([c for c in calls if c == 'prewarm']) == 1
    assert emulator.data['Yp'].shape == (2, 2)
    assert emulator.data['DH'].shape == (2, 2)
    assert emulator.data['Li7H'].shape == (2, 2)
    assert np.isfinite(emulator.data['Yp']).all()


def test_grid_emulator_build_accepts_keyword_solver_callable():
    from rabbit.inference.sampler import GridEmulator, GridEmulatorConfig

    def fake_solver(*, Sigma_H=0.0, eta=6.104e-10):
        return {'Yp': 0.24 + Sigma_H, 'DH': 2.5e-5 * (6.104e-10 / eta)}

    cfg = GridEmulatorConfig(param_ranges={'Sigma_H': (0.0, 1.0e-3, 2), 'eta': (6.0e-10, 6.1e-10, 2)}, observables=('Yp', 'DH'))
    emulator = GridEmulator.build(cfg, fake_solver, verbose=False)

    assert emulator.data['Yp'].shape == (2, 2)
    assert emulator.data['DH'].shape == (2, 2)
