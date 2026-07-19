import jax.numpy as jnp
import pytest

from rabbit.jax.solver_jax_rodas5p import _cached_solver_runner, _solver_runner_cache_key


def _rhs(N, y):
    return -y


def test_solver_runner_cache_key_includes_state_dim_for_schur_runner():
    key4 = _solver_runner_cache_key(_rhs, None, 24, None, (0, 2), 4)
    key5 = _solver_runner_cache_key(_rhs, None, 24, None, (0, 2), 5)
    assert key4 != key5


def test_cached_solver_runner_requires_state_dim_when_active_indices_are_used():
    with pytest.raises(ValueError, match='state_dim is required'):
        _cached_solver_runner(_rhs, None, event_refine_steps=24, active_indices=jnp.array([0, 2], dtype=jnp.int32))


def test_cached_solver_runner_reuses_callable_when_state_dim_is_fixed():
    active = jnp.array([0, 2], dtype=jnp.int32)
    r1 = _cached_solver_runner(_rhs, None, event_refine_steps=24, active_indices=active, state_dim=4)
    r2 = _cached_solver_runner(_rhs, None, event_refine_steps=24, active_indices=active, state_dim=4)
    assert r1 is r2
