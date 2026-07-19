"""Canonical likelihood wrapper locks after F06 JAX endpoint retirement."""
from __future__ import annotations

import os
import subprocess
import sys

import pytest

from rabbit.inference import forward_likelihood as fl


def test_forward_likelihood_import_keeps_jax_runtime_lazy() -> None:
    code = "\n".join(
        [
            "import sys",
            "import rabbit.inference.forward_likelihood",
            "loaded = [name for name in ('rabbit.jax',) if name in sys.modules]",
            "raise SystemExit(1 if loaded else 0)",
        ]
    )
    env = dict(os.environ)
    env.setdefault("JAX_PLATFORMS", "cpu")
    result = subprocess.run(
        [sys.executable, "-c", code],
        env=env,
        text=True,
        capture_output=True,
        check=False,
    )
    assert result.returncode == 0, result.stderr or result.stdout


def test_high_level_jax_prewarm_request_fails_closed():
    with pytest.raises(ValueError, match="high-level Type-I JAX runtime was removed"):
        fl.make_canonical_forward_model(prewarm_jax=True)


def test_likelihood_defaults_to_no_automatic_prewarm(monkeypatch):
    class FakeModel:
        def predict(self, **params):
            return fl.BBNPrediction(Yp=0.245, DH=2.55e-5, params=params, success=True)

    monkeypatch.setattr(fl, "make_canonical_forward_model", lambda **kwargs: FakeModel())
    like = fl.make_canonical_likelihood(backend="auto")
    assert like.auto_prewarm_on_first_loglike is False
