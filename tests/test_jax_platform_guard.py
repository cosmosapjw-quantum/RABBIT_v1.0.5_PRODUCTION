from __future__ import annotations


def test_jax_platform_guard_pins_cpu_without_device_hint(monkeypatch):
    from rabbit.config import jax_config as cfg

    monkeypatch.delenv("JAX_PLATFORMS", raising=False)
    monkeypatch.delenv("JAX_PLATFORM_NAME", raising=False)
    monkeypatch.delenv("RABBIT_JAX_ASSUME_ACCELERATOR", raising=False)
    monkeypatch.setattr(cfg, "local_accelerator_device_hint_available", lambda: False)

    out = cfg.maybe_pin_jax_platform_to_cpu_without_accelerator(reason="test")

    assert out["applied"] is True
    assert out["platforms"] == "cpu"
    assert out["reason"] == "test"
    assert cfg.os.environ["JAX_PLATFORMS"] == "cpu"


def test_jax_platform_guard_respects_explicit_platform(monkeypatch):
    from rabbit.config import jax_config as cfg

    monkeypatch.setenv("JAX_PLATFORMS", "rocm")
    monkeypatch.delenv("JAX_PLATFORM_NAME", raising=False)
    monkeypatch.setattr(cfg, "local_accelerator_device_hint_available", lambda: False)

    out = cfg.maybe_pin_jax_platform_to_cpu_without_accelerator(reason="test")

    assert out["applied"] is False
    assert out["reason"] == "explicit_jax_platform"
    assert cfg.os.environ["JAX_PLATFORMS"] == "rocm"


def test_jax_platform_guard_respects_device_hint(monkeypatch):
    from rabbit.config import jax_config as cfg

    monkeypatch.delenv("JAX_PLATFORMS", raising=False)
    monkeypatch.delenv("JAX_PLATFORM_NAME", raising=False)
    monkeypatch.delenv("RABBIT_JAX_ASSUME_ACCELERATOR", raising=False)
    monkeypatch.setattr(cfg, "local_accelerator_device_hint_available", lambda: True)

    out = cfg.maybe_pin_jax_platform_to_cpu_without_accelerator(reason="test")

    assert out["applied"] is False
    assert out["reason"] == "accelerator_device_hint_present"
    assert "JAX_PLATFORMS" not in cfg.os.environ
