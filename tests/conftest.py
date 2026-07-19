"""Pytest configuration and shared fixtures."""
from __future__ import annotations

import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

# Phase α adjoint stability hardening: on systems with mis-detected ROCm/HIP
# devices (no compatible GPU binary) JAX 0.10 may attempt GPU dispatch and
# fail mid-test with HIP_ERROR_NoBinaryForGpu. Force CPU dispatch unless the
# user explicitly opts into another platform via JAX_PLATFORMS. The env var
# must be set BEFORE jax is imported anywhere.
os.environ.setdefault("JAX_PLATFORMS", "cpu")
os.environ.setdefault("JAX_ENABLE_X64", "1")

import pytest


def pytest_configure(config):
    config.addinivalue_line("markers", "slow: marks tests as slow (>30s)")
    config.addinivalue_line("markers", "jax: marks tests requiring JAX")
    config.addinivalue_line("markers", "production: production-readiness gate")
    config.addinivalue_line("markers", "gold: BBN observable gold-lock")
    config.addinivalue_line("markers", "cross_code: live external-code parity")
