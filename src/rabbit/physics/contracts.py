"""Minimal physics contracts for backend-aligned provider evolution.

These are deliberately small. The project is not yet ready for a fully generic
provider architecture, but the weak/thermo/geometry interfaces below define the
semantics that future JAX/SciPy parity work should preserve.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol, Tuple


@dataclass(frozen=True)
class WeakResult:
    lambda_np: float
    lambda_pn: float
    mode: str


class WeakProvider(Protocol):
    def __call__(self, *args, **kwargs) -> WeakResult: ...


@dataclass(frozen=True)
class ThermoResult:
    hubble: float
    dT_gamma_dN: float
    tier: int
    T_nu_for_weak: float
    T_nu_e: float | None = None
    T_nu_x: float | None = None
    dT_nu_e_dN: float | None = None
    dT_nu_x_dN: float | None = None
    N_eff_effective: float | None = None
    mode: str = ""


class ThermoProvider(Protocol):
    def __call__(self, *args, **kwargs) -> ThermoResult: ...


@dataclass(frozen=True)
class GeometryResult:
    dSigma_plus_dN: float
    dSigma_minus_dN: float
    q: float


class GeometryProvider(Protocol):
    def __call__(self, *args, **kwargs) -> GeometryResult: ...


@dataclass(frozen=True)
class ClassAGeometryResult:
    dSigma_plus_dN: float
    dSigma_minus_dN: float
    dN1_dN: float
    dN2_dN: float
    dN3_dN: float
    q: float
    Omega: float
    K: float
    constraint_residual: float


class ClassAGeometryProvider(Protocol):
    def __call__(self, *args, **kwargs) -> ClassAGeometryResult: ...


@dataclass(frozen=True)
class TransportResult:
    pi_plus: float
    pi_minus: float
    mode: str


class TransportProvider(Protocol):
    def __call__(self, *args, **kwargs) -> TransportResult: ...


@dataclass(frozen=True)
class ClassATransportResult:
    kappa: float
    mode: str


class ClassATransportProvider(Protocol):
    def __call__(self, *args, **kwargs) -> ClassATransportResult: ...


@dataclass(frozen=True)
class WeakBudgetResult:
    lambda_np: float
    lambda_pn: float
    I0: float
    correction_level: int
    mode: str


class WeakBudgetProvider(Protocol):
    def __call__(self, *args, **kwargs) -> WeakBudgetResult: ...


@dataclass(frozen=True)
class TeffClosureBudgetResult:
    pi_tilde: float
    T2: float
    Sigma2: float
    mode: str


class TeffClosureProvider(Protocol):
    def __call__(self, *args, **kwargs) -> TeffClosureBudgetResult: ...
