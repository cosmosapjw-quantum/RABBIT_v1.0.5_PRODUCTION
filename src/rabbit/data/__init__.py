"""rabbit.data — versioned observational data registry.

Single source of truth for the observational constraints used across the
inference / likelihood layers (forward_likelihood, bbn_inference, sampler).

Loaded values are returned as frozen Observation dataclasses so callers
cannot accidentally mutate them.
"""
from __future__ import annotations

import json
from dataclasses import dataclass
from importlib import resources
from typing import Mapping


@dataclass(frozen=True)
class Observation:
    """Single observational constraint with optional named provenance.

    Back-compat: ``ref`` defaults to the empty string so legacy call sites
    that constructed ``Observation(name, value, sigma)`` with three
    positional args continue to work.  Production registry entries always
    populate ``ref``.
    """
    name: str
    value: float
    sigma: float
    ref: str = ""

    def chi2(self, prediction: float) -> float:
        return ((prediction - self.value) / self.sigma) ** 2

    def log_likelihood(self, prediction: float) -> float:
        return -0.5 * self.chi2(prediction)


_SUPPORTED_VERSIONS = ("1.0",)


def load_observations(version: str = "1.0") -> Mapping[str, Observation]:
    """Load the versioned BBN observation registry.

    Parameters
    ----------
    version
        Registry version. Currently only "1.0" is supported.

    Returns
    -------
    Mapping[str, Observation]
        Keys: "Yp", "DH", "tau_n", "eta", "Li7H".
    """
    if version not in _SUPPORTED_VERSIONS:
        raise ValueError(
            f"Unknown observation registry version {version!r}. "
            f"Supported: {_SUPPORTED_VERSIONS}"
        )

    with resources.files(__name__).joinpath("observations.json").open("r") as fh:
        payload = json.load(fh)

    if payload.get("version") != version:
        raise ValueError(
            f"observations.json reports version {payload.get('version')!r} "
            f"but caller requested {version!r}"
        )

    out: dict[str, Observation] = {}
    for key, entry in payload["observations"].items():
        out[key] = Observation(
            name=key,
            value=float(entry["value"]),
            sigma=float(entry["sigma"]),
            ref=str(entry["ref"]),
        )
    return out


__all__ = ["Observation", "load_observations"]
