"""rabbit.external.finite_shear_anchor — external finite-shear Bianchi-I anchor probe.

Why this exists (external audit 2026-06-30)
-------------------------------------------
The anisotropic (finite-shear) Bianchi-I BBN path has **no external cross-code anchor**.
The FLRW limit is anchored (PRIMAT / Mangano / NUDEC_BSM) and the internal anisotropic gold
(``tests/fixtures/jax_bbn_gold.json``) is a *self-consistency regression lock only* — a shared-bias
error in the anisotropic transport would pass it undetected. No comparable external Bianchi-I BBN
code is known to exist, so finite-shear anisotropic BBN is the honest open frontier (plan item B8).

This module makes that frontier **machine-checkable**: it probes for an externally-supplied
finite-shear benchmark and, when absent, the gate test
(``tests/test_finite_shear_external_anchor.py``) **fails closed as NOT VALIDATED** rather than
skipping (a skip would be counted as a pass by ``scripts/promotion_check.py`` and could promote the
"anisotropic BBN validated" claim on internal self-consistency alone).

Supplying an external benchmark
-------------------------------
Set ``RABBIT_FINITE_SHEAR_BENCHMARK_PATH=/path/to/benchmark.json`` (or drop the file at
``tests/fixtures/finite_shear_external_benchmark_v1.json``). The JSON schema is::

    {
      "source": "<independent code name + version, or analytic small-shear derivation>",
      "tolerances": {"Yp_abs": 1.0e-4, "DH_rel": 5.0e-3},
      "cells": [
        {"Sigma_H_plus": 0.1, "Sigma_H_minus": 0.0, "correction_level": 0,
         "Yp": 0.2440, "DH": 2.50e-5},
        ...
      ]
    }

``source`` MUST name an INDEPENDENT origin (another code, or a closed-form small-shear result) —
NOT a RABBIT self-run. ``cells`` are finite-shear (at least one ``Sigma_H_plus != 0`` or
``Sigma_H_minus != 0``); a benchmark containing only ``Sigma == 0`` cells is rejected as it would
re-anchor the already-anchored FLRW limit, not the anisotropic transport.
"""

from __future__ import annotations

import json
import os
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Optional, Tuple

from rabbit.external import ExternalCodeUnavailable

#: Env var pointing at an external finite-shear benchmark JSON (preferred).
ENV_VAR = "RABBIT_FINITE_SHEAR_BENCHMARK_PATH"

#: Fallback fixture location (not committed; an integrator drops it here).
_FIXTURE = (
    Path(__file__).resolve().parents[3]
    / "tests" / "fixtures" / "finite_shear_external_benchmark_v1.json"
)


@dataclass(frozen=True)
class FiniteShearBenchmark:
    """A loaded external finite-shear benchmark."""
    source: str
    tolerances: dict
    cells: Tuple[dict, ...]
    path: str

    def is_finite_shear(self) -> bool:
        """True iff at least one cell carries genuine (non-zero) shear."""
        return any(
            abs(float(c.get("Sigma_H_plus", 0.0))) > 0.0
            or abs(float(c.get("Sigma_H_minus", 0.0))) > 0.0
            for c in self.cells
        )


def resolve_benchmark_path() -> Optional[str]:
    """Return the path to an external finite-shear benchmark, or None if absent.

    Probe order: the ``RABBIT_FINITE_SHEAR_BENCHMARK_PATH`` env var (preferred), then the
    ``tests/fixtures/finite_shear_external_benchmark_v1.json`` fallback. Returns None — never raises —
    so callers can use it as a pure availability check.
    """
    candidate = os.environ.get(ENV_VAR)
    if candidate:
        candidate = os.path.expanduser(candidate)
        if os.path.isfile(candidate):
            return candidate
        return None
    if _FIXTURE.is_file():
        return str(_FIXTURE)
    return None


def has_finite_shear_external_benchmark() -> bool:
    """Return True iff an external finite-shear benchmark is reachable."""
    return resolve_benchmark_path() is not None


def load_benchmark() -> FiniteShearBenchmark:
    """Load + validate the external finite-shear benchmark.

    Raises :class:`ExternalCodeUnavailable` if no benchmark is reachable (the caller's gate then
    fails closed). Raises ``ValueError`` if a benchmark is present but malformed or FLRW-only
    (a benchmark with no genuine shear would not anchor the anisotropic transport).
    """
    path = resolve_benchmark_path()
    if path is None:
        raise ExternalCodeUnavailable(
            "finite_shear_anchor",
            f"no external finite-shear Bianchi-I benchmark found; set {ENV_VAR} or place "
            f"{_FIXTURE}. No comparable external code is known to exist (plan B8).",
        )
    raw = json.loads(Path(path).read_text(encoding="utf-8"))
    if not isinstance(raw, dict):
        raise ValueError(f"finite-shear benchmark {path}: top-level JSON must be an object")
    raw_cells = raw.get("cells", ())
    if not isinstance(raw_cells, (list, tuple)):
        raise ValueError(f"finite-shear benchmark {path}: 'cells' must be a list")
    cells = tuple(raw_cells)
    if not cells:
        raise ValueError(f"finite-shear benchmark {path} has no 'cells'")
    tolerances = dict(raw.get("tolerances", {}))
    require_dh = "DH_rel" in tolerances  # if the author declares a D/H tolerance, D/H is mandatory

    # Validate the cell schema at LOAD time so a malformed benchmark fails loudly here, not with an
    # opaque KeyError deep in the comparison loop. Each cell must carry the 'Yp' acceptance target
    # and numeric shear/CL fields; 'Sigma_H_minus'/'correction_level' default to 0. 'DH' is optional
    # UNLESS a 'DH_rel' tolerance is declared, in which case every cell must carry it (a missing 'DH'
    # would otherwise be silently half-checked — Yp only — while the author believes both are gated).
    for i, cell in enumerate(cells):
        if not isinstance(cell, dict):
            raise ValueError(f"finite-shear benchmark {path} cell {i} is not an object: {cell!r}")
        if "Yp" not in cell:
            raise ValueError(
                f"finite-shear benchmark {path} cell {i} is missing the required 'Yp' target: {cell!r}")
        if require_dh and "DH" not in cell:
            raise ValueError(
                f"finite-shear benchmark {path} cell {i} is missing 'DH', but the benchmark declares a "
                f"'DH_rel' tolerance — every cell must then carry 'DH' (no silent half-check): {cell!r}")
        try:
            float(cell["Yp"])
            float(cell.get("Sigma_H_plus", 0.0))
            float(cell.get("Sigma_H_minus", 0.0))
            int(cell.get("correction_level", 0))
            if "DH" in cell:
                float(cell["DH"])
        except (TypeError, ValueError) as exc:
            raise ValueError(
                f"finite-shear benchmark {path} cell {i} has a non-numeric field: {cell!r}") from exc

    source = str(raw.get("source", "")).strip()
    if not source:
        raise ValueError(f"finite-shear benchmark {path} is missing an independent 'source'")
    # Enforce the independence claim the module docstring makes: an obvious RABBIT self-run must be
    # rejected loudly, never silently honored as the "external" anchor (the exact masquerade this
    # gate exists to prevent). Independence cannot be fully proven programmatically, but a self-run
    # label is a clear tell.
    if re.search(r"rabbit|self[\s_-]?run|self[\s_-]?consist", source, re.IGNORECASE):
        raise ValueError(
            f"finite-shear benchmark {path} 'source'={source!r} looks like a RABBIT self-run; the "
            "external anchor must name an INDEPENDENT origin (another code or an analytic derivation), "
            "not a RABBIT-generated table."
        )

    bench = FiniteShearBenchmark(source=source, tolerances=tolerances, cells=cells, path=path)
    if not bench.is_finite_shear():
        raise ValueError(
            f"finite-shear benchmark {path} contains only Sigma==0 cells — that re-anchors the "
            "already-anchored FLRW limit, not the anisotropic transport. Supply finite-shear cells."
        )
    return bench
