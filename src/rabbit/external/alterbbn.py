"""rabbit.external.alterbbn — AlterBBN live wrapper via Docker (Phase ζ).

Physics meaning
---------------
AlterBBN (Arbey+ 2018, https://alterbbn.hepforge.org/) is a Fortran/C
BBN code used as a community-standard cross-check. We invoke it through
a Docker container so that no Fortran toolchain is required on the host.

Availability gate
-----------------
- ``has_docker()`` : returns True if ``docker`` is on PATH.
- ``has_alterbbn_image()`` : checks for the bundled
  ``rabbit/alterbbn:2.4`` image. This repository does not currently ship
  an AlterBBN Dockerfile; provide the image locally or pull it from the
  configured registry.

When unavailable, raises :class:`ExternalCodeUnavailable` with a
descriptive reason; tests skip rather than fail.

Conversions
-----------
- AlterBBN takes ``eta_10 = eta * 1e10`` (mapped from RABBIT eta).
- AlterBBN ≥ 2.0 reports Y_p as **mass fraction** (matches RABBIT).
- D/H by number ratio (matches RABBIT).
"""

from __future__ import annotations

import json
import shutil
import subprocess
from dataclasses import dataclass
from typing import Optional

from rabbit.external import ExternalCodeUnavailable


_IMAGE_TAG = "rabbit/alterbbn:2.4"


@dataclass(frozen=True)
class AlterBbnResult:
    """Subset of AlterBBN output used by cross-code parity tests."""
    Y_p: float
    DH: float
    Li7H: Optional[float] = None
    raw_stdout: Optional[str] = None


def has_docker() -> bool:
    return shutil.which("docker") is not None


def has_alterbbn_image() -> bool:
    """Return True iff the bundled AlterBBN docker image is present."""
    if not has_docker():
        return False
    out = subprocess.run(
        ["docker", "image", "inspect", _IMAGE_TAG],
        capture_output=True, text=True,
    )
    return out.returncode == 0


def run_alterbbn(
    eta: float = 6.104e-10,
    tau_n: float = 878.4,
    n_eff: float = 3.044,
    timeout_s: int = 120,
) -> AlterBbnResult:
    """Invoke AlterBBN inside ``rabbit/alterbbn:2.4`` and return parsed output.

    Parameters
    ----------
    eta, tau_n, n_eff : standard cosmological inputs.
    timeout_s : hard upper bound on the AlterBBN subprocess.

    Raises
    ------
    ExternalCodeUnavailable
        If docker or the image are missing.
    RuntimeError
        If AlterBBN itself returns non-zero or unparseable output.
    """
    if not has_docker():
        raise ExternalCodeUnavailable(
            "alterbbn",
            reason=(
                "docker not on PATH. Install Docker and provide the "
                f"{_IMAGE_TAG} image before running live AlterBBN parity."
            ),
        )
    if not has_alterbbn_image():
        raise ExternalCodeUnavailable(
            "alterbbn",
            reason=(
                f"docker image {_IMAGE_TAG} not present. Provide it locally "
                f"or run `docker pull {_IMAGE_TAG}` if a registry is configured."
            ),
        )

    eta_10 = float(eta) * 1.0e10
    cmd = [
        "docker", "run", "--rm", _IMAGE_TAG,
        "/alterbbn/alter_etannutau",
        f"{eta_10:.6e}", f"{n_eff:.6f}", f"{tau_n:.6f}",
    ]
    try:
        out = subprocess.run(
            cmd, capture_output=True, text=True, timeout=timeout_s,
        )
    except subprocess.TimeoutExpired as exc:
        raise RuntimeError(
            f"AlterBBN timed out after {timeout_s}s: {' '.join(cmd)}"
        ) from exc
    if out.returncode != 0:
        raise RuntimeError(
            f"AlterBBN returned non-zero ({out.returncode}): {out.stderr}"
        )
    return _parse_alterbbn_stdout(out.stdout)


def _parse_alterbbn_stdout(stdout: str) -> AlterBbnResult:
    """Parse the alter_etannutau line-formatted stdout.

    AlterBBN prints lines of the form ``Yp = 0.24735`` and ``D/H =
    2.547e-5``. We use a robust line-parser that handles whitespace
    drift between AlterBBN versions.
    """
    yp = None
    dh = None
    li7 = None
    for line in stdout.splitlines():
        s = line.strip()
        if s.startswith("Yp") and "=" in s:
            yp = float(s.split("=", 1)[1].strip())
        elif s.startswith("D/H") and "=" in s:
            dh = float(s.split("=", 1)[1].strip())
        elif s.startswith("Li7/H") and "=" in s:
            li7 = float(s.split("=", 1)[1].strip())
    if yp is None or dh is None:
        raise RuntimeError(
            "could not parse AlterBBN stdout for Yp / D/H. "
            f"stdout was:\n{stdout!r}"
        )
    return AlterBbnResult(Y_p=yp, DH=dh, Li7H=li7, raw_stdout=stdout)
