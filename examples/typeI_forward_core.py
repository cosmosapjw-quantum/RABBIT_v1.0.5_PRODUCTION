"""Canonical Type I forward-solver example.

This example intentionally uses the Type I characteristic core only. It does
not exercise Teff, QKE, evidence, or non-Type-I Bianchi candidate surfaces.
"""

from __future__ import annotations

from rabbit.inference.forward_likelihood import canonical_forward_solver


def main() -> None:
    flrw = canonical_forward_solver(Sigma_H=0.0, backend="scipy", N_q=6)
    shear = canonical_forward_solver(Sigma_H=0.05, backend="auto", N_q=20)
    print(f"FLRW:       Yp={flrw.Yp:.9f} D/H={flrw.DH:.9e}")
    print(f"Type I:     Yp={shear.Yp:.9f} D/H={shear.DH:.9e}")
    print(f"Type I backend: {shear.metadata.get('dispatch_backend')}")


if __name__ == "__main__":
    main()
