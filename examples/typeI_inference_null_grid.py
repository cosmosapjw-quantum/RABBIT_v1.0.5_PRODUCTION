"""Full-forward Type I synthetic-null grid example.

This is a parameter-estimation diagnostic, not an evidence or Bayes-factor
calculation.
"""

from __future__ import annotations

import numpy as np

from rabbit.data import Observation
from rabbit.inference.forward_likelihood import (
    BBNLikelihood,
    canonical_forward_solver,
    make_canonical_forward_model,
)


def main() -> None:
    truth = canonical_forward_solver(Sigma_H=0.0, backend="scipy", N_q=6)
    observations = [
        Observation("Yp", truth.Yp, 5.0e-4),
        Observation("DH", truth.DH, 3.0e-7),
    ]
    likelihood = BBNLikelihood(
        make_canonical_forward_model(backend="scipy", N_q=6),
        observations=observations,
    )

    grid = np.linspace(0.0, 0.2, 5)
    values = np.array([
        likelihood.log_likelihood(Sigma_H=float(sigma))
        for sigma in grid
    ])
    best = int(np.argmax(values))
    print(f"best_sigma={grid[best]:.3f}")
    for sigma, loglike in zip(grid, values):
        print(f"Sigma_H={sigma:.3f} logL={loglike:.6f}")


if __name__ == "__main__":
    main()
