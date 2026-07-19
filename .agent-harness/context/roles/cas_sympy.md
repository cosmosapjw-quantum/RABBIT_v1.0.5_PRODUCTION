# SymPy axis

- Record Python, SymPy, and numerical backend versions.
- Declare symbol assumptions explicitly; never rely on unrecorded global assumptions.
- Compare `simplify`, `cancel`, `factor`, `together`, and targeted transformations rather than treating one simplifier as an oracle.
- Report complex-domain/real-domain differences, principal branches, removable singularities, and piecewise conditions.
- Use high-precision numerical substitutions away from and near singular/branch loci when useful.
