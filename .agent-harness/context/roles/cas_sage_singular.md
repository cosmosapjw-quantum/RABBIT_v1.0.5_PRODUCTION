# SageMath + Singular axis

- Record SageMath and Singular versions, coefficient domain, polynomial/rational-function ring, monomial order, and extension fields.
- Record denominator clearing, saturation, localization, primary/branch decomposition, and excluded components.
- Save an executable `.sage`/`.py` script and any Singular script emitted.
- Distinguish equality in a polynomial ideal, equality on a variety, and equality after localization.
- Run every SageMath and Singular process with its working directory set to a newly created assignment-specific directory under `/tmp`; repository cwd is forbidden. Do not repurpose `HOME`, `home`, or `CODEX_HOME` and do not suppress history by redirecting it into the repository.
- Record the temporary cwd, tool commands, and SHA-256 hashes of every ephemeral script/output/history artifact in the declared result. Remove nothing from the repository, and verify that the preserved repository-root `.singularhistory` hash and mtime are unchanged before finishing.
