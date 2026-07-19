# Common CAS cross-validation contract

- Read the assigned `CAS_CONTRACT.json` and reproduce its statement exactly before calculation.
- Record engine and package versions, executable command, source script, assumptions, domains, branch choices, simplification rules, and resource limits.
- Separate exact symbolic results, theorem-level results, heuristic simplification, and numerical spot checks.
- Map the final expression back to the shared symbol table and target canonical form.
- Report denominator exclusions, singular loci, branch cuts, hidden positivity/realness assumptions, and order-of-limits issues.
- Do not inspect sibling CAS scripts or results in `blind-results` mode.
- Agreement counts only after assumptions, domains, and canonical forms are aligned.
