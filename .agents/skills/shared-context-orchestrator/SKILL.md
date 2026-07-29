---
name: shared-context-orchestrator
description: Orchestrate a spec-driven multi-subagent development or audit run with mandatory shared context, registered assignments, bounded fan-out, blind result isolation, and machine-readable handoffs. Use for explicit multi-agent implementation, audit, CAS cross-validation, or adjudication. Do not use for ordinary single-agent edits.
---

# Shared-context orchestrator

1. Read the repository `AGENTS.md` and `.agent-harness/README.md`.
2. Run `python3 .agent-harness/scripts/build_context_pack.py`.
3. If there is no suitable active run, initialize one with `init_run.py` against the governing spec and Git refs.
4. Build a non-overlapping assignment graph before spawning agents.
5. Register every assignment through `new_assignment.py`; never spawn an unregistered child.
6. Mint the assignment's single-use admission receipt with `admit_agent.py`, then include the printed RUN_ID, ASSIGNMENT_ID, CONTEXT_VERSION, INDEPENDENCE_MODE, and ADMISSION_TOKEN header verbatim in the spawn prompt.
7. Keep at most four active subagents, eight total assignments, and depth at most two.
8. Use `blind-results` for independent CAS/reviewer axes. Share the contract, conventions, assumptions, target form, and test vectors, but withhold sibling derivations and verdicts.
9. Wait for required results, run `merge_results.py`, and route only genuine disputes to an adjudicator or targeted rerun.
10. Permit one designated writer to modify production code/spec/shared gates. Other agents write only their unique result artifacts.
11. Run `validate_harness.py` before final gate evaluation.
