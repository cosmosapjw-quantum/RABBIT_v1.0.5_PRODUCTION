# FAILURE_LOG.md

## F-001

DATE: 2026-08-23

ATTEMPT: Initial runtime-version receipt.

COMMAND/ENVIRONMENT: Bash here-document in the current checkout.

RESULT: Quoting/termination error; no scientific body ran and no evidence was produced.

ROOT_CAUSE_OR_HYPOTHESIS: Command-construction error.

WHAT_IT_RULES_OUT: Nothing scientific.

RETRY_POLICY: Replaced once by a simple read-only `python -c` version query; succeeded. No source/dependency change.

## F-002

DATE: 2026-08-23

ATTEMPT: First composite X2--X6 probe invocation.

COMMAND/ENVIRONMENT: Isolated helper without `PYTHONPATH=src`.

RESULT: Stopped during import with `ModuleNotFoundError`; no experiment body ran.

ROOT_CAUSE_OR_HYPOTHESIS: Repository source path omitted from the process environment.

WHAT_IT_RULES_OUT: Nothing scientific; it is not a failed solver/probe result.

RETRY_POLICY: Composite command was abandoned. Each frozen X2--X6 experiment was invoked independently with `PYTHONPATH=src:/tmp`, exited 0, and its scientific pass/fail was recorded. The same import failure was not repeated.

## F-003

DATE: 2026-08-23

ATTEMPT: Broader independent relevant-test receipt.

COMMAND/ENVIRONMENT: Independent audit reported an 18-file pytest selection with `PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=src`.

RESULT: Reported 129 passed in 28.73 s, but the exact 18-file argv was not preserved in the primary research record.

ROOT_CAUSE_OR_HYPOTHESIS: Delegated evidence summary omitted the complete filename manifest.

WHAT_IT_RULES_OUT: It cannot serve as an exact reproducibility gate or replace X7's fully recorded 40-test command.

RETRY_POLICY: No retry: the result is retained as supplemental characterization only and makes no closure claim.
